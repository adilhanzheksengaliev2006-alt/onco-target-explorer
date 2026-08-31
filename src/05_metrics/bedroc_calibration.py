"""
bedroc_calibration.py — общая логика для Теста A (калибровка), Теста N
(перемешивание меток) и Теста A' (размерное смещение).

BEDROC/AUC/enrichment считаются встроенными функциями RDKit
(rdkit.ML.Scoring.Scoring) — не переизобретаются, чтобы не внести
тонкую ошибку в саму метрику.

Случайный базовый уровень BEDROC считается ЭМПИРИЧЕСКИ (много перемешиваний
меток, среднее), а не по замкнутой формуле из статьи Truchon & Bayly —
сознательный выбор: и Тест N (обязателен по протоколу отдельно), и
"случайный базовый уровень" для Теста A используют один и тот же код,
результат честно воспроизводим прогоном, а не набранной по памяти
константой из формулы. См. docs/methodology.md.
"""

import json
import os
import random

import numpy as np
import pandas as pd
from rdkit.ML.Scoring import Scoring


def _scores_matrix(df: pd.DataFrame, score_col: str, label_col: str):
    """Готовит [[score, is_active], ...], отсортированный по score
    ВОЗРАСТАНИЕ (более отрицательный докинг-скор = лучше = первым)."""
    sub = df[[score_col, label_col]].dropna(subset=[score_col]).copy()
    sub = sub.sort_values(score_col, ascending=True)
    return sub[[score_col, label_col]].values.tolist(), sub


def compute_bedroc(scores_matrix, alpha: float) -> float:
    return Scoring.CalcBEDROC(scores_matrix, 1, alpha)


def compute_enrichment(scores_matrix, fractions=(0.01, 0.05, 0.10)) -> dict:
    ef = Scoring.CalcEnrichment(scores_matrix, 1, list(fractions))
    return {f"EF_{int(f * 100)}%": v for f, v in zip(fractions, ef)}


def compute_auc_roc(scores_matrix) -> float:
    return Scoring.CalcAUC(scores_matrix, 1)


def empirical_random_baseline(sub_df: pd.DataFrame, score_col: str, label_col: str,
                               alpha: float, n_shuffles: int, seed: int = 0):
    """Тест N: перемешивает МЕТКИ (не скоры) n_shuffles раз, считает BEDROC
    на каждой перестановке. Даёт и "случайный базовый уровень" для Теста A,
    и сам Тест N (доказательство, что код метрики не врёт -- на
    перемешанных метках распределение должно лежать вокруг случайного
    уровня, не давать систематически высокий BEDROC)."""
    rng = random.Random(seed)
    labels = sub_df[label_col].tolist()
    scores = sub_df[score_col].tolist()
    bedroc_values = []
    for _ in range(n_shuffles):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        matrix = list(zip(scores, shuffled))
        # уже отсортировано по score, т.к. sub_df пришёл отсортированным
        bedroc_values.append(compute_bedroc(matrix, alpha))
    return np.array(bedroc_values)


def bootstrap_ci_bedroc(sub_df: pd.DataFrame, score_col: str, label_col: str,
                         alpha: float, n_iterations: int, seed: int = 0,
                         ci: float = 0.95):
    """Бутстрап ПО АКТИВНЫМ (ресэмплинг активных с возвращением, декои
    остаются как есть -- стандартный подход для доверительного интервала
    BEDROC/enrichment, отражает неопределённость из-за конечной выборки
    активных)."""
    rng = np.random.RandomState(seed)
    actives = sub_df[sub_df[label_col] == 1]
    inactives = sub_df[sub_df[label_col] == 0]
    n_actives = len(actives)
    values = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n_actives, size=n_actives)
        resampled_actives = actives.iloc[idx]
        combined = pd.concat([resampled_actives, inactives]).sort_values(score_col, ascending=True)
        matrix = combined[[score_col, label_col]].values.tolist()
        values.append(compute_bedroc(matrix, alpha))
    values = np.array(values)
    lo = np.percentile(values, (1 - ci) / 2 * 100)
    hi = np.percentile(values, (1 - (1 - ci) / 2) * 100)
    return float(lo), float(hi), values


def permutation_p_value(observed_bedroc: float, null_distribution: np.ndarray) -> float:
    """Доля перестановок с BEDROC >= наблюдаемого (односторонний тест)."""
    return float((null_distribution >= observed_bedroc).mean())


def run_test_a(df: pd.DataFrame, score_col: str, label_col: str, alpha: float,
               n_bootstrap: int, n_permutations: int, seed: int = 0) -> dict:
    matrix, sub = _scores_matrix(df, score_col, label_col)
    n_active = int(sub[label_col].sum())
    n_total = len(sub)
    ra = n_active / n_total if n_total else 0.0

    observed_bedroc = compute_bedroc(matrix, alpha)
    null_dist = empirical_random_baseline(sub, score_col, label_col, alpha, n_permutations, seed=seed)
    random_baseline_mean = float(null_dist.mean())
    random_baseline_std = float(null_dist.std())
    p_value = permutation_p_value(observed_bedroc, null_dist)

    ci_lo, ci_hi, _ = bootstrap_ci_bedroc(sub, score_col, label_col, alpha, n_bootstrap, seed=seed)

    ef = compute_enrichment(matrix)
    auc = compute_auc_roc(matrix)

    return {
        "n_active": n_active,
        "n_total": n_total,
        "ra_active_fraction": ra,
        "bedroc_alpha": alpha,
        "bedroc_observed": observed_bedroc,
        "bedroc_random_baseline_empirical_mean": random_baseline_mean,
        "bedroc_random_baseline_empirical_std": random_baseline_std,
        "bedroc_bootstrap_ci_95_lo": ci_lo,
        "bedroc_bootstrap_ci_95_hi": ci_hi,
        "permutation_p_value": p_value,
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        **ef,
        "auc_roc": auc,
    }


def run_test_n_shuffle(df: pd.DataFrame, score_col: str, label_col: str, alpha: float,
                        n_permutations: int, seed: int = 0) -> dict:
    """Отдельный отчёт Теста N (может переиспользовать null_dist из run_test_a,
    но остаётся отдельной точкой входа для прозрачности отчёта)."""
    matrix, sub = _scores_matrix(df, score_col, label_col)
    null_dist = empirical_random_baseline(sub, score_col, label_col, alpha, n_permutations, seed=seed)
    return {
        "n_permutations": n_permutations,
        "bedroc_shuffled_mean": float(null_dist.mean()),
        "bedroc_shuffled_std": float(null_dist.std()),
        "bedroc_shuffled_min": float(null_dist.min()),
        "bedroc_shuffled_max": float(null_dist.max()),
        "values": null_dist.tolist(),
    }


def run_test_a_prime_size_bias(df: pd.DataFrame, score_col: str, heavy_atom_col: str,
                                r2_threshold: float) -> dict:
    """Регрессия скора по числу тяжёлых атомов. R^2 > порог = скор
    в основном отражает размер молекулы, а не связывание."""
    sub = df[[score_col, heavy_atom_col]].dropna()
    x = sub[heavy_atom_col].values.astype(float)
    y = sub[score_col].values.astype(float)
    if len(x) < 3:
        return {"error": "недостаточно данных для регрессии", "n": len(x)}
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "n": len(x), "slope": float(slope), "intercept": float(intercept),
        "r2": float(r2), "threshold": r2_threshold,
        "size_bias_detected": bool(r2 > r2_threshold),
    }
