"""
test_b_generation.py — Тест B: DiffSBDD против baseline, 6 сидов.

ОТКЛОНЕНИЕ ОТ ПРОТОКОЛА (как и в Тесте A, задокументировано честно):
baseline берётся из общего пула ChEMBL (тот же источник, что и декои
Теста A), а не из ZINC -- та же причина (надёжность/время на пилоте),
тот же принцип (большая внешняя база органических drug-like молекул, не
специально подобранная под карман). См. docs/limitations.md.

Для каждого из 6 сидов:
  1. DiffSBDD генерирует N_PER_SEED молекул в карман 4JPS (переиспользует
     чекпоинт module_generative/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt,
     скачанный ранее в этом проекте).
  2. Фильтр валидности: RDKit-парсинг + санитайзинг (--sanitize уже делает
     сам generate_ligands.py), плюс проверка, что центр масс молекулы
     внутри докинг-бокса (простая, быстрая геометрическая проверка;
     полноценная проверка стерических клэшей с белком НЕ реализована --
     упрощение, отмечено в docs/limitations.md).
  3. baseline того же размера рисуется ЗАНОВО из ChEMBL (не один фиксированный
     набор на все 6 сидов).
  4. Обе группы докуются ОДИНАКОВО (Vina против 4JPS, exhaustiveness=8) --
     сравниваются докинг-скоры, а НЕ поза генератора.
  5. Ligand efficiency = -docking_score / heavy_atom_count.

Финально: 10-й перцентиль LE сгенерированных против baseline, перестановочный
тест (объединить обе группы, перемешать метки N раз, посчитать разницу
статистики) по ВСЕМ 6 сидам вместе (для мощности -- 1 сид даёт минимальное
p=0.5, см. protocol.yaml note про min achievable p-value при малом n).
"""

import json
import os
import random
import subprocess
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src", "04_docking"))

from dock_dataframe import dock_dataframe  # noqa: E402
from chembl_webresource_client.new_client import new_client  # noqa: E402

DIFFSBDD_DIR = os.path.join(BASE_DIR, "module_generative", "DiffSBDD")
DIFFSBDD_PYTHON = os.path.join(BASE_DIR, "module_generative", "diffsbdd_env", "Scripts", "python.exe")
DIFFSBDD_CHECKPOINT = os.path.join(DIFFSBDD_DIR, "checkpoints", "crossdocked_fullatom_cond.ckpt")
POCKET_PDB = os.path.join(DIFFSBDD_DIR, "example_4jps.pdb")  # уже скопирован ранее
REF_LIGAND = "A:1102"

BOX_CENTER = np.array([-1.3186999999999998, -9.512666666666666, 16.9481])
BOX_SIZE = np.array([20, 20, 23.114])

N_SEEDS = 6
SEEDS = [1, 2, 3, 4, 5, 6]
N_PER_SEED = 60
EXHAUSTIVENESS = 8
LE_PERCENTILE = 10
N_PERMUTATIONS = 1000

OUT_DIR = os.path.join(BASE_DIR, "results", "testB_generation")
os.makedirs(OUT_DIR, exist_ok=True)
FAILED_LOG = os.path.join(BASE_DIR, "failed.log")


def generate_for_seed(seed: int, n_samples: int) -> list:
    out_sdf = os.path.join(OUT_DIR, f"generated_seed{seed}.sdf")
    cmd = [
        DIFFSBDD_PYTHON,
        os.path.join(DIFFSBDD_DIR, "generate_ligands.py"),
        DIFFSBDD_CHECKPOINT,
        "--pdbfile", POCKET_PDB,
        "--outfile", out_sdf,
        "--ref_ligand", REF_LIGAND,
        "--n_samples", str(n_samples),
        "--sanitize",
    ]
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    result = subprocess.run(cmd, cwd=DIFFSBDD_DIR, capture_output=True, text=True, timeout=1800, env=env)
    if result.returncode != 0:
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(f"test_b_generation\tseed={seed}\tgenerate_ligands.py failed: "
                    f"{result.stderr[-500:]}\n")
        return []

    if not os.path.exists(out_sdf):
        return []

    suppl = Chem.SDMolSupplier(out_sdf)
    smiles_list = []
    for mol in suppl:
        if mol is None:
            continue
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        com = coords.mean(axis=0)
        half = BOX_SIZE / 2
        in_box = np.all(np.abs(com - BOX_CENTER) <= half)
        if not in_box:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            continue
        smiles_list.append(smi)
    return smiles_list


def fetch_baseline_smiles(n: int, seed: int, exclude_smiles: set) -> list:
    """Случайная выборка drug-like молекул из общего пула ChEMBL, заново на
    каждый сид (не фиксированный набор на все 6 сидов)."""
    molecule = new_client.molecule
    rng = random.Random(seed + 1000)
    # окно MW, типичное для drug-like малых молекул -- широкое, не
    # подогнано специально под сгенерированные (в отличие от Теста A,
    # где матчинг по свойствам -- часть самого теста; здесь baseline
    # должен быть НЕподобранным)
    mw_lo, mw_hi = rng.choice([(250, 350), (300, 400), (350, 450), (400, 500)])
    recs = molecule.filter(
        molecule_properties__mw_freebase__range=[mw_lo, mw_hi],
    ).only(["molecule_chembl_id", "molecule_structures"])
    out = []
    for r in recs:
        smi = (r.get("molecule_structures") or {}).get("canonical_smiles")
        if not smi or smi in exclude_smiles:
            continue
        out.append(smi)
        if len(out) >= n * 3:  # запас на невалидные/дубликаты
            break
    rng.shuffle(out)
    return out[:n]


def ligand_efficiency(df: pd.DataFrame) -> pd.Series:
    valid = df["docking_score_kcal_mol"].notna() & (df["heavy_atom_count"] > 0)
    le = pd.Series(index=df.index, dtype=float)
    le[valid] = -df.loc[valid, "docking_score_kcal_mol"] / df.loc[valid, "heavy_atom_count"]
    return le


def permutation_test_percentile_diff(group_a: np.ndarray, group_b: np.ndarray,
                                      percentile: int, n_permutations: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    observed = np.percentile(group_a, percentile) - np.percentile(group_b, percentile)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        diffs[i] = np.percentile(perm[:n_a], percentile) - np.percentile(perm[n_a:], percentile)
    p_value = float((diffs >= observed).mean())
    return float(observed), p_value, diffs


def main():
    all_generated_rows = []
    all_baseline_rows = []

    for seed in SEEDS:
        print(f"[test_b] Сид {seed}: генерация {N_PER_SEED} молекул DiffSBDD...")
        gen_smiles = generate_for_seed(seed, N_PER_SEED)
        print(f"  Валидных (SMILES + центр масс в боксе): {len(gen_smiles)}/{N_PER_SEED}")
        for smi in gen_smiles:
            all_generated_rows.append({"smiles": smi, "seed": seed, "group": "generated"})

        baseline_smiles = fetch_baseline_smiles(max(len(gen_smiles), 1), seed, exclude_smiles=set(gen_smiles))
        print(f"  Baseline (ChEMBL, заново на этот сид): {len(baseline_smiles)}")
        for smi in baseline_smiles:
            all_baseline_rows.append({"smiles": smi, "seed": seed, "group": "baseline"})

    df_gen = pd.DataFrame(all_generated_rows)
    df_base = pd.DataFrame(all_baseline_rows)
    df_gen.to_csv(os.path.join(OUT_DIR, "generated_smiles.csv"), index=False)
    df_base.to_csv(os.path.join(OUT_DIR, "baseline_smiles.csv"), index=False)

    print(f"[test_b] Докинг {len(df_gen)} сгенерированных молекул...")
    df_gen_docked = dock_dataframe(df_gen, exhaustiveness=EXHAUSTIVENESS, failed_log_path=FAILED_LOG)
    print(f"[test_b] Докинг {len(df_base)} baseline-молекул...")
    df_base_docked = dock_dataframe(df_base, exhaustiveness=EXHAUSTIVENESS, failed_log_path=FAILED_LOG)

    df_gen_docked["ligand_efficiency"] = ligand_efficiency(df_gen_docked)
    df_base_docked["ligand_efficiency"] = ligand_efficiency(df_base_docked)

    df_gen_docked.to_csv(os.path.join(OUT_DIR, "generated_docked.csv"), index=False)
    df_base_docked.to_csv(os.path.join(OUT_DIR, "baseline_docked.csv"), index=False)

    le_gen = df_gen_docked["ligand_efficiency"].dropna().values
    le_base = df_base_docked["ligand_efficiency"].dropna().values

    result = {
        "n_seeds": N_SEEDS, "n_per_seed_requested": N_PER_SEED,
        "n_generated_valid": len(df_gen), "n_generated_docked": len(le_gen),
        "n_baseline": len(df_base), "n_baseline_docked": len(le_base),
        "le_generated_median": float(np.median(le_gen)) if len(le_gen) else None,
        "le_baseline_median": float(np.median(le_base)) if len(le_base) else None,
        "le_generated_p10": float(np.percentile(le_gen, LE_PERCENTILE)) if len(le_gen) else None,
        "le_baseline_p10": float(np.percentile(le_base, LE_PERCENTILE)) if len(le_base) else None,
    }

    if len(le_gen) >= 5 and len(le_base) >= 5:
        observed, p_value, _ = permutation_test_percentile_diff(
            le_gen, le_base, LE_PERCENTILE, N_PERMUTATIONS
        )
        result["permutation_observed_diff_p10"] = observed
        result["permutation_p_value"] = p_value
        result["n_permutations"] = N_PERMUTATIONS
        result["min_achievable_p_at_n_seeds"] = round(2 / (2 ** N_SEEDS), 4)
    else:
        result["error"] = "недостаточно задокированных молекул для перестановочного теста"

    with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== Тест B: результат ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
