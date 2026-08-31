"""
module_generative/iterative_finetune_loop.py
=========================================================================
Итеративный пайплайн: генерация (MolGPT) -> фильтр валидности (RDKit)
-> докинг (AutoDock Vina) -> фильтр токсичности (PAINS/Brenk) ->
фильтр селективности (ChEMBL) -> дообучение MolGPT на прошедших
молекулах -> повтор.

ВЕСЬ пайплайн определяется одним параметром GENE_NAME — мишень в
ChEMBL, 3D-структура для докинга и т.д. подхватываются автоматически
(см. gene_target_utils.py). Для смены гена достаточно поменять
GENE_NAME ниже (или передать первым аргументом командной строки).

Обычное дообучение (supervised fine-tuning) на "хороших" примерах —
не RL.
"""

import os
import sys
import time

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gene_target_utils import GeneTargetError, resolve_gene_to_docking_target, get_chembl_new_client  # noqa: E402
from dock_existing_candidates import (  # noqa: E402
    DockingError,
    prepare_receptor,
    dock_smiles_isolated,
)
from generate_molecules_v1 import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    load_molgpt,
    generate_raw_smiles,
    filter_valid_unique,
)
from module5_admet_filters import (  # noqa: E402
    Admet5Error,
    compute_sa_scores,
    fetch_admet_batch,
    count_admet_red_flags,
    rank_candidates,
)

# =========================================================================
# ЕДИНСТВЕННЫЙ ПАРАМЕТР, ЗАДАЮЩИЙ ГЕН ДЛЯ ВСЕГО ПАЙПЛАЙНА
# =========================================================================
GENE_NAME = "PIK3CA"  # поменять на EGFR / BRAF / ABL1 / ALK для проверки
                        # на других известных парах ген-препарат — весь
                        # пайплайн подхватывает это значение автоматически

# =========================================================================
# Параметры цикла
# =========================================================================
DEV_MODE = False  # True -> быстрый прогон: 20 молекул/цикл, максимум 2 цикла

N_CYCLES = 2 if DEV_MODE else 20
N_GENERATE_PER_CYCLE = 20 if DEV_MODE else 500
VINA_EXHAUSTIVENESS = 4 if DEV_MODE else 8
VINA_TIMEOUT_SEC = 60  # молекула, на которой vina зависает дольше этого — считается неудачной, не блокирует цикл
FINETUNE_EPOCHS = 1 if DEV_MODE else 3
FINETUNE_LR = 5e-5

DOCKING_SCORE_THRESHOLD = -7.0  # ккал/моль; докинг лучше (отрицательнее) порога -> проходит
CONTROL_DOCKING_MAX_SCORE = -5.0  # контрольный докинг известного ингибитора хуже этого -> setup сломан, прогон не запускается
SELECTIVITY_IC50_THRESHOLD_NM = 1000.0  # "активность сильнее" этого порога -> считается хитом на мишени
SELECTIVITY_MAX_OTHER_TARGETS = 5  # больше этого числа посторонних мишеней -> неселективна

RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


class PipelineError(Exception):
    """Понятная, не-молчаливая ошибка любого внешнего шага пайплайна."""


# =========================================================================
# Фильтр токсичности: RDKit FilterCatalog (PAINS + Brenk)
# =========================================================================
def build_toxicity_catalog():
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        return FilterCatalog(params)
    except Exception as e:
        raise PipelineError(f"Не удалось построить каталог PAINS/Brenk (RDKit FilterCatalog): {e}")


def passes_toxicity_filter(smiles: str, catalog) -> bool:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return len(catalog.GetMatches(mol)) == 0


# =========================================================================
# Фильтр селективности: ChEMBL — сколько ЕЩЁ мишеней у молекулы
# =========================================================================
def _short_error(e: Exception, limit: int = 200) -> str:
    """ChEMBL иногда отдаёт HTML-страницу ошибки (500) вместо JSON —
    её текст попадает в str(исключения) целиком (десятки КБ) и не
    несёт полезной информации. Обрезаем, чтобы не раздувать лог."""
    msg = str(e)
    return msg if len(msg) <= limit else msg[:limit] + f"... [обрезано, всего {len(msg)} симв.]"


def _chembl_call_with_retry(fn, attempts: int = 3, delay_sec: float = 5.0):
    """ChEMBL/EBI API изредка отдаёт транзиентные 500-е ошибки под
    нагрузкой — несколько попыток с паузой спасают от того, чтобы
    одна молекула теряла весь фильтр селективности из-за случайного
    сбоя на стороне сервера."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                time.sleep(delay_sec)
    raise last_exc


_chembl_new_client_cache = None


def _get_chembl_new_client():
    """chembl_webresource_client.new_client делает сетевой запрос схемы
    (/spore) ПРИ ИМПОРТЕ модуля. Если он падает (EBI изредка отдаёт
    500 на /spore), а мы наивно делаем `from ... import` внутри
    check_selectivity() на каждую молекулу — сбой воспроизводится
    заново для КАЖДОЙ молекулы (Python не кеширует неудавшийся
    импорт). Поэтому импорт с ретраями делается один раз за процесс
    и кешируется в модульной переменной."""
    global _chembl_new_client_cache
    if _chembl_new_client_cache is not None:
        return _chembl_new_client_cache

    def _import():
        from chembl_webresource_client.new_client import new_client
        return new_client

    _chembl_new_client_cache = _chembl_call_with_retry(_import, attempts=3, delay_sec=8.0)
    return _chembl_new_client_cache


def check_selectivity(smiles: str, own_target_chembl_id: str):
    """Возвращает (n_other_targets, note). Если молекула не найдена в
    ChEMBL (обычный случай для НОВЫХ сгенерированных молекул) —
    считается, что посторонней активности не задокументировано, и
    молекула проходит фильтр (это честно отражено в note)."""
    try:
        new_client = _get_chembl_new_client()
        molecule = new_client.molecule
        matches = _chembl_call_with_retry(
            lambda: list(
                molecule.filter(
                    molecule_structures__canonical_smiles__flexmatch=smiles
                ).only(["molecule_chembl_id"])
            )
        )
    except Exception as e:
        raise PipelineError(f"Ошибка запроса ChEMBL (structure match) для селективности: {_short_error(e)}")

    if not matches:
        return 0, "молекула не найдена в ChEMBL (новая) — посторонняя активность неизвестна"

    mol_ids = list(set(m["molecule_chembl_id"] for m in matches))
    try:
        activity = new_client.activity
        acts = _chembl_call_with_retry(
            lambda: list(
                activity.filter(
                    molecule_chembl_id__in=mol_ids,
                    standard_type__in=["IC50", "EC50", "Ki", "Kd"],
                    standard_units="nM",
                    standard_value__isnull=False,
                ).only(["target_chembl_id", "standard_value"])
            )
        )
    except Exception as e:
        raise PipelineError(f"Ошибка запроса ChEMBL activity для селективности: {_short_error(e)}")

    other_targets = set()
    for a in acts:
        try:
            value = float(a["standard_value"])
        except (TypeError, ValueError):
            continue
        if value < SELECTIVITY_IC50_THRESHOLD_NM and a["target_chembl_id"] != own_target_chembl_id:
            other_targets.add(a["target_chembl_id"])

    note = f"найдена в ChEMBL ({mol_ids[0]}), посторонних мишеней с активностью <{SELECTIVITY_IC50_THRESHOLD_NM:.0f}нМ: {len(other_targets)}"
    return len(other_targets), note


# =========================================================================
# Дообучение (обычный supervised fine-tuning, не RL)
# =========================================================================
def finetune_model(model, tokenizer, smiles_list, epochs, lr):
    import torch

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_loss, steps = 0.0, 0
    for epoch in range(epochs):
        for smi in smiles_list:
            enc = tokenizer(smi, return_tensors="pt", truncation=True, max_length=128)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        print(f"    эпоха {epoch + 1}/{epochs} завершена")
    model.eval()
    avg_loss = total_loss / steps if steps else None
    return model, avg_loss


# =========================================================================
# Один цикл
# =========================================================================
def run_cycle(cycle_idx, model_path, gene_name, target_info, receptor_pdbqt, tox_catalog):
    print(f"\n{'=' * 70}\nЦИКЛ {cycle_idx} (модель: {model_path})\n{'=' * 70}")

    counts = {"generated": 0, "valid": 0, "docked_pass": 0, "tox_pass": 0, "selectivity_pass": 0}

    # --- генерация ---
    try:
        model, tokenizer = load_molgpt(model_path)
    except Exception as e:
        raise PipelineError(f"Не удалось загрузить MolGPT ({model_path}): {e}")

    try:
        raw_smiles = generate_raw_smiles(model, tokenizer, n=N_GENERATE_PER_CYCLE)
    except Exception as e:
        raise PipelineError(f"Ошибка генерации SMILES моделью MolGPT: {e}")
    counts["generated"] = len(raw_smiles)

    # --- валидность + дедуп ---
    valid_smiles = filter_valid_unique(raw_smiles)
    counts["valid"] = len(valid_smiles)
    if not valid_smiles:
        print("Ни одной валидной молекулы — цикл завершён без результатов.")
        return counts, pd.DataFrame(), model, tokenizer

    # --- докинг ---
    docking_scores = {}
    import tempfile

    with tempfile.TemporaryDirectory() as workdir:
        for i, smi in enumerate(valid_smiles):
            print(f"  докинг {i + 1}/{len(valid_smiles)}...", flush=True)
            score = dock_smiles_isolated(
                smi, receptor_pdbqt, target_info["box_center"], target_info["box_size"],
                workdir, tag=f"c{cycle_idx}_{i}", exhaustiveness=VINA_EXHAUSTIVENESS, timeout=VINA_TIMEOUT_SEC,
            )
            docking_scores[smi] = score
            print(f"    -> {score}", flush=True)
    docked_pass = [s for s in valid_smiles if docking_scores.get(s) is not None and docking_scores[s] <= DOCKING_SCORE_THRESHOLD]
    counts["docked_pass"] = len(docked_pass)
    print(f"Докинг: {sum(1 for v in docking_scores.values() if v is not None)}/{len(valid_smiles)} успешно задокировано, "
          f"{len(docked_pass)} прошли порог {DOCKING_SCORE_THRESHOLD} ккал/моль")
    if not docked_pass:
        print("Ни одна молекула не прошла докинг-фильтр — цикл завершён без результатов.")
        return counts, pd.DataFrame(), model, tokenizer

    # --- токсичность (PAINS/Brenk) ---
    tox_pass = [s for s in docked_pass if passes_toxicity_filter(s, tox_catalog)]
    counts["tox_pass"] = len(tox_pass)
    print(f"Токсичность (PAINS/Brenk): {len(tox_pass)}/{len(docked_pass)} без алертов")
    if not tox_pass:
        print("Ни одна молекула не прошла фильтр токсичности — цикл завершён без результатов.")
        return counts, pd.DataFrame(), model, tokenizer

    # --- SA score (RDKit, дёшево, локально, НЕ фильтрует — только колонка) ---
    sa_scores = compute_sa_scores(tox_pass)
    print(f"SA score посчитан для {len(tox_pass)} молекул, прошедших PAINS/Brenk (не фильтрует)")

    # --- ADMETlab 3.0 (полный ADME + токсикология, НЕ фильтрует — только колонки) ---
    try:
        admet_df = fetch_admet_batch(tox_pass)
    except Admet5Error as e:
        print(f"  [ADMETlab] ОШИБКА батча: {e} — ADMET-колонки будут пустыми в этом цикле, цикл продолжается без них.")
        admet_df = pd.DataFrame()

    # --- селективность (ChEMBL) ---
    selectivity_pass = []
    selectivity_notes = {}
    for smi in tox_pass:
        try:
            n_other, note = check_selectivity(smi, target_info["chembl_target_id"])
        except PipelineError as e:
            print(f"  [селективность] пропуск молекулы из-за ошибки ChEMBL: {e}")
            continue
        selectivity_notes[smi] = note
        if n_other <= SELECTIVITY_MAX_OTHER_TARGETS:
            selectivity_pass.append(smi)
    counts["selectivity_pass"] = len(selectivity_pass)
    print(f"Селективность (<= {SELECTIVITY_MAX_OTHER_TARGETS} посторонних мишеней): {len(selectivity_pass)}/{len(tox_pass)}")

    result_rows = []
    for smi in selectivity_pass:
        row = {
            "SMILES": smi,
            "docking_score_kcal_mol": docking_scores[smi],
            "selectivity_note": selectivity_notes.get(smi, ""),
            "SA_score": sa_scores.get(smi),
        }
        if smi in admet_df.index:
            admet_row = admet_df.loc[smi]
            for c in admet_df.columns:
                row[c] = admet_row[c]
            row["ADMET_red_flags"] = count_admet_red_flags(admet_row)
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    if not result_df.empty:
        # Композитное ранжирование: докинг (уже отфильтрован по порогу) ->
        # меньше явных ADMET красных флагов -> меньше SA score (легче синтезировать).
        # Не отсеивает — только сортирует, см. module5_admet_filters.rank_candidates.
        result_df = rank_candidates(result_df)
    return counts, result_df, model, tokenizer


# =========================================================================
# Контроль-проверка настройки докинга: докуем известный ингибитор
# ПЕРЕД тем, как гнать всю партию сгенерированных молекул. Если скор
# аномальный — значит структура/бокс/подготовка сломаны, и продолжать
# бессмысленно (см. историю с 9CMK: скор алпелисиба там был
# "разумным на вид", но карман был не тот — поэтому здесь порог
# скорее защита от совсем поломанной настройки, а не гарантия
# правильности кармана; уверенность в кармане даёт сам подбор
# структуры по известному лиганду в gene_target_utils).
# =========================================================================
def run_control_docking(target_info: dict, receptor_pdbqt: str) -> float:
    ligand_chembl_id = target_info.get("matched_known_ligand_chembl_id")
    ligand_name = target_info.get("matched_known_ligand")
    if not ligand_chembl_id:
        raise PipelineError(
            "Нет ChEMBL ID известного лиганда структуры (matched_known_ligand_chembl_id) — "
            "не могу выполнить контрольный докинг перед прогоном."
        )

    try:
        new_client = get_chembl_new_client()
        rec = new_client.molecule.get(ligand_chembl_id)
        smiles = (rec.get("molecule_structures") or {}).get("canonical_smiles")
    except Exception as e:
        raise PipelineError(f"Не удалось получить SMILES контрольного лиганда {ligand_chembl_id}: {str(e)[:200]}")

    if not smiles:
        raise PipelineError(f"У контрольного лиганда {ligand_chembl_id} нет SMILES в ChEMBL.")

    print(f"\n--- Контрольный докинг: {ligand_name} ({ligand_chembl_id}) в {target_info['pdb_id']} ---")
    print(f"  SMILES: {smiles}")

    import tempfile

    with tempfile.TemporaryDirectory() as workdir:
        score = dock_smiles_isolated(
            smiles, receptor_pdbqt, target_info["box_center"], target_info["box_size"],
            workdir, tag="control", exhaustiveness=VINA_EXHAUSTIVENESS, timeout=120,
        )

    print(f"  Docking score контроля: {score}")

    if score is None:
        raise PipelineError(
            f"Контрольный докинг {ligand_name} не удался (докинг вернул None) — "
            f"настройка рецептора/бокса сломана, прогон остановлен."
        )
    if score > CONTROL_DOCKING_MAX_SCORE:
        raise PipelineError(
            f"Контрольный докинг {ligand_name} дал аномальный скор {score} ккал/моль "
            f"(ожидался разумный диапазон примерно -7...-10, порог отсечения "
            f"{CONTROL_DOCKING_MAX_SCORE}). Это значит настройка докинга (структура/бокс/"
            f"подготовка рецептора) скорее всего сломана — прогон остановлен, чтобы не "
            f"тратить время на заведомо бессмысленные результаты."
        )

    print(f"  Контроль пройден: скор в разумном диапазоне.\n")
    return score


# =========================================================================
# Главный запуск
# =========================================================================
def run_iterative_pipeline(gene_name: str):
    print(f"########## Итеративный пайплайн генерации+фильтрации+дообучения: {gene_name} ##########")
    print(f"DEV_MODE={DEV_MODE}, циклов={N_CYCLES}, молекул/цикл={N_GENERATE_PER_CYCLE}\n")

    try:
        target_info = resolve_gene_to_docking_target(gene_name)
    except GeneTargetError as e:
        print(f"ОШИБКА поиска мишени/структуры: {e}")
        return
    if target_info is None:
        print(f"Докинг-структура для {gene_name} не найдена — пайплайн для этого гена остановлен.")
        return

    receptor_basename = os.path.join(
        os.path.dirname(target_info["pdb_path"]), f"{target_info['pdb_id']}_receptor"
    )
    try:
        receptor_pdbqt = prepare_receptor(
            target_info["pdb_path"], target_info["box_center"], target_info["box_size"], receptor_basename
        )
    except DockingError as e:
        print(f"ОШИБКА подготовки рецептора: {e}")
        return
    print(f"Рецептор готов: {receptor_pdbqt}\n")

    try:
        run_control_docking(target_info, receptor_pdbqt)
    except PipelineError as e:
        print(f"ОШИБКА КОНТРОЛЯ: {e}")
        return

    try:
        tox_catalog = build_toxicity_catalog()
    except PipelineError as e:
        print(f"ОШИБКА: {e}")
        return

    current_model_path = DEFAULT_MODEL_NAME
    all_cycle_summaries = []

    for cycle in range(1, N_CYCLES + 1):
        t0 = time.time()
        try:
            counts, result_df, model, tokenizer = run_cycle(
                cycle, current_model_path, gene_name, target_info, receptor_pdbqt, tox_catalog
            )
        except PipelineError as e:
            print(f"ОШИБКА в цикле {cycle}: {e}")
            break

        elapsed = time.time() - t0
        summary = {
            "cycle": cycle,
            "model_path": current_model_path,
            **counts,
            "elapsed_sec": round(elapsed, 1),
        }
        all_cycle_summaries.append(summary)

        print(
            f"\n--- Сводка цикла {cycle} ---\n"
            f"  сгенерировано:        {counts['generated']}\n"
            f"  прошло валидность:    {counts['valid']}\n"
            f"  прошло докинг:        {counts['docked_pass']}\n"
            f"  прошло токсичность:   {counts['tox_pass']}\n"
            f"  прошло селективность: {counts['selectivity_pass']}\n"
            f"  время цикла:          {elapsed:.1f}с"
        )

        cycle_csv = os.path.join(RESULTS_DIR, f"cycle_{cycle}_results_{gene_name}.csv")
        result_df.to_csv(cycle_csv, index=False)
        print(f"  сохранено: {cycle_csv} ({len(result_df)} молекул)")

        if result_df.empty:
            print(f"  В цикле {cycle} НИ ОДНА молекула не прошла все фильтры — "
                  f"дообучение на этом цикле ПРОПУЩЕНО (нельзя обучать на пустом наборе).")
            continue

        ckpt_dir = os.path.join(CHECKPOINTS_DIR, gene_name, f"cycle_{cycle}")
        print(f"  Дообучение MolGPT на {len(result_df)} молекулах, прошедших все фильтры...")
        try:
            model, avg_loss = finetune_model(
                model, tokenizer, result_df["SMILES"].tolist(), FINETUNE_EPOCHS, FINETUNE_LR
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Чекпоинт сохранён: {ckpt_dir} (средний loss: {avg_loss:.4f})" if avg_loss is not None else f"  Чекпоинт сохранён: {ckpt_dir}")
            current_model_path = ckpt_dir
        except Exception as e:
            print(f"  ОШИБКА дообучения/сохранения чекпоинта: {e} — следующий цикл продолжит с прежней моделью.")

    print(f"\n{'=' * 70}\nИТОГ ПО ВСЕМ ЦИКЛАМ ({gene_name})\n{'=' * 70}")
    summary_df = pd.DataFrame(all_cycle_summaries)
    if summary_df.empty:
        print("Ни одного цикла не завершилось успешно.")
    else:
        print(summary_df.to_string(index=False))
        total_pass = summary_df["selectivity_pass"].sum()
        if total_pass == 0:
            print("\nЗа весь прогон НИ ОДНА молекула не прошла все фильтры целиком.")
        else:
            print(f"\nВсего молекул, прошедших все фильтры за все циклы: {total_pass}")
        summary_csv = os.path.join(RESULTS_DIR, f"run_summary_{gene_name}.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Сводка сохранена: {summary_csv}")


if __name__ == "__main__":
    gene = sys.argv[1] if len(sys.argv) > 1 else GENE_NAME
    run_iterative_pipeline(gene)
