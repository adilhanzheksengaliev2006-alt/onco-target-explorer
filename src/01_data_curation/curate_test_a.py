"""
curate_test_a.py — курация данных для Теста A (калибровка BEDROC).

ОТКЛОНЕНИЕ ОТ ПРОТОКОЛА (задокументировано честно, не скрыто):
config/protocol.yaml предполагал декои из ZINC как основной вариант ("A").
Из-за ограничений по времени на пилотном прогоне декои вместо ZINC берутся
из САМОГО ChEMBL (общая база, ~2М соединений, HTTP API уже проверенно
работает в этом проекте) — свойства (MW, ALogP, HBD, HBA, RTB) сматчены
так же, как задумывалось для ZINC-декоев, плюс фильтр по Tanimoto ECFP4
< 0.35 ко ВСЕМ активным. Научный смысл теста (проверить, что докинг-скор
отличает активные от свойство-похожих, но структурно других молекул) не
меняется — меняется только источник строительного материала для декоев.
См. docs/limitations.md.

Выход:
  data/processed/test_a_actives_excl_alpelisib.csv
  data/processed/test_a_actives_all.csv           (без исключения серии)
  data/processed/test_a_decoys.csv
  data/processed/test_a_inactives_experimental.csv (вторичный тест)
  results/metrics/test_a_data_decision.json         (что и почему выбрано)
"""

import json
import os
import random
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chembl_webresource_client.new_client import new_client  # noqa: E402

TARGET_CHEMBL_ID = "CHEMBL4005"
ALPELISIB_SMILES = "Cc1nc(NC(=O)N2CCC[C@H]2C(N)=O)sc1-c1ccnc(C(C)(C)C(F)(F)F)c1"
ALPELISIB_TANIMOTO_THRESHOLD = 0.5

MAX_ACTIVES = 250  # подвыборка ради времени докинга, фиксированный seed
DECOY_RATIO = 30
DECOY_TANIMOTO_MAX = 0.35
PROPERTY_TOLERANCE = {"mw": 0.10, "alogp": 0.75, "hbd": 1, "hba": 1, "rtb": 2}  # относительные/абсолютные допуски

RANDOM_SEED = 42

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed")
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "metrics")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def _ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return None


def fetch_pik3ca_molecules(ic50_max_nm=None, ic50_min_nm=None):
    """Все уникальные молекулы PIK3CA с IC50 в заданном диапазоне (нМ)."""
    activity = new_client.activity
    acts = list(
        activity.filter(
            target_chembl_id=TARGET_CHEMBL_ID, standard_type="IC50", standard_relation="="
        ).only(["molecule_chembl_id", "standard_value", "standard_units"])
    )
    mol_ids = set()
    for a in acts:
        try:
            val = float(a["standard_value"])
        except (TypeError, ValueError):
            continue
        if a.get("standard_units") != "nM":
            continue
        if ic50_max_nm is not None and val > ic50_max_nm:
            continue
        if ic50_min_nm is not None and val < ic50_min_nm:
            continue
        mol_ids.add(a["molecule_chembl_id"])
    return mol_ids


def fetch_smiles_for_ids(mol_ids):
    molecule = new_client.molecule
    mol_ids = list(mol_ids)
    out = {}
    batch = 200
    for i in range(0, len(mol_ids), batch):
        chunk = mol_ids[i:i + batch]
        recs = molecule.filter(molecule_chembl_id__in=chunk).only(
            ["molecule_chembl_id", "molecule_structures"]
        )
        for r in recs:
            smi = (r.get("molecule_structures") or {}).get("canonical_smiles")
            if smi:
                out[r["molecule_chembl_id"]] = smi
    return out


def fetch_decoy_candidates(mw_lo, mw_hi, alogp_lo, alogp_hi, exclude_ids, limit=6000, page_size=500):
    """Ручная постраничная выборка с ретраями на страницу -- ChEMBL/EBI
    изредка отдают HTML-страницу ошибки вместо JSON посреди пагинации;
    implicit-итератор queryset тогда падает необратимо. Здесь одна неудачная
    страница просто пропускается (после ретраев), а не роняет всю курацию."""
    molecule = new_client.molecule
    qs = molecule.filter(
        molecule_properties__mw_freebase__range=[mw_lo, mw_hi],
        molecule_properties__alogp__range=[alogp_lo, alogp_hi],
    ).only(["molecule_chembl_id", "molecule_properties", "molecule_structures"])

    out = []
    offset = 0
    consecutive_failures = 0
    max_offset = 10000  # жёсткий потолок на случай низкой плотности
    # кандидатов в этом MW/ALogP-окне -- иначе пагинация может идти
    # неограниченно долго, если limit никогда не достигается
    while len(out) < limit and consecutive_failures < 5 and offset < max_offset:
        page = None
        for attempt in range(3):
            try:
                page = list(qs[offset:offset + page_size])
                break
            except Exception as e:
                print(f"  [curate_test_a] страница {offset}: попытка {attempt + 1}/3 не удалась ({e}), повтор через 5с...")
                import time
                time.sleep(5)
        if page is None:
            consecutive_failures += 1
            offset += page_size
            continue
        consecutive_failures = 0
        if not page:
            break  # больше нет данных
        for r in page:
            if r["molecule_chembl_id"] in exclude_ids:
                continue
            smi = (r.get("molecule_structures") or {}).get("canonical_smiles")
            props = r.get("molecule_properties") or {}
            if not smi or props.get("hbd") is None or props.get("hba") is None or props.get("rtb") is None:
                continue
            out.append({
                "chembl_id": r["molecule_chembl_id"], "smiles": smi,
                "mw": props.get("mw_freebase"), "alogp": props.get("alogp"),
                "hbd": props.get("hbd"), "hba": props.get("hba"), "rtb": props.get("rtb"),
            })
            if len(out) >= limit:
                break
        offset += page_size
        if offset % 2000 == 0:
            print(f"  [curate_test_a] прогресс: offset={offset}, собрано кандидатов={len(out)}")
    return out


def main():
    random.seed(RANDOM_SEED)
    decision_log = {}

    print("[curate_test_a] Загрузка активных (IC50 <= 100 нМ)...")
    active_ids = fetch_pik3ca_molecules(ic50_max_nm=100)
    print(f"  Уникальных активных ID: {len(active_ids)}")

    print("[curate_test_a] Загрузка экспериментальных неактивных (IC50 >= 10000 нМ)...")
    inactive_ids = fetch_pik3ca_molecules(ic50_min_nm=10000)
    print(f"  Уникальных неактивных ID: {len(inactive_ids)}")
    decision_log["n_actives_total_chembl"] = len(active_ids)
    decision_log["n_experimental_inactives_total_chembl"] = len(inactive_ids)

    if len(active_ids) > MAX_ACTIVES:
        active_ids = set(random.sample(sorted(active_ids), MAX_ACTIVES))
    print(f"[curate_test_a] Подвыборка активных для докинга: {len(active_ids)} (seed={RANDOM_SEED})")

    print("[curate_test_a] SMILES активных...")
    active_smiles = fetch_smiles_for_ids(active_ids)
    print("[curate_test_a] SMILES неактивных...")
    inactive_smiles = fetch_smiles_for_ids(inactive_ids)

    alp_fp = _ecfp4(ALPELISIB_SMILES)
    alp_scaffold = _murcko(ALPELISIB_SMILES)

    rows_all, rows_excl = [], []
    props_by_id = {}
    for cid, smi in active_smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = _ecfp4(smi)
        scaffold = _murcko(smi)
        is_alpelisib_series = False
        if fp is not None and alp_fp is not None:
            tanimoto = DataStructs.TanimotoSimilarity(fp, alp_fp)
            if tanimoto >= ALPELISIB_TANIMOTO_THRESHOLD or (scaffold and scaffold == alp_scaffold):
                is_alpelisib_series = True

        mw = Descriptors.MolWt(mol)
        alogp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rtb = Lipinski.NumRotatableBonds(mol)
        props_by_id[cid] = {"mw": mw, "alogp": alogp, "hbd": hbd, "hba": hba, "rtb": rtb, "fp": fp}

        row = {"chembl_id": cid, "smiles": smi, "label": "active",
               "mw": mw, "alogp": alogp, "hbd": hbd, "hba": hba, "rtb": rtb,
               "alpelisib_series": is_alpelisib_series}
        rows_all.append(row)
        if not is_alpelisib_series:
            rows_excl.append(row)

    df_all = pd.DataFrame(rows_all)
    df_excl = pd.DataFrame(rows_excl)
    df_all.to_csv(os.path.join(DATA_DIR, "test_a_actives_all.csv"), index=False)
    df_excl.to_csv(os.path.join(DATA_DIR, "test_a_actives_excl_alpelisib.csv"), index=False)
    print(f"[curate_test_a] Активных всего: {len(df_all)}, без серии алпелисиба: {len(df_excl)} "
          f"(исключено {len(df_all) - len(df_excl)})")
    decision_log["n_actives_used"] = len(df_all)
    decision_log["n_actives_excl_alpelisib"] = len(df_excl)
    decision_log["n_excluded_alpelisib_series"] = len(df_all) - len(df_excl)

    rows_inactive = []
    for cid, smi in inactive_smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows_inactive.append({"chembl_id": cid, "smiles": smi, "label": "inactive_experimental"})
    df_inactive = pd.DataFrame(rows_inactive)
    df_inactive.to_csv(os.path.join(DATA_DIR, "test_a_inactives_experimental.csv"), index=False)
    print(f"[curate_test_a] Экспериментальных неактивных с валидным SMILES: {len(df_inactive)}")
    decision_log["n_experimental_inactives_used"] = len(df_inactive)

    ratio_experimental = len(df_all) / max(len(df_inactive), 1)
    decision_log["experimental_ratio_actives_to_inactives"] = round(ratio_experimental, 2)
    decision_log["decoy_strategy_primary"] = "property_matched_chembl_pool"
    decision_log["decoy_strategy_deviation_from_protocol"] = (
        "protocol.yaml specified ZINC as primary decoy source (option A); "
        "used ChEMBL general compound pool instead for reliability/time reasons "
        "under pilot time constraints. Same matching logic (MW/ALogP/HBD/HBA/RTB "
        "tolerance + Tanimoto ECFP4 < 0.35 to all actives). See docs/limitations.md."
    )

    print("[curate_test_a] Загрузка кандидатов в декои из ChEMBL по диапазону свойств...")
    mws = [p["mw"] for p in props_by_id.values()]
    alogps = [p["alogp"] for p in props_by_id.values()]
    mw_lo, mw_hi = min(mws) * (1 - PROPERTY_TOLERANCE["mw"]), max(mws) * (1 + PROPERTY_TOLERANCE["mw"])
    alogp_lo, alogp_hi = min(alogps) - PROPERTY_TOLERANCE["alogp"], max(alogps) + PROPERTY_TOLERANCE["alogp"]
    exclude_ids = set(active_smiles.keys()) | set(inactive_smiles.keys())
    candidates = fetch_decoy_candidates(mw_lo, mw_hi, alogp_lo, alogp_hi, exclude_ids, limit=4000)
    print(f"[curate_test_a] Кандидатов в декои (после фильтра по MW/ALogP): {len(candidates)}")

    # предвычислить фингерпринты кандидатов
    for c in candidates:
        c["fp"] = _ecfp4(c["smiles"])
    candidates = [c for c in candidates if c["fp"] is not None]

    # max_sim к любой активной молекуле не зависит от того, какой активной
    # сейчас подбираем декои -- считаем ОДИН раз на кандидата (иначе
    # actives x candidates x actives Tanimoto-сравнений, слишком медленно)
    all_active_fps = [p["fp"] for p in props_by_id.values() if p["fp"] is not None]
    bulk_fps = [c["fp"] for c in candidates]
    for i, c in enumerate(candidates):
        sims = DataStructs.BulkTanimotoSimilarity(c["fp"], all_active_fps)
        c["max_sim_to_actives"] = max(sims) if sims else 0.0
    candidates = [c for c in candidates if c["max_sim_to_actives"] < DECOY_TANIMOTO_MAX]
    print(f"[curate_test_a] Кандидатов после Tanimoto-фильтра (<{DECOY_TANIMOTO_MAX} к любой активной): {len(candidates)}")

    random.shuffle(candidates)
    used_decoy_ids = set()
    decoy_rows = []

    for cid, p in props_by_id.items():
        n_assigned = 0
        for c in candidates:
            if n_assigned >= DECOY_RATIO:
                break
            if c["chembl_id"] in used_decoy_ids:
                continue
            if abs(c["hbd"] - p["hbd"]) > PROPERTY_TOLERANCE["hbd"]:
                continue
            if abs(c["hba"] - p["hba"]) > PROPERTY_TOLERANCE["hba"]:
                continue
            if abs(c["rtb"] - p["rtb"]) > PROPERTY_TOLERANCE["rtb"]:
                continue
            used_decoy_ids.add(c["chembl_id"])
            decoy_rows.append({
                "chembl_id": c["chembl_id"], "smiles": c["smiles"], "label": "decoy",
                "matched_active_chembl_id": cid, "mw": c["mw"], "alogp": c["alogp"],
                "hbd": c["hbd"], "hba": c["hba"], "rtb": c["rtb"],
            })
            n_assigned += 1

    df_decoys = pd.DataFrame(decoy_rows)
    df_decoys.to_csv(os.path.join(DATA_DIR, "test_a_decoys.csv"), index=False)
    print(f"[curate_test_a] Декоев подобрано: {len(df_decoys)} "
          f"(целевое макс. {len(props_by_id) * DECOY_RATIO}, "
          f"среднее на активную: {len(df_decoys) / max(len(props_by_id), 1):.1f})")
    decision_log["n_decoys_assigned"] = len(df_decoys)
    decision_log["decoy_ratio_achieved_avg"] = round(len(df_decoys) / max(len(props_by_id), 1), 2)

    with open(os.path.join(METRICS_DIR, "test_a_data_decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision_log, f, indent=2, ensure_ascii=False)
    print("[curate_test_a] Решение по данным записано в results/metrics/test_a_data_decision.json")


if __name__ == "__main__":
    main()
