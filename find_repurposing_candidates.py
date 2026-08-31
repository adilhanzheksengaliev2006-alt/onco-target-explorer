"""
find_repurposing_candidates.py — Модуль 3: поиск кандидатов для
repurposing среди ОДОБРЕННЫХ препаратов (max_phase=4)
==========================================================================
Задаётся один параметр GENE_NAME. Всё остальное — автоматически:
  1. Мишень в ChEMBL (gene_target_utils.find_chembl_target)
  2. ChEMBL mechanism endpoint — курируемые связи препарат-мишень
     (какие вещества ИЗВЕСТНЫ как действующие на эту мишень)
  3. Среди них — только max_phase=4 (одобренные регуляторами препараты)
  4. Ранжирование по лучшей измеренной активности (мин. IC50/EC50 в нМ,
     через pChEMBL) там, где такие данные есть в ChEMBL

Это "слепой тест" метода: для известных пар ген-препарат (PIK3CA ->
алпелисиб и т.д.) программа не знает заранее правильный ответ — она
просто возвращает ранжированный список, который нужно сверить вручную.
"""

import sys

import pandas as pd

from gene_target_utils import GeneTargetError, find_chembl_target, get_chembl_new_client


GENE_NAME = "PIK3CA"


def get_approved_drugs_for_target(target_chembl_id: str) -> list[dict]:
    """Курируемые (mechanism) связи препарат-мишень, отфильтрованные
    по max_phase=4."""
    try:
        new_client = get_chembl_new_client()

        mechanism = new_client.mechanism
        mech_records = list(
            mechanism.filter(target_chembl_id=target_chembl_id).only(
                ["molecule_chembl_id", "mechanism_of_action", "action_type"]
            )
        )
    except Exception as e:
        raise GeneTargetError(f"Ошибка запроса ChEMBL mechanism endpoint: {e}")

    if not mech_records:
        return []

    mol_ids = list(set(r["molecule_chembl_id"] for r in mech_records))
    moa_by_id = {r["molecule_chembl_id"]: r for r in mech_records}

    try:
        molecule = new_client.molecule
        mol_recs = list(
            molecule.filter(molecule_chembl_id__in=mol_ids, max_phase=4).only(
                ["molecule_chembl_id", "pref_name", "max_phase"]
            )
        )
    except Exception as e:
        raise GeneTargetError(f"Ошибка запроса ChEMBL molecule endpoint: {e}")

    candidates = []
    for m in mol_recs:
        moa = moa_by_id.get(m["molecule_chembl_id"], {})
        candidates.append(
            {
                "molecule_chembl_id": m["molecule_chembl_id"],
                "pref_name": m.get("pref_name") or m["molecule_chembl_id"],
                "max_phase": m.get("max_phase"),
                "mechanism_of_action": moa.get("mechanism_of_action"),
                "action_type": moa.get("action_type"),
            }
        )
    return candidates


def attach_best_potency(target_chembl_id: str, candidates: list[dict]) -> list[dict]:
    """Для каждого кандидата подтягивает лучшую (минимальную) измеренную
    активность IC50/EC50 против этой же мишени, если она есть в ChEMBL."""
    if not candidates:
        return candidates

    try:
        new_client = get_chembl_new_client()

        activity = new_client.activity
        mol_ids = [c["molecule_chembl_id"] for c in candidates]
        acts = list(
            activity.filter(
                target_chembl_id=target_chembl_id,
                molecule_chembl_id__in=mol_ids,
                standard_type__in=["IC50", "EC50", "Ki", "Kd"],
                standard_units="nM",
                standard_value__isnull=False,
            ).only(
                [
                    "molecule_chembl_id",
                    "standard_type",
                    "standard_value",
                    "standard_units",
                ]
            )
        )
    except Exception as e:
        raise GeneTargetError(f"Ошибка запроса ChEMBL activity для потентности: {e}")

    best_by_mol = {}
    for a in acts:
        try:
            value = float(a["standard_value"])
        except (TypeError, ValueError):
            continue
        cur = best_by_mol.get(a["molecule_chembl_id"])
        if cur is None or value < cur["standard_value"]:
            best_by_mol[a["molecule_chembl_id"]] = {
                "standard_type": a["standard_type"],
                "standard_value": value,
                "standard_units": a["standard_units"],
            }

    for c in candidates:
        best = best_by_mol.get(c["molecule_chembl_id"])
        if best:
            c["best_potency_nM"] = best["standard_value"]
            c["best_potency_type"] = best["standard_type"]
        else:
            c["best_potency_nM"] = None
            c["best_potency_type"] = None

    return candidates


def find_repurposing_candidates(gene_name: str) -> pd.DataFrame:
    print(f"=== Модуль 3: поиск одобренных препаратов (repurposing) для гена {gene_name} ===")

    target_info = find_chembl_target(gene_name)
    print(
        f"Мишень: {target_info['target_chembl_id']} — {target_info['pref_name']}"
    )

    candidates = get_approved_drugs_for_target(target_info["target_chembl_id"])
    print(f"Курируемых связей препарат-мишень (mechanism, max_phase=4): {len(candidates)}")

    if not candidates:
        print(
            f"НЕ НАЙДЕНО ни одного одобренного препарата с известным механизмом "
            f"действия против {target_info['target_chembl_id']} в ChEMBL."
        )
        return pd.DataFrame()

    candidates = attach_best_potency(target_info["target_chembl_id"], candidates)

    df = pd.DataFrame(candidates)
    df = df.sort_values(
        by=["best_potency_nM"], ascending=True, na_position="last"
    ).reset_index(drop=True)

    print("\nТоп-кандидаты (ранжированы по потентности, где известна):")
    for i, row in df.iterrows():
        potency = (
            f"{row['best_potency_nM']:.1f} нМ ({row['best_potency_type']})"
            if row["best_potency_nM"] is not None
            else "нет данных по IC50/EC50"
        )
        print(f"  {i+1}. {row['pref_name']} ({row['molecule_chembl_id']}) — {potency}")

    return df


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else GENE_NAME
    try:
        df = find_repurposing_candidates(gene)
    except GeneTargetError as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)

    if not df.empty:
        out_path = f"repurposing_candidates_{gene}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
