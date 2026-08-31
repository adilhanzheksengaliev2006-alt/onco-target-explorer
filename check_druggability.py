"""
check_druggability.py — Модуль 2: проверка "драгуемости" мишени в ChEMBL
=========================================================================
Задаётся один параметр GENE_NAME. Дальше всё определяется автоматически:
  1. Ищем мишень в ChEMBL (gene_target_utils.find_chembl_target)
  2. Считаем количество известных активностей (pChEMBL) против неё —
     прокси того, насколько мишень изучена медицинскими химиками
  3. Считаем, сколько ОДОБРЕННЫХ препаратов (max_phase=4) уже имеют
     измеренную активность против этой мишени — прямой сигнал
     "есть ли смысл искать среди одобренных препаратов" (Модуль 3)

Никаких докинг-структур здесь не ищем — это отдельно, в
gene_target_utils.resolve_gene_to_docking_target() /
dock_existing_candidates.py.
"""

import sys

from gene_target_utils import GeneTargetError, find_chembl_target, get_chembl_new_client

GENE_NAME = "PIK3CA"  # единственный параметр; меняется здесь или через CLI-аргумент

MIN_ACTIVITIES_FOR_DRUGGABLE = 50


def count_activities(target_chembl_id: str) -> int:
    try:
        new_client = get_chembl_new_client()

        activity = new_client.activity
        qs = activity.filter(
            target_chembl_id=target_chembl_id, pchembl_value__isnull=False
        ).only(["molecule_chembl_id"])
        return len(qs)
    except Exception as e:
        raise GeneTargetError(f"Ошибка подсчёта активностей ChEMBL: {e}")


def count_approved_drug_activities(target_chembl_id: str) -> int:
    """Использует ChEMBL mechanism endpoint (курируемые связи
    препарат-мишень) — быстрее и точнее, чем перебор тысяч записей
    activity вручную."""
    try:
        new_client = get_chembl_new_client()

        mechanism = new_client.mechanism
        mech_records = list(
            mechanism.filter(target_chembl_id=target_chembl_id).only(
                ["molecule_chembl_id"]
            )
        )
        mol_ids = list(set(r["molecule_chembl_id"] for r in mech_records))
        if not mol_ids:
            return 0

        molecule = new_client.molecule
        recs = molecule.filter(
            molecule_chembl_id__in=mol_ids, max_phase=4
        ).only(["molecule_chembl_id"])
        return len(list(recs))
    except Exception as e:
        raise GeneTargetError(f"Ошибка подсчёта одобренных препаратов ChEMBL: {e}")


def check_druggability(gene_name: str) -> dict:
    print(f"=== Модуль 2: проверка драгуемости мишени для гена {gene_name} ===")

    target_info = find_chembl_target(gene_name)
    print(
        f"Мишень: {target_info['target_chembl_id']} — {target_info['pref_name']} "
        f"(тип: {target_info['target_type']}, UniProt {target_info['uniprot_accession']})"
    )

    n_activities = count_activities(target_info["target_chembl_id"])
    print(f"Известных активностей (pChEMBL) против мишени: {n_activities}")

    n_approved = count_approved_drug_activities(target_info["target_chembl_id"])
    print(f"Из них с одобренными препаратами (max_phase=4): {n_approved}")

    if n_activities == 0:
        verdict = "НЕТ ДАННЫХ: в ChEMBL нет измеренной активности против этой мишени — драгуемость неизвестна."
    elif n_activities < MIN_ACTIVITIES_FOR_DRUGGABLE:
        verdict = f"ОГРАНИЧЕННЫЕ ДАННЫЕ: только {n_activities} записей активности — мишень мало изучена."
    else:
        verdict = f"ДРАГУЕМА: {n_activities} записей активности, мишень активно изучается медхимиками."

    print(f"\nВЕРДИКТ: {verdict}")

    return {
        "gene_name": gene_name,
        "target_chembl_id": target_info["target_chembl_id"],
        "pref_name": target_info["pref_name"],
        "uniprot_accession": target_info["uniprot_accession"],
        "n_activities": n_activities,
        "n_approved_drug_activities": n_approved,
        "verdict": verdict,
    }


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else GENE_NAME
    try:
        check_druggability(gene)
    except GeneTargetError as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
