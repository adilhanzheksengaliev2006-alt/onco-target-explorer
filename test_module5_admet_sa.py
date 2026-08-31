"""
test_module5_admet_sa.py — тест SA score + ADMETlab 3.0 на маленьком наборе
=============================================================================
Прогон ДО встраивания в module_generative/iterative_finetune_loop.py, как
просили: 5-10 молекул, включая контроль (алпелисиб — уже одобренный
препарат, должен выглядеть как разумный кандидат по SA score и ADMET).

SMILES для реальных препаратов взяты из уже существующего
docking_results_PIK3CA.csv (Модуль docking, прошлый прогон) — это
настоящие структуры из ChEMBL, а не придуманные вручную (см. заметку в
module5_admet_filters.py про риск ошибиться в структуре контроля).
"""

import pandas as pd

from module5_admet_filters import (
    Admet5Error,
    SA_REASONABLE_MAX,
    compute_sa_scores,
    fetch_admet_batch,
    count_admet_red_flags,
    rank_candidates,
)

TEST_MOLECULES = [
    ("ALPELISIB (контроль, одобренный препарат)", "Cc1nc(NC(=O)N2CCC[C@H]2C(N)=O)sc1-c1ccnc(C(C)(C)C(F)(F)F)c1"),
    ("INAVOLISIB", "C[C@H](Nc1ccc2c(c1)OCCn1cc(N3C(=O)OC[C@H]3C(F)F)nc1-2)C(N)=O"),
    ("COPANLISIB", "COc1c(OCCCN2CCOCC2)ccc2c1N=C(NC(=O)c1cnc(N)nc1)N1CCN=C21"),
    ("COPANLISIB HYDROCHLORIDE", "COc1c(OCCCN2CCOCC2)ccc2c1N=C(NC(=O)c1cnc(N)nc1)N1CCN=C21.Cl.Cl"),
    ("aspirin (простая, санити-чек SA score)", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ("caffeine (простая, санити-чек SA score)", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
    ("этанол (тривиальная, нижняя граница SA score)", "CCO"),
    ("заведомо невалидный SMILES (проверка, что не роняет батч)", "C1CC(not_a_real_smiles"),
]

if __name__ == "__main__":
    names = [n for n, _ in TEST_MOLECULES]
    smiles = [s for _, s in TEST_MOLECULES]

    print("=== SA score ===")
    sa_scores = compute_sa_scores(smiles)
    for (name, smi), score in zip(TEST_MOLECULES, sa_scores.values()):
        tag = "OK" if score is not None and score <= SA_REASONABLE_MAX else ("невалиден" if score is None else "сложна для синтеза")
        print(f"  {name}: SA_score={score} ({tag})")

    print("\n=== ADMETlab 3.0 (батч через /server/screeningCal) ===")
    try:
        admet_df = fetch_admet_batch(smiles)
    except Admet5Error as e:
        print(f"ОШИБКА ADMETlab: {e}")
        raise SystemExit(1)

    rows = []
    for name, smi in TEST_MOLECULES:
        row = {"name": name, "SMILES": smi, "SA_score": sa_scores.get(smi)}
        admet_row = admet_df.loc[smi] if smi in admet_df.index else None
        if admet_row is not None:
            for c in admet_df.columns:
                row[c] = admet_row[c]
            row["ADMET_red_flags"] = count_admet_red_flags(admet_row)
        rows.append(row)
    result_df = pd.DataFrame(rows)

    print("\n--- Сводка (ключевые ADME/токс показатели) ---")
    key_cols = ["name", "SA_score", "ADMET_matched", "ADMET_red_flags",
                "ADMET_caco2", "ADMET_f20", "ADMET_PPB", "ADMET_BBB",
                "ADMET_CYP3A4-inh", "ADMET_cl-plasma", "ADMET_t0.5",
                "ADMET_hERG", "ADMET_Ames", "ADMET_DILI"]
    key_cols = [c for c in key_cols if c in result_df.columns]
    print(result_df[key_cols].to_string(index=False))

    out_path = "test_module5_admet_sa_results.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nПолный результат (SA_score + все ADMET-колонки + ADMET_uncertainty_note) сохранён: {out_path}")

    print("\n=== Композитное ранжирование (rank_candidates) ===")
    ranked = rank_candidates(result_df)
    print(ranked[["name", "SA_score", "ADMET_red_flags"]].to_string(index=False))

    alpelisib_row = result_df[result_df["name"].str.contains("ALPELISIB", case=False)].iloc[0]
    print("\n=== Проверка контроля (алпелисиб) ===")
    print(f"  SA_score = {alpelisib_row['SA_score']} (ожидается разумное значение, обычно 2-5 для одобренного препарата)")
    print(f"  ADMET_matched = {alpelisib_row['ADMET_matched']}")
    print(f"  ADMET_red_flags = {alpelisib_row['ADMET_red_flags']} (ожидается немного — это одобренный препарат)")
    if not alpelisib_row["ADMET_matched"]:
        print("  ВНИМАНИЕ: ADMETlab не смог сопоставить алпелисиб — интеграция настроена неправильно, разбираться ПЕРЕД встраиванием в полный цикл.")
