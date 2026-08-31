"""
analyze_mutations.py — Модуль 1: анализ мутаций для гена GENE_NAME
=========================================================================
Источник данных: когорта METABRIC (рак молочной железы, ~2000
пациентов, data/metabric_data/). Это ОДНА конкретная когорта — для
генов, не характерных для рака молочной железы (например ABL1, ALK),
частота мутаций в этих данных может быть близка к нулю. Это не баг,
а честный результат: METABRIC не покрывает все виды рака.

Задаётся один параметр GENE_NAME — остальное (частота мутаций, типы
замен, влияние на выживаемость) считается автоматически.
"""

import os
import sqlite3
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUTATIONS_FILE = os.path.join(BASE_DIR, "data", "metabric_data", "data_mutations.txt")
CLINICAL_FILE = os.path.join(BASE_DIR, "data", "metabric_data", "data_clinical_patient.txt")
DB_PATH = os.path.join(BASE_DIR, "cancer_data.db")

GENE_NAME = "PIK3CA"


class MutationAnalysisError(Exception):
    pass


def _load_data():
    if not os.path.exists(MUTATIONS_FILE) or not os.path.exists(CLINICAL_FILE):
        raise MutationAnalysisError(
            f"Не найдены файлы METABRIC ({MUTATIONS_FILE}, {CLINICAL_FILE})."
        )
    try:
        mutations = pd.read_csv(MUTATIONS_FILE, sep="\t", comment="#", low_memory=False)
        clinical = pd.read_csv(CLINICAL_FILE, sep="\t", comment="#", low_memory=False)
    except Exception as e:
        raise MutationAnalysisError(f"Ошибка чтения METABRIC файлов: {e}")

    patient_ids = clinical["PATIENT_ID"].astype(str).tolist()
    patient_ids_sorted = sorted(patient_ids, key=len, reverse=True)

    def find_patient(sample):
        sample = str(sample)
        for patient in patient_ids_sorted:
            if sample.startswith(patient):
                return patient
        return None

    mutations["PATIENT_ID"] = mutations["Tumor_Sample_Barcode"].apply(find_patient)
    return mutations, clinical


def _build_db(mutations: pd.DataFrame, clinical: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    mutations.to_sql("mutations", conn, if_exists="replace", index=False)
    clinical.to_sql("clinical", conn, if_exists="replace", index=False)
    return conn


def analyze_mutations(gene_name: str) -> dict:
    print(f"=== Модуль 1: анализ мутаций гена {gene_name} (когорта METABRIC) ===")

    mutations, clinical = _load_data()
    n_total_patients = clinical["PATIENT_ID"].nunique()
    print(f"Всего пациентов в когорте: {n_total_patients}")

    conn = _build_db(mutations, clinical)
    try:
        gene_mutations = pd.read_sql(
            "SELECT * FROM mutations WHERE Hugo_Symbol = ?", conn, params=(gene_name,)
        )
        n_mutated_patients = gene_mutations["PATIENT_ID"].nunique()
        frequency = n_mutated_patients / n_total_patients if n_total_patients else 0
        print(
            f"Пациентов с мутацией в {gene_name}: {n_mutated_patients}/{n_total_patients} "
            f"({frequency:.1%})"
        )

        if n_mutated_patients == 0:
            print(
                f"В когорте METABRIC (рак молочной железы) не найдено ни одной "
                f"мутации в {gene_name}. Это ожидаемо для генов, не характерных "
                f"для рака молочной железы — анализ выживаемости пропущен."
            )
            variant_counts = {}
            survival = pd.DataFrame()
        else:
            variant_counts = (
                gene_mutations["Variant_Classification"].value_counts().to_dict()
                if "Variant_Classification" in gene_mutations.columns
                else {}
            )
            print(f"Типы замен: {variant_counts}")

            survival = pd.read_sql(
                f"""
                SELECT
                    CASE WHEN mut.PATIENT_ID IS NOT NULL THEN 'Mutation' ELSE 'No mutation' END AS group_name,
                    COUNT(DISTINCT clinical.PATIENT_ID) AS patients,
                    ROUND(AVG(clinical.OS_MONTHS), 1) AS average_survival_months
                FROM clinical
                LEFT JOIN (
                    SELECT DISTINCT PATIENT_ID FROM mutations WHERE Hugo_Symbol = ?
                ) AS mut
                ON clinical.PATIENT_ID = mut.PATIENT_ID
                GROUP BY group_name
                """,
                conn,
                params=(gene_name,),
            )
            print("\nСредняя выживаемость (мес.) по группам:")
            print(survival.to_string(index=False))
    finally:
        conn.close()

    return {
        "gene_name": gene_name,
        "n_total_patients": n_total_patients,
        "n_mutated_patients": n_mutated_patients,
        "frequency": frequency,
        "variant_counts": variant_counts,
        "survival": survival,
    }


def main():
    gene = sys.argv[1] if len(sys.argv) > 1 else GENE_NAME
    try:
        analyze_mutations(gene)
    except MutationAnalysisError as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
