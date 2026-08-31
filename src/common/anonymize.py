"""
anonymize.py — обезличивание перед коммитом в публичный репозиторий.

ligand_id = sha256(канонический SMILES)[:16]. Публичные копии содержат
id, скор, число тяжёлых атомов, LE, метку класса -- без SMILES/InChI.

ПРИМЕЧАНИЕ: отдельного приватного GitHub-репозитория в этом пилоте не
настроено (нет URL/учётных данных пользователя, см. config/paths.yaml).
Файлы с полными SMILES (ligand_id -> SMILES маппинг, позы) остаются
ТОЛЬКО локально (в .gitignore), не коммитятся вообще, пока пользователь
не даст доступ ко второму репозиторию -- это безопаснее, чем коммитить
их в единственный имеющийся (публичный по умолчанию) репозиторий.
"""

import hashlib
import os

import pandas as pd
from rdkit import Chem

PUBLIC_SAFE_COLUMNS = {
    "docking_score_kcal_mol", "heavy_atom_count", "ligand_efficiency",
    "label", "seed", "group", "mw", "alogp", "hbd", "hba", "rtb",
    "alpelisib_series", "matched_active_chembl_id",
}


def ligand_id(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol) if mol is not None else smiles
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def anonymize_csv(input_csv: str, output_csv: str, smiles_col: str = "smiles"):
    """Пишет публично-безопасную копию: smiles -> ligand_id, остальные
    столбцы -- только те, что в PUBLIC_SAFE_COLUMNS (плюс chembl_id, он не
    раскрывает структуру сам по себе, только идентификатор в базе)."""
    df = pd.read_csv(input_csv)
    out = pd.DataFrame()
    if smiles_col in df.columns:
        out["ligand_id"] = df[smiles_col].apply(lambda s: ligand_id(s) if isinstance(s, str) else None)
    for col in df.columns:
        if col in PUBLIC_SAFE_COLUMNS or col == "chembl_id":
            out[col] = df[col]
    out.to_csv(output_csv, index=False)
    return out


def anonymize_all(base_dir: str):
    """Проходит по известным файлам с SMILES в data/processed и
    results/testB_generation, пишет *_public.csv рядом."""
    targets = [
        os.path.join(base_dir, "data", "processed", "test_a_docked.csv"),
        os.path.join(base_dir, "data", "processed", "test_a_docked_experimental_secondary.csv"),
        os.path.join(base_dir, "results", "testB_generation", "generated_docked.csv"),
        os.path.join(base_dir, "results", "testB_generation", "baseline_docked.csv"),
    ]
    written = []
    for path in targets:
        if not os.path.exists(path):
            continue
        out_path = path.rsplit(".", 1)[0] + "_public.csv"
        try:
            anonymize_csv(path, out_path)
            written.append(out_path)
        except Exception as e:
            print(f"[anonymize] не удалось обработать {path}: {e}")
    return written


if __name__ == "__main__":
    import sys
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for p in anonymize_all(base):
        print("написано:", p)
