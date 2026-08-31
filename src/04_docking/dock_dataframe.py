"""
dock_dataframe.py — докует все SMILES из CSV (колонка `smiles`) параллельно
против 4JPS-кармана, дописывает колонки docking_score_kcal_mol и
heavy_atom_count. Переиспользуется Тестом A (активные+декои+неактивные) и
Тестом B (сгенерированные DiffSBDD + baseline-набор).

Устойчивость: каждая молекула в своём изолированном процессе
(dock_smiles_isolated, тот же воркер, что и в основном докинг-модуле и в
REINVENT4-компоненте), падение одной молекулы не роняет остальные, ошибки
пишутся в failed.log (общий для всего пайплайна, см. src/common/state.py).

Использование:
    python dock_dataframe.py <input.csv> <output.csv> [exhaustiveness] [n_workers]
"""

import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from dock_existing_candidates import dock_smiles_isolated  # noqa: E402

RECEPTOR_PDBQT = os.path.join(BASE_DIR, "structures", "4JPS_receptor.pdbqt")
BOX_CENTER = (-1.3186999999999998, -9.512666666666666, 16.9481)
BOX_SIZE = (20, 20, 23.114)


def _default_n_workers():
    cpu = os.cpu_count() or 4
    return max(1, cpu - 4)


def dock_dataframe(df: pd.DataFrame, exhaustiveness=8, n_workers=None, failed_log_path=None) -> pd.DataFrame:
    n_workers = n_workers or _default_n_workers()
    smiles_list = df["smiles"].tolist()

    heavy_atoms = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        heavy_atoms.append(mol.GetNumHeavyAtoms() if mol is not None else None)

    def _dock_one(args):
        i, smi = args
        with tempfile.TemporaryDirectory() as workdir:
            try:
                return dock_smiles_isolated(
                    smi, RECEPTOR_PDBQT, BOX_CENTER, BOX_SIZE, workdir,
                    tag=f"df{i}", exhaustiveness=exhaustiveness, timeout=120,
                )
            except Exception as e:
                if failed_log_path:
                    with open(failed_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tdock_dataframe\t"
                                f"idx={i}\terror={e}\n")
                return None

    scores = [None] * len(smiles_list)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, score in zip(range(len(smiles_list)), pool.map(_dock_one, enumerate(smiles_list))):
            scores[i] = score
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [dock_dataframe] {i + 1}/{len(smiles_list)} за {elapsed:.0f}с "
                      f"({elapsed / (i + 1):.2f}с/мол среднее)")

    out = df.copy()
    out["docking_score_kcal_mol"] = scores
    out["heavy_atom_count"] = heavy_atoms
    n_failed = sum(1 for s in scores if s is None)
    print(f"[dock_dataframe] Готово: {len(out) - n_failed}/{len(out)} успешно задокировано "
          f"({n_failed} не удалось)")
    return out


def main():
    input_csv, output_csv = sys.argv[1], sys.argv[2]
    exhaustiveness = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else None

    df = pd.read_csv(input_csv)
    print(f"[dock_dataframe] Докинг {len(df)} молекул из {input_csv} "
          f"(exhaustiveness={exhaustiveness}, workers={n_workers or _default_n_workers()})...")
    failed_log = os.path.join(BASE_DIR, "failed.log")
    out = dock_dataframe(df, exhaustiveness=exhaustiveness, n_workers=n_workers, failed_log_path=failed_log)
    out.to_csv(output_csv, index=False)
    print(f"[dock_dataframe] Сохранено: {output_csv}")


if __name__ == "__main__":
    main()
