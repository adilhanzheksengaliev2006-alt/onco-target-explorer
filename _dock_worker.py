"""
_dock_worker.py — вспомогательный воркер: докинг ОДНОЙ молекулы в
отдельном процессе.

Зачем отдельный процесс, а не просто вызов функции: у небольшой,
недообученной MolGPT иногда проскакивают синтаксически валидные, но
химически патологические SMILES (сильно сшитые/большие циклические
системы). На таких молекулах RDKit-embedding (построение 3D) или сам
Vina могут зависать на много минут — это не баг нашего кода, а
известная особенность RDKit/Vina на вырожденных графах. Запуская
докинг каждой молекулы в отдельном процессе, мы можем гарантированно
прибить его по таймауту на уровне ОС (subprocess timeout), не
дожидаясь зависшего шага и не роняя весь итеративный цикл.

Использование: python _dock_worker.py <smiles> <receptor_pdbqt>
                 <cx> <cy> <cz> <sx> <sy> <sz> <out_pdbqt> <exhaustiveness>
Печатает "SCORE=<float>" при успехе или "FAIL" при неудаче.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dock_existing_candidates import prepare_ligand_pdbqt, run_vina  # noqa: E402


def main():
    (
        smiles, receptor_pdbqt,
        cx, cy, cz, sx, sy, sz,
        out_pdbqt, exhaustiveness,
    ) = sys.argv[1:12]

    box_center = (float(cx), float(cy), float(cz))
    box_size = (float(sx), float(sy), float(sz))
    ligand_pdbqt = out_pdbqt.replace("_out.pdbqt", ".pdbqt")

    ok = prepare_ligand_pdbqt(smiles, ligand_pdbqt)
    if not ok:
        print("FAIL")
        return

    score = run_vina(
        receptor_pdbqt, ligand_pdbqt, box_center, box_size, out_pdbqt,
        exhaustiveness=int(exhaustiveness), timeout=None,  # таймаут снаружи, на уровне процесса
    )
    if score is None:
        print("FAIL")
    else:
        print(f"SCORE={score}")


if __name__ == "__main__":
    main()
