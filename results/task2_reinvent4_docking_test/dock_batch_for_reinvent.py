"""
dock_batch_for_reinvent.py — батч-докинг для REINVENT4 custom scoring
component (comp_pik3ca_dock.py в module_generative/REINVENT4/reinvent_plugins/components/).

Вызывается как ВНЕШНИЙ процесс (по образцу встроенного DockStream-компонента
REINVENT4, reinvent_plugins/components/comp_dockstream.py) через ОТДЕЛЬНЫЙ
python-интерпретатор — системный, где уже стоят rdkit/meeko/gemmi и есть
tools/vina.exe (окружение REINVENT4 .venv их не содержит и не должно:
это исследовательский RL-стек, докинг — отдельная зависимость).

Переиспользует dock_smiles_isolated() из dock_existing_candidates.py
(тот же изолированный воркер-процесс на молекулу с жёстким OS-таймаутом,
что и в основном докинг-модуле) — карман и рецептор НЕ пересчитываются
заново, только ссылаются на уже готовый structures/4JPS_receptor.pdbqt
и координаты бокса из find_ligand_center().

Использование:
    python dock_batch_for_reinvent.py <receptor_pdbqt> <cx> <cy> <cz>
        <sx> <sy> <sz> <exhaustiveness> <smiles1;smiles2;...>

Печатает на stdout одну аффинность (ккал/моль, float) на строку в том же
порядке, что и входные SMILES; "NA" для молекул, которые не удалось
задокировать (не считается фатальной ошибкой — как и везде в пайплайне).
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dock_existing_candidates import dock_smiles_isolated  # noqa: E402


def main():
    (
        receptor_pdbqt,
        cx, cy, cz,
        sx, sy, sz,
        exhaustiveness,
        smiles_joined,
    ) = sys.argv[1:10]

    box_center = (float(cx), float(cy), float(cz))
    box_size = (float(sx), float(sy), float(sz))
    smiles_list = smiles_joined.split(";")

    with tempfile.TemporaryDirectory() as workdir:
        for i, smi in enumerate(smiles_list):
            score = dock_smiles_isolated(
                smi, receptor_pdbqt, box_center, box_size, workdir,
                tag=f"rl{i}", exhaustiveness=int(exhaustiveness), timeout=90,
            )
            print(score if score is not None else "NA")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
