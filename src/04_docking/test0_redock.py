"""
test0_redock.py — Тест 0: редокинг со-кристаллического лиганда.

Извлекает лиганд 1LT (алпелисиб) из structures/4JPS.pdb с его реальными
кристаллографическими координатами, докует ЗАНОВО ту же молекулу (SMILES
из ChEMBL, CHEMBL2396661) в тот же карман и сравнивает лучшую позу с
исходной кристаллической через RMSD (с учётом симметрии/эквивалентных
атомов — rdMolAlign.GetBestRMS). Порог: <= 2.0 A (config/protocol.yaml).

Если не проходит — карман/рецептор подготовлены неправильно, продолжать
Тест A бессмысленно (garbage in, garbage out).
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import gemmi  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem, rdMolAlign  # noqa: E402
import meeko  # noqa: E402

from dock_existing_candidates import (  # noqa: E402
    prepare_ligand_pdbqt, run_vina,
)

ALPELISIB_SMILES = "Cc1nc(NC(=O)N2CCC[C@H]2C(N)=O)sc1-c1ccnc(C(C)(C)C(F)(F)F)c1"
PDB_PATH = os.path.join(BASE_DIR, "structures", "4JPS.pdb")
RECEPTOR_PDBQT = os.path.join(BASE_DIR, "structures", "4JPS_receptor.pdbqt")
BOX_CENTER = (-1.3186999999999998, -9.512666666666666, 16.9481)
BOX_SIZE = (20, 20, 23.114)
RMSD_THRESHOLD = 2.0
EXHAUSTIVENESS = 8

OUT_DIR = os.path.join(BASE_DIR, "results", "test0_redock")
os.makedirs(OUT_DIR, exist_ok=True)


def extract_crystal_ligand_pdb(pdb_path: str, resname: str, out_pdb: str):
    """Пишет отдельный PDB-файл только с атомами лиганда resname (первая модель)."""
    st = gemmi.read_structure(pdb_path)
    st.setup_entities()
    new_st = gemmi.Structure()
    new_st.name = "ligand"
    model = gemmi.Model("1")
    for chain in st[0]:
        for residue in chain:
            if residue.name.strip() == resname:
                new_chain = gemmi.Chain(chain.name)
                new_chain.add_residue(residue)
                model.add_chain(new_chain)
    new_st.add_model(model)
    new_st.write_pdb(out_pdb)


def mol_from_crystal_pdb(pdb_path: str, template_smiles: str):
    raw = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=True)
    if raw is None:
        raise RuntimeError(f"RDKit не смог прочитать {pdb_path}")
    template = Chem.MolFromSmiles(template_smiles)
    if template is None:
        raise RuntimeError("Не удалось распарсить SMILES-шаблон алпелисиба")
    mol = AllChem.AssignBondOrdersFromTemplate(template, raw)
    return mol


def mol_from_docked_pdbqt(pdbqt_path: str, template_smiles: str):
    pdbqt_mol = meeko.PDBQTMolecule.from_file(pdbqt_path, skip_typing=True)
    rdkit_mols = meeko.RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    # берём лучшую (первую) позу первой молекулы
    best = rdkit_mols[0]
    if best.GetNumConformers() > 1:
        # RDKitMolCreate может вернуть все позы как конформеры одной молекулы
        pass
    template = Chem.MolFromSmiles(template_smiles)
    mol = AllChem.AssignBondOrdersFromTemplate(template, Chem.RemoveHs(best))
    return mol


def main():
    print("[test0_redock] Извлечение кристаллической позы лиганда 1LT из 4JPS.pdb...")
    crystal_pdb = os.path.join(OUT_DIR, "crystal_ligand_1LT.pdb")
    extract_crystal_ligand_pdb(PDB_PATH, "1LT", crystal_pdb)
    mol_crystal = mol_from_crystal_pdb(crystal_pdb, ALPELISIB_SMILES)
    print(f"  Кристаллическая поза: {mol_crystal.GetNumAtoms()} тяжёлых атомов")

    print(f"[test0_redock] Редокинг алпелисиба (exhaustiveness={EXHAUSTIVENESS})...")
    with tempfile.TemporaryDirectory() as workdir:
        ligand_pdbqt = os.path.join(workdir, "redock.pdbqt")
        out_pdbqt = os.path.join(workdir, "redock_out.pdbqt")
        ok = prepare_ligand_pdbqt(ALPELISIB_SMILES, ligand_pdbqt)
        if not ok:
            raise RuntimeError("Не удалось подготовить лиганд алпелисиба для докинга")
        score = run_vina(
            RECEPTOR_PDBQT, ligand_pdbqt, BOX_CENTER, BOX_SIZE, out_pdbqt,
            exhaustiveness=EXHAUSTIVENESS, timeout=300,
        )
        if score is None or not os.path.exists(out_pdbqt):
            raise RuntimeError("Vina не вернула результат для редокинга")
        print(f"  Docking score: {score:.2f} ккал/моль")

        mol_docked = mol_from_docked_pdbqt(out_pdbqt, ALPELISIB_SMILES)
        # сохраняем результирующие структуры для отчёта до удаления workdir
        import shutil
        shutil.copy(out_pdbqt, os.path.join(OUT_DIR, "redocked_pose.pdbqt"))

    rmsd = rdMolAlign.GetBestRMS(mol_docked, mol_crystal)
    passed = rmsd <= RMSD_THRESHOLD

    result = {
        "docking_score_kcal_mol": score,
        "rmsd_angstrom": rmsd,
        "threshold_angstrom": RMSD_THRESHOLD,
        "passed": passed,
        "exhaustiveness": EXHAUSTIVENESS,
        "ligand": "alpelisib (CHEMBL2396661)",
        "receptor": "4JPS chain A, ATP-binding pocket",
    }
    with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    verdict = "ПРОЙДЕН" if passed else "НЕ ПРОЙДЕН"
    print(f"\n=== Тест 0: {verdict} ===")
    print(f"RMSD = {rmsd:.3f} A (порог {RMSD_THRESHOLD} A)")
    print(f"Docking score = {score:.2f} ккал/моль")
    if not passed:
        print("ВНИМАНИЕ: редокинг не прошёл порог. Дальнейшие тесты (A/N/A') "
              "будут диагностическими, не подтверждающими -- см. протокол.")

    return result


if __name__ == "__main__":
    main()
