"""
vina_variance.py — собственная дисперсия Vina: 10 лигандов x 5 сидов
(--seed передаётся в vina.exe), стандартное отклонение скора на молекулу.
Без этого числа неясно, значима ли разница средних в Тесте B.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from dock_existing_candidates import prepare_ligand_pdbqt, VINA_EXE  # noqa: E402
import re  # noqa: E402

RECEPTOR_PDBQT = os.path.join(BASE_DIR, "structures", "4JPS_receptor.pdbqt")
BOX_CENTER = (-1.3186999999999998, -9.512666666666666, 16.9481)
BOX_SIZE = (20, 20, 23.114)
N_LIGANDS = 10
SEEDS = [1, 2, 3, 4, 5]
EXHAUSTIVENESS = 8

OUT_DIR = os.path.join(BASE_DIR, "results", "metrics")
os.makedirs(OUT_DIR, exist_ok=True)


def run_vina_with_seed(receptor_pdbqt, ligand_pdbqt, box_center, box_size, out_pdbqt, seed, exhaustiveness=8):
    cmd = [
        VINA_EXE, "--receptor", receptor_pdbqt, "--ligand", ligand_pdbqt,
        "--center_x", str(box_center[0]), "--center_y", str(box_center[1]), "--center_z", str(box_center[2]),
        "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
        "--out", out_pdbqt, "--exhaustiveness", str(exhaustiveness), "--seed", str(seed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    combined = result.stdout + result.stderr
    matches = re.findall(r"^\s*\d+\s+(-?\d+\.\d+)", combined, re.MULTILINE)
    return float(matches[0]) if matches else None


def main():
    # 10 разных активных молекул из ChEMBL-курации Теста A, если уже готова;
    # иначе -- алпелисиб + несколько других контролей как fallback
    actives_path = os.path.join(BASE_DIR, "data", "processed", "test_a_actives_excl_alpelisib.csv")
    if os.path.exists(actives_path):
        df = pd.read_csv(actives_path)
        smiles_list = df["smiles"].head(N_LIGANDS).tolist()
    else:
        smiles_list = [
            "Cc1nc(NC(=O)N2CCC[C@H]2C(N)=O)sc1-c1ccnc(C(C)(C)C(F)(F)F)c1",  # alpelisib
        ] * N_LIGANDS

    rows = []
    with tempfile.TemporaryDirectory() as workdir:
        for i, smi in enumerate(smiles_list):
            ligand_pdbqt = os.path.join(workdir, f"lig{i}.pdbqt")
            if not prepare_ligand_pdbqt(smi, ligand_pdbqt):
                continue
            scores = []
            for seed in SEEDS:
                out_pdbqt = os.path.join(workdir, f"lig{i}_s{seed}_out.pdbqt")
                score = run_vina_with_seed(RECEPTOR_PDBQT, ligand_pdbqt, BOX_CENTER, BOX_SIZE,
                                            out_pdbqt, seed, EXHAUSTIVENESS)
                scores.append(score)
                print(f"  лиганд {i}, seed {seed}: {score}")
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                rows.append({
                    "ligand_idx": i, "smiles": smi,
                    "scores": valid_scores, "mean": float(np.mean(valid_scores)),
                    "std": float(np.std(valid_scores)),
                })

    all_stds = [r["std"] for r in rows]
    result = {
        "n_ligands": len(rows), "n_seeds": len(SEEDS),
        "per_ligand": rows,
        "mean_std_across_ligands": float(np.mean(all_stds)) if all_stds else None,
        "max_std_across_ligands": float(np.max(all_stds)) if all_stds else None,
    }
    with open(os.path.join(OUT_DIR, "vina_own_variance.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[vina_variance] Средний std по {len(rows)} лигандам: {result['mean_std_across_ligands']}")
    return result


if __name__ == "__main__":
    main()
