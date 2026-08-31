"""
generate_molecules_v1.py — генерация SMILES через MolGPT
=========================================================================
Стек: transformers + torch (MolGPT с HuggingFace), RDKit.
Реально запущено и проверено на этой машине (conda env molgen).

Изменение по сравнению с исходной версией: load_molgpt() теперь
принимает необязательный model_name_or_path — это нужно
module_generative/iterative_finetune_loop.py, чтобы на 2-м и
последующих циклах загружать не оригинальный чекпоинт с HuggingFace
Hub, а локальный чекпоинт, дообученный на предыдущем цикле.
"""

import pandas as pd

DEFAULT_MODEL_NAME = "jonghyunlee/MolGPT_pretrained-by-ZINC15"
N_GENERATE = 500
TOP_N_OUTPUT = 20


def load_molgpt(model_name_or_path: str = DEFAULT_MODEL_NAME):
    from transformers import GPT2LMHeadModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    model.eval()
    return model, tokenizer


def generate_raw_smiles(model, tokenizer, n=N_GENERATE, temperature=1.0, batch_size=50):
    """Генерирует n сырых SMILES-строк батчами, чтобы не упереться
    в память на слабом железе."""
    import torch

    all_smiles = []
    with torch.no_grad():
        while len(all_smiles) < n:
            batch_n = min(batch_size, n - len(all_smiles))
            outputs = model.generate(
                max_length=128,
                num_return_sequences=batch_n,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            batch = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            all_smiles.extend(batch)
            print(f"  сгенерировано {len(all_smiles)}/{n}")
    return all_smiles[:n]


def filter_valid_unique(smiles_list):
    """Часть сгенерированных строк не будут настоящими молекулами —
    это нормальное, задокументированное свойство таких моделей."""
    from rdkit import Chem

    valid, seen = [], set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical not in seen:
            seen.add(canonical)
            valid.append(canonical)
    rate = len(valid) / len(smiles_list) if smiles_list else 0
    print(f"Валидных уникальных молекул: {len(valid)}/{len(smiles_list)} ({rate:.0%})")
    return valid


def compute_descriptors_and_lipinski(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mw, logp = Descriptors.MolWt(mol), Descriptors.MolLogP(mol)
        hdon, hacc = Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)
        rows.append(
            {
                "SMILES": smi,
                "MW": round(mw, 1),
                "LogP": round(logp, 2),
                "HDonors": hdon,
                "HAcceptors": hacc,
                "Lipinski_Violations": count_lipinski_violations(mw, logp, hdon, hacc),
            }
        )
    return pd.DataFrame(rows)


def count_lipinski_violations(mw, logp, hdonors, hacceptors):
    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if hdonors > 5:
        violations += 1
    if hacceptors > 10:
        violations += 1
    return violations


def main():
    print(f"=== Генерация {N_GENERATE} молекул через MolGPT ===")
    model, tokenizer = load_molgpt()
    raw_smiles = generate_raw_smiles(model, tokenizer, n=N_GENERATE)

    valid_smiles = filter_valid_unique(raw_smiles)
    if not valid_smiles:
        print("Ни одной валидной молекулы не получилось.")
        return

    result_df = compute_descriptors_and_lipinski(valid_smiles)
    result_df = result_df.sort_values("Lipinski_Violations").head(TOP_N_OUTPUT)

    result_df.to_csv("generated_molecules.csv", index=False)
    print(f"\nСохранено: generated_molecules.csv ({len(result_df)} молекул)")


if __name__ == "__main__":
    main()
