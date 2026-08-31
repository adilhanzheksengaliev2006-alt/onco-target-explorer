"""
resume_from_cycle.py — продолжает прерванный прогон iterative_finetune_loop.py
с последнего сохранённого чекпоинта, не редактируя сам pipeline-файл
(он сейчас параллельно дорабатывается в другом чате — SA score/ADMET).

Использование: python resume_from_cycle.py <GENE_NAME> <START_CYCLE>
Пример: python resume_from_cycle.py PIK3CA 13
  -> продолжит с чекпоинта checkpoints/PIK3CA/cycle_12, начиная с цикла 13.

Причина: фоновый прогон (bvml1g30x, PIK3CA/4JPS, 500x20) был убит
извне (не программной ошибкой — в логе нет traceback, процесс просто
исчез) на 13-м цикле. 12 циклов уже отработали и сохранены
(checkpoints/PIK3CA/cycle_1..cycle_12, results/cycle_1..12_results_PIK3CA.csv)
— перезапуск с нуля потерял бы эту работу.
"""

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "module_generative"))

import iterative_finetune_loop as ifl  # noqa: E402


def resume(gene_name: str, start_cycle: int):
    print(f"########## Продолжение прогона: {gene_name}, с цикла {start_cycle} ##########")
    print(f"N_CYCLES={ifl.N_CYCLES}, молекул/цикл={ifl.N_GENERATE_PER_CYCLE}\n")

    ckpt_dir_prev = os.path.join(ifl.CHECKPOINTS_DIR, gene_name, f"cycle_{start_cycle - 1}")
    if not os.path.isdir(ckpt_dir_prev):
        print(f"ОШИБКА: чекпоинт {ckpt_dir_prev} не найден — не могу продолжить с цикла {start_cycle}.")
        return
    print(f"Стартовая модель: {ckpt_dir_prev}\n")

    try:
        target_info = ifl.resolve_gene_to_docking_target(gene_name)
    except ifl.GeneTargetError as e:
        print(f"ОШИБКА поиска мишени/структуры: {e}")
        return
    if target_info is None:
        print(f"Докинг-структура для {gene_name} не найдена — остановлено.")
        return

    receptor_basename = os.path.join(
        os.path.dirname(target_info["pdb_path"]), f"{target_info['pdb_id']}_receptor"
    )
    try:
        receptor_pdbqt = ifl.prepare_receptor(
            target_info["pdb_path"], target_info["box_center"], target_info["box_size"], receptor_basename
        )
    except ifl.DockingError as e:
        print(f"ОШИБКА подготовки рецептора: {e}")
        return
    print(f"Рецептор готов: {receptor_pdbqt}\n")

    try:
        ifl.run_control_docking(target_info, receptor_pdbqt)
    except ifl.PipelineError as e:
        print(f"ОШИБКА КОНТРОЛЯ: {e}")
        return

    try:
        tox_catalog = ifl.build_toxicity_catalog()
    except ifl.PipelineError as e:
        print(f"ОШИБКА: {e}")
        return

    current_model_path = ckpt_dir_prev
    all_cycle_summaries = []

    for cycle in range(start_cycle, ifl.N_CYCLES + 1):
        t0 = time.time()
        try:
            counts, result_df, model, tokenizer = ifl.run_cycle(
                cycle, current_model_path, gene_name, target_info, receptor_pdbqt, tox_catalog
            )
        except ifl.PipelineError as e:
            print(f"ОШИБКА в цикле {cycle}: {e}")
            break

        elapsed = time.time() - t0
        summary = {
            "cycle": cycle,
            "model_path": current_model_path,
            **counts,
            "elapsed_sec": round(elapsed, 1),
        }
        all_cycle_summaries.append(summary)

        print(
            f"\n--- Сводка цикла {cycle} ---\n"
            f"  сгенерировано:        {counts['generated']}\n"
            f"  прошло валидность:    {counts['valid']}\n"
            f"  прошло докинг:        {counts['docked_pass']}\n"
            f"  прошло токсичность:   {counts['tox_pass']}\n"
            f"  прошло селективность: {counts['selectivity_pass']}\n"
            f"  время цикла:          {elapsed:.1f}с"
        )

        cycle_csv = os.path.join(ifl.RESULTS_DIR, f"cycle_{cycle}_results_{gene_name}.csv")
        result_df.to_csv(cycle_csv, index=False)
        print(f"  сохранено: {cycle_csv} ({len(result_df)} молекул)")

        if result_df.empty:
            print(f"  В цикле {cycle} НИ ОДНА молекула не прошла все фильтры — дообучение ПРОПУЩЕНО.")
            continue

        ckpt_dir = os.path.join(ifl.CHECKPOINTS_DIR, gene_name, f"cycle_{cycle}")
        print(f"  Дообучение MolGPT на {len(result_df)} молекулах...")
        try:
            model, avg_loss = ifl.finetune_model(
                model, tokenizer, result_df["SMILES"].tolist(), ifl.FINETUNE_EPOCHS, ifl.FINETUNE_LR
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Чекпоинт сохранён: {ckpt_dir}" + (f" (средний loss: {avg_loss:.4f})" if avg_loss is not None else ""))
            current_model_path = ckpt_dir
        except Exception as e:
            print(f"  ОШИБКА дообучения/сохранения чекпоинта: {e} — следующий цикл продолжит с прежней моделью.")

    print(f"\n{'=' * 70}\nИТОГ ЦИКЛОВ {start_cycle}-{ifl.N_CYCLES} ({gene_name})\n{'=' * 70}")
    summary_df = pd.DataFrame(all_cycle_summaries)
    if summary_df.empty:
        print("Ни одного нового цикла не завершилось успешно.")
    else:
        print(summary_df.to_string(index=False))
        total_pass = summary_df["selectivity_pass"].sum()
        print(f"\nВсего молекул, прошедших все фильтры (циклы {start_cycle}-{ifl.N_CYCLES}): {total_pass}")
        summary_csv = os.path.join(ifl.RESULTS_DIR, f"run_summary_resumed_{gene_name}_from{start_cycle}.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Сводка сохранена: {summary_csv}")


if __name__ == "__main__":
    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    resume(gene, start)
