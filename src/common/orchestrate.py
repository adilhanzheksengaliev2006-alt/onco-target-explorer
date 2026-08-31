"""
orchestrate.py — автономный запуск всего пилотного пайплайна по порядку:
Test 0 -> Test A (curation + docking + метрики) -> Test N -> Test A'
(на ChEMBL и на генерации) -> vina_variance -> Test B.

УПРОЩЕНИЯ ОТНОСИТЕЛЬНО ПОЛНОЙ СПЕЦИФИКАЦИИ (задокументировано честно):
- Устойчивость к падению процесса/перезагрузке машины через
  Windows Task Scheduler НЕ реализована -- вместо этого один долгоживущий
  python-процесс с try/except на каждом этапе (падение этапа не убивает
  весь прогон, следующий этап всё равно запускается). Считается
  достаточным, т.к. по вводным машина гарантированно не выключается
  48 часов.
- git push в удалённый GitHub-репозиторий НЕ настроен (нет URL/учётных
  данных пользователя) -- коммиты идут только локально, что явно
  отмечается в STATUS.md.
- CONTROL.json проверяется перед КАЖДЫМ этапом (не каждые 30 минут внутри
  этапа) -- для этапов такой длительности, как здесь, разница
  несущественна.

Идемпотентность: каждый этап пропускается, если его result.json/выходной
файл уже существует (можно перезапускать скрипт без потери прогресса).
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src", "05_metrics"))

STATE_PATH = os.path.join(BASE_DIR, "state.json")
CONTROL_PATH = os.path.join(BASE_DIR, "CONTROL.json")
STATUS_PATH = os.path.join(BASE_DIR, "STATUS.md")
RUN_LOG_PATH = os.path.join(BASE_DIR, "run.log")
FAILED_LOG_PATH = os.path.join(BASE_DIR, "failed.log")
PROTOCOL_PATH = os.path.join(BASE_DIR, "config", "protocol.yaml")


def log(msg: str):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line)
    with open(RUN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def protocol_hash():
    with open(PROTOCOL_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_state(max_wallclock_hours: float):
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        current_hash = protocol_hash()
        if state.get("protocol_sha256") != current_hash:
            log(f"ОТКАЗ СТАРТА: config/protocol.yaml изменился после первого запуска "
                f"(было {state.get('protocol_sha256')}, стало {current_hash}). "
                f"Предрегистрация теряет смысл, если пороги можно менять по ходу.")
            sys.exit(1)
        return state
    deadline = (datetime.now() + timedelta(hours=max_wallclock_hours)).isoformat()
    state = {
        "started_at": datetime.now().isoformat(),
        "deadline": deadline,
        "protocol_sha256": protocol_hash(),
        "stages_completed": [],
        "stage_timings": {},
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return state


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def check_control():
    if not os.path.exists(CONTROL_PATH):
        with open(CONTROL_PATH, "w", encoding="utf-8") as f:
            json.dump({"abort": False, "skip_stage": None, "reduce_scope": None, "note": ""}, f, indent=2)
        return {"abort": False, "skip_stage": None}
    try:
        with open(CONTROL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"abort": False, "skip_stage": None}


def git_commit(message: str):
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, "src", "common"))
        import anonymize
        for p in anonymize.anonymize_all(BASE_DIR):
            log(f"обезличенная публичная копия: {p}")
    except Exception as e:
        log(f"anonymize перед коммитом не удался (не блокирует коммит): {e}")
    try:
        subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, capture_output=True, timeout=60)
        subprocess.run(["git", "commit", "-m", message], cwd=BASE_DIR, capture_output=True, timeout=60)
    except Exception as e:
        log(f"git commit не удался: {e}")
    try:
        push = subprocess.run(["git", "push", "origin", "main"], cwd=BASE_DIR,
                               capture_output=True, text=True, timeout=120)
        if push.returncode != 0:
            log(f"git push не удался (продолжаю без остановки): {push.stderr[-300:]}")
        else:
            log("git push: ok")
    except Exception as e:
        log(f"git push не удался: {e}")


def update_status(state, stage_name: str, stage_status: str):
    deadline = datetime.fromisoformat(state["deadline"])
    hours_left = (deadline - datetime.now()).total_seconds() / 3600
    n_failed = 0
    if os.path.exists(FAILED_LOG_PATH):
        with open(FAILED_LOG_PATH, "r", encoding="utf-8") as f:
            n_failed = sum(1 for _ in f)

    lines = [
        "# STATUS",
        "",
        f"Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Дедлайн: {state['deadline']} (осталось {hours_left:.1f} ч.)",
        f"Текущий этап: {stage_name} -- {stage_status}",
        f"Завершённые этапы: {', '.join(state['stages_completed']) or '(ещё ни один)'}",
        f"Ошибок в failed.log: {n_failed}",
        "",
        "**git remote не настроен** -- push в GitHub не выполняется, "
        "изменения только в локальном репозитории.",
        "",
        "## Тайминги этапов",
    ]
    for name, t in state.get("stage_timings", {}).items():
        lines.append(f"- {name}: {t:.1f} мин")

    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_stage(state, name: str, output_marker: str, fn):
    if name in state["stages_completed"] and os.path.exists(output_marker):
        log(f"[{name}] уже завершён, пропуск")
        return True

    control = check_control()
    if control.get("abort"):
        log(f"[{name}] CONTROL.json: abort=true, останавливаюсь перед этим этапом")
        return False
    if control.get("skip_stage") == name:
        log(f"[{name}] CONTROL.json: явно пропущен по запросу")
        return True

    log(f"[{name}] старт")
    update_status(state, name, "выполняется")
    t0 = time.time()
    try:
        fn()
        ok = True
    except Exception as e:
        log(f"[{name}] ОШИБКА: {e}\n{traceback.format_exc()}")
        with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}\tSTAGE_FAILURE\t{name}\t{e}\n")
        ok = False

    elapsed_min = (time.time() - t0) / 60
    state["stage_timings"][name] = elapsed_min
    if ok:
        state["stages_completed"].append(name)
    save_state(state)
    update_status(state, name, "завершён" if ok else "ОШИБКА (см. failed.log), пайплайн продолжает следующий этап")
    git_commit(f"[pilot] этап {name}: {'ok' if ok else 'failed'} ({elapsed_min:.1f} мин)")
    log(f"[{name}] {'готово' if ok else 'провалился'} за {elapsed_min:.1f} мин")
    return ok


def main():
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        protocol = yaml.safe_load(f)

    state = load_state(protocol["runtime"]["max_wallclock_hours"])
    log(f"=== Пилотный пайплайн PIK3CA/4JPS запущен. Дедлайн: {state['deadline']} ===")

    def stage_test0():
        sys.path.insert(0, os.path.join(BASE_DIR, "src", "04_docking"))
        import test0_redock
        result = test0_redock.main()
        if not result["passed"]:
            log("Тест 0 НЕ ПРОЙДЕН -- дальнейшие тесты помечены как диагностические, "
                "не подтверждающие (см. протокол). Пайплайн продолжается.")

    def stage_test_a_curation():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "curate_test_a", os.path.join(BASE_DIR, "src", "01_data_curation", "curate_test_a.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    def stage_test_a_docking():
        sys.path.insert(0, os.path.join(BASE_DIR, "src", "04_docking"))
        from dock_dataframe import dock_dataframe
        import pandas as pd

        actives = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test_a_actives_excl_alpelisib.csv"))
        decoys = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test_a_decoys.csv"))
        actives["label"] = 1
        decoys["label"] = 0
        combined = pd.concat([actives[["smiles", "label"]], decoys[["smiles", "label"]]], ignore_index=True)
        docked = dock_dataframe(combined, exhaustiveness=protocol["test_a_calibration"]["exhaustiveness"],
                                 failed_log_path=FAILED_LOG_PATH)
        docked.to_csv(os.path.join(BASE_DIR, "data", "processed", "test_a_docked.csv"), index=False)

        inactives_exp = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test_a_inactives_experimental.csv"))
        inactives_exp["label"] = 0
        actives_all = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test_a_actives_all.csv"))
        actives_all["label"] = 1
        combined_exp = pd.concat([actives_all[["smiles", "label"]], inactives_exp[["smiles", "label"]]], ignore_index=True)
        docked_exp = dock_dataframe(combined_exp, exhaustiveness=protocol["test_a_calibration"]["exhaustiveness"],
                                     failed_log_path=FAILED_LOG_PATH)
        docked_exp.to_csv(os.path.join(BASE_DIR, "data", "processed", "test_a_docked_experimental_secondary.csv"), index=False)

    def stage_test_a_metrics():
        import pandas as pd
        import bedroc_calibration as bc

        df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test_a_docked.csv"))
        cfg = protocol["test_a_calibration"]
        result = bc.run_test_a(df, "docking_score_kcal_mol", "label", cfg["bedroc_alpha"],
                                cfg["bootstrap_iterations"], cfg["permutation_iterations"])
        result["passed_threshold"] = result["bedroc_observed"] >= cfg["bedroc_threshold"]
        out_dir = os.path.join(BASE_DIR, "results", "testA_calibration")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        log(f"Тест A: BEDROC={result['bedroc_observed']:.3f} (порог {cfg['bedroc_threshold']}, "
            f"случайный уровень {result['bedroc_random_baseline_empirical_mean']:.3f}), "
            f"p={result['permutation_p_value']:.4f}")

        # секундарный тест на экспериментальных inactive
        df_exp = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test_a_docked_experimental_secondary.csv"))
        result_exp = bc.run_test_a(df_exp, "docking_score_kcal_mol", "label", cfg["bedroc_alpha"],
                                    cfg["bootstrap_iterations"], cfg["permutation_iterations"])
        with open(os.path.join(out_dir, "result_secondary_experimental.json"), "w", encoding="utf-8") as f:
            json.dump(result_exp, f, indent=2, ensure_ascii=False)

    def stage_test_n():
        import pandas as pd
        import bedroc_calibration as bc

        df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test_a_docked.csv"))
        cfg = protocol["test_n_shuffle"]
        result = bc.run_test_n_shuffle(df, "docking_score_kcal_mol", "label",
                                        protocol["test_a_calibration"]["bedroc_alpha"],
                                        cfg["n_permutations"])
        out_dir = os.path.join(BASE_DIR, "results", "testN_shuffle")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def stage_test_a_prime():
        import pandas as pd
        import bedroc_calibration as bc
        from rdkit import Chem

        df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "test_a_docked.csv"))
        df["heavy_atom_count"] = df["smiles"].apply(
            lambda s: (Chem.MolFromSmiles(s).GetNumHeavyAtoms() if Chem.MolFromSmiles(s) else None)
        )
        cfg = protocol["test_a_prime_size_bias"]
        result = bc.run_test_a_prime_size_bias(df, "docking_score_kcal_mol", "heavy_atom_count", cfg["r2_threshold"])
        out_dir = os.path.join(BASE_DIR, "results", "testA_prime_size_bias")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "result_chembl.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def stage_vina_variance():
        sys.path.insert(0, os.path.join(BASE_DIR, "src", "05_metrics"))
        import vina_variance
        vina_variance.main()

    def stage_test_b():
        sys.path.insert(0, os.path.join(BASE_DIR, "src", "06_generation"))
        import test_b_generation
        test_b_generation.main()

        # Тест A' на генерации (не только на ChEMBL)
        import pandas as pd
        import bedroc_calibration as bc
        gen_docked_path = os.path.join(BASE_DIR, "results", "testB_generation", "generated_docked.csv")
        if os.path.exists(gen_docked_path):
            df_gen = pd.read_csv(gen_docked_path)
            result = bc.run_test_a_prime_size_bias(
                df_gen, "docking_score_kcal_mol", "heavy_atom_count",
                protocol["test_a_prime_size_bias"]["r2_threshold"],
            )
            with open(os.path.join(BASE_DIR, "results", "testA_prime_size_bias", "result_generation.json"),
                      "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    stages = [
        ("test0_redock", os.path.join(BASE_DIR, "results", "test0_redock", "result.json"), stage_test0),
        ("test_a_curation", os.path.join(BASE_DIR, "data", "processed", "test_a_decoys.csv"), stage_test_a_curation),
        ("test_a_docking", os.path.join(BASE_DIR, "data", "processed", "test_a_docked.csv"), stage_test_a_docking),
        ("test_a_metrics", os.path.join(BASE_DIR, "results", "testA_calibration", "result.json"), stage_test_a_metrics),
        ("test_n_shuffle", os.path.join(BASE_DIR, "results", "testN_shuffle", "result.json"), stage_test_n),
        ("test_a_prime", os.path.join(BASE_DIR, "results", "testA_prime_size_bias", "result_chembl.json"), stage_test_a_prime),
        ("vina_variance", os.path.join(BASE_DIR, "results", "metrics", "vina_own_variance.json"), stage_vina_variance),
        ("test_b_generation", os.path.join(BASE_DIR, "results", "testB_generation", "result.json"), stage_test_b),
    ]

    for name, marker, fn in stages:
        deadline = datetime.fromisoformat(state["deadline"])
        hours_left = (deadline - datetime.now()).total_seconds() / 3600
        if hours_left <= protocol["runtime"]["deadline_checkpoints"]["stop_new_ligands_before_deadline_min"] / 60:
            log(f"До дедлайна {hours_left:.1f} ч. -- новые этапы больше не запускаются "
                f"(порог {protocol['runtime']['deadline_checkpoints']['stop_new_ligands_before_deadline_min']} мин).")
            break
        run_stage(state, name, marker, fn)

    log("=== Пайплайн завершён (все этапы пройдены или остановлен по дедлайну/CONTROL.json) ===")
    git_commit("[pilot] финальный коммит пайплайна")


if __name__ == "__main__":
    main()
