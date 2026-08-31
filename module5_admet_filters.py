"""
module5_admet_filters.py — Модуль 5 (продолжение): SA score + ADMETlab 3.0
=============================================================================
Расширяет существующий Модуль 5 (PAINS/Brenk в
module_generative/iterative_finetune_loop.py: build_toxicity_catalog /
passes_toxicity_filter) двумя дешёвыми фильтрами, которые встают в цепочку
ПОСЛЕ докинга и ПОСЛЕ PAINS/Brenk, но ПЕРЕД селективностью (ChEMBL):

    докинг -> PAINS/Brenk -> SA score -> ADMETlab (полный ADME) -> селективность

(PAINS/Brenk оставлен первым, т.к. он уже есть, самый дешёвый — чистый
RDKit без сети — и по духу пайплайна дешёвые локальные фильтры должны идти
перед сетевыми; сам порядок "докинг -> SA score -> ADMETlab -> селективность",
заданный в задаче, соблюдён один в один.)

Оба фильтра НЕ отсеивают кандидатов жёстко — они добавляют колонки к
результату, а не режут список (в отличие от докинга/токсичности/
селективности, которые именно фильтруют). Итоговое ранжирование
(rank_candidates) учитывает их вместе.

--------------------------------------------------------------------------
1. SA score (Ertl & Schuffenhauer, 2009)
--------------------------------------------------------------------------
Через rdkit.Chem.RDConfig.RDContribDir/SA_Score/sascorer.py — этот скрипт
идёт В КОМПЛЕКТЕ с установленной версией RDKit (2026.03.5, conda env
molgen; проверено: RDConfig.RDContribDir/SA_Score существует и sascorer
считает разумные значения из коробки). Отдельно скачивать sascorer.py /
fpscores.pkl.gz с github.com/rdkit/rdkit не потребовалось — сохранено на
случай, если в другом окружении их не окажется (см. _sascorer()).

Шкала: ~1 (легко синтезировать) .. ~10 (очень сложно). Порог отсечения
НЕ введён по запросу задачи — колонка SA_score просто сортируемая, ниже
предложен ориентир (SA_REASONABLE_MAX) для справки в отчётах.

--------------------------------------------------------------------------
2. ADMETlab 3.0 (полный ADME + токсикология)
--------------------------------------------------------------------------
Официальной программной документации эндпоинтов на сайте нет — /apis/ и
/documentation/ рендерятся клиентским JS и не отдают спецификацию
статическому HTTP-клиенту. Ниже — то, что подтверждено реверс-инжинирингом
живого сервиса (см. историю разработки этого модуля):

  - "Публичный REST API" (ninja-роутер, схема на /api/openapi.json)
    СЕЙЧАС СЛОМАН на стороне ADMETlab: /api/admet вообще снят с продакшена
    (404), а актуальный /api/single/admet стабильно падает с 500
    (KeyError: "['BSEP'] not in index") на ЛЮБОМ SMILES, включая
    тривиальный "CCO" — это баг в их backend-коде (колонка BSEP не
    досчитывается для одиночных предсказаний), не в формате нашего
    запроса. Независимо подтверждено сторонним проектом на GitHub
    (senseibelbi/ADMETlab_MCP), документирующим ту же ошибку как известную
    проблему сервиса.

  - Реально рабочий путь — тот же, которым пользуется веб-форма batch-
    скрининга: GET /server/screening (получить csrf-токен) -> POST
    /server/screeningCal (поле "smiles-list", CRLF-разделитель — именно
    так это шлёт браузерная <textarea>; ОБЫЧНЫЙ "\n" сервер не разбивает
    на строки — проверено на практике) -> редирект на
    /server/result/<taskId> -> CSV по прямой ссылке
    /server/result/<taskId>/download/csv (кнопка "Download as CSV" на
    странице результата). На этом пути все конечные точки считаются
    нормально (проверено на алпелисибе и контрольных молекулах,
    2026-08-22). ПРИМЕЧАНИЕ: изначально этот путь был
    /static/results/csv/<taskId>.csv — сайт сменил схему без
    предупреждения (стабильно отдавал 404), если снова начнёт падать
    404-кой, первым делом проверить вручную кнопку "Download as CSV" на
    /server/result/<taskId> и обновить ADMET_CSV_URL.

  - Численной оценки НЕОПРЕДЕЛЁННОСТИ (uncertainty) в живом выводе сервиса
    обнаружить не удалось — ни в этом CSV (123 колонки, ни одна не похожа
    на std/uncertainty), ни на странице /server/detail/<id>/<i> (там же
    цветные индикаторы success/warning/danger — это НЕ confidence, а
    "попадает ли предсказанное ЗНАЧЕНИЕ в оптимальный диапазон", обычная
    ADMET-подсветка, подтверждено по тексту подсказок в самой разметке).
    Поле "uncertain" существует в одной из Pydantic-схем API (ADMETSchema
    у /api/washmol), но ни на что не влияет: /api/washmol просто
    стандартизирует SMILES и игнорирует этот флаг. Это подтверждённый
    пробел ЖИВОГО сервиса относительно того, что заявлено в статье
    (ADMETlab 3.0, Nucleic Acids Research 2024) — не недосмотр в нашем
    запросе. Поэтому численной uncertainty-колонки здесь нет: вместо
    фиктивных чисел результат явно помечен через ADMET_uncertainty_note.
"""

import os
import time

import pandas as pd
import requests

ADMET_BASE_URL = "https://admetlab3.scbdd.com"
ADMET_SCREENING_PAGE = ADMET_BASE_URL + "/server/screening"
ADMET_SCREENING_CAL = ADMET_BASE_URL + "/server/screeningCal"
ADMET_CSV_URL = ADMET_BASE_URL + "/server/result/{}/download/csv"

ADMET_BATCH_SIZE = 25          # запас относительно заявленного лимита формы (1000) — не грузим чужой бесплатный сервис
ADMET_PAUSE_BETWEEN_BATCHES_SEC = 3.0
ADMET_RETRY_ATTEMPTS = 3
ADMET_RETRY_DELAY_SEC = 10.0

ADMET_UNCERTAINTY_NOTE = (
    "ADMETlab 3.0 не отдаёт численную uncertainty ни в одном из проверенных "
    "живых выходов сервиса (batch CSV, /server/detail) — см. docstring "
    "module5_admet_filters.py. Численного показателя доверия к предсказанию нет."
)

# Красные флаги токсикологии: ADMETlab выдаёт вероятность "быть токсичным/
# ингибитором" (0..1); >0.5 — стандартная граница классификации, тот же
# порог, что подписан в подсказках самого сайта ("Category 1: ... The
# output value is the probability of being ... within the range of 0 to 1").
ADMET_RED_FLAG_ENDPOINTS = {
    "hERG": 0.5,
    "Ames": 0.5,
    "DILI": 0.5,
    "BSEP": 0.5,
    "H-HT": 0.5,
}

SA_REASONABLE_MAX = 6.0  # ориентир для отчётов; НЕ используется как жёсткий фильтр


class Admet5Error(Exception):
    """Понятная ошибка любого шага SA score / ADMETlab (в духе GeneTargetError/DockingError/PipelineError)."""


# =============================================================================
# SA score (RDKit Contrib/SA_Score)
# =============================================================================
_sascorer_module = None


def _sascorer():
    """Лениво импортирует rdkit.Contrib.SA_Score.sascorer, кэширует модуль.
    Если Contrib/SA_Score не идёт в комплекте установленного RDKit (в этом
    окружении — conda env molgen, RDKit 2026.03.5 — идёт, проверено),
    сообщение об ошибке прямо укажет, что и откуда скачать вручную
    (github.com/rdkit/rdkit, папка Contrib/SA_Score: sascorer.py +
    fpscores.pkl.gz), как и предполагалось в задаче."""
    global _sascorer_module
    if _sascorer_module is not None:
        return _sascorer_module

    try:
        from rdkit.Chem import RDConfig
    except Exception as e:
        raise Admet5Error(f"Не удалось импортировать rdkit.Chem.RDConfig: {e}")

    sa_score_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if not os.path.isdir(sa_score_dir):
        raise Admet5Error(
            f"В установленном RDKit нет Contrib/SA_Score ({sa_score_dir}). "
            f"Скачай sascorer.py и fpscores.pkl.gz из "
            f"github.com/rdkit/rdkit (папка Contrib/SA_Score) в эту директорию."
        )

    import sys

    if sa_score_dir not in sys.path:
        sys.path.append(sa_score_dir)

    try:
        import sascorer  # noqa: F401 — сторонний модуль без типов/интерфейса пакета
    except Exception as e:
        raise Admet5Error(f"Не удалось импортировать sascorer из {sa_score_dir}: {e}")

    _sascorer_module = sascorer
    return sascorer


def compute_sa_score(smiles: str) -> float | None:
    """SA score (Ertl & Schuffenhauer) для одной молекулы. None, если SMILES
    не парсится RDKit-ом (не считается фатальной ошибкой — как и везде в
    этом пайплайне для отдельных молекул)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    sascorer = _sascorer()
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def compute_sa_scores(smiles_list: list[str]) -> dict:
    """SA score для списка SMILES. Ключ — исходная (не канонизированная) строка."""
    return {smi: compute_sa_score(smi) for smi in smiles_list}


# =============================================================================
# ADMETlab 3.0 — через тот же путь, что использует веб-форма batch-скрининга
# =============================================================================
def _canonical_smiles_and_inchikey(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = None
    return Chem.MolToSmiles(mol), inchikey


def _with_retry(fn, attempts=ADMET_RETRY_ATTEMPTS, delay_sec=ADMET_RETRY_DELAY_SEC, what=""):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                print(f"  [ADMETlab] {what}: попытка {attempt}/{attempts} не удалась ({e}), "
                      f"повтор через {delay_sec:.0f}с...")
                time.sleep(delay_sec)
    raise Admet5Error(f"ADMETlab: {what} не удалось после {attempts} попыток: {last_exc}")


def _get_screening_csrf(session: requests.Session) -> str:
    import re

    resp = session.get(ADMET_SCREENING_PAGE, timeout=30)
    resp.raise_for_status()
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', resp.text)
    if not m:
        raise Admet5Error("Не нашёл csrfmiddlewaretoken на странице /server/screening — вёрстка сайта могла измениться.")
    return m.group(1)


def _submit_screening_batch(session: requests.Session, token: str, smiles_batch: list) -> str:
    """POST'ит один батч SMILES на /server/screeningCal (как это делает
    браузерная форма method=2, textarea "smiles-list"). Возвращает taskId
    из URL редиректа (/server/result/<taskId>)."""
    payload = {
        "csrfmiddlewaretoken": token,
        "smiles-list": "\r\n".join(smiles_batch),  # CRLF — как реальная <textarea>, обычный \n сервер не разбивает
        "method": "2",
    }
    headers = {"Referer": ADMET_SCREENING_PAGE, "Origin": ADMET_BASE_URL}
    resp = session.post(ADMET_SCREENING_CAL, data=payload, headers=headers, timeout=120)
    resp.raise_for_status()

    task_id = resp.url.rstrip("/").rsplit("/", 1)[-1]
    if not task_id or "screening" in resp.url:
        raise Admet5Error(f"screeningCal не вернул ожидаемый редирект на /server/result/<taskId> (получили {resp.url}).")
    return task_id


def _fetch_admet_csv(session: requests.Session, task_id: str) -> pd.DataFrame:
    resp = session.get(ADMET_CSV_URL.format(task_id), timeout=60)
    resp.raise_for_status()
    if not resp.text.strip():
        raise Admet5Error(f"CSV результатов ADMETlab для taskId={task_id} пустой.")
    import io

    return pd.read_csv(io.StringIO(resp.text))


def fetch_admet_batch_raw(smiles_batch: list) -> pd.DataFrame:
    """Один батч (<= ADMET_BATCH_SIZE) через реальный batch-путь ADMETlab.
    Возвращает сырой DataFrame сервиса (колонки как в их CSV)."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (onco-target-explorer pipeline)"})

    token = _with_retry(lambda: _get_screening_csrf(session), what="получение csrf-токена /server/screening")
    task_id = _with_retry(lambda: _submit_screening_batch(session, token, smiles_batch), what="POST /server/screeningCal")
    return _with_retry(lambda: _fetch_admet_csv(session, task_id), what=f"скачивание CSV (taskId={task_id})")


def fetch_admet_batch(smiles_list: list) -> pd.DataFrame:
    """Полный ADME + токсикология для списка SMILES через ADMETlab 3.0,
    с батчингом и паузами (не бьём чужой бесплатный сервис одним потоком)
    и сопоставлением результатов обратно к исходным SMILES по InChIKey
    (ADMETlab "washes"/стандартизирует структуру на входе, поэтому сверять
    по точной строке SMILES ненадёжно — тот же принцип химического
    сопоставления, что и в gene_target_utils.py для PDB-лигандов).

    Возвращает DataFrame с индексом = исходные SMILES (как переданы) и
    колонками ADMET_<endpoint> для каждой конечной точки сервиса, плюс
    ADMET_matched (bool) и ADMET_uncertainty_note. Для SMILES, которые
    ADMETlab не смог сопоставить обратно (невалидные/не распознанные) —
    строка с ADMET_matched=False и NaN по остальным колонкам, БЕЗ падения
    всего батча (как и для докинга/токсичности отдельных молекул в этом
    пайплайне)."""
    if not smiles_list:
        return pd.DataFrame()

    input_by_inchikey = {}
    for smi in smiles_list:
        _, inchikey = _canonical_smiles_and_inchikey(smi)
        if inchikey:
            input_by_inchikey.setdefault(inchikey, []).append(smi)

    all_raw_frames = []
    batches = [smiles_list[i:i + ADMET_BATCH_SIZE] for i in range(0, len(smiles_list), ADMET_BATCH_SIZE)]
    for i, batch in enumerate(batches):
        print(f"  [ADMETlab] батч {i + 1}/{len(batches)} ({len(batch)} молекул)...")
        raw_df = fetch_admet_batch_raw(batch)
        all_raw_frames.append(raw_df)
        if i < len(batches) - 1:
            time.sleep(ADMET_PAUSE_BETWEEN_BATCHES_SEC)

    raw_df = pd.concat(all_raw_frames, ignore_index=True)

    from rdkit import Chem

    matched_by_inchikey = {}
    for _, row in raw_df.iterrows():
        out_smi = row.get("smiles")
        if not isinstance(out_smi, str):
            continue
        mol = Chem.MolFromSmiles(out_smi)
        if mol is None:
            continue
        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            continue
        matched_by_inchikey.setdefault(inchikey, row)

    endpoint_cols = [c for c in raw_df.columns if c not in ("raw_smiles", "smiles", "molstr")]

    rows = []
    n_matched = 0
    for smi in smiles_list:
        _, inchikey = _canonical_smiles_and_inchikey(smi)
        matched_row = matched_by_inchikey.get(inchikey) if inchikey else None
        out = {"SMILES": smi, "ADMET_matched": matched_row is not None, "ADMET_uncertainty_note": ADMET_UNCERTAINTY_NOTE}
        for c in endpoint_cols:
            out[f"ADMET_{c}"] = matched_row[c] if matched_row is not None else None
        if matched_row is not None:
            n_matched += 1
        else:
            print(f"  [ADMETlab] ПРЕДУПРЕЖДЕНИЕ: не нашёл {smi[:60]} в ответе сервиса (невалидный SMILES с точки зрения ADMETlab, либо он его отбросил) — ADMET-колонки будут пустыми для этой молекулы.")
        rows.append(out)

    print(f"  [ADMETlab] сопоставлено с результатом: {n_matched}/{len(smiles_list)}")
    return pd.DataFrame(rows).set_index("SMILES", drop=False)


def count_admet_red_flags(admet_row: pd.Series) -> int | None:
    """Сколько токсикологических конечных точек с ЯВНЫМ красным флагом
    (вероятность > порога) у молекулы. None, если ADMET для неё не
    посчитан (ADMET_matched=False) — отличать "не флагов нет" от "не
    считали"."""
    if not admet_row.get("ADMET_matched", False):
        return None
    n = 0
    for endpoint, threshold in ADMET_RED_FLAG_ENDPOINTS.items():
        val = admet_row.get(f"ADMET_{endpoint}")
        try:
            if val is not None and float(val) > threshold:
                n += 1
        except (TypeError, ValueError):
            continue
    return n


# =============================================================================
# Композитное ранжирование
# =============================================================================
def rank_candidates(df: pd.DataFrame, docking_col: str = "docking_score_kcal_mol") -> pd.DataFrame:
    """Сортирует кандидатов приоритетом: докинг лучше порога (уже
    отфильтровано раньше в пайплайне) -> меньше SA score (легче
    синтезировать) -> меньше явных ADMET красных флагов у показателей,
    которые вообще удалось посчитать. Не отсеивает — только сортирует;
    молекулы без SA/ADMET (например, сеть подвела) уходят в конец, а не
    выбрасываются."""
    df = df.copy()
    if "ADMET_matched" in df.columns:
        df["_admet_red_flags"] = df.apply(
            lambda r: count_admet_red_flags(r) if r.get("ADMET_matched", False) else None, axis=1
        )
    else:
        df["_admet_red_flags"] = None

    sort_cols, ascending = [], []
    if docking_col in df.columns:
        sort_cols.append(docking_col)
        ascending.append(True)
    sort_cols.append("_admet_red_flags")
    ascending.append(True)
    if "SA_score" in df.columns:
        sort_cols.append("SA_score")
        ascending.append(True)

    df = df.sort_values(by=sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    df = df.drop(columns=["_admet_red_flags"])
    return df
