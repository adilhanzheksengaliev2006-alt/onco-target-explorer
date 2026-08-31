"""
gene_target_utils.py
=====================
Общая логика "ген -> мишень ChEMBL -> 3D-структура с лигандом", которую
переиспользуют check_druggability.py, dock_existing_candidates.py и
module_generative/iterative_finetune_loop.py.

Всё определяется от GENE_NAME, без хардкода PDB ID или ChEMBL ID —
КРОМЕ явно подтверждённых человеком записей в confirmed_structures.json
(см. ниже).

Цепочка:
  find_chembl_target(gene_name)         -> ChEMBL target (SINGLE PROTEIN, human)
  get_known_target_ligands(target_id)   -> известные ChEMBL-ингибиторы мишени (любая фаза)
  find_pdb_structure_matching_known_ligand(...)
      -> PDB-структура, где связанный лиганд ХИМИЧЕСКИ СОВПАДАЕТ (по InChI,
         через RCSB chemical search) с одним из известных ингибиторов —
         не просто "любой лиганд с лучшим разрешением".
  download_pdb(pdb_id)                   -> скачивает .pdb/.cif в structures/
  find_ligand_center(pdb_path)           -> координаты центра бокса докинга + resname лиганда

ВАЖНЫЙ УРОК (см. историю: PIK3CA/9CMK): "лучшее разрешение среди
структур с любым лигандом" — недостаточный критерий. 9CMK оказалась
структурой RAS-binding domain с молекулярным клеем для другой задачи
(диабет), а не каталитического ATP-кармана, где работают реальные
ингибиторы вроде алпелисиба. Поэтому автопоиск теперь ТРЕБУЕТ, чтобы
связанный лиганд совпадал с известным ингибитором гена из ChEMBL, а
если совпадения нет — останавливается с явным сообщением вместо того,
чтобы молча брать первую попавшуюся структуру.
"""

import json
import os
import time
import requests

STRUCTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "structures")
os.makedirs(STRUCTURES_DIR, exist_ok=True)

# Сеть на этой машине изредка отдаёт транзиентные сбои DNS-резолвинга
# (repo.anaconda.com, search.rcsb.org и т.д.) — несколько попыток с
# паузой спасают от того, чтобы весь прогон падал из-за одного блипа.
_RETRY_ATTEMPTS = 3
_RETRY_DELAY_SEC = 5


def _request_with_retry(method: str, url: str, **kwargs):
    last_exc = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return requests.request(method, url, **kwargs)
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < _RETRY_ATTEMPTS:
                print(
                    f"[gene_target_utils] Сетевая ошибка (попытка {attempt}/{_RETRY_ATTEMPTS}) "
                    f"для {url}: {e}. Повтор через {_RETRY_DELAY_SEC}с..."
                )
                time.sleep(_RETRY_DELAY_SEC)
    raise last_exc

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{}"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{}.pdb"
RCSB_DOWNLOAD_URL_CIF = "https://files.rcsb.org/download/{}.cif"

# Гетероатомные остатки, которые НЕ считаются "настоящим" лигандом
# (вода, ионы, обычные криопротекторы/буферные добавки кристаллизации).
IGNORE_HET_RESIDUES = {
    "HOH", "WAT", "NA", "K", "CL", "MG", "CA", "ZN", "MN", "FE", "CO", "NI",
    "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "DMS", "ACT", "TRS", "IMD",
    "MPD", "BME", "FMT", "ACY", "CIT", "NO3", "UNX", "1PE", "P6G", "PGE",
    "MES", "HEPES", "BOG", "LDA", "MRD",
}


class GeneTargetError(Exception):
    """Понятная ошибка на любом шаге поиска мишени/структуры."""


_chembl_new_client_cache = None


def get_chembl_new_client():
    """chembl_webresource_client.new_client делает сетевой запрос схемы
    (/spore) ПРИ ИМПОРТЕ — EBI изредка отдаёт 500 на этот эндпоинт.
    Ретраим импорт с паузой и кешируем успешный результат в модульной
    переменной, чтобы не повторять хрупкий импорт на каждый вызов."""
    global _chembl_new_client_cache
    if _chembl_new_client_cache is not None:
        return _chembl_new_client_cache
    last_exc = None
    for attempt in range(1, 4):
        try:
            from chembl_webresource_client.new_client import new_client
            _chembl_new_client_cache = new_client
            return new_client
        except Exception as e:
            last_exc = e
            if attempt < 3:
                time.sleep(8)
    raise last_exc


def find_chembl_target(gene_name: str) -> dict:
    """Ищет ген в ChEMBL (как Модуль 2) и возвращает лучшую цель
    SINGLE PROTEIN / Homo sapiens с её UniProt accession.
    """
    try:
        new_client = get_chembl_new_client()
    except Exception as e:
        raise GeneTargetError(
            f"Не удалось импортировать chembl_webresource_client: {str(e)[:300]}"
        )

    try:
        target = new_client.target
        candidates = list(
            target.filter(
                target_synonym__icontains=gene_name,
                organism="Homo sapiens",
            ).only(["target_chembl_id", "pref_name", "target_type", "organism"])
        )
    except Exception as e:
        raise GeneTargetError(f"Ошибка запроса к ChEMBL API для гена {gene_name}: {e}")

    if not candidates:
        raise GeneTargetError(f"В ChEMBL не найдено ни одной мишени для гена {gene_name}")

    single_protein = [c for c in candidates if c.get("target_type") == "SINGLE PROTEIN"]
    chosen = single_protein[0] if single_protein else candidates[0]

    try:
        full = target.get(chosen["target_chembl_id"])
    except Exception as e:
        raise GeneTargetError(
            f"Ошибка получения деталей мишени {chosen['target_chembl_id']}: {e}"
        )

    uniprot_accession = None
    for comp in full.get("target_components", []):
        if comp.get("accession"):
            uniprot_accession = comp["accession"]
            break

    if uniprot_accession is None:
        raise GeneTargetError(
            f"У мишени {chosen['target_chembl_id']} ({chosen.get('pref_name')}) "
            f"нет UniProt accession — не могу искать 3D-структуру."
        )

    return {
        "gene_name": gene_name,
        "target_chembl_id": chosen["target_chembl_id"],
        "pref_name": chosen.get("pref_name"),
        "target_type": chosen.get("target_type"),
        "uniprot_accession": uniprot_accession,
    }


def get_known_target_ligands(target_chembl_id: str) -> list:
    """Известные ингибиторы/лиганды мишени из ChEMBL (mechanism endpoint —
    курируемые связи препарат-мишень, ЛЮБАЯ фаза разработки, не только
    одобренные — потому что большинство со-кристаллизованных в PDB
    лигандов являются исследовательскими tool compounds, а не готовыми
    препаратами). Возвращает список {chembl_id, pref_name, smiles, inchi}.
    """
    try:
        new_client = get_chembl_new_client()
    except Exception as e:
        raise GeneTargetError(f"Не удалось импортировать chembl_webresource_client: {str(e)[:300]}")

    try:
        mechanism = new_client.mechanism
        mech_records = list(
            mechanism.filter(target_chembl_id=target_chembl_id).only(["molecule_chembl_id"])
        )
    except Exception as e:
        raise GeneTargetError(f"Ошибка запроса ChEMBL mechanism endpoint: {str(e)[:300]}")

    mol_ids = list(set(r["molecule_chembl_id"] for r in mech_records))
    if not mol_ids:
        return []

    try:
        molecule = new_client.molecule
        recs = list(
            molecule.filter(molecule_chembl_id__in=mol_ids).only(
                ["molecule_chembl_id", "pref_name", "molecule_structures"]
            )
        )
    except Exception as e:
        raise GeneTargetError(f"Ошибка запроса ChEMBL molecule endpoint: {str(e)[:300]}")

    from rdkit import Chem

    ligands = []
    for r in recs:
        smiles = (r.get("molecule_structures") or {}).get("canonical_smiles")
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            inchi = Chem.MolToInchi(mol)
        except Exception:
            continue
        if not inchi:
            continue
        ligands.append(
            {
                "chembl_id": r["molecule_chembl_id"],
                "pref_name": r.get("pref_name") or r["molecule_chembl_id"],
                "smiles": smiles,
                "inchi": inchi,
            }
        )
    return ligands


def _rcsb_entry_resolution_and_title(pdb_id: str):
    entry_resp = _request_with_retry("get", RCSB_DATA_ENTRY_URL.format(pdb_id), timeout=30)
    entry_resp.raise_for_status()
    entry = entry_resp.json()
    resolution_list = entry.get("rcsb_entry_info", {}).get("resolution_combined")
    title = entry.get("struct", {}).get("title")
    resolution = resolution_list[0] if resolution_list else None
    return resolution, title


def find_pdb_structure_matching_known_ligand(uniprot_accession: str, known_ligands: list, top_n: int = 5):
    """Ищет PDB-структуры человека с данным UniProt accession, где
    связанный лиганд ХИМИЧЕСКИ СОВПАДАЕТ (RCSB chemical search по
    InChI, match_type=graph-relaxed-stereo — совпадение графа связей
    с учётом стереохимии) с одним из known_ligands.

    Возвращает (best_match, all_matches) где best_match — словарь с
    лучшим разрешением, all_matches — список всех найденных совпадений
    (для отчёта). Если совпадений нет — (None, []).
    """
    all_matches = []
    for lig in known_ligands:
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                            "operator": "exact_match",
                            "value": uniprot_accession,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                            "operator": "exact_match",
                            "value": "UniProt",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "chemical",
                        "parameters": {
                            "value": lig["inchi"],
                            "type": "descriptor",
                            "descriptor_type": "InChI",
                            "match_type": "graph-relaxed-stereo",
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": top_n}},
        }
        try:
            resp = _request_with_retry("post", RCSB_SEARCH_URL, json=query, timeout=30)
        except Exception as e:
            raise GeneTargetError(f"Ошибка сети при химическом поиске RCSB (лиганд {lig['pref_name']}): {e}")

        if resp.status_code == 204:
            continue
        if resp.status_code != 200:
            # некоторые InChI (соли, необычная стереохимия) могут быть
            # отклонены RCSB как невалидный дескриптор — не фатально,
            # просто пропускаем этот известный лиганд
            continue

        for hit in resp.json().get("result_set", []):
            pdb_id = hit["identifier"]
            try:
                resolution, title = _rcsb_entry_resolution_and_title(pdb_id)
            except Exception as e:
                raise GeneTargetError(f"Ошибка получения деталей PDB-записи {pdb_id}: {e}")
            if resolution is not None:
                all_matches.append(
                    {
                        "pdb_id": pdb_id,
                        "resolution": resolution,
                        "title": title,
                        "matched_ligand": lig["pref_name"],
                        "matched_ligand_chembl_id": lig["chembl_id"],
                    }
                )

    if not all_matches:
        return None, []

    all_matches.sort(key=lambda m: m["resolution"])
    return all_matches[0], all_matches


CONFIRMED_STRUCTURES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "confirmed_structures.json"
)


def load_confirmed_structure(gene_name: str):
    """Читает вручную подтверждённые структуры из confirmed_structures.json.
    Формат: {"GENE": {"pdb_id": "XXXX", "confirmed_by": "...", "note": "..."}}.
    Используется, когда автопоиск не может сам подтвердить релевантность
    кармана (см. find_pdb_structure_matching_known_ligand) — человек
    один раз проверяет структуру вручную и фиксирует её здесь."""
    if not os.path.exists(CONFIRMED_STRUCTURES_PATH):
        return None
    try:
        with open(CONFIRMED_STRUCTURES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data.get(gene_name.upper())


def find_pdb_structure_with_ligand(uniprot_accession: str, top_n: int = 5) -> dict | None:
    """ДИАГНОСТИЧЕСКИЙ режим (НЕ для автоматического выбора структуры):
    через RCSB Search API ищет PDB-структуры человека с этим UniProt
    accession, у которых есть хотя бы один связанный не-полимерный
    компонент (потенциальный лиганд, БЕЗ проверки, что это известный
    ингибитор), сортирует по разрешению и возвращает лучшую из первых
    top_n. Используется ТОЛЬКО чтобы показать человеку, что вообще
    нашлось, когда find_pdb_structure_matching_known_ligand не дал
    совпадений — сама эта функция не должна использоваться для выбора
    структуры под докинг напрямую.

    Возвращает None, если ничего подходящего не нашлось.
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": uniprot_accession,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater",
                        "value": 0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
            "paginate": {"start": 0, "rows": top_n},
        },
    }

    try:
        resp = _request_with_retry("post", RCSB_SEARCH_URL, json=query, timeout=30)
    except Exception as e:
        raise GeneTargetError(f"Ошибка сети при запросе к RCSB Search API: {e}")

    if resp.status_code == 204:
        return None
    if resp.status_code != 200:
        raise GeneTargetError(
            f"RCSB Search API вернул код {resp.status_code}: {resp.text[:300]}"
        )

    result_set = resp.json().get("result_set", [])
    if not result_set:
        return None

    for hit in result_set:
        pdb_id = hit["identifier"]
        try:
            entry_resp = _request_with_retry("get", RCSB_DATA_ENTRY_URL.format(pdb_id), timeout=30)
            entry_resp.raise_for_status()
            entry = entry_resp.json()
        except Exception as e:
            raise GeneTargetError(f"Ошибка получения деталей PDB-записи {pdb_id}: {e}")

        resolution_list = entry.get("rcsb_entry_info", {}).get("resolution_combined")
        title = entry.get("struct", {}).get("title")
        resolution = resolution_list[0] if resolution_list else None
        if resolution is not None:
            return {"pdb_id": pdb_id, "resolution": resolution, "title": title}

    # ни у одного из top_n не оказалось числового разрешения (напр. NMR-структуры)
    return None


def download_pdb(pdb_id: str) -> str:
    """Скачивает структуру в structures/{pdb_id}.pdb (переиспользует
    кэш, если уже скачана) и возвращает путь. Некоторые новые/крупные
    записи RCSB не имеют legacy PDB-формата — в этом случае скачивается
    mmCIF (.cif), который gemmi читает так же нативно."""
    dest_pdb = os.path.join(STRUCTURES_DIR, f"{pdb_id}.pdb")
    dest_cif = os.path.join(STRUCTURES_DIR, f"{pdb_id}.cif")
    if os.path.exists(dest_pdb) and os.path.getsize(dest_pdb) > 0:
        return dest_pdb
    if os.path.exists(dest_cif) and os.path.getsize(dest_cif) > 0:
        return dest_cif

    try:
        resp = _request_with_retry("get", RCSB_DOWNLOAD_URL.format(pdb_id), timeout=60)
    except Exception as e:
        raise GeneTargetError(f"Не удалось скачать структуру {pdb_id}: {e}")

    if resp.status_code == 200:
        with open(dest_pdb, "wb") as f:
            f.write(resp.content)
        return dest_pdb

    # legacy .pdb недоступен (частый случай для новых/крупных записей) -> mmCIF
    try:
        resp_cif = _request_with_retry("get", RCSB_DOWNLOAD_URL_CIF.format(pdb_id), timeout=60)
        resp_cif.raise_for_status()
    except Exception as e:
        raise GeneTargetError(
            f"Не удалось скачать структуру {pdb_id} ни в .pdb (код {resp.status_code}), "
            f"ни в .cif формате: {e}"
        )
    with open(dest_cif, "wb") as f:
        f.write(resp_cif.content)
    return dest_cif


def find_ligand_center(pdb_path: str):
    """Парсит PDB-файл через gemmi, находит связанный низкомолекулярный
    лиганд (HETATM, не вода/ион/криопротектор) и возвращает
    (center_x, center_y, center_z, resname, box_size) для докинг-бокса.

    Если в файле нет ни одного подходящего лиганда — возвращает None.
    """
    try:
        import gemmi
    except Exception as e:
        raise GeneTargetError(f"Не удалось импортировать gemmi: {e}")

    try:
        structure = gemmi.read_structure(pdb_path)
    except Exception as e:
        raise GeneTargetError(f"Ошибка чтения структуры {pdb_path}: {e}")

    best = None  # (n_atoms, center, resname, box_size)
    for model in structure:
        for chain in model:
            for residue in chain:
                if not residue.het_flag == "H":
                    continue
                resname = residue.name.strip()
                if resname in IGNORE_HET_RESIDUES:
                    continue
                atoms = [a for a in residue if a.element.name != "H"]
                if len(atoms) < 5:
                    # слишком маленький фрагмент, вряд ли настоящий лиганд
                    continue
                xs = [a.pos.x for a in atoms]
                ys = [a.pos.y for a in atoms]
                zs = [a.pos.z for a in atoms]
                center = (
                    sum(xs) / len(xs),
                    sum(ys) / len(ys),
                    sum(zs) / len(zs),
                )
                box_size = (
                    max(max(xs) - min(xs) + 10, 20),
                    max(max(ys) - min(ys) + 10, 20),
                    max(max(zs) - min(zs) + 10, 20),
                )
                if best is None or len(atoms) > best[0]:
                    best = (len(atoms), center, resname, box_size)
        break  # только первая модель

    if best is None:
        return None

    n_atoms, center, resname, box_size = best
    return {
        "center": center,
        "box_size": box_size,
        "resname": resname,
        "n_atoms": n_atoms,
    }


def resolve_gene_to_docking_target(gene_name: str) -> dict | None:
    """Полная цепочка: ген -> ChEMBL target -> UniProt -> PDB-структура,
    где связанный лиганд подтверждённо совпадает с известным
    ингибитором гена (или вручную подтверждена в confirmed_structures.json)
    -> скачанный файл -> координаты бокса докинга.

    Если функционально релевантную структуру подтвердить не удаётся —
    НЕ выбирает что-то наугад. Печатает, что было найдено, и
    останавливается (raise GeneTargetError) с просьбой проверить
    вручную и добавить в confirmed_structures.json.
    """
    print(f"[gene_target_utils] Поиск мишени ChEMBL для гена {gene_name}...")
    chembl_info = find_chembl_target(gene_name)
    print(
        f"[gene_target_utils] Найдена мишень: {chembl_info['target_chembl_id']} "
        f"({chembl_info['pref_name']}), UniProt {chembl_info['uniprot_accession']}"
    )

    confirmed = load_confirmed_structure(gene_name)
    matched_ligand_name = None
    matched_ligand_chembl_id = None

    if confirmed is not None:
        pdb_id = confirmed["pdb_id"]
        print(
            f"[gene_target_utils] Используется ВРУЧНУЮ ПОДТВЕРЖДЁННАЯ структура "
            f"{pdb_id} из confirmed_structures.json "
            f"(подтвердил: {confirmed.get('confirmed_by', '?')}; {confirmed.get('note', '')})"
        )
        try:
            resolution, title = _rcsb_entry_resolution_and_title(pdb_id)
        except Exception as e:
            raise GeneTargetError(f"Ошибка получения деталей PDB-записи {pdb_id}: {e}")
        matched_ligand_name = confirmed.get("ligand_name")
        matched_ligand_chembl_id = confirmed.get("ligand_chembl_id")
    else:
        print(
            f"[gene_target_utils] Поиск известных ингибиторов {gene_name} в ChEMBL "
            f"(для проверки релевантности кармана)..."
        )
        known_ligands = get_known_target_ligands(chembl_info["target_chembl_id"])
        print(f"[gene_target_utils] Известных ингибиторов в ChEMBL: {len(known_ligands)}"
              + (f" ({', '.join(l['pref_name'] for l in known_ligands[:8])}{'...' if len(known_ligands) > 8 else ''})" if known_ligands else ""))

        best_match, all_matches = (None, [])
        if known_ligands:
            print(f"[gene_target_utils] Поиск PDB-структур, где связанный лиганд СОВПАДАЕТ "
                  f"с одним из известных ингибиторов (RCSB chemical search по InChI)...")
            best_match, all_matches = find_pdb_structure_matching_known_ligand(
                chembl_info["uniprot_accession"], known_ligands
            )

        if best_match is None:
            # Ничего подтверждённого не нашлось — показать человеку, что вообще
            # есть (диагностика), и ОСТАНОВИТЬСЯ, а не выбирать наугад.
            print(
                f"[gene_target_utils] НЕ НАЙДЕНО PDB-структуры, где лиганд подтверждённо "
                f"совпадает с известным ингибитором {gene_name} из ChEMBL."
            )
            diag = find_pdb_structure_with_ligand(chembl_info["uniprot_accession"], top_n=5)
            if diag is not None:
                print(
                    f"[gene_target_utils] Для справки (НЕ выбрано автоматически): "
                    f"лучшая по разрешению структура с любым лигандом — "
                    f"{diag['pdb_id']} ({diag['resolution']} A): {diag['title']}"
                )
            raise GeneTargetError(
                f"Не могу автоматически подтвердить функционально релевантный карман для "
                f"{gene_name}. Нужна проверка человеком: посмотри структуру"
                + (f" {diag['pdb_id']}" if diag is not None else "")
                + f" (и другие кандидаты по UniProt {chembl_info['uniprot_accession']} на rcsb.org), "
                f"убедись, что связанный лиганд — известный ингибитор в функционально "
                f"релевантном сайте, и добавь запись в confirmed_structures.json вида "
                f'{{"{gene_name.upper()}": {{"pdb_id": "XXXX", "confirmed_by": "...", "note": "..."}}}}.'
            )

        pdb_id = best_match["pdb_id"]
        resolution = best_match["resolution"]
        title = best_match["title"]
        matched_ligand_name = best_match["matched_ligand"]
        matched_ligand_chembl_id = best_match["matched_ligand_chembl_id"]
        other_matches_note = (
            f" (всего найдено {len(all_matches)} совпадений с известными ингибиторами)"
            if len(all_matches) > 1 else ""
        )
        print(
            f"[gene_target_utils] Выбрана структура {pdb_id} (разрешение {resolution} A): {title}"
            f" — связанный лиганд совпадает с известным ингибитором {matched_ligand_name}"
            f"{other_matches_note}"
        )

    pdb_path = download_pdb(pdb_id)
    print(f"[gene_target_utils] Скачано: {pdb_path}")

    ligand_info = find_ligand_center(pdb_path)
    if ligand_info is None:
        print(
            f"[gene_target_utils] В структуре {pdb_id} не удалось "
            f"выделить связанный лиганд (только вода/ионы/криопротекторы). "
            f"Докинг для этого гена будет пропущен."
        )
        return None

    print(
        f"[gene_target_utils] Лиганд в структуре: {ligand_info['resname']} "
        f"({ligand_info['n_atoms']} тяжёлых атомов), центр бокса "
        f"{tuple(round(c, 2) for c in ligand_info['center'])}"
    )

    return {
        "gene_name": gene_name,
        "chembl_target_id": chembl_info["target_chembl_id"],
        "target_pref_name": chembl_info["pref_name"],
        "uniprot_accession": chembl_info["uniprot_accession"],
        "pdb_id": pdb_id,
        "resolution": resolution,
        "pdb_path": pdb_path,
        "ligand_resname": ligand_info["resname"],
        "matched_known_ligand": matched_ligand_name,
        "matched_known_ligand_chembl_id": matched_ligand_chembl_id,
        "box_center": ligand_info["center"],
        "box_size": ligand_info["box_size"],
    }


if __name__ == "__main__":
    import sys

    gene = sys.argv[1] if len(sys.argv) > 1 else "PIK3CA"
    result = resolve_gene_to_docking_target(gene)
    print("\n=== РЕЗУЛЬТАТ ===")
    print(result)
