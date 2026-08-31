# Воспроизводимость

## Что запушено, что нет

**В репозитории**: код, конфиги (`config/protocol.yaml`, `config/paths.yaml`),
обезличенные результаты (`ligand_id` = `sha256(канонический SMILES)[:16]`
вместо SMILES, см. `src/common/anonymize.py`), логи, метрики, графики,
`structures/4JPS_receptor.pdbqt`.

**НЕ в репозитории** (восстанавливается локально командами ниже):
промежуточные позы докинга (`*.pdbqt`, кроме готового рецептора), полные
SMILES-таблицы (`data/processed/test_a_*.csv` без суффикса `_public`,
`results/testB_generation/*_smiles.csv`/`*_docked.csv`), клонированные
DiffSBDD/REINVENT4 репозитории, их окружения, чекпоинты/приоры.

## Как восстановить

```bash
# 1. Окружение
conda env create -f environment.yml -n pik3ca-pilot
conda activate pik3ca-pilot
pip install -r requirements.txt

# 2. DiffSBDD (для Теста B)
git clone https://github.com/arneschneuing/DiffSBDD module_generative/DiffSBDD
# checkpoint (17.9MB): Zenodo record 20701824, файл crossdocked_fullatom_cond.ckpt
curl -L "https://zenodo.org/api/records/20701824/files/crossdocked_fullatom_cond.ckpt/content" \
     -o module_generative/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt

# 3. Полный пайплайн (все этапы идемпотентны -- пропускают уже готовые)
python src/common/orchestrate.py
```

Каждая строка результатов (JSONL/CSV) несёт: `ligand_id`, скор, число
тяжёлых атомов, LE, метку, seed, exhaustiveness Vina, и т.д. -- этого
достаточно, чтобы по коду+конфигу+`ligand_id` полностью пересчитать любую
конкретную молекулу локально, ИМЕЯ доступ к маппингу `ligand_id -> SMILES`
(который хранится только локально на машине, где считался пилот -- см.
docs/limitations.md, пункт про отсутствующий приватный репозиторий).

## Целостность протокола

`config/protocol.yaml` захеширован (SHA256) в `state.json` при первом
запуске `src/common/orchestrate.py`; при последующих запусках хеш
сверяется, и скрипт отказывается стартовать, если файл протокола изменился
после первого запуска (см. `run.log` в этом случае).
