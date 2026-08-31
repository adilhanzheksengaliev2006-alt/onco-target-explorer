# REINVENT4 bounded smoke test — 4JPS/PIK3CA project context

Config: `4jps_test_run.toml` (copy of the config actually run).
Prior: `priors/reinvent.prior` (base REINVENT4 SMILES prior, Zenodo 20701824).
Device: cuda:0 (NVIDIA RTX A4000). Peak GPU memory: 2086 MiB / 16376 MiB available.
Batch size: 128 (default). Max steps: 150. Terminated early at step 51
(max_score=0.6 exceeded — see below).
Wall-clock time: 29.8s total.

## Scoring used in this test

- `SAScore` (REINVENT4's built-in Ertl & Schuffenhauer component — the same
  metric as `module5_admet_filters.compute_sa_score`), reverse_sigmoid
  transform, low=2.0/high=6.0 (matches `SA_REASONABLE_MAX` from module5).
- `custom_alerts`: the same unwanted-SMARTS list used in REINVENT4's own
  example config (peroxides, hydrazines, thioesters, strained macrocycles,
  etc.) — a PAINS-adjacent filter, not the project's actual PAINS/Brenk
  catalog.
- Aggregation: geometric_mean of the two components.

**NOT included in this test:** live Vina docking against the 4JPS pocket as
part of the per-step RL reward. 128 molecules x up to 150 steps implies up
to ~19,200 docking calls; at realistic per-molecule Vina runtimes that is
far outside a 1-hour bounded test and is architecturally a separate task
(a custom REINVENT4 scoring component, or wiring DockStream — a component
REINVENT4 already ships, `comp_dockstream.py`, but not configured here).
This is a flagged follow-up decision, not a silent omission.

## Result: the RL loop demonstrably works with our own scoring wired in

| step | mean Score | fraction zeroed by custom_alerts |
|-----:|-----------:|----------------------------------:|
| 1    | 0.626      | 27.3%                              |
| 5    | 0.736      | 17.2%                              |
| 21   | 0.775      | 18.0%                              |
| 31   | 0.840      | 12.5%                              |
| 41   | 0.855      | 12.5%                              |
| 50   | 0.889      | 9.4%                               |
| 51   | 0.932      | 5.5%                               |

Mean raw SA score barely moved (2.14 -> 2.18) because the base prior
already generates molecules in the "easy to synthesize" range (low=2.0
threshold) — there was little headroom there. Nearly all of the score
improvement came from the agent learning to avoid the unwanted-SMARTS
alerts (27% -> 5.5% of generated molecules flagged), which is exactly the
kind of steerable behavior curriculum learning is supposed to produce.
6656 molecules scored across 52 steps; validity stayed at 98-100% per step
throughout.

## Files in this folder

- `4jps_test_run.toml` — the config that was actually run.
- `4jps_test_run.log` — full REINVENT4 run log.
- `4jps_test_run_1.csv` — per-molecule scores for every step (SMILES,
  Score, SA score raw/transformed, custom_alerts raw/transformed).
- `4jps_test.chkpt` — the trained agent checkpoint at termination (24MB);
  reusable as `agent_file` to continue this run (manual curriculum
  learning) instead of restarting from the base prior.

## Fixes required to get here (all applied directly in the cloned repo)

1. `reinvent/utils/hw_report.py`: unconditional `import resource` (POSIX-only
   stdlib module) crashed on Windows at import time even though its actual
   *usage* was already correctly guarded behind `if SYSTEM != "Windows"`.
   Made the import conditional too.
2. `scipy` is imported by `reinvent/runmodes/utils/plot.py` but was missing
   from `pyproject.toml`'s dependency list — installed separately.
3. The example `configs/staged_learning.toml` sets
   `parameters.unique_sequences`, which this REINVENT4 version (4.8.24)
   rejects as an unrecognized field (pydantic `extra_forbidden`) — removed
   it from our config.
