# REINVENT4 docking-in-the-loop test — 4JPS/PIK3CA pocket

Real AutoDock Vina docking against the 4JPS ATP pocket wired in as a live
component of the REINVENT4 RL reward (not a post-hoc filter) — via a new
custom scoring component, `comp_pik3ca_dock.py`, that reuses the project's
existing docking code (`dock_smiles_isolated` from
`dock_existing_candidates.py`) rather than reimplementing docking.

Config: `4jps_docking_test.toml`. Device: cuda:0. Batch size: 8 (reduced
from the project's eventual target of 128 — see "Scale" below).
Exhaustiveness: 4 (reduced from Vina's usual default of 8, for speed).
Max steps: 10. Wall-clock: 7m25s (all 10 steps ran; did not terminate
early). Docking itself runs on CPU via `tools/vina.exe`, independent of
the RL training's GPU usage.

## Result: docking scores flowed through correctly

Raw Vina scores across all 10 steps: min -10.54, max -4.74, mean -7.49
kcal/mol — squarely in the range this project has already established as
meaningful for this pocket (the alpelisib control docks at -10.18/-10.22
in independent checks this session). No steps came back all-zero or
all-NA; a handful of individual molecules failed to dock (embedding
failure, or the one invalid SMILES at step 10) and were scored 0.0/omitted
rather than crashing the batch, same convention as REINVENT4's own
DockStream component.

| step | mean composite Score | mean docking raw (kcal/mol) |
|-----:|----------------------:|------------------------------:|
| 1    | 0.097                 | -6.60                          |
| 2    | 0.270                 | -7.21                          |
| 3    | 0.157                 | -7.64                          |
| 4    | 0.187                 | -7.53                          |
| 5    | 0.271                 | -7.15                          |
| 6    | 0.303                 | -7.96                          |
| 7    | 0.221                 | -7.87                          |
| 8    | 0.427                 | -8.11                          |
| 9    | 0.165                 | -7.26                          |
| 10   | 0.318                 | -7.53                          |

10 steps x batch 8 is too short a run to expect a clean monotonic
docking-score trend (that needs hundreds of steps, per the project's
original REINVENT4 plan) — this test's purpose was narrower: prove the
full loop (RDKit SMILES -> 3D embed -> meeko PDBQT -> Vina -> score ->
transform -> RL gradient) works end-to-end without crashing, timing out,
or silently returning fake data, using the project's real receptor and
pocket coordinates. It does.

## Scale: why this was 8x10, not 128x100+

Measured on this machine: ~5.4s per molecule at exhaustiveness=4
(sequential, one Vina subprocess at a time — the isolated-worker pattern
in `dock_existing_candidates.py` is deliberately one-at-a-time, for the
hang-timeout protection it provides). At that rate:

- This test (8 x 10 = 80 dockings): ~7.5 min (matches measured).
- The project's originally-discussed scale (128 x 100-200 steps =
  12,800-25,600 dockings): **19-38 hours** sequential, even at this
  reduced exhaustiveness=4. At full exhaustiveness=8 (used for the
  alpelisib control and DiffSBDD-adjacent work) it roughly doubles again.

Getting to production scale needs one of:
1. Parallelize docking across CPU cores (the current worker pattern is
   single-threaded; Vina itself doesn't need the GPU, so this and the
   RL training could run concurrently on this machine's 13th-gen i7).
2. Reduce exhaustiveness further and/or dock only a sampled subset of
   each batch, using SA score alone to reward the rest.
3. Accept a much smaller batch_size for the RL loop itself.

This is a design decision for the next session, not something resolved
here — flagged rather than silently worked around.

## Files in this folder

- `4jps_docking_test.toml` — the config that was run.
- `4jps_docking_test.log` — full REINVENT4 run log (per-molecule scores
  every step, including which SMILES failed to dock).
- `4jps_docking_test_1.csv` — all 80 scored molecules with SA score,
  docking score, and alert-filter columns.
- `4jps_docking_test.chkpt` — trained agent checkpoint (23.5MB).
- `comp_pik3ca_dock.py` — the new REINVENT4 scoring component (copy of
  `module_generative/REINVENT4/reinvent_plugins/components/comp_pik3ca_dock.py`).
- `dock_batch_for_reinvent.py` — the new project-root wrapper script the
  component shells out to (copy of the project-root file).
