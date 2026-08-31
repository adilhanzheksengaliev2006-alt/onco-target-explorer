"""AutoDock Vina docking against the PIK3CA/4JPS pocket, as a REINVENT4
scoring component.

Modeled directly on the built-in DockStream component
(comp_dockstream.py): docking needs rdkit/meeko/gemmi + tools/vina.exe,
none of which live in the REINVENT4 .venv (a research RL stack has no
business depending on a separate docking toolchain) — so, like
DockStream, this shells out to a specific external python interpreter
running a standalone script (dock_batch_for_reinvent.py at the project
root), rather than importing docking code in-process.

Reuses dock_smiles_isolated() from dock_existing_candidates.py (same
per-molecule isolated-process worker + OS-level timeout used everywhere
else in this pipeline) via that wrapper script — the receptor
(structures/4JPS_receptor.pdbqt) and pocket box coordinates are NOT
recomputed here; they come from gene_target_utils.find_ligand_center()
and are passed in through the TOML config, config.pik3ca_dock.toml.

Cost note: unlike SAScore (near-instant, in-process RDKit), each
molecule here is a full Vina docking run in its own subprocess. Do not
point a full-size batch (128) x hundreds of RL steps at this without
first reducing exhaustiveness and/or batch size, or accepting a
multi-hour run — see results/task2_reinvent4_docking_test/summary.md
for measured per-molecule timing on this machine.
"""

from __future__ import annotations

__all__ = ["PIK3CADocking"]

import logging
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .run_program import run_command
from .add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles

logger = logging.getLogger(__name__)


@add_tag("__parameters")
@dataclass
class Parameters:
    """All fields are lists because components can have multiple endpoints;
    only the first value of each is used (single-endpoint component)."""

    python_path: List[str]  # system python with rdkit/meeko/gemmi installed
    script_path: List[str]  # dock_batch_for_reinvent.py
    receptor_pdbqt: List[str]
    box_center: List[str]  # "cx,cy,cz"
    box_size: List[str]  # "sx,sy,sz"
    exhaustiveness: List[int]


@add_tag("__component")
class PIK3CADocking:
    """AutoDock Vina docking score against the 4JPS (PIK3CA) ATP pocket.

    Molecules that fail to embed in 3D or fail to dock are scored 0.0
    (same convention as DockStream) rather than dropped, so a bad batch
    doesn't crash the RL step.
    """

    def __init__(self, params: Parameters):
        self.python_path = params.python_path[0]
        self.script_path = params.script_path[0]
        self.receptor_pdbqt = params.receptor_pdbqt[0]
        cx, cy, cz = params.box_center[0].split(",")
        sx, sy, sz = params.box_size[0].split(",")
        self.box_center = (cx, cy, cz)
        self.box_size = (sx, sy, sz)
        self.exhaustiveness = str(params.exhaustiveness[0])
        self.smiles_type = "rdkit_smiles"

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        command = [
            self.python_path,
            self.script_path,
            self.receptor_pdbqt,
            *self.box_center,
            *self.box_size,
            self.exhaustiveness,
            ";".join(smilies),
        ]

        result = run_command(command)
        lines = result.stdout.split()

        scores = []
        for line in lines:
            try:
                scores.append(float(line))
            except ValueError:
                scores.append(0.0)

        return ComponentResults([scores])
