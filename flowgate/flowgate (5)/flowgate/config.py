"""The gating hierarchy.

Each GateSpec is one step. `x`/`y` may be either a detector name ("FSC-A") or a
marker name ("CD45", "Zombie") all are lasers; markers are resolved to the file's real detector
channel at load time via the session's marker map. Add subpopulation gates
 just by appending more Gates with the right `parent`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateSpec:
    name: str          # output population name
    parent: str        # parent population ("root" for top level)
    x: str             # laser for X axis
    y: str             # laser for Y axis
    note: str = ""


# The hierarchy .
#   Beads and Real_Cells both sit at root on FSC/SSC (beads identified, cells kept).
GATE_HIERARCHY: list[GateSpec] = [
    GateSpec("Beads",               "root",                 "FSC-A", "SSC-A",
             note="Identify beads on forward/side scatter"),
    GateSpec("Real_Cells",          "root",                 "FSC-A", "SSC-A",
             note="First gate: cells vs debris on FSC/SSC"),
    GateSpec("Real_Fwd_Cells",      "Real_Cells",           "FSC-A", "FSC-H",
             note="Second gate: forward singlets (FSC-A vs FSC-H)"),
    GateSpec("Real_Fwd_Side_Cells", "Real_Fwd_Cells",       "SSC-A", "SSC-H",
             note="Third gate: side singlets (SSC-A vs SSC-H)"),
    GateSpec("Live_Cells",          "Real_Fwd_Side_Cells",  "FSC-A", "Zombie",
             note="Fourth gate: live cells (FSC-A vs Zombie viability)"),
    GateSpec("Category_Cells",      "Live_Cells",           "FSC-A", "CD45",
             note="Fifth gate: CD45+ population (FSC-A vs CD45)"),
    # --- Step 7: add subpopulation gates below, e.g.:
    # GateSpec("CD4_T",  "Category_Cells", "CD4",  "CD8", note="subpopulation"), example
]


def resolve_channel(token: str, marker_map: dict[str, str], channels: list[str]) -> str:
    """Turn a spec token into a real detector channel for a given file.

    - If token is already a detector present in the file, use it.
    - Else in cases where naming isn't global treat it as a marker and find the detector whose marker matches
      (case-insensitive, substring-tolerant so "Zombie" matches "Zombie NIR").
    """
    if token in channels:
        return token
    tl = token.lower()
    # marker match
    for det, mk in marker_map.items():
        if mk and (tl == mk.lower() or tl in mk.lower() or mk.lower() in tl):
            if det in channels:
                return det
    # if no marker is found: case-insensitive detector match
    for c in channels:
        if c.lower() == tl:
            return c
    raise KeyError(
        f"Could not resolve '{token}' to a channel. "
        f"Available detectors: {channels}. Markers: {marker_map}"
    )
