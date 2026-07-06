"""FlowGate core engine , no GUI dependencies.

Design in one paragraph
-----------------------
A :class:`Session` holds many :class:`FlowSample` objects (your mice) and ONE
shared :class:`GatingTree`. A gate is defined once (as a polygon in display
coordinates) and the *same* gate object is applied to every sample. That is the
whole trick behind requirement #8: edit a gate's vertices, call
``session.apply()`` again, and every mouse is re-gated and re-compared. Gates
form a hierarchy via ``parent`` — a child gate only sees the events its parent
kept.

Coordinate spaces
-----------------
Users draw polygons on a display plot whose axes may be transformed (linear for
scatter, arcsinh/biexponential for fluorescence). We store gate vertices in that
display space and, to apply a gate, we transform a sample's raw values into the
same display space and test point-in-polygon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.path import Path

import flowkit as fk


# --------------------------------------------------------------------------- #
# Axis transforms (forward only is needed for both plotting and gating)
# --------------------------------------------------------------------------- #
class AxisTransform:
    """Monotonic forward transform from raw channel value -> display value."""

    name = "identity"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float)


class LinearAxis(AxisTransform):
    name = "linear"


class AsinhAxis(AxisTransform):
    """arcsinh transform , invertible cousin of a log/biex axis.

    ``cofactor`` sets where the axis switches from ~linear (near 0, including
    negatives from compensation) to ~log. 150 is a sane default for fluorescence
    on an 18-bit scale; 262 or so also common.
    """

    name = "asinh"

    def __init__(self, cofactor: float = 150.0):
        self.cofactor = float(cofactor)

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        return np.arcsinh(x / self.cofactor)

    def inverse(self, y):
        y = np.asarray(y, dtype=float)
        return np.sinh(y) * self.cofactor


class LogAxis(AxisTransform):
    """log10 transform. Non-positive values are clamped to ``floor`` (flow data
    is non-negative, so this only affects exact zeros)."""

    name = "log"

    def __init__(self, floor: float = 1.0):
        self.floor = float(floor)

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        return np.log10(np.clip(x, self.floor, None))

    def inverse(self, y):
        return np.power(10.0, np.asarray(y, dtype=float))


# name -> constructor, for the GUI's scale dropdown
AXIS_TRANSFORMS = {"linear": LinearAxis, "log": LogAxis, "asinh": AsinhAxis}


def default_transform_for(channel: str, marker: str = "") -> AxisTransform:
    """Scatter channels are linear; anything with a fluorophore/marker is asinh."""
    scatter_prefixes = ("FSC", "SSC", "TIME", "Time")
    if channel.upper().startswith(scatter_prefixes):
        return LinearAxis()
    return AsinhAxis()


def transform_to_dict(t: AxisTransform) -> dict:
    d = {"name": t.name}
    if isinstance(t, AsinhAxis):
        d["cofactor"] = t.cofactor
    return d


def transform_from_dict(d: dict) -> AxisTransform:
    if d["name"] == "asinh":
        return AsinhAxis(d.get("cofactor", 150.0))
    if d["name"] == "log":
        return LogAxis(d.get("floor", 1.0))
    return LinearAxis()


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #
class FlowSample:
    """One FCS file loaded into a DataFrame, with detector<->marker awareness."""

    def __init__(self, path: str, sample_id: Optional[str] = None):
        self.path = path
        self._fk = fk.Sample(path, sample_id=sample_id)
        self.sample_id = self._fk.id
        # DataFrame with raw values, columns = detector (PnN) labels
        df = self._fk.as_dataframe(source="raw")
        df.columns = [c[0] for c in df.columns]  # drop the (pnn, pns) multiindex
        self.data: pd.DataFrame = df.reset_index(drop=True)

        # detector -> marker map, e.g. {"BV421-A": "CD45"}
        self.markers: dict[str, str] = {}
        for pnn, pns in zip(self._fk.pnn_labels, self._fk.pns_labels):
            if pns:
                self.markers[pnn] = pns

    @property
    def channels(self) -> list[str]:
        return list(self.data.columns)

    def label(self, channel: str) -> str:
        """ label: 'BV421-A (CD45)' when a marker is known."""
        m = self.markers.get(channel)
        return f"{channel} ({m})" if m else channel

    def n_events(self) -> int:
        return len(self.data)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
@dataclass
class PolygonGate:
    """One polygon gate: "keep the events inside this shape on these two axes".

    A gate is defined once and applied to every sample. The polygon corners
    (``vertices``) are stored in *display* coordinates — i.e. after the axis
    transforms are applied — which is exactly what you see and draw on screen.
    To test a sample, we transform its raw values the same way and check which
    points fall inside the polygon.

    Parameters
    ----------
    name : str
        Unique population name (also the tree key), e.g. "Live_Cells".
    x_channel, y_channel : str
        Detector channels shown on the X and Y axes, e.g. "FSC-A", "BV421-A".
    parent : str or None
        Name of the population this gate refines. ``None``/"root" = top level.
    vertices : list[(x, y)]
        Polygon corners in display coordinates. Fewer than 3 = "not drawn yet",
        in which case the gate keeps everything (a pass-through).
    x_transform, y_transform : AxisTransform
        How each axis is scaled (linear for scatter, arcsinh for fluorescence).
    """

    name: str
    x_channel: str
    y_channel: str
    parent: Optional[str] = None
    vertices: list[tuple[float, float]] = field(default_factory=list)
    x_transform: AxisTransform = field(default_factory=LinearAxis)
    y_transform: AxisTransform = field(default_factory=LinearAxis)
    # Optional per-plot view limits in RAW data units (None = auto-range).
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    def is_defined(self) -> bool:
        """True once at least 3 vertices exist (a real polygon has been drawn)."""
        return len(self.vertices) >= 3

    def _remap(self, old_xt, old_yt, new_xt, new_yt):
        """Move vertices from one display space to another via raw coordinates,
        so the polygon stays around the same events when a transform changes."""
        remapped = []
        for vx, vy in self.vertices:
            rx = old_xt.inverse(np.array([vx]))[0]
            ry = old_yt.inverse(np.array([vy]))[0]
            remapped.append((float(new_xt.forward(np.array([rx]))[0]),
                             float(new_yt.forward(np.array([ry]))[0])))
        self.vertices = remapped

    def set_transforms(self, x_transform, y_transform):
        """Change axis scaling (linear/log/asinh) without losing the drawn gate."""
        if self.vertices:
            self._remap(self.x_transform, self.y_transform, x_transform, y_transform)
        self.x_transform = x_transform
        self.y_transform = y_transform

    def set_channels(self, x_channel, y_channel, x_transform, y_transform):
        """Point this gate at different channels. The old polygon no longer makes
        sense on new axes, so it is reset."""
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.vertices = []
        self.x_min = self.x_max = self.y_min = self.y_max = None

    def display_xy(self, sample: FlowSample) -> np.ndarray:
        """This sample's events as an (N, 2) array in the gate's display space."""
        x = self.x_transform.forward(sample.data[self.x_channel].to_numpy())
        y = self.y_transform.forward(sample.data[self.y_channel].to_numpy())
        return np.column_stack([x, y])

    def contains(self, sample: FlowSample) -> np.ndarray:
        """Which of the sample's events fall inside this polygon.

        Returns a boolean array over *all* the sample's events. Parent membership
        is NOT considered here — :meth:`GatingTree.apply` handles the hierarchy by
        intersecting this with the parent's mask. An undrawn gate keeps everything.
        """
        if not self.is_defined():
            return np.ones(sample.n_events(), dtype=bool)
        pts = self.display_xy(sample)
        path = Path(np.asarray(self.vertices, dtype=float))
        return path.contains_points(pts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x_channel": self.x_channel,
            "y_channel": self.y_channel,
            "parent": self.parent,
            "vertices": [list(map(float, v)) for v in self.vertices],
            "x_transform": transform_to_dict(self.x_transform),
            "y_transform": transform_to_dict(self.y_transform),
            "view": [self.x_min, self.x_max, self.y_min, self.y_max],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolygonGate":
        view = d.get("view", [None, None, None, None])
        return cls(
            name=d["name"],
            x_channel=d["x_channel"],
            y_channel=d["y_channel"],
            parent=d.get("parent"),
            vertices=[tuple(v) for v in d.get("vertices", [])],
            x_transform=transform_from_dict(d["x_transform"]),
            y_transform=transform_from_dict(d["y_transform"]),
            x_min=view[0], x_max=view[1], y_min=view[2], y_max=view[3],
        )


# --------------------------------------------------------------------------- #
# Gating tree (shared across all samples)
# --------------------------------------------------------------------------- #
class GatingTree:
    """A hierarchy of polygon gates, shared by every sample in a Session.

    Think of it as the "gating strategy" in FlowJo: a set of named gates, each
    pointing at its ``parent`` gate, forming a tree. The SAME tree is applied to
    every sample, which is what lets you compare mice and adjust one threshold
    for all of them at once.

    A gate whose ``parent`` is ``None`` or ``"root"`` sits at the top level and
    sees all events. Any other gate only sees the events its parent kept.

    Attributes
    ----------
    gates : dict[str, PolygonGate]
        Every gate, keyed by its unique name.
    order : list[str]
        Gate names in creation order. Parents always come before their
        children, so applying gates in this order is always valid.
    """

    ROOT = "root"  # the implicit top-level population (all events)

    def __init__(self):
        self.gates: dict[str, PolygonGate] = {}
        self.order: list[str] = []

    # -- building the tree -------------------------------------------------- #
    def add(self, gate: PolygonGate) -> PolygonGate:
        """Add an already-built :class:`PolygonGate` to the tree.

        Raises if the name is taken or the named parent doesn't exist yet.
        Returns the gate so you can keep a handle to it.
        """
        if gate.name in self.gates:
            raise ValueError(f"a gate named '{gate.name}' already exists")
        if gate.parent and gate.parent != self.ROOT and gate.parent not in self.gates:
            raise ValueError(
                f"parent '{gate.parent}' of gate '{gate.name}' does not exist")
        self.gates[gate.name] = gate
        self.order.append(gate.name)
        return gate

    def add_gate(self, name, parent, x_channel, y_channel,
                 x_transform=None, y_transform=None, vertices=None) -> PolygonGate:
        """Convenience builder: create a gate and add it in one call.

        Parameters
        ----------
        name : str            unique name / population label
        parent : str          name of the parent gate, or "root"
        x_channel, y_channel : str   detector channels for the two axes
        x_transform, y_transform : AxisTransform, optional
            Defaults to linear for scatter channels, arcsinh otherwise.
        vertices : list[(x, y)], optional
            Polygon corners in display coordinates; empty = "not drawn yet".
        """
        return self.add(PolygonGate(
            name=name, x_channel=x_channel, y_channel=y_channel, parent=parent,
            vertices=list(vertices or []),
            x_transform=x_transform or default_transform_for(x_channel),
            y_transform=y_transform or default_transform_for(y_channel),
        ))

    # -- navigating the tree ------------------------------------------------ #
    def children_of(self, name: Optional[str]) -> list[PolygonGate]:
        """Gates whose direct parent is ``name`` (use "root"/None for top level).

        Top-level gates may store their parent as either ``None`` or ``"root"``;
        both are treated the same here.
        """
        if name in (None, self.ROOT):
            return [self.gates[n] for n in self.order
                    if self.gates[n].parent in (None, self.ROOT)]
        return [self.gates[n] for n in self.order if self.gates[n].parent == name]

    def descendants(self, name: str) -> list[str]:
        """All gate names below ``name`` (children, grandchildren, ...)."""
        out, stack = [], [c.name for c in self.children_of(name)]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(c.name for c in self.children_of(n))
        return out

    def ancestors(self, name: str) -> list[str]:
        """Names of all gates above ``name``, nearest parent first."""
        out, cur = [], self.gates.get(name)
        while cur and cur.parent and cur.parent != self.ROOT:
            out.append(cur.parent)
            cur = self.gates.get(cur.parent)
        return out

    def is_at_or_below(self, name: str, ancestor: str) -> bool:
        """True if ``name`` is ``ancestor`` itself or somewhere beneath it."""
        return name == ancestor or ancestor in self.ancestors(name)

    def remove(self, name: str) -> list[str]:
        """Delete a gate and everything beneath it. Returns the removed names."""
        to_remove = [name] + self.descendants(name)
        for n in to_remove:
            self.gates.pop(n, None)
            if n in self.order:
                self.order.remove(n)
        return to_remove

    # -- applying the tree -------------------------------------------------- #
    def apply(self, sample: FlowSample) -> dict[str, np.ndarray]:
        """Gate one sample and return a mask per population.

        Returns
        -------
        dict[str, np.ndarray]
            ``{gate_name: boolean array over the sample's events}``. Each mask is
            already intersected with its parent, so ``masks["Live_Cells"]`` is the
            true Live population. The key ``"root"`` maps to an all-True mask.
        """
        masks: dict[str, np.ndarray] = {self.ROOT: np.ones(sample.n_events(), bool)}
        for name in self.order:                       # parents precede children
            gate = self.gates[name]
            parent_mask = masks[gate.parent or self.ROOT]
            masks[name] = parent_mask & gate.contains(sample)
        return masks

    def describe(self) -> str:
        """A readable, indented outline of the whole hierarchy (for printing)."""
        lines: list[str] = []

        def walk(parent, depth):
            for g in self.children_of(parent):
                drawn = f"{len(g.vertices)} pts" if g.is_defined() else "not drawn"
                lines.append(f"{'  ' * depth}- {g.name} "
                             f"[{g.x_channel} x {g.y_channel}] ({drawn})")
                walk(g.name, depth + 1)

        walk(self.ROOT, 0)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
@dataclass
class GateStat:
    sample_id: str
    gate: str
    n_in: int
    n_parent: int

    @property
    def pct_of_parent(self) -> float:
        return 100.0 * self.n_in / self.n_parent if self.n_parent else 0.0


class Session:
    """The whole analysis: all loaded samples plus one shared gating tree.

    Typical use::

        s = Session()
        s.add_sample("Mouse1.fcs")
        s.add_sample("Mouse2.fcs")
        s.tree.add_gate("Cells", "root", "FSC-A", "SSC-A", vertices=[...])
        s.apply()                 # gate every sample
        s.stats_frame()           # counts and % per (sample, gate)

    ``apply()`` fills a cache of masks; call it again after editing any gate.
    """

    def __init__(self):
        self.samples: dict[str, FlowSample] = {}
        self.tree = GatingTree()
        # cache: sample_id -> {gate_name: boolean mask}
        self._masks: dict[str, dict[str, np.ndarray]] = {}

    # ---- samples ----
    def add_sample(self, path: str, sample_id: Optional[str] = None) -> FlowSample:
        """Load an FCS file and register it. Returns the new FlowSample."""
        s = FlowSample(path, sample_id=sample_id)
        self.samples[s.sample_id] = s
        return s

    def common_channels(self) -> list[str]:
        if not self.samples:
            return []
        sets = [set(s.channels) for s in self.samples.values()]
        common = set.intersection(*sets)
        # preserve first sample's order
        first = next(iter(self.samples.values()))
        return [c for c in first.channels if c in common]

    def marker_map(self) -> dict[str, str]:
        """Union of detector->marker maps across samples (first wins on conflict)."""
        out: dict[str, str] = {}
        for s in self.samples.values():
            for k, v in s.markers.items():
                out.setdefault(k, v)
        return out

    # ---- gating ----
    def apply(self) -> None:
        """(Re)gate every sample with the current tree. Call after any edit."""
        self._masks = {sid: self.tree.apply(s) for sid, s in self.samples.items()}

    def remove_gate(self, name: str) -> list[str]:
        """Delete a gate (and its descendants) and re-gate. Returns removed names."""
        removed = self.tree.remove(name)
        self.apply()
        return removed

    def masks_for(self, sample_id: str) -> dict[str, np.ndarray]:
        if sample_id not in self._masks:
            self._masks[sample_id] = self.tree.apply(self.samples[sample_id])
        return self._masks[sample_id]

    def population(self, sample_id: str, gate_name: str) -> pd.DataFrame:
        """The DataFrame of events belonging to ``gate_name`` in one sample."""
        s = self.samples[sample_id]
        if gate_name in (None, "root"):
            return s.data
        mask = self.masks_for(sample_id)[gate_name]
        return s.data.loc[mask].reset_index(drop=True)

    def stats(self) -> list[GateStat]:
        """One row per (sample, gate): counts and % of parent."""
        rows: list[GateStat] = []
        for sid, s in self.samples.items():
            masks = self.masks_for(sid)
            for name in self.tree.order:
                gate = self.tree.gates[name]
                parent_key = gate.parent if gate.parent else "root"
                n_parent = int(masks[parent_key].sum())
                n_in = int(masks[name].sum())
                rows.append(GateStat(sid, name, n_in, n_parent))
        return rows

    # ---- persistence: thresholds/gates travel across sessions ----
    def save_gates(self, path: str) -> None:
        import json
        payload = {"gates": [self.tree.gates[n].to_dict() for n in self.tree.order]}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load_gates(self, path: str) -> None:
        import json
        with open(path) as f:
            payload = json.load(f)
        self.tree = GatingTree()
        for gd in payload["gates"]:
            self.tree.add(PolygonGate.from_dict(gd))
        self._masks = {}

    def stats_frame(self) -> pd.DataFrame:
        rows = self.stats()
        return pd.DataFrame(
            [
                {
                    "sample": r.sample_id,
                    "gate": r.gate,
                    "events": r.n_in,
                    "parent_events": r.n_parent,
                    "pct_of_parent": round(r.pct_of_parent, 2),
                }
                for r in rows
            ]
        )


# --------------------------------------------------------------------------- #
# Pseudocolor density 
# --------------------------------------------------------------------------- #
def box_blur(a: np.ndarray, passes: int = 2) -> np.ndarray:
    """Tiny dependency-free 5-point blur to smooth a 2D density grid."""
    for _ in range(passes):
        a = (a
             + np.pad(a[1:, :], ((0, 1), (0, 0)))
             + np.pad(a[:-1, :], ((1, 0), (0, 0)))
             + np.pad(a[:, 1:], ((0, 0), (0, 1)))
             + np.pad(a[:, :-1], ((0, 0), (1, 0)))) / 5.0
    return a


def pseudocolor_density(x, y, xlim, ylim, bins: int = 256) -> np.ndarray:
    """Per-event, log-scaled, smoothed local density for coloring a dot plot."""
    x = np.asarray(x); y = np.asarray(y)
    h, xe, ye = np.histogram2d(x, y, bins=bins, range=[list(xlim), list(ylim)])
    h = box_blur(h, passes=2)
    xi = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, bins - 1)
    return np.log1p(h[xi, yi])
