"""FlowGate GUI 

Layout
------
  ┌─────────────┬───────────────────────────────┐
  │ Samples     │  Density plot for (sample,     │
  │ (mice)      │  gate) with an editable        │
  ├─────────────┤  polygon. Draw / edit /        │
  │ Populations │  apply the gate here.          │
  │ (gate tree) │                                │
  └─────────────┴───────────────────────────────┘
  │ Stats table: every sample × every gate       │
  └───────────────────────────────────────────────┘

The polygon you draw is stored on the shared gate and applied to ALL samples,
so re-gating and the "Compare across samples" grid update every mouse at once.
"""
from __future__ import annotations

import os
import sys
import numpy as np

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# Density colormap: jet, but truncated so the densest points are BRIGHT red
FLOW_CMAP = LinearSegmentedColormap.from_list(
    "flowjet", mpl.colormaps["jet"](np.linspace(0.0, 0.9, 256)))

from PyQt5 import QtWidgets, QtCore

from .core import (
    Session, PolygonGate, LinearAxis, AsinhAxis, LogAxis, AXIS_TRANSFORMS,
    default_transform_for, pseudocolor_density,
)
from .config import GATE_HIERARCHY, resolve_channel


class DraggablePolygon:
    """An editable polygon overlay: drag a vertex to reshape, or drag inside the
    polygon to move the whole thing. Reports changes back via callbacks.

    on_change(verts)  — fired continuously while dragging (cheap; redraw overlays)
    on_release(verts) — fired once on mouse-up (do the expensive re-gating here)
    """

    HIT_PX = 10  # how close (in pixels) a click must be to grab a vertex

    def __init__(self, ax, vertices, on_change=None, on_release=None, color="black"):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.verts = [[float(x), float(y)] for x, y in vertices]
        self.on_change = on_change
        self.on_release = on_release
        xs = [v[0] for v in self.verts] + [self.verts[0][0]]
        ys = [v[1] for v in self.verts] + [self.verts[0][1]]
        (self.line,) = ax.plot(xs, ys, "-o", color=color, lw=1.6, ms=6,
                               mfc="white", mec=color, zorder=20)
        self._mode = None  # None | ("vertex", i) | ("poly", (x0,y0), verts0)
        self._cids = [
            self.canvas.mpl_connect("button_press_event", self._press),
            self.canvas.mpl_connect("motion_notify_event", self._motion),
            self.canvas.mpl_connect("button_release_event", self._release),
        ]

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        try:
            self.line.remove()
        except Exception:
            pass

    def _toolbar_busy(self):
        tb = getattr(self.canvas, "toolbar", None)
        return tb is not None and getattr(tb, "mode", "")

    def _redraw(self):
        xs = [v[0] for v in self.verts] + [self.verts[0][0]]
        ys = [v[1] for v in self.verts] + [self.verts[0][1]]
        self.line.set_data(xs, ys)
        self.canvas.draw_idle()

    def _nearest_vertex(self, event):
        best, idx = self.HIT_PX, None
        for i, (vx, vy) in enumerate(self.verts):
            px, py = self.ax.transData.transform((vx, vy))
            d = ((px - event.x) ** 2 + (py - event.y) ** 2) ** 0.5
            if d < best:
                best, idx = d, i
        return idx

    def _press(self, event):
        if event.inaxes != self.ax or event.button != 1 or self._toolbar_busy():
            return
        if event.xdata is None:
            return
        i = self._nearest_vertex(event)
        if i is not None:
            self._mode = ("vertex", i)
        elif Path(self.verts).contains_point((event.xdata, event.ydata)):
            self._mode = ("poly", (event.xdata, event.ydata),
                          [list(v) for v in self.verts])

    def _motion(self, event):
        if self._mode is None or event.inaxes != self.ax or event.xdata is None:
            return
        if self._mode[0] == "vertex":
            self.verts[self._mode[1]] = [event.xdata, event.ydata]
        else:
            (x0, y0), verts0 = self._mode[1], self._mode[2]
            dx, dy = event.xdata - x0, event.ydata - y0
            self.verts = [[vx + dx, vy + dy] for vx, vy in verts0]
        self._redraw()
        if self.on_change:
            self.on_change([tuple(v) for v in self.verts])

    def _release(self, event):
        if self._mode is None:
            return
        self._mode = None
        if self.on_release:
            self.on_release([tuple(v) for v in self.verts])


# --------------------------------------------------------------------------- #
# A density plot that can host an editable polygon
# --------------------------------------------------------------------------- #
class DensityCanvas(FigureCanvas):
    """Renders a square density plot for one (sample, gate) and lets
    you draw a new polygon or drag an existing one to reshape/move it."""

    polygon_committed = QtCore.pyqtSignal(list)  # display-coord vertices

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5.5, 5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._selector: PolygonSelector | None = None
        self._editor: DraggablePolygon | None = None
        self._sample = None
        self._gate: PolygonGate | None = None
        self._parent_mask = None
        self._color_channel = None   # None = color by density; else a channel name
        self._cbar = None            # current colorbar (if any)
        self._show_gate = True       # draw the gate polygon overlay?
        self._editable = True        # attach the draggable editor?

    # ---- drawing ----
    def show_target(self, sample, gate: PolygonGate, display_mask=None,
                    show_gate=True, editable=True):
        """Render one plot.

        display_mask : boolean array over the sample's events to plot. Pass the
            PARENT population's mask to gate on it (edit mode), or the gate's OWN
            mask to view only the events that passed (population view).
        show_gate : draw the gate polygon overlay.
        editable  : make that polygon draggable/reshapeable.
        """
        self._sample = sample
        self._gate = gate
        self._parent_mask = display_mask
        self._show_gate = show_gate
        self._editable = editable
        self._deactivate_editor()
        self._deactivate_selector()
        self._redraw()

    def set_color_channel(self, channel):
        """Color the dots by this channel's intensity (None = density coloring)."""
        self._color_channel = channel
        self._redraw()

    def _display_mask_bool(self):
        n = self._sample.n_events()
        if self._parent_mask is None:
            return np.ones(n, dtype=bool)
        return np.asarray(self._parent_mask, dtype=bool)

    def _display_points(self):
        g, s = self._gate, self._sample
        sel = self._display_mask_bool()
        x = g.x_transform.forward(s.data[g.x_channel].to_numpy()[sel])
        y = g.y_transform.forward(s.data[g.y_channel].to_numpy()[sel])
        return x, y

    def _view_limits(self, x, y):
        """Return (xlo, xhi, ylo, yhi) in DISPLAY coords.

        Uses the gate's saved raw min/max where set (forward-transformed),
        otherwise an auto range from the data percentiles.
        """
        g = self._gate
        def auto(v):
            lo, hi = np.percentile(v, [0.05, 99.5])
            r = (hi - lo) or 1.0
            return lo - 0.03 * r, hi + 0.03 * r
        axlo, axhi = auto(x) if len(x) else (0.0, 1.0)
        aylo, ayhi = auto(y) if len(y) else (0.0, 1.0)
        xlo = g.x_transform.forward(np.array([g.x_min]))[0] if g.x_min is not None else axlo
        xhi = g.x_transform.forward(np.array([g.x_max]))[0] if g.x_max is not None else axhi
        ylo = g.y_transform.forward(np.array([g.y_min]))[0] if g.y_min is not None else aylo
        yhi = g.y_transform.forward(np.array([g.y_max]))[0] if g.y_max is not None else ayhi
        return xlo, xhi, ylo, yhi

    def _raw_tick_formatter(self, transform):
        """Show RAW data values on the axis even when it's log/asinh."""
        def fmt(display_val, _pos):
            raw = float(transform.inverse(np.array([display_val]))[0])
            a = abs(raw)
            if a >= 1e6:
                return f"{raw/1e6:.1f}M"
            if a >= 1e3:
                return f"{raw/1e3:.0f}K"
            return f"{raw:.0f}"
        return FuncFormatter(fmt)

    def _redraw(self):
        # remove any previous colorbar before clearing
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        self.ax.clear()
        self.ax.set_facecolor("white")
        if self._sample is None or self._gate is None:
            self.draw_idle()
            return
        g = self._gate
        x, y = self._display_points()
        xlo, xhi, ylo, yhi = self._view_limits(x, y)

        if len(x) > 0:
            if self._color_channel and self._color_channel in self._sample.channels:
                # --- colour each event by a chosen laser's intensity ---
                sel = self._display_mask_bool()
                ct = default_transform_for(self._color_channel)
                craw = self._sample.data[self._color_channel].to_numpy()[sel]
                cvals = ct.forward(craw)
                order = np.argsort(cvals)
                sc = self.ax.scatter(x[order], y[order], c=cvals[order], s=3,
                                     cmap=FLOW_CMAP, linewidths=0, rasterized=True)
                self._cbar = self.fig.colorbar(sc, ax=self.ax, fraction=0.046,
                                               pad=0.04)
                self._cbar.ax.yaxis.set_major_formatter(
                    self._raw_tick_formatter(ct))
                marker = self._sample.markers.get(self._color_channel, "")
                clabel = (f"{self._color_channel} ({marker})"
                          if marker and marker != self._color_channel
                          else self._color_channel)
                self._cbar.set_label(clabel)
            else:
                # --- default: colour by local density  ---
                c = pseudocolor_density(x, y, (xlo, xhi), (ylo, yhi))
                order = np.argsort(c)
                self.ax.scatter(x[order], y[order], c=c[order], s=3, cmap=FLOW_CMAP,
                                linewidths=0, rasterized=True)

        self.ax.set_xlim(xlo, xhi)
        self.ax.set_ylim(ylo, yhi)
        self.ax.set_box_aspect(1)  # square plotting box
        self.ax.xaxis.set_major_formatter(self._raw_tick_formatter(g.x_transform))
        self.ax.yaxis.set_major_formatter(self._raw_tick_formatter(g.y_transform))

        self.ax.set_xlabel(self._axis_label(g.x_channel, g.x_transform))
        self.ax.set_ylabel(self._axis_label(g.y_channel, g.y_transform))
        self.ax.set_title(f"{g.name}  ({self._sample.sample_id})")

        # Gate overlay / draggable editor (only in edit mode).
        if self._show_gate and g.is_defined():
            self._attach_editor() if self._editable else self.ax.add_patch(
                MplPolygon(np.asarray(g.vertices), closed=True, fill=False,
                           edgecolor="black", lw=1.6))
        self.draw_idle()

    def _axis_label(self, channel, transform):
        marker = self._sample.markers.get(channel, "") if self._sample else ""
        base = f"{channel} ({marker})" if marker and marker != channel else channel
        suffix = "" if isinstance(transform, LinearAxis) else f" [{transform.name}]"
        return base + suffix

    # ---- axis controls (called live from the main window) ----
    def set_axis_scale(self, axis: str, name: str):
        """Change X or Y scale to 'linear'/'log'/'asinh', keeping the gate."""
        g = self._gate
        if g is None:
            return
        new = AXIS_TRANSFORMS[name]()
        if axis == "x":
            g.set_transforms(new, g.y_transform)
        else:
            g.set_transforms(g.x_transform, new)
        self._redraw()
        # transforms affect gating -> tell the app to re-gate
        self.polygon_committed.emit(list(g.vertices))

    def set_axis_limit(self, which: str, value):
        """Set one of x_min/x_max/y_min/y_max (raw units, or None for auto)."""
        g = self._gate
        if g is None:
            return
        setattr(g, which, value)
        self._redraw()

    # ---- polygon editing ----
    def start_drawing(self):
        """Draw a NEW polygon from scratch (replaces any existing one)."""
        if self._gate is None:
            return
        self._deactivate_editor()
        self._deactivate_selector()

        def on_select(verts):
            if len(verts) >= 3:
                self._gate.vertices = [tuple(map(float, v)) for v in verts]
                self._deactivate_selector()
                self._redraw()                       # becomes draggable
                self.polygon_committed.emit(list(self._gate.vertices))

        self._selector = PolygonSelector(
            self.ax, on_select, useblit=True,
            props=dict(color="black", linewidth=2),
        )
        self.draw_idle()

    def commit_polygon(self) -> bool:
        """Commit an in-progress freehand draw (if the selector has ≥3 points)."""
        if self._selector is not None and len(self._selector.verts) >= 3:
            self._gate.vertices = [tuple(map(float, v)) for v in self._selector.verts]
            self._deactivate_selector()
            self._redraw()
            self.polygon_committed.emit(list(self._gate.vertices))
            return True
        return self._gate is not None and self._gate.is_defined()

    def clear_polygon(self):
        if self._gate is not None:
            self._gate.vertices = []
        self._deactivate_editor()
        self._deactivate_selector()
        self._redraw()
        self.polygon_committed.emit([])

    def _attach_editor(self):
        self._deactivate_editor()

        def on_release(verts):
            self._gate.vertices = [tuple(v) for v in verts]
            self.polygon_committed.emit(list(self._gate.vertices))

        self._editor = DraggablePolygon(
            self.ax, self._gate.vertices, on_release=on_release, color="black")

    def _deactivate_editor(self):
        if self._editor is not None:
            self._editor.disconnect()
            self._editor = None

    def _deactivate_selector(self):
        if self._selector is not None:
            try:
                self._selector.disconnect_events()
                self._selector.set_visible(False)
            except Exception:
                pass
            self._selector = None


# --------------------------------------------------------------------------- #
# Comparison window: edit one gate on a chosen sample, see it on all samples
# --------------------------------------------------------------------------- #
class CompareDialog(QtWidgets.QDialog):
    """Left: an editable density plot of the gate on one chosen sample.
    Right: small read-only plots of the SAME gate on every sample, with %.

    Because the gate is shared, drawing on the left and clicking Apply updates
    every sample at once — the panels on the right refresh to match.
    """

    def __init__(self, session: Session, gate: PolygonGate, parent=None):
        super().__init__(parent)
        self.session = session
        self.gate = gate
        self.setWindowTitle(f"Edit & compare gate '{gate.name}'")
        self.resize(1150, 720)

        root = QtWidgets.QVBoxLayout(self)

        # -- controls row --
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Edit on sample:"))
        self.sample_pick = QtWidgets.QComboBox()
        self.sample_pick.addItems(list(session.samples.keys()))
        row.addWidget(self.sample_pick)
        self.btn_draw = QtWidgets.QPushButton("Draw / edit polygon")
        self.btn_apply = QtWidgets.QPushButton("Apply to all samples")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        for b in (self.btn_draw, self.btn_apply, self.btn_clear):
            row.addWidget(b)
        row.addStretch(1)
        root.addLayout(row)

        # -- split: editor (left) | grid (right) --
        split = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        self.editor = DensityCanvas()
        left.addWidget(NavigationToolbar(self.editor, self))
        left.addWidget(self.editor, 1)
        split.addLayout(left, 3)

        self.grid = FigureCanvas(Figure(figsize=(5, 6), tight_layout=True))
        split.addWidget(self.grid, 2)
        root.addLayout(split, 1)

        # wiring
        self.sample_pick.currentTextChanged.connect(self._load_editor)
        self.btn_draw.clicked.connect(lambda: self.editor.start_drawing())
        self.btn_clear.clicked.connect(self._clear)
        self.btn_apply.clicked.connect(self._apply)
        self.editor.polygon_committed.connect(lambda _v: self._after_edit())

        self._load_editor()
        self._refresh_grid()

    def _current_sample(self):
        return self.session.samples[self.sample_pick.currentText()]

    def _load_editor(self, *_):
        s = self._current_sample()
        masks = self.session.masks_for(s.sample_id)
        g = self.gate
        pmask = masks[g.parent] if g.parent and g.parent != "root" else None
        self.editor.show_target(s, g, pmask)

    def _apply(self):
        # commit whatever is drawn; _after_edit refreshes everything
        if not self.editor.commit_polygon():
            self._after_edit()

    def _clear(self):
        self.editor.clear_polygon()

    def _after_edit(self):
        self.session.apply()
        self._load_editor()
        self._refresh_grid()

    def _refresh_grid(self):
        fig = self.grid.figure
        fig.clear()
        g = self.gate
        ids = list(self.session.samples.keys())
        n = len(ids)
        cols = 1 if n <= 3 else 2
        rows = int(np.ceil(n / cols))
        for i, sid in enumerate(ids):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_facecolor("white")
            s = self.session.samples[sid]
            masks = self.session.masks_for(sid)
            pmask = masks[g.parent] if g.parent and g.parent != "root" else None
            x = g.x_transform.forward(s.data[g.x_channel].to_numpy())
            y = g.y_transform.forward(s.data[g.y_channel].to_numpy())
            if pmask is not None:
                x, y = x[pmask], y[pmask]
            if len(x):
                xlo, xhi = np.percentile(x, [0.05, 99.5])
                ylo, yhi = np.percentile(y, [0.05, 99.5])
                c = pseudocolor_density(x, y, (xlo, xhi), (ylo, yhi), bins=160)
                order = np.argsort(c)
                ax.scatter(x[order], y[order], c=c[order], s=2, cmap=FLOW_CMAP,
                           linewidths=0, rasterized=True)
                ax.set_xlim(xlo, xhi)
                ax.set_ylim(ylo, yhi)
            if g.is_defined():
                ax.add_patch(MplPolygon(np.asarray(g.vertices), closed=True,
                                        fill=False, edgecolor="black", lw=1.4))
            ax.set_box_aspect(1)
            denom = pmask.sum() if pmask is not None else s.n_events()
            pct = 100.0 * masks[g.name].sum() / max(1, denom)
            ax.set_title(f"{sid} — {pct:.1f}%", fontsize=9)
            ax.tick_params(labelsize=7)
        self.grid.draw_idle()


# --------------------------------------------------------------------------- #
# Subpopulation dialog: pick two lasers, draw several polygons, name each
# --------------------------------------------------------------------------- #
class SubpopulationDialog(QtWidgets.QDialog):
    """Create one or more subpopulations under a chosen parent population.

    You pick the two channels (lasers/markers) to plot, draw a polygon, and give
    it a name — that becomes a new child gate (a new branch). Repeat to add as
    many sibling subpopulations as you like on the same axes.
    """

    def __init__(self, session: Session, parent_name: str, parent=None):
        super().__init__(parent)
        self.session = session
        self.parent_name = parent_name
        self.setWindowTitle(f"Add subpopulations under '{parent_name}'")
        self.resize(900, 760)
        mm = session.marker_map()
        chans = session.common_channels()

        root = QtWidgets.QVBoxLayout(self)

        # --- channel pickers (choose the two relevant lasers) ---
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(QtWidgets.QLabel(f"<b>Parent:</b> {parent_name}"))
        prow.addSpacing(16)
        prow.addWidget(QtWidgets.QLabel("X laser:"))
        self.x_pick = QtWidgets.QComboBox()
        prow.addWidget(self.x_pick)
        prow.addWidget(QtWidgets.QLabel("Y laser:"))
        self.y_pick = QtWidgets.QComboBox()
        prow.addWidget(self.y_pick)
        prow.addStretch(1)
        root.addLayout(prow)

        for cb in (self.x_pick, self.y_pick):
            for c in chans:
                label = f"{c} ({mm[c]})" if mm.get(c) and mm[c] != c else c
                cb.addItem(label, c)          # detector stored as itemData
        if self.y_pick.count() > 1:
            self.y_pick.setCurrentIndex(1)

        # --- editable density plot ---
        self.editor = DensityCanvas()
        root.addWidget(NavigationToolbar(self.editor, self))
        root.addWidget(self.editor, 1)

        # --- actions ---
        brow = QtWidgets.QHBoxLayout()
        self.btn_draw = QtWidgets.QPushButton("Draw polygon")
        self.btn_add = QtWidgets.QPushButton("Add as subpopulation…")
        self.btn_add.setStyleSheet("font-weight: bold;")
        brow.addWidget(self.btn_draw)
        brow.addWidget(self.btn_add)
        brow.addStretch(1)
        self.btn_done = QtWidgets.QPushButton("Done")
        brow.addWidget(self.btn_done)
        root.addLayout(brow)

        root.addWidget(QtWidgets.QLabel("Subpopulations added this session:"))
        self.listw = QtWidgets.QListWidget()
        self.listw.setMaximumHeight(120)
        root.addWidget(self.listw)

        self.x_pick.currentIndexChanged.connect(self._reload)
        self.y_pick.currentIndexChanged.connect(self._reload)
        self.btn_draw.clicked.connect(lambda: self.editor.start_drawing())
        self.btn_add.clicked.connect(self._add)
        self.btn_done.clicked.connect(self.accept)

        self.added: list[str] = []
        self.display_gate: PolygonGate | None = None
        self._reload()

    def _channels(self):
        return self.x_pick.currentData(), self.y_pick.currentData()

    def _reload(self, *_):
        """Rebuild the preview plot for the currently chosen channels."""
        xch, ych = self._channels()
        mm = self.session.marker_map()
        self.display_gate = PolygonGate(
            name="__preview__", x_channel=xch, y_channel=ych,
            parent=self.parent_name, vertices=[],
            x_transform=default_transform_for(xch, mm.get(xch, "")),
            y_transform=default_transform_for(ych, mm.get(ych, "")),
        )
        sid = next(iter(self.session.samples))
        masks = self.session.masks_for(sid)
        pmask = (masks[self.parent_name]
                 if self.parent_name and self.parent_name != "root" else None)
        self.editor.show_target(self.session.samples[sid], self.display_gate, pmask)

    def _add(self):
        """Turn the drawn polygon into a named child gate (a new branch)."""
        if not self.editor.commit_polygon():
            QtWidgets.QMessageBox.information(
                self, "No polygon", "Draw a polygon first, then add it.")
            return
        verts = list(self.display_gate.vertices)
        if len(verts) < 3:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Name subpopulation", "Subpopulation name:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if name in self.session.tree.gates:
            QtWidgets.QMessageBox.warning(
                self, "Name taken", f"A gate named '{name}' already exists.")
            return
        xch, ych = self._channels()
        mm = self.session.marker_map()
        self.session.tree.add_gate(
            name, self.parent_name, xch, ych,
            default_transform_for(xch, mm.get(xch, "")),
            default_transform_for(ych, mm.get(ych, "")),
            vertices=verts,
        )
        self.session.apply()
        self.added.append(name)
        self.listw.addItem(f"{name}   [{xch} × {ych}]   {len(verts)} pts")
        self.editor.clear_polygon()   # ready for the next polygon
        self._reload()


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class GatingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlowGate v6 — brighter density colormap")
        self.resize(1200, 850)
        self.session = Session()

        # --- central density canvas + controls ---
        central = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(central)
        self.canvas = DensityCanvas()
        cv.addWidget(NavigationToolbar(self.canvas, self))
        cv.addWidget(self.canvas, stretch=1)

        # --- live axis controls: lasers, scale, min/max ---
        ax = QtWidgets.QGridLayout()
        ax.addWidget(QtWidgets.QLabel("X laser:"), 0, 0)
        self.x_channel_pick = QtWidgets.QComboBox()
        ax.addWidget(self.x_channel_pick, 0, 1)
        ax.addWidget(QtWidgets.QLabel("Y laser:"), 0, 2)
        self.y_channel_pick = QtWidgets.QComboBox()
        ax.addWidget(self.y_channel_pick, 0, 3)
        self.lbl_laser_hint = QtWidgets.QLabel("(laser choice enabled for Category "
                                               "and its subpopulations)")
        self.lbl_laser_hint.setStyleSheet("color:#888; font-size:11px;")
        ax.addWidget(self.lbl_laser_hint, 0, 4, 1, 4)

        ax.addWidget(QtWidgets.QLabel("X scale:"), 1, 0)
        self.x_scale = QtWidgets.QComboBox()
        self.x_scale.addItems(["linear", "log", "asinh"])
        ax.addWidget(self.x_scale, 1, 1)
        ax.addWidget(QtWidgets.QLabel("X min/max:"), 1, 2)
        self.x_min = QtWidgets.QLineEdit(); self.x_min.setPlaceholderText("auto")
        self.x_max = QtWidgets.QLineEdit(); self.x_max.setPlaceholderText("auto")
        ax.addWidget(self.x_min, 1, 3)
        ax.addWidget(self.x_max, 1, 4)

        ax.addWidget(QtWidgets.QLabel("Y scale:"), 2, 0)
        self.y_scale = QtWidgets.QComboBox()
        self.y_scale.addItems(["linear", "log", "asinh"])
        ax.addWidget(self.y_scale, 2, 1)
        ax.addWidget(QtWidgets.QLabel("Y min/max:"), 2, 2)
        self.y_min = QtWidgets.QLineEdit(); self.y_min.setPlaceholderText("auto")
        self.y_max = QtWidgets.QLineEdit(); self.y_max.setPlaceholderText("auto")
        ax.addWidget(self.y_min, 2, 3)
        ax.addWidget(self.y_max, 2, 4)

        ax.addWidget(QtWidgets.QLabel("Color by:"), 3, 0)
        self.color_pick = QtWidgets.QComboBox()
        ax.addWidget(self.color_pick, 3, 1)
        self.chk_gated = QtWidgets.QCheckBox("Show only events inside this gate "
                                             "(the gated population)")
        ax.addWidget(self.chk_gated, 3, 2, 1, 3)
        cv.addLayout(ax)

        self.x_channel_pick.activated.connect(lambda: self._change_channel("x"))
        self.y_channel_pick.activated.connect(lambda: self._change_channel("y"))
        self.x_scale.activated.connect(lambda: self._change_scale("x"))
        self.y_scale.activated.connect(lambda: self._change_scale("y"))
        self.color_pick.activated.connect(self._change_color)
        self.chk_gated.toggled.connect(self._refresh_canvas)
        for w in (self.x_min, self.x_max, self.y_min, self.y_max):
            w.editingFinished.connect(self._change_limits)

        controls = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("① Load FCS files…")
        self.btn_load.setStyleSheet("font-weight: bold;")
        controls.addWidget(self.btn_load)
        self.btn_draw = QtWidgets.QPushButton("Draw / edit polygon")
        self.btn_apply = QtWidgets.QPushButton("Apply polygon → gate")
        self.btn_clear = QtWidgets.QPushButton("Clear polygon")
        self.btn_compare = QtWidgets.QPushButton("Compare across samples")
        self.btn_subpop = QtWidgets.QPushButton("➕ Add subpopulations…")
        self.btn_delete = QtWidgets.QPushButton("🗑 Delete gate")
        for b in (self.btn_draw, self.btn_apply, self.btn_clear, self.btn_compare,
                  self.btn_subpop, self.btn_delete):
            controls.addWidget(b)
        cv.addLayout(controls)
        self.setCentralWidget(central)

        self.btn_load.clicked.connect(self._load_fcs)
        self.btn_draw.clicked.connect(self._draw_clicked)
        self.btn_apply.clicked.connect(self._apply_polygon)
        self.btn_clear.clicked.connect(self.canvas.clear_polygon)
        self.btn_compare.clicked.connect(self._compare)
        self.btn_subpop.clicked.connect(self._add_subpopulations)
        self.btn_delete.clicked.connect(self._delete_gate)

        # --- left dock: samples + populations ---
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(QtWidgets.QLabel("Samples"))
        self.sample_list = QtWidgets.QListWidget()
        self.sample_list.currentTextChanged.connect(self._refresh_canvas)
        lv.addWidget(self.sample_list)
        lv.addWidget(QtWidgets.QLabel("Populations (gate hierarchy)"))
        self.gate_tree = QtWidgets.QTreeWidget()
        self.gate_tree.setHeaderLabels(["Population"])
        self.gate_tree.currentItemChanged.connect(self._refresh_canvas)
        lv.addWidget(self.gate_tree)
        dock_l = QtWidgets.QDockWidget("Navigator", self)
        dock_l.setWidget(left)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock_l)

        # --- bottom dock: stats ---
        self.stats_table = QtWidgets.QTableWidget()
        dock_b = QtWidgets.QDockWidget("Statistics (all samples × gates)", self)
        dock_b.setWidget(self.stats_table)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock_b)

        self._build_menu()

    # ---- menu / actions ----
    def _build_menu(self):
        mb = self.menuBar()
        # Keep the menu INSIDE the window. On macOS Qt otherwise moves it to the
        # top-of-screen menu bar; on some Linux setups the global menu hides it.
        mb.setNativeMenuBar(False)
        m = mb.addMenu("&File")
        m.addAction("Load FCS files…", self._load_fcs)
        m.addAction("Save gates…", self._save_gates)
        m.addAction("Load gates…", self._load_gates)
        m.addSeparator()
        m.addAction("Export stats CSV…", self._export_stats)
        g = mb.addMenu("&Gating")
        g.addAction("Re-gate all samples", self._regate)

        # A visible toolbar as well, so file loading is never hidden by a
        # missing/global menu bar regardless of platform.
        tb = self.addToolBar("Main")
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        tb.addAction("Load FCS files…", self._load_fcs)
        tb.addAction("Load gates…", self._load_gates)
        tb.addAction("Save gates…", self._save_gates)
        tb.addAction("Re-gate all", self._regate)
        tb.addAction("Export stats CSV…", self._export_stats)

    def _load_fcs(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load FCS files", "", "FCS files (*.fcs)")
        if not paths:
            return
        for p in paths:
            self.session.add_sample(p)
        self._init_gates_from_config()
        self._refresh_sample_list()
        self._regate()

    def _init_gates_from_config(self):
        """Populate the shared tree with (initially empty) gates from config,
        resolving marker names to this panel's detectors."""
        if self.session.tree.order:
            return  # already built
        mm = self.session.marker_map()
        chans = self.session.common_channels()
        for spec in GATE_HIERARCHY:
            try:
                xch = resolve_channel(spec.x, mm, chans)
                ych = resolve_channel(spec.y, mm, chans)
            except KeyError as e:
                print(f"[skip gate {spec.name}] {e}")
                continue
            self.session.tree.add(PolygonGate(
                name=spec.name, x_channel=xch, y_channel=ych, parent=spec.parent,
                vertices=[],
                x_transform=default_transform_for(xch, mm.get(xch, "")),
                y_transform=default_transform_for(ych, mm.get(ych, "")),
            ))
        self._refresh_gate_tree()
        self._populate_channel_lists()

    # ---- UI refreshers ----
    def _refresh_sample_list(self):
        self.sample_list.clear()
        for sid in self.session.samples:
            self.sample_list.addItem(sid)
        if self.sample_list.count() and self.sample_list.currentRow() < 0:
            self.sample_list.setCurrentRow(0)

    def _refresh_gate_tree(self):
        self.gate_tree.clear()
        items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for name in self.session.tree.order:
            gate = self.session.tree.gates[name]
            item = QtWidgets.QTreeWidgetItem([name])
            item.setData(0, QtCore.Qt.UserRole, name)
            items[name] = item
            if gate.parent and gate.parent in items:
                items[gate.parent].addChild(item)
            else:
                self.gate_tree.addTopLevelItem(item)
        self.gate_tree.expandAll()

    def _current_sample(self):
        it = self.sample_list.currentItem()
        return self.session.samples.get(it.text()) if it else None

    def _current_gate(self):
        it = self.gate_tree.currentItem()
        if not it:
            return None
        return self.session.tree.gates.get(it.data(0, QtCore.Qt.UserRole))

    def _refresh_canvas(self, *_):
        s, g = self._current_sample(), self._current_gate()
        if s is None or g is None:
            return
        masks = self.session.masks_for(s.sample_id)
        if self.chk_gated.isChecked() and g.is_defined():
            # population view: show ONLY the events that passed this gate
            self.canvas.show_target(s, g, masks[g.name],
                                    show_gate=False, editable=False)
        else:
            # edit view: show the parent population with this gate on top
            pmask = masks[g.parent] if g.parent and g.parent != "root" else None
            self.canvas.show_target(s, g, pmask, show_gate=True, editable=True)
        self.canvas.set_color_channel(self.color_pick.currentData())
        self._sync_axis_controls()

    def _change_color(self):
        self.canvas.set_color_channel(self.color_pick.currentData())

    def _draw_clicked(self):
        # drawing always happens on the parent population (edit view)
        if self.chk_gated.isChecked():
            self.chk_gated.setChecked(False)   # triggers _refresh_canvas
        else:
            self._refresh_canvas()
        self.canvas.start_drawing()

    # ---- live axis controls ----
    def _populate_channel_lists(self):
        """Fill the X/Y laser and Color-by dropdowns with available channels."""
        mm = self.session.marker_map()
        chans = self.session.common_channels()
        for cb in (self.x_channel_pick, self.y_channel_pick):
            cb.blockSignals(True)
            cb.clear()
            for c in chans:
                label = f"{c} ({mm[c]})" if mm.get(c) and mm[c] != c else c
                cb.addItem(label, c)
            cb.blockSignals(False)
        # Color-by: "Density" (default) plus every channel
        self.color_pick.blockSignals(True)
        self.color_pick.clear()
        self.color_pick.addItem("Density", None)
        for c in chans:
            label = f"{c} ({mm[c]})" if mm.get(c) and mm[c] != c else c
            self.color_pick.addItem(label, c)
        self.color_pick.blockSignals(False)

    def _laser_choice_allowed(self, gate) -> bool:
        """Lasers are user-choosable for Category_Cells and anything below it.
        If there's no Category_Cells gate, allow it for every gate."""
        tree = self.session.tree
        if "Category_Cells" not in tree.gates:
            return True
        return tree.is_at_or_below(gate.name, "Category_Cells")

    def _sync_axis_controls(self):
        """Make the control widgets reflect the currently selected gate."""
        g = self._current_gate()
        if g is None:
            return
        widgets = [self.x_channel_pick, self.y_channel_pick, self.x_scale,
                   self.y_scale, self.x_min, self.x_max, self.y_min, self.y_max]
        for w in widgets:
            w.blockSignals(True)
        # channels
        ix = self.x_channel_pick.findData(g.x_channel)
        iy = self.y_channel_pick.findData(g.y_channel)
        if ix >= 0:
            self.x_channel_pick.setCurrentIndex(ix)
        if iy >= 0:
            self.y_channel_pick.setCurrentIndex(iy)
        allow = self._laser_choice_allowed(g)
        self.x_channel_pick.setEnabled(allow)
        self.y_channel_pick.setEnabled(allow)
        # scales
        self.x_scale.setCurrentText(g.x_transform.name)
        self.y_scale.setCurrentText(g.y_transform.name)
        # limits (raw units; blank = auto)
        self.x_min.setText("" if g.x_min is None else str(g.x_min))
        self.x_max.setText("" if g.x_max is None else str(g.x_max))
        self.y_min.setText("" if g.y_min is None else str(g.y_min))
        self.y_max.setText("" if g.y_max is None else str(g.y_max))
        for w in widgets:
            w.blockSignals(False)

    def _change_channel(self, axis):
        g = self._current_gate()
        if g is None or not self._laser_choice_allowed(g):
            return
        if g.is_defined() and QtWidgets.QMessageBox.question(
                self, "Change laser",
                "Changing the laser resets this gate's polygon. Continue?"
        ) != QtWidgets.QMessageBox.Yes:
            self._sync_axis_controls()
            return
        mm = self.session.marker_map()
        xch = self.x_channel_pick.currentData()
        ych = self.y_channel_pick.currentData()
        g.set_channels(xch, ych,
                       default_transform_for(xch, mm.get(xch, "")),
                       default_transform_for(ych, mm.get(ych, "")))
        self._regate()

    def _change_scale(self, axis):
        name = (self.x_scale if axis == "x" else self.y_scale).currentText()
        self.canvas.set_axis_scale(axis, name)   # remaps polygon + emits re-gate
        self._regate()

    def _change_limits(self):
        g = self._current_gate()
        if g is None:
            return
        def val(w):
            t = w.text().strip()
            try:
                return float(t) if t else None
            except ValueError:
                return None
        g.x_min, g.x_max = val(self.x_min), val(self.x_max)
        g.y_min, g.y_max = val(self.y_min), val(self.y_max)
        self.canvas._redraw()

    def _apply_polygon(self):
        if self.canvas.commit_polygon():
            self._regate()

    def _regate(self):
        self.session.apply()
        self._refresh_stats()
        self._refresh_canvas()

    def _compare(self):
        g = self._current_gate()
        if g is None:
            return
        self.session.apply()
        CompareDialog(self.session, g, self).exec_()

    def _add_subpopulations(self):
        """Open the subpopulation dialog under the selected population (or root)."""
        if not self.session.samples:
            return
        g = self._current_gate()
        parent_name = g.name if g is not None else "root"
        dlg = SubpopulationDialog(self.session, parent_name, self)
        dlg.exec_()
        if dlg.added:
            self._refresh_gate_tree()
            self._refresh_stats()
            self._refresh_canvas()

    def _delete_gate(self):
        """Delete the selected gate and everything beneath it."""
        g = self._current_gate()
        if g is None:
            return
        extra = self.session.tree.descendants(g.name)
        msg = f"Delete gate '{g.name}'"
        if extra:
            msg += f" and its {len(extra)} descendant(s)"
        msg += "?"
        if QtWidgets.QMessageBox.question(self, "Delete gate", msg) != \
                QtWidgets.QMessageBox.Yes:
            return
        self.session.remove_gate(g.name)
        self._refresh_gate_tree()
        self._refresh_stats()

    def _refresh_stats(self):
        df = self.session.stats_frame()
        self.stats_table.setColumnCount(len(df.columns))
        self.stats_table.setHorizontalHeaderLabels(list(df.columns))
        self.stats_table.setRowCount(len(df))
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                self.stats_table.setItem(
                    r, c, QtWidgets.QTableWidgetItem(str(df.iloc[r, c])))
        self.stats_table.resizeColumnsToContents()

    def _save_gates(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save gates", "gates.json", "JSON (*.json)")
        if path:
            self.session.save_gates(path)

    def _load_gates(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load gates", "", "JSON (*.json)")
        if path:
            self.session.load_gates(path)
            self._refresh_gate_tree()
            self._regate()

    def _export_stats(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export stats", "stats.csv", "CSV (*.csv)")
        if path:
            self.session.stats_frame().to_csv(path, index=False)

    def load_paths(self, paths):
        """Load FCS files programmatically (e.g. from the command line)."""
        added = False
        for p in paths:
            if os.path.exists(p):
                self.session.add_sample(p)
                added = True
            else:
                print(f"[skip] file not found: {p}")
        if added:
            self._init_gates_from_config()
            self._refresh_sample_list()
            self._regate()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = GatingApp()
    # Any .fcs paths on the command line are loaded immediately, so you can
    # bypass the file dialog entirely:  python run.py file1.fcs file2.fcs
    paths = [a for a in sys.argv[1:] if a.lower().endswith(".fcs")]
    if paths:
        w.load_paths(paths)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
