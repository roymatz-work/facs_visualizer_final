# FlowGate

An interactive gating tool for flow cytometry (FCS) data,
built in Python. It reads FCS files into DataFrames, lets you draw polygon gates
by hand on density plots, applies each gate hierarchically, and
applies **one shared gate definition across all your samples (mice)** so you can
compare and re-adjust thresholds across the whole cohort at once.

## What it does today

- **FCS → DataFrame** via `flowkit`, keeping both detector names (`BV421-A`) and
  marker/stain names (`CD45`), so gates can be specified by marker.
- **The gating hierarchy**:
  Beads → Real_Cells → Real_Fwd_Cells → Real_Fwd_Side_Cells → Live_Cells →
  Category_Cells, plus a place to add subpopulation gates.
- **Manual polygon gating** — draw arbitrary polygons directly on the plot
  (matplotlib `PolygonSelector`). Any shape, any number of
  vertices.
- **Per-axis transforms** — linear for scatter, arcsinh (biexp-like) for
  fluorescence; gate math happens in the same display space you draw in.
- **One gate, all samples** — the shared `GatingTree` is applied to every loaded
  sample. Edit a gate once and re-gate the whole cohort.
- **Compare across samples** — a grid view showing the current gate on every
  mouse side by side, with the % gated on each.
- **Live statistics** — every sample × every gate, with % of parent.
- **Save / load gates** to JSON, so thresholds persist and can be reused or
  shared.

## Install & run

```bash
pip install -r requirements.txt        # flowkit, flowio, matplotlib, PyQt5, ...
python make_synthetic_data.py          # optional: writes ./data/Mouse1..4.fcs to try it
python run.py                          # launch the GUI
```

In the GUI: **File → Load FCS files**, pick your `.fcs` files (or the synthetic
ones). Select a sample and a population, click **Draw / edit polygon**, click to
place vertices, then **Apply polygon → gate**. Statistics update for every
sample. Use **Compare across samples** to see the gate on all mice at once, and
**File → Save gates** to persist your thresholds.

## Architecture

```
flowgate/
  core.py     # engine, no GUI: transforms, FlowSample, PolygonGate,
              #   GatingTree (shared), Session (samples + apply-to-all + stats)
  config.py   # the gate hierarchy as data; marker->detector resolution
  gui.py      # PyQt5 app: density canvas + PolygonSelector, tree, stats, compare
make_synthetic_data.py  # generates test mice
test_core.py            # headless test of the whole engine
run.py                  # entry point
```

Key idea: a **gate is defined once** (a polygon in display coordinates) and the
same object is applied to every sample. To gate a new sample you transform its
raw values into the gate's display space and test point-in-polygon. So "what you
drew" and "what gets gated" are identical across all mice — that's what makes the
cross-sample comparison and re-adjustment (your requirement #8) work.

## Extending

- **Subpopulation gates (step 7):** add `GateSpec(...)` rows to `GATE_HIERARCHY`
  in `config.py` with the right `parent`. They appear in the tree automatically.
- **Compensation:** `flowkit` supports spillover matrices; add a step in
  `FlowSample.__init__` to apply `apply_compensation` before gating.
- **Editing a saved gate in the Compare view:** the grid is currently read-only;
  making its polygon draggable and writing back to the shared gate is the natural
  next step for live cross-sample threshold tuning.
- **FlowJo interop:** `flowkit` can import/export GatingML and read FlowJo `.wsp`
  workspaces if you later want to exchange gates with FlowJo.

## Notes / limitations

- Density plots use a 2D histogram (fast for 10^4–10^6 events). Percentile-based
  axis limits keep outliers from squashing the view.
- Marker resolution is tolerant (`"Zombie"` matches `"Zombie NIR"`), but if a
  panel lacks a referenced marker, that gate is skipped with a console note.
- The app assumes samples share a common channel/panel layout.
