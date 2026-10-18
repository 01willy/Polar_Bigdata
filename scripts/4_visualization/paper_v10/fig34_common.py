# -*- coding: utf-8 -*-
"""Shared style, layout and QA helpers for paper v10 Fig. 3 (fig3_accuracy.py) and Fig. 4 (fig4_labels.py).

Style (docs/PAPER_PLAN_v10_2026-10-16.md section 2, design/style_tokens_v4.json color.paper_v10, read through src/polar/style_v5.py
with POLAR_PALETTE=paper): white background, black text, thin black axes, colour only for methods, regions and products in greys
with marker shapes, FreeSans 7 pt (ticks 6.5 pt, nothing below 6 pt), panel letters 8 pt bold, no titles or sentences in the figure.
Layout is placed on a fixed 180 mm canvas in mm (ax_mm, y measured from the top edge). Export: pdf, svg (text kept as text), png 600 dpi.
QA (qa()): minimum font size, text-text overlaps (rotated tick labels by perpendicular spacing), text outside the canvas,
text over a foreign plotting area, and in-axes labels touching data lines or markers.
"""
from __future__ import annotations
import itertools, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgb, to_hex
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
if os.environ.get("POLAR_PALETTE") != "paper":
    raise SystemExit("run with POLAR_PALETTE=paper (restrained paper palette in polar.style_v5)")
import polar.style_v5 as S  # noqa: E402

OUT = ROOT / "outputs" / "figures" / "paper" / "v10"; SRC = OUT / "source_data"
MM = 1 / 25.4
W = 180.0                                   # figure width (mm)

# ---------------------------------------------------------------- sizes (pt)
FS = 7.0          # axis labels, legends, facet labels
FS_TICK = 6.5     # tick labels, direct labels, printed values
FS_CELL = 6.0     # heat-map cell values (smallest text in the figures)
FS_LETTER = 8.0   # panel letters, bold
LW_AXIS = 0.5     # axes, ticks
LW_DATA = 1.0     # method lines (as paper_v10/common.py LW_MAIN, Figs 5-6)
LW_P0 = 0.7       # Stefan (source coeff.) thin grey line
LW_REF = 0.5      # zero and reference lines
LW_CI = 0.8       # interval whiskers
MS = 3.2          # circle diameter (pt), as paper_v10/common.py
MS_SQ = 2.8       # square side (pt): same area as a 3.2 pt circle (Stefan methods use squares, as in Fig. 5)
MS_P0 = 2.4
MEW = 0.6         # open-marker edge
DASH = (0, (3.2, 1.6))   # Direct ML line

# ---------------------------------------------------------------- colours (from tokens via polar.style_v5, paper palette)
BLACK = "#000000"
ZERO = S.NEUTRAL["zero_line"]                     # #808080
REG = ["Alaska", "Lena", "Canada"]
REG_EN = dict(S.REGION_EN)                        # Alaska, Lena Delta, Canada (NW)
REG_HEX = dict(S.REGION_HEX)                      # greys
REG_MK = dict(S.REGION_MARKER)                    # o, s, ^
REG_MS = {"Alaska": 3.0, "Lena": 2.8, "Canada": 3.4}   # visually equal marker areas


def tint(hex_, f):
    """Mix a colour with white: f = share of the colour (0 → white, 1 → colour). Solid colour, no transparency."""
    c = np.array(to_rgb(hex_)); return to_hex(1 - f * (1 - c))


METH = {   # method styles (one place for both figures)
    "P0": dict(label="Stefan (source coeff.)", color=S.METHOD_HEX["P0"], ls="-", lw=LW_P0, marker="s", ms=MS_P0, open=False),
    "P1": dict(label="Stefan (recalibrated)", color=S.METHOD_HEX["P1"], ls="-", lw=LW_DATA, marker="s", ms=MS_SQ, open=False),
    "R1": dict(label="Physics + residual ML", color=S.METHOD_HEX["R1"], ls="-", lw=LW_DATA, marker="o", ms=MS, open=False),
    "D0": dict(label="Direct ML", color=S.METHOD_HEX["D0"], ls=DASH, lw=0.9, marker="o", ms=MS, open=True),
}
BAND = tint(S.METHOD_HEX["R1"], 0.22)             # light vermillion 95 % interval band
HALO = tint(S.METHOD_HEX["P1"], 0.40)             # light blue (Stefan recalibrated under Physics + residual ML, Fig. 4c)


def setup():
    for f in (S.F["latin_regular_file"], S.F["latin_bold_file"]):
        fm.fontManager.addfont(f)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [S.F["latin"]], "font.size": FS, "text.color": BLACK,
        "axes.labelsize": FS, "axes.labelcolor": BLACK, "axes.edgecolor": BLACK, "axes.linewidth": LW_AXIS, "axes.labelpad": 2.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "white", "figure.facecolor": "white",
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "xtick.color": BLACK, "ytick.color": BLACK,
        "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "legend.fontsize": FS_TICK, "legend.frameon": False, "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
        "hatch.linewidth": 0.4, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True, "mathtext.default": "regular",
        "figure.dpi": 150, "savefig.facecolor": "white",
    })


def new_fig(h_mm):
    fig = plt.figure(figsize=(W * MM, h_mm * MM)); fig.patch.set_facecolor("white")
    fig._h_mm = h_mm
    return fig


def ax_mm(fig, x, y_top, w, h, **kw):
    """Axes in mm: left x, top y measured from the top edge of the canvas, width, height."""
    H = fig._h_mm
    return fig.add_axes([x / W, (H - y_top - h) / H, w / W, h / H], **kw)


def text_mm(fig, x, y_top, s, **kw):
    """Figure text at mm position (y from the top)."""
    H = fig._h_mm; kw.setdefault("fontsize", FS); kw.setdefault("color", BLACK)
    return fig.text(x / W, 1 - y_top / H, s, **kw)


def letter(fig, x, y_top, s):
    """Panel letter, 8 pt bold, baseline at y_top (mm from the top)."""
    return text_mm(fig, x, y_top, s, fontsize=FS_LETTER, fontweight="bold", ha="left", va="baseline")


def facet(ax, s, dy_mm=1.3):
    """Region label above a small panel (black, 7 pt, left aligned with the axes)."""
    h = ax.get_position().height * ax.figure.get_figheight() / MM
    return ax.text(0.0, 1 + dy_mm / h, s, transform=ax.transAxes, ha="left", va="baseline", fontsize=FS, color=BLACK)


def num(v, nd=1, plus=False):
    """Number with a true minus sign (U+2212); optional '+' for positive values; '-0.0' printed as 0.0."""
    t = f"{abs(v):.{nd}f}"
    if float(t) == 0:
        return t
    return ("−" if v < 0 else ("+" if plus else "")) + t


def method_handle(m, marker=True):
    st = METH[m]
    return Line2D([], [], color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"] if marker else None, ms=st["ms"],
                  mfc="white" if st["open"] else st["color"], mec=st["color"], mew=MEW if st["open"] else 0.0)


def plot_method(ax, x, y, m, zorder=3, mfc=None, clip_on=True, line=True, marker=True):
    st = METH[m]
    if line:
        ax.plot(x, y, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=zorder, clip_on=clip_on)
    if marker:
        ax.plot(x, y, ls="none", marker=st["marker"], ms=st["ms"], mfc=(("white" if st["open"] else st["color"]) if mfc is None else mfc),
                mec=st["color"], mew=MEW if (st["open"] or mfc == "white") else 0.0, zorder=zorder + 0.5, clip_on=clip_on)


def zero_line(ax, vertical=False):
    (ax.axvline if vertical else ax.axhline)(0, color=ZERO, lw=LW_REF, zorder=1)


def export(fig, stem):
    """pdf, svg, png (600 dpi) on the fixed canvas (no tight bbox), so the page size equals the layout size."""
    OUT.mkdir(parents=True, exist_ok=True); out = []
    for ext in ("pdf", "svg", "png"):
        p = OUT / f"{stem}.{ext}"; fig.savefig(p, dpi=600 if ext == "png" else None, facecolor="white"); out.append(p)
    return out


# ---------------------------------------------------------------- QA
def _drawn_tick_labels(ax):
    out = []
    for axis in (ax.xaxis, ax.yaxis):
        try:
            ticks = axis._update_ticks()                # ticks that are actually drawn (inside the view interval)
        except Exception:
            ticks = axis.get_major_ticks()
        for t in ticks:
            for lab in (t.label1, t.label2):
                if lab.get_visible() and lab.get_text().strip():
                    out.append(lab)
    return out


def _texts(fig):
    """All drawn, non-empty texts with an owner tag: ('tick', ax) | ('axes', ax) | ('figure', None) | ('legend', ax or None)."""
    items = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        if ax.axison:                                   # axis('off') hides ticks and axis labels
            for t in _drawn_tick_labels(ax):
                items.append((t, "tick", ax))
            for t in (ax.xaxis.label, ax.yaxis.label):
                if t.get_visible() and t.get_text().strip():
                    items.append((t, "label", ax))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                items.append((t, "axes", ax))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                items.append((t, "legend", ax))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            items.append((t, "figure", None))
    for leg in fig.legends:
        for t in leg.get_texts():
            items.append((t, "legend", None))
    return items


def _seg_hits_box(p0, p1, b):
    """Does segment p0-p1 (display px) intersect box b (x0, y0, x1, y1)? Liang-Barsky clipping."""
    x0, y0 = p0; dx, dy = p1[0] - p0[0], p1[1] - p0[1]; t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - b[0]), (dx, b[2] - x0), (-dy, y0 - b[1]), (dy, b[3] - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


def qa(fig, min_pt=6.0, edge_mm=0.3, ignore_pairs=()):
    """Return a list of problems (empty = clean). Pixel units are converted to mm for the report."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); px2mm = 25.4 / fig.dpi
    Wpx, Hpx = fig.bbox.width, fig.bbox.height
    items = _texts(fig); probs = []
    boxes = []
    for t, kind, ax in items:
        fs = t.get_fontsize()
        if fs < min_pt - 1e-6:
            probs.append(f"font {fs:.2f} pt < {min_pt}: '{t.get_text()}'")
        bb = t.get_window_extent(r)
        boxes.append((t, kind, ax, bb))
        m = edge_mm / px2mm
        if bb.x0 < m or bb.y0 < m or bb.x1 > Wpx - m or bb.y1 > Hpx - m:
            probs.append(f"outside canvas: '{t.get_text()}' ({bb.x0 * px2mm:.1f}, {bb.y0 * px2mm:.1f}, {bb.x1 * px2mm:.1f}, {bb.y1 * px2mm:.1f}) mm")
    # text-text overlaps
    for (t1, k1, a1, b1), (t2, k2, a2, b2) in itertools.combinations(boxes, 2):
        if (t1.get_text(), t2.get_text()) in ignore_pairs or (t2.get_text(), t1.get_text()) in ignore_pairs:
            continue
        rot1, rot2 = t1.get_rotation() % 180, t2.get_rotation() % 180
        if rot1 not in (0, 90) and rot1 == rot2 and k1 == k2 == "tick" and a1 is a2:
            # parallel rotated tick labels: perpendicular distance between baselines must exceed the text height
            th = np.deg2rad(rot1); p1 = np.array(t1.get_transform().transform(t1.get_position()))
            p2 = np.array(t2.get_transform().transform(t2.get_position())); d = p2 - p1
            perp = abs(-d[0] * np.sin(th) + d[1] * np.cos(th)); h = t1.get_fontsize() / 72 * fig.dpi * 1.0
            if perp < h:
                probs.append(f"rotated tick labels too close: '{t1.get_text()}' / '{t2.get_text()}' ({perp * px2mm:.2f} mm < {h * px2mm:.2f} mm)")
            continue
        if b1.overlaps(b2):
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0); iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 0.5 and iy > 0.5:
                probs.append(f"text overlap: '{t1.get_text()}' [{k1}] / '{t2.get_text()}' [{k2}] ({ix * px2mm:.2f} x {iy * px2mm:.2f} mm)")
    # text over a plotting area it does not belong to
    plot_axes = [a for a in fig.axes if a.get_visible() and a.axison]
    for t, kind, ax, bb in boxes:
        for a in plot_axes:
            if a is ax:
                continue
            ab = a.get_window_extent(r)
            ix = min(bb.x1, ab.x1) - max(bb.x0, ab.x0); iy = min(bb.y1, ab.y1) - max(bb.y0, ab.y0)
            if ix > 0.5 and iy > 0.5:
                probs.append(f"text inside a foreign axes: '{t.get_text()}' [{kind}] ({ix * px2mm:.2f} x {iy * px2mm:.2f} mm)")
    # in-axes labels touching lines or markers of the same axes
    for t, kind, ax, bb in boxes:
        if kind != "axes" or getattr(t, "_qa_skip", False):
            continue
        b = (bb.x0, bb.y0, bb.x1, bb.y1)
        for ln in ax.get_lines():
            if not ln.get_visible() or getattr(ln, "_qa_skip", False):
                continue
            xy = ln.get_transform().transform(np.column_stack(ln.get_data()).astype(float))
            xy = xy[np.isfinite(xy).all(axis=1)]
            if len(xy) == 0:
                continue
            hit = False
            if ln.get_linestyle() not in ("None", "none", "", " ") and len(xy) > 1:
                hit = any(_seg_hits_box(xy[i], xy[i + 1], b) for i in range(len(xy) - 1))
            if ln.get_marker() not in (None, "None", "none", "", " "):
                rad = ln.get_markersize() / 72 * fig.dpi / 2
                hit = hit or bool(((xy[:, 0] > b[0] - rad) & (xy[:, 0] < b[2] + rad) & (xy[:, 1] > b[1] - rad) & (xy[:, 1] < b[3] + rad)).any())
            if hit:
                probs.append(f"label on data: '{t.get_text()}' touches a line/marker ({ln.get_color()})")
    return probs


def qa_report(fig, name, **kw):
    p = qa(fig, **kw)
    sizes = sorted({round(t.get_fontsize(), 2) for t, _, _ in _texts(fig)})
    print(f"[QA {name}] font sizes used (pt): {sizes}; problems: {len(p)}")
    for x in p:
        print("   -", x)
    return p


def ink_margins_mm(fig):
    """Distance (mm) from the tight ink box to each canvas edge."""
    fig.canvas.draw(); bb = fig.get_tightbbox(fig.canvas.get_renderer()); k = 25.4
    return dict(left=round(bb.x0 * k, 2), right=round(W - bb.x1 * k, 2), bottom=round(bb.y0 * k, 2), top=round(fig._h_mm - bb.y1 * k, 2))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return path
