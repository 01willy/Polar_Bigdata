"""Shared helpers for the v11 supplementary figures S1 (Stefan), S2 (structures), S3 (SHAP) and S10 (label ladder).

Copied from paper_v10/common.py (2026-10-16) and restyled for v11 (docs/PAPER_PLAN_v11_2026-10-16.md section 4; design/style_tokens_v4.json
color.paper_v13 read through src/polar/style_v5.py with POLAR_PALETTE=paper13): white background, dark-grey text #262626 (no pure black),
#404040 axes 0.5 pt, light #E8E8E8 grid lines where they help, data lines 0.8-1.0 pt, markers 3-4 pt, methods / regions (plus marker
shape) / products in their v11 colours, FreeSans 7 pt (ticks 6.5 pt, nothing below 6 pt), panel letters 8 pt bold.
Named si13_common (not common) so that other v11 scripts in this folder can keep their own helper modules.
This module never writes into outputs/figures/paper/v7, v10, v10_si or deck/assets.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if os.environ.get("POLAR_PALETTE") != "paper13":
    raise SystemExit("run with POLAR_PALETTE=paper13 (v11 paper palette in polar.style_v5)")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "scripts" / "4_visualization", ROOT / "scripts" / "4_visualization" / "paper_v3"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import polar.style_v5 as S  # noqa: E402

OUT_MAIN = ROOT / "outputs" / "figures" / "paper" / "v15"
OUT_SI = ROOT / "outputs" / "figures" / "paper" / "v15_si"
SRC_MAIN = OUT_MAIN / "source_data"
FORBIDDEN = [ROOT / "outputs" / "figures" / "paper" / d for d in ("v7", "v10", "v10_si", "v11", "v11_si", "v12", "v12_si", "v13", "v13_si")] + [ROOT / "deck" / "assets"]
MM = 1.0 / 25.4
PT_MM = 25.4 / 72.0                       # mm per pt

# ------------------------------------------------------------------ palette (tokens color.paper_v13; marker shapes as in color.paper_v10.markers)
PP = S.C["paper_v13"]
MK = S.C["paper_v10"]["markers"]                  # shapes only (paper_v13 has no marker table): regions o / s / ^, products D / v / s / s / o
METHOD = dict(PP["methods"])
METHOD_NAME = {"R1": "Physics + residual ML", "P1": "Stefan (recalibrated)", "P0": "Stefan (source coeff.)", "D0": "Direct ML",
               "F1": "Physics-input ML", "D1": "Physics pseudo-labels"}
PRODUCT = {"wei": ("Wei 2026", PP["products"]["Wei"], MK["Wei"]), "liu": ("Liu 2024", PP["products"]["Liu"], MK["Liu"]),
           "cci4y": ("ESA CCI v4", PP["products"]["CCI"], MK["CCI"]), "cci5y": ("ESA CCI v5", PP["products"]["CCI5"], MK["CCI5"]),
           "aalto": ("Aalto 2018", PP["products"]["Aalto"], MK["Aalto"])}
REGION = {"Alaska": ("Alaska", PP["regions"]["Alaska"], MK["Alaska"]),
          "Lena": ("Lena Delta", PP["regions"]["Lena Delta"], MK["Lena Delta"]),
          "Canada": ("Canada (NW)", PP["regions"]["Canada"], MK["Canada"])}
REGION_MS = {"Alaska": 3.4, "Lena": 3.1, "Canada": 3.8}  # visually equal marker areas (pt)
REGIONS = ["Alaska", "Lena", "Canada"]
TEXT = PP["text"]; AXIS = PP["axis"]; GRID = PP["grid"]; ZERO = PP["zero_line"]; WHITE = "#FFFFFF"
TEXT_TICK = PP.get("text_tick", TEXT); AUX = PP.get("text_aux", "#595959")
EDGE_DARK = "#4D4D4D"                    # marker edge for light product markers (keeps open markers visible on white)
DASH = (0, (3.2, 1.6)); DOT = (0, (1.0, 1.4))

# sizes (pt)
FS = 7.0; FS_TICK = 6.5; FS_SMALL = 6.5; FS_LETTER = 8.0
LW_AX = 0.6; LW_TICK = 0.6; TICK_LEN = 2.2
LW_MAIN = 0.9; LW_AUX = 0.7; LW_REF = 0.5; LW_THIN = 0.45; LW_GRID = 0.0
MS = 3.4; MS_SQ = 3.1; MEW = 0.7


def setup():
    """matplotlib rc for paper figures (FreeSans 7 pt, dark-grey text, thin dark-grey axes, Unicode minus)."""
    S.setup(scale=1.0)
    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
        "axes.linewidth": LW_AX, "xtick.major.width": LW_TICK, "ytick.major.width": LW_TICK, "xtick.minor.width": LW_TICK * 0.8,
        "ytick.minor.width": LW_TICK * 0.8, "xtick.major.size": TICK_LEN, "ytick.major.size": TICK_LEN, "xtick.minor.size": TICK_LEN * 0.6,
        "ytick.minor.size": TICK_LEN * 0.6, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.0,
        "axes.edgecolor": AXIS, "axes.labelcolor": TEXT, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": TEXT_TICK, "ytick.labelcolor": TEXT_TICK,
        "text.color": TEXT, "axes.unicode_minus": True, "lines.linewidth": LW_MAIN, "lines.markersize": MS, "lines.markeredgewidth": MEW,
        "axes.grid": False, "hatch.color": "#B3B3B3",
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.0, "legend.borderpad": 0.0,
        "legend.labelspacing": 0.35, "legend.columnspacing": 1.2, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "hatch.linewidth": 0.3, "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150, "image.interpolation": "antialiased",
    })


def figure(w_mm, h_mm):
    S.PANELS.clear()
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def ax_mm(fig, x, y_top, w, h, **kw):
    """Axes from the top-left origin in mm."""
    Wf, Hf = fig.get_figwidth() / MM, fig.get_figheight() / MM
    return fig.add_axes([x / Wf, 1 - (y_top + h) / Hf, w / Wf, h / Hf], **kw)


def text_mm(fig, x, y_top, s, **kw):
    Wf, Hf = fig.get_figwidth() / MM, fig.get_figheight() / MM
    kw.setdefault("color", TEXT)
    return fig.text(x / Wf, 1 - y_top / Hf, s, **kw)


def letter(fig, x, y_top, s):
    """Panel letter, 8 pt bold, top-left anchored at (x, y_top) mm."""
    return text_mm(fig, x, y_top, s, fontsize=FS_LETTER, fontweight="bold", ha="left", va="top")


def grid(ax, axis="y"):
    """v12: no background grid lines. Kept as a no-op for the callers."""
    ax.set_axisbelow(True)


def tint(hex_, f):
    """Mix a colour with white: f = share of the colour (0 -> white, 1 -> colour). Solid colour, no transparency."""
    c = np.array(to_rgb(hex_)); return to_hex(1 - f * (1 - c))


def _rel_lum(c):
    c = np.array(to_rgb(c)); c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def on_colour(bg):
    """Text colour (dark grey or white) with the higher WCAG contrast ratio on the background colour bg."""
    lb = _rel_lum(bg); cw = (1.05) / (lb + 0.05); ct = (max(lb, _rel_lum(TEXT)) + 0.05) / (min(lb, _rel_lum(TEXT)) + 0.05)
    return WHITE if cw > ct else TEXT


def clean_axes(ax, left=True, bottom=True):
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)
    for k in ("left", "bottom"):
        ax.spines[k].set_linewidth(LW_AX); ax.spines[k].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=TEXT_TICK, width=LW_TICK, length=TICK_LEN, labelsize=FS_TICK)


def darker(hex_, f=0.62):
    c = np.array(to_rgb(hex_)) * f
    return to_hex(c)


def product_style(key):
    """(label, face colour, edge colour, marker) for a product: face = token grey, edge darker so open markers stay visible."""
    name, col, mk = PRODUCT[key]
    edge = col if to_rgb(col)[0] < 0.45 else darker(col, 0.62)
    return name, col, edge, mk


def export(fig, stem, outdir, exts=("pdf", "svg", "png"), dpi=600):
    """Fixed canvas (exact figure size), white background. Refuses to write into v7 or deck/assets."""
    outdir = Path(outdir)
    for f in FORBIDDEN:
        if str(outdir.resolve()).startswith(str(f.resolve())):
            raise RuntimeError(f"refusing to write into {f}")
    outdir.mkdir(parents=True, exist_ok=True); out = []
    for ext in exts:
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, facecolor="white")
        if ext == "png":                                 # flatten to RGB (no alpha channel), keep the dpi tag
            from PIL import Image
            with Image.open(p) as im:
                rgb = im.convert("RGB")
            rgb.save(p, dpi=(dpi, dpi), optimize=True)
        out.append(p)
    return out


# ------------------------------------------------------------------ QA: text size, colour, bounds, overlaps (text-text, text-ink)
def _drawn_tick_label_ids(fig):
    alltick, drawn = set(), set()
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                alltick.add(id(t.label1)); alltick.add(id(t.label2))
            if not a.get_visible() or not a.axison or not axis.get_visible():
                continue
            try:
                ticks = axis._update_ticks()
            except Exception:
                ticks = []
            for t in ticks:
                if t.label1.get_visible():
                    drawn.add(id(t.label1))
                if t.label2.get_visible():
                    drawn.add(id(t.label2))
    return alltick, drawn


def _visible(artist):
    a = artist
    while a is not None:
        if hasattr(a, "get_visible") and not a.get_visible():
            return False
        a = getattr(a, "axes", None) if a is not getattr(a, "axes", None) else None
    return True


def collect_texts(fig):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    alltick, drawn = _drawn_tick_label_ids(fig)
    out = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s or not s.strip() or not t.get_visible():
            continue
        if t.axes is not None and (not t.axes.get_visible()):
            continue
        if id(t) in alltick and id(t) not in drawn:
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        kind = "tick" if id(t) in alltick else "text"
        out.append(dict(t=t, bb=bb, kind=kind, s=s.replace("\n", " | ")[:60], fs=float(t.get_fontsize()), color=to_hex(t.get_color())))
    return out, r


def _ink_points(fig, r, skip=(), axes=None):
    """Display coordinates of drawn data ink: Line2D vertices (lines sampled), markers (with radius), scatter offsets, bar rectangles."""
    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection, LineCollection
    from matplotlib.patches import Rectangle
    pts = []          # (x, y, radius_px)
    rects = []        # bboxes
    for a in (fig.axes if axes is None else axes):
        if not a.get_visible():
            continue
        leg = a.get_legend()
        legend_children = set(id(c) for c in (leg.findobj() if leg is not None else []))
        for ln in a.lines:
            if id(ln) in legend_children or not ln.get_visible() or ln in skip:
                continue
            xy = ln.get_xydata()
            if xy is None or len(xy) == 0:
                continue
            try:
                d = ln.get_transform().transform(np.asarray(xy, float))
            except Exception:
                continue
            d = d[np.isfinite(d).all(axis=1)]
            if not len(d):
                continue
            if ln.get_clip_on():
                bb = a.get_window_extent(r)
                d = d[(d[:, 0] >= bb.x0 - 1) & (d[:, 0] <= bb.x1 + 1) & (d[:, 1] >= bb.y0 - 1) & (d[:, 1] <= bb.y1 + 1)]
            mk = ln.get_marker()
            if mk not in (None, "None", "none", "", " ") and ln.get_markersize() > 0:
                rad = ln.get_markersize() / 2 * fig.dpi / 72.0
                pts += [(x, y, rad) for x, y in d]
            if ln.get_linestyle() not in ("None", "none", "", " ") and len(d) > 1:
                lw = ln.get_linewidth() / 2 * fig.dpi / 72.0
                for (x0, y0), (x1, y1) in zip(d[:-1], d[1:]):
                    n = int(max(2, np.hypot(x1 - x0, y1 - y0) / 1.5))
                    for k in np.linspace(0, 1, n):
                        pts.append((x0 + k * (x1 - x0), y0 + k * (y1 - y0), lw))
        for c in a.collections:
            if id(c) in legend_children or not c.get_visible():
                continue
            if isinstance(c, PathCollection):
                off = c.get_offsets()
                if off is None or len(off) == 0:
                    continue
                d = c.get_offset_transform().transform(np.asarray(off, float))
                sz = c.get_sizes(); rad = (np.sqrt(sz[0]) if len(sz) else 3.0) / 2 * fig.dpi / 72.0
                pts += [(x, y, rad) for x, y in d if np.isfinite(x) and np.isfinite(y)]
            elif isinstance(c, LineCollection):
                for seg in c.get_segments():
                    d = c.get_transform().transform(np.asarray(seg, float))
                    for (x0, y0), (x1, y1) in zip(d[:-1], d[1:]):
                        n = int(max(2, np.hypot(x1 - x0, y1 - y0) / 1.5))
                        for k in np.linspace(0, 1, n):
                            pts.append((x0 + k * (x1 - x0), y0 + k * (y1 - y0), 0.5))
        abb = a.get_window_extent(r)
        for p in a.patches:
            if isinstance(p, Rectangle) and p.get_visible() and id(p) not in legend_children and p.get_width() != 0:
                pb = p.get_window_extent(r)
                if p.get_clip_on():                       # bars are clipped by the axes (log-scale bars start at -inf)
                    x0, x1 = max(pb.x0, abb.x0), min(pb.x1, abb.x1); y0, y1 = max(pb.y0, abb.y0), min(pb.y1, abb.y1)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    from matplotlib.transforms import Bbox
                    pb = Bbox.from_extents(x0, y0, x1, y1)
                rects.append(pb)
    return pts, rects


def qa_figure(fig, min_pt=6.0, margin_mm=0.3, text_ink=True, ink_skip=(), allow_overlap=(), ignore_ink_for=(), extra_boxes=(), ink_axes=None):
    """Return a dict: min font size, texts below min_pt, non-black texts, texts outside the canvas (margin_mm), overlapping text pairs,
    texts touching data ink. allow_overlap: substrings of text pairs to ignore. ignore_ink_for: substrings of texts placed on maps on purpose."""
    T, r = collect_texts(fig)
    W, H = fig.bbox.width, fig.bbox.height; m = margin_mm / 25.4 * fig.dpi
    rep = dict(n_texts=len(T), min_font_pt=min(t["fs"] for t in T) if T else None, small=[], color=[], outside=[], overlaps=[], text_ink=[])
    for t in T:
        if t["fs"] < min_pt - 1e-6:
            rep["small"].append((t["s"], t["fs"]))
        if t["color"].lower() not in (TEXT.lower(), WHITE.lower(), TEXT_TICK.lower(), AUX.lower()):   # v13: black labels, grey ticks and notes
            rep["color"].append((t["s"], t["color"]))
        bb = t["bb"]
        if bb.x0 < m or bb.y0 < m or bb.x1 > W - m or bb.y1 > H - m:
            rep["outside"].append((t["s"], [round(v / fig.dpi * 25.4, 2) for v in bb.extents]))
    boxes = [(t["s"], t["bb"].shrunk(0.98, 0.9)) for t in T] + [(n, b) for n, b in extra_boxes]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1:
                pair = (boxes[i][0], boxes[j][0])
                if any(s in pair[0] or s in pair[1] for s in allow_overlap):
                    continue
                rep["overlaps"].append(pair)
    if text_ink:
        pts, rects = _ink_points(fig, r, skip=ink_skip, axes=ink_axes)
        P = np.array(pts) if pts else np.zeros((0, 3))
        for t in T:
            if t["kind"] == "tick" or any(s in t["s"] for s in ignore_ink_for):
                continue
            bb = t["bb"].shrunk(0.96, 0.82)
            if len(P):
                hit = (P[:, 0] + P[:, 2] > bb.x0) & (P[:, 0] - P[:, 2] < bb.x1) & (P[:, 1] + P[:, 2] > bb.y0) & (P[:, 1] - P[:, 2] < bb.y1)
                if hit.any():
                    rep["text_ink"].append((t["s"], int(hit.sum())))
            for rb in rects:
                if rb.x0 < bb.x1 and bb.x0 < rb.x1 and rb.y0 < bb.y1 and bb.y0 < rb.y1:
                    rep["text_ink"].append((t["s"], "bar")); break
    rep["ok"] = not (rep["small"] or rep["color"] or rep["outside"] or rep["overlaps"] or rep["text_ink"])
    return rep


def print_qa(name, rep):
    print(f"[QA {name}] texts {rep['n_texts']}, min font {rep['min_font_pt']:.2f} pt, ok={rep['ok']}")
    for k in ("small", "color", "outside", "overlaps", "text_ink"):
        if rep[k]:
            print(f"   {k}: {rep[k][:12]}{' ...' if len(rep[k]) > 12 else ''}")


def pdf_page_mm(p):
    import re
    import subprocess
    try:
        o = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True, timeout=30).stdout
        w, h = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", o).groups()
        return [round(float(w) * 25.4 / 72.0, 1), round(float(h) * 25.4 / 72.0, 1)]
    except Exception:
        return None


def file_info(paths):
    return {Path(p).name: dict(path=str(Path(p).relative_to(ROOT)), bytes=Path(p).stat().st_size) for p in paths}


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    for f in FORBIDDEN:
        if str(path.resolve()).startswith(str(f.resolve())):
            raise RuntimeError(f"refusing to write into {f}")
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return path


# ------------------------------------------------------------------ supplementary figures: export (PDF + PNG) and the shared SI JSON
SI_JSON = OUT_SI / "SI_figures.json"


def export_si(fig, stem):
    return export(fig, stem, OUT_SI, exts=("pdf", "png"))


def update_si_json(key, entry):
    """Read-modify-write OUT_SI/SI_figures.json (one entry per supplementary figure, v11 numbering) under a file lock with an atomic
    replace, because other v11 SI scripts may update the same file."""
    import fcntl
    import tempfile
    OUT_SI.mkdir(parents=True, exist_ok=True)
    with open(Path(tempfile.gettempdir()) / "polar_paper_v14_si_figures.lock", "w") as lk:   # lock file kept out of the deliverable folder
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.loads(SI_JSON.read_text()) if SI_JSON.exists() else {}
        d[key] = entry
        d = dict(sorted(d.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 99))
        tmp = SI_JSON.with_suffix(".json.tmp"); write_json(tmp, d); os.replace(tmp, SI_JSON)
        fcntl.flock(lk, fcntl.LOCK_UN)


def label_year_ranges():
    """Label cell-years used in the analyses (labels_cell_year_v5.csv, in_forcing == 1): per region year range and counts."""
    import pandas as pd
    lab = pd.read_csv(ROOT / "data" / "processed" / "v5" / "labels_cell_year_v5.csv", usecols=["region", "year", "loc_id", "in_forcing"])
    lab = lab[lab.in_forcing == 1]
    return {r: dict(years=[int(g.year.min()), int(g.year.max())], n_cell_years=int(len(g)), n_cells=int(g.loc_id.nunique()))
            for r, g in lab.groupby("region")}
