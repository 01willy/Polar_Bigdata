"""Shared helpers for the v10 manuscript figures (Fig. 5, Fig. 6, Supplementary Figs S1-S11).

Style: Nature-family figure grammar (white background, black text, thin black axes, 7 pt FreeSans, ticks >= 6 pt,
panel letters 8 pt bold). Colour is used only to separate our methods (design/style_tokens_v4.json color.paper_v10):
Physics + residual ML vermillion, Stefan (recalibrated) blue, Direct ML black; products and regions are greys with marker shapes.
This module never writes into outputs/figures/paper/v7 or deck/assets.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLAR_PALETTE", "paper")
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

OUT_MAIN = ROOT / "outputs" / "figures" / "paper" / "v10"
OUT_SI = ROOT / "outputs" / "figures" / "paper" / "v10_si"
SRC_MAIN = OUT_MAIN / "source_data"
FORBIDDEN = [ROOT / "outputs" / "figures" / "paper" / "v7", ROOT / "deck" / "assets"]
MM = 1.0 / 25.4
PT_MM = 25.4 / 72.0                       # mm per pt

# ------------------------------------------------------------------ palette (tokens color.paper_v10)
PP = S.C["paper_v10"]
METHOD = dict(PP["methods"])
METHOD_NAME = {"R1": "Physics + residual ML", "P1": "Stefan (recalibrated)", "P0": "Stefan (source coeff.)", "D0": "Direct ML",
               "F1": "Physics-input ML", "D1": "Physics pseudo-labels"}
PRODUCT = {"wei": ("Wei 2026", PP["products"]["Wei"], PP["markers"]["Wei"]), "liu": ("Liu 2024", PP["products"]["Liu"], PP["markers"]["Liu"]),
           "cci4y": ("ESA CCI v4", PP["products"]["CCI"], PP["markers"]["CCI"]), "cci5y": ("ESA CCI v5", PP["products"]["CCI5"], PP["markers"]["CCI5"]),
           "aalto": ("Aalto 2018", PP["products"]["Aalto"], PP["markers"]["Aalto"])}
REGION = {"Alaska": ("Alaska", PP["regions"]["Alaska"], PP["markers"]["Alaska"]),
          "Lena": ("Lena Delta", PP["regions"]["Lena Delta"], PP["markers"]["Lena Delta"]),
          "Canada": ("Canada (NW)", PP["regions"]["Canada"], PP["markers"]["Canada"])}
REGIONS = ["Alaska", "Lena", "Canada"]
TEXT = PP["text"]; AXIS = PP["axis"]; GRID = PP["grid"]; ZERO = PP["zero_line"]
EDGE_DARK = "#4D4D4D"                    # marker edge for light-grey product markers (keeps open markers visible on white)
DASH = (0, (3.2, 1.6)); DOT = (0, (1.0, 1.4))

# sizes (pt)
FS = 7.0; FS_TICK = 6.5; FS_SMALL = 6.0; FS_LETTER = 8.0
LW_AX = 0.5; LW_TICK = 0.5; TICK_LEN = 2.0
LW_MAIN = 1.0; LW_AUX = 0.6; LW_REF = 0.5; LW_THIN = 0.4


def setup():
    """matplotlib rc for paper figures (FreeSans 7 pt, black text, thin black axes, Unicode minus)."""
    S.setup(scale=1.0)
    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
        "axes.linewidth": LW_AX, "xtick.major.width": LW_TICK, "ytick.major.width": LW_TICK, "xtick.minor.width": LW_TICK * 0.8,
        "ytick.minor.width": LW_TICK * 0.8, "xtick.major.size": TICK_LEN, "ytick.major.size": TICK_LEN, "xtick.minor.size": TICK_LEN * 0.6,
        "ytick.minor.size": TICK_LEN * 0.6, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.0,
        "axes.edgecolor": AXIS, "axes.labelcolor": TEXT, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": TEXT, "ytick.labelcolor": TEXT,
        "text.color": TEXT, "axes.unicode_minus": True, "lines.linewidth": LW_MAIN, "lines.markersize": 3.2, "lines.markeredgewidth": 0.6,
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


def clean_axes(ax, left=True, bottom=True):
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)
    for k in ("left", "bottom"):
        ax.spines[k].set_linewidth(LW_AX); ax.spines[k].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=TEXT, width=LW_TICK, length=TICK_LEN, labelsize=FS_TICK)


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
        if t["color"] not in ("#000000",):
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
    """Read-modify-write OUT_SI/SI_figures.json (one entry per supplementary figure)."""
    d = json.loads(SI_JSON.read_text()) if SI_JSON.exists() else {}
    d[key] = entry
    d = dict(sorted(d.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 99))
    write_json(SI_JSON, d)


def label_year_ranges():
    """Label cell-years used in the analyses (labels_cell_year_v5.csv, in_forcing == 1): per region year range and counts."""
    import pandas as pd
    lab = pd.read_csv(ROOT / "data" / "processed" / "v5" / "labels_cell_year_v5.csv", usecols=["region", "year", "loc_id", "in_forcing"])
    lab = lab[lab.in_forcing == 1]
    return {r: dict(years=[int(g.year.min()), int(g.year.max())], n_cell_years=int(len(g)), n_cells=int(g.loc_id.nunique()))
            for r, g in lab.groupby("region")}
