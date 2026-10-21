"""Shared helpers for the v11 manuscript figures made by role C: Fig. 1 (data), Fig. 5 (products), Fig. 6 (product maps), Fig. 7 (yearly maps)
and Supplementary Figs S4-S9 (v11 numbering). Copied from scripts/4_visualization/paper_v10/common.py and edited for the v11 style.

Style (docs/PAPER_PLAN_v11_2026-10-16.md section 4, design/style_tokens_v4.json color.paper_v13, read through POLAR_PALETTE=paper13):
  white background, dark-grey text #262626 (no pure black), axes #404040 0.5 pt, grid #E8E8E8, FreeSans 6-7 pt (panel letters 8 pt bold),
  data lines 0.8-1.0 pt, markers 3-4 pt. Colour per series: methods (Physics + residual ML #D1603D, Stefan recalibrated #2E6DA4, Stefan source
  #8C9BAB, Direct ML #8E5B9E dashed), products (Wei 2026 #5B8C5A, Liu 2024 #B65A72, ESA CCI v4 #A07D4A, ESA CCI v5 #C4A869, Aalto #8273A6,
  Yi-Kimball #7E8F3C) and regions (Alaska #3B3B8F circle, Lena Delta #4A90C2 square, Canada #3A9384 triangle). Marker shapes are kept as a
  second cue so that no information is carried by colour alone.
Outputs go to outputs/figures/paper/v11 (main) and v11_si (supplementary). Writing into v7, v10, v10_si or deck/assets is refused.
Module name c11_* (not common.py) so that it cannot collide with helpers of the other v11 figure scripts in the same folder.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLAR_PALETTE", "paper13")
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

if S.PALETTE != "paper13":
    raise RuntimeError("run with POLAR_PALETTE=paper13 (style_v5 palette is %r)" % S.PALETTE)

OUT_MAIN = ROOT / "outputs" / "figures" / "paper" / "v15"
OUT_SI = ROOT / "outputs" / "figures" / "paper" / "v15_si"
SRC_MAIN = OUT_MAIN / "source_data"
_P = ROOT / "outputs" / "figures" / "paper"
FORBIDDEN = [_P / "v7", _P / "v10", _P / "v10_si", _P / "v11", _P / "v11_si", _P / "v12", _P / "v12_si", _P / "v13", _P / "v13_si", _P / "v14", _P / "v14_si", ROOT / "deck" / "assets"]
MM = 1.0 / 25.4
PT_MM = 25.4 / 72.0                       # mm per pt

# ------------------------------------------------------------------ palette (tokens color.paper_v13)
PP = S.C["paper_v13"]
METHOD = dict(PP["methods"])
METHOD_NAME = {"R1": "Physics + residual ML", "P1": "Stefan (recalibrated)", "P0": "Stefan (source coeff.)", "D0": "Direct ML",
               "F1": "Physics-input ML", "D1": "Physics pseudo-labels"}
MARKER = {"Alaska": "o", "Lena": "s", "Canada": "^", "wei": "D", "liu": "v", "cci4y": "s", "cci5y": "s", "aalto": "o", "yk": "X"}
PRODUCT = {"wei": ("Wei 2026", PP["products"]["Wei"], MARKER["wei"]), "liu": ("Liu 2024", PP["products"]["Liu"], MARKER["liu"]),
           "cci4y": ("ESA CCI v4", PP["products"]["CCI"], MARKER["cci4y"]), "cci5y": ("ESA CCI v5", PP["products"]["CCI5"], MARKER["cci5y"]),
           "aalto": ("Aalto 2018", PP["products"]["Aalto"], MARKER["aalto"]), "yk": ("Yi–Kimball 2019", PP["products"]["YK"], MARKER["yk"])}
REGION = {"Alaska": ("Alaska", PP["regions"]["Alaska"], MARKER["Alaska"]),
          "Lena": ("Lena Delta", PP["regions"]["Lena Delta"], MARKER["Lena"]),
          "Canada": ("Canada (NW)", PP["regions"]["Canada"], MARKER["Canada"])}
REGIONS = ["Alaska", "Lena", "Canada"]
TEXT = PP["text"]; AXIS = PP["axis"]; GRID = PP["grid"]; ZERO = PP["zero_line"]
TEXT_TICK = PP.get("text_tick", TEXT); AUX = PP.get("text_aux", "#595959")   # v13 text hierarchy: labels black, ticks #333333, notes #595959
DARK = "#595959"                          # neutral series (inputs, static covariates) and neutral legend swatches
LIGHT = "#D9D9D9"                         # neutral light swatch
DASH = (0, (3.2, 1.6)); DOT = (0, (1.0, 1.4))

# sizes (pt)
FS = 7.0; FS_TICK = 6.5; FS_SMALL = 6.5; FS_LETTER = 8.0      # v13: labels 7 pt, ticks and notes 6.5 pt
LW_AX = 0.6; LW_TICK = 0.6; TICK_LEN = 2.2
LW_MAIN = 0.9; LW_AUX = 0.7; LW_REF = 0.5; LW_THIN = 0.45
MS = 3.4


def setup():
    """matplotlib rc for the v13 paper figures (FreeSans 7 / 6.5 pt, black labels, #333333 ticks, 0.6 pt axes, no grid)."""
    S.setup(scale=1.0)
    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
        "axes.linewidth": LW_AX, "xtick.major.width": LW_TICK, "ytick.major.width": LW_TICK, "xtick.minor.width": LW_TICK * 0.8,
        "ytick.minor.width": LW_TICK * 0.8, "xtick.major.size": TICK_LEN, "ytick.major.size": TICK_LEN, "xtick.minor.size": TICK_LEN * 0.6,
        "ytick.minor.size": TICK_LEN * 0.6, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.0,
        "axes.edgecolor": AXIS, "axes.labelcolor": TEXT, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": TEXT_TICK, "ytick.labelcolor": TEXT_TICK,
        "text.color": TEXT, "axes.unicode_minus": True, "lines.linewidth": LW_MAIN, "lines.markersize": MS, "lines.markeredgewidth": 0.7, "axes.grid": False,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.0, "legend.borderpad": 0.0,
        "legend.labelspacing": 0.35, "legend.columnspacing": 1.2, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "hatch.linewidth": 0.3, "hatch.color": DARK, "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150, "image.interpolation": "antialiased",
        "axes.grid": False,
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
    ax.tick_params(colors=AXIS, labelcolor=TEXT_TICK, width=LW_TICK, length=TICK_LEN, labelsize=FS_TICK)


def darker(hex_, f=0.62):
    c = np.array(to_rgb(hex_)) * f
    return to_hex(c)


def tint(hex_, f):
    """Blend towards white: f = 1 keeps the colour, f = 0 is white."""
    c = np.array(to_rgb(hex_)); return to_hex(1.0 - f * (1.0 - c))


def product_style(key):
    """(label, face colour, edge colour, marker) for a product: face = token colour, edge slightly darker so open markers stay crisp."""
    name, col, mk = PRODUCT[key]
    return name, col, darker(col, 0.85), mk


def export(fig, stem, outdir, exts=("pdf", "svg", "png"), dpi=600):
    """Fixed canvas (exact figure size), white background. Refuses to write into v7, v10, v10_si or deck/assets."""
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
        out.append(dict(t=t, bb=bb, kind=kind, s=s.replace("\n", " | ")[:60], fs=float(t.get_fontsize()), color=to_hex(t.get_color()),
                        font=t.get_fontname()))
    return out, r


def _ink_points(fig, r, skip=(), axes=None):
    """Display coordinates of drawn data ink: Line2D vertices (lines sampled), markers (with radius), scatter offsets, bar rectangles."""
    from matplotlib.lines import Line2D  # noqa: F401
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


def qa_figure(fig, min_pt=6.0, margin_mm=0.3, text_ink=True, ink_skip=(), allow_overlap=(), ignore_ink_for=(), extra_boxes=(), ink_axes=None,
              allow_colors=()):
    """Return a dict: min font size, texts below min_pt, texts not in the v11 text colour, non-FreeSans texts, texts outside the canvas
    (margin_mm), overlapping text pairs, texts touching data ink. allow_overlap: substrings of text pairs to ignore. ignore_ink_for: substrings of
    texts placed on maps on purpose. allow_colors: extra hex colours accepted for text (e.g. white numbers on dark bars)."""
    T, r = collect_texts(fig)
    W, H = fig.bbox.width, fig.bbox.height; m = margin_mm / 25.4 * fig.dpi
    ok_col = {to_hex(TEXT), to_hex(TEXT_TICK), to_hex(AUX)} | {to_hex(c) for c in allow_colors}
    rep = dict(n_texts=len(T), min_font_pt=min(t["fs"] for t in T) if T else None, font_sizes_pt=sorted({round(t["fs"], 2) for t in T}),
               fonts=sorted({t["font"] for t in T}), small=[], color=[], font=[], outside=[], overlaps=[], text_ink=[])
    for t in T:
        if t["fs"] < min_pt - 1e-6:
            rep["small"].append((t["s"], t["fs"]))
        if t["color"] not in ok_col:
            rep["color"].append((t["s"], t["color"]))
        if t["font"] != "FreeSans":
            rep["font"].append((t["s"], t["font"]))
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
    rep["ok"] = not (rep["small"] or rep["color"] or rep["font"] or rep["outside"] or rep["overlaps"] or rep["text_ink"])
    return rep


def print_qa(name, rep):
    print(f"[QA {name}] texts {rep['n_texts']}, sizes {rep['font_sizes_pt']}, fonts {rep['fonts']}, ok={rep['ok']}")
    for k in ("small", "color", "font", "outside", "overlaps", "text_ink"):
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


def pdf_fonts(p):
    """Font names embedded in a PDF (pdffonts)."""
    import subprocess
    try:
        o = subprocess.run(["pdffonts", str(p)], capture_output=True, text=True, timeout=30).stdout.splitlines()[2:]
        return sorted({ln.split()[0] for ln in o if ln.strip()})
    except Exception:
        return None


def file_info(paths):
    return {Path(p).name: dict(path=str(Path(p).relative_to(ROOT)), bytes=Path(p).stat().st_size) for p in paths}


def _check_path(path):
    for f in FORBIDDEN:
        if str(Path(path).resolve()).startswith(str(f.resolve())):
            raise RuntimeError(f"refusing to write into {f}")


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); _check_path(path)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return path


# ------------------------------------------------------------------ supplementary figures: export (PDF + PNG) and the shared SI JSON
SI_JSON = OUT_SI / "SI_figures.json"


def export_si(fig, stem):
    return export(fig, stem, OUT_SI, exts=("pdf", "png"))


def update_si_json(key, entry):
    """Read-modify-write OUT_SI/SI_figures.json under an exclusive file lock (other v11 SI scripts write other keys of the same file)."""
    OUT_SI.mkdir(parents=True, exist_ok=True); _check_path(SI_JSON)
    lock = OUT_MAIN / "_cache" / "SI_figures.lock"; lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.loads(SI_JSON.read_text()) if SI_JSON.exists() and SI_JSON.stat().st_size else {}
        d[key] = entry
        d = dict(sorted(d.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 99))
        tmp = SI_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        os.replace(tmp, SI_JSON)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return SI_JSON


def label_year_ranges():
    """Label cell-years used in the analyses (labels_cell_year_v5.csv, in_forcing == 1): per region year range and counts."""
    import pandas as pd
    lab = pd.read_csv(ROOT / "data" / "processed" / "v5" / "labels_cell_year_v5.csv", usecols=["region", "year", "loc_id", "in_forcing"])
    lab = lab[lab.in_forcing == 1]
    return {r: dict(years=[int(g.year.min()), int(g.year.max())], n_cell_years=int(len(g)), n_cells=int(g.loc_id.nunique()))
            for r, g in lab.groupby("region")}
