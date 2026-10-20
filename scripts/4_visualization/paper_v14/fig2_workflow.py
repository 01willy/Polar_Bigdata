# -*- coding: utf-8 -*-
"""Paper v12 Fig. 2 (Scientific Reports main text): cell-year design and Stefan anchor, physics-residual model, validation designs and yearly outputs. 180 x 124 mm.
v12 (2026-10-18, docs/FIGURE_AUDIT_v12_2026-10-18.md): black text and connectors, filled arrowheads, no row titles or hairlines, one text size (7 pt);
the cell-year table, the schematic label-number curves, the model-structure glyphs and the single-cell product comparison of v11 were removed.
--- v11 layout notes (kept for reference) ---

Rows (letter + short header + hairline; element labels on one line per row; all numbers read from files)
  a  Cell-year design (Lena Delta labels)
     soil columns  vector schematic: observation day (thawed to the thaw front, still-frozen active layer, permafrost) and end of season
                   (thawed to the permafrost table); dimension lines for the thaw depth on the observation day and ALT; key thawed / frozen.
     TDD curve     daily cumulative TDD of the curve cell-year (linear between ERA5-Land month-end values, 0 on 1 January, as
                   polar.testbed_v5.with_forcing('year_obs')); open circle = TDD to the observation day (TDD_obs, model input), filled circle =
                   full-season TDD (maps), dashed = the cell's 2011-2024 mean TDD (multi-year input), light lines = the cell's other years 2011-2024.
     table         four cell-years: two Lena Delta cells x the coldest and warmest year 2011-2024 (mean full-season TDD over all Lena label cells);
                   cells = the two with the most label years among cells with dated observations in both years (ties: smaller id);
                   curve cell-year = the row with the smallest TDD_obs / full-season ratio (open circle in the table).
  b  Physics-residual model
     Stefan anchor  observed ALT against sqrt(TDD_obs), all Alaska labels (light points); line ALT = E sqrt(TDD_obs), E by least squares through the
                    origin (= data/processed/v5/maps/Alaska/meta.json E); five residuals highlighted (rule in the source data).
     flow           Alaska 2014 map model: climate (cmcrameri lajolla), terrain (grayC) and soil (bamako) input thumbnails -> bracket (all 23 inputs)
                    -> gradient boosting g(x) -> residual map (vik, symmetric); TDD -> Stefan E sqrt(TDD) -> Stefan map (oslo_r); plus node with
                    weight lambda -> ALT 2014 (oslo_r, accent frame). Training target (dashed, training only). Equation strip above the flow
                    (the hat over ALT is a drawn vector chevron; mathtext widehat renders an oversized wedge with FreeSans).
  c  Evaluation and outputs
     validation designs  Lena Delta label cells under random, spatial block 0.5 deg, year hold-out, block x year and region hold-out (folds as the
                         harness, polar.testbed_v5.make_folds k 5 seed 0; checked against data/processed/v5/runs/core_v6/preds); test dark,
                         training mid grey, excluded open grey ring; 50 km scale bar.
     label number, new region  schematic RMSE curves (no values, tagged 'Schematic') with one marker per design step read from runs/ladder_v6/ladder_metrics.csv.
     compared structures  Direct ML, Physics-input ML (Stefan and Kudryavtsev predictions as inputs), Physics pseudo-labels (training only),
                          Physics + residual ML, with the line keys of Figs 3 and 4.
     yearly ALT anomaly   Alaska 2012 and 2019: pred_<year> r1 minus mean_2011_2024 r1, vik +-15 cm.
     same cell, same year  one Lena Delta label cell-year (14568, 2013) on an ALT axis: observed (diamond), physics + residual ML spatial-block
                          out-of-fold value, Wei 2026, Liu 2024, ESA CCI v4 and v5 of the same cell and year; dotted line = observed value.
                          Choice: cell-years with all four products and TDD_obs >= 90 % of the full season; lower median of |out-of-fold error|.
Data  data/processed/v5/{labels_cell_year_v5,forcing_year,forcing_clim,static_v5}.csv (polar.testbed_v5.load_v5),
      data/processed/v5/maps/Alaska/{pred_2012,pred_2014,pred_2019,mean_2011_2024,forcing_2014_thumb}.csv.gz and meta.json,
      data/processed/map_alaska/alaska_grid_x25_v1.csv.gz, data/processed/v5/runs/core_v6/preds (fold check), runs/ladder_v6/ladder_metrics.csv.
Style docs/PAPER_PLAN_v11_2026-10-16.md section 4; design/style_tokens_v4.json color.paper_v13 via polar.style_v5 (POLAR_PALETTE=paper13);
      FreeSans 6-7 pt (#262626), letters 8 pt bold, equation 8 pt; connectors 0.6 pt with open chevrons; operation blocks only (light tint, #808080).
Run   cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v14/fig2_workflow.py
      (FIG2_CACHE=<dir> writes a 300 dpi draft and a raster cache there instead of the outputs)
Out   outputs/figures/paper/v14/Fig2_workflow.{pdf,svg,png} (180 x 160 mm, png 600 dpi), outputs/figures/paper/v14/source_data/Fig2_workflow.json
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
import calendar, datetime as dt, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
if os.environ.get("POLAR_PALETTE") != "paper13":
    raise SystemExit("run with POLAR_PALETTE=paper13 (v11 paper palette in polar.style_v5)")
import polar.style_v5 as S  # noqa: E402
import polar.testbed_v5 as TB  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm, to_rgb, to_hex  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Polygon  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter  # noqa: E402
import cartopy.io.shapereader as shpreader  # noqa: E402
import shapely  # noqa: E402
from cmcrameri import cm as cmc  # noqa: E402

PROC = ROOT / "data" / "processed"; V5 = PROC / "v5"; MAPS = V5 / "maps"
OUT = ROOT / "outputs" / "figures" / "paper" / "v14"; STEM = "Fig2_workflow"
CACHE = Path(os.environ["FIG2_CACHE"]) if os.environ.get("FIG2_CACHE") else None
SRC: dict = {}
P11 = S.C["paper_v13"]

# ================================================================ style (docs/PAPER_PLAN_v11_2026-10-16.md section 4)
W_MM, H_MM = 180.0, 132.0
FS = 7.0; FS_S = 6.5; FS_T = 6.5; FS_LETTER = 8.0; FS_EQ = 8.0      # v13: labels 7 pt, small labels and ticks 6.5 pt, letters and equation 8 pt
INK = P11["text"]                       # #000000 (tokens color.paper_v13)
AXIS = P11["axis"]                      # #000000
TICK = P11.get("text_tick", INK)        # #333333 tick labels
AUX = P11.get("text_aux", "#595959")    # #595959 notes
FOLD_HEX = list(P11.get("folds", ["#2A7BB5", "#6BB5DE", "#D8A24A", "#C98AAE", "#3A9E7F"]))
GRID = P11["grid"]                      # #E8E8E8
ZERO = S.NEUTRAL["zero_line"]           # #8C8C8C
ACCENT = S.METHOD_HEX["R1"]             # #D1603D proposed method
STEFAN = S.METHOD_HEX["P1"]             # #2E6DA4 Stefan (recalibrated)
TRAIN = "#595959"                       # training-only path (dashed)
EDGE = "#000000"                        # block edges
FRAME = "#4D4D4D"                       # thumbnail frames
LW_LINE = 0.7; LW_BOX = 0.6; LW_AX = 0.6
DASH = (0, (2.6, 1.6))
HEAD_L, HEAD_W = 1.4, 0.55              # filled triangular arrowhead: length, half width (mm)
X0, X1 = 4.8, 179.2                     # content edges (mm)


def tint(hex_, f):
    """Solid mix of a colour with white (f = share of the colour)."""
    c = np.array(to_rgb(hex_)); return to_hex(1 - f * (1 - c))


FACE_STEFAN = tint(STEFAN, 0.16)
FACE_GB = tint(ACCENT, 0.16)
THAW = "#E8D6BC"                        # thawed soil (light warm)
FROZEN = "#DCE5EE"                      # seasonally frozen active layer (light cool)
PERMA = "#BFCDDC"                       # permafrost (cool)
SOIL_EDGE = "#8C8C8C"


def trunc(cmap, lo, hi, n=256):
    return ListedColormap(cmap(np.linspace(lo, hi, n)))


CM_CLIM = trunc(cmc.lajolla_r, 0.03, 0.80)     # climate inputs: light = low, dark = high
CM_TERR = trunc(cmc.grayC_r, 0.06, 0.86)       # terrain inputs
CM_SOIL = trunc(cmc.bamako_r, 0.03, 0.84)      # soil inputs
CM_ALT = trunc(cmc.oslo_r, 0.12, 0.90)         # ALT (same range as the v10 maps)
CM_DIV = cmc.vik                               # residual and anomaly (symmetric)
CM_YEAR = ListedColormap(cmc.bilbao_r(np.linspace(0.25, 0.95, 200)))   # year hold-out: sand -> dark brown over the scored years
ALASKA_ASP = 1.0 / np.cos(np.deg2rad(65.5))
THUMB_LAND = "#E6E6E6"; THUMB_COAST = "#6E6E6E"
LENA_EXT = (123.5, 130.4, 71.35, 72.95); LENA_LAT = 72.15
C_TEST, C_TRAIN, C_EXCL = "#000000", "#8C8C8C", "#8C8C8C"      # excluded = open ring (edge colour)
C_LAND, C_COAST = "#F2F2F2", "#8C8C8C"


def setup():
    S.setup()
    for f in ("/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf", "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_T, "ytick.labelsize": FS_T, "legend.fontsize": FS_S,
        "axes.linewidth": LW_AX, "xtick.major.width": LW_AX, "ytick.major.width": LW_AX, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0, "lines.linewidth": 0.9, "axes.grid": False,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": TICK, "ytick.labelcolor": TICK,
        "mathtext.fontset": "custom", "mathtext.rm": "FreeSans", "mathtext.it": "FreeSans:italic", "mathtext.bf": "FreeSans:bold",
        "mathtext.sf": "FreeSans", "mathtext.cal": "FreeSans", "mathtext.default": "it", "mathtext.fallback": None, "svg.fonttype": "none", "pdf.fonttype": 42})


def rect(x, y_top, w, h):
    return [x / W_MM, 1 - (y_top + h) / H_MM, w / W_MM, h / H_MM]


# ================================================================ drawing tools (mm canvas: x to the right, y downwards)
def canvas(fig):
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_MM); ax.set_ylim(H_MM, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def T(ax, x, y, s, fs=FS_S, ha="left", va="center", color=INK, **kw):
    return ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=color, **kw)


def tw_mm(ax, s, fs):
    r = ax.figure.canvas.get_renderer(); t = ax.text(0, 0, s, fontsize=fs); b = t.get_window_extent(r); t.remove()
    return b.width / ax.figure.dpi * 25.4


def th_mm(ax, s, fs):
    r = ax.figure.canvas.get_renderer(); t = ax.text(0, 0, s, fontsize=fs); b = t.get_window_extent(r); t.remove()
    return b.height / ax.figure.dpi * 25.4


def line(ax, pts, color=AXIS, lw=LW_LINE, ls="-", head=True, z=5, gap=None):
    """Polyline of horizontal / vertical segments with a filled triangular arrowhead at the end (v12). gap=(x, y, half) leaves a crossing gap."""
    pts = np.asarray(pts, float).copy()
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        L = float(np.hypot(x1 - x0, y1 - y0)); ux, uy = (x1 - x0) / L, (y1 - y0) / L
        tip = (float(x1), float(y1)); pts[-1] = (x1 - ux * HEAD_L * 0.85, y1 - uy * HEAD_L * 0.85)
    segs = [pts]
    if gap is not None:
        gx, gy, hh = gap; segs = []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if abs(x0 - x1) < 1e-9 and min(y0, y1) < gy < max(y0, y1) and abs(x0 - gx) < 1e-6:
                sg = np.sign(y1 - y0); segs += [np.array([[x0, y0], [x0, gy - sg * hh]]), np.array([[x0, gy + sg * hh], [x1, y1]])]
            elif abs(y0 - y1) < 1e-9 and min(x0, x1) < gx < max(x0, x1) and abs(y0 - gy) < 1e-6:
                sg = np.sign(x1 - x0); segs += [np.array([[x0, y0], [gx - sg * hh, y0]]), np.array([[gx + sg * hh, y0], [x1, y1]])]
            else:
                segs.append(np.array([[x0, y0], [x1, y1]]))
    for sg in segs:
        ax.plot(sg[:, 0], sg[:, 1], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt", dash_capstyle="butt", solid_joinstyle="miter")
    if head:
        nx, ny = -uy, ux
        tri = [tip, (tip[0] - ux * HEAD_L + nx * HEAD_W, tip[1] - uy * HEAD_L + ny * HEAD_W), (tip[0] - ux * HEAD_L - nx * HEAD_W, tip[1] - uy * HEAD_L - ny * HEAD_W)]
        ax.add_patch(Polygon(tri, closed=True, facecolor=color, edgecolor=color, lw=0.2, zorder=z + 0.1, joinstyle="miter"))


def dot(ax, x, y, color=AXIS, r=0.42, z=7):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=z))


def block(ax, x, y, w, h, face, text=None, fs=FS, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.3", facecolor=face, edgecolor=EDGE, lw=LW_BOX, zorder=z))
    if text:
        T(ax, x + w / 2, y + h / 2, text, fs=fs, ha="center", va="center", zorder=z + 1, fontweight="bold", linespacing=1.05)


def op_node(ax, x, y, r=1.45, z=7):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=AXIS, lw=LW_LINE, zorder=z))
    k = 0.55 * r
    ax.plot([x - k, x + k], [y, y], color=AXIS, lw=LW_LINE, zorder=z + 1, solid_capstyle="butt")
    ax.plot([x, x], [y - k, y + k], color=AXIS, lw=LW_LINE, zorder=z + 1, solid_capstyle="butt")


def dim_v(ax, x, y0, y1, cap=0.7, color=INK, lw=0.6, z=6):
    """Vertical dimension line without arrowheads: line + end ticks."""
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=z, solid_capstyle="butt")
    for yy in (y0, y1):
        ax.plot([x - cap / 2, x + cap / 2], [yy, yy], color=color, lw=lw, zorder=z, solid_capstyle="butt")


def thumb(fig, x, y_top, w, h, edge=FRAME, lw=0.45):
    ia = fig.add_axes(rect(x, y_top, w, h))
    ia.set_xticks([]); ia.set_yticks([])
    for sp in ia.spines.values():
        sp.set_visible(True); sp.set_edgecolor(edge); sp.set_linewidth(lw)
    ia.patch.set_facecolor("white")
    return ia


_NE = {}


def ne_layers(key):
    """Natural Earth 10 m land and coastline clipped and simplified (as maps_alt_v3.ne_layers)."""
    if key in _NE:
        return _NE[key]
    box = {"alaska": (-168.0, -141.0, 59.0, 71.5, 0.01), "lena": (123.3, 130.1, 71.5, 73.6, 0.002)}[key]
    clip = shapely.box(box[0] - 7, box[2] - 4, box[1] + 7, box[3] + 3)
    out = {}
    for name in ("land", "coastline"):
        gs = []
        for g in shpreader.Reader(shpreader.natural_earth(resolution="10m", category="physical", name=name)).geometries():
            if not g.intersects(clip):
                continue
            gi = shapely.intersection(g, clip)
            if gi.is_empty:
                continue
            gi = shapely.simplify(gi, box[4], preserve_topology=True)
            if not gi.is_empty:
                gs.append(gi)
        out[name] = gs
    _NE[key] = out
    return out


def alaska_img(ia, A, ras, cmap, norm):
    ne = ne_layers("alaska")
    for g in ne["land"]:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.exterior.xy; ia.fill(xs, ys, facecolor=THUMB_LAND, edgecolor="none", lw=0, zorder=0.5)
    ia.imshow(np.where(A["mask"], 1.0, np.nan), extent=A["extent"], origin="lower", cmap=ListedColormap([THUMB_LAND]), vmin=0, vmax=1,
              aspect=ALASKA_ASP, interpolation="nearest", zorder=1)
    if ras is not None:
        ia.imshow(ras, extent=A["extent"], origin="lower", cmap=cmap, norm=norm, aspect=ALASKA_ASP, interpolation="nearest", zorder=2)
    for g in ne["coastline"]:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.xy; ia.plot(xs, ys, color=THUMB_COAST, lw=0.3, zorder=3)
    ia.set_xlim(A["extent"][0], A["extent"][1]); ia.set_ylim(A["extent"][2], A["extent"][3]); ia.set_aspect(ALASKA_ASP)


def _fmt(v, _=None):
    return f"{v:g}".replace("-", "−")


# ================================================================ data
def load_labels():
    df = TB.load_v5(V5, products=False)
    d = TB.with_forcing(df, "year_obs")
    d = d[TB.usable_mask(d, None)].reset_index(drop=True)
    return d


def doy_label(year, doy):
    d_ = dt.date(int(year), 1, 1) + dt.timedelta(days=int(round(doy)) - 1)
    return f"{d_.day} {d_.strftime('%b')}"


def daily_cum(row, year):
    """Month-end cumulative TDD (tdd_cum_1..12) -> daily cumulative TDD, linear between month ends, 0 on 1 January (fig1_data.daily_cum)."""
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)]); ends = np.cumsum(days)
    cum = np.array([row[f"tdd_cum_{k}"] for k in range(1, 13)], float)
    doy = np.arange(0, ends[-1] + 1, dtype=float)
    return doy, np.interp(doy, np.r_[0, ends], np.r_[0, cum]), cum, ends


def cell_year_design(d):
    """Rows of a(iii) and the curve of a(ii) (selection rules recorded in the source data)."""
    L = d[(d.region == "Lena") & d.obs_doy.notna()]
    fy = pd.read_csv(V5 / "forcing_year.csv", usecols=["loc_id", "year", "e5_tdd"] + [f"tdd_cum_{k}" for k in range(1, 13)])
    lab_cells = L.loc_id.unique()
    ym = fy[fy.loc_id.isin(lab_cells) & fy.year.between(2011, 2024)].groupby("year").e5_tdd.mean()
    y_cold, y_warm = int(ym.idxmin()), int(ym.idxmax())
    has = L.groupby("loc_id").year.apply(lambda s: {y_cold, y_warm} <= set(int(v) for v in s))
    cand = [int(c) for c in has[has].index]
    nyr = d[d.loc_id.isin(cand)].groupby("loc_id").year.nunique()
    cells = sorted(cand, key=lambda c: (-int(nyr[c]), c))[:2]
    rows = L[L.loc_id.isin(cells) & L.year.isin([y_cold, y_warm])].copy()
    rows["order"] = rows.loc_id.map({c: i for i, c in enumerate(cells)})
    rows = rows.sort_values(["order", "year"]).reset_index(drop=True)
    rows["date"] = [doy_label(y, o) for y, o in zip(rows.year, rows.obs_doy)]
    k = int(rows.tdd_obs_frac.values.argmin()); cur = rows.iloc[k]
    cell, year = int(cur.loc_id), int(cur.year)
    fr = fy[(fy.loc_id == cell) & (fy.year == year)].iloc[0]
    doy, cum, cum_m, ends = daily_cum(fr, year)
    t_obs = float(np.interp(float(cur.obs_doy), doy, cum))
    assert abs(t_obs - float(cur.e5_tdd)) < 0.5, (t_obs, float(cur.e5_tdd))
    t_full = float(cur.y_e5_tdd); t_clim = float(cur.c_e5_tdd)
    assert abs(t_full - cum_m[-1]) < 1e-6
    inc = np.diff(np.r_[0.0, cum_m]); m_end = int(np.where(inc > 1e-9)[0].max())       # last month with positive TDD increment (0-based)
    doy_end = float(ends[m_end])
    others = {}
    for yy in range(2011, 2025):
        if yy == year:
            continue
        r_ = fy[(fy.loc_id == cell) & (fy.year == yy)]
        if len(r_):
            dd, cc, _, _ = daily_cum(r_.iloc[0], yy); others[yy] = (dd, cc)
    ann = fy[(fy.loc_id == cell) & fy.year.between(2011, 2024)].e5_tdd
    assert abs(float(ann.mean()) - t_clim) < 1e-3
    return dict(rows=rows, cells=cells, n_years={int(c): int(nyr[c]) for c in cells}, years=(y_cold, y_warm), year_mean_tdd={int(k_): float(v) for k_, v in ym.items()},
                cell=cell, year=year, lat=float(cur.lat), lon=float(cur.lon), obs_doy=float(cur.obs_doy), doy=doy, cum=cum, cum_m=cum_m,
                t_obs=t_obs, t_full=t_full, t_clim=t_clim, doy_end=doy_end, others=others, k_row=k)


def stefan_data(d, meta):
    a = d[d.region == "Alaska"]
    x = a.e5_sqrt_tdd.values.astype(float); y = a.alt_cm.values.astype(float)
    E = TB.ls_scale(x, y)
    assert abs(E - float(meta["E"])) < 1e-6, (E, meta["E"])
    assert len(a) == int(meta["n_train"])
    r = y - E * x; sd = float(np.std(r))
    picks = []
    for q, sgn in zip((0.1, 0.3, 0.5, 0.7, 0.9), (1, -1, 1, -1, 1)):
        xq = float(np.quantile(x, q))
        near = np.argsort(np.abs(x - xq), kind="stable")[:400]
        j = int(near[np.argmin(np.abs(r[near] - sgn * sd))])
        picks.append(j)
    P = a.iloc[picks]
    return dict(x=x, y=y, E=E, sd=sd, n=len(a), picks=P, r_picks=(P.alt_cm - E * P.e5_sqrt_tdd).values,
                years=sorted(int(v) for v in a.year.unique()))


def forcing_year_thumb(year=2014):
    cache = MAPS / "Alaska" / f"forcing_{year}_thumb.csv.gz"
    cols = ["e5_maat", "e5_fdd", "e5_swe"]
    d = pd.read_csv(cache, dtype={"cell_id": str})
    assert all(c in d.columns for c in cols)
    return d, str(cache.relative_to(ROOT))


def alaska_rasters(lam, year=2014, anom_years=(2012, 2019)):
    """Alaska 1 km map cells -> 0.05 deg x 0.02 deg box means (v10 fig2_method.alaska_thumbs rule)."""
    ck = CACHE / f"fig2_alaska_{year}_{'_'.join(map(str, anom_years))}.npz" if CACHE else None
    if ck is not None and ck.exists():
        z = np.load(ck, allow_pickle=True); return z["A"].item()
    p = pd.read_csv(MAPS / "Alaska" / f"pred_{year}.csv.gz", dtype={"cell_id": str})
    g = pd.read_csv(PROC / "map_alaska" / "alaska_grid_x25_v1.csv.gz", dtype={"cell_id": str},
                    usecols=["cell_id", "lat", "lon", "dem_elev", "dem_slope", "sg_soc_0_5", "sg_clay_5_15", "land"])
    fy, fy_src = forcing_year_thumb(year)
    fy = fy.rename(columns={"e5_fdd": "fdd", "e5_swe": "swe"})[["cell_id", "fdd", "swe"]]
    mean = pd.read_csv(MAPS / "Alaska" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).rename(columns={"r1": "r1_mean"})
    m = p.merge(g.drop(columns=["lat", "lon"]), on="cell_id", how="left").merge(fy, on="cell_id", how="left").merge(mean, on="cell_id", how="left")
    m["resid"] = (m.r1 - m.p1) / lam
    for ay in anom_years:
        q = pd.read_csv(MAPS / "Alaska" / f"pred_{ay}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).rename(columns={"r1": f"r1_{ay}"})
        m = m.merge(q, on="cell_id", how="left"); m[f"anom_{ay}"] = m[f"r1_{ay}"] - m.r1_mean
    lon0, lon1 = np.floor(m.lon.min() * 10) / 10, np.ceil(m.lon.max() * 10) / 10
    lat0, lat1 = np.floor(m.lat.min() * 10) / 10, np.ceil(m.lat.max() * 10) / 10
    xe = np.arange(lon0, lon1 + 1e-9, 0.05); ye = np.arange(lat0, lat1 + 1e-9, 0.02)
    gl = g[g.land == 1]; cnt, _, _ = np.histogram2d(gl.lat, gl.lon, bins=[ye, xe])
    keys = ["e5_tdd", "fdd", "swe", "dem_elev", "dem_slope", "sg_soc_0_5", "sg_clay_5_15", "p1", "r1", "resid"] + [f"anom_{ay}" for ay in anom_years]
    ras, stats = {}, {}
    for k in keys:
        v = m[k].values.astype(float); ok = np.isfinite(v)
        s_, _, _ = np.histogram2d(m.lat.values[ok], m.lon.values[ok], bins=[ye, xe], weights=v[ok])
        c_, _, _ = np.histogram2d(m.lat.values[ok], m.lon.values[ok], bins=[ye, xe])
        with np.errstate(invalid="ignore", divide="ignore"):
            ras[k] = np.where(c_ > 0, s_ / c_, np.nan)
        stats[k] = dict(cells=int(ok.sum()), p1=float(np.nanpercentile(v, 1)), p2=float(np.nanpercentile(v, 2)), p98=float(np.nanpercentile(v, 98)),
                        p99=float(np.nanpercentile(v, 99)), abs_p99=float(np.nanpercentile(np.abs(v[ok]), 99)), mean=float(np.nanmean(v)))
    ext = [lon0, lon1, lat0, lat1]; wh = (lon1 - lon0) / ((lat1 - lat0) * ALASKA_ASP)
    A = dict(ras=ras, mask=cnt > 0, extent=ext, wh=wh, stats=stats, n_cells=int(len(m)), forcing_src=fy_src, year=year, anom_years=list(anom_years))
    if ck is not None:
        ck.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(ck, A=np.array(A, dtype=object))
    return A


def lena_folds(d_all):
    """Lena rows and folds as the harness (v5_compare.py); fold assignments checked against data/processed/v5/runs/core_v6/preds."""
    dr = d_all[d_all.region == "Lena"].reset_index(drop=True)
    F = {s: TB.make_folds(dr, s, 5, 0) for s in ("random", "block", "year", "block_year")}
    F["region"] = [f for f in TB.make_folds(d_all, "region", 5, 0) if f.name == "L_Lena"]
    check = {}
    for s in ("random", "block", "year", "block_year"):
        fp = V5 / "runs" / "core_v6" / "preds" / f"Lena__{s}__year_obs__F8TS__P1__-__lam-1.csv.gz"
        p = pd.read_csv(fp, usecols=["fold", "cell_year_id"])
        mine = pd.Series({cid: f.name for f in F[s] for cid in dr.cell_year_id.values[f.te]})
        check[s] = bool(len(p) == len(mine) and (p.set_index("cell_year_id").fold.reindex(mine.index) == mine).all())
    fp = V5 / "runs" / "core_v6" / "preds" / "ALL__region__year_obs__F8TS__P1__-__lam-1.csv.gz"
    p = pd.read_csv(fp, usecols=["fold", "cell_year_id"]); pl = p[p.fold == "L_Lena"]
    check["region"] = bool(set(pl.cell_year_id) == set(d_all.cell_year_id.values[F["region"][0].te]))
    assert all(check.values()), check
    return dr, F, check


# ================================================================ panel letters (bold letters only; no row titles, no hairlines)
def letter_at(ax, x, y, s):
    return T(ax, x, y, s, fs=FS_LETTER, ha="left", va="top", fontweight="bold")


# ================================================================ row a: cell-year design and the Stefan relation
def row_a(fig, ax, y0, D, SD, meta):
    """a: soil-column schematic | cumulative TDD of one cell-year | Stefan anchor (all Alaska labels). Keys sit above the axes; in-plot
    labels are placed in empty regions (checked by the QA) so that no text touches a line or a point."""
    out = {}
    # ---------- (i) soil columns: observation day (left) and end of season (right)
    cw, gap = 8.0, 8.0
    w_lab = max(tw_mm(ax, "Thaw", FS_S), tw_mm(ax, "depth", FS_S))
    xL = X0 + w_lab + 3.2; xR = xL + cw + gap
    ys, yo, ya, yb = y0 + 7.0, y0 + 13.1, y0 + 19.9, y0 + 29.1         # surface, thaw front on obs. day, permafrost table, bottom
    for (x, segs) in ((xL, ((ys, yo, THAW), (yo, ya, FROZEN), (ya, yb, PERMA))), (xR, ((ys, ya, THAW), (ya, yb, PERMA)))):
        for (a_, b_, c_) in segs:
            ax.add_patch(Rectangle((x, a_), cw, b_ - a_, facecolor=c_, edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((x, ys), cw, yb - ys, facecolor="none", edgecolor=INK, lw=0.45, zorder=3))
    ax.plot([xL - 0.9, xR + cw + 0.9], [ys, ys], color=INK, lw=0.9, zorder=4, solid_capstyle="butt")       # ground surface
    for xa_, xb_ in ((xL, xL + cw), (xR, xR + cw)):
        ax.plot([xa_, xb_], [ya, ya], color=INK, lw=0.6, zorder=4, solid_capstyle="butt")                 # permafrost table
    ax.plot([xL + cw, xR], [ya, ya], color=INK, lw=0.45, ls=(0, (1.0, 1.0)), zorder=4)
    ax.plot([xL, xL + cw], [yo, yo], color=INK, lw=0.6, ls=(0, (2.0, 1.1)), zorder=4)                     # thaw front, obs. day
    yh = y0 + 2.6
    for x, s, filled in ((xL, "Obs. day", False), (xR, "End of season", True)):
        cx = x + cw / 2; w = tw_mm(ax, s, FS_S) + 1.85
        ax.add_patch(Circle((cx - w / 2 + 0.65, yh), 0.62, facecolor=INK if filled else "white", edgecolor=INK, lw=0.55, zorder=6))
        T(ax, cx - w / 2 + 1.95, yh, s, ha="left")
    dim_v(ax, xL - 1.5, ys, yo)
    T(ax, xL - 2.4, (ys + yo) / 2, "Thaw\ndepth", ha="right", linespacing=1.0)
    dim_v(ax, xR + cw + 1.5, ys, ya)
    xr = xR + cw + 2.8
    T(ax, xr, ys, "Ground surface", ha="left")
    T(ax, xr, (ys + ya) / 2, "ALT", ha="left")
    T(ax, xr, ya, "Permafrost table", ha="left")
    T(ax, xr, (ya + yb) / 2 + 0.6, "Permafrost", ha="left")
    yk = y0 + 32.8; xk = xL - 0.3
    for c_, s in ((THAW, "Thawed"), (FROZEN, "Seasonally frozen")):
        ax.add_patch(Rectangle((xk, yk - 0.9), 2.4, 1.8, facecolor=c_, edgecolor=INK, lw=0.35, zorder=3))
        T(ax, xk + 3.0, yk, s, ha="left"); xk += 3.0 + tw_mm(ax, s, FS_S) + 3.2
    x_end_i = xr + tw_mm(ax, "Permafrost table", FS_S)

    # ---------- (ii) cumulative TDD of one cell-year (key above the axes, two in-plot labels in empty regions)
    xa = x_end_i + 11.5; wa = 40.0; ya_, ha = y0 + 9.0, 22.0
    a2 = fig.add_axes(rect(xa, ya_, wa, ha))
    yr = D["year"]; doy, cum = D["doy"], D["cum"]
    X_LO, X_HI = 146.0, 304.0
    for yy, (dd, cc) in sorted(D["others"].items()):
        a2.plot(dd, cc, color="#CFCFCF", lw=0.45, zorder=2, solid_capstyle="butt")
    a2.plot(doy, cum, color=INK, lw=1.0, zorder=4, solid_capstyle="butt")
    a2.axhline(D["t_clim"], color=ZERO, lw=0.7, ls=(0, (3.0, 1.6)), zorder=3)
    od = D["obs_doy"]
    a2.plot([od, od], [0, D["t_obs"]], color=ZERO, lw=0.5, ls=(0, (1.0, 1.0)), zorder=3)
    a2.plot([od], [D["t_obs"]], "o", ms=3.4, mfc="white", mec=INK, mew=0.7, zorder=6)
    t_ref = float(np.interp(244.0, doy, cum))
    a2.plot([244.0], [t_ref], "o", ms=3.2, mfc=INK, mec=INK, mew=0.0, zorder=6)
    a2.set_xlim(X_LO, X_HI); a2.set_ylim(0, 1200)
    starts = np.r_[0, np.cumsum([calendar.monthrange(yr, m)[1] for m in range(1, 13)])]
    a2.xaxis.set_major_locator(FixedLocator([s_ for s_ in starts if X_LO <= s_ <= X_HI]))
    mids = (starts[:-1] + starts[1:]) / 2
    a2.xaxis.set_minor_locator(FixedLocator([m_ for m_ in mids if X_LO <= m_ <= X_HI]))
    a2.xaxis.set_major_formatter(NullFormatter())
    a2.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: calendar.month_abbr[int(np.searchsorted(starts, x))]))
    a2.tick_params(axis="x", which="minor", length=0, pad=1.8)
    a2.yaxis.set_major_locator(FixedLocator([0, 500, 1000])); a2.yaxis.set_minor_locator(MultipleLocator(250))
    for k in ("top", "right"):
        a2.spines[k].set_visible(False)
    a2.set_xlabel(f"Month ({yr})", labelpad=1.5); a2.set_ylabel("Cumulative TDD (°C d)", labelpad=2.0)
    hs = [Line2D([], [], color=INK, lw=1.0), Line2D([], [], color="#BDBDBD", lw=0.8), Line2D([], [], color=ZERO, lw=0.7, ls=(0, (3.0, 1.6)))]
    a2.legend(hs, [f"{yr}", "Other years 2011–2024", "Mean 2011–2024"], loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False,
              fontsize=FS_T, handlelength=1.5, handletextpad=0.5, labelspacing=0.2, borderaxespad=0.0, borderpad=0.0)
    labs = [a2.text(od + 3.5, 110, doy_label(yr, od), ha="left", va="center", fontsize=FS_T, color=AUX, zorder=7),
            a2.text(od - 4.0, D["t_obs"] - 45, "To obs. day", ha="right", va="top", fontsize=FS_T, color=AUX, zorder=7),
            a2.text(246.0, t_ref - 45, "To 1 Sep (maps)", ha="left", va="top", fontsize=FS_T, color=AUX, zorder=7)]
    out["a2"] = a2; out["a2_labels"] = labs

    # ---------- (iii) Stefan anchor: observed ALT against sqrt(TDD_obs), all Alaska labels
    E = SD["E"]
    xs0 = xa + wa + 14.5; ws = X1 - xs0 - 1.0; ys0, hs_ = y0 + 9.0, 22.0
    a1 = fig.add_axes(rect(xs0, ys0, ws, hs_))
    a1.scatter(SD["x"], SD["y"], s=0.8, color="#C4C4C4", lw=0, zorder=2, rasterized=True)
    XM, YM = 50.0, 200.0
    xx = np.array([0.0, XM]); a1.plot(xx, E * xx, color=STEFAN, lw=1.1, zorder=4, solid_capstyle="butt")
    P = SD["picks"]
    for (_, r), rr in zip(P.iterrows(), SD["r_picks"]):
        a1.plot([r.e5_sqrt_tdd, r.e5_sqrt_tdd], [E * r.e5_sqrt_tdd, r.alt_cm], color=ACCENT, lw=0.9, zorder=5, solid_capstyle="butt")
        a1.plot([r.e5_sqrt_tdd], [r.alt_cm], "o", ms=2.8, mfc=INK, mec="white", mew=0.4, zorder=6)
    a1.set_xlim(0, XM); a1.set_ylim(0, YM)
    a1.xaxis.set_major_locator(FixedLocator([0, 25, 50])); a1.xaxis.set_minor_locator(MultipleLocator(5))
    a1.yaxis.set_major_locator(FixedLocator([0, 100, 200])); a1.yaxis.set_minor_locator(MultipleLocator(25))
    for k in ("top", "right"):
        a1.spines[k].set_visible(False)
    a1.set_xlabel(r"$\sqrt{\mathrm{TDD}}$ to observation day ($\sqrt{\mathrm{°C\,d}}$)", labelpad=1.5)
    a1.set_ylabel("Observed ALT (cm)", labelpad=2.0)
    n_off = int((SD["y"] > YM).sum())
    hs1 = [Line2D([], [], color=STEFAN, lw=1.1), Line2D([], [], color=ACCENT, lw=0.9, marker="|", ms=4.5, mew=0.9, ls="none")]
    a1.legend(hs1, ["Stefan anchor (least squares)", "Residual"], loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=1, frameon=False, fontsize=FS_T, handlelength=1.5, handletextpad=0.5, labelspacing=0.2, borderaxespad=0.0, borderpad=0.0)
    labs1 = [a1.text(0.03, 0.97, "Alaska\nn = %s" % f"{SD['n']:,}", transform=a1.transAxes, ha="left", va="top", fontsize=FS_T, linespacing=1.05)]
    if n_off:
        labs1.append(a1.text(0.03, 0.70, f"{n_off} > {YM:g} cm", transform=a1.transAxes, ha="left", va="top", fontsize=FS_T, color=AUX))
    out["a1"] = a1; out["a1_labels"] = labs1
    SRC["a"] = dict(
        layout_mm=dict(soil_columns=[round(xL, 2), round(xR + cw, 2)], curve_axes=[round(xa, 2), round(xa + wa, 2)], stefan_axes=[round(xs0, 2), round(xs0 + ws, 2)]),
        soil_column="vector schematic, no data: two columns (observation day, end of season); thawed %s, seasonally frozen %s, permafrost %s; dashed line = thaw front on the observation day; solid = permafrost table" % (THAW, FROZEN, PERMA),
        curve=dict(cell=D["cell"], lat=D["lat"], lon=D["lon"], year=yr, observation_day=dict(doy=D["obs_doy"], date=doy_label(yr, D["obs_doy"])),
                   tdd_obs_degC_day=round(D["t_obs"], 2), tdd_full_season_degC_day=round(D["t_full"], 2), tdd_to_1_sep_degC_day=round(t_ref, 2), tdd_mean_2011_2024_degC_day=round(D["t_clim"], 2),
                   season_end_doy=D["doy_end"], season_end_date=doy_label(yr, D["doy_end"]), month_end_cumulative_tdd=[round(float(v), 2) for v in D["cum_m"]],
                   light_lines="the same cell's daily cumulative TDD in the other years 2011-2024 (%d lines)" % len(D["others"]),
                   cell_rule="Lena Delta cell with dated observations in the coldest and warmest year 2011-2024 and the most label years; the cell-year with the smallest TDD_obs / full-season ratio",
                   inputs=["data/processed/v5/forcing_year.csv (tdd_cum_1..12, e5_tdd)", "data/processed/v5/forcing_clim.csv (clim1124_tdd via polar.testbed_v5)",
                           "data/processed/v5/labels_cell_year_v5.csv (obs_doy)"],
                   rule="daily linear interpolation of month-end cumulative TDD (0 on 1 January); TDD_obs equals polar.testbed_v5.with_forcing('year_obs') e5_tdd (checked < 0.5 degC d)"),
        stefan_anchor=dict(region="Alaska", n_cell_years=SD["n"], label_years=f"{SD['years'][0]}-{SD['years'][-1]}", E_cm_per_sqrt_degC_day=round(E, 4),
                           E_printed=f"{E:.2f}", E_rule="least squares through the origin on all Alaska labels (polar.testbed_v5.ls_scale), equal to data/processed/v5/maps/Alaska/meta.json E",
                           x="sqrt(TDD to the observation day), polar.testbed_v5.with_forcing('year_obs')", axis_range=dict(x=[0, XM], y=[0, YM]),
                           n_above_y_range=n_off, residual_sd_cm=round(SD["sd"], 2),
                           highlighted=[dict(cell=int(r.loc_id), year=int(r.year), sqrt_tdd_obs=round(float(r.e5_sqrt_tdd), 2), alt_cm=round(float(r.alt_cm), 1),
                                             residual_cm=round(float(rr), 1)) for (_, r), rr in zip(P.iterrows(), SD["r_picks"])],
                           highlight_rule="at the 10/30/50/70/90th percentiles of sqrt(TDD_obs): among the 400 nearest cell-years, the one whose residual is closest to +1, -1, +1, -1, +1 residual s.d."))
    return out


# ================================================================ row b: physics-residual model (Alaska 2014 map flow)
def eq_alt_hat(ax, x, y, fs, color=INK):
    """'ALT' with a thin drawn hat (mathtext \\widehat renders an oversized wedge in FreeSans). Baseline at y (mm). Returns the right edge (mm)."""
    t = ax.text(x, y, r"$\mathrm{ALT}$", fontsize=fs, ha="left", va="baseline", color=color, zorder=6)
    r = ax.figure.canvas.get_renderer(); bb = t.get_window_extent(r)
    inv = ax.transData.inverted(); (xa, yt), (xb, _) = inv.transform([(bb.x0, bb.y1), (bb.x1, bb.y1)])
    cap = 0.716 * fs * 25.4 / 72.0                          # FreeSans cap height (mm)
    ytop = y - cap; w = xb - xa; xm = (xa + xb) / 2
    ax.plot([xm - 0.30 * w, xm, xm + 0.30 * w], [ytop - 0.35, ytop - 1.05, ytop - 0.35], color=color, lw=0.6, zorder=6,
            solid_joinstyle="miter", solid_capstyle="butt")
    return xb, t


# input thumbnails: one colour map per variable (deck v7 grammar, docs/FIGURE_AUDIT_v12 9절 사용자 지적 10-18 밤), each stretched to its own percentile range
INPUTS = {"swe": ("Snow", trunc(cmc.oslo_r, 0.08, 0.92), (2, 98)), "fdd": ("FDD", trunc(cmc.devon_r, 0.08, 0.92), (2, 98)),
          "e5_tdd": ("TDD", trunc(cmc.lajolla_r, 0.15, 0.88), (2, 98)),
          "dem_elev": ("Elevation", trunc(cmc.lapaz_r, 0.06, 0.94), (2, 98)), "dem_slope": ("Slope", trunc(cmc.acton_r, 0.06, 0.94), (2, 98)),
          "sg_soc_0_5": ("Soil carbon", trunc(cmc.bamako_r, 0.05, 0.92), (2, 98)), "sg_clay_5_15": ("Clay", trunc(cmc.turku_r, 0.06, 0.92), (2, 98))}


def row_b(fig, ax, y0, A, meta):
    out = {}
    lam = float(meta["lam"]); yr = A["year"]
    # (v13.1) no equations in the figure: the model is equation (5) of the text
    # ---------- input thumbnails (Alaska, 2014)
    TH = 16.5; TW = TH * A["wh"]
    xf0 = X0 + 2.0
    yU = y0 + 13.0; yL = yU + TH + 8.0
    g_in, g_grp = 1.9, 3.6
    xs_L = [xf0, xf0 + TW + g_in, xf0 + 2 * TW + g_in + g_grp, xf0 + 3 * TW + 2 * g_in + g_grp]
    x_in_r = xs_L[-1] + TW
    xs_U = [xf0, xf0 + TW + g_in, xf0 + 2 * (TW + g_in)]
    CLIM = ["swe", "fdd", "e5_tdd"]; TERR = ["dem_elev", "dem_slope"]; SOIL = ["sg_soc_0_5", "sg_clay_5_15"]
    norms = {}
    for key, x, yc in [(c, x, yU) for c, x in zip(CLIM, xs_U)] + [(c, x, yL) for c, x in zip(TERR, xs_L[:2])] + [(c, x, yL) for c, x in zip(SOIL, xs_L[2:])]:
        nm, cmap, pct = INPUTS[key]
        ia = thumb(fig, x, yc - TH / 2, TW, TH)
        lo, hi = np.nanpercentile(A["ras"][key], pct); norms[key] = (float(lo), float(hi))
        alaska_img(ia, A, A["ras"][key], cmap, Normalize(lo, hi))
        T(ax, x + TW / 2, yc + TH / 2 + 0.8, nm, ha="center", va="top")
    feats = list(meta["feats"]); n_clim = sum(f in TB.F8 for f in feats); n_terr = sum(f in TB.TERRAIN for f in feats); n_soil = sum(f in TB.SOIL for f in feats)
    assert n_clim + n_terr + n_soil == len(feats) == 23
    T(ax, xs_U[0], yU - TH / 2 - 0.9, f"Climate {yr}, {n_clim} variables", ha="left", va="bottom")
    T(ax, xs_L[0], yL - TH / 2 - 0.9, f"Terrain, {n_terr} variables", ha="left", va="bottom")
    T(ax, xs_L[2], yL - TH / 2 - 0.9, f"Soil, {n_soil} variables", ha="left", va="bottom")
    T(ax, xf0, yL + TH / 2 + 4.6, "Each input stretched to its 2nd–98th percentile", ha="left", va="top", color=AUX)
    # bracket (all 23 inputs) -> gradient boosting; TDD -> Stefan crosses the bracket through a gap
    xbr = x_in_r + 2.4; ytb, ybb = yU - TH / 2, yL + TH / 2
    for (ya_, yb_) in ((ytb, yU - 1.1), (yU + 1.1, ybb)):
        ax.plot([xbr, xbr], [ya_, yb_], color=AXIS, lw=LW_LINE, zorder=5, solid_capstyle="butt")
    for yy in (ytb, ybb):
        ax.plot([xbr - 1.0, xbr], [yy, yy], color=AXIS, lw=LW_LINE, zorder=5, solid_capstyle="butt")
    bx, bw, bh = xbr + 5.6, 18.0, 8.0
    block(ax, bx, yU - bh / 2, bw, bh, FACE_STEFAN, "Stefan\nmodel", fs=FS)
    block(ax, bx, yL - bh / 2, bw, bh, FACE_GB, "Gradient\nboosting", fs=FS)
    line(ax, [(xs_U[2] + TW, yU), (bx, yU)])
    T(ax, (xs_U[2] + TW + xbr) / 2, yU - 0.9, "To 1 Sep", ha="center", va="bottom", color=AUX)
    line(ax, [(xbr, yL), (bx, yL)])
    # training target (dashed, training only)
    xg = bx + bw / 2; yt = yL + bh / 2 + 5.6
    line(ax, [(xg, yt), (xg, yL + bh / 2)], color=TRAIN, ls=DASH)
    T(ax, xg, yt + 0.6, "Training target", ha="center", va="top", color=AUX)
    # outputs: Stefan map, residual map, weighted sum, ALT map
    ox = bx + bw + 6.0
    lo_alt = float(5 * math.floor(min(A["stats"]["p1"]["p1"], A["stats"]["r1"]["p1"]) / 5))
    hi_alt = float(5 * math.ceil(max(A["stats"]["p1"]["p99"], A["stats"]["r1"]["p99"]) / 5))
    nALT = Normalize(lo_alt, hi_alt)
    vres = float(5 * math.ceil(A["stats"]["resid"]["abs_p99"] / 5)); nRES = TwoSlopeNorm(0.0, -vres, vres)
    line(ax, [(bx + bw, yU), (ox, yU)])
    line(ax, [(bx + bw, yL), (ox, yL)], color=ACCENT)
    ia = thumb(fig, ox, yU - TH / 2, TW, TH, edge=STEFAN, lw=0.7); alaska_img(ia, A, A["ras"]["p1"], CM_ALT, nALT)
    T(ax, ox + TW / 2, yU + TH / 2 + 0.8, f"Stefan, {yr}", ha="center", va="top")
    ia = thumb(fig, ox, yL - TH / 2, TW, TH, edge=ACCENT, lw=0.7); alaska_img(ia, A, A["ras"]["resid"], CM_DIV, nRES)
    T(ax, ox + TW / 2, yL + TH / 2 + 0.8, f"Residual, {yr}", ha="center", va="top")
    xp = ox + TW + 6.2; rp_ = 1.6
    line(ax, [(ox + TW, yU), (xp - rp_, yU)])
    line(ax, [(ox + TW, yL), (xp, yL), (xp, yU + rp_)], color=ACCENT)
    T(ax, xp + 1.3, (yU + yL) / 2 + 1.6, "weight 0.5", ha="left", color=AUX)
    op_node(ax, xp, yU, rp_)
    gx = xp + rp_ + 4.8
    line(ax, [(xp + rp_, yU), (gx, yU)], color=ACCENT)
    ia = thumb(fig, gx, yU - TH / 2, TW, TH, edge=ACCENT, lw=0.9); alaska_img(ia, A, A["ras"]["r1"], CM_ALT, nALT)
    T(ax, gx + TW / 2, yU + TH / 2 + 0.8, f"ALT {yr}", ha="center", va="top")
    tk = [t for t in range(0, 200, 10) if lo_alt <= t <= hi_alt]; tk = tk[::2] if len(tk) > 4 else tk
    vcax = fig.add_axes(rect(gx + TW + 1.6, yU - TH / 2, 1.2, TH))
    cb = fig.colorbar(ScalarMappable(norm=nALT, cmap=CM_ALT), cax=vcax, orientation="vertical", ticks=tk)
    cb.outline.set_linewidth(0.45); cb.outline.set_edgecolor(FRAME)
    vcax.tick_params(labelsize=FS_T, width=0.45, length=1.5, pad=1.0, labelcolor=TICK)
    T(ax, gx + TW + 2.2, yU - TH / 2 - 0.7, "cm", ha="center", va="bottom", color=AUX)
    hcax = fig.add_axes(rect(ox - 0.2, yL + TH / 2 + 4.6, TW + 0.4, 1.0))
    cb2 = fig.colorbar(ScalarMappable(norm=nRES, cmap=CM_DIV), cax=hcax, orientation="horizontal", ticks=[-vres, 0, vres])
    cb2.outline.set_linewidth(0.45); cb2.outline.set_edgecolor(FRAME)
    hcax.tick_params(labelsize=FS_T, width=0.45, length=1.5, pad=1.0, labelcolor=TICK); hcax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    T(ax, ox + TW + 1.2, yL + TH / 2 + 5.1, "cm", ha="left", va="center", color=AUX)
    SRC["b"] = dict(
        flow=dict(year=yr, inputs={k: dict(name=INPUTS[k][0], colormap=INPUTS[k][1].name if hasattr(INPUTS[k][1], "name") else "listed", percentiles=list(INPUTS[k][2]))
                                   for k in CLIM + TERR + SOIL},
                  colour_rule="one cmcrameri colour map per variable (snow oslo_r, FDD devon_r, TDD lajolla_r, elevation lapaz_r, slope acton_r, soil carbon bamako_r, clay turku_r); darker = larger; each thumbnail stretched to its own percentile range (no absolute scale)",
                  n_features=dict(climate=n_clim, terrain=n_terr, soil=n_soil, total=len(feats)), features=feats,
                  input_ranges=norms, stefan_map="p1 = E sqrt(TDD to 1 September, %d)" % yr, residual_map="(r1 - p1) / lambda = residual prediction before weighting",
                  alt_map="r1 = p1 + lambda * residual (seed mean of 3 CatBoost models trained on all Alaska labels)", alt_range_cm=[lo_alt, hi_alt], alt_ticks_cm=tk,
                  residual_range_cm=[-vres, vres], lam=lam, learner=meta["learner"], seeds=meta["seeds"], n_train=int(meta["n_train"]), n_map_cells=A["n_cells"],
                  inputs_files=["data/processed/v5/maps/Alaska/pred_%d.csv.gz (e5_tdd, p1, r1)" % yr, A["forcing_src"] + " (e5_fdd, e5_swe)",
                                "data/processed/map_alaska/alaska_grid_x25_v1.csv.gz (dem_elev, dem_slope, sg_soc_0_5, sg_clay_5_15, land)",
                                "data/processed/v5/maps/Alaska/meta.json (E, lam, feats, learner, seeds, n_train)"],
                  raster="1 km map cells averaged in 0.05 deg x 0.02 deg boxes", thumbnail_mm=[round(TW, 2), TH]),
        equation=dict(main="ALT_hat = E_hat sqrt(TDD_obs) + lambda g(x)", target="g ~ ALT - E_hat sqrt(TDD_obs) (training only)", lam=lam,
                      note="maps use the TDD accumulated to 1 September in place of TDD_obs"),
        diagram_style=dict(connectors=f"black {LW_LINE} pt with filled triangular heads {HEAD_L} mm", boxes=f"{LW_BOX} pt black edge, light tint of the method colour",
                           box_titles="7 pt bold above the box"))
    return out


# ================================================================ row c: validation designs (all folds coloured) and yearly outputs
def lena_map(fig, x, y_top, mw, mh):
    ne = ne_layers("lena"); asp = 1.0 / np.cos(np.deg2rad(LENA_LAT)); ext = LENA_EXT
    mx = fig.add_axes(rect(x, y_top, mw, mh))
    for g in ne["land"]:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.exterior.xy; mx.fill(xs, ys, facecolor=C_LAND, edgecolor="none", lw=0, zorder=1)
    for g in ne["coastline"]:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.xy; mx.plot(xs, ys, color=C_COAST, lw=0.3, zorder=2)
    mx.set_xlim(ext[0], ext[1]); mx.set_ylim(ext[2], ext[3]); mx.set_aspect(asp); mx.set_xticks([]); mx.set_yticks([])
    for sp in mx.spines.values():
        sp.set_visible(True); sp.set_edgecolor(FRAME); sp.set_linewidth(0.45)
    return mx


def row_c(fig, ax, y0, dr, F, check, A):
    out = {}
    show = [("Random", "random"), ("Spatial block 0.5°", "block"), ("Year hold-out", "year"), ("Block × year", "block_year"), ("Region hold-out", "region")]
    ext = LENA_EXT; asp = 1.0 / np.cos(np.deg2rad(LENA_LAT))
    mw = 23.0; gap = 2.4; mh = mw * (ext[3] - ext[2]) * asp / (ext[1] - ext[0])
    yl = y0 + 1.8; ytop = y0 + 8.2
    T(ax, X0, yl, "Validation designs, Lena Delta label cells (all folds)", ha="left")
    # fold indices (harness folds: polar.testbed_v5.make_folds k = 5, seed 0)
    fold_of = {}
    for s in ("random", "block"):
        idx = np.full(len(dr), -1)
        for k, f in enumerate(sorted(F[s], key=lambda f_: f_.name)):
            idx[f.te] = k
        assert (idx >= 0).all(); fold_of[s] = idx
    blk_fold = {}
    for k, f in enumerate(sorted(F["block"], key=lambda f_: f_.name)):
        for b in set(dr.block.values[f.te]):
            blk_fold[int(b)] = k
    y_scored = sorted(int(f.name[1:]) for f in F["year"])
    nYR = Normalize(min(y_scored), max(y_scored))
    bq = dr.groupby("block").agg(lon=("lon", "min"), lat=("lat", "min"))
    bq["x"] = np.floor(bq.lon / 0.5) * 0.5; bq["y"] = np.floor(bq.lat / 0.5) * 0.5
    cell = dr.groupby("loc_id").agg(lon=("lon", "first"), lat=("lat", "first"), block=("block", "first"))
    info = {}
    d_all = F["_d_all"]; f_reg = F["region"][0]; tr_reg = d_all.iloc[f_reg.tr].region.value_counts().to_dict()
    yk1 = ytop + mh + 2.3; yk2 = yk1 + 2.8
    for i, (title, s) in enumerate(show):
        x = X0 + i * (mw + gap)
        mx = lena_map(fig, x, ytop, mw, mh)
        if s in ("block", "block_year"):
            for b, r_ in bq.iterrows():
                mx.add_patch(Rectangle((r_.x, r_.y), 0.5, 0.5, facecolor="none", edgecolor=FOLD_HEX[blk_fold[int(b)]], lw=0.5, zorder=3))
        if s in ("random", "block"):
            q = dr.assign(f=fold_of[s])
            if s == "block":
                cf = q.groupby("loc_id").f.first().reindex(cell.index).values
                for k in range(5):
                    cc = cell[cf == k]; mx.scatter(cc.lon, cc.lat, s=3.0, color=FOLD_HEX[k], lw=0, zorder=4, rasterized=True)
            else:                                           # random split is by cell-year: a cell's years fall into several folds (overplotted dots)
                q = q.sample(frac=1.0, random_state=0)
                mx.scatter(q.lon, q.lat, s=3.0, c=[FOLD_HEX[int(k)] for k in q.f], lw=0, zorder=4, rasterized=True)
            info[title] = dict(scheme=s, rows_per_fold={int(k): int(v) for k, v in q.f.value_counts().sort_index().items()})
        elif s in ("year", "block_year"):
            qs = dr.sort_values("year"); sc_ = qs.year.isin(y_scored).values
            mx.scatter(qs.lon[~sc_], qs.lat[~sc_], s=3.0, color="#C8C8C8", lw=0, zorder=3.5, rasterized=True)
            mx.scatter(qs.lon[sc_], qs.lat[sc_], s=3.0, c=qs.year[sc_], cmap=CM_YEAR, norm=nYR, lw=0, zorder=4, rasterized=True)
            info[title] = dict(scheme=s, years_scored=y_scored, rows_not_scored=int((~sc_).sum()))
        else:
            mx.scatter(cell.lon, cell.lat, s=3.0, color=INK, lw=0, zorder=4, rasterized=True)
            info[title] = dict(scheme=s, test="all Lena Delta cell-years", training_rows_other_regions={k_: int(v) for k_, v in tr_reg.items()})
        T(ax, x + mw / 2, ytop - 0.8, title, ha="center", va="bottom")
        if i == 0:
            km = 50.0; lat_sb = ext[2] + 0.16; dlon = km / (111.32 * np.cos(np.deg2rad(lat_sb))); lon_sb = ext[0] + 0.3
            mx.plot([lon_sb, lon_sb + dlon], [lat_sb, lat_sb], color=INK, lw=0.8, solid_capstyle="butt", zorder=8)
            mx.text(lon_sb + dlon / 2, lat_sb + 0.07, "50 km", ha="center", va="bottom", fontsize=FS_T, zorder=8)
            # fold key: five dots with numbers (random, spatial block, block x year boxes)
            T(ax, x, yk1, "Fold", ha="left", color=AUX)
            xk = x + tw_mm(ax, "Fold", FS_T) + 1.6
            for k in range(5):
                ax.add_patch(Circle((xk, yk1), 0.6, facecolor=FOLD_HEX[k], edgecolor="none", zorder=5))
                T(ax, xk + 1.0, yk1, str(k + 1), ha="left", color=AUX); xk += 1.0 + tw_mm(ax, str(k + 1), FS_T) + 1.9
        if s == "year":
            cax = fig.add_axes(rect(x + 6.0, yk1 - 0.55, mw - 12.0, 1.1))
            cb = fig.colorbar(ScalarMappable(norm=nYR, cmap=CM_YEAR), cax=cax, orientation="horizontal", ticks=[])
            cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(FRAME)
            T(ax, x + 5.2, yk1, str(min(y_scored)), ha="right", color=AUX); T(ax, x + mw - 5.2, yk1, str(max(y_scored)), ha="left", color=AUX)
            T(ax, x + mw / 2, yk2, "Dot = year, grey = not scored", ha="center", color=AUX)
        if s == "block_year":
            T(ax, x + mw / 2, yk1, "Box = block fold", ha="center", color=AUX)
            T(ax, x + mw / 2, yk2, "Dot = year", ha="center", color=AUX)
        if s == "region":
            T(ax, x + mw / 2, yk1, "Test: all Lena cells", ha="center", color=AUX)
            T(ax, x + mw / 2, yk2, "Training: Alaska, Canada", ha="center", color=AUX)
    x_maps_r = X0 + 5 * mw + 4 * gap
    # ---------- (ii) yearly outputs: ALT anomaly maps of two years (Alaska)
    TH2 = 15.5; TW2 = TH2 * A["wh"]
    xo = x_maps_r + 7.0
    T(ax, xo, yl, "Yearly ALT anomaly, Alaska (cm)", ha="left")
    nA = TwoSlopeNorm(0.0, -15.0, 15.0)
    for k, ay in enumerate(A["anom_years"]):
        xx = xo + k * (TW2 + 1.8)
        ia = thumb(fig, xx, ytop, TW2, TH2); alaska_img(ia, A, A["ras"][f"anom_{ay}"], CM_DIV, nA)
        T(ax, xx + TW2 / 2, ytop - 0.8, str(ay), ha="center", va="bottom")
    xcb = xo + 2 * TW2 + 1.8 + 1.4
    cax = fig.add_axes(rect(xcb, ytop, 1.2, TH2))
    cb = fig.colorbar(ScalarMappable(norm=nA, cmap=CM_DIV), cax=cax, orientation="vertical", ticks=[-15, 0, 15])
    cb.outline.set_linewidth(0.45); cb.outline.set_edgecolor(FRAME)
    cax.tick_params(labelsize=FS_T, width=0.45, length=1.5, pad=1.0, labelcolor=TICK); cax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    SRC["c"] = dict(validation=dict(inputs=["polar.testbed_v5 load_v5 -> with_forcing('year_obs') -> usable_mask -> region == 'Lena' -> make_folds(k = 5, seed = 0)",
                                            "checked against data/processed/v5/runs/core_v6/preds/Lena__<scheme>__year_obs__F8TS__P1__-__lam-1.csv.gz and ALL__region__...: %s" % check],
                                    n_rows=int(len(dr)), n_cells=int(dr.loc_id.nunique()), n_blocks=int(dr.block.nunique()), years=sorted(int(v) for v in dr.year.unique()),
                                    panels=info, fold_palette=FOLD_HEX, year_cmap="cmcrameri bilbao_r 0.25-0.95 over the scored years", map_mm=[mw, round(mh, 2)],
                                    encoding="random: dot colour = fold of the cell-year; spatial block: 0.5 deg box edge and dot colour = block fold; year hold-out: dot = year (grey = year with < 20 cell-years, not scored); block x year: box = block fold, dot = year; region hold-out: all Lena Delta cell-years are test, training = Alaska and Canada (NW) cell-years more than 100 km from any Lena Delta cell"),
                    anomaly=dict(years=A["anom_years"], definition="r1(year) - mean_2011_2024 r1 per 1 km cell, box means 0.05 x 0.02 deg", range_cm=[-15, 15],
                                 inputs=["data/processed/v5/maps/Alaska/pred_<year>.csv.gz (r1)", "data/processed/v5/maps/Alaska/mean_2011_2024.csv.gz (r1)"],
                                 stats={str(ay): dict(mean=round(A["stats"][f"anom_{ay}"]["mean"], 2), p1=round(A["stats"][f"anom_{ay}"]["p1"], 2), p99=round(A["stats"][f"anom_{ay}"]["p99"], 2)) for ay in A["anom_years"]}))
    return out


# ================================================================ QA
def _all_axes(fig):
    out = []
    def rec(a):
        out.append(a); [rec(c) for c in a.child_axes]
    for a in fig.axes:
        rec(a)
    return out


def collect_texts(fig):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); items = []
    for ax in _all_axes(fig):
        ts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                loc = tk.get_loc()
                if loc is None or not (lo - 1e-9 <= loc <= hi + 1e-9):
                    continue
                for lab in (tk.label1, tk.label2):
                    if lab.get_visible() and lab.get_text().strip():
                        ts.append(lab)
        for t in ts:
            if t.get_visible() and t.get_text().strip():
                items.append((t.get_text(), t.get_window_extent(r), t.get_fontsize(), t.get_fontname(), ax, t))
    return items


def qa(fig, canvas_ax):
    items = collect_texts(fig); r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height; ppt = fig.dpi / 72.0
    bad = []
    for name, bb, fs, fn, ax, t in items:
        if fs < 6.0 - 1e-9:
            bad.append(f"size {fs} < 6: '{name}'")
        if fn != "FreeSans":
            bad.append(f"font {fn}: '{name}'")
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            bad.append(f"outside canvas: '{name}'")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            w = min(a.x1, b.x1) - max(a.x0, b.x0); h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > 0.5 and h > 0.5:
                bad.append(f"text overlap: '{items[i][0]}' x '{items[j][0]}'")
    frames = [a for a in _all_axes(fig) if a is not canvas_ax]
    for name, bb, fs, fn, ax, t in items:
        for fr in frames:
            if ax is fr or (ax is not None and ax in fr.child_axes):
                continue
            fb = fr.get_window_extent(r)
            w = min(bb.x1, fb.x1) - max(bb.x0, fb.x0); h = min(bb.y1, fb.y1) - max(bb.y0, fb.y0)
            if w > 0.5 and h > 0.5:
                bad.append(f"text over frame: '{name}'")
    m = 0.5 * ppt
    for name, bb, fs, fn, ax, t in items:
        if ax is not canvas_ax:
            continue
        for ln in canvas_ax.lines:
            xy = ln.get_xydata()
            seg = [np.c_[np.linspace(x0, x1, 80), np.linspace(y0, y1, 80)] for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:])] or [xy]
            pts = canvas_ax.transData.transform(np.concatenate(seg))
            if ((pts[:, 0] >= bb.x0 - m) & (pts[:, 0] <= bb.x1 + m) & (pts[:, 1] >= bb.y0 - m) & (pts[:, 1] <= bb.y1 + m)).any():
                bad.append(f"text touches line: '{name}'")
                break
        for p in canvas_ax.patches:
            pb = p.get_window_extent(r)
            if not (bb.x1 + m < pb.x0 or bb.x0 - m > pb.x1 or bb.y1 + m < pb.y0 or bb.y0 - m > pb.y1):
                inside = pb.x0 <= bb.x0 and bb.x1 <= pb.x1 and pb.y0 <= bb.y0 and bb.y1 <= pb.y1
                if not inside:
                    bad.append(f"text touches patch: '{name}'")
    sizes = sorted({round(float(it[2]), 2) for it in items}); fonts = sorted({it[3] for it in items})
    print(f"[QA] texts {len(items)}, sizes {sizes}, fonts {fonts}, problems {len(bad)}" + "".join("\n   " + b for b in bad))
    return dict(n_texts=len(items), font_sizes_pt=sizes, fonts=fonts, problems=bad)


def save(fig, final):
    if not final:
        p = CACHE / f"{STEM}_draft.png"; fig.savefig(p, dpi=300, facecolor="white"); return [p]
    OUT.mkdir(parents=True, exist_ok=True); out = []
    for ext in ("pdf", "svg", "png"):
        p = OUT / f"{STEM}.{ext}"; fig.savefig(p, dpi=600, facecolor="white"); out.append(p)      # 600 dpi also sets the embedded raster resolution of pdf/svg
    return out


def main():
    final = os.environ.get("FIG2_FINAL") == "1" or CACHE is None
    setup()
    meta = json.loads((MAPS / "Alaska" / "meta.json").read_text())
    d = load_labels()
    D = cell_year_design(d)
    SD = stefan_data(d, meta)
    A = alaska_rasters(float(meta["lam"]))
    dr, F, check = lena_folds(d); F["_d_all"] = d
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax = canvas(fig)
    YA, YB, YC = 3.0, 42.0, 97.0
    letter_at(ax, 0.6, 0.8, "a"); row_a(fig, ax, YA, D, SD, meta)
    letter_at(ax, 0.6, YB + 0.8, "b"); row_b(fig, ax, YB, A, meta)
    letter_at(ax, 0.6, YC + 0.8, "c"); row_c(fig, ax, YC, dr, F, check, A)
    SRC["qa"] = qa(fig, ax)
    SRC["figure"] = dict(width_mm=W_MM, height_mm=H_MM, png_dpi=600, font="FreeSans", sizes_pt=dict(text=FS, letter=FS_LETTER, equation=FS_EQ),
                         palette="POLAR_PALETTE=%s" % os.environ.get("POLAR_PALETTE"), script="scripts/4_visualization/paper_v14/fig2_workflow.py",
                         rows=dict(a="cell-year design (soil columns, cumulative TDD) and the Stefan anchor", b="physics-residual model flow, Alaska 2014",
                                   c="validation designs (Lena Delta) and yearly ALT anomaly maps (Alaska)"))
    out = save(fig, final); plt.close(fig)
    if final:
        sd = OUT / "source_data"; sd.mkdir(parents=True, exist_ok=True)
        (sd / f"{STEM}.json").write_text(json.dumps(SRC, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        out.append(sd / f"{STEM}.json")
    for p in out:
        print(p)


if __name__ == "__main__":
    main()
