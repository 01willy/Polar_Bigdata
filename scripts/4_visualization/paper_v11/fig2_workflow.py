# -*- coding: utf-8 -*-
"""Paper v11 Fig. 2 (Scientific Reports main text): cell-year design, physics-residual model, evaluation and outputs. 180 mm wide.

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
Style docs/PAPER_PLAN_v11_2026-10-16.md section 4; design/style_tokens_v4.json color.paper_v11 via polar.style_v5 (POLAR_PALETTE=paper11);
      FreeSans 6-7 pt (#262626), letters 8 pt bold, equation 8 pt; connectors 0.6 pt with open chevrons; operation blocks only (light tint, #808080).
Run   cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper11 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v11/fig2_workflow.py
      (FIG2_CACHE=<dir> writes a 300 dpi draft and a raster cache there instead of the outputs)
Out   outputs/figures/paper/v11/Fig2_workflow.{pdf,svg,png} (180 x 160 mm, png 600 dpi), outputs/figures/paper/v11/source_data/Fig2_workflow.json
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
if os.environ.get("POLAR_PALETTE") != "paper11":
    raise SystemExit("run with POLAR_PALETTE=paper11 (v11 paper palette in polar.style_v5)")
import polar.style_v5 as S  # noqa: E402
import polar.testbed_v5 as TB  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm, to_rgb, to_hex  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Polygon  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter  # noqa: E402
import cartopy.io.shapereader as shpreader  # noqa: E402
import shapely  # noqa: E402
from cmcrameri import cm as cmc  # noqa: E402

PROC = ROOT / "data" / "processed"; V5 = PROC / "v5"; MAPS = V5 / "maps"
OUT = ROOT / "outputs" / "figures" / "paper" / "v11"; STEM = "Fig2_workflow"
CACHE = Path(os.environ["FIG2_CACHE"]) if os.environ.get("FIG2_CACHE") else None
SRC: dict = {}
P11 = S.C["paper_v11"]

# ================================================================ style (docs/PAPER_PLAN_v11_2026-10-16.md section 4)
W_MM, H_MM = 180.0, 160.0
FS = 7.0; FS_S = 6.5; FS_T = 6.0; FS_LETTER = 8.0; FS_EQ = 8.0
INK = P11["text"]                       # #262626
AXIS = P11["axis"]                      # #404040
GRID = P11["grid"]                      # #E8E8E8
ZERO = S.NEUTRAL["zero_line"]           # #8C8C8C
ACCENT = S.METHOD_HEX["R1"]             # #D1603D proposed method
STEFAN = S.METHOD_HEX["P1"]             # #2E6DA4 Stefan (recalibrated)
TRAIN = "#8C8C8C"                       # training-only path (dashed)
EDGE = "#808080"                        # block edges
FRAME = "#A6A6A6"                       # thumbnail frames
RULE = "#BFBFBF"                        # header hairline, table inner rules
LW_LINE = 0.6; LW_BOX = 0.5; LW_AX = 0.5
DASH = (0, (2.6, 1.6))
CHEV_L, CHEV_ANG = 0.95, 32.0           # open chevron: stroke length (mm), half angle (deg)
X0, X1 = 4.8, 179.2                     # content edges (mm)


def tint(hex_, f):
    """Solid mix of a colour with white (f = share of the colour)."""
    c = np.array(to_rgb(hex_)); return to_hex(1 - f * (1 - c))


FACE_STEFAN = tint(STEFAN, 0.13)
FACE_GB = tint(ACCENT, 0.13)
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
ALASKA_ASP = 1.0 / np.cos(np.deg2rad(65.5))
THUMB_LAND = "#E6E6E6"; THUMB_COAST = "#9A9A9A"
LENA_EXT = (123.5, 130.4, 71.35, 72.95); LENA_LAT = 72.15
C_TEST, C_TRAIN, C_EXCL = "#262626", "#8C8C8C", "#9A9A9A"      # excluded = open ring (edge colour)
C_LAND, C_COAST = "#F6F6F6", "#A6A6A6"


def setup():
    S.setup()
    for f in ("/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf", "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_T, "ytick.labelsize": FS_T, "legend.fontsize": FS_S,
        "axes.linewidth": LW_AX, "xtick.major.width": LW_AX, "ytick.major.width": LW_AX, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0, "lines.linewidth": 0.9,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
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
    """Polyline of horizontal / vertical segments with an open chevron at the end. gap=(x, y, half) leaves a crossing gap."""
    pts = np.asarray(pts, float); segs = [pts]
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
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        L = float(np.hypot(x1 - x0, y1 - y0)); ux, uy = (x1 - x0) / L, (y1 - y0) / L
        for sgn in (-1, 1):
            a = np.deg2rad(180 - sgn * CHEV_ANG)
            vx = ux * np.cos(a) - uy * np.sin(a); vy = ux * np.sin(a) + uy * np.cos(a)
            ax.plot([x1, x1 + CHEV_L * vx], [y1, y1 + CHEV_L * vy], color=color, lw=lw, zorder=z + 0.1, solid_capstyle="round")


def dot(ax, x, y, color=AXIS, r=0.42, z=7):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=z))


def block(ax, x, y, w, h, face, text=None, fs=FS, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.45", facecolor=face, edgecolor=EDGE, lw=LW_BOX, zorder=z))
    if text:
        T(ax, x + w / 2, y + h / 2, text, fs=fs, ha="center", va="center", zorder=z + 1)


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
            xs, ys = part.xy; ia.plot(xs, ys, color=THUMB_COAST, lw=0.22, zorder=3)
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


# ================================================================ row headers
def row_header(ax, y, letter, title):
    T(ax, 0.6, y, letter, fs=FS_LETTER, ha="left", va="top", fontweight="bold")
    t = T(ax, X0, y + 0.05, title, fs=FS, ha="left", va="top")
    w = tw_mm(ax, title, FS)
    yc = y + 1.25
    ax.plot([X0 + w + 2.0, X1], [yc, yc], color=RULE, lw=0.4, zorder=1, solid_capstyle="butt")
    return t


# ================================================================ row a
def row_a(fig, ax, y0, D):
    """a: soil-column schematic | cumulative TDD of one cell-year | cell-year table."""
    out = {}
    # ---------- (i) soil columns: observation day (left) and end of season (right)
    cw, gap = 7.8, 7.8
    w_lab = max(tw_mm(ax, "Thaw", FS_S), tw_mm(ax, "depth", FS_S))
    xL = X0 + w_lab + 3.2; xR = xL + cw + gap
    ys, yo, ya, yb = y0 + 11.8, y0 + 17.9, y0 + 24.7, y0 + 33.9         # surface, thaw front on obs. day, permafrost table, bottom
    for (x, segs) in ((xL, ((ys, yo, THAW), (yo, ya, FROZEN), (ya, yb, PERMA))), (xR, ((ys, ya, THAW), (ya, yb, PERMA)))):
        for (a_, b_, c_) in segs:
            ax.add_patch(Rectangle((x, a_), cw, b_ - a_, facecolor=c_, edgecolor="none", zorder=2))
        ax.add_patch(Rectangle((x, ys), cw, yb - ys, facecolor="none", edgecolor=SOIL_EDGE, lw=0.4, zorder=3))
    ax.plot([xL - 0.9, xR + cw + 0.9], [ys, ys], color=INK, lw=0.9, zorder=4, solid_capstyle="butt")       # ground surface
    for xa_, xb_ in ((xL, xL + cw), (xR, xR + cw)):
        ax.plot([xa_, xb_], [ya, ya], color=INK, lw=0.6, zorder=4, solid_capstyle="butt")                 # permafrost table
    ax.plot([xL + cw, xR], [ya, ya], color=INK, lw=0.4, ls=(0, (1.0, 1.0)), zorder=4)
    ax.plot([xL, xL + cw], [yo, yo], color=INK, lw=0.6, ls=(0, (2.0, 1.1)), zorder=4)                     # thaw front, obs. day
    yh = y0 + 7.4
    for x, s, filled in ((xL, "Obs. day", False), (xR, "End of season", True)):
        cx = x + cw / 2; w = tw_mm(ax, s, FS_S) + 1.75
        ax.add_patch(Circle((cx - w / 2 + 0.6, yh), 0.6, facecolor=INK if filled else "white", edgecolor=INK, lw=0.5, zorder=6))
        T(ax, cx - w / 2 + 1.75, yh, s, ha="left")
    dim_v(ax, xL - 1.5, ys, yo)
    T(ax, xL - 2.4, (ys + yo) / 2, "Thaw\ndepth", ha="right", linespacing=1.0)
    dim_v(ax, xR + cw + 1.5, ys, ya)
    xr = xR + cw + 2.8
    T(ax, xr, ys, "Ground surface", ha="left")
    T(ax, xr, (ys + ya) / 2, "ALT", ha="left")
    T(ax, xr, ya, "Permafrost table", ha="left")
    T(ax, xr, (ya + yb) / 2 + 0.6, "Permafrost", ha="left")
    yk = y0 + 37.6; xk = xL - 0.3
    for c_, s in ((THAW, "Thawed"), (FROZEN, "Seasonally frozen")):
        ax.add_patch(Rectangle((xk, yk - 0.85), 2.3, 1.7, facecolor=c_, edgecolor=SOIL_EDGE, lw=0.3, zorder=3))
        T(ax, xk + 2.9, yk, s, ha="left"); xk += 2.9 + tw_mm(ax, s, FS_S) + 3.2
    x_end_i = xr + tw_mm(ax, "Permafrost table", FS_S)

    # ---------- (iii) table geometry first (its width sets the curve width)
    rows = D["rows"]
    cell_txt = {c: f"{rows[rows.loc_id == c].lat.iloc[0]:.2f}°N, {rows[rows.loc_id == c].lon.iloc[0]:.2f}°E" for c in D["cells"]}
    vals = [[None, f"{int(r.year)}", r.date, f"{r.e5_tdd:.0f}", f"{r.alt_cm:.1f}"] for _, r in rows.iterrows()]
    heads = [("Cell", None), ("Year", None), ("Obs. day", None), (r"$\mathrm{TDD_{obs}}$", "(°C d)"), ("ALT", "(cm)")]
    widths = [max(tw_mm(ax, s, FS_S) for s in list(cell_txt.values()) + ["Cell"])]
    for i in range(1, 5):
        cand = [v[i] for v in vals] + [heads[i][0]] + ([heads[i][1]] if heads[i][1] else [])
        widths.append(max(tw_mm(ax, s, FS_S) for s in cand))
    widths[3] += 1.9                                                     # room for the open-circle marker
    gapc = 3.0
    tab_w = sum(widths) + gapc * 4
    xt = X1 - tab_w

    # ---------- (ii) cumulative TDD of one cell-year
    xa = x_end_i + 11.0; wa = xt - 6.5 - xa; ya_, ha = y0 + 10.4, 24.0
    a2 = fig.add_axes(rect(xa, ya_, wa, ha))
    yr = D["year"]; doy, cum = D["doy"], D["cum"]
    X_LO, X_HI = 146.0, 304.0
    for yy, (dd, cc) in sorted(D["others"].items()):
        a2.plot(dd, cc, color="#D6D6D6", lw=0.45, zorder=2, solid_capstyle="butt")
    a2.plot(doy, cum, color=INK, lw=1.0, zorder=4, solid_capstyle="butt")
    a2.axhline(D["t_clim"], color=ZERO, lw=0.7, ls=(0, (3.0, 1.6)), zorder=3)
    od = D["obs_doy"]
    a2.plot([od, od], [0, D["t_obs"]], color=ZERO, lw=0.5, ls=(0, (1.0, 1.0)), zorder=3)
    a2.plot([od], [D["t_obs"]], "o", ms=3.4, mfc="white", mec=INK, mew=0.7, zorder=6)
    a2.plot([D["doy_end"]], [D["t_full"]], "o", ms=3.2, mfc=INK, mec=INK, mew=0.0, zorder=6)
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
    labs = [a2.text(od + 4.0, D["t_obs"] - 30, r"$\mathrm{TDD_{obs}}$", ha="left", va="top", fontsize=FS),
            a2.text(D["doy_end"] + 3.5, D["t_full"] - 34, "Full\nseason", ha="left", va="top", fontsize=FS_S, linespacing=1.0),
            a2.text(X_LO + 3, D["t_clim"] + 28, "Mean 2011–2024", ha="left", va="bottom", fontsize=FS_S),
            a2.text(od + 2.5, 35, doy_label(yr, od), ha="left", va="bottom", fontsize=FS_T),
            a2.text(X_HI - 2, 1185, "Other years", ha="right", va="top", fontsize=FS_S)]
    out["a2"] = a2; out["a2_labels"] = labs

    # ---------- (iii) table
    xs = [xt]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + gapc)
    x_right = X1
    yt0 = y0 + 10.4; rh = 3.75
    T(ax, xt, y0 + 7.4, "Lena Delta cell-years", ha="left", va="center")
    ax.plot([xt, x_right], [yt0, yt0], color=INK, lw=0.6, zorder=4, solid_capstyle="butt")
    yh1 = yt0 + 2.1; yh2 = yh1 + 2.75
    for i, (h1, h2) in enumerate(heads):
        al = "left" if i == 0 else "right"; xx = xs[i] if i == 0 else xs[i] + widths[i]
        T(ax, xx, yh1, h1, ha=al)
        if h2:
            T(ax, xx, yh2, h2, ha=al)
    ym = yh2 + 2.0
    ax.plot([xt, x_right], [ym, ym], color=INK, lw=0.4, zorder=4, solid_capstyle="butt")
    yr0 = ym + 2.5
    for j, v in enumerate(vals):
        yy = yr0 + j * rh
        for i in range(1, 5):
            T(ax, xs[i] + widths[i], yy, v[i], ha="right")
        if j == D["k_row"]:
            xm = xs[3] + widths[3] - tw_mm(ax, v[3], FS_S) - 1.35
            ax.add_patch(Circle((xm, yy), 0.6, facecolor="white", edgecolor=INK, lw=0.5, zorder=6))
    for i, c in enumerate(D["cells"]):
        T(ax, xs[0], yr0 + 2 * i * rh, cell_txt[c], ha="left")
    yl = yr0 + 1.5 * rh
    ax.plot([xt, x_right], [yl, yl], color=RULE, lw=0.35, zorder=4, solid_capstyle="butt")
    yb_ = yr0 + 3.5 * rh
    ax.plot([xt, x_right], [yb_, yb_], color=INK, lw=0.6, zorder=4, solid_capstyle="butt")
    SRC["a"] = dict(
        layout_mm=dict(soil_columns=[round(xL, 2), round(xR + cw, 2)], curve_axes=[round(xa, 2), round(xa + wa, 2)], table=[round(xt, 2), round(x_right, 2)]),
        soil_column="vector schematic, no data: two columns (observation day, end of season); thawed %s, seasonally frozen %s, permafrost %s; dashed line = thaw front on the observation day; solid = permafrost table" % (THAW, FROZEN, PERMA),
        curve=dict(cell=D["cell"], lat=D["lat"], lon=D["lon"], year=yr, observation_day=dict(doy=D["obs_doy"], date=doy_label(yr, D["obs_doy"])),
                   tdd_obs_degC_day=round(D["t_obs"], 2), tdd_full_season_degC_day=round(D["t_full"], 2), tdd_mean_2011_2024_degC_day=round(D["t_clim"], 2),
                   season_end_doy=D["doy_end"], season_end_date=doy_label(yr, D["doy_end"]), month_end_cumulative_tdd=[round(float(v), 2) for v in D["cum_m"]],
                   light_lines="the same cell's daily cumulative TDD in the other years 2011-2024 (%d lines)" % len(D["others"]),
                   inputs=["data/processed/v5/forcing_year.csv (tdd_cum_1..12, e5_tdd)", "data/processed/v5/forcing_clim.csv (clim1124_tdd via polar.testbed_v5)",
                           "data/processed/v5/labels_cell_year_v5.csv (obs_doy)"],
                   rule="daily linear interpolation of month-end cumulative TDD (0 on 1 January); TDD_obs equals polar.testbed_v5.with_forcing('year_obs') e5_tdd (checked < 0.5 degC d)"),
        table=dict(rows=[dict(cell=int(r.loc_id), lat=round(float(r.lat), 4), lon=round(float(r.lon), 4), year=int(r.year), obs_doy=float(r.obs_doy), obs_date=r.date,
                              tdd_obs_degC_day=round(float(r.e5_tdd), 2), tdd_obs_printed=f"{r.e5_tdd:.0f}", alt_cm=round(float(r.alt_cm), 2), alt_printed=f"{r.alt_cm:.1f}",
                              n_obs=int(r.n_obs), source=str(r.source), tdd_obs_fraction_of_full_season=round(float(r.tdd_obs_frac), 3)) for _, r in rows.iterrows()],
                   years_rule="coldest and warmest year 2011-2024 by mean full-season TDD over all Lena Delta label cells (%d, %d)" % D["years"],
                   year_mean_full_season_tdd={k: round(v, 1) for k, v in D["year_mean_tdd"].items()},
                   cells_rule="Lena Delta cells with dated observations in both years; the two with the most label years (ties: smaller cell id)",
                   n_label_years=D["n_years"], curve_row_rule="row with the smallest TDD_obs / full-season TDD ratio (marked with the open circle)",
                   obs_day_note="obs_doy is the mean day of the probe observations aggregated into the cell-year; printed as the rounded calendar date"))
    return out


# ================================================================ row b
def eq_alt_hat(ax, x, y, fs, color=INK):
    """'ALT' with a thin drawn hat (mathtext \\widehat renders an oversized wedge in FreeSans). Baseline at y (mm). Returns the right edge (mm)."""
    t = ax.text(x, y, r"$\mathrm{ALT}$", fontsize=fs, ha="left", va="baseline", color=color, zorder=6)
    r = ax.figure.canvas.get_renderer(); bb = t.get_window_extent(r)
    inv = ax.transData.inverted(); (xa, yt), (xb, _) = inv.transform([(bb.x0, bb.y1), (bb.x1, bb.y1)])
    cap = 0.716 * fs * 25.4 / 72.0                          # FreeSans cap height (mm)
    ytop = y - cap; w = xb - xa; xm = (xa + xb) / 2
    ax.plot([xm - 0.30 * w, xm, xm + 0.30 * w], [ytop - 0.35, ytop - 1.05, ytop - 0.35], color=color, lw=0.5, zorder=6,
            solid_joinstyle="miter", solid_capstyle="butt")
    return xb, t


def row_b(fig, ax, y0, SD, A, meta):
    out = {}
    lam = float(meta["lam"]); E = SD["E"]
    # ---------- (i) Stefan anchor
    xs0, ws, ys0, hs = X0 + 9.4, 35.5, y0 + 10.6, 32.0
    a1 = fig.add_axes(rect(xs0, ys0, ws, hs))
    a1.scatter(SD["x"], SD["y"], s=0.75, color="#CCCCCC", lw=0, zorder=2, rasterized=True)
    XM, YM = 50.0, 200.0
    xx = np.array([0.0, XM]); a1.plot(xx, E * xx, color=STEFAN, lw=1.1, zorder=4, solid_capstyle="butt")
    P = SD["picks"]
    for (_, r), rr in zip(P.iterrows(), SD["r_picks"]):
        a1.plot([r.e5_sqrt_tdd, r.e5_sqrt_tdd], [E * r.e5_sqrt_tdd, r.alt_cm], color=ACCENT, lw=0.9, zorder=5, solid_capstyle="butt")
        a1.plot([r.e5_sqrt_tdd], [r.alt_cm], "o", ms=2.6, mfc=INK, mec="white", mew=0.35, zorder=6)
    a1.set_xlim(0, XM); a1.set_ylim(0, YM)
    a1.xaxis.set_major_locator(FixedLocator([0, 25, 50])); a1.xaxis.set_minor_locator(MultipleLocator(5))
    a1.yaxis.set_major_locator(FixedLocator([0, 100, 200])); a1.yaxis.set_minor_locator(MultipleLocator(25))
    for k in ("top", "right"):
        a1.spines[k].set_visible(False)
    a1.set_xlabel(r"$\sqrt{\mathrm{TDD_{obs}}}$ ($\sqrt{\mathrm{°C\,d}}$)", labelpad=1.5)
    a1.set_ylabel("Observed ALT (cm)", labelpad=2.0)
    # in-plot key (upper left, no data there): sample size, Stefan line with E, residual
    tr = a1.transAxes; xk0, xk1, xt_ = 0.035, 0.115, 0.145
    yk = [0.965, 0.855, 0.745, 0.635]
    labs = [a1.text(xk0, yk[0], "Alaska, n = %s" % f"{SD['n']:,}", transform=tr, ha="left", va="center", fontsize=FS_S),
            a1.text(xt_, yk[1], r"$\hat{E}\sqrt{\mathrm{TDD_{obs}}}$", transform=tr, ha="left", va="center", fontsize=FS_S),
            a1.text(xt_, yk[2], r"$\hat{E}$ = %.2f cm (°C d)$^{-1/2}$" % E, transform=tr, ha="left", va="center", fontsize=FS_S),
            a1.text(xt_, yk[3], "Residual", transform=tr, ha="left", va="center", fontsize=FS_S)]
    a1.plot([xk0, xk1], [yk[1], yk[1]], transform=tr, color=STEFAN, lw=1.1, solid_capstyle="butt", zorder=6, clip_on=False)
    a1.plot([(xk0 + xk1) / 2] * 2, [yk[3] - 0.045, yk[3] + 0.045], transform=tr, color=ACCENT, lw=0.9, solid_capstyle="butt", zorder=6, clip_on=False)
    n_off = int((SD["y"] > YM).sum())
    if n_off:
        labs.append(a1.text(XM, YM + 4, f"{n_off} > {YM:g} cm", ha="right", va="bottom", fontsize=FS_T, clip_on=False))
    out["a1"] = a1; out["a1_labels"] = labs
    T(ax, X0, y0 + 7.4, "Stefan anchor", ha="left")

    # ---------- (ii) map flow, Alaska 2014
    yr = A["year"]
    TH = 10.8; TW = TH * A["wh"]
    yU, yL = y0 + 22.0, y0 + 40.6
    xf0 = 63.0
    g_in, g_grp = 1.9, 3.6
    xs_L = [xf0, xf0 + TW + g_in, xf0 + 2 * TW + g_in + g_grp, xf0 + 3 * TW + 2 * g_in + g_grp]
    x_in_r = xs_L[-1] + TW
    xs_U = [xf0, xf0 + TW + g_in, xf0 + 2 * (TW + g_in)]
    pct = {"dem_slope": (0, 98)}
    CLIM = [("swe", "Snow"), ("fdd", "FDD"), ("e5_tdd", "TDD")]
    TERR = [("dem_elev", "Elevation"), ("dem_slope", "Slope")]
    SOIL = [("sg_soc_0_5", "Soil carbon"), ("sg_clay_5_15", "Clay")]
    norms = {}
    for (key, nm), x, cmap, yc in [(c, x, CM_CLIM, yU) for c, x in zip(CLIM, xs_U)] + [(c, x, CM_TERR, yL) for c, x in zip(TERR, xs_L[:2])] + \
                                   [(c, x, CM_SOIL, yL) for c, x in zip(SOIL, xs_L[2:])]:
        ia = thumb(fig, x, yc - TH / 2, TW, TH)
        lo, hi = np.nanpercentile(A["ras"][key], pct.get(key, (2, 98))); norms[key] = (float(lo), float(hi))
        alaska_img(ia, A, A["ras"][key], cmap, Normalize(lo, hi))
        T(ax, x + TW / 2, yc + TH / 2 + 0.75, nm, fs=FS_T, ha="center", va="top")
    feats = list(meta["feats"]); n_clim = sum(f in TB.F8 for f in feats); n_terr = sum(f in TB.TERRAIN for f in feats); n_soil = sum(f in TB.SOIL for f in feats)
    assert n_clim + n_terr + n_soil == len(feats) == 23
    T(ax, xs_U[0], yU - TH / 2 - 0.9, f"Climate {yr}, {n_clim} variables", ha="left", va="bottom")
    T(ax, xs_L[0], yL - TH / 2 - 0.9, f"Terrain, {n_terr} variables", ha="left", va="bottom")
    T(ax, xs_L[2], yL - TH / 2 - 0.9, f"Soil, {n_soil} variables", ha="left", va="bottom")
    # input colour key: low -> high for the three groups
    yk0 = yL + TH / 2 + 4.6; kw_, kh_ = 7.0, 0.75
    T(ax, xf0, yk0 + 1.0, "Low", fs=FS_T, ha="left", va="center")
    xk = xf0 + tw_mm(ax, "Low", FS_T) + 1.0
    for i, cm_ in enumerate((CM_CLIM, CM_TERR, CM_SOIL)):
        kax = fig.add_axes(rect(xk, yk0 + i * (kh_ + 0.25), kw_, kh_))
        kax.imshow(np.linspace(0, 1, 128)[None, :], aspect="auto", cmap=cm_, extent=(0, 1, 0, 1)); kax.set_axis_off()
    T(ax, xk + kw_ + 1.0, yk0 + 1.0, "High", fs=FS_T, ha="left", va="center")
    # bracket x (all inputs) -> gradient boosting; TDD -> Stefan (crosses the bracket with a gap)
    xbr = x_in_r + 2.2; ytb, ybb = yU - TH / 2, yL + TH / 2
    for (ya_, yb_) in ((ytb, yU - 0.9), (yU + 0.9, ybb)):
        ax.plot([xbr, xbr], [ya_, yb_], color=AXIS, lw=LW_LINE, zorder=5, solid_capstyle="butt")
    for yy in (ytb, ybb):
        ax.plot([xbr - 0.9, xbr], [yy, yy], color=AXIS, lw=LW_LINE, zorder=5, solid_capstyle="butt")
    bx, bw, bh = xbr + 5.2, 15.5, 7.0
    block(ax, bx, yU - bh / 2, bw, bh, FACE_STEFAN, r"$\hat{E}\sqrt{\mathrm{TDD}}$", fs=FS)
    block(ax, bx, yL - bh / 2, bw, bh, FACE_GB, r"$g(\mathbf{x})$", fs=FS)
    T(ax, bx + bw / 2, yU - bh / 2 - 0.8, "Stefan", ha="center", va="bottom")
    T(ax, bx + bw / 2, yL - bh / 2 - 0.8, "Gradient boosting", ha="center", va="bottom")
    line(ax, [(xs_U[2] + TW, yU), (bx, yU)])
    T(ax, (xs_U[2] + TW + xbr) / 2, yU - 0.7, "Full season", fs=FS_T, ha="center", va="bottom")
    line(ax, [(xbr, yL), (bx, yL)])
    # training target (dashed, training only)
    xg = bx + bw / 2; yt = yL + bh / 2 + 5.0
    line(ax, [(xg, yt), (xg, yL + bh / 2)], color=TRAIN, ls=DASH)
    T(ax, xg, yt + 0.5, "Training target", ha="center", va="top")
    # outputs
    ox = bx + bw + 5.5
    lo_alt = float(5 * math.floor(min(A["stats"]["p1"]["p1"], A["stats"]["r1"]["p1"]) / 5))
    hi_alt = float(5 * math.ceil(max(A["stats"]["p1"]["p99"], A["stats"]["r1"]["p99"]) / 5))
    nALT = Normalize(lo_alt, hi_alt)
    vres = float(5 * math.ceil(A["stats"]["resid"]["abs_p99"] / 5)); nRES = TwoSlopeNorm(0.0, -vres, vres)
    line(ax, [(bx + bw, yU), (ox, yU)])
    line(ax, [(bx + bw, yL), (ox, yL)], color=ACCENT)
    ia = thumb(fig, ox, yU - TH / 2, TW, TH, edge=STEFAN, lw=0.6); alaska_img(ia, A, A["ras"]["p1"], CM_ALT, nALT)
    T(ax, ox + TW / 2, yU + TH / 2 + 0.75, f"Stefan, {yr}", fs=FS_T, ha="center", va="top")
    ia = thumb(fig, ox, yL - TH / 2, TW, TH, edge=ACCENT, lw=0.6); alaska_img(ia, A, A["ras"]["resid"], CM_DIV, nRES)
    T(ax, ox + TW / 2, yL + TH / 2 + 0.75, f"Residual, {yr}", fs=FS_T, ha="center", va="top")
    xp = ox + TW + 5.2; rp_ = 1.45
    line(ax, [(ox + TW, yU), (xp - rp_, yU)])
    line(ax, [(ox + TW, yL), (xp, yL), (xp, yU + rp_)], color=ACCENT)
    T(ax, xp + 1.0, (yU + yL) / 2 + 1.5, r"$\lambda$", fs=FS, ha="left")
    op_node(ax, xp, yU, rp_)
    gx = xp + rp_ + 4.0
    line(ax, [(xp + rp_, yU), (gx, yU)], color=ACCENT)
    ia = thumb(fig, gx, yU - TH / 2, TW, TH, edge=ACCENT, lw=0.9); alaska_img(ia, A, A["ras"]["r1"], CM_ALT, nALT)
    T(ax, gx + TW / 2, yU + TH / 2 + 0.75, f"ALT {yr}", fs=FS_T, ha="center", va="top")
    tk = [t for t in range(0, 200, 10) if lo_alt <= t <= hi_alt]; tk = tk[::2] if len(tk) > 4 else tk
    vcax = fig.add_axes(rect(gx + TW + 1.3, yU - TH / 2, 1.1, TH))
    cb = fig.colorbar(ScalarMappable(norm=nALT, cmap=CM_ALT), cax=vcax, orientation="vertical", ticks=tk)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(FRAME)
    vcax.tick_params(labelsize=FS_T, width=0.4, length=1.2, pad=0.8)
    T(ax, gx + TW + 1.85, yU - TH / 2 - 0.6, "cm", fs=FS_T, ha="center", va="bottom")
    hcax = fig.add_axes(rect(ox - 0.2, yL + TH / 2 + 4.0, TW + 0.4, 1.0))
    cb2 = fig.colorbar(ScalarMappable(norm=nRES, cmap=CM_DIV), cax=hcax, orientation="horizontal", ticks=[-vres, 0, vres])
    cb2.outline.set_linewidth(0.4); cb2.outline.set_edgecolor(FRAME)
    hcax.tick_params(labelsize=FS_T, width=0.4, length=1.2, pad=0.8); hcax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    T(ax, ox + TW + 1.0, yL + TH / 2 + 4.5, "cm", fs=FS_T, ha="left", va="center")
    # equation (training: TDD to the observation day; maps: full-season TDD)
    eq1 = r"$= \hat{E}\,\sqrt{\mathrm{TDD_{obs}}} + \lambda\,g(\mathbf{x})$"
    eq2 = r"$g \,\approx\, \mathrm{ALT} - \hat{E}\,\sqrt{\mathrm{TDD_{obs}}}$" + ",   " + r"$\lambda$ = %g" % lam
    w0 = tw_mm(ax, r"$\mathrm{ALT}$", FS_EQ); w1 = tw_mm(ax, eq1, FS_EQ); w2 = tw_mm(ax, eq2, FS); sp = 8.0
    xe0 = xf0; ye = y0 + 8.6
    xa_, t0 = eq_alt_hat(ax, xe0, ye, FS_EQ)
    t1 = T(ax, xa_, ye, eq1, fs=FS_EQ, ha="left", va="baseline")
    t2 = T(ax, xa_ + w1 + sp, ye, eq2, fs=FS, ha="left", va="baseline")
    SRC["b"] = dict(
        stefan_anchor=dict(region="Alaska", n_cell_years=SD["n"], label_years=f"{SD['years'][0]}-{SD['years'][-1]}", E_cm_per_sqrt_degC_day=round(E, 4),
                           E_printed=f"{E:.2f}", E_rule="least squares through the origin on all Alaska labels (polar.testbed_v5.ls_scale), equal to data/processed/v5/maps/Alaska/meta.json E",
                           x="sqrt(TDD to the observation day), polar.testbed_v5.with_forcing('year_obs')", axis_range=dict(x=[0, XM], y=[0, YM]),
                           n_above_y_range=int((SD["y"] > YM).sum()), residual_sd_cm=round(SD["sd"], 2),
                           highlighted=[dict(cell=int(r.loc_id), year=int(r.year), sqrt_tdd_obs=round(float(r.e5_sqrt_tdd), 2), alt_cm=round(float(r.alt_cm), 1),
                                             residual_cm=round(float(rr), 1)) for (_, r), rr in zip(P.iterrows(), SD["r_picks"])],
                           highlight_rule="at the 10/30/50/70/90th percentiles of sqrt(TDD_obs): among the 400 nearest cell-years, the one whose residual is closest to +1, -1, +1, -1, +1 residual s.d."),
        flow=dict(year=yr, inputs={"climate (cmcrameri lajolla)": [c[1] for c in CLIM], "terrain (grayC)": [c[1] for c in TERR], "soil (bamako)": [c[1] for c in SOIL]},
                  n_features=dict(climate=n_clim, terrain=n_terr, soil=n_soil, total=len(feats)), features=feats, input_stretch="2-98th percentile per thumbnail (slope 0-98th); darker = larger (key: Low-High ramps)",
                  input_ranges=norms, stefan_map="p1 = E sqrt(full-season TDD %d)" % yr, residual_map="(r1 - p1) / lambda = residual prediction before weighting",
                  alt_map="r1 = p1 + lambda * residual (seed mean of 3 CatBoost models trained on all Alaska labels)", alt_range_cm=[lo_alt, hi_alt], alt_ticks_cm=tk,
                  residual_range_cm=[-vres, vres], lam=lam, learner=meta["learner"], seeds=meta["seeds"], n_train=int(meta["n_train"]), n_map_cells=A["n_cells"],
                  inputs_files=["data/processed/v5/maps/Alaska/pred_%d.csv.gz (e5_tdd, p1, r1)" % yr, A["forcing_src"] + " (e5_fdd, e5_swe)",
                                "data/processed/map_alaska/alaska_grid_x25_v1.csv.gz (dem_elev, dem_slope, sg_soc_0_5, sg_clay_5_15, land)",
                                "data/processed/v5/maps/Alaska/meta.json (E, lam, feats, learner, seeds, n_train)"],
                  raster="1 km map cells averaged in 0.05 deg x 0.02 deg boxes"),
        equation=dict(main="ALT_hat = E_hat sqrt(TDD_obs) + lambda g(x)", target="g ~ ALT - E_hat sqrt(TDD_obs) (training only)", lam=lam,
                      note="maps use the full-season TDD in place of TDD_obs; the hat over ALT is drawn as a thin vector chevron"))
    return out


def product_example(d):
    """One Lena Delta label cell-year with the four yearly products (Wei 2026, Liu 2024, ESA CCI v4, v5) and the physics + residual ML out-of-fold value.
    Candidates: Lena cell-years with all four products and TDD_obs >= 90 % of the full-season TDD (products give end-of-season ALT).
    Choice: the cell-year at the lower median of the absolute out-of-fold error of physics + residual ML (a typical case, not the best one)."""
    L = d[d.region == "Lena"][["cell_year_id", "loc_id", "year", "lat", "lon", "obs_doy", "tdd_obs_frac", "alt_cm"]]
    fp = V5 / "runs" / "core_v6" / "preds" / "Lena__block__year_obs__F8TS__R1__catboost__lam0.5.csv.gz"
    p = pd.read_csv(fp)
    r1 = p.groupby("cell_year_id").agg(r1=("pred", "mean"), y=("y", "first"), fold=("fold", "first"), n_seed=("seed", "nunique")).reset_index()
    pr = pd.read_csv(V5 / "products_cell_year_v5.csv")
    keys = ["wei", "liu", "cci4y", "cci5y"]
    w = pr[pr["product"].isin(keys)].pivot_table(index=["loc_id", "year"], columns="product", values="value_cm").reset_index()
    m = L.merge(r1, on="cell_year_id").merge(w, on=["loc_id", "year"], how="left").dropna(subset=keys)
    assert (np.abs(m.y - m.alt_cm) < 1e-6).all()
    c = m[m.tdd_obs_frac >= 0.9].copy(); c["ae"] = (c.r1 - c.alt_cm).abs()
    med_lo = float(np.percentile(c.ae, 50, method="lower"))
    row = c[np.isclose(c.ae, med_lo)].sort_values(["year", "loc_id"]).iloc[0]
    return dict(row=row, n_candidates=int(len(c)), n_all_products=int(len(m)), median_lower=med_lo, keys=keys, pred_file=str(fp.relative_to(ROOT)))


# ================================================================ row c
def lena_map(fig, x, y_top, mw, mh):
    ne = ne_layers("lena"); asp = 1.0 / np.cos(np.deg2rad(LENA_LAT)); ext = LENA_EXT
    mx = fig.add_axes(rect(x, y_top, mw, mh))
    for g in ne["land"]:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.exterior.xy; mx.fill(xs, ys, facecolor=C_LAND, edgecolor="none", lw=0, zorder=1)
    for g in ne["coastline"]:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.xy; mx.plot(xs, ys, color=C_COAST, lw=0.25, zorder=2)
    mx.set_xlim(ext[0], ext[1]); mx.set_ylim(ext[2], ext[3]); mx.set_aspect(asp); mx.set_xticks([]); mx.set_yticks([])
    for sp in mx.spines.values():
        sp.set_visible(True); sp.set_edgecolor(FRAME); sp.set_linewidth(0.45)
    return mx


def ladder_design():
    """Label numbers of the two experiments (data/processed/v5/runs/ladder_v6/ladder_metrics.csv): LL = label-number ladder, TL = new-region transfer."""
    m = pd.read_csv(V5 / "runs" / "ladder_v6" / "ladder_metrics.csv", usecols=["exp", "n_label"])
    def order(v):
        return [s for s in sorted(set(v), key=lambda s: (s == "all", float(s) if s != "all" else 0.0))]
    return {e: order(m[m.exp == e].n_label.astype(str)) for e in ("LL", "TL")}


def mini_curve(fig, x, y_top, w, h, steps, open_first=False):
    """Schematic RMSE-against-labels curve: one marker per design step (equal spacing), generic decreasing shape, no values."""
    m = fig.add_axes(rect(x, y_top, w, h))
    n = np.arange(float(len(steps))); yv = 1.0 + 2.0 * np.exp(-n / (0.32 * len(steps)))
    m.plot(n, yv, color=INK, lw=0.8, zorder=3, solid_capstyle="butt")
    if open_first:
        m.plot(n[1:], yv[1:], "o", ms=2.3, mfc=INK, mec="white", mew=0.3, zorder=4)
        m.plot(n[:1], yv[:1], "o", ms=2.5, mfc="white", mec=INK, mew=0.6, zorder=4)
    else:
        m.plot(n, yv, "o", ms=2.3, mfc=INK, mec="white", mew=0.3, zorder=4)
    m.set_xlim(-0.6, n[-1] + 0.6); m.set_ylim(0.6, 3.4)
    m.set_xticks([0, n[-1]]); m.set_xticklabels([steps[0], "All" if steps[-1] == "all" else steps[-1]]); m.set_yticks([])
    m.tick_params(axis="x", length=1.5, pad=1.0, labelsize=FS_T)
    for k in ("top", "right"):
        m.spines[k].set_visible(False)
    return m


def structure_glyphs(ax, x0, x1, y0):
    """Four compared structures (compact), each with its line key (Figs 3 and 4 styles). Slot widths follow the content; spare width goes to the gaps."""
    names = ["Direct ML", "Physics-input ML", "Physics pseudo-labels", "Physics + residual ML"]
    keys = ["D0", "F1", "D1", "R1"]
    bw, bh = 7.6, 4.4; sh = 3.4; yc = y0 + 8.4; ys = yc - 6.2
    top = {"F1": "Stefan, Kudryavtsev", "D1": "Stefan", "R1": "Stefan"}
    k_len, k_gap = 2.8, 0.7
    w_y = tw_mm(ax, r"$\hat{y}$", FS_S)
    glyph_w = {k: 4.6 + bw + (4.8 + 1.15 * 2 + 3.0 + 0.5 if k == "R1" else 3.7) + w_y for k in keys}
    top_w = {k: tw_mm(ax, top[k], FS_T) + 2.0 for k in top}
    lead = {k: max(0.0, top_w.get(k, 0) / 2 - (4.6 + bw / 2)) if k != "R1" else 0.0 for k in keys}   # room on the left for a wide top block
    slot = {k: lead[k] + max(glyph_w[k], k_len + k_gap + tw_mm(ax, nm, FS_S)) for k, nm in zip(keys, names)}
    g = (x1 - x0 - sum(slot.values())) / 3.0
    assert g >= 2.5, ("structure slots too wide", g)
    xg = x0
    for nm, k in zip(names, keys):
        xg0 = xg + lead[k]
        T(ax, xg0 + 1.2, yc, r"$\mathbf{x}$", fs=FS_S, ha="right")
        xb = xg0 + 4.6
        line(ax, [(xg0 + 1.9, yc), (xb, yc)])
        block(ax, xb, yc - bh / 2, bw, bh, "#F2F2F2", "ML", fs=FS_T)
        xe = xb + bw; cx = xb + bw / 2
        if k == "R1":
            xp = xe + 4.8; r = 1.15; sw = top_w[k]
            line(ax, [(xe, yc), (xp - r, yc)], color=ACCENT)
            T(ax, (xe + xp - r) / 2, yc - 0.7, r"$\lambda$", fs=FS_T, ha="center", va="bottom")
            op_node(ax, xp, yc, r)
            block(ax, xp - sw / 2, ys - sh / 2, sw, sh, FACE_STEFAN, top[k], fs=FS_T)
            line(ax, [(xp, ys + sh / 2), (xp, yc - r)])
            line(ax, [(xp + r, yc), (xp + r + 3.0, yc)])
            T(ax, xp + r + 3.5, yc, r"$\hat{y}$", fs=FS_S, ha="left")
        else:
            line(ax, [(xe, yc), (xe + 3.2, yc)])
            T(ax, xe + 3.7, yc, r"$\hat{y}$", fs=FS_S, ha="left")
            if k in ("F1", "D1"):
                sw = top_w[k]
                block(ax, cx - sw / 2, ys - sh / 2, sw, sh, FACE_STEFAN, top[k], fs=FS_T)
                if k == "F1":
                    line(ax, [(cx, ys + sh / 2), (cx, yc - bh / 2)])
                else:
                    line(ax, [(cx, ys + sh / 2), (cx, yc - bh / 2)], color=TRAIN, ls=(0, (1.1, 0.8)))
        yn = yc + bh / 2 + 3.0
        col = S.METHOD_HEX[k]; lsk = (0, (2.0, 1.1)) if k == "D0" else "-"
        ax.plot([xg, xg + k_len], [yn, yn], color=col, lw=1.0, ls=lsk, zorder=5, solid_capstyle="butt", dash_capstyle="butt")
        T(ax, xg + k_len + k_gap, yn, nm, fs=FS_S, ha="left")
        xg += slot[k] + g
    return names


def row_c(fig, ax, y0, dr, F, check, A, PX):
    out = {}
    # ---------- (i) validation designs (Lena Delta label cells)
    pick = dict(random="r0", block="b0", year="y2016", block_year="b0y2016")
    show = [("Random", "random"), ("Spatial block", "block"), ("Year hold-out", "year"), ("Block × year", "block_year"), ("Region hold-out", "region")]
    ext = LENA_EXT; asp = 1.0 / np.cos(np.deg2rad(LENA_LAT))
    mw = 21.6; gap = 2.3; mh = mw * (ext[3] - ext[2]) * asp / (ext[1] - ext[0])
    yl = y0 + 7.4; ytop = y0 + 12.6
    T(ax, X0, yl, "Validation designs, Lena Delta", ha="left")
    info = {}
    d_all = F["_d_all"]
    for i, (title, s) in enumerate(show):
        x = X0 + i * (mw + gap)
        mx = lena_map(fig, x, ytop, mw, mh)
        if s in ("block", "block_year"):
            for gl in np.arange(np.ceil(ext[0] / 0.5) * 0.5, ext[1], 0.5):
                mx.plot([gl, gl], [ext[2], ext[3]], color="#D9D9D9", lw=0.3, zorder=1.5)
            for gl in np.arange(np.ceil(ext[2] / 0.5) * 0.5, ext[3], 0.5):
                mx.plot([ext[0], ext[1]], [gl, gl], color="#D9D9D9", lw=0.3, zorder=1.5)
        if s == "region":
            f = F["region"][0]; te_ids = set(d_all.cell_year_id.values[f.te])
            st = np.where(dr.cell_year_id.isin(te_ids), "test", "excluded").astype(object); fold_name = f.name
            tr_reg = d_all.iloc[f.tr].region.value_counts().to_dict()
        else:
            f = [ff for ff in F[s] if ff.name == pick[s]][0]
            st = np.full(len(dr), "excluded", dtype=object); st[f.tr] = "training"; st[f.te] = "test"; fold_name = f.name; tr_reg = None
        q = dr.assign(status=st); rank = {"excluded": 0, "training": 1, "test": 2}
        cell = q.assign(r=q.status.map(rank)).groupby("loc_id").agg(lon=("lon", "first"), lat=("lat", "first"), r=("r", "max"))
        for k, col, z in ((0, C_EXCL, 4.5), (1, C_TRAIN, 4), (2, C_TEST, 5)):
            cc = cell[cell.r == k]
            if k == 0:
                mx.scatter(cc.lon, cc.lat, s=2.6, facecolor="white", edgecolor=C_EXCL, lw=0.35, zorder=z, rasterized=True)
            else:
                mx.scatter(cc.lon, cc.lat, s=2.4, color=col, lw=0, zorder=z, rasterized=True)
        T(ax, x + mw / 2, ytop - 0.8, title, ha="center", va="bottom")
        if i == 0:
            km = 50.0; lat_sb = ext[2] + 0.16; dlon = km / (111.32 * np.cos(np.deg2rad(lat_sb))); lon_sb = ext[0] + 0.3
            mx.plot([lon_sb, lon_sb + dlon], [lat_sb, lat_sb], color=INK, lw=0.8, solid_capstyle="butt", zorder=8)
            mx.text(lon_sb + dlon / 2, lat_sb + 0.07, "50 km", ha="center", va="bottom", fontsize=FS_T, zorder=8)
        info[title] = dict(scheme=s, fold_shown=fold_name, rows=dict(test=int((st == "test").sum()), training=int((st == "training").sum()), excluded=int((st == "excluded").sum())),
                           cells=dict(test=int((cell.r == 2).sum()), training=int((cell.r == 1).sum()), excluded=int((cell.r == 0).sum())),
                           years_tested=sorted(int(v) for v in q[q.status == "test"].year.unique()),
                           **({"training_rows_other_regions": {k_: int(v) for k_, v in tr_reg.items()}} if tr_reg else {}))
    x_maps_r = X0 + 5 * mw + 4 * gap
    xk = x_maps_r
    for nm_, col in (("Excluded", C_EXCL), ("Training", C_TRAIN), ("Test", C_TEST)):
        w = tw_mm(ax, nm_, FS_S); xk -= w
        T(ax, xk, yl, nm_, ha="left")
        if nm_ == "Excluded":
            ax.add_patch(Circle((xk - 1.25, yl), 0.55, facecolor="white", edgecolor=C_EXCL, lw=0.45, zorder=5))
        else:
            ax.add_patch(Circle((xk - 1.25, yl), 0.6, facecolor=col, edgecolor="none", zorder=5))
        xk -= 1.25 + 0.6 + 2.8
    # ---------- (ii) label number, new region (schematic curves at the design steps)
    LD = ladder_design()
    xi0 = x_maps_r + 8.5; wi = (X1 - xi0 - 6.5) / 2
    for j, (lab, steps, xl, op) in enumerate((("Label number", LD["LL"], "Training labels", False), ("New region", LD["TL"], "Target-region labels", True))):
        xi = xi0 + j * (wi + 6.5)
        T(ax, xi, yl, lab, ha="left")
        m = mini_curve(fig, xi + 3.0, ytop, wi - 3.0, mh - 3.6, steps, open_first=op)
        m.set_xlabel(xl, fontsize=FS_T, labelpad=0.8); m.set_ylabel("RMSE", fontsize=FS_T, labelpad=1.2)
        m.text(0.98, 0.97, "Schematic", transform=m.transAxes, ha="right", va="top", fontsize=FS_T, color="#595959")
    # ---------- (iii) compared structures + outputs (second line of row c)
    y2 = ytop + mh + 6.4
    r = PX["row"]
    series = [("Observed", float(r.alt_cm), INK, "D"), ("This study", float(r.r1), ACCENT, "o"), ("Wei 2026", float(r.wei), S.PRODUCT_HEX["Wei"], "o"),
              ("Liu 2024", float(r.liu), S.PRODUCT_HEX["Liu"], "o"), ("ESA CCI v4", float(r.cci4y), S.PRODUCT_HEX["CCI"], "o"),
              ("ESA CCI v5", float(r.cci5y), S.PRODUCT_HEX["CCI5"], "o")]
    w_lab = max(tw_mm(ax, sname, FS_T) for sname, *_ in series)
    wp = 14.0
    w_cm = 1.8 + tw_mm(ax, "cm", FS_T)
    w_gl = max(tw_mm(ax, "Same cell, same year", FS_S), w_lab + 1.2 + wp + w_cm)
    xg0 = X1 - w_gl
    TH2 = 13.0; TW2 = TH2 * A["wh"]; yo_ = y2 + 5.6
    w_an = 2 * TW2 + 1.8 + 1.3 + 1.1 + tw_mm(ax, "\u221215", FS_T) + 1.2
    xo = xg0 - 5.0 - w_an
    xs1 = xo - 4.5
    T(ax, X0, y2, "Compared structures", ha="left")
    structure_glyphs(ax, X0, xs1, y2 + 1.0)
    T(ax, xo, y2, "Yearly ALT anomaly", ha="left")
    nA = TwoSlopeNorm(0.0, -15.0, 15.0)
    for k, ay in enumerate(A["anom_years"]):
        xx = xo + k * (TW2 + 1.8)
        ia = thumb(fig, xx, yo_, TW2, TH2); alaska_img(ia, A, A["ras"][f"anom_{ay}"], CM_DIV, nA)
        T(ax, xx + TW2 / 2, yo_ - 0.6, str(ay), fs=FS_T, ha="center", va="bottom")
    xcb = xo + 2 * TW2 + 1.8 + 1.3
    cax = fig.add_axes(rect(xcb, yo_, 1.1, TH2))
    cb = fig.colorbar(ScalarMappable(norm=nA, cmap=CM_DIV), cax=cax, orientation="vertical", ticks=[-15, 0, 15])
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(FRAME)
    cax.tick_params(labelsize=FS_T, width=0.4, length=1.2, pad=0.8); cax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    T(ax, xcb + 0.55, yo_ - 1.5, "cm", fs=FS_T, ha="center", va="bottom")
    # same cell, same year: one Lena Delta label cell-year, observed ALT, this study (out-of-fold) and four yearly products
    T(ax, xg0, y2, "Same cell, same year", ha="left")
    xpl = X1 - w_cm - wp; rh_ = 2.5; ytp = y2 + 3.0; hp = rh_ * len(series)
    pa = fig.add_axes(rect(xpl, ytp, wp, hp))
    vals = [v for _, v, _, _ in series]
    lo_p = 10.0 * math.floor((min(vals) - 4.0) / 10.0); hi_p = 10.0 * math.ceil((max(vals) + 4.0) / 10.0)
    pa.axvline(float(r.alt_cm), color=ZERO, lw=0.5, ls=(0, (1.6, 1.2)), zorder=1)
    for i, (sname, v, col, mk) in enumerate(series):
        yy_ = len(series) - 1 - i
        pa.plot([v], [yy_], marker=mk, ms=3.0 if mk == "o" else 2.8, mfc=col, mec="white" if mk == "o" else col, mew=0.3, ls="none", zorder=4)
    pa.set_ylim(-0.6, len(series) - 0.4); pa.set_xlim(lo_p, hi_p)
    pa.set_yticks(range(len(series))); pa.set_yticklabels([sname for sname, *_ in series][::-1])
    pa.tick_params(axis="y", length=0, pad=1.2, labelsize=FS_T)
    step = 20.0 if hi_p - lo_p > 30 else 10.0
    pa.xaxis.set_major_locator(MultipleLocator(step)); pa.tick_params(axis="x", labelsize=FS_T, length=1.5, pad=1.0)
    for k in ("top", "right", "left"):
        pa.spines[k].set_visible(False)
    pa.set_xlim(lo_p, hi_p)
    T(ax, xpl + wp + 1.8, ytp + hp + 1.0 + 0.85, "cm", fs=FS_T, ha="left", va="center")
    out["product_axes"] = pa
    SRC["c"] = dict(validation=dict(inputs=["polar.testbed_v5 load_v5 -> with_forcing('year_obs') -> usable_mask -> region == 'Lena' -> make_folds(k = 5, seed = 0)",
                                            "checked against data/processed/v5/runs/core_v6/preds/Lena__<scheme>__year_obs__F8TS__P1__-__lam-1.csv.gz and ALL__region__...: %s" % check],
                                    n_rows=int(len(dr)), n_cells=int(dr.loc_id.nunique()), years=sorted(int(v) for v in dr.year.unique()), panels=info,
                                    colours=dict(test=C_TEST, training=C_TRAIN, excluded=C_EXCL),
                                    cell_status_rule="a cell is test if any of its cell-years is in the test fold, else training if any is in training, else excluded",
                                    region_rule="all Lena cell-years are test; training = Alaska and Canada (NW) cell-years more than 100 km from any Lena cell"),
                    experiments=dict(note="schematic curves, no data values; markers = design steps read from data/processed/v5/runs/ladder_v6/ladder_metrics.csv",
                                     label_number_steps=LD["LL"], new_region_steps=LD["TL"],
                                     label_number="spatial-block folds; training fold subsampled to n labels", new_region="training = the two other regions + n target-region labels from blocks outside the test fold; open marker = n = 0 (source regions only)"),
                    structures=dict(names=["Direct ML", "Physics-input ML", "Physics pseudo-labels", "Physics + residual ML"], codes=["D0", "F1", "D1", "R1"],
                                    key_colours={k: S.METHOD_HEX[k] for k in ("D0", "F1", "D1", "R1")}),
                    anomaly=dict(years=A["anom_years"], definition="r1(year) - mean_2011_2024 r1 per 1 km cell, box means 0.05 x 0.02 deg", range_cm=[-15, 15],
                                 inputs=["data/processed/v5/maps/Alaska/pred_<year>.csv.gz (r1)", "data/processed/v5/maps/Alaska/mean_2011_2024.csv.gz (r1)"],
                                 stats={str(ay): dict(mean=round(A["stats"][f"anom_{ay}"]["mean"], 2), p1=round(A["stats"][f"anom_{ay}"]["p1"], 2), p99=round(A["stats"][f"anom_{ay}"]["p99"], 2)) for ay in A["anom_years"]}),
                    product_example=dict(cell=int(r.loc_id), lat=round(float(r.lat), 4), lon=round(float(r.lon), 4), year=int(r.year), obs_doy=float(r.obs_doy),
                                         obs_date=doy_label(int(r.year), float(r.obs_doy)), tdd_obs_fraction_of_full_season=round(float(r.tdd_obs_frac), 3),
                                         values_cm={sname: round(v, 2) for sname, v, _, _ in series}, block_cv_fold=str(r.fold),
                                         x_range_cm=[lo_p, hi_p],
                                         rule="Lena Delta label cell-years with Wei 2026, Liu 2024, ESA CCI v4 and v5 values of the same cell and year and TDD_obs >= 90 %% of the full-season TDD (%d of %d with all four products); the one at the lower median of the absolute out-of-fold error of physics + residual ML (%.2f cm)" % (PX["n_candidates"], PX["n_all_products"], PX["median_lower"]),
                                         inputs=["data/processed/v5/products_cell_year_v5.csv (product, loc_id, year, value_cm)", PX["pred_file"] + " (seed mean of 3, spatial-block out-of-fold)"],
                                         colours={sname: col for sname, _, col, _ in series}, markers="observed diamond, others circles; dotted line = observed value"))
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
    YA, YB, YC = 0.0, 44.0, 101.0
    row_header(ax, YA + 0.8, "a", "Cell-year design")
    row_a(fig, ax, YA, D)
    row_header(ax, YB + 0.8, "b", "Physics-residual model")
    row_b(fig, ax, YB, SD, A, meta)
    row_header(ax, YC + 0.8, "c", "Evaluation and outputs")
    PX = product_example(d)
    row_c(fig, ax, YC, dr, F, check, A, PX)
    SRC["qa"] = qa(fig, ax)
    SRC["figure"] = dict(width_mm=W_MM, height_mm=H_MM, png_dpi=600, font="FreeSans", sizes_pt=dict(text=FS, small=FS_S, min=FS_T, letter=FS_LETTER, equation=FS_EQ),
                         palette="POLAR_PALETTE=%s" % os.environ.get("POLAR_PALETTE"), script="scripts/4_visualization/paper_v11/fig2_workflow.py")
    out = save(fig, final); plt.close(fig)
    if final:
        sd = OUT / "source_data"; sd.mkdir(parents=True, exist_ok=True)
        (sd / f"{STEM}.json").write_text(json.dumps(SRC, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        out.append(sd / f"{STEM}.json")
    for p in out:
        print(p)


if __name__ == "__main__":
    main()
