# -*- coding: utf-8 -*-
"""Manuscript v11 Fig. 1: study regions, label cell-years and data periods (Scientific Reports main-text figure, 180 mm wide).
Copied from scripts/4_visualization/paper_v10/fig1_data.py and edited for v11 (panel d removed: the cumulative-TDD time definition moved to the
new methods figure Fig. 2; layout re-balanced; v11 colours).

Panels
  a  Pan-Arctic map: ESA CCI permafrost probability (1997-2021 mean, light blue ramp), the three study regions outlined in their region colours,
     label cells per region (labels outside the circle with leader lines).
  b  Cell-years per observation year 2010-2024, one small bar panel per region (independent y axes), bars in the region colours.
  c  Data periods on a 1997-2025 axis: label years per region (region colours), ERA5-Land monthly climate input (dark grey), static covariates
     (hatched, no year), six ALT products (available period = light tint of the product colour, years used for the comparison = product colour,
     Aalto 2018 = one multi-year mean, outlined band). Analysis window 2010-2024 lightly shaded. Key above the panel.
Data (all numbers are read from files)
  data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1), data/processed/cci_pfr_mean_1997_2021.nc, data/processed/v5/forcing_meta.json,
  data/processed/v5/products_cell_year_v5_meta.json, data/processed/v5/products_cell_year_v5.csv. Liu 2024 available period (2003-2020) is longer
  than the downloaded files (2010-2020) and is taken from docs/research/2026-10-04/alt_products.md (Zenodo 10070610, ALT_{YYYY}.tif, 2003-2020).
  'Years used' = minimum-maximum year of label cell-years with a value of that product (scripts/2_evaluation/v10_numbers.py nyrsprod*).
Run  cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper12 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v12/fig1_data.py
Out  outputs/figures/paper/v12/Fig1_data.{pdf,svg,png} (180 x 106 mm, png 600 dpi), outputs/figures/paper/v12/source_data/Fig1_data.json
Style white background, text #262626 (labels 7 pt, ticks 6.5 pt, map labels 6 pt, panel letters 8 pt bold), FreeSans, axes #404040 0.5 pt.
      No titles or sentences inside the figure. Saved at the exact canvas size (no bbox cropping).
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c12_common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.path as mpath  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.transforms  # noqa: E402
import matplotlib.lines  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle, Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
from cmcrameri import cm as cmc  # noqa: E402
from pyproj import Geod  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

ROOT = C.ROOT
PROC = ROOT / "data" / "processed"; V5 = PROC / "v5"
OUT = C.OUT_MAIN; STEM = "Fig1_data"
PC = ccrs.PlateCarree()
REG = ["Alaska", "Lena", "Canada"]
REG_EN = {r: C.REGION[r][0] for r in REG}
REG_COL = {r: C.REGION[r][1] for r in REG}
WINDOW = (2010, 2024)                                     # analysis window (checked against forcing_meta.json years)
XLIM_C = (1996.5, 2025.5)                                 # c year axis (1997-2025; year y occupies [y - 0.5, y + 0.5])
LIU_AVAILABLE = (2003, 2020)                              # Liu 2024 available period: docs/research/2026-10-04/alt_products.md (Zenodo 10070610)
LIU_AVAILABLE_SRC = "docs/research/2026-10-04/alt_products.md (Zenodo 10.5281/zenodo.10070610, ALT_YYYY.tif for 2003–2020)"
PROD_KEY = {"wei": "wei", "liu": "liu", "cci4y": "cci4y", "cci5y": "cci5y", "yk": "yk", "aalto": "aalto"}

# ---------------------------------------------------------------- style (tokens color.paper_v12 + plan section 4)
W_MM, H_MM = 180.0, 106.0
FS, FS_TICK, FS_SMALL, FS_MAP = C.FS, C.FS_TICK, 7.0, 7.0      # v12: one text size
INK = C.TEXT
USED_T, AVAIL_T = 1.0, 0.36                               # (v11 tint rule, kept for the source-data note)
AVAIL_COL = "#B3B3B3"                                      # v12: available period = thin grey line; years used = bar in the series colour
SHADE = "#F0F0F0"                                         # analysis-window shade
HATCH_COL = "#8C8C8C"
OCEAN = "#EBEBEB"; LAND = "#FFFFFF"; COAST = "#7A7A7A"; GRAT = "#C4C4C4"
PFR_RANGE = (0.0, 0.30)                                   # oslo_r section for permafrost probability (0 % white, 100 % light steel blue)
PFR_CMAP = ListedColormap(cmc.oslo_r(np.linspace(*PFR_RANGE, 256)))
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
SRC: dict = {}


def rect(x, y_top, w, h):
    """mm rectangle (left, top, width, height; y from the top) -> figure fraction."""
    return [x / W_MM, 1 - (y_top + h) / H_MM, w / W_MM, h / H_MM]


def year_runs(ys):
    out = []
    for y in sorted(int(v) for v in ys):
        if out and y == out[-1][1] + 1:
            out[-1] = (out[-1][0], y)
        else:
            out.append((y, y))
    return out


def runs_str(runs):
    return ", ".join(f"{a}–{b}" if b > a else f"{a}" for a, b in runs)


# ================================================================ data
def load():
    lab = pd.read_csv(V5 / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1].copy()
    n_cells = lab.groupby("region").loc_id.nunique()
    n_rows = lab.groupby("region").size()
    years = np.arange(WINDOW[0], WINDOW[1] + 1)
    cnt = lab.groupby(["year", "region"]).size().unstack("region").reindex(index=years, columns=REG).fillna(0).astype(int)
    fmeta = json.loads((V5 / "forcing_meta.json").read_text())
    y_frc = (min(fmeta["years"]), max(fmeta["years"])); assert y_frc == WINDOW, (y_frc, WINDOW)
    pmeta = json.loads((V5 / "products_cell_year_v5_meta.json").read_text())["products"]
    pr = pd.read_csv(V5 / "products_cell_year_v5.csv")
    prods = []
    for key, name in (("wei", "Wei 2026"), ("liu", "Liu 2024"), ("cci4y", "ESA CCI v4"), ("cci5y", "ESA CCI v5"), ("yk", "Yi–Kimball 2019 (Alaska)"),
                      ("aalto", "Aalto 2018")):
        m = pmeta[key]; period = tuple(int(v) for v in m["period"])
        if key == "liu":
            assert LIU_AVAILABLE[0] <= period[0] and period[1] <= LIU_AVAILABLE[1], period
            avail, avail_src = LIU_AVAILABLE, LIU_AVAILABLE_SRC
        else:
            avail = (min(m["years"]), max(m["years"])) if m.get("years") else period
            avail_src = "data/processed/v5/products_cell_year_v5_meta.json (products.%s.%s)" % (key, "years" if m.get("years") else "period")
        if key == "aalto":                                   # one multi-year mean value (year = -1): no yearly comparison
            assert (pr.loc[pr["product"] == key, "year"] == -1).all()
            used = None
        else:                                                # label cell-years with a product value (as v10_numbers.py nyrsprod*)
            q = lab.merge(pr[(pr["product"] == key) & pr.value_cm.notna()][["loc_id", "year"]], on=["loc_id", "year"])
            used = (int(q.year.min()), int(q.year.max()))
        prods.append(dict(key=key, name=name, available=avail, available_source=avail_src, used=used, extracted_period=period,
                          colour=C.PRODUCT[key][1]))
    rows = [dict(group="Labels", name=REG_EN[r], kind="label", runs=year_runs(cnt.index[cnt[r] > 0]), colour=REG_COL[r]) for r in REG]
    rows += [dict(group="Inputs", name="ERA5-Land monthly climate", kind="used", runs=[y_frc], colour=C.DARK),
             dict(group="Inputs", name="Terrain, soil", kind="static", runs=[], colour=HATCH_COL)]
    for p in prods:
        if p["key"] == "aalto":
            rows.append(dict(group="Products", name=p["name"], kind="mean", runs=[p["available"]], prod=p, colour=p["colour"]))
        else:
            rows.append(dict(group="Products", name=p["name"], kind="product", runs=[p["available"]], used=p["used"], prod=p, colour=p["colour"]))
    return dict(lab=lab, n_cells=n_cells, n_rows=n_rows, cnt=cnt, years=years, rows=rows, prods=prods, fmeta=fmeta)


# ================================================================ a pan-Arctic map
LAT_MIN = 56.0
CLON = -90.0
BOX = {"Alaska": (-168.0, -141.0, 59.0, 71.5), "Lena": (123.3, 130.1, 71.5, 73.6), "Canada": (-141.0, -112.0, 60.0, 70.0)}
A_X0, A_Y0, A_DIA = 18.6, 3.6, 57.5


def pfr_image(proj, lat_min, n):
    """ESA CCI permafrost probability (0.1 deg) bilinearly resampled to the projected n x n grid (missing-weight normalised)."""
    import xarray as xr
    ds = xr.open_dataset(PROC / "cci_pfr_mean_1997_2021.nc"); P = ds["pfr_mean"]
    lat = P["lat"].values.astype(np.float64); lon = P["lon"].values.astype(np.float64); V = P.values.astype(np.float32)
    if lat[0] > lat[-1]:
        lat = lat[::-1]; V = V[::-1]
    dlat = (lat[-1] - lat[0]) / (len(lat) - 1); dlon = (lon[-1] - lon[0]) / (len(lon) - 1)
    valid = np.isfinite(V).astype(np.float32); Vf = np.where(np.isfinite(V), V, 0.0).astype(np.float32)
    Vf = np.c_[Vf, Vf[:, :1]]; valid = np.c_[valid, valid[:, :1]]
    px, py = proj.transform_point(CLON, lat_min, PC); Rm = float(np.hypot(px, py))
    xs = np.linspace(-Rm, Rm, n); X, Y = np.meshgrid(xs, xs)
    ll = PC.transform_points(proj, X.ravel(), Y.ravel()); lo, la = ll[:, 0], ll[:, 1]
    fi = (la - lat[0]) / dlat; fj = ((lo - lon[0]) % 360.0) / dlon
    v = map_coordinates(Vf, [fi, fj], order=1, mode="nearest"); w = map_coordinates(valid, [fi, fj], order=1, mode="nearest")
    out = np.where(w > 0.5, v / np.maximum(w, 1e-6), np.nan)
    out[(la < lat_min) | (la < lat[0])] = np.nan
    return np.ma.masked_invalid(out.reshape(n, n)), [-Rm, Rm, -Rm, Rm], ds.attrs.get("title", "")


def box_line(lo0, lo1, la0, la1, n=60):
    lons = np.r_[np.linspace(lo0, lo1, n), np.full(n, lo1), np.linspace(lo1, lo0, n), np.full(n, lo0)]
    lats = np.r_[np.full(n, la0), np.linspace(la0, la1, n), np.full(n, la1), np.linspace(la1, la0, n)]
    return lons, lats


def draw_a(fig, D):
    x0, y0, dia = A_X0, A_Y0, A_DIA
    proj = ccrs.NorthPolarStereo(central_longitude=CLON)
    ax = fig.add_axes(rect(x0, y0, dia, dia), projection=proj)
    ax.set_extent([-180, 180, LAT_MIN, 90], PC)
    th = np.linspace(0, 2 * np.pi, 361)
    ax.set_boundary(mpath.Path(np.c_[0.5 + 0.5 * np.cos(th), 0.5 + 0.5 * np.sin(th)]), transform=ax.transAxes)
    ax.set_facecolor(OCEAN)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND, edgecolor="none", zorder=0.5)
    img, ext, title = pfr_image(proj, LAT_MIN - 1.0, 1500)
    im = ax.imshow(img, extent=ext, transform=proj, origin="lower", cmap=PFR_CMAP, vmin=0, vmax=100, interpolation="bilinear", zorder=1,
                   rasterized=True)
    ax.coastlines("50m", linewidth=0.4, color=COAST, zorder=2)
    ax.gridlines(crs=PC, draw_labels=False, linewidth=0.25, color=GRAT, xlocs=range(-180, 180, 30), ylocs=[60, 70, 80], zorder=3)
    ax.spines["geo"].set_linewidth(0.5); ax.spines["geo"].set_edgecolor(C.AXIS)
    for la in (60, 70, 80):                                   # latitude labels on the 0 deg meridian (right, over the ocean)
        ax.text(0.0, la + 0.6, f"{la}°N", transform=PC, fontsize=FS_MAP, ha="center", va="bottom", zorder=6, path_effects=HALO, color=INK)
    R = float(np.hypot(*proj.transform_point(CLON, LAT_MIN, PC)))
    # region outlines: white halos of all boxes first, then the coloured lines (Alaska and Canada share the 141 W edge; Alaska drawn last)
    for r in REG:
        lons, lats = box_line(*BOX[r])
        ax.plot(lons, lats, transform=PC, color="white", lw=2.2, zorder=4.9, solid_joinstyle="miter")
    for r in ("Canada", "Lena", "Alaska"):
        lons, lats = box_line(*BOX[r])
        ax.plot(lons, lats, transform=PC, color=REG_COL[r], lw=1.3, zorder=5, solid_joinstyle="miter")
    # names and label-cell counts outside the circle (right-aligned), thin leader lines to the box edge
    fig.canvas.draw()
    to_mm = lambda xy: np.c_[fig.transFigure.inverted().transform(ax.transData.transform(xy))[:, 0] * W_MM,
                             (1 - fig.transFigure.inverted().transform(ax.transData.transform(xy))[:, 1]) * H_MM]
    ANCHOR = {"Lena": (BOX["Lena"][0], (BOX["Lena"][2] + BOX["Lena"][3]) / 2), "Alaska": (BOX["Alaska"][0], (BOX["Alaska"][2] + BOX["Alaska"][3]) / 2),
              "Canada": ((BOX["Canada"][0] + BOX["Canada"][1]) / 2, BOX["Canada"][2])}
    anc = {r: to_mm(np.array([proj.transform_point(*ANCHOR[r], PC)]))[0] for r in REG}
    ymm = {r: float(anc[r][1]) for r in REG}
    order = sorted(REG, key=lambda r: ymm[r]); MIN_GAP = 8.0
    for i in range(1, len(order)):
        ymm[order[i]] = max(ymm[order[i]], ymm[order[i - 1]] + MIN_GAP)
    XL = x0 - 1.4
    lab_txt = {}; leaders = {}
    for r in REG:
        t = fig.text(XL / W_MM, 1 - ymm[r] / H_MM, f"{REG_EN[r]}\n{D['n_cells'][r]:,} cells", ha="right", va="center", fontsize=FS, linespacing=1.1,
                     color=INK)
        lab_txt[r] = t
        a = np.array([XL + 0.6, ymm[r]]); b = anc[r]
        ln = matplotlib.lines.Line2D([a[0] / W_MM, b[0] / W_MM], [1 - a[1] / H_MM, 1 - b[1] / H_MM], transform=fig.transFigure, color=C.AXIS, lw=0.45,
                                     zorder=9, solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
        fig.add_artist(ln); leaders[r] = dict(label_mm=[round(float(v), 2) for v in a], box_point_mm=[round(float(v), 2) for v in b], box_point_lonlat=list(ANCHOR[r]))
    # 1,000 km scale bar inside the circle, lower right (Labrador Sea); projected horizontal bar with geodesic length 1,000 km
    geod = Geod(ellps="WGS84")
    xc, yc = 0.53 * R, -0.70 * R
    def dist(L):
        a = PC.transform_point(xc - L / 2, yc, proj); b = PC.transform_point(xc + L / 2, yc, proj)
        return geod.inv(a[0], a[1], b[0], b[1])[2]
    L = 1.0e6
    for _ in range(30):
        L *= 1.0e6 / dist(L)
    ax.plot([xc - L / 2, xc + L / 2], [yc, yc], color=INK, lw=0.9, solid_capstyle="butt", zorder=8, transform=proj, path_effects=HALO)
    ax.text(xc, yc + 0.025 * R, "1,000 km", ha="center", va="bottom", fontsize=FS_MAP, zorder=8, transform=proj, path_effects=HALO, color=INK)
    sb_ll = PC.transform_point(xc, yc, proj)
    # colour bar (vertical, right of the circle, lower half)
    cax = fig.add_axes(rect(x0 + dia + 2.0, y0 + dia * 0.50, 1.7, dia * 0.42))
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 50, 100])
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(C.AXIS)
    cax.tick_params(labelsize=FS_TICK, length=2.0, width=0.4, pad=1.5, colors=C.AXIS, labelcolor=INK)
    cb.set_label("Permafrost probability (%)", fontsize=FS, labelpad=2.0, color=INK)
    C.letter(fig, 0.9, 0.9, "a")
    SRC["a"] = dict(inputs=["data/processed/cci_pfr_mean_1997_2021.nc (variable pfr_mean; attrs.title: %s)" % title,
                            "data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1, unique loc_id per region)"],
                    years=dict(permafrost_probability="1997–2021 mean", labels=f"{D['lab'].year.min()}–{D['lab'].year.max()}"),
                    projection=f"North Polar Stereographic, central longitude {CLON:g}°, boundary {LAT_MIN:g}°N; bilinear resampling to 1500 × 1500 px",
                    colormap=f"cmcrameri oslo_r, section {PFR_RANGE[0]}–{PFR_RANGE[1]} (0 % white, 100 % light steel blue); colour bar 0–100 %",
                    ocean=OCEAN, land_without_value=LAND, region_outline_colours={REG_EN[r]: REG_COL[r] for r in REG},
                    region_boxes_lonlat={REG_EN[r]: dict(lon=[BOX[r][0], BOX[r][1]], lat=[BOX[r][2], BOX[r][3]]) for r in REG},
                    label_cell_extent={REG_EN[r]: dict(lat=[float(g.lat.min()), float(g.lat.max())], lon=[float(g.lon.min()), float(g.lon.max())])
                                       for r, g in D["lab"].groupby("region").__iter__() if r in REG_EN},
                    leaders_mm=leaders,
                    scale_bar=dict(km=1000, centre_lonlat=[round(float(sb_ll[0]), 2), round(float(sb_ll[1]), 2)],
                                   note="horizontal bar in projected coordinates; geodesic length between its ends = 1,000 km"),
                    numbers_printed={f"{REG_EN[r]} cells": int(D["n_cells"][r]) for r in REG} | {"colour bar ticks (%)": [0, 50, 100],
                                     "graticule labels": ["60°N", "70°N", "80°N"], "scale bar": "1,000 km"},
                    cell_years_per_region={REG_EN[r]: int(D["n_rows"][r]) for r in REG})
    return ax, cax, lab_txt


# ================================================================ b cell-years per observation year
def nice_ticks(vmax):
    for s in (1, 2, 2.5, 4, 5, 10, 20, 25, 40, 50, 100, 200, 250, 400, 500, 1000, 2000, 2500, 4000, 5000, 10000):
        if vmax <= 2 * s:
            return [0, s, 2 * s]
    return [0, vmax]


B_X, B_W, B_TOPS, B_H = 105.0, 73.5, (3.6, 21.6, 39.6), 14.6


def draw_b(fig, D):
    axes = []
    for i, r in enumerate(REG):
        ax = fig.add_axes(rect(B_X, B_TOPS[i], B_W, B_H)); axes.append(ax)
        C.clean_axes(ax)
        v = D["cnt"][r].values
        ax.bar(D["years"], v, width=0.66, color=REG_COL[r], lw=0, zorder=2)
        tk = nice_ticks(float(v.max())); ax.set_ylim(0, tk[-1]); ax.set_yticks(tk)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_xlim(WINDOW[0] - 0.6, WINDOW[1] + 0.6)
        ax.xaxis.set_major_locator(FixedLocator(list(range(WINDOW[0], WINDOW[1] + 1, 2))))
        ax.xaxis.set_minor_locator(FixedLocator(list(range(WINDOW[0], WINDOW[1] + 1))))
        ax.tick_params(axis="x", which="minor", length=1.2, width=0.4)
        if i < 2:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.text(0.012, 1.0, REG_EN[r], transform=ax.transAxes, ha="left", va="top", fontsize=FS, color=INK)
    axes[-1].set_xlabel("Observation year")
    C.text_mm(fig, B_X - 10.2, B_TOPS[0] + (B_TOPS[-1] + B_H - B_TOPS[0]) / 2, "Cell-years", rotation=90, ha="center", va="center", fontsize=FS)
    C.letter(fig, 93.5, 0.9, "b")
    SRC["b"] = dict(inputs=["data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1; rows = cell-years)"],
                    years=f"{WINDOW[0]}–{WINDOW[1]}",
                    cell_years={REG_EN[r]: {int(y): int(c) for y, c in D["cnt"][r].items()} for r in REG},
                    totals={REG_EN[r]: int(D["cnt"][r].sum()) for r in REG},
                    y_ticks={REG_EN[r]: nice_ticks(float(D["cnt"][r].max())) for r in REG}, bar_colours={REG_EN[r]: REG_COL[r] for r in REG},
                    numbers_printed="axis ticks only (y ticks per region listed in y_ticks; x ticks 2010–2024 every 2 years)")
    return axes


# ================================================================ c data periods
C_X, C_W, C_Y, C_H = 44.0, 134.5, 70.0, 28.5


def draw_c(fig, D):
    ax = fig.add_axes(rect(C_X, C_Y, C_W, C_H))
    C.clean_axes(ax, left=False)
    rows = D["rows"]; gap = 0.55; pos = []; y = 0.0; prev = None
    for r in rows:
        if prev is not None and r["group"] != prev:
            y += gap
        pos.append(y); y += 1.0; prev = r["group"]
    ymax = y - 1.0
    bh = 0.58
    ax.axvspan(WINDOW[0] - 0.5, WINDOW[1] + 0.5, color=SHADE, lw=0, zorder=0)
    for r, yy in zip(rows, pos):
        col = r["colour"]
        if r["kind"] in ("label", "used"):
            for a, b in r["runs"]:
                ax.add_patch(Rectangle((a - 0.5, yy - bh / 2), b - a + 1, bh, facecolor=col, edgecolor="none", zorder=3))
        elif r["kind"] == "static":
            ax.plot([XLIM_C[0] + 2.6, WINDOW[1] + 0.5], [yy, yy], color=AVAIL_COL, lw=1.0, ls=(0, (1.0, 1.4)), zorder=3, solid_capstyle="butt")
            ax.text(XLIM_C[0] + 0.4, yy, "static", ha="left", va="center", fontsize=FS, color=INK, clip_on=False, zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        elif r["kind"] == "product":
            a, b = r["runs"][0]
            ax.plot([a - 0.5, b + 0.5], [yy, yy], color=AVAIL_COL, lw=1.0, zorder=2, solid_capstyle="butt")
            ua, ub = r["used"]
            ax.add_patch(Rectangle((ua - 0.5, yy - bh / 2), ub - ua + 1, bh, facecolor=col, edgecolor="none", zorder=3))
        elif r["kind"] == "mean":
            a, b = r["runs"][0]
            ax.plot([a - 0.5, b + 0.5], [yy, yy], color=AVAIL_COL, lw=1.0, zorder=2, solid_capstyle="butt")
            ax.plot([(a + b) / 2], [yy], marker="D", ms=3.2, mfc=col, mec=col, mew=0.0, ls="none", zorder=4)
            ax.text(b + 0.5 + 0.35, yy, "one value, 2000–2014", ha="left", va="center", fontsize=FS, color=INK, clip_on=False)
    ax.set_xlim(*XLIM_C); ax.set_ylim(ymax + 0.6, -0.6)
    ax.xaxis.set_major_locator(FixedLocator([2000, 2005, 2010, 2015, 2020, 2025]))
    ax.xaxis.set_minor_locator(FixedLocator(list(range(1997, 2026))))
    ax.tick_params(axis="x", which="minor", length=1.2, width=0.4)
    ax.set_yticks([])
    ax.set_xlabel("Year")
    tr = ax.get_yaxis_transform()
    for r, yy in zip(rows, pos):
        ax.text(-0.008, yy, r["name"], transform=tr, ha="right", va="center", fontsize=FS_SMALL, color=INK)
    groups = {}
    for r, yy in zip(rows, pos):
        groups.setdefault(r["group"], []).append(yy)
    blend = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for g, ys in groups.items():
        ax.text(0.9 / W_MM, float(np.mean(ys)), g, transform=blend, fontsize=FS, ha="left", va="center", clip_on=False, color=INK)
    # key above the panel (right-aligned): neutral swatches with the same tint rule as the product rows
    hs = [Patch(facecolor=C.DARK, edgecolor="none"), matplotlib.lines.Line2D([], [], color=AVAIL_COL, lw=1.0),
          Patch(facecolor=SHADE, edgecolor="none")]
    leg = ax.legend(hs, ["Years used", "Available", "Analysis period 2010–2024"], loc="lower right",
                    bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=FS_SMALL, handlelength=1.6, handleheight=0.9, handletextpad=0.5,
                    columnspacing=1.3, borderaxespad=0.55, borderpad=0.0)
    for t in leg.get_texts():
        t.set_color(INK)
    C.letter(fig, 0.9, C_Y - 5.6, "c")
    SRC["c"] = dict(inputs=["data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1)", "data/processed/v5/forcing_meta.json (years)",
                            "data/processed/v5/products_cell_year_v5_meta.json (products.*.years / period)",
                            "data/processed/v5/products_cell_year_v5.csv (years with a value at label cell-years)", LIU_AVAILABLE_SRC],
                    axis_years=[1997, 2025], analysis_window=list(WINDOW),
                    colours=dict(rule=f"years used = bar in the series colour; available period = thin grey line {AVAIL_COL}; static inputs = dotted grey line with the note 'static'; "
                                      f"Aalto 2018 = grey line 2000-2014 with a diamond at the midpoint and the note 'one value, 2000-2014'", label_rows="region colours",
                                 climate_input=C.DARK, products={p["name"]: p["colour"] for p in D["prods"]}, analysis_window=SHADE),
                    rows=[dict(group=r["group"], name=r["name"], colour=r["colour"],
                               encoding={"label": "bar in the region colour = years with labels", "used": "dark bar = years used",
                                         "static": "dotted grey line = no time dimension; note 'static'",
                                         "product": "grey line = available period, bar in the product colour = years used for the comparison",
                                         "mean": "grey line 2000-2014 with a diamond = one multi-year mean value; note 'one value, 2000-2014'"}[r["kind"]],
                               years=(runs_str(r["runs"]) if r["runs"] else "static (no year)"),
                               **({"years_used": f"{r['used'][0]}–{r['used'][1]}", "available_source": r["prod"]["available_source"],
                                   "extracted_years_in_project": f"{r['prod']['extracted_period'][0]}–{r['prod']['extracted_period'][1]}"} if r["kind"] == "product" else {}),
                               **({"available_source": r["prod"]["available_source"]} if r["kind"] == "mean" else {})) for r in rows],
                    years_used_definition="years of label cell-years (in_forcing == 1) that have a product value; same as scripts/2_evaluation/v10_numbers.py nyrsprod*",
                    numbers_printed="axis ticks only (2000, 2005, 2010, 2015, 2020, 2025); product names carry publication years")
    return ax


def main():
    C.setup()
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "FreeSans", "mathtext.default": "regular"})
    D = load()
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax_a, cax, lab_txt = draw_a(fig, D)
    axes_b = draw_b(fig, D)
    ax_c = draw_c(fig, D)
    rep = C.qa_figure(fig, ink_axes=axes_b + [ax_c], ignore_ink_for=["°N", "km"])
    C.print_qa(STEM, rep)
    paths = C.export(fig, STEM, OUT)
    plt.close(fig)
    SRC["qa"] = rep
    SRC["figure"] = dict(width_mm=W_MM, height_mm=H_MM, pdf_page_mm=C.pdf_page_mm(paths[0]), pdf_fonts=C.pdf_fonts(paths[0]), png_dpi=600,
                         font="FreeSans", text_pt=FS, tick_pt=FS_TICK, letter_pt=C.FS_LETTER, text_colour=C.TEXT, axis_colour=C.AXIS,
                         palette="POLAR_PALETTE=%s (design/style_tokens_v4.json color.paper_v12)" % os.environ.get("POLAR_PALETTE", "deck"),
                         script="scripts/4_visualization/paper_v12/fig1_data.py", outputs=C.file_info(paths),
                         change_from_v10="panel d (cumulative TDD time definition) removed (moved to the new methods figure Fig. 2); layout re-balanced; "
                                         "region boxes, b bars and c label rows in region colours, c product rows in product colours")
    SRC["years_used"] = dict(a="permafrost probability 1997–2021 mean; label cells 2010–2024", b=f"{WINDOW[0]}–{WINDOW[1]}",
                             c={r["name"]: (f"used {r['used'][0]}–{r['used'][1]}, available {runs_str(r['runs'])}" if r["kind"] == "product"
                                            else (runs_str(r["runs"]) if r["runs"] else "static")) for r in D["rows"]})
    C.write_json(C.SRC_MAIN / f"{STEM}.json", SRC)
    for p in paths:
        print(p, p.stat().st_size)


if __name__ == "__main__":
    main()
