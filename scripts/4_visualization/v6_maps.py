"""v6 지도 그림(F6): 250 m 지형 음영 위에 1 km 예측을 얹은 드레이프, 발산 색의 연도 편차, 1 km 확대 창(관측 셀 겹침), 구간·AOA, 레나 델타.
입력: data/processed/v5/maps/<Region>/{pred_<year>,mean_2011_2024}.csv.gz(모델 예측), data/processed/map_<region>/hillshade_v1/shade_250m.tif(음영),
      map_<region>/<region>_grid_cells_v1.csv(셀 ky·kx), labels_cell_year_v5.csv(관측 셀). 실행: python3 scripts/4_visualization/v6_maps.py [--maps-dir data/processed/v5/maps] [--slide]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3"))
import polar.style_v5 as S
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgb
import matplotlib.patheffects as pe
import maps_alt_v3 as MV
PROC = ROOT / "data" / "processed"; OUT = ROOT / "outputs" / "figures" / "paper" / "v6"
W_SHADE = 0.38


def drape(key, values_by_cell, cmap, norm, shown_cells, water=None, fade=None):
    """250 m 음영 격자에 셀 값을 최근접(≤ 1 km)으로 옮겨 색을 입히고 음영과 곱 혼합. 반환 rgb(H,W,3), extent(투영 m), crs."""
    hs = PROC / f"map_{key}" / "hillshade_v1" / "shade_250m.tif"
    with rasterio.open(hs) as src:
        shade = src.read(1).astype(np.float32) / 255.0; tr = src.transform; shape = (src.height, src.width); crs = src.crs
    cells = pd.read_csv(PROC / f"map_{key}" / f"{key}_grid_cells_v1.csv", dtype={"cell_id": str})
    cells = cells[cells.cell_id.isin(shown_cells)]
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tf.transform(cells.lon.values, cells.lat.values)
    v = cells.cell_id.map(values_by_cell).values.astype(float)
    ok = np.isfinite(v); tree = cKDTree(np.c_[x[ok], y[ok]])
    col = cmap(norm(v[ok]))[:, :3]
    if fade is not None:  # 적용 가능 영역 밖 셀은 바탕색 쪽으로 흐리게
        outside = ~cells.cell_id.map(fade).fillna(False).values.astype(bool)[ok]
        col[outside] = 0.4 * col[outside] + 0.6 * np.array(to_rgb("#efefef"), np.float32)
    gx = tr.c + (np.arange(shape[1]) + 0.5) * tr.a; gy = tr.f + (np.arange(shape[0]) + 0.5) * tr.e
    rgb = np.ones(shape + (3,), np.float32)
    land_rgb = np.array(to_rgb("#efefef"), np.float32)
    # 육지 바탕: 음영 값이 유한한 곳(DEM 있음)
    land = np.isfinite(shade) & (shade > 0)
    rgb[land] = land_rgb
    for r0 in range(0, shape[0], 512):
        r1 = min(shape[0], r0 + 512)
        XX, YY = np.meshgrid(gx, gy[r0:r1])
        d, idx = tree.query(np.c_[XX.ravel(), YY.ravel()], distance_upper_bound=800.0)
        hit = np.isfinite(d)
        blk = rgb[r0:r1].reshape(-1, 3); blk[hit] = col[idx[hit]]; rgb[r0:r1] = blk.reshape(r1 - r0, shape[1], 3)
    sh = np.where(land, shade, 1.0)
    out = np.clip(rgb * ((1 - W_SHADE) + W_SHADE * sh)[:, :, None], 0, 1)
    extent = [gx[0] - tr.a / 2, gx[-1] + tr.a / 2, gy[-1] + tr.e / 2, gy[0] - tr.e / 2]
    return out, extent, crs, tf


def coast(ax, key, tf, lw=0.4, color="#8a8a8a"):
    from shapely.ops import transform as shp_transform
    ne = MV.ne_layers(key)
    for g in ne["coastline"]:
        gg = shp_transform(tf.transform, g)
        for part in getattr(gg, "geoms", [gg]):
            xs, ys = part.xy; ax.plot(xs, ys, color=color, lw=lw, zorder=4)


def scalebar(ax, x0, y0, km, label=None):
    ax.plot([x0, x0 + km * 1000], [y0, y0], color="black", lw=1.2, zorder=6, solid_capstyle="butt")
    ax.text(x0 + km * 500, y0 + 4000, label or f"{km} km", ha="center", va="bottom", fontsize=5.5, zorder=6, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])


def draw(ax, rgb, extent, key, tf, ext_box=None):
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="nearest", zorder=1)
    coast(ax, key, tf)
    if ext_box is not None:
        ax.set_xlim(ext_box[0], ext_box[1]); ax.set_ylim(ext_box[2], ext_box[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_edgecolor("#555555")


def latlon_box(tf, lon0, lon1, lat0, lat1):
    lons = np.r_[np.linspace(lon0, lon1, 40), np.full(40, lon1), np.linspace(lon1, lon0, 40), np.full(40, lon0)]
    lats = np.r_[np.full(40, lat0), np.linspace(lat0, lat1, 40), np.full(40, lat1), np.linspace(lat1, lat0, 40)]
    x, y = tf.transform(lons, lats); return x, y


def f6(maps_dir):
    S.setup(scale=S.SCALE)
    MD = ROOT / maps_dir
    lab = pd.read_csv(PROC / "v5" / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    H = 190.0; KS = 180.0 / H; fig = S.figure(S.W2, H)
    # ---------------- 알래스카
    mean = pd.read_csv(MD / "Alaska" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    p14 = pd.read_csv(MD / "Alaska" / "pred_2014.csv.gz", dtype={"cell_id": str}).set_index("cell_id").r1
    p22 = pd.read_csv(MD / "Alaska" / "pred_2022.csv.gz", dtype={"cell_id": str}).set_index("cell_id").r1
    shown = mean.index
    lo, hi = MV.pct_range(mean.r1.values); normA = Normalize(lo, hi)
    fadeA = mean.aoa_in.astype(bool)
    rgbA, extA, crsA, tfA = drape("alaska", mean.r1, MV.CM_ALT, normA, shown, fade=fadeA)
    rgbA0, _, _, _ = drape("alaska", mean.r1, MV.CM_ALT, normA, shown)  # 확대 창용(흐림 없음)
    # 전체 범위(투영): 라벨 셀 상자
    bx, by = latlon_box(tfA, -168, -141, 59, 71.5); boxA = [bx.min() - 2e4, bx.max() + 2e4, by.min() - 2e4, by.max() + 2e4]
    aspA = (boxA[3] - boxA[2]) / (boxA[1] - boxA[0])
    ax = fig.add_axes([0.03, 0.965 - 0.44 * KS * aspA, 0.44, 0.44 * KS * aspA]); draw(ax, rgbA, extA, "alaska", tfA, boxA)
    # 관측 셀(2010–2024) 작은 점
    ox, oy = tfA.transform(lab[lab.region == "Alaska"].drop_duplicates("loc_id").lon.values, lab[lab.region == "Alaska"].drop_duplicates("loc_id").lat.values)
    ax.scatter(ox, oy, s=0.4, c="#222222", lw=0, zorder=5, rasterized=True, alpha=0.6)
    # 확대 창 위치
    Z = {"North Slope (Toolik)": (-150.0, -148.0, 68.4, 69.2), "Interior (Fairbanks)": (-148.6, -146.8, 64.5, 65.3)}
    for k, (lo0, lo1, la0, la1) in Z.items():
        x, y = latlon_box(tfA, lo0, lo1, la0, la1); ax.plot(x, y, color="black", lw=0.8, zorder=6)
        ax.text(x.min(), y.max() + 8000, k, fontsize=5.5, va="bottom", path_effects=[pe.withStroke(linewidth=1.5, foreground="white")], zorder=7)
    scalebar(ax, boxA[0] + 5e4, boxA[2] + 6e4, 200)
    ax.text(0.01, 0.99, "Mean ALT 2011–2024, physics + residual ML\ndots = observed cells; faded = outside area of applicability", transform=ax.transAxes, va="top", ha="left", fontsize=6, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    S.panel_letter(ax, "a", x=0.0, y=1.01); axA = ax
    cax = fig.add_axes([0.07, 0.965 - 0.44 * KS * aspA - 0.03, 0.20, 0.009]); S.register("a", cax); MV.colorbar(fig, cax, MV.CM_ALT, normA, "ALT (cm)", MV.ticks_between(lo, hi), MV.Med("paper"), MV.extend_of(mean.r1.values, lo, hi), orientation="horizontal")
    # b, c: 연도 편차
    v = max(MV.sym_max((p14.reindex(shown) - mean.r1).values), MV.sym_max((p22.reindex(shown) - mean.r1).values)); normD = TwoSlopeNorm(0.0, -v, v)
    for j, (yr, pp, L) in enumerate(((2014, p14, "b"), (2022, p22, "c"))):
        rgb, ext, _, _ = drape("alaska", pp.reindex(shown) - mean.r1, MV.CM_DIFF, normD, shown, fade=fadeA)
        ax = fig.add_axes([0.50 + j * 0.25, 0.965 - 0.235 * KS * aspA, 0.235, 0.235 * KS * aspA]); draw(ax, rgb, ext, "alaska", tfA, boxA)
        ax.text(0.01, 0.99, f"ALT {yr} − mean (ERA5-Land {yr})", transform=ax.transAxes, va="top", ha="left", fontsize=6, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        S.panel_letter(ax, L, x=0.0, y=1.02)
    yD = 0.965 - 0.235 * KS * aspA - 0.03; cax = fig.add_axes([0.765, yD, 0.21, 0.009]); S.register("c", cax); MV.colorbar(fig, cax, MV.CM_DIFF, normD, "ALT difference from 2011–2024 mean (cm)", MV.ticks_between(-v, v), MV.Med("paper"), "both", orientation="horizontal")
    # d: 확대 창 2곳(1 km 셀 그대로, 관측 셀 테두리)
    for j, (k, (lo0, lo1, la0, la1)) in enumerate(Z.items()):
        x, y = latlon_box(tfA, lo0, lo1, la0, la1)
        ax = fig.add_axes([0.50 + j * 0.25, yD - 0.075 - 0.19 * KS, 0.19, 0.19 * KS]); draw(ax, rgbA0, extA, "alaska", tfA, [x.min(), x.max(), y.min(), y.max()])
        s = lab[(lab.region == "Alaska") & lab.lon.between(lo0, lo1) & lab.lat.between(la0, la1)].drop_duplicates("loc_id")
        sx, sy = tfA.transform(s.lon.values, s.lat.values); ax.scatter(sx, sy, s=3, facecolor="none", edgecolor="black", lw=0.3, zorder=5)
        scalebar(ax, x.min() + 5000, y.min() + 5000, 20)
        ax.text(0.01, 0.99, k, transform=ax.transAxes, va="top", ha="left", fontsize=6, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        if j == 0:
            S.panel_letter(ax, "d", x=0.0, y=1.02)
        else:
            S.register("d", ax)
    # e: AOA 와 보정량
    aoa = mean.aoa_in.astype(float)
    from matplotlib.colors import ListedColormap
    rgb, ext, _, _ = drape("alaska", aoa, ListedColormap([S.NEUTRAL["no_data"], "#6BAED6"]), Normalize(-0.5, 1.5), shown)
    yE = 0.965 - 0.44 * KS * aspA - 0.075
    ax = fig.add_axes([0.03, yE - 0.30 * KS * aspA, 0.30, 0.30 * KS * aspA]); draw(ax, rgb, ext, "alaska", tfA, boxA)
    ax.text(0.01, 0.99, f"Area of applicability (blue = inside, {aoa.mean():.0%})", transform=ax.transAxes, va="top", ha="left", fontsize=6, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    S.panel_letter(ax, "e", x=0.0, y=1.01)
    # f: 레나 델타 2016 + 확대
    lm = pd.read_csv(MD / "Lena" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    l16 = pd.read_csv(MD / "Lena" / "pred_2016.csv.gz", dtype={"cell_id": str}).set_index("cell_id").r1.reindex(lm.index)
    loL, hiL = MV.pct_range(lm.r1.values); normL = Normalize(loL, hiL)
    rgbL, extL, crsL, tfL = drape("lena", l16, MV.CM_ALT, normL, lm.index, fade=lm.aoa_in.astype(bool))
    bx, by = latlon_box(tfL, 123.3, 130.1, 71.5, 73.6); boxL = [bx.min(), bx.max(), by.min(), by.max()]
    aspL = (boxL[3] - boxL[2]) / (boxL[1] - boxL[0]); yTop = yD - 0.075 - 0.19 * KS - 0.04
    ax = fig.add_axes([0.36, yTop - 0.28 * KS * aspL, 0.28, 0.28 * KS * aspL]); draw(ax, rgbL, extL, "lena", tfL, boxL)
    ox, oy = tfL.transform(lab[lab.region == "Lena"].drop_duplicates("loc_id").lon.values, lab[lab.region == "Lena"].drop_duplicates("loc_id").lat.values)
    ax.scatter(ox, oy, s=0.4, c="#222222", lw=0, zorder=5, rasterized=True, alpha=0.6)
    x, y = latlon_box(tfL, 126.2, 126.8, 72.2, 72.5); ax.plot(x, y, color="black", lw=0.8, zorder=6)
    scalebar(ax, boxL[0] + 1e4, boxL[2] + 1e4, 50)
    ax.text(0.01, 0.99, "Lena Delta, ALT 2016, physics + residual ML (ERA5-Land 2016)", transform=ax.transAxes, va="top", ha="left", fontsize=6, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    S.panel_letter(ax, "f", x=0.0, y=1.01)
    cax = fig.add_axes([0.39, yTop - 0.28 * KS * aspL - 0.03, 0.22, 0.009]); S.register("f", cax); MV.colorbar(fig, cax, MV.CM_ALT, normL, "ALT (cm)", MV.ticks_between(loL, hiL), MV.Med("paper"), MV.extend_of(lm.r1.values, loL, hiL), orientation="horizontal")
    aspG = (y.max() - y.min()) / (x.max() - x.min())
    ax = fig.add_axes([0.70, yTop - 0.17 * KS * aspG, 0.17, 0.17 * KS * aspG]); draw(ax, rgbL, extL, "lena", tfL, [x.min(), x.max(), y.min(), y.max()])
    s = lab[(lab.region == "Lena") & lab.lon.between(126.2, 126.8) & lab.lat.between(72.2, 72.5)].drop_duplicates("loc_id")
    sx, sy = tfL.transform(s.lon.values, s.lat.values); ax.scatter(sx, sy, s=3, facecolor="none", edgecolor="black", lw=0.3, zorder=5)
    scalebar(ax, x.min() + 1500, y.min() + 1500, 5)
    ax.text(0.01, 0.99, "Samoylov–Kurungnakh (1 km cells; circles = observed cells)", transform=ax.transAxes, va="top", ha="left", fontsize=6, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    S.panel_letter(ax, "g", x=0.0, y=1.01)
    S.register("abcd", *[a for k in ("a", "b", "c", "d") for a in S.PANELS.get(k, [])])
    S.register("efg", *[a for k in ("e", "f", "g") for a in S.PANELS.get(k, [])])
    return S.export(fig, "F6_maps", outdir=OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--maps-dir", default="data/processed/v5/maps"); ap.add_argument("--slide", action="store_true")
    a = ap.parse_args(); S.setup(scale=1.69 if a.slide else 1.0); f6(a.maps_dir); print("[F6] done")
