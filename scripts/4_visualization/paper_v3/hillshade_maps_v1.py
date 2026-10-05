"""showpiece · 음영기복(hillshade) 위에 얹은 1 km 잔차 ML ALT 지도(알래스카, 레나 델타).

DEM   Copernicus DEM GLO-30 타일(<region>_grid_dem_tiles_v1.csv 목록, data/raw/dem)을 지도 투영의 250 m 격자로 재투영·평균(rasterio reproject,
      Resampling.average). 음영: 방위 315°, 고도 45°, 수직 과장 2(ESRI 식), 평지 = 1 이 되도록 cos(천정각)으로 나눈 뒤 [0, 1] 로 자른다.
드레이프 1 km 셀의 ALT 색(승인된 oslo_r, v3 지도와 같은 범위)을 250 m 격자에 최근접(≤ 1 km)으로 옮기고 곱 혼합 rgb·((1 − w) + w·shade), w = 0.35.
      바다 흰색(음영 없음), 영역 밖 육지는 바탕 육지색에 음영, 수체 흰색, 자료 없음 회색.
산출  outputs/figures/paper/v3_restructure/maps/<Region>_ALT_hillshade_v1.{pdf,png}(170 mm), _legend.md, _source_values.json
      deck/assets/paper_report/maps/<Region>_ALT_hillshade_slide.png(12.0 × 5.2 in)
      data/processed/map_<region>/hillshade_v1/{dem_250m.tif, shade_250m.tif, drape_rgb.npz}(표지 배경·3D 그림이 다시 쓴다)
실행  nice -n 10 python3 scripts/4_visualization/paper_v3/hillshade_maps_v1.py --region lena|alaska|both [--stage dem,fig]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                                                                         # noqa: E402
import json                                                                                             # noqa: E402
import math                                                                                             # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402
import matplotlib                                                                                       # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                         # noqa: E402
import rasterio                                                                                         # noqa: E402
from rasterio.crs import CRS                                                                            # noqa: E402
from rasterio.enums import Resampling                                                                   # noqa: E402
from rasterio.features import rasterize                                                                 # noqa: E402
from rasterio.transform import Affine                                                                   # noqa: E402
from rasterio.warp import reproject                                                                     # noqa: E402
from matplotlib.colors import Normalize, to_rgb                                                         # noqa: E402
from scipy.spatial import cKDTree                                                                       # noqa: E402
from pyproj import Transformer                                                                          # noqa: E402
import shapely                                                                                          # noqa: E402
from shapely.ops import transform as shp_transform                                                      # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import style as S                                                                                       # noqa: E402
import maps_alt_v3 as MV                                                                                # noqa: E402

PROC = ROOT / "data" / "processed"
RAW_DEM = ROOT / "data" / "raw" / "dem"
RES_M = 250.0
AZ, ALT_DEG, VE, W_SHADE = 315.0, 45.0, 2.0, 0.35
NOTE = "1 km ALT on 250 m hillshade; information resolution ~10 km (climate input)"
STEM = {"lena": "Lena_ALT_hillshade", "alaska": "Alaska_ALT_hillshade"}
TILES = {"lena": PROC / "map_lena" / "lena_grid_dem_tiles_v1.csv", "alaska": PROC / "map_alaska" / "alaska_grid_dem_tiles_v1.csv"}
HS_DIR = {"lena": PROC / "map_lena" / "hillshade_v1", "alaska": PROC / "map_alaska" / "hillshade_v1"}


def log(*a):
    print(*a, flush=True)


def dst_crs(key, proj) -> CRS:
    if key == "alaska":
        return CRS.from_epsg(3338)
    return CRS.from_proj4(proj.proj4_init)


def dst_grid(ext):
    """250 m 격자(투영 좌표, 북쪽이 첫 행). 반환 (transform, (nrow, ncol))."""
    x0, x1, y0, y1 = ext
    ncol, nrow = int(math.ceil((x1 - x0) / RES_M)), int(math.ceil((y1 - y0) / RES_M))
    return Affine(RES_M, 0.0, x0, 0.0, -RES_M, y1), (nrow, ncol)


# ================================================================ DEM → 250 m, 음영
def build_dem(key, d, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    crs = dst_crs(key, d["proj"])
    tr, shape = dst_grid(d["ext"])
    tiles = pd.read_csv(TILES[key])
    dem = np.full(shape, np.nan, np.float32)
    n_ok, n_missing, t0 = 0, [], time.time()
    for name in tiles.tile:
        p = RAW_DEM / f"{name}.tif"
        if not p.exists():
            n_missing.append(name)
            continue
        with rasterio.open(p) as src:
            reproject(rasterio.band(src, 1), dem, dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, src_nodata=src.nodata,
                      resampling=Resampling.average, init_dest_nodata=False, num_threads=4, warp_mem_limit=512)
        n_ok += 1
    t_dem = round(time.time() - t0, 1)
    log(f"[{key}] DEM 250 m {shape} · 타일 {n_ok} · 없음 {len(n_missing)} · {t_dem}s")
    shade = hillshade(dem, RES_M)
    with rasterio.open(out_dir / "dem_250m.tif", "w", driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="float32", crs=crs,
                       transform=tr, nodata=np.nan, compress="lzw") as dst:
        dst.write(dem, 1)
    with rasterio.open(out_dir / "shade_250m.tif", "w", driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="uint8", crs=crs,
                       transform=tr, compress="lzw") as dst:
        dst.write(np.round(shade * 255).astype(np.uint8), 1)
    fin = dem[np.isfinite(dem)]
    return dict(shape=list(shape), n_tiles=n_ok, missing_tiles=n_missing, seconds=t_dem, res_m=RES_M, crs=crs.to_string(),
                dem_stats=dict(min=float(fin.min()), p50=float(np.median(fin)), p99=float(np.percentile(fin, 99)), max=float(fin.max()),
                               frac_finite=float(np.isfinite(dem).mean())))


def hillshade(z, cell_m, az=AZ, alt=ALT_DEG, ve=VE):
    """ESRI 음영 식. 결측·바다는 0 m 로 두고 수직 과장 ve 를 곱한 뒤 기울기·향을 구한다. 평지 = 1(cos(천정각)으로 나눔), [0, 1]."""
    zz = np.where(np.isfinite(z), z, 0.0).astype(np.float64) * ve
    gy, gx = np.gradient(zz, cell_m, cell_m)                           # gy: 남쪽이 높으면 양수(첫 행이 북쪽), gx: 동쪽이 높으면 양수
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(gy, -gx)
    zen = np.radians(90.0 - alt)
    azm = np.radians((360.0 - az + 90.0) % 360.0)
    sh = np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(azm - aspect)
    return np.clip(sh / np.cos(zen), 0.0, 1.0).astype(np.float32)


# ================================================================ 드레이프
def land_mask(key, d, tr, shape, crs) -> np.ndarray:
    ne = MV.ne_layers(key)["land"]
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    geoms = [shp_transform(tf, g) for g in ne]
    return rasterize([(g, 1) for g in geoms if not g.is_empty], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)


def drape(key, d, out_dir: Path, values, rng) -> dict:
    with rasterio.open(out_dir / "shade_250m.tif") as src:
        shade = src.read(1).astype(np.float32) / 255.0
        tr, shape, crs = src.transform, (src.height, src.width), src.crs
    land = land_mask(key, d, tr, shape, crs)
    # 1 km 셀 → 250 m 최근접(≤ 1 km)
    xy = d["xy"]
    sel = d["shown"] | d["water"] | d["nodata"]
    tree = cKDTree(xy[sel])
    gx = tr.c + (np.arange(shape[1]) + 0.5) * tr.a
    gy = tr.f + (np.arange(shape[0]) + 0.5) * tr.e
    rgb = np.empty(shape + (3,), np.float32)
    rgb[:] = to_rgb("#ffffff")
    rgb[land] = to_rgb(S.BASEMAP["land"])
    norm = Normalize(*rng)
    cls = np.zeros(sel.sum(), np.int8)                                   # 1 표시, 2 수체, 3 자료 없음
    cls[d["shown"][sel]] = 1
    cls[d["water"][sel]] = 2
    cls[d["nodata"][sel]] = 3
    col = np.zeros((sel.sum(), 3), np.float32)
    v = np.asarray(values, float)[sel]
    ok = cls == 1
    col[ok] = MV.CM_ALT(norm(v[ok]))[:, :3]
    col[cls == 2] = to_rgb(MV.WATER)
    col[cls == 3] = to_rgb(MV.NODATA)
    n_hit = 0
    for r0 in range(0, shape[0], 512):
        r1 = min(shape[0], r0 + 512)
        XX, YY = np.meshgrid(gx, gy[r0:r1])
        dist, idx = tree.query(np.c_[XX.ravel(), YY.ravel()], distance_upper_bound=1000.0)
        hit = np.isfinite(dist)
        n_hit += int(hit.sum())
        blk = rgb[r0:r1].reshape(-1, 3)
        blk[hit] = col[idx[hit]]
        rgb[r0:r1] = blk.reshape(r1 - r0, shape[1], 3)
    shade_land = np.where(land, shade, 1.0)                              # 바다는 평지
    out = rgb * ((1.0 - W_SHADE) + W_SHADE * shade_land)[:, :, None]
    out = np.clip(out, 0, 1)
    np.savez_compressed(out_dir / "drape_rgb.npz", rgb=np.round(out * 255).astype(np.uint8), transform=np.array(tr.to_gdal()),
                        land=land, shape=np.array(shape))
    return dict(n_hit_px=n_hit, frac_land_px=float(land.mean()), shape=list(shape), rgb=out, extent=[gx[0] - RES_M / 2, gx[-1] + RES_M / 2,
                                                                                                        gy[-1] - RES_M / 2, gy[0] + RES_M / 2])


# ================================================================ 그림
def draw_map(fig, d, M, x_mm, y_mm, mw, mh, rgb, extent, left=True, bottom=True):
    ax = S.axes_mm(fig, x_mm, y_mm, mw, mh, projection=d["proj"])
    ax.set_extent(d["ext"], crs=d["proj"])
    ax.set_facecolor("#ffffff")
    im = ax.imshow(rgb, extent=extent, origin="upper", transform=d["proj"], interpolation="nearest", zorder=1)
    im.set_rasterized(True)
    ne = MV.ne_layers(d["key"])
    ax.add_geometries(ne["coastline"], crs=MV.PC, facecolor="none", edgecolor=S.BASEMAP["coast"], linewidth=M.lw_coast, zorder=4)
    ax.spines["geo"].set_linewidth(M.lw)
    ax.spines["geo"].set_edgecolor("#000000")
    MV.graticule(ax, d["xlocs"] if d["key"] == "alaska" else d["xlocs"][::2], d["ylocs"], M, left=left, bottom=bottom)
    return ax


def fig_hillshade(d, rgb, extent, medium):
    M = MV.Med(medium)
    MV.use_medium(medium)
    proj, ext = d["proj"], d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    lo, hi = d["rng"]["alt"]
    sh = d["shown"]
    if medium == "paper":
        W, LM, KEYR, TOP, BOT = 170.0, 8.5, 16.0, 1.5, 9.5
        mw = W - LM - KEYR - 0.3
        mh = mw * asp
        if TOP + mh + BOT > S.HMAX_MM:
            mh = S.HMAX_MM - TOP - BOT
            mw = mh / asp
        H = TOP + mh + BOT
        fig = S.fig_mm(W, H)
        ax = draw_map(fig, d, M, LM, TOP, mw, mh, rgb, extent)
        cax = S.axes_mm(fig, LM + mw + 2.0, TOP + 0.30 * mh, M.cbar, 0.40 * mh)
        MV.colorbar(fig, cax, MV.CM_ALT, Normalize(lo, hi), "ALT (cm)", MV.ticks_between(lo, hi), M, MV.extend_of(d["r1"][sh], lo, hi),
                    orientation="vertical")
        k_sb, c_sb = MV.scale_in_corner(ax, d, ext, M, mw)
        t = fig.text(LM / W, 1 - (H - 1.0) / H, NOTE, fontsize=M.fs_small, color=MV.GRAY_TXT, ha="left", va="bottom")
        t.set_gid("note")
        lay = dict(size_mm=[W, round(H, 1)], map_mm=[round(mw, 1), round(mh, 1)], scale_bar=dict(corner=k_sb, counts=c_sb))
    else:
        W, H = MV.SLIDE_W_MM, MV.SLIDE_H_MAX_MM
        LATW, TOP, BOT, KEYR = 14.0, 3.0, 13.0, 24.0
        mh = H - TOP - BOT
        mw = mh / asp
        LM = (W - (LATW + mw + 3.0 + KEYR)) / 2 + LATW                 # 지도·색 막대 묶음을 가운데에 둔다
        fig = plt.figure(figsize=(W / 25.4, H / 25.4))
        ax = draw_map(fig, d, M, LM, TOP, mw, mh, rgb, extent)
        cax = S.axes_mm(fig, LM + mw + 3.0, TOP + 0.30 * mh, M.cbar, 0.40 * mh)
        MV.colorbar(fig, cax, MV.CM_ALT, Normalize(lo, hi), "ALT (cm)", MV.ticks_between(lo, hi), M, MV.extend_of(d["r1"][sh], lo, hi),
                    orientation="vertical")
        k_sb, c_sb = MV.scale_in_corner(ax, d, ext, M, mw)
        t = fig.text(LM / W, 1 - (H - 1.5) / H, NOTE, fontsize=M.fs_small, color=MV.GRAY_TXT, ha="left", va="bottom")
        t.set_gid("note")
        lay = dict(size_in=[round(W / 25.4, 2), round(H / 25.4, 2)], map_mm=[round(mw, 1), round(mh, 1)], scale_bar=dict(corner=k_sb, counts=c_sb))
    return fig, lay


def legend_md(d, info) -> str:
    lo, hi = d["rng"]["alt"]
    reg = d["name"]
    src = "map_alaska_v1 (E = 1.617, λ = 0.5)" if d["key"] == "alaska" else "the registered Lena final map (h49 R1, λ = 0.25)"
    return (f"# {reg} ALT on hillshade\n\n"
            f"**Residual-ML active-layer thickness (ALT) of the {reg} at 1 km, draped on terrain relief.** Colour, ALT of the final residual-ML map "
            f"({src}), {lo:.0f}–{hi:.0f} cm on the same scale as the v3 map figure. Relief, hillshade of the Copernicus DEM GLO-30 averaged to "
            f"250 m in the map projection (azimuth 315°, altitude 45°, vertical exaggeration 2, {info['dem']['n_tiles']} tiles), blended by "
            f"multiplication with weight 0.35 (flat ground keeps the ALT colour). Land outside the mapped domain is grey with the same relief; "
            f"white, sea or water; light grey, no data. The relief is display context only: the ALT value of each 250 m pixel is that of its "
            f"1 km cell, and the climate inputs behind the map are ERA5-Land 0.1° (about 10 km). "
            f"Coastline, Natural Earth 10 m. Projection and extent as in the v3 map figure.\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="both")
    ap.add_argument("--stage", default="dem,fig")
    a = ap.parse_args(argv)
    regions = ["lena", "alaska"] if a.region == "both" else [a.region]
    stages = set(a.stage.split(","))
    for key in regions:
        t0 = time.time()
        d = MV.prepare(MV.load_region(key))
        out_dir = HS_DIR[key]
        info_path = out_dir / "hillshade_v1_meta.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        if "dem" in stages or not (out_dir / "shade_250m.tif").exists():
            info["dem"] = build_dem(key, d, out_dir)
        if "fig" in stages:
            dr = drape(key, d, out_dir, d["r1"], d["rng"]["alt"])
            rgb, extent = dr.pop("rgb"), dr.pop("extent")
            info["drape"] = dict(dr, w_shade=W_SHADE, alt_range=d["rng"]["alt"], extent=extent)
            stem = STEM[key]
            fig, lay = fig_hillshade(d, rgb, extent, "paper")
            S.save_fig(fig, f"{stem}_v1", MV.OUT_PAPER, formats=("pdf", "png"))
            au = S.audit_v3(fig)
            info["paper"] = dict(layout=lay, audit=dict(sizes=sorted(au["sizes"]), chars=au["chars"], thin=au["thin_lines"][:4]))
            plt.close(fig)
            fig, lay = fig_hillshade(d, rgb, extent, "slide")
            fig.savefig(MV.OUT_SLIDE / f"{stem}_slide.png", dpi=300)
            info["slide"] = dict(layout=lay)
            plt.close(fig)
            sv = dict(region=d["name"], alt_range=d["rng"]["alt"], coef=d["coef"], n_shown=int(d["shown"].sum()), hillshade=dict(
                azimuth_deg=AZ, altitude_deg=ALT_DEG, vertical_exaggeration=VE, weight=W_SHADE, res_m=RES_M), dem=info["dem"],
                drape={k: v for k, v in info["drape"].items()}, layout=dict(paper=info["paper"]["layout"], slide=lay))
            (MV.OUT_PAPER / f"{stem}_v1_source_values.json").write_text(json.dumps(sv, ensure_ascii=False, indent=1, default=str))
            (MV.OUT_PAPER / f"{stem}_v1_legend.md").write_text(legend_md(d, info))
        info["elapsed_s"] = round(time.time() - t0, 1)
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str))
        log(f"[{key}] {json.dumps({k: v for k, v in info.items() if k != 'drape'}, ensure_ascii=False, default=str)[:900]} · {info['elapsed_s']}s")


if __name__ == "__main__":
    main()
