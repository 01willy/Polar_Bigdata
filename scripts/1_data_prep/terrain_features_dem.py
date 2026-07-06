"""Copernicus DEM 30m → ALT 관측점 고유위치별 지형특징 + DEM 패치 추출.
지형특징: elev, slope, aspect(sin/cos), tpi(지형위치지수), roughness.
패치: 33x33 (≈1km) 상대고도(relief) — 패치 CNN 입력.
산출: data/processed/dl_locations.csv, data/processed/dl_patches.npy
"""
import os, glob
import numpy as np
import pandas as pd
import rasterio

DEM = "data/raw/dem"
W = 33; H = W // 2         # 패치 33x33 (30m → ~1km)
KEY = 4                    # 위경도 반올림 자릿수(~11m)

# ---------- 고유 위치 집합 ----------
ab = pd.read_csv("data/processed/alt_above_pointlevel.csv")[["lat", "lon"]]
calm = pd.read_csv("data/processed/alt_global.csv")
calm = calm[(calm.lat > 50) & (calm.lon < -100)][["lat", "lon"]]
pts = pd.concat([ab, calm], ignore_index=True)
pts["klat"] = pts.lat.round(KEY); pts["klon"] = pts.lon.round(KEY)
loc = pts.drop_duplicates(["klat", "klon"])[["klat", "klon"]].reset_index(drop=True)
loc.columns = ["lat", "lon"]
loc["tlat"] = np.floor(loc.lat).astype(int); loc["tlon"] = np.floor(loc.lon).astype(int)
print(f"ALT 점 {len(pts):,} → 고유위치 {len(loc):,} (DEM 30m 격자 근사)")

def tname(tlat, tlon):
    ns = f"N{abs(tlat):02d}" if tlat >= 0 else f"S{abs(tlat):02d}"
    ew = f"E{abs(tlon):03d}" if tlon >= 0 else f"W{abs(tlon):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"

feats = np.full((len(loc), 6), np.nan)          # elev,slope,aspect_sin,aspect_cos,tpi,rough
patches = np.zeros((len(loc), W, W), dtype=np.float16)
have = np.zeros(len(loc), dtype=bool)
mperdeg = 111320.0

for (tlat, tlon), g in loc.groupby(["tlat", "tlon"]):
    path = os.path.join(DEM, tname(tlat, tlon) + ".tif")
    if not os.path.exists(path):
        continue
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        arr[arr == ds.nodata] = np.nan
        ny, nx = arr.shape
        dy = mperdeg * abs(ds.transform.e)                       # m/px (lat)
        for idx, r in g.iterrows():
            row, col = ds.index(r.lon, r.lat)
            if not (0 <= row < ny and 0 <= col < nx):
                continue
            dx = mperdeg * abs(ds.transform.a) * np.cos(np.radians(r.lat))
            r0, r1 = max(0, row - H), min(ny, row + H + 1)
            c0, c1 = max(0, col - H), min(nx, col + H + 1)
            win = arr[r0:r1, c0:c1]
            if win.size == 0 or np.isnan(win).all():
                continue
            # 패치를 WxW로 패딩(가장자리 복제) — 상대고도
            pad = np.pad(win, ((H - (row - r0), H - (r1 - 1 - row)),
                               (H - (col - c0), H - (c1 - 1 - col))), mode="edge")
            pad = pad[:W, :W]
            cen = arr[row, col]
            patches[idx] = (pad - cen).astype(np.float16)
            gy, gx = np.gradient(win, dy, dx)
            i, j = row - r0, col - c0
            sl = np.degrees(np.arctan(np.hypot(gy[i, j], gx[i, j])))
            asp = np.arctan2(-gy[i, j], gx[i, j])
            feats[idx] = [cen, sl, np.sin(asp), np.cos(asp),
                          cen - np.nanmean(win), float(np.nanstd(win))]
            have[idx] = True

loc = loc[["lat", "lon"]].copy()          # lat/lon = 반올림 키(KEY자리)
loc.insert(0, "loc_id", np.arange(len(loc)))   # 패치 행 인덱스와 일치
for k, nm in enumerate(["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos",
                        "dem_tpi", "dem_rough"]):
    loc[nm] = feats[:, k]
loc["has_dem"] = have
loc.to_csv("data/processed/dl_locations.csv", index=False)
np.save("data/processed/dl_patches.npy", patches)
print(f"지형특징 추출: {have.sum():,}/{len(loc):,} 위치 유효 "
      f"(DEM 미보유 타일 {len(loc)-have.sum():,})")
print(f"  elev {np.nanmin(feats[have,0]):.0f}~{np.nanmax(feats[have,0]):.0f}m, "
      f"slope 0~{np.nanmax(feats[have,1]):.0f}°")
print(f"저장: dl_locations.csv, dl_patches.npy {patches.shape} ({patches.nbytes/1e6:.0f}MB)")
