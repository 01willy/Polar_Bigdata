"""B0 준비: weak label 408만 점에 공변량(ERA5 8종 + DEM 6종) 벡터화 부착.
- ERA5: 기후값 파생격자를 한 번 만들고 searchsorted 인덱싱(4M 점 일괄).
- DEM: 타일별로 전체 배열 로드 → gradient/uniform_filter로 slope/TPI/roughness 격자 → 일괄 인덱싱.
산출: data/processed/pretrain_weaklabels.parquet (실측 dl_dataset과 동일 피처 스키마)
"""
import os, calendar, glob
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from scipy.ndimage import uniform_filter

W = 33  # 지형 통계 창(≈1km) — 실측 파이프라인과 동일 스케일

# ---------------- ERA5 파생 격자 ----------------
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
elat = clim["latitude"].values          # 내림차순
elon = clim["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]
t = clim["t2m"].values - 273.15
stl = clim["stl1"].values - 273.15
sd = clim["sd"].values
tdd = np.nansum(np.clip(t, 0, None) * days, 0)
fdd = np.nansum(np.clip(-t, 0, None) * days, 0)
E5 = dict(e5_maat=np.nanmean(t, 0), e5_tdd=tdd, e5_fdd=fdd, e5_sqrt_tdd=np.sqrt(tdd),
          e5_twarm=np.nanmax(t, 0), e5_tcold=np.nanmin(t, 0),
          e5_stl1=np.nanmean(stl, 0), e5_swe=np.nanmean(sd, 0))
print("ERA5 파생격자 완료", E5["e5_maat"].shape)

wl = pd.read_parquet("data/processed/resalt_weaklabels.parquet")
n = len(wl)
iy = np.clip(np.searchsorted(-elat, -wl.lat.values), 0, len(elat) - 1)   # 내림차순 대응
ix = np.clip(np.searchsorted(elon, wl.lon.values), 0, len(elon) - 1)
for k, g in E5.items():
    wl[k] = g[iy, ix].astype(np.float32)
print(f"ERA5 부착: 유효 {np.isfinite(wl.e5_maat).sum():,}/{n:,}")

# ---------------- DEM 타일별 지형특징 ----------------
wl["tlat"] = np.floor(wl.lat).astype(int)
wl["tlon"] = np.floor(wl.lon).astype(int)
for c in ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]:
    wl[c] = np.nan
mperdeg = 111320.0

def tname(a, b):
    return f"Copernicus_DSM_COG_10_N{abs(a):02d}_00_W{abs(b):03d}_00_DEM"

groups = wl.groupby(["tlat", "tlon"])
done = 0
for (a, b), g in groups:
    path = f"data/raw/dem/{tname(a, b)}.tif"
    if not os.path.exists(path):
        continue
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nod = src.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        fc, fr = (~src.transform) * (g.lon.values, g.lat.values)
        rows = np.clip(np.floor(fr).astype(int), 0, arr.shape[0] - 1)
        cols = np.clip(np.floor(fc).astype(int), 0, arr.shape[1] - 1)
        dy = mperdeg * abs(src.transform.e)
        dx = mperdeg * abs(src.transform.a) * max(np.cos(np.radians(a + 0.5)), 0.05)
    arrf = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
    gy, gx = np.gradient(arrf, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gy, gx)))
    asp = np.arctan2(-gy, gx)
    m1 = uniform_filter(arrf, W)
    m2 = uniform_filter(arrf ** 2, W)
    std = np.sqrt(np.clip(m2 - m1 ** 2, 0, None))
    idx = g.index
    wl.loc[idx, "dem_elev"] = arr[rows, cols]
    wl.loc[idx, "dem_slope"] = slope[rows, cols]
    wl.loc[idx, "dem_aspect_sin"] = np.sin(asp[rows, cols])
    wl.loc[idx, "dem_aspect_cos"] = np.cos(asp[rows, cols])
    wl.loc[idx, "dem_tpi"] = arr[rows, cols] - m1[rows, cols]
    wl.loc[idx, "dem_rough"] = std[rows, cols]
    done += 1
print(f"DEM 처리 타일 {done}/{groups.ngroups}, 유효 {wl.dem_elev.notna().sum():,}/{n:,}")

out = wl.dropna(subset=["dem_elev", "e5_maat"]).drop(columns=["tlat", "tlon"]).reset_index(drop=True)
out["y"] = np.log1p(out.alt_cm)
out.to_parquet("data/processed/pretrain_weaklabels.parquet", index=False)
print(f"\n[사전학습셋] {len(out):,} 점 (공변량 완비) → pretrain_weaklabels.parquet")
print(out[["alt_cm", "e5_maat", "dem_elev", "dem_slope"]].describe().round(1).to_string())
