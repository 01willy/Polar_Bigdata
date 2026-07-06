"""공간 DL 학습셋 조립: ALT 점(ABoVE 22.4만 + CALM 북미) 각각에
  - 지형특징(Copernicus DEM 30m; dl_locations)  + DEM 패치 인덱스
  - ERA5-Land 공변량(도일/적설/토양온도; 격자 최근접)
를 부착 → data/processed/dl_dataset.csv  (+ 패치는 dl_patches.npy 재사용)
"""
import os, calendar
import numpy as np
import pandas as pd
import xarray as xr

KEY = 4
ERA5F = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]

# ---------- 위치별 ERA5 공변량 ----------
loc = pd.read_csv("data/processed/dl_locations.csv")
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tname = "valid_time" if "valid_time" in ds.coords else "time"
clim = ds.assign_coords(month=ds[tname].dt.month).groupby("month").mean(tname)
lat = clim["latitude"].values; lon = clim["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])
t2m = clim["t2m"].values - 273.15; stl = clim["stl1"].values - 273.15; swe = clim["sd"].values

def nn(a, v):
    return int(np.abs(a - v).argmin())

E = np.full((len(loc), len(ERA5F)), np.nan)
for i, r in loc.iterrows():
    iy, ix = nn(lat, r.lat), nn(lon, r.lon)
    tm = t2m[:, iy, ix]
    if np.all(np.isnan(tm)):
        continue
    tdd = float(np.nansum(np.clip(tm, 0, None) * days))
    fdd = float(np.nansum(np.clip(-tm, 0, None) * days))
    E[i] = [np.nanmean(tm), tdd, fdd, np.sqrt(tdd), np.nanmax(tm), np.nanmin(tm),
            np.nanmean(stl[:, iy, ix]), np.nanmean(swe[:, iy, ix])]
for k, nm in enumerate(ERA5F):
    loc[nm] = E[:, k]
print(f"위치 {len(loc)}개에 ERA5 공변량 부착 (유효 {np.isfinite(E[:,0]).sum()})")

# ---------- ALT 점 + 위치 병합 ----------
ab = pd.read_csv("data/processed/alt_above_pointlevel.csv")[["lat", "lon", "year", "alt_cm"]]
ab["source"] = "ABoVE"
ab["region"] = np.where(ab.lon < -141, "ABoVE_AK", "ABoVE_CA")
calm = pd.read_csv("data/processed/alt_global.csv")
calm = calm[(calm.lat > 50) & (calm.lon < -100)][["lat", "lon", "year", "alt_cm", "country"]].copy()
calm["source"] = "CALM"; calm["region"] = calm["country"]; calm = calm.drop(columns="country")
pts = pd.concat([ab, calm], ignore_index=True)
pts["klat"] = pts.lat.round(KEY); pts["klon"] = pts.lon.round(KEY)

loc_key = loc.rename(columns={"lat": "klat", "lon": "klon"})
ds_out = pts.merge(loc_key, on=["klat", "klon"], how="left")
ds_out = ds_out.dropna(subset=["dem_elev", "e5_maat", "alt_cm"]).copy()
ds_out["y"] = np.log1p(ds_out["alt_cm"])
ds_out = ds_out.drop(columns=["klat", "klon"])
ds_out.to_csv("data/processed/dl_dataset.csv", index=False)
print(f"학습셋: {len(ds_out):,} 점 (DEM+ERA5 완비), {ds_out.loc_id.nunique():,} 고유위치, "
      f"{ds_out.region.nunique()} region")
print("  region 분포:", ds_out.region.value_counts().head(6).to_dict())
print("  피처:", [c for c in ds_out.columns if c.startswith(('dem_', 'e5_'))])
