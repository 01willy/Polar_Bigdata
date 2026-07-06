"""ERA5-Land NH 월별 → ALT 사이트 위치에서 실측 도일/적설/토양온도 공변량 산출.
2015-2020 월평균을 월별 기후값으로 집계 후, 각 사이트에서 최근접 격자 샘플.
산출: MAAT, TDD(융해도일), FDD(동결도일), 적설(SWE), stl1 연평균, 최난/최한월.
→ data/processed/alt_era5_covariates.csv  (WorldClim 대체 공변량; 재채점 step에서 사용)
"""
import os, sys, calendar
import numpy as np
import pandas as pd
import xarray as xr

NC = "data/raw/era5land/nh_monthly_2015-2020.nc"
if not os.path.exists(NC):
    sys.exit(f"아직 다운로드 안 됨: {NC}")

ds = xr.open_dataset(NC)
print("변수:", list(ds.data_vars), "| 차원:", dict(ds.sizes))

# 시간 좌표명 정규화
tname = "valid_time" if "valid_time" in ds.coords else ("time" if "time" in ds.coords else None)
ds = ds.assign_coords(month=ds[tname].dt.month)
clim = ds.groupby("month").mean(tname)   # 12개월 기후값

# 변수명 매핑(ERA5-Land short name)
tvar = "t2m" if "t2m" in clim else [v for v in clim.data_vars if "t2m" in v][0]
svar = "sd" if "sd" in clim else [v for v in clim.data_vars if v == "sd" or "snow" in v][0]
gvar = "stl1" if "stl1" in clim else [v for v in clim.data_vars if "stl1" in v][0]

# 사이트 로드(통합 ALT)
sites = pd.read_csv("data/processed/alt_combined.csv")
locs = sites[["lat", "lon"]].drop_duplicates().reset_index(drop=True)

lat = clim["latitude"].values
lon = clim["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])

def nearest(arr, v):
    return int(np.abs(arr - v).argmin())

recs = []
t2m_all = clim[tvar].values - 273.15   # (month, lat, lon) °C
stl_all = clim[gvar].values - 273.15
sd_all = clim[svar].values             # m of water equiv
for _, r in locs.iterrows():
    iy, ix = nearest(lat, r.lat), nearest(lon, r.lon)
    tm = t2m_all[:, iy, ix]            # 12개월 기온
    if np.all(np.isnan(tm)):
        recs.append(dict(lat=r.lat, lon=r.lon))   # 해양/결측
        continue
    tdd = float(np.nansum(np.clip(tm, 0, None) * days))
    fdd = float(np.nansum(np.clip(-tm, 0, None) * days))
    recs.append(dict(
        lat=r.lat, lon=r.lon,
        e5_maat=float(np.nanmean(tm)),
        e5_tdd=tdd, e5_fdd=fdd,
        e5_sqrt_tdd=float(np.sqrt(tdd)),      # Stefan 근사
        e5_twarm=float(np.nanmax(tm)), e5_tcold=float(np.nanmin(tm)),
        e5_stl1=float(np.nanmean(stl_all[:, iy, ix])),
        e5_swe=float(np.nanmean(sd_all[:, iy, ix])),
    ))
cov = pd.DataFrame(recs)
cov.to_csv("data/processed/alt_era5_covariates.csv", index=False)
n_ok = cov["e5_maat"].notna().sum() if "e5_maat" in cov else 0
print(f"공변량 산출: {len(cov)} 위치 중 {n_ok} 유효 → data/processed/alt_era5_covariates.csv")
if n_ok:
    print(cov[["e5_maat", "e5_tdd", "e5_fdd", "e5_swe"]].describe().round(1).to_string())
