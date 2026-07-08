"""스레드 R ㉠ — 시간정합 공변량: 각 (위치,연도)에 '그 해' 기후 파생.

기존 era5land_covariates.py는 2015-2020 '평균'(정적)을 붙임 → 연도차가 잡음.
여기선 nh_monthly_2010-2024.nc에서 **연도별** 12개월값으로 도일/적설/토양온도를 계산,
각 (고유위치, 연도)에 정합. 추가로 **전년 겨울 적설**(단열)도 파생.

출력: data/processed/alt_era5_temporal.csv  (lat, lon, year → e5t_* 시변 공변량)
실행: python3 scripts/1_data_prep/era5land_temporal_covariates.py
"""
import os, calendar
import numpy as np
import pandas as pd
import xarray as xr

PROC = "data/processed"
NC = "data/raw/era5land/nh_monthly_2010-2024.nc"
ds = xr.open_dataset(NC)
tname = "valid_time" if "valid_time" in ds.coords else "time"
ds = ds.assign_coords(yr=ds[tname].dt.year, mo=ds[tname].dt.month)
lat = ds["latitude"].values
lon = ds["longitude"].values
YEARS = np.unique(ds["yr"].values)
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])

# 라벨의 (고유위치, 연도) 집합
lab = pd.read_csv(os.path.join(PROC, "dl_dataset.csv"), usecols=["lat", "lon", "year", "loc_id"])
lab = lab[(lab.year >= YEARS.min()) & (lab.year <= YEARS.max())]
locs = lab[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
print(f"고유 위치 {len(locs):,} · 연도 {YEARS.min()}–{YEARS.max()} · (위치,연도) {lab.groupby(['loc_id','year']).ngroups:,}")

# 위치별 최근접 격자 인덱스(한 번만)
iy = np.array([int(np.abs(lat - v).argmin()) for v in locs.lat])
ix = np.array([int(np.abs(lon - v).argmin()) for v in locs.lon])

# 연도별 월 큐브 미리 추출: t2m/stl1(°C), sd(m). shape (year, month, lat, lon)
def cube(var):
    a = ds[var].values  # (time, lat, lon), time=180 = 15yr×12mo (연도오름차 월오름차 가정)
    return a.reshape(len(YEARS), 12, a.shape[1], a.shape[2])

t2m = cube("t2m") - 273.15
stl = cube("stl1") - 273.15
sd = cube("sd")

recs = []
for yi, yr in enumerate(YEARS):
    tm = t2m[yi][:, iy, ix]     # (12, nloc)
    gm = stl[yi][:, iy, ix]
    sm = sd[yi][:, iy, ix]
    tdd = np.nansum(np.clip(tm, 0, None) * days[:, None], axis=0)
    fdd = np.nansum(np.clip(-tm, 0, None) * days[:, None], axis=0)
    # 전년 겨울(전년 10~12 + 당해 1~4월) 적설 평균 = thaw 직전 단열
    if yi > 0:
        prev_winter = np.concatenate([sd[yi - 1][9:12], sd[yi][0:4]], axis=0)[:, iy, ix]
    else:
        prev_winter = sm[0:4]
    df = pd.DataFrame({
        "lat": locs.lat.values, "lon": locs.lon.values, "year": int(yr),
        "e5t_maat": np.nanmean(tm, axis=0), "e5t_tdd": tdd, "e5t_fdd": fdd,
        "e5t_sqrt_tdd": np.sqrt(np.clip(tdd, 0, None)),
        "e5t_twarm": np.nanmax(tm, axis=0), "e5t_tcold": np.nanmin(tm, axis=0),
        "e5t_stl1": np.nanmean(gm, axis=0), "e5t_swe": np.nanmean(sm, axis=0),
        "e5t_swe_prevwinter": np.nanmean(prev_winter, axis=0),
    })
    recs.append(df)
cov = pd.concat(recs, ignore_index=True)
cov = cov.dropna(subset=["e5t_maat"])   # 해양/결측 제거
out = os.path.join(PROC, "alt_era5_temporal.csv")
cov.to_csv(out, index=False)
print(f"시간정합 공변량: {len(cov):,}행 (위치×연도) → {out}")
print(cov[["e5t_tdd", "e5t_fdd", "e5t_swe", "e5t_swe_prevwinter"]].describe().round(1).to_string())
# 연도차 실재 확인: 같은 위치의 연도별 TDD 표준편차
sd_tdd = cov.groupby(["lat", "lon"]).e5t_tdd.std().mean()
print(f"\n같은 위치의 연도별 TDD 표준편차 평균 = {sd_tdd:.0f} (>0이면 연도신호 존재 → 시간정합 의미)")
