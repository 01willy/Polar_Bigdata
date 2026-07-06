"""연속 ALT 예측 표면(직관적 2D 레이어) — 알래스카 북사면 밀집구역.
ERA5-Land를 고해상(0.02°) 보간 → 공변량 산출 → GBM 예측 → 매끄러운 색면 + 관측점 오버레이.
→ outputs/maps/alt_surface_northslope.png
"""
import sys, calendar
import numpy as np
import pandas as pd
import xarray as xr
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, add_cbar, style_geo, BAD
from polar.outputs import mappath
plt = use_polar()
plt.rcParams["image.cmap"] = "cmc.batlow"  # 기본 컬러맵 이름을 이 mpl 버전에 등록된 형태로(contour 폴백용)
from sklearn.ensemble import HistGradientBoostingRegressor

E5F = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
N, S, Wl, E = 72, 68, -162, -143      # 북사면

# 학습(북미 dl_dataset의 ERA5 피처)
df = pd.read_csv("data/processed/dl_dataset.csv")
gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(df[E5F], df["y"])

# ERA5 고해상 보간 격자
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
clim = clim.sel(latitude=slice(N, S), longitude=slice(Wl, E))
glat = np.arange(S, N, 0.02); glon = np.arange(Wl, E, 0.02)
fine = clim.interp(latitude=glat, longitude=glon, method="linear")
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]
t = fine["t2m"].values - 273.15; stl = fine["stl1"].values - 273.15; sd = fine["sd"].values
tdd = np.nansum(np.clip(t, 0, None) * days, 0); fdd = np.nansum(np.clip(-t, 0, None) * days, 0)
G = {"e5_maat": np.nanmean(t, 0), "e5_tdd": tdd, "e5_fdd": fdd, "e5_sqrt_tdd": np.sqrt(tdd),
     "e5_twarm": np.nanmax(t, 0), "e5_tcold": np.nanmin(t, 0),
     "e5_stl1": np.nanmean(stl, 0), "e5_swe": np.nanmean(sd, 0)}
LAT, LON = G["e5_maat"].shape
Xg = np.column_stack([G[f].ravel() for f in E5F])
land = ~np.isnan(Xg).any(1)
alt = np.full(LAT * LON, np.nan)
alt[land] = np.expm1(np.clip(gbm.predict(Xg[land]), np.log1p(1), np.log1p(600)))
alt = alt.reshape(LAT, LON)
print(f"표면 예측: {land.sum():,} 육지셀, ALT {np.nanmin(alt):.0f}~{np.nanmax(alt):.0f}cm")

# 렌더
fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_facecolor(BAD)   # 해양/결측(NaN) = 중립 회색 배경 → 저ALT 육지와 명확히 구분
mesh = ax.pcolormesh(glon, glat, alt, cmap=CMAP.alt, vmin=20, vmax=90, shading="auto")
cs = ax.contour(glon, glat, alt, levels=[30, 45, 60, 75], colors="#3a3a3a", linewidths=0.5, alpha=0.6)
ax.clabel(cs, fmt="%dcm", fontsize=7.5)
cb = add_cbar(fig, mesh, ax, "예측 활성층 두께 ALT (cm)")
# 관측 오버레이
obs = df[(df.lat.between(S, N)) & (df.lon.between(Wl, E))].groupby("loc_id").agg(
    lat=("lat", "mean"), lon=("lon", "mean"), alt=("alt_cm", "mean")).reset_index()
ax.scatter(obs.lon, obs.lat, c=obs.alt, s=20, cmap=CMAP.alt, vmin=20, vmax=90,
           edgecolors="#222222", linewidths=0.5, zorder=5)
ax.set_xlim(Wl, E); ax.set_ylim(S, N)
style_geo(ax, xlabel="경도 (°E)", ylabel="위도 (°N)")
ax.set_title(f"알래스카 북사면 활성층 두께(ALT) 예측 표면 — 연속 2D 레이어\n"
             f"(색면=예측, 테두리점=실측 관측 {len(obs)}곳, 등고선=ALT)")
fig.tight_layout(); fig.savefig(mappath("alt_surface_northslope"))
plt.close(fig)
print("saved", mappath("alt_surface_northslope"))
