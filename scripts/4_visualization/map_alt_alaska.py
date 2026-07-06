"""직관적 2D 레이어: 알래스카 ALT 예측 지도.
ERA5-Land 격자에서 공변량(도일/적설/토양온도) 산출 → ERA5 GBM으로 셀별 ALT(cm) 예측 →
실제 지도 위 색으로 렌더 + 관측점(CALM/ABoVE) 오버레이.  → outputs/maps/
"""
import os, sys, calendar
import numpy as np
import pandas as pd
import xarray as xr
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, add_cbar, style_geo, THAWED
from polar.outputs import mappath
plt = use_polar()
from sklearn.ensemble import HistGradientBoostingRegressor

E5F = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
BBOX = dict(n=72, s=60, w=-168, e=-140)   # 알래스카

# ---------- 1) 학습: CALM 사이트 + ERA5 공변량으로 GBM ----------
df = pd.read_csv("data/processed/alt_global.csv")
e5 = pd.read_csv("data/processed/alt_era5_covariates.csv")
for d in (df, e5):
    d["_k"] = d["lat"].round(4).astype(str) + "_" + d["lon"].round(4).astype(str)
df = df.merge(e5[["_k"] + E5F], on="_k", how="left").dropna(subset=E5F + ["alt_cm"])
y = np.log1p(df["alt_cm"].to_numpy())
gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                    max_leaf_nodes=31, random_state=0).fit(df[E5F], y)
print(f"GBM 학습: {len(df)} site-year (ERA5 공변량)")

# ---------- 2) ERA5-Land 격자에서 공변량 산출(알래스카) ----------
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tname = "valid_time" if "valid_time" in ds.coords else "time"
ds = ds.assign_coords(month=ds[tname].dt.month).groupby("month").mean(tname)
ds = ds.sel(latitude=slice(BBOX["n"], BBOX["s"]), longitude=slice(BBOX["w"], BBOX["e"]))
lat = ds["latitude"].values; lon = ds["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]

t = ds["t2m"].values - 273.15         # (12, LAT, LON) °C
stl = ds["stl1"].values - 273.15
swe = ds["sd"].values
maat = np.nanmean(t, 0); tdd = np.nansum(np.clip(t, 0, None) * days, 0)
fdd = np.nansum(np.clip(-t, 0, None) * days, 0)
grids = {"e5_maat": maat, "e5_tdd": tdd, "e5_fdd": fdd, "e5_sqrt_tdd": np.sqrt(tdd),
         "e5_twarm": np.nanmax(t, 0), "e5_tcold": np.nanmin(t, 0),
         "e5_stl1": np.nanmean(stl, 0), "e5_swe": np.nanmean(swe, 0)}

LATN, LONN = maat.shape
X = np.column_stack([grids[f].ravel() for f in E5F])
land = ~np.isnan(X).any(1)                     # ERA5-Land: 육지만 유효(해양 NaN=자연 마스크)
pred = np.full(LATN * LONN, np.nan)
pred[land] = np.expm1(np.clip(gbm.predict(X[land]), np.log1p(1), np.log1p(600)))
alt_grid = pred.reshape(LATN, LONN)
print(f"격자 예측: {land.sum():,} 육지셀, ALT {np.nanmin(alt_grid):.0f}~{np.nanmax(alt_grid):.0f}cm "
      f"(중앙 {np.nanmedian(alt_grid):.0f})")

# ---------- 3) 2D 지도 렌더 + 관측점 오버레이 ----------
fig, ax = plt.subplots(figsize=(12, 7))
extent = [lon.min(), lon.max(), lat.min(), lat.max()]
im = ax.imshow(alt_grid, extent=extent, origin="upper", cmap=CMAP.alt,
               vmin=20, vmax=110, aspect="auto")
add_cbar(fig, im, ax, "예측 활성층 두께 ALT (cm)  — 짙을수록 깊음(활성층 두꺼움)")

# 관측점 오버레이 — 위치 마커에 온도의미색(적) 재사용 금지 → 흰 채움+검은 테두리(중립 고대비)
calm = df[(df.lat.between(BBOX["s"], BBOX["n"])) & (df.lon.between(BBOX["w"], BBOX["e"]))]
ax.scatter(calm.lon, calm.lat, s=34, c="white", marker="o", edgecolors="#111111",
           linewidths=0.8, label=f"CALM 관측 ({len(calm)})", zorder=5)
try:
    ab = pd.read_csv("data/processed/alt_above_siteyear.csv")
    ab = ab[(ab.lat.between(BBOX["s"], BBOX["n"])) & (ab.lon.between(BBOX["w"], BBOX["e"]))]
    ax.scatter(ab.lon, ab.lat, s=22, c="#ffd166", marker="^", edgecolors="#333333",
               linewidths=0.5, label=f"ABoVE 관측 ({ab.lat.nunique()})", zorder=4, alpha=0.95)
except Exception:
    pass

style_geo(ax, title="알래스카 활성층 두께(ALT) 예측 지도 — ERA5-Land 공변량 + GBM\n"
                    "(회색=해양/결측, 옅은청→짙은청=얕음→깊음)")
ax.legend(loc="lower left", fontsize=9)
fig.savefig(mappath("alt_alaska_pred"))
plt.close(fig)
print("saved", mappath("alt_alaska_pred"))
