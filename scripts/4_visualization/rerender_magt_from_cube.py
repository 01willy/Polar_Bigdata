"""MAGT 지도(magt_alaska_2m_20m) 재렌더 전용 스크립트 — 재학습·큐브 재계산 금지.

vol_thermal_field_alaska.py의 (2) MAGT 블록을 그대로 재현하되,
캐시된 data/processed/thermal_cube_alaska.npy(80×120×250)를 로드해 렌더만 수행한다.
격자 좌표·실측 오버레이는 결정론적으로 재구성(모델 추론 없음).
출력: outputs/maps/magt_alaska_2m_20m.png(dpi 300) + .pdf(벡터).
"""
import sys, calendar
import numpy as np
import pandas as pd
import xarray as xr
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, tnorm, FROZEN
from polar.outputs import mappath
plt = use_polar()
plt.rcParams["image.cmap"] = "cmc.batlow"
plt.rcParams["pdf.fonttype"] = 42          # PDF 텍스트를 TrueType로 임베딩

N, S, Wl, E = 72.0, 60.0, -166.0, -141.0
DEPTHS = np.arange(0.25, 20.01, 0.25)

# 엔진 라벨(제목 표기용): 원본과 동일 규칙으로 승자 표시만 재구성
engine = "GBM"
try:
    r = pd.read_csv("data/processed/b1b_results.csv")
    tr_ = r[r.split == "지역전이"].sort_values("rmse")
    if tr_.iloc[0].model.startswith("NF"):
        engine = "NF"
except Exception:
    pass

# 캐시 큐브 로드(재계산 금지)
cube = np.load("data/processed/thermal_cube_alaska.npy")

# ERA5 격자 좌표 결정론적 재구성(원본 sub 슬라이스와 동일)
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim0 = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
sub = clim0.sel(latitude=slice(N, S), longitude=slice(Wl, E))
glat, glon = sub["latitude"].values, sub["longitude"].values
ds.close()
assert cube.shape[1:] == (len(glat), len(glon)), "캐시 큐브와 격자 불일치"

# 실측 지중온도(오버레이) 결정론적 재구성
g = pd.read_csv("data/processed/ground_temp_all.csv")
g = g[(g.depth_m > 0) & (g.depth_m <= 30) & (g.temp_c > -25) & (g.temp_c < 25)].reset_index(drop=True)


def slice_at(d):
    return cube[int(np.argmin(np.abs(DEPTHS - d)))]


bh = g[(g.lat.between(S, N)) & (g.lon.between(Wl, E))]
fig, axes = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
for ax_, d, dlo, dhi in [(axes[0], 2, 1.0, 3.5), (axes[1], 20, 12, 28)]:
    Z = slice_at(d)
    mesh = ax_.pcolormesh(glon, glat, Z, cmap=CMAP.temp, norm=tnorm(-8, 4), shading="auto",
                          rasterized=True)
    cs = ax_.contour(glon, glat, Z, levels=[0], colors=FROZEN, linewidths=1.6)
    ax_.clabel(cs, fmt="0°C", fontsize=9)
    ob = bh[bh.depth_m.between(dlo, dhi)].groupby("site").agg(
        lat=("lat", "first"), lon=("lon", "first"), t=("temp_c", "mean")).reset_index()
    ax_.scatter(ob.lon, ob.lat, c=ob.t, s=55, cmap=CMAP.temp, norm=tnorm(-8, 4),
                edgecolors="k", linewidths=0.8, zorder=5)
    fig.colorbar(mesh, ax=ax_, shrink=0.9).set_label("연평균 지중온도 (°C)")
    ax_.set_ylabel("위도"); ax_.set_xlim(Wl, E); ax_.set_ylim(S, N)
    ax_.set_title(f"깊이 {d}m · 파란 0°C 등고선=영구동토 경계, 테두리점=시추공 실측({len(ob)}곳)",
                  fontsize=12, weight="bold")
axes[1].set_xlabel("경도")
fig.suptitle(f"알래스카 연평균 지중온도(MAGT) 지도 (엔진 {engine} 조건장)\n"
             "남부(온난·비동토) ↔ 북사면(한랭·연속 영구동토) 대비", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(mappath("magt_alaska_2m_20m"), dpi=300)
fig.savefig(mappath("magt_alaska_2m_20m", ext="pdf"))
plt.close(fig)
print("saved", mappath("magt_alaska_2m_20m"))
print("saved", mappath("magt_alaska_2m_20m", ext="pdf"))
