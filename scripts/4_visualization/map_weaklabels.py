"""InSAR weak label(408만 점) 지도 — 사전학습 데이터가 공간적으로 어떻게 생겼나.
전 사이트 개관 + 대표 사이트 30m 디테일 줌. → outputs/maps/
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, FROZEN, add_cbar, style_geo
from polar.outputs import mappath
plt = use_polar()
# 이 환경(cmcrameri 1.8)은 batlow를 'cmc.batlow'로 등록 → 기본 image.cmap 검증 오류 방지.
if plt.rcParams.get("image.cmap") not in plt.colormaps():
    plt.rcParams["image.cmap"] = "cmc.batlow" if "cmc.batlow" in plt.colormaps() else "viridis"

wl = pd.read_parquet("data/processed/resalt_weaklabels.parquet")
obs = pd.read_csv("data/processed/dl_dataset.csv")

fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})

# ---- (1) 전 사이트 개관 ----
ax = axes[0]
sub = wl.sample(300_000, random_state=0)
sc = ax.scatter(sub.lon, sub.lat, c=sub.alt_cm, s=1.5, cmap=CMAP.alt, vmin=20, vmax=110,
                edgecolors="none", rasterized=True)
o = obs.groupby("loc_id").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
# 배경 ALT(청 계열)와 색상환 충돌 방지 → 흰 채움+검은 테두리 십자(연청/네이비 모두 대비)
ax.scatter(o.lon, o.lat, s=26, c="white", marker="P", edgecolors="#111111", linewidths=0.5,
           alpha=0.95, label=f"실측 위치 ({len(o):,})")
add_cbar(fig, sc, ax, "InSAR ALT (cm)")
style_geo(ax, f"InSAR weak label {len(wl)/1e6:.1f}M 점 (51 사이트, 표시 30만)\n"
              "— DL 사전학습 데이터의 공간 분포")
ax.legend(loc="lower left", markerscale=2)

# ---- (2) 대표 사이트 줌(30m 디테일) ----
ax = axes[1]
big = wl[wl.site == "tukhwy"]          # Tuktoyaktuk Highway — 대형 사이트
if len(big) < 1000:
    big = wl[wl.site == wl.site.value_counts().index[0]]
sc = ax.scatter(big.lon, big.lat, c=big.alt_cm, s=3, cmap=CMAP.alt, vmin=20, vmax=110,
                edgecolors="none", rasterized=True)
add_cbar(fig, sc, ax, "InSAR ALT (cm)")
style_geo(ax, f"사이트 '{big.site.iloc[0]}' 줌 — 30m 해상도 sub-grid 변동\n"
              "(기존 공변량 9km가 못 담던 디테일)")

fig.suptitle("Phase A2: 사전학습용 InSAR weak label — 실측(흰 십자) 대비 285배 밀도", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(mappath("weaklabels_overview"))
plt.close(fig)
print("saved", mappath("weaklabels_overview"))
