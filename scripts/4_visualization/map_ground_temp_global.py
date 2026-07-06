"""전지구 3D 라벨(지중온도 프로파일) 지도 + 깊이 커버리지 — Phase A1 결과 직관 시각화.
→ outputs/maps/ground_temp_global.png, figures/01_data/ground_temp_depth_coverage.png
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, tnorm, FROZEN, add_cbar, style_geo
from polar.outputs import mappath, figpath
plt = use_polar()
# 이 matplotlib/cmcrameri 조합에서 기본 cmap은 'cmc.' 접두어가 필요 — 잘못된 기본값 보정(스타일만)
plt.rcParams["image.cmap"] = "cmc.batlow"

g = pd.read_csv("data/processed/ground_temp_all.csv")
sites = g.groupby(["source", "site"]).agg(lat=("lat", "first"), lon=("lon", "first"),
                                          maxd=("depth_m", "max"), n=("depth_m", "size")).reset_index()

# ---- (1) 세계지도: 소스별 사이트 + 10m 온도 색 ----
m10 = g[(g.depth_m.between(5, 20))].groupby(["source", "site"]).agg(
    lat=("lat", "first"), lon=("lon", "first"), t=("temp_c", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(14, 6.5))
mk = {"GTNP": "o", "GGD200": "^", "PERMOS": "s"}
tn = tnorm(-6, 3)  # 0°C(동결/융해 경계)=흰색 중심. 관측 대부분 −6~+3°C라 대비 확보
for src, s in m10.groupby("source"):
    sc = ax.scatter(s.lon, s.lat, c=s.t, s=34 if src != "GTNP" else 26, marker=mk.get(src, "o"),
                    cmap=CMAP.temp, norm=tn, edgecolors="k", linewidths=0.4,
                    label=f"{src} ({s.site.nunique()} 사이트)")
cb = add_cbar(fig, sc, ax, "5~20m 평균 지중온도 (°C) — 청=동결(영구동토), 적=융해")
cb.ax.axhline(tn(0.0), color=FROZEN, lw=1.0)  # 0°C 경계 표시
ax.axhline(66.56, color=FROZEN, ls="--", lw=0.8, alpha=0.7)  # 북극권
ax.text(-178, 67.6, "북극권 66.6°N", color=FROZEN, fontsize=8, alpha=0.8)
ax.set_xlim(-172, 180); ax.set_ylim(28, 84)  # 관측 밀집대(북반구 한대)로 크롭 — 남반구 공백 제거
style_geo(ax,
          title=f"전지구 3D 라벨(깊이별 지중온도) — {g.site.nunique()}개 사이트 / {len(g):,} 점 / 8개국+α\n"
                "(Phase A1: GTN-P 전지구 파싱 + 서시베리아 + 알프스 통합)")
ax.legend(loc="lower left", fontsize=9)
fig.savefig(mappath("ground_temp_global")); plt.close(fig)
print("saved", mappath("ground_temp_global"))

# ---- (2) 지역×깊이 커버리지 ----
g["depth_bin"] = pd.cut(g.depth_m, [0, 1, 2, 5, 10, 20, 50, 140, 2000],
                        labels=["0-1", "1-2", "2-5", "5-10", "10-20", "20-50", "50-140", ">140"])
g.loc[g.source == "GTNP", "region2"] = g.loc[g.source == "GTNP", "region"]
g.loc[g.source != "GTNP", "region2"] = g.loc[g.source != "GTNP", "region"]
top = g.region2.value_counts().head(8).index
piv = (g[g.region2.isin(top)].groupby(["region2", "depth_bin"], observed=True)
        .size().unstack(fill_value=0).reindex(top))
fig, ax = plt.subplots(figsize=(10.5, 5.5))
logv = np.log10(piv.values + 1)
im = ax.imshow(logv, cmap=CMAP.count, aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=9)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=9)
ax.grid(False)
_thr = 0.62 * logv.max()  # 짙은 칸(높은 밀도) 위에서만 흰 글씨
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if v:
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=7,
                    color="white" if logv[i, j] > _thr else "#222222")
ax.set_xlabel("깊이 구간 (m)"); ax.set_ylabel("지역")
ax.set_title("지역 × 깊이 라벨 커버리지 (점 수) — 3D 신경장 학습 가능 범위")
add_cbar(fig, im, ax, "log10(점 수 + 1)", shrink=0.8)
fig.savefig(figpath("data", "ground_temp_depth_coverage")); plt.close(fig)
print("saved", figpath("data", "ground_temp_depth_coverage"))
