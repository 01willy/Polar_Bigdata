"""공간 DL 결과 분석 시각화: (1) 패치CNN vs GBM 공간블록 RMSE, (2) 변수 중요도.
→ outputs/figures/06_spatial_dl/
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, FROZEN, THAWED
from polar.outputs import figpath
plt = use_polar()

# 범주 강조색: 지리 컬러맵 대신 한대 순차 팔레트(CMAP.count)에서 추출해 일관성 유지.
#   CNN=짙은 청록(강조), GBM=옅은 청록 — 붉은 임의색 제거.
C_CNN = CMAP.count(0.72)   # 짙은 청록 (제안 모델 강조)
C_GBM = CMAP.count(0.32)   # 옅은 청록 (기준 모델)
# 변수 그룹 강조색: 동결/융해 대비색 사용 — 기후(ERA5)=동결계열, 지형(DEM)=융해계열.
C_TERR = THAWED            # 지형(DEM)
C_CLIM = FROZEN            # 기후(ERA5)

SCAL = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
NAMES = {"dem_elev": "고도", "dem_slope": "경사", "dem_aspect_sin": "향(sin)", "dem_aspect_cos": "향(cos)",
         "dem_tpi": "지형위치(TPI)", "dem_rough": "거칠기", "e5_maat": "연평균기온",
         "e5_tdd": "융해도일", "e5_fdd": "동결도일", "e5_sqrt_tdd": "√융해도일", "e5_twarm": "최난월",
         "e5_tcold": "최한월", "e5_stl1": "토양온도", "e5_swe": "적설"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (1) CNN vs GBM
res = pd.read_csv("data/processed/dl_cnn_results.csv")
ax = axes[0]
c = [C_CNN if "CNN" in m else C_GBM for m in res.model]
b = ax.bar(range(len(res)), res.rmse, color=c, edgecolor="#444444", linewidth=0.6, width=0.62)
ax.set_xticks(range(len(res))); ax.set_xticklabels(["패치 CNN\n(DEM패치+ERA5/지형)", "GBM\n(스칼라)"], fontsize=10)
for bar, v in zip(b, res.rmse):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}", ha="center", weight="bold")
ax.set_ylabel("ALT RMSE (cm) — 공간블록 CV")
ax.set_ylim(0, max(res.rmse) * 1.15)
ax.set_title("공간 DL vs GBM (북미 ALT)\n패치(공간맥락)가 도움되나?", weight="bold")
ax.grid(alpha=0.35, axis="y"); ax.set_axisbelow(True)

# (2) 변수 중요도 (GBM permutation, 공간블록 홀드아웃 샘플)
df = pd.read_csv("data/processed/dl_dataset.csv")
block = (np.floor(df.lat/0.5).astype(int).astype(str)+"_"+np.floor(df.lon/0.5).astype(int).astype(str))
ub = block.unique(); rng = np.random.RandomState(0)
te_b = set(rng.choice(ub, max(1, len(ub)//4), replace=False))
tr = df[~block.isin(te_b)]; te = df[block.isin(te_b)].sample(min(5000, (block.isin(te_b)).sum()), random_state=0)
g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(tr[SCAL], tr["y"])
pi = permutation_importance(g, te[SCAL], te["y"], n_repeats=5, random_state=0, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=[NAMES[s] for s in SCAL]).sort_values()
ax = axes[1]
_TERR_LBL = ["고도", "경사", "향(sin)", "향(cos)", "지형위치(TPI)", "거칠기"]
colors = [C_TERR if lbl in _TERR_LBL else C_CLIM for lbl in imp.index]
ax.barh(range(len(imp)), imp.values, color=colors, edgecolor="#444444", linewidth=0.5, height=0.72)
ax.set_yticks(range(len(imp))); ax.set_yticklabels(imp.index, fontsize=9)
ax.set_xlabel("중요도(순열, 홀드아웃)")
ax.set_title("변수 중요도 — 지형(DEM) vs 기후(ERA5)", weight="bold")
ax.grid(alpha=0.35, axis="x"); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=C_TERR, edgecolor="#444444", label="지형 (DEM)"),
                   Patch(facecolor=C_CLIM, edgecolor="#444444", label="기후 (ERA5)")],
          loc="lower right", fontsize=9)

fig.suptitle("공간 DL 결과 분석", fontsize=13, weight="bold")
fig.tight_layout(); fig.savefig(figpath("spatial_dl", "cnn_vs_gbm_importance"))
plt.close(fig)
print("saved", figpath("spatial_dl", "cnn_vs_gbm_importance"))
print("\nCNN vs GBM:\n", res.to_string(index=False))
print("\n지형 vs 기후 중요도 상위:\n", imp.sort_values(ascending=False).head(8).to_string())
