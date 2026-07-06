"""공간 DL 평가 결과를 '지도 레이어'로 직관 시각화(그래프 아님):
 (1) 공간블록 CV 구성  (2) 관측 vs 예측 vs 오차 ALT 지도  (3) 정확도(예측vs관측).
OOF(홀드아웃) 예측 = 공간블록 GroupKFold + GBM. → outputs/maps/, figures/06_spatial_dl/
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, tnorm, add_cbar, style_geo, FROZEN
from polar.outputs import mappath, figpath
plt = use_polar()
# 이 matplotlib(3.8)는 Crameri 맵을 'cmc.batlow'로 등록 → 기본 image.cmap을 유효값으로 정렬
plt.rcParams["image.cmap"] = "cmc.batlow"

SCAL = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
df = pd.read_csv("data/processed/dl_dataset.csv").reset_index(drop=True)
block = (np.floor(df.lat / 0.5).astype(int).astype(str) + "_" + np.floor(df.lon / 0.5).astype(int).astype(str))
X = df[SCAL].to_numpy(); y = df["y"].to_numpy(); alt = df["alt_cm"].to_numpy()

# ---- OOF 예측 + 폴드 라벨 ----
oof = np.full(len(df), np.nan); fold_of = np.full(len(df), -1)
gkf = GroupKFold(4)
for f, (tr, te) in enumerate(gkf.split(df, groups=block)):
    g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X[tr], y[tr])
    oof[te] = np.expm1(np.clip(g.predict(X[te]), np.log1p(1), np.log1p(600)))
    fold_of[te] = f
df["pred"] = oof; df["fold"] = fold_of; df["err"] = df.pred - alt

# ---- 위치별 집계(과밀 방지) ----
loc = df.groupby("loc_id").agg(lat=("lat", "mean"), lon=("lon", "mean"),
                               obs=("alt_cm", "mean"), pred=("pred", "mean"),
                               fold=("fold", "first"), n=("alt_cm", "size")).reset_index()
loc["err"] = loc.pred - loc.obs
ext = [loc.lon.min()-1, loc.lon.max()+1, loc.lat.min()-0.5, loc.lat.max()+0.5]
rmse = float(np.sqrt(np.mean(df.err**2)))
print(f"위치 {len(loc)}개, OOF RMSE {rmse:.1f}cm")

def base(ax):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    style_geo(ax)

# ===== (1) 공간블록 CV 지도 =====
fig, ax = plt.subplots(figsize=(11, 6.5))
cols = plt.cm.tab10(np.arange(4))
for f in range(4):
    s = loc[loc.fold == f]
    ax.scatter(s.lon, s.lat, s=16, color=cols[f], label=f"블록그룹 {f+1}",
               alpha=0.85, edgecolors="white", linewidths=0.3)
for gx in np.arange(np.floor(ext[0]), ext[1], 0.5):
    ax.axvline(gx, color="#bbbbbb", lw=0.3, alpha=0.4, zorder=0)
for gy in np.arange(np.floor(ext[2]), ext[3], 0.5):
    ax.axhline(gy, color="#bbbbbb", lw=0.3, alpha=0.4, zorder=0)
base(ax)
ax.set_title("공간블록 교차검증 구성 — 50km 블록을 4그룹으로 분할\n"
             "(같은 블록은 학습/시험에 섞지 않음 = 공간 누설 차단)")
ax.legend(loc="lower left", markerscale=1.6, title="폴드")
fig.tight_layout(); fig.savefig(mappath("spatial_cv_blocks")); plt.close(fig)
print("saved", mappath("spatial_cv_blocks"))

# ===== (2) 관측 / 예측 / 오차 ALT 지도 =====
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
vmax = np.nanpercentile(loc.obs, 97)
# 오차(부호 있는 편차) = 발산맵, 0중심. 대칭 스케일로 편향 방향 직관화.
enorm = tnorm(-40, 40)
for ax, col, ttl, cmap, kw, lab in [
    (axes[0], "obs", "관측 ALT (정답)", CMAP.alt, dict(vmin=20, vmax=vmax), "ALT (cm)"),
    (axes[1], "pred", "예측 ALT (OOF)", CMAP.alt, dict(vmin=20, vmax=vmax), "ALT (cm)"),
    (axes[2], "err", "오차 (예측 - 관측)", CMAP.diff, dict(norm=enorm), "편차 (cm)")]:
    sc = ax.scatter(loc.lon, loc.lat, c=loc[col], s=18, cmap=cmap, **kw,
                    edgecolors="white", linewidths=0.2)
    add_cbar(fig, sc, ax, lab)
    base(ax); ax.set_title(ttl)
fig.suptitle(f"북미 활성층 두께 ALT — 관측 vs 예측 vs 오차 지도 (위치 {len(loc):,}개, OOF RMSE {rmse:.1f}cm)",
             fontsize=14, weight="bold")
fig.tight_layout(); fig.savefig(mappath("alt_obs_pred_error")); plt.close(fig)
print("saved", mappath("alt_obs_pred_error"))

# ===== (3) 정확도: 예측 vs 관측 =====
fig, ax = plt.subplots(figsize=(6.5, 6))
hb = ax.hexbin(df.alt_cm, df.pred, gridsize=45, bins="log", cmap=CMAP.count, extent=(0, 200, 0, 200))
ax.plot([0, 200], [0, 200], "--", color=FROZEN, lw=1.4, label="완벽예측(1:1)")
ax.set_xlabel("관측 ALT (cm)"); ax.set_ylabel("예측 ALT (cm)")
ax.set_xlim(0, 200); ax.set_ylim(0, 200)
ax.set_aspect("equal")
ax.set_title(f"예측 정확도 (RMSE {rmse:.1f}cm)")
add_cbar(fig, hb, ax, "점 밀도 (log 개수)")
ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(figpath("spatial_dl", "pred_vs_obs")); plt.close(fig)
print("saved", figpath("spatial_dl", "pred_vs_obs"))
