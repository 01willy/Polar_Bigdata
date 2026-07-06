"""모델 토너먼트 다방면 진단 시각화(극지 컬러맵 표준). CPU 전용.
 (A) figures/06_deep_learning/model_tournament.png — 4패널:
     ① 순위 막대 + 95% 블록부트스트랩 신뢰구간(GBM 대비 유의성) ② fold×모델 RMSE 히트맵
     ③ 모델별 부호오차 바이올린 ④ UQ 90% 커버리지 보정
 (B) figures/06_deep_learning/model_tournament_diagnostics.png — 상위3 예측-실측 1:1 hexbin
 (C) maps/model_tournament_best_vs_gbm.png — 최고모델 vs GBM |오차| 개선 지도
입력: model_tournament_results.csv, _predictions.csv, _perfold.csv
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, FROZEN, THAWED, add_cbar, style_geo
from polar.outputs import figpath, mappath
plt = use_polar()

res = pd.read_csv("data/processed/model_tournament_results.csv").sort_values("rmse_cm").reset_index(drop=True)
pred = pd.read_csv("data/processed/model_tournament_predictions.csv")
try:
    perfold = pd.read_csv("data/processed/model_tournament_perfold.csv", index_col=0)
except Exception:
    perfold = None
models = res.model.tolist()
best = models[0]
mean_rmse = np.sqrt(np.mean((pred.alt_cm - pred.alt_cm.mean()) ** 2))
print("최고:", best, "| 모델", models)

# ---------------- 블록 부트스트랩(공간구조 존중) ----------------
blk = (np.floor(pred.lat / 0.5).astype(int).astype(str) + "_" + np.floor(pred.lon / 0.5).astype(int).astype(str)).values
ublk = np.unique(blk)
idx_by = {b: np.where(blk == b)[0] for b in ublk}
se = {m: (pred[f"pred_{m}"].values - pred.alt_cm.values) ** 2 for m in models}
B = 400
rng = np.random.RandomState(0)
boot = {m: np.empty(B) for m in models}
for i in range(B):
    sb = rng.choice(ublk, len(ublk), replace=True)
    rows = np.concatenate([idx_by[b] for b in sb])
    for m in models:
        boot[m][i] = np.sqrt(se[m][rows].mean())
ci = {m: np.percentile(boot[m], [2.5, 97.5]) for m in models}
# GBM 대비 ΔRMSE 유의성(부트스트랩 쌍차이 CI가 0 배제 시 유의)
sig = {}
if "GBM" in boot:
    for m in models:
        d = boot[m] - boot["GBM"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig[m] = "유의" if (lo > 0 or hi < 0) else "동률"
pd.DataFrame({"model": models, "rmse": [res.loc[res.model == m, "rmse_cm"].iloc[0] for m in models],
              "ci_lo": [ci[m][0] for m in models], "ci_hi": [ci[m][1] for m in models],
              "vs_GBM": [sig.get(m, "-") for m in models]}).to_csv(
    "data/processed/model_tournament_significance.csv", index=False)

# ==================== (A) 4패널 진단 ====================
fig, ax = plt.subplots(2, 2, figsize=(15.5, 11))

# ① 순위 막대 + 부트스트랩 CI
r = res.iloc[::-1].reset_index(drop=True)
cols = [FROZEN if m == best else ("#9aa7b4" if sig.get(m) != "유의" or m == "GBM" else "#6b8fb0") for m in r.model]
y = np.arange(len(r))
ax[0, 0].barh(y, r.rmse_cm, color=cols, edgecolor="#333", linewidth=0.6)
ax[0, 0].errorbar(r.rmse_cm, y, xerr=[[r.rmse_cm.values[i] - ci[m][0] for i, m in enumerate(r.model)],
                                      [ci[m][1] - r.rmse_cm.values[i] for i, m in enumerate(r.model)]],
                  fmt="none", ecolor="#333", elinewidth=1.0, capsize=3)
for i, m in enumerate(r.model):
    tag = "" if m == "GBM" else f"  [{sig.get(m,'-')}]"
    ax[0, 0].text(ci[m][1] + 0.1, i, f"{r.rmse_cm.values[i]:.2f}cm{tag}", va="center", fontsize=8.5)
ax[0, 0].set_yticks(y); ax[0, 0].set_yticklabels(r.model, fontsize=9)
gbm_r = res.loc[res.model == "GBM", "rmse_cm"].iloc[0]
ax[0, 0].axvline(gbm_r, color=THAWED, ls="--", lw=1.0, alpha=0.8)
ax[0, 0].axvline(mean_rmse, color="#888", ls=":", lw=1.0)
ax[0, 0].text(mean_rmse, -0.8, "평균예측기", color="#888", fontsize=8, ha="center")
ax[0, 0].set_xlabel("공간블록 CV RMSE (cm) + 95% 블록부트스트랩 CI")
ax[0, 0].set_title("① 순위와 통계적 유의성 (GBM 대비)", weight="bold")
ax[0, 0].set_xlim(min(v[0] for v in ci.values()) - 0.4, max(v[1] for v in ci.values()) + 2.2)
ax[0, 0].grid(axis="x", alpha=0.3)

# ② fold×모델 히트맵
if perfold is not None:
    M = perfold[models].values  # (folds, models)
    im = ax[0, 1].imshow(M.T, cmap=CMAP.err, aspect="auto")
    ax[0, 1].set_xticks(range(M.shape[0])); ax[0, 1].set_xticklabels([f"f{i}" for i in range(M.shape[0])])
    ax[0, 1].set_yticks(range(len(models))); ax[0, 1].set_yticklabels(models, fontsize=9)
    ax[0, 1].grid(False)
    for i in range(len(models)):
        for j in range(M.shape[0]):
            v = M[j, i]
            ax[0, 1].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                          color="white" if v > np.nanpercentile(M, 60) else "#222")
    add_cbar(fig, im, ax[0, 1], "RMSE (cm)", shrink=0.8)
    ax[0, 1].set_xlabel("공간블록 fold (held-out 지역)")
    ax[0, 1].set_title("② fold별 변동 — 지역마다 난이도 상이(순위 불안정 원인)", weight="bold")
else:
    ax[0, 1].axis("off")

# ③ 부호오차 바이올린
sub = pred.sample(min(40000, len(pred)), random_state=0)
data = [(sub[f"pred_{m}"] - sub.alt_cm).values for m in models]
vp = ax[1, 0].violinplot(data, showmedians=True, widths=0.85)
for pc in vp["bodies"]:
    pc.set_facecolor("#7fa8c9"); pc.set_edgecolor("#333"); pc.set_alpha(0.8)
for key in ("cbars", "cmins", "cmaxes", "cmedians"):
    vp[key].set_color("#333"); vp[key].set_linewidth(0.8)
ax[1, 0].axhline(0, color=THAWED, ls="--", lw=1.0)
ax[1, 0].set_xticks(range(1, len(models) + 1)); ax[1, 0].set_xticklabels(models, rotation=20, ha="right", fontsize=8.5)
ax[1, 0].set_ylabel("예측 - 실측 (cm)"); ax[1, 0].set_ylim(-60, 60)
ax[1, 0].set_title("③ 오차 분포 — 편향(중앙선)과 산포", weight="bold")
ax[1, 0].grid(axis="y", alpha=0.3)

# ④ UQ 커버리지
uqm = res.dropna(subset=["cov90_%"]) if "cov90_%" in res else res.iloc[0:0]
if len(uqm):
    x = np.arange(len(uqm))
    ax[1, 1].bar(x, uqm["cov90_%"], color="#7fa8c9", edgecolor="#333", linewidth=0.6, width=0.6)
    ax[1, 1].axhline(90, color=FROZEN, ls="--", lw=1.2); ax[1, 1].text(len(uqm) - .5, 91, "이상적 90%", color=FROZEN, fontsize=9, ha="right")
    for i, (c, s) in enumerate(zip(uqm["cov90_%"], uqm["sharp90_cm"])):
        ax[1, 1].text(i, c + 1.5, f"{c:.0f}%\n±{s:.0f}cm", ha="center", fontsize=8.5)
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(uqm.model, rotation=18, ha="right", fontsize=9)
    ax[1, 1].set_ylabel("90% 예측구간 실제 커버리지 (%)"); ax[1, 1].set_ylim(0, 105)
    ax[1, 1].set_title("④ 불확실성 보정 — <90%는 과신(conformal 보정 필요)", weight="bold")
    ax[1, 1].grid(axis="y", alpha=0.3)
else:
    ax[1, 1].axis("off")

fig.suptitle("모델 토너먼트 진단 — 현 데이터(14 공변량)·동일 공간블록 CV·동일 게이트", fontsize=15, weight="bold")
fig.tight_layout()
fig.savefig(figpath("deep_learning", "model_tournament")); plt.close(fig)
print("saved", figpath("deep_learning", "model_tournament"))

# ==================== (B) 예측-실측 1:1 (상위 3) ====================
top3 = models[:3]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax_, m in zip(axes, top3):
    p = pred[f"pred_{m}"].values; a = pred.alt_cm.values
    hb = ax_.hexbin(a, p, gridsize=45, cmap=CMAP.count, mincnt=1, extent=(0, 150, 0, 150))
    ax_.plot([0, 150], [0, 150], color=THAWED, ls="--", lw=1.2)
    rm = res.loc[res.model == m, "rmse_cm"].iloc[0]
    ax_.set_xlim(0, 150); ax_.set_ylim(0, 150); ax_.set_aspect("equal")
    ax_.set_xlabel("실측 ALT (cm)"); ax_.set_ylabel("예측 ALT (cm)")
    ax_.set_title(f"{m}\nRMSE {rm:.2f} cm", weight="bold", fontsize=11)
    ax_.grid(alpha=0.3)
    fig.colorbar(hb, ax=ax_, shrink=0.85).set_label("점 수")
fig.suptitle("예측 vs 실측 (1:1선) — 상위 3 모델. 수평 쏠림=평균회귀(정보병목 신호)", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(figpath("deep_learning", "model_tournament_diagnostics")); plt.close(fig)
print("saved", figpath("deep_learning", "model_tournament_diagnostics"))

# ==================== (C) 최고모델 vs GBM 개선 지도 ====================
if f"pred_{best}" in pred and "pred_GBM" in pred and best != "GBM":
    pred["e_best"] = (pred[f"pred_{best}"] - pred.alt_cm).abs()
    pred["e_gbm"] = (pred.pred_GBM - pred.alt_cm).abs()
    loc = pred.groupby([pred.lat.round(2), pred.lon.round(2)]).agg(
        e_best=("e_best", "mean"), e_gbm=("e_gbm", "mean")).reset_index()
    d = loc.e_gbm - loc.e_best
    fig, ax = plt.subplots(figsize=(12, 6.2))
    vlim = np.nanpercentile(np.abs(d), 95)
    sc = ax.scatter(loc.lon, loc.lat, c=d, s=11, cmap=CMAP.diff, vmin=-vlim, vmax=vlim,
                    edgecolors="none", rasterized=True)
    add_cbar(fig, sc, ax, f"GBM |오차| - {best} |오차| (cm)  — 파랑=신모델 우세")
    style_geo(ax, f"공간블록 CV: {best} vs GBM 오차 개선 지도(위치평균)\n"
                  f"전체 RMSE {res.rmse_cm.iloc[0]:.2f} vs GBM {gbm_r:.2f} cm")
    fig.tight_layout(); fig.savefig(mappath("model_tournament_best_vs_gbm")); plt.close(fig)
    print("saved", mappath("model_tournament_best_vs_gbm"))
