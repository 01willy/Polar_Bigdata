"""횡단 — AOA(적용가능영역) + Conformal UQ (기술 차별성 핵심).

두 스토리:
 (1) within-domain 보정 UQ: quantile-GBM → CQR(Romano 2019)로 90% 구간을 보정.
     생성모델 raw 커버리지(Diffusion 74% 과신)를 conformal로 90%에 맞춘다.
 (2) transfer AOA: LORO에서 held-out region의 각 점이 학습 환경공간 안(inside)인지 밖(outside)인지
     AOA(Meyer 2021 dissimilarity index)로 판정 → AOA 밖은 RMSE↑·coverage↓ (정직한 외삽 표기).

실행: python3 scripts/2_evaluation/aoa_conformal_alt.py
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from polar.eval_metrics import all_metrics, coverage, interval_width
from polar.outputs import figpath, mappath
from polar.plotstyle import use_polar, CMAP, lon_formatter, lat_formatter, despine

plt = use_polar()
PROC = "data/processed"
CLIP = (np.log1p(1), np.log1p(600))
ALPHA = 0.10  # 90% 구간

base = pd.read_csv(os.path.join(PROC, "dl_dataset.csv"))
BASE14 = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
          "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
X = base[BASE14].values
y = base["alt_cm"].values
ylog = np.log1p(y)
base["block"] = (np.floor(base.lat / 0.5).astype(int) * 100000 + np.floor(base.lon / 0.5).astype(int))
to_cm = lambda p: np.expm1(np.clip(p, *CLIP))


def gbm(q=None):
    if q is None:
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                                             l2_regularization=1.0, early_stopping=True, random_state=0)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=63, l2_regularization=1.0, random_state=0)


# ================= Part 1: within-domain CQR =================
print("=== Part 1: within-domain 보정 UQ (spatial-block CQR) ===")
gkf = GroupKFold(n_splits=6)
oof_pt = np.full(len(base), np.nan); oof_lo = np.full(len(base), np.nan); oof_hi = np.full(len(base), np.nan)
for tr, te in gkf.split(X, ylog, groups=base.block.values):
    oof_pt[te] = to_cm(gbm().fit(X[tr], ylog[tr]).predict(X[te]))
    oof_lo[te] = to_cm(gbm(ALPHA / 2).fit(X[tr], ylog[tr]).predict(X[te]))
    oof_hi[te] = to_cm(gbm(1 - ALPHA / 2).fit(X[tr], ylog[tr]).predict(X[te]))

blocks = base.block.values
ub = pd.unique(blocks)
rng = np.random.default_rng(0); rng.shuffle(ub)
cal_b = set(ub[: len(ub) // 2]); calm = np.array([b in cal_b for b in blocks])  # calib/test 블록 분리
# CQR conformity score: E = max(lo - y, y - hi)
E = np.maximum(oof_lo - y, y - oof_hi)
n_cal = calm.sum()
qlevel = min(1.0, np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal)
Q = np.quantile(E[calm], qlevel)
lo_c = oof_lo - Q; hi_c = oof_hi + Q
tst = ~calm
raw_cov = coverage(y[tst], oof_lo[tst], oof_hi[tst])
cqr_cov = coverage(y[tst], lo_c[tst], hi_c[tst])
raw_w = interval_width(oof_lo[tst], oof_hi[tst]); cqr_w = interval_width(lo_c[tst], hi_c[tst])
print(f"  raw quantile-GBM  coverage={raw_cov*100:.1f}%  width={raw_w:.1f}cm")
print(f"  CQR 보정          coverage={cqr_cov*100:.1f}%  width={cqr_w:.1f}cm  (목표 90%)")

# ================= Part 2: transfer AOA (LORO) =================
print("\n=== Part 2: transfer AOA (LORO, 14 환경변수) ===")
mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
Xs = (X - mu) / sd
reg = base.region.values
aoa_rows = []
di_all = np.full(len(base), np.nan); aoa_all = np.full(len(base), np.nan)
predT = np.full(len(base), np.nan); loT = np.full(len(base), np.nan); hiT = np.full(len(base), np.nan)
for r in pd.unique(reg):
    te = np.where(reg == r)[0]; tr = np.where(reg != r)[0]
    if len(te) < 200:
        continue
    # AOA: train 환경공간에서의 dissimilarity index
    nn = NearestNeighbors(n_neighbors=2).fit(Xs[tr])
    d_tr, _ = nn.kneighbors(Xs[tr]); d_bar = d_tr[:, 1].mean()  # 최근접 '다른' train점
    di_tr = d_tr[:, 1] / d_bar
    thr = np.quantile(di_tr, 0.75) + 1.5 * (np.quantile(di_tr, 0.75) - np.quantile(di_tr, 0.25))
    d_te, _ = nn.kneighbors(Xs[te], n_neighbors=1); di_te = d_te[:, 0] / d_bar
    inside = di_te <= thr
    di_all[te] = di_te; aoa_all[te] = inside.astype(float)
    # 예측 + CQR 구간(train regions로 보정)
    pt = to_cm(gbm().fit(X[tr], ylog[tr]).predict(X[te]))
    lo = to_cm(gbm(ALPHA / 2).fit(X[tr], ylog[tr]).predict(X[te]))
    hi = to_cm(gbm(1 - ALPHA / 2).fit(X[tr], ylog[tr]).predict(X[te]))
    predT[te] = pt; loT[te] = lo; hiT[te] = hi
    if inside.sum() > 30 and (~inside).sum() > 30:
        aoa_rows.append({"region": r, "n": len(te), "pct_inside": round(100 * inside.mean(), 1),
                         "rmse_in": all_metrics(y[te][inside], pt[inside])["rmse_cm"],
                         "rmse_out": all_metrics(y[te][~inside], pt[~inside])["rmse_cm"],
                         "cov_in": round(100 * coverage(y[te][inside], lo[inside], hi[inside]), 1),
                         "cov_out": round(100 * coverage(y[te][~inside], lo[~inside], hi[~inside]), 1)})
aoa_df = pd.DataFrame(aoa_rows)
if len(aoa_df):
    print(aoa_df.to_string(index=False))
    print(f"  집계: AOA안 RMSE {aoa_df.rmse_in.mean():.1f} < 밖 {aoa_df.rmse_out.mean():.1f}cm | "
          f"cov 안 {aoa_df.cov_in.mean():.0f}% > 밖 {aoa_df.cov_out.mean():.0f}%")

# 결과 저장
pd.DataFrame([
    {"setting": "within-domain raw quantile", "coverage_pct": round(raw_cov * 100, 1), "width_cm": round(raw_w, 1)},
    {"setting": "within-domain CQR(보정)", "coverage_pct": round(cqr_cov * 100, 1), "width_cm": round(cqr_w, 1)},
]).to_csv(os.path.join(PROC, "alt_conformal_aoa_results.csv"), index=False)
if len(aoa_df):
    aoa_df.to_csv(os.path.join(PROC, "alt_aoa_transfer_results.csv"), index=False)

# ================= 시각화 =================
# Fig1: coverage 보정
fig, ax = plt.subplots(figsize=(6.4, 4.6))
bars = ax.bar([0, 1], [raw_cov * 100, cqr_cov * 100], color=["#8a8f98", "#0b7285"], width=0.55)
ax.axhline(90, color="#b5651d", ls="--", lw=1.8, label="목표 90%")
ax.set_xticks([0, 1]); ax.set_xticklabels(["raw quantile-GBM\n(비보정)", "CQR 보정\n(conformal)"], fontsize=10)
ax.set_ylabel("관측 커버리지 (%)", fontsize=11); ax.set_ylim(0, 100)
for i, v in enumerate([raw_cov * 100, cqr_cov * 100]):
    if v > 85:  # 90선/범례와 겹침 방지 → 막대 안쪽 흰글씨
        ax.text(i, v - 5, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold", color="white")
    else:
        ax.text(i, v + 1.8, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
ax.annotate(f"n(calib)={n_cal}, n(test)={int(tst.sum())}", (0.5, 4), ha="center", fontsize=8, color="#555")
ax.set_title("보정된 불확실성: 90% 구간이 실제 90%를 담는가\n(생성모델 raw는 과신 → conformal이 교정)", fontsize=13, fontweight="bold")
despine(ax); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(figpath("eval", "coverage_calibration", ext=ext), dpi=300 if ext == "png" else None, bbox_inches="tight")
print("\n저장:", figpath("eval", "coverage_calibration"))

# Fig2: transfer AOA — 대표 held-out region 지도 + inside/outside 성능
if len(aoa_df):
    rbest = aoa_df.sort_values("n", ascending=False).iloc[0]["region"]
    sel = (reg == rbest) & np.isfinite(aoa_all)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    ins = sel & (aoa_all == 1); out = sel & (aoa_all == 0)
    ax.scatter(base.lon[ins], base.lat[ins], s=4, c=CMAP.alt(0.6), label="AOA 안(신뢰)", rasterized=True)
    ax.scatter(base.lon[out], base.lat[out], s=6, c="#B8BEC6", marker="x", label="AOA 밖(외삽·회색)", rasterized=True)
    ax.set_xlabel("경도 (°E)", fontsize=11); ax.set_ylabel("위도 (°N)", fontsize=11)
    ax.set_title(f"transfer AOA — held-out region '{rbest}'\n학습 환경 밖 점을 회색 표기(정직한 외삽)", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax = axes[1]
    rr = aoa_df[aoa_df.region == rbest].iloc[0]
    x = np.arange(2)
    ax.bar(x - 0.2, [rr.rmse_in, rr.rmse_out], width=0.4, color=CMAP.err(0.5), label="RMSE (cm)")
    ax.set_ylabel("RMSE (cm)", fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, [rr.cov_in, rr.cov_out], "o-", color="#0b7285", lw=1.8, ms=7, label="coverage (%)")
    ax2.axhline(90, color="#b5651d", ls="--", lw=1.2)
    ax2.set_ylabel("coverage (%)", color="#0b7285", fontsize=11); ax2.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(["AOA 안", "AOA 밖"], fontsize=10)
    ax.set_title("AOA 밖 = RMSE↑ · coverage↓\n(외삽 영역을 정직하게 경고)", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(mappath("alt_aoa_mask"), dpi=300, bbox_inches="tight")
    fig.savefig(mappath("alt_aoa_mask", ext="pdf"), bbox_inches="tight")
    print("저장:", mappath("alt_aoa_mask"))

# Fig3: 불확실성 폭 지도 (within-domain CQR width, 점 산포)
w = (hi_c - lo_c)
fig, ax = plt.subplots(figsize=(7.6, 6.0))
finite = np.isfinite(w) & np.isfinite(oof_pt)
sc = ax.scatter(base.lon[finite], base.lat[finite], c=w[finite], s=8, alpha=0.75, cmap=CMAP.err,
                vmin=np.nanpercentile(w, 5), vmax=np.nanpercentile(w, 95), rasterized=True)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("90% 예측구간 폭 (cm)", fontsize=10, rotation=270, labelpad=16)
ax.set_xlabel("경도", fontsize=11); ax.set_ylabel("위도", fontsize=11)
ax.xaxis.set_major_formatter(lon_formatter()); ax.yaxis.set_major_formatter(lat_formatter())
ax.set_aspect(1.0 / np.cos(np.deg2rad(65))); despine(ax)
ax.set_title(f"보정된 셀별 불확실성 폭 (CQR, 90%, n={int(finite.sum())})\n넓을수록 예측 신뢰도 낮음", fontsize=12.5)
fig.tight_layout()
fig.savefig(mappath("alt_uncertainty_width"), dpi=300, bbox_inches="tight")
fig.savefig(mappath("alt_uncertainty_width", ext="pdf"), bbox_inches="tight")
print("저장:", mappath("alt_uncertainty_width"))
