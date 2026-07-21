"""트랙 1 시각화 4종 — 알래스카 내부 라벨 증강 × 다중 DL.

1. 모델×증강조합 held-out RMSE·skill 히트맵(real_only 기준 Δ 병기).
2. 물리유도 비율-성능 곡선(모델별).
3. 최적 모델 예측지도 vs 실측(알래스카).
4. pseudo 라벨(물리·시추공) vs 실측 ALT 분포(정합성).

시각 규약: polar.plotstyle 냉색 순차, 붉은/rainbow 금지, 축 단위, PNG+PDF.
실행: /home/anaconda3/bin/python scripts/3_deep_learning/aug_within_alaska_figs.py
"""
import os, sys, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from polar.plotstyle import (use_polar, CMAP, add_cbar, style_geo, despine,
                             tnorm)
from polar.outputs import figpath, mappath

plt = use_polar()
PROC = "data/processed"
CAT = "12_aug_alaska"  # outputs/figures/12_aug_alaska/

sweep = pd.read_csv(os.path.join(PROC, "aug_within_alaska_sweep.csv"))
results = pd.read_csv(os.path.join(PROC, "aug_within_alaska_results.csv"))
oof = pd.read_csv(os.path.join(PROC, "aug_within_alaska_oof.csv"))
with open(os.path.join(PROC, "aug_within_alaska_meta.json")) as f:
    meta = json.load(f)

MODELS = ["GBM", "MLP", "FTTransformer", "TabM"]
SOTA = meta["sota_band_cm"]


def save(fig, name):
    fig.savefig(figpath(CAT, name, "png"), dpi=200, bbox_inches="tight")
    fig.savefig(figpath(CAT, name, "pdf"), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 그림 1: 모델×증강조합 RMSE 히트맵(real_only 대비 Δ)
# ============================================================
def fig1_heatmap():
    # 공통 조합만(모든 모델에 존재하는 config): real_only + DL 축소셋
    common = ["real_only", "real+phys_r0.5_w1.0_const", "real+phys_r1.0_w1.0_const",
              "real+phys_r2.0_w1.0_const", "real+phys_r1.0_w0.3_const",
              "real+borehole", "real+phys+borehole"]
    common = [c for c in common if (sweep.config == c).any()]
    M = np.full((len(MODELS), len(common)), np.nan)
    D = np.full_like(M, np.nan)
    for i, mo in enumerate(MODELS):
        base = sweep[(sweep.model == mo) & (sweep.config == "real_only")]
        base_rmse = float(base.rmse_cm.iloc[0]) if len(base) else np.nan
        for j, cf in enumerate(common):
            r = sweep[(sweep.model == mo) & (sweep.config == cf)]
            if len(r):
                M[i, j] = float(r.rmse_cm.iloc[0])
                D[i, j] = M[i, j] - base_rmse

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.2))
    lab = [c.replace("real+", "").replace("_const", "").replace("real_only", "real")
           for c in common]

    # 좌: 절대 RMSE(냉색 순차, 낮을수록 진함)
    im = axL.imshow(M, cmap=CMAP.err, aspect="auto",
                    vmin=np.nanmin(M), vmax=np.nanmax(M))
    for i in range(len(MODELS)):
        for j in range(len(common)):
            if np.isfinite(M[i, j]):
                axL.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                         fontsize=8, color="white" if M[i, j] > np.nanmean(M) else "black")
    axL.set_xticks(range(len(common))); axL.set_xticklabels(lab, rotation=35, ha="right", fontsize=8)
    axL.set_yticks(range(len(MODELS))); axL.set_yticklabels(MODELS)
    axL.set_title(f"held-out 실측 ALT RMSE (cm)\nSOTA 정직대역 {SOTA[0]}-{SOTA[1]}cm")
    add_cbar(fig, im, axL, "RMSE (cm)")

    # 우: real_only 대비 Δ(0중심 발산, 음수=증강개선)
    vmax = np.nanmax(np.abs(D[:, 1:])) if D.shape[1] > 1 else 1.0
    im2 = axR.imshow(D, cmap=CMAP.diff, norm=tnorm(-vmax, vmax, 0.0), aspect="auto")
    for i in range(len(MODELS)):
        for j in range(len(common)):
            if np.isfinite(D[i, j]) and common[j] != "real_only":
                axR.text(j, i, f"{D[i, j]:+.1f}", ha="center", va="center",
                         fontsize=8, color="black")
    axR.set_xticks(range(len(common))); axR.set_xticklabels(lab, rotation=35, ha="right", fontsize=8)
    axR.set_yticks(range(len(MODELS))); axR.set_yticklabels(MODELS)
    axR.set_title("증강 효과 ΔRMSE (cm)\n음수(파랑)=real_only 대비 개선")
    add_cbar(fig, im2, axR, "ΔRMSE (cm)")
    fig.suptitle("모델 × 증강조합: 알래스카 내부 held-out ALT 정확도", y=1.02, fontsize=13)
    save(fig, "fig1_model_aug_heatmap")


# ============================================================
# 그림 2: 물리유도 비율-성능 곡선(모델별)
# ============================================================
def fig2_ratio_curves():
    ratios = [0.0, 0.5, 1.0, 2.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, mo in enumerate(MODELS):
        rr, ss = [], []
        for pr in ratios:
            if pr == 0.0:
                r = sweep[(sweep.model == mo) & (sweep.config == "real_only")]
            else:
                r = sweep[(sweep.model == mo) &
                          (sweep.config == f"real+phys_r{pr}_w1.0_const")]
            if len(r):
                rr.append(float(r.rmse_cm.iloc[0])); ss.append(float(r.skill_over_mean.iloc[0]))
            else:
                rr.append(np.nan); ss.append(np.nan)
        ax1.plot(ratios, rr, "-o", color=colors[k], label=mo, lw=1.8, ms=6)
        ax2.plot(ratios, ss, "-o", color=colors[k], label=mo, lw=1.8, ms=6)
    ax1.axhspan(SOTA[0], SOTA[1], color="0.85", zorder=0, label="SOTA 대역")
    ax1.set_xlabel("물리유도 pseudo 비율 (× 실측 수)")
    ax1.set_ylabel("held-out RMSE (cm)")
    ax1.set_title("물리 증강 비율 대비 RMSE")
    ax1.legend(fontsize=8, loc="best"); despine(ax1)
    ax2.set_xlabel("물리유도 pseudo 비율 (× 실측 수)")
    ax2.set_ylabel("skill over mean")
    ax2.set_title("물리 증강 비율 대비 skill")
    ax2.axhline(0, color="0.6", lw=0.8, ls="--")
    ax2.legend(fontsize=8, loc="best"); despine(ax2)
    fig.suptitle("물리유도 라벨 증강: 비율-성능 곡선(가중 1.0, Stefan 상수E)", y=1.02, fontsize=13)
    save(fig, "fig2_phys_ratio_curves")


# ============================================================
# 그림 3: 최적 모델 예측지도 vs 실측(알래스카)
# ============================================================
def fig3_pred_maps():
    lat, lon = oof.lat.values, oof.lon.values
    y = oof.alt_cm.values
    yb = oof.yhat_best.values
    vmax = np.nanpercentile(np.r_[y, yb], 98)
    vmin = 0
    best = meta["best_overall"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    sc0 = axes[0].scatter(lon, lat, c=y, cmap=CMAP.alt, s=4, vmin=vmin, vmax=vmax)
    style_geo(axes[0], title="실측 ALT")
    add_cbar(fig, sc0, axes[0], "ALT (cm)")

    sc1 = axes[1].scatter(lon, lat, c=yb, cmap=CMAP.alt, s=4, vmin=vmin, vmax=vmax)
    style_geo(axes[1], title=f"예측 ALT (최적 {best['model']}/{best['config']})")
    add_cbar(fig, sc1, axes[1], "ALT (cm)")

    resid = yb - y
    rmax = np.nanpercentile(np.abs(resid), 98)
    sc2 = axes[2].scatter(lon, lat, c=resid, cmap=CMAP.diff,
                          norm=tnorm(-rmax, rmax, 0.0), s=4)
    style_geo(axes[2], title="예측 − 실측 (cm)")
    add_cbar(fig, sc2, axes[2], "잔차 (cm)")
    fig.suptitle(f"알래스카 내부 held-out ALT: 최적 모델 예측 vs 실측 "
                 f"(RMSE={best['rmse_cm']:.1f}cm)", y=1.03, fontsize=13)
    save(fig, "fig3_best_pred_map")
    # 지도 산출물도 maps/에 병행 저장
    fig2, ax = plt.subplots(figsize=(6, 5))
    scm = ax.scatter(lon, lat, c=yb, cmap=CMAP.alt, s=5, vmin=vmin, vmax=vmax)
    style_geo(ax, title=f"알래스카 예측 ALT ({best['model']})")
    add_cbar(fig2, scm, ax, "ALT (cm)")
    fig2.savefig(mappath("aug_alaska_best_pred", "png"), dpi=200, bbox_inches="tight")
    fig2.savefig(mappath("aug_alaska_best_pred", "pdf"), bbox_inches="tight")
    plt.close(fig2)


# ============================================================
# 그림 4: pseudo 라벨 vs 실측 ALT 분포(정합성)
# ============================================================
def fig4_label_dist():
    real = oof.alt_cm.values
    phys = pd.read_parquet("/tmp/aug_phys_rep.parquet")["alt_cm"].values \
        if os.path.exists("/tmp/aug_phys_rep.parquet") else None
    bh = pd.read_parquet("/tmp/aug_bh_rep.parquet")["alt_cm"].values \
        if os.path.exists("/tmp/aug_bh_rep.parquet") else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 250, 40)
    ax1.hist(real, bins=bins, density=True, alpha=0.6, label=f"실측(n={len(real)})",
             color=CMAP.count(0.6))
    if phys is not None:
        ax1.hist(phys, bins=bins, density=True, histtype="step", lw=2,
                 label=f"물리유도(n={len(phys)})", color="#1f6f8b")
    if bh is not None and len(bh):
        ax1.hist(bh, bins=bins, density=True, histtype="step", lw=2,
                 label=f"시추공유도(n={len(bh)})", color="#5b8c5a")
    ax1.set_xlabel("ALT (cm)"); ax1.set_ylabel("밀도")
    ax1.set_title("라벨 소스별 ALT 분포")
    ax1.legend(fontsize=8); despine(ax1)

    # 요약 통계 표
    ax2.axis("off")
    rows = [["소스", "n", "평균", "표준편차", "범위(cm)"]]
    rs = meta["real_alt_stats"]
    rows.append(["실측", f"{meta['n_anchor']}", f"{rs['mean']:.1f}",
                 f"{rs['std']:.1f}", f"{rs['min']:.0f}-{rs['max']:.0f}"])
    ps_ = meta["phys_alt_stats"]
    rows.append(["물리유도", f"{meta['n_phys_pool']}", f"{ps_['mean']:.1f}",
                 f"{ps_['std']:.1f}", f"{ps_['min']:.0f}-{ps_['max']:.0f}"])
    bs = meta.get("borehole_alt_stats", {})
    if bs:
        rows.append(["시추공유도", f"{bs['n']}", f"{bs['mean']:.1f}",
                     f"{bs['std']:.1f}", "0-400(클립)"])
    tbl = ax2.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center",
                    loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    ax2.set_title("라벨 소스 정합성 요약\n(물리유도=Stefan 상수E 전체적합, 시추공=t_max 0°C 깊이)",
                  fontsize=10)
    fig.suptitle("pseudo 라벨 vs 실측 ALT 정합성", y=1.02, fontsize=13)
    save(fig, "fig4_label_consistency")


if __name__ == "__main__":
    fig1_heatmap()
    fig2_ratio_curves()
    fig3_pred_maps()
    fig4_label_dist()
    print("[figs] 4종 저장 완료 → outputs/figures/12_aug_alaska/ + outputs/maps/")
