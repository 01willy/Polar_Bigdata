"""S1 시각화: 실제 지도 배경 위 관측·예측·잔차 + Taylor diagram + 모델 비교.

`docs/RESEARCH_PLAN_...` §11.5. 냉색 규약, cartopy 지도 배경(위경도 지역 식별),
표<그래프<지도 우선. 여러 모델을 나란히(한 모델 단정 금지 시각화).

실행: python scripts/4_visualization/s1_baseline_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar, CMAP, tnorm
from polar.geomap import (make_ax, scatter_map, hexbin_map, add_colorbar, add_scalebar,
                          add_inset_locator, ALASKA)

use_polar()
PROC = C.PROCESSED
OUT = C.FIGURES / "s1_baseline"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


oof = pd.read_csv(PROC / "s1_baseline_oof.csv")
res = pd.read_csv(PROC / "s1_baseline_results.csv")
pred_cols = [c for c in oof.columns if c.startswith("pred_")]
models = [c.replace("pred_", "") for c in pred_cols]
# in-domain RMSE 순위(seed 평균). 발산 모델(RMSE>30)은 시각화에서 제외(별도 안정화 대상).
rank = (res[res.cv == "spatial_block_AK"].groupby("model").rmse_cm.mean().sort_values())
diverged = rank[rank > 30].index.tolist()
ordered = [m for m in rank.index if rank[m] <= 30 and f"pred_{m}" in oof.columns]
if diverged:
    print(f"[warn] 발산 모델 시각화 제외: {diverged} (별도 안정화 필요)")
print(f"[load] {len(oof):,} OOF · 모델 {ordered}")

vmin, vmax = 20, 110


# ---------------- 1. 관측 + 상위 3모델 예측 지도 (hexbin 셀통계 + 위치 inset) ----------------
from polar.geomap import _proj
proj = _proj(ALASKA)
top = ordered[:3]
panels = [("(a) 관측 ALT", oof.alt_cm)] + [
    (f"({chr(98+i)}) {m} 예측 (RMSE {rank[m]:.1f})", oof[f"pred_{m}"]) for i, m in enumerate(top)]
fig = plt.figure(figsize=(13, 4.6))
for i, (title, vals) in enumerate(panels):
    ax = fig.add_subplot(1, 4, i + 1, projection=proj)
    make_ax(ALASKA, ax=ax, fig=fig, title=title)
    hb = hexbin_map(ax, oof.lon, oof.lat, vals, gridsize=40, cmap=CMAP.alt, vmin=vmin, vmax=vmax)
    if i == 0:
        add_inset_locator(fig, ax, ALASKA); add_scalebar(ax)
cb = fig.colorbar(hb, ax=fig.axes, fraction=0.012, pad=0.02)
cb.set_label("ALT (cm)", fontsize=9)
fig.suptitle("S1 알래스카 in-domain: 관측 vs 상위 모델 예측 (hexbin 셀 중앙값, 공간블록 OOF)", fontsize=12, y=1.02)
save(fig, "alaska_obs_vs_pred_maps")


# ---------------- 2. 모델별 잔차 지도 (hexbin, broc 0중심) ----------------
n = len(ordered)
ncol = min(4, n); nrow = int(np.ceil(n / ncol))
fig = plt.figure(figsize=(3.4 * ncol, 3.4 * nrow))
for i, m in enumerate(ordered):
    ax = fig.add_subplot(nrow, ncol, i + 1, projection=proj)
    make_ax(ALASKA, ax=ax, fig=fig, title=f"{m}  (bias {(oof[f'pred_{m}']-oof.alt_cm).mean():+.1f})")
    resid = oof[f"pred_{m}"] - oof.alt_cm
    hb = hexbin_map(ax, oof.lon, oof.lat, resid, gridsize=38, cmap=CMAP.diff, norm=tnorm(-40, 40, 0))
cb = fig.colorbar(hb, ax=fig.axes, fraction=0.012, pad=0.02)
cb.set_label("예측 - 관측 (cm)", fontsize=9)
fig.suptitle("S1 모델별 잔차 지도 (hexbin 중앙값; 파랑=과소, 갈색=과대)", fontsize=12, y=1.0)
save(fig, "alaska_residual_maps")


# ---------------- 3. Taylor diagram (in-domain) ----------------
obs = oof.alt_cm.values
sd_obs = obs.std()
fig = plt.figure(figsize=(6.4, 6))
ax = fig.add_subplot(111, polar=True)
ax.set_thetamin(0); ax.set_thetamax(90)
ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
cmap = plt.get_cmap("cmc.batlow", len(ordered))
for i, m in enumerate(ordered):
    p = oof[f"pred_{m}"].values
    r = np.corrcoef(obs, p)[0, 1]
    sd = p.std()
    theta = np.arccos(np.clip(r, -1, 1))
    ax.plot(theta, sd, "o", ms=10, color=cmap(i), label=f"{m} (r={r:.2f})")
ax.plot(0, sd_obs, "k*", ms=16, label="관측")
# RMSE 반원 등고선(관측 기준)
for rms in [8, 12, 16, 20]:
    th = np.linspace(0, np.pi / 2, 100)
    xr = sd_obs + rms * np.cos(th + np.pi)  # 근사
ax.set_rlabel_position(90)
ax.set_xlabel("표준편차 (cm)")
ax.set_title("Taylor diagram — 알래스카 in-domain\n(관측★에 가까울수록 우수)", fontsize=11, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=8)
save(fig, "taylor_indomain")


# ---------------- 4. 모델 비교 막대 (in-domain RMSE + LORO) ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
# in-domain
ind = res[res.cv == "spatial_block_AK"].groupby("model").rmse_cm.agg(["mean", "std"]).reindex(ordered)
axes[0].bar(range(len(ind)), ind["mean"], yerr=ind["std"].fillna(0), capsize=3,
            color=[cmap(i) for i in range(len(ind))])
axes[0].set_xticks(range(len(ind))); axes[0].set_xticklabels(ind.index, rotation=45, ha="right", fontsize=8)
axes[0].axhline(14, color="0.4", ls="--", lw=1, label="대표성 하한 ~14cm")
axes[0].set_ylabel("RMSE (cm)"); axes[0].set_title("알래스카 in-domain (공간블록)", fontsize=10)
axes[0].legend(fontsize=8)
# LORO (region별)
lo = res[res.cv == "LORO"].groupby(["model", "region"]).rmse_cm.mean().unstack()
lo = lo.reindex(ordered)
lo.plot(kind="bar", ax=axes[1], colormap="cmc.batlow", width=0.8)
axes[1].set_ylabel("RMSE (cm)"); axes[1].set_title("LORO 전이 (공유 코어, SAR 제외)", fontsize=10)
axes[1].set_xticklabels(lo.index, rotation=45, ha="right", fontsize=8)
axes[1].legend(title="test 지역", fontsize=8)
fig.suptitle("S1 모델 비교 — in-domain은 동률권(하한 근접), 전이는 지역별 편차 큼", fontsize=12)
save(fig, "model_comparison_bars")

print("[done] S1 시각화 4종 완료")
