"""S4 시각화: shrinkage λ 곡선 (Stefan 앵커 + 저용량 잔차) — LORO 게이트 대비.

핵심 그림: 지역별(Alaska·Lena·Canada) + 게이트(비가중평균) RMSE = f(λ).
λ=0 = Stefan-only 앵커. S2 게이트(22.24cm) 수평선. 자동선택 λ*(inner CV) 마커.
실행: python scripts/4_visualization/s4_residual_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar

use_polar()
PROC = C.PROCESSED
OUT = C.FIGURES / "s4_residual"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


import json
res = pd.read_csv(PROC / "s4_residual_results.csv")
meta = json.loads((PROC / "s4_residual_meta.json").read_text())
GATE = meta["s2_gate_cm"]

M_COLOR = {"ridge": "#8a8fa3", "catboost_lo": "#1f6f8b", "catboost": "#b0651a"}
M_LABEL = {"ridge": "ridge(선형)", "catboost_lo": "CatBoost 저용량(d3)", "catboost": "CatBoost 표준(d6)"}
FS_STYLE = {"shift14": "-", "shared25": "--"}
FS_LABEL = {"shift14": "지형+기후 14", "shared25": "+토양·CCI 25"}

# ---------------- λ 곡선: 지역 3 + 게이트 ----------------
loro = res[(res.cv == "LORO") & (res.seed != "SUMMARY")].copy()
gate = res[res.cv == "LORO_gate"].copy()
auto = res[res.cv == "LORO_autolam"].copy()
regions = ["Alaska", "Lena", "Canada"]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
for ax, reg in zip(axes[:3], regions):
    sub = loro[loro.region == reg]
    for model in M_COLOR:
        for fs in FS_STYLE:
            g = sub[(sub.model == model) & (sub.featset == fs)].groupby("lam").rmse_cm.mean()
            ax.plot(g.index, g.values, FS_STYLE[fs], color=M_COLOR[model], lw=1.8, ms=5,
                    marker="o", label=f"{M_LABEL[model]} · {FS_LABEL[fs]}")
    a = auto[auto.region == reg]
    for _, r in a.iterrows():
        ax.plot(r.lam, r.rmse_cm, "*", color=M_COLOR[r.model], ms=13, mec="k", mew=0.5, zorder=5)
    stefan_only = sub[sub.lam == 0].rmse_cm.mean()
    ax.axhline(stefan_only, color="0.35", lw=1.0, ls=":", label="Stefan-only(λ=0)")
    ax.set_xlabel("shrinkage λ", fontsize=9)
    ax.set_ylabel("LORO RMSE (cm)", fontsize=9)
    ax.set_title(f"{reg} 전이", fontsize=10)
ax = axes[3]
for model in M_COLOR:
    for fs in FS_STYLE:
        g = gate[(gate.model == model) & (gate.featset == fs)].sort_values("lam")
        ax.plot(g.lam, g.rmse_cm, FS_STYLE[fs], color=M_COLOR[model], lw=1.8, ms=5, marker="o")
ax.axhline(GATE, color="#7a1f2b", lw=1.4, ls="--", label=f"S2 게이트 {GATE:.2f}cm")
ag = res[res.cv == "LORO_autolam_gate"]
for _, r in ag.iterrows():
    ax.plot(r.lam, r.rmse_cm, "*", color=M_COLOR[r.model], ms=13, mec="k", mew=0.5, zorder=5)
ax.set_xlabel("shrinkage λ", fontsize=9)
ax.set_ylabel("비가중평균 RMSE (cm)", fontsize=9)
ax.set_title("게이트(Alaska·Lena·Canada 비가중평균)", fontsize=10)
ax.legend(fontsize=8, loc="best")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.02))
fig.suptitle("S4 잔차학습 shrinkage 곡선: 예측 = Stefan 앵커 + λ·잔차 g(x) (★=inner CV 자동 λ*)",
             fontsize=12, y=1.02)
save(fig, "s4_lambda_curves")

# ---------------- 부트스트랩 CI: 지역별 ΔRMSE(λ) ----------------
boot = res[res.cv == "LORO_boot"].copy()
if len(boot):
    fig, axes = plt.subplots(1, len(regions), figsize=(4.6 * len(regions), 3.8), sharey=False)
    for ax, reg in zip(axes, regions):
        b = boot[boot.region == reg].sort_values("lam")
        ax.errorbar(b.lam, b.delta_rmse,
                    yerr=[b.delta_rmse - b.ci_lo, b.ci_hi - b.delta_rmse],
                    fmt="o-", color="#1f6f8b", lw=1.6, ms=5, capsize=3)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_xlabel("shrinkage λ", fontsize=9)
        ax.set_ylabel("ΔRMSE vs Stefan-only (cm, 양수=개선)", fontsize=9)
        ax.set_title(f"{reg}", fontsize=10)
    fig.suptitle("S4 블록부트스트랩 95% CI (catboost_lo·지형+기후14, CI가 0 걸치면 개선 없음)",
                 fontsize=11, y=1.03)
    save(fig, "s4_bootstrap_ci")

# ---------------- in-domain vs 전이 대비 ----------------
ind = res[res.cv == "spatial_block_AK"].copy()
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for model in M_COLOR:
    g = ind[(ind.model == model) & (ind.featset == "shift14")].groupby("lam").rmse_cm.mean()
    ax.plot(g.index, g.values, "-o", color=M_COLOR[model], lw=1.8, ms=5,
            label=f"{M_LABEL[model]} (in-domain)")
    gg = gate[(gate.model == model) & (gate.featset == "shift14")].sort_values("lam")
    ax.plot(gg.lam, gg.rmse_cm, "--s", color=M_COLOR[model], lw=1.4, ms=4, alpha=0.7,
            label=f"{M_LABEL[model]} (LORO 게이트)")
ax.set_xlabel("shrinkage λ", fontsize=9)
ax.set_ylabel("RMSE (cm)", fontsize=9)
ax.set_title("잔차의 양면성: in-domain(AK 공간블록)은 개선, 전이(LORO 게이트)는 악화",
             fontsize=10.5)
ax.legend(fontsize=8)
save(fig, "s4_indomain_vs_transfer")

print("[done] S4 시각화 완료")
