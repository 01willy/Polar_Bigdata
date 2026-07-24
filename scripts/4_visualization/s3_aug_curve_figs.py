"""S3 시각화: 증강비율 반응곡선 (물리 vs placebo) + 조건별 Δskill.

`RESEARCH_PLAN_...` §11.5. 핵심 그림: Δskill=f(r) 곡선에 물리(stefan·ku)와 placebo 병기.
물리>placebo 간격이 '물리 정보의 순가치'. 냉색 규약.
실행: python scripts/4_visualization/s3_aug_curve_figs.py
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
OUT = C.FIGURES / "s3_aug"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


res = pd.read_csv(PROC / "s3_aug_curve_results.csv")
seed_rows = res[res.seed != "SUMMARY"].copy()
seed_rows["r"] = pd.to_numeric(seed_rows["r"])
# 발산 모델 제외(전이 covariate shift 하 신경망 발산; base RMSE>100 = 비물리)
div = seed_rows[seed_rows.rmse_cm > 100][["target", "model"]].drop_duplicates()
for _, row in div.iterrows():
    seed_rows = seed_rows[~((seed_rows.target == row.target) & (seed_rows.model == row.model))]
if len(div):
    print(f"[warn] 발산 제외(전이 발산): {div.to_dict('records')}")
targets = sorted(seed_rows.target.unique())
models = sorted(seed_rows.model.unique())
PHYS_LABEL = {"stefan": "Stefan pseudo", "kudryavtsev": "Kudryavtsev pseudo", "placebo": "placebo(상수)"}
PHYS_COLOR = {"stefan": "#1f6f8b", "kudryavtsev": "#5a9e6f", "placebo": "#b0651a"}

# ---------------- 반응곡선: target×model 그리드, r vs ΔRMSE(개선) ----------------
fig, axes = plt.subplots(len(models), len(targets), figsize=(5.2 * len(targets), 4.2 * len(models)),
                         squeeze=False)
for mi, model in enumerate(models):
    for ti, target in enumerate(targets):
        ax = axes[mi][ti]
        sub = seed_rows[(seed_rows.model == model) & (seed_rows.target == target)]
        base = sub[sub.r == 0].groupby("seed").rmse_cm.mean().mean()  # r=0 기준 RMSE
        for phys in ["stefan", "kudryavtsev", "placebo"]:
            s = sub[sub.phys == phys]
            if not len(s):
                continue
            g = s.groupby("r").rmse_cm.agg(["mean", "std"])
            delta = base - g["mean"]           # 양수=개선
            ax.plot(g.index, delta, "o-", color=PHYS_COLOR[phys], label=PHYS_LABEL[phys], lw=1.8, ms=5)
            ax.fill_between(g.index, delta - g["std"], delta + g["std"], color=PHYS_COLOR[phys], alpha=0.15)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xlabel("증강비율 r (pseudo/실측)", fontsize=9)
        ax.set_ylabel("ΔRMSE 개선 (cm, 양수=개선)", fontsize=9)
        ax.set_title(f"{model} → {target} 전이", fontsize=10)
        if mi == 0 and ti == 0:
            ax.legend(fontsize=8, loc="best")
fig.suptitle("S3 물리 pseudo-label 증강 반응곡선 — 물리(Stefan/Ku) vs placebo(상수)\n"
             "물리선이 placebo 위에 있어야 '물리 정보의 순가치'", fontsize=12, y=1.01)
save(fig, "aug_response_curves")

# ---------------- 물리 순가치: 물리 − placebo (같은 r에서) ----------------
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for model in models:
    for target in targets:
        sub = seed_rows[(seed_rows.model == model) & (seed_rows.target == target)]
        base = sub[sub.r == 0].rmse_cm.mean()
        st = base - sub[sub.phys == "stefan"].groupby("r").rmse_cm.mean()
        pl = base - sub[sub.phys == "placebo"].groupby("r").rmse_cm.mean()
        net = (st - pl).reindex(sorted(sub.r.unique()))
        ax.plot(net.index, net.values, "o-", lw=1.6, ms=5, label=f"{model}→{target}")
ax.axhline(0, color="0.4", lw=0.8, ls="--")
ax.set_xscale("symlog", linthresh=0.25)
ax.set_xlabel("증강비율 r", fontsize=9)
ax.set_ylabel("Stefan − placebo ΔRMSE (cm)", fontsize=9)
ax.set_title("물리 정보의 순가치 (Stefan pseudo − placebo 상수)\n양수=물리 구조가 단순 앵커링보다 유효", fontsize=11)
ax.legend(fontsize=8)
save(fig, "physics_net_value")

print("[done] S3 시각화 완료")
