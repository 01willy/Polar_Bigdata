"""S5 시각화: dense Stefan pseudo 사전학습 효과 (scratch vs pretrain).

핵심 그림: 지역별 LORO + in-domain에서 scratch→pretrain ΔRMSE(덤벨). transductive 여부 표기.
실행: python scripts/4_visualization/s5_pretrain_figs.py
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
OUT = C.FIGURES / "s5_pretrain"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


res = pd.read_csv(PROC / "s5_pretrain_results.csv")
loro = res[res.cv == "LORO"].copy()
ind = res[res.cv == "spatial_block_AK"].copy()
models = sorted(loro.model.unique())
V_COLOR = {"scratch": "#8a8fa3", "pretrain": "#1f6f8b"}

# ---------------- 덤벨: 지역×모델, scratch→pretrain ----------------
cases = []  # (라벨, model, sub_df)
for reg in ["Alaska", "Lena", "Canada"]:
    sub = loro[loro.region == reg]
    td = bool(sub.transductive.iloc[0]) if len(sub) else False
    for model in models:
        s = sub[sub.model == model]
        if len(s):
            cases.append((f"{reg}{'†' if td else ''} · {model}", model, s))
for model in models:
    s = ind[ind.model == model]
    if len(s):
        cases.append((f"in-domain AK · {model}", model, s))

fig, ax = plt.subplots(figsize=(7.8, 0.62 * len(cases) + 1.6))
for i, (label, model, s) in enumerate(cases):
    g = s.groupby("variant").rmse_cm.agg(["mean", "std"])
    if not {"scratch", "pretrain"} <= set(g.index):
        continue
    sc, pt = g.loc["scratch", "mean"], g.loc["pretrain", "mean"]
    ax.plot([sc, pt], [i, i], "-", color="0.6", lw=1.2, zorder=1)
    ax.errorbar(sc, i, xerr=g.loc["scratch", "std"], fmt="o", color=V_COLOR["scratch"],
                ms=6, capsize=2, zorder=2)
    ax.errorbar(pt, i, xerr=g.loc["pretrain", "std"], fmt="o", color=V_COLOR["pretrain"],
                ms=6, capsize=2, zorder=2)
    ax.annotate(f"{sc - pt:+.2f}", ((sc + pt) / 2, i + 0.22), fontsize=8,
                ha="center", color="0.25")
ax.set_yticks(range(len(cases)))
ax.set_yticklabels([c[0] for c in cases], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("RMSE (cm, 3-seed 평균 ± SD)", fontsize=9)
ax.plot([], [], "o", color=V_COLOR["scratch"], label="scratch(실측만)")
ax.plot([], [], "o", color=V_COLOR["pretrain"], label="Stefan pseudo 사전학습→finetune")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("S5 물리 사전학습 효과 (숫자=ΔRMSE, 양수=개선. †=격자가 test 지역 공변량 포함(transductive))",
             fontsize=10.5)
save(fig, "s5_pretrain_dumbbell")

# ---------------- 게이트 요약 ----------------
gate = res[res.cv == "LORO_gate"]
if len(gate):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    w, xs = 0.35, np.arange(len(models))
    for k, variant in enumerate(["scratch", "pretrain"]):
        vals = [gate[(gate.model == m) & (gate.variant == variant)].rmse_cm.mean() for m in models]
        ax.bar(xs + (k - 0.5) * w, vals, w, color=V_COLOR[variant], label=variant)
        for x, v in zip(xs + (k - 0.5) * w, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (x, v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(models)
    ax.set_ylabel("LORO 게이트 RMSE (cm, 비가중평균)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title("S5 게이트: LORO 비가중평균(Alaska·Lena·Canada)", fontsize=10.5)
    save(fig, "s5_gate_bars")

print("[done] S5 시각화 완료")
