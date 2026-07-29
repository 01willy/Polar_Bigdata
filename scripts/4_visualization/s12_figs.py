"""S12 보고서용 그림: 방법 계열 진행과 편향 상쇄 메커니즘.

순위 나열 대신 (a) 대표 방법의 지역별 오차, (b) 편향 상쇄 메커니즘,
(c) 잔차 가중에 대한 반응을 보인다.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
try:
    from polar.plotstyle import use_polar
    use_polar()
except Exception:
    pass

OUT = ROOT / "data" / "processed"
FIGD = ROOT / "outputs" / "figures" / "s12_hybrid"
FIGD.mkdir(parents=True, exist_ok=True)

df = pd.concat([pd.read_csv(p) for p in glob.glob(str(OUT / "s12_hybrid_transfer_shard*.csv"))],
               ignore_index=True)
df = df[np.isfinite(df.rmse_cm)]

def pick(proto, **kw):
    d = df[df.proto == proto]
    for k, v in kw.items():
        d = d[d[k] == v]
    return d.groupby("target").agg(rmse=("rmse_cm", "mean"), bias=("bias_cm", "mean"))

# 대표 방법 5종 (계열 진행)
def series(proto):
    if proto == "half":
        best = dict(family="anchored", anchor="stefan", resid="mlp", lam=0.25, pseudo="cci", r=1.0)
    else:
        best = dict(family="anchored", anchor="stefan_cci", resid="catboost", lam=0.25, pseudo="none", r=0.0)
    return [
        ("기계학습 단독",        pick(proto, family="direct", resid="catboost", pseudo="none")),
        ("위성제품 단독",        pick(proto, family="analytic", anchor="cci")),
        ("물리식(Stefan) 단독",  pick(proto, family="analytic", anchor="stefan")),
        ("물리식+위성제품 결합",  pick(proto, family="analytic", anchor="stefan_cci")),
        ("결합 + 잔차 학습",     pick(proto, **best)),
    ]

C_LENA, C_CAN = "#4a7c8c", "#2f4b6e"
fig = plt.figure(figsize=(11.6, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], hspace=0.52, wspace=0.24)

# ---- (a),(b) 지역별 RMSE
for j, (proto, pn) in enumerate([("half", "대상 지역 공변량 사용 가능"),
                                 ("loro", "대상 지역 정보 없음")]):
    ax = fig.add_subplot(gs[0, j])
    S = series(proto)
    x = np.arange(len(S)); w = 0.36
    le = [s[1].loc["Lena", "rmse"] if "Lena" in s[1].index else np.nan for s in S]
    ca = [s[1].loc["Canada", "rmse"] if "Canada" in s[1].index else np.nan for s in S]
    ax.bar(x - w / 2, le, w, color=C_LENA, label="레나델타", zorder=3)
    ax.bar(x + w / 2, ca, w, color=C_CAN, label="캐나다", zorder=3)
    for xi, (a, b) in enumerate(zip(le, ca)):
        ax.text(xi - w / 2, a + 0.7, f"{a:.1f}", ha="center", fontsize=8.2, color="#444")
        ax.text(xi + w / 2, b + 0.7, f"{b:.1f}", ha="center", fontsize=8.2, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in S], fontsize=8.6, rotation=18, ha="right")
    ax.set_ylabel("RMSE (cm)", fontsize=10)
    ax.set_title(f"({'ab'[j]}) {pn}", fontsize=10.5, loc="left", pad=7)
    ax.set_ylim(0, max(max(le), max(ca)) * 1.24)
    ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if j == 0:
        ax.legend(fontsize=9, frameon=False, loc="upper right")

# ---- (c) 편향 상쇄
ax = fig.add_subplot(gs[1, 0])
labels, stef_b, cci_b, comb_b = [], [], [], []
for proto, pn in [("half", "공변량 사용"), ("loro", "정보 없음")]:
    for tg in ["Lena", "Canada"]:
        labels.append(f"{'레나델타' if tg=='Lena' else '캐나다'}\n({pn})")
        stef_b.append(pick(proto, family="analytic", anchor="stefan").loc[tg, "bias"])
        cci_b.append(pick(proto, family="analytic", anchor="cci").loc[tg, "bias"])
        comb_b.append(pick(proto, family="analytic", anchor="stefan_cci").loc[tg, "bias"])
x = np.arange(len(labels)); w = 0.26
ax.axhline(0, color="#3a4a5a", lw=0.9, zorder=2)
ax.bar(x - w, stef_b, w, color="#8296a8", label="물리식", zorder=3)
ax.bar(x, cci_b, w, color="#b8c4d0", label="위성제품", zorder=3)
ax.bar(x + w, comb_b, w, color="#2f4b6e", label="결합", zorder=3)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
ax.set_ylabel("평균 편향 (cm)", fontsize=10)
ax.set_title("(c) 편향의 상쇄", fontsize=10.5, loc="left", pad=7)
ax.legend(fontsize=8.8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_ylim(min(cci_b) * 1.5, max(max(stef_b), max(cci_b)) * 1.85)

# ---- (d) 잔차 가중 반응
# 각 λ에서 두 지역이 모두 산출된 구성 중 최소값(포락선)을 그린다. 한 계열만 그리면
# 표의 "결합 + 잔차 학습" 행과 어긋나므로, 표와 같은 선택 규칙을 쓴다.
ax = fig.add_subplot(gs[1, 1])
for proto, cc, mk, pn in [("half", C_LENA, "o", "공변량 사용"), ("loro", C_CAN, "s", "정보 없음")]:
    d = df[(df.proto == proto) & (df.family == "anchored")]
    per = d.groupby(["anchor", "resid", "lam", "pseudo", "r", "target"]).rmse_cm.mean().reset_index()
    agg = (per.groupby(["anchor", "resid", "lam", "pseudo", "r"])
              .agg(nreg=("target", "nunique"), rmse=("rmse_cm", "mean")).reset_index())
    m = agg[agg.nreg == 2].groupby("lam").rmse.min()
    ax.plot(m.index, m.values, marker=mk, ms=5, lw=1.6, color=cc, label=pn, zorder=3)
    lo = m.idxmin()
    ax.plot([lo], [m.loc[lo]], marker=mk, ms=9, mfc="none", mec=cc, mew=1.6, ls="none", zorder=4)
ax.set_xlabel("잔차 가중 $\\lambda$", fontsize=10)
ax.set_ylabel("두 지역 평균 RMSE (cm)", fontsize=10)
ax.set_title("(d) 잔차 반영 정도에 따른 반응", fontsize=10.5, loc="left", pad=7)
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25, lw=0.5); ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

for ext in ("png", "pdf"):
    fig.savefig(FIGD / f"s12_hybrid_summary.{ext}", dpi=300, bbox_inches="tight")
print(f"saved: {FIGD}/s12_hybrid_summary.png")
