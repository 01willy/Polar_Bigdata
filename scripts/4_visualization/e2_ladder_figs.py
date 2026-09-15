"""E2 물리식 사다리 그림 — 같은 보정 자유도 k에서 Stefan 대비 ΔRMSE (지역 내·전이) + 멱지수 b.

스펙: figures/figure_spec.json id=e2_physics_ladder, e2_exponent_b.
입력: data/processed/e2_physics_ladder_results{TAG}.csv, e2_physics_ladder_coefs{TAG}.csv
산출: outputs/figures/e2_physics_ladder/e2_ladder_delta.{png,pdf}, e2_exponent_b.{png,pdf}
실행: python3 scripts/4_visualization/e2_ladder_figs.py [--tag _v2]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C                     # noqa: E402
from polar.plotstyle import use_polar             # noqa: E402

plt = use_polar()
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
args = ap.parse_args()
TAG = args.tag
PROC = C.PROCESSED
OUT = C.FIGURES / "e2_physics_ladder"
OUT.mkdir(parents=True, exist_ok=True)

DEEP, TEAL, GREY, LIGHT = "#2f4b6e", "#4a7c8c", "#8296a8", "#b8c4d0"
res = pd.read_csv(PROC / f"e2_physics_ladder_results{TAG}.csv")
coefs = pd.read_csv(PROC / f"e2_physics_ladder_coefs{TAG}.csv")

R = r"\sqrt{\mathrm{TDD}}"
LABEL = {
    "F1k1": f"Stefan  $E{R}$", "F1k2": f"Stefan  $a+E{R}$", "F2k2": r"멱법칙  $a\,\mathrm{TDD}^{b}$",
    "F3k2": r"TDD 선형  $a+b\,\mathrm{TDD}$", "F4k2": r"MAAT 선형  $a+b\,\mathrm{MAAT}$",
    "F5k0": "edaphic(무보정)", "F5k1": r"edaphic  $c\,(\cdot)$", "F5k2": r"edaphic  $a+c\,(\cdot)$",
    "F6k0": "Kudryavtsev(무보정)", "F6k1": r"Kudryavtsev  $c\,(\cdot)$", "F6k2": r"Kudryavtsev  $a+c\,(\cdot)$",
    "F7k1": r"토양 도일 Stefan  $E\sqrt{\mathrm{TDD}_{stl1}}$", "F7k2": r"토양 도일 Stefan  $a+E\sqrt{\mathrm{TDD}_{stl1}}$",
    "F8k2": "2층 Stefan(유기층)", "F9k2": f"Stefan+FDD  $E{R}+\\gamma\\sqrt{{\\mathrm{{FDD}}}}$",
    "F10k2": f"Stefan×눈  $E{R}\\,(1+\\gamma\\,\\mathrm{{SWE}})$",
}
ORDER = ["F1k1", "F7k1", "F5k1", "F6k1",                                  # k=1
         "F1k2", "F7k2", "F2k2", "F3k2", "F4k2", "F5k2", "F6k2", "F8k2", "F9k2", "F10k2"]  # k=2
K0 = ["F5k0", "F6k0"]


def paperize(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.7)
        ax.spines[s].set_color("#7a7a7a")
    ax.grid(False)
    ax.grid(True, axis=grid_axis, color="#aab3bd", lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=2.5, width=0.7, color="#8a8a8a", labelcolor="#333333")


ind = res[res.axis == "indomain_AK"].set_index("formula")
loro = res[res.axis == "LORO"]
regions = [r for r in ["Alaska", "Lena", "Canada"] if r in set(loro.region)]
others = sorted(set(loro.region) - set(regions))
regions += others
MARK = {"Alaska": "o", "Lena": "s", "Canada": "D"}
for i, r in enumerate(others):
    MARK[r] = ["^", "v", "P", "X", "*", "<", ">"][i % 7]
RNAME = {"Alaska": "알래스카", "Lena": "레나델타", "Canada": "캐나다", "Russia_W": "러시아 서부", "Russia_C": "러시아 중부",
         "Russia_E": "러시아 동부", "Svalbard": "스발바르", "Greenland": "그린란드", "Mongolia_CAsia": "몽골·중앙아시아",
         "Alps": "알프스", "Tibet": "티베트"}

XA_MAX, XB_MAX = 3.2, 6.0
y = np.arange(len(ORDER))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), sharey=True, gridspec_kw=dict(wspace=0.08))

# (a) 지역 내
ax = axes[0]
for yi, f in zip(y, ORDER):
    r = ind.loc[f]
    col = DEEP if f.startswith("F7") else (TEAL if f == "F1k1" else GREY)
    ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=col, lw=1.4, solid_capstyle="round")
    ax.plot(r.delta_vs_stefan, yi, marker="o" if f.endswith("k1") else "s", ms=6, color=col,
            mec="white", mew=0.6, zorder=3)
    if r.ci_hi > XA_MAX:                      # 축 밖으로 잘린 CI 상한은 값으로 표기
        ax.annotate(f"CI 상한 {r.ci_hi:+.1f}", xy=(XA_MAX, yi), xytext=(-3, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7.8, color=col)
ax.axvline(0, color="#555555", lw=0.8, ls="--")
ax.set_yticks(y)
ax.set_yticklabels([LABEL[f] for f in ORDER], fontsize=9)
ax.set_xlabel(f"ΔRMSE 대 Stefan $E{R}$ (cm, 같은 셀, 음수=개선)")
ax.set_xlim(-1.6, XA_MAX)
# k 구분선·라벨
sep = (y[3] + y[4]) / 2
ax.axhline(sep, color="#bbbbbb", lw=0.6)
ax.text(3.15, y[0] + 0.15, "k = 1", ha="right", va="bottom", fontsize=9, color="#333333")
ax.text(3.15, y[4] + 0.15, "k = 2", ha="right", va="bottom", fontsize=9, color="#333333")
k0txt = "  ·  ".join(f"{LABEL[f].replace('(무보정)', '')} k=0: {ind.loc[f].delta_vs_stefan:+.1f}" for f in K0)
ax.text(0.0, -0.22, f"무보정(k=0, 축 밖): {k0txt} cm", transform=ax.transAxes, fontsize=8.0, color="#555555", va="top")
ax.text(0.01, 1.01, "(a) 지역 내: 알래스카 0.5° 공간블록 6-fold", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
paperize(ax)

# (b) 전이 LORO: 지역별 Δ + 비가중 평균
ax = axes[1]
gate = res[res.axis == "LORO_gate"].set_index("formula")
ref_gate = gate.loc["F1k1", "rmse_cm"]
for yi, f in zip(y, ORDER):
    col = DEEP if f.startswith("F7") else (TEAL if f == "F1k1" else GREY)
    sub = loro[loro.formula == f].set_index("region")
    for r in regions:
        if r in sub.index and np.isfinite(sub.loc[r, "delta_vs_stefan"]):
            ax.plot(sub.loc[r, "delta_vs_stefan"], yi, marker=MARK[r], ms=5.5, color=col, mec="white", mew=0.5,
                    ls="none", zorder=3, alpha=0.95)
    d_mean = gate.loc[f, "rmse_cm"] - ref_gate
    ax.plot([d_mean, d_mean], [yi - 0.32, yi + 0.32], color=col, lw=2.2, solid_capstyle="butt", zorder=2)
ax.axvline(0, color="#555555", lw=0.8, ls="--")
ax.axhline(sep, color="#bbbbbb", lw=0.6)
ax.set_xlabel(f"ΔRMSE 대 Stefan $E{R}$ (cm, 지역별 짝지은 셀)")
ax.set_xlim(-1.6, XB_MAX)
from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], marker=MARK[r], color="#666666", ls="none", ms=5.5, label=RNAME.get(r, r)) for r in regions]
handles.append(Line2D([], [], color="#666666", lw=2.2, label="지역 비가중 평균"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8.5, frameon=False, ncol=len(handles))
clipped = []
for f in ORDER:
    sub = loro[loro.formula == f].set_index("region")
    for r in regions:
        if r in sub.index and sub.loc[r, "delta_vs_stefan"] > XB_MAX:
            clipped.append(f"{LABEL[f].split('  ')[0]}·{RNAME.get(r, r)} {sub.loc[r, 'delta_vs_stefan']:+.1f}")
    dm = gate.loc[f, "rmse_cm"] - ref_gate
    if dm > XB_MAX:
        clipped.append(f"{LABEL[f].split('  ')[0]} 평균 {dm:+.1f}")
k0b = "  ·  ".join(f"{LABEL[f].replace('(무보정)', '')} k=0 평균 {gate.loc[f, 'rmse_cm'] - ref_gate:+.1f}" for f in K0)
ax.text(0.01, 1.01, f"(b) 전이: 지역 단위 홀드아웃(LORO, {len(regions)}지역), Stefan 비가중 평균 {ref_gate:.2f} cm", transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="bottom")
ax.text(0.0, -0.22, "축 밖(cm): " + k0b + ("  ·  " + "  ·  ".join(clipped) if clipped else ""),
        transform=ax.transAxes, fontsize=8.0, color="#555555", va="top", wrap=True)
paperize(ax)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"e2_ladder_delta{TAG}.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[fig] {OUT.name}/e2_ladder_delta{TAG}.png+pdf")

# ---------------- 멱지수 b ----------------
b = coefs[coefs.formula == "F2k2"]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
bi = b[b.axis == "indomain_AK"]
ax.plot(np.zeros(len(bi)) + np.linspace(-0.12, 0.12, len(bi)), bi.b, "o", color=TEAL, ms=6, mec="white", mew=0.6, label="지역 내 fold (알래스카, n=6)")
bl = b[b.axis == "LORO"]
for i, (_, r) in enumerate(bl.iterrows()):
    ax.plot(1 + (i - (len(bl) - 1) / 2) * 0.12, r.b, marker=MARK.get(r.region, "o"), color=GREY, ms=6, mec="white", mew=0.6, ls="none")
ax.axhline(0.5, color="#555555", lw=0.8, ls="--")
ax.text(1.42, 0.505, "Stefan  b = 0.5", fontsize=8.5, color="#555555", va="bottom", ha="right")
ax.set_xticks([0, 1])
ax.set_xticklabels(["지역 내 fold\n(알래스카)", "전이 학습집합\n(홀드아웃 지역별)"], fontsize=9)
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel(r"멱지수 $b$  (ALT $= a\,\mathrm{TDD}^{b}$)")
handles = [Line2D([], [], marker="o", color=TEAL, ls="none", ms=6, label="지역 내 fold")]
handles += [Line2D([], [], marker=MARK.get(r, "o"), color=GREY, ls="none", ms=6, label=f"홀드아웃 {RNAME.get(r, r)}") for r in bl.region]
ax.legend(handles=handles, fontsize=8, frameon=True, edgecolor="#d5dae0", loc="upper left")
paperize(ax, grid_axis="y")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"e2_exponent_b{TAG}.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[fig] {OUT.name}/e2_exponent_b{TAG}.png+pdf")
