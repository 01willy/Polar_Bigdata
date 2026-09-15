"""E1 통합 요인 설계 그림 — (1) 조건별 잔차 가중 λ 반응, (2) 증강 대조군 사다리.

스펙: figures/figure_spec.json id=e1_lambda_by_condition, e1_control_ladder.
입력: data/processed/e1_summary{TAG}.csv (e1_analysis.py 산출, seed 평균)
산출: outputs/figures/e1_factorial/e1_lambda_by_condition{TAG}.{png,pdf}, e1_control_ladder{TAG}.{png,pdf}
실행: python3 scripts/4_visualization/e1_figs.py [--tag _smoke]
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
from polar.fidelity import TRANSFER_MAIN          # noqa: E402

plt = use_polar()
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
ap.add_argument("--per-region", action="store_true", help="정보 없음 조건을 지역별 패널로(지역 평균이 소표본 폭주에 지배될 때)")
args = ap.parse_args()
TAG = args.tag
PROC = C.PROCESSED
OUT = C.FIGURES / "e1_factorial"
OUT.mkdir(parents=True, exist_ok=True)
DEEP, TEAL, GREY, LIGHT = "#2f4b6e", "#4a7c8c", "#8296a8", "#b8c4d0"
COND_NAME = {"labels": "라벨 있음(지역 내)", "covonly": "공변량만", "noinfo": "정보 없음"}
MODEL_STYLE = {"ridge": dict(color=TEAL, marker="o"), "catboost_lo": dict(color=GREY, marker="s"), "mlp": dict(color=DEEP, marker="D")}
MODEL_NAME = {"ridge": "능형 회귀", "catboost_lo": "저용량 CatBoost", "mlp": "MLP"}
ANCHOR_NAME = {"stefan": "Stefan", "stefan_cci": "Stefan+위성 제품", "stefan_soil": "토양 도일 Stefan",
               "stefan_soil_cci": "토양 도일 Stefan+위성", "cci_cal": "위성 제품(보정)", "none": "앵커 없음",
               "ku_cal": "Kudryavtsev 보정", "stefan_ku": "Stefan+Kudryavtsev", "stefan_ku_cci": "Stefan+Ku+위성"}
PSEUDO_NAME = {"none": "증강 없음", "stefan": "Stefan", "cci": "위성 제품", "const": "상수(대조)", "shuffle": "셔플 Stefan(대조)", "tddlin": "TDD 선형(대조)"}


def paperize(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.7)
        ax.spines[s].set_color("#7a7a7a")
    ax.grid(False)
    ax.grid(True, axis=grid_axis, color="#aab3bd", lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=2.5, width=0.7, color="#8a8a8a", labelcolor="#333333")


summ = pd.read_csv(PROC / f"e1_summary{TAG}.csv")
targets = sorted(set(summ.target) - {"Alaska"})
main_targets = [t for t in targets if t in TRANSFER_MAIN] or targets

# ---------------- (1b) 지역별 λ 반응(정보 없음, 지역 전체 셀) ----------------
if args.per_region:
    anchors_pr = [a for a in ["stefan", "stefan_cci", "stefan_ku_cci"] if a in set(summ.anchor)]
    regs = [t for t in ["Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"] if t in set(summ.target)]
    RN = {"Lena": "레나델타", "Canada": "캐나다", "Russia_W": "러시아 서부", "Russia_E": "러시아 동부", "Russia_C": "러시아 중부", "Greenland": "그린란드"}
    fig, axes = plt.subplots(len(anchors_pr), len(regs), figsize=(2.3 * len(regs), 2.4 * len(anchors_pr)), sharex=True, squeeze=False)
    for i, anc in enumerate(anchors_pr):
        for j, tg in enumerate(regs):
            ax = axes[i, j]
            sub = summ[(summ.anchor == anc) & (summ.cond == "noinfo") & (summ.pseudo == "none") & (summ.target == tg)]
            base = sub[sub.lam == 0].rmse.mean()
            ymax = base * 1.6
            clipped = []
            for mdl, st in MODEL_STYLE.items():
                s2 = sub[sub.resid == mdl].groupby("lam").rmse.mean()
                if not len(s2):
                    continue
                v = s2.values.copy()
                over = v > ymax
                if over.any():
                    clipped.append(f"{MODEL_NAME[mdl]} λ=1: {v[-1]:.0f}")
                ax.plot(s2.index, np.where(over, ymax, v), ms=4, lw=1.1, mec="white", mew=0.4, label=MODEL_NAME[mdl], **st)
            ax.axhline(base, color="#555555", lw=0.8, ls="--")
            ax.set_ylim(min(base * 0.8, sub.rmse.min() * 0.95), ymax * 1.02)
            if clipped:
                ax.text(0.02, 0.97, "축 밖:\n" + "\n".join(clipped), transform=ax.transAxes, fontsize=6.3, va="top", color="#555555")
            n = int(sub.n.iloc[0]) if len(sub) else 0
            if i == 0:
                ax.set_title(f"{RN.get(tg, tg)} (n={n})", fontsize=9, fontweight="normal")
            if j == 0:
                ax.set_ylabel(f"{ANCHOR_NAME[anc]}\nRMSE (cm)", fontsize=8.5)
            if i == len(anchors_pr) - 1:
                ax.set_xlabel("λ", fontsize=9)
            ax.set_xticks([0, 0.5, 1.0])
            paperize(ax)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.text(0.01, 0.995, "정보 없음 조건(알래스카만 학습, 대상 지역 전체 셀). 파선 = 앵커 단독(λ=0). y 상한 = 앵커 × 1.6", fontsize=8, color="#555555", va="top")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"e1_lambda_per_region{TAG}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/e1_lambda_per_region{TAG}.png+pdf")

# ---------------- (1) λ 반응: 앵커별 패널 열 × 조건 패널 행 ----------------
anchors = [a for a in ["stefan", "stefan_cci", "stefan_soil_cci", "stefan_ku_cci"] if a in set(summ.anchor)]
conds = [c for c in ["labels", "covonly", "noinfo"] if c in set(summ.cond)]
has_ak = "Alaska" in set(summ.target)
ncol = len(conds) + (1 if has_ak else 0)
fig, axes = plt.subplots(len(anchors), ncol, figsize=(3.3 * ncol, 2.9 * len(anchors)), sharex=True, squeeze=False)
for i, anc in enumerate(anchors):
    panels = [("Alaska", "labels")] if has_ak else []
    panels += [(None, c) for c in conds]
    for j, (tg, cond) in enumerate(panels):
        ax = axes[i, j]
        # 공변량만 조건은 증강 없는 구성이 정보 없음과 같은 학습집합이므로 Stefan 증강(r=10) 구성으로 표시
        ps = "stefan" if cond == "covonly" else "none"
        sub = summ[(summ.anchor == anc) & (summ.cond == cond) & (summ.pseudo == ps)]
        sub = sub[sub.target == "Alaska"] if tg == "Alaska" else sub[sub.target.isin(main_targets)]
        for mdl, st in MODEL_STYLE.items():
            s2 = sub[sub.resid == mdl].groupby("lam").rmse.mean()      # 지역 비가중 평균
            if len(s2):
                ax.plot(s2.index, s2.values, ms=5, lw=1.3, mec="white", mew=0.5, label=MODEL_NAME[mdl], **st)
        base = sub[sub.lam == 0].rmse.mean()
        if np.isfinite(base):
            ax.axhline(base, color="#555555", lw=0.8, ls="--")
        if i == 0:
            ttl = ("알래스카 지역 내(6-fold)" if tg == "Alaska" else
                   (COND_NAME[cond] + (", Stefan 증강 r=10" if cond == "covonly" else "") + f" ({len(main_targets)}지역 평균)"))
            ax.set_title(ttl, fontsize=9.5, fontweight="normal")
        if j == 0:
            ax.set_ylabel(f"{ANCHOR_NAME[anc]}\nRMSE (cm)", fontsize=9)
        if i == len(anchors) - 1:
            ax.set_xlabel("잔차 가중 λ")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        paperize(ax)
axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
fig.text(0.01, 0.995, "파선 = 앵커 단독(λ=0)", fontsize=8.5, color="#555555", va="top")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"e1_lambda_by_condition{TAG}.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[fig] {OUT.name}/e1_lambda_by_condition{TAG}.png+pdf")

# ---------------- (2) 증강 대조군 사다리 (공변량만, r=10, 직접 회귀) ----------------
sub = summ[(summ.cond == "covonly") & (summ.anchor == "none") & (summ.lam == 1.0)]
if len(sub):
    pseudos = [p for p in ["none", "stefan", "cci", "tddlin", "shuffle", "const"] if p in set(sub.pseudo)]
    models = [m for m in ["catboost_lo", "mlp", "ridge"] if m in set(sub.resid)]
    fig, axes = plt.subplots(1, len(targets), figsize=(3.6 * len(targets), 3.6), sharey=False, squeeze=False)
    x = np.arange(len(pseudos))
    for j, tg in enumerate(targets):
        ax = axes[0, j]
        for k, mdl in enumerate(models):
            vals = [sub[(sub.target == tg) & (sub.pseudo == p) & (sub.resid == mdl)].rmse.mean() for p in pseudos]
            ax.plot(x + (k - (len(models) - 1) / 2) * 0.16, vals, ls="none", ms=6, mec="white", mew=0.5,
                    label=MODEL_NAME[mdl], **MODEL_STYLE[mdl])
        ref = summ[(summ.cond == "covonly") & (summ.target == tg) & (summ.anchor == "stefan") & (summ.lam == 0) & (summ.pseudo == "none")].rmse.mean()
        if np.isfinite(ref):
            ax.axhline(ref, color="#555555", lw=0.8, ls="--")
            ax.text(len(pseudos) - 0.5, ref, " Stefan 앵커", fontsize=8, color="#555555", va="bottom", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([PSEUDO_NAME[p] for p in pseudos], rotation=25, ha="right", fontsize=8.5)
        ax.set_title({"Lena": "레나델타", "Canada": "캐나다"}.get(tg, tg), fontsize=9.5, fontweight="normal")
        if j == 0:
            ax.set_ylabel("RMSE (cm, 대상 B블록)")
            ax.legend(fontsize=8, frameon=False)
        paperize(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"e1_control_ladder{TAG}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/e1_control_ladder{TAG}.png+pdf")
else:
    print("[skip] control ladder: covonly direct 구성 없음")
