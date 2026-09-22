"""H25–H30 라벨 예산 프로토콜 그림 (논문급). 스펙 figures/figure_spec.json (id h3_*). 규칙은 h2_figs.py 와 동일(검증 팔레트·단위·채점 셀 명시·PDF+PNG).

입력 data/processed/h3/{h25_curve,h25_breakeven,h25_targets,h27_loo_S1,h27_loo_S3,h27_rule,h28_tests,h28_nested,h29_block_value,h30_deploy_worst,h30_deploy_tests,h3_hurts}.csv
산출 outputs/figures/h3/<id>.{png,pdf}
실행: python3 scripts/4_visualization/h3_figs.py [--only fig00,fig01,...]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.plotstyle import use_polar, despine, lon_formatter, CMAP, tnorm   # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--dpi", type=int, default=300); args = ap.parse_args()
plt = use_polar(); plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.fontsize": 9, "lines.linewidth": 1.8, "lines.markersize": 5})
LW, MS, SUP = 1.8, 5, 15                                                     # brand_tokens: linewidth 1.8, marker 5, title 15 bold
H3 = ROOT / "data" / "processed" / "h3"; H2 = ROOT / "data" / "processed" / "h2"
FIG = ROOT / "outputs" / "figures" / "h3"; FIG.mkdir(parents=True, exist_ok=True)
PAL = dict(blue="#3a6ea5", teal="#2a9d8f", purple="#8e6bbf", ochre="#b8791f", ref="#6b7280", navy="#17365d", light="#c9d3df")
KR = {"Lena": "레나델타", "Canada": "캐나다", "Russia_W": "러시아 서부", "Russia_E": "러시아 동부", "Alaska": "알래스카"}
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
PARENT_COL = {"Alaska": PAL["blue"], "Canada": PAL["teal"], "Lena": PAL["purple"], "Russia_W": PAL["ochre"], "Russia_E": PAL["navy"]}
EXPORTS = []


def save(fig, name):
    for ext in ("png", "pdf"):
        p = FIG / f"{name}.{ext}"; fig.savefig(p, dpi=args.dpi if ext == "png" else None, bbox_inches="tight", pad_inches=0.45); EXPORTS.append(str(p.relative_to(ROOT)))
    plt.close(fig); print(f"[fig] {name}", flush=True)


def label_of(t):
    return KR.get(t, t)


def ci_bar(ax, y, lo, hi, color, lw=1.4):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=2)


# ====================================================================== fig00 개요 도식
def fig00_overview():
    fig, ax = plt.subplots(figsize=(13.0, 5.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f6f9", ec="#444", fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.0, color="#444"))

    box(0.2, 4.35, 12.6, 0.65, "입력: 원천 지역 실측(알래스카 등) + 대상 지역 공변량 25종(기후·토양·지형·CCI). 물리 앵커 ALT = E·√TDD", fc="#e8eef5", bold=True)
    cols = [(0.2, "라벨 0개", "1. 물리 앵커 E0(원천 최소제곱)\n2. Stefan+CCI 등가중 결합\n3. 물리 유사라벨 증강 ML = 물리식 수준\n4. AOA 밖 셀은 물리식만(최악 완화)", "#f3ede4"),
            (4.4, "라벨 3~10개", "1. 관측 설계: 공변량 k-중심 대표점 선택(비지도)\n2. E 수축 적합 (n·Ê + κ·E0)/(n+κ)\n3. 수준 오차 지역은 여기서 대부분 회복\n(잔차 ML 몫 ≈ 0)", "#e6f2ef"),
            (8.6, "라벨 수십 개 이상", "1. E 재적합(수축)\n2. 원천 잔차 ∪ 대상 잔차로 CatBoost 잔차 학습\n3. 예측 = 앵커 + λ·잔차 (λ 0.25~0.5)\n4. 구조 오차 지역에서 물리식을 넘음", "#e9e7f3")]
    for x, title, body, fc in cols:
        box(x, 3.55, 4.2, 0.55, title, fc=fc, bold=True, fs=11); box(x, 1.25, 4.2, 2.2, body, fc="#ffffff", fs=9.2)
        arrow(x + 2.1, 4.35, x + 2.1, 4.12)
    for x0 in (4.4, 8.6):
        ax.add_patch(FancyArrowPatch((x0 - 0.02, 2.35), (x0 + 0.02, 2.35), arrowstyle="-|>", mutation_scale=14, lw=1.2, color="#444"))
        ax.text(x0 + 0.12, 2.6, "라벨 추가", ha="left", fontsize=8, color="#444")
    box(0.2, 0.25, 12.6, 0.8, "채점: 같은 셀에서 짝지은 ΔRMSE, 블록 분할 A/B, 지역 층화 블록 부트스트랩 95% CI\n확정: 라벨 있음 -0.7~-1.1 cm(H13) · 라벨 3개 k-중심 선택 -0.7~-2.4 cm 대 무작위(H24; 레나·캐나다·러시아 W, 러시아 E 는 +0.2) · 라벨 0 = 물리식 수준(H3·H18~H22)", fc="#f4f6f9", fs=9.0)
    ax.set_title("라벨 예산이 배포 절차를 결정한다: 0개는 물리 앵커, 3~10개는 관측 설계와 E 수축, 수십 개부터 잔차 ML", loc="left", fontsize=SUP)
    save(fig, "h3_fig00_overview")


# ====================================================================== fig01 라벨 예산 계단
def fig01_budget_curves():
    C = pd.read_csv(H3 / "h25_curve.csv"); B = pd.read_csv(H3 / "h25_breakeven.csv"); T = pd.read_csv(H3 / "h25_targets.csv")
    nA = T.groupby("target").n_A.min()
    subs = sorted([t for t in C.target.unique() if t not in MAIN4 and nA.get(t, 0) >= 6])
    series = [("S1", "random", 1.0, 0.0, "E 수축 · 무작위", PAL["ref"], "o", ":"), ("S1", "kmedoid", 1.0, 0.0, "E 수축 · k-중심", PAL["blue"], "o", "-"),
              ("S2", "kmedoid", 1.0, 1.0, "+유사라벨 증강 ML", PAL["teal"], "^", "-"), ("S3", "kmedoid", 1.0, 0.25, "+잔차 ML(λ=.25)", PAL["purple"], "D", "-"),
              ("S3", "kmedoid", 10.0, 0.25, "+잔차 ML(α=10)", PAL["ochre"], "v", "--")]
    LEG = [Line2D([], [], marker=mk, color=col, ls=ls, label=lab) for _, _, _, _, lab, col, mk, ls in series] + \
          [Line2D([], [], color=PAL["ref"], ls="--", label="물리식(E0, n=0)"), Line2D([], [], color=PAL["purple"], ls=":", label="라벨 전량(A 전체) 잔차 ML"),
           Line2D([], [], color=PAL["purple"], lw=0.9, alpha=0.7, label="손익분기 n(회복률 50%·승률 75%)")]

    def panel(ax, t, big):
        sub = C[(C.target == t) & (C.scope == "n")]
        if "n_splits" in sub:
            sub = sub[sub.n_splits == sub.n_splits.max()]                       # 분할 일부에만 있는 n 제외(AL-1 등 |A| 불균형)
        phys = float((sub.rmse_mean - sub.d_phys_mean).iloc[0]); ax.axhline(phys, color=PAL["ref"], ls="--", lw=1.2)
        for stage, rule, alpha, lam, lab, col, mk, ls in series:
            q = sub[(sub.stage == stage) & (sub.rule == rule) & (sub.alpha == alpha) & (sub.lam == lam)].sort_values("n")
            if len(q):
                ax.plot(q.n, q.rmse_mean, marker=mk, color=col, ls=ls, lw=LW, ms=MS)
        full = C[(C.target == t) & (C.scope == "allA") & (C.stage == "S3") & (C.alpha == 1.0) & (C.lam == 0.25)]
        if len(full):
            ax.axhline(float(full.rmse_mean.mean()), color=PAL["purple"], ls=":", lw=1.1)
        be = B[(B.target == t) & (B.rule == "kmedoid") & (B.stage == "S3") & (B.lam == 0.25) & (B.alpha == 1.0)]
        nmin = int(sub.n.min()); nmax = int(sub.n.max())
        if len(be) and be.breakeven_n_rec50.iloc[0] > 0:
            bn = int(be.breakeven_n_rec50.iloc[0])
            if bn > nmin:
                ax.axvline(bn, color=PAL["purple"], lw=0.9, alpha=0.7)
            ax.text(0.98, 0.96, f"손익분기 n = {bn}", transform=ax.transAxes, ha="right", va="top", fontsize=9 if big else 8.5, color=PAL["purple"], bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_xscale("log"); ax.set_xlim(2.6, nmax * 1.25)
        ax.set_xticks([v for v in [3, 10, 40, 160] if v <= nmax]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        tr = T[T.target == t]; er = float(tr.E_own.iloc[0] / tr.E0.mean())
        ax.set_title(f"{label_of(t)} (A {int(tr.n_A.mean())}셀 · E비 {er:.2f})", loc="left", fontsize=12 if big else 10.5)
        ax.tick_params(labelsize=9); despine(ax); ax.set_xlabel("라벨 수 n", fontsize=10.5)

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.6))
    for ax, t in zip(axes, MAIN4):
        panel(ax, t, True); ax.set_ylabel("RMSE (cm), B블록", fontsize=10.5)
    fig.legend(handles=LEG, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("라벨 예산 계단(주 4지역): 수준 오차가 큰 러시아 서부만 라벨 3개로 물리식을 넘고, 레나·캐나다는 라벨을 늘려도 E 재적합 경로로는 못 넘는다", fontsize=SUP, fontweight="bold", y=1.03)
    fig.text(0.01, -0.22, "원천 = 대상 외 전체 라벨(100 km 버퍼), 앵커 E0 = 원천 최소제곱. 분할 3 × 반복(해석적 20, CatBoost 5 × seed 2). 채점 = B블록 실측·CCI·토양 도일 유효 셀(분할 3 풀링; 세 분할 모두에 있는 n 만). 근거: data/processed/h3/h25_{curve,breakeven,targets}.csv", fontsize=8, color="#555")
    fig.tight_layout(); save(fig, "h3_fig01_label_budget_curves")
    ncol = 4; nrow = int(np.ceil(len(subs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16.0, 3.9 * nrow)); axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.flat):
        if k < len(subs):
            panel(ax, subs[k], False)
            ax.set_ylabel("RMSE (cm), B블록", fontsize=9.5)
        else:
            ax.set_visible(False)
    fig.legend(handles=LEG, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("라벨 예산 계단(하위 지역 11개): 손익분기는 E비가 1에서 먼 하위 지역(AL-1·AL-3·CA-2)에서만 나타난다", fontsize=SUP, fontweight="bold", y=1.0)
    fig.text(0.01, -0.09, "하위 지역 = 블록 중심 k-means(알래스카 6·캐나다 3·레나 2, 라벨 미사용). |A| < 6 인 CA-1 제외. AL-1 은 4,716셀 단일 블록 때문에 A/B 가 불균형(288~4,716)이라 n ≤ 160 만 표시. 원천·채점·반복은 주 지역 그림과 동일.", fontsize=8, color="#555")
    fig.tight_layout(); save(fig, "h3_fig01b_label_budget_curves_subregions")


# ====================================================================== fig02 회복률·손익분기
def fig02_recovery():
    C = pd.read_csv(H3 / "h25_curve.csv"); B = pd.read_csv(H3 / "h25_breakeven.csv"); T = pd.read_csv(H3 / "h25_targets.csv")
    nA = T.groupby("target").n_A.min(); er = (T.groupby("target").E_own.first() / T.groupby("target").E0.mean())
    targets = [t for t in C.target.unique() if nA.get(t, 0) >= 6]
    order = sorted(targets, key=lambda t: -er[t])                                   # E 비 큰 순(수준 오차 큰 지역 위)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.0), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    series = [("S1", "random", 1.0, 0.0, "E 수축 · 무작위", PAL["ref"], "o"), ("S1", "kmedoid", 1.0, 0.0, "E 수축 · k-중심", PAL["blue"], "o"),
              ("S3", "kmedoid", 1.0, 0.25, "+잔차 ML(λ=.25)", PAL["purple"], "D"), ("S3", "kmedoid", 100.0, 0.25, "+잔차 ML(α=100)", PAL["ochre"], "v")]
    for k, n in enumerate((3, 10)):
        ax = axes[k]; y = np.arange(len(order))[::-1]
        for j, (stage, rule, alpha, lam, lab, col, mk) in enumerate(series):
            off = (j - 1.5) * 0.18
            for i, t in enumerate(order):
                q = C[(C.target == t) & (C.scope == "n") & (C.stage == stage) & (C.rule == rule) & (C.alpha == alpha) & (C.lam == lam) & (C.n == n)]
                if len(q):
                    q = q.iloc[0]; ci_bar(ax, y[i] - off, q.d_phys_lo, q.d_phys_hi, col, lw=1.1); ax.plot(q.d_phys_mean, y[i] - off, mk, color=col, ms=4.5, zorder=3)
        ax.axvline(0, color=PAL["ref"], lw=1); ax.set_yticks(y); ax.set_yticklabels([f"{label_of(t)} (E비 {er[t]:.2f})" for t in order], fontsize=8)
        ax.set_xlabel("ΔRMSE 대 물리식 (cm)  ← 개선 | 악화 →"); ax.set_title(f"({'ab'[k]}) 라벨 {n}개로 얻는 것 (분할·반복 짝지음 95% CI)", loc="left"); despine(ax)
        ax.set_xlim(-12, 6)
    ax = axes[2]
    b = B[(B.rule == "kmedoid") & (B.alpha == 1.0) & (B.stage.isin(["S1", "S3"])) & (B.lam.isin([0.0, 0.25])) & (B.target.isin(order))].copy()
    b["be"] = np.where(b.breakeven_n_rec50 > 0, b.breakeven_n_rec50, np.nan)
    piv = b.pivot_table(index="target", columns="stage", values="be").reindex(order); nmax = b.groupby("target").n_max.first()
    y = np.arange(len(order))[::-1]
    for j, (stage, col, mk) in enumerate((("S1", PAL["blue"], "o"), ("S3", PAL["purple"], "D"))):
        v = piv[stage].values if stage in piv else np.full(len(order), np.nan)
        for i, t in enumerate(order):
            if np.isfinite(v[i]):
                ax.plot(v[i], y[i] + (0.15 if j == 0 else -0.15), mk, color=col, ms=6)
            else:
                ax.plot(nmax[t] * 1.4, y[i] + (0.15 if j == 0 else -0.15), ">", color=col, ms=6, mfc="none")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([label_of(t) for t in order], fontsize=8); ax.set_xlabel("손익분기 n (로그; ▷ = 격자 최댓값까지 미달성)")
    ax.set_title("(c) 손익분기 n (회복률 50%·승률 75%)", loc="left"); despine(ax)
    fig.legend(handles=[Line2D([], [], marker=mk, color=col, ls="none", label=lab) for _, _, _, _, lab, col, mk in series] +
               [Line2D([], [], marker="o", color=PAL["blue"], ls="none", label="손익분기: E 수축"), Line2D([], [], marker="D", color=PAL["purple"], ls="none", label="손익분기: +잔차 ML")],
               loc="lower center", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("H25·H26 라벨 3개·10개로 얻는 것과 손익분기: 수준 오차(E비가 1에서 먼) 지역만 소수 라벨로 이득", fontsize=SUP, fontweight="bold", y=1.01)
    fig.text(0.01, -0.08, "대상은 E비 내림차순. E비 = 자체 E / 원천 E0(라벨 기반 진단). 근거: data/processed/h3/h25_{curve,breakeven,targets}.csv", fontsize=8, color="#555")
    fig.tight_layout(); save(fig, "h3_fig02_recovery_breakeven")


# ====================================================================== fig03 필요 라벨 수 규칙
def fig03_rule():
    f = H3 / "h27_loo_S3.csv"
    if not f.exists():
        print("[skip] fig03"); return
    M = pd.read_csv(f); R = pd.read_csv(H3 / "h27_rule.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for k, (xcol, xlab, ttl) in enumerate((("abs_logE", "|log(자체 E / 원천 E0)| (라벨 기반 진단; 1 에서의 거리)", "(a) 수준 오차가 클수록 손익분기 n 이 작다"),
                                          ("smd_x25", "공변량 표준화 평균차 SMD (라벨 미사용)", "(b) 공변량 이동 대 손익분기 n"))):
        ax = axes[k]
        Ms = M.sort_values([ "be_n", xcol]); offs = [(4, 4), (4, -11), (4, 13), (-4, -20), (4, 22), (-4, 30)]
        for j, (_, r) in enumerate(Ms.iterrows()):
            ax.plot(r[xcol], r.be_n, "o" if not r.censored else ">", color=PARENT_COL.get(r.parent, PAL["ref"]), ms=7, mfc="none" if r.censored else PARENT_COL.get(r.parent, PAL["ref"]))
            same = Ms[(np.abs(np.log(Ms.be_n) - np.log(r.be_n)) < 0.05)]; k = int(np.where(same.index == r.name)[0][0]) if len(same) > 1 else 0
            ax.annotate(label_of(r.target), (r[xcol], r.be_n), fontsize=7, xytext=offs[k % len(offs)], textcoords="offset points", ha="left" if offs[k % len(offs)][0] > 0 else "right")
        ax.set_yscale("log"); ax.set_ylim(2, float(M.be_n.max()) * 5); ax.set_xlabel(xlab); ax.set_ylabel("손익분기 n (S3, 로그)"); ax.set_title(ttl, loc="left"); despine(ax)
    ax = axes[2]
    if "pred_labelfree" in M:
        ax.plot(M.pred_labelfree, M.be_n, "o", color=PAL["blue"], ms=7); lim = [2, max(M.be_n.max(), M.pred_labelfree.max()) * 1.5]
        ax.plot(lim, lim, color=PAL["ref"], lw=1, ls="--"); ax.set_xscale("log"); ax.set_yscale("log")
        Ms = M.sort_values(["be_n", "pred_labelfree"]); offs = [(4, 4), (4, -11), (4, 13), (-4, -20), (4, 22), (-4, 30)]
        for _, r in Ms.iterrows():
            same = Ms[(np.abs(np.log(Ms.be_n) - np.log(r.be_n)) < 0.05)]; k = int(np.where(same.index == r.name)[0][0]) if len(same) > 1 else 0
            ax.annotate(label_of(r.target), (r.pred_labelfree, r.be_n), fontsize=7, xytext=offs[k % len(offs)], textcoords="offset points", ha="left" if offs[k % len(offs)][0] > 0 else "right")
        rr = R[(R.stage == "S3") & (R.features == "labelfree")].iloc[0]; re_ = R[(R.stage == "S3") & (R.features == "E_only")].iloc[0]
        ax.set_title(f"(c) 라벨 미사용 규칙의 LOO 예측 (R² {rr.loo_r2:.2f}, ρ {rr.spearman_pred:.2f}; |log E비| ρ {re_.spearman_Eratio_vs_be:.2f})", loc="left")
    ax.set_xlabel("예측 손익분기 n (leave-one-target-out)"); ax.set_ylabel("실제 손익분기 n"); despine(ax)
    fig.legend(handles=[Line2D([], [], marker="o", color=c, ls="none", label=label_of(p)) for p, c in PARENT_COL.items()] + [Line2D([], [], marker=">", color="#333", ls="none", mfc="none", label="미달성(우측 절단)")],
               loc="lower center", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("H27 필요 라벨 수는 라벨 없이 예측되지 않지만, 대표점 3개로 잰 E비(|log E비|)가 손익분기 n 을 ρ -0.84 로 예측한다", fontsize=SUP, fontweight="bold", y=1.02)
    nc = int(M.censored.sum()) if "censored" in M else 0
    fig.text(0.01, -0.12, f"손익분기 n = 회복률 50%·승률 75% 를 처음 넘는 n(S3 λ=.25, k-중심). 미달성 {nc}/{len(M)} 대상은 격자 최댓값 × 2 로 우측 절단 대치해 순위 상관·LOO 회귀에 포함(절단 처리에 민감). 근거: data/processed/h3/h27_{{loo_S3,rule}}.csv", fontsize=8, color="#555")
    fig.tight_layout(); save(fig, "h3_fig03_breakeven_rule")


# ====================================================================== fig04 블록 라벨 가치 지도
def fig04_block_value():
    f = H3 / "h29_block_value.csv"
    if not f.exists():
        print("[skip] fig04"); return
    R = pd.read_csv(f); g = R.groupby(["target", "block"]).agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("n_cells", "mean"), v1=("value_S1", "mean"), v3=("value_S3", "mean"), freq=("kmedoid_freq", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6)); fig.subplots_adjust(left=0.05, right=0.96, top=0.80, bottom=0.12, wspace=0.42)
    vmax = min(6.0, float(np.nanpercentile(np.abs(g[g.target.isin(["Canada", "Lena"])].v3), 95)) or 1.0)      # 두 패널 공통 색 범위(±6 cm 상한, 초과는 화살표)
    for ax, t in zip(axes, ["Canada", "Lena"]):
        d = g[g.target == t]
        sc = ax.scatter(d.lon, d.lat, c=-d.v3, s=20 + 4 * np.sqrt(d.n), cmap=CMAP.diff, norm=tnorm(-vmax, vmax, 0.0), edgecolor=np.where(d.freq > 0.5, "black", "none"), linewidths=0.8, zorder=3)
        ax.set_aspect(1 / np.cos(np.radians(d.lat.mean()))); ax.set_anchor("N"); ax.xaxis.set_major_formatter(lon_formatter()); despine(ax)
        dec = 1 if (d.lat.max() - d.lat.min()) < 5 else 0
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p, dec=dec: f"{v:.{dec}f}°N"))
        ax.set_title(f"{label_of(t)}: 블록 {len(d)}개, 해로운 블록 {int((d.v3 < 0).sum())}개", loc="left", pad=8)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, extend="both"); cb.set_label("ΔRMSE = (E 수축 앵커+잔차 ML) - 물리식 (cm), 청 = 개선 · 갈 = 악화", fontsize=9)
        for _, r in d[np.abs(d.v3) > vmax].iterrows():
            ax.annotate(f"{r.v3:+.0f}", (r.lon, r.lat), fontsize=7, xytext=(5, 3), textcoords="offset points", color="#333")
        ax.legend(handles=[Line2D([], [], marker="o", ls="none", color="#888", mfc="none", ms=np.sqrt(20 + 4 * np.sqrt(n)) , label=f"{n}셀") for n in (5, 50, 200)],
                  title="블록 셀 수", fontsize=8, title_fontsize=8, loc="lower left" if t == "Canada" else "upper right", frameon=True)
    harm = {t: float((g[g.target == t].v3 < 0).mean()) for t in ("Canada", "Lena")}
    fig.suptitle(f"H29 한 블록에 몰린 라벨의 가치는 지역에 따라 다르다: 해로운 블록 레나델타 {harm['Lena']*100:.0f}%, 캐나다 {harm['Canada']*100:.0f}% (검은 테두리 = k-중심이 자주 고르는 블록)", fontsize=SUP, fontweight="bold", y=0.97)
    fig.text(0.01, 0.01, "원 크기 ∝ √셀 수. 분할 3 평균, 잔차 CatBoost λ=.25 seed 2. 두 패널 공통 색 범위 ±6 cm(초과는 화살표, 극단값은 수치 표기). 등장방형 근사라 캐나다 고위도에서 동서 거리가 과장됨. 근거: data/processed/h3/h29_block_value.csv", fontsize=8, color="#555")
    save(fig, "h3_fig04_block_label_value")


# ====================================================================== fig05 라벨 있음 레시피 확정
def fig05_recipe():
    f = H3 / "h28_tests.csv"
    if not f.exists():
        print("[skip] fig05"); return
    T = pd.read_csv(f); PK = pd.read_csv(H3 / "h28_nested.csv")
    specs = [("H28_nested_all_vs_stefan", "중첩 선택(전 후보) - Stefan  [H28]"), ("H28_nested_cb_vs_stefan", "중첩 선택(CatBoost 계열) - Stefan"), ("H28x_nested_nn_vs_stefan", "중첩 선택(신경망 계열) - Stefan"),
             ("H28x_nested_analytic_vs_stefan", "중첩 선택(해석적 앵커) - Stefan"), ("H28b_nested_cb_vs_nested_nn", "CatBoost 계열 - 신경망 계열"), ("H28ref_prespec25_vs_stefan", "사전 지정 Stefan+CatBoost λ=.25 - Stefan"),
             ("H28ref_prespec50_vs_stefan", "사전 지정 λ=.5 - Stefan"), ("H28ref_oracle_vs_stefan", "자기 지역 최선(선택 효과 포함, 참고) - Stefan"), ("H28_alaska_fold_nested_vs_stefan", "알래스카 지역 내 fold 중첩 - Stefan")]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    ax = axes[0]; y = np.arange(len(specs))[::-1]; xl, xr = -11.0, 3.0
    for i, (tid, lab) in enumerate(specs):
        sub = T[T.test == tid]
        for r in MAIN4:
            q = sub[sub.target == r]
            if len(q):
                ax.plot(q.delta.iloc[0], y[i], "o", color=PAL["blue"], ms=4, mfc="none", alpha=0.8)
        m = sub[sub.target.astype(str).str.startswith("MEAN") | (sub.target == "Alaska")]
        if len(m):
            m = m.iloc[0]; ci_bar(ax, y[i], m.ci_lo, m.ci_hi, PAL["blue"], lw=1.6); ax.plot(m.delta, y[i], "o", color=PAL["blue"], ms=6.5, zorder=4)
            ax.text(xr + 0.15, y[i], f"{m.delta:+.2f} [{m.ci_lo:+.1f}, {m.ci_hi:+.1f}] · 블록 등가중 {m.delta_blockeq:+.1f}", fontsize=7.5, va="center", ha="left", color=PAL["blue"], clip_on=False)
    ax.axvline(0, color=PAL["ref"], lw=1); ax.set_yticks(y); ax.set_yticklabels([l for _, l in specs], fontsize=8.5); ax.set_xlabel("ΔRMSE 대 Stefan(알래스카 E) (cm)  ← 개선 | 악화 →"); despine(ax); ax.set_xlim(xl, xr)
    ax.set_title("(a) 라벨 있음 조건(대상 A블록 실측 학습, B 채점), 4지역 평균·층화 블록 부트스트랩 95% CI + 지역 점", loc="left")
    ax = axes[1]; ax.axis("off")
    AN = {"ku_cal": "Ku 보정", "stefan_ku_cci": "Stefan+Ku+CCI", "stefan_cci": "Stefan+CCI", "stefan": "Stefan", "stefan_cci_cal": "Stefan+CCI(보정)", "cci_cal": "CCI 보정", "stefan_ku": "Stefan+Ku", "stefan_soil": "토양 Stefan"}
    RN = {"catboost_lo": "CatBoost", "catboost": "CatBoost(deep)", "tabm": "TabM", "ridge": "ridge", "mlp": "MLP", "ftt": "FT-T", "realmlp": "RealMLP"}
    rows = PK[PK.choice.isin(["nested_all", "nested_cb", "nested_nn", "prespec25", "oracle"])].copy()
    rows["레시피"] = rows.anchor.map(lambda a: AN.get(a, a)) + " + " + rows.resid.map(lambda r: RN.get(r, r)) + " λ" + rows.lam.map(lambda v: f"{v:g}")
    tbl = rows.pivot_table(index="target", columns="choice", values="rmse_target").reindex(MAIN4).round(1)
    rec = rows[rows.choice == "nested_all"].set_index("target").reindex(MAIN4)["레시피"]; rec_nn = rows[rows.choice == "nested_nn"].set_index("target").reindex(MAIN4)["레시피"]
    cell = [[label_of(t), (rec[t][:-len(" + CatBoost λ0")] + " (앵커만)") if rec[t].endswith(" λ0") else rec[t], f"{tbl.loc[t, 'nested_all']:.1f}", f"{tbl.loc[t, 'nested_nn']:.1f}", f"{tbl.loc[t, 'prespec25']:.1f}", f"{tbl.loc[t, 'oracle']:.1f}"] for t in MAIN4]
    tb = ax.table(cellText=cell, colLabels=["대상", "중첩 선택 레시피(전체 후보)", "중첩", "신경망\n중첩", "사전 지정\nλ.25", "자기 최선\n(참고)"], loc="center", cellLoc="center",
                  colWidths=[0.15, 0.43, 0.10, 0.11, 0.11, 0.11])
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1.0, 1.9)
    ax.text(0.0, 0.12, "신경망 중첩 선택은 4지역 모두 Stefan+CCI 앵커 + TabM 잔차(λ 0.25~0.5)", transform=ax.transAxes, fontsize=8, color="#555")
    ax.set_title("(b) 대상별 선택 레시피와 RMSE (cm, B블록 3분할 풀링)", loc="left")
    fig.suptitle("H28 라벨 있는 지역: 사전 지정 Stefan+CatBoost 잔차만 확정적으로 물리식을 넘고, 중첩 선택 레시피의 큰 이득은 집계 의존", fontsize=SUP, fontweight="bold", y=1.02)
    fig.text(0.01, -0.05, "후보 = 앵커 14 × λ 5(CatBoost 잔차) ∪ 앵커 3 × 모델 7 × λ 5. 중첩 선택은 셀 가중에서 크게 개선되나 CI 가 0 을 포함하고 블록 등가중에서는 이득이 사라진다(집계 의존). 자기 최선은 선택 효과 포함(참고). 근거: data/processed/h3/h28_{tests,nested,scores}.csv", fontsize=8, color="#555")
    fig.tight_layout(); save(fig, "h3_fig05_recipe_nested")


# ====================================================================== fig06 해로운 조합 열지도
def fig06_hurts():
    f = H3 / "h3_hurts.csv"
    if not f.exists():
        print("[skip] fig06"); return
    H = pd.read_csv(f)
    H = H[(H.n.isin([0, 3, 10, 40])) & (~H.method.str.contains("α=100"))]
    piv = H.pivot_table(index="method", columns=["target", "n"], values="d_phys")
    cols = [(t, n) for t in MAIN4 for n in [0, 3, 10, 40] if (t, n) in piv.columns]
    piv = piv[cols]
    order = piv.rank(axis=0, pct=True).mean(1).sort_values().index        # 열별 백분위 순위의 평균(러시아 서부의 큰 값·n=0 행의 열 수 차이가 지배하지 않도록)
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(13.5, 0.36 * len(piv) + 2.6))
    vmax = 4.0                                                   # 러시아 서부 -10 이 색을 지배하지 않도록 ±4 cm 로 제한(수치는 셀에 표기)
    im = ax.imshow(piv.values, cmap=CMAP.diff, norm=tnorm(-vmax, vmax, 0.0), aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{label_of(t)}\nn={n}" for t, n in cols], fontsize=8); ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=8.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 * vmax else "#222")
    for k in range(1, 4):
        ax.axvline(k * 4 - 0.5, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, extend="both"); cb.set_label("ΔRMSE 대 물리식 (cm), 청 = 개선 · 갈 = 악화", fontsize=9)
    ax.set_title("해로운 조합은 무작위 라벨 E 재적합·블록 순환 선택·증강 없는 직접 회귀다: 방법 × 라벨 수 × 지역의 Δ(대 물리식)", loc="left"); ax.grid(False)
    fig.text(0.01, -0.02, "n=0 은 정보 없음 직접 회귀(M1). 행 순서 = 열별 백분위 순위의 평균(위 = 이득, 아래 = 해). 색은 ±4 cm 에서 포화(러시아 서부 -10 cm 등 정확한 값은 셀 숫자). 근거: data/processed/h3/h3_hurts.csv(h25·h24·h23·M1 통합)", fontsize=8, color="#555")
    save(fig, "h3_fig06_what_hurts")


# ====================================================================== fig07 배포 규칙
def fig07_deploy():
    W = pd.read_csv(H3 / "h30_deploy_worst.csv"); T = pd.read_csv(H3 / "h30_deploy_tests.csv")
    rules = ["ml_direct", "aoa_direct", "stefan", "ml_resid", "aoa_resid", "stefan_cci", "cci_agree20"]
    lab = {"ml_direct": "직접 ML(전 셀)", "aoa_direct": "AOA 안 직접 ML / 밖 물리식", "stefan": "물리식 Stefan", "ml_resid": "물리+잔차 ML(전 셀)", "aoa_resid": "AOA 안 잔차 ML / 밖 물리식",
           "stefan_cci": "Stefan+CCI 등가중", "cci_agree20": "|Stefan-CCI|≤20 이면 결합, 아니면 Stefan"}
    piv = W[W.target.isin(MAIN4)].pivot_table(index="rule", columns="target", values="rmse").reindex(rules)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]; y = np.arange(len(rules))[::-1]
    ax.barh(y + 0.18, piv.mean(1), 0.34, color=PAL["blue"], label="4지역 평균 RMSE")
    ax.barh(y - 0.18, piv.max(1), 0.34, color=PAL["ochre"], label="최악 지역 RMSE")
    ax.set_yticks(y); ax.set_yticklabels([lab[r] for r in rules], fontsize=8.5); ax.set_xlabel("RMSE (cm), 정보 없음 조건"); despine(ax)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    ax.set_title("(a) 규칙별 평균과 최악 지역 오차", loc="left"); ax.set_xlim(0, piv.max(1).max() * 1.12)
    ax = axes[1]
    specs = [("H30_aoa_resid_vs_ml_direct", "AOA 게이팅 잔차 ML - 직접 ML  [H30 사전 등록 문구]"), ("H30_aoa_direct_vs_ml_direct", "AOA 게이팅 직접 ML - 직접 ML  [H30 동일 계열]"), ("H30_aoa_resid_vs_stefan", "AOA 게이팅 잔차 ML - Stefan  [H30]"), ("H30x_aoa_direct_vs_stefan", "AOA 게이팅 직접 ML - Stefan"),
             ("H30x_ml_resid_vs_stefan", "잔차 ML(전 셀) - Stefan"), ("H30x_stefan_cci_vs_stefan", "Stefan+CCI - Stefan"), ("H30x_cci_agree20_vs_stefan_cci", "CCI 일치 게이팅 - Stefan+CCI")]
    yy = np.arange(len(specs))[::-1]
    for i, (tid, l) in enumerate(specs):
        sub = T[T.test == tid]
        for r in MAIN4:
            q = sub[sub.target == r]
            if len(q):
                ax.plot(q.delta.iloc[0], yy[i], "o", color=PAL["teal"], ms=4, mfc="none")
        m = sub[sub.target.astype(str).str.startswith("MEAN")]
        if len(m):
            m = m.iloc[0]; ci_bar(ax, yy[i], m.ci_lo, m.ci_hi, PAL["teal"], lw=1.6); ax.plot(m.delta, yy[i], "s", color=PAL["teal"], ms=6.5, zorder=4)
    ax.axvline(0, color=PAL["ref"], lw=1); ax.set_yticks(yy); ax.set_yticklabels([l for _, l in specs], fontsize=8.5); ax.set_xlabel("ΔRMSE (cm)  ← 개선 | 악화 →"); despine(ax)
    ax.set_title("(b) 4지역 평균 Δ(층화 블록 부트스트랩 95% CI) + 지역 점", loc="left")
    fig.suptitle("H30 배포 규칙: AOA 게이팅 + 잔차 ML 은 물리식과 동급(+0.1), AOA 게이팅 + 직접 ML 은 최악을 줄여도 물리식보다 +1.6 열세", fontsize=SUP, fontweight="bold", y=1.02)
    fig.text(0.01, -0.05, "AOA = 적용 가능 영역(공변량 이질성 지수 DI 임계 안). 사전 등록 H30 문구의 비교(AOA 잔차 - 직접 ML)와 동일 계열 비교(AOA 직접 - 직접 ML)를 모두 표시. 근거: data/processed/h3/h30_deploy_{worst,tests}.csv (M1 모델 축 예측 재사용)", fontsize=8, color="#555")
    fig.tight_layout(); save(fig, "h3_fig07_deploy_rule")


REG = {"fig00": fig00_overview, "fig01": fig01_budget_curves, "fig02": fig02_recovery, "fig03": fig03_rule, "fig04": fig04_block_value, "fig05": fig05_recipe, "fig06": fig06_hurts, "fig07": fig07_deploy}
only = [s for s in args.only.split(",") if s]
for k, fn in REG.items():
    if only and k not in only:
        continue
    try:
        fn()
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"[skip] {k}: {type(e).__name__} {e}")
print("exports:", json.dumps(EXPORTS, ensure_ascii=False))
