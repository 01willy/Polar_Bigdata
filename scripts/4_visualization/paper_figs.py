"""논문 제출용 그림(Sci Rep 규격) — 영문, 그림 안 제목·각주 없음, 패널 문자 a/b, 2단 폭 180 mm(또는 1단 88 mm), 7–8 pt, 벡터 PDF + 600 dpi PNG.

보고서용 그림(outputs/figures/h2, h3)과 같은 데이터·같은 결과 CSV 에서 생성하되 표현만 저널 규격으로 바꾼다. 캡션 문안은 outputs/figures/paper/CAPTIONS.md 에 함께 쓴다.
실행: python3 scripts/4_visualization/paper_figs.py [--only fig2]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
H3 = ROOT / "data" / "processed" / "h3"; H2 = ROOT / "data" / "processed" / "h2"; M1 = ROOT / "data" / "processed" / "m1"
OUT = ROOT / "outputs" / "figures" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); args = ap.parse_args()

MM = 1 / 25.4
W2, W1 = 180 * MM, 88 * MM                                   # Sci Rep 2단·1단 폭
COL = dict(phys="#4d4d4d", s1="#1f77b4", s1r="#9ecae1", s2="#2ca25f", s3="#7b3294", s3a="#e08214", ref="#999999")
EN = {"Lena": "Lena Delta", "Canada": "Canada", "Russia_W": "Russia W", "Russia_E": "Russia E"}
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
CAP = []


def journal_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5,
        "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.2, "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.8, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
        "axes.unicode_minus": False, "mathtext.default": "regular",
    })


def label_panels(axes, letters=None, x=-0.18, y=1.04):
    letters = letters or "abcdefghijklmnop"
    for ax, l in zip(np.ravel(axes), letters):
        ax.text(x, y, l, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name, caption):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); CAP.append((name, caption)); print(f"[paper] {name}")


# ---------------------------------------------------------------- Fig. 2  Label-budget staircase (main 4 regions) + gains at n = 3 and 10
def fig2_label_budget():
    C = pd.read_csv(H3 / "h25_curve.csv"); T = pd.read_csv(H3 / "h25_targets.csv"); B = pd.read_csv(H3 / "h25_breakeven.csv")
    series = [("S1", "random", 1.0, 0.0, "E re-fit, random labels", COL["s1r"], "o", "--"),
              ("S1", "kmedoid", 1.0, 0.0, "E re-fit, representative labels", COL["s1"], "o", "-"),
              ("S2", "kmedoid", 1.0, 1.0, "+ physics pseudo-label augmentation", COL["s2"], "^", "-"),
              ("S3", "kmedoid", 1.0, 0.25, "+ residual ML (λ = 0.25)", COL["s3"], "D", "-")]
    fig = plt.figure(figsize=(W2, 112 * MM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.95, wspace=0.5, left=0.07, right=0.99, top=0.93, bottom=0.09)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, t in zip(axes, MAIN4):
        sub = C[(C.target == t) & (C.scope == "n")]
        if "n_splits" in sub:
            sub = sub[sub.n_splits == sub.n_splits.max()]
        phys = float((sub.rmse_mean - sub.d_phys_mean).iloc[0]); ax.axhline(phys, color=COL["phys"], ls=":", lw=0.9)
        for stage, rule, alpha, lam, lab, col, mk, ls in series:
            q = sub[(sub.stage == stage) & (sub.rule == rule) & (sub.alpha == alpha) & (sub.lam == lam)].sort_values("n")
            ax.plot(q.n, q.rmse_mean, marker=mk, color=col, ls=ls, mfc="white" if rule == "random" else col, mew=0.8)
        full = C[(C.target == t) & (C.scope == "allA") & (C.stage == "S3") & (C.alpha == 1.0) & (C.lam == 0.25)]
        if len(full):
            ax.axhline(float(full.rmse_mean.mean()), color=COL["s3"], ls=":", lw=0.8, alpha=0.7)
        nmax = int(sub.n.max()); ax.set_xscale("log"); ax.set_xlim(2.5, nmax * 1.3)
        ax.set_xticks([v for v in [3, 10, 30, 100, 300] if v <= nmax]); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        tr = T[T.target == t]; er = float(tr.E_own.iloc[0] / tr.E0.mean())
        ax.set_title(f"{EN[t]}, $E_{{\\rm own}}/E_0$ = {er:.2f}", loc="left", fontsize=7, pad=2)
        ax.set_xlabel("Target labels $n$")
        be = B[(B.target == t) & (B.rule == "kmedoid") & (B.stage == "S3") & (B.lam == 0.25) & (B.alpha == 1.0)]
        if len(be) and be.breakeven_n_rec50.iloc[0] > 0:
            ax.annotate(f"$n^*$ = {int(be.breakeven_n_rec50.iloc[0])}", xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom", fontsize=6.5, color=COL["s3"])
    axes[0].set_ylabel("RMSE (cm)")
    handles = [Line2D([], [], marker=mk, color=col, ls=ls, mfc="white" if rule == "random" else col, mew=0.8, label=lab) for _, rule, _, _, lab, col, mk, ls in series] + \
              [Line2D([], [], color=COL["phys"], ls=":", label="Physics anchor ($n$ = 0)"), Line2D([], [], color=COL["s3"], ls=":", alpha=0.7, label="Residual ML with all A-block labels")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.53, 0.505), ncol=3, columnspacing=1.6, handletextpad=0.5, fontsize=6.5)
    # (e) Δ at n = 3 and (f) n = 10 for the 14 targets, ordered by |log E ratio|
    nA = T.groupby("target").n_A.min(); er = (T.groupby("target").E_own.first() / T.groupby("target").E0.mean())
    targets = [t for t in C.target.unique() if nA.get(t, 0) >= 6]; order = sorted(targets, key=lambda t: -abs(np.log(er[t])))
    for k, n in enumerate((3, 10)):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2]); y = np.arange(len(order))[::-1]
        for j, (stage, rule, alpha, lam, lab, col, mk) in enumerate([("S1", "kmedoid", 1.0, 0.0, "E re-fit", COL["s1"], "o"), ("S3", "kmedoid", 1.0, 0.25, "+ residual ML", COL["s3"], "D")]):
            off = (j - 0.5) * 0.3
            for i, t in enumerate(order):
                q = C[(C.target == t) & (C.scope == "n") & (C.stage == stage) & (C.rule == rule) & (C.alpha == alpha) & (C.lam == lam) & (C.n == n)]
                if len(q):
                    q = q.iloc[0]
                    if np.isfinite(q.d_phys_lo):
                        ax.plot([q.d_phys_lo, q.d_phys_hi], [y[i] - off] * 2, color=col, lw=0.8)
                    ax.plot(q.d_phys_mean, y[i] - off, mk, color=col, ms=3.2, zorder=3)
        ax.axvline(0, color=COL["phys"], lw=0.7); ax.set_yticks(y)
        ax.set_yticklabels([f"{EN.get(t, t)} ({abs(np.log(er[t])):.2f})" for t in order], fontsize=6.2)
        ax.set_xlim(-12, 6); ax.set_xlabel(f"ΔRMSE vs physics anchor at $n$ = {n} (cm)")
        if k == 1:
            ax.set_yticklabels([])
            ax.legend(handles=[Line2D([], [], marker="o", color=COL["s1"], ls="none", label="E re-fit"), Line2D([], [], marker="D", color=COL["s3"], ls="none", label="+ residual ML")], loc="lower left")
    for ax, l in zip(fig.axes[:4], "abcd"):
        ax.text(-0.28, 1.06, l, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    for ax, l in zip(fig.axes[4:], "ef"):
        ax.text(-0.02 if l == "f" else -0.32, 1.02, l, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    save(fig, "Fig2_label_budget", "Label-budget staircase. (a–d) B-block RMSE versus the number of target labels n for the four main transfer regions; "
         "curves show E re-fit with random or representative (k-centre) labels, physics pseudo-label augmentation and residual ML (λ = 0.25). "
         "Dotted grey: physics anchor with the source coefficient E0; dotted purple: residual ML with all A-block labels; n* = break-even labels (50 % recovery, win rate ≥ 0.75). "
         "(e, f) ΔRMSE relative to the physics anchor at n = 3 and n = 10 for the 14 targets (main regions and sub-regions), ordered by |log(E_own/E0)| (in parentheses); "
         "points are means over 3 block splits × 20 label draws (residual ML: 5 draws × 2 seeds); horizontal bars are 95 % percentile bootstrap intervals over the paired (split, draw) differences. "
         "n per target (B-block cells): Lena 1,478–1,687; Canada 374–393; Russia W 12–17; Russia E 11–14; sub-regions 125–4,866. Source data: data/processed/h3/h25_curve.csv.")


REG = {"fig2": fig2_label_budget}
journal_style()
for k, fn in REG.items():
    if args.only and k not in args.only.split(","):
        continue
    fn()
(OUT / "CAPTIONS.md").write_text("\n\n".join(f"**{n}.** {c}" for n, c in CAP))
