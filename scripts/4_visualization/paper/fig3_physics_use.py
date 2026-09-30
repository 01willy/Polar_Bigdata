"""Fig 3 (v2). 물리 정보의 사용: 증강 위약, 결합 구조, 라벨 전량, 계수 부분 풀링. 스펙: figure_spec.json display_items 'Fig 3'(f3_v2).

패널
  a  L15 위약 대조: D1 − 위약 4종, n = 0(실선 CI)·10(파선 CI). 층화 평균 P4(지역 4/4).
  b  L10(물리 입력 대 잔차), L12(곱셈 대 덧셈 잔차), L17(대상 라벨 셔플). λ 0.25 실선, 1.0 파선. 동등성 한계 ±0.5 cm 띠.
  c  라벨 전량: R1 − P0(L8, 지역·P4·3지역·Alaska 참조), R1 − P*(L29), R1 − P1(L4). 빗금 = 오차 하한(LGX-N2) 아래.
     오른쪽 띠 = 분할 50회 가운데 Δ < 0 비율(L31, 서술).
  d  계수 부분 풀링 P1, P2, P3, V1 의 Δ vs P0(E1·E2). P* 점선, P1* 빈 사각형.
자료: data/processed/paper_figs/fig3_*.csv(v2_data.py). 판정 기호 = 원천 표 verdict4.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
import matplotlib                                                        # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Rectangle                                 # noqa: E402

import v2_style as V                                                     # noqa: E402
from v2_style import ps, NPOS, N_ALL                                     # noqa: E402

NAME = "Fig3_physics_use"
SPEC_ID = "paper_fig3_physics_use"
OFF = 0.17
DASH = ps.VARIANT_LS["dashed"]
PLACEBO = [("shuffle", "Shuffled pseudo-labels"), ("const_t", "Constant, pool mean"), ("const_src", "Constant, source mean"),
           ("tddlin", "Linear TDD")]
COEF = {  # method → (선종, 마커, 표시 이름)
    "P1": ("-", "o", "P1: E re-fit, κ = 10 shrinkage"),
    "P2": (DASH, "s", "P2: E re-fit, least squares"),
    "P3": (":", "^", "P3: E offset, maximum likelihood"),
    "V1": ("-.", "h", "V1: covariate-dependent E"),
}
COEF_DX = {"P1": -0.15, "P2": -0.05, "P3": 0.05, "V1": 0.15}


def _vcol(fig, ax_ref, x_mm, w_mm, ys, verdicts, smalls):
    """포레스트 오른쪽 판정 기호 열(같은 y 좌표)."""
    x0, y0, w, h = ax_ref._box_mm
    ax = ps.axes_mm(fig, x_mm, y0, w_mm, h)
    ax.set_ylim(ax_ref.get_ylim()); ax.set_xlim(0, 1)
    ax.set_axis_off()
    for y, v, s in zip(ys, verdicts, smalls):
        V.draw_verdict(ax, 0.35, y, v, bool(s))
    ax.set_gid("verdict_col")
    return ax


def _ci(ax, y, lo, hi, color, ls="-", lw=None):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=color, ls=ls, lw=lw or ps.LW["ci_forest"], solid_capstyle="round", dash_capstyle="butt", zorder=2.5)


def _pt(ax, x, y, color, marker, ms=None):
    ax.plot([x], [y], ls="none", marker=marker, ms=ms or ps.MS["main"], mfc=color, mec=color, mew=ps.LW["marker_edge"], zorder=3)


def _forest_frame(ax, labels, xlim, xticks, xlabel, symlog=False):
    n = len(labels)
    ys = np.arange(n)[::-1].astype(float)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    if symlog:
        V.symlog_y(ax, lim=xlim, ticks=xticks, linscale=0.7, axis="x")
    else:
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0 if float(v).is_integer() else 1)))
    ax.set_xlabel(xlabel)
    return ys


def panel_a(fig, rect, vx):
    A = V.rd("fig3_a")
    ax = ps.axes_mm(fig, *rect)
    c = ps.COLOR["augment"]
    ys = _forest_frame(ax, [lab for _, lab in PLACEBO], (-5, 1), (-4, -3, -2, -1, 0, 1), "ΔRMSE, D1 − placebo (cm)")
    ps.zero_line(ax, "x", better_text=False)
    vy, vv, vs = [], [], []
    for y, (pl, _) in zip(ys, PLACEBO):
        for n, dy, ls in ((0, OFF, "-"), (10, -OFF, DASH)):
            r = A[(A.placebo == pl) & (A.n == n)].iloc[0]
            _ci(ax, y + dy, r.ci_lo, r.ci_hi, c, ls)
            _pt(ax, r.delta, y + dy, c, "^")
            vy.append(y + dy); vv.append(r.verdict); vs.append(r.small_effect)
    ax.text(0.99, 1.0, "solid n = 0, dashed n = 10", transform=ax.transAxes, ha="right", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    _vcol(fig, ax, vx, 6.0, vy, vv, vs)
    return ax, A


def panel_b(fig, rect, vx):
    B = V.rd("fig3_b")
    ax = ps.axes_mm(fig, *rect)
    c = ps.COLOR["residual"]
    rows = [("L10", "R0-F1k|n0", "L10  R0 − F1k, n = 0"), ("L10", "R0-F1a|n0", "L10  R0 − F1a, n = 0"),
            ("L10", "R1-F1k|n10", "L10  R1 − F1k, n = 10"), ("L10", "R1-F1n|n10", "L10  R1 − F1n, n = 10"),
            ("L12", "RM-R1|n10", "L12  RM − R1, n = 10"), ("L12", "RM-R1|n-1", "L12  RM − R1, all labels"),
            ("L17", "R1-R1s|n10", "L17  R1 − R1s, n = 10"), ("L17", "R1-R1s|n-1", "L17  R1 − R1s, all labels")]
    ys = _forest_frame(ax, [lab for *_, lab in rows], (-5.5, 2), (-5, -4, -3, -2, -1, 0, 1, 2), "ΔRMSE, first − second method (cm)")
    ax.axvspan(-0.5, 0.5, facecolor=ps.GREY["band"], edgecolor="none", zorder=0.1)
    ps.zero_line(ax, "x", better_text=False)
    vy, vv, vs = [], [], []
    for y, (h, key, _) in zip(ys, rows):
        if h == "L10":
            r = B[B.contrast == key].iloc[0]
            _ci(ax, y, r.ci_lo, r.ci_hi, c); _pt(ax, r.delta, y, c, "D")
            vy.append(y); vv.append(r.verdict); vs.append(r.small_effect)
        else:
            for lam, dy, ls in (("0.25", OFF, "-"), ("1.0", -OFF, DASH)):
                r = B[B.contrast == f"{key}|lam{lam}"].iloc[0]
                _ci(ax, y + dy, r.ci_lo, r.ci_hi, c, ls); _pt(ax, r.delta, y + dy, c, "D")
                vy.append(y + dy); vv.append(r.verdict); vs.append(r.small_effect)
    for yb in (ys[3] - 0.5, ys[5] - 0.5):
        ax.axhline(yb, color=ps.GREY["light"], lw=0.5, zorder=0.2)
    ax.text(0.99, 1.0, "L12, L17: solid λ = 0.25, dashed λ = 1.0; grey band ±0.5 cm", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    _vcol(fig, ax, vx, 6.0, vy, vv, vs)
    return ax, B


def panel_c(fig, rect, vx, sx):
    Cc = V.rd("fig3_c")
    S = V.rd("fig3_c_splits")
    F = V.rd("fig3_c_floor")
    ax = ps.axes_mm(fig, *rect)
    col = ps.COLOR["residual"]
    lay = []                                       # (라벨, 행 또는 None=머리, 종류)
    for blk, hyp in (("R1 − P0", "L8"), ("R1 − P*", "L29"), ("R1 − P1", "L4")):
        lay.append((f"{blk}  ({hyp})", None))
        q = Cc[Cc.block == blk]
        for r in q.itertuples():
            nm = {"P4 mean": "P4 mean", "Lena, Canada, Alaska mean": "Mean incl. Alaska (3)", "Alaska": "Alaska (reference)"}.get(r.target, ps.region_name(r.target))
            lay.append((nm, r))
    labels = [l for l, _ in lay]
    ys = _forest_frame(ax, labels, (-18, 6), (-15, -5, -2, -1, 0, 1, 2, 5), "ΔRMSE, R1 − baseline (cm)", symlog=True)
    ax.spines["left"].set_visible(True); ax.spines["left"].set_color(ps.GREY["light"]); ax.spines["left"].set_linewidth(0.5)
    ps.zero_line(ax, "x", better_text=False)
    ticks = ax.get_yticklabels()
    vy, vv, vs = [], [], []
    fl = dict(zip(F.target, F.floor_minus_p0))
    for (lab, r), y, tl in zip(lay, ys, ticks):
        if r is None:
            tl.set_fontweight("bold")
            continue
        mean = r.kind in ("mean4", "mean3")
        mk = "D" if not mean else "o"
        _ci(ax, y, r.ci_lo, r.ci_hi, col)
        _pt(ax, r.delta, y, col, mk, ms=ps.MS["mean"] if mean else ps.MS["main"])
        vy.append(y); vv.append(r.verdict); vs.append(r.small_effect)
        if r.block == "R1 − P0" and r.target in fl:                   # 오차 하한(LGX-N2): 회색 굵은 세로 눈금
            ax.plot([fl[r.target]] * 2, [y - 0.38, y + 0.38], color="#8c8c8c", lw=1.6, solid_capstyle="butt", zorder=2.2)
    ax.text(0.0, 1.0, "grey tick: error floor − P0 (LGX-N2)", transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["annot"],
            color=ps.GREY["text2"])
    _vcol(fig, ax, vx, 5.0, vy, vv, vs)
    # 분할 비율 띠
    x0, y0, w, h = ax._box_mm
    sa = ps.axes_mm(fig, sx, y0, 9.5, h)
    sa.set_ylim(ax.get_ylim()); sa.set_xlim(0, 1)
    sa.set_yticks([])
    sa.spines["left"].set_visible(False)
    sa.set_xticks([0, 0.5, 1]); sa.set_xticklabels(["0", "0.5", "1"])
    sa.set_xlabel("Share of splits\nwith Δ < 0", linespacing=1.0)
    sa.axvline(0.8, color=ps.GREY["mid"], lw=0.5, ls=(0, (2, 1.5)), zorder=0.5)
    cmap = {"R1 − P0": "R1-P0|all", "R1 − P1": "R1-P1|all"}
    srows = []
    for (lab, r), y in zip(lay, ys):
        if r is None or r.kind not in ("region", "ref") or r.block not in cmap:
            continue
        q = S[(S.target == r.target) & (S.contrast == cmap[r.block])]
        if len(q):
            f = float(q.frac_neg.iloc[0])
            sa.barh(y, f, height=0.55, color="#d9d9d9", edgecolor="#4d4d4d", lw=0.4, zorder=2)
            srows.append(dict(block=r.block, target=r.target, frac_neg=f, n_splits=int(q.n_splits.iloc[0]),
                              lg5_outside_10_90=bool(q.lg5_outside_10_90.iloc[0])))
    return ax, Cc, pd.DataFrame(srows), F


def panel_d(fig, x_e, x_f, y, h):
    D = V.rd("fig3_d")
    P1S = V.rd("fig3_d_p1star")
    PSP = V.rd("fig2_pstar_pool")
    c = ps.COLOR["refit"]
    ae = ps.axes_mm(fig, x_e, y, 19.0, h)
    af = ps.axes_mm(fig, x_f, y, 47.0, h)
    for ax, ed in ((ae, "E1_P4_n_le_10"), (af, "E2_LenaCanada_all_n")):
        V.linear_y(ax, lim=(-3.6, 10.0), ticks=(-2, 0, 2, 4, 6, 8, 10))
        ps.zero_line(ax, "y", better_text=False)
        ln = ax.axhline(float(PSP[PSP.edition == ed].pstar.iloc[0]), **V.PSTAR_STYLE, zorder=1.7); ln.set_gid("pstar")
        q = D[D.edition == ed]
        for m, (ls, mk, _) in COEF.items():
            g = q[q.method == m]
            for part in (g[g.n != N_ALL].sort_values("n"), g[g.n == N_ALL]):
                if not len(part):
                    continue
                xs = np.array([NPOS[int(n)] for n in part.n]) + COEF_DX[m]
                if m == "P1":                                   # CI 는 P1 만(나머지는 Source Data). 겹침 방지
                    for x, lo, hi in zip(xs, part.ci_lo, part.ci_hi):
                        ax.plot([x, x], [lo, hi], color=c, lw=ps.LW["ci"], zorder=2.4)
                ax.plot(xs, part.delta, color=c, ls=ls if len(part) > 1 else "none", lw=ps.LW["main"], marker=mk, ms=ps.MS["main"],
                        mfc=c, mec=c, mew=ps.LW["marker_edge"], zorder=3)
    for r in P1S.itertuples():
        af.plot([NPOS[int(r.n)] + 0.28], [r.p1star], ls="none", marker="s", ms=ps.MS["main"] + 0.4, mfc="white", mec=c, mew=0.9, zorder=3.5)
    V.n_axis(ae, ns=[0, 3, 10], lim=(-0.5, 2.5))
    V.n_axis(af, ns=[0, 3, 10, 40, 160, 320, N_ALL])                       # E2 에는 n = 1,000 이 없다(레나만)
    ae.set_ylabel("ΔRMSE vs P0 (cm)")
    ps.hide_ylabels(af)
    for ax, t in ((ae, "P4, n ≤ 10"), (af, "Lena + Canada pool")):
        ax.text(0.0, 1.03, t, transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["base"])
    V.better_note(af, "negative = lower error than P0")
    return ae, af, D, P1S


def build(draft: bool = False):
    ps.use_paper()
    W, H = 180.0, 140.0
    fig = ps.paper_figure(W, H)
    # 왼쪽 열: a(위), c(아래). 오른쪽 열: b(위), d(아래)
    ax_a, A = panel_a(fig, (33.0, 88.0, 44.0, 26.0), vx=78.0)
    ax_c, Cc, S, F = panel_c(fig, (30.0, 12.0, 43.0, 60.0), vx=73.5, sx=80.0)
    ax_b, B = panel_b(fig, (128.0, 76.0, 44.0, 38.0), vx=172.5)
    ae, af, D, P1S = panel_d(fig, 108.0, 130.0, 12.0, 45.0)
    ps.panel_label(ax_a, "a", dx_mm=-31.0, dy_mm=1.2)
    ps.panel_label(ax_c, "c", dx_mm=-28.0, dy_mm=1.2)
    ps.panel_label(ax_b, "b", dx_mm=-36.0, dy_mm=1.2)
    ps.panel_label(ae, "d", dx_mm=-12.0, dy_mm=1.2)
    handles = [Line2D([], [], color=ps.COLOR["refit"], ls=ls, lw=ps.LW["main"], marker=mk, ms=ps.MS["main"], mfc=ps.COLOR["refit"],
                      mec=ps.COLOR["refit"], label=lab) for ls, mk, lab in COEF.values()]
    handles += [V.pstar_handle(), V.p1star_handle()]
    ps.legend_below(fig, handles, (4.0, H - 12.0, W - 8.0, 10.0), ncol=3)
    # 판정 기호 설명(범례 객체 아님)
    kax = ps.axes_mm(fig, 4.0, 122.5, 172.0, 4.5)
    kax.set_axis_off(); kax.set_xlim(0, 1); kax.set_ylim(0, 1)
    V.verdict_key(kax, 0.0, 0.5, include_small=True)
    kax.set_gid("verdict_key")

    cap = dict(
        definition=("ΔRMSE is the RMSE difference in cm (first method minus second; negative is lower error for the first). D1 adds "
                    "Stefan pseudo-labels (κ = 10 re-fit E times √TDD) to direct CatBoost; placebos replace them. F1k, F1a and F1n give "
                    "physics outputs to direct ML as inputs; RM is a multiplicative residual; R1s shuffles the target labels. P* is P0 "
                    "with year-matched TDD and P1* the edaphic Stefan re-fit (L29)."),
        statistics=("Points and bars, cell-weighted stratified means with block-bootstrap 95% CI (10,000 resamples), four-region pool P4 "
                    "unless stated. Verdict symbols (right of each forest) are copied from the LGX tables and use both cell-weighted and "
                    "block-equal-weighted CIs with a 0.5 cm margin; † marks |Δ| < 0.5 cm. Panel d has no verdicts (descriptive)."),
        panels=("a, placebo controls (L15, registered verdict: mixed). b, combination structure: physics as input versus residual (L10, "
                "supported), multiplicative versus additive residual (L12, rejected), shuffled target labels (L17). c, all target labels: "
                "R1 against P0 (L8), P* and P1 (L4) by region, P4 and the three-region mean with Alaska (x); Alaska is shown as a "
                "reference row and is not pooled with P4. Grey ticks, covariate-conditional error floor minus P0 (LGX-N2). Bars, "
                "share of 50 splits (49 for Lena) with Δ < 0 (L31, dashed line 0.8). d, coefficient models against P0; the Lena and Canada P* line "
                "and P1* points are point estimates; CIs are drawn for P1 only (all CIs in Source Data)."),
        data=("LG and LGX runs on the Rescale platform; LGX tables are a local re-aggregation of the same shards (cross-environment "
              "check not performed)."),
    )
    src = dict(a=A, b=B, c=Cc, c_splits=S, c_floor=F, d=D, d_p1star=P1S)
    spec_extra = dict(module="fig3_physics_use.py", panels=4, message="physics as residual anchor; all-label gain depends on Russia W; P* row added",
                      pending=["AB3, AB4, AB7, AB8, AB9 markers (lgw_bundle not opened)", "SD/SE ratio flag (lgw_splitratio not opened)"])
    return V.save_v2(fig, NAME, cap, SPEC_ID, src, title="How physics enters the learner: pseudo-label controls, combination structure, "
                     "all-label effect and coefficient pooling.", spec_extra=spec_extra)


if __name__ == "__main__":
    build()
