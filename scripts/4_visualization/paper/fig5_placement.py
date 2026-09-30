"""Fig 5 (v2). 관측 배치: 블록 분산 대 셀 무작위(L43), 근접 성분(L23), 거리 의존(L24). 스펙: figure_spec.json display_items 'Fig 5' f3_v2.

패널
  a  L43(주): Δ_L43 = RMSE(블록 분산) − RMSE(셀 무작위), P1, n = 10(P4) 과 n = 40(레나·캐나다). 2단 재표집 CI(실선, 주)와
     추출 조건부 CI(파선, 보조). 오른쪽 기호 두 열 = 두 CI 의 4분 판정.
  b  L23: 층화 평균 주 네 행(근접 대비)과 지역 점 추정(작은 회색 점).
  c  L24: 적격 대상 × n 의 근 − 원 차(R1 − P1 실선, P1 − P0 파선). 기호 두 줄.
  d  거리 층 값(서술, n = 40): near·mid·far 의 Δ(R1 − P1), Δ(P1 − P0).
자료: data/processed/paper_figs/fig5_*.csv(v2_data.py). 판정 기호 = 원천 표 verdict4.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                       # noqa: E402
import matplotlib                                                        # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402

import v2_style as V                                                     # noqa: E402
from v2_style import ps                                                  # noqa: E402

NAME = "Fig5_placement"
SPEC_ID = "paper_fig5_placement"
DASH = ps.VARIANT_LS["dashed"]
OFF = 0.17
TGT = [("Lena|x", "Lena"), ("Canada|x", "Canada"), ("Alaska|x", "Alaska (x)"), ("AL-2|i", "AL-2 (i)")]
TMARK = {"Lena|x": "D", "Canada|x": "s", "Alaska|x": "o", "AL-2|i": "p"}


def _ci(ax, y, lo, hi, color, ls="-"):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=color, ls=ls, lw=ps.LW["ci_forest"], solid_capstyle="round", dash_capstyle="butt", zorder=2.5)


def _pt(ax, x, y, color, marker, ms=None):
    ax.plot([x], [y], ls="none", marker=marker, ms=ms or ps.MS["main"], mfc=color, mec=color, mew=ps.LW["marker_edge"], zorder=3)


def _frame(ax, labels, xlim, xticks, xlabel, bold=()):
    n = len(labels)
    ys = np.arange(n)[::-1].astype(float)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(*xlim); ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0 if float(v).is_integer() else 1)))
    ax.set_xlabel(xlabel)
    ps.zero_line(ax, "x", better_text=False)
    for tl, lab in zip(ax.get_yticklabels(), labels):
        if lab in bold:
            tl.set_fontweight("bold")
    return ys


def _vcols(fig, ax_ref, xs_mm, ys_list, verdicts_list, smalls_list, heads=None):
    x0, y0, w, h = ax_ref._box_mm
    out = []
    for k, x in enumerate(xs_mm):
        ax = ps.axes_mm(fig, x, y0, 5.0, h)
        ax.set_ylim(ax_ref.get_ylim()); ax.set_xlim(0, 1); ax.set_axis_off()
        for y, v, s in zip(ys_list[k], verdicts_list[k], smalls_list[k]):
            V.draw_verdict(ax, 0.4, y, v, bool(s))
        if heads:
            ax.text(0.4, 1.0, heads[k], transform=ax.transAxes, ha="center", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"],
                    linespacing=1.0)
        ax.set_gid("verdict_col")
        out.append(ax)
    return out


def panel_a(fig, rect, vx):
    A = V.rd("fig5_a")
    ax = ps.axes_mm(fig, *rect)
    c = ps.COLOR["refit"]
    lay = [("P1, n = 10 (P4)", None, None)] + [(nm, 10, nm) for nm in ("Lena", "Canada", "Russia_W", "Russia_E", "P4 mean")]
    lay += [("P1, n = 40 (Lena, Canada)", None, None)] + [(nm, 40, nm) for nm in ("Lena", "Canada", "Lena + Canada mean")]
    labels = [ps.region_name(l) if n is not None else l for l, n, _ in lay]
    ys = _frame(ax, labels, (-4.5, 4.0), (-4, -2, 0, 2, 4), "Δ$_{L43}$ = RMSE(block-spread) − RMSE(cell-random) (cm)",
                bold=[l for l, n, _ in lay if n is None])
    v1, v2, s1, s2, y1, y2 = [], [], [], [], [], []
    for (lab, n, key), y in zip(lay, ys):
        if n is None:
            continue
        q = A[(A.n == n) & (A.region == key)]
        mean = "mean" in key
        mk = "o" if mean else "D"
        r2 = q[q.ci_kind == "2단 재표집"].iloc[0]
        rc = q[q.ci_kind == "추출 조건부"].iloc[0]
        _ci(ax, y + OFF, r2.ci_lo, r2.ci_hi, c, "-")
        _ci(ax, y - OFF, rc.ci_lo, rc.ci_hi, c, DASH)
        _pt(ax, r2.delta, y + OFF, c, mk, ps.MS["mean"] if mean else None)
        _pt(ax, rc.delta, y - OFF, c, mk, ps.MS["mean"] if mean else None)
        y1.append(y); v1.append(r2.verdict); s1.append(False)
        y2.append(y); v2.append(rc.verdict); s2.append(False)
    ax.text(0.0, 1.01, "solid: two-stage CI (draws resampled)\ndashed: draw-conditional CI", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"], linespacing=1.05)
    _vcols(fig, ax, vx, [y1, y2], [v1, v2], [s1, s2], heads=["two-\nstage", "draw-\ncond."])
    return ax, A


def panel_b(fig, rect, vx):
    B = V.rd("fig5_b")
    ax = ps.axes_mm(fig, *rect)
    rows = [("D0[S_rand5]-D0[S_block]", "L23a  D0, random split\n    − region holdout", ps.COLOR["direct"]),
            ("[D0-P1](S_randm)-[D0-P1](S_block)", "L23b  D0 − P1, near labels\n    − holdout labels", ps.COLOR["direct"]),
            ("[R1-P1](S_randm)-[R1-P1](S_block)", "L23b  R1 − P1, near labels\n    − holdout labels", ps.COLOR["residual"]),
            ("[R1-P1](inblk)-[R1-P1](A)|n10", "L23c  R1 − P1, n = 10 in\n    scored blocks − in A", ps.COLOR["residual"])]
    ys = _frame(ax, [r[1] for r in rows], (-8.5, 3.0), (-8, -6, -4, -2, 0, 2), "Δ, first − second label set (cm)")
    for tl in ax.get_yticklabels():
        tl.set_linespacing(1.0)
    vv, ss = [], []
    for (key, _, col), y in zip(rows, ys):
        m = B[(B.contrast == key) & (B.kind == "mean")].iloc[0]
        rg = B[(B.contrast == key) & (B.kind == "region")]
        ax.scatter(rg.delta, np.full(len(rg), y - 0.26), s=5, color=ps.GREY["mid"], lw=0, zorder=2.2)
        _ci(ax, y, m.ci_lo, m.ci_hi, col)
        _pt(ax, m.delta, y, col, "o", ps.MS["mean"])
        vv.append(m.verdict); ss.append(m.small_effect)
    ax.text(1.0, 1.0, "P4 mean with CI; grey dots: regions", transform=ax.transAxes, ha="right", va="bottom", fontsize=ps.FS["annot"],
            color=ps.GREY["text2"])
    _vcols(fig, ax, [vx], [list(ys)], [vv], [ss])
    return ax, B


def panel_c(fig, rect, vx):
    Cc = V.rd("fig5_c")
    ax = ps.axes_mm(fig, *rect)
    lay = [(t, n) for t, _ in TGT for n in (40, 160)]
    labels = [f"{dict(TGT)[t]}, n = {n}" for t, n in lay]
    ys = _frame(ax, labels, (-3.5, 3.0), (-3, -2, -1, 0, 1, 2, 3), "Near − far difference in Δ (cm)")
    y1, y2, v1, v2, s1, s2 = [], [], [], [], [], []
    for (t, n), y in zip(lay, ys):
        for which, dy, col, ls, yl, vl, sl in (("R1 − P1", OFF, ps.COLOR["residual"], "-", y1, v1, s1),
                                               ("P1 − P0", -OFF, ps.COLOR["refit"], DASH, y2, v2, s2)):
            r = Cc[(Cc.target == t) & (Cc.n == n) & (Cc.which == which)].iloc[0]
            _ci(ax, y + dy, r.ci_lo, r.ci_hi, col, ls)
            _pt(ax, r.delta, y + dy, col, TMARK[t])
            yl.append(y); vl.append(r.verdict); sl.append(r.small_effect)
    for k in (1, 3, 5):
        ax.axhline(ys[k] - 0.5, color=ps.GREY["light"], lw=0.5, zorder=0.2)
    _vcols(fig, ax, [vx, vx + 7.0], [y1, y2], [v1, v2], [s1, s2], heads=["R1 −\nP1", "P1 −\nP0"])
    return ax, Cc


def panel_d(fig, x_mm, y_mm, w_mm, h_mm, gap_mm):
    D = V.rd("fig5_d")
    D = D[D.n == 40]
    axes = []
    st = ["near", "mid", "far"]
    for k, (con, title, col) in enumerate((("R1-P1", "R1 − P1", ps.COLOR["residual"]), ("P1-P0", "P1 − P0", ps.COLOR["refit"]))):
        ax = ps.axes_mm(fig, x_mm + k * (w_mm + gap_mm), y_mm, w_mm, h_mm)
        axes.append(ax)
        for j, (t, _) in enumerate(TGT):
            q = D[(D.contrast == con) & (D.target == t)].set_index("stratum").loc[st]
            xs = np.arange(3) + (j - 1.5) * 0.16
            for x, lo, hi in zip(xs, q.ci_lo, q.ci_hi):
                ax.plot([x, x], [lo, hi], color=col, lw=ps.LW["ci"], zorder=2.4)
            ax.plot(xs, q.delta, ls="-", lw=0.5, color=col, alpha=0.6, zorder=2.5)
            ax.plot(xs, q.delta, ls="none", marker=TMARK[t], ms=ps.MS["main"], mfc=col, mec=col, zorder=3)
        ax.set_xlim(-0.6, 2.6); ax.set_xticks(range(3)); ax.set_xticklabels(["near", "mid", "far"])
        ax.set_ylim(-2.2, 6.2); ax.set_yticks([-2, 0, 2, 4, 6])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))
        ps.zero_line(ax, "y", better_text=False)
        ax.text(0.5, 1.02, f"{title}, n = 40", transform=ax.transAxes, ha="center", va="bottom", fontsize=ps.FS["base"])
        ax.set_xlabel("Distance to nearest label")
        if k == 0:
            ax.set_ylabel("ΔRMSE (cm)")
        else:
            ps.hide_ylabels(ax)
    return axes, D


def build():
    ps.use_paper()
    W, H = 180.0, 132.0
    fig = ps.paper_figure(W, H)
    ax_a, A = panel_a(fig, (38.0, 64.0, 48.0, 44.0), vx=[87.0, 95.0])
    ax_b, B = panel_b(fig, (133.0, 76.0, 38.0, 32.0), vx=171.5)
    ax_c, Cc = panel_c(fig, (26.0, 12.0, 46.0, 40.0), vx=72.5)
    axd, D = panel_d(fig, 104.0, 12.0, 32.0, 36.0, 6.0)
    ps.panel_label(ax_a, "a", dx_mm=-36.0, dy_mm=4.0)
    ps.panel_label(ax_b, "b", dx_mm=-29.0, dy_mm=4.0)
    ps.panel_label(ax_c, "c", dx_mm=-24.0, dy_mm=4.5)
    ps.panel_label(axd[0], "d", dx_mm=-12.0, dy_mm=4.5)
    handles = [Line2D([], [], color=ps.COLOR["residual"], ls="-", lw=ps.LW["ci_forest"], label="R1 − P1 (c, d)"),
               Line2D([], [], color=ps.COLOR["refit"], ls=DASH, lw=ps.LW["ci_forest"], label="P1 − P0 (c, d)")]
    handles += [Line2D([], [], ls="none", marker=TMARK[t], ms=ps.MS["main"] + 0.3, mfc="#4d4d4d", mec="#4d4d4d", label=lab) for t, lab in TGT]
    handles += [Line2D([], [], ls="none", marker="o", ms=2.2, mfc=ps.GREY["mid"], mec=ps.GREY["mid"], label="Region estimate (b)")]
    ps.legend_below(fig, handles, (4.0, H - 10.5, W - 8.0, 9.0), ncol=4)
    kax = ps.axes_mm(fig, 4.0, H - 16.5, 172.0, 4.5)
    kax.set_axis_off(); kax.set_xlim(0, 1); kax.set_ylim(0, 1)
    V.verdict_key(kax, 0.0, 0.5, include_small=True)
    kax.set_gid("verdict_key")
    T = V.rd("fig5_a_verdict")
    bk = V.rd("fig5_a_blocks")
    bk = bk[bk.method == "P1"].set_index(["n", "placement"]).n_blocks_lab_mean
    cap = dict(
        definition=("The L43 difference is the RMSE with labels drawn across A blocks (block-spread) minus the RMSE with the same number drawn at "
                    "random cells; negative favours spreading. L23 compares label sets with scored cells held fixed; L24 is the Δ in "
                    "the scored third nearest to a label minus Δ in the farthest third."),
        statistics=("a, two-stage bootstrap (scored blocks and draws resampled; 10,000 resamples) and the draw-conditional CI of the "
                    "paired contrast. b–d, block-bootstrap 95% CI (10,000 resamples). Symbols copy the four-way verdicts of the source "
                    "tables (both weightings, 0.5 cm margin); † marks |Δ| < 0.5 cm."),
        panels=("a, L43 with P1 at n = 10 (four regions) and n = 40 (Lena and Canada only); registered verdict: no placement effect "
                "was established. Labelled blocks per draw: "
                f"{bk[(10, 'cell')]:.1f} (cell-random) and {bk[(10, 'block')]:.1f} (block-spread) at n = 10, {bk[(40, 'cell')]:.1f} and "
                f"{bk[(40, 'block')]:.1f} at n = 40. b, L23 proximity contrasts, four-region means; registered verdict: the gain from random "
                "splitting contains a proximity component that does not depend on label count. c, L24 for the four eligible "
                "targets; registered verdicts: distance dependence was not established, and the recalibration gain did not depend "
                "on distance to the labels. d, stratum values at n = 40 (descriptive; n = 160 in Source Data)."),
        data=("LG shards (Rescale) aggregated by h39 (a) and h42 (b–d, local re-aggregation of Rescale shards). The design gives a "
              "method-level rule only; no site priority is implied."),
    )
    assert "배치 효과는 확인되지 않았다" in str(T.verdict.iloc[0])
    src = dict(a=A, a_blocks=V.rd("fig5_a_blocks"), b=B, c=Cc, d=D)
    spec_extra = dict(module="fig5_placement.py", panels=4, message="no placement effect (L43); proximity component (L23); no distance dependence (L24)")
    return V.save_v2(fig, NAME, cap, SPEC_ID, src, title="Label placement: block-spread versus cell-random draws, proximity and distance.",
                     spec_extra=spec_extra)


if __name__ == "__main__":
    build()
