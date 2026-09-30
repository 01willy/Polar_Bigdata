"""Fig 4 (v2). 최소 라벨 수 n* 와 대상별 개선·악화 분포. 스펙: figure_spec.json display_items 'Fig 4'(f3_v2).

패널
  a  대상별 n*(P1 − P0, R1 − P0, R1 − P1). 달성 = 채운 표지와 구간 막대 'n_lo < n* ≤ n_hi', 미달성 = n_max 에서 오른쪽 화살표('> n_max').
     대상: 독립 지역(주 4지역 x, Alaska x 참조), 하위 지역 10개(모드 i). point_only 대상은 없다(러시아 C·그린란드 제외).
  b  L4: R1 − P1 층화 평균(E1 = P4, n ≤ 10, + P4 전량 점; E2 = 레나·캐나다), R1 − P1*(n 40·160, 빈 사각형). 아래 기호 줄.
  c  L3: 적격 4대상의 R0 n*(α = 1 대 중첩 선택 α).
  d  NOVELTY N1: n 마다 대상별 4분 판정 수(P4, 하위 지역 i), 대비 R1 − P0, P1 − P0, R1 − P1.
자료: data/processed/paper_figs/fig4_*.csv(v2_data.py). 판정 기호·판정 수 = 원천 열(lgx_lg_aux, lgx_tests, n1_target_summary).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                       # noqa: E402
import matplotlib                                                        # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Rectangle                                 # noqa: E402

import v2_style as V                                                     # noqa: E402
from v2_style import ps, NPOS, N_ALL                                     # noqa: E402

NAME = "Fig4_min_labels"
SPEC_ID = "paper_fig4_min_labels"
SER = {  # contrast → (방법 색 키, 마커, y 오프셋, 표시 이름)
    "P1 − P0": ("refit", "o", 0.25, "n*, P1 − P0"),
    "R1 − P0": ("residual", "D", 0.0, "n*, R1 − P0"),
    "R1 − P1": ("residual", "s", -0.25, "n*, R1 − P1"),
}
ALPHA_ST = {"alpha1": dict(marker="o", mfc="#b0b0b0", mec="#000000", label="L3: α = 1", dy=0.22),
            "nested": dict(marker="D", mfc="#4d4d4d", mec="#000000", label="L3: nested α", dy=-0.22)}
FILL = [  # (열, 표시 이름, 면 색, 빗금)
    ("n_superior", "Lower error", "#1a1a1a", None),
    ("n_equivalent", "Equivalent", "#ececec", "...."),
    ("n_undecided", "Undecided", "#bdbdbd", None),
    ("n_inferior", "Higher error", "white", "//////"),
    ("n_na", "Not determinable", "white", "xxxx"),
]
ARROW = 0.55


def _nstar_row(ax, y, n_star, n_lo, cens, n_max, color, marker, ms=None):
    if not cens and np.isfinite(n_star):
        x1, x0 = NPOS[int(n_star)], NPOS[int(n_lo)]
        ax.plot([x0, x1], [y, y], color=color, lw=0.6, solid_capstyle="butt", zorder=2.5)       # 격자 구간(CI 아님): 가는 선 + 왼쪽 끝 눈금
        ax.plot([x0], [y], ls="none", marker="|", ms=2.6, mec=color, mew=0.8, zorder=2.6)
        ax.plot([x1], [y], ls="none", marker=marker, ms=ms or ps.MS["main"] + 0.3, mfc=color, mec=color, mew=ps.LW["marker_edge"], zorder=3)
    else:
        x0 = NPOS[int(n_max)]
        ax.plot([x0], [y], ls="none", marker="|", ms=3.2, mec=color, mew=0.9, zorder=3)
        ax.annotate("", xy=(x0 + ARROW, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color, mutation_scale=5.5,
                                                                             shrinkA=0, shrinkB=0), zorder=3)


def _nstar_axis(ax, xlabel="Minimum target labels, n*"):
    ax.set_xlim(-0.5, 6.75)
    ax.set_xticks([NPOS[n] for n in V.N_GRID])
    ax.set_xticklabels([V.N_TICK[n] for n in V.N_GRID])
    ax.set_xlabel(xlabel)
    for x in range(len(V.N_GRID)):
        ax.axvline(x, color=ps.GREY["band"], lw=0.5, zorder=0.1)


def panel_a(fig, rect):
    A = V.rd("fig4_a")
    ax = ps.axes_mm(fig, *rect)
    order = [("h", "Regions (mode x)")] + [("t", t, "x") for t in ["Lena", "Canada", "Russia_W", "Russia_E", "Alaska"]]
    order += [("h", "Sub-regions (mode i)")] + [("t", t, "i") for t in ["AL-1", "AL-2", "AL-3", "AL-4", "AL-5", "AL-6", "CA-2", "CA-3", "LE-1", "LE-2"]]
    n = len(order)
    ys = np.arange(n)[::-1].astype(float)
    labels = []
    for o, y in zip(order, ys):
        if o[0] == "h":
            labels.append(o[1]); continue
        t, md = o[1], o[2]
        labels.append(ps.region_name(t) + (" (reference)" if t == "Alaska" else ""))
        if len([l for l in labels if l]) % 2 == 0:                    # 대상 행 묶음 음영(세 계열이 한 대상에 속함을 보인다)
            ax.axhspan(y - 0.46, y + 0.46, facecolor=ps.GREY["band"], edgecolor="none", zorder=0.05)
        for con, (mk_c, mk, dy, _) in SER.items():
            q = A[(A.target == t) & (A["mode"] == md) & (A.contrast == con)]
            if not len(q):
                continue
            r = q.iloc[0]
            _nstar_row(ax, y + dy, r.n_star, r.n_lo, bool(r.censored), r.n_max, ps.COLOR[mk_c], mk)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    for tl, o in zip(ax.get_yticklabels(), order):
        if o[0] == "h":
            tl.set_fontweight("bold")
    _nstar_axis(ax)
    ax.text(1.0, 1.005, "bar: grid interval n_lo < n* ≤ n_hi (not a CI); arrow: not reached", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    return ax, A


def panel_b(fig, rect, strip_rect):
    B = V.rd("fig4_b")
    PS1 = V.rd("fig4_b_p1star")
    VB = V.rd("fig4_b_verdicts")
    ax = ps.axes_mm(fig, *rect)
    c = ps.COLOR["residual"]
    V.linear_y(ax, lim=(-1.7, 2.3), ticks=(-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0 if float(v).is_integer() else 1)))
    ax.axhspan(-0.5, 0.5, facecolor=ps.GREY["band"], edgecolor="none", zorder=0.1)
    ps.zero_line(ax, "y", better_text=False)
    for ed, ls, dx in (("E1_P4_n_le_10", "-", -0.09), ("E2_LenaCanada_all_n", ps.VARIANT_LS["dashed"], 0.09)):
        g = B[(B.edition == ed) & (B.n != N_ALL)].sort_values("n")
        xs = np.array([NPOS[int(n)] for n in g.n]) + dx
        for x, lo, hi in zip(xs, g.ci_lo, g.ci_hi):
            ax.plot([x, x], [lo, hi], color=c, lw=ps.LW["ci"], ls=ls, zorder=2.4)
        ax.plot(xs, g.delta, color=c, ls=ls, lw=ps.LW["main"], marker="D", ms=ps.MS["main"], mfc=c, mec=c, zorder=3)
        al = B[(B.edition == ed) & (B.n == N_ALL)]
        if ed.startswith("E1"):
            al = B[B.edition.str.startswith("P4_all")]
        for r in al.itertuples():
            x = NPOS[N_ALL] + dx
            ax.plot([x, x], [r.ci_lo, r.ci_hi], color=c, lw=ps.LW["ci"], ls=ls, zorder=2.4)
            ax.plot([x], [r.delta], ls="none", marker="D", ms=ps.MS["main"], mfc=c, mec=c, zorder=3)
    for r in PS1.itertuples():
        x = NPOS[int(r.n)] + 0.27
        ax.plot([x, x], [r.ci_lo, r.ci_hi], color=c, lw=ps.LW["ci"], zorder=3.3)
        ax.plot([x], [r.delta], ls="none", marker="s", ms=ps.MS["main"] + 0.4, mfc="white", mec=c, mew=0.9, zorder=3.5)
    V.n_axis(ax)
    ax.set_ylabel("ΔRMSE, R1 − P1 or P1* (cm)")
    V.better_note(ax, "negative = lower error than the re-fitted physics", loc=(0.01, 0.03), ha="left", va="bottom")
    ax.text(0.99, 0.97, "grey band ±0.5 cm", transform=ax.transAxes, ha="right", va="top", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    # 기호 줄(열 = b 의 n 위치, 판정은 원천 열 그대로)
    st = ps.axes_mm(fig, *strip_rect)
    cols = [3, 10, 40, 160, 320, 1000, N_ALL]
    cells = {(r.row, int(r.n)): (r.verdict, bool(r.small_effect)) for r in VB.itertuples()}
    for n in (3, 320, 1000):                                          # P1* 는 n ∈ {10, 40, 160, all} 에서만 정했다(LG 7.1a 보충)
        cells.setdefault(("R1-P1*", n), ("not computed", False))
    pools = {int(r.n): V.pool_code(r.pool) for r in VB.itertuples() if V.pool_code(r.pool)}
    V.verdict_table(st, ["R1-P1", "R1-P1*"], cols, cells, ["R1 − P1  (L4)", "R1 − P1*  (L29)"], [""] * len(cols),
                    sub_labels=[pools.get(c_, "") for c_ in cols], col_x=[NPOS[c_] for c_ in cols], xlim=ax.get_xlim(), sub_title="pool")
    ABc = V.rd("ab_cells")
    for r in ABc[ABc.figure == "fig4b"].itertuples():                    # AB5(n 10), AB9(전량): P4 점 위(E1 = 실선 계열)
        q = B[(B.n == int(r.n)) & (B.edition.str.startswith("E1") if int(r.n) != N_ALL else B.edition.str.startswith("P4_all"))]
        y = float(q.ci_hi.iloc[0])
        if int(r.n) == N_ALL:
            V.ab_tag(ax, NPOS[int(r.n)] - 0.09, float(q.delta.iloc[0]), r.ab, dx_pt=-5.0, dy_pt=0.0, ha="right")
        else:
            V.ab_tag(ax, NPOS[int(r.n)] - 0.09, y, r.ab, dx_pt=0.0, dy_pt=5.0, ha="center")
    st.set_gid("verdict_table")
    return ax, B, PS1, VB


def panel_c(fig, rect):
    Cq = V.rd("fig4_c")
    ax = ps.axes_mm(fig, *rect)
    tg = [("Lena", "x"), ("Canada", "x"), ("AL-2", "i"), ("AL-5", "i")]
    ys = np.arange(len(tg))[::-1].astype(float)
    for (t, md), y in zip(tg, ys):
        pts = []
        for cond, st in ALPHA_ST.items():
            r = Cq[(Cq.target == t) & (Cq["mode"] == md) & (Cq.cond == cond)].iloc[0]
            yy = y + st["dy"]
            if not bool(r.censored):
                ax.plot([NPOS[int(r.n_star)]], [yy], ls="none", marker=st["marker"], ms=ps.MS["main"] + 0.4, mfc=st["mfc"], mec=st["mec"],
                        mew=0.6, zorder=3)
                pts.append(NPOS[int(r.n_star)])
            else:
                _nstar_row(ax, yy, np.nan, np.nan, True, r.n_max, "#4d4d4d" if cond == "nested" else "#808080", st["marker"])
        if len(pts) == 2:                                                 # 같은 대상의 두 조건을 잇는 선(덤벨)
            ax.plot(pts, [y + ALPHA_ST["alpha1"]["dy"], y + ALPHA_ST["nested"]["dy"]], color=ps.GREY["mid"], lw=0.5, zorder=2.5)
    ax.set_ylim(-0.6, len(tg) - 0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{ps.region_name(t)} ({md})" for t, md in tg])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    _nstar_axis(ax, xlabel="n* of R0 (L3)")
    ax.tick_params(axis="x", labelsize=ps.MIN_FONT_PT)
    return ax, Cq


def panel_c_aux(fig, rect):
    """c 오른쪽: L3 4분 보조(lgw_aux4). 열 = (R0@nested − R0@α1, R0@nested − P0) × n ∈ {10, 40}."""
    X = V.rd("fig4_c_aux4")
    tg = [("Lena", "x"), ("Canada", "x"), ("AL-2", "i"), ("AL-5", "i")]
    cols = [("nested − α1", 10), ("nested − α1", 40), ("nested − P0", 10), ("nested − P0", 40)]
    cells = {}
    for r in X.itertuples():
        cells[(r.target, (r.kind, int(r.n)))] = (r.verdict, bool(r.small_effect))
    ax = ps.axes_mm(fig, *rect)
    rows = [f"{t}|{m}" for t, m in tg]
    V.verdict_table(ax, rows, cols, cells, [""] * len(rows), ["vs α1", "", "vs P0", ""], sub_labels=["10", "40", "10", "40"],
                    col_x=[0, 1, 2.2, 3.2], xlim=(-0.55, 3.75), sub_title="n")
    ax.set_yticklabels([])
    ax.axvline(1.6, color=ps.GREY["light"], lw=0.5)
    ax.set_gid("verdict_table_L3aux")
    return ax, X


def panel_d(fig, x0s, y_rows, w, h):
    D = V.rd("fig4_d")
    groups = [("P4", "P4 regions", 4), ("sub_i", "Sub-regions (i)", 10)]
    cons = ["R1 − P0", "P1 − P0", "R1 − P1"]
    axes = []
    ns = V.N_GRID + [N_ALL]
    for gi, (g, gname, gsize) in enumerate(groups):
        for ci, con in enumerate(cons):
            ax = ps.axes_mm(fig, x0s[ci], y_rows[gi], w, h)
            axes.append(ax)
            q = D[(D.group == g) & (D.contrast == con)]
            for n in ns:
                r = q[q.n == n]
                if not len(r):
                    continue
                r = r.iloc[0]
                bottom = 0.0
                for col, _, fc, hatch in FILL:
                    v = float(r[col])
                    if v > 0:
                        ax.bar(NPOS[n], v, bottom=bottom, width=0.72, facecolor=fc, edgecolor="#000000", lw=0.4, hatch=hatch, zorder=2)
                        bottom += v
            ax.set_ylim(0, gsize)
            ax.set_yticks([0, gsize // 2, gsize] if gsize > 4 else [0, 2, 4])
            V.n_axis(ax, xlabel="Target labels, n" if gi == 1 else None, labels=(gi == 1), lim=(-0.6, V.X_ALL + 0.6))
            ax.tick_params(axis="x", labelsize=ps.MIN_FONT_PT)
            if ci == 0:
                ax.set_ylabel(f"{gname}\n(targets)", linespacing=1.0)
            else:
                ps.hide_ylabels(ax)
            if gi == 0:
                ax.text(0.5, 1.04, con, transform=ax.transAxes, ha="center", va="bottom", fontsize=ps.FS["base"])
    return axes, D


def _fill_key(fig, rect):
    ax = ps.axes_mm(fig, *rect)
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    x = 0.0
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    for _, lab, fc, hatch in FILL:
        ax.add_patch(Rectangle((x, 0.2), 0.018, 0.6, facecolor=fc, edgecolor="#000000", lw=0.4, hatch=hatch, transform=ax.transAxes))
        t = ax.text(x + 0.025, 0.5, lab, transform=ax.transAxes, ha="left", va="center", fontsize=ps.FS["annot"])
        x = inv.transform((t.get_window_extent(rend).x1, 0))[0] + 0.03
    t = ax.text(x, 0.5, "blank = n above the labelled pool", transform=ax.transAxes, ha="left", va="center", fontsize=ps.FS["annot"],
                color=ps.GREY["text2"])
    ax.set_gid("fill_key")
    return ax


def build(draft: bool = False):
    ps.use_paper()
    W, H = 180.0, 180.0
    fig = ps.paper_figure(W, H)
    ax_a, A = panel_a(fig, (30.0, 66.0, 54.0, 94.0))
    ax_b, B, PS1, VB = panel_b(fig, (106.0, 128.0, 70.0, 32.0), (106.0, 105.5, 70.0, 9.5))
    ABc = V.rd("ab_cells")
    ABc = ABc[ABc.figure == "fig4b"]
    ab_items = [(r.ab, V.AB_RULE_TEXT[r.rule]) for r in ABc.drop_duplicates("ab").itertuples()]
    V.verdict_key_grid(fig, (96.0, 90.6, 83.0, 13.5), ncol=2, col_w=[0.42, 0.58],
                       extra=[("n.c.", "P1* not computed at this n")] + ab_items, text_dx_mm=5.5)
    ax_c, Cq = panel_c(fig, (116.0, 64.0, 43.0, 18.0))
    ax_c2, X3 = panel_c_aux(fig, (161.5, 64.0, 17.0, 24.0))
    axd, D = panel_d(fig, [22.0, 75.0, 128.0], [31.0, 11.0], 48.0, 14.0)
    _fill_key(fig, (22.0, 51.5, 154.0, 4.0))
    ps.panel_label(ax_a, "a", dx_mm=-28.0, dy_mm=1.2)
    ps.panel_label(ax_b, "b", dx_mm=-13.0, dy_mm=1.2)
    ps.panel_label(ax_c, "c", dx_mm=-25.0, dy_mm=3.0)
    ps.panel_label(axd[0], "d", dx_mm=-20.0, dy_mm=5.0)
    handles = [Line2D([], [], color=ps.COLOR[mc], lw=0.9, marker=mk, ms=ps.MS["main"] + 0.3, mfc=ps.COLOR[mc], mec=ps.COLOR[mc], label=lab)
               for mc, mk, _, lab in SER.values()]
    c = ps.COLOR["residual"]
    handles += [Line2D([], [], color=c, ls="-", lw=ps.LW["main"], marker="D", ms=ps.MS["main"], mfc=c, mec=c, label="R1 − P1, P4 pool"),
                Line2D([], [], color=c, ls=ps.VARIANT_LS["dashed"], lw=ps.LW["main"], marker="D", ms=ps.MS["main"], mfc=c, mec=c,
                       label="R1 − P1, Lena + Canada pool"),
                Line2D([], [], ls="none", marker="s", ms=ps.MS["main"] + 0.4, mfc="white", mec=c, mew=0.9, label="R1 − P1* (P1* = edaphic re-fit)")]
    handles += [Line2D([], [], ls="none", marker=s["marker"], ms=ps.MS["main"] + 0.4, mfc=s["mfc"], mec=s["mec"], mew=0.6, label=s["label"])
                for s in ALPHA_ST.values()]
    ps.legend_below(fig, handles, (4.0, H - 14.0, W - 8.0, 12.0), ncol=4)

    cap = dict(
        definition=("n* is the smallest grid n at which the method has lower error than its baseline under three registered conditions "
                    "(block-CI upper bound < 0, split win rate ≥ 2/3, repeat win rate ≥ 0.75). It is reported as an interval "
                    "n_lo < n* ≤ n_hi; an arrow means not reached up to the largest tested n (no imputed value). n* is registered "
                    "against P0 and P1 only; no n* was computed against P* or P1*."),
        statistics=("a, c, block-bootstrap CIs with 1,000 resamples. b, fixed-composition stratified means with cell-weighted 95% CI "
                    "(10,000 resamples). Verdicts use both cell-weighted and block-equal-weighted CIs with a 0.5 cm margin: symbols in b "
                    "are copied from the LGX tables (10,000 resamples); counts in d apply the same rule to the per-target LG curve CIs "
                    "(1,000 resamples). † marks |Δ| < 0.5 cm."),
        panels=("a, n* by target; sub-regions share cells or sources with their parent region and are not independent. b, R1 − P1 "
                "(L4, registered verdict: net value of ML at sparse labels, minimum n = 10 for the four-region mean) and R1 − P1* at "
                "n = 40 and 160, where P1* (edaphic Stefan re-fit) outperformed P1 (L29); the pool row gives the regions behind each "
                "verdict. c, L3 targets, R0 with α = 1 versus nested α (registered verdict: supported; nested α reduced n* for no "
                "target). Right of c, L3 auxiliary four-way verdicts (R0 with nested α against α = 1 and against P0, n = 10 and 40). "
                "d, number of targets per verdict at each n; sub-regions are counted, not averaged. AB5 and AB9 are abstract "
                "contrasts (Holm over ten); n.c., P1* not determined at that n."),
        data=("LG and LGX runs on the Rescale platform; LGX tables and the pooled curves are local re-aggregations of the same shards "
              "(cross-environment check not performed). The n = 80 grid point (C12) was not run."),
    )
    src = dict(a=A, b=B, b_p1star=PS1, b_verdicts=VB, c=Cq, c_L3aux=X3, d=D, ab_tags=ABc)
    spec_extra = dict(module="fig4_min_labels.py", panels=4, message="n* reported as intervals; R1 − P1* undecided at n = 40, 160",
                      part2="AB5, AB9 tags; L3 four-way auxiliary grid (lgw_aux4); n.c. where P1* was not computed")
    return V.save_v2(fig, NAME, cap, SPEC_ID, src, title="Minimum number of target labels and the distribution of improvement and "
                     "deterioration across targets.", spec_extra=spec_extra)


if __name__ == "__main__":
    build()
