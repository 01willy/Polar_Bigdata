"""Fig 2 (v2). 라벨 수 곡선 Δ(n), 원천 계수 물리식 P0 대비. 스펙: figures/figure_spec.json display_items 'Fig 2'(f3_v2 블록).

패널
  a–d  지역 곡선(레나, 캐나다, 러시아 W, 러시아 E, 모드 x): P1, R0, R1, R2, D0. 띠 = P1·R1 의 셀 가중 CI(1,000회).
       P* 가로 점선(P0@tddm − P0, 지역 값), P1* 빈 사각형(P1@ed − P0, n 40·160, 레나·캐나다).
  e    풀 판 E1(P4, n ≤ 10). P* 선 = lgx_tests L29 MEAN.
  f    풀 판 E2(레나·캐나다, 공통 n). P* 선·P1* 점 = 지역 값 평균(점 추정).
  오른쪽 기호 표: R1 − P1(L4), R1 − P1*(L29), R2 − R1(L2), R1 − P0(L8), R1 − P*(L29). 원천 표의 verdict4 그대로.
자료: data/processed/paper_figs/fig2_*.csv(v2_data.py). 이 모듈은 계산하지 않는다.
실행: CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/4_visualization/paper/fig2_label_curve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                       # noqa: E402
import textwrap                                                          # noqa: E402

import v2_style as V                                                     # noqa: E402
from v2_style import ps, NPOS, N_ALL                                     # noqa: E402

NAME = "Fig2_label_curve"
SPEC_ID = "paper_fig2_label_curve"
METHODS = ["P1", "R0", "R1", "R2", "D0"]
DODGE = {"P1": -0.24, "R0": -0.12, "R1": 0.0, "R2": 0.12, "D0": 0.24}
P1S_DX = -0.36                              # a–d 는 왼쪽(D0 선 회피), f 는 오른쪽(부호 반대)
BAND = {"P1", "R1"}
REGIONS = [("Lena", "a"), ("Canada", "b"), ("Russia_W", "c"), ("Russia_E", "d")]
Y_REG = dict(lim=(-17, 15.0), ticks=(-10, -5, -2, -1, 0, 1, 2, 5), linscale=0.7)
Y_STRIP, Y_SEP = 10.0, 6.4                 # a–d 위쪽 L1 기호 줄(D0 − P0)과 구분선
Y_POOL = dict(lim=(-3.8, 3.8), ticks=(-3, -2, -1, 0, 1, 2, 3))
YLAB = "ΔRMSE vs P0 (cm)"


def _curve(ax, q, key, band: bool):
    """한 방법 곡선. 격자 n(0–1,000)은 선으로 잇고 'all' 은 떨어진 점 + CI 막대."""
    s = V.mstyle(key)
    dx = DODGE[key]
    g = q[q.n != N_ALL].sort_values("n")
    xa = np.array([NPOS[int(n)] for n in g.n]) + dx
    if band and len(g) > 1:
        ax.fill_between(xa, g.lo, g.hi, color=s["color"], alpha=ps.BAND_ALPHA, lw=0, zorder=1.8)
    ax.plot(xa, g.d, **{k: v for k, v in s.items()}, zorder=3)
    al = q[q.n == N_ALL]
    for _, r in al.iterrows():
        x = NPOS[N_ALL] + dx
        if band:
            ax.plot([x, x], [r.lo, r.hi], color=s["color"], lw=ps.LW["ci"], solid_capstyle="round", zorder=2.5)
        ax.plot([x], [r.d], ls="none", marker=s["marker"], ms=s["ms"], mfc=s["mfc"], mec=s["mec"], mew=s["mew"], zorder=3)


def _p1star(ax, xs, ys, lo=None, hi=None):
    c = ps.COLOR["refit"]
    for i, (x, y) in enumerate(zip(xs, ys)):
        if lo is not None and np.isfinite(lo[i]):
            ax.plot([x, x], [lo[i], hi[i]], color=c, lw=ps.LW["ci"], zorder=3.4)
        ax.plot([x], [y], ls="none", marker="s", ms=ps.MS["main"] + 0.4, mfc="white", mec=c, mew=0.9, zorder=3.5)


def _header(ax, text):
    t = ax.text(0.0, 1.035, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["base"])
    t.set_gid("panel_header")
    return t


def build(draft: bool = False):
    ps.use_paper()
    R = V.rd("fig2_region_curves").rename(columns={"d_p0": "d", "d_p0_lo": "lo", "d_p0_hi": "hi"})
    PS = V.rd("fig2_pstar_region")
    P1S = V.rd("fig2_p1star_region")
    PL = V.rd("fig2_pool_curves").rename(columns={"delta": "d", "ci_lo": "lo", "ci_hi": "hi"})
    PSP = V.rd("fig2_pstar_pool")
    P1SP = V.rd("fig2_p1star_pool")
    VM = V.rd("fig2_verdicts")
    L1 = V.rd("fig2_L1_region")

    W, H = 180.0, 182.0
    fig = ps.paper_figure(W, H)
    rows_y = {1: 123.0, 2: 75.0, 3: 18.0}
    AH = 34.0
    ax_pos = {"a": (14.0, rows_y[1]), "b": (97.0, rows_y[1]), "c": (14.0, rows_y[2]), "d": (97.0, rows_y[2])}
    AW = 72.0
    axes = {}
    for reg, letter in REGIONS:
        x, y = ax_pos[letter]
        ax = ps.axes_mm(fig, x, y, AW, AH)
        axes[letter] = ax
        q = R[R.target == reg]
        V.symlog_y(ax, **Y_REG)
        ps.zero_line(ax, "y", better_text=False)
        if letter == "a":
            V.better_note(ax, "negative = lower error than P0", loc=(0.99, 0.03), va="bottom")
        # L1 기호 줄: D0 − P0 의 지역 4분 판정(lgx_lg_aux, 원천 열 그대로)
        ax.axhline(Y_SEP, color=ps.GREY["light"], lw=0.5, zorder=0.4)
        lq = L1[L1.target == f"{reg}|x"]
        for r in lq.itertuples():
            if r.verdict != "no row":
                V.draw_verdict(ax, NPOS[int(r.n)], Y_STRIP, r.verdict, False)
        xl = NPOS[40] + 0.45 if len(lq[lq.verdict != "no row"]) > 3 else NPOS[10] + 0.45
        ax.text(xl, Y_STRIP, "D0 − P0 (L1)", ha="left", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"], zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        pv = float(PS[PS.target == reg].pstar_d.iloc[0])
        ln = ax.axhline(pv, **V.PSTAR_STYLE, zorder=1.7); ln.set_gid("pstar")
        for m in METHODS:
            _curve(ax, q[q.method == m], m, m in BAND)
        ps1 = P1S[P1S.target == reg]
        if len(ps1):
            _p1star(ax, [NPOS[int(n)] + P1S_DX for n in ps1.n], ps1.d_p0.to_numpy(), ps1.d_p0_lo.to_numpy(), ps1.d_p0_hi.to_numpy())
        if reg.startswith("Russia"):
            V.unavailable_band(ax, 2.55, V.X_BREAK - 0.12, "n ≥ 40 exceeds the labelled pool", y_text=0.22)
        V.n_axis(ax, xlabel="Target labels, n" if letter in "cd" else None)
        nl = q[q.n == N_ALL].n_lab.mean()
        _header(ax, f"{ps.region_name(reg)}  (all = {nl:,.0f} labels)")
        if letter in "bd":
            ps.hide_ylabels(ax)
        else:
            ax.set_ylabel(YLAB)

    # e, f: 풀 판
    ex, fx = 14.0, 44.0
    ae = ps.axes_mm(fig, ex, rows_y[3], 22.0, AH)
    af = ps.axes_mm(fig, fx, rows_y[3], 62.0, AH)
    for ax, ed, letter in ((ae, "E1_P4_n_le_10", "e"), (af, "E2_LenaCanada_all_n", "f")):
        axes[letter] = ax
        q = PL[PL.edition == ed]
        V.linear_y(ax, **Y_POOL)
        ps.zero_line(ax, "y", better_text=False)
        pv = float(PSP[PSP.edition == ed].pstar.iloc[0])
        ln = ax.axhline(pv, **V.PSTAR_STYLE, zorder=1.7); ln.set_gid("pstar")
        for m in METHODS:
            _curve(ax, q[q.method == m], m, m in BAND)
        if ed.startswith("E2"):
            _p1star(ax, [NPOS[int(n)] - P1S_DX for n in P1SP.n], P1SP.p1star.to_numpy())
            V.n_axis(ax, ns=[0, 3, 10, 40, 160, 320, N_ALL])               # E2 에는 n = 1,000 이 없다(레나만)
            _header(ax, "Lena + Canada pool, all common n")
            ps.hide_ylabels(ax)
        else:
            V.n_axis(ax, ns=[0, 3, 10], lim=(-0.5, 2.5))
            _header(ax, "P4 pool, n ≤ 10")
            ax.set_ylabel(YLAB)

    # 오른쪽 기호 표(판정은 원천 열 그대로)
    cols = [3, 10, 40, 160, 320, 1000, N_ALL]
    rows = [("R1-P1", "R1 − P1  (L4)"), ("R1-P1*", "R1 − P1*  (L29)"), ("R2-R1", "R2 − R1  (L2)"), ("R1-P0", "R1 − P0  (L8)"),
            ("R1-P*", "R1 − P*  (L29)")]
    cells = {(r.row, int(r.n)): (r.verdict, bool(r.small_effect)) for r in VM.itertuples()}
    for n in (3, 320, 1000):                                          # P1* 는 n ∈ {10, 40, 160, all} 에서만 정했다(LG 7.1a 보충)
        cells.setdefault(("R1-P1*", n), ("not computed", False))
    ABc = V.rd("ab_cells")
    ABc = ABc[ABc.figure == "fig2"]
    tags = {(r.key, int(r.n)): r.ab for r in ABc.itertuples()}
    pools = {}
    for r in VM.itertuples():
        pc = V.pool_code(r.pool)
        if pc:
            assert pools.get(int(r.n), pc) == pc, f"pool mismatch at n={r.n}"
            pools[int(r.n)] = pc
    at = ps.axes_mm(fig, 132.0, 24.5, 46.0, 36.0)
    V.verdict_table(at, [r for r, _ in rows], cols, cells, [lab for _, lab in rows], [V.N_TICK[c] for c in cols],
                    sub_labels=[pools.get(c, "") for c in cols], head_title="n", sub_title="pool", tags=tags)
    at.set_gid("verdict_table")
    V.verdict_key_grid(fig, (110.0, 11.0, 68.0, 7.0), ncol=2, extra=[("n.c.", "P1* not computed at this n")], text_dx_mm=5.5,
                       verdicts=False, col_w=[0.42, 0.58])
    kax = ps.axes_mm(fig, 4.0, H - 17.5, 172.0, 4.5)                     # 판정 기호 설명은 위(첫 사용 패널 a–d 가까이)
    kax.set_axis_off(); kax.set_xlim(0, 1); kax.set_ylim(0, 1)
    V.verdict_key(kax, 0.0, 0.5, include_small=True)
    kax.set_gid("verdict_key")
    note = "AB tags (abstract contrasts, Holm m = 10): " + V.ab_note(ABc, list(dict.fromkeys(ABc.ab)))
    nt = fig.text(110.0 / W, 9.6 / H, "\n".join(textwrap.wrap(note, 60)), ha="left", va="top", fontsize=ps.FS["annot"],
                  color=ps.GREY["text2"], linespacing=1.1)
    nt.set_gid("ab_note")

    for letter in "abcdef":
        ps.panel_label(axes[letter], letter, dx_mm=-12.0 if letter in "ace" else -4.0, dy_mm=1.2)

    handles = [V.method_handle(m) for m in METHODS] + [V.pstar_handle(), V.p1star_handle()]
    ps.legend_below(fig, handles, (4.0, H - 12.0, W - 8.0, 10.0), ncol=4)

    cap = dict(
        definition=("ΔRMSE = RMSE(method) − RMSE(P0) on the scored B half of each target; negative is lower error. P0 applies the source "
                    "least-squares Stefan coefficient E0; P1 re-fits E on the n labels with κ = 10 shrinkage; R0 and R1 add residual CatBoost "
                    "(λ = 0.25) to P0 and P1; R2 adds physics pseudo-label rows to R1; D0 is direct CatBoost without a physics anchor. P* and "
                    "P1* are the stronger physics baselines found under L29."),
        statistics=("a–d, cell-weighted means over five splits with block-bootstrap 95% CI (1,000 resamples; bands for P1 and R1 only). "
                    "e, f, fixed-composition stratified means (10,000 resamples). The a–d axes are linear within ±2 cm and logarithmic beyond. Verdict symbols are copied from the LGX tables (10,000 "
                    "resamples): both "
                    "cell-weighted and block-equal-weighted CIs, equivalence margin 0.5 cm; † marks |Δ| < 0.5 cm."),
        panels=("a–d, Lena, Canada, Russia W and Russia E (source excludes the target; mode x); hatched, n larger than the labelled pool. "
                "e, four-region pool P4 (n ≤ 10). f, Lena and Canada pool at every n present in both (n = 1,000 exists for Lena only). "
                "Dotted line, P* − P0; open squares, P1* − P0 at n = 40 and 160. In f both are point estimates (means of region values). "
                "Table right of f, registered pooled contrasts; its pool row gives the regions behind each column (L+C, Lena and Canada). "
                "Symbols above the grey line in a–d, region-level D0 − P0 verdicts for L1 (registered verdict: supported). "
                "AB tags mark cells that are abstract contrasts (lgw_bundle, same verdicts); n.c., P1* not determined at that n."),
        data=("All values come from LG and LGX runs on the Rescale platform; the LGX tables are a local re-aggregation of the same shards "
              "(cross-environment check not performed)."),
    )
    src = dict(a_d=R, a_d_pstar=PS, a_d_p1star=P1S, a_d_L1_verdicts=L1, e_f=PL, e_f_pstar=PSP, f_p1star=P1SP, verdicts=VM)
    spec_extra = dict(module="fig2_label_curve.py", panels=6, message="Δ(n) vs P0 with P* and P1* shown; verdicts from lgx tables only",
                      part2="AB5, AB6, AB8, AB9 tags from lgw_bundle; n.c. where P1* was not computed",
                      pending=["curve additions for L34(a), LGF-F1, LGF-N1 not evaluated (LGF window)"])
    return V.save_v2(fig, NAME, cap, SPEC_ID, src, title="Error change with the number of target labels, relative to the "
                     "source-coefficient Stefan model.", spec_extra=spec_extra)


if __name__ == "__main__":
    build()
