"""Fig 7 (v2). 배포 절차와 초록 주 대비 묶음. 스펙: figure_spec.json display_items 'Fig 7' f3_v2.

패널
  a  배포 절차 도식 T0 → T3 → T10 → T40(→ T160 보조). 판정 표지 문구는 lgw_tests 의 verdict 열에서 규칙으로 만든다
     (판정 이름 사전 VERDICT_EN 한 곳, 숫자는 정규식으로 추출). 손으로 쓴 판정 문구는 없다.
  b  시나리오 칸 행렬(독립 지역 5곳 × 단계·대비): verdict4 기호, 비열등(noninf) 칸 테두리, 러시아 W·E 의 T40·T160 빗금.
     아래 줄 = 주 4지역 층화 평균(main4_verdict4, 풀 표기).
  c  SC3w: 라벨 비 대 Δ_SC3(3지역 평균 CI, 지역 점), τ ∈ {0.025, 0.05, 0.10, 0.20}. 기준선 0.3·0.5 cm, 라벨 비 0.5.
  d  AB1–AB9(RMSE 차, cm, P4 층화 평균, 셀 가중 CI, 지역 점 추정) 와 AB10(구간 점수 차, cm, 별도 축).
     오른쪽 = 4분 판정 기호(보정 전, 두 가중) 와 초록 문구 규칙(Holm m = 10, lgw_bundle abstract_rule).
자료: data/processed/paper_figs/fig7_*.csv, ab_bundle*.csv, fig2_verdicts.csv(R1 − P1* 병기)(v2_data.py).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                       # noqa: E402
import matplotlib                                                        # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch  # noqa: E402

import v2_style as V                                                     # noqa: E402
from v2_style import ps                                                  # noqa: E402

NAME = "Fig7_deployment"
SPEC_ID = "paper_fig7_deployment"
VERDICT_EN = {"안전성 미확인": "safety not confirmed", "기각": "rejected", "문장 A": "sentence A", "문장 B": "sentence B",
              "규칙 비작동": "rule inactive", "배치 효과는 확인되지 않았다": "no placement effect established", "지지": "supported"}
REG = [("Lena", "Lena"), ("Canada", "Canada"), ("Russia_W", "Russia W"), ("Russia_E", "Russia E"), ("Alaska", "Alaska (x)")]
COLS = [("T0", "R0", "P0"), ("T3", "P1", "P0"), ("T3", "R1", "P0"), ("T10", "P1", "P0"), ("T10", "R1", "P0"), ("T40", "P1", "P0"),
        ("T40", "R1", "P0"), ("T40", "R1", "P1"), ("T160", "P1", "P0"), ("T160", "R1", "P0"), ("T160", "R1", "P1")]
C3 = {"Lena": "D", "Canada": "s", "Alaska": "o"}
AB_LABEL = {"AB1": "D0 − P0, n = 0 (L1)", "AB2": "B:ens − P0, n = 0 (L29)", "AB3": "D1 − D1@shuffle, n = 0 (L15)",
            "AB4": "P1 − P0, n = 10", "AB5": "R1 − P1, n = 10 (L4)", "AB6": "R2 − R1, n = 10 (L2)", "AB7": "R1 − F1k, n = 10 (L10)",
            "AB8": "R1 − P0, all labels (L8)", "AB9": "R1 − P1, all labels (L4)", "AB10": "interval score, (iii) − B4, n = 10 (LGU-A1)"}


# ---------------------------------------------------------------- 판정 문구(표에서 생성)
def _head(v: str) -> str:
    for k, e in VERDICT_EN.items():
        if str(v).startswith(k):
            return e
    raise KeyError(v)


def sc_texts(T, V2) -> dict:
    """판정 행 → 영문 표지(줄 목록). 판정 이름은 VERDICT_EN, 숫자는 표 글자에서 정규식으로 뽑는다."""
    g = {r.test_id: r for r in T.itertuples()}
    out = {}
    v = g["SC1w"].verdict
    k = re.search(r"비열등 칸 (\d+)/(\d+)", v)
    no_worse = "열세 판정 칸은 없었" in v
    out["SC1w"] = [f"SC1w, R1 − P0: {_head(v)}",
                   f"non-inferior cells {k.group(1)}/{k.group(2)}{'; no higher-error cell' if no_worse else ''}"]
    v = g["SC1w-P"].verdict
    cells = re.findall(r"(\w+) n = (\d+)\(Δ ([+\-−][\d.]+)", v)
    reg = sorted({c[0] for c in cells})
    ns = "; ".join(f"n = {c[1]}: {c[2].replace('-', '−')} cm" for c in cells)
    out["SC1w-P"] = [f"SC1w-P, P1 − P0: {_head(v)}", f"higher error in {', '.join(ps.region_name(r) for r in reg)} ({ns})"]
    sc2 = []
    for key, lab in (("SC2w", "n = 40"), ("SC2w-n160", "n = 160, aux.")):
        v = g[key].verdict
        m = re.search(r"(\d)지역 가운데 (\d)곳", v)
        _head(v)
        sc2.append((lab, m.group(2), m.group(1)))
    out["SC2w"] = [f"SC2w, R1 − P1: lower error in {sc2[0][1]}/{sc2[0][2]} regions ({sc2[0][0]})",
                   f"{sc2[1][1]}/{sc2[1][2]} regions at {sc2[1][0]}"]
    p1s = {int(r.n): r.verdict for r in V2[V2.row == "R1-P1*"].itertuples() if r.verdict not in ("same as P1",)}
    w40, w160 = V.VERDICT_TEXT[p1s[40]].lower(), V.VERDICT_TEXT[p1s[160]].lower()
    out["P1star"] = [f"R1 − P1* (edaphic re-fit, L29): {w40} at n = 40 and 160" if w40 == w160 else
                     f"R1 − P1* (edaphic re-fit, L29): {w40} at n = 40, {w160} at n = 160"]
    v = g["SC3w"].verdict
    r_ = re.search(r"라벨 비 ([\d.]+)", v)
    out["SC3w"] = [f"SC3w stopping rule 3 → 10 → 40 labels (τ = 0.05): {_head(v)}, label ratio {r_.group(1)}"]
    out["L43"] = [f"L43 placement (block-spread vs cell-random, n = 10 and 40): {_head(g['L43'].verdict)}"]
    return out


# ---------------------------------------------------------------- a
def panel_a(fig, rect, txt, SS):
    ax = ps.axes_mm(fig, *rect)
    W, H = rect[2], rect[3]
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    fs = ps.FS["annot"]
    stages = ["T0", "T3", "T10", "T40", "T160"]
    bw, gap = 27.0, 8.0
    x0s = [1.0 + k * (bw + gap) for k in range(5)]
    by, bh = H - 14.5, 13.5
    for k, st in enumerate(stages):
        q = SS[SS.stage == st]
        n = int(q.n.iloc[0])
        rec = ", ".join(dict.fromkeys(q.recipe))
        ref = ", ".join(dict.fromkeys(q.reference))
        aux = st == "T160"
        ax.add_patch(FancyBboxPatch((x0s[k], by), bw, bh, boxstyle="round,pad=0,rounding_size=1.2", fc="white" if not aux else "#f7f7f7",
                                    ec="#000000", lw=0.7, ls=(0, (2.5, 1.5)) if aux else "-", zorder=2))
        ax.text(x0s[k] + bw / 2, by + bh - 2.6, f"{st}: n = {n}{' (aux.)' if aux else ''}", ha="center", va="center", fontsize=ps.FS["base"],
                fontweight="bold", zorder=3)
        ax.text(x0s[k] + bw / 2, by + bh - 7.0, f"recipes {rec}", ha="center", va="center", fontsize=fs, zorder=3)
        ax.text(x0s[k] + bw / 2, by + bh - 10.8, f"compared with {ref}", ha="center", va="center", fontsize=fs, zorder=3)
        if k:
            ax.add_patch(FancyArrowPatch((x0s[k - 1] + bw + 0.6, by + bh / 2), (x0s[k] - 0.6, by + bh / 2), arrowstyle="-|>", mutation_scale=6,
                                         lw=0.8, color="#000000", ls="-" if not aux else (0, (2.5, 1.5)), zorder=3))
    # 판정 표지(표에서 생성한 문구). 괄호선으로 단계 범위를 표시한다
    def bracket(xa, xb, y):
        ax.plot([xa, xa, xb, xb], [y + 1.0, y, y, y + 1.0], color=ps.GREY["text2"], lw=0.5)
    def lines(x, y, rows, dy=3.4):
        for i, t in enumerate(rows):
            ax.text(x, y - i * dy, t, ha="left", va="center", fontsize=fs)
    y1 = by - 2.0
    bracket(x0s[1], x0s[2] + bw, y1)
    lines(x0s[1], y1 - 2.6, txt["SC1w"] + txt["SC1w-P"])
    bracket(x0s[3], x0s[4] + bw, y1)
    lines(x0s[3], y1 - 2.6, txt["SC2w"] + txt["P1star"])
    lines(x0s[1], y1 - 17.0, ["All stages: " + txt["SC3w"][0], "All stages: " + txt["L43"][0]])
    ax.set_gid("workflow")
    return ax


# ---------------------------------------------------------------- b
def panel_b(fig, rect):
    B = V.rd("fig7_b")
    SS = V.rd("fig7_b_summary")
    ax = ps.axes_mm(fig, *rect)
    nr = len(REG) + 1
    ys = {r: nr - 1 - i for i, (r, _) in enumerate(REG)}
    y_mean = -0.35
    for j, (st, rec, ref) in enumerate(COLS):
        for reg, _ in REG:
            q = B[(B.stage == st) & (B.recipe == rec) & (B.reference == ref) & (B.region == reg)]
            y = ys[reg]
            if not len(q):
                ax.add_patch(Rectangle((j - 0.45, y - 0.42), 0.9, 0.84, facecolor="none", edgecolor="#c8c8c8", hatch="//////", lw=0,
                                       zorder=0.5))
                continue
            r = q.iloc[0]
            V.draw_verdict(ax, j, y, r.verdict, bool(r.small_effect))
            if str(r.noninf).lower() == "true":
                ax.add_patch(Rectangle((j - 0.34, y - 0.36), 0.68, 0.72, fc="none", ec="#000000", lw=0.5, zorder=4))
        s = SS[(SS.stage == st) & (SS.recipe == rec) & (SS.reference == ref)].iloc[0]
        V.draw_verdict(ax, j, y_mean, s.verdict, False)
        if str(s.main4_noninf).lower() == "true":
            ax.add_patch(Rectangle((j - 0.34, y_mean - 0.36), 0.68, 0.72, fc="none", ec="#000000", lw=0.5, zorder=4))
        pool = {"지역 4/4": "P4", "부분(지역 2/4)": "L+C"}.get(s.main4_pool, s.main4_pool)
        ax.text(j, y_mean - 0.62, pool, ha="center", va="top", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    ax.axhline(0.5 - 0.02, color=ps.GREY["light"], lw=0.5)
    # 열 머리: 단계, 대비
    stg = {}
    for j, (st, rec, ref) in enumerate(COLS):
        stg.setdefault(st, []).append(j)
        ax.text(j, nr - 0.25, f"{rec}−{ref}", ha="center", va="bottom", fontsize=ps.FS["annot"], rotation=0)
    for st, js in stg.items():
        xa, xb = min(js) - 0.42, max(js) + 0.42
        ax.plot([xa, xb], [nr + 0.55, nr + 0.55], color="#000000", lw=0.5)
        ax.text((xa + xb) / 2, nr + 0.7, st, ha="center", va="bottom", fontsize=ps.FS["tick"])
    for j in (0.5, 2.5, 4.5, 7.5):
        ax.axvline(j, color=ps.GREY["band"], lw=3.0, zorder=0.1)
    # 가설 표지 행(아래)
    hyp = {1: "SC1w-P", 3: "SC1w-P", 2: "SC1w", 4: "SC1w", 7: "SC2w"}
    for j, h in hyp.items():
        ax.text(j, y_mean - 1.35, h, ha="center", va="top", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
    ax.set_yticks([ys[r] for r, _ in REG] + [y_mean])
    ax.set_yticklabels([lab for _, lab in REG] + ["P4 mean"])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.6, len(COLS) - 0.4)
    ax.set_ylim(y_mean - 1.9, nr + 1.5)
    return ax, B, SS


# ---------------------------------------------------------------- c
def panel_c(fig, rect):
    Cq = V.rd("fig7_c")
    ax = ps.axes_mm(fig, *rect)
    col = ps.COLOR["refit"]
    m = Cq[Cq.scope == "MEAN3"].sort_values("tau")
    for r in Cq[Cq.scope == "region"].itertuples():
        reg = r.region
        ax.plot([r.label_ratio], [r.delta], ls="none", marker=C3[reg], ms=ps.MS["main"] - 0.4, mfc=ps.GREY["light"], mec=ps.GREY["mid"],
                mew=0.4, zorder=2.5)
    ax.plot(m.label_ratio, m.delta, color=col, lw=0.6, zorder=2.6)
    for r in m.itertuples():
        main = abs(r.tau - 0.05) < 1e-9
        ax.plot([r.label_ratio, r.label_ratio], [r.ci_lo, r.ci_hi], color=col, lw=ps.LW["ci"], zorder=2.7)
        ax.plot([r.label_ratio], [r.delta], ls="none", marker="o", ms=ps.MS["mean"] + (1.2 if main else -0.6), mfc=col, mec=col, zorder=3)
        off = {0.025: (-3, 7, "right"), 0.05: (-5, -8, "right"), 0.1: (-4, -8, "right"), 0.2: (5, -2, "left")}[round(float(r.tau), 3)]
        ax.annotate(f"τ = {r.tau:g}", xy=(r.label_ratio, r.delta), xytext=off[:2], textcoords="offset points", ha=off[2], va="center",
                    fontsize=ps.FS["annot"], fontweight="bold" if main else "normal")
    ax.axhline(0.3, color=ps.GREY["mid"], lw=0.6, ls=(0, (1.2, 1.4)), zorder=1)
    ax.axhline(0.5, color=ps.GREY["mid"], lw=0.6, ls=(0, (3.5, 1.8)), zorder=1)
    ax.axvline(0.5, color=ps.GREY["mid"], lw=0.6, ls=(0, (1.2, 1.4)), zorder=1)
    ax.text(0.02, 0.3, "0.3 cm", ha="left", va="top", fontsize=ps.FS["annot"], color=ps.GREY["text2"],
            transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
    ax.text(0.02, 0.5, "0.5 cm", ha="left", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"],
            transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
    ps.zero_line(ax, "y", better_text=False)
    ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(-2.4, 0.9); ax.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0 if float(v).is_integer() else 1)))
    ax.set_xlabel("Label ratio (mean labels used / 40)")
    ax.set_ylabel("Δ$_{SC3}$ (cm)")
    return ax, Cq


# ---------------------------------------------------------------- d
def panel_d(fig, rect, rect10, xv, xt):
    AB = V.rd("fig7_d")
    R = V.rd("fig7_d_regions")
    A10 = V.rd("fig7_d_ab10").iloc[0]
    codes = [f"AB{k}" for k in range(1, 10)]
    ax = ps.axes_mm(fig, *rect)
    n = len(codes)
    ys = np.arange(n)[::-1].astype(float)
    V.symlog_y(ax, lim=(-18, 7), ticks=(-15, -5, -2, -1, 0, 1, 2, 5), linthresh=2.0, linscale=0.7, axis="x")
    ps.zero_line(ax, "x", better_text=False)
    ax.axvspan(-0.5, 0.5, facecolor=ps.GREY["band"], edgecolor="none", zorder=0.1)
    for c, y in zip(codes, ys):
        r = AB[AB.ab == c].iloc[0]
        rg = R[R.ab == c]
        ax.scatter(rg.delta, np.full(len(rg), y - 0.24), s=5, color=ps.GREY["mid"], lw=0, zorder=2.2)
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color="#000000", lw=ps.LW["ci_forest"], solid_capstyle="round", zorder=2.5)
        ax.plot([r.delta], [y], "o", ms=ps.MS["mean"], color="#000000", zorder=3)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(ys); ax.set_yticklabels([f"{c}  {AB_LABEL[c]}" for c in codes])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ΔRMSE, P4 mean (cm; log scale beyond ±2 cm)")
    # AB10 별도 축(구간 점수 차, cm)
    a10 = ps.axes_mm(fig, *rect10)
    a10.set_xlim(-5, 25); a10.set_xticks([0, 5, 10, 15, 20, 25])
    a10.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))
    ps.zero_line(a10, "x", better_text=False)
    a10.plot([A10.ci_lo_cm, A10.ci_hi_cm], [0, 0], color="#000000", lw=ps.LW["ci_forest"], solid_capstyle="round", zorder=2.5)
    a10.plot([A10.delta_cm], [0], "o", ms=ps.MS["mean"], color="#000000", zorder=3)
    a10.set_ylim(-0.6, 0.6); a10.set_yticks([0]); a10.set_yticklabels([f"AB10  {AB_LABEL['AB10']}"])
    a10.tick_params(axis="y", length=0, pad=2)
    a10.spines["left"].set_visible(False)
    a10.set_xlabel("Δ interval score, Lena, Canada, Alaska (x) (cm)")
    # 오른쪽 열: 기호(보정 전 4분 판정), 초록 규칙 문구(Holm)
    for axx, cc, yy in ((ax, codes, ys), (a10, ["AB10"], [0.0])):
        x0, y0, w, h = axx._box_mm
        va = ps.axes_mm(fig, xv, y0, 5.0, h); va.set_ylim(axx.get_ylim()); va.set_xlim(0, 1); va.set_axis_off(); va.set_gid("verdict_col")
        ta = ps.axes_mm(fig, xt, y0, 40.0, h); ta.set_ylim(axx.get_ylim()); ta.set_xlim(0, 1); ta.set_axis_off(); ta.set_gid("rule_col")
        for c, y in zip(cc, yy):
            r = AB[AB.ab == c].iloc[0]
            V.draw_verdict(va, 0.4, y, r.verdict, bool(r.small_effect))
            ta.text(0.0, y, V.AB_RULE_TEXT[r.rule], ha="left", va="center", fontsize=ps.FS["annot"])
        if axx is ax:
            va.text(0.4, n - 0.35, "verdict", ha="center", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
            ta.text(0.0, n - 0.35, "abstract wording after Holm (m = 10)", ha="left", va="bottom", fontsize=ps.FS["annot"],
                    color=ps.GREY["text2"])
    ax.text(0.0, 1.0, "grey band ±0.5 cm; grey dots: regions", transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["annot"],
            color=ps.GREY["text2"])
    return ax, a10, AB, R, A10


def build():
    ps.use_paper()
    W, H = 180.0, 182.0
    fig = ps.paper_figure(W, H)
    T = V.rd("fig7_tests")
    V2 = V.rd("fig2_verdicts")
    SS = V.rd("fig7_b_summary")
    txt = sc_texts(T, V2)
    ax_a = panel_a(fig, (2.0, 124.0, 176.0, 40.0), txt, SS)
    ax_b, B, SSb = panel_b(fig, (22.0, 72.0, 90.0, 44.0))
    ax_c, Cq = panel_c(fig, (130.0, 79.0, 46.0, 35.0))
    ax_d, a10, AB, R, A10 = panel_d(fig, (58.0, 28.0, 66.0, 36.0), (58.0, 10.0, 66.0, 4.5), xv=124.5, xt=131.0)
    sep = fig.add_artist(Line2D([2.0 / W, 178.0 / W], [17.3 / H, 17.3 / H], transform=fig.transFigure, color=ps.GREY["light"], lw=0.5))
    sep.set_gid("ab10_separator")
    ps.panel_label(ax_a, "a", dx_mm=-1.0, dy_mm=0.3)
    ps.panel_label(ax_b, "b", dx_mm=-20.0, dy_mm=1.0)
    ps.panel_label(ax_c, "c", dx_mm=-12.0, dy_mm=4.0)
    ps.panel_label(ax_d, "d", dx_mm=-56.0, dy_mm=2.5)
    handles = [Line2D([], [], color="#000000", lw=ps.LW["ci_forest"], marker="o", ms=ps.MS["mean"], label="P4 mean, 95% CI (d)"),
               Line2D([], [], ls="none", marker="o", ms=2.2, mfc=ps.GREY["mid"], mec=ps.GREY["mid"], label="Region estimate (d)"),
               Rectangle((0, 0), 1, 1, fc="none", ec="#000000", lw=0.5, label="Non-inferior to column reference (b)"),
               Rectangle((0, 0), 1, 1, fc="none", ec="#c8c8c8", hatch="//////", lw=0, label="n above the labelled pool (b)"),
               Line2D([], [], color=ps.COLOR["refit"], lw=0.6, marker="o", ms=ps.MS["mean"], label="3-region mean, 95% CI (c)")]
    handles += [Line2D([], [], ls="none", marker=mk, ms=ps.MS["main"] - 0.4, mfc=ps.GREY["light"], mec=ps.GREY["mid"], mew=0.4,
                       label=f"{ps.region_name(r) if r != 'Alaska' else 'Alaska (x)'} (c)") for r, mk in C3.items()]
    ps.legend_below(fig, handles, (2.0, H - 11.0, W - 4.0, 10.0), ncol=4)
    kax = ps.axes_mm(fig, 4.0, H - 16.0, 172.0, 4.2)
    kax.set_axis_off(); kax.set_xlim(0, 1); kax.set_ylim(0, 1)
    V.verdict_key(kax, 0.0, 0.5, include_small=True)
    kax.set_gid("verdict_key")
    rel = V.rd("fig7_delta_rel").iloc[0]
    Rg = V.rd("fig7_d_regions")
    def nsup(code):
        q = Rg[Rg.ab == code]
        sup = q[q.verdict == "superior"].region.map(ps.region_name).tolist()
        inf = q[q.verdict == "inferior"].region.map(ps.region_name).tolist()
        return f"{len(sup)}/{len(q)} regions lower error ({', '.join(sup)}); higher error in {', '.join(inf) or 'none'}"
    same = "unchanged for all nine" if int(rel.n_same) == int(rel.n_rows) else f"unchanged for {int(rel.n_same)} of {int(rel.n_rows)}"
    cap = dict(
        definition=("Stages T0 to T40 add target labels (T160 auxiliary); recipes P1 (κ = 10 re-fit) and R1 (P1 + residual CatBoost). "
                    "Non-inferior: both CI upper bounds below +0.5 cm against the column reference (SC1w counts only the R1 − P0 cells at "
                    "n = 3 and 10). Δ_SC3 compares a jackknife stopping rule with always using 40 "
                    "labels (Δ_SC3 = RMSE(rule) − RMSE(40 labels)). AB1 to AB10 are the ten pre-specified abstract contrasts."),
        statistics=("b–d, block bootstrap with 10,000 resamples, both weightings; AB10 uses the LGU two-stage bootstrap with 1,000 "
                    "resamples. Verdict symbols are the four-way verdicts before correction; the wording column applies Holm "
                    f"correction over the ten contrasts (AB5 Holm p = {AB[AB.ab == 'AB5'].holm_p.iloc[0]:.3f}; AB6 equivalence after "
                    f"correction). With a margin of 2% of the P0 RMSE ({rel.margin_cm_P4:.2f} cm) the verdicts are {same}."),
        panels=("a, stage flow; texts are generated from the registered verdicts (SC1w, SC1w-P, SC2w, SC3w, L43). b, cells by region; "
                "P4 mean row uses Lena and Canada only at n ≥ 40 (L+C). c, 3-region mean (Lena, Canada, Alaska (x)) and regions. "
                f"d, AB4: {nsup('AB4')}. AB8: {nsup('AB8')}. AB10 is an interval-score contrast in cm on its own axis. "
                "AB1 to AB5, AB8 and AB9 had been seen in direction in earlier experiments with other protocols; AB6 and AB7 are blind."),
        data=("LG and LGX shards (Rescale) aggregated by h39; LGU (local). AB4, AB10 and SC1w-P existed at registration (not viewed); "
              "AB4 and SC1w-P are quoted after viewing S-a."),
    )
    src = dict(a_texts=__import__("pandas").DataFrame([dict(key=k, text=" | ".join(v)) for k, v in txt.items()]), b_cells=B, b_summary=SSb, c=Cq,
               d=AB, d_regions=R, d_ab10=V.rd("fig7_d_ab10"))
    spec_extra = dict(module="fig7_deployment.py", panels=4, message="deployment stages with registered safety verdicts; abstract contrast bundle with Holm wording")
    return V.save_v2(fig, NAME, cap, SPEC_ID, src, title="Deployment stages by label budget and the pre-specified abstract contrasts.",
                     spec_extra=spec_extra)


if __name__ == "__main__":
    build()
