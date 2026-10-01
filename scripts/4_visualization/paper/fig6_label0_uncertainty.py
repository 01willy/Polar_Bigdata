"""Fig 6 (v2). 라벨 0 전이와 불확실성. 스펙: figure_spec.json display_items 'Fig 6'(layout_B, 지도 없음. LGU-B2 미결정이라
지도는 SI, DISPLAY_ITEMS 2절 'Fig 6 지도 병합').

패널
  a  라벨 0 대비 포레스트(n = 0, P4 층화 평균, 대비 = 방법 − P0). 행 = 스펙 rows_fixed(결과 전 고정)와 L29_pstar(P* 성립).
     플랫폼 묶음(Rescale, 로컬 GPU) 사이에 틈과 괄호 이름. 플랫폼 사이의 차는 그리지 않는다(LGF 2.2).
     같은 학습기의 두 조건은 CI 선종(실선 = 컨텍스트 10,000행·조정판, 파선 = 원천 전체·로컬 기본판).
     오른쪽 = 4분 판정 기호 두 열(실선, 파선, AB 표지는 기호 옆), 지지 갈래 열(LGF 2.2 의 두 갈래, 원천 A·B → 그림 글자 1·2),
     머리 행 = 등록 판정(표에서 생성), 묶음 범위 괄호.
  b  라벨 0 구간(LGU-B, n = 0, 범위 all): 정규화기 6종의 P4 평균(검정 마커 모양 + 약칭), 지역 값(작은 회색, 같은 모양),
     옛 C2 hier2_cdf 참조(빈 회색 원, CI 없음). 회색 띠 0.85–0.95. CI 는 그리지 않는다(스펙 b: 수준 값, CI 는 Source Data).
     거의 겹치는 P4 평균 두 무리(cfm·perm.·nflow, const·phys·C2)는 오른쪽 아래 확대 삽도 두 개로 푼다.
  c  구간 점수 대비(Δ / 기준 방법 점수, %): LGU-B2(n = 0, P4), LGU-A1(n 10, 40, 레나·캐나다·Alaska(x)). LGU 2단 CI 1,000회.
     ±5 % 띠(주 동등성 한계). ‡ = 2단과 1단 판정이 다름(보정 불확실성 의존). AB10 표지. 두 풀은 패널 머리 글자.
  d  L25: R1 90 % 구간 커버리지(CI) 대 폭 비(R0, n = 0 폭 대비), 레나·캐나다·Alaska(x) × n 40·160.
자료: data/processed/paper_figs/fig6_*.csv(v2_data.py --only fig6). 판정 기호·문구는 원천 표 열을 옮긴다(다시 계산하지 않는다).
실행: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 taskset -c 0-3 python3 scripts/4_visualization/paper/fig6_label0_uncertainty.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                       # noqa: E402
import matplotlib                                                        # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import Rectangle                                 # noqa: E402

import v2_style as V                                                     # noqa: E402
from v2_style import ps                                                  # noqa: E402

NAME = "Fig6_label0_uncertainty"
SPEC_ID = "paper_fig6_label0_uncertainty"
DASH = ps.VARIANT_LS["dashed"]
OFF = 0.19                                     # 두 조건 행의 세로 오프셋(행 단위)
U_MM = 3.7                                     # a 의 행 단위(mm)
C_BASE = "#000000"                             # 물리 기준선
C_DIRECT = ps.COLOR["direct"]
C_R1 = ps.COLOR["residual"]

# ---------------------------------------------------------------- a 의 행(위 → 아래). 행 키는 v2_data.FIG6A_LINES 와 같다
ROWS_A = [
    ("head", "baselines", "Physics baselines (LGX N1)"),
    ("row", "baselines", "B:ens", "B:ens, anchor ensemble"),
    ("row", "baselines", "P*", "P*: P0 with year-matched TDD"),
    ("head", "direct_rescale", "Direct ML (LG, LGX)"),
    ("row", "direct_rescale", "catboost_lo", "CatBoost, low capacity"),
    ("row", "direct_rescale", "catboost", "CatBoost, 600 trees, depth 6"),
    ("row", "direct_rescale", "rf", "Random forest, 500 trees"),
    ("row", "direct_rescale", "catboost_tuned", "CatBoost, tuned by source CV"),
    ("row", "direct_rescale", "F1k", "F1k: CatBoost + physics inputs"),
    ("gap",),
    ("head", "lgt", "Direct ML (LGT)"),
    ("row", "lgt", "tabpfn", "TabPFN v2"),
    ("row", "lgt", "catboost_ctx", "CatBoost, 10,000-row context"),
    ("head", "lgf_f", "Direct ML (LGF-F)"),
    ("row", "lgf_f", "tabicl", "TabICL v2"),
    ("row", "lgf_f", "catboost_ctx", "CatBoost, same contexts"),
    ("head", "lgf_n", "Direct ML (LGF-N)"),
    ("row", "lgf_n", "mlp", "MLP"),
    ("row", "lgf_n", "tabm", "Multi-head MLP"),
    ("row", "lgf_n", "ftt", "FT-T-type (reduced)"),
    ("row", "lgf_n", "realmlp", "RealMLP"),
]
BLOCK_HYP = {"baselines": ["L29"], "direct_rescale": ["L1", "L30", "L11"], "lgt": ["L34(a)"], "lgf_f": ["LGF-F1"], "lgf_n": ["LGF-N1"]}
PLATFORM_TXT = {"Rescale": "Rescale (LG, LGX shards)", "local GPU": "local GPU (RTX 3090)"}
H_HEAD, H_ROW, H_GAP = 0.9, 1.0, 0.75

# ---------------------------------------------------------------- b 의 정규화기(마커 모양 + 약칭, 방법 색 재사용 금지)
NORM = {  # key → (약칭, 마커, P4 평균 채움)
    "const": ("const", "s", "#000000"),
    "phys": ("phys", "P", "#000000"),
    "cfm": ("cfm", "<", "#000000"),
    "nflow": ("nflow", "h", "#000000"),
    "nflow#placebo": ("perm.", "h", "#a6a6a6"),
    "cbq": ("cbq", ">", "#000000"),
}
B_MAIN_LABEL = {"cbq": (141.0, 0.935)}                           # 본 축 약칭(자료 좌표) + 지시선. 겹치는 두 무리는 삽도에서 이름을 단다
# 거의 같은 P4 평균 두 무리의 확대 삽도(같은 마커, 같은 약칭). 창 = (x0, x1, y0, y1), 삽도 위치 = 본 축 자료 좌표(x, y, 폭, 높이),
# 약칭 위치 = {key: (dx pt, dy pt, ha)}. 창 안의 지역 값도 같은 모양으로 그린다. 연결선은 창 오른쪽 아래 → 삽도 왼쪽 위 한 줄
B_INSETS = [
    dict(keys=("cfm", "nflow#placebo", "nflow"), win=(93.0, 97.4, 0.8822, 0.8946), pos=(101.5, 0.813, 56.0, 0.068),
         labels={"cfm": (-4.5, 0.0, "right"), "nflow#placebo": (4.5, 0.0, "left"), "nflow": (-4.5, 0.0, "right")}, ref=False),
    dict(keys=("const", "phys"), win=(77.0, 84.4, 0.8565, 0.8775), pos=(101.5, 0.733, 56.0, 0.068),
         labels={"const": (4.5, 0.0, "left"), "phys": (-4.5, 0.0, "right"), "ref": (-5.0, 0.0, "right")}, ref=True),
]
CLASS_TXT = {"A": "1", "B": "2"}                                  # LGF 2.2 의 두 갈래(h47·h48 의 A·B) → 그림 글자 1·2(AB 표지와 겹치지 않게)
COV_LIM, COV_TICKS = (0.72, 1.0), (0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
TMARK_D = {"Lena": "D", "Canada": "s", "Alaska": "o"}
TNAME_D = {"Lena": "Lena", "Canada": "Canada", "Alaska": "Alaska (x)"}
D_LABEL_POS = {("Alaska", 160): ("above", "center", 0.0), ("Alaska", 40): ("below", "center", 0.0), ("Canada", 160): ("below", "center", 0.0),
               ("Canada", 40): ("above", "center", 0.0), ("Lena", 40): ("below", "center", 0.0), ("Lena", 160): ("above", "right", 2.0)}


def _ci(ax, y, lo, hi, color, ls="-", lw=None, z=2.5):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=color, ls=ls, lw=lw or ps.LW["ci_forest"], solid_capstyle="round", dash_capstyle="butt", zorder=z)


def dash_fit(ax, lo, hi, lw=None, pattern=DASH, lim=(0.75, 1.25)):
    """파선 CI 의 양 끝이 대시로 끝나도록 무늬 배율을 고른다(위상 0). 길이 L = k·(d + g)·f·lw + d·f·lw 인 정수 k 가운데
    배율 f 가 1 에 가장 가까운 것. 무늬는 선 굵기로 곱해진다(lines.scale_dashes). 반환 = matplotlib ls."""
    lw = lw or ps.LW["ci_forest"]
    d, g = pattern[1]
    x0, x1 = ax.get_xlim()
    px = ax.transData.transform([(x0, 0), (x1, 0)])
    pt_per = (px[1, 0] - px[0, 0]) / (x1 - x0) * 72.0 / ax.figure.dpi
    L = (hi - lo) * pt_per
    best = None
    for k in range(0, 400):
        f = L / (lw * (k * (d + g) + d))
        if lim[0] <= f <= lim[1] and (best is None or abs(f - 1) < abs(best - 1)):
            best = f
    if best is None:
        raise RuntimeError(f"파선 CI(길이 {L:.1f} pt)에 맞는 무늬 배율 없음")
    return (0.0, (d * best, g * best))


def _pt(ax, x, y, color, marker, ms=None, mew=None, z=3):
    ax.plot([x], [y], ls="none", marker=marker, ms=ms or ps.MS["main"], mfc=color, mec=color, mew=mew or ps.LW["marker_edge"], zorder=z)


def _side_axes(fig, ax_ref, x_mm, w_mm, gid):
    x0, y0, w, h = ax_ref._box_mm
    ax = ps.axes_mm(fig, x_mm, y0, w_mm, h)
    ax.set_ylim(ax_ref.get_ylim()); ax.set_xlim(0, 1); ax.set_axis_off(); ax.set_gid(gid)
    return ax


def layout_a():
    """행 y 좌표(위 → 아래, 행 단위). 반환 [(항목, y)], (y 하한, y 상한)."""
    y, out = 0.0, []
    for it in ROWS_A:
        if it[0] == "gap":
            y -= H_GAP
            continue
        step = H_HEAD if it[0] == "head" else H_ROW
        y -= step / 2
        out.append((it, y))
        y -= step / 2
    return out, (y - 0.15, 0.15)


# ---------------------------------------------------------------- a
def panel_a(fig, x_axis, y_bot, w_axis, x_sym, x_txt, x_brk):
    A = V.rd("fig6_a")
    R = V.rd("fig6_a_registered").set_index("hypothesis")
    items, (ylo, yhi) = layout_a()
    h_mm = (yhi - ylo) * U_MM
    ax = ps.axes_mm(fig, x_axis, y_bot, w_axis, h_mm)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-4.3, 8.3); ax.set_xticks([-4, -2, 0, 2, 4, 6, 8])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))
    ax.set_xlabel("ΔRMSE vs P0 at n = 0, P4 mean (cm)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ps.zero_line(ax, "x", better_text=False)
    ys = [y for _, y in items]
    ax.set_yticks(ys)
    ax.set_yticklabels([it[2] if it[0] == "head" else it[3] for it, _ in items])
    for tl, (it, _) in zip(ax.get_yticklabels(), items):
        if it[0] == "head":
            tl.set_fontweight("bold")
    sv = _side_axes(fig, ax, x_sym, 17.0, "verdict_cols")          # 열 x(축 좌표): 실선 0.12, 파선 0.44, 갈래 0.76
    tx = _side_axes(fig, ax, x_txt, 178.0 - x_txt, "registered_col")
    xs = {"solid": 0.12, "dashed": 0.44, "class": 0.76}
    drawn = []
    for it, y in items:
        if it[0] == "head":
            hyps = BLOCK_HYP[it[1]]
            vv = [R.loc[h, "verdict_en"] for h in hyps]
            if len(set(vv)) == 1:
                txt = f"{', '.join(hyps)}: {vv[0]}"
            else:
                txt = "; ".join(f"{h}: {v}" for h, v in zip(hyps, vv))
            if it[1] == "baselines" and vv[0] == "lower-error baseline found":
                txt = f"{hyps[0]}: lower-error baseline (P*)"
            if it[1] == "lgf_n":
                det = json.loads(R.loc["LGF-N1", "support_detail"])
                txt += f" ({len(det)} learners)"
            tx.text(0.0, y, txt, ha="left", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
            continue
        blk, key = it[1], it[2]
        q = A[(A.block == blk) & (A.key == key)]
        two = len(q) == 2
        col = C_BASE if blk == "baselines" else C_DIRECT
        mk, ms, mew = ("o", ps.MS["main"] + 0.3, ps.LW["marker_edge"]) if blk == "baselines" else ("x", ps.MS["main"] + 0.5, 0.9)
        for r in q.itertuples():
            yy = y + (OFF if r.variant == "solid" else -OFF) if two else y
            _ci(ax, yy, r.ci_lo, r.ci_hi, col, "-" if r.variant == "solid" else dash_fit(ax, r.ci_lo, r.ci_hi))
            _pt(ax, r.delta, yy, col, mk, ms=ms, mew=mew)
            V.draw_verdict(sv, xs[r.variant], y, r.verdict, bool(r.small_effect))
            drawn.append(dict(block=blk, key=key, variant=r.variant, y=yy, delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, verdict=r.verdict))
            if isinstance(r.ab, str) and r.ab.startswith("AB"):              # 행 단위 표지: 기호 바로 오른쪽
                V.ab_tag(sv, xs[r.variant], y, r.ab, dx_pt=5.0, dy_pt=0.0, ha="left", va="center")
        if blk in ("lgf_f", "lgf_n") and key not in ("catboost_ctx",):
            det = json.loads(R.loc["LGF-F1" if blk == "lgf_f" else "LGF-N1", "support_detail"])
            code = det.get(key)
            if code in CLASS_TXT:
                t = sv.text(xs["class"], y, CLASS_TXT[code], ha="center", va="center", fontsize=ps.FS["annot"], fontweight="bold")
                t.set_gid("support_class")
    # 열 머리: 선 견본(실선, 파선)과 'class', 오른쪽 열 'registered verdict'
    tr = matplotlib.transforms.blended_transform_factory(sv.transData, sv.transAxes)
    for k, ls in (("solid", "-"), ("dashed", DASH)):
        ln = sv.plot([xs[k] - 0.09, xs[k] + 0.09], [1.012, 1.012], transform=tr, color=C_DIRECT, lw=ps.LW["ci_forest"], ls=ls,
                     solid_capstyle="butt", dash_capstyle="butt", clip_on=False)[0]
        ln.set_gid("col_head")
    sv.text(xs["class"], 1.0, "class", transform=tr, ha="center", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"], clip_on=False)
    tx.text(0.0, 1.0, "registered verdict", transform=tx.transAxes, ha="left", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"],
            clip_on=False)
    # 묶음 범위 괄호(오른쪽 열): 머리 행의 등록 판정과 두 조건의 뜻은 묶음 전체에 걸린다
    for blk in BLOCK_HYP:
        yy = [y for it, y in items if it[1] == blk]
        top, bot = max(yy) + 0.42, min(yy) - 0.42
        ln = tx.plot([-0.012, -0.022, -0.022, -0.012], [top, top, bot, bot], color=ps.GREY["light"], lw=0.5, solid_capstyle="butt",
                     clip_on=False)[0]
        ln.set_gid("block_bracket")
    # 두 조건의 뜻(그 묶음 괄호 안, 오른쪽 열)
    notes = {("lgf_f", "tabicl"): "solid: 10,000-row context", ("lgf_f", "catboost_ctx"): "dashed: full source as context",
             ("lgf_n", "mlp"): "solid: tuned (source-region CV)", ("lgf_n", "tabm"): "dashed: local default"}
    for it, y in items:
        if it[0] == "row" and (it[1], it[2]) in notes:
            t = tx.text(0.0, y, notes[(it[1], it[2])], ha="left", va="center", fontsize=ps.FS["annot"], color=ps.GREY["text2"])
            t.set_gid("variant_note")
    # 플랫폼 괄호(가장 왼쪽)
    bk = _side_axes(fig, ax, x_brk, 5.0, "platform_bracket")
    for plat in ("Rescale", "local GPU"):
        blks = [b for b, p in (("baselines", "Rescale"), ("direct_rescale", "Rescale"), ("lgt", "local GPU"), ("lgf_f", "local GPU"),
                               ("lgf_n", "local GPU")) if p == plat]
        yy = [y for it, y in items if it[1] in blks]
        top, bot = max(yy) + 0.45, min(yy) - 0.45
        bk.plot([0.75, 0.55, 0.55, 0.75], [top, top, bot, bot], color="#000000", lw=0.6, solid_capstyle="butt", clip_on=False)
        t = bk.text(0.25, (top + bot) / 2, PLATFORM_TXT[plat], rotation=90, ha="center", va="center", fontsize=ps.FS["annot"])
        t.set_gid("platform_label")
    return ax, A, R, drawn


# ---------------------------------------------------------------- b
def _cov_axis(ax):
    ax.set_ylim(*COV_LIM); ax.set_yticks(COV_TICKS)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: "1" if v >= 1 else f"{v:.2f}".rstrip("0")))
    ax.axhspan(0.85, 0.95, facecolor=ps.GREY["band"], edgecolor="none", zorder=0.1)
    ax.axhline(0.90, color=ps.GREY["mid"], lw=0.5, ls=(0, (1.2, 1.4)), zorder=0.3)
    ax.set_ylabel("90% interval coverage")


def panel_b(fig, rect):
    B = V.rd("fig6_b")
    Rf = V.rd("fig6_b_ref")
    ax = ps.axes_mm(fig, *rect)
    _cov_axis(ax)
    for m, (lab, mk, fill) in NORM.items():
        rg = B[(B.method == m) & (B.kind == "region")]
        ax.plot(rg.wid10, rg.cov10, ls="none", marker=mk, ms=2.4, mfc="#d4d4d4", mec="#8c8c8c", mew=0.4, zorder=2.0)
    ref = Rf[Rf.kind == "mean4"].iloc[0]
    ax.plot([ref.width_cm], [ref.coverage], ls="none", marker="o", ms=6.0, mfc="none", mec=ps.GREY["mid"], mew=0.8, zorder=3.4)
    M = B[B.kind == "mean4"].set_index("method")
    for m, (lab, mk, fill) in NORM.items():
        r = M.loc[m]
        ax.plot([r.wid10], [r.cov10], ls="none", marker=mk, ms=ps.MS["mean"], mfc=fill, mec="#000000", mew=0.6, zorder=3.6)
        if m in B_MAIN_LABEL:
            t = ax.annotate(lab, xy=(r.wid10, r.cov10), xytext=B_MAIN_LABEL[m], textcoords="data", ha="center", va="center",
                            fontsize=ps.FS["annot"], arrowprops=dict(arrowstyle="-", lw=0.4, color=ps.GREY["mid"], shrinkA=1.5, shrinkB=2.5),
                            zorder=4)
            t.set_gid("norm_label")
    ax.set_xlim(60, 160); ax.set_xticks([60, 80, 100, 120, 140, 160])
    ax.set_xlabel("Mean 90% interval width (cm)")
    # 거의 겹치는 P4 평균 두 무리의 확대 삽도(눈금 없음, 같은 마커·약칭). 본 축에는 창 사각형과 연결선 한 줄
    inv = fig.transFigure.inverted()
    for spec in B_INSETS:
        x0, x1, y0, y1 = spec["win"]
        bx, by, bw, bh = spec["pos"]
        (fx0, fy0), (fx1, fy1) = inv.transform(ax.transData.transform([(bx, by), (bx + bw, by + bh)]))
        ins = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])
        ins.set_gid("inset")
        ins.set_xlim(x0, x1); ins.set_ylim(y0, y1)
        ins.set_xticks([]); ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.5); sp.set_edgecolor(ps.GREY["mid"])
        ins.axhspan(0.85, 0.95, facecolor=ps.GREY["band"], edgecolor="none", zorder=0.1)
        inwin = B[(B.kind == "region") & B.wid10.between(x0, x1) & B.cov10.between(y0, y1)]
        for m, (lab, mk, fill) in NORM.items():
            q = inwin[inwin.method == m]
            ins.plot(q.wid10, q.cov10, ls="none", marker=mk, ms=2.4, mfc="#d4d4d4", mec="#8c8c8c", mew=0.4, zorder=2.0)
        for m in spec["keys"]:
            lab, mk, fill = NORM[m]
            r = M.loc[m]
            ins.plot([r.wid10], [r.cov10], ls="none", marker=mk, ms=ps.MS["mean"] + 0.6, mfc=fill, mec="#000000", mew=0.6, zorder=3)
            dx, dy, ha = spec["labels"][m]
            ins.annotate(lab, xy=(r.wid10, r.cov10), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=ps.FS["annot"])
        if spec["ref"] and x0 <= ref.width_cm <= x1 and y0 <= ref.coverage <= y1:
            ins.plot([ref.width_cm], [ref.coverage], ls="none", marker="o", ms=6.0, mfc="none", mec=ps.GREY["mid"], mew=0.8, zorder=3.4)
            dx, dy, ha = spec["labels"]["ref"]
            ins.annotate("C2", xy=(ref.width_cm, ref.coverage), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                         fontsize=ps.FS["annot"], color=ps.GREY["text2"])
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=ps.GREY["mid"], lw=0.5, zorder=3.8))
        cl = ax.plot([x1, bx], [y0, by + bh], color=ps.GREY["light"], lw=0.4, zorder=1.9)[0]
        cl.set_gid("inset_connector")
    ax.text(0.0, 1.02, "n = 0; large: P4 mean, small: regions", transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["annot"],
            color=ps.GREY["text2"])
    return ax, B, Rf


# ---------------------------------------------------------------- c
def panel_c(fig, rect, x_sym):
    Cc = V.rd("fig6_c")
    ax = ps.axes_mm(fig, *rect)
    order = [("LGU-B2", 0, "B2, n = 0", "nflow − perm."), ("LGU-A1", 10, "A1, n = 10", "(iii) − B4"), ("LGU-A1", 40, "A1, n = 40", "(iii) − B4")]
    pools = {"LGU-B2": "P4", "LGU-A1": "Lena, Canada, Alaska (x)"}
    ys = np.arange(len(order))[::-1].astype(float)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_yticks(ys); ax.set_yticklabels([f"{o[2]}\n{o[3]}" for o in order])
    for tl in ax.get_yticklabels():
        tl.set_linespacing(1.15)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.axvspan(-5, 5, facecolor=ps.GREY["band"], edgecolor="none", zorder=0.1)
    ps.zero_line(ax, "x", better_text=False)
    ax.axhline(ys[0] - 0.5, color=ps.GREY["light"], lw=0.5, zorder=0.2)
    sv = _side_axes(fig, ax, x_sym, 16.0, "verdict_col_c")
    for (test, n, _, _), y in zip(order, ys):
        r = Cc[(Cc.test == test) & (Cc.n == n)].iloc[0]
        _ci(ax, y, r.ci_lo_pct, r.ci_hi_pct, "#000000")
        _pt(ax, r.delta_pct, y, "#000000", "o", ms=ps.MS["mean"])
        V.draw_verdict(sv, 0.12, y, r.verdict, False)
        dx = 4.2
        if bool(r.calib_dependent):
            t = sv.annotate("‡", xy=(0.12, y), xytext=(dx, 0.4), textcoords="offset points", ha="left", va="center", fontsize=ps.FS["annot"],
                            annotation_clip=False)
            t.set_gid("calib_flag")
            dx += 5.0
        if isinstance(r.ab, str) and r.ab.startswith("AB"):
            V.ab_tag(sv, 0.12, y, r.ab, dx_pt=dx + 1.0, dy_pt=0.0, ha="left", va="center")
    ax.set_xlim(-15, 25); ax.set_xticks([-10, 0, 10, 20])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: ps.fmt_num(v, 0)))
    ax.set_xlabel("Δ interval score (% of reference score)")
    ax.text(-0.55, 1.03, f"B2 pool: {pools['LGU-B2']}\nA1 pool: {pools['LGU-A1']}\ngrey band: ±5% of reference",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["annot"], color=ps.GREY["text2"], linespacing=1.15)
    return ax, Cc


# ---------------------------------------------------------------- d
def panel_d(fig, rect):
    D = V.rd("fig6_d")
    Dr = V.rd("fig6_d_registered").iloc[0]
    ax = ps.axes_mm(fig, *rect)
    _cov_axis(ax)
    ax.axvline(1.0, color=ps.GREY["text2"], lw=0.6, ls=(0, (3.0, 1.6)), zorder=0.4)
    for r in D.itertuples():
        ax.plot([r.width_ratio, r.width_ratio], [r.cov_lo, r.cov_hi], color=C_R1, lw=ps.LW["ci"], zorder=2.4)
        _pt(ax, r.width_ratio, r.coverage, C_R1, TMARK_D[r.target], ms=ps.MS["main"] + 0.4, z=3)
        pos, ha, dx = D_LABEL_POS[(r.target, int(r.n))]
        yy, va, dy = (r.cov_hi, "bottom", 1.2) if pos == "above" else (r.cov_lo, "top", -1.2)
        t = ax.annotate(f"{int(r.n)}", xy=(r.width_ratio, yy), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                        fontsize=ps.FS["annot"], color=ps.GREY["text2"])
        t.set_gid("n_label")
    ax.set_xlim(0.52, 1.04); ax.set_xticks([0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.1f}"))
    ax.set_xlabel("Width / R0 width at n = 0")
    ax.text(0.04, 0.03, f"{int(Dr.c1)} of {int(Dr.n_cells)} in band,\n{int(Dr.c2)} of {int(Dr.c1)} narrower", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=ps.FS["annot"], color=ps.GREY["text2"], linespacing=1.1)
    ax.text(0.0, 1.02, "R1, n = 40 and 160 (L25)", transform=ax.transAxes, ha="left", va="bottom", fontsize=ps.FS["annot"],
            color=ps.GREY["text2"])
    return ax, D, Dr


def verdict_key6(ax, x0, y, include_small):
    """4분 판정 기호 설명 한 줄(동등성 한계가 패널마다 다르다: a 0.5 cm, c 기준 점수의 5 %)."""
    text = dict(V.VERDICT_TEXT)
    text["equivalent"] = "Equivalent (a ±0.5 cm, c ±5%)"
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    x = x0
    items = [(v, text[v]) for v in V.VERDICT_ORDER] + ([("dagger", f"{V.DAGGER} |Δ| < 0.5 cm")] if include_small else [])
    for v, lab in items:
        if v == "dagger":
            t = ax.text(x, y, lab, transform=ax.transAxes, fontsize=ps.FS["annot"], ha="left", va="center")
        else:
            V.draw_verdict(ax, x + 0.012, y, v, transform=ax.transAxes)
            t = ax.text(x + 0.028, y, lab, transform=ax.transAxes, fontsize=ps.FS["annot"], ha="left", va="center")
        bb = t.get_window_extent(rend)
        x = inv.transform((bb.x1, bb.y0))[0] + 0.03
    return x


def build():
    ps.use_paper()
    W = 180.0
    # 세로 배치(mm, 아래에서 위로)
    y_bcd, h_bcd = 12.0, 46.0
    items, (ylo, yhi) = layout_a()
    h_a = (yhi - ylo) * U_MM
    y_a = y_bcd + h_bcd + 18.5
    H = round(y_a + h_a + 6.5 + 24.0, 1)
    fig = ps.paper_figure(W, H)
    ax_a, A, R, drawn = panel_a(fig, x_axis=53.0, y_bot=y_a, w_axis=63.0, x_sym=117.0, x_txt=134.5, x_brk=1.5)
    ax_b, B, Rf = panel_b(fig, (14.0, y_bcd, 46.0, h_bcd))
    ax_c, Cc = panel_c(fig, (84.0, y_bcd, 28.0, h_bcd - 13.0), x_sym=113.0)
    ax_d, D, Dr = panel_d(fig, (142.0, y_bcd, 36.0, h_bcd))
    ps.panel_label(ax_a, "a", dx_mm=-51.5, dy_mm=3.4)
    ps.panel_label(ax_b, "b", dx_mm=-12.5, dy_mm=4.2)
    ps.panel_label(ax_c, "c", dx_mm=-18.5, dy_mm=13.0 + 4.2)
    ps.panel_label(ax_d, "d", dx_mm=-12.5, dy_mm=4.2)
    # 범례(1개)
    handles = [Line2D([], [], color=C_BASE, lw=ps.LW["ci_forest"], marker="o", ms=ps.MS["main"] + 0.3, label="Physics baseline (a)"),
               Line2D([], [], color=C_DIRECT, lw=ps.LW["ci_forest"], marker="x", ms=ps.MS["main"] + 0.5, mew=0.9, label="Direct ML (a)"),
               Line2D([], [], ls="none", marker="s", ms=2.4, mfc="#d4d4d4", mec="#8c8c8c", mew=0.4, label="Region value (b)"),
               Line2D([], [], ls="none", marker="o", ms=6.0, mfc="none", mec=ps.GREY["mid"], mew=0.8, label="C2 interval, earlier protocol (b)")]
    handles += [Line2D([], [], ls="none", marker=TMARK_D[t], ms=ps.MS["main"] + 0.4, mfc=C_R1, mec=C_R1, label=f"{TNAME_D[t]} (d)")
                for t in ("Lena", "Canada", "Alaska")]
    ps.legend_below(fig, handles, (2.0, H - 10.0, W - 4.0, 9.0), ncol=4)
    # 기호 설명(범례 객체 아님) 세 줄
    small_any = bool(A.small_effect.astype(str).str.lower().eq("true").any())
    k1 = ps.axes_mm(fig, 4.0, H - 15.0, 172.0, 4.0)
    k1.set_axis_off(); k1.set_xlim(0, 1); k1.set_ylim(0, 1)
    verdict_key6(k1, 0.0, 0.5, include_small=small_any)
    k1.set_gid("verdict_key")
    k2 = ps.axes_mm(fig, 4.0, H - 19.2, 172.0, 4.0)
    k2.set_axis_off(); k2.set_xlim(0, 1); k2.set_ylim(0, 1)
    k2.text(0.026, 0.5, "1, 2", ha="right", va="center", fontsize=ps.FS["annot"], fontweight="bold")
    k2.text(0.032, 0.5, "support class of LGF-F1 and LGF-N1 (a): 1, every judged contrast has higher error than P0 or is equivalent; "
            "2, none has lower error and one or more are undecided", ha="left", va="center", fontsize=ps.FS["annot"])
    k2.set_gid("class_key")
    AB = V.rd("ab_bundle")
    k3 = ps.axes_mm(fig, 4.0, H - 23.4, 172.0, 4.0)
    k3.set_axis_off(); k3.set_xlim(0, 1); k3.set_ylim(0, 1)
    k3.text(0.026, 0.5, "‡", ha="right", va="center", fontsize=ps.FS["annot"])
    k3.text(0.032, 0.5, "two-stage and one-stage CI verdicts differ (c)", ha="left", va="center", fontsize=ps.FS["annot"])
    k3.text(0.37, 0.5, "AB tags (Holm m = 10): " + V.ab_note(AB, ["AB1", "AB2", "AB10"]), ha="left", va="center", fontsize=ps.FS["annot"],
            color=ps.GREY["text2"])
    k3.set_gid("key3")
    cap = caption(A, R, B, Rf, Cc, D, Dr)
    import pandas as pd
    src = dict(a=A, a_registered=R.reset_index(), a_drawn=pd.DataFrame(drawn), b=B, b_ref=Rf, c=Cc,
               c_registered=V.rd("fig6_c_registered"), d=D, d_registered=V.rd("fig6_d_registered"))
    spec_extra = dict(module="fig6_label0_uncertainty.py", panels=4, layout_variant="layout_B(지도 없음, LGU-B2 미결정)",
                      message="no tested direct learner has lower error than P0 at n = 0 (the physics baseline P* has); "
                              "region-held-out intervals: conditional width not established (LGU-B2), hierarchical model not established (LGU-A1)")
    return V.save_v2(fig, NAME, cap, SPEC_ID, src, title="Zero-label transfer across learners and region-held-out prediction intervals.",
                     spec_extra=spec_extra)


def caption(A, R, B, Rf, Cc, D, Dr) -> dict:
    def ci(r, nd=2):
        return f"{ps.fmt_num(r.delta, nd)} [{ps.fmt_num(r.ci_lo, nd)}, {ps.fmt_num(r.ci_hi, nd)}]"
    bens = A[A.key == "B:ens"].iloc[0]
    b4 = B[B.kind == "mean4"].set_index("method")
    ref = Rf[Rf.kind == "mean4"].iloc[0]
    ctx = A[A.key == "catboost_ctx"]
    same_ctx = bool(np.isclose(ctx[ctx.block == "lgt"].delta.iloc[0], ctx[(ctx.block == "lgf_f") & (ctx.variant == "solid")].delta.iloc[0], atol=1e-9))
    return dict(
        definition=("ΔRMSE = RMSE(method) − RMSE(P0) at n = 0, where P0 is the Stefan model with the source least-squares coefficient E0; "
                    "negative is lower error. B:ens averages the scale-calibrated Stefan, Kudryavtsev and CCI anchors; P* is P0 with "
                    "year-matched TDD. Interval scores use α = 0.1. B2 compares nflow-normalized conformal intervals with the same intervals "
                    "after permuting σ within regions (perm.); A1 compares the hierarchical interval (stage iii) with B4, a calibrated "
                    "constant-width interval around P1."),
        statistics=("a, d, block bootstrap (10,000 resamples); c, LGU two-stage bootstrap (1,000 resamples); b, no CI drawn (CIs in Source Data). Points and "
                    "bars are cell-weighted means with 95% CI, four-region pool (P4) unless stated; c divides each difference and its CI by "
                    "the reference score. Symbols copy the four-way verdicts of the source tables, which require both cell-weighted and "
                    f"block-equal-weighted CIs; B:ens ({ci(bens)} cm) is undecided because its block-equal-weighted CI includes zero."),
        panels=("a, rows fixed before the results; Rescale and local GPU runs form separate blocks and are not differenced. Supported "
                "refers to the registered hypotheses that direct ML does not surpass P0 (L1 covers n ≤ 40). LGF-N learners were tuned by "
                "leave-one-source-region-out validation; FT-T-type is a reduced FT-Transformer and multi-head MLP is not the published "
                "TabM." + (" The 10,000-row-context CatBoost gives identical values in LGT and LGF-F." if same_ctx else "") +
                " b, normalizers: const, constant σ; phys, physics prior; cbq, CatBoost quantiles; cfm and nflow, flow models. "
                f"P4 coverage {b4.cov10.min():.2f}–{b4.cov10.max():.2f}; insets enlarge the boxed near-coincident means; C2 interval "
                f"{ref.coverage:.2f}, {ref.width_cm:.0f} cm. c, registered verdicts undecided for B2 and A1; A1 pools Lena, Canada and Alaska (x). "
                f"d, R1 coverage within 0.85–0.95 in "
                f"{int(Dr.c1)} of {int(Dr.n_cells)} cells, all {int(Dr.c2)} narrower than R0 at n = 0; the width ratio has no CI."),
        data=("LG and LGX shards on the Rescale platform (h39, h42); LGT, LGF and LGU runs on the local GPU server (RTX 3090); earlier C2 table. "
              "Values in Source Data."),
    )


if __name__ == "__main__":
    build()
