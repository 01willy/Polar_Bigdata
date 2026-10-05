"""Fig 2 v3. 라벨 수 곡선(그림 명세 FIGURE_SPEC_v3.md 4절, 지침 2–3절과 6.2절).

주장(설명문 첫 문장, 초록 대비 AB4·AB5 규칙 (a)·(b)): With ten labels, Stefan recalibration lowered the four-region mean
error; no further difference was established for residual ML.

패널(170 × 135 mm, 왼쪽 위 원점 mm; 그림 명세 4.2)
  a  주 4지역 평균, 라벨 10개 이하(구성 고정 층화 평균). n = 0 별도 축(6 mm) + log10 축(3, 10).
     연도 정합 Stefan 은 n = 0 축의 점선 짧은 수평선과 95% CI 세로 막대.
  b  레나델타·캐나다 평균, 전체 n. n = 0 별도 축 + log10 축(3–320) + 'All' 별도 축(7 mm).
     연도 정합 Stefan 은 점추정만. 방법 직접 라벨 4개는 b 에만 한 번 단다.
  c–f 레나델타, 캐나다, W Russia, E Russia(n ≥ 3, log10). c, d, f 는 y [−6, 6] cm 공유, e 는 자체 범위.
     후보 셀 수(A 셀 최댓값)를 넘는 n 은 #B8BEC6 + '//' 마스크.
  계열: Recalibrated Stefan(#2b5c8f 실선), Anchor + residual ML(#9a7bc9 실선), Direct ML(#6b7280 파선).
  CI 띠는 앞의 두 계열만(셀 가중 95%, alpha 0.20). n = 0 과 All 별도 축에서는 같은 띠를 세로 조각으로 그린다.
  y 축 이름과 x 축 이름은 그림 전체에 한 번씩 둔다(같은 문구 반복 0, 지침 H13).

자료(등록 원천, 이 모듈은 값을 계산하지 않고 읽어서 그린다)
  data/processed/paper_figs/fig2_pool_curves.csv   edition ∈ {E1_P4_n_le_10, E2_LenaCanada_all_n}, method ∈ {P1, R1, D0}
  data/processed/paper_figs/fig2_pstar_pool.csv    같은 edition(연도 정합 Stefan)
  data/processed/paper_figs/fig2_region_curves.csv target ∈ {Lena, Canada, Russia_W, Russia_E}, method ∈ {P1, R1, D0}, n ≥ 3
  후보 셀 수: paper/claims/D_data_and_design/README.md E27–E30(A 셀 범위).
산출(outputs/figures/paper/v3_restructure/): Fig2.pdf, Fig2.png(600 dpi), Fig2_source_data.csv, Fig2_values.txt.
  설명문 Fig2_legend.md 는 이 모듈이 쓰지 않는다(손으로 쓴 문장, 수치 대조는 Fig2_values.txt).
실행: nice -n 10 python3 scripts/4_visualization/paper_v3/fig2.py            (논문판, 산출 저장)
      nice -n 10 python3 scripts/4_visualization/paper_v3/fig2.py --medium slide --out <폴더>   (덱판 a·b, 지정 폴더에만)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S                                                        # noqa: E402

import matplotlib                                                        # noqa: E402
from matplotlib import font_manager                                      # noqa: E402
from matplotlib.collections import PolyCollection                        # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle                # noqa: E402
from matplotlib.text import Text                                         # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory              # noqa: E402

NAME = "Fig2"
SRC_POOL = S.PAPER_FIGS / "fig2_pool_curves.csv"
SRC_PSTAR = S.PAPER_FIGS / "fig2_pstar_pool.csv"
SRC_REGION = S.PAPER_FIGS / "fig2_region_curves.csv"
ED_A, ED_B = "E1_P4_n_le_10", "E2_LenaCanada_all_n"
N_ALL = -1

# 원천 method 값 → (방법 키, CI 띠 여부). 띠는 재보정 계열 2개만(그림 명세 4.3, 패널당 띠 2개 이하)
SERIES = {"P1": ("recalibrated_stefan", True), "R1": ("anchor_residual", True), "D0": ("direct_ml", False)}
DRAW_ORDER = ["direct_ml", "recalibrated_stefan", "anchor_residual"]
REGIONS = [("Lena", "c"), ("Canada", "d"), ("Russia_W", "e"), ("Russia_E", "f")]
# 후보 라벨 셀 수(A 셀) 범위: paper/claims/D_data_and_design/README.md E27–E30(유효 분할 5개의 최솟값–최댓값)
CANDIDATE_A = {"Lena": (1120, 1737), "Canada": (325, 406), "Russia_W": (14, 16), "Russia_E": (15, 17)}
GRID = [3, 10, 40, 160, 320, 1000]

# y 범위와 눈금(cm). 지역 패널 c, d, f 공유. e 는 R1 n 10 CI 하한 −14.31 cm 를 포함하도록 [−15, 5](명세 [−14, 4] 에서 조정)
Y_POOL = dict(lim=(-4.0, 4.0), ticks=[-4, -2, 0, 2, 4])
Y_REG = dict(lim=(-6.0, 6.0), ticks=[-6, -3, 0, 3, 6])
Y_REG_W = dict(lim=(-15.0, 5.0), ticks=[-15, -10, -5, 0, 5])
X_A = (2.55, 12.0)
X_B = (2.55, 400.0)
X_REG = (2.4, 1400.0)
DODGE = {"recalibrated_stefan": -0.38, "anchor_residual": 0.38, "direct_ml": 0.0}   # 별도 축(n = 0, All) 안의 가로 위치
SLICE_W_MM = 1.3                                                                      # 별도 축의 CI 띠 조각 폭

TEXT = {
    "paper": dict(ylabel="Error change vs source Stefan (cm)", xlabel="Target labels, $n$", head_a="Four-region mean",
                  head_b="Two-region mean", all_tick="All", direction="Lower error",
                  method={k: S.METHOD[k]["short"] for k in ("year_matched_stefan", "recalibrated_stefan", "anchor_residual",
                                                            "direct_ml")},
                  region={k: S.REGION_NAME[k] for k, _ in REGIONS}),
    "slide": dict(ylabel="원천 계수 Stefan 대비 오차 변화 (cm)", xlabel="대상 라벨 수, n", head_a="주 4지역 평균",
                  head_b="레나델타·캐나다 평균", all_tick="전량", direction="오차 감소",
                  method={"year_matched_stefan": "연도 정합 Stefan", "recalibrated_stefan": "재보정 Stefan",
                          "anchor_residual": "재보정 물리 잔차 결합", "direct_ml": "직접 ML"},
                  region={"Lena": "레나델타", "Canada": "캐나다", "Russia_W": "러시아 서부", "Russia_E": "러시아 동부"}),
}

# 매체별 크기(지침 2.2–2.3 논문, slide_archetypes 1.3 슬라이드)
TOK = {
    # v4 토큰(design/style_tokens_v4.json): 논문 FreeSans 7 pt, 선 1.2/0.8/0.6 pt, 표지 4 pt; 슬라이드 13/14 pt, 선 2.2/1.4/1.0 pt, 표지 7 pt
    "paper": dict(fs=7.0, fs_axis=7.0, fs_head=7.0, lw_main=S.LW["main"], lw_aux=S.LW["aux"], lw_zero=S.LW["ref"], ms=S.MS["main"],
                  arrow_ms=5.0, pad=1.5, font=S.FONT_LATIN),
    "slide": dict(fs=S.SLIDE_FONT["direct"], fs_axis=S.SLIDE_FONT["axis_label"], fs_head=S.SLIDE_FONT["direct"], lw_main=S.LW_SLIDE["main"],
                  lw_aux=S.LW_SLIDE["aux"], lw_zero=S.LW_SLIDE["ref"], ms=S.MS_SLIDE["main"], arrow_ms=10.0, pad=4.0, font="Pretendard"),
}


# ---------------------------------------------------------------- 자료
def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def n_tick(n: int) -> str:
    return "All" if n == N_ALL else str(int(n))


def load() -> tuple[pd.DataFrame, dict]:
    """그릴 값 전부를 한 표(tidy)로 모은다. 원천 열 이름과 필터는 그림 명세 4.3 그대로."""
    pool = pd.read_csv(SRC_POOL)
    pst = pd.read_csv(SRC_PSTAR)
    reg = pd.read_csv(SRC_REGION)
    rows = []
    for panel, ed, group in (("a", ED_A, "Four main regions"), ("b", ED_B, "Lena Delta and Canada")):
        q = pool[(pool.edition == ed) & pool.method.isin(list(SERIES))].copy()
        if not ((q.status == "ok").all() and q.composition_complete.astype(bool).all()):
            raise ValueError(f"{ed}: status/composition 점검 실패")
        expect_n = {ED_A: {0, 3, 10}, ED_B: {0, 3, 10, 40, 160, 320, N_ALL}}[ed]
        for m in SERIES:
            got = set(q[q.method == m].n.astype(int))
            if got != expect_n:
                raise ValueError(f"{ed} {m}: n 격자 {sorted(got)} ≠ {sorted(expect_n)}")
        for r in q.itertuples():
            key, band = SERIES[r.method]
            degenerate = bool(np.isclose(r.ci_lo, r.ci_hi))
            rows.append(dict(panel=panel, group=group, series=key, n=int(r.n), value=float(r.delta), lo=float(r.ci_lo),
                             hi=float(r.ci_hi), ci_drawn=bool(band and not degenerate), resamples=int(r.nboot),
                             axis="n0" if r.n == 0 else ("all" if r.n == N_ALL else "log"),
                             src=f"{SRC_POOL.name}: edition=={ed}, method=={r.method}, n=={int(r.n)}; delta, ci_lo, ci_hi"))
        p = pst[pst.edition == ed]
        if len(p) != 1:
            raise ValueError(f"{SRC_PSTAR.name}: {ed} 행 {len(p)}")
        p = p.iloc[0]
        has_ci = bool(np.isfinite(p.pstar_lo) and np.isfinite(p.pstar_hi))
        rows.append(dict(panel=panel, group=group, series="year_matched_stefan", n=0, value=float(p.pstar),
                         lo=float(p.pstar_lo) if has_ci else np.nan, hi=float(p.pstar_hi) if has_ci else np.nan,
                         ci_drawn=has_ci, resamples=10000 if has_ci else 0, axis="n0",
                         src=f"{SRC_PSTAR.name}: edition=={ed}; pstar, pstar_lo, pstar_hi"))
    masks = {}
    for region, panel in REGIONS:
        q = reg[(reg.target == region) & reg.method.isin(list(SERIES)) & (reg.n >= 3)].copy()
        if (q.point_only.astype(str).str.lower() == "true").any() or (q.n_splits_used < 5).any():
            raise ValueError(f"{region}: 점추정 전용 행 또는 분할 5 미만")
        for r in q.itertuples():
            key, band = SERIES[r.method]
            rows.append(dict(panel=panel, group=S.REGION_NAME[region], series=key, n=int(r.n), value=float(r.d_p0),
                             lo=float(r.d_p0_lo), hi=float(r.d_p0_hi), ci_drawn=band, resamples=int(r.nboot), axis="log",
                             src=f"{SRC_REGION.name}: target=={region}, method=={r.method}, n=={int(r.n)}; d_p0, d_p0_lo, d_p0_hi"))
        a_max = CANDIDATE_A[region][1]
        n_all_mean = float(reg[(reg.target == region) & (reg.n == N_ALL)].n_lab.mean())
        if not (CANDIDATE_A[region][0] - 0.5 <= n_all_mean <= a_max + 0.5):
            raise ValueError(f"{region}: 전량 n_lab 평균 {n_all_mean} 이 A 셀 범위 {CANDIDATE_A[region]} 밖")
        drawn_n = set(q.n.astype(int))
        if max(drawn_n) > a_max:
            raise ValueError(f"{region}: 후보 수 {a_max} 를 넘는 n 이 있다")
        if any(g > a_max for g in GRID):
            masks[panel] = dict(region=region, start=float(a_max), a_range=CANDIDATE_A[region], n_all_mean=n_all_mean)
    D = pd.DataFrame(rows)
    return D, masks


# ---------------------------------------------------------------- 그리기 도우미
def _fmt(v: float) -> str:
    return S.fmt_int(v)


def _yticks(ax, ticks, labels=True):
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter((lambda v, p: _fmt(v)) if labels else (lambda v, p: "")))


def _xticks_log(ax, ticks, labelled):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: str(int(round(v))) if int(round(v)) in labelled else ""))


def _single_axis(ax, label):
    """n = 0 또는 'All' 별도 축(눈금 하나)."""
    ax.set_xlim(-1.0, 1.0)
    ax.xaxis.set_major_locator(FixedLocator([0.0]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: label))


def _style_axes(ax, T, left: bool, ylim, yticks, ylabels: bool):
    ax.set_ylim(*ylim)
    _yticks(ax, yticks, labels=ylabels)
    ax.spines["left"].set_visible(left)
    if not left:
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="both", pad=T["pad"])
    ax.patch.set_visible(False)


def _zero(ax, T):
    ln = ax.axhline(0.0, color=S.INK_AUX, lw=T["lw_zero"], ls="-", zorder=1.2)
    ln.set_gid("zero_line")


def _slice(ax, x, lo, hi, color, w_mm):
    """별도 축(n = 0, All)의 CI 띠 조각: 띠와 같은 채움(alpha 0.20, 테두리 없음)."""
    w = w_mm / (ax._w_mm / 2.0)                                           # 축 폭(mm) → 자료 단위(xlim −1–1)
    r = Rectangle((x - w / 2, lo), w, hi - lo, facecolor=color, edgecolor="none", lw=0, alpha=S.CI_BAND_ALPHA, zorder=1.5)
    r.set_gid("ci_slice")
    ax.add_patch(r)
    return r


def _point(ax, x, y, color, T, gid):
    ln, = ax.plot([x], [y], ls="none", marker="o", ms=T["ms"], mfc=color, mec="none", mew=0, zorder=3.2)
    ln.set_gid(gid)
    return ln


def _draw_series(D, panel, axes, T):
    """한 패널의 세 계열. axes = dict(n0=?, log=ax, all=?)."""
    for key in DRAW_ORDER:
        m = S.METHOD[key]
        s = D[(D.panel == panel) & (D.series == key)].sort_values("n")
        g = s[s.axis == "log"]
        ax = axes["log"]
        if len(g):
            if bool(g.ci_drawn.all()):
                pc = ax.fill_between(g.n, g.lo, g.hi, color=m["color"], alpha=S.CI_BAND_ALPHA, lw=0, zorder=1.5)
                pc.set_gid(f"band|{panel}|{key}")
            ln, = ax.plot(g.n, g.value, color=m["color"], ls=m["ls"], lw=T["lw_main"], marker="o", ms=T["ms"],
                          mfc=m["color"], mec="none", mew=0, zorder=3.0 + DRAW_ORDER.index(key) * 0.05)
            ln.set_gid(f"data|{panel}|{key}|log")
        for axis_key in ("n0", "all"):
            r = s[s.axis == axis_key]
            axs = axes.get(axis_key)
            if axs is None or r.empty:
                continue
            r = r.iloc[0]
            x = DODGE[key]
            if r.ci_drawn:
                _slice(axs, x, r.lo, r.hi, m["color"], SLICE_W_MM if T["fs"] < 10 else 3.0)
            _point(axs, x, r.value, m["color"], T, f"data|{panel}|{key}|{axis_key}")


def _draw_pstar(D, panel, ax0, T):
    m = S.METHOD["year_matched_stefan"]
    r = D[(D.panel == panel) & (D.series == "year_matched_stefan")].iloc[0]
    ln, = ax0.plot([-0.85, 0.85], [r.value, r.value], color=m["color"], ls=m["ls"], lw=T["lw_main"], zorder=2.5,
                   solid_capstyle="butt")
    ln.set_gid(f"data|{panel}|year_matched_stefan|n0")
    if r.ci_drawn:
        b, = ax0.plot([0.0, 0.0], [r.lo, r.hi], color=m["color"], ls="-", lw=T["lw_aux"], zorder=2.4, solid_capstyle="butt")
        b.set_gid(f"ci|{panel}|year_matched_stefan|n0")
    return r


def _mask(ax, x0, x1):
    p = ax.axvspan(x0, x1, facecolor=S.BASEMAP["nodata"], edgecolor=S.BASEMAP["hatch"], hatch="//", lw=0, zorder=1.3)
    p.set_gid("mask")
    return p


# ---------------------------------------------------------------- 배치
def geometry(medium: str, fig=None, T=None) -> dict:
    """mm 배치(왼쪽 위 원점). 논문판은 그림 명세 4.2 의 칸을 따른다."""
    if medium == "paper":
        # 그림 명세 4.2: 170 × 약 135 mm, a·b 축 y 6–70(머리·패널 문자 포함 0–70), c–f 축 y 82–124(각 42 mm), b 본 축은 x 144 까지,
        # 직접 라벨 열 x 144–170. 1차 렌더(127 mm, 축 높이 60·38)는 명세보다 작아 2차 검토에서 명세 값으로 되돌렸다.
        g = dict(W=170.0, H=135.0, row1=(6.0, 64.0), row2=(82.0, 42.0),
                 a0=9.5, aL=(17.5, 38.5), b0=62.0, bL=(70.0, 65.0), bA=(137.0, 7.0), w0=6.0, label_x=145.5,
                 reg_x=[9.5, 51.0, 92.5, 134.0], reg_w=34.0,
                 # 왼쪽 열 패널 문자(a, c)는 그림 전체 y 축 이름(x 0.4–2.9 mm) 오른쪽에 둔다
                 letters=dict(a=(4.3, 0.6), b=(58.5, 0.6), c=(4.3, 76.6), d=(45.5, 76.6), e=(87.0, 76.6), f=(128.5, 76.6)),
                 ylabel_x=2.9, xlabel_y=130.6)
        return g
    # 슬라이드판(A2 단독 그림 칸 12.00 × 5.20 in): a, b 만
    g = dict(W=304.8, H=132.0, row1=(13.0, 96.0), a0=27.0, aL=(45.0, 70.0), b0=128.0, w0=13.0, ylabel_x=9.0,
             xlabel_y=124.0)
    g["bL"] = (g["b0"] + 13.0 + 5.0, 0.0)
    g["bA"] = (0.0, 15.0)
    return g


def build(medium: str = "paper"):
    """그림을 만들고 (fig, D, masks, 축 사전)을 돌려준다. 저장은 main() 이 한다."""
    T = dict(TOK[medium])
    X = TEXT[medium]
    S.use_v3(medium)
    if medium == "paper" and hasattr(S, "mathtext_liberation"):
        S.mathtext_liberation()                                          # x 축 이름의 "$n$" 기울임(Liberation Sans Italic)
    if medium == "slide":
        for p in (Path.home() / ".fonts").glob("Pretendard-*.otf"):
            try:
                font_manager.fontManager.addfont(str(p))
            except Exception:                                            # noqa: BLE001
                pass
        if not any(f.name == "Pretendard" for f in font_manager.fontManager.ttflist):
            T["font"] = S.FONT_LATIN
        matplotlib.rcParams["font.sans-serif"] = [T["font"]]
        matplotlib.rcParams["font.weight"] = "normal"
    matplotlib.rcParams["axes.labelpad"] = 2.0 if medium == "paper" else 6.0
    D, masks = load()
    G = geometry(medium)
    fig = S.fig_mm(G["W"], G["H"])
    W, H = G["W"], G["H"]

    def fx(x_mm):
        return x_mm / W

    def fy(y_top_mm):
        return 1.0 - y_top_mm / H

    if medium == "slide":                                                  # 직접 라벨 열 폭을 실측해 b 폭을 정한다
        widths = []
        for k in ("recalibrated_stefan", "anchor_residual", "direct_ml"):
            t = fig.text(0, 0, X["method"][k], fontsize=T["fs"])
            fig.canvas.draw()
            widths.append(t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4)
            t.remove()
        label_w = max(widths) + 1.5
        bA_x = W - 1.0 - label_w - 2.5 - G["bA"][1]
        G["bA"] = (bA_x, G["bA"][1])
        G["bL"] = (G["bL"][0], bA_x - 4.0 - G["bL"][0])
        G["label_x"] = bA_x + G["bA"][1] + 2.5

    top, h = G["row1"]
    ax = {}
    # ---- a
    ax["a0"] = S.axes_mm(fig, G["a0"], top, G["w0"], h)
    ax["aL"] = S.axes_mm(fig, G["aL"][0], top, G["aL"][1], h)
    # ---- b
    ax["b0"] = S.axes_mm(fig, G["b0"], top, G["w0"], h)
    ax["bL"] = S.axes_mm(fig, G["bL"][0], top, G["bL"][1], h)
    ax["bA"] = S.axes_mm(fig, G["bA"][0], top, G["bA"][1], h)
    for k, w in (("a0", G["w0"]), ("b0", G["w0"]), ("bA", G["bA"][1])):
        ax[k]._w_mm = w
    for k in ("a0", "aL", "b0", "bL", "bA"):
        _zero(ax[k], T)
    _style_axes(ax["a0"], T, True, Y_POOL["lim"], Y_POOL["ticks"], True)
    _style_axes(ax["aL"], T, False, Y_POOL["lim"], Y_POOL["ticks"], False)
    _style_axes(ax["b0"], T, True, Y_POOL["lim"], Y_POOL["ticks"], False)
    _style_axes(ax["bL"], T, False, Y_POOL["lim"], Y_POOL["ticks"], False)
    _style_axes(ax["bA"], T, False, Y_POOL["lim"], Y_POOL["ticks"], False)
    _single_axis(ax["a0"], "0")
    _single_axis(ax["b0"], "0")
    _single_axis(ax["bA"], X["all_tick"])
    ax["aL"].set_xlim(*X_A)
    _xticks_log(ax["aL"], [3, 10], {3, 10})
    ax["bL"].set_xlim(*X_B)
    # 슬라이드판 16 pt 에서는 160 과 320 라벨 간격이 0.10 in 미만이라 160 라벨을 뺀다(눈금은 남김, slide_archetypes 1.3)
    _xticks_log(ax["bL"], [3, 10, 40, 160, 320], {3, 10, 40, 160, 320} if medium == "paper" else {3, 10, 40, 320})
    _draw_series(D, "a", dict(n0=ax["a0"], log=ax["aL"]), T)
    _draw_series(D, "b", dict(n0=ax["b0"], log=ax["bL"], all=ax["bA"]), T)
    _draw_pstar(D, "a", ax["a0"], T)
    rb = _draw_pstar(D, "b", ax["b0"], T)

    # ---- 직접 라벨(b 의 선 오른쪽 끝, 한 번씩). 연도 정합 Stefan 은 n = 0 축 짧은 선의 오른쪽 끝
    labels = {}
    tr_all = blended_transform_factory(fig.transFigure, ax["bA"].transData)
    for key in ("recalibrated_stefan", "anchor_residual", "direct_ml"):
        r = D[(D.panel == "b") & (D.series == key) & (D.n == N_ALL)].iloc[0]
        t = fig.text(fx(G["label_x"]), r.value, X["method"][key], transform=tr_all, ha="left", va="center",
                     fontsize=T["fs"], color=S.METHOD[key]["color"])              # v4: 직접 라벨은 계열 색
        t.set_gid("direct_label")
        labels[key] = t
    tr_b0 = blended_transform_factory(fig.transFigure, ax["b0"].transData)
    t = fig.text(fx(G["b0"] + G["w0"] + (1.2 if medium == "paper" else 2.5)), rb.value, X["method"]["year_matched_stefan"],
                 transform=tr_b0, ha="left", va="center", fontsize=T["fs"], color=S.METHOD["year_matched_stefan"]["color"])
    t.set_gid("direct_label")
    labels["year_matched_stefan"] = t

    # ---- 방향 표지: v4 토큰에서 화살촉 주석을 쓰지 않는다(방향은 설명문과 0선의 뜻으로 읽는다)

    # ---- 축 이름(그림 전체에 한 번씩)
    ax["a0"].set_ylabel(X["ylabel"], fontsize=T["fs_axis"])
    row_bottom = (G["row2"][0] + G["row2"][1]) if medium == "paper" else (top + h)
    ax["a0"].yaxis.set_label_coords(fx(G["ylabel_x"]), fy((top + row_bottom) / 2.0), transform=fig.transFigure)

    heads = {}
    if medium == "paper":
        # ---- c–f 지역 패널
        top2, h2 = G["row2"]
        for (region, panel), x0 in zip(REGIONS, G["reg_x"]):
            a = S.axes_mm(fig, x0, top2, G["reg_w"], h2)
            ax[panel] = a
            _zero(a, T)
            yl = Y_REG_W if region == "Russia_W" else Y_REG
            _style_axes(a, T, True, yl["lim"], yl["ticks"], True)
            a.set_xlim(*X_REG)
            _xticks_log(a, GRID, {10, 40, 160, 1000})
            _draw_series(D, panel, dict(log=a), T)
            if panel in masks:
                _mask(a, masks[panel]["start"], X_REG[1])
        ax["c"].set_xlabel(X["xlabel"], fontsize=T["fs_axis"])
        ax["c"].xaxis.set_label_coords(fx((G["reg_x"][0] + G["reg_x"][-1] + G["reg_w"]) / 2.0), fy(G["xlabel_y"]),
                                       transform=fig.transFigure)
        ax["c"].xaxis.label.set_va("top")
        # ---- 패널 문자와 열 머리(같은 행은 같은 높이)
        for letter, (x, y) in G["letters"].items():
            S.panel_letter(fig, x, y, letter).set_gid("panel_letter")
        spans = dict(a=(G["a0"], G["aL"][0] + G["aL"][1]), b=(G["b0"], G["bA"][0] + G["bA"][1]))
        for (region, panel), x0 in zip(REGIONS, G["reg_x"]):
            spans[panel] = (x0, x0 + G["reg_w"])
        head_text = dict(a=X["head_a"], b=X["head_b"], **{p: X["region"][r] for r, p in REGIONS})
        for p, (x0, x1) in spans.items():
            y = G["letters"][p][1]
            heads[p] = fig.text(fx((x0 + x1) / 2.0), fy(y), head_text[p], ha="center", va="top", fontsize=T["fs_head"])
            heads[p].set_gid("column_head")
    else:
        # 슬라이드판: x 축 이름은 a·b 아래 가운데, 패널 머리 2개
        ax["aL"].set_xlabel(X["xlabel"], fontsize=T["fs_axis"])
        ax["aL"].xaxis.set_label_coords(fx((G["a0"] + G["bA"][0] + G["bA"][1]) / 2.0), fy(G["xlabel_y"]),
                                        transform=fig.transFigure)
        ax["aL"].xaxis.label.set_va("top")
        spans = dict(a=(G["a0"], G["aL"][0] + G["aL"][1]), b=(G["b0"], G["bA"][0] + G["bA"][1]))
        for p, (x0, x1) in spans.items():
            heads[p] = fig.text(fx((x0 + x1) / 2.0), fy(2.0), X["head_a" if p == "a" else "head_b"], ha="center", va="top",
                                fontsize=T["fs_head"])
    return fig, D, masks, ax, dict(labels=labels, heads=heads, G=G, T=T)


# ---------------------------------------------------------------- 점검
def plotted_values(fig) -> pd.DataFrame:
    """그림에 실제로 그려진 자료 점(gid 'data|패널|계열|축')을 읽는다."""
    out = []
    for ln in fig.findobj(Line2D):
        gid = ln.get_gid() or ""
        if not gid.startswith("data|"):
            continue
        _, panel, key, axis = gid.split("|")
        xs, ys = ln.get_xdata(), ln.get_ydata()
        if key == "year_matched_stefan":
            out.append(dict(panel=panel, series=key, axis=axis, n=0, y=float(ys[0])))
            continue
        for xv, yv in zip(xs, ys):
            n = int(round(xv)) if axis == "log" else (0 if axis == "n0" else N_ALL)
            out.append(dict(panel=panel, series=key, axis=axis, n=n, y=float(yv)))
    return pd.DataFrame(out)


def clipping_check(D, ax) -> list[str]:
    """그린 값과 CI 가 축 범위 안에 있는지(삼각 표지 0개 조건)."""
    bad = []
    lim = {p: Y_POOL["lim"] for p in "ab"}
    lim.update({p: (Y_REG_W if r == "Russia_W" else Y_REG)["lim"] for r, p in REGIONS})
    for r in D.itertuples():
        lo, hi = lim[r.panel]
        vals = [r.value] + ([r.lo, r.hi] if r.ci_drawn else [])
        if min(vals) < lo or max(vals) > hi:
            bad.append(f"{r.panel} {r.series} n={r.n}: {min(vals):.2f}–{max(vals):.2f} 밖 [{lo}, {hi}]")
    return bad


def overlap_check(fig) -> list[str]:
    """글자끼리, 글자와 자료(선·점·띠)가 겹치는 곳을 찾는다(지침 2.2 '글자 겹침 0')."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_window_extent(rend)) for t in texts]
    issues = []
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        if b1.overlaps(b2) and b1.width > 0 and b2.width > 0:
            issues.append(f"글자 겹침: '{t1.get_text()}' / '{t2.get_text()}'")
    data_lines = [ln for ln in fig.findobj(Line2D) if (ln.get_gid() or "").split("|")[0] in ("data", "ci", "zero_line")]
    bands = [c for c in fig.findobj(PolyCollection) if (c.get_gid() or "").startswith("band|")]
    slices = [p for p in fig.findobj(Rectangle) if p.get_gid() == "ci_slice"]
    for t, b in boxes:
        gid = t.get_gid() or ""
        if gid not in ("direct_label", "direction_label", "column_head"):
            continue
        pts = np.array([(x, y) for x in np.linspace(b.x0, b.x1, 9) for y in np.linspace(b.y0, b.y1, 4)])
        for ln in data_lines:
            xy = ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
            if len(xy) == 1:
                seg = xy
            else:
                seg = np.vstack([np.linspace(xy[i], xy[i + 1], 60) for i in range(len(xy) - 1)])
            r_px = (ln.get_markersize() / 2 + ln.get_linewidth() / 2) * fig.dpi / 72.0
            inside = ((seg[:, 0] >= b.x0 - r_px) & (seg[:, 0] <= b.x1 + r_px) & (seg[:, 1] >= b.y0 - r_px) & (seg[:, 1] <= b.y1 + r_px))
            if inside.any():
                issues.append(f"글자-자료 겹침: '{t.get_text()}' / {ln.get_gid()}")
        for c in bands:
            path = c.get_paths()[0]
            if path.contains_points(pts, transform=c.get_transform()).any():
                issues.append(f"글자-띠 겹침: '{t.get_text()}' / {c.get_gid()}")
        for p in slices:
            if p.get_window_extent(rend).overlaps(b):
                issues.append(f"글자-띠 조각 겹침: '{t.get_text()}'")
    return issues


def tick_spacing_check(fig, ax) -> list[str]:
    """눈금 라벨 사이 간격 1.5 mm 이상(지침 2.12)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    out = []
    for k, a in ax.items():
        labs = sorted([t for t in a.xaxis.get_ticklabels() if t.get_text()], key=lambda t: t.get_window_extent(rend).x0)
        for t1, t2 in zip(labs, labs[1:]):
            gap = (t2.get_window_extent(rend).x0 - t1.get_window_extent(rend).x1) / fig.dpi * 25.4
            if gap < 1.5:
                out.append(f"{k}: '{t1.get_text()}'–'{t2.get_text()}' 간격 {gap:.2f} mm")
    return out


def extent_check(fig) -> list[str]:
    """모든 글자가 캔버스 안(여백 0.3 mm 이상)에 있는지."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    m = 0.3 / 25.4 * fig.dpi
    out = []
    for t in fig.findobj(Text):
        if not (t.get_visible() and t.get_text().strip()):
            continue
        b = t.get_window_extent(rend)
        if b.x0 < m or b.y0 < m or b.x1 > W - m or b.y1 > H - m:
            out.append(f"캔버스 밖 또는 여백 부족: '{t.get_text()}'")
    return out


def source_data_table(D: pd.DataFrame, masks: dict) -> pd.DataFrame:
    """제출용 Source Data(내부 약호·경로 없음)."""
    name = {k: S.METHOD[k]["short"] for k in S.METHOD}
    axis_name = {"n0": "n = 0 axis", "log": "log axis", "all": "All axis"}
    rows = []
    order = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
    ser_order = {"year_matched_stefan": 0, "recalibrated_stefan": 1, "anchor_residual": 2, "direct_ml": 3}
    for r in D.sort_values(["panel", "series", "n"], key=lambda s: s.map(order) if s.name == "panel" else (
            s.map(ser_order) if s.name == "series" else s.where(s != N_ALL, 10 ** 6))).itertuples():
        weighting = ("fixed-composition stratified mean of cell-weighted region values" if r.panel in "ab"
                     else "cell-weighted mean over five splits")
        if r.series == "year_matched_stefan" and r.panel == "b":
            weighting = "mean of the two region values (point estimate)"
        rows.append({"panel": r.panel, "pool_or_region": r.group, "series": name[r.series], "target_labels_n": n_tick(r.n),
                     "x_axis": axis_name[r.axis], "error_change_cm": round(r.value, 4),
                     "ci95_low_cm": round(r.lo, 4) if np.isfinite(r.lo) else "",
                     "ci95_high_cm": round(r.hi, 4) if np.isfinite(r.hi) else "",
                     "ci_drawn": "yes" if r.ci_drawn else "no", "estimate": weighting,
                     "ci_type": "95% block bootstrap, cell-weighted" if np.isfinite(r.lo) else "",
                     "resamples": r.resamples if np.isfinite(r.lo) else ""})
    for panel, m in masks.items():
        rows.append({"panel": panel, "pool_or_region": S.REGION_NAME[m["region"]], "series": "Masked range",
                     "target_labels_n": f"above {m['start']:.0f}", "x_axis": "log axis", "error_change_cm": "",
                     "ci95_low_cm": "", "ci95_high_cm": "", "ci_drawn": "",
                     "estimate": f"candidate label cells per split {m['a_range'][0]} to {m['a_range'][1]}",
                     "ci_type": "", "resamples": ""})
    return pd.DataFrame(rows)


def write_values(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 수치 대조(Fig2_values.txt)
LGW_BUNDLE = S.ROOT / "data" / "processed" / "lgw" / "lgw_bundle.csv"          # C2 README 2.1 A6, A7, CA1 의 원천
LGX_TESTS = S.ROOT / "data" / "processed" / "lgx" / "lgx_tests.csv"            # 연도 정합 Stefan CI(L29 MEAN)
LG_TARGETS = S.ROOT / "results" / "rescale_lg" / "data" / "processed" / "lg" / "lg_targets.csv"   # D README E27–E30
C2_README = S.ROOT / "paper" / "claims" / "C2_bias_diagnosis" / "README.md"
D_README = S.ROOT / "paper" / "claims" / "D_data_and_design" / "README.md"
LEGEND_MD = S.OUT / f"{NAME}_legend.md"
MEAN4 = "MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"


def _chk(name, ok, detail):
    return dict(name=name, ok=bool(ok), detail=detail)


def value_checks(D: pd.DataFrame, sd_path: Path, G: dict, fig) -> list[dict]:
    out = []
    # 1. 표(tidy)와 원천 CSV 를 독립 조회로 다시 대조
    pool = pd.read_csv(SRC_POOL)
    pst = pd.read_csv(SRC_PSTAR)
    reg = pd.read_csv(SRC_REGION)
    inv = {v[0]: k for k, v in SERIES.items()}
    diffs = []
    for r in D.itertuples():
        if r.series == "year_matched_stefan":
            q = pst[pst.edition == (ED_A if r.panel == "a" else ED_B)].iloc[0]
            raw = (q.pstar, q.pstar_lo, q.pstar_hi)
        elif r.panel in "ab":
            q = pool[(pool.edition == (ED_A if r.panel == "a" else ED_B)) & (pool.method == inv[r.series]) & (pool.n == r.n)]
            assert len(q) == 1
            q = q.iloc[0]
            raw = (q.delta, q.ci_lo, q.ci_hi)
        else:
            region = [rg for rg, p in REGIONS if p == r.panel][0]
            q = reg[(reg.target == region) & (reg.method == inv[r.series]) & (reg.n == r.n)]
            assert len(q) == 1
            q = q.iloc[0]
            raw = (q.d_p0, q.d_p0_lo, q.d_p0_hi)
        for a, b in zip((r.value, r.lo, r.hi), raw):
            if np.isfinite(a) or np.isfinite(b):
                diffs.append(abs(float(a) - float(b)))
    out.append(_chk("표 = 원천 CSV(독립 조회)", max(diffs) == 0.0, f"{len(D)}행, 값·CI {len(diffs)}개, 최대 차 {max(diffs):.3g} cm"))
    # 2. Source Data 왕복
    sd = pd.read_csv(sd_path)
    sd = sd[sd.series != "Masked range"].copy()
    name_to_key = {S.METHOD[k]["short"]: k for k in S.METHOD}
    sd["series_key"] = sd.series.map(name_to_key)
    sd["n"] = sd.target_labels_n.map(lambda s: N_ALL if s == "All" else int(s))
    m = sd.merge(D, left_on=["panel", "series_key", "n"], right_on=["panel", "series", "n"], how="outer", indicator=True)
    dmax = float(np.max(np.abs(m.error_change_cm.astype(float) - m.value)))
    out.append(_chk("Source Data = 표(반올림 4자리)", (m["_merge"] == "both").all() and dmax <= 5e-5,
                    f"{len(sd)}행 일치, 최대 차 {dmax:.2g} cm"))
    # 3. 설명문 수치: 초록 대비 AB4·AB5(C2 README 2.1 A6, A7, CA1 의 원천)
    b = pd.read_csv(LGW_BUNDLE)
    ab4 = b[(b.ab == "AB4") & (b.target == MEAN4)].iloc[0]
    ab5 = b[(b.ab == "AB5") & (b.target == MEAN4)].iloc[0]
    c4 = b[(b.ab == "AB4") & (b.target == "Canada|x")].iloc[0]
    out.append(_chk("AB4 4지역 평균 −2.45 [−3.04, −1.88], Holm P 0.001, 보정 전 P 0.0001",
                    (f"{ab4.delta:.2f}", f"{ab4.ci_lo:.2f}", f"{ab4.ci_hi:.2f}", f"{ab4.holm_p:.3f}", f"{ab4.holm_input_p:.4f}")
                    == ("-2.45", "-3.04", "-1.88", "0.001", "0.0001"),
                    f"lgw_bundle.csv ab==AB4, target==MEAN: delta {ab4.delta:.4f}, CI [{ab4.ci_lo:.4f}, {ab4.ci_hi:.4f}], "
                    f"holm_p {ab4.holm_p}, holm_input_p {ab4.holm_input_p}, nboot {ab4.nboot}"))
    out.append(_chk("AB5 4지역 평균 −0.18 [−0.53, −0.01], Holm P 0.136, 보정 전 P 0.034",
                    (f"{ab5.delta:.2f}", f"{ab5.ci_lo:.2f}", f"{ab5.ci_hi:.2f}", f"{ab5.holm_p:.3f}", f"{ab5.holm_input_p:.3f}")
                    == ("-0.18", "-0.53", "-0.01", "0.136", "0.034"),
                    f"lgw_bundle.csv ab==AB5, target==MEAN: delta {ab5.delta:.4f}, CI [{ab5.ci_lo:.4f}, {ab5.ci_hi:.4f}], "
                    f"holm_p {ab5.holm_p}, holm_input_p {ab5.holm_input_p}"))
    out.append(_chk("AB4 캐나다 +2.26 [0.71, 2.94] 열세", (f"{c4.delta:.2f}", f"{c4.ci_lo:.2f}", f"{c4.ci_hi:.2f}", c4.verdict4)
                    == ("2.26", "0.71", "2.94", "열세"), f"lgw_bundle.csv ab==AB4, target==Canada|x: {c4.delta:.4f} "
                    f"[{c4.ci_lo:.4f}, {c4.ci_hi:.4f}], verdict4 {c4.verdict4}"))
    v4 = dict(zip(b[(b.ab == "AB4") & b.target.str.endswith("|x")].target, b[(b.ab == "AB4") & b.target.str.endswith("|x")].verdict4))
    w4 = b[(b.ab == "AB4") & (b.target == "Russia_W|x")].iloc[0]
    out.append(_chk("AB4 지역 판정: 오차 감소는 W Russia 하나(11.96 cm)", v4 == {"Lena|x": "미결정", "Canada|x": "열세",
                                                                       "Russia_W|x": "우세", "Russia_E|x": "미결정"}
                    and f"{-w4.delta:.2f}" == "11.96", f"verdict4 {v4}; W Russia delta {w4.delta:.4f}"))
    readme = C2_README.read_text(encoding="utf-8")
    out.append(_chk("C2 README 2.1 문자열(A6, A7, CA1)", all(s in readme for s in ("−2.45 [−3.04, −1.88]", "−0.18 [−0.53, −0.01]",
                                                                                 "+2.26 [0.71, 2.94]")), "문자열 포함 여부"))
    # 그림의 풀 곡선(E1, P1 n 10)과 AB4 값의 관계: 같은 대비, 재표집 표가 다르다
    p1 = D[(D.panel == "a") & (D.series == "recalibrated_stefan") & (D.n == 10)].iloc[0]
    r1 = D[(D.panel == "a") & (D.series == "anchor_residual") & (D.n == 10)].iloc[0]
    out.append(_chk("그림 a 의 재보정 n 10 점추정 = AB4 점추정", abs(p1.value - ab4.delta) < 1e-6,
                    f"그림 {p1.value:.4f} [{p1.lo:.4f}, {p1.hi:.4f}] / AB4 {ab4.delta:.4f} [{ab4.ci_lo:.4f}, {ab4.ci_hi:.4f}]. "
                    f"CI 끝값 차 최대 {max(abs(p1.lo - ab4.ci_lo), abs(p1.hi - ab4.ci_hi)):.3f} cm(재표집 표가 다르다. 설명문은 AB4 값)"))
    out.append(_chk("그림 a 의 R1 − P1(n 10) = AB5 점추정", abs((r1.value - p1.value) - ab5.delta) < 1e-6,
                    f"{r1.value - p1.value:.4f} / {ab5.delta:.4f}"))
    # 4. W Russia 값(설명문 '−12.5 cm')과 e 의 y 범위
    w = D[(D.panel == "e") & (D.series == "anchor_residual") & (D.n == 10)].iloc[0]
    out.append(_chk("W Russia 앵커 + 잔차 n 10 = −12.5 cm(설명문)", f"{w.value:.1f}" == "-12.5",
                    f"{w.value:.4f} [{w.lo:.4f}, {w.hi:.4f}]; e 의 y 범위 {Y_REG_W['lim']}"))
    # 5. 후보 셀 수(D README E27–E30 와 그 원천)
    rd = D_README.read_text(encoding="utf-8")
    ok_readme = all(f"{lo}–{hi}" in rd for lo, hi in CANDIDATE_A.values())
    det = f"D README 문자열 {'일치' if ok_readme else '불일치'}"
    ok_src = True
    if LG_TARGETS.exists():
        t = pd.read_csv(LG_TARGETS)
        t = t[(t.part == "cpu") & (t["mode"] == "x") & (t.valid.astype(str).str.lower() == "true")]
        got = {k: (int(t[t.target == k].n_A.min()), int(t[t.target == k].n_A.max())) for k in CANDIDATE_A}
        ok_src = got == CANDIDATE_A
        det += f"; lg_targets.csv n_A 범위 {got}"
    out.append(_chk("후보 라벨 셀 수(마스크 시작점)", ok_readme and ok_src, det))
    # 6. 재표집 횟수와 분할 수
    nb_pool = sorted(pool[pool.edition.isin([ED_A, ED_B])].nboot.unique().tolist())
    nb_reg = sorted(reg.nboot.unique().tolist())
    lx = pd.read_csv(LGX_TESTS)
    lq = lx[(lx.test_id == "L29") & (lx.contrast == "P0@tddm-P0|n0") & (lx.target == MEAN4)].iloc[0]
    out.append(_chk("재표집 횟수: a·b 10000, c–f 1000, 연도 정합 Stefan(a) 10000",
                    nb_pool == [10000] and nb_reg == [1000] and int(lq.nboot) == 10000 and abs(lq.delta - D[(D.panel == "a") & (
                        D.series == "year_matched_stefan")].value.iloc[0]) < 1e-9,
                    f"pool nboot {nb_pool}, region nboot {nb_reg}, lgx_tests L29 MEAN nboot {int(lq.nboot)} "
                    f"(delta {lq.delta:.4f} [{lq.ci_lo:.4f}, {lq.ci_hi:.4f}], ci_dependence {lq.ci_dependence})"))
    out.append(_chk("지역 곡선 분할 수 5", set(reg[reg.target.isin(CANDIDATE_A)].n_splits_used.unique()) == {5},
                    f"n_splits_used {sorted(reg.n_splits_used.unique())}"))
    # 7. 설명문 단어 수와 문장 점검
    if LEGEND_MD.exists():
        txt = LEGEND_MD.read_text(encoding="utf-8")
        body = [ln for ln in txt.splitlines() if ln.strip() and not ln.startswith("#") and not ln.startswith("<!--")]
        legend = " ".join(body).replace("**", "")
        nw = len(legend.split())
        ta = S.text_audit(legend)
        file_words = len(txt.split())
        out.append(_chk("설명문 350단어 이하(파일 전체 포함)", nw <= 350 and file_words <= 350,
                        f"설명문 {nw}단어, 파일 전체 {file_words}단어"))
        bad = {k: v for k, v in ta.items() if v}
        out.append(_chk("설명문 문장 점검(대시·금지어·가운뎃점 숫자·네 자리 쉼표·내부 약호)", not bad, f"{bad or '0건'}"))
        out.append(_chk("설명문 첫 문장이 'This figure' 로 시작하지 않음", not legend.lower().startswith("this figure"), legend[:60]))
    else:
        out.append(_chk("설명문 파일", False, f"{LEGEND_MD} 없음"))
    return out


# 3형(tritan) Machado 2009 severity 1.0 행렬. src/polar/cvd.py 에는 1형·2형만 있다(지침 8절 3)
TRITAN = np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]])


def cvd_pairs() -> list[str]:
    """F-09: 이 그림에 공존하는 색 4개의 쌍별 ΔE00(정상, 1형, 2형, 3형)과 흰 바탕 대비."""
    sys.path.insert(0, str(S.ROOT / "src"))
    from polar import cvd
    cols = {k: S.METHOD[k]["color"] for k in ("recalibrated_stefan", "anchor_residual", "direct_ml", "source_stefan")}

    def sim(rgb, kind):
        if kind == "normal":
            return rgb
        M = {"tritan": TRITAN, "deut": cvd.DEU, "prot": cvd.PRO}[kind]
        return cvd._delin(np.clip(cvd._lin(np.asarray(rgb, float)) @ M.T, 0, 1))
    lines, mins = [], []
    for (k1, c1), (k2, c2) in itertools.combinations(cols.items(), 2):
        d = {kind: cvd.de00(cvd.lab(sim(cvd.hex2rgb(c1), kind)), cvd.lab(sim(cvd.hex2rgb(c2), kind)))
             for kind in ("normal", "prot", "deut", "tritan")}
        mins.append(min(d.values()))
        lines.append(f"- {S.METHOD[k1]['short']} {c1} / {S.METHOD[k2]['short']} {c2}: "
                     + ", ".join(f"{kk} {vv:.1f}" for kk, vv in d.items()))
    lines.append(f"- 최소 ΔE00 {min(mins):.1f}(기준 12 이상). 흰 바탕 대비: "
                 + ", ".join(f"{S.METHOD[k]['short']} {cvd.contrast(c):.2f}" for k, c in cols.items()) + "(기준 3:1 이상)")
    lines.append("- 흑백 렌더(사람 판정): Direct ML 은 파선으로 구분된다. Recalibrated Stefan 과 Anchor + residual ML 은 지침 2.4 에서"
                 " 둘 다 실선이라 명도(L* 38, 57)와 선 끝 직접 라벨로만 구분된다. Year-matched Stefan 은 점선이다.")
    return lines


def values_report(summary: dict, vals: list[dict], D: pd.DataFrame, masks: dict, G: dict) -> list[str]:
    aud = summary["audit"]
    L = ["Fig 2 v3 수치·점검 기록", "",
         "작성: scripts/4_visualization/paper_v3/fig2.py 실행 때 자동 생성. 그림 명세 FIGURE_SPEC_v3.md 4절, 지침 2–3절과 6.2절.", ""]
    L += ["[1] 원천 파일(등록)과 sha256"]
    for p in (SRC_POOL, SRC_PSTAR, SRC_REGION, LGW_BUNDLE, LGX_TESTS):
        L.append(f"- {p.relative_to(S.ROOT)}  {sha256(p)[:16]}…")
    L += ["- 행 필터: fig2_pool_curves edition ∈ {E1_P4_n_le_10 (a), E2_LenaCanada_all_n (b)}, method ∈ {P1, R1, D0};",
          "  fig2_pstar_pool 같은 edition; fig2_region_curves target ∈ {Lena, Canada, Russia_W, Russia_E}, method ∈ {P1, R1, D0}, n ≥ 3.",
          "- 계열 대응(코드 안에서만): P1 = Recalibrated Stefan, R1 = Anchor + residual ML, D0 = Direct ML, P* = Year-matched Stefan.",
          f"- 그린 값 {len(D)}행(패널별 {D.groupby('panel').size().to_dict()}).", ""]
    L += ["[2] 대조 결과"]
    L.append(f"- 그림 객체의 자료 점 = 표: 최대 차 {summary['plotted_vs_table_max_abs_diff_cm']:.3g} cm, 짝 없는 점 {summary['unmatched']}개"
             f" ({'통과' if summary['unmatched'] == 0 and summary['plotted_vs_table_max_abs_diff_cm'] == 0 else '실패'})")
    for c in vals:
        L.append(f"- {'통과' if c['ok'] else '실패'}: {c['name']}. {c['detail']}")
    L.append(f"- 축 범위 밖 값(삼각 표지 필요): {summary['clipping'] or '0건'}")
    L.append("")
    L += ["[3] 그림 점검(지침 부록 A.1 audit_v3, A.2 pdf_audit, 7.1)"]
    L.append(f"- audit_v3: 글자 크기 {sorted(aud['sizes'])} pt, 문자 수 {aud['chars']}(한도 600), 얇은 선 {aud['thin_lines'] or '0'},"
             f" 제목 {aud['titles'] or '0'}, 글씨 상자 {aud['boxed_text']}, 내부 약호 {aud['codes'] or '0'}, 네 자리 쉼표 "
             f"{aud['comma4'] or '0'}, 긴 라벨 {aud['long_labels'] or '0'}, 그림 안 수치 {aud['loose_numbers'] or '0'};"
             f" 실패 항목 {aud['fails'] or '없음'}")
    pa = summary["pdf"]
    L.append(f"- pdf_audit: {pa['width_mm']} × {pa['height_mm']} mm, 문자 {pa['chars']}, 크기 분포 {pa['sizes']}, 5 pt 미만 {pa['lt5']},"
             f" 글꼴 계열 {pa['families']}, Type 3 {'있음' if pa['type3'] else '없음'}")
    for ln in summary["fonts"].strip().splitlines()[2:]:
        L.append(f"- pdffonts: {' '.join(ln.split()[:2])} 내장 {ln.split()[-5]}")
    if aud.get("audit_fig") is not None:
        af = aud["audit_fig"]
        L.append(f"- digest 5.3 audit_fig: 글씨 상자 {af['text_boxes']}, 곡선 화살표 {af['curved']}, 사선 화살표 {af['diagonal']}"
                 f"(방향 표지 화살표 1개는 수직), 긴 라벨 {af['long_labels'] or '0'}, 제목 {af['titles'] or '0'}")
    L.append(f"- 글자 겹침·글자와 자료 겹침: {summary['overlaps'] or '0건'}")
    L.append(f"- 눈금 라벨 간격 1.5 mm 미만: {summary['tick_spacing'] or '0건'}")
    L.append(f"- 캔버스 밖 글자: {summary['extent'] or '0건'}")
    L.append("- 직접 라벨 실측 폭(mm): " + ", ".join(f"{S.METHOD[k]['short']} {v:.1f}" for k, v in summary['label_widths_mm'].items())
             + f"; 라벨 열 오른쪽 끝 {summary['label_right_mm']:.1f} mm(캔버스 170 mm)")
    W, H = G["W"], G["H"]
    a_area = (G["aL"][0] + G["aL"][1]) * (G["row1"][0] + G["row1"][1] + 6.0)
    b_area = (G["bA"][0] + G["bA"][1] - 58.5) * (G["row1"][0] + G["row1"][1] + 6.0)
    L.append(f"- 주 패널 면적(F-17, 슬롯 기준, 직접 라벨 열 제외): a {a_area:.0f}, b {b_area:.0f} mm², 합 {100 * (a_area + b_area) / (W * H):.0f}%"
             f"(기준 35% 이상), 가장 큰 패널 b")
    L.append("")
    L += ["[4] 판단과 명세에서 바꾼 점([판단])",
          f"- e(W Russia) y 범위를 명세의 [−14, 4] cm 에서 {list(Y_REG_W['lim'])} cm 로 넓혔다. 앵커 + 잔차 n 10 의 CI 하한 "
          f"{D[(D.panel == 'e') & (D.series == 'anchor_residual') & (D.n == 10)].lo.iloc[0]:.2f} cm 가 −14 밖이라 띠가 잘리기 때문이다."
          " 축 밖 값은 0건이고 삼각 표지는 쓰지 않았다.",
          "- y 축 이름(\"Error change vs source Stefan (cm)\")과 x 축 이름(\"Target labels, n\")은 그림 전체에 한 번씩 두었다."
          " 여섯 패널이 같은 양을 쓰고, 패널마다 두면 같은 문구가 반복된다(지침 H13, F-04).",
          "- a·b 위에 열 머리 \"Four-region mean\", \"Two-region mean\" 을 두었다(c–f 의 지역 이름 열 머리와 같은 형식)."
          " 지역 이름을 a·b 머리에 다시 쓰지 않아 반복을 피했다. 어느 지역인지는 설명문에 쓴다.",
          "- a 와 b 는 같은 y 범위 [−4, 4] cm 를 쓰고 b 의 y 눈금 라벨은 생략했다(같은 양, 같은 기준선).",
          "- n = 0 축과 All 축의 CI 는 곡선 띠와 같은 채움(alpha 0.20)의 세로 조각(폭 1.3 mm)으로 그렸다. 세 계열은 이 별도 축 안에서만"
          " 가로로 조금 비켜 두었다(재보정 왼쪽, 앵커 + 잔차 오른쪽, 직접 ML 가운데). 연도 정합 Stefan 의 a CI 는 1.0 pt 세로 막대다.",
          "- 연도 정합 Stefan 직접 라벨은 b 의 n = 0 축 짧은 점선 오른쪽 끝에 달았다. 그 값은 n = 0 에만 있어 오른쪽 라벨 열에 두면"
          f" 기호와 떨어진다. 나머지 세 라벨은 All 축 점과 같은 높이로 라벨 열(x {G['label_x']} mm, 명세 4.2 의 144–170 mm 열)에 두었다.",
          f"- 캔버스 {G['W']:.0f} × {G['H']:.0f} mm, a·b 축 높이 {G['row1'][1]:.0f} mm(y {G['row1'][0]:.0f}–{G['row1'][0] + G['row1'][1]:.0f}),"
          f" c–f 축 {G['reg_w']:.0f} × {G['row2'][1]:.0f} mm(y {G['row2'][0]:.0f}–{G['row2'][0] + G['row2'][1]:.0f}); 명세 4.2 의 c–f 폭 38 mm 는"
          " y 눈금 라벨 자리를 포함한 슬롯 폭으로 읽었다(축 34 mm + 라벨 4 mm).",
          "- 마스크 시작점은 지역별 후보 라벨 셀(A 셀) 수의 최댓값이다(W Russia 16, E Russia 17, Canada 406;"
          " D README E28–E30). 이보다 큰 n 은 어떤 분할에도 없다. 전량 행의 평균 라벨 수(15.4, 15.6, 361.8)는 이 범위 안이다.",
          "- 방향 표지 \"Lower error\" 와 아래 방향 화살표(1.0 pt, #4d4d4d)는 a 본 축 왼쪽 아래 빈 곳에 하나만 두었다.",
          "- 패널 a·c 의 패널 문자는 그림 전체 y 축 이름 오른쪽(x 4.3 mm)에 두었다. 같은 행의 패널 문자는 같은 높이다.",
          "- dash 패턴은 공용 모듈(style.py)의 rc 설정을 따른다(matplotlib 기본 lines.scale_dashes, 선 굵기에 비례). 다른 v3 그림과 같다.",
          "- 설명문 수치는 원고 명세(paper/manuscript/MANUSCRIPT_SPEC.md R3 첫 문장)와 같은 소수 둘째 자리로 썼다. 지침 4.5 의"
          " 자릿수 규칙(|Δ| ≥ 1 cm 는 소수 첫째 자리)과 다르다. 원고 전체에서 한 규칙으로 맞추는 결정이 필요하다.",
          "- 그림 a 의 재보정 n 10 CI(풀 곡선 표, [−3.04, −1.86])와 설명문의 AB4 CI([−3.04, −1.88], lgw_bundle)는 끝값이 최대 0.02 cm"
          " 다르다. 같은 점추정에 재표집 표가 다르다. 설명문은 초록 대비(AB4)의 값을 쓴다.", ""]
    L += ["[5] 색각·흑백(F-09)"] + cvd_pairs() + [""]
    L += ["[6] 마스크와 후보 수"]
    for p, m in masks.items():
        L.append(f"- 패널 {p}({S.REGION_NAME[m['region']]}): n > {m['start']:.0f} 마스크, A 셀 {m['a_range'][0]}–{m['a_range'][1]},"
                 f" 전량 평균 {m['n_all_mean']:.1f}")
    L.append("")
    L += ["[7] 산출 파일"] + [f"- {p}" for p in summary["paths"]] + [f"- {S.OUT / (NAME + '_legend.md')}(손으로 작성)", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medium", default="paper", choices=["paper", "slide"])
    ap.add_argument("--out", default=None, help="저장 폴더(슬라이드판은 필수, 논문판 기본은 v3_restructure)")
    args = ap.parse_args()
    if args.medium == "slide":
        if not args.out:
            raise SystemExit("슬라이드판은 --out 폴더를 지정한다(v3_restructure 에는 쓰지 않는다)")
        fig, D, masks, ax, extra = build("slide")
        p = Path(args.out) / f"{NAME}_slide_ab.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)                                       # 배치 크기 그대로 300 dpi(경로 B)
        print("saved", p, "overlaps:", overlap_check(fig), "tick spacing:", tick_spacing_check(fig, ax))
        return
    out_dir = Path(args.out) if args.out else S.OUT
    fig, D, masks, ax, extra = build("paper")
    G = extra["G"]

    # ---- 점검(저장 전)
    aud = S.audit_v3(fig)
    aud["audit_fig"] = S.audit_fig(fig) if hasattr(S, "audit_fig") else None            # digest 5.3(상자·곡선·사선 화살표)
    pv = plotted_values(fig)
    mrg = pv.merge(D[["panel", "series", "axis", "n", "value"]], on=["panel", "series", "axis", "n"], how="outer", indicator=True)
    unmatched = mrg[mrg["_merge"] != "both"]
    max_diff = float(np.nanmax(np.abs(mrg.y - mrg.value))) if len(mrg) else np.nan
    clip = clipping_check(D, ax)
    ovl = overlap_check(fig)
    tsp = tick_spacing_check(fig, ax)
    ext = extent_check(fig)
    # 직접 라벨 열 오른쪽 끝(mm)
    rend = fig.canvas.get_renderer()
    lab_right = max(t.get_window_extent(rend).x1 for t in extra["labels"].values()) / fig.dpi * 25.4
    lab_widths = {k: t.get_window_extent(rend).width / fig.dpi * 25.4 for k, t in extra["labels"].items()}

    paths = S.save_fig(fig, NAME, out_dir, formats=("pdf", "png"))
    sd = source_data_table(D, masks)
    sd_path = out_dir / f"{NAME}_source_data.csv"
    sd.to_csv(sd_path, index=False, encoding="utf-8")

    pdf = paths[0]
    pa = S.pdf_audit(pdf)
    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    summary = dict(audit=aud, plotted_vs_table_max_abs_diff_cm=max_diff, unmatched=len(unmatched), clipping=clip,
                   overlaps=ovl, tick_spacing=tsp, extent=ext, label_right_mm=lab_right, label_widths_mm=lab_widths,
                   pdf=pa, fonts=fonts, masks=masks, n_rows=len(D), paths=[str(p) for p in paths] + [str(sd_path)])
    vals = value_checks(D, sd_path, G, fig)
    write_values(out_dir / f"{NAME}_values.txt", values_report(summary, vals, D, masks, G))
    print(json.dumps({k: (sorted(v) if isinstance(v, set) else v) for k, v in summary.items() if k != "audit"}, default=str,
                     ensure_ascii=False, indent=1))
    print("audit_v3:", {k: (sorted(v) if isinstance(v, set) else v) for k, v in aud.items()})
    print("value checks failed:", [c["name"] for c in vals if not c["ok"]])
    return summary


if __name__ == "__main__":
    main()
