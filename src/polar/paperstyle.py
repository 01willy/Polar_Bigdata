"""논문 그림 공용 스타일(Sci Rep·Nature 계열 규격). 정본 스펙: figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §1.

모든 논문 그림(scripts/4_visualization/paper/fig*.py)은 이 모듈의 상수·헬퍼만 쓴다.
    from polar import paperstyle as ps
    ps.use_paper()
    fig = ps.paper_figure(180, 118)
    ax = ps.slot_mm(fig, x_mm=0, y_mm=64, label_mm=14, w_axis_mm=36, h_mm=40)   # 좌하 원점 mm 절대 좌표
    ps.plot_curve(ax, n, d, method="residual")                                   # Δ 축이면 0선·gid 자동
    ps.label_panels([ax, ...])                                                   # 굵은 소문자 a, b, …
    ps.save_figure(fig, "outputs/figures/paper/Fig3_label_budget")               # PDF(Type 42) + 600 dpi PNG

설계 규칙 요약
  - 배치는 mm 절대 좌표(axes_mm·slot_mm)만 쓴다. 저장은 bbox tight 를 쓰지 않으므로 캔버스 폭이 그대로 180 mm 가 된다.
    캔버스 밖으로 나간 글자는 qa_check(_common) 가 '잘림'으로 실패 처리한다.
  - 방법 색(METHOD)은 전 그림 고정. 지역은 색이 아니라 마커(REGION_MARKER)로만 구분한다.
  - Δ 축은 mark_delta_axis 로 gid 를 달고 DELTA_ZERO 0선을 반드시 그린다(forest·plot_curve 가 자동 처리).
  - 빈 마커(mfc 흰색)는 그림당 한 의미만 갖는다(기본 = 블록 등가중 채점).
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle          # noqa: F401  (Patch: 범례 핸들용 재노출)

# ---------------------------------------------------------------- 1) 규격 상수
MM = 1 / 25.4                                    # mm → inch
W2_MM, W1_MM, HMAX_MM = 180.0, 88.0, 200.0
W2, W1, HMAX = W2_MM * MM, W1_MM * MM, HMAX_MM * MM
MIN_FONT_PT = 6.5
FS = dict(base=7.5, label=7.5, tick=7.0, legend=6.5, annot=6.5, panel=9.0)   # §1.4 글자 크기(pt)
LW = dict(axis=0.6, tick=0.5, main=1.0, ref=0.7, ci=0.8, ci_forest=1.2, marker_edge=0.6)
MS = dict(main=3.0, mean=3.6, point=1.8)          # 마커 크기(pt)
POINT_ALPHA = 0.35
BAND_ALPHA = 0.18

FONT_STACK = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
PAPER_RC = {
    "font.family": "sans-serif", "font.sans-serif": FONT_STACK, "font.size": FS["base"],
    "axes.titlesize": FS["base"], "axes.labelsize": FS["label"], "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "legend.fontsize": FS["legend"], "legend.title_fontsize": FS["legend"], "figure.titlesize": FS["base"],
    "axes.linewidth": LW["axis"], "axes.edgecolor": "#000000", "axes.labelcolor": "#000000", "axes.labelpad": 2.0,
    "xtick.major.width": LW["tick"], "ytick.major.width": LW["tick"], "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
    "xtick.color": "#000000", "ytick.color": "#000000",
    "lines.linewidth": LW["main"], "lines.markersize": MS["main"], "lines.markeredgewidth": LW["marker_edge"],
    "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt",
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5, "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.2, "legend.borderpad": 0.2, "legend.labelspacing": 0.35,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.unicode_minus": True,
    # 수식 글꼴도 Liberation Sans 로 고정(기본 dejavusans 는 PDF 에 DejaVu 를 임베드한다)
    "mathtext.fontset": "custom", "mathtext.default": "regular",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    "mathtext.sf": "Liberation Sans", "mathtext.cal": "Liberation Sans", "mathtext.tt": "Liberation Mono",
    "mathtext.fallback": None,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.facecolor": "white", "savefig.pad_inches": 0.02,
    "hatch.linewidth": 0.4, "hatch.color": "#808080",
    "errorbar.capsize": 0,
}

# ---------------------------------------------------------------- 2) 방법 색·선·마커(§1.1, 개정 색)
METHOD: dict[str, dict] = {
    "phys":          dict(color="#4d4d4d", ls="-", lw=LW["ref"], marker=None, label="Physics anchor (E0, n = 0)"),
    "ref_allA":      dict(color="#000000", ls=(0, (4, 2)), lw=LW["ref"], marker=None, label="All A-block labels (reference)"),
    "oracle_branch": dict(color="#000000", ls="none", lw=LW["main"], marker="*", label="Oracle branch (post hoc)"),
    "direct":        dict(color="#6b7280", ls="-", lw=LW["main"], marker="x", label="Direct ML (no physics)"),
    "refit":         dict(color="#2b5c8f", ls="-", lw=LW["main"], marker="o", label="E re-fit"),
    "augment":       dict(color="#2e6b2e", ls="-.", lw=LW["main"], marker="^", label="Physics pseudo-label augmentation"),
    "residual":      dict(color="#9a7bc9", ls="-", lw=LW["main"], marker="D", label="Physics anchor + residual ML"),
    "tfm":           dict(color="#ad921a", ls="-", lw=LW["main"], marker="s", label="Tabular foundation model"),
    "cci":           dict(color="#568f72", ls="-", lw=LW["main"], marker="v", label="CCI combination"),
}
COLOR = {k: v["color"] for k, v in METHOD.items()}
# 방법 변형(선종). 같은 방법 안의 처리 구분에만 쓴다(§1.1 채움·선종 규칙).
VARIANT_LS = {"solid": "-", "dashed": (0, (3.5, 1.8)), "dashdot": "-.", "dotted": ":"}
PHYS_ABS_LS = ":"                                  # 절대 RMSE 패널에서 물리식 수평선
# 회색 보조
GREY = dict(band="#f2f2f2", land="#f4f4f4", pf_cont="#d0d0d0", pf_disc="#e6e6e6", missing="#e9ecef", aoa="#B8BEC6",
            edge="#808080", grid="#999999", text2="#4d4d4d", light="#b0b0b0", mid="#808080")
DELTA_ZERO = dict(color="#4d4d4d", lw=0.7, ls="-", zorder=1.5)
AOA_MASK = dict(facecolor="#B8BEC6", hatch="///", edgecolor="#808080", lw=0)
CENSOR = dict(reached="#9a7bc9", censored="#4d4d4d", range_limited="#B8BEC6")

# ---------------------------------------------------------------- 3) 지역 마커(§1.2)
REGION_MARKER = {"Alaska": "o", "Canada": "s", "Lena": "D", "Russia_W": "^", "Russia_E": "v", "Russia_C": "P",
                 "Greenland": "*", "Mongolia_CAsia": "h"}
SUBREGION_PREFIX = {"AL-": "Alaska", "CA-": "Canada", "LE-": "Lena"}
SUBREGION_SCALE = 0.85
REGION_EN = {"Alaska": "Alaska", "Canada": "Canada", "Lena": "Lena", "Russia_W": "Russia W", "Russia_E": "Russia E",
             "Russia_C": "Russia C", "Greenland": "Greenland", "Mongolia_CAsia": "Mongolia/Central Asia"}


def parent_region(name: str) -> str:
    """'AL-3' → 'Alaska'. 상위 지역 이름은 그대로 돌려준다."""
    for p, r in SUBREGION_PREFIX.items():
        if str(name).startswith(p):
            return r
    return str(name)


def region_marker(name: str) -> tuple[str, float]:
    """(마커, 크기 배율). 하위 지역은 상위 마커 + 0.85배(채움 유지, 빈 마커 금지)."""
    par = parent_region(name)
    return REGION_MARKER.get(par, "o"), (SUBREGION_SCALE if par != str(name) else 1.0)


def region_name(name: str) -> str:
    return REGION_EN.get(str(name), str(name))


def region_label(name: str, abslogE: float | None = None, nd: int = 2) -> str:
    """'Canada (0.00)'. abslogE 가 없으면 이름만."""
    s = region_name(name)
    return s if abslogE is None or not np.isfinite(abslogE) else f"{s} ({abslogE:.{nd}f})"


# ---------------------------------------------------------------- 4) 연속 색(§1.3)
def cmap(name: str):
    """'diverging' = cmc.broc(0 중심), 'alt' = cmc.oslo_r, 'width' = cmc.acton."""
    from cmcrameri import cm as cmc
    return {"diverging": cmc.broc, "alt": cmc.oslo_r, "width": cmc.acton}[name]


def shared_diverging_norm(map_arrays: Iterable, step_cm: float = 5.0) -> TwoSlopeNorm:
    """같은 색막대를 쓰는 지도 패널들의 |값| 99 백분위를 step 단위로 올림 → ±vmax 0 중심 정규화."""
    v = np.concatenate([np.abs(np.asarray(a, float).ravel()) for a in map_arrays])
    v = v[np.isfinite(v)]
    vmax = step_cm * math.ceil(max(np.percentile(v, 99), 1e-9) / step_cm) if len(v) else step_cm
    return TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)


def circle_size(n, k: float = 0.6, smin: float = 4.0, smax: float = 120.0):
    """면적 비례 원 크기(pt²) = clip(k·n, smin, smax)."""
    return np.clip(k * np.asarray(n, float), smin, smax)


# ---------------------------------------------------------------- 5) 그림·축 배치(mm 절대 좌표)
def use_paper() -> None:
    """rcParams 적용. 그림 모듈 첫머리에서 1회 호출한다."""
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)     # Arial 부재 경고 억제(Liberation Sans 로 대체)
    plt.rcParams.update(PAPER_RC)


def paper_figure(width_mm: float = W2_MM, height_mm: float | None = None) -> plt.Figure:
    """빈 그림. 축은 axes_mm·slot_mm 으로만 추가한다. 높이 기본 = 폭 × 0.66."""
    height_mm = float(height_mm if height_mm is not None else width_mm * 0.66)
    if height_mm > HMAX_MM + 1e-6:
        raise ValueError(f"그림 높이 {height_mm:.1f} mm > {HMAX_MM} mm")
    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
    fig._paper_mm = (float(width_mm), height_mm)
    return fig


def fig_mm(fig) -> tuple[float, float]:
    w, h = fig.get_size_inches()
    return w / MM, h / MM


def mm_to_fig(fig, x_mm, y_mm):
    W, H = fig_mm(fig)
    return x_mm / W, y_mm / H


def axes_mm(fig, x_mm: float, y_mm: float, w_mm: float, h_mm: float, **kw):
    """좌하 원점 mm 좌표의 축 상자(눈금 라벨 제외). kw 는 fig.add_axes 로 전달(projection 등)."""
    W, H = fig_mm(fig)
    ax = fig.add_axes([x_mm / W, y_mm / H, w_mm / W, h_mm / H], **kw)
    ax._slot_mm = (x_mm, y_mm, w_mm, h_mm)          # 패널 문자 기준(슬롯 좌상단)
    ax._box_mm = (x_mm, y_mm, w_mm, h_mm)
    return ax


def slot_mm(fig, x_mm: float, y_mm: float, label_mm: float, w_axis_mm: float, h_mm: float, **kw):
    """슬롯 = 좌측 라벨 여백(label_mm) + 축 상자(w_axis_mm). 축 상자는 x_mm + label_mm 에서 시작한다.
    y_mm·h_mm 는 축 상자 기준(x 라벨 여백은 호출자가 y_mm 에 포함해 둔다)."""
    ax = axes_mm(fig, x_mm + label_mm, y_mm, w_axis_mm, h_mm, **kw)
    ax._slot_mm = (x_mm, y_mm, label_mm + w_axis_mm, h_mm)
    return ax


def panel_label(ax, letter: str, dx_mm: float = 0.0, dy_mm: float = 1.5, fig=None):
    """굵은 소문자 패널 문자(9 pt). 위치 = 슬롯 좌측 끝(dx_mm 보정), 축 상자 위 dy_mm."""
    fig = fig or ax.figure
    x0, y0, w, h = getattr(ax, "_slot_mm", None) or _ax_box_mm(ax)
    fx, fy = mm_to_fig(fig, x0 + dx_mm, y0 + h + dy_mm)
    t = fig.text(fx, fy, letter, fontsize=FS["panel"], fontweight="bold", ha="left", va="bottom")
    t.set_gid("panel_label")
    return t


def label_panels(axes, letters: str = "abcdefgh", dx_mm: float = 0.0, dy_mm: float = 1.5):
    """축 순서대로 a, b, c … 부여. letters 에 공백을 넣으면 해당 축은 건너뛴다."""
    return [panel_label(ax, l, dx_mm, dy_mm) for ax, l in zip(axes, letters) if l.strip()]


def _ax_box_mm(ax):
    W, H = fig_mm(ax.figure)
    b = ax.get_position()
    return b.x0 * W, b.y0 * H, b.width * W, b.height * H


def hide_ylabels(ax):
    """y 공유 패널의 눈금 라벨 숨김(눈금은 유지)."""
    ax.tick_params(axis="y", labelleft=False)


# ---------------------------------------------------------------- 6) 방법 스타일
def style_of(method: str, variant: str | None = None, filled: bool = True, ms: float | None = None) -> dict:
    """plot() 에 바로 넘길 kwargs. variant ∈ {None, 'solid', 'dashed', 'dashdot', 'dotted'} 는 선종만 바꾼다.
    filled=False 는 빈 마커(그림당 한 의미만 허용, 기본 = 블록 등가중 채점)."""
    m = METHOD[method]
    ls = VARIANT_LS.get(variant, m["ls"]) if variant else m["ls"]
    d = dict(color=m["color"], ls=ls, lw=m["lw"], marker=m["marker"] or "", ms=ms or MS["main"],
             mec=m["color"], mew=LW["marker_edge"], mfc=m["color"] if filled else "white")
    return d


def method_handle(method: str, label: str | None = None, variant: str | None = None, line: bool = True,
                  marker: bool = True, filled: bool = True) -> Line2D:
    """범례 핸들(방법 색·선종·마커)."""
    s = style_of(method, variant, filled)
    if not line:
        s["ls"] = "none"
    if not marker:
        s["marker"] = ""
    return Line2D([], [], label=label or METHOD[method]["label"], **s)


def mark_delta_axis(ax, which: str = "x") -> None:
    """Δ 축 표지(gid). qa_check 가 이 축에서 0선을 검사한다."""
    assert which in ("x", "y")
    ax.set_gid(f"delta_{which}")


def zero_line(ax, axis: str = "x", better_text: bool = True, text: str | None = None):
    """Δ = 0 기준선(DELTA_ZERO)과 'better' 방향 표지(6.5 pt, 1회).
    axis='x'(가로 포레스트): 0 세로선, 축 위쪽 0선 왼편에 'better ←'.
    axis='y'(세로 곡선): 0 가로선, 축 오른쪽 끝 0선 아래에 'better ↓'."""
    if axis == "x":
        ln = ax.axvline(0.0, **DELTA_ZERO); ln.set_gid("delta_zero")
        mark_delta_axis(ax, "x")
        if better_text:
            tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
            t = ax.annotate(text or "better ←", xy=(0.0, 1.0), xycoords=tr, xytext=(-1.5, 1.0), textcoords="offset points",
                            ha="right", va="bottom", fontsize=FS["annot"], color=GREY["text2"], annotation_clip=False)
            t.set_gid("better")
    else:
        ln = ax.axhline(0.0, **DELTA_ZERO); ln.set_gid("delta_zero")
        mark_delta_axis(ax, "y")
        if better_text:
            tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
            t = ax.annotate(text or "better ↓", xy=(1.0, 0.0), xycoords=tr, xytext=(-1.0, -1.5), textcoords="offset points",
                            ha="right", va="top", fontsize=FS["annot"], color=GREY["text2"], annotation_clip=False)
            t.set_gid("better")
    return ln


def ref_line(ax, value: float, method: str = "ref_allA", axis: str = "y", **kw):
    """참조 수평·수직선(전량 A 라벨 참조 = 검정 파선, 절대 RMSE 패널 물리식 = 점선)."""
    s = METHOD[method]
    d = dict(color=s["color"], ls=s["ls"] if method != "phys" else PHYS_ABS_LS, lw=s["lw"], zorder=1.6)
    d.update(kw)
    return ax.axhline(value, **d) if axis == "y" else ax.axvline(value, **d)


# ---------------------------------------------------------------- 7) 곡선·포레스트·CI
def ci_errorbar(ax, at, lo, hi, orient: str = "h", method: str = "residual", variant: str | None = None,
                lw: float | None = None, color: str | None = None, zorder: float = 2.5):
    """CI 막대(캡 없음, 둥근 끝). orient='h' → x 방향 [lo, hi] 를 y=at 에, 'v' → y 방향 [lo, hi] 를 x=at 에.
    variant('dashed' 등)는 같은 방법의 두 변형 구분용 선종."""
    s = style_of(method, variant)
    lw = lw or LW["ci"]
    lo, hi, at = np.atleast_1d(lo).astype(float), np.atleast_1d(hi).astype(float), np.atleast_1d(at).astype(float)
    out = []
    for a, b, c in zip(lo, hi, at):
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        xs, ys = ([a, b], [c, c]) if orient == "h" else ([c, c], [a, b])
        out += ax.plot(xs, ys, color=color or s["color"], ls=VARIANT_LS.get(variant, "-") if variant else "-", lw=lw, solid_capstyle="round",
                       dash_capstyle="round", zorder=zorder)
    return out


def offscale_marker(ax, value: float, at: float, lim: tuple[float, float], orient: str = "h", method: str = "residual",
                    ms: float | None = None, zorder: float = 4):
    """축 범위 밖 값: 축 끝에 계열 색 채운 '>'·'<'(가로) 또는 '^'·'v'(세로). 범위 안이면 None."""
    lo, hi = lim
    if lo <= value <= hi or not np.isfinite(value):
        return None
    c = COLOR[method]
    if orient == "h":
        mk, x = (">", hi) if value > hi else ("<", lo)
        return ax.plot([x], [at], mk, color=c, mec=c, ms=ms or MS["main"] + 0.6, clip_on=False, zorder=zorder)[0]
    mk, y = ("^", hi) if value > hi else ("v", lo)
    return ax.plot([at], [y], mk, color=c, mec=c, ms=ms or MS["main"] + 0.6, clip_on=False, zorder=zorder)[0]


def plot_curve(ax, x, y, lo=None, hi=None, method: str = "residual", ls=None, band: bool = False, filled_mask=None,
               label: str | None = None, delta: bool = True, variant: str | None = None, marker: bool = True,
               ms: float | None = None, zorder: float = 3, **kw):
    """n 곡선(Δ = y). band=True 면 lo·hi 를 띠(alpha 0.18)로, 아니면 세로 CI 막대로 그린다.
    filled_mask: 점별 채움 불리언(Fig 4a–d: 최소 n 세 조건 충족 = 채움). delta=True 면 0선·gid 를 자동 추가한다."""
    s = style_of(method, variant, ms=ms)
    if ls is not None:
        s["ls"] = VARIANT_LS.get(ls, ls)
    if not marker:
        s["marker"] = ""
    s.update(kw)
    x = np.asarray(x, float); y = np.asarray(y, float)
    o = np.argsort(x); x, y = x[o], y[o]
    if lo is not None and hi is not None:
        lo = np.asarray(lo, float)[o]; hi = np.asarray(hi, float)[o]
        if band:
            ax.fill_between(x, lo, hi, color=s["color"], alpha=BAND_ALPHA, lw=0, zorder=zorder - 1)
        else:
            ci_errorbar(ax, x, lo, hi, orient="v", method=method, zorder=zorder - 0.5)
    if filled_mask is not None:
        fm_ = np.asarray(filled_mask, bool)[o]
        line, = ax.plot(x, y, **{**s, "marker": ""}, label=None, zorder=zorder)
        if s["marker"]:
            ax.plot(x[fm_], y[fm_], ls="none", marker=s["marker"], ms=s["ms"], mec=s["color"], mfc=s["color"], mew=s["mew"], zorder=zorder + 0.1)
            ax.plot(x[~fm_], y[~fm_], ls="none", marker=s["marker"], ms=s["ms"], mec=s["color"], mfc="white", mew=s["mew"], zorder=zorder + 0.1)
        line.set_label(label)
    else:
        line, = ax.plot(x, y, label=label, zorder=zorder, **s)
    if delta and ax.get_gid() != "delta_y":
        zero_line(ax, "y", better_text=True)
    return line


def forest(ax, labels: Sequence[str], est, lo=None, hi=None, method: str = "residual", offset: float = 0.0, ls="-",
           offscale: tuple[float, float] | None = None, sharey_with=None, short_labels: Sequence[str] | None = None,
           filled=True, ms: float | None = None, lw: float | None = None, marker: str | None = None,
           set_labels: bool = True, delta: bool = True, better_text: bool = True, zorder: float = 3):
    """가로 포레스트(Δ = x). 첫 행이 맨 위. offset: 같은 행 안 세로 오프셋(±0.18 권장, 두 변형 병치).
    ls: CI 막대 선종(변형 구분, '-'·'dashed'). offscale=(xmin, xmax) 이면 범위 밖 점을 축 끝 '>'/'<' 로.
    filled: bool 또는 행별 불리언 배열(False = 빈 마커, 그림당 한 의미). 반환: 행 y 좌표 배열."""
    n = len(labels)
    y = np.arange(n)[::-1].astype(float)
    est = np.asarray(est, float)
    s = style_of(method, ms=ms)
    mk = marker or s["marker"] or "o"
    fl = np.broadcast_to(np.asarray(filled, bool), (n,))
    ls_ = VARIANT_LS.get(ls, ls)
    if lo is not None and hi is not None:
        lo_, hi_ = np.asarray(lo, float), np.asarray(hi, float)
        if offscale is not None:
            lo_c, hi_c = np.clip(lo_, *offscale), np.clip(hi_, *offscale)
        else:
            lo_c, hi_c = lo_, hi_
        for yi, a, b in zip(y + offset, lo_c, hi_c):
            if np.isfinite(a) and np.isfinite(b):
                ax.plot([a, b], [yi, yi], color=s["color"], ls=ls_, lw=lw or LW["ci_forest"], solid_capstyle="round",
                        dash_capstyle="round", zorder=zorder - 0.5)
    for yi, v, f in zip(y + offset, est, fl):
        if not np.isfinite(v):
            continue
        if offscale is not None and not (offscale[0] <= v <= offscale[1]):
            offscale_marker(ax, v, yi, offscale, "h", method)
            continue
        ax.plot([v], [yi], ls="none", marker=mk, ms=s["ms"], mec=s["color"], mew=LW["marker_edge"],
                mfc=s["color"] if f else "white", zorder=zorder)
    if set_labels:
        ax.set_yticks(y)
        if sharey_with is not None:
            ax.set_yticklabels([]); ax.tick_params(axis="y", length=0)
        else:
            ax.set_yticklabels(list(short_labels) if short_labels is not None else list(labels))
        ax.set_ylim(-0.6, n - 0.4)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.spines["left"].set_visible(False)
    if offscale is not None:
        ax.set_xlim(*offscale)
    if delta and ax.get_gid() != "delta_x":
        zero_line(ax, "x", better_text=better_text)
    return y


def group_bands(ax, rows: Sequence[int], color: str = GREY["band"], text: str | None = None, side: str = "right"):
    """행 묶음 배경 띠(#f2f2f2) + 우측 6.5 pt 브래킷 텍스트('level'·'structure'). rows = forest 가 돌려준 y 좌표 부분집합."""
    if not len(rows):
        return None
    y0, y1 = min(rows) - 0.5, max(rows) + 0.5
    tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    r = Rectangle((0, y0), 1, y1 - y0, transform=tr, facecolor=color, edgecolor="none", zorder=0)
    ax.add_patch(r)
    if text:
        t = ax.text(1.01 if side == "right" else -0.01, (y0 + y1) / 2, text, transform=tr, fontsize=FS["annot"], rotation=90,
                    ha="left" if side == "right" else "right", va="center", color=GREY["text2"], clip_on=False)
        t.set_gid("bracket")
    return r


def censor_band(ax, x, n_max, names: Sequence[str] | None = None, range_limited=None, y_marker: float = 0.22,
                label_rows=(0.52, 0.82), merge_pt: float = 6.0):
    """'not reached' 띠 축(Fig 4e 아래 영역, 높이 ~8 mm)에 미달성 대상을 ▲ 로 찍고 '> n_max' 를 띠 안에 쓴다.
    ax: 띠 전용 축(이 함수가 배경 #f2f2f2·y 범위 0–1·y 눈금을 정리). x: 가로 위치(|log E비|). n_max: 대상별 최대 검사 n.
    range_limited: 불리언(True = n_max < 40, 회색 ▲ + ' (range-limited)'). names: 라벨 앞에 붙일 이름(선택, 예: Table 1 번호).
    라벨 배치: x 간격 < merge_pt(pt)이고 라벨이 같은 이웃 ▲ 는 한 라벨로 합치고(▲ 개수가 대상 수를 보여 준다),
    라벨 상자가 겹치면 label_rows 중 오른쪽 이동이 가장 작은 줄에 둔다(렌더러 기반 탐욕 배치, 축 왼쪽 밖이면 안으로 민다)."""
    x = np.asarray(x, float); n_max = np.asarray(n_max)
    rl = np.zeros(len(x), bool) if range_limited is None else np.asarray(range_limited, bool)
    ax.set_facecolor(GREY["band"]); ax.set_ylim(0, 1); ax.set_yticks([])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    arts = []
    labs = []
    for i, (xi, nm, r) in enumerate(zip(x, n_max, rl)):
        c = CENSOR["range_limited"] if r else CENSOR["censored"]
        arts += ax.plot([xi], [y_marker], "^", color=c, mec=c, ms=MS["main"] + 0.4, zorder=3, clip_on=False)
        lab = f"> {int(nm)}" + (" (range-limited)" if r else "")
        labs.append(((names[i] + " ") if names is not None else "") + lab)
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts_per_data = abs(ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]) * 72 / fig.dpi
    o = np.argsort(x)
    groups = []                                           # [(x 목록, 라벨)]
    for k in o:
        if groups and groups[-1][1] == labs[k] and (x[k] - groups[-1][0][-1]) * pts_per_data < merge_pt:
            groups[-1][0].append(x[k])
        else:
            groups.append(([x[k]], labs[k]))
    placed = {r_: [] for r_ in range(len(label_rows))}
    inv = ax.transData.inverted()
    for xs, lab in groups:
        xm = float(np.mean(xs))
        t = ax.text(xm, label_rows[0], lab, ha="center", va="center", fontsize=FS["annot"], color=GREY["text2"], clip_on=False)
        bb0 = t.get_window_extent(rend)
        lead = max(0.0, ax.bbox.x0 - bb0.x0)                  # 축 왼쪽 밖으로 나가면 오른쪽으로 민다
        best = None
        for r_, yr in enumerate(label_rows):
            x0, x1 = bb0.x0 + lead, bb0.x1 + lead
            for b in sorted(placed[r_], key=lambda b: b.x0):   # 겹치는 상자 뒤로 밀기
                if x0 < b.x1 + 2 and b.x0 < x1 + 2:
                    d = b.x1 + 2 - x0; x0 += d; x1 += d
            shift = x0 - bb0.x0
            if best is None or shift < best[0] - 1e-6:
                best = (shift, r_)
            if shift <= lead + 1e-6:
                break
        shift, r_ = best
        xd = inv.transform((ax.transData.transform((xm, 0))[0] + shift, 0))[0]
        t.set_x(xd); t.set_y(label_rows[r_])
        placed[r_].append(t.get_window_extent(rend))
        arts.append(t)
    return arts


def legend_below(fig, handles: Sequence, rect_mm: tuple[float, float, float, float], ncol: int | None = None, **kw):
    """그림당 1개 범례를 mm 사각형(x, y, w, h) 중앙에 둔다. 항목 ≤ 8 을 강제한다."""
    handles = list(handles)
    if len(handles) > 8:
        raise ValueError(f"범례 항목 {len(handles)} > 8")
    x, y, w, h = rect_mm
    fx, fy = mm_to_fig(fig, x, y); fw, fh = mm_to_fig(fig, w, h)
    lg = fig.legend(handles=handles, loc="center", bbox_to_anchor=(fx, fy, fw, fh), bbox_transform=fig.transFigure,
                    ncol=ncol or len(handles), fontsize=FS["legend"], frameon=False, **kw)
    lg.set_gid("paper_legend")
    return lg


one_legend = legend_below                     # 스펙 §1.5 이름 호환


def wrap(s: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(s, width))


def log_n_axis(ax, ticks=(3, 10, 30, 100, 300), lim=(2.5, 380), axis: str = "x"):
    """라벨 수 n 로그 축(눈금 3·10·30·100·300, 보조 눈금 라벨 없음)."""
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    (ax.set_xlim if axis == "x" else ax.set_ylim)(*lim)
    a.set_major_locator(FixedLocator(list(ticks))); a.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def symlog_delta_axis(ax, linthresh: float = 2.0, ticks=(-10, -5, -2, -1, 0, 1, 2, 5), lim=(-12, 6), axis: str = "y"):
    """Fig 3·4 공유 Δ symlog 축(선형 ±linthresh cm)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    (ax.set_yscale if axis == "y" else ax.set_xscale)("symlog", linthresh=linthresh, linscale=1.0)
    (ax.set_ylim if axis == "y" else ax.set_xlim)(*lim)
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(list(ticks))); a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, p: fmt_num(v, 0)))


def fmt_num(v: float, nd: int = 1) -> str:
    """유니코드 마이너스(U+2212) 숫자 문자열."""
    if v is None or not np.isfinite(v):
        return "–"
    s = f"{v:.{nd}f}"
    if s.startswith("-"):
        s = "−" + s[1:]
        if float(s[1:]) == 0:
            s = s[1:]
    return s


# ---------------------------------------------------------------- 8) 지도(cartopy, §1.4 지도 행)
def paper_map_ax(fig, rect_mm: tuple[float, float, float, float], extent: tuple[float, float, float, float],
                 lon0: float | None = None, grid_lat: float = 5, grid_lon: float = 10, scalebar_km: float | None = 200,
                 inset: bool = True, land: bool = True, coast_res: str = "50m"):
    """극 입체(NorthPolarStereo, true_scale_latitude 70) 지도 축. extent = (lon_min, lon_max, lat_min, lat_max).
    NE 50 m 해안선 0.4 pt #808080, 육지 #f4f4f4, 위도·경도선 0.4 pt #999999(라벨 없음), 축척 막대 좌하, 삽도(범북극 위치) 0.22 폭."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    lon0 = float(np.mean(extent[:2])) if lon0 is None else float(lon0)
    proj = ccrs.NorthPolarStereo(central_longitude=lon0, true_scale_latitude=70)
    x, y, w, h = rect_mm
    ax = axes_mm(fig, x, y, w, h, projection=proj)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    if land:
        ax.add_feature(cfeature.LAND.with_scale(coast_res), facecolor=GREY["land"], edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale(coast_res), edgecolor=GREY["edge"], lw=0.4, zorder=1)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, lw=0.4, color=GREY["grid"], alpha=1.0, zorder=0.5,
                      xlocs=np.arange(-180, 181, grid_lon), ylocs=np.arange(40, 91, grid_lat))
    ax.spines["geo"].set_linewidth(0.5)
    ax._paper_proj = proj
    if scalebar_km:
        scale_bar(ax, scalebar_km)
    if inset:
        inset_locator(fig, ax, extent)
    return ax


def scale_bar(ax, km: float, loc=(0.05, 0.06), lw: float = 1.2):
    """투영 좌표(m) 기준 축척 막대(true_scale 70°N 기준, 캡션에 기준 위도 명시)."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xs = x0 + loc[0] * (x1 - x0); ys = y0 + loc[1] * (y1 - y0)
    ax.plot([xs, xs + km * 1000], [ys, ys], color="#000000", lw=lw, solid_capstyle="butt", transform=ax.transData, zorder=5)
    t = ax.text(xs + km * 500, ys + 0.015 * (y1 - y0), f"{int(km):,} km", ha="center", va="bottom", fontsize=FS["annot"],
                transform=ax.transData, zorder=5)
    t.set_gid("scalebar")
    return t


def inset_locator(fig, ax, extent, frac: float = 0.22, corner: str = "upper right"):
    """범북극 위치 삽도(폭 = 부모 축 × frac). 대상 영역을 0.6 pt 사각형으로."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    x, y, w, h = ax._box_mm
    s = w * frac
    ix = x + w - s if "right" in corner else x
    iy = y + h - s if "upper" in corner else y
    iax = axes_mm(fig, ix, iy, s, s, projection=ccrs.NorthPolarStereo())
    iax.set_extent((-180, 180, 50, 90), crs=ccrs.PlateCarree())
    iax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#e0e0e0", edgecolor="none")
    iax.spines["geo"].set_linewidth(0.4)
    lo0, lo1, la0, la1 = extent
    xs = np.r_[np.linspace(lo0, lo1, 20), np.full(10, lo1), np.linspace(lo1, lo0, 20), np.full(10, lo0)]
    ys = np.r_[np.full(20, la0), np.linspace(la0, la1, 10), np.full(20, la1), np.linspace(la1, la0, 10)]
    iax.plot(xs, ys, color="#000000", lw=0.6, transform=ccrs.PlateCarree())
    iax.set_gid("inset")
    return iax


def density_circles(ax, lon, lat, n, k: float = 0.6, smin: float = 4, smax: float = 120, color: str | None = None,
                    alpha: float = 0.45, edge: str = GREY["edge"], c=None, norm=None, cmap_=None, zorder: float = 3):
    """면적 비례 원. color(단색) 또는 c+norm+cmap_(발산 색). 테두리 0.3 pt #808080. 작은 원이 위(n 내림차순으로 그림)."""
    import cartopy.crs as ccrs
    n = np.asarray(n, float); o = np.argsort(-n)
    kw = dict(s=circle_size(n[o], k, smin, smax), transform=ccrs.PlateCarree(), linewidths=0.3, edgecolors=edge, zorder=zorder)
    if c is not None:
        return ax.scatter(np.asarray(lon)[o], np.asarray(lat)[o], c=np.asarray(c)[o], norm=norm, cmap=cmap_, **kw)
    return ax.scatter(np.asarray(lon)[o], np.asarray(lat)[o], color=color or COLOR["refit"], alpha=alpha, **kw)


def size_legend_handles(values=(10, 100, 1000), k: float = 0.6, smin: float = 4, smax: float = 120, color: str | None = None,
                        alpha: float = 0.45, fmt: str = "{:,} cells"):
    """면적 비례 원 범례 핸들(markersize = √s)."""
    return [Line2D([], [], ls="none", marker="o", ms=float(np.sqrt(circle_size(v, k, smin, smax))), mfc=color or COLOR["refit"],
                   mec=GREY["edge"], mew=0.3, alpha=alpha, label=fmt.format(v)) for v in values]


# ---------------------------------------------------------------- 9) 저장
def save_figure(fig, path_stem, formats=("pdf", "png"), dpi: int = 600, expect_width_mm: float | None = W2_MM,
                tol_mm: float = 1.0, max_height_mm: float = HMAX_MM) -> dict:
    """PDF(Type 42 글꼴) + 600 dpi PNG 저장(bbox tight 미사용: 캔버스 = 설계 mm).
    폭·높이 검사 실패 시 ValueError. 반환 {pdf, png, width_mm, height_mm}."""
    W, H = fig_mm(fig)
    if expect_width_mm is not None and abs(W - expect_width_mm) > tol_mm:
        raise ValueError(f"그림 폭 {W:.1f} mm ≠ {expect_width_mm} ± {tol_mm} mm")
    if H > max_height_mm + 1e-6:
        raise ValueError(f"그림 높이 {H:.1f} mm > {max_height_mm} mm")
    stem = Path(path_stem); stem.parent.mkdir(parents=True, exist_ok=True)
    out = dict(width_mm=round(W, 2), height_mm=round(H, 2))
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        for f in formats:
            p = stem.with_suffix(f".{f}")
            fig.savefig(p, dpi=dpi if f == "png" else None, facecolor="white")
            out[f] = str(p)
    return out
