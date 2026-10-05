"""v3 논문 그림·표 공용 양식 모듈(재구성 비교판).

근거: design/journal_grade_style_guide.md(이하 '지침') 2절(그림), 3절(표), 2.14절(상수), 부록 A(점검 함수),
outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md 1절(공통 규격).

- v2 모듈(scripts/4_visualization/paper/)과 src/polar/paperstyle.py 는 가져오지 않는다(지침 2.14 의 v3 값을 여기에 둔다).
- 산출 폴더: outputs/figures/paper/v3_restructure/ (사용자 결정 2026-10-04, 그림 명세 1.7).
- 그림 함수는 medium="paper" 또는 "slide" 를 받아 같은 자료·색·범위로 두 매체를 그린다(지침 0.4 3).
- 여러 그림 모듈이 함께 쓰므로 이름을 바꾸거나 지우지 말고 더하기만 한다.
- v4(2026-10-05): 글꼴·선·색 값은 design/style_tokens_v4.json 하나에서 읽는다(사용자 검토: 흑백에 가까움, 굵은 글씨, 화살촉 과다).
  글꼴 FreeSans(Helvetica 계열 TrueType, Regular), 한글 Pretendard Regular, 패널 문자 8 pt 굵게. 방법마다 색, 강조색은 주홍 하나.
"""
from __future__ import annotations

import re
import subprocess
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
from matplotlib.collections import Collection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parents[3]
PAPER_FIGS = ROOT / "data" / "processed" / "paper_figs"          # 그림용 원천 표(등록)
OUT = ROOT / "outputs" / "figures" / "paper" / "v3_restructure"  # v3 비교 폴더
SOURCE_DATA = OUT / "source_data"                                # 패널 값(그림 명세 1.7)

# ---------------------------------------------------------------- v4 토큰(design/style_tokens_v4.json, 단일 원천)
TOKENS_PATH = ROOT / "design" / "style_tokens_v4.json"
TOK = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))
_F, _LP, _LS, _C = TOK["font"], TOK["lines"]["paper"], TOK["lines"]["slide"], TOK["color"]
FONT_LATIN = _F["latin"]                                         # FreeSans
FONT_KOREAN = _F["korean"]                                       # Pretendard(Regular)
FONT_FILES = [_F["latin_regular_file"], _F["latin_bold_file"],
              str(Path(_F["latin_regular_file"]).with_name("FreeSansOblique.ttf")),
              str(Path(_F["latin_regular_file"]).with_name("FreeSansBoldOblique.ttf")),
              str(Path.home() / ".fonts" / "Pretendard-Regular.otf")]


def register_fonts() -> None:
    """FreeSans 네 굵기·기울임과 Pretendard Regular 를 matplotlib 에 등록한다(여러 번 불러도 된다)."""
    have = {f.fname for f in _fm.fontManager.ttflist}
    for f in FONT_FILES:
        if Path(f).exists() and f not in have:
            _fm.fontManager.addfont(f)


register_fonts()

# ---------------------------------------------------------------- 크기(지침 2.1, 2.14; 글꼴·선 값은 v4 토큰)
MM_PER_IN = 25.4
W2_MM, W15_MM, W1_MM = 170.0, 120.0, 85.0
HMAX_MM, HTARGET_MM = 180.0, 165.0
FONT_PT = float(_F["paper"]["text_pt"])                          # 7 pt
PANEL_PT = float(_F["paper"]["panel_letter_pt"])                 # 패널 문자 8 pt 굵게
MIN_FONT_PT = float(_F["paper"]["min_pt"])                       # 5 pt
SLIDE_FONT = dict(axis_label=float(_F["slide"]["axis_label_pt"]), tick=float(_F["slide"]["tick_pt"]),
                  direct=float(_F["slide"]["direct_label_pt"]), min=float(_F["slide"]["min_pt"]),
                  panel=float(_F["slide"]["panel_letter_pt"]))
MIN_LW_PT = min(_LP["graticule_pt"], 0.5)   # 점검 하한: v4 토큰의 가장 가는 선(경위선 0.4 pt). 자료·축 선은 0.6 pt 이상

# 선 굵기(pt)와 마커(pt): v4 토큰 lines.paper
LW = dict(axis=_LP["axis_pt"], tick=_LP["tick_pt"], main=_LP["data_pt"], aux=_LP["data_aux_pt"], ref=_LP["reference_pt"],
          ci=_LP["data_aux_pt"], ci_forest_cell=_LP["ci_cell_pt"], ci_forest_block=_LP["ci_block_pt"],
          hatch=_LP["reference_pt"], grid_map=_LP["graticule_pt"], scale_bar=_LP["data_pt"], marker_edge_open=_LP["data_aux_pt"],
          coast=_LP["map_coast_pt"])
MS = dict(main=_LP["marker_pt"], region_point=3.0)
TICK_LEN_PT = _LP["tick_len_pt"]
# 슬라이드판 값(v4 토큰 lines.slide). 슬라이드 모듈이 LW·MS 를 이 값으로 바꿔 쓴다
LW_SLIDE = dict(axis=_LS["axis_pt"], tick=_LS["tick_pt"], main=_LS["data_pt"], aux=_LS["data_aux_pt"], ref=_LS["reference_pt"],
                ci=_LS["data_aux_pt"], ci_forest_cell=_LS["ci_cell_pt"], ci_forest_block=_LS["ci_block_pt"],
                hatch=_LS["reference_pt"], grid_map=_LS["graticule_pt"], scale_bar=_LS["data_pt"], marker_edge_open=_LS["data_aux_pt"],
                coast=_LS["map_coast_pt"])
MS_SLIDE = dict(main=_LS["marker_pt"], region_point=5.0)
TICK_LEN_SLIDE = _LS["tick_len_pt"]
CI_BAND_ALPHA = float(_C["ci_band_alpha"])
FOREST_BLOCK_OFFSET_MM = 0.8      # 블록 등가중 막대를 셀 가중 막대 아래(또는 오른쪽) 0.8 mm

# ---------------------------------------------------------------- 색(v4 토큰 color)
_N = _C["neutral"]
INK = _N["text_paper"]
INK_SLIDE = _N["text_slide"]
INK_AUX = _N["text_aux"]          # 보조 글자(묶음 머리 보조, 띠 머리)
EQUIV_BAND = _N["equiv_band"]     # ±0.5 cm 동등 띠
EQUIV_HALF_WIDTH_CM = 0.5
ZERO_LINE = dict(color=_N["zero_line"], lw=_LP["reference_pt"], ls="-")
REF_COLOR = _N["zero_line"]       # 기준선(오차 하한 등, 자료 계열 아님)

_M = _C["methods"]
# 방법 색 코드: 키는 코드 안에서만 쓰는 이름이다. 그림 글자에는 short(그림 짧은 이름)만 쓴다(H3).
# 선 모양: 토큰 line(solid, dashed, dotted). 같은 색 계열 두 개는 선 모양으로 가른다
_LS_MAP = {"solid": "-", "dashed": (0, (3.0, 1.6)), "dotted": (0, (1.0, 1.4))}
METHOD = {
    "source_stefan":       dict(short=_M["P0"]["en"], color=_M["P0"]["hex"], ls=_LS_MAP[_M["P0"]["line"]]),
    "year_matched_stefan": dict(short=_M["Pstar"]["en"], color=_M["Pstar"]["hex"], ls=_LS_MAP[_M["Pstar"]["line"]]),
    "recalibrated_stefan": dict(short=_M["P1"]["en"], color=_M["P1"]["hex"], ls=_LS_MAP[_M["P1"]["line"]]),
    "soil_adjusted_stefan": dict(short="Soil-adjusted Stefan", color=_M["P1"]["hex"], ls=(0, (4.0, 1.5, 1.0, 1.5))),  # SI 전용
    "anchor_residual":     dict(short=_M["R1"]["en"], color=_M["R1"]["hex"], ls=_LS_MAP[_M["R1"]["line"]]),
    "source_anchor_residual": dict(short="Source anchor + residual", color=_M["R1"]["hex"], ls=(0, (4, 2))),
    "physics_pseudo":      dict(short=_M["D1"]["en"], color=_M["D1"]["hex"], ls=_LS_MAP[_M["D1"]["line"]]),
    "direct_ml":           dict(short=_M["D0"]["en"], color=_M["D0"]["hex"], ls=_LS_MAP[_M["D0"]["line"]]),
    "physics_input":       dict(short=_M["F1"]["en"], color=_M["F1"]["hex"], ls=_LS_MAP[_M["F1"]["line"]]),
    "stefan_cci_anchor":   dict(short="Stefan-CCI anchor", color=_C["products"]["CCI"]["hex"], ls=(0, (5, 1.5, 1.5, 1.5))),
    "target_cv_selection": dict(short=_M["W"]["en"], color=_M["W"]["hex"], ls=_LS_MAP[_M["W"]["line"]]),
    "workflow":            dict(short=_M["WF"]["en"], color=_M["WF"]["hex"], ls=_LS_MAP[_M["WF"]["line"]]),
    "stack":               dict(short=_M["Stack"]["en"], color=_M["Stack"]["hex"], ls=_LS_MAP[_M["Stack"]["line"]]),
    "placebo":             dict(short=_M["placebo"]["en"], color=_M["placebo"]["hex"], ls=_LS_MAP[_M["placebo"]["line"]]),
    "error_floor":         dict(short="Error floor", color=_N["zero_line"], ls=(0, (1.0, 1.4))),   # 기준선(계열 아님): 점선
}
METHOD_HEX = {v["color"] for v in METHOD.values() if v["color"]}
PRODUCT = {k: dict(short=v["en"], color=v["hex"]) for k, v in _C["products"].items()}
PLACEMENT = {k: v for k, v in _C.get("placement", {}).items() if k != "note"}     # 배치 전략 색(Fig 5)
INTERVALS = {k: v for k, v in _C.get("intervals", {}).items() if k != "note"}     # 예측 구간 방법 색(Fig 6d·e)
STATUS = {k: v for k, v in _C.get("status", {}).items() if k != "note"}           # 상태·자료 종류 색(덱 메타 차트)
REGION_COLOR = {k: v for k, v in _C["regions"].items() if k != "note"}
ACCENT = _M["R1"]["hex"]          # 그림마다 강조색 하나(제안 방법)
ACCENT_FORBIDDEN = _C["deck"]["title_orange"]   # 덱 제목 주황, 자료 표현 금지

# 연속 색표 이름(cmcrameri), 지침 2.6(사용자 승인, v4 에서도 그대로)
CMAP = dict(alt="oslo_r", delta="broc", pred_change="broc", residual="bam", width="acton_r",
            density="davos_r", ground_temp="vik")
# 바탕 지도 색: v4 토큰 color.map_base(2026-10-05 15:50, 매체별). use_v3(medium) 이 BASEMAP 을 제자리에서 갱신한다
# (모듈이 S.BASEMAP[...] 을 그릴 때 읽으므로 매체 전환이 모든 지도에 적용된다).
MAP_BASE = _C["map_base"]
BASEMAP = dict(continuous=MAP_BASE["paper"]["pf_continuous"], discontinuous=MAP_BASE["paper"]["pf_discontinuous"],
               land=MAP_BASE["paper"]["land"], sea=MAP_BASE["paper"]["ocean"],
               label_block="#bdbdbd", score_block="#737373", graticule=MAP_BASE["paper"]["graticule"], coast=MAP_BASE["paper"]["coast"],
               buffer_line=MAP_BASE["paper"]["buffer_line"],
               candidate="#bdbdbd", selected="#000000", nodata=_N["no_data"], hatch="#9a9a9a")
PF_KEY_LABELS = (("continuous", "Continuous permafrost"), ("discontinuous", "Discontinuous permafrost"))   # 열쇠에 구역 이름을 적는다


def set_basemap(medium: str = "paper") -> None:
    """BASEMAP 의 바탕 지도 항목을 매체 토큰으로 바꾼다(제자리 갱신)."""
    m = MAP_BASE["slide" if medium == "slide" else "paper"]
    BASEMAP.update(continuous=m["pf_continuous"], discontinuous=m["pf_discontinuous"], land=m["land"], sea=m["ocean"],
                   graticule=m["graticule"], coast=m["coast"], buffer_line=m["buffer_line"])

# ---------------------------------------------------------------- 지역 이름과 모양(지침 2.5, 그림 명세 D-14)
# 키는 원천 표의 target 값이다. 이름은 Table 1 에서 한 번 정하고 모든 그림이 같게 쓴다.
REGION_NAME = {
    "Lena": "Lena Delta", "Canada": "Canada", "Russia_W": "W Russia", "Russia_E": "E Russia",
    "Russia_C": "Central Russia", "Greenland": "Greenland", "Alaska": "Alaska",
    "Russia_C_LGD": "Central Russia (expanded)", "Tibet_LGD": "Tibetan Plateau", "NAtlantic": "North Atlantic",
}
REGION_MARKER = {"Alaska": "o", "Canada": "s", "Lena": "D", "Russia_W": "^", "Russia_E": "v",
                 "Russia_C": "P", "Russia_C_LGD": "P", "Greenland": "*"}   # 티베트·북대서양은 미정(D-15)

# ---------------------------------------------------------------- 숫자 표기(지침 2.2, R-13)
MINUS = "−"


def fmt_int(n) -> str:
    """정수. 네 자리 이하는 쉼표 없이(3037), 다섯 자리 이상만 쉼표(13,606). 음수는 U+2212."""
    v = int(round(float(n)))
    s = f"{abs(v):,}" if abs(v) >= 10000 else f"{abs(v)}"
    return (MINUS if v < 0 else "") + s


def fmt_num(x, nd: int) -> str:
    """소수 nd 자리. 음수는 U+2212."""
    s = f"{float(x):.{nd}f}"
    return s.replace("-", MINUS)


def mm(x_mm: float) -> float:
    """mm → in."""
    return x_mm / MM_PER_IN


# ---------------------------------------------------------------- rc 설정
def use_v3(medium: str = "paper") -> None:
    """v4 토큰으로 rcParams 를 맞춘다. 논문: FreeSans 7 pt Regular, 선 토큰 lines.paper.
    슬라이드: FreeSans(한글은 Pretendard Regular 로 대체), 눈금 12 pt, 축 이름 14 pt, 직접 라벨 13 pt, 선 토큰 lines.slide."""
    if medium not in ("paper", "slide"):
        raise ValueError(medium)
    register_fonts()
    set_basemap(medium)
    paper = medium == "paper"
    L_ = LW if paper else LW_SLIDE
    M_ = MS if paper else MS_SLIDE
    fs = FONT_PT if paper else SLIDE_FONT["direct"]
    ink = INK if paper else INK_SLIDE
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_LATIN, FONT_KOREAN],
        "font.weight": "normal", "axes.labelweight": "normal", "axes.titleweight": "normal", "figure.titleweight": "normal",
        "font.size": fs, "axes.labelsize": fs if paper else SLIDE_FONT["axis_label"],
        "xtick.labelsize": fs if paper else SLIDE_FONT["tick"], "ytick.labelsize": fs if paper else SLIDE_FONT["tick"],
        "legend.fontsize": fs, "axes.titlesize": fs, "figure.titlesize": fs,
        "mathtext.fontset": "custom", "mathtext.rm": FONT_LATIN, "mathtext.it": f"{FONT_LATIN}:italic",
        "mathtext.bf": f"{FONT_LATIN}:bold", "mathtext.sf": FONT_LATIN, "mathtext.cal": FONT_LATIN, "mathtext.tt": FONT_LATIN,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": L_["axis"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.edgecolor": ink, "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": ink, "ytick.color": ink, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": L_["tick"], "ytick.major.width": L_["tick"],
        "xtick.major.size": TICK_LEN_PT if paper else TICK_LEN_SLIDE, "ytick.major.size": TICK_LEN_PT if paper else TICK_LEN_SLIDE,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "lines.linewidth": L_["main"], "lines.markersize": M_["main"],
        "lines.markeredgewidth": 0.0, "patch.linewidth": L_["axis"],
        "hatch.linewidth": L_["hatch"], "hatch.color": BASEMAP["hatch"],
        "legend.frameon": False, "savefig.dpi": 600 if paper else 300, "savefig.transparent": False,
        "figure.dpi": 100,
    }
    matplotlib.rcParams.update(rc)


def fig_mm(w_mm: float = W2_MM, h_mm: float = 150.0):
    """최종 크기(배율 1.00)의 빈 그림. bbox_inches='tight' 를 쓰지 않으므로 이 크기가 그대로 저장된다."""
    if h_mm > HMAX_MM:
        raise ValueError(f"height {h_mm} mm > {HMAX_MM} mm")
    return plt.figure(figsize=(mm(w_mm), mm(h_mm)))


def axes_mm(fig, x_mm: float, y_top_mm: float, w_mm: float, h_mm: float, **kw):
    """그림 왼쪽 위를 원점으로 한 mm 좌표(x, 위끝 y, 폭, 높이)에 축을 둔다."""
    W, H = fig.get_size_inches() * MM_PER_IN
    return fig.add_axes([x_mm / W, 1 - (y_top_mm + h_mm) / H, w_mm / W, h_mm / H], **kw)


def panel_letter(fig, x_mm: float, y_top_mm: float, letter: str):
    """8 pt 굵은 직립 소문자 패널 문자(v4 토큰 font.paper). 슬롯 왼쪽 위 끝. 굵게는 패널 문자에만 쓴다."""
    W, H = fig.get_size_inches() * MM_PER_IN
    t = fig.text(x_mm / W, 1 - y_top_mm / H, letter, fontsize=PANEL_PT, fontweight="bold", ha="left", va="top")
    t.set_gid("panel_letter")
    return t


def save_fig(fig, stem: str, out_dir: Path | None = None, formats=("pdf", "svg", "png")) -> list[Path]:
    """PDF(정본)·SVG·PNG 600 dpi. bbox_inches 를 쓰지 않는다(지침 2.1, 2.13)."""
    out_dir = Path(out_dir) if out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None)
        paths.append(p)
    return paths


# ---------------------------------------------------------------- 점검 함수(지침 부록 A, 전문 그대로)
CODE_RE = re.compile(r"\b(L\d{1,2}|AB\d{1,2}|LG[A-Z]?(-[A-Z]\d)?|WF\d{1,2}|SC\d\w*|H\d{2}\w?|X[A-J]|P4|L\+C|T\d{1,3}"
                     r"|P[0-2]\*?|R[0-2]|D[01])\b|\((x|i)\)")
NUM_RE = re.compile(r"[−\-]?\d+(\.\d+)?")
COMMA4_RE = re.compile(r"(?<![\d,.])\d,\d{3}(?![\d,])")


def audit_v3(fig, size_pt=7.0, min_lw=MIN_LW_PT, max_chars=600, max_words=3, allowed_num_gids=("scale", "sizekey")):
    """글자 크기 집합, 문자 수, 최소 선 굵기, 제목, 글씨 상자, 내부 코드, 네 자리 쉼표, 라벨 단어 수, 그림 안 수치를 센다."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    out = dict(sizes=set(), chars=0, thin_lines=[], titles=[], boxed_text=0, codes=[], comma4=[],
               long_labels=[], loose_numbers=[])
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    tick_texts = set()                                   # 눈금 라벨과 축 라벨은 수치·단어 수 검사에서 뺀다
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            tick_texts.update(axis.get_ticklabels())
            tick_texts.add(axis.label)
    cbar_axes = {a for a in fig.axes if a.get_label() == "<colorbar>"}
    for t in texts:
        s = t.get_text().strip()
        out["sizes"].add(round(t.get_fontsize(), 2))
        out["chars"] += len(s.replace(" ", ""))
        if t.get_bbox_patch() is not None:
            out["boxed_text"] += 1
        if CODE_RE.search(s):
            out["codes"].append(s[:30])
        if re.search(r"(?<![\d,.])\d,\d{3}(?![\d,])", s):
            out["comma4"].append(s[:30])
        is_tick_or_axis = t in tick_texts or getattr(t, "axes", None) in cbar_axes
        if not is_tick_or_axis and len([w for w in s.split() if w != "+"]) > max_words and t.get_gid() != "category":
            out["long_labels"].append(s[:40])           # 범주 축 라벨은 gid "category"로 표시해 4단어까지 따로 본다
        if not is_tick_or_axis and NUM_RE.search(s) and t.get_gid() not in allowed_num_gids:
            out["loose_numbers"].append(s[:30])         # 그림 안 수치(R-25)
    rects = [p for p in fig.findobj(Patch) if isinstance(p, (Rectangle, FancyBboxPatch)) and p.get_visible()
             and p not in [a.patch for a in fig.axes] and p is not fig.patch and p.get_gid() not in ("allowed_frame", "key_backing")]
    for p in rects:
        pb = p.get_window_extent(rend)
        if pb.width < 2 or pb.height < 2:
            continue
        for t in texts:
            tb = t.get_window_extent(rend)
            if pb.x0 <= tb.x0 and tb.x1 <= pb.x1 and pb.y0 <= tb.y0 and tb.y1 <= pb.y1:
                out["boxed_text"] += 1
    for ln in fig.findobj(Line2D):
        if ln.get_visible() and ln.get_linestyle() not in ("None", "none", "") and ln.get_linewidth() < min_lw - 1e-6:
            out["thin_lines"].append((ln.get_gid() or ln.get_label())[:20])
    for c in fig.findobj(Collection):
        lws = [w for w in (c.get_linewidths() if hasattr(c, "get_linewidths") else []) if w > 0]
        if lws and min(lws) < min_lw - 1e-6:
            out["thin_lines"].append(type(c).__name__)
    for a in fig.axes:
        vis = [sp for sp in a.spines.values() if sp.get_visible()]
        if vis and max(sp.get_linewidth() for sp in vis) < min_lw - 1e-6:
            out["thin_lines"].append("spine")
        for axis in (a.xaxis, a.yaxis):
            tk = axis.get_major_ticks()
            if tk and tk[0].tick1line.get_visible() and tk[0].tick1line.get_markeredgewidth() < min_lw - 1e-6:
                out["thin_lines"].append("tick")
        if a.get_title().strip():
            out["titles"].append(a.get_title())
    if fig._suptitle is not None and fig._suptitle.get_text().strip():
        out["titles"].append(fig._suptitle.get_text())
    out["fails"] = [k for k, bad in [("size", out["sizes"] - {size_pt, PANEL_PT} != set()), ("chars", out["chars"] > max_chars),
                    ("thin", bool(out["thin_lines"])), ("title", bool(out["titles"])), ("box", out["boxed_text"] > 0),
                    ("code", bool(out["codes"])), ("comma4", bool(out["comma4"])),
                    ("label", bool(out["long_labels"])), ("number", bool(out["loose_numbers"]))] if bad]
    return out


def char_size(o):
    """글자 크기. LTChar.size 는 가로쓰기 글꼴에서 bbox 높이를 돌려주므로,
    90° 회전한 글자(세로 축 라벨)에서는 진행 폭이 나온다. 진행 방향이 세로이면 bbox 폭을 쓴다.
    0°와 90° 회전에서 정확하고, 그 밖의 각도는 근사이다."""
    a, b = o.matrix[0], o.matrix[1]
    return o.width if abs(b) > abs(a) else o.height


def pdf_audit(pdf, size_pt=7.0, tol=0.05, max_chars=600):
    """최종 PDF 에서 글자 크기 분포, 문자 수, 글꼴, 쪽 크기(mm)를 잰다(지침 부록 A.2)."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTChar

    sizes, fonts, n = Counter(), set(), 0

    def walk(o):
        nonlocal n
        if isinstance(o, LTChar):
            if o.get_text().strip():
                sizes[round(char_size(o), 1)] += 1; fonts.add(o.fontname.split("+")[-1]); n += 1
        elif hasattr(o, "__iter__"):
            for c in o:
                walk(c)
    pages = list(extract_pages(str(pdf)))
    for p in pages:
        walk(p)
    w_mm, h_mm = pages[0].width / 72 * 25.4, pages[0].height / 72 * 25.4
    off = {s: c for s, c in sizes.items() if abs(s - size_pt) > tol}
    t3 = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    fam = {re.sub(r"-(Bold|Italic|BoldItalic|Regular)$", "", f) for f in fonts}
    return dict(width_mm=round(w_mm, 1), height_mm=round(h_mm, 1), chars=n, sizes=dict(sizes), off_size=off,
                lt5=sum(c for s, c in sizes.items() if s < 5.0), families=sorted(fam), type3="Type 3" in t3)


# ---------------------------------------------------------------- 수식 글꼴(추가, 2026-10-04: Fig 3·Fig 4)
def mathtext_liberation() -> None:
    """(이름은 호환용) mathtext 의 모든 글꼴을 FreeSans 계열로 맞춘다(한 글자 변수의 기울임 "$n$" 용).
    v4 부터 Liberation Sans 를 쓰지 않는다. PDF 에는 FreeSansOblique 가 같은 계열로 내장된다."""
    register_fonts()
    matplotlib.rcParams.update({
        "mathtext.fontset": "custom", "mathtext.rm": FONT_LATIN, "mathtext.it": f"{FONT_LATIN}:italic",
        "mathtext.bf": f"{FONT_LATIN}:bold", "mathtext.cal": FONT_LATIN, "mathtext.sf": FONT_LATIN,
        "mathtext.tt": FONT_LATIN, "mathtext.default": "it",
    })


mathtext_freesans = mathtext_liberation


# ---------------------------------------------------------------- 글자 겹침 점검(추가, 2026-10-04: 지침 2.2 '글자 겹침 0곳')
def text_overlaps(fig) -> dict:
    """보이는 글자끼리의 경계 상자 겹침 쌍과 캔버스 밖으로 나간 글자를 센다(글자와 자료의 겹침은 사람이 PNG 로 본다)."""
    import itertools
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    bbs = [(t.get_text().strip(), t.get_window_extent(rend)) for t in texts]
    pairs = [(a[:25], b[:25]) for (a, ba), (b, bb) in itertools.combinations(bbs, 2) if ba.overlaps(bb)]
    W, H = fig.get_size_inches() * fig.dpi
    outside = [s[:25] for s, b in bbs if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5]
    return dict(n_texts=len(bbs), overlap_pairs=pairs, outside=outside)


def audit_fig(fig, max_words=3):
    """digest 5.3 전문: 글씨 상자, 곡선·사선 화살표(FancyArrowPatch), 문장형 라벨, 축 제목을 센다(F-07 보조)."""
    from matplotlib.patches import FancyArrowPatch
    out = {"text_boxes": 0, "curved": 0, "diagonal": 0, "long_labels": [], "titles": []}
    for t in fig.findobj(Text):
        s = t.get_text().strip()
        if not s or not t.get_visible():
            continue
        if t.get_bbox_patch() is not None:
            out["text_boxes"] += 1
        if len(s.split()) > max_words or s.endswith((".", "다")):
            out["long_labels"].append(s[:40])
    for p in fig.findobj(FancyArrowPatch):
        cs = p.get_connectionstyle()
        if type(cs).__name__ in ("Angle3", "Arc", "Bar") or getattr(cs, "rad", 0) != 0:
            out["curved"] += 1
            continue
        ab = p._posA_posB
        if ab and abs(ab[0][0] - ab[1][0]) > 1e-9 and abs(ab[0][1] - ab[1][1]) > 1e-9:
            out["diagonal"] += 1
    out["titles"] = [a.get_title() for a in fig.axes if a.get_title().strip()]
    if fig._suptitle is not None and fig._suptitle.get_text().strip():
        out["titles"].append(fig._suptitle.get_text())
    return out


# ---------------------------------------------------------------- 문장 점검(digest 5.1, 지침 7.3 M-02)
DASH_RE = re.compile(r"(?<![0-9])\s*[—–]\s*(?![0-9])")           # 연결어 대시(숫자 범위 제외)
BANNED_EN_RE = re.compile(
    r"novel|groundbreaking|unprecedented|state-of-the-art|in this study,? we propose|the idea in one line|big picture"
    r"|two things to remember|honesty note|sits at|ships with|field-side|standard tool|parameter-free|takeaway"
    r"|all ten learners|(?<!internally )pre-registered|\bsafe\b|leverage|harness|unlock|comprehensive|robust"
    r"|crucial|pivotal|notably|importantly|interestingly|this figure", re.I)
MIDDOT_NUM_RE = re.compile(r"[0-9](\.[0-9]+)?\s*·\s*[0-9]")


def text_audit(text: str) -> dict:
    """원고·설명문 문자열의 연결어 대시, 금지 어휘, 가운뎃점 숫자 나열, 네 자리 쉼표, 내부 코드를 센다."""
    def hits(rx):
        return [m.group(0) for m in rx.finditer(text)]
    return dict(dash=hits(DASH_RE), banned=hits(BANNED_EN_RE), middot_num=hits(MIDDOT_NUM_RE),
                comma4=hits(COMMA4_RE), codes=hits(CODE_RE))
