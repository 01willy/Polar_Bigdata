"""v3 논문 그림·표 공용 양식 모듈(재구성 비교판).

근거: design/journal_grade_style_guide.md(이하 '지침') 2절(그림), 3절(표), 2.14절(상수), 부록 A(점검 함수),
outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md 1절(공통 규격).

- v2 모듈(scripts/4_visualization/paper/)과 src/polar/paperstyle.py 는 가져오지 않는다(지침 2.14 의 v3 값을 여기에 둔다).
- 산출 폴더: outputs/figures/paper/v3_restructure/ (사용자 결정 2026-10-04, 그림 명세 1.7).
- 그림 함수는 medium="paper" 또는 "slide" 를 받아 같은 자료·색·범위로 두 매체를 그린다(지침 0.4 3).
- 여러 그림 모듈이 함께 쓰므로 이름을 바꾸거나 지우지 말고 더하기만 한다.
"""
from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import Collection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

# ---------------------------------------------------------------- 경로
ROOT = Path(__file__).resolve().parents[3]
PAPER_FIGS = ROOT / "data" / "processed" / "paper_figs"          # 그림용 원천 표(등록)
OUT = ROOT / "outputs" / "figures" / "paper" / "v3_restructure"  # v3 비교 폴더
SOURCE_DATA = OUT / "source_data"                                # 패널 값(그림 명세 1.7)

# ---------------------------------------------------------------- 크기(지침 2.1, 2.14)
MM_PER_IN = 25.4
W2_MM, W15_MM, W1_MM = 170.0, 120.0, 85.0
HMAX_MM, HTARGET_MM = 180.0, 165.0
FONT_PT = 7.0                     # 모든 글자 7 pt(지침 2.2, R-04)
MIN_LW_PT = 1.0                   # 모든 선 1.0 pt 이상(지침 2.3, R-07)

# 선 굵기(pt)와 마커(pt), 지침 2.3 과 2.14
LW = dict(axis=1.0, tick=1.0, main=1.25, aux=1.0, ref=1.0, ci=1.0, ci_forest_cell=2.0, ci_forest_block=1.0,
          hatch=1.0, grid_map=1.0, scale_bar=1.5, marker_edge_open=1.0)
MS = dict(main=3.5, region_point=2.5)
TICK_LEN_PT = 3.0
CI_BAND_ALPHA = 0.20
FOREST_BLOCK_OFFSET_MM = 0.8      # 블록 등가중 막대를 셀 가중 막대 아래(또는 오른쪽) 0.8 mm

# ---------------------------------------------------------------- 색(지침 2.4, 2.6, 그림 명세 1.3, 1.4)
INK = "#000000"
INK_AUX = "#4d4d4d"               # 보조 표지·방향 표지·0선
EQUIV_BAND = "#ededed"            # ±0.5 cm 동등 띠
EQUIV_HALF_WIDTH_CM = 0.5
ZERO_LINE = dict(color="#4d4d4d", lw=1.0, ls="-")

# 방법 색 코드: 키는 코드 안에서만 쓰는 이름이다. 그림 글자에는 short(그림 짧은 이름)만 쓴다(H3).
METHOD = {
    "source_stefan":       dict(short="Source Stefan", color="#4d4d4d", ls="-"),
    "year_matched_stefan": dict(short="Year-matched Stefan", color="#4d4d4d", ls=(0, (1.2, 1.4))),
    "recalibrated_stefan": dict(short="Recalibrated Stefan", color="#2b5c8f", ls="-"),
    "soil_adjusted_stefan": dict(short="Soil-adjusted Stefan", color="#2b5c8f", ls=(0, (1.2, 1.4))),  # SI 전용
    "anchor_residual":     dict(short="Anchor + residual ML", color="#9a7bc9", ls="-"),
    "source_anchor_residual": dict(short="Source anchor + residual", color="#9a7bc9", ls=(0, (4, 2))),
    "physics_pseudo":      dict(short="Physics pseudo-labels", color="#568f72", ls=(0, (5, 1.5, 1.5, 1.5))),
    "direct_ml":           dict(short="Direct ML", color="#6b7280", ls=(0, (2, 1.5))),
    "physics_input":       dict(short="Physics-input ML", color="#ad921a", ls=(0, (7, 2))),
    "stefan_cci_anchor":   dict(short="Stefan-CCI anchor", color="#84480c", ls=(0, (5, 1.5, 1.5, 1.5, 1.5, 1.5))),
    "target_cv_selection": dict(short="Target-label CV selection", color=None, ls="-"),  # 고유 색 없음(검정 요소로만)
    "error_floor":         dict(short="Error floor", color="#b0b0b0", ls="-"),
}
METHOD_HEX = {"#4d4d4d", "#2b5c8f", "#9a7bc9", "#568f72", "#6b7280", "#ad921a", "#84480c"}
ACCENT_FORBIDDEN = "#EA851B"      # 덱 강조색, 자료 표현 금지

# 연속 색표 이름(cmcrameri), 지침 2.6
CMAP = dict(alt="oslo_r", delta="broc", pred_change="broc", residual="bam", width="acton_r",
            density="davos_r", ground_temp="vik")
BASEMAP = dict(continuous="#d0d0d0", discontinuous="#e6e6e6", land="#f4f4f4", sea="#ffffff",
               label_block="#bdbdbd", score_block="#737373", graticule="#e0e0e0", coast="#bdbdbd",
               candidate="#bdbdbd", selected="#000000", nodata="#B8BEC6", hatch="#9a9a9a")

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
    """지침 2.2–2.3(논문)과 5.3(슬라이드) 값으로 rcParams 를 맞춘다."""
    if medium not in ("paper", "slide"):
        raise ValueError(medium)
    paper = medium == "paper"
    fs = FONT_PT if paper else 16.0
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans"] if paper else ["Pretendard"],
        "font.size": fs, "axes.labelsize": fs if paper else 18.0, "xtick.labelsize": fs, "ytick.labelsize": fs,
        "legend.fontsize": fs, "axes.titlesize": fs, "figure.titlesize": fs,
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans" if paper else "Pretendard",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": LW["axis"] if paper else 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": LW["tick"] if paper else 1.5, "ytick.major.width": LW["tick"] if paper else 1.5,
        "xtick.major.size": TICK_LEN_PT if paper else 6.0, "ytick.major.size": TICK_LEN_PT if paper else 6.0,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "lines.linewidth": LW["main"] if paper else 3.0, "lines.markersize": MS["main"] if paper else 8.0,
        "lines.markeredgewidth": 0.0, "patch.linewidth": 1.0,
        "hatch.linewidth": LW["hatch"], "hatch.color": BASEMAP["hatch"],
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
    """7 pt 굵은 직립 소문자 패널 문자(지침 2.2). 슬롯 왼쪽 위 끝."""
    W, H = fig.get_size_inches() * MM_PER_IN
    return fig.text(x_mm / W, 1 - y_top_mm / H, letter, fontsize=FONT_PT, fontweight="bold", ha="left", va="top")


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


def audit_v3(fig, size_pt=7.0, min_lw=1.0, max_chars=600, max_words=3, allowed_num_gids=("scale", "sizekey")):
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
             and p not in [a.patch for a in fig.axes] and p is not fig.patch and p.get_gid() not in ("allowed_frame",)]
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
    out["fails"] = [k for k, bad in [("size", out["sizes"] - {size_pt} != set()), ("chars", out["chars"] > max_chars),
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
    """mathtext 의 모든 글꼴을 Liberation Sans 계열로 맞춘다(한 글자 변수의 기울임 "$n$" 용, 지침 2.2).
    use_v3() 는 mathtext.rm 만 지정하므로 cal·it·bf·sf·tt 가 기본값(cursive 등)으로 남아 findfont 경고가 난다.
    use_v3("paper") 뒤에 부른다. PDF 에는 LiberationSans-Italic 이 같은 계열로 내장된다(pdf_audit families 1개)."""
    matplotlib.rcParams.update({
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold", "mathtext.cal": "Liberation Sans", "mathtext.sf": "Liberation Sans",
        "mathtext.tt": "Liberation Sans", "mathtext.default": "it",
    })


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
