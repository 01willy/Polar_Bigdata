"""방법 도식 v4(국문 슬라이드판). 명세: deck/assets/paper_report/method/METHOD_FIGS_SPEC.md

그림마다 함수 하나(m1_problem ... m9_deepsets). 좌표는 인치(캔버스 축 xlim 0–W, ylim 0–H), 글자는 pt.
양식: design/style_tokens_v4.json(방법 색, 지역 색, 공정 채움, 글꼴). 대회 덱 v5 승인 문법(상자는 실자료 그림·공정 셰브런·
신경망 블록만, 직선 화살표, 학습 전용 경로는 붉은 파선). 수치는 명세 1절의 기록 문서 값만 쓴다.
실자료 축소 그림은 v3 논문 그림 모듈의 자료 함수(fig1.lena_design, fig3.load_concept)와 원천 표(paper_figs/*.csv)를 읽기만 한다.

실행: OMP_NUM_THREADS=4 nice -n 10 python3 scripts/4_visualization/method_figs_v4.py [--only M4,M6]
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "deck" / "assets" / "paper_report" / "method"
PF = ROOT / "data" / "processed" / "paper_figs"
TOK = json.loads((ROOT / "design" / "style_tokens_v4.json").read_text())
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3"))
sys.path.insert(0, str(ROOT / "src"))

# ================================================================ 토큰
_C = TOK["color"]
_M = _C["methods"]
INK = _C["neutral"]["text_slide"]          # #1A1A1A
AUX = _C["neutral"]["text_aux"]            # #555555
ZERO = _C["neutral"]["zero_line"]          # #4D4D4D
BAND = _C["neutral"]["equiv_band"]         # #ECECEC
LAND = _C["neutral"]["land"]               # #F2F2F2
COAST = _C["neutral"]["coast"]             # #BDBDBD
HAIR = "#BDBDBD"                           # 헤어라인·자연 테두리
SEP = "#DDDDDD"                            # 열 구분 헤어라인(덱 헤어라인과 같은 값)
HEAD = _C["deck"]["title_orange"]          # 열 머리 주황 #EA851B(자료 표현에는 쓰지 않는다)
YEL = _C["process"]["learning"]            # #F7E9AE 학습·신경망 블록
GRN = _C["process"]["evaluation_physics"]  # #DCEBDD 물리·평가 셰브런
YEL_EDGE = "#C9AE5B"
GRN_EDGE = "#86AE8A"
TRAIN_RED = "#C0392B"                      # 학습 때만 쓰는 경로(붉은 파선)
ARROW = ZERO
COL = dict(P0=_M["P0"]["hex"], P1=_M["P1"]["hex"], Pstar=_M["Pstar"]["hex"], R1=_M["R1"]["hex"], W=_M["W"]["hex"],
           D0=_M["D0"]["hex"], D1=_M["D1"]["hex"], F1=_M["F1"]["hex"], placebo=_M["placebo"]["hex"])
NAME = dict(P0=_M["P0"]["ko"], P1=_M["P1"]["ko"], R1=_M["R1"]["ko"], D0=_M["D0"]["ko"], D1=_M["D1"]["ko"], F1=_M["F1"]["ko"],
            W=_M["W"]["ko"])
REG = {k: v for k, v in _C["regions"].items() if k != "note"}
REG_KO = {"Alaska": "알래스카", "Lena Delta": "레나델타", "Canada": "캐나다", "W Russia": "러시아 서부", "E Russia": "러시아 동부",
          "Central Russia": "러시아 중부", "Tibetan Plateau": "티베트 고원"}
# fig1_blocks.csv 의 region 값 → 토큰 지역 이름
BLK2REG = {"Alaska": "Alaska", "Lena": "Lena Delta", "Canada": "Canada", "Russia_W": "W Russia", "Russia_E": "E Russia",
           "Russia_C": "Central Russia", "Tibet_LGD": "Tibetan Plateau"}

FS = dict(head=15.0, body=13.5, small=12.0, chev=14.0, panel=14.0)   # pt
DPI = 300
LOG: dict = {}


# ================================================================ 글꼴
FONT_CACHE = Path.home() / ".cache" / "polar_fonts"


def _otf_to_ttf(src: Path, dst: Path):
    """Pretendard OTF(CFF) → TrueType(2차 곡선). matplotlib PDF 가 CFF 글꼴을 묻지 못해 바꾼다(fontTools otf2ttf 방식)."""
    from fontTools.ttLib import TTFont, newTable
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    f = TTFont(str(src))
    order = f.getGlyphOrder()
    gs = f.getGlyphSet()
    glyphs = {}
    for g in gs.keys():
        tp = TTGlyphPen(gs)
        gs[g].draw(Cu2QuPen(tp, 1.0, reverse_direction=True))
        glyphs[g] = tp.glyph()
    f["loca"] = newTable("loca")
    f["glyf"] = gl = newTable("glyf")
    gl.glyphOrder = order
    gl.glyphs = glyphs
    del f["CFF "]
    if "VORG" in f:
        del f["VORG"]
    gl.compile(f)
    f["maxp"] = mx = newTable("maxp")
    mx.tableVersion = 0x00010000
    for k in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
              "maxSizeOfInstructions", "maxComponentElements"):
        setattr(mx, k, 1 if k == "maxZones" else 0)
    post = f["post"]
    post.formatType = 2.0; post.extraNames = []; post.mapping = {}; post.glyphOrder = order
    f.sfntVersion = "\000\001\000\000"
    dst.parent.mkdir(parents=True, exist_ok=True)
    f.save(str(dst))


def setup_fonts():
    """라틴 FreeSans, 한글 Pretendard(글리프 대체). font.family 를 목록으로 줘야 Agg·PDF 모두 대체가 된다.
    수식(mathtext) 문자열 안에서는 대체가 되지 않으므로 한글과 수식은 RT() 로 따로 놓는다."""
    for w in ("Regular", "Bold"):
        dst = FONT_CACHE / f"PretendardTT-{w}.ttf"
        if not dst.exists():
            _otf_to_ttf(Path.home() / ".fonts" / f"Pretendard-{w}.otf", dst)
        fm.fontManager.addfont(str(dst))
    for f in (TOK["font"]["latin_regular_file"], TOK["font"]["latin_bold_file"],
              "/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf"):
        fm.fontManager.addfont(f)
    matplotlib.rcParams.update({
        "font.family": ["FreeSans", "Pretendard"], "font.weight": "normal",
        "font.size": FS["body"], "axes.unicode_minus": True,
        "mathtext.fontset": "custom", "mathtext.rm": "FreeSans", "mathtext.it": "FreeSans:italic",
        "mathtext.bf": "FreeSans:bold", "mathtext.sf": "FreeSans", "mathtext.default": "it",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


# ================================================================ 캔버스와 기본 요소(인치 좌표)
def canvas(W, H):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    fig._W, fig._H = W, H
    return fig, ax


def T(ax, x, y, s, size=None, color=INK, ha="left", va="center", weight="normal", z=6, **kw):
    t = ax.text(x, y, s, fontsize=size or FS["body"], color=color, ha=ha, va=va, fontweight=weight, zorder=z, **kw)
    return t


def RT(ax, x, y, segs, size=None, color=INK, ha="left", z=6, gap_em=0.28):
    """한글과 수식을 나란히 놓는 글줄. segs = 문자열 목록('$..$' 는 수식). 기준선을 맞추고, y 는 글줄 가운데 높이.
    한 Text 안에 한글과 수식을 섞으면 수식 해석기가 한글 글리프를 찾지 못한다(글꼴 대체 없음)."""
    size = size or FS["body"]
    fig = ax.figure
    r = fig.canvas.get_renderer()
    yb = y - 0.33 * size / 72.0
    arts, cx = [], x
    colors = color if isinstance(color, (list, tuple)) else [color] * len(segs)
    for s_, c_ in zip(segs, colors):
        gap = gap_em * size / 72.0 if s_.endswith(" ") else 0.0
        lead = gap_em * size / 72.0 if s_.startswith(" ") else 0.0
        cx += lead
        t = ax.text(cx, yb, s_.strip(), fontsize=size, color=c_, ha="left", va="baseline", zorder=z)
        w = t.get_window_extent(r).width / fig.dpi
        arts.append(t)
        cx += w + gap
    total = cx - x
    shift = {"left": 0.0, "center": -total / 2, "right": -total}[ha]
    if shift:
        for t in arts:
            t.set_x(t.get_position()[0] + shift)
    return arts, total


def head(ax, x, y, s, ha="left", size=None):
    """열 머리: 덱 제목 주황, 굵게(그림 안 굵은 글씨는 열 머리와 패널 문자만)."""
    return T(ax, x, y, s, size=size or FS["head"], color=HEAD, ha=ha, va="center", weight="bold")


def panel_letter(ax, x, y, s):
    return T(ax, x, y, s, size=FS["panel"], color=INK, ha="left", va="center", weight="bold")


def hline(ax, x0, x1, y, color=HAIR, lw=0.8, ls="-", z=2):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=z)


def vline(ax, x, y0, y1, color=HAIR, lw=0.8, ls="-", z=2):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=z)


def arrow(ax, p0, p1, color=ARROW, lw=1.1, ls="-", head=True, hl=0.085, hw=0.062, z=5):
    """수평·수직 직선 화살표. 사선은 거부한다(대회 덱 v5 문법)."""
    (x0, y0), (x1, y1) = p0, p1
    if abs(x0 - x1) > 1e-9 and abs(y0 - y1) > 1e-9:
        raise ValueError(f"사선 화살표 금지: {p0} → {p1}")
    L = np.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    xe, ye = (x1 - ux * hl, y1 - uy * hl) if head else (x1, y1)
    ax.plot([x0, xe], [y0, ye], color=color, lw=lw, ls=ls, solid_capstyle="butt", dash_capstyle="butt", zorder=z)
    if head:
        px, py = -uy, ux
        tri = [(x1, y1), (xe + px * hw / 2, ye + py * hw / 2), (xe - px * hw / 2, ye - py * hw / 2)]
        ax.add_patch(Polygon(tri, closed=True, facecolor=color, edgecolor="none", zorder=z))
    LOG.setdefault("_arrows", []).append(dict(p0=[round(x0, 3), round(y0, 3)], p1=[round(x1, 3), round(y1, 3)]))


def chevron(ax, x, y, w, h, fill, text=None, first=False, tip=0.17, size=None, edge=None, tcolor=INK):
    """공정 셰브런(왼쪽 홈, 오른쪽 뾰족). first=True 는 왼쪽이 평평하다."""
    edge = edge or (YEL_EDGE if fill == YEL else GRN_EDGE)
    pts = [(x, y), (x + w - tip, y), (x + w, y + h / 2), (x + w - tip, y + h), (x, y + h)]
    if not first:
        pts.append((x + tip, y + h / 2))
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, lw=0.9, joinstyle="miter", zorder=3))
    if text:
        cx = x + (w - tip) / 2 + (0 if first else tip / 2)
        T(ax, cx, y + h / 2, text, size=size or FS["chev"], ha="center", va="center", color=tcolor, linespacing=1.15)


def block(ax, x, y, w, h, text=None, fill=YEL, edge=None, size=None, r=0.035, lw=0.9, tcolor=INK, z=3, **kw):
    """신경망·학습기 블록(노랑) 또는 물리 계산 블록(초록). (x, y) = 왼쪽 아래."""
    edge = edge or (YEL_EDGE if fill == YEL else GRN_EDGE)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fill, edgecolor=edge,
                                lw=lw, zorder=z))
    if text is not None:
        T(ax, x + w / 2, y + h / 2, text, size=size or FS["body"], ha="center", va="center", color=tcolor, z=z + 1,
          linespacing=1.15, **kw)


def key_line(ax, x, y, color, w=0.32, lw=2.6, ls="-"):
    ax.plot([x, x + w], [y, y], color=color, lw=lw, ls=ls, solid_capstyle="butt", dash_capstyle="butt", zorder=5)


def frame(axt, color=HAIR, lw=0.7):
    for s in axt.spines.values():
        s.set_visible(True); s.set_edgecolor(color); s.set_linewidth(lw)


def sub_axes(fig, x, y, w, h, **kw):
    W, H = fig._W, fig._H
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


# ================================================================ 실자료
_CACHE: dict = {}


def blocks_table():
    if "blocks" not in _CACHE:
        b = pd.read_csv(PF / "fig1_blocks.csv")
        # Table 1 과 같은 구성: 주 4지역·알래스카는 기존 표(v3), 새 지역은 약관 확인 추가 행(러시아 중부, 티베트 고원)
        keep = ((b.kind == "v3") & b.region.isin(["Alaska", "Lena", "Canada", "Russia_W", "Russia_E"])) | \
               ((b.region == "Russia_C")) | ((b.kind == "lgd_added") & (b.region == "Tibet_LGD"))
        b = b[keep].copy()
        b["reg"] = b.region.map(BLK2REG)
        _CACHE["blocks"] = b
    return _CACHE["blocks"]


def table1():
    if "t1" not in _CACHE:
        _CACHE["t1"] = pd.read_csv(PF / "table1_rows.csv")
    return _CACHE["t1"]


def lena():
    if "lena" not in _CACHE:
        import fig1  # v3 논문 그림 모듈(읽기만)
        vals = {}
        L = fig1.lena_design(vals)
        proj, ext, lon_c, lat_c = fig1.zoom_projection(L)
        _CACHE["lena"] = dict(L=L, vals=vals, proj=proj, ext=ext, fig1=fig1)
    return _CACHE["lena"]


def concept():
    if "concept" not in _CACHE:
        import fig3
        _CACHE["concept"] = fig3.load_concept()
    return _CACHE["concept"]


def land_feature(scale="50m"):
    import cartopy.feature as cfeature
    return cfeature.LAND.with_scale(scale)


def polar_map(fig, x, y, w, h, color_by_region=True, size_k=3.2, highlight=None, show_frame=True, lat_min=50.0,
              mono=None, hollow_regions=(), alpha=0.9, edge_lw=0.5, hollow_color=None):
    """범북극 라벨 지도(Fig 1a 와 같은 블록 자료). 원 면적 ∝ 블록의 1 km 라벨 위치 수. 원형 경계 지도."""
    import cartopy.crs as ccrs
    import matplotlib.path as mpath
    proj = ccrs.NorthPolarStereo(central_longitude=127.0, true_scale_latitude=70.0)
    axm = sub_axes(fig, x, y, w, h, projection=proj)
    axm.set_extent([-180, 180, lat_min, 90], ccrs.PlateCarree())
    th = np.linspace(0, 2 * np.pi, 200)
    circ = mpath.Path(np.c_[np.sin(th), np.cos(th)] * 0.5 + 0.5)
    axm.set_boundary(circ, transform=axm.transAxes)
    axm.add_feature(land_feature("50m"), facecolor=LAND, edgecolor="none", zorder=0).set_rasterized(True)
    axm.spines["geo"].set_edgecolor(HAIR if show_frame else "none"); axm.spines["geo"].set_linewidth(0.7)
    b = blocks_table()
    b = b[b.lat >= lat_min]
    order = ["Alaska", "Canada", "Lena Delta", "W Russia", "E Russia", "Central Russia"]
    for r in order:
        d = b[b.reg == r]
        if not len(d):
            continue
        c = mono or (REG[r] if color_by_region else INK)
        if highlight is not None and r != highlight and highlight != "all":
            pass
        fc = "white" if r in hollow_regions else c
        ec = (hollow_color or c) if r in hollow_regions else "white"
        axm.scatter(d.lon.values, d.lat.values, s=size_k * np.sqrt(d.n_loc_1km.values) * 4.0, facecolor=fc, edgecolor=ec,
                    linewidths=edge_lw if r not in hollow_regions else 1.2, alpha=alpha, transform=ccrs.PlateCarree(), zorder=3)
    return axm, proj


def lena_zoom(fig, x, y, w, h, mode="design", label_size=None, scale_bar=True, buffer=True, dots=True):
    """레나델타 확대도. mode = 'holdout'(대상 셀 + 100 km 완충) 또는 'design'(라벨·채점 블록 + 라벨 40개)."""
    import cartopy.crs as ccrs
    from matplotlib.collections import PolyCollection
    D = lena()
    L, proj, ext, f1 = D["L"], D["proj"], D["ext"], D["fig1"]
    axm = sub_axes(fig, x, y, w, h, projection=proj)
    axm.set_xlim(ext[0], ext[1]); axm.set_ylim(ext[2], ext[3])
    axm.add_feature(land_feature("10m"), facecolor=LAND, edgecolor="none", zorder=0).set_rasterized(True)
    frame(axm); axm.spines["geo"].set_edgecolor(HAIR); axm.spines["geo"].set_linewidth(0.7)
    d = L["df"].iloc[L["t_idx"]]
    out = dict(ax=axm, proj=proj, ext=ext)
    if mode in ("design", "both"):
        pa, _ = f1.block_polys(list(L["A_blocks"]), proj)
        pb, _ = f1.block_polys(list(L["score_blocks"]), proj)
        axm.add_collection(PolyCollection(pa, facecolors=LBL_BLOCK, edgecolors="white", linewidths=0.8, zorder=2))
        axm.add_collection(PolyCollection(pb, facecolors=SCO_BLOCK, edgecolors="white", linewidths=0.8, zorder=2))
        out["pa"], out["pb"] = pa, pb
        lab = L["lab"]
        if dots:
            axm.scatter(lab.lon.values, lab.lat.values, s=7.0, color=INK, linewidths=0, transform=ccrs.PlateCarree(), zorder=4)
    if mode in ("holdout", "both"):
        if dots:
            axm.scatter(d.lon.values, d.lat.values, s=1.6, color=INK, linewidths=0, transform=ccrs.PlateCarree(), zorder=3,
                        rasterized=True)
    if buffer:
        from sklearn.neighbors import BallTree
        ng = 300
        X, Y = np.meshgrid(np.linspace(ext[0], ext[1], ng), np.linspace(ext[2], ext[3], ng))
        ll = ccrs.PlateCarree().transform_points(proj, X, Y)
        uq = d[["lat", "lon"]].drop_duplicates().values
        tree = BallTree(np.radians(uq), metric="haversine")
        dist, _ = tree.query(np.radians(np.c_[ll[..., 1].ravel(), ll[..., 0].ravel()]), k=1)
        Dkm = dist[:, 0].reshape(X.shape) * 6371.0
        cs = axm.contour(X, Y, Dkm, levels=[100.0], colors=[INK], linewidths=[1.0], linestyles=[(0, (3, 2))], zorder=4)
        out["buffer_verts"] = np.vstack([p.vertices for p in cs.get_paths() if len(p.vertices)])
    if scale_bar:
        x0 = ext[0] + 0.07 * (ext[1] - ext[0]); y0 = ext[2] + 0.07 * (ext[3] - ext[2])
        lon, lat = ccrs.PlateCarree().transform_point(x0, y0, proj)
        k = (1 + np.sin(np.radians(70.0))) / (1 + np.sin(np.radians(lat)))
        Lm = 100 * 1e3 * k
        axm.plot([x0, x0 + Lm], [y0, y0], color=INK, lw=2.0, solid_capstyle="butt", zorder=8)
        axm.text(x0 + Lm / 2, y0 + 0.02 * (ext[3] - ext[2]), "100 km", ha="center", va="bottom", fontsize=label_size or FS["small"],
                 color=INK, zorder=8)
    return out


LBL_BLOCK = "#EFD98A"   # 라벨 블록(학습에 쓰는 절반, 학습 노랑 계열을 진하게)
SCO_BLOCK = "#9DC7A4"   # 채점 블록(평가 초록 계열을 진하게)


def alaska_png(panel="c"):
    """알래스카 1 km ALT 지도 슬라이드판 PNG 에서 지도 부분을 자른다(컬러바·글자 제외)."""
    from PIL import Image
    if panel == "c_single":
        im = Image.open(ROOT / "deck/assets/paper_report/slide_panels/Alaska_ALT_map_v3_c_slide.png").convert("RGB")
        a = np.asarray(im)
        return a[60:1255, 160:1490]
    im = Image.open(ROOT / "deck/assets/paper_report/maps/Alaska_ALT_map_v3_slide.png").convert("RGB")
    a = np.asarray(im)
    # 네 패널의 검정 테두리 찾기: 세로로 길게 이어진 어두운 열(지도 안 어두운 화소와 구분)
    dark = (a.astype(int).sum(2) < 120)
    ys = np.where(dark.mean(1) > 0.8)[0]                 # 네 패널을 가로지르는 위·아래 테두리 행
    ytop, ybot = int(ys[ys < dark.shape[0] / 2].max()), int(ys[ys > dark.shape[0] / 2].min())
    colfrac = dark[ytop + 10: ybot - 10].mean(0)
    xs = np.where(colfrac > 0.95)[0]
    groups = np.split(xs, np.where(np.diff(xs) > 5)[0] + 1)
    edges = [(int(g.min()), int(g.max())) for g in groups]
    pairs = [(edges[i][1], edges[i + 1][0]) for i in range(0, len(edges) - 1, 2)]
    idx = dict(a=0, b=1, c=2, d=3)[panel]
    x0, x1 = pairs[idx]
    return a[ytop + 3: ybot - 2, x0 + 3: x1 - 2]


def img_thumb(fig, x, y, w, h, arr, framed=True):
    axt = sub_axes(fig, x, y, w, h)
    axt.imshow(arr, interpolation="lanczos", aspect="auto")
    axt.set_xticks([]); axt.set_yticks([])
    if framed:
        frame(axt)
    else:
        for s in axt.spines.values():
            s.set_visible(False)
    return axt


def fit_box(arr_shape, w, h):
    """이미지 종횡비를 지키며 (w, h) 상자 안 최대 크기."""
    ih, iw = arr_shape[:2]
    r = iw / ih
    if w / h > r:
        return h * r, h
    return w, w / r


def stefan_scatter(fig, x, y, w, h, numbers=False, show_recal=True, size=None):
    """러시아 서부 31셀의 ALT 대 √TDD(Fig 3a 자료), 원천 계수와 라벨 10개 재보정 직선. 썸네일은 숫자 눈금 없음."""
    C = concept()
    axs = sub_axes(fig, x, y, w, h)
    s, yv = C["s"], C["y"]
    xm = np.nanmax(s) * 1.08
    xs = np.array([0, xm])
    axs.plot(xs, C["E0"] * xs, color=COL["P0"], lw=2.2, solid_capstyle="butt", zorder=3)
    if show_recal:
        axs.plot(xs, C["E1"] * xs, color=COL["P1"], lw=2.2, solid_capstyle="butt", zorder=3)
    axs.scatter(s, yv, s=14, facecolor="white", edgecolor=INK, linewidths=0.8, zorder=4)
    axs.scatter(C["s_sel"], C["y_sel"], s=14, facecolor=INK, edgecolor=INK, linewidths=0.8, zorder=5)
    axs.set_xlim(np.nanmin(s) * 0.85, xm); axs.set_ylim(0, max(np.nanmax(yv), C["E1"] * xm) * 1.05)
    axs.invert_yaxis()
    for k in ("top", "right"):
        axs.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        axs.spines[k].set_linewidth(1.0); axs.spines[k].set_color(INK)
    if not numbers:
        axs.set_xticks([]); axs.set_yticks([])
    axs.set_xlabel("√TDD", fontsize=size or FS["small"], labelpad=2)
    axs.set_ylabel("ALT", fontsize=size or FS["small"], labelpad=2)
    axs.patch.set_alpha(0)
    return axs


def overlay(fig):
    """지도 축 위에 그리는 투명 덧층(인치 좌표). 지도 축을 다 넣은 뒤 부른다."""
    W, H = fig._W, fig._H
    ov = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ov.set_xlim(0, W); ov.set_ylim(0, H); ov.set_aspect("equal"); ov.axis("off")
    return ov


def staircase(ax, xa, xb, y0, labels=("원천 계수 Stefan", "재보정 + 저가중 잔차", "교차검증 선정"), dy=0.50, size=None):
    """라벨 수 구간(0, 3–10, 40 이상; Fig 7d·M8 과 같은 구간) 위 권고 방법(도식). y0 = 축 높이."""
    size = size or FS["small"]
    cats = ("0", "3–10", "40 이상")
    tx = np.array([xa + 0.08, xa + 0.62, xb - 0.40])
    hline(ax, xa, xb, y0, color=INK, lw=1.0)
    for xv, lb in zip(tx, cats):
        vline(ax, xv, y0 - 0.06, y0, color=INK, lw=1.0)
        T(ax, xv, y0 - 0.20, lb, size=size, ha="center")
    segs = [(COL["P0"], labels[0], 3, None), (COL["R1"], labels[1], 2, (tx[1] - 0.22, tx[1] + 0.22)),
            (COL["W"], labels[2], 1, (tx[2] - 0.30, xb))]
    for c, lb, lev, span in segs:
        y_ = y0 + 0.30 + (lev - 1) * dy
        if span is None:
            ax.scatter([tx[0]], [y_], s=46, color=c, zorder=5, linewidths=0)
        else:
            ax.plot(list(span), [y_, y_], color=c, lw=3.2, solid_capstyle="butt", zorder=5)
        T(ax, xa if lev > 1 else xb, y_ + 0.19, lb, size=size, color=INK, ha="left" if lev > 1 else "right")
    return tx


# ================================================================ 점검과 저장
def qa_and_save(fig, name, allow_overlap=()):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    Wpx, Hpx = fig._W * DPI, fig._H * DPI
    rec = dict(name=name, size_in=[fig._W, fig._H], min_font_pt=None, texts=0, outside=[], overlaps=[], small=[])
    boxes = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.width < 1:
            continue
        fs = t.get_fontsize()
        rec["texts"] += 1
        rec["min_font_pt"] = fs if rec["min_font_pt"] is None else min(rec["min_font_pt"], fs)
        if fs < 12.0 - 1e-6:
            rec["small"].append([t.get_text()[:30], fs])
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > Wpx + 1 or bb.y1 > Hpx + 1:
            rec["outside"].append(t.get_text()[:40])
        boxes.append((t.get_text(), bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ix = min(a.x1, b.x1) - max(a.x0, b.x0); iy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ix > 2 and iy > 2:
                pair = (boxes[i][0][:24], boxes[j][0][:24])
                if any(k in pair[0] or k in pair[1] for k in allow_overlap):
                    continue
                rec["overlaps"].append(list(pair))
    rec["arrows"] = len(LOG.pop("_arrows", []))
    png = OUT / f"{name}_slide.png"
    pdf = OUT / f"{name}_slide.pdf"
    fig.savefig(png, dpi=DPI)
    try:
        fig.savefig(pdf)
    except Exception as e:  # 글꼴 묻기 실패 시 Type 3 로
        matplotlib.rcParams["pdf.fonttype"] = 3
        fig.savefig(pdf)
        rec["pdf_fonttype"] = f"3 ({type(e).__name__})"
        matplotlib.rcParams["pdf.fonttype"] = 42
    plt.close(fig)
    from PIL import Image
    im = Image.open(png)
    rec["png_px"] = list(im.size)
    LOG[name] = rec
    status = "OK" if not (rec["outside"] or rec["overlaps"] or rec["small"]) else "CHECK"
    print(f"[{name}] {status} 글자 {rec['texts']} · 최소 {rec['min_font_pt']} pt · 화살표 {rec['arrows']} · "
          f"밖 {rec['outside']} · 겹침 {rec['overlaps'][:6]} · 작은 글자 {rec['small'][:4]}", flush=True)
    return rec


# ================================================================ M4 전체 워크플로
def m4_workflow():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    n = 6
    gap = 0.06
    cw = (W - 0.04 - gap * (n - 1)) / n          # 열 폭
    xs = [0.02 + i * (cw + gap) for i in range(n)]
    ych, hch = 4.42, 0.74                        # 셰브런
    steps = [("① 자료 구축", GRN), ("② 물리 기준선", GRN), ("③ ML 결합 방식", YEL), ("④ 평가 설계", GRN),
             ("⑤ 라벨 수별\n워크플로", YEL), ("⑥ 1 km 지도와\n제품 비교", GRN)]
    for i, (s_, f) in enumerate(steps):
        chevron(ax, xs[i], ych, cw + (0.0 if i == n - 1 else 0.10), hch, f, s_, first=(i == 0), size=FS["chev"])
    ty0, th = 2.05, 2.15          # 축소 그림 상자 아래·높이
    pad = 0.12
    tw = cw - 2 * pad
    # ① 범북극 라벨 지도(실자료, 지역 색)
    d = min(tw, th)
    polar_map(fig, xs[0] + pad + (tw - d) / 2, ty0 + (th - d) / 2, d, d, size_k=3.0)
    # ② Stefan 산점도(러시아 서부 31셀, 원천 계수와 라벨 10개 재보정 직선)
    stefan_scatter(fig, xs[1] + pad + 0.28, ty0 + 0.34, tw - 0.32, th - 0.38)
    # ③ 결합 구조 4종(도식): 방법 색 표지, 이름, 물리 정보가 들어가는 위치
    rows3 = [("D0", "직접 ML", "물리 정보 없음"), ("D1", "물리 유사라벨 증강", "학습 라벨에"), ("F1", "물리 입력 ML", "입력에"),
             ("R1", "물리 잔차 결합", "앵커 + 잔차")]
    y3 = ty0 + th - 0.12
    for k, (code, nm, where) in enumerate(rows3):
        yy = y3 - k * 0.54
        key_line(ax, xs[2] + pad, yy, COL[code], w=0.26, lw=3.0)
        T(ax, xs[2] + pad + 0.36, yy, nm, size=FS["body"], color=COL[code] if code == "R1" else INK)
        T(ax, xs[2] + pad + 0.36, yy - 0.24, where, size=FS["small"], color=AUX)
    # ④ 레나델타 지역 홀드아웃·블록 분할·라벨 40개(실자료), 아래 두 줄 열쇠
    d4 = min(tw, th - 0.52)
    lena_zoom(fig, xs[3] + pad + (tw - d4) / 2, ty0 + 0.52 + (th - 0.52 - d4) / 2, d4, d4, mode="design", scale_bar=False,
              buffer=True)
    kx = xs[3] + pad
    for yk, fc, lb in ((ty0 + 0.34, LBL_BLOCK, "라벨 블록 · 점 = 라벨"), (ty0 + 0.08, SCO_BLOCK, "채점 블록")):
        ax.add_patch(Rectangle((kx, yk - 0.07), 0.14, 0.14, facecolor=fc, edgecolor="none", zorder=5))
        T(ax, kx + 0.20, yk, lb, size=FS["small"], color=AUX)
    # ⑤ 라벨 수 구간 위 권고 방법(도식, Fig 7d 의 구간 이름)
    x5a, x5b = xs[4] + pad, xs[4] + cw - pad
    yax = ty0 + 0.34
    staircase(ax, x5a, x5b, yax)
    T(ax, (x5a + x5b) / 2, yax - 0.46, "라벨 수", size=FS["small"], ha="center", color=AUX)
    # ⑥ 알래스카 1 km ALT 지도(실자료)
    arr = alaska_png("c_single")
    w6, h6 = fit_box(arr.shape, tw, th)
    img_thumb(fig, xs[5] + pad + (tw - w6) / 2, ty0 + (th - h6) / 2, w6, h6, arr)
    # 설명 줄(명사구, 열 폭 안)
    yl = [1.70 - k * 0.33 for k in range(4)]
    lines = {
        0: [["평가 지역 7곳"], ["1 km 셀 라벨"], ["공변량 25종"]],
        1: [["$\mathrm{ALT} = E\sqrt{\mathrm{TDD}}$"], ["KEY:P0", "원천 계수 ", "$E_0$"], ["KEY:P1", "재보정 계수 ", "$E_n$"],
            ["$\kappa = 10$ 수축"]],
        2: [["같은 학습기 CatBoost"], ["물리 정보 위치 비교"]],
        3: [["지역 홀드아웃"], ["100 km 완충"], ["라벨 수 0개–전량"], ["블록 CI · 사전 등록"]],
        4: [["라벨 10개 편향 진단"], ["교차검증 선정"], ["라벨 분산 배치"]],
        5: [["알래스카 1 km 지도"], ["90% 구간"], ["학습 범위 밖 표시"], ["기존 ALT 지도와 비교"]],
    }
    # 라벨 축적 고리: ⑤ 아래 → 바닥 → ① 아래(직선, 화살촉 1개). 라벨이 늘면 ① 부터 다시
    ylo = 0.22
    xr5 = xs[4] + cw - 0.10
    xr1 = xs[0] + cw - 0.10
    vline(ax, xr5, ylo, 1.92, color=COL["R1"], lw=1.3)
    hline(ax, xr1, xr5, ylo, color=COL["R1"], lw=1.3)
    arrow(ax, (xr1, ylo), (xr1, 1.92), color=COL["R1"], lw=1.3)
    RT(ax, (xr1 + xr5) / 2, ylo + 0.20, ["라벨 추가 → 라벨 수 ", "$n$", " 증가 · 다음 관측은 블록·공변량 분산 배치"], size=FS["small"],
       ha="center", color=AUX)
    for i, L in lines.items():
        for k, segs_ in enumerate(L):
            if len(segs_) == 1 and "$" in segs_[0] and "수축" in segs_[0]:
                RT(ax, xs[i] + pad + 0.36, yl[k], ["$\kappa = 10$ ", "수축"], size=FS["body"])
            elif segs_[0].startswith("KEY:"):
                key_line(ax, xs[i] + pad, yl[k], COL[segs_[0][4:]], w=0.26, lw=3.0)
                RT(ax, xs[i] + pad + 0.36, yl[k], segs_[1:], size=FS["body"])
            else:
                RT(ax, xs[i] + pad, yl[k], segs_, size=FS["body"])
    return qa_and_save(fig, "M4_workflow")


# ================================================================ M6 비교한 방법의 구조
def plus_node(ax, x, y, r=0.10):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, lw=1.0, zorder=6))
    ax.plot([x - r * 0.55, x + r * 0.55], [y, y], color=INK, lw=1.0, zorder=7, solid_capstyle="butt")
    ax.plot([x, x], [y - r * 0.55, y + r * 0.55], color=INK, lw=1.0, zorder=7, solid_capstyle="butt")


def m6_models():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    yh = 4.98
    head(ax, 0.10, yh, "입력")
    head(ax, 2.05, yh, "결합 구조")
    head(ax, 10.20, yh, "학습기 비교")
    # 학습 전용 경로 열쇠
    ax.plot([7.40, 7.72], [yh, yh], color=TRAIN_RED, lw=1.4, ls=(0, (3, 2)), zorder=5)
    T(ax, 7.80, yh, "학습 때만 쓰는 경로", size=FS["small"], color=AUX)
    vline(ax, 1.92, 0.25, 5.12, color=SEP, lw=0.8)
    vline(ax, 10.05, 0.25, 5.12, color=SEP, lw=0.8)

    # ---- 입력 X: 공변량 25종 막대(지형 6, 기후 8, 토양 9, CCI 2)
    bx, bw, y_bot, y_top = 1.25, 0.30, 0.80, 4.40
    groups = [("지형", 6, "#D9E0E8"), ("기후", 8, "#B4C3D3"), ("토양", 9, "#8EA6BF"), ("CCI", 2, _C["products"]["CCI"]["hex"])]
    unit = (y_top - y_bot) / 25.0
    yc = y_bot
    for nm, k, c in groups:
        ax.add_patch(Rectangle((bx, yc), bw, k * unit, facecolor=c, edgecolor="white", lw=1.0, zorder=3))
        T(ax, bx - 0.10, yc + k * unit / 2, f"{nm} {k}", size=FS["body"], ha="right")
        if nm == "기후":
            T(ax, bx - 0.10, yc + k * unit / 2 - 0.27, "√TDD 포함", size=FS["small"], ha="right", color=AUX)
        yc += k * unit
    RT(ax, bx + bw / 2, y_bot - 0.30, ["공변량 25종"], size=FS["small"], ha="center", color=AUX)

    # ---- 방법 행(위에서 아래로 물리 정보 결합이 강한 순서)
    yL = [4.33, 3.41, 2.49, 1.57, 0.65]
    xin, xb0, xb1, xout, xnote = 2.12, 4.15, 4.95, 6.02, 7.62
    xdrop = 5.70
    bh = 0.36
    # 공용 분배선: 막대 → 분배선 → 각 행 입력
    xbus = 1.78
    ymid = (y_bot + y_top) / 2
    hline(ax, bx + bw, xbus, ymid, color=ARROW, lw=1.1)
    vline(ax, xbus, yL[-1], yL[0], color=ARROW, lw=1.1)
    for y in yL:
        arrow(ax, (xbus, y), (xin - 0.04, y))
    names = [
        [("P0", "원천 계수 Stefan"), ("P1", "재보정 Stefan")],
        [("R1", "물리 잔차 결합")],
        [("F1", "물리 입력 ML")],
        [("D1", "물리 유사라벨 증강")],
        [("D0", "직접 ML")],
    ]
    for y, nm in zip(yL, names):
        cx = xin
        for code, label in nm:
            key_line(ax, cx, y + 0.27, COL[code], w=0.24, lw=3.0)
            _, wtxt = RT(ax, cx + 0.32, y + 0.27, [label], size=FS["body"], color=COL["R1"] if code == "R1" else INK)
            cx += 0.32 + wtxt + 0.22
    # 행별 입력 글자
    inputs = [["$\\sqrt{\\mathrm{TDD}}$"], ["$X$"], ["$X$", ", ", "$a$"], ["$X$"], ["$X$"]]
    in_cols = [[INK], [INK], [INK, INK, COL["P1"]], [INK], [INK]]
    for y, segs_, cols_ in zip(yL, inputs, in_cols):
        _, wv = RT(ax, xin, y, segs_, size=FS["body"] + 1, color=cols_, gap_em=0.0)
        arrow(ax, (xin + wv + 0.08, y), (xb0, y))
    # 블록(물리 행은 초록 최소제곱, 학습 행은 노랑 f·g)
    blk_txt = ["최소제곱", "$g$", "$f$", "$f$", "$f$"]
    for k, (y, t_) in enumerate(zip(yL, blk_txt)):
        block(ax, xb0, y - bh / 2, xb1 - xb0, bh, t_, fill=GRN if k == 0 else YEL, size=FS["body"] + (0 if k == 0 else 2))
    # 학습 목표(블록 아래, 붉은 파선 위로)
    targets = [["라벨 ", "$y$", " (원천 셀, 대상 ", "$n$", "개)"], ["$y - a$"], ["$y$"],
               ["$y$", " + 유사라벨"], ["$y$"]]
    xbc = (xb0 + xb1) / 2
    for y, segs_ in zip(yL, targets):
        yt = y - 0.42
        RT(ax, xbc, yt, segs_, size=FS["small"] + 0.5, ha="center", color=INK)
        arrow(ax, (xbc, yt + 0.12), (xbc, y - bh / 2), color=TRAIN_RED, lw=1.3, ls=(0, (2.5, 1.6)), hl=0.06, hw=0.06)
    # 출력
    # 물리 행: a(x) = E√TDD 와 계수 식
    arrow(ax, (xb1, yL[0]), (xout - 0.42, yL[0]))
    RT(ax, xout - 0.36, yL[0], ["$a(x) = E\\,\\sqrt{\\mathrm{TDD}}$"], size=FS["body"] + 1, color=COL["P1"])
    RT(ax, xnote, yL[0] + 0.30, ["$E_0 = \\sum s\\,y \\,/ \\sum s^2$", "  원천 셀"], size=FS["body"])
    RT(ax, xnote, yL[0] - 0.02, ["$E_n = (n\\,E_{\\mathrm{ls}} + \\kappa E_0)\\,/\\,(n + \\kappa)$"], size=FS["body"])
    RT(ax, xnote, yL[0] - 0.34, ["$\\kappa = 10$", "  대상 라벨 ", "$n$", "개로 재보정"], size=FS["body"])
    # 잔차 행: g → ×λ → ⊕ ← a(위) → ŷ
    plus_node(ax, xdrop, yL[1])
    arrow(ax, (xb1, yL[1]), (xdrop - 0.11, yL[1]))
    T(ax, (xb1 + xdrop) / 2, yL[1] + 0.17, "$\\times\\,\\lambda$", size=FS["body"], ha="center")
    arrow(ax, (xdrop, yL[0] - 0.16), (xdrop, yL[1] + 0.11), color=COL["P1"], lw=1.3)
    arrow(ax, (xdrop + 0.11, yL[1]), (xout - 0.04, yL[1]))
    RT(ax, xout, yL[1], ["$\\hat{y} = a + \\lambda\\, g(X)$"], size=FS["body"] + 1, color=COL["R1"])
    # 나머지 행
    outs = {2: "$\\hat{y} = f(X, a)$", 3: "$\\hat{y} = f(X)$", 4: "$\\hat{y} = f(X)$"}
    for k, t_ in outs.items():
        arrow(ax, (xb1, yL[k]), (xout - 0.04, yL[k]))
        RT(ax, xout, yL[k], [t_], size=FS["body"] + 1)
    # 설정 줄(회색, 두 줄)
    notes = {
        1: [["$\\lambda \\in \\{0.25, 0.5, 1.0\\}$"], ["판정 기준 ", "$\\lambda = 0.25$"]],
        2: [["$a$", " 또는 Kudryavtsev·"], ["토양 물성 Stefan 출력"]],
        3: [["유사라벨 ", "$E_n\\sqrt{\\mathrm{TDD}}$"], ["대상 라벨 절반 셀 · 원천 행당 10개"]],
        4: [["물리 결합 없음"], ["√TDD·CCI 는 공변량 안"]],
    }
    for k, lines_ in notes.items():
        for j, segs_ in enumerate(lines_):
            RT(ax, xnote, yL[k] + 0.14 - j * 0.30, segs_, size=FS["small"],
               color=[COL["P1"] if sg == "$a$" else AUX for sg in segs_])
    # ---- 학습기 열(노랑 블록)
    learners = ["CatBoost", "CatBoost 큰 설정", "CatBoost 원천 CV 조정", "랜덤 포레스트", "TabPFN v2", "TabICL v2", "MLP",
                "다중 헤드 MLP", "FT-Transformer", "RealMLP"]
    lx, lw_, lh, lg = 10.20, 1.75, 0.29, 0.065
    ly = 4.62
    for k, nm in enumerate(learners):
        yk = ly - k * (lh + lg)
        block(ax, lx, yk - lh, lw_, lh, nm, fill=YEL, size=FS["small"])
    yb = ly - len(learners) * (lh + lg)
    T(ax, lx, yb - 0.16, "방법 비교 CatBoost", size=FS["small"], color=AUX)
    T(ax, lx, yb - 0.45, "(200회, 깊이 3)", size=FS["small"], color=AUX)
    T(ax, lx, yb - 0.74, "라벨 0개 직접 ML 10종", size=FS["small"], color=AUX)
    return qa_and_save(fig, "M6_models")


# ================================================================ M5 물리 유사라벨 증강 대조 설계
def m5_augmentation():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    yh = 4.98
    head(ax, 0.10, yh, "대상 지역")
    head(ax, 2.30, yh, "유사라벨 조건")
    head(ax, 6.05, yh, "학습과 채점")
    head(ax, 9.40, yh, "위약 대비 오차 변화")
    rows = [4.22, 3.47, 2.77, 2.07, 1.37]           # 조건 행(0 = 물리 유사라벨, 1–4 = 위약)
    # ---- 대상 지역: 라벨 블록 셀(유사라벨 자리), 채점 블록
    import cartopy.crs as ccrs
    d0 = 1.95
    Z = lena_zoom(fig, 0.10, 2.05, d0, d0, mode="design", scale_bar=True, buffer=False, dots=False)
    D = lena()
    L = D["L"]
    dfA = L["df"].iloc[L["A_idx"]]
    Z["ax"].scatter(dfA.lon.values, dfA.lat.values, s=1.3, color="#6B5A1E", linewidths=0, transform=ccrs.PlateCarree(),
                    zorder=3, rasterized=True)
    for yk, fc, lb in ((1.72, LBL_BLOCK, "라벨 블록(점 = 셀)"), (1.44, SCO_BLOCK, "채점 블록")):
        ax.add_patch(Rectangle((0.10, yk - 0.07), 0.14, 0.14, facecolor=fc, edgecolor="none", zorder=5))
        T(ax, 0.30, yk, lb, size=FS["small"], color=AUX)
    T(ax, 0.10, 1.12, "유사라벨 자리 = 라벨 블록 셀", size=FS["small"], color=AUX)
    T(ax, 0.10, 0.84, "레나델타 예 · 분할 1", size=FS["small"], color=AUX)
    # 지도 → 분배선 → 조건 행
    xbus1 = 2.18
    hline(ax, 0.10 + d0, xbus1, 3.02, color=ARROW, lw=1.1)
    vline(ax, xbus1, rows[-1], rows[0], color=ARROW, lw=1.1)
    for y in rows:
        arrow(ax, (xbus1, y), (2.34, y))
    # ---- 조건 5행: 색 표지, 이름, 식
    conds = [
        ("D1", "물리 유사라벨", ["$\\tilde{y}_i = E_n\\sqrt{\\mathrm{TDD}_i}$"]),
        ("placebo", "순서를 섞은 유사라벨", ["$\\tilde{y}_i = E_n\\sqrt{\\mathrm{TDD}_{\\pi(i)}}$"]),
        ("placebo", "풀 평균 상수", ["$\\tilde{y}_i = \\mathrm{mean}_A\\,(E_n\\sqrt{\\mathrm{TDD}})$"]),
        ("placebo", "원천 평균 상수", ["$\\tilde{y}_i = \\mathrm{mean}_S\\,(y)$"]),
        ("placebo", "선형 융해 지수", ["$\\tilde{y}_i = \\alpha + \\beta\\,\\mathrm{TDD}_i$", "  원천 적합"]),
    ]
    xk, xn = 2.42, 2.80
    for y, (code, nm, f_) in zip(rows, conds):
        key_line(ax, xk, y + 0.13, COL[code], w=0.28, lw=3.0, ls="-" if code == "D1" else (0, (2.0, 1.2)))
        T(ax, xn, y + 0.13, nm, size=FS["body"], color=INK)
        RT(ax, xn, y - 0.18, f_, size=FS["small"] + 0.5, color=AUX)
    xbus2 = 5.70
    for y in rows:
        hline(ax, 5.30, xbus2, y, color=ARROW, lw=1.1)
    vline(ax, xbus2, rows[-1], rows[0], color=ARROW, lw=1.1)
    ymid = (rows[1] + rows[2]) / 2
    T(ax, 2.42, 0.82, "다섯 조건은 셀·행 수·난수가 같고 유사라벨 값만 다르다", size=FS["small"], color=AUX)
    # ---- 학습 집합 → CatBoost → 예측 → 채점
    bx0, bx1 = 6.05, 7.25
    arrow(ax, (xbus2, ymid), (bx0, ymid))
    block(ax, bx0, ymid - 0.24, bx1 - bx0, 0.48, "CatBoost", fill=YEL, size=FS["body"])
    T(ax, (bx0 + bx1) / 2, ymid + 1.30, "학습 집합", size=FS["small"], ha="center", color=AUX)
    T(ax, (bx0 + bx1) / 2, ymid + 1.02, "원천 실측", size=FS["body"], ha="center")
    RT(ax, (bx0 + bx1) / 2, ymid + 0.74, ["+ 대상 라벨 ", "$n$", "개"], size=FS["body"], ha="center")
    T(ax, (bx0 + bx1) / 2, ymid + 0.46, "+ 유사라벨", size=FS["body"], ha="center")
    arrow(ax, ((bx0 + bx1) / 2, ymid + 0.34), ((bx0 + bx1) / 2, ymid + 0.24))
    T(ax, (bx0 + bx1) / 2, ymid - 0.46, "조건마다 따로 학습", size=FS["small"], ha="center", color=AUX)
    T(ax, (bx0 + bx1) / 2, ymid - 0.74, "원천 행당 유사라벨 10개", size=FS["small"], ha="center", color=AUX)
    RT(ax, (bx0 + bx1) / 2, ymid - 1.02, ["$n$", " = 0, 10"], size=FS["small"], ha="center", color=AUX)
    cx0 = 7.50
    arrow(ax, (bx1, ymid), (cx0, ymid))
    chevron(ax, cx0, ymid - 0.36, 1.62, 0.72, GRN, "채점 블록\nRMSE", first=True, size=FS["body"])
    arrow(ax, (cx0 + 1.62, ymid), (9.40, ymid))
    # ---- 위약 대비 Δ(실자료 fig3_a.csv): 행은 위약 행과 같은 높이
    a = pd.read_csv(PF / "fig3_a.csv").set_index(["placebo", "n"])
    keys = ["shuffle", "const_t", "const_src", "tddlin"]
    fx0, fx1 = 10.28, 11.88
    vmin, vmax = -4.5, 1.0
    X_ = lambda v: fx0 + (v - vmin) / (vmax - vmin) * (fx1 - fx0)
    ytop, ybot = rows[1] + 0.40, rows[-1] - 0.40
    ax.add_patch(Rectangle((X_(-0.5), ybot), X_(0.5) - X_(-0.5), ytop - ybot, facecolor=BAND, edgecolor="none", zorder=1))
    vline(ax, X_(0), ybot, ytop, color=ZERO, lw=1.0, z=2)
    for y, kk in zip(rows[1:], keys):
        for n_, dy, filled in ((0, 0.13, True), (10, -0.13, False)):
            r = a.loc[(kk, n_)]
            yy = y + dy
            ax.plot([X_(r.ci_lo), X_(r.ci_hi)], [yy, yy], color=COL["D1"], lw=3.0, solid_capstyle="butt", zorder=4)
            ax.plot([X_(r.ci_lo_beq), X_(r.ci_hi_beq)], [yy - 0.065, yy - 0.065], color=COL["D1"], lw=1.3,
                    solid_capstyle="butt", zorder=4)
            ax.scatter([X_(r.delta)], [yy], s=40, facecolor=COL["D1"] if filled else "white", edgecolor=COL["D1"],
                       linewidths=1.4, zorder=5)
    # 축
    hline(ax, fx0, fx1, ybot, color=INK, lw=1.0)
    for v in (-4, -2, 0):
        vline(ax, X_(v), ybot - 0.06, ybot, color=INK, lw=1.0)
        T(ax, X_(v), ybot - 0.20, f"{v:d}".replace("-", "−"), size=FS["small"], ha="center")
    T(ax, (fx0 + fx1) / 2, ybot - 0.47, "오차 변화 (cm)", size=FS["small"], ha="center")
    # 행 이름(위약 행과 같은 높이)
    for y, nm in zip(rows[1:], ["순서 섞음", "풀 평균", "원천 평균", "선형 지수"]):
        T(ax, 9.46, y, nm, size=FS["small"], color=AUX)
    # 열쇠(맨 위 행 자리: 물리 유사라벨 행은 비교 기준)
    kx0 = 9.46
    yk = rows[0] + 0.13
    ax.scatter([kx0 + 0.08], [yk], s=40, facecolor=COL["D1"], edgecolor=COL["D1"], linewidths=1.4, zorder=5)
    T(ax, kx0 + 0.22, yk, "라벨 0개", size=FS["small"])
    ax.scatter([kx0 + 1.42], [yk], s=40, facecolor="white", edgecolor=COL["D1"], linewidths=1.4, zorder=5)
    T(ax, kx0 + 1.56, yk, "10개", size=FS["small"])
    ax.plot([kx0, kx0 + 0.24], [yk - 0.33, yk - 0.33], color=COL["D1"], lw=3.0, solid_capstyle="butt")
    T(ax, kx0 + 0.32, yk - 0.33, "셀 가중", size=FS["small"])
    ax.plot([kx0 + 1.34, kx0 + 1.58], [yk - 0.33, yk - 0.33], color=COL["D1"], lw=1.3, solid_capstyle="butt")
    T(ax, kx0 + 1.66, yk - 0.33, "블록 등가중", size=FS["small"])
    T(ax, 9.46, ybot - 0.76, "음수 = 물리 유사라벨 오차 작음", size=FS["small"], color=AUX)
    T(ax, 9.40, 4.64, "주 4지역 평균 · 95% CI", size=FS["small"], color=AUX)
    return qa_and_save(fig, "M5_augmentation_design")


# ================================================================ M7 평가 설계
def panel_head(ax, x, y, letter, title):
    panel_letter(ax, x, y, letter)
    head(ax, x + 0.24, y, title)


def m7_evaluation():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    yh = 4.98
    xa, xb, xc, xd = 0.05, 3.00, 6.05, 8.80
    panel_head(ax, xa, yh, "a", "지역 홀드아웃")
    panel_head(ax, xb, yh, "b", "블록 절반 분할")
    panel_head(ax, xc, yh, "c", "라벨 수 격자")
    panel_head(ax, xd, yh, "d", "판정 규칙")
    dm = 2.55
    ym = 2.05
    # ---- a: 대상 셀 + 100 km 완충
    lena_zoom(fig, xa + 0.02, ym, dm, dm, mode="holdout", scale_bar=True, buffer=True)
    ky = [1.72, 1.45, 1.18]
    ax.scatter([xa + 0.12], [ky[0]], s=12, color=INK, linewidths=0, zorder=5)
    T(ax, xa + 0.28, ky[0], "대상 지역 셀(원천에서 제외)", size=FS["small"])
    ax.plot([xa + 0.02, xa + 0.22], [ky[1], ky[1]], color=INK, lw=1.0, ls=(0, (3, 2)))
    T(ax, xa + 0.28, ky[1], "100 km 완충", size=FS["small"])
    T(ax, xa + 0.02, ky[2], "원천 = 완충 밖 모든 라벨 셀", size=FS["small"], color=AUX)
    # ---- b: 라벨 블록·채점 블록 + 라벨 40개
    lena_zoom(fig, xb + 0.02, ym, dm, dm, mode="design", scale_bar=False, buffer=False, dots=True)
    for yk, fc, lb in ((ky[0], LBL_BLOCK, "라벨 블록 · 점 = 라벨 40개"), (ky[1], SCO_BLOCK, "채점 블록")):
        ax.add_patch(Rectangle((xb + 0.04, yk - 0.07), 0.14, 0.14, facecolor=fc, edgecolor="none", zorder=5))
        T(ax, xb + 0.26, yk, lb, size=FS["small"])
    T(ax, xb + 0.02, ky[2], "0.5° 블록 무작위 절반 · 분할 5회", size=FS["small"], color=AUX)
    # ---- c: 라벨 수 격자(범주 축)
    grid = ["0", "3", "10", "40", "160", "320", "1000", "전량"]
    yg = np.linspace(2.06, 4.42, len(grid))
    xax = xc + 0.30
    vline(ax, xax, yg[0], yg[-1], color=INK, lw=1.0)
    for y, g_ in zip(yg, grid):
        hline(ax, xax - 0.07, xax, y, color=INK, lw=1.0)
        T(ax, xax - 0.13, y, g_, size=FS["body"], ha="right")
    # 괄호: 평균을 내는 지역
    def bracket(y0, y1, x, text, lines=None):
        vline(ax, x, y0, y1, color=AUX, lw=1.0)
        hline(ax, x - 0.06, x, y0, color=AUX, lw=1.0)
        hline(ax, x - 0.06, x, y1, color=AUX, lw=1.0)
        T(ax, x + 0.10, (y0 + y1) / 2, text, size=FS["small"], color=INK)
    bracket(yg[0] - 0.05, yg[2] + 0.05, xax + 0.42, "주 4지역 평균")
    bracket(yg[3] - 0.05, yg[4] + 0.05, xax + 0.42, "레나델타·캐나다")
    T(ax, xc + 0.02, ky[0], "분할 5회 × 추출 5회", size=FS["small"])
    T(ax, xc + 0.02, ky[1], "모든 방법이 같은 라벨", size=FS["small"])
    T(ax, xc + 0.02, ky[2], "라벨 블록 셀 수보다 작은 값만", size=FS["small"], color=AUX)
    # ---- d: 판정 규칙 도식(숫자 눈금 없음)
    rows = [("오차 감소", "두 CI 상한 < 0", (-1.65, -0.75), (-1.90, -0.45)),
            ("오차 증가", "두 CI 하한 > 0", (0.70, 1.55), (0.45, 1.80)),
            ("동등", "네 한계 모두 ±0.5 cm 안", (-0.28, 0.22), (-0.40, 0.36)),
            ("미결정", "그 밖", (-0.95, 0.35), (-1.30, 0.85))]
    yr = [4.30, 3.62, 2.94, 2.26]
    sx0, sx1 = xd + 1.60, xd + 3.05          # 도식 축 범위(인치)
    vmin, vmax = -2.1, 2.1
    X_ = lambda v: sx0 + (v - vmin) / (vmax - vmin) * (sx1 - sx0)
    ytop, ybot = yr[0] + 0.32, yr[-1] - 0.32
    ax.add_patch(Rectangle((X_(-0.5), ybot), X_(0.5) - X_(-0.5), ytop - ybot, facecolor=BAND, edgecolor="none", zorder=1))
    vline(ax, X_(0), ybot, ytop, color=ZERO, lw=1.0, z=2)
    T(ax, X_(0), ytop + 0.14, "0", size=FS["small"], ha="center", color=AUX)
    for y, (verdict, cond, cc, cb) in zip(yr, rows):
        T(ax, xd + 0.02, y + 0.12, verdict, size=FS["body"])
        T(ax, xd + 0.02, y - 0.16, cond, size=FS["small"], color=AUX)
        ax.plot([X_(cc[0]), X_(cc[1])], [y + 0.04, y + 0.04], color=INK, lw=3.0, solid_capstyle="butt", zorder=4)
        ax.plot([X_(cb[0]), X_(cb[1])], [y - 0.08, y - 0.08], color=INK, lw=1.3, solid_capstyle="butt", zorder=4)
    # 열쇠
    ax.plot([xd + 0.02, xd + 0.26], [ky[0], ky[0]], color=INK, lw=3.0, solid_capstyle="butt")
    T(ax, xd + 0.34, ky[0], "셀 가중 CI", size=FS["small"])
    ax.plot([xd + 1.45, xd + 1.69], [ky[0], ky[0]], color=INK, lw=1.3, solid_capstyle="butt")
    T(ax, xd + 1.77, ky[0], "블록 등가중 CI", size=FS["small"])
    ax.add_patch(Rectangle((xd + 0.02, ky[1] - 0.07), 0.24, 0.14, facecolor=BAND, edgecolor="none"))
    T(ax, xd + 0.34, ky[1], "±0.5 cm 동등 한계", size=FS["small"])
    T(ax, xd + 0.02, ky[2], "95% · 블록 재표집 1000 또는 10,000회", size=FS["small"], color=AUX)
    # ---- 사전 등록 시간선(가는 선, 점 4개: 이름 위, 설명 아래)
    yt = 0.48
    pts = [(0.40, "등록", "가설·판정 규칙 커밋"), (3.40, "실행", "같은 코드 · 고정 seed"),
           (6.40, "재현 관문", "계산 환경 간 대조"), (9.15, "열람", "봉인 표 첫 열람 기록")]
    arrow(ax, (pts[0][0], yt), (11.90, yt), color=AUX, lw=1.0, hl=0.07, hw=0.055)
    for x, nm, sub in pts:
        ax.scatter([x], [yt], s=30, facecolor="white", edgecolor=INK, linewidths=1.2, zorder=6)
        T(ax, x - 0.06, yt + 0.21, nm, size=FS["body"])
        T(ax, x - 0.06, yt - 0.22, sub, size=FS["small"], color=AUX)
    return qa_and_save(fig, "M7_evaluation")


# ================================================================ M1 문제 정의
def m1_problem():
    import cartopy.crs as ccrs
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    yh = 4.98
    head(ax, 0.10, yh, "원천 지역과 대상 지역")
    head(ax, 6.30, yh, "대상 라벨 수에 따른 방법 선택")
    vline(ax, 6.10, 0.25, 5.12, color=SEP, lw=0.8)
    # 범북극 지도: 원천 = 채운 원(지역 색), 대상 = 레나델타 빈 원
    d = 3.85
    axm, proj = polar_map(fig, 0.10, 0.62, d, d, size_k=3.0, hollow_regions=("Lena Delta",), hollow_color=INK)
    # 대상 확대도(지역 홀드아웃 + 100 km 완충)
    z = 1.62
    zx, zy = 4.25, 2.95
    lena_zoom(fig, zx, zy, z, z, mode="holdout", scale_bar=True, buffer=True, label_size=FS["small"])
    # 지도 위 레나델타 위치 → 확대도: 수평·수직 직선
    D = lena()
    dd = D["L"]["df"].iloc[D["L"]["t_idx"]]
    P = proj.transform_point(float(dd.lon.mean()), float(dd.lat.mean()), ccrs.PlateCarree())
    xy_disp = axm.transData.transform(P)
    xin, yin = ax.transData.inverted().transform(xy_disp)
    ov = overlay(fig)
    vline(ov, xin, yin + 0.16, zy + z / 2, color=INK, lw=1.0)
    arrow(ov, (xin, zy + z / 2), (zx - 0.02, zy + z / 2), color=INK, lw=1.0)
    T(ax, zx, zy - 0.20, "대상: 레나델타", size=FS["small"])
    T(ax, zx, zy - 0.47, "점 = 대상 셀", size=FS["small"], color=AUX)
    ax.plot([zx, zx + 0.22], [zy - 0.74, zy - 0.74], color=INK, lw=1.0, ls=(0, (3, 2)))
    T(ax, zx + 0.30, zy - 0.74, "100 km 완충", size=FS["small"], color=AUX)
    T(ax, 0.10, 0.40, "원천 = 대상과 완충 밖 라벨 셀 · 원천 셀의 78–94%가 알래스카", size=FS["small"], color=AUX)
    # 오른쪽: 라벨 수 축과 후보 방법 4계열
    cats = ["0", "3", "10", "40", "160", "전량"]
    xs_ = np.linspace(8.55, 11.75, len(cats))
    yax = 1.40
    hline(ax, xs_[0] - 0.15, xs_[-1] + 0.12, yax, color=INK, lw=1.0)
    for x_, c_ in zip(xs_, cats):
        vline(ax, x_, yax - 0.07, yax, color=INK, lw=1.0)
        T(ax, x_, yax - 0.25, c_, size=FS["body"], ha="center")
    RT(ax, (xs_[0] + xs_[-1]) / 2, yax - 0.55, ["대상 지역 라벨 수 ", "$n$"], size=FS["body"], ha="center")
    rows = [("P0", "원천 계수 Stefan", "대상 라벨을 쓰지 않음", 0),
            ("P1", "재보정 Stefan", "라벨로 계수 E 만 보정", 1),
            ("R1", "물리 잔차 결합", "재보정 앵커 + 잔차 학습", 0),
            ("D0", "직접 ML", "물리 결합 없이 학습", 0)]
    yr = [4.05, 3.42, 2.79, 2.16]
    for x_ in xs_:
        vline(ax, x_, yax + 0.12, yr[0] + 0.30, color=SEP, lw=0.8, z=1)
    for y, (code, nm, sub, i0) in zip(yr, rows):
        key_line(ax, 6.30, y + 0.12, COL[code], w=0.26, lw=3.0)
        T(ax, 6.66, y + 0.12, nm, size=FS["body"], color=COL["R1"] if code == "R1" else INK)
        T(ax, 6.66, y - 0.16, sub, size=FS["small"], color=AUX)
        ax.plot([xs_[i0], xs_[-1]], [y, y], color=COL[code], lw=2.2, ls="-" if code != "D0" else (0, (3.0, 1.6)),
                solid_capstyle="butt", zorder=3)
        ax.scatter(xs_[i0:], [y] * len(xs_[i0:]), s=34, facecolor="white", edgecolor=COL[code], linewidths=1.4, zorder=4)
    T(ax, 9.15, 4.62, "라벨 n개일 때 어떤 방법을 쓰는가", size=FS["head"], ha="center", color=INK)
    T(ax, 6.30, 0.40, "모든 방법은 같은 라벨 n개와 같은 채점 블록을 쓴다", size=FS["small"], color=AUX)
    return qa_and_save(fig, "M1_problem")


# ================================================================ M3 자료 요약
def m3_data():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    yh = 4.98
    panel_head(ax, 0.05, yh, "a", "라벨 위치")
    panel_head(ax, 3.78, yh, "b", "라벨 출처")
    panel_head(ax, 8.62, yh, "c", "입력 해상도")
    # a: 범북극 지도(지역 색) + 티베트 고원 삽도
    d = 3.45
    polar_map(fig, 0.12, 0.95, d, d, size_k=3.0)
    import cartopy.crs as ccrs
    b = blocks_table()
    tb = b[b.reg == "Tibetan Plateau"]
    axi = sub_axes(fig, 2.60, 0.30, 1.05, 0.70, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=90, central_latitude=34))
    axi.set_extent([77, 103, 27.5, 39.5], ccrs.PlateCarree())
    axi.add_feature(land_feature("50m"), facecolor=LAND, edgecolor="none").set_rasterized(True)
    axi.scatter(tb.lon.values, tb.lat.values, s=3.0 * np.sqrt(tb.n_loc_1km.values) * 4.0, facecolor=REG["Tibetan Plateau"],
                edgecolor="white", linewidths=0.5, transform=ccrs.PlateCarree(), zorder=3)
    axi.spines["geo"].set_edgecolor(HAIR); axi.spines["geo"].set_linewidth(0.7)
    T(ax, 2.55, 0.65, "티베트 고원", size=FS["small"], ha="right", color=AUX)
    T(ax, 0.10, 0.30, "원 면적 ∝ 블록의 1 km 위치 수", size=FS["small"], color=AUX)
    # b: 헤어라인 표(Table 1)
    t1 = table1().set_index("target")
    rows = [("Lena Delta", "Lena", "점 관측(ALLena)"), ("Canada", "Canada", "점 관측(ABoVE)"),
            ("W Russia", "Russia_W", "CALM 지점 평균"), ("E Russia", "Russia_E", "CALM 지점 평균"),
            ("Alaska", "Alaska", "점 관측(ABoVE)"), ("Central Russia", "Russia_C_LGD", "1 km 셀 평균"),
            ("Tibetan Plateau", "Tibet_LGD", "1 km 셀 평균")]
    cx = [3.80, 4.08, 5.22, 7.32, 7.86, 8.32]     # 점, 지역, 출처, 라벨(오른쪽), 1 km(오른쪽), 블록(오른쪽)
    ytop = 4.55
    rh = 0.40
    hline(ax, 3.78, 8.34, ytop + 0.05, color=INK, lw=1.2)
    T(ax, cx[1], ytop - 0.17, "지역", size=FS["small"], color=AUX)
    T(ax, cx[2], ytop - 0.17, "라벨 출처", size=FS["small"], color=AUX)
    T(ax, cx[3], ytop - 0.17, "라벨", size=FS["small"], color=AUX, ha="right")
    T(ax, cx[4], ytop - 0.17, "1 km", size=FS["small"], color=AUX, ha="right")
    T(ax, cx[5], ytop - 0.17, "블록", size=FS["small"], color=AUX, ha="right")
    hline(ax, 3.78, 8.34, ytop - 0.38, color=INK, lw=0.7)

    def fmt(v):
        v = int(round(float(v)))
        return f"{v:,}" if v >= 10000 else f"{v:d}"
    y = ytop - 0.38 - rh / 2 - 0.02
    groups = [("주 4지역", rows[:4]), ("참조", rows[4:5]), ("새 지역", rows[5:])]
    for gname, rr in groups:
        T(ax, cx[1], y, gname, size=FS["small"], color=AUX)
        y -= rh * 0.80
        for reg, key, src in rr:
            ax.scatter([cx[0] + 0.07], [y], s=60, color=REG[reg], linewidths=0, zorder=4)
            T(ax, cx[1], y, REG_KO[reg], size=FS["body"])
            T(ax, cx[2], y, src, size=FS["small"] + 0.5)
            T(ax, cx[3], y, fmt(t1.loc[key, "label_rows"]), size=FS["body"], ha="right")
            T(ax, cx[4], y, fmt(t1.loc[key, "loc_1km"]), size=FS["body"], ha="right")
            T(ax, cx[5], y, fmt(t1.loc[key, "blocks"]), size=FS["body"], ha="right")
            y -= rh
    hline(ax, 3.78, 8.34, y + rh / 2, color=INK, lw=1.2)
    T(ax, 3.80, y + rh / 2 - 0.24, "라벨 = 행 수 · 블록 = 0.5° · 라벨 연도 1990–2024", size=FS["small"], color=AUX)
    # c: 입력 해상도 점 그림(로그 m 축)
    items = [("라벨", 11, 1000, INK), ("지형 DEM", 30, 30, INK), ("지형 창", 30, 990, INK), ("기후 ERA5-Land", 11100, 11100, INK),
             ("토양 SoilGrids", 250, 5000, INK), ("ESA CCI ALT", 1000, 1000, INK), ("MODIS(추가 시험)", 250, 500, AUX)]
    lx0, lx1 = 10.22, 11.88
    lo, hi = np.log10(8), np.log10(20000)
    X_ = lambda v: lx0 + (np.log10(v) - lo) / (hi - lo) * (lx1 - lx0)
    yi = np.linspace(4.30, 1.70, len(items))
    ax.add_patch(Rectangle((X_(5000), yi[-1] - 0.25), X_(11100) - X_(5000), yi[0] - yi[-1] + 0.50, facecolor="#E3EEF7",
                           edgecolor="none", zorder=1))
    vline(ax, X_(1000), yi[-1] - 0.25, yi[0] + 0.25, color=ZERO, lw=1.2, z=2)
    for y_, (nm, a_, b_, c_) in zip(yi, items):
        T(ax, lx0 - 0.08, y_, nm, size=FS["small"], ha="right", color=c_)
        if nm.startswith("MODIS"):
            ax.plot([X_(a_), X_(b_)], [y_, y_], color=AUX, lw=4.0, solid_capstyle="butt", zorder=3, alpha=0.6)
            continue
        if a_ != b_:
            ax.plot([X_(a_), X_(b_)], [y_, y_], color=INK, lw=1.2, zorder=3)
            ax.scatter([X_(a_)], [y_], s=40, facecolor="white", edgecolor=INK, linewidths=1.2, zorder=4)
        ax.scatter([X_(b_)], [y_], s=40, facecolor=INK, edgecolor=INK, linewidths=1.2, zorder=5)
    yb = yi[-1] - 0.30
    hline(ax, lx0, lx1, yb, color=INK, lw=1.0)
    for v, lb in ((10, "10"), (100, "100"), (1000, "1000"), (10000, "10,000")):
        vline(ax, X_(v), yb - 0.06, yb, color=INK, lw=1.0)
        T(ax, X_(v), yb - 0.21, lb, size=FS["small"], ha="center")
    T(ax, (lx0 + lx1) / 2, yb - 0.48, "공간 규모 (m)", size=FS["small"], ha="center")
    T(ax, X_(1000) + 0.05, yi[0] + 0.40, "1 km 모형 셀", size=FS["small"], ha="center")
    ax.scatter([8.72], [yb - 0.80], s=40, facecolor="white", edgecolor=INK, linewidths=1.2)
    T(ax, 8.84, yb - 0.80, "원자료", size=FS["small"], color=AUX)
    ax.scatter([9.62], [yb - 0.80], s=40, facecolor=INK, edgecolor=INK, linewidths=1.2)
    T(ax, 9.74, yb - 0.80, "모형 입력", size=FS["small"], color=AUX)
    ax.add_patch(Rectangle((10.62, yb - 0.87), 0.20, 0.14, facecolor="#E3EEF7", edgecolor="none"))
    T(ax, 10.88, yb - 0.80, "정보 해상도", size=FS["small"], color=AUX)
    T(ax, 8.72, yb - 1.08, "정보 해상도: 토양 약 5 km, 기후 0.1°", size=FS["small"], color=AUX)
    return qa_and_save(fig, "M3_data")


# ================================================================ M2 선행 연구 한계 → 개선 → 기대 효과(IMAGE 3쪽 배치)
def m2_gap():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    yh = 4.98
    c0, c1, c2 = 0.10, 4.05, 8.45
    head(ax, c0, yh, "선행 연구의 한계")
    head(ax, c1, yh, "본 연구: 라벨 수별 전이 평가")
    head(ax, c2, yh, "기대 효과")
    for x in (3.85, 8.25):
        vline(ax, x, 0.25, 5.12, color=SEP, lw=0.9, ls=(0, (3, 3)))
    # 왼쪽 위: 검증 설계에 따른 RMSE(실자료, Fig 1e 세 지역 평균)
    e = pd.read_csv(ROOT / "outputs/figures/paper/v3_restructure/Fig1_source_data.csv")
    e = e[(e.panel == "e") & (e.region == "MEAN3[Alaska,Lena,Canada]") & (e.element == "rmse_mean_equal_weight")]
    order = ["Random", "Site", "Block", "kNNDM", "Region holdout"]
    dml = [float(e[e.kind == f"D0w|{k}"].value.iloc[0]) for k in order]
    stf = [float(e[e.kind == f"PSw|{k}"].value.iloc[0]) for k in order]
    axl = sub_axes(fig, 0.66, 3.32, 2.80, 1.28)
    xx = np.arange(5)
    axl.plot(xx, dml, color=COL["D0"], lw=2.2, ls=(0, (3.0, 1.6)), marker="o", ms=5, mfc=COL["D0"], mec=COL["D0"])
    axl.plot(xx, stf, color=COL["P1"], lw=2.2, marker="o", ms=5, mfc=COL["P1"], mec=COL["P1"])
    axl.set_xlim(-0.3, 4.3); axl.set_ylim(12, 31)
    axl.set_yticks([15, 20, 25, 30]); axl.set_xticks([0, 4]); axl.set_xticklabels(["무작위 셀", "지역 홀드아웃"])
    for k in ("top", "right"):
        axl.spines[k].set_visible(False)
    axl.tick_params(labelsize=FS["small"], length=3, width=1.0, pad=2)
    axl.set_ylabel("RMSE (cm)", fontsize=FS["small"])
    axl.text(4.25, dml[-1] + 0.4, "직접 ML", color=INK, fontsize=FS["small"], ha="right", va="bottom")
    axl.text(4.25, stf[-1] - 0.9, "Stefan", color=INK, fontsize=FS["small"], ha="right", va="top")
    T(ax, c0, 2.78, "검증 설계별 오차 · 알래스카·레나델타·캐나다 평균", size=FS["small"], color=AUX)
    # 가운데 위: 작은 흐름(원천 라벨 → Stefan · 물리 기반 ML → 대상 채점 블록)
    d_ = 1.30
    polar_map(fig, c1 + 0.02, 3.20, d_, d_, size_k=1.8)
    T(ax, c1 + 0.02 + d_ / 2, 2.98, "원천 라벨", size=FS["small"], ha="center", color=AUX)
    chx, chw = 5.66, 1.12
    chevron(ax, chx, 4.00, chw, 0.44, GRN, "Stefan", first=True, size=FS["small"])
    chevron(ax, chx, 3.30, chw, 0.44, YEL, "물리 기반 ML", first=True, size=FS["small"])
    xbus = 5.50
    yc_ = 3.85
    hline(ax, c1 + 0.02 + d_, xbus, yc_, color=ARROW, lw=1.1)
    vline(ax, xbus, 3.52, 4.22, color=ARROW, lw=1.1)
    arrow(ax, (xbus, 4.22), (chx - 0.02, 4.22)); arrow(ax, (xbus, 3.52), (chx - 0.02, 3.52))
    xm2 = chx + chw + 0.14
    hline(ax, chx + chw, xm2, 4.22, color=ARROW, lw=1.1); hline(ax, chx + chw, xm2, 3.52, color=ARROW, lw=1.1)
    vline(ax, xm2, 3.52, 4.22, color=ARROW, lw=1.1)
    dz = 1.12
    arrow(ax, (xm2, yc_), (7.05, yc_))
    lena_zoom(fig, 7.07, yc_ - dz / 2, dz, dz, mode="design", scale_bar=False, buffer=False, dots=True)
    T(ax, 7.07 + dz / 2, 2.98, "대상 채점 블록", size=FS["small"], ha="center", color=AUX)
    RT(ax, (chx + chw / 2), 2.98, ["같은 라벨 ", "$n$", "개"], size=FS["small"], ha="center", color=AUX)
    # 오른쪽 위: 알래스카 1 km 지도 + 라벨 수별 방법(도식)
    arr = alaska_png("c_single")
    w6, h6 = fit_box(arr.shape, 1.40, 1.30)
    img_thumb(fig, c2 + 0.02, 3.20, w6, h6, arr)
    T(ax, c2 + 0.02 + w6 / 2, 2.98, "1 km ALT 지도", size=FS["small"], ha="center", color=AUX)
    sa = c2 + w6 + 0.25
    staircase(ax, sa, 11.92, 3.36, labels=("원천 계수", "재보정 + 잔차", "교차검증 선정"), dy=0.44)
    T(ax, (sa + 11.92) / 2, 2.98, "라벨 수별 방법", size=FS["small"], ha="center", color=AUX)
    # 세 행(헤어라인 구분)
    rows = [
        ("평가 지역", "학습 지역 안 평가", "무작위 교차검증", "지역 홀드아웃", "100 km 완충 · 블록 절반 분할", "새 지역 오차 추정",
         "학습에 없는 지역 기준"),
        ("라벨 수", "라벨 한 조건", "0개 또는 전량", "라벨 0개–전량 곡선", "모든 방법이 같은 라벨", "라벨 투자 판단 근거",
         "몇 개부터 ML 을 더할지"),
        ("물리 기준선", "기준선 없음 또는 고정 계수", "같은 라벨 보정 없음", "원천·재보정 Stefan 기준선", "같은 라벨로 계수 재보정",
         "ML 순이득 분리", "재보정 몫과 ML 몫 구분"),
    ]
    yr = [2.18, 1.30, 0.42]
    for i, (topic, l1, l2, m1, m2, r1, r2) in enumerate(rows):
        y = yr[i]
        hline(ax, 0.10, 11.90, y + 0.40, color=HAIR, lw=0.8)
        T(ax, c0, y + 0.18, topic, size=FS["small"], color=AUX)
        T(ax, c0 + 1.10, y + 0.18, l1, size=FS["body"])
        T(ax, c0 + 1.10, y - 0.12, l2, size=FS["small"], color=AUX)
        T(ax, c1, y + 0.18, m1, size=FS["body"], color=COL["R1"])
        T(ax, c1, y - 0.12, m2, size=FS["small"], color=AUX)
        T(ax, c2, y + 0.18, r1, size=FS["body"])
        T(ax, c2, y - 0.12, r2, size=FS["small"], color=AUX)
        arrow(ax, (3.55, y + 0.18), (3.95, y + 0.18), color=AUX, lw=1.0, hl=0.07, hw=0.055)
        arrow(ax, (7.95, y + 0.18), (8.35, y + 0.18), color=AUX, lw=1.0, hl=0.07, hw=0.055)
    return qa_and_save(fig, "M2_gap")


# ================================================================ M8 라벨 수별 워크플로 알고리즘
def m8_label_workflow():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    yh = 4.98
    head(ax, 0.10, yh, "라벨 수에 따른 단계")
    head(ax, 9.65, yh, "출력")
    vline(ax, 9.45, 0.25, 5.12, color=SEP, lw=0.8)
    # 라벨 수 축(구간): 0 | 3–10 | 40 이상
    x0, x1 = 1.55, 9.20
    bounds = [x0, 3.30, 6.05, x1]
    cats = [("0", (bounds[0] + bounds[1]) / 2), ("3–10", (bounds[1] + bounds[2]) / 2), ("40 이상", (bounds[2] + bounds[3]) / 2)]
    yax = 4.40
    hline(ax, x0, x1, yax, color=INK, lw=1.0)
    for xb_ in bounds:
        vline(ax, xb_, yax - 0.07, yax + 0.07, color=INK, lw=1.0)
    for lb, xc_ in cats:
        T(ax, xc_, yax + 0.22, lb, size=FS["body"], ha="center")
    T(ax, 0.10, yax + 0.22, "라벨 수", size=FS["small"], color=AUX)
    for xb_ in bounds[1:-1]:
        vline(ax, xb_, 1.90, yax - 0.07, color=SEP, lw=0.8, ls=(0, (3, 3)))
    lanes = [("방법", 3.90), ("진단", 3.00), ("라벨 배치", 2.25)]
    for nm, y in lanes:
        T(ax, 0.10, y, nm, size=FS["small"], color=AUX)
    # 방법 행
    yM = lanes[0][1]
    ax.scatter([(bounds[0] + bounds[1]) / 2], [yM - 0.16], s=60, color=COL["P0"], zorder=5, linewidths=0)
    T(ax, (bounds[0] + bounds[1]) / 2, yM + 0.16, "원천 계수 Stefan", size=FS["small"] + 0.5, ha="center")
    ax.plot([bounds[1] + 0.10, bounds[2] - 0.10], [yM - 0.16, yM - 0.16], color=COL["R1"], lw=3.2, solid_capstyle="butt")
    T(ax, (bounds[1] + bounds[2]) / 2, yM + 0.16, "재보정 + 저가중 잔차", size=FS["small"] + 0.5, ha="center")
    RT(ax, (bounds[1] + bounds[2]) / 2, yM - 0.42, ["$\\kappa = 10$", " · ", "$\\lambda = 0.25$"], size=FS["small"], ha="center",
       color=AUX)
    ax.plot([bounds[2] + 0.10, bounds[3] - 0.10], [yM - 0.16, yM - 0.16], color=COL["W"], lw=3.2, solid_capstyle="butt")
    T(ax, (bounds[2] + bounds[3]) / 2, yM + 0.16, "대상 라벨 교차검증 선정(후보 7개)", size=FS["small"] + 0.5, ha="center", color=COL["W"])
    # 진단 행
    yD = lanes[1][1]
    T(ax, (bounds[0] + bounds[1]) / 2, yD + 0.13, "학습 범위 밖 표시", size=FS["small"] + 0.5, ha="center")
    T(ax, (bounds[0] + bounds[1]) / 2, yD - 0.17, "공변량 외삽 셀", size=FS["small"], ha="center", color=AUX)
    T(ax, (bounds[1] + bounds[2]) / 2, yD + 0.13, "라벨 10개 편향 진단", size=FS["small"] + 0.5, ha="center")
    T(ax, (bounds[1] + bounds[2]) / 2, yD - 0.17, "원천 계수 Stefan 의 평균 절대 오차", size=FS["small"], ha="center", color=AUX)
    T(ax, (bounds[2] + bounds[3]) / 2, yD + 0.13, "교차검증 RMSE", size=FS["small"] + 0.5, ha="center")
    T(ax, (bounds[2] + bounds[3]) / 2, yD - 0.17, "고정 레시피 대비 비열등(40·160개)", size=FS["small"], ha="center", color=AUX)
    # 라벨 배치 행(전 구간)
    yP = lanes[2][1]
    ax.plot([bounds[0] + 0.10, bounds[3] - 0.10], [yP - 0.16, yP - 0.16], color=INK, lw=1.2, solid_capstyle="butt")
    T(ax, (x0 + x1) / 2, yP + 0.13, "블록·공변량 분산 배치 · 효과는 지역 의존(캐나다 이득, 레나델타 손해)", size=FS["small"] + 0.5,
      ha="center")
    T(ax, (x0 + x1) / 2, yP - 0.40, "사전 지정 배치 절차는 후보가 모두 제외되어 무작위 배치로 확정", size=FS["small"], ha="center",
      color=AUX)
    # 선정 규칙 5단계(methods.tex)
    head(ax, 0.10, 1.55, "교차검증 선정 규칙")
    steps = ["1  라벨 10개 미만이면 재보정 Stefan",
             "2  라벨 블록을 최대 5겹으로 나눔(2겹 미만이면 재보정 Stefan)",
             "3  겹마다 계수 재계산, 후보 7개 적합",
             "4  교차검증 RMSE 최소 후보 선택",
             "5  전체 라벨로 다시 적합"]
    for k, st in enumerate(steps):
        T(ax, 0.10 + (k // 3) * 4.75, 1.18 - (k % 3) * 0.31, st, size=FS["small"] + 0.5)
    T(ax, 4.85, 0.56, "후보: 원천·재보정·현지 최소제곱 Stefan, 물리 잔차 결합 λ 0.25·1.0,", size=FS["small"], color=AUX)
    T(ax, 4.85, 0.27, "증강 + 앵커 + 잔차, 물리 유사라벨", size=FS["small"], color=AUX)
    # 출력: 알래스카 1 km 지도와 학습 범위
    arr = alaska_png("c")
    w_, h_ = fit_box(arr.shape, 2.25, 1.45)
    img_thumb(fig, 9.65, 3.12, w_, h_, arr)
    T(ax, 9.65, 2.90, "1 km ALT 지도", size=FS["body"])
    T(ax, 9.65, 2.62, "90% 구간 ±20.2 cm", size=FS["small"], color=AUX)
    arr2 = alaska_png("d")
    w2, h2 = fit_box(arr2.shape, w_, 1.45)
    img_thumb(fig, 9.65, 0.80, w2, h2, arr2)
    T(ax, 9.65, 0.58, "학습 범위 밖 표시", size=FS["body"])
    T(ax, 9.65, 0.30, "공변량 외삽 셀 73%", size=FS["small"], color=AUX)
    return qa_and_save(fig, "M8_label_workflow")


# ================================================================ M9 학습된 배치 정책(DeepSets)
def m9_deepsets():
    W, H = 12.0, 4.6
    fig, ax = canvas(W, H)
    yh = 4.38
    head(ax, 0.10, yh, "라벨 집합")
    head(ax, 2.35, yh, "집합 효용 신경망")
    head(ax, 9.70, yh, "배치 정책")
    vline(ax, 9.50, 0.20, 4.50, color=SEP, lw=0.8)
    ax.plot([6.80, 7.12], [yh, yh], color=TRAIN_RED, lw=1.4, ls=(0, (3, 2)))
    T(ax, 7.20, yh, "학습 때만 쓰는 경로", size=FS["small"], color=AUX)
    ym = 2.75
    # 라벨 집합: 알래스카 라벨 셀(실자료, 학습 범위 패널의 점)
    arr = alaska_png("d")
    w_, h_ = fit_box(arr.shape, 1.90, 1.80)
    img_thumb(fig, 0.12, ym - h_ / 2, w_, h_, arr)
    T(ax, 0.12, ym - h_ / 2 - 0.22, "원소 = 라벨 셀", size=FS["small"], color=AUX)
    RT(ax, 0.12, ym - h_ / 2 - 0.50, ["집합 ", "$L$", " · 크기 ", "$n$"], size=FS["small"], color=AUX)
    # 원소 특징
    x1 = 2.35
    arrow(ax, (0.12 + w_, ym), (x1 - 0.02, ym))
    # 원소 특징 막대(29칸 중 묶음 표시)
    fx, fw = x1 + 0.05, 0.26
    groups = [("공변량 25", 25, "#B4C3D3"), ("√TDD", 1, COL["P1"]), ("블록 비율", 1, "#D9E0E8"), ("외삽 비유사도", 1, "#8EA6BF"),
              ("최근접 라벨 거리", 1, "#6E8AA8")]
    top, bot = ym + 1.15, ym - 1.15
    unit = (top - bot) / 29.0
    yc = top
    for nm, k, c in groups:
        ax.add_patch(Rectangle((fx, yc - k * unit), fw, k * unit, facecolor=c, edgecolor="white", lw=0.8, zorder=3))
        yc -= k * unit
    T(ax, fx + fw / 2, bot - 0.20, "원소 특징 29", size=FS["small"], ha="center")
    T(ax, fx + fw + 0.10, ym + 0.42, "공변량 25", size=FS["small"], color=AUX)
    T(ax, fx + fw + 0.10, bot + 2.0 * unit, "그 밖 4", size=FS["small"], color=AUX)
    # φ: 원소마다 같은 MLP(겹친 블록)
    px = 4.05
    for k_ in (2, 1):
        block(ax, px + 0.06 * k_, ym - 0.62 + 0.06 * k_, 1.00, 1.24, None, fill=YEL, z=2)
    block(ax, px, ym - 0.62, 1.00, 1.24, None, fill=YEL, z=3)
    T(ax, px + 0.50, ym + 0.32, "φ", size=FS["body"] + 3, ha="center", z=5)
    T(ax, px + 0.50, ym - 0.05, "MLP", size=FS["small"], ha="center", z=5)
    T(ax, px + 0.50, ym - 0.36, "128 · 128", size=FS["small"], ha="center", z=5)
    T(ax, px + 0.55, ym - 0.86, "원소마다 같은 가중치", size=FS["small"], ha="center", color=AUX)
    arrow(ax, (fx + fw, ym), (px - 0.02, ym))
    # 합·평균 풀링
    qx = 5.55
    arrow(ax, (px + 1.12, ym), (qx - 0.02, ym))
    block(ax, qx, ym - 0.42, 0.80, 0.84, "합·평균\n풀링", fill=YEL, size=FS["small"])
    T(ax, qx + 0.40, ym - 0.62, "256", size=FS["small"], ha="center", color=AUX)
    # 문맥 6(아래에서 결합)
    cx_ = 6.75
    from matplotlib.patches import Circle
    arrow(ax, (qx + 0.80, ym), (cx_ - 0.11, ym))
    ax.add_patch(Circle((cx_, ym), 0.10, facecolor="white", edgecolor=INK, lw=1.0, zorder=6))
    T(ax, cx_, ym, "‖", size=FS["small"], ha="center", z=7)
    arrow(ax, (cx_, ym - 0.92), (cx_, ym - 0.11))
    T(ax, cx_, ym - 1.08, "과제 문맥 6", size=FS["small"], ha="center")
    T(ax, cx_, ym - 1.36, "후보·블록 수, 공변량 차이, n", size=FS["small"], ha="center", color=AUX)
    # ρ
    rx = 7.10
    arrow(ax, (cx_ + 0.11, ym), (rx - 0.02, ym))
    block(ax, rx, ym - 0.42, 0.95, 0.84, None, fill=YEL)
    T(ax, rx + 0.475, ym + 0.17, "ρ", size=FS["body"] + 3, ha="center", z=5)
    T(ax, rx + 0.475, ym - 0.18, "128 → 1", size=FS["small"], ha="center", z=5)
    arrow(ax, (rx + 0.95, ym), (8.35, ym))
    RT(ax, 8.40, ym, ["$\\hat{U}(L)$"], size=FS["body"] + 1)
    T(ax, 8.40, ym - 0.30, "예측 효용", size=FS["small"], color=AUX)
    # 학습: RankNet 순위 손실(붉은 파선)
    yl = ym + 1.15
    T(ax, 7.58, yl + 0.20, "RankNet 순위 손실", size=FS["body"], ha="center")
    RT(ax, 7.58, yl - 0.10, ["실제 효용 ", "$U$", ": 무작위 배치 대비 RMSE 변화"], size=FS["small"], ha="center", color=AUX)
    arrow(ax, (rx + 0.475, yl - 0.26), (rx + 0.475, ym + 0.44), color=TRAIN_RED, lw=1.3, ls=(0, (2.5, 1.6)), hl=0.07, hw=0.06)
    T(ax, 2.35, 0.62, "원소 특징 = 표준화 공변량 25, √TDD, 블록 셀 비율, 외삽 비유사도, 집합 안 최근접 라벨 거리", size=FS["small"],
      color=AUX)
    T(ax, 2.35, 0.32, "AdamW · 학습률 1e-3 · seed 3개 평균 · 알래스카 하위 지역 하나 제외 교차검증으로 epoch 선택", size=FS["small"],
      color=AUX)
    # 배치 정책(탐욕): 초록 셰브런
    chevron(ax, 9.70, ym + 0.55, 2.15, 0.62, GRN, "탐욕 선택", first=True, size=FS["body"])
    T(ax, 9.70, ym + 0.20, "5개씩 추가", size=FS["small"])
    T(ax, 9.70, ym - 0.08, "후보: farthest-first 상위 300", size=FS["small"])
    T(ax, 9.70, ym - 0.36, "∪ 무작위 200", size=FS["small"])
    RT(ax, 9.70, ym - 0.64, ["$\\hat{U}$", " 가 가장 작은 셀"], size=FS["small"])
    T(ax, 9.70, ym - 1.08, "알래스카 밖 시험 5과제", size=FS["small"], color=AUX)
    T(ax, 9.70, ym - 1.36, "(서술, 검정 없음)", size=FS["small"], color=AUX)
    return qa_and_save(fig, "M9_deepsets")


# ================================================================ 실행
FIGS = {"M4": m4_workflow, "M6": m6_models, "M5": m5_augmentation, "M7": m7_evaluation, "M1": m1_problem, "M3": m3_data, "M2": m2_gap, "M8": m8_label_workflow, "M9": m9_deepsets}


def contact_sheet():
    from PIL import Image
    files = sorted(OUT.glob("M*_slide.png"))
    if not files:
        return
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        r = 1100 / im.size[0]
        thumbs.append((f.stem, im.resize((1100, int(im.size[1] * r)), Image.LANCZOS)))
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    hmax = max(t.size[1] for _, t in thumbs)
    sheet = Image.new("RGB", (cols * 1140 + 40, rows * (hmax + 60) + 40), "white")
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sheet)
    for k, (nm, t) in enumerate(thumbs):
        cx, cy = 40 + (k % cols) * 1140, 40 + (k // cols) * (hmax + 60)
        sheet.paste(t, (cx, cy + 30))
        dr.rectangle([cx - 1, cy + 29, cx + t.size[0], cy + 30 + t.size[1]], outline=(200, 200, 200))
        dr.text((cx, cy + 8), nm, fill=(0, 0, 0))
    sheet.save(OUT / "contact_sheet.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    o = ap.parse_args()
    setup_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = [k.strip() for k in o.only.split(",") if k.strip()] or list(FIGS)
    t0 = time.time()
    for k in keys:
        FIGS[k]()
    qa_path = OUT / "method_figs_qa.json"
    old = json.loads(qa_path.read_text()) if qa_path.exists() else {}
    old.update({k: v for k, v in LOG.items() if not k.startswith("_")})
    qa_path.write_text(json.dumps(old, ensure_ascii=False, indent=1))
    contact_sheet()
    print(f"완료 {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
