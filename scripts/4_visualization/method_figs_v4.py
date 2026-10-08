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
_MB = _C.get("map_base", {}).get("slide", {})
LAND = _MB.get("land", _C["neutral"]["land"])          # 슬라이드 육지(토큰 map_base.slide)
COAST = _MB.get("coast", _C["neutral"]["coast"])
PF_CONT = _MB.get("pf_continuous", "#AABFD6")
PF_DISC = _MB.get("pf_discontinuous", "#CDD9E7")
GRATICULE = _MB.get("graticule", "#CCCCCC")
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


def pfr():
    if "pfr" not in _CACHE:
        import fig1
        _CACHE["pfr"] = fig1.pfr_classes()
    return _CACHE["pfr"]


def draw_pfr_slide(axm, proj, w_in, h_in):
    """영구동토 2단계(연속 ≥ 90 %, 불연속 50–90 %)를 map_base.slide 색으로(fig1.draw_pfr 와 같은 표본화)."""
    import cartopy.crs as ccrs
    from matplotlib.colors import ListedColormap
    lat, lon, M, _ = pfr()
    nx, ny = int(w_in * 300), int(h_in * 300)
    x0, x1 = axm.get_xlim(); y0, y1 = axm.get_ylim()
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y1, y0, ny))
    ll = ccrs.PlateCarree().transform_points(proj, X, Y)
    LON, LAT = ll[..., 0], ll[..., 1]
    i = np.clip(np.round((LAT - lat[0]) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
    j = np.clip(np.round((LON - lon[0]) / (lon[1] - lon[0])).astype(int), 0, len(lon) - 1)
    v = M[i, j]
    v[(LAT < lat.min() - 0.05) | (LAT > lat.max() + 0.05) | ~np.isfinite(LAT)] = np.nan
    cls = np.full(v.shape, np.nan)
    cls[(v >= 50) & (v < 90)] = 0
    cls[v >= 90] = 1
    axm.imshow(np.ma.masked_invalid(cls), extent=(x0, x1, y0, y1), origin="upper", cmap=ListedColormap([PF_DISC, PF_CONT]),
               vmin=0, vmax=1, interpolation="nearest", transform=proj, zorder=0.3, rasterized=True)


def pfr_key(ax, x, y, size=None):
    """영구동토 구역 열쇠(색만으로 전하지 않도록 이름을 적는다)."""
    size = size or FS["small"]
    ax.add_patch(Rectangle((x, y - 0.07), 0.16, 0.14, facecolor=PF_CONT, edgecolor="none", zorder=5))
    T(ax, x + 0.22, y, "연속 영구동토", size=size, color=AUX)
    ax.add_patch(Rectangle((x + 1.40, y - 0.07), 0.16, 0.14, facecolor=PF_DISC, edgecolor="none", zorder=5))
    T(ax, x + 1.62, y, "불연속", size=size, color=AUX)


def polar_map(fig, x, y, w, h, color_by_region=True, size_k=3.2, highlight=None, show_frame=True, lat_min=50.0,
              mono=None, hollow_regions=(), alpha=0.9, edge_lw=0.5, hollow_color=None, permafrost=False, coast=True):
    """범북극 라벨 지도(Fig 1a 와 같은 블록 자료). 원 면적 ∝ 블록의 1 km 라벨 위치 수. 원형 경계 지도."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.path as mpath
    proj = ccrs.NorthPolarStereo(central_longitude=127.0, true_scale_latitude=70.0)
    axm = sub_axes(fig, x, y, w, h, projection=proj)
    axm.set_extent([-180, 180, lat_min, 90], ccrs.PlateCarree())
    th = np.linspace(0, 2 * np.pi, 200)
    circ = mpath.Path(np.c_[np.sin(th), np.cos(th)] * 0.5 + 0.5)
    axm.set_boundary(circ, transform=axm.transAxes)
    axm.add_feature(land_feature("50m"), facecolor=LAND, edgecolor="none", zorder=0).set_rasterized(True)
    if permafrost:
        draw_pfr_slide(axm, proj, w, h)
    if coast:
        axm.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=COAST, linewidth=0.5, zorder=0.5).set_rasterized(True)
    axm.spines["geo"].set_edgecolor(HAIR if show_frame else "none"); axm.spines["geo"].set_linewidth(0.7)
    b = blocks_table()
    b = b[b.lat >= lat_min]
    order = ["Alaska", "Canada", "Lena Delta", "W Russia", "E Russia", "Central Russia"]
    for r in order:
        d = b[b.reg == r]
        if not len(d):
            continue
        c = mono or (REG[r] if color_by_region else INK)
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
    axm.add_feature(land_feature("10m"), facecolor=LAND, edgecolor=COAST, linewidth=0.5, zorder=0).set_rasterized(True)
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


def stefan_scatter(fig, x, y, w, h, numbers=False, show_recal=True, size=None, labelpad=2):
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
    axs.set_xlim(np.nanmin(s) * 0.85, xm); axs.set_ylim(0, max(np.nanmax(yv), C["E1"] * xm) * 1.15)
    axs.invert_yaxis()
    for k in ("top", "right"):
        axs.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        axs.spines[k].set_linewidth(1.0); axs.spines[k].set_color(INK)
    if not numbers:
        axs.set_xticks([]); axs.set_yticks([])
    axs.set_xlabel("√TDD", fontsize=size or FS["small"], labelpad=labelpad)
    axs.set_ylabel("ALT", fontsize=size or FS["small"], labelpad=2)
    axs.patch.set_alpha(0)
    return axs


def overlay(fig):
    """지도 축 위에 그리는 투명 덧층(인치 좌표). 지도 축을 다 넣은 뒤 부른다."""
    W, H = fig._W, fig._H
    ov = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ov.set_xlim(0, W); ov.set_ylim(0, H); ov.set_aspect("equal"); ov.axis("off")
    return ov


def staircase(ax, xa, xb, y0, labels=("원천 계수 Stefan", "재보정 + 저가중 잔차", "교차검증 방법 선택"), dy=0.50, size=None):
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
def _seg_hits_rect(p0, p1, bb, pad=1.0):
    """선분 (p0, p1)(표시 좌표)이 상자 bb(안쪽으로 pad 만큼 줄임)를 지나는지(Liang–Barsky)."""
    x0, y0, x1, y1 = bb.x0 + pad, bb.y0 + pad, bb.x1 - pad, bb.y1 - pad
    if x1 <= x0 or y1 <= y0:
        return False
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for pq in ((-dx, p0[0] - x0), (dx, x1 - p0[0]), (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        pp, q = pq
        if abs(pp) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / pp
        if pp < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _overlap_px(a, b):
    ix = min(a.x1, b.x1) - max(a.x0, b.x0); iy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return ix, iy


def qa_and_save(fig, name, allow_overlap=()):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    Wpx, Hpx = fig._W * DPI, fig._H * DPI
    rec = dict(name=name, size_in=[fig._W, fig._H], min_font_pt=None, texts=0, outside=[], overlaps=[], small=[],
               text_line=[], text_axes=[], text_patch=[])
    main_ax = fig.axes[0]
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
        boxes.append((t.get_text(), bb, t.axes))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ix, iy = _overlap_px(a, b)
            if ix > 2 and iy > 2:
                pair = (boxes[i][0][:24], boxes[j][0][:24])
                if any(k in pair[0] or k in pair[1] for k in allow_overlap):
                    continue
                rec["overlaps"].append(list(pair))
    # 글자 vs 캔버스 축의 선(헤어라인, 화살표, 열쇠 선): 글자 상자 안쪽(2 px 줄임)을 지나는 선분
    segs = []
    for ln in main_ax.findobj(Line2D):
        if not ln.get_visible() or ln.get_linewidth() <= 0:
            continue
        xy = ln.get_transform().transform(ln.get_xydata())
        for k in range(len(xy) - 1):
            segs.append((xy[k], xy[k + 1], ln))
    for txt, bb, axt in boxes:
        if axt is not main_ax:
            continue
        for p0, p1, ln in segs:
            if _seg_hits_rect(p0, p1, bb, pad=2.0):
                rec["text_line"].append(txt[:30])
                break
    # 글자 vs 다른 축(지도·축소 그림·차트)의 상자: 캔버스 글자가 축 상자와 부분적으로 겹치면 표시(완전히 안쪽은 의도한 배치)
    sub_bbs = [(a_, a_.get_window_extent(r)) for a_ in fig.axes[1:] if a_.get_visible()]
    for txt, bb, axt in boxes:
        if axt is not main_ax:
            continue
        for a_, sb in sub_bbs:
            ix, iy = _overlap_px(bb, sb)
            if ix > 2 and iy > 2:
                inside = bb.x0 >= sb.x0 - 1 and bb.x1 <= sb.x1 + 1 and bb.y0 >= sb.y0 - 1 and bb.y1 <= sb.y1 + 1
                if not inside:
                    rec["text_axes"].append(txt[:30])
                    break
    # 글자 vs 캔버스 도형(블록·셰브런·막대): 부분 겹침만 표시
    pats = []
    for pt in main_ax.findobj(Patch):
        if not pt.get_visible() or pt.get_facecolor()[3] == 0 and pt.get_edgecolor()[3] == 0:
            continue
        pb = pt.get_window_extent(r)
        if pb.width < 8 or pb.height < 8:
            continue
        pats.append((pt, pb))
    for txt, bb, axt in boxes:
        if axt is not main_ax:
            continue
        for pt, pb in pats:
            ix, iy = _overlap_px(bb, pb)
            if ix > 2 and iy > 2:
                inside = bb.x0 >= pb.x0 - 2 and bb.x1 <= pb.x1 + 2 and bb.y0 >= pb.y0 - 2 and bb.y1 <= pb.y1 + 2
                if not inside:
                    rec["text_patch"].append(txt[:30])
                    break
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
    status = "OK" if not (rec["outside"] or rec["overlaps"] or rec["small"] or rec["text_line"] or rec["text_axes"] or rec["text_patch"]) else "CHECK"
    print(f"[{name}] {status} 글자 {rec['texts']} · 최소 {rec['min_font_pt']} pt · 화살표 {rec['arrows']} · "
          f"밖 {rec['outside']} · 겹침 {rec['overlaps'][:6]} · 작은 글자 {rec['small'][:4]} · 글자–선 {rec['text_line'][:6]} · "
          f"글자–축 {rec['text_axes'][:6]} · 글자–도형 {rec['text_patch'][:6]}", flush=True)
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
    stefan_scatter(fig, xs[1] + pad + 0.28, ty0 + 0.46, tw - 0.32, th - 0.50, labelpad=5)
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
        4: [["라벨 10개 편향 진단"], ["교차검증 방법 선택"], ["라벨 분산 배치"]],
        5: [["알래스카 1 km 지도"], ["90% 구간"], ["학습 범위 밖 표시"], ["기존 ALT 지도와 비교"]],
    }
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


def covariate_list(ax, x0, x1, ytop, size=None):
    """공변량 25종의 헤어라인 목록(군 머리 + 변수). 반환: 마지막 y."""
    size = size or FS["small"]
    groups = [("기후 8 · ERA5-Land 0.1°", ["연평균 기온, TDD, FDD, √TDD", "최난·최한월 기온", "표층 토양 온도, 적설(SWE)"]),
              ("토양 9 · SoilGrids(약 5 km)", ["점토·모래·실트, 용적밀도", "조립질, pH (5–15 cm)", "유기 탄소 3층"]),
              ("지형 6 · Copernicus DEM 30 m", ["고도, 경사, 향(sin·cos)", "TPI, 거칠기(33 × 33 창)"]),
              ("ESA CCI 2 · 1 km", ["ALT 1997–2021 평균, 유효 표지"])]
    yg = ytop
    for gh, items in groups:
        T(ax, x0, yg, gh, size=size + 0.5, color=INK)
        hline(ax, x0, x1, yg - 0.14, color=HAIR, lw=0.7)
        for k_, it in enumerate(items):
            T(ax, x0 + 0.10, yg - 0.36 - k_ * 0.235, it, size=size, color=AUX)
        yg -= 0.36 + 0.235 * len(items) + 0.17
    return yg


def m6_models():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    yh = 4.98
    head(ax, 0.10, yh, "입력")
    head(ax, 2.05, yh, "결합 구조")
    head(ax, 10.20, yh, "학습기")
    ax.plot([7.40, 7.72], [yh, yh], color=TRAIN_RED, lw=1.4, ls=(0, (3, 2)), zorder=5)
    T(ax, 7.80, yh, "학습 때만 쓰는 경로", size=FS["small"], color=AUX)
    vline(ax, 1.92, 0.25, 5.12, color=SEP, lw=0.8)
    vline(ax, 10.05, 0.25, 5.12, color=SEP, lw=0.8)
    # ---- 입력 막대: 공변량 25종(지형 6, 기후 8, 토양 9, CCI 2)
    bx, bw, y_bot, y_top = 1.22, 0.32, 0.80, 4.40
    groups = [("지형", 6, "#D9E0E8"), ("기후", 8, "#B4C3D3"), ("토양", 9, "#8EA6BF"), ("CCI", 2, _C["products"]["CCI"]["hex"])]
    unit = (y_top - y_bot) / 25.0
    yc = y_bot
    for nm, k, c in groups:
        ax.add_patch(Rectangle((bx, yc), bw, k * unit, facecolor=c, edgecolor="white", lw=1.0, zorder=3))
        T(ax, bx - 0.10, yc + k * unit / 2, f"{nm} {k}", size=FS["body"] + 1, ha="right")
        if nm == "기후":
            T(ax, bx - 0.10, yc + k * unit / 2 - 0.30, "√TDD 포함", size=FS["small"], ha="right", color=AUX)
        yc += k * unit
    T(ax, bx + bw / 2, y_bot - 0.30, "공변량 25종", size=FS["small"], ha="center", color=AUX)
    # ---- 방법 행
    yL = [4.38, 3.47, 2.56, 1.65, 0.74]
    xin, xb0, xb1, xout, xnote = 2.12, 4.30, 5.15, 6.25, 7.75
    xdrop = 5.90
    bh = 0.40
    xbus = 1.78
    hline(ax, bx + bw, xbus, (y_bot + y_top) / 2, color=ARROW, lw=1.1)
    vline(ax, xbus, yL[-1], yL[0], color=ARROW, lw=1.1)
    for y in yL:
        arrow(ax, (xbus, y), (xin - 0.04, y))
    names = [[("P0", "원천 계수 Stefan"), ("P1", "재보정 Stefan")], [("R1", "물리 잔차 결합")], [("F1", "물리 입력 ML")],
             [("D1", "물리 유사라벨 증강")], [("D0", "직접 ML")]]
    for y, nm in zip(yL, names):
        cx = xin
        for code, label in nm:
            key_line(ax, cx, y + 0.31, COL[code], w=0.26, lw=3.2)
            _, wtxt = RT(ax, cx + 0.34, y + 0.31, [label], size=FS["body"] + 1, color=COL["R1"] if code == "R1" else INK)
            cx += 0.34 + wtxt + 0.24
    inputs = [["$\\sqrt{\\mathrm{TDD}}$"], ["$X$"], ["$X$", ", ", "$a$"], ["$X$"], ["$X$"]]
    in_cols = [[INK], [INK], [INK, INK, COL["P1"]], [INK], [INK]]
    for y, segs_, cols_ in zip(yL, inputs, in_cols):
        _, wv = RT(ax, xin, y, segs_, size=FS["body"] + 2, color=cols_, gap_em=0.0)
        arrow(ax, (xin + wv + 0.10, y), (xb0, y))
    blk_txt = ["최소제곱", "$g$", "$f$", "$f$", "$f$"]
    for k, (y, t_) in enumerate(zip(yL, blk_txt)):
        bh_ = 0.34 if k == 0 else bh
        block(ax, xb0, y - bh_ / 2, xb1 - xb0, bh_, t_, fill=GRN if k == 0 else YEL, size=FS["body"] + (0 if k == 0 else 3))
    targets = [["라벨 ", "$y$"], ["잔차 ", "$y - a$"], ["$y$"], ["$y$", " + 유사라벨"], ["$y$"]]
    xbc = (xb0 + xb1) / 2
    for k, (y, segs_) in enumerate(zip(yL, targets)):
        top = y - (0.34 if k == 0 else bh) / 2
        y_start = y - 0.62
        arrow(ax, (xbc, y_start), (xbc, top), color=TRAIN_RED, lw=1.6, ls=(0, (3.0, 1.8)), hl=0.08, hw=0.07)
        RT(ax, xbc + 0.10, y - 0.50, segs_, size=FS["body"], ha="left", color=TRAIN_RED)
    # 출력과 식
    arrow(ax, (xb1, yL[0]), (xout - 0.42, yL[0]))
    RT(ax, xout - 0.36, yL[0], ["$a = E\\,\\sqrt{\\mathrm{TDD}}$"], size=FS["body"] + 2, color=COL["P1"])
    RT(ax, xnote, yL[0] + 0.30, ["$E_0 = \\sum s\\,y \\,/ \\sum s^2$", "  (원천 셀)"], size=FS["body"] + 1)
    RT(ax, xnote, yL[0] - 0.06, ["$E_n = (n\\,E_{\\mathrm{ls}} + \\kappa E_0)\\,/\\,(n + \\kappa)$"], size=FS["body"] + 1)
    RT(ax, xnote, yL[0] - 0.40, ["$\\kappa = 10$", "  (대상 라벨 ", "$n$", "개)"], size=FS["body"], color=AUX)
    plus_node(ax, xdrop, yL[1])
    arrow(ax, (xb1, yL[1]), (xdrop - 0.11, yL[1]))
    T(ax, (xb1 + xdrop) / 2, yL[1] + 0.18, "$\\times\\,\\lambda$", size=FS["body"] + 1, ha="center")
    arrow(ax, (xdrop, yL[0] - 0.18), (xdrop, yL[1] + 0.11), color=COL["P1"], lw=1.3)
    arrow(ax, (xdrop + 0.11, yL[1]), (xout - 0.04, yL[1]))
    RT(ax, xout, yL[1], ["$\\hat{y} = a + \\lambda\\, g(X)$"], size=FS["body"] + 2, color=COL["R1"])
    outs = {2: "$\\hat{y} = f(X, a)$", 3: "$\\hat{y} = f(X)$", 4: "$\\hat{y} = f(X)$"}
    for k, t_ in outs.items():
        arrow(ax, (xb1, yL[k]), (xout - 0.04, yL[k]))
        RT(ax, xout, yL[k], [t_], size=FS["body"] + 2)
    notes = {1: [["$\\lambda \\in \\{0.25, 0.5, 1.0\\}$"], ["판정 기준 0.25"]], 2: [["$a$", " 또는 Kudryavtsev·"], ["토양 물성 Stefan 출력"]],
             3: [["유사라벨 ", "$E_n\\sqrt{\\mathrm{TDD}}$"], ["원천 행당 10개"]], 4: [["물리 결합 없음"]]}
    for k, lines_ in notes.items():
        for j, segs_ in enumerate(lines_):
            RT(ax, xnote + (0.45 if k == 1 else 0.0), yL[k] + 0.14 - j * 0.30, segs_, size=FS["body"],
               color=[COL["P1"] if sg == "$a$" else AUX for sg in segs_])
    # ---- 학습기 열
    learners = ["CatBoost (기본 설정)", "CatBoost (큰 모형)", "CatBoost (원천 지역\n교차검증 조정)", "랜덤 포레스트", "TabPFN", "TabICL", "MLP",
                "다중 헤드 MLP", "FT-Transformer", "RealMLP"]
    lx, lw_, lh, lg = 10.20, 1.72, 0.29, 0.06
    yk = 4.66
    for k, nm in enumerate(learners):
        h_ = 0.50 if "\n" in nm else lh
        block(ax, lx, yk - h_, lw_, h_, nm, fill=YEL, size=FS["small"], lw=1.5 if k == 0 else 0.9,
              edge=COL["R1"] if k == 0 else None)
        yk -= h_ + lg
    T(ax, lx, max(yk - 0.14, 0.14), "테두리 = 방법 비교 학습기", size=FS["small"], color=AUX)
    return qa_and_save(fig, "M6_models")


def m6b_models_detail():
    """M6 의 상세: 잔차 결합의 학습 목표와 λ 규칙, 학습기 10종과 설정, 유사라벨·물리 입력 상세."""
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    yh = 4.98
    head(ax, 0.10, yh, "물리 잔차 결합의 학습 절차")
    head(ax, 5.30, yh, "학습기")
    head(ax, 8.55, yh, "유사라벨 증강과 물리 입력")
    for xv in (5.12, 8.38):
        vline(ax, xv, 0.25, 5.12, color=SEP, lw=0.8)
    # ---- 왼쪽: 절차 4단계(초록 물리, 노랑 학습) 세로 흐름
    x0 = 0.10
    steps = [(GRN, "① 앵커 계산", [["$a_i = E\\,s_i$", ",  원천 셀 ", "$E_0$", ", 대상 라벨 셀 ", "$E_n$", " (κ = 10)"]]),
             (GRN, "② 잔차 목표", [["$r_i = y_i - a_i$", "  (원천 행 ", "$y - E_0 s$", ", 대상 행 ", "$y - E_n s$", ")"]]),
             (YEL, "③ 잔차 학습기 적합", [["$g$", " = CatBoost(", "$X$", " → ", "$r$", ") · 원천·대상 행 등가중"]]),
             (YEL, "④ 가중 결합", [["$\\hat{y} = a + \\lambda\\,g(X)$", "   λ 는 적합 뒤 적용(잔차 수축)"]])]
    y = 4.50
    for fill, nm, lines_ in steps:
        block(ax, x0, y - 0.19, 1.65, 0.38, nm, fill=fill, size=FS["small"] + 0.5)
        for j, segs_ in enumerate(lines_):
            RT(ax, x0 + 1.80, y - j * 0.28, segs_, size=FS["small"] + 0.5)
        y -= 0.78
        if y > 2.0:
            arrow(ax, (x0 + 0.82, y + 0.78 - 0.19), (x0 + 0.82, y + 0.19), lw=1.1, hl=0.07, hw=0.06)
    hline(ax, x0, 4.95, 1.86, color=HAIR, lw=0.8)
    T(ax, x0, 1.62, "λ 선택 규칙", size=FS["body"])
    RT(ax, x0, 1.32, ["후보 ", "$\\lambda \\in \\{0.25, 0.5, 1.0\\}$"], size=FS["small"] + 0.5)
    T(ax, x0, 1.04, "전이 시험: 판정 기준 λ = 0.25 (사전 고정, 저가중 잔차)", size=FS["small"], color=AUX)
    T(ax, x0, 0.78, "지역 내 시험·지도: 뽑은 라벨 안 0.5° 블록 교차검증으로 선택", size=FS["small"], color=AUX)
    T(ax, x0, 0.52, "알래스카 1 km 지도: λ = 0.5 (5겹 블록 CV)", size=FS["small"], color=AUX)
    T(ax, x0, 0.26, "λ = 0 이면 재보정 Stefan, λ = 1 이면 잔차 전체 반영", size=FS["small"], color=AUX)
    # ---- 가운데: 학습기 10종
    learners = [("CatBoost (방법 비교)", True), ("CatBoost 큰 설정", False), ("CatBoost 원천 CV 조정", False), ("랜덤 포레스트", False),
                ("TabPFN (미세조정 없음)", False), ("TabICL (미세조정 없음)", False), ("MLP", False), ("다중 헤드 MLP", False),
                ("FT-Transformer 축소형", False), ("RealMLP", False)]
    lx, lw_, lh, lg = 5.30, 2.85, 0.29, 0.065
    ly = 4.62
    for k, (nm, main) in enumerate(learners):
        yk = ly - k * (lh + lg)
        block(ax, lx, yk - lh, lw_, lh, nm, fill=YEL, size=FS["small"], lw=1.4 if main else 0.9,
              edge=COL["R1"] if main else None)
    yb = ly - len(learners) * (lh + lg)
    T(ax, lx, yb - 0.16, "CatBoost: 200회 · 학습률 0.05 · 깊이 3", size=FS["small"], color=AUX)
    T(ax, lx, yb - 0.42, "seed 2개 · CPU 결정적 실행", size=FS["small"], color=AUX)
    T(ax, lx, yb - 0.68, "신경망 4종: 원천 지역 하나 제외 CV 로 조정", size=FS["small"], color=AUX)
    T(ax, lx, yb - 0.94, "라벨 0개 직접 ML 비교에 10종 사용", size=FS["small"], color=AUX)
    # ---- 오른쪽: 유사라벨·물리 입력
    xr = 8.55
    T(ax, xr, 4.56, "물리 유사라벨 증강", size=FS["body"], color=COL["D1"])
    RT(ax, xr, 4.26, ["대상 라벨 절반 셀에 ", "$\\tilde{y} = E_n\\sqrt{\\mathrm{TDD}}$"], size=FS["small"] + 0.5)
    T(ax, xr, 3.98, "원천 행당 10개 추가 · 라벨 셀은 실측 유지", size=FS["small"], color=AUX)
    T(ax, xr, 3.72, "같은 CatBoost 로 직접 학습", size=FS["small"], color=AUX)
    T(ax, xr, 3.46, "위약 4종(순서 섞음·상수 2종·선형 지수)과 대조", size=FS["small"], color=AUX)
    hline(ax, xr, 11.90, 3.18, color=HAIR, lw=0.8)
    T(ax, xr, 2.94, "증강 + 앵커 + 잔차", size=FS["body"], color=COL["R1"])
    T(ax, xr, 2.64, "잔차 구조에 잔차 0 인 유사 행을 추가", size=FS["small"] + 0.5)
    T(ax, xr, 2.38, "(라벨 절반 셀, 원천 행당 10개)", size=FS["small"], color=AUX)
    hline(ax, xr, 11.90, 2.10, color=HAIR, lw=0.8)
    T(ax, xr, 1.86, "물리 입력 ML", size=FS["body"], color=COL["F1"])
    T(ax, xr, 1.56, "직접 ML 의 입력에 물리 출력을 추가", size=FS["small"] + 0.5)
    T(ax, xr, 1.30, "Kudryavtsev 모형 · 토양 물성 Stefan 출력", size=FS["small"], color=AUX)
    T(ax, xr, 1.04, "또는 앵커 값(원천·재보정 Stefan)", size=FS["small"], color=AUX)
    hline(ax, xr, 11.90, 0.76, color=HAIR, lw=0.8)
    T(ax, xr, 0.52, "Stefan·CCI 평균 앵커", size=FS["body"], color=_C["products"]["CCI"]["hex"])
    RT(ax, xr, 0.24, ["$a$", " = (", "$E_n\\sqrt{\\mathrm{TDD}}$", " + CCI ALT)/2, CCI 유효 셀"], size=FS["small"], color=AUX)
    return qa_and_save(fig, "M6b_models_detail")


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
    d = 3.62
    axm, proj = polar_map(fig, 0.10, 0.92, d, d, size_k=3.0, hollow_regions=("Lena Delta",), hollow_color=INK, permafrost=True)
    # 대상 확대도(지역 홀드아웃 + 100 km 완충)
    z = 1.62
    zx, zy = 4.25, 3.05
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
    pfr_key(ax, 0.10, 0.66)
    T(ax, 0.10, 0.36, "원천 = 대상과 완충 밖 라벨 셀 · 원천 셀의 78–94%가 알래스카", size=FS["small"], color=AUX)
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
    d = 3.30
    polar_map(fig, 0.42, 1.12, d, d, size_k=3.0, permafrost=True)
    import cartopy.crs as ccrs
    b = blocks_table()
    tb = b[b.reg == "Tibetan Plateau"]
    axi = sub_axes(fig, 0.10, 0.22, 1.05, 0.72, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=90, central_latitude=34))
    axi.set_extent([77, 103, 27.5, 39.5], ccrs.PlateCarree())
    axi.add_feature(land_feature("50m"), facecolor=LAND, edgecolor="none").set_rasterized(True)
    axi.scatter(tb.lon.values, tb.lat.values, s=3.0 * np.sqrt(tb.n_loc_1km.values) * 4.0, facecolor=REG["Tibetan Plateau"],
                edgecolor="white", linewidths=0.5, transform=ccrs.PlateCarree(), zorder=3)
    axi.spines["geo"].set_edgecolor(HAIR); axi.spines["geo"].set_linewidth(0.7)
    T(ax, 1.25, 0.84, "티베트 고원(삽도)", size=FS["small"], color=AUX)
    T(ax, 1.25, 0.58, "원 면적 ∝ 블록의 1 km 위치 수", size=FS["small"], color=AUX)
    pfr_key(ax, 1.25, 0.30)
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
    staircase(ax, sa, 11.92, 3.36, labels=("원천 계수", "재보정 + 잔차", "교차검증 방법 선택"), dy=0.44)
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
    ym = 3.98
    bh = 0.56
    steps = [(0.10, 1.25, GRN, "원천 계수\nStefan"), (2.30, 1.40, GRN, "계수 재보정\nκ = 10"), (4.55, 1.45, GRN, "라벨 10개\n편향 진단"),
             (7.25, 1.55, YEL, "교차검증 방법 선택\n후보 7개"), (9.75, 2.15, GRN, "1 km 지도 +\n90% 구간")]
    for x, w, f, t_ in steps:
        block(ax, x, ym - bh / 2, w, bh, t_, fill=f, size=FS["small"] + 1)
    conds = ["라벨 n ≥ 3", "n ≥ 10", "n ≥ 40", ""]
    for k_ in range(4):
        x0_ = steps[k_][0] + steps[k_][1]
        x1_ = steps[k_ + 1][0]
        arrow(ax, (x0_, ym), (x1_ - 0.02, ym))
        if conds[k_]:
            T(ax, (x0_ + x1_) / 2, ym + 0.17, conds[k_], size=FS["small"], ha="center", color=INK)
    # 조건 미충족 경로(수직 → 바닥 선 → 출력)
    yl = 3.00
    for k_ in (0, 1, 2):
        x, w = steps[k_][0], steps[k_][1]
        xc = x + w / 2
        vline(ax, xc, yl, ym - bh / 2, color=AUX, lw=1.0)
        T(ax, xc + 0.08, (yl + ym - bh / 2) / 2, ["n = 0", "n < 10", "n < 40"][k_], size=FS["small"], color=AUX)
    xo = steps[4][0] + steps[4][1] / 2
    hline(ax, steps[0][0] + steps[0][1] / 2, xo, yl, color=AUX, lw=1.0)
    arrow(ax, (xo, yl), (xo, ym - bh / 2 - 0.02), color=AUX, lw=1.0)
    T(ax, 8.60, yl - 0.18, "현재 단계의 모형으로 지도 산출", size=FS["small"], color=AUX, ha="center")
    # 단계 설명(블록 위, 두 줄)
    desc = {0: ["대상 라벨 없음", "학습 범위 밖 셀 표시"], 1: ["RT:E_n", "저가중 잔차(λ 0.25) 결합"],
            2: ["원천 계수 Stefan 의 평균 절대 오차", "크면 라벨 보정 이득 기대"],
            3: ["라벨 블록 ≤ 5겹 CV · RMSE 최소", "전체 라벨로 다시 적합"],
            4: ["90% 구간 ±20.2 cm(알래스카)", "학습 범위 밖 공변량 셀 73%"]}
    for k_, lines_ in desc.items():
        x = steps[k_][0]
        for j, t_ in enumerate(lines_):
            yy = ym + bh / 2 + 0.25 + (1 - j) * 0.27
            if t_ == "RT:E_n":
                RT(ax, x, yy, ["$E_n = (n\\,E_{\\mathrm{ls}} + \\kappa E_0)/(n + \\kappa)$"], size=FS["small"], color=AUX)
            else:
                T(ax, x, yy, t_, size=FS["small"], color=AUX)
    # 후보 7개(두 줄)
    hline(ax, 0.10, 11.90, 2.52, color=HAIR, lw=0.8)
    T(ax, 0.10, 2.28, "교차검증 후보 7개", size=FS["body"])
    cands = [("P0", "원천 계수 Stefan"), ("P1", "재보정 Stefan"), ("P1", "현지 최소제곱 Stefan"), ("R1", "물리 잔차 결합 λ 0.25"),
             ("R1", "물리 잔차 결합 λ 1.0"), ("R1", "증강 + 앵커 + 잔차"), ("D1", "물리 유사라벨 증강")]
    xk, yk = 2.30, 2.28
    for i, (code, nm) in enumerate(cands):
        if i == 4:
            xk, yk = 2.30, 1.98
        key_line(ax, xk, yk, COL[code], w=0.22, lw=3.0)
        _, wtxt = RT(ax, xk + 0.30, yk, [nm], size=FS["small"])
        xk += 0.30 + wtxt + 0.32
    T(ax, 0.10, 1.98, "(같은 라벨로 적합)", size=FS["small"], color=AUX)
    # 라벨 배치
    hline(ax, 0.10, 11.90, 1.68, color=HAIR, lw=0.8)
    T(ax, 0.10, 1.44, "라벨 배치", size=FS["body"])
    T(ax, 2.30, 1.44, "블록·공변량 공간에 분산 배치 · 효과는 지역 의존(캐나다 이득, 레나델타 손해)", size=FS["small"], color=AUX)
    T(ax, 2.30, 1.16, "사전 지정 배치 절차는 알래스카 선택 규칙에서 후보가 모두 제외되어 무작위 배치로 확정", size=FS["small"], color=AUX)
    # 출력 축소 그림(오른쪽 아래)
    arr = alaska_png("c")
    w_, h_ = fit_box(arr.shape, 1.05, 0.82)
    img_thumb(fig, 9.75, 0.14, w_, h_, arr)
    arr2 = alaska_png("d")
    w2, h2 = fit_box(arr2.shape, 1.05, 0.82)
    img_thumb(fig, 9.75 + w_ + 0.08, 0.14, w2, h2, arr2)
    T(ax, 11.90, 0.14 + h_ + 0.12, "출력: ALT 지도 · 학습 범위 밖 표시", size=FS["small"], color=AUX, ha="right")
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


# ================================================================ 실자료: ERA5-Land 월 기온(라벨 셀 하나)
def era5_cell():
    """Fig 3a 추출의 첫 라벨 셀(러시아 서부)에서 ERA5-Land 2015–2020 월평균 2 m 기온과 TDD = Σ max(T̄_m, 0)·d_m.
    자료 표의 e5_tdd 와 같은 정의(검산: 차 < 0.01 °C d)."""
    if "era5" in _CACHE:
        return _CACHE["era5"]
    import netCDF4 as nc
    C = concept()
    i = int(np.where(C["drawn"])[0][0])
    lat, lon = float(C["lat"][i]), float(C["lon"][i])
    with nc.Dataset(ROOT / "data/raw/era5land/nh_monthly_2015-2020.nc") as f:
        la, lo = f.variables["latitude"][:], f.variables["longitude"][:]
        j, k = int(np.argmin(np.abs(la - lat))), int(np.argmin(np.abs(lo - lon)))
        t2m = f.variables["t2m"][:, j, k].astype(float) - 273.15
        tt = pd.to_datetime(f.variables["valid_time"][:], unit="s")
    m = pd.DataFrame(dict(t=tt, T=t2m))
    m["mon"] = m.t.dt.month; m["days"] = m.t.dt.days_in_month
    clim = m.groupby("mon").agg(T=("T", "mean"), days=("days", "mean"))
    pos = np.clip(clim["T"].values, 0, None) * clim["days"].values
    tdd = float(pos.sum())
    tdd_table = float(C["s"][i] ** 2)
    assert abs(tdd - tdd_table) < 0.01, (tdd, tdd_table)
    _CACHE["era5"] = dict(lat=lat, lon=lon, T=clim["T"].values, days=clim["days"].values, tdd_cum=np.cumsum(pos), tdd=tdd,
                          alt=float(C["y"][i]), s=float(C["s"][i]), E0=float(C["E0"]), E1=float(C["E1"]))
    return _CACHE["era5"]


HEAT = "#C9704A"      # 지표 열 유입(따뜻한 색, 기온 막대와 같은 계열)
THAW = "#E8D9B5"      # 녹은 층(활동층)
FROZEN = "#C9D6E3"    # 언 땅(영구동토)
SURFACE = "#8C6D3F"   # 지표


def soil_column(fig, x, y, w, h, compact=False, size=None):
    """토양 기둥 도식: 가로 = 달(5–10월), 세로 = 깊이. 융해 전선 깊이 = E0·√TDD(t), TDD(t)는 라벨 셀의 ERA5-Land 월 기후값 누적(실자료).
    compact=True 는 글자를 줄인 축소판(B1 용)."""
    size = size or FS["small"]
    E = era5_cell()
    axc = sub_axes(fig, x, y, w, h)
    months = np.arange(1, 13)
    cum = np.r_[0.0, E["tdd_cum"]]                           # 달 끝의 누적 TDD
    # 달 안을 선형 보간해 전선 곡선을 매끈하게(월 경계 값은 실자료)
    tm = np.linspace(0, 12, 241)
    cum_i = np.interp(tm, np.arange(0, 13), cum)
    front = E["E0"] * np.sqrt(cum_i)
    xm0, xm1 = 4.0, 10.0                                      # 5월 초–10월 말
    zmax = max(front.max() * 1.25, 10)
    axc.add_patch(Rectangle((xm0, 0), xm1 - xm0, zmax, facecolor=FROZEN, edgecolor="none", zorder=1))
    m_ = (tm >= xm0) & (tm <= xm1)
    axc.fill_between(tm[m_], 0, front[m_], color=THAW, zorder=2, linewidth=0)
    axc.plot(tm[m_], front[m_], color=INK, lw=1.4, zorder=3)
    axc.plot([xm0, xm1], [0, 0], color=SURFACE, lw=2.4, zorder=4, solid_capstyle="butt")
    # 열 유입: 짧고 가는 수직 화살표 3개(지표 바로 위 → 녹은 층 안), 따뜻한 색
    for k_, xa in enumerate((xm0 + 0.45, xm0 + 0.75, xm0 + 1.05)):
        axc.annotate("", xy=(xa, 0.16 * zmax), xytext=(xa, -0.10 * zmax),
                     arrowprops=dict(arrowstyle="-|>", color=HEAT, lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0), zorder=5)
    axc.set_xlim(xm0, xm1); axc.set_ylim(zmax, -0.14 * zmax)
    axc.set_autoscale_on(False)
    axc.set_xticks([5, 6, 7, 8, 9, 10]); axc.set_xticklabels(["5월", "6월", "7월", "8월", "9월", "10월"], fontsize=size)
    axc.xaxis.tick_top(); axc.tick_params(axis="x", length=0, pad=3)
    for k_ in ("top", "right", "bottom"):
        axc.spines[k_].set_visible(False)
    axc.spines["left"].set_linewidth(1.0)
    axc.set_ylabel("깊이 (cm)", fontsize=size, labelpad=2)
    axc.tick_params(axis="y", labelsize=size, length=3, width=1.0)
    axc.set_yticks([t_ for t_ in (0, 50, 100) if t_ <= zmax])
    # 글자
    zf = float(front[m_][-1])
    axc.text(xm0 + 1.22, 0.02 * zmax, "지표 열 유입", fontsize=size, color=HEAT, ha="left", va="top")
    axc.text(7.4, 0.5 * E["E0"] * np.sqrt(cum[7]), "녹은 층(활동층)", fontsize=size, ha="center", va="center", color=INK)
    axc.text((xm0 + xm1) / 2, zmax * 0.86, "언 땅(영구동토)", fontsize=size, ha="center", va="center", color=INK)
    axc.text(9.0, E["E0"] * np.sqrt(cum[9]) + 0.07 * zmax, "융해 전선", fontsize=size, ha="center", va="top", color=INK)
    axc.annotate("", xy=(xm1 - 0.15, zf), xytext=(xm1 - 0.15, 0),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0, mutation_scale=10, shrinkA=0, shrinkB=0), zorder=5)
    axc.text(xm1 - 0.25, zf / 2, "ALT", fontsize=size + 1, ha="right", va="center", color=INK)
    return axc, E, zf


# ================================================================ M10 Stefan 물리와 계수
def m10_stefan_physics():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    yh = 4.98
    head(ax, 0.10, yh, "활동층과 계절 융해")
    head(ax, 4.30, yh, "Stefan 해와 계수 E")
    head(ax, 8.35, yh, "재보정이 바꾸는 것")
    for xv in (4.12, 8.17):
        vline(ax, xv, 0.25, 5.12, color=SEP, lw=0.8)
    # ---- 왼쪽: 토양 기둥(위) + 월 기온·누적 TDD(아래)
    axc, E, zf = soil_column(fig, 0.60, 2.72, 2.85, 1.72)
    axt = sub_axes(fig, 0.60, 1.20, 2.85, 1.10)
    mon = np.arange(1, 13)
    axt.bar(mon, E["T"], width=0.72, color=np.where(E["T"] > 0, "#D98B6A", FROZEN), edgecolor="none", zorder=2)
    axt.axhline(0, color=ZERO, lw=1.0, zorder=3)
    axt.set_xlim(0.4, 12.6); axt.set_xticks([1, 4, 7, 10]); axt.set_xticklabels(["1월", "4", "7", "10"], fontsize=FS["small"])
    axt.set_ylabel("기온 (°C)", fontsize=FS["small"], labelpad=2)
    axt.set_yticks([-20, 0])
    axt.tick_params(labelsize=FS["small"], length=3, width=1.0)
    for k_ in ("top", "right"):
        axt.spines[k_].set_visible(False)
    ax2 = axt.twinx()
    cum = np.r_[0.0, E["tdd_cum"]]
    ax2.plot(np.arange(0.5, 13.5), cum, color=INK, lw=1.6, zorder=4)
    ax2.set_ylim(0, E["tdd"] * 1.15)
    ax2.set_yticks([0, 1000]); ax2.tick_params(labelsize=FS["small"], length=3, width=1.0)
    ax2.set_ylabel("누적 TDD (°C d)", fontsize=FS["small"], labelpad=3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(1.0)
    T(ax, 0.10, 0.86, "막대 = 월평균 기온, 선 = 누적 TDD = Σ max(T̄ₘ, 0)·dₘ", size=FS["small"], color=AUX)
    T(ax, 0.10, 0.58, "ERA5-Land 2015–2020 월 기후값 · 러시아 서부 라벨 셀", size=FS["small"], color=AUX)
    T(ax, 0.10, 0.30, f"TDD {E['tdd']:.0f} °C d · 실측 ALT {E['alt']:.0f} cm · 예측 {E['E0'] * E['s']:.0f} cm", size=FS["small"], color=AUX)
    # ---- 가운데: 식 사슬
    xm = 4.30
    RT(ax, xm, 4.30, ["융해 깊이의 열수지 해(Stefan)"], size=FS["body"], color=AUX)
    RT(ax, xm + 0.10, 3.70, ["$\\mathrm{ALT} = \\sqrt{\\dfrac{2\\,k\\,\\mathrm{TDD}}{\\rho\\, w\\, L}}$"], size=FS["body"] + 3)
    RT(ax, xm + 0.10, 2.98, ["$= E\\,\\sqrt{\\mathrm{TDD}}$", "     ", "$E = \\sqrt{2k\\,/\\,(\\rho\\, w\\, L)}$"], size=FS["body"] + 3)
    rows = [("$k$", "열전도도(녹은 토양)"), ("$w$", "토양 수분 함량"), ("$\\rho$", "토양 밀도"), ("$L$", "얼음의 융해 잠열"),
            ("$\\mathrm{TDD}$", "융해 도일(기후 입력)")]
    for k_, (sym, nm) in enumerate(rows):
        yy = 2.44 - k_ * 0.29
        RT(ax, xm + 0.10, yy, [sym], size=FS["body"])
        T(ax, xm + 0.90, yy, nm, size=FS["small"], color=AUX)
    hline(ax, xm, 7.98, 1.08, color=HAIR, lw=0.8)
    T(ax, xm, 0.84, "E 는 토양·수분·지표 조건을 한 수로 묶는다", size=FS["body"])
    RT(ax, xm, 0.54, ["단위 cm (°C d)", "$^{-1/2}$", " · 지역마다 다르다"], size=FS["small"], color=AUX)
    T(ax, xm, 0.26, "지역 기하 평균 1.31(레나델타)–11.03(티베트 고원)", size=FS["small"], color=AUX)
    # ---- 오른쪽: 실자료 산점도(러시아 서부 31셀)와 두 직선, 재보정 식
    C = concept()
    axs = sub_axes(fig, 8.95, 1.98, 2.85, 2.30)
    sv, yv = C["s"], C["y"]
    xs_ = np.array([30.0, 50.0])
    axs.plot(xs_, C["E0"] * xs_, color=COL["P0"], lw=2.4, zorder=3)
    axs.plot(xs_, C["E1"] * xs_, color=COL["P1"], lw=2.4, zorder=3)
    axs.scatter(sv, yv, s=22, facecolor="white", edgecolor=INK, linewidths=0.9, zorder=4)
    axs.scatter(C["s_sel"], C["y_sel"], s=22, facecolor=INK, edgecolor=INK, linewidths=0.9, zorder=5)
    axs.set_xlim(30, 50); axs.set_ylim(0, 140); axs.invert_yaxis()
    axs.set_xticks([30, 40, 50]); axs.set_yticks([0, 50, 100])
    axs.tick_params(labelsize=FS["small"], length=3, width=1.0)
    for k_ in ("top", "right"):
        axs.spines[k_].set_visible(False)
    axs.set_xlabel("√TDD (√(°C d))", fontsize=FS["small"], labelpad=2)
    axs.set_ylabel("ALT (cm)", fontsize=FS["small"], labelpad=2)
    axs.text(30.6, C["E0"] * 30.6 - 5, f"원천 계수 E₀ = {C['E0']:.2f}", color=COL["P0"], fontsize=FS["small"], ha="left", va="bottom")
    axs.text(49.4, C["E1"] * 49.4 + 7, f"재보정 Eₙ = {C['E1']:.2f}", color=COL["P1"], fontsize=FS["small"], ha="right", va="top")
    T(ax, 8.35, 4.52, "러시아 서부 라벨 셀 31개 · 검정 = 뽑은 라벨 10개", size=FS["small"], color=AUX)
    RT(ax, 8.35, 1.20, ["$E_n = (n\\,E_{\\mathrm{ls}} + \\kappa\\,E_0)\\,/\\,(n + \\kappa)$", "   ", "$\\kappa = 10$"], size=FS["body"])
    T(ax, 8.35, 0.86, "원천 계수는 라벨 10개 무게로 남는다(수축)", size=FS["small"], color=AUX)
    RT(ax, 8.35, 0.58, ["$E_{\\mathrm{ls}}$", " = 대상 라벨 n개의 원점 통과 최소제곱 기울기"], size=FS["small"], color=AUX)
    T(ax, 8.35, 0.30, "주 지역의 원천 계수 1.59–1.62 (본 연구 자료)", size=FS["small"], color=AUX)
    return qa_and_save(fig, "M10_stefan_physics")


# ================================================================ B1 배경: ALT 와 관측 공백
def xl_panel(panel):
    """XL 알래스카 제품 비교 슬라이드 PNG 에서 위 행 패널(a 잔차 ML, b CCI v5, c Wei 2026, d Aalto 2018, e Yi-Kimball)을 자른다."""
    from PIL import Image
    im = Image.open(ROOT / "deck/assets/paper_report/maps/XL_Alaska_product_differences_slide.png").convert("RGB")
    a = np.asarray(im)
    dark = (a.astype(int).sum(2) < 120)
    H_ = dark.shape[0]
    top = dark[: H_ // 2]
    ys = np.where(top.mean(1) > 0.5)[0]
    yg = np.split(ys, np.where(np.diff(ys) > 5)[0] + 1)
    ytop, ybot = int(yg[0].max()), int(yg[-1].min())
    colfrac = dark[ytop + 10: ybot - 10].mean(0)
    xs = np.where(colfrac > 0.95)[0]
    groups = np.split(xs, np.where(np.diff(xs) > 5)[0] + 1)
    edges = [(int(g.min()), int(g.max())) for g in groups]
    pairs = [(edges[i][1], edges[i + 1][0]) for i in range(0, len(edges) - 1, 2)]
    idx = dict(a=0, b=1, c=2, d=3, e=4)[panel]
    x0, x1 = pairs[idx]
    return a[ytop + 3: ybot - 2, x0 + 3: x1 - 2]


def b1_background():
    """배경: 왼쪽 큰 관측 지도(지역 색, 영구동토 구역), 가운데 지역·구역 열쇠, 오른쪽 위 토양 기둥. 오른쪽 아래는 덱이 세 줄을 넣는 자리."""
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    d = 5.05
    polar_map(fig, 0.05, 0.08, d, d, size_k=3.6, permafrost=True, edge_lw=0.6)
    # 지역 열쇠(라벨 행 수) + 영구동토 구역
    t1 = table1().set_index("target")
    regs = [("Alaska", "Alaska"), ("Lena Delta", "Lena"), ("Canada", "Canada"), ("W Russia", "Russia_W"), ("E Russia", "Russia_E"),
            ("Central Russia", "Russia_C_LGD"), ("Tibetan Plateau", "Tibet_LGD")]
    kx, ky0 = 5.35, 4.70
    T(ax, kx, ky0 + 0.30, "관측 지역 · 라벨 행 수", size=FS["small"], color=AUX)
    for i, (reg, key) in enumerate(regs):
        yy = ky0 - i * 0.33
        ax.scatter([kx + 0.09], [yy], s=70, color=REG[reg], linewidths=0, zorder=5)
        T(ax, kx + 0.28, yy, REG_KO[reg], size=FS["small"] + 0.5)
        n = int(float(t1.loc[key, "label_rows"]))
        T(ax, kx + 2.00, yy, f"{n:,}" if n >= 10000 else f"{n}", size=FS["small"] + 0.5, ha="right")
    yk = ky0 - len(regs) * 0.33 - 0.12
    ax.add_patch(Rectangle((kx, yk - 0.07), 0.16, 0.14, facecolor=PF_CONT, edgecolor="none", zorder=5))
    T(ax, kx + 0.28, yk, "연속 영구동토", size=FS["small"], color=AUX)
    ax.add_patch(Rectangle((kx, yk - 0.40), 0.16, 0.14, facecolor=PF_DISC, edgecolor="none", zorder=5))
    T(ax, kx + 0.28, yk - 0.33, "불연속 영구동토", size=FS["small"], color=AUX)
    T(ax, kx, yk - 0.68, "원 면적 ∝ 1 km 위치 수", size=FS["small"], color=AUX)
    T(ax, kx, yk - 0.96, "티베트 고원은 지도 범위 밖", size=FS["small"], color=AUX)
    # 토양 기둥(오른쪽 위)
    T(ax, 7.75, 5.02, "활동층 두께(ALT) = 계절 최대 융해 깊이 (cm)", size=FS["body"])
    soil_column(fig, 8.30, 2.72, 3.45, 1.80)
    T(ax, 7.75, 2.46, "ERA5-Land 월 기후값으로 그린 융해 전선(러시아 서부 라벨 셀)", size=FS["small"], color=AUX)
    # 오른쪽 아래(x 7.75–11.9, y 0.2–2.1)는 덱이 글 세 줄을 넣는 자리
    return qa_and_save(fig, "B1_background")


# ================================================================ B2 선행 연구 표
def b2_prior_work():
    W, H = 12.0, 4.6
    fig, ax = canvas(W, H)
    cols = [("연구", 0.10), ("지역", 1.62), ("평가 유형", 3.62), ("평가 방법", 4.82), ("라벨 조건", 7.52), ("물리 기준선", 8.72)]
    rows = [
        ("Gautam 2025", "알래스카(CALM 68지점)", "무작위 CV", "무작위 70/30 분할 1회", "한 조건", "병렬 비교(지점 역산 E), 재보정 없음"),
        ("Pilyugina 2023", "—", "무작위 CV", "시간 분할 · 무작위 5겹", "한 조건", "없음(Kudryavtsev 출력은 입력)"),
        ("Wang G. 2025", "칭하이-티베트 고원", "무작위 CV", "무작위 10겹", "한 조건", "없음(Stefan 출력은 입력)"),
        ("Ran 2022 (CEE)", "청장고원", "무작위 CV", "지역 안 10겹", "한 조건", "없음(E 를 공변량으로 회귀)"),
        ("Zhang C. 2024", "내륙 알래스카 실험지 2곳", "무작위 CV", "실험지 안 5겹", "한 조건", "없음(E 를 원격탐사로 회귀)"),
        ("Aalto 2018", "환북극", "공간 CV", "거리 블록 교차검증", "한 조건", "없음"),
        ("Karjalainen 2019", "환북극", "공간 CV", "500 km 거리 제외", "한 조건", "없음"),
        ("Ran 2022 (ESSD)", "범북극", "공간 CV", "거리 블록 교차검증", "한 조건", "없음"),
        ("Wei 2026", "북반구", "공간 CV", "지점 하나 제외", "한 조건", "없음"),
        ("본 연구", "7개 지역(새 지역 2곳 포함)", "지역 홀드아웃", "100 km 완충 · 블록 절반 분할", "0개–전량 곡선", "원천 계수 + 같은 라벨 재보정(κ = 10)"),
    ]
    ytop = 4.30
    rh = 0.335
    hline(ax, 0.10, 11.90, ytop + 0.06, color=INK, lw=1.2)
    for nm, x in cols:
        T(ax, x, ytop - 0.14, nm, size=FS["small"], color=AUX)
    hline(ax, 0.10, 11.90, ytop - 0.34, color=INK, lw=0.7)
    y = ytop - 0.34 - rh / 2 - 0.02
    for i, r in enumerate(rows):
        last = i == len(rows) - 1
        if last:
            hline(ax, 0.10, 11.90, y + rh / 2, color=INK, lw=0.7)
        c = COL["R1"] if last else INK
        for (nm, x), v in zip(cols, r):
            T(ax, x, y, v, size=FS["small"] + 0.5, color=c)
        y -= rh
    hline(ax, 0.10, 11.90, y + rh / 2, color=INK, lw=1.2)
    T(ax, 0.10, y + rh / 2 - 0.22, "— = 지역 기재 없음 · 무작위 CV = 학습 지역 안 무작위 교차검증 · 공간 CV = 거리 블록·지점 제외 교차검증", size=FS["small"], color=AUX)
    T(ax, 0.10, y + rh / 2 - 0.48, "기존 ALT 기계학습 연구 9편은 모두 지역 홀드아웃 없음, 라벨 한 조건, 같은 라벨로 재보정한 물리 기준선 없음",
      size=FS["small"], color=AUX)
    return qa_and_save(fig, "B2_prior_work")


# ================================================================ R0 결과 요약: 라벨 수별 오차 변화와 권고 방법
def r0_summary_ladder():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    pc = pd.read_csv(PF / "fig2_pool_curves.csv")
    f7 = pd.read_csv(ROOT / "outputs/figures/paper/v3_restructure/Fig7_source_data.csv")
    wpath = f7[(f7.panel == "d") & (f7.element == "path") & (f7.method == "cv_selection")]
    series = [("P1", "recalibrated_stefan", NAME["P1"]), ("R1", "anchor_residual", "물리 잔차 결합(λ 0.25)"), ("D0", "direct_ml", NAME["D0"])]
    panels = [("a", "E1_P4_n_le_10", "주 4지역 평균", [0, 3, 10], 0.75, 4.95),
              ("b", "E2_LenaCanada_all_n", "레나델타·캐나다 평균", [0, 3, 10, 40, 160, 320, -1], 6.05, 11.80)]
    ylo, yhi = -4.2, 4.2
    lab = {0: "0", 3: "3", 10: "10", 40: "40", 160: "160", 320: "320", -1: "전량"}
    for pl, ed, title, ns, x0, x1 in panels:
        xpos = {n: x0 + 0.25 + (x1 - x0 - 0.5) * k / (len(ns) - 1) for k, n in enumerate(ns)}
        ybot, ytop = 2.05, 4.60
        Y_ = lambda v: ybot + (v - ylo) / (yhi - ylo) * (ytop - ybot)
        panel_letter(ax, x0 - 0.62, 4.92, pl)
        T(ax, x0 - 0.36, 4.92, title, size=FS["body"])
        ax.add_patch(Rectangle((x0, Y_(-0.5)), x1 - x0, Y_(0.5) - Y_(-0.5), facecolor=BAND, edgecolor="none", zorder=1))
        hline(ax, x0, x1, Y_(0), color=COL["P0"], lw=2.0, z=2)
        # 축
        vline(ax, x0, ybot, ytop, color=INK, lw=1.0)
        for v in (-4, -2, 0, 2, 4):
            hline(ax, x0 - 0.06, x0, Y_(v), color=INK, lw=1.0)
            T(ax, x0 - 0.12, Y_(v), f"{v:d}".replace("-", "−"), size=FS["small"], ha="right")
        hline(ax, x0, x1, ybot, color=INK, lw=1.0)
        for n in ns:
            vline(ax, xpos[n], ybot - 0.06, ybot, color=INK, lw=1.0)
            T(ax, xpos[n], ybot - 0.20, lab[n], size=FS["small"], ha="center")
        T(ax, (x0 + x1) / 2, ybot - 0.44, "대상 라벨 수", size=FS["small"], ha="center")
        d = pc[pc.edition == ed]
        for k_, (code, _, nm) in enumerate(series):
            dd = d[d.method == code].set_index("n")
            xs_, ys_ = [], []
            for n in ns:
                if n not in dd.index:
                    continue
                r = dd.loc[n]
                xs_.append(xpos[n]); ys_.append(Y_(r.delta))
                off = (k_ - 1) * 0.045
                ax.plot([xpos[n] + off] * 2, [Y_(r.ci_lo), Y_(r.ci_hi)], color=COL[code], lw=2.6, solid_capstyle="butt", zorder=4)
                ax.plot([xpos[n] + off + 0.05] * 2, [Y_(r.ci_lo_beq), Y_(r.ci_hi_beq)], color=COL[code], lw=1.0,
                        solid_capstyle="butt", zorder=4)
            ax.plot(xs_, ys_, color=COL[code], lw=1.8, ls="-" if code != "D0" else (0, (3.0, 1.6)), zorder=5)
            ax.scatter(xs_, ys_, s=26, facecolor="white", edgecolor=COL[code], linewidths=1.3, zorder=6)
        # 권고 방법 표시(강조색): a 는 n 3·10 의 물리 잔차 결합, b 는 40·160·전량의 교차검증 방법 선택 점(wf_curve 서술값)
        if pl == "a":
            dd = d[d.method == "R1"].set_index("n")
            for n in (3, 10):
                ax.scatter([xpos[n]], [Y_(dd.loc[n].delta)], s=70, facecolor=COL["R1"], edgecolor=COL["R1"], zorder=7)
            T(ax, xpos[10] - 0.10, Y_(dd.loc[10].delta) - 0.30, "재보정 + 저가중 잔차", size=FS["small"], color=COL["R1"], va="top", ha="right")
            ax.scatter([xpos[0]], [Y_(0)], s=70, facecolor=COL["P0"], edgecolor=COL["P0"], zorder=7)
        else:
            wp = wpath.set_index("n")
            xs_, ys_ = [], []
            for n in (40, 160, -1):
                v = float(wp.loc[float(n)].delta)
                xs_.append(xpos[n]); ys_.append(Y_(v))
            ax.plot(xs_, ys_, color=COL["W"], lw=1.8, zorder=6)
            ax.scatter(xs_, ys_, s=70, facecolor=COL["W"], edgecolor=COL["W"], zorder=7)
            key_line(ax, 6.05, 1.30, COL["W"], w=0.26, lw=3.0)
            T(ax, 6.39, 1.30, "교차검증 방법 선택(점 추정, 40개 이상의 권고)", size=FS["small"], color=COL["W"])
    T(ax, 0.75, 1.30, "오차 변화 (cm) = 방법 RMSE − 원천 계수 Stefan RMSE · 음수 = 오차 감소", size=FS["small"], color=AUX)
    T(ax, 0.75, 1.02, "b 풀에서는 재보정이 캐나다 오차를 키워 라벨을 쓰는 방법이 0 위에 놓인다", size=FS["small"], color=AUX)
    # 열쇠
    kx = 6.05
    for k_, (code, _, nm) in enumerate(series):
        xk_ = kx + [0.0, 1.65, 3.95][k_]
        key_line(ax, xk_, 1.02, COL[code], w=0.26, lw=2.6, ls="-" if code != "D0" else (0, (3.0, 1.6)))
        T(ax, xk_ + 0.34, 1.02, nm, size=FS["small"])
    key_line(ax, 0.75, 0.74, COL["P0"], w=0.26, lw=2.0)
    T(ax, 1.09, 0.74, "원천 계수 Stefan = 0 (기준선)", size=FS["small"], color=AUX)
    ax.plot([kx, kx], [0.67, 0.81], color=INK, lw=2.6, solid_capstyle="butt"); T(ax, kx + 0.10, 0.74, "셀 가중 95% CI", size=FS["small"], color=AUX)
    ax.plot([kx + 1.65, kx + 1.65], [0.67, 0.81], color=INK, lw=1.0, solid_capstyle="butt"); T(ax, kx + 1.75, 0.74, "블록 등가중 CI", size=FS["small"], color=AUX)
    ax.add_patch(Rectangle((kx + 3.30, 0.67), 0.22, 0.14, facecolor=BAND, edgecolor="none")); T(ax, kx + 3.60, 0.74, "±0.5 cm 동등 한계", size=FS["small"], color=AUX)
    # 아래 띠: 라벨 수별 권고 단계
    yb = 0.22
    steps = [(0.75, 1.75, COL["P0"], "물리식(원천 계수)"), (1.95, 4.95, COL["R1"], "재보정 + 잔차 결합"),
             (6.05, 7.80, COL["P0"], "물리식"), (7.95, 8.90, COL["R1"], "재보정 + 잔차"), (9.05, 11.80, COL["W"], "교차검증 방법 선택")]
    for xa_, xb_, c, nm in steps:
        ax.plot([xa_, xb_], [yb, yb], color=c, lw=3.2, solid_capstyle="butt", zorder=4)
        T(ax, (xa_ + xb_) / 2, yb + 0.19, nm, size=FS["small"], ha="center", color=c if c != COL["P0"] else AUX)
    return qa_and_save(fig, "R0_summary_ladder")


# ================================================================ E1 증거의 규모
def e1_evidence_scope():
    W, H = 12.0, 4.6
    fig, ax = canvas(W, H)
    yh = 4.38
    head(ax, 0.10, yh, "시험 설계와 규모")
    head(ax, 6.90, yh, "지역 × 라벨 수: 시험한 칸")
    vline(ax, 6.70, 0.20, 4.50, color=SEP, lw=0.8)
    src = {}
    kv = [
        ("평가 지역", "7곳 (주 4지역, 알래스카, 새 지역 2곳)", "outputs/figures/paper/v3_restructure/Table1_data.tex"),
        ("전이 대상", "27개 (지역 7 + 하위 지역 10 × 2모드)", "results/rescale_lg/data/processed/lg/lg_meta.json targets; paper/claims/D_data_and_design/README.md E25"),
        ("라벨 수 격자", "8단계 (0, 3, 10, 40, 160, 320, 1000, 전량)", "results/rescale_lg/data/processed/lg/lg_meta.json n_grid; D README E24"),
        ("방법·학습기", "방법 계열 5, 방법 12종, 학습기 10종", "deck/deck_spec_paper_report.json S10(5계열); lg_meta.json methods(12); methods.tex(학습기 10종)"),
        ("분할·추출·seed", "블록 절반 분할 5회 × 라벨 추출 5회 × 학습 seed 2", "lg_meta.json splits, draws, seeds; D README E25"),
        ("등록 가설", "174개 (실행 전 등록 문서 7종)", "docs/EXPERIMENT_PLAN_{LG,LGF,LGU,WRAPUP,WF,FINAL_BATCH,FINAL_BATCH_ADDENDUM_XM}: 45+11+17+15+24+58+4"),
        ("주 가설 10개", "다중 비교 보정 뒤 방향 확정 6(감소 5, 증가 1) · 동등 1 · 미결정 3",
         "data/processed/paper_figs/fig7_d_ab10.csv abstract_rule a/b/c/d and verdict4 (AB1–AB10); methods.tex Statistical analysis (Holm, m = 10)"),
        ("모형 적합", "최소 863,229건 (실행 기록의 합)", "lg_meta.json 129,887; lgx_meta.json 335,089; ladder 5,556; lgt_meta.json 43,544; lgd shards 27,770; lgf_meta.json 15,320; wf_meta 81,836; wf2b_meta 25,794; wf3b_meta 14,818; XB 40,376; XC 59,073; XD 46,104; XE r1b 19,650; XG 280; XI 5,380; XM 10,412"),
        ("신뢰구간", "0.5° 블록 재표집 10,000회 (라벨 수 격자 집계 1000회)", "methods.tex Statistical analysis"),
        ("두 계산 환경 재현", "물리식·CatBoost 결과 차 3.6e-14 cm (34,360개 값)", "methods.tex Computing environments"),
    ]
    y = 3.98
    x0, x1 = 0.10, 6.50
    for k_, (key, val, source) in enumerate(kv):
        T(ax, x0, y, key, size=FS["small"], color=AUX)
        T(ax, x0 + 1.55, y, val, size=FS["small"], color=INK if key != "모형 적합" else COL["R1"])
        hline(ax, x0, x1, y - 0.19, color=HAIR, lw=0.6)
        src[key] = dict(value=val, source=source)
        y -= 0.385
    # 지역 × 라벨 수 행렬(시험한 칸 = 지역 색 채움)
    ns = ["0", "3", "10", "40", "160", "320", "1000", "전량"]
    regions = [("Alaska", "알래스카", {"0", "3", "10", "40", "160", "320", "1000", "전량"}),
               ("Lena Delta", "레나델타", {"0", "3", "10", "40", "160", "320", "1000", "전량"}),
               ("Canada", "캐나다", {"0", "3", "10", "40", "160", "320", "전량"}),
               ("W Russia", "러시아 서부", {"0", "3", "10", "전량"}),
               ("E Russia", "러시아 동부", {"0", "3", "10", "전량"}),
               ("Central Russia", "러시아 중부", {"0", "3", "10", "전량"}),
               ("Tibetan Plateau", "티베트 고원", {"0", "3", "10", "40", "전량"})]
    mx0, my0 = 8.35, 3.70
    cw, rh = 0.42, 0.40
    for j, n in enumerate(ns):
        T(ax, mx0 + j * cw + cw / 2, my0 + 0.26, n, size=FS["small"], ha="center", color=AUX)
    for i, (reg, ko, tested) in enumerate(regions):
        yy = my0 - i * rh
        T(ax, mx0 - 0.12, yy - rh / 2, ko, size=FS["small"], ha="right")
        for j, n in enumerate(ns):
            xx = mx0 + j * cw
            if n in tested:
                ax.add_patch(Rectangle((xx + 0.05, yy - rh + 0.05), cw - 0.10, rh - 0.10, facecolor=REG[reg], edgecolor="none", zorder=3))
            else:
                ax.add_patch(Rectangle((xx + 0.05, yy - rh + 0.05), cw - 0.10, rh - 0.10, facecolor="white", edgecolor=HAIR, lw=0.7, zorder=3))
    T(ax, mx0 + 4 * cw, my0 + 0.52, "대상 라벨 수", size=FS["small"], ha="center", color=AUX)
    ybot = my0 - len(regions) * rh
    T(ax, 6.90, ybot - 0.24, "채움 = 결과가 있는 칸(라벨 블록 셀 수보다 작은 n) · 빈 칸 = 시험하지 않음", size=FS["small"], color=AUX)
    T(ax, 6.90, ybot - 0.50, "알래스카는 참조 대상 · 러시아 중부(확충판)·티베트 고원은 새 지역", size=FS["small"], color=AUX)
    src["matrix"] = dict(source="results/rescale_lg/data/processed/lg/lg_curve.csv (method axis, mode x); data/processed/lgd/lgd_curve_lic.csv (Russia_C expanded, Tibet_LGD)",
                         cells={ko: sorted(tested, key=ns.index) for _, ko, tested in regions})
    (OUT / "E1_evidence_scope_source_values.json").write_text(json.dumps(src, ensure_ascii=False, indent=1))
    return qa_and_save(fig, "E1_evidence_scope")


# ================================================================ 결과 그림 공용: 작은 포레스트 행
def forest_row(ax, x0, x1, y, vmin, vmax, d, lo, hi, dlo, dhi, color, band=True, zero=True, lw_c=3.0, lw_b=1.3, dot=46, show_beq=False):
    X_ = lambda v: x0 + (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin) * (x1 - x0)
    ax.plot([X_(lo), X_(hi)], [y, y], color=color, lw=lw_c, solid_capstyle="butt", zorder=4)
    if show_beq:
        ax.plot([X_(dlo), X_(dhi)], [y - 0.075, y - 0.075], color=color, lw=lw_b, solid_capstyle="butt", zorder=4)
    ax.scatter([X_(d)], [y], s=dot, facecolor=color, edgecolor=color, linewidths=1.2, zorder=5)
    return X_


def mini_axis(ax, x0, x1, ybot, ytop, vmin, vmax, ticks, band=True, label=None, size=None, spans=None):
    """작은 수평 축. spans = [(y0, y1), ...] 이면 동등 띠와 0 선을 그 구간에만 그린다(행 머리글과 겹치지 않게)."""
    size = size or FS["small"]
    X_ = lambda v: x0 + (v - vmin) / (vmax - vmin) * (x1 - x0)
    for (ya, yb) in (spans or [(ybot, ytop)]):
        if band:
            ax.add_patch(Rectangle((X_(-0.5), ya), X_(0.5) - X_(-0.5), yb - ya, facecolor=BAND, edgecolor="none", zorder=1))
        vline(ax, X_(0), ya, yb, color=ZERO, lw=1.0, z=2)
    hline(ax, x0, x1, ybot, color=INK, lw=1.0)
    for v in ticks:
        vline(ax, X_(v), ybot - 0.06, ybot, color=INK, lw=1.0)
        T(ax, X_(v), ybot - 0.19, f"{v:+d}".replace("-", "−").replace("+0", "0") if isinstance(v, int) else str(v), size=size, ha="center")
    if label:
        T(ax, (x0 + x1) / 2, ybot - 0.45, label, size=size, ha="center", color=AUX)
    return X_


def r1_results_by_range():
    """라벨 수 구간별 권고 방법과 핵심 대비(오차 변화, 두 가중 95 % CI). 수치는 열람한 기록에서만."""
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    pc = pd.read_csv(PF / "fig2_pool_curves.csv").set_index(["edition", "method", "n"])
    pa = pd.read_csv(PF / "fig3_a.csv").set_index(["placebo", "n"])
    lt = pd.read_csv(ROOT / "results/rescale_lg/data/processed/lg/lg_tests.csv")
    l4 = lt[(lt.test_id == "L4") & (lt.scope == "MEAN4") & (lt.n == 10) & (lt.lam == 0.25)].iloc[0]
    f7 = pd.read_csv(ROOT / "outputs/figures/paper/v3_restructure/Fig7_source_data.csv")
    xc = f7[(f7.panel == "e") & (f7.element == "pooled") & (f7.contrast == "A1-A3")].set_index("n")
    wr = f7[(f7.panel == "a") & (f7.region == "Alaska") & (f7.method == "anchor_residual") & (f7.n == 1000)].iloc[0]
    xm = pd.read_csv(ROOT / "data/processed/xbatch/XM_landcover_vegetation/sealed/xm_tests.csv")
    xmb = xm[(xm.hypothesis == "XM-b") & (xm.n == -1) & (xm.target == "Lena|r")].iloc[0]
    # 구간 축
    bounds = [0.30, 3.12, 5.98, 8.84, 11.80]
    yax = 4.58
    hline(ax, bounds[0], bounds[-1], yax, color=INK, lw=1.0)
    for xb_ in bounds:
        vline(ax, xb_, yax - 0.07, yax + 0.07, color=INK, lw=1.0)
    segs = ["라벨 0개", "3–10개", "40–160개", "수백 개 이상"]
    for k, lb in enumerate(segs):
        T(ax, (bounds[k] + bounds[k + 1]) / 2, yax + 0.26, lb, size=FS["body"] + 0.5, ha="center")
    for xb_ in bounds[1:-1]:
        vline(ax, xb_, 0.62, yax - 0.28, color=SEP, lw=0.8, ls=(0, (3, 3)))
    cols = [(bounds[k] + 0.12, bounds[k + 1] - 0.12) for k in range(4)]
    methods = [[("P0", "원천 계수 Stefan"), ("D1", "물리 유사라벨 증강")], [("P1", "재보정 Stefan"), ("R1", "+ 저가중 잔차")],
               [("W", "교차검증 방법 선택")], [("R1", "물리 잔차 결합")]]
    for k, ms in enumerate(methods):
        x0, x1 = cols[k]
        yy = 4.10
        for code, nm in ms:
            key_line(ax, x0, yy, COL[code], w=0.26, lw=3.2)
            T(ax, x0 + 0.34, yy, nm, size=FS["body"], color=COL["R1"] if code == "R1" else INK)
            yy -= 0.30
    rows = [
        (0, "물리 유사라벨 − 섞은 유사라벨", COL["D1"], *[pa.loc[("shuffle", 0)][c] for c in ("delta", "ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq")]),
        (0, "직접 ML − 원천 계수 Stefan", COL["D0"], *[pc.loc[("E1_P4_n_le_10", "D0", 0)][c] for c in ("delta", "ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq")]),
        (1, "재보정 − 원천 계수 Stefan (10개)", COL["P1"], *[pc.loc[("E1_P4_n_le_10", "P1", 10)][c] for c in ("delta", "ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq")]),
        (1, "잔차 결합 − 재보정 Stefan (10개)", COL["R1"], l4.delta, l4.ci_lo, l4.ci_hi, l4.ci_lo_beq, l4.ci_hi_beq),
        (2, "방법 선택 − 재보정 (40개)", COL["W"], *[xc.loc[40.0][c] for c in ("delta", "ci_lo", "ci_hi", "delta_blockeq_lo", "delta_blockeq_hi")]),
        (2, "방법 선택 − 재보정 (160개)", COL["W"], *[xc.loc[160.0][c] for c in ("delta", "ci_lo", "ci_hi", "delta_blockeq_lo", "delta_blockeq_hi")]),
        (3, "잔차 결합 − 재보정 (알래스카 1000개)", COL["R1"], wr.delta, wr.ci_lo, wr.ci_hi, wr.delta_blockeq_lo, wr.delta_blockeq_hi),
        (3, "+10–20 m 입력 − 25종 (레나델타)", COL["R1"], xmb.delta, xmb.ci_lo, xmb.ci_hi, xmb.ci_lo_beq, xmb.ci_hi_beq),
    ]
    vmin, vmax = -4.2, 4.2
    ybot, ytop = 1.95, 3.38
    rows_y = [ytop - 0.42 - j * 0.66 for j in range(2)]
    for k in range(4):
        x0, x1 = cols[k]
        mini_axis(ax, x0 + 0.05, x1 - 0.05, ybot, ytop, vmin, vmax, [-4, -2, 0, 2, 4], label="오차 변화 (cm, 음수 = 오차 감소)",
                  spans=[(y - 0.17, y + 0.12) for y in rows_y])
    slot = {0: 0, 1: 0, 2: 0, 3: 0}
    for col, nm, c, d, lo, hi, dlo, dhi in rows:
        x0, x1 = cols[col]
        j = slot[col]; slot[col] += 1
        y = rows_y[j]
        T(ax, x0 + 0.05, y + 0.26, nm, size=FS["small"], color=INK, z=7)
        forest_row(ax, x0 + 0.05, x1 - 0.05, y, vmin, vmax, float(d), float(lo), float(hi), float(dlo), float(dhi), c)
    facts = {
        0: ["30개 대상 중 2 cm 넘게 악화:", "직접 ML 18개, 유사라벨 증강 2개", "(라벨 0개, 사후 서술)"],
        1: ["라벨 10개 재보정 −2.45 cm", "(주 4지역 평균) · 잔차 추가 이득은", "0.5 cm 미만(보정 뒤 비유의)"],
        2: ["레나델타·캐나다 평균 −1.35 / −1.51 cm", "이득은 캐나다에서(−2.96 cm),", "레나델타는 동등"],
        3: ["알래스카 1000개: 14.41 → 13.87 cm", "레나델타: 10–20 m 피복·식생 입력", "추가로 −1.50 cm(탐색)"],
    }
    for k, lines_ in facts.items():
        x0, _ = cols[k]
        for j, t_ in enumerate(lines_):
            T(ax, x0, 1.22 - j * 0.26, t_, size=FS["small"], color=AUX)
    T(ax, 0.30, 0.36, "막대 = 95% 신뢰구간 · 회색 띠 ±0.5 cm 동등 한계 · '재보정' = 재보정 Stefan", size=FS["small"], color=AUX)
    src = dict(
        n0_pseudo_vs_shuffle="data/processed/paper_figs/fig3_a.csv shuffle n0",
        n0_direct_vs_source="data/processed/paper_figs/fig2_pool_curves.csv E1 D0 n0",
        n10_recal="fig2_pool_curves.csv E1 P1 n10", n10_resid_vs_recal="results/rescale_lg/data/processed/lg/lg_tests.csv L4 MEAN4 n10 λ 0.25",
        n40_160_cv_vs_recal="outputs/figures/paper/v3_restructure/Fig7_source_data.csv panel e A1-A3 (XC-2b, XC-2c)",
        alaska_1000="Fig7_source_data.csv panel a Alaska anchor_residual n 1000 (rmse_recal 14.41, rmse_method 13.87)",
        lena_xm="data/processed/xbatch/XM_landcover_vegetation/sealed/xm_tests.csv XM-b n -1 Lena|r",
        risk_table="docs/RESEARCH_OVERVIEW_2026-10-02.md 9절(직접 ML 18/30 대 증강 2/30, WF0 사후 서술); paper/claims/C1_label0_safety/README.md",
        canada_40="Fig7_source_data.csv panel e region_value Canada n 40 (−2.96)")
    (OUT / "R1_results_by_range_source_values.json").write_text(json.dumps(src, ensure_ascii=False, indent=1))
    return qa_and_save(fig, "R1_results_by_range")


def r2_summary_curve():
    """(a) 전이 주 4지역 평균 n 0·3·10(원천 계수 Stefan 대비), (b) 지역 내 라벨 200개–전량, 지역별(재보정 Stefan 대비)."""
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    pc = pd.read_csv(PF / "fig2_pool_curves.csv")
    d = pc[pc.edition == "E1_P4_n_le_10"]
    # ---- (a)
    panel_letter(ax, 0.10, 4.92, "a"); T(ax, 0.36, 4.92, "전이 · 주 4지역 평균 · 라벨 0–10개", size=FS["body"])
    xa0, xa1, ybot, ytop = 1.00, 5.20, 1.30, 4.38
    ns = [0, 3, 10]
    xpos = {n: xa0 + 0.35 + (xa1 - xa0 - 0.7) * k / 2 for k, n in enumerate(ns)}
    ylo, yhi = -4.0, 4.0
    Y_ = lambda v: ybot + (np.clip(v, ylo, yhi) - ylo) / (yhi - ylo) * (ytop - ybot)
    ax.add_patch(Rectangle((xa0, Y_(-0.5)), xa1 - xa0, Y_(0.5) - Y_(-0.5), facecolor=BAND, edgecolor="none", zorder=1))
    hline(ax, xa0, xa1, Y_(0), color=COL["P0"], lw=2.0, z=2)
    T(ax, xa1 - 0.05, Y_(0.5) + 0.12, "원천 계수 Stefan = 0", size=FS["small"], color=AUX, ha="right")
    vline(ax, xa0, ybot, ytop, color=INK, lw=1.0)
    for v in (-4, -2, 0, 2, 4):
        hline(ax, xa0 - 0.06, xa0, Y_(v), color=INK, lw=1.0)
        T(ax, xa0 - 0.12, Y_(v), f"{v:d}".replace("-", "−"), size=FS["small"], ha="right")
    hline(ax, xa0, xa1, ybot, color=INK, lw=1.0)
    for n in ns:
        vline(ax, xpos[n], ybot - 0.06, ybot, color=INK, lw=1.0)
        T(ax, xpos[n], ybot - 0.20, str(n), size=FS["small"], ha="center")
    T(ax, (xa0 + xa1) / 2, ybot - 0.46, "대상 라벨 수", size=FS["small"], ha="center")
    T(ax, 0.30, (ybot + ytop) / 2, "원천 계수 Stefan 대비 오차 변화 (cm)", size=FS["small"], ha="center", rotation=90)
    series = [("P1", NAME["P1"], "-", 2.0), ("R1", "물리 잔차 결합", "-", 3.2), ("D0", NAME["D0"], (0, (3.0, 1.6)), 2.0)]
    for code, nm, ls, lw in series:
        dd = d[d.method == code].set_index("n")
        xs_ = [xpos[n] for n in ns]; ys_ = [Y_(dd.loc[n].delta) for n in ns]
        if code in ("P1", "R1"):
            ax.fill_between(xs_, [Y_(dd.loc[n].ci_lo) for n in ns], [Y_(dd.loc[n].ci_hi) for n in ns], color=COL[code], alpha=0.20,
                            linewidth=0, zorder=2)
        ax.plot(xs_, ys_, color=COL[code], lw=lw, ls=ls, zorder=5, solid_capstyle="round")
        ax.scatter(xs_, ys_, s=30, facecolor="white", edgecolor=COL[code], linewidths=1.3, zorder=6)
        yl = ys_[-1] + {"P1": 0.16, "R1": -0.18, "D0": 0.16}[code]
        T(ax, xpos[10] + 0.10, yl, nm, size=FS["small"], color=COL[code], ha="left", va="center")
    T(ax, 0.10, 4.60, "직접 ML +2.2 → +1.3 cm · 재보정 −2.45 cm(10개) · 잔차 결합은 재보정과 0.5 cm 안", size=FS["small"], color=AUX)
    # ---- (b)
    panel_letter(ax, 6.20, 4.92, "b"); T(ax, 6.46, 4.92, "지역 내 · 라벨 200개–전량 · 지역별", size=FS["body"])
    f7 = pd.read_csv(ROOT / "outputs/figures/paper/v3_restructure/Fig7_source_data.csv")
    a = f7[(f7.panel == "a") & f7.method.isin(["anchor_residual", "direct_ml"])]
    xb0, xb1 = 7.10, 11.75
    ylo2, yhi2 = -3.0, 4.0
    Y2 = lambda v: ybot + (np.clip(v, ylo2, yhi2) - ylo2) / (yhi2 - ylo2) * (ytop - ybot)
    ax.add_patch(Rectangle((xb0, Y2(-0.5)), xb1 - xb0, Y2(0.5) - Y2(-0.5), facecolor=BAND, edgecolor="none", zorder=1))
    hline(ax, xb0, xb1, Y2(0), color=COL["P1"], lw=2.0, z=2)
    T(ax, xb0 + 0.08, Y2(0.5) + 0.12, "재보정 Stefan = 0", size=FS["small"], color=AUX)
    vline(ax, xb0, ybot, ytop, color=INK, lw=1.0)
    for v in (-2, 0, 2, 4):
        hline(ax, xb0 - 0.06, xb0, Y2(v), color=INK, lw=1.0)
        T(ax, xb0 - 0.12, Y2(v), f"{v:d}".replace("-", "−"), size=FS["small"], ha="right")
    hline(ax, xb0, xb1, ybot, color=INK, lw=1.0)
    T(ax, 6.40, (ybot + ytop) / 2, "재보정 Stefan 대비 오차 변화 (cm)", size=FS["small"], ha="center", rotation=90)
    groups = [("Alaska", "알래스카", [200, 500, 1000, -1]), ("Lena", "레나델타", [200, 500, 1000, -1]), ("Canada", "캐나다", [200, -1])]
    gx = xb0 + 0.22
    gw = {4: 1.55, 2: 0.90}
    for reg, ko, ns2 in groups:
        w = gw[len(ns2)]
        xs_map = {n: gx + 0.12 + (w - 0.24) * k / (len(ns2) - 1) for k, n in enumerate(ns2)}
        for n in ns2:
            vline(ax, xs_map[n], ybot - 0.06, ybot, color=INK, lw=1.0)
            T(ax, xs_map[n], ybot - 0.20, "전량" if n == -1 else str(n), size=FS["small"], ha="center")
        T(ax, gx + w / 2, ybot - 0.46, ko, size=FS["small"], ha="center")
        for code, meth, ls in (("R1", "anchor_residual", "-"), ("D0", "direct_ml", (0, (3.0, 1.6)))):
            dd = a[(a.region == reg) & (a.method == meth)].set_index("n")
            xs_ = [xs_map[n] + (0.05 if code == "D0" else -0.05) for n in ns2]
            ys_ = [Y2(dd.loc[float(n)].delta) for n in ns2]
            for n, x_ in zip(ns2, xs_):
                r_ = dd.loc[float(n)]
                ax.plot([x_, x_], [Y2(r_.ci_lo), Y2(r_.ci_hi)], color=COL[code], lw=2.4, solid_capstyle="butt", zorder=4)
                ax.plot([x_ + 0.04, x_ + 0.04], [Y2(r_.delta_blockeq_lo), Y2(r_.delta_blockeq_hi)], color=COL[code], lw=0.9,
                        solid_capstyle="butt", zorder=4)
            ax.plot(xs_, ys_, color=COL[code], lw=1.8 if code == "D0" else 2.6, ls=ls, zorder=5)
            ax.scatter(xs_, ys_, s=26, facecolor="white", edgecolor=COL[code], linewidths=1.3, zorder=6)
        gx += w + 0.30
    T(ax, 6.20, 4.60, "알래스카: 잔차 결합 −0.54 cm(1000개) · 레나델타·캐나다: 차이 확인되지 않음", size=FS["small"], color=AUX)
    # 열쇠
    kx, ky = 7.60, 0.42
    key_line(ax, kx, ky, COL["P1"], w=0.26, lw=2.0); T(ax, kx + 0.34, ky, "재보정 Stefan", size=FS["small"])
    key_line(ax, kx + 1.70, ky, COL["R1"], w=0.26, lw=3.2); T(ax, kx + 2.04, ky, "물리 잔차 결합", size=FS["small"])
    key_line(ax, kx + 3.45, ky, COL["D0"], w=0.26, lw=2.0, ls=(0, (3.0, 1.6))); T(ax, kx + 3.79, ky, "직접 ML", size=FS["small"])
    T(ax, 0.30, 0.42, "음수 = 오차 감소 · 95% CI(a 셀 가중 띠, b 굵은 선 셀 가중·가는 선 블록 등가중) · 회색 띠 ±0.5 cm", size=FS["small"], color=AUX)
    src = dict(a="data/processed/paper_figs/fig2_pool_curves.csv edition E1_P4_n_le_10 (P1, R1, D0; n 0, 3, 10)",
               b="outputs/figures/paper/v3_restructure/Fig7_source_data.csv panel a (Alaska, Lena, Canada; anchor_residual, direct_ml)")
    (OUT / "R2_summary_curve_source_values.json").write_text(json.dumps(src, ensure_ascii=False, indent=1))
    return qa_and_save(fig, "R2_summary_curve")


PRODUCT_RESULTS = dict(   # 조정 지시(2026-10-06)의 값. 알래스카 라벨 셀 13,606개, 같은 셀(Yi–Kimball 은 10,606셀)
    r=[("우리 방법", 0.62, "R1"), ("Wei 2026", 0.55, "Wei"), ("CCI v5", 0.60, "CCI"), ("Yi–Kimball", 0.43, "YK"), ("Aalto 2018", -0.50, "Aalto")],
    rmse=[("우리 방법", 13.8, "R1"), ("Wei 2026", 17.2, "Wei"), ("Aalto 2018", 43.6, "Aalto"), ("Yi–Kimball", 61.0, "YK"), ("CCI v5", 74.9, "CCI")])


def _pcol(key):
    return COL["R1"] if key == "R1" else _C["products"][key]["hex"]


def hbars(ax, x0, x1, ytop, items, vmin, vmax, fmt, size=13.5, rh=0.42, bh=0.26, label_w=1.35, ticks=None, xlabel=None):
    """가로 막대(값을 막대 끝에). items = [(이름, 값, 색 키)]."""
    xa = x0 + label_w
    X_ = lambda v: xa + (v - vmin) / (vmax - vmin) * (x1 - xa)
    for i, (nm, v, key) in enumerate(items):
        y = ytop - i * rh
        c = _pcol(key) if key in ("R1", "Wei", "CCI", "Aalto", "YK") else COL.get(key, key)
        T(ax, xa - 0.10, y, nm, size=size, ha="right", color=COL["R1"] if key == "R1" else INK)
        x_a, x_b = sorted([X_(0), X_(v)])
        ax.add_patch(Rectangle((x_a, y - bh / 2), x_b - x_a, bh, facecolor=c, edgecolor="none", zorder=3))
        T(ax, (X_(v) if v >= 0 else X_(0)) + 0.07, y, fmt(v), size=size, ha="left", color=c)
    ybot = ytop - (len(items) - 1) * rh - rh * 0.75
    vline(ax, X_(0), ybot, ytop + rh * 0.6, color=ZERO, lw=1.0, z=2)
    if ticks is not None:
        hline(ax, xa, x1, ybot, color=INK, lw=1.0)
        for t_ in ticks:
            vline(ax, X_(t_), ybot - 0.06, ybot, color=INK, lw=1.0)
            T(ax, X_(t_), ybot - 0.20, (f"{t_:g}").replace("-", "−"), size=13, ha="center")
    if xlabel:
        T(ax, (xa + x1) / 2, ybot - 0.46, xlabel, size=13, ha="center", color=AUX)
    return X_


def p1_products_vs_ours():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    panel_letter(ax, 0.10, 4.92, "a"); T(ax, 0.36, 4.92, "상관계수 r (실측 대비, 위가 좋음)", size=15)
    hbars(ax, 0.30, 5.60, 4.30, PRODUCT_RESULTS["r"], -0.6, 0.8, lambda v: f"{v:.2f}".replace("-", "−"), ticks=[-0.5, 0, 0.5],
          xlabel="상관계수 r")
    vline(ax, 6.00, 0.95, 5.12, color=SEP, lw=0.8)
    panel_letter(ax, 6.20, 4.92, "b"); T(ax, 6.46, 4.92, "RMSE (cm, 작을수록 좋음)", size=15)
    hbars(ax, 6.40, 11.55, 4.30, PRODUCT_RESULTS["rmse"], 0.0, 80.0, lambda v: f"{v:.1f}", ticks=[0, 20, 40, 60, 80], xlabel="RMSE (cm)")
    T(ax, 0.30, 0.62, "같은 실측 셀(알래스카 13,606개). 우리 값은 해당 블록을 학습에서 뺀 교차검증 예측(물리 잔차 결합).", size=13, color=AUX)
    T(ax, 0.30, 0.30, "Yi–Kimball 은 10,606셀(같은 셀에서 우리 r 0.51, RMSE 14.5 cm).", size=13, color=AUX)
    (OUT / "P1_products_vs_ours_source_values.json").write_text(json.dumps(dict(
        values=PRODUCT_RESULTS, note="조정 지시 2026-10-06 의 값. 우리 RMSE 13.80 은 data/processed/xbatch/XK_support_scale/xk_support_scale_v1.csv (Alaska, models, 1km, r1 rmse_cw 13.7977) 와 일치. 제품 값의 원천 파일은 이 작업에서 대조하지 못했다"),
        ensure_ascii=False, indent=1))
    return qa_and_save(fig, "P1_products_vs_ours")


def c1_cv_selection():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    head(ax, 0.10, 4.98, "절차 예시: 라벨 40개인 새 지역")
    head(ax, 7.55, 4.98, "실제 선택 비율 (새 지역 시험)")
    vline(ax, 7.35, 0.25, 5.12, color=SEP, lw=0.8)
    # ① 라벨 40개 → 블록 5묶음(점 도식)
    rng = np.random.RandomState(7)
    gcols = [REG["Alaska"], REG["Canada"], REG["E Russia"], REG["Tibetan Plateau"], REG["Lena Delta"]]
    centers = [(0.55, 3.95), (1.35, 4.15), (0.75, 3.15), (1.55, 3.30), (1.10, 2.45)]
    for g, (cx, cy) in enumerate(centers):
        pts = rng.normal(0, 0.16, size=(8, 2))
        ax.scatter(cx + pts[:, 0], cy + pts[:, 1], s=34, color=gcols[g], linewidths=0, zorder=4)
    T(ax, 0.20, 1.86, "라벨 40개를 블록 단위", size=13.5)
    T(ax, 0.20, 1.58, "5묶음으로 나눔", size=13.5)
    arrow(ax, (2.10, 3.30), (2.55, 3.30), lw=1.3)
    # ② 후보 7개 교차검증 RMSE(예시 막대, 숫자 없음)
    cands = ["원천 계수 Stefan", "재보정 Stefan", "대상 지역 최소제곱 Stefan", "물리 잔차 결합(λ 0.25)", "물리 잔차 결합(λ 1.0)",
             "증강+잔차", "물리 유사라벨 증강"]
    ex = [0.86, 0.80, 0.83, 0.70, 0.60, 0.74, 0.90]      # 예시 길이(무차원), 수치 아님
    best = int(np.argmin(ex))
    x0b, wmax = 4.85, 1.75
    for i, (nm, v) in enumerate(zip(cands, ex)):
        y = 4.45 - i * 0.40
        T(ax, x0b - 0.10, y, nm, size=13, ha="right", color=COL["R1"] if i == best else INK)
        ax.add_patch(Rectangle((x0b, y - 0.12), wmax * v, 0.24, facecolor=COL["R1"] if i == best else "#C9CDD3", edgecolor="none", zorder=3))
    T(ax, x0b + wmax * ex[best] + 0.10, 4.45 - best * 0.40, "선택", size=13.5, color=COL["R1"])
    vline(ax, x0b, 4.45 - 6 * 0.40 - 0.22, 4.67, color=INK, lw=1.0)
    T(ax, 2.75, 1.72, "CV RMSE (예시, 블록 하나씩 빼고 채점)", size=13, color=AUX)
    arrow(ax, (x0b + 0.85, 1.48), (x0b + 0.85, 1.12), lw=1.3)
    T(ax, x0b - 2.30, 0.84, "선택된 방법을 라벨 40개 전부로 다시 학습 → 지도", size=13.5)
    # 오른쪽: 실제 선택 비율
    shares = [("원천 계수 Stefan", 30, "P0"), ("물리 잔차 결합 λ 1.0", 27, "R1"), ("대상 지역 최소제곱 Stefan", 17, "P1"),
              ("물리 잔차 결합 λ 0.25", 12, "R1"), ("물리 유사라벨 증강", 11, "D1"), ("재보정 Stefan", 3, "P1"), ("증강+잔차", 2, "R1")]
    xa, x1 = 9.95, 11.55
    X_ = lambda v: xa + v / 35.0 * (x1 - xa)
    for i, (nm, v, key) in enumerate(shares):
        y = 4.45 - i * 0.40
        T(ax, xa - 0.10, y, nm, size=13, ha="right")
        ax.add_patch(Rectangle((xa, y - 0.12), X_(v) - xa, 0.24, facecolor=COL[key], edgecolor="none", zorder=3))
        T(ax, X_(v) + 0.07, y, f"{v}%", size=13, ha="left")
    vline(ax, xa, 4.45 - 6 * 0.40 - 0.22, 4.67, color=INK, lw=1.0)
    T(ax, 7.55, 1.62, "선택 1,742회", size=13, color=AUX)
    T(ax, 7.55, 1.20, "지역 특성에 따라 물리식만으로 충분한 곳과", size=13.5)
    T(ax, 7.55, 0.90, "ML 보정이 필요한 곳을 절차가 스스로 가름", size=13.5)
    (OUT / "C1_cv_selection_source_values.json").write_text(json.dumps(dict(
        shares=shares, n_selections=1742, note="조정 지시 2026-10-06 의 값. 원천 파일은 이 작업에서 대조하지 못했다. 왼쪽 막대는 예시(수치 아님)"),
        ensure_ascii=False, indent=1))
    return qa_and_save(fig, "C1_cv_selection")


def r4_summary_simple():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    cols = [(0.15, 2.95), (3.15, 5.95), (6.15, 8.95), (9.15, 11.95)]
    heads = ["라벨 0개", "라벨 10개", "라벨 40–160개", "수백 개 이상 (알래스카)"]
    caps = ["2 cm 넘게 나빠진 대상 수", "원천 계수 Stefan 대비 오차 변화", "재보정 Stefan 대비 오차 변화", "재보정 Stefan 대비 오차 변화"]
    ylabs = ["대상 수 (아래가 좋음)", "오차 변화 (cm, 아래가 좋음)", "오차 변화 (cm, 아래가 좋음)", "오차 변화 (cm, 아래가 좋음)"]
    data = [[("직접 ML", 18, COL["D0"], "18 / 30"), ("물리 유사라벨 ML", 2, COL["R1"], "2 / 30")],
            [("직접 ML", 1.31, COL["D0"], "+1.3"), ("계수 재보정", -2.45, COL["R1"], "−2.45")],
            [("고정 레시피", -0.64, "#9E9E9E", "−0.64"), ("교차검증 방법 선택", -1.51, COL["R1"], "−1.51")],
            [("직접 ML", 1.27, COL["D0"], "+1.3"), ("물리 잔차 결합", -0.52, COL["R1"], "−0.52")]]
    ranges = [(0, 20), (-3, 2), (-2, 0.5), (-1, 2)]
    for k, ((x0, x1), hd, cap, yl, dd, (vmin, vmax)) in enumerate(zip(cols, heads, caps, ylabs, data, ranges)):
        head(ax, x0 + 0.05, 4.98, hd)
        T(ax, x0 + 0.05, 4.62, cap, size=13, color=AUX)
        ab, at = 1.35, 4.10
        Y_ = lambda v, vmin=vmin, vmax=vmax: ab + (v - vmin) / (vmax - vmin) * (at - ab)
        xa = x0 + 0.75
        vline(ax, xa, ab, at, color=INK, lw=1.0)
        hline(ax, xa, x1 - 0.15, Y_(0), color=ZERO, lw=1.0)
        T(ax, x0 + 0.20, (ab + at) / 2, yl, size=13, ha="center", rotation=90)
        for j, (nm, v, c, lab) in enumerate(dd):
            bx = xa + 0.25 + j * 0.88
            y0_, y1_ = sorted([Y_(0), Y_(v)])
            ax.add_patch(Rectangle((bx, y0_), 0.62, y1_ - y0_, facecolor=c, edgecolor="none", zorder=3))
            T(ax, bx + 0.31, (Y_(v) + 0.16) if v >= 0 else (Y_(v) - 0.16), lab, size=14, ha="center", color=c)
            for i_, ln in enumerate(nm.split(" ") if len(nm) > 8 else [nm]):
                pass
            T(ax, bx + 0.31, ab - 0.10, nm if len(nm) <= 6 else nm.replace(" ", "\n", 1), size=13, ha="center", va="top", color=c,
              linespacing=1.15)
    T(ax, 0.20, 0.30, "주황 = 권고 방법 · 0개: 전이 대상 30개(사후 서술) · 10개: 주 4지역 평균 · 40–160개: 레나델타·캐나다 평균(라벨 160개) · 수백 개 이상: 알래스카 라벨 전량",
      size=12.5, color=AUX)
    (OUT / "R4_summary_simple_source_values.json").write_text(json.dumps(dict(
        n0="RESEARCH_OVERVIEW 9절 위험표(직접 ML 18/30, 물리 유사라벨 증강 2/30, 사후 서술)",
        n10="fig2_pool_curves.csv E1: D0 n10 +1.31, P1 n10 −2.45",
        n160="lg_tests.csv L4 MEAN4 n160 λ0.25 −0.64 (고정 레시피); Fig7_source_data panel e A1-A3 n160 −1.51 (교차검증 방법 선택)",
        all_alaska="Fig7_source_data panel a Alaska n all: direct_ml +1.27, anchor_residual −0.52"), ensure_ascii=False, indent=1))
    return qa_and_save(fig, "R4_summary_simple")


def s1_strengths():
    """(a) 안정성 (b) 정확도 (c) 검증 설계 (d) 기존 제품 대비. 막대 = 95% 신뢰구간."""
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    X0 = [0.10, 3.10, 6.10, 9.10]
    for xv in (2.95, 5.95, 8.95):
        vline(ax, xv, 0.25, 5.12, color=SEP, lw=0.8)
    # (a)
    panel_letter(ax, X0[0], 4.92, "a"); T(ax, X0[0] + 0.26, 4.92, "안정성", size=FS["body"])
    T(ax, X0[0], 4.56, "라벨 0개, 2 cm 넘게 나빠진", size=FS["small"], color=AUX)
    T(ax, X0[0], 4.30, "대상 수 (30개 중)", size=FS["small"], color=AUX)
    bx0, bx1 = 0.20, 2.40
    for k, (code, nm, val) in enumerate([("D0", "직접 ML", 18), ("D1", "물리 유사라벨 증강", 2)]):
        y = 3.55 - k * 0.95
        T(ax, bx0, y + 0.34, nm, size=FS["body"], color=COL[code])
        ax.add_patch(Rectangle((bx0, y - 0.16), (bx1 - bx0) * val / 30, 0.32, facecolor=COL[code], edgecolor="none", zorder=3))
        T(ax, bx0 + (bx1 - bx0) * val / 30 + 0.08, y, f"{val}/30", size=FS["body"])
    T(ax, X0[0], 1.10, "전이 대상 30개 · 사후 서술", size=FS["small"], color=AUX)
    # (b)
    panel_letter(ax, X0[1], 4.92, "b"); T(ax, X0[1] + 0.26, 4.92, "정확도", size=FS["body"])
    T(ax, X0[1], 4.56, "권고 방법 − 재보정 Stefan (cm)", size=FS["small"], color=AUX)
    f7 = pd.read_csv(ROOT / "outputs/figures/paper/v3_restructure/Fig7_source_data.csv")
    xc = f7[(f7.panel == "e") & (f7.element == "pooled") & (f7.contrast == "A1-A3")].set_index("n")
    wr = f7[(f7.panel == "a") & (f7.region == "Alaska") & (f7.method == "anchor_residual") & (f7.n == 1000)].iloc[0]
    pts = [("방법 선택\n40개", COL["W"], xc.loc[40.0]), ("방법 선택\n160개", COL["W"], xc.loc[160.0]), ("잔차 결합\n알래스카", COL["R1"], wr)]
    px0, px1, pyb, pyt = 3.55, 5.80, 1.95, 4.10
    ylo, yhi = -2.4, 0.8
    Y_ = lambda v: pyb + (v - ylo) / (yhi - ylo) * (pyt - pyb)
    ax.add_patch(Rectangle((px0, Y_(-0.5)), px1 - px0, Y_(0.5) - Y_(-0.5), facecolor=BAND, edgecolor="none", zorder=1))
    hline(ax, px0, px1, Y_(0), color=COL["P1"], lw=2.0, z=2)
    vline(ax, px0, pyb, pyt, color=INK, lw=1.0)
    for v in (-2, -1, 0):
        hline(ax, px0 - 0.06, px0, Y_(v), color=INK, lw=1.0)
        T(ax, px0 - 0.12, Y_(v), f"{v:d}".replace("-", "−"), size=FS["small"], ha="right")
    hline(ax, px0, px1, pyb, color=INK, lw=1.0)
    for k, (nm, c, r_) in enumerate(pts):
        x_ = px0 + 0.40 + k * 0.72
        d, lo, hi = float(r_["delta"]), float(r_["ci_lo"]), float(r_["ci_hi"])
        ax.plot([x_, x_], [Y_(lo), Y_(hi)], color=c, lw=3.0, solid_capstyle="butt", zorder=4)
        ax.scatter([x_], [Y_(d)], s=60, facecolor=c, edgecolor=c, zorder=5)
        T(ax, x_, Y_(lo) - 0.16, f"{d:+.2f}".replace("-", "−"), size=FS["small"], color=c, ha="center")
        for j, ln in enumerate(nm.split("\n")):
            T(ax, x_, pyb - 0.22 - j * 0.25, ln, size=FS["small"], ha="center", color=INK if j == 0 else AUX)
    T(ax, X0[1], 1.10, "막대 = 95% 신뢰구간 · 띠 ±0.5 cm", size=FS["small"], color=AUX)
    T(ax, X0[1], 0.82, "가로선 = 재보정 Stefan", size=FS["small"], color=AUX)
    # (c)
    panel_letter(ax, X0[2], 4.92, "c"); T(ax, X0[2] + 0.26, 4.92, "검증 설계", size=FS["body"])
    T(ax, X0[2], 4.56, "검증 분리가 클수록 직접 ML 만 악화", size=FS["small"], color=AUX)
    e = pd.read_csv(ROOT / "outputs/figures/paper/v3_restructure/Fig1_source_data.csv")
    e = e[(e.panel == "e") & (e.region == "MEAN3[Alaska,Lena,Canada]") & (e.element == "rmse_mean_equal_weight")]
    order = ["Random", "Site", "Block", "kNNDM", "Region holdout"]
    dml = [float(e[e.kind == f"D0w|{k}"].value.iloc[0]) for k in order]
    stf = [float(e[e.kind == f"PSw|{k}"].value.iloc[0]) for k in order]
    cx0, cx1, cyb, cyt = 6.65, 8.75, 1.95, 4.10
    Y3 = lambda v: cyb + (v - 12.0) / (31.0 - 12.0) * (cyt - cyb)
    xs_ = [cx0 + 0.15 + (cx1 - cx0 - 0.3) * k / 4 for k in range(5)]
    vline(ax, cx0, cyb, cyt, color=INK, lw=1.0)
    for v in (15, 20, 25, 30):
        hline(ax, cx0 - 0.06, cx0, Y3(v), color=INK, lw=1.0)
        T(ax, cx0 - 0.12, Y3(v), str(v), size=FS["small"], ha="right")
    hline(ax, cx0, cx1, cyb, color=INK, lw=1.0)
    T(ax, xs_[0], cyb - 0.22, "무작위", size=FS["small"], ha="center")
    T(ax, xs_[-1], cyb - 0.22, "지역 홀드아웃", size=FS["small"], ha="right")
    ax.plot(xs_, [Y3(v) for v in dml], color=COL["D0"], lw=2.2, ls=(0, (3.0, 1.6)), zorder=5)
    ax.scatter(xs_, [Y3(v) for v in dml], s=26, facecolor=COL["D0"], edgecolor=COL["D0"], zorder=6)
    ax.plot(xs_, [Y3(v) for v in stf], color=COL["P1"], lw=2.2, zorder=5)
    ax.scatter(xs_, [Y3(v) for v in stf], s=26, facecolor=COL["P1"], edgecolor=COL["P1"], zorder=6)
    T(ax, xs_[-1], Y3(dml[-1]) + 0.20, f"직접 ML {dml[-1]:.1f}", size=FS["small"], color=COL["D0"], ha="right")
    T(ax, xs_[0], Y3(dml[0]) - 0.22, f"{dml[0]:.1f}", size=FS["small"], color=COL["D0"], ha="left")
    T(ax, xs_[0], Y3(24.0), f"Stefan {min(stf):.1f}–{max(stf):.1f}", size=FS["small"], color=COL["P1"], ha="left")
    T(ax, X0[2], 1.10, "RMSE (cm) · 세 지역 평균", size=FS["small"], color=AUX)
    # (d)
    panel_letter(ax, X0[3], 4.92, "d"); T(ax, X0[3] + 0.26, 4.92, "기존 제품 대비", size=FS["body"])
    T(ax, X0[3], 4.56, "알래스카 실측 13,606셀과의 상관 r", size=FS["small"], color=AUX)
    hbars(ax, X0[3] + 0.05, 11.75, 3.95, PRODUCT_RESULTS["r"], -0.6, 0.8, lambda v: f"{v:.2f}".replace("-", "−"), size=FS["small"],
          rh=0.44, bh=0.24, label_w=1.10)
    T(ax, X0[3], 1.10, "우리 0.62 · 가장 좋은 제품 0.60", size=FS["small"], color=AUX)
    T(ax, X0[3], 0.82, "(위가 좋음)", size=FS["small"], color=AUX)
    src = dict(a="RESEARCH_OVERVIEW 9절 위험표(사후 서술)", b="Fig7_source_data panel e A1-A3 n40·160; panel a Alaska anchor_residual n1000",
               c="Fig1_source_data panel e MEAN3", d="조정 지시 2026-10-06 의 제품 상관값(P1 과 같음)")
    (OUT / "S1_strengths_source_values.json").write_text(json.dumps(src, ensure_ascii=False, indent=1))
    return qa_and_save(fig, "S1_strengths")


def r3_alaska_inregion():
    """알래스카 지역 내 라벨 200·500·1000·전량: 재보정 Stefan 대비 오차 변화(물리 잔차 결합, 직접 ML). 글자 14 pt 이상."""
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    f7 = pd.read_csv(ROOT / "outputs/figures/paper/v3_restructure/Fig7_source_data.csv")
    a = f7[(f7.panel == "a") & (f7.region == "Alaska") & f7.method.isin(["anchor_residual", "direct_ml"])]
    ns = [200, 500, 1000, -1]
    x0, x1, ybot, ytop = 1.55, 8.70, 1.05, 4.85
    xpos = {n: x0 + 0.60 + (x1 - x0 - 1.20) * k / 3 for k, n in enumerate(ns)}
    ylo, yhi = -1.5, 4.0
    Y_ = lambda v: ybot + (v - ylo) / (yhi - ylo) * (ytop - ybot)
    ax.add_patch(Rectangle((x0, Y_(-0.5)), x1 - x0, Y_(0.5) - Y_(-0.5), facecolor=BAND, edgecolor="none", zorder=1))
    hline(ax, x0, x1, Y_(0), color=COL["P1"], lw=2.4, z=2)
    vline(ax, x0, ybot, ytop, color=INK, lw=1.2)
    for v in (-1, 0, 1, 2, 3, 4):
        hline(ax, x0 - 0.07, x0, Y_(v), color=INK, lw=1.2)
        T(ax, x0 - 0.14, Y_(v), f"{v:+d}".replace("-", "−").replace("+0", "0"), size=15, ha="right")
    hline(ax, x0, x1, ybot, color=INK, lw=1.2)
    for n in ns:
        vline(ax, xpos[n], ybot - 0.07, ybot, color=INK, lw=1.2)
        T(ax, xpos[n], ybot - 0.26, "전량" if n == -1 else f"{n}", size=15, ha="center")
    T(ax, (x0 + x1) / 2, ybot - 0.62, "알래스카 지역 내 라벨 수", size=15, ha="center")
    T(ax, 0.42, (ybot + ytop) / 2, "재보정 Stefan 대비 오차 변화\n(cm, 음수 = 오차 감소)", size=14, ha="center", rotation=90, linespacing=1.3)
    for code, meth, ls, lw, off in (("R1", "anchor_residual", "-", 3.4, -0.10), ("D0", "direct_ml", (0, (3.2, 1.8)), 2.4, 0.10)):
        dd = a[a.method == meth].set_index("n")
        xs_ = [xpos[n] + off for n in ns]
        ys_ = [Y_(float(dd.loc[float(n)].delta)) for n in ns]
        for n, x_ in zip(ns, xs_):
            r_ = dd.loc[float(n)]
            ax.plot([x_, x_], [Y_(r_.ci_lo), Y_(r_.ci_hi)], color=COL[code], lw=3.2, solid_capstyle="butt", zorder=4)
        ax.plot(xs_, ys_, color=COL[code], lw=lw, ls=ls, zorder=5, solid_capstyle="round")
        ax.scatter(xs_, ys_, s=70, facecolor="white" if code == "D0" else COL[code], edgecolor=COL[code], linewidths=1.8, zorder=6)
    rr = a[a.method == "anchor_residual"].set_index("n")
    dm = a[a.method == "direct_ml"].set_index("n")
    T(ax, (xpos[200] + xpos[500]) / 2 + 0.10, Y_(-1.0), "물리 잔차 결합", size=16, color=COL["R1"], ha="center")
    T(ax, xpos[200] + 0.20, Y_(float(dm.loc[200.0].delta)) + 0.25, "직접 ML", size=16, color=COL["D0"], ha="left")
    T(ax, x0 + 0.08, Y_(0.5) + 0.18, "재보정 Stefan = 0", size=15, color=COL["P1"], ha="left")
    # 오른쪽 사실 목록
    xr = 9.15
    T(ax, xr, 4.70, "물리 잔차 결합", size=15, color=COL["R1"])
    T(ax, xr, 4.38, f"1000개: {float(rr.loc[1000.0].delta):+.2f} cm".replace("-", "−"), size=14)
    T(ax, xr, 4.10, f"RMSE {float(rr.loc[1000.0].rmse_recal):.2f} → {float(rr.loc[1000.0].rmse_method):.2f} cm", size=14)
    T(ax, xr, 3.82, "500개 이상 두 가중 모두 감소", size=14, color=AUX)
    T(ax, xr, 3.30, "직접 ML", size=15, color=COL["D0"])
    T(ax, xr, 2.98, "+1.3 ~ +3.5 cm", size=14)
    T(ax, xr, 2.70, "가중 방식에 따라 달라 미결정", size=14, color=AUX)
    ax.plot([xr, xr], [2.02, 2.26], color=INK, lw=3.2, solid_capstyle="butt"); T(ax, xr + 0.14, 2.14, "막대 = 95% 신뢰구간", size=14, color=AUX)
    ax.add_patch(Rectangle((xr - 0.05, 1.67), 0.22, 0.16, facecolor=BAND, edgecolor="none")); T(ax, xr + 0.24, 1.75, "±0.5 cm 동등 한계", size=14, color=AUX)
    T(ax, xr, 1.28, "분할 25회 · 블록 재표집", size=14, color=AUX)
    (OUT / "R3_alaska_inregion_source_values.json").write_text(json.dumps(dict(
        source="outputs/figures/paper/v3_restructure/Fig7_source_data.csv panel a, region Alaska, methods anchor_residual and direct_ml, n 200/500/1000/all",
        values={f"{m}|{int(n)}": dict(delta=float(r.delta), ci=[float(r.ci_lo), float(r.ci_hi)], ci_beq=[float(r.delta_blockeq_lo), float(r.delta_blockeq_hi)])
                for m, g in a.groupby("method") for n, r in g.set_index("n").iterrows()}), ensure_ascii=False, indent=1))
    return qa_and_save(fig, "R3_alaska_inregion")


def figS14_panel(idx=1):
    """FigS14(레나델타, 라벨 0개)의 패널 b(90% 구간 폭) 지도 부분. 테두리 안쪽을 고정 좌표로 자른다(4015 × 2031 px 판)."""
    from PIL import Image
    a = np.asarray(Image.open(ROOT / "outputs/figures/paper/v3_restructure/si/FigS14.png").convert("RGB"))
    dark = (a.astype(int).sum(2) < 120)
    # 패널 b 의 테두리: 가로 1450–2800, 세로 150–1450 범위에서 어두운 행·열이 가장 긴 곳
    sub = dark[150:1450, 1450:2800]
    rows = np.where(sub.mean(1) > 0.6)[0]
    cols = np.where(sub.mean(0) > 0.6)[0]
    y0, y1 = 150 + int(rows.min()), 150 + int(rows.max())
    x0, x1 = 1450 + int(cols.min()), 1450 + int(cols.max())
    return a[y0 + 5: y1 - 4, x0 + 5: x1 - 4]


def g1_workflow_guide():
    """새 지역에서 워크플로를 쓰는 순서(4단계, 라벨 수 축). 글자 13 pt 이상."""
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    cw = 2.70
    xs = [0.15 + k * 3.0 for k in range(4)]
    heads = ["① 라벨 0개", "② 다음 관측 위치 선택", "③ 라벨 약 10개", "④ 라벨 40개 이상"]
    yaxis = 4.60
    hline(ax, xs[0], xs[3] + cw, yaxis, color=INK, lw=1.0)
    for k in range(4):
        cx = xs[k] + cw / 2
        vline(ax, cx, yaxis - 0.07, yaxis + 0.07, color=INK, lw=1.0)
        T(ax, cx, yaxis + 0.30, heads[k], size=15, ha="center", color=HEAD, weight="bold")
    # 축소 그림(실자료)
    tb, th_ = 2.62, 1.72
    thumbs = [figS14_panel(1), None, None, alaska_png("c_single")]
    from PIL import Image
    thumbs[1] = np.asarray(Image.open(ROOT / "deck/assets/paper_report/slide_panels/Fig5_c_slide.png").convert("RGB"))
    boxes = []
    for k in range(4):
        if k == 2:
            w_, h_ = 2.05, th_
            x_ = xs[k] + (cw - w_) / 2 + 0.18
            stefan_scatter(fig, x_, tb + 0.12, w_ - 0.18, h_ - 0.12, size=13, labelpad=3)
            boxes.append((x_ - 0.30, x_ + w_ - 0.18))
            continue
        arr = thumbs[k]
        w_, h_ = fit_box(arr.shape, 2.05, th_)
        x_ = xs[k] + (cw - w_) / 2
        img_thumb(fig, x_, tb + (th_ - h_) / 2, w_, h_, arr)
        boxes.append((x_, x_ + w_))
    ymid = tb + th_ / 2
    for k in range(3):
        arrow(ax, (boxes[k][1] + 0.12, ymid), (boxes[k + 1][0] - 0.12, ymid), lw=1.4, hl=0.10, hw=0.08)
    caps = ["90% 구간 폭 (레나델타, 라벨 0개)", "공변량 분산 배치 예 (캐나다)", "원천 → 재보정 계수 (러시아 서부)", "최종 ALT 지도 (알래스카)"]
    for k in range(4):
        T(ax, xs[k] + cw / 2, tb - 0.24, caps[k], size=13, ha="center", color=AUX)
    acts = [["Stefan·유사라벨 ML 로 첫 지도", "구간 폭·학습 범위 밖 확인"],
            ["구간 넓고 범위 밖인 곳 우선", "블록·공변량 공간에 고르게"],
            ["Stefan 계수 재보정 (κ = 10)", "편향 진단 → 지도 갱신"],
            ["후보 7개 교차검증으로 방법 선택", "최종 지도 + 90% 구간"]]
    for k in range(4):
        for j, t_ in enumerate(acts[k]):
            T(ax, xs[k] + 0.05, 1.82 - j * 0.34, t_, size=13.5, color=INK)
    # 반복 고리(가는 선 하나, 화살촉 하나)
    yl = 0.82
    x2 = xs[1] + cw / 2
    x4 = xs[3] + cw / 2
    vline(ax, x4, yl, 1.18, color=AUX, lw=1.0)
    hline(ax, x2, x4, yl, color=AUX, lw=1.0)
    arrow(ax, (x2, yl), (x2, 1.18), color=AUX, lw=1.0, hl=0.07, hw=0.06)
    T(ax, (x2 + x4) / 2, yl - 0.24, "관측이 늘 때마다 ②–④ 반복", size=13.5, ha="center", color=AUX)
    (OUT / "G1_workflow_guide_source_values.json").write_text(json.dumps(dict(
        thumb1="outputs/figures/paper/v3_restructure/si/FigS14.png panel b (Lena Delta, 0 labels, width of 90% interval)",
        thumb2="deck/assets/paper_report/slide_panels/Fig5_c_slide.png (covariate spread, Canada)",
        thumb3="fig3.load_concept (W Russia 31 cells, E0 1.59, E_n 2.11)",
        thumb4="deck/assets/paper_report/slide_panels/Alaska_ALT_map_v3_c_slide.png",
        rules="methods.tex selection rule (7 candidates), recalibration κ = 10, placement"), ensure_ascii=False, indent=1))
    return qa_and_save(fig, "G1_workflow_guide")


# ================================================================ 실행
FIGS = {"M4": m4_workflow, "M6": m6_models, "M6b": m6b_models_detail, "M5": m5_augmentation, "M7": m7_evaluation, "M1": m1_problem, "M3": m3_data, "M2": m2_gap, "M8": m8_label_workflow, "M9": m9_deepsets, "M10": m10_stefan_physics, "B1": b1_background, "B2": b2_prior_work, "R0": r0_summary_ladder, "E1": e1_evidence_scope, "R1": r1_results_by_range, "R2": r2_summary_curve, "S1": s1_strengths, "R3": r3_alaska_inregion, "G1": g1_workflow_guide, "P1": p1_products_vs_ours, "C1": c1_cv_selection, "R4": r4_summary_simple}


def contact_sheet():
    from PIL import Image
    files = sorted(OUT.glob("*_slide.png"))
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
