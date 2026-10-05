"""논문 v3 · ALT 1 km 지도(알래스카, 레나 델타). 개정 2(2026-10-05 검토 반영). 파일 이름과 출력 폴더는 개정 1 과 같다.

구성(두 지역 같다)
  a 재보정 Stefan ALT(P1), b 잔차 ML 보정량(R1 − P1, broc 0 중심 대칭), c 잔차 ML ALT(R1, 최종), d 학습 범위(범위 밖 공변량 수, 라벨 셀).
  a 와 c 는 같은 색표(oslo_r 0.12–0.90)와 같은 범위(두 지도 표시 셀 합동 1–99 백분위를 5 cm 단위로 바깥 반올림)를 쓴다.
  b 의 범위는 |보정량| 99 백분위를 5 cm 단위로 올린 값이다. 90 % 구간은 상수 폭이라 지도로 그리지 않고 설명문에 적는다.
  ERA5-Land 격자 확대는 XL_display_resolution 으로 옮겼다.
그리기
  셀은 원 격자(ky, kx; 위도 0.009°, 경도 0.009°/cos φ)의 사각형 그대로 QuadMesh 하나로 그린다. 행마다 경도 간격이 달라 행을 쌓고
  행 사이 높이 0 인 줄은 투명으로 둔다(build_mesh). 재표집하지 않으므로 육각형 모자이크가 생기지 않는다.
  QuadMesh 와 d 의 해칭 층만 래스터(600 dpi)이고 글자·선·해안선·경위선은 벡터다. Natural Earth 10m 육지·해안선은 지역 상자로 잘라
  단순화한다(알래스카 0.01°, 레나 0.002°). 축척 막대와 위치 삽도는 자료 셀과 해안선이 없는 모서리에 둔다(pick_corner).
논문판  170 mm, 2 × 2, 패널 문자만, 각 열 오른쪽 세로 색 막대(a·c 공유, b)와 d 의 열쇠, 위도 왼쪽 열·경도 아래 행만.
슬라이드판  12.6 in 폭, 1 × 4, 패널 아래 색 막대, 좌표는 a 에만, Pretendard.
입력  알래스카 data/processed/map_alaska/alaska_pred_v1.csv.gz(map_alaska_v1.py), 레나 data/processed/map_lena/lena_pred_v1.csv.gz(h49),
      lena_final_extras_v1.csv.gz(map_lena_final_extras_v1.py), 라벨 위치 polar.m1_core.load_base(v3 F4_direct)
산출  outputs/figures/paper/v3_restructure/maps/{Alaska_ALT_map_v3, Lena_ALT_map_v3}.{pdf,png}, *_source_values.json,
      deck/assets/paper_report/maps/{Alaska_ALT_map_v3, Lena_ALT_map_v3}_slide.png
실행  nice -n 10 python3 scripts/4_visualization/paper_v3/maps_alt_v3.py [--region alaska|lena|both] [--medium paper|slide|both]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse                                                                                         # noqa: E402
import json                                                                                             # noqa: E402
import math                                                                                             # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402
import matplotlib                                                                                       # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                                         # noqa: E402
import matplotlib.ticker as mticker                                                                     # noqa: E402
import matplotlib.patheffects as pe                                                                     # noqa: E402
from matplotlib.cm import ScalarMappable                                                                # noqa: E402
from matplotlib.collections import QuadMesh                                                             # noqa: E402
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm, to_rgba                          # noqa: E402
from matplotlib.lines import Line2D                                                                     # noqa: E402
from matplotlib.patches import Patch                                                                    # noqa: E402
import cartopy                                                                                          # noqa: E402
import cartopy.crs as ccrs                                                                              # noqa: E402
import cartopy.io.shapereader as shpreader                                                              # noqa: E402
import cmcrameri.cm as cmc                                                                              # noqa: E402
import shapely                                                                                          # noqa: E402
from pyproj import Geod                                                                                 # noqa: E402
from scipy.spatial import cKDTree                                                                       # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(HERE), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import style as S                                                                                       # noqa: E402

OUT_PAPER = ROOT / "outputs" / "figures" / "paper" / "v3_restructure" / "maps"
OUT_SLIDE = ROOT / "deck" / "assets" / "paper_report" / "maps"
SLIDE_W_MM = 12.0 * 25.4               # 덱 본문 폭 12.0 in(2026-10-05 정정), 높이 5.2 in 이하
SLIDE_H_MAX_MM = 5.2 * 25.4
PROC = ROOT / "data" / "processed"
PC = ccrs.PlateCarree()
GEOD = Geod(ellps="WGS84")
CELL_DEG = 0.009
NODATA = S.BASEMAP["nodata"]           # #C9CDD3
# 육지·해안·경위선 색은 v4 토큰 color.map_base(매체별)이며 그릴 때 S.BASEMAP 에서 읽는다(use_medium 이 바꾼다)
MAPPED = "#e6e6e6"                     # 학습 범위 패널의 표시 셀 바탕
WATER = "#ffffff"
GRAY_TXT = "#6b6b6b"
HATCH_CAT = {1: "//", 2: "////", 3: "xxx"}           # 12·24 줄/인치 사선, 18 줄/인치 교차
MINUS = "−"


def lut(cm, lo, hi, n=250):
    """끝을 자른 색표(250 단계: PDF 래스터가 256색 이하 색인 영상이 되도록)."""
    c = ListedColormap(cm(np.linspace(lo, hi, n)))
    c.set_bad((0.0, 0.0, 0.0, 0.0))
    return c


CM_ALT = lut(cmc.oslo_r, 0.12, 0.90)
CM_DIFF = lut(cmc.broc, 0.0, 1.0)

REG = {
    "alaska": dict(name="Alaska", box=dict(lat0=59.0, lat1=71.5, lon0=-168.0, lon1=-141.0), pad_m=20000.0,
                   xlocs=[-170, -160, -150, -140], ylocs=[60, 65, 70], scale_km=200, simplify=0.01, zoom_half_m=25000.0, zoom_min_cover=0.8),
    "lena": dict(name="Lena Delta", box=dict(lat0=71.5, lat1=73.6, lon0=123.3, lon1=130.1), pad_m=3000.0,
                 xlocs=[124, 126, 128, 130], ylocs=[72, 73], scale_km=50, simplify=0.002),
}


def projection(key):
    if key == "alaska":
        return ccrs.epsg(3338)                                         # NAD83 / Alaska Albers
    return ccrs.NorthPolarStereo(central_longitude=126.7, true_scale_latitude=70.0)


def register_pretendard():
    """슬라이드판 글꼴(지침 5.3). ~/.fonts 의 Pretendard 를 matplotlib 에 등록한다."""
    import glob
    from matplotlib import font_manager as fm
    for f in sorted(glob.glob(os.path.expanduser("~/.fonts/Pretendard-*.otf"))):
        fm.fontManager.addfont(f)


def use_medium(medium):
    """v4 토큰: 두 매체 모두 FreeSans Regular(한글은 Pretendard Regular 로 대체), 선은 lines.paper·lines.slide."""
    S.use_v3(medium)
    S.mathtext_liberation()


class Med:
    """매체별 크기(pt, mm)."""

    def __init__(self, medium):
        self.paper = medium == "paper"
        F, Lp, Ls = S.TOK["font"], S.TOK["lines"]["paper"], S.TOK["lines"]["slide"]
        self.fs = F["paper"]["text_pt"] if self.paper else F["slide"]["tick_pt"]                # 눈금, 열쇠, 축척(v4: 7 / 12 pt)
        self.fs_head = F["paper"]["text_pt"] if self.paper else F["slide"]["direct_label_pt"]   # 패널 제목(7 / 13 pt)
        self.fs_lab = F["paper"]["text_pt"] if self.paper else F["slide"]["axis_label_pt"]     # 색 막대 라벨(7 / 14 pt)
        self.fs_small = F["paper"]["text_pt"] if self.paper else F["slide"]["tick_pt"]         # 회색 부가 글(7 / 12 pt)
        self.fs_letter = F["paper"]["panel_letter_pt"] if self.paper else F["slide"]["panel_letter_pt"]   # 패널 문자 굵게(8 / 14 pt)
        self.lw = (Lp if self.paper else Ls)["axis_pt"]                  # 틀, 색 막대 테두리, 눈금(0.6 / 1.0 pt)
        self.lw_coast = (Lp if self.paper else Ls)["map_coast_pt"]       # 해안선(0.5 / 0.8 pt)
        self.lw_grat = (Lp if self.paper else Ls)["graticule_pt"]        # 경위선(0.4 / 0.6 pt)
        self.lw_scale = (Lp if self.paper else Ls)["data_pt"]            # 축척 막대(1.2 / 2.2 pt)
        self.dot = 0.9 if self.paper else 5.0           # 라벨 점 면적(pt²)
        self.cbar = 2.5 if self.paper else 4.0          # 색 막대 두께(mm)
        self.tick_len = (Lp if self.paper else Ls)["tick_len_pt"]


# ================================================================ 자료
def label_cells(macro: str) -> pd.DataFrame:
    from polar.m1_core import load_base
    df = load_base(PROC)
    d = df[df.macro == macro]
    return pd.DataFrame(dict(lat=d.lat.values, lon=d.lon.values, block=d.block.values))


def load_region(key: str) -> dict:
    if key == "alaska":
        p = pd.read_csv(PROC / "map_alaska" / "alaska_pred_v1.csv.gz", dtype={"cell_id": str})
        meta = json.loads((PROC / "map_alaska" / "alaska_pred_v1_meta.json").read_text())
        d = dict(p1=p.pred_p1.values.astype(float), r1=p.pred_r1.values.astype(float), corr=p.corr_r1_minus_p1.values.astype(float),
                 cat=p.extrap_cat.values.astype(float), n_out=p.n_extrap_out.values.astype(float), interval_q=float(meta["interval"]["q"]),
                 coef=dict(E_n=meta["model"]["E_n"], lam=meta["model"]["lam"]), meta=meta, labels=label_cells("Alaska"))
        land_like = np.ones(len(p), bool)                               # 알래스카 격자는 모두 Natural Earth 육지
    else:
        p = pd.read_csv(PROC / "map_lena" / "lena_pred_v1.csv.gz", dtype={"cell_id": str})
        ex = pd.read_csv(PROC / "map_lena" / "lena_final_extras_v1.csv.gz", dtype={"cell_id": str})
        assert (ex.cell_id.values == p.cell_id.values).all()
        meta = json.loads((PROC / "map_lena" / "lena_pred_v1_meta.json").read_text())
        xm = json.loads((PROC / "map_lena" / "lena_final_extras_v1_meta.json").read_text())
        d = dict(p1=p.m_P1_all.values.astype(float), r1=p.pred_c.values.astype(float), corr=(p.m_R1_all - p.m_P1_all).values.astype(float),
                 cat=ex.extrap_cat_final.values.astype(float), n_out=ex.n_extrap_out_final.values.astype(float), interval_q=float(xm["interval"]["q"]),
                 coef=dict(E_n=meta["coefs"]["R1_all"]["E_n"], lam=meta["coefs"]["R1_all"]["lam"]), meta=meta, extras_meta=xm, labels=label_cells("Lena"))
        land_like = p.land.values == 1
    reason = p.gray_reason.fillna("").astype(str).values
    gray = p.gray.values == 1
    d.update(key=key, cells=p, proj=projection(key), **REG[key])
    d["shown"] = (~gray) & np.isfinite(d["r1"]) & np.isfinite(d["p1"])
    d["water"] = gray & (reason == "water")
    d["nodata"] = gray & ~np.isin(reason, ["water", "dem_tile_absent", "pfr_missing"]) & land_like
    return d


def box_extent(proj, box, pad_m):
    lon = np.r_[np.linspace(box["lon0"], box["lon1"], 60), np.full(60, box["lon1"]), np.linspace(box["lon1"], box["lon0"], 60), np.full(60, box["lon0"])]
    lat = np.r_[np.full(60, box["lat0"]), np.linspace(box["lat0"], box["lat1"], 60), np.full(60, box["lat1"]), np.linspace(box["lat1"], box["lat0"], 60)]
    xy = proj.transform_points(PC, lon, lat)
    return [xy[:, 0].min() - pad_m, xy[:, 0].max() + pad_m, xy[:, 1].min() - pad_m, xy[:, 1].max() + pad_m]


def square_extent(ext):
    """짧은 변을 늘려 정사각형 범위로 만든다(같은 크기 패널용)."""
    x0, x1, y0, y1 = ext
    w, h = x1 - x0, y1 - y0
    if w > h:
        c = (y0 + y1) / 2
        return [x0, x1, c - w / 2, c + w / 2]
    c = (x0 + x1) / 2
    return [c - h / 2, c + h / 2, y0, y1]


# ================================================================ 원 격자 셀 메시
def build_mesh(proj, ky, kx):
    """셀(ky, kx)의 사각형을 QuadMesh 좌표로 만든다. 반환 coords (2R, N+1, 2), 셀별 사각형 색인 (qi, qj), 색 배열 모양 (2R−1, N)."""
    ky = np.asarray(ky, np.int64)
    kx = np.asarray(kx, np.int64)
    r0 = int(ky.min())
    R = int(ky.max()) - r0 + 1
    ri = ky - r0
    kmin = np.full(R, np.iinfo(np.int64).max)
    kmax = np.full(R, np.iinfo(np.int64).min)
    np.minimum.at(kmin, ri, kx)
    np.maximum.at(kmax, ri, kx)
    emp = kmin > kmax
    kmin[emp] = 0
    kmax[emp] = -1
    n = kmax - kmin + 1
    N = int(max(n.max(), 1))
    rows = np.arange(R)
    cphi = np.cos(np.radians((r0 + rows + 0.5) * CELL_DEG))
    jj = np.minimum(np.arange(N + 1)[None, :], n[:, None])
    lon = (kmin[:, None] + jj) * CELL_DEG / cphi[:, None]
    LON = np.repeat(lon, 2, axis=0)
    LAT = np.empty_like(LON)
    LAT[0::2] = ((r0 + rows) * CELL_DEG)[:, None]
    LAT[1::2] = ((r0 + rows + 1) * CELL_DEG)[:, None]
    xy = proj.transform_points(PC, LON.ravel(), LAT.ravel())
    coords = np.stack([xy[:, 0].reshape(LON.shape), xy[:, 1].reshape(LON.shape)], -1)
    return dict(coords=coords, qi=2 * ri, qj=kx - kmin[ri], shape=(2 * R - 1, N))


def add_cells(ax, mesh, rgba, zorder=2.0):
    """셀 색(rgba, 셀 순서)을 QuadMesh 하나로 그린다. 래스터 층(600 dpi)."""
    fc = np.zeros(mesh["shape"] + (4,), np.float32)
    fc[mesh["qi"], mesh["qj"]] = rgba
    qm = QuadMesh(mesh["coords"], antialiased=False)
    qm.set_facecolor(fc.reshape(-1, 4))
    qm.set_edgecolor("none")
    qm.set_linewidth(0.0)
    qm.set_transform(ax.transData)
    qm.set_zorder(zorder)
    qm.set_rasterized(True)
    ax.add_collection(qm, autolim=False)
    return qm


def cell_rgba(d, values, cmap, norm, idx=None):
    """표시 셀은 색표 색, 수체는 흰색, 자료 없는 육지는 #B8BEC6, 그 밖은 투명. idx 가 있으면 그 셀만."""
    sel = np.arange(len(d["shown"])) if idx is None else np.asarray(idx)
    sh, wa, nd = d["shown"][sel], d["water"][sel], d["nodata"][sel]
    v = np.asarray(values, float)[sel] if len(values) == len(d["shown"]) else np.asarray(values, float)
    out = np.zeros((len(sel), 4), np.float32)
    ok = sh & np.isfinite(v)
    if cmap is None:
        out[ok] = to_rgba(MAPPED)
    else:
        out[ok] = cmap(norm(v[ok]))
    out[wa] = to_rgba(WATER)
    out[nd] = to_rgba(NODATA)
    return out


# ================================================================ 범위(지침 2.6)
def pct_range(v, lo=1.0, hi=99.0, step=5.0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(step * math.floor(np.percentile(v, lo) / step)), float(step * math.ceil(np.percentile(v, hi) / step))]


def sym_max(v, q=99.0, step=5.0):
    v = np.abs(np.asarray(v, float))
    v = v[np.isfinite(v)]
    return float(step * math.ceil(np.percentile(v, q) / step))


def extend_of(v, lo, hi):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    a, b = bool((v < lo).any()), bool((v > hi).any())
    return "both" if a and b else ("min" if a else ("max" if b else "neither"))


def ticks_between(lo, hi):
    for step in (1, 2, 5, 10, 4, 20, 25, 50, 100, 200):             # 5·10 단위를 먼저, 그다음 4 단위
        t = np.arange(math.ceil(lo / step) * step, hi + 1e-9, step)
        if 3 <= len(t) <= 5:
            return t
    return np.linspace(lo, hi, 3)


def prepare(d):
    proj = d["proj"]
    d["ext"] = box_extent(proj, d["box"], d["pad_m"])
    c = d["cells"]
    t0 = time.time()
    d["mesh"] = build_mesh(proj, c.ky.values, c.kx.values)
    d["xy"] = proj.transform_points(PC, c.lon.values, c.lat.values)[:, :2]
    d["t_mesh"] = round(time.time() - t0, 1)
    sh = d["shown"]
    d["rng"] = dict(alt=pct_range(np.r_[d["p1"][sh], d["r1"][sh]]), corr_max=sym_max(d["corr"][sh]))
    return d


# ================================================================ 바탕 지도
_NE: dict = {}


def ne_layers(key):
    """Natural Earth 10m 육지·해안선을 지역 상자(여유 포함)로 잘라 단순화한다. 해안선 점(0.02° 간격)은 모서리 검사용."""
    if key in _NE:
        return _NE[key]
    b = REG[key]["box"]
    clip = shapely.box(b["lon0"] - 7, b["lat0"] - 4, b["lon1"] + 7, b["lat1"] + 3)
    tol = REG[key]["simplify"]
    out = {}
    for name in ("land", "coastline"):
        path = shpreader.natural_earth(resolution="10m", category="physical", name=name)
        gs = []
        for g in shpreader.Reader(path).geometries():
            if not g.intersects(clip):
                continue
            gi = shapely.intersection(g, clip)
            if gi.is_empty:
                continue
            gi = shapely.simplify(gi, tol, preserve_topology=True)
            if not gi.is_empty:
                gs.append(gi)
        out[name] = gs
    pts = [shapely.get_coordinates(shapely.segmentize(g, 0.02)) for g in out["coastline"]]
    out["coast_pts"] = np.vstack(pts) if pts else np.zeros((0, 2))
    _NE[key] = out
    return out


def base_map(ax, d, ext, M, coast=True):
    ax.set_extent(ext, crs=d["proj"])
    ax.set_facecolor("#ffffff")
    ne = ne_layers(d["key"])
    ax.add_geometries(ne["land"], crs=PC, facecolor=S.BASEMAP["land"], edgecolor="none", zorder=0)
    if coast:
        ax.add_geometries(ne["coastline"], crs=PC, facecolor="none", edgecolor=S.BASEMAP["coast"], linewidth=M.lw_coast, zorder=4)
    ax.spines["geo"].set_linewidth(M.lw)
    ax.spines["geo"].set_edgecolor("#000000")


def graticule(ax, xlocs, ylocs, M, left=False, bottom=False):
    lab = {}
    if left:
        lab["left"] = "y"
    if bottom:
        lab["bottom"] = "x"
    gl = ax.gridlines(crs=PC, draw_labels=lab if lab else False, linewidth=M.lw_grat, color=S.BASEMAP["graticule"], xlocs=mticker.FixedLocator(xlocs),
                      ylocs=mticker.FixedLocator(ylocs), x_inline=False, y_inline=False, zorder=3)
    if lab:
        gl.rotate_labels = False
        gl.xlabel_style = dict(size=M.fs, rotation=0)
        gl.ylabel_style = dict(size=M.fs, rotation=0)
        gl.xpadding = 2 if M.paper else 4
        gl.ypadding = 2 if M.paper else 4
    return gl


def occupancy_points(d):
    """모서리 검사용 점: 색이 칠해지는 셀 중심(표시·자료 없음)과 해안선 점(투영 좌표). 수체 셀은 흰색이라 빈 곳으로 본다."""
    sel = d["shown"] | d["nodata"]
    pts = [d["xy"][sel]]
    cp = ne_layers(d["key"])["coast_pts"]
    if len(cp):
        pts.append(d["proj"].transform_points(PC, cp[:, 0], cp[:, 1])[:, :2])
    return np.vstack(pts)


def corner_box(ext, corner, w, h, m=0.025):
    x0, x1, y0, y1 = ext
    W, H = x1 - x0, y1 - y0
    bx = x0 + m * W if corner[1] == "l" else x1 - m * W - w * W
    by = y1 - m * H - h * H if corner[0] == "t" else y0 + m * H
    return (bx, by, w * W, h * H)


def pick_corner(pts, ext, w, h, order=("tl", "tr", "bl", "br"), avoid=()):
    """자료 셀·해안선 점이 없는 첫 모서리. 없으면 점이 가장 적은 모서리. 반환 (모서리, 상자, {모서리: 점 수})."""
    counts = {}
    for k in ("tl", "tr", "bl", "br"):
        if k in avoid:
            continue
        bx, by, bw, bh = corner_box(ext, k, w, h)
        inside = (pts[:, 0] >= bx) & (pts[:, 0] <= bx + bw) & (pts[:, 1] >= by) & (pts[:, 1] <= by + bh)
        counts[k] = int(inside.sum())
    for k in order:
        if k in counts and counts[k] == 0:
            return k, corner_box(ext, k, w, h), counts
    k = min(counts, key=counts.get)
    return k, corner_box(ext, k, w, h), counts


def bar_length(proj, x, y, km):
    """투영 좌표 (x, y) 에서 동서 방향 km 의 투영 길이."""
    lo0, la0 = PC.transform_point(x, y, proj)
    lo1, la1 = PC.transform_point(x + 10000.0, y, proj)
    _, _, d10 = GEOD.inv(lo0, la0, lo1, la1)
    return km * 1000.0 * 10000.0 / d10


def scale_bar(ax, proj, box, km, M):
    """상자 안 왼쪽 아래에 막대, 그 위에 길이 글(흰 테두리). 상자는 pick_corner 의 결과."""
    bx, by, bw, bh = box
    xs, ys = bx + 0.04 * bw, by + 0.18 * bh
    L = bar_length(proj, xs, ys, km)
    ax.plot([xs, xs + L], [ys, ys], color="#000000", lw=M.lw_scale, solid_capstyle="butt", transform=proj, zorder=8, gid="scale")
    t = ax.text(xs + L / 2, ys + 0.22 * bh, f"{km:g} km", ha="center", va="bottom", fontsize=M.fs, transform=proj, zorder=8,
                path_effects=[pe.withStroke(linewidth=1.5 if M.paper else 3.0, foreground="#ffffff")])
    t.set_gid("scale")
    return L


def scale_bar_below(fig, x_mm, y_mm, L_mm, km, M):
    """지도 틀 아래(그림 좌표)에 축척 막대와 길이 글을 한 줄로 둔다. 패널들의 축척이 같을 때만 쓴다."""
    W, H = fig.get_size_inches() * 25.4
    fig.add_artist(Line2D([x_mm / W, (x_mm + L_mm) / W], [1 - y_mm / H] * 2, color="#000000", lw=M.lw_scale, solid_capstyle="butt",
                          transform=fig.transFigure, gid="scale"))
    t = fig.text((x_mm + L_mm + 2.0) / W, 1 - y_mm / H, f"{km:g} km", ha="left", va="center", fontsize=M.fs)
    t.set_gid("scale")


def scale_box_size(d, ext, km, M, panel_w_mm):
    """축척 막대 상자의 크기(축 비율). 막대 길이와 글 폭 가운데 큰 쪽."""
    x0, x1, y0, y1 = ext
    L = bar_length(d["proj"], (x0 + x1) / 2, (y0 + y1) / 2, km) / (x1 - x0)
    txt_mm = 0.55 * (M.fs * 0.3528) * len(f"{km:g} km")
    return max(L, txt_mm / panel_w_mm) / 0.9 + 0.02, (M.fs * 0.3528 * 2.4) / panel_w_mm


def location_inset(fig, rect, box, M):
    """범북극 위치 삽도(육지 회색, 대상 상자 검정)."""
    ia = fig.add_axes(rect, projection=ccrs.NorthPolarStereo(central_longitude=-45))
    ia.set_extent([-180, 180, 52, 90], crs=PC)
    ia.set_facecolor("#ffffff")
    ia.add_geometries(list(shpreader.Reader(shpreader.natural_earth(resolution="110m", category="physical", name="land")).geometries()),
                      crs=PC, facecolor=S.BASEMAP["land"], edgecolor=S.BASEMAP["coast"], linewidth=M.lw_coast)
    lon = np.r_[np.linspace(box["lon0"], box["lon1"], 30), np.linspace(box["lon1"], box["lon0"], 30), box["lon0"]]
    lat = np.r_[np.full(30, box["lat0"]), np.full(30, box["lat1"]), box["lat0"]]
    ia.plot(lon, lat, color="#000000", lw=M.lw_coast, transform=PC)
    ia.spines["geo"].set_linewidth(M.lw)
    return ia


def inset_in_corner(fig, ax, d, ext, M, size_frac=0.22, outside_rect=None):
    """위치 삽도를 자료 셀·해안선이 없는 모서리에 둔다. 빈 모서리가 없으면 outside_rect(그림 비율 [x, y, w, h], 지도 밖)에 두고,
    그것도 없으면 그리지 않는다. 반환 (자리, 점 수 기록)."""
    pts = occupancy_points(d)
    k, (bx, by, bw, bh), counts = pick_corner(pts, ext, size_frac, size_frac, order=("tl", "tr", "br", "bl"))
    if counts[k] == 0:
        x0, x1, y0, y1 = ext
        pos = ax.get_position()
        fx = pos.x0 + (bx - x0) / (x1 - x0) * pos.width
        fy = pos.y0 + (by - y0) / (y1 - y0) * pos.height
        location_inset(fig, [fx, fy, bw / (x1 - x0) * pos.width, bh / (y1 - y0) * pos.height], d["box"], M)
        return k, counts
    if outside_rect is not None:
        location_inset(fig, outside_rect, d["box"], M)
        return "outside", counts
    return "none", counts


def scale_in_corner(ax, d, ext, M, panel_w_mm, km=None, avoid=()):
    km = km or d["scale_km"]
    w, h = scale_box_size(d, ext, km, M, panel_w_mm)
    pts = occupancy_points(d)
    k, box, counts = pick_corner(pts, ext, w, h, order=("bl", "tl", "tr", "br"), avoid=avoid)
    if counts[k] > 0:
        print(f"  [warn] {d['key']}: 축척 막대 자리에 빈 모서리가 없다({counts}). 점이 가장 적은 {k} 에 둔다", flush=True)
    scale_bar(ax, d["proj"], box, km, M)
    return k, counts


# ================================================================ 학습 범위 층과 열쇠
def rasterize(xy, layers: dict, extent, res_m, maxdist_m):
    """정규 격자(투영 좌표)에 가장 가까운 셀 값(거리 ≤ maxdist). d 의 해칭 윤곽 전용."""
    tree = cKDTree(xy)
    x0, x1, y0, y1 = extent
    nx, ny = int(math.ceil((x1 - x0) / res_m)), int(math.ceil((y1 - y0) / res_m))
    gx = x0 + (np.arange(nx) + 0.5) * res_m
    gy = y0 + (np.arange(ny) + 0.5) * res_m
    out = {k: np.full(ny * nx, np.nan) for k in layers}
    for r0 in range(0, ny, 256):
        r1 = min(ny, r0 + 256)
        XX, YY = np.meshgrid(gx, gy[r0:r1])
        dist, idx = tree.query(np.c_[XX.ravel(), YY.ravel()], distance_upper_bound=maxdist_m)
        hit = np.isfinite(dist)
        sl = slice(r0 * nx, r1 * nx)
        for k, v in layers.items():
            a = np.full(hit.size, np.nan)
            a[hit] = np.asarray(v, float)[idx[hit]]
            out[k][sl] = a
    return {k: v.reshape(ny, nx) for k, v in out.items()}, gx, gy


def range_layers(ax, d, M):
    """d 패널: 표시 셀 #e6e6e6(메시), 범위 밖 공변량 수의 해칭(정규 격자 윤곽, 래스터), 라벨 셀 점."""
    add_cells(ax, d["mesh"], cell_rgba(d, np.ones(len(d["shown"])), None, None))
    res = 1000.0 if d["key"] == "alaska" else 250.0
    R, gx, gy = rasterize(d["xy"], dict(cat=np.where(d["shown"], d["cat"], np.nan)), d["ext"], res, 1000.0 if d["key"] == "alaska" else 750.0)
    XX, YY = np.meshgrid(gx, gy)
    cs = ax.contourf(XX, YY, np.ma.masked_invalid(R["cat"]), levels=[0.5, 1.5, 2.5, 3.5], colors="none",
                     hatches=[HATCH_CAT[1], HATCH_CAT[2], HATCH_CAT[3]], transform=d["proj"], zorder=2.6)
    cs.set_rasterized(True)
    xy = d["proj"].transform_points(PC, d["labels"].lon.values, d["labels"].lat.values)
    ax.scatter(xy[:, 0], xy[:, 1], s=M.dot, c="#000000", linewidths=0, transform=d["proj"], zorder=5)


def range_key(fig, cax, M, orientation):
    """범위 밖 공변량 수의 해칭 열쇠(색 막대 형식). 감춘 축의 contourf 로 만든다."""
    dummy = fig.add_axes([0.0, 0.0, 0.001, 0.001])
    cs = dummy.contourf(np.array([[0.0, 1.0], [2.0, 3.0]]), levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=[MAPPED] * 4,
                        hatches=["", HATCH_CAT[1], HATCH_CAT[2], HATCH_CAT[3]])
    dummy.set_visible(False)
    cb = fig.colorbar(cs, cax=cax, orientation=orientation, ticks=[0, 1, 2, 3])
    cb.set_ticklabels(["0", "1", "2", "≥3"])
    for p_ in getattr(cb, "solids_patches", []):                       # 해칭 색 = 모서리 색(선 굵기 0). 지도 해칭과 같은 회색
        p_.set_edgecolor(S.BASEMAP["hatch"])
        p_.set_linewidth(0.0)
    style_cbar(cb, "Covariates outside range", M)
    cb.set_label("Covariates outside range", fontsize=M.fs_lab, labelpad=3.5 if M.paper else 8)
    return cb


def style_cbar(cb, label, M):
    cb.set_label(label, fontsize=M.fs_lab, labelpad=1.5 if M.paper else 4)
    cb.ax.tick_params(labelsize=M.fs, width=M.lw, length=M.tick_len, pad=1.5 if M.paper else 3)
    cb.outline.set_linewidth(M.lw)
    try:
        cb.dividers.set_linewidth(M.lw)
    except AttributeError:
        pass


def colorbar(fig, cax, cmap, norm, label, ticks, M, extend="neither", orientation="horizontal"):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation=orientation, extend=extend, ticks=ticks)
    fmt = mticker.FuncFormatter(lambda v, _: S.fmt_num(v, 0) if abs(v - round(v)) < 1e-9 else S.fmt_num(v, 1))
    (cb.ax.xaxis if orientation == "horizontal" else cb.ax.yaxis).set_major_formatter(fmt)
    style_cbar(cb, label, M)
    return cb


def nodata_label_legend(ax_or_fig, M, loc, ncol=1, dots=True, bbox=None, small=False):
    hs = [Patch(facecolor=NODATA, edgecolor="none", label="No data")]
    if dots:
        hs.append(Line2D([], [], marker="o", ls="none", color="#000000", markersize=math.sqrt(M.dot) * 1.8, label="Label cells"))
    kw = dict(handles=hs, loc=loc, ncol=ncol, fontsize=M.fs_small if small else M.fs, frameon=False, handlelength=1.2 if M.paper else 1.4,
              handleheight=1.0,
              handletextpad=0.4, borderaxespad=0.0, labelspacing=0.45, columnspacing=1.0)
    if bbox is not None:
        kw["bbox_to_anchor"] = bbox
    return ax_or_fig.legend(**kw)


def letter(fig, x_mm, y_mm, k, M, title=None):
    W, H = fig.get_size_inches() * 25.4
    fig.text(x_mm / W, 1 - y_mm / H, k, fontsize=M.fs_letter, fontweight="bold", ha="left", va="bottom")
    if title:
        fig.text((x_mm + (3.0 if M.paper else 5.5)) / W, 1 - y_mm / H, title, fontsize=M.fs_head, ha="left", va="bottom")


# ================================================================ 지도 그림
PANEL_TITLE = {"a": "Recalibrated Stefan", "b": "ML correction", "c": "Residual ML", "d": "Training range"}


def draw_panel(ax, d, k, M):
    norm_alt = Normalize(*d["rng"]["alt"])
    v = d["rng"]["corr_max"]
    if k == "a":
        add_cells(ax, d["mesh"], cell_rgba(d, d["p1"], CM_ALT, norm_alt))
    elif k == "b":
        add_cells(ax, d["mesh"], cell_rgba(d, d["corr"], CM_DIFF, TwoSlopeNorm(0.0, -v, v)))
    elif k == "c":
        add_cells(ax, d["mesh"], cell_rgba(d, d["r1"], CM_ALT, norm_alt))
    else:
        range_layers(ax, d, M)


def fig_map_paper(d):
    M = Med("paper")
    use_medium("paper")
    proj, ext = d["proj"], d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    W, LM, KEYL, GAP, KEYR, RM = 170.0, 8.5, 13.0, 2.5, 19.0, 0.3
    mw = (W - LM - KEYL - GAP - KEYR - RM) / 2
    mh = mw * asp
    TOP, ROWGAP, BOT = 3.2, 3.6, 4.6
    H = TOP + 2 * mh + ROWGAP + BOT
    fig = S.fig_mm(W, H)
    xc = [LM, LM + mw + KEYL + GAP]
    yr = [TOP, TOP + mh + ROWGAP]
    pos = {"a": (0, 0), "b": (0, 1), "c": (1, 0), "d": (1, 1)}
    axes = {}
    for k, (i, j) in pos.items():
        ax = S.axes_mm(fig, xc[j], yr[i], mw, mh, projection=proj)
        base_map(ax, d, ext, M)
        draw_panel(ax, d, k, M)
        graticule(ax, d["xlocs"], d["ylocs"], M, left=(j == 0), bottom=(i == 1))
        letter(fig, xc[j], yr[i] - 0.5, k, M)
        axes[k] = ax
    sh = d["shown"]
    lo, hi = d["rng"]["alt"]
    v = d["rng"]["corr_max"]
    span = 2 * mh + ROWGAP
    cax = S.axes_mm(fig, xc[0] + mw + 1.5, yr[0] + 0.15 * span, M.cbar, 0.7 * span)
    colorbar(fig, cax, CM_ALT, Normalize(lo, hi), "ALT (cm)", ticks_between(lo, hi), M,
             extend_of(np.r_[d["p1"][sh], d["r1"][sh]], lo, hi), orientation="vertical")
    cax = S.axes_mm(fig, xc[1] + mw + 1.5, yr[0] + 0.1 * mh, M.cbar, 0.8 * mh)
    colorbar(fig, cax, CM_DIFF, TwoSlopeNorm(0.0, -v, v), "ALT change (cm)", ticks_between(-v, v), M, extend_of(d["corr"][sh], -v, v),
             orientation="vertical")
    cax = S.axes_mm(fig, xc[1] + mw + 1.5, yr[1] + 0.04 * mh, 4.0, 0.5 * mh)
    range_key(fig, cax, M, "vertical")
    lg = S.axes_mm(fig, xc[1] + mw + 1.0, yr[1] + 0.62 * mh, KEYR - 1.2, 0.3 * mh)
    lg.set_axis_off()
    nodata_label_legend(lg, M, "upper left")
    rec = {}
    side = 11.5                                                         # 지도 밖 자리: 왼쪽 열쇠 칸 위(색 막대 위 빈 곳)
    out_rect = [(xc[0] + mw + 0.9) / W, 1 - (yr[0] + side) / H, side / W, side / H]
    k_in, c_in = inset_in_corner(fig, axes["a"], d, ext, M, size_frac=0.22, outside_rect=out_rect)
    rec["inset"] = dict(place=k_in, counts=c_in)
    k_sb, c_sb = scale_in_corner(axes["a"], d, ext, M, mw, avoid=(k_in,) if k_in in ("tl", "tr", "bl", "br") else ())
    rec["scale_bar"] = dict(corner=k_sb, counts=c_sb)
    d.setdefault("layout", {})["paper"] = dict(size_mm=[W, round(H, 1)], map_mm=[round(mw, 1), round(mh, 1)], **rec)
    return fig


def fig_map_slide(d):
    M = Med("slide")
    use_medium("slide")
    proj, ext = d["proj"], d["ext"]
    asp = (ext[3] - ext[2]) / (ext[1] - ext[0])
    W, LM, GAP, RM = SLIDE_W_MM, 17.0, 5.0, 1.5
    mw = (W - LM - 3 * GAP - RM) / 4
    mh = mw * asp
    TOP, LONLAB, CB = 10.0, 8.0, 21.5
    H = TOP + mh + LONLAB + CB + 1.5
    assert H <= SLIDE_H_MAX_MM, H
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    xs = [LM + k * (mw + GAP) for k in range(4)]
    sh = d["shown"]
    lo, hi = d["rng"]["alt"]
    v = d["rng"]["corr_max"]
    axes = {}
    for j, k in enumerate("abcd"):
        ax = S.axes_mm(fig, xs[j], TOP, mw, mh, projection=proj)
        base_map(ax, d, ext, M)
        draw_panel(ax, d, k, M)
        graticule(ax, d["xlocs"][::2] if d["key"] == "lena" else d["xlocs"], d["ylocs"], M, left=(j == 0), bottom=(j == 0))
        letter(fig, xs[j], TOP - 1.5, k, M, PANEL_TITLE[k])
        axes[k] = ax
    ycb = TOP + mh + LONLAB + 1.0
    for j, k in enumerate("abc"):
        cax = S.axes_mm(fig, xs[j] + 0.08 * mw, ycb, 0.84 * mw, M.cbar)
        if k == "b":
            colorbar(fig, cax, CM_DIFF, TwoSlopeNorm(0.0, -v, v), "ALT change (cm)", ticks_between(-v, v), M, extend_of(d["corr"][sh], -v, v))
        else:
            colorbar(fig, cax, CM_ALT, Normalize(lo, hi), "ALT (cm)", ticks_between(lo, hi), M, extend_of(np.r_[d["p1"][sh], d["r1"][sh]], lo, hi))
    cax = S.axes_mm(fig, xs[3] + 0.08 * mw, ycb, 0.84 * mw, M.cbar)
    range_key(fig, cax, M, "horizontal")
    lg = S.axes_mm(fig, xs[3], TOP + mh + 1.0, mw, LONLAB - 1.5)            # 좌표 글이 없는 d 아래 띠
    lg.set_axis_off()
    nodata_label_legend(lg, M, "center left", ncol=2, small=True)
    rec = {}
    k_in, c_in = inset_in_corner(fig, axes["a"], d, ext, M, size_frac=0.24, outside_rect=None)   # 슬라이드: 빈 모서리가 없으면 생략
    rec["inset"] = dict(place=k_in, counts=c_in)
    L_mm = bar_length(proj, (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, d["scale_km"]) / (ext[1] - ext[0]) * mw
    scale_bar_below(fig, xs[1], TOP + mh + 4.2, L_mm, d["scale_km"], M)       # 좌표 글이 없는 b 아래 띠(같은 축척)
    rec["scale_bar"] = dict(place="below_b", length_mm=round(L_mm, 2))
    d.setdefault("layout", {})["slide"] = dict(size_in=[round(W / 25.4, 2), round(H / 25.4, 2)], **rec)
    return fig


# ================================================================ 확대 위치(XL_display_resolution 에서 쓴다)
def zoom_center(d):
    """라벨 셀 수가 많은 0.5° 블록부터, 블록 라벨 중심의 ±zoom_half 상자에서 표시 셀이 상자 면적의 zoom_min_cover 이상인 첫 블록."""
    L = d["labels"]
    b = L.groupby("block").agg(n=("lat", "size"), lat=("lat", "mean"), lon=("lon", "mean")).sort_values("n", ascending=False)
    proj = d["proj"]
    xy = d["xy"][d["shown"]]
    hz = d["zoom_half_m"]
    for blk, r in b.iterrows():
        xc, yc = proj.transform_point(r.lon, r.lat, PC)
        n_in = int(((np.abs(xy[:, 0] - xc) <= hz) & (np.abs(xy[:, 1] - yc) <= hz)).sum())
        cover = n_in / ((2 * hz / 1000.0) ** 2)
        if cover >= d["zoom_min_cover"]:
            return float(r.lat), float(r.lon), int(r.n), str(blk), float(cover)
    r = b.iloc[0]
    return float(r.lat), float(r.lon), int(r.n), str(b.index[0]), float("nan")


# ================================================================ 기록
def source_values(d) -> dict:
    sh = d["shown"]

    def q(v):
        v = np.asarray(v, float)[sh]
        v = v[np.isfinite(v)]
        return dict(n=int(len(v)), p01=float(np.percentile(v, 1)), p25=float(np.percentile(v, 25)), p50=float(np.median(v)),
                    p75=float(np.percentile(v, 75)), p99=float(np.percentile(v, 99)), mean=float(v.mean()))
    return dict(region=d["name"], n_cells=int(len(d["cells"])), n_shown=int(sh.sum()), n_water=int(d["water"].sum()), n_nodata_land=int(d["nodata"].sum()),
                n_label_cells=int(len(d["labels"])), ranges=dict(alt_a_c=d["rng"]["alt"], corr_b=[-d["rng"]["corr_max"], d["rng"]["corr_max"]]),
                p1=q(d["p1"]), r1=q(d["r1"]), corr=q(d["corr"]),
                extrap_cat_shown={str(int(k)): int(v) for k, v in pd.Series(d["cat"][sh]).value_counts().sort_index().items()},
                frac_outside_shown=float((d["n_out"][sh] >= 1).mean()), interval90_halfwidth_cm=d["interval_q"], coef=d["coef"],
                mesh_rows_cols=list(d["mesh"]["shape"]), layout=d.get("layout", {}), cartopy=cartopy.__version__)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="both")
    ap.add_argument("--medium", default="both")
    a = ap.parse_args(argv)
    regions = ["alaska", "lena"] if a.region == "both" else [a.region]
    media = ["paper", "slide"] if a.medium == "both" else [a.medium]
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    OUT_SLIDE.mkdir(parents=True, exist_ok=True)
    for key in regions:
        t0 = time.time()
        d = prepare(load_region(key))
        stem = "Alaska_ALT_map_v3" if key == "alaska" else "Lena_ALT_map_v3"
        for medium in media:
            fig = fig_map_paper(d) if medium == "paper" else fig_map_slide(d)
            if medium == "paper":
                S.save_fig(fig, stem, OUT_PAPER, formats=("pdf", "png"))
                au = S.audit_v3(fig)
                print(f"[audit] {stem}: sizes {sorted(au['sizes'])} chars {au['chars']} thin {au['thin_lines'][:4]} long {au['long_labels'][:4]} "
                      f"numbers {au['loose_numbers'][:4]} codes {au['codes'][:4]}", flush=True)
            else:
                fig.savefig(OUT_SLIDE / f"{stem}_slide.png", dpi=300)
            plt.close(fig)
        sv = source_values(d)
        (OUT_PAPER / f"{stem}_source_values.json").write_text(json.dumps(sv, ensure_ascii=False, indent=1))
        print(f"[{key}] 범위 {sv['ranges']} · 배치 {json.dumps(sv['layout'], ensure_ascii=False)} · {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
