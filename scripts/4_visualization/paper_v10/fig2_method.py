# -*- coding: utf-8 -*-
"""논문 초안 v10 Fig. 2: 방법 구조와 검증 분할(Scientific Reports 본문 그림, 180 mm 폭).

패널
  a  물리 + 잔차 ML 구조(scripts/4_visualization/v7_f3_method.py 패널 a · 덱 M01 을 논문 mm 좌표로 다시 그림).
     Input | Model | Output 옅은 파선 영역. 입력 썸네일(알래스카 2014): 기후 MAAT · FDD · Snow · TDD, 정적 Elevation · Slope · Soil carbon · Clay,
     모두 한 색표(INPUT_CMAP). 모델 상자는 옅은 회색 면 + 얇은 진회색 테두리(Stefan, Gradient boosting). 잔차 ML 경로(학습기 → 잔차 → ⊕, 적용 범위)와
     최종 ALT 썸네일 테두리만 주홍 #D55E00. 학습 때만 쓰는 경로(관측 y, 물리 ŷ_P, 목표 y − ŷ_P)는 회색 파선.
  b  비교한 네 구조의 작은 도식: Direct ML, Physics-input ML, Physics pseudo-labels, Physics + residual ML.
  c  다섯 검증 분할(레나 델타 라벨 셀, 겹 하나씩 예시): Random(겹 1), Spatial block 0.5°(겹 1), Year hold-out(2016), Block × year(겹 1 × 2016),
     Region hold-out(레나 델타 전체 시험, 학습 = 알래스카 · 캐나다). 시험 셀 주홍, 학습 셀 진회색, 제외 셀 옅은 회색. 열쇠 하나.
자료(수치는 모두 파일에서 읽는다)
  a: data/processed/v5/maps/Alaska/pred_2014.csv.gz(e5_tdd, p1, r1) · meta.json(λ, E, feats) · mean_2011_2024.csv.gz(aoa_in) ·
     forcing_2014_thumb.csv.gz(e5_maat, e5_fdd, e5_swe; 없으면 grid/alaska_forcing_year.csv 에서 만든다),
     data/processed/map_alaska/alaska_grid_x25_v1.csv.gz(dem_elev, dem_slope, sg_soc_0_5, sg_clay_5_15, land), labels_cell_year_v5.csv(2014 관측 셀).
     잔차 썸네일 = (r1 − p1)/λ(가중 전 잔차 예측), ALT 2014 = r1, Physics ALT = p1(같은 색 눈금).
  c: polar.testbed_v5 load_v5 → with_forcing('year_obs') → usable_mask → 레나 행 → make_folds(k 5, seed 0). 실험 하네스
     scripts/3_deep_learning/v5_compare.py 와 같은 경로이고, data/processed/v5/runs/core_v6/preds 의 겹 배정과 일치함을 실행 때 확인한다.
실행  cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v10/fig2_method.py
산출  outputs/figures/paper/v10/Fig2_method.{pdf,svg,png}(180 mm 폭, png 600 dpi), outputs/figures/paper/v10/source_data/Fig2_method.json
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
import inspect, json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S
import polar.testbed_v5 as TB
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import cartopy.io.shapereader as shpreader
import shapely
from cmcrameri import cm as cmc

PROC = ROOT / "data" / "processed"; V5 = PROC / "v5"
OUT = ROOT / "outputs" / "figures" / "paper" / "v10"; STEM = "Fig2_method"
SRC: dict = {}

# ---------------------------------------------------------------- 양식
W_MM, H_MM = 180.0, 118.3
FS = 7.0; FS_SMALL = 6.5; FS_TINY = 6.0; FS_LETTER = 8.0
INK = "#000000"; DARK = "#333333"
ACCENT = S.METHOD_HEX["R1"]                       # 제안 방법 주홍(paper_v10: #D55E00)
TRAIN = "#8C8C8C"                                  # 학습 전용 경로(회색 파선)
DASH = (0, (2.6, 1.6))
BOX_FACE = "#EEEEEE"; BOX_EDGE = DARK
REGION_EDGE = "#B3B3B3"
FRAME = "#9A9A9A"; THUMB_LAND = "#E6E6E6"; THUMB_COAST = "#9A9A9A"
LW_LINE = 0.6; LW_BOX = 0.6
CHEV_L, CHEV_ANG = 1.05, 32.0                      # 열린 셰브런(두 짧은 획) 길이(mm) · 반각(도)
INPUT_CMAP = ListedColormap(cmc.grayC_r(np.linspace(0.04, 0.82, 256)))      # 입력 썸네일 공통 색표(무채색 한 가지)
ALT_CMAP = ListedColormap(cmc.oslo_r(np.linspace(0.12, 0.90, 256)))        # ALT(maps_alt_v3.CM_ALT 와 같은 구간)
RES_CMAP = cmc.vik                                                           # 잔차(0 중심 대칭)
AOA_IN, AOA_OUT = "#8FA6C8", "#DCDCDC"                               # 적용 범위 안(oslo 계열 옅은 청회) · 밖(옅은 회색)
ALASKA_ASP = 1.0 / np.cos(np.deg2rad(65.5))
# c 지도
LENA_EXT = (123.5, 130.4, 71.35, 72.95); LENA_LAT = 72.15
C_TEST, C_TRAIN, C_EXCL = "#000000", "#9A9A9A", "#D9D9D9"     # 2026-10-16: 시험 셀 검정(주홍은 제안 방법 전용), 학습 회색, 제외 옅은 회색
C_LAND, C_COAST = "#EFEFEF", "#A0A0A0"


def setup():
    S.setup()
    try:
        fm.fontManager.addfont("/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf")
    except Exception:
        pass
    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_TINY, "ytick.labelsize": FS_TINY, "legend.fontsize": FS_SMALL,
        "axes.linewidth": 0.6, "lines.linewidth": LW_LINE, "hatch.linewidth": 0.4, "text.color": INK, "axes.edgecolor": INK,
        "mathtext.fontset": "custom", "mathtext.rm": "FreeSans", "mathtext.it": "FreeSans:italic", "mathtext.bf": "FreeSans:bold",
        "mathtext.sf": "FreeSans", "mathtext.default": "it", "mathtext.fallback": None, "svg.fonttype": "none", "pdf.fonttype": 42})


def rect(x, y_top, w, h):
    return [x / W_MM, 1 - (y_top + h) / H_MM, w / W_MM, h / H_MM]


# ================================================================ 자료(a)
def forcing_year_thumb(year=2014):
    """격자 연별 기후(grid/alaska_forcing_year.csv)에서 그 해 MAAT · FDD · SWE 만 추린 캐시(v7_f3_method.forcing_year_thumb 과 같은 파일)."""
    cache = V5 / "maps" / "Alaska" / f"forcing_{year}_thumb.csv.gz"
    cols = ["e5_maat", "e5_fdd", "e5_swe"]
    if cache.exists():
        d = pd.read_csv(cache, dtype={"cell_id": str})
        if all(c in d.columns for c in cols):
            return d, str(cache.relative_to(ROOT))
    parts = []
    for ch in pd.read_csv(V5 / "grid" / "alaska_forcing_year.csv", usecols=["cell_id", "year"] + cols, dtype={"cell_id": str}, chunksize=2_000_000):
        parts.append(ch[ch.year == year])
    d = pd.concat(parts, ignore_index=True)
    return d, "data/processed/v5/grid/alaska_forcing_year.csv (year %d)" % year


def alaska_thumbs(year, lam):
    """그 해 알래스카 1 km 지도 셀에 정적 공변량 · 그 해 기후 · 적용 범위를 붙여 0.05° × 0.02° 상자 평균 래스터로 만든다(v7_f3_method.alaska_thumbs 와 같은 규칙)."""
    p = pd.read_csv(V5 / "maps" / "Alaska" / f"pred_{year}.csv.gz", dtype={"cell_id": str})
    g = pd.read_csv(PROC / "map_alaska" / "alaska_grid_x25_v1.csv.gz", dtype={"cell_id": str},
                    usecols=["cell_id", "lat", "lon", "dem_elev", "dem_slope", "sg_soc_0_5", "sg_clay_5_15", "land"])
    fy, fy_src = forcing_year_thumb(year)
    fy = fy.rename(columns={"e5_maat": "maat", "e5_fdd": "fdd", "e5_swe": "swe"})[["cell_id", "maat", "fdd", "swe"]]
    a = pd.read_csv(V5 / "maps" / "Alaska" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "aoa_in"])
    m = p.merge(g.drop(columns=["lat", "lon"]), on="cell_id", how="left").merge(fy, on="cell_id", how="left").merge(a, on="cell_id", how="left")
    m["resid"] = (m.r1 - m.p1) / lam                                       # 가중 전 잔차 예측(R1 = P1 + λ · 잔차)
    lon0, lon1 = np.floor(m.lon.min() * 10) / 10, np.ceil(m.lon.max() * 10) / 10
    lat0, lat1 = np.floor(m.lat.min() * 10) / 10, np.ceil(m.lat.max() * 10) / 10
    xe = np.arange(lon0, lon1 + 1e-9, 0.05); ye = np.arange(lat0, lat1 + 1e-9, 0.02)
    gl = g[g.land == 1]; cnt, _, _ = np.histogram2d(gl.lat, gl.lon, bins=[ye, xe])
    ras = {}
    for k in ("e5_tdd", "maat", "fdd", "swe", "dem_elev", "dem_slope", "sg_soc_0_5", "sg_clay_5_15", "p1", "r1", "resid", "aoa_in"):
        v = m[k].values.astype(float); ok = np.isfinite(v)
        s_, _, _ = np.histogram2d(m.lat.values[ok], m.lon.values[ok], bins=[ye, xe], weights=v[ok])
        c_, _, _ = np.histogram2d(m.lat.values[ok], m.lon.values[ok], bins=[ye, xe])
        with np.errstate(invalid="ignore", divide="ignore"):
            ras[k] = np.where(c_ > 0, s_ / c_, np.nan)
    ras["aoa_bin"] = np.where(np.isfinite(ras["aoa_in"]), (ras["aoa_in"] >= 0.5).astype(float), np.nan)
    ext = [lon0, lon1, lat0, lat1]; wh = (lon1 - lon0) / ((lat1 - lat0) * ALASKA_ASP)
    return dict(ras=ras, mask=cnt > 0, extent=ext, vals=m, wh=wh, forcing_src=fy_src)


_NE = {}


def ne_layers(key):
    """Natural Earth 10m 육지 · 해안선을 상자로 잘라 단순화(maps_alt_v3.ne_layers 와 같은 규칙)."""
    if key in _NE:
        return _NE[key]
    box = {"alaska": (-168.0, -141.0, 59.0, 71.5, 0.01), "lena": (123.3, 130.1, 71.5, 73.6, 0.002)}[key]
    clip = shapely.box(box[0] - 7, box[2] - 4, box[1] + 7, box[3] + 3)
    out = {}
    for name in ("land", "coastline"):
        gs = []
        for g in shpreader.Reader(shpreader.natural_earth(resolution="10m", category="physical", name=name)).geometries():
            if not g.intersects(clip):
                continue
            gi = shapely.intersection(g, clip)
            if gi.is_empty:
                continue
            gi = shapely.simplify(gi, box[4], preserve_topology=True)
            if not gi.is_empty:
                gs.append(gi)
        out[name] = gs
    _NE[key] = out
    return out


def pct_range(v, lo=1.0, hi=99.0, step=5.0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(step * math.floor(np.percentile(v, lo) / step)), float(step * math.ceil(np.percentile(v, hi) / step))]


def sym_max(v, q=99.0, step=5.0):
    v = np.abs(np.asarray(v, float)); v = v[np.isfinite(v)]
    return float(step * math.ceil(np.percentile(v, q) / step))


def learner_spec():
    src = inspect.getsource(TB.fit_learner)
    m = re.search(r"iterations=(\d+),\s*learning_rate=([\d.]+),\s*depth=(\d+)\)\s*if name == \"catboost\"", src)
    assert m, "testbed_v5.fit_learner catboost 설정을 찾지 못함"
    return dict(iterations=int(m.group(1)), learning_rate=float(m.group(2)), depth=int(m.group(3)))


# ================================================================ 그리기 도구(mm 좌표 캔버스: x 오른쪽, y 아래쪽)
def canvas(fig):
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_MM); ax.set_ylim(H_MM, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def line(ax, pts, color=INK, lw=LW_LINE, ls="-", head=True, z=5, gap=None):
    """꺾은선(가로 · 세로 구간) + 끝의 열린 셰브런. gap=(x, y, half) 이면 그 점 주변을 비운다(교차 표시)."""
    pts = np.asarray(pts, float)
    segs = [pts]
    if gap is not None:
        gx, gy, hh = gap; segs = []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if abs(x0 - x1) < 1e-9 and min(y0, y1) < gy < max(y0, y1) and abs(x0 - gx) < 1e-6:
                sgn = np.sign(y1 - y0); segs += [np.array([[x0, y0], [x0, gy - sgn * hh]]), np.array([[x0, gy + sgn * hh], [x1, y1]])]
            elif abs(y0 - y1) < 1e-9 and min(x0, x1) < gx < max(x0, x1) and abs(y0 - gy) < 1e-6:
                sgn = np.sign(x1 - x0); segs += [np.array([[x0, y0], [gx - sgn * hh, y0]]), np.array([[gx + sgn * hh, y0], [x1, y1]])]
            else:
                segs.append(np.array([[x0, y0], [x1, y1]]))
    for sg in segs:
        ax.plot(sg[:, 0], sg[:, 1], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt", dash_capstyle="butt", solid_joinstyle="miter")
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        L = float(np.hypot(x1 - x0, y1 - y0)); ux, uy = (x1 - x0) / L, (y1 - y0) / L
        for sgn in (-1, 1):
            a = np.deg2rad(180 - sgn * CHEV_ANG)
            vx = ux * np.cos(a) - uy * np.sin(a); vy = ux * np.sin(a) + uy * np.cos(a)
            ax.plot([x1, x1 + CHEV_L * vx], [y1, y1 + CHEV_L * vy], color=color, lw=lw, zorder=z + 0.1, solid_capstyle="round")


def dot(ax, x, y, color=INK, r=0.45, z=7):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=z))


def box(ax, x, y, w, h, text=None, fs=FS, z=4):
    """모델 상자(왼쪽 위 x, y, 폭, 높이; mm)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6", facecolor=BOX_FACE, edgecolor=BOX_EDGE, lw=LW_BOX, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=z + 1)


def op_node(ax, x, y, r=1.7, op="+", z=7):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=DARK, lw=LW_LINE, zorder=z))
    k = 0.55 * r
    ax.plot([x - k, x + k], [y, y], color=DARK, lw=LW_LINE, zorder=z + 1, solid_capstyle="butt")
    if op == "+":
        ax.plot([x, x], [y - k, y + k], color=DARK, lw=LW_LINE, zorder=z + 1, solid_capstyle="butt")


def thumb(fig, x, y_top, w, h, edge=FRAME, lw=0.5):
    ia = fig.add_axes(rect(x, y_top, w, h))
    ia.set_xticks([]); ia.set_yticks([])
    for sp in ia.spines.values():
        sp.set_visible(True); sp.set_edgecolor(edge); sp.set_linewidth(lw)
    ia.patch.set_facecolor("white")
    return ia


def alaska_img(ia, A, key, cmap, norm):
    ne = ne_layers("alaska")
    for g in ne["land"]:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.exterior.xy; ia.fill(xs, ys, facecolor=THUMB_LAND, edgecolor="none", lw=0, zorder=0.5)
    ia.imshow(np.where(A["mask"], 1.0, np.nan), extent=A["extent"], origin="lower", cmap=ListedColormap([THUMB_LAND]), vmin=0, vmax=1,
              aspect=ALASKA_ASP, interpolation="nearest", zorder=1)
    im = None
    if key is not None:
        im = ia.imshow(A["ras"][key], extent=A["extent"], origin="lower", cmap=cmap, norm=norm, aspect=ALASKA_ASP, interpolation="nearest", zorder=2)
    for g in ne["coastline"]:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.xy; ia.plot(xs, ys, color=THUMB_COAST, lw=0.25, zorder=3)
    ia.set_xlim(A["extent"][0], A["extent"][1]); ia.set_ylim(A["extent"][2], A["extent"][3]); ia.set_aspect(ALASKA_ASP)
    return im


def tw_mm(ax, s, fs):
    """글자 폭(mm)."""
    r = ax.figure.canvas.get_renderer(); t = ax.text(0, 0, s, fontsize=fs); w = t.get_window_extent(r).width; t.remove()
    return w / ax.figure.dpi * 25.4


def _fmt(v, _=None):
    return f"{v:g}".replace("-", "−")


# ================================================================ a 구조도
def panel_a(fig, ax, A, lab, meta):
    yr = int(meta["year"]); lam = float(meta["lam"]); m = A["vals"]
    TH = 11.5; TW = TH * A["wh"]
    OY = 1.2                                                     # 위 여백(패널 문자 위 1.2 mm)만큼 a 전체를 내린다
    R_TOP, R_BOT = 3.0 + OY, 61.5 + OY
    RX = dict(Input=(1.5, 66.5), Model=(69.5, 111.5), Output=(114.5, 179.0))
    for nm, (xa, xb) in RX.items():
        ax.add_patch(FancyBboxPatch((xa, R_TOP), xb - xa, R_BOT - R_TOP, boxstyle="round,pad=0,rounding_size=1.0", facecolor="none",
                                    edgecolor=REGION_EDGE, lw=0.5, ls=(0, (3.0, 2.0)), zorder=1))
        ax.text(xa + 1.6, R_TOP + 1.2, nm, ha="left", va="top", fontsize=FS, zorder=2)
    yU, yL, yT = 17.5 + OY, 36.8 + OY, 53.0 + OY
    feats = list(meta["feats"]); n_clim = sum(f in TB.F8 for f in feats); n_stat = len(feats) - n_clim
    assert len(feats) == len(TB.FEATSETS["F8TS"]) and n_clim == 8
    # ---- Input: 기후(그 해) · 정적 썸네일(같은 열), 열 폭 = max(썸네일, 이름)
    CLIM = [("maat", "MAAT"), ("fdd", "FDD"), ("swe", "Snow"), ("e5_tdd", "TDD")]
    STAT = [("dem_elev", "Elevation"), ("dem_slope", "Slope"), ("sg_soc_0_5", "Soil carbon"), ("sg_clay_5_15", "Clay")]
    widths = [max(TW, tw_mm(ax, a[1], FS_SMALL), tw_mm(ax, b[1], FS_SMALL)) for a, b in zip(CLIM, STAT)]
    gap = 2.6; xr = 61.0; xs = []; x = xr
    for wd in reversed(widths):
        xs.append(x - wd / 2); x -= wd + gap
    cx = xs[::-1]
    pct = {"dem_slope": (0, 98)}
    for row, yc, head in ((CLIM, yU, f"Climate, {yr} ({n_clim} variables)"), (STAT, yL, f"Static ({n_stat} variables)")):
        for (key, nm), c in zip(row, cx):
            ia = thumb(fig, c - TW / 2, yc - TH / 2, TW, TH)
            a_ = A["ras"][key]; lo, hi = np.nanpercentile(a_, pct.get(key, (2, 98)))
            alaska_img(ia, A, key, INPUT_CMAP, Normalize(lo, hi))
            ax.text(c, yc + TH / 2 + 0.8, nm, ha="center", va="top", fontsize=FS_SMALL)
        ax.text(cx[0] - TW / 2, yc - TH / 2 - 0.9, head, ha="left", va="bottom", fontsize=FS)
    x_tdd_r = cx[-1] + TW / 2
    o = lab[(lab.region == "Alaska") & (lab.year == yr)]
    ia = thumb(fig, xr - TW, yT - TH / 2, TW, TH); alaska_img(ia, A, None, None, None)
    ia.scatter(o.lon, o.lat, s=0.6, color=INK, lw=0, zorder=4, rasterized=True)
    ax.text(xr - TW - 1.4, yT, "Observed ALT", ha="right", va="center", fontsize=FS_SMALL)
    # ---- Model: Stefan(위) · Gradient boosting(가운데), ⊖(아래, 학습 전용)
    bx, bw = 76.0, 28.0; hP, hL = 7.5, 7.5                          # 두 모델 상자 같은 높이
    box(ax, bx, yU - hP / 2, bw, hP, "Stefan")
    ax.text(bx + bw / 2, yU + hP / 2 + 1.0, r"$\mathrm{ALT}=E\,\sqrt{\mathrm{TDD}}$", ha="center", va="top", fontsize=FS_SMALL)
    box(ax, bx, yL - hL / 2, bw, hL, "Gradient boosting")
    line(ax, [(x_tdd_r, yU), (bx, yU)])
    xbr = xr + 2.6; ytb, ybb = yU - TH / 2, yL + TH / 2; tick = 1.0; gp = 1.0      # 괄호: 두 행(기후 · 정적)의 위 끝 ~ 아래 끝
    for (ya_, yb_) in ((ytb, yU - gp), (yU + gp, ybb)):
        ax.plot([xbr, xbr], [ya_, yb_], color=INK, lw=LW_LINE, zorder=5, solid_capstyle="butt")
    ax.plot([xbr - tick, xbr], [ytb, ytb], color=INK, lw=LW_LINE, zorder=5, solid_capstyle="butt")
    ax.plot([xbr - tick, xbr], [ybb, ybb], color=INK, lw=LW_LINE, zorder=5, solid_capstyle="butt")
    line(ax, [(xbr, yL), (bx, yL)])
    xm = bx + bw / 2; rm = 1.7
    op_node(ax, xm, yT, rm, "−")
    # ---- Output: Physics ALT · Residual → ⊕(λ) → ALT 2014, Applicability
    ox = 121.0; xb_ = 139.5; xp = 151.5; rp = 1.7; gx = 158.0
    lo, hi = pct_range(np.r_[m.p1.values, m.r1.values]); nALT = Normalize(lo, hi)
    v = sym_max(m.resid.values); nRES = TwoSlopeNorm(0.0, -v, v)
    line(ax, [(bx + bw, yU), (ox, yU)])
    line(ax, [(bx + bw, yL), (ox, yL)], color=ACCENT)
    ia = thumb(fig, ox, yU - TH / 2, TW, TH); alaska_img(ia, A, "p1", ALT_CMAP, nALT)
    ax.text(ox + TW / 2, yU + TH / 2 + 0.8, "Physics ALT", ha="center", va="top", fontsize=FS_SMALL)
    ia = thumb(fig, ox, yL - TH / 2, TW, TH); alaska_img(ia, A, "resid", RES_CMAP, nRES)
    t_res = ax.text(ox + TW / 2, yL + TH / 2 + 0.8, "Residual", ha="center", va="top", fontsize=FS_SMALL)
    # 잔차 소형 색 막대(가로)
    cbw = TW + 1.0; cby = yL + TH / 2 + 4.1
    cax = fig.add_axes(rect(ox - 0.5, cby, cbw, 1.3))
    cb = fig.colorbar(ScalarMappable(norm=nRES, cmap=RES_CMAP), cax=cax, orientation="horizontal", ticks=[-v, 0, v])
    cb.outline.set_linewidth(0.5); cb.outline.set_edgecolor(FRAME)
    cax.tick_params(labelsize=FS_TINY, width=0.5, length=1.2, pad=0.8); cax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.text(ox - 0.5 + cbw + 0.8, cby + 0.65, "cm", ha="left", va="center", fontsize=FS_TINY)
    # 흐름
    line(ax, [(ox + TW, yU), (xp - rp, yU)]); dot(ax, xb_, yU)
    line(ax, [(ox + TW, yL), (xp, yL)], color=ACCENT, head=False); dot(ax, xp, yL, color=ACCENT)
    line(ax, [(xp, yL), (xp, yU + rp)], color=ACCENT)
    ax.text(xp - 1.2, (yU + yL) / 2, r"$\lambda$ = %g" % lam, ha="right", va="center", fontsize=FS_SMALL)
    op_node(ax, xp, yU, rp, "+")
    line(ax, [(xp + rp, yU), (gx, yU)])
    ia = thumb(fig, gx, yU - TH / 2, TW, TH, edge=ACCENT, lw=0.9); alaska_img(ia, A, "r1", ALT_CMAP, nALT)
    ax.text(gx + TW / 2, yU + TH / 2 + 0.8, f"ALT {yr}", ha="center", va="top", fontsize=FS_SMALL)
    tk = [t for t in (20, 30, 40, 50, 60, 70, 80, 90, 100) if lo <= t <= hi]; tk = tk[::2] if len(tk) > 3 else tk
    vcax = fig.add_axes(rect(gx + TW + 1.6, yU - TH / 2, 1.3, TH))
    cb2 = fig.colorbar(ScalarMappable(norm=nALT, cmap=ALT_CMAP), cax=vcax, orientation="vertical", ticks=tk)
    cb2.outline.set_linewidth(0.5); cb2.outline.set_edgecolor(FRAME)
    vcax.tick_params(labelsize=FS_TINY, width=0.5, length=1.2, pad=0.8)
    ax.text(gx + TW + 1.6 + 0.65, yU - TH / 2 - 0.7, "cm", ha="center", va="bottom", fontsize=FS_TINY)
    line(ax, [(xp, yL), (gx, yL)], color=ACCENT)
    ia = thumb(fig, gx, yL - TH / 2, TW, TH)
    alaska_img(ia, A, "aoa_bin", ListedColormap([AOA_OUT, AOA_IN]), Normalize(0, 1))
    ax.text(gx + TW / 2, yL + TH / 2 + 0.8, "Applicability\n2011–2024 mean", ha="center", va="top", fontsize=FS_SMALL, linespacing=1.1)
    yk = yL + TH / 2 + 7.7; sq = 1.6
    w_in = tw_mm(ax, "Inside", FS_TINY); w_out = tw_mm(ax, "Outside", FS_TINY)
    tot = sq + 0.7 + w_in + 2.2 + sq + 0.7 + w_out; xk = gx + TW / 2 - tot / 2
    for c_, s_ in ((AOA_IN, "Inside"), (AOA_OUT, "Outside")):
        ax.add_patch(Rectangle((xk, yk - sq / 2), sq, sq, facecolor=c_, edgecolor=FRAME, lw=0.3, zorder=5))
        ax.text(xk + sq + 0.7, yk, s_, ha="left", va="center", fontsize=FS_TINY)
        xk += sq + 0.7 + (w_in if s_ == "Inside" else w_out) + 2.2
    # ---- 학습 전용 경로(회색 파선): ŷ_P(분기 → 아래 → 왼쪽 → ⊖), 관측 y(→ ⊖), y − ŷ_P(⊖ → 학습기)
    line(ax, [(xb_, yU), (xb_, yT), (xm + rm, yT)], color=TRAIN, ls=DASH, gap=(xb_, yL, 1.1))
    ax.text(xb_ + 1.0, yT - 1.0, r"$\hat{y}_{\mathrm{P}}$", ha="left", va="bottom", fontsize=FS_SMALL)
    line(ax, [(xr, yT), (xm - rm, yT)], color=TRAIN, ls=DASH)
    ax.text((xr + xm - rm) / 2, yT - 0.7, r"$y$", ha="center", va="bottom", fontsize=FS_SMALL)
    line(ax, [(xm, yT - rm), (xm, yL + hL / 2)], color=TRAIN, ls=DASH)
    ax.text(xm + 1.1, (yT - rm + yL + hL / 2) / 2, r"$y-\hat{y}_{\mathrm{P}}$", ha="left", va="center", fontsize=FS_SMALL)
    # ---- 열쇠(Output 영역 오른쪽 위): 흐름 · 학습 전용
    xk = 178.0; yk = R_TOP + 2.45
    w2 = tw_mm(ax, "Training only", FS_SMALL); w1 = tw_mm(ax, "Data flow", FS_SMALL); hl = 4.0
    x2 = xk - w2; ax.text(x2, yk, "Training only", ha="left", va="center", fontsize=FS_SMALL)
    ax.plot([x2 - 0.9 - hl, x2 - 0.9], [yk, yk], color=TRAIN, lw=LW_LINE, ls=DASH, solid_capstyle="butt", dash_capstyle="butt")
    x1 = x2 - 0.9 - hl - 2.6 - w1; ax.text(x1, yk, "Data flow", ha="left", va="center", fontsize=FS_SMALL)
    ax.plot([x1 - 0.9 - hl, x1 - 0.9], [yk, yk], color=INK, lw=LW_LINE, solid_capstyle="butt")
    ax.text(0.9, 1.2, "a", ha="left", va="top", fontsize=FS_LETTER, fontweight="bold")
    SRC["a"] = dict(inputs=["data/processed/v5/maps/Alaska/pred_%d.csv.gz (e5_tdd, p1 = Stefan recalibrated, r1 = physics + residual ML)" % yr,
                            "data/processed/v5/maps/Alaska/meta.json (lam, E, feats, learner)", "data/processed/v5/maps/Alaska/mean_2011_2024.csv.gz (aoa_in)",
                            A["forcing_src"] + " (e5_maat, e5_fdd, e5_swe, year %d)" % yr,
                            "data/processed/map_alaska/alaska_grid_x25_v1.csv.gz (dem_elev, dem_slope, sg_soc_0_5, sg_clay_5_15, land)",
                            "data/processed/v5/labels_cell_year_v5.csv (Alaska, year %d, in_forcing == 1: observed cells)" % yr],
                    years=dict(example_year=yr, applicability="area of applicability of the residual learner computed once from 2011–2024 mean climate + static covariates (scripts/3_deep_learning/v5_map_predict.py: DI threshold = 95th percentile of cross-block nearest-neighbour distances of training rows in standardised F8TS space); not year-specific; thumbnail box = inside when ≥ 50 % of its 1 km cells are inside"),
                    lam=lam, E_cm_per_sqrt_degC_day=float(meta["E"]), n_train=int(meta["n_train"]), learner=dict(name=meta["learner"], **learner_spec()),
                    n_features=len(feats), n_climate_features=n_clim, n_static_features=n_stat, features=feats,
                    thumbnails=dict(input_cmap="cmcrameri grayC_r 0.04–0.82 (one colormap for all eight input thumbnails), each stretched to its own 2–98th percentile (slope 0–98th)",
                                    alt_cmap="cmcrameri oslo_r 0.12–0.90, Physics ALT and ALT %d share one scale" % yr, alt_range_cm=[lo, hi], alt_ticks_cm=tk,
                                    residual="(r1 − p1)/λ = residual prediction before weighting; cmcrameri vik, symmetric", residual_range_cm=[-v, v],
                                    applicability_colours=dict(inside=AOA_IN, outside=AOA_OUT), bins_deg=[0.05, 0.02], n_map_cells=int(len(m)),
                                    observed_cells=int(o.loc_id.nunique()), observed_rows=int(len(o))),
                    encoding=dict(accent=f"{ACCENT}: residual-ML path (gradient boosting → residual → ⊕, applicability) and the ALT {yr} frame",
                                  training_only=f"dashed {TRAIN}: observed y → ⊖, ŷ_P → ⊖, y − ŷ_P → gradient boosting",
                                  boxes=f"face {BOX_FACE}, edge {BOX_EDGE} {LW_BOX} pt", connectors=f"{LW_LINE} pt lines with open chevrons",
                                  learner_inputs="bracket grouping the eight input thumbnails (all 23 variables) → gradient boosting; TDD → Stefan crosses the bracket"),
                    numbers_printed={"lambda": lam, "climate variables": n_clim, "static variables": n_stat, "example year": yr,
                                     "residual colour bar ticks (cm)": [-v, 0, v], "ALT colour bar ticks (cm)": tk})
    return dict(yU=yU, yL=yL, yT=yT)


# ================================================================ b 비교 구조 4종
def panel_b(fig, ax, y0):
    yc = y0 + 10.0
    names = ["Direct ML", "Physics-input ML", "Physics pseudo-labels", "Physics + residual ML"]
    cw = W_MM / 4; bw, bh = 11.0, 5.4; sw, sh = 11.0, 4.2; ys = yc - 8.0
    gw = {0: 9.0 + bw + 5.8 + 2.0, 1: 9.0 + bw + 5.8 + 2.0, 2: 9.0 + bw + 5.8 + 2.0, 3: 9.0 + bw + 8.0 + 1.5 + 5.8 + 2.0}   # X 왼쪽 끝 → ŷ 오른쪽 끝(mm)
    for i, nm in enumerate(names):
        x0 = i * cw + (cw - gw[i]) / 2 - 0.5
        ax.text(x0 + 3.0, yc, r"$X$", ha="right", va="center", fontsize=FS)
        xb = x0 + 9.0
        line(ax, [(x0 + 3.9, yc), (xb, yc)])
        box(ax, xb, yc - bh / 2, bw, bh, "ML", fs=FS)
        xe = xb + bw
        if i == 3:
            xp = xe + 8.0; r = 1.5
            line(ax, [(xe, yc), (xp - r, yc)], color=ACCENT)
            ax.text((xe + xp - r) / 2, yc - 0.8, r"$\lambda$", ha="center", va="bottom", fontsize=FS_SMALL)
            op_node(ax, xp, yc, r, "+")
            box(ax, xp - sw / 2, ys - sh / 2, sw, sh, "Stefan", fs=FS_SMALL)
            line(ax, [(xp, ys + sh / 2), (xp, yc - r)])
            line(ax, [(xp + r, yc), (xp + r + 5.0, yc)])
            ax.text(xp + r + 5.8, yc, r"$\hat{y}$", ha="left", va="center", fontsize=FS)
            xc = (x0 + xp + r + 7.5) / 2
        else:
            line(ax, [(xe, yc), (xe + 5.0, yc)])
            ax.text(xe + 5.8, yc, r"$\hat{y}$", ha="left", va="center", fontsize=FS)
            if i >= 1:
                cxm = xb + bw / 2
                lab_ = "Stefan, Kudryavtsev" if i == 1 else "Stefan"           # F1 은 Stefan(P1) · Kudryavtsev(Ku) 예측을 함께 특징으로 쓴다
                sw_ = max(sw, tw_mm(ax, lab_, FS_SMALL) + 2.4)
                box(ax, cxm - sw_ / 2, ys - sh / 2, sw_, sh, lab_, fs=FS_SMALL)
                if i == 1:
                    line(ax, [(cxm, ys + sh / 2), (cxm, yc - bh / 2)])
                else:
                    line(ax, [(cxm, ys + sh / 2), (cxm, yc - bh / 2)], color=TRAIN, ls=(0, (1.1, 0.8)))   # 짧은 연결선: 잘게 끊은 파선
            xc = (x0 + xe + 7.5) / 2
        ax.text(xc, yc + bh / 2 + 2.2, nm, ha="center", va="top", fontsize=FS)
    ax.text(0.9, y0 + 0.5, "b", ha="left", va="top", fontsize=FS_LETTER, fontweight="bold")
    SRC["b"] = dict(structures=names, encoding=dict(physics_input="recalibrated Stefan and Kudryavtsev predictions enter the learner as input features (solid line; testbed_v5.run_method F1)",
                                                    pseudo_labels="Stefan values used as additional training targets (dashed grey: training only)",
                                                    residual="learner output added to the Stefan anchor with weight λ (vermillion)"),
                    methods_code=dict(zip(names, ["D0", "F1", "D1", "R1"])), source="polar.testbed_v5.run_method", numbers_printed="none")


# ================================================================ c 다섯 분할(레나 델타)
def lena_folds():
    """실험 하네스(v5_compare.py)와 같은 경로로 레나 행 · 겹을 만들고, core_v6 예측 파일의 겹 배정과 같은지 확인한다."""
    df = TB.load_v5(V5, products=False)
    d_all = TB.with_forcing(df, "year_obs"); d_all = d_all[TB.usable_mask(d_all, None)].reset_index(drop=True)
    dr = d_all[d_all.region == "Lena"].reset_index(drop=True)
    F = {s: TB.make_folds(dr, s, 5, 0) for s in ("random", "block", "year", "block_year")}
    F["region"] = [f for f in TB.make_folds(d_all, "region", 5, 0) if f.name == "L_Lena"]
    check = {}
    for s in ("random", "block", "year", "block_year"):
        fp = V5 / "runs" / "core_v6" / "preds" / f"Lena__{s}__year_obs__F8TS__P1__-__lam-1.csv.gz"
        if fp.exists():
            p = pd.read_csv(fp, usecols=["fold", "cell_year_id"])
            mine = pd.Series({cid: f.name for f in F[s] for cid in dr.cell_year_id.values[f.te]})
            check[s] = bool(len(p) == len(mine) and (p.set_index("cell_year_id").fold.reindex(mine.index) == mine).all())
    fp = V5 / "runs" / "core_v6" / "preds" / "ALL__region__year_obs__F8TS__P1__-__lam-1.csv.gz"
    if fp.exists():
        p = pd.read_csv(fp, usecols=["fold", "cell_year_id"]); pl = p[p.fold == "L_Lena"]
        check["region"] = bool(set(pl.cell_year_id) == set(d_all.cell_year_id.values[F["region"][0].te]))
    assert all(check.values()), check
    return d_all, dr, F, check


def status_rows(n, te, tr):
    s = np.full(n, "excluded", dtype=object); s[tr] = "training"; s[te] = "test"; return s


def panel_c(fig, ax, y0):
    d_all, dr, F, check = lena_folds()
    pick = dict(random="r0", block="b0", year="y2016", block_year="b0y2016")
    show = [("Random", "random"), ("Spatial block (0.5°)", "block"), ("Year hold-out", "year"), ("Block × year", "block_year"),
            ("Region hold-out", "region")]
    ne = ne_layers("lena"); asp = 1.0 / np.cos(np.deg2rad(LENA_LAT)); ext = LENA_EXT
    mw = 32.4; gap = (W_MM - 2.0 - 5 * mw) / 4; mh = mw * (ext[3] - ext[2]) * asp / (ext[1] - ext[0])
    ytop = y0 + 8.4
    info = {}; sb_info = None
    for i, (title, s) in enumerate(show):
        x = 1.0 + i * (mw + gap)
        mx = fig.add_axes(rect(x, ytop, mw, mh))
        for g in ne["land"]:
            for part in getattr(g, "geoms", [g]):
                xs, ys = part.exterior.xy; mx.fill(xs, ys, facecolor=C_LAND, edgecolor="none", lw=0, zorder=1)
        for g in ne["coastline"]:
            for part in getattr(g, "geoms", [g]):
                xs, ys = part.xy; mx.plot(xs, ys, color=C_COAST, lw=0.3, zorder=2)
        if s in ("block", "block_year"):
            for gl in np.arange(np.ceil(ext[0] / 0.5) * 0.5, ext[1], 0.5):
                mx.plot([gl, gl], [ext[2], ext[3]], color="#CFCFCF", lw=0.3, zorder=1.5)
            for gl in np.arange(np.ceil(ext[2] / 0.5) * 0.5, ext[3], 0.5):
                mx.plot([ext[0], ext[1]], [gl, gl], color="#CFCFCF", lw=0.3, zorder=1.5)
        if s == "region":
            f = F["region"][0]; te_ids = set(d_all.cell_year_id.values[f.te]); st = np.where(dr.cell_year_id.isin(te_ids), "test", "excluded").astype(object)
            tr_reg = d_all.iloc[f.tr].region.value_counts().to_dict()
            fold_name = f.name
        else:
            f = [ff for ff in F[s] if ff.name == pick[s]][0]; st = status_rows(len(dr), f.te, f.tr); tr_reg = None; fold_name = f.name
        q = dr.assign(status=st)
        rank = {"excluded": 0, "training": 1, "test": 2}
        cell = q.assign(r=q.status.map(rank)).groupby("loc_id").agg(lon=("lon", "first"), lat=("lat", "first"), r=("r", "max"))
        for k, col, z in ((0, C_EXCL, 3), (1, C_TRAIN, 4), (2, C_TEST, 5)):
            cc = cell[cell.r == k]
            mx.scatter(cc.lon, cc.lat, s=2.2, color=col, lw=0, zorder=z, rasterized=True)
        mx.set_xlim(ext[0], ext[1]); mx.set_ylim(ext[2], ext[3]); mx.set_aspect(asp); mx.set_xticks([]); mx.set_yticks([])
        for sp in mx.spines.values():
            sp.set_visible(True); sp.set_edgecolor(INK); sp.set_linewidth(0.5)
        ax.text(x + mw / 2, ytop - 0.9, title, ha="center", va="bottom", fontsize=FS)
        if i == 0:                                                      # 축척(50 km, 첫 지도 왼쪽 위)
            km = 50.0; lat_sb = ext[2] + 0.13; dlon = km / (111.32 * np.cos(np.deg2rad(lat_sb))); lon_sb = ext[0] + 0.25
            mx.plot([lon_sb, lon_sb + dlon], [lat_sb, lat_sb], color=INK, lw=1.0, solid_capstyle="butt", zorder=8)
            mx.text(lon_sb + dlon / 2, lat_sb + 0.05, "50 km", ha="center", va="bottom", fontsize=FS_TINY, zorder=8)
            sb_info = dict(km=km, lon=[round(lon_sb, 3), round(lon_sb + dlon, 3)], lat=round(lat_sb, 3))
        info[title] = dict(scheme=s, fold_shown=fold_name,
                           rows=dict(test=int((st == "test").sum()), training=int((st == "training").sum()), excluded=int((st == "excluded").sum())),
                           cells=dict(test=int((cell.r == 2).sum()), training=int((cell.r == 1).sum()), excluded=int((cell.r == 0).sum())),
                           n_folds=len(F[s]), fold_names=[ff.name for ff in F[s]] if s != "region" else ["L_Lena"],
                           years_tested=sorted(int(y) for y in q[q.status == "test"].year.unique()),
                           **({"training_rows_other_regions": {k: int(v) for k, v in tr_reg.items()}} if tr_reg else {}))
    # 열쇠(패널 문자 줄, 오른쪽 맞춤)
    yk = y0 + 2.0; xk = W_MM - 1.0
    for nm_, col in (("Excluded", C_EXCL), ("Training", C_TRAIN), ("Test", C_TEST))[::1]:
        w = tw_mm(ax, nm_, FS_SMALL); xk -= w
        ax.text(xk, yk, nm_, ha="left", va="center", fontsize=FS_SMALL)
        ax.add_patch(Circle((xk - 1.6, yk), 0.75, facecolor=col, edgecolor="none", zorder=5)); xk -= 1.6 + 0.75 + 3.0
    ax.text(0.9, y0 + 0.5, "c", ha="left", va="top", fontsize=FS_LETTER, fontweight="bold")
    yrs = sorted(int(y) for y in dr.year.unique())
    SRC["c"] = dict(inputs=["data/processed/v5/labels_cell_year_v5.csv, forcing_year.csv, forcing_clim.csv, static_v5.csv via polar.testbed_v5.load_v5",
                            "polar.testbed_v5.with_forcing('year_obs') → usable_mask → region == 'Lena' → make_folds(k = 5, seed = 0)",
                            "check against data/processed/v5/runs/core_v6/preds/Lena__<scheme>__year_obs__F8TS__P1__-__lam-1.csv.gz and ALL__region__...: %s" % check],
                    years=dict(lena_labels=f"{yrs[0]}–{yrs[-1]}", year_list=yrs), n_rows=int(len(dr)), n_cells=int(dr.loc_id.nunique()),
                    n_blocks=int(dr.block.nunique()),
                    cell_status_rule="a cell is test if any of its cell-years is in the test fold, else training if any is in training, else excluded",
                    year_holdout_rule="years with < 20 Lena cell-years (2010, 2011, 2022) are never test years and stay in training",
                    block_year_rule="test = block fold i and year y; training = other block folds and other years; excluded = same fold other years or same year other folds",
                    region_rule="all Lena cell-years are test; training = Alaska and Canada (NW) cell-years more than 100 km from any Lena cell",
                    panels=info, colours=dict(test=C_TEST, training=C_TRAIN, excluded=C_EXCL), map_extent_lonlat=list(LENA_EXT),
                    scale_bar=sb_info, numbers_printed={"scale bar": "50 km", "titles": "Spatial block (0.5°)"})
    return info


# ================================================================ 점검
def _all_axes(fig):
    out = []
    def rec(a):
        out.append(a); [rec(c) for c in a.child_axes]
    for a in fig.axes:
        rec(a)
    return out


def collect_texts(fig):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); items = []
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            items.append((t.get_text(), t.get_window_extent(r), t.get_fontsize(), t.get_fontname(), None, t))
    for ax in _all_axes(fig):
        ts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                loc = tk.get_loc()
                if loc is None or not (lo - 1e-9 <= loc <= hi + 1e-9):
                    continue
                for lab in (tk.label1, tk.label2):
                    if lab.get_visible() and lab.get_text().strip():
                        ts.append(lab)
        lg = ax.get_legend()
        if lg is not None:
            ts += list(lg.get_texts())
        for t in ts:
            if t.get_visible() and t.get_text().strip():
                items.append((t.get_text(), t.get_window_extent(r), t.get_fontsize(), t.get_fontname(), ax, t))
    return items


def qa(fig, canvas_ax):
    """글자 크기(≥ 6 pt) · 글꼴 · 캔버스 안 · 글자끼리 겹침 · 글자와 썸네일/지도 틀 겹침 · 캔버스 글자와 선(여백 0.5 pt) 겹침."""
    items = collect_texts(fig); r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height; ppt = fig.dpi / 72.0
    bad = []
    for name, bb, fs, fn, ax, t in items:
        if fs < 6.0 - 1e-9:
            bad.append(f"size {fs} < 6: '{name}'")
        if fn != "FreeSans":
            bad.append(f"font {fn}: '{name}'")
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            bad.append(f"outside canvas: '{name}'")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            w = min(a.x1, b.x1) - max(a.x0, b.x0); h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > 0.5 and h > 0.5:
                bad.append(f"text overlap: '{items[i][0]}' x '{items[j][0]}'")
    frames = [a for a in _all_axes(fig) if a is not canvas_ax]
    for name, bb, fs, fn, ax, t in items:
        for fr in frames:
            if ax is fr or (ax is not None and ax in fr.child_axes):
                continue
            fb = fr.get_window_extent(r)
            w = min(bb.x1, fb.x1) - max(bb.x0, fb.x0); h = min(bb.y1, fb.y1) - max(bb.y0, fb.y0)
            if w > 0.5 and h > 0.5:
                bad.append(f"text over frame: '{name}'")
    m = 0.5 * ppt
    for name, bb, fs, fn, ax, t in items:
        if ax is not canvas_ax:
            continue
        for ln in canvas_ax.lines:
            xy = ln.get_xydata()
            seg = [np.c_[np.linspace(x0, x1, 60), np.linspace(y0, y1, 60)] for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:])] or [xy]
            pts = canvas_ax.transData.transform(np.concatenate(seg))
            if ((pts[:, 0] >= bb.x0 - m) & (pts[:, 0] <= bb.x1 + m) & (pts[:, 1] >= bb.y0 - m) & (pts[:, 1] <= bb.y1 + m)).any():
                bad.append(f"text touches line: '{name}'")
                break
        for p in canvas_ax.patches:
            if isinstance(p, FancyBboxPatch) and p.get_linestyle() not in ("-", "solid"):
                continue                                             # 영역 파선 사각형(머리 라벨이 안에 있다)
            pb = p.get_window_extent(r)
            if not (bb.x1 + m < pb.x0 or bb.x0 - m > pb.x1 or bb.y1 + m < pb.y0 or bb.y0 - m > pb.y1):
                inside = pb.x0 <= bb.x0 and bb.x1 <= pb.x1 and pb.y0 <= bb.y0 and bb.y1 <= pb.y1
                if not inside:
                    bad.append(f"text touches patch: '{name}'")
    sizes = sorted({round(float(it[2]), 2) for it in items}); fonts = sorted({it[3] for it in items})
    print(f"[QA] texts {len(items)}, sizes {sizes}, fonts {fonts}, problems {len(bad)}" + "".join("\n   " + b for b in bad))
    return dict(n_texts=len(items), font_sizes_pt=sizes, fonts=fonts, problems=bad)


def save(fig):
    OUT.mkdir(parents=True, exist_ok=True); out = []
    for ext in ("pdf", "svg", "png"):
        p = OUT / f"{STEM}.{ext}"; fig.savefig(p, dpi=600 if ext == "png" else None, facecolor="white"); out.append(p)
    return out


def main():
    setup()
    meta = json.loads((V5 / "maps" / "Alaska" / "meta.json").read_text()); meta["year"] = 2014
    lab = pd.read_csv(V5 / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    A = alaska_thumbs(2014, float(meta["lam"]))
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax = canvas(fig)
    panel_a(fig, ax, A, lab, meta)
    panel_b(fig, ax, 65.3)
    panel_c(fig, ax, 84.0)
    SRC["qa"] = qa(fig, ax)
    SRC["figure"] = dict(width_mm=W_MM, height_mm=H_MM, png_dpi=600, font="FreeSans", text_pt=FS, small_pt=FS_SMALL, min_pt=FS_TINY,
                         letter_pt=FS_LETTER, palette="POLAR_PALETTE=%s" % os.environ.get("POLAR_PALETTE", "deck"),
                         script="scripts/4_visualization/paper_v10/fig2_method.py")
    out = save(fig); plt.close(fig)
    sd = OUT / "source_data"; sd.mkdir(parents=True, exist_ok=True)
    (sd / f"{STEM}.json").write_text(json.dumps(SRC, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    for p in out + [sd / f"{STEM}.json"]:
        print(p)


if __name__ == "__main__":
    main()
