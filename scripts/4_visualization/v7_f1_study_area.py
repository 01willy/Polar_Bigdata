"""v7 F1 연구 지역·관측·기간(docs/FIGURE_CONTRACT_v7.md, DESIGN_BRIEF_v8_2026-10-15.md 2절 8쪽).
a 범북극 영구동토 확률(ESA CCI 1997–2021 평균, 투영 격자에 쌍선형 재표집)과 세 지역 상자, b–d 지역 지도(음영 바탕, 관측 셀 = 관측 연도 수,
  0.5° 블록, 축척, b·c 는 밀집부 확대 창, 크기 열쇠는 b 에 한 번),
e·f 자료 목록(한 연도 축 1997–2025 공유, 큰 눈금 2000–2025 5년 + 해마다 작은 눈금): e 연도별 셀-연도 누적 막대(지역 색, 선형 축; 열쇠에 마지막 관측 월 비율 한 줄),
  f 자료별 기간 띠(행 묶음 Labels 지역 3행(관측 있는 해만) · Climate ERA5-Land · Satellite MODIS/InSAR(위성 상태색의 밝은·어두운 두 명도 SAT_SHADE: 레나 델타 #3E92CC 와
  CIELAB ΔE ≥ 20, color_qa 가 확인) · Static 지형·토양(회색 띠, 'static') · Products 제품 6종(제품 색, CCI v4/v5 두 명도, Aalto 는 기간 평균이라 윤곽 띠)), 분석 창 2010–2024 음영.
  캡션에 적을 부호: 윤곽 띠 = 기간 평균 제품, CCI·Wei·Aalto 는 2000년 이전 시작(작은 눈금으로 읽는다), e 는 선형 축(연도별 표는 source_data).
원천  outputs/figures/paper/v7/source_data/F1_study_area.json(e 표·f 띠·월 비율·색) + F1_e_cell_years.csv(해 × 지역 셀-연도).
입력  data/processed/v5/labels_cell_year_v5.csv(in_forcing==1), cci_pfr_mean_1997_2021.nc, map_<key>/hillshade_v1/{shade,dem}_250m.tif,
      v5/forcing_meta.json, v5/modis_cell_year_v5_meta.json, v5/products_cell_year_v5_meta.json. InSAR ALT 연도(2017)만 docs 표에서 옮긴 상수.
실행  OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v7_f1_study_area.py [--fig 1] [--slide | --deck]   (CPU 만)
산출  outputs/figures/paper/v7/F1_study_area.{pdf,svg,png}, --slide: v7/slide/F1_study_area.png + 패널 크롭(a b c d e f bcd ef)
      --deck: v7/deck/F1_study_area__deck_<N01|N05|N06>.png (300 dpi, RGB, 폭 12.0 in). N01·N06 = 12.0 × 4.65 in(3600 × 1395 px),
        N05 = 12.0 × 4.35 in(3600 × 1305 px; 세 지도의 가로세로 비가 넓어 한 줄 배치).
        N01 = v7 보고 1쪽 구성 복원(10-15 v9 항 9): a 범북극 지도 + b 지역별 연도별 셀-연도(지역마다 독립 y 축의 작은 막대 3단, 지역 이름·마지막 관측 월 비율).
        N05 = a 알래스카(크게) | b 레나 델타 | c 캐나다(같은 높이, 위 맞춤) + 아래 띠에 크기 열쇠 한 줄(10-15 v9 항 11: 오른쪽 빈 여백 제거).
        10-16: 확대 창 자리 재조정. b 레나 = 왼쪽 아래 관측 없는 육지(높이 0.31; 옛 0.40 은 밀집부 369 셀과 확대 상자를 덮었다),
        a 알래스카 = 왼쪽 위, 높이 0.38, 틀 위에서 0.10(창 아래 관측 0, 위 연결선이 배로 밀집부 아래로). 창 아래 관측 셀 수를 출력하고 덱에서 0 을 확인한다.
        N06 = e + f 자료 목록 한 패널(패널 문자 없음). 덱 글자: 축 라벨 14, 눈금·직접 라벨 12, 패널 문자 14 굵게(슬라이드 안 순서 a, b, c).
개정(10-15 v9): 지역색 토큰 변경(레나 #3E92CC 등) 반영, f 의 묶음 이름 'Forcing' → 'Climate'(강제력 표현 금지, 항 7). 위성 띠(MODIS·InSAR)는
      상태색 #56B4E9 의 어두운 명도라 새 레나 색과의 ΔE 를 color_qa 가 확인한다(≥ 20).
구조  load() 자료 → draw_a/draw_region/draw_canada/draw_bars/draw_bands/draw_inventory(그리기, Axes 단위) → f1()(논문·슬라이드 배치) / deck()(덱 배치). 자료 논리는 한 곳.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3"))
import polar.style_v5 as S
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
from matplotlib.patches import ConnectionPatch, Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio import features
from pyproj import Transformer
from shapely.ops import transform as shp_transform
from scipy.ndimage import map_coordinates
import maps_alt_v3 as MV

PROC = ROOT / "data" / "processed"; V5 = PROC / "v5"
OUT = ROOT / "outputs" / "figures" / "paper" / "v7"
REG = ["Alaska", "Lena", "Canada"]
PC = ccrs.PlateCarree()
# 지역 상자(경도 0, 경도 1, 위도 0, 위도 1): a 의 상자와 b–d 의 표시 범위
BOX = {"Alaska": (-168.0, -141.0, 59.0, 71.5), "Lena": (123.3, 130.1, 71.5, 73.6), "Canada": (-141.0, -112.0, 60.0, 70.0)}
# a 의 표시 상자. 레나 델타는 범위(6.8° × 2.1°)가 범북극 축척에서 약 2 mm 라 같은 선 두께로 그리면 마름모 표식처럼 보이므로,
# 중심(126.7°E, 72.55°N)을 유지한 채 약 2.3 배 넓힌 윤곽 상자로 그린다(실제 범위는 c 패널). 알래스카·캐나다는 BOX 그대로
BOX_A = dict(BOX); BOX_A["Lena"] = (118.7, 134.7, 70.0, 75.1)
EXT = {"Alaska": (-167.5, -140.5, 60.3, 71.6), "Lena": (123.3, 130.1, 71.4, 73.6), "Canada": (-142.0, -112.0, 60.0, 70.0)}
ZOOM = {"Alaska": (-151.3, -147.7, 68.3, 70.55), "Lena": (125.65, 127.05, 72.24, 72.55)}   # 확대 창(관측 밀집부)
STEP = {"Alaska": 3, "Lena": 1}   # 음영 250 m 픽셀 묶음
GRID = {"Alaska": ([-170, -160, -150, -140], [60, 65, 70]), "Lena": ([124, 126, 128, 130], [72, 73]), "Canada": ([-140, -130, -120], [62, 65, 68])}
GRID_DECK = dict(GRID, Lena=([124, 126, 128], [72, 73]))   # 덱: 레나 오른쪽 끝 '130°E' 라벨이 틀 밖으로 나가 c 와의 간격을 줄이므로 뺀다
SB = {"Alaska": 200, "Lena": 50, "Canada": 200}
SB_POS = {"Alaska": (0.04, 0.05), "Lena": (0.705, 0.92)}   # 축척 위치(패널 분율): 레나는 오른쪽 위 바다(128°E·130°E 자오선 사이, 관측 없음)
PAD = {"Alaska": 1.5e4, "Lena": 2e3}   # 투영 상자 여유(m)
LAND = "#efefef"; WATER = "#e1e4e8"; COAST = "#8a8a8a"; EDGE = "#333333"; FRAME = "#555555"; TEXT = "#333333"
W_SHADE = 0.45
INSAR_YEAR = 2017   # ABoVE ReSALT InSAR ALT(알래스카), docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md 자료 표
SZ0 = {1: 2.0, 2: 5.5, 4: 24.0}   # 관측 연도 수 등급별 마커 면적(pt², 논문): 1년 점 < 2–3년 원 < ≥ 4년 별(단조). 범례도 같은 값에서 만든다
A_SMALL = 0.6   # 1년·2–3년 마커 면 투명도(밀집부에서 밀도가 보이게); 외곽선은 흰색, 별은 불투명·어두운 외곽선으로 맨 위
W_MM = S.W2; H_MM = 172.0   # 캔버스(mm). f1()·deck() 이 배치 계산으로 덮어쓴다
XA = 4.0; G_AB = 10.0   # a 왼쪽 여백, a–b 상자 최소 간격(mm). b 위도 라벨(≈ 6 mm)이 간격 안에 있고 슬라이드 b 크롭이 그 라벨에서 시작하므로
                        # a 의 '0°' 라벨(원 밖 ≈ 2.4 mm)이 크롭에 들어오지 않으려면 ≥ 9.5. 원 지름 D 는 f1() 이 남는 폭·세로 한도로 정한다
DECK_W_IN = 12.0; DECK_H_IN = {"N01": 4.65, "N05": 4.35, "N06": 4.65}   # 덱 본문(in, 설계 지침 v9 1.4). N05 는 지도 비율 때문에 4.35
_SHADE = {}; _PROJ = {}

# ---------------- 글자 크기(pt)·선 배율: 논문 / 덱. 덱은 12 in 폭을 300 dpi 로 그대로 저장하므로 pt 를 직접 지정한다
FS_PAPER = dict(letter=8, tick=6, cb_label=6.5, grat=6, a_lbl=5.5, a_box=6, sb=6, sb_in=5.5, legend=6, region=6.5, share=5.5, band=6, head=6, group=6)
FS_DECK = dict(letter=14, tick=12, cb_label=14, grat=12, a_lbl=12, a_box=12, sb=12, sb_in=12, legend=12, region=14, share=12, band=12, head=12, group=12)
XL_INV = (1996.5, 2025.5); XT_INV = [2000, 2005, 2010, 2015, 2020, 2025]   # e·f 공유 연도 축(해 y 는 [y − 0.5, y + 0.5]); CCI 1997 시작이 축 안에 들어온다
WINDOW = (2010, 2024)   # 분석 창(forcing_meta years 와 같다; load() 에서 확인)
XT_MINOR = list(range(XT_INV[0] - 3, XT_INV[-1] + 1))   # 해마다 작은 눈금(1997 부터): 2000 이전에 시작하는 띠(CCI·Wei·Aalto)의 시작 해를 읽는 기준
BAND_H = 0.64; GROUP_GAP = 0.4   # f 띠 높이·묶음 사이 간격(행 간격 단위)
SAT_SHADE = {"MODIS": 0.35, "InSAR": -0.60}   # 위성 상태색(토큰 #56B4E9)의 두 명도. 10-15 지역색 변경(레나 #3E92CC) 뒤 원색은 레나와 ΔE 13, 옛 −0.35 명도는 17 이라
                                               # MODIS 는 밝게(+0.35, #91CEF1: 레나 ΔE 26 · ERA5-Land 39), InSAR 는 어둡게(−0.60, #22485D: 레나 36 · ERA5-Land 31)
FS = dict(FS_PAPER); K = 1.0; MODE = "paper"
SZ = dict(SZ0); HALO = [pe.withStroke(linewidth=1.6, foreground="white")]


def set_mode(mode):
    """paper: 토큰 논문 rc(S.setup). deck: 토큰 slide 선 두께·글자(축 라벨 14, 눈금 12)와 선·마커 배율 K."""
    global FS, K, MODE, SZ, HALO
    MODE = mode
    if mode == "deck":
        FS = dict(FS_DECK); K = 1.8
        Ls = S.TOK["lines"]["slide"]; Fs = S.TOK["font"]["slide"]
        plt.rcParams.update({"font.size": Fs["axis_label_pt"], "axes.labelsize": Fs["axis_label_pt"], "xtick.labelsize": Fs["tick_pt"], "ytick.labelsize": Fs["tick_pt"],
                             "legend.fontsize": Fs["tick_pt"], "axes.linewidth": Ls["axis_pt"], "xtick.major.width": Ls["tick_pt"], "ytick.major.width": Ls["tick_pt"],
                             "xtick.major.size": Ls["tick_len_pt"], "ytick.major.size": Ls["tick_len_pt"], "lines.linewidth": Ls["data_pt"], "lines.markersize": Ls["marker_pt"]})
    else:
        FS = dict(FS_PAPER); K = 1.0
    SZ = {k: v * K * K for k, v in SZ0.items()}
    HALO = [pe.withStroke(linewidth=1.6 * K, foreground="white")]


def R(x, y_top, w, h):
    """mm 사각형(왼쪽, 위, 폭, 높이) → figure 분율."""
    return [x / W_MM, 1 - (y_top + h) / H_MM, w / W_MM, h / H_MM]


def panel_letter(ax, s, x=-0.12, y=1.04):
    """패널 문자(논문 8 pt, 덱 14 pt 굵게). S.PANELS 에 묶어 슬라이드 크롭에 쓴다."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=FS["letter"], fontweight="bold", va="bottom", ha="left")
    S.PANELS.setdefault(s, []).append(ax)


def letter(ax, s, rect_mm, dx_mm=-6.5, dy_mm=1.0):
    panel_letter(ax, s, x=dx_mm / rect_mm[2], y=1 + dy_mm / rect_mm[3])


def proj_of(r):
    if r not in _PROJ:
        _PROJ[r] = ccrs.epsg(3338) if r == "Alaska" else ccrs.NorthPolarStereo(central_longitude=126.7, true_scale_latitude=70.0)
    return _PROJ[r]


def proj_box(proj, lon0, lon1, lat0, lat1, pad=0.0):
    lons = np.r_[np.linspace(lon0, lon1, 60), np.full(60, lon1), np.linspace(lon1, lon0, 60), np.full(60, lon0)]
    lats = np.r_[np.full(60, lat0), np.linspace(lat0, lat1, 60), np.full(60, lat1), np.linspace(lat1, lat0, 60)]
    p = proj.transform_points(PC, lons, lats)
    return [p[:, 0].min() - pad, p[:, 0].max() + pad, p[:, 1].min() - pad, p[:, 1].max() + pad]


def region_box(r):
    """b·c 의 투영 표시 상자(m)."""
    return proj_box(proj_of(r), *EXT[r], pad=PAD[r])


def region_aspect(r):
    """패널 높이/폭 비(캐나다는 65°N 기준 등장방형)."""
    if r == "Canada":
        lo0, lo1, la0, la1 = EXT["Canada"]; return (la1 - la0) / ((lo1 - lo0) * np.cos(np.radians(65.0)))
    b = region_box(r); return (b[3] - b[2]) / (b[1] - b[0])


def _shade_full(key):
    """250 m 음영(0–1)과 육지(Natural Earth 10m) 마스크를 음영 CRS 전체 해상도로 읽어 캐시한다."""
    if key in _SHADE:
        return _SHADE[key]
    with rasterio.open(PROC / f"map_{key}" / "hillshade_v1" / "shade_250m.tif") as src:
        shade = src.read(1).astype(np.float32) / 255.0; tr = src.transform; shape = (src.height, src.width); crs = src.crs
    with rasterio.open(PROC / f"map_{key}" / "hillshade_v1" / "dem_250m.tif") as src:
        has_dem = np.isfinite(src.read(1))
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    geoms = [shp_transform(tf.transform, g) for g in MV.ne_layers(key)["land"]]
    land = features.rasterize([(g, 1) for g in geoms], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(np.float32)
    lb = land > 0.5
    if (lb & ~has_dem).any():   # DEM 타일 밖 육지: 밝기를 음영 육지 평균에 맞춰 경계가 드러나지 않게 한다
        shade[lb & ~has_dem] = float(shade[lb & has_dem].mean())
    _SHADE[key] = (shade, land, tr, crs)
    return _SHADE[key]


def shade_rgba(key, step, box=None):
    """음영만 입힌 육지 RGBA(값 없음). box=(x0,x1,y0,y1, 투영 m) 이면 그 창만 잘라 낸다. 반환 rgba, extent(투영 m), crs."""
    shade, land, tr, crs = _shade_full(key)
    r0, c0 = 0, 0
    if box is not None:
        c0 = int(max(0, np.floor((box[0] - tr.c) / tr.a)) - 2); c1 = int(np.ceil((box[1] - tr.c) / tr.a) + 2)
        r0 = int(max(0, np.floor((box[3] - tr.f) / tr.e)) - 2); r1 = int(np.ceil((box[2] - tr.f) / tr.e) + 2)
        r0 = max(r0, 0); c0 = max(c0, 0); shade = shade[r0:r1, c0:c1]; land = land[r0:r1, c0:c1]
    H = shade.shape[0] // step * step; W = shade.shape[1] // step * step
    shade = shade[:H, :W]; land = land[:H, :W]
    if step > 1:
        shade = shade.reshape(H // step, step, W // step, step).mean((1, 3))
        land = land.reshape(H // step, step, W // step, step).mean((1, 3))
    base = np.array(to_rgb(LAND), np.float32)
    rgba = np.empty(shade.shape + (4,), np.float32)
    rgba[..., :3] = base[None, None, :] * ((1 - W_SHADE) + W_SHADE * shade)[..., None]
    rgba[..., 3] = np.clip(land, 0, 1)
    x0 = tr.c + c0 * tr.a; y0 = tr.f + r0 * tr.e
    extent = [x0, x0 + W * tr.a, y0 + H * tr.e, y0]
    return rgba, extent, crs


def pfr_image(proj, lat_min=52.0, n=1400):
    """ESA CCI 영구동토 확률(0.1°)을 투영 격자(n×n, 원 지름 = n 픽셀)에 쌍선형 재표집한다. 반환 masked 배열, extent(투영 m)."""
    import xarray as xr
    ds = xr.open_dataset(PROC / "cci_pfr_mean_1997_2021.nc"); P = ds[list(ds.data_vars)[0]]
    latn = [d for d in P.dims if "lat" in d][0]; lonn = [d for d in P.dims if "lon" in d][0]
    lat = P[latn].values.astype(np.float64); lon = P[lonn].values.astype(np.float64); V = P.values.astype(np.float32)
    if lat[0] > lat[-1]:
        lat = lat[::-1]; V = V[::-1]
    dlat = (lat[-1] - lat[0]) / (len(lat) - 1); dlon = (lon[-1] - lon[0]) / (len(lon) - 1)
    valid = np.isfinite(V).astype(np.float32); Vf = np.where(np.isfinite(V), V, 0.0).astype(np.float32)
    Vf = np.c_[Vf, Vf[:, :1]]; valid = np.c_[valid, valid[:, :1]]   # 경도 둘레 이음
    px, py = proj.transform_point(proj.proj4_params.get("lon_0", 0.0), lat_min, PC); Rm = float(np.hypot(px, py))
    xs = np.linspace(-Rm, Rm, n); X, Y = np.meshgrid(xs, xs)
    ll = PC.transform_points(proj, X.ravel(), Y.ravel()); lo, la = ll[:, 0], ll[:, 1]
    fi = (la - lat[0]) / dlat; fj = ((lo - lon[0]) % 360.0) / dlon
    v = map_coordinates(Vf, [fi, fj], order=1, mode="nearest"); w = map_coordinates(valid, [fi, fj], order=1, mode="nearest")
    out = np.where(w > 0.5, v / np.maximum(w, 1e-6), np.nan)
    out[(la < lat_min) | (la > lat[-1] + dlat / 2) | (la < lat[0])] = np.nan
    return np.ma.masked_invalid(out.reshape(n, n)), [-Rm, Rm, -Rm, Rm]


def compose(paths, out, gap_mm=6.0):
    """슬라이드 패널 크롭 여러 장을 같은 배율로 가로로 이어 붙인다(위 맞춤, 300 dpi 메타)."""
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    gap = int(round(gap_mm / 25.4 * 300)); W = sum(i.width for i in ims) + gap * (len(ims) - 1); H = max(i.height for i in ims)
    canvas = Image.new("RGB", (W, H), (255, 255, 255)); x = 0
    for im in ims:
        canvas.paste(im, (x, 0)); x += im.width + gap
    canvas.save(out, dpi=(300, 300)); return out


def scalebar(ax, x0, y0, length, label, dy, lw=1.0, fs=None, ha="center"):
    """축척 막대 + 라벨. ha='left' 면 라벨을 막대 시작점에 왼쪽 맞춤(좁은 확대 창에서 라벨이 틀 밖으로 나가지 않게)."""
    ax.plot([x0, x0 + length], [y0, y0], color="black", lw=lw * K, zorder=8, solid_capstyle="butt", clip_on=False)
    ax.text(x0 if ha == "left" else x0 + length / 2, y0 + dy, label, ha=ha, va="bottom", fontsize=FS["sb"] if fs is None else fs, zorder=8, path_effects=HALO)


def marker_kw(k, color):
    """관측 연도 수 등급(1, 2, 4) → scatter 인자. 범례 손잡이도 이 함수로 만든다(크기 대응 보장)."""
    if k == 1:
        return dict(s=SZ[1], marker="o", facecolors=[to_rgba(color, A_SMALL)], edgecolors=[(1.0, 1.0, 1.0, 0.9)], linewidths=0.15 * K, zorder=5)
    if k == 2:
        return dict(s=SZ[2], marker="o", facecolors=[to_rgba(color, A_SMALL)], edgecolors=[(1.0, 1.0, 1.0, 0.9)], linewidths=0.25 * K, zorder=6)
    return dict(s=SZ[4], marker="*", c=color, linewidths=0.3 * K, edgecolors=EDGE, zorder=7)


def legend_kw(k):
    """크기 열쇠 손잡이: 지도 마커와 같은 크기(SZ)·모양. 1년·2–3년은 흰 바탕 위에서도 보이도록 불투명 회색 면·어두운 외곽선(단조 사다리가 드러나게)."""
    kw = marker_kw(k, "#555555")
    if k != 4:
        kw.update(facecolors=["#555555"], edgecolors=[EDGE], linewidths=0.3 * K)
    return kw


def cells_scatter(ax, c, color, proj=None):
    """관측 셀: 1년 = 점, 2–3년 = 원, ≥ 4년 = 별(모두 어두운 외곽선). 좌표는 proj 투영(없으면 경위도)."""
    if proj is not None:
        p = proj.transform_points(PC, c.lon.values, c.lat.values); x, y = p[:, 0], p[:, 1]
    else:
        x, y = c.lon.values, c.lat.values
    n = c.n_years.values
    for k, m in ((1, n == 1), (2, (n >= 2) & (n <= 3)), (4, n >= 4)):
        if m.any():
            ax.scatter(x[m], y[m], rasterized=(k == 1), **marker_kw(k, color))


def block_grid(ax, lon0, lon1, lat0, lat1, lw=0.2, alpha=0.5):
    xs = np.arange(np.floor(lon0 / 0.5) * 0.5, lon1 + 0.01, 0.5); ys = np.arange(np.floor(lat0 / 0.5) * 0.5, lat1 + 0.01, 0.5)
    ax.gridlines(crs=PC, draw_labels=False, xlocs=xs, ylocs=ys, linewidth=lw * K, color="#ffffff", alpha=alpha, zorder=3)


def graticule(ax, xlocs, ylocs):
    gl = ax.gridlines(crs=PC, draw_labels=True, xlocs=xlocs, ylocs=ylocs, linewidth=0.4 * K, color="#9a9a9a", alpha=0.9, linestyle=(0, (4, 3)), zorder=4,
                      x_inline=False, y_inline=False, rotate_labels=False)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"size": FS["grat"], "color": "#333333"}; gl.ylabel_style = {"size": FS["grat"], "color": "#333333"}
    gl.xpadding = 3 * K; gl.ypadding = 3 * K
    return gl


def zoom_inset(fig, ax, proj, zb, rect_mm, side, draw_base, cells, color, sb_km):
    """확대 창: 주 지도에 사각형(zb, 투영 m), rect_mm 에 같은 바탕·셀을 그린 축, 마주 보는 모서리 연결선 2개.
    side: 창이 사각형의 어느 쪽에 있는가('left' = 창이 왼쪽, 'top' = 창이 위, 'below' = 창이 왼쪽 아래)."""
    ax.add_patch(Rectangle((zb[0], zb[2]), zb[1] - zb[0], zb[3] - zb[2], fill=False, edgecolor=EDGE, lw=0.5 * K, zorder=8))
    axi = fig.add_axes(R(*rect_mm), projection=proj); axi.set_facecolor(WATER)
    axi.set_xlim(zb[0], zb[1]); axi.set_ylim(zb[2], zb[3])
    draw_base(axi)
    cells_scatter(axi, cells, color, proj)
    axi.spines["geo"].set_linewidth(0.6 * K); axi.spines["geo"].set_edgecolor(EDGE)
    L = sb_km * 1000.0; w = zb[1] - zb[0]; h = zb[3] - zb[2]
    scalebar(axi, zb[0] + 0.06 * w, zb[2] + 0.05 * h, L, f"{sb_km} km", dy=0.015 * h, lw=0.8, fs=FS["sb_in"], ha="left" if MODE == "deck" else "center")
    if side == "left":    # 창 오른쪽 모서리 ↔ 사각형 왼쪽 모서리
        pairs = [((zb[0], zb[3]), (1, 1)), ((zb[0], zb[2]), (1, 0))]
    elif side == "below":   # 창이 사각형 왼쪽 아래: 창 위 모서리 ↔ 사각형 아래 모서리(가는 회색 직선)
        pairs = [((zb[0], zb[2]), (0, 1)), ((zb[1], zb[2]), (1, 1))]
    else:                 # 창 아래 모서리 ↔ 사각형 위 모서리
        pairs = [((zb[0], zb[3]), (0, 0)), ((zb[1], zb[3]), (1, 0))]
    for (xa, ya), (xb, yb) in pairs:
        con = ConnectionPatch(xyA=(xa, ya), coordsA=ax.transData, xyB=(xb, yb), coordsB=axi.transAxes, color="#777777", lw=0.4 * K, zorder=2.5)
        fig.add_artist(con)
    return axi


# ======================================================================================= 자료
def load():
    """라벨(in_forcing==1)·셀 표·연도·메타. 논문·슬라이드·덱이 같은 객체를 쓴다."""
    lab = pd.read_csv(V5 / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1].copy()
    lab["obs_month_last"] = lab.obs_months.astype(str).map(lambda v: 9 if v == "eos" else max(int(x) for x in v.split(";") if x.strip().isdigit()))
    cells = lab.groupby("loc_id").agg(lat=("lat", "first"), lon=("lon", "first"), region=("region", "first"), n_years=("year", "nunique")).reset_index()
    n_cells = cells.groupby("region").size()
    fmeta = json.loads((V5 / "forcing_meta.json").read_text()); mmeta = json.loads((V5 / "modis_cell_year_v5_meta.json").read_text())
    pmeta = json.loads((V5 / "products_cell_year_v5_meta.json").read_text())["products"]
    y_frc = (min(fmeta["years"]), max(fmeta["years"])); y_mod = (min(mmeta["years_vi"]), max(mmeta["years_vi"]))
    assert y_frc == WINDOW, (y_frc, WINDOW)
    years = np.arange(WINDOW[0], WINDOW[1] + 1)
    cnt = lab.groupby(["year", "region"]).size().unstack("region").reindex(index=years, columns=REG).fillna(0).astype(int)   # e 누적 막대(해 × 지역)
    share = {r: lab[lab.region == r].obs_month_last.value_counts(normalize=True).reindex([8, 9, 10], fill_value=0) for r in REG}   # 마지막 관측 월 비율
    sat = S.STATUS["satellite"]; sat_m = shade(sat, SAT_SHADE["MODIS"]); sat_i = shade(sat, SAT_SHADE["InSAR"])
    # f 행: (묶음, 이름, [(y0, y1), ...] 해 구간(끝 포함), 색, 종류). Labels 는 관측이 있는 해만(캐나다 2020 빈 해가 드러난다)
    rows = [("Labels", S.REGION_EN[r], year_runs(cnt.index[cnt[r] > 0]), S.REGION_HEX[r], "band") for r in REG]
    rows += [("Climate", "ERA5-Land", [y_frc], S.STATUS["climate"], "band"),     # 묶음 이름: 'Forcing' 대신 'Climate'(지침 v9 항 7)
             ("Satellite", "MODIS", [y_mod], sat_m, "band"), ("Satellite", "InSAR ALT (Alaska)", [(INSAR_YEAR, INSAR_YEAR)], sat_i, "band"),
             ("Static", "Terrain, soil", [], S.NEUTRAL["no_data"], "static")]
    rows += [("Products", S.PRODUCT_EN[k] + sfx, [tuple(pmeta[k]["period"])], S.PRODUCT_HEX[k], kind)
             for k, sfx, kind in (("cci4y", "", "band"), ("cci5y", "", "band"), ("wei", "", "band"), ("liu", "", "band"), ("yk", " (Alaska)", "band"), ("aalto", " (mean)", "mean"))]
    color_qa(rows)
    return SimpleNamespace(lab=lab, cells=cells, n_cells=n_cells, years=years, cnt=cnt, share=share, rows=rows)


def cielab(hex_):
    """sRGB → CIELAB(D65). 띠 색 분리 점검(ΔE)에만 쓴다."""
    c = np.array(to_rgb(hex_)); c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ c / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def color_qa(rows, min_de=20.0):
    """f 띠 색의 쌍별 CIELAB ΔE(1976). 가장 가까운 쌍을 출력하고, 위성 두 행이 라벨 세 행과 ΔE ≥ min_de 인지 확인한다."""
    cols = [(r[1], r[3]) for r in rows if r[4] != "static"]
    pairs = sorted(((float(np.linalg.norm(cielab(a[1]) - cielab(b[1]))), a[0], b[0]) for i, a in enumerate(cols) for b in cols[i + 1:]))
    print("band colour ΔE, closest pairs: " + "; ".join(f"{a} – {b} {d:.0f}" for d, a, b in pairs[:5]))
    labels = {r[1]: r[3] for r in rows if r[0] == "Labels"}; sats = {r[1]: r[3] for r in rows if r[0] == "Satellite"}
    for sn, sc in sats.items():
        for ln, lc in labels.items():
            de = float(np.linalg.norm(cielab(sc) - cielab(lc))); assert de >= min_de, (sn, ln, de)


def write_source(d):
    """e 표·f 띠·월 비율·색을 source_data 에 적는다(e 막대는 선형 축이라 작은 지역의 해별 값은 표로 읽는다)."""
    sd = OUT / "source_data"; sd.mkdir(parents=True, exist_ok=True)
    tab = d.cnt.copy(); tab.index.name = "year"; tab["total"] = tab.sum(axis=1); tab.to_csv(sd / "F1_e_cell_years.csv")
    src = {"e_cell_years_csv": "F1_e_cell_years.csv", "e_axis": "linear; 2014 Alaska campaign dominates, read small regions from the CSV",
           "n_cells": {r: int(d.n_cells[r]) for r in REG}, "window": list(WINDOW), "x_range": list(XL_INV), "x_major_ticks": XT_INV, "x_minor_ticks": "yearly from 1997",
           "last_month_share": {r: {str(m): round(float(d.share[r][m]), 4) for m in (8, 9, 10)} for r in REG},
           "f_rows": [{"group": g, "name": n, "years": [list(s) for s in segs] if segs else "static (all years)", "hex": c,
                       "encoding": {"band": "filled band = yearly values", "mean": "outline = one multi-year mean value", "static": "grey band = time-invariant input"}[k]}
                      for g, n, segs, c, k in d.rows],
           "satellite_shades": SAT_SHADE, "insar_year": INSAR_YEAR, "labels_csv": "data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1)",
           "meta": ["data/processed/v5/forcing_meta.json", "data/processed/v5/modis_cell_year_v5_meta.json", "data/processed/v5/products_cell_year_v5_meta.json"]}
    (sd / "F1_study_area.json").write_text(json.dumps(src, ensure_ascii=False, indent=1, default=float))


def year_runs(ys):
    """정렬된 해 목록 → 연속 구간 [(y0, y1), ...]."""
    ys = [int(y) for y in ys]; out = []
    for y in ys:
        if out and y == out[-1][1] + 1:
            out[-1] = (out[-1][0], y)
        else:
            out.append((y, y))
    return out


def shade(hex_, f):
    """같은 색상의 다른 명도: f < 0 어둡게(검정 쪽), f > 0 밝게(흰색 쪽)."""
    c = np.array(to_rgb(hex_)); c = c * (1 + f) if f < 0 else c + (1 - c) * f
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in np.clip(c, 0, 1))


def text_mm(s, pt):
    """FreeSans 글자 폭(mm). 열 배치(열쇠 이름 열, 행 이름·묶음 열)에 쓴다."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    return TextPath((0, 0), s, size=pt, prop=FontProperties(family=S.F["latin"])).get_extents().width * 25.4 / 72


# ======================================================================================= 패널 그리기(Axes 단위)
def draw_a(fig, rect, cb_rect, n_cells, lt="a", ref=True, lt_xy=(-0.02, 0.98), canada_side="below", lat_lon=170.0):
    """a 범북극: rect=(x, y, D, D) mm 원, cb_rect 컬러바(mm). ref=True 면 지역 이름 앞에 b·c·d 패널 문자(논문). lt_xy 패널 문자 위치(축 분율).
    canada_side: 캐나다 이름·셀 수의 자리('below' = 상자 아래(논문), 'right' = 상자 오른쪽 아래(덱: 원 아래 '120°W' 경도 라벨과 겹치지 않게)).
    lat_lon: 위도 라벨을 놓을 자오선(°E). 논문 170°E(왼쪽 가로), 덱 35°E(오른쪽 위 대각선: 12 pt 라벨이 가로로 붙지 않고 바렌츠해·비영구동토 육지 위). 반환 (축, 컬러바 축)."""
    proj_a = ccrs.NorthPolarStereo(central_longitude=-100)
    img_a, ext_a = pfr_image(proj_a, lat_min=52.0, n=1800)   # 원 지름 73 mm × 600 dpi ≈ 1,720 px
    axm = fig.add_axes(R(*rect), projection=proj_a); axm.set_extent([-180, 180, 52, 90], PC)
    th = np.linspace(0, 2 * np.pi, 240); axm.set_boundary(mpath.Path(np.c_[0.5 + 0.5 * np.cos(th), 0.5 + 0.5 * np.sin(th)]), transform=axm.transAxes)
    axm.set_facecolor(WATER)
    axm.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f6f6f6", edgecolor="none", zorder=0)
    pcm = LinearSegmentedColormap.from_list("pfr", ["#ffffff", "#e4ecf5", "#bdd0e5", "#8fb0d2", "#5f8bbb", "#3a6699"])
    im = axm.imshow(img_a, extent=ext_a, transform=proj_a, origin="lower", cmap=pcm, vmin=0, vmax=100, interpolation="bilinear", zorder=1, rasterized=True)
    axm.coastlines("50m", linewidth=0.3 * K, color=COAST, zorder=2)
    axm.gridlines(crs=PC, draw_labels=False, linewidth=0.3 * K, color="#c8c8c8", xlocs=range(-180, 180, 60), ylocs=[60, 70, 80], zorder=3)
    for la in (60, 70, 80):   # 위도 라벨(lat_lon 자오선)
        axm.text(lat_lon, la, f"{la}°N", transform=PC, fontsize=FS["a_lbl"], color="#555555", ha="center", va="center", path_effects=HALO, zorder=6)
    for lo in (-180, -120, -60, 0, 60, 120):   # 경도 라벨(원 둘레 바깥). 덱: 각도에 따라 바깥쪽 모서리를 원 밖 51°N 에 붙인다(12 pt 라벨이 원 테두리에 걸치지 않게)
        s = f"{abs(lo)}°{'W' if lo < 0 else 'E'}" if lo not in (0, -180) else ("0°" if lo == 0 else "180°")
        if MODE == "deck":
            x, y = proj_a.transform_point(lo, 51.0, PC); th = np.arctan2(y, x); c_, s_ = np.cos(th), np.sin(th)
            ha = "left" if c_ > 0.35 else ("right" if c_ < -0.35 else "center"); va = "bottom" if s_ > 0.35 else ("top" if s_ < -0.35 else "center")
        else:
            (x, y), ha, va = proj_a.transform_point(lo, 50.2, PC), "center", "center"
        axm.text(x, y, s, fontsize=FS["a_lbl"], color="#555555", ha=ha, va=va, clip_on=False, zorder=6)
    for r in REG:   # 지역 윤곽 상자(경위도 사각형을 투영; 레나는 BOX_A 의 넓힌 상자)
        lo0, lo1, la0, la1 = BOX_A[r]
        lons = np.r_[np.linspace(lo0, lo1, 40), np.full(40, lo1), np.linspace(lo1, lo0, 40), np.full(40, lo0)]
        lats = np.r_[np.full(40, la0), np.linspace(la0, la1, 40), np.full(40, la1), np.linspace(la1, la0, 40)]
        axm.plot(lons, lats, transform=PC, color=S.REGION_HEX[r], lw=1.0 * K, zorder=5, path_effects=[pe.withStroke(linewidth=2.0 * K, foreground="white")])
    for r, lt_ in (("Alaska", "b"), ("Lena", "c"), ("Canada", "d")):   # 이름 + 셀 수(상자 옆: 알래스카·레나는 왼쪽, 캐나다는 아래)
        pb = proj_box(proj_a, *BOX_A[r]); off = 90e3
        if r == "Canada" and canada_side == "right":
            x, y, ha, va = pb[1] + off, pb[2] + 0.25 * (pb[3] - pb[2]), "left", "top"
        elif r == "Canada":
            x, y, ha, va = (pb[0] + pb[1]) / 2, pb[2] - off, "center", "top"
        else:
            x, y, ha, va = pb[0] - off, (pb[2] + pb[3]) / 2, "right", "center"
        name = f"{lt_}  {S.REGION_EN[r]}" if ref else S.REGION_EN[r]
        axm.text(x, y, f"{name}\n{n_cells[r]:,} cells", fontsize=FS["a_box"], color=TEXT, ha=ha, va=va, path_effects=HALO, zorder=7, clip_on=False, linespacing=1.15)
    axm.spines["geo"].set_linewidth(0.5 * K); axm.spines["geo"].set_edgecolor("#777777")
    cax = fig.add_axes(R(*cb_rect)); cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 25, 50, 75, 100]); cb.set_label("Permafrost probability, ESA CCI (%)", fontsize=FS["cb_label"], labelpad=2 * K)
    cb.ax.tick_params(labelsize=FS["tick"], length=2 * K, width=0.5 * K)
    cb.outline.set_linewidth(0.5 * K)
    if lt:
        panel_letter(axm, lt, x=lt_xy[0], y=lt_xy[1]); S.register(lt, cax)
    return axm, cax


def draw_region(fig, r, rect, cells, lt=None, legend=False, inset_frac=None, dx_letter=-6.5, inset_top=None):
    """b 알래스카 / c 레나(음영 CRS) + 확대 창. rect=(x, y, w, h) mm(폭·높이 비는 region_aspect). legend=True 면 크기 열쇠(오른쪽 아래)."""
    key = r.lower(); proj = proj_of(r); bx = region_box(r); ext_r = EXT[r]
    rgba, ext, crs = shade_rgba(key, STEP[r])

    def base(ax_, rgba_=rgba, ext_=ext, key_=key, ext_r_=ext_r, lw_=0.2):
        ax_.imshow(rgba_, extent=ext_, origin="upper", transform=proj, interpolation="antialiased", zorder=1)
        ax_.add_feature(cfeature.LAKES.with_scale("10m"), facecolor=WATER, edgecolor="none", zorder=2)
        ax_.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor=WATER, facecolor="none", linewidth=0.35 * K, zorder=2)
        ax_.add_geometries(MV.ne_layers(key_)["coastline"], crs=PC, facecolor="none", edgecolor=COAST, linewidth=0.35 * K, zorder=4)
        block_grid(ax_, *ext_r_, lw=lw_)
    ax = fig.add_axes(R(*rect), projection=proj); ax.set_facecolor(WATER)
    ax.set_xlim(bx[0], bx[1]); ax.set_ylim(bx[2], bx[3])
    base(ax); graticule(ax, *(GRID_DECK if MODE == "deck" else GRID)[r])
    cr = cells[cells.region == r]
    cells_scatter(ax, cr, S.REGION_HEX[r], proj)
    ax.spines["geo"].set_linewidth(0.5 * K); ax.spines["geo"].set_edgecolor(FRAME)
    km = SB[r]; L = km * 1000.0; w = bx[1] - bx[0]; h = bx[3] - bx[2]
    sbx, sby = SB_POS[r]
    scalebar(ax, bx[0] + sbx * w, bx[2] + sby * h, L, f"{km} km", dy=0.012 * h)
    if lt:
        letter(ax, lt, rect, dx_mm=dx_letter)
    if legend:   # 크기 열쇠(관측 연도 수, b–d 공통) 한 번만. True: 지도 오른쪽 아래(관측 없는 산지, 논문). (x_mm, y_mm): 캔버스 좌표(왼쪽 위)에 그림 범례(덱).
        hs = [ax.scatter([], [], **legend_kw(k)) for k in (1, 2, 4)]   # 손잡이는 지도 마커와 같은 크기 함수(legend_kw)
        kw = dict(frameon=False, fontsize=FS["legend"], handletextpad=0.4, labelspacing=0.35, borderaxespad=0.0, title="Observed years",
                  title_fontsize=FS["legend"], alignment="left", scatterpoints=1, markerscale=1.0)
        if legend is True:
            leg = ax.legend(hs, ["1 year", "2–3 years", "≥ 4 years"], loc="lower right", bbox_to_anchor=(0.955, 0.045), **kw)
            leg.set_zorder(9)
            for t in list(leg.get_texts()) + [leg.get_title()]:
                t.set_path_effects(HALO)
            p_ = proj.transform_points(PC, cr.lon.values, cr.lat.values); fx_ = (p_[:, 0] - bx[0]) / w; fy_ = (p_[:, 1] - bx[2]) / h
            print(f"{r} cells under legend area (fx 0.77-0.96, fy 0.04-0.21): {int(((fx_ > 0.77) & (fx_ < 0.96) & (fy_ > 0.04) & (fy_ < 0.21)).sum())}")
        elif len(legend) == 3:   # (가운데 x_mm, 위 y_mm, 'h'): 가로 한 줄 = 제목(왼쪽 글자) + 항목 3개. 제목·항목 묶음의 가운데가 x_mm
            lx, ly, _ = legend
            kw_h = dict(kw); [kw_h.pop(k) for k in ("title", "title_fontsize", "alignment")]
            tw = text_mm("Observed years", FS["legend"]) + 4.0                                       # 제목 폭 + 간격(mm)
            lg = fig.legend(hs, ["1 year", "2–3 years", "≥ 4 years"], ncol=3, loc="upper center", bbox_to_anchor=((lx + tw / 2) / W_MM, 1 - ly / H_MM),
                            columnspacing=1.6, **kw_h)
            fig.canvas.draw(); bb = lg.get_window_extent(fig.canvas.get_renderer())
            fig.text(bb.x0 / (fig.get_figwidth() * fig.dpi) - 4.0 / W_MM, (bb.y0 + bb.y1) / 2 / (fig.get_figheight() * fig.dpi), "Observed years",
                     ha="right", va="center", fontsize=FS["legend"])
        else:
            lx, ly = legend
            fig.legend(hs, ["1 year", "2–3 years", "≥ 4 years"], loc="upper left", bbox_to_anchor=(lx / W_MM, 1 - ly / H_MM), **kw)
    # 확대 창: 알래스카 북사면 → 왼쪽 위(척치해 빈자리), 레나 델타 중심부 → 왼쪽 아래(델타 밖 본토, 관측 없음; 상자 아래 모서리와 직선 연결)
    zb = proj_box(proj, *ZOOM[r]); aspz = (zb[3] - zb[2]) / (zb[1] - zb[0])
    zrgba, zext, _ = shade_rgba(key, 1, box=zb)
    if r == "Alaska":
        fz = 0.46 if inset_frac is None else inset_frac                       # inset_top: 창 위 변의 틀 위에서 비율(기본 0.015). 덱 N05 는 0.10:
        hz = fz * rect[3]; wz = hz / aspz                                       # 위 연결선이 우트키아빅(배로) 관측 밀집부 아래로 지나간다
        rz = (rect[0] + 0.012 * rect[2], rect[1] + (0.015 if inset_top is None else inset_top) * rect[3], wz, hz); side = "left"; sbz = 50
    else:
        fz = 0.30 if inset_frac is None else inset_frac
        hz = fz * rect[3]; wz = hz / aspz; rz = (rect[0] + 0.02 * rect[2], rect[1] + 0.98 * rect[3] - hz, wz, hz); side = "below"; sbz = 10
    zc = cr[cr.lon.between(ZOOM[r][0], ZOOM[r][1]) & cr.lat.between(ZOOM[r][2], ZOOM[r][3])]

    def base_z(ax_, rgba_=zrgba, ext_=zext, key_=key, ext_r_=ZOOM[r]):
        ax_.imshow(rgba_, extent=ext_, origin="upper", transform=proj, interpolation="antialiased", zorder=1)
        ax_.add_feature(cfeature.LAKES.with_scale("10m"), facecolor=WATER, edgecolor="none", zorder=2)
        ax_.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor=WATER, facecolor="none", linewidth=0.35 * K, zorder=2)
        ax_.add_geometries(MV.ne_layers(key_)["coastline"], crs=PC, facecolor="none", edgecolor=COAST, linewidth=0.35 * K, zorder=4)
        block_grid(ax_, *ext_r_)
    axi = zoom_inset(fig, ax, proj, zb, rz, side, base_z, zc, S.REGION_HEX[r], sbz)
    if lt:
        S.register(lt, axi)
    # 창 아래 육지 비율(배치 점검용 출력)
    xi0, xi1 = bx[0] + (rz[0] - rect[0]) / rect[2] * w, bx[0] + (rz[0] + rz[2] - rect[0]) / rect[2] * w
    yi1, yi0 = bx[3] - (rz[1] - rect[1]) / rect[3] * h, bx[3] - (rz[1] + rz[3] - rect[1]) / rect[3] * h
    _, land_f, tr_f, _ = _shade_full(key)
    c0, c1 = int((xi0 - tr_f.c) / tr_f.a), int((xi1 - tr_f.c) / tr_f.a); r0, r1 = int((yi1 - tr_f.f) / tr_f.e), int((yi0 - tr_f.f) / tr_f.e)
    p_ = proj.transform_points(PC, cr.lon.values, cr.lat.values)              # 창 아래에 가려지는 관측 셀(창 틀 바깥 1 mm 여유 포함) = 0 이어야 한다
    m_x, m_y = 1.0 / rect[2] * w, 1.0 / rect[3] * h
    n_hid = int(((p_[:, 0] > xi0 - m_x) & (p_[:, 0] < xi1 + m_x) & (p_[:, 1] > yi0 - m_y) & (p_[:, 1] < yi1 + m_y)).sum())
    print(f"{r} inset {wz:.1f}x{hz:.1f} mm, land under inset {float(land_f[max(r0,0):r1, max(c0,0):c1].mean()):.2f}, cells in zoom {len(zc)} "
          f"(stars {(zc.n_years >= 4).sum()}), observed cells under inset (+1 mm) {n_hid}")
    assert MODE != "deck" or n_hid == 0, f"{r}: zoom inset hides {n_hid} observed cells"
    return ax, axi


def draw_canada(fig, rect, cells, lt=None, dx_letter=-6.5):
    """d 캐나다 NW(등장방형, 65°N 기준; 음영 자료 없음 → 블록 격자를 더 옅게). 크기 열쇠는 b 에 한 번만 둔다."""
    lo0, lo1, la0, la1 = EXT["Canada"]
    ax = fig.add_axes(R(*rect), projection=PC); ax.set_extent([lo0, lo1, la0, la1], PC); ax.set_aspect(1 / np.cos(np.radians(65.0)))
    ax.set_facecolor(WATER)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor=LAND, edgecolor="none", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor=WATER, edgecolor="none", zorder=2)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor=WATER, facecolor="none", linewidth=0.35 * K, zorder=2)
    ax.coastlines("10m", linewidth=0.35 * K, color=COAST, zorder=4)
    block_grid(ax, lo0, lo1, la0, la1, lw=0.12, alpha=0.22); graticule(ax, *GRID["Canada"])   # 0.5° 망: 음영 없는 바탕이라 가장 옅게
    cells_scatter(ax, cells[cells.region == "Canada"], S.REGION_HEX["Canada"], None)
    ax.spines["geo"].set_linewidth(0.5 * K); ax.spines["geo"].set_edgecolor(FRAME)
    lat_sb = la0 + 0.6; L = SB["Canada"] / (111.32 * np.cos(np.radians(lat_sb)))
    scalebar(ax, lo0 + 0.04 * (lo1 - lo0), lat_sb, L, f"{SB['Canada']} km", dy=0.15)
    if lt:
        letter(ax, lt, rect, dx_mm=dx_letter)
    return ax


def region_key(ax, rect, d, x_mm, y_mm):
    """지역 색 열쇠: 네모 + 지역 이름 + 마지막 관측 월 비율 한 줄(회색). 축 왼쪽 위 모서리에서 (x_mm, y_mm) 안쪽부터 한 줄씩 아래로."""
    pt = FS["share"]; em = pt * 25.4 / 72; sw = 0.8 * em; pitch = 1.55 * em; gap = 0.7 * em
    name_w = max(text_mm(S.REGION_EN[r], pt) for r in REG) + 1.2 * em
    for i, r in enumerate(REG):
        yc = 1 - (y_mm + i * pitch + 0.5 * em) / rect[3]
        ax.add_patch(Rectangle((x_mm / rect[2], yc - 0.5 * sw / rect[3]), sw / rect[2], sw / rect[3], transform=ax.transAxes, facecolor=S.REGION_HEX[r], lw=0, zorder=6, clip_on=False))
        xt = x_mm + sw + gap; mo = d.share[r]
        ax.text(xt / rect[2], yc, S.REGION_EN[r], transform=ax.transAxes, ha="left", va="center", fontsize=pt, color=TEXT, zorder=6)
        ax.text((xt + name_w) / rect[2], yc, f"Aug {mo[8]:.0%}, Sep {mo[9]:.0%}, Oct {mo[10]:.0%}".replace("%", " %"), transform=ax.transAxes, ha="left", va="center",
                fontsize=pt, color=S.NEUTRAL["text_aux"], zorder=6)


def draw_bars(fig, rect, d, xl, xticks, key_x_mm, shared=False, window_label=False, lt=None, dx_letter=-8.0):
    """e 연도별 셀-연도 누적 막대(지역 색, 아래부터 알래스카·레나·캐나다). rect=(x, y, w, h) mm. shared=True 면 f 와 축을 공유(x 눈금 라벨 없음).
    key_x_mm: 열쇠의 왼쪽 위치(축 왼쪽에서 mm). window_label=True 면 오른쪽 위에 'analysis window'(막대가 낮은 2022–2024 위)."""
    ax = fig.add_axes(R(*rect)); ax.patch.set_alpha(0)
    bottom = np.zeros(len(d.years), float)
    for r in REG:
        v = d.cnt[r].values.astype(float); ax.bar(d.years, v, bottom=bottom, width=0.72, color=S.REGION_HEX[r], lw=0, zorder=3); bottom += v
    top = bottom.max(); mag = 10 ** np.floor(np.log10(top)); ytop = np.ceil(top / mag) * mag
    ax.set_ylim(0, ytop * 1.08); ax.set_yticks([0, ytop / 2, ytop]); ax.set_yticklabels([f"{int(v):,}" for v in (0, ytop / 2, ytop)])
    ax.tick_params(length=2 * K, pad=1.5 * K, labelsize=FS["tick"]); ax.set_ylabel("Cell-years", labelpad=2.0 * K)
    ax.set_xlim(*xl); ax.set_xticks(xticks)
    if shared:
        ax.tick_params(axis="x", length=0, labelbottom=False)
    else:   # 축 라벨은 f·N06 의 공유 축과 같은 말('Year'). 해마다 작은 눈금
        ax.set_xlabel("Year", labelpad=1.5 * K); minor_years(ax, xl)
    region_key(ax, rect, d, key_x_mm, 0.4 * FS["share"] * 25.4 / 72)
    if window_label:
        ax.text(WINDOW[1] + 0.4, 0.985, "analysis window", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=FS["share"], color=S.NEUTRAL["text_aux"], zorder=6)
    if lt:
        letter(ax, lt, rect, dx_mm=dx_letter)
    return ax


def nice_top(v):
    """막대 축의 위 눈금: v 이상인 가장 작은 '좋은 수'(1 · 1.5 · 2 · 2.5 · 3 · 4 · 5 · 6 · 7 · 8 · 10 × 10^k)."""
    mag = 10 ** np.floor(np.log10(v))
    return float(next(m * mag for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10) if m * mag >= v))


def draw_bars_multi(fig, rect, d, xl, xticks, gap_mm, lt=None, dx_letter=-17.0):
    """N01 b(v7 보고 1쪽 구성): 지역별 연도별 셀-연도 막대 3단(알래스카 · 레나 델타 · 캐나다, 지역마다 독립 y 축 0 · 반 · 위 눈금), x 축 공유
    (아래 단에만 눈금 라벨·'Observation year'). 단마다 왼쪽 위에 지역 이름(지역 색)과 마지막 관측 월 비율(회색 한 줄). 글자 두 줄이 막대와 겹치지
    않도록 y 상한을 글자 높이만큼 늘리고 y 축선은 0–위 눈금까지만. rect=(x, y_top, w, h) mm. 반환 축 목록."""
    x, top, w, h = rect; hk = (h - 2 * gap_mm) / 3; axes = []
    em = FS["share"] * 25.4 / 72; txt_mm = 1.0 + FS["region"] * 25.4 / 72 * 1.15 + em * 1.25 + 1.2      # 지역 이름 + 비율 줄 + 여유(mm)
    yl_x = None
    for i, r in enumerate(REG):
        ax = fig.add_axes(R(x, top + i * (hk + gap_mm), w, hk)); ax.patch.set_alpha(0)
        v = d.cnt[r].values.astype(float); yt = nice_top(v.max())
        ax.bar(d.years, v, width=0.72, color=S.REGION_HEX[r], lw=0, zorder=3)
        frac = max(0.25, min(0.6, txt_mm / hk))                                                       # 글자 띠가 차지하는 축 높이 비율
        ax.set_ylim(0, yt / (1 - frac)); ax.set_yticks([0, yt / 2, yt]); ax.set_yticklabels([f"{int(t):,}" for t in (0, yt / 2, yt)])
        ax.spines["left"].set_bounds(0, yt); ax.tick_params(length=2 * K, pad=1.5 * K, labelsize=FS["tick"])
        ax.set_xlim(*xl); ax.set_xticks(xticks); minor_years(ax, xl)
        if i < len(REG) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Observation year", labelpad=3.0 * K)
        mo = d.share[r]
        ax.text(0.0, 1.0, S.REGION_EN[r], transform=ax.transAxes, ha="left", va="top", fontsize=FS["region"], color=S.REGION_HEX[r], zorder=6)
        ax.text(0.0, 1.0 - (FS["region"] * 25.4 / 72 * 1.25) / hk, f"Aug {mo[8]:.0%}, Sep {mo[9]:.0%}, Oct {mo[10]:.0%}".replace("%", " %"),
                transform=ax.transAxes, ha="left", va="top", fontsize=FS["share"], color=S.NEUTRAL["text_aux"], zorder=6)
        axes.append(ax)
    # y 라벨 하나(가운데 단), 가장 넓은 눈금 라벨(알래스카) 왼쪽에 x 를 맞춘다
    wmax = max(text_mm(f"{int(nice_top(d.cnt[r].values.max())):,}", FS["tick"]) for r in REG) + 2 * K * 25.4 / 72 + 1.5 * K * 25.4 / 72 + 2.0
    axes[1].set_ylabel("Cell-years"); axes[1].yaxis.set_label_coords(-wmax / w, 0.5)
    if lt:
        letter(axes[0], lt, (x, top, w, hk), dx_mm=dx_letter)
        S.register(lt, *axes[1:])
    return axes


def minor_years(ax, xl):
    """해마다 작은 눈금(라벨 없음): 큰 눈금 사이의 해(띠 시작·끝)를 셀 수 있게 한다."""
    ax.set_xticks([y for y in XT_MINOR if xl[0] <= y <= xl[1]], minor=True)
    ax.tick_params(axis="x", which="minor", length=1.1 * K, width=0.4 * K)


def row_positions(rows):
    """f 행의 y(위에서 0, 아래로 −1씩, 묶음이 바뀌면 GROUP_GAP 더)."""
    ys, y, prev = [], 0.0, None
    for g, *_ in rows:
        if prev is not None and g != prev:
            y -= GROUP_GAP
        ys.append(y); y -= 1.0; prev = g
    return ys


def draw_bands(fig, rect, d, xl, xticks, lt=None, dx_letter=-8.0):
    """f 자료별 기간 띠(행 이름은 y 눈금 라벨, 묶음 이름은 그 왼쪽 열 회색). 해 y 의 띠는 [y − 0.5, y + 0.5]. 정적 입력은 축 전체 회색 띠 + 'static'.
    Aalto(기간 평균)는 윤곽 띠. 분석 창 음영은 draw_inventory 의 바탕 축이 그린다."""
    ax = fig.add_axes(R(*rect)); ax.patch.set_alpha(0)
    ys = row_positions(d.rows)
    for y, (g, name, segs, col, kind) in zip(ys, d.rows):
        if kind == "static":
            ax.barh(y, xl[1] - xl[0], left=xl[0], height=BAND_H, color=col, lw=0, zorder=3)
            ax.text(0.5 * (xl[0] + WINDOW[0] - 0.5), y, "static", ha="center", va="center", fontsize=FS["share"], color=S.NEUTRAL["text_aux"], zorder=4)
        elif kind == "mean":
            (y0, y1), = segs; ax.barh(y, y1 - y0 + 1, left=y0 - 0.5, height=BAND_H, facecolor="none", edgecolor=col, lw=0.8 * K, zorder=3)
        else:
            for y0, y1 in segs:
                ax.barh(y, y1 - y0 + 1, left=y0 - 0.5, height=BAND_H, color=col, lw=0, zorder=3)
    pad_pt = 2.0 * K
    ax.set_yticks(ys); ax.set_yticklabels([r[1] for r in d.rows], fontsize=FS["band"]); ax.tick_params(axis="y", length=0, pad=pad_pt)
    ax.set_ylim(min(ys) - 0.6, 0.6); ax.set_xlim(*xl); ax.set_xticks(xticks); ax.set_xlabel("Year", labelpad=(3.0 if MODE == "deck" else 1.5) * K)
    ax.tick_params(axis="x", length=2 * K, pad=1.5 * K, labelsize=FS["tick"]); minor_years(ax, xl)
    ax.spines["left"].set_visible(False)
    # 묶음 이름: 행 이름 열(가장 긴 이름 폭)의 왼쪽, 묶음 가운데 높이에 오른쪽 맞춤
    name_w = max(text_mm(r[1], FS["band"]) for r in d.rows) + pad_pt * 25.4 / 72
    xg = -(name_w + 0.35 * FS["group"] * 25.4 / 72 * 1.0 + 1.0) / rect[2]
    tr = ax.get_yaxis_transform()
    groups = {}
    for y, r in zip(ys, d.rows):
        groups.setdefault(r[0], []).append(y)
    for g, yy in groups.items():
        ax.text(xg, float(np.mean(yy)), g, transform=tr, ha="right", va="center", fontsize=FS["group"], color=S.NEUTRAL["text_aux"], zorder=6)
    if lt:
        letter(ax, lt, rect, dx_mm=dx_letter)
    return ax, name_w


def inventory_cols(fs_band, fs_group, rows):
    """e·f 왼쪽 글자 열 폭(mm): 묶음 이름 + 간격 + 행 이름 + 눈금 여백. 축 왼쪽 x 를 정하는 데 쓴다."""
    em = fs_group * 25.4 / 72
    return max(text_mm(g, fs_group) for g, *_ in rows) + 0.35 * em + 1.0 + max(text_mm(r[1], fs_band) for r in rows) + 2.0 * K * 25.4 / 72


def draw_inventory(fig, x, top, w, h_bars, gap, pitch, d, lt=(None, None), dx_letter=-8.0):
    """e(위 누적 막대) + f(아래 띠) 한 연도 축 1997–2025. 바탕 축 하나가 둘을 덮어 분석 창 음영을 끊김 없이 그린다. 반환 (바탕, e, f 축, 전체 높이 mm)."""
    n_pitch = len(d.rows) + GROUP_GAP * (len({r[0] for r in d.rows}) - 1) + 1.2
    h_bands = n_pitch * pitch; h_all = h_bars + gap + h_bands
    axbg = fig.add_axes(R(x, top, w, h_all)); axbg.set_xlim(*XL_INV); axbg.set_ylim(0, 1); axbg.axis("off"); axbg.set_zorder(-1)
    axbg.axvspan(WINDOW[0] - 0.5, WINDOW[1] + 0.5, color=S.NEUTRAL["equiv_band"], lw=0, zorder=0)
    axe = draw_bars(fig, (x, top, w, h_bars), d, XL_INV, XT_INV, key_x_mm=0.6 * K, shared=True, window_label=True, lt=lt[0], dx_letter=dx_letter)
    axf, _ = draw_bands(fig, (x, top + h_bars + gap, w, h_bands), d, XL_INV, XT_INV, lt=lt[1], dx_letter=dx_letter)
    return axbg, axe, axf, h_all


# ======================================================================================= 논문·슬라이드 배치
def f1():
    global W_MM, H_MM
    d = load(); W_MM = S.W2
    # ---------------- 배치(mm). 1행: a 원 + b 알래스카(높이 hb 가 폭을 정한다), 2행: c 레나 | d 캐나다(폭을 채운다), 3행: e·f 자료 목록(전체 폭, 축 공유)
    asp = {r: region_aspect(r) for r in REG}
    # 1행: b 높이 hb 가 폭 wb 를 정한다. a 원 지름 D 는 남는 폭과 세로 한도(원 + 경도 라벨 + 컬러바가 b 아래선에서 끝난다) 중 작은 쪽, 남는 폭 안에서 가운데
    top1 = 3.5; hb = 79.5
    wb = hb / asp["Alaska"]; xb = W_MM - 3.0 - wb; ya = top1 + 2.0   # 원 위 경도 라벨(원 밖 ≈ 1 mm)
    D = min(xb - G_AB - XA, top1 + hb - ya - 10.0); xa = XA + 0.5 * (xb - G_AB - XA - D)
    cb_w = 0.80 * D; cb_rect = (xa + (D - cb_w) / 2, ya + D + 2.5, cb_w, 2.0)
    top2 = top1 + hb + 8.0; xc = 12.0; g_cd = 16.0
    h2 = (W_MM - 3.0 - xc - g_cd) / (1 / asp["Lena"] + 1 / asp["Canada"]); wc = h2 / asp["Lena"]; xd = xc + wc + g_cd; wd = h2 / asp["Canada"]
    top3 = top2 + h2 + 9.5; xinv = 3.0 + inventory_cols(FS["band"], FS["group"], d.rows); winv = W_MM - 3.0 - xinv   # 9.5 = c 경도 라벨(≈ 3 mm) + e 패널 문자
    h_bars = 15.5; g_inv = 2.0; pitch = 2.2   # 행 간격 2.2 mm(띠 글자 6 pt ≈ 2.1 mm 한 줄), 높이 한도 230 mm 아래 ≥ 3 mm 여유
    rects = {"Alaska": (xb, top1, wb, hb), "Lena": (xc, top2, wc, h2), "Canada": (xd, top2, wd, h2)}
    n_pitch = len(d.rows) + GROUP_GAP * 4 + 1.2; H_MM = top3 + h_bars + g_inv + n_pitch * pitch + 6.5
    print(f"layout mm: H={H_MM:.1f} a=({xa:.1f},{ya},{D:.1f}) a-b gutter={xb - xa - D:.1f} b={tuple(round(v, 1) for v in rects['Alaska'])} "
          f"c={tuple(round(v, 1) for v in rects['Lena'])} d={tuple(round(v, 1) for v in rects['Canada'])} ef=({xinv:.1f},{top3:.1f},{winv:.1f})")
    assert H_MM <= 227.0, H_MM   # 계약 한도 230 mm 에서 3 mm 여유를 남긴다(다음 수정이 한도에 바로 걸리지 않게)

    if S.SCALE == 1.0:
        write_source(d)
    fig = S.figure(W_MM, H_MM)
    draw_a(fig, (xa, ya, D, D), cb_rect, d.n_cells, "a", ref=True)
    draw_region(fig, "Alaska", rects["Alaska"], d.cells, "b", legend=True, inset_frac=0.38, inset_top=0.10)   # 덱 N05 와 같은 확대 창 자리(10-16)
    draw_region(fig, "Lena", rects["Lena"], d.cells, "c", dx_letter=-7.5)
    draw_canada(fig, rects["Canada"], d.cells, "d", dx_letter=-7.5)
    axbg, axe, axf, _ = draw_inventory(fig, xinv, top3, winv, h_bars, g_inv, pitch, d, lt=("e", "f"), dx_letter=-(xinv - 3.0))
    S.register("ef", axe, axf)   # 묶음 크롭: 한 열에 세로로 놓여 축 상자 합집합이 곧 패널
    out = S.export(fig, "F1_study_area", outdir=OUT)
    if S.SCALE != 1.0:   # 묶음 크롭 bcd: 축 상자 합집합은 이웃 패널(a)을 품으므로 패널 크롭을 이어 붙여 만든다
        sd = OUT / "slide"; byk = {p.stem.split("__")[-1]: p for p in out if "__" in p.stem}
        out.append(compose([byk[k] for k in ("b", "c", "d")], sd / "F1_study_area__bcd.png"))
    return out


# ======================================================================================= 덱 배치(슬라이드 본문 12 in 폭, 300 dpi)
def deck(slides=("N01", "N05", "N06")):
    """슬라이드마다 캔버스를 채우는 배치(여백은 축 라벨 몫). 패널 문자는 슬라이드 안 순서(a, b, c). 저장 후 픽셀 크기를 확인한다."""
    from PIL import Image
    global W_MM, H_MM
    set_mode("deck"); d = load(); sd = OUT / "deck"; sd.mkdir(parents=True, exist_ok=True); out = []
    asp = {r: region_aspect(r) for r in REG}
    for sl in slides:
        H_in = DECK_H_IN[sl]; W_MM = DECK_W_IN * 25.4; H_MM = H_in * 25.4
        fig = S.figure(W_MM, H_MM); fig.set_size_inches(DECK_W_IN, H_in)   # mm → in 반올림 오차(3599 px)를 막는다
        if sl == "N01":   # v7 보고 1쪽 구성: a 원 지도 + 컬러바(왼쪽) | b 지역별 셀-연도 막대 3단(오른쪽). 원 지름은 세로(경도 라벨 + 컬러바 블록 ≈ 26 mm)로 정한다
            D = 86.0; xa = 24.0; ya = 6.5
            cb_w = 0.80 * D; cb_rect = (xa + (D - cb_w) / 2, ya + D + 8.5, cb_w, 3.0)
            draw_a(fig, (xa, ya, D, D), cb_rect, d.n_cells, "a", ref=False, lt_xy=((3.0 - xa) / D, 1 + (ya - 5.6) / D), canada_side="right", lat_lon=35.0)
            xe = 146.0; top = 6.5; he = H_MM - 14.8 - top; xl = (2009.4, 2024.6)
            draw_bars_multi(fig, (xe, top, W_MM - 3.0 - xe, he), d, xl, [2010, 2015, 2020, 2024], gap_mm=5.0, lt="b", dx_letter=-(xe - 128.0))
        elif sl == "N05":   # a 알래스카(크게, 전체 높이) | b 레나 델타 | c 캐나다(같은 높이, 위 맞춤). 크기 열쇠는 b·c 아래 띠에 가로 한 줄.
            # 가로: 왼쪽 위도 라벨 13 mm, 패널 사이 = 다음 패널 위도 라벨(≈ 11.5 mm) + 여백 ≥ 9 mm(0.35 in). b·c 높이는 남는 폭으로 정한다
            top = 6.0; hb = 92.0; xb = 13.5; wb = hb / asp["Alaska"]; g = 22.0
            hs_ = (W_MM - 2.0 - xb - wb - 2 * g) / (1 / asp["Lena"] + 1 / asp["Canada"])
            xc = xb + wb + g; wc = hs_ / asp["Lena"]; xd = xc + wc + g; wd = hs_ / asp["Canada"]
            leg_y = top + hs_ + 0.5 * (hb - hs_) - 1.5                                                    # 아래 띠(경도 라벨 아래)의 가운데쯤
            draw_region(fig, "Alaska", (xb, top, wb, hb), d.cells, "a", legend=((xc + xd + wd) / 2, leg_y, "h"), dx_letter=-12.5,
                        inset_frac=0.38, inset_top=0.10)          # 10-16: 창이 관측 셀을 덮지 않고(여유 포함) 위 연결선이 배로 밀집부를 지나지 않는 자리
            draw_region(fig, "Lena", (xc, top, wc, hs_), d.cells, "b", inset_frac=0.31, dx_letter=-12.5)   # 10-16: 0.40 은 관측 밀집부·확대 상자와 겹쳤다
            draw_canada(fig, (xd, top, wd, hs_), d.cells, "c", dx_letter=-12.5)
            print(f"N05 layout mm: a ({xb:.1f}, {wb:.1f} x {hb:.1f}), b ({xc:.1f}, {wc:.1f} x {hs_:.1f}), c ({xd:.1f}, {wd:.1f} x {hs_:.1f}), right edge {xd + wd:.1f} / {W_MM:.1f}")
        elif sl == "N06":   # e·f 자료 목록 한 패널(패널 문자 없음): 위 누적 막대, 아래 띠 13행, 연도 축 공유. 왼쪽 글자 열은 글꼴 폭에서 계산
            top = 3.0; xinv = 2.0 + inventory_cols(FS["band"], FS["group"], d.rows); winv = W_MM - 3.0 - xinv
            pitch = 4.75; gap = 3.0; n_pitch = len(d.rows) + GROUP_GAP * 4 + 1.2   # 행 간격 4.75 mm: 12 pt 행 이름 사이 ≥ 3 pt
            h_bars = H_MM - top - gap - n_pitch * pitch - 14.5   # 아래 14.5 mm = 눈금 라벨 12 pt + 'Year' 14 pt(라벨 간격 포함)
            draw_inventory(fig, xinv, top, winv, h_bars, gap, pitch, d)
        else:
            raise ValueError(sl)
        p = sd / f"F1_study_area__deck_{sl}.png"
        fig.savefig(p, dpi=300, facecolor="white"); plt.close(fig)
        with Image.open(p) as im:
            rgb = im.convert("RGB")
        rgb.save(p, dpi=(300, 300))
        want = (int(round(DECK_W_IN * 300)), int(round(H_in * 300)))
        assert rgb.size == want, (sl, rgb.size, want)
        print(f"{sl}: {p} {rgb.size[0]}x{rgb.size[1]} px")
        out.append(p)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", type=int, default=1); ap.add_argument("--slide", action="store_true")
    ap.add_argument("--deck", action="store_true", help="덱 슬라이드 본문 PNG(N01 N05 N06)")
    ap.add_argument("--slides", nargs="*", default=None, help="--deck 에서 그릴 슬라이드(기본 N01 N05 N06)")
    a = ap.parse_args(); S.setup(scale=1.69 if (a.slide or a.deck) else 1.0)
    # PDF·SVG 의 래스터 층(CCI 확률, 음영, 1년 점)은 savefig dpi 로 그려진다. 기본값(figure 150 dpi)이면 64 mm 원이 약 380 px 로 뭉개지므로 500 dpi 로 둔다
    # (PNG 600 dpi·슬라이드판은 S.export 가 dpi 를 직접 넘기므로 영향 없음)
    plt.rcParams["savefig.dpi"] = 500
    paths = deck(tuple(a.slides) if a.slides else ("N01", "N05", "N06")) if a.deck else f1()
    for p in paths:
        print(p)
