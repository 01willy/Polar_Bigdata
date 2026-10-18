"""v7 F7: 방법이 내놓는 것. a 알래스카 1 km ALT 연도 편차 지도 2011–2024(연도 r1 − 2011–2024 평균 r1, 소형 다중, broc 발산 색표, 모든 연도 공통의
대칭 범위 ±MV.sym_max(5 cm 단위)), b 격자 전체 지역 평균의 연도 편차 시계열(알래스카·레나, 관측 셀 평균 겹침), c 제품 학습 지점에서 25 km 밖 공통 셀 RMSE
(물리 + 잔차 ML 대 최선 제품, 지역별), d 적용 범위 안 비율·90 % 구간 반폭.
입력: data/processed/v5/maps/<Region>/{pred_<year>,mean_2011_2024}.csv.gz, meta.json; labels_cell_year_v5.csv; runs/core_v6/common_cells_eval.csv.
실행(CPU 만): OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v7_f7_series_significance.py [--slide | --deck]
수치는 모두 CSV·JSON 에서 읽는다. 지도 셀은 원 격자 해상도(0.009° × 0.018°)의 래스터로 그린다(셀당 한 칸, 평균 없음). 편차는 셀마다 뺀 뒤 래스터화한다.
적용 범위 밖은 mean_2011_2024 의 aoa_in(연도와 무관한 정적 셀 마스크)이다. 색은 그대로 두고 마스크 윤곽으로 자른 45° 사선(자료 좌표, 수직 간격 HATCH_MM)을
얹는다. 사선은 자료 좌표의 선분이므로 모든 연도 패널에서 같고, PNG 축소·PDF 에서 해칭 무늬의 위상에 따라 사라지지 않는다.
관측 셀은 0.5° 블록마다 고리 하나(셀 수 3 등급, 가는 테두리, 투명도 0.7)로 그린다. 지도 없음(회색 육지) = 격자 밖 셀(영구동토 비율 < 50 %, 물·입력 결측).
산출 수치는 source_data/F7_series_significance.json 에 쓴다(c 의 계열은 series_code(결과 CSV 의 내부 이름)와 series_label(그림 라벨)을 함께 적고 label_map 에 대응을 둔다).
--deck: 국문 덱 슬라이드 본문 PNG(폭 12 in, 300 dpi, RGB, 흰 바탕)를 outputs/figures/paper/v7/deck/<stem>__deck_<slide>.png 에 쓴다.
R10(12.0 × 4.65 in, 10-15 v9 항 24): a 연도 편차 지도 3 × 5(지도 확대, 아래 빈 공간 없음) + 위 띠 한 줄 열쇠(마스크 · 색 막대 축소)
+ 15 번째 칸 고리 열쇠, b 지역 평균 편차(알래스카·레나 세로 배치, a–b 여백 0.45 in). C01(3.70 in): c 공통 셀 RMSE, d 적용 범위·구간 반폭.
C02(3.70 in): 대표 연도 편차 지도 4 장(2012·2016·2020·2024, 같은 색 범위) + 알래스카 편차. 자료 적재·패널 그리기 함수는 논문판과 같다(Med 로 크기만 바뀐다).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3"))
import polar.style_v5 as S
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath
import shapely
import cartopy.io.shapereader as shpreader
from scipy.spatial import cKDTree
import maps_alt_v3 as MV

PROC = ROOT / "data" / "processed" / "v5"; MAPS = PROC / "maps"; RUN = PROC / "runs" / "core_v6"
OUT = ROOT / "outputs" / "figures" / "paper" / "v7"; CACHE = OUT / "_cache"; DECK = OUT / "deck"; STEM = "F7_series_significance"
YEARS = list(range(2011, 2025)); DECK_YEARS = [2012, 2016, 2020, 2024]
BOX = (-168.0, -141.0, 59.5, 71.6)              # lon0, lon1, lat0, lat1 (알래스카 지도 창)
ASPECT = 1.0 / np.cos(np.radians(65.0))         # PlateCarree 세로 늘림
DLAT, DLON = 0.009, 0.018                       # 1 km 격자의 원 간격(위도 0.009°, 경도 0.0184°)
NLON, NLAT = int(round((BOX[1] - BOX[0]) / DLON)), int(round((BOX[3] - BOX[2]) / DLAT))   # 1500 × 1344
GRAT_Y = [60, 65, 70]
MB = S.TOK["color"]["map_base"]["paper"]; COAST, GRAT, FRAME = MB["coast"], MB["graticule"], "#555555"
LAND = MB["no_data_diverging"]                  # 격자 밖 육지(영구동토 비율 < 50 % 또는 입력 결측, 지도 없음): 발산 색표용 토큰(무채색 L* ≈ 83, 규칙은 토큰 파일 map_base.no_data_diverging_rule).
                                                # neutral.no_data(#C9CDD3)는 푸른 기가 있어 broc 의 음의 편차 셀과 섞인다. 바다는 흰색
ANOM_STEP = 5.0                                 # 편차 색 범위·눈금 단위(cm)
HATCH_COL, HATCH_K, HATCH_MM, HATCH_LEG = "#4A4A4A", 4, 1.0, "/" * 12   # 사선 색, 마스크 묶음(4 × 4 셀 = 0.036° × 0.072°), 선 수직 간격(mm, 패널 폭 기준), 열쇠 패치 해칭
RING, RING_ALPHA, BLOCK_DEG = "#111111", 0.7, 0.5; CLASSES = [(1, 9, 1.4), (10, 99, 2.4), (100, None, 3.6)]   # 관측 셀 수 등급 → 고리 지름(pt, 논문)
W_FIG, H_FIG = S.W2, 173.0
DECK_W, DECK_DPI = 12.0, 300
PRODS = {"Wei 2026": "wei", "Liu 2024": "liu", "ESA CCI v4": "cci4y"}
LABEL_C = {"R1 (CatBoost)": "Physics + residual ML"}   # common_cells_eval.csv 의 series(내부 코드) → 그림 라벨. 제품 이름은 그대로. source_data JSON 의 c.label_map 에도 쓴다
SRC: dict = {}


class Med:
    """매체별 크기. 논문(k = 1, 글꼴 5.5–7 pt)과 덱(k = 1.69, 글꼴은 토큰 font.slide: 눈금 12·직접 라벨 13·축 라벨 14·패널 문자 14 pt)."""

    def __init__(self, deck=False):
        self.deck = deck; k = self.k = 1.69 if deck else 1.0
        Fs = S.TOK["font"]["slide"]
        self.fs_tick = Fs["tick_pt"] if deck else 5.5          # 지도 눈금
        self.fs = Fs["tick_pt"] if deck else 6.0               # 열쇠, 값 라벨, 주석, c 의 계열 이름
        self.fs_ytick = Fs["tick_pt"] if deck else 6.5         # d 의 지역 이름
        self.fs_year = Fs["direct_label_pt"] if deck else 6.5  # 지도 연도
        self.fs_reg = Fs["direct_label_pt"] if deck else 7.0   # 지역 이름(b, c 머리)
        self.fs_lab = Fs["axis_label_pt"] if deck else 7.0
        self.fs_letter = Fs["panel_letter_pt"] if deck else S.TOK["font"]["paper"]["panel_letter_pt"]
        self.lw_grat, self.lw_coast, self.lw_frame, self.lw_hatch = 0.3 * k, 0.35 * k, 0.5 * k, 0.3 * k
        self.lw_line, self.lw_zero, self.lw_grid, self.lw_lead, self.lw_scale = 1.2 * k, 0.5 * k, 0.4 * k, 0.4 * k, 1.0 * k
        self.ms, self.mew = 3.4 * k, 0.7 * k                   # b 관측 표식
        self.ring_mew = 0.3 * k; self.halo = [pe.withStroke(linewidth=0.8 * k, foreground="white")]
        self.white = [pe.withStroke(linewidth=1.6 * k, foreground="white")]
        self.tick_len, self.tick_w, self.tick_pad = 1.5 * k, 0.4 * k, 1.2 * k
        self.grat_x = [-160, -145] if deck else [-160, -150]   # 덱은 경도 라벨(12 pt, 0.54 in)이 겹치지 않도록 15° 간격
        self.note_dy = -0.58 if deck else -0.80                # d 아래 주석 위치(축 높이 배)


def deck_rc(M):
    """덱 rcParams: 토큰 font.slide·lines.slide."""
    Ls = S.TOK["lines"]["slide"]
    plt.rcParams.update({"font.size": M.fs, "axes.labelsize": M.fs_lab, "xtick.labelsize": M.fs_tick, "ytick.labelsize": M.fs_tick, "legend.fontsize": M.fs,
                         "axes.linewidth": Ls["axis_pt"], "xtick.major.width": Ls["tick_pt"], "ytick.major.width": Ls["tick_pt"],
                         "xtick.major.size": Ls["tick_len_pt"], "ytick.major.size": Ls["tick_len_pt"], "lines.linewidth": Ls["data_pt"],
                         "lines.markersize": Ls["marker_pt"], "hatch.linewidth": M.lw_hatch})


def ax_mm(fig, x, y_top, w, h):
    """mm 단위(왼쪽 위 원점)로 축을 만든다(논문판)."""
    return fig.add_axes([x / W_FIG, 1.0 - (y_top + h) / H_FIG, w / W_FIG, h / H_FIG])


def letter(ax, s, x_mm, y_mm):
    """패널 문자를 그림 mm 좌표에 둔다(S.panel_letter 가 축 좌표를 받으므로 변환)."""
    b = ax.get_position(); x0, y0 = b.x0 * W_FIG, (1 - b.y1) * H_FIG; w, h = b.width * W_FIG, b.height * H_FIG
    S.panel_letter(ax, s, x=(x_mm - x0) / w, y=1.0 + (y0 - y_mm) / h)


# ---------------------------------------------------------------- 자료
def grid_index(lat, lon):
    ix = np.floor((lon - BOX[0]) / (BOX[1] - BOX[0]) * NLON).astype(int); iy = np.floor((lat - BOX[2]) / (BOX[3] - BOX[2]) * NLAT).astype(int)
    ok = (ix >= 0) & (ix < NLON) & (iy >= 0) & (iy < NLAT)
    return np.where(ok, iy * NLON + ix, -1), ok


def bin_mean(idx, ok, v):
    s = np.bincount(idx[ok], weights=v[ok], minlength=NLON * NLAT); c = np.bincount(idx[ok], minlength=NLON * NLAT)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    return m.reshape(NLAT, NLON)


def alaska_series():
    """연도별 편차 래스터(연도 r1 − 2011–2024 평균 r1, 셀마다 뺀 뒤 원 해상도로 래스터화), 적용 범위 래스터(정적), 연도별 격자 평균(절대 ALT), 셀 평균 표.
    캐시(npz, float16: 편차 ±40 cm 범위에서 정밀도 0.03 cm) 사용."""
    CACHE.mkdir(parents=True, exist_ok=True); cp = CACHE / f"f7_alaska_anom_{NLON}x{NLAT}.npz"
    mean = pd.read_csv(MAPS / "Alaska" / "mean_2011_2024.csv.gz", dtype={"cell_id": str})
    if cp.exists():
        z = np.load(cp); return {int(k[4:]): z[k].astype(np.float32) for k in z.files if k.startswith("img_")}, z["aoa"], dict(zip(z["years"].tolist(), z["means"].tolist())), mean
    idx, ok = grid_index(mean.lat.values, mean.lon.values)
    aoa = bin_mean(idx, ok, mean.aoa_in.values.astype(float))
    imgs, means = {}, {}
    for y in YEARS:
        p = pd.read_csv(MAPS / "Alaska" / f"pred_{y}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1.reindex(mean.cell_id).values
        imgs[y] = bin_mean(idx, ok, p - mean.r1.values).astype(np.float32); means[y] = float(np.nanmean(p))
    np.savez_compressed(cp, aoa=aoa.astype(np.float32), years=np.array(YEARS), means=np.array([means[y] for y in YEARS]), **{f"img_{y}": imgs[y].astype(np.float16) for y in YEARS})
    return imgs, aoa, means, mean


def region_means(region):
    return {y: float(pd.read_csv(MAPS / region / f"pred_{y}.csv.gz", usecols=["r1"]).r1.mean()) for y in YEARS}


def observed_anomaly(lab, region, mean, pred_anom, min_n=50, max_km=1.5):
    """관측 셀의 연도 편차: 셀-연도별 (관측 − 그 셀의 2011–2024 지도 평균)의 연도 평균(셀 구성 효과 제거)을, 같은 연도들의 예측 편차 평균에
    맞춰 중심화한다(관측일이 융해 종료 전이라 생기는 상수 차이를 뺀다). 셀 대응은 최근접 ≤ max_km, 연도는 셀-연도 ≥ min_n 만."""
    L = lab[lab.region == region]; c0 = np.cos(np.radians(mean.lat.mean()))
    d, i = cKDTree(np.c_[mean.lat.values, mean.lon.values * c0]).query(np.c_[L.lat.values, L.lon.values * c0])
    ok = d * 111.32 <= max_km
    L = L[ok].assign(anom=L.alt_cm.values[ok] - mean.r1.values[i[ok]])
    g = L.groupby("year").anom.agg(["mean", "size"]); g = g[(g["size"] >= min_n) & g.index.isin(YEARS)]
    g["mean"] = g["mean"] - g["mean"].mean() + np.mean([pred_anom[y] for y in g.index])
    return g


def block_counts(o):
    """관측 셀을 0.5° 블록으로 모아 (중심 경도, 중심 위도, 셀 수) 표를 만든다."""
    g = o.groupby([np.floor(o.lat / BLOCK_DEG), np.floor(o.lon / BLOCK_DEG)]).size()
    return pd.DataFrame({"lon": (g.index.get_level_values(1) + 0.5) * BLOCK_DEG, "lat": (g.index.get_level_values(0) + 0.5) * BLOCK_DEG, "n": g.values})


def ne_clip(name):
    clip = shapely.box(BOX[0] - 0.5, BOX[2] - 0.5, BOX[1] + 0.5, BOX[3] + 0.5); out = []
    for g in shpreader.Reader(shpreader.natural_earth(resolution="50m", category="physical", name=name)).geometries():
        if g.intersects(clip):
            gi = shapely.intersection(g, clip)
            if not gi.is_empty:
                out.append(gi)
    return out


def land_patch(geoms, **kw):
    paths = []
    for g in geoms:
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon":
                continue
            paths.append(MPath(np.asarray(p.exterior.coords)))
            for r in p.interiors:
                paths.append(MPath(np.asarray(r.coords)))
    return PathPatch(MPath.make_compound_path(*paths), **kw)


def load():
    """논문·덱 공통 자료. 지도 래스터, 색 범위, 해안·육지, 적용 범위 마스크, 지역 평균·관측 편차, 공통 셀 평가표, 지도 메타."""
    lab = pd.read_csv(PROC / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    imgs, aoa, meanA, cellA = alaska_series()
    allv = np.concatenate([v[np.isfinite(v)] for v in imgs.values()]); vmax = MV.sym_max(allv, step=ANOM_STEP)   # 모든 연도 공통 대칭 범위(|편차| 99 백분위, 5 cm 단위)
    extend = MV.extend_of(allv, -vmax, vmax)
    meanL = region_means("Lena"); cellL = pd.read_csv(MAPS / "Lena" / "mean_2011_2024.csv.gz", dtype={"cell_id": str})
    anomA = {y: meanA[y] - np.mean(list(meanA.values())) for y in YEARS}; anomL = {y: meanL[y] - np.mean(list(meanL.values())) for y in YEARS}
    ev = pd.read_csv(RUN / "common_cells_eval.csv"); mask = int(ev.mask_km.max()); ev = ev[ev.mask_km == mask]
    return dict(lab=lab, imgs=imgs, aoa=aoa, vmax=vmax, extend=extend, norm=Normalize(-vmax, vmax), land=ne_clip("land"), coast=ne_clip("coastline"), hatch_xy=outside_mask(aoa),
                cellA=cellA, cellL=cellL, meanA=meanA, meanL=meanL, anom={"Alaska": anomA, "Lena": anomL},
                obs={"Alaska": observed_anomaly(lab, "Alaska", cellA, anomA), "Lena": observed_anomaly(lab, "Lena", cellL, anomL)},
                ev=ev, mask=mask, meta={r: json.loads((MAPS / r / "meta.json").read_text()) for r in ("Alaska", "Lena")})


# ---------------------------------------------------------------- 지도 패널
def colorize(img, norm):
    rgba = MV.CM_DIFF(norm(np.ma.masked_invalid(img)))
    rgba[..., 3] = np.isfinite(img).astype(float)
    return rgba


def outside_mask(aoa):
    """적용 범위 밖 마스크를 HATCH_K × HATCH_K 묶음으로 줄인다(자료 없는 칸은 안쪽으로 둔다). 반환 (lon 중심, lat 중심, 0/1)."""
    k = HATCH_K; H, W = (NLAT // k) * k, (NLON // k) * k
    a = np.nan_to_num(aoa[:H, :W], nan=1.0).reshape(H // k, k, W // k, k).mean((1, 3))
    x = BOX[0] + (np.arange(W // k) + 0.5) * k * (BOX[1] - BOX[0]) / NLON; y = BOX[2] + (np.arange(H // k) + 0.5) * k * (BOX[3] - BOX[2]) / NLAT
    return x, y, (a < 0.5).astype(float)


def draw_hatch(ax, hatch_xy, M):
    """적용 범위 밖: 마스크 윤곽(contourf 경로)으로 자른 45° 사선. 선은 자료 좌표의 선분(수직 간격 HATCH_MM, 패널 폭 기준)이라 연도 패널마다 같다."""
    x, y, m = hatch_xy
    cs = ax.contourf(x, y, m, levels=[0.5, 1.5], colors="none"); paths = [p for p in cs.get_paths() if len(p.vertices)]; cs.remove()
    if not paths:
        return
    clip = PathPatch(MPath.make_compound_path(*paths), transform=ax.transData, facecolor="none", edgecolor="none", lw=0); ax.add_patch(clip)
    w_mm = ax.get_position().width * ax.figure.get_figwidth() * 25.4
    d = HATCH_MM * np.sqrt(2.0) * (BOX[1] - BOX[0]) / w_mm            # 화면 45° 선의 X 절편 간격(경도 단위)
    X1, Y1 = BOX[1] - BOX[0], (BOX[3] - BOX[2]) * ASPECT; segs = []
    for c in np.arange(-Y1, X1 + d, d):                                 # 화면 좌표 X − Y = c, X = lon − lon0, Y = (lat − lat0) · ASPECT
        xa, xb = max(0.0, c), min(X1, c + Y1)
        if xb > xa:
            segs.append([(BOX[0] + xa, BOX[2] + (xa - c) / ASPECT), (BOX[0] + xb, BOX[2] + (xb - c) / ASPECT)])
    lc = LineCollection(segs, colors=HATCH_COL, linewidths=M.lw_hatch, zorder=1.2, capstyle="butt"); lc.set_clip_path(clip); ax.add_collection(lc)


def draw_map(ax, rgba, hatch_xy, land, coast, left, bottom, M):
    ax.add_patch(land_patch(land, facecolor=LAND, edgecolor="none", zorder=0.5))
    ax.imshow(rgba, extent=[BOX[0], BOX[1], BOX[2], BOX[3]], origin="lower", interpolation="antialiased", zorder=1, rasterized=True)
    draw_hatch(ax, hatch_xy, M)
    for x in M.grat_x:
        ax.axvline(x, color=GRAT, lw=M.lw_grat, zorder=1.5)
    for y in GRAT_Y:
        ax.axhline(y, color=GRAT, lw=M.lw_grat, zorder=1.5)
    for g in coast:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.xy; ax.plot(xs, ys, color=COAST, lw=M.lw_coast, zorder=3, solid_capstyle="round")
    ax.set_xlim(BOX[0], BOX[1]); ax.set_ylim(BOX[2], BOX[3]); ax.set_aspect(ASPECT)
    ax.set_xticks(M.grat_x); ax.set_yticks(GRAT_Y)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}°W")); ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    ax.tick_params(length=M.tick_len, width=M.tick_w, labelsize=M.fs_tick, pad=M.tick_pad, labelleft=left, labelbottom=bottom)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(M.lw_frame); sp.set_edgecolor(FRAME)


def draw_blocks(ax, bc, M):
    """0.5° 블록 고리: 가는 테두리, 투명도 RING_ALPHA(겹친 고리가 밀도로 읽힌다)."""
    for lo, hi, ms in CLASSES:
        sel = bc[(bc.n >= lo) & ((bc.n <= hi) if hi is not None else True)]
        if len(sel):
            ax.plot(sel.lon, sel.lat, ls="none", marker="o", ms=ms * M.k, mfc="none", mec=RING, mew=M.ring_mew, alpha=RING_ALPHA, zorder=5, path_effects=M.halo)


def scalebar(ax, lon0, lat0, km, M):
    dlon = km / (111.32 * np.cos(np.radians(lat0)))
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color="black", lw=M.lw_scale, solid_capstyle="butt", zorder=6)
    ax.text(lon0 + dlon / 2, lat0 + (0.45 if M.deck else 0.3), f"{km} km", ha="center", va="bottom", fontsize=M.fs_tick, zorder=6, path_effects=M.white)


def year_map(ax, y, D, M, left, bottom, scale):
    """연도 y 의 알래스카 지도 한 장(래스터, 사선, 관측 블록 고리, 연도, 축척). 반환: 블록 표."""
    draw_map(ax, colorize(D["imgs"][y], D["norm"]), D["hatch_xy"], D["land"], D["coast"], left, bottom, M)
    lab = D["lab"]; bc = block_counts(lab[(lab.region == "Alaska") & (lab.year == y)].drop_duplicates("loc_id"))
    draw_blocks(ax, bc, M)
    ax.text(0.03, 0.965, str(y), transform=ax.transAxes, ha="left", va="top", fontsize=M.fs_year, zorder=7, path_effects=M.white)
    if scale:
        scalebar(ax, -149.5, 60.0, 200, M)
    return bc


def draw_keys(fig, cax, lax, pax, D, M, MVm):
    """색 막대(편차, 대칭, 5 cm 눈금), 블록 고리 열쇠, 사선 열쇠(색표 중앙 위의 사선, 가는 테두리)."""
    v = D["vmax"]; MV.colorbar(fig, cax, MV.CM_DIFF, D["norm"], "ALT anomaly (cm)", np.arange(-v, v + 1e-6, ANOM_STEP), MVm, D["extend"], orientation="horizontal")
    lax.axis("off"); pax.axis("off")
    hnd = [Line2D([], [], ls="none", marker="o", ms=ms * M.k, mfc="none", mec=RING, mew=M.ring_mew, alpha=RING_ALPHA, label=(f"{a}–{b}" if b else f"≥ {a}")) for a, b, ms in CLASSES]
    lax.legend(handles=hnd, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, title="Observed cells per 0.5° block", title_fontsize=M.fs, alignment="left",
               handlelength=1.4, handletextpad=0.4, columnspacing=1.0, fontsize=M.fs, borderaxespad=0, borderpad=0)
    pax.legend(handles=[Patch(facecolor=MV.CM_DIFF(0.5), edgecolor=HATCH_COL, hatch=HATCH_LEG, lw=M.lw_hatch, label="outside area of applicability"),
                        Patch(facecolor=LAND, edgecolor=HATCH_COL, lw=M.lw_hatch, label="not mapped")],
               loc="upper left", bbox_to_anchor=(0.0, 1.0), title="Masks", title_fontsize=M.fs, alignment="left", handlelength=1.6, handleheight=1.1,
               handletextpad=0.4, fontsize=M.fs, borderaxespad=0, borderpad=0, labelspacing=0.35)


# ---------------------------------------------------------------- b, c, d 패널
def draw_anom(ax, reg, D, M, legend=True, xlabel=True):
    """지역 평균 편차 시계열(지역마다 독립 y 범위). 예측 범위의 두 배를 넘는 관측 연도는 연도·셀-연도 수를 점 아래 14 pt(논문; 덱은 × 1.69)에 두고 세로 인출선으로
    점에 잇는다(왼쪽에 두면 덱의 좁은 축에서 지역 이름과 겹친다; 사선 인출선은 계약상 금지). 반환 (관측표, ylim)."""
    m, o = D["anom"][reg], D["obs"][reg]; aux = S.NEUTRAL["text_aux"]
    pred = np.array([m[y] for y in YEARS]); pmax = float(np.abs(pred).max())
    vmax = max(pmax, float(np.abs(o["mean"]).max())); ylim = 5.0 * np.ceil(vmax / 5.0) + 2.0
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=M.lw_zero, zorder=1)
    ax.plot(YEARS, pred, color=S.REGION_HEX[reg], lw=M.lw_line, zorder=3)
    ax.plot(o.index.values, o["mean"].values, ls="none", marker="o", ms=M.ms, mfc="none", mec="black", mew=M.mew, zorder=4)
    for yr, row in o[np.abs(o["mean"]) > 2.0 * pmax].iterrows():
        ax.annotate(f"{yr}, n = {int(row['size'])}", xy=(yr, row["mean"]), xytext=(0, -(M.ms / 2 + 14.0 * M.k)), textcoords="offset points", ha="center", va="top",
                    fontsize=M.fs, color=aux, zorder=5, arrowprops=dict(arrowstyle="-", linewidth=M.lw_lead, color=aux, shrinkA=1.0 * M.k, shrinkB=M.ms / 2 + 0.6 * M.k))
    ax.set_xlim(2010.4, 2024.6); ax.set_xticks([2012, 2016, 2020, 2024]); ax.set_ylim(-ylim, ylim)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20 if ylim > 30 else (10 if ylim > 15 else 5)))
    ax.text(0.03, 0.97, S.REGION_EN[reg], transform=ax.transAxes, ha="left", va="top", fontsize=M.fs_reg, color=S.REGION_HEX[reg])
    ax.set_ylabel("ALT anomaly (cm)")
    if xlabel:
        ax.set_xlabel("Year")
    else:
        ax.tick_params(labelbottom=False)
    if legend:
        ax.legend(handles=[Line2D([], [], color=aux, lw=M.lw_line, label="predicted, all cells"),
                           Line2D([], [], ls="none", marker="o", ms=M.ms, mfc="none", mec="black", mew=M.mew, label="observed cells")],
                  loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=M.fs, handlelength=1.6, handletextpad=0.5, borderaxespad=0.2)
    return o, ylim


def end_label(ax, x, y, s, dx, M):
    ax.text(x + dx, y, s, ha="left", va="center", fontsize=M.fs, color=S.NEUTRAL["text_aux"], zorder=3)


def draw_rmse(ax, D, M, head_x):
    """공통 셀(제품 학습 지점 mask km 밖) RMSE: 지역마다 물리 + 잔차 ML 과 최선 제품. head_x = 지역 머리글 x(축 폭 배, 음수). 반환: source 행."""
    ev = D["ev"]; rows = []; ypos = 0.0; heads = []; src = []
    for reg in ("Alaska", "Lena", "Canada"):
        e = ev[ev.region == reg].set_index("series")
        best = e.loc[list(PRODS)].rmse.idxmin()
        heads.append((ypos, reg))
        rows.append((ypos - 1.0, LABEL_C["R1 (CatBoost)"], float(e.loc["R1 (CatBoost)", "rmse"]), S.METHOD_HEX["R1"]))
        rows.append((ypos - 2.0, best, float(e.loc[best, "rmse"]), S.PRODUCT_HEX[PRODS[best]]))
        for s in ("R1 (CatBoost)", best):
            src.append({"region": reg, "series_code": s, "series_label": LABEL_C.get(s, s), "rmse_cm": float(e.loc[s, "rmse"]), "n": int(e.loc[s, "n"])})
        ypos -= 3.4
    xmax = max(r[2] for r in rows); xlim = 10.0 * np.ceil(xmax * 1.15 / 10.0)
    for y, lbl, v, col in rows:
        ax.barh(y, v, height=0.78, color=col, lw=0, zorder=2); end_label(ax, v, y, f"{v:.1f}", 0.012 * xlim, M)
    ax.set_yticks([r[0] for r in rows]); ax.set_yticklabels([r[1] for r in rows], fontsize=M.fs); ax.tick_params(axis="y", length=0, pad=2 * M.k)
    for y, reg in heads:
        ax.text(head_x, y - 0.10, S.REGION_EN[reg], transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=M.fs_reg, color=S.REGION_HEX[reg])   # 첫 행 라벨과 ≥ 3 pt
    ax.set_xlim(0, xlim); ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_ylim(rows[-1][0] - 0.7, 0.6); ax.set_xlabel(f"RMSE, cells > {D['mask']} km from\nproduct training sites (cm)"); ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=S.NEUTRAL["grid"], lw=M.lw_grid, zorder=0); ax.set_axisbelow(True)
    return src


def draw_aoa(axes, D, M):
    """적용 범위 안 비율, 90 % 구간 반폭(지도가 있는 두 지역). 첫 축 아래에 지도 지역 주석(캐나다 행이 없는 이유)."""
    meta = D["meta"]; regs = ["Alaska", "Lena"]
    for j, (ax, key, lbl, scale, xmax_, step, fmt) in enumerate(((axes[0], "frac", "Inside area of\napplicability (%)", 100.0, 100.0, 50, "{:.0f}"),
                                                                 (axes[1], "hw", "90 % interval\nhalf-width (cm)", 1.0, None, 10, "{:.1f}"))):
        vals = [meta[r]["aoa"]["frac_inside"] * scale if key == "frac" else meta[r]["interval"]["half_width_90"] for r in regs]
        xl = xmax_ if xmax_ else 10.0 * np.ceil(max(vals) * 1.15 / 10.0)
        ax.barh([1, 0], vals, height=0.7, color=[S.REGION_HEX[r] for r in regs], lw=0, zorder=2)
        for yy, v in zip([1, 0], vals):
            end_label(ax, v, yy, fmt.format(v), 0.015 * xl, M)
        ax.set_yticks([1, 0]); ax.set_yticklabels([S.REGION_EN[r] for r in regs], fontsize=M.fs_ytick); ax.tick_params(axis="y", length=0, pad=2 * M.k)
        ax.set_xlim(0, xl); ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
        ax.set_ylim(-0.6, 1.6); ax.set_xlabel(lbl); ax.spines["left"].set_visible(False); ax.tick_params(labelleft=(j == 0))
        ax.grid(axis="x", color=S.NEUTRAL["grid"], lw=M.lw_grid, zorder=0); ax.set_axisbelow(True)
    axes[0].text(0.0, M.note_dy, "maps: Alaska, Lena Delta", transform=axes[0].transAxes, ha="left", va="top", fontsize=M.fs, color=S.NEUTRAL["text_aux"])
    return {r: {"frac_inside_aoa": meta[r]["aoa"]["frac_inside"], "half_width_90_cm": meta[r]["interval"]["half_width_90"]} for r in regs}


# ---------------------------------------------------------------- 논문 그림
def f7():
    M = Med(deck=False); MVm = MV.Med("paper"); plt.rcParams["hatch.linewidth"] = M.lw_hatch
    D = load(); vmax = D["vmax"]
    fig = S.figure(W_FIG, H_FIG)
    # ---- a: 2 × 7 소형 다중 지도(원 해상도 래스터, 0.5° 블록 관측 고리)
    w = 23.0; h = w * (BOX[3] - BOX[2]) * ASPECT / (BOX[1] - BOX[0]); gap = 1.2; x0, y0 = 9.5, 6.0
    axes_a = []; blocks = {}
    for k, y in enumerate(YEARS):
        r, c = divmod(k, 7)
        ax = ax_mm(fig, x0 + c * (w + gap), y0 + r * (h + gap), w, h)
        blocks[y] = year_map(ax, y, D, M, left=(c == 0), bottom=(r == 1), scale=(c == 0)); axes_a.append(ax)
    letter(axes_a[0], "a", 3.5, 3.0); S.register("a", *axes_a[1:])
    y_cb = y0 + 2 * h + gap + 5.3
    cax = ax_mm(fig, 14.0, y_cb, 42.0, 2.5); lax = ax_mm(fig, 66.0, y_cb - 2.2, 62.0, 9.0); pax = ax_mm(fig, 134.0, y_cb - 2.2, 40.0, 9.0)
    draw_keys(fig, cax, lax, pax, D, M, MVm); S.register("a", cax, lax, pax)
    SRC["a"] = {"variable": "r1_year - r1_mean_2011_2024", "anomaly_range_cm": [-vmax, vmax], "colorbar_extend": D["extend"], "n_cells": int(len(D["cellA"])),
                "frac_inside_aoa": float(D["cellA"].aoa_in.mean()), "raster_deg": [DLAT, DLON],
                "hatch_mask_deg": [DLAT * HATCH_K, DLON * HATCH_K], "hatch_spacing_mm": HATCH_MM, "ring_alpha": RING_ALPHA,
                "regional_mean_cm": {str(y): D["meanA"][y] for y in YEARS}, "observed_cells": {str(y): int(blocks[y].n.sum()) for y in YEARS},
                "observed_blocks_0p5deg": {str(y): int(len(blocks[y])) for y in YEARS}}
    # ---- b: 지역 평균 편차 시계열(지역마다 독립 y 범위)
    yb, hb = 83.0, 27.0; axes_b = []; SRC["b"] = {}       # a 의 열쇠 줄 아래 ≥ 4 mm: 슬라이드판 패널 a 크롭(여백 0.76 mm)이 b 문자에 닿지 않는다
    for j, reg in enumerate(("Alaska", "Lena")):
        ax = ax_mm(fig, 12.0 + j * 90.0, yb, 76.0, hb); axes_b.append(ax)
        o, ylim = draw_anom(ax, reg, D, M, legend=(j == 0)); m = D["anom"][reg]
        SRC["b"][reg] = {"mean_2011_2024_cm": float(np.mean(list(D["meanA" if reg == "Alaska" else "meanL"].values()))), "predicted_anomaly_cm": {str(y): m[y] for y in YEARS},
                         "observed_anomaly_cm": {str(y): float(o.loc[y, "mean"]) for y in o.index}, "observed_n_cell_years": {str(y): int(o.loc[y, "size"]) for y in o.index},
                         "min_cell_years": 50, "ylim_cm": float(ylim)}
    letter(axes_b[0], "b", 3.5, 79.0); S.register("b", axes_b[1])
    # ---- c: 공통 셀(제품 학습 지점 25 km 밖) RMSE, 물리 + 잔차 ML 대 최선 제품. c·d 는 b 의 축 라벨과 3 mm 더 띄운다(패널 b 크롭에 문자 c·d 가 들어오지 않게)
    yc, yl = 128.0, 124.0
    axc = ax_mm(fig, 30.0, yc, 52.0, 32.0)
    SRC["c"] = {"mask_km": D["mask"], "source": "runs/core_v6/common_cells_eval.csv", "label_map": dict(LABEL_C), "rows": draw_rmse(axc, D, M, head_x=-24.0 / 52.0)}
    letter(axc, "c", 3.5, yl)
    # ---- d: 적용 범위 안 비율, 90 % 구간 반폭
    axes_d = [ax_mm(fig, 110.0 + j * 37.0, yc, 30.0, 16.0) for j in range(2)]
    SRC["d"] = draw_aoa(axes_d, D, M)
    letter(axes_d[0], "d", 92.0, yl); S.register("d", axes_d[1])
    S.register("ab", *S.PANELS["a"], *S.PANELS["b"]); S.register("cd", *S.PANELS["c"], *S.PANELS["d"])
    (OUT / "source_data").mkdir(parents=True, exist_ok=True)
    (OUT / "source_data" / f"{STEM}.json").write_text(json.dumps(SRC, indent=1, ensure_ascii=False, default=float))
    print(f"[F7] anomaly range ±{vmax} cm ({D['extend']}); obs years Alaska {list(D['obs']['Alaska'].index)}, Lena {list(D['obs']['Lena'].index)}; mask {D['mask']} km")
    return S.export(fig, STEM, outdir=OUT)


# ---------------------------------------------------------------- 덱 슬라이드(12 in 폭, 300 dpi)
def deck_canvas(H):
    fig = S.figure(DECK_W * 25.4, H * 25.4); fig.set_size_inches(DECK_W, H); fig.patch.set_facecolor("white"); return fig   # mm 왕복 오차(11.9999 in → 3599 px) 제거


def ax_in(fig, x, y_top, w, h):
    """inch 단위(왼쪽 위 원점)로 축을 만든다(덱)."""
    H = fig.get_figheight(); return fig.add_axes([x / DECK_W, 1.0 - (y_top + h) / H, w / DECK_W, h / H])


def deck_letter(fig, s, x, y_top, M):
    fig.text(x / DECK_W, 1.0 - y_top / fig.get_figheight(), s, fontsize=M.fs_letter, fontweight="bold", ha="left", va="top")


def deck_save(fig, slide, H):
    from PIL import Image
    DECK.mkdir(parents=True, exist_ok=True); p = DECK / f"{STEM}__deck_{slide}.png"
    fig.savefig(p, dpi=DECK_DPI, facecolor="white"); plt.close(fig)
    with Image.open(p) as im:
        size = im.size; rgb = im.convert("RGB")
    rgb.save(p, dpi=(DECK_DPI, DECK_DPI))
    want = (int(round(DECK_W * DECK_DPI)), int(round(H * DECK_DPI)))
    assert size == want, f"{slide}: {size} != {want}"
    print(f"[deck {slide}] {p} {size[0]} x {size[1]} px"); return p


def deck_r10(D, M, MVm):
    """R10(12.0 × 4.65 in, 10-15 v9 항 24): a 연도 편차 지도 3 × 5(14 장, 높이를 채우는 크기) + 위 띠 한 줄 열쇠(마스크 2 항목 · 편차 색 막대),
    15 번째 칸(오른쪽 아래)에 관측 블록 고리 열쇠, b 알래스카·레나 평균 편차(오른쪽, 세로 배치). a–b 사이 여백 ≥ 0.45 in."""
    H = 4.65; fig = deck_canvas(H)
    ncol, nrow, gap = 5, 3, 0.06; ytop, ybot, xa0 = 0.43, 0.05, 0.50          # 아래 띠 없음: 경도 라벨을 빼고(같은 범위의 소형 다중, 위도 라벨·경선·축척은 둔다) 지도를 키운다
    h = (H - ytop - ybot - (nrow - 1) * gap) / nrow; w = h * (BOX[1] - BOX[0]) / ((BOX[3] - BOX[2]) * ASPECT)
    n_last = len(YEARS) - (nrow - 1) * ncol                                       # 마지막 줄 지도 수(4)
    for k, y in enumerate(YEARS):
        r, c = divmod(k, ncol)
        year_map(ax_in(fig, xa0 + c * (w + gap), ytop + r * (h + gap), w, h), y, D, M, left=(c == 0), bottom=False,
                 scale=(r == nrow - 1 and c == 0))
    xa1 = xa0 + ncol * w + (ncol - 1) * gap                                      # a 오른쪽 끝(in)
    deck_letter(fig, "a", 0.04, 0.03, M)
    # 위 띠: 마스크 열쇠(왼쪽) · 편차 색 막대(오른쪽 끝 = a 오른쪽 끝). 색 막대 라벨은 막대 왼쪽, 눈금은 −v · 0 · v(5 cm 작은 눈금)
    v = D["vmax"]; cb_w, cb_h = 1.30, 0.085; yb = ytop - 0.05 - cb_h / 2                       # 색 막대 가운데(in, 위에서): 지도 위 0.05 in, 눈금 라벨은 막대 위
    cax = ax_in(fig, xa1 - cb_w - 0.12, yb - cb_h / 2, cb_w, cb_h)
    cb = MV.colorbar(fig, cax, MV.CM_DIFF, D["norm"], "", [-v, 0, v], MVm, D["extend"], orientation="horizontal")
    cax.set_xticks(np.arange(-v, v + 1e-6, ANOM_STEP), minor=True)
    cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cax.tick_params(which="both", top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=M.fs, pad=M.tick_pad)
    cax.tick_params(which="minor", length=M.tick_len * 0.8, width=M.tick_w)
    fig.text((xa1 - cb_w - 0.12 - 0.12) / DECK_W, 1.0 - yb / H, "ALT anomaly (cm)", ha="right", va="center", fontsize=M.fs)
    lg = fig.legend(handles=[Patch(facecolor=MV.CM_DIFF(0.5), edgecolor=HATCH_COL, hatch=HATCH_LEG, lw=M.lw_hatch, label="outside applicability"),
                             Patch(facecolor=LAND, edgecolor=HATCH_COL, lw=M.lw_hatch, label="not mapped")],
                    loc="center left", bbox_to_anchor=(0.34 / DECK_W, 1.0 - yb / H), ncol=2, fontsize=M.fs, frameon=False, handlelength=1.6,
                    handleheight=1.0, handletextpad=0.4, columnspacing=1.4, borderaxespad=0.0, borderpad=0.0)
    # 15 번째 칸: 관측 블록 고리 열쇠(제목 두 줄 + 등급 3 개)
    xs, ys = xa0 + (ncol - 1) * (w + gap), ytop + (nrow - 1) * (h + gap)
    hnd = [Line2D([], [], ls="none", marker="o", ms=ms * M.k, mfc="none", mec=RING, mew=M.ring_mew, alpha=RING_ALPHA, label=(f"{a}–{b}" if b else f"≥ {a}"))
           for a, b, ms in CLASSES]
    fig.legend(handles=hnd, loc="upper left", bbox_to_anchor=((xs + 0.06) / DECK_W, 1.0 - (ys + 0.04) / H), ncol=1, title="Observed cells\nper 0.5° block",
               title_fontsize=M.fs, alignment="left", fontsize=M.fs, frameon=False, handlelength=1.4, handletextpad=0.5, borderaxespad=0.0, borderpad=0.0,
               labelspacing=0.45)
    # b: 알래스카 · 레나 평균 편차(지도 위 끝 ~ 아래 x 라벨). 왼쪽 = a 오른쪽 끝 + 여백 0.45 in + y 라벨·눈금(0.78 in)
    xb0 = xa1 + 0.45 + 0.78; wb = DECK_W - 0.22 - xb0; gb = 0.24; ybb = 0.56; hb = (H - ytop - ybb - gb) / 2
    draw_anom(ax_in(fig, xb0, ytop, wb, hb), "Alaska", D, M, legend=False, xlabel=False)
    draw_anom(ax_in(fig, xb0, ytop + hb + gb, wb, hb), "Lena", D, M, legend=True, xlabel=True)
    deck_letter(fig, "b", xa1 + 0.45, 0.03, M)
    print(f"[R10] maps {w:.3f} x {h:.3f} in, a right {xa1:.2f} in, b x {xb0:.2f}–{xb0 + wb:.2f} in")
    return deck_save(fig, "R10", H)


def deck_c01(D, M):
    """C01: c 공통 셀 RMSE(왼쪽), d 적용 범위 안 비율·90 % 구간 반폭(오른쪽)."""
    H = 3.70; fig = deck_canvas(H)
    draw_rmse(ax_in(fig, 1.95, 0.40, 3.5, 2.35), D, M, head_x=-1.72 / 3.5); deck_letter(fig, "c", 0.08, 0.10, M)
    draw_aoa([ax_in(fig, 7.0, 0.40, 2.1, 1.55), ax_in(fig, 9.7, 0.40, 2.2, 1.55)], D, M); deck_letter(fig, "d", 6.0, 0.10, M)
    return deck_save(fig, "C01", H)


def deck_c02(D, M, MVm):
    """C02: 대표 연도 편차 지도 4 장(같은 색 범위, 왼쪽) + 알래스카 평균 편차(오른쪽)."""
    H = 3.70; fig = deck_canvas(H)
    xa0, gap, ytop, h = 0.55, 0.1, 0.45, 1.8; w = h * (BOX[1] - BOX[0]) / ((BOX[3] - BOX[2]) * ASPECT)
    for c, y in enumerate(DECK_YEARS):
        year_map(ax_in(fig, xa0 + c * (w + gap), ytop, w, h), y, D, M, left=(c == 0), bottom=True, scale=(c == 0))
    deck_letter(fig, "a", 0.08, 0.10, M)
    yk = ytop + h + 0.68                            # 색 막대 2.4 in; 열쇠 글은 b 축(8.7 in) 왼쪽에서 끝난다
    draw_keys(fig, ax_in(fig, 0.95, yk, 2.4, 0.18), ax_in(fig, 3.65, yk - 0.12, 2.6, 0.7), ax_in(fig, 6.25, yk - 0.12, 2.6, 0.7), D, M, MVm)
    draw_anom(ax_in(fig, 8.7, ytop, 3.2, 2.65), "Alaska", D, M, legend=True, xlabel=True); deck_letter(fig, "b", 7.95, 0.10, M)
    return deck_save(fig, "C02", H)


def deck():
    M = Med(deck=True); deck_rc(M); MVm = MV.Med("slide"); D = load()
    return [deck_r10(D, M, MVm), deck_c01(D, M), deck_c02(D, M, MVm)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--slide", action="store_true"); ap.add_argument("--deck", action="store_true"); ap.add_argument("--fig", default="7")
    a = ap.parse_args(); S.setup(scale=1.69 if (a.slide or a.deck) else 1.0)
    for p in (deck() if a.deck else f7()):
        print(p)
