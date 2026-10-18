# -*- coding: utf-8 -*-
"""논문 초안 v10 Fig. 1: 라벨 셀 · 입력 · 연도(Scientific Reports 본문 그림, 180 mm 폭).

패널
  a  북극 지도: ESA CCI 영구동토 확률(1997–2021 평균, cci_pfr_mean_1997_2021.nc)과 세 연구 지역 윤곽(검정), 지역별 라벨 셀 수.
  b  관측 연도별 셀-연도 수(2010–2024), 지역마다 작은 막대 패널(독립 y 축), 막대는 한 가지 진회색.
  c  자료 기간(1997–2025 축): 지역별 라벨 연도, ERA5-Land 월 기후 입력, 정적 공변량(연도 없음, 빗금), ALT 제품 6종
     (제공 기간 = 옅은 회색, 비교에 쓴 연도 = 진회색, Aalto 2018 = 다년 평균 윤곽 띠). 분석 창 2010–2024 옅은 음영.
  d  기후 입력의 시간 정의: 알래스카 라벨 셀 한 곳의 2014 누적 TDD(월말 누적값의 일별 선형 보간), 관측일 · 계절 전체 · 2011–2024 평균.
자료(수치는 모두 파일에서 읽는다)
  data/processed/v5/labels_cell_year_v5.csv(in_forcing == 1), data/processed/cci_pfr_mean_1997_2021.nc, data/processed/v5/forcing_meta.json,
  data/processed/v5/products_cell_year_v5_meta.json, data/processed/v5/products_cell_year_v5.csv, data/processed/v5/forcing_year.csv,
  data/processed/v5/forcing_clim.csv. Liu 2024 의 제공 기간(2003–2020)은 내려받은 파일(2010–2020)보다 길어 docs/research/2026-10-04/alt_products.md
  (Zenodo 10070610, ALT_{YYYY}.tif, YYYY = 2003–2020)에서 옮긴다.
  '비교에 쓴 연도' = 라벨 셀-연도 중 그 제품 값이 있는 해의 최소–최대(scripts/2_evaluation/v10_numbers.py 의 nyrsprod* 와 같은 정의).
  d 의 셀 선택 규칙 = scripts/4_visualization/v7_f2_problem_time.py pick_alaska_cell(관측 연도 수 최다, 동률이면 관측일 누적 TDD 비율 중앙값 최소,
  그다음 loc_id). 날짜가 없는 CALM 관측(eos)은 파이프라인 규약 9월 1일(doy 244, polar.testbed_v5.with_forcing).
실행  cd /home/willy010313/Polar_Bigdata && POLAR_PALETTE=paper CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v10/fig1_data.py
산출  outputs/figures/paper/v10/Fig1_data.{pdf,svg,png}(180 × 126 mm, png 600 dpi), outputs/figures/paper/v10/source_data/Fig1_data.json
양식  흰 바탕, 검정 글자(축 라벨 · 주석 7 pt, 눈금 6.5 pt, 최소 6 pt, 패널 문자 8 pt 굵게), FreeSans, 축선 0.6 pt, 자료선 1.0 pt 이하.
      색은 지도 색표(cmcrameri oslo_r)뿐이고 나머지는 회색 명도. 그림 안 제목 · 문장 없음. 저장은 bbox 자르기 없이 캔버스 크기 그대로(폭 180 mm).
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
import calendar, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S
import polar.testbed_v5 as T
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patheffects as pe
import matplotlib.transforms
import matplotlib.lines
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import Rectangle, Patch
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cmcrameri import cm as cmc
from pyproj import Geod
from scipy.ndimage import map_coordinates

PROC = ROOT / "data" / "processed"; V5 = PROC / "v5"
OUT = ROOT / "outputs" / "figures" / "paper" / "v10"; STEM = "Fig1_data"
PC = ccrs.PlateCarree()
REG = ["Alaska", "Lena", "Canada"]
REG_EN = {"Alaska": "Alaska", "Lena": "Lena Delta", "Canada": "Canada (NW)"}
WINDOW = (2010, 2024)                                     # 분석 창(forcing_meta.json years 와 같음을 확인한다)
XLIM_C = (1996.5, 2025.5)                                 # c 연도 축(1997–2025, 해 y 는 [y − 0.5, y + 0.5])
LIU_AVAILABLE = (2003, 2020)                              # Liu 2024 제공 기간: docs/research/2026-10-04/alt_products.md(Zenodo 10070610)
LIU_AVAILABLE_SRC = "docs/research/2026-10-04/alt_products.md (Zenodo 10.5281/zenodo.10070610, ALT_YYYY.tif for 2003–2020)"
EOS_DOY = 244                                             # 날짜 없는 CALM 관측(eos)의 관측일 규약 = 9월 1일(polar.testbed_v5.with_forcing)

# ---------------------------------------------------------------- 양식(토큰 + 논문 v10 지시)
W_MM, H_MM = 180.0, 121.3
FS = 7.0; FS_TICK = 6.5; FS_SMALL = 6.5; FS_LETTER = 8.0
LW_AX = 0.6; LW_DATA = 1.0; LW_REF = 0.6
INK = "#000000"
DARK = "#4D4D4D"                                          # 막대 · 사용 연도 띠(진회색)
LIGHT = "#C4C4C4"                                         # 제공 기간 띠(옅은 회색)
SHADE = "#EFEFEF"                                         # 분석 창 음영
REFGREY = S.NEUTRAL["zero_line"]                          # 기준선 회색(paper_v10 zero_line #808080)
OCEAN = "#E9E9E9"; LAND = "#FFFFFF"; COAST = "#8C8C8C"; GRAT = "#C8C8C8"
PFR_RANGE = (0.0, 0.6)
PFR_CMAP = ListedColormap(cmc.oslo_r(np.linspace(*PFR_RANGE, 256)))   # 확률 0 % = 흰색, 100 % = 중간 청색(배경 지도라 0.6 에서 자름: 검은 윤곽 · 글자가 보이게)
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
SRC: dict = {}


def setup():
    S.setup()
    for f in ("/usr/share/fonts/truetype/freefont/FreeSansOblique.ttf",):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_SMALL,
        "axes.linewidth": LW_AX, "xtick.major.width": LW_AX, "ytick.major.width": LW_AX, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
        "axes.labelpad": 2.5, "lines.linewidth": LW_DATA, "lines.markersize": 3.0, "hatch.linewidth": 0.4,
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
        "mathtext.fontset": "custom", "mathtext.rm": "FreeSans", "mathtext.it": "FreeSans:italic", "mathtext.bf": "FreeSans:bold",
        "mathtext.sf": "FreeSans", "mathtext.default": "it", "mathtext.fallback": None, "svg.fonttype": "none", "pdf.fonttype": 42})


def rect(x, y_top, w, h):
    """mm 사각형(왼쪽, 위, 폭, 높이; 위에서 잰 y) → figure 분율."""
    return [x / W_MM, 1 - (y_top + h) / H_MM, w / W_MM, h / H_MM]


def letter(fig, s, x_mm, y_mm):
    """패널 문자: 8 pt 굵게, 글자 위 가장자리를 (x, y) mm(위에서 잰 y)에 맞춘다."""
    return fig.text(x_mm / W_MM, 1 - y_mm / H_MM, s, fontsize=FS_LETTER, fontweight="bold", ha="left", va="top")


def year_runs(ys):
    """정렬된 해 목록 → 연속 구간 [(y0, y1), ...]."""
    out = []
    for y in sorted(int(v) for v in ys):
        if out and y == out[-1][1] + 1:
            out[-1] = (out[-1][0], y)
        else:
            out.append((y, y))
    return out


def runs_str(runs):
    return ", ".join(f"{a}–{b}" if b > a else f"{a}" for a, b in runs)


# ================================================================ 자료
def load():
    lab = pd.read_csv(V5 / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1].copy()
    n_cells = lab.groupby("region").loc_id.nunique()
    n_rows = lab.groupby("region").size()
    years = np.arange(WINDOW[0], WINDOW[1] + 1)
    cnt = lab.groupby(["year", "region"]).size().unstack("region").reindex(index=years, columns=REG).fillna(0).astype(int)
    fmeta = json.loads((V5 / "forcing_meta.json").read_text())
    y_frc = (min(fmeta["years"]), max(fmeta["years"])); assert y_frc == WINDOW, (y_frc, WINDOW)
    pmeta = json.loads((V5 / "products_cell_year_v5_meta.json").read_text())["products"]
    pr = pd.read_csv(V5 / "products_cell_year_v5.csv")
    prods = []
    for key, name in (("wei", "Wei 2026"), ("liu", "Liu 2024"), ("cci4y", "ESA CCI v4"), ("cci5y", "ESA CCI v5"), ("yk", "Yi–Kimball 2019 (Alaska)"),
                      ("aalto", "Aalto 2018")):
        m = pmeta[key]; period = tuple(int(v) for v in m["period"])
        if key == "liu":
            assert LIU_AVAILABLE[0] <= period[0] and period[1] <= LIU_AVAILABLE[1], period
            avail, avail_src = LIU_AVAILABLE, LIU_AVAILABLE_SRC
        else:
            avail = (min(m["years"]), max(m["years"])) if m.get("years") else period
            avail_src = "data/processed/v5/products_cell_year_v5_meta.json (products.%s.%s)" % (key, "years" if m.get("years") else "period")
        if key == "aalto":                                   # 다년 평균 대표값(연도 = −1): 연도별 비교 없음
            assert (pr.loc[pr["product"] == key, "year"] == -1).all()
            used = None
        else:                                                # 라벨 셀-연도 중 제품 값이 있는 해(v10_numbers.py nyrsprod* 와 같은 정의)
            q = lab.merge(pr[(pr["product"] == key) & pr.value_cm.notna()][["loc_id", "year"]], on=["loc_id", "year"])
            used = (int(q.year.min()), int(q.year.max()))
        prods.append(dict(key=key, name=name, available=avail, available_source=avail_src, used=used, extracted_period=period))
    rows = [dict(group="Labels", name=REG_EN[r], kind="label", runs=year_runs(cnt.index[cnt[r] > 0])) for r in REG]
    rows += [dict(group="Inputs", name="ERA5-Land monthly climate", kind="used", runs=[y_frc]),
             dict(group="Inputs", name="Terrain, soil", kind="static", runs=[])]
    for p in prods:
        if p["key"] == "aalto":
            rows.append(dict(group="Products", name=p["name"], kind="mean", runs=[p["available"]], prod=p))
        else:
            rows.append(dict(group="Products", name=p["name"], kind="product", runs=[p["available"]], used=p["used"], prod=p))
    return dict(lab=lab, n_cells=n_cells, n_rows=n_rows, cnt=cnt, years=years, rows=rows, prods=prods, fmeta=fmeta)


def pick_alaska_cell(d):
    """v7_f2_problem_time.pick_alaska_cell 과 같은 규칙: 관측 연도 수 최다, 동률이면 관측일 누적 TDD 비율(중앙값) 최소, 그다음 loc_id."""
    ak = d[d.region == "Alaska"]; n = ak.groupby("loc_id").year.nunique(); ids = n[n == n.max()].index
    f = ak[ak.loc_id.isin(ids)].groupby("loc_id").tdd_obs_frac.median().sort_index()
    return int(f.sort_values(kind="stable").index[0]), int(n.max()), int(len(ids))


def daily_cum(row, year):
    """forcing_year 한 행의 tdd_cum_1..12(월말 누적) → 일별 누적 TDD(월말 값 사이 선형 보간, 1월 1일 0)."""
    days = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)]); ends = np.cumsum(days)
    cum = np.array([row[f"tdd_cum_{k}"] for k in range(1, 13)], float)
    doy = np.arange(0, ends[-1] + 1, dtype=float)
    return doy, np.interp(doy, np.r_[0, ends], np.r_[0, cum]), cum


def load_d(lab, year=2014):
    d = T.with_forcing(T.load_v5(V5, products=False), "year_obs")
    cell, nyr, ntie = pick_alaska_cell(d)
    fy = pd.read_csv(V5 / "forcing_year.csv"); fc = pd.read_csv(V5 / "forcing_clim.csv")
    row = fy[(fy.loc_id == cell) & (fy.year == year)].iloc[0]
    doy, cum, cum_m = daily_cum(row, year)
    lr = lab[(lab.loc_id == cell) & (lab.year == year)].iloc[0]
    od_raw = lr.obs_doy; od = float(od_raw) if np.isfinite(od_raw) else float(EOS_DOY)
    t_obs_pipe = float(d[(d.loc_id == cell) & (d.year == year)].e5_tdd.iloc[0])          # with_forcing('year_obs') 의 관측일 누적 TDD
    t_obs = float(np.interp(od, doy, cum)); t_full = float(row.e5_tdd); t_clim = float(fc.loc[fc.loc_id == cell, "clim1124_tdd"].iloc[0])
    assert abs(t_obs - t_obs_pipe) < 0.5, (t_obs, t_obs_pipe)
    assert abs(t_full - cum_m[-1]) < 1e-6
    ann = fy[(fy.loc_id == cell) & fy.year.between(2011, 2024)].e5_tdd
    assert abs(float(ann.mean()) - t_clim) < 1e-3, (float(ann.mean()), t_clim)          # clim1124_tdd = 2011–2024 연간 TDD 평균
    return dict(cell=cell, n_years=nyr, n_tied=ntie, year=year, lat=float(lr.lat), lon=float(lr.lon), source=str(lr.source), method=str(lr.method),
                obs_months=str(lr.obs_months), obs_doy_raw=None if not np.isfinite(od_raw) else float(od_raw), obs_doy=od,
                doy=doy, cum=cum, cum_month_end=cum_m.tolist(), t_obs=t_obs, t_obs_pipeline=t_obs_pipe, t_full=t_full, t_clim=t_clim,
                annual_tdd_2011_2024={int(y): float(v) for y, v in zip(fy[(fy.loc_id == cell) & fy.year.between(2011, 2024)].year, ann)})


# ================================================================ a 북극 지도
LAT_MIN = 56.0
CLON = -90.0
BOX = {"Alaska": (-168.0, -141.0, 59.0, 71.5), "Lena": (123.3, 130.1, 71.5, 73.6), "Canada": (-141.0, -112.0, 60.0, 70.0)}


def pfr_image(proj, lat_min, n):
    """ESA CCI 영구동토 확률(0.1°)을 투영 격자(n × n)에 쌍선형 재표집(결측 가중 정규화). 반환 masked 배열, extent(투영 m)."""
    import xarray as xr
    ds = xr.open_dataset(PROC / "cci_pfr_mean_1997_2021.nc"); P = ds["pfr_mean"]
    lat = P["lat"].values.astype(np.float64); lon = P["lon"].values.astype(np.float64); V = P.values.astype(np.float32)
    if lat[0] > lat[-1]:
        lat = lat[::-1]; V = V[::-1]
    dlat = (lat[-1] - lat[0]) / (len(lat) - 1); dlon = (lon[-1] - lon[0]) / (len(lon) - 1)
    valid = np.isfinite(V).astype(np.float32); Vf = np.where(np.isfinite(V), V, 0.0).astype(np.float32)
    Vf = np.c_[Vf, Vf[:, :1]]; valid = np.c_[valid, valid[:, :1]]
    px, py = proj.transform_point(CLON, lat_min, PC); Rm = float(np.hypot(px, py))
    xs = np.linspace(-Rm, Rm, n); X, Y = np.meshgrid(xs, xs)
    ll = PC.transform_points(proj, X.ravel(), Y.ravel()); lo, la = ll[:, 0], ll[:, 1]
    fi = (la - lat[0]) / dlat; fj = ((lo - lon[0]) % 360.0) / dlon
    v = map_coordinates(Vf, [fi, fj], order=1, mode="nearest"); w = map_coordinates(valid, [fi, fj], order=1, mode="nearest")
    out = np.where(w > 0.5, v / np.maximum(w, 1e-6), np.nan)
    out[(la < lat_min) | (la < lat[0])] = np.nan
    return np.ma.masked_invalid(out.reshape(n, n)), [-Rm, Rm, -Rm, Rm], ds.attrs.get("title", "")


def box_line(lo0, lo1, la0, la1, n=60):
    lons = np.r_[np.linspace(lo0, lo1, n), np.full(n, lo1), np.linspace(lo1, lo0, n), np.full(n, lo0)]
    lats = np.r_[np.full(n, la0), np.linspace(la0, la1, n), np.full(n, la1), np.linspace(la1, la0, n)]
    return lons, lats


def draw_a(fig, D):
    x0, y0, dia = 18.0, 4.5, 61.0
    proj = ccrs.NorthPolarStereo(central_longitude=CLON)
    ax = fig.add_axes(rect(x0, y0, dia, dia), projection=proj)
    ax.set_extent([-180, 180, LAT_MIN, 90], PC)
    th = np.linspace(0, 2 * np.pi, 361)
    ax.set_boundary(mpath.Path(np.c_[0.5 + 0.5 * np.cos(th), 0.5 + 0.5 * np.sin(th)]), transform=ax.transAxes)
    ax.set_facecolor(OCEAN)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND, edgecolor="none", zorder=0.5)
    img, ext, title = pfr_image(proj, LAT_MIN - 1.0, 1500)
    im = ax.imshow(img, extent=ext, transform=proj, origin="lower", cmap=PFR_CMAP, vmin=0, vmax=100, interpolation="bilinear", zorder=1,
                   rasterized=True)
    ax.coastlines("50m", linewidth=0.3, color=COAST, zorder=2)
    ax.gridlines(crs=PC, draw_labels=False, linewidth=0.25, color=GRAT, xlocs=range(-180, 180, 30), ylocs=[60, 70, 80], zorder=3)
    ax.spines["geo"].set_linewidth(0.6); ax.spines["geo"].set_edgecolor(INK)
    for la in (60, 70, 80):                                   # 위도 라벨: 0° 자오선(오른쪽, 바다 위)
        ax.text(0.0, la + 0.6, f"{la}°N", transform=PC, fontsize=6.0, ha="center", va="bottom", zorder=6, path_effects=HALO)
    R = float(np.hypot(*proj.transform_point(CLON, LAT_MIN, PC)))
    # 지역 윤곽(경위도 사각형 투영)
    outl = {}
    for r in REG:                                             # 1차: 흰 테두리(모든 상자), 2차: 검은 선(모든 상자). 알래스카 · 캐나다는 141°W 변을 공유한다
        lons, lats = box_line(*BOX[r])
        ax.plot(lons, lats, transform=PC, color="white", lw=1.8, zorder=4.9, solid_joinstyle="miter")
        outl[r] = proj.transform_points(PC, lons, lats)[:, :2]
    for r in REG:
        lons, lats = box_line(*BOX[r])
        ax.plot(lons, lats, transform=PC, color=INK, lw=0.8, zorder=5, solid_joinstyle="miter")
    # 이름 · 라벨 셀 수: 원 밖 왼쪽(오른쪽 맞춤), 가는 지시선으로 윤곽의 가장 가까운 점에 잇는다
    fig.canvas.draw()
    to_mm = lambda xy: np.c_[fig.transFigure.inverted().transform(ax.transData.transform(xy))[:, 0] * W_MM,
                             (1 - fig.transFigure.inverted().transform(ax.transData.transform(xy))[:, 1]) * H_MM]
    # 지시선 끝점(경위도): 레나 = 서쪽 변 가운데, 알래스카 = 서쪽 변 가운데, 캐나다 = 남쪽 변 가운데(알래스카 상자와 맞닿은 서쪽 변을 피한다)
    ANCHOR = {"Lena": (BOX["Lena"][0], (BOX["Lena"][2] + BOX["Lena"][3]) / 2), "Alaska": (BOX["Alaska"][0], (BOX["Alaska"][2] + BOX["Alaska"][3]) / 2),
              "Canada": ((BOX["Canada"][0] + BOX["Canada"][1]) / 2, BOX["Canada"][2])}
    anc = {r: to_mm(np.array([proj.transform_point(*ANCHOR[r], PC)]))[0] for r in REG}
    ymm = {r: float(anc[r][1]) for r in REG}
    order = sorted(REG, key=lambda r: ymm[r]); MIN_GAP = 8.0
    for i in range(1, len(order)):                             # 라벨 사이 최소 간격(mm)
        ymm[order[i]] = max(ymm[order[i]], ymm[order[i - 1]] + MIN_GAP)
    XL = x0 - 1.6
    lab_txt = {}; leaders = {}
    for r in REG:
        t = fig.text(XL / W_MM, 1 - ymm[r] / H_MM, f"{REG_EN[r]}\n{D['n_cells'][r]:,} cells", ha="right", va="center", fontsize=FS, linespacing=1.1)
        lab_txt[r] = t
        a = np.array([XL + 0.6, ymm[r]]); b = anc[r]
        ln = matplotlib.lines.Line2D([a[0] / W_MM, b[0] / W_MM], [1 - a[1] / H_MM, 1 - b[1] / H_MM], transform=fig.transFigure, color=INK, lw=0.5,
                                     zorder=9, solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        fig.add_artist(ln); leaders[r] = dict(label_mm=[round(float(v), 2) for v in a], box_point_mm=[round(float(v), 2) for v in b], box_point_lonlat=list(ANCHOR[r]))
    # 축척(1,000 km): 원 안 오른쪽 아래 바다(래브라도해). 투영 좌표 가로 막대, 길이는 측지 거리로 맞춘다
    geod = Geod(ellps="WGS84")
    xc, yc = 0.53 * R, -0.70 * R
    def dist(L):
        a = PC.transform_point(xc - L / 2, yc, proj); b = PC.transform_point(xc + L / 2, yc, proj)
        return geod.inv(a[0], a[1], b[0], b[1])[2]
    L = 1.0e6
    for _ in range(30):
        L *= 1.0e6 / dist(L)
    ax.plot([xc - L / 2, xc + L / 2], [yc, yc], color=INK, lw=1.0, solid_capstyle="butt", zorder=8, transform=proj)
    ax.text(xc, yc + 0.025 * R, "1,000 km", ha="center", va="bottom", fontsize=6.0, zorder=8, transform=proj, path_effects=HALO)
    sb_ll = PC.transform_point(xc, yc, proj)
    # 색 막대(세로, 원 오른쪽 아래)
    cax = fig.add_axes(rect(x0 + dia + 2.2, y0 + dia * 0.52, 1.8, dia * 0.42))
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 50, 100])
    cb.outline.set_linewidth(0.5); cax.tick_params(labelsize=FS_TICK, length=2.0, width=0.5, pad=1.5)
    cb.set_label("Permafrost probability (%)", fontsize=FS, labelpad=2.0)
    letter(fig, "a", 0.9, 1.2)
    SRC["a"] = dict(inputs=["data/processed/cci_pfr_mean_1997_2021.nc (variable pfr_mean; attrs.title: %s)" % title,
                            "data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1, unique loc_id per region)"],
                    years=dict(permafrost_probability="1997–2021 mean", labels=f"{D['lab'].year.min()}–{D['lab'].year.max()}"),
                    projection=f"North Polar Stereographic, central longitude {CLON:g}°, boundary {LAT_MIN:g}°N; bilinear resampling to 1500 × 1500 px",
                    colormap=f"cmcrameri oslo_r, {PFR_RANGE[0]}–{PFR_RANGE[1]} of the map (0 % white); colour bar 0–100 %", ocean=OCEAN,
                    land_without_value=LAND,
                    region_boxes_lonlat={REG_EN[r]: dict(lon=[BOX[r][0], BOX[r][1]], lat=[BOX[r][2], BOX[r][3]]) for r in REG},
                    label_cell_extent={REG_EN[r]: dict(lat=[float(g.lat.min()), float(g.lat.max())], lon=[float(g.lon.min()), float(g.lon.max())])
                                       for r, g in D["lab"].groupby("region")},
                    leaders_mm=leaders,
                    scale_bar=dict(km=1000, centre_lonlat=[round(float(sb_ll[0]), 2), round(float(sb_ll[1]), 2)],
                                   note="horizontal bar in projected coordinates; geodesic length between its ends = 1,000 km"),
                    numbers_printed={f"{REG_EN[r]} cells": int(D["n_cells"][r]) for r in REG} | {"colour bar ticks (%)": [0, 50, 100],
                                     "graticule labels": ["60°N", "70°N", "80°N"], "scale bar": "1,000 km"},
                    cell_years_per_region={REG_EN[r]: int(D["n_rows"][r]) for r in REG})
    return ax, cax, lab_txt


# ================================================================ b 셀-연도(지역별 작은 막대)
def nice_ticks(vmax):
    for s in (1, 2, 2.5, 4, 5, 10, 20, 25, 40, 50, 100, 200, 250, 400, 500, 1000, 2000, 2500, 4000, 5000, 10000):
        if vmax <= 2 * s:
            return [0, s, 2 * s]
    return [0, vmax]


def draw_b(fig, D):
    xb, wb = 106.0, 72.6; tops = [4.5, 25.0, 45.5]; hb = 16.5
    axes = []
    for i, r in enumerate(REG):
        ax = fig.add_axes(rect(xb, tops[i], wb, hb)); axes.append(ax)
        v = D["cnt"][r].values
        ax.bar(D["years"], v, width=0.68, color=DARK, lw=0, zorder=2)
        tk = nice_ticks(float(v.max())); ax.set_ylim(0, tk[-1]); ax.set_yticks(tk)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_xlim(WINDOW[0] - 0.6, WINDOW[1] + 0.6)
        ax.xaxis.set_major_locator(FixedLocator(list(range(WINDOW[0], WINDOW[1] + 1, 2))))
        ax.xaxis.set_minor_locator(FixedLocator(list(range(WINDOW[0], WINDOW[1] + 1))))
        ax.spines["bottom"].set_bounds(WINDOW[0] - 0.6, WINDOW[1] + 0.6)
        if i < 2:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.text(0.015, 1.0, REG_EN[r], transform=ax.transAxes, ha="left", va="top", fontsize=FS)
    axes[-1].set_xlabel("Observation year")
    fig.text((xb - 9.6) / W_MM, 1 - (tops[0] + (tops[-1] + hb - tops[0]) / 2) / H_MM, "Cell-years", rotation=90, ha="center", va="center", fontsize=FS)
    letter(fig, "b", 93.0, 1.2)
    SRC["b"] = dict(inputs=["data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1; rows = cell-years)"],
                    years=f"{WINDOW[0]}–{WINDOW[1]}",
                    cell_years={REG_EN[r]: {int(y): int(c) for y, c in D["cnt"][r].items()} for r in REG},
                    totals={REG_EN[r]: int(D["cnt"][r].sum()) for r in REG},
                    y_ticks={REG_EN[r]: nice_ticks(float(D["cnt"][r].max())) for r in REG}, bar_colour=DARK,
                    numbers_printed="axis ticks only (y ticks per region listed in y_ticks; x ticks 2010–2024 every 2 years)")
    return axes


# ================================================================ c 자료 기간
def draw_c(fig, D):
    xa, wa, ya, ha = 44.5, 78.0, 81.0, 32.5
    ax = fig.add_axes(rect(xa, ya, wa, ha))
    rows = D["rows"]; gap = 0.55; pos = []; y = 0.0; prev = None
    for r in rows:
        if prev is not None and r["group"] != prev:
            y += gap
        pos.append(y); y += 1.0; prev = r["group"]
    ymax = y - 1.0
    bh = 0.62
    ax.axvspan(WINDOW[0] - 0.5, WINDOW[1] + 0.5, color=SHADE, lw=0, zorder=0)
    for r, yy in zip(rows, pos):
        if r["kind"] in ("label", "used"):
            for a, b in r["runs"]:
                ax.add_patch(Rectangle((a - 0.5, yy - bh / 2), b - a + 1, bh, facecolor=DARK, edgecolor="none", zorder=3))
        elif r["kind"] == "static":
            ax.add_patch(Rectangle((XLIM_C[0], yy - bh / 2), XLIM_C[1] - XLIM_C[0], bh, facecolor="none", edgecolor="#7A7A7A", lw=0.0,
                                   hatch="//////", zorder=3))
        elif r["kind"] == "product":
            a, b = r["runs"][0]
            ax.add_patch(Rectangle((a - 0.5, yy - bh / 2), b - a + 1, bh, facecolor=LIGHT, edgecolor="none", zorder=2))
            ua, ub = r["used"]
            ax.add_patch(Rectangle((ua - 0.5, yy - bh / 2), ub - ua + 1, bh, facecolor=DARK, edgecolor="none", zorder=3))
        elif r["kind"] == "mean":
            a, b = r["runs"][0]
            ax.add_patch(Rectangle((a - 0.5 + 0.06, yy - bh / 2 + 0.03), b - a + 1 - 0.12, bh - 0.06, facecolor="none", edgecolor=DARK, lw=0.6,
                                   zorder=3))
    ax.set_xlim(*XLIM_C); ax.set_ylim(ymax + 0.6, -0.6)
    ax.xaxis.set_major_locator(FixedLocator([2000, 2005, 2010, 2015, 2020, 2025]))
    ax.xaxis.set_minor_locator(FixedLocator(list(range(1997, 2026))))
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Year")
    # 행 이름(오른쪽 맞춤, 축 왼쪽) · 묶음 이름(왼쪽 맞춤, 묶음 가운데)
    tr = ax.get_yaxis_transform()
    for r, yy in zip(rows, pos):
        ax.text(-0.012, yy, r["name"], transform=tr, ha="right", va="center", fontsize=FS_SMALL)
    groups = {}
    for r, yy in zip(rows, pos):
        groups.setdefault(r["group"], []).append(yy)
    blend = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for g, ys in groups.items():
        ax.text(0.9 / W_MM, float(np.mean(ys)), g, transform=blend, fontsize=FS, ha="left", va="center", clip_on=False)
    # 열쇠(축 위, 오른쪽 맞춤)
    hs = [Patch(facecolor=DARK, edgecolor="none"), Patch(facecolor=LIGHT, edgecolor="none"), Patch(facecolor="none", edgecolor=DARK, lw=0.6),
          Patch(facecolor="none", edgecolor="#7A7A7A", lw=0.0, hatch="//////"), Patch(facecolor=SHADE, edgecolor="none")]
    ax.legend(hs, ["Years used", "Available", "Multi-year mean", "Static (no year)", "Analysis window"], loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=5, frameon=False,
              fontsize=FS_SMALL, handlelength=1.4, handleheight=0.9, handletextpad=0.5, columnspacing=1.2, borderaxespad=0.45, borderpad=0.0)
    letter(fig, "c", 0.9, 74.5)
    SRC["c"] = dict(inputs=["data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1)", "data/processed/v5/forcing_meta.json (years)",
                            "data/processed/v5/products_cell_year_v5_meta.json (products.*.years / period)",
                            "data/processed/v5/products_cell_year_v5.csv (years with a value at label cell-years)", LIU_AVAILABLE_SRC],
                    axis_years=[1997, 2025], analysis_window=list(WINDOW),
                    rows=[dict(group=r["group"], name=r["name"], encoding={"label": "dark band = years with labels", "used": "dark band = years used",
                                                                         "static": "hatched full-width band = no time dimension; legend: Static (no year)",
                                                                         "product": "light band = available period, dark band = years used for comparison",
                                                                         "mean": "outlined band = one multi-year mean value (2000–2014); legend: Multi-year mean"}[r["kind"]],
                               years=(runs_str(r["runs"]) if r["runs"] else "static (no year)"),
                               **({"years_used": f"{r['used'][0]}–{r['used'][1]}", "available_source": r["prod"]["available_source"],
                                   "extracted_years_in_project": f"{r['prod']['extracted_period'][0]}–{r['prod']['extracted_period'][1]}"} if r["kind"] == "product" else {}),
                               **({"available_source": r["prod"]["available_source"]} if r["kind"] == "mean" else {})) for r in rows],
                    years_used_definition="years of label cell-years (in_forcing == 1) that have a product value; same as scripts/2_evaluation/v10_numbers.py nyrsprod*",
                    numbers_printed="axis ticks only (2000, 2005, 2010, 2015, 2020, 2025); product names carry publication years")
    return ax


# ================================================================ d 기후 입력의 시간 정의
def draw_d(fig, Dd):
    xa, wa, ya, ha = 140.0, 38.5, 81.0, 32.5
    ax = fig.add_axes(rect(xa, ya, wa, ha))
    yr = Dd["year"]; doy, cum = Dd["doy"], Dd["cum"]; od = Dd["obs_doy"]
    import datetime as _dt
    _d = _dt.date(yr, 1, 1) + _dt.timedelta(days=int(round(od)) - 1); obs_lab = f"{_d.day} {_d.strftime('%b')}"   # doy 244 → 1 Sep
    ndays = 366 if calendar.isleap(yr) else 365
    ax.plot(doy, cum, color=INK, lw=LW_DATA, zorder=4, solid_capstyle="butt")
    ax.plot([od, od], [0, Dd["t_obs"]], color=REFGREY, lw=LW_REF, ls=(0, (1.2, 1.2)), zorder=2)
    ax.plot([0, od], [Dd["t_obs"], Dd["t_obs"]], color=REFGREY, lw=LW_REF, ls=(0, (1.2, 1.2)), zorder=2)   # 관측일 값 → y 축
    ax.plot([od], [Dd["t_obs"]], "o", ms=3.4, mfc="white", mec=INK, mew=0.7, zorder=6)
    ax.plot([ndays], [Dd["t_full"]], "o", ms=3.0, mfc=INK, mec=INK, mew=0.0, zorder=6, clip_on=False)
    ax.axhline(Dd["t_clim"], color=REFGREY, lw=0.8, ls=(0, (3.5, 1.8)), zorder=3)
    ytop = float(np.ceil(Dd["t_clim"] * 1.17 / 100.0) * 100.0)
    ax.set_xlim(0, ndays); ax.set_ylim(0, ytop)
    starts = np.r_[0, np.cumsum([calendar.monthrange(yr, m)[1] for m in range(1, 13)])]
    ax.xaxis.set_major_locator(FixedLocator(starts.tolist()))
    ax.xaxis.set_minor_locator(FixedLocator(((starts[:-1] + starts[1:]) / 2).tolist()))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: ""))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: "JFMAMJJASOND"[int(np.searchsorted(starts, x) - 1)]))
    ax.tick_params(axis="x", which="minor", length=0, labelsize=FS_TICK, pad=2.0)
    ax.tick_params(axis="x", which="major", length=2.0)
    ax.yaxis.set_major_locator(MultipleLocator(100 if ytop <= 600 else 200))
    ax.set_xlabel(f"Month ({yr})"); ax.set_ylabel("Cumulative TDD (°C·d)")
    t1 = ax.text(od + 6, Dd["t_obs"] * 0.42, "Observation\nday (%s)" % obs_lab, ha="left", va="center", fontsize=FS_SMALL, linespacing=1.05, zorder=7)
    t2 = ax.text(ndays - 3, Dd["t_full"] + 0.035 * ytop, "Full season", ha="right", va="bottom", fontsize=FS_SMALL, zorder=7)
    t3 = ax.text(6, Dd["t_clim"] + 0.03 * ytop, "Mean 2011–2024", ha="left", va="bottom", fontsize=FS_SMALL, zorder=7)
    letter(fig, "d", 128.5, 74.5)
    SRC["d"] = dict(inputs=["data/processed/v5/forcing_year.csv (tdd_cum_1..12, e5_tdd)", "data/processed/v5/forcing_clim.csv (clim1124_tdd)",
                            "data/processed/v5/labels_cell_year_v5.csv (cell selection, obs_doy)",
                            "polar.testbed_v5.with_forcing('year_obs') (tdd_obs_frac for cell selection, observation-day TDD cross-check)"],
                    years=dict(curve=str(yr), mean="2011–2024"),
                    cell=dict(loc_id=Dd["cell"], lat=Dd["lat"], lon=Dd["lon"], source=Dd["source"], method=Dd["method"], n_label_years=Dd["n_years"],
                              n_cells_tied_on_years=Dd["n_tied"],
                              selection_rule="most label years in Alaska; ties → smallest median observation-day TDD fraction; then loc_id (v7_f2_problem_time.pick_alaska_cell)"),
                    observation_day=dict(doy=Dd["obs_doy"], doy_in_label_table=Dd["obs_doy_raw"], obs_months=Dd["obs_months"],
                                         rule="CALM 'eos' records carry no date: pipeline convention 1 September (doy 244)"),
                    curve="daily linear interpolation of month-end cumulative TDD (0 on 1 January)",
                    cum_month_end_tdd=[round(v, 3) for v in Dd["cum_month_end"]],
                    values_degC_day=dict(observation_day=round(Dd["t_obs"], 2), observation_day_pipeline=round(Dd["t_obs_pipeline"], 2),
                                         full_season=round(Dd["t_full"], 2), mean_2011_2024=round(Dd["t_clim"], 2)),
                    annual_tdd_2011_2024={k: round(v, 2) for k, v in Dd["annual_tdd_2011_2024"].items()},
                    numbers_printed="axis ticks; date in the observation-day label (1 Sep = CALM eos convention); the three values are read from the y axis (listed in values_degC_day)")
    return ax, (t1, t2, t3)


# ================================================================ 점검(겹침 · 크기 · 글꼴 · 캔버스 안)
def _all_axes(fig):
    out = []
    def rec(a):
        out.append(a); [rec(c) for c in a.child_axes]
    for a in fig.axes:
        rec(a)
    return out


def collect_texts(fig):
    """그려지는 글자 목록(이름, 상자, 크기, 글꼴, 축). 눈금 라벨은 보이는 범위의 눈금만."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); items = []
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            items.append((t.get_text(), t.get_window_extent(r), t.get_fontsize(), t.get_fontname(), None, t))
    for ax in _all_axes(fig):
        ts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for ticks in (axis.get_major_ticks(), axis.get_minor_ticks()):
                for tk in ticks:
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


def qa(fig, extra_checks=()):
    items = collect_texts(fig); W, H = fig.bbox.width, fig.bbox.height; ppt = fig.dpi / 72.0
    bad = []
    for name, bb, fs, fn, ax, t in items:
        if fs < 6.0 - 1e-9:
            bad.append(f"size {fs} < 6: '{name}'")
        if fn != "FreeSans":
            bad.append(f"font {fn}: '{name}'")
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            bad.append(f"outside canvas: '{name}' {bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            w = min(a.x1, b.x1) - max(a.x0, b.x0); h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > 0.5 and h > 0.5:
                bad.append(f"text overlap: '{items[i][0]}' x '{items[j][0]}'")
    for fn_ in extra_checks:
        bad += fn_()
    sizes = sorted({round(float(it[2]), 2) for it in items}); fonts = sorted({it[3] for it in items})
    print(f"[QA] texts {len(items)}, sizes {sizes}, fonts {fonts}, problems {len(bad)}" + "".join("\n   " + b for b in bad))
    return dict(n_texts=len(items), font_sizes_pt=sizes, fonts=fonts, problems=bad)


def text_vs_lines(ax, texts, margin_pt=1.0, skip=()):
    """축 안 글자 상자와 그 축의 선(Line2D) · 막대(Rectangle) 사이 겹침(여백 margin_pt)."""
    def run():
        fig = ax.figure; fig.canvas.draw(); r = fig.canvas.get_renderer(); m = margin_pt * fig.dpi / 72.0; out = []
        for t in texts:
            bb = t.get_window_extent(r)
            for ln in ax.lines:
                if ln in skip:
                    continue
                xy = ln.get_xydata()
                if len(xy) < 2:
                    pts = ln.get_transform().transform(xy)
                else:
                    seg = [np.c_[np.linspace(x0, x1, 80), np.linspace(y0, y1, 80)] for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:])]
                    pts = ln.get_transform().transform(np.concatenate(seg))
                hit = ((pts[:, 0] >= bb.x0 - m) & (pts[:, 0] <= bb.x1 + m) & (pts[:, 1] >= bb.y0 - m) & (pts[:, 1] <= bb.y1 + m)).any()
                if hit:
                    out.append(f"text '{t.get_text()}' touches line ({ln.get_label()})")
            for p in ax.patches:
                pb = p.get_window_extent(r)
                if pb.width > 0 and not (bb.x1 + m < pb.x0 or bb.x0 - m > pb.x1 or bb.y1 + m < pb.y0 or bb.y0 - m > pb.y1):
                    out.append(f"text '{t.get_text()}' touches patch")
        return out
    return run


def save(fig):
    OUT.mkdir(parents=True, exist_ok=True); out = []
    for ext in ("pdf", "svg", "png"):
        p = OUT / f"{STEM}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None, facecolor="white")
        out.append(p)
    return out


def main():
    setup()
    D = load(); Dd = load_d(D["lab"], 2014)
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax_a, cax, lab_txt = draw_a(fig, D)
    axes_b = draw_b(fig, D)
    ax_c = draw_c(fig, D)
    ax_d, td = draw_d(fig, Dd)
    checks = [text_vs_lines(ax_d, td, 1.0)] + [text_vs_lines(a, a.texts, 1.0) for a in axes_b]
    SRC["qa"] = qa(fig, checks)
    SRC["figure"] = dict(width_mm=W_MM, height_mm=H_MM, png_dpi=600, font="FreeSans", text_pt=FS, tick_pt=FS_TICK, letter_pt=FS_LETTER,
                         palette="POLAR_PALETTE=%s" % os.environ.get("POLAR_PALETTE", "deck"), script="scripts/4_visualization/paper_v10/fig1_data.py")
    out = save(fig); plt.close(fig)
    sd = OUT / "source_data"; sd.mkdir(parents=True, exist_ok=True)
    (sd / f"{STEM}.json").write_text(json.dumps(SRC, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    for p in out + [sd / f"{STEM}.json"]:
        print(p)


if __name__ == "__main__":
    main()
