"""v7 F6 지도 그림: 1 km 연도 라벨 ALT 지도(알래스카 평균·연도 편차·확대 2곳·적용 범위, 레나 델타 2016·사모일로프 확대).
250 m 지형 음영 위에 1 km 셀 값을 얹는다(드레이프). v6_maps.py 를 바탕으로 다음을 고쳤다.
  셀 → 화소 대응은 셀 격자 산술(ky = ⌊lat/0.009⌋, kx = ⌊lon/(0.009/cos φ)⌋)이라 셀이 정확한 사각형이다.
  그리는 셀 = 지도 격자 영역(gray == 0, pfr_mean ≥ 10, dem_tile_missing == 0) ∩ DEM 음영 있음. 적용 범위 밖은 육지색과 40 % 혼합(경계는 가우스 완화).
  화소 범주: 그린 셀 / 자료 없음(격자 안이지만 못 그린 셀: 빗금) / 비영구동토 육지(격자 밖 육지: 평평한 중립 회색, 음영 없음) / 물(옅은 파랑).
  표시 보간(v8, 2026-10-15): 개관 패널(알래스카 평균·편차, 레나 2016)은 1 km 셀 중심값을 250 m 음영 격자에 쌍선형 보간해 얹는다(interp_cells).
    보간은 지리 좌표에서 한다(위도 방향 = 셀 행, 경도 방향 = 행마다 다른 간격 CELL/cos φ; (ky, kx) 격자 번호 공간은 행 사이 kx 가 2 셀 넘게 어긋나므로 쓰지 않는다).
    값이 없는 이웃은 가중치에서 빼고(NaN 인식), 그릴지 여부·자료 없음·적용 범위는 화소가 든 셀 그대로라 영역 경계는 1 km 셀 경계를 지킨다.
    확대 창(논문 d·g, 덱 R07 b·c, R08 a·b, R09 b)과 적용 범위 지도는 1 km 셀 그대로(보간 없음). source_data/F6_maps.json 의 display 에 기록한다.
    틈 메우기(v8 2차): 개관 패널에서 그려지는 셀 사이의 물·자료 없음 틈 가운데 4-연결 성분이 FILL_MAX(2) 셀 이하인 것은 보간값으로 메운다(fill_rows; 레나 북부 평원의
      흰 점 얼룩 제거). 3 셀 이상 틈(수로·호수)은 그대로. 마스크(그릴지·비영구동토·자료 없음)는 1 km 그대로라 서남부 산발 영구동토대의 각진 경계는 자료 경계다(메타에 적음).
  관측 셀 = 관측 지점을 담은 1 km 지도 셀(빈 원; v8 2차: 원마다 흰 테를 먼저 그려 겹친 원이 따로 보인다, obs_marks). 축척 막대(길이 위, 끝 눈금), 큰 패널 경위선 라벨,
    확대 창 연결선, 가로 컬러바(ALT 10 cm·차이 5 cm 눈금). 발산 색표(broc)의 색각 이상 모의 ΔE 를 메타에 적는다(cvd_check).
  범례는 지도 틀 밖(v8): 패널 a 의 범주 범례(비영구동토 육지·자료 없음·관측 셀·물)는 경도 라벨 아래 2 열, 적용 범위 범례(안·밖)도 패널 아래 1 줄.
    '자료 없음' 손잡이는 지도와 같은 주기·선 폭의 빗금을 직접 그린다(HatchHandler). 범례의 육지·빗금 색은 지도처럼 평균 음영을 곱한 색이다.
  배치(논문): 행 1 = a(2×2 칸) | d1, d2(왼 열) | b, c(오른 열) 가 같은 세로 띠를 차지한다. 행 2 = e | f | g 같은 높이. 확대 2곳은 문자 d 하나에 지명 라벨.
    컬러바: ALT 는 d 열 아래, 차이는 c 열 아래, 레나 ALT 는 f·g 가로 중앙(모두 같은 띠). 범례는 a·e 아래 왼쪽.
  슬라이드판(--slide): 패널마다 컬러바 사본(b·c·d·g 단독 크롭에도 색-값 대응이 들어가게), 단독 크롭은 연결선 없이, 묶음 크롭(abcd, efg, fg)은 연결선 포함.
    a 의 컬러바는 범례 오른쪽(범례 폭을 재서 놓는다). 확대 창 관측 원은 저장 배율 1.69 를 넣어 최종 지름 ≥ 6 pt·테두리 ≥ 1 pt.
  덱 원판(--deck): 슬라이드 본문 한 장 크기(12.0 × 5.10 in, 300 dpi, 3600 × 1530 px)에 패널을 직접 배치한다. 글자 pt 를 직접 지정(눈금·범례 12, 직접 라벨 13,
    컬러바 라벨 14, 패널 문자 14 굵게; 세 장 모두 같은 Med("deck") 값). 패널 문자는 장마다 읽는 순서대로 다시 매긴다(논문 문자와 다르다).
    R07 = 알래스카 평균(a) | 확대 2곳 세로(b, c) | 2014·2022 편차 세로(d, e). 세 열의 아래 끝을 맞추고, a 아래 = 경도 라벨 + 범주 범례(2 열),
      b·c 아래 = ALT 컬러바, d·e 아래 = 차이 컬러바.
    R08 = 확대 2곳 가로(a, b, 원 ≥ 6 pt·테두리 1 pt) + 적용 범위(c, 범례는 c 아래). 확대 아래 ALT 컬러바와 '적용 범위 안/밖' 범례(밖 = 육지색 40 % 혼합).
    R09 = 레나 델타 2016(a) + 사모일로프-쿠룽나흐 확대(b, 창 비율은 패널 비율). 확대 아래 ALT 컬러바, 창에 적용 범위 밖 셀이 있으면 범례.
    같은 자료·그리기 함수(draw_*)를 논문판과 공유한다. 확대 창 범위는 패널 비율을 따른다(R07·R08 은 논문과 같음, R09 는 폭 24 km × 패널 비율).
    v8 2차: 세 장 모두 폭 12 in 을 거의 다 쓴다(높이에 묶인 열을 뺀 남는 폭은 열 간격과 양쪽 여백으로 나눔; 덱 빌더는 그림을 12 in 폭에 맞춰 가운데 놓고 캡션은 아래).
입력  data/processed/v5/maps/<Alaska|Lena>/{mean_2011_2024, pred_<year>}.csv.gz, data/processed/map_<key>/{<key>_grid_x25_v1.csv.gz, hillshade_v1/{shade,dem}_250m.tif},
      data/processed/v5/labels_cell_year_v5.csv(in_forcing == 1).
산출  outputs/figures/paper/v7/F6_maps.{pdf,svg,png}, --slide 이면 v7/slide/F6_maps.png 와 패널 크롭(a–g, abcd, efg, fg),
      --deck 이면 v7/deck/F6_maps__deck_{R07,R08,R09}.png. 그림 메타·수치 v7/source_data/F6_maps.json (판마다 덮어쓴다: display, 색 범위, 관측 셀 수).
실행  OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v7_f6_maps.py [--slide | --deck] [--cache DIR]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rfeat
from rasterio.windows import Window
from affine import Affine
from pyproj import Transformer
from scipy.ndimage import gaussian_filter, label as ndi_label
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3"))
import polar.style_v5 as S
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap, to_rgb
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import matplotlib.patheffects as pe
import maps_alt_v3 as MV
PROC = ROOT / "data" / "processed"; MAPS = PROC / "v5" / "maps"; OUT = ROOT / "outputs" / "figures" / "paper" / "v7"
CACHE = Path(os.environ.get("F6_CACHE", OUT / "_cache"))
DECK_OUT = OUT / "deck"; DECK_W_IN = 12.0; DECK_H_IN = 5.10          # 덱 본문 한 장(상자 없는 슬라이드)
CELL = 0.009                      # 셀 위도 간격(도). 경도 간격 = CELL / cos(행 중심 위도)
PX = 250.0                        # 음영 화소(m)
PT_MM = 72.0 / 25.4               # 1 mm 당 pt
W_SHADE = 0.38                    # 음영 혼합 비중
FADE = 0.40                       # 적용 범위 밖 셀: 육지색 혼합 비율
FILL_MAX = 2                      # 개관 패널 표시용 틈 메우기: 그려지는 셀 사이 물·자료 없음 틈의 4-연결 성분이 이 셀 수 이하면 보간값으로 메운다(확대 창은 0, 메우지 않음; v8).
                                  #   1 셀만 메우면 레나 북부 평원에 2 셀 틈 얼룩이 남고, 3 셀은 2 셀과 차이가 작다(2026-10-15 비교 렌더) → 2
LAND = np.array(to_rgb("#E6E6E6"), np.float32)      # 비영구동토 육지(격자 밖)
WATER = np.array(to_rgb("#EAF2F9"), np.float32)     # 물
HATCH = np.array(to_rgb("#A9A9A9"), np.float32)     # 자료 없음 빗금 선
HATCH_MM = (0.75, 0.20)                              # 빗금 주기·선 폭(mm, 종이 기준)
COAST = "#8C8C8C"; GRAT = "#9A9A9A"; FRAME = "#555555"; LINK = "#8C8C8C"; OBS = "#1A1A1A"
AOA_IN = "#8BBCDF"; AOA_OUT = "#B4BAC5"             # 적용 범위 안·밖(밖 = 푸른 기 있는 회색; 비영구동토 육지 #E6E6E6 보다 뚜렷이 짙어 범례 없이도 구분된다, v8)
REG = {"alaska": dict(region="Alaska", box=(-168.0, -141.0, 59.0, 71.5), pad=2e4, lons=[-170, -165, -160, -155, -150, -145, -140], lats=[60, 65, 70],
                      lat_lab=[60, 65, 70], step=1),
       "lena": dict(region="Lena", box=(123.3, 130.1, 71.5, 73.6), pad=3e3, lons=[122, 124, 126, 128, 130, 132], lats=[71, 72, 73, 74],
                    lat_lab=[72, 73], step=1)}
ZOOM = {"alaska": [("North Slope, Toolik", (-149.55, 68.75), 25e3), ("Interior, Fairbanks", (-147.85, 64.85), 25e3)],
        "lena": [("Samoylov–Kurungnakh", (126.4, 72.36), 12e3)]}
LINKS: list = []                  # 확대 창 연결선(슬라이드 단독 크롭에서는 숨긴다)
SLIDE_SCALE = 1.69                # 슬라이드판 저장 배율(style_v5.setup(scale=1.69)): 종이 크기로 그려 1.69 배로 저장한다
META: dict = {}                   # source_data/F6_maps.json 에 쓸 그림 메타(표시 보간·색 범위·관측 셀 수)


class Med:
    """매체별 크기(pt·mm). paper = 논문판(슬라이드 크롭판도 종이 크기로 그린 뒤 1.69 배 확대하므로 같다). deck = 덱 원판(12 in 폭, 토큰 slide 값)."""

    def __init__(self, kind="paper"):
        """kind: paper(논문판) · slide(논문 배치를 1.69 배로 저장하는 슬라이드판) · deck(덱 원판)."""
        self.kind = kind; self.paper = kind != "deck"; self.slide = kind == "slide"; Fs = S.F["slide"]; Ls = S.TOK["lines"]["slide"]
        if self.paper:
            self.fs_tick, self.fs_label, self.fs_cbar, self.fs_letter = 6.0, 6.5, 6.5, S.F["paper"]["panel_letter_pt"]   # 눈금·범례·축척 / 직접 라벨 / 컬러바 라벨 / 패널 문자
            self.lw_frame, self.lw_coast, self.lw_grat, self.lw_scale, self.lw_box, self.lw_link = 0.5, 0.35, 0.35, 0.9, 0.7, 0.5
            self.tick_len, self.tick_w, self.tick_pad = 1.5, 0.5, 1.2
            self.halo = 1.8; self.obs_over = (1.2, 0.25); self.obs_zoom_lw = 1.0 / SLIDE_SCALE if self.slide else 0.35  # 관측 셀 원: 개관(면적 pt², 선 pt), 확대(선 pt; 슬라이드 최종 1 pt)
            self.obs_case = (0.25, 0.8 / SLIDE_SCALE if self.slide else 0.3)        # 원 테두리 바깥 흰 테(pt; 개관, 확대): 겹친 원이 따로 보이게 한다(v8)
            self.hatch_mm = HATCH_MM; self.cb_h = 2.2; self.leg_ms, self.leg_mew, self.leg_edge = 2.6, 0.4, 0.3
            self.leg = dict(handlelength=1.5, handleheight=1.0, labelspacing=0.25, handletextpad=0.5, columnspacing=1.2)
        else:
            self.fs_tick, self.fs_label, self.fs_cbar, self.fs_letter = Fs["tick_pt"], Fs["direct_label_pt"], Fs["axis_label_pt"], Fs["panel_letter_pt"]
            self.lw_frame, self.lw_coast, self.lw_grat, self.lw_scale, self.lw_box, self.lw_link = Ls["axis_pt"], 0.7, Ls["graticule_pt"], Ls["data_pt"], 1.3, 0.9
            self.tick_len, self.tick_w, self.tick_pad = 3.0, 1.0, 2.0
            self.halo = 3.0; self.obs_over = (3.6, 0.45); self.obs_zoom_lw = 1.0; self.obs_case = (0.55, 0.8)
            self.hatch_mm = (HATCH_MM[0] * SLIDE_SCALE, HATCH_MM[1] * SLIDE_SCALE); self.cb_h = 3.5; self.leg_ms, self.leg_mew, self.leg_edge = 4.5, 0.7, 0.5
            self.leg = dict(handlelength=1.6, handleheight=1.0, labelspacing=0.3, handletextpad=0.6, columnspacing=1.5)
        self.halo_fx = [pe.withStroke(linewidth=self.halo, foreground="white")]

    def obs_zoom_s(self, mm_per_km):
        """확대 패널 관측 셀 원의 면적(pt²). 논문판: 지름 0.7 km 상당(2.5 pt–1.0 mm). 슬라이드판: 저장 배율을 넣어 최종 지름 ≥ 6 pt. 덱: 지름 1.2 km 상당, 6–10 pt."""
        if self.paper:
            d_pt = max(2.5, min(1.0, 0.7 * mm_per_km) * PT_MM)
            if self.slide:
                d_pt = max(d_pt, 6.0 / SLIDE_SCALE)
        else:
            d_pt = float(np.clip(1.2 * mm_per_km * PT_MM, 6.0, 10.0))
        return d_pt ** 2


# ================================================================ 자료
def cell_table(key):
    """지도 격자 셀 표(x25). domain = 그릴 수 있는 셀, water = GSW 물 셀, nodata = 격자 안이지만 값이 없는 셀. (ky, kx) → 행 번호 조회표를 함께 돌려준다."""
    g = pd.read_csv(PROC / f"map_{key}" / f"{key}_grid_x25_v1.csv.gz", usecols=["cell_id", "ky", "kx", "gray", "gray_reason", "pfr_mean", "dem_tile_missing"], dtype={"cell_id": str})
    g["water"] = (g.gray_reason == "water").values
    g["domain"] = (g.gray == 0) & (g.pfr_mean >= 10) & (g.dem_tile_missing == 0)
    g = g.reset_index(drop=True)
    ky0, kx0 = int(g.ky.min()), int(g.kx.min())
    lut = np.full((int(g.ky.max()) - ky0 + 1, int(g.kx.max()) - kx0 + 1), -1, np.int32)
    lut[g.ky.values - ky0, g.kx.values - kx0] = np.arange(len(g), dtype=np.int32)
    return g, lut, ky0, kx0


def cell_values(key, cells, series):
    """cell_id → 값 시리즈를 셀 표 순서의 배열로(영역 밖 셀은 NaN)."""
    v = series.reindex(cells.cell_id.values).values.astype(np.float32)
    v[~cells.domain.values] = np.nan
    return v


def obs_cells(lab, region):
    """관측 지점(loc_id)을 담은 1 km 지도 셀의 중심 경위도(중복 없이). 셀 산술은 Base 의 화소 → 셀 대응과 같다."""
    d = lab[lab.region == region].drop_duplicates("loc_id")
    ky = np.floor(d.lat.values / CELL).astype(np.int64); dx = CELL / np.cos(np.radians((ky + 0.5) * CELL)); kx = np.floor(d.lon.values / dx).astype(np.int64)
    u = pd.DataFrame({"ky": ky, "kx": kx}).drop_duplicates()
    lat = (u.ky.values + 0.5) * CELL; lon = (u.kx.values + 0.5) * CELL / np.cos(np.radians(lat))
    return lon, lat, len(d)


class Base:
    """한 지역의 음영 격자(250 m × step)와 화소별 셀 번호·육지·DEM 유무·경위도(표시 보간용). window = 투영 좌표 (x0, x1, y0, y1).
    캐시 이름에 화소 창을 넣어 창이 다르면 다시 만든다(v2: 경위도 포함)."""

    def __init__(self, key, cells, lut, ky0, kx0, step=1, window=None, tag="full"):
        self.key = key; self.step = step; self.lut, self.ky0, self.kx0 = lut, ky0, kx0
        hs = PROC / f"map_{key}" / "hillshade_v1"
        with rasterio.open(hs / "shade_250m.tif") as src:
            tr, crs, H, W = src.transform, src.crs, src.height, src.width
        self.crs = crs; self.tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        c0, r0, w, h = 0, 0, W, H
        if window is not None:
            x0, x1, y0, y1 = window
            c0 = max(0, int(np.floor((x0 - tr.c) / tr.a))); c1 = min(W, int(np.ceil((x1 - tr.c) / tr.a)))
            r0 = max(0, int(np.floor((y1 - tr.f) / tr.e))); r1 = min(H, int(np.ceil((y0 - tr.f) / tr.e)))
            w, h = c1 - c0, r1 - r0
        w -= w % step; h -= h % step
        cache = CACHE / (f"{key}_{tag}_s{step}_v2.npz" if window is None else f"{key}_{tag}_{c0}_{r0}_{w}x{h}_s{step}_v2.npz")
        if cache.exists():
            z = np.load(cache); self.shade, self.valid, self.idx, self.land, self.lat, self.lon = z["shade"], z["valid"], z["idx"], z["land"], z["lat"], z["lon"]
        else:
            t = time.time()
            with rasterio.open(hs / "shade_250m.tif") as src:
                shade = src.read(1, window=Window(c0, r0, w, h)).astype(np.float32) / 255.0
            with rasterio.open(hs / "dem_250m.tif") as src:
                valid = np.isfinite(src.read(1, window=Window(c0, r0, w, h)))
            if step > 1:
                shade = shade.reshape(h // step, step, w // step, step).mean(axis=(1, 3))
                valid = valid.reshape(h // step, step, w // step, step).mean(axis=(1, 3)) >= 0.5
            hh, ww = shade.shape
            gx = tr.c + (c0 + (np.arange(ww) + 0.5) * step) * tr.a; gy = tr.f + (r0 + (np.arange(hh) + 0.5) * step) * tr.e
            XX, YY = np.meshgrid(gx, gy)
            inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = inv.transform(XX.ravel(), YY.ravel()); del XX, YY
            ky = np.floor(lat / CELL).astype(np.int64); dx = CELL / np.cos(np.radians((ky + 0.5) * CELL)); kx = np.floor(lon / dx).astype(np.int64)
            ky -= ky0; kx -= kx0
            ok = (ky >= 0) & (ky < lut.shape[0]) & (kx >= 0) & (kx < lut.shape[1])
            idx = np.full(ky.shape, -1, np.int32); idx[ok] = lut[ky[ok], kx[ok]]; idx = idx.reshape(hh, ww); del ky, kx, dx, ok
            self.lat = lat.reshape(hh, ww).astype(np.float32); self.lon = lon.reshape(hh, ww).astype(np.float32); del lat, lon
            tr2 = Affine(tr.a * step, 0.0, tr.c + c0 * tr.a, 0.0, tr.e * step, tr.f + r0 * tr.e)
            from shapely.ops import transform as shp_transform
            geoms = [shp_transform(self.tf.transform, g) for g in MV.ne_layers(key)["land"]]
            land = rfeat.rasterize(((g, 1) for g in geoms), out_shape=(hh, ww), transform=tr2, fill=0, dtype="uint8").astype(bool)
            self.shade, self.valid, self.idx, self.land = shade.astype(np.float32), valid, idx, land
            CACHE.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, shade=self.shade, valid=valid, idx=idx, land=land, lat=self.lat, lon=self.lon)
            print(f"  base {key}/{tag} s{step}: {hh}×{ww} px, {time.time() - t:.1f} s")
        hh, ww = self.shade.shape
        lv = self.valid & self.land; self.shade_fill = float(self.shade[lv].mean()) if lv.any() else 1.0   # DEM 없는 육지의 밝기 = 육지 음영 평균
        self.extent = [tr.c + c0 * tr.a, tr.c + (c0 + ww * step) * tr.a, tr.f + (r0 + hh * step) * tr.e, tr.f + r0 * tr.e]
        self.water_cell = np.zeros(self.idx.shape, bool); m = self.idx >= 0; self.water_cell[m] = cells.water.values[self.idx[m]]

    def ppm(self, panel_mm, lim):
        """종이 1 mm 당 이 격자의 화소 수(패널 폭 panel_mm 에 투영 범위 lim 이 들어간다)."""
        return (lim[1] - lim[0]) / (PX * self.step) / panel_mm

    def hatch_px(self, panel_mm, lim, mm=HATCH_MM):
        """종이 위 빗금 주기·선 폭(mm)을 화소 수로 바꾼다."""
        ppm = self.ppm(panel_mm, lim)
        return max(3, int(round(mm[0] * ppm))), max(1, int(round(mm[1] * ppm)))


def interp_cells(base, val, rows=256):
    """표시용 쌍선형 보간: 1 km 셀 중심값을 음영 화소(250 m × step)로. 지리 좌표에서 한다(위도 방향 = 셀 행 ky, 경도 방향 = 행마다 다른 간격 CELL/cos φ).
    값이 없는 이웃(NaN, 격자 밖)은 가중치에서 빼고 남은 가중치로 정규화한다. 유효 이웃이 없으면 NaN. 행 묶음(rows)으로 나눠 메모리를 아낀다. 반환 float32 (H, W)."""
    lut, ky0, kx0 = base.lut, base.ky0, base.kx0; H, W = base.lat.shape
    out = np.full((H, W), np.nan, np.float32)
    for r0 in range(0, H, rows):
        lat = base.lat[r0:r0 + rows].astype(np.float64); lon = base.lon[r0:r0 + rows].astype(np.float64)
        fy = lat / CELL - 0.5; i0 = np.floor(fy).astype(np.int32); wy = (fy - i0).astype(np.float32)
        num = np.zeros(lat.shape, np.float32); den = np.zeros(lat.shape, np.float32)
        for i, wi in ((i0, 1.0 - wy), (i0 + 1, wy)):
            fx = lon / (CELL / np.cos(np.radians((i + 0.5) * CELL))) - 0.5; j0 = np.floor(fx).astype(np.int32); wx = (fx - j0).astype(np.float32)
            for j, wj in ((j0, 1.0 - wx), (j0 + 1, wx)):
                ii = i - ky0; jj = j - kx0
                ok = (ii >= 0) & (ii < lut.shape[0]) & (jj >= 0) & (jj < lut.shape[1])
                idx = np.full(lat.shape, -1, np.int32); idx[ok] = lut[ii[ok], jj[ok]]
                v = np.full(lat.shape, np.nan, np.float32); h = idx >= 0; v[h] = val[idx[h]]
                w = wi * wj * np.isfinite(v).astype(np.float32)
                num += w * np.nan_to_num(v); den += w
        out[r0:r0 + rows] = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    return out


def fill_rows(base, val, max_cells):
    """표시용 틈 메우기(개관 패널): 그려지는 셀 사이의 작은 틈(물·자료 없음; 4-연결 성분 크기 ≤ max_cells 셀)을 메울 셀(bool, 셀 표 순서).
    셀 격자(lut) 위에서 성분을 센다. 격자 밖(lut < 0)은 행이 없어 메우지 않는다. 메운 셀의 색은 interp_cells 의 이웃 보간값이다."""
    lut = base.lut; has = lut >= 0
    drawn = np.zeros(lut.shape, bool); drawn[has] = np.isfinite(val[lut[has]])
    hole = ~drawn; lab, _ = ndi_label(hole, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]); sz = np.bincount(lab.ravel())
    small = hole & has & (sz[lab] <= max_cells); del lab
    rows = np.zeros(len(val), bool); rows[lut[small]] = True
    return rows


def render(base, val, cmap, norm, aoa=None, feather=0.0, aoa_sigma=0.0, hatch=(6, 1), smooth=False, fill_max=0):
    """셀 값을 색으로 바꿔 음영과 곱한다. val = 셀 표 순서 값(NaN = 안 그림), aoa = 셀 표 순서 0/1, hatch = 자료 없음 빗금(주기, 선 폭; 화소).
    smooth = 셀 중심값을 화소로 쌍선형 보간해 색을 정한다(개관 패널). 그릴지 여부·자료 없음·적용 범위는 화소가 든 셀 그대로.
    fill_max > 0 (smooth 와 함께) = 그려지는 셀 사이의 ≤ fill_max 셀 틈(물·자료 없음)을 보간값으로 메운다(표시 전용; 확대 창은 0). 반환 uint8 (H, W, 3)."""
    idx = base.idx; has = idx >= 0
    v = np.full(idx.shape, np.nan, np.float32); v[has] = val[idx[has]]
    drawn = np.isfinite(v) & base.valid; drawn_src = drawn; water_cell = base.water_cell
    if smooth:
        vs = interp_cells(base, val)
        if fill_max > 0:
            fill = np.zeros(idx.shape, bool); fill[has] = fill_rows(base, val, fill_max)[idx[has]]; fill &= np.isfinite(vs)
            drawn = drawn | fill; water_cell = water_cell & ~fill; del fill
        v = np.where(drawn & np.isfinite(vs), vs, v); del vs
    water = water_cell | (~base.land & ~drawn)
    nodata = has & base.land & ~drawn & ~water_cell                            # 격자 안(영구동토 영역)인데 값·음영이 없는 셀
    rgb = np.empty(idx.shape + (3,), np.float32); rgb[:] = LAND; rgb[water] = WATER
    hh, ww = idx.shape; P, wd = hatch
    line = ((np.arange(hh, dtype=np.int32)[:, None] + np.arange(ww, dtype=np.int32)[None, :]) % P) < wd
    rgb[nodata & line] = HATCH; del line
    col = cmap(norm(np.where(drawn, v, np.nanmean(v[drawn]) if drawn.any() else 0.0)))[..., :3].astype(np.float32)
    if aoa is not None:
        w = np.zeros(idx.shape, np.float32); w[has] = aoa[idx[has]]
        if aoa_sigma > 0:
            d = drawn_src.astype(np.float32); den = gaussian_filter(d, aoa_sigma); num = gaussian_filter(w * d, aoa_sigma)   # 메운 틈은 이웃의 적용 범위를 따른다
            w = np.where(den > 1e-3, num / np.maximum(den, 1e-6), w)
        f = (FADE * (1.0 - w))[..., None]
        col = col * (1.0 - f) + f * LAND
    if feather > 0:
        d = drawn.astype(np.float32); a = gaussian_filter(d, feather)
        num = np.stack([gaussian_filter(col[..., c] * d, feather) for c in range(3)], axis=-1)
        cs = np.where(a[..., None] > 1e-3, num / np.maximum(a, 1e-6)[..., None], col)
        a = np.clip(a, 0, 1)[..., None]; rgb = a * cs + (1 - a) * rgb
    else:
        rgb[drawn] = col[drawn]
    sh = np.where((drawn | nodata) & base.valid, base.shade, base.shade_fill)   # 음영은 지도 영역 안에만. 밖(비영구동토 육지)은 평평한 회색(DEM 타일 경계가 드러나지 않게)
    sh = np.where(water, 1.0, sh)
    out = rgb * ((1 - W_SHADE) + W_SHADE * sh)[..., None]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def zoom_outside(base, val, aoa, lim):
    """확대 창(lim) 안에 그려지는 셀 가운데 적용 범위 밖 셀 수(범례 필요 여부)."""
    x0, x1, y0, y1 = base.extent; hh, ww = base.idx.shape
    cols = np.arange(ww); rows = np.arange(hh)
    gx = x0 + (cols + 0.5) * (x1 - x0) / ww; gy = y1 + (rows + 0.5) * (y0 - y1) / hh
    m = (gy >= lim[2]) & (gy <= lim[3]); n = (gx >= lim[0]) & (gx <= lim[1])
    idx = base.idx[np.ix_(m, n)]; has = idx >= 0
    v = np.full(idx.shape, np.nan, np.float32); v[has] = val[idx[has]]; a = np.zeros(idx.shape, np.float32); a[has] = aoa[idx[has]]
    drawn = np.isfinite(v) & base.valid[np.ix_(m, n)]
    return int(len(np.unique(idx[drawn & (a < 0.5)])))


# ================================================================ 그리기 도우미
def ax_mm(fig, x, y_top, w, h, Wmm, Hmm):
    return fig.add_axes([x / Wmm, 1 - (y_top + h) / Hmm, w / Wmm, h / Hmm])


def latlon_box(tf, lon0, lon1, lat0, lat1, n=60):
    lons = np.r_[np.linspace(lon0, lon1, n), np.full(n, lon1), np.linspace(lon1, lon0, n), np.full(n, lon0)]
    lats = np.r_[np.full(n, lat0), np.linspace(lat0, lat1, n), np.full(n, lat1), np.linspace(lat1, lat0, n)]
    return tf.transform(lons, lats)


def region_limits(key, tf):
    lo0, lo1, la0, la1 = REG[key]["box"]; p = REG[key]["pad"]
    x, y = latlon_box(tf, lo0, lo1, la0, la1); return [x.min() - p, x.max() + p, y.min() - p, y.max() + p]


def frame(ax, rgb, extent, lim, M):
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="antialiased", zorder=1)
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(M.lw_frame); sp.set_edgecolor(FRAME)


def coast(ax, key, tf, M):
    from shapely.ops import transform as shp_transform
    for g in MV.ne_layers(key)["coastline"]:
        gg = shp_transform(tf.transform, g)
        for part in getattr(gg, "geoms", [gg]):
            xs, ys = part.xy; ax.plot(xs, ys, color=COAST, lw=M.lw_coast, zorder=4, solid_capstyle="round")


def _cross(xs, ys, c, axis):
    """선(xs, ys)이 x = c(axis 0) 또는 y = c(axis 1)를 지나는 점들의 다른 좌표."""
    a, b = (xs, ys) if axis == 0 else (ys, xs); out = []
    s = a - c
    for i in np.where(s[:-1] * s[1:] <= 0)[0]:
        if s[i + 1] == s[i]:
            continue
        t = s[i] / (s[i] - s[i + 1]); out.append(b[i] + t * (b[i + 1] - b[i]))
    return out


def graticule(ax, key, tf, lim, M, lon_lab=(), lat_lab=()):
    """경위선. lon_lab·lat_lab 에 든 선만 아래·왼쪽 라벨을 단다(큰 패널만)."""
    r = REG[key]; x0, x1, y0, y1 = lim; yt, yl, xt, xl = [], [], [], []
    lo = np.linspace(r["box"][0] - 12, r["box"][1] + 12, 400); la = np.linspace(r["box"][2] - 6, r["box"][3] + 6, 400)
    for lat in r["lats"]:
        xs, ys = tf.transform(lo, np.full_like(lo, lat)); ax.plot(xs, ys, color=GRAT, lw=M.lw_grat, alpha=0.55, zorder=3)
        for yc in _cross(xs, ys, x0, 0):
            if y0 < yc < y1 and lat in lat_lab:
                yt.append(yc); yl.append(f"{abs(lat)}°{'N' if lat >= 0 else 'S'}"); break
    for lon in r["lons"]:
        xs, ys = tf.transform(np.full_like(la, lon), la); ax.plot(xs, ys, color=GRAT, lw=M.lw_grat, alpha=0.55, zorder=3)
        for xc in _cross(xs, ys, y0, 1):
            if x0 < xc < x1 and lon in lon_lab:
                xt.append(xc); xl.append(f"{abs(lon)}°{'E' if lon >= 0 else 'W'}"); break
    ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.tick_params(length=M.tick_len, width=M.tick_w, pad=M.tick_pad, labelsize=M.fs_tick, color=FRAME)


def scalebar(ax, lim, km, M, fx=0.05, fy=0.06):
    """왼쪽 아래 축척 막대: 길이 라벨은 막대 위, 양 끝 눈금."""
    x0, x1, y0, y1 = lim; L = km * 1000.0; tick = 0.014 * (y1 - y0)
    xs = x0 + fx * (x1 - x0); ys = y0 + fy * (y1 - y0)
    halo = [pe.withStroke(linewidth=M.halo + 0.4, foreground="white")]
    ax.plot([xs, xs + L], [ys, ys], color="black", lw=M.lw_scale, zorder=8, solid_capstyle="butt", path_effects=halo)
    for x in (xs, xs + L):
        ax.plot([x, x], [ys, ys + tick], color="black", lw=0.8 * M.lw_scale, zorder=8, solid_capstyle="butt", path_effects=halo)
    ax.text(xs + L / 2, ys + 1.4 * tick, f"{km} km", ha="center", va="bottom", fontsize=M.fs_tick, zorder=9, path_effects=M.halo_fx)


def label(ax, s, M, x=0.025, y=0.97):
    ax.text(x, y, s, transform=ax.transAxes, va="top", ha="left", fontsize=M.fs_label, zorder=9, path_effects=M.halo_fx)


def letter(ax, s, M, x=0.0, y=1.01):
    """패널 문자. 논문판은 style_v5.panel_letter(8 pt), 덱은 토큰 slide 크기(14 pt 굵게). 둘 다 PANELS 에 등록한다."""
    if M.paper:
        S.panel_letter(ax, s, x=x, y=y)
    else:
        ax.text(x, y, s, transform=ax.transAxes, fontsize=M.fs_letter, fontweight="bold", va="bottom", ha="left"); S.PANELS.setdefault(s, []).append(ax)


def obs_marks(ax, ox, oy, lim, s, lw, case=0.0, pad_frac=0.0):
    """관측 셀을 빈 원(지름 √s pt, 테두리 lw pt)으로 표시한다. 패널 테두리에서 pad_frac(범위 비율)만큼 안쪽 셀만 그려 원이 잘리지 않게 한다.
    case > 0 이면 원마다 테두리 바깥에 흰 테(case pt)를 먼저 그린다(withStroke: 마커는 그리는 순서대로 테 → 선이라 뒤 원의 흰 테가 앞 원의 선을 가려
    겹친 원이 따로 보인다, v8). 북쪽부터 그려 남쪽 원이 위에 놓인다."""
    px = pad_frac * (lim[1] - lim[0]); py = pad_frac * (lim[3] - lim[2])
    m = (ox >= lim[0] + px) & (ox <= lim[1] - px) & (oy >= lim[2] + py) & (oy <= lim[3] - py)
    o = np.argsort(-oy[m], kind="stable")
    fx = [pe.withStroke(linewidth=lw + 2.0 * case, foreground="white")] if case > 0 else None
    ax.plot(ox[m][o], oy[m][o], ls="none", marker="o", ms=float(np.sqrt(s)), mfc="none", mec=OBS, mew=lw, zorder=6, path_effects=fx)
    return int(m.sum())


def zoom_lim(tf, centre, hx, hy):
    cx, cy = tf.transform(*centre); return [cx - hx, cx + hx, cy - hy, cy + hy]


def zoom_box(ax, lim, M):
    x0, x1, y0, y1 = lim; ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="black", lw=M.lw_box, zorder=7)


def link(fig, ax_src, lim, ax_dst, M):
    """확대 상자(ax_src 자료 좌표)의 오른쪽 두 모서리를 확대 패널(ax_dst)의 왼쪽 두 모서리와 곧은 회색 선으로 잇는다."""
    x0, x1, y0, y1 = lim
    for (xa, ya), (xb, yb) in (((x1, y1), (0, 1)), ((x1, y0), (0, 0))):
        c = ConnectionPatch(xyA=(xa, ya), coordsA="data", axesA=ax_src, xyB=(xb, yb), coordsB="axes fraction", axesB=ax_dst, color=LINK, lw=M.lw_link, zorder=0.5, clip_on=False)
        fig.add_artist(c); LINKS.append(c)


def cbar(fig, rect, cmap, norm, lab, ticks, extend, M):
    cax = fig.add_axes(rect)
    cb = MV.colorbar(fig, cax, cmap, norm, lab, ticks, MV.Med("paper" if M.paper else "slide"), extend, orientation="horizontal")
    if M.paper:
        cb.ax.tick_params(labelsize=M.fs_tick); cb.set_label(lab, fontsize=M.fs_cbar, labelpad=1.5)
    return cax


class HatchHandler(HandlerBase):
    """'자료 없음' 범례 손잡이: 지도와 같은 주기·선 폭(pt)의 45° 빗금을 사각형 안에 직접 그린다."""

    def __init__(self, face, line, period_pt, lw_pt, edge="#BEBEBE", edge_lw=0.3):
        super().__init__(); self.face, self.line, self.period, self.lw, self.edge, self.edge_lw = face, line, period_pt, lw_pt, edge, edge_lw

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x0, y0 = -xdescent, -ydescent
        arts = [Rectangle((x0, y0), width, height, facecolor=self.face, edgecolor=self.edge, lw=self.edge_lw, transform=trans)]
        c = -height + self.period / 2.0
        while c < width:                                   # 선 x − y = c, 사각형 안쪽 구간만
            xa, xb = max(0.0, c), min(width, c + height)
            if xb > xa:
                arts.append(Line2D([x0 + xa, x0 + xb], [y0 + xa - c, y0 + xb - c], color=self.line, lw=self.lw, solid_capstyle="butt", transform=trans))
            c += self.period
        return arts


def _anchor(loc, bbox, transform):
    kw = dict(loc=loc, bbox_to_anchor=bbox, frameon=False, borderaxespad=0.0)
    if transform is not None:
        kw["bbox_transform"] = transform; kw["borderpad"] = 0.0
    return kw


def legend_categories(ax, M, base, ncol=2, loc="upper left", bbox=(0.0, -0.06), transform=None):
    """범주 범례: 비영구동토 육지 / 자료 없음 / 관측 셀 / 물. 지도 틀 밖(기본: 패널 아래 왼쪽, 2 열)에 둔다. transform 을 주면 bbox 는 그 좌표계(그림 mm 배치용).
    육지·빗금 색은 지도와 같게 평균 음영(shade_fill)을 곱한다. 물은 음영을 곱하지 않는다. 반환 Legend(폭을 재서 컬러바를 비켜 놓는 데 쓴다)."""
    f = (1.0 - W_SHADE) + W_SHADE * base.shade_fill
    land = tuple((LAND * f).tolist()); hatch = tuple((HATCH * f).tolist()); edge = "#BEBEBE"
    h_hatch = Patch(facecolor=land, edgecolor=hatch)
    h = [Patch(facecolor=land, edgecolor=edge, lw=M.leg_edge), h_hatch,
         Line2D([], [], ls="none", marker="o", ms=M.leg_ms, mfc="none", mec=OBS, mew=M.leg_mew),
         Patch(facecolor=tuple(WATER.tolist()), edgecolor=edge, lw=M.leg_edge)]
    hm = {h_hatch: HatchHandler(land, hatch, M.hatch_mm[0] * PT_MM, M.hatch_mm[1] * PT_MM, edge, M.leg_edge)}
    return ax.legend(h, ["non-permafrost land", "no data", "observed cell", "water"], fontsize=M.fs_tick, handler_map=hm, ncol=ncol,
                     **_anchor(loc, bbox, transform), **M.leg)        # 열 우선 채움(ncol=2): 1 열 = 육지·자료 없음, 2 열 = 관측 셀·물


def legend_aoa(ax, M, ncol=2, loc="upper left", bbox=(0.0, -0.06), transform=None):
    """적용 범위 범례(안·밖 평면색), 지도 틀 밖 한 줄."""
    return ax.legend(handles=[Patch(facecolor=AOA_IN, edgecolor="none"), Patch(facecolor=AOA_OUT, edgecolor="none")], labels=["inside", "outside"],
                     fontsize=M.fs_tick, ncol=ncol, **_anchor(loc, bbox, transform), **M.leg)


def legend_right_mm(fig, leg, Wmm):
    """범례의 오른쪽 끝(그림 mm). 큰 그림을 다시 그리지 않고 렌더러로 범례 상자만 잰다."""
    r = fig.canvas.get_renderer(); bb = leg.get_window_extent(r)
    return bb.x1 / fig.dpi * 25.4


def legend_aoa_zoom(ax, M, cmap, loc="upper right", bbox=(1.0, -0.03)):
    """확대 패널의 적용 범위 범례(패널 아래 띠, 두 줄): 셀 색(중간 ALT) = 안, 육지색과 40 % 혼합 = 밖(render 와 같은 규칙)."""
    col = np.array(cmap(0.55)[:3], np.float32); out = col * (1.0 - FADE) + FADE * LAND
    h = [Patch(facecolor=tuple(col.tolist()), edgecolor="none"), Patch(facecolor=tuple(out.tolist()), edgecolor="none")]
    return ax.legend(h, ["inside area of applicability", "outside area of applicability"], loc=loc, bbox_to_anchor=bbox, frameon=False, fontsize=M.fs_tick,
                     borderaxespad=0.0, **M.leg)


# ================================================================ 자료 적재(논문판·덱 공용)
def load_data():
    """알래스카·레나 지도 자료와 색 범위, 관측 셀 좌표, 전역 음영 격자. 반환 dict A(알래스카), L(레나)."""
    lab = pd.read_csv(PROC / "v5" / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    key = "alaska"; cells, lut, ky0, kx0 = cell_table(key)
    mean = pd.read_csv(MAPS / "Alaska" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    vmean = cell_values(key, cells, mean.r1); aoa = np.nan_to_num(cell_values(key, cells, mean.aoa_in.astype(float)), nan=0.0)
    preds = {yr: cell_values(key, cells, pd.read_csv(MAPS / "Alaska" / f"pred_{yr}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1) for yr in (2014, 2022)}
    base = Base(key, cells, lut, ky0, kx0, step=REG[key]["step"], tag="full"); tf = base.tf; lim = region_limits(key, tf)
    lo, hi = MV.pct_range(mean.r1.values); normA = Normalize(lo, hi); ticksA = np.arange(lo, hi + 1e-6, 10.0); extA = MV.extend_of(mean.r1.values, lo, hi)
    diffs = {yr: preds[yr] - vmean for yr in preds}
    v = max(MV.sym_max(diffs[yr][np.isfinite(diffs[yr])]) for yr in diffs); normD = TwoSlopeNorm(0.0, -v, v); ticksD = np.arange(-v, v + 1e-6, 5.0)
    olon, olat, nA = obs_cells(lab, "Alaska"); ox, oy = tf.transform(olon, olat)
    A = dict(key=key, cells=cells, lut=lut, ky0=ky0, kx0=kx0, vmean=vmean, aoa=aoa, diffs=diffs, base=base, tf=tf, lim=lim, asp=(lim[3] - lim[2]) / (lim[1] - lim[0]),
             lo=lo, hi=hi, norm=normA, ticks=ticksA, ext=extA, dmax=v, normD=normD, ticksD=ticksD, ox=ox, oy=oy, nloc=nA)
    keyL = "lena"; cellsL, lutL, ky0L, kx0L = cell_table(keyL)
    lm = pd.read_csv(MAPS / "Lena" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    l16 = pd.read_csv(MAPS / "Lena" / "pred_2016.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1
    vL = cell_values(keyL, cellsL, l16); aoaL = np.nan_to_num(cell_values(keyL, cellsL, lm.aoa_in.astype(float)), nan=0.0)
    baseL = Base(keyL, cellsL, lutL, ky0L, kx0L, step=REG[keyL]["step"], tag="full"); tfL = baseL.tf; limL = region_limits(keyL, tfL)
    loL, hiL = MV.pct_range(l16.values, step=10.0); normL = Normalize(loL, hiL); ticksL = np.arange(loL, hiL + 1e-6, 10.0); extL = MV.extend_of(l16.values, loL, hiL)
    olonL, olatL, nL = obs_cells(lab, "Lena"); oxL, oyL = tfL.transform(olonL, olatL)
    L = dict(key=keyL, cells=cellsL, lut=lutL, ky0=ky0L, kx0=kx0L, v=vL, aoa=aoaL, base=baseL, tf=tfL, lim=limL, asp=(limL[3] - limL[2]) / (limL[1] - limL[0]),
             lo=loL, hi=hiL, norm=normL, ticks=ticksL, ext=extL, ox=oxL, oy=oyL, nloc=nL)
    return dict(A=A, L=L)


# ================================================================ 패널 그리기(축 하나씩; 논문판·덱 공용)
def draw_mean(ax, D, M, w_mm, zooms, legend=True):
    """알래스카 2011–2024 평균 ALT: 적용 범위 밖은 흐리게, 관측 셀, 확대 창 상자, 경위선 라벨, 축척, 범주 범례."""
    A = D["A"]; base = A["base"]
    fe = 2.0 * 2.0 / base.step; sg = 3.0 * 2.0 / base.step            # 완화 폭은 화소 수: 500 m 기준값을 격자 간격에 맞춘다
    rgb = render(base, A["vmean"], MV.CM_ALT, A["norm"], aoa=A["aoa"], feather=fe, aoa_sigma=sg, hatch=base.hatch_px(w_mm, A["lim"], M.hatch_mm), smooth=True, fill_max=FILL_MAX)
    frame(ax, rgb, base.extent, A["lim"], M); coast(ax, A["key"], A["tf"], M)
    graticule(ax, A["key"], A["tf"], A["lim"], M, lon_lab=(-165, -160, -155, -150, -145), lat_lab=REG[A["key"]]["lat_lab"])
    obs_marks(ax, A["ox"], A["oy"], A["lim"], s=M.obs_over[0], lw=M.obs_over[1], case=M.obs_case[0])
    for _, zl in zooms:
        zoom_box(ax, zl, M)
    scalebar(ax, A["lim"], 200, M); label(ax, "mean ALT 2011–2024", M)
    if legend:
        legend_categories(ax, M, base)


def draw_zoom(ax, D, M, w_mm, j, zl):
    """알래스카 1 km 확대(전해상도, 셀 단위) j 번째 창. 반환: 창 안 적용 범위 밖 셀 수."""
    A = D["A"]; name = ZOOM[A["key"]][j][0]
    bz = Base(A["key"], A["cells"], A["lut"], A["ky0"], A["kx0"], step=1, window=[zl[0] - 2e3, zl[1] + 2e3, zl[2] - 2e3, zl[3] + 2e3], tag=f"zoom{j}")
    frame(ax, render(bz, A["vmean"], MV.CM_ALT, A["norm"], aoa=A["aoa"], hatch=bz.hatch_px(w_mm, zl, M.hatch_mm)), bz.extent, zl, M)
    obs_marks(ax, A["ox"], A["oy"], zl, s=M.obs_zoom_s(w_mm / ((zl[1] - zl[0]) / 1e3)), lw=M.obs_zoom_lw, case=M.obs_case[1], pad_frac=0.012)
    scalebar(ax, zl, 20, M); label(ax, name, M)
    return zoom_outside(bz, A["vmean"], A["aoa"], zl)


def draw_diff(ax, D, M, w_mm, yr):
    """연도 편차(그 해 예측 − 평균), 발산 색표, 같은 범위."""
    A = D["A"]; base = A["base"]
    fe = 2.0 * 2.0 / base.step; sg = 3.0 * 2.0 / base.step
    rgb = render(base, A["diffs"][yr], MV.CM_DIFF, A["normD"], aoa=A["aoa"], feather=fe, aoa_sigma=sg, hatch=base.hatch_px(w_mm, A["lim"], M.hatch_mm), smooth=True, fill_max=FILL_MAX)
    frame(ax, rgb, base.extent, A["lim"], M); coast(ax, A["key"], A["tf"], M); graticule(ax, A["key"], A["tf"], A["lim"], M)
    scalebar(ax, A["lim"], 200, M); label(ax, f"ALT {yr} − mean", M)


def draw_aoa(ax, D, M, w_mm, lon_lab=(-160, -150), legend=True):
    """적용 범위 안·밖(평면색) 지도(1 km 셀 그대로, 범주이므로 보간 없음). legend=False 면 범례는 호출 쪽이 틀 밖에 놓는다."""
    A = D["A"]; base = A["base"]
    rgb = render(base, np.where(np.isfinite(A["vmean"]), A["aoa"], np.nan), ListedColormap([AOA_OUT, AOA_IN]), Normalize(-0.5, 1.5), feather=1.5 * 2.0 / base.step,
                 hatch=base.hatch_px(w_mm, A["lim"], M.hatch_mm))
    frame(ax, rgb, base.extent, A["lim"], M); coast(ax, A["key"], A["tf"], M); graticule(ax, A["key"], A["tf"], A["lim"], M, lon_lab=lon_lab, lat_lab=REG[A["key"]]["lat_lab"])
    scalebar(ax, A["lim"], 200, M); label(ax, "area of applicability", M)
    if legend:
        legend_aoa(ax, M)


def draw_lena(ax, D, M, w_mm, zl):
    """레나 델타 2016 ALT, 관측 셀, 확대 창 상자."""
    L = D["L"]; base = L["base"]
    rgb = render(base, L["v"], MV.CM_ALT, L["norm"], aoa=L["aoa"], feather=1.5, aoa_sigma=4.0, hatch=base.hatch_px(w_mm, L["lim"], M.hatch_mm), smooth=True, fill_max=FILL_MAX)
    frame(ax, rgb, base.extent, L["lim"], M); coast(ax, L["key"], L["tf"], M); graticule(ax, L["key"], L["tf"], L["lim"], M, lon_lab=(124, 126, 128, 130), lat_lab=REG[L["key"]]["lat_lab"])
    obs_marks(ax, L["ox"], L["oy"], L["lim"], s=M.obs_over[0], lw=M.obs_over[1], case=M.obs_case[0]); zoom_box(ax, zl, M)
    scalebar(ax, L["lim"], 50, M); label(ax, "Lena Delta, ALT 2016", M)


def draw_lena_zoom(ax, D, M, w_mm, zl):
    """사모일로프–쿠룽나흐 1 km 확대. 반환: 창 안 적용 범위 밖 셀 수."""
    L = D["L"]; name = ZOOM[L["key"]][0][0]
    bz = Base(L["key"], L["cells"], L["lut"], L["ky0"], L["kx0"], step=1, window=[zl[0] - 1e3, zl[1] + 1e3, zl[2] - 1e3, zl[3] + 1e3], tag="zoom0")
    frame(ax, render(bz, L["v"], MV.CM_ALT, L["norm"], aoa=L["aoa"], hatch=bz.hatch_px(w_mm, zl, M.hatch_mm)), bz.extent, zl, M)
    obs_marks(ax, L["ox"], L["oy"], zl, s=M.obs_zoom_s(w_mm / ((zl[1] - zl[0]) / 1e3)), lw=M.obs_zoom_lw, case=M.obs_case[1], pad_frac=0.012)
    scalebar(ax, zl, 5, M); label(ax, name, M)
    return zoom_outside(bz, L["v"], L["aoa"], zl)


# ================================================================ 논문판 그림
def f6(slide=False):
    t0 = time.time(); LINKS.clear(); M = Med("slide" if slide else "paper"); D = load_data(); A, L = D["A"], D["L"]
    aspA, aspL = A["asp"], L["asp"]; tfA, tfL = A["tf"], L["tf"]; limA, limL = A["lim"], L["lim"]
    # ---------------- 배치(mm). 행 1: a(2 띠) | d1, d2 | b, c 가 같은 세로 띠. 행 2: e | f | g 같은 높이
    Wmm = S.W2; xa, ya = 7.5, 4.5; g1, g2, rm = 5.0, 5.0, 2.2
    gv = 15.0 if slide else 4.0                                                   # 2×2 칸의 세로 간격(슬라이드판은 칸마다 컬러바가 들어간다)
    cw = (Wmm - xa - g1 - g2 - rm - gv / aspA) / 4.0; ch = cw * aspA; ha = 2 * ch + gv; wa = ha / aspA
    xd = xa + wa + g1; xb = xd + cw + g2; cb_h = M.cb_h
    y_cb1 = ya + ha + 5.5; y_leg = ya + ha + 5.0; y2 = y_cb1 + cb_h + 5.5 + 5.5          # 띠 1 = 경도 라벨 아래: 범례(a 왼쪽)와 컬러바(d·c 열)
    H2 = 51.0; we, wf, wg = H2 / aspA, H2 / aspL, H2; xe = xa; xf = xe + we + 7.5; xg = xf + wf + 5.0
    y_cb2 = y2 + H2 + 5.5; Hmm = y_cb2 + cb_h + 5.5 + 1.0
    fig = S.figure(Wmm, Hmm)
    AX = lambda x, y, w, h: ax_mm(fig, x, y, w, h, Wmm, Hmm)
    RECT = lambda x, y, w, h: [x / Wmm, 1 - (y + h) / Hmm, w / Wmm, h / Hmm]
    zoomsA = [(name, zoom_lim(tfA, c, hx, hx * aspA)) for name, c, hx in ZOOM[A["key"]]]          # 확대 창 비율 = 칸 비율
    # a 알래스카 평균(범주 범례는 틀 밖, 경도 라벨 아래 왼쪽 2 열)
    ax = AX(xa, ya, wa, ha); draw_mean(ax, D, M, wa, zoomsA, legend=False); letter(ax, "a", M, x=0.0, y=1.01); axA = ax
    legA = legend_categories(axA, M, A["base"], ncol=2, bbox=(xa / Wmm, 1 - y_leg / Hmm), transform=fig.transFigure)
    # d 확대 2곳(문자 d 하나, 지명 라벨 각각)
    for j, (name, zl) in enumerate(zoomsA):
        yj = ya + j * (ch + gv); ax = AX(xd, yj, cw, ch); draw_zoom(ax, D, M, cw, j, zl)
        if j == 0:
            letter(ax, "d", M, x=0.0, y=1.02)
        else:
            S.register("d", ax)
        link(fig, axA, zl, ax, M)
        if slide:
            S.register("d", cbar(fig, RECT(xd + 3, yj + ch + (2.5 if j == 0 else 5.5), cw - 6, cb_h), MV.CM_ALT, A["norm"], "ALT (cm)", A["ticks"], A["ext"], M))
    # b, c 연도 편차(같은 발산 범위)
    for j, (yr, Lt) in enumerate(((2014, "b"), (2022, "c"))):
        yj = ya + j * (ch + gv); ax = AX(xb, yj, cw, ch); draw_diff(ax, D, M, cw, yr); letter(ax, Lt, M, x=0.0, y=1.02)
        if slide:
            S.register(Lt, cbar(fig, RECT(xb + 3, yj + ch + (2.5 if j == 0 else 5.5), cw - 6, cb_h), MV.CM_DIFF, A["normD"], "ALT difference (cm)", A["ticksD"], "both", M))
    if not slide:   # 논문판: 공유 컬러바를 열 아래에 둔다(ALT = d 열, 차이 = c 열; 같은 폭). 범례는 a 아래 왼쪽
        S.register("d", cbar(fig, RECT(xd + 2.5, y_cb1, cw - 5.0, cb_h), MV.CM_ALT, A["norm"], "ALT (cm)", A["ticks"], A["ext"], M))
        S.register("c", cbar(fig, RECT(xb + 2.5, y_cb1, cw - 5.0, cb_h), MV.CM_DIFF, A["normD"], "ALT difference (cm)", A["ticksD"], "both", M))
    else:           # 슬라이드판: a 의 컬러바는 범례 오른쪽(범례 폭을 재서 겹치지 않게), 폭은 d·c 열 컬러바와 같게
        x_cb = max(legend_right_mm(fig, legA, Wmm) + 5.0, xa + wa - (cw - 5.0) - 2.5); w_cb = min(cw - 5.0, xa + wa + 4.0 - x_cb)
        S.register("a", cbar(fig, RECT(x_cb, y_cb1, w_cb, cb_h), MV.CM_ALT, A["norm"], "ALT (cm)", A["ticks"], A["ext"], M))
    # ---------------- 행 2: e 적용 범위 | f 레나 2016 | g 사모일로프 확대
    ax = AX(xe, y2, we, H2); draw_aoa(ax, D, M, we, legend=False); letter(ax, "e", M, x=0.0, y=1.01)
    legend_aoa(ax, M, ncol=2, bbox=(xe / Wmm, 1 - (y_cb2 - 0.5) / Hmm), transform=fig.transFigure)        # 틀 밖: 경도 라벨 아래, 컬러바 띠와 같은 높이
    name, c, h = ZOOM[L["key"]][0]; zlL = zoom_lim(tfL, c, h, h)
    ax = AX(xf, y2, wf, H2); draw_lena(ax, D, M, wf, zlL); letter(ax, "f", M, x=0.0, y=1.01); axF = ax
    ax = AX(xg, y2, wg, H2); draw_lena_zoom(ax, D, M, wg, zlL); letter(ax, "g", M, x=0.0, y=1.02); link(fig, axF, zlL, ax, M)
    if not slide:   # 공유 컬러바: f·g 의 가로 중앙
        cx = (xf + xg + wg) / 2.0; S.register("f", cbar(fig, RECT(cx - 22.0, y_cb2, 44.0, cb_h), MV.CM_ALT, L["norm"], "ALT (cm)", L["ticks"], L["ext"], M))
    else:
        S.register("f", cbar(fig, RECT(xf + wf / 2 - 20.0, y_cb2, 40.0, cb_h), MV.CM_ALT, L["norm"], "ALT (cm)", L["ticks"], L["ext"], M))
        S.register("g", cbar(fig, RECT(xg + wg / 2 - 20.0, y_cb2, 40.0, cb_h), MV.CM_ALT, L["norm"], "ALT (cm)", L["ticks"], L["ext"], M))
    # ---------------- 내보내기. 묶음 크롭은 실제로 직사각형이 되는 조합만(abcd = 행 1, efg = 행 2, fg)
    groups = {"abcd": [a for k in "abcd" for a in S.PANELS.get(k, [])], "efg": [a for k in "efg" for a in S.PANELS.get(k, [])], "fg": [a for k in "fg" for a in S.PANELS.get(k, [])]}
    if slide:
        singles = {k: list(v) for k, v in S.PANELS.items()}
        for c_ in LINKS:
            c_.set_visible(False)                      # 1차: 단독 패널 크롭(연결선 없음)
        out = S.export(fig, "F6_maps", outdir=OUT)
        for c_ in LINKS:
            c_.set_visible(True)                       # 2차: 합성본과 묶음 크롭(연결선 포함)
        S.PANELS.clear(); S.PANELS.update(groups)
        out += S.export(fig, "F6_maps", outdir=OUT)
        S.PANELS.clear(); S.PANELS.update(singles)
    else:
        for k, v_ in groups.items():
            S.register(k, *v_)
        out = S.export(fig, "F6_maps", outdir=OUT)
    print(f"[F6] {len(out)} files, {time.time() - t0:.0f} s; size {Wmm:.0f}×{Hmm:.1f} mm; ALT range {A['lo']}–{A['hi']}, diff ±{A['dmax']}, Lena {L['lo']}–{L['hi']}; "
          f"observed cells Alaska {len(A['ox'])} (from {A['nloc']} loc), Lena {len(L['ox'])} (from {L['nloc']} loc)")
    km = lambda z: [round((z[1] - z[0]) / 1e3, 1), round((z[3] - z[2]) / 1e3, 1)]
    sc = SLIDE_SCALE if slide else 1.0; dz = lambda w_mm, z: round(float(np.sqrt(M.obs_zoom_s(w_mm / ((z[1] - z[0]) / 1e3)))) * sc, 2)
    write_meta(D, "slide" if slide else "paper", dict(size_mm=[Wmm, round(Hmm, 1)], zoom_windows_km={"Alaska": [km(z) for _, z in zoomsA], "Lena": [km(zlL)]},
               png_declared=dict(dpi=300, width_in=round(Wmm / 25.4 * sc, 2)) if slide else dict(dpi=600, width_mm=Wmm),
               zoom_circle_pt=dict(diameter={"Alaska_d": dz(cw, zoomsA[0][1]), "Lena_g": dz(wg, zlL)}, edge=round(M.obs_zoom_lw * sc, 2), casing=round(M.obs_case[1] * sc, 2),
                                   note="at the PNG's declared size (slide: 300 dpi metadata, width_in above; the deck builder places the file at that size)")))
    return out


def cvd_check(cmap, vmax):
    """발산 색표의 색각 이상 모의(Machado et al. 2009, 심도 1.0 행렬; 선형 RGB): ±vmax, ±vmax/2, ±vmax/5 쌍의 CIE76 ΔE 를 정상·적색맹·녹색맹·청색맹에 대해 돌려준다."""
    MAT = {"protanopia": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
           "deuteranopia": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
           "tritanopia": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]}
    lin = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    srgb = lambda c: np.where(np.clip(c, 0, 1) <= 0.0031308, 12.92 * np.clip(c, 0, 1), 1.055 * np.clip(c, 0, 1) ** (1 / 2.4) - 0.055)
    def lab(rgb):
        xyz = lin(np.asarray(rgb)) @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T / np.array([0.95047, 1.0, 1.08883])
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0); return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])
    dE = lambda a, b: round(float(np.linalg.norm(lab(a) - lab(b))), 1)
    pairs = [vmax, vmax / 2.0, vmax / 5.0]; col = lambda x: np.array(cmap(0.5 + 0.5 * x / vmax)[:3])
    out = {"pairs_cm": [[-x, x] for x in pairs], "dE_CIE76": {"normal": [dE(col(-x), col(x)) for x in pairs]}}
    ends = {"normal": [round(float(lab(col(-vmax))[0])), round(float(lab(col(vmax))[0]))]}
    for name, Mx in MAT.items():
        sim = lambda c, Mx=np.array(Mx): srgb(Mx @ lin(c))
        out["dE_CIE76"][name] = [dE(sim(col(-x)), sim(col(x))) for x in pairs]
        ends[name] = [round(float(lab(sim(col(-vmax)))[0])), round(float(lab(sim(col(vmax)))[0]))]
    out["L_star_ends"] = ends
    return out


def write_meta(D, variant, extra):
    """source_data/F6_maps.json: 표시 보간(덱 캡션 '표시용 250 m 보간' 의 근거)·마스크·틈 메우기·원 테·색각 점검·색 범위·관측 셀 수. 판(variant)마다 항목을 덧붙인다."""
    A, L = D["A"], D["L"]
    nfill = {"Alaska_mean": int(fill_rows(A["base"], A["vmean"], FILL_MAX).sum()), "Lena_2016": int(fill_rows(L["base"], L["v"], FILL_MAX).sum())}
    display = dict(overview_panels="1 km cell values (cell centres) interpolated bilinearly in geographic coordinates, NaN-aware, onto the hillshade display grid before draping; "
                                   "drawn/no-data/area-of-applicability masks follow the containing 1 km cell",
                   display_grid_m={"Alaska": PX * A["base"].step, "Lena": PX * L["base"].step}, hillshade_m=PX,
                   exact_cell_panels="zoom windows (paper d, g; deck R07 b, c; R08 a, b; R09 b) and the area-of-applicability map (paper e; deck R08 c)",
                   masks="only the continuous ALT and ALT-difference values are interpolated; the drawn / non-permafrost land / no-data masks stay at native 1 km cell resolution. "
                         "In the sporadic-permafrost belt of west-central Alaska (about 60-63 N) this leaves 1 km block edges next to smooth gradients: they are data boundaries "
                         "(CCI permafrost fraction < 10 %, missing ERA5-Land soil forcing or CCI value), not a rendering defect",
                   gap_fill=dict(rule=f"overview panels only: gaps (water or no data) between drawn cells whose 4-connected component is <= {FILL_MAX} cell are filled with the "
                                      "interpolated value for display; the analysis masks and the zoom windows are unchanged", cells_filled=nfill,
                                 remaining=f"fine white specks left in the Lena Delta overview are water / no-data gaps of > {FILL_MAX} cells (channels and lakes of > {FILL_MAX} km2) at native 1 km resolution"),
                   observed_cell_circles="open circles with a white casing drawn per circle (later circles overprint earlier ones), so overlapping circles stay separable; "
                                         "circles still overlap in dense clusters of the overview panels and where station spacing approaches the 1 km cell pitch (paper g, deck R09 b)",
                   shade_weight=W_SHADE, outside_aoa_fade=FADE, aoa_outside_hex=AOA_OUT, non_permafrost_land_hex="#E6E6E6",
                   caption_note_ko="표시: 개관 지도는 1 km 셀 값을 250 m 음영 격자에 쌍선형 보간(마스크는 1 km 그대로, 1셀 틈은 메움), 확대 창은 1 km 셀 그대로. "
                                   "원 = 관측 셀(촘촘한 곳은 겹침). 서남부 산발 영구동토대의 각진 경계는 자료 경계(영구동토 확률 < 10 %, 강제력 결측)이다.",
                   caption_note_en="Overview maps drape 1 km cell values interpolated to the 250 m hillshade grid (masks at native 1 km; single-cell gaps filled for display); "
                                   "zoom windows show exact 1 km cells. Circles mark observed cells and may overlap in dense clusters; block edges in the sporadic-permafrost "
                                   "belt are data boundaries.")
    cvd = cvd_check(MV.CM_DIFF, float(A["dmax"])); dmin = {k: min(v) for k, v in cvd["dE_CIE76"].items()}
    cvd["verdict"] = ("sign (loss vs gain) distinguishable in all simulations at |difference| <= vmax/2 (min dE %.0f); " % min(min(v[1:]) for v in cvd["dE_CIE76"].values())
                      + ", ".join(f"{k}: end-pair dE {v[0]}" for k, v in cvd["dE_CIE76"].items())
                      + ". Palette (crameri broc, MV.CM_DIFF) kept for consistency with F7; the two saturated ends are close only under tritanopia")
    meta = json.loads((OUT / "source_data" / "F6_maps.json").read_text()) if (OUT / "source_data" / "F6_maps.json").exists() else {}
    meta.update(dict(display=display, cvd_check_diff_colormap=dict(colormap="crameri broc (MV.CM_DIFF)", simulation="Machado 2009 severity 1.0", **cvd), alt_range_cm={"Alaska_mean_2011_2024": [float(A["lo"]), float(A["hi"])], "Lena_2016": [float(L["lo"]), float(L["hi"])]},
                     diff_sym_max_cm=float(A["dmax"]), observed_cells={"Alaska": int(len(A["ox"])), "Lena": int(len(L["ox"]))}, observed_locations={"Alaska": int(A["nloc"]), "Lena": int(L["nloc"])},
                     sources=["data/processed/v5/maps/<Alaska|Lena>/{mean_2011_2024,pred_<year>}.csv.gz", "data/processed/map_<key>/<key>_grid_x25_v1.csv.gz",
                              "data/processed/map_<key>/hillshade_v1/{shade,dem}_250m.tif", "data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1)"]))
    meta[variant] = extra
    (OUT / "source_data").mkdir(parents=True, exist_ok=True)
    (OUT / "source_data" / "F6_maps.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))


# ================================================================ 덱 원판(슬라이드 본문 크기)
def deck_canvas(h_in=DECK_H_IN):
    fig = S.figure(DECK_W_IN * 25.4, h_in * 25.4); fig.set_size_inches(DECK_W_IN, h_in)     # mm → in 반올림 오차(11.9999 in → 3599 px)를 없앤다
    return fig, DECK_W_IN * 25.4, h_in * 25.4


def deck_save(fig, sid, h_in=DECK_H_IN):
    """300 dpi, 여백 자르기 없음 → 3600 × (h_in·300) px. 저장 뒤 PIL 로 크기와 캔버스 사용률을 확인한다."""
    from PIL import Image
    DECK_OUT.mkdir(parents=True, exist_ok=True); p = DECK_OUT / f"F6_maps__deck_{sid}.png"
    fig.savefig(p, dpi=300, facecolor="white")
    r = fig.canvas.get_renderer(); bb = fig.get_tightbbox(r); W, H = fig.get_size_inches()
    use_w, use_h = bb.width / W, bb.height / H
    plt.close(fig)
    with Image.open(p) as im:
        rgb = im.convert("RGB")                        # Agg 는 RGBA 로 저장한다 → RGB(흰 배경)로 바꿔 덱에 넣는다
    rgb.save(p, dpi=(300, 300)); sz = rgb.size
    want = (int(round(DECK_W_IN * 300)), int(round(h_in * 300)))
    assert sz == want, f"{p.name}: {sz} != {want}"
    print(f"[F6 deck] {p.name}: {sz[0]}×{sz[1]} px, canvas use {use_w * 100:.0f} % × {use_h * 100:.0f} %")
    return p


def deck_R07(D):
    """R07(5.10 in): a 평균 | b·c 확대(세로) | d·e 편차(세로). 세 열의 아래 끝을 맞춘다. a 아래 = 경도 라벨 + 범주 범례(2 열, 틀 밖),
    b·c 아래 = ALT 컬러바(a 와 공유), d·e 아래 = 차이 컬러바."""
    M = Med("deck"); LINKS.clear(); A = D["A"]; asp = A["asp"]
    fig, Wmm, Hmm = deck_canvas()
    top, xa, lon_h, cb_band, bot, rm = 6.0, 12.5, 6.0, 17.6, 1.0, 2.0
    col_bot = Hmm - bot - cb_band; ha = col_bot - top; wa = ha / asp
    gv = 6.0; ch = (col_bot - top - gv) / 2.0; cw = ch / asp
    spare = Wmm - rm - xa - wa - 2 * cw                      # 높이에 묶인 세 열을 뺀 남는 폭: 열 간격(≤ 24 mm)으로 쓰고 나머지는 양쪽 여백으로 나눠 가운데 맞춤(v8, 오른쪽 빈 띠 제거)
    gap = min(24.0, max(6.0, spare / 2.0)); xa += (spare - 2.0 * gap) / 2.0
    xd = xa + wa + gap; xb = xd + cw + gap
    AX = lambda x, y, w, h: ax_mm(fig, x, y, w, h, Wmm, Hmm)
    RECT = lambda x, y, w, h: [x / Wmm, 1 - (y + h) / Hmm, w / Wmm, h / Hmm]
    zooms = [(name, zoom_lim(A["tf"], c, hx, hx * asp)) for name, c, hx in ZOOM[A["key"]]]
    ax = AX(xa, top, wa, ha); draw_mean(ax, D, M, wa, zooms, legend=False); letter(ax, "a", M); axA = ax
    legend_categories(axA, M, A["base"], ncol=2, bbox=(xa / Wmm, 1 - (col_bot + lon_h + 0.8) / Hmm), transform=fig.transFigure)
    for j, (name, zl) in enumerate(zooms):
        ax = AX(xd, top + j * (ch + gv), cw, ch); draw_zoom(ax, D, M, cw, j, zl); letter(ax, "bc"[j], M); link(fig, axA, zl, ax, M)
    for j, yr in enumerate((2014, 2022)):
        ax = AX(xb, top + j * (ch + gv), cw, ch); draw_diff(ax, D, M, cw, yr); letter(ax, "de"[j], M)
    y_cb = col_bot + 2.5
    cbar(fig, RECT(xd + 2.0, y_cb, cw - 4.0, M.cb_h), MV.CM_ALT, A["norm"], "ALT (cm)", A["ticks"], A["ext"], M)
    cbar(fig, RECT(xb + 2.0, y_cb, cw - 4.0, M.cb_h), MV.CM_DIFF, A["normD"], "ALT difference (cm)", A["ticksD"], "both", M)
    return deck_save(fig, "R07"), dict(zoom_circle_diameter_pt=round(float(np.sqrt(M.obs_zoom_s(cw / ((zooms[0][1][1] - zooms[0][1][0]) / 1e3)))), 2), column_gap_mm=round(gap, 1))


def deck_R08(D):
    """R08(5.10 in): a·b 확대 2곳 가로(원 ≥ 6 pt, 테두리 1 pt) + c 적용 범위. 확대 아래 공유 ALT 컬러바와 적용 범위 안/밖 범례."""
    M = Med("deck"); LINKS.clear(); A = D["A"]; asp = A["asp"]
    fig, Wmm, Hmm = deck_canvas()
    top, bot, lon_h = 6.0, 1.0, 6.0
    wz, gz, x1 = 86.0, 5.0, 2.0; hz = wz * asp
    xe = x1 + 2 * wz + gz + 12.5; we = Wmm - xe - 2.0; he = we * asp
    if top + he + lon_h + bot > Hmm:
        he = Hmm - top - lon_h - bot; we = he / asp
    AX = lambda x, y, w, h: ax_mm(fig, x, y, w, h, Wmm, Hmm)
    RECT = lambda x, y, w, h: [x / Wmm, 1 - (y + h) / Hmm, w / Wmm, h / Hmm]
    zooms = [(name, zoom_lim(A["tf"], c, hx, hx * asp)) for name, c, hx in ZOOM[A["key"]]]
    n_out = 0
    for j, (name, zl) in enumerate(zooms):
        ax = AX(x1 + j * (wz + gz), top, wz, hz); n_out += draw_zoom(ax, D, M, wz, j, zl); letter(ax, "ab"[j], M); axz = ax
    ax = AX(xe, top, we, he); draw_aoa(ax, D, M, we, legend=False); letter(ax, "c", M)
    legend_aoa(ax, M, ncol=2, bbox=(xe / Wmm, 1 - (top + he + lon_h + 0.8) / Hmm), transform=fig.transFigure)          # 틀 밖: c 의 경도 라벨 아래
    cbar(fig, RECT(x1 + wz / 2.0 - 30.0, top + hz + 2.5, 60.0, M.cb_h), MV.CM_ALT, A["norm"], "ALT (cm)", A["ticks"], A["ext"], M)   # a 아래 중앙; 범례는 b 아래 오른쪽
    if n_out > 0:
        legend_aoa_zoom(axz, M, MV.CM_ALT)
    print(f"[F6 deck] R08 zoom cells outside area of applicability: {n_out}")
    return deck_save(fig, "R08"), dict(zoom_circle_diameter_pt=round(float(np.sqrt(M.obs_zoom_s(wz / ((zooms[0][1][1] - zooms[0][1][0]) / 1e3)))), 2), zoom_cells_outside_aoa=n_out)


def deck_R09(D):
    """R09(5.10 in): a 레나 델타 2016 + b 사모일로프–쿠룽나흐 확대(창 폭 24 km, 높이는 패널 비율). 확대 아래 ALT 컬러바, 창에 적용 범위 밖 셀이 있으면 범례."""
    M = Med("deck"); LINKS.clear(); L = D["L"]; aspL = L["asp"]
    fig, Wmm, Hmm = deck_canvas()
    top, bot, lon_h, cb_band = 6.0, 1.0, 6.0, 17.6
    xf = 12.5; hf = Hmm - top - lon_h - bot; wf = hf / aspL; hg = Hmm - top - cb_band - bot
    spare = Wmm - 2.0 - xf - wf - 150.0                     # 확대 패널 폭 ≤ 150 mm(창 비율 = 패널 비율), 남는 폭은 간격(≤ 20 mm)과 양쪽 여백으로(v8, 오른쪽 빈 띠 제거)
    gap = min(20.0, max(8.0, spare)); xf += max(0.0, spare - gap) / 2.0; xg = xf + wf + gap; wg = min(150.0, Wmm - xg - 2.0)
    AX = lambda x, y, w, h: ax_mm(fig, x, y, w, h, Wmm, Hmm)
    RECT = lambda x, y, w, h: [x / Wmm, 1 - (y + h) / Hmm, w / Wmm, h / Hmm]
    name, c, hx = ZOOM[L["key"]][0]; zl = zoom_lim(L["tf"], c, hx, hx * hg / wg)
    ax = AX(xf, top, wf, hf); draw_lena(ax, D, M, wf, zl); letter(ax, "a", M); axF = ax
    ax = AX(xg, top, wg, hg); n_out = draw_lena_zoom(ax, D, M, wg, zl); letter(ax, "b", M); link(fig, axF, zl, ax, M)
    cbar(fig, RECT(xg + 4.0, top + hg + 2.5, 50.0, M.cb_h), MV.CM_ALT, L["norm"], "ALT (cm)", L["ticks"], L["ext"], M)   # b 아래 왼쪽; 범례는 오른쪽
    if n_out > 0:
        legend_aoa_zoom(ax, M, MV.CM_ALT)
    print(f"[F6 deck] R09 zoom {(zl[1] - zl[0]) / 1e3:.1f} × {(zl[3] - zl[2]) / 1e3:.1f} km, cells outside area of applicability: {n_out}")
    return deck_save(fig, "R09"), dict(zoom_circle_diameter_pt=round(float(np.sqrt(M.obs_zoom_s(wg / ((zl[1] - zl[0]) / 1e3)))), 2), zoom_window_km=[round((zl[1] - zl[0]) / 1e3, 1), round((zl[3] - zl[2]) / 1e3, 1)],
                                       zoom_cells_outside_aoa=n_out, gap_mm=round(gap, 1))


def deck():
    t0 = time.time(); D = load_data()
    res = [deck_R07(D), deck_R08(D), deck_R09(D)]; out = [r[0] for r in res]; M = Med("deck")
    write_meta(D, "deck", dict(canvas_in=[DECK_W_IN, DECK_H_IN], dpi=300, font_pt=dict(tick=M.fs_tick, label=M.fs_label, cbar=M.fs_cbar, letter=M.fs_letter),
                               zoom_circle_pt=dict(edge=M.obs_zoom_lw, casing=M.obs_case[1], note="diameter = ring centre line; outer diameter = diameter + edge"),
                               slides={sid: info for sid, (_, info) in zip(("R07", "R08", "R09"), res)}))
    print(f"[F6 deck] {len(out)} files, {time.time() - t0:.0f} s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--slide", action="store_true"); ap.add_argument("--deck", action="store_true"); ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    if a.cache:
        CACHE = Path(a.cache)
    if a.deck:
        S.setup(scale=1.69); deck()
    else:
        S.setup(scale=1.69 if a.slide else 1.0); f6(slide=a.slide)
    print("[F6] done")
