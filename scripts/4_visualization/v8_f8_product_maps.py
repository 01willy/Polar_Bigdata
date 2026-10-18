"""v8 F8 제품 지도 비교: 같은 해·같은 지역의 우리 지도(물리 + 잔차 ML = this study), Wei 2026, Liu 2024 ALT 지도와 관측 셀 잔차, 잔차 ML 보정량 지도.
구성(논문 180 × ~178 mm, 3 행; 행 사이 17 mm 띠에 경도 라벨·컬러바·범례)
  행 1 알래스카 2014: a 우리 1 km 지도(r1, 적용 범위 밖 흐림) | b Wei 2026 ALT_2014 | c Liu 2024 ALT_2014 | d 관측 셀 잔차(작은 지도 3: this study·Wei·Liu, 예측 − 관측).
  행 2 레나 델타 2016: e–h 같은 구성. 행 3 보정량 r1 − p1(물리 + 잔차 ML − 재보정 Stefan): i 알래스카 2011–2024 평균, j 레나 2016(행 폭을 채우는 높이).
  범례(지도 틀 밖, v8.1): a 아래 = 화소 범주(비영구동토 육지·자료 없음 빗금·물, 2 열), e 아래 = 적용 범위(안·밖, 1 열). ALT 컬러바는 b–c(f–g) 아래 중앙.
  작은 지도 제목은 위("this study"·"Wei 2026"·"Liu 2024"), RMSE 는 아래 설명 줄(지도 안에 글자 없음).
자료·표시
  우리 지도 data/processed/v5/maps/<Region>/pred_<year>.csv.gz(r1, p1), mean_2011_2024.csv.gz(aoa_in; 행 3 알래스카 r1 − p1). 1 km 셀 값을 음영 격자(알래스카 500 m,
    레나 250 m)에 지리 좌표 쌍선형 보간해 얹는다(행 = 위도, 열 = 행마다 간격이 다른 경도; 값 없는 이웃은 가중치에서 뺀다). 그릴지 여부·자료 없음·물은 화소가 든 셀 그대로.
  Wei 2026 data/raw/wei2026_alt_v2/extracted/ALT_<year>.tif(EPSG:3995, 1 km, nodata −9999, cm), Liu 2024 data/raw/liu2024_alt/ALT_<year>.tif(EPSG:4326, 1/120°,
    int16 cm × 100, ≤ 0 은 없음). 지역 창만 읽어 rasterio.warp.reproject(쌍선형)로 음영 격자에 옮긴다. 세 지도는 같은 색표(oslo_r)·같은 범위(우리 지도 1–99 백분위).
  화소 범주는 v7 F6 과 같다: 값 / 자료 없음(우리 격자 안인데 그 지도에 값이 없음: 빗금) / 비영구동토 육지(격자 밖: 평평한 회색) / 물.
    빗금은 F6 처럼 지도 화소 배열에 직접 그린다(45°, 주기 0.75 mm·선 0.20 mm, 종이 기준; SVG <pattern> 이 아니다). 범례 손잡이는 같은 주기·선 폭의 선(HatchHandler).
  관측 셀 잔차(d·h): 그 해 라벨(labels_cell_year_v5, in_forcing == 1)을 1 km 셀로 묶어 셀 평균 관측값을 만들고, 우리 = 지도 r1(그 셀), Wei·Liu = products_cell_year_v5 의
    그 해 값(셀 평균). 원 = 지점 묶음(지도 폭의 3 % 안의 셀을 단일 연결로 묶음), 색 = 묶음 안 셀 잔차 평균(broc, ±20 cm), 넓이 ∝ 그 지도에 값이 있는 셀 수, 값이 없으면 빈 원.
    RMSE 는 세 지도 모두 값이 있는 셀(공통 셀)에서 셀 단위로 계산해 적는다. 우리 지도는 전량 학습 지도라 관측 셀에서는 표본 안 값이다(블록 CV OOF 값은 JSON 에 함께 적는다).
  행 3 보정량: r1 − p1, broc 0 중심, 두 지도 공통 대칭 범위(|보정량| 99 백분위, 5 cm 단위).
산출  outputs/figures/paper/v7/F8_product_maps.{pdf,svg,png}; --slide 이면 v7/slide/F8_product_maps.png 와 크롭(a–j, abcd, efgh, ij);
      --deck 이면 v7/deck/F8_product_maps__deck_{R11,R12,R13}.png(12.0 × 5.10 in, 300 dpi). R11·R12 는 열 배치: 위 = 지도 a·b·c, 아래 = 같은 열의 잔차 작은 지도(d),
      오른쪽 열 = ALT 컬러바·범례(행 1), 잔차 컬러바·원 크기 열쇠(행 2). 사양·수치 v7/source_data/F8_product_maps.json(캡션 문장 초안 caption_notes 포함).
실행  OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v8_f8_product_maps.py [--slide | --deck]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window, from_bounds, intersection as win_intersection
from affine import Affine
from pyproj import Transformer
from scipy.ndimage import map_coordinates
from scipy.cluster.hierarchy import fcluster, linkage
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts" / "4_visualization", ROOT / "scripts" / "4_visualization" / "paper_v3"):
    sys.path.insert(0, str(_p))
import polar.style_v5 as S
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgb
import maps_alt_v3 as MV
import v7_f6_maps as F6
from v7_f6_maps import (Base, Med, REG, cell_table, cell_values, frame, coast, graticule, scalebar, label, letter, cbar, ax_mm, region_limits,
                        deck_canvas, HatchHandler, _anchor, LAND, WATER, HATCH, W_SHADE, FADE, CELL, PT_MM)
from matplotlib.patches import Patch

PROC = ROOT / "data" / "processed"; MAPS = PROC / "v5" / "maps"; RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "figures" / "paper" / "v7"; DECK_OUT = OUT / "deck"; SRC_OUT = OUT / "source_data"; CACHE = F6.CACHE
STEM = "F8_product_maps"; DECK_W_IN, DECK_H_IN = 12.0, 5.10
ROWS = [dict(key="alaska", region="Alaska", year=2014, lon_lab=(-160, -150), lon_lab_big=(-165, -155, -145), lon_lab_deck=(-160, -150), lat_lab=(60, 65, 70),
             scale_km=200, alt_step=5.0, mini_step=4, letters="abcd"),
        dict(key="lena", region="Lena", year=2016, lon_lab=(124, 128), lon_lab_big=(124, 126, 128, 130), lon_lab_deck=(124, 126, 128, 130), lat_lab=(72, 73),
             scale_km=50, alt_step=10.0, mini_step=2, letters="efgh")]
PRODUCTS = {"wei": dict(name="Wei 2026", dir=RAW / "wei2026_alt_v2" / "extracted", kind="float_nodata"),
            "liu": dict(name="Liu 2024", dir=RAW / "liu2024_alt", kind="int16_x100")}
METHODS = [("ours", "this study"), ("wei", "Wei 2026"), ("liu", "Liu 2024")]      # 작은 지도 제목. this study = 물리 + 잔차 ML(a·e 라벨)
CAT_LABELS = ["non-permafrost land", "no data", "water"]; AOA_LABELS = ["inside area of applicability", "outside area of applicability"]
RES_MAX = 20.0                                           # 잔차 원 색 범위(cm, broc 0 중심)
CLUSTER_FRAC = 0.03                                      # 지점 묶음 연결 거리 = 지도 폭 × 0.03
MINI_FILL = np.array(to_rgb("#F3F3F3"), np.float32); MINI_SHADE = 0.25     # 잔차 작은 지도: 옅은 바탕, 약한 음영
EDGE = "#4D4D4D"; MINUS = "−"
SIZE_REF = (1, 20)                                       # 원 크기 열쇠(셀 수)
HATCH_PAPER = (0.50, 0.15)                               # 논문판 빗금 주기·선 폭(mm): 33 mm 지도에서 1.5 mm 급 자료 없음 조각에도 줄이 보이게(F6 의 0.75·0.20 보다 가늘게)


# ================================================================ 자료: 셀 값 → 화소(지리 좌표 쌍선형)
def geo_index(base, ky0, kx0, tag):
    """화소마다 (아래 행 r0, 세로 가중치 t, 행 r0·r0+1 안의 소수 열 번호 fx0·fx1). 행마다 경도 간격이 달라 열 번호는 행별로 따로 둔다."""
    cache = CACHE / f"f8_{base.key}_{tag}_s{base.step}_geo.npz"
    if cache.exists():
        z = np.load(cache); return z["r0"], z["t"], z["fx0"], z["fx1"]
    t0 = time.time(); x0, x1, y0, y1 = base.extent; hh, ww = base.idx.shape
    gx = x0 + (np.arange(ww) + 0.5) * (x1 - x0) / ww; gy = y1 + (np.arange(hh) + 0.5) * (y0 - y1) / hh
    XX, YY = np.meshgrid(gx, gy); inv = Transformer.from_crs(base.crs, "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(XX.ravel(), YY.ravel()); del XX, YY
    fy = lat / CELL - 0.5 - ky0; r0 = np.floor(fy); t = (fy - r0).astype(np.float32); r0 = r0.astype(np.int32)
    def fx(r):
        dx = CELL / np.cos(np.radians((r + ky0 + 0.5) * CELL)); return (lon / dx - 0.5 - kx0).astype(np.float32)
    out = dict(r0=r0.reshape(hh, ww), t=t.reshape(hh, ww), fx0=fx(r0).reshape(hh, ww), fx1=fx(r0 + 1).reshape(hh, ww))
    CACHE.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, **out)
    print(f"  geo index {base.key}/{tag} s{base.step}: {hh}×{ww}, {time.time() - t0:.1f} s")
    return out["r0"], out["t"], out["fx0"], out["fx1"]


def grid2d(cells, shape, ky0, kx0, val):
    V = np.full(shape, np.nan, np.float32); ok = np.isfinite(val)
    V[cells.ky.values[ok] - ky0, cells.kx.values[ok] - kx0] = val[ok]; return V


def interp_px(V2d, geo):
    """셀 격자 값(NaN = 없음)을 화소에 쌍선형 보간. 값 없는 이웃은 가중치에서 뺀다(가중치 합 < 0.05 면 NaN)."""
    r0, t, fx0, fx1 = geo; hh, ww = r0.shape
    W = np.isfinite(V2d).astype(np.float32); Vz = np.where(W > 0, V2d, 0.0).astype(np.float32)
    num = np.zeros(hh * ww, np.float32); den = np.zeros(hh * ww, np.float32)
    for r, fx, w in ((r0, fx0, 1.0 - t), (r0 + 1, fx1, t)):
        c = np.vstack([r.ravel().astype(np.float32), fx.ravel()])
        num += w.ravel() * map_coordinates(Vz, c, order=1, mode="constant", cval=0.0)
        den += w.ravel() * map_coordinates(W, c, order=1, mode="constant", cval=0.0)
    return np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan).astype(np.float32).reshape(hh, ww)


# ================================================================ 자료: 제품 래스터 → 음영 격자
def product_px(base, prod, year):
    """제품 GeoTIFF 의 지역 창을 읽어 음영 격자(base)에 쌍선형 재투영. 반환 float32 (H, W), 없음 = NaN. 캐시."""
    cache = CACHE / f"f8_{base.key}_{prod}_{year}_s{base.step}.npz"
    if cache.exists():
        return np.load(cache)["v"]
    t0 = time.time(); P = PRODUCTS[prod]; path = P["dir"] / f"ALT_{year}.tif"
    x0, x1, y0, y1 = base.extent; hh, ww = base.idx.shape
    dst_tr = Affine((x1 - x0) / ww, 0.0, x0, 0.0, -(y1 - y0) / hh, y1)
    with rasterio.open(path) as s:
        tf = Transformer.from_crs(base.crs, s.crs, always_xy=True); n = 300
        xs = np.r_[np.linspace(x0, x1, n), np.full(n, x1), np.linspace(x1, x0, n), np.full(n, x0)]
        ys = np.r_[np.full(n, y0), np.linspace(y0, y1, n), np.full(n, y1), np.linspace(y1, y0, n)]
        px, py = tf.transform(xs, ys); pad = 3.0 * max(abs(s.res[0]), abs(s.res[1]))
        win = from_bounds(px.min() - pad, py.min() - pad, px.max() + pad, py.max() + pad, s.transform).round_offsets().round_lengths()
        win = win_intersection(win, Window(0, 0, s.width, s.height))
        arr = s.read(1, window=win); wtr = s.window_transform(win); src_crs = s.crs; nod = s.nodata
    if P["kind"] == "float_nodata":
        src = np.where((arr == nod) | ~np.isfinite(arr), np.nan, arr).astype(np.float32)
    else:
        src = np.where(arr <= 0, np.nan, arr.astype(np.float32) * 0.01).astype(np.float32)
    dst = np.full((hh, ww), np.nan, np.float32)
    reproject(src, dst, src_transform=wtr, src_crs=src_crs, src_nodata=np.nan, dst_transform=dst_tr, dst_crs=base.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    CACHE.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, v=dst)
    print(f"  product {prod} {year} → {base.key} s{base.step}: window {arr.shape}, finite {np.isfinite(dst).mean() * 100:.0f} %, {time.time() - t0:.1f} s")
    return dst


# ================================================================ 그리기: 화소 값 드레이프
def render_px(base, vpx, cmap, norm, drawn, nodata, aoa=None, hatch=(6, 1), w_shade=W_SHADE, fill=None):
    """화소 값(vpx)을 색으로 바꿔 음영과 곱한다. drawn = 색칠 화소, nodata = 빗금 화소, aoa = 화소별 적용 범위 가중치(0–1, 밖은 육지색 혼합), fill = 상수 바탕색."""
    water = base.water_cell | (~base.land & ~drawn)
    rgb = np.empty(base.idx.shape + (3,), np.float32); rgb[:] = LAND; rgb[water] = WATER
    hh, ww = base.idx.shape; P, wd = hatch
    if wd > 0 and nodata is not None and nodata.any():
        line = ((np.arange(hh, dtype=np.int32)[:, None] + np.arange(ww, dtype=np.int32)[None, :]) % P) < wd
        rgb[nodata & line] = HATCH; del line
    if fill is not None:
        col = np.broadcast_to(fill, rgb.shape)
    else:
        v = np.where(drawn & np.isfinite(vpx), vpx, float(np.nanmean(vpx[drawn])) if drawn.any() else 0.0)
        col = cmap(norm(v))[..., :3].astype(np.float32)
    if aoa is not None:
        f = (FADE * (1.0 - np.clip(aoa, 0.0, 1.0)))[..., None]; col = col * (1.0 - f) + f * LAND
    rgb[drawn] = col[drawn]
    nd = nodata if nodata is not None else np.zeros_like(drawn)
    sh = np.where((drawn | nd) & base.valid, base.shade, base.shade_fill); sh = np.where(water, 1.0, sh)
    out = rgb * ((1 - w_shade) + w_shade * sh)[..., None]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def nearest_px(base, val):
    """셀 표 순서 값을 화소(가장 가까운 셀)로. 반환 (값, 그릴 화소, 자료 없음 화소)."""
    has = base.idx >= 0; v = np.full(base.idx.shape, np.nan, np.float32); v[has] = val[base.idx[has]]
    drawn = np.isfinite(v) & base.valid
    nodata = has & base.land & ~drawn & ~base.water_cell
    return v, drawn, nodata


def product_masks(base, vpx):
    has = base.idx >= 0
    drawn = np.isfinite(vpx) & base.valid & base.land & ~base.water_cell
    return drawn, has & base.land & ~drawn & ~base.water_cell


# ================================================================ 자료: 관측 셀 잔차
def residual_table(row, pred_r1, tf, lim):
    region, yr = row["region"], row["year"]
    lab = pd.read_csv(PROC / "v5" / "labels_cell_year_v5.csv"); lab = lab[lab.in_forcing == 1]
    pr = pd.read_csv(PROC / "v5" / "products_cell_year_v5.csv")
    l = lab[(lab.region == region) & (lab.year == yr)].copy()
    ky = np.floor(l.lat.values / CELL).astype(np.int64); dx = CELL / np.cos(np.radians((ky + 0.5) * CELL)); kx = np.floor(l.lon.values / dx).astype(np.int64)
    l["cell_id"] = [f"{a}_{b}" for a, b in zip(ky, kx)]
    for p in PRODUCTS:
        l[p] = pr[(pr["product"] == p) & (pr.year == yr)].set_index("loc_id").value_cm.reindex(l.loc_id).values
    oof_path = PROC / "v5" / "runs" / "core_v6" / "preds" / f"{region}__block__year_obs__F8TS__R1__catboost__lam0.5.csv.gz"
    if oof_path.exists():
        o = pd.read_csv(oof_path, usecols=["loc_id", "year", "pred"]); o = o[o.year == yr].groupby("loc_id").pred.mean()
        l["oof"] = o.reindex(l.loc_id).values
    else:
        l["oof"] = np.nan
    l["ours"] = pred_r1.reindex(l.cell_id).values
    c = l.groupby("cell_id").agg(lat=("lat", "mean"), lon=("lon", "mean"), n_lab=("alt_cm", "size"), obs=("alt_cm", "mean"), wei=("wei", "mean"), liu=("liu", "mean"),
                                 oof=("oof", "mean"), ours=("ours", "mean"))
    c["x"], c["y"] = tf.transform(c.lon.values, c.lat.values)
    thr = CLUSTER_FRAC * (lim[1] - lim[0])
    c["cl"] = fcluster(linkage(c[["x", "y"]].values, "single"), thr, "distance") if len(c) > 1 else 1
    ms = ("ours", "wei", "liu", "oof")
    for m in ms:
        c[f"d_{m}"] = c[m] - c.obs
    agg = dict(x=("x", "mean"), y=("y", "mean"), n=("obs", "size"), lat=("lat", "mean"), lon=("lon", "mean"))
    agg.update({f"d_{m}": (f"d_{m}", "mean") for m in ms}); agg.update({f"n_{m}": (m, "count") for m in ms})
    g = c.groupby("cl").agg(**agg).reset_index()
    common = c[c[["ours", "wei", "liu"]].notna().all(axis=1)]
    rmse = {m: float(np.sqrt(np.mean(common[f"d_{m}"] ** 2))) for m in ms}
    rmse_own = {m: dict(rmse=float(np.sqrt(np.nanmean(c[f"d_{m}"] ** 2))), n=int(c[m].notna().sum())) for m in ms}
    lab_level = {m: dict(rmse=float(np.sqrt(np.nanmean((l[m] - l.alt_cm) ** 2))), n=int(l[m].notna().sum())) for m in ms}
    return dict(cells=c, clusters=g, n_common=int(len(common)), rmse=rmse, rmse_own=rmse_own, label_level=lab_level, n_labels=int(len(l)), thr_m=float(thr))


# ================================================================ 자료 적재(행 하나)
def load_row(row):
    t0 = time.time(); key, region, yr = row["key"], row["region"], row["year"]
    cells, lut, ky0, kx0 = cell_table(key)
    pred = pd.read_csv(MAPS / region / f"pred_{yr}.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    mean = pd.read_csv(MAPS / region / "mean_2011_2024.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    r1 = cell_values(key, cells, pred.r1); p1 = cell_values(key, cells, pred.p1)
    aoa = np.nan_to_num(cell_values(key, cells, mean.aoa_in.astype(float)), nan=0.0)
    corr = cell_values(key, cells, mean.r1 - mean.p1) if key == "alaska" else (r1 - p1)
    base = Base(key, cells, lut, ky0, kx0, step=REG[key]["step"], tag="full"); tf = base.tf; lim = region_limits(key, tf)
    geo = geo_index(base, ky0, kx0, "full"); G = lambda v: grid2d(cells, lut.shape, ky0, kx0, v)
    v_r1, drawn, nodata = nearest_px(base, r1); v_c, drawn_c, nodata_c = nearest_px(base, corr)
    px = dict(r1=interp_px(G(r1), geo), corr=interp_px(G(corr), geo), aoa=np.nan_to_num(interp_px(G(aoa), geo), nan=0.0))
    px["r1"] = np.where(np.isfinite(px["r1"]), px["r1"], v_r1); px["corr"] = np.where(np.isfinite(px["corr"]), px["corr"], v_c)
    prod = {p: product_px(base, p, yr) for p in PRODUCTS}
    pm = {p: product_masks(base, prod[p]) for p in PRODUCTS}
    fin = r1[np.isfinite(r1)]; lo, hi = MV.pct_range(fin, step=row["alt_step"]); norm = Normalize(lo, hi); ticks = np.arange(lo, hi + 1e-6, 10.0)
    allv = np.concatenate([fin] + [prod[p][pm[p][0]] for p in PRODUCTS]); ext = MV.extend_of(allv, lo, hi)
    bm = Base(key, cells, lut, ky0, kx0, step=row["mini_step"], tag="full")
    _, drawn_m, _ = nearest_px(bm, r1)
    rgb_mini = render_px(bm, None, None, None, drawn_m, None, fill=MINI_FILL, hatch=(6, 0), w_shade=MINI_SHADE)
    res = residual_table(row, pred.r1, tf, lim)
    corr_fin = corr[np.isfinite(corr)]
    D = dict(row, cells=cells, base=base, tf=tf, lim=lim, asp=(lim[3] - lim[2]) / (lim[1] - lim[0]), px=px, drawn=drawn, nodata=nodata, drawn_c=drawn_c, nodata_c=nodata_c,
             prod=prod, pm=pm, lo=lo, hi=hi, norm=norm, ticks=ticks, ext=ext, base_mini=bm, rgb_mini=rgb_mini, res=res, corr_sym=MV.sym_max(corr_fin),
             prod_range={p: [float(np.nanpercentile(prod[p][pm[p][0]], 1)), float(np.nanpercentile(prod[p][pm[p][0]], 99))] for p in PRODUCTS},
             n_cells_drawn=int(np.isfinite(r1).sum()))
    print(f"  load {region} {yr}: ALT range {lo}–{hi} ({ext}), corr sym {D['corr_sym']}, cells {res['n_labels']} labels → {len(res['cells'])} cells, {len(res['clusters'])} clusters, "
          f"common {res['n_common']}, RMSE {({k: round(v, 1) for k, v in res['rmse'].items()})}, {time.time() - t0:.0f} s")
    return D


def load_all():
    D = {r["key"]: load_row(r) for r in ROWS}
    v = max(D[k]["corr_sym"] for k in D); D["normC"] = TwoSlopeNorm(0.0, -v, v); D["ticksC"] = np.arange(-v, v + 1e-6, 5.0 if v <= 20 else 10.0); D["vC"] = v
    D["normR"] = TwoSlopeNorm(0.0, -RES_MAX, RES_MAX); D["ticksR"] = np.arange(-RES_MAX, RES_MAX + 1e-6, 10.0)
    return D


# ================================================================ 패널
def hatch_mm(M):
    return HATCH_PAPER if M.paper else M.hatch_mm


def draw_map(ax, Dk, M, w_mm, kind, text, D, lon_lab=(), lat_lab=(), scale=False):
    base = Dk["base"]; hp = base.hatch_px(w_mm, Dk["lim"], hatch_mm(M))
    if kind == "ours":
        rgb = render_px(base, Dk["px"]["r1"], MV.CM_ALT, Dk["norm"], Dk["drawn"], Dk["nodata"], aoa=Dk["px"]["aoa"], hatch=hp)
    elif kind == "corr":
        rgb = render_px(base, Dk["px"]["corr"], MV.CM_DIFF, D["normC"], Dk["drawn_c"], Dk["nodata_c"], aoa=Dk["px"]["aoa"], hatch=hp)
    else:
        dr, nd = Dk["pm"][kind]; rgb = render_px(base, Dk["prod"][kind], MV.CM_ALT, Dk["norm"], dr, nd, hatch=hp)
    frame(ax, rgb, base.extent, Dk["lim"], M); coast(ax, Dk["key"], Dk["tf"], M)
    graticule(ax, Dk["key"], Dk["tf"], Dk["lim"], M, lon_lab=lon_lab, lat_lab=lat_lab)
    if scale:
        scalebar(ax, Dk["lim"], Dk["scale_km"], M)
    label(ax, text, M)


def circle_d1(M):
    """묶음 원 지름(pt) = d1 × √(셀 수)."""
    return 1.6 if M.paper else 2.8


def draw_mini(ax, Dk, M, m, name, D):
    """잔차 작은 지도: 옅은 음영 바탕 + 묶음 원(색 = 평균 잔차, 넓이 ∝ 값 있는 셀 수), 값 없는 묶음은 ×. 제목은 위 중앙, RMSE 는 아래 설명 줄(지도 안에 글자 없음)."""
    bm = Dk["base_mini"]; frame(ax, Dk["rgb_mini"], bm.extent, Dk["lim"], M); coast(ax, Dk["key"], Dk["tf"], M); graticule(ax, Dk["key"], Dk["tf"], Dk["lim"], M)
    g = Dk["res"]["clusters"]; d1 = circle_d1(M); lw = 0.3 if M.paper else 0.6
    for _, r in g.sort_values("n", ascending=False).iterrows():
        n = int(r[f"n_{m}"]); val = r[f"d_{m}"]
        if n > 0 and np.isfinite(val):
            ax.scatter(r.x, r.y, s=(d1 * np.sqrt(n)) ** 2, facecolor=MV.CM_DIFF(D["normR"](val)), edgecolor=EDGE, lw=lw, zorder=6)
        else:
            ax.scatter(r.x, r.y, s=(1.7 * d1) ** 2, marker="x", c=EDGE, lw=1.4 * lw, zorder=6)
    rm = Dk["res"]["rmse"][m]
    ax.text(0.5, 1.03 if M.paper else 1.02, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=M.fs_tick if M.paper else M.fs_label)
    ax.text(0.5, -0.04 if M.paper else -0.02, f"RMSE {rm:.1f} cm", transform=ax.transAxes, ha="center", va="top", fontsize=M.fs_tick)


def legend_f8(ax, M, base, cmap=None, items=("cat",), ncol=1, loc="upper left", bbox=(0.0, -0.06), transform=None):
    """지도 틀 밖 범례. cat = 화소 범주(비영구동토 육지 / 자료 없음 빗금 / 물; 육지·빗금 색은 F6 처럼 평균 음영을 곱한다), aoa = 적용 범위(안 = 셀 색, 밖 = 육지색 40 % 혼합).
    빗금 손잡이는 지도와 같은 주기·선 폭의 선을 직접 그린다(HatchHandler). 반환 Legend."""
    f = (1.0 - W_SHADE) + W_SHADE * base.shade_fill
    land = tuple((LAND * f).tolist()); hatch = tuple((HATCH * f).tolist()); edge = "#BEBEBE"
    h, lab, hm = [], [], {}
    if "cat" in items:
        h_hatch = Patch(facecolor=land, edgecolor=hatch)
        h += [Patch(facecolor=land, edgecolor=edge, lw=M.leg_edge), h_hatch, Patch(facecolor=tuple(WATER.tolist()), edgecolor=edge, lw=M.leg_edge)]
        lab += CAT_LABELS; hm[h_hatch] = HatchHandler(land, hatch, hatch_mm(M)[0] * PT_MM, hatch_mm(M)[1] * PT_MM, edge, M.leg_edge)
    if "aoa" in items:
        col = np.array(cmap(0.55)[:3], np.float32); out = col * (1.0 - FADE) + FADE * LAND
        h += [Patch(facecolor=tuple(col.tolist()), edgecolor="none"), Patch(facecolor=tuple(out.tolist()), edgecolor="none")]; lab += AOA_LABELS
    return ax.legend(h, lab, fontsize=M.fs_tick, handler_map=hm, ncol=ncol, **_anchor(loc, bbox, transform), **M.leg)


def key_line(fig, rect_mm, M, Wmm, Hmm, items=("circles", "x"), text_right=None):
    """한 줄 열쇠: 원 크기(셀 수 1·20), × 값 없음, 오른쪽 글. 축은 보이지 않는다. 반환 축."""
    x, y, w, h = rect_mm; ax = ax_mm(fig, x, y, w, h, Wmm, Hmm); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    d1 = circle_d1(M); lw = 0.3 if M.paper else 0.6; cx = 0.6; ch = M.fs_tick / PT_MM * 0.52; gap = 1.6 if M.paper else 2.6; tp = 0.8 if M.paper else 1.8   # 손잡이–글 간격
    if "circles" in items:
        for n in SIZE_REF:
            d_pt = d1 * np.sqrt(n); d_mm = d_pt / PT_MM
            ax.scatter(cx + d_mm / 2, h / 2, s=d_pt ** 2, facecolor="white", edgecolor=EDGE, lw=lw, clip_on=False)
            t = f"{n} cell" + ("s" if n > 1 else ""); ax.text(cx + d_mm + tp, h / 2, t, va="center", ha="left", fontsize=M.fs_tick)
            cx += d_mm + tp + len(t) * ch + gap
    if "x" in items:
        d_mm = 1.7 * d1 / PT_MM
        ax.scatter(cx + d_mm / 2, h / 2, s=(1.7 * d1) ** 2, marker="x", c=EDGE, lw=1.4 * lw, clip_on=False)
        t = "no value"; ax.text(cx + d_mm + tp, h / 2, t, va="center", ha="left", fontsize=M.fs_tick); cx += d_mm + tp + len(t) * ch + gap
    if text_right:
        ax.text(w, h / 2, text_right, va="center", ha="right", fontsize=M.fs_tick)
    return ax


def text_line(fig, rect_mm, M, Wmm, Hmm, text, ha="left"):
    x, y, w, h = rect_mm; ax = ax_mm(fig, x, y, w, h, Wmm, Hmm); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    ax.text(0 if ha == "left" else w, h / 2, text, va="center", ha=ha, fontsize=M.fs_tick); return ax


def cbar_v(fig, rect, cmap, norm, lab, ticks, extend, M):
    cax = fig.add_axes(rect)
    cb = MV.colorbar(fig, cax, cmap, norm, lab, ticks, MV.Med("paper" if M.paper else "slide"), extend, orientation="vertical")
    if M.paper:
        cb.ax.tick_params(labelsize=M.fs_tick); cb.set_label(lab, fontsize=M.fs_cbar, labelpad=1.5)
    return cax


# ================================================================ 논문판
def f8(D, slide=False):
    """논문판. 행 1·2: 지도 3(33 mm) + 잔차 작은 지도 열; 행 사이 17 mm 띠 = 경도 라벨 + 컬러바(b–c 아래 중앙 50 mm) + 범례(a 아래 범주 2 열, e 아래 적용 범위 1 열).
    행 3: 두 보정량 지도가 세로 컬러바와 함께 그림 폭을 채우는 높이(H3)로."""
    t0 = time.time(); M = Med("paper"); A, L = D["alaska"], D["lena"]
    Wmm = S.W2; xa, mw, gm = 7.0, 33.0, 2.5; xr = xa + 3 * mw + 2 * gm + 4.5; wr = Wmm - xr - 1.5; gmini = 2.0; mini_w = (wr - 2 * gmini) / 3
    blk = 17.0; g3 = 8.0; cb_col = 4.0 + M.cb_h + 8.5 + 1.5                 # 행 3 세로 컬러바 열: 간격·막대·눈금 글자와 라벨·오른쪽 여백
    H3 = float(np.floor(2.0 * (Wmm - xa - g3 - cb_col) / (1.0 / A["asp"] + 1.0 / L["asp"])) / 2.0)
    y1 = 3.5; y2 = y1 + mw * A["asp"] + blk; y3 = y2 + mw * L["asp"] + blk; Hmm = y3 + H3 + 5.5
    fig = S.figure(Wmm, Hmm)
    AX = lambda x, y, w, h: ax_mm(fig, x, y, w, h, Wmm, Hmm)
    RECT = lambda x, y, w, h: [x / Wmm, 1 - (y + h) / Hmm, w / Wmm, h / Hmm]
    x_cb = xa + mw + gm + (2 * mw + gm) / 2.0 - 25.0                        # b–c 아래 중앙
    for Dk, y0 in ((A, y1), (L, y2)):
        mh = mw * Dk["asp"]; lt = Dk["letters"]; name = S.REGION_EN[Dk["region"]]
        texts = {"ours": f"{name} {Dk['year']}\nphysics + residual ML", "wei": "Wei 2026", "liu": "Liu 2024"}
        for j, kind in enumerate(("ours", "wei", "liu")):
            ax = AX(xa + j * (mw + gm), y0, mw, mh)
            draw_map(ax, Dk, M, mw, kind, texts[kind], D, lon_lab=Dk["lon_lab"], lat_lab=Dk["lat_lab"] if j == 0 else (), scale=(j == 0)); letter(ax, lt[j], M)
            if j == 0:
                ax0 = ax
        S.register(lt[1], cbar(fig, RECT(x_cb, y0 + mh + 5.0, 50.0, M.cb_h), MV.CM_ALT, Dk["norm"], "ALT (cm)", Dk["ticks"], Dk["ext"], M))
        y_leg = y0 + mh + 4.5                                               # 경도 라벨 아래. 범례는 축 a·e 에 붙어 크롭에 함께 들어간다
        if Dk is A:
            legend_f8(ax0, M, Dk["base"], items=("cat",), ncol=2, bbox=(xa / Wmm, 1 - y_leg / Hmm), transform=fig.transFigure)
        else:
            legend_f8(ax0, M, Dk["base"], cmap=MV.CM_ALT, items=("aoa",), ncol=1, bbox=(xa / Wmm, 1 - y_leg / Hmm), transform=fig.transFigure)
        yi = y0 + 2.5; hmini = mini_w * Dk["asp"]
        for j, (m, nm) in enumerate(METHODS):
            ax = AX(xr + j * (mini_w + gmini), yi, mini_w, hmini); draw_mini(ax, Dk, M, m, nm, D)
            if j == 0:
                letter(ax, lt[3], M, x=0.0, y=1.0 + 3.6 / hmini)
            else:
                S.register(lt[3], ax)
        ycb = yi + hmini + 5.0
        S.register(lt[3], cbar(fig, RECT(xr + 7.0, ycb, wr - 14.0, M.cb_h), MV.CM_DIFF, D["normR"], "prediction − observed (cm)", D["ticksR"], "both", M))
        S.register(lt[3], key_line(fig, (xr, ycb + 9.5, wr, 3.5), M, Wmm, Hmm))
        S.register(lt[3], text_line(fig, (xr, ycb + 13.0, wr, 3.0), M, Wmm, Hmm, f"RMSE on {Dk['res']['n_common']} common cells"))
    wA3, wL3 = H3 / A["asp"], H3 / L["asp"]; xL3 = xa + wA3 + g3
    ax = AX(xa, y3, wA3, H3); draw_map(ax, A, M, wA3, "corr", "Alaska, mean 2011–2024", D, lon_lab=A["lon_lab_big"], lat_lab=A["lat_lab"], scale=True); letter(ax, "i", M)
    ax = AX(xL3, y3, wL3, H3); draw_map(ax, L, M, wL3, "corr", "Lena Delta, 2016", D, lon_lab=L["lon_lab_big"], lat_lab=L["lat_lab"], scale=True); letter(ax, "j", M)
    S.register("j", cbar_v(fig, RECT(xL3 + wL3 + 4.0, y3 + (H3 - 40.0) / 2.0, M.cb_h, 40.0), MV.CM_DIFF, D["normC"], "residual ML correction (cm)", D["ticksC"], "both", M))
    for k in ("abcd", "efgh", "ij"):
        S.register(k, *[a for c in k for a in S.PANELS.get(c, [])])
    out = S.export(fig, STEM, outdir=OUT)
    print(f"[F8] {len(out)} files, {time.time() - t0:.0f} s; size {Wmm:.0f}×{Hmm:.1f} mm, row 3 {H3:.1f} mm tall, right edge of row 3 colour bar column {xL3 + wL3 + cb_col - 1.5:.1f} mm")
    return out, dict(size_mm=[Wmm, round(Hmm, 1)], map_w_mm=mw, mini_w_mm=round(mini_w, 2), row_band_mm=blk, row3_h_mm=H3,
                     legend=dict(a="pixel categories, 2 columns under a", e="area of applicability, 1 column under e"), alt_cbar="50 mm centred under b–c and f–g")


# ================================================================ 덱 원판
def deck_save(fig, sid, h_in=DECK_H_IN):
    from PIL import Image
    DECK_OUT.mkdir(parents=True, exist_ok=True); p = DECK_OUT / f"{STEM}__deck_{sid}.png"
    fig.savefig(p, dpi=300, facecolor="white")
    r = fig.canvas.get_renderer(); bb = fig.get_tightbbox(r); W, H = fig.get_size_inches(); use_w, use_h = bb.width / W, bb.height / H
    plt.close(fig)
    with Image.open(p) as im:
        rgb = im.convert("RGB")
    rgb.save(p, dpi=(300, 300)); sz = rgb.size; want = (int(round(DECK_W_IN * 300)), int(round(h_in * 300)))
    assert sz == want, f"{p.name}: {sz} != {want}"
    print(f"[F8 deck] {p.name}: {sz[0]}×{sz[1]} px, canvas use {use_w * 100:.0f} % × {use_h * 100:.0f} %")
    return p


def deck_row(D, Dk, sid):
    """R11/R12 열 배치(12.0 × 5.10 in): 위 = a 우리 | b Wei | c Liu(70 mm), 아래 = 같은 열의 잔차 작은 지도(d, 제목 위·RMSE 아래), 두 행이 캔버스 높이를 채운다.
    오른쪽 열(≈ 66 mm): 행 1 = ALT 가로 컬러바 + 범례(범주 3 + 적용 범위 2, 1 열), 행 2 = 잔차 가로 컬러바 + 원 크기 열쇠 3 줄."""
    M = Med("deck"); fig, Wmm, Hmm = deck_canvas(); asp = Dk["asp"]
    top, xa, gm, lon_h, title_h, cap_h, bot = 5.5, 12.5, 4.0, 5.0, 4.5, 4.5, 1.5
    mw = 70.0; wm = min(44.0, (Hmm - top - lon_h - 1.0 - title_h - cap_h - bot) / asp - mw); mh, hm = mw * asp, wm * asp
    xr = xa + 3 * mw + 2 * gm + 6.0; wr = Wmm - xr - 2.0; cbw = 54.0; y_mini = top + mh + lon_h + 1.0 + title_h
    AX = lambda x, y, w, h: ax_mm(fig, x, y, w, h, Wmm, Hmm)
    RECT = lambda x, y, w, h: [x / Wmm, 1 - (y + h) / Hmm, w / Wmm, h / Hmm]
    name = S.REGION_EN[Dk["region"]]; texts = {"ours": f"{name} {Dk['year']}\nphysics + residual ML", "wei": "Wei 2026", "liu": "Liu 2024"}
    for j, kind in enumerate(("ours", "wei", "liu")):
        x = xa + j * (mw + gm)
        ax = AX(x, top, mw, mh)
        draw_map(ax, Dk, M, mw, kind, texts[kind], D, lon_lab=Dk["lon_lab_deck"], lat_lab=Dk["lat_lab"] if j == 0 else (), scale=(j == 0)); letter(ax, "abc"[j], M)
        if j == 0:
            ax0 = ax
        m, nm = METHODS[j]; draw_mini(AX(x + (mw - wm) / 2.0, y_mini, wm, hm), Dk, M, m, nm, D)
    fig.text((xa + (mw - wm) / 2.0) / Wmm, 1 - (y_mini - 0.8) / Hmm, "d", fontsize=M.fs_letter, fontweight="bold", ha="left", va="bottom")   # 첫 작은 지도 왼쪽 위, 제목 줄
    cbar(fig, RECT(xr + 2.0, top + 1.0, cbw, M.cb_h), MV.CM_ALT, Dk["norm"], "ALT (cm)", Dk["ticks"], Dk["ext"], M)
    legend_f8(ax0, M, Dk["base"], cmap=MV.CM_ALT, items=("cat", "aoa"), ncol=1, bbox=(xr / Wmm, 1 - (top + 1.0 + M.cb_h + 14.0) / Hmm), transform=fig.transFigure)
    cbar(fig, RECT(xr + 2.0, y_mini, cbw, M.cb_h), MV.CM_DIFF, D["normR"], "prediction − observed (cm)", D["ticksR"], "both", M)
    yk = y_mini + M.cb_h + 14.0
    key_line(fig, (xr, yk, wr, 5.0), M, Wmm, Hmm, items=("circles",))
    key_line(fig, (xr, yk + 5.5, wr, 5.0), M, Wmm, Hmm, items=("x",))
    text_line(fig, (xr, yk + 11.0, wr, 5.0), M, Wmm, Hmm, f"RMSE on {Dk['res']['n_common']} common cells")
    print(f"  {sid}: map {mw:.1f}×{mh:.1f} mm, mini {wm:.1f}×{hm:.1f} mm, right column {wr:.1f} mm, bottom of captions {y_mini + hm + cap_h:.1f} / {Hmm:.1f} mm")
    return deck_save(fig, sid)


def deck_R13(D):
    """R13: a 알래스카 평균 보정량 | b 레나 2016 보정량, 오른쪽 세로 컬러바."""
    M = Med("deck"); fig, Wmm, Hmm = deck_canvas(); A, L = D["alaska"], D["lena"]
    top, xa = 6.0, 12.5; h = Hmm - top - 6.0 - 1.0; wA, wL = h / A["asp"], h / L["asp"]; xL = xa + wA + 12.0
    AX = lambda x, y, w, hh: ax_mm(fig, x, y, w, hh, Wmm, Hmm)
    RECT = lambda x, y, w, hh: [x / Wmm, 1 - (y + hh) / Hmm, w / Wmm, hh / Hmm]
    ax = AX(xa, top, wA, h); draw_map(ax, A, M, wA, "corr", "Alaska, mean 2011–2024", D, lon_lab=(-165, -155, -145), lat_lab=A["lat_lab"], scale=True); letter(ax, "a", M)
    ax = AX(xL, top, wL, h); draw_map(ax, L, M, wL, "corr", "Lena Delta, 2016", D, lon_lab=L["lon_lab_deck"], lat_lab=L["lat_lab"], scale=True); letter(ax, "b", M)
    cbar_v(fig, RECT(xL + wL + 6.0, top + (h - 60.0) / 2.0, M.cb_h, 60.0), MV.CM_DIFF, D["normC"], "residual ML correction (cm)", D["ticksC"], "both", M)
    return deck_save(fig, "R13")


# ================================================================ 사양 JSON
def patch_stats(Dk):
    """자료 없음 화소의 연결 조각 통계(km²): 빗금이 눈에 보이려면 조각이 빗금 주기보다 커야 한다. 논문 33 mm 알래스카 지도에서 1 mm ≈ 48 km."""
    from scipy.ndimage import label as cc_label
    km2 = (0.25 * Dk["base"].step) ** 2; out = {}
    for m, nd in (("ours", Dk["nodata"]), *[(k, Dk["pm"][k][1]) for k in PRODUCTS]):
        lab, n = cc_label(nd); sizes = np.bincount(lab.ravel())[1:] if n else np.array([0]); del lab
        out[m] = dict(area_km2=round(float(nd.sum() * km2), 1), patches=int(n), largest_patch_km2=round(float(sizes.max() * km2), 1),
                      share_of_area_in_patches_over_25km2=round(float(sizes[sizes * km2 >= 25].sum() / max(1, sizes.sum())), 3))
    return out


def crs_name(crs):
    e = crs.to_epsg()
    return f"EPSG:{e}" if e else f"local polar stereographic on the WGS 84 ellipsoid, no EPSG code (hillshade grid CRS; {crs.to_proj4()})"


def write_spec(D, outputs, layout=None, medium="paper"):
    SRC_OUT.mkdir(parents=True, exist_ok=True); p = SRC_OUT / f"{STEM}.json"
    spec = json.loads(p.read_text()) if p.exists() else {}
    rows = {}
    for r in ROWS:
        Dk = D[r["key"]]; res = Dk["res"]; yr = r["year"]
        nd_px = dict(ours=int(Dk["nodata"].sum()), **{k: int(Dk["pm"][k][1].sum()) for k in PRODUCTS})
        rows[r["region"]] = dict(
            year=yr, letters=r["letters"],
            sources=dict(ours=str((MAPS / r["region"] / f"pred_{yr}.csv.gz").relative_to(ROOT)) + " (r1; p1 for row 3 Lena)",
                         aoa=str((MAPS / r["region"] / "mean_2011_2024.csv.gz").relative_to(ROOT)) + " (aoa_in; Alaska row 3: r1 − p1)",
                         wei=str((PRODUCTS["wei"]["dir"] / f"ALT_{yr}.tif").relative_to(ROOT)) + " (EPSG:3995, 1 km, nodata −9999, cm)",
                         liu=str((PRODUCTS["liu"]["dir"] / f"ALT_{yr}.tif").relative_to(ROOT)) + " (EPSG:4326, 1/120°, int16 cm×100, ≤0 = no data)",
                         labels="data/processed/v5/labels_cell_year_v5.csv (in_forcing == 1, region, year)",
                         products_at_cells="data/processed/v5/products_cell_year_v5.csv (product wei/liu, year)",
                         hillshade=f"data/processed/map_{r['key']}/hillshade_v1/shade_250m.tif", grid=f"data/processed/map_{r['key']}/{r['key']}_grid_x25_v1.csv.gz"),
            display=dict(crs=crs_name(Dk["base"].crs), pixel_m=250.0 * Dk["base"].step, pixels=list(map(int, Dk["base"].idx.shape)),
                         ours="geographic bilinear interpolation of 1 km cell values (interp_px)",
                         products="rasterio.warp.reproject bilinear onto the hillshade grid", aoa="outside area of applicability blended 40 % with land grey",
                         water="#EAF2F9", non_permafrost_land="#E6E6E6 flat (land outside our 1 km grid)",
                         no_data=dict(meaning="inside our 1 km grid (cell table) but no value in that map", rendering="45° hatch drawn into the map pixel array as in F6 "
                                      f"(paper and slide: period {HATCH_PAPER[0]:g} mm, line {HATCH_PAPER[1]:g} mm; deck: {Med('deck').hatch_mm[0]:.2f} / {Med('deck').hatch_mm[1]:.2f} mm); "
                                      "raster, not an SVG <pattern>; the legend handle draws the same lines as vector paths",
                                      pixels_hatched=nd_px, pixels_hatched_where_our_map_has_a_value={k: int((Dk["pm"][k][1] & Dk["drawn"]).sum()) for k in PRODUCTS},
                                      patches=patch_stats(Dk)),
                         n_cells_drawn=Dk["n_cells_drawn"]),
            colour=dict(cmap="oslo_r (cmcrameri, 0.12–0.90)", alt_range_cm=[Dk["lo"], Dk["hi"]], alt_range_rule=f"1–99 percentile of our map, {r['alt_step']:g} cm step",
                        extend=Dk["ext"], product_p1_p99={k: [round(a, 1), round(b, 1)] for k, (a, b) in Dk["prod_range"].items()}),
            residual=dict(unit="prediction − observed (cm), cell level", cells_from=f"{res['n_labels']} labels → {len(res['cells'])} one-km cells (cell mean of observed ALT)",
                          ours_at_cell="map r1 at the cell (full-data map; in-sample at observed cells)", products_at_cell="cell mean of products_cell_year_v5 values",
                          circle=f"site cluster: single-linkage within {res['thr_m'] / 1e3:.0f} km ({CLUSTER_FRAC * 100:.0f} % of map width); colour = mean residual over cells with a value; "
                                 f"area ∝ number of cells with a value (d = d1·√n, d1 = {circle_d1(Med('paper'))} pt paper / {circle_d1(Med('deck'))} pt deck); hollow = no value",
                          cmap="broc", range_cm=[-RES_MAX, RES_MAX], n_common_cells=res["n_common"],
                          rmse_common_cells_cm={k: round(v, 2) for k, v in res["rmse"].items()},
                          rmse_own_cells_cm={k: dict(rmse=round(v["rmse"], 2), n=v["n"]) for k, v in res["rmse_own"].items()},
                          rmse_label_level_cm={k: dict(rmse=round(v["rmse"], 2), n=v["n"]) for k, v in res["label_level"].items()},
                          oof_note="oof = block-CV out-of-fold R1 prediction (runs/core_v6/preds, seeds averaged), for reference only; the figure prints map r1",
                          clusters=[dict(cluster=int(g.cl), lat=round(float(g.lat), 4), lon=round(float(g.lon), 4), n_cells=int(g.n),
                                         **{f"d_{m}": (None if not np.isfinite(g[f"d_{m}"]) else round(float(g[f"d_{m}"]), 2)) for m in ("ours", "wei", "liu")},
                                         **{f"n_{m}": int(g[f"n_{m}"]) for m in ("ours", "wei", "liu")}) for _, g in res["clusters"].iterrows()]))
    A, L = D["alaska"], D["lena"]
    spec.update(dict(figure=STEM, created=time.strftime("%Y-%m-%d %H:%M:%S"), script="scripts/4_visualization/v8_f8_product_maps.py", rows=rows,
                     method_alias=dict(this_study="physics + residual ML (map r1); labelled 'physics + residual ML' on panels a/e and 'this study' on the residual minis d/h"),
                     legend=dict(categories=CAT_LABELS, applicability=AOA_LABELS,
                                 paper="categories under panel a (2 columns), applicability under panel e (1 column); both outside the map frames",
                                 deck="R11/R12 right column, one list under the ALT colour bar; R13 has none (colour bar only)",
                                 handles="land and hatch swatches multiplied by the mean hillshade as on the map; applicability = cmap(0.55) and its 40 % land blend"),
                     correction=dict(definition="r1 − p1 (physics + residual ML − Stefan, recalibrated)", cmap="broc", range_cm=[-D["vC"], D["vC"]],
                                     range_rule="max over the two maps of the 99th percentile of |r1 − p1|, 5 cm step",
                                     alaska="mean_2011_2024.csv.gz r1 − p1", lena="pred_2016.csv.gz r1 − p1", per_map_sym={k: D[k]["corr_sym"] for k in ("alaska", "lena")}),
                     caption_notes=[
                         f"Panels i and j share one colour scale, ±{D['vC']:g} cm, fixed to the larger of the two panels' 99th percentiles of |r1 − p1| "
                         f"(Alaska {A['corr_sym']:g} cm, Lena Delta {L['corr_sym']:g} cm) so that the two regions are directly comparable; the Lena field is therefore muted relative to its own range.",
                         "'this study' = physics + residual ML (the r1 map); the maps in a and e are full-data maps, so their values at observed cells are in-sample.",
                         f"RMSE in d and h is computed on the cells where all three maps have a value ({A['res']['n_common']} cells in Alaska 2014, {L['res']['n_common']} in Lena Delta 2016).",
                         "Hatching marks cells inside our 1 km grid for which the respective map has no value; flat grey is land outside the grid (non-permafrost); water is white-blue.",
                         f"ALT colour range per row: Alaska {A['lo']:g}–{A['hi']:g} cm, Lena Delta {L['lo']:g}–{L['hi']:g} cm (1st–99th percentile of our map), shared by the three maps of the row."],
                     fonts=dict(paper="FreeSans 6–7 pt, panel letters 8 pt bold", deck="FreeSans 12 pt ticks/legend, 13 pt labels, 14 pt colorbar labels"),
                     size_key_cells=list(SIZE_REF)))
    spec.setdefault("outputs", {})[medium] = [str(Path(o).relative_to(ROOT)) for o in outputs]
    if layout:
        spec.setdefault("layout", {})[medium] = layout
    p.write_text(json.dumps(spec, ensure_ascii=False, indent=1)); print(f"[F8] spec {p.relative_to(ROOT)}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--slide", action="store_true"); ap.add_argument("--deck", action="store_true"); a = ap.parse_args()
    t0 = time.time()
    if a.deck:
        S.setup(scale=1.69); D = load_all()
        out = [deck_row(D, D["alaska"], "R11"), deck_row(D, D["lena"], "R12"), deck_R13(D)]
        write_spec(D, out, medium="deck")
    else:
        S.setup(scale=1.69 if a.slide else 1.0); D = load_all()
        out, lay = f8(D, slide=a.slide); write_spec(D, out, lay, medium="slide" if a.slide else "paper")
    print(f"[F8] done, {time.time() - t0:.0f} s")
