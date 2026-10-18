"""v8 표지 그림(덱 전용). 알래스카 2022년 ALT 지도(물리 잔차 결합 R1, 적용 범위 안)를 250 m 지형 음영 위에 드레이프해 남색 바탕의 오른쪽에 두고,
왼쪽을 남색으로 녹여 제목 자리(왼쪽 40 %)를 비운다. 그림 안 글자는 없다(제목·문장은 덱이 얹는다).
v9(2026-10-15, 설계 지침 v9 항 2): 오른쪽 아래의 라벨 'Alaska · ALT 2022 · …' 와 200 km 축척 막대를 지우고, 그 자리를 비우던 모서리 녹임도 없앴다.
장식용 자산이다: 색 범위가 논문 지도와 다르고 색막대가 없으므로 정량 그림으로 재사용하지 않는다(meta 의 decorative_only).

구성
  캔버스 4000 × 2250 px(16:9), 바탕 남색 #0F1C2E, 오른쪽 위에 약한 방사형 밝은 톤(#152338, L* 13). 지도는 음영 격자 좌표계(EPSG:3338, 북쪽이 위)에서 그대로 두고
  캔버스 오른쪽 가장자리를 −141° 자오선의 북단(71.5°N)에 맞춰 격자 영역의 동쪽 경계(사선)가 캔버스 밖에 놓이게 한다. 세로는 북극해 연안이 위,
  아래는 약 60.5°N 에서 캔버스 밖으로 흘러나가고 아래 가장자리는 짧게 녹인다.
  화소 범주(음영 격자 500 m 화소 기준)
    그린 셀      oslo_r(2022 값 1–99 백분위, 5 cm 단위), 음영 평균 0 변조(그늘만 어둡게, 양지는 +8 % 까지). 셀 안 DEM 결손 화소는 평평한 음영.
    적용 범위 밖  색을 55 % 탈색(같은 휘도의 회색과 혼합, 8 % 어둡게). 어두운 회색과 섞지 않는다(섞으면 비영구동토 육지와 구분되지 않는다).
    내륙 호수 셀  1 km 물 셀(gray_reason == water)이 Natural Earth 육지 안에 있으면 이웃 셀 색(정규화 합성곱 σ 2.5 px, 큰 호수 안쪽은 σ 12 px)의 94 % 로 채운다.
                 투명하게 두면 노스슬로프의 열카르스트 호수 5 만여 화소가 남색 점으로 흩어져 결손처럼 보인다(v8 1차 결함).
    작은 구멍     지도 안에 갇힌 값 없는 셀 덩어리(격자 밖 셀·cci/soil 결측 셀, 연결 성분 < 4,000 px = 1,000 km²; 테셰크푹 호수 둘레·북동 브룩스산맥 3 블록 등)는
                 이웃 색을 30 % 탈색한 색 × 음영으로 채우고 가장자리를 σ 3 px 로 녹인다(값은 없고 지형만 잇는다). 어두운 회색으로 두면 밝은 지도 안의 얼룩으로 보인다.
    비영구동토 육지(큰 덩어리)  평평한 짙은 회색 #2A3340. 음영을 입히지 않는다: 이 영역의 DEM 은 1° 타일 단위로 듬성듬성해 입히면 직사각형 질감 조각이 생긴다.
                 격자 경계(25 km 타일 계단)는 σ 10 px(5 km) 가우스로 양쪽을 녹여 계단을 지운다. 해안은 녹이지 않는다(육지 화소만으로 정규화).
    물           투명 → 남색 바탕. 해안선은 옅은 청회색 가는 선.
  색표는 논문 지도와 같은 oslo_r 이되 밝은 쪽 0.12–0.62 만 쓴다(0.62 의 L* 37; 남색 L* 10, 육지 회색 L* 21 과 ΔL* ≥ 15). 가로 알파 경사(캔버스 42 → 58 %)로
  지도 왼쪽을 남색에 녹이고(해안선은 경사의 세제곱), 아래 가장자리 600 px(비영구동토 육지 덩어리가 바탕으로 가라앉는다)도 녹인다.
  저장 뒤 PNG 를 다시 읽어 측정한다(meta 의 qa): 육지 회색 대 바다 L*, 지도 바닥 L* 대 남색, 지도 안 어두운 고립 화소 수, 격자 밖 육지의 최대 연결 성분.
입력  data/processed/v5/maps/Alaska/{pred_2022, mean_2011_2024}.csv.gz(r1, aoa_in), data/processed/map_alaska/{alaska_grid_x25_v1.csv.gz, hillshade_v1/shade_250m.tif}
      (v7_f6_maps.Base 의 캐시 outputs/figures/paper/v7/_cache/alaska_full_s2_v2.npz 를 쓴다), Natural Earth 10m 해안선(maps_alt_v3.ne_layers).
산출  deck/assets/v8/cover_bg_v8.png(4000 × 2250 px RGB), cover_bg_v8_preview1600.png(1600 px 폭 미리보기), cover_bg_v8_meta.json(색 범위·측정값).
실행  OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v8_cover_art.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, uniform_filter
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "4_visualization", ROOT / "scripts" / "4_visualization" / "paper_v3"):
    sys.path.insert(0, str(p))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import Normalize, to_rgb
import cmcrameri.cm as cmc
import polar.style_v5 as S
import v7_f6_maps as F6
import maps_alt_v3 as MV

OUT_DIR = ROOT / "deck" / "assets" / "v8"
W, H = 4000, 2250                      # 캔버스(px)
DPI = 250                              # 16 × 9 in × 250 dpi = 4000 × 2250 px
YEAR = 2022
NAVY = np.array(to_rgb("#0F1C2E"), np.float32)
NAVY_GLOW = np.array(to_rgb("#152338"), np.float32)     # 방사형 밝은 톤의 중심 색(L* 13: 육지 회색 L* 21 보다 충분히 어둡게)
GLOW_CENTRE, GLOW_SIGMA = (0.80, 0.32), 0.36            # 중심(캔버스 폭·높이 비율)·폭(폭 비율): 지도의 가장 밝은 노스슬로프 뒤
LAND_DARK = np.array(to_rgb("#2A3340"), np.float32)     # 비영구동토 육지·격자 안 값 없는 셀(평지·평균 색)
COAST = "#6F8098"
CM_RANGE = (0.12, 0.62)                # oslo_r 사용 구간(논문 지도 0.12–0.90). 0.62 의 L* 37 이 남색(10)·육지 회색(21)과 ΔL* ≥ 15
CM_COVER = MV.lut(cmc.oslo_r, *CM_RANGE)
W_SHADE = 0.42                         # 그린 셀 음영 변조 폭(평균 0 변조: 1 + W·(shade − 평균))
FADE_OUT = 0.55                        # 적용 범위 밖 셀: 탈색 비율(같은 휘도 회색과 혼합)
FADE_OUT_DIM = 0.92                    # 적용 범위 밖 셀: 밝기 배율
LAKE_SIGMA, LAKE_SIGMA_FAR, LAKE_DIM = 2.5, 12.0, 0.94   # 내륙 호수 셀 채움: 이웃 색 정규화 합성곱 σ(px; 가까운 이웃이 없으면 먼 σ)·밝기 배율
HOLE_MAX_PX = 4000                     # 지도 안 작은 구멍으로 보고 채우는 값 없는 덩어리의 최대 연결 성분 크기(500 m 화소; 1,000 km²)
HOLE_DESAT, HOLE_DIM, HOLE_SIGMA = 0.3, 0.98, 3.0   # 작은 구멍 채움: 이웃 색 탈색 비율·밝기 배율·가장자리 녹임 σ(px)
EDGE_SIGMA = 10.0                      # 격자 경계 녹임 σ(음영 격자 px; 500 m/px → 5 km)
AOA_SIGMA = 3.0                        # 적용 범위 가중 완화 σ(px)
X_FADE = (0.42, 0.58)                  # 가로 알파 경사(캔버스 폭 비율): 왼쪽은 0, 오른쪽은 1. 왼쪽 42 % 는 글자 자리
Y_FADE_PX = 600                        # 아래 가장자리 녹임 높이(px)
X_RIGHT_LONLAT = (-141.0, 71.5)        # 캔버스 오른쪽 가장자리를 맞출 점(격자 영역 북동 모서리)
Y_TOP_M = 2.462e6                      # 캔버스 위 가장자리(EPSG:3338 y, 음영 격자 위 끝)
LAT_BOTTOM = 60.5                      # 캔버스 아래 가장자리 위도(중앙 자오선 −154° 기준)
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def setup_fonts():
    for f in (S.F["latin_regular_file"], S.F["latin_bold_file"]):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [S.F["latin"], "DejaVu Sans"], "axes.unicode_minus": False, "svg.fonttype": "none"})


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0); return t * t * (3.0 - 2.0 * t)


def lstar(rgb):
    """sRGB(0–1, (..., 3)) → CIE L*."""
    rgb = np.asarray(rgb, np.float64)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = lin @ LUM.astype(np.float64)
    return 116.0 * np.where(y > 0.008856, np.cbrt(y), 7.787 * y + 16.0 / 116.0) - 16.0


def nconv(col, m, sigma):
    """정규화 합성곱: 마스크 m(0/1) 안의 색을 σ 로 퍼뜨려 이웃 색을 만든다(마스크 밖 화소도 값을 가진다)."""
    d = m.astype(np.float32); den = gaussian_filter(d, sigma)
    num = np.stack([gaussian_filter(col[..., c] * d, sigma) for c in range(3)], axis=-1)
    return np.where(den[..., None] > 1e-3, num / np.maximum(den, 1e-6)[..., None], col), den


# ================================================================ 자료
def load():
    key = "alaska"; cells, lut, ky0, kx0 = F6.cell_table(key)
    pred = pd.read_csv(F6.MAPS / "Alaska" / f"pred_{YEAR}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1
    aoa_in = pd.read_csv(F6.MAPS / "Alaska" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "aoa_in"]).set_index("cell_id").aoa_in.astype(float)
    val = F6.cell_values(key, cells, pred); aoa = np.nan_to_num(F6.cell_values(key, cells, aoa_in), nan=0.0)
    base = F6.Base(key, cells, lut, ky0, kx0, step=2, tag="full")
    lo, hi = MV.pct_range(pred.values)
    n_in = int(np.isfinite(val[aoa >= 0.5]).sum()); n_all = int(np.isfinite(val).sum())
    print(f"  {YEAR} r1: cells {n_all} (inside AoA {n_in}), colour range {lo:.0f}–{hi:.0f} cm")
    return dict(key=key, base=base, val=val, aoa=aoa, lo=lo, hi=hi, norm=Normalize(lo, hi), n_in=n_in, n_all=n_all)


# ================================================================ 지도 층(음영 격자 해상도)
def render_layer(D):
    """셀 값 → 색 × 음영. 반환 rgb float32 (H, W, 3), alpha (H, W; 물 = 0), 범주 마스크 dict, 화소 수 dict.
    범주: inside(그린 셀 + 채운 호수 셀 + 채운 작은 구멍) / land_out(큰 비영구동토 육지 덩어리; 평평한 회색) / ocean(투명)."""
    base, val, aoa, norm = D["base"], D["val"], D["aoa"], D["norm"]
    idx = base.idx; has = idx >= 0; valid = base.valid
    v = np.full(idx.shape, np.nan, np.float32); v[has] = val[idx[has]]
    drawn = np.isfinite(v)                                            # 값 있는 셀(DEM 결손 화소 포함: 평평한 음영으로 그린다)
    lake = base.water_cell & base.land & ~drawn                        # 내륙 호수 셀 → 이웃 색으로 채움
    ocean = ~base.land & ~drawn                                        # 바다(해안 밖 물 셀 포함)
    nodata = base.land & ~drawn & ~lake                                # 격자 밖 육지 + 격자 안 값 없는 셀
    lab, ncomp = label(nodata); sizes = np.bincount(lab.ravel())
    hole = nodata & (sizes[lab] < HOLE_MAX_PX)                         # 지도 안 작은 구멍(이웃 휘도의 회색으로 채움)
    land_out = nodata & ~hole                                          # 큰 덩어리: 평평한 #2A3340
    # 그린 셀 색 · 적용 범위 밖 탈색 · 음영
    col = CM_COVER(norm(np.where(drawn, v, np.nanmean(v[drawn]))))[..., :3].astype(np.float32)
    w = np.zeros(idx.shape, np.float32); w[has] = aoa[idx[has]]
    d = drawn.astype(np.float32); den = gaussian_filter(d, AOA_SIGMA); num = gaussian_filter(w * d, AOA_SIGMA)
    w = np.where(den > 1e-3, num / np.maximum(den, 1e-6), w); del d, den, num
    f = (FADE_OUT * (1.0 - w))[..., None]; grey = (col @ LUM)[..., None]
    col = (col * (1.0 - f) + f * grey) * (1.0 - (1.0 - FADE_OUT_DIM) * f); del f, grey
    m_sh = float(base.shade[drawn & valid].mean()); sh = np.where(valid, base.shade, m_sh)
    mod = np.clip(1.0 + W_SHADE * (sh - m_sh), 0.0, 1.08)[..., None]    # 평균 0 변조: 평지는 색표 값 그대로, 그늘은 어둡게, 양지는 +8 % 까지만
    col = col * mod; del sh
    # 내륙 호수 셀: 이웃 색(가까운 σ, 없으면 먼 σ)의 94 %. 호수 화소의 음영은 평평하므로 변조하지 않는다
    cs, den = nconv(col, drawn, LAKE_SIGMA); cs_far, _ = nconv(col, drawn, LAKE_SIGMA_FAR)
    cs = np.where((den > 0.05)[..., None], cs, cs_far); del cs_far, den
    col = np.where(lake[..., None], cs * LAKE_DIM, col); del cs
    # 작은 구멍: 이웃 색을 탈색한 색 × 음영(값은 없고 지형만 잇는다), 가장자리 녹임
    cs, _ = nconv(col, drawn | lake, LAKE_SIGMA_FAR); grey = (cs @ LUM)[..., None]
    hole_rgb = (cs * (1.0 - HOLE_DESAT) + grey * HOLE_DESAT) * HOLE_DIM * mod; del cs, grey, mod
    hm = gaussian_filter(hole.astype(np.float32), HOLE_SIGMA)[..., None]
    col = col * (1.0 - hm) + hole_rgb * hm; del hole_rgb, hm
    inside = drawn | lake | hole
    # 격자 경계 녹임: 육지 화소만으로 정규화(해안은 그대로). a = 1 지도 안쪽, 0 회색 육지 안쪽
    land_any = inside | land_out
    a = gaussian_filter(inside.astype(np.float32), EDGE_SIGMA) / np.maximum(gaussian_filter(land_any.astype(np.float32), EDGE_SIGMA), 1e-6)
    a = np.clip(np.where(land_any, a, 0.0), 0.0, 1.0)[..., None]
    cs2, _ = nconv(col, inside, EDGE_SIGMA)
    own = np.where(inside[..., None], col, cs2); del cs2
    rgb = a * own + (1.0 - a) * LAND_DARK[None, None, :]; del own, a
    rgb[ocean] = NAVY
    alpha = land_any.astype(np.float32)
    masks = dict(inside=inside, land_out=land_out, ocean=ocean)
    n = dict(px_drawn=int(drawn.sum()), px_lake_filled=int(lake.sum()), px_dem_void_in_cells=int((drawn & ~valid).sum()),
             px_hole_filled=int(hole.sum()), n_hole_components=int(len(np.unique(lab[hole]))),
             px_grid_nodata_total=int((has & nodata).sum()), px_land_outside_grid_total=int((~has & nodata).sum()),
             px_land_out_flat=int(land_out.sum()), px_ocean=int(ocean.sum()))
    return np.clip(rgb, 0, 1), alpha, masks, n


# ================================================================ 캔버스 합성
def canvas_geometry(base):
    """캔버스 px ↔ EPSG:3338 m. 오른쪽 가장자리 = X_RIGHT_LONLAT 의 x, 위 = Y_TOP_M, 아래 = LAT_BOTTOM(중앙 자오선). 반환 (x_left, y_top, m/px)."""
    xr, _ = base.tf.transform(*X_RIGHT_LONLAT); _, yb = base.tf.transform(-154.0, LAT_BOTTOM)
    s = (Y_TOP_M - yb) / H
    return xr - W * s, Y_TOP_M, s


def _place(arr, base, geom, resample):
    """음영 격자 배열(H, W[, 3], uint8)을 캔버스 해상도로 재표집해 캔버스 크기 배열(uint8)로 돌려준다(캔버스 밖은 0)."""
    x_left, y_top, s = geom; x0, x1, y0, y1 = base.extent
    cx0, cx1 = (x0 - x_left) / s, (x1 - x_left) / s; cy0, cy1 = (y_top - y1) / s, (y_top - y0) / s
    tw, th = int(round(cx1 - cx0)), int(round(cy1 - cy0)); ox, oy = int(round(cx0)), int(round(cy0))
    im = np.asarray(Image.fromarray(arr).resize((tw, th), resample))
    out = np.zeros((H, W) + arr.shape[2:], np.uint8)
    sx0, sy0 = max(0, -ox), max(0, -oy); dx0, dy0 = max(0, ox), max(0, oy)
    ww, hh = min(tw - sx0, W - dx0), min(th - sy0, H - dy0)
    out[dy0:dy0 + hh, dx0:dx0 + ww] = im[sy0:sy0 + hh, sx0:sx0 + ww]
    return out


def place_layer(rgb, alpha, masks, base, geom):
    L = _place((rgb * 255).astype(np.uint8), base, geom, Image.LANCZOS).astype(np.float32) / 255.0
    A = _place((alpha * 255).astype(np.uint8), base, geom, Image.LANCZOS).astype(np.float32) / 255.0
    M = {k: _place(m.astype(np.uint8), base, geom, Image.NEAREST).astype(bool) for k, m in masks.items()}
    return L, A, M


def background():
    """남색 바탕 + 오른쪽 위 방사형 밝은 톤(가우스)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy, sig = GLOW_CENTRE[0] * W, GLOW_CENTRE[1] * H, GLOW_SIGMA * W
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig * sig))[..., None]
    return NAVY * (1 - g) + NAVY_GLOW * g


def fade_mask():
    """알파 경사: 가로(왼쪽 0 → 오른쪽 1) × 아래 가장자리 녹임. 반환 (지도용, 해안선용: 왼쪽 경사를 세제곱해 윤곽선 잔상을 줄인다).
    v9: 오른쪽 아래 라벨을 지웠으므로 라벨 자리 둘레 녹임은 없다."""
    x = np.arange(W, dtype=np.float32) / W; gx = smoothstep((x - X_FADE[0]) / (X_FADE[1] - X_FADE[0]))
    y = np.arange(H, dtype=np.float32); gy = smoothstep((H - y) / Y_FADE_PX)
    return gx[None, :] * gy[:, None], (gx ** 3)[None, :] * gy[:, None]


def coast_layer(base, geom, lw_pt=0.9):
    """해안선을 같은 캔버스 크기의 투명 Agg 버퍼에 그려 알파(0–1)로 돌려준다."""
    from shapely.ops import transform as shp_transform
    x_left, y_top, s = geom
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_alpha(0.0)
    for g in MV.ne_layers(base.key)["coastline"]:
        gg = shp_transform(base.tf.transform, g)
        for part in getattr(gg, "geoms", [gg]):
            xs, ys = np.asarray(part.xy[0]), np.asarray(part.xy[1])
            ax.plot((xs - x_left) / s, (y_top - ys) / s, color=COAST, lw=lw_pt, solid_capstyle="round", solid_joinstyle="round")
    fig.canvas.draw(); buf = np.asarray(fig.canvas.buffer_rgba()); plt.close(fig)
    assert buf.shape[:2] == (H, W), buf.shape
    return buf[..., 3].astype(np.float32) / 255.0


def compose(D):
    base = D["base"]; geom = canvas_geometry(base)
    rgb, alpha, masks, stats = render_layer(D)
    L, A, M = place_layer(rgb, alpha, masks, base, geom); del rgb, alpha, masks
    fade, fade_c = fade_mask(); A = A * fade
    bg = background(); out = bg * (1 - A[..., None]) + L * A[..., None]
    ca = coast_layer(base, geom) * fade_c * 0.85
    out = out * (1 - ca[..., None]) + np.array(to_rgb(COAST), np.float32) * ca[..., None]
    M["unfaded"] = fade > 0.98; M["fade90"] = fade > 0.90
    return np.clip(out, 0, 1), geom, stats, M


# ================================================================ 저장·측정
def qa(png, M):
    """저장된 PNG 를 다시 읽어 잰다. 녹이지 않은 영역(unfaded)에서 (1) 육지 회색 대 바다 L*, (2) 지도 바닥 L* 대 남색,
    (3) 지도 안 어두운 고립 화소(L* < 16, 주변 9×9 평균 L* > 40) 수, (4) 격자 밖 육지의 최대 연결 성분(캔버스 육지 화소 대비), (5) 왼쪽 40 %·45 % 의 최대 RGB."""
    img = np.asarray(Image.open(png).convert("RGB"), np.float32) / 255.0; Ls = lstar(img)
    u = M["unfaded"]; ocean = M["ocean"] & u; inside = M["inside"] & u
    dist = distance_transform_edt(M["land_out"]); land_flat = M["land_out"] & M["fade90"] & (dist > 40)   # 경계 녹임 밖의 평평한 육지 회색(녹임 ≥ 0.9 인 곳)
    if not land_flat.any():
        land_flat = M["land_out"] & M["fade90"] & (dist > 15)
    navy_L = float(lstar(NAVY)); L_land = Ls[land_flat]; L_oc = Ls[ocean]; L_in = Ls[inside]
    nb = uniform_filter(Ls, 9, mode="nearest")
    spk = inside & (Ls < 16.0) & (nb > 40.0)
    lab, n = label(M["land_out"]); sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    land_px = int((M["inside"] | M["land_out"]).sum())
    left = img[:, : int(0.40 * W)]
    pct = lambda x, q: round(float(np.percentile(x, q)), 1) if x.size else None
    r = dict(navy_Lstar=round(navy_L, 1),
             land_flat_Lstar=dict(mean=round(float(L_land.mean()), 1) if L_land.size else None, p10=pct(L_land, 10), p90=pct(L_land, 90), n_px=int(land_flat.sum())),
             ocean_Lstar=dict(mean=round(float(L_oc.mean()), 1), max=round(float(L_oc.max()), 1), n_px=int(ocean.sum())),
             delta_Lstar_land_vs_ocean_mean=round(float(L_land.mean() - L_oc.mean()), 1) if L_land.size else None,
             map_floor_Lstar=dict(p1=round(float(np.percentile(L_in, 1)), 1), p5=round(float(np.percentile(L_in, 5)), 1), min=round(float(L_in.min()), 1)),
             delta_Lstar_map_p1_vs_navy=round(float(np.percentile(L_in, 1) - navy_L), 1),
             dark_isolated_px_inside_map=int(spk.sum()),
             land_outside_grid_canvas=dict(px=int(M["land_out"].sum()), frac_of_land_px=round(float(M["land_out"].sum() / max(land_px, 1)), 3),
                                           largest_component_px=int(sizes.max()), largest_component_frac_of_land_px=round(float(sizes.max() / max(land_px, 1)), 3)),
             left40_max_rgb=[int(v) for v in (left.reshape(-1, 3).max(axis=0) * 255).round()],
             left45_max_rgb=[int(v) for v in (img[:, : int(0.45 * W)].reshape(-1, 3).max(axis=0) * 255).round()])
    return r


def finalize(img, geom, D, stats, M):
    setup_fonts(); OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_left, y_top, s = geom
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI); ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=(0, W, H, 0), interpolation="none"); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    p = OUT_DIR / "cover_bg_v8.png"; fig.savefig(p, dpi=DPI, facecolor=tuple(NAVY.tolist())); plt.close(fig)
    with Image.open(p) as im:
        rgb = im.convert("RGB")
    assert rgb.size == (W, H), rgb.size
    rgb.save(p, dpi=(300, 300))
    prev = OUT_DIR / "cover_bg_v8_preview1600.png"; rgb.resize((1600, 900), Image.LANCZOS).save(prev)
    q = qa(p, M)
    meta = dict(decorative_only=True, note="Cover background for the deck. Colour sub-range differs from the paper maps and there is no colour bar: not a quantitative figure.",
                year=YEAR, colour_range_cm=[D["lo"], D["hi"]], cmap=f"oslo_r {CM_RANGE[0]:.2f}-{CM_RANGE[1]:.2f} (paper maps: 0.12-0.90)", cells_drawn=D["n_all"], cells_inside_aoa=D["n_in"],
                m_per_px=s, text="none (v9: label line and 200 km scale bar removed)", x_fade=X_FADE, y_fade_px=Y_FADE_PX, size_px=[W, H],
                layer_px_counts_500m=stats, render=dict(edge_sigma_px=EDGE_SIGMA, lake_sigma_px=LAKE_SIGMA, lake_dim=LAKE_DIM, fade_out_desaturate=FADE_OUT, fade_out_dim=FADE_OUT_DIM,
                                                      w_shade=W_SHADE, hole_max_px=HOLE_MAX_PX, hole_desat=HOLE_DESAT, hole_dim=HOLE_DIM, hole_sigma_px=HOLE_SIGMA, land_hex="#2A3340", land_shaded=False, glow_hex="#152338"), qa=q)
    (OUT_DIR / "cover_bg_v8_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[cover] {p} {rgb.size[0]}×{rgb.size[1]} px; preview {prev.name}; {s:.0f} m/px; colour {D['lo']:.0f}–{D['hi']:.0f} cm; no text")
    print("[cover] qa " + json.dumps(q))
    return p, prev


def main():
    t0 = time.time(); D = load()
    img, geom, stats, M = compose(D); out = finalize(img, geom, D, stats, M)
    print(f"[cover] done {time.time() - t0:.0f} s"); return out


if __name__ == "__main__":
    main()
