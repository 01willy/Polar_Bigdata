"""F3 방법 구조·평가 설계(v7, docs/FIGURE_CONTRACT_v7.md · docs/DESIGN_BRIEF_v8_2026-10-15.md §2 항 13). 10-15 4차 개정(검토 6건 반영).

a 구조도(10-15 v9 개정), 세 영역 Input | Model | Output = 옅은 파선 둥근 사각형(토큰 neutral.coast) + 사각형 안 왼쪽 위 작은 회색 머리 라벨.
  패널 문자 옆 회색 'Alaska, 2014'(a 의 예시 지역·연도).
  Input  = 항목별 실자료 썸네일(어긋난 스택 아님, 단위 없음, 층마다 자기 백분위 범위): 'Climate, 2014 (8)' 행 MAAT · FDD · Snow · TDD(TDD 오른쪽 끝 → Stefan;
           10-16 MAAT 추가, 정적 행과 같은 열 가운데),
           'Static (15)' 행 Elevation · Slope · Soil carbon · Clay(괄호 = 묶음별 특징 수, testbed_v5.F8 과 meta.json feats 에서 센다; 합 23 = F8TS),
           훈련 전용 'Observed ALT'(적색 점 = 그 해 관측 셀, 적색 파선 → ⊖). 영구동토 확률(pfr_mean)은 F8TS 특징이 아니라 넣지 않는다.
  Model  = 'Stefan' 상자(녹, 이름만; 아래 회색 mathtext 'ALT = E√TDD') · 'Gradient boosting' 상자(황, 이름만). Input 사각형 오른쪽 변 → 학습 상자.
           훈련 전용 ⊖(관측 y − 물리 ŷ_P → 목표 y − ŷ_P, 적색 파선).
  Output = 썸네일 4장: Physics ALT · Residual (cm)(소형 색 막대) → ⊕(λ = meta.json lam) → ALT 2014(세로 색 막대 cm) · Applicability(AOA 안·밖 열쇠).
  범례(data flow · training only)는 Output 사각형 오른쪽 위. 흐름 선 끝 = 작은 채움 셰브런(삼각 화살촉 없음). E·λ·AOA 는 meta.json 에서 읽는다.
b 비교 구조 4종, a 와 같은 문법(학습 = 'ML' 황 상자, 물리 = 'Stefan' 녹 상자): 물리가 들어가는 자리를 한 단어로 적는다(input · labels · anchor).
c 다섯 분할(레나 델타 실제 셀), d 블록 부트스트랩 ΔRMSE 분포(x 눈금은 판 공통 5 cm 간격·왼쪽 한계는 5 의 배수로 내림).
자체 점검: 그리기 뒤 글자 상자끼리·글자 상자와 썸네일 틀의 겹침을 렌더러 좌표로 검사해 출력한다(_qa_overlap).

배치: a·b 는 덱 mm(304.8 폭) 좌표 하나로 그리고 논문판은 축 크기만 0.5906 배(글자 pt 는 판별 fsz()). 썸네일 가로/세로 = 자료 범위(위도 보정)로 계산.
자료: data/processed/v5/maps/Alaska/pred_2014.csv.gz(+ meta.json), mean_2011_2024.csv.gz(aoa_in), forcing_2014_thumb.csv.gz(grid/alaska_forcing_year.csv 에서
      2014 FDD·SWE 를 추려 캐시; 없으면 만든다), data/processed/map_alaska/alaska_grid_x25_v1.csv.gz(정적 공변량), labels_cell_year_v5.csv,
      runs/core_v6/preds/Lena__block__year_obs__F8TS__{P1,R1}__*.csv.gz, runs/core_v6/deltas_vs_ref.csv(대조).
실행: OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v7_f3_method.py [--slide | --deck]   (CPU 만)
산출: 논문판 outputs/figures/paper/v7/F3_method.{pdf,svg,png}, source_data/F3_method.json
      슬라이드판(--slide) slide/F3_method.png, slide/F3_method__{a,b,c,d,ab,cd}.png
      덱판(--deck) deck/F3_method__deck_M01.png(a + b, 12.0 × 4.65 in, 3600 × 1395 px) · deck/F3_method__deck_M02.png(c 한 줄 + d 아래, 같은 크기)
개정(10-15, 설계 지침 v9 항 15): a 를 Input · Model · Output 옅은 파선 사각형 + 머리 라벨로 나누고, 모델 상자에는 이름만('Stefan', 'Gradient boosting';
      Stefan 식은 상자 아래 mathtext ALT = E√TDD), 입력은 어긋난 특징 스택 대신 항목별 썸네일(기후 그 해: FDD · Snow · TDD, 정적: Elevation · Slope ·
      Soil carbon · Clay; 소제목에 묶음별 특징 수 8 · 15), 관측 ALT(훈련 전용) 유지. 영구동토 확률은 모델 특징(F8TS)이 아니라 입력에서 뺐다.
      흐름 선 끝은 삼각 화살촉 대신 작은 채움 셰브런. b 는 학습 상자 'ML' · 물리 상자 'Stefan'(글리프 '√' 없음), 방법 이름은 지침 1.1 표기.
      덱 글자: 축 라벨·블록 14 pt, 직접 라벨 13 pt, 눈금·소형 12 pt, 패널 문자 14 pt 굵게.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
import argparse, inspect, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3"))
import polar.style_v5 as S
import polar.testbed_v5 as TB
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch, Circle, Patch, Rectangle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe
from cmcrameri import cm as cmc
import maps_alt_v3 as MV

PROC = ROOT / "data" / "processed"; V5 = PROC / "v5"; RUN = V5 / "runs" / "core_v6"
OUT = ROOT / "outputs" / "figures" / "paper" / "v7"
STEM = "F3_method"
SRC: dict = {}        # 그림에 쓴 수치(source_data JSON)

# ---- 색·선(토큰 design/style_tokens_v4.json; 공정 블록은 계약서의 두 가지만)
GRN, GRN_EDGE = S.TOK["color"]["process"]["evaluation_physics"], "#86AE8A"   # 물리 #DCEBDD
YEL, YEL_EDGE = S.TOK["color"]["process"]["learning"], "#C9AE5B"             # 학습 #F7E9AE
INK = S.NEUTRAL["text_paper"]; AUX = S.NEUTRAL["text_aux"]; ZERO = S.NEUTRAL["zero_line"]
FRAME = "#9A9A9A"; LAND = "#EBEBEB"; COAST = "#9A9A9A"; GRAT = "#D0D0D0"; NODATA = S.NEUTRAL["no_data"]
TRAIN_RED = "#C0392B"; DASH = (0, (3.0, 1.6))
CHEV_W, CHEV_H = 2.6, 2.4                                                  # 흐름 셰브런 크기(덱 mm)
REGION_EDGE = S.NEUTRAL["coast"]                                            # Input·Model·Output 파선 사각형(옅은 회색, 토큰)
AOA_IN, AOA_OUT = "#9DBBD3", "#D2D2D2"                                       # 적용 범위 안(연청)·밖(회색)
# 채점 겹 5색: Okabe–Ito(청·하늘·주황·분홍·청록). 10-14 검토 반영
FOLD_HEX = ["#0072B2", "#56B4E9", "#E69F00", "#CC79A7", "#009E73"]
CM_YEAR = ListedColormap(cmc.bilbao_r(np.linspace(0.25, 0.95, 200)))         # 연도 홀드아웃: 순차(모래 → 짙은 갈색)
ALASKA_ASP = 1.0 / np.cos(np.deg2rad(65.5)); LENA_LAT = 72.2
CM_TDD = ListedColormap(cmc.lajolla_r(np.linspace(0.15, 0.88, 250)))      # 낮은 TDD = 밝음, 높은 TDD = 짙은 갈색
FS_SMALL = 6.0; FS_TINY = 5.5; FS_STACK = 6.5     # 스택 층 라벨은 6.5 pt(검토: 6 pt 는 바닥 5 pt 에 가깝다)

# ---- 판별 배율·글자 대응. 논문·슬라이드판은 모두 1(슬라이드판은 S.export 가 렌더 dpi 로 1.69배 키운다). 덱판은 deck() 이 바꾼다
DECK = False
K = 1.0                                   # 선 굵기·화살촉·점 반지름 배율
KM = 1.0                                  # 지도(c) 마커 배율(지도 폭 비)
FS_DECK = {7.0: 14.0, 6.0: 13.0, 5.5: 12.0}   # 논문 pt → 덱 pt(토큰 font.slide: 축 라벨 14, 직접 라벨 13, 눈금·최소 12)


def sc(v):
    """선 굵기·화살촉·점 반지름 배율 훅."""
    return v * K


def fsz(pt):
    """글자 크기 훅: 덱판은 FS_DECK 대응(최소 12 pt), 그 밖은 그대로."""
    return (FS_DECK.get(pt, max(12.0, 2.0 * pt)) if DECK else pt)


def msz(s, k=None):
    """산점 마커 면적(pt²) 배율 훅."""
    return s * (K if k is None else k) ** 2


def letter(ax, s, x, y):
    """패널 문자: 논문·슬라이드판은 S.panel_letter(8 pt 굵게, 크롭 등록), 덱판은 14 pt 굵게."""
    if DECK:
        ax.text(x, y, s, transform=ax.transAxes, fontsize=S.TOK["font"]["slide"]["panel_letter_pt"], fontweight="bold", va="bottom", ha="left")
        S.PANELS.setdefault(s, []).append(ax)
    else:
        S.panel_letter(ax, s, x=x, y=y)


# ---- 배치표(덱 mm: 캔버스 폭 304.8). 논문판은 같은 좌표를 0.5906 배 축에 그린다
DECK_W_IN, DECK_H_IN, DECK_DPI = 12.0, 4.65, 300   # 설계 지침 v9 1.4: 일반 쪽 12.0 × 4.65 in(3600 × 1395 px)
WD = DECK_W_IN * 25.4                     # 304.8
HB = 25.0                                 # b 띠 높이(덱 mm)
GAP_AB = 8.0                              # a 파선 사각형 아래 – b 위(Stefan 상자) 간격(덱 mm, 지침 v9 1.3: 패널 간격 ≥ 0.35 in = 8.9 mm 를 사각형 안쪽 1 mm 와 함께 채운다)
HA = DECK_H_IN * 25.4 - HB - GAP_AB       # a 높이(덱 mm) = 85.11
LETTER_Y = 0.943                          # 패널 문자 a(va=bottom)의 축 높이 비율: 덱 14 pt·논문 8 pt 모두 글자 위가 축 위(HA) 안
# a(10-15 v9 항 15): Input · Model · Output 범위를 옅은 파선 사각형 + 머리 라벨로 표시. 모델 상자 안에는 이름만('Stefan', 'Gradient boosting').
# 입력은 실자료 썸네일을 항목별로 나눠 둔다: 기후(그 해) 3 · 정적 공변량 4(소제목에 묶음별 특징 수), 관측 ALT(훈련 전용, 적색 파선).
GEO_A = dict(
    top_pad=6.5, ry0=1.0, rr=1.6,                                   # 영역 사각형: 위 = HA − top_pad, 아래 ry0, 모서리 반지름
    rx=dict(Input=(6.0, 110.0), Model=(113.0, 176.0), Output=(179.0, 303.5)),
    head_dx=1.8, head_dy=1.2,                                       # 머리 라벨(사각형 안 왼쪽 위)
    yU=59.5, yL=33.1, yt=9.5,                                       # 행 높이: 물리 · 학습 · 훈련 전용(머리 라벨·소제목·이름 줄 높이에서 계산)
    TH=14.0, in_right=104.0, gap_item=4.0, name_dy=1.0, sub_dy=1.0,  # 입력 썸네일 높이, 오른쪽 끝, 항목 간격, 이름·소제목 간격
    obs_lab_dx=2.0,                                                 # 관측 ALT 라벨(썸네일 왼쪽)
    bx=120.0, bw=48.0, bhP=12.0, bhL=14.0, f_dy=1.2,                # 모델 상자(물리 높이 · 학습 높이), 수식(물리 상자 아래)
    r_minus=3.0, y_dy=-1.4, tgt_dx=2.4, cross=1.8, yp_dx=2.0,       # 훈련 경로(⊖, y 라벨, 목표 라벨, 교차 끊김, ŷ_P 라벨)
    THo=15.0, ox=186.0, g_j2=9.0, g_px=21.0, r_op=3.2, g_gx=9.0, lam_dx=2.6, t1=1.0,   # 출력 썸네일·분기점·⊕·최종 썸네일
    cb_dy=5.4, cb_h=1.9, cb_gap=3.5, cb_w=2.4, key_dy=7.8, key_pitch=19.0, key_sq=2.2,  # 잔차 소형 색 막대, ALT 색 막대, 적용 범위 열쇠
    leg_anchor=(302.0, None),                                       # 범례: Output 사각형 오른쪽 위(y 는 머리 라벨 높이)
    letter=dict(x=0.0), ex_x=6.0,                                   # 패널 문자 · 예시 라벨(지역, 연도)
)
GEO_B = dict(cw=WD / 4, x0=0.0, yc=11.0, bw=24.0, bh=8.0, gw=20.0, gh=6.5, gap=3.5, in_dx=9.0, a_gap=1.4,
             out_len=7.0, out_lab=8.4, op_dx=12.0, r_op=2.7, lam_dy=1.2, o2_len=6.0, o2_lab=7.4, note_dx=1.6, name_y=2.4,
             letter=dict(x=0.0))
GEO_C_PAPER = dict(ysub=-0.30, ysub2=-0.47)          # 지도 아래 둘째·셋째 줄(축 비율)
GEO_C_DECK = dict(ysub=-0.33, ysub2=-0.485)
GEO_C = GEO_C_PAPER


# ================================================================ 자료
THUMB_FORCING = ["e5_maat", "e5_fdd", "e5_swe"]      # 그 해 기후 썸네일 열(TDD 는 pred_<year> 의 e5_tdd). 10-16: MAAT 추가(기후 4 · 정적 4 대칭)


def forcing_year_thumb(year=2014):
    """격자 연별 강제력(3.6 GB)에서 그 해의 MAAT·FDD·SWE 만 추려 캐시한다(maps/Alaska/forcing_<year>_thumb.csv.gz). 캐시에 열이 모자라면 다시 만든다."""
    cache = V5 / "maps" / "Alaska" / f"forcing_{year}_thumb.csv.gz"
    if cache.exists():
        d = pd.read_csv(cache, dtype={"cell_id": str})
        if all(c in d.columns for c in THUMB_FORCING):
            return d
    parts = []
    for ch in pd.read_csv(V5 / "grid" / "alaska_forcing_year.csv", usecols=["cell_id", "year"] + THUMB_FORCING, dtype={"cell_id": str}, chunksize=2_000_000):
        parts.append(ch[ch.year == year])
    d = pd.concat(parts, ignore_index=True); d.to_csv(cache, index=False)
    return d


def alaska_thumbs(year=2014):
    """그 해 알래스카 1 km 격자(예측 파일 셀)에 정적 공변량·그 해 강제력·AOA 를 붙여 0.05° × 0.02° 상자 평균 래스터로 만든다."""
    p = pd.read_csv(V5 / "maps" / "Alaska" / f"pred_{year}.csv.gz", dtype={"cell_id": str})
    g = pd.read_csv(PROC / "map_alaska" / "alaska_grid_x25_v1.csv.gz", dtype={"cell_id": str},
                    usecols=["cell_id", "lat", "lon", "dem_elev", "dem_slope", "sg_soc_0_5", "sg_clay_5_15", "pfr_mean", "land"])
    fy = forcing_year_thumb(year).rename(columns={"e5_maat": "maat", "e5_fdd": "fdd", "e5_swe": "swe"})[["cell_id", "maat", "fdd", "swe"]]
    a = pd.read_csv(V5 / "maps" / "Alaska" / "mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "aoa_in"])
    m = p.merge(g.drop(columns=["lat", "lon"]), on="cell_id", how="left").merge(fy, on="cell_id", how="left").merge(a, on="cell_id", how="left")
    m["resid"] = m.r1 - m.p1
    lon0, lon1 = np.floor(m.lon.min() * 10) / 10, np.ceil(m.lon.max() * 10) / 10
    lat0, lat1 = np.floor(m.lat.min() * 10) / 10, np.ceil(m.lat.max() * 10) / 10
    xe = np.arange(lon0, lon1 + 1e-9, 0.05); ye = np.arange(lat0, lat1 + 1e-9, 0.02)
    gl = g[g.land == 1]; cnt, _, _ = np.histogram2d(gl.lat, gl.lon, bins=[ye, xe])      # 육지 바탕 = 전체 격자의 육지 셀
    ras = {}
    for k in ("e5_tdd", "maat", "fdd", "swe", "dem_elev", "dem_slope", "sg_soc_0_5", "sg_clay_5_15", "pfr_mean", "p1", "r1", "resid", "aoa_in"):
        v = m[k].values.astype(float); ok = np.isfinite(v)
        s, _, _ = np.histogram2d(m.lat.values[ok], m.lon.values[ok], bins=[ye, xe], weights=v[ok])
        c, _, _ = np.histogram2d(m.lat.values[ok], m.lon.values[ok], bins=[ye, xe])
        with np.errstate(invalid="ignore", divide="ignore"):
            ras[k] = np.where(c > 0, s / c, np.nan)
    ras["aoa_bin"] = np.where(np.isfinite(ras["aoa_in"]), (ras["aoa_in"] >= 0.5).astype(float), np.nan)   # 상자 안 적용 범위 셀이 반 이상이면 '안'
    ext = [lon0, lon1, lat0, lat1]
    wh = (lon1 - lon0) / ((lat1 - lat0) * ALASKA_ASP)                 # 썸네일 가로/세로(위도 보정)
    return dict(ras=ras, mask=cnt > 0, extent=ext, vals=m, wh=wh)


def labels():
    d = pd.read_csv(V5 / "labels_cell_year_v5.csv"); return d[d.in_forcing == 1].copy()


def learner_spec():
    """학습기 설정(트리 수·깊이)은 실험 코드 testbed_v5.fit_learner 의 catboost 인자에서 읽는다(손으로 적지 않는다)."""
    src = inspect.getsource(TB.fit_learner)
    m = re.search(r"iterations=(\d+),\s*learning_rate=([\d.]+),\s*depth=(\d+)\)\s*if name == \"catboost\"", src)
    assert m, "testbed_v5.fit_learner 의 catboost 설정을 찾지 못함"
    return dict(iterations=int(m.group(1)), learning_rate=float(m.group(2)), depth=int(m.group(3)))


def oof(method, lam):
    f = RUN / "preds" / f"Lena__block__year_obs__F8TS__{method}__{'catboost' if method == 'R1' else '-'}__lam{lam}.csv.gz"
    p = pd.read_csv(f)
    return p.groupby("cell_year_id").agg(y=("y", "first"), pred=("pred", "mean"), block=("block", "first"))


def block_bootstrap(y, pA, pB, blocks, n=2000, seed=0):
    """짝지은 ΔRMSE(A − B)의 블록 부트스트랩 분포. polar.testbed_v5.boot_delta 와 같은 난수(RandomState(seed), 다항 가중)·같은 합산 경로."""
    y, pA, pB = (np.asarray(v, float) for v in (y, pA, pB))
    m = np.isfinite(y) & np.isfinite(pA) & np.isfinite(pB)
    y, pA, pB, b = y[m], pA[m], pB[m], np.asarray(blocks)[m]
    ub, inv = np.unique(b, return_inverse=True); nb = len(ub)
    eA, eB = (pA - y) ** 2, (pB - y) ** 2
    sA = np.bincount(inv, eA, nb); sB = np.bincount(inv, eB, nb); cnt = np.bincount(inv, minlength=nb).astype(float)
    rng = np.random.RandomState(seed)
    W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n).astype(float)
    dA = np.sqrt((W @ sA) / (W @ cnt)) - np.sqrt((W @ sB) / (W @ cnt))
    d0 = float(np.sqrt(sA.sum() / cnt.sum()) - np.sqrt(sB.sum() / cnt.sum()))
    return dA, d0


# ================================================================ 그리기 도구(덱 mm 좌표 축)
def ax_mm(fig, x, y, w, h, **kw):
    W, H = fig.get_size_inches() * 25.4
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def canvas(fig, x, y, w, h, xlim, ylim):
    ax = ax_mm(fig, x, y, w, h); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([]); ax.patch.set_visible(False)      # axis("off") 는 숨은 축의 tightbbox 가 남아 슬라이드 크롭이 넘친다
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def arrow(ax, p0, p1, color=INK, ls="-", head=True, lw=0.7, z=6):
    """흐름 선(가로·세로만). head=True 면 끝에 작은 채움 셰브런(뒤쪽 홈이 파인 화살 모양; 설계 지침 v9 1.3 '흐름 도식의 셰브런 허용',
    삼각 화살촉 대신). 셰브런 끝이 p1 에 닿고 선은 셰브런 홈까지만 그린다. 크기는 덱 mm(논문판은 같은 좌표계를 축소해 그린다)."""
    (x0, y0), (x1, y1) = p0, p1
    assert abs(x0 - x1) < 1e-9 or abs(y0 - y1) < 1e-9, f"사선 금지 {p0}→{p1}"
    L = float(np.hypot(x1 - x0, y1 - y0)); ux, uy = ((x1 - x0) / L, (y1 - y0) / L) if L > 0 else (1.0, 0.0)
    w, h = CHEV_W, CHEV_H; xe, ye = (x1 - ux * w * 0.55, y1 - uy * w * 0.55) if head else (x1, y1)
    ax.plot([x0, xe], [y0, ye], color=color, lw=sc(lw), ls=ls, zorder=z, solid_capstyle="butt", dash_capstyle="butt")
    if head:
        from matplotlib.patches import Polygon
        tip = 0.55 * w; px_, py_ = -uy, ux                                            # 진행 방향(ux, uy), 수직(px_, py_)
        loc = [(-w, h / 2), (-tip, h / 2), (0.0, 0.0), (-tip, -h / 2), (-w, -h / 2), (-w + tip * 0.75, 0.0)]   # 끝 = (0, 0), 뒤로 −w
        pts = [(x1 + a * ux + b * px_, y1 + a * uy + b * py_) for a, b in loc]
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=z + 0.1))


def dot(ax, x, y, z=7):
    ax.add_patch(Circle((x, y), 0.75, color=INK, zorder=z))


def block(ax, x, y, w, h, text=None, kind="learned", fs=7.0, r=1.2, z=4):
    fill, edge = (GRN, GRN_EDGE) if kind == "physics" else (YEL, YEL_EDGE)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fill, edgecolor=edge, lw=sc(0.7), zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fsz(fs), color=INK, zorder=z + 1, linespacing=1.15)


def trees(ax, x, y, w, h, n=5, lw=0.5, color=INK, z=6):
    """트리 앙상블 글리프: 이진 트리 n 그루(3단, 선만) 를 상자 (x, y, w, h) 안에 가로로 나란히."""
    pitch = w / n
    for i in range(n):
        cx = x + (i + 0.5) * pitch; tw = 0.9 * pitch
        y0, y1, y2 = y + h, y + h / 2, y
        for s in (-1, 1):
            x1 = cx + s * tw / 4
            ax.plot([cx, x1], [y0, y1], color=color, lw=sc(lw), zorder=z, solid_capstyle="round")
            for s2 in (-1, 1):
                ax.plot([x1, x1 + s2 * tw / 8], [y1, y2], color=color, lw=sc(lw), zorder=z, solid_capstyle="round")


def op_node(ax, x, y, r=2.0, op="+", z=7):
    """원 연산 노드: ⊕ 또는 ⊖."""
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, lw=sc(0.7), zorder=z))
    ax.plot([x - r * 0.55, x + r * 0.55], [y, y], color=INK, lw=sc(0.7), zorder=z + 1, solid_capstyle="butt")
    if op == "+":
        ax.plot([x, x], [y - r * 0.55, y + r * 0.55], color=INK, lw=sc(0.7), zorder=z + 1, solid_capstyle="butt")


def hbrace(ax, x0, x1, y, h, color=INK, lw=0.6, z=6):
    """가로 중괄호(대상 아래, 꼭지는 아래쪽): 양끝 (x0, y)·(x1, y), 몸통 y − h/2, 꼭지 (가운데, y − h)."""
    xm = (x0 + x1) / 2; q = h / 2; yb = y - h / 2
    verts = [(x0, y), (x0, yb), (x0, yb), (x0 + q, yb), (xm - q, yb), (xm, yb), (xm, yb), (xm, y - h),
             (xm, yb), (xm, yb), (xm + q, yb), (x1 - q, yb), (x1, yb), (x1, yb), (x1, y)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor="none", edgecolor=color, lw=sc(lw), zorder=z, capstyle="round"))


def thumb(ax, x, y, w, h, z=3):
    """썸네일 축(얇은 회색 틀, 눈금 없음)."""
    ia = ax.inset_axes([x, y, w, h], transform=ax.transData, zorder=z)
    ia.set_xticks([]); ia.set_yticks([])
    for sp in ia.spines.values():
        sp.set_visible(True); sp.set_edgecolor(FRAME); sp.set_linewidth(sc(0.5))
    ia.patch.set_facecolor("white")
    return ia


def label(ax, x, y, s, fs=7.0, color=INK, ha="center", va="top", **kw):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=fsz(fs), color=color, **kw)


def alaska_img(ia, A, key, cmap, norm, coast, land=()):
    for g in land:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.exterior.xy; ia.fill(xs, ys, facecolor=LAND, edgecolor="none", lw=0, zorder=0.5)
    ia.imshow(np.where(A["mask"], 1.0, np.nan), extent=A["extent"], origin="lower", cmap=ListedColormap([LAND]), vmin=0, vmax=1,
              aspect=ALASKA_ASP, interpolation="nearest", zorder=1)
    if key is not None:
        im = ia.imshow(A["ras"][key], extent=A["extent"], origin="lower", cmap=cmap, norm=norm, aspect=ALASKA_ASP, interpolation="nearest", zorder=2)
    else:
        im = None
    for g in coast:
        for part in getattr(g, "geoms", [g]):
            xs, ys = part.xy; ia.plot(xs, ys, color=COAST, lw=sc(0.3), zorder=3)
    ia.set_xlim(A["extent"][0], A["extent"][1]); ia.set_ylim(A["extent"][2], A["extent"][3]); ia.set_aspect(ALASKA_ASP)
    return im


def _fmt(v, _=None):
    return f"{v:g}".replace("-", "−")


def mini_cbar(ax, x, y, w, h, cmap, norm, ticks, unit):
    """가로 소형 컬러바(눈금 아래, 단위는 오른쪽)."""
    cax = ax.inset_axes([x, y, w, h], transform=ax.transData, zorder=5)
    cb = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(sc(0.4)); cb.outline.set_edgecolor(FRAME)
    cax.tick_params(labelsize=fsz(FS_TINY), width=sc(0.4), length=sc(1.5), pad=sc(1.0), colors=INK)
    cax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_fmt))
    if unit:
        ax.text(x + w + 1.2, y + h / 2, unit, ha="left", va="center", fontsize=fsz(FS_TINY), color=INK)
    return cax


def mini_vcbar(ax, x, y, w, h, cmap, norm, ticks, unit, side="right"):
    """세로 소형 컬러바(눈금 오른쪽 또는 왼쪽, 단위는 위)."""
    cax = ax.inset_axes([x, y, w, h], transform=ax.transData, zorder=5)
    cb = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical", ticks=ticks)
    cb.outline.set_linewidth(sc(0.4)); cb.outline.set_edgecolor(FRAME)
    if side == "left":
        cax.yaxis.set_ticks_position("left")
    cax.tick_params(labelsize=fsz(FS_TINY), width=sc(0.4), length=sc(1.5), pad=sc(1.0), colors=INK)
    cax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_fmt))
    ax.text(x + w / 2, y + h + 1.4, unit, ha="center", va="bottom", fontsize=fsz(FS_TINY), color=INK)
    return cax


# ================================================================ a 구조도(Input | Model | Output)
# 입력 썸네일(열, 이름, 색표, 백분위 범위). 기후 = 그 해 ERA5-Land(TDD 는 오른쪽 끝: Stefan 으로 가는 선), 정적 = 지형·토양.
# 단위는 쓰지 않는다(층마다 자기 백분위 범위로 늘린 예시 그림, 절대 눈금 없음). 영구동토 확률(pfr_mean)은 모델 특징(F8TS)이 아니라 넣지 않는다
CLIMATE = [("maat", "MAAT", cmc.lipari, (2, 98)), ("fdd", "FDD", cmc.devon_r, (2, 98)), ("swe", "Snow", cmc.oslo_r, (2, 98)),
           ("e5_tdd", "TDD", CM_TDD, (2, 98))]                 # 10-16: MAAT(연평균 기온, 그 해) 추가 → 기후 4 · 정적 4. 썸네일은 정적 행과 같은 열 가운데에 맞춘다
STATIC = [("dem_elev", "Elevation", cmc.lapaz_r, (2, 98)), ("dem_slope", "Slope", cmc.acton_r, (0, 98)),
          ("sg_soc_0_5", "Soil carbon", cmc.bamako_r, (2, 98)), ("sg_clay_5_15", "Clay", cmc.turku_r, (2, 98))]


def _tw(ax, s_, fs):
    """글자 폭(축 자료 단위 = 덱 mm). 렌더러로 잰다."""
    r = ax.figure.canvas.get_renderer(); t = ax.text(0, 0, s_, fontsize=fsz(fs)); wpx = t.get_window_extent(r).width; t.remove()
    return wpx / (ax.transData.transform((1.0, 0.0)) - ax.transData.transform((0.0, 0.0)))[0]


def _qa_overlap(fig, tag):
    """글자 상자끼리, 글자 상자와 썸네일(틀 있는 inset 축) 상자의 겹침을 렌더러 좌표로 검사해 출력한다(0.5 px 안쪽으로 줄여 맞닿음은 제외)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    axes = []
    def rec(a):
        axes.append(a); [rec(c) for c in a.child_axes]                        # inset 축(썸네일·소형 컬러바)은 fig.axes 에 없다
    for a in fig.axes:
        rec(a)
    texts, frames = [], []
    for ax in axes:
        ts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
        for axis in (ax.xaxis, ax.yaxis):                                     # 눈금 라벨은 한계 안의 눈금만(밖의 것은 그려지지 않는다)
            lo, hi = sorted(axis.get_view_interval())
            ts += [t for loc, t in zip(axis.get_majorticklocs(), axis.get_majorticklabels()) if lo - 1e-9 <= loc <= hi + 1e-9]
        for t in ts:
            if t.get_visible() and t.get_text().strip():
                texts.append((t.get_text().replace("\n", " / "), t.get_window_extent(r), ax))
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                texts.append((f"legend:{t.get_text()}", t.get_window_extent(r), ax))
        if ax.spines["left"].get_visible() and not ax.get_xticks().size and not ax.get_yticks().size:
            frames.append(("thumb", ax.get_window_extent(r), ax))
    bad = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if _ov(texts[i][1], texts[j][1]) > 0.5:
                bad.append(f"text×text: '{texts[i][0]}' ↔ '{texts[j][0]}'")
        for f in frames:
            if f[2] is not texts[i][2] and _ov(texts[i][1], f[1]) > 0.5:     # 자기 축 안의 글자(경위선·축척)는 제외
                bad.append(f"text×thumb: '{texts[i][0]}'")
    print(f"[QA {tag}] texts {len(texts)}, thumbs {len(frames)}, overlaps {len(bad)}" + ("".join("\n   " + b for b in bad) if bad else ""))
    return bad


def _ov(a, b):
    """두 상자가 겹치는 폭·높이 중 작은 쪽(px). 겹치지 않으면 0."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0); h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return min(w, h) if (w > 0 and h > 0) else 0.0


def panel_a(fig, A, lab, meta, spec, x0, y0, w, h):
    """축 좌표 = 덱 mm(왼쪽 아래 0,0; 폭 WD, 높이 HA). 논문판은 같은 좌표를 작은 축에 그린다. 행: 위 = 물리(yU), 가운데 = 학습(yL), 아래 = 훈련 전용(yt).
    Input · Model · Output 의 범위는 옅은 파선 둥근 사각형과 머리 라벨. 모델 상자 안에는 이름만, Stefan 식은 상자 아래 mathtext."""
    G = GEO_A
    ax = canvas(fig, x0, y0, w, h, (0, WD), (0, HA)); S.register("a", ax)
    ne = MV.ne_layers("alaska"); coast = ne["coastline"]; land = ne["land"]
    yr = int(meta["year"]); m = A["vals"]; wh = A["wh"]
    lo, hi = MV.pct_range(np.r_[m.p1.values, m.r1.values]); nALT = Normalize(lo, hi)
    v = MV.sym_max(m.resid.values); nRES = TwoSlopeNorm(0.0, -v, v)
    o = lab[(lab.region == "Alaska") & (lab.year == yr)]
    feats = list(meta["feats"]); n_feat = len(feats); assert n_feat == len(TB.FEATSETS["F8TS"]), "meta.feats 와 FEATSETS['F8TS'] 길이가 다름"
    n_clim = sum(f in TB.F8 for f in feats); n_stat = n_feat - n_clim                    # 기후 8(그 해) · 정적 15(지형 6 + 토양 9)
    hw = float(meta["interval"]["half_width_90"]); aoa_frac = float(meta["aoa"]["frac_inside"])
    ytop = HA - G["top_pad"]; yU, yL, yt = G["yU"], G["yL"], G["yt"]
    # ---- 영역 사각형(옅은 파선) + 머리 라벨(사각형 안 왼쪽 위, 작은 회색)
    for nm, (xa_, xb_) in G["rx"].items():
        ax.add_patch(FancyBboxPatch((xa_, G["ry0"]), xb_ - xa_, ytop - G["ry0"], boxstyle=f"round,pad=0,rounding_size={G['rr']}",
                                    facecolor="none", edgecolor=REGION_EDGE, lw=sc(0.6), ls=(0, (3.0, 2.0)), zorder=1))
        label(ax, xa_ + G["head_dx"], ytop - G["head_dy"], nm, fs=FS_TINY, color=AUX, ha="left", va="top")
    # ---- Input: 기후(그 해) 행 · 정적 공변량 행(오른쪽 맞춤, 항목 폭 = max(썸네일, 이름)), 관측 ALT(훈련 전용)
    TH = G["TH"]; TW = TH * wh; xr = G["in_right"]; rows_src = {}

    def slots_of(items):
        """오른쪽 맞춤 칸(항목 폭 = max(썸네일, 이름)) 의 가운데 x 목록."""
        widths = [max(TW, _tw(ax, nm, FS_TINY)) for _, nm, *_ in items]
        x = xr; slots = []
        for wd in reversed(widths):
            slots.append((x - wd, x)); x -= wd + G["gap_item"]
        return [(a + b) / 2 for a, b in slots[::-1]]

    def row(items, yc, head, centres=None):
        cxs_in = slots_of(items) if centres is None else centres; cxs = []
        for (key, nm, cm, pct), cx in zip(items, cxs_in):
            cxs.append(cx)
            ia = thumb(ax, cx - TW / 2, yc - TH / 2, TW, TH)
            a_ = A["ras"][key]; alaska_img(ia, A, key, cm, Normalize(*np.nanpercentile(a_, pct)), coast, land)
            label(ax, cx, yc - TH / 2 - G["name_dy"], nm, fs=FS_TINY, va="top")
        label(ax, cxs[0] - TW / 2, yc + TH / 2 + G["sub_dy"], head, fs=FS_TINY, color=AUX, ha="left", va="bottom")
        rows_src[head] = [nm for _, nm, *_ in items]
        return cxs
    cs = slots_of(STATIC); assert len(CLIMATE) == len(STATIC)                 # 기후 썸네일 = 정적 썸네일과 같은 열(같은 크기·간격·세로 정렬)
    cx_cl = row(CLIMATE, yU, f"Climate, {yr} ({n_clim})", centres=cs)
    row(STATIC, yL, f"Static ({n_stat})", centres=cs)
    x_tdd_r = cx_cl[-1] + TW / 2                                              # TDD 썸네일 오른쪽 끝 → Stefan
    ia = thumb(ax, xr - TW, yt - TH / 2, TW, TH); alaska_img(ia, A, None, None, None, coast, land)
    ia.scatter(o.lon, o.lat, s=msz(2.0), color=TRAIN_RED, lw=0, zorder=4, rasterized=True)
    label(ax, xr - TW - G["obs_lab_dx"], yt, "Observed ALT", fs=FS_TINY, ha="right", va="center")
    # ---- Model: 이름만 든 상자 두 개, Stefan 식은 상자 아래(mathtext)
    bx, bw = G["bx"], G["bw"]
    block(ax, bx, yU - G["bhP"] / 2, bw, G["bhP"], "Stefan", kind="physics")
    label(ax, bx + bw / 2, yU - G["bhP"] / 2 - G["f_dy"], r"ALT = $E\sqrt{\mathrm{TDD}}$", fs=FS_TINY, color=AUX)
    block(ax, bx, yL - G["bhL"] / 2, bw, G["bhL"], "Gradient boosting", kind="learned")
    arrow(ax, (x_tdd_r, yU), (bx, yU))                                       # TDD → Stefan
    arrow(ax, (G["rx"]["Input"][1], yL), (bx, yL))                           # 입력 특징 전체(기후 + 정적) → 학습
    # ---- Output: 물리 ALT · 잔차 → ⊕(λ) → ALT 지도, 적용 범위(잔차 학습기의 부산물)
    THo = G["THo"]; TWo = THo * wh; ox = G["ox"]
    arrow(ax, (bx + bw, yU), (ox, yU)); arrow(ax, (bx + bw, yL), (ox, yL))
    ia = thumb(ax, ox, yU - THo / 2, TWo, THo); alaska_img(ia, A, "p1", MV.CM_ALT, nALT, coast, land)
    label(ax, ox + TWo / 2, yU - THo / 2 - G["t1"], "Physics ALT", fs=FS_TINY)
    ia = thumb(ax, ox, yL - THo / 2, TWo, THo); alaska_img(ia, A, "resid", MV.CM_DIFF, nRES, coast, land)
    label(ax, ox + TWo / 2, yL - THo / 2 - G["t1"], "Residual (cm)", fs=FS_TINY)
    mini_cbar(ax, ox - 1.5, yL - THo / 2 - G["t1"] - G["cb_dy"] - G["cb_h"], TWo + 3.0, G["cb_h"], MV.CM_DIFF, nRES, [-v, 0, v], None)
    xj2 = ox + TWo + G["g_j2"]; px = xj2 + G["g_px"]; r = G["r_op"]
    arrow(ax, (ox + TWo, yU), (px - r, yU)); dot(ax, xj2, yU)                 # 분기점: ŷ_P → 훈련 목표
    arrow(ax, (ox + TWo, yL), (px, yL), head=False); arrow(ax, (px, yL), (px, yU - r)); dot(ax, px, yL)
    label(ax, px - G["lam_dx"], (yU + yL) / 2, f"$\\lambda$ = {meta['lam']:g}", ha="right", va="center", fs=FS_SMALL)
    op_node(ax, px, yU, r, "+")
    gx = px + r + G["g_gx"]
    arrow(ax, (px + r, yU), (gx, yU))
    ia = thumb(ax, gx, yU - THo / 2, TWo, THo); alaska_img(ia, A, "r1", MV.CM_ALT, nALT, coast, land)
    label(ax, gx + TWo / 2, yU - THo / 2 - G["t1"], f"ALT {yr}", fs=FS_TINY)
    tk = MV.ticks_between(lo, hi); tk = tk[::2] if len(tk) > 3 else tk
    mini_vcbar(ax, gx + TWo + G["cb_gap"], yU - THo / 2, G["cb_w"], THo, MV.CM_ALT, nALT, tk, "cm")
    arrow(ax, (px, yL), (gx, yL))
    ia = thumb(ax, gx, yL - THo / 2, TWo, THo)
    alaska_img(ia, A, "aoa_bin", ListedColormap([AOA_OUT, AOA_IN]), Normalize(0, 1), coast, land)
    label(ax, gx + TWo / 2, yL - THo / 2 - G["t1"], "Applicability", fs=FS_TINY)
    yk = yL - THo / 2 - G["t1"] - G["key_dy"]; sq = G["key_sq"]
    for i, (c_, s_) in enumerate(((AOA_IN, "inside"), (AOA_OUT, "outside"))):
        xk = gx + TWo / 2 - G["key_pitch"] * 0.95 + i * G["key_pitch"]
        ax.add_patch(Rectangle((xk, yk - sq / 2), sq, sq, facecolor=c_, edgecolor=FRAME, lw=sc(0.3), zorder=5))
        label(ax, xk + sq + 1.0, yk, s_, fs=FS_TINY, color=AUX, ha="left", va="center")
    # ---- 훈련 전용 경로: 물리 ŷ_P(실선 분기, 교차 끊김) 와 관측 y(적색 파선) → ⊖ → 목표 y − ŷ_P → 학습 상자 아래
    rm = G["r_minus"]; xm = bx + bw / 2; gap = G["cross"]
    arrow(ax, (xj2, yU), (xj2, yL + gap), head=False); arrow(ax, (xj2, yL - gap), (xj2, yt), head=False); arrow(ax, (xj2, yt), (xm + rm, yt))
    label(ax, xj2 + G["yp_dx"], (yL - gap + yt) / 2, r"$\hat{y}_{\mathrm{P}}$", ha="left", va="center", fs=FS_SMALL)
    op_node(ax, xm, yt, rm, "−")
    arrow(ax, (xr, yt), (xm - rm, yt), color=TRAIN_RED, ls=DASH)
    label(ax, (G["rx"]["Input"][1] + xm - rm) / 2 + 2.0, yt + G["y_dy"], "$y$", color=TRAIN_RED, fs=FS_SMALL, va="top")
    arrow(ax, (xm, yt + rm), (xm, yL - G["bhL"] / 2), color=TRAIN_RED, ls=DASH)
    label(ax, xm + G["tgt_dx"], (yt + rm + yL - G["bhL"] / 2) / 2, r"$y-\hat{y}_{\mathrm{P}}$", ha="left", va="center", color=TRAIN_RED, fs=FS_SMALL)
    # ---- 범례(Output 사각형 오른쪽 위) · 패널 문자 · 예시 라벨
    hs = [Line2D([], [], color=INK, lw=sc(0.7)), Line2D([], [], color=TRAIN_RED, lw=sc(0.7), ls=DASH)]
    ax.legend(hs, ["data flow", "training only"], loc="upper right", bbox_to_anchor=(G["leg_anchor"][0], ytop - 0.6), bbox_transform=ax.transData,
              frameon=False, fontsize=fsz(FS_TINY), handlelength=1.8, handletextpad=0.5, borderaxespad=0.0, borderpad=0.0, labelspacing=0.35)
    letter(ax, "a", x=G["letter"]["x"], y=LETTER_Y)
    label(ax, G["ex_x"], LETTER_Y * HA, f"Alaska, {yr}", fs=FS_TINY, color=AUX, ha="left", va="bottom")   # a 의 예시 지역·연도(c·d 는 레나)
    SRC["a"] = dict(year=yr, E=meta["E"], lam=meta["lam"], n_train=meta["n_train"], n_features=n_feat, n_climate_features=n_clim, n_static_features=n_stat,
                    features=meta["feats"], learner=dict(name=meta["learner"], **spec), n_cells_map=int(len(m)), alt_range_cm=[lo, hi], resid_sym_max_cm=v,
                    interval_half_width_90_cm=hw, aoa_frac_inside=aoa_frac, n_observed_cells=int(o.loc_id.nunique()), n_observed_rows=int(len(o)),
                    thumb_bins_deg=[0.05, 0.02], input_rows=rows_src, input_examples={k: [c[0] for c in CLIMATE + STATIC] for k in ("columns",)},
                    input_scales="each thumbnail stretched to its own percentile range (no colour bar, no units)",
                    regions="Input / Model / Output = light dashed rounded rectangles with head labels (deck mm x ranges in GEO_A.rx)",
                    model_boxes="names only: 'Stefan', 'Gradient boosting'; Stefan formula below the box in mathtext",
                    connectors="thin lines with small filled chevrons (no triangular arrowheads)", example_label=f"Alaska, {yr}",
                    pfr_mean="not a model feature (F8TS); removed from the input panel 2026-10-15",
                    forcing_year_layers="TDD · FDD · SWE of the map year (grid/alaska_forcing_year.csv)", thumb_aspect_w_over_h=round(A["wh"], 4))
    return ax


# ================================================================ b 비교 구조 4종(같은 문법: 학습 = 'ML' 황 상자, 물리 = 'Stefan' 녹 상자)
def panel_b(fig, x0, y0, w, h):
    B = GEO_B
    ax = canvas(fig, x0, y0, w, h, (0, WD), (0, HB)); S.register("b", ax)
    names = ["Direct ML", "Physics-input ML", "Physics pseudo-labels", "Physics + residual ML"]
    cw, yc, bw, bh, gw, gh = B["cw"], B["yc"], B["bw"], B["bh"], B["gw"], B["gh"]; gy = yc + bh / 2 + B["gap"]
    for i, nm in enumerate(names):
        gx = B["x0"] + i * cw
        xin, xb = gx + B["in_dx"], gx + B["in_dx"] + B["a_gap"] + 6.0
        label(ax, xin, yc, "$X$", ha="right", va="center")
        arrow(ax, (xin + B["a_gap"], yc), (xb, yc))
        block(ax, xb, yc - bh / 2, bw, bh, "ML", kind="learned", fs=FS_SMALL, r=1.0)
        xe = xb + bw; cx = xb + bw / 2
        if i == 3:
            px = xe + B["op_dx"]; r = B["r_op"]
            arrow(ax, (xe, yc), (px - r, yc)); label(ax, (xe + px - r) / 2, yc + B["lam_dy"], "$\\lambda$", va="bottom", fs=FS_SMALL)
            op_node(ax, px, yc, r, "+")
            block(ax, px - gw / 2, gy, gw, gh, "Stefan", kind="physics", fs=FS_SMALL, r=1.0)
            arrow(ax, (px, gy), (px, yc + r))
            label(ax, px - gw / 2 - B["note_dx"], gy + gh / 2, "anchor", ha="right", va="center", fs=FS_TINY, color=AUX)   # 왼쪽: 오른쪽은 캔버스 끝
            arrow(ax, (px + r, yc), (px + r + B["o2_len"], yc)); label(ax, px + r + B["o2_lab"], yc, r"$\hat{y}$", ha="left", va="center")
        else:
            arrow(ax, (xe, yc), (xe + B["out_len"], yc)); label(ax, xe + B["out_lab"], yc, r"$\hat{y}$", ha="left", va="center")
            if i >= 1:
                block(ax, cx - gw / 2, gy, gw, gh, "Stefan", kind="physics", fs=FS_SMALL, r=1.0)
                if i == 1:
                    arrow(ax, (cx, gy), (cx, yc + bh / 2))                                       # 물리 출력 = 입력 특징
                    label(ax, cx + gw / 2 + B["note_dx"], gy + gh / 2, "input", ha="left", va="center", fs=FS_TINY, color=AUX)
                else:
                    arrow(ax, (cx, gy), (cx, yc + bh / 2), color=TRAIN_RED, ls=DASH)               # 물리 값 = 추가 라벨(학습 때만)
                    label(ax, cx + gw / 2 + B["note_dx"], gy + gh / 2, "labels", ha="left", va="center", fs=FS_TINY, color=TRAIN_RED)
        label(ax, gx + cw / 2, B["name_y"], nm, va="center", fs=FS_SMALL)
    letter(ax, "b", x=B["letter"]["x"], y=(HB - 4.8) / HB)
    return ax


# ================================================================ c 다섯 분할(레나 델타)
LENA_EXT = (123.5, 130.4, 71.35, 72.95)


def map_th(tw):
    """레나 소지도 높이(mm): 폭 tw 와 범위·위도 보정 비율로 정한다."""
    asp = 1.0 / np.cos(np.deg2rad(LENA_LAT)); ext = LENA_EXT
    return tw * (ext[3] - ext[2]) * asp / (ext[1] - ext[0])


def panel_c(fig, lab, x0, y0, tw, gap):
    C = GEO_C; ysub, ysub2 = C["ysub"], C["ysub2"]                               # 둘째·셋째 줄(범례·연도 컬러바·설명) 위치, 축 비율
    q = lab[lab.region == "Lena"].reset_index(drop=True)
    f_rand = np.random.RandomState(0).permutation(len(q)) % 5                   # testbed_v5.folds_random 과 같은 규칙(행 단위)
    f_blk = TB._balanced_group_folds(q.block.values, 5, 0)                       # testbed_v5.folds_block
    yrs = sorted(int(y) for y in q.year.unique()); y_scored = sorted(int(f.name[1:]) for f in TB.folds_year(q))   # 연도 홀드아웃: 시험 행 ≥ 20 인 해만 채점
    nYR = Normalize(min(y_scored), max(y_scored))                               # 색 눈금 = 채점되는 해의 범위
    # 0.5° 블록 사각형(블록 id → 왼쪽 아래 모서리), 테두리 = 블록 겹 색
    bq = q.assign(f=f_blk).groupby("block").agg(lon=("lon", "min"), lat=("lat", "min"), f=("f", "first"))
    bq["x"] = np.floor(bq.lon / 0.5) * 0.5; bq["y"] = np.floor(bq.lat / 0.5) * 0.5
    assert (np.floor(q.lon / 0.5) * 0.5 == q.block.map(bq.x)).all() and (np.floor(q.lat / 0.5) * 0.5 == q.block.map(bq.y)).all(), "블록이 0.5° 격자와 어긋남"
    schemes = [("Random", "fold", f_rand), ("Spatial block 0.5°", "fold", f_blk), ("Year hold-out", "year", None), ("Block × year", "year", f_blk), ("Region hold-out", "region", None)]
    ne = MV.ne_layers("lena"); asp = 1.0 / np.cos(np.deg2rad(LENA_LAT))
    ext = LENA_EXT; th = map_th(tw)
    axes = []
    for i, (nm, mode, f_) in enumerate(schemes):
        ax = ax_mm(fig, x0 + i * (tw + gap), y0, tw, th); axes.append(ax)
        for g in ne["land"]:
            for part in getattr(g, "geoms", [g]):
                xs, ys = part.exterior.xy; ax.fill(xs, ys, facecolor=LAND, edgecolor="none", lw=0, zorder=1)
        for g in ne["coastline"]:
            for part in getattr(g, "geoms", [g]):
                xs, ys = part.xy; ax.plot(xs, ys, color=COAST, lw=sc(0.3), zorder=2)
        for gl in (124, 126, 128, 130):
            ax.plot([gl, gl], [ext[2], ext[3]], color=GRAT, lw=sc(0.3), zorder=1.5)
        for gl in (72, 73):
            ax.plot([ext[0], ext[1]], [gl, gl], color=GRAT, lw=sc(0.3), zorder=1.5)
        if mode == "fold" and f_ is f_blk or nm == "Block × year":            # 블록 겹: 0.5° 사각형 테두리 = 겹 색
            for _, b in bq.iterrows():
                ax.add_patch(Rectangle((b.x, b.y), 0.5, 0.5, facecolor="none", edgecolor=FOLD_HEX[int(b.f)], lw=sc(0.5), zorder=3))
        if mode == "fold":
            for k in range(5):
                qq = q[f_ == k]
                ax.scatter(qq.lon, qq.lat, s=msz(1.6, KM), color=FOLD_HEX[k], lw=0, zorder=4, rasterized=True)
        elif mode == "year":                                                    # 점 = 연도(순차), 채점 안 되는 해(행 < 20)는 중립색
            qs = q.sort_values("year"); sc_ = qs.year.isin(y_scored).values
            ax.scatter(qs.lon[~sc_], qs.lat[~sc_], s=msz(1.6, KM), color=NODATA, lw=0, zorder=3.5, rasterized=True)
            ax.scatter(qs.lon[sc_], qs.lat[sc_], s=msz(1.6, KM), c=qs.year[sc_], cmap=CM_YEAR, norm=nYR, lw=0, zorder=4, rasterized=True)
        else:                                                                   # 지역 홀드아웃: 레나 전부 시험, 학습은 다른 두 지역
            ax.scatter(q.lon, q.lat, s=msz(1.6, KM), color=INK, lw=0, zorder=4, rasterized=True)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect(asp); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(FRAME); sp.set_linewidth(sc(0.5))
        ax.text(0.5, -0.06, nm, transform=ax.transAxes, ha="center", va="top", fontsize=fsz(7.0), color=INK)
        if i == 0:   # 경위선 라벨 · 축척(첫 지도; 다섯 지도는 같은 범위)
            for gl in (124, 128):
                ax.text(gl, ext[2] + 0.02, f"{gl}°E", ha="center", va="bottom", fontsize=fsz(FS_TINY), color=AUX, zorder=5,
                        path_effects=[pe.withStroke(linewidth=sc(1.2), foreground="white")])
            ax.text(ext[0] + 0.05, 72.0, "72°N", ha="left", va="bottom", fontsize=fsz(FS_TINY), color=AUX, zorder=5,
                    path_effects=[pe.withStroke(linewidth=sc(1.2), foreground="white")])
            km = 50.0; dlon = km / (111.32 * np.cos(np.deg2rad(72.85)))
            ax.plot([123.75, 123.75 + dlon], [72.85, 72.85], color=INK, lw=sc(1.0), zorder=6, solid_capstyle="butt")
            ax.text(123.75 + dlon / 2, 72.80, f"{km:g} km", ha="center", va="top", fontsize=fsz(FS_TINY), color=INK, zorder=6,
                    path_effects=[pe.withStroke(linewidth=sc(1.2), foreground="white")])
            letter(ax, "c", x=-0.02, y=1.04)
            # 겹 범례(점 색 = 채점 겹 1–5; 무작위·공간 블록·블록 × 연도 테두리 공통)
            ax.text(0.0, ysub, "fold", transform=ax.transAxes, ha="left", va="center", fontsize=fsz(FS_SMALL), color=INK)
            for k in range(5):
                xk = 0.26 + k * 0.16
                ax.scatter([xk], [ysub], s=msz(9, KM), color=FOLD_HEX[k], lw=0, transform=ax.transAxes, clip_on=False, zorder=5)
                ax.text(xk + 0.045, ysub, str(k + 1), transform=ax.transAxes, ha="left", va="center", fontsize=fsz(FS_SMALL), color=INK)
        else:
            S.register("c", ax)
        if nm == "Year hold-out":   # 연도 색 눈금(순차)
            cax = ax.inset_axes([0.27, ysub - 0.045, 0.46, 0.09], transform=ax.transAxes, zorder=5)
            cb = fig.colorbar(ScalarMappable(norm=nYR, cmap=CM_YEAR), cax=cax, orientation="horizontal", ticks=[])
            cb.outline.set_linewidth(sc(0.4)); cb.outline.set_edgecolor(FRAME); S.register("c", cax)
            ax.text(0.25, ysub, str(min(y_scored)), transform=ax.transAxes, ha="right", va="center", fontsize=fsz(FS_TINY), color=INK)
            ax.text(0.75, ysub, str(max(y_scored)), transform=ax.transAxes, ha="left", va="center", fontsize=fsz(FS_TINY), color=INK)
        if nm == "Block × year":
            ax.text(0.5, ysub, "box = fold · dot = year", transform=ax.transAxes, ha="center", va="center", fontsize=fsz(FS_SMALL), color=AUX)
        if nm == "Region hold-out":
            ax.text(0.5, ysub, "test: Lena Delta", transform=ax.transAxes, ha="center", va="center", fontsize=fsz(FS_SMALL), color=AUX)
            ax.text(0.5, ysub2, "train: Alaska · Canada", transform=ax.transAxes, ha="center", va="center", fontsize=fsz(FS_SMALL), color=AUX)
    SRC["c"] = dict(region="Lena", n_cell_years=int(len(q)), n_cells=int(q.loc_id.nunique()), n_blocks=int(q.block.nunique()), years=yrs,
                    years_scored_year_holdout=y_scored, k_folds=5, seed=0, fold_palette=FOLD_HEX, fold_palette_name="Okabe–Ito",
                    year_cmap="cmcrameri bilbao_r 0.25–0.95, norm = scored years",
                    block_year_encoding="point colour = year, 0.5° box edge = block fold", region_holdout="all Lena cells scored (test); training = Alaska + Canada")
    return axes, th


# ================================================================ d 블록 부트스트랩 ΔRMSE
def panel_d(fig, x0, y0, w, h, letter_x=-0.28, ylab="Block resamples", letter_y=1.04, xlab="ΔRMSE (cm)\nPhysics + residual ML − Stefan (recalibrated)"):
    rP = oof("P1", "-1"); rR = oof("R1", "0.5")
    j = rP.join(rR, lsuffix="_p", rsuffix="_r", how="inner")
    assert np.allclose(j.y_p, j.y_r), "같은 cell_year_id 의 관측값이 다르다"
    d, d0 = block_bootstrap(j.y_r.values, j.pred_r.values, j.pred_p.values, j.block_p.values, 2000, 0)
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    bd = TB.boot_delta(j.y_r.values, j.pred_r.values, j.pred_p.values, j.block_p.values, 2000, 0)      # 같은 경로의 정본 함수
    assert abs(bd["lo"] - lo) < 1e-9 and abs(bd["hi"] - hi) < 1e-9 and abs(bd["delta"] - d0) < 1e-9, "boot_delta 와 불일치"
    ref = pd.read_csv(RUN / "deltas_vs_ref.csv"); ref = ref[(ref.region == "Lena") & (ref.scheme == "block") & (ref.forcing == "year_obs") & (ref.method == "R1") & (ref.lam == 0.5) & (ref.ref == "P1")]
    rd, rl, rh = (float(ref[c].iloc[0]) for c in ("delta", "lo", "hi"))
    match = bool(abs(rd - d0) < 1e-6 and abs(rl - lo) < 1e-6 and abs(rh - hi) < 1e-6)
    print(f"[F3d] Lena block ΔRMSE R1−P1: point {d0:+.3f} cm, 95 % [{lo:+.3f}, {hi:+.3f}] (n blocks {bd['n_blocks']}, n {bd['n']}); "
          f"deltas_vs_ref: {rd:+.3f} [{rl:+.3f}, {rh:+.3f}] → {'일치' if match else '불일치'}")
    SRC["d"] = dict(region="Lena", scheme="block", forcing="year_obs", featset="F8TS", pair="R1(lam 0.5) − P1", n_boot=2000, seed=0, rng="numpy RandomState(0) multinomial (testbed_v5.boot_delta)",
                    n_blocks=int(bd["n_blocks"]), n=int(bd["n"]), delta_cm=round(d0, 3), ci95_cm=[round(lo, 3), round(hi, 3)],
                    deltas_vs_ref_cm=[round(rd, 3), round(rl, 3), round(rh, 3)], matches_deltas_vs_ref=match)
    ax = ax_mm(fig, x0, y0, w, h); S.register("d", ax)
    bins = np.linspace(np.floor(d.min()), np.ceil(d.max()), 31)
    n, _, _ = ax.hist(d, bins=bins, color=S.METHOD_HEX["R1"], alpha=0.85, lw=0)
    ax.axvline(0, color=ZERO, lw=sc(0.6), zorder=3)
    top = n.max() * 1.10; t = n.max() * 0.03
    ax.plot([lo, hi], [top, top], color=INK, lw=sc(0.7), solid_capstyle="butt"); ax.plot([lo, lo], [top - t, top + t], color=INK, lw=sc(0.7)); ax.plot([hi, hi], [top - t, top + t], color=INK, lw=sc(0.7))
    ax.plot([d0, d0], [top - t * 1.6, top + t * 1.6], color=INK, lw=sc(0.7))
    ax.text((lo + hi) / 2, top + t * 2.0, "95 %", ha="center", va="bottom", fontsize=fsz(7.0), color=INK)
    ax.set_ylim(0, n.max() * 1.32); ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    x_lo = np.floor(d.min() / 5) * 5
    ax.set_xlim(x_lo - 0.05 * (1.0 - x_lo), 1.0); ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(5))   # 판 공통 눈금. 왼쪽 5 % 여유: '−15' 와 y 눈금 '0' 이 모서리에서 붙지 않게
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(_fmt))
    letter(ax, "d", x=letter_x, y=letter_y)
    return ax


# ================================================================ 조립(논문·슬라이드판)
PAPER_U = S.W2 / WD                       # 덱 mm → 논문 mm 배율(0.5906)


def f3():
    W = S.W2; ha, hb = HA * PAPER_U, HB * PAPER_U          # 58.8 · 17.7 mm
    y_c = 11.0; th = map_th(22.0); y_b = y_c + th + 9.5; y_a = y_b + hb + GAP_AB * PAPER_U; H = y_a + ha
    fig = S.figure(W, H)
    A = alaska_thumbs(2014); lab = labels(); meta = json.loads((V5 / "maps" / "Alaska" / "meta.json").read_text()); meta["year"] = 2014
    spec = learner_spec()
    panel_a(fig, A, lab, meta, spec, 0.0, y_a, W, ha)
    panel_b(fig, 0.0, y_b, W, hb)
    axes_c, th = panel_c(fig, lab, 4.0, y_c, 22.0, 2.6)
    panel_d(fig, 140.0, y_c, 32.0, th, letter_x=-0.42, xlab="ΔRMSE (cm)\nPhysics + residual ML\n− Stefan (recalibrated)")   # 논문: 세 줄(폭 180 mm 안)
    S.register("ab", *S.PANELS["a"], *S.PANELS["b"]); S.register("cd", *S.PANELS["c"], *S.PANELS["d"])
    SRC["qa_overlaps"] = _qa_overlap(fig, "paper")
    (OUT / "source_data").mkdir(parents=True, exist_ok=True); (OUT / "source_data" / f"{STEM}.json").write_text(json.dumps(SRC, indent=1, ensure_ascii=False))
    return S.export(fig, STEM, outdir=OUT)


# ================================================================ 덱판(슬라이드 본문 12.0 × 5.10 in, 300 dpi, 여백 없이 저장)
def deck_setup():
    """덱판 rc: 토큰 font.slide·lines.slide(축 라벨 14, 눈금 12, 범례 13 pt; 축선 1.0, 자료선 2.2 pt). 배율을 덱용으로 바꾼다."""
    global DECK, K, KM, GEO_C
    S.setup(scale=1.69)
    fs = S.TOK["font"]["slide"]; ln = S.TOK["lines"]["slide"]
    plt.rcParams.update({"font.size": fs["axis_label_pt"], "axes.labelsize": fs["axis_label_pt"], "xtick.labelsize": fs["tick_pt"], "ytick.labelsize": fs["tick_pt"],
                         "legend.fontsize": fs["direct_label_pt"], "axes.linewidth": ln["axis_pt"], "xtick.major.width": ln["tick_pt"], "ytick.major.width": ln["tick_pt"],
                         "xtick.major.size": ln["tick_len_pt"], "ytick.major.size": ln["tick_len_pt"], "lines.linewidth": ln["data_pt"], "lines.markersize": ln["marker_pt"]})
    DECK = True; K = 1.69; KM = DECK_C_TW / 22.0
    GEO_C = GEO_C_DECK


DECK_C_TW, DECK_C_GAP = 48.5, 7.0          # M02 소지도 폭·간격(mm). 4.65 in 캔버스에서 d 와 세로 여백을 두려고 2.0 in → 48.5 mm


def deck_save(fig, slide):
    """여백 없이 300 dpi 저장 → RGB 로 바꾸고 픽셀 크기를 확인한다."""
    from PIL import Image
    dd = OUT / "deck"; dd.mkdir(parents=True, exist_ok=True)
    p = dd / f"{STEM}__deck_{slide}.png"
    fig.savefig(p, dpi=DECK_DPI, facecolor="white"); plt.close(fig)
    with Image.open(p) as im:
        im.convert("RGB").save(p, dpi=(DECK_DPI, DECK_DPI))
    with Image.open(p) as im:
        want = (round(DECK_W_IN * DECK_DPI), round(fig.get_figheight() * DECK_DPI))
        assert im.size == want and im.mode == "RGB", f"{p.name}: {im.size} {im.mode} ≠ {want} RGB"
        print(f"[deck] {p}  {im.size[0]} × {im.size[1]} px {im.mode}")
    return p


def deck():
    deck_setup()
    Wmm, Hmm = WD, DECK_H_IN * 25.4
    A = alaska_thumbs(2014); lab = labels(); meta = json.loads((V5 / "maps" / "Alaska" / "meta.json").read_text()); meta["year"] = 2014
    spec = learner_spec()
    out = []
    # M01: a(구조도, 전체 폭) + b(글리프 띠, 아래)
    fig = S.figure(Wmm, Hmm); fig.set_size_inches(DECK_W_IN, DECK_H_IN)      # mm → in 왕복 오차로 3599 px 가 되지 않게 인치로 고정
    panel_a(fig, A, lab, meta, spec, 0.0, HB + GAP_AB, Wmm, HA)
    panel_b(fig, 0.0, 0.0, Wmm, HB)
    SRC["qa_overlaps_M01"] = _qa_overlap(fig, "deck M01")
    out.append(deck_save(fig, "M01"))
    # M02: c(다섯 소지도 한 줄, 전체 폭) + d(아래 넓고 낮은 패널)
    fig = S.figure(Wmm, Hmm); fig.set_size_inches(DECK_W_IN, DECK_H_IN)
    tw, gp = DECK_C_TW, DECK_C_GAP; th = map_th(tw)
    xc = (Wmm - (5 * tw + 4 * gp)) / 2; yc = Hmm - 7.5 - th
    panel_c(fig, lab, xc, yc, tw, gp)
    panel_d(fig, 62.0, 19.5, 180.0, 24.5, letter_x=-20.0 / 180.0, ylab="Resamples", letter_y=1.0)   # 4.65 in: d 위 = c 의 셋째 줄 아래, x 라벨 두 줄이 캔버스 안
    SRC["qa_overlaps_M02"] = _qa_overlap(fig, "deck M02")
    out.append(deck_save(fig, "M02"))
    (OUT / "source_data").mkdir(parents=True, exist_ok=True); (OUT / "source_data" / f"{STEM}_deck.json").write_text(json.dumps(SRC, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--slide", action="store_true"); ap.add_argument("--deck", action="store_true"); ap.add_argument("--fig", default="3")
    a = ap.parse_args()
    if a.deck:
        for p in deck():
            print(p)
    else:
        S.setup(scale=1.69 if a.slide else 1.0)
        for p in f3():
            print(p)
