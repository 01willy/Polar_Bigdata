"""XE_hires_covariates 시험(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.5절 '구현·시험' (a)–(f), 1절 누설 시험·출력 제한).

계획 2.5 는 시험 파일 이름을 tests/test_x_hires.py 로 적었다. 작업 지시의 이름(tests/test_x_xe.py)을 따랐고 내용은 등록 목록 (a)–(f)를 모두 담는다.

합성 자료 시험(라벨 값을 화면에 쓰지 않는다)
(a) 합성 래스터에서 점 추출·창 통계가 수동 계산과 같다: 창 평균·TPI·등급 비율·점 화소, 타일 모자이크와 화소 색인, 평면의 경사·TPI·D8 누적·TWI,
    ArcticDEM 창 지표(평면과 포물면), T2 창 묶음 작업(합성 DEM·GSW·Hansen 타일).
(b) 입력 집합 전환이 맞는 열을 넘긴다: 변형별 열 목록(90 % 규칙의 군 제외, 토양 교체, 점 화소 민감도), 적합에 들어가는 행렬의 열.
(c) 분해 항등식 두 개: 블록마다 SSE_총 = SSE_w + SSE_b, SSE_w = SSE_gl + SSE_l(함수와 단위 저장소 모두).
(d) x25 판이 h54.R9Unit 과 같다(모든 키의 예측, 총·격자 안·격자 사이 저장소). 고해상 변형의 P1 은 x25 의 P1 과 같은 키·같은 값이다.
(e) 새 열 추출 코드가 라벨 열을 읽지 않는다(읽기 기록, 거부, 원천 문자열, 합성 표의 라벨 열 독 값).
(f) MOD10A2 대체 정의(적설 표시 합성 수 × 8, 첫·마지막 무적설 전이 합성의 시작일)가 합성 시계열에서 맞다. MOD10A1·MOD13Q1 정의도 본다.
(g) 누설 시험(1절): 선택 밖 A 라벨과 B 라벨을 바꿔도 모든 예측이 같고, 선택 라벨 하나를 바꾸면 달라진다.
(h) 판정: Holm 가족 m = 6(행 없음은 p 1), _rule3, 해석 문장 갈래와 지역 열세 덧붙임.
(i) 특징 표 결합·최종 해시 단계: 90 % 규칙, 마감 초과 원자료의 결측 처리, 최종 해시와 다시 쓰기 거부, 군 열 해시의 판 사이 비교.
(j) 조각 이름·인자·몫 나누기와 합성 조각의 집계(봉인 폴더에만 쓰고 화면에는 행 수와 해시만).
실제 자료 시험(세기 범주: 셀 수만 쓰고 라벨 값은 쓰지 않는다)
(k) --count-only 가 라벨 유래 통계를 화면에 쓰지 않는다. 특징 표의 loc_id 결합이 문맥의 A·채점 셀 순서와 맞는다.
(l) XE-e: VWC 층 분류·캠페인 우선·위치·기기별 평균·15 m 결합, P1 격자 안 잔차가 E1 과 무관함, 블록 교차검증 설명 비율(합성).
(m) 실행 보호(검토 반영): 스모크 조각의 봉인 경로, --allow-unfinal 범위, 빈 특징 표·최종 해시 없음의 적합 전 거부, 최종 해시의 개정 이력 대조,
    집계의 최종판 조각 조건, 재현 관문(포괄률·단위 누락)과 판정 불가, n 200 행의 XE-c 표기, 문장 수치의 같은 대비·같은 n, 맹검 어휘,
    재표집 등록 이탈 열, SI 상태 표, 스모크 코어·스레드, 로컬 메모리 확인, 묶음 선택 파일.
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 taskset -c <코어 4개> python3 -m pytest -q tests/test_x_xe.py
특징 추출 모듈(scripts/1_data_prep/xe_feature_tools.py, xe_point_covariates.py)과 rasterio·pysheds 는 xbatch 묶음에 없으므로 지연 읽기와
skipif 로 둔다(Rescale 에서는 그 시험만 건너뛴다).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "3_deep_learning"))


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


XB = _load("xbatch_core", ROOT / "scripts" / "3_deep_learning" / "xbatch_core.py")
M = _load("x_hires_covariates", ROOT / "scripts" / "3_deep_learning" / "x_hires_covariates.py")
RG = M.RG
H, X, W = XB.H, XB.X, XB.W
PREP = ROOT / "scripts" / "1_data_prep"


def _has(*mods):
    return all(importlib.util.find_spec(m_) is not None for m_ in mods)


class _Lazy:
    """특징 추출 모듈의 지연 읽기(묶음에 없는 파일을 모듈 수준에서 읽지 않는다)."""

    _OWN = ("name", "path", "mod")

    def __init__(self, name, path):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "mod", None)

    def _real(self):
        if self.mod is None:
            object.__setattr__(self, "mod", _load(self.name, self.path))
        return self.mod

    def __getattr__(self, k):
        return getattr(self._real(), k)

    def __setattr__(self, k, v):                                          # monkeypatch.setattr(P, ...) 가 모듈 자체를 바꾸게 한다
        if k in self._OWN:
            object.__setattr__(self, k, v)
        else:
            setattr(self._real(), k, v)

    def __delattr__(self, k):
        delattr(self._real(), k)


T = _Lazy("xe_feature_tools", PREP / "xe_feature_tools.py")
P = _Lazy("xe_point_covariates", PREP / "xe_point_covariates.py")
NEED_T = pytest.mark.skipif(not (PREP / "xe_feature_tools.py").exists(), reason="특징 추출 모듈이 없다(xbatch 묶음 밖)")
NEED_P = pytest.mark.skipif(not ((PREP / "xe_point_covariates.py").exists() and (PREP / "xe_feature_tools.py").exists()),
                            reason="특징 추출 모듈이 없다(xbatch 묶음 밖)")
NEED_RIO = pytest.mark.skipif(not _has("rasterio", "affine"), reason="rasterio 가 없다")
NEED_SHEDS = pytest.mark.skipif(not _has("pysheds"), reason="pysheds 가 없다")
REAL = pytest.mark.skipif(not (ROOT / "data/processed/fidelity_base_v3.csv").exists(), reason="실제 자료(v3)가 없다")
NOMEM = ["--min-mem-gb", "0", "--max-mem-gb", "0"]                     # 시험 안의 main 호출: 메모리 대기·감시를 끈다(시험 프로세스에 상한을 남기지 않는다)


# ---------------------------------------------------------------- 합성 문맥
def make_ctx(seed=0, target="T", split=1, n_extra=6, nA=96, nbB=6):
    """합성 지역 내 문맥(h54.RCtx). 채점 셀은 블록마다 격자 묶음 3개(같은 √TDD), 묶음마다 위치 칸 1–3개, 위치마다 1–3셀(한 셀 위치 포함)."""
    rng = np.random.RandomState(seed)
    D = len(XB.FEATS)
    XA = rng.randn(nA, D).astype(np.float32); sA = rng.uniform(20, 40, nA); yA = 1.4 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    blkA = 100 * split + np.repeat(np.arange(8), nA // 8)
    latA, lonA = 65 + 0.5 * rng.rand(nA), -150 + 0.5 * rng.rand(nA)
    sB, blkB, latB, lonB = [], [], [], []
    for b in range(nbB):
        for g in range(3):
            s_ = 25.0 + 3 * g + 0.37 * b
            for loc in range(rng.randint(1, 4)):
                for _ in range(rng.randint(1, 4)):
                    sB.append(s_); blkB.append(1000 * split + b)
                    latB.append(66.0045 + 0.009 * (loc + 4 * g) + 0.6 * b + 0.001 * rng.rand()); lonB.append(-150.2 + 0.001 * rng.rand())
    sB = np.array(sB); nB = len(sB)
    XBm = rng.randn(nB, D).astype(np.float32); yB = 1.4 * sB + 3 * XBm[:, 1] + 2 * rng.randn(nB)
    c = W.RCtx(target, split, target, XA, yA, sA, blkA, latA, lonA, XBm, yB, sB, np.array(blkB), np.array(latB), np.array(lonB), 1.5,
               meta=dict(dup_of=-1, valid=True, n_valid_splits=1, n_unique_splits=1))
    if n_extra:
        eA, eB = rng.randn(nA, n_extra), rng.randn(nB, n_extra)
        eA[rng.rand(nA, n_extra) < 0.05] = np.nan
        M.attach_variant(c, "xh", np.hstack([XA, eA]), np.hstack([XBm, eB]))
    return c


def small_args(grid=(20, -1), seeds=1, splits=(1, 2), nboot=200):
    return XB.h54_args(exp="wf9", splits=list(splits), grid=list(grid), threads=1, cb_iters=10, seeds=seeds, draws_cap=1, nboot=nboot)


# ---------------------------------------------------------------- (a) 래스터 창과 지형
@NEED_T
def test_a_window_stats_manual():
    rng = np.random.RandomState(1)
    a = rng.rand(9, 11) * 100
    a[0, 0] = np.nan
    for r, c in ((4, 5), (0, 0), (8, 10), (1, 9)):
        w = np.full((3, 3), np.nan)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= r + i < 9 and 0 <= c + j < 11:
                    w[i + 1, j + 1] = a[r + i, c + j]
        assert np.allclose(T.window(a, r, c, 1), w, equal_nan=True)
        assert T.win_mean(a, r, c, 1) == pytest.approx(np.nanmean(w))
        if np.isfinite(a[r, c]):
            assert T.tpi_at(a, r, c, 1) == pytest.approx(a[r, c] - np.nanmean(w))
    cls = np.array([[10, 10, 20, 0], [90, 100, 60, 80], [10, 30, 0, 0]], float)
    fr = T.class_fractions(cls, 1, 1, 1, RG.WC_CODES, nodata=0)
    valid = cls[0:3, 0:3][cls[0:3, 0:3] != 0]
    assert fr["wc_tree"] == pytest.approx(np.sum(valid == 10) / len(valid)) and fr["wc_moss"] == pytest.approx(1 / len(valid))
    assert sum(fr.values()) == pytest.approx(np.isin(valid, list(RG.WC_CODES.values())).mean())
    oh = T.class_onehot(cls, 1, 1, RG.WC_CODES)
    assert oh["wc_moss"] == 1.0 and sum(oh.values()) == 1.0
    assert all(np.isnan(v) for v in T.class_fractions(np.zeros((3, 3)), 1, 1, 1, RG.WC_CODES).values())


def _write_tif(path, arr, left, top, res_x, res_y, nodata=None, dtype="float32", crs="EPSG:4326"):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype, crs=crs,
                       transform=from_origin(left, top, res_x, res_y), nodata=nodata) as d:
        d.write(arr.astype(dtype), 1)


@NEED_P
@NEED_RIO
def test_a_mosaic_and_pixel_index(tmp_path):
    rng = np.random.RandomState(2)
    A = rng.rand(10, 10).astype(np.float32); B = rng.rand(10, 10).astype(np.float32)
    _write_tif(tmp_path / "a.tif", A, 0.0, 10.0, 1.0, 1.0)
    _write_tif(tmp_path / "b.tif", B, 10.0, 10.0, 1.0, 1.0)
    arr, tr, grid = P.read_mosaic([tmp_path / "a.tif", tmp_path / "b.tif"], (8.2, 3.5, 12.7, 9.1), ref=tmp_path / "a.tif")
    assert np.allclose(arr, np.hstack([A, B])[0:7, 8:13]) and grid["row_off"] == 0 and grid["col_off"] == 8
    r, c = P.grid_rc(grid, np.array([9.5, 10.5, 12.01]), np.array([8.9, 4.2, 3.6]))
    r2, c2 = T.pixel_rc(tr, np.array([9.5, 10.5, 12.01]), np.array([8.9, 4.2, 3.6]))
    assert np.array_equal(r, r2) and np.array_equal(c, c2)
    full = np.hstack([A, B])
    for i, (x, y) in enumerate(((9.5, 8.9), (10.5, 4.2), (12.01, 3.6))):
        assert arr[r[i], c[i]] == full[int(np.floor(10 - y)), int(np.floor(x))]
    v = P.sample_points(tmp_path / "a.tif", np.array([0.5, 20.0]), np.array([9.5, 5.0]))
    assert v[0] == pytest.approx(A[0, 0]) and np.isnan(v[1])


@NEED_T
@NEED_RIO
@NEED_SHEDS
def test_a_plane_terrain_and_twi():
    H_, W_ = 40, 30
    dy, dx = 30.0, 25.0
    rr, cc = np.mgrid[0:H_, 0:W_]
    z = 500.0 - 0.05 * dy * rr + 0.0 * cc                                  # 남쪽(행 증가)으로 낮아지는 평면, 경사 0.05
    sl = T.slope_deg_grid(z, dy, dx)
    assert np.allclose(sl[1:-1, 1:-1], np.degrees(np.arctan(0.05)))
    from affine import Affine
    tr = Affine(dx / 111320.0, 0, -150.0, 0, -dy / 111320.0, 65.0)
    acc = T.d8_accumulation(z, tr)
    col = acc[5:35, 15]
    assert np.all(np.diff(col) == 1.0), "평면의 D8 누적은 남쪽으로 한 칸마다 1 늘어야 한다"
    out = T.terrain_points(z, tr, np.array([20]), np.array([15]), dy, dx, acc)
    a_spec = (acc[20, 15] + 1) * dx * dy / (0.5 * (dx + dy))
    assert out["twi_pt"][0] == pytest.approx(np.log(a_spec / 0.05), rel=1e-9)
    assert abs(out["tpi_90"][0]) < 1e-9 and abs(out["tpi_270"][0]) < 1e-9
    assert out["slope_90"][0] == pytest.approx(np.degrees(np.arctan(0.05))) and out["slope_px"][0] == pytest.approx(out["slope_90"][0])


@NEED_T
def test_a_arcticdem_points():
    res = 10.0
    rr, cc = np.mgrid[0:41, 0:41]
    plane = 100 + 0.1 * res * cc
    o = T.arcticdem_points(plane, res, np.array([20]), np.array([20]))
    assert o["ad_slope_30"][0] == pytest.approx(np.degrees(np.arctan(0.1))) and abs(o["ad_tpi_50"][0]) < 1e-9 and abs(o["ad_tpi_150"][0]) < 1e-9
    assert abs(o["ad_curv_30"][0]) < 1e-12
    assert o["ad_rough_50"][0] == pytest.approx(np.std(plane[18:23, 18:23]))
    xx, yy = cc * res, rr * res
    para = 0.01 * (xx - 200) ** 2 + 0.01 * (yy - 200) ** 2
    o2 = T.arcticdem_points(para, res, np.array([20]), np.array([20]))
    assert o2["ad_curv_30"][0] == pytest.approx(0.04, rel=1e-6)            # ∇²(0.01x² + 0.01y²) = 0.04
    assert T.half_for(30, 10) == 1 and T.half_for(50, 10) == 2 and T.half_for(150, 10) == 7 and T.half_for(150, 32) == 2 and T.half_for(30, 32) == 1


@NEED_P
@NEED_RIO
@NEED_SHEDS
def test_a_t2_cluster_synthetic_tiles(tmp_path, monkeypatch):
    """합성 1° DEM 타일 두 개(위도에 따라 연속인 평면)와 GSW·Hansen 타일로 T2 창 묶음 작업이 이름 규칙·창·수동 값과 맞는지 본다."""
    res = 0.002
    n = int(round(1 / res))
    for d in ("dem", "gsw", "hansen"):
        (tmp_path / d).mkdir()
    for la, lo in ((65, -150), (64, -150)):
        lat_c = (la + 1) - (np.arange(n) + 0.5) * res
        dem = np.repeat((5000.0 - 0.02 * 111320.0 * (66.0 - lat_c))[:, None], n, axis=1).astype(np.float32)   # 남쪽으로 낮아지는 평면(경사 0.02)
        _write_tif(tmp_path / "dem" / P.dem_tile(la, lo).name, dem, lo, la + 1, res, res)
    gres = 0.0005
    g = (np.arange(2000 * 2000).reshape(2000, 2000) % 101).astype(np.uint8)
    _write_tif(tmp_path / "gsw" / P.gsw_tile(70, -150).name, g, -150, 66, gres, gres, dtype="uint8")
    _write_tif(tmp_path / "hansen" / P.hansen_tile(70, -150).name, g, -150, 66, gres, gres, dtype="uint8")
    monkeypatch.setattr(P, "D_DEM", tmp_path / "dem"); monkeypatch.setattr(P, "D_GSW", tmp_path / "gsw"); monkeypatch.setattr(P, "D_HANSEN", tmp_path / "hansen")
    lat, lon = np.array([65.31, 65.33, 65.02]), np.array([-149.71, -149.75, -149.60])
    key, loc, out, note = P.t2_cluster(((0, 0), np.array([1, 2, 3]), lat, lon))
    assert len(note["dem_tiles"]) == 2 and all(np.isfinite(out["twi_pt"])) and np.allclose(out["tpi_90"], 0, atol=1e-3)
    assert np.allclose(out["tpi_270"], 0, atol=1e-3) and np.allclose(out["slope_90"], np.degrees(np.arctan(0.02)), rtol=1e-3)
    for i in range(3):
        r, c = int(np.floor((66 - lat[i]) / gres)), int(np.floor((lon[i] + 150) / gres))
        assert out["gsw_occ_pt"][i] == pytest.approx(np.mean(g[r - 1:r + 2, c - 1:c + 2].astype(float)))
        assert out["treecover_pt"][i] == pytest.approx(out["gsw_occ_pt"][i]) and out["treecover_px"][i] == g[r, c]


@NEED_P
def test_a_arcticdem_tile_rule():
    """ArcticDEM v4.1 타일 이름 규칙이 STAC 항목의 proj:bbox(07_41: x 0–100 km, y −3,400–−3,300 km; 08_40; 30_30)와 맞는다."""
    x = np.array([50e3, -50e3, -1050e3]); y = np.array([-3350e3, -3250e3, -1050e3])
    assert P.ad_tile_of(x, y) == ["07_41", "08_40", "30_30"]
    pts = pd.DataFrame(dict(lat=[70.0, 70.00001], lon=[-150.0, -150.00001]), index=[5, 9])
    k, _, _ = P.ad_window_keys(pts)
    assert k.iloc[0] == k.iloc[1] and list(k.index) == [5, 9]


# ---------------------------------------------------------------- (b) 입력 집합 전환
def test_b_variant_columns():
    x25, soil = list(XB.FEATS), M.SOIL_COLS
    allin = {g: True for g in RG.H_GROUPS}
    assert RG.variant_columns("xh0", "Alaska", x25, soil, allin)[0] == x25 + RG.GROUPS["H0"]
    assert RG.variant_columns("xt2", "Alaska", x25, soil, allin)[0] == x25 + RG.GROUPS["H0"] + RG.GROUPS["T2"]
    assert RG.variant_columns("xh", "Alaska", x25, soil, allin)[0] == x25 + RG.H_COLS and len(x25 + RG.H_COLS) == 70
    dec = dict(allin, O=False, M=False)
    cols, info = RG.variant_columns("xh", "Lena", x25, soil, dec)
    assert cols == x25 + [c for g in ("H0", "T2", "V", "S") for c in RG.GROUPS[g]] and info["dropped"] == ["M", "O"]
    dec2 = dict(allin, M=False, O=False, V=False, S=False)
    assert RG.variant_columns("xh", "Lena", x25, soil, dec2)[0] == RG.variant_columns("xt2", "Lena", x25, soil, dec2)[0], "모든 2단계 군이 빠지면 xh = xt2"
    sg = RG.variant_columns("sg250", "Alaska", x25, soil, allin)[0]
    assert len(sg) == 23 and not set(soil) & set(sg) and sg[-7:] == RG.SG250_COLS
    assert RG.variant_columns("sg250", "Lena", x25, soil, dec)[0] is None
    px = RG.variant_columns("xh_px", "Alaska", x25, soil, allin)[0]
    assert "slope_px" in px and "slope_90" not in px and "wc_moss_px" in px and "wc_moss" not in px and "tpi_90" in px and len(px) == 70
    assert RG.variant_columns("add_M", "Lena", x25, soil, dec)[0] is None and RG.variant_columns("add_V", "Lena", x25, soil, dec)[0] == x25 + RG.GROUPS["V"]
    assert RG.variant_columns("cov_AK", "Lena", x25, soil, allin, ["abv_tree"])[0] is None
    assert RG.variant_columns("cov_AK", "Alaska", x25, soil, dict(allin, cov_AK=True), ["abv_tree", "x"])[0] == x25 + ["abv_tree"]
    assert RG.variant_columns("cov_AK", "Alaska", x25, soil, None, ["abv_tree"])[0] == x25 + ["abv_tree"], "결정 없음(세기)은 포함으로 센다"
    assert RG.variant_columns("cov_AK", "Alaska", x25, soil, allin, ["abv_tree"])[0] is None, "결정이 있는데 피복 항목이 없으면 뺀다(실패 쪽)"
    assert RG.variant_columns("cov_LE", "Lena", x25, soil, allin, ["abv_tree"])[1]["skip"] == RG.COVER_NOT_RUN
    assert RG.COVER_NOT_RUN.startswith("미실행(등록 이탈)")
    assert RG.method_suffix("x25") == "" and RG.method_suffix("xh") == "@xh"


class SpyUnit(M.XERUnit):
    """적합을 하지 않고 학습 행렬과 예측 행렬의 열 수를 기록한다."""

    def __init__(self, *args, **kw):
        self.seen = []
        super().__init__(*args, **kw)

    def fit(self, lr, method, build, nrow, seed, preds, n, d, lab_idx, placement="cell", **tr):
        Xtr, ytr, _ = build()
        self.seen.append((method, Xtr.shape[1], tuple(p.shape[1] for p in preds)))
        return [np.zeros(len(p)) for p in preds]


def test_b_fit_matrices_follow_variant():
    c = make_ctx(n_extra=6)
    a = small_args()
    for v, k in (("x25", 25), ("xh", 31)):
        U = SpyUnit(a, c, v)
        U.run()
        assert U.seen and all(s_[1] == k and all(q == k for q in s_[2]) for s_ in U.seen), v
        if v == "xh":
            assert {m for m, *_ in U.seen} == {"R1@xhcv", "R1@xh", "D0@xh"}
        else:
            assert {"R1", "R1cv", "D0", "Re", "Recv"} <= {m for m, *_ in U.seen}
    assert np.array_equal(c.XAf("xh")[:, :25], c.XA) and c.XAf("x25") is c.XA
    with pytest.raises(ValueError):
        M.XERUnit(a, c, "xt2")
    f = M.FeatTable(None)
    assert f.is_dummy and f.decisions("Alaska") is None


# ---------------------------------------------------------------- (c) 분해 항등식
def test_c_decomposition_identities_functions():
    rng = np.random.RandomState(3)
    c = make_ctx(seed=3, n_extra=0)
    y, p = rng.randn(len(c.yB)) * 10, rng.randn(len(c.yB)) * 10
    gid, mg, lid, ml = M.loc_groups(c.sB, c.blkB, c.latB, c.lonB)
    w_fn, b_fn = W.decomp_fns(gid, mg)
    wl, bl = M.loc_decomp_fns(gid, mg, lid, ml)
    blk = np.asarray(c.blkB)

    def sse(mask_blk, yy, pp):
        return pd.Series((yy - pp) ** 2).groupby(mask_blk).sum()
    tot = sse(blk, y, p)
    yw, pw = w_fn(y, p); yb, pb = b_fn(y, p); yl, pl = wl(y, p); yg, pg = bl(y, p)
    Sw, Sb, Sl, Sg = sse(blk[mg], yw, pw), sse(blk, yb, pb), sse(blk[ml], yl, pl), sse(blk[mg], yg, pg)
    assert np.allclose(tot.values, Sw.reindex(tot.index).fillna(0).values + Sb.reindex(tot.index).values)
    assert np.allclose(Sw.values, Sg.reindex(Sw.index).values + Sl.reindex(Sw.index).fillna(0).values)
    assert ml.sum() > 0 and (mg & ~ml).sum() > 0, "합성 자료에 한 셀 위치와 여러 셀 위치가 모두 있어야 한다"
    assert all(set(np.where(lid == l_)[0]) <= set(np.where(gid == gid[np.where(lid == l_)[0][0]])[0]) for l_ in np.unique(lid)), "위치 묶음은 격자 묶음 안이다"
    ky, kx = M.loc_cell(np.array([66.0045, 66.0135]), np.array([-150.0, -150.0]))
    assert ky[1] - ky[0] == 1 and kx[0] == int(np.floor(-150 * np.cos(np.radians((ky[0] + 0.5) * 0.009)) / 0.009))


def _store_sse(st, key):
    s_, _ = st.get(key)
    return pd.Series(s_, index=st.blocks)


def test_c_decomposition_identities_unit_stores():
    c = make_ctx(seed=4)
    a = small_args(grid=(20,))
    U = M.XERUnit(a, c, "xh").run()
    rows, stores, stats = U.finish()
    by = {st.target: st for st in stores}
    assert set(by) == {"T|r", "T~w|r", "T~b|r", "T~l|r", "T~gl|r"} and stats["notes"]["loc"]["n_multi_loc_cells"] > 0
    for key in by["T|r"].keys:
        if key[0] == "P0":
            continue
        tot, w_, b_ = _store_sse(by["T|r"], key), _store_sse(by["T~w|r"], key), _store_sse(by["T~b|r"], key)
        l_, g_ = _store_sse(by["T~l|r"], key), _store_sse(by["T~gl|r"], key)
        assert np.allclose(tot.values, w_.reindex(tot.index).fillna(0).values + b_.reindex(tot.index).values, rtol=1e-9, atol=1e-6)
        assert np.allclose(w_.values, g_.reindex(w_.index).values + l_.reindex(w_.index).fillna(0).values, rtol=1e-9, atol=1e-6)


# ---------------------------------------------------------------- (d) x25 = R9Unit
def test_d_x25_equals_r9unit():
    c = make_ctx(seed=5, n_extra=0)
    a = small_args()
    with XB.h54_trace() as (p1, _):
        r9 = W.R9Unit(a, c, "wf9", "").run()
    with XB.h54_trace() as (p2, _):
        xe = M.XERUnit(a, c, "x25").run()
    k1 = {k[4:]: v for k, v in p1.items()}
    k2 = {k[4:]: v for k, v in p2.items()}
    assert set(k1) == set(k2) and len(k1) > 20
    assert all(np.array_equal(k1[k], k2[k]) for k in k1), "x25 판의 예측이 R9Unit 과 다르다"
    _, s9, _ = r9.finish()
    _, sx, _ = xe.finish()
    s9 = {st.target: st for st in s9}
    sx = {st.target: st for st in sx}
    for nm in ("T|r", "T~w|r", "T~b|r"):
        assert s9[nm].keys == sx[nm].keys
        for k in s9[nm].keys:
            assert np.array_equal(s9[nm].get(k)[0], sx[nm].get(k)[0])
    c2 = make_ctx(seed=5, n_extra=4)
    with XB.h54_trace() as (p3, _):
        M.XERUnit(a, c2, "xh").run()
    k3 = {k[4:]: v for k, v in p3.items()}
    p1keys = [k for k in k3 if k[0] == "P1"]
    assert p1keys and all(np.array_equal(k3[k], k2[k]) for k in p1keys) and any(k[0] == "R1@xh" and k[7] == -1.0 for k in k3)
    assert any(k[0] == "D0@xh" and k[1] == "catboost" for k in k3)


# ---------------------------------------------------------------- (e) 라벨 열을 읽지 않는다
@NEED_P
def test_e_extraction_reads_no_label_columns(tmp_path, monkeypatch):
    fb = tmp_path / "fb.csv"
    n = 30
    pd.DataFrame(dict(loc_id=np.arange(n), lat=np.linspace(60, 72, n), lon=np.linspace(-150, 125, n),
                      region=(["ABoVE_AK", "Lena_RU", "ABoVE_CA"] * 10), alt_cm=["독"] * n, sigma_prior_cm=["독"] * n, source_id="F4_direct")).to_csv(fb, index=False)
    P.TRACE_READS.clear()
    pts = P.read_points(fb)
    assert len(pts) == n and set(pts.target) == {"Alaska", "Lena", "Canada"}
    for path, cols in P.TRACE_READS:
        assert not set(cols) & set(RG.LABEL_COLS), (path, cols)
    with pytest.raises(RuntimeError):
        P.read_csv_cols(fb, ["loc_id", "alt_cm"])
    for f in ("scripts/1_data_prep/xe_point_covariates.py", "scripts/1_data_prep/xe_feature_tools.py"):
        src = (ROOT / f).read_text()
        assert "alt_cm" not in src and '"y"' not in src and "sigma_prior" not in src, f


# ---------------------------------------------------------------- (f) MODIS 정의
def _mod10a2_series(cloud=()):
    rows = []
    for Y in range(2014, 2022):
        for doy in range(1, 366, 8):
            d = pd.Timestamp(f"{Y}-01-01") + pd.Timedelta(days=doy - 1)
            if d < pd.Timestamp("2014-09-01") or d > pd.Timestamp("2021-08-31"):
                continue
            snow = doy < 150 or doy >= 280
            code = 200 if snow else 25
            if (Y, doy) in cloud:
                code = 50
            rows.append((d, code))
    return pd.DataFrame(rows, columns=["date", "code"])


@NEED_T
def test_f_mod10a2_registered_definition():
    s = _mod10a2_series()
    m = T.mod10a2_metrics(s.date.values, s.code.values)
    hy = T.hydro_year(s.date.values)
    exp_scd = np.mean([8 * int(((hy == Y) & (s.code.values == 200)).sum()) for Y in range(2015, 2021)])
    assert m["scd500"] == pytest.approx(exp_scd) and m["n_hy"] == 6
    assert m["snowoff_doy500"] == pytest.approx(153.0) and m["snowon_doy500"] == pytest.approx(281.0)
    s2 = _mod10a2_series(cloud={(2017, 153), (2018, 281)})                # 전이 합성이 구름이면 다음 판정 합성이 전이 합성이다
    m2 = T.mod10a2_metrics(s2.date.values, s2.code.values)
    assert m2["by_year"]["off"][2017] == 161 and m2["by_year"]["on"][2018] == 289
    assert m2["snowoff_doy500"] == pytest.approx((153 * 5 + 161) / 6) and m2["snowon_doy500"] == pytest.approx((281 * 5 + 289) / 6)
    hy2 = T.hydro_year(s2.date.values)
    assert m2["by_year"]["scd"][2017] == 8 * int(((hy2 == 2017) & (s2.code.values == 200)).sum())
    st = T.mod10a2_state([0, 1, 11, 25, 37, 39, 50, 100, 200, 254, 255])
    assert np.isnan(st[[0, 1, 2, 4, 5, 6, 7, 9, 10]]).all() and st[3] == 0 and st[8] == 1
    assert T.hydro_year(["2014-09-01", "2015-08-31", "2015-09-01"]).tolist() == [2015, 2015, 2016]


@NEED_T
def test_f_mod10a1_and_mod13q1():
    d = pd.date_range("2014-09-01", "2021-08-31", freq="D")
    v = np.where((d.dayofyear < 140) | (d.dayofyear >= 290), 80, 3).astype(float)
    v[(d.year == 2016) & (d.dayofyear == 140)] = 250                        # 구름: 가장 가까운 판정 날(같은 거리면 앞 날)로 채운다
    m = T.mod10a1_metrics(d, v)
    assert m["by_year"]["off"][2016] == 141 and m["by_year"]["off"][2015] == 140 and m["by_year"]["on"][2015] == 290
    assert T.fill_nearest(np.array([np.nan, 1, np.nan, 0, np.nan, np.nan])).tolist() == [1, 1, 1, 0, 0, 0]
    assert T.fill_nearest(np.array([1, np.nan, np.nan, 0])).tolist() == [1, 1, 0, 0]
    dates = ["2015-06-10", "2015-06-26", "2015-08-29", "2016-06-09", "2016-05-24", "2016-06-25"]
    nd = [0.5, 0.7, 0.6, 0.3, 0.9, 0.8]
    rel = [0, 1, 0, 2, 0, 0]
    o = T.mod13q1_metrics(dates, nd, nd, [0.3] * 6, [0.1] * 6, rel)
    assert o["n_comp"] == 4 and o["ndvi250_jja"] == pytest.approx(np.mean([0.5, 0.7, 0.6, 0.8])) and o["ndwi250_jja"] == pytest.approx(0.5)
    assert o["ndvi250_max"] == pytest.approx(0.75)                          # 일차 177 의 기후값 (0.7 + 0.8)/2


# ---------------------------------------------------------------- (g) 누설 시험
def test_g_leakage_invariance():
    a = small_args(grid=(20, 40))
    c = make_ctx(seed=6)

    def run_fn(cc):
        M.XERUnit(a, cc, "xh").run()
    res = XB.leakage_invariance(run_fn, c)
    assert res["ok"] and res["same_keys"] and res["max_abs_diff"] == 0.0 and res["n_keys"] > 10
    assert 0 < res["n_keep"] < res["n_A"]
    with XB.h54_trace() as (p1, tr):
        run_fn(c)
    keep = XB.selected_union(tr)
    c_bad = XB.perturb_labels(c, keep[1:], seed=3)
    with XB.h54_trace() as (p2, _):
        run_fn(c_bad)
    assert not XB.compare_predictions(p1, p2)["ok"], "선택 라벨을 바꿔도 예측이 같다(시험이 둔하다)"


# ---------------------------------------------------------------- (h) 판정·Holm·문장
def _row(delta, lo, hi, v="미결정", p=0.2, und=False):
    return dict(scope="MEAN", delta=delta, ci_lo=lo, ci_hi=hi, ci_lo_beq=lo, ci_hi_beq=hi, verdict4=v, p_two=p, undetermined=und)


def test_h_holm_and_sentences():
    A, B = M.MAIN_CONTRASTS[0][1], M.MAIN_CONTRASTS[1][1]
    pooled = {("XE-a", A, "격자 안", 500): _row(-0.8, -1.2, -0.3, "우세", 0.001), ("XE-a", A, "격자 안", -1): _row(-0.6, -1.0, -0.2, "우세", 0.004),
              ("XE-b", B, "격자 안", -1): _row(0.2, -0.3, 0.6, "미결정", 0.3)}
    region = [dict(scope="region", hypothesis="XE-a", contrast=A, part="격자 안", n=-1, target="Lena~w|r", delta=0.9, ci_lo=0.2, ci_hi=1.5, worse=True),
              dict(scope="region", hypothesis="XE-c", contrast=A, part="총", n=-1, target="Canada|r", delta=0.3, ci_lo=-0.2, ci_hi=0.8, worse=False)]
    vr, ht = M.verdict_rows(pooled, region, None, {"group_decisions": {"Canada": {"O": {"included": False}}}}, ["Alaska", "Lena", "Canada"])
    assert len(ht) == 6 and set(ht.m) == {6} and (ht.p[ht.label == "XE-a|n1000"] == 1.0).all()
    assert ht.loc[ht.label == "XE-a|n500", "p_holm"].iloc[0] == pytest.approx(min(1.0, 0.001 * 6))
    va = vr[vr.hypothesis == "XE-a"].iloc[0]
    assert va.verdict.startswith("부분 지지") and va.branch == "우세"
    s_ = vr[vr.scope == "sentence"].sentence.iloc[0]
    assert "확인하지 못했다" in s_ and "지역 레나(XE-a, n 전량) 에서는 오차가 컸다" in s_ and "캐나다" in s_ and "O 군" in s_ and "고친다" in s_
    pooled2 = {("XE-a", A, "격자 안", n): _row(0.1, -0.3, 0.4, "동등", 0.5) for n in M.MAIN_N}
    vr2, _ = M.verdict_rows(pooled2, [], None, {}, ["Alaska", "Lena", "Canada"])
    assert vr2[vr2.scope == "sentence"].sentence.iloc[0].startswith("고해상 입력을 더한 ML 의 격자 안 오차는 x25 와 0.5 cm 안에서 같았다")
    pooled3 = {("XE-a", A, "격자 안", n): _row(0.1, -0.6, 0.9, "미결정", 0.5) for n in M.MAIN_N}
    s3 = M.verdict_rows(pooled3, [], None, {}, ["Alaska", "Lena", "Canada"])[0]
    assert "100 m 셀 안 분산 비율은 알래스카 49.2 %" in s3[s3.scope == "sentence"].sentence.iloc[0]
    s4 = M.verdict_rows({}, [], None, {}, ["Alaska"])[0]
    assert s4[s4.hypothesis == "XE-a"].verdict.iloc[0].startswith("판정 불가") and "판정할 수 없었다" in s4[s4.scope == "sentence"].sentence.iloc[0]
    assert M.branch("기각: 열세인 대비가 있다(n=500)", ["열세", "미결정"]) == "열세"


# ---------------------------------------------------------------- (i) 특징 표 결합과 최종 해시
@NEED_P
def test_i_assemble_finalize(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XE_OUT_ROOTS", str(tmp_path))
    n = 40
    fb, cov = tmp_path / "fb.csv", tmp_path / "cov.csv"
    reg = ["ABoVE_AK"] * 20 + ["Lena_RU"] * 10 + ["ABoVE_CA"] * 10
    pd.DataFrame(dict(loc_id=np.arange(n), lat=np.linspace(60, 72, n), lon=-140.0, region=reg, alt_cm="독", source_id="F4_direct")).to_csv(fb, index=False)
    rng = np.random.RandomState(0)
    h0 = pd.DataFrame(rng.rand(n, 10), columns=RG.GROUPS["H0"]); h0.insert(0, "loc_id", np.arange(n))
    h0.loc[20:21, RG.GROUPS["H0"]] = np.nan                                # 레나 10행 가운데 2행 결측 → 유한값 0.8 < 0.9 → 레나 H0 제외
    h0.to_csv(cov, index=False)
    monkeypatch.setattr(P, "FB", fb); monkeypatch.setattr(P, "COV_EXT", cov)
    out = tmp_path / "xe"
    args = ["--out-dir", str(out), "--workers", "1", "--mem-gb", "0", "--min-free-gb", "0"]
    P.main(["h0"] + args)
    pts = P.read_points(fb)
    t2 = pd.DataFrame(dict(loc_id=np.arange(n), **{c: rng.rand(n) for c in RG.GROUPS["T2"] + ["slope_px", "gsw_occ_px", "treecover_px"]}))
    P.write_part(out, "T2", t2, pts, dict(T2=dict(latest_mtime="2026-10-04 15:00:00 +0900")))
    o = pd.DataFrame(dict(loc_id=np.arange(n), **{c: rng.rand(n) for c in RG.GROUPS["O"]}))
    P.write_part(out, "O", o, pts, dict(O=dict(latest_mtime="2026-10-09 00:00:00 +0900")))   # 마감(T0 + 72 h) 뒤 → 결측으로 둔다
    P.main(["assemble"] + args)
    meta = json.loads((out / "xe_feat_v1_meta.json").read_text())
    dec = meta["group_decisions"]
    assert dec["Alaska"]["H0"]["included"] and not dec["Lena"]["H0"]["included"] and dec["Canada"]["T2"]["included"]
    assert "O" in meta["late_sources"] and not dec["Alaska"]["O"]["included"] and not dec["Alaska"]["M"]["included"]
    feat = pd.read_csv(out / "xe_feat_v1.csv")
    assert list(feat.columns[:2]) == ["loc_id", "target"] and feat[RG.GROUPS["O"]].isna().all().all() and len(feat) == n
    assert meta["stage"] == "stage2" and meta["group_cols_sha256"]["H0"] == RG.cols_sha(feat, RG.GROUPS["H0"])
    sha_t2 = meta["group_cols_sha256"]["T2"]
    P.main(["finalize"] + args)
    fin = json.loads((out / "xe_feat_v1_final.json").read_text())
    assert fin["sha256"] == RG.sha256_file(out / "xe_feat_v1.csv") and "Lena" in fin["dropped_groups"] and "H0" in fin["dropped_groups"]["Lena"]
    assert (out / "xe_feat_v1_revision_entry.md").read_text().startswith("- ")
    ft = M.FeatTable(out / "xe_feat_v1.csv")
    assert ft.final_local_ok and not ft.final_ok, "최종 해시 파일만 있고 개정 이력 커밋이 없으면 최종판이 아니다"
    monkeypatch.setattr(M, "final_committed", lambda sha, **k: dict(ok=True, how="시험"))
    ft = M.FeatTable(out / "xe_feat_v1.csv")
    assert ft.final_ok and ft.decisions("Lena")["H0"] is False and ft.group_sha(["T2"])["T2"] == sha_t2
    mtx = ft.matrix(np.array([3, 999, 1]), ["twi", "rel_1km"])
    assert np.allclose(mtx[0], h0.loc[3, ["twi", "rel_1km"]].values) and np.isnan(mtx[1]).all()
    o.loc[0, RG.GROUPS["O"][0]] = 0.123456
    P.write_part(out, "O", o, pts, dict(O=dict(latest_mtime="2026-10-05 00:00:00 +0900")))
    P.main(["assemble"] + args)
    with pytest.raises(SystemExit):
        P.main(["finalize"] + args)
    meta2 = json.loads((out / "xe_feat_v1_meta.json").read_text())
    assert meta2["group_cols_sha256"]["T2"] == sha_t2 and meta2["group_cols_sha256"]["H0"] == meta["group_cols_sha256"]["H0"], "1단계 군 열 해시는 판 사이에서 같다"
    captured = capsys.readouterr().out
    assert "독" not in captured


@NEED_P
def test_i_s_path_choice():
    pts = pd.DataFrame(dict(loc_id=[1, 2, 3, 4]))
    pm = pd.Series(["a", "b", "c", "d"], index=[1, 2, 3, 4])

    def tab(cover):
        return lambda fs, prod: pd.DataFrame(dict(pix=list(cover[prod])))
    files = lambda have: (lambda prod: ["f"] if prod in have else [])      # noqa: E731
    ch = P.s_choose(pts, files({"MOD10A1", "MOD10A2"}), tab({"MOD10A1": "abcd", "MOD10A2": "abcd"}), pm)
    assert ch[0] == "MOD10A1"
    ch = P.s_choose(pts, files({"MOD10A1", "MOD10A2"}), tab({"MOD10A1": "ab", "MOD10A2": "abcd"}), pm)
    assert ch[0] == "MOD10A2" and ch[3]["MOD10A1"] == 0.5
    assert P.s_choose(pts, files({"MOD10A1"}), tab({"MOD10A1": "ab"}), pm)[0] == "MOD10A1"
    assert P.s_choose(pts, files(set()), tab({}), pm)[0] == "none"


# ---------------------------------------------------------------- (j) 조각 이름·인자·집계
def test_j_args_shards_and_names(tmp_path):
    a = M.parse_args(["--stage", "r1b", "--feat", "none", "--count-only"])
    assert a.VARIANTS == ["x25", "xh0", "xt2"] and a.SPLIT_LIST == list(range(1, 26)) and a.GRID == [200, 500, 1000, -1] and a.TAG == "xe_r1b"
    assert a.h.SPLITS == list(range(1, 26)) and a.h.G["wf9"] == [200, 500, 1000, -1] and a.threads <= 4 and a.FEAT.is_dummy
    assert a.SHARDS == a.OUT / "shards" and a.GATE_LEVEL == "elm_hematite" and a.gate_wf9 == str(M.WF9_SHARDS)
    a3 = M.parse_args(["--stage", "r3", "--feat", "none", "--count-only"])
    assert a3.VARIANTS[:2] == ["x25", "xh"] and "sg250" in a3.VARIANTS and "xh0" not in a3.VARIANTS
    s = M.parse_args(["--smoke", "--feat", "none", "--count-only", "--threads", "4"])
    assert s.TARGETS == ["Canada"] and s.SPLIT_LIST == [7] and s.GRID == [200, -1] and s.seeds == 1 and s.TAG == "xe_r1b_smoke"
    assert s.SEALED_NAME.endswith("/smoke") and s.nboot <= 500 and s.threads == 2, "제한 스모크는 스레드 2"
    assert s.SHARDS == s.OUT / "smoke" / "sealed" / "shards" and "sealed" in s.SHARDS.parts, "스모크 조각은 봉인 폴더 안"
    assert s.GATE_LEVEL == "local_rescale"
    units = [("Alaska", sp, v) for sp in range(1, 5) for v in ("x25", "xh")]
    assert M.parse_shard("2/3", units) == units[1::3] and M.parse_shard("Alaska:3:xh", units) == [("Alaska", 3, "xh")]
    with pytest.raises(SystemExit):
        M.parse_shard("Lena:3:xh", units)
    p = XB.shard_paths(tmp_path, "xe_r1b", "Alaska", "r", 7, "xh0")
    assert p["unit"].name == "xe_r1b__cpu__Alaska__r__s7__xh0_unit.json"
    with pytest.raises(SystemExit):
        M.parse_args(["--variants", "x99", "--feat", "none"])
    e = M.parse_args(["--xe-e", "--feat", "none"])
    assert e.TAG == "xe_e" and e.TARGETS == ["Alaska", "Canada"] and M.run_mode(e) == "run"
    es = M.parse_args(["--xe-e", "--smoke", "--feat", "none"])
    assert es.TAG == "xe_e_smoke" and es.TARGETS == ["Canada"] and M.run_mode(es) == "smoke" and "sealed" in str(es.SEALED_NAME + "/sealed")
    assert M.run_mode(M.parse_args(["--xe-e-prep", "--feat", "none"])) == "count"


def _final_feat(tmp_path, monkeypatch, commit=True):
    """합성 최종판 특징 표(모든 군 포함, 열은 H_COLS·PX_COLS). commit 이면 개정 이력 대조를 통과시킨다."""
    d = tmp_path / "xe"
    d.mkdir(parents=True, exist_ok=True)
    fp = d / "xe_feat_v1.csv"
    pd.DataFrame({"loc_id": [1], **{c: [0.0] for c in RG.H_COLS + RG.PX_COLS}}).to_csv(fp, index=False)
    sha = RG.sha256_file(fp)
    dec = {t: {g: dict(included=True, finite_frac=1.0) for g in RG.H_GROUPS} for t in RG.TARGETS}
    (d / "xe_feat_v1_meta.json").write_text(json.dumps(dict(sha256=sha, group_decisions=dec, group_cols_sha256={"H0": "h" * 64, "T2": "t" * 64})))
    (d / "xe_feat_v1_final.json").write_text(json.dumps(dict(sha256=sha, group_cols_sha256={"H0": "h" * 64, "T2": "t" * 64})))
    if commit:
        monkeypatch.setattr(M, "final_committed", lambda s_, **k: dict(ok=s_ == sha, how="시험"))
    return fp, sha


def _write_synth(a, sha, ref_dir=None, unit_extra=None, variants=("x25", "xh"), nA=96):
    """합성 세 대상 × 두 분할 × 변형 조각. ref_dir 이면 x25 조각을 WF9 이름(tag wf9, 변형 없음)으로도 쓴다(재현 관문 기준)."""
    for k, t in enumerate(("Alaska", "Lena", "Canada")):
        for sp in (1, 2):
            c = make_ctx(seed=10 * k + sp, target=t, split=sp, n_extra=5, nA=nA)
            for v in variants:
                U = M.XERUnit(a.h, c, v).run()
                rows, stores, stats = U.finish()
                unit = dict(stats, split=sp, dup_of=-1, valid=True, variant=v, exp="xe", feat_group_sha={}, allow_unfinal=False,
                            feat_final_ok=True, feat_sha256=sha)
                unit.update(unit_extra or {})
                cfg = XB.make_unit_cfg("xe", v, f"synthetic:{v}", grid=a.GRID, splits=[1, 2])
                XB.write_shard(a.SHARDS, a.TAG, t, "r", sp, rows, stores, cfg, unit, v, [1, 2], M.__file__)
                if ref_dir is not None and v == "x25":
                    XB.write_shard(ref_dir, "wf9", t, "r", sp, rows, stores, cfg, unit, "", [1, 2], M.__file__)


def test_j_summarize_synthetic_shards_sealed(tmp_path, monkeypatch, capsys):
    """합성 세 대상 × 두 분할 × (x25, xh) 조각을 등록 이름으로 쓰고 집계한다. 표는 봉인 폴더에만, 화면에는 행 수와 해시만.
    재현 관문이 봉인 표보다 먼저 돌고(합성 WF9 기준, 같은 노드 0), 재표집 1,000회 미만은 등록 이탈 열을 단다."""
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    fp, sha = _final_feat(tmp_path, monkeypatch)
    ref = tmp_path / "wf9ref"
    a = M.parse_args(["--stage", "r3", "--variants", "x25,xh", "--feat", str(fp), "--out-dir", str(tmp_path / "XE"), "--nboot", "200",
                      "--grid", "200,all", "--splits", "1,2", "--draws-cap", "1", "--seeds", "1", "--cb-iters", "10", "--gate-wf9", str(ref),
                      "--gate-level", "same_node"])
    assert a.FEAT.final_ok
    _write_synth(a, sha, ref_dir=ref, nA=240)                             # 후보 240셀: n 200 행이 생긴다(XE-c 표기 확인)
    capsys.readouterr()
    with XB.restricted_output():
        res = M.summarize(a)
    out = capsys.readouterr().out
    assert res and f"{a.TAG}_xh_tests.csv" in res["tables"] and f"{a.TAG}_xh_hypotheses.csv" in res["tables"]
    assert res["gate"]["passed"] and res["gate"]["coverage_x25_in_wf9"] == 1.0 and res["gate"]["coverage_wf9_in_x25"] == 1.0
    assert out.index("[gate]") < out.index("[봉인]"), "재현 관문은 봉인 표보다 먼저 돈다"
    sd = tmp_path / "XE" / "sealed"
    assert (sd / "sealed_manifest.json").exists() and (sd / f"{a.TAG}_xh_holm.csv").exists() and (sd / f"{a.TAG}_xh_decomp.csv").exists()
    assert not XB.FORBIDDEN_OUT.search(out) and "[봉인]" in out, "집계 화면에 RMSE·Δ·판정이 있다"
    dec = pd.read_csv(sd / f"{a.TAG}_xh_decomp.csv")
    assert np.nanmax(np.abs(dec.check_tot_w_b.values)) < 1e-6 and np.nanmax(np.abs(dec.check_w_gl_l.values)) < 1e-6
    tests = pd.read_csv(sd / f"{a.TAG}_xh_tests.csv")
    assert {"XE-a", "XE-b", "XE-c", "XE-d"} <= set(tests.hypothesis) and (tests[tests.scope == "MEAN"].n_regions_registered == 3).all()
    assert not len(tests[tests.hypothesis.isin(["XE-a", "XE-b", "XE-a0"]) & ~tests.n.isin(M.MAIN_N)]), "n 200 행은 XE-c 다"
    moved = tests[tests.hypothesis_registered.isin(["XE-a", "XE-b"]) & ~tests.n.isin(M.MAIN_N)]
    assert len(moved) and (moved.hypothesis == "XE-c").all() and (moved.role == "보조").all()
    assert set(tests.blind) <= set(XB.BLIND_LABELS) and tests.blind_reason.notna().all()
    assert (tests.nboot == 200).all() and tests.nboot_note.str.startswith("재표집 200회(등록 이탈").all() and (tests.gate == "통과").all()
    hyp = pd.read_csv(sd / f"{a.TAG}_xh_hypotheses.csv")
    assert (hyp.nboot_note.str.contains("등록 이탈")).all() and not hyp[hyp.scope == "verdict"].verdict.str.contains("관문").any()
    meta = json.loads((sd / f"{a.TAG}_meta.json").read_text())
    assert meta["gate_wf9"]["passed"] and meta["nboot_registered"] == 10000 and meta["feat_final_ok"]
    with pytest.raises(PermissionError):
        XB.assert_not_sealed(sd / f"{a.TAG}_xh_tests.csv")
    # 기대 x25 단위가 빠지면 관문 실패
    units = [json.loads(s_["unit"].read_text()) for s_ in XB.find_shards(a.SHARDS, a.TAG)]
    XB.shard_paths(a.SHARDS, a.TAG, "Canada", "r", 2, "x25")["unit"].unlink()
    with XB.restricted_output():
        g = M.gate_wf9(a, units)
    assert not g["passed"] and "Canada:2" in g["missing_x25_units"] and g["status"].startswith("관문 실패")


def test_j_summarize_rejects_unfinal_and_marks_gate(tmp_path, monkeypatch, capsys):
    """2단계 변형 조각이 최종판 조건(allow_unfinal 거짓, feat_final_ok 참, 같은 최종 해시)을 어기면 집계를 거부한다. 관문을 끄면 판정 불가로 쓴다."""
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    fp, sha = _final_feat(tmp_path, monkeypatch)
    base = ["--stage", "r3", "--variants", "x25,xh", "--feat", str(fp), "--nboot", "200", "--grid", "20,all", "--splits", "1,2", "--draws-cap", "1",
            "--seeds", "1", "--cb-iters", "10", "--gate-wf9", "none"]
    a = M.parse_args(base + ["--out-dir", str(tmp_path / "bad")])
    _write_synth(a, sha, unit_extra=dict(allow_unfinal=True))
    with pytest.raises(SystemExit):
        M.summarize(a)
    b = M.parse_args(base + ["--out-dir", str(tmp_path / "other")])
    _write_synth(b, "f" * 64)
    with pytest.raises(SystemExit):
        M.summarize(b)
    c = M.parse_args(base + ["--out-dir", str(tmp_path / "ok")])
    _write_synth(c, sha)
    capsys.readouterr()
    with XB.restricted_output():
        res = M.summarize(c)
    assert res["gate"]["status"].startswith("관문 미실시")
    hyp = pd.read_csv(tmp_path / "ok" / "sealed" / f"{c.TAG}_xh_hypotheses.csv")
    v = hyp[hyp.scope == "verdict"]
    assert v.verdict.str.startswith("판정 불가(관문").all() and "판정할 수 없었다" in hyp[hyp.scope == "sentence"].sentence.iloc[0]
    with pytest.raises(SystemExit):
        M.parse_args(base + ["--out-dir", str(tmp_path / "x"), "--summarize-only", "--allow-unfinal"])


# ---------------------------------------------------------------- (l) XE-e
def _above_rows():
    """합성 ABoVE 행. 위치 A(ALT 2015): 같은 해 shallow 2기기(30, 40), 다른 해 shallow 90(캠페인 행이 있어 뺀다), deep 은 2016 만(50, 모든 행),
    GPR 2015(10), 하단 결측 비 GPR 1행(층 없음), VWC 결측 1행. 위치 B: 미표기 기기 shallow 20(ALT 없음)."""
    r = []
    A, B = (65.00001, -150.00001), (66.0, -151.0)
    r.append(dict(latitude=A[0], longitude=A[1], date="2015-08-01", ALT_instrument="Probe", VWC_instrument=None, depth_bottom=-9999, VWC=-9999))
    r.append(dict(latitude=A[0], longitude=A[1], date="2015-08-01", ALT_instrument=None, VWC_instrument="HydroSense II", depth_bottom=6, VWC=30))
    r.append(dict(latitude=A[0], longitude=A[1], date="2015-08-02", ALT_instrument=None, VWC_instrument="CD659 12cm rods", depth_bottom=12, VWC=40))
    r.append(dict(latitude=A[0], longitude=A[1], date="2016-08-01", ALT_instrument=None, VWC_instrument="HydroSense II", depth_bottom=6, VWC=90))
    r.append(dict(latitude=A[0], longitude=A[1], date="2016-08-01", ALT_instrument=None, VWC_instrument="HydroSense II", depth_bottom=20, VWC=50))
    r.append(dict(latitude=A[0], longitude=A[1], date="2015-08-01", ALT_instrument=None, VWC_instrument="GPR", depth_bottom=40, VWC=10))
    r.append(dict(latitude=A[0], longitude=A[1], date="2015-08-01", ALT_instrument=None, VWC_instrument="HydroSense I", depth_bottom=-9999, VWC=70))
    r.append(dict(latitude=A[0], longitude=A[1], date="2015-08-01", ALT_instrument=None, VWC_instrument="HydroSense I", depth_bottom=6, VWC=-9999))
    r.append(dict(latitude=B[0], longitude=B[1], date="2019-07-01", ALT_instrument=None, VWC_instrument=None, depth_bottom=10, VWC=20))
    # 위치 C: 하단 결측 비 GPR 행만(층 없음). 등록 부분 집합('15 m 안에 VWC 위치가 있는 셀')의 소속에는 세고 예측 변수는 결측이다
    r.append(dict(latitude=67.0, longitude=-152.0, date="2018-07-01", ALT_instrument=None, VWC_instrument="HydroSense I", depth_bottom=-9999, VWC=55))
    return pd.DataFrame(r)


def test_l_xe_e_vwc_layers_campaign_and_join():
    raw = _above_rows()
    assert "ALT" not in M.ABOVE_COLS and "ALT_err" not in M.ABOVE_COLS and set(raw.columns) == set(M.ABOVE_COLS)
    rows = M.vwc_rows(raw)
    assert len(rows) == 8 and (rows.cls == "").sum() == 2 and set(rows.cls) == {"shallow", "deep", "gpr", ""}
    yrs = M.alt_years(raw)
    assert yrs == {"65.0000_-150.0000": {2015}}
    lv = M.vwc_location_values(rows, yrs).set_index(["key", "cls"])
    kA = "65.0000_-150.0000"
    assert lv.loc[(kA, "shallow"), "vwc"] == pytest.approx(35.0) and bool(lv.loc[(kA, "shallow"), "campaign"]), "같은 해 행의 기기별 평균의 평균"
    assert lv.loc[(kA, "deep"), "vwc"] == pytest.approx(50.0) and not bool(lv.loc[(kA, "deep"), "campaign"]), "같은 해 행이 없으면 모든 행"
    assert lv.loc[(kA, "gpr"), "vwc"] == pytest.approx(10.0) and lv.loc[("66.0000_-151.0000", "shallow"), "vwc"] == pytest.approx(20.0)
    assert "67.0000_-152.0000" not in lv.index.get_level_values(0), "층 없는 위치는 층 값이 없다"
    la = M.vwc_locations_any(rows)
    assert set(la.key) == {kA, "66.0000_-151.0000", "67.0000_-152.0000"}, "소속 기준 위치는 층 없는 위치도 센다"
    xy = lambda lat, lon: np.column_stack([np.asarray(lon, float) * 1e5, np.asarray(lat, float) * 1e5])   # noqa: E731  1e-5° = 1 m(시험용)
    cells = pd.DataFrame(dict(loc_id=[1, 2, 3, 4], lat=[65.00001 + 10e-5, 65.00001 + 20e-5, 66.0, 67.0], lon=[-150.00001, -150.00001, -151.0, -152.0],
                              target="Alaska", block=["b1", "b1", "b2", "b3"]))
    cv = M.cell_vwc(cells, M.vwc_location_values(rows, yrs), xy_fn=xy, loc_any=la).set_index("loc_id")
    assert cv.loc[1, "vwc_shallow"] == pytest.approx(35.0) and cv.loc[1, "vwc_gpr"] == pytest.approx(10.0) and cv.loc[1, "n_loc_deep"] == 1
    assert np.isnan(cv.loc[2, "vwc_shallow"]) and cv.loc[2, "n_loc_shallow"] == 0 and cv.loc[2, "n_loc_any"] == 0, "15 m 밖"
    assert cv.loc[3, "vwc_shallow"] == pytest.approx(20.0) and np.isnan(cv.loc[3, "vwc_gpr"])
    assert cv.loc[4, "n_loc_any"] == 1 and cv.loc[4, "n_loc_shallow"] == 0 and np.isnan(cv.loc[4, "vwc_shallow"]), "층 없는 위치만 있는 셀: 소속만"
    assert list(cv.n_loc_any) == [1, 0, 1, 1]
    cv0 = M.cell_vwc(cells, M.vwc_location_values(rows, yrs), xy_fn=xy).set_index("loc_id")
    assert list(cv0.n_loc_any) == [3, 0, 1, 0], "loc_any 가 없으면 층 위치 수의 합"
    assert not set(M.COORD_READ) & set(RG.LABEL_COLS)


def test_l_xe_e_reads_no_label_columns(tmp_path):
    fb = tmp_path / "fb.csv"
    pd.DataFrame(dict(loc_id=[1, 2], lat=[65.0, 60.0], lon=[-150.0, -120.0], region=["ABoVE_AK", "ABoVE_CA"], block=["a", "b"], alt_cm=["독", "독"],
                      y=["독", "독"])).to_csv(fb, index=False)
    d = M.read_coords(fb)
    assert list(d.target) == ["Alaska", "Canada"] and "alt_cm" not in d and "y" not in d


def test_l_xe_e_residual_and_ratio():
    rng = np.random.RandomState(0)
    n = 360
    blk = np.repeat([f"b{i}" for i in range(12)], n // 12)
    s = np.repeat(np.arange(60) * 0.5 + 20, n // 60)                       # 묶음 60개(셀 6개씩), 블록 안에 중첩
    y = 1.3 * s + rng.randn(n) * 5
    e1, m1 = M.within_grid_residual(y, s, blk)
    e2, m2 = M.within_grid_residual(y - 0.7 * s, s, blk)
    assert np.allclose(e1, e2) and m1.all(), "P1 격자 안 잔차는 E1 과 무관하다"
    x = rng.uniform(5, 60, n)
    e = 0.4 * (x - x.mean()) + rng.randn(n) * 2
    X_ = np.column_stack([x, np.where(rng.rand(n) < 0.2, np.nan, rng.randn(n))])
    args = SimpleNamespace(cb_iters=20, threads=1)
    rows = M.xe_e_fit(e, X_, blk, "Alaska", "main", [0], args)
    r_ols = [r for r in rows if r["learner"] == "ols"][0]
    r_cb = [r for r in rows if r["learner"] == "catboost_lo"][0]
    assert r_ols["n_folds"] == 5 and r_ols["n_blocks"] == 12 and r_ols["expl_pct"] > 50 and r_cb["expl_pct"] > 20
    assert r_ols["ols_coef_vwc_shallow"] == pytest.approx(0.4, abs=0.05)
    rows0 = M.xe_e_fit(rng.randn(n), X_, blk, "Alaska", "main", [0], args)
    assert [r for r in rows0 if r["learner"] == "ols"][0]["expl_pct"] < 5, "관계가 없으면 설명 비율이 작다"
    one = M.xe_e_fit(e[:10], X_[:10], np.array(["b0"] * 10), "Canada", "gpr", [0], args)
    assert all(r["status"].startswith("계산 불가") for r in one)
    f1, K = M.block_folds(blk, ("xe-e", "Alaska", "main", "r", 0))
    f2, _ = M.block_folds(blk, ("xe-e", "Alaska", "main", "r", 0))
    assert K == 5 and np.array_equal(f1, f2) and all(len(set(f1[blk == b])) == 1 for b in np.unique(blk)), "같은 블록은 같은 묶음, seed 고정"
    p, beta = M.ols_fit_predict(np.array([[1.0], [2.0], [np.nan], [4.0]]), np.array([2.0, 4.0, 5.0, 8.0]), np.array([[3.0], [np.nan]]))
    assert len(beta) == 3 and np.isfinite(p).all()


def test_l_xe_e_guard_and_blind():
    with pytest.raises(SystemExit):                                        # XE-e 본 실행은 허용 표지가 있을 때만
        M.main(["--xe-e", "--feat", "none"] + NOMEM)
    assert M.blind_of("XE-e")[0] in XB.BLIND_LABELS


# ---------------------------------------------------------------- (m) 실행 보호(검토 반영)
def test_m_unfinal_dummy_and_requested(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        M.parse_args(["--feat", "none", "--allow-unfinal"])                # 본 실행
    assert M.parse_args(["--feat", "none", "--allow-unfinal", "--count-only"]).allow_unfinal
    assert M.parse_args(["--feat", "none", "--allow-unfinal", "--smoke", "--count-only"]).allow_unfinal
    a = M.parse_args(["--feat", "none", "--count-only"])
    with pytest.raises(SystemExit):
        M.check_feat_for_fit(a, "run")
    with pytest.raises(SystemExit):
        M.check_feat_for_fit(a, "smoke")
    M.check_feat_for_fit(a, "count")
    a1 = M.parse_args(["--feat", "none", "--count-only", "--variants", "x25"])
    M.check_feat_for_fit(a1, "run")
    sk = [dict(target="Alaska", split=-1, variant="xh", status=M.FINAL_REQUIRED_TXT), dict(target="Lena", split=-1, variant="add_M", status="군 제외(90 % 규칙)")]
    with pytest.raises(SystemExit):
        M.check_requested(a, sk, "run")
    with pytest.raises(SystemExit):
        M.check_requested(a, sk, "smoke")
    M.check_requested(a, sk[1:], "run")
    M.check_requested(a, sk, "count")
    fp, sha = _final_feat(tmp_path, monkeypatch, commit=False)
    meta = json.loads(fp.with_name("xe_feat_v1_meta.json").read_text())
    meta.pop("group_decisions")
    fp.with_name("xe_feat_v1_meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):
        M.FeatTable(fp).decisions("Alaska")
    meta["group_decisions"] = {"Alaska": {g: dict(included=True) for g in RG.H_GROUPS}}
    fp.with_name("xe_feat_v1_meta.json").write_text(json.dumps(meta))
    ft = M.FeatTable(fp)
    assert ft.decisions("Alaska")["H0"] is True
    with pytest.raises(RuntimeError):
        ft.decisions("Lena")


def test_m_final_committed():
    sha = "ab" * 32
    head = "# 계획\n본문 " + "cd" * 32 + "\n## 개정 이력\n- 2026-10-08 (XE xe_feat 최종판): sha256 `" + sha + "`\n"
    r = M.final_committed(sha, head_text=head)
    assert r["ok"] and "git HEAD" in r["how"]
    assert not M.final_committed("cd" * 32, head_text=head)["ok"], "개정 이력 절 밖의 해시는 커밋으로 보지 않는다"
    r2 = M.final_committed(sha, head_text="", file_text=head)
    assert r2["ok"] and "사본" in r2["how"]
    assert not M.final_committed("", head_text=head)["ok"] and not M.final_committed(sha, head_text="", file_text="")["ok"]
    plan = (ROOT / XB.PLAN_DOC).read_text() if (ROOT / XB.PLAN_DOC).exists() else ""
    assert M.revision_section(plan).startswith("## 개정 이력") or not plan


def test_m_sentence_same_contrast_same_n():
    A, B = M.MAIN_CONTRASTS[0][1], M.MAIN_CONTRASTS[1][1]
    pooled = {("XE-a", A, "격자 안", 500): _row(-0.3, -0.5, -0.1, "우세", 0.001), ("XE-a", A, "격자 안", -1): _row(-0.9, -2.0, 0.3, "미결정", 0.2),
              ("XE-b", B, "격자 안", -1): _row(0.2, -0.3, 0.6, "미결정", 0.3)}
    vr, ht = M.verdict_rows(pooled, [], None, {}, ["Alaska", "Lena", "Canada"])
    s_ = vr[vr.scope == "sentence"].iloc[0]
    assert s_.sentence_source_hyp == "XE-a" and s_.sentence_n == "500" and s_.sentence_delta == pytest.approx(-0.3)
    assert "0.30 cm 줄였으나" in s_.sentence and "통계적으로 구별되나" in s_.sentence and "0.90" not in s_.sentence
    assert s_.sentence_holm_p == pytest.approx(ht.loc[ht.label == "XE-a|n500", "p_holm"].iloc[0])
    pooled2 = dict(pooled)
    pooled2[("XE-b", B, "격자 안", 1000)] = _row(-1.2, -2.0, -0.6, "우세", 0.03)
    vr2, ht2 = M.verdict_rows(pooled2, [], None, {}, ["Alaska", "Lena", "Canada"])
    s2 = vr2[vr2.scope == "sentence"].iloc[0]
    assert s2.sentence_source_hyp == "XE-b" and s2.sentence_n == "1000" and "1.20 cm 작았다" in s2.sentence
    hb, ha = ht2.loc[ht2.label == "XE-b|n1000", "p_holm"].iloc[0], ht2.loc[ht2.label == "XE-a|n500", "p_holm"].iloc[0]
    assert s2.sentence_holm_p == pytest.approx(hb) and s2.sentence_holm_p_marker == pytest.approx(max(hb, ha))
    assert ("보정 전 유의" in s2.sentence) == (max(hb, ha) >= 0.05)
    # 첫째 문장(두 가설 모두 우세): XE-b 는 보정 뒤 유의, XE-a 는 보정 뒤 유의하지 않음 → 표지를 붙인다(두 Holm p 가운데 큰 값)
    pooled3 = {("XE-a", A, "격자 안", 500): _row(-0.3, -0.5, -0.1, "우세", 0.04), ("XE-b", B, "격자 안", 1000): _row(-1.2, -2.0, -0.6, "우세", 0.0001)}
    vr3b, ht3 = M.verdict_rows(pooled3, [], None, {}, ["Alaska", "Lena", "Canada"])
    s3 = vr3b[vr3b.scope == "sentence"].iloc[0]
    assert ht3.loc[ht3.label == "XE-b|n1000", "p_holm"].iloc[0] < 0.05 < ht3.loc[ht3.label == "XE-a|n500", "p_holm"].iloc[0]
    assert "보정 전 유의" in s3.sentence and s3.sentence_holm_p < 0.05 <= s3.sentence_holm_p_marker and "XE-a|n500" in s3.sentence_holm_p_other
    vr3, _ = M.verdict_rows(pooled, [], None, {}, ["Alaska"], gate=dict(passed=False, status="관문 실패(키 실패 1)"))
    assert vr3[vr3.scope == "verdict"].verdict.str.startswith("판정 불가(관문").all()
    assert "판정할 수 없었다" in vr3[vr3.scope == "sentence"].sentence.iloc[0]
    assert M.row_label("XE-a", "주", 200) == ("XE-c", "보조") and M.row_label("XE-b", "주", -1) == ("XE-b", "주")
    assert M.row_label("XE-a0", "서술", 200) == ("XE-c", "보조") and M.row_label("XE-d", "서술", 200) == ("XE-d", "서술")


def test_m_blind_vocabulary_and_si_status():
    for hyp, v, c in (("XE-a", "xh", "R1(λ cv, xh)−R1(λ cv, x25)"), ("XE-c", "xh0", "R1(λ cv, xh0)−R1(λ cv, x25)"), ("XE-c", "xh", "D0(xh)−D0(x25)"),
                      ("XE-c", "xt2", "R1(λ cv, xt2)−R1(λ cv, x25)"), ("XE-d", "add_M", "R1(λ cv, add_M)−P1"), ("XE-d", "sg250", "R1(λ cv, sg250)−R1(λ cv, x25)")):
        lab, why = M.blind_of(hyp, v, c)
        assert lab in XB.BLIND_LABELS and why and lab != why
    assert M.blind_of("XE-c", "xh0", "x")[0] == "비맹검 부분 포함" and "xh0" in M.blind_of("XE-c", "xh0", "x")[1]
    a = M.parse_args(["--stage", "r3", "--feat", "none", "--count-only"])
    st = M.variant_status(a, ["cov_AK", "cov_LE", "sg250"]).set_index(["variant", "target"])
    assert st.loc[("cov_AK", "Alaska"), "status"] == RG.COVER_NOT_RUN and st.loc[("cov_LE", "Lena"), "status"] == RG.COVER_NOT_RUN
    assert "cfvo" in st.loc[("sg250", "Alaska"), "note"] and st.loc[("sg250", "Alaska"), "status"] == "실행"


def test_m_smoke_env_and_memory(monkeypatch):
    s = M.parse_args(["--smoke", "--feat", "none", "--count-only"])
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)))
    with pytest.raises(SystemExit):
        M.check_smoke_env(s, "smoke")
    M.check_smoke_env(s, "count")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(4)))
    M.check_smoke_env(s, "smoke")
    seen = []
    monkeypatch.setattr(XB, "require_memory", lambda g, wait_s=0.0, poll_s=60.0: seen.append((g, wait_s)) or 40.0)
    a = M.parse_args(["--feat", "none", "--count-only", "--max-mem-gb", "0"])
    r = M.local_resources(a)
    assert seen == [(30.0, 1800.0)] and r["mem_available_gb"] == 40.0 and not r["watchdog"]
    a.PERMIT = True
    assert M.local_resources(a)["mem_available_gb"] is None and len(seen) == 1


def test_m_payload_optional_files():
    for f in ("data/processed/xe/xe_feat_v1.csv", "data/processed/xe/xe_feat_v1_meta.json", "data/processed/xe/xe_feat_v1_final.json",
              "data/processed/xbatch/XE_hires_covariates/inputs/xe_vwc_v1.csv", XB.PLAN_DOC):
        assert f in XB.PAYLOAD_OPTIONAL, f
    assert XB.PAYLOAD_REQUIRES["data/processed/xe/xe_feat_v1.csv"] == ("data/processed/xe/xe_feat_v1_meta.json",)


# ---------------------------------------------------------------- (k) 실제 자료(세기 범주)
@REAL
def test_k_count_only_prints_no_label_statistics(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    rc = M.main(["--count-only", "--stage", "r1b", "--targets", "Canada", "--splits", "3", "--feat", "none", "--out-dir", str(tmp_path / "XE")] + NOMEM)
    out = capsys.readouterr().out
    assert rc == 0 and "[count-only]" in out and "총 적합" in out
    assert not XB.FORBIDDEN_OUT.search(out), "세기 화면에 RMSE·Δ·판정이 있다"
    a = M.parse_args(["--stage", "r1b", "--targets", "Canada", "--splits", "3", "--feat", "none", "--count-only"])
    stats = []
    for sp in (1, 2, 3):
        c, _, _ = M.build_ctx(a, "Canada", sp, "x25")
        stats += [c.E0, c.meta["E_A"], float(np.mean(c.yA)), float(np.mean(c.yB)), float(np.std(c.yB)), float(np.sqrt(np.mean((c.E0 * c.sB - c.yB) ** 2)))]
    for v in stats:                                                       # 라벨 유래 통계(계수, 평균, P0 RMSE)가 화면에 없다
        for fmt in ("{:.3f}", "{:.4f}", "{:.2f}"):
            s_ = fmt.format(v)
            if len(s_.replace("-", "").replace(".", "")) >= 4:
                assert s_ not in out, f"세기 화면에 라벨 유래 값 {s_} 가 있다"
    cnt = pd.read_csv(tmp_path / "XE" / "xe_r1b_count.csv")
    assert set(cnt.variant) == {"x25", "xh0", "xt2"} and cnt.fit_total.gt(0).all() and set(cnt.split) <= {1, 2, 3}
    assert not [c_ for c_ in cnt.columns if re.search("rmse|delta|bias|alt|E0|E_A", c_)]


@REAL
def test_k_feature_join_matches_context_order(tmp_path):
    """특징 표의 loc_id 결합이 h54.build_rctx 의 A·채점 셀 순서와 맞는다(특징 값 = loc_id 로 둔 합성 표, 라벨 값은 비교만 하고 쓰지 않는다)."""
    pts = M.read_coords(ROOT / "data/processed/fidelity_base_v3.csv")      # 좌표·블록 열만(추출 모듈 없이)
    can = pts[pts.target == "Canada"].sort_values("loc_id")
    feat = pd.DataFrame(dict(loc_id=can.loc_id.values, target="Canada"))
    for c in RG.H_COLS + RG.PX_COLS:
        feat[c] = can.loc_id.values.astype(float)
    fp = tmp_path / "xe_feat_v1.csv"
    feat.to_csv(fp, index=False)
    dec = {t: {g: dict(included=True, finite_frac=1.0) for g in RG.H_GROUPS} for t in RG.TARGETS}
    (tmp_path / "xe_feat_v1_meta.json").write_text(json.dumps(dict(group_decisions=dec)))
    a = M.parse_args(["--stage", "r1b", "--targets", "Canada", "--splits", "3", "--feat", str(fp), "--count-only"])
    c, cols, info = M.build_ctx(a, "Canada", 3, "xt2")
    _, D = W.get_data(a.h)
    t_idx = D.target_idx("Canada")
    A_idx, B_idx = XB.half_split_blocks(D.df, t_idx, 3)
    evB = B_idx[XB.eval_mask(D.df.iloc[B_idx])]
    XA, XBm = c.XAf("xt2"), c.XBf("xt2")
    assert XA.shape[1] == len(cols) == 25 + 16 and np.array_equal(XA[:, :25], c.XA, equal_nan=True)
    assert np.array_equal(XA[:, 25], D.df.loc_id.values[A_idx].astype(np.float32)) and np.array_equal(XBm[:, -1], D.df.loc_id.values[evB].astype(np.float32))
