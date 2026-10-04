"""XG(x_product_comparison.py)와 XH(x_validation_ladder.py, src/polar/cv_schemes.py) 단위 시험.
계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.7(구현·시험 행)과 2.8(구현·시험 행), 1절 '누설 시험'.

합성 자료 시험(라벨 값을 화면에 쓰지 않는다)
XG (a) 합성 래스터 표집: 최근접 화소 값, 3 × 3 평균, 결측값(nodata), 래스터 밖 점, /vsizip/ 경로가 같은 값을 낸다.
   (b) 추출 기록: zip md5 를 대조하고(다르면 중단, 관문 g3), 단위(값 범위·명령행)·좌표계·결측값을 기록한다.
   (c) 연도 정합 규칙(겹친 연도 평균, 겹침 없으면 기간 평균과 match_flag, 정적)과 최근접 색인(오름·내림 좌표축).
   (d) 누설 마스크: 학습 지점에서 d km 안(≤ d)의 셀이 있는 블록을 빼고, L0 제품은 빈 집합이다.
   (e) 재채점: 마스크로 블록 열을 고른 저장소가 셀에서 직접 만든 저장소와 같고, 그 위의 대비도 같다.
   (f) P1@{p} 가 P1 과 같은 수축 식(h42.shrink, κ 10)을 쓴다(제품 = √TDD 이면 P1@{p} = P1, P0@{p} = P0).
   (g) 누설: 제품 키는 선택 밖 A 라벨과 B 라벨을 바꿔도 같고, 선택 라벨 하나를 바꾸면 달라진다(음성 대조).
   (h) WRAPUP 1.1 (a)–(d) 문구 갈래, 두 마스크 결합('마스크 의존'), 해석 문장(CALM 절, 학습 자료 중복 가능, 등록 이탈 표지).
   (i) 확인 대비 6개의 합성 저장소 경로: Holm m = 6, 모든 행에 '등록 이탈(WRAPUP 10)', L0 제품은 두 마스크 판정이 같다.
   (j) 셀 단위 민감도 단위(h54 전이 단위 하위 클래스)의 누설 시험(h54 시험 b·d·m 형식)과 셀 단위 5 km 저장소.
XH (k) 단 W1R·W1S·W1B 의 묶음이 대상 셀을 덮고 겹치지 않으며, 학습·채점 교집합 0, 지점·블록 분리. W1B 는 h41 W1 묶음과 같다.
   (l) kNNDM 색인: 만들기·쓰기·sha256 대조(바꾼 파일 거부), W1K 학습·채점 교집합 0.
   (m) cv_schemes 의 kNNDM·묶음 함수가 m1_cv_scheme_comparison.py 의 함수와 같은 입력에서 같은 묶음을 낸다.
   (n) 예측 영역의 결정성(상자 격자, 격자 파일, m1 정의 세분)과 육지 ∩ MAAT < 0 조건.
   (o) models_w1x 가 h41.models_pooled 의 PSw·D0w·RSw 와 같은 예측을 낸다(결정적 대체 적합기).
   (p) 누설: W1 모형은 채점 셀 라벨을 쓰지 않는다(채점 라벨을 바꿔도 예측이 같다, catboost_lo·능형).
   (q) 집계: 합성 조각에서 ΔΔ(XH-1)가 반복·seed 평균 블록 SSE 로 손 계산한 값과 같고, 3지역 평균은 지역 점 추정의 평균이다.
   (r) 명령행: 두 모듈에 --count-only, --smoke, --shard, --allow-local 이 있고, 허용 표지 없는 본 실행을 자료를 읽기 전에 거부한다.
검토 반영(2026-10-04 적대 검토 결함 14건, impl_notes 의 '검토 반영' 절)
XG (u) 판정 표의 마스크별 보조 판정·의존 표지·풀 표지·HK 조건, 문장 표지와 열세 지역의 두 마스크 합집합(결함 5).
   (v) CALM 학습 제품의 XG-4 풀에서 러시아 W·E 제외와 별표 참고값(결함 6), L2 제품의 누설 점검 불가·SI 전용 열(결함 7), XG-4r 문장(결함 8).
   (w) ρ·s 대체 채점 셀 비율(fill_frac_B)과 표지, 대체 셀 블록을 두 팔에서 같이 뺀 민감도(결함 4).
   (x) 작업 R3 본 실행의 입력 확인과 조각 설정의 입력 sha256(결함 2), 본 봉인 폴더의 재표집 10,000회(결함 12), GDAL 캐시·임시 폴더(결함 9),
       XG 의 --gate-level 제거(결함 14).
   (y) 두 모듈의 묶음 점검(PAYLOAD_EXTRA → payload_manifest(extra=), 공용 PAYLOAD_OPTIONAL, 결함 3), 환경 변수 기반 상주 메모리 감시(결함 1),
       집계·관문 경로의 가용 메모리 대기(결함 13), W1K 색인 사전 점검.
XH (u) 관문 (2) 실패 시 알래스카 연속성 행 제거와 관문 요약의 봉인 메타 기록(결함 10), 관문 (1)의 1절 허용 오차 단독 규칙(결함 11).
실제 자료 시험(세기 범주)
   (s) --count-only 의 화면 출력에 출력 제한 패턴이 없고, 라벨 값을 바꿔도 출력이 같다(라벨에서 나온 통계를 쓰지 않는다).
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c <코어 4개> python3 -m pytest -q tests/test_x_xg_xh.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import contextlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "3_deep_learning"))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


XB = _load("xbatch_core", "scripts/3_deep_learning/xbatch_core.py")
XG = _load("x_product_comparison", "scripts/3_deep_learning/x_product_comparison.py")
XH = _load("x_validation_ladder", "scripts/3_deep_learning/x_validation_ladder.py")
H, X, W, H4 = XB.H, XB.X, XB.W, XB.H4
H41 = XB.frozen("h41")
from polar import cv_schemes as CV                                        # noqa: E402

REAL = pytest.mark.skipif(not (ROOT / "data/processed/fidelity_base_v3.csv").exists()
                          or not (ROOT / "results/rescale_lg/data/processed/lg/shards").exists(), reason="실제 자료가 없다")


# ================================================================ 합성 도우미
def write_tif(path, arr, lon0, lat0, res, nodata=-9999.0, crs="EPSG:4326"):
    import rasterio
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32", crs=crs,
                       transform=from_origin(lon0, lat0, res, res), nodata=nodata, tiled=True, blockxsize=16, blockysize=16) as r:
        r.write(arr.astype(np.float32), 1)


def synth_D(seed=0, ns=200, nA=60, nB=40, b_mult=None):
    """합성 자료(제품 키 계산용): 원천·A·B 셀의 y, s, loc_id, 제품 열 b."""
    rng = np.random.RandomState(seed)
    n = ns + nA + nB
    s = rng.uniform(20, 40, n)
    y = 1.5 * s + 2 * rng.randn(n)
    b = s.copy() if b_mult is None else b_mult * s + rng.randn(n)
    b[rng.choice(n, 5, replace=False)] = np.nan                         # 결측 제품 값(ρ·s 로 바뀐다)
    df = pd.DataFrame(dict(loc_id=np.arange(n), y=y, s=s, b=b))
    return SimpleNamespace(df=df), np.arange(ns), np.arange(ns, ns + nA), np.arange(ns + nA, n)


def fake_V(seed=0, n=600, nblk=24):
    """합성 VData(h41 의 필드). 지역 R 한 곳, 블록 24개, 좌표는 블록 안에 흩어 둔다."""
    rng = np.random.RandomState(seed)
    blk = np.sort(rng.randint(0, nblk, n))
    lat = 65 + (blk // 6) * 0.5 + rng.uniform(0, 0.5, n)
    lon = -150 + (blk % 6) * 0.5 + rng.uniform(0, 0.5, n)
    s = rng.uniform(20, 40, n); Xf = rng.randn(n, len(XB.FEATS)).astype(np.float32)
    y = 1.5 * s + 3 * Xf[:, 1] + rng.randn(n)
    V = SimpleNamespace(N=n, lat=lat, lon=lon, block=(1000 + blk).astype(np.int64), loc_id=np.arange(n, dtype=np.int64) + 7,
                        macro=np.array(["R"] * n, object), y=y, s=s, X=Xf, trainable=np.ones(n, bool),
                        score=rng.rand(n) > 0.05, D=SimpleNamespace(target_idx=lambda r: np.arange(n)))
    V.take = lambda idx: H41.Cells(V.X[idx], V.y[idx], V.s[idx], np.full(len(idx), np.nan), np.full(len(idx), np.nan), np.full(len(idx), np.nan),
                                   np.full(len(idx), np.nan))
    return V


class StubFitter:
    """결정적 대체 적합기: 예측 = 학습 목표 평균 + seed + 0.01·학습기 번호 + X[:, 1] 의 최소제곱 기울기 × X[:, 1]."""

    def fit(self, learner, axis, build, seed, preds, info=None, init=None, iters=None, cont=False):
        Xtr, ytr, _ = build()
        x = np.asarray(Xtr, float)[:, 1]
        bet = float(np.polyfit(x, np.asarray(ytr, float), 1)[0])
        off = float(np.mean(ytr)) + float(seed) + 0.01 * ["catboost_lo", "catboost", "rf"].index(learner)
        return None, [off + bet * np.asarray(P, float)[:, 1] for P in preds]


# ================================================================ XG
def test_xg_a_raster_sampling(tmp_path):
    arr = (np.arange(20)[:, None] * 100 + np.arange(30)[None, :]).astype(float)
    arr[5, 7] = -9999.0
    tif = tmp_path / "t.tif"
    write_tif(tif, arr, -150.0, 70.0, 0.1)
    rows, cols = np.array([0, 5, 10, 19, 3]), np.array([0, 7, 15, 29, 4])
    lat = 70.0 - (rows + 0.5) * 0.1; lon = -150.0 + (cols + 0.5) * 0.1
    cells = pd.DataFrame(dict(lat=np.r_[lat, 80.0], lon=np.r_[lon, -150.05]))
    v, v3, dist, rec = XG._tif_points(str(tif), cells)
    want = arr[rows, cols].astype(float); want[1] = np.nan
    assert np.allclose(v[:5], want, equal_nan=True) and np.isnan(v[5]), "최근접 화소·nodata·래스터 밖"
    A = np.where(arr == -9999.0, np.nan, arr)
    m3 = [np.nanmean(A[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]) for r, c in zip(rows, cols)]
    assert np.allclose(v3[:5], m3), "3 × 3 평균(결측 제외, 가장자리 창)"
    assert np.all(dist[:5] < 0.01) and rec["crs"].endswith("4326") and rec["nodata"] == -9999.0
    z = tmp_path / "t.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        f.write(tif, "dir/t.tif")
    vz, _, _, _ = XG._tif_points(f"/vsizip/{z}/dir/t.tif", cells)
    assert np.array_equal(np.nan_to_num(vz, nan=-1), np.nan_to_num(v, nan=-1)), "/vsizip/ 경로가 같은 값을 낸다"


def test_xg_b_extract_record_md5_units(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; (raw / "fake").mkdir(parents=True)
    arr = 0.5 + np.random.RandomState(0).rand(20, 30)                    # m 단위 값(중앙값 < 10)
    tif = tmp_path / "ALT_Baseline.tif"
    write_tif(tif, arr, -150.0, 70.0, 0.1)
    z = raw / "fake" / "ALT_Rasters.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        f.write(tif, "ALT_Baseline.tif")
    monkeypatch.setattr(XG, "RAW", raw)
    spec = dict(kind="zip_tif", folder="fake", period=(2000, 2014), rule="static", zips={"ALT_Rasters.zip": XG.md5_file(z)}, member="ALT_Baseline.tif")
    cells = pd.DataFrame(dict(lat=[69.95, 69.05], lon=[-149.95, -147.05]))
    V, V3, years, dist, rec = XG.extract_tif(spec, cells)
    assert years == [None] and rec["units"]["units"] == "m" and rec["units"]["how"].startswith("값 범위")
    assert np.allclose(V[:, 0], 100.0 * arr[[0, 9], [0, 29]]), "m → cm"
    assert rec["md5"]["ALT_Rasters.zip"]["ok"] and "crs" in rec and rec["nodata"] == -9999.0
    V2, _, _, _, rec2 = XG.extract_tif(spec, cells, override="cm")
    assert np.allclose(V2[:, 0], arr[[0, 9], [0, 29]]) and rec2["units"]["how"] == "명령행"
    bad = dict(spec, zips={"ALT_Rasters.zip": "0" * 32})
    with pytest.raises(SystemExit):
        XG.extract_tif(bad, cells)
    mult, r = XG.infer_units(np.array([50.0, 120.0, np.nan]))
    assert mult == 1.0 and r["units"] == "cm"
    assert XG.infer_units(np.array([1.0]), declared="metres")[0] == 100.0


def test_xg_c_year_rule_and_nearest_index():
    V = np.array([[1.0, 2.0, 3.0, np.nan], [10.0, 20.0, 30.0, 40.0], [np.nan] * 4, [5.0, 5.0, 5.0, 5.0]])
    years = [2000, 2001, 2002, 2003]
    val, flag, ny = XG.year_rule(V, years, np.array([2001, 1990, 2001, np.nan]), np.array([2003, 1995, 2001, np.nan]), "matched")
    assert np.allclose(val[:2], [2.5, 25.0]) and np.isnan(val[2]) and val[3] == 5.0
    assert list(flag) == ["overlap", "period_mean", "missing", "period_mean"] and list(ny[:2]) == [2, 4]
    sv, sf, _ = XG.year_rule(V, years, None, None, "static")
    assert np.allclose(sv[:2], [2.0, 25.0]) and sf[0] == "static"
    ax = np.arange(85.0, 25.0, -0.01)
    q = np.array([84.996, 70.0049, 25.004, 50.0])
    assert np.array_equal(XG.nearest_index(ax, q), np.array([np.argmin(np.abs(ax - v)) for v in q]))
    ax2 = np.arange(-180.0, 180.0, 0.25)
    q2 = np.array([-179.9, 0.13, 179.0])
    assert np.array_equal(XG.nearest_index(ax2, q2), np.array([np.argmin(np.abs(ax2 - v)) for v in q2]))


def test_xg_d_block_mask():
    lat = np.array([65.0, 65.0, 66.0, 66.0, 67.0])
    lon = np.array([-150.0, -150.2, -150.0, -150.1, -150.0])
    blk = np.array([1, 1, 2, 2, 3])
    km_per_deg = 6371.0 * np.pi / 180.0
    train = np.array([[65.0 + 3.0 / km_per_deg, -150.0], [66.0 + 10.0 / km_per_deg, -150.0]])   # 블록 1 에서 3 km, 블록 2 에서 10 km
    m5, mind, dist = XG.block_mask(lat, lon, blk, train, 5.0)
    m25, _, _ = XG.block_mask(lat, lon, blk, train, 25.0)
    assert m5 == {"1"} and m25 == {"1", "2"} and abs(mind["1"] - 3.0) < 0.01
    m_eq, _, _ = XG.block_mask(lat, lon, blk, train, float(mind["2"]))
    assert "2" in m_eq, "d km 안(≤ d)은 가린다"
    assert XG.block_mask(lat, lon, blk, np.zeros((0, 2)), 25.0)[0] == set(), "L0 제품(학습 지점 없음)은 빈 집합"
    assert XG.masked_blocks(None, "Lena|x", "cci5y", "blk25") == set() and XG.masked_blocks({}, "Lena|x", "wei", "none") == set()
    with pytest.raises(SystemExit):
        XG.masked_blocks(None, "Lena|x", "wei", "blk5")


def test_xg_e_mask_store_rescore():
    rng = np.random.RandomState(3)
    n = 120
    blk = np.repeat([f"b{j}" for j in range(12)], n // 12)
    y = rng.uniform(30, 90, n)
    keys = [H.P0_KEY] + [("R1", "catboost_lo", "1", "cell", 10, d, s, 0.25) for d in range(3) for s in (0, 1)]
    preds = {k: y + rng.randn(n) * (3 + i % 3) for i, k in enumerate(keys)}
    drop = {"b2", "b7"}
    full, sub = H4.BlockStore("T|x", 1, blk), H4.BlockStore("T|x", 1, blk[~np.isin(blk, list(drop))])
    for k in keys:
        full.add(k, y, preds[k])
        sub.add(k, y[~np.isin(blk, list(drop))], preds[k][~np.isin(blk, list(drop))])
    ms = XG.mask_store(full, drop)
    assert list(ms.blocks) == list(sub.blocks) and np.array_equal(ms.ncell, sub.ncell)
    for k in keys:
        assert np.allclose(ms.get(k)[0], sub.get(k)[0]) and np.array_equal(ms.get(k)[1], sub.get(k)[1])
    assert XG.mask_store(full, set(blk)) is None
    a = SimpleNamespace(nboot=200)
    units = [dict(split=1, dup_of=-1, valid=True)]
    tm1, nb1 = XG.make_tm(a, "T|x", {1: full}, units, drop)
    tm2 = W.make_tm("T|x", {1: sub}, units, 200, False)
    g = ("R1", "catboost_lo", "1", "cell", 10, 0.25)
    r1, r2 = H.contrast(tm1, g, H.P0_GRP, return_dist=True), H.contrast(tm2, g, H.P0_GRP, return_dist=True)
    assert nb1 == {1: 10} and r1["delta"] == r2["delta"] and np.array_equal(r1["dist"], r2["dist"])


def test_xg_f_product_key_uses_p1_shrink(monkeypatch):
    monkeypatch.setitem(XG.PRODUCTS, "psyn", dict(kind="column", column="b", period=(2000, 2000), rule="static", leak="L0", train=None,
                                                 label="합성", alt_def="합성"))
    D, src, A, B = synth_D(b_mult=None)
    D.df.loc[D.df.b.isna(), "b"] = D.df.s[D.df.b.isna()]               # 제품 = √TDD
    keys, rec = XG.product_keys(D, src, A, B, "psyn", {}, "T", "x", 1, (0, 3, 10, 40, -1), 3)
    df = D.df
    E0 = H.ls_E(df.y.values[src], df.s.values[src])
    sB = df.s.values[B]
    assert np.allclose(keys[("P0@psyn", "none", "1", "cell", 0, 0, -1, 0.0)], E0 * sB) and np.allclose(keys[("B:psyn_raw", "none", "1", "cell", 0, 0, -1, 0.0)], sB)
    c = SimpleNamespace(E0=E0, yA=df.y.values[A], sA=df.s.values[A])
    for n, d in H.cells_of([0, 3, 10, 40, -1], 3, len(A)):
        sel = H.draw_cells("T", "x", 1, n, d, len(A))
        E1 = X.shrink(H.ls_E(c.yA[sel], c.sA[sel]), E0, len(sel), 10.0) if len(sel) else E0
        assert np.allclose(keys[("P1@psyn", "none", "1", "cell", n, d, -1, 0.0)], E1 * sB), f"P1@p ≠ P1 (n {n})"
    a0, b0 = X.affine_ls(df.b.values[src], df.y.values[src])
    assert np.allclose(keys[("B:psyn_aff", "none", "1", "cell", 0, 0, -1, 0.0)], a0 + b0 * sB)
    k2, rec2 = XG.product_keys(D, src, A, B, "psyn", {}, "T", "r", 1, (10,), 3, inregion=True)
    assert list(k2) == [("B:psyn_raw", "none", "1", "cell", 0, 0, -1, 0.0)]


def test_xg_g_product_key_leakage(monkeypatch):
    monkeypatch.setitem(XG.PRODUCTS, "psyn", dict(kind="column", column="b", period=(2000, 2000), rule="static", leak="L0", train=None,
                                                 label="합성", alt_def="합성"))
    D, src, A, B = synth_D(seed=5, b_mult=1.3)
    grid, draws = (0, 3, 10, 40, -1), 3
    k1, _ = XG.product_keys(D, src, A, B, "psyn", {}, "T", "x", 2, (0, 3, 10, 40), draws)
    sel_union = np.unique(np.concatenate([H.draw_cells("T", "x", 2, n, d, len(A)) for n, d in H.cells_of([3, 10, 40], draws, len(A))]))
    assert 0 < len(sel_union) < len(A)
    D2 = SimpleNamespace(df=D.df.copy())
    rng = np.random.RandomState(9)
    outA = np.setdiff1d(np.arange(len(A)), sel_union)
    D2.df.loc[A[outA], "y"] = rng.uniform(1, 500, len(outA))
    D2.df.loc[B, "y"] = rng.uniform(1, 500, len(B))
    k2, _ = XG.product_keys(D2, src, A, B, "psyn", {}, "T", "x", 2, (0, 3, 10, 40), draws)
    assert set(k1) == set(k2) and all(np.array_equal(k1[k], k2[k]) for k in k1), "선택 밖 A·B 라벨이 제품 키에 새었다"
    D3 = SimpleNamespace(df=D.df.copy())
    D3.df.loc[A[sel_union[0]], "y"] += 50.0
    k3, _ = XG.product_keys(D3, src, A, B, "psyn", {}, "T", "x", 2, (0, 3, 10, 40), draws)
    assert any(not np.array_equal(k1[k], k3[k]) for k in k1), "선택 라벨을 바꿔도 같다(시험이 둔하다)"
    full, _ = XG.product_keys(D, src, A, B, "psyn", {}, "T", "x", 2, grid, draws)
    assert ("P1@psyn", "none", "1", "cell", -1, 0, -1, 0.0) in full


def test_xg_h_wording_masks_sentences():
    assert XG.wording("우세", 0.01, np.nan) == ("우세", "")
    assert XG.wording("열세", 0.06, np.nan) == ("미결정", XB.UNCORRECTED_TXT)
    assert XG.wording("동등", np.nan, 0.01) == ("동등", "") and XG.wording("동등", np.nan, 0.03) == ("미결정", "보정 전 동등")
    assert XG.wording("미결정", 0.5, 0.5) == ("미결정", "") and XG.wording("판정 불가", 1.0, 1.0)[0] == "판정 불가"
    assert XG.combine_masks(["우세", "우세"]) == ("우세", "") and XG.combine_masks(["우세", "미결정"]) == ("미결정", "마스크 의존")
    assert XG.combine_masks(["우세", "열세"]) == ("미결정", "마스크 의존") and XG.combine_masks(["동등", "판정 불가"]) == ("판정 불가", "마스크 의존")
    s1 = XG.sentence("XG-1", "열세", "wei", 2.345, 2, 2, None, [], [])
    assert "2.35 cm 컸다" in s1 and "CALM 지점도 학습에" in s1 and "—" not in s1
    s2 = XG.sentence("XG-1", "열세", "cci5y", 1.0, 4, 4, None, [], [])
    assert "CALM" not in s2, "L0 제품에는 CALM 학습 문장을 붙이지 않는다"
    s3 = XG.sentence("XG-2", "열세", "wei", 0.8, 2, 2, 10, [], [("Lena|x", 1.2, 0.3, 2.0)])
    assert XG.OVERLAP_TXT in s3 and "지역 Lena|x 에서는 오차가 컸다" in s3
    s4 = XG.sentence("XG-3", "우세", "cci5y", 0.8, 4, 4, -1, ["마스크 의존"], [])
    assert "라벨 전량을 쓴" in s4 and XG.OVERLAP_TXT not in s4 and "(마스크 의존)" in s4


def _synth_rescore(nboot=300, products=("wei", "cci5y"), with_p1=False):
    """확인 대비 6개에 필요한 키를 가진 합성 저장소의 재채점 상태(Rescore 의 필드만 채운다). with_p1 이면 서술 대비(XG-4)의 P1 키도 넣는다.
    products 에 aalto 가 있으면 B:aalto_raw·P1@aalto 키를 넣는다(L2 제품 행의 누설 점검 열 시험)."""
    rng = np.random.RandomState(11)
    R = XG.Rescore.__new__(XG.Rescore)
    R.a = SimpleNamespace(nboot=nboot, smoke=False)
    R.products = list(products); R.pv = {}; R.recs = []; R._tm = {}; R.fill = {}
    R.raw, R.units, R.masks = {}, {}, {}
    names = sorted(set(XG.POOL_W) | set(XG.POOL_C))
    levels = {"wei": 31.0, "cci5y": 33.0, "aalto": 34.0}
    for nm in names:
        by = {}
        for sp in (1, 2, 3):
            nb = 12
            st = H4.BlockStore._from_arrays(nm, sp, np.array([f"{nm[:2]}{sp}b{j:02d}" for j in range(nb)]), rng.randint(3, 20, nb), [], [], [], {})
            cnt = st.ncell

            def put(key, level):
                st.add_sse(key, cnt * (level + rng.gamma(2.0, 1.0, nb)) ** 2, cnt)
            put(H.P0_KEY, 30.0)
            for p in products:
                lev = levels[p]
                put((f"B:{p}_raw", "none", "1", "cell", 0, 0, -1, 0.0), lev)
                for n in (10, -1):
                    for d in (range(5) if n == 10 else [0]):
                        put((f"P1@{p}", "none", "1", "cell", n, d, -1, 0.0), lev - 3)
            for n in (10, -1):
                for d in (range(5) if n == 10 else [0]):
                    if with_p1:
                        put(("P1", "none", "1", "cell", n, d, -1, 0.0), 29.0)
                    for s in (0, 1):
                        put(("R1", "catboost_lo", "1", "cell", n, d, s, 0.25), 26.0)
            by[sp] = st
        R.raw[nm] = by
        R.units[nm] = [dict(split=sp, dup_of=-1, valid=True) for sp in by]
        blocks = sorted(set().union(*[set(st.blocks) for st in by.values()]))
        for p in ("wei", "aalto"):
            R.masks[(nm, p, 5.0)] = set(blocks[:2]); R.masks[(nm, p, 25.0)] = set(blocks[:5])
    R.load = lambda name: R.raw.get(name, {})
    return R


def test_xg_i_confirm_pipeline_synthetic():
    R = _synth_rescore()
    tests, rows = XG.confirm_tests(R)
    assert list(tests.hypothesis) == [h["id"] for h in XG.CONFIRM] and (tests.holm_m == 6).all()
    assert (tests.deviation == XG.DEVIATION).all() and (rows.deviation == XG.DEVIATION).all()
    assert set(tests.branch) <= set(XB.BRANCHES)
    cc = tests[tests["product"] == "cci5y"]
    assert (cc.verdict4_blk5 == cc.verdict4_blk25).all() and (cc.mask_dependence == "").all(), "L0 제품은 두 마스크가 같다"
    w = rows[(rows.hypothesis == "XG-1w") & (rows.scope == "region")]
    assert (w[w["mask"] == "blk25"].nb_split_min_postmask.values <= w[w["mask"] == "blk5"].nb_split_min_premask.values).all()
    assert (w[w["mask"] == "blk25"].nb_split_min_postmask.values < 12).any()
    assert all("등록 이탈(WRAPUP 10)" in s for s in tests.sentence)
    ph = XB.holm([1e-4, 0.03, 0.2, None, 0.01, 0.5], 6)
    assert ph[3] == 1.0 and abs(ph[0] - 6e-4) < 1e-12


def test_xg_u_tests_table_carries_mask_tags():
    """검토 반영(결함 5): 판정 표에 마스크별 보조 판정·의존 표지·풀 표지·HK 조건이 있고, 문장 표지와 열세 지역은 두 마스크의 합집합이다."""
    R = _synth_rescore()
    tests, rows = XG.confirm_tests(R)
    for m in XG.CO_PRIMARY:
        for c in ("verdict4_d10", "verdict4_rel", "limit_dependence", "ci_dependence", "pool_label", "few_block_regions", "mask_few_blocks",
                  "fill_frac_B", "rmse_p0_postmask", "n_ci_regions", "region_general"):
            assert f"{c}_{m}" in tests.columns, f"{c}_{m} 열이 없다"
    assert "region_general" in tests.columns and "tags" in tests.columns and "worse_regions_union" in tests.columns
    assert (tests.pool_label_blk5.str.startswith("지역")).all()
    rg = tests[tests.region_general]
    assert all(XG.REGION_GENERAL_TXT in t for t in rg.tags) and all(XG.REGION_GENERAL_TXT not in t for t in tests[~tests.region_general].tags)
    # 두 마스크 합집합: 부분 풀 표지, 1절 의존 표지, 제품 결측 대체, 열세 지역(앞 마스크 값, 이름 뒤 마스크 표기)
    mr5 = dict(pool="부분(지역 1/2)", limit_dependence=XB.LIMIT_DEP_TXT, ci_dependence="", fill_frac_B=0.0, delta=-1.2, n_ci_regions=1)
    mr25 = dict(pool="지역 2/2", limit_dependence="", ci_dependence=XB.CI_DEP_TXT, fill_frac_B=0.1, delta=-1.0, n_ci_regions=2)
    reg5 = [dict(scope="region", target="Lena|x", worse=True, delta=1.5, ci_lo=0.5, ci_hi=2.5)]
    reg25 = [dict(scope="region", target="Lena|x", worse=True, delta=1.1, ci_lo=0.2, ci_hi=2.0),
             dict(scope="region", target="Canada|x", worse=True, delta=0.9, ci_lo=0.1, ci_hi=1.7)]
    info = {"blk5": dict(mr=mr5, rows=reg5, note=XB.UNCORRECTED_TXT, branch="미결정"), "blk25": dict(mr=mr25, rows=reg25, note="", branch="우세")}
    notes = XG.combined_notes(info, "미결정", "마스크 의존", -1.2)
    for want in ("부분(지역 1/2)", XB.UNCORRECTED_TXT, "마스크 의존", XB.LIMIT_DEP_TXT, XB.CI_DEP_TXT, "제품 결측 대체 10.0%"):
        assert want in notes, want
    wu = XG.worse_union(info)
    assert [w[0] for w in wu] == ["Lena|x(5·25 km 마스크)", "Canada|x(25 km 마스크)"] and wu[0][1] == 1.5, "앞 마스크 값과 두 마스크 표기"
    mc = XG.mask_cols({"blk5": dict(mr=dict(mr5, region_general=True)), "blk25": dict(mr=dict(mr25, region_general=False))})
    assert mc["pool_label_blk5"] == "부분(지역 1/2)" and mc["limit_dependence_blk5"] == XB.LIMIT_DEP_TXT and mc["fill_frac_B_blk25"] == 0.1
    assert mc["region_general_blk5"] and not mc["region_general"], "지역 일반은 두 마스크 모두 참일 때만"
    s = XG.sentence("XG-1", "미결정", "wei", -1.2, 1, 2, None, notes, wu)
    assert "부분(지역 1/2)" in s and "마스크 의존" in s and "지역 Lena|x(5·25 km 마스크) 에서는 오차가 컸다" in s
    assert XG.fill_note(0.0) == "" and XG.fill_note(np.nan) == "" and XG.fill_note(0.0002) == "제품 결측 대체 0.1% 미만"


def test_xg_v_descriptive_pools_leak_cols_and_xg4r_sentence():
    """검토 반영(결함 6·7·8): CALM 학습 제품의 XG-4 풀에서 러시아 W·E 를 빼고 별표 참고값으로만 싣는다, L2 제품은 누설 점검 불가·SI 전용,
    XG-4r 은 두 마스크 결합 갈래와 사전 고정 문장을 가진다."""
    R = _synth_rescore(products=("wei", "cci5y", "aalto"), with_p1=True)
    d = XG.descriptive_tests(R)
    x4 = d[d.hypothesis == "XG-4"]
    for p in ("wei", "aalto"):                                             # CALM 학습 제품: 풀에서 러시아 W·E 제외, 별표 참고 행
        mean_t = x4[(x4["product"] == p) & (x4.scope == "MEAN")].target.unique().tolist()
        assert mean_t == ["MEAN[Lena|x,Canada|x]"], mean_t
        ru = x4[(x4["product"] == p) & (x4.target.isin(XG.XG7))]
        assert len(ru) and (ru.ref_star == XG.REF_STAR).all() and (ru.role == XG.REF_ROLE).all()
        assert (ru.verdict4.astype(str) == "").all(), "별표 참고값에는 판정 열을 비운다"
    cm = x4[(x4["product"] == "cci5y") & (x4.scope == "MEAN")].target.unique().tolist()
    assert cm == ["MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"] and (x4[x4["product"] == "cci5y"].ref_star == "").all()
    aa = d[d["product"] == "aalto"]
    assert (aa.leak_check == XG.LEAK_CHECK["L2"]).all() and aa.si_only.all() and all(XG.SI_ONLY_TXT in r or r == XG.REF_ROLE for r in aa.role)
    assert (d[d["product"] != "aalto"].leak_check == "").all() and not d[d["product"] != "aalto"].si_only.any()
    x4r = d[(d.hypothesis == "XG-4r") & (d.scope == "MEAN")]
    assert len(x4r) == 2 * 2 * 2 and set(x4r.branch) <= set(XB.BRANCHES) and (x4r.holm == "없음(서술)").all()
    assert all(isinstance(s, str) and XG.DEVIATION in s and ("원값" in s) for s in x4r.sentence)
    assert set(x4r.mask_dependence) <= {"", "마스크 의존"}
    k = XG.rmse_table(R, sorted(R.raw), R.products)
    star = k[k.ref_star != ""][["target", "product"]].drop_duplicates()
    assert set(star.target) == set(XG.XG7) and set(star["product"]) == {"wei", "aalto"}
    assert (k[k["product"] == "aalto"].leak_check == XG.LEAK_CHECK["L2"]).all() and (k[k["product"] == "cci5y"].ref_star == "").all()


def test_xg_w_fill_fraction_and_sensitivity():
    """검토 반영(결함 4): 대비 행에 ρ·s 대체 채점 셀 비율(fill_frac_B)과 '제품 결측 대체 k%' 표지가 실리고, 민감도 표는 대체 셀이 있는 블록을
    두 팔에서 같이 뺀다."""
    R = _synth_rescore()
    for nm, by in R.raw.items():
        R.fill[(nm, "wei")] = {sp: np.r_[np.array([2, 1]), np.zeros(len(st.blocks) - 2, int)] for sp, st in by.items()}
    gA, gB = XG.grp("B:wei_raw", 0), XG.grp("P0", 0)
    rows, _ = XG.contrast_rows(R, XG.POOL_W, gA, gB, "wei", "blk5", "XG-1w", registered=2)
    mr = rows[-1]
    assert mr["scope"] == "MEAN" and mr["fill_frac_B"] > 0 and mr["fill_note"].startswith(XG.FILL_TXT) and not mr["fill_excluded"]
    assert mr["n_B_filled"] == sum(r["n_B_filled"] for r in rows[:-1]) and all(r["fill_frac_B"] > 0 for r in rows[:-1])
    nb_full = {nm: dict(R.tm(nm, "wei", "blk5")[1]) for nm in XG.POOL_W}
    rows2, _ = XG.contrast_rows(R, XG.POOL_W, gA, gB, "wei", "blk5", "XG-1w", registered=2, excl_fill=True)
    mr2 = rows2[-1]
    assert mr2["fill_excluded"] and mr2["n_blocks_fill_dropped"] > 0 and mr2["fill_frac_B"] == 0 and mr2["fill_note"] == ""
    for nm in XG.POOL_W:
        nb2 = R.tm(nm, "wei", "blk5", excl_fill=True)[1]
        assert all(nb2[sp] <= nb_full[nm][sp] for sp in nb2), "대체 셀 블록을 뺀 뒤 채점 블록이 줄어든다"
        tmA, _ = R.tm(nm, "wei", "blk5", excl_fill=True)
        for sp, st in tmA.used.items():                                     # 두 팔(B:wei_raw, P0)이 같은 블록 집합에서 채점된다
            assert set(st.blocks) == set(R.raw[nm][sp].blocks) - R.drop_set(nm, "wei", "blk5", excl_fill=True)
    fs = XG.fill_sensitivity(R)
    assert len(fs) and fs.fill_excluded.all() and (fs.holm == "없음(서술)").all() and (fs.role == XG.FILL_SENS_TXT).all()
    mm = fs[fs.scope == "MEAN"]
    assert set(mm.branch_descriptive) <= set(XB.BRANCHES) and set(mm.hypothesis) == {h["id"] for h in XG.CONFIRM}
    tests, _ = XG.confirm_tests(R)
    w = tests[tests["product"] == "wei"]
    assert (w.fill_frac_B > 0).all() and all(XG.FILL_TXT in t for t in w.tags), "판정 표에 결측 대체 비율과 표지"
    assert tests[tests["product"] == "cci5y"].fill_frac_B.isna().all() or (tests[tests["product"] == "cci5y"].fill_frac_B == 0).all()


def test_xg_x_strict_cell_inputs_cfg_nboot_gdal(tmp_path, monkeypatch):
    """검토 반영(결함 2·9·12): 작업 R3 본 실행은 Wei 값·셀 거리 표가 없으면 적합 전에 멈춘다, 조각 설정에 입력 표 sha256 이 들어간다,
    본 봉인 폴더는 재표집 10,000회만, GDAL 캐시·임시 폴더는 data/raw 아래."""
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    a = XG.parse_args(["--out-dir", str(tmp_path / "xg")])
    a.BASE_OUT.mkdir(parents=True)
    with pytest.raises(SystemExit, match="wei"):
        XG.require_cell_inputs(a, {"cci5y": pd.Series(dtype=float)}, "wei")
    assert XG.require_cell_inputs(a, {"wei": pd.Series(dtype=float)}, "wei")
    with pytest.raises(SystemExit, match="묶음"):
        XG.cell_dist(a, "Lena|x", [1, 2], [0, 0], [0, 0], strict=True)      # 표 없음
    pd.DataFrame(dict(target="Lena|x", product="wei", loc_id=[1, 2], dist_km=[7.5, np.nan], within_cell_d=[0, 0])).to_csv(
        a.BASE_OUT / XG.CELLDIST_FILE, index=False)
    with pytest.raises(SystemExit, match="거리가 없다"):
        XG.cell_dist(a, "Lena|x", [1, 2], [0, 0], [0, 0], strict=True)      # 비유한 거리
    with pytest.raises(SystemExit, match="행이 없다"):
        XG.cell_dist(a, "Canada|x", [1, 2], [0, 0], [0, 0], strict=True)    # 대상 행 없음
    with pytest.raises(SystemExit):
        XG.preflight_cell(a)                                               # 제품 값 표 없음
    c1 = XG.cell_cfg(a)
    assert c1["values_sha256"] == "" and c1["celldist_sha256"] != "" and c1["lg_grid"] == list(XG.LG_GRID)
    pd.DataFrame(dict(loc_id=[1], product="wei", value_cm=[50.0])).to_csv(a.BASE_OUT / XG.VALUES_FILE, index=False)
    a.INSHA = None
    c2 = XG.cell_cfg(a)
    assert c2["values_sha256"] != "" and XB.cfg_hash(c1) != XB.cfg_hash(c2), "입력 표가 다르면 설정 해시가 달라 --resume 이 조각을 다시 쓰지 않는다"
    assert XG.preflight_cell(a)
    # 재표집 수: 허용 표지 없는 로컬(1,000회 상한)은 본 봉인 폴더에 쓰지 않는다
    with pytest.raises(SystemExit, match="10000"):
        XG.check_nboot(XG.parse_args(["--summarize-only"]))
    assert XG.parse_args(["--summarize-only"]).nboot == XB.LOCAL_NBOOT_MAX
    XG.check_nboot(XG.parse_args(["--summarize-only", "--allow-local"]))
    XG.check_nboot(XG.parse_args(["--smoke"]))
    with pytest.raises(SystemExit):
        XG.check_nboot(XG.parse_args(["--summarize-only", "--allow-local", "--nboot", "5000"]))
    # GDAL: 캐시 상한(MB)과 임시 폴더(data/raw 아래)
    monkeypatch.delenv("GDAL_CACHEMAX", raising=False); monkeypatch.delenv("CPL_TMPDIR", raising=False)
    rec = XG.gdal_env(tmp_path / "raw")
    assert os.environ["GDAL_CACHEMAX"] == str(XG.GDAL_CACHEMAX_MB) and Path(os.environ["CPL_TMPDIR"]) == tmp_path / "raw" / XG.GDAL_TMP_NAME
    assert (tmp_path / "raw" / XG.GDAL_TMP_NAME).is_dir() and rec["GDAL_CACHEMAX"] == "512" and XG.RAW == XB.ROOT / "data" / "raw"
    with pytest.raises(SystemExit):
        XG.parse_args(["--gate-level", "same_node"])                      # XG 에는 쓰지 않는 인자라 뺐다(결함 14). XH 는 관문 (1)에 쓴다
    assert XH.parse_args(["--gate-level", "same_node"]).gate_level == "same_node"


def test_xg_y_payload_check_and_main_guards(tmp_path, monkeypatch):
    """검토 반영(결함 1·3·13·14): 묶음 점검이 PAYLOAD_EXTRA 를 payload_manifest(extra=) 로 넘기고 없는 입력을 돌려준다(파일 위치 이탈 기록 포함),
    상주 메모리 감시는 허용 표지와 관계없이 Rescale 환경 변수가 없으면 건다, 집계·관문 경로는 가용 메모리를 기다린다(wait_memory)."""
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    monkeypatch.delenv("WF_RESCALE", raising=False); monkeypatch.delenv("LG_RESCALE", raising=False)
    for mod in (XG, XH):                                                   # 가용 메모리 대기: 하한 0 이면 바로 돌아온다(값은 /proc 의 MemAvailable)
        g = mod.wait_memory(min_gb=0.0, wait_s=0.0)
        assert (not np.isfinite(g)) or g >= 0.0
        assert mod.MEM_MIN_GB == 30.0 and mod.MEM_WAIT_S >= 3600.0 and mod.RSS_LIMIT_GB == 10.0
    seen = {}

    def fake_manifest(extra=(), out=None, allowed=None):
        seen["extra"] = tuple(extra)
        return dict(missing=list(extra), sha256={}, n_files=0)
    monkeypatch.setattr(XB, "payload_manifest", fake_manifest)
    for mod, name, job in ((XG, "xg_payload_manifest.json", "R3"), (XH, "xh_payload_manifest.json", "R1b")):
        a = mod.parse_args(["--out-dir", str(tmp_path / mod.EXP_ID)])
        miss = mod.payload_check(a, write=True)
        assert miss == list(mod.PAYLOAD_EXTRA) and seen["extra"] == tuple(mod.PAYLOAD_EXTRA)
        assert all(f.startswith("data/processed/xbatch/") for f in mod.PAYLOAD_EXTRA)
        assert set(mod.PAYLOAD_EXTRA) <= set(XB.PAYLOAD_OPTIONAL), "공용 골격의 묶음 목록(PAYLOAD_OPTIONAL)에도 있어야 한다"
        out = (tmp_path / mod.EXP_ID) / name
        man = json.loads(out.read_text())
        assert man["module_inputs"]["job"] == job and man["module_inputs"]["missing"] == miss and "파일 위치 이탈" in man["module_inputs"]["file_location"]
    for k, req in XB.PAYLOAD_REQUIRES.items():
        if "XG_product_comparison" in k or "XH_validation_ladder" in k:
            assert set(req) <= set(XB.PAYLOAD_OPTIONAL)
    # 주 함수의 보호: 감시 스레드는 Rescale 환경 변수가 없으면 --allow-local 이 있어도 건다, 집계는 wait_memory 를 거친다
    for mod in (XG, XH):
        calls = dict(watch=0, wait=0)
        monkeypatch.setattr(mod, "rss_watchdog", lambda limit_gb=10.0, poll_s=2.0: calls.__setitem__("watch", calls["watch"] + 1))
        monkeypatch.setattr(mod, "wait_memory", lambda *a_, **k_: calls.__setitem__("wait", calls["wait"] + 1) or 99.0)
        monkeypatch.setattr(mod, "summarize", lambda a_, **k_: {})
        rc = mod.main(["--payload-check", "--allow-local", "--out-dir", str(tmp_path / mod.EXP_ID)])
        assert rc == 1 and calls["watch"] == 1 and calls["wait"] == 0, "--allow-local 이어도 로컬이면 감시를 건다. 묶음 점검은 메모리를 기다리지 않는다"
        rc = mod.main(["--summarize-only", "--allow-local", "--out-dir", str(tmp_path / mod.EXP_ID)])
        assert rc == 0 and calls["watch"] == 2 and calls["wait"] == 1, "집계 경로는 wait_memory 를 거친다"
        with pytest.raises(SystemExit, match="10000"):
            mod.main(["--summarize-only", "--out-dir", str(tmp_path / mod.EXP_ID)])   # 허용 표지 없음 → 1,000회 → 본 봉인 폴더 거부
        assert calls["watch"] == 3 and calls["wait"] == 1, "거부는 재표집 전에 난다(감시는 걸렸고 대기 전이다)"
        monkeypatch.setenv("WF_RESCALE", "1")
        rc = mod.main(["--payload-check", "--out-dir", str(tmp_path / mod.EXP_ID)])
        assert rc == 1 and calls["watch"] == 3, "Rescale 환경에서는 감시를 걸지 않는다"
        monkeypatch.delenv("WF_RESCALE", raising=False)
    monkeypatch.setattr(XH, "wait_memory", lambda *a_, **k_: 99.0)
    monkeypatch.setattr(XB, "execute", lambda *a_, **k_: (_ for _ in ()).throw(AssertionError("색인 없이 적합이 시작됐다")))
    with pytest.raises(SystemExit, match=r"\[W1K\]"):
        XH.main(["--shard", "Canada:W1K:1", "--allow-local", "--out-dir", str(tmp_path / "XH")])
    assert XH.preflight_w1k(XH.parse_args(["--out-dir", str(tmp_path / "XH")]), [("Canada", "W1R", 1, "")]) == ""


def make_tctx(seed=0, split=1, target="T", mode="x"):
    rng = np.random.RandomState(seed)
    D = len(XB.FEATS)
    ns, nA, nB = 300, 80, 60
    Xs = rng.randn(ns, D); ss = rng.uniform(20, 40, ns); ys = 1.6 * ss + 3 * Xs[:, 1] + 2 * rng.randn(ns)
    XA = rng.randn(nA, D); sA = rng.uniform(20, 40, nA); yA = 1.3 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    XBm = rng.randn(nB, D); sB = rng.uniform(20, 40, nB); yB = 1.3 * sB + 3 * XBm[:, 1] + 2 * rng.randn(nB)
    return H.Ctx(target, mode, split, target, Xs, ys, ss, np.repeat(["r1", "r2"], ns // 2), XA, yA, sA, np.repeat(np.arange(8), nA // 8),
                 XBm, yB, sB, 500 + np.repeat(np.arange(6), nB // 6), meta=dict(dup_of=-1, valid=True))


def test_xg_j_cell_unit_leakage():
    a = XB.h54_args(exp="wf4", threads=1, cb_iters=10, seeds=1, nboot=100)
    keep = np.ones(60, bool); keep[:9] = False                          # 학습 지점 5 km 안 채점 셀(합성)
    rng = np.random.RandomState(4)
    bsrc, bA, bB = rng.uniform(30, 80, 300), rng.uniform(30, 80, 80), rng.uniform(30, 80, 60)
    out = {}

    def run_fn(c):
        def prod_fn():
            return XG.product_keys_arrays(c.y_src, c.s_src, c.yA, c.sA, c.sB, bsrc, bA, bB, "wei", c.target, c.mode, c.split, [0, 10], 2)[0]
        u = XG.XGCellUnit(a, c, "T", keep, prod_fn, [0, 10], 2).run()
        out["u"] = u
    c = make_tctx()
    res = XB.leakage_invariance(run_fn, c)
    assert res["ok"] and res["n_keys"] > 10 and 0 < res["n_keep"] < res["n_A"]
    rows, stores, stats = out["u"].finish()
    names = [s_.target for s_ in stores]
    assert names == ["T|x", "T~c5|x"] and stores[1].ncell.sum() == keep.sum() and stores[0].ncell.sum() == 60
    ks = set(stores[0].keys)
    assert ("B:wei_raw", "none", "1", "cell", 0, 0, -1, 0.0) in ks and ("P1@wei", "none", "1", "cell", 10, 1, -1, 0.0) in ks
    assert ("R1", "catboost_lo", "1", "cell", 10, 0, 0, 0.25) in ks and ("D0", "catboost_lo", "1", "cell", 0, 0, 0, 1.0) in ks
    assert set(stores[1].keys) == ks and stats["status"] == "ok"
    assert sum(1 for r in rows if r["method"] == "P1@wei" and int(r["n"]) == 0) == 1, "n 0 의 P1@wei 는 한 번만 저장한다"


# ================================================================ XH
def test_xh_k_folds_cover_disjoint():
    V = fake_V()
    a = SimpleNamespace()
    t_idx, sc = XH.region_index(V, "R")
    assert len(t_idx) == V.N and len(sc) == int(V.score.sum())
    for st in ("W1R", "W1S", "W1B"):
        for rep in (1, 2):
            f = XH.stage_folds(a, V, "R", st, rep)
            assert len(f) == len(t_idx) and set(np.unique(f)) == set(range(5)), "모든 셀이 묶음 0–4 가운데 하나에 있다"
            seen = np.zeros(V.N, int)
            for k in range(5):
                tr, te = XH.fold_cells(V, "R", f, k, st)
                assert len(np.intersect1d(tr, te)) == 0
                seen[te] += 1
            assert np.array_equal(seen[sc], np.ones(len(sc), int)) and seen[np.setdiff1d(t_idx, sc)].sum() == 0, "채점 셀은 반복마다 한 번"
    fb = XH.stage_folds(a, V, "R", "W1B", 2)
    assert np.array_equal(fb, H41.folds_grouped(V.block, 5, XB.seed_of("lgv", "W", "R", 2))), "W1B = h41 W1 묶음"
    assert not np.array_equal(XH.stage_folds(a, V, "R", "W1R", 1), XH.stage_folds(a, V, "R", "W1R", 2))
    V.trainable[5] = False
    f = XH.stage_folds(a, V, "R", "W1R", 1)
    tr, te = XH.fold_cells(V, "R", f, int(f[5]) - 1 if f[5] > 0 else 1, "W1R")
    assert 5 not in tr


def test_xh_l_knndm_index_and_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    V = fake_V(n=300)
    monkeypatch.setattr(XH, "get_vdata", lambda a: V)
    monkeypatch.setattr(XH, "VARIANTS", {"R": ("", "pm2")})
    a = XH.parse_args(["--out-dir", str(tmp_path / "xh"), "--regions", "Alaska", "--reps", "1,2"])
    a.REGIONS = ["R"]
    rng = np.random.RandomState(2)

    def dom(region, variant, lat, lon):
        pad = 1.0 if variant == "" else 2.0
        pts = np.c_[rng.uniform(lon.min() - pad, lon.max() + pad, 400), rng.uniform(lat.min() - pad, lat.max() + pad, 400)]
        return pts, dict(definition="합성")
    monkeypatch.setattr(CV, "region_crs", lambda r, lat=None, lon=None: "+proj=laea +lat_0=66 +lon_0=-149 +datum=WGS84 +units=m +no_defs")
    idx, dm, meta = XH.build_knndm(a, regions=["R"], reps=[1, 2], n_q=12, log=None, domain_fn=dom)
    assert len(idx) == 2 * 2 * V.N and set(idx.variant) == {"", "pm2"} and len(meta["entries"]) == 4
    meta = XH.write_knndm(a, idx, dm, meta)
    tab, sha = XH.load_index(a)
    assert sha == meta["index_sha256"] == XB.sha256_file(a.INDEX)
    for v in ("", "pm2"):
        f = XH.stage_folds(a, V, "R", "W1K", 2, v, tab)
        assert set(np.unique(f)) == set(range(5))
        for k in range(5):
            tr, te = XH.fold_cells(V, "R", f, k, "W1K")
            assert len(np.intersect1d(tr, te)) == 0, "W1K 학습·채점 교집합 0"
    txt = a.INDEX.read_text().splitlines()
    a.INDEX.write_text("\n".join(txt[:-1] + [txt[-1][:-1] + ("0" if txt[-1][-1] != "0" else "1")]) + "\n")
    with pytest.raises(SystemExit):
        XH.load_index(a)
    with pytest.raises(SystemExit):
        XH.stage_folds(a, V, "R", "W1K", 1, "", None)


def _load_m1(tmp_path, monkeypatch):
    name = "m1_cv_scheme_comparison"
    if name in sys.modules:
        return sys.modules[name]
    monkeypatch.setattr(sys, "argv", ["m1", "--out-dir", str(tmp_path / "m1o"), "--fig-dir", str(tmp_path / "m1f")])
    return _load(name, "scripts/3_deep_learning/m1_cv_scheme_comparison.py")


def test_xh_m_knndm_equals_m1(tmp_path, monkeypatch):
    m1 = _load_m1(tmp_path, monkeypatch)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rng = np.random.RandomState(0)
    cen = np.array([[65.0, -150.0], [66.5, -147.0], [64.2, -145.5]])
    lab = rng.randint(0, 3, 240)
    lat = cen[lab, 0] + 0.15 * rng.randn(240); lon = cen[lab, 1] + 0.3 * rng.randn(240)
    XY = CV.project_km(lon, lat, "EPSG:3338"); rad = CV.to_rad(lon, lat)
    plat, plon = rng.uniform(63.5, 67.5, 500), rng.uniform(-151, -144, 500)
    gij = CV.nnd_km(CV.to_rad(plon, plat), rad)
    assert np.allclose(gij, m1.nnd_km(m1.to_rad(plon, plat), rad))
    for seed in (0, 1):
        f1, i1, g1 = CV.knndm(XY, rad, gij, 5, seed, 15, 0.5)
        f2, i2, g2 = m1.knndm(XY, rad, gij, 5, seed, 15, 0.5, lambda s: None)
        assert np.array_equal(f1, f2) and i1["q_selected"] == i2["q_selected"] and i1["W_selected_km"] == i2["W_selected_km"]
        assert np.allclose(g1, g2) and i1["fold_sizes"] == i2["fold_sizes"]
    r1, r2 = np.random.RandomState(4), np.random.RandomState(4)
    grp = rng.randint(0, 40, 240)
    assert np.array_equal(CV.group_folds_random(grp, 6, r1, 240), m1.group_folds_random(grp, 6, r2, 240))
    labs = m1.kmeans_labels(XY, 20, 3)
    assert np.array_equal(labs, CV.kmeans_labels(XY, 20, 3))
    pc1 = np.linalg.svd(XY - XY.mean(0), full_matrices=False)[2][0]
    a1, a2 = CV.merge_clusters(labs, XY, pc1, 5, 0.5), m1.merge_clusters(labs, XY, pc1, 5, 0.5)
    assert (a1 is None and a2 is None) or np.array_equal(a1, a2)
    f = CV.m1_folds(lat, lon, grp, 2)
    rng2 = np.random.RandomState(2)
    from sklearn.model_selection import KFold
    rc = np.empty(240, int)
    for fi, (_, te) in enumerate(KFold(6, shuffle=True, random_state=2).split(np.arange(240))):
        rc[te] = fi
    site = m1.group_folds_random(m1.grid_key(lat, lon, 0.05), 6, rng2, 240)
    blk = m1.group_folds_random(grp, 6, rng2, 240)
    assert np.array_equal(f["random_cell"], rc) and np.array_equal(f["site_0.05"], site) and np.array_equal(f["block_0.5"], blk)


def _synth_nc(path, lat0=60.0, lat1=64.0, lon0=-130.0, lon1=-124.0, res=0.1):
    import xarray as xr
    lat = np.round(np.arange(lat1, lat0 - res / 2, -res), 4); lon = np.round(np.arange(lon0, lon1 + res / 2, res), 4)
    tt = pd.date_range("2015-01-01", periods=24, freq="MS")
    base = 273.15 + 10.0 - 4.0 * (lat[None, :, None] - lat0) + 0.0 * lon[None, None, :]
    season = 15.0 * np.sin(2 * np.pi * (np.arange(24) % 12) / 12.0)[:, None, None]
    t2m = (base + season).astype("float32")
    t2m[:, :, :3] = np.nan                                               # 바다(서쪽 세 열)
    xr.Dataset(dict(t2m=(("valid_time", "latitude", "longitude"), t2m)), coords=dict(valid_time=tt, latitude=lat, longitude=lon)).to_netcdf(path)
    return lat, lon


def test_xh_n_prediction_domain_determinism(tmp_path):
    nc = tmp_path / "e.nc"
    lat, lon = _synth_nc(nc)
    box = (-129.5, -124.5, 60.5, 63.5)
    p1, m1 = CV.prediction_domain_box(box, nc=nc)
    p2, _ = CV.prediction_domain_box(box, nc=nc)
    assert np.array_equal(p1, p2) and len(p1) > 0 and m1["n_grid_permafrost"] == len(p1)
    assert np.all(p1[:, 0] >= -129.5 - 1e-9) and np.all(p1[:, 0] <= -124.5 + 1e-9) and np.all(p1[:, 0] > lon[2])
    assert np.all(p1[:, 1] > 62.5 - 1e-6), "MAAT < 0 인 북쪽 격자만(합성: 위도 62.5° 북쪽)"
    p3, _ = CV.prediction_domain_box(box, nc=nc, n_sample=20, seed=0)
    p4, _ = CV.prediction_domain_box(box, nc=nc, n_sample=20, seed=0)
    assert len(p3) == 20 and np.array_equal(p3, p4)
    g = tmp_path / "g.csv.gz"
    pd.DataFrame(dict(lat=np.linspace(70, 72, 50), lon=np.linspace(120, 130, 50), x=1)).to_csv(g, index=False, compression="gzip")
    q1, _ = CV.prediction_domain_grid(g); q2, _ = CV.prediction_domain_grid(g, n_sample=10, seed=0); q3, _ = CV.prediction_domain_grid(g, n_sample=10, seed=0)
    assert len(q1) == 50 and np.array_equal(q2, q3) and len(q2) == 10
    d1, _ = CV.prediction_domain(300, 0.05, 0, None, nc=nc, box=box)
    d2, _ = CV.prediction_domain(300, 0.05, 0, None, nc=nc, box=box)
    assert np.array_equal(d1["permafrost"], d2["permafrost"]) and len(d1["permafrost"]) == 300
    assert CV.label_box([60.0, 61.0], [-130.0, -128.0], 1.0) == (-131.0, -127.0, 59.0, 62.0)


def _cells(seed, n):
    rng = np.random.RandomState(seed)
    Xf = rng.randn(n, len(XB.FEATS)).astype(np.float32); s = rng.uniform(20, 40, n); y = 1.4 * s + 3 * Xf[:, 1] + rng.randn(n)
    nan = np.full(n, np.nan)
    return H41.Cells(Xf, y, s, nan, nan, nan, nan)


def test_xh_o_models_equal_h41():
    TR, TE = _cells(0, 150), _cells(1, 40)
    a = SimpleNamespace(LEARNERS=["catboost_lo", "catboost", "rf"], SEEDS=[0, 1], LAMS=[0.25, 0.5, 1.0])
    ref, _ = H41.models_pooled(StubFitter(), a, TR, TE, "w1", sfx="w", extended=False)
    new, rec = XH.models_w1x(StubFitter(), TR, TE, "w1", a.LEARNERS, a.SEEDS, (0.25, 1.0))
    assert set(new) <= set(ref) and len(new) == 1 + 3 * 2 * (1 + 2)
    for k in new:
        assert np.array_equal(new[k], ref[k]), k
    assert rec["E"] == H.ls_E(TR.y, TR.s)


def test_xh_p_leakage_test_labels():
    a = XH.parse_args(["--threads", "2", "--learners", "catboost_lo"])
    F = H41.make_fitter(XH.h41_args(a))
    TR, TE = _cells(2, 120), _cells(3, 30)
    o1, _ = XH.models_w1x(F, TR, TE, "w1", ["catboost_lo"], [0], (0.25, 1.0), cont=True)
    TE2 = H41.Cells(TE.X, np.random.RandomState(7).uniform(1, 500, len(TE)), TE.s, TE.ku, TE.ed, TE.cci, TE.soil)
    o2, _ = XH.models_w1x(F, TR, TE2, "w1", ["catboost_lo"], [0], (0.25, 1.0), cont=True)
    assert set(o1) == set(o2) and all(np.array_equal(o1[k], o2[k]) for k in o1), "채점 라벨이 예측에 새었다"
    assert ("RSw", "ridge", -1, 0.75) in o1 and ("D0w", "ridge", -1, 1.0) in o1
    TR2 = H41.Cells(TR.X, TR.y + np.r_[50.0, np.zeros(len(TR) - 1)], TR.s, TR.ku, TR.ed, TR.cci, TR.soil)
    o3, _ = XH.models_w1x(F, TR2, TE, "w1", ["catboost_lo"], [0], (0.25, 1.0), cont=True)
    assert any(not np.array_equal(o1[k], o3[k]) for k in o1), "학습 라벨을 바꿔도 같다(시험이 둔하다)"


def _synth_xh_shards(tmp_path, with_ridge=False):
    """합성 XH 조각(지역 3, 단 W1R·W1K, 반복 2, seed 2). with_ridge 이면 알래스카에 대회 구성 키(D0w[ridge], RSw[ridge, λ 0.75])를 더한다."""
    a = XH.parse_args(["--out-dir", str(tmp_path / "xh"), "--nboot", "200", "--learners", "catboost_lo"])
    rng = np.random.RandomState(5)
    cfg = XH.unit_cfg(a, "")
    lev = dict(W1R=dict(P=20.0, D=15.0), W1K=dict(P=21.0, D=19.0))
    for R in XH.REGIONS:
        nb = 10
        blocks = np.repeat([f"{R[:2]}{j:02d}" for j in range(nb)], rng.randint(3, 9, nb))
        for stg in ("W1R", "W1K"):
            for rep in (1, 2):
                st = H4.BlockStore(XH.store_name(R, stg), rep, blocks)
                cnt = st.ncell
                st.add_sse(("PSw", "none", "1", stg, -1, rep, -1, 0.0), cnt * (lev[stg]["P"] + rng.rand(nb)) ** 2, cnt)
                for sd in (0, 1):
                    st.add_sse(("D0w", "catboost_lo", "1", stg, -1, rep, sd, 1.0), cnt * (lev[stg]["D"] + rng.rand(nb)) ** 2, cnt)
                    st.add_sse(("RSw", "catboost_lo", "1", stg, -1, rep, sd, 0.25), cnt * (lev[stg]["P"] - 1 + rng.rand(nb)) ** 2, cnt)
                if with_ridge and R == XH.CONT_REGION:
                    st.add_sse(("D0w", "ridge", "1", stg, -1, rep, -1, 1.0), cnt * (lev[stg]["D"] + 2 + rng.rand(nb)) ** 2, cnt)
                    st.add_sse(("RSw", "ridge", "1", stg, -1, rep, -1, XH.CONT_LAM), cnt * (lev[stg]["P"] - 0.5 + rng.rand(nb)) ** 2, cnt)
                XB.write_shard(a.SHARDS, a.TAG, R, stg, rep, [dict(fold=0)], [st], cfg, dict(nnd_median_km=1.0 if stg == "W1R" else 30.0), "",
                               expected=[1, 2], code_file=XH.__file__)
    return a


def test_xh_q_summarize_dd(tmp_path, monkeypatch):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    a = _synth_xh_shards(tmp_path)
    out = XH.summarize(a, write=False)
    dd = out["xh_dd"]
    stores = H4.load_stores([s_["npz"] for s_ in XB.find_shards(a.SHARDS, a.TAG)])

    def rm(nm, g):
        t, _ = XH.avg_terms(stores, nm)
        S, C, _ = t[g]
        return float(np.sqrt(S.sum() / C.sum()))
    for R in XH.REGIONS:
        want = (rm(XH.store_name(R, "W1K"), ("D0w", "catboost_lo", 1.0)) - rm(XH.store_name(R, "W1R"), ("D0w", "catboost_lo", 1.0))) - \
               (rm(XH.store_name(R, "W1K"), ("PSw", "none", 0.0)) - rm(XH.store_name(R, "W1R"), ("PSw", "none", 0.0)))
        row = dd[(dd["item"] == "XH-1") & (dd.target == R) & (dd.variant == "")]
        assert len(row) == 1 and abs(float(row.dd.iloc[0]) - want) < 1e-9
        assert row.ci_lo.iloc[0] <= row.ci_hi.iloc[0]
    mr = dd[(dd["item"] == "XH-1") & (dd.scope == "MEAN") & (dd.variant == "")]
    assert len(mr) == 1 and abs(float(mr.dd.iloc[0]) - dd[(dd["item"] == "XH-1") & (dd.scope == "region") & (dd.variant == "")].dd.mean()) < 1e-12
    lad = out["xh_ladder"]
    assert set(lad.stage) == {"W1R", "W1K"} and (lad[lad.stage == "W1K"].nnd_median_km == 30.0).all()
    assert not any(XB.FORBIDDEN_OUT.search(str(v)) for v in dd["item"].unique())


def test_xh_u_gate2_continuity_rows_and_gate1_rule(tmp_path, monkeypatch):
    """검토 반영(결함 10·11): 집계가 관문 기록을 읽어 관문 (2) 실패면 알래스카 연속성 행(ridge 키)을 빼고, 통과·미확인은 표지를 달며, 관문 요약을
    봉인 메타에 쓴다. 관문 (1)의 통과는 1절 허용 오차만으로 정하고 1 ulp 경로는 정보 열이다."""
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    a = _synth_xh_shards(tmp_path, with_ridge=True)
    out = XH.summarize(a, write=False)                                     # 관문 기록 없음 → 능형 행 유지 + '관문 2 미확인'
    lad = out["xh_ladder"]
    rg = lad[lad.learner == "ridge"]
    assert len(rg) == 2 * 2 and (rg.region == "Alaska").all() and set(rg.lam) == {1.0, XH.CONT_LAM}
    assert all(XH.CONT_UNCHECKED_TXT in s for s in rg.continuity_gate) and (lad[lad.learner != "ridge"].continuity_gate == "").all()
    gs = a.SUMMARY_META["gates"]
    assert not gs["found"] and gs["cont_action"] == "flag" and gs["ridge_rows_dropped"] == 0
    gm = a.OUT / XH.GATE_META
    import time as _t
    gm.write_text(json.dumps(dict(gate1=dict(n_keys=7, n_fail=0, passed=True), gate2=dict(n_keys=36, n_fail=3, passed=False),
                                  gate2_note=XH.CONT_FAIL_TXT, level="local_rescale", created="x")))
    out2 = XH.summarize(a, write=False)                                    # 관문 (2) 실패(기록이 조각보다 새롭다) → 능형 행 제거
    assert not (out2["xh_ladder"].learner == "ridge").any() and len(out2["xh_ladder"]) == len(lad) - 4
    gs2 = a.SUMMARY_META["gates"]
    assert gs2["found"] and gs2["gate1"] == "통과" and gs2["gate2"] == "실패" and gs2["cont_action"] == "drop" and gs2["ridge_rows_dropped"] == 4
    assert XH.CONT_FAIL_TXT in gs2["ridge_note"] and gs2["summary"]["gate2"]["n_fail"] == 3
    assert len(out2["xh_dd"]) == len(out["xh_dd"]) and len(out2["xh_stage_contrasts"]) == len(out["xh_stage_contrasts"]), "그 밖의 표는 그대로"
    gm.write_text(json.dumps(dict(gate1=dict(n_keys=7, n_fail=0, passed=True), gate2=dict(n_keys=36, n_fail=0, passed=True), gate2_note="",
                                  level="local_rescale", created="x")))
    out3 = XH.summarize(a, write=False)                                    # 관문 (2) 통과 → 능형 행 유지 + '관문 2 통과'
    r3 = out3["xh_ladder"]
    assert (r3[r3.learner == "ridge"].continuity_gate == "관문 2 통과").all() and a.SUMMARY_META["gates"]["cont_action"] == "flag"
    gm.write_text(json.dumps(dict(gate1=dict(n_keys=7, n_fail=0, passed=True), gate2=dict(n_keys=36, n_fail=3, passed=False),
                                  gate2_note=XH.CONT_FAIL_TXT, level="local_rescale", created="x")))
    old = _t.time() - 86400.0
    os.utime(gm, (old, old))                                               # 관문 기록이 조각보다 오래됨 → 빼지 않고 표지
    out4 = XH.summarize(a, write=False)
    r4 = out4["xh_ladder"]
    assert (r4.learner == "ridge").sum() == 4 and all("오래됨" in s for s in r4[r4.learner == "ridge"].continuity_gate)
    assert a.SUMMARY_META["gates"]["stale"] and a.SUMMARY_META["gates"]["cont_action"] == "flag"
    gm.write_text(json.dumps(dict(gate1=dict(n_keys=0, n_fail=0, passed=False), gate2=dict(n_keys=0, n_fail=0, passed=False),
                                  gate2_note="M1C 조각 없음(관문 2 미실행)", level="local_rescale", created="x")))
    XH.summarize(a, write=False)
    assert a.SUMMARY_META["gates"]["gate2"] == "미실행" and a.SUMMARY_META["gates"]["cont_action"] == "flag"
    # 관문 (1) 규칙: 1절 허용 오차(로컬과 Rescale 사이 |ΔSSE| ≤ 1e-4 cm² 또는 상대 1e-9)만. ulp 는 통과 근거가 아니다
    assert XH.gate1_decision(0.0, 0.0, 0.0, True) == (True, False)
    assert XH.gate1_decision(5e-5, 1e-6, 3.0, True) == (True, False), "절대 허용 안이면 ulp 와 무관하게 통과"
    assert XH.gate1_decision(1.0, 1e-10, 0.5, True) == (True, False), "상대 허용 안"
    assert XH.gate1_decision(1.0, 1e-3, 0.5, True) == (False, True), "허용 오차를 넘으면 예측 차가 1 ulp 이내라도 실패(정보 열만 참)"
    assert XH.gate1_decision(1.0, 1e-3, 0.5, False) == (False, False) and XH.gate1_decision(np.nan, np.nan, np.inf, True) == (False, False)
    assert XH.gate1_decision(1e-12, 1e-15, 2.0, True, level="same_node") == (False, False), "같은 노드 종류는 |ΔSSE| 0"
    assert XH.gate1_decision(0.0, 0.0, 0.0, True, level="elm_hematite") == (True, False)


def test_xh_r_cli_flags_and_guard(monkeypatch):
    monkeypatch.delenv("WF_RESCALE", raising=False); monkeypatch.delenv("LG_RESCALE", raising=False)
    for mod in (XG, XH):
        a = mod.parse_args(["--count-only"])
        assert a.count_only and not a.PERMIT
        assert mod.parse_args(["--smoke"]).smoke and mod.parse_args(["--allow-local"]).PERMIT
    assert XH.parse_shard("Canada:W1K:2:pm2") == ("Canada", "W1K", 2, "pm2") and XH.parse_args(["--shard", "Alaska:M1C:0"]).shard
    with pytest.raises(SystemExit):
        XH.main(["--shard", "Canada:W1R:1"])                             # 허용 표지 없는 본 실행
    with pytest.raises(SystemExit):
        XG.main(["--part", "cell", "--shard", "Lena:x:1"])
    a = XH.parse_args(["--smoke"])
    assert a.REGIONS == ["Canada"] and a.STAGES == ["W1R", "W1B"] and a.TAG == "xh_smoke" and a.OUT.name == "smoke"
    assert XB.shard_paths(a.SHARDS, a.TAG, "Canada", "W1K", 1, "pm2")["unit"].name == "xh_smoke__cpu__Canada__W1K__s1__pm2_unit.json"
    g = XG.parse_args(["--smoke"])
    assert g.TARGETS == ["Lena|x"] and g.SPLITS == [1] and g.nboot <= 200 and g.TAG == "xgc_smoke"
    th = XH.rss_watchdog(1000.0)
    assert (th is None or th.daemon) and XH.rss_watchdog(1000.0) is None, "감시 스레드는 한 번만 띄운다"


def test_xg_t_cell_distance_table(tmp_path, monkeypatch):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    a = XG.parse_args(["--out-dir", str(tmp_path / "xg")])
    a.BASE_OUT.mkdir(parents=True)
    pd.DataFrame(dict(target="Lena|x", product="wei", loc_id=[3, 1, 2], dist_km=[7.5, 0.4, 30.0], within_cell_d=[0, 1, 0])).to_csv(
        a.BASE_OUT / XG.CELLDIST_FILE, index=False)
    d = XG.cell_dist(a, "Lena|x", [1, 2, 3], [0, 0, 0], [0, 0, 0])
    assert np.allclose(d, [0.4, 30.0, 7.5]) and list(~(d <= XG.CELL_D)) == [False, True, True]


# ================================================================ 실제 자료(세기 범주)
def _run_capture(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(argv)
    return rc, buf.getvalue()


@REAL
def test_xh_s_count_only_no_label_statistics():
    argv = ["--count-only", "--regions", "Canada", "--stages", "W1R,W1B", "--reps", "1"]
    rc, out1 = _run_capture(XH.main, argv)
    assert rc == 0 and "[세기]" in out1
    assert all(XB.allowed_line(ln) for ln in out1.splitlines()), "출력 제한 패턴이 있다"
    a = XH.parse_args(argv)
    V = XH.get_vdata(a)
    y0 = V.y.copy()
    try:
        fin = np.isfinite(V.y)
        V.y[fin] = np.random.RandomState(0).permutation(V.y[fin]) * 1.7 + 11.0
        rc2, out2 = _run_capture(XH.main, argv)
    finally:
        V.y[:] = y0
    assert rc2 == 0 and out1 == out2, "세기 출력이 라벨 값에 따라 달라진다"


@REAL
def test_xg_s_count_only_no_label_statistics():
    argv = ["--count-only", "--targets", "Canada|x,Canada|r", "--cell-targets", "Canada"]
    rc, out1 = _run_capture(XG.main, argv)
    assert rc == 0 and "[세기]" in out1 and "셀 단위 Canada s1" in out1
    assert all(XB.allowed_line(ln) for ln in out1.splitlines()), "출력 제한 패턴이 있다"
    saved = []
    try:
        for HA_, D_ in list(W._DATA.values()):
            saved.append((D_, D_.df["y"].values.copy()))
            fin = np.isfinite(D_.df["y"].values)
            v = D_.df["y"].values.copy(); v[fin] = np.random.RandomState(1).permutation(v[fin]) * 0.6 + 5.0
            D_.df["y"] = v
        rc2, out2 = _run_capture(XG.main, argv)
    finally:
        for D_, y_ in saved:
            D_.df["y"] = y_
    assert rc2 == 0 and out1 == out2, "세기 출력이 라벨 값에 따라 달라진다"
