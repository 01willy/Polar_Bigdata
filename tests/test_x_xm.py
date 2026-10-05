"""XM_landcover_vegetation 시험(추가 등록 docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md, 커밋 ff7c97f).

합성 자료 시험(라벨 값을 화면에 쓰지 않는다)
(a) 1 km 라벨 셀: 특징 추출의 셀 색인이 하네스 위치 묶음(x_hires_covariates.loc_cell)과 같고, 셀 범위가 점을 담으며 남북·동서 약 1.00 km 다.
(b) WorldCover 셀 비율: 합성 경위도 래스터에서 화소 중심 포함, 등급 수, 분모(자료 화소), 자료 10 % 미만 결측이 수동 계산과 같다.
(c) Sentinel-2: 2 × 2 평균(결측 전파), 반사도와 offset 진단(1 백분위수 ≥ 1,000), 유효 관측(SCL 4·5, 반사도 > 0), NDVI·NDMI, 시간 중앙값, 셀 값의
    하한(10 %, 20 화소).
(d) 장면 선택: COG 자산 거르기, 같은 타일·같은 날 중복 제거(s2:sequence 최대), r ≥ 0.9 먼저·운량·자료 없음 비율·시각 순 6장면, 타일 배정(꼭짓점 여유 최대).
(e) 특징 추출이 라벨 열을 읽지 않는다(읽기 기록과 거부).
(f) 변형 열: xw = x25 + WC 11 + S2 3, xw_lc = x25 + WC 11, S2 가 빠지면 xw 는 xw_lc 와 같고, 모두 빠지면 돌지 않는다.
(g) 하네스: --stage xm 인자(변형, tag, 산출 경로, 관문 수준), S2 전부 제외 때 xw 건너뜀, 적합 행렬의 열 수, 누설 시험(1절, h54 시험 b·d·m 형식).
(h) 판정과 문장: Holm m = 3, _rule3, 다섯 갈래 문장, 0.5 cm 미만 표기, 대체(xw_lc) 문구.
(i) 합성 조각 집계: XE x25 조각을 다시 쓰고 XM 조각과 합쳐 봉인 폴더에만 쓴다. 관문이 봉인 표보다 먼저 돌고, 화면에는 RMSE·Δ·판정이 없다.
    공통 설정 해시가 다르면 멈춘다.
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python3 -m pytest -q tests/test_x_xm.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util
import json
import sys
from pathlib import Path

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
NEED_F = pytest.mark.skipif(not (PREP / "xm_landcover_s2_features.py").exists(), reason="특징 추출 모듈이 없다(xbatch 묶음 밖)")
NEED_RIO = pytest.mark.skipif(importlib.util.find_spec("rasterio") is None, reason="rasterio 가 없다")


def _F():
    return _load("xm_landcover_s2_features", PREP / "xm_landcover_s2_features.py")


# ---------------------------------------------------------------- 합성 문맥
def make_ctx(seed=0, target="T", split=1, n_extra=14, nA=96, nbB=6, variant="xw"):
    """합성 지역 내 문맥(h54.RCtx). 채점 셀은 블록마다 격자 묶음 3개(같은 √TDD), 묶음마다 위치 칸 1–3개, 위치마다 1–3셀."""
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
        eA, eB = rng.rand(nA, n_extra), rng.rand(nB, n_extra)
        eA[rng.rand(nA, n_extra) < 0.05] = np.nan
        M.attach_variant(c, variant, np.hstack([XA, eA]), np.hstack([XBm, eB]))
    return c


def small_args(grid=(20, -1), seeds=1, splits=(1, 2), nboot=200):
    return XB.h54_args(exp="wf9", splits=list(splits), grid=list(grid), threads=1, cb_iters=10, seeds=seeds, draws_cap=1, nboot=nboot)


def _xm_feat(tmp_path, s2_included=True, name="xm_feat_v1.csv"):
    """합성 XM 특징 표와 메타(group_decisions). s2_included = bool 또는 {대상: bool}."""
    d = tmp_path / "xm_inputs"
    d.mkdir(parents=True, exist_ok=True)
    fp = d / name
    pd.DataFrame({"loc_id": [1, 2], **{c: [0.1, 0.2] for c in RG.XM_COLS}}).to_csv(fp, index=False)
    inc = s2_included if isinstance(s2_included, dict) else {t: bool(s2_included) for t in RG.TARGETS}
    dec = {t: {"WC": dict(included=True, finite_frac=1.0), "S2": dict(included=inc[t], finite_frac=1.0 if inc[t] else 0.5)} for t in RG.TARGETS}
    (d / (fp.stem + "_meta.json")).write_text(json.dumps(dict(sha256=RG.sha256_file(fp), group_decisions=dec, group_cols_sha256={})))
    return fp


# ---------------------------------------------------------------- (a) 셀 기하
@NEED_F
def test_a_cell_geometry_matches_harness():
    F = _F()
    rng = np.random.RandomState(0)
    lat = np.r_[60 + 12 * rng.rand(300), 66.0045, 71.993]
    lon = np.r_[-165 + 300 * rng.rand(300), -150.0, 126.4]
    ky, kx = F.cell_index(lat, lon)
    ky2, kx2 = M.loc_cell(lat, lon)
    assert np.array_equal(ky, ky2) and np.array_equal(kx, kx2), "셀 색인이 하네스 위치 묶음과 다르다"
    w, s, e, n = F.cell_bounds(ky, kx)
    assert F.in_cell(lon, lat, (w, s, e, n)).all() and (w < e).all() and (s < n).all()
    phi = np.radians((ky + 0.5) * F.CELL_DEG)
    ew_km = (e - w) * 111.32 * np.cos(phi)
    ns_km = (n - s) * 111.32
    assert np.allclose(ns_km, 1.00188, atol=1e-4) and np.allclose(ew_km, 1.00188, atol=1e-4)
    assert not F.in_cell(np.array([e[0]]), np.array([lat[0]]), (w[0], s[0], e[0], n[0]))[0], "범위는 반열림(동쪽 끝 제외)"


# ---------------------------------------------------------------- (b) WorldCover 셀 비율
@NEED_F
@NEED_RIO
def test_b_worldcover_fractions_manual():
    from affine import Affine
    F = _F()
    a = 1 / 1200.0                                                       # 합성 화소(약 93 m)
    tr = Affine(a, 0.0, -150.0, 0.0, -a, 66.0)
    H_, W_ = 60, 120
    rng = np.random.RandomState(1)
    codes = np.array(list(F.WC_CODES.values()))
    arr = codes[rng.randint(0, len(codes), size=(H_, W_))].astype(np.uint8)
    arr[:5, :] = 0                                                       # 자료 없음
    ky, kx = F.cell_index(np.array([65.98]), np.array([-149.97]))
    b = tuple(float(v[0]) for v in F.cell_bounds(ky, kx))
    win = F.geo_window(tr, b, W_, H_)
    r0, c0, h, w = win
    lon, lat = F.geo_centers(tr, r0, c0, h, w)
    m = F.in_cell(lon, lat, b)
    cnt, n_in, n_data = F.wc_counts(arr[r0:r0 + h, c0:c0 + w], m)
    # 수동: 전체 격자의 화소 중심으로 직접 센다
    LON, LAT = np.meshgrid(-150.0 + (np.arange(W_) + 0.5) * a, 66.0 - (np.arange(H_) + 0.5) * a)
    mm = F.in_cell(LON, LAT, b)
    assert n_in == int(mm.sum()) and n_data == int((mm & (arr != 0)).sum())
    for k, code in F.WC_CODES.items():
        assert cnt[k] == int((mm & (arr == code)).sum())
    fr, dfrac = F.wc_fractions(cnt, n_in, n_data)
    assert abs(sum(fr.values()) - 1.0) < 1e-12 and dfrac == pytest.approx(n_data / n_in)
    fr2, _ = F.wc_fractions(cnt, 100, 5)                                 # 자료 5 % < 10 % → 결측
    assert all(np.isnan(v) for v in fr2.values())
    assert F.wc_tiles_for((-150.2, 65.99, -149.9, 66.01)) == ["N63W153", "N63W150", "N66W153", "N66W150"]


# ---------------------------------------------------------------- (c) Sentinel-2 계산
@NEED_F
def test_c_s2_reflectance_indices_and_cell_values():
    F = _F()
    a10 = np.arange(16, dtype=np.float32).reshape(4, 4)
    a10[0, 0] = np.nan
    g = F.agg2x2(a10)
    assert np.isnan(g[0, 0]) and g[0, 1] == pytest.approx(np.mean([2, 3, 6, 7])) and g.shape == (2, 2)
    dn = np.array([0, 500, 1500], np.uint16)
    r0 = F.reflectance(dn, offset=False); r1 = F.reflectance(dn, offset=True)
    assert np.isnan(r0[0]) and r0[1] == pytest.approx(0.05) and r1[2] == pytest.approx(0.05) and np.isnan(r1[0])
    scl = np.full((10, 10), 4, np.uint8)
    off, p, npx = F.dn_offset_check([np.full((10, 10), 1300.0)], [scl])
    assert off and p == pytest.approx(1300.0) and npx == 100
    off2, p2, _ = F.dn_offset_check([np.full((10, 10), 400.0)], [scl])
    assert not off2 and p2 == pytest.approx(400.0)
    assert F.dn_offset_check([np.full((3, 3), 400.0)], [np.full((3, 3), 9, np.uint8)])[2] == 0, "구름 화소는 진단에 넣지 않는다"
    # 장면 지수: SCL 4·5 만, 반사도 > 0 만
    red10 = np.full((4, 4), 400, np.uint16); nir10 = np.full((4, 4), 3000, np.uint16)
    sw = np.full((2, 2), 1500, np.uint16)
    s20 = np.array([[4, 5], [6, 9]], np.uint8)
    nd, nq = F.scene_indices(red10, nir10, sw, s20, False)
    assert nd[0, 0] == pytest.approx((0.3 - 0.04) / (0.3 + 0.04), rel=1e-5) and nq[0, 1] == pytest.approx((0.3 - 0.15) / (0.3 + 0.15), rel=1e-5)
    assert np.isnan(nd[1, 0]) and np.isnan(nd[1, 1]), "SCL 6·9 는 유효 관측이 아니다"
    nd_o, _ = F.scene_indices(red10, nir10, sw, s20, True)               # offset −0.1 이면 red 반사도 < 0 → 유효하지 않다
    assert np.isnan(nd_o).all()
    st = np.array([[[0.1, np.nan]], [[0.3, np.nan]], [[0.5, 0.2]]], np.float32)
    comp = F.composite(st)
    assert comp[0, 0] == pytest.approx(0.3) and comp[0, 1] == pytest.approx(0.2)
    inm = np.ones((10, 10), bool)
    v = np.full((10, 10), np.nan, np.float32); v[:3, :] = 0.5; v[0, 0] = 0.7
    out = F.cell_s2_values(v, v, inm)
    assert out["s2_valid_frac"] == pytest.approx(0.3) and out["s2_ndvi_med"] == pytest.approx(0.5) and out["s2_ndvi_sd"] == pytest.approx(np.std(v[:3, :]))
    v2 = np.full((10, 10), np.nan, np.float32); v2[0, :5] = 0.4                # 5 % < 10 %
    assert np.isnan(F.cell_s2_values(v2, v2, inm)["s2_ndvi_med"])
    v3 = np.full((10, 10), np.nan, np.float32); v3[:1, :] = 0.4; v3[1, :9] = 0.4  # 19 화소 < 20, 19 %
    assert np.isnan(F.cell_s2_values(v3, v3, inm)["s2_ndvi_med"])


# ---------------------------------------------------------------- (d) 장면 선택과 타일 배정
def _item(i, tile="MGRS-6WVB", date="2021-07-10", seq=0, cloud=1.0, nodata=0.0, poly=None, href="https://x/B04.tif"):
    poly = poly or [[-151, 68], [-147, 68], [-147, 70], [-151, 70], [-151, 68]]
    hr = {an: href.replace("B04", an) for an in ("red", "nir", "swir16", "scl")}
    return dict(id=f"S2A_{i}", properties={"grid:code": tile, "datetime": f"{date}T21:00:00Z", "s2:sequence": seq, "eo:cloud_cover": cloud,
                                           "s2:nodata_pixel_percentage": nodata, "proj:epsg": 32606},
                geometry={"type": "Polygon", "coordinates": [poly]}, assets={an: {"href": h} for an, h in hr.items()})


@NEED_F
def test_d_scene_selection_and_tiles():
    F = _F()
    assert F.cog_ok(_item(1)) and not F.cog_ok(_item(2, href="s3://sentinel-s2-l2a/tiles/B04.jp2"))
    its = [_item(1, seq=0, cloud=0.5), _item(2, seq=1, cloud=9.0), _item(3, date="2021-07-11", seq=0, cloud=2.0),
           _item(4, date="2021-07-11", seq=0, cloud=1.0)]
    dd = F.dedupe(its)
    ids = sorted(it["id"] for it in dd)
    assert ids == ["S2A_2", "S2A_4"], "같은 날은 s2:sequence 최대, 같으면 운량 최소"
    small = [[-149.5, 68.9], [-149.4, 68.9], [-149.4, 69.1], [-149.5, 69.1], [-149.5, 68.9]]
    cand = [_item(10, date="2021-07-01", cloud=0.0, poly=small), _item(11, date="2021-07-02", cloud=5.0), _item(12, date="2021-07-03", cloud=3.0),
            _item(13, date="2021-07-04", cloud=3.0, nodata=10.0)] + [_item(20 + k, date=f"2021-08-{10 + k}", cloud=20.0 + k) for k in range(5)]
    lon, lat = np.array([-148.0, -148.5, -149.0]), np.array([69.0, 69.2, 69.4])
    sel = F.select_scenes(cand, lon, lat)
    assert [it["id"] for it, _ in sel] == ["S2A_12", "S2A_13", "S2A_11", "S2A_20", "S2A_21", "S2A_22"]
    assert all(r >= 0.9 for _, r in sel) and len(sel) == 6
    assert F.coverage_r(cand[0], lon, lat) == 0.0
    frames = {"A": dict(epsg=32606, x0=400000.0, y0=7700040.0, width_m=109800.0, height_m=109800.0),
              "B": dict(epsg=32606, x0=490200.0, y0=7700040.0, width_m=109800.0, height_m=109800.0)}
    from pyproj import Transformer
    lo, la = Transformer.from_crs("EPSG:32606", "EPSG:4326", always_xy=True).transform(np.array([495000.0, 590000.0]), np.array([7650000.0, 7650000.0]))
    ky, kx = F.cell_index(la, lo)
    w, s, e, n = F.cell_bounds(ky, kx)
    cells = pd.DataFrame(dict(cell=["c1", "c2"], w=w, s=s, e=e, n=n))
    tiles, margin = F.assign_tiles(cells, frames)
    assert list(tiles) == ["A", "B"] and (margin > 0).all(), "꼭짓점 여유가 큰 타일에 배정한다"
    tiles2, _ = F.assign_tiles(cells, frames, cand={"c1": {"B"}, "c2": {"B"}})
    assert list(tiles2) == ["B", "B"], "항목이 셀 중심을 덮는 타일 가운데서 고른다"


# ---------------------------------------------------------------- (e) 라벨 열을 읽지 않는다
@NEED_F
def test_e_reads_no_label_columns(tmp_path):
    F = _F()
    fb = tmp_path / "fb.csv"
    pd.DataFrame(dict(loc_id=[1, 2, 3], lat=[66.0, 66.1, 72.0], lon=[-150.0, -150.1, 126.0], region=["ABoVE_AK", "ABoVE_AK", "Lena_RU"],
                      alt_cm=[11111.0, 22222.0, 33333.0])).to_csv(fb, index=False)
    F.TRACE_READS.clear()
    pts = F.read_points(fb)
    assert list(pts.columns) == ["loc_id", "lat", "lon", "region", "target"] and set(pts.target) == {"Alaska", "Lena"}
    assert all(set(c) <= set(RG.COORD_COLS) for _, c in F.TRACE_READS)
    with pytest.raises(RuntimeError):
        F.read_csv_cols(fb, ["loc_id", "alt_cm"])
    cells, lab = F.cell_table(pts)
    assert "alt_cm" not in cells.columns and "alt_cm" not in lab.columns and cells.n_labels.sum() == 3
    src = (PREP / "xm_landcover_s2_features.py").read_text()
    assert "alt_cm" not in src and "sigma_prior_cm" not in src, "추출 코드에 라벨 열 이름이 있다"


# ---------------------------------------------------------------- (f) 변형 열
def test_f_variant_columns():
    x25 = list(XB.FEATS)
    cols, info = RG.xm_variant_columns("xw", "Alaska", x25)
    assert cols == x25 + RG.XM_GROUPS["WC"] + RG.XM_GROUPS["S2"] and len(cols) == 39 and info["same_as"] == ""
    assert RG.xm_variant_columns("xw_lc", "Alaska", x25)[0] == x25 + RG.XM_GROUPS["WC"]
    cols2, info2 = RG.xm_variant_columns("xw", "Lena", x25, {"WC": True, "S2": False})
    assert cols2 == x25 + RG.XM_GROUPS["WC"] and info2["same_as"] == "xw_lc" and info2["dropped"] == ["S2"]
    assert RG.xm_variant_columns("xw", "Lena", x25, {"WC": False, "S2": False})[0] is None
    assert RG.xm_variant_columns("xw_lc", "Lena", x25, {"S2": True})[0] is None, "결정이 없는 군은 빼는 쪽으로 닫는다"
    assert len(RG.XM_GROUPS["WC"]) == 11 and len(RG.XM_GROUPS["S2"]) == 3 and RG.method_suffix("xw_lc") == "@xw_lc"
    assert RG.STAGE_VARIANTS["xm"] == ("xw", "xw_lc") and "xw" not in RG.VARIANTS_ALL, "XE 의 변형 목록은 바뀌지 않는다"
    with pytest.raises(ValueError):
        RG.xm_variant_columns("xh", "Alaska", x25)


# ---------------------------------------------------------------- (g) 하네스
def test_g_args_and_fallback(tmp_path):
    fp = _xm_feat(tmp_path)
    a = M.parse_args(["--stage", "xm", "--feat-xm", str(fp), "--count-only"])
    assert a.VARIANTS == ["xw", "xw_lc"] and a.TAG == "xm" and a.OUT == M.XM_OUT and a.SHARDS == M.XM_OUT / "shards"
    assert a.GATE_LEVEL == "local_rescale" and a.X25_TAG == "xe_r1b" and a.X25_FROM == M.XM_X25_FROM and a.FEAT.is_dummy and not a.FEAT_XM.is_dummy
    assert a.SEALED_ROOT / a.SEALED_NAME == XB.XBATCH_ROOT / "XM_landcover_vegetation"
    assert a.GRID == [200, 500, 1000, -1] and a.SPLIT_LIST == list(range(1, 26)) and a.seeds == 2 and a.cb_iters == 200
    with pytest.raises(SystemExit):
        M.parse_args(["--stage", "xm", "--variants", "x25,xw", "--feat-xm", str(fp)])
    with pytest.raises(SystemExit):
        M.parse_args(["--stage", "r1b", "--variants", "xw", "--feat", "none"])
    with pytest.raises(SystemExit):
        M.parse_args(["--stage", "xm", "--feat-xm", str(tmp_path / "missing.csv")])
    cnt = M.parse_args(["--stage", "xm", "--feat-xm", str(tmp_path / "missing.csv"), "--count-only"])
    assert cnt.FEAT_XM.is_dummy
    assert not a.FEAT_XM.s2_all_dropped() and a.FEAT_XM.decisions("Alaska") == {"WC": True, "S2": True}
    fp2 = _xm_feat(tmp_path, s2_included=False, name="xm_feat_s2none.csv")
    b = M.parse_args(["--stage", "xm", "--feat-xm", str(fp2), "--count-only"])
    assert b.FEAT_XM.s2_all_dropped() and b.FEAT_XM.dropped_targets("S2") == list(M.TARGETS)
    fp3 = _xm_feat(tmp_path, s2_included={"Alaska": True, "Lena": False, "Canada": True}, name="xm_feat_s2lena.csv")
    c3 = M.parse_args(["--stage", "xm", "--feat-xm", str(fp3), "--count-only"])
    assert not c3.FEAT_XM.s2_all_dropped() and c3.FEAT_XM.dropped_targets("S2") == ["Lena"]
    cols, info = M.variant_cols(c3, "xw", "Lena")
    assert info["same_as"] == "xw_lc" and len(cols) == 36
    assert M.needs_final("xw") is False and M.needs_final("xw_lc") is False and M.needs_final("xh") is True


@pytest.mark.skipif(not (ROOT / "data/processed/fidelity_base_v3.csv").exists(), reason="실제 자료(v3)가 없다")
def test_g_fallback_skips_xw_units(tmp_path):
    fp2 = _xm_feat(tmp_path, s2_included=False, name="xm_feat_s2none.csv")
    b = M.parse_args(["--stage", "xm", "--feat-xm", str(fp2), "--count-only", "--targets", "Canada", "--splits", "1,2"])
    units, skipped, _ = M.enumerate_units(b)
    assert units and all(v == "xw_lc" for _, _, v in units)
    assert any(s_["variant"] == "xw" and s_["status"] == M.XM_S2_FALLBACK_TXT for s_ in skipped)


class SpyUnit(M.XERUnit):
    """적합을 하지 않고 학습 행렬과 예측 행렬의 열 수를 기록한다."""

    def __init__(self, *args, **kw):
        self.seen = []
        super().__init__(*args, **kw)

    def fit(self, lr, method, build, nrow, seed, preds, n, d, lab_idx, placement="cell", **tr):
        Xtr, ytr, _ = build()
        self.seen.append((method, Xtr.shape[1], tuple(p.shape[1] for p in preds)))
        return [np.zeros(len(p)) for p in preds]


def test_g_fit_matrices_and_leakage():
    c = make_ctx(n_extra=14, variant="xw")
    a = small_args()
    U = SpyUnit(a, c, "xw")
    U.run()
    assert U.seen and all(s_[1] == 39 and all(q == 39 for q in s_[2]) for s_ in U.seen)
    assert {m for m, *_ in U.seen} == {"R1@xwcv", "R1@xw", "D0@xw"}
    c2 = make_ctx(seed=6, n_extra=14, variant="xw")
    a2 = small_args(grid=(20, 40))

    def run_fn(cc):
        M.XERUnit(a2, cc, "xw").run()
    res = XB.leakage_invariance(run_fn, c2)
    assert res["ok"] and res["same_keys"] and res["max_abs_diff"] == 0.0 and res["n_keys"] > 10
    with XB.h54_trace() as (p1, tr):
        run_fn(c2)
    keep = XB.selected_union(tr)
    c_bad = XB.perturb_labels(c2, keep[1:], seed=3)
    with XB.h54_trace() as (p2, _):
        run_fn(c_bad)
    assert not XB.compare_predictions(p1, p2)["ok"], "선택 라벨을 바꿔도 예측이 같다(시험이 둔하다)"


# ---------------------------------------------------------------- (h) 판정과 문장
def _row(delta, lo, hi, v="미결정", p=0.2, und=False):
    return dict(scope="MEAN", delta=delta, ci_lo=lo, ci_hi=hi, ci_lo_beq=lo, ci_hi_beq=hi, verdict4=v, p_two=p, undetermined=und)


def test_h_verdicts_and_sentences():
    A = "R1(λ cv, xw)−R1(λ cv, x25)"
    pooled = {("XM-a", A, "격자 안", 500): _row(-0.8, -1.2, -0.3, "우세", 0.001), ("XM-a", A, "격자 안", -1): _row(-0.6, -1.0, -0.2, "우세", 0.03)}
    region = [dict(scope="region", hypothesis="XM-a", contrast=A, part="격자 안", n=-1, target="Lena~w|r", delta=0.9, ci_lo=0.2, ci_hi=1.5, worse=True)]
    vr, ht = M.xm_verdict_rows(pooled, region, None, "xw", None)
    assert len(ht) == 3 and set(ht.m) == {3} and (ht.p[ht.label == "XM-a|n1000"] == 1.0).all()
    assert ht.loc[ht.label == "XM-a|n500", "p_holm"].iloc[0] == pytest.approx(0.003)
    v = vr[vr.scope == "verdict"].iloc[0]
    assert v.verdict.startswith("부분 지지") and v.branch == "우세" and v.design == M.XM_DESIGN and v.blind == "비맹검 부분 포함"
    s_ = vr[vr.scope == "sentence"].iloc[0]
    assert s_.sentence.startswith("1 km 셀의 WorldCover 10 m 피복 비율과 Sentinel-2 20 m 여름 식생 지수를 더하면 R1 의 ERA5 격자 안 오차가 0.60 cm 작아졌다")
    assert "보정 전 유의" in s_.sentence and "지역 레나(n 전량) 에서는 오차가 컸다" in s_.sentence and "고치지 않고" in s_.sentence
    assert s_.sentence_n == "전량" and s_.sentence_delta == pytest.approx(-0.6)
    small = {("XM-a", A, "격자 안", n): _row(-0.3, -0.45, -0.1, "우세", 0.001) for n in M.MAIN_N}
    s2 = M.xm_verdict_rows(small, [], None, "xw", None)[0]
    t2 = s2[s2.scope == "sentence"].sentence.iloc[0]
    assert "통계적으로 구별되나 크기는 0.5 cm 미만" in t2 and s2[s2.scope == "verdict"].verdict.iloc[0].startswith("지지")
    eq = {("XM-a", A, "격자 안", n): _row(0.1, -0.3, 0.4, "동등", 0.5) for n in M.MAIN_N}
    t3 = M.xm_verdict_rows(eq, [], None, "xw", None)[0]
    assert t3[t3.scope == "sentence"].sentence.iloc[0].startswith("피복·식생 입력을 더한 R1 의 격자 안 오차는 x25 와 0.5 cm 안에서 같았다")
    und = {("XM-a", A, "격자 안", n): _row(0.1, -0.6, 0.9, "미결정", 0.5) for n in M.MAIN_N}
    t4 = M.xm_verdict_rows(und, [], None, "xw", None, s2_dropped=["Lena"])[0]
    s4 = t4[t4.scope == "sentence"].sentence.iloc[0]
    assert "확인하지 못했다" in s4 and "1 km 셀 사이 몫은 알래스카 18.8 %, 레나 47.8 %, 캐나다 17.7 %" in s4 and "S2 군이 90 % 규칙으로 빠진 대상: 레나" in s4
    na = M.xm_verdict_rows({}, [], None, "xw", None)[0]
    assert na[na.scope == "verdict"].verdict.iloc[0].startswith("판정 불가") and "판정할 수 없었다" in na[na.scope == "sentence"].sentence.iloc[0]
    g = M.xm_verdict_rows(pooled, [], None, "xw", dict(passed=False, status="관문 실패(키 실패 3)"))[0]
    assert g[g.scope == "verdict"].verdict.iloc[0].startswith("판정 불가(관문")
    B = "R1(λ cv, xw_lc)−R1(λ cv, x25)"
    fb = {("XM-a", B, "격자 안", n): _row(-0.7, -1.0, -0.4, "우세", 0.001) for n in M.MAIN_N}
    t5 = M.xm_verdict_rows(fb, [], None, "xw_lc", None, fallback=True)[0]
    s5 = t5[t5.scope == "sentence"].sentence.iloc[0]
    assert s5.startswith("1 km 셀의 WorldCover 10 m 피복 비율을 더하면") and "xw_lc 로 대체했다" in s5
    assert M.xm_row_label("XM-a", 200) == ("XM-d", "보조") and M.xm_row_label("XM-a", -1) == ("XM-a", "주")


# ---------------------------------------------------------------- (i) 합성 조각 집계
def _write_synth(a, x25_dir, xm_feat_sha, ref_dir=None, variants=("xw", "xw_lc"), nA=240):
    """합성 세 대상 × 두 분할: x25 조각은 x25_dir(tag xe_r1b, XE 와 같은 이름), XM 조각은 a.SHARDS(tag xm). ref_dir 이면 x25 를 WF9 이름으로도 쓴다."""
    for k, t in enumerate(("Alaska", "Lena", "Canada")):
        for sp in (1, 2):
            c = make_ctx(seed=10 * k + sp, target=t, split=sp, n_extra=14, nA=nA, variant="xw")
            M.attach_variant(c, "xw_lc", c.xmats["xw"][0][:, :36], c.xmats["xw"][1][:, :36])
            for v in ("x25",) + tuple(variants):
                U = M.XERUnit(a.h, c, v).run()
                rows, stores, stats = U.finish()
                unit = dict(stats, split=sp, dup_of=-1, valid=True, variant=v, exp="xe", feat_group_sha={}, allow_unfinal=False, feat_sha256=xm_feat_sha)
                cfg = XB.make_unit_cfg("xe", v, f"synthetic:{v}", grid=a.GRID, splits=[1, 2])
                if v == "x25":
                    XB.write_shard(x25_dir, "xe_r1b", t, "r", sp, rows, stores, cfg, unit, v, [1, 2], M.__file__)
                    if ref_dir is not None:
                        XB.write_shard(ref_dir, "wf9", t, "r", sp, rows, stores, cfg, unit, "", [1, 2], M.__file__)
                else:
                    XB.write_shard(a.SHARDS, a.TAG, t, "r", sp, rows, stores, cfg, unit, v, [1, 2], M.__file__)


def test_i_summarize_xm_synthetic(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    fp = _xm_feat(tmp_path)
    ref, x25d = tmp_path / "wf9ref", tmp_path / "XE" / "shards"
    a = M.parse_args(["--stage", "xm", "--feat-xm", str(fp), "--out-dir", str(tmp_path / "XM"), "--x25-from", str(x25d), "--nboot", "200",
                      "--grid", "200,all", "--splits", "1,2", "--draws-cap", "1", "--seeds", "1", "--cb-iters", "10", "--gate-wf9", str(ref),
                      "--gate-level", "same_node"])
    _write_synth(a, x25d, a.FEAT_XM.sha, ref_dir=ref)
    capsys.readouterr()
    with XB.restricted_output():
        res = M.summarize(a)
    out = capsys.readouterr().out
    assert res and res["primary"] == "xw" and res["gate"]["passed"]
    assert out.index("[gate]") < out.index("[봉인]"), "재현 관문은 봉인 표보다 먼저 돈다"
    assert not XB.FORBIDDEN_OUT.search(out) and "[봉인]" in out, "집계 화면에 RMSE·Δ·판정이 있다"
    sd = tmp_path / "XM" / "sealed"
    for f in ("xm_tests.csv", "xm_hypotheses.csv", "xm_holm.csv", "xm_decomp.csv", "xm_meta.json", "sealed_manifest.json"):
        assert (sd / f).exists(), f
    tests = pd.read_csv(sd / "xm_tests.csv")
    assert {"XM-a", "XM-b", "XM-c", "XM-d"} <= set(tests.hypothesis) and set(tests.part) >= {"격자 안", "총", "격자 사이", "위치 안", "격자 안·위치 사이"}
    assert not len(tests[(tests.hypothesis == "XM-a") & ~tests.n.isin(M.MAIN_N)]), "n 200 행은 XM-d 다"
    assert set(tests[tests.hypothesis == "XM-a"].variant) == {"xw"} and "xw_lc" in set(tests.variant)
    assert (tests.design == M.XM_DESIGN).all() and (tests.blind == "비맹검 부분 포함").all() and (tests.nboot == 200).all()
    assert (tests[tests.scope == "MEAN"].n_regions_registered == 3).all()
    dec = pd.read_csv(sd / "xm_decomp.csv")
    assert np.nanmax(np.abs(dec.check_tot_w_b.values)) < 1e-6 and np.nanmax(np.abs(dec.check_w_gl_l.values)) < 1e-6 and "xw" in set(dec.variant)
    holm = pd.read_csv(sd / "xm_holm.csv")
    assert len(holm) == 3 and set(holm.m) == {3}
    meta = json.loads((sd / "xm_meta.json").read_text())
    assert meta["n_units_x25"] == 6 and meta["n_units_xm"] == 12 and meta["plan_commit"] == "ff7c97f" and meta["x25_reuse_sha256"]
    assert (tmp_path / "XM" / "xm_x25_reuse.csv").exists() and (tmp_path / "XM" / "xm_gate_wf9.csv").exists()
    # 공통 설정이 다르면 멈춘다
    b = M.parse_args(["--stage", "xm", "--feat-xm", str(fp), "--out-dir", str(tmp_path / "XM2"), "--x25-from", str(x25d), "--nboot", "200",
                      "--grid", "200,all", "--splits", "1,2", "--draws-cap", "1", "--seeds", "1", "--cb-iters", "10", "--gate-wf9", "none"])
    for t in ("Alaska",):
        c = make_ctx(seed=99, target=t, split=1, n_extra=14, nA=240, variant="xw")
        U = M.XERUnit(b.h, c, "xw").run()
        rows, stores, stats = U.finish()
        cfg = XB.make_unit_cfg("xe", "xw", "synthetic:xw", grid=b.GRID, splits=[1, 2], cb_iters=999)
        XB.write_shard(b.SHARDS, b.TAG, t, "r", 1, rows, stores, cfg, dict(stats, split=1, variant="xw"), "xw", [1, 2], M.__file__)
    with pytest.raises(SystemExit):
        M.summarize(b)
