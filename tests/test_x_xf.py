"""scripts/3_deep_learning/x_new_regions.py 단위 시험(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.6 XF, 1절, 0.3, 5절).

합성 자료 시험(라벨 값을 화면에 쓰지 않는다. 실제 v3 는 공변량과 좌표만 빌려 쓴다)
(a) v5 조립: v4 원문 줄이 바이트 단위로 같고, 새 행은 loc_id 30000 부터, region(REGION_OF, 알래스카 하위 과제는 AL-7)·source_id·토양 도일 규약.
    공변량 결측률(covariate_completeness 의 x25_all·eval_ready)이 합성 결측 형태와 같다.
(b) 셀 표지: 6B.4 지리 macro(geo_macro), 지리 정의와 다른 점 자료 macro 는 멈춘다(Svalbard, Scandinavia, Greenland, Yukon 표기 등),
    dup_v4, merge_v4(in_v4 = 1 만), v1_cell_not_v4(표시만), 비상업 약관, 역할, in_v5, xf_target(알래스카 하위 과제 포함).
(c) spec 과 실행 표: 약관 확인분만, NAtlantic 새 셀 0 이면 적합하지 않음, 약관 회신 판, 셀 보강(하위 지역 CA-1 대상과 같은 작업의 기준 표),
    AL-7 F2 전용 별칭. 실행 표는 고른 새 행만 F4_direct, table.dir 은 상대 경로, 하위 지역 대상 표에는 블록 대응표. 확장 블록 대응표 규칙.
(d) 적격 세기: structure15 가 lgd_eligibility_v1.structure(라벨 평균 제외)와 같다. parent 원천 제외. 분할 201–210 은 XB.split_plan_new 규칙.
(e) --count-only: 대상 라벨 값을 바꿔도 화면 출력과 적격·적합 수 표가 같다. 공변량 결측률(xf_x25_all, xf_eval_ready). 동결 뒤 거부.
(f) 누설 시험(1절): run_unit 이 쓰는 compute_unit 에서 선택 밖 A 라벨과 B 라벨을 바꿔도 모든 예측이 같고, 선택 라벨 하나를 바꾸면 달라진다.
(g) 대비·판정: XF-1–XF-3 키, Holm m = 3 × 적격 지역 수(빠진 행 p 1), 빠진 지역 행의 판정 불가 문장과 사유, 다섯 갈래 문장, 적격 0곳 문장,
    NAtlantic v5 점 추정(첫 세기 마감 규칙), 셀 보강의 L40 형식(L1 조건, R1 − P0, P1 − P0, L28 분류), XF-4 행.
(h) 부록 XC-F3: 0곳이면 '시험하지 못함', 새 macro 만, 스크립트·v5·spec 해시, 상대 경로, F2 전용 별칭, 최종판 한 번만 쓰기(적격 동결).
(i) 끝에서 끝: (spec, 분할) 단위 실행이 compute_unit 을 지난다, 조각 이름, 공통 설정 해시 하나, --resume, 봉인 집계, 조각 표 해시 대조.
(j) build_ext_cells_v1 재사용: 입력·산출 폴더만 바꿔 부르고 화면 출력이 없으며 메타에 라벨 통계 항목이 없다. 공변량 단계는 전역 변수만 바꾼다.
(k) 실행 보호와 로컬 자원: 허용 표지 없이 본 실행 거부, R4 는 XF 자료 마감과 최종 XC-F3 뒤에만, --allow-local 도 30 GB 대기·RSS 감시·
    워커 × 스레드 32 이하, 스모크 규약(코어 4, 스레드 2) 어긋나면 거부, --shard, --units.
(m) 마감: 점 자료 목록 기록과 마감 뒤 동결(목록 밖·바뀐 파일 거부), NAtlantic v5 첫 세기 기록(한 번만)과 점 추정 규칙.
(n) 적격 동결: 최종 XC-F3 해시 대조(check_frozen), register_xf_aliases 의 동결 요구.
(o) XB 진입점: v5 연도 정합 도일(write_tdd_v5), XF 별칭 등록과 XB tdd_tables 연결(xb_prepare), XB 의 별칭 해석, --workers 0 요구.
실제 자료 시험(HEAVY, XF_HEAVY=1 일 때만): (l) 재현 관문(v4 메타, LGD 실행 표 4개의 실행 기록 해시, Tibet·NAtlantic_lic 재생성 바이트, 부록 A).
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c <코어 4개> python3 -m pytest -q -p no:cacheprovider tests/test_x_xf.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
XF = _load("x_new_regions", ROOT / "scripts" / "3_deep_learning" / "x_new_regions.py")
H, X, W, H4 = XB.H, XB.X, XB.W, XB.H4

V3 = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
S3 = ROOT / "data" / "processed" / "e5_soil_tdd_v3.csv"
REAL = pytest.mark.skipif(not (V3.exists() and S3.exists()), reason="v3 표가 없다(공변량을 빌려 쓰는 합성 시험)")
HEAVY = pytest.mark.skipif(os.environ.get("XF_HEAVY", "") != "1", reason="실제 자료 재현 관문(XF_HEAVY=1)")
BEFORE = XF.DEADLINE_DATA - XF.dt.timedelta(days=1)
AFTER = XF.DEADLINE_DATA + XF.dt.timedelta(hours=1)
KNOWN = {"Tibet", "Tibet_LGD", "NAtlantic", "Canada", "Alaska", "Lena", "Svalbard", "Scandinavia", "Alps", "Mongolia_CAsia"}


@pytest.fixture(autouse=True)
def _roots(tmp_path, monkeypatch):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    monkeypatch.delenv("WF_RESCALE", raising=False)
    monkeypatch.delenv("LG_RESCALE", raising=False)
    saved = dict(XB.RUN_TABLES)
    yield
    XB.RUN_TABLES.clear()
    XB.RUN_TABLES.update(saved)


# ---------------------------------------------------------------- 합성 실행 표(공변량은 실제 v3 행을 빌린다. 대상 라벨은 합성)
_V3C = {}


def _v3():
    if "v3" not in _V3C:
        _V3C["v3"] = pd.read_csv(V3, low_memory=False)
        _V3C["s3"] = pd.read_csv(S3)
    return _V3C["v3"], _V3C["s3"]


def synth_table(d, target="Testland", nbl=(4, 4), per_block=4, seed=0, label_scale=1.0, loc0=30000):
    """원천 = v3 의 CALM 러시아 W·E·C 행(실제 좌표, 라벨은 합성), 대상 = v3 캐나다 행의 공변량을 빌린 새 셀(위도 50–52°, 경도 95–97°,
    0.5° 블록 nbl[0] × nbl[1]). 대상 라벨 = label_scale × 1.5·√TDD·exp(0.15ε). 반환 (디렉터리, 대상 loc_id)."""
    v3, s3 = _v3()
    rng = np.random.RandomState(seed)
    f4 = v3[v3.source_id == "F4_direct"]
    src = pd.concat([f4[f4.region.isin(["CALM_Russia_W", "CALM_Russia_E", "CALM_Russia_C"])],     # h40.Data 의 하위 지역 k-평균이 캐나다·레나를 요구한다
                     f4[f4.region == "Lena_RU"].sample(60, random_state=7), f4[f4.region == "ABoVE_CA"].sample(60, random_state=8)]).copy()
    src["alt_cm"] = np.clip(1.4 * src.e5_sqrt_tdd.values * np.exp(0.2 * np.random.RandomState(1).randn(len(src))), 5, 590)
    pool = v3[(v3.region == "ABoVE_CA") & v3.cci_alt.notna()].merge(s3[s3.e5_sqrt_tdd_soil.notna()][["loc_id"]], on="loc_id")
    nt = nbl[0] * nbl[1] * per_block
    tg = pool.sample(nt, random_state=seed).copy()
    old_ids = tg.loc_id.values.copy()
    tg["loc_id"] = loc0 + np.arange(nt)
    k = np.arange(nt)
    bi, bj, w = (k // per_block) // nbl[1], (k // per_block) % nbl[1], k % per_block
    tg["lat"] = 50.0 + 0.5 * bi + 0.05 + 0.1 * w
    tg["lon"] = 95.0 + 0.5 * bj + 0.05 + 0.1 * w
    tg["block"] = np.floor(tg.lat / 0.5).astype(int) * 100000 + np.floor(tg.lon / 0.5).astype(int)
    tg["region"] = target
    tg["alt_cm"] = label_scale * 1.5 * tg.e5_sqrt_tdd.values * np.exp(0.15 * rng.randn(nt))
    tab = pd.concat([src, tg], ignore_index=True)[list(v3.columns)]
    sl = s3[s3.loc_id.isin(src.loc_id)]
    st = s3.set_index("loc_id").loc[old_ids].reset_index()
    st["loc_id"] = tg.loc_id.values
    st["lat"], st["lon"] = tg.lat.values, tg.lon.values
    soil = pd.concat([sl, st], ignore_index=True)[list(s3.columns)]
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    tab.to_csv(d / "fidelity_base_v3.csv", index=False)
    soil.to_csv(d / "e5_soil_tdd_v3.csv", index=False)
    return d, tg.loc_id.values.astype(int)


def synth_spec(P, alias="Testland~xf", target="Testland", seed=0, label_scale=1.0, nbl=(4, 4)):
    d, ids = synth_table(XF.spec_dir(P, alias), target=target, seed=seed, label_scale=label_scale, nbl=nbl)
    s = dict(alias=alias, target=target, kind=XF.KIND_NEW, blind="맹검", keep_v4=[], keep_v5=[int(v) for v in ids], drop_v3=[],
             n_lic_dropped_v4=0, n_lic_dropped_v5=0, note="", run=True, run_skip="",
             table=dict(dir=os.path.relpath(d, P.root), sha256=XB.sha256_file(d / "fidelity_base_v3.csv")))
    return s


def no_forbidden(text):
    bad = [ln for ln in str(text).splitlines() if XB.FORBIDDEN_OUT.search(ln)]
    assert not bad, f"출력 제한 패턴이 화면에 있다: {bad[:3]}"


# ---------------------------------------------------------------- (a) v5 조립과 공변량 결측률
def _v3_header():
    with open(V3) as f:
        return f.readline().rstrip("\n").split(",")


@REAL
def test_a_v5_assembly(tmp_path):
    cols = _v3_header()
    scols = pd.read_csv(S3, nrows=0).columns.tolist()
    rng = np.random.RandomState(0)
    v4 = pd.DataFrame({c: rng.rand(4).round(5) for c in cols})
    v4["loc_id"] = [1, 2, 20000, 20001]
    v4["region"] = ["ABoVE_AK", "Lena_RU", "Tibet_LGD", "NAtlantic"]
    v4["source_id"] = ["F4_direct", "F4_direct", "F4_ext_direct", "F4_ext_direct"]
    p4, ps4 = tmp_path / "v4.csv", tmp_path / "soil4.csv"
    v4.to_csv(p4, index=False)
    pd.DataFrame({c: rng.rand(4).round(4) for c in scols}).assign(loc_id=[1, 2, 20000, 20001], src="x").to_csv(ps4, index=False)
    cells = pd.DataFrame(dict(cell_uid=["direct|1_1", "direct|2_2", "direct|3_3", "direct|4_4", "direct|5_5"], label_set="direct",
                              macro=["Testland", "Tibet", "NAtlantic", "Testland", "Alaska"], ky=[3, 1, 2, 4, 5], kx=[3, 1, 2, 4, 5],
                              lat=[50.1, 33.0, 68.3, 50.3, 64.8], lon=[95.1, 92.0, 19.0, 95.3, -147.7], alt_cm=[80.0, 200.0, 70.0, 90.0, 60.0],
                              alt_sd_all=[np.nan, 5.0, 50.0, 1.0, 2.0], in_v5=[1, 1, 1, 0, 1],
                              xf_role=["new_macro", "augment_L40", "natl_v5", "new_macro", "alaska_subtask"],
                              xf_target=[1, 1, 1, 0, 1], license="CC BY 4.0", lic_unverified=0, lic_nc=0))
    feat = [c for c in cols if c.startswith(("dem_", "e5_", "sg_", "cci_"))]
    cov = pd.DataFrame({c: rng.rand(4) + 1 for c in feat})
    cov["cell_uid"] = ["direct|1_1", "direct|2_2", "direct|3_3", "direct|5_5"]
    for c in ("e5_tdd_soil", "e5_fdd_soil", "e5_sqrt_tdd_soil", "e5_stl1_twarm", "e5_stl1_tcold", "soil_fallback_deg"):
        cov[c] = [10.0, -1.0, 4.0, 9.0] if c == "e5_tdd_soil" else 1.0
    cov.loc[cov.cell_uid == "direct|3_3", "sg_clay_5_15"] = np.nan             # 공변량 결측 형태: NAtlantic 셀의 x25 하나
    cov.loc[cov.cell_uid == "direct|5_5", "cci_alt"] = np.nan                  # 알래스카 셀의 CCI 결측(채점 불가)
    P = XF.Paths(tmp_path)
    meta = XF.build_v5(P, cells, cov, v4_table=p4, v4_soil=ps4)
    assert meta["v4_rows_bytes_equal"] and meta["soil_v4_rows_bytes_equal"] and meta["n_rows"]["new"] == 4
    l4, l5 = p4.read_text().splitlines(), P.v5_table.read_text().splitlines()
    assert l5[:len(l4)] == l4 and len(l5) == len(l4) + 4
    t5 = pd.read_csv(P.v5_table)
    new = t5[t5.loc_id >= XF.LOC0_V5].sort_values("loc_id")
    assert new.loc_id.tolist() == [30000, 30001, 30002, 30003]
    assert new.region.tolist() == [XF.AL7, "NAtlantic", "Testland", "Tibet_LGD"]       # macro, ky, kx 순. 알래스카 하위 과제 = AL-7
    assert (new.source_id == "F4_ext_direct").all() and (new.spatial_support_m == 1000).all() and (new.right_censored == 0).all()
    assert new.sigma_prior_cm.tolist() == [3.0, 40.0, 12.0, 5.0]                    # 3–40 자름, 없으면 12
    s5 = pd.read_csv(P.v5_soil)
    tib = s5[s5.loc_id == 30003].iloc[0]
    assert np.isnan(tib.e5_tdd_soil) and np.isnan(tib.e5_sqrt_tdd_soil)                 # e5_tdd_soil <= 0 → 결측(v3 규칙)
    lab = pd.read_csv(P.v5_labels)
    assert "alt_cm" not in lab.columns and lab.macro_v5.tolist() == [XF.AL7, "NAtlantic", "Testland", "Tibet_LGD"]
    assert {"block", "lat", "lon", "geo_macro"} <= set(lab.columns)
    # 공변량 결측률(2.6 '시험: 공변량 결측률'): 합성 결측 형태와 같아야 한다
    comp = meta["covariate_completeness"]
    assert comp["natl_v5|NAtlantic"]["x25_all"] == 0.0 and comp["new_macro|Testland"]["x25_all"] == 1.0
    assert comp["alaska_subtask|Alaska"]["x25_all"] == 0.0 and comp["alaska_subtask|Alaska"]["eval_ready"] == 0
    assert comp["augment_L40|Tibet"]["eval_ready"] == 0 and comp["augment_L40|Tibet"]["soil_tdd"] == 0.0     # 토양 도일 결측
    assert comp["new_macro|Testland"]["eval_ready"] == 1 and comp["natl_v5|NAtlantic"]["eval_ready"] == 1
    # 새 행이 없으면 v5 는 v4 와 바이트가 같다
    P2 = XF.Paths(tmp_path / "zero")
    XF.build_v5(P2, cells.iloc[:0].assign(in_v5=[]), cov.iloc[:0], v4_table=p4, v4_soil=ps4)
    assert P2.v5_table.read_bytes() == p4.read_bytes() and P2.v5_soil.read_bytes() == ps4.read_bytes()


# ---------------------------------------------------------------- (b) 셀 표지와 6B.4 지리 macro
@pytest.mark.parametrize("country,lat,lon,expect", [
    ("Norway", 78.2, 15.8, "NAtlantic"), ("Svalbard and Jan Mayen", 78.9, 11.9, "NAtlantic"), ("Greenland", 69.2, -51.1, "NAtlantic"),
    ("Denmark", 72.0, -40.0, "NAtlantic"), ("Sweden", 68.35, 19.05, "NAtlantic"), ("Finland", 69.8, 27.2, "NAtlantic"),
    ("Canada", 64.0, -139.4, "Canada"), ("Russia", 72.5, 126.0, "Lena"), ("Russia", 67.0, 60.0, "Russia_W"), ("Russia", 62.0, 129.7, "Russia_C"),
    ("Russia", 68.0, 161.0, "Russia_E"), ("Russia", 66.0, -173.0, "Russia_E"), ("United States", 64.8, -147.7, "Alaska"), ("USA", 40.0, -105.0, ""),
    ("China", 34.0, 92.0, "Tibet"), ("China", 52.0, 122.0, ""), ("Mongolia", 48.0, 100.0, "Mongolia_CAsia"), ("Iceland", 64.0, -19.0, "")])
def test_b0_geo_macro(country, lat, lon, expect):
    assert XF.geo_macro(country, lat, lon)[0] == expect


def _cell(macro, country, lat, lon, uid="direct|1_1"):
    c = XF.empty_cells().reindex(range(1)).copy()
    c["cell_uid"], c["label_set"], c["macro"], c["country"], c["lat"], c["lon"] = uid, "direct", macro, country, lat, lon
    c["dup_v3"], c["indep_ok"], c["license"] = 0, 1, "CC BY 4.0"
    return c


V4C0 = pd.DataFrame(columns=["cell_uid", "label_set", "lat", "lon", "in_v4", "excl_reason"])


@pytest.mark.parametrize("macro,country,lat,lon", [("Svalbard", "Norway", 78.2, 15.8), ("Scandinavia", "Sweden", 68.35, 19.05),
                                                   ("Greenland", "Greenland", 69.2, -51.1), ("Yukon", "Canada", 64.0, -139.4),
                                                   ("other", "Iceland", 64.0, -19.0), ("Tibet", "Mongolia", 48.0, 100.0),
                                                   ("Alps", "Iceland", 64.0, -19.0), ("Iceland", "", 64.0, -19.0)])
def test_b1_geo_mismatch_stops(macro, country, lat, lon):
    with pytest.raises(SystemExit):
        XF.classify_cells(_cell(macro, country, lat, lon), V4C0, known=KNOWN)


@pytest.mark.skipif(not XF.V4_TABLE.exists(), reason="v4 표가 없다")
@pytest.mark.parametrize("macro,country,lat,lon", [("Greenland", "Greenland", 69.2, -51.1), ("Yukon", "Canada", 64.0, -139.4),
                                                   ("Svalbard", "Norway", 78.2, 15.8), ("Scandinavia", "Sweden", 68.35, 19.05)])
def test_b1r_review_cases_with_real_known_macros(macro, country, lat, lon):
    """검토 지적의 네 경우: 실제 known_macros() 로도 새 macro·셀 보강이 되지 않고 멈춘다."""
    with pytest.raises(SystemExit):
        XF.classify_cells(_cell(macro, country, lat, lon), V4C0, known=XF.known_macros())


def test_b2_classify_cells():
    cells = XF.empty_cells().reindex(range(8)).copy()
    cells["cell_uid"] = [f"direct|{i}_{i}" for i in range(7)] + ["temp|9_9"]
    cells["label_set"] = ["direct"] * 7 + ["temp"]
    cells["macro"] = ["Testland", "NAtlantic", "Tibet", "Alaska", "Testland", "Testland", "Canada", "Testland"]
    cells["country"] = ["Testland", "Sweden", "China", "United States", "Testland", "Testland", "Canada", "Testland"]
    cells["lat"] = [50.0, 68.3, 33.0, 65.0, 60.0, 61.0, 62.0, 50.0]
    cells["lon"] = [95.0, 19.0, 92.0, -150.0, 100.0, 101.0, -110.0, 95.0]
    cells["dup_v3"] = [0, 0, 0, 0, 0, 1, 0, 0]
    cells["indep_ok"] = [1, 1, 1, 1, 0, 1, 1, 1]
    cells["license"] = ["CC BY 4.0", "CC BY 4.0", "CC BY-NC-SA 3.0"] + ["CC BY 4.0"] * 5
    v4c = pd.DataFrame(dict(cell_uid=["direct|0_0", "direct|77_77", "direct|6_6"], label_set="direct", lat=[10.0, 68.305, 0.0],
                            lon=[10.0, 19.004, 0.0], in_v4=[1, 1, 0], excl_reason=["", "", "macro=Alaska"]))
    c = XF.classify_cells(cells, v4c, known=KNOWN)
    assert c.xf_role.tolist() == ["new_macro", "natl_v5", "augment_L40", "alaska_subtask", "new_macro", "new_macro", "augment_L40", "new_macro"]
    assert c.geo_macro.tolist() == ["", "NAtlantic", "Tibet", "Alaska", "", "", "Canada", ""]
    assert c.merge_v4.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]                             # v4 에 들어간 셀만
    assert c.v1_cell_not_v4.tolist() == [0, 0, 0, 0, 0, 0, 1, 0]                       # v4 밖 v1 셀은 표시만
    assert c.dup_v4.tolist() == [0, 1, 0, 0, 0, 0, 0, 0]                               # 체비쇼프 0.01° 이내
    assert c.lic_nc.tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    assert c.in_v5.tolist() == [0, 0, 0, 1, 1, 0, 1, 0]
    assert c.xf_target.tolist() == [0, 0, 0, 1, 0, 0, 1, 0]                            # 알래스카 하위 과제도 대상(AL-7), 독립성 실패 제외
    assert "merge_v4" in c.xf_excl[0] and "license_nc" in c.xf_excl[2] and "indep<100km" in c.xf_excl[4]
    assert c.xf_excl[6] == "" and "v1_cell_not_v4(macro=Alaska" in c.xf_note[6] and "alaska_subtask" in c.xf_note[3]


def _points(path, src="xf_test", macro="Testland", country="XX", lat0=50.1, lon0=95.2):
    cols = ["src_id", "site_id", "site_name", "lat", "lon", "year", "month", "alt_cm", "method", "label_def", "n_obs", "country", "macro",
            "citation", "license", "subunit", "date", "eos_basis", "value_kind", "year_min", "year_max", "alt_sd_cm", "coord_prec_deg",
            "disturbed", "disturb_type", "right_censored", "orig_source", "dup_of", "qc_flag", "notes"]
    rows = []
    for i in range(6):
        for y in (2015, 2016):
            rows.append(dict(src_id=src, site_id=f"S{i}", site_name=f"S{i}", lat=lat0 + 0.6 * i, lon=lon0 + 0.05 * i, year=y, month=8,
                             alt_cm=60 + 7 * i + (y - 2015), method="probe", label_def="direct_eos", n_obs=1, country=country, macro=macro,
                             citation="test", license="CC BY 4.0", subunit="", date=f"{y}-08-20", eos_basis="record_date", value_kind="single_visit",
                             qc_flag="ok", disturbed=0, right_censored=0))
    pd.DataFrame(rows).reindex(columns=cols).to_csv(path, index=False)
    return path


def test_b3_check_point_macros(tmp_path):
    ok = _points(tmp_path / "a_points.csv", "a", macro="NAtlantic", country="Norway", lat0=78.0, lon0=15.0)
    assert XF.check_point_macros([ok], KNOWN) == 12
    bad = _points(tmp_path / "b_points.csv", "b", macro="Svalbard", country="Norway", lat0=78.0, lon0=15.0)
    with pytest.raises(SystemExit):
        XF.check_point_macros([ok, bad], KNOWN)


# ---------------------------------------------------------------- (c) spec 과 실행 표
CA_BLK1, CA_BLK2 = 120 * 100000 + (-230), 124 * 100000 + (-260)


def _submap():
    return pd.DataFrame(dict(parent=["Canada", "Canada"], block=[CA_BLK1, CA_BLK2], subregion=["CA-1", "CA-2"], n_cells=[5, 5],
                             lat=[60.2, 62.2], lon=[-114.8, -129.8], extra=[0, 0]))


def _labs():
    lab_v4 = pd.DataFrame(dict(loc_id=[20000, 20001, 20002, 20003, 20004], part="new", lgd_role="target",
                               macro_v4=["NAtlantic", "NAtlantic", "Tibet_LGD", "Tibet_LGD", "Canada"], lic_unverified=[0, 1, 0, 0, 0],
                               sources=["grl_disko", "calm_web_subsites", "qtp_du_gpr", "qtp_du_gpr", "ggd353"]))
    lab_xf = pd.DataFrame(dict(loc_id=[30000, 30001, 30002, 30003, 30004, 30005], xf_target=[1] * 6,
                               xf_role=["new_macro", "new_macro", "augment_L40", "natl_v5", "augment_L40", "alaska_subtask"],
                               macro_v5=["Testland", "Testland", "Tibet_LGD", "NAtlantic", "Canada", XF.AL7], lic_unverified=[0, 1, 0, 0, 0, 0],
                               block=[0, 0, 0, 0, CA_BLK1, 0], sources=["a", "b", "qtp_xiao", "swe_stordalen_crill", "naat", "viper"]))
    return lab_v4, lab_xf


def test_c_specs_rules():
    lab_v4, lab_xf = _labs()
    s = XF.make_specs(lab_v4, lab_xf, submap=_submap())
    assert set(s) == {"Testland~xf", XF.NATL_ALIAS, "Tibet_LGD~aug~v5", "Tibet_LGD~aug~ref", "CA-1~aug~v5", "CA-1~aug~ref", XF.AL7_ALIAS}
    assert s["Testland~xf"]["keep_v5"] == [30000] and s["Testland~xf"]["n_lic_dropped_v5"] == 1 and s["Testland~xf"]["blind"] == "맹검"
    n = s[XF.NATL_ALIAS]
    assert n["keep_v4"] == [20000] and n["keep_v5"] == [30003] and n["run"] and n["blind"] == "비맹검 부분 포함"
    t, tr = s["Tibet_LGD~aug~v5"], s["Tibet_LGD~aug~ref"]
    assert t["keep_v4"] == [20002, 20003] and t["keep_v5"] == [30002] and t["ref"] == "Tibet_LGD~aug~ref" and "parent" not in t
    assert tr["keep_v4"] == [20002, 20003] and tr["keep_v5"] == [] and tr["kind"] == XF.KIND_AUG_REF and tr["parts"] == ["lg"] and tr["run"]
    ca, car = s["CA-1~aug~v5"], s["CA-1~aug~ref"]
    assert ca["target"] == "CA-1" and ca["parent"] == "Canada" and ca["keep_v4"] == [20004] and ca["keep_v5"] == [30004]
    assert car["target"] == "CA-1" and car["parent"] == "Canada" and car["keep_v5"] == []
    ak = s[XF.AL7_ALIAS]
    assert ak["target"] == XF.AL7 and ak["kind"] == XF.KIND_AK and ak["keep_v5"] == [30005] and not ak["run"] and ak["f2_only"]
    s0 = XF.make_specs(lab_v4, lab_xf[lab_xf.xf_role != "natl_v5"], submap=_submap())
    assert not s0[XF.NATL_ALIAS]["run"] and "새 셀 0" in s0[XF.NATL_ALIAS]["run_skip"]
    sf = XF.make_specs(lab_v4, lab_xf, natl_confirmed=("calm_web_subsites",), submap=_submap())
    assert sf[XF.NATL_FULL_ALIAS]["keep_v4"] == [20000, 20001]
    with pytest.raises(ValueError):
        XF.make_specs(lab_v4, lab_xf.assign(macro_v5=["Bad__name"] * 2 + list(lab_xf.macro_v5[2:])), submap=_submap())
    with pytest.raises(SystemExit):                                                    # 하위 지역 대응표에 없는 블록
        XF.make_specs(lab_v4, lab_xf.assign(block=[0, 0, 0, 0, 999, 0]), submap=_submap())


def test_c2_extended_submap(tmp_path):
    mp = tmp_path / "map.csv"
    _submap().drop(columns=["extra"]).to_csv(mp, index=False)
    t = pd.DataFrame(dict(loc_id=[1, 2, 3, 30000], lat=[60.1, 62.1, 62.4, 62.3], lon=[-114.9, -129.9, -131.2, -131.4],
                          region=["ABoVE_CA", "ABoVE_CA", "Canada", "Canada"]))
    t.to_csv(tmp_path / "v5.csv", index=False)
    sm = XF.extended_submap(tmp_path / "v5.csv", mp)
    orig = sm[sm.extra == 0]
    assert orig[["parent", "block", "subregion"]].values.tolist() == _submap()[["parent", "block", "subregion"]].values.tolist()
    ex = sm[sm.extra == 1]
    assert len(ex) == 1 and ex.subregion.iloc[0] == "CA-2" and int(ex.block.iloc[0]) == 124 * 100000 + int(np.floor(-131.3 / 0.5))
    assert XF.sub_of(sm, "Canada", CA_BLK1) == "CA-1" and XF.sub_of(sm, "Canada", 1) == ""


@REAL
def test_c3_run_table_text(tmp_path):
    cols = _v3_header()
    rng = np.random.RandomState(1)
    t = pd.DataFrame({c: rng.rand(6).round(5) for c in cols})
    t["loc_id"] = [1, 2, 20000, 20001, 30000, 30001]
    t["region"] = ["CALM_Russia_W", "Lena_RU", "NAtlantic", "NAtlantic", "NAtlantic", "Testland"]
    t["source_id"] = ["F4_direct", "F4_direct", "F4_ext_direct", "F4_ext_direct", "F4_ext_direct", "F4_ext_direct"]
    p5 = tmp_path / "v5.csv"
    t.to_csv(p5, index=False)
    soil = tmp_path / "soil.csv"
    soil.write_text("loc_id\n")
    P = XF.Paths(tmp_path)
    specs = {"NAtlantic~lic~v5": dict(alias="NAtlantic~lic~v5", target="NAtlantic", kind=XF.KIND_NATL, keep_v4=[20000], keep_v5=[30000], drop_v3=[]),
             "v4only": dict(alias="v4only", target="NAtlantic", kind=XF.KIND_NATL, keep_v4=[20000], keep_v5=[], drop_v3=[]),
             "CA-1~aug~v5": dict(alias="CA-1~aug~v5", target="CA-1", parent="Canada", kind=XF.KIND_AUG, keep_v4=[], keep_v5=[], drop_v3=[])}
    XF.write_tables(P, specs, p5, soil, submap=_submap())
    out = (P.runs / "NAtlantic~lic~v5" / "fidelity_base_v3.csv").read_text().splitlines()
    src = p5.read_text().splitlines()
    assert out[:3] == src[:3]                                                           # 머리글과 v3 행은 원문 줄 그대로
    ids = [int(ln.split(",")[0]) for ln in out[1:]]
    assert ids == [1, 2, 20000, 30000]
    isrc = cols.index("source_id")
    assert all(ln.split(",")[isrc] == "F4_direct" for ln in out[1:])
    assert os.path.islink(P.runs / "NAtlantic~lic~v5" / "e5_soil_tdd_v3.csv")
    # v4 만 고른 표는 v4 경로(lgd_eligibility_v1.write_run_table_text)와 바이트가 같다
    p4 = tmp_path / "v4.csv"
    p4.write_text("\n".join(src[:5]) + "\n")
    XF.E1.write_run_table_text(p4, tmp_path / "ref.csv", [20000])
    assert (P.runs / "v4only" / "fidelity_base_v3.csv").read_bytes() == (tmp_path / "ref.csv").read_bytes()
    js = json.loads(P.specs.read_text())
    assert js["specs"]["NAtlantic~lic~v5"]["table"]["n_new"] == 2
    for al, sp in js["specs"].items():                                                 # 표 경로는 산출 뿌리 기준 상대 경로(Rescale 노드에서도 같은 곳)
        assert not os.path.isabs(sp["table"]["dir"]) and (P.root / sp["table"]["dir"]).resolve() == XF.spec_dir(P, al).resolve()
    assert (P.runs / "CA-1~aug~v5" / XF.SUBMAP_NAME).exists() and not (P.runs / "v4only" / XF.SUBMAP_NAME).exists()
    assert js["specs"]["CA-1~aug~v5"]["table"]["submap_sha256"]


# ---------------------------------------------------------------- (d) 적격 세기
def _fake_df(nb=12, per=3, seed=0, few_eval=False):
    rng = np.random.RandomState(seed)
    n = nb * per
    lat = 60 + 0.5 * np.repeat(np.arange(nb), per) + 0.1
    lon = 30 + 0.1 * np.tile(np.arange(per), nb)
    t = pd.DataFrame(dict(lat=lat, lon=lon, macro="T", alt_cm=50 + rng.rand(n), cci_alt=1.0, e5_sqrt_tdd_soil=20.0, loc_id=np.arange(n)))
    if few_eval:
        t.loc[t.index[:-per * 2], "cci_alt"] = np.nan
    s = pd.DataFrame(dict(lat=10 + rng.rand(40), lon=10 + rng.rand(40), macro="S", alt_cm=50 + rng.rand(40), cci_alt=1.0, e5_sqrt_tdd_soil=20.0,
                          loc_id=1000 + np.arange(40)))
    df = pd.concat([t, s], ignore_index=True)
    df["block"] = np.floor(df.lat / 0.5).astype(int) * 100000 + np.floor(df.lon / 0.5).astype(int)
    return df


@pytest.mark.parametrize("few_eval", [False, True])
def test_d_structure_equals_lgd_eligibility(few_eval):
    df = _fake_df(few_eval=few_eval)
    t_idx = XF.target_index(df, "T")
    mine, info = XF.structure15(df, t_idx)
    ref, rinfo, _ = XF.E1.structure(df, "T")
    for k, v in ref.items():
        if k == "alt_mean":
            assert "alt_mean" not in mine                                               # 라벨 평균은 계산하지 않는다
            continue
        assert (v == mine[k]) or (isinstance(v, float) and np.isnan(v) and np.isnan(mine[k])), k
    for sp in info:
        assert {k: info[sp][k] for k in ("dup_of", "mirror_of", "n_A", "nb_eval", "valid")} == {k: rinfo[sp][k] for k in ("dup_of", "mirror_of", "n_A", "nb_eval", "valid")}
    rs = XF.run_splits15(info)
    assert all(info[s]["dup_of"] < 0 for s in rs)
    D = type("D", (), dict(df=df, target_idx=lambda self, t: t_idx))()
    keep, rows = XB.split_plan_new(D, "T", XF.XC_SPLITS, XB.PRIOR_LG, XB.MIN_EVAL_BLOCKS)
    assert [r["split"] for r in rows] == list(XF.XC_SPLITS) and set(keep) <= set(XF.XC_SPLITS)
    assert XF.structure15(df, t_idx, parent="S")[0]["n_src"] == 0                      # 하위 지역 대상(모드 x)은 parent 를 원천에서 뺀다


# ---------------------------------------------------------------- (e) --count-only: 라벨 값 미사용, 공변량 결측률, 동결 뒤 거부
def _count_run(base, capsys, label_scale):
    P = XF.Paths(base)
    spec = synth_spec(P, label_scale=label_scale)
    XF.save_specs(P, {spec["alias"]: spec})
    rc = XF.main(["--count-only", "--out-root", str(base), "--threads", "2", "--min-mem-gb", "0", "--max-mem-gb", "0"])
    out = capsys.readouterr().out
    out = "\n".join(ln for ln in out.splitlines() if not ln.strip().startswith("[자원]"))   # 최대 RSS 는 실행마다 다르다
    return rc, out.replace(str(P.root), "<ROOT>"), P


@REAL
def test_e_count_only_label_free(tmp_path, capsys):
    rc1, o1, P1 = _count_run(tmp_path / "a", capsys, 1.0)
    rc2, o2, P2 = _count_run(tmp_path / "b", capsys, 1.7)
    assert rc1 == 0 and rc2 == 0
    assert o1 == o2, "대상 라벨 값을 바꾸자 화면 출력이 달라졌다(라벨 통계가 출력된다)"
    no_forbidden(o1)
    assert "[출력 제한]" not in o1, "출력 제한이 줄을 걸렀다(세기 출력에 RMSE·Δ·판정 문자열이 있었다)"
    for f in ("elig", "elig_splits", "count", "count_sum"):
        a_, b_ = pd.read_csv(getattr(P1, f)), pd.read_csv(getattr(P2, f))
        a_, b_ = a_.drop(columns=["table_sha256"], errors="ignore"), b_.drop(columns=["table_sha256"], errors="ignore")
        pd.testing.assert_frame_equal(a_, b_)
    el = pd.read_csv(P1.elig)
    assert not any("alt" in c or "mean" in c or "rmse" in c.lower() for c in el.columns)
    r = el.iloc[0]
    assert r.n_cells == 64 and r.n_blocks == 16 and r.n_xf_target == 64 and bool(r.check_target_rows) and r.xf_x25_all == 1.0
    assert bool(r.eligible) == (r.n_valid_splits >= 1 and r.nb_union >= 8)
    cnt = pd.read_csv(P1.count)
    assert len(cnt) == len(XF._splits_of(r.run_splits)) and (cnt.fits_lg > 0).all() and (cnt.fits_wf4 > 0).all()
    cm = json.loads(P1.count_meta.read_text())
    assert cm["max_rss_mb"] > 0 and "기록하지 않음" in cm["natl_first_count"]["status"]
    # 동결(최종 XC-F3) 뒤에는 세기를 거부한다
    P1.xcf3_json.write_text(json.dumps(dict(final=True, created=AFTER.isoformat())))
    with pytest.raises(SystemExit):
        XF.main(["--count-only", "--out-root", str(tmp_path / "a"), "--threads", "2", "--min-mem-gb", "0", "--max-mem-gb", "0", "--no-fit-count"])


@REAL
def test_e2_covariate_missingness_count(tmp_path):
    """2.6 '시험: 공변량 결측률': 대상 새 행에 x25 결측과 CCI 결측을 넣으면 xf_x25_all·xf_eval_ready 가 그 형태대로 나온다."""
    P = XF.Paths(tmp_path)
    spec = synth_spec(P)
    d = XF.spec_dir(P, spec["alias"])
    t = pd.read_csv(d / "fidelity_base_v3.csv", low_memory=False)
    new = t.index[t.loc_id >= XF.LOC0_V5]
    t.loc[new[:16], "sg_clay_5_15"] = np.nan                                           # 64 가운데 16 행 x25 결측
    t.loc[new[16:24], "cci_alt"] = np.nan                                              # 8 행 CCI 결측(채점 불가, x25 결측이기도 하다)
    t.to_csv(d / "fidelity_base_v3.csv", index=False)
    row, _ = XF.count_spec(spec, d)
    assert row["n_xf_target"] == 64 and row["xf_x25_all"] == round(40 / 64, 4) and row["xf_eval_ready"] == 56


# ---------------------------------------------------------------- (f) 누설 시험(run_unit 이 쓰는 compute_unit)
def make_tctx(seed=0, split=1, target="T"):
    rng = np.random.RandomState(seed)
    D = len(XB.FEATS)
    ns, nA, nB = 300, 80, 60
    Xs = rng.randn(ns, D); ss = rng.uniform(20, 40, ns); ys = 1.6 * ss + 3 * Xs[:, 1] + 2 * rng.randn(ns)
    XA = rng.randn(nA, D); sA = rng.uniform(20, 40, nA); yA = 1.3 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    XBm = rng.randn(nB, D); sB = rng.uniform(20, 40, nB); yB = 1.3 * sB + 3 * XBm[:, 1] + 2 * rng.randn(nB)
    return H.Ctx(target, "x", split, target, Xs, ys, ss, np.repeat(["r1", "r2"], ns // 2), XA, yA, sA, np.repeat(np.arange(8), nA // 8),
                 XBm, yB, sB, 500 + np.repeat(np.arange(6), nB // 6), meta=dict(dup_of=-1, valid=True))


def _record(monkeypatch):
    rec = {}
    orig = H4.BlockStore.add

    def add(self, key, y, pred):
        rec[(self.target, self.split, tuple(key))] = np.asarray(pred, float).copy()
        return orig(self, key, y, pred)
    monkeypatch.setattr(H4.BlockStore, "add", add)
    return rec


def test_f_leakage_lg_axis_and_wf4(tmp_path, monkeypatch):
    rec = _record(monkeypatch)
    HA = H.parse_args(XF.h40_argv(tmp_path, "T", threads=1, cb_iters=10, n_grid="0,3,10", draws=2, seeds=1))
    a54 = XB.h54_args(exp="wf4", grid=[10, 20], threads=1, cb_iters=10, seeds=1, draws_cap=2, nboot=200)
    c = make_tctx()
    nA = len(c.yA)
    keep = np.unique(np.concatenate([H.draw_cells("T", "x", 1, n, d, nA) for n in (3, 10, 20) for d in (0, 1)]))
    assert 0 < len(keep) < nA
    r0 = XF.compute_unit(c, "T~xf", HA, a54)
    assert set(r0) == {"lg", "wf4", "elapsed"}
    p1 = dict(rec); rec.clear()
    assert any(k[0] == "T|x" for k in p1) and any(k[0] == "T~xf~wf4|x" for k in p1)   # LG 저장소 이름은 적합 뒤 별칭으로 바꾼다
    assert r0["lg"][1].target == "T~xf|x"
    assert any(k[2][0] == "W" for k in p1) and any(k[2][0] == "D0" for k in p1) and any(k[2][0] == "P3" for k in p1)
    XF.compute_unit(XB.perturb_labels(c, keep, seed=5), "T~xf", HA, a54)
    p2 = dict(rec); rec.clear()
    cmp_ = XB.compare_predictions(p1, p2)
    assert cmp_["ok"] and cmp_["same_keys"] and cmp_["n_keys"] > 30, cmp_
    XF.compute_unit(XB.perturb_labels(c, keep[1:], seed=5), "T~xf", HA, a54)        # 음성 대조: 선택 라벨 하나도 바꾼다
    p3 = dict(rec); rec.clear()
    assert not XB.compare_predictions(p1, p3)["ok"], "선택 라벨을 바꿔도 예측이 같다(시험이 둔하다)"
    r_lg = XF.compute_unit(c, "T~xf", HA, a54, parts=("lg",))                          # --resume 에서 남은 부분만
    assert set(r_lg) == {"lg", "elapsed"} and not any(k[0].endswith("~wf4|x") for k in rec)


# ---------------------------------------------------------------- (g) 대비·판정·문장
def synth_store_tm(name, levels, splits=(1, 2, 3, 4, 5), nb=10, draws=5, seeds=(0, 1), rng_seed=0, nboot=300, point_only=False, wf4=False):
    """levels: (method, learner, n, lam) → RMSE 수준. 물리식(learner none)은 seed −1, n 0·−1 은 추출 하나."""
    rng = np.random.RandomState(rng_seed)
    by = {}
    for sp in splits:
        ncell = rng.randint(3, 20, nb)
        st = H4.BlockStore(name, sp, np.repeat([f"{name}s{sp}b{j:02d}" for j in range(nb)], ncell))
        cnt = st.ncell
        st.add_sse(H.P0_KEY, cnt * (30.0 + rng.gamma(2.0, 1.0, st.nb)) ** 2, cnt)
        for (m, lr, n, lam), lev in levels.items():
            for d in range(1 if n in (0, -1) else draws):
                for s in ((-1,) if lr == "none" else seeds):
                    st.add_sse((m, lr, "1", "cell", int(n), d, int(s), float(lam)), cnt * (lev + rng.gamma(2.0, 1.0, st.nb)) ** 2, cnt)
        by[sp] = st
    units = [dict(split=sp, dup_of=-1, valid=True, expected_splits=list(splits), store_names=[name]) for sp in splits]
    return W.make_tm(name, by, units, nboot, point_only)


LG_LEVELS = {("D0", "catboost_lo", 0, 1.0): 34.0, ("P1", "none", 10, 0.0): 29.0, ("R1", "catboost_lo", 10, 0.25): 26.0,
             ("R1", "catboost_lo", -1, 0.25): 22.0}
W4_LEVELS = {("W", "catboost_lo", n, 0.0): 24.0 for n in (10, 40, 160)} | {("R1", "catboost_lo", n, 0.25): 24.2 for n in (10, 40, 160)}


def _elig(rows):
    return pd.DataFrame([dict(alias=a, eligible=e, eligible_xc=x, kind=k, nb_union=12 if e else 6,
                              xf_sources="swe_stordalen_crill;sjm_adventdalen_wendt2023") for a, e, x, k in rows])


def _sentence_matches(s, verdict):
    key = {"우세": "작았다", "열세": "컸다", "동등": "0.5 cm 안에서", "미결정": "확인하지 못했다"}.get(verdict, "판정할 수 없었다")
    return key in s


def _specs_g():
    return {"Aland~xf": dict(alias="Aland~xf", target="Aland", kind=XF.KIND_NEW, blind="맹검", run=True, keep_v4=[], keep_v5=[1]),
            "Bland~xf": dict(alias="Bland~xf", target="Bland", kind=XF.KIND_NEW, blind="맹검", run=True, keep_v4=[], keep_v5=[2]),
            XF.NATL_ALIAS: dict(alias=XF.NATL_ALIAS, target="NAtlantic", kind=XF.KIND_NATL, blind="비맹검 부분 포함", run=True, keep_v4=[], keep_v5=[3])}


def _tms_g(bland_levels=None):
    tms_lg = {"Aland~xf|x": synth_store_tm("Aland~xf|x", LG_LEVELS, rng_seed=1),
              "Bland~xf|x": synth_store_tm("Bland~xf|x", bland_levels or LG_LEVELS, rng_seed=2),
              f"{XF.NATL_ALIAS}|x": synth_store_tm(f"{XF.NATL_ALIAS}|x", LG_LEVELS, rng_seed=3, point_only=True)}
    tms_w4 = {f"{k.split('|')[0]}~wf4|x": synth_store_tm(f"{k.split('|')[0]}~wf4|x", W4_LEVELS, splits=(1, 2, 3), rng_seed=4) for k in tms_lg}
    return tms_lg, tms_w4


def test_g_contrasts_holm_sentences():
    specs = _specs_g()
    tms_lg, tms_w4 = _tms_g()
    units_w4 = [dict(target="Aland~xf", diag=[dict(abs_bias=3.0, bias=-3.0), dict(abs_bias=5.0, bias=5.0)])]
    # 적격 2곳
    el = _elig([("Aland~xf", True, True, XF.KIND_NEW), ("Bland~xf", True, False, XF.KIND_NEW), (XF.NATL_ALIAS, False, False, XF.KIND_NATL)])
    res = XF.summarize_core(tms_lg, tms_w4, specs, el, units_w4, nboot=300)
    t = res["tests"]
    assert res["meta"]["k_eligible"] == 2 and res["meta"]["holm_m"] == 6
    h = res["holm"]
    assert len(h) == 6 and (h.m == 6).all() and set(h.label) == {f"XF-{i}|{a}~xf|x" for i in (1, 2, 3) for a in ("Aland", "Bland")}
    reg = t[(t.group == "new_region") & (t.scope == "region") & t.test_id.isin(["XF-1", "XF-2", "XF-3"])]
    assert len(reg) == 6 and reg.p_holm.notna().all() and (reg.p_holm >= reg.p_two - 1e-12).all()
    assert set(t[(t.group == "new_region") & t.test_id.isin(["XF-1", "XF-2", "XF-3"])].scope) == {"region", "MEAN"}
    xf1 = reg[reg.test_id == "XF-1"].iloc[0]
    assert xf1.verdict4 == "열세"                                                       # D0(34) − P0(약 32): 직접 ML 이 나쁘다
    for s in res["sentences"]["XF"]:
        assert "독립 지역 확인" in s["sentence"] and _sentence_matches(s["sentence"], s["verdict"]), s
    assert any(s["scope"] == "MEAN" and "추가 독립 지역 2곳" in s["sentence"] for s in res["sentences"]["XF"])
    natl = t[(t.group == "natl_v5") & t.test_id.isin(["XF-1", "XF-2", "XF-3"])]
    assert len(natl) == 3 and natl.ci_lo.isna().all() and natl.point_only.all()        # 점 추정만
    assert all("점 추정" in s["sentence"] and "공개 자료 2건" in s["sentence"] and "앞선 분석에서 열람" in s["sentence"] for s in res["sentences"]["NAtlantic"])
    x4 = t[(t.test_id == "XF-4") & (t.item == "W-R1(0.25)")]
    assert len(x4) == 9 and set(x4.n) == {10, 40, 160}
    dg = t[t.item == "진단값 |b10|"].iloc[0]
    assert dg.diag_abs_mean == 4.0 and dg.n_diag == 2
    # 적격 1곳: 지역 행 문장, Holm m = 3
    el1 = _elig([("Aland~xf", True, True, XF.KIND_NEW), ("Bland~xf", False, False, XF.KIND_NEW), (XF.NATL_ALIAS, False, False, XF.KIND_NATL)])
    r1 = XF.summarize_core(tms_lg, tms_w4, specs, el1, [], nboot=300)
    assert r1["meta"]["holm_m"] == 3 and len(r1["holm"]) == 3 and r1["meta"]["new_point_only"] == ["Bland~xf"]
    assert len([s for s in r1["sentences"]["XF"] if s["scope"] == "region"]) == 3
    assert (r1["tests"].group == "new_region_point_only").any()
    # 적격 0곳: 사전 고정 문장, Holm 표 없음
    el0 = _elig([("Aland~xf", False, False, XF.KIND_NEW), ("Bland~xf", False, False, XF.KIND_NEW), (XF.NATL_ALIAS, False, False, XF.KIND_NATL)])
    r0 = XF.summarize_core(tms_lg, tms_w4, specs, el0, [], nboot=300)
    assert r0["meta"]["k_eligible"] == 0 and len(r0["holm"]) == 0
    assert r0["sentences"]["XF"][0]["sentence"] == XF.ZERO_SENTENCE and "시험하지 못함" in r0["sentences"]["XF"][0]["status"]


def test_g1_missing_region_row():
    """1절 '행 없음·실패': 적격 지역의 대비 행이 없으면 판정 불가 행과 문장(사유)을 남기고 Holm 가족에 p 1 로 넣는다."""
    specs = _specs_g()
    lv = {k: v for k, v in LG_LEVELS.items() if k != ("R1", "catboost_lo", 10, 0.25)}  # Bland 의 XF-2 행이 없다
    tms_lg, tms_w4 = _tms_g(bland_levels=lv)
    el = _elig([("Aland~xf", True, True, XF.KIND_NEW), ("Bland~xf", True, True, XF.KIND_NEW), (XF.NATL_ALIAS, False, False, XF.KIND_NATL)])
    res = XF.summarize_core(tms_lg, tms_w4, specs, el, [], nboot=300)
    t = res["tests"]
    miss = t[(t.test_id == "XF-2") & (t.target == "Bland~xf|x") & (t.scope == "region")]
    assert len(miss) == 1 and miss.iloc[0].verdict4 == "행 없음" and miss.iloc[0].p_two == 1.0 and "Holm 가족에 p 1" in miss.iloc[0].reason
    h = res["holm"].set_index("label")
    assert h.loc["XF-2|Bland~xf|x", "p"] == 1.0 and len(h) == 6
    s = [x for x in res["sentences"]["XF"] if x["hyp"] == "XF-2" and x.get("target") == "Bland~xf|x"]
    assert len(s) == 1 and s[0]["verdict"] == "판정 불가" and "판정할 수 없었다" in s[0]["sentence"] and "Holm 가족에 p 1" in s[0]["sentence"]
    # 판정 불가 지역 행의 사유: 분할 수, 채점 블록 합집합, CI 유무
    rr = XF.region_reason(dict(target="Bland~xf|x", verdict4="판정 불가", n_splits=3, n_splits_expected=5, ci_lo=np.nan, ci_hi=np.nan,
                               ci_lo_beq=np.nan, ci_hi_beq=np.nan), el.assign(nb_union=[6, 6, 6]).set_index("alias"))
    assert "분할 3/5" in rr and "채점 블록 합집합 6 < 8" in rr and "CI 없음" in rr


def test_g2_compose_rules():
    T = XF._templates("XF-2")
    s = XF.compose(T, "우세", 1, delta=-0.3, holm_p=0.2)
    assert "작았다" in s and "독립 지역 확인" in s and XB.UNCORRECTED_TXT in s and XB.SMALL_EFFECT_TXT in s and "0.30" in s
    s2 = XF.compose(T, "열세", 3, delta=2.0, holm_p=0.01, worse=[("X|x", 2.0, 0.5, 3.5)])
    assert "추가 독립 지역 3곳" in s2 and "컸다" in s2 and XB.UNCORRECTED_TXT not in s2 and "지역 X|x 에서는 오차가 컸다" in s2
    s3 = XF.compose(T, "판정 불가", 1, reason="pool<2")
    assert "판정할 수 없었다" in s3 and "pool<2" in s3
    assert "0.5 cm 안에서 같았다" in XF.compose(T, "동등", 1, delta=0.1) and "확인하지 못했다" in XF.compose(T, "미결정", 1, delta=0.1)


def test_g3_natl_deadline_status():
    """NAtlantic v5: 적격이어도 첫 세기 마감 규칙(natl_info)이 점 추정이면 점 추정 문장이고, 규칙을 통과하면 4분 판정 문장이다."""
    specs = _specs_g()
    tms_lg, tms_w4 = _tms_g()
    tms_lg[f"{XF.NATL_ALIAS}|x"] = synth_store_tm(f"{XF.NATL_ALIAS}|x", LG_LEVELS, rng_seed=3, point_only=False)
    el = _elig([("Aland~xf", True, True, XF.KIND_NEW), ("Bland~xf", True, True, XF.KIND_NEW), (XF.NATL_ALIAS, True, False, XF.KIND_NATL)])
    late = XF.summarize_core(tms_lg, tms_w4, specs, el, [], nboot=300, natl_info={XF.NATL_ALIAS: (True, "첫 세기가 마감 뒤")})
    s = late["sentences"]["NAtlantic"]
    assert all(x["point_only"] and "점 추정" in x["sentence"] and x["reason"] == "첫 세기가 마감 뒤" for x in s)
    assert late["meta"]["natl_status"][XF.NATL_ALIAS]["point_only"]
    ok = XF.summarize_core(tms_lg, tms_w4, specs, el, [], nboot=300, natl_info={XF.NATL_ALIAS: (False, "")})
    assert all(not x["point_only"] and "4분 판정" in x["sentence"] for x in ok["sentences"]["NAtlantic"])


def test_g4_augment_l40_and_l28():
    """셀 보강(L40 형식): 확충판과 같은 작업의 기준 표에서 L1 조건(D0 − P0), 전량 R1 − P0, n 10 P1 − P0 를 비교하고 L28 로 분류한다.
    XF-1–XF-3 행을 만들지 않고, 기준 표는 XF-4 에 넣지 않는다."""
    assert XF.l28_shift("우세", "우세") == "강건" and XF.l28_shift("동등", "미결정") == "강건" and XF.l28_shift("우세", "미결정") == "약화"
    assert XF.l28_shift("우세", "열세") == "의존" and XF.l28_shift("행 없음", "우세") == "판정 불가"
    assert XF.worst_shift(["강건", "약화"]) == "약화" and XF.worst_shift(["강건", "판정 불가"]) == "강건(판정한 대비 1/2)"
    specs = {"Tibet_LGD~aug~v5": dict(alias="Tibet_LGD~aug~v5", target="Tibet_LGD", kind=XF.KIND_AUG, blind="비맹검 부분 포함", run=True, keep_v4=[7],
                                      keep_v5=[30001, 30002], ref="Tibet_LGD~aug~ref"),
             "Tibet_LGD~aug~ref": dict(alias="Tibet_LGD~aug~ref", target="Tibet_LGD", kind=XF.KIND_AUG_REF, blind="비맹검 부분 포함", run=True,
                                       keep_v4=[7], keep_v5=[], aug="Tibet_LGD~aug~v5", parts=["lg"])}
    ref_lv = dict(LG_LEVELS) | {("P1", "none", 10, 0.0): 25.0}
    aug_lv = dict(LG_LEVELS) | {("P1", "none", 10, 0.0): 37.0}                        # P1 − P0: 기준 우세(25), 확충판 열세(37) → 의존
    tms_lg = {"Tibet_LGD~aug~v5|x": synth_store_tm("Tibet_LGD~aug~v5|x", aug_lv, rng_seed=5),
              "Tibet_LGD~aug~ref|x": synth_store_tm("Tibet_LGD~aug~ref|x", ref_lv, rng_seed=6)}
    tms_w4 = {"Tibet_LGD~aug~v5~wf4|x": synth_store_tm("Tibet_LGD~aug~v5~wf4|x", W4_LEVELS, splits=(1, 2, 3), rng_seed=7)}
    el = _elig([("Tibet_LGD~aug~v5", True, False, XF.KIND_AUG), ("Tibet_LGD~aug~ref", True, False, XF.KIND_AUG_REF)])
    res = XF.summarize_core(tms_lg, tms_w4, specs, el, [], nboot=300, nlab={"Tibet_LGD~aug~v5": 120, "Tibet_LGD~aug~ref": 118})
    t = res["tests"]
    l40 = t[t.test_id == "L40(XF 셀 보강)"]
    vr = l40[l40.scope == "verdict"].iloc[0]
    assert vr.l28 == "의존" and "L1 지역 조건 같음" in vr.stat and "P1-P0|n10: 의존" in vr.stat and "R1-P0|nall: 강건" in vr.stat
    items = set(l40[l40.scope == "region"].item)
    assert items == {"D0-P0(L1 조건)", "R1-P0", "P1-P0"} and set(l40[l40.scope == "region"].side) == {"확충판", "기준"}
    assert not t.test_id.isin(["XF-1", "XF-2", "XF-3"]).any() if "test_id" in t else True
    assert set(t[t.test_id == "XF-4"].target) == {"Tibet_LGD~aug~v5~wf4|x"}
    assert res["sentences"]["augment"][0]["l28"] == "의존" and res["meta"]["augment_ref"] == ["Tibet_LGD~aug~ref"]


# ---------------------------------------------------------------- (h) 부록 XC-F3
def test_h_xc_f3_list(tmp_path, capsys):
    P = XF.Paths(tmp_path)
    cols = dict(table_sha256="ab", xc_n_valid=8, xc_nb_union=12, xc_nb_min=5, xc_few_blocks=False, n_cells=40, n_blocks=16, lic_ok=True, target="T")
    el = pd.DataFrame([dict(alias="Aland~xf", kind=XF.KIND_NEW, eligible=True, eligible_xc=True, xc_keep="201,202,203", **cols),
                       dict(alias="Bland~xf", kind=XF.KIND_NEW, eligible=True, eligible_xc=False, xc_keep="201", **cols),
                       dict(alias=XF.NATL_ALIAS, kind=XF.KIND_NATL, eligible=True, eligible_xc=True, xc_keep="201", **cols),
                       dict(alias="Tibet_LGD~aug~v5", kind=XF.KIND_AUG, eligible=True, eligible_xc=True, xc_keep="201", **cols),
                       dict(alias="Tibet_LGD~aug~ref", kind=XF.KIND_AUG_REF, eligible=True, eligible_xc=True, xc_keep="201", **cols),
                       dict(alias=XF.AL7_ALIAS, kind=XF.KIND_AK, eligible=False, eligible_xc=False, xc_keep="", **cols)])
    P.root.mkdir(parents=True, exist_ok=True)
    el.to_csv(P.elig, index=False)
    doc = XF.write_xc_f3(P, el, {}, now=BEFORE)
    assert doc["k"] == 1 and doc["f3"][0]["alias"] == "Aland~xf" and doc["f3"][0]["xc_keep"] == "201,202,203" and not doc["final"]
    assert not os.path.isabs(doc["f3"][0]["run_table"]) and doc["f3"][0]["run_table"] == os.path.relpath(XF.spec_dir(P, "Aland~xf"), XF.ROOT)
    assert doc["script_sha256"] == XB.sha256_file(XF.__file__) and doc["elig_sha256"] == XB.sha256_file(P.elig) and "v5_sha256" in doc and "specs_sha256" in doc
    assert any(XB.sha256_file(XF.__file__) in t for t in doc["appendix_text"])
    assert [x["alias"] for x in doc["f2_only"]] == [XF.AL7_ALIAS]
    ex = {e["alias"]: e["reason"] for e in doc["excluded"]}
    assert "민감도" in ex[XF.NATL_ALIAS] and "셀 보강" in ex["Tibet_LGD~aug~v5"] and "201–210" in ex["Bland~xf"] and "기준 표" in ex["Tibet_LGD~aug~ref"]
    assert pd.read_csv(P.xcf3_csv).alias.tolist() == ["Aland~xf"]
    el0 = el[el.alias != "Aland~xf"]
    el0.to_csv(P.elig, index=False)
    doc0 = XF.write_xc_f3(P, el0, {}, now=AFTER)
    assert doc0["k"] == 0 and doc0["final"] and "시험하지 못함" in doc0["status"]
    assert any("외부 계열 시험은 하지 못했다" in t for t in doc0["appendix_text"])
    again = XF.write_xc_f3(P, el0, {}, now=AFTER + XF.dt.timedelta(days=1))            # 최종판은 다시 쓰지 않는다(같은 해시면 그대로)
    assert again["created"] == doc0["created"]
    el.to_csv(P.elig, index=False)                                                      # 동결 뒤 적격 표가 바뀌면 멈춘다
    with pytest.raises(SystemExit):
        XF.write_xc_f3(P, el, {}, now=AFTER)
    no_forbidden(capsys.readouterr().out)


# ---------------------------------------------------------------- (i) 끝에서 끝: 단위 실행, 조각, 봉인 집계
@REAL
def test_i_unit_shards_and_sealed_summary(tmp_path, capsys, monkeypatch):
    a = XF.parse_args(["--out-root", str(tmp_path), "--threads", "2", "--cb-iters", "10", "--resume"])
    a.LG_NGRID, a.LG_DRAWS, a.LG_SEEDS = "0,10,all", 1, 1
    a.W4_GRID, a.W4_SEEDS, a.W4_DRAWS_CAP = (10, -1), 1, 1
    spec = synth_spec(a.P)
    spec["run_splits"] = [1]
    a.SPECS = {spec["alias"]: spec}
    XF.save_specs(a.P, a.SPECS)
    calls = []
    orig = XF.compute_unit

    def spy(*args, **kw):
        calls.append(kw.get("parts"))
        return orig(*args, **kw)
    monkeypatch.setattr(XF, "compute_unit", spy)
    with XB.restricted_output():
        out = XF.run_unit(a, spec["alias"], 1)
    assert out["fits_lg"] > 0 and out["fits_wf4"] > 0 and calls == [("lg", "wf4")]   # 본 실행이 누설 시험과 같은 compute_unit 을 지난다
    for part in ("lg", "wf4"):
        b = XB.shard_base(a.P.shards, "xf", spec["alias"], "x", 1, part)
        assert b.name == f"xf__cpu__{spec['alias']}__x__s1__{part}"
        for suf in ("_runs.csv", "_blocksse.npz", "_unit.json"):
            assert Path(str(b) + suf).exists()
        u = json.loads(Path(str(b) + "_unit.json").read_text())
        assert u["variant"] == part and u["expected_splits"] == [1] and u["spec_kind"] == XF.KIND_NEW and u["table_sha256"] == spec["table"]["sha256"]
    with XB.restricted_output():
        again = XF.run_unit(a, spec["alias"], 1)                                        # --resume: 완료 조각은 건너뛴다
    assert again["status"] == "resumed"
    tms, units, _ = XB.load_tms(a.P.shards, "xf", 200, point_only=lambda nm: False)    # 공통 설정 해시가 하나여야 읽힌다
    assert set(tms) == {f"{spec['alias']}|x", f"{spec['alias']}~wf4|x"}
    pd.DataFrame([dict(alias=spec["alias"], eligible=True, eligible_xc=True, kind=XF.KIND_NEW, run_splits="1", nb_union=16)]).to_csv(a.P.elig, index=False)
    with pytest.raises(SystemExit):
        XF.summarize(a, nboot=200)                                                      # 재현 관문 기록이 없으면 거부
    with pytest.raises(SystemExit):
        XF.summarize(a, nboot=200, require_gate=False)                                  # 적격 동결 기록(최종 XC-F3)이 없으면 거부
    capsys.readouterr()
    with XB.restricted_output():
        res = XF.summarize(a, nboot=200, require_gate=False, require_freeze=False)
    out_txt = capsys.readouterr().out
    no_forbidden(out_txt)
    assert (a.P.sealed / "xf_tests.csv").exists() and (a.P.sealed / "xf_sentences.json").exists() and (a.P.sealed / "sealed_manifest.json").exists()
    with pytest.raises(PermissionError):
        XB.assert_not_sealed(a.P.sealed / "xf_tests.csv")
    t = res["tests"]
    assert set(t.test_id) >= {"XF-1", "XF-2", "XF-3", "XF-4"} and res["meta"]["k_eligible"] == 1 and res["meta"]["holm_m"] == 3
    sp2 = dict(spec, table=dict(spec["table"], sha256="0" * 64))                        # 조각의 표 해시가 spec 과 다르면 거부
    XF.save_specs(a.P, {spec["alias"]: sp2})
    with pytest.raises(SystemExit):
        XF.summarize(a, nboot=200, require_gate=False, require_freeze=False)


# ---------------------------------------------------------------- (j) build_ext_cells_v1·ext_cells_covariates_v1 재사용
@REAL
def test_j_b1_reuse_and_cov_redirect(tmp_path, capsys):
    p = _points(tmp_path / "xf_test_points.csv")
    P = XF.Paths(tmp_path)
    cells, meta = XF.run_b1([p], P.build)
    assert capsys.readouterr().out == ""                                                # B1 의 화면 출력(라벨 평균 포함)은 버린다
    assert not (P.build / "ext_cells_v1_meta.json").exists() and (P.build / "cells_xf_meta.json").exists()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                assert not XF.LABEL_KEY.search(str(k)), k
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(json.loads((P.build / "cells_xf_meta.json").read_text()))
    assert B_EXT_RESTORED()
    d = cells[cells.label_set == "direct"]
    assert len(d) == 6 and (d.macro == "Testland").all() and (d.indep_ok == 1).all() and (d.dup_v3 == 0).all()
    v4c = pd.DataFrame(columns=["cell_uid", "label_set", "lat", "lon", "in_v4"])
    c = XF.classify_cells(cells, v4c, known={"Tibet", "NAtlantic"})
    assert (c[c.label_set == "direct"].xf_target == 1).all() and (c.xf_role == XF.ROLE_NEW).all()

    class FakeCov:                                                                     # 단계 함수 대신: 전역 변수 교체와 호출 순서만 본다
        PARTS, DEM, SG_RAW = tmp_path / "orig_parts", tmp_path / "orig_dem", tmp_path / "orig_sg"
        calls = []

        @staticmethod
        def tname(a, b):
            return f"T{a}_{b}"

        @staticmethod
        def fetch_tile(a, b):
            raise AssertionError("내려받기 금지")

        @classmethod
        def _st(cls, name, cell, tag):
            cls.calls.append((name, str(cls.PARTS), str(cls.DEM), str(cls.SG_RAW), len(cell), tag))
            pd.DataFrame(dict(id=cell.id, **{f"{name}_v": 1.0})).to_csv(cls.PARTS / f"{tag}_{name}.csv", index=False)

        @classmethod
        def stage_dem(cls, cell, tag): cls._st("dem", cell, tag)

        @classmethod
        def stage_e5(cls, cell, tag): cls._st("e5", cell, tag)

        @classmethod
        def stage_cci(cls, cell, tag): cls._st("cci", cell, tag)

        @classmethod
        def stage_sg(cls, cell, tag, allow_download=True):
            assert not allow_download and all(m.startswith("xf5_") for m in cell.macro)
            cls._st("sg", cell, tag)

        @classmethod
        def stage_elev(cls, cell, tag):
            assert cls.fetch_tile(0, 0) == "not_fetched"
            cls._st("elev", cell, tag)

        @classmethod
        def merge(cls, cell, tag):
            o = cell[["id", "lat", "lon", "macro"]].copy()
            for st in ("dem", "e5", "cci", "sg", "elev"):
                o = o.merge(pd.read_csv(cls.PARTS / f"{tag}_{st}.csv"), on="id")
            return o
    for d_ in (FakeCov.DEM, FakeCov.SG_RAW):
        d_.mkdir()
    (FakeCov.DEM / "tileA.tif").write_text("x")
    (FakeCov.SG_RAW / "windows_wcs_meta_v4.json").write_text("{}")
    ov = (tmp_path / "ov_dem", tmp_path / "ov_sg")
    cov = XF.run_cov(P, c, fetch=False, cov_mod=FakeCov, overlays=ov)
    assert len(cov) == int(c.in_v5.sum()) and [x[0] for x in FakeCov.calls] == ["dem", "e5", "cci", "sg", "elev"]
    assert all(x[1] == str(P.cov) and x[2] == str(ov[0]) and x[3] == str(ov[1]) for x in FakeCov.calls)
    assert FakeCov.PARTS == tmp_path / "orig_parts" and FakeCov.DEM == tmp_path / "orig_dem"          # 전역 변수 복원
    assert os.path.islink(ov[0] / "tileA.tif") and not os.path.islink(ov[1] / "windows_wcs_meta_v4.json")   # 기존 파일은 연결, v4 창 메타는 복사
    assert set(cov.cell_uid) == set(c.cell_uid[c.in_v5 == 1]) and (cov.macro == "Testland").all()


def B_EXT_RESTORED():
    return XF.B1.EXT == XF.EXT and XF.B1.PROC == XF.PROC


# ---------------------------------------------------------------- (k) 실행 보호와 로컬 자원
def test_k_run_guard(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        XF.main(["--out-root", str(tmp_path), "--min-mem-gb", "0", "--max-mem-gb", "0"])
    a = XF.parse_args(["--threads", "16", "--shard", "1/3", "--units", "A~xf:1,B~xf:2"])
    assert a.threads <= XB.LOCAL_MAX_THREADS and a.SHARD == (1, 3) and a.UNITS == [("A~xf", 1), ("B~xf", 2)]
    specs = {al: dict(run=True, run_splits=[1, 2, 3]) for al in ("A~xf", "B~xf")}
    specs["C~aug~v5"] = dict(run=False, run_splits=[1])
    a2 = XF.parse_args(["--shard", "1/2"])
    u = XF.plan_units(a2, specs)
    assert u == [("A~xf", 2), ("B~xf", 1), ("B~xf", 3)]
    with pytest.raises(SystemExit):
        XF.parse_args(["--shard", "3/2"])
    assert np.isfinite(XF.rss_mb()) and XF.rss_mb() > 0 and not XF.rss_watchdog(0)          # RSS 상한 감시(0 = 끔)
    assert a.max_mem_gb == 10.0 and a.as_cap_gb == 0.0


def test_k2_local_resources_allow_local(tmp_path, monkeypatch):
    """1절 로컬 자원: --allow-local 은 로컬 실행이므로 30 GB 대기·RSS 감시·워커당 4스레드·합계 32스레드를 지킨다. WF_RESCALE=1 에서만 푼다."""
    with pytest.raises(SystemExit):
        XF.parse_args(["--allow-local", "--workers", "22", "--threads", "4"])           # 88 > 32
    a = XF.parse_args(["--allow-local", "--workers", "8", "--threads", "16"])
    assert a.threads == XB.LOCAL_MAX_THREADS                                           # --allow-local 이어도 4
    rec = []
    monkeypatch.setattr(XB, "require_memory", lambda *x, **k: rec.append("mem") or 50.0)
    monkeypatch.setattr(XF, "rss_watchdog", lambda gb, **k: rec.append(("rss", gb)) or True)
    r = XF._resources(a)
    assert rec == ["mem", ("rss", 10.0)] and r["mem_wait"] and r["watchdog"] and not r["on_rescale"]
    monkeypatch.setenv("WF_RESCALE", "1")
    rec.clear()
    a3 = XF.parse_args(["--workers", "22", "--threads", "4"])                          # Rescale 노드: 상한 없음
    assert a3.threads == 4 and XF._resources(a3)["on_rescale"] and rec == []


def test_k3_run_all_needs_deadline_and_final_xcf3(tmp_path):
    a = XF.parse_args(["--out-root", str(tmp_path), "--allow-local", "--min-mem-gb", "0", "--max-mem-gb", "0"])
    with pytest.raises(SystemExit, match="마감"):
        XF.run_all(a, now=BEFORE)
    with pytest.raises(SystemExit, match="XC-F3"):
        XF.run_all(a, now=AFTER)


def test_k4_smoke_env_refused(tmp_path, monkeypatch):
    a = XF.parse_args(["--out-root", str(tmp_path), "--smoke", "--min-mem-gb", "0", "--max-mem-gb", "0"])
    monkeypatch.setattr(XB, "smoke_env_report", lambda: dict(threads="8", cores=96, warn=["코어 묶음 96개 > 4"], mem_available_gb=50.0, permitted=False))
    with pytest.raises(SystemExit, match="스모크 규약"):
        XF.smoke(a)


# ---------------------------------------------------------------- (m) 마감: 점 자료 목록 동결, NAtlantic v5 첫 세기
def test_m1_points_manifest_freeze(tmp_path):
    P = XF.Paths(tmp_path)
    P.points.mkdir(parents=True)
    fa = _points(P.points / "a_points.csv", "a")
    fb = _points(P.points / "b_points.csv", "b")
    d0 = XF.points_manifest(P, [fa, fb], now=BEFORE)
    assert not d0["frozen"] and set(d0["files"]) == {"a_points.csv", "b_points.csv"}
    fb.write_text(fb.read_text() + "\n")                                               # 마감 전 수정은 기록만 갱신
    d1 = XF.points_manifest(P, [fa, fb], now=BEFORE + XF.dt.timedelta(hours=1))
    assert d1["files"]["b_points.csv"]["sha256"] == XB.sha256_file(fb) and not d1["frozen"]
    d2 = XF.points_manifest(P, [fa, fb], now=AFTER)                                    # 마감 뒤 첫 실행에서 동결
    assert d2["frozen"] and d2["freeze_basis"] == "마감 전 마지막 기록"
    fc = _points(P.points / "c_points.csv", "c")
    with pytest.raises(SystemExit, match="동결 목록 밖"):
        XF.points_manifest(P, [fa, fb, fc], now=AFTER + XF.dt.timedelta(hours=1))
    fa.write_text(fa.read_text() + "\n")
    with pytest.raises(SystemExit):
        XF.points_manifest(P, [fa, fb], now=AFTER + XF.dt.timedelta(hours=2))
    # 마감 전 기록이 없으면 수정 시각으로 동결한다
    P2 = XF.Paths(tmp_path / "q")
    P2.points.mkdir(parents=True)
    g1, g2 = _points(P2.points / "g1_points.csv", "g1"), _points(P2.points / "g2_points.csv", "g2")
    os.utime(g1, (BEFORE.timestamp(), BEFORE.timestamp()))
    os.utime(g2, (AFTER.timestamp(), AFTER.timestamp()))
    with pytest.raises(SystemExit, match="g2_points.csv"):
        XF.points_manifest(P2, [g1, g2], now=AFTER + XF.dt.timedelta(hours=3))
    # 새 점 자료는 산출 폴더의 ext_labels 에서만 읽고, v4 조립에 쓴 이름은 받지 않는다
    assert [p.name for p in XF.xf_point_files(P, v1={})] == ["a_points.csv", "b_points.csv", "c_points.csv"]
    with pytest.raises(SystemExit):
        XF.xf_point_files(P, v1={"a_points.csv": "x"})


def _natl_setup(tmp_path, eligible=True, with_priority=True):
    P = XF.Paths(tmp_path)
    P.root.mkdir(parents=True, exist_ok=True)
    files = {"swe_stordalen_crill_points.csv": dict(sha256="s1")} if with_priority else {"zz_points.csv": dict(sha256="s2")}
    P.points_manifest.write_text(json.dumps(dict(files=files, frozen=False)))
    specs = {XF.NATL_ALIAS: dict(alias=XF.NATL_ALIAS, target="NAtlantic", kind=XF.KIND_NATL, run=True, keep_v4=[], keep_v5=[30000])}
    el = pd.DataFrame([dict(alias=XF.NATL_ALIAS, table_sha256="t", n_cells=24, n_xf_target=2, n_blocks=9, n_valid_splits=5,
                            nb_union=8 if eligible else 6, eligible=eligible, xf_sources="swe_stordalen_crill")])
    return P, specs, el


def test_m2_natl_first_count_record(tmp_path):
    P, specs, el = _natl_setup(tmp_path / "a")
    st = XF.natl_first_record(P, specs, el, now=XF.DEADLINE_COUNT - XF.dt.timedelta(hours=2))
    assert st["status"] == "기록함" and st["before_deadline"]
    first = P.natl_first.read_text()
    assert XF.natl_first_record(P, specs, el, now=XF.DEADLINE_COUNT + XF.dt.timedelta(hours=2))["status"].startswith("기록 있음")
    assert P.natl_first.read_text() == first                                           # 한 번만 쓴다
    assert XF.natl_status(P, specs, el) == {XF.NATL_ALIAS: (False, "")}
    assert XF.natl_status(P, specs, el.assign(eligible=False))[XF.NATL_ALIAS][0]       # 지금 부적격
    # 마감 뒤 첫 세기 → 점 추정
    P2, specs2, el2 = _natl_setup(tmp_path / "b")
    XF.natl_first_record(P2, specs2, el2, now=XF.DEADLINE_COUNT + XF.dt.timedelta(minutes=1))
    po, why = XF.natl_status(P2, specs2, el2)[XF.NATL_ALIAS]
    assert po and "마감" in why
    # 첫 세기에서 부적격이면 나중에 적격이 되어도 점 추정(2.6 '점 추정으로 확정')
    P3, specs3, el3 = _natl_setup(tmp_path / "c", eligible=False)
    XF.natl_first_record(P3, specs3, el3, now=XF.DEADLINE_COUNT - XF.dt.timedelta(hours=1))
    po3, why3 = XF.natl_status(P3, specs3, el3.assign(eligible=True, nb_union=9))[XF.NATL_ALIAS]
    assert po3 and "확정" in why3
    # 기록 없음 → 점 추정, 우선 자료원 점 자료가 아직 없으면 기록하지 않는다
    P4, specs4, el4 = _natl_setup(tmp_path / "d", with_priority=False)
    assert XF.natl_first_record(P4, specs4, el4, now=XF.DEADLINE_COUNT - XF.dt.timedelta(hours=1))["status"].startswith("기록하지 않음")
    assert XF.natl_status(P4, specs4, el4)[XF.NATL_ALIAS][0] and not P4.natl_first.exists()
    assert XF.point_only_map(P2, specs2, el2) == {XF.NATL_ALIAS: True}
    # 전체 판(약관 회신 뒤 생김): 첫 세기 기록에 없으면 약관 확인분 판의 첫 세기로 마감 규칙을 판정하고, 지금 적격은 전체 판 자신의 적격 표로 본다
    full = dict(alias=XF.NATL_FULL_ALIAS, target="NAtlantic", kind=XF.KIND_NATL_FULL, run=True, keep_v4=[1], keep_v5=[30000])
    el_full = pd.concat([el, pd.DataFrame([dict(alias=XF.NATL_FULL_ALIAS, eligible=True)])], ignore_index=True)
    st1 = XF.natl_status(P, dict(specs, **{XF.NATL_FULL_ALIAS: full}), el_full)
    assert st1[XF.NATL_FULL_ALIAS] == (False, "") and st1[XF.NATL_ALIAS] == (False, "")
    assert XF.natl_status(P, dict(specs, **{XF.NATL_FULL_ALIAS: full}), el_full.assign(eligible=[True, False]))[XF.NATL_FULL_ALIAS][0]
    po_f, why_f = XF.natl_status(P3, dict(specs3, **{XF.NATL_FULL_ALIAS: full}), pd.concat([el3.assign(eligible=True, nb_union=9),
                                                                                           pd.DataFrame([dict(alias=XF.NATL_FULL_ALIAS, eligible=True)])]))[XF.NATL_FULL_ALIAS]
    assert po_f and XF.NATL_ALIAS in why_f and "확정" in why_f                          # 약관 확인분 판의 첫 세기가 부적격이면 전체 판도 점 추정
    assert XF.natl_status(P2, dict(specs2, **{XF.NATL_FULL_ALIAS: full}), el2)[XF.NATL_FULL_ALIAS][0]   # 첫 세기가 마감 뒤면 전체 판도 점 추정


# ---------------------------------------------------------------- (n) 적격 동결
def _frozen_setup(tmp_path):
    P = XF.Paths(tmp_path)
    P.root.mkdir(parents=True, exist_ok=True)
    P.v5.mkdir(parents=True, exist_ok=True)
    P.v5_table.write_text("loc_id,lat,lon,region\n1,60,30,Testland\n")
    XF.save_specs(P, {"Aland~xf": dict(alias="Aland~xf", target="Aland", kind=XF.KIND_NEW, run=True, keep_v4=[], keep_v5=[1],
                                       table=dict(dir="run_tables/Aland~xf", sha256="x"))}, P.v5_table)
    el = pd.DataFrame([dict(alias="Aland~xf", kind=XF.KIND_NEW, eligible=True, eligible_xc=False, xc_keep="", table_sha256="x", xc_n_valid=0,
                            xc_nb_union=0, xc_nb_min=0, xc_few_blocks=False, n_cells=10, n_blocks=9, lic_ok=True, target="Aland")])
    el.to_csv(P.elig, index=False)
    return P, el


def test_n_eligibility_freeze(tmp_path):
    P, el = _frozen_setup(tmp_path)
    assert XF.check_frozen(P, require=False) is None
    with pytest.raises(SystemExit):
        XF.check_frozen(P, require=True)
    with pytest.raises(SystemExit):
        XF.register_xf_aliases(P)                                                      # R4: 동결 기록이 있어야 한다
    add = XF.register_xf_aliases(P, require_frozen=False)
    assert add["Aland~xf"][0] == str(XF.spec_dir(P, "Aland~xf").resolve()) and add["Aland~xf"][1] == "Aland" and add["Aland~xf"][2] is False
    assert XB.resolve("Aland~xf")[0] == add["Aland~xf"][0]
    doc = XF.write_xc_f3(P, now=AFTER)
    assert doc["final"] and XF.check_frozen(P)["created"] == doc["created"]
    assert XF.register_xf_aliases(P)["Aland~xf"][1] == "Aland"
    a = XF.parse_args(["--out-root", str(P.base), "--min-mem-gb", "0", "--max-mem-gb", "0"])
    with pytest.raises(SystemExit):
        XF.count_only(a)                                                               # 동결 뒤 세기 거부
    with pytest.raises(SystemExit):
        XF.build(a, ["tables"])                                                        # 동결 뒤 실행 표 다시 만들기 거부
    el.assign(eligible=False).to_csv(P.elig, index=False)                              # 동결 뒤 적격 표가 바뀌면 멈춘다
    with pytest.raises(SystemExit):
        XF.check_frozen(P)
    with pytest.raises(SystemExit):
        XF.select_specs(a)


# ---------------------------------------------------------------- (o) XB 진입점과 v5 연도 정합 도일
class FakeXS:
    """x_multisource_stacking 의 쓰는 부분만 흉내 낸다(a2_tdd_cells, A2_CELLS, tdd_tables, main)."""
    A2_CELLS = None
    called = []

    @staticmethod
    def a2_tdd_cells(lat, lon, ymin, ymax):
        n = len(lat)
        return pd.DataFrame(dict(tdd_matched=1000.0 + np.arange(n), match_flag="full", n_years_matched=1, match_lo=2015, match_hi=2015, fallback_deg=0.0))

    @staticmethod
    def tdd_tables(*a, **k):
        return pd.Series([500.0, 600.0], index=[1, 2]), pd.Series(["full", "full"], index=[1, 2]), dict(v1="x")

    @classmethod
    def main(cls, argv):
        cls.called.append(list(argv))
        return 0


def _tdd_setup(tmp_path, n_new=2):
    P = XF.Paths(tmp_path)
    P.v5.mkdir(parents=True, exist_ok=True)
    rows = [dict(loc_id=1, lat=60.0, lon=30.0, region="CALM_Russia_W")] + [dict(loc_id=30000 + i, lat=68.35, lon=19.05 + i, region="NAtlantic")
                                                                           for i in range(n_new)]
    pd.DataFrame(rows).to_csv(P.v5_table, index=False)
    pd.DataFrame(dict(loc_id=[30000 + i for i in range(n_new)], year_min=2010, year_max=2017)).to_csv(P.v5_labels, index=False)
    a2 = tmp_path / "a2.csv"
    pd.DataFrame(dict(loc_id=[1], lat=[60.0], lon=[30.0], year_min=[2015], year_max=[2015], match_flag=["full"])).to_csv(a2, index=False)
    FakeXS.A2_CELLS = a2
    return P


def test_o1_write_tdd_v5(tmp_path):
    P = _tdd_setup(tmp_path)
    m = XF.write_tdd_v5(P, n_check=0, xs=FakeXS)
    t = pd.read_csv(P.tdd_v5)
    assert m["n_new"] == 2 and t.loc_id.tolist() == [30000, 30001] and (t.part == "v5new").all() and m["sha256"] == XB.sha256_file(P.tdd_v5)
    assert not m["label_values_used"] and m["v5_sha256"] == XB.sha256_file(P.v5_table)
    P0 = _tdd_setup(tmp_path / "z", n_new=0)                                           # 새 행이 없으면 XB 를 부르지 않고 빈 표
    m0 = XF.write_tdd_v5(P0, xs=None)
    assert m0["n_new"] == 0 and len(pd.read_csv(P0.tdd_v5)) == 0


def test_o2_xb_prepare_and_entry(tmp_path):
    P = _tdd_setup(tmp_path)
    XF.write_tdd_v5(P, n_check=0, xs=FakeXS)
    XF.save_specs(P, {"Aland~xf": dict(alias="Aland~xf", target="Aland", kind=XF.KIND_NEW, run=True, keep_v4=[], keep_v5=[30000])}, P.v5_table)
    pd.DataFrame([dict(alias="Aland~xf", eligible=True)]).to_csv(P.elig, index=False)
    with pytest.raises(SystemExit):
        XF.xb_prepare(P, require_frozen=True, xs=FakeXS)                               # 실제 적합은 동결 기록이 있어야 한다
    XS, add = XF.xb_prepare(P, require_frozen=False, xs=FakeXS)
    v, flags, src = XS.tdd_tables()
    assert v.loc[30000] == 1000.0 and v.loc[1] == 500.0 and "v5" in src and flags.loc[30001] == "full"
    XF.xb_prepare(P, require_frozen=False, xs=FakeXS)                                  # 두 번 감싸지 않는다
    assert len(XS.tdd_tables()[0]) == 4
    assert XB.resolve("Aland~xf")[1] == "Aland"
    P.tdd_v5.write_text(P.tdd_v5.read_text() + "\n")                                   # 해시가 메타와 다르면 멈춘다
    with pytest.raises(SystemExit):
        XF.xb_prepare(P, require_frozen=False, xs=FakeXS)
    with pytest.raises(SystemExit, match="workers 0"):
        XF.main(["--out-root", str(P.base), "--xb", "--", "--exps", "xb_t", "--smoke"])


def test_o3_xb_resolves_xf_alias(tmp_path):
    """실제 XB 모듈: XF 별칭을 등록하면 --targets-t '<별칭>:x' 가 XF 실행 표로 풀린다(XB 파일은 고치지 않는다)."""
    XS = importlib.import_module("x_multisource_stacking")
    assert all(hasattr(XS, k) for k in ("tdd_tables", "a2_tdd_cells", "A2_CELLS", "main", "parse_args"))
    d = tmp_path / "run_tables" / "Foo~xf"
    XB.RUN_TABLES["Foo~xf"] = (str(d), "Foo", False)
    a = XS.parse_args(["--exps", "xb_t", "--targets-t", "Foo~xf:x", "--workers", "0", "--count-only"])
    assert a.T_T == [("Foo~xf", "x")] and XB.resolve("Foo~xf") == (str(d), "Foo", False)


# ---------------------------------------------------------------- (l) 실제 자료 재현 관문(HEAVY)
@HEAVY
def test_l_repro_gate_real(tmp_path):
    rc = XF.main(["--stage", "cells,cov,v5,tdd,tables", "--out-root", str(tmp_path), "--threads", "2", "--min-mem-gb", "0", "--max-mem-gb", "0"])
    assert rc == 0
    rc = XF.main(["--repro-gate", "--out-root", str(tmp_path), "--threads", "2", "--min-mem-gb", "0", "--max-mem-gb", "0"])
    g = json.loads(XF.Paths(tmp_path).gate.read_text())
    assert rc == 0 and g["passed"], [c for c in g["checks"] if not c["ok"]]
    names = {c["name"] for c in g["checks"]}
    assert {"v4_table_matches_meta", "lgd_table_unchanged:Tibet", "lgd_table_unchanged:Russia_C"} <= names and g["max_rss_mb"] > 0
