"""scripts/3_deep_learning/h51_lgd_run.py(LGD 실행기)와 scripts/2_evaluation/h52_lgd_pool.py(확장 풀 집계기) 시험.
계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B(개정 13, 14), docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 7절.

가벼운 시험(학습 없음, 기본 실행)
(a) 표 정의: 새 행 수, 뺀 v3 행, 기대 셀 수(ext_cells_v1 + v3)가 적격 표(개정 13)와 같다. GGD402 보조 행은 어느 표에도 없다.
    L42 는 다른 새 지역의 대상 셀을 더하고 NAtlantic 셀을 더할 때 v3 하위 지점 평균 행을 뺀다. (실자료가 있을 때만)
(b) v3 행만 고른 실행 표는 v3 원문과 region 을 바꾼 3줄만 다르다. (실자료)
(c) h40 인자: 등록 범위(방법 축 12방법, CatBoost, 분할 5, 추출 5, seed 2, n 격자)와 꺼진 축(α, 배치, 학습기, 중첩 선택), tag lgd,
    스레드 4, 워커 2. 대응표 없는 자료 디렉터리의 설정은 kmeans 다.
(d) 실행 보호: --allow-local 없는 학습 거부, 워커 상한 2, load average 64 초과 거부, 점검 기록 없는 표는 학습하지 않는다.
    시험 전용 인자(--skip-hash-check, --h40-extra, h52 --allow-no-repro-gate)는 LGD_TEST=1 이 없으면 거부한다(개정 14).
(e) 점검의 가짜 라벨(√TDD) dry 실행이 합성 라벨(1.3·s + 잡음) dry 실행과 같은 적합 수를 낸다(WRAPUP 7.2 (a)7).
(f) WRAPUP 7.4 (b)1: MAIN_POINT 를 적격 표로 바꾼 뒤 만든 적격 Russia_C 의 TMx 는 has_ci 가 참이고 nboot 가 등록 횟수이며
    minn 에서 절단되지 않는다. 원래 상수로 만들면 CI 가 없다. 점 추정 전용 변형 이름은 CI 가 없다.
(g) L1e 의 ⌊N/4⌋ 기준, L28 분류, 대칭 이름, 6B.5 병기 문구.
(h) WRAPUP 7.2 (a)5 의 분할 변동 범위 표지(대체 표시, seed 고정).
(i) 재현 점검 대조(개정 14): 키 분류(물리식 P0–P3 1e-9, ridge = V1·ridge 학습기 0.02, CatBoost 0.02 cm), 범위 base(6A.6)·all,
    h51 과 교차 환경 점검 스크립트(lgw_xenv_gate_i)의 분류가 같다.
(j) 합성 조각의 끝-끝 집계: 산출 파일, P4 의 L1e·L4e·L8e 판정이 h40 의 L1·L4·L8 판정과 같다, PE 행의 교차 환경 표지, (a)1 비교 행,
    L38 대칭 참조 행, 지역 수준 추론, 공통 재표집 열(L1e·L4e·L8e 지역·MEAN 행 포함), L4e-PE2 의 (a)10 병기 행, same_as 변형(주 설정과
    같은 값), 공통 난수 seed, L41 의 L28 분류에서 (a)10 병기 제외, 변형 행의 소수 블록, 6B.7 표시(L39, L40), 적격 표 기반 '변형 불가',
    교차 환경 점검 범위 표시(부분), 재현 점검 물리식 불통과 때 중단(범위 base 행).
(k) 실자료 점검 한 표(repro_Russia_W): 셀 수와 분할 구조가 적격 표와 같다. (실자료)
(l) same_as 의 수치 비교(1e-9), 묶음별 추정, 스모크 실행 표(대상 셀만 합성 라벨, h40 대상 셀과 일치). (l 의 스모크 표는 실자료)
(m) 약관 확인분 판(LG 개정 15 (m), 실행 계획 A6): 묶음 lic 의 이름(선택 묶음, all 에 들지 않음, 파이프라인 grep), 표 정의(lic 셀만 정확히
    빠지고 나머지 정의는 등록 표와 같다, (m)1 셀 수 19/38·37/75·7/8, (m)4 야말 보조 행, 저장소 이름 '~lic' 와 충돌 없음; loc_id 와 약관 열만 쓴다),
    등록 표 39개의 정의 해시·h40 설정 해시·실행 표 원문 해시 불변, 원문 줄 대조(합성 표), 6B.4 적격 규칙과 레짐의 값 없는 상한, 학습 계획의
    '변형 불가'와 묶음 추정, --count-only 경로(합성 라벨 v4: 모든 줄의 alt_cm 을 √TDD 기반 합성값으로 바꾼다), h52 의 (m)2 허락 규칙,
    두 판 집계(합성 조각: 판별 풀, 판 열, same_as·공통 난수, 두 판 병기 표, 전체 판이 약관 확인분 판의 유무와 무관하게 같음, 부적격이면
    그 판의 PE1 에서 빠짐, 조각 없음 '미완')
학습을 하는 시험(환경 변수 LGD_RUN_E2E=1)
(E) 실제 공변량과 합성 라벨로 만든 작은 자료 디렉터리에서 h51 의 h40 호출 경로가 조각(tag lgd)을 만든다.
실행: OMP_NUM_THREADS=2 nice -n 10 python3 -m pytest -q tests/test_h51_lgd.py
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
PROC = ROOT / "data" / "processed"
REAL = (PROC / "fidelity_base_v4.csv").exists() and (PROC / "lgd_eligibility_v1.csv").exists()
realdata = pytest.mark.skipif(not REAL, reason="실자료(v4, 적격 표)가 없다")
E2E = pytest.mark.skipif(os.environ.get("LGD_RUN_E2E", "") != "1", reason="학습을 하는 시험은 LGD_RUN_E2E=1 일 때만")


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("h51_lgd_run", "scripts/3_deep_learning/h51_lgd_run.py")
A = _load("h52_lgd_pool", "scripts/2_evaluation/h52_lgd_pool.py")
H = R.H
X = A.X
from polar.h4_common import BlockStore, save_stores                                                  # noqa: E402

CB = H.BASE_LEARNER


@pytest.fixture(autouse=True)
def _restore_main_point():
    orig = list(H.MAIN_POINT)
    yield
    H.MAIN_POINT = orig


# ================================================================ (a)(b) 표 정의
@pytest.fixture(scope="module")
def tables():
    return R.Tables()


@realdata
def test_a_specs_match_eligibility(tables):
    S = tables.specs()
    el = tables.el.set_index("spec")
    for name, s in S.items():
        if s["kind"] in ("repro", "v3local", "source_L42") or s.get("lic_of"):      # 약관 확인분 표(LG 개정 15 (m))는 등록 적격 표에 행이 없다
            continue
        r = el.loc[name]
        e = tables.expected_counts(s)
        assert len(s["keep"]) == int(r.n_new_cells), name
        assert e["expected_new"] == int(r.n_new_cells) and e["expected_new"] + e["expected_v3"] == int(r.n_cells), name
        dv = "" if pd.isna(r.drop_v3) else str(r.drop_v3)
        assert ";".join(str(x) for x in s["drop_v3"]) == dv, name
    yamal = set(tables.new[tables.new.lgd_role == "aux_unknown_yamal"].loc_id)
    temp = set(tables.new[tables.new.lgd_role == "aux_temp_L39"].loc_id)
    for name, s in S.items():
        assert not (set(s["keep"]) & yamal), f"{name}: GGD402 보조 행이 들어갔다"
        if name != "Tibet_L39_temp":
            assert not (set(s["keep"]) & temp), f"{name}: 지온 유도 행은 L39 표에서만 쓴다"
    # L42: 다른 새 지역의 대상 셀을 원천에 더한다. NAtlantic 셀을 더하면 17520·17569 를 뺀다
    tgt_ids = {t: set(tables.ids(t)) for t in R.NEW_MAIN.values()}
    for sp, t in R.NEW_MAIN.items():
        s = S[f"{sp}_L42"]
        assert set(s["keep"]) == set().union(*tgt_ids.values())
        assert (17520 in s["drop_v3"]) == (sp != "NAtlantic") and (17569 in s["drop_v3"]) == (sp != "NAtlantic")
        assert (17557 in s["drop_v3"]) == (sp == "Russia_C")
    assert S["Russia_C"]["drop_v3"] == [17557]
    assert S["NAtlantic_L41h"]["point_only"] and S["Russia_C_L41h"]["point_only"] and not S["NAtlantic"]["point_only"]


@realdata
def test_b_v3_only_table_differs_only_in_region(tables, tmp_path):
    s = tables.specs()["repro_Russia_W"]
    d, rec = R.write_table(tables, s, tmp_path)
    chk = R.v3_only_text_check(tables, d / "fidelity_base_v3.csv")
    assert chk["ok"] and chk["n_lines_differ"] == 3 and chk["differ_only_in_region"]
    assert (d / "e5_soil_tdd_v3.csv").is_symlink()
    assert (tmp_path / "run_tables" / ".gitignore").read_text().splitlines()[-2:] == ["*", "!.gitignore"]


# ================================================================ (c) h40 인자
def test_c_h40_args_registered_scope(tmp_path):
    av = R.h40_argv(tmp_path, tmp_path / "out", "NAtlantic", R.SPLITS, R.THREADS, R.WORKERS_MAX)
    a = H.parse_args(av)
    assert a.TARGETS == [("NAtlantic", "x")] and a.SPLITS == [1, 2, 3, 4, 5] and a.DRAWS == 5 and a.SEEDS == [0, 1]
    assert a.N_GRID == [0, 3, 10, 40, 160, 320, 1000, -1] and a.METHODS == H.METHODS_ALL and a.LEARNERS == ["catboost_lo"]
    assert a.ALPHAS == ["1"] and a.PLACE_N == [] and a.NESTED_ALPHAS == []
    assert ("NAtlantic", "x") not in a.NESTED_TARGETS and ("NAtlantic", "x") not in a.LEARNER_TARGETS
    assert not H.is_learner_unit(a, "NAtlantic", "x", 1)
    assert a.TAG == "lgd" and a.threads == 4 and a.workers == 2 and a.allow_local and a.no_summarize and a.resume
    assert a.PROC == tmp_path and a.buffer_km == 100.0 and a.kappa == 10.0 and a.r == 10.0 and a.LAMS == [0.25, 0.5, 1.0]
    cfg = H.unit_cfg(a, "cpu", "NAtlantic", "x", 1)
    assert cfg["subregion_map"] == "kmeans" and cfg["alphas"] == ["1"] and cfg["place_n"] == [] and cfg["nested"] is False
    ac = H.parse_args(R.h40_argv(tmp_path, tmp_path / "o", "Russia_W", 1, 2, 0, resume=False, count=True))
    assert ac.count_only and not ac.allow_local and ac.SPLITS == [1]


# ================================================================ (d) 실행 보호
def test_d_guards(monkeypatch, tmp_path):
    monkeypatch.delenv("LGD_TEST", raising=False)
    with pytest.raises(SystemExit, match="시험 전용"):
        R.parse_args(["--skip-hash-check"])
    with pytest.raises(SystemExit, match="시험 전용"):
        R.parse_args(["--h40-extra", "--n-grid 0"])
    with pytest.raises(SystemExit, match="시험 전용"):
        A.parse_args(["--allow-no-repro-gate"])
    with pytest.raises(SystemExit, match="함께 쓰지 않는다"):
        R.parse_args(["--smoke", "--count-only"])
    monkeypatch.setenv("LGD_TEST", "1")
    assert R.parse_args(["--skip-hash-check"]).skip_hash_check and R.parse_args(["--specs", "main"]).ARGV == ["--specs", "main"]
    with pytest.raises(SystemExit, match="allow-local"):
        R.main(["--specs", "NAtlantic", "--out-dir", str(tmp_path), "--skip-hash-check"])
    with pytest.raises(SystemExit, match="이하"):
        R.parse_args(["--workers", "3"])
    monkeypatch.setattr(R.os, "getloadavg", lambda: (70.0, 1.0, 1.0))
    with pytest.raises(SystemExit, match="load average"):
        R.main(["--specs", "NAtlantic", "--allow-local", "--out-dir", str(tmp_path), "--skip-hash-check"])
    a = R.parse_args(["--specs", "main,l39,repro"])
    assert a.SPEC_NAMES == ["Tibet", "NAtlantic", "Russia_C", "Tibet_L39_temp", "repro_Russia_W"]
    assert R.parse_args([]).SPEC_NAMES[:3] == ["Tibet", "NAtlantic", "Russia_C"] and "repro_Russia_W" not in R.parse_args([]).SPEC_NAMES


@realdata
def test_d2_training_needs_count_record(monkeypatch, tmp_path):
    """점검 기록(manifest 의 check_ok 와 같은 실행 표 해시)이 없으면 학습하지 않는다."""
    called = []
    monkeypatch.setenv("LGD_TEST", "1")
    monkeypatch.setattr(R, "run_spec", lambda av: called.append(av) or {})
    monkeypatch.setattr(R.os, "getloadavg", lambda: (1.0, 1.0, 1.0))
    res = R.main(["--specs", "NAtlantic", "--allow-local", "--out-dir", str(tmp_path), "--skip-hash-check", "--workers", "0"])
    assert not called and res["blocked"] and res["blocked"][0]["spec"] == "NAtlantic"


# ================================================================ (e) 가짜 라벨 dry 실행
def _syn_ctx(dummy=False, seed=0):
    rng = np.random.RandomState(seed)
    ns, nA, nB, d = 300, 24, 20, len(H.FEATS)
    Xs, XA, XB = rng.randn(ns, d), rng.randn(nA, d), rng.randn(nB, d)
    ss, sA, sB = rng.uniform(20, 40, ns), rng.uniform(20, 40, nA), rng.uniform(20, 40, nB)
    ys, yA, yB = 1.6 * ss + rng.randn(ns), 1.3 * sA + rng.randn(nA), 1.3 * sB + rng.randn(nB)
    if dummy:
        ys, yA, yB = ss.copy(), sA.copy(), sB.copy()
    c = H.Ctx("T", "x", 1, "T", Xs, ys, ss, np.repeat(["r1", "r2", "r3"], ns // 3), XA, yA, sA, np.repeat(np.arange(4), nA // 4),
              XB, yB, sB, np.repeat(np.arange(5), nB // 5))
    if dummy:
        c.prior = None
    return c


def test_e_dummy_label_counts_equal_real(tmp_path):
    a = H.parse_args(R.h40_argv(tmp_path, tmp_path, "T", 1, 1, 0, resume=False, count=True))
    _, _, s_real = H.run_ctx(_syn_ctx(False), "cpu", a, learner_axis=False, nested=False, dry=True)
    _, _, s_dum = H.run_ctx(_syn_ctx(True), "cpu", a, learner_axis=False, nested=False, dry=True)
    assert s_real["n_fit"] == s_dum["n_fit"] and s_real["n_rows"] == s_dum["n_rows"] and s_real["n_fit_detail"] == s_dum["n_fit_detail"]
    assert sum(s_real["n_fit"].values()) > 0 and "alpha" not in s_real["n_fit"] and "place" not in s_real["n_fit"]


# ================================================================ (f) MAIN_POINT(WRAPUP 7.4 (b)1)
def _mk_store(name, split, nb, keys, seed, p0_sd=6.0):
    rng = np.random.RandomState(seed + 17 * split)
    blocks = np.repeat(np.arange(nb), 5)
    st = BlockStore(name, split, blocks)
    y = rng.randn(len(blocks)) * 5 + 50
    u = rng.randn(len(y))
    st.add(H.P0_KEY, y, y + p0_sd * u)
    for k, sd in keys.items():
        st.add(k, y, y + float(sd) * u)
    return st


def K(method, n, lam=None, draw=0):
    if method in ("P1", "P2", "P3", "V1"):
        return (method, "none", "1", "cell", int(n), draw, -1, 0.0)
    if method in ("D0", "D1"):
        return (method, CB, "1", "cell", int(n), draw, 0, 1.0)
    return (method, CB, "1", "cell", int(n), draw, 0, float(H.LAM_BASE if lam is None else lam))


def _keys(ns, sd=dict(P1=5.0, P2=5.5, P3=5.2, D0=7.0, R1=4.0), draws=(0, 1)):
    out = {}
    for n in ns:
        for dr in ((0,) if n in (0, -1) else draws):
            for m, s in sd.items():
                out[K(m, n, draw=dr)] = s
    return out


def test_f_main_point_override():
    keys = _keys((0, 3, 10, -1))
    by = {sp: _mk_store("Russia_C|x", sp, 9, keys, 3) for sp in (1, 2)}
    info = {1: dict(dup_of=-1, valid=True), 2: dict(dup_of=-1, valid=True)}
    tm_orig = H.TMx("Russia_C|x", by, info, 1000)                    # 원래 상수: 러시아 C 는 점 추정만
    assert tm_orig.point_only and not tm_orig.has_ci and tm_orig.nboot == 0
    el = pd.DataFrame(dict(spec=["NAtlantic", "Russia_C", "Tibet"], kind=["new_region", "new_region(v3 셀 확충)", "new_region"],
                           target=["NAtlantic", "Russia_C", "Tibet_LGD"], eligible=[True, True, True], regime=["shallow", "shallow", "deep"],
                           min_nb_eval_used=[3.0, 6.0, 14.0]))
    pools = A.pools_from_eligibility(el)
    assert pools["PE1"] == A.P4 + ["NAtlantic|x", "Russia_C|x"] and pools["PE2"] == pools["PE1"] + ["Tibet_LGD|x"] and pools["ineligible"] == []
    mp = A.set_main_point(pools["ineligible"], ["Russia_C~L41h"])
    assert mp == ["Russia_C~L41h"] and H.MAIN_POINT == mp
    tm = H.TMx("Russia_C|x", by, info, 1000)
    assert tm.has_ci and not tm.point_only and tm.nboot == 1000
    tmv = H.TMx("Russia_C~L41h|x", by, info, 1000)
    assert tmv.point_only and not tmv.has_ci
    cur = H.build_curve({"Russia_C|x": A.method_view(tm)}, _runs_of(by, "Russia_C", 20))
    mn = H.build_minn(cur)
    r = mn[(mn.method == "R1") & (mn.lam == H.LAM_BASE)].iloc[0]
    assert not bool(r.point_only) and "point_only" not in str(r.flag) and np.isfinite(r.n_star)
    el2 = el.copy(); el2.loc[el2.target == "NAtlantic", "eligible"] = False
    p2 = A.pools_from_eligibility(el2)
    assert p2["ineligible"] == ["NAtlantic"] and "NAtlantic|x" not in p2["PE1"]
    assert A.set_main_point(p2["ineligible"]) == ["NAtlantic"]


def _runs_of(by, target, n_lab_all):
    rows = []
    for sp, st in by.items():
        for k in st.keys:
            rows.append(dict(target=target, mode="x", parent=target, split=sp, part="cpu", axis="method", method=k[0], learner=k[1], alpha=str(k[2]),
                             placement=k[3], n=int(k[4]), n_lab=(n_lab_all if int(k[4]) == -1 else max(int(k[4]), 0)), draw=int(k[5]), seed=int(k[6]),
                             lam=float(k[7]), rmse_cm=st.rmse(k), rmse_beq_cm=st.rmse(k), bias_cm=0.0, E_used=np.nan, alpha_sel="", n_blocks_lab=0,
                             n_nonfinite=0, fit_flag=""))
    return pd.DataFrame(rows)


# ================================================================ (g) 규칙 함수
def _cur_l1(sigs, nlab_all=None):
    rows = []
    for t, m in sigs.items():
        for n, s in m.items():
            rows.append(dict(target=t, mode="x", method="D0", learner=CB, alpha="1", placement="cell", n=n, lam=1.0, sig_p0=s,
                             n_lab=(nlab_all or {}).get(t, 20) if n == -1 else n, d_p0=-1.0, d_p0_lo=-2, d_p0_hi=-0.5, d_p0_beq=-1,
                             d_p0_beq_lo=-2, d_p0_beq_hi=-0.5, ci_flag=""))
    return pd.DataFrame(rows)


def test_g_l1e_threshold_and_rules():
    names = [f"r{i}|x" for i in range(8)]
    base = {f"r{i}": {0: "ns", 3: "ns", 10: "ns"} for i in range(8)}
    for N, nbeat, want in ((4, 1, "지지"), (4, 2, "기각"), (6, 1, "지지"), (6, 2, "기각"), (7, 1, "지지"), (8, 2, "지지"), (8, 3, "기각")):
        sig = {k: dict(v) for k, v in list(base.items())[:N]}
        for i in range(nbeat):
            sig[f"r{i}"][3] = "improve"
        rows, b = A.l1e(_cur_l1(sig), names[:N], "T")
        v = [r for r in rows if r["scope"] == "verdict"][0]
        assert v["verdict"] == want and b == want and v["k_improve"] == nbeat and f"k/N = {nbeat}/{N}" in v["stat"], (N, nbeat)
    # 전량 행은 실제 라벨 수 ≤ 40 인 지역에서만 센다
    sig = {"r0": {0: "ns", -1: "improve"}, "r1": {0: "ns", -1: "improve"}, "r2": {0: "ns"}, "r3": {0: "ns"}}
    rows, b = A.l1e(_cur_l1(sig, dict(r0=300, r1=30)), names[:4], "T")
    assert b == "지지"
    assert A.l28_shift("우세", "우세") == "강건" and A.l28_shift("동등", "미결정") == "강건" and A.l28_shift("우세", "미결정") == "약화"
    assert A.l28_shift("열세", "우세") == "의존" and A.l28_shift("행 없음", "우세") == "판정 불가"
    assert A.worst_shift(["강건", "약화"]) == "약화" and A.worst_shift(["강건", "의존", "약화"]) == "의존"
    assert A.worst_shift(["판정 불가", "판정 불가"]).startswith("판정 불가") and A.worst_shift(["강건", "판정 불가"]) == "강건(판정한 대비 1/2)"
    assert A.sym_dir("열세", 1.2, +1) == "같은 방향(우세)" and A.sym_dir("우세", -1.2, +1) == "반대(열세)"
    assert A.sym_dir("미결정", 0.3, +1) == "같은 부호(비유의)" and A.sym_dir("동등", -0.1, +1) == "반대 부호(비유의)"
    assert A.sym_dir("우세", -2.0, -1) == "같은 방향(우세)" and A.sym_dir("미결정", 0.4, -1) == "반대 부호(비유의)"
    assert A.sym_dir("판정 불가", 0.4, -1) == ""
    assert A.phrase_6b5("지지", "지지", "지지", 3, 7) == "4지역의 판정이 확장 풀(지역 7개)에서 유지된다"
    assert A.phrase_6b5("지지", "지지", "기각", 3, 7).startswith("얕은 레짐의 확장 풀에서 유지된다")
    assert A.phrase_6b5("지지", "기각", "기각", 3, 7) == "4지역의 판정은 확장 풀에서 유지되지 않는다"
    assert A.phrase_6b5("지지", None, "기각", 3, 7).startswith("판정 불가") and A.phrase_6b5("지지", "지지", "지지", 0, 4) == "확장 풀을 구성하지 못했다"


# ================================================================ (h) 분할 변동 범위
def test_h_split_range_flag():
    v = {str(i): float(x) for i, x in enumerate(np.linspace(-2, 2, 50), 1)}
    sd = pd.DataFrame([dict(source="n4", target="Canada|x", contrast=c, delta_splits=json.dumps(v), primary_source=True)
                       for c in ("D0-P0|n0", "P1-P0|n10", "R1-P0|all")])
    f_in = A.split_range_flag(sd, "Canada|x", "R1", "P0", -1, 0.05, 2000, "")
    f_out = A.split_range_flag(sd, "Canada|x", "R1", "P0", -1, 1.9, 2000, "")
    assert f_in["split_range"] == "분할 변동 범위" and f_out["split_range"] == "" and f_in["split_p10"] < 0 < f_in["split_p90"]
    f_sub = A.split_range_flag(sd, "Canada|x", "D0", "P0", 10, 0.0, 2000, "D0 − P0(n = 0) 분포로 대신")
    assert f_sub["split_range"].startswith("분할 변동 범위") and "대신" in f_sub["split_range"] and f_sub["split_source"].startswith("D0-P0|n0")
    assert A.split_range_flag(sd, "Canada|x", "R1", "P0", -1, 0.05, 2000, "")["split_p10"] == f_in["split_p10"]      # seed 고정
    assert A.split_range_flag(None, "Canada|x", "R1", "P0", -1, 0.0, 10, "")["split_range"].startswith("분포 없음")


# ================================================================ (i) 재현 점검 대조
def test_i_compare_stores():
    keys = _keys((0, 3), sd=dict(P1=5.0, P3=5.2, D0=7.0, R1=4.0, V1=4.8, V1r=4.5))
    a_ = _mk_store("Russia_W|x", 1, 6, keys, 1)
    b_ = _mk_store("Russia_W|x", 1, 6, keys, 1)
    ra, rb = R.compare_stores(a_, b_, scope="all"), R.compare_stores(a_, b_, scope="base")
    assert ra["pass_phys"] and ra["pass_ml"] and ra["blocks_equal"] and ra["max_diff_phys"] == 0.0 and ra["n_keys"] == len(a_.keys)
    assert rb["n_keys"] == sum(1 for k in a_.keys if k[0] in R.BASE_METHODS) < ra["n_keys"]
    # 분류: P0–P3 물리식, V1 ridge(학습기 none 이어도 물리식이 아니다), V1r·R1·D0 CatBoost
    assert R.key_class(K("P3", 3)) == "phys" and R.key_class(K("V1", 3)) == "ridge" and R.key_class(K("V1r", 0)) == "catboost"
    assert R.key_class(("R2", "ridge", "1", "cell", 3, 0, 0, 1.0)) == "ridge"
    assert ra["n_keys_ridge"] == sum(1 for k in a_.keys if k[0] == "V1") and rb["n_keys_ridge"] == 0
    s, c = b_.get(K("V1r", 0))
    b_.add_sse(K("V1r", 0), s * 1.05, c)                                 # 기준 방법 밖의 CatBoost 키(V1r) 하나를 바꾼다
    ra, rb = R.compare_stores(a_, b_, scope="all"), R.compare_stores(a_, b_, scope="base")
    assert ra["pass_phys"] and not ra["pass_ml"] and ra["methods_ml_over_tol"] == "V1r" and ra["n_ml_over_tol"] == 1
    assert rb["pass_phys"] and rb["pass_ml"]                             # 주 판정 범위(6A.6)에는 V1r 이 없다
    assert R.gate_status(dict(rb)) == "ok" and R.gate_status(dict(ra)).startswith("병기: CatBoost 불통과")
    s, c = b_.get(K("V1", 3))
    b_.add_sse(K("V1", 3), s * (1 + 1e-7), c)                            # V1 의 아주 작은 차: 물리식 허용 차가 아니라 ridge 허용 차로 본다
    assert R.compare_stores(a_, b_, scope="all")["pass_phys"]
    s, c = b_.get(H.P0_KEY)
    b_.add_sse(H.P0_KEY, s + 1e-6, c)
    assert not R.compare_stores(a_, b_, scope="base")["pass_phys"] and not R.compare_stores(a_, b_, scope="all")["pass_phys"]
    assert R.gate_status(R.compare_stores(a_, b_, scope="base")).startswith("물리식 불통과")
    XE = _load("lgw_xenv_gate_i", "scripts/2_evaluation/lgw_xenv_gate_i.py")
    for k in list(a_.keys) + [("R2", "ridge", "1", "cell", 3, 0, 0, 1.0)]:
        assert XE.key_class(k) == R.key_class(k), k
    assert XE.BASE_METHODS == R.BASE_METHODS and XE.TOL_PHYS == R.GATE_TOL_PHYS and XE.TOL_ML == R.GATE_TOL_ML


# ================================================================ (j) 합성 조각의 끝-끝 집계
def _write_shards(root, tag, target, splits, nb, keys, seed, n_lab_all, p0_sd=6.0):
    d = root / "shards"; d.mkdir(parents=True, exist_ok=True)
    for sp in splits:
        st = _mk_store(f"{target}|x", sp, nb, keys, seed, p0_sd)
        b = d / f"{tag}__cpu__{target}__x__s{sp}"
        save_stores([st], Path(str(b) + "_blocksse.npz"))
        _runs_of({sp: st}, target, n_lab_all).to_csv(str(b) + "_runs.csv", index=False)
        Path(str(b) + "_unit.json").write_text(json.dumps(dict(target=target, mode="x", split=sp, part="cpu", status="ok", dup_of=-1, valid=True,
                                                               code_sha="test", cfg_common="test")))


def _synthetic_world(tmp_path, xenv_ml=0.46, xenv_units=(4, 4)):
    lg, lgd = tmp_path / "lg", tmp_path / "lgd"
    p4_keys = _keys((0, 3, 10, 40, -1))
    for i, (t, nl) in enumerate((("Lena", 1500), ("Canada", 370), ("Russia_W", 16), ("Russia_E", 15))):
        _write_shards(lg, "lg", t, (1, 2), 10, p4_keys, 10 + i, nl)
    new_keys = _keys((0, 3, 10, -1))
    new_keys.update({K("P2", 10): 5.5, K("P3", 10): 5.2})
    specs = {}
    for i, (sp, t) in enumerate((("NAtlantic", "NAtlantic"), ("Russia_C", "Russia_C"), ("Tibet", "Tibet_LGD"))):
        _write_shards(lgd / sp, "lgd", t, (1, 2), 9, new_keys, 30 + i, 22)
        specs[sp] = dict(target=t, tm_name=f"{t}|x", point_only=False, check_ok=True, n_units=2)
    for i, (sp, t) in enumerate((("Russia_W_expanded", "Russia_W"), ("Russia_E_expanded", "Russia_E"), ("Canada_expanded", "Canada"))):
        _write_shards(lgd / sp, "lgd", t, (1, 2), 10, p4_keys, 50 + i, 20 if t != "Canada" else 400)
        specs[sp] = dict(target=t, tm_name=f"{t}~exp|x", point_only=False, check_ok=True, n_units=2)
    _write_shards(lgd / "NAtlantic_L41c", "lgd", "NAtlantic", (1, 2), 9, new_keys, 70, 18)
    specs["NAtlantic_L41c"] = dict(target="NAtlantic", tm_name="NAtlantic~L41c|x", point_only=False, check_ok=True, n_units=2)
    specs["NAtlantic_L41b"] = dict(target="NAtlantic", tm_name="NAtlantic~L41b|x", point_only=False, check_ok=True, n_units=2, same_as="NAtlantic")
    specs["NAtlantic_L41a"] = dict(target="NAtlantic", tm_name="NAtlantic~L41a|x", point_only=True, check_ok=False, n_units=0,
                                   note="; 변형 불가(부적격)")
    _write_shards(lgd / "NAtlantic_L41h", "lgd", "NAtlantic", (1, 2), 9, new_keys, 71, 15)
    specs["NAtlantic_L41h"] = dict(target="NAtlantic", tm_name="NAtlantic~L41h|x", point_only=True, check_ok=True, n_units=2,
                                   note="점 추정만(WRAPUP 7.3 (a)9)")
    _write_shards(lgd / "Tibet_L41d", "lgd", "Tibet_LGD", (1, 2), 9, new_keys, 72, 22)
    specs["Tibet_L41d"] = dict(target="Tibet_LGD", tm_name="Tibet_LGD~L41d|x", point_only=False, check_ok=True, n_units=2)
    specs["Tibet_L41a"] = dict(target="Tibet_LGD", tm_name="Tibet_LGD~L41a|x", point_only=True, check_ok=True, n_units=0)   # 적격 표가 정한다
    _write_shards(lgd / "Tibet_L39_temp", "lgd", "Tibet_LGD", (1, 2), 9, new_keys, 73, 22)
    specs["Tibet_L39_temp"] = dict(target="Tibet_LGD", tm_name="Tibet_LGD~L39|x", point_only=False, check_ok=True, n_units=2)
    (lgd).mkdir(parents=True, exist_ok=True)
    (lgd / "lgd_run_manifest.json").write_text(json.dumps(dict(specs=specs)))
    el = pd.DataFrame(dict(spec=["NAtlantic", "Russia_C", "Tibet", "NAtlantic_L41a", "NAtlantic_L41b", "NAtlantic_L41c", "NAtlantic_L41h",
                                 "Tibet_L41a", "Tibet_L41d"],
                           kind=["new_region", "new_region(v3 셀 확충)", "new_region", "variant_L41a", "variant_L41b", "variant_L41c", "variant_L41h",
                                 "variant_L41a", "variant_L41d"],
                           target=["NAtlantic", "Russia_C", "Tibet_LGD", "NAtlantic", "NAtlantic", "NAtlantic", "NAtlantic", "Tibet_LGD", "Tibet_LGD"],
                           eligible=[True, True, True, False, True, True, True, False, True],
                           regime=["shallow", "shallow", "deep", "shallow", "shallow", "shallow", "shallow", "deep", "deep"],
                           min_nb_eval_used=[3.0, 6.0, 14.0, 3.0, 3.0, 4.0, 4.0, np.nan, 14.0]))
    elp = tmp_path / "elig.csv"; el.to_csv(elp, index=False)
    xj = tmp_path / "xenv.json"
    xj.write_text(json.dumps(dict(max_phys=0.0, max_ml=xenv_ml, blocks_equal=True,
                                  scope=dict(units_done=xenv_units[0], units_expected=xenv_units[1]))))
    v = {str(i): float(x) for i, x in enumerate(np.linspace(-1, 1, 50), 1)}
    sd = pd.DataFrame([dict(source="n4", target=f"{t}|x", contrast=c, delta_splits=json.dumps(v), primary_source=True)
                       for t in ("Russia_W", "Russia_E", "Canada") for c in ("D0-P0|n0", "P1-P0|n10", "R1-P0|all")])
    sdp = tmp_path / "splitdist.csv"; sd.to_csv(sdp, index=False)
    return lg, lgd, elp, xj, sdp


def _agg_args(tmp_path, lg, lgd, elp, xj, sdp, extra=()):
    return ["--lg-dir", str(lg), "--lgd-dir", str(lgd), "--eligibility", str(elp), "--xenv-gate", str(xj), "--splitdist", str(sdp),
            "--nboot-h40", "200", "--nboot", "300", "--l40-draws", "200", "--out-dir", str(tmp_path / "out")] + list(extra)


def test_j_end_to_end_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("LGD_TEST", "1")
    lg, lgd, elp, xj, sdp = _synthetic_world(tmp_path)
    with pytest.raises(SystemExit, match="재현 점검"):
        A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp))
    res = A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp, ["--allow-no-repro-gate"]))
    O = tmp_path / "out"
    for f in ("lgd_curve.csv", "lgd_minn.csv", "lgd_tests.csv", "lgd_region_inference.csv", "lgd_pool.csv", "lgd_meta.json"):
        assert (O / f).exists(), f
    t = res["tests"]
    meta = res["meta"]
    assert meta["main_point"]["overridden"] == ["NAtlantic~L41a", "NAtlantic~L41h", "Tibet_LGD~L41a"] and meta["xenv_i"] == "(i) CatBoost 불통과"
    assert meta["pools"]["PE2"] == A.P4 + ["NAtlantic|x", "Russia_C|x", "Tibet_LGD|x"]
    v = t[t.scope == "verdict"]
    for hyp in ("L1e", "L4e", "L8e"):
        assert set(v[v.test_id == hyp].pool) == {"P4", "PE1", "PE2"}, hyp
    # P4 의 L1e·L4e·L8e 는 h40 의 L1·L4·L8 판정과 같다(같은 조각, 같은 재표집)
    tms = {nm: A.method_view(H.TMx(nm, *_by_info(lg, nm.split("|")[0]), 200)) for nm in A.P4}
    runs = pd.concat([_runs_from_dir(lg, nm.split("|")[0]) for nm in A.P4], ignore_index=True)
    cur = H.build_curve(tms, runs)
    ht = H.build_tests(tms, cur, H.build_minn(cur), H.parse_args([]))
    hv = ht[ht.scope == "verdict"].set_index("test_id").verdict
    for hyp, h40id in (("L1e", "L1"), ("L4e", "L4"), ("L8e", "L8")):
        assert v[(v.test_id == hyp) & (v.pool == "P4")].verdict.iloc[0] == hv[h40id], hyp
    pe = t[(t.pool.isin(["PE1", "PE2"])) & (t.test_id.isin(["L1e", "L4e", "L8e"]))]
    assert (pe.cross_env == "교차 환경; (i) CatBoost 불통과").all()
    cmp_ = t[t.scope == "compare"]
    assert set(cmp_.test_id) == {"L1e", "L4e", "L8e"}
    l38v = v[v.test_id == "L38"].iloc[0]
    assert "대칭 참조(P4)" in l38v.stat and "NOVELTY-N7" in str(l38v.note)
    l38r = t[(t.test_id == "L38") & (t.scope == "region")]
    assert "소수 블록" in set(";".join(l38r[l38r.target == "NAtlantic|x"]["flags"].astype(str)).split(";"))
    tib = l38r[l38r.target == "Tibet_LGD|x"]
    assert {"(c) R1-P2", "(c) R1-P3", "(c) P2-P1"} <= set(tib.item) and tib[tib.item == "(b) R1-P0"]["flags"].str.contains("예측된 결과").all()
    assert {"verdict4_common", "ci_dependence", "direction"} <= set(t.columns)
    l41 = v[(v.test_id == "L41") & v.target.astype(str).str.startswith("NAtlantic")].set_index("variant")
    assert v[(v.test_id == "L41") & v.target.astype(str).str.startswith("Russia_C")].verdict.str.startswith("행 없음").all()
    assert l41.loc["(a)"].verdict.startswith("변형 불가") and l41.loc["(b)"].verdict == "강건"
    assert "주 설정과 같은 실행 표" in str(l41.loc["(b)"].note) and l41.loc["(h)"].verdict.startswith("점 추정만")
    assert "공통 난수" in str(l41.loc["(c)"].note) and meta["crn"]["NAtlantic_L41c"] == "seed ← NAtlantic|x"
    assert meta["crn"]["NAtlantic_L41b"].startswith("same_as")
    # same_as 변형의 행은 주 설정 행과 같은 값이다(같은 TMx)
    rb_ = t[(t.test_id == "L41") & (t.scope == "region") & (t.variant == "(b)")].set_index(["item", "n"])
    rm_ = l38r[l38r.target == "NAtlantic|x"].set_index(["item", "n"])
    for col in ("delta", "ci_lo", "ci_hi", "ci_lo_c", "ci_hi_c", "verdict4"):
        assert (rb_[col].astype(str) == rm_.loc[rb_.index, col].astype(str)).all(), col
    # 변형 행의 소수 블록(적격 표 min_nb_eval_used < 5)
    assert (t[(t.test_id == "L41") & (t.variant.isin(["(b)", "(c)"])) & t.target.astype(str).str.startswith("NAtlantic")].few_blocks == "소수 블록").all()
    # Tibet L41: (a)는 적격 표에서 부적격, (d)의 L28 분류는 L38 (a)–(c)만(병기 대비는 aux_l28)
    tv = v[(v.test_id == "L41") & v.target.astype(str).str.startswith("Tibet")].set_index("variant")
    assert tv.loc["(a)"].verdict.startswith("변형 불가") and "R1-P2" not in str(tv.loc["(d)"].stat) and "R1-P2" in str(tv.loc["(d)"].aux_stat)
    assert str(tv.loc["(d)"].aux_l28) != "" and "(c) R1-P1|n10" in str(tv.loc["(d)"].stat)
    l40 = v[v.test_id == "L40"]
    assert len(l40) == 4 and (l40[~l40.verdict.str.startswith("행 없음")].cross_env == "교차 환경(보조); (i) CatBoost 불통과").all()
    assert (t[t.test_id == "L40"]["flags"].astype(str).str.contains("재현(비맹검)", regex=False)).all()
    l39 = t[t.test_id == "L39"]
    assert len(l39) > 1 and l39["flags"].astype(str).str.contains("비맹검(라벨 통계 열람)", regex=False).all()
    # (a)4 공통 재표집 CI: L1e 지역 행, L4e·L8e 지역·MEAN 행
    for hyp in ("L1e", "L4e", "L8e"):
        q = t[(t.test_id == hyp) & t.scope.isin(["region", "MEAN"])]
        assert q.ci_lo_c.notna().any() and "sig_common" in q and q.sig_common.astype(str).isin(["improve", "ns", "worse", "", "nan"]).all(), hyp
    q = t[(t.test_id == "L4e") & (t.scope == "MEAN") & (t.item == "R1-P1")]
    assert q.ci_lo_c.notna().all() and q.n_common_regions.min() >= 2
    # (a)10: L4e-PE2 병기 행(PE2 층화 평균과 티베트 행), 다른 풀에는 없다
    ax = t[(t.test_id == "L4e") & (t.item.isin(["R1-P2", "R1-P3", "P2-P1"]))]
    assert set(ax.pool) == {"PE2"} and {"R1-P2", "R1-P3", "P2-P1"} <= set(ax.item) and (ax.scope == "MEAN").any()
    assert ax.target.astype(str).str.startswith("Tibet").any() and ax["flags"].astype(str).str.contains("병기").all()
    # 6B.7: 티베트 P0 기반 행
    tib1 = t[(t.test_id.isin(["L1e", "L8e"])) & (t.scope == "region") & t.target.astype(str).str.startswith("Tibet")]
    assert len(tib1) and tib1["flags"].astype(str).str.contains("비맹검(라벨 통계 열람)", regex=False).all()
    l40r = t[(t.test_id == "L40") & (t.scope == "region") & (t.side == "확충판")]
    assert l40r.split_range.notna().any() and l40r.split_source.astype(str).str.contains("n0").any()
    ri = res["region_inference"]
    assert {"P4", "PE1", "PE2"} <= set(ri.pool) and "hk_lo" in ri and ri.wording.notna().any()
    pool = res["pool"].set_index("pool")
    assert pool.loc["PE1", "n_regions"] == 6 and pool.loc["PE2", "n_regions"] == 7 and "NAtlantic" in pool.loc["PE1", "few_blocks"]
    # 재현 점검 물리식 불통과면 멈춘다(주 판정 범위 base 행으로 정한다)
    pd.DataFrame([dict(scope="base", pass_phys=False, pass_ml=True, blocks_equal=True),
                  dict(scope="all", pass_phys=True, pass_ml=True, blocks_equal=True)]).to_csv(tmp_path / "gate.csv", index=False)
    with pytest.raises(SystemExit, match="물리식"):
        A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp, ["--repro-gate", str(tmp_path / "gate.csv")]))
    # 병기 범위(all)만 CatBoost 불통과면 상태는 통과다
    pd.DataFrame([dict(scope="base", pass_phys=True, pass_ml=True, blocks_equal=True, status="ok"),
                  dict(scope="all", pass_phys=True, pass_ml=False, blocks_equal=True, status="병기: CatBoost 불통과")]).to_csv(tmp_path / "gate.csv", index=False)
    r3 = A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp, ["--repro-gate", str(tmp_path / "gate.csv")]))
    assert r3["meta"]["repro_gate"] == "재현 점검 통과" and r3["meta"]["l40_reference"].startswith("LG 본 실행")
    assert r3["meta"]["repro_gate_all_scope"][0]["status"].startswith("병기")
    # CatBoost 만 불통과면 L40 은 로컬 v3 기준값(없으면 보류)
    pd.DataFrame([dict(scope="base", pass_phys=True, pass_ml=False, blocks_equal=True)]).to_csv(tmp_path / "gate.csv", index=False)
    r2 = A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp, ["--repro-gate", str(tmp_path / "gate.csv")]))
    assert r2["meta"]["l40_reference"].startswith("판정 보류") and r2["meta"]["repro_gate"] == "재현 점검 CatBoost 불통과"
    # 교차 환경 점검 범위가 등록 범위보다 작으면 '부분(k/K 단위, 잠정)'
    xj.write_text(json.dumps(dict(max_phys=0.0, max_ml=0.46, blocks_equal=True, scope=dict(units_done=1, units_expected=4))))
    r4 = A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp, ["--repro-gate", str(tmp_path / "gate.csv")]))
    assert r4["meta"]["xenv_i"] == "(i) CatBoost 불통과; 부분(1/4 단위, 잠정)"
    xj.write_text(json.dumps(dict(max_phys=0.0, max_ml=0.46)))
    assert A.xenv_status(A.parse_args(["--xenv-gate", str(xj)]))[0].endswith("범위 미기록(부분 점검일 수 있다)")


def _by_info(lg, target):
    from polar.h4_common import load_stores
    sh = sorted((lg / "shards").glob(f"lg__cpu__{target}__x__s*_blocksse.npz"))
    st = load_stores(sh)
    by = {sp: s for (nm, sp), s in st.items()}
    return by, {sp: dict(dup_of=-1, valid=True) for sp in by}


def _runs_from_dir(lg, target):
    return pd.concat([pd.read_csv(p, dtype=dict(alpha=str, alpha_sel=str, fit_flag=str), keep_default_na=False, na_values=["", "nan", "NaN"])
                      for p in sorted((lg / "shards").glob(f"lg__cpu__{target}__x__s*_runs.csv"))], ignore_index=True)


# ================================================================ (k) 실자료 점검 한 표
@realdata
def test_k_count_one_real_spec(tables, tmp_path):
    s = tables.specs()["repro_Russia_W"]
    d, rec = R.write_table(tables, s, tmp_path)
    rows, summ = R.count_spec(tables, s, d, tmp_path, 1)
    assert summ["check_ok"] and summ["n_cells"] == 31 and summ["n_split_mismatch"] == 0 and summ["n_fit"] > 0
    assert rows[0]["n_src"] == int(tables.el.set_index("spec").loc["v3ref_Russia_W", "n_src"])


# ================================================================ (E) 학습 경로(합성 라벨)
@E2E
def test_E_h40_path_on_synthetic_labels(tmp_path):
    v3 = pd.read_csv(PROC / "fidelity_base_v3.csv", low_memory=False)
    keep = v3[v3.region.isin(["CALM_Russia_W"])].index.tolist() + v3[v3.region.isin(["Lena_RU"])].index[:300].tolist()
    sub = v3.loc[keep].copy()
    rng = np.random.RandomState(0)
    sub["alt_cm"] = np.clip(1.4 * sub.e5_sqrt_tdd.fillna(30).values * np.exp(0.2 * rng.randn(len(sub))), 5, 590)   # 합성 라벨
    d = tmp_path / "data"; d.mkdir()
    sub.to_csv(d / "fidelity_base_v3.csv", index=False)
    os.symlink(PROC / "e5_soil_tdd_v3.csv", d / "e5_soil_tdd_v3.csv")
    out = tmp_path / "out"
    av = R.h40_argv(d, out, "Russia_W", 1, 1, 0, resume=False,
                    extra=["--n-grid", "0,3,all", "--draws", "1", "--seeds", "1", "--cb-iters", "15"])
    res = R.run_spec(av)
    assert not res.get("n_fail")
    u = json.loads((out / "shards" / "lgd__cpu__Russia_W__x__s1_unit.json").read_text())
    assert u["status"] == "ok" and u["cfg"]["subregion_map"] == "kmeans" and u["tag"] == "lgd" and u["threads"] == 1


# ================================================================ (l) same_as 수치 비교, 묶음별 추정, 스모크 실행 표
def test_l_same_table_numeric(tmp_path):
    a_, b_, c_, d_ = (tmp_path / f"{x}.csv" for x in "abcd")
    a_.write_text("loc_id,region,alt_cm\n1,X,10.0\n2,Y,20.000000000000004\n")
    b_.write_text("loc_id,region,alt_cm\n1,X,10.0\n2,Y,20.0\n")
    c_.write_text("loc_id,region,alt_cm\n1,X,10.0\n2,Z,20.0\n")
    d_.write_text("loc_id,region,alt_cm\n1,X,10.0\n2,Y,20.01\n")
    assert R.same_table(b_, b_) == (True, dict(basis="바이트 동일"))
    eq, info = R.same_table(a_, b_)
    assert eq and info["basis"].startswith("수치 동일") and info["diff_cols"][0]["col"] == "alt_cm"
    assert not R.same_table(b_, c_)[0] and not R.same_table(b_, d_)[0]


def test_l_bundle_estimates():
    man = dict(specs=dict(
        Tibet=dict(group="main", est_h_1proc=0.3, n_fit=100, check_ok=True),
        Tibet_L41b=dict(group="l41", est_h_1proc=0.3, n_fit=100, check_ok=True, same_as="Tibet"),
        NAtlantic_L41c=dict(group="l41", est_h_1proc=0.2, n_fit=80, check_ok=True),
        repro_Russia_W=dict(group="repro", est_h_1proc=0.05, n_fit=10, check_ok=True),
        v3local_Canada=dict(group="v3local", est_h_1proc=0.5, n_fit=200, check_ok=True)))
    b = R.bundle_estimates(man)
    assert b["main"]["n_tables"] == 2 and b["main"]["n_fit"] == 180 and b["main"]["est_h_1proc"] == 0.5 and b["main"]["est_h_workers2"] == 0.25
    assert b["optional"]["n_tables"] == 2 and b["optional"]["n_fit"] == 210 and b["same_as"] == ["Tibet_L41b"]


@realdata
def test_l_smoke_table_synthetic_targets_only(tables, tmp_path):
    s = tables.specs()["Russia_C"]
    d, rec = R.write_table(tables, s, tmp_path)
    info = R.smoke_table(tables, s, d, tmp_path / "smoke")
    run = pd.read_csv(d / "fidelity_base_v3.csv", low_memory=False)
    sm = pd.read_csv(tmp_path / "smoke" / "fidelity_base_v3.csv", low_memory=False)
    tgt = set(info["target_loc_ids"])
    m = run.loc_id.astype(int).isin(tgt).values
    assert info["n_target_rows_synthetic"] == int(m.sum()) == 57 and 17557 not in tgt
    assert np.allclose(run.loc[~m, "alt_cm"].values, sm.loc[~m, "alt_cm"].values, equal_nan=True)       # 대상 밖 행은 그대로다
    assert not np.allclose(run.loc[m, "alt_cm"].values, sm.loc[m, "alt_cm"].values)                     # 대상 행은 합성 라벨이다
    assert sm.loc[m, "alt_cm"].between(5, 590).all() and (tmp_path / "smoke" / "e5_soil_tdd_v3.csv").is_symlink()
    args = H.parse_args(R.h40_argv(tmp_path / "smoke", tmp_path / "o", "Russia_C", 1, 1, 0, resume=False, count=True))
    D = H.get_data(args)
    assert sorted(int(x) for x in D.df.loc_id.values[D.target_idx("Russia_C")]) == info["target_loc_ids"]


# ================================================================ (m) 약관 확인분 판(LG 개정 15 (m), 실행 계획 A6)
# 등록 표 정의의 고정값(A6 구현 전 HEAD 7f0ce14 의 h51 로 계산). 등록 표 39개의 정의 해시와 h40 설정 해시가 바뀌면 조각 이름·해시가 달라진다
REG_SPEC_DIGEST = "e962259d899461b63b2332f91e2928b879bf8b79e9b2879d72c56664286f142d"
REG_CFG_HASH = "8ca3e18fe401"


def _spec_digest(s):
    import hashlib
    d = dict(name=s["name"], target=s["target"], kind=s["kind"], keep=[int(x) for x in s["keep"]], suffix=s["suffix"], drop_v3=s["drop_v3"],
             alt=sorted((int(k), repr(float(v))) for k, v in (s["alt"] or {}).items()), roles=s["roles"], splits=s["splits"], base=s["base"],
             point_only=s["point_only"], el_name=s["el_name"], group=s["group"], tm_name=R.tm_name(s))
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def test_m_groups_and_names():
    """lic 는 선택 묶음이다(all 에 들지 않는다). 등록 묶음 목록은 그대로다. 파이프라인의 grep '"lic"' 이 맞는다."""
    reg = {k: v for k, v in R.GROUPS.items() if k != R.LIC}
    assert list(reg) == ["main", "l40", "l39", "l41", "l42", "repro", "v3local"] and R.DEFAULT_GROUPS == ("main", "l40", "l39", "l41", "l42")
    assert reg["main"] == ["Tibet", "NAtlantic", "Russia_C"] and len(reg["l41"]) == 24 and reg["l42"] == ["Tibet_L42", "NAtlantic_L42", "Russia_C_L42"]
    assert reg["l40"] == ["Russia_W_expanded", "Russia_E_expanded", "Canada_expanded", "Russia_E_expanded_noKytalyk"]
    assert R.GROUPS[R.LIC] == [f"{b}_lic" for b in R.LIC_BASES] and len(R.GROUPS[R.LIC]) == 14
    assert R.parse_args(["--specs", "lic"]).SPEC_NAMES == R.GROUPS[R.LIC]
    assert not any(n.endswith("_lic") for n in R.parse_args(["--specs", "all"]).SPEC_NAMES + R.parse_args([]).SPEC_NAMES)
    assert '"lic"' in (ROOT / "scripts" / "3_deep_learning" / "h51_lgd_run.py").read_text(encoding="utf-8")   # run_post_results.sh lgd-lic 의 점검
    assert R.tm_name(dict(target="NAtlantic", suffix="lic")) == "NAtlantic~lic|x"
    assert R.tm_name(dict(target="NAtlantic", suffix="L41c~lic")) == "NAtlantic~L41c~lic|x"
    assert R.LIC_M1 == {"NAtlantic": (19, 38), "Canada_expanded": (37, 75), "Russia_W_expanded": (7, 8)}


@realdata
def test_m_lic_specs_real(tables):
    """약관 확인분 표 정의(loc_id 집합과 약관 열만 쓴다. 라벨 값은 읽지 않는다): lic 셀만 정확히 빠지고, 나머지 정의는 등록 표와 같다.
    (m)1 의 셀 수, (m)4 의 야말 보조 행, 이름 충돌 없음, 등록 표 정의와 h40 설정 해시 불변."""
    S = tables.specs()
    lic_ids = tables.lic_ids
    reg = {n: s for n, s in S.items() if not s.get("lic_of")}
    lic = {n: s for n, s in S.items() if s.get("lic_of")}
    # 등록 표 정의·설정 해시 불변(조각 이름과 cfg_hash 가 같다)
    import hashlib
    dig = {n: _spec_digest(s) for n, s in reg.items()}
    assert len(reg) == 39 and hashlib.sha256(json.dumps(dig, sort_keys=True).encode()).hexdigest() == REG_SPEC_DIGEST
    for n, s in S.items():
        a_ = H.parse_args(R.h40_argv(Path("/nonexistent_dir"), Path("/nonexistent_out"), s["target"], s["splits"], R.THREADS, R.WORKERS_MAX))
        assert H.cfg_hash(H.unit_cfg(a_, "cpu", s["target"], "x", 1)) == REG_CFG_HASH, n
    # 자료로 센 목록 = 등록 목록 = lic_unverified 셀이 든 본 묶음 표
    has = sorted(n for n, s in reg.items() if s["group"] in R.DEFAULT_GROUPS and set(int(x) for x in s["keep"]) & lic_ids)
    assert tables.lic_set_ok and has == sorted(R.LIC_BASES) == sorted(tables.lic_tables)
    assert sorted(lic) == sorted(R.GROUPS[R.LIC]) == sorted(f"{n}_lic" for n in has)
    for n, s in lic.items():
        b = S[s["lic_of"]]
        kb = set(int(x) for x in b["keep"])
        assert set(int(x) for x in s["keep"]) == kb - lic_ids and not (set(int(x) for x in s["keep"]) & lic_ids), n
        assert s["lic_dropped"] == sorted(kb & lic_ids) and len(s["lic_dropped"]) > 0, n
        assert sum(s["lic_dropped_by_source"].values()) == len(s["lic_dropped"]), n
        for k in ("target", "kind", "drop_v3", "roles", "splits"):
            assert s[k] == b[k], (n, k)
        assert s["group"] == R.LIC and s["elig"] is None and s["el_name"] is None
        assert R.tm_name(s) == R.tm_name(b)[:-2] + "~lic|x", n
        if b["alt"] is not None:
            assert set(s["alt"]) == set(int(x) for x in s["keep"]), n
    assert lic["NAtlantic_L41c_lic"]["base"] == "NAtlantic_lic" and lic["NAtlantic_L42_lic"]["base"] == "NAtlantic_lic"
    assert lic["Tibet_L42_lic"]["base"] == "Tibet" and lic["Russia_C_L42_lic"]["base"] == "Russia_C" and lic["NAtlantic_lic"]["base"] is None
    # (m)1: 주 판정 풀의 대상 새 셀(약관 확인분 / 전체 판)
    for b, want in (("NAtlantic", (19, 38)), ("Canada_expanded", (37, 75)), ("Russia_W_expanded", (7, 8))):
        tl, tb = tables.target_ids(S[f"{b}_lic"]), tables.target_ids(S[b])
        assert (len([i for i in tl if i >= R.LOC0]), len([i for i in tb if i >= R.LOC0])) == want, b
    # (m)4: 야말 GGD402 보조 148셀은 약관 미확인이고 어느 표에도 없다
    yamal = set(int(x) for x in tables.new[tables.new.lgd_role == "aux_unknown_yamal"].loc_id)
    assert len(yamal) == 148 and yamal <= lic_ids
    assert all(not (set(int(x) for x in s["keep"]) & yamal) for s in S.values())
    # 이름: 표 이름·저장소 이름이 모두 다르고, 등록 표에는 '~lic' 가 없다
    tms = [R.tm_name(s) for s in S.values()]
    assert len(set(S)) == len(S) and len(set(tms)) == len(tms)
    assert not any("~lic" in R.tm_name(s) or n.endswith("_lic") for n, s in reg.items())
    assert all("~lic|x" in R.tm_name(s) for s in lic.values())


@realdata
def test_m_registered_tables_unchanged(tables, tmp_path):
    """등록 표의 실행 표 원문이 manifest 에 적힌 해시(본 실행 전 점검)와 같다(약관 확인분 판 구현이 등록 표를 바꾸지 않았다)."""
    man = json.loads((PROC / "lgd" / "lgd_run_manifest.json").read_text())
    S = tables.specs()
    n = 0
    for name, s in S.items():
        rec = man["specs"].get(name, {})
        if s.get("lic_of") or "table_sha256" not in rec:
            continue
        _, r = R.write_table(tables, s, tmp_path)
        assert r["table_sha256"] == rec["table_sha256"], name
        n += 1
    assert n >= 30


def _syn_v4(tmp_path, n_v3=4, lic=(20001, 20003)):
    """합성 v4 원문(라벨은 합성). v3 행 loc_id < 20000, 새 행 20000–20004."""
    rows = ["loc_id,region,alt_cm,source_id,e5_sqrt_tdd"]
    for i in range(n_v3):
        rows.append(f"{17000 + i},R{i},{40.0 + i},F4_direct,30.5")
    for j in range(5):
        rows.append(f"{20000 + j},NEW,{'' if j == 4 else repr(55.0 + j)},F4_ext_direct,31.25")
    p = tmp_path / "v4.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p, set(lic)


def test_m_lic_text_check_synthetic(tmp_path):
    """약관 확인분 표 = 등록 표에서 lic 줄만 뺀 표(원문 줄·순서 동일). 변조, 남은 lic 줄, 머리줄 차이는 불통과(합성 표)."""
    E = R.elig_mod()
    v4, lic = _syn_v4(tmp_path)
    keep = [20000, 20001, 20002, 20003, 20004]
    E.write_run_table_text(v4, tmp_path / "base.csv", keep, drop_v3=[17002])
    E.write_run_table_text(v4, tmp_path / "lic.csv", [k for k in keep if k not in lic], drop_v3=[17002])
    tgt_b, tgt_l = [17000, 17001, 17003] + keep, [17000, 17001, 17003, 20000, 20002, 20004]
    c = R.lic_text_check(tmp_path / "base.csv", tmp_path / "lic.csv", lic, tgt_b, tgt_l)
    assert c["ok"] and c["n_removed"] == 2 and c["removed_ids"] == [20001, 20003] and c["n_lic_left"] == 0
    assert c["n_lines_base"] - c["n_lines_lic"] == 2 and c["n_finite_target_base"] == 7 and c["n_finite_target_lic"] == 5   # 20004 는 결측
    # 변조: 남은 줄 하나의 값이 다르다
    t = (tmp_path / "lic.csv").read_text().replace("57.0", "57.5")
    (tmp_path / "bad1.csv").write_text(t)
    assert not R.lic_text_check(tmp_path / "base.csv", tmp_path / "bad1.csv", lic)["ok"]
    # lic 줄이 남았다
    E.write_run_table_text(v4, tmp_path / "bad2.csv", [20000, 20001, 20002, 20004], drop_v3=[17002])
    c2 = R.lic_text_check(tmp_path / "base.csv", tmp_path / "bad2.csv", lic)
    assert not c2["ok"] and c2["n_lic_left"] == 1
    # lic 가 아닌 줄까지 빠졌다
    E.write_run_table_text(v4, tmp_path / "bad3.csv", [20000, 20004], drop_v3=[17002])
    assert not R.lic_text_check(tmp_path / "base.csv", tmp_path / "bad3.csv", lic)["ok"]


def test_m_lic_eligibility_rule():
    """6B.4 적격 규칙(유효 분할 ≥ 1, 사용 분할의 채점 블록 합집합 ≥ 8, 소수 블록 < 5)과 레짐의 값 없는 상한."""
    E = R.elig_mod()
    assert E.MIN_BLOCKS_CI == R.MIN_BLOCKS_CI == 8 and float(E.DEEP_CM) == R.DEEP_CM == 150.0 and int(E.FEW_BLOCKS) == R.FEW_BLOCKS == 5
    rows = [dict(split=s, dup_of=-1, valid=True, nb_eval=v) for s, v in zip(range(1, 6), (2, 3, 4, 2, 3))]
    e = R.lic_eligibility(rows, 6)
    assert not e["eligible"] and e["n_valid_splits"] == 5 and e["min_nb_eval_used"] == 2 and e["few_blocks"]
    assert R.lic_eligibility(rows, 8)["eligible"]
    rows2 = [dict(split=1, dup_of=-1, valid=False, nb_eval=1), dict(split=2, dup_of=1, valid=False, nb_eval=6),
             dict(split=3, dup_of=-1, valid=False, nb_eval=1)]
    e2 = R.lic_eligibility(rows2, 12)
    assert not e2["eligible"] and e2["n_valid_splits"] == 0 and e2["n_unique_splits"] == 2 and e2["min_nb_eval_used"] == 1
    rows3 = [dict(split=1, dup_of=-1, valid=True, nb_eval=6), dict(split=2, dup_of=1, valid=True, nb_eval=1),
             dict(split=3, dup_of=-1, valid=False, nb_eval=1)]
    e3 = R.lic_eligibility(rows3, 9)
    assert e3["eligible"] and e3["min_nb_eval_used"] == 6 and not e3["few_blocks"] and e3["n_valid_splits"] == 1
    ok, ub, _ = R.regime_bound("shallow", 70.9, 41, 22)
    assert ok and abs(ub - 70.95 * 41 / 22) < 0.01
    assert not R.regime_bound("shallow", 100.0, 40, 20)[0] and not R.regime_bound("deep", 200.0, 10, 5)[0]
    assert not R.regime_bound("shallow", 70.9, 41, 0)[0]
    assert R.lic_role("new_region", False, "shallow").endswith("point_only") and "PE1+PE2" in R.lic_role("new_region", True, "shallow")
    assert "변형 불가" in R.lic_role("variant_L41c", False, "shallow")


@realdata
def test_m_plan_and_bundles(tables):
    """학습에서는 manifest 의 약관 확인분 적격으로 부적격 L41 변형을 뺀다. 점검에서는 모두 센다. 등록 표의 '변형 불가' 목록과 묶음 추정에
    lic 표가 섞이지 않는다."""
    man = dict(specs={"NAtlantic_L41c_lic": dict(eligible=False, run_skip="변형 불가(약관 확인분 판에서 부적격, LG 개정 15 (m)1)"),
                      "NAtlantic_lic": dict(eligible=False)})
    a = R.parse_args(["--specs", "lic"])
    _, run, skip = R.plan_specs(a, tables, man)
    assert [x["spec"] for x in skip] == ["NAtlantic_L41c_lic"] and "NAtlantic_lic" in [s["name"] for s in run] and len(run) == 13
    a2 = R.parse_args(["--specs", "lic", "--count-only"])
    _, run2, skip2 = R.plan_specs(a2, tables, man)
    assert not skip2 and len(run2) == 14
    allspec = tables.specs()
    assert not any(x["spec"].endswith("_lic") for x in R.ineligible_variants(allspec))
    b = R.bundle_estimates(dict(specs=dict(
        NAtlantic=dict(group="main", est_h_1proc=0.3, n_fit=100, check_ok=True),
        NAtlantic_lic=dict(group="lic", est_h_1proc=0.2, n_fit=60, check_ok=True),
        NAtlantic_L41b_lic=dict(group="lic", est_h_1proc=0.2, n_fit=60, check_ok=True, same_as="NAtlantic_lic"),
        NAtlantic_L41c_lic=dict(group="lic", est_h_1proc=0.1, n_fit=40, check_ok=True, run_skip="변형 불가"))))
    assert b["main"]["n_tables"] == 1 and b["lic"]["n_tables"] == 1 and b["lic"]["n_fit"] == 60


@realdata
def test_m_count_lic_synthetic_labels(tables, tmp_path):
    """--count-only 의 약관 확인분 점검 경로(count_spec + lic_summary)를 합성 라벨 v4 로 돈다. 실제 대상 라벨은 쓰지 않는다: v4 의 모든 줄의
    alt_cm 을 √TDD 기반 합성값으로 바꾼 표를 만든다(공변량만 읽는다). 셀 수·(m)1·원문 줄 대조·적격 열을 확인한다."""
    proc = tmp_path / "proc"; proc.mkdir()
    lines = (PROC / "fidelity_base_v4.csv").read_text(encoding="utf-8").splitlines()
    hdr = lines[0].split(","); ia, iq = hdr.index("alt_cm"), hdr.index("e5_sqrt_tdd")
    rng = np.random.RandomState(0)
    out = [lines[0]]
    for ln in lines[1:]:
        f = ln.split(",")
        try:
            q = float(f[iq])
        except ValueError:
            q = 30.0
        f[ia] = repr(float(np.clip(1.4 * (q if np.isfinite(q) else 30.0) * np.exp(0.2 * rng.randn()), 5, 590)))       # 합성 라벨
        out.append(",".join(f))
    (proc / "fidelity_base_v4.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    for f in ("fidelity_base_v4_labels.csv", "lgd_eligibility_v1.csv", "lgd_eligibility_v1_splits.csv", "e5_soil_tdd_v4.csv"):
        os.symlink(PROC / f, proc / f)
    T = R.Tables(proc)
    S = T.specs()
    for name, (n_new, n_full, n_v3) in (("NAtlantic_lic", (19, 38, 3)), ("Russia_W_expanded_lic", (7, 8, 31))):
        s = S[name]
        d, rec = R.write_table(T, s, tmp_path / "o")
        rows, summ = R.count_spec(T, s, d, tmp_path / "o", 1)
        summ.update(R.lic_summary(T, s, S, d, rows, summ, tmp_path / "o"))
        assert summ["ok_cells"] and summ["n_split_mismatch"] == 0 and summ["n_cells"] == n_new + n_v3, name
        assert (summ["n_new_target"], summ["n_new_target_full"], summ["n_v3_target"]) == (n_new, n_full, n_v3) and summ["ok_m1"], name
        assert summ["text_check"]["ok"] and summ["text_check"]["removed_equals_dropped"] and summ["text_check"]["n_lic_left"] == 0, name
        assert summ["n_lic_dropped"] == n_full - n_new and summ["lic_ok"], name
        assert isinstance(summ["eligible"], bool) and summ["n_valid_splits"] >= 0 and summ["regime"] in ("shallow", "deep"), name
        assert summ["eligible"] == bool(summ["n_valid_splits"] >= 1 and summ["nb_union"] >= 8), name
        lel = R.lic_eligibility_table([dict(summ, spec=name)], S)
        assert list(lel.columns) == R.LIC_EL_COLS and lel.name.iloc[0] == R.tm_name(s) and lel.lic_of.iloc[0] == s["lic_of"]


# ---------------- h52: 두 판 병기(합성 조각)
LIC_SRC = {"calm_web_subsites": 2, "cusp_v1_1": 1}


def _add_lic(lgd, nat_eligible=True, shards=True):
    """합성 세계에 약관 확인분 표(조각, manifest 기록, lgd_eligibility_lic.csv)를 더한다. NAtlantic_lic 의 적격을 nat_eligible 로 둔다."""
    new_keys = _keys((0, 3, 10, -1)); new_keys.update({K("P2", 10): 5.5, K("P3", 10): 5.2})
    p4_keys = _keys((0, 3, 10, 40, -1))
    if shards:
        _write_shards(lgd / "NAtlantic_lic", "lgd", "NAtlantic", (1, 2), 9, new_keys, 80, 12)
        _write_shards(lgd / "NAtlantic_L41c_lic", "lgd", "NAtlantic", (1, 2), 9, new_keys, 81, 10)
        _write_shards(lgd / "Canada_expanded_lic", "lgd", "Canada", (1, 2), 10, p4_keys, 82, 380)
    man = json.loads((lgd / "lgd_run_manifest.json").read_text())
    skip = "" if nat_eligible else "변형 불가(약관 확인분 판에서 부적격, LG 개정 15 (m)1)"
    rows = [("NAtlantic_lic", "NAtlantic", "NAtlantic", "new_region", "NAtlantic~lic|x", nat_eligible, 3.0, "", None, LIC_SRC),
            ("NAtlantic_L41a_lic", "NAtlantic_L41a", "NAtlantic", "variant_L41a", "NAtlantic~L41a~lic|x", False, 2.0,
             "변형 불가(약관 확인분 판에서 부적격, LG 개정 15 (m)1)", None, {"calm_web_subsites": 2}),
            ("NAtlantic_L41b_lic", "NAtlantic_L41b", "NAtlantic", "variant_L41b", "NAtlantic~L41b~lic|x", nat_eligible, 3.0, skip, "NAtlantic_lic", LIC_SRC),
            ("NAtlantic_L41c_lic", "NAtlantic_L41c", "NAtlantic", "variant_L41c", "NAtlantic~L41c~lic|x", nat_eligible, 3.0, skip, None, LIC_SRC),
            ("Canada_expanded_lic", "Canada_expanded", "Canada", "expanded_L40", "Canada~exp~lic|x", True, 9.0, "", None,
             {"cusp_v1_1": 1, "nsidc_ggd353_thawtube": 3})]
    le = []
    for sp, of, t, kind, nm, el_, mn, rs, same, src in rows:
        po = bool((kind.startswith("new_region") or kind.startswith("variant_L41")) and not el_)
        man["specs"][sp] = dict(target=t, tm_name=nm, group="lic", lic_of=of, point_only=po, eligible=el_, check_ok=True, n_units=2,
                                **({"same_as": same} if same else {}), **({"run_skip": rs} if rs else {}))
        le.append(dict(spec=sp, lic_of=of, target=t, kind=kind, name=nm, eligible=el_, regime="shallow", min_nb_eval_used=mn, point_only=po,
                       run_skip=rs, lic_dropped_sources=json.dumps(src)))
    (lgd / "lgd_run_manifest.json").write_text(json.dumps(man))
    pd.DataFrame(le).to_csv(lgd / "lgd_eligibility_lic.csv", index=False)


FAST = ["--allow-no-repro-gate", "--nboot-h40", "100", "--nboot", "100", "--l40-draws", "50"]


def test_m_h52_permission_rule(tmp_path):
    """(m)2: 약관 확인분 판이 뺀 셀의 자료원이 모두 granted 일 때만 주 판정이 전체 판."""
    req = ["calm_web_subsites", "cusp_v1_1", "nsidc_ggd353_thawtube"]
    p = tmp_path / "perm.json"
    a = A.parse_args(["--lic-permission", str(p)])
    assert A.lic_main_version(a, req)[0] == "lic"                                                     # 기록 없음
    p.write_text(json.dumps(dict(sources={k: dict(granted=True, date="2026-10-01") for k in req})))
    v, why, rec = A.lic_main_version(a, req)
    assert v == "full" and rec["missing"] == [] and rec["granted"] == req
    p.write_text(json.dumps(dict(sources={k: dict(granted=(k != "cusp_v1_1")) for k in req})))
    v, why, rec = A.lic_main_version(a, req)
    assert v == "lic" and rec["missing"] == ["cusp_v1_1"] and "cusp_v1_1" in why
    p.write_text(json.dumps(dict(sources={k: dict(granted="yes") for k in req})))                     # 참(true)만 허락으로 본다
    assert A.lic_main_version(a, req)[0] == "lic"
    p.write_text("{깨진 JSON")
    assert A.lic_main_version(a, req)[0] == "lic"
    assert A.lic_main_version(a, [])[0] == "lic"                                                      # 필요한 자료원 목록이 없으면 판단하지 않는다
    le = pd.DataFrame(dict(lic_dropped_sources=[json.dumps({"a": 1}), json.dumps({"b": 2, "a": 1}), np.nan]))
    assert A.lic_required_sources(le) == ["a", "b"] and A.lic_required_sources(None) == []


def test_m_h52_two_versions_synthetic(tmp_path, monkeypatch):
    """약관 확인분 판(적격 NAtlantic~lic): 판별 풀, 판 열, same_as·공통 난수, L40 확충판 이름, 두 판 병기 표. 전체 판은 약관 확인분 판의
    유무와 무관하게 같다. 허락 기록이 모두 있으면 주 판정 판이 전체 판으로 바뀐다."""
    monkeypatch.setenv("LGD_TEST", "1")
    lg, lgd, elp, xj, sdp = _synthetic_world(tmp_path)
    r0 = A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp, FAST + ["--out-dir", str(tmp_path / "o0")]))
    assert "lic" not in r0 and r0["meta"]["lic"]["status"].startswith("계산하지 않음") and r0["main_version"] == "lic"
    assert not (tmp_path / "o0" / "lgd_tests_lic.csv").exists() and (tmp_path / "o0" / "lgd_versions.csv").exists()
    assert (r0["tests"].main_verdict == False).all() and (r0["tests"].version_role == A.ROLE_FULL_SI).all()           # noqa: E712
    _add_lic(lgd, nat_eligible=True)
    r1 = A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp, FAST + ["--out-dir", str(tmp_path / "o1")]))
    O1 = tmp_path / "o1"
    for k in A.LIC_OUT:
        assert (O1 / f"lgd_{k}.csv").exists() and (O1 / f"lgd_{k}_lic.csv").exists(), k
    # 전체 판은 약관 확인분 판이 있어도 같다
    for k in A.LIC_OUT:
        assert (O1 / f"lgd_{k}.csv").read_text() == (tmp_path / "o0" / f"lgd_{k}.csv").read_text(), k
    m = r1["meta"]
    assert m["pools"]["PE1"] == A.P4 + ["NAtlantic|x", "Russia_C|x"] and m["lic"]["status"] == "완료"
    assert m["lic"]["pools"]["PE1"] == A.P4 + ["NAtlantic~lic|x", "Russia_C|x"] and m["lic"]["pools"]["PE2"][-1] == "Tibet_LGD|x"
    assert m["lic"]["required_sources"] == ["calm_web_subsites", "cusp_v1_1", "nsidc_ggd353_thawtube"] and m["lic"]["main_version"] == "lic"
    assert m["lic"]["spec_map"]["NAtlantic"] == "NAtlantic_lic" and m["lic"]["spec_map"]["Canada_expanded"] == "Canada_expanded_lic"
    tl = r1["lic"]["tests"]
    assert (tl.lic_version == "lic").all() and tl.main_verdict.all() and (tl.version_role == A.ROLE_MAIN).all()
    assert (r1["tests"].lic_version == "full").all() and not r1["tests"].main_verdict.any()
    assert r1["lic"]["crn"]["NAtlantic_L41c"] == "seed ← NAtlantic~lic|x" and r1["lic"]["crn"]["NAtlantic_L41b"].startswith("same_as → NAtlantic~lic|x")
    v = tl[tl.scope == "verdict"]
    l41 = v[(v.test_id == "L41") & v.target.astype(str).str.startswith("NAtlantic")].set_index("variant")
    assert l41.loc["(a)"].verdict.startswith("변형 불가") and l41.loc["(b)"].verdict == "강건" and l41.loc["(c)"].target == "NAtlantic~L41c~lic|x"
    assert "Canada~exp~lic|x" in set(v[v.test_id == "L40"].target) and "Russia_W~exp|x" in set(v[v.test_id == "L40"].target)
    l38r = tl[(tl.test_id == "L38") & (tl.scope == "region")]
    assert "NAtlantic~lic|x" in set(l38r.target) and "NAtlantic|x" not in set(l38r.target)
    cl = r1["lic"]["curve"]
    assert "NAtlantic~lic" in set(cl.target) and "NAtlantic_lic" in set(cl.spec)
    # 두 판 병기 표: P4 행은 두 판이 같고, 주 판정 판의 판정을 verdict_main 에 싣는다
    vt = pd.read_csv(O1 / "lgd_versions.csv")
    p4 = vt[(vt.test_id == "L1e") & (vt.pool == "P4") & (vt.scope == "verdict")]
    assert len(p4) == 1 and p4.verdict_full.iloc[0] == p4.verdict_lic.iloc[0] and not p4.differs.iloc[0]
    assert (vt.verdict_main.astype(str) == vt.verdict_lic.astype(str)).all() and (vt.main_version == "lic").all()
    assert {"L1e", "L4e", "L8e", "L38", "L39", "L40", "L41"} <= set(vt.test_id)
    assert not vt.target.astype(str).str.contains("~lic").any()
    # 허락 기록이 모두 있으면 주 판정 판은 전체 판
    (lgd / "lgd_license_permission.json").write_text(json.dumps(dict(sources={k: dict(granted=True) for k in m["lic"]["required_sources"]})))
    r2 = A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp, FAST + ["--out-dir", str(tmp_path / "o2")]))
    assert r2["main_version"] == "full" and r2["tests"].main_verdict.all() and (r2["tests"].version_role == A.ROLE_MAIN).all()
    assert not r2["lic"]["tests"].main_verdict.any() and (r2["lic"]["tests"].version_role == A.ROLE_LIC_AUX).all()
    vt2 = pd.read_csv(tmp_path / "o2" / "lgd_versions.csv")
    assert (vt2.verdict_main.astype(str) == vt2.verdict_full.astype(str)).all()


def test_m_h52_ineligible_lic_and_missing(tmp_path, monkeypatch):
    """(m)1: 약관 확인분 판에서 NAtlantic 이 부적격이면 그 판의 PE1 에서 빠지고(점 추정), PE1 은 남은 새 지역으로 계산한다. 전체 판의 풀은
    그대로다. 조각이 없는 약관 확인분 표가 있으면 '미완'."""
    monkeypatch.setenv("LGD_TEST", "1")
    lg, lgd, elp, xj, sdp = _synthetic_world(tmp_path)
    _add_lic(lgd, nat_eligible=False)
    r = A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp, FAST))
    m = r["meta"]
    assert m["pools"]["PE1"] == A.P4 + ["NAtlantic|x", "Russia_C|x"]
    assert m["lic"]["pools"]["PE1"] == A.P4 + ["Russia_C|x"] and m["lic"]["pools"]["ineligible"] == ["NAtlantic~lic"]
    assert "NAtlantic~lic" in m["lic"]["main_point"] and "NAtlantic~lic" not in m["main_point"]["overridden"]
    tl = r["lic"]["tests"]
    v = tl[tl.scope == "verdict"]
    nat = v[v.test_id.isin(["L41", "L42"]) & v.target.astype(str).str.startswith("NAtlantic")]
    assert len(nat) == 9 and nat.verdict.str.startswith("행 없음(주 설정이 이 판의 적격 표에서 부적격").all()
    assert "NAtlantic~lic|x" not in set(tl[(tl.test_id == "L38") & (tl.scope == "region")].target)
    pl = r["lic"]["pool"].set_index("pool")
    assert pl.loc["PE1", "n_regions"] == 5 and pl.loc["point_only", "regions"] == "NAtlantic~lic"
    cl = r["lic"]["curve"]
    assert "NAtlantic~lic" in set(cl.target)                                                         # 점 추정은 곡선에 남는다
    for hyp in ("L1e", "L8e"):
        pe1 = v[(v.test_id == hyp) & (v.pool == "PE1")]
        assert len(pe1) == 1 and pe1.n_regions.iloc[0] == 5, hyp
    # 조각이 없는 약관 확인분 표
    import shutil
    shutil.rmtree(lgd / "Canada_expanded_lic")
    r2 = A.main(_agg_args(tmp_path, lg, lgd, elp, xj, sdp, FAST + ["--out-dir", str(tmp_path / "o3")]))
    assert r2["meta"]["lic"]["status"].startswith("미완") and r2["meta"]["lic"]["missing_shards"] == ["Canada_expanded_lic"]
    assert r2["lic"]["tests"].version_role.str.contains("미완").all()
    l40 = r2["lic"]["tests"]
    assert l40[(l40.test_id == "L40") & (l40.scope == "verdict") & (l40.target == "Canada~exp~lic|x")].verdict.str.startswith("행 없음").all()
