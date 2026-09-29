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
        if s["kind"] in ("repro", "v3local", "source_L42"):
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
