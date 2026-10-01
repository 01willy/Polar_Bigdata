"""scripts/3_deep_learning/h48_nn_tuning.py 단위 시험(LGF 의 N 부분, 계획 docs/EXPERIMENT_PLAN_LGF_2026-09-29.md 4절).

가벼운 시험은 실자료와 GPU 를 쓰지 않는다. 기본판 적합(h40.Fitter._fit)과 조정판 적합(fit_tuned)은 대체 함수(학습 목표 평균 + 입력 1열의 선형 항)로
바꿔 학습 행렬, 키, 저장 경로, 선택 규칙만 확인한다. 실제 신경망·CatBoost 적합 시험은 환경 변수 LG_RUN_HEAVY=1 일 때만(CPU) 돈다.

(a) 설정 목록: 결정성, 시도별 seed 라서 k = 16 목록이 k = 32 목록의 앞부분, 값의 범위, FT-T 시도 1, RealMLP 키.
(b) QuantileTransformer 재현(같은 입력과 적합 식별자이면 같은 변환, 식별자가 다르면 random_state 가 다르다).
(c) 30셀 fold 규칙과 fold 표(학습 행 수, 제외 지역 100 km 안의 학습 행 수).
(d) 기본값 우선 규칙(합성 d_f): c* 채택, sel_default_pref, fold 승수 규칙, no_cv, all_failed, 동률의 매개변수 수, default_failed.
(e) 민감도 선택(셀 가중 점수, λ 0.25 fold 값).
(f) 통과 0 공유 적합의 분할별 예측이 분할별 단독 적합(h40.run_ctx n = 0)의 채점 셀과 같다(블록 SSE 일치).
(g) 통과 0 의 sel_default 복사(시도 0 인 조정판과 민감도 판은 기본판 적합을 쓴다. 같은 시도는 한 번만 적합).
(h) 통과 1: 기본판 학습 행렬(h40.TRACE)과 조정판 학습 행렬이 같고(trace_check), 시도 0 인 종류는 기본판 SSE 를 복사한다.
(i) 집계: k_default, N2 의 강건 문장 규칙(세 경우), N2s 분류 행, N1 의 지지 두 갈래와 Holm 묶음(conf 4, aux).
(j) eq_verdict 의 갈래(λ 조합 규칙, 정밀도 미달)와 support_class, finish_tests 의 열.
(k) catboost_tuned_loc 선택 표가 h42.pick_tuned 와 같은 선택을 한다(합성 행).
(l) TabM 묶음 헤드와 루프 헤드의 출력 차 ≤ 1e-6, MLPp·TabMp·FTTp 의 출력 모양, fit_tuned(CPU, 2 epoch) 의 예측 모양.
(m) 창 파일: t0 는 한 번만 쓴다, 마감 판정, 본 실행의 E1·reduce 대조. 드레인(종료 코드 3)과 종료 코드 순서.
(n) GPU 거부(8, 허용 표지 없는 0·1, 허용 밖, 빈 --gpus, --allow-local 없음, --cpu-only 와 --gpus), h43·h40 SHA 불일치 거부와 '개정 필요' 표지.
(o) 작업 계획: 1단계가 끝나면 2단계 단위가 생기고, 2단계가 끝나면 선택 표에 쓰며, 그 뒤 통과 0 이 배정 가능해진다.
(p) 선택 단위(run_sel_unit, 대체 학습기): JSON 의 fold 점수, unit_sig, 재개 판정.
(q) 조각 기록과 재개 판정(설정 해시, failed, --rerun-partial).
무거운 시험(LG_RUN_HEAVY=1, CPU)
(H1) 실제 mlp 기본판으로 (f)를 다시 확인한다. (H2) fit_tuned 의 RealMLP 경로. (H3) cb_select_rows 와 h42.pick_tuned(작은 격자).
실행: .venv_lgf/bin/python -m pytest -q tests/test_h48_nn.py              (가벼운 시험, 스레드 2개)
      LG_RUN_HEAVY=1 .venv_lgf/bin/python -m pytest -q tests/test_h48_nn.py -k H
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"                                      # numpy 를 부르기 전에 둔다
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # 시험은 GPU 를 쓰지 않는다
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HEAVY = os.environ.get("LG_RUN_HEAVY", "") == "1"
heavy = pytest.mark.skipif(not HEAVY, reason="실제 적합 시험은 LG_RUN_HEAVY=1 일 때만 실행한다")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


N = _load("h48_nn_tuning", "scripts/3_deep_learning/h48_nn_tuning.py")
H = N.H
T43 = N.T43
DF = len(H.FEATS)


def args(extra=(), tmp=None):
    out = ["--out-dir", str(tmp)] if tmp is not None else []
    a = N.parse_args(list(extra) + out)
    return a


def with_cfg(a):
    N.ensure_configs(a)
    return a


# ---------------------------------------------------------------- 합성 자료
def make_src(seed=0, sizes=(("r1", 200), ("r2", 120), ("r3", 40), ("r4", 12))):
    rng = np.random.RandomState(seed)
    mac = np.concatenate([[nm] * k for nm, k in sizes])
    n = len(mac)
    X = rng.randn(n, DF).astype(np.float32)
    X[5, 3] = np.nan
    s = rng.uniform(20, 40, n)
    y = 1.6 * s + 3 * np.nan_to_num(X[:, 1]) + rng.randn(n) * 2
    lat = np.concatenate([np.full(k, 60.0 + 3 * i) + rng.rand(k) * 0.5 for i, (_, k) in enumerate(sizes)])
    lon = np.concatenate([np.full(k, -150.0 + 2 * i) + rng.rand(k) * 0.5 for i, (_, k) in enumerate(sizes)])
    return X, y, s, mac, lat, lon


def make_ctx(split, seed=0, NA=60, NB=50):
    """같은 원천(분할과 무관)에 분할별 대상 A·B 를 붙인 합성 h40.Ctx. 채점 블록 10개."""
    Xs, ys, ss, mac, _, _ = make_src(0)
    rng = np.random.RandomState(100 + int(split))
    XA = rng.randn(NA, DF).astype(np.float32); XA[3, 2] = np.nan
    sA = rng.uniform(20, 40, NA); yA = 1.3 * sA + 3 * np.nan_to_num(XA[:, 1]) + rng.randn(NA) * 2
    XB = rng.randn(NB, DF).astype(np.float32)
    sB = rng.uniform(20, 40, NB); yB = 1.3 * sB + 3 * XB[:, 1] + rng.randn(NB) * 2
    return H.Ctx("Lena", "x", int(split), "Lena", Xs, ys, ss, mac, XA, yA, sA, np.repeat(np.arange(6), NA // 6), XB, yB, sB,
                 np.repeat(np.arange(10), NB // 10))


def stub_fit(self, learner, axis, build, seed, preds, info, init, iters, cont):
    """h40.Fitter._fit 대체: 학습 목표 평균 + 0.01·x1 + 0.001·seed."""
    Xtr, ytr, w = build()
    self._trace(dict(info or {}, learner=learner, axis=axis, seed=seed), Xtr, ytr, w)
    y = np.asarray(ytr, float)
    return None, [float(y.mean()) + 0.01 * np.nan_to_num(np.asarray(p, float)[:, 1]) + 0.001 * int(seed) for p in preds]


CALLS = []


def stub_tuned(learner, entry, Xtr, ytr, preds, seed, fit_id, threads=2, max_epochs=300, patience=16, realmlp_epochs=256, dev=None):
    CALLS.append(dict(learner=learner, trial=int(entry["trial"]), kind=entry["kind"], fit_id=fit_id, seed=int(seed), Xtr=np.array(Xtr, copy=True),
                      ytr=np.array(ytr, copy=True)))
    y = np.asarray(ytr, float)
    return [float(y.mean()) + 0.02 * np.nan_to_num(np.asarray(p, float)[:, 1]) + 0.01 * int(entry["trial"]) for p in preds], 7, 1234


@pytest.fixture()
def stubs(monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(H.Fitter, "_fit", stub_fit)
    monkeypatch.setattr(N, "fit_tuned", stub_tuned)
    return True


class FakeD:
    """분할 구조만 있는 자료 대역(h40.enumerate_units, h42.make_tm 이 부른다)."""

    def __init__(self, splits=(1, 2, 3, 4, 5)):
        self.sp = tuple(splits)

    def split_structure(self, t):
        return {sp: dict(dup_of=-1, mirror_of=-1, nb_eval=10, n_eval=50, n_A=60, nb_A=6, valid=True, n_unique_splits=len(self.sp),
                         n_valid_splits=len(self.sp)) for sp in self.sp}


def inject_src(a, D, target="Lena", mode="x", seed=0, sizes=(("r1", 200), ("r2", 120), ("r3", 40), ("r4", 12))):
    X, y, s, mac, lat, lon = make_src(seed, sizes)
    regs = sorted(r for r, k in pd.Series(mac).value_counts().items() if k >= int(a.min_region_cells))
    si = SimpleNamespace(target=target, mode=mode, src=np.arange(len(y)), X=X, y=y, s=s, mac=mac.astype(str), lat=lat, lon=lon, regions=regs,
                         src_hash=f"h{seed}", folds=None)
    N._SRC[(id(D), target, mode, int(a.min_region_cells))] = si
    return si


# ================================================================ (a) 설정 목록
def test_a_configs_deterministic_and_prefix():
    for lr in ("mlp", "tabm", "ftt"):
        for kind in N.KINDS:
            for t in (1, 2, 5, 17, 32):
                assert N.sample_config(lr, kind, t) == N.sample_config(lr, kind, t)
    assert N.n_trials_list("mlp", 16) == N.n_trials_list("mlp", 32)[:17]
    assert N.n_trials_list("ftt", 16) == N.n_trials_list("ftt", 32)[:18]
    cf = N.build_configs(32)
    assert len(cf["mlp"]["D"]) == 33 and len(cf["ftt"]["R"]) == 34 and len(cf["realmlp"]["D"]) == 33
    assert cf["mlp"]["D"][0]["params"] == dict(default=True)
    assert cf["ftt"]["D"][1]["params"] == N.FTT_PAPER and cf["ftt"]["R"][1]["params"] == N.FTT_PAPER
    assert cf["mlp"]["D"][5]["params"] != cf["mlp"]["R"][5]["params"], "종류마다 seed 가 다르다"
    assert N.configs_sig(cf) == N.configs_sig(N.build_configs(32))
    for e in cf["mlp"]["D"][1:]:
        p = e["params"]
        assert 1 <= p["n_layers"] <= 6 and 64 <= p["width"] <= 1024 and p["width"] % 16 == 0 and p["batch"] in (256, 512, 1024, 2048)
        assert 1e-4 <= p["lr"] <= 1e-2 and (p["wd"] == 0.0 or 1e-6 <= p["wd"] <= 1e-3) and 0.0 <= p["dropout"] <= 0.5
    for e in cf["tabm"]["R"][1:]:
        p = e["params"]
        assert 1 <= p["n_blocks"] <= 4 and 64 <= p["width"] <= 1024 and (p["width"] - 64) % 16 == 0 and p["k"] in (8, 16, 32)
    for e in cf["ftt"]["D"][2:]:
        p = e["params"]
        assert p["d_token"] in (64, 96, 128, 160, 192, 256) and p["d_token"] % 8 == 0 and 1 <= p["blocks"] <= 4 and p["batch"] in (256, 512, 1024)
        assert round(2 / 3 * p["d_token"]) - 1 <= p["ffn"] <= round(8 / 3 * p["d_token"]) + 1
    for e in cf["realmlp"]["D"][1:]:
        assert set(e["params"]) == set(N.RMLP_KEYS)
        assert all(type(v) in (str, bool, float, int, list) for v in e["params"].values()), "numpy 자료형이 남아 있다"


# ================================================================ (b) 분위 변환 재현
def test_b_quantile_transform_repro():
    assert N.qt_repro() == 0.0
    assert N.qt_seed("mlp", "D", 3, "sel|Lena|x|r1", 0) != N.qt_seed("mlp", "D", 3, "sel|Lena|x|r2", 0)
    X = np.random.RandomState(0).randn(1500, DF)
    f, info = N.make_transform(dict(transform="qnorm"), X, "mlp", "D", 1, "x", 0)
    assert info["n_quantiles"] == 1000 and np.all(np.isfinite(f(X)))
    f2, info2 = N.make_transform(dict(transform="std"), X, "mlp", "D", 1, "x", 0)
    assert np.allclose(f2(X), H._prep_apply(X, H._prep_stats(X)))


# ================================================================ (c) 30셀 fold 규칙
def test_c_fold_rule_30_cells(tmp_path):
    a = args((), tmp_path)
    D = FakeD()
    si = inject_src(a, D)
    assert si.regions == ["r1", "r2", "r3"], "30셀 미만 지역(r4)은 fold 가 아니다(항상 학습 쪽)"
    fr = N.fold_rows(si)
    assert [f["region"] for f in fr] == ["r1", "r2", "r3"]
    assert all(f["n_te"] + f["n_tr"] == len(si.y) for f in fr)
    assert fr[2]["n_te"] == 40 and fr[2]["n_tr"] == 332
    m = N.near_mask(np.array([60.0, 70.0]), np.array([-150.0, -150.0]), np.array([60.5]), np.array([-150.0]))
    assert m.tolist() == [True, False], "0.5° 위도(약 56 km)는 100 km 안, 10° 는 밖"


# ================================================================ (d) 기본값 우선 규칙
def test_d_choose_default_preference():
    npar = {0: 100, 1: 50, 2: 80}
    S = {0: np.array([10.0, 10.0, 10.0, 10.0]), 1: np.array([8.0, 8.5, 9.0, 8.0]), 2: np.array([12.0, 12.0, 12.0, 12.0])}
    r = N.choose(S, npar)
    assert r["trial"] == 1 and r["flag"] == "" and r["wins"] == 4 and r["need"] == 3
    S = {0: np.array([10.0, 10.0, 10.0, 10.0]), 1: np.array([9.9, 10.1, 9.8, 10.15])}
    r = N.choose(S, {0: 1, 1: 1})                                         # d̄ = −0.0125, se ≈ 0.08, 승 2 < 3
    assert r["cstar"] == 1 and r["trial"] == 0 and r["flag"] == "sel_default_pref"
    S = {0: np.array([10.0, 10.0, 10.0, 10.0]), 1: np.array([9.99, 9.99, 9.99, 10.5])}
    r = N.choose(S, {0: 1, 1: 1})                                         # 주 점수 최소가 시도 0 이면 c* = 0(표지 없음)
    assert r["cstar"] == 0 and r["trial"] == 0 and r["flag"] == ""
    S = {0: np.array([10.0, 10.0, 10.0, 10.0]), 1: np.array([9.5, 9.5, 9.5, 11.5])}
    r = N.choose(S, {0: 100, 1: 1})                                       # 동률(d̄ = 0): 매개변수 수로 c* = 1 이지만 d̄ < 0 이 아니므로 승 3 과 무관하게 시도 0
    assert r["cstar"] == 1 and r["wins"] == 3 and r["trial"] == 0 and r["flag"] == "sel_default_pref"
    S = {0: np.array([10.0, 10.0, 10.0, 10.0]), 1: np.array([9.9, 9.9, 9.9, 10.2])}
    r = N.choose(S, {0: 1, 1: 1})                                         # d̄ = −0.025 < 0, −d̄ < se(0.075) 이지만 승 3 ≥ 3
    assert r["trial"] == 1 and r["wins"] == 3
    assert N.choose({0: np.array([1.0]), 1: np.array([0.5])}, {})["flag"] == "no_cv"
    r = N.choose({0: np.array([np.inf, 1.0]), 1: np.array([np.inf, np.inf])}, {})
    assert r["trial"] == 0 and r["flag"] == "all_failed"
    r = N.choose({0: np.array([10.0, 10.0]), 1: np.array([8.0, 8.0]), 2: np.array([8.0, 8.0])}, {0: 5, 1: 300, 2: 200})
    assert r["cstar"] == 2, "동률이면 매개변수 수가 작은 설정"
    r = N.choose({0: np.array([np.inf, 10.0]), 1: np.array([8.0, 8.0])}, {})
    assert r["trial"] == 1 and r["flag"] == "default_failed"


# ================================================================ (e) 민감도 선택
def test_e_sensitivity_choice(tmp_path):
    a = args(("--learners", "mlp"), tmp_path)

    def rec(tr, sd, r_main, r25, sse, n=(100, 50, 30, 30)):
        return dict(trial=tr, seed=sd, n_params=10 + tr, folds=[dict(rmse=r1, rmse_l25=r2, sse=s_, n_te=nn) for r1, r2, s_, nn in
                                                                 zip(r_main, r25, sse, n)])
    recs = {}
    for sd in (1, 2):
        recs[(0, sd)] = rec(0, sd, [10, 10, 10, 10], [9, 9, 9, 9], [100 * 100, 100 * 50, 100 * 30, 100 * 30])
        recs[(1, sd)] = rec(1, sd, [9, 9, 9, 9], [9.5, 9.5, 9.5, 9.5], [100 * 100 * 2, 100 * 50, 81 * 30, 81 * 30])
    d = N.decide_group(a, "mlp", "R", recs, [0, 1])
    assert d["main"]["trial"] == 1, "주 점수: 시도 1 이 모든 fold 에서 낫다"
    assert d["cw"]["trial"] == 0, "셀 가중 점수: 큰 fold 의 SSE 가 커서 시도 0"
    assert d["l25"]["trial"] == 0, "λ 0.25 fold 값: 시도 0 이 낫다"
    assert N.decide_group(a, "mlp", "D", recs, [0, 1])["l25"] is None


# ================================================================ (f) 통과 0 공유 적합 = 분할별 단독 적합
def dec_all(trial_D=0, trial_R=0, cw=None, l25=None):
    out = {}
    for kind, tr in (("D", trial_D), ("R", trial_R)):
        out[kind] = dict(main=dict(trial=tr, flag=""), cw=dict(trial=tr if cw is None else cw, flag=""))
        if kind == "R":
            out[kind]["l25"] = dict(trial=tr if l25 is None else l25, flag="")
    return out


def test_f_p0_shared_equals_per_split(tmp_path, stubs):
    a = with_cfg(args(("--learners", "mlp"), tmp_path))
    cs = [make_ctx(sp) for sp in (1, 2, 3)]
    books, fits, h = N.p0_books(a, "mlp", "Lena", "x", cs, dec_all(0, 0))
    assert len(fits) == 4, "기본판만: 종류 2 × seed 2"
    for c, b_ in zip(cs, books):
        _, st_ref, _ = H.run_ctx(c, "gpu", N.ha_n(a, "mlp", "0"), learner_axis=True, learners=["mlp"])
        keys = [k for k in st_ref.keys if k[1] == "mlp"]
        assert keys and all(k in b_.st for k in keys)
        for k in keys:
            assert np.array_equal(st_ref.get(k)[0], b_.st.get(k)[0]) and np.array_equal(st_ref.get(k)[1], b_.st.get(k)[1])
        assert H.P0_KEY in b_.st and ("P1", "none", "1", "cell", 0, 0, -1, 0.0) in b_.st
    bad = make_ctx(2); bad.y_src = bad.y_src + 1.0
    with pytest.raises(RuntimeError, match="원천 배열"):
        N.p0_books(a, "mlp", "Lena", "x", [cs[0], bad], dec_all(0, 0))


# ================================================================ (g) 통과 0 의 sel_default 복사
def test_g_p0_sel_default_copy(tmp_path, stubs):
    a = with_cfg(args(("--learners", "mlp"), tmp_path))
    cs = [make_ctx(sp) for sp in (1, 2)]
    books, fits, _ = N.p0_books(a, "mlp", "Lena", "x", cs, dec_all(trial_D=0, trial_R=3, cw=0, l25=3))
    assert sorted(fits) == sorted([(k, s, t) for k in ("D", "R") for s in (0, 1) for t in ((0,) if k == "D" else (0, 3))])
    assert len(CALLS) == 2 and {c_["trial"] for c_ in CALLS} == {3}
    st = books[0].st
    for seed in (0, 1):
        kd, kt = ("D0", "mlp", "1", "cell", 0, 0, seed, 1.0), ("D0", "mlp_tuned", "1", "cell", 0, 0, seed, 1.0)
        assert np.array_equal(st.get(kd)[0], st.get(kt)[0]), "선택이 시도 0 이면 조정판 = 기본판"
        for lam in a.LAMS:
            r_def = ("R1", "mlp", "1", "cell", 0, 0, seed, lam)
            r_cw, r_t, r_25 = [("R1", f"mlp_{arm}", "1", "cell", 0, 0, seed, lam) for arm in ("tuned_cw", "tuned", "tuned_l25")]
            assert np.array_equal(st.get(r_cw)[0], st.get(r_def)[0]), "cw 선택이 시도 0 이면 기본판 예측"
            assert np.array_equal(st.get(r_25)[0], st.get(r_t)[0]), "l25 선택이 주 선택과 같으면 조정판 예측"
    rows = pd.DataFrame(books[0].rows)
    assert rows[(rows.learner == "mlp_tuned") & (rows.method == "D0")].sel_default.all()
    assert not rows[(rows.learner == "mlp_tuned") & (rows.method == "R1")].sel_default.any()
    assert rows[rows.learner != "none"].shared_n0.all()
    assert not any(k[1] == "mlp_tuned_l25" and k[0] == "D0" for k in st.keys), "l25 는 종류 R 만"


# ================================================================ (h) 통과 1: TRACE 대조와 복사
def test_h_p1_trace_and_copy(tmp_path, stubs):
    a = with_cfg(args(("--learners", "mlp", "--n-grid", "0,10,40,all", "--draws", "2"), tmp_path))
    c = make_ctx(1)
    book, stats, fit_log, trace = N.p1_book(a, "mlp", "Lena", "x", 1, c, dec_all(trial_D=0, trial_R=2), trace_on=True)
    chk = N.trace_check(c, trace, "mlp", a.kappa)
    assert chk["trace_ok"] and chk["trace_fits"] == len([e for e in trace if e["method"] in ("D0", "R1") and len(e["sel"])])
    cells = H.cells_of([10, 40, -1], 2, len(c.yA))
    assert len(CALLS) == len(cells) * 2, "종류 R 만 조정판 적합(칸 × seed 2)"
    for call in CALLS:
        n = int(call["fit_id"].split("|")[4]); d = int(call["fit_id"].split("|")[5])
        sel = H.draw_cells("Lena", "x", 1, n, d, len(c.yA))
        E_n = T43.coef_n(c, sel, a.kappa)
        assert np.array_equal(call["Xtr"].astype(np.float32), np.vstack([c.X_src, c.XA[sel]]).astype(np.float32), equal_nan=True)
        assert np.allclose(call["ytr"], np.concatenate([c.r0_src, c.yA[sel] - E_n * c.sA[sel]]))
    st = book.st
    for n, d in cells:
        for seed in (0, 1):
            assert np.array_equal(st.get(("D0", "mlp_tuned", "1", "cell", n, d, seed, 1.0))[0], st.get(("D0", "mlp", "1", "cell", n, d, seed, 1.0))[0])
            assert ("R1", "mlp_tuned", "1", "cell", n, d, seed, 0.25) in st
    assert book.status()[0] == "ok"
    assert H.P0_KEY in st and not any(k[4] == 0 for k in st.keys if k[1] != "none"), "통과 1 조각에는 n = 0 학습기 키가 없다"


# ================================================================ (i) 집계: k_default, N2 강건 문장, N2s, N1
def synth_stores(learners=("mlp",), same=True, seed=0, splits=(1, 2)):
    """주 4지역 × 분할의 합성 저장소. same = True 이면 조정판·민감도 판의 SSE 가 기본판과 같다."""
    rng = np.random.RandomState(seed)
    stores = {}
    names = [f"{t}|x" for t in H.MAIN4]
    for nm in names:
        for sp in splits:
            blocks = np.repeat(np.arange(10), 5)
            st = H4_BlockStore(nm, sp, blocks)
            cnt = st.ncell.copy()

            def put(key, base=None, scale=1.0):
                sse = (base if base is not None else rng.uniform(80, 120, st.nb)) * scale * cnt
                st.add_sse(key, sse, cnt)
                return sse / cnt / scale
            put(H.P0_KEY, scale=1.3)
            for n in (0, 10, 40, 160, -1):
                for d in ((0,) if n in (0, -1) else (0, 1)):
                    put(("P1", "none", "1", "cell", n, d, -1, 0.0), scale=1.2)
                    for lr in list(learners) + ["catboost_lo"]:
                        for seed_ in (0, 1):
                            b = put(("D0", lr, "1", "cell", n, d, seed_, 1.0))
                            arms = ("tuned", "tuned_cw") if lr != "catboost_lo" else ()
                            for arm in arms:
                                put(("D0", f"{lr}_{arm}", "1", "cell", n, d, seed_, 1.0), base=b if same else None)
                            for lam in (0.25, 0.5, 1.0):
                                br = put(("R1", lr, "1", "cell", n, d, seed_, lam))
                                for arm in (("tuned", "tuned_cw", "tuned_l25") if lr != "catboost_lo" else ()):
                                    put(("R1", f"{lr}_{arm}", "1", "cell", n, d, seed_, lam), base=br if same else None)
            stores[(nm, sp)] = st
    return stores


def H4_BlockStore(nm, sp, blocks):
    return N.BlockStore(nm, sp, blocks)


def _tests_for(tmp_path, dec, same=True):
    a = args(("--learners", "mlp", "--nboot", "200"), tmp_path)
    X = N._load_h42()
    stores = synth_stores(same=same)
    D = FakeD(splits=(1, 2))
    tms = {nm: X.make_tm(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, D, a.nboot) for nm in sorted({k[0] for k in stores})}
    df = N.build_tests_n(a, tms, D, None, X, dec, pd.DataFrame())
    return a, N.finish_tests(df, a), X


def dec_table(trials):
    """trials = {(종류, 대상): 시도} → build_tests_n 의 dec."""
    out = {}
    for (kind, t), tr in trials.items():
        out[("mlp", kind, t, "x")] = dict(main=dict(trial=tr, flag=""))
    return out


def n2_text(df):
    v = df[(df.test_id == "LGF-N2") & df.scope.isin(["verdict", "verdict_aux"]) & (df.learner == "mlp")]
    return str(v.verdict.iloc[0])


def test_i_n2_robust_rules_and_n1(tmp_path):
    allz = dec_table({(k, t): 0 for k in ("D", "R") for t in H.MAIN4})
    a, df, X = _tests_for(tmp_path, allz)
    t_ = n2_text(df)
    assert "강건 후보" in t_ and "모든 대상에서 기본 설정을 골랐다" in t_ and t_.startswith("Δ = 0(선택 결과) 지역 4/4(종류 R), 4/4(종류 D)")
    nd = dec_table({(k, t): 5 for k in ("D", "R") for t in H.MAIN4})
    _, df2, _ = _tests_for(tmp_path, nd)
    t2 = n2_text(df2)
    assert "초모수 선택에 강건하다" in t2 and not t2.startswith("Δ = 0")
    whole = df2[(df2.test_id == "LGF-N2") & (df2["clause"].astype(str) == "전체")]
    assert len(whole) == 1 and whole.verdict.iloc[0] == "신경망의 결론은 초모수 선택에 강건하다"
    one = dec_table({**{(k, t): 0 for k in ("D", "R") for t in H.MAIN4}, ("D", "Lena"): 5, ("R", "Lena"): 5})
    _, df3, _ = _tests_for(tmp_path, one)
    t3 = n2_text(df3)
    assert "적용해도 차이는 한계 안이다" in t3 and "3/4(종류 R)" in t3
    n1 = df2[(df2.test_id == "LGF-N1") & (df2.scope == "verdict")]
    assert len(n1) == 1 and json.loads(n1.support_class.iloc[0])["mlp"] in (N.SUP_STRONG, N.SUP_WEAK, "기각")
    conf = df2[(df2.holm_family == "conf")]
    assert len(conf) == 1 and conf.confirmatory.all(), "학습기 1개: conf 묶음 1개(학습기 4개면 4개)"
    aux = df2[(df2.holm_family == "aux")]
    assert len(aux) == 6 + 3, "N2 기준 λ 6개(R1 λ0.25 × 3 + D0 × 3) + N3 3개"
    assert (df2[df2.holm_family == "aux"].holm_family_eq == "eq").all()
    st = df2[(df2.test_id == "LGF-N2s") & (df2.scope == "stability")]
    assert len(st) == 7 and set(st.stability) <= {"강건", "약화", "의존", "판정 불가", "판정 불가(풀 지역 불일치)"}
    assert (st.stability == "강건").all(), "기본판과 조정판의 SSE 가 같으면 모든 핵심 대비가 강건"
    kd = df2[(df2.test_id == "LGF-N2") & (df2.scope == "MEAN") & (df2.contrast.astype(str).str.startswith("R1[mlp*]"))]
    assert kd.k_default.astype(str).str.contains("0/4").all()
    l1 = df2[df2.scope == "l1_count"]
    assert len(l1) == 1 and int(l1.n_regions_beat.iloc[0]) >= 0
    n3 = df2[(df2.test_id == "LGF-N3") & df2.scope.isin(["verdict_aux"])]
    assert n3.verdict.iloc[0].startswith("주 비교 대상 catboost_lo"), "CBT 가 없으면 catboost_lo 로 바꾸고 문구를 한정한다"


def test_i2_pairing_drops_unpaired(tmp_path):
    a = args(("--learners", "mlp", "--nboot", "200"), tmp_path)
    X = N._load_h42()
    stores = synth_stores()
    k_drop = ("R1", "mlp_tuned", "1", "cell", 10, 1, 1, 0.25)
    st = stores[("Lena|x", 1)]
    keep = [k for k in st.keys if k != k_drop]
    stores[("Lena|x", 1)] = T43._sub_store(st, keep)
    D = FakeD(splits=(1, 2))
    tms = {nm: X.make_tm(nm, {sp: s_ for (n_, sp), s_ in stores.items() if n_ == nm}, D, a.nboot) for nm in sorted({k[0] for k in stores})}
    df = N.finish_tests(N.build_tests_n(a, tms, D, None, X, dec_table({(k, t): 3 for k in ("D", "R") for t in H.MAIN4}), pd.DataFrame()), a)
    r = df[(df.contrast == "R1[mlp*]-R1[mlp]|n10|lam0.25") & (df.scope == "MEAN")]
    assert int(r.n_unpaired.iloc[0]) == 1 and r.key_set.iloc[0] == "pair"


# ================================================================ (j) eq_verdict, support_class, finish_tests
def R(v, lo=-0.1, hi=0.1, delta=0.0):
    return dict(verdict4=v, ci_lo=lo, ci_hi=hi, ci_lo_beq=lo, ci_hi_beq=hi, delta=delta)


def test_j_eq_verdict_branches(tmp_path):
    a = args((), tmp_path)
    X = N._load_h42()
    eq = [("x", R("동등"))]
    assert N.eq_verdict(X, a, eq * 3, eq * 3, "UP", "DOWN", "EQ")[1] == "eq"
    t_, br = N.eq_verdict(X, a, eq * 3, [("λ1", R("우세", -2, -1, -1.5))], "UP", "DOWN", "EQ")
    assert br == "lam" and "동등으로 쓰지 않는다" in t_
    assert N.eq_verdict(X, a, [("a", R("우세", -2, -1, -1.5)), ("b", R("미결정", -1, 1))], [], "UP", "DOWN", "EQ")[1] == "up"
    assert N.eq_verdict(X, a, [("a", R("열세", 1, 2, 1.5))], [], "UP", "DOWN", "EQ")[1] == "down"
    assert N.eq_verdict(X, a, [("a", R("열세", 1, 2, 1.5)), ("b", R("우세", -2, -1, -1.5))], [], "UP", "DOWN", "EQ")[1] == "mixed"
    t_, br = N.eq_verdict(X, a, [("a", R("미결정", -2, 2)), ("b", R("동등"))], [], "UP", "DOWN", "EQ")
    assert br == "prec" and t_.startswith("정밀도 미달(동등성 판정 불가, 대비 1/2)")
    t_, br = N.eq_verdict(X, a, [("a", R("미결정", -0.6, 0.3)), ("b", R("미결정", -2, 2))], [], "UP", "DOWN", "EQ")
    assert br == "ns" and "(정밀도 미달 대비 1/2)" in t_
    t_, br = N.eq_verdict(X, a, [("a", R("우세", -2, -1, -1.5)), ("b", None)], [], "UP", "DOWN", "EQ")
    assert br == "na" and t_.startswith("판정 불가") and "우세" in t_ and "부분(대비 1/2)" in t_
    assert N.support_class(X, R("열세", 0.5, 1)) == N.SUP_STRONG and N.support_class(X, R("동등")) == N.SUP_STRONG
    assert N.support_class(X, R("미결정", -1, 1)) == N.SUP_WEAK and N.support_class(X, R("우세", -2, -1)) == "기각"
    assert N.support_class(X, None) == "판정 불가"
    df = pd.DataFrame([dict(test_id="LGF-N2", item="aux", primary=True, eq_test=True, verdict4="미결정", ci_lo=-1.0, ci_hi=0.2, ci_lo_beq=-0.3,
                            ci_hi_beq=0.3, p_eq=0.2, holm_p=0.01, verdict4_d10="동등", verdict=np.nan),
                       dict(test_id="LGF-N2", item="aux", primary=True, eq_test=True, verdict4="우세", ci_lo=-1.0, ci_hi=-0.2, ci_lo_beq=-0.9,
                            ci_hi_beq=-0.1, p_eq=0.5, holm_p=0.2, verdict4_d10="우세", verdict=np.nan),
                       dict(test_id="LGF-N1", item="conf", primary=True, eq_test=False, verdict4="동등", ci_lo=-0.1, ci_hi=0.1, ci_lo_beq=-0.1,
                            ci_hi_beq=0.1, p_eq=0.001, holm_p=0.9, verdict4_d10="동등", verdict=np.nan)])
    df["scope"] = "MEAN"
    extra = [dict(test_id="LGF-N1", item="N1aux", primary=False, eq_test=False, scope="MEAN", verdict4="동등", ci_lo=-0.1, ci_hi=0.1, ci_lo_beq=-0.1,
                  ci_hi_beq=0.1, verdict=np.nan),
             dict(test_id="LGF-N1", item="conf", primary=False, eq_test=False, scope="region", verdict4="동등", ci_lo=-0.1, ci_hi=0.1, ci_lo_beq=-0.1,
                  ci_hi_beq=0.1, verdict=np.nan),
             dict(test_id="LGF-N1", item="conf", scope="verdict", verdict="지지"), dict(test_id="LGF-N1", item="N1aux", scope="verdict_aux", verdict="x"),
             dict(test_id="LGF-N2", item="aux", primary=True, scope="MEAN", verdict4="미결정", ci_lo=np.nan, ci_hi=0.9, ci_lo_beq=-1.0, ci_hi_beq=0.8,
                  verdict=np.nan)]
    out = N.finish_tests(pd.concat([df, pd.DataFrame(extra)], ignore_index=True), a)
    assert np.isclose(out.halfwidth.iloc[0], 0.6) and not out.precision_ok.iloc[0] and out.precision_note.iloc[0] == "정밀도 미달(동등성 판정 불가)"
    assert out.flag_uncorr.tolist()[:3] == ["", "보정 전 유의", ""]
    assert out.holm_family.tolist()[:3] == ["aux", "aux", "conf"] and out.holm_family_eq.tolist()[:3] == ["eq", "eq", ""]
    assert out.confirmatory.tolist() == [False, False, True, False, False, True, False, False], "확인적 = N1 의 conf 층화 평균 행과 주 판정 행만"
    assert (out.platform == N.PLATFORM).all()
    assert np.isnan(out.halfwidth.iloc[-1]) and np.isnan(N.halfwidth(dict(ci_lo=np.nan, ci_hi=0.4, ci_lo_beq=-0.5, ci_hi_beq=0.3))), \
        "반폭은 네 끝값이 모두 유한할 때만(h47 과 같은 정의)"
    assert out.precision_note.iloc[-1] == "", "반폭을 계산할 수 없으면 정밀도 미달 표기를 붙이지 않는다"
    t_eq, _ = N.eq_verdict(X, a, [("a", R("미결정", -0.3, 0.3)), ("b", R("미결정", -0.2, 0.3))], [], "UP", "DOWN", "EQ")
    assert t_eq == "차이를 확인하지 못함", "정밀도 미달 대비가 없으면 표기를 붙이지 않는다(h47.prec_suffix 와 같은 규칙)"
    assert list(out.columns[:5]) == ["test_id", "item", "contrast", "scope", "n"]


# ================================================================ (k) CatBoost 선택 표 = h42.pick_tuned
def test_k_cb_select_matches_pick_tuned(tmp_path):
    a = args(("--cpu-only",), tmp_path)
    X = N._load_h42()
    rng = np.random.RandomState(0)
    rows = []
    for t in ("Lena", "Canada"):
        for kind in N.KINDS:
            for dp, it in X.TUNE_GRID:
                rows.append(dict(target=t, mode="x", n_src=100, n_regions=3, regions="r1,r2,r3", sel_sig=N.cb_sel_sig(a), kind=kind, depth=dp,
                                 iterations=it, learning_rate=X.TUNE_LR, score_rmse=float(rng.uniform(10, 12)), score_rmse_lam025=np.nan,
                                 rmse_by_region="{}", n_fit=3, sec=0.1))
    rows[3]["score_rmse"] = rows[5]["score_rmse"] = 5.0                    # 동률: 용량(반복 × 2^깊이)이 작은 쪽
    N.write_cb_select(a, rows)
    ref = X.pick_tuned(rows)
    for t in ("Lena", "Canada"):
        got = N.load_cb_tuned(a, t, "x")
        for kind in N.KINDS:
            q = ref[(ref.target == t) & (ref.kind == kind) & ref.selected].iloc[0]
            assert got[kind]["depth"] == int(q.depth) and got[kind]["iterations"] == int(q.iterations)
    assert N.load_cb_tuned(a, "Lena", "x")["D"] == dict(iterations=X.TUNE_GRID[3][1], depth=X.TUNE_GRID[3][0], learning_rate=X.TUNE_LR,
                                                        l2_leaf_reg=3.0)


# ================================================================ (l) 모듈 모양과 TabM 헤드
def test_l_modules_and_tabm_heads():
    import torch
    assert N.tabm_loop_diff("cpu") <= 1e-6
    cf = N.build_configs(32)
    for lr in ("mlp", "tabm", "ftt"):
        for e in cf[lr]["D"][1:4]:
            net = N.build_net(lr, e["params"], DF).eval()
            with torch.no_grad():
                assert tuple(net(torch.randn(7, DF)).shape) == (7,)
    net = N._mods().MLPp(DF, 3, 64, shape="half", norm="none")
    lin = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in lin] == [64, 32, 16, 1]


def test_l2_fit_tuned_cpu_short():
    rng = np.random.RandomState(0)
    X = rng.randn(300, DF); X[0, 2] = np.nan
    y = 3 * np.nan_to_num(X[:, 1]) + rng.randn(300)
    for lr, tr in (("mlp", 1), ("tabm", 1), ("ftt", 1)):
        e = N.cfg_entry(lr, "D", tr)
        out, ep, npar = N.fit_tuned(lr, e, X, y, [X[:11], X[:0]], 0, "t|a", max_epochs=2, patience=16, dev="cpu")
        assert out[0].shape == (11,) and out[1].shape == (0,) and np.all(np.isfinite(out[0])) and 1 <= ep <= 2 and npar > 0
    e = N.cfg_entry("mlp", "D", 2)
    o1, _, _ = N.fit_tuned("mlp", e, X, y, [X[:5]], 1, "t|b", max_epochs=2, dev="cpu")
    o2, _, _ = N.fit_tuned("mlp", e, X, y, [X[:5]], 1, "t|b", max_epochs=2, dev="cpu")
    assert np.allclose(o1[0], o2[0]), "같은 seed 와 적합 식별자이면 같은 예측(CPU)"


# ================================================================ (m) 창 파일, 드레인, 종료 코드
def test_m_window_drain_exit(tmp_path):
    a = args(("--learners", "mlp"), tmp_path)
    w1 = N.window_init_t0(a)
    assert N._parse_time(w1["t0"]) is not None and abs(N._parse_time(w1["deadline"]) - N._parse_time(w1["t0"]) - 48 * 3600) < 2
    time.sleep(1.1)
    w2 = N.window_init_t0(a)
    assert w2["t0"] == w1["t0"], "t0 는 한 번만 쓴다"
    assert not N.window_closed(a)
    w2["deadline"] = time.time() - 10
    N._write_json(a.WINDOW, w2, strict=True)
    assert N.window_closed(a), "epoch 초 형식의 deadline 도 읽는다"
    with pytest.raises(SystemExit, match="사전 점검 개정"):
        N.check_window_args(a)
    w2.update(E1=False, reduce=["S2", "S5"], restored=["S5"])
    N._write_json(a.WINDOW, w2, strict=True)
    with pytest.raises(SystemExit, match="B, E1, reduce"):
        N.check_window_args(a)
    w2.update(B=46.0)
    N._write_json(a.WINDOW, w2, strict=True)
    N.check_window_args(a)
    with pytest.raises(SystemExit, match="--reduce"):
        N.check_window_args(args(("--learners", "mlp", "--reduce", "S5"), tmp_path))
    with pytest.raises(SystemExit, match="--e1"):
        N.check_window_args(args(("--learners", "mlp", "--e1", "1"), tmp_path))
    b = args(("--cpu-only",), tmp_path)
    N.drain_path(b).parent.mkdir(parents=True, exist_ok=True)
    N.drain_path(b).write_text("")
    P = N.StaticPlanner(b, None, [("chk",)])
    res = N.run_cpu(b, P, time.time())
    assert res["drained"] == "drain" and not N.drain_path(b).exists() and P.state["chk"] == "pending"
    assert N.exit_code_n(dict(res, n_fail=0)) == 3
    assert N.exit_code_n(dict(interrupted="x", n_fail=1)) == 130 and N.exit_code_n(dict(n_fail=1, drained="d")) == 1
    assert N.exit_code_n(dict(closed="window", n_partial=1)) == 4 and N.exit_code_n(dict(n_partial=1)) == 2 and N.exit_code_n({}) == 0


# ================================================================ (n) GPU 거부와 SHA
def test_n_gpu_reject_and_sha(tmp_path, monkeypatch):
    ok = args(("--allow-local", "--gpus", "5,9"), tmp_path)
    N.check_run_args(ok)
    for extra, pat in ((("--allow-local", "--gpus", "8"), "쓰지 않는다"), (("--allow-local", "--gpus", "1"), "allow-gpus-01"),
                       (("--allow-local", "--gpus", "11"), "허용 후보"), (("--allow-local",), "비어 있다"), (("--gpus", "5"), "allow-local"),
                       (("--allow-local", "--cpu-only", "--gpus", "5"), "cpu-only")):
        with pytest.raises(SystemExit, match=pat):
            N.check_run_args(args(extra, tmp_path))
    N.check_run_args(args(("--allow-local", "--gpus", "0", "--allow-gpus-01"), tmp_path))
    with pytest.raises(SystemExit, match="n_cb"):
        args(("--jobs", "n_cb"), tmp_path)
    with pytest.raises(SystemExit, match="하한"):
        args(("--n-random", "mlp=8"), tmp_path)
    info = {5: dict(uuid="u5", used=2), 6: dict(uuid="u6", used=2), 9: dict(uuid="u9", used=900)}
    a = args(("--allow-local", "--gpus", "5,6,9"), tmp_path)
    a.LGT_LOCK = tmp_path / "lock.json"
    a.LGT_LOCK.write_text(json.dumps(dict(pid=os.getpid())))
    g, dropped = N.screen_gpus_n(a, info, [])
    assert g == [5] and {d[0] for d in dropped} == {6, 9}, "9 는 메모리, 6 은 LGT 잠금"
    real = T43.file_sha
    monkeypatch.setattr(T43, "file_sha", lambda p, short=0: "0" * 40 if "h43" in str(p) else real(p, short))
    with pytest.raises(SystemExit, match="h43"):
        N.check_sha(args((), tmp_path))
    b = args(("--h43-sha", "0" * 40), tmp_path)
    N.check_sha(b)
    assert b.REVISION_NEEDED and "개정 필요" in b.REVISION_NEEDED[0]


# ================================================================ (o) 작업 계획: 2단계 생성과 선택 기록
def write_sel_json(a, D, u, rmse):
    si = N.src_info(a, D, u[3], u[4])
    folds = [dict(region=r, n_te=40, n_tr=100, n_tr_within100km=0, rmse=float(v), rmse_l25=float(v) + 0.1, sse=float(v) ** 2 * 40,
                  sse_l25=0.0, fail=False, err="", epochs_run=5, n_params=10, fit_s=0.1) for r, v in zip(si.regions, rmse)]
    rec = dict(learner=u[1], kind=u[2], target=u[3], mode=u[4], stage=u[5], trial=u[6], seed=u[7], cfg_entry=N.cfg_entry(u[1], u[2], u[6]),
               folds=folds, n_params=10 + u[6], status="ok", n_fail=0, unit_sig=N.sel_unit_sig(a, u, si), fit_s_sum=0.1, epochs_run=[5] * len(folds),
               **N.sel_scores(folds, u[2]), run_id="")
    N._write_json(N.sel_path(a, u), rec, strict=True)


def test_o_planner_stage2_and_decision(tmp_path):
    a = with_cfg(args(("--learners", "mlp", "--jobs", "n_sel_fast,n_p0_fast", "--targets", "Lena:x", "--n-random", "mlp=16", "--resume"), tmp_path))
    D = FakeD()
    inject_src(a, D)
    a.NSEL.mkdir(parents=True, exist_ok=True)
    good = {3: [8.0, 8.0, 8.0], 7: [8.5, 8.5, 8.5], 11: [8.8, 8.8, 8.8]}
    for kind in N.KINDS:
        for u in N.stage1_units(a, "mlp", kind, "Lena", "x"):
            write_sel_json(a, D, u, good.get(u[6], [10.0 + 0.01 * u[6]] * 3))
    P = N.build_planner(a, D, fresh=False)
    s2 = [nm for nm, s_ in P.state.items() if s_ == "pending" and "|2|" in nm]
    assert len(s2) == 2 * 4 * 2, "종류 2 × (상위 3 + 시도 0) × seed 2"
    assert {int(nm.split("|")[6]) for nm in s2} == {0, 3, 7, 11}
    assert P.next_ready()[0] == "sel" and not P.decided("mlp", "Lena", "x")
    for nm in sorted(s2):
        u = P.units[nm]
        write_sel_json(a, D, u, good.get(u[6], [10.0] * 3))
        P.on_done(u)
    assert P.decided("mlp", "Lena", "x")
    assert P.next_ready() == ("p0", "mlp", "Lena", "x")
    dec = N.load_decision(a, "mlp", "Lena", "x")
    assert dec["D"]["main"]["trial"] == 3 and dec["R"]["main"]["trial"] == 3
    sel = N.read_select(a)
    assert set(sel.row_type) == {"fit", "decision"}
    fit_sel = sel[(sel.row_type == "fit") & sel.selected.astype(str).str.lower().eq("true")]
    assert set(fit_sel.trial) == {3} and set(fit_sel.stage) == {2}
    assert a.SENS_T.exists() and a.STAB_T.exists()


# ================================================================ (p) 선택 단위(대체 학습기)
def test_p_sel_unit_stub(tmp_path, stubs, monkeypatch):
    a = with_cfg(args(("--learners", "mlp", "--targets", "Lena:x"), tmp_path))
    D = FakeD()
    si = inject_src(a, D)
    monkeypatch.setattr(N, "get_data", lambda a_: D)
    u0 = ("sel", "mlp", "R", "Lena", "x", 1, 0, 0)
    res = N.run_sel_unit(a, u0)
    assert res["status"] == "ok" and res["n_fit"] == 3
    rec = N.load_sel(a, u0)
    te = si.mac == "r1"
    E = H.ls_E(si.y[~te], si.s[~te])
    g = si.y[~te] - E * si.s[~te]
    pred = E * si.s[te] + float(g.mean()) + 0.01 * np.nan_to_num(si.X[te][:, 1])
    assert np.isclose(rec["folds"][0]["rmse"], np.sqrt(np.mean((pred - si.y[te]) ** 2)))
    assert np.isclose(rec["score_main"], np.mean([f["rmse"] for f in rec["folds"]]))
    assert N.sel_done(a, D, u0) and not N.sel_done(a, D, u0, run_id="other")
    u1 = ("sel", "mlp", "D", "Lena", "x", 1, 4, 0)
    N.run_sel_unit(a, u1)
    assert len(CALLS) == 3 and all(c_["trial"] == 4 and c_["fit_id"].startswith("sel|Lena|x|") for c_ in CALLS)
    assert np.isnan(N.load_sel(a, u1)["score_l25"]), "종류 D 에는 λ 0.25 점수가 없다"
    b = with_cfg(args(("--learners", "mlp", "--targets", "Lena:x", "--smoke"), tmp_path))
    inject_src(b, D)
    r_b = N.run_sel_unit(b, ("sel", "mlp", "D", "Lena", "x", 1, 1, 0))
    assert r_b["status"] == "ok" and np.isnan(N.load_sel(b, ("sel", "mlp", "D", "Lena", "x", 1, 1, 0))["folds"][0]["rmse"]), "스모크는 RMSE 를 쓰지 않는다"


# ================================================================ (q) 조각 기록과 재개 판정
def test_q_shard_state(tmp_path, stubs):
    a = with_cfg(args(("--learners", "mlp"), tmp_path))
    cs = [make_ctx(sp) for sp in (1, 2)]
    dec = dec_all(0, 0)
    books, _, _ = N.p0_books(a, "mlp", "Lena", "x", cs, dec)
    u = N.write_shard(a, "p0", "mlp", books[0], N.shard_cfg(a, "p0", "mlp", dec=dec), dict(shared_n0=True, decisions=dec))
    assert u["status"] == "ok" and u["shared_n0"] and u["cfg_hash"] == H.cfg_hash(N.shard_cfg(a, "p0", "mlp", dec=dec))
    assert N.shard_state(a, "p0", "mlp", "Lena", "x", 1, dec=dec)[0]
    assert N.shard_state(a, "p0", "mlp", "Lena", "x", 1) == (False, "선택 표에 이 학습기·대상의 선택이 없다"), "선택 표가 없으면 미완료"
    dec2 = dec_all(0, 3)
    assert N.shard_state(a, "p0", "mlp", "Lena", "x", 1, dec=dec2) == (False, "설정 불일치(cfg_hash)"), "선택 결과가 바뀌면 미완료(명세 4.3)"
    assert N.shard_state(a, "p0", "mlp", "Lena", "x", 1, dec=dec_all(0, 0, cw=5))[0] is False, "민감도 선택(cw)도 통과 0 설정에 들어간다"
    assert u["cfg_common"] == N.write_shard(a, "p0", "mlp", books[1], N.shard_cfg(a, "p0", "mlp", dec=dec2), {})["cfg_common"], \
        "선택 결과는 조각 사이 설정 대조(cfg_common)에서 뺀다"
    runs = pd.read_csv(N.shard_paths(a, "p0", "mlp", "Lena", "x", 1)["runs"])
    assert list(runs.columns[:len(N.RUN_COLS)]) == N.RUN_COLS and runs.rmse_cm.notna().any()
    a2 = with_cfg(args(("--learners", "mlp", "--epochs", "50"), tmp_path))
    assert N.shard_state(a2, "p0", "mlp", "Lena", "x", 1, dec=dec) == (False, "설정 불일치(cfg_hash)")
    p = N.shard_paths(a, "p0", "mlp", "Lena", "x", 1)["unit"]
    uj = json.loads(p.read_text()); uj["status"] = "partial"; p.write_text(json.dumps(uj))
    a3 = with_cfg(args(("--learners", "mlp", "--rerun-partial"), tmp_path))
    assert N.shard_state(a, "p0", "mlp", "Lena", "x", 1, dec=dec)[0] and not N.shard_state(a3, "p0", "mlp", "Lena", "x", 1, dec=dec)[0]
    s = with_cfg(args(("--learners", "mlp", "--smoke"), tmp_path))
    books_s, _, _ = N.p0_books(s, "mlp", "Lena", "x", [make_ctx(1)], dec)
    N.write_shard(s, "p0", "mlp", books_s[0], N.shard_cfg(s, "p0", "mlp", dec=dec), {})
    ps = N.shard_paths(s, "p0", "mlp", "Lena", "x", 1)
    assert not ps["npz"].exists() and pd.read_csv(ps["runs"]).rmse_cm.isna().all(), "산출 제한: BlockStore 미저장, RMSE NaN"


def test_q2_check_cfg_common_across_targets():
    """집계의 조각 대조: 대상마다 선택 결과(cfg_hash)가 달라도 공통 설정(cfg_common)이 같으면 통과, 공통 설정이 다르거나 한 대상 안에서
    cfg_hash 가 여러 종이면 거부(2026-10-01 집계 수정, LGF 개정 이력)."""
    a = SimpleNamespace(allow_mixed_cfg=False)
    sh = [dict(pass_="cpu", learner="catboost_tuned_loc") for _ in range(4)]
    us = [dict(target=t, mode="x", cfg_hash=h, cfg_common="c1", code_sha={"h48": "s"})
          for t, h in (("Lena", "h1"), ("Lena", "h1"), ("Canada", "h2"), ("Canada", "h2"))]
    info = N.check_cfg_n(a, sh, us)
    assert info["cpu|catboost_tuned_loc"]["cfg_common"] == {"c1": 4}
    with pytest.raises(SystemExit):
        N.check_cfg_n(a, sh, [dict(u, cfg_common=("c2" if i == 0 else "c1")) for i, u in enumerate(us)])
    with pytest.raises(SystemExit):
        N.check_cfg_n(a, sh, [dict(u, cfg_hash=("h3" if i == 0 else u["cfg_hash"])) for i, u in enumerate(us)])
    N.check_cfg_n(SimpleNamespace(allow_mixed_cfg=True), sh, [dict(u, cfg_common=("c2" if i == 0 else "c1")) for i, u in enumerate(us)])


# ================================================================ (r) 검증 지적 반영: 창 파일 설정, 조각 선택 기록, 비교 대상, 문구, 실행 제어
def write_window(a, **kw):
    w = dict(window_h=48.0, gpu_events=[], decisions=[])
    w.update(kw)
    N._write_json(a.WINDOW, w, strict=True)


def test_r1_adopt_window_for_summary_and_cpu(tmp_path):
    a = args(("--summarize-only", "--allow-local"), tmp_path)
    assert N.adopt_window(a) == "명령행(창 파일에 사전 점검 결정 없음)" and a.N_RANDOM == dict(mlp=32, tabm=32, ftt=16, realmlp=16)
    write_window(a, B=46.0, E1=True, cbt_enabled=True, reduce=["S2", "S5", "S6", "S7"], restored=["S6"])
    a = args(("--summarize-only", "--allow-local"), tmp_path)
    s_before = N.sel_sig(a, "ftt")
    assert N.adopt_window(a) == "창 파일"
    assert a.E1 and a.REDUCE_N == ["S5", "S7"] and a.N_RANDOM == dict(mlp=16, tabm=16, ftt=32, realmlp=32)
    assert a.DRAWS_P1 == 1 and a.P1_SPLITS == [1, 2, 3, 4, 5], "S6 은 복원되어 적용하지 않는다"
    assert N.sel_sig(a, "ftt") != s_before, "k 가 sel_sig 에 들어가므로 창 파일의 E1 을 써야 선택 표 행을 찾는다"
    with pytest.raises(SystemExit, match="--e1"):
        N.adopt_window(args(("--summarize-only", "--e1", "0"), tmp_path))
    with pytest.raises(SystemExit, match="--reduce"):
        N.adopt_window(args(("--summarize-only", "--reduce", "S5"), tmp_path))
    b = args(("--summarize-only", "--e1", "1", "--reduce", "S5,S7"), tmp_path)
    assert N.adopt_window(b) == "창 파일" and b.N_RANDOM["ftt"] == 32
    c = args(("--cpu-only", "--allow-local", "--threads", "2"), tmp_path)
    w = N.check_window_args(c)
    assert w["B"] == 46.0 and c.E1 and c.DRAWS_P1 == 1, "CPU 작업은 --e1 없이 창 파일의 값을 받는다"
    g = args(("--allow-local", "--gpus", "5"), tmp_path)
    with pytest.raises(SystemExit, match="--reduce"):
        N.check_window_args(g)
    N.check_window_args(args(("--allow-local", "--gpus", "5", "--e1", "1", "--reduce", "S5,S7"), tmp_path))
    write_window(a, B=46.0, E1=False, cbt_enabled=False, reduce=["S1"], restored=[])
    d = args(("--summarize-only",), tmp_path)
    N.adopt_window(d)
    assert d.T2 == [] and d.threads == 1 and d.nboot <= 1000 and d.LOCAL_SUMMARY


def test_r2_shard_decisions(tmp_path):
    sh = [dict(pass_="p0", learner="mlp", target="Lena", mode="x", split=1), dict(pass_="p1", learner="mlp", target="Lena", mode="x", split=2),
          dict(pass_="cpu", learner="catboost_lo", target="Lena", mode="x", split=1)]
    d0 = {"D": {"main": dict(trial=3, flag=""), "cw": dict(trial=0, flag="sel_default_pref")},
          "R": {"main": dict(trial=0, flag="sel_default_pref"), "cw": dict(trial=0, flag=""), "l25": dict(trial=5, flag="")}}
    d1 = {"D": dict(trial=3, flag=""), "R": dict(trial=0, flag="sel_default_pref")}
    tab = {("mlp", "D", "Lena", "x"): {"main": dict(trial=3), "cw": dict(trial=0)}, ("mlp", "R", "Lena", "x"): {"main": dict(trial=0), "cw": dict(trial=0),
                                                                                                                "l25": dict(trial=5)}}
    out, probs = N.shard_decisions(sh, [dict(decisions=d0), dict(decisions=d1), {}], tab)
    assert not probs and out[("mlp", "D", "Lena", "x")]["main"]["trial"] == 3 and out[("mlp", "R", "Lena", "x")]["l25"]["trial"] == 5
    out, probs = N.shard_decisions(sh, [dict(decisions=d0), dict(decisions=d1), {}], {})
    assert probs and all("선택 표에" in p_ for p_ in probs), "선택 표에 행이 없으면(예: sel_sig 불일치) 문제로 적는다"
    bad = {"D": dict(trial=7, flag=""), "R": dict(trial=0, flag="")}
    _, probs = N.shard_decisions(sh, [dict(decisions=d0), dict(decisions=bad), {}], tab)
    assert any("다른 조각" in p_ for p_ in probs)
    _, probs = N.shard_decisions(sh, [dict(decisions=d0), {}, {}], tab)
    assert any("선택 기록이 없다" in p_ for p_ in probs)


def test_r3_comparator_from_window(tmp_path):
    assert N.comparator_n3(True)[:2] == ("catboost_tuned_loc", "CBT")
    assert N.comparator_n3(False)[0] == "catboost_lo" and "거짓" in N.comparator_n3(False)[2]
    assert N.comparator_n3(None)[0] == "catboost_lo" and "결정 없음" in N.comparator_n3(None)[2]
    a = args(("--learners", "mlp", "--nboot", "200"), tmp_path)
    X = N._load_h42()
    stores = synth_stores()
    D = FakeD(splits=(1, 2))
    tms = {nm: X.make_tm(nm, {sp: st for (n_, sp), st in stores.items() if n_ == nm}, D, a.nboot) for nm in sorted({k[0] for k in stores})}
    dec = dec_table({(k, t): 3 for k in ("D", "R") for t in H.MAIN4})
    df = N.finish_tests(N.build_tests_n(a, tms, D, None, X, dec, pd.DataFrame(), cbt_enabled=True), a)
    v = df[(df.test_id == "LGF-N3") & (df.scope == "verdict_aux")]
    assert v.comparator.iloc[0] == "catboost_tuned_loc" and v.verdict.iloc[0].startswith("판정 불가"), \
        "cbt_enabled 가 참이면 CBT 조각이 없어도 CBT 가 주 비교 대상이고 판정 불가로 처리한다"
    df2 = N.finish_tests(N.build_tests_n(a, tms, D, None, X, dec, pd.DataFrame(), cbt_enabled=False), a)
    v2 = df2[(df2.test_id == "LGF-N3") & (df2.scope == "verdict_aux")]
    assert v2.comparator.iloc[0] == "catboost_lo" and "양쪽 조정" not in v2.verdict.iloc[0]
    n2s = df2[(df2.test_id == "LGF-N2s") & (df2.scope == "verdict_aux")]
    assert n2s.verdict.astype(str).str.contains("Δ = 0(선택 결과) 지역", regex=False).all(), "N2s 판정 문구에 k/4 를 적는다"
    allz = dec_table({(k, t): 0 for k in ("D", "R") for t in H.MAIN4})
    df3 = N.finish_tests(N.build_tests_n(a, tms, D, None, X, allz, pd.DataFrame(), cbt_enabled=False), a)
    n2s3 = df3[(df3.test_id == "LGF-N2s") & (df3.scope == "verdict_aux")]
    assert n2s3.verdict.astype(str).str.contains("조정판 = 기본판(선택 결과", regex=False).all()
    n1 = df3[(df3.test_id == "LGF-N1") & (df3.scope == "verdict")]
    assert "Δ = 0(선택 결과) 지역(종류 D): mlp 4/4" in n1.verdict.iloc[0] and "조정판 = 기본판" in n1.verdict.iloc[0]
    rows = df3[(df3.test_id == "LGF-N1") & (df3.scope == "MEAN") & (df3.holm_family == "conf")]
    assert rows.blind.all() and not n1.blind.iloc[0] and "비맹검 부분 포함" in str(n1.prior_info.iloc[0]), "조정판 대비 행은 맹검, 판정 행에 비맹검 표시"


def test_r4_n1_text_branches():
    X = N._load_h42()
    strong, weak, win = R("열세", 0.5, 1.0, 0.7), R("미결정", -0.4, 0.3, -0.05), R("우세", -2, -1, -1.5)
    lrs = ["mlp", "tabm", "ftt", "realmlp"]
    cls = {lr: N.support_class(X, r) for lr, r in zip(lrs, [strong] * 4)}
    t_ = N.n1_text(X, [(lr, strong) for lr in lrs], cls, lrs)
    assert t_.startswith(N.SUP_STRONG) and "판별 신경망 4종(MLP, 다중 헤드 MLP, 축소 FT-Transformer, RealMLP)" in t_
    res = [("mlp", strong), ("tabm", weak), ("ftt", strong), ("realmlp", strong)]
    cls = {lr: N.support_class(X, r) for lr, r in res}
    t_ = N.n1_text(X, res, cls, lrs)
    assert t_.startswith("지지. ") and "판별 신경망 3종(MLP, 축소 FT-Transformer, RealMLP)" in t_
    assert "조정한 다중 헤드 MLP 에서 우세가 관찰되지 않았다(CI [-0.40, 0.30] cm)" in t_
    res = [("mlp", weak), ("tabm", weak)]
    t_ = N.n1_text(X, res, {lr: N.SUP_WEAK for lr, _ in res}, ["mlp", "tabm"])
    assert t_.startswith(N.SUP_WEAK) and "판별 신경망" not in t_
    res = [("mlp", win), ("tabm", strong)]
    t_ = N.n1_text(X, res, {lr: N.support_class(X, r) for lr, r in res}, ["mlp", "tabm"])
    assert t_.startswith("기각: 조정한 MLP") and "마스터 곡선에 더한다" in t_ and "기본 초모수 한정" in t_
    t_ = N.n1_text(X, [("mlp", win), ("tabm", None)], {"mlp": "기각", "tabm": "판정 불가"}, ["mlp", "tabm"])
    assert t_ == "기각(일부 대비 판정 불가, 대비 1/2)"


def test_r5_run_control_helpers(tmp_path, monkeypatch):
    assert N._peek_threads(argv=["x", "--summarize-only"]) == "1" and N._peek_threads(argv=["x", "--summarize-only", "--allow-local"]) == "2"
    assert N._peek_threads(argv=["x", "--threads", "8"]) == "2"
    g = args(("--allow-local", "--gpus", "2,6"), tmp_path)
    info = {2: dict(uuid="u2", used=2), 6: dict(uuid="u6", used=2), 0: dict(uuid="u0", used=13000)}
    monkeypatch.setattr(T43, "query_gpu_info", lambda: info)
    monkeypatch.setattr(T43, "query_gpu_apps", lambda: [])
    monkeypatch.setattr(T43, "GPU_SETTLE_S", 0)
    assert N.rescreen_n(g, [2]) == [2], "허용 후보 GPU 2 의 재확인이 h43 기본 예약(0–4)으로 거부되지 않는다"
    g.LGT_LOCK = tmp_path / "lgt_lock.json"
    g.LGT_LOCK.write_text(json.dumps(dict(pid=os.getpid())))
    assert N.rescreen_n(g, [2, 6]) == [2], "LGT 잠금이 살아 있으면 6 을 뺀다"
    assert N.rescreen_n(g, [2], excluded=[2]) == []
    monkeypatch.setattr(T43, "query_gpu_info", lambda: (_ for _ in ()).throw(OSError("smi")))
    assert N.rescreen_n(g, [2]) == [], "nvidia-smi 오류는 잡아 빈 목록"
    P = N.StaticPlanner(g, None, [("chk",), ("pre_d",)])
    P.mark(("chk",), "done")
    un, why = N.unrun_units(P, dict(gpu_lost=True))
    assert un == ["pre_d"] and "GPU" in why
    assert N.unrun_units(P, dict(drained="drain")) == ([], "") and N.unrun_units(P, dict(closed="window"))[0] == []
    assert N.exit_code_n(dict(n_fail=len(un))) == 1 and N.exit_code_n(dict(refused=True)) == 1
    c = args(("--cpu-only",), tmp_path)
    assert N.status_path(c).name == "lgfn_cpu_run_status.json" and N.status_path(g).name == "lgfn_run_status.json"
    assert N.drain_path(c).parent.name == "run_lgfn_cpu"


def test_r6_sel_rerun_partial_and_cb_pool(tmp_path, stubs, monkeypatch):
    a = with_cfg(args(("--learners", "mlp", "--targets", "Lena:x"), tmp_path))
    D = FakeD()
    inject_src(a, D)
    monkeypatch.setattr(N, "get_data", lambda a_: D)
    u = ("sel", "mlp", "D", "Lena", "x", 1, 2, 0)
    N.run_sel_unit(a, u)
    rec = N.load_sel(a, u)
    rec["status"] = "partial"
    N._write_json(N.sel_path(a, u), rec, strict=True)
    b = with_cfg(args(("--learners", "mlp", "--targets", "Lena:x", "--rerun-partial"), tmp_path))
    inject_src(b, D)
    assert N.sel_done(a, D, u) and not N.sel_done(b, D, u), "--rerun-partial 이면 fold 실패가 있는 선택 JSON 을 다시 돈다"
    import catboost
    seen = []

    class FakePool:
        def __init__(self, data, label=None, thread_count=-1, **kw):
            seen.append(int(thread_count)); self.n = len(data)

    class FakeCB:
        def __init__(self, **kw):
            self.kw = kw

        def fit(self, X, y=None):
            assert isinstance(X, FakePool) and y is None, "학습 Pool 을 직접 만든다"

        def predict(self, P_, thread_count=-1):
            seen.append(int(thread_count)); return np.zeros(P_.n)
    monkeypatch.setattr(catboost, "Pool", FakePool)
    monkeypatch.setattr(catboost, "CatBoostRegressor", FakeCB)
    out = N._cb_tuned_fit(dict(iterations=5, learning_rate=0.05, depth=3), np.zeros((7, 3)), np.zeros(7), np.zeros((4, 3)), 0, 2)
    assert out.shape == (4,) and seen == [2, 2, 2], "학습·채점 Pool 과 예측의 thread_count 는 --threads"
    assert T43.is_cuda_oom(RuntimeError("CUDA out of memory")) and N.fail_text(RuntimeError("CUDA out of memory")).startswith("[OOM]")


def test_r7_projection_rows(tmp_path, monkeypatch):
    a = with_cfg(args(("--precheck",), tmp_path))
    D = FakeD()
    for t in list(H.MAIN4) + [H.ALASKA]:
        inject_src(a, D, target=t)
    monkeypatch.setattr(N, "get_data", lambda a_: D)
    tm = []
    for lr in N.LEARNERS_ALL:
        for tr in N.n_trials_list(lr, N.K_LIST_MAX):
            tm.append(dict(part="a", learner=lr, trial=tr, sec_per_epoch=0.01 if tr == 0 else 0.1))
        tm += [dict(part="c", learner=lr, what="p0_shared", fit_s=1.0), dict(part="c", learner=lr, what="p1_n10", fit_s=1.0),
               dict(part="c", learner=lr, what="p1_nall", fit_s=1.0)]
    tm += [dict(part="d", what="select", fit_s=10.0), dict(part="d", what="full_fit", fit_s=0.5)]
    ep = pd.DataFrame([dict(part="b", learner=lr, trial=tr, epochs_run=40) for lr in N.LEARNERS_ALL for tr in (0, 1, 2)])
    pj = N.projection_n(a, pd.DataFrame(tm), ep)
    rows = set(pj.row)
    assert {"J2", "J3", "J4", "J5", "J7", "J10", "P_N", "E1", "S1", "S5", "S6", "S7", "C1"} <= rows and (pj.script == "N").all()
    v = pj.set_index("row").est_gpu_h
    assert np.isclose(v["P_N"], sum(v[j] for j in ("J2", "J3", "J4", "J5", "J7", "J10")), atol=0.01) and np.isclose(v["S1"], v["J10"])
    assert v["E1"] > 0 and v["S7"] > 0 and v["S5"] > 0 and v["S6"] >= 0 and np.isnan(v["C1"])
    out = N.write_projection(a, pj, "N")
    pd.DataFrame([dict(script="F", row="J1", est_gpu_h=0.1)]).pipe(lambda f: pd.concat([pd.read_csv(tmp_path / "lgf_projection.csv"), f])).to_csv(
        tmp_path / "lgf_projection.csv", index=False)
    out = N.write_projection(a, pj, "N")
    assert set(out.script) == {"N", "F"} and (out.script == "F").sum() == 1, "N 행만 바꾸고 F 행은 남긴다"


# ================================================================ 무거운 시험(CPU, 실제 적합)
@heavy
def test_H1_p0_shared_real_mlp(tmp_path):
    a = with_cfg(args(("--learners", "mlp", "--epochs", "3"), tmp_path))
    cs = [make_ctx(sp) for sp in (1, 2)]
    books, _, _ = N.p0_books(a, "mlp", "Lena", "x", cs, dec_all(0, 0))
    for c, b_ in zip(cs, books):
        _, st_ref, _ = H.run_ctx(c, "gpu", N.ha_n(a, "mlp", "0"), learner_axis=True, learners=["mlp"])
        for k in [k for k in st_ref.keys if k[1] == "mlp"]:
            assert np.allclose(st_ref.get(k)[0], b_.st.get(k)[0], rtol=1e-6, atol=1e-6)


@heavy
def test_H2_fit_tuned_realmlp():
    rng = np.random.RandomState(0)
    X = rng.randn(200, DF); y = 2 * X[:, 1] + rng.randn(200)
    out, ep, npar = N.fit_tuned("realmlp", N.cfg_entry("realmlp", "D", 1), X, y, [X[:9]], 0, "t", threads=2, realmlp_epochs=2, dev="cpu")
    assert out[0].shape == (9,) and np.all(np.isfinite(out[0])) and ep == 2 and npar > 0
    assert N.rmlp_key_check()["rmlp_keys_ok"]


@heavy
def test_H3_cb_select_rows_small(tmp_path, monkeypatch):
    a = args(("--cpu-only", "--targets", "Lena:x"), tmp_path)
    X = N._load_h42()
    monkeypatch.setattr(X, "TUNE_GRID", [(3, 20), (4, 30)])
    D = FakeD()
    inject_src(a, D)
    rows = N.cb_select_rows(a, D, "Lena", "x")
    assert len(rows) == 2 * 2 and all(r_["n_fit"] == 3 for r_ in rows)
    df = N.write_cb_select(a, rows)
    assert df.groupby("kind").selected.sum().tolist() == [1, 1]
