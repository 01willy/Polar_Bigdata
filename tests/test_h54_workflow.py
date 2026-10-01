"""scripts/3_deep_learning/h54_workflow.py 단위 시험(합성 자료, 실제 자료를 읽지 않는다).

(a) 라벨 위치 전략 S1–S7 은 A 색인 범위 안의 서로 다른 정수를 요청한 수만큼 돌려주고, 같은 seed 에서 같은 결과를 낸다.
    S5 는 결정적이라 추출 사이에 같다. S6 의 선택은 선택된 라벨만 쓴다(비선택 A 라벨을 바꿔도 같다).
(b) 지역 내 실험(wf1 x25·x34, wf2 S6, wf3)의 누설 점검: 적합·계수·교차검증·크리깅·선택이 쓴 라벨 색인은 모두 그 추출의 선택 라벨 안에 있고,
    B 라벨과 비선택 A 라벨을 바꿔도 저장한 모든 예측이 같다.
(c) 채점은 B 블록 셀만 쓴다: 저장소의 블록 = B 블록, P0 의 블록별 SSE 가 직접 계산과 같다.
(d) 전이 실험(wf4)의 규칙 W 는 선택 라벨만으로 고른다: 교차검증 묶음의 학습·평가 색인이 선택 라벨 안에 있고, B 라벨을 바꿔도 선택과 예측이 같다.
    W 의 예측은 고른 후보의 예측과 같다. n = 10 의 편향 진단값은 선택 라벨의 E0·s − y 평균이다.
(e) 같은 설정을 두 번 돌리면 저장소가 같다(seed 결정성).
(f) 국소 크리깅은 같은 변이도 모수에서 pykrige(backend loop, n_closest_points)와 예측이 같다. 변이도 적합은 유한한 모수를 낸다.
(g) dry 실행은 학습 없이 적합 수와 추정 시간을 센다.
(h) 조각 기록과 --resume 상태(설정 해시 불일치는 다시 실행).
(i) 허용 표지 없는 본 실행은 자료를 읽기 전에 거부된다.
(j) [HEAVY] 합성 조각으로 집계(곡선, 가설 표, meta)가 끝까지 돈다. LG_RUN_HEAVY=1 일 때만 실행한다.
(k) WF5 우선순위 함수는 전략마다 후보 색인 범위 안의 서로 다른 정수를 요청한 수만큼 돌려준다.
2차 보강(WF6–WF8)
(l) wf6 의 새 분할(split_seed 6–25)은 결정적이고 서로 다르며, 분할 1–25 안에서 계산한 분할 1–5 의 구조는 분할 1–5 만으로 계산한 것과 같다.
    중복 분할과 무효 분할은 h40 의 규칙으로 빠진다.
(m) wf6 의 누설 점검(선택 라벨 밖의 A 라벨과 B 라벨을 바꿔도 모든 예측이 같다)과 방법 구성(D0 은 catboost 만, R1·R2·Re 의 교차검증 λ).
(n) wf7 은 n = 0 을 포함한 LG 추출을 쓰고, Pe = 앵커(E_n), Re·R1 의 학습 행(원천 목표와 선택 라벨 목표)이 정의와 같으며 라벨 누설이 없다.
(o) wf8 의 전략(S1, S2, S4)은 전이 모드에서 A 색인 범위 안의 서로 다른 정수를 결정적으로 고르고, S1·S2 = LG 추출, S4 = wf2 의 S4 다.
    선택과 예측은 선택 라벨만 쓰고, |A| 가 작은 대상은 n = 10 만 두며, 같은 라벨 집합의 추출은 적합을 재사용한다.
(p) 1차(wf1–wf4)의 설정 해시, 공통 해시, 조각 이름, 기본 대상·격자가 2차 보강 추가 전의 값(커밋 1b42b4d 의 하네스)과 같다.
(q) [HEAVY] 합성 조각으로 2차 보강의 집계(wf2b_* 표, WF6–WF8 판정 행)가 끝까지 돌고 1차 표를 쓰지 않는다.
(r) --exp 별칭, 차수별 표 이름, 2차 보강의 기본 대상(wf7 = LG 27)과 n = 0 규칙.
3차 보강(WF9·WF10)
(s) 격자 분해: 블록마다 총 SSE = 격자 안 SSE + 격자 사이 SSE, 묶음 안 상수 예측(P1)의 격자 안 SSE = 묶음 안 실측 제곱합, 혼자 묶음은 격자 안에서 빠진다.
(t) wf10 분할: W 는 블록 평균 √TDD 의 가장 따뜻한(cold 는 가장 추운) 블록이고 셀 25 % 이상, W·I·A 는 겹치지 않고 대상 셀을 덮는다.
    W 는 분할 사이에 같고 I 는 분할마다 다르며 같은 분할은 결정적이다.
(u) wf9 지역 내 단위(R9Unit): 세 저장소(총, ~w, ~b)의 키가 같고 분해 항등식이 모든 키에서 성립한다. 방법 구성과 누설 점검(선택 밖 라벨을 바꿔도 예측이 같다).
(v) wf9x 전이 단위(T9Unit): n = 0 을 포함하고 R1·R2·D0·D1 키가 있으며 분해 항등식이 성립한다.
(w) wf10 단위(R10Unit): W·I 저장소는 채점 셀을 나눈 것이고 SSE 의 합이 총 저장소와 같다. 외삽 손실 EP = Δ_W − Δ_I(점 추정과 분포).
(x) [HEAVY] 합성 조각으로 3차 집계(wf3b_* 표, WF9·WF10 판정 행, 서술 표)가 끝까지 돌고 1·2차 표를 쓰지 않는다.
(y) --exp wf9,wf9x,wf10 의 차수(r3)와 표 이름, 기본 실행 목록(1·2차)은 그대로이고, 스모크·사전 점검 설정.
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c <코어 4개> python3 -m pytest -q tests/test_h54_workflow.py
(CatBoost 는 thread_count 밖의 고정 비용 부분도 여러 스레드로 돌린다. 공유 서버에서는 taskset 으로 코어를 묶는다)
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
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "3_deep_learning"))
_spec = importlib.util.spec_from_file_location("h54_workflow", ROOT / "scripts" / "3_deep_learning" / "h54_workflow.py")
W = importlib.util.module_from_spec(_spec)
sys.modules["h54_workflow"] = W
_spec.loader.exec_module(W)
H, X = W.H, W.X

HEAVY = pytest.mark.skipif(os.environ.get("LG_RUN_HEAVY", "") != "1", reason="집계 통합 시험은 LG_RUN_HEAVY=1 일 때만 실행한다(공유 서버 CPU 보호)")
X.CB_HI = dict(iterations=10, learning_rate=0.1, depth=3, l2_leaf_reg=3.0)        # 시험 속도(설정 해시에 들어가지만 시험 안에서만 쓴다)
ARGS = ["--threads", "1", "--cb-iters", "10", "--seeds", "1", "--draws-cap", "2", "--wf1-grid", "20,50,all", "--wf2-budgets", "20,40",
        "--wf3-grid", "40,all", "--wf4-grid", "10,40,all", "--nboot", "200",
        "--wf6-grid", "20,40,all", "--wf7-grid", "0,3,10,all", "--wf8-grid", "10,20"]


def args(extra=()):
    return W.parse_args(ARGS + list(extra))


def make_rctx(seed=0, split=1, target="T", nA=160, nB=120, nbA=10, nbB=10, y_shift=None):
    """합성 지역 내 문맥. A 블록 0..nbA−1, B 블록 1000.. 으로 겹치지 않는다. cci 열은 일부 결측."""
    rng = np.random.RandomState(seed)
    D = len(W.FEATS)
    XA = rng.randn(nA, D).astype(np.float32); XB = rng.randn(nB, D).astype(np.float32)
    for Xm, k in ((XA, nA), (XB, nB)):
        Xm[:, W.CCI_COL] = 40 + 10 * rng.randn(k); Xm[:, W.CCIV_COL] = 1.0
    XA[:7, W.CCI_COL] = np.nan; XA[:7, W.CCIV_COL] = 0.0
    sA, sB = rng.uniform(20, 40, nA), rng.uniform(20, 40, nB)
    yA = 1.4 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    yB = 1.4 * sB + 3 * XB[:, 1] + 2 * rng.randn(nB)
    blkA = np.repeat(np.arange(nbA), int(np.ceil(nA / nbA)))[:nA]; blkB = 1000 + np.repeat(np.arange(nbB), int(np.ceil(nB / nbB)))[:nB]
    latA = 65 + 0.5 * (blkA % 5) + 0.05 * rng.rand(nA); lonA = -150 + 0.5 * (blkA // 5) + 0.05 * rng.rand(nA)
    latB = 68 + 0.5 * ((blkB - 1000) % 5) + 0.05 * rng.rand(nB); lonB = -150 + 0.5 * ((blkB - 1000) // 5) + 0.05 * rng.rand(nB)
    sar = lambda k: np.where(rng.rand(k, len(W.SAR_COLS)) < 0.2, np.nan, rng.randn(k, len(W.SAR_COLS)))   # noqa: E731
    XA34 = np.hstack([XA, sar(nA)]).astype(np.float32); XB34 = np.hstack([XB, sar(nB)]).astype(np.float32)
    if y_shift is not None:
        yA, yB = y_shift(yA.copy(), yB.copy())
    meta = dict(dup_of=-1, valid=True, n_valid_splits=2, n_unique_splits=2)
    return W.RCtx(target, split, target, XA, yA, sA, blkA, latA, lonA, XB, yB, sB, blkB, latB, lonB, 1.6, XA34=XA34, XB34=XB34,
                  src_X=rng.randn(200, D) + 0.5, meta=meta)


def make_tctx(seed=0, split=1, target="T", mode="x", y_shift=None):
    """합성 전이 문맥(h40.Ctx). 원천 300행(두 지역), A 80셀 8블록, B 60셀 6블록."""
    rng = np.random.RandomState(seed)
    D = len(W.FEATS)
    ns, nA, nB = 300, 80, 60
    Xs = rng.randn(ns, D); ss = rng.uniform(20, 40, ns); ys = 1.6 * ss + 3 * Xs[:, 1] + 2 * rng.randn(ns)
    XA = rng.randn(nA, D); sA = rng.uniform(20, 40, nA); yA = 1.3 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    XB = rng.randn(nB, D); sB = rng.uniform(20, 40, nB); yB = 1.3 * sB + 3 * XB[:, 1] + 2 * rng.randn(nB)
    if y_shift is not None:
        yA, yB = y_shift(yA.copy(), yB.copy())
    c = H.Ctx(target, mode, split, target, Xs, ys, ss, np.repeat(["r1", "r2"], ns // 2), XA, yA, sA, np.repeat(np.arange(8), nA // 8),
              XB, yB, sB, 500 + np.repeat(np.arange(6), nB // 6), meta=dict(dup_of=-1, valid=True))
    return c


def run_r(a, c, exp, variant=""):
    U = W.RUnit(a, c, exp, variant).run()
    rows, st, stats = U.finish()
    return U, rows, st, stats


def sel_sets(trace):
    """추적 기록에서 (n, 추출) → 선택 라벨 집합."""
    out = {}
    for e in trace:
        if e["kind"] in ("coef", "select"):
            out.setdefault((int(e["n"]), str(e["draw"])), set()).update(int(v) for v in e["idx"])
    return out


# ---------------------------------------------------------------- (a) 전략
@pytest.mark.parametrize("strat", list(W.STRATEGIES))
def test_a_strategies_valid_and_deterministic(strat):
    a = args()
    c = make_rctx()
    sets = {d: W.RUnit(a, c, "wf2", strat).strategy_sets(strat, [20, 40], d) for d in range(3)}
    again = {d: W.RUnit(a, c, "wf2", strat).strategy_sets(strat, [20, 40], d) for d in range(3)}
    for d in range(3):
        for n in (20, 40):
            s_ = np.asarray(sets[d][n])
            assert len(s_) == n and len(np.unique(s_)) == n, f"{strat} n={n}: 개수 또는 중복"
            assert s_.dtype.kind in "iu" and s_.min() >= 0 and s_.max() < len(c.yA), f"{strat}: A 색인 범위 밖"
            assert np.array_equal(np.sort(s_), np.sort(np.asarray(again[d][n]))), f"{strat}: 같은 seed 에서 결과가 다르다"
    if strat == "S5":
        assert all(np.array_equal(np.sort(sets[0][40]), np.sort(sets[d][40])) for d in range(3)), "S5 는 결정적이어야 한다"
    if strat == "S1":
        assert np.array_equal(np.sort(sets[0][20]), H.draw_cells("T", "r", 1, 20, 0, len(c.yA)))
    if strat == "S6":                                                         # 선택 적합이 실패해 무작위로 대체되지 않았다
        U = W.RUnit(a, c, "wf2", strat)
        U.strategy_sets(strat, [20, 40], 0)
        assert not U.notes.get("s6_fallback") and not U.F.fail, f"S6 선택 적합 실패: {U.F.errors}"


def test_a2_s6_uses_only_selected_labels():
    a = args()
    c = make_rctx()
    base = W.RUnit(a, c, "wf2", "S6").strategy_sets("S6", [20, 40], 0)
    used = set(base[40].tolist())
    rng = np.random.RandomState(5)

    def shift(yA, yB):
        m = np.array([i not in used for i in range(len(yA))])
        yA[m] = rng.uniform(1, 500, m.sum()); yB[:] = rng.uniform(1, 500, len(yB))
        return yA, yB
    c2 = make_rctx(y_shift=shift)
    other = W.RUnit(a, c2, "wf2", "S6").strategy_sets("S6", [20, 40], 0)
    assert np.array_equal(base[40], other[40]) and np.array_equal(base[20], other[20])
    assert np.array_equal(base[20], H.draw_cells("T", "r", 1, 20, 0, len(c.yA))), "S6 의 시작은 S1 의 n = 20 추출이다"


# ---------------------------------------------------------------- (b) 지역 내 누설
@pytest.mark.parametrize("exp,variant", [("wf1", ""), ("wf1", "x34"), ("wf2", "S6"), ("wf2", "S3"), ("wf3", "")])
def test_b_inregion_no_leakage(exp, variant):
    a = args()
    W.WF_TRACE = []; W.PRED_TRACE = {}
    try:
        run_r(a, make_rctx(), exp, variant)
        tr, p1 = list(W.WF_TRACE), dict(W.PRED_TRACE)
    finally:
        W.WF_TRACE = None; W.PRED_TRACE = None
    sets = sel_sets(tr)
    assert sets, "선택 기록이 없다"
    allsel = set().union(*sets.values())
    for e in tr:
        idx = set(int(v) for v in e["idx"])
        if e["kind"] in ("fit", "cv", "krige", "s6"):
            assert idx <= allsel, f"{e['kind']} {e.get('method')}: 선택하지 않은 A 라벨을 썼다"
            key = (int(e["n"]), str(e["draw"]).split(".")[0]) if e["kind"] != "s6" else None
            if key in sets:
                assert idx <= sets[key], f"{e['kind']} {e.get('method')} n={e['n']}: 다른 추출의 라벨을 썼다"
        if e["kind"] == "cv":
            assert set(int(v) for v in e["held"]) <= allsel and not (set(int(v) for v in e["held"]) & idx), "교차검증 묶음이 겹친다"
    rng = np.random.RandomState(9)

    def shift(yA, yB):
        m = np.array([i not in allsel for i in range(len(yA))])
        yA[m] = rng.uniform(1, 500, m.sum()); yB[:] = rng.uniform(1, 500, len(yB))
        return yA, yB
    W.PRED_TRACE = {}
    try:
        run_r(a, make_rctx(y_shift=shift), exp, variant)
        p2 = dict(W.PRED_TRACE)
    finally:
        W.PRED_TRACE = None
    assert set(p1) == set(p2) and len(p1) > 5
    for k in p1:
        assert np.allclose(p1[k], p2[k], rtol=0, atol=1e-9), f"B 라벨 또는 비선택 A 라벨이 예측 {k} 에 영향을 준다"


# ---------------------------------------------------------------- (c) 채점 = B 블록
def test_c_scoring_only_B():
    a = args()
    c = make_rctx()
    U, rows, st, stats = run_r(a, c, "wf3")
    assert set(st.blocks.tolist()) == set(np.unique(c.blkB).astype(str).tolist())
    assert not (set(np.unique(c.blkA).astype(str)) & set(st.blocks.tolist()))
    sse, cnt = st.get(H.P0_KEY)
    want = {str(b): float(np.sum((c.E0 * c.sB[c.blkB == b] - c.yB[c.blkB == b]) ** 2)) for b in np.unique(c.blkB)}
    for b, s_, n_ in zip(st.blocks, sse, cnt):
        assert np.isclose(s_, want[b]) and n_ == int((c.blkB.astype(str) == b).sum())
    assert stats["status"] == "ok" and stats["n_stored_ml"] > 0


# ---------------------------------------------------------------- (d) 전이·규칙 W
def test_d_transfer_selector_uses_only_A():
    a = args()
    W.WF_TRACE = []; W.PRED_TRACE = {}
    try:
        U = W.TUnit(a, make_tctx(), "T").run()
        rows, st, stats = U.finish()
        tr, p1 = list(W.WF_TRACE), dict(W.PRED_TRACE)
    finally:
        W.WF_TRACE = None; W.PRED_TRACE = None
    sets = sel_sets(tr)
    for e in tr:
        if e["kind"] in ("fit", "cv"):
            key = (int(e["n"]), str(e["draw"]))
            assert set(int(v) for v in e["idx"]) <= sets[key]
            if e["kind"] == "cv":
                held = set(int(v) for v in e["held"])
                assert held <= sets[key] and not (held & set(int(v) for v in e["idx"]))
    wr = [r for r in rows if r["method"] == "W"]
    assert wr and all(r["alpha_sel"] in W.W_CANDS for r in wr)
    mp = {"P0": ("P0", "none", "1", "cell", 0, 0, -1, 0.0)}
    for r in wr:                                                              # W 의 예측 = 고른 후보의 예측
        n, d, s_ = r["n"], r["draw"], r["seed"]
        ch = r["alpha_sel"]
        if ch == "P0":
            k = mp["P0"]
        elif ch in ("P1", "P2"):
            k = (ch, "none", "1", "cell", n, d, -1, 0.0)
        elif ch == "D1":
            k = ("D1", W.LO, "1", "cell", n, d, s_, 1.0)
        else:
            m, lam = ch.split("@")
            k = (m, W.LO, "1", "cell", n, d, s_, float(lam))
        kw = ("W", W.LO, "1", "cell", n, d, s_, 0.0)
        assert np.allclose(p1[("wf4", "T|x", 1, "") + kw], p1[("wf4", "T|x", 1, "") + k])
    assert stats["diag"], "n = 10 의 편향 진단값이 없다"
    c = make_tctx()
    for dg in stats["diag"]:
        sel = H.draw_cells("T", "x", 1, 10, dg["draw"], len(c.yA))
        assert np.isclose(dg["bias"], np.mean(c.E0 * c.sA[sel] - c.yA[sel]))
    rng = np.random.RandomState(3)

    def shift(yA, yB):
        yB[:] = rng.uniform(1, 500, len(yB))
        return yA, yB
    W.PRED_TRACE = {}
    try:
        U2 = W.TUnit(a, make_tctx(y_shift=shift), "T").run()
        rows2, _, _ = U2.finish()
        p2 = dict(W.PRED_TRACE)
    finally:
        W.PRED_TRACE = None
    assert [r["alpha_sel"] for r in rows if r["method"] == "W"] == [r["alpha_sel"] for r in rows2 if r["method"] == "W"]
    for k in p1:
        assert np.allclose(p1[k], p2[k], rtol=0, atol=1e-9), f"B 라벨이 {k} 에 영향을 준다"


def test_d2_w_rule_small_n_is_p1():
    a = args(["--wf4-grid", "5"])
    U = W.TUnit(a, make_tctx(), "T").run()
    rows, _, _ = U.finish()
    assert {r["alpha_sel"] for r in rows if r["method"] == "W"} == {"P1"}


# ---------------------------------------------------------------- (e) 결정성
def test_e_deterministic_rerun():
    a = args()
    _, _, st1, _ = run_r(a, make_rctx(), "wf1")
    _, _, st2, _ = run_r(a, make_rctx(), "wf1")
    assert st1.keys == st2.keys
    S1, C1 = st1.matrices(st1.keys); S2, C2 = st2.matrices(st2.keys)
    assert np.array_equal(C1, C2) and np.allclose(S1, S2, rtol=0, atol=1e-9)


# ---------------------------------------------------------------- (f) 크리깅
def test_f_kriging_matches_pykrige():
    pk = pytest.importorskip("pykrige.ok")
    rng = np.random.RandomState(0)
    P = rng.uniform(0, 100, (80, 2)); z = np.sin(P[:, 0] / 15) + 0.1 * rng.randn(80)
    Q = rng.uniform(0, 100, (40, 2))
    vp = dict(psill=1.0, range=30.0, nugget=0.1)
    mine = W.ok_predict(P, z, Q, vp, k=16)
    ok = pk.OrdinaryKriging(P[:, 0], P[:, 1], z, variogram_model="exponential",          # 목록은 [sill, range, nugget] 이라 사전으로 준다
                            variogram_parameters=dict(psill=1.0, range=30.0, nugget=0.1))
    ref, _ = ok.execute("points", Q[:, 0], Q[:, 1], backend="loop", n_closest_points=16)
    assert np.allclose(mine, np.asarray(ref), atol=1e-6)


def test_f2_variogram_fit_finite():
    rng = np.random.RandomState(1)
    lat, lon = 65 + rng.rand(300) * 3, -150 + rng.rand(300) * 6
    P = W.xyz_km(lat, lon)
    z = np.sin(P[:, 0] / 80.0) + 0.2 * rng.randn(300)
    vp = W.fit_variogram(P, z, 0)
    assert vp is not None and all(np.isfinite([vp["psill"], vp["range"], vp["nugget"]])) and vp["range"] > 0
    p, _ = W.krige(P, z, P[:5] + 1.0, 0)
    assert p is not None and np.all(np.isfinite(p))
    assert W.fit_variogram(P[:5], z[:5], 0) is None, "라벨 10개 미만은 크리깅하지 않는다"


# ---------------------------------------------------------------- (g) dry
def test_g_dry_counts_without_training():
    a = args()
    U = W.RUnit(a, make_rctx(), "wf1", "", dry=True).run()
    _, _, stats = U.finish()
    assert sum(stats["n_fit"].values()) > 0 and sum(stats["est_detail"].values()) > 0 and stats["n_krige"] > 0
    assert not stats["sec"], "dry 실행이 학습했다"
    U4 = W.TUnit(a, make_tctx(), "T", dry=True).run()
    _, _, s4 = U4.finish()
    assert s4["n_fit"]["wf4"] > 0 and not s4["sec"]


# ---------------------------------------------------------------- (h) 조각과 재개
def test_h_shard_and_resume(tmp_path):
    a = args(["--out-dir", str(tmp_path)])
    c = make_rctx()
    U, rows, st, stats = run_r(a, c, "wf3")
    W.write_shard(a, "wf3", "T", "r", 1, "", c, rows, st, stats, 1.0, [1])
    ok, why = W.unit_state(a, "wf3", "T", "r", 1, "")
    assert ok and why == "ok"
    a2 = args(["--out-dir", str(tmp_path), "--wf3-grid", "40"])
    ok2, why2 = W.unit_state(a2, "wf3", "T", "r", 1, "")
    assert not ok2 and "cfg_hash" in why2
    u = json.loads(W.shard_paths(a, "wf3", "T", "r", 1)["unit"].read_text())
    assert u["expected_splits"] == [1] and u["cfg_hash"] and u["exp"] == "wf3"


# ---------------------------------------------------------------- (i) 실행 거부
def test_i_refuses_without_permission(monkeypatch):
    monkeypatch.delenv("WF_RESCALE", raising=False)
    monkeypatch.delenv("LG_RESCALE", raising=False)
    monkeypatch.setattr(sys, "argv", ["h54"])
    with pytest.raises(SystemExit):
        W.main(["--threads", "1"])
    a = W.parse_args(["--threads", "16"])
    assert a.threads <= W.LOCAL_MAX_THREADS


# ---------------------------------------------------------------- (j) 집계 통합(HEAVY)
@HEAVY
def test_j_summarize_end_to_end(tmp_path):
    a = args(["--out-dir", str(tmp_path), "--strategies", "S1,S5,S6", "--wf2-budgets", "20,40"])
    for split in (1, 2):
        for tgt in ("Alaska", "Lena", "Canada"):
            c = make_rctx(seed=split * 10 + len(tgt), split=split, target=tgt)
            jobs = [("wf1", ""), ("wf2", "S1"), ("wf2", "S5"), ("wf2", "S6"), ("wf3", "")] + ([("wf1", "x34")] if tgt == "Alaska" else [])
            for exp, var in jobs:
                U, rows, st, stats = run_r(a, c, exp, var)
                W.write_shard(a, exp, tgt, "r", split, var, c, rows, st, stats, 1.0, [1, 2])
        for tgt in ("Lena", "Canada", "Russia_W", "Russia_E"):
            c = make_tctx(seed=split * 7 + len(tgt), split=split, target=tgt)
            U = W.TUnit(a, c, tgt).run()
            rows, st, stats = U.finish()
            W.write_shard(a, "wf4", tgt, "x", split, "", c, rows, st, stats, 1.0, [1, 2])
    out = W.summarize(a)
    assert out is not None
    t = out["tests"]
    for hyp in ("WF1-a", "WF1-b", "WF1-c", "WF2-a", "WF3-a", "WF4-a", "WF4-b", "WF4-c"):
        v = t[(t.test_id == hyp) & t.scope.isin(["verdict", "verdict_aux"])]
        assert len(v), f"{hyp} 판정 행이 없다"
        assert not v.verdict.astype(str).str.contains("계산 실패").any()
    for f in ("wf_curve.csv", "wf_tests.csv", "wf_meta.json", "wf_timing.csv"):
        assert (tmp_path / f).exists()
    assert len(out["curve"]) > 0 and set(out["curve"].exp) >= {"wf1", "wf2", "wf3", "wf4"}


# ---------------------------------------------------------------- (k) WF5 우선순위
@pytest.mark.parametrize("strat", list(W.STRATEGIES))
def test_k_wf5_priority(strat):
    rng = np.random.RandomState(0)
    N, D = 300, len(W.FEATS)
    Xc = rng.randn(N, D); sc = rng.uniform(20, 40, N); lat = 70 + rng.rand(N) * 3; lon = 125 + rng.rand(N) * 5
    Xl = rng.randn(40, D); sl = rng.uniform(20, 40, 40); yl = 1.4 * sl + rng.randn(40)
    out = W.wf5_priority(strat, Xc, sc, lat, lon, Xl, yl, sl, 1.5, 30, 0, src_X=rng.randn(200, D), threads=1, cb_iters=10)
    o = np.asarray(out["order"])
    assert len(o) == 30 and len(np.unique(o)) == 30 and o.min() >= 0 and o.max() < N
    again = W.wf5_priority(strat, Xc, sc, lat, lon, Xl, yl, sl, 1.5, 30, 0, src_X=rng.randn(200, D) * 0 + 1, threads=1, cb_iters=10)
    if strat not in ("S5",):                                                  # S5 만 원천 공변량에 의존한다
        assert np.array_equal(np.sort(o), np.sort(np.asarray(again["order"])))


# ================================================================ 2차 보강(WF6–WF8)
def make_tctx2(seed=0, split=1, target="T", mode="x", nA=80, nB=60, nbA=8, nbB=6, y_shift=None):
    """CCI 열이 있는 합성 전이 문맥(h40.Ctx). 원천 300행(두 지역), cci_alt ~ 40 ± 10, cci_valid 1, 원천·A·B 의 일부 셀은 CCI 무효(NaN 또는 0)."""
    rng = np.random.RandomState(seed)
    D = len(W.FEATS)
    ns = 300
    Xs = rng.randn(ns, D); ss = rng.uniform(20, 40, ns); ys = 1.6 * ss + 3 * Xs[:, 1] + 2 * rng.randn(ns)
    XA = rng.randn(nA, D); sA = rng.uniform(20, 40, nA); yA = 1.3 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    XB = rng.randn(nB, D); sB = rng.uniform(20, 40, nB); yB = 1.3 * sB + 3 * XB[:, 1] + 2 * rng.randn(nB)
    for Xm in (Xs, XA, XB):
        k = len(Xm)
        Xm[:, W.CCI_COL] = 40 + 10 * rng.randn(k); Xm[:, W.CCIV_COL] = 1.0
        Xm[: max(2, k // 10), W.CCI_COL] = np.nan                         # 무효: cci_alt 결측
        Xm[max(2, k // 10): max(4, k // 5), W.CCIV_COL] = 0.0              # 무효: cci_valid < 0.5
    if y_shift is not None:
        yA, yB = y_shift(yA.copy(), yB.copy())
    blkA = np.repeat(np.arange(nbA), int(np.ceil(nA / nbA)))[:nA]
    blkB = 500 + np.repeat(np.arange(nbB), int(np.ceil(nB / nbB)))[:nB]
    return H.Ctx(target, mode, split, target, Xs, ys, ss, np.repeat(["r1", "r2"], ns // 2), XA, yA, sA, blkA, XB, yB, sB, blkB,
                 meta=dict(dup_of=-1, valid=True))


def _shift_outside(sel_all, seed):
    """선택 라벨(sel_all) 밖의 A 라벨과 모든 B 라벨을 바꾸는 y_shift."""
    rng = np.random.RandomState(seed)

    def shift(yA, yB):
        m = np.array([i not in sel_all for i in range(len(yA))])
        yA[m] = rng.uniform(1, 500, m.sum()); yB[:] = rng.uniform(1, 500, len(yB))
        return yA, yB
    return shift


def _assert_B_only(st, c):
    """채점은 B 블록 셀만 쓴다: 저장소의 블록 = B 블록(A 블록과 겹치지 않음), P0 의 블록별 SSE·셀 수 = 직접 계산."""
    blocks = set(st.blocks.tolist())
    assert blocks == set(np.unique(c.blkB).astype(str).tolist()) and not (set(np.unique(c.blkA).astype(str).tolist()) & blocks)
    sse, cnt = st.get(H.P0_KEY)
    for b, s_, n_ in zip(st.blocks, sse, cnt):
        m = c.blkB.astype(str) == b
        assert np.isclose(s_, float(np.sum((c.E0 * c.sB[m] - c.yB[m]) ** 2))) and n_ == int(m.sum())


def _assert_same_preds(p1, p2, what):
    assert set(p1) == set(p2) and len(p1) > 3, f"{what}: 저장 키가 다르다"
    for k in p1:
        assert np.allclose(p1[k], p2[k], rtol=0, atol=1e-9), f"{what}: 선택 밖 라벨이 예측 {k} 에 영향을 준다"


# ---------------------------------------------------------------- (l) wf6 분할
def _split_df(n_blocks, cells_per_block=6, seed=0):
    import pandas as pd
    rng = np.random.RandomState(seed)
    blk = np.repeat(np.arange(n_blocks), cells_per_block)
    n = len(blk)
    return pd.DataFrame({"block": blk, "alt_cm": rng.uniform(30, 120, n), "cci_alt": rng.uniform(30, 120, n),
                         "e5_sqrt_tdd_soil": rng.uniform(20, 40, n)})


def test_l_wf6_splits_deterministic_distinct():
    from types import SimpleNamespace
    from polar.m1_core import half_split_blocks
    df = _split_df(40)
    idx = np.arange(len(df))
    D = SimpleNamespace(df=df, target_idx=lambda t: idx)
    sp25 = list(range(1, 26))
    info = W.split_structure_ext(D, "T", sp25)
    again = W.H.Data.split_structure(W._SplitView(D, sp25), "T")
    assert info == again, "같은 분할 목록에서 구조가 다르다(결정성)"
    first5 = W.H.Data.split_structure(W._SplitView(D, range(1, 6)), "T")
    for sp in range(1, 6):                                                # 분할 1–5 의 구조는 LG(분할 1–5 만)와 같다
        drop = ("n_unique_splits", "n_valid_splits")
        assert {k: v for k, v in info[sp].items() if k not in drop} == {k: v for k, v in first5[sp].items() if k not in drop}
    A_sets = []
    for sp in sp25:
        A1, B1 = half_split_blocks(df, idx, sp)
        A2, B2 = half_split_blocks(df, idx, sp)
        assert np.array_equal(A1, A2) and np.array_equal(B1, B2), "새 분할이 결정적이지 않다"
        assert not (set(df.block.values[A1]) & set(df.block.values[B1])), "A·B 블록이 겹친다"
        A_sets.append(frozenset(df.block.values[A1]))
    assert len(set(A_sets)) == 25, "분할 1–25 의 A 블록 집합이 서로 다르지 않다"
    keep, skip, _ = W.split_plan(D, "T", sp25, info)
    assert keep == sp25 and not skip
    # 블록이 3개인 작은 지역: 중복 분할은 빠지고 남은 분할의 A 집합은 서로 다르다
    df3 = _split_df(3)
    idx3 = np.arange(len(df3))
    D3 = SimpleNamespace(df=df3, target_idx=lambda t: idx3)
    info3 = W.split_structure_ext(D3, "T3", sp25)
    keep3, skip3, _ = W.split_plan(D3, "T3", sp25, info3)
    dups = [sp for sp, st_, _ in skip3 if st_.startswith("dup_of")]
    assert dups and len(keep3) < 25
    kept_sets = [frozenset(df3.block.values[half_split_blocks(df3, idx3, sp)[0]]) for sp in keep3]
    assert len(set(kept_sets)) == len(kept_sets), "남은 분할에 중복이 있다"
    for sp in dups:
        assert info3[sp]["dup_of"] in sp25 and info3[sp]["dup_of"] < sp
    # 채점 블록 2개 미만(무효) 분할은 유효 분할이 있으면 빠진다
    inval = [sp for sp in keep3 if not info3[sp]["valid"]]
    if any(info3[sp]["valid"] for sp in sp25):
        assert not inval


# ---------------------------------------------------------------- (m) wf6 누설과 방법 구성
def test_m_wf6_no_leakage_and_methods():
    a = args(["--wf6-grid", "20,40"])
    W.WF_TRACE = []; W.PRED_TRACE = {}
    try:
        U, rows, st, stats = run_r(a, make_rctx(), "wf6")
        tr, p1 = list(W.WF_TRACE), dict(W.PRED_TRACE)
    finally:
        W.WF_TRACE = None; W.PRED_TRACE = None
    _assert_B_only(st, make_rctx())
    sets = sel_sets(tr)
    allsel = set().union(*sets.values())
    assert len(allsel) < len(make_rctx().yA), "선택 밖 A 라벨이 있어야 시험이 의미가 있다"
    for e in tr:
        idx = set(int(v) for v in e["idx"])
        if e["kind"] in ("fit", "cv", "krige"):
            key = (int(e["n"]), str(e["draw"]).split(".")[0])
            assert idx <= sets[key], f"{e['kind']} {e.get('method')}: 그 추출의 선택 라벨 밖을 썼다"
        if e["kind"] == "cv":
            assert not (set(int(v) for v in e["held"]) & idx), "교차검증 묶음이 겹친다"
    W.PRED_TRACE = {}
    try:
        run_r(a, make_rctx(y_shift=_shift_outside(allsel, 11)), "wf6")
        p2 = dict(W.PRED_TRACE)
    finally:
        W.PRED_TRACE = None
    _assert_same_preds(p1, p2, "wf6")
    keys = set(st.keys)
    meth = {(k[0], k[1]) for k in keys}
    assert ("D0", W.HI) in meth and ("D0", W.LO) not in meth, "wf6 의 D0 은 catboost 만이다"
    assert not any(k[0] in ("D1", "RK") or k[0].startswith("R1c") for k in keys)
    for m in ("R1", "R2", "Re"):
        assert any(k[0] == m and k[1] == W.LO and k[7] == W.LAM_CV for k in keys), f"{m} 의 교차검증 λ 키가 없다"
    for m in ("P1", "Pk", "Pbest", "P2", "Pc"):
        assert any(k[0] == m for k in keys), f"{m} 키가 없다"
    lam_sel = {r["alpha_sel"] for r in rows if r["method"] in ("R1", "R2", "Re") and r["lam"] == W.LAM_CV}
    assert lam_sel and lam_sel <= {f"lam={v}" for v in W.LAMS}
    assert stats["status"] == "ok"


# ---------------------------------------------------------------- (n) wf7
def test_n_wf7_anchor_rows_and_no_leakage():
    a = args(["--wf7-grid", "0,3,10"])
    c = make_tctx2()
    W.WF_TRACE = []; W.PRED_TRACE = {}; H.TRACE = []
    try:
        U = W.T7Unit(a, c, "T").run()
        rows, st, stats = U.finish()
        tr, p1, ht = list(W.WF_TRACE), dict(W.PRED_TRACE), list(H.TRACE)
    finally:
        W.WF_TRACE = None; W.PRED_TRACE = None; H.TRACE = None
    assert stats["status"] == "ok"
    _assert_B_only(st, c)
    sets = sel_sets(tr)
    assert set(sets) == {(0, "0"), (3, "0"), (3, "1"), (10, "0"), (10, "1")}, "n = 0 과 LG 추출(추출 상한 2)이어야 한다"
    for (n, d), s_ in sets.items():                                        # 추출 = LG 의 h40.draw_cells
        assert s_ == set(H.draw_cells("T", "x", 1, n, int(d), len(c.yA)).tolist())
    nsrc = len(c.y_src)
    re_src = W.re_src_resid(c)
    cci_ok = np.isfinite(c.X_src[:, W.CCI_COL]) & (c.X_src[:, W.CCIV_COL] >= 0.5)
    assert np.allclose(re_src[~cci_ok], c.r0_src[~cci_ok]) and not np.allclose(re_src[cci_ok], c.r0_src[cci_ok])
    seen = set()
    for h in ht:                                                          # 학습 행: 원천(목표 = y − 앵커(E0)) ∪ 선택 라벨(목표 = y − 앵커(E_n))
        sel = np.asarray(h["sel"], int)
        E_n = W.X.shrink(H.ls_E(c.yA[sel], c.sA[sel]), c.E0, len(sel), W.KAPPA) if len(sel) else c.E0
        assert len(h["ytr"]) == nsrc + len(sel)
        if h["method"] == "Re":
            assert np.allclose(h["ytr"][:nsrc], re_src)
            want = c.yA[sel] - W.re_anchor(E_n * c.sA[sel], c.XA[sel, W.CCI_COL], c.XA[sel, W.CCIV_COL])
        else:
            assert h["method"] == "R1" and np.allclose(h["ytr"][:nsrc], c.r0_src)
            want = c.yA[sel] - E_n * c.sA[sel]
        assert np.allclose(h["ytr"][nsrc:], want)
        seen.add((h["method"], int(h["n"])))
    assert ("Re", 0) in seen and ("R1", 0) in seen, "n = 0 의 적합(원천 행만)이 없다"
    pre = ("wf7", "T|x", 1, "")
    p0 = p1[pre + ("P0", "none", "1", "cell", 0, 0, -1, 0.0)]
    assert np.allclose(p1[pre + ("P1", "none", "1", "cell", 0, 0, -1, 0.0)], p0), "n = 0 의 P1 은 P0 이다"
    for (n, d), s_ in sets.items():
        sel = np.array(sorted(s_), int)
        E_n = W.X.shrink(H.ls_E(c.yA[sel], c.sA[sel]), c.E0, len(sel), W.KAPPA) if len(sel) else c.E0
        pe = p1[pre + ("Pe", "none", "1", "cell", n, int(d), -1, 0.0)]
        assert np.allclose(pe, W.re_anchor(E_n * c.sB, c.XB[:, W.CCI_COL], c.XB[:, W.CCIV_COL]))
        okB = np.isfinite(c.XB[:, W.CCI_COL]) & (c.XB[:, W.CCIV_COL] >= 0.5)
        assert np.allclose(pe[~okB], E_n * c.sB[~okB]) and np.allclose(pe[okB], 0.5 * (E_n * c.sB[okB] + c.XB[okB, W.CCI_COL]))
        for m in ("R1", "Re"):
            assert pre + (m, W.LO, "1", "cell", n, int(d), 0, 0.25) in p1
    allsel = set().union(*sets.values())
    W.PRED_TRACE = {}
    try:
        W.T7Unit(a, make_tctx2(y_shift=_shift_outside(allsel, 5)), "T").run()
        p2 = dict(W.PRED_TRACE)
    finally:
        W.PRED_TRACE = None
    _assert_same_preds(p1, p2, "wf7")


# ---------------------------------------------------------------- (o) wf8
@pytest.mark.parametrize("strat", list(W.WF8_STRATEGIES))
def test_o_wf8_strategies_transfer(strat):
    a = args()
    c = make_tctx2()
    nA = len(c.yA)
    rc = W.RCtx("T", 1, "T", c.XA, c.yA, c.sA, c.blkA, np.zeros(nA), np.zeros(nA), c.XB, c.yB, c.sB, c.blkB, np.zeros(len(c.yB)),
                np.zeros(len(c.yB)), c.E0)
    for d in range(3):
        sets = W.T8Unit(a, c, "T", strat).strategy_sets(strat, [10, 20], d)
        again = W.T8Unit(a, make_tctx2(), "T", strat).strategy_sets(strat, [10, 20], d)
        for n in (10, 20):
            s_ = np.sort(np.asarray(sets[n]))
            assert len(s_) == n and len(np.unique(s_)) == n and s_.dtype.kind in "iu" and s_.min() >= 0 and s_.max() < nA
            assert np.array_equal(s_, np.sort(np.asarray(again[n]))), f"{strat}: 같은 seed 에서 결과가 다르다"
            if strat == "S1":
                assert np.array_equal(s_, H.draw_cells("T", "x", 1, n, d, nA)), "S1 은 LG 셀 추출이다"
            if strat == "S2":
                assert np.array_equal(s_, H.draw_blocks("T", "x", 1, n, d, c.blkA)), "S2 는 LG 블록 분산 추출이다"
            if strat == "S4":
                w2 = W.RUnit(a, rc, "wf2", "S4").strategy_sets("S4", [10, 20], d)
                assert np.array_equal(s_, np.sort(np.asarray(w2[n]))), "S4 는 wf2 의 S4 와 같다"
    with pytest.raises(ValueError):
        W.T8Unit(a, c, "T", "S6")


def test_o2_wf8_run_no_leakage_small_A_and_dup_reuse():
    a = args(["--wf8-grid", "10,20"])
    for strat in W.WF8_STRATEGIES:
        W.WF_TRACE = []; W.PRED_TRACE = {}
        try:
            U = W.T8Unit(a, make_tctx2(), "T", strat).run()
            rows, st, stats = U.finish()
            tr, p1 = list(W.WF_TRACE), dict(W.PRED_TRACE)
        finally:
            W.WF_TRACE = None; W.PRED_TRACE = None
        assert stats["status"] == "ok" and {k[3] for k in st.keys if k[0] != "P0"} == {strat}
        _assert_B_only(st, make_tctx2())
        sets = sel_sets(tr)
        for e in tr:
            if e["kind"] == "fit":
                assert set(int(v) for v in e["idx"]) <= sets[(int(e["n"]), str(e["draw"]))]
        allsel = set().union(*sets.values())
        W.WF_TRACE = []; W.PRED_TRACE = {}
        try:
            W.T8Unit(a, make_tctx2(y_shift=_shift_outside(allsel, 7)), "T", strat).run()
            tr2, p2 = list(W.WF_TRACE), dict(W.PRED_TRACE)
        finally:
            W.WF_TRACE = None; W.PRED_TRACE = None
        assert sel_sets(tr2) == sets, f"{strat}: 선택이 라벨에 의존한다"
        _assert_same_preds(p1, p2, f"wf8 {strat}")
    # |A| 가 작은 대상(15셀): n = 40 은 |A| 이상이라 빠지고 n = 10 만 남는다(LG 규칙)
    a40 = args(["--wf8-grid", "10,40"])
    U = W.T8Unit(a40, make_tctx2(nA=15, nbA=10), "T", "S4").run()
    _, st, _ = U.finish()
    assert {k[4] for k in st.keys if k[0] == "R1"} == {10}
    # 같은 라벨 집합이 여러 추출에서 나오면 적합은 한 번이고 예측은 추출마다 저장한다

    class Same(W.T8Unit):
        def strategy_sets(self, strat, budgets, d):
            return {n: np.arange(n) for n in budgets}
    W.PRED_TRACE = {}
    try:
        U = Same(a, make_tctx2(), "T", "S4").run()
        _, st, stats = U.finish()
        pt = dict(W.PRED_TRACE)
    finally:
        W.PRED_TRACE = None
    assert stats["n_fit"]["wf8"] == 2 * len(a.SEEDS), "중복 집합에서 다시 적합했다"
    assert stats["notes"].get("dup_draws")
    pre = ("wf8", "T|x", 1, "S4")
    for n in (10, 20):
        k0 = pre + ("R1", W.LO, "1", "S4", n, 0, 0, 0.25); k1 = pre + ("R1", W.LO, "1", "S4", n, 1, 0, 0.25)
        assert k0 in pt and k1 in pt and np.allclose(pt[k0], pt[k1])


# ---------------------------------------------------------------- (p) 1차 설정 불변
R1_CFG_HASH = {                                                           # 2차 보강 추가 전(커밋 1b42b4d)의 h54 로 계산한 값, data_sha = "TEST"
    "default|wf1|": ("79f292c0b60d", "a9eecca296b8"), "default|wf1|x34": ("83e93a56ae3e", "a9eecca296b8"),
    "default|wf2|S1": ("a9d38454c770", "dc02ac9e0bf0"), "default|wf2|S2": ("8cc280d55100", "dc02ac9e0bf0"),
    "default|wf2|S3": ("ef7e6b523368", "dc02ac9e0bf0"), "default|wf2|S4": ("fd09b3da0f43", "dc02ac9e0bf0"),
    "default|wf2|S5": ("8c626350356d", "dc02ac9e0bf0"), "default|wf2|S6": ("b9be964d2fc1", "dc02ac9e0bf0"),
    "default|wf2|S7": ("05cf228d6c14", "dc02ac9e0bf0"), "default|wf3|": ("f3ff92566e23", "511def187d71"),
    "default|wf4|": ("2c160dd13ee1", "fb6ab8550b07"),
    "smoke|wf1|": ("2f14f2a5893b", "627863625321"), "smoke|wf1|x34": ("de31a2edc947", "627863625321"),
    "smoke|wf2|S1": ("c53ba338dc9f", "07575e618415"), "smoke|wf2|S2": ("5920a1a19601", "07575e618415"),
    "smoke|wf2|S3": ("00d978d6a592", "07575e618415"), "smoke|wf2|S4": ("99778a35aaa1", "07575e618415"),
    "smoke|wf2|S5": ("346c156db714", "07575e618415"), "smoke|wf2|S6": ("d77e714bd610", "07575e618415"),
    "smoke|wf2|S7": ("7768f014f21f", "07575e618415"), "smoke|wf3|": ("8b7047c7617b", "ea738b0210e9"),
    "smoke|wf4|": ("f4a6844af492", "dc6967779768"),
    "precheck|wf1|": ("79f292c0b60d", "a9eecca296b8"), "precheck|wf1|x34": ("83e93a56ae3e", "a9eecca296b8"),
    "precheck|wf2|S1": ("a9d38454c770", "dc02ac9e0bf0"), "precheck|wf2|S2": ("8cc280d55100", "dc02ac9e0bf0"),
    "precheck|wf2|S3": ("ef7e6b523368", "dc02ac9e0bf0"), "precheck|wf2|S4": ("fd09b3da0f43", "dc02ac9e0bf0"),
    "precheck|wf2|S5": ("8c626350356d", "dc02ac9e0bf0"), "precheck|wf2|S6": ("b9be964d2fc1", "dc02ac9e0bf0"),
    "precheck|wf2|S7": ("05cf228d6c14", "dc02ac9e0bf0"), "precheck|wf3|": ("f3ff92566e23", "511def187d71"),
    "precheck|wf4|": ("2c160dd13ee1", "fb6ab8550b07"),
}
R1_WF4_DEFAULT = [("Lena", "x"), ("Canada", "x"), ("Russia_W", "x"), ("Russia_E", "x"), ("Russia_C", "x"), ("Greenland", "x"), ("Alaska", "x")] + \
    [(f"AL-{i}", "i") for i in range(1, 7)] + [(f"AL-{i}", "x") for i in range(1, 7)] + \
    [(t, m) for t in ("CA-2", "CA-3", "LE-1", "LE-2") for m in ("i", "x")] + [("Tibet_LGD", "x"), ("NAtlantic~lic", "x"), ("Russia_C~lgd", "x")]


def test_p_round1_config_unchanged(monkeypatch):
    monkeypatch.setattr(X, "CB_HI", dict(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0))   # 본 실행 값(시험 모듈이 줄인 값을 되돌린다)
    variants = {"wf1": ["", "x34"], "wf2": list(W.STRATEGIES), "wf3": [""], "wf4": [""]}
    for label, argv, suf in (("default", [], ""), ("smoke", ["--smoke"], "_smoke"), ("precheck", ["--precheck"], "_precheck")):
        a = W.parse_args(argv)
        assert a.TAG == f"wf{suf}" and a.TAG2 == f"wf2b{suf}"
        for exp, vs in variants.items():
            assert W.exp_tag(a, exp) == f"{exp}{suf}"
            for v in vs:
                cfg = W.unit_cfg(a, exp, v, "TEST")
                assert (W.wf_cfg_hash(cfg), W.wf_cfg_hash(cfg, common=True)) == R1_CFG_HASH[f"{label}|{exp}|{v}"], f"{label}|{exp}|{v}: 설정 해시가 바뀌었다"
                mode = "x" if exp == "wf4" else "r"
                assert W.shard_base(a, exp, "Alaska", mode, 3, v).name == f"{exp}{suf}__cpu__Alaska__{mode}__s3" + (f"__{v}" if v else "")
    a = W.parse_args([])
    assert a.T["wf1"] == ["Alaska", "Lena", "Canada", "AL-1", "AL-2", "AL-6"] and a.T["wf2"] == ["Alaska", "Lena", "Canada", "AL-1", "AL-2"]
    assert a.T["wf3"] == ["Alaska", "Lena"] and [tuple(v) for v in a.T["wf4"]] == R1_WF4_DEFAULT
    assert {k: a.G[k] for k in W.EXPS_R1} == {"wf1": [20, 50, 100, 200, 500, 1000, 2000, 5000, -1], "wf2": [20, 50, 100, 200, 500],
                                              "wf3": [100, 500, 2000, -1], "wf4": [10, 40, 160, -1]}
    assert a.SPLITS == [1, 2, 3, 4, 5] and a.SEEDS == [0, 1] and a.STRAT == list(W.STRATEGIES) and a.X34 == ["Alaska"]
    s = W.parse_args(["--smoke"])
    assert {k: s.G[k] for k in W.EXPS_R1} == {"wf1": [20, 50, -1], "wf2": [20, 50], "wf3": [100, -1], "wf4": [10, -1]}
    assert s.T["wf1"] == ["Canada", "AL-3"] and s.T["wf4"] == [("Russia_W", "x"), ("Tibet_LGD", "x")] and s.SPLITS == [1] and s.nboot == 500
    p = W.parse_args(["--precheck"])
    assert p.T["wf4"] == [("Lena", "x"), ("AL-2", "i")] and p.STRAT == ["S1", "S6"] and p.SPLITS == [1]
    for exp in W.EXPS_R1:                                                 # 1차 실험의 (n, 추출) 규칙(0 은 넣지 않는다)
        assert (0, 0) not in W.cells_for(a, exp, 100, grid=[0, 20, -1])


# ---------------------------------------------------------------- (q) 2차 보강 집계(HEAVY)
@HEAVY
def test_q_summarize_round2_end_to_end(tmp_path):
    a = args(["--out-dir", str(tmp_path)])
    for split in (1, 2):
        for tgt in ("Alaska", "Lena", "Canada"):
            c = make_rctx(seed=split * 10 + len(tgt), split=split, target=tgt)
            U, rows, st, stats = run_r(a, c, "wf6")
            W.write_shard(a, "wf6", tgt, "r", split, "", c, rows, st, stats, 1.0, [1, 2])
        for tgt, m in (("Lena", "x"), ("Canada", "x"), ("Russia_W", "x"), ("Russia_E", "x"), ("AL-2", "i")):
            c = make_tctx2(seed=split * 7 + len(tgt), split=split, target=tgt, mode=m)
            U = W.T7Unit(a, c, tgt).run()
            rows, st, stats = U.finish()
            W.write_shard(a, "wf7", tgt, m, split, "", c, rows, st, stats, 1.0, [1, 2])
        for tgt in ("Lena", "Canada", "Russia_W", "Russia_E", "Alaska"):
            c = make_tctx2(seed=split * 5 + len(tgt), split=split, target=tgt)
            for s in W.WF8_STRATEGIES:
                U = W.T8Unit(a, c, tgt, s).run()
                rows, st, stats = U.finish()
                W.write_shard(a, "wf8", tgt, "x", split, s, c, rows, st, stats, 1.0, [1, 2])
    out = W.summarize(a)
    assert out is not None
    t = out["tests"]
    for hyp in ("WF6-a", "WF6-b", "WF7-a", "WF7-b", "WF8-a", "WF8-b"):
        v = t[(t.test_id == hyp) & t.scope.isin(["verdict", "verdict_aux"])]
        assert len(v), f"{hyp} 판정 행이 없다"
        assert not v.verdict.astype(str).str.contains("계산 실패").any()
    assert len(t[(t.test_id == "WF6-a") & (t.scope == "verdict")]) == 3, "WF6-a 는 대상별 판정이다"
    assert not t[t.test_id.isin(["WF7-a", "WF7-b"])].hypothesis.any() and t[t.test_id == "WF8-a"].hypothesis.all()
    for f in ("wf2b_curve.csv", "wf2b_tests.csv", "wf2b_meta.json", "wf2b_timing.csv", "wf2b_targets.csv", "wf2b_failed.csv"):
        assert (tmp_path / f).exists(), f
    assert not (tmp_path / "wf_tests.csv").exists() and not (tmp_path / "wf_curve.csv").exists(), "1차 표를 쓰면 안 된다"
    meta = json.loads((tmp_path / "wf2b_meta.json").read_text())
    assert meta["plan_commit"] == "1b42b4d" and set(meta["exps"]) == {"wf6", "wf7", "wf8"}
    assert set(out["curve"].exp) == {"wf6", "wf7", "wf8"}
    cur = out["curve"]
    assert set(cur[cur.exp == "wf8"].placement) >= {"S1", "S2", "S4"} and (cur[cur.exp == "wf7"].n == 0).any()


# ---------------------------------------------------------------- (r) CLI·차수·n = 0
def test_r_cli_rounds_and_zero_n():
    a = W.parse_args(["--exp", "wf6,wf8"])
    assert a.EXPS == ["wf6", "wf8"]
    assert W.rounds_of(a) == [("r2", ["wf6", "wf8"], "wf2b")]
    b = W.parse_args([])
    assert b.EXPS == list(W.EXPS_ALL) and [r for r, _, _ in W.rounds_of(b)] == ["r1", "r2"]
    assert [tuple(v) for v in b.T["wf7"]] == [tuple(v) for v in H.default_targets(["i", "x"])] and len(b.T["wf7"]) == 27
    assert b.T["wf8"] == [(t, "x") for t in W.WF8_TARGETS] and b.T["wf6"] == list(W.WF6_TARGETS)
    assert b.SPLITS6 == list(range(1, 26)) and b.G["wf7"] == [0, 3, 10, 40, 160, 320, 1000, -1] and b.G["wf8"] == [10, 40]
    cells = W.cells_for(b, "wf7", 50)
    assert cells[0] == (0, 0) and (-1, 0) in cells and sum(1 for n, _ in cells if n == 10) == 5 and not any(n >= 50 for n, _ in cells)
    assert {n for n, _ in W.cells_for(b, "wf6", 600)} == {200, 500, -1} and sum(1 for n, _ in W.cells_for(b, "wf6", 600) if n == 200) == 3
    s = W.parse_args(["--smoke", "--exp", "wf6,wf7,wf8"])
    assert s.SPLITS6 == [6] and s.TAG2 == "wf2b_smoke" and W.exp_tag(s, "wf7") == "wf7_smoke"


# ---------------------------------------------------------------- (s) 격자 분해 항등식
def test_s_grid_decomposition_identity():
    rng = np.random.RandomState(3)
    n = 400
    blk = 100 + rng.randint(0, 12, n)
    sv = np.round(rng.choice([25.0, 27.5, 30.25, 33.0, 36.5], n) + 0.0, 6)
    sv[:3] = np.nan                                                       # 비유한 √TDD 는 혼자 묶음
    y = rng.uniform(20, 120, n)
    gid, multi = W.grid_groups(sv, blk)
    assert len(gid) == n and not multi[:3].any(), "비유한 √TDD 셀은 혼자 묶음이라 격자 안에서 빠진다"
    within, between = W.decomp_fns(gid, multi)
    for pred in (rng.uniform(20, 120, n), np.where(np.isfinite(sv), 1.5 * np.nan_to_num(sv), 50.0)):
        st_t = H.BlockStore("T|r", 1, blk); st_w = H.BlockStore("T~w|r", 1, blk[multi]); st_b = H.BlockStore("T~b|r", 1, blk)
        st_t.add(("k",), y, pred)
        yw, pw = within(y, pred); st_w.add(("k",), yw, pw)
        yb, pb = between(y, pred); st_b.add(("k",), yb, pb)
        tot = dict(zip(st_t.blocks, st_t.get(("k",))[0])); wi = dict(zip(st_w.blocks, st_w.get(("k",))[0])); be = dict(zip(st_b.blocks, st_b.get(("k",))[0]))
        for b in tot:
            assert np.isclose(tot[b], wi.get(b, 0.0) + be[b], rtol=1e-10, atol=1e-8), f"블록 {b}: 총 SSE ≠ 격자 안 + 격자 사이"
    const = np.zeros(n)                                                   # 묶음 안 상수 예측: 격자 안 SSE = 묶음 안 실측 제곱합
    for g in np.unique(gid):
        const[gid == g] = rng.uniform(0, 100)
    yw, pw = within(y, const)
    assert np.allclose(pw, 0.0)
    ss = sum(float(np.sum((y[(gid == g)] - y[(gid == g)].mean()) ** 2)) for g in np.unique(gid) if (gid == g).sum() >= 2)
    assert np.isclose(float(np.sum((yw - pw) ** 2)), ss)


# ---------------------------------------------------------------- (t) wf10 분할
def _wf10_D(n_blocks=24, cells=10, seed=0):
    import pandas as pd
    from types import SimpleNamespace
    rng = np.random.RandomState(seed)
    blk = np.repeat(np.arange(n_blocks), cells)
    s_b = rng.uniform(20, 45, n_blocks)
    sv = s_b[blk] + 0.3 * rng.randn(len(blk))
    sv[5] = np.nan
    df = pd.DataFrame({"block": blk, "s": sv, "alt_cm": rng.uniform(30, 120, len(blk)), "cci_alt": 50.0, "e5_sqrt_tdd_soil": sv})
    idx = np.arange(len(df))
    return SimpleNamespace(df=df, target_idx=lambda t: idx), s_b


def test_t_wf10_split_rules():
    D, _ = _wf10_D()
    df = D.df
    N = len(df)
    bm = df.groupby("block").s.mean()
    for var in W.WF10_VARIANTS:
        outs = [W.wf10_split(D, "T", var, k) for k in (1, 2, 3)]
        again = W.wf10_split(D, "T", var, 2)
        assert all(np.array_equal(outs[1][k_], again[k_]) for k_ in ("A", "W", "I")), "같은 분할이 결정적이지 않다"
        for o in outs:
            A_, W_, I_ = set(o["A"].tolist()), set(o["W"].tolist()), set(o["I"].tolist())
            assert not (A_ & W_) and not (A_ & I_) and not (W_ & I_) and len(A_ | W_ | I_) == N, "W·I·A 가 겹치거나 대상 셀을 덮지 않는다"
            assert len(W_) >= W.WF10_FRAC * N and len(I_) >= W.WF10_FRAC * N
            wb = {int(b) for b in o["W_blocks"]}
            rest = [int(b) for b in bm.index if int(b) not in wb]
            if var == "warm":
                assert min(bm[list(wb)]) >= max(bm[rest]) - 1e-12, "W 가 가장 따뜻한 블록이 아니다"
            else:
                assert max(bm[list(wb)]) <= min(bm[rest]) + 1e-12, "W 가 가장 추운 블록이 아니다"
            wb_less = sorted(wb, key=lambda b: (-bm[b] if var == "warm" else bm[b]))[:-1]   # 마지막 블록을 빼면 25 % 미만(최소 집합)
            assert df.block.isin(wb_less).sum() < W.WF10_FRAC * N
        assert outs[0]["W_blocks"] == outs[1]["W_blocks"] == outs[2]["W_blocks"], "W 는 분할 사이에 같아야 한다"
        assert len({tuple(o["I_blocks"]) for o in outs}) == 3, "I 는 분할마다 달라야 한다"


# ---------------------------------------------------------------- (u) wf9 지역 내 단위
def _check_decomp(sts, name):
    by = {st.target: st for st in sts}
    t, m = name.split("|")
    tot, wi, be = by[name], by.get(f"{t}~w|{m}"), by[f"{t}~b|{m}"]
    assert wi is not None and set(tot.keys) == set(wi.keys) == set(be.keys), "세 저장소의 키가 다르다"
    for k in tot.keys:
        s_t = dict(zip(tot.blocks, tot.get(k)[0])); s_w = dict(zip(wi.blocks, wi.get(k)[0])); s_b = dict(zip(be.blocks, be.get(k)[0]))
        for b in s_t:
            assert np.isclose(s_t[b], s_w.get(b, 0.0) + s_b[b], rtol=1e-9, atol=1e-6), f"{k} 블록 {b}: 분해 항등식 불성립"
    return tot, wi, be


def _grid_rctx(**kw):
    c = make_rctx(**kw)
    c.sB = np.round(c.sB / 4.0) * 4.0                                     # 채점 셀의 √TDD 를 격자처럼 묶는다(블록 안 2셀 이상 묶음이 생긴다)
    return c


def test_u_wf9_inregion_unit():
    a = args(["--wf9-grid", "20,40,all"])
    W.WF_TRACE = []; W.PRED_TRACE = {}
    try:
        U = W.R9Unit(a, _grid_rctx(), "wf9").run()
        rows, sts, stats = U.finish()
        tr, p1 = list(W.WF_TRACE), dict(W.PRED_TRACE)
    finally:
        W.WF_TRACE = None; W.PRED_TRACE = None
    assert isinstance(sts, list) and stats["store_names"] == ["T|r", "T~w|r", "T~b|r"]
    tot, wi, be = _check_decomp(sts, "T|r")
    meth = {(k[0], k[1]) for k in tot.keys}
    assert {("P1", "none"), ("Pk", "none"), ("Pc", "none"), ("D0", W.HI), ("R1", W.LO), ("Re", W.LO)} <= meth
    assert not any(k[0] in ("R2", "P2", "Pbest", "D1") for k in tot.keys), "wf9 는 R2·P2·Pbest·D1 을 두지 않는다"
    k1 = next(k for k in tot.keys if k[0] == "P1")
    c = _grid_rctx()
    gid, multi = W.grid_groups(c.sB, c.blkB)
    ss = sum(float(np.sum((c.yB[gid == g] - c.yB[gid == g].mean()) ** 2)) for g in np.unique(gid) if (gid == g).sum() >= 2)
    assert np.isclose(wi.get(k1)[0].sum(), ss), "P1 의 격자 안 SSE 가 묶음 안 실측 제곱합과 다르다"
    sets = sel_sets(tr)
    allsel = set().union(*sets.values())
    W.PRED_TRACE = {}
    try:
        W.R9Unit(a, _grid_rctx(y_shift=_shift_outside(allsel, 5)), "wf9").run()
        p2 = dict(W.PRED_TRACE)
    finally:
        W.PRED_TRACE = None
    _assert_same_preds(p1, p2, "wf9")


# ---------------------------------------------------------------- (v) wf9x 전이 단위
def test_v_wf9x_transfer_unit():
    a = args(["--wf9x-grid", "0,10,all"])
    c = make_tctx2()
    c.sB = np.round(c.sB / 4.0) * 4.0
    U = W.T9Unit(a, c, "T").run()
    rows, sts, stats = U.finish()
    tot, wi, be = _check_decomp(sts, "T|x")
    ks = set(tot.keys)
    assert any(k[0] == "P1" and k[4] == 0 for k in ks) and any(k[0] == "R1" and k[4] == 0 for k in ks), "n = 0 의 P1·R1 키가 없다"
    for m in ("R1", "R2", "D0", "D1"):
        assert any(k[0] == m and k[1] == W.LO for k in ks), f"{m} 키가 없다"
    assert {k[7] for k in ks if k[0] == "R1"} == set(W.LAMS)


# ---------------------------------------------------------------- (w) wf10 단위와 외삽 손실
def _w10_ctx(seed=0, split=1):
    c = make_rctx(seed=seed, split=split, target="T")
    nW = len(c.yB) // 2
    c.maskW = np.r_[np.ones(nW, bool), np.zeros(len(c.yB) - nW, bool)]
    c.sB = c.sB + 8.0 * c.maskW                                           # W 는 더 따뜻하다
    c.yB = c.yB + 1.4 * 8.0 * c.maskW
    c.variant = "warm"
    return c


def test_w_wf10_unit_and_ep():
    a = args(["--wf10-grid", "20,all", "--nboot", "200"])
    sts_by = {}
    for split in (1, 2):
        c = _w10_ctx(seed=split, split=split)
        U = W.R10Unit(a, c, "wf10", "warm").run()
        rows, sts, stats = U.finish()
        assert stats["store_names"] == ["T~warm|r", "T~warmW|r", "T~warmI|r"]
        by = {st.target: st for st in sts}
        tot, sw, si = by["T~warm|r"], by["T~warmW|r"], by["T~warmI|r"]
        assert set(sw.blocks) == set(np.unique(c.blkB[c.maskW]).astype(str)) and set(si.blocks) == set(np.unique(c.blkB[~c.maskW]).astype(str))
        for k in tot.keys:
            assert np.isclose(tot.get(k)[0].sum(), sw.get(k)[0].sum() + si.get(k)[0].sum()), "W·I SSE 의 합이 총 SSE 와 다르다"
        meth = {(k[0], k[1]) for k in tot.keys}
        assert {("P1", "none"), ("P2", "none"), ("D0", W.HI), ("D0", W.LO), ("D1", W.LO), ("R1", W.LO), ("R2", W.LO)} <= meth
        assert any(k[0] == "R1" and k[7] == W.LAM_CV for k in tot.keys) and any(k[0] == "R2" and k[7] == W.LAM_CV for k in tot.keys)
        for nm, st in by.items():
            sts_by.setdefault(nm, {})[split] = st
    units = [dict(split=sp, dup_of=-1, valid=True, expected_splits=[1, 2]) for sp in (1, 2)]
    tms = {nm: W.make_tm(nm, bs, units, 200, False) for nm, bs in sts_by.items()}
    gA, gB = W.gk("D0", -1, W.HI, 1.0), W.gk("P1", -1, "none")
    w_, i_ = X.region_stats(tms["T~warmW|r"], gA, gB), X.region_stats(tms["T~warmI|r"], gA, gB)
    ep = W.region_ep(tms["T~warmW|r"], tms["T~warmI|r"], gA, gB)
    assert np.isclose(ep["delta"], w_["delta"] - i_["delta"]) and np.isclose(ep["rmse_A"], w_["delta"]) and np.isclose(ep["rmse_B"], i_["delta"])
    if ep["dist"] is not None:
        assert np.allclose(ep["dist"], w_["dist"] - i_["dist"])
    assert W.region_ep(None, tms["T~warmI|r"], gA, gB) is None


# ---------------------------------------------------------------- (x) 3차 집계(HEAVY)
@HEAVY
def test_x_summarize_round3_end_to_end(tmp_path):
    a = args(["--out-dir", str(tmp_path), "--wf9-grid", "20,40,all", "--wf9x-grid", "0,10,all", "--wf10-grid", "20,all"])
    for split in (1, 2):
        for tgt in ("Alaska", "Lena", "Canada"):
            c = _grid_rctx(seed=split * 10 + len(tgt), split=split, target=tgt)
            U = W.R9Unit(a, c, "wf9").run()
            rows, sts, stats = U.finish()
            W.write_shard(a, "wf9", tgt, "r", split, "", c, rows, sts, stats, 1.0, [1, 2])
        for tgt in ("Lena", "Canada", "Russia_W", "Russia_E", "Alaska"):
            c = make_tctx2(seed=split * 7 + len(tgt), split=split, target=tgt)
            c.sB = np.round(c.sB / 4.0) * 4.0
            U = W.T9Unit(a, c, tgt).run()
            rows, sts, stats = U.finish()
            W.write_shard(a, "wf9x", tgt, "x", split, "", c, rows, sts, stats, 1.0, [1, 2])
        for tgt in ("Alaska", "Canada"):
            for var in W.WF10_VARIANTS:
                c = _w10_ctx(seed=split * 3 + len(tgt) + len(var), split=split)
                c.target = tgt
                U = W.R10Unit(a, c, "wf10", var).run()
                rows, sts, stats = U.finish()
                W.write_shard(a, "wf10", tgt, "r", split, var, c, rows, sts, stats, 1.0, [1, 2])
    b = W.parse_args(ARGS + ["--out-dir", str(tmp_path), "--exp", "wf9,wf9x,wf10", "--wf9-grid", "20,40,all", "--wf9x-grid", "0,10,all",
                             "--wf10-grid", "20,all"])
    out = W.summarize(b)
    assert out is not None
    t = out["tests"]
    for hyp in ("WF9-a", "WF9-c", "WF10-a", "WF10-b", "WF10-c", "WF10-d"):
        v = t[(t.test_id == hyp) & t.scope.isin(["verdict", "verdict_aux"])]
        assert len(v), f"{hyp} 판정 행이 없다"
        assert not v.verdict.astype(str).str.contains("계산 실패").any(), v.verdict.tolist()
    for f in ("wf3b_curve.csv", "wf3b_tests.csv", "wf3b_meta.json", "wf3b_rmse.csv", "wf3b_decomp.csv"):
        assert (tmp_path / f).exists(), f
    assert not (tmp_path / "wf_tests.csv").exists() and not (tmp_path / "wf2b_tests.csv").exists(), "1·2차 표를 쓰면 안 된다"
    meta = json.loads((tmp_path / "wf3b_meta.json").read_text())
    assert meta["plan_commit"] == "8180632" and set(meta["exps"]) == {"wf9", "wf9x", "wf10"}
    import pandas as pd
    dt = pd.read_csv(tmp_path / "wf3b_decomp.csv")
    assert len(dt) and np.nanmax(np.abs(dt.check_tot_eq_w_plus_b.values)) < 1e-6, "분해 표의 총 = 격자 안 + 격자 사이 점검 실패"
    p1 = dt[(dt.method == "P1")]
    assert np.allclose(p1.expl_within.values, 0.0, atol=1e-9)


# ---------------------------------------------------------------- (y) 3차 CLI·차수
def test_y_cli_round3():
    a = W.parse_args(["--exp", "wf9,wf9x,wf10"])
    assert a.EXPS == ["wf9", "wf9x", "wf10"] and W.rounds_of(a) == [("r3", ["wf9", "wf9x", "wf10"], "wf3b")]
    assert a.SPLITS9 == list(range(1, 26)) and a.SPLITS10 == list(range(1, 11)) and a.VAR10 == ["warm", "cold"]
    assert a.T["wf9"] == list(W.WF9_TARGETS) and a.T["wf9x"] == [(t, "x") for t in W.WF9X_TARGETS] and a.T["wf10"] == list(W.WF10_TARGETS)
    assert a.G["wf9"] == [200, 500, 1000, -1] and a.G["wf9x"] == [0, 10, 40, 160, -1] and a.G["wf10"] == [100, 500, -1]
    cells = W.cells_for(a, "wf9x", 50)
    assert cells[0] == (0, 0) and sum(1 for n, _ in cells if n == 10) == 5 and (-1, 0) in cells
    assert sum(1 for n, _ in W.cells_for(a, "wf9", 600) if n == 200) == 3 and sum(1 for n, _ in W.cells_for(a, "wf10", 600) if n == 100) == 3
    assert W.exp_tag(a, "wf9x") == "wf9x" and W.exp_tag(a, "wf10") == "wf10"
    assert W.shard_base(a, "wf10", "Alaska", "r", 3, "warm").name == "wf10__cpu__Alaska__r__s3__warm"
    b = W.parse_args([])
    assert b.EXPS == list(W.EXPS_ALL) and not (set(b.EXPS) & set(W.EXPS_R3)), "기본 실행 목록에 3차를 넣으면 안 된다"
    s = W.parse_args(["--smoke", "--exp", "wf9,wf9x,wf10"])
    assert s.SPLITS9 == [7] and s.VAR10 == ["warm"] and s.TAG3 == "wf3b_smoke" and s.T["wf10"] == ["Canada"]
    q = W.parse_args(["--precheck", "--exp", "wf9,wf9x,wf10"])
    assert q.SPLITS9 == [1] and q.T["wf9x"] == [("Alaska", "x"), ("Lena", "x")]
    for v in W.WF10_VARIANTS:
        c1, c2 = W.unit_cfg(a, "wf10", v, "S"), W.unit_cfg(a, "wf10", "cold" if v == "warm" else "warm", "S")
        assert W.wf_cfg_hash(c1, common=True) == W.wf_cfg_hash(c2, common=True), "변형은 공통 설정 해시에 들어가지 않는다"
