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
        "--wf3-grid", "40,all", "--wf4-grid", "10,40,all", "--nboot", "200"]


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
