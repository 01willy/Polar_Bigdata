"""scripts/3_deep_learning/x_multisource_stacking.py 단위 시험(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.2절 '구현·시험').

계획이 등록한 시험 이름은 tests/test_x_stacking.py 이고, 작업 지시에 따라 이 파일(tests/test_x_xb.py)에 같은 항목을 둔다(구현 기록 8절).
tests/test_x_stacking.py 는 이 파일을 다시 내보내는 얇은 파일이다(그 파일 이름으로만 실행할 때 같은 시험이 돈다).
모든 시험은 합성 자료만 쓴다(실제 라벨을 읽지 않는다). 화면에 RMSE·Δ·판정을 쓰지 않는다.

등록 항목(계획 2.2)
(a) simplex_ls 가 무작위 문제에서 w ≥ 0, Σw = 1 이고 SLSQP 기준해보다 SSE 가 같거나 작다.
(b) 교차 적합 Z 의 각 행은 자기 묶음 밖 라벨로만 보정된다(추적과 묶음 라벨 교체 불변).
(c) 누설 시험: 선택되지 않은 A 라벨과 B 라벨을 바꿔도 선택·가중·예측이 같다(xb_t, xb_r). 선택 라벨 하나를 바꾸면 달라진다(음성 대조).
(d) γ = 1 또는 n < 10(또는 cv_folds_of 의 묶음 수 K < 2. 라벨이 한 블록에만 있어 셀 묶음 K ≥ 2 이면 대체가 아니다)이면 Stack = P1(정확히 같음)이고
    stack_fallback 이 기록된다.
(e) anchor_fill 의 ρ 와 원천 계수는 원천에서만 정해진다(A·채점 셀의 앵커 값과 대상 라벨을 바꿔도 같다).
(f) P1·R1·Re 키가 h54 의 RUnit(wf6)·T7Unit(wf7)과 같다(합성 자료, 블록 SSE 가 비트 단위로 같다).
(g) 세기: dry 실행이 적합 수와 라벨 없이 정한 대체 사유를 세고, --count-only 의 화면 출력에 라벨에서 나온 통계가 없다
    (라벨을 모두 바꿔도 출력이 같고 출력 제한 패턴에 걸리는 줄이 없다).
(h) 설정 해시 고정(합성 표 기준의 고정값), 조각 이름 규약.
(i) Wei 후보는 Stack0 에 들어가지 않고, 확장 판 StackR 원천 행에서 학습 지점 5 km 안 행이 빠진다. Wei 표에 mask_ok·train_dist_km 가 없거나
    원천 행 거리가 비유한이면 중단한다(누설 통제를 선택 열에 맡기지 않는다).
추가 항목
(j) W+: select_w_wplus 의 W 선택이 h54.TUnit.select_w 와 같고, W+ 선택·예측이 선택 라벨만 쓴다(누설 시험).
(k) P* 결측 규칙(A 또는 채점 셀의 5 % 초과 → 후보에서 뺀다)과 B:ens(h42 x9 정의).
(l) 집계: 합성 조각에서 summarize 가 봉인 폴더에만 쓰고 화면에는 행 수와 해시만 남긴다. 퇴화 추출 규칙(대체 비율 > 50 % → 판정 불가)과 동등 재확인.
(m) --shard i/N 이 단위 목록을 겹치지 않게 모두 덮는다.
(n) 1c 입력 a2_tdd_cells 가 a2 정의(연별 TDD, 관측 연도 정합, pre2010·partial 표지, 육지 격자 폴백)를 따른다(합성 NetCDF).
(o) v4 tdd 표: 없으면 중단(--allow-missing-tdd-v4 일 때만 진행, unit.json 기록), sha256 앞 16자가 다르면 중단. 묶음 목록에 있다.
(p) P* 결측 규칙은 대상 수준(대상 셀 전체, 대상 채점 셀)으로 한 번 정한다.
(q) 판정 불가 갈래: 모든 지역이 퇴화하면 판정 불가(적층 미작동) 풀 행, 적층 미작동이 아닌 판정 불가는 사유 문장, XB-4 는 비열등이 없고
    판정할 수 없는 n 이 있으면 판정 불가.
(r) 집계·관문은 이 변형의 조각만 읽는다(핵심·변형 조각이 한 폴더에 있어도 섞지 않는다).
(s) 조각 runs.csv 에 RMSE·편향 열이 없다(계획 0.3). 로컬 자원 규약(30 GB 대기, 적합 모드는 할당 영역 2개 뒤에만 주소 공간 상한).
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c <코어 4개> python3 -m pytest -q -p no:cacheprovider tests/test_x_xb.py
(prlimit --as 로 가상 메모리를 10 GB 로 묶으면 CatBoost 적합이 수백 건 쌓인 뒤 멈춘다. 최대 RSS 는 약 0.35 GB 다. 구현 기록 9절)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import copy
import importlib.util

import sys
from pathlib import Path

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
M = _load("x_multisource_stacking", "scripts/3_deep_learning/x_multisource_stacking.py")
H, X, W, H4 = XB.H, XB.X, XB.W, XB.H4
CCI, CCIV = W.CCI_COL, W.CCIV_COL
NF = len(XB.FEATS)


def hargs(seeds=1, draws_cap=2, extra=()):
    """h54 인자(합성 시험용: 반복 10, 스레드 1)."""
    a = XB.h54_args(exp="wf6", threads=1, cb_iters=10, seeds=seeds, nboot=100, draws_cap=draws_cap,
                    extra=["--wf6-grid", "20,40,all", "--wf7-grid", "0,3,10,all"] + list(extra))
    a.draws_cap = draws_cap
    return a


def _cci(Xm, rng, frac_bad=0.1):
    Xm[:, CCI] = 40 + 10 * rng.randn(len(Xm)); Xm[:, CCIV] = 1.0
    bad = rng.rand(len(Xm)) < frac_bad
    Xm[bad, CCI] = np.nan; Xm[bad, CCIV] = 0.0
    return Xm


def _raw(rng, s):
    """후보 원값(P* = tdd, Ku, Ed, Ss). 일부 결측·0 이하 값을 넣어 anchor_fill 경로를 지난다."""
    tdd = (s ** 2) * rng.uniform(0.85, 1.15, len(s))
    ku = 1.3 * s + 3 * rng.randn(len(s)); ku[rng.rand(len(s)) < 0.05] = np.nan
    ed = 1.6 * s * rng.uniform(0.8, 1.2, len(s)); ed[rng.rand(len(s)) < 0.03] = -1.0
    ss = 0.9 * s + rng.randn(len(s))
    return dict(tdd=tdd, ku=ku, ed=ed, ss=ss)


def build_cands(s, y_src, X3, raw3, products=None, parent="T"):
    raw = {"P*": {k: raw3[k]["tdd"] for k in raw3}, "Ku": {k: raw3[k]["ku"] for k in raw3}, "Ed": {k: raw3[k]["ed"] for k in raw3},
           "Ss": {k: raw3[k]["ss"] for k in raw3}}
    cci = {k: np.asarray(X3[k][:, CCI], float) for k in X3}
    ccv = {k: np.asarray(X3[k][:, CCIV], float) for k in X3}
    return M.CandSet.build(s, y_src, raw, cci, ccv, products, parent)


def make_tctx(seed=0, split=1, target="T", mode="x", ns=300, nA=80, nB=60, nbA=8, products=None, slope=1.3):
    """합성 전이 문맥(h40.Ctx) + 후보 표. 원천 300행(세 거시 지역), A 80셀 8블록, B 60셀 6블록."""
    rng = np.random.RandomState(seed)
    Xs = _cci(rng.randn(ns, NF), rng); ss = rng.uniform(20, 40, ns); ys = 1.6 * ss + 3 * Xs[:, 1] + 2 * rng.randn(ns)
    XA = _cci(rng.randn(nA, NF), rng); sA = rng.uniform(20, 40, nA); yA = slope * sA + 3 * XA[:, 1] + 0.1 * np.nan_to_num(XA[:, CCI]) + 2 * rng.randn(nA)
    XBm = _cci(rng.randn(nB, NF), rng); sB = rng.uniform(20, 40, nB); yB = slope * sB + 3 * XBm[:, 1] + 0.1 * np.nan_to_num(XBm[:, CCI]) + 2 * rng.randn(nB)
    blkA = np.repeat(np.arange(nbA), int(np.ceil(nA / nbA)))[:nA]
    c = H.Ctx(target, mode, split, target, Xs, ys, ss, np.repeat(["r1", "r2", "r3"], ns // 3), XA, yA, sA, blkA, XBm, yB, sB,
              500 + np.repeat(np.arange(6), nB // 6), meta=dict(dup_of=-1, valid=True))
    raw3 = {k: _raw(rng, v) for k, v in (("src", c.s_src), ("A", c.sA), ("B", c.sB))}
    c.xb_cand = build_cands(dict(src=c.s_src, A=c.sA, B=c.sB), c.y_src, dict(src=c.X_src, A=c.XA, B=c.XB), raw3, products, target)
    return c


def make_rctx(seed=0, split=1, target="T", nA=160, nB=120, nbA=10, nbB=10, ns=250):
    """합성 지역 내 문맥(h54.RCtx) + 후보 표(원천 행은 c0·ρ 전용). E0 = 원천 최소제곱(xb 의 P1 c0 와 같다)."""
    rng = np.random.RandomState(seed)
    XA = _cci(rng.randn(nA, NF).astype(np.float32), rng); XBm = _cci(rng.randn(nB, NF).astype(np.float32), rng)
    sA, sB = rng.uniform(20, 40, nA), rng.uniform(20, 40, nB)
    yA = 1.4 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    yB = 1.4 * sB + 3 * XBm[:, 1] + 2 * rng.randn(nB)
    Xs = _cci(rng.randn(ns, NF), rng); s_src = rng.uniform(20, 40, ns); y_src = 1.6 * s_src + 2 * rng.randn(ns)
    E0 = H.ls_E(y_src, s_src)
    blkA = np.repeat(np.arange(nbA), int(np.ceil(nA / nbA)))[:nA]; blkB = 1000 + np.repeat(np.arange(nbB), int(np.ceil(nB / nbB)))[:nB]
    latA = 65 + 0.5 * (blkA % 5) + 0.05 * rng.rand(nA); lonA = -150 + 0.5 * (blkA // 5) + 0.05 * rng.rand(nA)
    latB = 68 + 0.5 * ((blkB - 1000) % 5) + 0.05 * rng.rand(nB); lonB = -150 + 0.5 * ((blkB - 1000) // 5) + 0.05 * rng.rand(nB)
    c = W.RCtx(target, split, target, XA, yA, sA, blkA, latA, lonA, XBm, yB, sB, blkB, latB, lonB, E0, src_X=Xs[:, :NF],
               meta=dict(dup_of=-1, valid=True, n_valid_splits=2, n_unique_splits=2))
    raw3 = {k: _raw(rng, v) for k, v in (("src", s_src), ("A", sA), ("B", sB))}
    c.xb_cand = build_cands(dict(src=s_src, A=sA, B=sB), y_src, dict(src=Xs, A=XA, B=XBm), raw3, None, target)
    return c


def run_t(a, c, alias="T", grid=(0, 3, 10, -1), variant=""):
    U = M.XBTUnit(a, c, alias, variant, grid=grid).run()
    return (U,) + U.finish()


def run_r(a, c, grid=(20, 40, -1)):
    U = M.XBRUnit(a, c, "xb_r", grid=grid).run()
    return (U,) + U.finish()


# ---------------------------------------------------------------- (a) simplex_ls
def test_a_simplex_vs_slsqp():
    from scipy.optimize import minimize
    rng = np.random.RandomState(1)
    for t in range(40):
        K = int(rng.randint(2, 8)); n = int(rng.randint(K + 2, 60))
        Z = rng.randn(n, K) * rng.uniform(0.5, 3, K) + rng.randn(n)[:, None]
        w0 = rng.dirichlet(np.ones(K)) if t % 3 else np.eye(K)[0]
        y = Z @ w0 + 0.3 * rng.randn(n) + (2.0 if t % 5 == 0 else 0.0)
        w, sse, S = M.simplex_ls(Z, y)
        assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-12, "단체 제약 위반"
        assert abs(float(np.sum((y - Z @ w) ** 2)) - sse) <= 1e-8 * max(1, sse)
        f = lambda v: float(np.sum((y - Z @ v) ** 2))                                          # noqa: E731
        r = minimize(f, np.ones(K) / K, method="SLSQP", bounds=[(0, 1)] * K,
                     constraints=[dict(type="eq", fun=lambda v: v.sum() - 1)], options=dict(ftol=1e-12, maxiter=500))
        assert sse <= f(np.clip(r.x, 0, None) / np.clip(r.x, 0, None).sum()) + 1e-7 * max(1.0, sse), "SLSQP 보다 SSE 가 크다"
    # 동률: 같은 열 두 개 → P1(색인 0)을 포함하고 원소가 적은 지지 집합
    Z = np.c_[np.arange(10.0), np.arange(10.0), np.ones(10)]
    w, _, S = M.simplex_ls(Z, np.arange(10.0))
    assert S == (0,) and w[0] == 1.0


# ---------------------------------------------------------------- (b) 교차 적합
def test_b_oof_rows_use_outside_labels_only():
    c = make_tctx(seed=2)
    cs = c.xb_cand
    idx = np.arange(0, 60, 2)
    fid, K, flag, reason = M.block_folds(c.blkA[idx], c.target, c.mode, c.split, 30, 0)
    assert K >= 2 and reason == ""
    tr = []
    Z = M.oof_base(cs, idx, c.yA, fid, K, trace=tr)
    folds = [e for e in tr if e["kind"] == "zfold"]
    assert len(folds) == K
    for e in folds:
        assert not set(e["cal"]) & set(e["pred"]), "보정 라벨이 예측 행과 겹친다"
        assert set(e["pred"]) == set(idx[fid == e["fold"]]) and set(e["cal"]) == set(idx[fid != e["fold"]])
    for j in range(K):                                                                      # 묶음 j 의 라벨을 바꿔도 묶음 j 행은 같다
        y2 = c.yA.copy(); y2[idx[fid == j]] += 50.0
        Z2 = M.oof_base(cs, idx, y2, fid, K)
        assert np.array_equal(Z[fid == j], Z2[fid == j])
        assert not np.array_equal(Z[fid != j], Z2[fid != j])


# ---------------------------------------------------------------- (c) 누설
def _collect(fn):
    store = {}

    def run(c):
        U = fn(c)
        store["stack"] = copy.deepcopy(U.stack_log)
        store["stack0"] = copy.deepcopy(U.notes.get("stack0"))
        return U
    return run, store


def test_c_leakage_transfer():
    a = hargs(seeds=1)
    c = make_tctx(seed=3)
    run1, s1 = _collect(lambda cc: M.XBTUnit(a, cc, "T", grid=(0, 3, 10, 40)).run())
    with XB.h54_trace() as (p1, tr):
        run1(c)
    keep = XB.selected_union(tr)
    assert 0 < len(keep) < len(c.yA)
    c2 = XB.perturb_labels(c, keep, seed=7)
    run2, s2 = _collect(lambda cc: M.XBTUnit(a, cc, "T", grid=(0, 3, 10, 40)).run())
    with XB.h54_trace() as (p2, _):
        run2(c2)
    cmp = XB.compare_predictions(p1, p2)
    assert cmp["ok"], cmp
    assert s1["stack"] == s2["stack"] and s1["stack0"] == s2["stack0"], "가중·γ·대체가 비선택 라벨에 따라 달라졌다"
    c3 = copy.copy(c); c3.yA = c.yA.copy(); c3.yA[keep[0]] += 40.0                         # 음성 대조: 선택 라벨 하나를 바꾸면 달라진다
    with XB.h54_trace() as (p3, _):
        M.XBTUnit(a, c3, "T", grid=(0, 3, 10, 40)).run()
    assert not XB.compare_predictions(p1, p3)["ok"]


def test_c_leakage_inregion():
    a = hargs(seeds=1)
    c = make_rctx(seed=4)
    res = XB.leakage_invariance(lambda cc: M.XBRUnit(a, cc, "xb_r", grid=(20, 40)).run(), c)
    assert res["ok"] and res["n_keep"] < res["n_A"], res


# ---------------------------------------------------------------- (d) 대체 규칙
def test_d_fallback_exact_p1(monkeypatch):
    a = hargs(seeds=1, draws_cap=1)
    c = make_tctx(seed=5)
    U, rows, st, stats = run_t(a, c, grid=(0, 3, 10, -1))
    rr = pd.DataFrame(rows)
    for n in (0, 3):                                                                         # n < 10
        k_s = ("Stack", "none", "1", "cell", n, 0, -1, 0.0); k_p = ("P1", "none", "1", "cell", n, 0, -1, 0.0)
        assert np.array_equal(st.get(k_s)[0], st.get(k_p)[0])
        assert rr[(rr.method == "Stack") & (rr.n == n)].stack_fallback.iloc[0] == M.FB_N
        assert np.array_equal(st.get(("StackR", W.LO, "1", "cell", n, 0, 0, 0.25))[0], st.get(("R1", W.LO, "1", "cell", n, 0, 0, 0.25))[0])
    # 묶음 규칙(계획 2.2 '가중 학습' 1, 문구 그대로): cv_folds_of 의 묶음 수 K < 2 일 때만 대체. 라벨이 한 블록에만 있으면 cv_folds_of 가
    # 셀 묶음(K ≥ 2)을 내므로 대체가 아니다(γ̂ = 1 대체는 자료에 따라 생길 수 있다)
    cs = c.xb_cand
    one = np.where(c.blkA == c.blkA[0])[0][:10]
    assert len(one) == 10
    sf = M.fit_stack(cs, one, c.yA, c.blkA, "T", "x", 1, 10, 0)
    assert sf.fold_flag == "cell_folds" and sf.n_folds >= 2 and sf.fallback in ("", M.FB_GAMMA)
    assert M.fallback_reason(10, c.blkA[one], "T", "x", 1, 10, 0) == ""
    # cv_folds_of 가 K < 2 를 내면 대체(folds<2)이고 적층 = P1(정확히 같음)
    monkeypatch.setattr(M.X, "cv_folds_of", lambda blk, *a_, **k_: (np.zeros(len(blk), int), 1, "forced_k1"))
    sf = M.fit_stack(cs, np.arange(40), c.yA, c.blkA, "T", "x", 1, 40, 0)
    assert sf.fallback == M.FB_FOLDS and sf.is_p1 and np.array_equal(sf.pred("B"), sf.params["P1"] * c.sB)
    assert M.fallback_reason(40, c.blkA[:40], "T", "x", 1, 40, 0) == M.FB_FOLDS
    monkeypatch.undo()
    # γ̂ = 1
    monkeypatch.setattr(M, "choose_gamma", lambda *a_, **k_: (1.0, {}))
    sf = M.fit_stack(cs, np.arange(40), c.yA, c.blkA, "T", "x", 1, 40, 0)
    assert sf.fallback == M.FB_GAMMA and sf.is_p1 and sf.reuse_r1
    E1, _ = U.coefs(np.arange(40))
    assert np.array_equal(sf.pred("B"), E1 * c.sB), "γ = 1 의 적층이 P1 과 정확히 같지 않다"
    a1 = M.fallback_reason(9, c.blkA[:9], "T", "x", 1, 9, 0)
    assert a1 == M.FB_N


# ---------------------------------------------------------------- (e) anchor_fill 의 ρ
def test_e_rho_from_source_only():
    rng = np.random.RandomState(6)
    s = {k: rng.uniform(20, 40, n) for k, n in (("src", 200), ("A", 50), ("B", 40))}
    X3 = {k: _cci(rng.randn(len(v), NF), rng) for k, v in s.items()}
    raw3 = {k: _raw(rng, v) for k, v in s.items()}
    y_src = 1.5 * s["src"] + rng.randn(200)
    cs1 = build_cands(s, y_src, X3, raw3)
    raw3b = copy.deepcopy(raw3)
    for part in ("A", "B"):                                                                 # 대상 쪽 앵커 값을 크게 바꾸고 결측을 늘린다
        for k in ("ku", "ed", "ss", "tdd"):
            v = raw3b[part][k] * 3.0; v[::4] = np.nan; raw3b[part][k] = v
    cs2 = build_cands(s, y_src, X3, raw3b)
    for nm in ("Ku", "Ed", "Ss"):
        assert cs1.info[nm]["rho"] == cs2.info[nm]["rho"] and cs1.c0[nm] == cs2.c0[nm], f"{nm}: ρ·c0 가 대상 쪽 값에 따라 달라졌다"
        bad = ~(np.isfinite(raw3b["A"][{"Ku": "ku", "Ed": "ed", "Ss": "ss"}[nm]]) & (raw3b["A"][{"Ku": "ku", "Ed": "ed", "Ss": "ss"}[nm]] > 0))
        assert np.allclose(cs2.b[nm]["A"][bad], cs2.info[nm]["rho"] * s["A"][bad])
    raw3c = copy.deepcopy(raw3); raw3c["src"]["ku"] = raw3c["src"]["ku"] * 1.7               # 원천 값을 바꾸면 ρ 가 바뀐다
    cs3 = build_cands(s, y_src, X3, raw3c)
    assert cs3.info["Ku"]["rho"] != cs1.info["Ku"]["rho"]


# ---------------------------------------------------------------- (f) h54 와 같은 키
def _same_keys(st_a, st_b, pred):
    n = 0
    for k in st_a.keys:
        if pred(k):
            assert k in st_b, k
            sa, ca = st_a.get(k); sb, cb = st_b.get(k)
            assert np.array_equal(sa, sb) and np.array_equal(ca, cb), f"키 {k} 의 블록 SSE 가 h54 와 다르다"
            n += 1
    return n


def test_f_keys_equal_h54(monkeypatch):
    monkeypatch.setattr(X, "CB_HI", dict(iterations=10, learning_rate=0.1, depth=3, l2_leaf_reg=3.0))
    a = hargs(seeds=1, draws_cap=2)
    c = make_rctx(seed=8)
    Ur = W.RUnit(a, c, "wf6").run()
    _, st_ref, _ = Ur.finish()
    _, _, st_xb, _ = run_r(a, c, grid=(20, 40, -1))
    n = _same_keys(st_xb, st_ref, lambda k: k[0] == "P1" or (k[0] in ("R1", "Re") and k[1] == W.LO and k[7] in (0.25, -1.0)) or k[0] == "P0")
    assert n >= 3 * 5
    ct = make_tctx(seed=9)
    Ut = W.T7Unit(a, ct, "T").run()
    _, st_ref, _ = Ut.finish()
    _, _, st_xt, _ = run_t(a, ct, grid=(0, 3, 10, -1))
    n = _same_keys(st_xt, st_ref, lambda k: k[0] in ("P0", "P1", "Pe", "R1", "Re"))
    assert n >= 4 * 3


# ---------------------------------------------------------------- (g) 세기
def _count_args(tmp_path, monkeypatch):
    out = tmp_path / "XB_multisource_stacking"
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    a = M.parse_args(["--threads", "1", "--cb-iters", "10", "--out-dir", str(out), "--tdd-matched", str(_tdd_table(tmp_path)),
                      "--tdd-v4", str(tmp_path / "none.csv"), "--allow-missing-tdd-v4"])
    return a


def _tdd_table(tmp_path):
    p = tmp_path / "tdd.csv"
    if not p.exists():
        pd.DataFrame(dict(loc_id=np.arange(5), tdd_matched=np.linspace(100, 200, 5))).to_csv(p, index=False)
    return p


def test_g_count_only_prints_no_label_statistics(tmp_path, monkeypatch, capsys):
    a = _count_args(tmp_path, monkeypatch)
    a.ha.draws_cap = 2
    labels = {"seed": 0}

    def fake_make_unit(a_, arm, target, mode, split, variant="", dry=False):
        if arm == "xb_r":
            c = make_rctx(seed=10 + int(split))
        else:
            c = make_tctx(seed=20 + int(split))
        if labels["seed"]:                                                                   # 라벨만 바꾼 문맥(공변량·블록 같음)
            rng = np.random.RandomState(labels["seed"])
            c.yA = rng.uniform(1, 300, len(c.yA)); c.yB = rng.uniform(1, 300, len(c.yB))
        U = M.XBRUnit(a_.ha, c, "xb_r", variant, dry, grid=(20, 40, -1)) if arm == "xb_r" else M.XBTUnit(a_.ha, c, target, variant, dry, grid=(0, 3, 10, 40, -1))
        return c, U
    monkeypatch.setattr(M, "make_unit", fake_make_unit)
    units = [("xb_r", "T", "r", 1, ""), ("xb_t", "T", "x", 1, ""), ("xb_t", "T", "x", 2, "")]
    import re
    outs = []
    for sd in (0, 99):
        labels["seed"] = sd
        df = M.count_only(a, units, [])
        o = capsys.readouterr().out
        assert XB.FORBIDDEN_OUT.search(o) is None, "세기 출력에 RMSE·Δ·판정 줄이 있다"
        outs.append(re.sub(r"세기 \d+s", "세기 Ns", o))
        assert int(df.fit_total.sum()) > 0
    assert outs[0] == outs[1], "라벨을 바꾸자 세기 출력이 달라졌다(라벨에서 나온 값이 출력에 있다)"
    fb = pd.read_csv(a.OUT / "xb_count_fallback.csv")
    assert set(fb.columns) >= {"arm", "target", "n", "n_draws", "n_fb_n", "n_fb_folds"}
    assert int(fb[(fb.arm == "xb_t") & (fb.n.isin([0, 3]))].n_fb_n.sum()) == int(fb[(fb.arm == "xb_t") & (fb.n.isin([0, 3]))].n_draws.sum())
    cnt = pd.read_csv(a.OUT / "xb_count.csv")
    assert not any("rmse" in c_.lower() for c_ in cnt.columns)


def test_g_dry_counts_match_real_fits():
    a = hargs(seeds=1, draws_cap=1)
    c = make_tctx(seed=11)
    Ud = M.XBTUnit(a, c, "T", dry=True, grid=(0, 3, 10, 40, -1)).run()
    _, _, sd = Ud.finish()
    Ur = M.XBTUnit(a, c, "T", grid=(0, 3, 10, 40, -1)).run()
    _, _, sr = Ur.finish()
    nd, nr = sd["n_fit_detail"], sr["n_fit_detail"]
    for k in nd:                                                                             # R1·Re 는 같고 StackR 은 세기가 상한(γ̂ = 1 재사용 미반영)
        if k.endswith("|StackR"):
            assert nd[k] >= nr.get(k, 0)
        else:
            assert nd[k] == nr.get(k, 0), k
    lf = [e["fallback"] for e in sd["stack"]]
    rf = [e["fallback"] for e in sr["stack"]]
    assert all((x_ == y_) or (x_ == "" and y_ == M.FB_GAMMA) for x_, y_ in zip(lf, rf)), "라벨 없이 정한 대체 사유가 본 실행과 다르다"


# ---------------------------------------------------------------- (h) 설정 해시·조각 이름
CFG_HASH_FIXED = {"xb_r": "06804715d0e5", "xb_t": "dd55546f0a48"}                          # 합성 표 기준의 고정값(2026-10-04 검토 반영:
# 묶음 규칙 문구 그대로, P* 대상 수준, Wei 5 km 범위·마스크 필수, v4 표 출처 항목. 첫 실행 값 a7f7257274e3·799c4dcfb332 에서 바뀌었다)


def test_h_cfg_hash_fixed(tmp_path, monkeypatch):
    a = _count_args(tmp_path, monkeypatch)
    h = {arm: XB.cfg_hash(M.unit_cfg(a, arm, "", "sha:fixed"), common=True) for arm in M.ARMS}
    h2 = {arm: XB.cfg_hash(M.unit_cfg(a, arm, "", "sha:other"), common=True) for arm in M.ARMS}
    assert h == h2, "공통 해시에 data_sha 가 들어갔다"
    cfg = M.unit_cfg(a, "xb_t", "", "sha:fixed")
    assert cfg["cands"] == list(M.CANDS_CORE) and cfg["gammas"] == [0.0, 0.25, 0.5, 0.75, 1.0] and cfg["stack_min_n"] == 10
    assert cfg["pstar_max_miss"] == 0.05 and cfg["grid"] == [0, 3, 10, 40, 160, 320, 1000, -1] and cfg["draws"] == 5
    assert M.unit_cfg(a, "xb_r", "", "x")["grid"] == [20, 50, 100, 200, 500, 1000, -1] and M.unit_cfg(a, "xb_r", "", "x")["draws"] == 3
    if all(CFG_HASH_FIXED.values()):
        assert h == CFG_HASH_FIXED, f"설정 해시가 고정값과 다르다: {h} ≠ {CFG_HASH_FIXED}"
    else:
        print(f"[cfg-hash] {h}")
    assert M.TAGS == {"xb_r": "xbr", "xb_t": "xbt"}
    p = XB.shard_paths(tmp_path, a.TAG["xb_t"], "Canada~exp~lic", "x", 3, "ext")
    assert p["unit"].name == "xbt__cpu__Canada~exp~lic__x__s3__ext_unit.json"


# ---------------------------------------------------------------- (i) Wei
def test_i_wei_not_in_stack0_and_5km_rule():
    rng = np.random.RandomState(12)
    c = make_tctx(seed=12)
    ns, nA, nB = len(c.y_src), len(c.yA), len(c.yB)
    dist = rng.uniform(0, 50, ns); dist[:40] = 1.0
    wei = dict(src=1.2 * c.s_src + rng.randn(ns), A=1.2 * c.sA + rng.randn(nA), B=1.2 * c.sB + rng.randn(nB),
               mask_A=np.ones(nA, bool), mask_B=np.ones(nB, bool), dist_src=dist)
    c2 = make_tctx(seed=12, products={"Wei": wei})
    cs = c2.xb_cand
    assert "Wei" in cs.names and "Wei" in cs.stack0_exclude
    a = hargs(seeds=1, draws_cap=1)
    U = M.XBTUnit(a, c2, "T", "ext", grid=(0, 10, -1)).run()
    assert "Wei" not in U.notes["stack0"]["names"] and "Wei" in U.notes["stack0"]["excluded"]
    sf = M.fit_stack(cs, np.arange(40), c2.yA, c2.blkA, "T", "x", 1, 40, 0)
    Xtr, ytr, _ = M.stackr_rows(U, sf, np.arange(40))
    keep = ~(dist < M.WEI_KM)
    assert len(ytr) == int(keep.sum()) + 40, "확장 판 StackR 원천 행에서 Wei 학습 지점 5 km 안 행이 빠지지 않았다"
    assert U.notes["xb_cands"]["resid_src_excluded"] == int((~keep).sum()) and cs.meta["km5_src_excluded_Wei"] == int((~keep).sum())
    wei_bad = dict(wei, mask_B=np.r_[False, np.ones(nB - 1, bool)])                          # 채점 셀 하나가 마스크 실패 → 그 대상에서 뺀다
    c3 = make_tctx(seed=12, products={"Wei": wei_bad})
    assert "Wei" not in c3.xb_cand.names and "Wei" in c3.xb_cand.dropped
    assert np.array_equal(c3.xb_cand.resid_src_keep, keep), "5 km 제외는 확장 판 전체(후보에서 빠진 대상 포함)에 적용한다"
    yk = dict(wei, dist_src=None, mask_A=None, mask_B=None)                                  # YK 는 L0(마스크 없음), 5 km 규칙 없음
    c4 = make_tctx(seed=12, products={"YK": yk})                                             # parent T ≠ Alaska → 뺀다
    assert "YK" not in c4.xb_cand.names
    c5 = make_tctx(seed=12, products={"YK": yk}, target="Alaska")                            # 알래스카 계열이면 마스크 열 없이 들어간다
    assert "YK" in c5.xb_cand.names and c5.xb_cand.resid_src_keep is None
    # 누설 통제 열이 없거나 비유한이면 중단한다(조용히 건너뛰지 않는다)
    for bad in (dict(wei, dist_src=None), dict(wei, dist_src=np.r_[np.nan, dist[1:]]), dict(wei, dist_src=dist[:-1]),
                dict(wei, mask_A=None), dict(wei, mask_B=None)):
        with pytest.raises(SystemExit):
            make_tctx(seed=12, products={"Wei": bad})


def test_i2_read_product_requires_leak_columns(tmp_path):
    base = pd.DataFrame(dict(loc_id=np.arange(6), value=np.linspace(50, 100, 6), mask_ok=[1, 1, 0, 1, np.nan, 1], train_dist_km=np.linspace(0, 10, 6)))
    p = tmp_path / "wei.csv"
    base.to_csv(p, index=False)
    t = M._read_product(str(p), "Wei")
    assert t.mask_ok.tolist() == [True, True, False, True, False, True] and np.isfinite(t.train_dist_km).all()
    for drop in ("mask_ok", "train_dist_km"):
        q = tmp_path / f"wei_no_{drop}.csv"
        base.drop(columns=[drop]).to_csv(q, index=False)
        with pytest.raises(SystemExit):
            M._read_product(str(q), "Wei")
    q = tmp_path / "yk.csv"
    base.drop(columns=["mask_ok", "train_dist_km"]).to_csv(q, index=False)
    t = M._read_product(str(q), "YK")
    assert bool(t.mask_ok.all()) and not np.isfinite(t.train_dist_km).any()
    with pytest.raises(SystemExit):
        M._read_product(f"{q}:alt_cm", "YK")                                                 # 값 열이 없다


# ---------------------------------------------------------------- (j) W+
def test_j_wplus_w_matches_h54_and_no_leakage():
    a = hargs(seeds=1, draws_cap=1)
    c = make_tctx(seed=13)
    U = M.XBTUnit(a, c, "T", grid=(40,))
    for n, d in ((40, 0), (60, 0)):
        sel = H.draw_cells(c.target, c.mode, c.split, n, d, len(c.yA))
        ref = W.TUnit.select_w(U, sel, n, d)
        got = M.select_w_wplus(U, sel, n, d)
        assert got["W"][:3] == ref[:3] and got["W"][3] == ref[3], "W 선택이 h54 와 다르다"
        assert got["Wplus"][0] in M.WPLUS_CANDS and set(got["Wplus"][3]) == set(M.WPLUS_CANDS)
    assert M.select_w_wplus(U, np.arange(5), 5, 0)["Wplus"][0] == "P1"

    def run(cc):
        V = M.XBTUnit(a, cc, "T", grid=(40,))
        for n in (40, 60):
            sel = V.draw(n, 0)
            V.trace("select", sel, n=n, draw="0")
            ch = M.select_w_wplus(V, sel, n, 0)["Wplus"][0]
            E1, _ = V.coefs(sel)
            (g1,) = V.fit(W.LO, "R1", lambda: V.rows_R(sel, E1 * cc.sA[sel]), V.nsrc + n, 0, [cc.XB], n, 0, sel)
            pr, log = M.wplus_predictions(V, sel, n, 0, 0, g1)
            V.add("Wplus_" + ch.replace("@", "_"), W.LO, "cell", n, 0, 0, 0.0, pr["StackR@0.25"])
            V.add("Wplus_stack", "none", "cell", n, 0, -1, 0.0, pr["Stack"])
        return V
    res = XB.leakage_invariance(run, c)
    assert res["ok"] and res["n_keep"] < res["n_A"], res


# ---------------------------------------------------------------- (k) P* 결측 규칙과 B:ens
def test_k_pstar_missing_rule_and_bens():
    rng = np.random.RandomState(14)
    s = {k: rng.uniform(20, 40, n) for k, n in (("src", 200), ("A", 100), ("B", 100))}
    X3 = {k: _cci(rng.randn(len(v), NF), rng) for k, v in s.items()}
    raw3 = {k: _raw(rng, v) for k, v in s.items()}
    y_src = 1.5 * s["src"] + rng.randn(200)
    raw3["A"]["tdd"][:5] = np.nan                                                           # 5 % = 5/100 → 남긴다
    assert "P*" in build_cands(s, y_src, X3, raw3).names
    raw3["B"]["tdd"][:6] = np.nan                                                           # 6 % → 뺀다
    cs = build_cands(s, y_src, X3, raw3)
    assert "P*" not in cs.names and "P*" in cs.dropped
    e = cs.ens
    ku_c, ku_rho, _ = X.anchor_fill(raw3["src"]["ku"], raw3["A"]["ku"], raw3["B"]["ku"], s["src"], s["A"], s["B"])
    cc, _, _ = X.anchor_fill(X3["src"][:, CCI], X3["A"][:, CCI], X3["B"][:, CCI], s["src"], s["A"], s["B"])
    assert np.allclose(e["ku_B"], ku_c["B"]) and abs(e["c0_ku"] - H.ls_E(y_src, ku_c["src"])) < 1e-12
    assert np.allclose(e["cci_B"], cc["B"]) and abs(e["c0_cci"] - H.ls_E(y_src, cc["src"])) < 1e-12


# ---------------------------------------------------------------- (l) 집계(합성 조각)
def _write_units(a, tmp_shards):
    """주 4지역(모드 x)과 지역 내 3대상의 합성 조각을 쓴다(분할 1·2). 대상 이름을 등록 풀의 이름으로 둔다."""
    for k, nm in enumerate(["Lena", "Canada", "Russia_W", "Russia_E"]):
        for sp in (1, 2):
            c = make_tctx(seed=100 + 10 * k + sp, split=sp, target=nm, nA=160, nbA=16, slope=1.2 + 0.1 * k)
            U = M.XBTUnit(a.ha, c, nm, grid=(10, 40, -1)).run()
            rows, st, stats = U.finish()
            XB.write_shard(tmp_shards, a.TAG["xb_t"], nm, "x", sp, rows, st, M.unit_cfg(a, "xb_t", "", "s"), unit=dict(stats, xb_cands=c.xb_cand.describe()),
                           expected=[1, 2])
    for k, nm in enumerate(["Alaska", "Lena", "Canada"]):
        for sp in (1, 2):
            c = make_rctx(seed=200 + 10 * k + sp, split=sp, target=nm, nA=260, nbA=13)
            U = M.XBRUnit(a.ha, c, "xb_r", grid=(200, -1)).run()
            rows, st, stats = U.finish()
            XB.write_shard(tmp_shards, a.TAG["xb_r"], nm, "r", sp, rows, st, M.unit_cfg(a, "xb_r", "", "s"), unit=dict(stats, xb_cands=c.xb_cand.describe()),
                           expected=[1, 2])


def test_l_summarize_sealed_only(tmp_path, monkeypatch, capsys):
    a = _count_args(tmp_path, monkeypatch)
    a.ha.draws_cap = 2; a.nboot = 100; a.ha.SEEDS = [0]
    a.G_T = [10, 40, -1]; a.G_R = [200, -1]
    _write_units(a, a.SHARDS)
    capsys.readouterr()
    with XB.restricted_output(report=False):
        t = M.summarize(a)
    o = capsys.readouterr().out
    assert XB.FORBIDDEN_OUT.search(o) is None
    assert all(line.startswith("[봉인]") for line in o.strip().splitlines()), "집계가 봉인 기록 밖의 줄을 화면에 썼다"
    sd = a.OUT / "sealed"
    names = {p.name for p in sd.iterdir()}
    assert {"xb_tests.csv", "xb_weights.csv", "xb_fallback.csv", "xb_meta.json", "xb_group_error.csv", "sealed_manifest.json"} <= names
    assert {p.name for p in a.OUT.iterdir()} == {"shards", "sealed"}, "봉인 폴더 밖에 집계 표를 썼다"
    for hyp in ("XB-1", "XB-2", "XB-3", "XB-4", "XB-5"):
        assert (t.hyp == hyp).any(), hyp
    assert t[t.hyp.isin(["XB-1", "XB-2", "XB-3"])].hypothesis_verdict.str.len().min() > 0
    assert (t[(t.hyp == "XB-1") & (t.scope == "MEAN")].holm_family == "XB-1–XB-3(m 12)").all()


def test_l_degenerate_rule_and_eq_recheck():
    """대체 비율 > 50 % 지역은 판정 불가(적층 미작동)로 풀에서 빠지고, 동등은 대체 없는 추출 판정과 같을 때만 남는다."""
    rng = np.random.RandomState(15)

    def tm_of(name, level_s, level_p, draws=4, splits=(1, 2, 3), nb=10):
        by = {}
        for sp in splits:
            ncell = rng.randint(3, 20, nb)
            st = H4.BlockStore(name, sp, np.repeat([f"{name[:2]}{sp}b{j}" for j in range(nb)], ncell))
            cnt = st.ncell
            st.add_sse(H.P0_KEY, cnt * 30.0 ** 2, cnt)
            for d in range(draws):
                e = rng.gamma(2.0, 0.05, st.nb)
                st.add_sse(("P1", "none", "1", "cell", 10, d, -1, 0.0), cnt * (level_p + e) ** 2, cnt)
                st.add_sse(("Stack", "none", "1", "cell", 10, d, -1, 0.0), cnt * (level_s + e + (0.0 if d < 2 else 0.0)) ** 2, cnt)
            by[sp] = st
        tm = H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in splits}, 300)
        tm.expected_splits = list(splits)
        return tm
    tms = {"A1|x": tm_of("A1|x", 20.0, 20.0), "A2|x": tm_of("A2|x", 21.0, 21.0), "A3|x": tm_of("A3|x", 22.0, 22.0)}
    fbr = {("A3|x", 10): dict(ratio=0.75, n_draws=12, n_fb=9, reasons={"folds<2": 9})}
    fbd = {("A3|x", 10): {(sp, 10, d) for sp in (1, 2, 3) for d in (0, 1, 2)}}
    rows, per = M.contrast_rows("XB-1", tms, list(tms), M._g("Stack", 10), M._g("P1", 10), 10, 3, fbr, fbd)
    r3 = [r for r in rows if r.get("target") == "A3|x"][0]
    assert r3["verdict_final"] == "판정 불가" and r3["verdict_note"] == "적층 미작동"
    assert "A3|x" not in per and rows[-1].get("pool", "").startswith("부분")
    assert rows[-1]["verdict_final"] == "동등"                                              # 대체 추출이 없는 지역뿐이라 동등이 유지된다


# ---------------------------------------------------------------- (m) --shard
def test_m_shard_partition():
    units = [(arm, t, m, sp, "") for arm in M.ARMS for t, m in (("Lena", "x"), ("AL-1", "i"), ("Alaska", "r")) for sp in range(1, 8)]
    parts = [M.shard_select(units, f"{i}/4") for i in range(1, 5)]
    flat = [u for p in parts for u in p]
    assert sorted(flat, key=M._unit_name) == M.shard_select(units) and len(set(flat)) == len(units)
    with pytest.raises(SystemExit):
        M.shard_select(units, "5/4")


# ---------------------------------------------------------------- (n) a2 정의의 tdd_matched
def test_n_a2_tdd_cells(tmp_path):
    xr = pytest.importorskip("xarray")
    times = pd.date_range("2010-01-01", "2024-12-01", freq="MS")
    lat = np.round(np.arange(70.0, 68.75, -0.1), 4); lon = np.round(np.arange(-150.0, -148.75, 0.1), 4)
    rng = np.random.RandomState(16)
    mon = np.array([t.month for t in times]); yr = np.array([t.year for t in times])
    base = (-20 + 30 * np.sin((mon - 4) / 12 * 2 * np.pi))[:, None, None] + 0.5 * (yr - 2010)[:, None, None]
    t2m = (base + rng.randn(len(times), len(lat), len(lon)) + 273.15).astype(np.float32)
    t2m[:, 0, 0] = np.nan                                                                   # 바다 격자(폴백 시험)
    ds = xr.Dataset(dict(t2m=(("valid_time", "latitude", "longitude"), t2m)), coords=dict(valid_time=times, latitude=lat, longitude=lon))
    p = tmp_path / "t.nc"
    ds.to_netcdf(p)
    res = M.a2_tdd_cells(np.array([69.5, 70.0, 69.2]), np.array([-149.5, -150.0, -149.0]), np.array([2015, 2005, 2008]), np.array([2016, 2008, 2012]), nc=p)
    assert list(res.match_flag) == ["full", "pre2010", "partial"]
    assert res.fallback_deg.iloc[1] == 0.2 and res.fallback_deg.iloc[0] == 0.0
    import calendar
    iy, ix = int(np.abs(lat - 69.5).argmin()), int(np.abs(lon + 149.5).argmin())
    T = t2m[:, iy, ix].astype(np.float32) - np.float32(273.15)
    days = np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(yr, mon)])
    tdd_y = {y: float((np.clip(T[yr == y], 0, None) * days[yr == y]).sum()) for y in range(2010, 2025)}
    assert abs(res.tdd_matched.iloc[0] - np.mean([tdd_y[2015], tdd_y[2016]])) < 1e-6
    assert res.n_years_matched.tolist() == [2, 5, 3]
    assert abs(res.tdd_matched.iloc[2] - np.mean([tdd_y_at(t2m, lat, lon, 69.2, -149.0, yr, mon)[y] for y in (2010, 2011, 2012)])) < 1e-6


def tdd_y_at(t2m, lat, lon, la, lo, yr, mon):
    import calendar
    iy, ix = int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin())
    T = t2m[:, iy, ix].astype(np.float32) - np.float32(273.15)
    days = np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(yr, mon)])
    return {y: float((np.clip(T[yr == y], 0, None) * days[yr == y]).sum()) for y in range(2010, 2025)}


def test_j_xc_provider_contract():
    """XC 공급자 계약(xc_wplus_cv, xc_wplus_final)이 select_w_wplus 의 묶음 안 계산·wplus_predictions 와 같은 값을 낸다."""
    a = hargs(seeds=1, draws_cap=1)
    c = make_tctx(seed=17)
    U = M.XBTUnit(a, c, "T", grid=(40,))
    n, d = 40, 0
    sel = H.draw_cells(c.target, c.mode, c.split, n, d, len(c.yA))
    got = M.select_w_wplus(U, sel, n, d)
    fid, K, _ = U.folds(sel, n, d)
    sse = {"Stack": 0.0, "StackR@0.25": 0.0}
    for j in range(K):
        tr, te = sel[fid != j], sel[fid == j]
        pr = M.xc_wplus_cv(U, tr, te, n, d, j, U.seeds[0])
        for k in sse:
            sse[k] += float(np.sum((pr[k] - c.yA[te]) ** 2))
    for k in sse:
        assert np.sqrt(sse[k] / len(sel)) == got["Wplus"][3][k], "XC 공급자의 묶음 예측이 select_w_wplus 와 다르다"
    fin = M.xc_wplus_final(U, sel, n, d, 0)
    pr, log = M.wplus_predictions(U, sel, n, d, 0, None)
    assert set(fin) == set(M.WPLUS_EXTRA) and all(np.array_equal(fin[k], pr[k]) for k in fin) and len(fin["Stack"]) == len(c.yB)


# ---------------------------------------------------------------- (o) v4 tdd 표(1c)의 존재·해시
def test_o_tdd_v4_required(tmp_path, monkeypatch):
    v1 = _tdd_table(tmp_path)
    missing = tmp_path / "missing_v4.csv"
    with pytest.raises(SystemExit):
        M.tdd_tables(v1, missing)
    v, flags, src = M.tdd_tables(v1, missing, allow_missing=True)
    assert src["v4_missing_allowed"] is True and len(flags) == 0
    v4 = tmp_path / "v4.csv"
    pd.DataFrame(dict(loc_id=np.arange(3, 8), tdd_matched=np.r_[np.linspace(100, 200, 5)[3:], 300.0, 310.0, 320.0],
                      match_flag=["full", "full", "pre2010", "full", "partial"])).to_csv(v4, index=False)
    with pytest.raises(SystemExit):                                                          # 기록된 해시와 다르다
        M.tdd_tables(v1, v4)
    monkeypatch.setattr(M, "TDD_V4_SHA16", XB.sha256_file(v4)[:16])
    v, flags, src = M.tdd_tables(v1, v4)
    assert src["v4_missing_allowed"] is False and src["v4_sha256_16"] == M.TDD_V4_SHA16
    assert len(v) == 8 and float(v.loc[7]) == 320.0 and flags.loc[5] == "pre2010"
    # 명령행: 허용 표지 없이 없는 표를 주면 입력 점검에서 중단, 허용 표지는 unit.json·설정에 남는다
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    a = M.parse_args(["--out-dir", str(tmp_path / "XB_multisource_stacking"), "--tdd-matched", str(v1), "--tdd-v4", str(missing)])
    assert a.TDDV4_ALLOW_MISSING is False and a.ha.TDDV4_ALLOW_MISSING is False
    with pytest.raises(SystemExit):
        M.check_inputs(a)
    a = _count_args(tmp_path, monkeypatch)
    assert a.TDDV4_ALLOW_MISSING and M.unit_cfg(a, "xb_t", "", "s")["tdd"]["v4_missing_allowed"] is True
    # 묶음(4절): xbatch_core 의 입력 목록과 모듈의 PAYLOAD_EXTRA 에 v4 표와 메타가 있다
    for f in M.PAYLOAD_EXTRA:
        assert f in XB.PAYLOAD_INPUTS, f
    assert M.PAYLOAD_EXTRA == ("data/processed/xbatch/XB_multisource_stacking/inputs/xb_tdd_matched_v4.csv",
                               "data/processed/xbatch/XB_multisource_stacking/inputs/xb_tdd_matched_v4_meta.json")
    real = ROOT / M.PAYLOAD_EXTRA[0]
    if real.exists():                                                                        # 1c 산출이 있으면 기록된 해시와 같아야 한다
        assert XB.sha256_file(real)[:16] == "4b22ca458685950c"


# ---------------------------------------------------------------- (p) P* 의 대상 수준 결정
def test_p_pstar_target_level(tmp_path):
    rng = np.random.RandomState(18)
    s = {k: rng.uniform(20, 40, n) for k, n in (("src", 200), ("A", 100), ("B", 100))}
    X3 = {k: _cci(rng.randn(len(v), NF), rng) for k, v in s.items()}
    raw3 = {k: _raw(rng, v) for k, v in s.items()}
    y_src = 1.5 * s["src"] + rng.randn(200)
    raw = {"P*": {k: raw3[k]["tdd"] for k in raw3}, "Ku": {k: raw3[k]["ku"] for k in raw3}, "Ed": {k: raw3[k]["ed"] for k in raw3},
           "Ss": {k: raw3[k]["ss"] for k in raw3}}
    cci = {k: X3[k][:, CCI] for k in X3}; ccv = {k: X3[k][:, CCIV] for k in X3}
    cs = M.CandSet.build(s, y_src, raw, cci, ccv, pstar_target=dict(miss_A=0.06, miss_B=0.0))   # 이 분할은 결측 0, 대상 수준 6 %
    assert "P*" not in cs.names and "target" in cs.dropped["P*"]
    raw["P*"]["B"] = raw["P*"]["B"].copy(); raw["P*"]["B"][:7] = np.nan                     # 이 분할은 7 %, 대상 수준 4 %
    cs = M.CandSet.build(s, y_src, raw, cci, ccv, pstar_target=dict(miss_A=0.0, miss_B=0.04))
    assert "P*" in cs.names and cs.info["P*"]["decision"] == "target"
    # pstar_target_miss: 대상 셀 전체와 대상의 채점 셀(eval_mask)의 결측(0 이하 포함) 비율. 분할에 기대지 않는다
    from polar.m1_core import TARGET
    n = 40
    df = pd.DataFrame({"loc_id": np.arange(n), TARGET: np.where(np.arange(n) % 4 == 0, np.nan, 50.0), "cci_alt": 60.0, "e5_sqrt_tdd_soil": 30.0})
    tv = pd.Series(np.where(np.arange(n) < 6, np.nan, 900.0), index=np.arange(n)); tv.iloc[6] = -1.0
    r = M.pstar_target_miss(tv, df, np.arange(n))
    ev = np.arange(n)[np.arange(n) % 4 != 0]
    assert r["miss_A"] == 7 / n and r["n_B"] == len(ev) and r["miss_B"] == float(np.mean(ev < 7))


# ---------------------------------------------------------------- (q) 판정 불가 갈래
def _tm_many(rng, name, levels, ns, draws=3, splits=(1, 2, 3), nb=10):
    """합성 TMx: levels = {방법: 오차 수준}, n 마다 같은 키 구조."""
    by = {}
    for sp in splits:
        ncell = rng.randint(3, 20, nb)
        st = H4.BlockStore(name, sp, np.repeat([f"{name[:2]}{sp}b{j}" for j in range(nb)], ncell))
        cnt = st.ncell
        st.add_sse(H.P0_KEY, cnt * 30.0 ** 2, cnt)
        for n in ns:
            for d in range(draws):
                e = rng.gamma(2.0, 0.05, st.nb)
                for m, lv in levels.items():
                    st.add_sse((m, "none", "1", "cell", n, d, -1, 0.0), cnt * (lv + e) ** 2, cnt)
        by[sp] = st
    tm = H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in splits}, 300)
    tm.expected_splits = list(splits)
    return tm


def test_q_undeterminable_branches():
    rng = np.random.RandomState(19)
    lv = {"Stack": 20.0, "P1": 20.0}
    tms = {"A1|x": _tm_many(rng, "A1|x", lv, (10, 40)), "A2|x": _tm_many(rng, "A2|x", lv, (10, 40))}
    fb = lambda r: dict(ratio=r, n_draws=9, n_fb=int(round(9 * r)), reasons={"gamma=1": int(round(9 * r))})   # noqa: E731
    fbr = {("A1|x", 10): fb(0.78), ("A2|x", 10): fb(0.67)}
    fbd = {("A1|x", 10): {(1, 10, 0)}, ("A2|x", 10): {(1, 10, 0)}}
    rows, _ = M.contrast_rows("XB-1", tms, list(tms), M._g("Stack", 10), M._g("P1", 10), 10, 2, fbr, fbd)
    mean = [r for r in rows if r.get("scope") == "MEAN"]
    assert len(mean) == 1 and mean[0]["verdict_final"] == "판정 불가" and mean[0]["verdict_note"] == M.STACK_NA_NOTE
    assert abs(mean[0]["fallback_ratio"] - (7 + 6) / 18) < 1e-12 and len([r for r in rows if r.get("scope") == "region"]) == 2
    # 한 지역만 퇴화해 풀 지역이 2 미만: 원인이 적층 미작동이라 그 사유를 적는다
    rows1, _ = M.contrast_rows("XB-1", tms, list(tms), M._g("Stack", 10), M._g("P1", 10), 10, 2, {("A1|x", 10): fb(0.78)}, fbd)
    assert rows1[-1]["scope"] == "MEAN" and rows1[-1]["verdict_final"] == "판정 불가" and rows1[-1]["verdict_note"].startswith(M.STACK_NA_NOTE)
    # 적층 미작동이 아닌 판정 불가(등록 지역 하나뿐): XB-2 n 40, 대체 없음
    tms2 = {"A1|x": _tm_many(rng, "A1|x", {"StackR": 20.0, "R1": 20.0}, (40,))}
    rows2, _ = M.contrast_rows("XB-2", tms2, ["A1|x", "A9|x"], M._g("StackR", 40), M._g("R1", 40), 40, 2, {}, {})
    assert rows2[-1]["verdict_final"] == "판정 불가" and rows2[-1].get("verdict_note", "") == ""
    # XB-4: 비열등이 하나도 없고 판정할 수 없는 n 이 있으면 판정 불가(기각이 아니다)
    ak = {"Alaska|r": _tm_many(rng, "Alaska|r", lv, (200, 500, 1000))}                      # 전량 행은 없다(행 없음)
    fbr4 = {("Alaska|r", n): fb(0.89) for n in (200, 500, 1000)}
    rows4 = []
    for n in (200, 500, 1000, -1):
        rr, _ = M.contrast_rows("XB-4", ak, ["Alaska|r"], M._g("Stack", n), M._g("P1", n), n, 1, fbr4, {})
        rows4 += [dict(r, role="주(비열등)") for r in rr if r.get("scope") == "region"]
    t = M._verdict_columns(XB.clean_rows(rows + rows2 + rows4))
    s1 = t[(t.hyp == "XB-1") & (t.n == 10) & (t.scope == "MEAN")].sentence.iloc[0]
    assert "적층이 작동하지 않아(대체 비율 0.72)" in s1 and "0.72" in s1
    assert t[t.hyp == "XB-1"].hypothesis_verdict.iloc[0].startswith("판정 불가")
    s2 = t[(t.hyp == "XB-2") & (t.n == 40) & (t.scope == "MEAN")].sentence.iloc[0]
    assert "작동하지 않아" not in s2 and "판정할 수 없었다(라벨 40개, 사유 풀 지역 2 미만)" in s2
    h4 = t[t.hyp == "XB-4"].hypothesis_verdict.iloc[0]
    q4 = t[(t.hyp == "XB-4") & (t.scope == "region")]
    assert set(q4.n) == {200, 500, 1000}
    assert h4.startswith("판정 불가(적층 미작동, 행 없음)") and "판정 불가 n = 200, 500, 1000, 전량" in h4, h4
    for n in (200, 500, 1000):
        assert "적층이 작동하지 않아(대체 비율 0.89) 판정할 수 없었다" in q4[q4.n == n].sentence.iloc[0]


# ---------------------------------------------------------------- (r) 변형 조각을 섞지 않는다
def test_r_variant_filter(tmp_path, monkeypatch):
    a = _count_args(tmp_path, monkeypatch)
    a.ha.draws_cap = 1; a.nboot = 100; a.ha.SEEDS = [0]
    for sp in (1, 2):
        c = make_tctx(seed=300 + sp, split=sp, target="Lena", nA=120, nbA=12)
        U = M.XBTUnit(a.ha, c, "Lena", grid=(10,)).run()
        rows, st, stats = U.finish()
        XB.write_shard(a.SHARDS, a.TAG["xb_t"], "Lena", "x", sp, rows, st, M.unit_cfg(a, "xb_t", "", "s"), expected=[1, 2])
        U = M.XBTUnit(a.ha, c, "Lena", "cci5y", grid=(10,)).run()
        rows, st, stats = U.finish()
        cfg = dict(M.unit_cfg(a, "xb_t", "cci5y", "s"), products={"cci5y": "다른표"})
        XB.write_shard(a.SHARDS, a.TAG["xb_t"], "Lena", "x", sp, rows, st, cfg, variant="cci5y", expected=[1, 2])
    with pytest.raises(SystemExit):                                                          # 걸러내지 않으면 설정 해시가 섞여 중단된다
        XB.load_tms(a.SHARDS, a.TAG["xb_t"], 100)
    tms, units, runs = M.load_tms_variant(a, "xb_t")
    assert set(tms) == {"Lena|x"} and len(units) == 2 and set(runs.variant.fillna("")) == {""}
    a.VARIANT = "cci5y"
    tms, units, runs = M.load_tms_variant(a, "xb_t")
    assert set(tms) == {"Lena~cci5y|x"} and len(units) == 2
    assert {s_["variant"] for s_ in M.variant_shards(a, "xb_t")} == {"cci5y"}


# ---------------------------------------------------------------- (s) runs 의 봉인 열·로컬 자원 규약
def test_s_runs_without_rmse_and_local_resources(tmp_path, monkeypatch):
    a = _count_args(tmp_path, monkeypatch)
    a.ha.draws_cap = 1; a.ha.SEEDS = [0]

    def fake_make_unit(a_, arm, target, mode, split, variant="", dry=False):
        c = make_tctx(seed=40 + int(split), split=int(split), target=target)
        return c, M.XBTUnit(a_.ha, c, target, variant, dry, grid=(0, 10))
    monkeypatch.setattr(M, "make_unit", fake_make_unit)
    monkeypatch.setattr(M, "data_sha_of", lambda a_, arm, target: "sha:test")
    u = M.run_unit(a, "xb_t", "Lena", "x", 1)
    p = XB.shard_paths(a.SHARDS, a.TAG["xb_t"], "Lena", "x", 1)
    cols = set(pd.read_csv(p["runs"], nrows=1).columns)
    assert not cols & set(M.RUNS_SEALED_COLS) and {"method", "n", "split", "draw", "stack_fallback"} <= cols
    assert u["tdd_v4_missing_allowed"] is True and u["K"] == u["xb_cands"]["K"] and u["runs_sealed_cols"] == list(M.RUNS_SEALED_COLS)
    st = H4.load_stores([p["npz"]])
    assert len(next(iter(st.values()))) > 0, "블록 SSE 는 그대로 저장한다"
    assert M.local_resource_plan("smoke", False) == dict(require=True, limit="arena_as")
    assert M.local_resource_plan("run", False) == dict(require=True, limit="arena_as")
    assert M.local_resource_plan("count", False) == dict(require=True, limit="as")
    assert M.local_resource_plan("summarize", False) == dict(require=True, limit="as")
    assert M.local_resource_plan("run", True) == dict(require=False, limit="none")
    a2 = M.parse_args(["--smoke", "--out-dir", str(tmp_path / "XB_multisource_stacking"), "--allow-missing-tdd-v4"])
    assert isinstance(a2.SMOKE_ENV, dict) and {"threads", "cores", "warn", "mem_available_gb"} <= set(a2.SMOKE_ENV)
