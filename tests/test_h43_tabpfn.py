"""scripts/3_deep_learning/h43_tabpfn_label_grid.py 단위 시험(계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6C).

가벼운 시험은 학습을 하지 않는다. 두 학습기 자리에 대체 함수(학습 목표의 평균 + 입력 1열의 선형 항)를 넣어 컨텍스트 행렬, 목표, 저장 경로만
확인한다. 실자료와 GPU 를 쓰지 않는다. TabPFN 을 실제로 적합하는 시험(표지 GPU)은 환경 변수 LGT_RUN_GPU=1 일 때만 실행한다.

가벼운 시험
(a) ctx_order: 순열, 재현성, 중첩 구조, 지역 비례(이론 상한), 지역 안 순위 순서.
(b) src_budget 의 경계값(T = 1,000, 1,001, 9,000, 9,001)과 대상 행 부분 추출(tsub).
(c) 방법별 목표 행렬: D0, R0, R1 의 컨텍스트 행과 목표. 라벨 셀은 h40.draw_cells 와 같다. 두 학습기가 같은 행렬을 받는다.
(d) 비선택 라벨과 채점 라벨을 교란해도 컨텍스트 행렬과 목표가 변하지 않는다.
(e) 키 형식과 n = 0 의 R1 = R0, 민감도 변형의 키(alpha, 방법 이름), rid 의 26번째 열, tgt 의 라벨 수 하한.
(f) 설정 해시와 재개(--resume, --rerun-partial). 스레드와 GPU 번호는 해시에 들지 않는다.
(g) GPU 목록 거부 규칙과 실행 거부 조건(--gpus 없음, 스레드 범위, 가중치 파일 없음).
(h) 기본값, 스모크 범위, 축별 h40 인자, 실행 순서 묶음.
(i) 조각 기록과 읽기: runs.csv, blocksse.npz, cells.npz, unit.json. 스모크 점검 함수.
(j) 학습 없는 계수(dry)의 적합 수가 대체 학습기 실행의 적합 수와 같다.
(k) 적합 한 건의 예외는 그 키만 실패로 남는다. ImportError 는 조각 전체의 실패다.
(l) 판정 안정성 규칙과 CUDA 메모리 부족 예외의 판별.
(m) 집계: 합성 저장소에서 L32–L37 의 대비 행과 판정 행이 만들어진다.
GPU 시험
(G) TabPFN 실제 적합(합성 자료): 예측이 유한하고 묶음 예측과 한 번 예측의 차이가 1e-3 이하다.
실행: python3 -m pytest -q tests/test_h43_tabpfn.py            (가벼운 시험, 스레드 2개)
      LGT_RUN_GPU=1 CUDA_VISIBLE_DEVICES=9 python3 -m pytest -q tests/test_h43_tabpfn.py -k G_
"""
import os
RUN_GPU = os.environ.get("LGT_RUN_GPU", "") == "1"
if not RUN_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                  # 가벼운 시험은 GPU 를 쓰지 않는다
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GPU = pytest.mark.skipif(not RUN_GPU, reason="TabPFN 을 적합하는 시험은 LGT_RUN_GPU=1 일 때만 실행한다(GPU 1장)")
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


T = _load("h43_tabpfn_label_grid", "scripts/3_deep_learning/h43_tabpfn_label_grid.py")
H = T.H
D_FEAT = len(H.FEATS)
NS, NA, NB = 400, 60, 50
CMAX, CRES, CMIN = 300, 50, 50
BASE_ARGS = ["--n-grid", "0,3,10,40,all", "--draws", "2", "--seeds", "2", "--splits", "1", "--threads", "2", "--ctx-max", str(CMAX),
             "--ctx-reserve", str(CRES), "--ctx-src-min", str(CMIN), "--model-path", "/nonexistent/lgt_test_weights.ckpt"]


def make_unit(yA_fn=None, yB_fn=None, seed=0):
    """합성 작업 단위. 특징 0 열은 행 식별자(원천 = −1−j, 대상 A = j, 채점 B = 1000 + j). 원천 지역 4개는 크기가 다르다."""
    rng = np.random.RandomState(seed)
    Xs = rng.randn(NS, D_FEAT); Xs[:, 0] = -1.0 - np.arange(NS)
    ss = rng.uniform(20, 40, NS); ys = 1.6 * ss + 3 * Xs[:, 1] + rng.randn(NS) * 2
    XA = rng.randn(NA, D_FEAT); XA[:, 0] = np.arange(NA)
    sA = rng.uniform(20, 40, NA); yA = 1.3 * sA + 3 * XA[:, 1] + rng.randn(NA) * 2
    XB = rng.randn(NB, D_FEAT); XB[:, 0] = 1000.0 + np.arange(NB)
    sB = rng.uniform(20, 40, NB); yB = 1.3 * sB + 3 * XB[:, 1] + rng.randn(NB) * 2
    XA[5, 3] = np.nan; Xs[7, 4] = np.nan                       # 결측은 NaN 그대로 넘긴다
    if yA_fn is not None:
        yA = yA_fn(yA.copy())
    if yB_fn is not None:
        yB = yB_fn(yB.copy())
    mac = np.array(["r1"] * 200 + ["r2"] * 120 + ["r3"] * 60 + ["r4"] * 20)
    return H.Ctx("T", "x", 1, "T", Xs, ys, ss, mac, XA, yA, sA, np.repeat(np.arange(6), NA // 6), XB, yB, sB, np.repeat(np.arange(5), NB // 5))


def stub_tab(a, X, y, XB, seed, cat_idx=None, check_chunk=0):
    y = np.asarray(y, float)
    return float(y.mean()) + 0.01 * np.nan_to_num(np.asarray(XB, float)[:, 1]) + 0.001 * int(seed), "", {}


def stub_cb(HA, X, y, XB, seed, cat_idx=None):
    y = np.asarray(y, float)
    return float(y.mean()) + 0.02 * np.nan_to_num(np.asarray(XB, float)[:, 1]) + 0.001 * int(seed), "", {}


@pytest.fixture()
def stub(monkeypatch):
    monkeypatch.setattr(T, "tabpfn_fit_predict", stub_tab)
    monkeypatch.setattr(T, "cb_fit_predict", stub_cb)
    return True


def args(extra=(), tmp=None):
    out = ["--out-dir", str(tmp)] if tmp is not None else []
    return T.parse_args(BASE_ARGS + out + list(extra))


def run(axis="main", c=None, extra=(), dry=False, trace=True, tmp=None):
    a = args(extra, tmp)
    HA = T.h40_args_t(a, axis)
    T.TRACE = [] if trace else None
    try:
        rows, st, stats, cells = T.run_ctx_t(c if c is not None else make_unit(), axis, a, HA, dry=dry)
        return a, HA, rows, st, stats, cells, list(T.TRACE or [])
    finally:
        T.TRACE = None


def ids(X):
    return np.asarray(X, float)[:, 0]


# ================================================================ (a) 원천 부분 추출의 순서
def test_a_ctx_order_nested_and_proportional():
    mac = np.array(["b"] * 500 + ["a"] * 300 + ["d"] * 150 + ["c"] * 50)
    o = T.ctx_order(mac, "Lena", "x", 1, 0)
    assert sorted(o.tolist()) == list(range(len(mac)))
    assert np.array_equal(o, T.ctx_order(mac, "Lena", "x", 1, 0)), "같은 (대상, 모드, 분할, seed)는 같은 순서다"
    assert not np.array_equal(o, T.ctx_order(mac, "Lena", "x", 1, 1)) and not np.array_equal(o, T.ctx_order(mac, "Lena", "x", 2, 0))
    assert not np.array_equal(o, T.ctx_order(mac, "Canada", "x", 1, 0))
    J, N = 4, len(mac)
    for m in (1, 7, 50, 333, 640, 999, 1000):
        sel = o[:m]
        assert set(o[:max(m - 1, 0)].tolist()) <= set(sel.tolist())                      # 작은 예산의 집합은 큰 예산의 집합에 포함된다
        cnt = T.region_counts(mac, sel)
        assert sum(cnt.values()) == m
        for k, c_j in (("a", 300), ("b", 500), ("c", 50), ("d", 150)):
            w = c_j / N
            assert abs(cnt.get(k, 0) - m * w) <= 0.5 * (1 - w) + w * (J - 1) / 2 + 1e-9, (m, k)   # 이론 상한
        assert T.region_dev(mac, sel) <= T.region_dev_bound(mac) + 1e-9 < J / 2
    # 지역 안에서는 순위 순서대로 들어간다: 앞의 m행에 든 지역 행은 그 지역 순열의 앞부분이다
    rng = np.random.RandomState(T.seed_of("lgt-ctx", "Lena", "x", 1, 0))
    for nm in sorted(set(mac.tolist())):                                                   # 난수 소비 순서 = 지역 이름 오름차순
        idx = np.where(mac == nm)[0]
        perm = idx[rng.permutation(len(idx))]
        got = [i for i in o[:400].tolist() if mac[i] == nm]
        assert got == perm[:len(got)].tolist(), nm
    assert T.region_dev(mac, o) == 0.0 and T.region_dev(mac, o[:0]) == 0.0
    assert T.region_dev_bound(mac) == pytest.approx(0.5 * (1 - 0.5) + 0.5 * (J - 1) / 2) and T.region_dev_bound(mac[:0]) == 0.0


# ================================================================ (b) 예산의 경계값과 대상 행 부분 추출
def test_b_src_budget_boundaries_and_tsub():
    C, R = 10000, 1000
    assert T.src_budget(17436, 0, C, R) == 9000 and T.src_budget(17436, 1000, C, R) == 9000
    assert T.src_budget(17436, 1001, C, R) == 8999
    assert T.src_budget(17436, 9000, C, R) == 1000 and T.src_budget(17436, 9001, C, R) == 999
    assert T.src_budget(3860, 0, C, R) == 3860 and T.src_budget(3860, 6139, C, R) == 3860 and T.src_budget(3860, 6141, C, R) == 3859
    assert T.src_budget(14429, 1505, C, R) == 8495 and T.src_budget(3860, 7436, C, R) == 2564           # 계획서 §6C.4 의 예
    assert T.src_budget(10, 20000, C, R) == 0
    c = make_unit()
    a = args(["--ctx-max", "100", "--ctx-reserve", "20", "--ctx-src-min", "20"])
    o = T.ctx_order(c.macro_src, c.target, c.mode, c.split, 0)
    sel = np.arange(55)
    src, tsel, fl = T.context_rows(c, o, sel, T.MAIN_SET, a, 55, 0)
    assert fl == "" and len(tsel) == 55 and len(src) == 45 and np.array_equal(src, o[:45])
    sel = np.arange(60)
    big = dict(T.MAIN_SET, dup=2)                                                          # T = 120 > 100 − 20
    src, tsel, fl = T.context_rows(c, o, sel, big, a, -1, 0)
    assert fl == "tsub" and len(tsel) == 40 and set(tsel.tolist()) <= set(sel.tolist()) and len(src) == 20
    src2, tsel2, _ = T.context_rows(c, o, sel, big, a, -1, 0)
    assert np.array_equal(tsel, tsel2) and np.array_equal(src, src2)
    src, tsel, fl = T.context_rows(c, o, sel, T.SENS_SETS["tgt"], a, -1, 0)                # tgt: 원천 행 없음, 상한은 C
    assert len(src) == 0 and len(tsel) == 60 and fl == ""


# ================================================================ (c) 방법별 목표 행렬
def test_c_method_targets_and_same_matrix(stub):
    c = make_unit()
    a, HA, rows, st, stats, cells, tr = run("main", c)
    assert stats["status"] == "ok" and len(tr) == 2 * (16 + 16 + 14)
    orders = {s_: T.ctx_order(c.macro_src, "T", "x", 1, s_) for s_ in (0, 1)}
    seen = set()
    for e in tr:
        n, d, seed, m = e["n"], e["draw"], e["seed"], e["method"]
        sel = H.draw_cells("T", "x", 1, n, d, NA)
        assert np.array_equal(e["sel"], sel) and np.array_equal(e["tsel"], sel), "라벨 셀은 h40.draw_cells 와 같다"
        nsrc = T.src_budget(NS, len(sel), CMAX, CRES)
        src = orders[seed][:nsrc]
        assert np.array_equal(e["src"], src) and nsrc == (250 if len(sel) <= 50 else 240)
        assert e["X"].shape == (nsrc + len(sel), D_FEAT) and e["X"].dtype == np.float32 and e["cat_idx"] is None
        assert np.array_equal(ids(e["X"]), np.concatenate([-1.0 - src, sel.astype(float)])), "원천 행(정렬 순) 다음에 대상 행"
        E0 = c.E0
        En = T.coef_n(c, sel, 10.0)
        if len(sel):
            Els = H.ls_E(c.yA[sel], c.sA[sel])
            assert np.isclose(En, (len(sel) * Els + 10.0 * E0) / (len(sel) + 10.0))
        else:
            assert En == E0
        want = dict(D0=np.concatenate([c.y_src[src], c.yA[sel]]), R0=np.concatenate([c.r0_src[src], c.yA[sel] - E0 * c.sA[sel]]),
                    R1=np.concatenate([c.r0_src[src], c.yA[sel] - En * c.sA[sel]]))[m]
        assert np.allclose(e["y"], want, atol=1e-12), (m, n, d, seed)
        assert not (m == "R1" and n == 0), "n = 0 의 R1 은 R0 의 적합을 쓴다(추가 적합 없음)"
        seen.add((m, n, d, seed, e["learner"]))
    by = {}
    for e in tr:
        by.setdefault((e["method"], e["n"], e["draw"], e["seed"]), {})[e["learner"]] = e
    for k, v in by.items():
        assert set(v) == set(T.LEARNERS_T), k
        assert np.array_equal(v[T.TP]["X"], v[T.CBX]["X"], equal_nan=True) and np.array_equal(v[T.TP]["y"], v[T.CBX]["y"]), k
        assert np.array_equal(v[T.TP]["XB"], v[T.CBX]["XB"], equal_nan=True)
    # 세 방법은 같은 (n, 추출, seed)에서 같은 컨텍스트 행렬을 쓴다
    for n, d, seed in {(k[1], k[2], k[3]) for k in by if k[1] != 0}:
        Xs = [by[(m, n, d, seed)][T.TP]["X"] for m in ("D0", "R0", "R1")]
        assert np.array_equal(Xs[0], Xs[1], equal_nan=True) and np.array_equal(Xs[0], Xs[2], equal_nan=True)
    # 원천 부분 집합은 n, 추출, 방법과 무관하고 seed 에 따라 다르다. n = 전량의 집합은 작은 n 의 집합의 부분집합이다
    s0 = {k[3]: set(v[T.TP]["src"].tolist()) for k, v in by.items() if k[0] == "D0" and k[1] == 3 and k[2] == 0}
    assert s0[0] != s0[1]
    for k, v in by.items():
        assert set(v[T.TP]["src"].tolist()) <= s0[k[3]]
    r = pd.DataFrame(rows)
    ml = r[r.learner != "none"]
    sha = ml.groupby(["method", "n", "draw", "seed"]).ctx_sha.nunique()
    assert (sha == 1).all() and (ml.ctx_sha.str.len() == 12).all()
    assert (ml.n_ctx == ml.n_ctx_src + ml.n_ctx_tgt).all() and ml.n_ctx.max() <= CMAX and set(r.ctx_set) == {"main"}


# ================================================================ (d) 누설: 비선택 라벨과 채점 라벨
def _free_cells(n_list=(3, 10), draws=2):
    used = set()
    for n in n_list:
        for d in range(draws):
            used |= set(H.draw_cells("T", "x", 1, n, d, NA).tolist())
    return np.array(sorted(set(range(NA)) - used))


def test_d_unselected_and_scoring_labels_never_used(stub):
    free = _free_cells()
    assert len(free) > 10

    def bump(y):
        y[free] += 500.0
        return y
    ex = ["--n-grid", "0,3,10"]
    for axis, extra in (("main", ex), ("sens", ex + ["--sens", "d3,d10,rid"])):
        _, _, r0, s0, _, _, t0 = run(axis, make_unit(), extra)
        _, _, r1, s1, _, _, t1 = run(axis, make_unit(yA_fn=bump), extra)
        assert len(t0) == len(t1) and len(t0) > 10 and s0.keys == s1.keys
        for e0, e1 in zip(t0, t1):
            assert np.array_equal(e0["X"], e1["X"], equal_nan=True) and np.array_equal(e0["y"], e1["y"]), (axis, e0["method"], e0["n"])
        S0, C0 = s0.matrices(s0.keys); S1, C1 = s1.matrices(s1.keys)
        assert np.array_equal(S0, S1) and np.array_equal(C0, C1), axis
        _, _, r2, s2, _, _, t2 = run(axis, make_unit(yB_fn=lambda y: y + 300.0), extra)
        for e0, e2 in zip(t0, t2):                                                         # 채점 셀 라벨은 채점에만 쓴다
            assert np.array_equal(e0["X"], e2["X"], equal_nan=True) and np.array_equal(e0["y"], e2["y"])
        assert not np.array_equal(S0, s2.matrices(s2.keys)[0])


# ================================================================ (e) 키 형식과 변형
def test_e_keys_and_variants(stub):
    c = make_unit()
    a, HA, rows, st, stats, cells, tr = run("main", c)
    keys = list(st.keys)
    assert keys[0] == tuple(H.P0_KEY) and all(len(k) == 8 for k in keys)
    cells_ = H.cells_of([0, 3, 10, 40, -1], 2, NA)
    assert {k for k in keys if k[0] == "P1"} == {("P1", "none", "1", "cell", n, d, -1, 0.0) for n, d in cells_}
    for m, lams in (("D0", [1.0]), ("R0", [0.25, 0.5, 1.0]), ("R1", [0.25, 0.5, 1.0])):
        want = {(m, lr, "1", "cell", n, d, s_, lam) for lr in T.LEARNERS_T for n, d in cells_ for s_ in (0, 1) for lam in lams}
        assert {k for k in keys if k[0] == m} == want, m
    assert len(keys) == 1 + 8 + 2 * 8 * 2 * 7
    for k in [k for k in keys if k[0] == "R1" and k[4] == 0]:                              # n = 0 에서 R1 = R0
        k0 = ("R0",) + tuple(k[1:])
        assert np.array_equal(st.get(k)[0], st.get(k0)[0])
    assert np.isclose(st.rmse(tuple(H.P0_KEY)), np.sqrt(np.mean((c.E0 * c.sB - c.yB) ** 2)))
    sel = H.draw_cells("T", "x", 1, 10, 1, NA)
    k = ("R1", T.TP, "1", "cell", 10, 1, 1, 0.5)
    src = T.ctx_order(c.macro_src, "T", "x", 1, 1)[:250]
    En = T.coef_n(c, sel, 10.0)
    g = stub_tab(a, None, np.concatenate([c.r0_src[src], c.yA[sel] - En * c.sA[sel]]), c.XB, 1)[0]
    assert np.isclose(st.rmse(k), np.sqrt(np.mean((En * c.sB + 0.5 * g - c.yB) ** 2)))
    assert cells is not None and len(cells.keys) == 2 * 8 * 2 * 3 and len(cells.coef) == 8
    # 민감도
    a, HA, rows, st, stats, cells, tr = run("sens", c)
    r = pd.DataFrame(rows)
    assert set(r.ctx_set) == {"sens", "d3", "d10", "rid", "tgt"} and set(r[r.ctx_set == "sens"].method) == {"P0"}
    ml = r[r.learner != "none"]
    assert set(ml[ml.ctx_set == "d3"].alpha) == {"3"} and set(ml[ml.ctx_set == "d3"].method) == {"D0", "R1"} and set(ml[ml.ctx_set == "d3"].n) == {3, 10, 40}
    assert set(ml[ml.ctx_set == "d10"].alpha) == {"10"} and set(ml[ml.ctx_set == "d10"].n) == {10, 40}
    assert set(ml[ml.ctx_set == "rid"].method) == {"D0@rid", "R1@rid"} and set(ml[ml.ctx_set == "rid"].n) == {0, 10, 40, -1}
    assert set(ml[ml.ctx_set == "tgt"].method) == {"D0@tgt", "R1@tgt"} and set(ml[ml.ctx_set == "tgt"].n) == {40, -1}
    assert (ml[ml.ctx_set == "tgt"].n_lab >= 40).all() and (ml[ml.ctx_set == "tgt"].n_ctx_src == 0).all()
    assert set(ml[ml.ctx_set.isin(["rid", "tgt"])].alpha) == {"1"} and "variant" not in r.columns
    assert (ml.n_ctx <= CMAX).all()
    d10 = ml[(ml.ctx_set == "d10") & (ml.n == 40)]
    assert set(d10.fit_flag) == {"tsub"} and set(d10.n_ctx_tgt) == {250} and set(d10.n_ctx_src) == {50} and (d10.n_nonfinite == 0).all()
    assert set(ml[(ml.ctx_set == "d3") & (ml.n == 40)].n_ctx_tgt) == {120} and set(ml[(ml.ctx_set == "d3") & (ml.n == 40)].n_ctx_src) == {180}
    for e in tr:
        if e["ctx_set"] == "rid":
            assert e["X"].shape[1] == D_FEAT + 1 and e["XB"].shape[1] == D_FEAT + 1 and e["cat_idx"] == D_FEAT
            code, new = T.macro_codes(list(c.macro_src) + [c.parent])
            nsrc = len(e["src"])
            assert np.array_equal(e["X"][:nsrc, -1], np.array([code[m] for m in c.macro_src[e["src"]]], np.float32))
            assert (e["X"][nsrc:, -1] == new).all() and (e["XB"][:, -1] == new).all()
        else:
            assert e["X"].shape[1] == D_FEAT and e["cat_idx"] is None
        if e["ctx_set"] in ("d3", "d10"):
            dup = 3 if e["ctx_set"] == "d3" else 10
            nsrc = len(e["src"])
            assert np.array_equal(ids(e["X"])[nsrc:], np.repeat(e["tsel"].astype(float), dup)), "중복 행은 행 반복이다"
            assert len(e["y"]) == nsrc + dup * len(e["tsel"])
        if e["ctx_set"] == "tgt":
            assert len(e["src"]) == 0 and np.array_equal(ids(e["X"]), e["sel"].astype(float))
    assert ("R1", T.TP, "3", "cell", 10, 0, 0, 0.25) in st and ("R1@rid", T.CBX, "1", "cell", 0, 0, 1, 1.0) in st
    assert ("D0@tgt", T.TP, "1", "cell", -1, 0, 0, 1.0) in st and ("P1", "none", "1", "cell", 3, 1, -1, 0.0) in st


# ================================================================ (f) 설정 해시와 재개
def test_f_cfg_hash_and_resume(tmp_path):
    a = args([], tmp_path)
    cfg = T.unit_cfg_t(a, "main")
    assert cfg["learners"] == list(T.LEARNERS_T) and cfg["methods"] == ["D0", "R0", "R1"] and cfg["feats"] == "x25" and cfg["ctx_max"] == CMAX
    assert "threads" not in cfg and "gpus" not in cfg and cfg["model_sha1"] == "none" and cfg["model"] == "lgt_test_weights.ckpt"
    h = H.cfg_hash(cfg)
    assert h == H.cfg_hash(T.unit_cfg_t(args(["--threads", "4", "--gpus", "9,7"], tmp_path), "main")), "스레드와 GPU 번호는 해시에 들지 않는다"
    for extra in (["--n-est", "4"], ["--ctx-max", "200"], ["--seeds", "1"], ["--kappa", "3"], ["--cb-iters", "50"], ["--ctx-reserve", "10"]):
        assert h != H.cfg_hash(T.unit_cfg_t(args(extra, tmp_path), "main")), extra
    assert H.cfg_hash(T.unit_cfg_t(a, "sens")) != h
    assert H.cfg_hash(T.unit_cfg_t(args(["--sens", "d3"], tmp_path), "sens")) != H.cfg_hash(T.unit_cfg_t(a, "sens"))
    assert H.cfg_hash(T.unit_cfg_t(args(["--sens", "d3"], tmp_path), "main")) == h, "민감도 변형 목록은 주 설정의 해시에 들지 않는다"
    p = T.shard_paths_t(a, "main", "T", "x", 1)
    assert p["runs"].name == "lgt__gpu__T__x__s1_runs.csv" and p["cells"].name == "lgt__gpu__T__x__s1_cells.npz"
    assert T.shard_paths_t(a, "sens", "T", "x", 1)["unit"].name == "lgts__gpu__T__x__s1_unit.json"
    assert T.unit_state_t(a, "main", "T", "x", 1) == (False, "조각 없음")
    p["runs"].parent.mkdir(parents=True, exist_ok=True)
    p["runs"].write_text("a\n1\n"); p["npz"].write_bytes(b"x")
    assert not T.unit_state_t(a, "main", "T", "x", 1)[0]
    p["unit"].write_text(json.dumps(dict(cfg_hash=h, status="ok")))
    assert T.unit_state_t(a, "main", "T", "x", 1) == (True, "ok")
    assert T.unit_state_t(args(["--n-est", "4"], tmp_path), "main", "T", "x", 1) == (False, "설정 불일치(cfg_hash)")
    p["unit"].write_text(json.dumps(dict(cfg_hash=h, status="failed")))
    assert T.unit_state_t(a, "main", "T", "x", 1) == (False, "이전 실행 실패")
    p["unit"].write_text(json.dumps(dict(cfg_hash=h, status="partial")))
    assert T.unit_state_t(a, "main", "T", "x", 1) == (True, "partial")
    assert not T.unit_state_t(args(["--rerun-partial"], tmp_path), "main", "T", "x", 1)[0]
    p["unit"].write_text("{broken")
    assert T.unit_state_t(a, "main", "T", "x", 1) == (False, "unit.json 을 읽을 수 없음")


# ================================================================ (g) GPU 목록과 실행 거부
def test_g_gpu_rules_and_refusals(tmp_path):
    used = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 13188, 9: 0}
    assert T.screen_gpus([9, 7, 6, 5], used) == ([9, 7, 6, 5], [])
    ok, dropped = T.screen_gpus([9, 8, 7], used)
    assert ok == [9, 7] and [g for g, _ in dropped] == [8]
    assert T.screen_gpus([9, 7], {9: 3, 7: 0})[0] == [7], "기본 기준은 0 MiB 다"
    assert T.screen_gpus([9, 7], {9: 300, 7: 600}, mem_max=500)[0] == [9]
    assert T.screen_gpus([9, 11], {9: 0})[1][0][0] == 11
    for bad in ([9, 4], [0], [3, 9, 7]):
        with pytest.raises(SystemExit) as e:
            T.screen_gpus(bad, used)
        assert "거부" in str(e.value)
    assert T.parse_args(["--gpus", "5,9,7"]).GPUS == [9, 7, 5] and T.parse_args([]).GPUS == []
    with pytest.raises(SystemExit):
        T.parse_args(["--gpus", "9,x"])
    out = ["--out-dir", str(tmp_path / "no"), "--no-summarize"]
    w = tmp_path / "w.ckpt"; w.write_bytes(b"weights")
    cases = [(["--threads", "2", "--model-path", str(w)], "--gpus"), (["--gpus", "", "--threads", "2", "--model-path", str(w)], "--gpus"),
             (["--gpus", "9,3", "--threads", "2", "--model-path", str(w)], "남겨 두는"), (["--gpus", "9", "--threads", "1", "--model-path", str(w)], "스레드"),
             (["--gpus", "9", "--threads", "8", "--model-path", str(w)], "스레드"),
             (["--gpus", "9", "--threads", "2", "--model-path", str(tmp_path / "none.ckpt")], "가중치"),
             (["--smoke", "--threads", "2", "--model-path", str(w)], "--gpus")]
    for extra, word in cases:
        with pytest.raises(SystemExit) as e:                                               # 자료를 읽기 전에 거부한다
            T.main(extra + out)
        assert "거부" in str(e.value) and word in str(e.value), extra
    assert not (tmp_path / "no").exists()
    T.check_run_args(T.parse_args(["--gpus", "9,7,6,5", "--threads", "4", "--model-path", str(w)]))
    assert T.resolve_model("tabpfn-v2-regressor.ckpt") == T.tabpfn_cache_dir() / "tabpfn-v2-regressor.ckpt"
    assert T.resolve_model(str(w)) == w and T.resolve_model("models/x.ckpt") == T.ROOT / "models/x.ckpt"


# ================================================================ (h) 기본값과 범위
def test_h_defaults_scope_and_order():
    a = T.parse_args([])
    assert a.AXES == ["main", "sens"] and a.SENS == ["d3", "d10", "rid", "tgt"] and a.TAG == "lgt" and a.threads == 2 and a.nboot == 10000
    assert a.delta_eq == 0.5 and a.delta_eq_aux == 1.0 and a.ctx_max == 10000 and a.ctx_reserve == 1000 and a.ctx_src_min == 1000
    assert a.n_est == 8 and a.pred_chunk == 4096 and a.gpu_mem_max_mib == 0 and a.save_cells and not a.cross_lg
    assert str(a.OUT).endswith("data/processed/lgt") and a.MODEL.name == "tabpfn-v2-regressor.ckpt"
    hm, hs = T.h40_args_t(a, "main"), T.h40_args_t(a, "sens")
    assert hm.TAG == "lgt" and hs.TAG == "lgts" and len(hm.TARGETS) == 27 and hs.TARGETS == list(H.LEARNER_TARGETS)
    assert hm.N_GRID == [0, 3, 10, 40, 160, 320, 1000, -1] and hs.N_GRID == [0, 3, 10, 40, 160, 320, 1000, -1]
    assert hm.SPLITS == [1, 2, 3, 4, 5] and hm.DRAWS == 5 and hs.DRAWS == 2 and hm.SEEDS == [0, 1] and hs.SEEDS == [0, 1]
    assert hm.kappa == 10.0 and hm.LAMS == [0.25, 0.5, 1.0] and hm.cb_iters == 200 and hm.threads == 2 and hm.SHARDS == a.SHARDS
    assert T.sens_grid(a, "d3") == [3, 10, 40, 160] and T.sens_grid(a, "d10") == [10, 40, 160] and T.sens_grid(a, "rid") == [0, 10, 40, 160, -1]
    assert T.sens_grid(a, "tgt") == [40, 160, 320, 1000, -1]
    s = T.parse_args(["--smoke", "--gpus", "7,9"])
    sm, ss = T.h40_args_t(s, "main"), T.h40_args_t(s, "sens")
    assert s.TAG == "lgt_smoke" and sm.TAG == "lgt_smoke" and ss.TAG == "lgts_smoke" and s.nboot == 1000 and s.GPUS == [9, 7]
    assert sm.TARGETS == [("Russia_W", "x"), ("Canada", "x")] and ss.TARGETS == [("Canada", "x"), ("Russia_W", "x")]   # sens 는 학습기 축 대상의 순서
    assert sm.SPLITS == [1] and sm.N_GRID == [0, 10, 40, -1] and sm.DRAWS == 1 and sm.SEEDS == [0] and ss.DRAWS == 1
    assert T.sens_grid(s, "d3") == [10, 40] and T.sens_grid(s, "tgt") == [40, -1]
    t = T.parse_args(["--targets", "AL-4:i,Lena"])
    assert T.h40_args_t(t, "main").TARGETS == [("AL-4", "i"), ("Lena", "x")] and T.h40_args_t(t, "sens").TARGETS == [("Lena", "x")]
    assert T.h40_args_t(T.parse_args(["--targets", "AL-4:i"]), "sens").TARGETS == []
    assert T.parse_args(["--draws", "1"])._ha == {} and T.h40_args_t(T.parse_args(["--draws", "3"]), "sens").DRAWS == 2
    assert T.parse_args(["--count-only", "--threads", "4"]).threads == 2 and T.parse_args(["--summarize-only", "--threads", "1"]).threads == 1
    assert T.parse_args(["--threads", "9"]).threads == 4 and T.parse_args(["--threads", "9"]).threads_asked == 9
    for bad in (["--axes", "main,x"], ["--sens", "d3,zz"]):
        with pytest.raises(SystemExit):
            T.parse_args(bad)
    g = [T.unit_group("main", "Lena", "x"), T.unit_group("sens", "Canada", "x"), T.unit_group("main", "Alaska", "x"),
         T.unit_group("main", "AL-2", "x"), T.unit_group("main", "Russia_C", "x"), T.unit_group("main", "AL-2", "i"),
         T.unit_group("main", "LE-1", "x"), T.unit_group("sens", "Alaska", "x"), T.unit_group("sens", "CA-2", "x")]
    assert g == [0, 1, 2, 3, 3, 4, 4, 5, 5]
    assert T.est_fit_s(T.TP, 9900, 1000) == pytest.approx(0.5 + 6.7 + 0.6) and T.est_fit_s(T.CBX, 10000, 0) == pytest.approx(0.4)
    assert T.DESIGN_FITS == dict(main=16800, d3=1424, d10=1000, rid=1440, tgt=1108)


# ================================================================ (i) 조각 기록과 읽기
def test_i_shard_write_read_and_smoke_report(stub, tmp_path):
    X = T._load_h42()
    c = make_unit()
    units = []
    for axis in ("main", "sens"):
        a, HA, rows, st, stats, cells, _ = run(axis, c, tmp=tmp_path, trace=False)
        u = T.write_shard_t(a, axis, c, rows, st, stats, cells, 1.5, extra=dict(gpu_mem_peak_mib=10.0, env=T.env_info()))
        units.append(u)
        p = T.shard_paths_t(a, axis, "T", "x", 1)
        assert all(p[k].exists() for k in ("runs", "npz", "cells", "unit")) and T.unit_state_t(a, axis, "T", "x", 1) == (True, "ok")
        assert not list(p["runs"].parent.glob("*.tmp*")), "임시 파일이 남지 않는다"
        uj = json.loads(p["unit"].read_text())
        assert uj["cfg_hash"] == H.cfg_hash(T.unit_cfg_t(a, axis)) and uj["axis"] == axis and uj["part"] == "gpu" and uj["status"] == "ok"
        assert uj["code_sha"] == T.code_sha_t() and uj["code_sha_h40"] == H.code_sha() and uj["has_cells"] and uj["threads"] == 2
        assert set(uj["ctx"]) == {"0", "1"} and all("regions" in v for s_ in uj["ctx"].values() for v in s_.values())
        assert uj["n_fit_total"] == sum(uj["n_fit_detail"].values()) and set(uj["n_fit_detail"]) == set(uj["sec_detail"]) == set(uj["rows_detail"])
        with np.load(p["cells"], allow_pickle=False) as z:
            assert z["P"].dtype == np.float32 and z["P"].shape == (len(z["keys"]), NB) and len(z["E"]) == len(z["keys"])
            assert np.allclose(z["y"], c.yB.astype(np.float32)) and float(z["E0"]) == pytest.approx(c.E0)
            k0 = json.loads(str(z["keys"][0]))
            assert len(k0) == 8 and k0[7] in ("pred", "g") and len(z["coef_keys"]) == len(z["coef_E"])
    a = args([], tmp_path)
    sh = T.find_shards_t(a, X)
    assert [s_["axis"] for s_ in sh] == ["main", "sens"] and [s_["tag"] for s_ in sh] == ["lgt", "lgts"]
    info = T.check_cfg_t(a, sh, units)
    assert set(info) == {"lgt|main", "lgts|sens"}
    runs = T.read_runs_t(sh)
    assert set(T.RUN_COLS) <= set(runs.columns) and runs.alpha.map(type).eq(str).all()
    assert not runs.duplicated(subset=["target", "mode", "split"] + list(H.KEY_COLS)).any()
    assert int((runs.method == "P0").sum()) == 1 and len(T.failed_rows(runs)) == 0
    stores = T.load_stores([s_["npz"] for s_ in sh])
    st = stores[("T|x", 1)]
    assert ("R1", T.TP, "1", "cell", 40, 0, 0, 0.25) in st and ("R1", T.TP, "10", "cell", 40, 1, 1, 0.25) in st
    rep = T.smoke_report(units, runs, stores)
    assert rep["same_matrix"] is True and rep["n_pairs"] > 20 and rep["r1_eq_r0_n0"] is True and rep["n_r1_r0"] == 2 * 2 * 3
    assert rep["gpu_mem_peak_mib"] == 10.0 and rep["gpu_mem_reserved_peak_mib"] is None and rep["chunk_maxdiff"] is None
    units[0]["cfg_hash"] = "zzz"
    sh2 = sh + [dict(sh[0])]
    with pytest.raises(SystemExit):
        T.check_cfg_t(a, sh2, [units[0], units[1], dict(units[0], cfg_hash="yyy")])
    tt = H.timing_table(units)
    assert set(tt.axis) == {"main", "d3", "d10", "rid", "tgt"} and set(tt.learner) == set(T.LEARNERS_T) and (tt.n_fit > 0).all()


# ================================================================ (j) 학습 없는 계수
def test_j_dry_count_matches_stub_run(stub):
    for axis in ("main", "sens"):
        _, _, rows, st, stats, cells, _ = run(axis)
        _, _, rows_d, st_d, stats_d, cells_d, tr = run(axis, dry=True)
        assert stats_d["n_fit_detail"] == stats["n_fit_detail"] and stats_d["rows_detail"] == stats["rows_detail"], axis
        assert stats_d["n_rows"] == stats["n_rows"] == len(rows) and rows_d == [] and cells_d is None and tr == []
        assert stats_d["n_ctx_max"] == stats["n_ctx_max"] <= CMAX and stats_d["ctx"] == stats["ctx"]
        _, _, _, _, stats_b, _, _ = run(axis, extra=["--dry-build"], dry=True)
        assert stats_b["n_fit_detail"] == stats["n_fit_detail"] and sum(stats_b["fail"].values()) == 0
    _, _, _, _, stats, _, _ = run("main")
    assert stats["n_fit_detail"] == {f"main|{lr}|{m}": k for lr in T.LEARNERS_T for m, k in (("D0", 16), ("R0", 16), ("R1", 14))}


# ================================================================ (k) 실패 처리
def test_k_fit_failure_is_per_key(monkeypatch):
    def bad_tab(a, X, y, XB, seed, cat_idx=None, check_chunk=0):
        if len(y) == 250 + 10 and seed == 1:
            raise RuntimeError("CUDA error: 시험용 실패")
        return stub_tab(a, X, y, XB, seed, cat_idx)
    monkeypatch.setattr(T, "tabpfn_fit_predict", bad_tab)
    monkeypatch.setattr(T, "cb_fit_predict", stub_cb)
    _, _, rows, st, stats, _, _ = run("main")
    r = pd.DataFrame(rows)
    bad = r[r.fit_flag == "fail"]
    assert stats["status"] == "partial" and len(stats["errors"]) > 0 and set(bad.learner) == {T.TP} and set(bad.n) == {10} and set(bad.seed) == {1}
    assert (bad.n_nonfinite == NB).all() and len(bad) == 2 * (1 + 3 + 3)
    assert not any(k[1] == T.TP and k[4] == 10 and k[6] == 1 for k in st.keys), "실패한 키는 저장하지 않는다"
    assert any(k[1] == T.CBX and k[4] == 10 and k[6] == 1 for k in st.keys) and any(k[1] == T.TP and k[4] == 10 and k[6] == 0 for k in st.keys)
    assert len(T.failed_rows(r)) == len(bad)

    def nan_tab(a, X, y, XB, seed, cat_idx=None, check_chunk=0):
        p, fl, ex = stub_tab(a, X, y, XB, seed, cat_idx)
        p[0] = np.nan
        return p, fl, ex
    monkeypatch.setattr(T, "tabpfn_fit_predict", nan_tab)
    _, _, rows, st, stats, _, _ = run("main")
    r = pd.DataFrame(rows)
    assert stats["status"] == "partial" and set(r[r.learner == T.TP].fit_flag) == {"nonfinite"} and not any(k[1] == T.TP for k in st.keys)

    def all_bad(a, X, y, XB, seed, cat_idx=None, check_chunk=0):
        raise ValueError("시험용 실패")
    monkeypatch.setattr(T, "tabpfn_fit_predict", all_bad)
    monkeypatch.setattr(T, "cb_fit_predict", lambda *q, **k: all_bad(*q))
    assert run("main")[4]["status"] == "failed"

    def imp(a, X, y, XB, seed, cat_idx=None, check_chunk=0):
        raise ImportError("tabpfn 없음")
    monkeypatch.setattr(T, "tabpfn_fit_predict", imp)
    with pytest.raises(ImportError):
        run("main")
    # 학습 목표에 비유한 값이 있으면 그 적합을 실패로 기록한다
    monkeypatch.setattr(T, "tabpfn_fit_predict", stub_tab)
    monkeypatch.setattr(T, "cb_fit_predict", stub_cb)
    sel = H.draw_cells("T", "x", 1, 3, 0, NA)

    def hole(y):
        y[sel[0]] = np.nan
        return y
    _, _, rows, st, stats, _, _ = run("main", make_unit(yA_fn=hole), ["--n-grid", "0,3"])
    r = pd.DataFrame(rows)
    assert set(r[(r.n == 3) & (r.draw == 0) & (r.learner != "none") & (r.method != "P1")].fit_flag) == {"fail"} and stats["status"] == "partial"


# ================================================================ (l) 안정성 규칙과 예외 판별
def test_l_stability_rule_and_oom_detection():
    X = T._load_h42()

    def r(v, pool="Lena|x,Canada|x"):
        return dict(verdict4=v, pool_regions=pool, delta=0.0)
    assert T.stability(X, r("우세"), r("우세")) == "강건" and T.stability(X, r("동등"), r("미결정")) == "강건"
    assert T.stability(X, r("미결정"), r("동등")) == "강건" and T.stability(X, r("우세"), r("열세")) == "의존"
    assert T.stability(X, r("열세"), r("우세")) == "의존" and T.stability(X, r("우세"), r("미결정")) == "약화"
    assert T.stability(X, r("동등"), r("열세")) == "약화"
    assert T.stability(X, r("우세"), None) == "판정 불가" and T.stability(X, r("판정 불가"), r("우세")) == "판정 불가"
    assert T.stability(X, r("우세"), r("우세", "Lena|x")) == "판정 불가(풀 지역 불일치)"

    class OutOfMemoryError(RuntimeError):
        pass

    class TabPFNCUDAOutOfMemoryError(Exception):
        pass
    assert T.is_cuda_oom(OutOfMemoryError("x")) and T.is_cuda_oom(TabPFNCUDAOutOfMemoryError("x"))
    assert T.is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate")) and not T.is_cuda_oom(RuntimeError("shape mismatch"))
    assert not T.is_cuda_oom(MemoryError()) and not T.is_cuda_oom(ValueError("out of memory"))


# ================================================================ (m) 집계: 합성 저장소
def K(method, n, lam, lr="none", alpha="1", seed=None):
    return (method, lr, str(alpha), "cell", int(n), 0, (-1 if lr == "none" else 0) if seed is None else seed, float(lam))


def _mk_tm(name, nb, seed, specs, splits=(1, 2), nboot=200, p0_sd=6.0):
    """합성 저장소. specs = {키: (오차 SD, 잡음 묶음 이름)}. 같은 묶음의 키는 같은 단위 잡음에 SD 를 곱한 오차를 쓴다."""
    rng = np.random.RandomState(seed)
    by = {}
    for sp in splits:
        blocks = np.repeat(np.arange(nb), 5)
        st = H.BlockStore(name, sp, blocks)
        y = rng.randn(len(blocks)) * 5 + 50
        noise = {"p0": rng.randn(len(y))}
        st.add(H.P0_KEY, y, y + p0_sd * noise["p0"])
        for k, (sd, grp) in specs.items():
            if grp not in noise:
                noise[grp] = rng.randn(len(y))
            st.add(k, y, y + float(sd) * noise[grp])
        by[sp] = st
    tm = H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in splits}, nboot)
    tm.base_target = tm.target
    return tm


def _specs(tp_sd=1.0, cb_sd=1.0, same=True, d0_sd=9.0, ns=(0, 3, 10, 40, 160, 320, 1000, -1)):
    sp = {}
    for n in ns:
        sp[K("P1", n, 0.0)] = (3.0, f"p1{n}")
        for lam in (0.25, 0.5, 1.0):
            sp[K("R1", n, lam, T.TP)] = (tp_sd, f"r{n}")
            sp[K("R1", n, lam, T.CBX)] = (cb_sd, f"r{n}" if same else f"rc{n}")
            sp[K("R0", n, lam, T.TP)] = (tp_sd, f"r{n}")
            sp[K("R0", n, lam, T.CBX)] = (cb_sd, f"r{n}" if same else f"rc{n}")
        sp[K("D0", n, 1.0, T.TP)] = (d0_sd, "p0")
        sp[K("D0", n, 1.0, T.CBX)] = (d0_sd, "p0")
    for n in (10, 40):
        for lam in (0.25, 0.5, 1.0):
            for lr in T.LEARNERS_T:
                sp[K("R1", n, lam, lr, alpha="3")] = (1.0, f"r{n}")
                sp[K("R1", n, lam, lr, alpha="10")] = (1.0, f"r{n}")
                sp[K("R1@rid", n, lam, lr)] = (1.0, f"r{n}")
    for lam in (0.25, 0.5, 1.0):
        for lr in T.LEARNERS_T:
            sp[K("R1@tgt", 40, lam, lr)] = (2.5 if lr == T.TP else 1.0, "r40")
    return sp


def _tms(**kw):
    names = [f"{t}|x" for t in H.MAIN4] + [f"{H.ALASKA}|x"]
    return {nm: _mk_tm(nm, 10, 100 + i, _specs(**kw)) for i, nm in enumerate(names)}


def _verdict(tests, tid, **where):
    v = tests[(tests.test_id == tid) & tests.scope.isin(["verdict", "verdict_aux"])]
    for k, val in where.items():
        v = v[v[k] == val]
    return [str(s_) for s_ in v.verdict]


def test_m_aggregation_on_synthetic_stores():
    ns = T.SimpleNamespace(nboot=200, delta_eq=0.5, delta_eq_aux=1.0)
    tests = T.build_tests_t(ns, _tms(), None, None)
    assert set(tests.test_id) == {"L32", "L33", "L34", "L35", "L36", "L37"} and set(tests.item) == {"LGT"}
    assert not tests.confirmatory.any(), "LGT 의 가설은 모두 보조다"
    assert set(tests[tests.scope.isin(["verdict", "verdict_aux"])].scope) == {"verdict_aux"}
    prim = tests[tests.primary == True]                                                    # noqa: E712
    assert sorted(prim.test_id.value_counts().to_dict().items()) == [("L32", 5), ("L33", 6), ("L34", 2), ("L35", 1)], "Holm 묶음의 주 대비는 14개다"
    assert prim.holm_p.notna().all() and set(prim[prim.test_id == "L32"].lam) == {0.25}
    v32 = _verdict(tests, "L32")
    assert len(v32) == 1 and v32[0].startswith("동등(한계 0.5 cm"), v32                   # 두 학습기의 예측이 같다(Δ = 0)
    v33 = _verdict(tests, "L33")
    assert len(v33) == 1 and "희소 라벨에서도 TabPFN 잔차의 순가치가 있다" in v33[0] and "4지역 평균 n=3" in v33[0], v33
    v34 = _verdict(tests, "L34")
    assert len(v34) == 2 and v34[0].startswith("지지") and "오차가 작다" in v34[1], v34
    assert _verdict(tests, "L35") == ["지지"]
    assert _verdict(tests, "L36") == ["TabPFN 에서도 앵커 + 잔차 구조가 직접 구조보다 오차가 작다"]
    v37 = {v: _verdict(tests, "L37", variant=v) for v in ("d3", "d10", "rid", "tgt")}
    assert v37["d3"] == ["강건"] and v37["d10"] == ["강건"] and v37["rid"] == ["강건"], v37
    assert len(v37["tgt"]) == 1 and "원천 컨텍스트가 기여한다(n=40)" in v37["tgt"][0], v37
    stab = tests[tests.scope == "stability"]
    assert len(stab) == 2 * 2 * 3 + 2 and set(stab.stability) <= {"강건", "약화", "의존"}
    bl = tests[(tests.test_id == "L32") & (tests.scope == "MEAN") & (tests.role == "보조")]
    assert dict(zip(bl.n, bl.blind)) == {0: True, 10: False, 40: False, 160: True, -1: True}
    assert not tests[(tests.test_id == "L32") & tests.scope.isin(["verdict_aux"])].blind.any()
    assert not tests[(tests.test_id == "L37") & (tests.scope == "verdict_aux") & tests.variant.isin(["d3", "rid"])].blind.any()
    assert tests[(tests.test_id == "L37") & (tests.scope == "verdict_aux") & tests.variant.isin(["d10", "tgt"])].blind.all()
    reg = tests[(tests.scope == "region") & (tests.n == 1000) & (tests.test_id == "L33")]
    assert set(reg.target) == {"Lena|x", f"{H.ALASKA}|x"} and len(reg) == 4
    assert (tests[tests.scope == "MEAN3"].role == "보조(3지역)").all() and len(tests[tests.scope == "MEAN3"]) > 0
    # TabPFN 이 같은 컨텍스트 CatBoost 보다 나쁜 경우(독립 잡음, SD 1 대 3)
    t2 = T.build_tests_t(ns, _tms(tp_sd=3.0, cb_sd=1.0, same=False), None, None)
    v = _verdict(t2, "L32")
    assert len(v) == 1 and "TabPFN 열세" in v[0] and "n=0" in v[0], v
    # 대비가 없는 경우: 행 없음과 판정 불가
    t3 = T.build_tests_t(ns, _tms(ns=(0, 10)), None, None)
    assert _verdict(t3, "L32")[0].startswith("판정 불가") and _verdict(t3, "L35")[0].startswith("판정 불가")
    assert _verdict(t3, "L36")[0].startswith("판정 불가") and "부분(" in _verdict(t3, "L33")[0]
    assert all(v.startswith("판정 불가") for v in _verdict(t3, "L37", variant="tgt"))
    t4 = T.build_tests_t(ns, {}, None, None)
    assert all(v.startswith("판정 불가") for tid in ("L32", "L33", "L35", "L36") for v in _verdict(t4, tid))


# ================================================================ (G) TabPFN 실제 적합(GPU)
@GPU
def test_G_tabpfn_real_fit_and_chunks():
    a = T.parse_args(["--gpus", "9", "--threads", "2"])
    assert a.MODEL.exists(), "가중치 파일이 없다"
    rng = np.random.RandomState(0)
    X = rng.randn(600, D_FEAT).astype(np.float32); X[3, 2] = np.nan
    y = 2.0 * X[:, 1] + rng.randn(600) * 0.1
    XB = rng.randn(90, D_FEAT).astype(np.float32)
    p, fl, ex = T.tabpfn_fit_predict(a, X, y, XB, 0, None, check_chunk=45)
    assert p.shape == (90,) and np.isfinite(p).all() and fl == "" and ex["chunk_maxdiff"] <= T.SMOKE_CHUNK_TOL
    assert np.corrcoef(p, 2.0 * XB[:, 1])[0, 1] > 0.9
    Xr = np.c_[X, rng.randint(0, 5, 600)].astype(np.float32); XBr = np.c_[XB, np.full(90, 5)].astype(np.float32)
    p2, _, _ = T.tabpfn_fit_predict(a, Xr, y, XBr, 0, D_FEAT)
    assert np.isfinite(p2).all()
    q, _, _ = T.cb_fit_predict(T.h40_args_t(a, "main"), Xr, y, XBr, 0, D_FEAT)
    assert np.isfinite(q).all() and q.shape == (90,)
