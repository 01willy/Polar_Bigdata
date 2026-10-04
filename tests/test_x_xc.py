"""scripts/3_deep_learning/x_workflow_end_to_end.py 단위 시험(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.3절 '구현·시험' (a)부터 (i)까지).

계획은 시험 파일 이름을 tests/test_x_workflow.py 로 적었고 작업 지시는 tests/test_x_xc.py 다. 내용은 계획의 (a)부터 (i)까지를 그대로 담는다.
합성 자료 시험(라벨 값을 화면에 쓰지 않는다)
(a) 배치 순서의 앞 n 개 = n 의 집합(중첩): S2, S4, S8a·b·c·p, 무작위 중첩 순열. 일정을 줄여도 앞부분이 같다.
(b) W·W+ 선택이 선택 라벨만 쓴다(누설 시험): 선택 밖 A 라벨과 B 라벨을 바꿔도 XC 단위(W+ 가짜 공급자 포함)와 XC-r 단위의 모든 예측이 같다.
    음성 대조: 선택 밖 라벨을 읽는 공급자는 시험에 걸린다. 칸 단위 판(b3, b4): (배치, n, 추출) 칸마다 그 칸의 선택 집합 밖 라벨을 바꿔도
    그 칸의 키 예측이 같다(집합 사이 누설). 음성 대조: 다른 칸(같은 n·추출의 무작위 집합) 라벨을 읽는 공급자는 합집합 판을 통과하고 칸 단위 판에 걸린다.
(c) b10 은 첫 10개 라벨만 쓴다: 진단값 = n 10 집합의 mean(E0·s − y), 그 밖 라벨을 바꿔도 같다.
(d) A5 = h54 TUnit 의 W(합성 자료): 관문 단위(h40.draw_cells, placement 'cell')의 W·R1·R2·D1·P1·P2 키가 TUnit 과 블록 SSE 까지 같고,
    무작위 팔의 W 선택이 TUnit.select_w 와 같다.
(e) 분할 제외 규칙이 부록 A 의 표를 다시 낸다(실제 자료, 세기 범주: 셀 수·블록 수만).
(f) 합성 저장소에서 판정 문구 생성: 다섯 갈래, 보정 전 유의, 지역 열세 문장, 레나 행, XC-2 의 독립 지역 문장, F3 0곳 문장, Holm m = 8.
(g) Algorithm P 의 블록 할당량(power_alloc, γ 0·0.5·1)과 결정성, 블록 순서 β, 블록 안 farthest-first(전수 대조).
(h) two_stage_ci 가 h39 l43_region 과 같다(팔 표지 ('block', 'cell')). 2단 공통 판은 추출이 하나일 때 h42.boot_delta_common 과 같다.
(i) 고정 순서 우월 시험 XC-2s 는 비열등이 성립하지 않은 n 에서 시험하지 않는다.
그 밖: 세기 범주(--count-only)는 라벨을 지운 문맥에서 같은 수를 내고 화면에 라벨 유래 통계를 쓰지 않는다(실제 자료 판 포함), 실행 보호,
부록 XC-0 선택 파일(본 실행은 사이드카 필수, XD 산출 확인, sha256 대조), 조각 → 집계 → 봉인의 왕복(합성 적합), S1(규칙 5)의 '시험하지 않음'
(XC-5·S-XC3 와 같은 키 대비 XC-F3·S-XC0·S-XC8·F2), 부록 XC-F3 전의 잠정 Holm 표(_provisional)와 재표집 이탈 이름(_nboot<N>),
문장의 의존 표지(분할 독립 가정 의존, 추출 변동 의존), Holm 동률 배수, 독립 지역 문장의 평가 지역 수, '선택 계열'·'소수 블록' 표지,
같은 라벨 집합 XC 팔 대비의 보조 2단 CI, 조각의 결과 열 제거와 교차검증 RMSE 봉인, 스모크의 분할(평가 분할 밖)과 봉인 조각 폴더,
XC-F3 단위의 W+ 미사용, 본 실행의 배치 구현·W+ 강제, Algorithm P 결정성 점검(관문 항목), 로컬 자원 규약(스모크 코어·스레드, 메모리 대기).
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 prlimit --as=10737418240 taskset -c <코어 4개>
      python3 -m pytest -q -p no:cacheprovider tests/test_x_xc.py
      (MALLOC_ARENA_MAX=2 가 없으면 CatBoost 적합을 되풀이할 때 가상 주소 공간이 10 GB 를 넘어 prlimit --as 아래에서 CatBoost 가 멈춘다.
       실측 VmPeak 10.8 GB 대 2.8 GB, RSS 는 둘 다 0.2 GB 대)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
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
if "xbatch_core" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("xbatch_core", ROOT / "scripts" / "3_deep_learning" / "xbatch_core.py")
    _m = importlib.util.module_from_spec(_spec)
    sys.modules["xbatch_core"] = _m
    _spec.loader.exec_module(_m)
import x_workflow_end_to_end as XC                                          # noqa: E402

XC.limit_malloc_arenas(2)                                                   # 실행 중 mallopt 로도 상한을 둔다(명령의 환경 변수와 같은 효과)

XB = XC.XB
H, X, W, H4 = XB.H, XB.X, XB.W, XB.H4
LO = XB.LO

HEAVY = pytest.mark.skipif(os.environ.get("XC_RUN_HEAVY", "") != "1", reason="실제 자료 적합 시험은 XC_RUN_HEAVY=1 일 때만(공유 서버 보호)")
REAL = pytest.mark.skipif(not (ROOT / "data/processed/fidelity_base_v3.csv").exists()
                          or not all((XB.LGD_DIR / s / "fidelity_base_v3.csv").exists() for s in XB.RUN_TABLE_DIRS),
                          reason="실제 자료(v3, LGD 실행 표)가 없다")


def _load_h39():
    name = "h39_scenarios"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "2_evaluation" / "h39_scenarios.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def hargs(grid=(10, 20), xcr_grid=(20, 40), draws=2, seeds=1, cb_iters=10, nboot=200, wf4_grid=None):
    a = XB.h54_args(exp="xc", splits=[1, 201, 202, 211], grid=list(grid), threads=1, cb_iters=cb_iters, seeds=seeds, nboot=nboot, draws_cap=draws)
    a.G["xcr"] = list(xcr_grid)
    a.G["wf4"] = list(wf4_grid or grid)
    a.XC_DRAWS = draws
    a.XCR_DRAWS = draws
    return a


def make_tctx(seed=0, split=201, target="T", mode="x", nA=160, nbA=16, nB=60, nbB=6, y_shift=None):
    """합성 전이 문맥(h40.Ctx). 원천 300행(두 지역), A nA 셀 nbA 블록(블록마다 셀 수가 다르다), B 블록 번호는 분할마다 다르다."""
    rng = np.random.RandomState(seed)
    D = len(XB.FEATS)
    ns = 300
    Xs = rng.randn(ns, D); ss = rng.uniform(20, 40, ns); ys = 1.6 * ss + 3 * Xs[:, 1] + 2 * rng.randn(ns)
    XA = rng.randn(nA, D); sA = rng.uniform(20, 40, nA); yA = 1.3 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    XBm = rng.randn(nB, D); sB = rng.uniform(20, 40, nB); yB = 1.3 * sB + 3 * XBm[:, 1] + 2 * rng.randn(nB)
    sizes = np.maximum(1, rng.multinomial(nA - nbA, np.ones(nbA) / nbA) + 1)
    blkA = np.repeat(np.arange(nbA), sizes)[:nA]
    if y_shift is not None:
        yA, yB = y_shift(yA.copy(), yB.copy())
    return H.Ctx(target, mode, split, target, Xs, ys, ss, np.repeat(["r1", "r2"], ns // 2), XA, yA, sA, blkA,
                 XBm, yB, sB, 500 + 10 * split + np.repeat(np.arange(nbB), nB // nbB), meta=dict(dup_of=-1, valid=True))


def make_rctx(seed=0, split=211, target="T", nA=160, nbA=10, nB=60, nbB=6):
    """합성 지역 내 문맥(h54.RCtx, tests/test_h54_workflow.py 와 같은 구성)."""
    rng = np.random.RandomState(seed)
    D = len(W.FEATS)
    XA = rng.randn(nA, D).astype(np.float32); XBm = rng.randn(nB, D).astype(np.float32)
    sA, sB = rng.uniform(20, 40, nA), rng.uniform(20, 40, nB)
    yA = 1.4 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    yB = 1.4 * sB + 3 * XBm[:, 1] + 2 * rng.randn(nB)
    blkA = np.repeat(np.arange(nbA), int(np.ceil(nA / nbA)))[:nA]; blkB = 1000 + np.repeat(np.arange(nbB), int(np.ceil(nB / nbB)))[:nB]
    latA = 65 + 0.5 * (blkA % 5) + 0.05 * rng.rand(nA); lonA = -150 + 0.5 * (blkA // 5) + 0.05 * rng.rand(nA)
    latB = 68 + 0.5 * ((blkB - 1000) % 5) + 0.05 * rng.rand(nB); lonB = -150 + 0.5 * ((blkB - 1000) // 5) + 0.05 * rng.rand(nB)
    return W.RCtx(target, split, target, XA, yA, sA, blkA, latA, lonA, XBm, yB, sB, blkB, latB, lonB, 1.6,
                  meta=dict(dup_of=-1, valid=True, n_valid_splits=1, n_unique_splits=1))


class FakeStack:
    """W+ 가짜 공급자(시험 전용). 적층 = P1 과 같은 형식, StackR = 원천 행 + tr 라벨의 잔차 모형.
    leak = True(음성 대조)이면 교차검증에서 Stack 이 te 라벨을 그대로 내어(선택 라벨이라 누설은 아니다) W+ 가 Stack 을 고르게 하고,
    최종 예측에 선택 밖 라벨(A 전체 평균)을 더한다(누설). leak = 'cross' 이면 최종 예측에 같은 (n, 추출)의 무작위 라벨 집합의 평균을 더한다
    (집합 사이 누설: 무작위 집합은 다른 칸의 선택 라벨이라 합집합 판 시험 (b)는 잡지 못하고 칸 단위 시험 (b2)가 잡는다)."""

    def __init__(self, leak=False):
        self.leak = leak

    def cv(self, unit, tr, te, n, d, fold, seed):
        c = unit.c
        E1, _ = unit.coefs(tr)
        (g,) = unit.fit(LO, "StackRcv", lambda: unit.rows_R(tr, E1 * c.sA[tr]), unit.nsrc + len(tr), seed, [c.XA[te]], n, d, tr, fold=fold)
        base = np.array(c.yA[te], float) if self.leak else E1 * c.sA[te]
        return {"Stack": base, "StackR@0.25": E1 * c.sA[te] + 0.25 * np.asarray(g, float)}

    def final(self, unit, sel, n, d, seed):
        c = unit.c
        E1, _ = unit.coefs(sel)
        (g,) = unit.fit(LO, "StackR", lambda: unit.rows_R(sel, E1 * c.sA[sel]), unit.nsrc + len(sel), seed, [c.XB], n, d, sel)
        if self.leak == "cross":
            extra = float(np.mean(np.asarray(c.yA, float)[unit.rand_set(n, d)])) * 1e-2
        else:
            extra = float(np.nanmean(c.yA)) * 1e-2 if self.leak else 0.0
        return {"Stack": E1 * c.sB + extra, "StackR@0.25": E1 * c.sB + 0.25 * np.asarray(g, float)}


def provider(leak=False):
    f = FakeStack(leak)
    return XC.WPlusProvider(f.cv, f.final, "test")


ALGP = XC.algp_spec("S8a", source="시험")


# ---------------------------------------------------------------- (a) 중첩
@pytest.mark.parametrize("name", list(XC.ALGP_CANDIDATES))
def test_a_placement_nested(name):
    rng = np.random.RandomState(3)
    Z = rng.randn(300, 6); blk = np.repeat(np.arange(25), 12).astype(str)
    sp = XC.algp_spec(name)
    full = XC.placement_order(sp, Z, blk, [10, 40, 160], 77, "ref")
    short = XC.placement_order(sp, Z, blk, [10, 40], 77, "ref")
    assert len(full) == 160 and len(set(full.tolist())) == 160 and full.min() >= 0 and full.max() < 300
    assert np.array_equal(full[:40], short), "일정을 줄여도 앞 40개가 같아야 한다(중첩)"
    for n1, n2 in ((10, 40), (40, 160)):
        assert set(full[:n1].tolist()) <= set(full[:n2].tolist())
    assert np.array_equal(full, XC.placement_order(sp, Z, blk, [10, 40, 160], 77, "ref")), "결정적이어야 한다"


def test_a2_random_nested_and_unit_sets():
    o1 = XB.draw_nested(XC.RAND_TAG, "T", "x", 201, 160, 0, 300)
    for n in (10, 40):
        assert np.array_equal(XB.draw_nested(XC.RAND_TAG, "T", "x", 201, n, 0, 300), o1[:n])
    a = hargs(grid=(10, 40, 100))
    U = XC.XCUnit(a, make_tctx(), "T", ALGP, dry=True)
    for d in range(2):
        s10, s40, s100 = (set(U.algp_set(n, d).tolist()) for n in (10, 40, 100))
        r10, r40 = (set(U.rand_set(n, d).tolist()) for n in (10, 40))
        assert s10 <= s40 <= s100 and r10 <= r40 and len(s100) == 100


# ---------------------------------------------------------------- (b) 누설
def test_b_leakage_xc_unit_with_wplus():
    a = hargs(grid=(10, 20), draws=2, seeds=1)
    c = make_tctx(nA=160)

    def run_fn(cc, pv=provider()):
        XC.XCUnit(a, cc, "T", ALGP, wplus=pv).run()
    res = XB.leakage_invariance(run_fn, c)
    assert res["ok"] and res["same_keys"] and res["max_abs_diff"] == 0.0 and res["n_keys"] > 20
    assert 0 < res["n_keep"] < res["n_A"], "선택 밖 A 라벨이 있어야 시험이 의미가 있다"
    with XB.h54_trace() as (p1, tr):
        XC.XCUnit(a, c, "T", ALGP, wplus=provider()).run()
    assert any(k[4] == "W+" for k in p1), "W+ 키가 저장되어야 한다"
    keep = XB.selected_union(tr)
    leaky = provider(leak=True)                                            # 음성 대조: 선택 밖 라벨을 읽는 공급자
    with XB.h54_trace() as (q1, _):
        XC.XCUnit(a, c, "T", ALGP, wplus=leaky).run()
    with XB.h54_trace() as (q2, _):
        XC.XCUnit(a, XB.perturb_labels(c, keep, seed=5), "T", ALGP, wplus=leaky).run()
    assert not XB.compare_predictions(q1, q2)["ok"], "선택 밖 라벨을 읽는 공급자가 시험에 걸리지 않았다"


def test_b2_leakage_xcr_unit():
    a = hargs(xcr_grid=(20, 40), draws=1, seeds=1)
    c = make_rctx(nA=200)

    def run_fn(cc):
        XC.XCRUnit(a, cc, ALGP).run()
    res = XB.leakage_invariance(run_fn, c)
    assert res["ok"] and res["n_keys"] > 10 and 0 < res["n_keep"] < res["n_A"]


def per_cell_leakage(run_fn, c, cells, sel_fn, seed=21):
    """칸 단위 누설 시험(1절 '선택되지 않은 A 라벨'을 칸마다 적용): 원래 문맥의 예측을 모은 뒤, 칸 (배치, n, 추출)마다 그 칸의 선택 집합 밖
    A 라벨과 B 라벨을 바꾼 문맥으로 단위 전체를 다시 돌려 그 칸의 키(placement, n, 추출이 같은 키) 예측이 같은지 본다. 다른 n·다른 배치·다른
    추출의 선택 라벨도 바뀌므로 집합 사이 누설(예: n 10 예측이 n 40 집합의 라벨을 읽음, algP 예측이 무작위 집합의 라벨을 읽음)을 잡는다.
    반환 [dict(cell, n_keys, n_sel, max_abs_diff)]."""
    with XB.h54_trace() as (p1, _):
        run_fn(c)
    out = []
    for j, (pl, n, d) in enumerate(cells):
        sel = np.asarray(sel_fn(pl, n, d), int)
        c2 = XB.perturb_labels(c, sel, seed=seed + j)
        with XB.h54_trace() as (p2, _):
            run_fn(c2)
        ks = [k for k in p1 if k[7] == pl and k[8] == n and k[9] == d]
        md = 0.0
        for k in ks:
            md = max(md, float(np.max(np.abs(np.asarray(p1[k], float) - np.asarray(p2[k], float)))) if k in p2 else np.inf)
        out.append(dict(cell=(pl, n, d), n_keys=len(ks), n_sel=len(sel), max_abs_diff=md))
    return out


def test_b3_per_cell_leakage_xc_unit_with_wplus():
    """(b) 칸 단위 판: XC 단위(W+ 가짜 공급자 포함)의 모든 유한 n 칸(무작위·algP × n 10·20 × 추출 0·1)에서 그 칸의 선택 집합 밖 라벨을 바꿔도
    그 칸의 키 예측이 같다. 음성 대조: 같은 (n, 추출)의 무작위 집합 라벨을 읽는 공급자('cross')는 합집합 판 (b)를 통과하지만 이 시험에 걸린다."""
    a = hargs(grid=(10, 20), draws=2, seeds=1)
    c = make_tctx(seed=12, nA=160)
    U0 = XC.XCUnit(a, c, "T", ALGP)

    def sel_fn(pl, n, d):
        return U0.rand_set(n, d) if pl == XC.PL_RAND else U0.algp_set(n, d)
    cells = [(pl, n, d) for pl in (XC.PL_RAND, XC.PL_P) for n in (10, 20) for d in (0, 1)]
    res = per_cell_leakage(lambda cc: XC.XCUnit(a, cc, "T", ALGP, wplus=provider()).run(), c, cells, sel_fn)
    assert all(r["n_keys"] > 0 and 0 < r["n_sel"] < len(c.yA) for r in res)
    assert all(r["max_abs_diff"] == 0.0 for r in res), [r for r in res if r["max_abs_diff"] != 0.0]
    assert sum(r["n_keys"] for r in res if r["cell"][0] == XC.PL_P and r["cell"][1] == 20) >= 2 * 3, "algP n 20 칸에 W·W+ 키가 있어야 한다"
    cross = XC.WPlusProvider(FakeStack("cross").cv, FakeStack("cross").final, "test")

    def run_cross(cc):
        XC.XCUnit(a, cc, "T", ALGP, wplus=cross).run()
    assert XB.leakage_invariance(run_cross, c)["ok"], "합집합 판은 집합 사이 누설을 잡지 못한다(이 시험을 더한 이유)"
    bad = per_cell_leakage(run_cross, c, [(XC.PL_P, 20, 0), (XC.PL_P, 20, 1)], sel_fn)
    assert all(r["max_abs_diff"] > 0 for r in bad), "칸 단위 시험이 집합 사이 누설을 잡아야 한다"


def test_b4_per_cell_leakage_xcr_unit():
    a = hargs(xcr_grid=(20, 40), draws=1, seeds=1)
    c = make_rctx(seed=3, nA=200)
    U0 = XC.XCRUnit(a, c, ALGP)

    def sel_fn(pl, n, d):
        return U0.rand_set(n, d) if pl == XC.PL_RAND else U0.algp_set(n, d)
    cells = [(pl, n, 0) for pl in (XC.PL_RAND, XC.PL_P) for n in (20, 40)]
    res = per_cell_leakage(lambda cc: XC.XCRUnit(a, cc, ALGP).run(), c, cells, sel_fn)
    assert all(r["n_keys"] > 0 and r["max_abs_diff"] == 0.0 for r in res), res


# ---------------------------------------------------------------- (c) b10
def test_c_b10_uses_first_10_labels_only():
    a = hargs(grid=(10, 20), draws=2, seeds=1)
    c = make_tctx(seed=4)
    U = XC.XCUnit(a, c, "T", ALGP).run()
    d10 = [e for e in U.diag if e["n"] == 10]
    assert len(d10) == 4 and {e["placement"] for e in d10} == {XC.PL_RAND, XC.PL_P}
    sets = {}
    for e in d10:
        sel = U.rand_set(10, e["draw"]) if e["placement"] == XC.PL_RAND else U.algp_set(10, e["draw"])
        sets[(e["placement"], e["draw"])] = sel
        np.testing.assert_allclose(e["bias"], np.mean(c.E0 * c.sA[sel] - c.yA[sel]), rtol=0, atol=1e-12)
        assert e["n_lab"] == 10
    U2 = XC.XCUnit(a, c, "T", ALGP)
    assert np.array_equal(np.sort(U2.algp_order(0)[:10]), sets[(XC.PL_P, 0)]), "algP 의 n 10 = 순서의 첫 10개"
    keep = np.unique(np.concatenate(list(sets.values())))
    c2 = XB.perturb_labels(c, keep, seed=9)
    V = XC.XCUnit(a, c2, "T", ALGP).run()
    assert [(e["placement"], e["draw"], e["bias"]) for e in V.diag if e["n"] == 10] == [(e["placement"], e["draw"], e["bias"]) for e in d10]


# ---------------------------------------------------------------- (d) A5 = TUnit 의 W
def test_d_gate_unit_equals_tunit():
    a = hargs(grid=(10, 20, -1), draws=2, seeds=2, wf4_grid=(10, 20, -1))
    c = make_tctx(seed=2, split=1)
    G = XC.XCUnit(a, c, "T", ALGP, gate=True).run()
    T = W.TUnit(a, c, "T").run()
    keys = [k for k in T.st.keys if k[0] in ("W", "R1", "R2", "D1", "P1", "P2", "P0")]
    assert len(keys) > 30
    for k in keys:
        assert k in G.st, f"관문 단위에 키 {k} 가 없다"
        s1, n1 = T.st.get(k); s2, n2 = G.st.get(k)
        assert np.array_equal(n1, n2) and np.array_equal(s1, s2), f"키 {k} 의 블록 SSE 가 다르다"
    assert set(G.st.keys) == set(T.st.keys)


def test_d2_random_arm_w_matches_select_w():
    a = hargs(grid=(10, 20), draws=2, seeds=1)
    c = make_tctx(seed=6)
    U = XC.XCUnit(a, c, "T", ALGP).run()
    T = W.TUnit(a, c, "T")
    runs = pd.DataFrame(U.rows)
    for n in (10, 20):
        for d in range(2):
            sel = U.rand_set(n, d)
            want = T.select_w(sel, n, d)[0]
            got = runs[(runs.method == "W") & (runs.placement == XC.PL_RAND) & (runs.n == n) & (runs.draw == d)].alpha_sel.iloc[0]
            assert got == want


# ---------------------------------------------------------------- (e) 부록 A(실제 자료, 세기 범주)
@REAL
def test_e_split_plan_reproduces_appendix_a():
    a = XB.h54_args(exp="xc", splits=list(XB.XC_SPLITS) + list(XB.XCR_SPLITS), threads=1)
    tot = {}
    for al, m in XC.XC_TARGETS_DEFAULT:
        keep, rows = XC.split_plan_201(a, al, "xc")                        # check=True: 부록 A 와 다르면 AssertionError
        tot[(al, m)] = len(keep)
    assert sum(tot.values()) == 68 + 2 * 56
    for t in XC.XCR_TARGETS_DEFAULT:
        keep, rows = XC.split_plan_201(a, t, "xcr")
        assert len(keep) == XB.APPENDIX_A[("xcr", t)]["n"]
    assert XB.APPENDIX_A[("xc", "Lena")]["n"] == tot[("Lena", "x")] == 9


# ---------------------------------------------------------------- (f) 판정 문구(합성 저장소)
def synth_tm(groups, name, splits=(201, 202, 203), nb=10, draws=5, seeds=(0, 1), rng_seed=0, nboot=300):
    rng = np.random.RandomState(rng_seed)
    by = {}
    for sp in splits:
        ncell = rng.randint(3, 30, nb)
        st = H4.BlockStore(name, sp, np.repeat([f"{name}s{sp}b{j:02d}" for j in range(nb)], ncell))
        cnt = st.ncell
        st.add_sse(H.P0_KEY, cnt * (40.0 + rng.gamma(2.0, 1.0, st.nb)) ** 2, cnt)
        for method, learner, placement, n, lam, level in groups:
            for d in range(1 if n in (0, -1) else draws):
                for s in ((-1,) if learner == "none" else seeds):
                    key = (method, learner, "1", placement, int(n), d, int(s), float(lam))
                    st.add_sse(key, cnt * (level + rng.gamma(2.0, 0.5, st.nb)) ** 2, cnt)
        by[sp] = st
    return H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in splits}, nboot)


def arm_groups(n, a1, a2, a3, a5, a4=None):
    p = XC.PL_P
    g = [("R1", LO, XC.PL_RAND, n, 0.25, a2), ("P1", "none", XC.PL_RAND, n, 0.0, a3), ("W", LO, XC.PL_RAND, n, 0.0, a5),
         ("R1", LO, XC.PL_RAND, n, 1.0, a2 + 1), ("D1", LO, XC.PL_RAND, n, 1.0, a2 + 2), ("P2", "none", XC.PL_RAND, n, 0.0, a3 + 1)]
    if n in XC.A1_R1_N:
        g.append(("R1", LO, p, n, 0.25, a1))
    else:
        g += [("W", LO, p, n, 0.0, a1), ("R1", LO, p, n, 0.25, a4 if a4 is not None else a1 + 1)]
    return g


def synth_world(lena_worse_vs_a3=False, nboot=300, a3_all=None):
    """PE1·Tibet·Alaska 의 합성 저장소. 레나·캐나다는 n 10·40·160, 러시아 W·E·Russia_C 는 n 10, 티베트 10·40, 알래스카 10·40·160.
    A1 은 A2 보다 4 cm 작고 A3 보다 6 cm 작다. lena_worse_vs_a3 이면 레나의 A3 가 A1 보다 6 cm 작다(지역 열세)."""
    tms, units = {}, []
    spec = {"Lena|x": (10, 40, 160), "Canada|x": (10, 40, 160), "Russia_W|x": (10,), "Russia_E|x": (10,), "Russia_C~lgd|x": (10,),
            "Tibet_LGD|x": (10, 40), "Alaska|x": (10, 40, 160)}
    for j, (nm, ns) in enumerate(spec.items()):
        gs = []
        for n in ns:
            a3 = 24.0 if (lena_worse_vs_a3 and nm == "Lena|x") else (36.0 if a3_all is None else float(a3_all))
            gs += arm_groups(n, 30.0, 34.0, a3, 30.2)
        gs += [("W", LO, XC.PL_RAND, -1, 0.0, 28.0), ("R1", LO, XC.PL_RAND, -1, 0.25, 29.0), ("P1", "none", XC.PL_RAND, -1, 0.0, 33.0 + j)]
        tms[nm] = synth_tm(gs, nm, rng_seed=j, nboot=nboot)
        t, m = nm.split("|")
        rng = np.random.RandomState(100 + j)
        for sp in (201, 202, 203):
            diag = [dict(n=10, draw=d, placement=pl, split=sp, n_lab=10, nb_lab=3, bias=float(b), abs_bias=abs(float(b)))
                    for d in range(5) for pl, b in ((XC.PL_P, rng.randn() * 5 + j), (XC.PL_RAND, rng.randn() * 6 + j))]
            units.append(dict(target=t, mode=m, split=sp, n_A=500 + 100 * j, diag=diag))
    return tms, units


def test_f_hypothesis_table_and_sentences():
    tms, units = synth_world(lena_worse_vs_a3=True)
    tms["AL-1|x"] = synth_tm(arm_groups(10, 30.0, 34.0, 36.0, 30.2), "AL-1|x", nb=4, rng_seed=40, nboot=300)   # F2, 분할마다 채점 블록 4
    out = XC.tests_xc(tms, units, ALGP, f3_names=(), nboot=300, floors={"Lena": 14.8, "Canada": 17.9, "Alaska": 11.3}, f3_status="none")
    hyp = out["hyp"].set_index("hypothesis")
    assert list(hyp.index) == [h[0] for h in XC.HYPS] and (hyp.m == 8).all() and len(out["holm"]) == 8
    assert hyp.loc["XC-1b", "branch"] == "우세" and bool(hyp.loc["XC-1b", "criterion_met"])
    assert "같은 지역을 다시 무작위로 나눈 분할에서" in hyp.loc["XC-1b", "sentence"] and "레나 행:" in hyp.loc["XC-1b", "sentence"]
    assert hyp.loc["XC-F3", "p_raw"] != hyp.loc["XC-F3", "p_raw"] and hyp.loc["XC-F3", "p_holm"] == 1.0          # NaN → Holm p 1
    assert hyp.loc["XC-F3", "sentence"] == XC.SENTENCES["XC-F3-none"] and "시험하지 못함" in hyp.loc["XC-F3", "reason"]
    assert out["_state"]["holm_final"] and hyp.holm_final.all(), "부록 XC-F3 0곳 확정이면 Holm 표가 확정이다"
    s2 = hyp.loc["XC-2b", "sentence"]
    assert "독립 지역 7곳" in s2 and "Lena|x" in s2 and "성립하지 않았다" in s2, "XC-2 문장 뒤에 독립 지역 손해 문장이 붙어야 한다"
    assert "지역 Lena|x 에서는 오차가 컸다" in s2, "풀 문장 뒤 지역 열세 문장"
    assert "(이 라벨 수에서 평가한 지역 4곳)" in s2 and "(이 라벨 수에서 평가한 지역 3곳)" in hyp.loc["XC-2c", "sentence"]
    assert "(이 라벨 수에서 평가한 지역 7곳)" in hyp.loc["XC-2a", "sentence"]
    hb = out["harm_b"]
    assert set(hb.columns) >= {"target", "worse_than_A3_05", "n_regions_evaluated", "selection_family"} and hb.worse_than_A3_05.any()
    assert dict(hb.groupby("n").n_regions_evaluated.first()) == {10: 7, 40: 4, 160: 3}
    assert set(hb[hb.target == "Alaska|x"].selection_family) == {XC.SEL_FAMILY_TXT} and set(hb[hb.target == "Lena|x"].selection_family) == {""}
    for k in ("contrasts_aux", "f2_counts", "harm_a", "diag", "labeleq", "alloc", "s12", "loo"):
        assert k in out
    assert (out["s12"].item == "S-XC12").all() and out["diag"].absb10_ge8.dtype == bool
    aux = out["contrasts_aux"]
    assert {"S-XC0", "S-XC1", "S-XC3", "S-XC5", "S-XC10"} <= set(aux["item"])
    a3 = aux[aux["item"] == "S-XC8|AUX3"]
    assert set(a3[a3.target == "Alaska|x"].selection_family) == {XC.SEL_FAMILY_TXT}, "3지역 보조 열의 알래스카 x 행은 '선택 계열'"
    assert a3[a3.scope == "MEAN"].selection_family.str.contains("선택 계열 포함").all()
    s1m = aux[(aux["item"] == "S-XC1") & (aux.scope == "MEAN") & (aux.n == 10)].iloc[0]                # 같은 라벨 집합 XC 팔 대비의 보조 2단 CI
    assert np.isfinite(s1m.ci_lo_2s) and np.isfinite(s1m.ci_hi_beq_2s) and s1m.verdict4_2s in XB.BRANCHES
    s10 = aux[(aux["item"] == "S-XC10") & (aux.scope == "MEAN")]
    assert "ci_lo_2s" not in s10.columns or s10.ci_lo_2s.isna().all(), "P0 대비에는 보조 2단 CI 를 내지 않는다"
    f2 = out["f2_counts"].set_index(["contrast", "n"])
    assert "AL-1|x:" in f2.loc[("A1-A2", 10), "detail"] and "소수 블록" in f2.loc[("A1-A2", 10), "detail"]
    assert "선택 계열" in f2.loc[("A1-A2", 10), "detail"] and f2.loc[("A1-A2", 10), "few_block_targets"] == "AL-1|x"
    s = XC.compose_hyp_sentence("XC-2b", "noninf", 40, dict(delta=0.1, ci_hi=0.3, ci_hi_beq=0.2), "우세", 0.01, indep=[("Alaska|x", 0.8)], indep_k=4)
    assert "Alaska|x(선택 계열) Δ +0.80 cm" in s


def test_f5_f3_pending_is_provisional(tmp_path):
    """부록 XC-F3 전(R2a 집계): XC-F3 은 '미정', 'F3 0곳' 문장을 쓰지 않고, Holm 의존 표는 _provisional 이름, meta 의 holm_final = false."""
    tms, units = synth_world()
    out = XC.tests_xc(tms, units, ALGP, f3_names=(), nboot=200, f3_status="pending")
    hyp = out["hyp"].set_index("hypothesis")
    assert hyp.loc["XC-F3", "branch"] == XC.F3_PENDING_TXT and hyp.loc["XC-F3", "p_holm"] == 1.0
    assert XC.SENTENCES["XC-F3-none"] not in " ".join(hyp.sentence) and XC.F3_PENDING_TXT in hyp.loc["XC-F3", "sentence"]
    assert not out["_state"]["holm_final"] and not hyp.holm_final.any() and not out["holm"].holm_final.any()
    o = XC.parse_args(["--out-dir", str(tmp_path / XC.EXP_NAME)])
    o.nboot = XB.NBOOT
    named, dev = XC.sealed_names(o, out, holm_final=False)
    assert {"xc_hyp_provisional.csv", "xc_holm_provisional.csv", "xc_sentences_provisional.json", "xc_contrasts_main.csv"} <= set(named)
    assert not any(k.startswith("xc__") for k in named) and dev["provisional_files"] and not dev["registration_deviation"]
    named2, _ = XC.sealed_names(o, out, holm_final=True)
    assert "xc_hyp.csv" in named2 and "xc_holm.csv" in named2
    o.nboot = 1000                                                         # 허용 표지 없는 로컬 집계의 상한
    named3, dev3 = XC.sealed_names(o, out, holm_final=True)
    assert "xc_hyp_nboot1000.csv" in named3 and "xc_hyp.csv" not in named3 and dev3["registration_deviation"].startswith("재표집 1000회")
    o.f3_status = "auto"
    assert XC.resolve_f3_status(o, False) == "pending" and XC.resolve_f3_status(o, True) == "final"
    for st, has in (("none", True), ("pending", True), ("final", False)):
        o.f3_status = st
        with pytest.raises(SystemExit):
            XC.resolve_f3_status(o, has)
    o.f3_status = "none"
    assert XC.resolve_f3_status(o, False) == "none"


def test_f2_five_branches_and_uncorrected():
    row = dict(delta=-1.2, ci_lo=-2.0, ci_hi=-0.4, ci_lo_beq=-2.1, ci_hi_beq=-0.3, ci_hi_c=-0.3, ci_hi_beq_c=-0.2, verdict4="우세",
               pool="부분(지역 2/5)")
    s = XC.compose_hyp_sentence("XC-1b", "superior", 40, row, "우세", 0.12)
    assert "1.20 cm(셀 가중) 작았다" in s and XB.UNCORRECTED_TXT in s, "Holm p ≥ 0.05 이면 '보정 전 유의'"
    s = XC.compose_hyp_sentence("XC-1b", "superior", 40, row, "우세", 0.01)
    assert XB.UNCORRECTED_TXT not in s
    for br, frag in (("열세", "컸다"), ("동등", "0.5 cm 안에서 같았다"), ("미결정", "확인하지 못했다(최소 검출 효과 약"), ("판정 불가", "판정할 수 없었다(행 없음)")):
        s = XC.compose_hyp_sentence("XC-1c", "superior", 160, dict(row, verdict4=br), br, 0.5, reason="행 없음" if br == "판정 불가" else "")
        assert frag in s, (br, s)
    s = XC.compose_hyp_sentence("XC-2a", "noninf", 10, dict(row, delta=0.1), "우세", 0.2, indep=[])
    assert "비열등, 라벨 10개" in s and XB.UNCORRECTED_TXT in s and XB.SMALL_EFFECT_TXT not in s and "0곳(없음)" in s
    s = XC.compose_hyp_sentence("XC-5b", "superior", 40, row, "열세", 0.01, worse=[("Lena|x", 2.0, 0.5, 3.1)])
    assert "컸다" in s and "지역 Lena|x 에서는 오차가 컸다" in s
    br, why = XC.hyp_branch("noninf", dict(row, ci_hi=0.7, ci_hi_beq=0.4, verdict4="미결정"))
    assert br == "미결정"
    br, why = XC.hyp_branch("superior", None)
    assert br == "판정 불가" and why == "행 없음"


def test_f4_sentence_marks_dependence():
    """주 2단 CI 로 우세(또는 비열등 성립)인데 2단 공통 CI 가 기준을 못 넘으면 기준 미충족이고 문장 괄호에 '분할 독립 가정 의존'이 붙는다.
    추출 조건부 판정이 다르면 '추출 변동 의존'도 붙는다."""
    row = dict(delta=-1.2, ci_lo=-2.0, ci_hi=-0.4, ci_lo_beq=-2.1, ci_hi_beq=-0.3, ci_hi_c=0.2, ci_hi_beq_c=-0.2, verdict4="우세",
               verdict4_common="미결정", ci_dependence=XB.CI_DEP_TXT)
    br, _ = XC.hyp_branch("superior", row)
    deps = XC.sentence_deps("superior", row, br)
    assert br == "우세" and deps == [XB.CI_DEP_TXT] and not XB.criterion_met(row, "superior")
    s = XC.compose_hyp_sentence("XC-1b", "superior", 40, row, br, 0.01, deps=deps)
    assert s.split(" 레나")[0].endswith("작았다(레나·캐나다 2지역)(분할 독립 가정 의존).")
    s = XC.compose_hyp_sentence("XC-1b", "superior", 40, row, br, 0.2, deps=deps)
    assert "(보정 전 유의, 분할 독립 가정 의존)." in s
    ni = dict(delta=0.1, ci_hi=0.3, ci_hi_beq=0.2, ci_hi_c=0.7, ci_hi_beq_c=0.3, verdict4="미결정", verdict4_common="미결정",
              ci_hi_x=0.6, ci_hi_beq_x=0.2, draw_dependence="")
    br, _ = XC.hyp_branch("noninf", ni)
    deps = XC.sentence_deps("noninf", ni, br)
    assert br == "우세" and deps == [XB.CI_DEP_TXT, XB.DRAW_DEP_TXT] and not XB.criterion_met(ni, "noninf")
    s = XC.compose_hyp_sentence("XC-2a", "noninf", 10, ni, br, 0.01, deps=deps, indep=[])
    assert "(비열등, 라벨 10개)(분할 독립 가정 의존, 추출 변동 의존)." in s
    assert XC.sentence_deps("superior", dict(row, ci_hi_c=-0.1, ci_dependence=""), "우세") == []
    assert XC.sentence_deps("superior", row, "미결정") == [], "판정을 말하지 않는 갈래에는 붙이지 않는다"


def test_f6_holm_ties_multiplier():
    """XC-2s 의 보정 배수: p 동률(비열등 p 0)이면 동률 묶음의 첫 순위 배수를 쓴다(Holm 보정 p 와 일치)."""
    labels = [h[0] for h in XC.HYPS]
    p = [0.004, 0.2, 0.0, 0.0, 0.3, np.nan, np.nan, np.nan]
    ht = XC.holm_with_ties(labels, p, 8).set_index("label")
    assert ht.loc["XC-2a", "multiplier_tie"] == ht.loc["XC-2b", "multiplier_tie"] == 8
    assert ht.loc["XC-2b", "multiplier"] == 7, "안정 정렬 순위의 배수(이전 구현이 XC-2s 에 쓰던 값)"
    assert ht.loc["XC-1b", "multiplier_tie"] == 6 and ht.loc["XC-F3", "p_holm"] == 1.0
    assert (ht.p_holm >= np.minimum(1.0, ht.multiplier_tie * ht.p) - 1e-12).all()


def test_f3_algp_s1_marks_xc5_untested():
    tms, units = synth_world()
    s1 = XC.algp_spec("S1", source="규칙 5")
    for nm, tm in tms.items():                                             # S1: algP 키가 없고 rand 키만 쓴다
        assert XC.arm_key("A1", 40, True) == ("W", LO, "1", XC.PL_RAND, 40, 0.0)
    out = XC.tests_xc(tms, units, s1, nboot=200, f3_status="pending")
    hyp = out["hyp"].set_index("hypothesis")
    for h in ("XC-5b", "XC-5c"):
        assert hyp.loc[h, "p_holm"] == 1.0 and hyp.loc[h, "reason"] == XC.RULE5_TXT and hyp.loc[h, "branch"] == "판정 불가"
    assert hyp.loc["XC-F3", "branch"] == "판정 불가" and "A1 = A2" in hyp.loc["XC-F3", "reason"] and hyp.loc["XC-F3", "p_holm"] == 1.0
    assert "동등" not in hyp.loc["XC-F3", "branch"] and hyp.loc["XC-F3", "sentence"] != XC.SENTENCES["XC-F3-none"]
    assert out["_state"]["holm_final"], "S1 이면 XC-F3 은 F3 상태와 관계없이 시험하지 않음이라 Holm 표가 확정이다"
    aux = out["contrasts_aux"]
    assert (aux[aux["item"] == "S-XC3"].note.astype(str) == XC.RULE5_TXT).all()
    s0 = aux[aux["item"] == "S-XC0"]
    assert len(s0) == 1 and "A1 = A2" in s0.note.iloc[0] and s0.verdict4.iloc[0] == "판정 불가"
    s8 = aux[aux["item"].str.startswith("S-XC8")]
    same = s8[(s8.contrast == "A1-A2") & (s8.n == 10)]
    assert len(same) == 2 and same.note.str.contains("시험하지 않음").all(), "S-XC8 의 같은 키 행(A1 − A2, n 10)"
    assert s8[(s8.contrast == "A1-A5") & s8.n.isin([40, 160])].note.str.contains("시험하지 않음").all()
    f2 = out["f2_counts"].set_index(["contrast", "n"])
    assert "시험하지 않음" in f2.loc[("A1-A2", 10), "note"] and "시험하지 않음" in f2.loc[("A1-A5", 40), "note"]
    s3 = XC.tests_xc(tms, units, s1, nboot=200, f3_status="none")["hyp"].set_index("hypothesis")
    assert s3.loc["XC-F3", "sentence"] == XC.SENTENCES["XC-F3-none"], "0곳 확정이면 F3 0곳 문장이 먼저다"


# ---------------------------------------------------------------- (g) Algorithm P 의 할당량과 결정성
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_g_power_alloc_rules(gamma):
    rng = np.random.RandomState(1)
    for _ in range(30):
        K = rng.randint(2, 12)
        N = rng.randint(1, 15, K)
        rank = rng.permutation(K)
        for n in (1, 3, 10, int(N.sum()), int(N.sum()) + 5):
            q = XC.power_alloc_ref(N, n, gamma, rank)
            assert q.sum() == min(n, N.sum()) and np.all(q <= N) and np.all(q >= 0)
    N = np.array([5, 9, 2, 7, 3])
    rank = np.array([3, 0, 4, 1, 2])
    q = XC.power_alloc_ref(N, 3, 0.0, rank)
    assert q.tolist() == [0, 1, 0, 1, 1], "γ 0, n < 블록 수: β 순서의 앞 n 블록에 1개씩"


@pytest.mark.parametrize("name", ["S8a", "S8b", "S8c", "S8p"])
def test_g2_algp_quota_order_and_determinism(name):
    rng = np.random.RandomState(11)
    sizes = np.array([40, 3, 25, 12, 8, 30, 5, 18, 2, 22, 15, 20])
    blk = np.repeat(np.arange(len(sizes)), sizes).astype(str)
    Z = rng.randn(len(blk), 5)
    sp = XC.algp_spec(name)
    sched = [10, 40, 160]
    seed = 4242
    pi = XC.algorithm_p_order_ref(Z, blk, sched, sp["gamma"], seed, sp["first"])
    assert np.array_equal(pi, XC.algorithm_p_order_ref(Z, blk, sched, sp["gamma"], seed, sp["first"])), "같은 seed 두 번 실행 순서 일치"
    ub, bc = np.unique(blk, return_inverse=True)
    mu = np.vstack([Z[bc == k].mean(0) for k in range(len(ub))])
    beta = XC.block_order_ref(mu, seed)
    rank = np.empty(len(ub), int); rank[beta] = np.arange(len(ub))
    prev = np.zeros(len(ub), int)
    for nk in sched:
        q = XC.power_alloc_ref(np.bincount(bc), nk, sp["gamma"], rank)
        cnt = np.bincount(bc[pi[:nk]], minlength=len(ub))
        if np.all(q >= prev):                                               # 할당이 단조이면 블록별 수 = q
            assert np.array_equal(cnt, q), (nk, cnt, q)
        prev = q
    if sp["gamma"] == 0.0:
        assert set(bc[pi[:10]].tolist()) == set(beta[:10].tolist()), "γ 0, n 10 < 블록 12: β 의 앞 10 블록"
    first = {}
    for c_ in pi:                                                          # 블록 안 첫 셀(S8a·c·p = 중심 근접)
        first.setdefault(int(bc[c_]), int(c_))
    if sp["first"] == "center":
        for b, c_ in first.items():
            cells = np.where(bc == b)[0]
            assert c_ == cells[np.argmin(np.sqrt(((Z[cells] - mu[b]) ** 2).sum(1)))]
    mind = np.full(len(Z), np.inf)                                         # 블록 안 farthest-first(전수 대조)
    seen = np.zeros(len(ub), int)
    taken = np.zeros(len(Z), bool)
    for i, c_ in enumerate(pi):
        b = bc[c_]
        if seen[b] > 0 or (sp["first"] == "ff" and i > 0):
            cand = np.where((bc == b) & ~taken)[0]
            assert np.isclose(mind[c_], mind[cand].max()), "블록 안 선택은 최소 거리 최대 셀이어야 한다"
        seen[b] += 1; taken[c_] = True
        mind = np.minimum(mind, np.sqrt(((Z - Z[c_]) ** 2).sum(1)))
    other = XC.algorithm_p_order_ref(Z, blk, sched, sp["gamma"], seed + 1, sp["first"])
    assert not np.array_equal(other, pi) or XC.block_order_ref(mu, seed + 1)[0] == beta[0]


def test_g3_s2_s4_nested_versions():
    rng = np.random.RandomState(5)
    blkA = np.repeat(np.arange(9), 7)
    for n in (5, 12, 30):                                                  # s2_order 의 앞 n 개 = h40.draw_blocks 의 절차를 같은 seed 로 n 에서 멈춘 것
        o = XC.s2_order(blkA, 40, 999)
        assert set(o[:n].tolist()) == set(XC.s2_order(blkA, n, 999).tolist())
    Z = rng.randn(63, 4)
    assert np.array_equal(XC.placement_order(XC.algp_spec("S4"), Z, blkA, [10, 40], 3, "ref"), W.kcenter_order(Z, 40, 3))


# ---------------------------------------------------------------- (h) 2단 CI = h39 l43_region
G_B = ("P1", "none", "1", "block", 10, 0.0)
G_C = ("P1", "none", "1", "cell", 10, 0.0)


def test_h_two_stage_ci_equals_l43_region():
    h39 = _load_h39()
    tm = synth_tm([("P1", "none", "block", 10, 0.0, 30.0), ("P1", "none", "cell", 10, 0.0, 31.0)], "Lena|x", splits=(1, 2, 3), nb=9, nboot=400)
    seed = XB.seed_of("lgw-l43", "Lena|x", "P1", 10)
    ref = h39.l43_region(tm, "P1", 10, 400, seed)
    got = XC.two_stage_ci(tm, G_B, G_C, 400, seed, tags=("block", "cell"))
    assert abs(got["delta"] - ref["delta"]) < 1e-12 and abs(got["delta_beq"] - ref["delta_beq"]) < 1e-12
    np.testing.assert_allclose(got["dist"], ref["dist"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["dist_beq"], ref["dist_beq"], rtol=0, atol=1e-12)
    gA, gB = ("R1", LO, "1", "cell", -1, 0.25), ("P1", "none", "1", "cell", -1, 0.0)
    tm2 = synth_tm([("R1", LO, "cell", -1, 0.25, 29.0), ("P1", "none", "cell", -1, 0.0, 30.0)], "Canada|x", nb=12, nboot=500)
    refc = X.boot_delta_common(tm2, gA, gB, nboot=500)
    gotc = XC.two_stage_ci_common(tm2, gA, gB, 500, tm2.seed, tags=("a", "b"), w_seed=XB.seed_of("lgxboot", tm2.name))
    np.testing.assert_allclose(gotc["dist"], refc["dist"], rtol=0, atol=1e-12)
    assert np.isfinite(gotc["ci_lo"]) and gotc["ci_lo"] <= gotc["ci_hi"]


# ---------------------------------------------------------------- (i) XC-2s
def test_i_fixed_sequence_superiority_only_after_noninferiority():
    row = dict(delta=-1.0, ci_hi=-0.2, ci_hi_beq=-0.1, ci_hi_c=-0.1, ci_hi_beq_c=-0.05, verdict4="우세", p_two=0.004)
    r = XC.xc2s_test(row, met_ni=False, multiplier=8)
    assert r["result"] == "시험하지 않음" and not np.isfinite(r["p_adj"])
    r = XC.xc2s_test(row, met_ni=True, multiplier=8)
    assert r["result"] == "우세" and abs(r["p_adj"] - 0.032) < 1e-12
    r = XC.xc2s_test(row, met_ni=True, multiplier=20)
    assert r["result"] == "우세 아님" and r["reason"] == "보정 p ≥ 0.05"
    tms_bad, units = synth_world(a3_all=24.0)                              # A3 가 A1 보다 6 cm 작다: 비열등 불성립
    out = XC.tests_xc(tms_bad, units, ALGP, nboot=300)
    hyp = out["hyp"].set_index("hypothesis")
    for h in ("XC-2a", "XC-2b", "XC-2c"):
        assert not bool(hyp.loc[h, "criterion_met"]) and hyp.loc[h, "xc2s"] == "시험하지 않음" and hyp.loc[h, "branch"] == "열세"
    tms_ok, units = synth_world()                                          # A3 가 6 cm 크다: 비열등 성립 → 우월 시험
    hyp = XC.tests_xc(tms_ok, units, ALGP, nboot=300)["hyp"].set_index("hypothesis")
    for h in ("XC-2b", "XC-2c"):
        assert bool(hyp.loc[h, "criterion_met"]) and hyp.loc[h, "xc2s"] in ("우세", "우세 아님")
        assert hyp.loc[h, "branch"] == "우세" and "비열등" in hyp.loc[h, "sentence"]


# ---------------------------------------------------------------- 세기 범주(라벨 값 미사용)
def test_count_scrubbed_counts_label_free():
    a = hargs(grid=(10, 40, -1), draws=2, seeds=2)
    c1 = make_tctx(seed=7)
    c2 = make_tctx(seed=7, y_shift=lambda yA, yB: (yA * 3 + 50, yB - 7))
    cnt = []
    for c in (c1, c2):
        XC.scrub_labels(c)
        assert np.isnan(c.yA).all() and np.isnan(c.yB).all() and c.E0 == 1.0
        U = XC.XCUnit(a, c, "T", ALGP, dry=True).run()
        rows, st, stats = U.finish()
        assert len(st) == 0 and not U.diag, "세기는 키와 진단값을 저장하지 않는다"
        cnt.append(dict(stats["n_fit_detail"]))
    assert cnt[0] == cnt[1] and sum(cnt[0].values()) > 0
    # 등록 구조: n 10 은 무작위 3방법 + W 교차검증, algP 는 R1 만. n 40·전량은 두 집합(전량은 하나) 모두 3방법 + W 교차검증
    det = cnt[0]
    assert det["xc|catboost_lo|R1"] == 2 * 2 * (2 + 2) + 2 * 1     # (n 10·40 × 추출 2 × (rand + algP) + 전량) × seed 2


def test_count_only_cli_dry_prints_no_label_stats(tmp_path, monkeypatch, capsys):
    """세기 CLI 경로(합성 문맥으로 바꾼 build_unit): 화면에 RMSE·Δ·판정 줄이 생기지 않아야 하고(걸러낸 줄 0), 산출 표에 그 열이 없다."""
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    o = XC.parse_args(["--count-only", "--part", "xc", "--targets", "Lena:x", "--grid", "10,40,all", "--draws-cap", "2", "--seeds", "1",
                       "--out-dir", str(tmp_path / "xc"), "--algorithm-p", "S8a"])
    o.algp = XC.resolve_algp(o, need_file=False)
    o.wplus, o.wplus_status = None, "off"

    def fake_build(oo, part, alias, mode, split, dry=False, scrub=False):
        c = make_tctx(seed=split)
        if scrub:
            XC.scrub_labels(c)
        return c, XC.XCUnit(oo.ha, c, alias, oo.algp, dry=dry), None
    monkeypatch.setattr(XC, "build_unit", fake_build)
    units = [("xc", "Lena", "x", sp, "") for sp in (201, 202)]
    with XB.restricted_output():
        df = XC.count_only(o, units, [])
    out = capsys.readouterr().out
    assert "[출력 제한]" not in out and XB.FORBIDDEN_OUT.search(out) is None
    assert "적합" in out and len(df) == 2
    det = pd.read_csv(tmp_path / "xc" / "xc_count_detail.csv")
    assert not any(XB.FORBIDDEN_OUT.search(col) for col in list(df.columns) + list(det.columns))


@REAL
def test_count_only_real_data_no_label_stats(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    rc = XC.main(["--count-only", "--part", "xc,xcr", "--targets", "Russia_W:x", "--xcr-targets", "Canada", "--threads", "1",
                  "--out-dir", str(tmp_path / "xc"), "--algorithm-p", "S8a", "--wplus", "off"])
    out = capsys.readouterr().out
    assert rc == 0 and "[출력 제한]" not in out and XB.FORBIDDEN_OUT.search(out) is None
    df = pd.read_csv(tmp_path / "xc" / "xc_count.csv")
    assert set(df.part) == {"xc", "xcr"} and (df[df.part == "xc"].split.between(201, 210)).all() and len(df[df.part == "xc"]) == 10
    assert len(df[df.part == "xcr"]) == 10 and (df[df.part == "xcr"].split.between(211, 220)).all()


# ---------------------------------------------------------------- 실행 보호, 부록 XC-0, F3, 조각 이름
def test_guard_and_xc0(tmp_path, monkeypatch):
    monkeypatch.delenv("WF_RESCALE", raising=False)
    monkeypatch.delenv("LG_RESCALE", raising=False)
    with pytest.raises(SystemExit):
        XC.main(["--part", "xc", "--targets", "Lena:x"])                    # 본 실행은 허용 표지가 없으면 자료를 읽기 전에 거부
    p = tmp_path / "xc0.json"
    p.write_text(json.dumps(dict(algorithm_p="S8c", source="XD-alg", table=[1, 2])))
    sp = XC.load_xc0(p)
    assert sp["name"] == "S8c" and sp["gamma"] == 0.5 and sp["first"] == "center" and "table" not in sp["xc0"] and len(sp["file_sha256"]) == 64
    p.write_text(json.dumps(dict(selected="S9")))
    with pytest.raises(ValueError):
        XC.load_xc0(p)
    o = XC.parse_args(["--algorithm-p", "S8a"])
    with pytest.raises(SystemExit):
        XC.resolve_algp(o, need_file=True)                                  # 본 실행·집계는 선택 파일만
    o2 = XC.parse_args(["--xc0", str(tmp_path / "없음.json")])
    with pytest.raises(SystemExit):
        XC.resolve_algp(o2, need_file=True)
    assert XC.resolve_algp(o2, need_file=False)["name"] == XC.P_DEFAULT


def test_f3_registration_and_eligibility(tmp_path):
    als = XC.register_f3(f"NewReg~xf={tmp_path}@NewReg")
    assert als == ["NewReg~xf"] and XB.resolve("NewReg~xf")[1] == "NewReg" and not XB.is_point_only("NewReg~xf")
    assert XB.prior_seeds_of("NewReg~xf") == XB.PRIOR_LG and XB.min_eval_blocks_of("NewReg~xf", "xc") == 2
    rows_ok = [dict(split=201, status="ok", n_A=30, nb_eval=5, eval_blocks=[str(i) for i in range(5)]),
               dict(split=202, status="ok", n_A=31, nb_eval=4, eval_blocks=[str(i) for i in range(3, 7)])]
    rows_ok += [dict(split=203, status="ok", n_A=29, nb_eval=3, eval_blocks=["7", "8", "1"])]
    assert XC.f3_eligible(rows_ok)                                          # 합집합 9 ≥ 8
    assert not XC.f3_eligible(rows_ok[:1])
    XB.RUN_TABLES.pop("NewReg~xf", None)
    with pytest.raises(SystemExit):
        XC.register_f3("bad")


def test_shard_select_and_tags(tmp_path):
    o = XC.parse_args(["--out-dir", str(tmp_path), "--tag", "xc"])
    assert (XC.tag_of(o, "xc"), XC.tag_of(o, "xcr"), XC.tag_of(o, "xcf3"), XC.tag_of(o, "gate")) == ("xc", "xcr", "xcf3", "xcgate")
    units = [("xc", "Lena", "x", sp, "") for sp in (201, 202, 203, 204)] + [("xcr", "Canada", "r", 211, "")]
    o.shard = "1/2"
    part = XC.select_shard(o, units)
    o.shard = "0/2"
    assert len(part) + len(XC.select_shard(o, units)) == len(units)
    o.shard = "xc__cpu__Lena__x__s203"
    assert XC.select_shard(o, units) == [("xc", "Lena", "x", 203, "")]
    os_ = XC.parse_args(["--smoke", "--out-dir", str(tmp_path)])
    assert XC.tag_of(os_, "xc") == "xc_smoke" and os_.ha.XC_DRAWS == 1 and os_.ha.SEEDS == [0] and os_.nboot <= 500


# ---------------------------------------------------------------- 조각 → 집계 → 봉인(합성 적합, 작은 왕복)
def test_j_shards_summarize_sealed_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    o = XC.parse_args(["--out-dir", str(tmp_path / XC.EXP_NAME), "--grid", "10,40,all", "--draws-cap", "2", "--seeds", "1", "--cb-iters", "10",
                       "--nboot", "200", "--wplus", "off"])
    o.algp = XC.algp_spec("S8a", source="시험")
    o.wplus, o.wplus_status = None, "off"
    for j, al in enumerate(("Lena", "Canada")):
        for sp in (201, 202):
            c = make_tctx(seed=10 * j + sp, split=sp, target=al, nA=120, nbA=12)
            U = XC.XCUnit(o.ha, c, al, o.algp).run()
            rows, st, stats = U.finish()
            cfg = XC.unit_cfg(o, "xc", "synthetic")
            XB.write_shard(o.SHARDS, XC.tag_of(o, "xc"), al, "x", sp, rows, st, cfg, unit=dict(stats, n_A=len(c.yA)), expected=[201, 202],
                           code_file=XC.__file__)
    capsys.readouterr()
    with XB.restricted_output():
        named = XC.summarize(o)
    out = capsys.readouterr().out
    assert XB.FORBIDDEN_OUT.search(out) is None and "[봉인]" in out
    sd = XB.sealed_dir(XC.EXP_NAME, XC.sealed_root(o))
    hyp_name = "xc_hyp_provisional_nboot200.csv"                           # 부록 XC-F3 전(xcf3 조각 없음) + 재표집 200(등록 이탈)
    assert sd == tmp_path / XC.EXP_NAME / "sealed" and (sd / hyp_name).exists() and (sd / "sealed_manifest.json").exists()
    assert hyp_name in named and len(named[hyp_name]) == 8 and "xc_hyp.csv" not in named
    meta = named["xc_meta_nboot200.json"]
    assert meta["holm_final"] is False and meta["f3_status"] == "pending" and meta["registration_deviation"] == "재표집 200회(등록 10000회)"
    tms, units, _ = XB.load_tms(o.SHARDS, "xc", 200)
    assert set(tms) == {"Lena|x", "Canada|x"} and all(len(u["diag"]) == 4 for u in units)


# ---------------------------------------------------------------- 조각 내용, 스모크, 본 실행 강제, 부록 XC-0, 배치 결정성
def test_k_run_unit_shard_has_no_result_columns(tmp_path, monkeypatch):
    """조각 runs.csv 에 RMSE·편향 열이 없고 unit.json 에 교차검증 RMSE 가 없으며, 교차검증 RMSE 는 봉인 폴더 unit_cv/ 에 있다(합성 적합)."""
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    o = XC.parse_args(["--out-dir", str(tmp_path / XC.EXP_NAME), "--grid", "10,20", "--draws-cap", "1", "--seeds", "1", "--cb-iters", "10",
                       "--wplus", "off"])
    o.algp = XC.algp_spec("S8a", source="시험")
    o.wplus, o.wplus_status = None, "off"

    def fake_build(oo, part, alias, mode, split, dry=False, scrub=False):
        c = make_tctx(seed=3, split=split, target=alias)
        return c, XC.XCUnit(oo.ha, c, alias, oo.algp, dry=dry, placement_impl=oo.placement_impl), None
    monkeypatch.setattr(XC, "build_unit", fake_build)
    monkeypatch.setattr(XB, "data_sha", lambda a=None, alias=None: "synthetic")
    XC.run_unit(o, "xc", "T", "x", 201)
    p = XB.shard_paths(o.SHARDS, "xc", "T", "x", 201)
    cols = set(pd.read_csv(p["runs"], nrows=0).columns)
    assert not cols & set(XC.LABEL_RESULT_COLS) and {"method", "placement", "n", "draw"} <= cols
    unit = json.loads(p["unit"].read_text())
    assert "w_cv_rmse" not in unit["notes"] and unit["cfg"]["shard_cols_dropped"] == list(XC.LABEL_RESULT_COLS) and unit["orders_head"]
    wcv = XB.sealed_dir(XC.EXP_NAME, XC.sealed_root(o)) / XC.UNIT_CV_DIR / "xc__cpu__T__x__s201_wcv.json"
    assert wcv.exists() and "sealed" in wcv.parts and set(json.loads(wcv.read_text())["w_cv_rmse"]) >= {f"{XC.PL_RAND}|20|0", f"{XC.PL_P}|20|0"}


def test_l_smoke_uses_non_eval_split_and_sealed_shards(tmp_path, monkeypatch):
    monkeypatch.delenv("WF_RESCALE", raising=False)
    monkeypatch.delenv("LG_RESCALE", raising=False)
    o = XC.parse_args(["--smoke", "--threads", "4", "--out-dir", str(tmp_path / XC.EXP_NAME)])
    assert o.SHARDS == XB.sealed_dir(XC.EXP_NAME, XC.sealed_root(o)) / XC.SMOKE_SHARDS and "sealed" in o.SHARDS.parts
    units, skipped, expected = XC.enumerate_units(o)
    used = {u[3] for u in units}
    assert used == {1} and not used & (set(XB.XC_SPLITS) | set(XB.XCR_SPLITS)), "스모크는 평가 분할(201–220)을 쓰지 않는다"
    assert {u[0] for u in units} == {"xc", "xcr"} and 1 in o.ha.SPLITS and o.threads == 2 and all(v == [1] for v in expected.values())


def test_m_xcf3_unit_has_no_wplus(monkeypatch):
    o = XC.parse_args(["--grid", "10", "--draws-cap", "1", "--seeds", "1"])
    o.algp = ALGP
    o.wplus, o.wplus_status = provider(), "test"
    monkeypatch.setattr(XC, "_plan_row", lambda a, part, al, sp: dict(status="ok"))
    monkeypatch.setattr(XB, "build_tctx", lambda a, al, m, sp, prow=None: make_tctx(seed=sp, split=sp, target=al))
    _, U3, _ = XC.build_unit(o, "xcf3", "NewReg~xf", "x", 201, dry=True)
    _, U1, _ = XC.build_unit(o, "xc", "Lena", "x", 201, dry=True)
    assert U3.wplus is None and U1.wplus is o.wplus
    assert "wplus_extra" not in XC.unit_cfg(o, "xcf3") and "wplus_extra" in XC.unit_cfg(o, "xc")


def test_n_main_run_enforces_xd_and_wplus():
    with pytest.raises(SystemExit):
        XC.enforce_main_impl(XC.parse_args(["--placement-impl", "ref"]))
    with pytest.raises(SystemExit):
        XC.enforce_main_impl(XC.parse_args(["--wplus", "off"]))
    with pytest.raises(SystemExit):
        XC.enforce_main_impl(XC.parse_args(["--summarize-only", "--placement-impl", "ref"]))
    o = XC.enforce_main_impl(XC.parse_args(["--smoke", "--placement-impl", "ref", "--wplus", "off"]))
    assert o.placement_impl == "ref" and o.wplus == "off", "스모크·세기·시험은 강제하지 않는다"
    if XC.xd_placement() is not None:
        o = XC.enforce_main_impl(XC.parse_args([]))
        assert o.placement_impl == "xd" and o.wplus == "require" and o.placement_sha != "ref"


def test_o_xc0_main_requires_sidecar_and_xd_origin(tmp_path):
    def side(p):
        Path(str(p) + ".sha256").write_text(f"{XB.sha256_file(p)}  {p.name}\n")
    p = tmp_path / "algorithm_p_selection.json"
    p.write_text(json.dumps(dict(chosen="S8c", spec=dict(gamma=0.5), reason="점수 최소", candidates={}, rule=XC.XC0_RULE,
                                 alaska_family=dict(transfer=["Alaska|x"]))))
    with pytest.raises(SystemExit):
        XC.load_xc0(p, require_sidecar=True)                               # 사이드카 없음
    assert XC.load_xc0(p)["name"] == "S8c"                                 # 스모크·세기·시험은 사이드카 없이 받는다
    side(p)
    sp = XC.load_xc0(p, require_sidecar=True)
    assert sp["name"] == "S8c" and sp["xc0_origin"] == "rule" and sp["xc0_sidecar"]
    assert XC.load_xc0(p, require_sidecar=True, expect_sha256=sp["file_sha256"][:16])["name"] == "S8c"
    for bad in ("0" * 16, sp["file_sha256"][:8]):
        with pytest.raises(SystemExit):
            XC.load_xc0(p, require_sidecar=True, expect_sha256=bad)
    o = XC.parse_args(["--xc0", str(p), "--xc0-sha256", sp["file_sha256"][:20]])
    assert XC.resolve_algp(o, need_file=True)["file_sha256"] == sp["file_sha256"]
    p2 = tmp_path / "other.json"
    p2.write_text(json.dumps(dict(chosen="S8c", rule="다른 규칙", alaska_family={})))
    side(p2)
    with pytest.raises(SystemExit):
        XC.load_xc0(p2, require_sidecar=True)                              # XD 선택 스크립트의 산출이 아니다
    p3 = tmp_path / "default.json"
    p3.write_text(json.dumps(dict(chosen="S8a", reason="XD-alg 결과가 R1a 제출 뒤 48 시간 안에 나오지 않았다: P_default", candidates={})))
    side(p3)
    assert XC.load_xc0(p3, require_sidecar=True)["xc0_origin"] == "P_default"   # 규칙 6(XD select_default)


def test_p_placement_determinism_and_shard_heads(tmp_path, monkeypatch):
    """관문의 Algorithm P 결정성 항목(라벨 미사용): 같은 seed 두 번 계산한 순서가 같고, R2a 조각의 orders_head 와 앞 10개가 같다.
    orders_head 가 다르면 통과하지 않는다."""
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    o = XC.parse_args(["--out-dir", str(tmp_path / XC.EXP_NAME), "--grid", "10,40", "--draws-cap", "2"])
    o.algp = ALGP
    monkeypatch.setattr(XC, "split_plan_201", lambda a, al, fam="xc", check=True: ([201, 202], []))
    monkeypatch.setattr(XC, "_plan_row", lambda a, part, al, sp: dict(status="ok"))
    monkeypatch.setattr(XB, "build_tctx", lambda a, al, m, sp, prow=None: make_tctx(seed=sp, split=sp, target=al))
    sh = tmp_path / "r2a_shards"
    sh.mkdir()
    for sp in (201, 202):
        U = XC.XCUnit(o.ha, make_tctx(seed=sp, split=sp, target="T"), "T", ALGP, dry=True, placement_impl=o.placement_impl).run()
        heads = U.finish()[2]["orders_head"]
        b = str(XB.shard_base(sh, "xc", "T", "x", sp))
        Path(b + "_unit.json").write_text(json.dumps(dict(orders_head=heads)))
        Path(b + "_runs.csv").write_text("method\n")
        Path(b + "_blocksse.npz").write_bytes(b"")
    res = XC.placement_determinism(o, shards_dir=sh, targets=[("T", "x")], out_name="det.csv")
    assert res["passed"] and res["n_cases"] == 4 and res["n_same"] == 4 and res["n_head"] == 4 and res["n_head_same"] == 4
    assert (tmp_path / XC.EXP_NAME / "det.csv").exists()
    b = str(XB.shard_base(sh, "xc", "T", "x", 202))
    u = json.loads(Path(b + "_unit.json").read_text())
    u["orders_head"]["1"] = list(reversed(u["orders_head"]["1"]))
    Path(b + "_unit.json").write_text(json.dumps(u))
    bad = XC.placement_determinism(o, shards_dir=sh, targets=[("T", "x")])
    assert not bad["passed"] and bad["n_head_same"] == 3
    assert XC.placement_determinism(o, shards_dir=None, targets=[("T", "x")])["n_head"] == 0


def test_q_local_resource_rules(tmp_path, monkeypatch):
    """로컬 자원 규약: 스모크는 코어 4개·스레드 2 를 어기면 거부하고, 가용 메모리 확인은 30 GB 하한에서 기다린다(바로 끝내지 않는다)."""
    monkeypatch.delenv("WF_RESCALE", raising=False)
    monkeypatch.delenv("LG_RESCALE", raising=False)
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    for v in XB.THREAD_VARS:
        monkeypatch.setenv(v, os.environ.get(v, "2"))
    monkeypatch.setattr(XC, "set_local_nice", lambda target=10: int(target))
    monkeypatch.setattr(XB, "smoke_env_report", lambda: dict(warn=["코어 묶음 8개 > 4(taskset -c 로 4개에 묶는다)"]))
    with pytest.raises(SystemExit):
        XC.main(["--smoke", "--out-dir", str(tmp_path / XC.EXP_NAME)])
    called = {}

    class Stop(Exception):
        pass

    def fake_req(min_gb=30.0, wait_s=0.0, poll_s=60.0):
        called.update(min_gb=min_gb, wait_s=wait_s, poll_s=poll_s)
        raise Stop()
    monkeypatch.setattr(XB, "require_memory", fake_req)
    with pytest.raises(Stop):
        XC.main(["--count-only", "--out-dir", str(tmp_path / XC.EXP_NAME)])
    assert called == dict(min_gb=30.0, wait_s=XC.MEM_WAIT_S, poll_s=XC.MEM_POLL_S) and XC.MEM_WAIT_S >= 600


# ---------------------------------------------------------------- XD·XB 모듈과의 연결
@pytest.mark.skipif(XC.xd_placement() is None, reason="XD 모듈(x_placement_policy)의 배치 함수가 없다")
def test_g4_reference_matches_xd_module():
    """참조 구현(2.4절 의사코드 문자 그대로)과 XD 모듈의 배치 순서·배분이 합성 후보 풀에서 같다(계획: XC 는 XD 의 함수를 쓴다)."""
    _, fn = XC.xd_placement()
    XP = sys.modules[XC.XD_MODULE]
    n_eq = n_all = 0
    for t in range(25):
        rng = np.random.RandomState(t)
        K = rng.randint(3, 30)
        sizes = rng.randint(1, 50, K)
        blk = np.repeat(np.arange(K), sizes).astype(str)
        Z = XC.zstd(rng.randn(len(blk), 25).astype(np.float32))
        sched = [n for n in (10, 40, 160) if n < len(blk)]
        for name in XC.ALGP_CANDIDATES:
            s_ = XB.seed_of(XC.P_TAG, "T", "x", 201, t % 5)
            a_ = XC.placement_order(XC.algp_spec(name), Z, blk, sched, s_, "ref")
            b_ = XC.placement_order(XC.algp_spec(name), Z, blk, sched, s_, "xd")
            n_all += 1
            n_eq += int(np.array_equal(a_, b_))
        for g in (0.0, 0.5, 1.0):
            for n in (5, 10, 40, 160):
                r = rng.permutation(K)
                assert np.array_equal(XC.power_alloc_ref(sizes, n, g, r), XP.power_alloc(sizes, n, g, r)), (t, g, n)
    assert n_eq == n_all, f"배치 순서 일치 {n_eq}/{n_all}"


@pytest.mark.skipif(importlib.util.find_spec(XC.XS_MODULE) is None, reason="XB 모듈(x_multisource_stacking)이 없다")
def test_wplus_provider_contract():
    pv, status = XC.load_wplus("auto")
    assert pv is not None and status.startswith(XC.XS_MODULE) and callable(pv.cv) and callable(pv.final) and callable(pv.prepare)
    assert XC.load_wplus("off") == (None, "off")


def test_xc0_xd_format_and_sidecar(tmp_path):
    p = tmp_path / "algorithm_p_selection.json"
    p.write_text(json.dumps(dict(chosen="S8p", spec=dict(gamma=1.0), reason="계획 2.3 고정 규칙", candidates=dict(S8a=0.1))))
    sp = XC.load_xc0(p)
    assert sp["name"] == "S8p" and sp["gamma"] == 1.0 and "candidates" not in sp["xc0"]
    (tmp_path / "algorithm_p_selection.json.sha256").write_text(f"{sp['file_sha256']}  {p.name}\n")
    assert XC.load_xc0(p)["name"] == "S8p"
    (tmp_path / "algorithm_p_selection.json.sha256").write_text("0" * 64 + f"  {p.name}\n")
    with pytest.raises(SystemExit):
        XC.load_xc0(p)
    assert XC.XC0_DEFAULT.parts[-3:] == ("XD_placement_policy", "selection", "algorithm_p_selection.json")


@HEAVY
@REAL
@pytest.mark.skipif(importlib.util.find_spec(XC.XS_MODULE) is None, reason="XB 모듈이 없다")
def test_b3_leakage_real_xb_provider_one_split():
    """누설 시험 (b)의 실제 경로 판(작은 크기): 캐나다 x 분할 201, n 40, 추출 1, seed 1, catboost 반복 10. 실제 XB 공급자(W+ 의 Stack, StackR)를
    포함한 XC 단위에서 선택 밖 A 라벨과 B 라벨을 바꿔도 모든 예측이 같다. 화면에는 통과 여부만 남는다."""
    o = XC.parse_args(["--grid", "40", "--draws-cap", "1", "--seeds", "1", "--cb-iters", "10", "--wplus", "require"])
    o.algp = XC.algp_spec("S8a", source="시험")
    o.wplus, o.wplus_status = XC.load_wplus("require")
    a = o.ha
    row = XC._plan_row(a, "xc", "Canada", 201)
    c = XB.build_tctx(a, "Canada", "x", 201, row)
    o.wplus.prepare(a, c, "Canada", "x", 201)

    def run_fn(cc):
        XC.XCUnit(a, cc, "Canada", o.algp, wplus=o.wplus).run()
    res = XB.leakage_invariance(run_fn, c)
    assert res["ok"] and res["same_keys"] and 0 < res["n_keep"] < res["n_A"]
    with XB.h54_trace() as (p1, _):
        run_fn(c)
    assert any(k[4] == "W+" for k in p1)
