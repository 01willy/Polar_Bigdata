"""scripts/3_deep_learning/xbatch_core.py 단위 시험(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 1절·4절, 부록 A).

합성 자료 시험(라벨 값을 화면에 쓰지 않는다)
(a) 동결 모듈: 네 파일의 sha256 앞 16자가 계획 머리말 값과 같고, 적재된 모듈이 동결 경로의 파일이다. 한 바이트를 바꾼 사본은 거부된다.
(b) 제약 파일의 고정 판이 1절 '패키지 판'과 같다(catboost 1.2.10, scikit-learn 1.9.1, pandas 2.3.3, scipy 1.17.1, numpy 2.1.1).
(c) 실행 보호: 허용 표지(WF_RESCALE=1 또는 --allow-local)가 없으면 본 실행을 거부하고 스레드 ≤ 4, 집계 재표집 ≤ 1,000 이다.
(d) 출력 제한: RMSE·Δ·판정 줄을 거르고 나머지 줄은 그대로 둔다.
(e) 인자: h54 인자 계약(a.SPLITS 에 새 seed, a.G[exp] 격자, a.SEEDS 0·1)을 만족한다.
(f) 추출: draw_cells = h40.draw_cells, 중첩 순열 추출의 앞 n 개 = n 의 집합(중첩), cells_grid 규칙.
(g) 분할 제외 규칙(합성 자료): 앞선 분할과 같거나 여집합, 새 범위 안 중복(여집합 포함), 채점 블록 2·5 미만을 빼고 빈 자리를 채우지 않는다.
(h) 2단 재표집: tags = ('block', 'cell') 에서 h39 l43_region 과 같은 분포, 추출이 하나면 boot_delta_blocks(h40.contrast)와 같고 '2단 공통'은
    h42.boot_delta_common 과 같다. 기본 팔 표지에서 대비 사이 가법성(Δ(A−C) = Δ(A−B) + Δ(B−C))과 결정성.
(i) 판정: 4분 판정(δ 0.5, 1.0, δ_rel)과 '한계 의존', 양측 p, 비열등 p 와 Holm 입력(× 2).
(j) Holm: 행 없음은 p 1 로 가족 크기를 유지하고, 보정 배수 m − 순위 + 1 을 낸다.
(k) 다수 n 종합 _rule3 과 해석 문장의 다섯 갈래(보정 전 유의, 효과 크기, 최소 검출 효과, 지역 열세 덧붙임).
(l) 풀: h42.pool_rows 감싸기의 보조 열(p, δ_rel, 비열등, 소수 블록, 부분 풀 표지, 지역 열세, HK 지역 수준 구간)과 XC 기준(criterion_met).
(m) 조각: 등록 이름 <tag>__cpu__<대상>__<모드>__s<분할>[__<변형>]_{runs.csv, blocksse.npz, unit.json}, --resume 상태, 공통 해시,
    조각 → TMx 왕복, 허용 경로 밖 쓰기 거부. 재현 관문 비교(같은 노드 0, 로컬·Rescale 1e-4 cm² 또는 상대 1e-9, 키 이름 대응).
(n) 봉인: 화면에는 행 수와 해시만, 목록 파일, 봉인 폴더 읽기 거부, 허용 경로 밖 거부.
(o) 누설 시험 틀: h54 전이 단위(규칙 W 포함)에서 선택 밖 A 라벨과 B 라벨을 바꿔도 모든 예측이 같고, 선택 라벨 하나를 바꾸면 달라진다(음성 대조).
(p) XI warm_trim: A 에서만 셀을 빼고 W·I 채점 셀은 같으며 frac_W_outside_A = 1, R10TrimUnit 의 저장소 이름과 dry 실행.
실제 자료 시험(세기 범주: 셀 수·블록 수만 쓰고 라벨 값은 쓰지 않는다)
(q) 부록 A: XC(201–210) 18행과 XC-r(211–220) 3행의 유효 분할 수, 뺀 seed 와 사유, |A| 범위, 채점 블록 범위, 합집합이 같다.
    분할 1–5 의 최소 채점 블록이 부록 A 머리 문장의 기록과 같다.
(r) 4절 약관 점검: 네 실행 표(Tibet, Russia_C, NAtlantic_lic, Canada_expanded_lic)에 약관 미확인 셀이 없다.
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c <코어 4개> python3 -m pytest -q tests/test_xbatch_core.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "3_deep_learning"))
_spec = importlib.util.spec_from_file_location("xbatch_core", ROOT / "scripts" / "3_deep_learning" / "xbatch_core.py")
XB = importlib.util.module_from_spec(_spec)
sys.modules["xbatch_core"] = XB
_spec.loader.exec_module(XB)
H, X, W, H4 = XB.H, XB.X, XB.W, XB.H4

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


# ---------------------------------------------------------------- 합성 저장소
def synth_tm(groups, name="T|x", splits=(1, 2, 3), nb=10, draws=5, seeds=(0, 1), rng_seed=0, nboot=300, with_p0=True):
    """합성 TMx. groups = [(method, learner, placement, n, lam, 수준)]. 물리식(learner none)은 seed −1 하나, n = 0·−1 은 추출 하나."""
    rng = np.random.RandomState(rng_seed)
    by = {}
    for sp in splits:
        ncell = rng.randint(3, 30, nb)
        st = H4.BlockStore(name, sp, np.repeat([f"s{sp}b{j:02d}" for j in range(nb)], ncell))
        cnt = st.ncell
        if with_p0:
            st.add_sse(H.P0_KEY, cnt * (30.0 + rng.gamma(2.0, 2.0, st.nb)) ** 2, cnt)
        for method, learner, placement, n, lam, level in groups:
            nd = 1 if n in (0, -1) else draws
            for d in range(nd):
                for s in ((-1,) if learner == "none" else seeds):
                    key = (method, learner, "1", placement, int(n), d, int(s), float(lam))
                    st.add_sse(key, cnt * (level + rng.gamma(2.0, 2.0, st.nb)) ** 2, cnt)
        by[sp] = st
    return H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in splits}, nboot)


def synth_stat(delta, sd=0.3, nboot=2000, seed=0, nbmin=8, p0=30.0):
    rng = np.random.RandomState(seed)
    d = delta + sd * rng.randn(nboot); db = delta + sd * rng.randn(nboot)
    return dict(delta=float(delta), delta_beq=float(delta), rmse_A=p0 + delta, rmse_B=p0, n_splits=5, n_splits_expected=5, n_blocks_min=nbmin,
                ok=True, dist=d, dist_beq=db, cdist=d + 0.01 * rng.randn(nboot), cdist_beq=db + 0.01 * rng.randn(nboot), p0_cell=p0, p0_beq=p0,
                ci_kind="two_stage", xdist=d, xdist_beq=db)


def make_tctx(seed=0, split=1, target="T", mode="x"):
    """합성 전이 문맥(h40.Ctx, tests/test_h54_workflow.py 와 같은 구성). 원천 300행(두 지역), A 80셀 8블록, B 60셀 6블록."""
    rng = np.random.RandomState(seed)
    D = len(XB.FEATS)
    ns, nA, nB = 300, 80, 60
    Xs = rng.randn(ns, D); ss = rng.uniform(20, 40, ns); ys = 1.6 * ss + 3 * Xs[:, 1] + 2 * rng.randn(ns)
    XA = rng.randn(nA, D); sA = rng.uniform(20, 40, nA); yA = 1.3 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    XBm = rng.randn(nB, D); sB = rng.uniform(20, 40, nB); yB = 1.3 * sB + 3 * XBm[:, 1] + 2 * rng.randn(nB)
    return H.Ctx(target, mode, split, target, Xs, ys, ss, np.repeat(["r1", "r2"], ns // 2), XA, yA, sA, np.repeat(np.arange(8), nA // 8),
                 XBm, yB, sB, 500 + np.repeat(np.arange(6), nB // 6), meta=dict(dup_of=-1, valid=True))


def make_rctx10(seed=0, nA=120, nW=30, nI=30):
    """합성 지역 내 WF10 문맥(h54.RCtx + maskW). W 셀의 s 가 가장 크다."""
    rng = np.random.RandomState(seed)
    D = len(XB.FEATS)
    XA = rng.randn(nA, D).astype(np.float32); sA = rng.uniform(20, 46, nA); yA = 1.4 * sA + 2 * rng.randn(nA)
    sW, sI = rng.uniform(44, 50, nW), rng.uniform(20, 40, nI)
    sB = np.r_[sW, sI]; XBm = rng.randn(nW + nI, D).astype(np.float32); yB = 1.4 * sB + 2 * rng.randn(nW + nI)
    blkA = np.repeat(np.arange(12), nA // 12); blkB = 1000 + np.repeat(np.arange(6), (nW + nI) // 6)
    latA, lonA = 65 + 0.1 * rng.rand(nA), -150 + 0.1 * rng.rand(nA)
    latB, lonB = 66 + 0.1 * rng.rand(nW + nI), -150 + 0.1 * rng.rand(nW + nI)
    c = W.RCtx("T", 1, "T", XA, yA, sA, blkA, latA, lonA, XBm, yB, sB, blkB, latB, lonB, 1.6,
               meta=dict(dup_of=-1, valid=True, n_valid_splits=1, n_unique_splits=1))
    c.maskW = np.r_[np.ones(nW, bool), np.zeros(nI, bool)]
    return c


# ---------------------------------------------------------------- (a) 동결 모듈
def test_a_frozen_hashes_and_paths(tmp_path):
    want = {"h40": "7098f59dabe2b73f", "h42": "22e215e435e157be", "h54": "cf9a1f6d0929c892", "h41": "492373e4d37b5ea4"}
    assert XB.FROZEN_SHA16 == want
    for k in want:
        assert XB.assert_frozen(k) == want[k]
    for k, mod in (("h40", H), ("h42", X), ("h54", W)):
        name, rel, _ = XB.FROZEN[k]
        assert sys.modules[name] is mod and Path(mod.__file__).resolve() == (ROOT / rel).resolve()
    assert X.H is H and W.H is H and W.X is X, "h42·h54 가 다른 h40·h42 를 읽었다"
    bad = tmp_path / "h40_copy.py"
    shutil.copy(ROOT / XB.FROZEN["h40"][1], bad)
    with open(bad, "ab") as f:
        f.write("\n# 변경\n".encode())
    with pytest.raises(XB.FrozenModuleError):
        XB.assert_frozen("h40", path=bad)


# ---------------------------------------------------------------- (b) 제약 파일
def test_b_constraints_file():
    pins = XB.read_constraints()
    assert pins == {"catboost": "1.2.10", "scikit-learn": "1.9.1", "pandas": "2.3.3", "scipy": "1.17.1", "numpy": "2.1.1"} == XB.PINNED
    mm = XB.check_constraints()
    assert all(len(t) == 3 for t in mm)                                   # 로컬에서는 차이가 있을 수 있다(형식만 본다)


# ---------------------------------------------------------------- (c) 실행 보호
def test_c_run_guard(monkeypatch):
    monkeypatch.delenv("WF_RESCALE", raising=False); monkeypatch.delenv("LG_RESCALE", raising=False)
    assert not XB.run_permitted([])
    assert XB.run_permitted(["--allow-local"])
    for m in ("smoke", "count", "summarize", "test"):
        assert XB.guard(m, [])
    with pytest.raises(SystemExit):
        XB.guard("run", [])
    assert XB.guard("run", ["--allow-local"])
    assert XB.local_limits(8, 10000, []) == (4, 1000)
    assert XB._peek_threads(["--threads", "16"]) == "4"
    monkeypatch.setenv("WF_RESCALE", "1")
    assert XB.run_permitted([]) and XB.guard("run", [])
    assert XB.local_limits(8, 10000, []) == (8, 10000)
    assert XB._peek_threads(["--threads", "16"]) == "16"


# ---------------------------------------------------------------- (d) 출력 제한
def test_d_output_filter(capsys):
    with XB.restricted_output(report=True):
        print("세기: 적합 120건")
        print("R1 rmse 12.3")
        print("Δ −0.4 [−1.0, 0.2]")
        print("판정 우세")
        XB.safe_print("delta", 0.3)
        XB.safe_print("단위", 5)
    out = capsys.readouterr().out
    assert "적합 120건" in out and "단위 5" in out
    assert XB.FORBIDDEN_OUT.search(out.replace("[출력 제한]", "")) is None
    assert "걸러낸 줄 3개" in out and "한 줄을 걸렀다" in out


# ---------------------------------------------------------------- (e) 인자
def test_e_h54_args_contract():
    a = XB.h54_args(exp="xc", splits=XB.XC_SPLITS, grid=[10, 40, 160, -1], threads=16, cb_iters=200)
    assert a.SPLITS == list(range(201, 211)) and a.G["xc"] == [10, 40, 160, -1] and a.SEEDS == [0, 1]
    assert a.cb_iters == 200 and a._ha is None and Path(a.PROC).name == "processed" and Path(a.LGD).name == "run_tables"
    if not XB.run_permitted():
        assert a.threads <= 4
    assert XB.KAPPA == 10.0 and XB.LAMS == (0.25, 0.5, 1.0) and XB.SEEDS == (0, 1) and XB.NBOOT == 10000
    assert XB.PE1 == ["Lena|x", "Canada|x", "Russia_W|x", "Russia_E|x", "Russia_C~lgd|x"] and XB.PE2 == XB.PE1 + ["Tibet_LGD|x"]
    assert XB.is_point_only("NAtlantic~lic") and not XB.is_point_only("Russia_C~lgd") and XB.is_point_only("Russia_C")
    assert XB.resolve("Canada~exp~lic") == ("Canada_expanded_lic", "Canada", False)


# ---------------------------------------------------------------- (f) 추출
def test_f_draws():
    for n in (0, 3, 10, 40, -1):
        assert np.array_equal(XB.draw_cells("Lena", "x", 201, n, 2, 500), H.draw_cells("Lena", "x", 201, n, 2, 500))
    big = XB.draw_nested("xc-rand", "Lena", "x", 203, 160, 1, 900)
    for n in (10, 40):
        small = XB.draw_nested("xc-rand", "Lena", "x", 203, n, 1, 900)
        assert np.array_equal(small, big[:n]) and len(np.unique(small)) == n
    assert np.array_equal(big, XB.draw_nested("xc-rand", "Lena", "x", 203, 160, 1, 900))
    assert not np.array_equal(big, XB.draw_nested("xc-rand", "Lena", "x", 203, 160, 2, 900))
    assert len(XB.draw_nested("xc-rand", "T", "x", 1, 0, 0, 30)) == 0 and np.array_equal(XB.draw_nested("xc-rand", "T", "x", 1, -1, 0, 30), np.arange(30))
    assert XB.cells_grid([0, 10, 40, 160, -1], 5, 100) == [(10, d) for d in range(5)] + [(40, d) for d in range(5)] + [(-1, 0)]
    assert XB.cells_grid([0, 10], 2, 15, zero_n=True) == [(0, 0), (10, 0), (10, 1)]


# ---------------------------------------------------------------- (g) 분할 제외 규칙(합성)
class _FakeD:
    def __init__(self, df):
        self.df = df

    def target_idx(self, t):
        return np.where(self.df.reg.values == t)[0]


def _fake_data(nblocks=6, seed=0, few_eval=False):
    rng = np.random.RandomState(seed)
    rows = []
    for b in range(nblocks):
        for _ in range(int(rng.randint(5, 40))):
            ok = not (few_eval and b % 2 == 1)
            rows.append(dict(reg="T", block=1000 + b, alt_cm=50.0, cci_alt=40.0 if ok else np.nan, e5_sqrt_tdd_soil=30.0))
    return _FakeD(pd.DataFrame(rows))


@pytest.mark.parametrize("nblocks,few_eval,min_nb", [(4, False, 2), (6, False, 2), (9, False, 5), (8, True, 2)])
def test_g_split_rule_synthetic(nblocks, few_eval, min_nb):
    D = _fake_data(nblocks, few_eval=few_eval)
    t_idx = D.target_idx("T"); blk = D.df.block.values; allb = frozenset(blk[t_idx].tolist())
    prior, new = list(range(1, 6)), list(range(201, 216))
    keep, rows = XB.split_plan_new(D, "T", new, prior, min_nb)
    assert [r["split"] for r in rows] == new and keep == [r["split"] for r in rows if r["status"] == "ok"]
    pa = {}
    for q in prior:
        a_ = frozenset(blk[XB.half_split_blocks(D.df, t_idx, q)[0]].tolist())
        pa.setdefault(a_, q); pa.setdefault(allb - a_, q)
    seen = []
    for r in rows:
        a_ = frozenset(blk[XB.half_split_blocks(D.df, t_idx, r["split"])[0]].tolist())
        if a_ in pa:
            assert r["status"].startswith("prior")
        elif any(a_ == q or a_ == allb - q for q in seen):
            assert r["status"] in ("dup_new", "mirror_new")
        elif r["nb_eval"] < 2:
            assert r["status"] == "nb_eval<2"
        elif r["nb_eval"] < min_nb:
            assert r["status"] == f"nb_eval<{min_nb}"
        else:
            assert r["status"] == "ok" and r["valid"] and r["dup_of"] == -1
        seen.append(a_)
        assert r["n_valid_splits"] == len(keep)
    keep2, _ = XB.split_plan_new(D, "T", new, prior, min_nb)
    assert keep2 == keep, "결정적이어야 한다"
    s = XB.plan_summary(rows)
    assert s["n_valid"] + len(s["excluded"]) == len(new)
    if nblocks == 4:
        assert len(keep) < len(new), "블록 4개에서는 중복 분할이 생겨야 한다(빈 자리를 채우지 않는다)"


def test_g2_prior_and_min_blocks_assignment():
    assert XB.prior_seeds_of("Lena") == tuple(range(1, 201)) and XB.prior_seeds_of("AL-3") == tuple(range(1, 201))
    assert XB.prior_seeds_of("Tibet_LGD") == (1, 2, 3, 4, 5) and XB.prior_seeds_of("NAtlantic~lic|x") == (1, 2, 3, 4, 5)
    assert XB.min_eval_blocks_of("Russia_C~lgd", "xc") == 5 and XB.min_eval_blocks_of("Alaska", "xcr") == 5
    assert XB.min_eval_blocks_of("AL-1", "xc") == 2 and XB.min_eval_blocks_of("NAtlantic~lic", "xc") == 2
    assert XB.min_eval_blocks_of("Lena", "transfer") == 2


# ---------------------------------------------------------------- (h) 2단 재표집
G_B = ("P1", "none", "1", "block", 10, 0.0)
G_C = ("P1", "none", "1", "cell", 10, 0.0)


def test_h1_two_stage_equals_l43_region():
    h39 = _load_h39()
    tm = synth_tm([("P1", "none", "block", 10, 0.0, 30.0), ("P1", "none", "cell", 10, 0.0, 31.0)], name="Lena|x", nb=9)
    seed = XB.seed_of("lgw-l43", "Lena|x", "P1", 10)
    ref = h39.l43_region(tm, "P1", 10, 400, seed)
    got = XB.two_stage(tm, G_B, G_C, 400, seed, tags=("block", "cell"))
    assert abs(got["delta"] - ref["delta"]) < 1e-12 and abs(got["delta_beq"] - ref["delta_beq"]) < 1e-12
    np.testing.assert_allclose(got["dist"], ref["dist"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["dist_beq"], ref["dist_beq"], rtol=0, atol=1e-12)
    assert got["n_splits"] == ref["n_splits"] == 3


def test_h2_single_draw_reduces_to_same_set_ci():
    gA, gB = ("R1", "catboost_lo", "1", "cell", -1, 0.25), ("P1", "none", "1", "cell", -1, 0.0)
    tm = synth_tm([("R1", "catboost_lo", "cell", -1, 0.25, 29.0), ("P1", "none", "cell", -1, 0.0, 30.0)], nb=12, nboot=500)
    ref = H.contrast(tm, gA, gB, return_dist=True)                      # h4_common.boot_delta_blocks, seed = tm.seed
    got = XB.two_stage(tm, gA, gB, tm.nboot, tm.seed, tags=("a", "b"))
    np.testing.assert_allclose(got["dist"], ref["dist"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(got["dist_beq"], ref["dist_beq"], rtol=0, atol=1e-12)
    assert abs(got["delta"] - ref["delta"]) < 1e-12
    refc = X.boot_delta_common(tm, gA, gB, nboot=500)                    # 2단 공통 = 공통 재표집(추출이 하나일 때)
    gotc = XB.two_stage(tm, gA, gB, 500, tm.seed, tags=("a", "b"), common=True, w_seed=XB.seed_of("lgxboot", tm.name))
    np.testing.assert_allclose(gotc["dist"], refc["dist"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(gotc["dist_beq"], refc["dist_beq"], rtol=0, atol=1e-12)


def test_h3_additivity_determinism_and_region_stat():
    gs = [("W", "catboost_lo", "algP", 40, 0.0, 28.0), ("R1", "catboost_lo", "rand", 40, 0.25, 29.0), ("P1", "none", "rand", 40, 0.0, 30.5)]
    tm = synth_tm(gs, name="Canada|x", nb=11, nboot=600)
    A, B, C = [(m, lr, "1", pl, n, lam) for m, lr, pl, n, lam, _ in gs]
    s = XB.seed_of("xbatch-2s", tm.name)
    ab, bc, ac = XB.two_stage(tm, A, B, 600, s), XB.two_stage(tm, B, C, 600, s), XB.two_stage(tm, A, C, 600, s)
    np.testing.assert_allclose(ac["dist"], ab["dist"] + bc["dist"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(ac["dist_beq"], ab["dist_beq"] + bc["dist_beq"], rtol=0, atol=1e-9)
    again = XB.two_stage(tm, A, B, 600, s)
    assert np.array_equal(again["dist"], ab["dist"])
    r = XB.region_stat_two_stage(tm, A, B)
    assert r["ok"] and r["cdist"] is not None and r["xdist"] is not None and r["ci_kind"] == "two_stage"
    assert np.array_equal(r["dist"], ab["dist"]) and not np.array_equal(r["cdist"], r["dist"])
    assert np.isfinite(r["p0_cell"]) and np.isfinite(r["p0_beq"])
    small = synth_tm(gs, name="Few|x", splits=(1,), nb=6, nboot=200)     # 채점 블록 합집합 6 < 8 → CI 없음
    r2 = XB.region_stat_two_stage(small, A, B)
    assert not r2["ok"] and r2["dist"] is None and np.isfinite(r2["delta"])
    rs = XB.region_stat_same(tm, B, C)
    assert rs["ci_kind"] == "same" and rs["dist"] is not None
    with pytest.raises(ValueError):
        XB.two_stage(tm, A, B, 10, s, tags=("x", "x"))


# ---------------------------------------------------------------- (i) 판정과 p
def test_i_verdicts_and_p():
    v = XB.verdict_cols(-2.0, -1.0, -2.2, -0.9, 30.0, 25.0)
    assert v["verdict4"] == v["verdict4_d10"] == v["verdict4_rel"] == "우세" and v["limit_dependence"] == ""
    v = XB.verdict_cols(-0.6, 0.4, -0.4, 0.4, 30.0, 30.0)                  # δ 0.5 미결정, δ 1.0 동등, δ_rel 0.6 동등
    assert (v["verdict4"], v["verdict4_d10"], v["verdict4_rel"]) == ("미결정", "동등", "동등") and v["limit_dependence"] == "한계 의존"
    assert abs(v["delta_rel"] - 0.6) < 1e-12
    assert XB.verdict_cols(np.nan, 1, 0, 1)["verdict4"] == "판정 불가"
    d = -1.0 + 0.1 * np.random.RandomState(0).randn(1000)
    assert XB.p_two(d, d) == pytest.approx(1.0 / 1000)
    assert XB.p_noninf(np.r_[np.zeros(90), np.ones(10)], np.r_[np.zeros(80), np.ones(20)], 0.5) == pytest.approx(0.2)
    assert XB.p_noninf_for_holm(0.3) == pytest.approx(0.6) and XB.p_noninf_for_holm(0.8) == 1.0
    assert XB.noninf(0.4, 0.49) and not XB.noninf(0.4, 0.5) and not XB.noninf(np.nan, 0.1)


# ---------------------------------------------------------------- (j) Holm
def test_j_holm():
    adj = XB.holm([0.01, 0.04, None, np.nan], m=8)
    ref = XB.MS.holm(np.r_[0.01, 0.04, 1.0, 1.0, np.ones(4)])[:4]
    np.testing.assert_allclose(adj, ref)
    assert adj[0] == pytest.approx(0.08) and adj[2] == 1.0
    t = XB.holm_table(["a", "b", "c"], [0.03, 0.01, 0.2], m=8)
    assert t["rank"].tolist() == [2, 1, 3] and t["multiplier"].tolist() == [7, 8, 6]
    assert t["p_holm"].tolist() == pytest.approx([0.21, 0.08, 1.0])
    with pytest.raises(ValueError):
        XB.holm([0.1, 0.2, 0.3], m=2)


# ---------------------------------------------------------------- (k) 종합과 해석 문장
def test_k_rule3_and_sentences():
    assert XB.rule3([("n40", "우세"), ("n160", "우세")]).startswith("지지")
    assert XB.rule3([("n40", "우세"), ("n160", "미결정")]).startswith("부분 지지")
    assert XB.rule3([("n40", "우세"), ("n160", "열세")]).startswith("기각")
    assert XB.rule3([("n40", "미결정"), ("n160", "판정 불가")]).startswith("판정 불가")
    assert XB.rule3([("n40", "미결정"), ("n160", "동등")]).startswith("기각")
    assert XB.rule3([("n40", None)]).startswith("판정 불가")
    tpl = {"우세": "오차가 a cm 작았다.", "열세": "오차가 a cm 컸다.", "동등": "0.5 cm 안에서 같았다.", "미결정": "차이를 확인하지 못했다.",
           "판정 불가": "판정할 수 없었다."}
    s = XB.compose_sentence(tpl, "우세", holm_p=0.2, delta=-0.3, worse_regions=[("Lena|x", 1.5, 0.2, 2.9)])
    assert "보정 전 유의" in s and XB.SMALL_EFFECT_TXT in s and "지역 Lena|x 에서는 오차가 컸다" in s and "\u2014" not in s
    assert "최소 검출 효과 약 1.2 cm" in XB.compose_sentence(tpl, "미결정", mde=1.2)
    assert XB.compose_sentence(tpl, "행 없음", reason="F3 0곳").startswith("판정할 수 없었다(F3 0곳)")
    assert "보정 전" not in XB.compose_sentence(tpl, "열세", holm_p=0.01, delta=2.0)


# ---------------------------------------------------------------- (l) 풀
def test_l_pool_wrapper():
    per = {"A|x": synth_stat(-1.5, seed=1), "B|x": synth_stat(-1.0, seed=2, nbmin=3), "C|x": synth_stat(1.2, sd=0.2, seed=3)}
    rows = XB.pool(per, ["A|x", "B|x", "C|x", "D|x"], label="XC-1b|n40", registered=4)
    reg = {r["target"]: r for r in rows if r["scope"] == "region"}
    mr = rows[-1]
    assert set(reg) == {"A|x", "B|x", "C|x"} and mr["scope"] == "MEAN"
    assert reg["A|x"]["verdict4"] == "우세" and reg["C|x"]["verdict4"] == "열세" and reg["C|x"]["worse"]
    assert reg["B|x"]["few_blocks"] == XB.FEW_BLOCKS_TXT and reg["A|x"]["few_blocks"] == ""
    assert mr["pool"] == "부분(지역 3/4)" and mr["n_regions_registered"] == 4 and mr["few_block_regions"] == "B|x"
    assert mr["worse_regions"] == "C|x" and XB.worse_regions(rows) == [("C|x", 1.2, reg["C|x"]["ci_lo"], reg["C|x"]["ci_hi"])]
    for r in rows:
        for k in ("p_two", "p_ni", "p_ni_x2", "verdict4_rel", "limit_dependence", "noninf", "noninf_d10", "ci_dependence", "draw_dependence"):
            assert k in r
    assert mr["ri_k"] == 3 and "region_general" in mr and np.isfinite(mr["ri_hk_lo"])
    np.testing.assert_allclose(mr["_dist"], XB.MS.strat([per[k]["dist"] for k in ("A|x", "B|x", "C|x")]))
    assert abs(mr["delta"] - np.mean([-1.5, -1.0, 1.2])) < 1e-12
    df = XB.clean_rows(rows)
    assert not any(str(c).startswith("_") for c in df.columns)
    good = {k: synth_stat(-1.0, sd=0.2, seed=i) for i, k in enumerate(("A|x", "B|x"))}
    mr2 = XB.pool(good, ["A|x", "B|x"], "t")[-1]
    assert mr2["pool"] == "지역 2/2" and mr2["verdict4"] == "우세" and XB.criterion_met(mr2, "superior") and XB.criterion_met(mr2, "noninf")
    one = XB.pool({"A|x": synth_stat(-1.0)}, ["A|x", "B|x"], "t")[-1]
    assert one["pool"] == "판정 불가" and one["verdict4"] == "판정 불가" and not XB.criterion_met(one)


# ---------------------------------------------------------------- (m) 조각
def _store(name, split, rng, keys):
    st = H4.BlockStore(name, split, np.repeat([f"b{j}" for j in range(9)], 4))
    for k in keys:
        st.add_sse(k, st.ncell * (20 + rng.rand(st.nb)) ** 2, st.ncell)
    return st


def test_m_shards_roundtrip(tmp_path):
    sh = tmp_path / "shards"
    rng = np.random.RandomState(0)
    keys = [H.P0_KEY] + [("P1", "none", "1", "rand", 10, d, -1, 0.0) for d in range(3)]
    cfg = XB.make_unit_cfg("xc", variant="", data_sha="abc", grid=[10])
    for sp in (201, 203):
        u = XB.write_shard(sh, "xc", "Lena", "x", sp, [dict(method="P1", n=10)], _store("Lena|x", sp, rng, keys), cfg,
                           unit=dict(n_A=100), expected=[201, 203], code_file=__file__, allowed=[tmp_path])
        assert u["cfg_hash"] == XB.cfg_hash(cfg) and u["store_names"] == ["Lena|x"] and u["frozen"] == XB.FROZEN_SHA16
    p = XB.shard_paths(sh, "xc", "Lena", "x", 201)
    assert p["runs"].name == "xc__cpu__Lena__x__s201_runs.csv" and p["npz"].name == "xc__cpu__Lena__x__s201_blocksse.npz"
    assert p["unit"].name == "xc__cpu__Lena__x__s201_unit.json"
    assert XB.shard_paths(sh, "xc", "Lena", "x", 201, "S8a")["unit"].name == "xc__cpu__Lena__x__s201__S8a_unit.json"
    assert XB.unit_state(sh, "xc", "Lena", "x", 201, cfg) == (True, "ok")
    assert XB.unit_state(sh, "xc", "Lena", "x", 201, dict(cfg, grid=[40]))[0] is False
    assert XB.unit_state(sh, "xc", "Lena", "x", 202, cfg) == (False, "조각 없음")
    uj = json.loads(p["unit"].read_text()); uj["status"] = "failed"; p["unit"].write_text(json.dumps(uj))
    assert XB.unit_state(sh, "xc", "Lena", "x", 201, cfg) == (False, "이전 실행 실패")
    uj["status"] = "ok"; p["unit"].write_text(json.dumps(uj))
    assert XB.cfg_hash(cfg, common=True) == XB.cfg_hash(dict(cfg, variant="S8a", data_sha="zz"), common=True)
    assert XB.cfg_hash(cfg) != XB.cfg_hash(dict(cfg, variant="S8a"))
    found = XB.find_shards(sh, "xc")
    assert [(f["target"], f["mode"], f["split"], f["variant"]) for f in found] == [("Lena", "x", 201, ""), ("Lena", "x", 203, "")]
    tms, units, runs = XB.load_tms(sh, "xc", nboot=100)
    tm = tms["Lena|x"]
    assert sorted(tm.used) == [201, 203] and tm.expected_splits == [201, 203] and tm.has_ci and len(runs) == 2
    assert np.isfinite(XB.grp_rmse2(tm, ("P1", "none", "1", "rand", 10, 0.0))[0])
    with pytest.raises(SystemExit):
        XB.write_shard(tmp_path / "x", "xc", "Lena", "x", 1, [], _store("Lena|x", 1, rng, keys), cfg)     # 기본 허용 뿌리 밖
    with pytest.raises(ValueError):
        XB.shard_base(sh, "x__c", "Lena", "x", 1)
    with pytest.raises(SystemExit):
        XB.check_out_dir(ROOT / "results" / "x")
    assert XB.check_out_dir(XB.XBATCH_ROOT / "XC_workflow_end_to_end" / "shards") is not None
    assert XB.out_dir("XC") == XB.XBATCH_ROOT / "XC_workflow_end_to_end"


def test_m2_gate_compare():
    """재현 관문: 같은 값은 통과, 같은 노드 수준에서는 아주 작은 차도 실패, 로컬·Rescale 수준은 1e-4 cm² 안이면 통과. key_map 으로 이름을 맞춘다."""
    rng = np.random.RandomState(1)
    kr = [("R1", "catboost_lo", "1", "cell", 10, d, 0, 0.25) for d in range(2)]
    ref = {("Lena|x", 1): _store("Lena|x", 1, rng, kr)}
    new = {("Lena|x", 1): H4.BlockStore._from_arrays("Lena|x", 1, ref[("Lena|x", 1)].blocks, ref[("Lena|x", 1)].ncell, [], [], [], {})}
    for k in kr:
        s, cn = ref[("Lena|x", 1)].get(k)
        new[("Lena|x", 1)].add_sse(k[:3] + ("rand",) + k[4:], s + (5e-5 if k[5] == 1 else 0.0), cn)
    kmap = lambda k: k[:3] + ("cell",) + k[4:]                            # noqa: E731
    g0 = XB.gate_compare(new, ref, "same_node", key_map=kmap)
    assert len(g0) == 2 and XB.gate_summary(g0) == dict(n_keys=2, n_fail=1, passed=False)
    g1 = XB.gate_compare(new, ref, "local_rescale", key_map=kmap)
    assert XB.gate_summary(g1)["passed"]
    assert len(XB.gate_compare(new, ref, "same_node")) == 0, "이름을 맞추지 않으면 공통 키가 없다"
    assert not any(XB.FORBIDDEN_OUT.search(c) for c in g0.columns)


# ---------------------------------------------------------------- (n) 봉인
def test_n_sealed_writer(tmp_path, capsys):
    df = pd.DataFrame(dict(test_id=["XB-1"] * 3, delta=[-0.2, 0.1, 0.4], verdict4=["미결정"] * 3))
    recs = XB.write_sealed("XB_multisource_stacking", {"xb_tests.csv": df, "xb_meta.json": dict(n=3)}, root=tmp_path, allowed=[tmp_path])
    out = capsys.readouterr().out
    assert XB.FORBIDDEN_OUT.search(out) is None and "행 3" in out and "-0.2" not in out
    d = XB.sealed_dir("XB_multisource_stacking", tmp_path)
    assert (d / "xb_tests.csv").exists() and (d / "README.txt").exists()
    man = json.loads((d / "sealed_manifest.json").read_text())
    assert {e["file"] for e in man["entries"]} == {"xb_tests.csv", "xb_meta.json"} and recs[0]["sha256"] == XB.sha256_file(d / "xb_tests.csv")
    with pytest.raises(PermissionError):
        XB.assert_not_sealed(d / "xb_tests.csv")
    assert XB.assert_not_sealed(tmp_path / "open.csv")
    with pytest.raises(SystemExit):
        XB.write_sealed("XB_multisource_stacking", {"t.csv": df}, root=tmp_path)                       # 기본 허용 뿌리 밖
    with pytest.raises(ValueError):
        XB.write_sealed("XB_multisource_stacking", {"../t.csv": df}, root=tmp_path, allowed=[tmp_path])


# ---------------------------------------------------------------- (o) 누설 시험 틀
def test_o_leakage_template_transfer_unit():
    """틀: (1) 단위를 한 번 돌려 선택 라벨 합집합을 기록에서 얻는다. (2) 선택 밖 A 라벨과 B 라벨을 바꾼 문맥에서 다시 돌린다. (3) 모든 키의 예측이
    같아야 한다. 음성 대조로 선택 라벨 하나를 바꾸면 예측이 달라져야 한다(시험이 민감함을 확인)."""
    a = XB.h54_args(exp="wf4", grid=[10, 20], threads=1, cb_iters=10, seeds=1, draws_cap=2, nboot=200)

    def run_fn(c):
        W.TUnit(a, c, "T").run()
    c = make_tctx()
    res = XB.leakage_invariance(run_fn, c)
    assert res["ok"] and res["same_keys"] and res["max_abs_diff"] == 0.0 and res["n_keys"] > 10
    assert 0 < res["n_keep"] < res["n_A"], "선택 밖 A 라벨이 있어야 시험이 의미가 있다"
    with XB.h54_trace() as (p1, tr):
        run_fn(c)
    keep = XB.selected_union(tr)
    c_bad = XB.perturb_labels(c, keep[1:], seed=3)                       # 선택 라벨 하나(keep[0])도 바꾼다
    with XB.h54_trace() as (p2, _):
        run_fn(c_bad)
    assert not XB.compare_predictions(p1, p2)["ok"], "선택 라벨을 바꿔도 예측이 같다(시험이 둔하다)"


# ---------------------------------------------------------------- (p) XI warm_trim
def test_p_warm_trim():
    c = make_rctx10()
    sW = c.sB[c.maskW]
    c2, keep = XB.trim_ctx_warm(c, float(np.min(sW)))
    assert np.array_equal(c2.yB, c.yB) and np.array_equal(c2.sB, c.sB) and np.array_equal(c2.maskW, c.maskW), "W·I 채점 셀은 같아야 한다"
    assert np.array_equal(c2.sA, c.sA[keep]) and np.all(c.sA[~keep] >= np.min(sW)) and np.all(c2.sA < np.min(sW))
    assert c2.meta["frac_W_outside_A"] == 1.0 and c2.meta["n_trim"] == int((~keep).sum()) > 0 and c2.meta["n_A"] == len(c2.yA)
    assert len(c2.XA) == len(c2.yA) == len(c2.blkA) == len(c2.PA) and c.meta["n_valid_splits"] == 1
    a = XB.h54_args(exp="wf10", grid=[20, -1], threads=1, cb_iters=10, seeds=1, draws_cap=1, nboot=100)
    U = XB.R10TrimUnit(a, c2, dry=True).run()
    rows, st, stats = U.finish()
    assert [s_.target for s_ in st] == ["T~warm_trim|r", "T~warm_trimW|r", "T~warm_trimI|r"] and stats["n_rows"] > 5
    with pytest.raises(ValueError):
        XB.R10TrimUnit(a, c2, variant="warm")


# ---------------------------------------------------------------- (q) 부록 A(실제 자료, 세기 범주)
@REAL
def test_q_appendix_a_split_structure():
    a = XB.h54_args(splits=XB.TRANSFER_SPLITS, threads=1)
    for fam, al in XB.APPENDIX_A:
        keep, rows = XB.split_plan(a, al, fam)                            # check=True: 부록 A 와 다르면 AssertionError
        assert len(keep) == XB.APPENDIX_A[(fam, al)]["n"]
        assert all(r["split"] in XB.SPLIT_FAMILIES[fam] for r in rows)
    for al, want in XB.SPLITS_1_5_MIN_NB.items():                         # 분할 1–5 재계산(h40 규칙)
        spec, tgt, _ = XB.resolve(al)
        _, D = XB.get_data(a, al)
        info = D.split_structure(tgt)
        nbs = [v["nb_eval"] for v in info.values() if v["valid"]]
        assert min(nbs) == want, f"{al}: 분할 1–5 최소 채점 블록"


@REAL
def test_r_licence_check():
    rs = XB.lic_check(strict=True)
    assert [r["spec"] for r in rs] == list(XB.RUN_TABLE_DIRS) and all(r["n_unverified"] == 0 and r["n_rows"] > 17000 for r in rs)
    assert all(r["soil_link"] == "../../../e5_soil_tdd_v4.csv" for r in rs)
