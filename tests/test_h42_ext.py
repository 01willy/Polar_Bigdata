"""scripts/3_deep_learning/h42_label_grid_ext.py 단위 시험(계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6A).

가벼운 시험은 학습을 하지 않는다. 학습기 자리에 대체 함수(학습 목표의 평균 + 입력 1열의 선형 항)를 넣어 학습 행렬과 저장 경로만 확인한다.
학습을 하는 시험(표지 HEAVY)은 환경 변수 LG_RUN_HEAVY=1 일 때만 실행한다. 이 서버는 공유 서버이므로 Rescale 사전 점검 작업에서 실행한다.

가벼운 시험(대체 학습기 또는 순수 함수)
(a) 기준 방법의 학습 행렬과 저장 SSE 가 h40.run_ctx 와 같다.
(b) 비선택 라벨을 교란해도 새 방법의 저장값이 변하지 않는다(축 x1, x2, x9, x9f, n3, x5, ridge).
(c) 위약: shuffle 은 풀 값의 다중집합을 보존하고, 라벨 셀은 실측이며, 복원 추출 색인이 D1 과 같다.
(d) R1s 는 선택 라벨의 다중집합을 보존하고 E 를 순열 뒤 값으로 구한다.
(e) n = 0 항등: F1n = F1a, RM(λ 1) = V2 = RM0(λ 1), R1@k3 = R0.
(f) verdict4 의 경계값, Holm, 보조 CI(지역 블록 공통 재표집).
(g) 분할 4·5 축의 열거와 공통 설정 해시.
(h) 셀 제외 변형에서 표지 셀이 원천, 라벨, 채점에 없다.
(i) 앵커 대체값이 유한하고 ρ 가 원천에서만 나온다.
(j) --resume 과 설정 해시, 실행 거부(스모크 포함), 기본값, 허용 표지 없는 실행의 자원 제한, 실행 순서, 선택 표의 복구 경로.
(k) 거리 층의 셀 수 합이 블록 셀 수와 같다.
(l) 구간 분위 색인.
(m) 근접 단: 채점 셀은 반복마다 한 번 채점되고 S_randm 의 라벨 수는 |A| 다. 채점 fold 의 셀은 학습에 없다.
    채점 fold 의 라벨을 교란해도 그 fold 의 학습 행렬과 학습 목표(계수 E1 포함)가 변하지 않는다.
(n) 집계: 합성 저장소에서 가설 표와 표들이 만들어진다(행이 없는 가설은 '행 없음').
(p) 판정 문구의 부분 표기: 풀 지역 3/4, 분할 2/3(확인적 가설은 판정 불가), 효과 크기 0.5 cm 미만.
(q) 판정할 수 없는 대비: L29 의 기준선 결측, L30 의 기각 조건 충족, L18 의 부분 표기.
(r) 세는 형식의 판정: tri_count, L24, L25, L27, L31.
(s) L17 의 λ 규칙, L28 의 풀 지역 불일치, 동등성 대비의 Holm 보정.
(t) 결측 대체 민감도 열: 결측이 없는 셀에서 load_base 의 열과 같다(실자료가 있을 때만).
(u) n = 0 의 F1n 은 F1a 의 적합 표지를 쓴다. 결측 대체 민감도 행이 x9 축에 저장된다.
학습을 하는 시험
(A) 기준 방법의 저장 SSE 가 h40.run_ctx 와 같다(CatBoost, 1e-9).
(B) 비선택 라벨 교란 불변(CatBoost).
(C) 모든 cpu 축이 합성 자료에서 돌고 상태가 ok 다. 변수 기여 행이 나온다.
(D) 실자료 한 단위의 실행, 재개, 집계.
(E) 프로세스 풀 경로(--workers 2)의 실행.
실행(Rescale): LG_RUN_HEAVY=1 CUDA_VISIBLE_DEVICES="" python3 -m pytest -q tests/test_h42_ext.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # 시험은 GPU 를 쓰지 않는다
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HEAVY = pytest.mark.skipif(os.environ.get("LG_RUN_HEAVY", "") != "1",
                           reason="학습을 하는 시험은 LG_RUN_HEAVY=1 일 때만 실행한다(공유 서버 CPU 보호, Rescale 사전 점검에서 실행)")
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


X = _load("h42_label_grid_ext", "scripts/3_deep_learning/h42_label_grid_ext.py")
H = X.H                                                       # h42 가 쓰는 h40 모듈(TRACE 는 이 모듈에 넣는다)
D_FEAT = len(X.FEATS)
NS, NA, NB = 400, 60, 50
BASE_ARGS = ["--n-grid", "0,3,10,all", "--draws", "2", "--seeds", "1", "--cb-iters", "15", "--threads", "1", "--splits", "1", "--r", "1.0",
             "--learner-n-grid", "0,3,10,all", "--place-n-grid", "10", "--epochs", "1"]
TUNED = dict(D=dict(iterations=20, depth=3, learning_rate=0.05, l2_leaf_reg=3.0), R=dict(iterations=20, depth=3, learning_rate=0.05, l2_leaf_reg=3.0))


def make_unit(y_override=None, seed=0, flags=False, yB_override=None):
    """합성 작업 단위. 특징 0 열은 행 식별자(원천 = −1−j, 대상 A = j, 채점 B = 1000 + j).
    yB_override 는 채점 셀의 라벨을 바꾼다(난수 순서는 바뀌지 않는다)."""
    rng = np.random.RandomState(seed)
    Xs = rng.randn(NS, D_FEAT); Xs[:, 0] = -1.0 - np.arange(NS)
    ss = rng.uniform(20, 40, NS); ys = 1.6 * ss + 3 * Xs[:, 1] + rng.randn(NS) * 2
    XA = rng.randn(NA, D_FEAT); XA[:, 0] = np.arange(NA)
    sA = rng.uniform(20, 40, NA); yA = 1.3 * sA + 3 * XA[:, 1] + rng.randn(NA) * 2
    XB = rng.randn(NB, D_FEAT); XB[:, 0] = 1000.0 + np.arange(NB)
    sB = rng.uniform(20, 40, NB); yB = 1.3 * sB + 3 * XB[:, 1] + rng.randn(NB) * 2
    XA[5, 3] = np.nan                                           # 결측 대체 경로
    if y_override is not None:
        yA = y_override(yA.copy())
    if yB_override is not None:
        yB = yB_override(yB.copy())
    mac = np.repeat(["r1", "r2", "r3", "r4"], NS // 4)
    c = H.Ctx("T", "x", 1, "T", Xs, ys, ss, mac, XA, yA, sA, np.repeat(np.arange(6), NA // 6), XB, yB, sB, np.repeat(np.arange(5), NB // 5))
    ext = {}
    for k, n, s, off in (("src", NS, ss, 0), ("A", NA, sA, 10000), ("B", NB, sB, 20000)):
        ku = 1.5 * s + rng.randn(n)
        ku[::7] = np.nan                                        # p4_ku 결측(비동토 판정 셀)
        fl = np.zeros(n, bool)
        if flags:
            fl[::5] = True
        ext[k] = dict(loc_id=off + np.arange(n), lat=60 + rng.rand(n) * 3, lon=-150 + rng.rand(n) * 6, e5_tdd=s ** 2, p4_ku=ku,
                      p2_edaphic=1.2 * s + rng.randn(n), cci_alt=1.4 * s + rng.randn(n) * 3, e5_sqrt_tdd_soil=0.8 * s + rng.randn(n) * 0.5,
                      tdd_matched=(s + rng.randn(n) * 0.5) ** 2, early=fl.copy(), gpr=np.roll(fl, 1))
    ext.update(flags_src="synthetic", tdd_src="synthetic")
    # 결측 대체 민감도 열(학습 쪽 중앙값으로 채운 값의 자리): 기본 열에 작은 차이를 준 값
    ext["phys_tr"] = {k: dict(p4_ku=ext[k]["p4_ku"] + 0.05, p2_edaphic=ext[k]["p2_edaphic"] - 0.05) for k in ("src", "A", "B")}
    ext["phys_tr"]["info"] = dict(n_filled={}, n_unfilled=0)
    return c, ext


FAIL_METHODS: set = set()      # 시험용: 이 방법 이름의 적합을 실패시킨다


def stub_fit(self, learner, axis, build, seed, preds, info, init, iters, cont):
    """대체 학습기: 학습 없이 학습 목표의 (가중) 평균 + 입력 1열의 선형 항을 돌려준다. 학습 행렬은 추적 목록에 남는다."""
    Xtr, ytr, w = build()
    self._trace(dict(info or {}, learner=learner, axis=axis, seed=seed), Xtr, ytr, w)
    if (info or {}).get("method") in FAIL_METHODS:
        raise RuntimeError("시험용 적합 실패")
    ytr = np.asarray(ytr, float)
    if not np.all(np.isfinite(ytr)):
        raise ValueError("학습 목표에 비유한 값이 있다")
    mu = float(np.average(ytr, weights=w)) if w is not None else float(ytr.mean())
    return None, [mu + 0.01 * np.nan_to_num(np.asarray(p, float)[:, 1]) + 0.001 * int(seed) if len(p) else np.zeros(0) for p in preds]


@pytest.fixture()
def stub(monkeypatch):
    monkeypatch.setattr(H.Fitter, "_fit", stub_fit)
    monkeypatch.setattr(X.FitterX, "_fit", stub_fit)
    return True


def ha(extra=(), part="cpu"):
    return H.parse_args(["--part", part] + BASE_ARGS + list(extra))


def run_x(axis, unit=None, extra=(), learners=None, trace=True, dry=False):
    c, ext = unit or make_unit()
    H.TRACE = [] if trace else None
    try:
        rows, st, stats, cells = X.run_ctx_x(c, ext, axis, ha(extra, X.AX[axis]["part"]), dry=dry, learners=learners, tuned=TUNED)
        return rows, st, stats, cells, (H.TRACE or [])
    finally:
        H.TRACE = None


def run_h(extra=(), unit=None):
    c, _ = unit or make_unit()
    args = ha(["--methods", "P0,P1,P2,D0,D1,R0,R1,R2", "--learners", "catboost_lo", "--alphas", "1"] + list(extra))
    H.TRACE = []
    try:
        rows, st, stats = H.run_ctx(c, "cpu", args, learner_axis=False, nested=False)
        return rows, st, stats, (H.TRACE or [])
    finally:
        H.TRACE = None


def _same_base(extra=()):
    _, sx, stx, _, tx = run_x("base", extra=extra)
    _, sh, sth, th = run_h(extra)
    kx = [k for k in sx.keys if "#" not in k[0]]
    kh = [k for k in sh.keys if k[3] == "cell"]
    assert set(kx) == set(kh) and len(kx) > 30
    Sx, Cx = sx.matrices(kx); Sh, Ch = sh.matrices(kx)
    assert np.array_equal(Cx, Ch) and np.allclose(Sx, Sh, rtol=1e-9, atol=1e-9)
    ex = {(e["method"], e["n"], e["draw"], e["seed"]): e for e in tx}
    eh = {(e["method"], e["n"], e["draw"], e["seed"]): e for e in th if e["placement"] == "cell"}
    assert set(ex) == set(eh) and len(ex) > 10
    for k, e in ex.items():
        assert e["Xtr"].shape == eh[k]["Xtr"].shape, k
        assert np.allclose(e["Xtr"], eh[k]["Xtr"], equal_nan=True, atol=1e-9) and np.allclose(e["ytr"], eh[k]["ytr"], atol=1e-9), k
        assert (e["w"] is None) == (eh[k]["w"] is None)
    return stx, sth


# ================================================================ 가벼운 시험
def test_a_base_matches_h40_matrices(stub):
    stx, sth = _same_base()
    assert stx["status"] == "ok" and stx["n_fit"]["base"] == sth["n_fit"]["method"], "적합 수가 h40 의 방법 축과 같아야 한다"


def _free_cells(n_list=(3, 10), draws=2):
    used = set()
    for n in n_list:
        for d in range(draws):
            used |= set(H.draw_cells("T", "x", 1, n, d, NA).tolist())
    return np.array(sorted(set(range(NA)) - used), int)


def _perturb_invariant(axes):
    free = _free_cells()
    assert len(free) > 10

    def perturb(y):
        y[free] = y[free] * 7.0 + 500.0
        return y
    grid = ["--n-grid", "0,3,10", "--learner-n-grid", "0,3,10"]
    for axis in axes:
        _, s1, st1, _, _ = run_x(axis, make_unit(flags=True), grid, trace=False)
        _, s2, st2, _, _ = run_x(axis, make_unit(perturb, flags=True), grid, trace=False)
        n_keys = 0
        for a_, b_ in zip([s1] + list(s1.extra), [s2] + list(s2.extra)):
            k1 = [k for k in a_.keys if k[3] == "cell"]; k2 = [k for k in b_.keys if k[3] == "cell"]
            assert k1 == k2, axis
            n_keys += len(k1)
            S1, C1 = a_.matrices(k1); S2, C2 = b_.matrices(k2)
            assert np.array_equal(C1, C2) and np.allclose(S1, S2, rtol=1e-9, atol=1e-9), f"{axis}: 비선택 라벨을 바꾸자 결과가 달라졌다(누설)"
        assert n_keys > 10, axis
    # 대조: 라벨 전량은 교란에 반응해야 한다(시험이 교란을 실제로 감지함을 확인)
    _, s1, _, _, _ = run_x("x1", make_unit(), trace=False)
    _, s2, _, _, _ = run_x("x1", make_unit(perturb), trace=False)
    ka = ("F1n", "catboost_lo", "1", "cell", -1, 0, 0, 1.0)
    assert not np.allclose(s1.get(ka)[0], s2.get(ka)[0])


def test_b_unselected_labels_never_used(stub):
    _perturb_invariant(["x1", "x2", "x9", "x9f", "n3", "x5", "ridge"])


def _check_trace_rows(tr):
    """학습 행렬의 대상 행은 선택 라벨(sel)뿐이다. 유사라벨을 넣는 방법의 비선택 행은 따로 본다. 채점 셀은 없다."""
    n = 0
    for e in tr:
        ids = e["Xtr"][:, 0]
        assert not np.any(ids >= 1000), f"{e['method']}: 채점 셀이 학습 행렬에 있다"
        if e["method"].startswith(("D1", "R2")):
            continue
        tgt = ids[ids >= 0].astype(int)
        assert set(tgt.tolist()) <= set(int(v) for v in e["sel"]), f"{e['method']} n={e['n']}: 비선택 대상 셀이 학습 행렬에 있다"
        n += 1
    return n


def test_b2_training_rows_are_selected_only(stub):
    for axis in ("x1", "x2", "x9", "n3", "x5", "ridge"):
        _, _, _, _, tr = run_x(axis, extra=["--n-grid", "0,3,10", "--learner-n-grid", "0,3,10"])
        assert _check_trace_rows(tr) > 5, axis


def test_c_placebo_pool_and_index(stub):
    c, ext = make_unit()
    base = 1.4 * c.sA
    sh = X.placebo_pool("shuffle", base, c.y_src, "T", "x", 1, 0)
    assert np.allclose(np.sort(sh), np.sort(base)) and not np.allclose(sh, base)
    assert np.allclose(X.placebo_pool("const_t", base, c.y_src, "T", "x", 1, 0), base.mean())
    assert np.allclose(X.placebo_pool("const_src", base, c.y_src, "T", "x", 1, 0), c.y_src.mean())
    ab = X.affine_ls(ext["src"]["e5_tdd"], c.y_src)
    tl = X.placebo_pool("tddlin", base, c.y_src, "T", "x", 1, 0, tdd=ext["A"]["e5_tdd"], tdd_ab=ab, ratio=0.9)
    assert np.allclose(tl, 0.9 * (ab[0] + ab[1] * ext["A"]["e5_tdd"]))
    ps = X.ps_index("T", "x", 1, 0, NA, NS, 1.0)
    want = np.random.RandomState(H.seed_of("lg-ps", "T", "x", 1, 0)).choice(NA, int(round(1.0 * NS)), replace=True)
    assert np.array_equal(ps, want)
    sel = np.array([3, 8, 20])
    v = X.pseudo_values(sh, ps, sel, c.yA)
    m = np.isin(ps, sel)
    assert m.any() and np.allclose(v[m], c.yA[ps[m]]) and np.allclose(v[~m], sh[ps[~m]])
    r = X.pseudo_values(None, ps, sel, c.yA, base)
    assert np.allclose(r[m], c.yA[ps[m]] - base[ps[m]]) and np.all(r[~m] == 0)
    # 실행 경로: 위약과 D1 의 유사라벨 행은 같은 셀(같은 복원 추출 색인)이고 라벨 셀은 실측이다
    _, _, _, _, tr = run_x("x2", extra=["--n-grid", "0,10"])
    _, _, _, _, tb = run_x("base", extra=["--n-grid", "0,10"])
    d1 = [e for e in tb if e["method"] == "D1" and e["n"] == 10 and e["draw"] == 0][0]
    for p in X.PLACEBOS:
        e = [e for e in tr if e["method"] == f"D1@{p}" and e["n"] == 10 and e["draw"] == 0][0]
        assert np.array_equal(e["Xtr"][:, 0], d1["Xtr"][:, 0]), p
        ids = e["Xtr"][:, 0].astype(int); sel10 = set(int(v) for v in e["sel"])
        lab = np.array([i >= 0 and i in sel10 for i in ids])
        assert lab.sum() >= 10 and np.allclose(e["ytr"][lab], c.yA[ids[lab]])
    e30 = [e for e in tr if e["method"] == "D1@r30" and e["n"] == 10 and e["draw"] == 0][0]
    assert e30["Xtr"].shape[0] == NS + 10 + 30 * NS


def test_d_label_shuffle(stub):
    c, _ = make_unit()
    _, st, _, _, tr = run_x("x2", extra=["--n-grid", "0,10,all"])
    for n, nl in ((10, 10), (-1, NA)):
        es = [e for e in tr if e["method"] == "R1s" and e["n"] == n and e["draw"] == 0]
        assert len(es) == 1
        e = es[0]; sel = np.asarray(e["sel"], int)
        y_sh = X.shuffled_labels(c.yA[sel], "T", "x", 1, n, 0)
        assert np.allclose(np.sort(y_sh), np.sort(c.yA[sel])) and not np.allclose(y_sh, c.yA[sel])
        E1s = X.shrink(H.ls_E(y_sh, c.sA[sel]), c.E0, nl, 10.0)
        lab = e["Xtr"][:, 0] >= 0
        assert lab.sum() == nl and np.allclose(e["ytr"][lab], y_sh - E1s * c.sA[sel])
        s_, cnt = st.get(("P1s", "none", "1", "cell", n, 0, -1, 0.0))
        assert np.isclose(s_.sum(), np.sum((E1s * c.sB - c.yB) ** 2))
    assert not any(k[0].startswith("D1@") and k[4] == -1 for k in st.keys), "전량에는 R1s, P1s 만 있다"


def test_e_n0_identities(stub):
    _, s1, _, _, _ = run_x("x1")
    _, sb, _, _, _ = run_x("base")
    _, s9, _, _, _ = run_x("x9")

    def sse(st, m, lam, lr="catboost_lo"):
        return st.get((m, lr, "1", "cell", 0, 0, 0, lam))[0]
    assert np.allclose(sse(s1, "F1n", 1.0), sse(s1, "F1a", 1.0))
    assert np.allclose(sse(s1, "RM", 1.0), sse(s1, "V2", 0.0)) and np.allclose(sse(s1, "RM0", 1.0), sse(s1, "V2", 0.0))
    assert np.allclose(sse(s1, "RMc", 1.0), sse(s1, "V2c", 0.0))
    for lam in (0.25, 0.5, 1.0):
        assert np.allclose(sse(s9, "R1@k3", lam), sse(sb, "R0", lam)) and np.allclose(sse(s9, "R1@k30", lam), sse(sb, "R1", lam))
        assert np.allclose(sse(s1, "RM", lam), sse(s1, "RM0", lam))


def test_f_verdict_holm_common_ci():
    v = X.verdict4
    assert v(-1.0, -0.1, -0.8, -0.05, 0.5) == "우세" and v(0.1, 1.0, 0.05, 0.8, 0.5) == "열세"
    assert v(-1.0, 0.0, -0.8, -0.05, 0.5) == "미결정", "상한 0 은 우세가 아니다"
    assert v(-0.5, 0.5, -0.3, 0.2, 0.5) == "동등" and v(-0.51, 0.2, -0.3, 0.2, 0.5) == "미결정" and v(-0.51, 0.2, -0.3, 0.2, 1.0) == "동등"
    assert v(-1.0, -0.1, -0.2, 0.3, 0.5) == "미결정", "한 가중만 0 을 제외하면 우세가 아니다"
    assert v(np.nan, 0.1, -0.1, 0.1, 0.5) == "판정 불가"
    assert np.allclose(X.MS.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    d = np.linspace(-1, 1, 2001)
    assert abs(X.eq_p(d, 0.5) - 0.25) < 0.01 and X.eq_p(d * 0.1, 0.5) <= 1.0 / 2001 + 1e-12
    tm = _mk_tm("Lena|x", 10, 0)
    gA, gB = ("R1", "catboost_lo", "1", "cell", 40, 0.25), ("P1", "none", "1", "cell", 40, 0.0)
    r = H.contrast(tm, gA, gB, return_dist=True)
    rc = X.boot_delta_common(tm, gA, gB)
    assert np.isclose(rc["delta"], r["delta"]) and np.isclose(rc["delta_beq"], r["delta_beq"])
    assert rc["dist"].shape == r["dist"].shape and np.isfinite(rc["dist"]).all()
    lo, hi = np.percentile(rc["dist"], [2.5, 97.5])
    assert lo < rc["delta"] < hi < 0
    # 분할의 채점 블록이 합집합의 일부일 때: 다중도가 모두 0 인 반복은 그 분할만 빠진다
    tm2 = _mk_tm("Canada|x", 6, 1, disjoint=True)
    rc2 = X.boot_delta_common(tm2, gA, gB)
    assert rc2["n_blocks_union"] == 12 and np.isfinite(rc2["dist"]).all()
    s_ = X.region_stats(tm, gA, gB)
    row = X.stats_row(s_, X.parse_args([]), "Lena|x")
    assert row["verdict4"] == "우세" and row["verdict4_common"] == "우세" and row["p_eq"] > 0.5


def _mk_tm(name, nb, seed, nboot=200, disjoint=False, extra=None, stats=None):
    """합성 저장소: 분할 2개, 블록마다 5셀. R1 의 오차 SD 1, P1 의 오차 SD 3. stats = 구간 키 → 셀당 값(미포함 비율 또는 폭)."""
    rng = np.random.RandomState(seed)
    by = {}
    for sp in (1, 2):
        blocks = np.repeat(np.arange(nb) + (nb * (sp - 1) if disjoint else 0), 5)
        st = H.BlockStore(name, sp, blocks)
        y = rng.randn(len(blocks)) * 5 + 50
        st.add(H.P0_KEY, y, y + rng.randn(len(y)) * 4.0)
        st.add(("P1", "none", "1", "cell", 40, 0, -1, 0.0), y, y + rng.randn(len(y)) * 3.0)
        st.add(("R1", "catboost_lo", "1", "cell", 40, 0, 0, 0.25), y, y + rng.randn(len(y)) * 1.0)
        for k, sd in (extra or {}).items():
            st.add(k, y, y + rng.randn(len(y)) * sd)
        for k, per_cell in (stats or {}).items():
            st.add_sse(k, per_cell * st.ncell.astype(float), st.ncell)
        by[sp] = st
    return H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in (1, 2)}, nboot)


def test_g_g45_enumeration_and_cfg():
    a = X.parse_args(["--part", "gpu", "--axes", "g45"])
    hg = H.parse_args(X.g45_argv(a))
    assert hg.TAG == "lgxg" and hg.SPLITS == [1, 2, 3, 4, 5] and hg.LEARNER_SPLITS == [1, 2, 3, 4, 5] and hg.LEARNERS == H.GPU_LEARNERS
    ref = H.parse_args(["--part", "gpu"])
    for lr in H.GPU_LEARNERS:
        c1 = H.cfg_hash(H.unit_cfg(hg, "gpu", "Canada", "x", 4, lr), common=True)
        assert c1 == H.cfg_hash(H.unit_cfg(ref, "gpu", "Canada", "x", 1, lr), common=True), "분할 4·5 조각의 공통 설정 해시가 본 실행과 다르다"
        assert H.cfg_hash(H.unit_cfg(hg, "gpu", "Canada", "x", 4, lr)) == H.cfg_hash(H.unit_cfg(ref, "gpu", "Canada", "x", 1, lr))
    assert X.g45_splits(a) == (4, 5) and X.g45_splits(X.parse_args(["--part", "gpu", "--smoke"])) == (4,)
    sub = X.parse_args(["--part", "gpu", "--learners", "realmlp"])
    assert X.axis_learners(sub, "g45") == ["realmlp"] and X.axis_learners(sub, "r0") == ["realmlp"] and X.axis_learners(sub, "nn") == []
    if not (ROOT / "data" / "processed" / "fidelity_base_v3.csv").exists():
        pytest.skip("실자료 없음")
    units, skipped = X.enumerate_x(a)
    assert len(units) > 0 and {u[3] for u in units} <= {4, 5} and {u[0] for u in units} == {"g45"}
    assert {(u[1], u[2]) for u in units} <= set(H.LEARNER_TARGETS)
    units.sort(key=lambda u: X.priority_x(a, u))
    order = [u[4] for u in units]
    assert order.index("realmlp") > max(i for i, v in enumerate(order) if v == "ftt") > max(i for i, v in enumerate(order) if v == "nflow")


def test_h_variant_excludes_flagged_cells(stub):
    unit = make_unit(flags=True)
    c, ext = unit
    X.TRACE_X = []
    try:
        rows, st, stats, _, tr = run_x("x9f", unit, extra=["--n-grid", "0,10,all"])
        tx = list(X.TRACE_X)
    finally:
        X.TRACE_X = None
    assert [s_.target for s_ in st.extra] == ["T~noearly|x", "T~probe|x"] and stats["n_stores"] == 3 and stats["status"] == "ok"
    assert {e["variant"] for e in tr} == {"noearly", "probe"}
    for variant, col in (("noearly", "early"), ("probe", "gpr")):
        bad_src = set((-1.0 - np.where(ext["src"][col])[0]).tolist())
        bad_A = set(float(v_) for v_ in np.where(ext["A"][col])[0])
        assert len(bad_src) > 10 and len(bad_A) > 5
        sv = [s_ for s_ in st.extra if s_.target == f"T~{variant}|x"][0]
        assert sv.ncell.sum() == int((~ext["B"][col]).sum()) < NB, "표지 셀이 채점에 남았다"
        v = [e for e in tx if e["kind"] == "variant" and e["variant"] == variant][0]
        assert not set(v["src_loc"].tolist()) & set(ext["src"]["loc_id"][ext["src"][col]].tolist())
        assert not set(v["B_loc"].tolist()) & set(ext["B"]["loc_id"][ext["B"][col]].tolist())
        sels = [e for e in tx if e["kind"] == "variant_sel" and e["variant"] == variant]
        assert len(sels) == 4
        for e in sels:
            assert not set(float(v_) for v_ in e["sel"]) & bad_A and set(e["sel"].tolist()) <= set(e["sel0"].tolist())
            assert np.array_equal(e["sel0"], H.draw_cells("T", "x", 1, e["n"], e["draw"], NA)), "추출 색인은 기본 추출과 같아야 한다"
        fits = [e for e in tr if e["variant"] == variant]
        assert len(fits) > 5
        for e in fits:
            ids = set(e["Xtr"][:, 0].tolist())
            assert not ids & bad_src, "표지 셀이 원천에 남았다"
            assert not ids & bad_A, "표지 셀이 라벨에 남았다"
            assert not any(i >= 1000 for i in ids)
        full = [r for r in rows if r["variant"] == variant and r["method"] == "R1" and r["n"] == -1]
        assert full and all(r["n_lab"] == int((~ext["A"][col]).sum()) for r in full), "실제 라벨 수를 기록한다"
    # 기본 저장소에는 P0 만 있다(변형의 값은 변형 저장소에 있다)
    assert set(k[0] for k in st.keys) == {"P0"}


def test_i_anchor_fill():
    rng = np.random.RandomState(0)
    s = rng.uniform(20, 40, 50); sA = rng.uniform(20, 40, 9); sB = rng.uniform(20, 40, 7)
    raw = 1.5 * s; raw[[1, 5]] = np.nan; raw[7] = -3.0; raw[9] = 0.0
    rA = 1.5 * sA; rA[2] = np.nan
    rB = 1.5 * sB; rB[0] = np.inf
    b, rho, nrep = X.anchor_fill(raw, rA, rB, s, sA, sB)
    assert np.isclose(rho, 1.5) and nrep == dict(src=4, A=1, B=1)
    assert all(np.all(np.isfinite(b[k])) and np.all(b[k] > 0) for k in ("src", "A", "B"))
    assert np.allclose(b["src"][[1, 5, 7, 9]], rho * s[[1, 5, 7, 9]]) and np.isclose(b["A"][2], rho * sA[2]) and np.isclose(b["B"][0], rho * sB[0])
    b2, rho2, _ = X.anchor_fill(raw, rA * 9.0, rB * 0.1, s, sA, sB)       # 대상 값을 바꿔도 ρ 는 같다(원천에서만 구한다)
    assert rho2 == rho and np.allclose(b2["src"], b["src"])
    _, rho3, _ = X.anchor_fill(np.full(5, np.nan), rA, rB, s[:5], sA, sB)
    assert not np.isfinite(rho3)


def test_j_resume_cfg_and_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("LG_RESCALE", raising=False)
    a = X.parse_args([])
    assert a.GPUS == [] and a.workers <= 2 and a.threads == 1 and a.AXES == X.AXES_CPU and a.TAG == "lgx" and a.nboot == 10000
    assert X.parse_args(["--threads", "8"]).threads == 1, "허용 표지가 없으면 스레드는 1 이다"
    assert X.parse_args(["--allow-local"]).threads == 4 and X.parse_args(["--allow-local", "--threads", "2"]).threads == 2
    monkeypatch.setenv("LG_RESCALE", "1")
    assert X.parse_args([]).threads == 4 and X.parse_args(["--threads", "2"]).threads == 2
    monkeypatch.delenv("LG_RESCALE", raising=False)
    assert str(a.OUT).endswith("data/processed/lgx") and a.delta_eq == 0.5 and a.delta_eq_aux == 1.0
    g = X.parse_args(["--part", "gpu"])
    assert g.AXES == ["r0", "g45", "nn"] and X.parse_args(["--part", "gpu", "--precheck"]).AXES == ["r0", "nn"]
    s = X.parse_args(["--smoke"])
    hb = X.h40_args(s, "base")
    assert s.TAG == "lgx_smoke" and hb.TAG == "lgxb_smoke" and hb.SPLITS == [1] and hb.N_GRID == [0, 10, 40, -1] and hb.DRAWS == 1 and hb.SEEDS == [0]
    assert hb.TARGETS == [("Russia_W", "x"), ("AL-3", "i"), ("Canada", "x")] and hb.threads == X.CB_THREADS
    p = X.parse_args(["--precheck"])
    assert X.h40_args(p, "x2").N_GRID == [0, 40] and X.h40_args(p, "base").TARGETS == [("Canada", "x")]
    full = X.parse_args(["--threads", "1"])
    assert X.h40_args(full, "base").threads == 4, "CPU 축의 CatBoost 스레드는 4 로 고정한다"
    assert len(X.h40_args(full, "base").TARGETS) == 27 and X.h40_args(full, "base").N_GRID == [0, 3, 10, 40, 160, 320, 1000, -1]
    assert X.h40_args(full, "n4").SPLITS == list(range(1, 51)) and X.h40_args(full, "n4a").SPLITS == list(range(1, 201))
    assert len(X.h40_args(full, "n4").TARGETS) == 15 and X.h40_args(full, "n3").TARGETS == list(H.LEARNER_TARGETS)
    assert X.h40_args(full, "x2").N_GRID == [0, 10, 40, 160, -1] and X.h40_args(full, "base").kappa == 10.0 and X.h40_args(full, "base").r == 10.0
    with pytest.raises(SystemExit):
        X.parse_args(["--part", "cpu", "--axes", "g45"])
    monkeypatch.delenv("LG_RESCALE", raising=False)
    assert not X.run_permitted(a) and X.run_permitted(X.parse_args(["--allow-local"]))
    for extra in ([], ["--smoke"], ["--precheck"]):                       # 스모크와 사전 점검도 허용 표지 없이는 자료를 읽기 전에 거부된다
        with pytest.raises(SystemExit) as e:
            X.main(["--part", "cpu", "--targets", "Russia_W", "--workers", "0", "--threads", "1", "--no-summarize", "--out-dir", str(tmp_path / "no")]
                   + extra)
        assert "거부" in str(e.value), extra
    assert not (tmp_path / "no").exists()
    # 허용 표지 없는 집계: 프로세스 1개, 재표집 1,000회 이하
    loc = X.limit_local(X.parse_args(["--summarize-only", "--workers", "14", "--threads", "4"]))
    assert loc.workers == 1 and loc.nboot == X.LOCAL_NBOOT_MAX and loc.nboot_asked == 10000 and loc.threads == 1
    assert X.curve_argv(loc)[-2:] == ["--nboot", str(X.LOCAL_NBOOT_MAX)]
    ok = X.limit_local(X.parse_args(["--summarize-only", "--workers", "14", "--allow-local"]))
    assert ok.workers == 14 and ok.nboot == 10000 and not hasattr(ok, "nboot_asked")
    # 실행 순서: 확인적 가설에 쓰는 단위(축 base, x1, x2, x9, n3 의 주 4지역 x 모드)가 먼저다
    assert X.confirm_grade("x9", "Lena", "x") == 0 and X.confirm_grade("n3", "Canada", "x") == 0
    assert X.confirm_grade("base", "Alaska", "x") == 1 and X.confirm_grade("prox", "Lena", "x") == 1 and X.confirm_grade("x1", "CA-3", "i") == 1
    # 선택 표의 복구 경로: 새 행이 없으면 표를 쓰지 않는다. 빈 파일과 깨진 파일은 빈 설정으로 읽는다
    s_ = X.parse_args(["--out-dir", str(tmp_path / "sel"), "--tag", "t"])
    assert len(X.write_select(s_, [])) == 0 and not s_.SELECT.exists()
    s_.OUT.mkdir(parents=True, exist_ok=True)
    s_.SELECT.write_text("\n")
    assert X.load_tuned(s_, "Canada", "x") == {} and not X.select_ready(s_, [("Canada", "x")])
    rows = [dict(target="Canada", mode="x", kind=k_, depth=dp, iterations=it, learning_rate=X.TUNE_LR, score_rmse=10.0 + dp + it / 1000.0,
                 n_regions=3, sel_sig=X.select_sig(s_)) for k_ in ("D", "R") for dp, it in X.TUNE_GRID]
    df = X.write_select(s_, rows)
    assert len(df) == 18 and int(df.selected.sum()) == 2
    got = X.load_tuned(s_, "Canada", "x")
    assert set(got) == {"D", "R"} and got["D"]["depth"] == 3 and got["D"]["iterations"] == 200
    assert X.select_ready(s_, [("Canada", "x")]) and not X.select_ready(s_, [("Canada", "x"), ("Lena", "x")])
    assert len(X.write_select(s_, [])) == 0 and len(X.load_tuned(s_, "Canada", "x")) == 2, "새 행이 없으면 기존 표를 보존한다"
    more = [dict(r_, target="Lena") for r_ in rows]
    assert len(X.write_select(s_, more)) == 36 and X.select_ready(s_, [("Canada", "x"), ("Lena", "x")])
    # 조각 이름, 설정 해시, 재개
    b = X.parse_args(["--out-dir", str(tmp_path), "--tag", "t"])
    u = ("x1", "Canada", "x", 1, None)
    pth = X.shard_paths_x(b, *u)
    assert pth["unit"].name == "t1__cpu__Canada__x__s1_unit.json" and pth["cells"].name == "t1__cpu__Canada__x__s1_cells.npz"
    gp = X.parse_args(["--part", "gpu", "--out-dir", str(tmp_path), "--tag", "t"])
    assert X.shard_paths_x(gp, "r0", "Canada", "x", 2, "ftt")["unit"].name == "tr0__gpu__Canada__x__s2__ftt_unit.json"
    assert X.shard_paths_x(b, "ridge", "Canada", "x", 2)["unit"].name == "tr0__cpu__Canada__x__s2_unit.json"
    assert X.unit_state_x(b, *u) == (False, "조각 없음")
    pth["runs"].parent.mkdir(parents=True, exist_ok=True)
    pth["runs"].write_text("x\n"); pth["npz"].write_bytes(b"0")
    cfg = X.unit_cfg_x(b, "x1")
    pth["unit"].write_text(json.dumps(dict(status="ok", cfg_hash=H.cfg_hash(cfg))))
    assert X.unit_state_x(b, *u) == (True, "ok")
    assert X.unit_state_x(X.parse_args(["--out-dir", str(tmp_path), "--tag", "t", "--workers", "9", "--axes", "x1"]), *u)[0]      # 실행 자원은 해시에 없다
    for extra in (["--n-grid", "0,10"], ["--seeds", "1"], ["--draws", "2"]):
        ok, why = X.unit_state_x(X.parse_args(["--out-dir", str(tmp_path), "--tag", "t"] + extra), *u)
        assert not ok and "설정" in why, extra
    pth["unit"].write_text(json.dumps(dict(status="failed", cfg_hash=H.cfg_hash(cfg))))
    assert not X.unit_state_x(b, *u)[0]
    pth["unit"].write_text(json.dumps(dict(status="partial", cfg_hash=H.cfg_hash(cfg), tuned_missing=True)))
    ok_, why = X.unit_state_x(b, *u)
    assert not ok_ and "catboost_tuned" in why, "선택 설정이 없던 조각은 미완료다"
    pth["unit"].write_text(json.dumps(dict(status="ok", cfg_hash=H.cfg_hash(cfg))))
    found = X.find_shards_x(b.SHARDS, "t1")
    assert len(found) == 1 and found[0]["target"] == "Canada" and found[0]["split"] == 1 and found[0]["part"] == "cpu"
    assert X.find_shards_x(b.SHARDS, "t") == [] and X.find_shards_x(b.SHARDS, "t1_smoke") == []
    assert X.unit_cfg_x(b, "x1") != X.unit_cfg_x(b, "base") and H.cfg_hash(X.unit_cfg_x(gp, "r0", "ftt"), common=True) == \
        H.cfg_hash(X.unit_cfg_x(gp, "r0", "tabm"), common=True)


def test_k_distance_strata_counts(stub):
    rows, st, stats, cells, _ = run_x("base", extra=["--n-grid", "0,10,all"])
    for n in (10, -1):
        for m, lr, seed, lam in (("R1", "catboost_lo", 0, 0.25), ("P1", "none", -1, 0.0), ("P0", "none", -1, 0.0), ("D0", "catboost_lo", 0, 1.0)):
            cnt = sum(st.get((f"{m}#{s_}", lr, "1", "cell", n, 0, seed, lam))[1] for s_ in ("near", "mid", "far"))
            assert np.array_equal(cnt, st.ncell), (m, n)
            tot = sum(st.get((f"{m}#{s_}", lr, "1", "cell", n, 0, seed, lam))[0] for s_ in ("near", "mid", "far"))
            ref = H.P0_KEY if m == "P0" else (m, lr, "1", "cell", n, 0, seed, lam)
            assert np.allclose(tot, st.get(ref)[0])
    assert not any("#" in k[0] and k[4] == 0 for k in st.keys), "n = 0 에는 거리 층이 없다"
    d = X.strata3(np.arange(30.0))
    assert d["near"].sum() == 10 and d["far"].sum() == 10 and d["mid"].sum() == 10 and not np.any(d["near"] & d["far"])
    # 셀 단위 저장: 키 × 채점 셀, (n, 추출)별 거리
    assert cells is not None and len(cells.keys) == len(cells.P) > 0 and all(p.shape == (NB,) and p.dtype == np.float32 for p in cells.P)
    assert {k[7] for k in cells.keys} == {"pred", "g"} and {k[0] for k in cells.keys} == {"D0", "D1", "R0", "R1", "R2"}
    assert [k for k in cells.nd] == [[0, 0], [10, 0], [10, 1], [-1, 0]] and np.isnan(cells.D[0]).all() and np.isfinite(cells.D[1]).all()
    # 최근접 거리는 haversine 과 같다
    from polar.m1_ext import haversine_km
    la, lo = np.array([60.0, 61.5]), np.array([-150.0, -148.0])
    lr_, lor = np.array([60.2, 63.0, 61.4]), np.array([-150.5, -149.0, -148.2])
    want = [min(haversine_km(a_, b_, lr_, lor)) for a_, b_ in zip(la, lo)]
    assert np.allclose(X.nearest_km(la, lo, lr_, lor), want, atol=1e-6) and np.isnan(X.nearest_km(la, lo, [], [])).all()


def test_l_conformal_quantile():
    assert X.conformal_k(10, 0.1) == 10 and X.conformal_k(10, 0.2) == 9 and X.conformal_k(9, 0.1) == 9 and X.conformal_k(8, 0.1) == 9
    assert X.conformal_k(40, 0.1) == 37 and X.conformal_k(160, 0.2) == 129
    s = np.arange(1.0, 11.0)
    assert X.conformal_q(s, 0.1) == 10.0 and X.conformal_q(s, 0.2) == 9.0 and np.isinf(X.conformal_q(s[:8], 0.1))
    assert X.weighted_quantile([1, 2, 3, 4], [1, 1, 1, 1], 0.5) == 2.0 and X.weighted_quantile([1, 2, 3, 4], [1, 1, 1, 1], 0.9) == 4.0
    assert X.weighted_quantile([1, 2, 3, 4], [0.97, 0.01, 0.01, 0.01], 0.9) == 1.0
    assert np.allclose(X.log_score([10.0, 10.0], [20.0, 0.1]), [np.log(2.0), np.log(10.0)]), "ŷ 는 1 cm 아래로 내려가지 않는다"
    blk = np.repeat(np.arange(6), 10)
    f, K, flag = X.cv_folds_of(blk, "T", "x", 1, 40, 0)
    assert K == 5 and flag == "" and all(len(set(f[blk == b])) == 1 for b in range(6)) and set(f) == set(range(5))
    f, K, flag = X.cv_folds_of(np.zeros(12, int), "T", "x", 1, 10, 0)
    assert K == 5 and flag == "cell_folds" and sorted(np.bincount(f).tolist()) == [2, 2, 2, 3, 3]
    f, K, flag = X.cv_folds_of(np.array([1, 1, 2]), "T", "x", 1, 3, 0)
    assert K == 2 and flag == ""


def test_l2_interval_rows(stub):
    c, _ = make_unit()
    rows, st, stats, _, tr = run_x("x5", extra=["--n-grid", "0,10,all"])
    assert stats["status"] == "ok" and stats["notes"]["n0_regions"] == ["r1", "r2", "r3", "r4"]
    for n, d, M, lr, seed, lam in ((10, 0, "R1", "catboost_lo", 0, 0.25), (10, 1, "P1", "none", -1, 0.0), (-1, 0, "R0", "catboost_lo", 0, 0.25),
                                   (0, 0, "R0", "catboost_lo", 0, 0.25), (0, 0, "P0", "none", -1, 0.0)):
        for lv in (90, 80):
            miss, cnt = st.get((f"cov{lv}:{M}", lr, "1", "cell", n, d, seed, lam))
            wid, _ = st.get((f"wid{lv}:{M}", lr, "1", "cell", n, d, seed, lam))
            assert np.array_equal(cnt, st.ncell) and np.all(miss >= 0) and np.all(miss <= cnt) and np.all(wid > 0)
        assert st.get((f"wid90:{M}", lr, "1", "cell", n, d, seed, lam))[0].sum() >= st.get((f"wid80:{M}", lr, "1", "cell", n, d, seed, lam))[0].sum()
    cv = [e for e in tr if "cv_fold" in e]
    assert len(cv) > 0
    for e in cv:                                                          # 교차 적합: 제외 묶음의 라벨은 학습 행렬에 없다
        full = H.draw_cells("T", "x", 1, e["n"], e["draw"], NA)
        tgt = set(e["Xtr"][e["Xtr"][:, 0] >= 0, 0].astype(int).tolist())
        assert tgt == set(int(v) for v in e["sel"]) and tgt < set(full.tolist())
    q = [r for r in rows if r["method"] == "cov90:R1" and r["n"] == 10]
    assert q and all(r["cv_folds"] == 5 and np.isfinite(r["q"]) for r in q)


def test_m_prox_tiers(stub):
    c, _ = make_unit()
    rows, st, stats, _, tr = run_x("prox", extra=["--n-grid", "10"])
    assert stats["status"] == "ok"
    pl = {k[3] for k in st.keys}
    assert pl == {"cell", "rand5", "randm", "inblk"} | {f"buf{d}" for d in X.BUFFERS}
    for p, n, reps in (("rand5", -1, X.PROX_REPS), ("randm", -1, X.PROX_REPS), ("inblk", 10, 2)):
        for d in range(reps):
            for key in (("P1", "none", "1", p, n, d, -1, 0.0), ("D0", "catboost_lo", "1", p, n, d, 0, 1.0), ("R1", "catboost_lo", "1", p, n, d, 0, 0.25)):
                assert np.array_equal(st.get(key)[1], st.ncell), "각 채점 셀은 반복마다 한 번 채점된다"
    for e in tr:
        if e["placement"].startswith("buf") or e["placement"] == "cell":
            continue
        ids = e["Xtr"][:, 0]
        nlab = int((ids >= 0).sum())
        rep = e["draw"]
        fid = X.prox_folds("T", "x", 1, rep, NB)
        te = set((1000.0 + np.where(fid == e["fold"])[0]).tolist())
        assert not te & set(ids.tolist()), "채점 fold 의 셀이 학습 행렬에 있다"
        if e["placement"] == "randm":
            assert nlab == NA
        elif e["placement"] == "rand5":
            assert nlab == NA + int((fid != e["fold"]).sum())
        else:
            assert nlab == 10 and np.all(ids[ids >= 0] >= 1000)
    # 채점 fold 의 라벨을 교란해도 그 fold 의 학습 행렬과 학습 목표가 변하지 않는다(R1 의 학습 목표에는 계수 E1 이 들어 있다).
    # 다른 fold 의 학습 목표는 변해야 한다(근접 단은 채점 fold 밖의 B 라벨을 학습에 쓴다. 시험이 교란을 감지함을 확인)
    k0 = 0
    fid0 = X.prox_folds("T", "x", 1, 0, NB)
    te0 = np.where(fid0 == k0)[0]
    assert 5 <= len(te0) <= NB // 2

    def perturb(y):
        y[te0] = y[te0] * 3.0 + 200.0
        return y
    _, st2, _, _, tr2 = run_x("prox", make_unit(yB_override=perturb), extra=["--n-grid", "10"])

    def pick(trace, same_fold):
        return [e for e in trace if e["placement"] in ("rand5", "randm", "inblk") and e["draw"] == 0 and (e["fold"] == k0) == same_fold]
    a1, a2 = pick(tr, True), pick(tr2, True)
    assert len(a1) == len(a2) == 6, "단 3개 × 방법 2개(D0, R1), 반복 0, fold 0, seed 1"
    for e1, e2 in zip(a1, a2):
        assert (e1["method"], e1["placement"], e1["n"]) == (e2["method"], e2["placement"], e2["n"])
        assert np.array_equal(e1["Xtr"], e2["Xtr"], equal_nan=True), (e1["method"], e1["placement"])
        assert np.allclose(e1["ytr"], e2["ytr"], rtol=0, atol=1e-10), f"{e1['method']} {e1['placement']}: 채점 fold 의 라벨이 학습 목표에 들어갔다(누설)"
    b1, b2 = pick(tr, False), pick(tr2, False)
    assert len(b1) == len(b2) > 0
    assert any(not np.allclose(e1["ytr"], e2["ytr"]) for e1, e2 in zip(b1, b2) if e1["placement"] == "rand5")
    iA, iB = X.prox_labels("randm", "T", "x", 1, 0, 0, np.arange(100), 30)
    assert len(iA) == 0 and len(iB) == 30 and len(set(iB.tolist())) == 30
    iA, iB = X.prox_labels("randm", "T", "x", 1, 0, 0, np.arange(10), 30)
    assert len(iA) == 20 and len(iB) == 10 and len(set(iA.tolist())) == 20
    f = X.prox_folds("T", "x", 1, 0, 53)
    assert sorted(np.bincount(f).tolist()) == [10, 10, 11, 11, 11]


def test_n_aggregation_on_synthetic_stores():
    a = X.parse_args(["--nboot", "200"])
    keys = {("D0", "catboost_lo", "1", "cell", 0, 0, 0, 1.0): 5.0, ("F1a", "catboost_lo", "1", "cell", 0, 0, 0, 1.0): 5.0,
            ("F1k", "catboost_lo", "1", "cell", 0, 0, 0, 1.0): 5.0, ("R0", "catboost_lo", "1", "cell", 0, 0, 0, 0.25): 2.0,
            ("R1#near", "catboost_lo", "1", "cell", 40, 0, 0, 0.25): 1.0}
    iv = {("cov90:R1", "catboost_lo", "1", "cell", 40, 0, 0, 0.25): 0.2, ("wid90:R1", "catboost_lo", "1", "cell", 40, 0, 0, 0.25): 30.0}
    tms = {f"{t}|x": _mk_tm(f"{t}|x", 10, i, extra=keys, stats=iv) for i, t in enumerate(X.MAIN4)}
    for tm in tms.values():
        tm.base_target = tm.target
    T = X.TestBook(a, tms, None, None)
    r = T.contrast("L10", "X1", "R0-F1k|n0", X.G("R0", 0), X.G("F1k", 0), n=0)
    assert r["verdict4"] == "우세" and r["n_ci_regions"] == 4 and r["pool"] == "지역 4/4" and r["verdict4_common"] == "우세"
    assert T.contrast("L9", "X1", "없는 키", X.G("F1n", 10), X.G("D0", 10), n=10) is None
    dd = T.dd("L23", "X3c", "이중 차분", (X.G("R0", 0), X.G("F1k", 0)), (X.G("R0", 0), X.G("F1a", 0)), n=0)
    assert dd is not None and abs(dd["delta"]) < 2.0 and dd["kind"] == "이중 차분"
    T.verdict("L10", "X1", "시험")
    df = T.frame()
    assert {"verdict4", "verdict4_common", "verdict4_d10", "holm_p", "n_ci_regions", "pool", "blind", "delta_pct_p0", "small_effect"} <= set(df.columns)
    assert {"holm_p_eq", "confirmatory", "small_note", "splits_min", "splits_expected", "pool_regions", "n_regions_floor"} <= set(df.columns)
    m = df[(df.scope == "MEAN") & df.primary.astype(bool) & df.p_boot.notna()]
    jd = m[~m.verdict4.isin(list(X.NA_VERDICTS))]                          # 판정 불가 행은 Holm 묶음에 넣지 않는다
    assert len(m) == 2 and len(jd) >= 1 and jd.holm_p.notna().all() and (jd.holm_p >= jd.p_boot - 1e-12).all()
    assert m[m.verdict4.isin(list(X.NA_VERDICTS))].holm_p.isna().all()
    assert df[df.test_id == "L10"].confirmatory.all() and not df[df.test_id.isin(["L9", "L23"])].confirmatory.any()
    assert int(r["n_regions_floor"]) == 0 and r["splits_min"] == 2 and r["splits_expected"] == 2 and r["pool_regions"].count("|x") == 4
    assert (df[df.contrast == "없는 키"].verdict4 == "행 없음").all()
    one = dict(list(tms.items())[:1])                                     # 풀 지역 1개: 판정 불가
    T1 = X.TestBook(a, one, None, None)
    assert T1.contrast("L10", "X1", "R0-F1k|n0", X.G("R0", 0), X.G("F1k", 0), n=0)["verdict4"] == "판정 불가"
    # 가설 표 전체: 행이 없는 가설은 '행 없음' 또는 '판정 불가'로 남고 예외가 없어야 한다
    tests = X.build_tests_x(a, tms, None, None, None, conf=None, n4=None)
    v = tests[tests.scope.isin(["verdict", "verdict_aux"])]
    assert {f"L{i}" for i in list(range(9, 19)) + list(range(23, 32))} <= set(v.test_id)
    assert set(v[v.test_id.isin(["L12", "L15", "L25", "L26", "L29", "L30", "L31"])].verdict.str[:5]) == {"판정 불가"}, "행이 없으면 판정하지 않는다"
    assert X.is_aux_method("R1#near") and X.is_aux_method("cov90:R1") and not X.is_aux_method("R1@k3")
    cv = X.curve_view(tms["Lena|x"])
    assert not any(X.is_aux_method(g[0]) for gd in cv.idx.values() for g in gd) and any(X.is_aux_method(g[0]) for gd in tms["Lena|x"].idx.values() for g in gd)
    ct = X.conformal_table(tms, a)
    assert len(ct) == 4 and set(ct.level) == {90} and np.allclose(ct.coverage, 0.8) and np.allclose(ct.width_cm, 30.0)
    assert np.allclose(ct.cov_lo, 0.8) and np.allclose(ct.width_hi, 30.0) and set(ct.method) == {"R1"}
    ri = X.region_inference([-1.0, -0.5, -0.2, -0.8], [0.04, 0.05, 0.02, 0.03])
    assert ri["k"] == 4 and ri["n_neg"] == 4 and np.isclose(ri["p_sign"], 0.125) and np.isclose(ri["p_perm"], 0.125) and ri["hk_hi"] < 0
    r2 = X.region_inference([-1.0, -0.5, 0.3], [0.04, np.nan, 0.02])
    assert r2["k"] == 3 and r2["k_re"] == 2 and r2["n_neg"] == 2 and np.isfinite(r2["mean_re"])
    rm = X.region_inference_main(tms, a)
    assert set(rm.region_set) >= {"main4", "no_russia_w"} and (rm[rm.region_set == "main4"].k == 4).all()
    assert X.base_lam_x("R1@ku") == 0.25 and X.base_lam_x("D1@shuffle") == 1.0 and X.base_lam_x("V2") == 0.0 and X.base_lam_x("RMc") == 1.0
    assert X.base_lam_x("P1@k3") == 0.0 and X.base_lam_x("B:ens") == 0.0 and X.base_lam_x("V2r") == 0.25 and X.base_lam_x("F1k") == 1.0
    assert X.G("P0", 40) == H.P0_GRP and X.G("P1@ku", 40) == ("P1@ku", "none", "1", "cell", 40, 0.0)
    assert X.G("R1", 40, alpha="10") == ("R1", "catboost_lo", "10", "cell", 40, 0.25)


def test_o_dry_count_matches_real_paths(stub):
    for axis in ("base", "x1", "x2", "x9", "x9f", "n3", "prox", "x5", "ridge", "n4a"):
        unit = make_unit(flags=True)
        _, _, s_dry, _, _ = run_x(axis, unit, trace=False, dry=True)
        _, _, s_run, _, _ = run_x(axis, unit, trace=False)
        assert s_dry["n_fit_detail"] == s_run["n_fit_detail"], axis
        assert s_dry["n_rows"] == s_run["n_rows"], axis


# ================================================================ 개정 7 의 판정 규칙(합성 저장소, 학습 없음)
CB = "catboost_lo"


def K(method, n, lam, lr=CB, alpha="1", placement="cell"):
    """저장소 키(추출 0). 물리식과 기준선은 seed −1, 그 밖은 seed 0."""
    return (method, lr, str(alpha), placement, int(n), 0, -1 if lr == "none" else 0, float(lam))


def _mk_tm3(name, nb, seed, specs, splits=(1, 2), nboot=200, expected=None, p0_sd=6.0):
    """합성 저장소(판정 규칙 시험용). specs = {키: (오차 SD, 잡음 묶음 이름)}. 같은 묶음의 키는 같은 단위 잡음에 SD 를 곱한 오차를 쓴다
    (SD 가 같으면 예측이 같아 Δ = 0 이고, SD 가 다르면 모든 재표집에서 부호가 같다). 묶음이 다르면 독립이다.
    P0 의 오차는 묶음 'p0' 의 잡음에 p0_sd 를 곱한 값이다. expected = 기대 분할 목록(주면 분할 완결성 표기를 시험한다)."""
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
    if expected is not None:
        tm.expected_splits = list(expected)
    return tm


def _tms4(specs, nb=10, **kw):
    """주 4지역의 합성 저장소. kw 의 값이 dict 이면 지역 이름 → 값으로 읽는다(예: nb={'Russia_E': 4})."""
    out = {}
    for i, t in enumerate(X.MAIN4):
        sp_ = specs(t) if callable(specs) else specs
        args = {k: (v.get(t, None) if isinstance(v, dict) else v) for k, v in kw.items()}
        args = {k: v for k, v in args.items() if v is not None}
        out[f"{t}|x"] = _mk_tm3(f"{t}|x", (nb.get(t, 10) if isinstance(nb, dict) else nb), 100 + i, sp_, **args)
    return out


def _verdicts(tests, tid, role=None):
    v = tests[(tests.test_id == tid) & tests.scope.isin(["verdict", "verdict_aux"])]
    if role is not None:
        v = v[v.role == role]
    return [str(s_) for s_ in v.verdict]


def _agg(tms, conf=None, n4=None):
    return X.build_tests_x(X.parse_args(["--nboot", "200"]), tms, None, None, None, conf=conf, n4=n4)


L10_SPECS = {K("R0", 0, 0.25): (1.0, "r0"), K("F1k", 0, 1.0): (5.0, "p0"), K("F1a", 0, 1.0): (5.0, "f1a"),
             K("R1", 10, 0.25): (1.0, "r1"), K("F1k", 10, 1.0): (5.0, "f1k10"), K("F1n", 10, 1.0): (5.0, "f1n10")}
L10_OK = "지지: 잔차 구조는 물리 입력 구조보다 오차가 작다"


def test_p_partial_marks():
    # 기준: 4지역, 분할 완결. F1k(n = 0)는 P0 과 같은 잡음의 5.0/5.3 배라 모든 재표집에서 P0 보다 낫고 차이는 0.5 cm 미만이다
    t0 = _agg(_tms4(L10_SPECS, p0_sd=5.3))
    assert _verdicts(t0, "L10", "주") == [L10_OK]
    v11 = _verdicts(t0, "L11")
    assert len(v11) == 1 and v11[0].startswith("기각: F1k 가 n = 0 에서 P0 보다 우세") and X.SMALL_EFFECT_TXT in v11[0]
    r11 = t0[(t0.test_id == "L11") & (t0.scope == "MEAN")].iloc[0]
    assert r11.verdict4 == "우세" and abs(r11.delta) < 0.5 and bool(r11.small_effect) and r11.small_note == X.SMALL_EFFECT_TXT
    rows = t0[t0.scope.isin(["verdict", "verdict_aux"])]
    assert set(rows[rows.confirmatory.astype(bool)].test_id) == set(X.CONFIRMATORY)
    assert set(rows[rows.test_id.isin(["L11", "L13"])].role) == {"보조"}, "L11, L13 의 판정 행은 보조다"
    assert set(rows[(rows.test_id == "L10") & (rows.role == "주")].scope) == {"verdict"}
    # 풀 지역 3/4: 러시아 E 의 채점 블록이 4개라 CI 풀에 들지 못한다
    t1 = _agg(_tms4(L10_SPECS, nb={"Russia_E": 4}, p0_sd=5.3))
    assert _verdicts(t1, "L10", "주") == [f"부분(지역 3/4): {L10_OK}"]
    m1 = t1[(t1.test_id == "L10") & (t1.scope == "MEAN") & t1.primary.astype(bool)]
    assert len(m1) == 4 and set(m1.pool) == {"지역 3/4"} and set(m1.verdict4) == {"우세"} and set(m1.pool_regions) == {"Lena|x,Canada|x,Russia_W|x"}
    # 분할 2/3: 확인적 가설은 판정하지 않고 보조 가설은 부분 표기를 붙인다
    t2 = _agg(_tms4(L10_SPECS, expected={"Lena": [1, 2, 3]}, p0_sd=5.3))
    v10 = _verdicts(t2, "L10", "주")
    assert len(v10) == 1 and v10[0].startswith("판정 불가(분할 2/3")
    assert _verdicts(t2, "L11")[0].startswith("부분(분할 2/3): 기각: F1k 가 n = 0 에서 P0 보다 우세")
    m2 = t2[(t2.test_id == "L10") & (t2.scope == "MEAN") & t2.primary.astype(bool)]
    assert set(m2.splits_min) == {2} and set(m2.splits_expected) == {3} and set(m2.splits_short) == {"Lena|x 2/3"}
    # 표기 함수 자체
    mk = X.TestBook.marks
    row = dict(scope="MEAN", verdict4="우세", delta=-0.2, n_ci_regions=2, n_regions_target=4, splits_min=5, splits_expected=5)
    assert mk([("a", row)]) == dict(pool=(2, 4), splits=None, small=["a"])
    assert mk([("a", dict(row, verdict4="판정 불가")), ("b", None)]) == dict(pool=None, splits=None, small=[])
    assert mk([("a", dict(row, scope="region", n_splits=3, n_splits_expected=5, splits_min=None, splits_expected=None))], point=True)["splits"] == (3, 5)
    assert X.count_valid([("a", row), ("b", None), ("c", dict(row, verdict4="행 없음"))]) == (1, 3, ["b", "c"])
    assert X.part_mark([("a", row)]) == [] and X.part_mark([("a", row), ("b", None)]) == ["대비 1/2(없는 대비: b)"]
    assert X.na_text([("a", row), ("b", None)], "기준선") == "판정 불가(기준선 1/2; 없는 기준선: b)"


def test_q_missing_contrasts():
    bl = ["B:s_aff", "B:ed_raw", "B:ku_raw", "B:cci_raw", "B:cci_aff", "B:ens"] + [f"P0@{b}" for b in X.ANCHORS]
    assert len(bl) == 11

    def base(sd, skip=(), over=None):
        d = {K(m, 0, 0.0, lr="none"): (sd, f"g{i}") for i, m in enumerate(bl) if m not in skip}
        d.update(over or {})
        return d
    # L29: 기준선이 모두 있고 우세가 없다(기준선 오차 SD 9, P0 의 SD 6)
    assert _verdicts(_agg(_tms4(base(9.0))), "L29", "주") == ["원천 계수 Stefan 은 시험한 기준선 가운데 약한 기준선이 아니다"]
    # L29: 기준선 하나가 없으면 지지 문구를 쓰지 않는다
    assert _verdicts(_agg(_tms4(base(9.0, skip=("P0@tddm",)))), "L29", "주") == ["판정 불가(기준선 10/11; 없는 기준선: P0@tddm)"]
    # L29: 우세인 기준선이 있으면 빠진 기준선이 있어도 그 사실을 적는다(P* 는 판정한 기준선 가운데 값)
    v = _verdicts(_agg(_tms4(base(9.0, skip=("P0@tddm",), over={K("B:ens", 0, 0.0, lr="none"): (1.0, "ens")}))), "L29", "주")
    assert len(v) == 1 and v[0].startswith("P0 보다 우세인 기준선이 있다. P* = B:ens.") and "기준선 10/11" in v[0] and "P0@tddm" in v[0]
    # L30: 기각 조건(우세인 학습기)이 이미 충족되면 빠진 학습기가 있어도 기각이다. 충족되지 않으면 판정 불가다
    s30 = {K("D0", 0, 1.0, lr="catboost"): (1.0, "a"), K("D0", 0, 1.0, lr="rf"): (9.0, "b")}
    v = _verdicts(_agg(_tms4(s30)), "L30", "주")
    assert len(v) == 1 and v[0].startswith("기각(일부 대비 판정 불가, 대비 2/3): ") and v[0].endswith("우세 학습기 catboost")
    s30b = {K("D0", 0, 1.0, lr="catboost"): (9.0, "a"), K("D0", 0, 1.0, lr="rf"): (9.0, "b")}
    assert _verdicts(_agg(_tms4(s30b)), "L30", "주") == ["판정 불가(대비 2/3; 없는 대비: catboost_tuned)"]
    s30c = {**s30b, K("D0", 0, 1.0, lr="catboost_tuned"): (9.0, "c")}
    assert _verdicts(_agg(_tms4(s30c)), "L30", "주")[0].startswith("지지: 용량을 키우거나")
    # L12: 열세인 대비가 있으면 빠진 대비가 있어도 기각이다
    s12 = {K("RM", 10, 0.25): (3.0, "g"), K("R1", 10, 0.25): (1.0, "g")}
    v = _verdicts(_agg(_tms4(s12)), "L12", "주")
    assert len(v) == 1 and v[0].startswith("기각(일부 대비 판정 불가, 대비 1/4): n=10 λ=0.25 RM 열세")
    # L18: '우세가 없으면 지지' 형식의 보조 가설은 빠진 대비를 부분 표기로 적는다
    s18 = {K(m, 10, 0.25, alpha=al): (sd, f"{m}{al}") for al in X.ALPHAS_X2 for m, sd in (("R2", 3.0), ("R1", 1.0))}
    v = _verdicts(_agg(_tms4(s18)), "L18")
    assert len(v) == 1 and v[0].startswith("부분(대비 2/6(없는 대비: α=10 n=40, α=10 n=160, α=100 n=40, α=100 n=160)): ")
    assert v[0].endswith("지지: 가중을 주어도 증강 결합은 R1 보다 낫지 않다")
    # 행이 전혀 없으면 판정 불가
    e = _agg(_tms4({}))
    assert _verdicts(e, "L18") == ["판정 불가(대비 0/6; 없는 대비: α=10 n=10, α=10 n=40, α=10 n=160, α=100 n=10, α=100 n=40, α=100 n=160)"]
    assert all(s_.startswith("판정 불가") for s_ in _verdicts(e, "L29") + _verdicts(e, "L30") + _verdicts(e, "L10") + _verdicts(e, "L24"))


def _conf_rows(cells, w0=None, cov=0.9, wid=20.0, n_splits=2, n_exp=2):
    """구간 표의 합성 행. cells = [(지역, n)], w0 = {지역: n = 0 의 R0 폭}."""
    import pandas as pd
    rows = [dict(target=t, mode="x", method="R1", n=n, level=90, coverage=cov, width_cm=wid, n_splits=n_splits, n_splits_expected=n_exp)
            for t, n in cells]
    rows += [dict(target=t, mode="x", method="R0", n=0, level=90, coverage=0.9, width_cm=w, n_splits=n_splits, n_splits_expected=n_exp)
             for t, w in (w0 or {}).items()]
    return pd.DataFrame(rows)


def test_r_counting_rules():
    import pandas as pd
    tc = X.tri_count
    assert tc([True, True, None, None], 2) == "yes" and tc([True, False, False, None], 2) == "unknown" and tc([True, False, False, False], 2) == "no"
    assert tc([None] * 4, 3) == "unknown" and tc([True, True, False, False], 3) == "no" and tc([True, True, True, None], 3) == "yes"
    assert tc([], 1) == "no"
    # ---- L27: 등록 학습기 8종 가운데 시험한 학습기가 적을 때
    lrs = ["ridge"] + list(H.GPU_LEARNERS)
    assert len(lrs) == X.N_LEARNERS_L27 == 8

    def s27(good, bad=()):
        d = {K("R0", n, 0.25): (1.0, f"c{n}") for n in (40, 160, -1)}
        d.update({K("R0", n, 0.25, lr=lr): (1.0, f"{lr}{n}") for lr in good for n in (40, 160, -1)})
        d.update({K("R0", n, 0.25, lr=lr): (9.0, f"{lr}{n}") for lr in bad for n in (40, 160, -1)})
        return {"Canada|x": _mk_tm3("Canada|x", 10, 7, d)}
    v = _verdicts(_agg(s27(lrs[:5])), "L27")
    assert len(v) == 1 and v[0].startswith("판정 불가(시험한 학습기 5/8, 부호가 같은 학습기 5")
    assert _verdicts(_agg(s27(lrs[:6])), "L27") == ["부분(시험 6/8): 학습기와 무관"]
    assert _verdicts(_agg(s27(lrs[:2], lrs[2:5])), "L27") == ["부분(시험 5/8): CatBoost 한정"]
    assert _verdicts(_agg(s27(lrs[:6], lrs[6:])), "L27") == ["학습기와 무관"]
    assert _verdicts(_agg(s27(lrs[:5], lrs[5:])), "L27") == ["CatBoost 한정"]
    t27 = _agg(s27(lrs[:6]))
    cnt = t27[(t27.test_id == "L27") & (t27.scope == "count") & (t27.target == "Canada|x")]
    assert len(cnt) == 3 and set(cnt.n_learners_judged + cnt.n_learners_na) == {9}, "기준 학습기와 등록 학습기 8종"
    # 분할이 기대보다 적은 학습기는 집계에서 뺀다(분할 2 의 저장소에 ridge 의 키가 없다)
    d6 = {K("R0", n, 0.25): (1.0, f"c{n}") for n in (40, 160, -1)}
    d6.update({K("R0", n, 0.25, lr=lr): (1.0, f"{lr}{n}") for lr in lrs[:6] for n in (40, 160, -1)})
    full = _mk_tm3("Canada|x", 10, 7, d6)
    part = _mk_tm3("Canada|x", 10, 7, {k: v for k, v in d6.items() if k[1] != "ridge"})
    by = {1: full.by_all[1], 2: part.by_all[2]}
    mix = H.TMx("Canada|x", by, {sp: dict(dup_of=-1, valid=True) for sp in (1, 2)}, 200)
    mix.base_target = "Canada"
    v = _verdicts(_agg({"Canada|x": mix}), "L27")
    assert len(v) == 1 and v[0].startswith("판정 불가(시험한 학습기 5/8"), "ridge 는 분할 1/2 이라 집계에서 빠진다"
    # ---- L25: 6칸 가운데 행이 없는 칸과 n = 0 의 R0 폭이 없는 칸
    six = [(t, n) for t in ("Lena", "Canada", H.ALASKA) for n in (40, 160)]
    w0 = {t: 30.0 for t in ("Lena", "Canada", H.ALASKA)}
    e = _tms4({})
    assert _verdicts(_agg(e, conf=_conf_rows(six, w0)), "L25") == ["라벨이 구간을 좁힌다"]
    assert _verdicts(_agg(e, conf=_conf_rows(six[:4], w0)), "L25") == ["부분(칸 4/6): 라벨이 구간을 좁힌다"]
    assert _verdicts(_agg(e, conf=_conf_rows(six[:3], w0)), "L25")[0].startswith("판정 불가(칸 3/6")
    assert _verdicts(_agg(e, conf=_conf_rows(six, {"Canada": 30.0, H.ALASKA: 30.0})), "L25") == ["부분(칸 4/6): 라벨이 구간을 좁힌다"]
    assert _verdicts(_agg(e, conf=_conf_rows(six, w0, wid=40.0)), "L25") == ["라벨은 커버리지를 맞추나 폭을 줄이지 못한다"]
    assert _verdicts(_agg(e, conf=_conf_rows(six, w0, cov=0.7)), "L25") == ["라벨 기반 보정이 목표 커버리지를 주지 못한다"]
    assert _verdicts(_agg(e, conf=_conf_rows(six[:2], w0, cov=0.7)), "L25")[0].startswith("판정 불가(칸 2/6")
    assert _verdicts(_agg(e, conf=_conf_rows(six[:3], w0, cov=0.7)), "L25") == ["부분(칸 3/6): 라벨 기반 보정이 목표 커버리지를 주지 못한다"]
    assert _verdicts(_agg(e, conf=_conf_rows(six, w0, n_splits=2, n_exp=3)), "L25") == ["부분(분할 2/3): 라벨이 구간을 좁힌다"]
    assert _verdicts(_agg(e, conf=None), "L25") == ["판정 불가(행 없음)"]
    # ---- L31: 주 4지역 가운데 행이 있는 지역이 적을 때
    def sd(fr, n_splits=50, n_exp=50):
        return dict(splitdist=pd.DataFrame([dict(source="n4", contrast="R1-P0|all", target=f"{t}|x", frac_neg=f, lg5_outside_10_90=False,
                                                 n_splits=n_splits, n_splits_expected=n_exp, primary_source=True) for t, f in fr.items()]),
                    region=pd.DataFrame())
    assert _verdicts(_agg(e, n4=sd(dict(Lena=0.9, Canada=0.9, Russia_W=0.85, Russia_E=0.2))), "L31") == ["분할 구성에 강건"]
    assert _verdicts(_agg(e, n4=sd(dict(Lena=0.9, Canada=0.9, Russia_W=0.85))), "L31") == ["부분(지역 3/4): 분할 구성에 강건"]
    assert _verdicts(_agg(e, n4=sd(dict(Lena=0.9, Canada=0.9))), "L31")[0].startswith("판정 불가(지역 2/4")
    assert _verdicts(_agg(e, n4=sd(dict(Lena=0.1, Canada=0.2))), "L31") == ["부분(지역 2/4): 지역 의존"]
    assert _verdicts(_agg(e, n4=sd(dict(Lena=0.9, Canada=0.9, Russia_W=0.1, Russia_E=0.2))), "L31") == ["지역 의존"]
    assert _verdicts(_agg(e, n4=sd(dict(Lena=0.9, Canada=0.9, Russia_W=0.9, Russia_E=0.9), n_splits=30)), "L31") == ["부분(분할 30/50): 분할 구성에 강건"]
    t31 = _agg(e, n4=sd(dict(Lena=0.9, Canada=0.9, Russia_W=0.85, Russia_E=0.2)))
    assert not t31[(t31.test_id == "L31") & t31.scope.isin(["verdict_aux"])].blind.any(), "판정 행은 캐나다를 포함하므로 비맹검이다"
    # ---- L24: 대상 단위 집계. 판정 불가와 행 없음은 알 수 없는 대상이다
    def s24():
        d = {}
        for n in (40, 160):
            d.update({K("R1#near", n, 0.25): (1.0, f"rn{n}"), K("R1#far", n, 0.25): (5.0, f"pf{n}"),
                      K("P1#near", n, 0.0, lr="none"): (5.0, f"pn{n}"), K("P1#far", n, 0.0, lr="none"): (5.0, f"pf{n}"),
                      K("P0#near", n, 0.0, lr="none"): (5.0, f"pn{n}"), K("P0#far", n, 0.0, lr="none"): (5.0, f"pf{n}")})
        return d
    names = ["Lena|x", "Canada|x", f"{H.ALASKA}|x", "AL-2|i"]
    t4 = {nm: _mk_tm3(nm, 10, 40 + i, s24()) for i, nm in enumerate(names)}
    assert _verdicts(_agg(t4), "L24") == ["잔차 학습의 순가치는 라벨 가까운 셀에 집중된다", "재보정 이득은 라벨과의 거리에 의존하지 않는다"]
    t2 = {nm: t4[nm] for nm in names[:2]}
    v = _verdicts(_agg(t2), "L24")
    assert v[0] == "부분(대상 2/4): 잔차 학습의 순가치는 라벨 가까운 셀에 집중된다" and v[1].startswith("판정 불가(대상 2/4")
    t3 = {nm: t4[nm] for nm in names[:3]}
    assert _verdicts(_agg(t3), "L24")[1] == "부분(대상 3/4): 재보정 이득은 라벨과의 거리에 의존하지 않는다"
    half = dict(t3)                                                       # 한 대상이 n = 40 의 행만 가지면 그 대상은 알 수 없는 대상이다
    half[names[2]] = _mk_tm3(names[2], 10, 42, {k: v for k, v in s24().items() if k[4] == 40})
    assert _verdicts(_agg(half), "L24")[1].startswith("판정 불가(대상 2/4")
    small = dict(t3)                                                      # 채점 블록이 8개 미만이면 CI 가 없어 판정 불가다(지지 쪽으로 세지 않는다)
    small[names[2]] = _mk_tm3(names[2], 4, 42, s24())
    assert _verdicts(_agg(small), "L24")[1].startswith("판정 불가(대상 2/4")


def test_s_lambda_rule_pool_mismatch_and_holm():
    # ---- L17: 동등은 λ 0.25 와 1.0 의 네 대비가 모두 동등일 때다
    def s17(sd025, sd1):
        d = {}
        for n in (10, -1):
            d.update({K("R1", n, 0.25): (1.0, f"a{n}"), K("R1s", n, 0.25): (sd025, f"a{n}"),
                      K("R1", n, 1.0): (1.0, f"b{n}"), K("R1s", n, 1.0): (sd1, f"b{n}")})
        return d
    t_eq = _agg(_tms4(s17(1.0, 1.0)))
    assert _verdicts(t_eq, "L17") == ["대상 라벨의 기여는 수준 정보다"]
    v = _verdicts(_agg(_tms4(s17(1.0, 3.0))), "L17")
    assert v == ["확인하지 못함"], "λ 1.0 에서 동등이 아니면 '수준 정보'를 쓰지 않는다(기준 λ 에서는 우세가 없다)"
    v = _verdicts(_agg(_tms4(s17(3.0, 3.0))), "L17")
    assert v == ["대상 라벨의 공변량과 잔차의 관계가 기여한다: n=10 λ=0.25, n=-1 λ=0.25"]
    only025 = {k: v_ for k, v_ in s17(1.0, 1.0).items() if k[7] == 0.25}
    assert _verdicts(_agg(_tms4(only025)), "L17")[0].startswith("판정 불가(대비 2/4; 없는 대비: n=10 λ=1.0, n=-1 λ=1.0)")
    # ---- 동등성 대비의 Holm 보정: 같은 항목의 동등성 주 대비 묶음. 판정 불가와 행 없음은 묶음에 넣지 않는다
    q = t_eq[(t_eq.test_id == "L17") & (t_eq.scope == "MEAN") & t_eq.primary.astype(bool)]
    assert len(q) == 4 and set(q.verdict4) == {"동등"} and q.holm_p_eq.notna().all() and (q.holm_p_eq >= q.p_eq - 1e-12).all()
    assert np.allclose(q.p_eq, 1.0 / 200) and np.allclose(q.holm_p_eq, 4.0 / 200) and set(q.holm_note_eq) == {""}
    na = t_eq[t_eq.verdict4.isin(list(X.NA_VERDICTS))]
    assert len(na) > 10 and na.holm_p.isna().all() and na.holm_p_eq.isna().all()
    not_eq = t_eq[(t_eq.scope == "MEAN") & t_eq.primary.astype(bool) & (t_eq.test_id == "L9")]
    assert not_eq.holm_p_eq.isna().all()
    # ---- L28: 변형과 기준의 풀 지역 집합이 다르면 그 대비의 안정성은 판정 불가다
    core = {K("D0", 0, 1.0): (9.0, "d0"), K("P1", 10, 0.0, lr="none"): (3.0, "p1"), K("R1", 10, 0.25): (1.0, "r1"),
            K("P1", -1, 0.0, lr="none"): (3.0, "p1a"), K("R1", -1, 0.25): (1.0, "r1a"),
            K("P1@k3", 10, 0.0, lr="none"): (3.0, "p1"), K("R1@k3", 10, 0.25): (1.0, "r1"),
            K("P1@k3", -1, 0.0, lr="none"): (3.0, "p1a"), K("R1@k3", -1, 0.25): (1.0, "r1a")}
    t = _agg(_tms4(core))
    k3 = t[(t.test_id == "L28") & t.scope.isin(["verdict_aux"]) & (t.variant == "κ 3")]
    assert list(k3.verdict) == ["강건"] and not k3.blind.any()
    assert set(t[(t.test_id == "L28") & (t.scope == "MEAN") & (t.variant == "κ 3")].stability) == {"강건"}

    def drop(tname):
        return {k: v_ for k, v_ in core.items() if not (tname == "Russia_E" and k[0] == "R1@k3" and k[4] == 10)}
    t = _agg(_tms4(drop))
    k3 = t[(t.test_id == "L28") & t.scope.isin(["verdict_aux"]) & (t.variant == "κ 3")]
    assert list(k3.verdict) == ["부분(대비 4/5): 강건"]
    r = t[(t.test_id == "L28") & (t.scope == "MEAN") & (t.variant == "κ 3") & (t.contrast == "κ 3|R1-P1|n10")].iloc[0]
    assert r.stability == "판정 불가(풀 지역 불일치)" and r.pool == "지역 3/4" and r.verdict4_base == "우세"
    # 맹검 표지: κ 와 앵커는 비맹검, 6–7월 제외·탐침만·CCI 입력 제외는 맹검이다
    vb = t[(t.test_id == "L28") & t.scope.isin(["verdict_aux"]) & t.variant.notna()]
    bl = dict(zip(vb.variant, vb.blind.astype(bool)))
    assert bl["6–7월 제외"] and bl["탐침만"] and bl["CCI 입력 제외"] and not bl["κ 30"] and not bl["앵커 ku"]


def test_t_phys_train_fill_real():
    """결측 대체 민감도 열: 공변량 결측이 없는 셀에서는 load_base 의 열과 같다. 실자료가 있을 때만 실행한다(학습 없음)."""
    if not (ROOT / "data" / "processed" / "fidelity_base_v3.csv").exists():
        pytest.skip("실자료 없음")
    a = X.parse_args(["--axes", "x9", "--targets", "Lena"])
    HA = X.h40_args(a, "x9")
    D = X.get_data_x(HA)
    c, ext = X.build_unit(D, HA, X.ext_tables(a, D, dry=True), "Lena", "x", 1)
    pt = ext["phys_tr"]()
    df = D.df
    miss = ~np.isfinite(df[list(X.PHYS_INPUT_COLS)].values.astype(float)).all(1)
    assert int(miss.sum()) > 0, "자료에 공변량 결측 셀이 있어야 한다"
    lid = {k: np.asarray(ext[k]["loc_id"]) for k in ("src", "A", "B")}
    bad_id = set(df.loc_id.values[miss].tolist())
    n_diff = 0
    for k in ("src", "A", "B"):
        clean = np.array([v not in bad_id for v in lid[k].tolist()])
        for col in ("p4_ku", "p2_edaphic"):
            a0, a1 = np.asarray(ext[k][col], float), np.asarray(pt[k][col], float)
            assert a0.shape == a1.shape and np.allclose(a0[clean], a1[clean], rtol=0, atol=1e-9, equal_nan=True), (k, col)
            n_diff += int(np.sum(~np.isclose(a0[~clean], a1[~clean], rtol=0, atol=1e-9, equal_nan=True)))
    assert sum(pt["info"]["n_filled"].values()) > 0 and pt["info"]["n_unfilled"] == 0
    assert n_diff >= 0                                                    # 결측 셀의 값은 달라질 수 있다(크기는 unit.json 의 notes 에 남는다)


def test_u_f1n_flag_and_sensitivity_rows(stub):
    # n = 0 의 F1n 은 F1a 의 예측을 나눠 쓴다. F1k 의 적합만 실패했을 때 F1n 행에 실패 표지가 붙지 않는다
    FAIL_METHODS.add("F1k")
    try:
        rows, st, stats, _, _ = run_x("x1", extra=["--n-grid", "0,10"])
    finally:
        FAIL_METHODS.clear()
    r0 = [r for r in rows if r["n"] == 0 and r["seed"] == 0]
    f1n = [r for r in r0 if r["method"] == "F1n"]; f1k = [r for r in r0 if r["method"] == "F1k"]; f1a = [r for r in r0 if r["method"] == "F1a"]
    assert len(f1n) == len(f1k) == len(f1a) == 1
    assert f1k[0]["fit_flag"] == "fail" and f1k[0]["n_nonfinite"] > 0 and f1a[0]["fit_flag"] == "" and f1n[0]["fit_flag"] == ""
    assert np.isclose(f1n[0]["rmse_cm"], f1a[0]["rmse_cm"]) and stats["status"] == "partial"
    # 결측 대체 민감도 행(x9 축): 해석식 기준선은 n = 0 에 한 번, F1k@tr 은 n {0, 10} 에 적합한다
    rows, st, stats, _, tr = run_x("x9", extra=["--n-grid", "0,3,10,all"])
    assert stats["status"] == "ok"
    for m in ("P0@ku_tr", "P0@ed_tr", "B:ku_raw_tr", "B:ed_raw_tr", "B:ens_tr"):
        assert ((m, "none", "1", "cell", 0, 0, -1, 0.0) in st), m
    fk = sorted({(k[4], k[5]) for k in st.keys if k[0] == "F1k@tr"})
    assert fk == [(0, 0), (10, 0), (10, 1)], "F1k@tr 은 n = 0 과 n = 10 에만 있다"
    e = [e_ for e_ in tr if e_["method"] == "F1k@tr" and e_["n"] == 10 and e_["draw"] == 0][0]
    c, ext = make_unit()
    sel = H.draw_cells("T", "x", 1, 10, 0, NA)
    assert e["Xtr"].shape == (NS + 10, D_FEAT + 2)
    assert np.allclose(e["Xtr"][:NS, D_FEAT], ext["phys_tr"]["src"]["p4_ku"], equal_nan=True)
    assert np.allclose(e["Xtr"][NS:, D_FEAT + 1], ext["phys_tr"]["A"]["p2_edaphic"][sel])
    assert set(stats["notes"]["phys_tr_ku"]) >= {"rho", "c0", "n_changed_B", "max_abs_diff_B"} and stats["notes"]["phys_tr_ku"]["n_changed_B"] > 0
    assert abs(stats["notes"]["phys_tr_ed"]["max_abs_diff_B"] - 0.05) < 1e-6


# ================================================================ 학습을 하는 시험
@HEAVY
def test_A_base_matches_h40_catboost():
    stx, _ = _same_base()
    assert stx["status"] == "ok" and stx["n_nonfinite_keys"] == 0


@HEAVY
def test_B_unselected_labels_never_used_catboost():
    _perturb_invariant(["x1", "x2", "x9", "x9f", "x5"])


@HEAVY
def test_C_all_cpu_axes_run():
    for axis in ("base", "x1", "x2", "x9", "x9f", "n3", "prox", "x5", "ridge", "n4", "n4a"):
        rows, st, stats, cells, _ = run_x(axis, make_unit(flags=True), extra=["--n-grid", "0,10,all"], trace=False)
        assert stats["status"] == "ok" and stats["n_nonfinite_keys"] == 0 and len(rows) > 0, (axis, stats["errors"])
        assert (cells is not None) == (axis in ("base", "x1"))
        if axis == "x5":
            sh = stats["shap_rows"]
            assert {r["method"] for r in sh} == {"D0", "F1k", "R1"} and {r["n"] for r in sh} == {0, -1}
            assert len([r for r in sh if r["method"] == "F1k" and r["n"] == 0]) == D_FEAT + 2
        if axis == "n3":
            assert {k[1] for k in st.keys} == {"none", "catboost", "rf", "catboost_tuned"}


@HEAVY
def test_D_real_unit_resume_and_summarize(tmp_path):
    if not (ROOT / "data" / "processed" / "fidelity_base_v3.csv").exists():
        pytest.skip("실자료 없음")
    argv = ["--part", "cpu", "--axes", "base,x1,prox", "--targets", "Russia_W", "--splits", "1", "--n-grid", "0,10", "--draws", "1", "--seeds", "1",
            "--workers", "0", "--threads", "2", "--out-dir", str(tmp_path), "--tag", "t", "--no-summarize", "--allow-local", "--nboot", "200"]
    r1 = X.main(argv)
    assert sorted(r1["executed"]) == [("base", "Russia_W", "x", 1), ("prox", "Russia_W", "x", 1), ("x1", "Russia_W", "x", 1)] and not r1["resumed"]
    sh = tmp_path / "shards"
    assert (sh / "tb__cpu__Russia_W__x__s1_unit.json").exists() and (sh / "tb__cpu__Russia_W__x__s1_cells.npz").exists()
    assert not list(sh.glob("*.tmp*")), "임시 파일이 남았다"
    with np.load(sh / "tb__cpu__Russia_W__x__s1_cells.npz", allow_pickle=False) as z:
        assert z["P"].dtype == np.float32 and z["P"].shape == (len(z["keys"]), len(z["loc_id"])) and len(z["d_src_km"]) == len(z["y"])
    r2 = X.main(argv + ["--resume"])
    assert r2["executed"] == [] and len(r2["resumed"]) == 3
    r3 = X.main(argv + ["--resume", "--n-grid", "0"])                     # 설정이 다르면 건너뛰지 않는다
    assert len(r3["executed"]) == 3
    X.main(argv)
    out = X.summarize(X.parse_args(argv))
    assert out is not None and len(out["curve"]) > 0 and len(out["tests"]) > 0 and len(out["floor"]) == 6
    assert {"verdict4_p0", "delta_pct_p0", "small_effect"} <= set(out["curve"].columns)
    for f in ("curve", "tests", "gate", "lg_aux", "region_inference", "splitdist", "conformal", "distance", "shap", "floor", "timing", "failed"):
        assert (tmp_path / f"t_{f}.csv").exists(), f
    assert (tmp_path / "t_meta.json").exists()


@HEAVY
def test_E_pool_path_workers2(tmp_path):
    """프로세스 풀 경로(spawn 워커 2개). 워커가 모듈을 다시 읽고 작업 단위를 실행한다."""
    if not (ROOT / "data" / "processed" / "fidelity_base_v3.csv").exists():
        pytest.skip("실자료 없음")
    argv = ["--part", "cpu", "--axes", "base", "--targets", "Russia_W,Russia_E", "--splits", "1", "--n-grid", "0,10", "--draws", "1", "--seeds", "1",
            "--workers", "2", "--threads", "2", "--out-dir", str(tmp_path), "--tag", "t", "--no-summarize", "--allow-local", "--nboot", "200"]
    r = X.main(argv)
    assert sorted(r["executed"]) == [("base", "Russia_E", "x", 1), ("base", "Russia_W", "x", 1)] and not r["n_fail"]
    assert not list((tmp_path / "shards").glob("*.tmp*")), "임시 파일이 남았다"
    out = X.summarize(X.parse_args(argv))                                 # 곡선 계산도 워커 2개에서 돈다(대상·모드 2개)
    assert out is not None and len(out["curve"]) > 0 and out["meta"]["curve_failed"] == [] and out["meta"]["summarize_workers"] == 2
    assert set(out["curve"].target) == {"Russia_W", "Russia_E"} and int(out["tests"].nboot.iloc[0]) == 200
