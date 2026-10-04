"""scripts/3_deep_learning/x_placement_policy.py·x_placement_deepsets.py 단위 시험(계획 2.4 '구현·시험' (a)–(h)와 누설·세기 시험).

합성 자료 시험(라벨 값을 화면에 쓰지 않는다)
(a) S8 계열이 A 색인 안의 서로 다른 정수를 내고 결정적·중첩적이다(일정 (10, 40) 의 앞 10 개 = 일정 (10) 의 순서).
(b) power_alloc: 합 = n, 블록 셀 수 이하, γ 0 이고 n < 블록 수이면 β 의 앞 n 블록에 1개씩, 상한 재배분.
(c) 블록 안 선택이 작은 문제에서 전수 계산의 최대 최소 거리와 같고, 라벨 없는 블록의 첫 셀은 블록 중심 근접 셀이다.
(d) 후보 집합(학습 자료 라벨 집합 200개) 생성 재현과 구성(40·30·30 %, 섭동 20–50 %). S2 순환 순서 = h40.draw_blocks.
(e) 정책 특징(비가중 AOA 포함)이 라벨을 읽지 않는다(라벨 속성에 접근하면 예외를 내는 대리 문맥으로 추적). 증분 특징 = 처음부터 계산한 특징.
(f) 개발·시험 분리와 교차검증 접기(AL-k 와 알래스카 x 함께 제외)에 과제 누설이 없다. S9-GBM 접기 기록도 같다.
(g) DeepSets 같은 seed 두 번 실행의 학습·선택 일치(CPU 결정적 설정).
(h) select_algorithm_p 가 알래스카 계열 조각만 열고(비알래스카 조각은 깨진 파일이어도 돈다) 비알래스카 행을 출력하지 않는다.
    제외 규칙(2단 CI 하한 > 0), 점수, 동률 순서(S8a 먼저), 모두 제외 → S1.
(i) 누설 시험(1절): 선택 밖 A 라벨과 B 라벨을 바꿔도 xd_t(S8a + S8w·S8r), xd_r, xd_learn, xd_test 단위의 모든 예측이 같고, 선택 라벨 하나를
    바꾸면 달라진다(음성 대조).
(j) 재현: xd_t 의 S1·S2·S4 저장소가 h54.T8Unit(WF8)과, xd_r 의 S1·S2·S4 가 h54.RUnit(WF2)과 같은 키에서 블록 SSE 가 같다.
(k) 계수 변형: 설계 가중 E_w 의 손 계산, 블록 무작위 효과 계수(블록당 라벨 1개면 E_ls, τ² 큰 경우 블록 평균 쪽).
(l) 탐욕 정책: 중첩, 단계 크기(1 → 5 → 10 …), 결정성, 정책 seed 는 첫 셀만 바꾼다.
(m) [실제 자료, 세기 범주] --count-only 가 라벨 값에서 나온 통계를 출력하지 않는다(출력 제한에 걸린 줄도 없다).
(n) 조각 이름·봉인: 합성 조각으로 XD-alg 집계(Holm m 20)가 봉인 폴더에만 쓰고 화면에는 행 수와 sha256 만 쓴다.
(o) XD-4 서술 지표 v_w(분할·추출 쌍 평균)와 과제 k/5, 계열 j/3 세기(두 가중 모두 음수일 때만).
(p) 색인 파일의 sha256·A 지문 대조와 시험 팔 단위(S9·S1·AP 키).
(q) S9* 동결: 교차검증 효용이 작은 변형, 동률은 S9-GBM, 화면에 효용 값을 쓰지 않는다.
(r) 동결 모듈 4개의 sha256 이 계획 머리말 값과 같다.
(s) S9-GBM 학습과 정책 점수 함수.
(t) 재현 관문(같은 SSE 면 통과, 다르면 실패, 화면에는 수만)과 마감 기본값(P_default = S8a), 등록 밖 후보 거부.
(v) XD-5: 계열 하나 제외 접기(검증 계열의 과제가 학습에 없다)와 계열 안 과제 하나 제외 접기, 오라클 대비 후회(U 정의 = 학습 자료와 같다).
(w) XC 호환: placement_order 의 S8 계열이 algorithm_p_order 와 같고, 블록 번호는 문자열 정렬(음수·자릿수 혼합 번호에서도 결정적)이다.
(u) 명령행 거부: 동결 기록 없는 xd_test, XD-5 전 개발 과제 밖 학습 자료, 허용 표지 없는 본 실행·10,000회 집계. --shard, 옵션 제거.
(x)–(ad) 2026-10-04 검토 반영 시험(스모크 대상, XD-4 AP 순서, 색인 과제·시작 전 확인, 문장 갈래, XD-4 화면 출력, 로컬 자원, 동결 커밋).
(ae) S9-DS 장치: 허용 GPU 0–4 하나만(nvidia-smi 표 대리, CUDA 미생성), CUBLAS_WORKSPACE_CONFIG 덮어쓰기, CUDA_DEVICE_ORDER, --mem-wait.
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 prlimit --data=10737418240 taskset -c <코어 4개> python3 -m pytest -q tests/test_x_xd.py
(메모리 상한은 prlimit --data 로 건다. prlimit --as 아래에서는 CatBoost 1.2.10 이 가끔 특징 중요도 계산에서 멈춘다)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
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
XP = _load("x_placement_policy", "scripts/3_deep_learning/x_placement_policy.py")
H, X, W, H4 = XB.H, XB.X, XB.W, XB.H4
LO = XB.LO

REAL = pytest.mark.skipif(not (ROOT / "data/processed/fidelity_base_v3.csv").exists(), reason="실제 자료(v3)가 없다")
NF = len(XB.FEATS)


# ---------------------------------------------------------------- 합성 자료
def args(tmp=None, extra=()):
    av = ["--exp", "xd_t", "--threads", "1", "--cb-iters", "10", "--seeds", "1", "--draws-cap", "2", "--nboot", "200", "--xdt-grid", "10,20",
          "--xdr-grid", "10,20", "--learn-grid", "10", "--test-grid", "10,20"]
    if tmp is not None:
        av += ["--out-dir", str(Path(tmp) / XP.EXP_NAME)]
    a = XP.parse(av + list(extra))
    if tmp is not None:
        a.XD_ALLOWED = [Path(tmp)]
    return a


def make_tctx(seed=0, split=1, target="T", nA=96, nbA=12, nB=60, nbB=10, mode="x"):
    """합성 전이 문맥(h40.Ctx). 원천 300행, A 셀 nA 개(블록 크기 불균등), B 셀 nB 개(분할마다 다른 블록 번호)."""
    rng = np.random.RandomState(seed)
    ns = 300
    Xs = rng.randn(ns, NF); ss = rng.uniform(20, 40, ns); ys = 1.6 * ss + 3 * Xs[:, 1] + 2 * rng.randn(ns)
    XA = rng.randn(nA, NF); sA = rng.uniform(20, 40, nA)
    w = np.arange(1, nbA + 1, dtype=float); w /= w.sum()
    blkA = np.sort(rng.choice(nbA, nA, p=w))
    blkA[:nbA] = np.arange(nbA)                                            # 모든 블록에 셀이 있다
    blkA = np.sort(blkA)
    yA = (1.2 + 0.03 * blkA) * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    XBm = rng.randn(nB, NF); sB = rng.uniform(20, 40, nB); yB = 1.3 * sB + 3 * XBm[:, 1] + 2 * rng.randn(nB)
    blkB = 500 + 100 * split + np.repeat(np.arange(nbB), int(np.ceil(nB / nbB)))[:nB]
    return H.Ctx(target, mode, split, target, Xs, ys, ss, np.repeat(["r1", "r2"], ns // 2), XA, yA, sA, blkA, XBm, yB, sB, blkB,
                 meta=dict(dup_of=-1, valid=True, n_valid_splits=1, n_unique_splits=1))


def make_rctx(seed=0, split=1, target="T", nA=120, nbA=10, nB=80, nbB=10):
    """합성 지역 내 문맥(h54.RCtx)."""
    rng = np.random.RandomState(seed)
    XA = rng.randn(nA, NF).astype(np.float32); XBm = rng.randn(nB, NF).astype(np.float32)
    sA, sB = rng.uniform(20, 40, nA), rng.uniform(20, 40, nB)
    blkA = np.repeat(np.arange(nbA), int(np.ceil(nA / nbA)))[:nA]
    blkB = 1000 + 100 * split + np.repeat(np.arange(nbB), int(np.ceil(nB / nbB)))[:nB]
    yA = (1.3 + 0.04 * blkA) * sA + 3 * XA[:, 1] + 2 * rng.randn(nA); yB = 1.4 * sB + 3 * XBm[:, 1] + 2 * rng.randn(nB)
    latA, lonA = 65 + 0.05 * rng.rand(nA), -150 + 0.05 * rng.rand(nA)
    latB, lonB = 66 + 0.05 * rng.rand(nB), -150 + 0.05 * rng.rand(nB)
    return W.RCtx(target, split, target, XA, yA, sA, blkA, latA, lonA, XBm, yB, sB, blkB, latB, lonB, 1.6, src_X=rng.randn(200, NF),
                  meta=dict(dup_of=-1, valid=True, n_valid_splits=1, n_unique_splits=1))


def synth_Z(seed=0, N=120, B=12):
    rng = np.random.RandomState(seed)
    Z = rng.randn(N, 5)
    w = np.arange(1, B + 1, dtype=float); w /= w.sum()
    blk = rng.choice(B, N, p=w); blk[:B] = np.arange(B)
    return Z, 100 + blk


# ---------------------------------------------------------------- (a) S8 계열
@pytest.mark.parametrize("name", list(XP.S8_SPECS))
def test_a_s8_valid_deterministic_nested(name):
    Z, blk = synth_Z(1)
    g, f = XP.S8_SPECS[name]
    o1 = XP.algorithm_p_order(Z, blk, (10, 40), g, seed=7, first=f)
    o2 = XP.algorithm_p_order(Z, blk, (10, 40), g, seed=7, first=f)
    assert np.array_equal(o1, o2) and len(o1) == 40 and len(set(o1.tolist())) == 40
    assert o1.min() >= 0 and o1.max() < len(Z) and o1.dtype.kind == "i"
    o10 = XP.algorithm_p_order(Z, blk, (10,), g, seed=7, first=f)
    assert np.array_equal(o10, o1[:10]), "일정 (10, 40) 의 앞 10 개가 일정 (10) 과 다르다(중첩 위반)"
    o3 = XP.algorithm_p_order(Z, blk, (10, 40), g, seed=8, first=f)
    assert len(o3) == 40
    # 단위 안 전략 집합: 정렬, 중첩
    a = args()
    c = make_tctx(3)
    U = XP.XDTUnit(a, c, "T", name)
    sets = U.xd_sets(name, [10, 20], 1)
    assert set(sets[10].tolist()) <= set(sets[20].tolist()) and np.all(np.diff(sets[20]) > 0)
    # placement_order(XC 용) = algorithm_p_order
    assert np.array_equal(XP.placement_order(name, Z, blk, (10, 40), 7), o1)


def test_a2_gamma0_one_per_block_then_round():
    Z, blk = synth_Z(2, N=200, B=12)
    o = XP.algorithm_p_order(Z, blk, (10, 40), 0.0, seed=3)
    b10 = blk[o[:10]]
    assert len(set(b10.tolist())) == 10, "γ 0, n < 블록 수: 앞 10 개가 서로 다른 블록이어야 한다"
    cnt = pd.Series(blk[o[:40]]).value_counts()
    ub, nb = np.unique(blk, return_counts=True)
    q = XP.power_alloc(nb, 40, 0.0)
    assert int(cnt.sum()) == 40 and all(cnt.get(b, 0) <= n for b, n in zip(ub, nb))
    assert sorted(cnt.values.tolist()) == sorted(q[q > 0].tolist())


# ---------------------------------------------------------------- (b) power_alloc
def test_b_power_alloc():
    rng = np.random.RandomState(0)
    for _ in range(200):
        B = rng.randint(1, 15)
        Nb = rng.randint(1, 30, B)
        n = rng.randint(0, Nb.sum() + 5)
        g = rng.choice([0.0, 0.5, 1.0])
        rank = rng.permutation(B)
        q = XP.power_alloc(Nb, n, g, rank)
        assert q.sum() == min(n, Nb.sum()) and np.all(q <= Nb) and np.all(q >= 0)
    Nb = np.array([5, 5, 5, 5, 5])
    rank = np.array([3, 0, 4, 1, 2])
    q = XP.power_alloc(Nb, 3, 0.0, rank)
    assert q.tolist() == [0, 1, 0, 1, 1], "γ 0, n < B: β 순위 앞 3 블록에 1개씩"
    assert XP.power_alloc([1, 1, 10], 6, 0.0).tolist() == [1, 1, 4], "상한 재배분"
    q = XP.power_alloc([10, 20, 30, 40], 50, 1.0)
    assert q.tolist() == [5, 10, 15, 20]
    q = XP.power_alloc([4, 16], 6, 0.5)
    assert q.tolist() == [2, 4]


# ---------------------------------------------------------------- (c) 블록 안 선택 = 전수 최대 최소
@pytest.mark.parametrize("first", ["center", "farthest"])
def test_c_within_block_bruteforce(first):
    rng = np.random.RandomState(5)
    Z = rng.randn(18, 3)
    blk = np.array([0] * 9 + [1] * 6 + [2] * 3)
    o = XP.algorithm_p_order(Z, blk, (4, 12), 0.0, seed=1, first=first)
    ub = np.unique(blk)
    mu = {b: Z[blk == b].mean(0) for b in ub}
    for k, c in enumerate(o):
        prev = o[:k]
        b = blk[c]
        in_blk_prev = [p for p in prev if blk[p] == b]
        cand = [i for i in np.where(blk == b)[0] if i not in set(prev.tolist())]
        if not in_blk_prev and (first == "center" or k == 0):
            best = min(cand, key=lambda i: (np.linalg.norm(Z[i] - mu[b]), i))
        else:
            best = max(cand, key=lambda i: (min(np.linalg.norm(Z[i] - Z[p]) for p in prev), -i))
        assert c == best, f"{k} 번째 선택이 전수 계산과 다르다"
    # 블록 하나 문제: 탐욕 최대 최소의 각 단계가 전수 비교의 최댓값
    Zs = rng.randn(10, 2)
    o = XP.algorithm_p_order(Zs, np.zeros(10, int), (6,), 0.0, seed=0, first=first)
    for k in range(1, 6):
        prev = o[:k]
        vals = {i: min(np.linalg.norm(Zs[i] - Zs[p]) for p in prev) for i in range(10) if i not in set(prev.tolist())}
        assert np.isclose(vals[o[k]], max(vals.values()))


# ---------------------------------------------------------------- (d) 후보 집합 생성 재현
def test_d_learning_sets_reproducible():
    Z, blk = synth_Z(4, N=150, B=10)
    s1 = XP.learning_sets(Z, blk, "AL-1", "x", 1, 20, 200)
    s2 = XP.learning_sets(Z, blk, "AL-1", "x", 1, 20, 200)
    assert len(s1) == 200
    assert all(k1 == k2 and g1 == g2 and np.array_equal(a1, a2) for (k1, g1, a1), (k2, g2, a2) in zip(s1, s2))
    kinds = [k for k, _, _ in s1]
    assert kinds.count("S1") == 80 and sum(k in XP.STRUCT_GENS for k in kinds) == 60 and sum(k.startswith("pert:") for k in kinds) == 60
    for k, _, a in s1:
        assert len(a) == 20 and len(set(a.tolist())) == 20 and a.min() >= 0 and a.max() < 150 and np.all(np.diff(a) > 0)
    struct = [a for k, _, a in s1 if k in XP.STRUCT_GENS]
    for j, (k, _, a) in enumerate([t for t in s1 if t[0].startswith("pert:")]):
        changed = len(set(a.tolist()) - set(struct[j % 60].tolist()))
        assert round(0.2 * 20) <= changed <= round(0.5 * 20)
    k2, g2, a2 = [t for t in s1 if t[0] == "S2"][0]
    assert np.array_equal(a2, np.sort(XP.round_robin_order(blk, g2)[:20]))
    other = XP.learning_sets(Z, blk, "AL-1", "x", 2, 20, 200)
    assert not all(np.array_equal(a, b) for (_, _, a), (_, _, b) in zip(s1, other)), "분할이 다르면 집합이 달라야 한다"


def test_d2_round_robin_equals_draw_blocks():
    _, blk = synth_Z(6, N=80, B=9)
    for n in (1, 5, 9, 10, 33, 79):
        for d in range(3):
            rs = XB.seed_of("T", "x", 2, n, d, "block")
            assert np.array_equal(np.sort(XP.round_robin_order(blk, rs)[:n]), H.draw_blocks("T", "x", 2, n, d, blk))


# ---------------------------------------------------------------- (e) 정책 특징은 라벨을 읽지 않는다
class LabelTrap:
    """공변량 속성만 내주고 라벨 속성(yA, yB, y_src, zA, r0_src)에 접근하면 예외를 낸다(추적)."""

    def __init__(self, c):
        self._c = c
        self.accessed = []

    def __getattr__(self, k):
        if k in ("yA", "yB", "y_src", "zA", "r0_src", "prior"):
            raise AssertionError(f"정책 특징이 라벨 속성 {k} 를 읽었다")
        self.__dict__.setdefault("accessed", []).append(k)
        return getattr(self._c, k)


def test_e_features_label_free_and_batch():
    c = make_tctx(7)
    trap = LabelTrap(c)
    tf_ = XP.TaskFeat.from_ctx(trap)
    assert set(trap.accessed) <= {"X_src", "src_X", "target", "mode", "split", "XA", "sA", "blkA"}
    st = W.standardize_fit(c.X_src)
    di = W.aoa_di(W.standardize(c.XA, st), W.standardize(c.X_src, st), XB.seed_of("wf-aoa", c.target, c.split))
    assert np.allclose(tf_.DI, di), "비가중 AOA 는 h54.aoa_di(원천 통계)와 같아야 한다"
    rng = np.random.RandomState(0)
    L = rng.choice(tf_.N, 9, replace=False)
    C = np.setdiff1d(np.arange(tf_.N), L)[:15]
    Fb = tf_.features_batch(L, C)
    Fs = np.vstack([tf_.features(np.r_[L, cc]) for cc in C])
    assert Fb.shape == (15, len(XP.FEATS)) and np.allclose(Fb, Fs, rtol=1e-9, atol=1e-9)
    # 라벨을 바꿔도 특징이 같다
    c2 = XB.perturb_labels(c, [], seed=3)
    tf2 = XP.TaskFeat.from_ctx(c2)
    assert np.array_equal(tf_.features(L), tf2.features(L)) and np.array_equal(tf_.element_static(), tf2.element_static())
    E = XP.elements_of(tf_.element_static(), L)
    assert E.shape == (9, len(XP.EL_FEATS))


# ---------------------------------------------------------------- (f) 개발·시험 분리와 접기
def test_f_cv_folds_no_task_leak():
    folds = XP.cv_folds(XP.DEV_TASKS)
    assert [f["val"] for f in folds] == list(XP.AL_SUBS)
    for f in folds:
        assert f["val"] not in f["train"] and "Alaska" not in f["train"] and f["val"] != "Alaska"
        assert set(f["train"]) | {f["val"], "Alaska"} == set(XP.DEV_TASKS)
        assert not set(f["train"]) & set(XP.TEST_TASKS)
    assert XP.assert_dev_test_disjoint()
    with pytest.raises(AssertionError):
        XP.assert_dev_test_disjoint(XP.DEV_TASKS, ("LE-1", "AL-2"))
    df = synth_learning_df()
    _, rep, it, log = XP.train_gbm(df, threads=1, iters_grid=(10, 20))
    for r in log:
        assert r["val"] not in r["train"] and "Alaska" not in r["train"]
    assert it in (10, 20) and len(rep) == 2


def synth_learning_df(tasks=("Alaska", "AL-1", "AL-2", "AL-3"), n_sets=20, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for t in tasks:
        for sp in (1, 2):
            for n in (10, 20):
                F = rng.randn(n_sets, len(XP.FEATS))
                U = 0.3 * F[:, 1] - 0.2 * F[:, 4] + 0.05 * rng.randn(n_sets)
                for j in range(n_sets):
                    rows.append(dict(task=t, mode="x", split=sp, n=n, j=j, kind="S1", U_cell=U[j], U_beq=U[j], U=U[j],
                                     **{f: float(v) for f, v in zip(XP.FEATS, F[j])}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (g) DeepSets 결정성
def test_g_deepsets_same_seed_same_selection(monkeypatch):
    DS = _load("x_placement_deepsets", "scripts/3_deep_learning/x_placement_deepsets.py")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")                 # 환경에 다른 값이 있어도 계획 값(:4096:8)으로 덮어쓴다
    info = DS.setup_device("cpu", None, 0, threads=2)
    assert info["device"] == "cpu" and info["cublas"] == ":4096:8" and os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert info["deterministic"] is True and info["gpu"] is None and info["gpu_allowed"] == [0, 1, 2, 3, 4]
    rng = np.random.RandomState(0)
    groups = []
    for t in ("Alaska", "AL-1", "AL-2"):
        for n in (5, 8):
            X_ = rng.randn(12, n, len(XP.EL_FEATS)).astype(np.float32)
            U = X_[:, :, 0].mean(1) + 0.1 * rng.randn(12)
            groups.append(dict(task=t, split=1, n=n, X=X_, C=np.tile(np.r_[rng.rand(5), n], (12, 1)).astype(np.float32), U=U, j=np.arange(12)))
    cv1 = DS.cv_select_epoch(groups, seeds=(0,), epochs=2, device="cpu")
    cv2 = DS.cv_select_epoch(groups, seeds=(0,), epochs=2, device="cpu")
    assert cv1["curve"] == cv2["curve"] and cv1["best_epoch"] == cv2["best_epoch"]
    b1 = DS.fit_final(groups, 2, seeds=(0, 1), device="cpu")
    b2 = DS.fit_final(groups, 2, seeds=(0, 1), device="cpu")
    for m1, m2 in zip(b1["models"], b2["models"]):
        for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
            assert k1 == k2 and np.array_equal(v1.numpy(), v2.numpy())
    tf_ = XP.TaskFeat.from_ctx(make_tctx(9, nA=80))
    o1 = XP.greedy_policy(tf_, DS.ds_score_fn(b1), 15, 0)
    o2 = XP.greedy_policy(tf_, DS.ds_score_fn(b2), 15, 0)
    assert np.array_equal(o1, o2) and len(set(o1.tolist())) == 15


# ---------------------------------------------------------------- (h) Algorithm P 선택은 알래스카 계열만
def _store(name, split, levels, nb=10, draws=3, seeds=(0, 1), ns=(10, 40), rng_seed=0, noise=0.05):
    rng = np.random.RandomState(rng_seed + split)
    ncell = rng.randint(4, 12, nb)
    st = H4.BlockStore(name, split, np.repeat([f"s{split}b{j}" for j in range(nb)], ncell))
    cnt = st.ncell
    st.add_sse(H.P0_KEY, cnt * 30.0 ** 2, cnt)
    base = {}
    for n in ns:
        for d in range(draws):
            e = 20.0 + rng.gamma(2.0, noise, nb)
            base[(n, d)] = e
            for pl, lev in levels.items():
                for s in seeds:
                    st.add_sse(("R1", LO, "1", pl, int(n), d, int(s), 0.25), cnt * (e + (lev - 20.0)) ** 2, cnt)
    return st


def _write_synth_shard(d, tag, target, mode, split, st, variant="S1"):
    cfg = XB.make_unit_cfg("xd_t", variant, "synthetic")
    XB.write_shard(d, tag, target, mode, split, [], [st], cfg, unit=dict(dup_of=-1, valid=True), variant=variant, expected=[1, 2, 3],
                   allowed=[d.parent.parent])


def test_h_select_alaska_family_only(tmp_path, capsys):
    root = tmp_path / XP.EXP_NAME
    d = root / "shards"
    lev = dict(S1=20.0, S2=25.0, S4=19.5, S8a=19.0, S8b=19.5, S8c=19.0, S8p=21.0)
    lev_r = dict(lev, S4=23.0)                                             # S4 는 지역 내(r) 행에서만 나쁘다
    for sp in (1, 2, 3):
        for t in ("Alaska", "AL-1", "AL-2"):
            _write_synth_shard(d, "xdt", t, "x", sp, _store(f"{t}|x", sp, lev, rng_seed={"Alaska": 0, "AL-1": 10, "AL-2": 20}[t]))
        _write_synth_shard(d, "xdr", "Alaska", "r", sp, _store("Alaska|r", sp, lev_r, ns=(20, 50)))
    for v in ("S1", "S8a"):                                                # 비알래스카 조각: 깨진 파일(열면 실패한다)
        for ext in ("_runs.csv", "_blocksse.npz", "_unit.json"):
            (d / f"xdt__cpu__Lena__x__s1__{v}{ext}").write_bytes(b"\x00broken")
            (d / f"xdt__cpu__Tibet_LGD__x__s1__{v}{ext}").write_bytes(b"\x00broken")
            (d / f"xdr__cpu__Lena__r__s1__{v}{ext}").write_bytes(b"\x00broken")       # 지역 내 비알래스카(xd_r 대상)
            (d / f"xdr__cpu__Canada__r__s1__{v}{ext}").write_bytes(b"\x00broken")
    # 같은 SSE 로 S8a 와 S8c 동률을 만든다(같은 수준과 같은 잡음)
    res = XP.select_algorithm_p(d, "xdt", "xdr", root / "selection", nboot=300, allowed=[tmp_path])
    out = capsys.readouterr().out
    assert res["chosen"] == "S8a", "S8a 와 S8c 가 동률이면 S8a(순서 규칙)"
    assert res["candidates"]["S2"]["excluded"] and res["candidates"]["S8p"]["excluded"]
    assert not res["candidates"]["S8a"]["excluded"] and not res["candidates"]["S8c"]["excluded"]
    s4 = res["candidates"]["S4"]
    assert s4["excluded"] and s4["excl_rows"] and all("|r" in v for v in s4["excl_rows"]), "S4 는 지역 내 행만으로 제외되어야 한다"
    assert np.isfinite(s4["score"]), "점수는 전이 행에서만 계산한다(제외와 별개)"
    assert abs(res["candidates"]["S8a"]["score"] - res["candidates"]["S8c"]["score"]) < XP.AP_TIE_TOL
    assert res["candidates"]["S8a"]["score"] < res["candidates"]["S4"]["score"]
    files = [root / "selection" / "algorithm_p_selection.json", root / "selection" / "algorithm_p_alaska_table.csv"]
    for f in files:
        txt = f.read_text()
        assert "Lena" not in txt and "Tibet" not in txt and "Russia" not in txt and "Canada" not in txt
        assert XP.verify_sidecar(f)
    assert "Lena" not in out and "Tibet" not in out and XB.FORBIDDEN_OUT.search(out) is None
    assert set(res["stores_read"]) == {"Alaska|x", "AL-1|x", "AL-2|x", "Alaska|r"}
    sel = XP.load_selection(files[0])
    assert sel["chosen"] == "S8a" and sel["spec"]["gamma"] == 0.0 and sel["spec"]["schedule_fixed"] == [10, 40, 160]
    with pytest.raises(SystemExit):                                         # 봉인 조각 폴더(sealed/shards)는 선택이 읽지 않는다
        XP.select_algorithm_p(root / "sealed" / "shards", "xdt", "xdr", root / "selection", nboot=300, allowed=[tmp_path])
    # 모든 후보 제외 → S1
    tab = pd.DataFrame([dict(arm="x", target="Alaska", store="Alaska|x", n=10, cand=c, rel_cell=0.1, rel_beq=0.1, excl_row=True)
                        for c in XP.AP_CANDIDATES])
    assert XP.choose_algorithm_p(tab)["chosen"] == "S1"
    tab.loc[tab.cand == "S4", "excl_row"] = False
    assert XP.choose_algorithm_p(tab)["chosen"] == "S4"
    # 점수 = 대상별(n 평균 뒤 나쁜 가중) 단순 평균
    rows = [dict(arm="x", target=t, store=f"{t}|x", n=n, cand="S8b", rel_cell=rc, rel_beq=rb, excl_row=False)
            for t, n, rc, rb in (("Alaska", 10, -0.1, 0.0), ("Alaska", 40, -0.1, -0.2), ("AL-1", 10, -0.3, -0.1))]
    r = XP.choose_algorithm_p(pd.DataFrame(rows))
    assert np.isclose(r["candidates"]["S8b"]["score"], np.mean([max(-0.1, -0.1), max(-0.3, -0.1)]))


# ---------------------------------------------------------------- (i) 누설 시험
def _neg_control(run_fn, c, keep):
    with XB.h54_trace() as (p1, _):
        run_fn(c)
    c3 = XB.perturb_labels(c, np.arange(len(c.yA)), seed=1)
    c3.yA = np.array(c.yA, float, copy=True)
    c3.yA[int(keep[0])] += 50.0
    with XB.h54_trace() as (p3, _):
        run_fn(c3)
    return XB.compare_predictions(p1, p3)


def test_i_leakage_xd_t_s8a_and_coef_variants():
    a = args()
    c = make_tctx(11)

    def run(cc):
        XP.XDTUnit(a, cc, "T", "S8a").run()
    r = XB.leakage_invariance(run, c)
    assert r["ok"] and r["n_keep"] < r["n_A"], r
    with XB.h54_trace() as (p, tr):
        run(c)
    assert any(k[7] == "S8w" for k in p) and any(k[7] == "S8r" for k in p)
    keep = XB.selected_union(tr)
    assert not _neg_control(run, c, keep)["ok"], "선택 라벨을 바꾸면 예측이 달라져야 한다(음성 대조)"


def test_i2_leakage_xd_r_learn_test():
    a = args()
    c = make_rctx(12)
    r = XB.leakage_invariance(lambda cc: XP.XDRUnit(a, cc, "S8b").run(), c)
    assert r["ok"] and r["n_keep"] < r["n_A"], r
    r = XB.leakage_invariance(lambda cc: XP.XDRUnit(a, cc, "S8a").run(), c)
    assert r["ok"], r
    ct = make_tctx(13, nA=80)
    r = XB.leakage_invariance(lambda cc: XP.XDLUnit(a, cc, "T", 10, n_sets=10).run(), ct)
    assert r["ok"] and r["n_keep"] < r["n_A"], r
    tf_ = XP.TaskFeat.from_ctx(ct)
    ent = {p: dict(nA=tf_.N, a_fp=tf_.a_fp, order=XP.greedy_policy(tf_, lambda t, L, C: -t.features_batch(L, C)[:, 1], 20, p).tolist())
           for p in (0, 1)}
    r = XB.leakage_invariance(lambda cc: XP.XDXUnit(a, cc, "T", ent, "S8a").run(), ct)
    assert r["ok"] and r["n_keep"] < r["n_A"], r


# ---------------------------------------------------------------- (j) 재현: h54 와 같은 계산
def _same_store(st1, st2, placements):
    keys = [k for k in st2.keys if k[3] in placements]
    assert keys, "비교할 키가 없다"
    for k in keys:
        assert k in st1, f"키 {k} 가 없다"
        s1, c1 = st1.get(k); s2, c2 = st2.get(k)
        assert np.array_equal(c1, c2) and np.array_equal(s1, s2), f"키 {k} 의 블록 SSE 가 다르다"
    return len(keys)


@pytest.mark.parametrize("strat", ["S1", "S2", "S4"])
def test_j_reproduces_wf8_and_wf2(strat):
    a = args()
    a.G["wf8"] = list(a.G["xd_t"]); a.G["wf2"] = list(a.G["xd_r"])
    c = make_tctx(14)
    U1 = XP.XDTUnit(a, c, "T", strat).run()
    U2 = W.T8Unit(a, c, "T", strat).run()
    assert _same_store(U1.st, U2.st, (strat,)) > 0
    cr = make_rctx(15)
    R1 = XP.XDRUnit(a, cr, strat).run()
    R2 = W.RUnit(a, cr, "wf2", strat).run()
    assert _same_store(R1.st, R2.st, (strat,)) > 0


# ---------------------------------------------------------------- (k) 계수 변형
def test_k_coef_variants():
    y = np.array([10.0, 12.0, 30.0, 33.0, 31.0]); s = np.array([5.0, 6.0, 10.0, 11.0, 10.0]); b = np.array([1, 1, 2, 2, 2])
    Nb_of = {1: 100, 2: 30}
    w = np.array([50.0, 50.0, 10.0, 10.0, 10.0])
    assert np.isclose(XP.coef_w(y, s, b, Nb_of), (w * s * y).sum() / (w * s * s).sum())
    e1, i1 = XP.coef_re(y[[0, 2]], s[[0, 2]], b[[0, 2]])
    assert i1["sigma2_src"] == "pooled" and np.isclose(i1["tau2"], 0.0) and np.isclose(e1, (s[[0, 2]] @ y[[0, 2]]) / (s[[0, 2]] @ s[[0, 2]]))
    rng = np.random.RandomState(0)
    bb = np.repeat(np.arange(6), 5); ss = rng.uniform(20, 40, 30); Eb = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 3.0])
    ss[bb == 5] = 5.0                                                     # 셀 가중이 작은 블록의 계수가 크다
    yy = Eb[bb] * ss + 0.01 * rng.randn(30)
    e_re, inf = XP.coef_re(yy, ss, bb)
    e_ls = (ss @ yy) / (ss @ ss)
    blk_mean = np.mean([(ss[bb == j] @ yy[bb == j]) / (ss[bb == j] @ ss[bb == j]) for j in range(6)])
    assert inf["tau2"] > 0 and abs(e_re - blk_mean) < abs(e_ls - blk_mean), "τ² 가 크면 블록 평균 쪽"
    a = args()
    c = make_tctx(16)
    U = XP.XDTUnit(a, c, "T", "S8a")
    sel = np.arange(10)
    Ec, Eraw, info = U.coef_variant("S8w", sel)
    assert Ec == Eraw and info["flag"] == "", "S8w 앵커는 수축하지 않은 E_w(계획 2.4)"
    assert not np.isclose(Ec, X.shrink(Eraw, c.E0, 10, XB.KAPPA)), "κ 수축을 하지 않는다"
    assert np.isclose(Ec, XP.coef_w(c.yA[sel], c.sA[sel], c.blkA[sel], U.Nb_of))
    Er, Er_raw, _ = U.coef_variant("S8r", sel)
    assert Er == Er_raw == XP.coef_re(c.yA[sel], c.sA[sel], c.blkA[sel])[0]
    c2 = make_tctx(16)
    c2.sA = np.array(c2.sA, float, copy=True); c2.sA[sel] = np.nan
    U2 = XP.XDTUnit(a, c2, "T", "S8a")
    Ec2, Eraw2, info2 = U2.coef_variant("S8w", sel)
    assert Ec2 == float(c2.E0) and np.isnan(Eraw2) and info2["flag"] == "coef_nonfinite_E0"


# ---------------------------------------------------------------- (l) 탐욕 정책
def test_l_greedy_policy():
    tf_ = XP.TaskFeat.from_ctx(make_tctx(17, nA=120))
    sizes = []

    def f(t, L, C):
        sizes.append(len(L))
        return -t.features_batch(L, C)[:, 1]
    o20 = XP.greedy_policy(tf_, f, 20, 0)
    assert sizes == [1, 5, 10, 15] and len(o20) == 20 and len(set(o20.tolist())) == 20
    o10 = XP.greedy_policy(tf_, f, 10, 0)
    assert np.array_equal(o10, o20[:10])
    assert np.array_equal(XP.greedy_policy(tf_, f, 20, 0), o20)
    firsts = {int(XP.greedy_policy(tf_, f, 1, p)[0]) for p in range(5)}
    assert len(firsts) >= 3, "정책 seed 가 첫 셀을 바꿔야 한다"
    # 정책 seed 는 첫 셀만 바꾼다: 첫 셀이 같으면 나머지 순서도 같다
    c0 = int(XP.greedy_policy(tf_, f, 1, 0)[0])
    other = next((p for p in range(5, 400) if int(XP.greedy_policy(tf_, f, 1, p)[0]) == c0), None)
    if other is not None:
        assert np.array_equal(XP.greedy_policy(tf_, f, 20, other), o20)


# ---------------------------------------------------------------- (m) 세기 출력(실제 자료, 라벨 값 미사용)
@REAL
def test_m_count_only_prints_no_label_stats(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    calls = []
    monkeypatch.setattr(XP, "local_resources", lambda wait_s=0.0, log=None: calls.append(float(wait_s)) or {})
    out_dir = tmp_path / XP.EXP_NAME
    rc = XP.main(["--count-only", "--exp", "xd_r,xd_t,xd_learn,xd_test", "--xdr-targets", "AL-2", "--xdt-targets", "Russia_W,AL-3",
                  "--dev-tasks", "AL-3", "--test-tasks", "CA-3", "--splits", "1", "--out-dir", str(out_dir)])
    out = capsys.readouterr().out
    assert rc == 0 and "[세기]" in out
    assert "[출력 제한]" not in out, "세기 출력에 걸러야 할 줄이 있었다"
    assert XB.FORBIDDEN_OUT.search(out) is None
    df = pd.read_csv(out_dir / "count" / "xd_count.csv")
    assert not [c for c in df.columns if "rmse" in c.lower() or "delta" in c.lower() or "bias" in c.lower()]
    assert df.fit_total.sum() > 0 and set(df.exp) == {"xd_r", "xd_t", "xd_learn", "xd_test"}
    assert not (out_dir / "shards").exists(), "세기는 조각을 쓰지 않는다"
    assert calls, "로컬 세기도 시작 때 메모리 확인·상한을 부른다(계획 1절)"


# ---------------------------------------------------------------- (n) 조각 이름과 봉인 집계
def test_n_shards_sealed_aggregation(tmp_path, capsys):
    a = args(tmp_path, ["--seeds", "1", "--draws-cap", "2"])
    targets = ["Lena", "Canada", "Alaska", "AL-1"]
    for i, t in enumerate(targets):
        for sp in (1, 2):
            c = make_tctx(20 + i, split=sp, target=t)
            for s in XP.STRATS:
                U = XP.XDTUnit(a, c, t, s).run()
                XP.write_unit_shard(a, "xd_t", t, "x", sp, s, U, c, 0.0, {("xd_t", t, "x"): [1, 2]}, "synthetic")
    for sp in (1, 2):
        c = make_rctx(40, split=sp, target="Alaska")
        for s in XP.STRATS:
            U = XP.XDRUnit(a, c, s).run()
            XP.write_unit_shard(a, "xd_r", "Alaska", "r", sp, s, U, c, 0.0, {("xd_r", "Alaska", "r"): [1, 2]}, "synthetic")
    for sp in (1, 2):
        c = make_rctx(41, split=sp, target="Lena")
        for s in XP.STRATS:
            U = XP.XDRUnit(a, c, s).run()
            XP.write_unit_shard(a, "xd_r", "Lena", "r", sp, s, U, c, 0.0, {("xd_r", "Lena", "r"): [1, 2]}, "synthetic")
    names = sorted(p.name for p in (a.XD_SHARDS).glob("*_unit.json"))
    assert "xdt__cpu__AL-1__x__s1__S8a_unit.json" in names and "xdr__cpu__Alaska__r__s2__S4_unit.json" in names
    assert not [n for n in names if "Lena" in n or "Canada" in n], "비알래스카 xd_t·xd_r 조각은 봉인하지 않는 폴더에 쓰지 않는다"
    sealed_names = sorted(p.name for p in a.XD_SEALED_SHARDS.glob("*_unit.json"))
    assert a.XD_SEALED_SHARDS == tmp_path / XP.EXP_NAME / "sealed" / "shards"
    assert "xdt__cpu__Lena__x__s1__S8a_unit.json" in sealed_names and "xdr__cpu__Lena__r__s1__S1_unit.json" in sealed_names
    assert not [n for n in sealed_names if "Alaska" in n or "AL-" in n]
    ok, why = XB.unit_state(a.XD_SHARDS, "xdt", "AL-1", "x", 1, XP.unit_cfg(a, "xd_t", "S8a", "synthetic"), "S8a")
    assert ok, why
    assert XP.shard_dir_of(a, "xd_t", "Lena", "x") == a.XD_SEALED_SHARDS and XP.shard_dir_of(a, "xd_t", "AL-1", "x") == a.XD_SHARDS
    assert XP.shard_dir_of(a, "xd_r", "Canada", "r") == a.XD_SEALED_SHARDS and XP.shard_dir_of(a, "xd_r", "AL-2", "r") == a.XD_SHARDS
    assert XP.shard_dir_of(a, "xd_t", "AL-1", "r") == a.XD_SEALED_SHARDS, "알래스카 계열 판정은 (대상, 모드) 짝으로 한다"
    assert XP.shard_dir_of(a, "xd_learn", "AL-3", "x") == a.XD_SHARDS and XP.shard_dir_of(a, "xd_test", "LE-1", "x") == a.XD_SHARDS
    ok, why = XB.unit_state(a.XD_SEALED_SHARDS, "xdt", "Lena", "x", 1, XP.unit_cfg(a, "xd_t", "S8a", "synthetic"), "S8a")
    assert ok, why
    with pytest.raises(SystemExit):
        XP.load_learning(a.XD_SEALED_SHARDS, "xdl")
    capsys.readouterr()
    recs = XP.summarize_alg(XP.alg_dirs(a), "xdt", "xdr", nboot=200, root=tmp_path, allowed=[tmp_path])
    out = capsys.readouterr().out
    sdir = tmp_path / XP.EXP_NAME / "sealed"
    assert {r["file"] for r in recs} == {"xd_alg_contrasts.csv", "xd_alg_tests.csv", "xd_alg_holm.csv", "xd6_traits.csv", "xd_alg_meta.json"}
    assert all((sdir / r["file"]).exists() for r in recs) and (sdir / "sealed_manifest.json").exists()
    assert XB.FORBIDDEN_OUT.search(out) is None
    for line in out.strip().splitlines():
        assert line.startswith("[봉인]"), f"집계기 화면 출력에 봉인 기록 밖 줄이 있다: {line[:40]}"
    holm = pd.read_csv(sdir / "xd_alg_holm.csv")
    assert len(holm) == XP.HOLM_M and int(holm.m.iloc[0]) == 20
    tests = pd.read_csv(sdir / "xd_alg_tests.csv")
    assert set(tests.test_id) == {"XD-1", "XD-2", "XD-3"} and (tests.test_id == "XD-3").sum() == 16
    con = pd.read_csv(sdir / "xd_alg_contrasts.csv")
    assert {"S8a-S1", "S8w-S8a", "S8c-S4"} <= set(con.contrast) and {"PE1", "region"} <= set(con.scope)
    assert {"Lena|x", "Canada|x", "Lena|r"} <= set(con.target.dropna()), "집계는 봉인 조각 폴더도 읽는다"
    assert set(con[con.contrast == "S8w-S8a"].kind) == {"same"} and set(con[con.contrast == "S8a-S1"].kind) == {"two_stage"}
    with pytest.raises(PermissionError):
        XB.assert_not_sealed(sdir / "xd_alg_tests.csv")


# ---------------------------------------------------------------- (o) XD-4 서술
def _tm_test(name, lev_s9, lev_ap, nb=10, splits=(1, 2), seed=0):
    rng = np.random.RandomState(seed)
    by = {}
    for sp in splits:
        ncell = rng.randint(4, 10, nb)
        st = H4.BlockStore(name, sp, np.repeat([f"b{sp}{j}" for j in range(nb)], ncell))
        cnt = st.ncell
        st.add_sse(H.P0_KEY, cnt * 30.0 ** 2, cnt)
        for n in (20, 40):
            for d in range(3):
                for s in (0, 1):
                    for pl, lev in (("S9", lev_s9), ("AP", lev_ap), ("S1", 20.0)):
                        st.add_sse(("R1", LO, "1", pl, n, d, s, 0.25), cnt * (lev + 0.01 * rng.rand(nb)) ** 2, cnt)
        by[sp] = st
    return H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in splits}, 200)


def test_o_xd4_description():
    tms = {"LE-1|x": _tm_test("LE-1|x", 18.0, 20.0), "LE-2|x": _tm_test("LE-2|x", 19.0, 20.0, seed=1), "CA-2|x": _tm_test("CA-2|x", 18.0, 20.0, seed=2),
           "CA-3|x": _tm_test("CA-3|x", 21.0, 20.0, seed=3), "Tibet_LGD|x": _tm_test("Tibet_LGD|x", 22.0, 20.0, seed=4)}
    tms["Russia_C~lgd|x"] = _tm_test("Russia_C~lgd|x", 10.0, 20.0, seed=5)            # 서술 행: k/5 에 넣지 않는다
    tdf, fdf, cdf = XP.xd4_rows(tms, "AP", nboot=200)
    assert int(((tdf.direction == "우세 방향") & tdf.counted).sum()) == 3 and not bool(tdf.set_index("task").loc["Russia_C~lgd", "counted"])
    assert fdf.set_index("family").agree.to_dict() == {"Lena": True, "Canada": True, "Tibet": False}
    v = tdf.set_index("task")
    assert abs(v.loc["LE-1", "v_cell"] - (-0.1)) < 0.01 and abs(v.loc["Tibet_LGD", "v_beq"] - 0.1) < 0.01
    assert len(cdf) == 12 and set(cdf.n) == {20, 40}
    vc, vb, k = XP.rel_pairs(tms["LE-1|x"], W.gk("R1", 20, LO, 0.25, "S9"), W.gk("R1", 20, LO, 0.25, "AP"))
    assert k == 2 * 3 * 3, "분할 2 × (A 추출 3 × B 추출 3) 쌍"


# ---------------------------------------------------------------- (p) 색인 파일과 시험 팔
def test_p_index_file_and_test_unit(tmp_path):
    a = args(tmp_path)
    c = make_tctx(30, nA=90)
    tf_ = XP.TaskFeat.from_ctx(c)
    path = tmp_path / XP.EXP_NAME / "policy" / "idx.json"
    sha = XP.export_index({("T", 1): tf_}, lambda t, L, C: -t.features_batch(L, C)[:, 1], "S9-GBM", "deadbeef", path, seeds=(0, 1), n_max=20,
                          allowed=[tmp_path])
    idx = XP.IndexFile(path)
    assert idx.sha256 == sha and set(idx.entries_for("T", "x", 1)) == {0, 1}
    U = XP.XDXUnit(a, c, "T", idx.entries_for("T", "x", 1), "S8a").run()
    pls = {k[3] for k in U.st.keys}
    assert {"S9", "S1", "AP"} <= pls
    assert {int(k[5]) for k in U.st.keys if k[3] == "S9"} == {0, 1}
    c_bad = make_tctx(31, nA=90)
    with pytest.raises(SystemExit):
        XP.XDXUnit(a, c_bad, "T", idx.entries_for("T", "x", 1), "S8a")
    fz = tmp_path / XP.EXP_NAME / "freeze" / "fz.json"
    XP._write_json(fz, dict(index_sha256="0" * 64), allowed=[tmp_path])
    with pytest.raises(SystemExit):
        XP.IndexFile(path, fz)
    path.write_text(path.read_text().replace('"policy_seed": 1', '"policy_seed": 2'))
    with pytest.raises(SystemExit):
        XP.IndexFile(path)


# ---------------------------------------------------------------- (q) 동결
def test_q_freeze_choice(tmp_path, capsys):
    pol = tmp_path / XP.EXP_NAME / "policy"
    for nm, u in (("gbm", -0.031234), ("ds", -0.045678)):
        ip = pol / f"{nm}_index.json"
        XP._write_json(ip, dict(format=XP.INDEX_FORMAT, entries=[]), allowed=[tmp_path])
        XP._write_json(pol / f"{nm}_meta.json", dict(cv_utility=u, index_path=str(ip), model_path="m", model_sha256="x", features=XP.FEATS, seeds=[0]),
                       allowed=[tmp_path])
    capsys.readouterr()
    out_p = tmp_path / XP.EXP_NAME / "freeze" / "m.json"
    man = XP.freeze(pol / "gbm_meta.json", pol / "ds_meta.json", out_p, allowed=[tmp_path])
    out = capsys.readouterr().out
    assert man["s9_star"] == "S9-DS" and "-0.045" not in out and "-0.031" not in out and XP.verify_sidecar(out_p)
    XP._write_json(pol / "ds_meta.json", dict(cv_utility=-0.031234, index_path=str(pol / "ds_index.json")), allowed=[tmp_path])
    assert XP.freeze(pol / "gbm_meta.json", pol / "ds_meta.json", out_p, allowed=[tmp_path])["s9_star"] == "S9-GBM", "동률은 S9-GBM"
    assert XP.freeze(pol / "gbm_meta.json", None, out_p, allowed=[tmp_path])["variants_available"] == ["S9-GBM"]


# ---------------------------------------------------------------- (r) 동결 모듈
def test_r_frozen_modules_unchanged():
    for k, v in {"h40": "7098f59dabe2b73f", "h42": "22e215e435e157be", "h54": "cf9a1f6d0929c892", "h41": "492373e4d37b5ea4"}.items():
        assert XB.assert_frozen(k) == v
    assert XP.W is XB.W and XP.H is XB.H


# ---------------------------------------------------------------- (s) S9-GBM
def test_s_gbm_train_and_policy():
    df = synth_learning_df(n_sets=15, seed=2)
    model, rep, it, _ = XP.train_gbm(df, threads=1, iters_grid=(10,))
    assert it == 10 and np.isfinite(rep[0]["cv_utility"])
    tf_ = XP.TaskFeat.from_ctx(make_tctx(33, nA=100))
    o = XP.greedy_policy(tf_, XP.gbm_score_fn(model, 1), 15, 2)
    assert len(o) == 15 and len(set(o.tolist())) == 15
    d = df[df.task == "AL-1"].copy()
    assert np.isclose(XP.cv_utility(d, d.U.values), d.groupby(["task", "split", "n"]).U.min().mean()), "오라클 예측이면 묶음 최솟값의 평균"


# ---------------------------------------------------------------- (t) 재현 관문과 마감 기본값
def test_t_gate_and_default(tmp_path, capsys):
    root = tmp_path / XP.EXP_NAME
    d, ref = root / "shards", tmp_path / "ref"
    lev = dict(S1=20.0, S2=19.0, S4=21.0)
    for sp in (1, 2):
        st = _store("Lena|x", sp, lev)
        _write_synth_shard(d, "xdt", "Lena", "x", sp, st, variant="S1")
        _write_synth_shard(ref, "wf8", "Lena", "x", sp, _store("Lena|x", sp, lev), variant="S1")
    capsys.readouterr()
    s = XP.gate_alg(d, "xdt", "xdr", ref, None, root / "gate" / "g.csv", allowed=[tmp_path])
    out = capsys.readouterr().out
    assert s["passed"] and s["n_keys"] > 0 and XB.FORBIDDEN_OUT.search(out) is None
    st = _store("Lena|x", 1, dict(S1=20.5, S2=19.0, S4=21.0))
    _write_synth_shard(ref, "wf8", "Lena", "x", 1, st, variant="S1")
    s = XP.gate_alg(d, "xdt", "xdr", ref, None, root / "gate" / "g.csv", allowed=[tmp_path])
    assert not s["passed"] and s["n_fail"] > 0
    r = XP.select_default(root / "selection", allowed=[tmp_path])
    assert XP.load_selection(r["path"])["chosen"] == "S8a"
    bad = root / "selection" / "bad.json"
    XP._write_json(bad, dict(chosen="S8w"), allowed=[tmp_path])
    with pytest.raises(SystemExit):
        XP.load_selection(bad)


# ---------------------------------------------------------------- (u) 명령행 거부 규칙
def test_u_cli_refusals(monkeypatch, tmp_path):
    monkeypatch.delenv("WF_RESCALE", raising=False); monkeypatch.delenv("LG_RESCALE", raising=False)
    monkeypatch.setattr(XP, "local_resources", lambda wait_s=0.0, log=None: {})
    with pytest.raises(SystemExit):
        XP.parse(["--exp", "xd_test"])                                     # 동결 기록 없이 시험 과제 평가
    with pytest.raises(SystemExit):
        XP.parse(["--exp", "xd_learn", "--dev-tasks", "LE-1"])              # XD-5 전 개발 과제 밖 학습 자료
    od = tmp_path / XP.EXP_NAME
    with pytest.raises(SystemExit):                                         # --xd5 자기 선언만으로는 안 된다(XD-4 봉인 표 없음)
        XP.parse(["--exp", "xd_learn", "--dev-tasks", "LE-1", "--xd5", "--out-dir", str(od)])
    sd = XB.sealed_dir(XP.EXP_NAME, tmp_path)
    sd.mkdir(parents=True)
    (sd / XP.XD4_SEALED).write_text("{}")                                  # 파일만 있고 봉인 기록 항목이 없다
    with pytest.raises(SystemExit):
        XP.parse(["--exp", "xd_learn", "--dev-tasks", "LE-1", "--xd5", "--out-dir", str(od)])
    with pytest.raises(SystemExit):
        XP.main(["--summarize-xd5", "--xd5", "--out-dir", str(tmp_path / "other" / XP.EXP_NAME)])
    XB.write_sealed(XP.EXP_NAME, {XP.XD4_SEALED: {"k": 1}}, root=tmp_path, allowed=[tmp_path], quiet=True)
    assert XP.parse(["--exp", "xd_learn", "--dev-tasks", "LE-1", "--xd5", "--out-dir", str(od)]).XD_T["xd_learn"] == ["LE-1"]
    with pytest.raises(SystemExit):
        XP.main(["--exp", "xd_t", "--xdt-targets", "Lena"])                 # 허용 표지 없는 본 실행
    with pytest.raises(SystemExit):
        XP.main(["--summarize-only", "--out-dir", str(tmp_path / XP.EXP_NAME)])   # 허용 표지 없는 10,000회 집계
    a = XP.parse(["--shard", "1/3"])
    assert a.XD_SHARD == (1, 3)
    assert XP._drop_opt(["--smoke", "--exp", "xd_r", "--exp=xd_t", "--threads", "2"], "--exp") == ["--smoke", "--threads", "2"]


# ---------------------------------------------------------------- (v) XD-5
def test_v_xd5_folds_and_regret():
    tasks = list(XP.DEV_TASKS) + list(XP.TEST_TASKS)
    fam = XP.lofo_folds(tasks)
    assert [f["fold"] for f in fam] == ["Alaska", "Lena", "Canada", "Tibet"]
    for f in fam:
        assert not {XP.family_of_task(t) for t in f["train"]} & {f["fold"]} and set(f["val"]) | set(f["train"]) == set(tasks)
    lo = XP.loto_folds(tasks)
    assert [f["fold"] for f in lo] == list(XP.TEST_TASKS) and all("LE-2" in f["train"] for f in lo if f["fold"] == "LE-1")
    df = synth_learning_df(tasks=("Alaska", "AL-1", "LE-1", "LE-2", "CA-2"), n_sets=10)
    tab = XP.cv_by_folds(df, XP.lofo_folds(sorted(df.task.unique())) + XP.loto_folds(sorted(df.task.unique())), 10)
    assert set(tab.scheme) == {"family_out", "task_out"} and np.isfinite(tab.cv_utility).all()
    st = H4.BlockStore("LE-1|x", 1, np.repeat([f"b{j}" for j in range(4)], 3))
    cnt = st.ncell
    for n in (20, 40):
        for d in range(2):
            st.add_sse(("R1", LO, "1", "S1", n, d, 0, 0.25), cnt * 10.0 ** 2, cnt)
            st.add_sse(("R1", LO, "1", "S9", n, d, 0, 0.25), cnt * 9.0 ** 2, cnt)
            st.add_sse(("R1", LO, "1", "AP", n, d, 0, 0.25), cnt * 11.0 ** 2, cnt)
            st.add_sse(("R1", LO, "1", "S9", n, d, 1, 0.25), cnt * 1.0 ** 2, cnt)      # seed 1 은 U 에 쓰지 않는다(학습 자료와 같은 seed 0)
    tm = H.TMx("LE-1|x", {1: st}, {1: dict(dup_of=-1, valid=True)}, 0)
    dl = pd.DataFrame([dict(task="LE-1", split=1, n=n, j=j, U=u) for n in (20, 40) for j, u in enumerate((-0.2, -0.05, 0.1))])
    r = XP.xd5_regret({"LE-1|x": tm}, dl).set_index("n")
    assert np.isclose(r.loc[20, "U_S9"], -0.1) and np.isclose(r.loc[20, "U_AP"], 0.1) and np.isclose(r.loc[20, "regret_S9"], 0.1)


# ---------------------------------------------------------------- (w) XC 호환
def test_w_placement_order_xc_contract():
    rng = np.random.RandomState(3)
    Z = rng.randn(150, 6)
    pool = np.array([-12600122, -330001, 1609985, 12099765, 16199857, -9900123, 77, -5])
    blk = pool[rng.randint(0, len(pool), 150)]
    for name, (g, f) in XP.S8_SPECS.items():
        o = XP.placement_order(name, Z, blk, [10, 40, 160], 12345)
        assert np.array_equal(o, XP.algorithm_p_order(Z, blk, (10, 40, 160), g, 12345, first=f))
        assert np.array_equal(o, XP.algorithm_p_order(Z, blk.astype(str), (10, 40, 160), g, 12345, first=f)), "블록 번호의 형(정수·문자열)과 무관"
    o2 = XP.placement_order("S2", Z, blk, [10, 40], 99)
    assert np.array_equal(np.sort(o2[:40]), np.sort(XP.round_robin_order(blk, 99)[:40]))
    assert np.array_equal(XP.placement_order("S4", Z, blk, [10, 40], 7), W.kcenter_order(Z, 40, 7))


# ================================================================ 2026-10-04 검토 반영 시험
# ---------------------------------------------------------------- (x) 스모크 대상은 알래스카 계열, 조각은 봉인 폴더
def test_x_smoke_targets_alaska_only(tmp_path):
    fam = set(XP.ALASKA_X) | set(XP.ALASKA_R)
    assert all(set(v) <= fam for v in XP.SMOKE_TARGETS.values()), "스모크 대상 ⊂ ALASKA_X ∪ ALASKA_R"
    assert XP.assert_smoke_alaska()
    with pytest.raises(AssertionError):
        XP.assert_smoke_alaska(dict(xd_t=("Russia_W", "AL-3")))
    a = XP.parse(["--smoke", "--out-dir", str(tmp_path / XP.EXP_NAME)])
    assert all(set(v) <= fam for v in a.XD_T.values()) and not set(a.XD_T["xd_t"]) & {"Russia_W", "Lena", "Canada", "Tibet_LGD"}
    assert a.XD_SHARDS == a.XD_SEALED_SHARDS == tmp_path / XP.EXP_NAME / "sealed" / XP.SMOKE_SHARDS
    for exp in ("xd_t", "xd_r", "xd_learn", "xd_test"):
        assert XP.shard_dir_of(a, exp, a.XD_T[exp][0], XP.MODE_OF[exp]).parent.name == "sealed"


# ---------------------------------------------------------------- (y) XD-4 의 고정된 Algorithm P = XC 의 Algorithm P 순서
def test_y_xd4_ap_matches_xc_order():
    XC = _load("x_workflow_end_to_end", "scripts/3_deep_learning/x_workflow_end_to_end.py")
    a = args()                                                             # 시험 격자 10·20(일정과 다르다)
    c = make_tctx(41, nA=96)
    Z = W.standardize(c.XA, W.standardize_fit(c.XA))
    differs = 0
    for ap in XP.S8_SPECS:
        U = XP.XDXUnit(a, c, "T", {}, ap)
        for d in range(4):
            seed = XB.seed_of(XP.S8_SEED_TAG, c.target, c.mode, c.split, d)
            sets = U.ap_sets([10, 20], d)
            o_xd = XP.placement_order(ap, Z, c.blkA, XP.AP_SCHEDULE, seed)
            o_xc = XC.placement_order(XC.algp_spec(ap), Z, c.blkA, [n for n in XC.XC_GRID if 0 < n < len(c.yA)], seed, "ref")
            for n in (10, 20):
                assert np.array_equal(sets[n], np.sort(o_xd[:n])), "XD-4 AP = placement_order(일정 10, 40, 160)"
                assert np.array_equal(sets[n], np.sort(o_xc[:n])), "XD-4 AP = XC 참조 구현의 Algorithm P 순서"
            grid = XP.algorithm_p_order(Z, c.blkA, (10, 20), *XP.S8_SPECS[ap][:1], seed, first=XP.S8_SPECS[ap][1])
            differs += int(not np.array_equal(np.sort(grid[:20]), sets[20]))
    assert differs > 0, "시험 격자 일정(10, 20)은 다른 라벨 집합을 낸다(회귀 방지: 일정을 시험 격자로 바꾸면 이 시험이 잡는다)"
    for ap in ("S2", "S4"):                                                 # 일정이 없는 후보는 XD-alg(WF8 구현)과 같다
        U = XP.XDXUnit(a, c, "T", {}, ap)
        s1, s2 = U.ap_sets([10, 20], 1), U.xd_sets(ap, [10, 20], 1)
        assert all(np.array_equal(s1[n], s2[n]) for n in (10, 20))
    assert XP.XDXUnit(a, c, "T", {}, "S1").ap_sets([10, 20], 0) is None


# ---------------------------------------------------------------- (z) 색인 과제 집합과 시작 전 색인 확인
def test_z_index_tasks_and_coverage(tmp_path, monkeypatch):
    DS = _load("x_placement_deepsets", "scripts/3_deep_learning/x_placement_deepsets.py")
    ds_tasks = XP._list(DS.build_parser().parse_args([]).test_tasks)
    gbm_tasks = XP.parse(["--exp", "xd_t"]).XD_T["xd_test"]
    assert set(ds_tasks) == set(gbm_tasks) == set(XP.TEST_TASKS + XP.TEST_DESC), "S9-DS·S9-GBM 색인 과제 = xd_test 기본 과제"
    tfs = {("LE-1", sp): XP.TaskFeat.from_ctx(make_tctx(50 + sp, split=sp, target="LE-1", nA=80)) for sp in (1, 2)}
    path = tmp_path / XP.EXP_NAME / "policy" / "idx.json"
    XP.export_index(tfs, lambda t, L, C: -t.features_batch(L, C)[:, 1], "S9-GBM", "x", path, seeds=(0, 1), n_max=20, allowed=[tmp_path])
    idx = XP.IndexFile(path)
    exp_ok = {("xd_test", "LE-1", "x"): [1, 2], ("xd_test", "Russia_C~lgd", "x"): [1]}
    cov = XP.index_coverage(idx, exp_ok, (0, 1), 20)
    assert not cov["counted"] and [m[0] for m in cov["desc"]] == ["Russia_C~lgd"] and not cov["short"]
    cov = XP.index_coverage(idx, {("xd_test", "LE-1", "x"): [1, 2, 3]}, (0, 1, 2), 40)
    assert {(m[0], m[1]) for m in cov["counted"]} == {("LE-1", 1), ("LE-1", 2), ("LE-1", 3)} and cov["short"]
    # run_units: 시험 과제 항목이 없으면 시작 전 거부, 서술 과제 항목이 없으면 그 단위만 건너뛴다(SystemExit 없음)
    a = args(tmp_path)
    a.XD_EXPS, a.XD_INDEX = ["xd_test"], idx
    seen = []
    monkeypatch.setattr(XP, "_worker_run", lambda *u: seen.append(u) or dict(status="ok"))
    units = [("xd_test", "LE-1", "x", 1, ""), ("xd_test", "LE-1", "x", 2, ""), ("xd_test", "Russia_C~lgd", "x", 1, "")]
    monkeypatch.setattr(XP, "enumerate_units", lambda a_: (list(units), dict(exp_ok), []))
    res = XP.run_units(a, [])
    assert seen == units[:2] and not res["failed"]
    monkeypatch.setattr(XP, "enumerate_units", lambda a_: (units + [("xd_test", "LE-2", "x", 1, "")], {**exp_ok, ("xd_test", "LE-2", "x"): [1]}, []))
    seen.clear()
    with pytest.raises(SystemExit):
        XP.run_units(a, [])
    assert not seen, "시험 과제의 색인 항목이 빠지면 어떤 단위도 시작하지 않는다"


# ---------------------------------------------------------------- (aa) XD-1·2·3 문장 갈래
def _prow(v4, delta, noninf=False, p=0.001, pni=0.001, pool="지역 5/5"):
    lo = 0.1 if v4 == "열세" else (-1.0 if v4 == "우세" else -0.2)
    return dict(verdict4=v4, delta=delta, noninf=noninf, p_two=p, p_ni=pni, p_ni_x2=min(1.0, 2 * pni), pool=pool, ci_lo=lo, ci_hi=0.4)


def _rrow(target, n, v4, delta, arm="x"):
    return dict(contrast="S8w-S8a", method="R1", scope="region", target=target, n=n, verdict4=v4, delta=delta, arm=arm, p_two=0.001,
                ci_lo=0.1, ci_hi=0.4, worse=v4 == "열세")


def test_aa_xd_sentence_branches():
    pools = {("S8a-S1", "R1", 10, "PE1"): _prow("우세", -0.8), ("S8a-S1", "R1", 40, "PE1"): _prow("우세", -0.9, pool="부분(지역 2/5)")}
    for v in XP.S8_SPECS:
        for ref in ("S2", "S4"):
            for n in XP.ALASKA_X_N:
                pools[(f"{v}-{ref}", "R1", n, "PE1")] = _prow("열세", 0.3, noninf=True, pni=0.001)     # CI (0.1, 0.4): 비열등 + 열세
    del pools[("S8p-S4", "R1", 40, "PE1")]                                                        # 행 없음
    df, ht = XP.xd_tests([], pools, [])
    x1 = df[(df.test_id == "XD-1") & df.n.notna()].set_index("n")
    assert "지역 지역" not in "".join(x1.sentence) and "(지역 5/5)" in x1.loc[10, "sentence"] and "(부분(지역 2/5))" in x1.loc[40, "sentence"]
    x2 = df[(df.test_id == "XD-2") & df.n.isna()].iloc[0]
    assert x2.verdict.startswith("판정 불가") and "판정할 수 없었다" in x2.sentence and "확인하지 못했다" not in x2.sentence
    x3 = df[df.test_id == "XD-3"].set_index("item")
    s = x3.loc["S8a-S2|R1(0.25)|n10|PE1", "sentence"]
    assert "0.5 cm 안이었다" in s and "컸다" in s and XB.SMALL_EFFECT_TXT in s, "비열등과 열세가 함께 성립하면 두 문장을 모두 쓴다"
    assert x3.loc["S8a-S2|R1(0.25)|n10|PE1", "sentence_branch"] == "비열등+열세"
    miss = x3.loc["S8p-S4|R1(0.25)|n40|PE1"]
    assert miss.verdict4 == "행 없음" and "판정할 수 없었다" in miss.sentence and "확인하지 못했다" not in miss.sentence
    assert len(ht) == XP.HOLM_M
    # XD-2: 레나 두 행 우세, 열세 없음 → 지지(보정 전 유의 표기는 Holm p 에 따른다)
    rows_x = [_rrow("Lena|x", 10, "우세", -0.3), _rrow("Lena|x", 40, "우세", -1.2), _rrow("Alaska|x", 10, "미결정", 0.1)]
    df, _ = XP.xd_tests(rows_x, pools, [])
    x2 = df[(df.test_id == "XD-2") & df.n.isna()].iloc[0]
    assert x2.verdict == "지지" and "손해를 줄였다" in x2.sentence and XB.SMALL_EFFECT_TXT in x2.sentence
    # 지역 내 캐나다 행 하나가 열세 → 기각(열세 문장), 레나 행이 없어도 판정할 수 있다
    rows_r = [_rrow("Canada|r", 100, "열세", 0.7, arm="r")]
    df, _ = XP.xd_tests([_rrow("Lena|x", 10, "우세", -0.9)], pools, rows_r)
    x2 = df[(df.test_id == "XD-2") & df.n.isna()].iloc[0]
    assert x2.verdict.startswith("기각") and "Canada|r n100" in x2.sentence and "오차를 키웠다" in x2.sentence
    # 레나 두 행 동등 → 기각(동등 문장)
    df, _ = XP.xd_tests([_rrow("Lena|x", 10, "동등", 0.1), _rrow("Lena|x", 40, "동등", -0.1)], pools, [])
    x2 = df[(df.test_id == "XD-2") & df.n.isna()].iloc[0]
    assert x2.verdict == "기각" and "0.5 cm 안에서 같았다" in x2.sentence
    assert XP.xd2_verdict({10: dict(verdict4="우세"), 40: dict(verdict4="판정 불가")}, [])[0] == "판정 불가"
    assert XP.xd3_parts(None) == ["판정 불가"] and XP.xd3_parts(dict(verdict4="미결정", noninf=False)) == ["미결정"]


# ---------------------------------------------------------------- (ab) XD-4 집계의 화면 출력은 봉인 기록뿐
def _xdx_store(name, split, lev, nb=8, seed=0):
    rng = np.random.RandomState(seed + split)
    st = H4.BlockStore(name, split, np.repeat([f"b{split}{j}" for j in range(nb)], rng.randint(4, 9, nb)))
    cnt = st.ncell
    st.add_sse(H.P0_KEY, cnt * 30.0 ** 2, cnt)
    for n in XP.TEST_N:
        for d in range(2):
            for s in (0, 1):
                for pl, lv in lev.items():
                    st.add_sse(("R1", LO, "1", pl, n, d, s, 0.25), cnt * (lv + 0.02 * rng.rand(nb)) ** 2, cnt)
    return st


def test_ab_summarize_test_prints_sealed_only(tmp_path, capsys):
    d = tmp_path / XP.EXP_NAME / "shards"
    for i, t in enumerate(XP.TEST_TASKS):
        for sp in (1, 2):
            st = _xdx_store(f"{t}|x", sp, dict(S9=19.0 + i, AP=20.0, S1=20.5), seed=i)
            XB.write_shard(d, "xdx", t, "x", sp, [], [st], XB.make_unit_cfg("xd_test", "", "synthetic"), unit=dict(dup_of=-1, valid=True),
                           variant="", expected=[1, 2], allowed=[tmp_path])
    capsys.readouterr()
    recs = XP.summarize_test(d, "xdx", nboot=200, root=tmp_path, allowed=[tmp_path])
    out = capsys.readouterr().out
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    assert lines and all(ln.startswith("[봉인]") for ln in lines), "XD-4 집계의 화면 출력은 봉인 기록 줄뿐이어야 한다"
    assert XB.FORBIDDEN_OUT.search(out) is None and "xd4_summary.json" in {r["file"] for r in recs}
    assert (tmp_path / XP.EXP_NAME / "sealed" / XP.XD4_SEALED).exists()
    assert XP.require_xd4_sealed(tmp_path / XP.EXP_NAME).name == XP.XD4_SEALED


# ---------------------------------------------------------------- (ac) 로컬 자원 규약(main 시작 때)
def test_ac_local_resources_called(tmp_path, monkeypatch):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    monkeypatch.delenv("WF_RESCALE", raising=False); monkeypatch.delenv("LG_RESCALE", raising=False)
    mem, lim = [], []
    monkeypatch.setattr(XB, "require_memory", lambda min_gb=30.0, wait_s=0.0, poll_s=60.0: mem.append((min_gb, wait_s)) or 99.0)
    monkeypatch.setattr(XP, "limit_data", lambda gb=10.0: lim.append(gb) or True)
    assert XP.main(["--select-default", "--mem-wait", "5", "--out-dir", str(tmp_path / XP.EXP_NAME)]) == 0
    assert mem == [(30.0, 5.0)] and lim == [10.0]
    monkeypatch.setenv("WF_RESCALE", "1")
    mem.clear(); lim.clear()
    assert XP.main(["--select-default", "--out-dir", str(tmp_path / XP.EXP_NAME)]) == 0
    assert not mem and not lim, "Rescale 작업 안에서는 로컬 자원 규약을 걸지 않는다"
    monkeypatch.delenv("WF_RESCALE")
    monkeypatch.setattr(XB, "require_memory", lambda *a_, **k_: (_ for _ in ()).throw(SystemExit("[대기 초과]")))
    with pytest.raises(SystemExit):
        XP.main(["--select-default", "--out-dir", str(tmp_path / XP.EXP_NAME)])


# ---------------------------------------------------------------- (ad) xd_test 는 커밋된 동결 기록에서만
def test_ad_freeze_commit_required(tmp_path, monkeypatch):
    monkeypatch.delenv("WF_RESCALE", raising=False); monkeypatch.delenv("LG_RESCALE", raising=False)
    od = tmp_path / XP.EXP_NAME
    tf_ = XP.TaskFeat.from_ctx(make_tctx(60, target="LE-1", nA=80))
    ip = od / "policy" / "idx.json"
    isha = XP.export_index({("LE-1", 1): tf_}, lambda t, L, C: -t.features_batch(L, C)[:, 1], "S9-GBM", "x", ip, seeds=(0,), n_max=20,
                           allowed=[tmp_path])
    fz = od / "freeze" / "s9_freeze_manifest.json"
    XP._write_json(fz, dict(s9_star="S9-GBM", index_path=str(ip), index_sha256=isha), allowed=[tmp_path])
    sel = XP.select_default(od / "selection", allowed=[tmp_path])
    argv = ["--exp", "xd_test", "--freeze-manifest", str(fz), "--selection-file", sel["path"], "--out-dir", str(od)]
    st = XP.freeze_commit_state(fz)
    assert not st["ok"], "저장소 밖·커밋 전 동결 기록은 통과하지 못한다"
    with pytest.raises(SystemExit):
        XP.parse(argv)                                                      # git 작업 트리: 커밋되지 않은 동결 기록 거부
    monkeypatch.setattr(XP, "git_available", lambda: False)                # Rescale 묶음(git 없음)
    with pytest.raises(SystemExit):
        XP.parse(argv)                                                      # 확인 기록 없음
    with pytest.raises(SystemExit):
        XP.write_freeze_attestation(fz, allowed=[tmp_path])                 # 커밋 확인 실패이면 확인 기록을 쓰지 않는다
    XP.write_freeze_attestation(fz, allowed=[tmp_path], state=dict(ok=True, commit="f" * 40, rel="x"))
    a = XP.parse(argv)
    assert a.XD_FREEZE_COMMIT["source"] == "attestation" and a.XD_INDEX.sha256 == isha and a.XD_AP == "S8a"
    XP._write_json(fz, dict(s9_star="S9-GBM", index_path=str(ip), index_sha256=isha, extra=1), allowed=[tmp_path])
    with pytest.raises(SystemExit):
        XP.parse(argv)                                                      # 확인 뒤 동결 기록이 바뀌었다
    tracked = "scripts/3_deep_learning/h40_label_grid.py"
    if XB._git("ls-files", tracked) and XB._git("diff", "--name-only", "HEAD", "--", tracked) == "":
        st = XP.freeze_commit_state(ROOT / tracked)
        assert st["ok"] and st["tracked"] and st["clean"] and len(st["commit"]) == 40
    assert XP.rebase_path("/elsewhere/root/data/processed/xbatch/x.json") == ROOT / "data/processed/xbatch/x.json"


# ---------------------------------------------------------------- (ae) S9-DS 의 GPU 는 0–4 하나만(nvidia-smi 표를 대리로, CUDA 없이)
def _fake_gpu_table(busy=(), used=None):
    used = used or {}
    return [dict(index=i, mem_mib=float(used.get(i, 0)), util=0.0, name="GPU-X", driver="999.0", busy=i in set(busy)) for i in range(10)]


def test_ae_gpu_restricted_to_0_4(monkeypatch):
    DS = _load("x_placement_deepsets", "scripts/3_deep_learning/x_placement_deepsets.py")
    monkeypatch.setattr(DS, "gpu_table", lambda: _fake_gpu_table(busy=(0, 1), used={2: 900.0}))
    assert DS.GPU_ALLOWED == (0, 1, 2, 3, 4)
    assert DS.pick_gpu()["index"] == 3, "0·1 은 계산 프로세스가 있고 2 는 사용 메모리가 커서 3 이 첫 빈 GPU 다"
    assert DS.pick_gpu(exclude=(3,))["index"] == 4
    for g in (5, 7, 9, -1):
        with pytest.raises(SystemExit):                                     # 허용 목록(0–4) 밖은 nvidia-smi 를 보기 전에 거부
            DS.setup_device("cuda", g, 0, threads=1)
    monkeypatch.setattr(DS, "gpu_table", lambda: _fake_gpu_table(busy=range(5)))
    assert DS.pick_gpu() is None, "5–9 는 비어 있어도 고르지 않는다"
    with pytest.raises(SystemExit):
        DS.setup_device("cuda", None, 0, threads=1)                         # 빈 GPU(0–4) 없음
    info = DS.setup_device("auto", None, 0, threads=1)                      # auto 는 CPU 대체 경로(계획 7.2-6)
    assert info["device"] == "cpu" and os.environ["CUDA_VISIBLE_DEVICES"] == "" and info["cublas"] == ":4096:8"
    monkeypatch.setattr(DS, "gpu_table", lambda: [g for g in _fake_gpu_table() if g["index"] != 2])
    with pytest.raises(SystemExit):
        DS.setup_device("cuda", 2, 0, threads=1)                            # 번호는 허용되나 nvidia-smi 목록에 없다(CUDA 를 만들기 전에 거부)
    assert DS.DEVICE_ORDER == "PCI_BUS_ID" and DS.CUBLAS_CFG == ":4096:8"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "", "시험은 GPU 를 만들지 않는다"
    assert DS.build_parser().parse_args([]).mem_wait == 1800.0, "S9-DS 도 30 GB 대기 규약(--mem-wait)을 x_placement_policy 와 같게 둔다"
