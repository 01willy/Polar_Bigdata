"""scripts/2_evaluation/h45_normalized_conformal.py 단위 시험(계획 docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 4절, spec_B 8절).

가벼운 시험(합성 자료, 학습 없음, 실제 자료를 읽지 않는다)
(a) σ = 1 인 정규화 구간이 const 와 같다.
(b) const@h37 의 보정 분위와 커버리지가 h37 의 hier2_cdf 식(점수 |log y − log(E0·s)|, 지역 등가중 분위)의 수계산과 같다.
(c) 교환 가능한 합성 자료에서 const 와 참 σ 정규화의 90 % 커버리지가 0.90 ± 0.03 이다.
(d) 이분산 합성 자료에서 참 σ 정규화의 구간 점수(α 0.1)가 위약 정규화보다 낮다(방향 확인).
(e) @mw 의 평균 로그 반폭이 const 와 같다.
(f) 위약 순열의 재현성, 지역 안 순열, 순열 번호별 차이.
(g) cqr_pool 의 유한표본 분위 색인(다중도 1, 정수 다중도, 재표집 판).
(h) 위치 전용 λ 규약: 대상 구간의 로그 폭은 λ 와 무관하고 중심만 λ·q_0.5 만큼 옮긴다.
(i) 실패한 seed 의 처리(유효 seed 만 읽는다, 비유한 σ 거부, 설정 불일치)와 '판정 불가(적합 실패)'.
(j) LGU-B2 의 판정 분기(전이, 조건부 우세 두 경우, 미결정, 악화, 판정 불가).
(k) --precision-only 출력(precision_rows)에 delta, CI 끝점, 판정, 기준 점수, 반폭 절댓값 열이 없다(개정 1).
(l) 허용 표지 없이 실행(채점, 스모크, 집계)을 거부한다. 로컬 스레드 상한과 GPU 0 MiB 확인. (개정 1) 배정 밖 GPU 거부,
    LG_RESCALE=1 에도 로컬 규칙 적용, fit 설정 해시에 --threads 가 없다, 1단 재표집 seed 가 대상마다 다르다,
    2단 재표집의 대상별 실패 처리.
(n) 다중도가 모두 1 인 2단 재표집 통계가 ScoreStore 의 점 추정과 같다(격자, 구간, 보조 n 의 추출·분할 결합 포함).
(o) 임의 다중도에서 2단 통계가 수계산(다중도 가중 분위, 셀 가중·블록 등가중 평균)과 같다.
(p) 스모크·사전 점검 설정, 층 표지(순위 등분), 구간 끝점 역전 처리.
학습을 하는 시험(표지 HEAVY, LG_RUN_HEAVY=1 일 때만)
(m) 스모크 경로(--smoke --part all, 풀 없이, CPU, --s-gen 16)의 종료 코드 0 과 산출 파일. 실제 자료를 읽고 CPU 로 학습한다.
실행(로컬 GPU 서버, 사용자 지시 2026-09-30. CPU 만 쓴다. 스레드 2, nice 10):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 python3 -m pytest -q tests/test_h45_normconf.py
  (HEAVY 포함: 앞에 LG_RUN_HEAVY=1 을 붙인다)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # 시험은 GPU 를 쓰지 않는다
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

HEAVY = pytest.mark.skipif(os.environ.get("LG_RUN_HEAVY", "") != "1",
                           reason="학습을 하는 시험은 LG_RUN_HEAVY=1 일 때만 실행한다(공유 서버 CPU 보호)")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import polar.lgu_common as LC  # noqa: E402


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


h45 = _load("h45_normalized_conformal", "scripts/2_evaluation/h45_normalized_conformal.py")


def _synth(seed=0, n_eval=400, regions=("A", "B", "C", "D"), n_per=300, hetero=True, norms=("nflow", "cbq", "cfm"), seeds=(0, 1),
           perms=3, lams=(0.0, 0.25, 1.0), aux=False, true_sigma=True, shift=None):
    """합성 cells dict(h45.build_cells 와 같은 키). 로그 비 = σ(x)·ε, σ(x) = 0.15 + 0.5x(이분산) 또는 0.4."""
    from scipy.special import ndtri
    rng = np.random.RandomState(seed)
    K = len(regions); E0 = 3.0

    def sig_of(x):
        return 0.15 + 0.5 * x if hetero else np.full(len(x), 0.4)
    x_e = rng.rand(n_eval); sig_e = sig_of(x_e)
    s_e = rng.uniform(5.0, 15.0, n_eval); center = E0 * s_e
    y_e = center * np.exp(sig_e * rng.randn(n_eval))
    col_e = 1000 + rng.randint(0, 40, n_eval)
    g = np.repeat(np.array(regions), n_per); N = len(g)
    x_c = rng.rand(N); sig_c = sig_of(x_c)
    off = np.zeros(N) if shift is None else np.repeat(np.asarray(shift, float), n_per)
    c_u = off + sig_c * rng.randn(N)
    c_col = np.repeat(np.arange(K), n_per) * 100 + rng.randint(0, 30, N)
    C = dict(e_loc=np.arange(n_eval), e_y=y_e, e_s=s_e, e_center=center, e_col=col_e.astype(np.int64),
             e_bkey=np.array([f"T|{c}" for c in col_e]), e_early=rng.rand(n_eval) < 0.2, e_gpr=rng.rand(n_eval) < 0.3,
             e_probe=rng.rand(n_eval) < 0.5, e_dsrc=rng.uniform(1.0, 500.0, n_eval), e_lat=rng.uniform(60, 70, n_eval),
             e_lon=rng.uniform(100, 120, n_eval), c_loc=np.arange(N) + 10_000, c_region=g.astype(str), c_col=c_col.astype(np.int64),
             c_u=c_u, c_h37=np.abs(c_u + 0.05), c_early=rng.rand(N) < 0.2, E0_mk=np.full(K, E0))
    S = len(seeds)
    for nm in norms:
        if true_sigma:
            sc = np.stack([sig_c * np.exp(0.02 * rng.randn(N)) for _ in range(S)])
            st = np.stack([sig_e * np.exp(0.02 * rng.randn(n_eval)) for _ in range(S)])
        else:
            sc = np.ones((S, N)); st = np.ones((S, n_eval))
        C[f"sig_c__{nm}"] = sc; C[f"sig_t__{nm}"] = st
        if nm in ("nflow", "cfm"):
            zq = ndtri(np.array(LC.Q5))
            C[f"q5_c__{nm}"] = np.stack([zq[:, None] * sc[j][None, :] + 0.01 for j in range(S)])
            C[f"q5_t__{nm}"] = np.stack([zq[:, None] * st[j][None, :] + 0.01 for j in range(S)])
            C[f"e0k__{nm}"] = np.full(K, E0); C[f"e0f__{nm}"] = np.array(E0)
    meta = dict(target="T", perms=perms, lams=list(lams), norms=list(norms), seeds={nm: list(seeds) for nm in norms}, G=list(regions),
                E0=E0, alaska=regions[0], aux_n=[], aux_splits=[], sigma_clip=[0.02, 2.0])
    if aux:
        n = 10
        cpos = np.concatenate([rng.choice(np.where(g == k)[0], 60, replace=False) for k in regions for _ in range(2)])
        C[f"ax_r__{n}"] = c_u[cpos] + 0.03 * rng.randn(len(cpos)); C[f"ax_g__{n}"] = g[cpos].astype(str)
        C[f"ax_col__{n}"] = c_col[cpos].astype(np.int64); C[f"ax_cpos__{n}"] = cpos.astype(np.int64)
        perm = rng.permutation(n_eval)
        b1, b2 = np.sort(perm[: n_eval // 2]), np.sort(perm[n_eval // 2:])
        C["ax_b__1"] = b1; C["ax_En__1"] = np.array([[n, 0, 2.9], [n, 1, 3.1]])
        C["ax_b__2"] = b2; C["ax_En__2"] = np.array([[n, 0, 3.05], [n, 1, 2.95]])
        meta["aux_n"] = [n]; meta["aux_splits"] = [1, 2]
    C["meta"] = meta
    return C


def _score(C):
    specs = h45.method_specs(C)
    stores, rows = h45.score_from_specs(C, specs)
    return specs, stores, pd.DataFrame(rows)


def _nc(C):
    return int(max(np.max(C["e_col"]), np.max(C["c_col"]))) + 1


# ---------------------------------------------------------------- (a)
def test_a_sigma_one_equals_const():
    C = _synth(norms=("nflow",), true_sigma=False)
    _, stores, _ = _score(C)
    st = stores[("T|all", 0)]
    for m in LC.METRICS:
        a_ = st.mean(("nflow", "", 0, 0, 0), m, "cell"); b_ = st.mean(("const", "", 0, 0, -1), m, "cell")
        assert (np.isnan(a_) and np.isnan(b_)) or a_ == pytest.approx(b_, rel=0, abs=1e-12), m
        assert st.mean(("nflow", "", 0, 0, 0), m, "beq") == pytest.approx(st.mean(("const", "", 0, 0, -1), m, "beq"), rel=0, abs=1e-12) \
            or np.isnan(st.mean(("const", "", 0, 0, -1), m, "beq"))


# ---------------------------------------------------------------- (b)
def _h37_wquantile(scores, w, level, w_test=0.0):
    o = np.argsort(scores, kind="stable"); s, ww = np.asarray(scores, float)[o], np.asarray(w, float)[o]
    cum = np.cumsum(ww) / (ww.sum() + w_test)
    k = int(np.searchsorted(cum, level - 1e-12))
    return float(s[k]) if k < len(s) else np.inf


def _h37_group_weights(groups, total_each=1.0):
    ug, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    return total_each / cnt[inv], len(ug)


def test_b_const_h37_matches_h37_formula():
    C = _synth(regions=("A", "B", "C", "D", "E"), n_per=150)
    _, stores, runs = _score(C)
    K = len(C["meta"]["G"])
    w_g, _ = _h37_group_weights(C["c_region"], 1.0 / K)
    q_h37 = _h37_wquantile(C["c_h37"], w_g, 0.9, w_test=0.0)
    r = runs[(runs.method == "const@h37") & (runs.scope == "all")].iloc[0]
    assert r.q90 == pytest.approx(q_h37, abs=0)
    y, yhat = C["e_y"], C["e_center"]
    cov = float(np.mean(np.abs(np.log(y) - np.log(yhat)) <= q_h37))
    assert stores[("T|all", 0)].mean(("const@h37", "", 0, 0, -1), "cov10", "cell") == pytest.approx(cov, abs=1e-12)
    w = (yhat * np.exp(q_h37) - yhat * np.exp(-q_h37)).mean()
    assert stores[("T|all", 0)].mean(("const@h37", "", 0, 0, -1), "wid10", "cell") == pytest.approx(w, rel=1e-12)


# ---------------------------------------------------------------- (c)
@pytest.mark.parametrize("hetero", [False, True])
def test_c_coverage_exchangeable(hetero):
    C = _synth(seed=3, n_eval=6000, n_per=2500, hetero=hetero, norms=("nflow",), seeds=(0,), perms=1)
    _, stores, _ = _score(C)
    st = stores[("T|all", 0)]
    assert st.mean(("const", "", 0, 0, -1), "cov10", "cell") == pytest.approx(0.90, abs=0.03)
    assert st.mean(("nflow", "", 0, 0, 0), "cov10", "cell") == pytest.approx(0.90, abs=0.03)
    assert st.mean(("nflow", "", 0, 0, 0), "cov20", "cell") == pytest.approx(0.80, abs=0.03)


# ---------------------------------------------------------------- (d)
def test_d_true_sigma_beats_placebo():
    C = _synth(seed=5, n_eval=3000, n_per=1500, hetero=True, norms=("nflow",), seeds=(0,), perms=5)
    _, stores, _ = _score(C)
    st = stores[("T|all", 0)]
    is_true = st.mean(("nflow", "", 0, 0, 0), "is10", "cell")
    is_pl = np.mean([st.mean((f"nflow#p{j}", "", 0, 0, 0), "is10", "cell") for j in range(5)])
    assert is_true < is_pl
    assert st.mean(("nflow#p0", "", 0, 0, 0), "cov10", "cell") == pytest.approx(0.90, abs=0.05)   # 위약도 한계 커버리지는 유지


# ---------------------------------------------------------------- (e)
def test_e_mw_mean_log_halfwidth_equals_const():
    C = _synth(seed=7)
    _, stores, runs = _score(C)
    st = stores[("T|all", 0)]
    for nm in ("nflow", "cbq", "cfm"):
        for s in (0, 1):
            assert st.mean((f"{nm}@mw", "", 0, 0, s), "lhw10", "cell") == pytest.approx(st.mean(("const", "", 0, 0, -1), "lhw10", "cell"), rel=1e-10)
            assert st.mean((f"{nm}@mw", "", 0, 0, s), "lhw20", "cell") == pytest.approx(st.mean(("const", "", 0, 0, -1), "lhw20", "cell"), rel=1e-10)
    q_c = runs[(runs.method == "const") & (runs.scope == "all")].q90.iloc[0]
    q_m = runs[(runs.method == "nflow@mw") & (runs.scope == "all")].q90.iloc[0]
    assert q_c == q_m                                        # 보정 분위는 const 의 것을 쓴다


# ---------------------------------------------------------------- (f)
def test_f_placebo_perm():
    g = np.array(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
    p1, t1 = h45.placebo_perm("Lena", "nflow", 0, 3, g, 77)
    p2, t2 = h45.placebo_perm("Lena", "nflow", 0, 3, g, 77)
    np.testing.assert_array_equal(p1, p2); np.testing.assert_array_equal(t1, t2)
    assert np.array_equal(np.sort(p1), np.arange(len(g))) and np.array_equal(np.sort(t1), np.arange(77))
    np.testing.assert_array_equal(g[p1], g)                  # 지역 안 순열
    p3, t3 = h45.placebo_perm("Lena", "nflow", 0, 4, g, 77)
    assert not np.array_equal(p1, p3) and not np.array_equal(t1, t3)
    p4, _ = h45.placebo_perm("Lena", "cbq", 0, 3, g, 77)
    assert not np.array_equal(p1, p4)


# ---------------------------------------------------------------- (g)
def test_g_cqr_pool_index():
    rng = np.random.RandomState(0)
    for N in (5, 19, 100):
        s = rng.randn(N)
        for al in (0.1, 0.2, 0.5):
            assert h45.pooled_quantile_mult(s, np.ones(N), al) == LC.pooled_quantile(s, al)
            k = int(np.ceil((N + 1) * (1 - al)))
            exp = np.sort(s)[k - 1] if k <= N else np.inf
            assert h45.pooled_quantile_mult(s, np.ones(N), al) == exp
        m = rng.randint(0, 4, N)
        if m.sum() > 0:
            assert h45.pooled_quantile_mult(s, m, 0.1) == LC.pooled_quantile(np.repeat(s, m), 0.1)
    s = rng.randn(40); cols = rng.randint(0, 10, 40)
    W = rng.randint(0, 3, size=(7, 10)); W[3] = 0
    qb = h45.pooled_quantile_boot(s, cols, W, 0.1, chunk=3)
    for r in range(7):
        want = h45.pooled_quantile_mult(s, W[r, cols], 0.1)
        assert (np.isnan(want) and np.isnan(qb[r])) or qb[r] == want
    assert np.isnan(qb[3])
    assert np.all(h45.pooled_quantile_boot(s, cols, np.ones((2, 10)), 0.1) == LC.pooled_quantile(s, 0.1))


# ---------------------------------------------------------------- (h)
def test_h_loc_only_moves_center_only():
    C = _synth(seed=2)
    arrs = {lam: h45.cqr_arrays(C, "nflow", 0, lam) for lam in (0.0, 0.25, 1.0)}
    q5t = C["q5_t__nflow"][0]
    anc = np.log(3.0) + np.log(C["e_s"])
    for lam, arr in arrs.items():
        for ai in (0, 1):
            np.testing.assert_allclose(arr["a_hi"][ai] - arr["a_lo"][ai], arrs[1.0]["a_hi"][ai] - arrs[1.0]["a_lo"][ai], atol=1e-12)
        np.testing.assert_allclose(np.log(arr["med"]), anc + lam * q5t[2], atol=1e-12)
        np.testing.assert_allclose(arr["a_lo"][0] - np.log(arr["med"]), q5t[0] - q5t[2], atol=1e-12)
    np.testing.assert_allclose(arrs[1.0]["a_lo"][0], anc + q5t[0], atol=1e-12)   # λ = 1 은 학습기의 원래 구간


# ---------------------------------------------------------------- (i)
def test_i_failed_seeds(tmp_path):
    a = h45.parse_args(["--out-dir", str(tmp_path), "--normalizers", "nflow", "--seeds", "3"])
    rng = np.random.RandomState(0)
    N_cal, N_t = 30, 20
    df = pd.DataFrame(dict(loc_id=np.arange(100) + 500))
    D = SimpleNamespace(df=df)
    info = dict(cal_idx=np.arange(0, 30), t_idx=np.arange(50, 70), ev=np.arange(50, 70)[[1, 3, 5, 7, 9, 11]])
    p = h45.fit_paths(a, "T", "nflow"); p["sigma"].parent.mkdir(parents=True, exist_ok=True)
    cal_sigma = rng.uniform(0.1, 1, (3, N_cal)); cal_sigma[1] = np.nan
    tgt_sigma = rng.uniform(0.1, 1, (3, N_t)); tgt_sigma[1] = np.nan
    LC.atomic_npz(p["sigma"], groups=np.array(["A"]), E0=np.array(3.0), E0_mk=np.array([3.0]), E0_train_k=np.array([3.0]),
                  E0_train_full=np.array(3.0), cal_loc=df.loc_id.values[:30], cal_region=np.array(["A"] * 30), cal_col=np.zeros(30, np.int32),
                  cal_u=rng.randn(30), cal_q=rng.randn(3, 5, N_cal).astype(np.float32), cal_sigma=cal_sigma, tgt_loc=df.loc_id.values[50:70],
                  tgt_sigma=tgt_sigma, tgt_q5=rng.randn(3, 5, N_t).astype(np.float32), tgt_qgrid=np.zeros((3, 99, N_t), np.float32),
                  seeds=np.array([0, 1, 2]), valid_seeds=np.array([0, 2]), fit_info=np.array("[]"), crossing=np.array("[]"))
    LC.atomic_text(p["runs"], "target\nT\n")
    cfg = h45.fit_cfg(a, "nflow")
    LC.atomic_text(p["unit"], json.dumps(dict(status="partial", cfg=cfg, cfg_hash=LC.cfg_hash(cfg))))
    f, why = h45.load_fit(a, "T", "nflow", info, D)
    assert why == "" and f["seeds"] == [0, 2]
    np.testing.assert_array_equal(f["sig_c"], cal_sigma[[0, 2]])
    np.testing.assert_array_equal(f["sig_t"], tgt_sigma[[0, 2]][:, [1, 3, 5, 7, 9, 11]])
    assert f["q5_t"].shape == (2, 5, 6)
    # 설정이 다르면 읽지 않는다
    a2 = h45.parse_args(["--out-dir", str(tmp_path), "--normalizers", "nflow", "--seeds", "3", "--epochs", "7"])
    f2, why2 = h45.load_fit(a2, "T", "nflow", info, D)
    assert f2 is None and "설정 불일치" in why2
    # 유효 seed 의 σ 가 비유한이면 거부한다
    cal_sigma[2, 0] = np.nan
    with np.load(p["sigma"]) as z:
        arrs = {k: z[k] for k in z.files}
    arrs["cal_sigma"] = cal_sigma
    LC.atomic_npz(p["sigma"], **arrs)
    f3, why3 = h45.load_fit(a, "T", "nflow", info, D)
    assert f3 is None and "비유한" in why3
    # 유효 seed 가 없으면 '유효 seed 없음'
    arrs["valid_seeds"] = np.array([], np.int64)
    LC.atomic_npz(p["sigma"], **arrs)
    f4, why4 = h45.load_fit(a, "T", "nflow", info, D)
    assert f4 is None and "유효 seed 없음" in why4
    # 명세는 유효 seed 만 만든다. 유효 seed 1개인 풀 지역이 있으면 LGU-B2 는 '판정 불가(적합 실패)'
    C = _synth(norms=("nflow",), seeds=(2,), perms=2)
    specs = h45.method_specs(C, aux=False)
    assert {s["key"][4] for s in specs if s["key"][0] == "nflow"} == {2}
    assert h45.b2_verdict((-3, -1, -3, -1), [-1, -1, -1, -1], 0.9, False, 4) == "판정 불가(적합 실패)"


# ---------------------------------------------------------------- (j)
def test_j_b2_branches():
    V = h45.b2_verdict
    assert V((-3, -1, -2, -0.5), [-1, -2, -0.5, 0.3], 0.86, True, 4) == "전이"
    v = V((-3, -1, -2, -0.5), [-1, 0.2, -0.5, 0.3], 0.86, True, 4)
    assert v.startswith("조건부 우세") and "(2)" in v
    v = V((-3, -1, -2, -0.5), [-1, -2, -0.5, -0.3], 0.79, True, 4)
    assert v.startswith("조건부 우세") and "(3)" in v
    v = V((-3, -1, 2, 5), [-1, -2, -0.5, -0.3], 0.9, True, 4)
    assert v == "미결정"                                     # 한 가중만 0 미만
    assert V((-3, 1, -2, 0.5), [-1, -2, -0.5, -0.3], 0.9, True, 4) == "미결정"
    assert V((0.5, 2, 0.1, 3), [1, 2, 1, 1], 0.9, True, 4) == "악화"
    assert V((0.5, 2, -0.1, 3), [1, 2, 1, 1], 0.9, True, 4) == "미결정"
    assert V((-3, -1, -2, -0.5), [-1, -2], 0.9, True, 2) == "판정 불가"   # 풀 3지역 미만
    assert V((np.nan, -1, -2, -0.5), [-1, -2, -1], 0.9, True, 3) == "판정 불가"
    assert V((-3, -1, -2, -0.5), [-1, -2, 0.5], 0.9, True, 3, need_neg=2, n_min=2) == "전이"      # 3지역 평균의 기준


# ---------------------------------------------------------------- (k)
def test_k_precision_rows_have_no_delta():
    rng = np.random.RandomState(0)
    e = [dict(contrast="LGU-B2 is10: nflow − 위약(nflow)", pool="Canada", n=0, d2=(rng.randn(200) - 5, rng.randn(200) - 4), ref=(100.0, 90.0))]
    rows = h45.precision_rows(e, 200)
    df = pd.DataFrame(rows)
    assert set(df.columns) == set(h45.PREC_COLS)
    for bad in ("delta", "delta_beq", "ci_lo", "ci_hi", "verdict", "p_boot", "ref_score", "half_width"):
        assert bad not in df.columns
    assert np.all(df.half_width_rel > 0) and np.all(np.isfinite(df.half_width_rel))
    assert set(df.main_rule_5pct) <= {"검정 가능", "검정 불가(정밀도 미달)"}


# ---------------------------------------------------------------- (l)
def test_l_permission_and_local_rules(monkeypatch, tmp_path):
    monkeypatch.delenv("LG_RESCALE", raising=False)
    for argv in (["--part", "score"], ["--smoke", "--part", "all"], ["--summarize-only"], ["--precheck"]):
        with pytest.raises(SystemExit) as ei:
            h45.main(argv + ["--out-dir", str(tmp_path)])
        assert "거부" in str(ei.value)
    monkeypatch.setattr(h45.os, "nice", lambda inc: 0)
    a = h45.parse_args(["--allow-local", "--workers", "3", "--threads", "4", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        h45.local_guard(a, ["score"], True)                  # 3 × 4 = 12 > 8
    a = h45.parse_args(["--allow-local", "--workers", "2", "--threads", "4", "--gpus", "2", "--out-dir", str(tmp_path)])
    h45.local_guard(a, ["fit", "score"], True)               # 2 × 4 = 8, GPU 1 × 4 = 4
    a = h45.parse_args(["--allow-local", "--workers", "1", "--threads", "4", "--gpus", "2,1", "--procs-per-gpu", "3", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        h45.local_guard(a, ["fit"], False)                   # GPU 2 × 3 × 4 = 24 > 8
    a = h45.parse_args(["--allow-local", "--gpus", "2,1", "--out-dir", str(tmp_path)])
    monkeypatch.setattr(h45.LC, "gpu_memory_used", lambda ids=None, timeout=30: {2: 1200.0, 1: 0.0})
    monkeypatch.setattr(h45.LC, "free_gpus", lambda ids, max_mib=0.0: [1])
    assert h45.usable_gpus(a) == ["1"]
    monkeypatch.setattr(h45.LC, "free_gpus", lambda ids, max_mib=0.0: [])
    with pytest.raises(SystemExit):
        h45.usable_gpus(a)
    # (개정 1) 배정 밖 GPU 는 경고가 아니라 거부
    a = h45.parse_args(["--allow-local", "--gpus", "5", "--workers", "1", "--threads", "4", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        h45.local_guard(a, ["fit"], False)
    with pytest.raises(SystemExit):
        h45.usable_gpus(a)
    # (개정 1) LG_RESCALE=1 로 허용된 실행에도 스레드 상한을 적용한다
    monkeypatch.setenv("LG_RESCALE", "1")
    a = h45.parse_args(["--workers", "3", "--threads", "4", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        h45.local_guard(a, ["score"], True)
    assert h45.local_guard(h45.parse_args(["--workers", "2", "--threads", "4", "--out-dir", str(tmp_path)]), ["score"], True) == 8
    # (개정 1) fit 설정 해시에 --threads 가 들어가지 않는다
    a1 = h45.parse_args(["--allow-local", "--threads", "4", "--out-dir", str(tmp_path)])
    a2 = h45.parse_args(["--allow-local", "--threads", "2", "--out-dir", str(tmp_path)])
    assert LC.cfg_hash(h45.fit_cfg(a1, "cbq")) == LC.cfg_hash(h45.fit_cfg(a2, "cbq"))
    # (개정 1) 1단 재표집 seed 는 대상·범위마다 다르고 같은 대상·범위 안에서는 같다
    assert h45.Summ.boot_seed("Russia_W", "all") != h45.Summ.boot_seed("Russia_E", "all")
    assert h45.Summ.boot_seed("Lena", "all") == h45.Summ.boot_seed("Lena", "all") != h45.Summ.boot_seed("Lena", "aux")
    # (개정 1) 2단 재표집: 한 대상의 예외는 그 대상만 실패로 적는다(풀 없는 경로)
    def fake_job(path, W, chunk):
        if "bad" in path:
            raise ValueError("boom")
        return "ok_t", {"k": 1}, 0.1
    monkeypatch.setattr(h45, "_ts_job", fake_job)
    a0 = h45.parse_args(["--allow-local", "--workers", "0", "--out-dir", str(tmp_path)])
    T2, secs, tf = h45.run_twostage(a0, {"ok_t": tmp_path / "ok.npz", "bad_t": tmp_path / "bad.npz"}, None)
    assert set(T2) == {"ok_t"} and [t for t, _ in tf] == ["bad_t"] and "boom" in tf[0][1]


# ---------------------------------------------------------------- (n)
def _store_value(st, key, metric, w):
    if metric == "se":
        return st.rmse(key, w)
    return st.mean(key, metric, w)


def test_n_twostage_unit_mult_equals_point():
    C = _synth(seed=11, aux=True, perms=2)
    specs, stores, _ = _score(C)
    W = np.ones((3, _nc(C)), np.uint8)
    T2 = h45.twostage_from_specs(C, specs, W, chunk=2)
    st = stores[("T|all", 0)]; ne = stores[("T~noearly", 0)]
    n_checked = 0
    for (store, key), res in T2.items():
        if store == "aux":
            continue
        s_ = st if store == "all" else ne
        for mt in LC.METRICS_2STAGE:
            for wi, w in enumerate(("cell", "beq")):
                want = _store_value(s_, key, mt, w)
                got = res[mt][wi]
                assert np.allclose(got, want, rtol=1e-10, atol=1e-10, equal_nan=True), (store, key, mt, w)
        n_checked += 1
    assert n_checked > 40
    assert any(k[1][0].startswith("cqr_pool") for k in T2 if k[0] == "all")
    # 보조 n: 분할 안 추출 평균 → 분할 평균
    for m, s in (("const", -1), ("nflow", 0), ("cbq", 1)):
        res = T2[("aux", (m, 10, s))]
        for mt in LC.METRICS_2STAGE:
            for wi, w in enumerate(("cell", "beq")):
                per = [np.mean([_store_value(stores[("T", sp)], (m, "", 10, d, s), mt, w) for d in (0, 1)]) for sp in (1, 2)]
                assert np.allclose(res[mt][wi], np.mean(per), rtol=1e-10, atol=1e-10), (m, s, mt, w)


# ---------------------------------------------------------------- (o)
def test_o_twostage_random_mult_matches_hand():
    C = _synth(seed=13, n_eval=300, n_per=200, perms=1)
    specs = h45.method_specs(C, aux=False)
    rng = np.random.RandomState(1)
    W = rng.randint(0, 3, size=(6, _nc(C))).astype(np.uint8)
    y = C["e_y"]; cols = C["e_col"]; ub, inv = np.unique(cols, return_inverse=True)
    for want_key in (("nflow", "", 0, 0, 0), ("cqr_pool:nflow@lam1.0", "", 0, 0, 1), ("cqr_hier:cfm@lam0.25", "", 0, 0, 0)):
        sp = [s for s in specs if s["key"] == want_key and s["store"] == "all"][0]
        res = h45.twostage_from_specs(C, [sp], W)[("all", sp["key"])]
        for r in range(W.shape[0]):
            m = W[r, sp["cal"]["c"]].astype(float)
            if sp["kind"] == "grid":
                q9, q8 = LC.hier_quantiles(sp["cal"]["s"], sp["cal"]["g"], [0.9, 0.8], mult=m)
                lo = sp["center"] * np.exp(-q9 * sp["b"]); hi = sp["center"] * np.exp(q9 * sp["b"])
            else:
                if sp["cal"]["pooled"]:
                    q9 = h45.pooled_quantile_mult(sp["cal"]["s"][0], m, 0.1)
                else:
                    q9 = LC.hier_quantiles(sp["cal"]["s"][0], sp["cal"]["g"], [0.9], mult=m)[0]
                lo, hi = h45.interval_ends(sp["a_lo"][0], sp["a_hi"][0], sp["med"], q9)
            IS = (hi - lo) + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
            cov = ((lo <= y) & (y <= hi)).astype(float)
            wc = W[r, cols].astype(float)
            assert res["is10"][0][r] == pytest.approx((wc * IS).sum() / wc.sum(), rel=1e-10)
            assert res["cov10"][0][r] == pytest.approx((wc * cov).sum() / wc.sum(), rel=1e-10)
            bm = np.bincount(inv, IS) / np.bincount(inv); wb = W[r, ub].astype(float)
            assert res["is10"][1][r] == pytest.approx((wb * bm).sum() / wb.sum(), rel=1e-10)


# ---------------------------------------------------------------- (p)
def test_p_settings_bins_inversion(tmp_path):
    a = h45.parse_args(["--smoke", "--out-dir", str(tmp_path)])
    assert a.TARGETS == ["Canada", "Russia_W"] and a.SEEDS == [0] and a.PERMS == 2 and a.nboot == 200 and a.epochs == 3
    assert a.AUX_EFF == ["Canada"] and a.N_AUX == [10] and a.SPLITS == [1] and a.DRAWS == 1
    assert a.TAG == "lgub_smoke" and a.PREFIX == "lgu_b_smoke"
    b = h45.parse_args(["--precheck", "--out-dir", str(tmp_path)])
    assert b.TARGETS == ["Canada"] and b.SEEDS == [0] and b.PERMS == 10 and b.precision_only and b.nboot == 2000 and b.epochs == 100
    c = h45.parse_args(["--n-grid", "10,40", "--targets", "Lena:x,Canada", "--out-dir", str(tmp_path)])
    assert c.N_AUX == [10, 40] and c.TARGETS == ["Lena", "Canada"] and c.PREFIX == "lgu_b"
    with pytest.raises(SystemExit):
        h45.parse_args(["--targets", "AL-3:i"])
    assert h45.lam_label(1) == "1.0" and h45.lam_label(0.25) == "0.25" and h45.lam_label(0) == "0.0"
    v = np.array([5.0, 1.0, np.nan, 3.0, 2.0, 4.0, 0.5, 7.0, 6.0, 8.0, 9.0])
    b5 = h45.rank_bins(v, 5)
    assert b5[2] == -1 and np.bincount(b5[b5 >= 0]).tolist() == [2, 2, 2, 2, 2]
    assert np.all(np.diff(b5[np.argsort(np.where(np.isnan(v), np.inf, v))][:-1]) >= 0)
    lo, hi = h45.interval_ends(np.array([0.0, 0.0]), np.array([0.2, 0.2]), np.array([1.1, 1.1]), -0.5)
    np.testing.assert_allclose(lo, [1.1, 1.1]); np.testing.assert_allclose(hi, [1.1, 1.1])
    lo, hi = h45.interval_ends(np.array([0.0]), np.array([0.2]), np.array([1.1]), 0.1)
    np.testing.assert_allclose(lo, np.exp(-0.1)); np.testing.assert_allclose(hi, np.exp(0.3))
    C = _synth(seed=4)
    ms = h45.strata_masks(C, C["e_center"])
    assert [m[0] for m in ms] == ["#gpr", "#probe", "#early", "#w1", "#w2", "#w3", "#w4", "#w5", "#d1", "#d2", "#d3"]
    assert sum(int(m.sum()) for n_, m in ms if n_.startswith("#w")) == len(C["e_y"])


# ---------------------------------------------------------------- (m) HEAVY
@HEAVY
def test_m_smoke_path(tmp_path):
    rc = h45.main(["--smoke", "--part", "all", "--allow-local", "--workers", "0", "--threads", "2", "--s-gen", "16", "--out-dir", str(tmp_path)])
    assert rc == 0
    for nm in ("intervals", "tests", "cal", "strata", "pit", "cells", "timing", "failed", "meta"):
        assert (tmp_path / f"lgu_b_smoke_{nm}.{'json' if nm == 'meta' else 'csv'}").exists(), nm
    t = pd.read_csv(tmp_path / "lgu_b_smoke_tests.csv")
    assert {"LGU-B1", "LGU-B2", "LGU-B3", "LGU-B4", "LGU-B6"} <= set(t.test)
    rc2 = h45.main(["--smoke", "--summarize-only", "--precision-only", "--allow-local", "--workers", "0", "--threads", "2", "--s-gen", "16",
                    "--out-dir", str(tmp_path)])
    assert rc2 == 0
    p = pd.read_csv(tmp_path / "lgu_b_smoke_precision.csv")
    assert "delta" not in p.columns and len(p) > 0
