"""src/polar/lgu_common.py 단위 시험(계획 docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 2절, spec_common 11절).

가벼운 시험(학습 없음, 합성 자료, torch 를 적재하지 않는다)
(a) prep_stats, prep_apply, ls_E 가 h40 의 값과 같다.
(b) q_from_samples, rearrange, crossing_stats, grid_index 의 기지값.
(c) grid_from_center 의 대칭, 단조화, interval_of.
(d) 구간 점수 항등식 IS_α = (2/α)(ρ_(α/2)(y − L) + ρ_(1−α/2)(y − U)), 정규 예측 분포의 CRPS 닫힌 형태와의 상대 차 2 % 이내,
    보정된 합성 자료의 PIT 평균 0.5 ± 0.02, pit_grid 와 np.interp 의 일치(동률 격자 포함), score_cells·score_interval_only 의 키와 항등.
(e) ScoreStore 저장·적재 왕복(inf 포함), 셀 가중·블록 등가중 평균, merge, 여러 파일 병합.
(f) wquantile 기지값, hier_quantiles 와 wquantile 의 일치, hier_quantiles_boot 가 반복 호출과 같은 값, pooled_quantile 색인.
(g) BlockIndex 와 global_mult(지역별 행 합 = 블록 수, 같은 seed 에서 같은 값).
(h) one_stage_delta 의 수계산 대조(구간 점수, RMSE), 키 평균의 NaN 처리, agg_* 의 inf 처리, verdict4 의 다섯 경우,
    eq_state 의 정밀도 규칙. (개정 1) (n, 추출) 단위 짝지음: 한쪽에만 있는 단위는 빠지고 n_unpaired_* 로 세며, NaN 키는 n_nan_keys_* 로 센다.
(i) emulation_plan 의 재현성, 제외 규칙, 라벨 블록과 채점 블록의 비중복, E0_mk.
(j) loc_only(λ = 1)은 항등, λ = 0 의 중앙값은 0, 폭은 λ 와 무관.
(m) block_val_mask 의 몫과 대체 규칙, row_weights.
(n) phys_log_samples 의 재현성, 유한성, TDD 비례.
(o) require_permission 이 허용 표지 없이 거부한다. peek_threads, set_thread_env 의 규칙.
(p) 조각 도우미(cfg_hash, unit_state, atomic_*), load_flags, nearest_km, center_coef.
(r) GPU 확인 도우미(gpu_memory_used 의 nvidia-smi 출력 해석, free_gpus 의 0 MiB 규칙), fit_gen 의 입력 확인(학습 전 ValueError).
(s) (개정 1) is_env_error 의 분류, cal_groups(로그 비가 정의되는 셀 수 기준), check_blockindex 의 쓰기·대조, 실행 중 스레드 등록부
    (claim_threads: 합계 상한 거부, 죽은 pid 항목 정리, release), atomic_text 의 실패 시 임시 파일 삭제.
학습을 하는 시험(표지 HEAVY, LG_RUN_HEAVY=1 일 때만. torch 스레드 2)
(k) legacy 설정의 fit_gen + gen_samples(S = 64, chunk 4096)가 tab_models.fit_predict 의 samples 와 최대 절대 차 1e-5 이내
    (cfm, ddpm, nflow. 합성 자료 n = 600, d = 5, epochs 3).
(l) nflow_quantiles 가 표본 200,000개(명세의 20,000개보다 많다. 꼬리 분위의 표집 오차를 줄이기 위해서다)의 경험 분위와 0.05·SD 이내,
    τ 에 단조, 위치 고정 변형의 τ = 0.5 분위가 0(1e-6 이내), nflow_logpdf 의 격자 적분이 1 ± 1e-3, CDF 표의 단조성.
(q) cfm 의 표본 분위와 결정적 분위 사상, fit_failed, CatBoost 다분위(cb_multiquantile)의 동작과 손실 방식 표기.
(t) (개정 1) CUDA_VISIBLE_DEVICES 가 없는 프로세스에서 fit_gen(device=None)은 CPU 를 쓴다.
실행(로컬 GPU 서버, 사용자 지시 2026-09-30. CPU 만 쓴다. 스레드 2, nice 10):
  LG_RUN_HEAVY=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      nice -n 10 python3 -m pytest -q tests/test_lgu_common.py
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
                           reason="학습을 하는 시험은 LG_RUN_HEAVY=1 일 때만 실행한다(공유 서버 CPU 보호. 수정 단계의 로컬 단위 시험에서 실행)")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import polar.lgu_common as LC  # noqa: E402
import polar.h4_common as H4  # noqa: E402


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _crps_normal(y, mu, sd):
    from scipy.stats import norm
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


# ---------------------------------------------------------------- (a)
def test_a_prep_matches_h40():
    H = _load("h40_label_grid", "scripts/3_deep_learning/h40_label_grid.py")
    rng = np.random.RandomState(0)
    X = rng.randn(200, 6) * np.array([1, 10, 0.1, 5, 2, 3]) + np.array([0, 5, -1, 100, 0, 3])
    X[rng.rand(*X.shape) < 0.1] = np.nan
    X[:, 5] = np.nan                                        # 전부 결측인 열
    s1, s2 = LC.prep_stats(X), H._prep_stats(X)
    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a, b)
    Xt = rng.randn(50, 6)
    Xt[rng.rand(*Xt.shape) < 0.2] = np.nan
    np.testing.assert_array_equal(LC.prep_apply(Xt, s1), H._prep_apply(Xt, s2))
    y = rng.rand(30) * 100; s = rng.rand(30) * 30; s[3] = np.nan; s[4] = -1.0
    assert LC.ls_E(y, s) == H.ls_E(y, s)


# ---------------------------------------------------------------- (b)
def test_b_quantile_grid_basics():
    S = np.tile(np.arange(101, dtype=float)[:, None], (1, 3))
    Q = LC.q_from_samples(S)
    assert Q.shape == (99, 3)
    np.testing.assert_allclose(Q[:, 0], LC.TAUS * 100, atol=1e-9)
    S2 = S.copy(); S2[5, 1] = np.nan
    Q2 = LC.q_from_samples(S2)
    assert np.isnan(Q2[:, 1]).all() and np.isfinite(Q2[:, [0, 2]]).all()
    Qc = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 2.0], [3.0, 3.0]])        # 열 0 에 역전 1쌍(1 → 0.5)
    cs = LC.crossing_stats(Qc)
    assert cs["frac_cells"] == 0.5 and cs["n_cross_pairs"] == 1
    assert abs(cs["max_drop"] - 0.5) < 1e-12 and abs(cs["frac_pairs"] - 1 / 6) < 1e-12
    R = LC.rearrange(Qc)
    assert (np.diff(R, axis=0) >= 0).all() and LC.crossing_stats(R)["n_cross_pairs"] == 0
    assert LC.grid_index(0.05) == 4 and LC.grid_index(0.95) == 94 and LC.grid_index(0.5) == 49
    with pytest.raises(ValueError):
        LC.grid_index(0.025)


# ---------------------------------------------------------------- (c)
def test_c_grid_from_center_and_interval():
    rng = np.random.RandomState(1)
    center = rng.uniform(20, 80, 7)
    half = np.cumsum(rng.uniform(0.01, 0.05, 49))
    Q = LC.grid_from_center(center, half)
    assert Q.shape == (99, 7)
    np.testing.assert_allclose(Q[49], center)
    np.testing.assert_array_equal(LC.median_of(Q), Q[49])
    for j in range(1, 50):
        np.testing.assert_allclose(Q[49 - j] * Q[49 + j], center ** 2, rtol=1e-12)
        np.testing.assert_allclose(np.log(Q[49 + j] / center), half[j - 1], rtol=1e-10)
    lo, hi = LC.interval_of(Q, 0.1)
    np.testing.assert_allclose(lo, center * np.exp(-half[44])); np.testing.assert_allclose(hi, center * np.exp(half[44]))
    lo2, hi2 = LC.interval_of(Q, 0.2)
    np.testing.assert_allclose(lo2, center * np.exp(-half[39])); np.testing.assert_allclose(hi2, center * np.exp(half[39]))
    hbad = half.copy(); hbad[10] = hbad[9] - 0.5                           # 단조가 아닌 반폭
    Qb = LC.grid_from_center(center, hbad)
    assert (np.diff(Qb, axis=0) >= 0).all()
    np.testing.assert_allclose(Qb[49 + 11], center * np.exp(hbad[9]))
    H2 = np.tile(half[:, None], (1, 7)) * np.linspace(0.5, 1.5, 7)[None, :]
    Q2 = LC.grid_from_center(center, H2)
    np.testing.assert_allclose(Q2[94], center * np.exp(H2[44]))
    Qi = LC.grid_from_center(center[:2], np.full(49, np.inf))              # 무한 반폭
    assert (Qi[0] == 0).all() and np.isinf(Qi[98]).all()


# ---------------------------------------------------------------- (d)
def test_d_scores_identities():
    from scipy.special import ndtri
    rng = np.random.RandomState(2)
    n = 4000
    y = rng.lognormal(3.5, 0.4, n)
    lo = y * np.exp(rng.normal(0, 0.3, n)) * 0.8
    hi = lo * np.exp(rng.uniform(0.1, 0.8, n))
    for a in (0.1, 0.2):
        p = LC.interval_parts(y, lo, hi, a)
        ident = (2 / a) * (LC.pinball(y, lo, a / 2) + LC.pinball(y, hi, 1 - a / 2))
        np.testing.assert_allclose(p["total"], ident, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(p["total"], p["width"] + p["under"] + p["over"], rtol=1e-12)
    p = LC.interval_parts(np.array([10.0]), np.array([0.0]), np.array([np.inf]), 0.1)
    assert np.isinf(p["total"][0]) and np.isinf(p["width"][0]) and p["inf"][0] == 1
    mu = rng.normal(50, 5, n); sd = rng.uniform(5, 15, n)
    yn = mu + sd * rng.randn(n)
    Q = mu[None, :] + sd[None, :] * ndtri(LC.TAUS)[:, None]
    c_grid, c_true = LC.crps_grid(yn, Q), _crps_normal(yn, mu, sd)
    assert abs(c_grid.mean() / c_true.mean() - 1) < 0.02
    assert np.median(np.abs(c_grid / c_true - 1)) < 0.02
    pit = LC.pit_grid(yn, Q)
    assert abs(np.nanmean(pit) - 0.5) < 0.02
    ref = np.array([np.interp(yn[i], Q[:, i], LC.TAUS, left=0.005, right=0.995) for i in range(300)])
    np.testing.assert_allclose(pit[:300], ref, atol=1e-12)
    ys = np.array([Q[0, 0] - 1.0, Q[0, 1], Q[98, 2] + 1.0, Q[98, 3]])      # 격자 밖과 끝점
    ps = LC.pit_grid(ys, Q[:, :4])
    np.testing.assert_allclose(ps, [np.interp(ys[i], Q[:, i], LC.TAUS, left=0.005, right=0.995) for i in range(4)])
    Qt = np.sort(np.round(np.random.RandomState(21).uniform(0, 10, (99, 6)), 0), axis=0)   # 동률이 많은 격자
    yt = np.array([Qt[0, 0], Qt[50, 1], Qt[98, 2], 3.0, Qt[10, 4] + 0.3, -1.0])
    np.testing.assert_allclose(LC.pit_grid(yt, Qt), [np.interp(yt[i], Qt[:, i], LC.TAUS, left=0.005, right=0.995) for i in range(6)],
                               rtol=0, atol=1e-12)
    hcnt = LC.pit_hist(pit)
    assert hcnt.sum() == n and len(hcnt) == LC.PIT_BINS
    sc = LC.score_cells(yn, Q)
    assert set(LC.METRICS) <= set(sc) and "pit" in sc and sc["valid"].all()
    np.testing.assert_allclose(sc["crps"], c_grid)
    np.testing.assert_allclose(sc["se"], (Q[49] - yn) ** 2)
    np.testing.assert_allclose(sc["bias"], Q[49] - yn)
    lo10, hi10 = LC.interval_of(Q, 0.1)
    np.testing.assert_allclose(sc["cov10"], ((lo10 <= yn) & (yn <= hi10)).astype(float))
    np.testing.assert_allclose(sc["is10_w"], sc["wid10"])
    np.testing.assert_allclose(sc["pb05"], LC.pinball(yn, Q[4], 0.05))
    assert abs(sc["cov10"].mean() - 0.9) < 0.03 and abs(sc["cov20"].mean() - 0.8) < 0.03
    assert (sc["inf10"] == 0).all()
    lo20, hi20 = LC.interval_of(Q, 0.2)
    si = LC.score_interval_only(yn, lo10, hi10, lo20, hi20, Q[49])
    np.testing.assert_allclose(si["is10"], sc["is10"]); np.testing.assert_allclose(si["cov20"], sc["cov20"])
    np.testing.assert_allclose(si["se"], sc["se"])
    assert np.isnan(si["crps"]).all() and np.isnan(si["pit"]).all()
    yb = yn.copy(); yb[0] = np.nan
    Qn = Q.copy(); Qn[:, 1] = np.nan
    sb = LC.score_cells(yb, Qn)
    assert np.isnan(sb["is10"][:2]).all() and not sb["valid"][0] and sb["valid"][1]


# ---------------------------------------------------------------- (e)
def test_e_scorestore_roundtrip(tmp_path):
    rng = np.random.RandomState(3)
    n = 60
    blk = rng.randint(0, 7, n) * 100 + 5
    cols = blk // 100 + 10
    y = rng.lognormal(3, 0.3, n)
    Q = np.sort(y[None, :] * np.exp(rng.normal(0, 0.3, (99, n))), axis=0)
    kA, kB, kC = ("B4", "", 10, 0, -1), ("iii", "", 10, 0, -1), ("P0", "", 0, 0, -1)
    st = LC.ScoreStore("Canada", 1, blk, cols, meta=dict(n_eval=n))
    sc = LC.score_cells(y, Q)
    st.add(kA, sc)
    Q2 = Q.copy(); Q2[94, :3] = np.inf
    st.add(kB, LC.score_cells(y, Q2))
    p = LC.save_score_stores([st], tmp_path / "a.npz")
    s2 = LC.load_score_stores(p)[("Canada", 1)]
    assert s2.keys == st.keys and s2.meta["n_eval"] == n
    np.testing.assert_array_equal(s2.blocks, st.blocks); np.testing.assert_array_equal(s2.cols, st.cols)
    np.testing.assert_array_equal(s2.ncell, st.ncell)
    for k in st.keys:
        a, c = st.get(k); b, d = s2.get(k)
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(c, d)
    assert abs(st.mean(kA, "is10") - sc["is10"].mean()) < 1e-9
    bm = pd.Series(sc["is10"]).groupby(blk).mean().mean()
    assert abs(st.mean(kA, "is10", "beq") - bm) < 1e-9
    assert abs(st.rmse(kA) - np.sqrt(np.mean(sc["se"]))) < 1e-9
    assert np.isinf(st.mean(kB, "is10"))
    st3 = LC.ScoreStore("Canada", 1, blk, cols)
    st3.add(kC, sc)
    s2.merge(st3)
    assert kC in s2 and len(s2) == 3
    p3 = LC.save_score_stores([st3], tmp_path / "b.npz")
    assert len(LC.load_score_stores([p, p3])[("Canada", 1)]) == 3
    assert not list(tmp_path.glob("*.tmp*"))
    with pytest.raises(ValueError):
        LC.ScoreStore("x", 1, blk, np.arange(n))                          # 같은 블록에 다른 열 번호
    with pytest.raises(RuntimeError):
        s2.add(kA, sc)                                                    # 적재본은 셀 순서가 없다


# ---------------------------------------------------------------- (f)
def test_f_wquantile_and_hier_boot():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    assert LC.wquantile(s, np.ones(4), 0.5) == 2.0
    assert LC.wquantile(s, np.ones(4), 0.75) == 3.0
    assert LC.wquantile(s, np.ones(4), 0.9) == 4.0
    assert np.isinf(LC.wquantile(s, np.ones(4), 0.9, w_test=1.0))
    assert LC.wquantile(s, [0, 0, 0, 1], 0.1) == 4.0
    rng = np.random.RandomState(4)
    N = 300
    sc = rng.exponential(1.0, N)
    g = rng.choice(np.array(["A", "B", "C"]), N, p=[0.6, 0.3, 0.1])
    cols = rng.randint(0, 25, N)
    lv = LC.LEVELS
    w = LC.hier_cell_weights(g)
    for r in "ABC":
        assert abs(w[g == r].sum() - 1 / 3) < 1e-12
    np.testing.assert_array_equal(LC.hier_quantiles(sc, g, lv), [LC.wquantile(sc, w, lev) for lev in lv])
    W = rng.poisson(1.0, (40, 25)).astype(np.uint8)
    W[7] = 0
    W[9, cols[g == "C"]] = 0                                             # 한 지역의 다중도가 모두 0(K 가 줄어든다)
    qb = LC.hier_quantiles_boot(sc, g, cols, W, lv, chunk=7)
    ref = np.array([LC.hier_quantiles(sc, g, lv, mult=W[r, cols]) for r in range(40)])
    np.testing.assert_array_equal(qb, ref)
    assert np.isnan(qb[7]).all() and np.isfinite(qb[9]).all()
    wm = LC.hier_cell_weights(g, mult=W[9, cols])
    assert abs(wm.sum() - 1) < 1e-12 and (wm[g == "C"] == 0).all()
    assert LC.pooled_quantile(np.arange(1, 20, dtype=float), 0.1) == 18.0
    assert np.isinf(LC.pooled_quantile(np.arange(1, 5, dtype=float), 0.1))


# ---------------------------------------------------------------- (g)
def test_g_global_mult():
    df = pd.DataFrame(dict(macro=["A"] * 12 + ["B"] * 6, block=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 1, 2, 2, 3, 3]))
    bi = LC.BlockIndex(df)
    assert bi.n_cols == 9 and bi.regions == ["A", "B"] and bi.keys[0] == "A|1" and bi.keys[6] == "B|1"
    assert bi.cols([12])[0] == 6 and bi.to_frame().n_cells.sum() == len(df)
    W = LC.global_mult(bi, nboot=50, seed=3)
    assert W.shape == (50, 9) and W.dtype == np.uint8
    for r, nb in (("A", 6), ("B", 3)):
        c = np.where(bi.region_of_col == r)[0]
        assert (W[:, c].astype(int).sum(1) == nb).all()
    np.testing.assert_array_equal(W, LC.global_mult(bi, 50, 3))
    assert not np.array_equal(W, LC.global_mult(bi, 50, 4))


# ---------------------------------------------------------------- (h)
def test_h_one_stage_delta_and_verdicts():
    rng = np.random.RandomState(6)
    n = 80
    blk = rng.randint(0, 9, n)
    y = rng.lognormal(3, 0.3, n)
    stores = {}
    for sp in (1, 2):
        st = LC.ScoreStore("T", sp, blk, None)
        for sd in (0, 1):
            QA = np.sort(y[None, :] * np.exp(rng.normal(0, 0.25, (99, n))), 0)
            st.add(("A", "", 10, 0, sd), LC.score_cells(y, QA))
        QB = np.sort(y[None, :] * np.exp(rng.normal(0.1, 0.3, (99, n))), 0)
        st.add(("B", "", 10, 0, -1), LC.score_cells(y, QB))
        stores[sp] = st
    fA, fB = (lambda k: k[0] == "A"), (lambda k: k[0] == "B")
    res = LC.one_stage_delta(stores, fA, fB, "is10", nboot=200, seed=11)
    dd, pts, dd_se, pts_se = [], [], [], []
    for sp, st in stores.items():
        W = H4.boot_weights(st.nb, 200, H4.seed_of(11, sp))
        SA, CA = st.matrices(st.select(fA), "is10"); SB, CB = st.matrices(st.select(fB), "is10")
        dd.append(((SA @ W.T) / (CA @ W.T)).mean(0) - ((SB @ W.T) / (CB @ W.T)).mean(0))
        pts.append((SA.sum(1) / CA.sum(1)).mean() - (SB.sum(1) / CB.sum(1)).mean())
        EA, _ = st.matrices(st.select(fA), "se"); EB, _ = st.matrices(st.select(fB), "se")
        dd_se.append(np.sqrt((EA @ W.T) / (CA @ W.T)).mean(0) - np.sqrt((EB @ W.T) / (CB @ W.T)).mean(0))
        pts_se.append(np.sqrt(EA.sum(1) / CA.sum(1)).mean() - np.sqrt(EB.sum(1) / CB.sum(1)).mean())
    dist = np.mean(dd, 0)
    np.testing.assert_allclose(res["dist"], dist, rtol=1e-12)
    assert abs(res["delta"] - np.mean(pts)) < 1e-9 and res["n_splits"] == 2
    assert abs(res["ci_lo"] - np.percentile(dist, 2.5)) < 1e-9 and abs(res["ci_hi"] - np.percentile(dist, 97.5)) < 1e-9
    assert res["p_boot"] == H4.boot_p(res["dist"]) and res["n_keys_A"] == 4 and res["n_keys_B"] == 2
    r2 = LC.one_stage_delta(stores, fA, fB, "se", nboot=200, seed=11)
    np.testing.assert_allclose(r2["dist"], np.mean(dd_se, 0), rtol=1e-12)
    assert abs(r2["delta"] - np.mean(pts_se)) < 1e-9
    for st in stores.values():                                          # 계산 실패 키(NaN)는 키 평균에서 빠진다(LG 규칙의 nanmean)
        st.add_sums(("A", "", 10, 0, 9), np.full((len(LC.METRICS), st.nb), np.nan), st.get(("B", "", 10, 0, -1))[1])
    r3 = LC.one_stage_delta(stores, fA, fB, "is10", nboot=200, seed=11)
    np.testing.assert_allclose(r3["dist"], res["dist"], rtol=1e-12)
    assert abs(r3["delta"] - res["delta"]) < 1e-12 and r3["n_keys_A"] == 6
    assert res["pair_ok"] and res["n_unpaired_A"] == 0 and res["n_nan_keys_A"] == 0
    assert r3["n_nan_keys_A"] == 2 and not r3["pair_ok"]                  # NaN 키는 표시한다(확인적 대비는 호출자가 판정 불가로 둔다)
    # (개정 1) 짝지음: 분할 1 에만 A 의 추출 1 이 있으면 그 단위는 빠지고 n_unpaired_A 로 센다
    st1 = stores[1]
    QX = np.sort(y[None, :] * np.exp(rng.normal(-0.5, 0.2, (99, n))), 0)
    st1.add(("A", "", 10, 1, 0), LC.score_cells(y, QX))
    r4 = LC.one_stage_delta(stores, fA, fB, "is10", nboot=200, seed=11)
    np.testing.assert_allclose(r4["dist"], r3["dist"], rtol=1e-12)
    assert r4["n_unpaired_A"] == 1 and r4["n_unpaired_B"] == 0 and not r4["pair_ok"]
    # B 에도 같은 단위가 생기면 짝지어 들어가고 추출 평균의 차가 된다
    st1.add(("B", "", 10, 1, -1), LC.score_cells(y, QX))
    r5 = LC.one_stage_delta(stores, fA, fB, "is10", nboot=0, seed=11)
    assert r5["n_unpaired_A"] == 0 and np.isfinite(r5["delta"])
    assert np.isnan(LC.one_stage_delta(stores, lambda k: k[0] == "Z", fB, "is10", nboot=10)["delta"])
    S = np.array([[1.0, np.inf, 2.0], [1.0, np.nan, 2.0]]); C = np.ones((2, 3))
    Wt = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    out = LC.agg_cell(S, C, Wt)
    np.testing.assert_allclose(out[:, 0], [5.0 / 3.0, 5.0 / 3.0])
    assert np.isinf(out[0, 1]) and np.isnan(out[1, 1])
    np.testing.assert_allclose(LC.agg_beq(np.array([[2.0, 9.0]]), np.array([[2, 3]]), np.ones(2)), [(1.0 + 3.0) / 2])
    assert LC.verdict4(-3, -1, -2, -0.5, 0.1) == "우세"
    assert LC.verdict4(1, 2, 0.5, 3, 0.1) == "열세"
    assert LC.verdict4(-0.05, 0.05, -0.04, 0.06, 0.1) == "동등"
    assert LC.verdict4(-1, 1, -1, 1, 0.1) == "미결정"
    assert LC.verdict4(np.nan, 1, -1, 1, 0.1) == "판정 불가"
    assert LC.eq_state(-0.3, 0.3, -0.3, 0.3, 0.1) == "검정 불가(정밀도 미달)"
    assert LC.eq_state(-0.05, 0.05, -0.04, 0.06, 0.1) == "동등"
    assert LC.eq_state(0.02, 0.12, 0.0, 0.1, 0.1) == "동등 아님"
    assert LC.eq_state(-0.05, 0.05, -0.2, 0.2, (0.1, 0.3)) == "동등"
    assert LC.eq_state(-0.05, 0.05, -0.2, 0.2, 0.1) == "검정 불가(정밀도 미달)"
    assert LC.rel_margin(100.0, 80.0, 0.05) == (5.0, 4.0)
    assert LC.half_width(-1, 1, -2, 0) == 1.0
    lo_, hi_, nn = LC.ci95(np.r_[np.arange(100.0), np.nan])
    assert nn == 1 and abs(lo_ - np.percentile(np.arange(100.0), 2.5)) < 1e-12
    np.testing.assert_allclose(LC.combine_same_index([np.array([1.0, np.nan]), np.array([3.0, 4.0])]), [2.0, 4.0])
    sm = LC.strat_mean({"a": np.array([1.0, np.nan]), "b": np.array([3.0, 4.0])})
    assert sm[0] == 2.0 and np.isnan(sm[1])


# ---------------------------------------------------------------- (i)
def _emu_df():
    rng = np.random.RandomState(7)
    rows = []
    for rid, (reg, nb, per) in enumerate((("R1", 12, 8), ("R2", 10, 6), ("R3", 3, 4), ("T", 5, 5))):
        for b in range(nb):
            for _ in range(per):
                rows.append(dict(macro=reg, block=(rid + 1) * 1000 + b, y=rng.lognormal(3.5, 0.3), s=rng.uniform(10, 30)))
    return pd.DataFrame(rows)


def test_i_emulation_plan():
    df = _emu_df()
    src = np.where(df.macro.values != "T")[0]
    G = ["R1", "R2", "R3"]
    grid = [0, 3, 10, 40]
    r1 = LC.emulation_plan(df, src, G, grid, "T", "x")
    r2 = LC.emulation_plan(df, src, G, grid, "T", "x")
    assert len(r1) == len(r2) > 0
    for a, b in zip(r1, r2):
        assert (a["region"], a["n"], a["e"], a["d"]) == (b["region"], b["n"], b["e"], b["d"])
        np.testing.assert_array_equal(a["cal_idx"], b["cal_idx"]); np.testing.assert_array_equal(a["lab_idx"], b["lab_idx"])
    blk, mac = df.block.values, df.macro.values
    y, s = df.y.values, df.s.values
    zero = [r for r in r1 if r["n"] == 0]
    assert [r["region"] for r in zero] == G
    for r in zero:
        np.testing.assert_array_equal(np.sort(r["cal_idx"]), np.where(mac == r["region"])[0])
        rest = src[mac[src] != r["region"]]
        assert abs(r["E0_mk"] - LC.ls_E(y[rest], s[rest])) < 1e-12 and r["E_center"] == r["E0_mk"]
    pos = [r for r in r1 if r["n"] > 0]
    assert pos and not [r for r in pos if r["region"] == "R3"]                  # R3 는 채점 블록이 1개뿐이다
    assert not [r for r in pos if r["region"] == "R2" and r["n"] == 40]         # |A_k| ≤ n
    for r in pos:
        assert len(r["lab_idx"]) == r["n"] < r["n_A"] and r["nb_cal"] >= 2
        assert (mac[r["lab_idx"]] == r["region"]).all() and (mac[r["cal_idx"]] == r["region"]).all()
        assert not set(blk[r["lab_idx"]]) & set(blk[r["cal_idx"]])
        assert abs(r["E_center"] - LC.center_coef(y[r["lab_idx"]], s[r["lab_idx"]], r["E0_mk"])) < 1e-12
    r = pos[0]
    idx_k = src[mac[src] == r["region"]]
    A_k, B_k = LC.half_split_blocks(df, idx_k, r["e"])
    sel = np.sort(np.random.RandomState(LC.seed_of("lgu-emu", "T", "x", r["region"], r["e"], r["n"], r["d"]))
                  .choice(len(A_k), r["n"], replace=False))
    np.testing.assert_array_equal(r["lab_idx"], A_k[sel]); np.testing.assert_array_equal(r["cal_idx"], B_k)


# ---------------------------------------------------------------- (j)
def test_j_loc_only():
    rng = np.random.RandomState(9)
    Qz = np.sort(rng.normal(0.2, 0.3, (99, 10)), 0)
    np.testing.assert_allclose(LC.loc_only(Qz, 1.0), Qz, atol=1e-12)
    Z0 = LC.loc_only(Qz, 0.0)
    assert np.abs(Z0[49]).max() == 0.0
    np.testing.assert_allclose(Z0 - Z0[49], Qz - Qz[49], atol=1e-12)
    Zq = LC.loc_only(Qz, 0.25)
    np.testing.assert_allclose(Zq[49], 0.25 * Qz[49], atol=1e-12)
    Q5 = np.sort(rng.normal(0, 0.3, (5, 4)), 0)
    np.testing.assert_allclose(LC.loc_only(Q5, 0.0)[2], 0.0, atol=1e-15)
    with pytest.raises(ValueError):
        LC.loc_only(np.zeros((7, 2)), 0.5)
    np.testing.assert_allclose(LC.to_cm(np.array([10.0, 20.0]), np.zeros((3, 2))), [[10.0, 20.0]] * 3)


# ---------------------------------------------------------------- (m)
def test_m_block_val_mask():
    rng = np.random.RandomState(8)
    sizes = rng.randint(1, 60, 30)
    g = np.repeat(np.arange(30) * 7 + 3, sizes)
    mask, flag = LC.block_val_mask(g, seed=5)
    assert flag == "val_block" and 0.1 <= mask.mean() <= 0.5
    for b in np.unique(g):
        assert mask[g == b].all() or (~mask[g == b]).all()
    np.testing.assert_array_equal(mask, LC.block_val_mask(g, seed=5)[0])
    ub = np.unique(g); order = np.random.RandomState(5).permutation(len(ub))
    vb = set(np.unique(g[mask]))
    k = len(vb)
    assert set(ub[order[:k]]) == vb                                   # 순열의 앞에서부터 넣었다
    assert (np.isin(g, ub[order[:k - 1]])).sum() < 0.1 * len(g)       # 마지막 블록 전에는 10 % 미만
    m1, f1 = LC.block_val_mask(np.zeros(100, int), seed=5)
    assert f1 == "val_random"
    np.testing.assert_array_equal(m1, np.random.RandomState(5).rand(100) < 0.1)
    m2, f2 = LC.block_val_mask(np.r_[np.zeros(95, int), np.ones(5, int)], seed=5)
    assert f2 == "val_random"
    w = LC.row_weights(np.r_[np.zeros(150, int), np.ones(50, int)])
    assert abs(w[0] - 100 / 150) < 1e-12 and w[-1] == 1.0


# ---------------------------------------------------------------- (n)
def _phys_df(n, rng):
    return pd.DataFrame(dict(e5_tdd=rng.uniform(300, 1500, n), e5_fdd=rng.uniform(2000, 5000, n), e5_sqrt_tdd=rng.uniform(17, 39, n),
                             e5_maat=rng.uniform(-15, -2, n), e5_twarm=rng.uniform(8, 16, n), e5_tcold=rng.uniform(-35, -20, n),
                             e5_swe=rng.uniform(0.02, 0.2, n), sg_bdod_5_15=rng.uniform(0.8, 1.5, n), sg_sand_5_15=rng.uniform(10, 60, n),
                             sg_cfvo_5_15=rng.uniform(0, 20, n), sg_soc_5_15=rng.uniform(10, 150, n)))


def test_n_phys_log_samples():
    rng = np.random.RandomState(10)
    dfc = _phys_df(30, rng); dff = _phys_df(100, rng)
    dfc.loc[3, "sg_bdod_5_15"] = np.nan; dfc.loc[5, "e5_tdd"] = np.nan
    a = LC.phys_log_samples(dfc, dff, 16, seed=0)
    b = LC.phys_log_samples(dfc, dff, 16, seed=0)
    assert a.shape == (16, 30) and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, LC.phys_log_samples(dfc, dff, 16, seed=1))
    c = LC.phys_log_samples(dfc, dff, 16, seed=0, unit_tdd=True)
    assert np.isfinite(c).all()
    np.testing.assert_allclose(a[:, 0] - c[:, 0], 0.5 * np.log(dfc.e5_tdd.values[0]), rtol=1e-10)
    np.testing.assert_allclose(a[:, 5] - c[:, 5], 0.5 * np.log(np.nanmedian(dff.e5_tdd.values)), rtol=1e-10)


# ---------------------------------------------------------------- (o)
def test_o_permission(monkeypatch):
    monkeypatch.delenv("LG_RESCALE", raising=False)
    a = SimpleNamespace(allow_local=False, count_only=False, summarize_only=True)
    with pytest.raises(SystemExit):
        LC.require_permission(a)
    with pytest.raises(SystemExit):
        LC.require_permission(SimpleNamespace(allow_local=False, count_only=False, smoke=True))
    assert LC.require_permission(SimpleNamespace(allow_local=False, count_only=True)) is False
    assert LC.require_permission(SimpleNamespace(allow_local=True, count_only=False)) is True
    monkeypatch.setenv("LG_RESCALE", "1")
    assert LC.require_permission(a) is True
    assert "같은 효과" not in LC.PERMIT_MSG and "그대로 적용" in LC.PERMIT_MSG   # 개정 1: LG_RESCALE=1 은 허용 표지일 뿐이다
    assert LC.peek_threads("4", argv=["x", "--threads", "8"], environ={}) == "1"
    assert LC.peek_threads("4", argv=["x", "--threads", "8"], environ={"LG_RESCALE": "1"}) == "8"
    assert LC.peek_threads("4", argv=["x", "--allow-local"], environ={}) == "4"
    assert LC.peek_threads("4", argv=["x", "--allow-local", "--threads=2"], environ={}) == "2"
    assert LC.run_permitted(False, environ={}) is False and LC.run_permitted(True, environ={}) is True
    for v in LC.THREAD_VARS:
        monkeypatch.setenv(v, "7")
    LC.set_thread_env(2)
    assert all(os.environ[v] == "2" for v in LC.THREAD_VARS)


# ---------------------------------------------------------------- (p)
def test_p_helpers(tmp_path):
    cfg = dict(a=1, b=[1, 2])
    assert LC.cfg_hash(cfg) == LC.cfg_hash(dict(b=[1, 2], a=1)) and len(LC.cfg_hash(cfg)) == 12
    uj, rf = tmp_path / "u_unit.json", tmp_path / "u_runs.csv"
    assert LC.unit_state(uj, [rf], cfg)[0] is False
    LC.atomic_text(rf, "x\n1\n")
    LC.atomic_text(uj, json.dumps(dict(cfg_hash=LC.cfg_hash(cfg), status="ok")))
    assert LC.unit_state(uj, [rf], cfg) == (True, "ok")
    assert LC.unit_state(uj, [rf], dict(a=2))[0] is False
    LC.atomic_text(uj, json.dumps(dict(cfg_hash=LC.cfg_hash(cfg), status="failed")))
    assert LC.unit_state(uj, [rf], cfg)[0] is False
    p = LC.atomic_npz(tmp_path / "z.npz", a=np.arange(3))
    with np.load(p) as z:
        np.testing.assert_array_equal(z["a"], np.arange(3))
    assert not list(tmp_path.glob("*.tmp*")) and LC.file_sha(p) != "none" and LC.file_sha(tmp_path / "nope") == "none"
    ft = tmp_path / "flags.csv"
    pd.DataFrame(dict(loc_id=[1, 2, 3], early=[1, 0, 0], gpr=[0, 1, 0], probe=[1, 0, 1], team="t", n_pts=1,
                      above_matched=1)).to_csv(ft, index=False)
    fl = LC.load_flags(ft, [3, 1])
    assert fl["early"].tolist() == [False, True] and fl["probe"].tolist() == [True, True] and fl["gpr"].tolist() == [False, False]
    with pytest.raises(SystemExit):
        LC.load_flags(ft, [1, 9])
    with pytest.raises(SystemExit):
        LC.load_flags(tmp_path / "none.csv", [1])
    assert LC.load_flags(tmp_path / "none.csv", [1, 2], missing_ok=True)["gpr"].tolist() == [False, False]
    deg = 2 * np.pi * LC.EARTH_R_KM / 360
    np.testing.assert_allclose(LC.nearest_km([61.0, 70.0], [150.0, 150.0], [60.0, 50.0], [150.0, 150.0]), [deg, 10 * deg], rtol=1e-9)
    assert np.isnan(LC.nearest_km([60.0], [150.0], [], [])).all()
    assert LC.center_coef([], [], 3.0) == 3.0
    assert abs(LC.center_coef(np.array([30.0, 40.0]), np.array([10.0, 10.0]), 3.0) - (2 * 3.5 + 10 * 3.0) / 12) < 1e-12
    assert LC.center_coef(np.array([30.0]), np.array([np.nan]), 3.0) == 3.0


# ---------------------------------------------------------------- (r)
def test_r_gpu_helpers_and_fit_gen_checks(monkeypatch):
    import subprocess

    class _R:
        returncode = 0
        stdout = "0, 0\n1, 512\n2, 0\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _R())
    assert LC.gpu_memory_used() == {0: 0.0, 1: 512.0, 2: 0.0}
    assert LC.gpu_memory_used([2, 5]) == {2: 0.0, 5: None}
    assert LC.free_gpus([2, 1, 5]) == [2]                                # 0 MiB 규칙, 읽지 못한 번호는 사용 중으로 본다
    assert LC.free_gpus([2, 1], max_mib=600) == [2, 1]

    def _boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", _boom)
    assert LC.gpu_memory_used() == {} and LC.free_gpus([2]) == []
    X = np.zeros((10, 3), np.float32); y = np.zeros(10)                  # 아래 호출은 모두 torch 적재 전에 거부된다
    with pytest.raises(ValueError):
        LC.fit_gen("gan", X, y)
    with pytest.raises(ValueError):
        LC.fit_gen("cfm", X, y, median_zero=True)
    Xn = X.copy(); Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        LC.fit_gen("nflow", Xn, y)
    with pytest.raises(ValueError):
        LC.fit_gen("nflow", X, y, weights=np.ones(9))
    with pytest.raises(ValueError):
        LC.fit_gen("nflow", X, y, groups=np.zeros(9), val="block")


# ---------------------------------------------------------------- (s) 개정 1 의 도우미
def test_s_revision1_helpers(tmp_path, monkeypatch):
    # is_env_error: 환경·자원 오류만 참
    class OutOfMemoryError(RuntimeError):
        pass
    assert LC.is_env_error(ImportError("x")) and LC.is_env_error(MemoryError())
    assert LC.is_env_error(OutOfMemoryError("CUDA out of memory"))
    assert LC.is_env_error(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not LC.is_env_error(RuntimeError("shape mismatch")) and not LC.is_env_error(ValueError("x"))
    # cal_groups: 로그 비가 정의되는 셀(ok) 수로 센다
    mac = np.array(["A"] * 25 + ["B"] * 22 + ["C"] * 30)
    ok = np.ones(len(mac), bool); ok[25:30] = False                      # B 의 ok 셀 17개
    assert LC.cal_groups(mac, np.arange(len(mac)), ok, 20) == ["A", "C"]
    assert LC.cal_groups(mac, np.arange(len(mac)), np.ones(len(mac), bool), 20) == ["A", "B", "C"]
    # check_blockindex: 처음에는 쓰고, 같은 색인이면 통과, 다르면 중단
    df = pd.DataFrame(dict(macro=["A", "A", "B"], block=[1, 2, 1]))
    bi = LC.BlockIndex(df)
    p = LC.check_blockindex(tmp_path / "bi.csv", bi)
    assert p.exists() and LC.check_blockindex(tmp_path / "bi.csv", bi) == p
    with pytest.raises(SystemExit):
        LC.check_blockindex(tmp_path / "bi.csv", LC.BlockIndex(pd.DataFrame(dict(macro=["A", "B"], block=[1, 1]))))
    # atomic_text: 교체가 실패하면 임시 파일을 남기지 않는다
    (tmp_path / "adir").mkdir()
    with pytest.raises(OSError):
        LC.atomic_text(tmp_path / "adir", "x")
    assert not list(tmp_path.glob("adir.tmp*"))
    # 실행 중 스레드 등록부
    reg = tmp_path / "running"
    monkeypatch.setenv(LC.RUN_REGISTRY_ENV, str(reg))
    p1 = LC.claim_threads("t1", 4, cap=8)
    assert p1.exists() and [int(r["threads"]) for r in LC.running_threads()] == [4]
    ppid = os.getppid()                                                  # 살아 있는 다른 프로세스(부모)를 흉내 낸다
    (reg / f"{ppid}.json").write_text(json.dumps(dict(pid=ppid, start=LC._start_ticks(ppid), tag="other", threads=6)), encoding="utf-8")
    with pytest.raises(SystemExit):
        LC.claim_threads("t2", 4, cap=8)                                 # 다른 실행 6 + 이번 4 > 8(자기 pid 항목은 세지 않는다)
    LC.claim_threads("t3", 2, cap=8)                                     # 6 + 2 = 8 은 허용(자기 항목은 덮어쓴다)
    (reg / "999999999.json").write_text(json.dumps(dict(pid=999999999, tag="dead", threads=8)), encoding="utf-8")
    alive = LC.running_threads()
    assert {r["tag"] for r in alive} == {"t3", "other"} and not (reg / "999999999.json").exists()
    (reg / f"{ppid}.json").write_text(json.dumps(dict(pid=ppid, start="1", tag="reused", threads=6)), encoding="utf-8")
    assert "reused" not in {r["tag"] for r in LC.running_threads()}      # 시작 시각이 다르면 pid 재사용으로 본다
    LC.release_threads(p1)
    assert not p1.exists()
    LC.release_threads(None)


# ---------------------------------------------------------------- 학습을 하는 시험
def _pin_cuda_invisible():
    """시험 프로세스의 CUDA 런타임을 CUDA_VISIBLE_DEVICES="" 상태에서 먼저 초기화해 장치 수 0 을 고정한다.

    CUDA 런타임은 처음 초기화될 때의 CUDA_VISIBLE_DEVICES 를 프로세스 끝까지 쓴다. 환경 변수를 지운 채 torch 학습을 하면(시험 t)
    그 시점에 전 GPU 를 보는 상태로 초기화되고, 뒤 시험의 tab_models._dev() 가 물리 GPU 0 을 고른다(2026-09-30 heavy 시험에서 확인:
    시험 k·q 가 이 순서에서만 실패했다). 초기화만 하고 컨텍스트는 만들지 않는다. 장치가 보이면 시험을 멈춘다(GPU 사용 금지 규칙)."""
    import torch
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "시험은 CUDA_VISIBLE_DEVICES=\"\" 로 실행한다"
    n = int(torch._C._cuda_getDeviceCount()) if hasattr(torch._C, "_cuda_getDeviceCount") else 0
    assert n == 0, f"시험 프로세스가 GPU {n}장을 본다(시험은 GPU 를 쓰지 않는다)"


def _synth(n=600, d=5, seed=0):
    LC.set_torch_threads(2)                                             # 로컬 규칙: 시험의 torch 스레드 2
    _pin_cuda_invisible()                                               # 시험 t 의 환경 변수 삭제가 뒤 시험의 장치를 바꾸지 않게 한다
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    y = 0.5 * X[:, 0] + (0.3 + 0.2 * np.abs(X[:, 1])) * rng.randn(n)
    blocks = rng.randint(0, 40, n)
    return X, y, blocks, rng


@HEAVY
def test_t_fit_gen_defaults_to_cpu_without_visible_devices(monkeypatch):
    X, y, blocks, _ = _synth(n=200, seed=3)                            # _synth 가 CUDA 런타임을 장치 0 개로 먼저 고정한다
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    h = LC.fit_gen("nflow", X, y, groups=blocks, weights=LC.row_weights(blocks), seed=0, epochs=1)
    assert str(h.device) == "cpu"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _pin_cuda_invisible()                                               # 환경 변수를 지운 동안에도 런타임이 GPU 를 보지 않았다


@HEAVY
def test_k_legacy_matches_tab_models():
    from polar import tab_models as TM
    X, y, _, rng = _synth(seed=11)
    Xte = rng.randn(50, 5).astype(np.float32)
    for name in ("cfm", "ddpm", "nflow"):
        ref = TM.fit_predict(name, X, y, Xte, seed=0, epochs=3)["samples"]
        h = LC.fit_gen(name, X, y, seed=0, epochs=3, clamp=LC.LEGACY_CLAMP, median_zero=False, val="random", weights=None)
        assert h.legacy
        got = LC.gen_samples(h, Xte, S=64, seed=0, chunk=4096)
        assert got.shape == ref.shape
        diff = float(np.max(np.abs(got - ref)))
        print(f"[k] 래퍼 재현 {name}: 최대 절대 차 {diff:.3e}(기준 1e-5, 계획서 2.9절)")
        assert diff <= 1e-5, name


@HEAVY
def test_l_nflow_quantiles_density():
    X, y, blocks, _ = _synth(seed=12)
    w = LC.row_weights(blocks)
    h = LC.fit_gen("nflow", X, y, groups=blocks, weights=w, seed=1, epochs=3, clamp=LC.NFLOW_CLAMP, median_zero=False, val="block")
    assert np.isfinite(h.val_loss) and h.flag.startswith("val_block") and not h.legacy
    q = LC.nflow_quantiles(h, X[:3])
    assert q.shape == (99, 3) and (np.diff(q, axis=0) >= 0).all()
    for i in range(3):
        smp = LC.gen_samples(h, np.repeat(X[i:i + 1], 200000, 0), S=1, seed=5 + i)[0]
        emp = np.quantile(smp, LC.TAUS)
        assert float(np.max(np.abs(emp - q[:, i]))) <= 0.05 * float(smp.std())
    h0 = LC.fit_gen("nflow", X, y, groups=blocks, weights=w, seed=1, epochs=3, median_zero=True)
    q0 = LC.nflow_quantiles(h0, X[:50], taus=np.array([0.5]))
    assert float(np.max(np.abs(q0))) <= 1e-6
    assert (np.diff(LC.nflow_quantiles(h0, X[:50]), axis=0) >= 0).all()
    t = np.linspace(-40, 40, 80001)
    grid = h.ymu + h.ysd * t
    f = np.exp(LC.nflow_logpdf(h, X[:2], np.tile(grid, (2, 1))))
    integ = ((f[:, 1:] + f[:, :-1]) / 2).sum(1) * (grid[1] - grid[0])
    assert np.all(np.abs(integ - 1) < 1e-3), integ
    eg = h.ymu + h.ysd * np.linspace(-4, 4, 801)
    cdf = LC.nflow_cdf_table(h, X[:4], eg)
    assert cdf.shape == (4, 801) and cdf.dtype == np.float32
    assert (np.diff(cdf, axis=1) >= 0).all() and cdf.min() >= 0 and cdf.max() <= 1


@HEAVY
def test_q_cfm_quantiles_and_catboost():
    X, y, blocks, _ = _synth(seed=13)
    w = LC.row_weights(blocks)
    hc = LC.fit_gen("cfm", X, y, groups=blocks, weights=w, seed=0, epochs=3, val="block")
    Qs = LC.gen_quantiles(hc, X[:20], S=64, seed=0)
    assert Qs.shape == (99, 20) and np.isfinite(Qs).all()
    np.testing.assert_array_equal(Qs, LC.gen_quantiles(hc, X[:20], S=64, seed=0))
    cs = LC.crossing_stats(LC.cfm_quantiles_ode(hc, X[:20]))
    assert 0.0 <= cs["frac_cells"] <= 1.0
    bad, why = LC.fit_failed(hc, LC.rearrange(Qs))
    assert isinstance(bad, bool) and isinstance(why, str)
    assert LC.fit_failed(None, np.full((99, 3), 5.0))[0] is True
    preds, info = LC.cb_multiquantile(X, y, [X[:30], X[30:50]], seed=0, weights=w, threads=1)
    assert preds[0].shape == (30, 5) and preds[1].shape == (20, 5)
    assert (np.diff(preds[0], axis=1) >= 0).all()
    assert info["flag"] == "multiquantile" or info["flag"].startswith("per_quantile")
    assert 0.0 <= info["frac_cells"] <= 1.0
    print(f"[q] CatBoost 다분위 방식: {info['flag']}, 교차 셀 비율 {info['frac_cells']:.4f}")
