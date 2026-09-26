"""src/polar/h4_common.py 빠른 단위 시험(합성 자료).

(a) 블록 간 이질성이 크고 반복 간 차이가 작으면 블록 부트스트랩 CI 가 반복 부트스트랩 CI 보다 넓다.
(b) 방법 A·B 와 모든 반복이 같은 재표집 인덱스를 공유한다(동일 예측이면 분포가 정확히 0).
(c) 절단 순위: 절단 대상은 관측값보다 큰 최대 순위 동률.
(d) 오프셋 최대우도: n → ∞ 에서 사후 평균이 대상 표본 평균으로 수렴, n = 0 이면 사전.
실행: python3 -m pytest -q tests/test_h4_common.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from polar.h4_common import (BlockStore, block_sse, boot_weights, boot_delta_blocks, rep_boot_ci, save_stores,  # noqa: E402
                             load_stores, stores_for_target, min_n, rank_with_censoring, spearman_censored,
                             kendall_censored, achieved_test, offset_mle_prior, offset_mle_estimate, recovery)


def _synthetic(n_splits=2, nb=12, cpb=30, reps=5, seeds=2, seed=0):
    """블록마다 물리식 편향이 크게 다르고, 방법 A 는 블록 편향을 블록별로 다른 비율로 줄인다. 반복 잡음은 작다."""
    rng = np.random.RandomState(seed)
    out = {}
    for sp in range(n_splits):
        blocks = np.repeat([f"b{j}" for j in range(nb)], cpb)
        y = rng.normal(50, 5, len(blocks))
        bias = np.repeat(rng.normal(0, 20, nb), cpb)
        shrink = np.repeat(rng.uniform(-0.3, 1.0, nb), cpb)          # 블록에 따라 개선 또는 악화
        st = BlockStore("T", sp, blocks)
        st.add(("physics", 0, -1, -1), y, y + bias)
        for r in range(reps):
            for s in range(seeds):
                st.add(("A", 10, r, s), y, y + bias * (1 - shrink) + rng.normal(0, 0.05, len(y)))
        out[sp] = st
    return out


def test_block_sse_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0]); p = np.array([1.0, 3.0, 5.0, np.nan]); codes = np.array([0, 0, 1, 1])
    sse, cnt = block_sse(y, p, codes, 2)
    assert np.allclose(sse, [1.0, 4.0]) and np.array_equal(cnt, [2, 1])


def test_a_block_ci_wider_than_rep_ci():
    sts = _synthetic()
    res = boot_delta_blocks(sts, lambda k: k[0] == "A", lambda k: k[0] == "physics", nboot=500, seed=3,
                            rep_fn=lambda k: k[2])
    # 반복 부트스트랩: (분할, 반복) 행 Δ
    d_rows = []
    for sp, st in sts.items():
        phys = st.rmse(("physics", 0, -1, -1))
        for r in range(5):
            d_rows.append(np.mean([st.rmse(("A", 10, r, s)) for s in range(2)]) - phys)
    lo_r, hi_r = rep_boot_ci(d_rows, nboot=500)
    assert res["ci_lo"] <= res["delta"] <= res["ci_hi"]
    assert (res["ci_hi"] - res["ci_lo"]) > 3 * (hi_r - lo_r)
    assert res["n_splits"] == 2 and res["n_keys_A"] == 20 and res["n_keys_B"] == 2
    assert np.isfinite(res["delta_beq"]) and res["ci_lo_beq"] <= res["ci_hi_beq"]
    assert 0.0 <= res["split_win"] <= 1.0 and 0.0 <= res["rep_win"] <= 1.0


def test_b_pairing_uses_same_indices():
    W1, W2 = boot_weights(9, 50, 7), boot_weights(9, 50, 7)
    assert np.array_equal(W1, W2) and np.all(W1.sum(1) == 9)
    rng = np.random.RandomState(1)
    blocks = np.repeat(np.arange(10), 20); y = rng.normal(0, 1, len(blocks))
    st = BlockStore("T", 0, blocks)
    for r in range(4):
        p = y + np.repeat(rng.normal(0, 3, 10), 20)
        st.add(("A", r), y, p); st.add(("B", r), y, p)                       # 반복마다 A 와 B 가 동일
    res = boot_delta_blocks({0: st}, lambda k: k[0] == "A", lambda k: k[0] == "B", nboot=200, seed=0,
                            rep_fn=lambda k: k[1], return_dist=True)
    assert np.allclose(res["dist"], 0.0) and np.allclose(res["dist_beq"], 0.0)
    assert res["delta"] == 0.0 and res["rep_win"] == 0.0


def test_store_roundtrip(tmp_path):
    sts = _synthetic(n_splits=2, reps=2, seeds=1)
    p = save_stores(sts.values(), tmp_path / "x_blocksse.npz")
    ld = stores_for_target(load_stores([p]), "T")
    assert sorted(ld) == [0, 1]
    for sp in (0, 1):
        for k in sts[sp].keys:
            a, b = sts[sp].get(k), ld[sp].get(k)
            assert np.allclose(a[0], b[0]) and np.array_equal(a[1], b[1])
    r1 = boot_delta_blocks(sts, lambda k: k[0] == "A", lambda k: k[0] == "physics", nboot=100, seed=5)
    r2 = boot_delta_blocks(ld, lambda k: k[0] == "A", lambda k: k[0] == "physics", nboot=100, seed=5)
    assert np.isclose(r1["ci_hi"], r2["ci_hi"]) and np.isclose(r1["delta"], r2["delta"])


def test_c_rank_censoring_and_min_n():
    r = rank_with_censoring([3, 10, 160, 5, 160], [False, False, True, False, True])
    assert np.allclose(r, [1, 3, 4.5, 2, 4.5])
    x = np.array([0.05, 0.1, 0.3, 0.4, 0.5, 0.6])
    ns = np.array([3, 5, 20, 40, np.nan, np.nan]); cen = np.isnan(ns)
    rho, p, n = spearman_censored(x, ns, cen)
    tau, pk, _ = kendall_censored(x, ns, cen)
    assert n == 6 and rho > 0.9 and tau > 0.8
    cur = pd.DataFrame(dict(n=[3, 5, 10, 20], ci_hi=[0.5, -0.1, -0.2, -0.3], split_win=[1, 1 / 3, 1, 1], rep_win=[1, 1, 0.8, 0.9]))
    out = min_n(cur)
    assert out["n_star"] == 10 and not out["censored"]
    out2 = min_n(cur.assign(ci_hi=0.1))
    assert out2["censored"] and np.isnan(out2["n_star"]) and out2["n_max"] == 20
    assert np.isnan(recovery(10.0, 9.0, 10.5)) and np.isclose(recovery(10.0, 9.0, 8.0), 0.5)
    at = achieved_test([True, False, True, False, True, False, False, True], [0.05, 0.3, 0.1, 0.4, 0.2, 0.15, 0.5, 0.35])
    assert 0 <= at["p_mw"] <= 1 and at["mw_method"] == "exact" and np.isfinite(at["logit_coef"])


def test_d_offset_mle_converges():
    rng = np.random.RandomState(0)
    rows = []
    for j, mu in enumerate([0.5, 0.7, 0.9, 0.6, 1.1]):
        s = rng.uniform(20, 40, 200); z = rng.normal(mu, 0.2, 200)
        rows.append(pd.DataFrame(dict(macro=f"R{j}", s=s, y=np.exp(z) * s)))
    prior = offset_mle_prior(pd.concat(rows), region_col="macro", min_cells=3)
    assert prior["n_regions"] == 5 and abs(prior["sigma2"] - 0.04) < 0.01 and prior["tau2"] > 0.01
    p0 = offset_mle_estimate([], prior)
    assert np.isclose(p0["logE_post"], prior["logE0"]) and np.isclose(p0["logE_sd"], np.sqrt(prior["tau2"]))
    zt = rng.normal(1.5, 0.2, 200_000)
    pn = offset_mle_estimate(zt, prior)
    assert abs(pn["logE_post"] - zt.mean()) < 1e-3 and pn["w"] > 0.999
    small = offset_mle_estimate(zt[:3], prior)
    assert prior["logE0"] < small["logE_post"] < zt[:3].mean() + 1e-9 or zt[:3].mean() < small["logE_post"] < prior["logE0"]
    with pytest.raises(ValueError):
        offset_mle_prior(rows[0], region_col="macro")
