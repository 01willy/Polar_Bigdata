"""scripts/2_evaluation/h46_spatial_diagnostics.py 단위 시험(계획 docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 5절, spec_C 8절).

가벼운 시험(합성 자료, 학습 없음, 실자료를 읽지 않는다)
(a) 지수 공분산으로 만든 합성 장(서로 200 km 넘게 떨어진 독립 조각 10개, 조각은 100 km 정사각형)에서 적합한 range_a_km 가 참값의 0.5–2배 안에 든다.
(b) 독립 잡음에서 쌍이 20,000개 이상인 구간의 rho 가 0 ± 0.05 다.
(c) 구간별 쌍 수와 합이 전수 계산과 같다(최대 거리 안의 쌍만 센다). 모든 쌍이 구간 안이면 합이 N(N − 1)/2 이고 쌍 가중 gamma 평균이
    var_w(ddof 1)와 같다. 청크 크기에 무관하다. 블록 하나 제외 gamma_(−b) 가 블록을 실제로 뺀 재계산과 같고 jk_se 가 정의식과 같다.
(d) 블록 내 상관이 알려진 분산 성분의 모의 자료에서 회복된다. 적률식의 수계산 대조, 다중도 가중과 복제 자료의 일치, CI 재현성.
(e) 근접 표의 추출이 h40.draw_cells 와 같고 frac_within 이 수계산과 같으며 d_km 에 단조 증가한다. n ≥ |A| 는 행이 없고 all 은 A 전체 기준과 같다.
    분할 평균 행의 최소·최대 규칙.
(f) SAR 정렬: r_within 과 그 CI 가 블록 평균을 더해도 변하지 않는다. r_cell, rho_cell 이 numpy·scipy 값과 같다. 셀 부족 단위는 NaN 과 flag.
    polsar 분류 부호 점검 행의 수계산 대조, 요약 행(SUMMARY9)의 개수.
(g) 허용 표지 없이 실행을 거부한다(자료를 읽기 전). 로컬 스레드 합계 상한(LG_RESCALE=1 에도 적용, 개정 1), 기본값, 스레드 결정 규칙.
(i) 집계: 합성 조각에서 표와 항목 unit.json 이 만들어지고, 설정 해시가 다르면 표를 쓰지 않고 종료 코드 1 이다.
(j) 합성 자료 객체로 하위 단위 실행(variogram, icc, sar), --resume, 실패 기록, 집계를 거친다.
실자료를 쓰는 시험(표지 HEAVY, LG_RUN_HEAVY=1 일 때만)
(h) 스모크 경로(--smoke --allow-local --workers 1 --threads 2)의 종료 코드 0 과 표 생성.
실행(로컬 GPU 서버, 사용자 지시 2026-09-30. GPU 를 쓰지 않는다. 스레드 2, nice 10):
  LG_RUN_HEAVY=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      nice -n 10 python3 -m pytest -q tests/test_h46_spatial.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # 시험은 GPU 를 쓰지 않는다
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

HEAVY = pytest.mark.skipif(os.environ.get("LG_RUN_HEAVY", "") != "1",
                           reason="실자료를 쓰는 시험은 LG_RUN_HEAVY=1 일 때만 실행한다(공유 서버 CPU 보호. 수정 단계의 로컬 단위 시험에서 실행)")
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


M = _load("h46_spatial_diagnostics", "scripts/2_evaluation/h46_spatial_diagnostics.py")
H = sys.modules["h40_label_grid"]
import polar.lgu_common as LC  # noqa: E402

EDGES = np.array([0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 200], float)
NB = len(EDGES) - 1


def _grid_block(lat, lon):
    """fidelity.add_group_keys 와 같은 0.5° 블록 번호."""
    return np.floor(np.asarray(lat) / 0.5).astype(int) * 100000 + np.floor(np.asarray(lon) / 0.5).astype(int)


def _box(rng, n, lat0, lon0, half_km):
    dlat = half_km / 111.2
    dlon = half_km / (111.2 * np.cos(np.radians(lat0)))
    return lat0 + rng.uniform(-dlat, dlat, n), lon0 + rng.uniform(-dlon, dlon, n)


def _brute(lat, lon, w, edges=EDGES):
    n = len(w)
    I, J = np.triu_indices(n, 1)
    d = M.pair_dist_f32(lat, lon, I, J)
    m = (d >= edges[0]) & (d < edges[-1])
    b = np.searchsorted(edges, d[m], side="right") - 1
    h2 = 0.5 * (w[I[m]] - w[J[m]]) ** 2
    nb = len(edges) - 1
    return np.bincount(b, minlength=nb), np.bincount(b, weights=h2, minlength=nb), int(m.sum())


# ---------------------------------------------------------------- (a)
def test_a_exponential_range_recovered():
    rng = np.random.RandomState(3)
    a_true, c0, c1 = 15.0, 0.05, 1.0
    lats, lons, ws = [], [], []
    for p in range(10):                                     # 조각 10개: 위도 3° 간격 × 경도 20° 간격(조각 사이 최소 약 230 km > 200 km)
        la, lo = _box(rng, 300, 60.0 + 3.0 * (p % 5), 10.0 + 20.0 * (p // 5), 50.0)
        lr, lnr = np.radians(la), np.radians(lo)
        D = M._hav_km(lr[:, None], lnr[:, None], lr[None, :], lnr[None, :])
        C = c1 * np.exp(-D / a_true) + c0 * np.eye(len(la))
        L = np.linalg.cholesky(C + 1e-10 * np.eye(len(la)))
        ws.append(L @ rng.randn(len(la))); lats.append(la); lons.append(lo)
    lat, lon, w = np.concatenate(lats), np.concatenate(lons), np.concatenate(ws)
    rows, info = M.variogram_rows("R", "all", lat, lon, w - w.mean(), _grid_block(lat, lon), EDGES, chunk=128)
    fit = [r for r in rows if r["row"] == "fit"][0]
    assert info["n_pairs_beyond"] > 0                       # 조각 사이 쌍은 세지 않는다
    assert "no_converge" not in fit["fit_flag"] and "few_bins" not in fit["fit_flag"], fit
    assert 0.5 * a_true <= fit["range_a_km"] <= 2.0 * a_true, fit
    assert abs(fit["practical_range_km"] - 3.0 * fit["range_a_km"]) < 1e-9
    assert len([r for r in rows if r["row"] == "bin"]) == NB


# ---------------------------------------------------------------- (b)
def test_b_independent_noise_rho_zero():
    rng = np.random.RandomState(5)
    lat, lon = _box(rng, 3000, 65.0, -150.0, 30.0)
    w = rng.randn(3000)
    rows, info = M.variogram_rows("R", "all", lat, lon, w - w.mean(), _grid_block(lat, lon), EDGES, chunk=512)
    assert info["n_pairs_beyond"] == 0
    big = [r for r in rows if r["row"] == "bin" and r["n_pairs"] >= 20000]
    assert len(big) >= 3
    for r in big:
        assert abs(r["rho"]) <= 0.05, r


# ---------------------------------------------------------------- (c)
def test_c_pair_counts_chunks_and_jackknife():
    rng = np.random.RandomState(7)
    n = 220
    lat = rng.uniform(60.0, 64.0, n); lon = rng.uniform(0.0, 12.0, n)       # 약 450 × 630 km: 200 km 밖 쌍이 있다
    w = rng.randn(n)
    blk = _grid_block(lat, lon)
    acc = M.pair_accumulate(lat, lon, w, blk, EDGES, chunk=17)
    cnt, s2, n_in = _brute(lat, lon, w)
    np.testing.assert_array_equal(acc["npair"], cnt)
    np.testing.assert_allclose(acc["s2"], s2, rtol=1e-10, atol=1e-12)
    assert acc["n_total"] == n * (n - 1) // 2 and acc["n_within"] == n_in == int(acc["npair"].sum())
    assert 0 < n_in < acc["n_total"]
    # 청크 크기 무관
    acc2 = M.pair_accumulate(lat, lon, w, blk, EDGES, chunk=512)
    for k in ("npair", "s2", "sr", "sh", "nsame", "cell_s2", "cell_n", "blk_s2", "blk_n", "tot_s2", "tot_n"):
        np.testing.assert_allclose(acc[k], acc2[k], rtol=1e-10, atol=1e-10)
    # 셀별 누적의 합 = 2 × 쌍 합, 블록 셀별 누적의 합 = 2 × 쌍 수
    np.testing.assert_allclose(acc["cell_s2"].sum(0), 2 * acc["s2"], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(acc["tot_n"].sum(0), 2 * acc["npair"])
    # 모든 쌍이 구간 안: 합 = N(N − 1)/2, 쌍 가중 gamma 평균 = var(w, ddof 1)
    la2, lo2 = _box(rng, 150, 62.0, 5.0, 20.0)
    w2 = rng.randn(150)
    acc3 = M.pair_accumulate(la2, lo2, w2, _grid_block(la2, lo2), EDGES)
    assert int(acc3["npair"].sum()) == 150 * 149 // 2
    np.testing.assert_allclose(acc3["s2"].sum() / acc3["npair"].sum(), np.var(w2, ddof=1), rtol=1e-10)
    # 블록 하나 제외 잭나이프: 실제로 블록을 뺀 재계산과 같다
    G = M.jackknife_gammas(acc)
    blocks = acc["blocks"]
    assert G.shape == (len(blocks), NB)
    with np.errstate(invalid="ignore", divide="ignore"):
        for bi in range(len(blocks)):
            keep = blk != blocks[bi]
            c_b, s_b, _ = _brute(lat[keep], lon[keep], w[keep])
            g_b = np.where(c_b > 0, s_b / np.where(c_b > 0, c_b, 1), np.nan)
            np.testing.assert_allclose(G[bi], g_b, rtol=1e-9, atol=1e-12, equal_nan=True)
    se = M.jackknife_se(G)
    Bn = len(blocks)
    for k in range(NB):
        col = G[:, k]
        if np.all(np.isfinite(col)):
            ref = np.sqrt((Bn - 1) / Bn * ((col - col.mean()) ** 2).sum())
            assert abs(se[k] - ref) < 1e-12
        else:
            assert np.isnan(se[k])


# ---------------------------------------------------------------- (d)
def test_d_icc_recovery_and_formula():
    rng = np.random.RandomState(11)
    B = 400
    sizes = rng.randint(2, 16, B)
    tau2, sig2 = 0.09, 0.16
    blk = np.repeat(np.arange(B), sizes)
    w = np.repeat(rng.randn(B) * np.sqrt(tau2), sizes) + rng.randn(int(sizes.sum())) * np.sqrt(sig2)
    r = M.icc_row("R", "all", w, blk, 500, 1)
    assert abs(r["sigma2"] / sig2 - 1) < 0.10, r
    assert abs(r["tau_b2"] / tau2 - 1) < 0.25, r
    assert abs(r["icc"] / (tau2 / (tau2 + sig2)) - 1) < 0.20, r
    assert r["icc_lo"] < r["icc_hi"] and r["sd_within_lo"] < r["sd_within_hi"]
    assert r["icc_lo"] <= r["icc"] <= r["icc_hi"]
    assert r["n_blocks"] == B and r["n_blocks_ge2"] == B and r["n_cells"] == int(sizes.sum())
    r2 = M.icc_row("R", "all", w, blk, 500, 1)
    assert (r2["icc_lo"], r2["icc_hi"]) == (r["icc_lo"], r["icc_hi"])
    # 수계산: 블록 [1, 2, 3], [10, 12], [5]
    wt = np.array([1.0, 2.0, 3.0, 10.0, 12.0, 5.0]); bt = np.array([0, 0, 0, 1, 1, 2])
    _, n, mean, ssw = M.block_moments(wt, bt)
    st = M.icc_stats(n, mean, ssw)
    s2_ref = 4.0 / 3.0
    tau_ref = 21.0 - s2_ref * (1 / 3 + 1 / 2 + 1) / 3
    assert abs(st["sigma2"][0] - s2_ref) < 1e-12 and abs(st["tau_b2"][0] - tau_ref) < 1e-12
    assert abs(st["icc"][0] - tau_ref / (tau_ref + s2_ref)) < 1e-12
    # 다중도 가중 = 복제한 자료
    stm = M.icc_stats(n, mean, ssw, np.array([[2.0, 0.0, 1.0]]))
    std_ = M.icc_stats(np.array([3.0, 3.0, 1.0]), np.array([2.0, 2.0, 5.0]), np.array([2.0, 2.0, 0.0]))
    for k in ("sigma2", "tau_b2", "icc", "sd_within"):
        assert abs(stm[k][0] - std_[k][0]) < 1e-12
    # 다중도 행렬: 행 합 = 블록 수, 같은 seed 에서 같은 값
    Mb = M.block_boot_mult(7, 50, 3)
    assert Mb.shape == (50, 7) and np.all(Mb.sum(1) == 7)
    np.testing.assert_array_equal(Mb, M.block_boot_mult(7, 50, 3))


# ---------------------------------------------------------------- (e)
def _prox_data(rng):
    lat_t, lon_t = _box(rng, 80, 62.0, -120.0, 50.0)
    lat_s, lon_s = _box(rng, 40, 64.0, -115.0, 40.0)
    lat = np.r_[lat_t, lat_s]; lon = np.r_[lon_t, lon_s]
    A_idx = np.sort(rng.choice(80, 50, replace=False))
    evB = np.setdiff1d(np.arange(80), A_idx)
    return lat, lon, A_idx, evB, np.arange(80, 120), _grid_block(lat, lon)


def test_e_proximity_matches_h40_draws():
    rng = np.random.RandomState(13)
    lat, lon, A_idx, evB, src, blk = _prox_data(rng)
    dists = [1, 2, 5, 10, 25, 50, 100]
    rows, sels = M.proximity_split("Canada", "x", 2, lat, lon, A_idx, evB, src, blk, [10, 40, 60, -1], 3, dists, return_sel=True)
    nA = len(A_idx)
    assert set(sels) == {(10, 0), (10, 1), (10, 2), (40, 0), (40, 1), (40, 2), (-1, 0)}     # n = 60 ≥ |A| 는 없다
    for (n, d), sel in sels.items():
        np.testing.assert_array_equal(sel, H.draw_cells("Canada", "x", 2, n, d, nA))
    df = pd.DataFrame(rows)
    assert set(df.n) == {10, 40, -1}
    # 수계산(n = 10, d = 5 km)
    fr = []
    for d in range(3):
        lab = A_idx[H.draw_cells("Canada", "x", 2, 10, d, nA)]
        dist = LC.nearest_km(lat[evB], lon[evB], lat[lab], lon[lab])
        fr.append(np.mean(dist <= 5.0))
    r = df[(df.n == 10) & (df.d_km == 5.0)].iloc[0]
    assert abs(r.frac_within - np.mean(fr)) < 1e-12 and r.frac_within_min == min(fr) and r.frac_within_max == max(fr)
    assert r.n_eval == len(evB) and r.n_lab == 10 and r.n_draws == 3
    # d_km 에 단조, all 은 A 전체 기준과 같다
    for n, g in df.groupby("n"):
        v = g.sort_values("d_km").frac_within.values
        assert np.all(np.diff(v) >= -1e-12)
    al = df[df.n == -1]
    np.testing.assert_allclose(al.frac_within.values, al.frac_within_allA.values)
    assert (al.n_lab == nA).all()
    dS = LC.nearest_km(lat[evB], lon[evB], lat[src], lon[src])
    assert abs(df.med_src_km.iloc[0] - np.median(dS)) < 1e-9
    # 분할 평균 행
    rows3 = M.proximity_split("Canada", "x", 3, lat, lon, A_idx, evB, src, blk, [10], 3, dists)
    summ = pd.DataFrame(M.proximity_summary(rows + rows3))
    s = summ[(summ.n == 10) & (summ.d_km == 5.0)].iloc[0]
    r3 = pd.DataFrame(rows3); r3 = r3[r3.d_km == 5.0].iloc[0]
    assert s.split == -1 and s.n_splits == 2 and s.n_draws == 6
    assert abs(s.frac_within - (r.frac_within + r3.frac_within) / 2) < 1e-12
    assert s.frac_within_min == min(r.frac_within_min, r3.frac_within_min)
    assert s.frac_within_max == max(r.frac_within_max, r3.frac_within_max)


# ---------------------------------------------------------------- (f)
def _sar_data(rng, B=12):
    sizes = rng.randint(5, 12, B)
    blk = np.repeat(np.arange(B) + 1000, sizes)
    n = int(sizes.sum())
    base = rng.randn(n)
    x = 60 + 8 * (0.5 * base + rng.randn(n))
    y = 60 + 8 * (base + 0.5 * rng.randn(n))
    return x, y, blk, sizes


def test_f_sar_within_invariance_and_cell_stats():
    from scipy.stats import spearmanr
    rng = np.random.RandomState(17)
    x, y, blk, sizes = _sar_data(rng)
    s1 = M.sar_row_stats(x, y, blk, "cm", 300, 5)
    sx = np.repeat(rng.randn(len(sizes)) * 20, sizes); sy = np.repeat(rng.randn(len(sizes)) * 30, sizes)
    s2 = M.sar_row_stats(x + sx, y + sy, blk, "cm", 300, 5)
    assert np.isfinite(s1["r_within"])
    assert abs(s1["r_within"] - s2["r_within"]) < 1e-10
    assert abs(s1["r_within_lo"] - s2["r_within_lo"]) < 1e-9 and abs(s1["r_within_hi"] - s2["r_within_hi"]) < 1e-9
    assert s1["r_within_lo"] <= s1["r_within"] <= s1["r_within_hi"]
    assert abs(s1["r_cell"] - np.corrcoef(x, y)[0, 1]) < 1e-12
    assert abs(s1["rho_cell"] - spearmanr(x, y).correlation) < 1e-12
    assert abs(s1["bias_cm"] - np.mean(x - y)) < 1e-12 and abs(s1["rmse_cm"] - np.sqrt(np.mean((x - y) ** 2))) < 1e-12
    assert s1["n_blocks_within"] == len(sizes) and s1["n_blocks_between"] == len(sizes)
    # 합동 Pearson(블록 평균 제거) 수계산
    dx = x - pd.Series(x).groupby(blk).transform("mean").values
    dy = y - pd.Series(y).groupby(blk).transform("mean").values
    assert abs(s1["r_within"] - np.corrcoef(dx, dy)[0, 1]) < 1e-10
    # 로그 척도와 셀 부족
    s3 = M.sar_row_stats(x, y, blk, "log", 0, 5)
    assert np.isfinite(s3["r_within"]) and np.isnan(s3["r_within_lo"])
    assert abs(s3["bias_cm"] - s1["bias_cm"]) < 1e-12 or s3["n_cells"] < s1["n_cells"]
    s4 = M.sar_row_stats(x[:20], y[:20], blk[:20], "cm", 100, 5)
    assert "n<30" in s4["flag"] and np.isnan(s4["r_cell"]) and s4["n_cells"] == 20
    with pytest.raises(ValueError):
        M.sar_row_stats(x, y, blk, "sqrt", 0, 5)


def test_f2_polsar_codes_and_summary():
    v = np.array([50, 60, 100, 100, 100, 99.7, np.nan, 120, 100], float)
    valid = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0], bool)
    y = np.array([40, 55, 70, 80, 90, 60, 50, 65, 75], float)
    rows = M.sar_code_rows("polsar_alt", "U", v, valid, y)
    s = rows[0]
    assert s["kind"] == "summary" and s["n_ge99_5_valid"] == 5 and s["n_ge99_5_invalid"] == 1
    assert s["n_eq_100"] == 3 and s["n_window_95_105"] == 4 and s["n_distinct_ge99_5"] == 3
    assert abs(s["frac_integer_ge99_5"] - 0.8) < 1e-12 and s["n_valid"] == 7 and s["max_value"] == 120
    assert s["y_median_ge99_5"] == np.median([70, 80, 90, 60, 65]) and s["y_median_lt99_5"] == np.median([40, 55])
    vr = [r for r in rows if r["kind"] == "value"]
    assert vr[0]["value"] == 100.0 and vr[0]["count"] == 3 and len(vr) == 3
    tab = M._frame([dict(product="polsar_alt", unit=u, stratum="all", scale="cm", n_cells=40, n_blocks=5, r_within_lo=lo, nboot=2000)
                    for u, lo in (("AL-1", 0.3), ("AL-2", 0.1), ("CA-1", np.nan), ("Alaska", 0.9))], M.SAR_COLS)
    summ = M.sar_summary_rows(tab)
    assert len(summ) == 1 and summ[0]["unit"] == "SUMMARY9"
    assert summ[0]["n_units_eval"] == 2 and summ[0]["n_units_pass"] == 1 and "AL-1" in summ[0]["flag"]


# ---------------------------------------------------------------- (g)
def test_g_refuses_without_permission(monkeypatch):
    monkeypatch.delenv("LG_RESCALE", raising=False)

    def boom(*a, **k):
        raise AssertionError("허용 표지 없이 자료를 읽으면 안 된다")

    monkeypatch.setattr(H, "Data", boom)
    for argv in (["--items", "icc"], ["--smoke"], ["--precheck"], ["--summarize-only"], ["--no-summarize"]):
        with pytest.raises(SystemExit):
            M.main(argv)
    with pytest.raises(SystemExit):                          # 로컬 스레드 합계 상한(4 × 4 > 8)
        M.main(["--allow-local", "--workers", "4", "--threads", "4"])
    monkeypatch.setenv("LG_RESCALE", "1")                    # 개정 1: LG_RESCALE=1 로 허용된 실행에도 상한을 적용한다
    with pytest.raises(SystemExit) as ei:
        M.main(["--workers", "4", "--threads", "4"])
    assert "스레드 합계" in str(ei.value)
    monkeypatch.delenv("LG_RESCALE", raising=False)
    a = M.parse_args([])
    assert a.workers == 2 and a.threads == 4 and a.gpus == "" and a.NBOOT == 2000 and a.chunk == 512
    assert a.OUT == ROOT / "data" / "processed" / "lgu" and a.PREFIX == "lgu_c" and a.TAG == "lguc"
    assert a.TARGETS == M.DEFAULT_TARGETS and len(a.TARGETS) == 15 and a.REGIONS == ["Alaska", "Lena", "Canada"]
    assert a.N_GRID == [10, 40, 160, -1] and a.SPLITS == [1, 2, 3, 4, 5] and list(a.EDGES) == list(EDGES)
    s = M.parse_args(["--smoke"])
    assert s.REGIONS == ["Canada"] and s.TARGETS == [("Canada", "x")] and s.SPLITS == [1] and s.NBOOT == 200 and s.PREFIX == "lgu_c_smoke"
    with pytest.raises(SystemExit):
        M.parse_args(["--smoke", "--precheck"])
    with pytest.raises(SystemExit):
        M.parse_args(["--bins", "0,5,2"])
    assert M._peek_threads("4", argv=["x", "--threads", "3"], environ={}) == "1"
    assert M._peek_threads("4", argv=["x", "--allow-local", "--threads", "3"], environ={}) == "3"
    assert M._peek_threads("4", argv=["x"], environ={"LG_RESCALE": "1"}) == "4"
    assert M._target_pairs("Canada,AL-3,LE-1:x") == [("Canada", "x"), ("AL-3", "i"), ("LE-1", "x")]


# ---------------------------------------------------------------- (i)
def _write_sar_shards(a, tasks):
    for _, u in tasks:
        p = M.shard_paths(a, "sar", u)
        rows = [dict(product="polsar_alt", unit=u, stratum="all", scale="cm", n_cells=40, n_blocks=5,
                     r_within_lo=0.3 if u == "AL-1" else 0.1, nboot=2000)]
        LC.atomic_text(p["part"], M._frame(rows, M.SAR_COLS).to_csv(index=False))
        LC.atomic_text(p["codes"], M._frame([], M.CODE_COLS).to_csv(index=False))
        cfg = M.task_cfg(a, "sar", u)
        LC.atomic_text(p["unit"], json.dumps(dict(status="ok", cfg=cfg, cfg_hash=LC.cfg_hash(cfg), meta={}, n_rows=1)))


def test_i_summarize_from_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "input_hashes", lambda a: {"fidelity_base_v3.csv": "test"})
    a = M.parse_args(["--out-dir", str(tmp_path), "--items", "sar", "--sar-units", "AL-1,AL-2"])
    tasks = M.build_tasks(a)
    assert tasks == [("sar", "AL-1"), ("sar", "AL-2")]
    _write_sar_shards(a, tasks)
    assert M.summarize(a, tasks) == 0
    tab = pd.read_csv(M.out_path(a, "sar.csv"))
    s = tab[tab.unit == "SUMMARY9"]
    assert len(s) == 1 and int(s.n_units_pass.iloc[0]) == 1 and int(s.n_units_eval.iloc[0]) == 2
    assert M.item_unit_path(a, "sar").exists() and M.out_path(a, "sar_codes.csv").exists()
    meta = json.loads(M.out_path(a, "meta.json").read_text(encoding="utf-8"))
    assert meta["blind"] == M.BLIND and meta["n_failed"] == 0
    assert list(tab.columns) == M.SAR_COLS
    # 설정 해시가 다르면 표를 쓰지 않고 종료 코드 1
    M.out_path(a, "sar.csv").unlink()
    a2 = M.parse_args(["--out-dir", str(tmp_path), "--items", "sar", "--sar-units", "AL-1,AL-2", "--nboot", "100"])
    assert M.summarize(a2, tasks) == 1
    assert not M.out_path(a2, "sar.csv").exists()
    fl = pd.read_csv(M.out_path(a2, "failed.csv"))
    assert len(fl) == 2 and fl.reason.str.contains("설정 불일치").all()


# ---------------------------------------------------------------- (j)
class _FakeD:
    def __init__(self, df):
        self.df = df
        self.macros = set(df.macro.unique())


def _fake(rng):
    parts = []
    for reg, n, lat0, lon0 in (("R1", 240, 64.0, -150.0), ("R2", 80, 60.0, 100.0)):
        la, lo = _box(rng, n, lat0, lon0, 60.0)
        parts.append(pd.DataFrame(dict(macro=reg, lat=la, lon=lo)))
    df = pd.concat(parts, ignore_index=True)
    n = len(df)
    df["loc_id"] = np.arange(n)
    df["block"] = _grid_block(df.lat.values, df.lon.values)
    df["s"] = rng.uniform(20, 40, n)
    df["y"] = df.s * np.exp(0.3 + 0.2 * rng.randn(n)) * 1.5
    df["z"] = np.log(df.y) - np.log(df.s)
    df["sub"] = np.where(df.macro == "R1", np.where(df.lat > 64.0, "S1", "S2"), "")
    df["insar_alt"] = df.y * np.exp(0.2 * rng.randn(n))
    pol = df.y * np.exp(0.3 * rng.randn(n))
    pol[rng.rand(n) < 0.1] = 100.0
    df["polsar_alt"] = pol
    df["polsar_valid"] = (rng.rand(n) < 0.9).astype(float)
    df["polsar_std"] = rng.uniform(5, 10, n)
    i = np.arange(n)
    r1 = df.macro.values == "R1"
    gpr = r1 & (i % 2 == 0)                                  # R1: gpr 120셀, probe 120셀
    gpr[np.where(~r1)[0][:10]] = True                        # R2: gpr 10셀(30 미만이므로 행을 만들지 않는다), probe 70셀
    FL = dict(early=(i % 3 == 0), gpr=gpr, probe=~gpr)
    return _FakeD(df), FL


def test_j_pipeline_on_fake_data(tmp_path, monkeypatch):
    rng = np.random.RandomState(23)
    D, FL = _fake(rng)
    monkeypatch.setattr(M, "get_data", lambda a: D)
    monkeypatch.setattr(M, "get_flags", lambda a, D_, missing_ok=False: FL)
    monkeypatch.setattr(M, "input_hashes", lambda a: {"fidelity_base_v3.csv": "fake"})
    a = M.parse_args(["--out-dir", str(tmp_path), "--items", "variogram,icc,sar", "--regions", "R1,R2", "--sar-units", "R1,S1",
                      "--nboot", "50", "--workers", "1", "--threads", "1"])
    tasks = M.build_tasks(a)
    recs = M.run_tasks(a, tasks, [], 1)
    assert [r["status"] for r in recs] == ["ok"] * len(tasks), recs
    assert M.summarize(a, tasks) == 0
    vg = pd.read_csv(M.out_path(a, "variogram.csv"))
    assert set(vg.region) == {"R1", "R2"} and set(vg.row) == {"bin", "fit"}
    assert set(vg[vg.region == "R1"].stratum) == {"all", "gpr", "probe", "noearly"}
    assert set(vg[vg.region == "R2"].stratum) == {"all", "probe", "noearly"}    # R2 의 gpr 층은 10셀(30 미만)
    meta0 = json.loads(M.out_path(a, "meta.json").read_text(encoding="utf-8"))
    assert any(r["region"] == "R2" and r["stratum"] == "gpr" for r in meta0["rows_not_made"])
    ic = pd.read_csv(M.out_path(a, "icc.csv"))
    assert len(ic[ic.region == "R1"]) == 4 and len(ic[ic.region == "R2"]) == 3 and ic.icc.between(0, 1).all()
    sar = pd.read_csv(M.out_path(a, "sar.csv"))
    assert set(sar[sar.unit != "SUMMARY9"].unit) == {"R1", "S1"}
    assert set(sar[sar["product"] == "insar_alt"].stratum) <= {"all", "gpr", "probe"}
    assert "lt99_5" in set(sar[sar["product"] == "polsar_alt"].stratum)
    codes = pd.read_csv(M.out_path(a, "sar_codes.csv"))
    s = codes[(codes["product"] == "polsar_alt") & (codes.unit == "R1") & (codes.kind == "summary")].iloc[0]
    ii = np.where(D.df.macro.values == "R1")[0]
    ok = D.df.polsar_valid.values[ii] > 0
    assert s.n_eq_100 == int((ok & (D.df.polsar_alt.values[ii] == 100.0)).sum())
    meta = json.loads(M.out_path(a, "meta.json").read_text(encoding="utf-8"))
    assert meta["blind"] == M.BLIND and len(meta["cell_counts"]) >= 5
    # --resume: 모두 건너뛴다
    a.resume = True
    recs2 = M.run_tasks(a, tasks, [], 1)
    assert [r["status"] for r in recs2] == ["resumed"] * len(tasks)
    # 실패 기록: 없는 지역
    r = M.run_task(a, "icc", "Nowhere")
    assert r["status"] == "failed" and "Nowhere" in r["error"]
    u = json.loads(M.shard_paths(a, "icc", "Nowhere")["unit"].read_text(encoding="utf-8"))
    assert u["status"] == "failed"
    ok_, why = LC.unit_state(M.shard_paths(a, "icc", "Nowhere")["unit"], [], M.task_cfg(a, "icc", "Nowhere"))
    assert not ok_


# ---------------------------------------------------------------- (h)
@HEAVY
def test_h_smoke_exit_zero(tmp_path):
    if not (ROOT / "data" / "processed" / "fidelity_base_v3.csv").exists():
        pytest.skip("실자료가 없다")
    env = dict(os.environ)
    env.pop("LG_RESCALE", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, str(ROOT / "scripts" / "2_evaluation" / "h46_spatial_diagnostics.py"), "--smoke", "--allow-local",
           "--workers", "1", "--threads", "2", "--out-dir", str(tmp_path)]
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for name in ("variogram", "icc", "proximity", "sar", "sar_codes", "timing", "failed"):
        assert (tmp_path / f"lgu_c_smoke_{name}.csv").exists(), name
    prox = pd.read_csv(tmp_path / "lgu_c_smoke_proximity.csv")
    assert set(prox.target) == {"Canada"} and (prox.split == -1).any()
