"""시험지 모듈(polar.testbed_v5) 단위 시험: 분할의 겹침 없음·덮음, 지표·부트스트랩의 기본 성질."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polar.testbed_v5 as T


def _toy(n=600, seed=0):
    rng = np.random.RandomState(seed)
    lat = 60 + rng.rand(n) * 5; lon = -150 + rng.rand(n) * 5
    d = pd.DataFrame(dict(lat=lat, lon=lon, year=rng.choice(np.arange(2010, 2020), n), loc_id=rng.choice(200, n),
                          region=rng.choice(["Alaska", "Lena", "Canada"], n)))
    d["block"] = np.floor(d.lat / 0.5).astype(int) * 100000 + np.floor(d.lon / 0.5).astype(int)
    d["alt_cm"] = 50 + rng.randn(n) * 10
    return d


def _check(folds, n):
    for f in folds:
        assert len(np.intersect1d(f.tr, f.te)) == 0
    te_all = np.concatenate([f.te for f in folds])
    return te_all


def test_random_block_year_cover():
    d = _toy()
    for scheme in ("random", "block", "year", "block_year"):
        folds = T.make_folds(d, scheme, k=4, seed=1)
        te = _check(folds, len(d))
        assert len(np.unique(te)) == len(te)
        if scheme in ("random", "block"):
            assert len(te) == len(d)
        if scheme == "block":
            for f in folds:
                assert not set(d.block.values[f.tr]) & set(d.block.values[f.te])
        if scheme == "year":
            for f in folds:
                assert len(set(d.year.values[f.te])) == 1 and set(d.year.values[f.tr]).isdisjoint(set(d.year.values[f.te]))
        if scheme == "block_year":
            for f in folds:
                assert set(d.year.values[f.tr]).isdisjoint(set(d.year.values[f.te]))
                assert not set(d.block.values[f.tr]) & set(d.block.values[f.te])


def test_region_buffer():
    d = _toy()
    d.loc[d.region == "Lena", "lon"] += 100
    d.loc[d.region == "Canada", "lon"] += 50
    folds = T.folds_region(d, buffer_km=100.0)
    assert len(folds) == 3
    for f in folds:
        assert len(set(d.region.values[f.te])) == 1
        assert set(d.region.values[f.tr]).isdisjoint(set(d.region.values[f.te]))


def test_metrics_and_boot():
    rng = np.random.RandomState(0)
    y = rng.randn(500) * 10 + 50; b = rng.choice(20, 500)
    pA = y + rng.randn(500) * 2; pB = y + rng.randn(500) * 6
    m = T.metrics(y, pA, b)
    assert abs(m["rmse"] - 2) < 0.5 and abs(m["bias"]) < 0.5 and m["r"] > 0.95
    bd = T.boot_delta(y, pA, pB, b, nboot=500)
    assert bd["delta"] < 0 and bd["hi"] < 0 and bd["n_blocks"] == 20
    loc = rng.choice(100, 500)
    am = T.anomaly_metrics(y, pA, loc)
    assert am["n_anom"] > 0 and am["r_anom"] > 0.9


def test_ls_scale():
    s = np.linspace(10, 40, 50); y = 1.7 * s
    assert abs(T.ls_scale(s, y) - 1.7) < 1e-9
    a, b = T.ls_affine(s, y + 3)
    assert abs(a - 3) < 1e-6 and abs(b - 1.7) < 1e-6
