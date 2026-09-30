"""scripts/4_visualization/paper/pool_fixed_curve.py 단위 시험(합성 조각만 쓴다. 실제 LG 조각과 결과 표를 열지 않는다).

(a) 구성 고정: 한 판의 모든 층화 평균 행이 같은 지역 집합을 쓰고, E2 의 n 은 두 지역 모두에 있는 n 이다(한 지역에만 있는 n 은 뺀다).
(b) 지역 하나의 대비가 없으면 그 n 의 층화 평균 행은 '판정 불가(구성 불완전)'이고 평균 값이 비어 있다(3지역 평균으로 줄이지 않는다).
(c) CI 풀 밖 지역(채점 블록 합집합 < 8)이 있으면 그 판의 모든 층화 평균 행이 판정 불가다.
(d) 두 가중 CI 가 계산되고 주 CI 와 점 추정이 h40.strat_mean(같은 seed·재표집 횟수)과 같다. 지역 행은 h40.contrast 와 같다.
(e) 4분 판정 열이 h42.verdict4(δ 0.5·1.0)와 같고, 우세·열세가 h40._sig 의 improve·worse 와 맞는다. 상수가 h42 기본값과 같다.
(f) 같은 입력이면 산출 CSV 가 바이트 단위로 같다. 메타의 seed 가 하네스 규칙(seed_of('lgboot', 이름))이다.
(g) 일부 분할에만 키가 있거나 조각이 모자라면 '부분(분할 k/K)' 표지가 붙고 값은 남는다.
(h) 명령행 --synthetic: 화면에 결과 값(소수)이 나오지 않고 _synthetic 산출과 메타가 생긴다.
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python3 -m pytest -q tests/test_pool_fixed_curve.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

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


P = _load("pool_fixed_curve", "scripts/4_visualization/paper/pool_fixed_curve.py")
X, H = P.X, P.H
from polar.h4_common import seed_of                                                                  # noqa: E402

NBOOT = 300
REGIONS = {
    "Lena": dict(nA=1200, grid=(0, 3, 10, 40, 160, 320, 1000, -1), blocks=16),
    "Canada": dict(nA=300, grid=(0, 3, 10, 40, 160, 320, -1), blocks=18),
    "Russia_W": dict(nA=16, grid=(0, 3, 10, -1), blocks=14),
    "Russia_E": dict(nA=15, grid=(0, 3, 10, -1), blocks=14),
}


def _run(tmp, name="base", regions=None, drop=None, nboot=NBOOT, remove=()):
    sh = tmp / f"{name}_shards"
    P.make_synthetic_shards(sh, regions=regions or REGIONS, drop=drop)
    for stem in remove:                                                 # 조각 누락 모사
        for suf in ("_blocksse.npz", "_runs.csv", "_unit.json"):
            (sh / f"{stem}{suf}").unlink()
    res = P.run(sh, tmp / f"{name}_out", nboot=nboot)
    return res, sh


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pool")
    res, sh = _run(tmp)
    return dict(res=res, df=res["df"], meta=json.loads(Path(res["meta"]).read_text()), shards=sh, tmp=tmp)


def _mean(df, ed, bl, m, n):
    q = df[(df.edition == ed) & (df.baseline == bl) & (df.method == m) & (df.n == n) & (df.scope == "MEAN")]
    assert len(q) == 1
    return q.iloc[0]


# ---------------------------------------------------------------- (a)
def test_composition_fixed_and_e2_n(base):
    df, meta = base["df"], base["meta"]
    for ed, spec in P.EDITIONS.items():
        m = df[(df.edition == ed) & (df.scope == "MEAN")]
        assert len(m) > 0
        assert set(m.edition_regions) == {",".join(spec["regions"])}
        assert set(m.target) == {f"MEAN[{','.join(spec['regions'])}]"}
        assert m.composition_complete.astype(bool).all()
        assert (m.n_ci_regions == len(spec["regions"])).all()
        assert set(m.pool_regions) == {",".join(spec["regions"])}
    e1 = df[(df.edition == "E1_P4_n_le_10") & (df.scope == "MEAN")]
    assert sorted(e1.n.unique()) == [0, 3, 10]
    e2 = df[(df.edition == "E2_LenaCanada_all_n") & (df.scope == "MEAN")]
    assert sorted(e2.n.unique()) == [-1, 0, 3, 10, 40, 160]                  # 320·1,000 은 레나에만 있다
    ex = meta["editions"]["E2_LenaCanada_all_n"]["excluded_n"]
    assert ex == {"320": ["Canada|x"], "1000": ["Canada|x"]}
    assert meta["editions"]["E2_LenaCanada_all_n"]["n_list"] == ["0", "3", "10", "40", "160", "all"]
    # 방법 × 기준선: P1 대 P1 은 없다
    assert not ((df.method == "P1") & (df.baseline == "P1")).any()
    assert set(df.method) == set(P.METHODS) and set(df.baseline) == {"P0", "P1"}


# ---------------------------------------------------------------- (b)
def test_missing_region_makes_pool_undetermined(tmp_path):
    def drop(t, mode, sp, key):
        return t == "Russia_W" and key[0] == "D0" and key[4] == 10
    res, sh = _run(tmp_path, "miss", drop=drop)
    df = res["df"]
    r = _mean(df, "E1_P4_n_le_10", "P0", "D0", 10)
    assert r.verdict4 == "판정 불가" and r.verdict4_d10 == "판정 불가"
    assert not bool(r.composition_complete)
    assert r.missing_regions == "Russia_W|x"
    assert str(r.status).startswith("판정 불가(구성 불완전")
    for c in ("delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq"):
        assert not np.isfinite(r[c])
    assert r.target == "MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"
    reg = df[(df.edition == "E1_P4_n_le_10") & (df.baseline == "P0") & (df.method == "D0") & (df.n == 10) & (df.scope == "region")]
    assert sorted(reg.target) == ["Canada|x", "Lena|x", "Russia_E|x"] and np.isfinite(reg.delta).all()
    # 같은 대비를 h42.pool_rows 로 그대로 내면 3지역 평균이 된다(이 모듈은 그 값을 쓰지 않는다)
    L = P.load_inputs(sh, "lg", P.P4, NBOOT)
    gA, gB = P.groups_for("D0", "P0", 10)
    per = {nm: X.region_stats(L["tms"][nm], gA, gB) for nm in P.P4}
    per = {k: v for k, v in per.items() if v is not None}
    raw = [q for q in X.pool_rows(per, P.P4, P.SimpleNamespace(delta_eq=0.5, delta_eq_aux=1.0), "x") if q["scope"] == "MEAN"][0]
    assert raw["n_ci_regions"] == 3 and np.isfinite(raw["delta"])
    # 다른 n 은 완전 구성
    for n in (0, 3):
        q = _mean(df, "E1_P4_n_le_10", "P0", "D0", n)
        assert bool(q.composition_complete) and q.status == "ok" and np.isfinite(q.delta)


# ---------------------------------------------------------------- (c)
def test_region_without_ci_blocks_whole_edition(tmp_path):
    reg = dict(REGIONS)
    reg["Russia_E"] = dict(nA=15, grid=(0, 3, 10, -1), blocks=5, eval_blocks=3)       # 채점 블록 합집합 ≤ 5 < 8
    res, _ = _run(tmp_path, "noci", regions=reg)
    df = res["df"]
    e1 = df[(df.edition == "E1_P4_n_le_10") & (df.scope == "MEAN")]
    assert (e1.verdict4 == "판정 불가").all() and (~e1.composition_complete.astype(bool)).all()
    assert set(e1.no_ci_regions) == {"Russia_E|x"} and e1.delta.isna().all()
    e2 = df[(df.edition == "E2_LenaCanada_all_n") & (df.scope == "MEAN")]
    assert e2.composition_complete.astype(bool).all() and np.isfinite(e2.delta).all()
    rr = df[(df.scope == "region") & (df.target == "Russia_E|x")]
    assert (rr.status == "CI 없음(CI 풀 밖)").all()


# ---------------------------------------------------------------- (d)
@pytest.mark.parametrize("ed,bl,m,n", [("E1_P4_n_le_10", "P1", "R1", 3), ("E1_P4_n_le_10", "P0", "D0", 0),
                                       ("E2_LenaCanada_all_n", "P0", "R2", -1), ("E2_LenaCanada_all_n", "P1", "V1", 160)])
def test_two_weight_ci_matches_strat_mean(base, ed, bl, m, n):
    df = base["df"]
    r = _mean(df, ed, bl, m, n)
    for c in ("ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq", "ci_lo_c", "ci_hi_c", "delta", "delta_blockeq", "p_eq"):
        assert np.isfinite(r[c]), c
    assert r.ci_lo <= r.delta <= r.ci_hi and r.ci_lo_beq <= r.delta_blockeq <= r.ci_hi_beq
    regions = P.EDITIONS[ed]["regions"]
    L = P.load_inputs(base["shards"], "lg", regions, NBOOT)
    gA, gB = P.groups_for(m, bl, n)
    ref = [q for q in H.strat_mean("ref", L["tms"], regions, lambda tm: gA, lambda tm: gB) if str(q["target"]).startswith("MEAN[")][0]
    for c in ("delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq"):
        assert r[c] == pytest.approx(ref[c], abs=1e-12), c
    assert ref["n_ci_regions"] == len(regions)
    for nm in regions:
        rg = df[(df.edition == ed) & (df.baseline == bl) & (df.method == m) & (df.n == n) & (df.scope == "region") & (df.target == nm)].iloc[0]
        c = H.contrast(L["tms"][nm], gA, gB, return_dist=True)
        assert rg.delta == pytest.approx(c["delta"], abs=1e-12) and rg.delta_blockeq == pytest.approx(c["delta_beq"], abs=1e-12)
        assert rg.ci_lo == pytest.approx(float(np.nanpercentile(c["dist"], 2.5)), abs=1e-12)
        assert rg.ci_hi_beq == pytest.approx(float(np.nanpercentile(c["dist_beq"], 97.5)), abs=1e-12)


def test_p1_baseline_equals_p0_at_n0(base):
    df = base["df"]
    for m in ("R1", "D0", "V1"):
        a = _mean(df, "E1_P4_n_le_10", "P0", m, 0); b = _mean(df, "E1_P4_n_le_10", "P1", m, 0)
        for c in ("delta", "ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq"):
            assert a[c] == pytest.approx(b[c], abs=1e-12)


# ---------------------------------------------------------------- (e)
def test_verdict_thresholds_match_h42_h40(base):
    df = base["df"]
    a = X.parse_args([])
    assert P.DELTA_EQ == a.delta_eq == 0.5 and P.DELTA_EQ_AUX == a.delta_eq_aux == 1.0
    ok = df[df.verdict4 != "판정 불가"]
    assert len(ok) > 50
    for r in ok.itertuples():
        assert r.verdict4 == X.verdict4(r.ci_lo, r.ci_hi, r.ci_lo_beq, r.ci_hi_beq, 0.5)
        assert r.verdict4_d10 == X.verdict4(r.ci_lo, r.ci_hi, r.ci_lo_beq, r.ci_hi_beq, 1.0)
        s = H._sig(r.ci_lo, r.ci_hi, r.ci_lo_beq, r.ci_hi_beq)
        assert (r.verdict4 == "우세") == (s == "improve") and (r.verdict4 == "열세") == (s == "worse")
    assert set(ok.verdict4) >= {"우세"}                                          # 합성 자료에서 판정이 실제로 갈린다
    assert (df.nboot == NBOOT).all() and (df.role == P.STATUS_LABEL).all()


# ---------------------------------------------------------------- (f)
def test_deterministic_and_harness_seeds(base, tmp_path):
    res2 = P.run(base["shards"], tmp_path / "again", nboot=NBOOT)
    assert Path(base["res"]["csv"]).read_bytes() == Path(res2["csv"]).read_bytes()
    meta = base["meta"]
    for nm, s in meta["seeds"].items():
        assert s["lgboot"] == seed_of("lgboot", nm) and s["lgxboot"] == seed_of("lgxboot", nm)
        assert s["split_seeds"] == {str(sp): seed_of(seed_of("lgboot", nm), sp) for sp in s["splits_used"]}
    assert meta["nboot"] == NBOOT and meta["delta_eq"] == 0.5 and meta["delta_eq_aux"] == 1.0
    assert meta["n_shards"] == 12 and len(meta["inputs"]) == 24 and all(len(i["sha256"]) == 64 for i in meta["inputs"])
    assert set(meta["code_sha256"]) >= {"module", "h40", "h42", "h4_common", "m1_stats"}


# ---------------------------------------------------------------- (g)
def test_partial_splits_are_marked(tmp_path):
    def drop(t, mode, sp, key):
        return t == "Lena" and key[4] == 40 and sp == 3 and key[0] != "P0"
    res, _ = _run(tmp_path, "part", drop=drop, remove=["lg__cpu__Canada__x__s3"])
    df = res["df"]
    meta = json.loads(Path(res["meta"]).read_text())
    assert meta["missing_shards"] == {"Canada|x": [2, 3]}
    r = _mean(df, "E2_LenaCanada_all_n", "P0", "R1", 40)
    assert bool(r.composition_complete) and np.isfinite(r.delta)
    assert str(r.status).startswith("부분(분할") and "Lena|x 2/3" in r.splits_short
    q = _mean(df, "E2_LenaCanada_all_n", "P0", "R1", 10)
    assert "Canada|x 2/3" in q.splits_short and np.isfinite(q.delta)


# ---------------------------------------------------------------- (h)
def test_cli_synthetic_prints_no_values(tmp_path, capsys):
    res = P.main(["--synthetic", "--out", str(tmp_path), "--nboot", "100"])
    out = capsys.readouterr().out
    assert Path(res["csv"]).name == "pool_fixed_curve_synthetic.csv" and Path(res["meta"]).exists()
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines and all(ln.startswith("[pool_fixed_curve]") for ln in lines)
    assert not re.search(r"\d\.\d", out)
    meta = json.loads(Path(res["meta"]).read_text())
    assert meta["synthetic"] is True and meta["nboot"] == 100
    with pytest.raises(SystemExit):
        P.parse_args([])                                                  # 실제 실행은 조각 폴더가 필요하다
