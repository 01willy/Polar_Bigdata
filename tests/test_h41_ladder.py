"""scripts/2_evaluation/h41_validation_ladder.py 단위 시험.

이 서버는 공유 서버이므로 시험은 Rescale 사전 점검 작업에서 실행한다(로컬에서 pytest 를 돌리지 않는다).
학습을 하지 않는 시험은 합성 자료와 대체 적합기(numpy 최소제곱)를 쓴다. CatBoost 학습과 실자료가 필요한 시험(표지 HEAVY)은
환경 변수 LG_RUN_HEAVY=1 일 때만 실행한다.
(a) 모든 단에서 학습과 채점의 교집합이 없다. 묶음 단은 같은 묶음이 두 fold 에 걸치지 않는다.
(b) 버퍼 단(V-C100, V-C500, V-G, W2)은 버퍼 안 학습 셀이 0 개다. 버퍼로 뺀 셀은 haversine 전수 계산과 같다.
(c) V-G 의 학습 집합이 h40.Data.source_idx 와 같다.
(d) 각 채점 셀이 반복마다 한 번 예측된다. 조각 → 집계 → 표(L19–L22)가 만들어진다. 채점 범위가 부분인 (단, 반복)은 집계에서 빠지고
    나머지 단은 집계된다. (d2) 비유한 예측은 지역 단위로 빠진다(V-G 의 한 지역의 적합 실패가 다른 지역의 같은 키 행을 빼지 않는다).
(e) [HEAVY] h40.build_ctx('Lena', 'x', 1)의 A 를 지역 내 학습 fold, B 를 채점으로 둔 w2 모델의 P0, P1w, R1w[catboost_lo] 가
    h40.run_ctx 의 P0, P1(전량), R1(전량)과 같다(물리식 1e-9, CatBoost 1e-6). h42 의 기준 방법은 h40 과 같은 값이다(h42 시험 a).
(f) 채점 fold 의 라벨을 바꿔도 예측과 R1w@nest 의 λ 선택이 변하지 않는다(교란 불변). 학습 fold 의 라벨을 바꾸면 변한다(대조).
(g) verdict4 의 경계값, 층화 평균, 풀 지역 수, Holm, 부트스트랩 식(h4_common.boot_delta_blocks 와 같다).
    (g2) 풀 지역 3/5 의 판정 문구에는 '부분(지역 3/5)'가 붙는다. (g4) 판정 문구의 부분 표기와 효과 크기 표기(mark_verdict).
(h) 분할 함수: 균형, 재현성, 동률 규칙. (i) 기본값과 실행 거부. (j) --resume 과 설정 해시. (k) 거리 함수. (l) 적합 수(dry).
(m) N1 기준선: 대체 통계와 계수는 학습 셀에서만 구한다. 앵커는 5종이다(연도 정합 √TDD 포함). (n) [HEAVY] 실자료의 단 구조.
(p) [HEAVY] 프로세스 풀 경로(--workers 2)의 실행.
실행(Rescale): LG_RUN_HEAVY=1 CUDA_VISIBLE_DEVICES="" python3 -m pytest -q tests/test_h41_ladder.py
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
import pandas as pd
import pytest

HEAVY = pytest.mark.skipif(os.environ.get("LG_RUN_HEAVY", "") != "1",
                           reason="학습 또는 실자료가 필요한 시험은 LG_RUN_HEAVY=1 일 때만 실행한다(공유 서버 CPU 보호, Rescale 사전 점검에서 실행)")

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


H = _load("h40_label_grid", "scripts/3_deep_learning/h40_label_grid.py")
V = _load("h41_validation_ladder", "scripts/2_evaluation/h41_validation_ladder.py")
from polar.fidelity import TARGET, CLIMATE, SOIL                                                      # noqa: E402
from polar.m1_ext import haversine_km                                                                 # noqa: E402
import polar.h4_common as H4                                                                         # noqa: E402

REGIONS = [("Alaska", 66.0, -150.0, 90, 2.0, 5.0, 1.30), ("Canada", 66.0, -140.5, 40, 2.0, 5.0, 1.10), ("Lena", 72.0, 127.0, 40, 1.5, 4.0, 1.60),
           ("Russia_W", 67.0, 60.0, 16, 2.0, 6.0, 1.40), ("Russia_E", 68.0, 162.0, 16, 2.0, 6.0, 1.50), ("Russia_C", 70.0, 100.0, 4, 1.0, 3.0, 1.45),
           ("Greenland", 70.0, -45.0, 2, 0.2, 0.4, 1.20), ("Tibet", 35.0, 92.0, 2, 0.5, 1.0, 3.00)]
CELLS_PER_SITE = 4


def make_df(seed=0):
    """합성 자료. 지역마다 지점(site)을 뿌리고 지점마다 셀 4개를 0.02° 안에 둔다. 기후·토양·CCI 열은 지점 안에서 같다(V-P 묶음 = 지점).
    dem_elev(입력 0 열)은 행 식별자다."""
    rng = np.random.RandomState(seed)
    rows = []
    for reg, la0, lo0, n_site, dla, dlo, E in REGIONS:
        for _ in range(n_site):
            la, lo = la0 + rng.uniform(-dla, dla), lo0 + rng.uniform(-dlo, dlo)
            site = {c: float(rng.randn()) for c in CLIMATE + SOIL}
            tdd = float(rng.uniform(400, 1600)); site["e5_tdd"] = tdd; site["e5_sqrt_tdd"] = float(np.sqrt(tdd))
            cci = float(rng.uniform(30, 120)) if rng.rand() > 0.05 else np.nan
            for _ in range(CELLS_PER_SITE):
                r = dict(site)
                r.update(macro=reg, lat=la + rng.uniform(0, 0.02), lon=lo + rng.uniform(0, 0.02), cci_alt=cci, cci_valid=float(np.isfinite(cci)),
                         **{c: float(rng.randn()) for c in ("dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough")})
                s = r["e5_sqrt_tdd"]
                r[TARGET] = float(max(E * s + 4.0 * r["dem_slope"] + rng.randn() * 3.0, 5.0))
                r["e5_sqrt_tdd_soil"] = float(0.9 * s) if rng.rand() > 0.01 else np.nan
                r["p4_ku"] = float(1.2 * s + rng.randn()) if rng.rand() > 0.15 else np.nan
                r["p2_edaphic"] = float(1.1 * s + rng.randn())
                rows.append(r)
    df = pd.DataFrame(rows)
    df["dem_elev"] = np.arange(len(df), dtype=float)
    df["loc_id"] = 1000 + np.arange(len(df))
    df["block"] = (np.floor(df.lat / 0.5).astype(int) * 100000 + np.floor(df.lon / 0.5).astype(int))
    df["s"] = df.e5_sqrt_tdd.values.astype(float); df["y"] = df[TARGET].values.astype(float)
    df["z"] = np.log(df.y.values) - np.log(df.s.values)
    assert set(H.FEATS) <= set(df.columns)
    return df


def make_D(df, HA):
    """h40.Data 를 파일 없이 만든다(source_idx 와 target_idx 는 h40 의 코드를 그대로 쓴다)."""
    D = H.Data.__new__(H.Data)
    D.df = df; D.args = HA
    D.macros = set(df.macro.unique()); D.sub_parent = {}; D.subs = pd.DataFrame(columns=["subregion", "parent"])
    D.sub_src = "synthetic"; D.sub_kmeans_diff = 0
    D._src = {}; D._split = {}
    return D


class FakeFitter(V.FitterV):
    """학습 없는 대체 적합기: 절편과 입력 1–4 열의 최소제곱(numpy). 학습기와 seed 에 따라 상수를 더한다. 적합 수·추적은 본 적합기와 같다."""
    OFFSET = {"catboost_lo": 0.0, "catboost": 0.3, "rf": -0.2}

    def _fit(self, learner, axis, build, seed, preds, info, init, iters, cont):
        Xtr, ytr, w = build()
        self._trace(dict(info or {}, learner=learner, axis=axis, seed=seed), Xtr, ytr, w)

        def A(X):
            X = np.asarray(X, float)
            return np.c_[np.ones(len(X)), np.nan_to_num(X[:, 1:5])]
        beta = np.linalg.lstsq(A(Xtr), np.asarray(ytr, float), rcond=None)[0]
        return None, [A(p) @ beta + self.OFFSET[learner] * 0.01 + 0.001 * seed for p in preds]


N_CLUSTERS = 12          # 합성 자료의 군집 수(큰 지역이 여러 군집으로 나뉘어 버퍼가 이웃 군집의 셀을 빼게 한다)
# 합성 자료의 시험은 실자료의 연도 정합 도일 표를 읽지 않는다(loc_id 가 다르다). 없는 파일 이름을 준다
BASE_ARGS = ["--threads", "1", "--nboot", "300", "--n-clusters", str(N_CLUSTERS), "--reps", "2", "--within-reps", "2", "--seeds", "2",
             "--workers", "0", "--tdd-matched", "lgx_tdd_matched_not_used_in_tests.csv"]


def make_env(tmp_path, extra=(), seed=0, y_fn=None):
    a = V.parse_args(BASE_ARGS + ["--out-dir", str(tmp_path / "out"), "--cluster-map", str(tmp_path / "cmap.csv")] + list(extra))
    df = make_df(seed)
    if y_fn is not None:
        df = y_fn(df)
    Vd = V.VData(a, D=make_D(df, a.HA))
    if not a.CMAP.exists():
        V.write_cluster_map(a, Vd)
    return a, Vd


@pytest.fixture()
def fake_fitter():
    V.FITTER = FakeFitter
    try:
        yield
    finally:
        V.FITTER = None; H.TRACE = None


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("h41"))


# ================================================================ (a)–(c) 단 구조
def test_a_no_overlap_and_groups_intact(env):
    a, Vd = env
    units, skipped = V.enumerate_units(a, Vd)
    assert {u[0] for u in units} == set(V.schemes_to_run(a)) and len(units) > 60
    for s, rep, k in units:
        c = V.unit_cells(a, Vd, s, rep, k, check=True)
        tr, te, held = c["tr"], c["te"], c["held"]
        assert len(te) > 0 and len(tr) > 0
        assert not set(tr.tolist()) & set(te.tolist()) and not set(tr.tolist()) & set(held.tolist())
        assert np.all(Vd.score[te]) and set(te.tolist()) <= set(held.tolist())
        if s in V.GROUPED:
            g = Vd.groups(s)
            assert not set(g[tr].tolist()) & set(g[held].tolist()), f"{s}: 묶음이 학습과 채점에 걸쳐 있다"
        if V.scheme_kind(s) == "within":
            R = s[3:]
            assert set(Vd.macro[te]) == {R} and not set(Vd.block[c["trw"]].tolist()) & set(Vd.block[held].tolist())
            assert not np.any(Vd.macro[c["src"]] == R)
            assert np.array_equal(tr, np.concatenate([c["src"], c["trw"]]) if s.startswith("W2-") else c["trw"])
    for s in V.GROUPED:                                          # 같은 묶음이 두 fold 에 걸치지 않는다
        for rep in a.REPS:
            f = Vd.fold_of(a, s, rep)
            t = pd.DataFrame(dict(g=Vd.groups(s), f=f)).groupby("g").f.nunique()
            assert (t == 1).all() and set(f.tolist()) == set(range(a.folds))
    assert len(np.unique(Vd.groups("V-P"))) == sum(r[3] for r in REGIONS)               # V-P 묶음 = 지점
    f1, f2 = Vd.fold_of(a, "W2-Lena", 1), Vd.fold_of(a, "W1-Lena", 1)
    assert np.array_equal(f1, f2) and set(f1[Vd.macro != "Lena"].tolist()) == {-1}


def _brute_min_km(Vd, q, r):
    return np.array([haversine_km(Vd.lat[i], Vd.lon[i], Vd.lat[r], Vd.lon[r]).min() for i in q])


def test_b_buffer_cells_excluded(env):
    a, Vd = env
    units, _ = V.enumerate_units(a, Vd)
    n_checked = n_removed = 0
    for s, rep, k in units:
        kind = V.scheme_kind(s)
        if kind == "cluster" and V.cluster_buffer(s) > 0:
            d = V.cluster_buffer(s)
            c = V.unit_cells(a, Vd, s, rep, k, check=True)
            cand = np.where((Vd.fold_of(a, s, rep) != k) & Vd.trainable)[0]
            dist = _brute_min_km(Vd, cand, c["held"])
            assert np.array_equal(c["tr"], cand[dist >= d]), f"{s} f{k}: 버퍼로 뺀 셀이 haversine 전수 계산과 다르다"
            assert c["info"]["n_buffer_excluded"] == int((dist < d).sum())
            n_checked += 1; n_removed += int((dist < d).sum())
        elif kind == "region" or s.startswith("W2-"):
            c = V.unit_cells(a, Vd, s, rep, k, check=True)
            R = c["info"]["region"]
            src = c["tr"] if kind == "region" else c["src"]
            t_idx = np.where(Vd.macro == R)[0]
            assert _brute_min_km(Vd, src, t_idx).min() >= a.buffer_km
            n_checked += 1
    assert n_checked > 10 and n_removed > 0, "버퍼가 실제로 셀을 빼는 경우가 없다(합성 자료 확인)"
    # 단언 함수 자체: 버퍼 안 셀을 넣으면 잡아낸다
    ak, ca = np.where(Vd.macro == "Alaska")[0], np.where(Vd.macro == "Canada")[0]
    n_in = V.count_within_km(Vd.lat[ca], Vd.lon[ca], Vd.lat[ak], Vd.lon[ak], 100.0)
    assert n_in == int((_brute_min_km(Vd, ca, ak) < 100.0).sum()) and n_in > 0


def test_c_vg_train_equals_source_idx(env):
    a, Vd = env
    for k, R in enumerate(V.REG7):
        c = V.unit_cells(a, Vd, "V-G", 1, k, check=True)
        t_idx, _, src, comp = Vd.D.source_idx(R, "x")
        assert np.array_equal(c["tr"], src) and c["info"]["n_buffer_excluded"] == comp["n_buffer_excluded"]
        assert np.array_equal(c["te"], t_idx[Vd.score[t_idx]]) and c["info"]["region"] == R
        w = V.unit_cells(a, Vd, f"W2-{R}", 1, 0, check=True) if R in a.WREGIONS else None
        if w is not None:
            assert np.array_equal(w["src"], src)
    n_ca = Vd.D.source_idx("Alaska", "x")[3]["n_buffer_excluded"]
    assert n_ca > 0, "합성 자료에서 알래스카 버퍼가 캐나다 셀을 빼야 한다"


# ================================================================ (d) 조각과 집계
def _run_all(a, Vd):
    units, skipped = V.enumerate_units(a, Vd)
    units.sort(key=V.unit_priority)
    return units, skipped, [V.run_unit(a, Vd, *u) for u in units]


def test_d_each_cell_once_and_summary(tmp_path, fake_fitter):
    a, Vd = make_env(tmp_path)
    units, skipped, done = _run_all(a, Vd)
    assert all(u["status"] == "ok" for u in done) and not list(a.SHARDS.glob("*.tmp*"))
    sh = V.find_shards(a)
    assert len(sh) == len(units)
    cnt = {}
    for s in sh:
        with np.load(s["pred"], allow_pickle=False) as z:
            assert z["P"].shape == (len(z["keys"]), len(z["idx"])) and z["P"].dtype == np.float32
            assert np.array_equal(z["loc_id"], Vd.loc_id[z["idx"]]) and np.allclose(z["y"], Vd.y[z["idx"]])
            c = cnt.setdefault((s["scheme"], s["rep"]), np.zeros(Vd.N, int))
            np.add.at(c, z["idx"], 1)
    for (scheme, rep), c in cnt.items():
        want = Vd.score_of(scheme)
        assert np.all(c[want] == 1) and np.all(c[~want] == 0), f"{scheme} r{rep}: 채점 셀이 정확히 한 번 예측되지 않았다"
    assert {k[1] for k in cnt if k[0] == "V-R"} == set(a.REPS) and {k[1] for k in cnt if k[0] == "V-G"} == {1}
    PR, failed, excluded = V.load_preds(a, Vd, sh)
    assert not failed and not excluded and all(d["coverage"] == 1.0 and not d["bad_rows"] and not d["bad_reg"] for d in PR.values())
    out = V.summarize(a, Vd)
    met, con, tests = out["metrics"], out["contrasts"], out["tests"]
    for f in ("metrics", "contrasts", "tests", "timing", "failed"):
        assert (a.OUT / f"{a.TAG}_{f}.csv").exists()
    meta = json.loads((a.OUT / f"{a.TAG}_meta.json").read_text())
    assert meta["n_score"] == int(Vd.score.sum()) and meta["n_units"] == len(units)
    assert set(met.scheme) == set(V.schemes_to_run(a)) and {"region", "MEAN5", "MEAN4", "POOL"} <= set(met.scope)
    # 물리식의 점 추정은 직접 계산과 같다(V-G 의 PS = 원천 계수 Stefan)
    r = met[(met.scheme == "V-G") & (met.method == "PS") & (met.scope == "region") & (met.target == "Lena")].iloc[0]
    t_idx, _, src, _ = Vd.D.source_idx("Lena", "x")
    te = t_idx[Vd.score[t_idx]]
    E0 = H.ls_E(Vd.y[src], Vd.s[src])
    assert abs(r.rmse - float(np.sqrt(np.mean((E0 * Vd.s[te] - Vd.y[te]) ** 2)))) < 1e-4 and r.n_cells == len(te)
    assert abs(r.bias - float(np.mean(E0 * Vd.s[te] - Vd.y[te]))) < 1e-4
    # 블록 8개 미만 지역은 점 추정만 낸다
    small = met[(met.scope == "region") & met.target.isin(["Russia_C", "Greenland"])]
    assert len(small) and small.rmse_lo.isna().all() and small.rmse.notna().all()
    big = met[(met.scope == "region") & (met.target == "Alaska") & (met.scheme == "V-R")]
    assert big.rmse_lo.notna().all() and (big.rmse_lo < big.rmse_hi).all()
    assert ((big.rmse_lo <= big.rmse) & (big.rmse <= big.rmse_hi)).mean() > 0.9
    # 판정 표
    v = tests[tests.scope.isin(["verdict", "verdict_aux"])]
    assert {"L19", "L20", "L21", "L22"} <= set(v.test_id) and (v[v.role == "주"].groupby("test_id").size() == 1).all()
    main = tests[tests.holm_bundle == "X3a"]                                                  # Holm 묶음 X3a: L19 둘, L20, L21(λ 0.25)의 5지역 평균
    assert len(main) == 4 and set(main.test_id) == {"L19", "L20", "L21"} and set(main.scope) == {"MEAN5"} and set(main.role) == {"주"}
    jd = main[main.verdict4 != "판정 불가"]
    assert jd.holm_p.notna().all() and (jd.holm_p >= jd.p_boot - 1e-12).all() and main[main.verdict4 == "판정 불가"].holm_p.isna().all()
    w3 = tests[tests.holm_bundle == "X3b"]                                                    # Holm 묶음 X3b: L22 의 기준 λ 지역 대비
    assert len(w3) == len(a.WREGIONS) and set(w3.target) == set(a.WREGIONS) and set(w3.test_id) == {"L22"} and set(w3.role) == {"주"}
    jw = w3[w3.verdict4 != "판정 불가"]
    assert jw.holm_p.notna().all() and (jw.holm_p >= jw.p_boot - 1e-12).all()
    assert set(tests.verdict4.dropna()) <= {"우세", "열세", "동등", "미결정", "판정 불가"}
    assert tests[tests.test_id == "L19"].confirmatory.all() and not tests[tests.test_id != "L19"].confirmatory.any()
    assert set(tests.nboot) == {300} and meta["excluded"] == [] and meta["reps_short"] == {}
    vt0 = {(r.test_id, r.role, r.stat.split(" | ")[0]): r.verdict for r in v.itertuples()}
    l22 = tests[(tests.test_id == "L22") & (tests.role == "주") & (tests.scope == "region")]
    assert set(l22.target) == set(a.WREGIONS) and not l22[l22.target == "Alaska"].blind.any() and l22[l22.target != "Alaska"].blind.all()
    # '#ex' 대비는 두 쪽을 같은 셀로 제한한다
    ex = con[(con.method_A == "P0@ku#ex") & (con.scope == "region") & (con.target == "Alaska")].iloc[0]
    full = con[(con.method_A == "P0@ku") & (con.scope == "region") & (con.target == "Alaska")].iloc[0]
    n_ak = int((Vd.score & (Vd.macro == "Alaska")).sum()); n_ok = int((Vd.score & (Vd.macro == "Alaska") & np.isfinite(Vd.ku)).sum())
    assert ex.n_cells == n_ok < n_ak == full.n_cells
    # 이중 차분은 두 대비의 차와 같다
    dd = con[(con.family == "double_diff") & (con.scheme_A == "V-G") & (con.learner_A == "catboost_lo") & (con.lam_A == 0.25) & (con.scope == "MEAN5")]
    rs = con[(con.family == "degradation") & (con.scheme_A == "V-G") & (con.method_A == "RS") & (con.learner_A == "catboost_lo") & (con.lam_A == 0.25)
             & (con.scope == "MEAN5")]
    d0 = con[(con.family == "degradation") & (con.scheme_A == "V-G") & (con.method_A == "D0") & (con.learner_A == "catboost_lo") & (con.scope == "MEAN5")]
    assert len(dd) == len(rs) == len(d0) == 1 and abs(float(dd.delta.iloc[0]) - (float(rs.delta.iloc[0]) - float(d0.delta.iloc[0]))) < 1e-9
    # 조각 하나를 지우면 그 (단, 반복)만 집계에서 빠지고 나머지 단은 집계된다. --allow-partial 이면 통계에 넣는다
    one = [s for s in sh if s["scheme"] == "V-B" and s["rep"] == 1][0]
    one["pred"].unlink(); one["unit"].unlink()
    out1 = V.summarize(a, Vd)
    assert out1["excluded"] == [("V-B", 1)] and out1["meta"]["excluded"] == [["V-B", 1]] and out1["meta"]["reps_short"] == {"V-B": dict(have=[2], expected=[1, 2])}
    assert any(f["scheme"] == "V-B" and f["reason"] == "채점 범위 부분, 집계 제외" for f in out1["failed"])
    m1 = out1["metrics"]
    vb = m1[(m1.scheme == "V-B") & (m1.method == "D0") & (m1.learner == "catboost_lo") & (m1.scope == "region") & (m1.target == "Alaska")].iloc[0]
    assert vb.n_rows == 2 and float(m1[m1.scheme == "V-B"].coverage.iloc[0]) == 1.0, "남은 반복 하나(seed 2개)만 쓴다"
    t1 = out1["tests"]
    v1 = t1[t1.scope.isin(["verdict", "verdict_aux"])]
    vt1 = {(r.test_id, r.role, r.stat.split(" | ")[0]): r.verdict for r in v1.itertuples()}
    for k_ in vt0:                                                                            # V-B 를 쓰지 않는 가설(L19, L21, L22)의 판정은 그대로다
        if k_[0] != "L20":
            assert vt1[k_] == vt0[k_], k_
    l20 = v1[(v1.test_id == "L20") & (v1.role == "주")].iloc[0]
    assert l20.verdict.startswith(("부분(", "판정 불가")) and "V-B 반복 [1]" in l20.stat
    a.allow_partial = True
    out2 = V.summarize(a, Vd)
    assert float(out2["metrics"][out2["metrics"].scheme == "V-B"].coverage.iloc[0]) < 1.0 and out2["excluded"] == []
    a.allow_partial = False


def test_d2_nonfinite_rows_are_dropped_by_region(tmp_path, fake_fitter):
    """V-G 의 한 지역(Russia_C)에서 D0[catboost_lo] seed 0 의 예측이 비유한이면 그 지역의 통계에서만 그 행을 뺀다."""
    a, Vd = make_env(tmp_path)
    a.schemes = "V-R,V-G"; a.SCHEMES = ["V-R", "V-G"]; a.part = "pooled"
    units, skipped, done = _run_all(a, Vd)
    out0 = V.summarize(a, Vd)
    k5 = V.REG7.index("Russia_C")
    p5 = V.shard_paths(a, "V-G", 1, k5)["pred"]
    with np.load(p5, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    keys = [tuple(json.loads(str(s))) for s in d["keys"]]
    i0 = [i for i, k in enumerate(keys) if (k[0], k[1], int(k[2]), float(k[3])) == ("D0", "catboost_lo", 0, 1.0)][0]
    P = d["P"].copy(); P[i0, :] = np.nan
    np.savez_compressed(p5, **dict(d, P=P))
    PR, failed, excluded = V.load_preds(a, Vd, V.find_shards(a))
    g = ("D0", "catboost_lo", 1.0)
    row = PR[("V-G", 1)]["keys"].index(("D0", "catboost_lo", 0, 1.0))
    assert not excluded and PR[("V-G", 1)]["bad_reg"] == {row: {"Russia_C"}} and PR[("V-G", 1)]["coverage"] == 1.0
    assert any("Russia_C" in f["detail"] and f["reason"].startswith("비유한 예측") for f in failed)
    G = V.Agg(a, Vd, PR, excluded)
    assert len(G.rows_of("V-G", g, "Alaska")) == 2 and len(G.rows_of("V-G", g, "Russia_C")) == 1 and len(G.rows_of("V-G", g, None)) == 1
    assert G.expected_rows("V-G", g) == 2 and G.expected_rows("V-R", g) == 4 and G.expected_rows("V-G", V.g_("PS")) == 1
    assert G.term("V-G", g, "Alaska")["n_rows"] == 2 and G.term("V-G", g, "Russia_C")["n_rows"] == 1 and G.term("V-G", g, "POOL")["n_rows"] == 1
    out1 = V.summarize(a, Vd)
    c0 = out0["contrasts"]; c1 = out1["contrasts"]

    def pick(c, scope, target=None):
        q = c[(c.contrast == "D0[catboost_lo,1.0]-PS") & (c.scheme_A == "V-G") & (c.scope == scope)]
        return (q if target is None else q[q.target == target]).iloc[0]
    for sc, tg in (("region", "Alaska"), ("region", "Lena"), ("MEAN5", None)):               # 주 5지역의 값은 변하지 않는다
        assert abs(pick(c0, sc, tg).delta - pick(c1, sc, tg).delta) < 1e-12 and pick(c1, sc, tg).n_rows == 2, (sc, tg)
    assert pick(c1, "region", "Russia_C").n_rows == 1 and pick(c1, "region", "Russia_C").n_rows_expected == 2
    v0 = out0["tests"]; v1 = out1["tests"]
    l19 = [list(v_[(v_.test_id == "L19") & (v_.role == "주") & (v_.scope == "verdict")].verdict) for v_ in (v0, v1)]
    assert l19[0] == l19[1] and len(l19[0]) == 1, "확인적 가설 L19 의 판정은 주 5지역 밖의 실패에 영향을 받지 않는다"


# ================================================================ (f) 교란 불변
def _preds(a, Vd, u):
    H.TRACE = []
    try:
        cells, preds, rec = V.predict_unit(a, Vd, *u, V.make_fitter(a), check=False)
        return cells, preds, rec, list(H.TRACE)
    finally:
        H.TRACE = None


def test_f_heldout_labels_never_used(tmp_path, fake_fitter):
    a, Vd = make_env(tmp_path)
    units = [("V-R", 1, 0), ("V-P", 1, 1), ("V-S", 2, 2), ("V-B", 1, 3), ("V-C0", 1, 0), ("V-C100", 1, 1), ("V-C500", 1, 2), ("V-G", 1, 0),
             ("V-G", 1, 2), ("W2-Lena", 1, 0), ("W2-Alaska", 2, 3), ("W1-Canada", 1, 1)]
    valid = set(V.enumerate_units(a, Vd)[0])
    units = [u for u in units if u in valid]
    assert len(units) >= 9
    for u in units:
        cells, p1, rec1, tr1 = _preds(a, Vd, u)
        held = cells["held"]
        ids = set(int(v) for v in held)
        assert len(tr1) > 0
        for e in tr1:                                             # 학습 행렬에 채점 fold 의 셀이 없다(입력 0 열 = 행 식별자, D0m 에도 남는다)
            assert not ids & set(int(v) for v in e["Xtr"][:, 0]), f"{u} {e['method']}: 채점 fold 의 셀이 학습 행렬에 있다"
        y0 = Vd.y.copy(); df_y0 = Vd.df["y"].values.copy()
        try:
            Vd.y[held] = Vd.y[held] * 3.0 + 200.0                 # 채점 fold 의 라벨 교란
            Vd.df.loc[Vd.df.index[held], "y"] = Vd.y[held]
            Vd.D._src = {}
            _, p2, rec2, _ = _preds(a, Vd, u)
        finally:
            Vd.y[:] = y0; Vd.df["y"] = df_y0; Vd.D._src = {}
        assert list(p1) == list(p2)
        for k in p1:
            assert np.allclose(p1[k], p2[k], rtol=0, atol=1e-10, equal_nan=True), f"{u} {k}: 채점 fold 의 라벨을 바꾸자 예측이 달라졌다(누설)"
        if u[0].startswith("W2-"):
            assert rec1["nest"] == rec2["nest"] and rec1["nest"]["lam"] in a.LAMS and rec1["nest"]["flag"] == ""
            nest = [e for e in tr1 if e["method"] == "R1w@nest"]
            assert len(nest) == a.nest_folds and all(e["seed"] == 0 for e in nest)
            assert ("R1w@nest", "catboost_lo", 0, -1.0) in p1
            lam = rec1["nest"]["lam"]
            assert np.allclose(p1[("R1w@nest", "catboost_lo", 1, -1.0)], p1[("R1w", "catboost_lo", 1, lam)])
    # 대조: 학습 fold 의 라벨을 바꾸면 예측이 달라진다(시험이 교란을 감지함을 확인)
    u = ("W2-Lena", 1, 0)
    cells, p1, _, _ = _preds(a, Vd, u)
    y0 = Vd.y.copy()
    try:
        Vd.y[cells["trw"]] = Vd.y[cells["trw"]] * 3.0 + 200.0
        _, p2, _, _ = _preds(a, Vd, u)
    finally:
        Vd.y[:] = y0
    assert not np.allclose(p1[("P1w", "none", -1, 0.0)], p2[("P1w", "none", -1, 0.0)])
    assert np.allclose(p1[("P0", "none", -1, 0.0)], p2[("P0", "none", -1, 0.0)])               # 원천 계수는 지역 내 라벨과 무관하다


def test_f2_w2_formulas(tmp_path, fake_fitter):
    """W2 의 계수와 학습 행렬: 행 순서 [원천; 지역 내], 원천 행은 E0 앵커, 지역 내 행은 E_w 앵커."""
    a, Vd = make_env(tmp_path)
    u = ("W2-Canada", 1, 2)
    cells, p, rec, tr = _preds(a, Vd, u)
    src, trw, te = cells["src"], cells["trw"], cells["te"]
    E0 = float((Vd.s[src] @ Vd.y[src]) / (Vd.s[src] @ Vd.s[src])); E_ls = float((Vd.s[trw] @ Vd.y[trw]) / (Vd.s[trw] @ Vd.s[trw]))
    E_w = (len(trw) * E_ls + 10.0 * E0) / (len(trw) + 10.0)
    assert abs(rec["E0"] - E0) < 1e-12 and abs(rec["E_w"] - E_w) < 1e-12 and abs(rec["E_ls"] - E_ls) < 1e-12
    assert np.allclose(p[("P0", "none", -1, 0.0)], E0 * Vd.s[te]) and np.allclose(p[("P1w", "none", -1, 0.0)], E_w * Vd.s[te])
    assert np.allclose(p[("P2w", "none", -1, 0.0)], E_ls * Vd.s[te])
    ids = np.concatenate([src, trw]).astype(float)
    for m, want in (("D0", np.concatenate([Vd.y[src], Vd.y[trw]])),
                    ("R0", np.concatenate([Vd.y[src] - E0 * Vd.s[src], Vd.y[trw] - E0 * Vd.s[trw]])),
                    ("R1w", np.concatenate([Vd.y[src] - E0 * Vd.s[src], Vd.y[trw] - E_w * Vd.s[trw]])),
                    ("RMw", np.concatenate([np.log(Vd.y[src] / (E0 * Vd.s[src])), np.log(Vd.y[trw] / (E_w * Vd.s[trw]))])),
                    ("V2", np.log(np.concatenate([Vd.y[src], Vd.y[trw]]) / np.concatenate([Vd.s[src], Vd.s[trw]])))):
        es = [e for e in tr if e["method"] == m and e["learner"] == "catboost_lo" and e["seed"] == 0]
        assert len(es) == 1 and np.array_equal(es[0]["Xtr"][:, 0], ids.astype(np.float32)) and np.allclose(es[0]["ytr"], want), m
        assert es[0]["Xtr"].shape[1] == 25 and es[0]["Xtr"].dtype == np.float32 and es[0]["w"] is None
    f1k = [e for e in tr if e["method"] == "F1k" and e["seed"] == 0][0]
    assert f1k["Xtr"].shape[1] == 27 and np.allclose(f1k["Xtr"][:, 25], np.concatenate([Vd.ku[src], Vd.ku[trw]]), equal_nan=True)
    assert np.allclose(f1k["Xtr"][:, 26], np.concatenate([Vd.ed[src], Vd.ed[trw]]))
    assert {e["learner"] for e in tr if e["method"] in ("D0", "R1w")} == {"catboost_lo", "catboost", "rf"}
    # 예측식: R1w = E_w·s + λ·g, RMw = E_w·s·exp(λ·ĥ)(ĥ 는 학습 h 의 범위 안)
    g = (p[("R1w", "catboost_lo", 0, 1.0)] - E_w * Vd.s[te])
    assert np.allclose(p[("R1w", "catboost_lo", 0, 0.25)], E_w * Vd.s[te] + 0.25 * g)
    hh = np.log(p[("RMw", "catboost_lo", 0, 1.0)] / (E_w * Vd.s[te]))
    h_tr = [e for e in tr if e["method"] == "RMw" and e["seed"] == 0][0]["ytr"]
    assert hh.min() >= h_tr.min() - 1e-9 and hh.max() <= h_tr.max() + 1e-9
    assert np.allclose(p[("RMw", "catboost_lo", 0, 0.5)], E_w * Vd.s[te] * np.exp(0.5 * hh))


def test_f3_pooled_formulas(tmp_path, fake_fitter):
    """통합 자료 단의 예측식: PS, RS, V2, RM0, D0m, F1k. W1 은 D0m 과 RM0 을 내지 않는다."""
    a, Vd = make_env(tmp_path)
    cells, p, rec, tr = _preds(a, Vd, ("V-B", 1, 0))
    trn, te = cells["tr"], cells["te"]
    E = float((Vd.s[trn] @ Vd.y[trn]) / (Vd.s[trn] @ Vd.s[trn]))
    assert abs(rec["E"] - E) < 1e-12 and np.allclose(p[("PS", "none", -1, 0.0)], E * Vd.s[te])
    g = p[("RS", "rf", 1, 1.0)] - E * Vd.s[te]
    assert np.allclose(p[("RS", "rf", 1, 0.5)], E * Vd.s[te] + 0.5 * g)
    zh = np.log(p[("V2", "catboost_lo", 0, 0.0)] / Vd.s[te])
    z_tr = np.log(Vd.y[trn] / Vd.s[trn])
    assert zh.min() >= z_tr.min() - 1e-9 and zh.max() <= z_tr.max() + 1e-9
    assert np.allclose(p[("RM0", "catboost_lo", 0, 0.25)], E * Vd.s[te] * np.exp(0.25 * (zh - np.log(E))))
    assert ("RM0", "catboost_lo", 0, 1.0) not in p                                           # λ = 1.0 은 V2 와 같다
    d0m = [e for e in tr if e["method"] == "D0m" and e["seed"] == 0][0]
    assert d0m["Xtr"].shape[1] == 21 and np.array_equal(d0m["Xtr"], Vd.X[trn][:, V.KEEP_M])
    assert [H.FEATS[i] for i in range(25) if i not in V.KEEP_M] == ["e5_tdd", "e5_sqrt_tdd", "cci_alt", "cci_valid"]
    rs = [e for e in tr if e["method"] == "RS" and e["learner"] == "catboost" and e["seed"] == 1][0]
    assert np.allclose(rs["ytr"], Vd.y[trn] - E * Vd.s[trn]) and np.array_equal(rs["Xtr"][:, 0], trn.astype(np.float32))
    assert not any(k[0].startswith(("B:", "P0@")) for k in p)                                 # N1 기준선은 V-G 에만 있다
    _, pw, _, _ = _preds(a, Vd, ("W1-Lena", 1, 0))
    assert {k[0] for k in pw} == {"PSw", "D0w", "F1kw", "V2w", "RSw"}
    _, pg, recg, _ = _preds(a, Vd, ("V-G", 1, 1))
    assert {"B:ed_raw", "B:ku_raw", "B:cci_raw", "P0@soil", "P0@ku", "P0@ed", "P0@cci", "B:s_aff", "B:cci_aff", "B:ens", "B:ku_raw#ex",
            "P0@ku#ex"} <= {k[0] for k in pg}
    assert {k[0] for k in pg if k[1] != "none"} == {"D0", "D0m", "F1k", "V2", "RM0", "RS"}


# ================================================================ (g) 판정과 통계
def test_g_verdict4_boundaries():
    assert V.verdict4(-1.0, -0.1, -2.0, -0.2, 0.5) == "우세"
    assert V.verdict4(0.1, 1.0, 0.2, 2.0, 0.5) == "열세"
    assert V.verdict4(-1.0, -0.1, -2.0, 0.0, 0.5) == "미결정"                                  # 한 가중의 상한이 0 이면 우세가 아니다
    assert V.verdict4(-0.3, 0.0, -0.2, -0.1, 0.5) == "동등"                                    # 상한 0 은 우세가 아니고 한계 안이면 동등
    assert V.verdict4(-0.5, 0.3, -0.4, 0.5, 0.5) == "동등"                                     # 경계값 포함
    assert V.verdict4(-0.5000001, 0.3, -0.4, 0.5, 0.5) == "미결정"
    assert V.verdict4(-0.8, 0.3, -0.4, 0.5, 1.0) == "동등" and V.verdict4(-0.8, 0.3, -0.4, 0.5, 0.5) == "미결정"
    assert V.verdict4(0.0, 0.4, 0.1, 0.3, 0.5) == "동등"                                       # 하한 0 은 열세가 아니다
    assert V.verdict4(np.nan, 0.4, 0.1, 0.3, 0.5) == "판정 불가" and V.verdict4(None, 0.4, 0.1, 0.3, 0.5) == "판정 불가"
    d = np.linspace(-1, 1, 2001)
    assert abs(V.p_equiv(d, 0.5) - max(np.mean(d <= -0.5), np.mean(d >= 0.5))) < 1e-12 and 0.24 < V.p_equiv(d, 0.5) < 0.26
    assert V.p_equiv(d - 0.4, 0.5) > 0.4 and V.p_equiv(0.1 * d, 0.5) == 0.0
    assert np.isnan(V.p_equiv(None, 0.5)) and all(np.isnan(v) for v in V.pct_ci(None))


def test_g2_strat_mean_and_pool():
    rng = np.random.RandomState(0)
    a = V.parse_args(["--threads", "1"])
    per = {r: dict(delta=-1.0 - i, delta_beq=-0.5 - i, dist=-1.0 - i + 0.1 * rng.randn(500), dist_beq=-0.5 - i + 0.1 * rng.randn(500), rmse_A=10.0,
                   rmse_B=11.0 + i, n_cells=100, n_blocks=10, n_rows=2) for i, r in enumerate(["Alaska", "Lena", "Canada"])}
    per["Russia_W"] = dict(per["Alaska"], delta=5.0, delta_beq=5.0, dist=None, dist_beq=None)          # 블록 8개 미만: 점 추정만
    res = V.strat_combine(per, V.MAIN5)
    assert res["n_ci_regions"] == 3 and res["regions"] == "Alaska,Lena,Canada" and abs(res["delta"] + 2.0) < 1e-12        # CI 풀 지역의 평균
    assert np.allclose(res["dist"], np.mean([per[r]["dist"] for r in ("Alaska", "Lena", "Canada")], 0))
    row = V.finish_row(res, a, need_regions=5)
    assert row["verdict4"] == "우세" and row["pool"] == "지역 3/5" and row["n_ci_regions"] == 3 and row["ci_lo"] < row["delta"] < row["ci_hi"]
    assert row["pool_partial"] and row["n_rows"] == 2 and row["n_rows_expected"] == 2 and row["rows_short"] == ""
    used = [dict(row, scope="MEAN5", contrast="c")]
    assert V.mark_verdict("L21", "지지", used) == "부분(지역 3/5): 지지", "대비 행의 4분 판정은 그대로 두고 판정 문구에 풀 지역 수를 붙인다"
    assert V.mark_verdict("L19", "지지", used) == "부분(지역 3/5): 지지" and V.mark_verdict("L19", "판정 불가", used) == "판정 불가"
    assert abs(row["delta_pct_ref"] - 100.0 * res["delta"] / res["rmse_B"]) < 1e-9 and not row["small_effect"]
    one = V.strat_combine({"Alaska": per["Alaska"], "Russia_W": per["Russia_W"]}, V.MAIN5)
    assert one["n_ci_regions"] == 1 and V.finish_row(one, a, need_regions=5)["verdict4"] == "판정 불가"
    none = V.strat_combine({"Russia_W": per["Russia_W"]}, V.MAIN5)
    assert none["dist"] is None and none["delta"] == 5.0 and V.finish_row(none, a, need_regions=5)["verdict4"] == "판정 불가"
    assert V.strat_combine({}, V.MAIN5) is None
    off = dict(res, delta=10.0)                                                                # 점 추정치가 백분위 CI 밖: 판정 불가
    r2 = V.finish_row(off, a, need_regions=5)
    assert r2["verdict4"] == "판정 불가" and "assert" in r2["ci_flag"]
    from polar.m1_stats import holm
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])


def test_g4_mark_verdict_and_p_equiv():
    full = dict(scope="MEAN5", contrast="c", verdict4="우세", delta=-2.0, n_ci_regions=5, n_regions_target=5, n_rows=6, n_rows_expected=6)
    assert V.mark_verdict("L19", "지지", [full]) == "지지" and V.mark_verdict("L21", "지지", [full, None]) == "지지"
    short = dict(full, n_rows=4, n_rows_expected=6)
    assert V.mark_verdict("L19", "지지", [short]).startswith("판정 불가(반복·seed 행 4/6"), "확인적 가설은 행이 기대보다 적으면 판정하지 않는다"
    assert V.mark_verdict("L21", "지지", [short]) == "부분(반복·seed 행 4/6): 지지"
    assert V.mark_verdict("L21", "지지", [dict(short, n_ci_regions=3)]) == "부분(지역 3/5, 반복·seed 행 4/6): 지지"
    small = dict(full, delta=-0.3)
    assert V.mark_verdict("L21", "지지", [small]) == f"지지 [{V.SMALL_EFFECT_TXT}: c]"
    assert V.mark_verdict("L21", "지지", [dict(small, verdict4="미결정")]) == "지지"
    assert V.mark_verdict("L21", "지지", [dict(short, verdict4="판정 불가")]) == "지지", "판정 불가 행은 표기에 쓰지 않는다"
    region = dict(scope="region", contrast="r", verdict4="열세", delta=0.2, n_rows=10, n_rows_expected=10)
    assert V.mark_verdict("L22", "우세 지역 1/3: Lena", [region]) == f"우세 지역 1/3: Lena [{V.SMALL_EFFECT_TXT}: r]"
    assert V.valid_row(full) and not V.valid_row(None) and not V.valid_row(dict(verdict4="판정 불가"))
    d = np.linspace(-1, 1, 2001)
    assert V.p_equiv2(d, 0.1 * d, 0.5) == V.p_equiv(d, 0.5) and V.p_equiv2(0.1 * d, d, 0.5) == V.p_equiv(d, 0.5), "두 가중 가운데 큰 값"
    assert V.p_equiv2(d, None, 0.5) == V.p_equiv(d, 0.5) and np.isnan(V.p_equiv2(None, None, 0.5))
    rng = np.random.RandomState(0)
    a = V.parse_args(["--threads", "1"])
    res = dict(delta=0.0, delta_beq=0.0, dist=0.05 * rng.randn(500), dist_beq=0.6 + 0.05 * rng.randn(500), rmse_A=10.0, rmse_B=10.0,
               n_cells=100, n_blocks=10, n_rows=2)
    row = V.finish_row(res, a)
    assert row["p_equiv"] > 0.9 and row["verdict4"] != "동등", "블록 등가중 분포가 한계 밖이면 동등성 p 가 크다"


def test_g3_bootstrap_matches_h4_common():
    """boot_group 의 식은 h4_common.boot_delta_blocks(분할 하나)와 같다."""
    rng = np.random.RandomState(1)
    nb, nboot = 12, 400
    blocks = np.repeat(np.arange(nb), 7)
    y = rng.randn(len(blocks)) * 5 + 50
    st = H.BlockStore("T", 1, blocks)
    kA = [("A", d, s) for d in range(3) for s in range(2)]; kB = [("B", 0, -1)]
    for k in kA:
        st.add(k, y, y + rng.randn(len(y)) * 2.0)
    st.add(kB[0], y, y + rng.randn(len(y)) * 3.0)
    seed = 77
    ref = H4.boot_delta_blocks({1: st}, kA, kB, nboot=nboot, seed=seed, return_dist=True)
    W = H4.boot_weights(nb, nboot, H4.seed_of(seed, 1))
    SA, CA = st.matrices(kA); SB, CB = st.matrices(kB)
    ca, ba = V.boot_group(SA, CA.astype(float), W); cb, bb = V.boot_group(SB, CB.astype(float), W)
    assert np.allclose(ca - cb, ref["dist"]) and np.allclose(ba - bb, ref["dist_beq"])
    lo, hi = V.pct_ci(ca - cb)
    assert abs(lo - ref["ci_lo"]) < 1e-12 and abs(hi - ref["ci_hi"]) < 1e-12


# ================================================================ (h) 분할 함수
def test_h_fold_functions():
    f = V.folds_random(103, 5, 11)
    assert np.array_equal(f, V.folds_random(103, 5, 11)) and not np.array_equal(f, V.folds_random(103, 5, 12))
    assert sorted(np.bincount(f).tolist()) == [20, 20, 21, 21, 21]
    perm = np.random.RandomState(11).permutation(103)
    assert np.array_equal(f[perm], np.arange(103) % 5)                                         # 순열에서의 위치 mod 5
    g = np.repeat(np.arange(10), 3)                                                            # 크기가 같은 묶음 10개
    fg = V.folds_grouped(g, 5, 3)
    order = np.random.RandomState(3).permutation(10)
    assert [int(fg[g == j][0]) for j in order] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]               # 동률이면 번호가 작은 fold
    g2 = np.concatenate([np.repeat(0, 50), np.repeat(np.arange(1, 21), 5)])                    # 큰 묶음 하나와 작은 묶음 20개
    f2 = V.folds_grouped(g2, 5, 0)
    assert pd.DataFrame(dict(g=g2, f=f2)).groupby("g").f.nunique().eq(1).all()
    assert np.bincount(f2, minlength=5).max() <= 70 and np.bincount(f2, minlength=5).min() >= 20      # 큰 묶음은 가장 적은 fold 에 들어간다
    s = V.groups_s(np.array([70.01, 70.04, 70.06, -0.01]), np.array([-150.01, -150.04, -150.01, 0.01]))
    assert s[0] == s[1] and s[0] != s[2] and len(set(s.tolist())) == 3
    assert V.scheme_kind("V-C250") == "cluster" and V.cluster_buffer("V-C250") == 250.0 and V.scheme_kind("W2-Lena") == "within"
    with pytest.raises(SystemExit):
        V.scheme_kind("V-X")


def test_h2_cluster_map_roundtrip(tmp_path):
    a, Vd = make_env(tmp_path)
    tab = pd.read_csv(a.CMAP)
    assert list(tab.columns) == ["block", "cluster", "n_cells", "lat", "lon"] and tab.cluster.nunique() == N_CLUSTERS and tab.block.is_unique
    assert sorted(tab.cluster.unique().tolist()) == list(range(N_CLUSTERS))
    assert int(tab.n_cells.sum()) == Vd.N and set(tab.block) == set(Vd.block.tolist())
    lat_c = tab.groupby("cluster").lat.mean()
    assert lat_c.is_monotonic_decreasing                                                       # 군집 번호는 평균 위도의 내림차순
    m0 = a.CMAP.stat().st_mtime_ns
    V.write_cluster_map(a, Vd)                                                                 # 같은 표: 다시 쓰지 않는다
    assert a.CMAP.stat().st_mtime_ns == m0
    bad = tab.copy(); bad.loc[0, "cluster"] = (bad.cluster.iloc[0] + 1) % N_CLUSTERS
    bad.to_csv(a.CMAP, index=False)
    with pytest.raises(SystemExit):                                                            # 다른 표는 덮어쓰지 않는다
        V.write_cluster_map(a, Vd)
    a.CMAP.unlink()
    Vn = V.VData(a, D=Vd.D)
    with pytest.raises(SystemExit):                                                            # 실행은 표만 읽는다
        Vn.cluster_of(a)
    a.count_only = True
    assert len(np.unique(Vn.cluster_of(a))) == N_CLUSTERS and "임시" in Vn.cluster_src and not a.CMAP.exists()


# ================================================================ (i)–(l) 기본값, 재개, 거리, 적합 수
def test_i_defaults_and_refusal(monkeypatch, tmp_path):
    monkeypatch.setenv("LG_RESCALE", "1")
    a = V.parse_args([])
    assert a.workers <= 2 and a.threads == 4 and a.tag == "lgv" and a.TAG == "lgv" and a.out_dir == "data/processed/lgx/ladder"
    assert a.TDDM.name == "lgx_tdd_matched_v1.csv" and V.ANCHOR_KINDS == ["soil", "ku", "ed", "cci", "tddm"]
    assert a.SCHEMES == ["V-R", "V-P", "V-S", "V-B", "V-C0", "V-C100", "V-C500", "V-G"] and a.WREGIONS == ["Alaska", "Lena", "Canada"]
    assert a.REPS == [1, 2, 3] and a.WREPS == [1, 2, 3, 4, 5] and a.folds == 5 and a.SEEDS == [0, 1] and a.LAMS == [0.25, 0.5, 1.0]
    assert a.LEARNERS == ["catboost_lo", "catboost", "rf"] and a.kappa == 10.0 and a.buffer_km == 100.0 and a.n_clusters == 30
    assert a.nboot == 10000 and a.delta_eq == 0.5 and a.delta_eq_aux == 1.0 and a.nest_folds == 4 and not a.allow_partial
    assert a.HA.threads == 4 and a.HA.cb_iters == 200 and a.HA.kappa == 10.0 and a.HA.buffer_km == 100.0
    assert V.schemes_to_run(V.parse_args(["--part", "pooled"])) == a.SCHEMES
    assert V.schemes_to_run(V.parse_args(["--part", "within"])) == [f"W2-{r}" for r in a.WREGIONS] + [f"W1-{r}" for r in a.WREGIONS]
    s = V.parse_args(["--smoke"])
    assert s.TAG == "lgv_smoke" and s.REPS == [1] and s.WREPS == [1] and s.SEEDS == [0] and s.SCHEMES == ["V-R", "V-B", "V-C100", "V-G"]
    assert s.WREGIONS == ["Lena"] and s.SMOKE_CLUSTERS == 3 and s.allow_partial and s.LEARNERS == a.LEARNERS
    with pytest.raises(SystemExit):
        V.parse_args(["--learners", "mlp"])
    with pytest.raises(SystemExit):
        V.parse_args(["--schemes", "W2-Lena"])
    monkeypatch.delenv("LG_RESCALE", raising=False)
    assert not V.run_permitted(a) and V.run_permitted(V.parse_args(["--allow-local"]))
    loc = V.parse_args(["--threads", "8", "--summarize-only"])                                 # 허용 표지가 없으면 스레드 1, 집계의 재표집은 1,000회 이하
    assert loc.threads == 1 and loc.HA.threads == 1 and V.parse_args(["--allow-local", "--threads", "2"]).threads == 2
    V.limit_local(loc)
    assert loc.nboot == V.LOCAL_NBOOT_MAX and loc.nboot_asked == 10000
    ok = V.limit_local(V.parse_args(["--allow-local", "--summarize-only"]))
    assert ok.nboot == 10000 and ok.threads == 4 and not hasattr(ok, "nboot_asked")
    for extra in ([], ["--smoke"]):                                                            # 스모크도 허용 표지 없이는 거부된다(자료를 읽기 전)
        with pytest.raises(SystemExit) as e:
            V.main(["--part", "pooled", "--schemes", "V-R", "--workers", "0", "--threads", "1", "--no-summarize", "--out-dir", str(tmp_path)] + extra)
        assert "거부" in str(e.value)
    assert not (tmp_path / "shards").exists()
    monkeypatch.setenv("LG_RESCALE", "1")
    assert V.run_permitted(a)
    order = sorted([("W1-Lena", 1, 0), ("V-S", 1, 0), ("V-C500", 1, 3), ("W2-Lena", 2, 1), ("V-R", 2, 0), ("V-G", 1, 4), ("V-R", 1, 3)],
                   key=V.unit_priority)
    assert [u[0] for u in order] == ["V-G", "V-R", "V-R", "W2-Lena", "V-C500", "V-S", "W1-Lena"] and order[1] == ("V-R", 1, 3)
    assert [V.g_(m) for m in ("PS", "P1w", "B:ens", "P0@ku#ex")] == [(m, "none", 0.0) for m in ("PS", "P1w", "B:ens", "P0@ku#ex")]
    assert V.g_("D0", "rf") == ("D0", "rf", 1.0) and V.g_("F1kw") == ("F1kw", "catboost_lo", 1.0) and V.g_("V2") == ("V2", "catboost_lo", 0.0)
    assert V.g_("RS") == ("RS", "catboost_lo", 0.25) and V.g_("R1w", "rf", 1.0) == ("R1w", "rf", 1.0)
    assert V.g_("R1w@nest") == ("R1w@nest", "catboost_lo", -1.0) and V.g_("RM0", lam=0.5) == ("RM0", "catboost_lo", 0.5)


def test_j_resume_checks_cfg_hash(tmp_path):
    base = ["--out-dir", str(tmp_path), "--tag", "t", "--cluster-map", str(tmp_path / "cmap.csv")]
    a = V.parse_args(base)
    u = ("V-B", 2, 3)
    p = V.shard_paths(a, *u)
    assert p["unit"].name == "t__V-B__r2__f3_unit.json" and p["pred"].name == "t__V-B__r2__f3_pred.npz"
    assert V.shard_paths(a, "W2-Lena", 1, 0)["pred"].name == "t__W2-Lena__r1__f0_pred.npz"
    assert V.unit_state(a, *u) == (False, "조각 없음")
    p["unit"].parent.mkdir(parents=True, exist_ok=True)
    p["pred"].write_bytes(b"0")
    cfg = V.unit_cfg(a, "V-B")
    p["unit"].write_text(json.dumps(dict(status="ok", cfg_hash=V.cfg_hash(cfg))))
    assert V.unit_state(a, *u) == (True, "ok")
    assert V.unit_state(V.parse_args(base + ["--workers", "14", "--threads", "2", "--reps", "1", "--nboot", "100"]), *u)[0]    # 실행 자원·집계 인자는 해시에 없다
    for extra in (["--kappa", "5"], ["--seeds", "1"], ["--learners", "catboost_lo"], ["--lams", "0.25,1.0"], ["--folds", "4"], ["--buffer-km", "50"]):
        ok, why = V.unit_state(V.parse_args(base + extra), *u)
        assert not ok and "설정" in why, extra
    p["unit"].write_text(json.dumps(dict(status="failed", cfg_hash=V.cfg_hash(cfg))))
    assert not V.unit_state(a, *u)[0]
    p["unit"].write_text(json.dumps(dict(status="ok")))
    assert not V.unit_state(a, *u)[0]
    found = V.find_shards(a)
    assert len(found) == 1 and (found[0]["scheme"], found[0]["rep"], found[0]["fold"]) == u
    assert V.cfg_hash(cfg, common=True) == V.cfg_hash(V.unit_cfg(a, "W2-Lena"), common=True) == V.cfg_hash(V.unit_cfg(a, "V-C100"), common=True)
    c1 = V.cfg_hash(V.unit_cfg(a, "V-C100"))
    (tmp_path / "cmap.csv").write_text("block,cluster,n_cells,lat,lon\n1,0,1,0,0\n")
    assert V.cfg_hash(V.unit_cfg(a, "V-C100")) != c1 and V.cfg_hash(V.unit_cfg(a, "V-B")) == V.cfg_hash(cfg)      # 군집 표가 바뀌면 V-C 조각만 다시 실행
    with pytest.raises(SystemExit):
        V.check_shard_cfg(a, [dict(cfg_common="a"), dict(cfg_common="b")])
    a.allow_mixed_cfg = True
    assert len(V.check_shard_cfg(a, [dict(cfg_common="a"), dict(cfg_common="b")])["cfg_common"]) == 2


def test_k_distance_functions():
    rng = np.random.RandomState(3)
    lq, oq = rng.uniform(55, 80, 300), rng.uniform(-180, 180, 300)
    lr, orr = rng.uniform(55, 80, 200), rng.uniform(-180, 180, 200)
    brute = np.array([haversine_km(lq[i], oq[i], lr, orr).min() for i in range(len(lq))])
    assert np.allclose(V.min_dist_km(lq, oq, lr, orr), brute, rtol=1e-9, atol=1e-6)
    for d in (50.0, 100.0, 500.0):
        assert V.count_within_km(lq, oq, lr, orr, d) == int((brute < d).sum())
    assert V.count_within_km(lq, oq, lr, orr, 0.0) == 0 and V.count_within_km(lq[:0], oq[:0], lr, orr, 100.0) == 0
    assert len(V.min_dist_km(lq[:0], oq[:0], lr, orr)) == 0 and np.all(np.isinf(V.min_dist_km(lq, oq, lr[:0], orr[:0])))
    # 날짜 변경선: 경도 179.9 와 −179.9 는 가깝다
    d = V.min_dist_km(np.array([70.0]), np.array([179.9]), np.array([70.0]), np.array([-179.9]))[0]
    assert abs(d - haversine_km(70.0, 179.9, 70.0, -179.9)) < 1e-6 and d < 10.0
    # 경계: 거리가 정확히 d 인 셀은 남긴다(거리 < d 만 뺀다)
    d0 = float(haversine_km(np.array([60.0]), np.array([10.0]), np.array([61.0]), np.array([10.0]))[0])
    assert V.min_dist_km(np.array([60.0]), np.array([10.0]), np.array([61.0]), np.array([10.0]), d_ref=d0)[0] == d0
    assert V.count_within_km(np.array([60.0]), np.array([10.0]), np.array([61.0]), np.array([10.0]), d0) == 0
    assert V.count_within_km(np.array([60.0]), np.array([10.0]), np.array([61.0]), np.array([10.0]), d0 + 1e-6) == 1


def test_l_dry_fit_counts(tmp_path):
    a, Vd = make_env(tmp_path)
    assert V.FITTER is None
    r = V.run_unit(a, Vd, "V-R", 1, 0, dry=True)
    assert r["fit_pooled"] == 18 and r["n_keys"] == 1 + 6 + 2 + 2 + 2 + 4 + 18                  # PS, D0, D0m, F1k, V2, RM0, RS
    g = V.run_unit(a, Vd, "V-G", 1, 0, dry=True)
    assert g["fit_pooled"] == 18 and g["n_keys"] == r["n_keys"] + 12                            # N1 기준선 10 + '#ex' 2
    w2 = V.run_unit(a, Vd, "W2-Lena", 1, 0, dry=True)
    assert w2["fit_w2"] == 20 and w2["fit_w2_nest"] == 4 and w2["n_keys"] == 3 + 6 + 2 + 2 + 6 + 18 + 6 + 2
    w1 = V.run_unit(a, Vd, "W1-Lena", 1, 0, dry=True)
    assert w1["fit_w1"] == 16 and w1["n_keys"] == 1 + 6 + 2 + 2 + 18
    assert not a.SHARDS.exists() or not list(a.SHARDS.glob("*"))                                # dry 는 조각을 쓰지 않는다
    b = V.parse_args(BASE_ARGS + ["--out-dir", str(tmp_path / "o2"), "--cluster-map", str(a.CMAP), "--learners", "catboost_lo", "--seeds", "1"])
    r1 = V.run_unit(b, V.VData(b, D=Vd.D), "V-R", 1, 0, dry=True)
    assert r1["fit_pooled"] == 5 and r1["n_keys"] == 1 + 1 + 1 + 1 + 1 + 2 + 3
    s = V.parse_args(BASE_ARGS + ["--out-dir", str(tmp_path / "o3"), "--cluster-map", str(a.CMAP), "--smoke"])
    us, sk = V.enumerate_units(s, V.VData(s, D=Vd.D))
    assert sum(1 for u in us if u[0] == "V-C100") == 3 and any(v["status"] == "smoke_limit" for v in sk)
    assert {u[0] for u in us} == {"V-R", "V-B", "V-C100", "V-G", "W2-Lena", "W1-Lena"} and {u[1] for u in us} == {1}


# ================================================================ (m) N1 기준선
def test_m_baselines_use_training_cells_only():
    rng = np.random.RandomState(5)

    def cells(n, nan_frac):
        s = rng.uniform(20, 40, n); y = 1.4 * s + rng.randn(n)
        ku = 1.2 * s + rng.randn(n); ku[rng.rand(n) < nan_frac] = np.nan
        return V.Cells(rng.randn(n, 25), y, s, ku, 1.1 * s + rng.randn(n), 0.8 * y + rng.randn(n), 0.9 * s)
    TR, TE = cells(300, 0.2), cells(80, 0.3)
    TE.ku[0] = -1.0                                                                            # 0 이하 값도 대체 대상이다
    E = H.ls_E(TR.y, TR.s)
    out, co = V.baselines_n1(TR, TE, E)
    ok = np.isfinite(TR.ku) & (TR.ku > 0)
    rho = float(np.median(TR.ku[ok] / TR.s[ok]))
    assert abs(co["ku"]["rho"] - rho) < 1e-12                                                  # ρ 는 학습 셀에서만 구한다
    ftr = np.where(ok, TR.ku, rho * TR.s); c0 = float((ftr @ TR.y) / (ftr @ ftr))
    bad = ~(np.isfinite(TE.ku) & (TE.ku > 0))
    fte = np.where(bad, rho * TE.s, TE.ku)
    assert bad.sum() > 5 and abs(co["ku"]["c0"] - c0) < 1e-12 and co["ku"]["n_fill_test"] == int(bad.sum())
    assert np.allclose(out[("P0@ku", "none", -1, 0.0)], c0 * fte) and np.allclose(out[("B:ku_raw", "none", -1, 0.0)], fte)
    ex = out[("P0@ku#ex", "none", -1, 0.0)]
    assert np.array_equal(np.isnan(ex), bad) and np.allclose(ex[~bad], c0 * TE.ku[~bad])
    assert np.array_equal(np.isnan(out[("B:ku_raw#ex", "none", -1, 0.0)]), bad)
    b, a0 = np.polyfit(TR.s, TR.y, 1)
    assert np.allclose(out[("B:s_aff", "none", -1, 0.0)], a0 + b * TE.s)
    b, a0 = np.polyfit(TR.cci, TR.y, 1)
    assert np.allclose(out[("B:cci_aff", "none", -1, 0.0)], a0 + b * TE.cci)
    assert np.allclose(out[("B:ens", "none", -1, 0.0)], (E * TE.s + out[("P0@ku", "none", -1, 0.0)] + out[("P0@cci", "none", -1, 0.0)]) / 3.0)
    assert ("B:soil_raw", "none", -1, 0.0) not in out and all(np.all(np.isfinite(v)) for k, v in out.items() if not k[0].endswith("#ex"))
    assert ("P0@tddm", "none", -1, 0.0) not in out and "skipped" in co["tddm"], "연도 정합 도일이 없으면 그 앵커를 건너뛴다"
    TR.tddm = 0.95 * TR.s; TE.tddm = 0.95 * TE.s; TE.tddm[3] = np.nan                         # 연도 정합 √TDD 앵커(h42 의 X9 와 같은 5종)
    out_t, co_t = V.baselines_n1(TR, TE, E)
    c0t = float(((0.95 * TR.s) @ TR.y) / ((0.95 * TR.s) @ (0.95 * TR.s)))
    want = c0t * np.where(np.isfinite(TE.tddm), TE.tddm, 0.95 * TE.s)
    assert abs(co_t["tddm"]["c0"] - c0t) < 1e-12 and abs(co_t["tddm"]["rho"] - 0.95) < 1e-12 and co_t["tddm"]["n_fill_test"] == 1
    assert np.allclose(out_t[("P0@tddm", "none", -1, 0.0)], want) and ("B:tddm_raw", "none", -1, 0.0) not in out_t
    assert set(out_t) == set(out) | {("P0@tddm", "none", -1, 0.0)}
    TR.tddm[:] = np.nan; TE.tddm[:] = np.nan
    y2 = TE.y.copy(); TE.y[:] = TE.y * 5 + 100                                                 # 채점 셀의 라벨은 쓰지 않는다
    out2, _ = V.baselines_n1(TR, TE, E)
    TE.y[:] = y2
    assert all(np.allclose(out[k], out2[k], equal_nan=True) for k in out)
    assert V.shrink(np.nan, 5, 1.5, 10.0) == 1.5 and V.shrink(2.0, 0, 1.5, 10.0) == 1.5 and abs(V.shrink(2.0, 10, 1.0, 10.0) - 1.5) < 1e-12


# ================================================================ 실자료·학습 시험
def _real_args(tmp_path, extra=()):
    return V.parse_args(["--threads", "2", "--out-dir", str(tmp_path / "out"), "--nboot", "200"] + list(extra))


@HEAVY
def test_e_w2_matches_h40_run_ctx(tmp_path):
    if not (ROOT / "data" / "processed" / "fidelity_base_v3.csv").exists():
        pytest.skip("실자료 없음")
    a = _real_args(tmp_path, ["--seeds", "1", "--learners", "catboost_lo"])
    Vd = V.get_vdata(a)
    HA = H.parse_args(["--part", "cpu", "--targets", "Lena", "--splits", "5", "--n-grid", "all", "--draws", "1", "--seeds", "1", "--methods", "P0,P1,R1",
                       "--learners", "catboost_lo", "--alphas", "1", "--threads", "2"])
    D = H.get_data(HA)
    c = H.build_ctx(D, HA, "Lena", "x", 1)
    rows, st, stats = H.run_ctx(c, "cpu", HA, learner_axis=False, nested=False)
    assert stats["status"] == "ok"
    t_idx, _, src, _ = D.source_idx("Lena", "x")
    from polar.m1_core import half_split_blocks, eval_mask
    A_idx, B_idx = half_split_blocks(D.df, t_idx, 1)
    evB = B_idx[eval_mask(D.df.iloc[B_idx])]
    src = src[Vd.trainable[src]]
    assert np.array_equal(D.df.y.values[src], c.y_src) and np.array_equal(D.df.y.values[A_idx], c.yA) and np.array_equal(D.df.y.values[evB], c.yB)
    F = V.FitterV(a.HA, False)
    p, rec = V.models_w2(F, a, Vd.take(src), Vd.take(A_idx), Vd.take(evB), Vd.block[A_idx], ("lgv", "nest", "Lena", 1, 0))
    assert abs(rec["E0"] - c.E0) < 1e-12
    ref = H.BlockStore("Lena|x", 1, c.blkB)
    for k41, k40, tol in ((("P0", "none", -1, 0.0), ("P0", "none", "1", "cell", 0, 0, -1, 0.0), 1e-9),
                          (("P1w", "none", -1, 0.0), ("P1", "none", "1", "cell", -1, 0, -1, 0.0), 1e-9),
                          (("R1w", "catboost_lo", 0, 0.25), ("R1", "catboost_lo", "1", "cell", -1, 0, 0, 0.25), 1e-6),
                          (("R1w", "catboost_lo", 0, 1.0), ("R1", "catboost_lo", "1", "cell", -1, 0, 0, 1.0), 1e-6)):
        ref.add(k41, c.yB, p[k41])
        s41, c41 = ref.get(k41); s40, c40 = st.get(k40)
        assert np.array_equal(c41, c40)
        r41, r40 = float(np.sqrt(s41.sum() / c41.sum())), float(np.sqrt(s40.sum() / c40.sum()))
        assert abs(r41 - r40) < tol, f"{k41}: RMSE {r41} 대 {r40}"
        assert np.allclose(s41, s40, rtol=tol, atol=tol)


@HEAVY
def test_n_real_data_structure(tmp_path):
    if not (ROOT / "data" / "processed" / "fidelity_base_v3.csv").exists():
        pytest.skip("실자료 없음")
    a = _real_args(tmp_path)
    if not a.CMAP.exists():
        a = _real_args(tmp_path, ["--schemes", "V-R,V-P,V-S,V-B,V-G"])
    Vd = V.get_vdata(a)
    assert Vd.N == 17467 and int(Vd.score.sum()) == 17374 and len(np.unique(Vd.gid[Vd.score])) == 174
    assert len(np.unique(Vd.groups("V-P"))) == 736 and len(np.unique(Vd.groups("V-S"))) == 437 and len(np.unique(Vd.groups("V-B"))) == 185
    assert Vd.trainable.all() and pd.Series(Vd.loc_id).is_unique
    units, skipped = V.enumerate_units(a, Vd)
    cnt = {}
    for s, rep, k in units:
        c = V.unit_cells(a, Vd, s, rep, k, check=True)                                          # 교집합과 버퍼 단언을 실자료에서 실행한다
        q = cnt.setdefault((s, rep), np.zeros(Vd.N, int)); np.add.at(q, c["te"], 1)
        if s == "V-G":
            assert np.array_equal(c["tr"], Vd.D.source_idx(V.REG7[k], "x")[2])
    for (s, rep), q in cnt.items():
        want = Vd.score_of(s)
        assert np.all(q[want] == 1) and np.all(q[~want] == 0), f"{s} r{rep}"


@HEAVY
def test_o_real_fitters_smoke():
    """catboost(고용량)와 rf 의 적합 경로. 결측은 rf 에서 학습 행렬 중앙값으로 바뀐다."""
    rng = np.random.RandomState(0)
    X = rng.randn(400, 25).astype(np.float32); X[::7, 3] = np.nan
    y = 3 * X[:, 0] + rng.randn(400)
    Xt = rng.randn(50, 25).astype(np.float32); Xt[::5, 3] = np.nan
    a = V.parse_args(["--threads", "2"])
    F = V.FitterV(a.HA, False)
    for lr in ("catboost_lo", "catboost", "rf"):
        _, (p,) = F.fit(lr, "t", lambda: (X, y, None), 0, [Xt], info=dict(method="D0", n_train=len(y)))
        _, (q,) = F.fit(lr, "t", lambda: (X, y, None), 0, [Xt], info=dict(method="D0", n_train=len(y)))
        assert np.all(np.isfinite(p)) and np.allclose(p, q) and np.corrcoef(p, 3 * Xt[:, 0])[0, 1] > 0.7, lr
    assert F.nd["t|rf|D0"] == 2 and F.rowsd["t|catboost|D0"] == 800 and not F.errors


@HEAVY
def test_p_pool_path_workers2(tmp_path):
    """프로세스 풀 경로(spawn 워커 2개). 워커가 이 모듈을 다시 읽고 V-G 단위를 실행한다. 연도 정합 도일 표는 시험용으로 만든다."""
    src = ROOT / "data" / "processed" / "fidelity_base_v3.csv"
    if not src.exists():
        pytest.skip("실자료 없음")
    tp = tmp_path / "tdd_test.csv"
    pd.read_csv(src, usecols=["loc_id", "e5_tdd"]).rename(columns={"e5_tdd": "tdd_matched"}).drop_duplicates("loc_id").to_csv(tp, index=False)
    argv = ["--part", "pooled", "--schemes", "V-G", "--seeds", "1", "--learners", "catboost_lo", "--workers", "2", "--threads", "2", "--allow-local",
            "--out-dir", str(tmp_path / "out"), "--tdd-matched", str(tp), "--nboot", "200"]
    r = V.main(argv)
    assert len(r["executed"]) == 7 and not r["n_fail"] and not r["n_excluded"]
    a = V.parse_args(argv)
    assert not list(a.SHARDS.glob("*.tmp*")), "임시 파일이 남았다"
    met = pd.read_csv(a.OUT / f"{a.TAG}_metrics.csv")
    assert {"P0@tddm", "P0@soil", "P0@ku", "P0@ed", "P0@cci", "B:ens", "D0", "PS"} <= set(met[met.scheme == "V-G"].method)
    assert set(met.nboot) == {200}
