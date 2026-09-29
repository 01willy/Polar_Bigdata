"""scripts/3_deep_learning/h40_label_grid.py 단위 시험.

학습을 하는 시험(a–e, 표지 HEAVY)은 환경 변수 LG_RUN_HEAVY=1 일 때만 실행한다. 이 서버는 공유 서버이므로 Rescale 사전 점검 작업에서 실행한다.
(a) 같은 (n, 추출 번호)에서 모든 방법·학습기·부분이 같은 sel 을 쓰고, 그 sel 이 seed_of(대상, 모드, 분할, n, 추출 번호) 추출과 같다.
(b) n = 0 에서 R0 의 학습 행렬은 원천 잔차(y − E0·s)뿐이다. MLP α 축 R0 의 n = 0 기준 행이 저장된다.
(c) 대상 비선택 셀의 라벨은 학습 행렬·계수·α 선택 어디에도 들어가지 않는다(행 추적 + 비선택 라벨 교란 불변).
    α 중첩 선택의 후보는 {10, 100, cont} 뿐이다.
(d) --resume 은 조각이 있는 작업 단위를 건너뛴다.
(e) mlp_fit 은 tab_models.fit_predict('mlp') 와 같은 예측을 낸다(같은 절차).
학습을 하지 않는 시험(f–k)은 항상 실행한다.
(f) 기본값은 GPU 를 쓰지 않고 워커 2개 이하이며, 허용 표지 없는 본 실행은 거부된다.
(g) --resume 은 설정 해시가 다른 조각, 실패한 조각, 해시가 없는 조각을 다시 실행 대상으로 둔다. gpu 조각 이름에는 학습기가 붙는다.
(h) 실행 순서: 빠른 학습기 → 주 4지역 → 분할 번호, 무효 분할은 맨 뒤. 방법별 기준 λ.
(i) 층화 평균: CI 풀 지역 수 기록, 풀 지역 2개 미만과 단언 실패는 판정 불가, 단언 실패 때 지역 행 보존.
(j) 비유한 예측이 있는 키는 저장하지 않고 runs 에 기록한다.
(k) 적합 수 세기(dry): 학습기 단위 실행과 R2 제외 목록.
실행(Rescale): LG_RUN_HEAVY=1 CUDA_VISIBLE_DEVICES="" python3 -m pytest -q tests/test_h40_smoke.py
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
if os.environ.get("LG_RUN_HEAVY", "") == "1":
    import torch
    torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("h40_label_grid", ROOT / "scripts" / "3_deep_learning" / "h40_label_grid.py")
H = importlib.util.module_from_spec(_spec)
sys.modules["h40_label_grid"] = H
_spec.loader.exec_module(H)

BASE_ARGS = ["--n-grid", "0,3,10,all", "--draws", "2", "--seeds", "1", "--cb-iters", "15", "--cont-trees", "5", "--threads", "1", "--splits", "1",
             "--r", "1.0", "--learner-n-grid", "0,3,10,all", "--place-n-grid", "10", "--epochs", "1"]
D_FEAT = 6


def make_ctx(y_override=None, seed=0):
    """합성 작업 단위. 특징 0 열은 행 식별자(원천 = −1−j, 대상 A = j, 채점 B = 1000 + j)."""
    rng = np.random.RandomState(seed)
    ns, nA, nB = 400, 60, 50
    Xs = rng.randn(ns, D_FEAT); Xs[:, 0] = -1.0 - np.arange(ns)
    ss = rng.uniform(20, 40, ns); ys = 1.6 * ss + 3 * Xs[:, 1] + rng.randn(ns) * 2
    XA = rng.randn(nA, D_FEAT); XA[:, 0] = np.arange(nA)
    sA = rng.uniform(20, 40, nA); yA = 1.3 * sA + 3 * XA[:, 1] + rng.randn(nA) * 2
    XB = rng.randn(nB, D_FEAT); XB[:, 0] = 1000.0 + np.arange(nB)
    sB = rng.uniform(20, 40, nB); yB = 1.3 * sB + 3 * XB[:, 1] + rng.randn(nB) * 2
    XA[5, 3] = np.nan                                           # 결측 대체 경로
    if y_override is not None:
        yA = y_override(yA.copy())
    return H.Ctx("T", "x", 1, "T", Xs, ys, ss, np.repeat(["r1", "r2"], ns // 2), XA, yA, sA, np.repeat(np.arange(6), nA // 6),
                 XB, yB, sB, np.repeat(np.arange(5), nB // 5))


def run(part, extra=(), ctx=None, nested=True, trace=True):
    args = H.parse_args(["--part", part] + BASE_ARGS + list(extra))
    H.TRACE = [] if trace else None
    try:
        rows, st, stats = H.run_ctx(ctx or make_ctx(), part, args, learner_axis=True, nested=nested)
        return args, rows, st, stats, (H.TRACE or [])
    finally:
        H.TRACE = None


@pytest.fixture(scope="module")
def cpu_run():
    return run("cpu")


@pytest.fixture(scope="module")
def gpu_run():
    return run("gpu", ["--learners", "mlp"])


@HEAVY
def test_a_methods_share_draws(cpu_run, gpu_run):
    seen = {}
    for part, (args, rows, st, stats, tr) in (("cpu", cpu_run), ("gpu", gpu_run)):
        assert len(tr) > 0
        for e in tr:
            if e["placement"] != "cell" or "cv_fold" in e:
                continue
            seen.setdefault((e["n"], e["draw"]), []).append((part, e["learner"], e["method"], e["alpha"], tuple(int(v) for v in e["sel"])))
    assert {(3, 0), (3, 1), (10, 0), (10, 1), (0, 0), (-1, 0)} <= set(seen)
    for (n, d), lst in seen.items():
        sels = {v[-1] for v in lst}
        assert len(sels) == 1, f"n={n} draw={d}: 방법마다 추출이 다르다"
        want = tuple(int(v) for v in H.draw_cells("T", "x", 1, n, d, 60))
        assert sels == {want}
        if n > 0:
            rng = np.random.RandomState(H.seed_of("T", "x", 1, n, d))
            assert want == tuple(sorted(int(v) for v in rng.choice(60, n, replace=False)))
    methods = {(v[0], v[1], v[2]) for v in seen[(10, 0)]}
    assert {("cpu", "catboost_lo", m) for m in ("D0", "D1", "R0", "R1", "R2", "R3", "V1r")} <= methods
    assert {("cpu", "ridge", m) for m in ("D0", "R1", "R2")} <= methods and {("gpu", "mlp", m) for m in ("D0", "R0", "R1", "R2")} <= methods
    assert {v[3] for v in seen[(10, 0)]} >= {"1", "10", "100", "cont"}


@HEAVY
def test_b_r0_n0_is_source_residual_only(cpu_run):
    args, rows, st, stats, tr = cpu_run
    c = make_ctx()
    e0 = [e for e in tr if e["method"] == "R0" and e["n"] == 0 and e["learner"] == "catboost_lo"]
    assert len(e0) == 1
    e = e0[0]
    assert e["Xtr"].shape[0] == len(c.y_src) and np.all(e["Xtr"][:, 0] < 0)
    assert np.allclose(e["ytr"], c.y_src - c.E0 * c.s_src) and e["w"] is None
    # 저장된 R0(n = 0) 예측은 E0·s + λ·g 이고 R1(n = 0) 과 같다
    for lam in args.LAMS:
        a = st.get(("R0", "catboost_lo", "1", "cell", 0, 0, 0, lam)); b = st.get(("R1", "catboost_lo", "1", "cell", 0, 0, 0, lam))
        assert np.allclose(a[0], b[0])


@HEAVY
def test_b2_mlp_r0_n0_reference_row(gpu_run):
    args, rows, st, stats, tr = gpu_run
    for lam in args.LAMS:                                     # MLP α 축 R0 의 n = 0 기준 행(라벨 순가치의 기준)
        a = st.get(("R0", "mlp", "1", "cell", 0, 0, 0, lam)); b = st.get(("R1", "mlp", "1", "cell", 0, 0, 0, lam))
        assert np.allclose(a[0], b[0])
    assert stats["status"] == "ok" and stats["n_nonfinite_keys"] == 0
    assert any(k.startswith("mlp_alpha|mlp|") for k in stats["n_fit_detail"]) and all(v >= 0 for v in stats["sec_detail"].values())


def _check_rows(tr, c, kappa=10.0):
    n_checked = 0
    for e in tr:
        ids = e["Xtr"][:, 0]; tgt = ids >= 0
        assert not np.any(ids >= 1000), "채점 셀이 학습 행렬에 있다"
        sel = set(int(v) for v in e["sel"])
        out = tgt & ~np.isin(ids.astype(int), list(sel))
        if e["method"] in ("D1", "R2") and "cv_fold" not in e:
            sl = np.array(sorted(sel), int)
            if len(sl):
                E_ls = float((c.sA[sl] @ c.yA[sl]) / (c.sA[sl] @ c.sA[sl])); E1 = (len(sl) * E_ls + kappa * c.E0) / (len(sl) + kappa)
            else:
                E1 = c.E0
            j = ids[out].astype(int)
            want = E1 * c.sA[j] if e["method"] == "D1" else np.zeros(len(j))
            assert np.allclose(e["ytr"][out], want), f"{e['method']} n={e['n']}: 비선택 셀 행이 유사라벨이 아니다"
            n_checked += int(out.sum())
        else:
            assert not out.any(), f"{e['method']} n={e['n']} α={e['alpha']}: 비선택 대상 셀이 학습 행렬에 있다"
        if e["n"] == 0 and "cv_fold" not in e and e["method"] not in ("D1", "R2"):
            assert not tgt.any()
    return n_checked


@HEAVY
def test_c_unselected_labels_never_used(cpu_run, gpu_run):
    c = make_ctx()
    assert _check_rows(cpu_run[4], c) > 0
    _check_rows(gpu_run[4], c)
    assert any("cv_fold" in e for e in cpu_run[4]), "α 중첩 선택 교차검증이 실행되지 않았다"
    assert {e["alpha"] for e in cpu_run[4] if "cv_fold" in e} <= {"10", "100", "cont"}, "중첩 선택 후보에 α = 1 이 있다"
    # 비선택 라벨 교란: n = all 을 빼면 어떤 추출에도 뽑히지 않은 A 셀이 있다. 그 라벨을 바꿔도 모든 저장값이 같아야 한다.
    grid = ["--n-grid", "0,3,10", "--learner-n-grid", "0,3,10"]
    args = H.parse_args(["--part", "cpu"] + BASE_ARGS)
    used = set()
    for n in (3, 10):
        for d in range(args.DRAWS):
            used |= set(H.draw_cells("T", "x", 1, n, d, 60).tolist()) | set(H.draw_blocks("T", "x", 1, n, d, make_ctx().blkA).tolist())
    free = np.array(sorted(set(range(60)) - used), int)
    assert len(free) > 10

    def perturb(y):
        y[free] = y[free] * 7.0 + 500.0
        return y
    for part, extra in (("cpu", []), ("gpu", ["--learners", "mlp"])):
        _, r1, s1, _, _ = run(part, grid + extra, ctx=make_ctx(), trace=False)
        _, r2, s2, _, _ = run(part, grid + extra, ctx=make_ctx(perturb), trace=False)
        k1 = [k for k in s1.keys if k[4] >= 0]; k2 = [k for k in s2.keys if k[4] >= 0]
        assert k1 == k2 and len(k1) > 20
        S1, _ = s1.matrices(k1); S2, _ = s2.matrices(k2)
        assert np.allclose(S1, S2, rtol=1e-9, atol=1e-9), f"{part}: 비선택 라벨을 바꾸자 결과가 달라졌다(누설)"
    # 대조: 라벨 전량(n = all)은 교란에 반응해야 한다(시험이 교란을 실제로 감지함을 확인)
    _, _, s1, _, _ = run("cpu", ["--learners", "ridge"], ctx=make_ctx(), trace=False)
    _, _, s2, _, _ = run("cpu", ["--learners", "ridge"], ctx=make_ctx(perturb), trace=False)
    ka = ("P1", "none", "1", "cell", -1, 0, -1, 0.0)
    assert not np.allclose(s1.get(ka)[0], s2.get(ka)[0])


@HEAVY
def test_d_resume_skips_done_units(tmp_path):
    if not (ROOT / "data" / "processed" / "fidelity_base_v3.csv").exists():
        pytest.skip("실자료 없음")
    argv = ["--part", "cpu", "--learners", "ridge", "--targets", "Russia_W", "--splits", "1", "--n-grid", "0,3", "--draws", "1", "--seeds", "1",
            "--workers", "0", "--threads", "2", "--out-dir", str(tmp_path), "--tag", "t", "--no-summarize", "--allow-local"]
    r1 = H.main(argv)
    assert r1["executed"] == [("Russia_W", "x", 1)] and not r1["resumed"]
    unit = tmp_path / "shards" / "t__cpu__Russia_W__x__s1_unit.json"
    runs = tmp_path / "shards" / "t__cpu__Russia_W__x__s1_runs.csv"
    assert unit.exists() and runs.exists() and (tmp_path / "shards" / "t__cpu__Russia_W__x__s1_blocksse.npz").exists()
    assert not list((tmp_path / "shards").glob("*.tmp*")), "임시 파일이 남았다"
    m0 = (unit.stat().st_mtime_ns, runs.stat().st_mtime_ns)
    r2 = H.main(argv + ["--resume"])
    assert r2["executed"] == [] and r2["resumed"] == [("Russia_W", "x", 1)]
    assert (unit.stat().st_mtime_ns, runs.stat().st_mtime_ns) == m0
    r3 = H.main(argv + ["--resume", "--kappa", "5"])                      # 설정이 다르면 건너뛰지 않는다
    assert r3["executed"] == [("Russia_W", "x", 1)] and not r3["resumed"]
    r4 = H.main(argv + ["--resume"])                                      # 덮어쓴 조각은 원래 설정과 해시가 다르다
    assert r4["executed"] == [("Russia_W", "x", 1)]
    out = H.summarize(H.parse_args(argv))
    assert out is not None and len(out["curve"]) > 0 and {"d_p0", "d_p1", "net_value", "ci_flag"} <= set(out["curve"].columns)
    p0 = out["curve"][out["curve"].method == "P0"]
    assert len(p0) == 1 and abs(float(p0.d_p0.iloc[0])) < 1e-12


@HEAVY
def test_e_mlp_fit_matches_tab_models():
    from polar.tab_models import fit_predict
    rng = np.random.RandomState(0)
    X = rng.randn(300, 5).astype(np.float32); y = (X[:, 0] * 2 + rng.randn(300)).astype(float); Xt = rng.randn(40, 5).astype(np.float32)
    a = fit_predict("mlp", X, y, Xt, seed=3, epochs=2)["pred"]
    b = H.mlp_predict(H.mlp_fit(X, y, 3, 2), Xt)
    assert np.allclose(a, b, atol=1e-5)


# ================================================================ 학습을 하지 않는 시험
def test_f_defaults_are_local_safe(monkeypatch):
    a = H.parse_args([])
    assert a.GPUS == [] and a.workers <= 2 and a.NESTED_ALPHAS == ["10", "100", "cont"] and a.R2_EXCLUDE == []
    g = H.parse_args(["--part", "gpu", "--smoke"])
    assert g.GPUS == [] and g.LEARNERS == ["mlp", "cfm"] and g.epochs <= 3 and g.realmlp_epochs <= 3 and g.TAG == "lg_smoke"
    p = H.parse_args(["--part", "gpu", "--precheck"])
    assert p.TAG == "lg_precheck" and p.LEARNERS == H.GPU_LEARNERS and p.epochs == 100 and p.N_GRID == [0, 40] and p.LEARNER_N == [0, 40]
    assert p.TARGETS == [("Canada", "x")] and p.LEARNER_TARGETS == [("Canada", "x")] and p.SPLITS == [1]
    monkeypatch.delenv("LG_RESCALE", raising=False)
    assert not H.run_permitted(a) and H.run_permitted(H.parse_args(["--allow-local"]))
    with pytest.raises(SystemExit):                                       # 자료를 읽기 전에 거부된다
        H.main(["--part", "cpu", "--targets", "Russia_W", "--workers", "0", "--threads", "1", "--no-summarize"])
    monkeypatch.setenv("LG_RESCALE", "1")
    assert H.run_permitted(a)


def test_g_resume_checks_cfg_hash(tmp_path):
    base = ["--part", "gpu", "--out-dir", str(tmp_path), "--tag", "t"]
    a = H.parse_args(base)
    u = ("Canada", "x", 1, "mlp")
    p = H.shard_paths(a, "gpu", *u)
    assert p["unit"].name == "t__gpu__Canada__x__s1__mlp_unit.json"
    assert H.shard_paths(H.parse_args(["--part", "cpu", "--out-dir", str(tmp_path), "--tag", "t"]), "cpu", "Canada", "x", 1)["unit"].name \
        == "t__cpu__Canada__x__s1_unit.json"
    assert H.unit_state(a, "gpu", *u) == (False, "조각 없음")
    p["runs"].parent.mkdir(parents=True, exist_ok=True)
    p["runs"].write_text("x\n"); p["npz"].write_bytes(b"0")
    cfg = H.unit_cfg(a, "gpu", *u)
    p["unit"].write_text(json.dumps(dict(status="ok", cfg_hash=H.cfg_hash(cfg))))
    assert H.unit_state(a, "gpu", *u) == (True, "ok")
    assert H.unit_state(H.parse_args(base + ["--learners", "mlp,cfm"]), "gpu", *u)[0]          # 학습기를 나눠 제출해도 조각 해시는 같다
    assert H.unit_state(H.parse_args(base + ["--gpus", "0,1", "--workers", "1"]), "gpu", *u)[0]  # 실행 자원은 해시에 없다
    for extra in (["--epochs", "3"], ["--n-grid", "0,10"], ["--seeds", "1"], ["--learner-r2-exclude", "mlp"], ["--r", "5"]):
        ok, why = H.unit_state(H.parse_args(base + extra), "gpu", *u)
        assert not ok and "설정" in why, extra
    p["unit"].write_text(json.dumps(dict(status="failed", cfg_hash=H.cfg_hash(cfg))))
    assert not H.unit_state(a, "gpu", *u)[0]
    p["unit"].write_text(json.dumps(dict(status="ok")))                                        # 개정 전 형식(해시 없음)
    assert not H.unit_state(a, "gpu", *u)[0]
    found = H.find_shards(a)
    assert len(found) == 1 and found[0]["learner"] == "mlp" and found[0]["part"] == "gpu"
    assert H.cfg_hash(cfg, common=True) == H.cfg_hash(H.unit_cfg(a, "gpu", "Lena", "x", 2, "realmlp"), common=True)


class _FakeD:
    def __init__(self, info):
        self.info = info

    def split_structure(self, t):
        return self.info[t]


def test_h_unit_priority_and_base_lam():
    names = ("CA-2", "Greenland", "AL-3", "Alaska", "Russia_W", "Lena")
    info = {t: {sp: dict(valid=t != "Greenland", n_A=15 if t == "Russia_W" else 100) for sp in (1, 2)} for t in names}
    a = H.parse_args(["--part", "gpu"])
    D = _FakeD(info)
    units = [(t, "x", sp, lr) for t in names for sp in (2, 1) for lr in ("realmlp", "ftt", "mlp")]
    units.sort(key=lambda u: H.unit_priority(a, D, u))
    assert all(u[0] == "Greenland" for u in units[-6:]), "무효 분할은 맨 뒤"
    ok = [u for u in units if u[0] != "Greenland"]
    assert [u[3] for u in ok] == ["mlp"] * 10 + ["ftt"] * 10 + ["realmlp"] * 10, "빠른 학습기 먼저"
    assert [(u[0], u[2]) for u in ok[:10]] == [("Lena", 1), ("Russia_W", 1), ("Lena", 2), ("Russia_W", 2), ("Alaska", 1), ("Alaska", 2),
                                                 ("AL-3", 1), ("AL-3", 2), ("CA-2", 1), ("CA-2", 2)]
    c = [("AL-3", "x", 1, None), ("AL-3", "i", 1, None), ("Lena", "x", 1, None)]
    c.sort(key=lambda u: H.unit_priority(a, D, u))
    assert [(u[0], u[1]) for u in c] == [("Lena", "x"), ("AL-3", "i"), ("AL-3", "x")]
    assert [H.base_lam(m) for m in ("P1", "V1", "D0", "D1", "R0", "R1", "R2", "R3", "V1r")] == [0.0, 0.0, 1.0, 1.0] + [H.LAM_BASE] * 5
    assert H.unit_name(("Lena", "x", 1, None)) == "Lena|x|s1" and H.unit_name(("Lena", "x", 1, "mlp")) == "Lena|x|s1|mlp"


G_A = ("R1", "catboost_lo", "1", "cell", 40, 0.25)
G_B = ("P1", "none", "1", "cell", 40, 0.0)


def _mk_tm(name, nb, seed, nboot=200):
    """합성 저장소: 분할 2개, 블록마다 5셀. 방법 A 의 오차 SD 1, 방법 B 의 오차 SD 3(A 가 뚜렷이 낫다)."""
    rng = np.random.RandomState(seed)
    by = {}
    for sp in (1, 2):
        blocks = np.repeat(np.arange(nb), 5)
        st = H.BlockStore(name, sp, blocks)
        y = rng.randn(len(blocks)) * 5 + 50
        st.add(("P1", "none", "1", "cell", 40, 0, -1, 0.0), y, y + rng.randn(len(y)) * 3.0)
        st.add(("R1", "catboost_lo", "1", "cell", 40, 0, 0, 0.25), y, y + rng.randn(len(y)) * 1.0)
        by[sp] = st
    return H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in (1, 2)}, nboot)


def _mean_row(rows):
    m = [r for r in rows if str(r["target"]).startswith("MEAN[")]
    assert len(m) == 1
    return m[0]


def test_i_strat_mean_pool_and_assert(monkeypatch):
    names = [f"{t}|x" for t in H.MAIN4]
    tms = {nm: _mk_tm(nm, 10, seed=i) for i, nm in enumerate(names)}
    rows = H.strat_mean("t", tms, names, lambda tm: G_A, lambda tm: G_B)
    m = _mean_row(rows)
    assert len(rows) == 5 and m["n_ci_regions"] == 4 and not m["undetermined"] and H.mean_state(m) == "improve" and m["ci_hi"] < 0
    # 채점 블록 합집합이 8 미만인 지역은 CI 풀에서 빠지고 지역 수에 반영된다
    tms2 = dict(tms); tms2["Russia_W|x"] = _mk_tm("Russia_W|x", 5, seed=9)
    m2 = _mean_row(H.strat_mean("t", tms2, names, lambda tm: G_A, lambda tm: G_B))
    assert m2["n_ci_regions"] == 3 and H.mean_state(m2) == "improve"
    # 지역 1개: 판정 불가
    m1 = _mean_row(H.strat_mean("t", {names[0]: tms[names[0]]}, names, lambda tm: G_A, lambda tm: G_B))
    assert m1["n_ci_regions"] == 1 and m1["undetermined"] and H.mean_state(m1) == "undetermined"
    # 점 추정 전용 대상은 CI 를 만들지 않는다
    tp = _mk_tm("Russia_C|x", 10, seed=3)
    assert tp.point_only and not tp.has_ci and tp.nboot == 0
    r = H.contrast(tp, G_A, G_B)
    assert np.isfinite(r["delta"]) and not np.isfinite(r["ci_hi"])
    # 단언 실패(점 추정치가 백분위 CI 밖): 지역 행은 보존하고 평균은 판정 불가로 둔다

    def boom(*a, **k):
        raise AssertionError("점 추정치가 CI 밖")
    monkeypatch.setattr(H, "summarize_delta", boom)
    rows = H.strat_mean("t", tms, names, lambda tm: G_A, lambda tm: G_B)
    m = _mean_row(rows)
    assert len(rows) == 5 and sorted(r["target"] for r in rows if r is not m) == sorted(names)
    assert str(m["ci_flag"]).startswith("assert") and m["undetermined"] and H.mean_state(m) == "undetermined" and m["n_ci_regions"] == 4
    assert np.isfinite(m["ci_hi"]) and np.isfinite(m["delta"])


def test_j_nonfinite_predictions_are_not_stored():
    c = make_ctx()
    c.sB[3] = np.nan                                           # P0·P1 예측에 비유한 값이 하나 생긴다
    args = H.parse_args(["--part", "cpu", "--methods", "P0,P1", "--learners", "ridge"] + BASE_ARGS)
    rows, st, stats = H.run_ctx(c, "cpu", args, learner_axis=True, nested=False)
    assert len(rows) > 0 and len(st) == 0 and stats["n_nonfinite_keys"] == len(rows) and stats["status"] != "ok"
    assert all(r["n_nonfinite"] == 1 and r["fit_flag"] == "nonfinite" and not np.isfinite(r["rmse_cm"]) for r in rows)
    rows, st, stats = H.run_ctx(make_ctx(), "cpu", args, learner_axis=True, nested=False)
    assert len(st) == len(rows) > 0 and stats["status"] == "ok" and all(r["n_nonfinite"] == 0 and r["fit_flag"] == "" for r in rows)


def test_k_dry_count_by_learner_and_r2_exclude():
    a = H.parse_args(["--part", "gpu", "--learners", "realmlp,mlp"] + BASE_ARGS)
    _, _, s = H.run_ctx(make_ctx(), "gpu", a, dry=True, learners=["realmlp"])
    d = s["n_fit_detail"]
    assert d["learner|realmlp|R2"] > 0 and d["learner|realmlp|R1"] > 0 and not any("|mlp|" in k for k in d) and s["status"] == "ok"
    assert s["rows_detail"]["learner|realmlp|R2"] > s["rows_detail"]["learner|realmlp|R1"] and s["est_detail"]["learner|realmlp|R2"] > 0
    b = H.parse_args(["--part", "gpu", "--learners", "realmlp,mlp", "--learner-r2-exclude", "realmlp"] + BASE_ARGS)
    _, _, s2 = H.run_ctx(make_ctx(), "gpu", b, dry=True, learners=["realmlp"])
    assert "learner|realmlp|R2" not in s2["n_fit_detail"] and s2["n_fit_detail"]["learner|realmlp|R1"] == d["learner|realmlp|R1"]
    _, _, s3 = H.run_ctx(make_ctx(), "gpu", a, dry=True, learners=["mlp"])
    assert any(k.startswith("mlp_alpha|mlp|") for k in s3["n_fit_detail"]) and any(k.startswith("mlp_cont|mlp|") for k in s3["n_fit_detail"])
