"""scripts/2_evaluation/h44_hier_predictive_ladder.py 단위 시험(계획 docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 3절, spec_A 11절).

가벼운 시험(합성 자료, 학습 없음, torch 를 적재하지 않는다)
(a) hyper_from_blocks 가 알려진 분산 성분의 모의 자료(지역 8, 블록 30)에서 σ², τ_b², τ_r² 를 회복한다(허용 상대 차 20 %).
    한 자료의 τ_b² 추정은 표집 오차(상대 표준오차 약 10 %)가 커서 우연히 기준을 넘을 수 있으므로 자료 5개의 평균으로 본다.
    다중도 m 은 블록 m 개로 본 것과 같다.
(b) 단 (ii)의 닫힌 형태와 posterior_grid + predictive_q(정규 사전)의 분위 차 0.002 이내. 단 (iv)의 구적(ν = ∞, σ 고정)과
    단 (v)의 CDF 표 합성(v_quantiles, 정규 잔차)이 같은 모형의 격자 해와 맞는다.
(c) 모의 기반 보정 점검(SBC): 단 (iii) 모형에서 생성한 자료의 PIT 평균 0.5 ± 0.03, 90 % 커버리지 0.90 ± 0.03.
(d) n = 0 의 예측 중앙값이 P0 와 같다(단 i0, i, ii, iii, B4, 범위 B 와 all).
(e) 라벨 추출이 h40.draw_cells 와 같다.
(f) n = 3 에서 무한 구간이 없다(B4, 단 i, ii, iii).
(g) 다중도 1 의 2단 통계가 점 추정(저장소 평균)과 같다.
(h) 판정 규칙의 분기(지지, 부분 지지, 동등, 기각, 미결정, 판정 불가, 커버리지 미달, 알래스카 의존). (개정 1) 열세가 판정 불가보다 먼저,
    2단 분포가 없는 확인적 대비는 '판정 불가(2단 CI 없음)', 짝지음 점검 실패는 '판정 불가(계산 실패)', 부분 지지의 n 비교.
(i) --precision-only 출력에 delta, CI 끝점, 판정, 기준 점수 열이 없다. 합성 저장소로 build_tests, build_curve 가 돈다(보조 열 aux:* 포함).
(j) 허용 표지 없이 실행을 거부한다(자료 적재 전). 허용 표지가 없으면 스레드 1. (개정 1) 스레드 합계 초과는 거부(낮추지 않는다),
    자동 집계 풀까지 센다, LG_RESCALE=1 에도 GPU 허용 목록을 적용한다.
(m) 게이트 판정(gate_decide)의 상태(통과, 미통과, 판정 불가(적합 실패), 판정 불가(지역 부족))와 강제 값.
(n) (개정 1) τ_r 사후의 t 성분이 칸 적분 질량이고 다시 정규화하지 않는다(수치 적분과 대조).
(o) (개정 1) 게이트의 척도 보정 계수 c 가 {k, q} 를 뺀 표본 밖 적합으로 계산된다(CatBoost 적합 수 K², 흐름 seed 없이 합성 자료).
(p) (개정 1) 흐름 조각은 게이트 unit.json 이 바뀌면(gate_ref 불일치) 완료로 보지 않는다.
학습을 하는 시험(HEAVY, LG_RUN_HEAVY=1 일 때만)
(k) 스모크 경로(cpu, gate, flow)의 종료 코드 0(실제 자료, CPU, 스레드 2, 산출은 임시 디렉터리).
실행(로컬 GPU 서버, 사용자 지시 2026-09-30. CPU 만 쓴다. 스레드 2, nice 10):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 python3 -m pytest -q tests/test_h44_ladder.py
  (무거운 시험까지: 앞에 LG_RUN_HEAVY=1)
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
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

HEAVY = pytest.mark.skipif(os.environ.get("LG_RUN_HEAVY", "") != "1",
                           reason="학습을 하는 시험은 LG_RUN_HEAVY=1 일 때만 실행한다(공유 서버 CPU 보호)")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "scripts" / "2_evaluation" / "h44_hier_predictive_ladder.py"


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("h44_hier_predictive_ladder", SCRIPT)
H = M.H
LC = M.LC
REGIONS = ["Lena", "Alaska", "Canada", "Russia_W", "Russia_E"]
FEATS = [f"x{j}" for j in range(5)]


def _synth(seed=0, nblk=14, cells=(6, 30), tau_r=0.2, tau_b=0.15, sigma=0.25, E=1.6):
    """합성 자료: 지역 5개(이름은 주 대상과 같다), 지역마다 0.5° 블록 nblk 개, u = a_r + b + ε, y = E·s·exp(u)."""
    rng = np.random.RandomState(seed)
    rows = []
    a_r = dict(zip(REGIONS, rng.normal(0, tau_r, len(REGIONS))))
    for ri, r in enumerate(REGIONS):
        for b in range(nblk):
            bb = rng.normal(0, tau_b)
            lat0, lon0 = 55.0 + 4.0 * ri + 0.5 * (b % 5), -160.0 + 25.0 * ri + 0.5 * (b // 5)
            for _ in range(rng.randint(*cells)):
                s = rng.uniform(15, 45)
                y = E * s * np.exp(a_r[r] + bb + rng.normal(0, sigma))
                rows.append(dict(loc_id=len(rows), lat=lat0 + 0.4 * rng.rand(), lon=lon0 + 0.4 * rng.rand(), block=100000 * (ri + 1) + b,
                                 macro=r, y=y, s=s, alt_cm=y, cci_alt=1.0, e5_sqrt_tdd_soil=1.0, **dict(zip(FEATS, rng.randn(len(FEATS))))))
    df = pd.DataFrame(rows)
    n = len(df)
    flags = dict(early=rng.rand(n) < 0.2, gpr=rng.rand(n) < 0.3, probe=rng.rand(n) < 0.6)
    return df, flags


def _cfg(**kw):
    c = SimpleNamespace(N_GRID=[0, 3, 10], DRAWS=2, RUNGS=["P0", "P1", "B4", "hier2@h37", "wconf", "i0", "i", "ii", "iii", "iii+c"],
                        SENS=["t3", "stau025", "w_cell", "w_beq", "sig_pool", "probe_sig", "noearly"], CB_SEEDS=[0], FLOW_SEEDS=[0, 1, 2],
                        kappa=10.0, nu=4.0, s_tau=0.5, min_cal_cells=20, threads=1, epochs=3, clamp=2.0, nboot=50)
    c.__dict__.update(kw)
    return c


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    df, flags = _synth(0)
    fr = M.Frame(df, flags, feats=FEATS)
    t_idx = np.where(df.macro.values == "Lena")[0]
    src_idx = np.where(df.macro.values != "Lena")[0]
    cfg = _cfg()
    emu = M.emu_compute(fr, t_idx, src_idx, "Lena", "x", cfg, do_iv=False)
    p = tmp_path_factory.mktemp("emu") / "emu.npz"
    M.save_emu(p, emu)
    ev = M.EmuView(M.load_emu(p), fr)
    res = {sp: M.cpu_compute(fr, ev, "Lena", "x", sp, cfg, all_scope=(sp == 1)) for sp in (1, 2)}
    return SimpleNamespace(df=df, flags=flags, fr=fr, t_idx=t_idx, src_idx=src_idx, cfg=cfg, emu=emu, ev=ev, res=res)


def _store(res, name):
    return [st for st in res[0] if st.target == name][0]


# ---------------------------------------------------------------- (a)
def _sim_blocks(rng, a_true, J=30, tau_b2=0.04, sigma2=0.09):
    reg, col, u = [], [], []
    for k, ak in enumerate(a_true):
        for j in range(J):
            nkj = rng.randint(5, 30)
            b = rng.normal(0, np.sqrt(tau_b2))
            reg += [f"R{k}"] * nkj; col += [k * 1000 + j] * nkj
            u += list(ak + b + rng.normal(0, np.sqrt(sigma2), nkj))
    return M.block_table(np.array(reg), np.array(col), np.array(u))


def test_a_hyper_recovery():
    rng = np.random.RandomState(1)
    a_true = np.array([0.3, -0.3] * 4)                       # 지역 8, 위치 0 기준 2차 적률 0.09
    hs = [M.hyper_bt(_sim_blocks(rng, a_true)) for _ in range(5)]
    for key, true in (("sigma2", 0.09), ("tau_b2", 0.04), ("tau_r2", 0.09)):
        est = float(np.mean([h[key] for h in hs]))
        assert abs(est / true - 1) < 0.2, (key, est, true)
    h = hs[0]
    assert h["K"] == 8 and not h["flags"]
    np.testing.assert_allclose(h["a_k"], a_true, atol=0.15)
    # 단 (i)의 값: 셀 평균 기준 분산은 σ² + τ_b² 근처다
    assert 0.09 < h["sigma1_2"] < 0.2


def test_a_hyper_mult_equals_duplicated_blocks():
    rng = np.random.RandomState(2)
    bt = _sim_blocks(rng, np.array([0.2, -0.1, 0.05]), J=12)
    m = rng.randint(0, 3, len(bt["n"])).astype(float)
    h_m = M.hyper_bt(bt, mult=m)
    rep = np.repeat(np.arange(len(bt["n"])), m.astype(int))
    bt_dup = {k: np.asarray(v)[rep] for k, v in bt.items()}
    bt_dup["col"] = np.arange(len(rep))                      # 뽑힌 사본을 서로 다른 블록으로 둔다
    h_d = M.hyper_bt(bt_dup)
    for key in ("sigma2", "tau_b2", "tau_r2", "sigma1_2", "tau_r1_2"):
        assert h_m[key] == pytest.approx(h_d[key], rel=1e-10), key
    np.testing.assert_allclose(h_m["a_k"], h_d["a_k"], rtol=1e-10)
    for wm in ("cell", "beq"):
        np.testing.assert_allclose(M.hyper_bt(bt, mult=m, weight_mode=wm)["v_k"], M.hyper_bt(bt_dup, weight_mode=wm)["v_k"], rtol=1e-10)


# ---------------------------------------------------------------- (b)
LABEL_SETS = [(np.zeros(0), np.zeros(0)), (np.array([3.0]), np.array([0.2])), (np.array([5.0, 2.0, 8.0]), np.array([0.1, -0.3, 0.25]))]


def test_b_closed_form_vs_grid():
    tau_r2, tau_b2, sigma2 = 0.05, 0.03, 0.08
    pa = M.normal_prior_mass(tau_r2)
    for n_j, ub in LABEL_SETS:
        m, v = M.posterior_normal(n_j, ub, tau_r2, tau_b2, sigma2)
        q_cf = M.normal_q(m, v + tau_b2 + sigma2)
        q_gr = M.predictive_q(M.posterior_grid(pa, n_j, ub, tau_b2, sigma2), np.sqrt(tau_b2 + sigma2))
        assert np.max(np.abs(q_cf - q_gr)) < 0.002
        assert np.all(np.diff(q_gr) > 0)


def test_b_iv_and_v_quadrature_match_normal_model():
    tau_r2, tau_b2, sigma2 = 0.05, 0.03, 0.08
    sig = np.sqrt(sigma2)
    pa = M.normal_prior_mass(tau_r2)
    u_L = np.array([0.15, 0.30, -0.05, 0.22, 0.10, 0.40]); blk = np.array([1, 1, 2, 2, 2, 3])
    nJ, ub = M.label_blocks(u_L, blk)
    post = M.posterior_grid(pa, nJ, ub, tau_b2, sigma2)
    q_ref = M.predictive_q(post, np.sqrt(tau_b2 + sigma2))
    # 단 (iv): ν = ∞, σ 고정이면 단 (iii)의 정규 잔차 모형과 같다(b 101점 구적, σ 수준 보간 오차 포함)
    post_iv = M.post_from_loglik(pa, M.iv_label_loglik(u_L, blk, np.full(len(u_L), sig), tau_b2, np.inf))
    assert np.max(np.abs(np.cumsum(post_iv) - np.cumsum(post))) < 2e-3
    pc, _ = M.pc_from_post(post_iv, tau_b2)
    q_iv = M.iv_cell_quantiles(M.iv_tables(pc, np.inf), np.array([sig]))[:, 0]
    assert np.max(np.abs(q_iv - q_ref)) < 6e-3
    # 단 (v): 셀별 ε CDF 표(정규)와 c 질량의 합성
    Fe = np.tile(M.ndtr(M.EPS_GRID / sig), (2, 1))
    q_v = M.v_quantiles(Fe, M.coarsen2(M.pc_from_post(post, tau_b2)[0]))
    assert q_v.shape == (99, 2)
    assert np.max(np.abs(q_v[:, 0] - q_ref)) < 0.012
    # 단 (v)의 라벨 우도: 정규 로그 밀도 표이면 단 (iv)(ν = ∞)와 같다
    lf = np.tile(-0.5 * (M.EPS_GRID / sig) ** 2 - 0.5 * np.log(2 * np.pi) - np.log(sig), (len(u_L), 1))
    post_v = M.post_from_loglik(pa, M.v_label_loglik(u_L, blk, lf, tau_b2))
    assert np.max(np.abs(np.cumsum(post_v) - np.cumsum(post_iv))) < 2e-3


def test_b_lattice_and_quantile_helpers():
    F = np.array([0.0, 0.1, 0.1, 0.5, 0.9, 1.0])
    g = np.arange(6) * 1.0
    q = M.q_from_cdf(F, g, [0.05, 0.1, 0.3, 0.5, 0.95, 1.0])
    np.testing.assert_allclose(q, [0.5, 1.0, 2.5, 3.0, 4.5, 5.0])
    np.testing.assert_allclose(M.q_from_cdf(np.vstack([F, F]), g, [0.3]), [[2.5], [2.5]])
    p = np.zeros(len(M.A_GRID)); p[M.A_HALF] = 1.0          # a = 0 점 질량과 N(0, 1)의 합성 = 표준 정규
    q = M.predictive_q(p, 1.0, [0.05, 0.5, 0.95])
    np.testing.assert_allclose(q, [-1.6449, 0.0, 1.6449], atol=2e-4)
    pc = np.zeros(len(M.U_GRID)); pc[M.U_HALF] = 1.0
    np.testing.assert_allclose(M.iv_tables(pc, 5.0, levels=[1.0], taus=[0.9])[0, 0], M.t_ppf(0.9, 5.0) * M.kappa_nu(5.0), atol=1e-3)
    assert M.coarsen2(np.full(1601, 1 / 1601)).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- (c)
def test_c_sbc_iii():
    rng = np.random.RandomState(3)
    tau_b2, sigma2 = 0.03, 0.06
    tg, pt = M.tau_r_posterior([0.2, -0.15, 0.05, 0.3], [0.002] * 4, 4.0, 0.5)
    pa = M.prior_a(tg, pt, 4.0)
    cdf_a = np.cumsum(pa)
    sd = np.sqrt(tau_b2 + sigma2)
    pits, cov = [], []
    for _ in range(1500):
        a = M.A_GRID[min(np.searchsorted(cdf_a, rng.rand()), len(M.A_GRID) - 1)]
        blk = np.repeat(np.arange(4), 3)
        u = a + rng.normal(0, np.sqrt(tau_b2), 4)[blk] + rng.normal(0, np.sqrt(sigma2), len(blk))
        nJ, ub = M.label_blocks(u, blk)
        q = M.predictive_q(M.posterior_grid(pa, nJ, ub, tau_b2, sigma2), sd)
        ustar = a + rng.normal(0, sd)                          # 새 블록의 셀: b 와 ε 를 새로 뽑는다
        pits.append(float(np.interp(ustar, q, LC.TAUS, left=0.005, right=0.995)))
        cov.append(q[4] <= ustar <= q[94])
    print(f"[c] SBC(단 iii, 1,500회): PIT 평균 {np.mean(pits):.4f}(0.5 ± 0.03), 90 % 커버리지 {np.mean(cov):.4f}(0.90 ± 0.03)")
    assert abs(np.mean(pits) - 0.5) < 0.03
    assert abs(np.mean(cov) - 0.90) < 0.03
    qc = M.quadrature_check(n_sim=200, seed=4)
    print(f"[c] 구적 격자 h = {M.HG}: 닫힌 형태 대비 분위 최대 절대 차 {float(qc.value.iloc[0]):.3e}(기준 0.002, 계획서 2.9절)")                     # 사전 점검용 함수의 형식과 닫힌 형태 대조(SBC 행은 표집 오차가 커서 형식만 본다)
    assert list(qc.check) == ["closed_form_max_abs_diff", "sbc_pit_mean", "sbc_cov90"]
    assert bool(qc.ok.iloc[0])


# ---------------------------------------------------------------- (d)
def test_d_n0_median_equals_P0(synth):
    for name, sp in (("Lena", 1), ("Lena|all", 0)):
        res = synth.res[1]
        st = _store(res, name)
        assert st.split == sp
        b0 = st.mean(("P0", "", 0, 0, -1), "bias")
        for mth in ("i0", "i", "ii", "iii", "B4"):
            key = (mth, "", 0, 0, -1)
            assert key in st, (name, mth)
            assert st.mean(key, "bias") == pytest.approx(b0, rel=1e-6, abs=1e-6), (name, mth)
    q, _, _ = M.iii_quantiles(synth.ev.prior("main"), np.zeros(0), np.zeros(0), synth.ev.hyp("main"))
    assert abs(q[49]) < 1e-6


# ---------------------------------------------------------------- (e)
def test_e_labels_match_h40(synth):
    for sp in (1, 2):
        stores, rows, cells, stats = synth.res[sp]
        for j in range(len(cells["nd_n"])):
            n, d = int(cells["nd_n"][j]), int(cells["nd_d"][j])
            sel = H.draw_cells("Lena", "x", sp, n, d, stats["n_A"])
            np.testing.assert_array_equal(cells["lab_pos"][cells["nd_ls"][j]:cells["nd_le"][j]], sel)
        assert [tuple(v) for v in zip(cells["nd_n"], cells["nd_d"])] == H.cells_of(synth.cfg.N_GRID, synth.cfg.DRAWS, stats["n_A"])


# ---------------------------------------------------------------- (f)
def test_f_no_infinite_intervals(synth):
    for sp in (1, 2):
        runs = pd.DataFrame(synth.res[sp][1])
        r3 = runs[(runs.n == 3) & (runs.variant == "") & runs.method.isin(["B4", "i", "ii", "iii"])]
        assert set(r3.method) == {"B4", "i", "ii", "iii"}
        assert (r3.inf10 == 0).all() and (r3.inf20 == 0).all()
        assert np.isfinite(r3.is10).all()


# ---------------------------------------------------------------- (g)
def test_g_two_stage_mult_one_equals_point(synth):
    cells = {sp: synth.res[sp][2] for sp in (1, 2)}
    out = M.two_stage_compute(synth.ev, cells, np.ones((1, synth.fr.bi.n_cols), np.uint8), synth.cfg)
    stores = {sp: _store(synth.res[sp], "Lena") for sp in (1, 2)}
    assert out["n_nan_rep"] == 0
    for mth in M.TWO_STAGE_METHODS:
        for n in out["n_list"]:
            for j, met in enumerate(M.M2):
                if mth == "P1" and met != "se":
                    continue
                lp = M.level_point(stores, M.kfn(mth, "", n), met)
                d = out["dist"][out["n_list"].index(n), out["methods"].index(mth), 0, j]
                np.testing.assert_allclose(d, [lp["cell"], lp["beq"]], rtol=1e-8, atol=1e-9, err_msg=f"{mth} n={n} {met}")


def test_g_two_stage_runs_with_resampling(synth):
    cells = {sp: synth.res[sp][2] for sp in (1, 2)}
    W = LC.global_mult(synth.fr.bi, 20, 0)
    out = M.two_stage_compute(synth.ev, cells, W, synth.cfg)
    d = out["dist"]
    assert d.shape == (len(out["n_list"]), len(M.TWO_STAGE_METHODS), 20, len(M.M2), 2)
    i3 = out["methods"].index("iii")
    assert np.isfinite(d[:, i3, :, 0, 0]).mean() > 0.9     # 구간 점수(셀 가중)


# ---------------------------------------------------------------- (h)
def test_h_a1_decision_branches():
    ok = dict(verdict="우세", cov_ok=True)
    assert M.a1_decide({10: ok, 40: ok})[0] == "지지"
    assert M.a1_decide({10: ok, 40: dict(verdict="미결정", cov_ok=True)})[0] == "부분 지지(n = 10)"
    assert M.a1_decide({10: dict(verdict="미결정", cov_ok=True), 40: ok})[0] == "부분 지지(n = 40)"
    assert M.a1_decide({10: dict(verdict="동등", cov_ok=True), 40: dict(verdict="동등", cov_ok=True)})[0] == "동등"
    assert M.a1_decide({10: ok, 40: dict(verdict="열세", cov_ok=True)})[0] == "기각"
    assert M.a1_decide({10: dict(verdict="미결정", cov_ok=True), 40: dict(verdict="동등", cov_ok=True)})[0] == "미결정"
    assert M.a1_decide({10: ok, 40: ok}, pool_ok=False)[0].startswith("판정 불가")
    assert M.a1_decide({10: ok, 40: ok}, splits_ok=False)[0].startswith("판정 불가")
    assert M.a1_decide({10: ok, 40: dict(verdict="판정 불가")})[0] == "판정 불가"
    assert M.a1_decide({10: ok})[0] == "판정 불가"                   # 행 없음은 지지 쪽으로 세지 않는다
    weak = dict(verdict="우세", cov_ok=False)
    assert M.a1_decide({10: weak, 40: weak})[0] == "우세(커버리지 미달)"
    assert M.a1_decide({10: ok, 40: weak})[0] == "부분 지지(n = 10)"
    # (개정 1) 열세 검사가 판정 불가 검사보다 먼저다. 판정 불가의 사유를 그대로 돌려준다
    assert M.a1_decide({10: dict(verdict="판정 불가"), 40: dict(verdict="열세")})[0] == "기각(다른 n 판정 불가)"
    assert M.a1_decide({10: ok, 40: dict(verdict="판정 불가(2단 CI 없음)")})[0] == "판정 불가(2단 CI 없음)"
    assert M.alaska_note("지지", "미결정") == "알래스카 의존"
    assert M.alaska_note("부분 지지(n = 10)", "부분 지지(n = 40)") == "알래스카 의존"     # 개정 1: 지지하는 n 까지 비교
    assert M.alaska_note("부분 지지(n = 10)", "부분 지지(n = 10)") == ""
    assert M.alaska_note("판정 불가(풀 지역 3개 미만)", "판정 불가") == ""
    # (개정 1) 확인적 대비의 2단 요구와 짝지음 점검
    base = dict(lo1=-3.0, hi1=-1.0, lob1=-3.0, hib1=-1.0, lo2=np.nan, hi2=np.nan, lob2=np.nan, hib2=np.nan, ref_cell=100.0, ref_beq=90.0,
                dist1=np.linspace(-3, -1, 50), dist2=None, has2=False, pair_ok=True)
    v = M.contrast_verdict(base, "is10", require2=True)
    assert v["verdict"] == "판정 불가(2단 CI 없음)" and v["verdict_1stage"] == "우세" and np.isnan(v["p_boot"])
    assert M.contrast_verdict(base, "is10")["verdict"] == "우세"                # 보조 대비는 1단으로 대신한다(ci_kind 1단)
    two = dict(base, lo2=-2.5, hi2=-0.5, lob2=-2.5, hib2=-0.5, dist2=np.linspace(-2.5, -0.5, 50), has2=True)
    assert M.contrast_verdict(two, "is10", require2=True)["verdict"] == "우세"
    assert M.contrast_verdict(dict(two, pair_ok=False), "is10", require2=True)["verdict"] == "판정 불가(계산 실패)"
    # 4분 판정(LC.verdict4)과 정밀도 규칙(LC.eq_state)
    assert LC.verdict4(-3, -1, -2, -0.5, (1.0, 1.0)) == "우세"
    assert LC.verdict4(-0.5, 0.4, -0.3, 0.2, (1.0, 1.0)) == "동등"
    assert LC.eq_state(-3, 3, -1, 1, (1.0, 1.0)) == "검정 불가(정밀도 미달)"


# ---------------------------------------------------------------- (i)
def _ts_from(synth, names):
    TSs = {}
    for t in names:
        ts = M.TS(t, "x")
        for sp in (1, 2):
            stores, rows, cells, stats = M.cpu_compute(synth.fr, synth.ev, t, "x", sp, synth.cfg, all_scope=(sp == 1))
            for st in stores:
                ts.stores[(st.target, st.split)] = st
            ts.runs.append(pd.DataFrame(rows))
        ts.splits_expected = [1, 2]
        ts.emu_meta = synth.ev.meta
        TSs[(t, "x")] = ts
    return TSs


def test_i_precision_frame_has_no_delta():
    rows = [dict(M._prec_row("LGU-A1", "iii − B4", "is10", "POOL3", 10, "cell", "2단", 0.8, 10.0, 2000, 3), delta=-1.0, ci_lo=-2.0,
                 ci_hi=0.1, verdict="미결정", ref_score=10.0, half_width=0.8)]
    df = M.precision_frame(rows)
    for c in ("delta", "delta_beq", "ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq", "verdict", "ref_score", "half_width"):
        assert c not in df.columns
    assert list(df.columns) == list(M.PREC_COLS)
    assert list(M.precision_frame([]).columns) == list(M.PREC_COLS)
    r = rows[0]                                                          # 반폭 비율 0.08: 본 실행 규칙(5 %)은 미달, 사전 점검 규칙(10 %)은 등록 없음
    assert abs(r["half_width_rel"] - 0.08) < 1e-12 and np.isnan(r["half_width_abs"])
    assert r["main_rule_5pct"] == "검정 불가(정밀도 미달)" and r["precheck_rule_10pct"] == "등록 없음"
    r2 = M._prec_row("LGU-A1", "iii − B4", "is10", "POOL3", 10, "cell", "2단", 1.2, 10.0, 2000, 3)
    assert r2["precheck_rule_10pct"] == "본 실행 전 동등 판정 검정 불가 등록"
    r3 = M._prec_row("LGU-A3", "iii − P1", "se", "주 4지역", 3, "max", "2단", 0.3, np.nan, 2000, 4, rel=False)
    assert r3["half_width_abs"] == 0.3 and np.isnan(r3["half_width_rel"]) and r3["main_rule_5pct"] == "검정 가능" and r3["precheck_rule_10pct"] == ""


def test_i_build_tests_and_curve_on_synthetic_stores(synth):
    TSs = _ts_from(synth, M.POOL3)
    a = SimpleNamespace(nboot=50)
    tests, sens, prec = M.build_tests(a, TSs)
    assert {"LGU-A1", "LGU-A2", "LGU-A3", "LGU-A5", "LGU-A7"} <= set(tests.test)
    a1 = tests[(tests.test == "LGU-A1") & (tests.n.astype(str) == "10,40")]
    assert len(a1) == 1 and isinstance(a1.verdict.iloc[0], str)
    # 2단 분포(ts.boot)가 없으므로 확인적 판정은 1단으로 대신하지 않는다(개정 1)
    a1n = tests[(tests.test == "LGU-A1") & (tests.n.astype(str).isin(["10", "40"]))]
    assert (a1n.verdict == "판정 불가(2단 CI 없음)").all() and a1n.verdict_1stage.notna().all()
    assert str(a1.verdict.iloc[0]).startswith("판정 불가") and a1.ci_kind.iloc[0] == "2단 없음"
    a3 = tests[tests.test == "LGU-A3"]
    assert (a3.verdict == "판정 불가(2단 CI 없음)").all() and "holm_p" in a3.columns
    assert prec and all("delta" not in r and "ci_lo" not in r and "verdict" not in r and "ref_score" not in r for r in prec)
    assert all(r["test"] in ("LGU-A1", "LGU-A3") for r in prec)
    curve = M.build_curve(a, TSs)
    assert {"is10", "is10_beq", "cov10", "is10_lo1", "net_is10", "ci_kind", "in_band"} <= set(curve.columns)
    assert (curve[(curve.n == 0)].net_is10.dropna() == 0).all()
    # 보조 열(개정 1): 절단 평균(i, ii)과 사후 평균 E(i0, i, ii, iii)
    aux = curve[curve.variant.astype(str).str.startswith("aux:")]
    assert {("i", "aux:tmean"), ("ii", "aux:tmean"), ("iii", "aux:Emean"), ("ii", "aux:Emean")} <= set(zip(aux.method, aux.variant))
    assert np.isfinite(aux.se).all() and aux.is10.isna().all()
    assert len(sens) > 0 and len(M.build_pit(pd.concat([pd.concat(ts.runs) for ts in TSs.values()]))) > 0


# ---------------------------------------------------------------- (j)
def test_j_refuses_without_permission(monkeypatch):
    monkeypatch.delenv("LG_RESCALE", raising=False)

    def boom(a):
        raise AssertionError("허용 표지 없이 자료를 읽었다")
    monkeypatch.setattr(M, "get_D", boom)
    for argv in (["--part", "cpu"], ["--smoke"], ["--summarize-only"], ["--part", "gate", "--gpus", "2"], ["--precheck"]):
        with pytest.raises(SystemExit) as e:
            M.main(argv)
        assert "거부" in str(e.value), argv
    with pytest.raises(AssertionError):                               # --count-only 는 허용된다(자료 적재까지 간다)
        M.main(["--count-only"])
    assert M.parse_args(["--threads", "8"]).threads == 1
    assert M.parse_args(["--threads", "8", "--allow-local"]).threads == 8


def test_j_local_guards(monkeypatch):
    monkeypatch.delenv("LG_RESCALE", raising=False)
    a = M.parse_args(["--allow-local", "--part", "gate", "--gpus", "5"])
    with pytest.raises(SystemExit):
        M._check_gpus(a)                                              # 배정되지 않은 GPU
    monkeypatch.setattr(LC, "free_gpus", lambda ids, max_mib=0.0: [])
    a = M.parse_args(["--allow-local", "--part", "gate", "--gpus", "2"])
    with pytest.raises(SystemExit):
        M._check_gpus(a)                                              # 사용 중(0 MiB 아님)
    monkeypatch.setattr(LC, "free_gpus", lambda ids, max_mib=0.0: [2])
    assert M._check_gpus(M.parse_args(["--allow-local", "--part", "gate", "--gpus", "2,1"])) == ["2"]
    monkeypatch.setattr(M.os, "nice", lambda inc: 10)
    a = M.parse_args(["--allow-local", "--workers", "4", "--threads", "4"])
    with pytest.raises(SystemExit):                                   # 개정 1: --threads 를 낮추지 않고 거부한다
        M._local_limits(a)
    assert a.threads == 4
    assert M._local_limits(M.parse_args(["--allow-local", "--workers", "2", "--threads", "4"])) == 8
    a = M.parse_args(["--allow-local", "--part", "flow", "--gpus", "2", "--threads", "8"])
    with pytest.raises(SystemExit):                                   # 자동 집계 풀(워커 2 × 8)까지 센다
        M._local_limits(a)
    assert M._local_limits(M.parse_args(["--allow-local", "--part", "flow", "--gpus", "2", "--threads", "8", "--no-summarize"])) == 8
    monkeypatch.setenv("LG_RESCALE", "1")                             # 개정 1: LG_RESCALE=1 도 GPU 허용 목록과 0 MiB 확인을 거친다
    with pytest.raises(SystemExit):
        M._check_gpus(M.parse_args(["--part", "gate", "--gpus", "5"]))
    with pytest.raises(SystemExit):
        M._local_limits(M.parse_args(["--workers", "3", "--threads", "4"]))


# ---------------------------------------------------------------- (m)
def _gate_store(k, rng, d_v, seeds):
    nb, per = 10, 12
    blk = np.repeat(np.arange(nb), per)
    st = LC.ScoreStore(f"T~gate:{k}", 0, blk.astype(str), blk, meta=dict(heldout=k))
    base = 0.3 + 0.05 * rng.rand(nb * per)
    st.add(("iv", "blk", 0, 0, 0), dict(crps_log=base, valid=np.ones(len(base), bool)))
    for s in seeds:
        st.add(("v", "blk", 0, 0, s), dict(crps_log=base + d_v + 0.002 * rng.randn(len(base)), valid=np.ones(len(base), bool)))
    for var in ("res", "n0"):
        st.add(("iv", var, 0, 0, 0), dict(crps_log=base, valid=np.ones(len(base), bool)))
        for s in seeds:
            st.add(("v", var, 0, 0, s), dict(crps_log=base, valid=np.ones(len(base), bool)))
    return st


def test_m_gate_decide():
    rng = np.random.RandomState(5)
    good = [_gate_store(k, rng, -0.05, [0, 1, 2]) for k in ("A", "B", "C")]
    out = M.gate_decide(good, 200, "T", "x", [0, 1, 2])
    assert out["gate_pass"] and out["gate_state"] == "통과" and out["n_valid_seeds"] == 3 and out["blk"]["ci_hi"] < 0
    one_seed = [_gate_store(k, rng, -0.05, [0]) for k in ("A", "B", "C")]
    o1 = M.gate_decide(one_seed, 200, "T", "x", [0, 1, 2])
    assert not o1["gate_pass"] and o1["gate_state"] == "판정 불가(적합 실패)"       # 개정 1: 미통과로 세지 않는다
    worse = [_gate_store(k, rng, 0.05, [0, 1, 2]) for k in ("A", "B", "C")]
    ow = M.gate_decide(worse, 200, "T", "x", [0, 1, 2])
    assert not ow["gate_pass"] and ow["gate_state"] == "미통과"
    fp = M.gate_decide(worse, 200, "T", "x", [0, 1, 2], force="pass")
    assert fp["gate_pass"] and fp["gate_state"] == "미통과" and fp["forced"] == "pass"
    no_iv = [_gate_store(k, rng, -0.05, [0, 1, 2]) for k in ("A", "B", "C")]
    no_iv[1] = LC.ScoreStore("T~gate:B", 0, no_iv[1].blocks[no_iv[1].codes], None, meta=dict(heldout="B"))   # 단 (iv) 키가 없는 지역
    for s_ in (0, 1, 2):
        no_iv[1].add(("v", "blk", 0, 0, s_), dict(crps_log=np.full(len(no_iv[1].codes), 0.3), valid=np.ones(len(no_iv[1].codes), bool)))
    assert M.gate_decide(no_iv, 200, "T", "x", [0, 1, 2])["gate_state"] == "판정 불가(적합 실패)"
    small = []
    for k in ("A", "B"):
        blk = np.repeat(np.arange(5), 6)
        st = LC.ScoreStore(f"T~gate:{k}", 0, blk.astype(str), blk, meta=dict(heldout=k))
        st.add(("iv", "blk", 0, 0, 0), dict(crps_log=np.full(30, 0.3), valid=np.ones(30, bool)))
        small.append(st)
    assert M.gate_decide(small, 200, "T", "x", [0, 1, 2])["gate_state"] == "판정 불가(지역 부족)"


# ---------------------------------------------------------------- (n)
def test_n_tau_component_is_bin_mass():
    from scipy import integrate
    comp = M._tau_comp(4.0)
    assert comp.shape == (len(M.TAU_R), len(M.A_GRID))
    rs = comp.sum(axis=1)
    assert np.all(rs <= 1 + 1e-12) and abs(rs[0] - 1) < 1e-6                 # 작은 τ 는 거의 모든 질량이 격자 안(t_4 꼬리 약 4e-9)
    j = int(np.argmin(np.abs(M.TAU_R - 2.0)))
    assert abs(rs[j] - (M.t_cdf(2.0025 / M.TAU_R[j], 4.0) - M.t_cdf(-2.0025 / M.TAU_R[j], 4.0))) < 1e-9   # 다시 정규화하지 않는다
    # τ_r 사후의 적분 ∫ t_4(a; 0, τ)·N(ā; a, v) da 를 수치 적분과 대조한다
    abar, v = 0.3, 0.002
    dens = M._bin_mass(M.A_GRID, [abar], [v])[:, 0] / M.HG
    for tau in (0.05, 0.3, 1.0, 2.0):
        i = int(np.argmin(np.abs(M.TAU_R - tau)))
        tt = float(M.TAU_R[i])
        ref, _ = integrate.quad(lambda x: np.exp(M.t_logpdf(x / tt, 4.0)) / tt * np.exp(-(abar - x) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v),
                                abar - 1.0, abar + 1.0, points=[abar], limit=200)
        assert abs(float(comp[i] @ dens) / ref - 1) < 5e-3, (tau, float(comp[i] @ dens), ref)
    pa = M.prior_a(*M.tau_r_posterior([0.2, -0.1, 0.05], [0.002] * 3))
    assert abs(pa.sum() - 1) < 1e-12 and np.all(pa >= 0)


# ---------------------------------------------------------------- (o)
def test_o_gate_scale_is_out_of_sample(synth, monkeypatch):
    cfg = _cfg(CB_SEEDS=[0], FLOW_SEEDS=[], RUNGS=["P0", "P1", "B4", "i", "ii", "iii", "iv"], SENS=[])
    emu = M.emu_compute(synth.fr, synth.t_idx, synth.src_idx, "Lena", "x", cfg, do_iv=True)
    ev = M.EmuView(emu, synth.fr)
    assert ev.iv(0) is not None
    sizes = []
    orig = M.cb_fit

    def spy(X, y, w, seed, threads):
        sizes.append(len(y))
        return orig(X, y, w, seed, threads)
    monkeypatch.setattr(M, "cb_fit", spy)
    stores, rows, info, fails, n_fit = M.gate_compute(synth.fr, ev, "Lena", "x", cfg)
    K = len(ev.groups)
    assert not fails and n_fit["catboost"] == K * K == len(sizes)
    n_reg = {k: int((np.asarray(ev.a["g_reg"]).astype(str) == k).sum()) for k in ev.groups}
    for i, k in enumerate(ev.groups):                                   # 뺀 지역 k 마다 전체 적합 1건 뒤 {k, q} 제외 적합 K − 1건
        blk = sizes[i * K:(i + 1) * K]
        n_tr = sum(v for kk, v in n_reg.items() if kk != k)
        assert blk[0] <= n_tr and all(b_ < blk[0] for b_ in blk[1:])
    r = pd.DataFrame(rows)
    assert (r.scale_c_method == "oos_leave_one_region").all() and (r.scale_c > 0).all() and (r.scale_c_regions == K - 1).all()
    dec = M.gate_decide(stores, 50, "Lena", "x", cfg.FLOW_SEEDS)
    assert dec["gate_state"] == "판정 불가(적합 실패)" and not dec["gate_pass"]    # 흐름 seed 가 없으면 판정 불가


# ---------------------------------------------------------------- (p)
def test_p_flow_requires_same_gate(tmp_path):
    a = M.parse_args(["--allow-local", "--part", "flow", "--out-dir", str(tmp_path)])
    pg, pf = M.shard_paths(a, "gate", "Canada", "x"), M.shard_paths(a, "flow", "Canada", "x")
    LC.atomic_text(pg["unit"], json.dumps(dict(status="ok", gate_pass=False, gate_state="미통과")))
    LC.atomic_text(pf["unit"], json.dumps(dict(status="skipped_gate", cfg_hash=LC.cfg_hash(M.unit_cfg(a, "flow")),
                                               gate_ref=M.gate_ref(a, "Canada", "x"))))
    assert M.unit_done(a, "flow", "Canada", "x")[0]
    LC.atomic_text(pg["unit"], json.dumps(dict(status="ok", gate_pass=True, gate_state="통과")))      # 게이트를 다시 실행했다
    ok, why = M.unit_done(a, "flow", "Canada", "x")
    assert not ok and "gate_ref" in why
    assert M.gate_state(a, "Canada", "x")[0] is None                   # 게이트 조각(runs, scores)이 없으면 흐름은 실행하지 않고 기다린다


# ---------------------------------------------------------------- (k)
@HEAVY
def test_k_smoke_paths(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    env.pop("LG_RESCALE", None)
    base = [sys.executable, str(SCRIPT), "--smoke", "--allow-local", "--threads", "2", "--workers", "0", "--out-dir", str(tmp_path)]
    for extra in (["--part", "cpu", "--no-summarize"], ["--part", "gate", "--no-summarize"], ["--part", "flow"]):
        r = subprocess.run(base + extra, env=env, capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
        assert r.returncode == 0, (extra, r.stdout[-4000:], r.stderr[-4000:])
    for stem in ("tests", "curve", "hyper", "gate", "meta"):
        ext = "json" if stem == "meta" else "csv"
        assert (tmp_path / f"lgu_a_smoke_{stem}.{ext}").exists(), stem
