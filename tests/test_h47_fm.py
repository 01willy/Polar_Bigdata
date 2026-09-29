"""scripts/3_deep_learning/h47_foundation_models.py 단위 시험(계획 docs/EXPERIMENT_PLAN_LGF_2026-09-29.md §3, 개정 1).

가벼운 시험은 학습을 하지 않는다. 두 학습기 자리에 대체 함수(학습 목표의 평균 + 입력 1열의 선형 항)를 넣어 컨텍스트 행렬, 목표, 저장 경로,
집계 규칙만 확인한다. 실자료와 GPU 를 쓰지 않는다. 학습을 하는 시험(CatBoost 실제 적합, TabICL 실제 적합)은 환경 변수 LG_RUN_HEAVY=1 일 때만
돈다(TabICL 시험은 CUDA 와 가중치 파일이 있을 때만). 스레드 환경 변수는 numpy 를 부르기 전에 2 로 둔다.

가벼운 시험
(a) h43 경로(h43.run_ctx_t)와 ctx_sha 가 모든 키에서 같다. 두 학습기가 같은 컨텍스트 행렬과 목표를 받는다.
(b) main 의 n = 0 에서 R1 은 R0 적합을 다시 쓴다(SSE 동일, 적합 없음). full 의 n = 0 R1@full 은 E0 앵커로 적합한다.
(c) 부분 n0 과 pos 의 저장 키 합집합이 부분 all 의 키 집합과 같고 값도 같다.
(d) full 컨텍스트는 원천 전부(ctx_order 순서)와 대상 라벨 행이다. main 은 상한을 넘지 않는다.
(e) LGT 짝 게이트의 다섯 조건(가짜 LGT 조각): 모두 맞으면 통과하고 tabpfn 키가 병합된다. 하나씩 어기면 통과하지 않는다.
(f) F1 판정 함수: 두 대비의 모든 조합(우세, 열세, 동등, 미결정, 판정 불가, 행 없음)과 지지 두 갈래.
(g) 정밀도 규칙: 반폭, precision_ok, 동등성 문구의 '정밀도 미달' 두 형식.
(h) Holm 묶음 구성(conf 2, aux 27, eq 17)과 lgf_tests.csv 의 열, 확인적 표지, 합성 저장소의 판정 문구.
(i) L1 형식 지역 수: 두 지역이 n = 10 에서 우세이면 예외 문구가 붙는다.
(j) 창 파일의 생성(t0 는 한 번만), 마감, 드레인, 배정 판단, 결정 기록(--window-set, --restore, --gpu-event), 본 실행의 축소 대조.
(k) GPU 거부 규칙(8, 허용 표지 없는 0·1, 후보 밖, 빈 --gpus, --allow-local 없음)과 점유 판정(메모리, 계산 프로세스, LGT 잠금).
(l) h43·h40 판 고정: SHA 불일치 거부, --h43-sha 로 받아들이면 '개정 필요' 표지.
(m) 패키지 판 확인. (n) 작업 명세와 축소(S2, S3, S4), f_t3 의 15대상. (o) 학습 없는 계수의 적합 수 = 대체 학습기 실행의 적합 수.
(p) 실행 경로(GPU 대체): 본 실행 조각 네 파일과 재개, 스모크의 산출 제한(BlockStore·cells 없음, RMSE NaN)과 경로 점검 기록.
(q) 종료 코드. (r) 결측 서술 표의 부분집합과 30셀 규칙. (s) 스모크 점검 파일의 원래 클래스 강제.
무거운 시험(LG_RUN_HEAVY=1)
(H1) catboost_ctx 실제 적합: 예측이 유한하고 Pool 경로와 h40.cb_fit 경로가 같다.
(H2) TabICL 실제 적합(CUDA 와 가중치가 있을 때): DataFrame(NaN 포함) 입력, 캐시 하위 클래스와 원래 클래스의 예측 차 ≤ 1e-6.
실행: PY=.venv_lgf/bin/python; $PY -m pytest -q tests/test_h47_fm.py
      LG_RUN_HEAVY=1 CUDA_VISIBLE_DEVICES=<빈 GPU> $PY -m pytest -q tests/test_h47_fm.py -k H
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
HEAVY = os.environ.get("LG_RUN_HEAVY", "") == "1"
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
HEAVY_GPU = HEAVY and _CVD.isdigit() and int(_CVD) in (2, 3, 4, 5, 6, 7, 9)     # 무거운 GPU 시험은 허용 후보 GPU 한 장을 명시했을 때만
if not HEAVY_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                   # 가벼운 시험과 CPU 무거운 시험은 GPU 를 쓰지 않는다
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HEAVY_MARK = pytest.mark.skipif(not HEAVY, reason="학습을 하는 시험은 LG_RUN_HEAVY=1 일 때만 실행한다")
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


M = _load("h47_foundation_models", "scripts/3_deep_learning/h47_foundation_models.py")
T43 = M.T43
H = M.H
D_FEAT = len(H.FEATS)
NS, NA, NB = 400, 60, 50
CMAX, CRES, CMIN = 300, 50, 50
BASE_ARGS = ["--n-grid", "0,3,10,40,all", "--draws", "2", "--seeds", "2", "--splits", "1", "--threads", "2", "--ctx-max", str(CMAX),
             "--ctx-reserve", str(CRES), "--ctx-src-min", str(CMIN), "--model-path", "/nonexistent/lgf_test_weights.ckpt"]
NSA = SimpleNamespace(nboot=200, delta_eq=0.5, delta_eq_aux=1.0, LOCAL_SUMMARY=False, INTERIM=False)


def make_unit(seed=0, xb_nan=0):
    """합성 작업 단위(h43 시험과 같은 구성). 특징 0 열은 행 식별자(원천 = −1−j, 대상 A = j, 채점 B = 1000 + j)."""
    rng = np.random.RandomState(seed)
    Xs = rng.randn(NS, D_FEAT); Xs[:, 0] = -1.0 - np.arange(NS)
    ss = rng.uniform(20, 40, NS); ys = 1.6 * ss + 3 * Xs[:, 1] + rng.randn(NS) * 2
    XA = rng.randn(NA, D_FEAT); XA[:, 0] = np.arange(NA)
    sA = rng.uniform(20, 40, NA); yA = 1.3 * sA + 3 * XA[:, 1] + rng.randn(NA) * 2
    XB = rng.randn(NB, D_FEAT); XB[:, 0] = 1000.0 + np.arange(NB)
    sB = rng.uniform(20, 40, NB); yB = 1.3 * sB + 3 * XB[:, 1] + rng.randn(NB) * 2
    XA[5, 3] = np.nan; Xs[7, 4] = np.nan
    if xb_nan:
        XB[:xb_nan, 2] = np.nan
    mac = np.array(["r1"] * 200 + ["r2"] * 120 + ["r3"] * 60 + ["r4"] * 20)
    return H.Ctx("T", "x", 1, "T", Xs, ys, ss, mac, XA, yA, sA, np.repeat(np.arange(6), NA // 6), XB, yB, sB, np.repeat(np.arange(5), NB // 5))


def stub_icl(a, X, y, XB, seed, check_chunk=0, check_stock=False):
    y = np.asarray(y, float)
    return (float(y.mean()) + 0.01 * np.nan_to_num(np.asarray(XB, float)[:, 1]) + 0.001 * int(seed), "",
            dict(amp_used=True, fa3_flag=False, has_fa3=False, offload_mode="GPU", impl="cached", check_s=0.0))


def stub_cb(ns, X, y, XB, seed):
    y = np.asarray(y, float)
    return float(y.mean()) + 0.02 * np.nan_to_num(np.asarray(XB, float)[:, 1]) + 0.001 * int(seed), "", {}


@pytest.fixture()
def stub(monkeypatch):
    monkeypatch.setattr(M, "tabicl_fit_predict", stub_icl)
    monkeypatch.setattr(M, "cb_fit_predict", stub_cb)
    return True


def args(extra=(), tmp=None):
    out = ["--out-dir", str(tmp)] if tmp is not None else []
    return M.parse_args(BASE_ARGS + out + list(extra))


def unit_of(a, axis="main", part="all", grid=None, job="t"):
    grid = list(a.N_USER) if grid is None else list(grid)
    return (job, axis, "T", "x", 1, part, tuple(int(n) for n in grid), M.axis_draws(a, axis), int(a.seeds))


def run(axis="main", part="all", grid=None, c=None, extra=(), dry=False, tmp=None):
    a = args(extra, tmp)
    u = unit_of(a, axis, part, grid)
    HA = M.ha_of(a, u)
    M.TRACE = []
    try:
        rows, st, stats, cells = M.run_ctx_f(c if c is not None else make_unit(), axis, part, a, HA, dry=dry)
        return a, HA, rows, st, stats, cells, list(M.TRACE or [])
    finally:
        M.TRACE = None


# ================================================================ (a) h43 경로와의 ctx_sha 일치
def test_a_ctx_sha_matches_h43_path(stub):
    a, HA, rows, st, stats, cells, tr = run("main")
    ref = M.h43_ctx_shas(make_unit(), a, a.N_USER, M.axis_draws(a, "main"), a.seeds)
    n_cmp, n_bad = M.sha_check(rows, ref)
    mine = {(r["method"], r["n"], r["draw"], r["seed"]) for r in rows if r["learner"] == M.TI}
    assert n_cmp == len(ref) == len(mine) and n_bad == 0, (n_cmp, n_bad, len(ref), len(mine))
    assert T43.run_ctx_t is not None and T43.tabpfn_fit_predict.__name__ == "tabpfn_fit_predict", "h43 의 학습기 함수는 되돌려 둔다"
    by = {}
    for t in tr:
        by.setdefault((t["method"], t["n"], t["draw"], t["seed"]), []).append(t)
    for k, ts in by.items():
        assert sorted(q["learner"] for q in ts) == sorted(M.LEARNERS_F), k
        assert np.array_equal(ts[0]["X"], ts[1]["X"], equal_nan=True) and np.array_equal(ts[0]["y"], ts[1]["y"]), k
        assert len(ts[0]["X"]) <= CMAX
    assert stats["status"] == "ok" and stats["failed_learners"] == []


# ================================================================ (b) n = 0 의 R1
def test_b_n0_reuse_main_and_full_r1_anchor(stub):
    c = make_unit()
    a, HA, rows, st, stats, cells, tr = run("main", "n0", grid=[0], c=c)
    assert sorted({t["method"] for t in tr}) == ["D0", "R0"] and len(tr) == 2 * 2 * 2, "main 의 n = 0 은 D0, R0 만 적합한다"
    n_r1 = 0
    for lr in M.LEARNERS_F:
        for s_ in (0, 1):
            for lam in (0.25, 0.5, 1.0):
                k1, k0 = ("R1", lr, "1", "cell", 0, 0, s_, lam), ("R0", lr, "1", "cell", 0, 0, s_, lam)
                assert k1 in st and k0 in st and np.array_equal(st.get(k1)[0], st.get(k0)[0])
                n_r1 += 1
    assert n_r1 == 12 and all(r["fit_s"] == 0.0 for r in rows if r["method"] == "R1")
    a, HA, rows, st, stats, cells, tr = run("full", "n0", grid=[0], c=c)
    assert sorted({t["method"] for t in tr}) == ["D0@full", "R1@full"] and len(tr) == 2 * 2 * 2
    for t in tr:
        assert len(t["src"]) == len(c.y_src) and len(t["tsel"]) == 0
        want = c.y_src[t["src"]] if t["method"] == "D0@full" else c.r0_src[t["src"]]
        assert np.allclose(t["y"], want), "R1@full 의 n = 0 은 E0 앵커의 원천 잔차를 학습한다"
    assert all(r["E_used"] == pytest.approx(c.E0) for r in rows if r["method"] == "R1@full")


# ================================================================ (c) 부분 합 = 전체
def test_c_parts_union_equals_all(stub):
    c = make_unit()
    _, _, _, s0, *_ = run("main", "n0", grid=[0], c=c)
    _, _, _, sp_, *_ = run("main", "pos", grid=[3, 10, 40, -1], c=c)
    _, _, _, sa, *_ = run("main", "all", grid=[0, 3, 10, 40, -1], c=c)
    assert set(s0.keys) | set(sp_.keys) == set(sa.keys) and M.H.P0_KEY in s0 and M.H.P0_KEY in sp_
    mg = s0.merge(sp_)
    for k in sa.keys:
        assert np.allclose(mg.get(k)[0], sa.get(k)[0]) and np.array_equal(mg.get(k)[1], sa.get(k)[1]), k


# ================================================================ (d) full 컨텍스트
def test_d_full_context_is_all_source(stub):
    c = make_unit()
    a, HA, rows, st, stats, cells, tr = run("full", "all", grid=[0, 10, 40, -1], c=c)
    assert tr and stats["n_ctx_max"] > CMAX, "full 은 상한이 없다"
    for t in tr:
        order = T43.ctx_order(c.macro_src, c.target, c.mode, c.split, t["seed"])
        assert np.array_equal(t["src"], order), "원천 전부를 ctx_order 순서로 넣는다"
        ids = t["X"][:, 0]
        assert np.array_equal(ids[:len(order)], c.X_src[order, 0]) and np.array_equal(ids[len(order):], np.asarray(t["tsel"], float))
        assert len(t["X"]) == len(c.y_src) + len(t["tsel"])
    ref = M.full_ctx_shas(c, a, [0, 10, 40, -1], M.axis_draws(a, "full"), a.seeds, HA.kappa)
    n_cmp, n_bad = M.sha_check(rows, ref)
    assert n_cmp == len(ref) > 0 and n_bad == 0
    _, _, _, _, st2, _, tr2 = run("main", "all", c=c)
    assert all(len(t["X"]) <= CMAX for t in tr2) and st2["n_ctx_max"] <= CMAX


# ================================================================ (e) LGT 짝 게이트
def _gate_fixture(tmp_path, *, lgt_status="ok", ctx_max=10000, p0_shift=0.0, sha_bad=False, blocks_shift=False):
    rng = np.random.RandomState(3)
    blocks = np.repeat(np.arange(6), 5)
    y = rng.randn(len(blocks)) * 5 + 50
    st = M.BlockStore("T|x", 1, blocks)
    stT = M.BlockStore("T|x", 1, blocks + (100 if blocks_shift else 0))
    st.add(M.H.P0_KEY, y, y + rng.randn(len(y)))
    stT.add_sse(M.H.P0_KEY, st.get(M.H.P0_KEY)[0] + p0_shift, st.get(M.H.P0_KEY)[1])
    kP1 = ("P1", "none", "1", "cell", 10, 0, -1, 0.0)
    st.add(kP1, y, y + rng.randn(len(y))); stT.add_sse(kP1, *st.get(kP1))
    runs_l, runs_t = [], []
    for s_ in (0, 1):
        for mth in ("D0", "R0", "R1"):
            kI, kC, kT = [(mth, lr, "1", "cell", 10, 0, s_, 1.0) for lr in (M.TI, M.CBX, M.TP)]
            st.add(kI, y, y + rng.randn(len(y))); st.add(kC, y, y + rng.randn(len(y)))
            stT.add(kT, y, y + rng.randn(len(y))); stT.add_sse(kC, *st.get(kC))
            sha = f"{mth}{s_}sha"
            runs_l.append(dict(target="T", mode="x", split=1, method=mth, learner=M.TI, n=10, draw=0, seed=s_, ctx_set="main", ctx_sha=sha))
            runs_t.append(dict(method=mth, learner=M.TP, alpha="1", n=10, draw=0, seed=s_, ctx_set="main",
                               ctx_sha=("x" + sha) if (sha_bad and mth == "R1" and s_ == 1) else sha, rmse_cm=0.0))
    d = tmp_path / "lgt" / "shards"
    d.mkdir(parents=True, exist_ok=True)
    b = d / "lgt__gpu__T__x__s1"
    M.save_stores([stT], Path(str(b) + "_blocksse.npz"))
    pd.DataFrame(runs_t).to_csv(str(b) + "_runs.csv", index=False)
    cfg_t = dict(axis="main", ctx_max=ctx_max, ctx_reserve=1000, ctx_src_min=1000, kappa=10.0, seeds=[0, 1], draws=5,
                 n_grid=[0, 3, 10, 40, 160, 320, 1000, -1], subregion_map="lg_subregion_map_v1.csv")
    Path(str(b) + "_unit.json").write_text(json.dumps(dict(status=lgt_status, cfg=cfg_t, threads=4, code_sha="13148bb899e9")))
    sh = [dict(target="T", mode="x", split=1, part="gpu", unit=Path(str(b) + "_unit.json"), runs=Path(str(b) + "_runs.csv"),
               npz=Path(str(b) + "_blocksse.npz"), cells=Path(str(b) + "_cells.npz"))]
    cfg_l = dict(axis="main", ctx_max=10000, ctx_reserve=1000, ctx_src_min=1000, dup=1, kappa=10.0, seeds=[0, 1], draws=5, n_grid=[3, 10, 40],
                 n_grid_axis=[0, 3, 10, 40, 160, 320, 1000, -1], subregion_map="lg_subregion_map_v1.csv")
    units = [dict(target="T", mode="x", split=1, axis="main", cfg=cfg_l, threads=2, code_sha_h43="13148bb899e92c49")]
    return {("T|x", 1): st}, pd.DataFrame(runs_l), units, sh


def test_e_gate_five_conditions(tmp_path):
    a = args([], tmp_path / "lgf")
    X = T43._load_h42()
    stores, runs, units, sh = _gate_fixture(tmp_path)
    g, out = M.gate_lgt(a, X, stores, runs, units, lgt_shards=sh)
    r = g.iloc[0]
    assert bool(r.passed) and r.status == "ok" and int(r.n_tabpfn_merged) == 6 and int(r.n_sha_keys) == 6 and int(r.n_sha_mismatch) == 0
    assert r.cfg_ok and r.cfg_diff == "" and r.code_sha_warn == "" and int(r.cb_n_keys) == 6 and float(r.cb_max_diff) == pytest.approx(0.0)
    assert any(k[1] == M.TP for k in out[("T|x", 1)].keys) and not any(k[1] == M.TP for k in stores[("T|x", 1)].keys), "원본 저장소는 그대로 둔다"
    cases = dict(status=dict(lgt_status="failed"), cfg=dict(ctx_max=9000), p01=dict(p0_shift=1e-3), sha=dict(sha_bad=True), blocks=dict(blocks_shift=True))
    for nm, kw in cases.items():
        stores, runs, units, sh = _gate_fixture(tmp_path / nm, **kw)
        g, out = M.gate_lgt(a, X, stores, runs, units, lgt_shards=sh)
        r = g.iloc[0]
        assert not bool(r.passed) and int(r.n_tabpfn_merged) == 0 and not any(k[1] == M.TP for k in out[("T|x", 1)].keys), nm
        want = dict(status="LGT 상태", cfg="설정 불일치", p01="P0·P1 불일치", sha="ctx_sha 불일치", blocks="채점 블록 불일치")[nm]
        assert want in r.status, (nm, r.status)
    assert "ctx_max" in M.gate_cfg_diff(dict(ctx_max=1), dict(ctx_max=2, axis="main"))[0]
    assert M.gate_cfg_diff(dict(dup=1), dict(axis="main")) == [], "LGT 주 설정에는 dup 이 없고 h43.MAIN_SET 의 1 이다"


# ================================================================ (f) F1 판정 함수
def _row(v, lo=-0.2, hi=0.3):
    if v is None:
        return None
    return dict(verdict4=v, ci_lo=lo, ci_hi=hi, ci_lo_beq=lo, ci_hi_beq=hi, delta=(lo + hi) / 2.0)


def test_f_f1_verdict_all_combinations():
    X = T43._load_h42()
    vals = ["우세", "열세", "동등", "미결정", "판정 불가", None]
    for v1 in vals:
        for v2 in vals:
            txt, sup = M.f1_verdict(X, [("D0[I]-P0", _row(v1)), ("D0@full[I]-P0", _row(v2))])
            valid = [v for v in (v1, v2) if v not in ("판정 불가", None)]
            if "우세" in (v1, v2):
                assert txt.startswith("기각") and sup == "", (v1, v2, txt)
                if len(valid) < 2:
                    assert txt.startswith("기각(일부 대비 판정 불가, 대비 1/2)"), txt
            elif len(valid) < 2:
                assert txt.startswith("판정 불가") and sup == "", (v1, v2, txt)
            elif set(valid) <= {"열세", "동등"}:
                assert txt.startswith("지지(물리식보다 오차가 크거나 구별되지 않음)") and sup == "A" and "두 조건" in txt, (v1, v2)
            else:
                assert txt.startswith("지지(우세 근거 없음, ") and sup == "B" and "CI [-0.20, 0.30] cm" in txt, (v1, v2, txt)


# ================================================================ (g) 정밀도 규칙
def test_g_precision_rule():
    X = T43._load_h42()
    assert M.halfwidth(dict(ci_lo=-0.2, ci_hi=0.4, ci_lo_beq=-0.5, ci_hi_beq=0.3)) == pytest.approx(0.4)
    assert M.precision_ok(dict(ci_lo=-0.2, ci_hi=0.4, ci_lo_beq=-0.5, ci_hi_beq=0.3), 0.5)
    assert not M.precision_ok(dict(ci_lo=-0.9, ci_hi=0.4, ci_lo_beq=-0.5, ci_hi_beq=0.3), 0.5)
    assert np.isnan(M.halfwidth(dict(ci_lo=np.nan, ci_hi=0.4, ci_lo_beq=-0.5, ci_hi_beq=0.3)))
    wide, narrow, eq = _row("미결정", -1.2, 0.9), _row("미결정", -0.6, 0.1), _row("동등", -0.3, 0.3)
    base = [(f"n={n}", eq) for n in (0, 10, 40, 160)] + [("전량", wide)]
    lam = [(f"n={n} λ=1.0", eq) for n in (0, 10, 40, 160, -1)]
    txt, used, part = M.equiv_text(X, base, lam, 0.5, "시험")
    assert txt == "정밀도 미달(동등성 판정 불가, 대비 1/10)", txt
    base2 = [(f"n={n}", eq) for n in (0, 10, 40)] + [("n=160", narrow), ("전량", wide)]
    txt2, *_ = M.equiv_text(X, base2, lam, 0.5, "시험")
    assert txt2 == "차이를 확인하지 못함(정밀도 미달 대비 1/10)", txt2
    txt3, *_ = M.equiv_text(X, [(f"n={n}", eq) for n in (0, 10, 40, 160, -1)], lam, 0.5, "시험")
    assert txt3.startswith("동등(한계 0.5 cm, λ 0.25 와 1.0)") and "정밀도" not in txt3
    df = M.finish_tests(NSA, pd.DataFrame([dict(test_id="LGF-F2", scope="MEAN", verdict4="미결정", ci_lo=-1.2, ci_hi=0.9, ci_lo_beq=-1.0, ci_hi_beq=0.8,
                                                holm_family="aux", eq_test=True, p_boot=0.5, p_eq=0.4, verdict="")]))
    r = df.iloc[0]
    assert r.halfwidth == pytest.approx(1.05) and r.precision_ok == False and r.precision_note == "정밀도 미달(동등성 판정 불가)"  # noqa: E712


# ================================================================ (h)(i) 집계: 합성 저장소
def K(method, n, lam, lr="none", draw=0, seed=None):
    return (method, lr, "1", "cell", int(n), int(draw), (-1 if lr == "none" else 0) if seed is None else int(seed), float(lam))


def _mk_tm(name, nb, seed, specs, splits=(1, 2), nboot=200, p0_sd=6.0):
    rng = np.random.RandomState(seed)
    by = {}
    for sp in splits:
        blocks = np.repeat(np.arange(nb), 5)
        st = H.BlockStore(name, sp, blocks)
        y = rng.randn(len(blocks)) * 5 + 50
        noise = {"p0": rng.randn(len(y))}
        st.add(H.P0_KEY, y, y + p0_sd * noise["p0"])
        for k, (sd, grp) in specs.items():
            if grp not in noise:
                noise[grp] = rng.randn(len(y))
            st.add(k, y, y + float(sd) * noise[grp])
        by[sp] = st
    tm = H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in splits}, nboot)
    tm.base_target = tm.target
    return tm


def _specs(d0_better_n10=False):
    sp = {}
    for n in (0, 3, 10, 40, 160, 320, 1000, -1):
        sp[K("P1", n, 0.0)] = (3.0, f"p1{n}")
        for lr in (M.TI, M.CBX, M.TP):
            for lam in (0.25, 0.5, 1.0):
                sp[K("R1", n, lam, lr)] = (1.0, f"r{n}")
                sp[K("R0", n, lam, lr)] = (1.0, f"r{n}")
            sp[K("D0", n, 1.0, lr)] = (9.0, "p0")
        for lr in (M.TI, M.CBX):
            for lam in (0.25, 0.5, 1.0):
                sp[K("R1@full", n, lam, lr)] = (1.0, f"r{n}")
            sp[K("D0@full", n, 1.0, lr)] = (9.0, "p0")
    if d0_better_n10:
        sp[K("D0", 10, 1.0, M.TI)] = (0.3, "d10own")
    return sp


def _tms(better=()):
    names = [f"{t}|x" for t in H.MAIN4] + [f"{H.ALASKA}|x"]
    return {nm: _mk_tm(nm, 10, 100 + i, _specs(d0_better_n10=nm in better)) for i, nm in enumerate(names)}


def _verdicts(tests, tid, **where):
    v = tests[(tests.test_id == tid) & tests.scope.isin(["verdict", "verdict_aux"])]
    for k, val in where.items():
        v = v[v[k] == val]
    return [str(s_) for s_ in v.verdict_text]


def test_h_holm_families_columns_and_verdicts():
    tests = M.build_tests_f(NSA, _tms(), None, None)
    assert [c for c in M.TEST_COLS if c not in tests.columns] == []
    mean = tests[tests.scope == "MEAN"]
    assert (mean.holm_family == "conf").sum() == 2 and (mean.holm_family == "aux").sum() == 27 and (mean.holm_family_eq == "eq").sum() == 17
    assert set(mean[mean.holm_family == "conf"].contrast) == {"D0[I]-P0|n0", "D0@full[I]-P0|n0"}
    assert set(mean[mean.holm_family_eq == "eq"].test_id) == {"LGF-F2", "LGF-F3", "LGF-F6"}
    assert (tests[tests.scope != "MEAN"].holm_family == "").all()
    assert mean[mean.holm_family != ""].holm_p.notna().all() and mean[mean.holm_family_eq == "eq"].holm_p_eq.notna().all()
    conf = tests[tests.confirmatory.astype(bool)]
    assert set(conf.test_id) == {"LGF-F1"} and set(conf.scope) == {"MEAN", "verdict"} and len(conf) == 3
    assert set(tests.platform) == {M.PLATFORM}
    v1 = tests[(tests.test_id == "LGF-F1") & (tests.scope == "verdict")]
    assert len(v1) == 1 and v1.support_class.iloc[0] == "A" and v1.verdict_text.iloc[0].startswith("지지(물리식보다 오차가 크거나 구별되지 않음)")
    assert _verdicts(tests, "LGF-F2")[0].startswith("동등(한계 0.5 cm, λ 0.25 와 1.0)")
    assert _verdicts(tests, "LGF-F3")[0].startswith("TabICL v2 와 TabPFN v2 의 차이는 한계 0.5 cm 안이다")
    v4 = _verdicts(tests, "LGF-F4")
    assert "희소 라벨에서도 TabICL 잔차의 순가치가 있다" in v4[0] and "라벨 전량에서 TabICL 잔차는 원천 계수 물리식을 넘는다" in v4[1]
    assert _verdicts(tests, "LGF-F5") == ["라벨 0 에서 TabICL 잔차는 원천 계수 물리식보다 오차가 작다"]
    assert _verdicts(tests, "LGF-F6")[0].startswith("컨텍스트 상한 10,000행의 영향은 한계 0.5 cm 안이다(TabICL)")
    cb = tests[(tests.test_id == "LGF-F1") & (tests.role == "병기(C)") & (tests.scope == "MEAN")]
    assert len(cb) == 2 and not cb.blind.astype(bool).any() and set(cb.prior_info) == {"재현(비맹검, M1·a2)"}
    f2 = mean[(mean.test_id == "LGF-F2") & (mean.holm_family == "aux")]
    assert set(f2.key_set) == {"pair_IC"} and (f2.n_unpaired == 0).all()
    assert set(mean[mean.test_id == "LGF-F6"].key_set) <= {"d01", "pair_IC"}
    assert not any("LGF-family" == t for t in tests.test_id), "lgt_tests.csv 가 없으면 계열 문장 표지를 쓰지 않는다"
    t0 = M.build_tests_f(NSA, {}, None, None)
    assert all(v.startswith("판정 불가") for v in _verdicts(t0, "LGF-F1", clause="판정"))


def test_i_l1_region_count():
    tests = M.build_tests_f(NSA, _tms(better=("Lena|x", "Canada|x")), None, None)
    vb = _verdicts(tests, "LGF-F1", clause="(b) L1 형식 지역 수")
    assert len(vb) == 1 and vb[0].startswith("L1 형식 지역 수 2/4") and "TabICL v2 는 n ≤ 40 에서 예외(Lena(10), Canada(10))" in vb[0], vb
    va = _verdicts(tests, "LGF-F1", clause="(a) n ∈ {3, 10, 40}")
    assert len(va) == 1 and va[0].startswith("보조 행 (a)"), va
    tests1 = M.build_tests_f(NSA, _tms(better=("Lena|x",)), None, None)
    vb1 = _verdicts(tests1, "LGF-F1", clause="(b) L1 형식 지역 수")
    assert vb1[0].startswith("L1 형식 지역 수 1/4") and "예외" not in vb1[0]


# ================================================================ (j) 창 파일
def test_j_window_file_and_assignment(tmp_path):
    a = args([], tmp_path / "w1")
    a.MAIN_MODE = True
    w = M.window_start(a, now=1000.0)
    t0 = M._to_ts(w["t0"])
    assert t0 == pytest.approx(1000.0) and M._to_ts(w["deadline"]) == pytest.approx(1000.0 + 48 * 3600)
    w2 = M.window_start(a, now=9000.0)
    assert w2["t0"] == w["t0"], "t0 는 처음 한 번만 쓴다"
    assert not M.window_deadline_passed(w2, now=1000.0 + 47.9 * 3600) and M.window_deadline_passed(w2, now=1000.0 + 48 * 3600)
    st = {}
    assert M.assign_gate(a, st, now=1000.0 + 49 * 3600, load_fn=lambda: (1.0,)) == "window"
    (a.RUN_DIR).mkdir(parents=True, exist_ok=True)
    (a.RUN_DIR / "drain").write_text("")
    assert M.assign_gate(a, {}, now=2000.0, load_fn=lambda: (1.0,)) == "drain" and M.drain_requested(a)
    M.clear_drain(a)
    assert not M.drain_requested(a)
    b = args([], tmp_path / "w2")
    b.MAIN_MODE = False
    st = {}
    assert M.assign_gate(b, st, now=10.0, load_fn=lambda: (200.0,)) == "load"
    assert M.assign_gate(b, st, now=20.0, load_fn=lambda: (1.0,)) == "load", "load average 는 300 s 마다 본다"
    assert M.assign_gate(b, st, now=10.0 + M.LOAD_CHECK_S, load_fn=lambda: (1.0,)) == "ok"
    assert not M.window_path(b).exists(), "본 실행이 아니면 창을 시작하지 않는다"
    with pytest.raises(SystemExit):
        M.window_tool(args(["--window-set", "--B", "46", "--e1", "0", "--cbt", "1"], tmp_path / "w2"))
    M.window_tool(args(["--window-set", "--B", "46", "--e1", "0", "--cbt", "1", "--reduce", "S1,S2", "--viewing-state", "LG 미회수"], tmp_path / "w2"))
    ww = M.read_window(b)
    assert ww["B"] == 46.0 and ww["E1"] is False and ww["cbt_enabled"] is True and ww["reduce"] == ["S1", "S2"] and "t0" not in ww
    assert M.check_window_for_run(args(["--reduce", "S2"], tmp_path / "w2"))
    for bad in ([], ["--reduce", "S3"], ["--reduce", "S1,S2"]):
        with pytest.raises(SystemExit):
            M.check_window_for_run(args(bad, tmp_path / "w2"))
    M.window_tool(args(["--window-set", "--restore", "S2", "--viewing-state", "LG 미회수"], tmp_path / "w2"))
    assert M.effective_reduce(M.read_window(b)) == ["S1"] and M.check_window_for_run(args([], tmp_path / "w2"))
    with pytest.raises(SystemExit):
        M.window_tool(args(["--window-set", "--restore", "S7", "--viewing-state", "x"], tmp_path / "w2"))
    M.window_tool(args(["--gpu-event", "5:add", "--viewing-state", "LGT 미열람"], tmp_path / "w2"))
    ev = M.read_window(b)["gpu_events"]
    assert ev[-1]["gpu"] == 5 and ev[-1]["event"] == "add" and ev[-1]["viewing_state"] == "LGT 미열람"
    assert len(M.read_window(b)["decisions"]) == 2
    with pytest.raises(SystemExit):
        M.check_window_for_run(args([], tmp_path / "empty"))


# ================================================================ (k) GPU 규칙
def test_k_gpu_refusals_and_screening(tmp_path):
    for g in ("", "8", "0", "1", "11", "5,8", "5,0"):
        with pytest.raises(SystemExit):
            M.check_gpu_list(args(["--gpus", g, "--allow-local"], tmp_path))
    assert M.check_gpu_list(args(["--gpus", "0,5", "--allow-gpus-01", "--allow-local"], tmp_path)) == [5, 0]
    with pytest.raises(SystemExit):
        M.check_gpu_list(args(["--gpus", "8", "--allow-gpus-01"], tmp_path))
    with pytest.raises(SystemExit) as e:
        M.check_run_args(args(["--gpus", "5"], tmp_path))
    assert "--allow-local" in str(e.value)
    with pytest.raises(SystemExit) as e:
        M.check_run_args(args(["--gpus", "5", "--allow-local"], tmp_path))
    assert "가중치" in str(e.value)
    with pytest.raises(SystemExit) as e:
        M.check_run_args(args(["--gpus", "5", "--allow-local", "--threads", "4"], tmp_path))
    assert "--threads" in str(e.value)
    a = args(["--gpus", "9,7,6,5,4", "--allow-local", "--lgt-dir", str(tmp_path / "lgt")], tmp_path)
    info = {g: dict(uuid=f"GPU-{g}", used=u) for g, u in ((4, 13000), (5, 2), (6, 0), (7, 0), (9, 0))}
    ok, dropped = M.screen_now(a, a.GPUS, info=info, apps=[(123, "GPU-9")])
    assert ok == [7, 6, 5] and {g for g, _ in dropped} == {9, 4}
    lk = tmp_path / "lgt" / "run_lgt"
    lk.mkdir(parents=True)
    (lk / "lock.json").write_text(json.dumps(dict(pid=os.getpid())))
    ok, dropped = M.screen_now(a, a.GPUS, info=info, apps=[])
    assert ok == [5] and {g for g, _ in dropped} == {4, 6, 7, 9}, "LGT 잠금의 PID 가 살아 있으면 6·7·9 를 뺀다"
    (lk / "lock.json").write_text(json.dumps(dict(pid=99999999)))
    ok, _ = M.screen_now(a, a.GPUS, info=info, apps=[])
    assert ok == [9, 7, 6, 5]


# ================================================================ (l) 판 고정
def test_l_version_pins(tmp_path):
    a = args([], tmp_path)
    real = T43.file_sha
    fake = lambda p: ("f" * 40) if "h43" in str(p) else real(p)                                  # noqa: E731
    with pytest.raises(SystemExit) as e:
        M.check_pins(a, fake)
    assert "h43" in str(e.value)
    fake40 = lambda p: ("e" * 40) if "h40" in str(p) else real(p)                                # noqa: E731
    with pytest.raises(SystemExit):
        M.check_pins(a, fake40)
    b = args(["--h43-sha", "f" * 40], tmp_path)
    p = M.check_pins(b, fake)
    assert p["revision_needed"] and p["note"] == "개정 필요"
    assert M.check_pins(args([], tmp_path))["revision_needed"] is False, "현재 h43·h40 은 고정 판이다"
    b.PINS = p
    assert M.unit_cfg_f(b, unit_of(b)).get("pin_note") == "개정 필요" and "pin_note" not in M.unit_cfg_f(a, unit_of(a))


# ================================================================ (m) 패키지 판
def test_m_env_check():
    ok = dict(M.PKG_PINS, torch="2.6.0+cu124")
    assert M.check_env(lambda k: ok[k])
    for k in M.PKG_PINS:
        bad = dict(ok, **{k: "0.0.0"})
        with pytest.raises(SystemExit):
            M.check_env(lambda q: bad[q])


# ================================================================ (n) 작업 명세와 축소
def test_n_job_specs_and_reductions(tmp_path):
    full = ["--jobs", ",".join(M.JOBS), "--n-grid", T43.FULL_GRID, "--draws", "0"]
    specs, notes = M.job_specs(args(full, tmp_path))
    got = [(s_["job"], s_["axis"], s_["part"], s_["draws"]) for s_ in specs]
    assert got == [("f_n0", "main", "n0", 5), ("f_n0", "full", "n0", 2), ("f_t1", "main", "pos", 5), ("f_full", "full", "pos", 2),
                   ("f_t2ak", "main", "all", 5), ("f_t2", "main", "all", 5), ("f_t3", "main", "all", 5)] and notes == []
    sp = {s_["job"] + s_["axis"]: s_ for s_ in specs}
    assert sp["f_fullfull"]["grid"] == [10, 40, 160, -1] and sp["f_t1main"]["grid"] == [3, 10, 40, 160, 320, 1000, -1] and sp["f_n0full"]["grid"] == [0]
    assert len(M.T3_TARGETS) == 15 and not set(M.T3_TARGETS) & set(H.LEARNER_TARGETS) and len(M.T2_TARGETS) == 7
    specs, notes = M.job_specs(args(full + ["--reduce", "S3"], tmp_path))
    assert [s_["draws"] for s_ in specs if s_["job"] == "f_t2"] == [2]
    specs, notes = M.job_specs(args(full + ["--reduce", "S2,S4"], tmp_path))
    assert {s_["job"] for s_ in specs} == {"f_n0", "f_t1", "f_t2ak", "f_t3"} and len(notes) == 2
    assert ("f_n0", "full") in {(s_["job"], s_["axis"]) for s_ in specs}, "S2 에서도 n = 0 의 full 은 남긴다"
    specs, _ = M.job_specs(args(["--smoke"], tmp_path))
    assert [(s_["axis"], s_["grid"], s_["draws"]) for s_ in specs] == [("main", [0, 10, 40, -1], 1), ("full", [0, 10, 40, -1], 1)]
    assert [(u[1], u[2], u[4], u[6]) for u in M.precheck_units(args(["--precheck"], tmp_path))] == [
        ("main", "Lena", 1, (0, 40, -1)), ("full", "Lena", 1, (0, 40, -1)), ("main", H.ALASKA, 5, (0, 40, -1)), ("full", "Russia_W", 1, (0,))]
    with pytest.raises(SystemExit):
        args(["--jobs", "f_x"], tmp_path)
    with pytest.raises(SystemExit):
        args(["--reduce", "S9"], tmp_path)
    a1, a2 = args(["--threads", "1"], tmp_path), args(["--threads", "2", "--gpus", "5"], tmp_path)
    assert H.cfg_hash(M.unit_cfg_f(a1, unit_of(a1))) == H.cfg_hash(M.unit_cfg_f(a2, unit_of(a2))), "스레드와 GPU 는 설정 해시에 들지 않는다"
    assert args(["--smoke"], tmp_path).TAG == "lgf_smoke" and M.axis_tag(args(["--precheck"], tmp_path), "full") == "lgfs_pre"


# ================================================================ (o) 학습 없는 계수
def test_o_dry_count_matches_stub_run(stub):
    for axis in ("main", "full"):
        _, _, _, _, s_run, _, _ = run(axis)
        _, _, _, _, s_dry, _, _ = run(axis, dry=True)
        assert s_run["n_fit_detail"] == s_dry["n_fit_detail"] and sum(s_dry["n_fit_detail"].values()) > 0, axis


# ================================================================ (p) 실행 경로(GPU 대체)
def _fake_backend(peak_mib=5.0, resv_mib=8.0):
    cuda = SimpleNamespace(reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: peak_mib * 1024 ** 2,
                           max_memory_reserved=lambda: resv_mib * 1024 ** 2)
    return SimpleNamespace(cuda=cuda)


@pytest.fixture()
def fake_run(monkeypatch, stub):
    c = make_unit()
    D = SimpleNamespace(df=pd.DataFrame(dict(macro=list(c.macro_src) + [c.parent])))
    monkeypatch.setattr(H, "get_data", lambda HA: D)
    monkeypatch.setattr(H, "build_ctx", lambda D_, HA, t, m, sp: c)
    monkeypatch.setattr(M, "cuda_backend", _fake_backend)

    class _CB:
        def __init__(self, n):
            self.n = n

        def predict(self, XB, thread_count=1):
            return np.zeros(len(XB))
    monkeypatch.setattr(H, "cb_fit", lambda ns, X, y, seed: _CB(0))
    monkeypatch.setattr(M, "cb_h40_pred", lambda ns, X, y, XB, seed: np.zeros(len(XB)))
    return c


def test_p_run_unit_paths_and_smoke_limits(fake_run, tmp_path):
    a = args([], tmp_path)
    a.RUN_ID = "rid-1"
    u = unit_of(a, "main", "all", job="f_t1")
    uj = M.run_unit_f(a, u)
    p = M.shard_paths_f(a, u)
    assert all(p[k].exists() for k in ("runs", "npz", "cells", "unit")) and uj["status"] == "ok" and uj["stored"]
    assert uj["gpu_mem_peak_mib"] == 5.0 and uj["run_id"] == "rid-1" and uj["job"] == "f_t1" and uj["pins"]["h43"] == M.PIN_H43
    assert p["unit"].name == "lgf__gpu__T__x__s1__all_unit.json"
    assert M.unit_state_f(a, u) == (True, "ok") and M.unit_done_run_f(a, u) is not None
    runs = pd.read_csv(p["runs"])
    assert [c_ for c_ in M.RUN_COLS_F if c_ not in runs.columns] == [] and runs[runs.learner != "none"].rmse_cm.notna().all()
    assert set(runs[runs.learner == M.TI].offload_mode) == {"GPU"}
    s_ = M.unit_summary_f(json.loads(p["unit"].read_text()))
    assert s_["n_fail"] == 0 and s_["part"] == "all"
    uf = unit_of(a, "full", "all", job="f_full")
    M.run_unit_f(a, uf)
    assert M.shard_paths_f(a, uf)["unit"].name == "lgfs__gpu__T__x__s1__all_unit.json"
    b = args(["--smoke"], tmp_path)
    ub = ("smoke", "main", "T", "x", 1, "all", (0, 10, 40, -1), 1, 1)
    ujb = M.run_unit_f(b, ub)
    pb = M.shard_paths_f(b, ub)
    assert pb["unit"].exists() and pb["runs"].exists() and not pb["npz"].exists() and not pb["cells"].exists() and not ujb["stored"]
    rb = pd.read_csv(pb["runs"])
    assert rb.rmse_cm.isna().all() and rb.rmse_beq_cm.isna().all() and rb.bias_cm.isna().all()
    chk = {q["check"]: q for q in ujb["smoke_check"]}
    assert chk["ctx_sha_h43"]["ok"] and chk["ctx_sha_h43"]["n_keys"] > 0 and chk["r1_eq_r0_n0"]["ok"] and "cb_pool" in chk
    assert M.unit_state_f(b, ub) == (True, "ok")
    ubf = ("smoke", "full", "T", "x", 1, "all", (0, 10, 40, -1), 1, 1)
    ujf = M.run_unit_f(b, ubf)
    assert {q["check"]: q for q in ujf["smoke_check"]}["ctx_sha_h43"]["ok"]
    res = M.smoke_summary(b)
    assert res is not None and (tmp_path / "lgf_smoke_check.csv").exists() and (tmp_path / "lgf_smoke_fit_timing.csv").exists()
    assert not (tmp_path / "lgf_smoke_tests.csv").exists() and not (tmp_path / "lgf_smoke_curve.csv").exists()


# ================================================================ (q) 종료 코드
def test_q_exit_codes():
    E = M.exit_code_f
    assert E(dict(interrupted="x", n_fail=1)) == 130 and E(dict(n_fail=1, stop="drain")) == 1 and E(dict(stop="drain", n_partial=1)) == 3
    assert E(dict(stop="window", n_partial=1)) == 4 and E(dict(n_partial=2)) == 2 and E(dict()) == 0


# ================================================================ (r) 결측 서술 표
def test_r_missing_desc_subsets(monkeypatch, stub, tmp_path):
    c = make_unit(xb_nan=5)
    monkeypatch.setattr(H, "build_ctx", lambda D_, HA, t, m, sp: c)
    a = args([], tmp_path)
    u = unit_of(a, "main", "all")
    HA = M.ha_of(a, u)
    rows, st, stats, cells = M.run_ctx_f(c, "main", "all", a, HA)
    pth = tmp_path / "c_cells.npz"
    cells.save(pth)
    df = M.missing_desc(a, None, {("T", "x", 1): [pth]}, {}, names=["T|x"])
    reg = df[df.target == "T|x"]
    assert set(reg.hypothesis) == {"F2"} and set(reg.subset) == {"결측 있음", "결측 없음", "전체"}
    miss = reg[reg.subset == "결측 있음"]
    assert (miss.n_cells == 5).all() and miss.delta.isna().all(), "30셀 미만은 비운다"
    full = reg[reg.subset == "결측 없음"]
    assert (full.n_cells == 45).all() and full.delta.notna().all()
    assert set(reg.n) == {0, 10, 40, -1}
    mean = df[df.target.str.startswith("MEAN[")]
    assert len(mean) and mean.note.str.contains("판정과 CI 없음").all()


# ================================================================ (s) 스모크 점검 파일
def test_s_smoke_forced_stock(tmp_path):
    a = args([], tmp_path)
    assert not M.smoke_forced_stock(a)
    pd.DataFrame([dict(check="cached_vs_stock", value=1e-7, ok=True)]).to_csv(tmp_path / "lgf_smoke_check.csv", index=False)
    assert not M.smoke_forced_stock(a)
    pd.DataFrame([dict(check="cached_vs_stock", value=1e-3, ok=False)]).to_csv(tmp_path / "lgf_smoke_check.csv", index=False)
    assert M.smoke_forced_stock(a)


# ================================================================ (t) 검증 지적 반영: 창 시작 시점, 잠금, 결정 기록, 복원 권고, GPU 수
def test_t1_window_start_after_load_and_deadline(tmp_path):
    a = args([], tmp_path / "t1")
    a.MAIN_MODE = True
    st = {}
    assert M.assign_gate(a, st, now=5000.0, load_fn=lambda: (500.0,)) == "load"
    assert not M.window_path(a).exists() and not st.get("started"), "load 보류 중에는 창(t0)을 시작하지 않는다"
    assert M.assign_gate(a, st, now=5000.0 + M.LOAD_CHECK_S, load_fn=lambda: (1.0,)) == "ok"
    assert M._to_ts(M.read_window(a)["t0"]) == pytest.approx(5000.0 + M.LOAD_CHECK_S), "t0 = 첫 배정(제출) 직전"
    b = args([], tmp_path / "t1b")
    M._write_json_excl(M.window_path(b), dict(t0="2026-09-01T00:00:00+09:00", deadline="2026-09-03T00:00:00+09:00", B=46.0, reduce=[], restored=[]))
    w = M.check_window_for_run(b)
    assert M.window_deadline_passed(w), "마감 경과는 거부(종료 코드 1)가 아니라 main 의 stop = 'window'(종료 코드 4)로 처리한다"
    assert M.exit_code_f(dict(n_fail=0, stop="window")) == 4


def test_t2_window_lock_dead_pid_and_after_t0(tmp_path):
    a = args([], tmp_path / "t2")
    lk = M.window_path(a).with_name("lgf_window.json.lock")
    lk.parent.mkdir(parents=True, exist_ok=True)
    lk.write_text("99999999")                                             # 죽은 PID 의 잠금
    import time as _t
    t0 = _t.time()
    M.window_tool(args(["--window-set", "--B", "46", "--e1", "0", "--cbt", "1", "--reduce", "S1,S5", "--viewing-state", "x"], tmp_path / "t2"))
    assert _t.time() - t0 < 5.0 and not lk.exists(), "죽은 PID 의 잠금은 바로 지운다"
    M.window_tool(args(["--window-set", "--restore", "S5", "--viewing-state", "x"], tmp_path / "t2"))
    M.window_start(a, now=1000.0)
    for bad in (["--e1", "1", "--cbt", "1", "--B", "46", "--reduce", "S1,S5"], ["--e1", "0", "--cbt", "0", "--B", "46", "--reduce", "S1,S5"],
                ["--e1", "0", "--cbt", "1", "--B", "50", "--reduce", "S1,S5"], ["--e1", "0", "--cbt", "1", "--B", "46", "--reduce", "S1,S5,S7"]):
        with pytest.raises(SystemExit, match="창이 시작된 뒤"):
            M.window_tool(args(["--window-set", "--viewing-state", "x"] + bad, tmp_path / "t2"))
    M.window_tool(args(["--window-set", "--B", "46", "--e1", "0", "--cbt", "1", "--reduce", "S1,S5,S6", "--viewing-state", "x"], tmp_path / "t2"))
    w = M.read_window(a)
    assert w["reduce"] == ["S1", "S5", "S6"] and w["restored"] == ["S5"], "t0 뒤에도 복원 기록을 지우지 않는다"
    assert M.effective_reduce(w) == ["S1", "S6"]


def test_t3_projection_rows_and_restore_advice(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(M, "enumerate_f", lambda a_: ([], []))            # 작업 완료 상태 표시(실자료)는 건너뛴다
    a = args([], tmp_path / "t3")
    rowsF = [dict(script="F", row="J8", est_gpu_h=1.4), dict(script="F", row="S2", est_gpu_h=1.4), dict(script="F", row="J12", est_gpu_h=9.0)]
    M.write_projection_f(a, rowsF)
    pd.concat([pd.read_csv(tmp_path / "t3" / "lgf_projection.csv"),
               pd.DataFrame([dict(script="N", row="S6", est_gpu_h=0.0), dict(script="N", row="S1", est_gpu_h=3.0)])]).to_csv(
        tmp_path / "t3" / "lgf_projection.csv", index=False)
    M.write_projection_f(a, rowsF + [dict(script="F", row="S3", est_gpu_h=2.0)])
    pj = pd.read_csv(tmp_path / "t3" / "lgf_projection.csv")
    assert set(pj[pj.script == "N"].row) == {"S6", "S1"} and "S3" in set(pj[pj.script == "F"].row), "F 행만 바꾸고 N 행은 남긴다"
    assert M.projection_value(a, "S2", script="F") == 1.4 and M.projection_value(a, "S6", script="N") == 0.0
    M.window_tool(args(["--window-set", "--B", "46", "--e1", "0", "--cbt", "1", "--reduce", "S1,S2,S6", "--viewing-state", "x"], tmp_path / "t3"))
    M.window_start(a, now=M.time.time())
    capsys.readouterr()
    M.window_status(a)
    out = capsys.readouterr().out
    assert "복원 권고 S6: 여유" in out and "복원 권고 S2: 여유" in out and "추정량이 없다" not in out, out


def test_t4_live_gpus_include_h48_workers(tmp_path, monkeypatch):
    a = args([], tmp_path / "t4")
    d = tmp_path / "t4" / "run_lgfn"
    d.mkdir(parents=True)
    (d / f"worker__{os.getpid()}.json").write_text(json.dumps(dict(pid=os.getpid(), gpu="6")))
    (d / "worker__99999999.json").write_text(json.dumps(dict(pid=99999999, gpu="7")))
    assert M.live_gpus(a) == [6] and M.live_gpus(a, include_n=False) == [], "살아 있는 h48 워커의 GPU 만 센다"
    b = args([], tmp_path / "t4b")
    (tmp_path / "t4b").mkdir(parents=True)
    for d_ in (tmp_path / "t4", tmp_path / "t4b"):
        pd.DataFrame([dict(script="F", row="J12", est_gpu_h=50.0)]).to_csv(d_ / "lgf_projection.csv", index=False)
    now = 1_000_000.0
    w = dict(t0="x", deadline=M._dt.datetime.fromtimestamp(now + 30 * 3600).astimezone().isoformat())
    monkeypatch.setattr(M, "default_scope_status", lambda a_, w_=None: (5, 5))
    a.n_jobs_done = b.n_jobs_done = True
    ok1, why1 = M.f_t3_allowed(a, [5], w, now=now)
    ok2, why2 = M.f_t3_allowed(b, [5], w, now=now)
    assert ok1 and "GPU [5, 6]" in why1, "h48 워커의 GPU 를 더하면 여유 30 h × 2 = 60 > 50 GPU-h"
    assert not ok2 and "GPU [5]" in why2, "h47 의 GPU 하나이면 여유 30 GPU-h ≤ 50 GPU-h"


# ================================================================ 무거운 시험(LG_RUN_HEAVY=1)
@HEAVY_MARK
def test_H1_catboost_ctx_real_fit(monkeypatch):
    monkeypatch.setattr(M, "tabicl_fit_predict", stub_icl)
    a, HA, rows, st, stats, cells, tr = run("main", "all", grid=[0, 10])
    assert stats["status"] == "ok" and all(np.isfinite(r["rmse_cm"]) for r in rows if r["learner"] == M.CBX)
    t = [q for q in tr if q["learner"] == M.CBX][0]
    ns = SimpleNamespace(threads=2, cb_iters=200)
    p, _, _ = M.cb_fit_predict(ns, t["X"], t["y"], t["XB"], t["seed"])
    q = M.cb_h40_pred(ns, t["X"], t["y"], t["XB"], t["seed"])                # h40.cb_fit 의 모형 정의, 스레드 제한 Pool
    assert np.max(np.abs(np.asarray(p) - np.asarray(q))) < 1e-9


def _cuda_ready():
    try:
        import torch
        return torch.cuda.is_available() and Path(M.MODEL_PATH_DEFAULT).exists()
    except Exception:                                                     # noqa: BLE001
        return False


@HEAVY_MARK
def test_H2_tabicl_real_fit_cached_equals_stock():
    if not HEAVY_GPU:
        pytest.skip("GPU 시험은 CUDA_VISIBLE_DEVICES 에 허용 후보 GPU 한 장(2, 3, 4, 5, 6, 7, 9)을 줄 때만 돈다")
    a = M.parse_args(["--threads", "2", "--lgt-dir", str(ROOT / "data/processed/lgt")])
    ok, dropped = M.screen_now(a, [int(_CVD)])                          # 메모리 ≤ 50 MiB, 계산 프로세스 없음, LGT 잠금(6·7·9)
    if not ok:
        pytest.skip(f"GPU {_CVD} 가 점유 판정을 통과하지 못했다: {dropped}")
    if not _cuda_ready():
        pytest.skip("CUDA 또는 TabICL 가중치가 없다")
    rng = np.random.RandomState(0)
    X = rng.randn(400, D_FEAT).astype(np.float32); X[rng.rand(400, D_FEAT) < 0.05] = np.nan
    y = X[:, 1].astype(float) * 3 + rng.randn(400)
    XB = rng.randn(120, D_FEAT).astype(np.float32); XB[:3, 2] = np.nan
    p, flag, info = M.tabicl_fit_predict(a, X, y, XB, 0, check_chunk=60, check_stock=True)
    assert np.all(np.isfinite(p)) and flag == "" and info["impl"] == "cached" and info["has_fa3"] is False
    assert info["stock_maxdiff"] <= M.CACHED_TOL and info["chunk_maxdiff"] <= M.CHUNK_TOL and info["offload_mode"] != ""
