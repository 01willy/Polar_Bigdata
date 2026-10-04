"""XI(계획 2.9)·XJ(계획 2.10) 단위 시험. scripts/3_deep_learning/x_climate_extrapolation_retest.py, x_tempderived_aux_labels.py.

합성 자료 시험(라벨 값·RMSE·Δ·판정을 화면에 쓰지 않는다)
XI
(a) 판 별칭 단위: 별칭이 대상 이름이면 h54.R10Unit 과 저장소 이름·키·블록 SSE 가 모두 같다(재현 관문의 전제). 별칭 저장소 이름 규칙.
(b) warm_trim 별칭 단위: A 에서만 셀을 빼고 W·I 채점 셀은 같으며 frac_W_outside_A = 1, 외삽 폭 > 0. cold 의 외삽 폭은 거울 정의.
(c) 누설: warm_trim·warm 단위에서 선택 밖 A 라벨과 B 라벨을 바꿔도 모든 예측이 같다(1절 누설 시험, h54 시험 b 형식). 음성 대조 포함.
(d) 사전 규칙과 약한 쪽: frac 분할 평균 < 0.5 → '외삽 시험 아님', 약한 쪽 순서, 반대 방향은 미결정.
(e) 집계(합성 조각): 봉인 폴더에만 판정 표를 쓰고 화면에는 행 수·해시만, 가설 6행·Holm m = 6, 문장에 공간 대용 표지와 외삽 폭,
    기본 판이 '외삽 시험 아님'이면 약한 쪽으로 판정 불가, 주 행이 모두 판정 불가이면 '기후 외삽은 판정할 수 없었다'.
(f) --count-only: 세기 표와 화면은 허용 열(셀·블록·적합 수, 공변량 √TDD 값)만 쓰고 라벨에서 나온 통계(합성 감시값)는 쓰지 않는다.
(g) 조각 이름(<tag>__cpu__<별칭>__r__s<분할>__<변형>), --shard I/K 분할, 공통 설정 해시가 판·변형 사이에 같다.
XJ
(h) w = 0 은 LG 와 같다: 원천 쪽 기준 키(D0, D1, R1)와 대상 쪽 P1 이 h40.run_ctx 의 같은 키와 블록 SSE 까지 같다.
(i) 가중이 학습 행렬에 맞게 들어간다(h40.TRACE): 기준 행렬 뒤에 보조 행, 가중 w(그 밖 1), 목표(D 는 y, R 은 y − E0·s), 표지 열.
(j) 보조 행은 채점에 없다: 저장소의 블록·셀 수 = 직접 라벨 채점 셀. 대상 쪽은 A 블록 보조 행만 쓰고 B 블록 보조 행은 예측을 바꾸지 않는다.
    aux_coef(w = 0) = h42.shrink(h40.ls_E) 와 정확히 같다.
(k) 보조 행의 100 km 버퍼와 대상 지역 제외: 합성 Data 에서 거리별 판정, 직접 라벨 행에 적용하면 h40.Data.source_idx 와 같은 집합.
(l) 누설: 원천 쪽·대상 쪽 단위에서 선택 밖 A 라벨과 B 라벨을 바꿔도 예측이 같다. 음성 대조 포함.
(m) 약관: --aux-lic-ref 가 없으면 원천 쪽 '시험하지 않음', 개정 이력에 없는 문구는 거부, 가설 15행·Holm m = 15(시험하지 않음 p 1), 대상 쪽 약관 거름.
(n) 집계(합성 조각): 봉인 폴더에만 쓰고 가설 15행, 사전 고정 문장의 둘째 문장, 화면에 판정어 없음.
(o) --count-only 허용 열(합성 감시값), 조각 이름, --shard.
실제 자료 시험(세기 범주)
(p) --count-only 의 화면 출력이 라벨 값에 의존하지 않는다: 라벨을 바꾼 자료로 다시 세어도 화면 출력이 같고, 출력 제한에 걸린 줄이 없다.
    로컬 세기에도 1절 로컬 자원 점검(local_resources)이 걸린다.
검토 결함 수정(2026-10-04)
(q) XI-c 는 비열등 상태(충족 > 미충족 > 판정 불가 > 외삽 시험 아님)로 두 판을 비교하고 문장 꼬리를 약한 쪽으로 쓴다. 가설 수준 판정의 약한 쪽.
(r) 허용 표지가 있는 본 실행에서 등록 입력(Canada~exp~lic, v3 판, XJ 대상 쪽)이 없으면 중단(전체판 825셀만 예외). 묶음 목록.
(s) 조각 runs.csv 에 키별 RMSE·편향 열이 없고 XJ unit.json 에 대상 라벨 통계가 없다. 패키지 판 기록.
(t)·(u) 재현 관문의 범위 점검(기대 분할·저장소·키, 등록 관문 단위)과 패키지 판 기록(XI, XJ).
(v) 허용 표지 없는 스모크 2스레드, 로컬 자원(MALLOC_ARENA_MAX 재시작·거부, 30 GB 대기 1시간, 10 GB 상한).
(w) 원천 쪽 약관 근거: 로컬 대조(작업 트리·HEAD·T0)와 표지, 계획 문서가 없는 환경의 표지 대조.
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 nice -n 10 prlimit --as=10737418240 taskset -c <코어 4개> python3 -m pytest -q -p no:cacheprovider tests/test_x_xi_xj.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util
import json
import re
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "3_deep_learning"))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


XB = _load("xbatch_core", "scripts/3_deep_learning/xbatch_core.py")
XI = _load("x_climate_extrapolation_retest", "scripts/3_deep_learning/x_climate_extrapolation_retest.py")
XJ = _load("x_tempderived_aux_labels", "scripts/3_deep_learning/x_tempderived_aux_labels.py")
H, X, W, H4 = XB.H, XB.X, XB.W, XB.H4

REAL = pytest.mark.skipif(not (ROOT / "data/processed/fidelity_base_v3.csv").exists()
                          or not (XB.LGD_DIR / "Canada_expanded_lic" / "fidelity_base_v3.csv").exists()
                          or not (XB.LGD_DIR / "Tibet" / "fidelity_base_v3.csv").exists()
                          or not (ROOT / "data/processed/fidelity_base_v4.csv").exists(), reason="실제 자료(v3, LGD 실행 표, v4)가 없다")
SENTINELS = ("123.456", "987.654", "55.5551")


@pytest.fixture
def small_hi(monkeypatch):
    """catboost(기본 600회)를 시험 속도로 줄인다(h54 시험과 같은 방식, 시험 안에서만)."""
    monkeypatch.setattr(X, "CB_HI", dict(iterations=10, learning_rate=0.1, depth=3, l2_leaf_reg=3.0))


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    """쓰기 허용 뿌리를 시험 폴더로 둔다(xbatch_core.check_out_dir 의 XBATCH_OUT_ROOTS)."""
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------- 합성 문맥
def make_rctx10(seed=0, nA=120, nW=30, nI=30, target="T"):
    """합성 WF10 문맥(h54.RCtx + maskW). W 셀의 s 가 가장 크다(tests/test_xbatch_core.py 와 같은 구성)."""
    rng = np.random.RandomState(seed)
    D = len(XB.FEATS)
    XA = rng.randn(nA, D).astype(np.float32); sA = rng.uniform(20, 46, nA); yA = 1.4 * sA + 2 * rng.randn(nA)
    sW, sI = rng.uniform(44, 50, nW), rng.uniform(20, 40, nI)
    sB = np.r_[sW, sI]; XBm = rng.randn(nW + nI, D).astype(np.float32); yB = 1.4 * sB + 2 * rng.randn(nW + nI)
    blkA = np.repeat(np.arange(12), nA // 12); blkB = 1000 + np.repeat(np.arange(6), (nW + nI) // 6)
    latA, lonA = 65 + 0.1 * rng.rand(nA), -150 + 0.1 * rng.rand(nA)
    latB, lonB = 66 + 0.1 * rng.rand(nW + nI), -150 + 0.1 * rng.rand(nW + nI)
    c = W.RCtx(target, 1, target, XA, yA, sA, blkA, latA, lonA, XBm, yB, sB, blkB, latB, lonB, 1.6,
               meta=dict(dup_of=-1, valid=True, n_valid_splits=1, n_unique_splits=1, nb_eval_W=3, nb_eval_I=3, frac_W_outside_A=0.5))
    c.maskW = np.r_[np.ones(nW, bool), np.zeros(nI, bool)]
    return c


def make_tctx(seed=0, split=1, target="T", mode="x"):
    """합성 전이 문맥(h40.Ctx). 원천 300행(두 지역), A 80셀 8블록, B 60셀 6블록(tests/test_h54_workflow.py 와 같은 구성)."""
    rng = np.random.RandomState(seed)
    D = len(XB.FEATS)
    ns, nA, nB = 300, 80, 60
    Xs = rng.randn(ns, D); ss = rng.uniform(20, 40, ns); ys = 1.6 * ss + 3 * Xs[:, 1] + 2 * rng.randn(ns)
    XA = rng.randn(nA, D); sA = rng.uniform(20, 40, nA); yA = 1.3 * sA + 3 * XA[:, 1] + 2 * rng.randn(nA)
    XBm = rng.randn(nB, D); sB = rng.uniform(20, 40, nB); yB = 1.3 * sB + 3 * XBm[:, 1] + 2 * rng.randn(nB)
    return H.Ctx(target, mode, split, target, Xs, ys, ss, np.repeat(["r1", "r2"], ns // 2), XA, yA, sA, np.repeat(np.arange(8), nA // 8),
                 XBm, yB, sB, 500 + np.repeat(np.arange(6), nB // 6), meta=dict(dup_of=-1, valid=True))


def make_aux(seed=5, n=12):
    rng = np.random.RandomState(seed)
    D = len(XB.FEATS)
    Xa = rng.randn(n, D).astype(np.float32); sa = rng.uniform(20, 40, n); ya = 1.9 * sa + 4 * rng.randn(n)
    return dict(X=Xa, y=ya, s=sa)


def xi_args(grid=(20, -1)):
    return XB.h54_args(exp="wf10", grid=list(grid), threads=1, cb_iters=10, seeds=1, draws_cap=1, nboot=100)


def xj_args():
    return XB.h54_args(exp="wf4", threads=1, cb_iters=10, seeds=1, draws_cap=2, nboot=100)


def stores_of(st):
    sts = st if isinstance(st, list) else [st]
    return {s_.target: s_ for s_ in sts}


# ================================================================ XI
def test_a_alias_unit_matches_h54_r10unit(small_hi):
    assert XI.W is W and XI.XB is XB and XI.H is H, "동결 모듈은 xbatch_core 가 읽은 것만 써야 한다"
    a = xi_args()
    c = make_rctx10()
    U1 = W.R10Unit(a, c, "wf10", "warm").run()
    r1, s1, _ = U1.finish()
    U2 = XI.XIUnit(a, c, "T", "warm").run()
    r2, s2, _ = U2.finish()
    d1, d2 = stores_of(s1), stores_of(s2)
    assert list(d1) == list(d2) == ["T~warm|r", "T~warmW|r", "T~warmI|r"]
    for nm in d1:
        assert d1[nm].keys == d2[nm].keys and len(d1[nm].keys) > 10
        for k in d1[nm].keys:
            assert np.array_equal(d1[nm].get(k)[0], d2[nm].get(k)[0]), f"{nm} {k} 블록 SSE 가 h54 와 다르다"
    U3 = XI.XIUnit(a, c, "Canada~exp~lic", "cold", dry=True).run()
    _, s3, _ = U3.finish()
    assert [s_.target for s_ in s3] == ["Canada~exp~lic~cold|r", "Canada~exp~lic~coldW|r", "Canada~exp~lic~coldI|r"]
    with pytest.raises(ValueError):
        XI.XIUnit(a, c, "T", "warm_trim")


def test_b_trim_alias_unit_and_width():
    c = make_rctx10()
    sW = c.sB[c.maskW]
    c2, keep = XB.trim_ctx_warm(c, float(np.min(sW)))
    c2.meta.update(XI.extrap_meta(c2, "warm"))
    assert np.array_equal(c2.yB, c.yB) and np.array_equal(c2.maskW, c.maskW) and len(c2.yA) == int(keep.sum()) < len(c.yA)
    assert c2.meta["frac_W_outside_A"] == 1.0 and c2.meta["extrap_width"] > 0
    assert np.isclose(c2.meta["extrap_width"], np.median(sW) - np.max(c2.sA))
    a = xi_args()
    U = XI.XITrimUnit(a, c2, "Canada~exp~lic", dry=True).run()
    rows, st, stats = U.finish()
    assert [s_.target for s_ in st] == ["Canada~exp~lic~warm_trim|r", "Canada~exp~lic~warm_trimW|r", "Canada~exp~lic~warm_trimI|r"]
    r = XI.count_row(c2, stats, "Canada~exp~lic", "warm_trim", 1)
    assert list(r) == XI.COUNT_COLS and r["fits"] > 0 and r["n_trim"] == int((~keep).sum()) and r["frac_W_outside_A"] == 1.0
    cc = make_rctx10(seed=3)
    m = XI.extrap_meta(cc, "cold")
    sWc = cc.sB[cc.maskW]
    assert np.isclose(m["extrap_width"], np.min(cc.sA) - np.median(sWc))


@pytest.mark.parametrize("variant", ["warm_trim", "warm"])
def test_c_xi_no_leakage(small_hi, variant):
    a = xi_args(grid=(20, 40))                                            # 전량(n = −1)은 A 라벨을 모두 쓰므로 선택 밖 라벨이 없다
    c = make_rctx10(seed=1)
    if variant == "warm_trim":
        c, _ = XB.trim_ctx_warm(c, float(np.min(c.sB[c.maskW])))

    def run_fn(cx):
        XI.make_unit(a, cx, "T", variant).run()
    res = XB.leakage_invariance(run_fn, c)
    assert res["ok"] and res["max_abs_diff"] == 0.0 and res["n_keys"] > 10 and 0 < res["n_keep"] < res["n_A"]
    with XB.h54_trace() as (p1, tr):
        run_fn(c)
    keep = XB.selected_union(tr)
    with XB.h54_trace() as (p2, _):
        run_fn(XB.perturb_labels(c, keep[1:], seed=4))
    assert not XB.compare_predictions(p1, p2)["ok"], "선택 라벨을 바꿔도 예측이 같다(시험이 둔하다)"


def test_d_frac_rule_and_weaker():
    st = pd.DataFrame([dict(alias="A", variant="warm", split=s, frac_W_outside_A=f, extrap_width=0.1) for s, f in ((1, 0.024), (2, 0.957), (3, 0.3))])
    fr = XI.frac_summary(st)
    assert np.isclose(fr[("A", "warm")]["frac_mean"], (0.024 + 0.957 + 0.3) / 3) and fr[("A", "warm")]["extrap_test"] is False
    st2 = st.assign(frac_W_outside_A=[0.5, 0.6, 0.4])
    assert XI.frac_summary(st2)[("A", "warm")]["extrap_test"] is True, "평균 0.5 는 규칙을 만족한다(미만만 외삽 시험 아님)"
    assert XI.effective_verdict(dict(verdict4="열세"), False) == XI.NOT_EXTRAP and XI.effective_verdict(None, True) == "행 없음"
    assert XI.weaker("열세", "미결정") == "미결정" and XI.weaker("동등", "우세") == "동등" and XI.weaker("우세", "열세") == "미결정"
    assert XI.weaker("열세", XI.NOT_EXTRAP) == XI.NOT_EXTRAP and XI.weaker("판정 불가", "미결정") == "판정 불가" and XI.weaker("열세", "열세") == "열세"


# ---------------------------------------------------------------- 합성 조각(XI)
XI_KEYS_N = (100, -1)


def _xi_groups():
    g = [("P0", "none", 0, 0.0, 30.0)]
    for n in XI_KEYS_N:
        g += [("P1", "none", n, 0.0, 30.0), ("P2", "none", n, 0.0, 31.0), ("D0", XB.HI, n, 1.0, 33.0), ("D0", XB.LO, n, 1.0, 32.0),
              ("D1", XB.LO, n, 1.0, 31.5), ("R1", XB.LO, n, XB.LAM_CV, 29.0), ("R1", XB.LO, n, 0.25, 29.5), ("R1", XB.LO, n, 1.0, 29.2),
              ("R2", XB.LO, n, XB.LAM_CV, 29.4)]
    return g


def write_xi_shards(o, combos, fracs, splits=(1, 2, 3), seed=0):
    rng = np.random.RandomState(seed)
    nW, nI = 10, 12
    for alias, v in combos:
        for sp in splits:
            cW, cI = rng.randint(3, 9, nW), rng.randint(3, 9, nI)
            bW = np.repeat([f"9{j:02d}" for j in range(nW)], cW); bI = np.repeat([f"{sp}{j:02d}" for j in range(nI)], cI)
            yW, yI = rng.uniform(40, 120, len(bW)), rng.uniform(40, 120, len(bI))
            tot = H4.BlockStore(f"{alias}~{v}|r", sp, np.r_[bW, bI])
            sW, sI = H4.BlockStore(f"{alias}~{v}W|r", sp, bW), H4.BlockStore(f"{alias}~{v}I|r", sp, bI)
            for m, lr, n, lam, lvl in _xi_groups():
                nd = 1 if n in (0, -1) else 3
                for d in range(nd):
                    for s in ((-1,) if lr == "none" else (0, 1)):
                        key = (m, lr, "1", "cell", int(n), d, int(s), float(lam))
                        pW = yW + lvl * 0.1 * rng.randn(len(yW)) + (1.0 if m == "D0" else 0.0)
                        pI = yI + lvl * 0.1 * rng.randn(len(yI))
                        sW.add(key, yW, pW); sI.add(key, yI, pI); tot.add(key, np.r_[yW, yI], np.r_[pW, pI])
            cfg = XI.unit_cfg(o, v, "S")
            unit = dict(frac_W_outside_A=fracs[(alias, v)], extrap_width=0.5, n_A=200, nb_A=20, n_trim=7 if v == XI.TRIM else 0,
                        role=XI.ROLE.get((alias, v), ""), data_version="합성")
            XB.write_shard(o.SHARDS, o.TAG, alias, "r", sp, [], [tot, sW, sI], cfg, unit=unit, variant=v, expected=list(splits))


def _clean(txt):
    assert not XB.FORBIDDEN_OUT.search(txt), f"화면에 출력 제한 대상이 있다: {[ln for ln in txt.splitlines() if XB.FORBIDDEN_OUT.search(ln)][:3]}"


def test_e_xi_summarize_sealed(out_root, capsys):
    o = XI.parse_args(["--out-dir", str(out_root / "XI_climate_extrapolation_retest"), "--nboot", "200"])
    combos = [XI.PRIMARY, XI.BASIC, ("Canada~exp~lic", "cold"), ("Alaska", "warm")]
    fracs = {XI.PRIMARY: 1.0, XI.BASIC: 0.6, ("Canada~exp~lic", "cold"): 0.4, ("Alaska", "warm"): 0.9}
    write_xi_shards(o, combos, fracs)
    capsys.readouterr()
    res = XI.summarize(o)
    out = capsys.readouterr().out
    _clean(out)
    sealed = o.OUT / "sealed"
    names = {p.name for p in sealed.iterdir()}
    assert {"xi_tests.csv", "xi_hyp.csv", "xi_holm.csv", "xi_curve.csv", "xi_meta.json", "sealed_manifest.json", "README.txt"} <= names
    assert not any(p.name.startswith("xi_") and p.suffix == ".csv" and "tests" in p.name for p in o.OUT.iterdir() if p.is_file()), "판정 표가 봉인 밖에 있다"
    assert "[봉인]" in out and "sha256" in out
    hyp, holm = res["hyp"], res["holm"]
    assert len(hyp) == 6 and set(hyp.test_id) == {"XI-a", "XI-b", "XI-c"} and int(holm.m.iloc[0]) == 6 and len(holm) == 6
    assert all(XI.SPACE_PROXY in s and "외삽 폭 √TDD 차" in s for s in hyp.sentence)
    hab, hc = hyp[hyp.test_id != "XI-c"], hyp[hyp.test_id == "XI-c"]
    assert set(hab.verdict_basic) <= set(XB.VERDICTS4) | {"판정 불가", "행 없음"}
    assert set(hc.verdict_basic) <= {XI.NI_OK, XI.NI_NO, "판정 불가", "행 없음"} and set(hc.verdict4_basic) <= set(XB.VERDICTS4) | {"판정 불가"}
    assert {"verdict_hypothesis", "verdict_hypothesis_basic", "verdict_hypothesis_written", "hyp_both_written"} <= set(hyp.columns)
    tests = res["tests"]
    assert set(tests.test_id) >= {"XI-a", "XI-b", "XI-c", "XI-d"} and not tests[tests.test_id != "XI-d"].alias.eq("Alaska").any()
    # 사전 규칙(2.9): frac 분할 평균 0.4 인 cold 판의 행은 판정 열이 모두 '외삽 시험 아님'이고 CI 열은 남는다
    cold = tests[(tests.alias == "Canada~exp~lic") & (tests.variant == "cold")]
    assert len(cold) > 10 and (cold.extrap_rule == XI.NOT_EXTRAP).all()
    for col in XI.NOT_EXTRAP_VERDICT_COLS:
        assert (cold[col] == XI.NOT_EXTRAP).all(), col
    assert cold.ci_hi.astype(float).notna().any() and (cold.limit_dependence.fillna("") == "").all()
    prim = tests[(tests.alias == XI.PRIMARY[0]) & (tests.variant == XI.PRIMARY[1])]
    assert (prim.extrap_rule.fillna("") == "").all() and not (prim.verdict4 == XI.NOT_EXTRAP).any()
    st = pd.read_csv(o.OUT / "xi_structure.csv")
    assert set(st.alias) == {"Canada~exp~lic", "Alaska"} and "frac_W_outside_A" in st and not any("rmse" in c_ for c_ in st.columns)
    meta = json.loads((sealed / "xi_meta.json").read_text())
    assert meta["holm_m"] == 6 and meta["fracs"]["Canada~exp~lic|cold"]["extrap_test"] is False


def test_e2_xi_basic_not_extrap_and_all_na(out_root, capsys):
    o = XI.parse_args(["--out-dir", str(out_root / "XI_b"), "--nboot", "200"])
    write_xi_shards(o, [XI.PRIMARY, XI.BASIC], {XI.PRIMARY: 1.0, XI.BASIC: 0.3}, seed=2)
    res = XI.summarize(o, quiet=True)
    _clean(capsys.readouterr().out)
    hyp = res["hyp"]
    assert (hyp.verdict_basic == XI.NOT_EXTRAP).all() and (hyp.verdict_sentence == XI.NOT_EXTRAP).all() and hyp.both_written.all()
    assert res["meta"]["overall_sentence"] == XI.ALL_NA_SENTENCE
    o2 = XI.parse_args(["--out-dir", str(out_root / "XI_c"), "--nboot", "200"])
    write_xi_shards(o2, [XI.PRIMARY], {XI.PRIMARY: 0.2}, seed=3)                # 주 판의 사전 규칙 미충족(합성)
    res2 = XI.summarize(o2, quiet=True)
    assert (res2["hyp"].verdict_trim == XI.NOT_EXTRAP).all() and (res2["holm"].p == 1.0).all()
    assert all(v == XI.NOT_EXTRAP for v in (h["verdict"] for h in res2["meta"]["hypotheses"].values()))


def test_f_xi_count_whitelist(out_root, capsys):
    o = XI.parse_args(["--count-only", "--out-dir", str(out_root / "XI_cnt")])

    def fake(u):
        r = {k: 1 for k in XI.COUNT_COLS}
        r.update(alias=u[0], variant=u[1], split=u[2], frac_W_outside_A=0.75, extrap_width=0.31, E_A=123.456, y_mean=987.654, rmse_cm=55.5551,
                 E0=55.5551)
        return r
    df = XI.count_only(o, [("Canada~exp~lic", "warm_trim", 1), ("Canada~exp~lic", "warm_trim", 2)], [], unit_fn=fake)
    out = capsys.readouterr().out
    _clean(out)
    assert not any(s in out for s in SENTINELS) and list(df.columns) == XI.COUNT_COLS
    csv = pd.read_csv(o.OUT / "xi_count.csv")
    assert list(csv.columns) == XI.COUNT_COLS and not any(s in (o.OUT / "xi_count.csv").read_text() for s in SENTINELS)


def test_g_xi_names_shard_cfg(out_root):
    assert XB.shard_base("d", "xi", "Canada~exp~lic", "r", 3, "warm_trim").name == "xi__cpu__Canada~exp~lic__r__s3__warm_trim"
    units = [(al, v, s) for al in ("A", "B") for v in ("warm", "cold") for s in (1, 2, 3)]
    parts = [XI.shard_select(units, f"{i}/3") for i in range(3)]
    assert sorted(sum(parts, [])) == sorted(units) and all(len(p) == 4 for p in parts)
    assert XI.shard_select(units, "") == units
    with pytest.raises(SystemExit):
        XI.shard_select(units, "3/3")
    o = XI.parse_args([])
    hs = {XB.cfg_hash(XI.unit_cfg(o, v, d), common=True) for v in XI.VARIANTS_ALL for d in ("s1", "s2")}
    assert len(hs) == 1, "판·변형 사이 공통 설정 해시가 달라 함께 집계할 수 없다"
    assert XI.parse_args(["--smoke"]).TAG == "xi_smoke" and XI.parse_args(["--threads", "16"]).threads <= 4


# ================================================================ XJ
def _lg_args(grid="0,10", draws=2):
    return H.parse_args(["--methods", "P0,P1,D0,D1,R1", "--learners", "catboost_lo", "--n-grid", grid, "--draws", str(draws), "--seeds", "1",
                         "--cb-iters", "10", "--threads", "1", "--alphas", "1", "--place-n-grid", "10"])


def test_h_w0_equals_lg():
    c = make_tctx()
    rows, st_lg, _ = H.run_ctx(c, "cpu", _lg_args(), learner_axis=False, nested=False)
    a = xj_args()
    U = XJ.XJSrcUnit(a, c, "T", make_aux(), grid=[0, 10], draws=2).run()
    _, st, _ = U.finish()
    base = [k for k in st.keys if k[0] in ("D0", "D1", "R1")]
    assert len(base) == 3 * (1 + 2) + 2 * (1 + 2) and st.target == st_lg.target == "T|x"
    for k in base:
        assert k in st_lg, f"LG 에 없는 키 {k}"
        assert np.array_equal(st.get(k)[0], st_lg.get(k)[0]) and np.array_equal(st.get(k)[1], st_lg.get(k)[1]), f"w = 0 키 {k} 가 LG 와 다르다"
    assert any(k[0] == "R1@w0.3" for k in st.keys) and any(k[0] == "D1@wf1" for k in st.keys) and any(k[0] == "D0@w0.1" for k in st.keys)
    T = XJ.XJTgtUnit(a, c, "T", dict(y=np.zeros(0), s=np.zeros(0)), grid=[0, 10], draws=2).run()
    _, stt, _ = T.finish()
    p1 = [k for k in stt.keys if k[0] == "P1"]
    assert len(p1) == 3
    for k in p1 + [k for k in stt.keys if k[0] == "P0"]:
        assert np.array_equal(stt.get(k)[0], st_lg.get(k)[0]), f"대상 쪽 {k} 가 LG 와 다르다"


def test_i_weights_in_training_matrix(monkeypatch):
    c = make_tctx(seed=2)
    aux = make_aux(n=9)
    a = xj_args()
    tr = []
    monkeypatch.setattr(H, "TRACE", tr)
    XJ.XJSrcUnit(a, c, "T", aux, grid=[10], draws=1).run()
    by = {(e["method"], int(e["n"]), int(e["seed"])): e for e in tr}
    for m, kind in (("R1", "R"), ("D0", "D"), ("D1", "D")):
        b = by[(m, 10, 0)]
        assert b["w"] is None, "기준(w = 0) 행렬에 가중이 있다(LG 와 달라진다)"
        n0 = len(b["ytr"])
        for w in XJ.W_GRID:
            for fl in (False, True):
                e = by[(XJ.mname(m, w, fl), 10, 0)]
                assert len(e["ytr"]) == n0 + 9 and np.array_equal(e["w"], np.r_[np.ones(n0), np.full(9, w)])
                Xe = e["Xtr"]
                assert np.array_equal(Xe[:n0, :len(XB.FEATS)], b["Xtr"]) and np.allclose(Xe[n0:, :len(XB.FEATS)], aux["X"])
                assert np.array_equal(e["ytr"][:n0], b["ytr"])
                ya = aux["y"] if kind == "D" else aux["y"] - c.E0 * aux["s"]
                assert np.allclose(e["ytr"][n0:], ya)
                if fl:
                    assert Xe.shape[1] == len(XB.FEATS) + 1 and np.array_equal(Xe[:, -1], np.r_[np.zeros(n0), np.ones(9)])
                else:
                    assert Xe.shape[1] == len(XB.FEATS)


def test_j_aux_not_scored_and_coef():
    c = make_tctx(seed=3)
    a = xj_args()
    U = XJ.XJSrcUnit(a, c, "T", make_aux(), grid=[0, 10], draws=1).run()
    _, st, _ = U.finish()
    assert set(st.blocks) == set(np.unique(c.blkB).astype(str)) and int(st.ncell.sum()) == len(c.yB)
    for k in st.keys:
        assert int(st.get(k)[1].sum()) == len(c.yB), "채점 셀 수가 직접 라벨 채점 셀과 다르다"
    rng = np.random.RandomState(0)
    ys, ss = rng.uniform(50, 150, 7), rng.uniform(20, 40, 7)
    E, m = XJ.aux_coef(ys, ss, rng.uniform(50, 150, 5), rng.uniform(20, 40, 5), 0.0, 1.6)
    assert E == X.shrink(H.ls_E(ys, ss), 1.6, 7, XB.KAPPA) and m == 7.0, "w = 0 의 계수가 LG 의 P1 과 정확히 같지 않다"
    assert XJ.aux_coef(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 0.3, 1.6) == (1.6, 0.0)
    E1, m1 = XJ.aux_coef(np.zeros(0), np.zeros(0), np.array([100.0]), np.array([25.0]), 1.0, 1.6)
    assert np.isclose(E1, (1.0 * 4.0 + 10 * 1.6) / 11.0) and m1 == 1.0
    # 대상 쪽: B 블록 보조 행은 쓰지 않는다(분할 함수가 A 블록만 고른다)
    D = SimpleNamespace(df=pd.DataFrame(dict(block=np.r_[np.repeat(np.arange(8), 10), 500 + np.repeat(np.arange(6), 10)])),
                        target_idx=lambda t: np.arange(140))
    aux_df = pd.DataFrame(dict(block=[0, 3, 7, 500, 503, 999]))
    import x_tempderived_aux_labels as XJm
    orig = XJm.half_split_blocks
    try:
        XJm.half_split_blocks = lambda df, idx, sp: (np.arange(80), np.arange(80, 140))
        cA = SimpleNamespace(blkA=np.repeat(np.arange(8), 10))
        inA, comp = XJ.tgt_aux_split(D, "T", 1, cA, aux_df)
    finally:
        XJm.half_split_blocks = orig
    assert inA.tolist() == [True, True, True, False, False, True], "A 풀은 B 블록만 뺀다(A 블록과 직접 라벨 없는 블록은 넣는다)"
    assert comp["n_aux_A"] == 3 and comp["n_aux_B"] == 2 and comp["n_aux_other"] == 1 and comp["n_aux_pool"] == 4
    T1 = XJ.XJTgtUnit(a, c, "T", dict(y=np.array([90.0, 110.0]), s=np.array([30.0, 33.0]))).run()
    s1 = T1.finish()[1]
    assert set(s1.blocks) == set(np.unique(c.blkB).astype(str)) and all(int(s1.get(k)[1].sum()) == len(c.yB) for k in s1.keys)
    assert any(k[0] == "P1@w0.3" for k in s1.keys) and any(k[0] == "P1" for k in s1.keys)


def _fake_D(n_t=20, seed=0):
    """합성 h40.Data 대용(df, target_idx, parent_of, macros, args.buffer_km). 대상 T 는 위도 60, 경도 100–101."""
    rng = np.random.RandomState(seed)
    lat_t, lon_t = 60 + 0.2 * rng.rand(n_t), 100 + rng.rand(n_t)
    lat_o = np.r_[60 + 0.2 * rng.rand(30), 70 + rng.rand(30), 60.1 + np.zeros(10)]
    lon_o = np.r_[102.0 + 1.5 * rng.rand(30), 100 + rng.rand(30), 101.5 + 0.02 * np.arange(10)]
    df = pd.DataFrame(dict(lat=np.r_[lat_t, lat_o], lon=np.r_[lon_t, lon_o], macro=np.r_[["T"] * n_t, ["O"] * 70],
                           y=np.r_[rng.uniform(50, 100, n_t), rng.uniform(50, 100, 70)]))
    D = SimpleNamespace(df=df, macros={"T", "O"}, args=SimpleNamespace(buffer_km=100.0), _src={})
    D.target_idx = lambda t: np.where(df.macro.values == t)[0]
    D.parent_of = lambda t: t
    return D


def test_k_aux_buffer_matches_source_idx():
    D = _fake_D()
    t_idx, parent, src, comp = H.Data.source_idx(D, "T", "x")
    keep, cmp_ = XJ.aux_source_mask(D, D.df.lat.values, D.df.lon.values, D.df.macro.values, "T")
    assert set(np.where(keep)[0]) == set(src.tolist()), "직접 라벨 행에 적용한 보조 행 규칙이 h40.Data.source_idx 와 다르다"
    assert cmp_["n_aux_excl_target"] == len(t_idx) and cmp_["n_aux_excl_buffer"] == comp["n_buffer_excluded"] > 0
    lat, lon = D.df.lat.values[D.target_idx("T")], D.df.lon.values[D.target_idx("T")]
    a_lat = np.array([60.1, 60.1, 60.1, 75.0, 60.1])
    a_lon = np.array([100.5, 102.2, 104.5, 100.5, 100.5])                 # 대상 안, 약 70–80 km, 약 200 km, 먼 곳, 대상 지역 표지
    a_mac = np.array(["O", "O", "O", "O", "T"])
    k2, c2 = XJ.aux_source_mask(D, a_lat, a_lon, a_mac, "T")
    dmin = [float(XJ.haversine_km(a_lat[j], a_lon[j], lat, lon).min()) for j in range(5)]
    assert dmin[1] < 100 < dmin[2]
    assert k2.tolist() == [False, False, True, True, False] and c2 == dict(n_aux_total=5, n_aux_excl_target=1, n_aux_excl_buffer=2, n_aux_used=2)
    with pytest.raises(ValueError):
        XJ.aux_source_mask(SimpleNamespace(macros={"O"}), a_lat, a_lon, a_mac, "AL-1")


@pytest.mark.parametrize("side", ["src", "tgt"])
def test_l_xj_no_leakage(side):
    a = xj_args()
    c = make_tctx(seed=4)
    aux = make_aux()

    def run_fn(cx):
        if side == "src":
            XJ.XJSrcUnit(a, cx, "T", aux, grid=[0, 10], draws=2).run()
        else:
            XJ.XJTgtUnit(a, cx, "T", dict(y=aux["y"][:4], s=aux["s"][:4]), grid=[0, 3, 10], draws=2).run()
    res = XB.leakage_invariance(run_fn, c)
    assert res["ok"] and res["max_abs_diff"] == 0.0 and res["n_keys"] > 5 and 0 < res["n_keep"] < res["n_A"]
    with XB.h54_trace() as (p1, tr):
        run_fn(c)
    keep = XB.selected_union(tr)
    with XB.h54_trace() as (p2, _):
        run_fn(XB.perturb_labels(c, keep[1:], seed=7))
    assert not XB.compare_predictions(p1, p2)["ok"], "선택 라벨을 바꿔도 예측이 같다(시험이 둔하다)"


def test_m_licence_gate_and_holm():
    assert XJ.check_aux_lic_ref("")["ok"] is False
    o = XJ.parse_args([])
    assert o.SRC_TESTED is False and "시험하지 않음" not in o.LIC["check"] and o.LIC["check"].startswith("근거 미기록")
    with pytest.raises(SystemExit):
        XJ.parse_args(["--aux-lic-ref", "이 문구는 개정 이력에 없다 0f3c"])
    s = XJ.parse_args(["--smoke"])
    assert s.SRC_TESTED and s.SMOKE_LIC and s.SRC_TARGETS == ["Russia_W"]
    hyp, holm = XJ.xj_hypotheses([], src_tested=False)
    assert len(hyp) == 15 and int(holm.m.iloc[0]) == 15 and (holm.p == 1.0).all()
    assert (hyp[hyp.test_id.isin(["XJ-1", "XJ-2"])].verdict4 == XJ.NOT_TESTED).all() and (hyp[hyp.test_id == "XJ-3"].verdict4 == "행 없음").all()
    assert list(hyp.test_id).count("XJ-1") == 6 and list(hyp.test_id).count("XJ-2") == 3 and list(hyp.test_id).count("XJ-3") == 6
    lab = pd.DataFrame(dict(loc_id=[1, 2, 3, 4], license=["CC BY 4.0", "", None, "CC BY 4.0"], lic_unverified=[0, 0, 0, 1]))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "lab.csv"
        lab.to_csv(p, index=False)
        df, info = XJ.aux_licence(pd.DataFrame(dict(loc_id=[1, 2, 3, 4, 5])), p)
    assert df.loc_id.tolist() == [1] and info == dict(n_in=5, n_ok=1, licenses=["CC BY 4.0"])


# ---------------------------------------------------------------- 합성 조각(XJ)
def write_xj_shards(o, splits=(1, 2), seed=0, src=True):
    rng = np.random.RandomState(seed)
    names = [nm.split("|")[0] for nm in XB.MAIN4] if src else []
    for t in names + [XJ.TGT_ALIAS]:
        for sp in splits:
            nb = 9
            cnt = rng.randint(3, 9, nb)
            blk = np.repeat([f"{sp}{j:02d}" for j in range(nb)], cnt)
            y = rng.uniform(40, 120, len(blk))
            st = H4.BlockStore(f"{t}|x", sp, blk)
            st.add(("P0", "none", "1", "cell", 0, 0, -1, 0.0), y, y + 3 * rng.randn(len(y)))
            if t == XJ.TGT_ALIAS:
                for n in XJ.TGT_GRID:
                    for d in range(1 if n == 0 else 5):
                        for m in ["P1"] + [XJ.mname("P1", w) for w in XJ.W_GRID]:
                            st.add((m, "none", "1", "cell", n, d, -1, 0.0), y, y + 2.5 * rng.randn(len(y)))
            else:
                for n in XJ.SRC_GRID:
                    for d in range(1 if n == 0 else 5):
                        for s in (0, 1):
                            for m0, lam in XJ.SRC_METHODS:
                                for w, fl in [(0.0, False)] + [(w, f) for w in XJ.W_GRID for f in (False, True)]:
                                    lams = W.LAMS if m0 == "R1" else (1.0,)
                                    for lm in lams:
                                        st.add((XJ.mname(m0, w, fl), XB.LO, "1", "cell", n, d, s, float(lm)), y, y + 2.0 * rng.randn(len(y)))
            side = "tgt" if t == XJ.TGT_ALIAS else "src"
            cfg = XJ.unit_cfg(o, side, "S")
            XB.write_shard(o.SHARDS, o.TAG, t, "x", sp, [], [st], cfg, unit=dict(side=side, n_A=50, nb_A=8, n_aux_used=3, aux_lic="합성"),
                           expected=list(splits))


def test_n_xj_summarize_sealed(out_root, capsys):
    o = XJ.parse_args(["--out-dir", str(out_root / "XJ_tempderived_aux_labels"), "--nboot", "200"])
    write_xj_shards(o)
    capsys.readouterr()
    res = XJ.summarize(o)
    out = capsys.readouterr().out
    _clean(out)
    sealed = o.OUT / "sealed"
    assert {"xj_tests.csv", "xj_hyp.csv", "xj_holm.csv", "xj_curve.csv", "xj_meta.json"} <= {p.name for p in sealed.iterdir()}
    hyp, holm = res["hyp"], res["holm"]
    assert len(hyp) == 15 and int(holm.m.iloc[0]) == 15 and res["meta"]["src_tested"] is True
    assert all(s.endswith(XJ.SENT_TAIL) for s in hyp.sentence)
    assert set(hyp.verdict4) <= set(XB.VERDICTS4) | {"판정 불가"}, "합성 조각에서 행 없음이 나왔다(키 이름 불일치)"
    tests = res["tests"]
    m1 = tests[(tests.test_id == "XJ-1") & (tests.scope == "MEAN")]
    assert len(m1) == 6 and set(m1.pool) <= {"지역 4/4"} and (tests[tests.test_id == "XJ-3"].scope == "region").all()
    st = pd.read_csv(o.OUT / "xj_structure.csv")
    assert not any("rmse" in c_ or "delta" in c_ for c_ in st.columns)


def test_o_xj_count_whitelist_names(out_root, capsys):
    o = XJ.parse_args(["--count-only", "--out-dir", str(out_root / "XJ_cnt")])

    def fake(u):
        r = {k: 2 for k in XJ.COUNT_COLS}
        r.update(side=u[0], alias=u[1], split=u[2], E_A=123.456, y_mean=987.654, rmse_cm=55.5551)
        return r
    df = XJ.count_only(o, [("tgt", XJ.TGT_ALIAS, 1), ("tgt", XJ.TGT_ALIAS, 2)], [dict(status="시험하지 않음")], unit_fn=fake)
    out = capsys.readouterr().out
    _clean(out)
    assert not any(s in out for s in SENTINELS) and list(df.columns) == XJ.COUNT_COLS
    assert list(pd.read_csv(o.OUT / "xj_count.csv").columns) == XJ.COUNT_COLS
    assert XB.shard_base("d", "xj", "Russia_W", "x", 2).name == "xj__cpu__Russia_W__x__s2"
    units = [("src", t, s) for t in ("Lena", "Canada") for s in (1, 2, 3, 4, 5)]
    assert sorted(XJ.shard_select(units, "0/2") + XJ.shard_select(units, "1/2")) == sorted(units)
    assert XB.cfg_hash(XJ.unit_cfg(o, "src", "a"), common=True) == XB.cfg_hash(XJ.unit_cfg(o, "tgt", "b"), common=True)
    c = make_tctx()
    U = XJ.XJSrcUnit(xj_args(), c, "T", make_aux(), dry=True, grid=[0, 10], draws=5).run()
    _, _, stats = U.finish()
    r = XJ.count_row("src", "T", 1, c, stats, dict(n_aux_total=5, n_aux_used=3))
    assert list(r) == XJ.COUNT_COLS and r["fits"] == (1 + 5) * 1 * 3 * 7


# ================================================================ 실제 자료(세기 범주)
def _clear_caches():
    W._DATA.clear(); W._WF10_PLAN.clear(); W._SPLIT_EXT.clear()
    H._DATA = None
    XI._ARGS.clear(); XJ._ARGS.clear(); XJ._AUX.clear()


@REAL
def test_p_count_only_output_label_invariant(out_root, monkeypatch, capsys):
    """--count-only 의 화면 출력이 라벨 값과 무관함을 확인한다: 라벨(ALT)을 아핀 변환한 자료로 다시 세어도 화면 출력이 같다.
    출력 제한에 걸린 줄도 없어야 한다. 시험은 라벨 통계를 계산하거나 출력하지 않는다."""
    import polar.m1_core as M1
    argv_i = ["--count-only", "--versions", "Canada~exp~lic", "--variants", "warm_trim", "--splits", "2", "--out-dir", str(out_root / "XIc")]
    argv_j = ["--count-only", "--smoke", "--out-dir", str(out_root / "XJc")]
    calls = []
    monkeypatch.setattr(XI, "local_resources", lambda cli_argv=None: calls.append(("XI", cli_argv)))
    monkeypatch.setattr(XJ, "local_resources", lambda cli_argv=None: calls.append(("XJ", cli_argv)))
    monkeypatch.delenv("WF_RESCALE", raising=False); monkeypatch.delenv("LG_RESCALE", raising=False)
    outs = []
    real_lb = M1.load_base
    for shift in (False, True):
        _clear_caches()
        if shift:
            def lb(*a, **k):
                df = real_lb(*a, **k)
                df[XJ.TARGET] = df[XJ.TARGET] * 1.37 + 3.0
                return df
            monkeypatch.setattr(H, "load_base", lb)
            monkeypatch.setattr(XJ, "load_base", lb)
        capsys.readouterr()
        assert XI.main(argv_i) == 0 and XJ.main(argv_j) == 0
        txt = capsys.readouterr().out
        assert "[출력 제한]" not in txt, "세기 출력에서 걸러진 줄이 있다"
        _clean(txt)
        outs.append(re.sub(r"\d+\.?\d*s\b", "", txt))
    _clear_caches()
    assert calls == [("XI", None), ("XJ", None)] * 2, "로컬 --count-only 에 1절 로컬 자원 점검이 걸리지 않았다"
    assert outs[0] == outs[1], "라벨을 바꾸면 세기 화면 출력이 달라진다(라벨에서 나온 값이 출력에 있다)"
    cnt = pd.read_csv(out_root / "XIc" / "xi_count.csv")
    assert list(cnt.columns) == XI.COUNT_COLS and (cnt.frac_W_outside_A == 1.0).all() and (cnt.n_trim > 0).all()
    cj = pd.read_csv(out_root / "XJc" / "xj_count_smoke.csv") if (out_root / "XJc" / "xj_count_smoke.csv").exists() else \
        pd.read_csv(out_root / "XJc" / "xj_smoke_count.csv")
    assert set(cj.side) == {"src", "tgt"} and (cj.fits[cj.side == "src"] > 0).all() and (cj.fits[cj.side == "tgt"] == 0).all()


# ================================================================ 검토 결함 수정(2026-10-04) 시험
def _xr(alias, variant, item, n, v4, hi, hib, lo=-0.4, lob=-0.4, d=0.1):
    """XI 가설 행 합성(대비 행의 필요한 열만). 값은 판정 규칙을 시험하기 위한 합성 값이다."""
    return dict(alias=alias, variant=variant, item=item, n=n, verdict4=v4, ci_lo=lo, ci_hi=hi, ci_lo_beq=lob, ci_hi_beq=hib, delta=d,
                p_two=0.2, p_ni_x2=0.1, small_note="", limit_dependence="", ci_dependence="")


def test_q_xi_weaker_noninf_and_hypothesis_level():
    """XI-c 는 비열등 상태로 두 판을 비교하고(미충족이 충족보다 약하다) 문장 꼬리는 약한 쪽 상태로 쓴다. 가설 수준 판정도 두 판을 따로 내고
    약한 쪽을 verdict_hypothesis_written 에 쓴다(계획 2.9 '두 판정을 모두 적고 약한 쪽으로 쓴다')."""
    P, B = XI.PRIMARY, XI.BASIC
    rows = [_xr(*P, "a", 100, "열세", 2.0, 2.1, lo=0.6, lob=0.7, d=1.3), _xr(*P, "a", -1, "열세", 2.0, 2.2, lo=0.6, lob=0.5, d=1.2),
            _xr(*B, "a", 100, "열세", 2.0, 2.1, lo=0.6, lob=0.7, d=1.3), _xr(*B, "a", -1, "미결정", 2.0, 2.2, lo=-0.6, lob=-0.5, d=0.9),
            _xr(*P, "b", 100, "미결정", 0.6, 0.7), _xr(*P, "b", -1, "미결정", 0.6, 0.7),
            _xr(*B, "b", 100, "미결정", 0.6, 0.7), _xr(*B, "b", -1, "미결정", 0.6, 0.7),
            _xr(*P, "c", 100, "미결정", 0.3, 0.2, lo=-0.6, lob=-0.7), _xr(*P, "c", -1, "미결정", 0.3, 0.25, lo=-0.6, lob=-0.7),
            _xr(*B, "c", 100, "미결정", 0.8, 0.3, lo=-0.6, lob=-0.7), _xr(*B, "c", -1, "미결정", 0.3, 0.2, lo=-0.6, lob=-0.7)]
    fr = {P: dict(extrap_test=True, frac_mean=1.0, width_mean=0.5), B: dict(extrap_test=True, frac_mean=0.6, width_mean=0.2)}
    hyp, holm, summ = XI.xi_hypotheses(rows, fr)
    c100 = hyp[(hyp.test_id == "XI-c") & (hyp.n == 100)].iloc[0]
    assert (c100.verdict_trim, c100.verdict_basic, c100.verdict_sentence) == (XI.NI_OK, XI.NI_NO, XI.NI_NO) and c100.both_written
    assert c100.verdict4_sentence == "미결정" and not c100.both_written_v4
    assert "미충족이다(warm_trim 판 충족, 기본 판 미충족, 약한 쪽으로 씀)" in c100.sentence
    cA = hyp[(hyp.test_id == "XI-c") & (hyp.n == -1)].iloc[0]
    assert cA.verdict_sentence == XI.NI_OK and not cA.both_written and "충족이다." in cA.sentence
    h = summ["hypotheses"]
    assert h["XI-c"]["verdict"].startswith("지지") and h["XI-c"]["verdict_basic"].startswith("부분 지지")
    assert h["XI-c"]["verdict_written"].startswith("부분 지지") and h["XI-c"]["both_written"]
    assert h["XI-a"]["verdict"].startswith("지지") and h["XI-a"]["verdict_basic"].startswith("부분 지지")
    assert h["XI-a"]["verdict_written"].startswith("부분 지지") and h["XI-a"]["both_written"]
    assert h["XI-b"]["verdict_written"] == h["XI-b"]["verdict"] and not h["XI-b"]["both_written"]
    ra = hyp[hyp.test_id == "XI-a"]
    assert ra.verdict_hypothesis_written.str.startswith("부분 지지").all() and ra.hyp_both_written.all()
    aA = ra[ra.n == -1].iloc[0]
    assert (aA.verdict_trim, aA.verdict_basic, aA.verdict_sentence) == ("열세", "미결정", "미결정") and aA.both_written
    assert XI.weaker_ni(XI.NI_OK, XI.NI_NO) == XI.NI_NO and XI.weaker_ni(XI.NI_NO, "판정 불가") == "판정 불가"
    assert XI.weaker_ni("행 없음", XI.NOT_EXTRAP) == XI.NOT_EXTRAP
    assert XI.weaker_hyp("지지: x", "기각: y") == "기각: y" and XI.weaker_hyp("기각: y", "판정 불가(행 없음)") == "판정 불가(행 없음)"
    assert XI.weaker_hyp("부분 지지: x", XI.NOT_EXTRAP) == XI.NOT_EXTRAP and XI.hyp_class("부분 지지: n=100") == "부분 지지"
    assert XI.ni_status(None, True) == "행 없음" and XI.ni_status(_xr(*P, "c", 100, "판정 불가", np.nan, np.nan), True) == "판정 불가"
    hyp2, _, s2 = XI.xi_hypotheses([r for r in rows if (r["alias"], r["variant"]) == P], {P: fr[P]})
    assert (hyp2.verdict_basic == XI.NOT_TESTED).all() and not hyp2.both_written.any()
    assert s2["hypotheses"]["XI-c"]["verdict_written"] == s2["hypotheses"]["XI-c"]["verdict"]


def test_r_permitted_run_requires_registered_inputs(tmp_path, monkeypatch):
    """허용 표지가 있는 본 실행(스모크 아님)에서 등록 입력이 없으면 중단한다(전체판 825셀만 예외). 묶음 목록에 XI·XJ 입력이 있다."""
    empty = tmp_path / "rt"
    empty.mkdir()
    monkeypatch.setenv("WF_RESCALE", "1")
    o = XI.parse_args(["--lgd-dir", str(empty), "--versions", "Canada~exp~lic,Canada~exp", "--out-dir", str(tmp_path / "XI")])
    with pytest.raises(SystemExit):
        XI.enumerate_units(o)
    o2 = XI.parse_args(["--lgd-dir", str(empty), "--versions", "Canada~exp", "--out-dir", str(tmp_path / "XI")])
    units, skipped, _ = XI.enumerate_units(o2)
    assert units == [] and {s_["status"] for s_ in skipped} == {"실행 표 없음"}
    o3 = XI.parse_args(["--data-dir", str(empty), "--versions", "Alaska", "--out-dir", str(tmp_path / "XI")])
    with pytest.raises(SystemExit):
        XI.enumerate_units(o3)
    o4 = XI.parse_args(["--lgd-dir", str(empty), "--data-dir", str(empty), "--smoke", "--out-dir", str(tmp_path / "XI")])   # 스모크는 예외
    units, skipped, _ = XI.enumerate_units(o4)
    assert units == [] and {s_["alias"] for s_ in skipped} == {XI.PRIMARY[0], "Canada"}
    oj = XJ.parse_args(["--lgd-dir", str(empty), "--sides", "tgt", "--out-dir", str(tmp_path / "XJ")])
    with pytest.raises(SystemExit):
        XJ.enumerate_units(oj)
    oj2 = XJ.parse_args(["--aux-v4", str(empty / "none.csv"), "--sides", "tgt", "--out-dir", str(tmp_path / "XJ")])
    with pytest.raises(SystemExit):
        XJ.enumerate_units(oj2)
    monkeypatch.delenv("WF_RESCALE")
    o5 = XI.parse_args(["--lgd-dir", str(empty), "--versions", "Canada~exp~lic", "--out-dir", str(tmp_path / "XI")])
    units, skipped, _ = XI.enumerate_units(o5)                              # 허용 표지 없음(세기): 건너뜀으로 기록한다
    assert units == [] and skipped
    for f in XI.PAYLOAD_EXTRA + XJ.PAYLOAD_EXTRA:
        assert f in XB.PAYLOAD_INPUTS, f
    assert XJ.LIC_TOKEN in XB.PAYLOAD_OPTIONAL
    XI._ARGS.clear(); XJ._ARGS.clear()


def test_s_shards_have_no_label_statistics(out_root, monkeypatch, small_hi):
    """조각 runs.csv 에 키별 RMSE·편향 열이 없고, XJ unit.json 에 대상 라벨 통계(E_A, E_B, y_mean 등)가 없다. 두 모듈 모두 패키지 판을 적는다."""
    o = XI.parse_args(["--out-dir", str(out_root / "XI_s")])
    a, c = xi_args(), make_rctx10(seed=5)
    monkeypatch.setattr(XI, "build_ctx", lambda o_, al, v, sp: (a, c))
    monkeypatch.setattr(XI, "data_sha", lambda o_, al: "S")
    XI.run_unit(o, "T", "warm", 1)
    p = XB.shard_paths(o.SHARDS, o.TAG, "T", "r", 1, "warm")
    runs = pd.read_csv(p["runs"])
    assert len(runs) > 10 and not set(XI.RUN_DROP_COLS) & set(runs.columns) and {"n_nonfinite", "fit_flag"} <= set(runs.columns)
    uj = json.loads(p["unit"].read_text())
    assert "numpy" in uj["deps"] and not any("rmse" in k for k in uj)
    oj = XJ.parse_args(["--out-dir", str(out_root / "XJ_s")])
    aj, cj = xj_args(), make_tctx(seed=6)
    cj.meta.update(E_A=123.456, E_B=987.654, E_own=55.5551, y_mean=987.654, y_sd=55.5551, logE_ratio_AB=0.123456, logE_ratio_own=0.654321,
                   E_block_cv=0.5, tau2=0.1, sigma2=0.2, n_A=80, nb_A=8, n_eval=60, nb_eval=6, n_src=300, smd_x25=0.3)
    U = XJ.XJTgtUnit(aj, cj, "T", dict(y=np.array([90.0, 110.0]), s=np.array([30.0, 33.0])), grid=[0, 3], draws=2)
    comp = dict(n_aux_total=3, n_aux_A=2, n_aux_B=1, n_aux_other=0, n_aux_pool=2)
    monkeypatch.setattr(XJ, "build_unit", lambda o_, side, al, sp, dry=False: (aj, cj, U, comp))
    monkeypatch.setattr(XJ, "data_sha", lambda o_, a_, side, al: "S")
    XJ.run_unit(oj, "tgt", "T", 1)
    pj = XB.shard_paths(oj.SHARDS, oj.TAG, "T", "x", 1)
    txt = pj["unit"].read_text()
    uj = json.loads(txt)
    for k in ("E_A", "E_B", "E_own", "y_mean", "y_sd", "logE_ratio_AB", "logE_ratio_own", "E_block_cv", "tau2", "sigma2"):
        assert k not in uj, k
    assert not any(s_ in txt for s_ in SENTINELS) and uj["n_A"] == 80 and uj["n_src"] == 300 and uj["n_aux_pool"] == 2 and "numpy" in uj["deps"]
    rj = pd.read_csv(pj["runs"])
    assert len(rj) and not set(XJ.RUN_DROP_COLS) & set(rj.columns)


# ---------------------------------------------------------------- 재현 관문 범위 점검(합성 조각)
DEPS_LOG = "[2026-10-01T16:34:01Z] [deps] 없는 필수 패키지\n[deps] 확인: catboost 1.2.10 · scikit-learn 1.9.1 · pandas 2.3.3 · scipy 1.17.1 · numpy 2.4.6\n"


def _gate_store(name, sp, keys, nb=5):
    """합성 저장소(같은 이름·분할이면 두 쪽의 블록 SSE 가 같다)."""
    rng = np.random.RandomState(1000 + int(sp))
    blk = np.repeat([f"{sp}{j:02d}" for j in range(nb)], 3)
    y = rng.uniform(40, 120, len(blk))
    st = H4.BlockStore(name, sp, blk)
    for k in keys:                                                        # 키마다 정해진 합성 오차(두 쪽에서 같은 키는 같은 SSE)
        st.add(tuple(k), y, y + 0.01 * (1 + zlib.crc32(str(tuple(k)).encode()) % 97))
    return st


def _xi_gate_keys(extra=()):
    ks = [("P0", "none", "1", "cell", 0, 0, -1, 0.0), ("P1", "none", "1", "cell", -1, 0, -1, 0.0)]
    ks += [("P1", "none", "1", "cell", 100, d, -1, 0.0) for d in range(3)]
    ks += [("D0", XB.HI, "1", "cell", 100, d, s, 1.0) for d in range(3) for s in (0, 1)]
    return ks + list(extra)


def _xi_gate_write(root, tag, alias, variant, splits, keys, unit=None):
    for sp in splits:
        sts = [_gate_store(f"{alias}~{variant}{part}|r", sp, keys) for part in ("", "W", "I")]
        XB.write_shard(root, tag, alias, "r", sp, [], sts, {}, unit=dict(unit or {}), variant=variant, expected=[1, 2])


def test_t_xi_gate_coverage(out_root, capsys):
    ref = out_root / "ref_wf10"
    deps = out_root / "deps.log"
    deps.write_text(DEPS_LOG)
    _xi_gate_write(ref, "wf10", "Canada", "warm", (1, 2), _xi_gate_keys(extra=[("D0", XB.HI, "1", "cell", 100, 0, 7, 1.0)]))   # seed 7: 설계 밖

    def run(name, mine_splits=(1, 2), mine_extra=(), drop=(), versions="Canada"):
        o = XI.parse_args(["--out-dir", str(out_root / name), "--gate-ref", str(ref), "--gate-ref-deps", str(deps), "--versions", versions,
                           "--variants", "warm", "--splits", "2"])
        keys = [k for k in _xi_gate_keys(extra=mine_extra) if k not in drop]
        _xi_gate_write(o.SHARDS, o.TAG, "Canada", "warm", mine_splits, keys, unit=dict(deps={"numpy": "2.1.1", "catboost": "1.2.10"}))
        rc = XI.gate(o)
        g = pd.read_csv(o.OUT / f"{o.TAG}_gate.csv")
        meta = json.loads((o.OUT / f"{o.TAG}_gate_meta.json").read_text())
        return rc, g, meta
    rc, g, meta = run("XI_g1")
    assert rc == 0 and (g.check == "SSE").sum() > 0 and not (g.check == "범위").any() and g.ok.all()
    assert meta["compared"]["Canada|warm"]["splits"] == [1, 2] and meta["numpy_mismatch"] is True and meta["passed"] is True
    assert "numpy 2.4.6" in g.deps_ref.iloc[0] and "numpy 2.1.1" in g.deps_new.iloc[0]
    rc, g, _ = run("XI_g2", mine_splits=(1,))                                 # 분할 2 조각 없음
    assert rc == 1 and g.note.fillna("").str.contains("이번 실행 조각 없음").any()
    rc, g, _ = run("XI_g3", mine_extra=[("D1", XB.LO, "1", "cell", 100, 0, 0, 1.0)])   # 기준에 없는 키
    assert rc == 1 and g.note.fillna("").str.contains("기준 키 없음").any()
    rc, g, _ = run("XI_g4", drop=[("D0", XB.HI, "1", "cell", 100, 2, 1, 1.0)])         # 설계 범위 안 기준 키가 이번 실행에 없다
    assert rc == 1 and g.note.fillna("").str.contains("이번 실행 키 없음").any()
    rc, g, _ = run("XI_g5", versions="Alaska,Canada")                         # 알래스카 warm(등록 관문 단위)이 두 쪽 모두 없다
    assert rc == 1 and g.note.fillna("").str.contains("등록 관문 단위 없음").any()
    _clean(capsys.readouterr().out)


def _xj_src_keys(extra=()):
    ks = [("P0", "none", "1", "cell", 0, 0, -1, 0.0)]
    for n in XJ.SRC_GRID:
        for d in range(1 if n == 0 else XJ.DRAWS):
            for s in (0, 1):
                ks += [("D0", XB.LO, "1", "cell", n, d, s, 1.0), ("D1", XB.LO, "1", "cell", n, d, s, 1.0)]
                ks += [("R1", XB.LO, "1", "cell", n, d, s, float(lm)) for lm in XB.LAMS]
    return ks + list(extra)


def _xj_tgt_keys(extra=()):
    ks = [("P0", "none", "1", "cell", 0, 0, -1, 0.0)]
    ks += [("P1", "none", "1", "cell", n, d, -1, 0.0) for n in XJ.TGT_GRID for d in range(1 if n == 0 else XJ.DRAWS)]
    return ks + list(extra)


def test_u_xj_gate_coverage(out_root, capsys):
    lg, lgd = out_root / "ref_lg", out_root / "ref_lgd"
    deps = out_root / "deps_lg.log"
    deps.write_text(DEPS_LOG)
    non_design_src = [("D0", "ridge", "1", "cell", 10, 0, 0, 1.0), ("R1", XB.LO, "1", "cell", 160, 0, 0, 0.25), ("P1", "none", "1", "cell", 10, 0, -1, 0.0)]
    non_design_tgt = [("P2", "none", "1", "cell", 3, 0, -1, 0.0), ("P1", "none", "1", "cell", 40, 0, -1, 0.0)]
    for sp in (1, 2):
        XB.write_shard(lg, "lg", "Lena", "x", sp, [], [_gate_store("Lena|x", sp, _xj_src_keys(non_design_src))], {}, unit={}, expected=[1, 2])
        XB.write_shard(lgd, "lgd", XJ.TGT_ALIAS, "x", sp, [], [_gate_store(f"{XJ.TGT_ALIAS}|x", sp, _xj_tgt_keys(non_design_tgt))], {}, unit={},
                       expected=[1, 2])

    def run(name, tgt_splits=(1, 2), src_splits=(1, 2), src_extra=(), tgt_drop=()):
        o = XJ.parse_args(["--out-dir", str(out_root / name), "--gate-ref-lg", str(lg), "--gate-ref-lgd", str(lgd), "--gate-ref-lg-deps", str(deps),
                           "--splits", "2", "--src-targets", "Lena"])
        flagged = [(XJ.mname("R1", 0.3), XB.LO, "1", "cell", 10, 0, 0, 0.25)]          # w > 0 키는 관문 대상이 아니다
        for sp in src_splits:
            XB.write_shard(o.SHARDS, o.TAG, "Lena", "x", sp, [], [_gate_store("Lena|x", sp, _xj_src_keys(list(src_extra) + flagged))], {},
                           unit=dict(side="src", deps={"numpy": "2.1.1"}), expected=[1, 2])
        for sp in tgt_splits:
            keys = [k for k in _xj_tgt_keys([(XJ.mname("P1", 0.1), "none", "1", "cell", 3, 0, -1, 0.0)]) if k not in tgt_drop]
            XB.write_shard(o.SHARDS, o.TAG, XJ.TGT_ALIAS, "x", sp, [], [_gate_store(f"{XJ.TGT_ALIAS}|x", sp, keys)], {},
                           unit=dict(side="tgt", deps={"numpy": "2.1.1"}), expected=[1, 2])
        rc = XJ.gate(o)
        return rc, pd.read_csv(o.OUT / f"{o.TAG}_gate.csv"), json.loads((o.OUT / f"{o.TAG}_gate_meta.json").read_text())
    rc, g, meta = run("XJ_g1")
    assert rc == 0 and set(g.side) == {"src", "tgt"} and not (g.check == "범위").any() and meta["passed"] is True
    assert meta["compared"]["src"]["Lena"]["splits"] == [1, 2] and meta["compared"]["tgt"][XJ.TGT_ALIAS]["splits"] == [1, 2]
    assert meta["deps_ref"]["src"]["numpy_mismatch"] is True and meta["deps_ref"]["tgt"]["deps"] == {}
    rc, g, _ = run("XJ_g2", tgt_splits=(1,))                                  # 대상 쪽 분할 2 없음
    assert rc == 1 and g.note.fillna("").str.contains("이번 실행 저장소 없음").any()
    rc, g, _ = run("XJ_g3", src_extra=[("D1", XB.LO, "1", "cell", 40, 9, 0, 1.0)])     # 기준에 없는 w = 0 키
    assert rc == 1 and g.note.fillna("").str.contains("기준 키 없음").any()
    rc, g, _ = run("XJ_g4", tgt_drop=[("P1", "none", "1", "cell", 10, 4, -1, 0.0)])    # 설계 범위 안 기준 키가 없다
    assert rc == 1 and g.note.fillna("").str.contains("이번 실행 키 없음").any()
    rc, g, _ = run("XJ_g5", src_splits=())                                    # 원천 쪽을 시험하지 않으면 원천 쪽 관문은 없다
    assert rc == 0 and set(g.side) == {"tgt"}
    _clean(capsys.readouterr().out)


def test_v_smoke_threads_and_local_resources(monkeypatch):
    """허용 표지 없는 스모크는 스레드 2(5절 1a). 1절 로컬 자원: MALLOC_ARENA_MAX 가 없으면 명령행은 다시 시작하고 함수 호출은 거부한다.
    가용 메모리 30 GB 대기(1시간 상한)와 10 GB 주소 공간 상한을 건다."""
    monkeypatch.delenv("WF_RESCALE", raising=False)
    monkeypatch.delenv("LG_RESCALE", raising=False)
    assert XI.parse_args(["--smoke", "--threads", "4"]).threads == 2 and XJ.parse_args(["--smoke", "--threads", "4"]).threads == 2
    assert XI.parse_args(["--threads", "4"]).threads == 4 and XI.parse_args(["--threads", "16"]).threads == 4

    class _Exec(Exception):
        pass
    for M in (XI, XJ):
        monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
        with pytest.raises(SystemExit):
            M.local_resources(None)
        seen = {}

        def fake_exec(path, args, env, seen=seen):
            seen.update(path=path, args=list(args), env=dict(env))
            raise _Exec()
        monkeypatch.setattr(os, "execve", fake_exec)
        with pytest.raises(_Exec):
            M.local_resources(["--count-only"])
        assert seen["env"]["MALLOC_ARENA_MAX"] == "2" and seen["args"][-1] == "--count-only"
        assert Path(seen["args"][1]).name == Path(M.__file__).name
        monkeypatch.setenv("MALLOC_ARENA_MAX", "2")
        rec = []
        monkeypatch.setattr(XB, "require_memory", lambda min_gb=30.0, wait_s=0.0, poll_s=60.0: rec.append(("mem", min_gb, wait_s)) or 50.0)
        monkeypatch.setattr(XB, "limit_memory", lambda gb=10.0: rec.append(("cap", gb)) or True)
        M.local_resources(None)
        assert rec == [("mem", 30.0, 3600.0), ("cap", 10.0)]


def test_w_licence_token(tmp_path, out_root):
    """원천 쪽 약관 근거: 로컬은 작업 트리·HEAD·T0 판을 대조하고 표지를 쓴다. 계획 문서가 없는 환경은 표지가 있어야 하고 문구가 같아야 한다."""
    ref = "XJ 원천 보조 행 약관 근거 시험 문구 7a1c"
    t0 = "# plan\n## 개정 이력\n- 개정 1\n"
    wt = t0 + f"- 개정 2: {ref}\n"
    assert XJ.lic_local_check(ref, wt, wt, t0)
    with pytest.raises(SystemExit):
        XJ.lic_local_check(ref, wt, t0, t0)                                   # 커밋 전(HEAD 의 개정 이력에 없다)
    with pytest.raises(SystemExit):
        XJ.lic_local_check(ref, wt, wt, wt)                                   # T0 판에 이미 있다
    with pytest.raises(SystemExit):
        XJ.lic_local_check(ref, "# plan\n" + ref + "\n## 개정 이력\n", wt, t0)   # 개정 이력 절 밖
    plan = tmp_path / "plan.md"
    plan.write_text(wt)

    def git(*args):
        if args[0] == "rev-parse":
            return "c0ffee" * 6
        if args[0] == "show":
            return wt if args[1].startswith("HEAD:") else t0
        if args[0] == "log":
            return "abc123" * 6 + "\n"
        return ""
    lic = XJ.check_aux_lic_ref(ref, plan_path=plan, git=git)
    assert lic["ok"] and lic["method"] == "local" and lic["commit"].startswith("abc123")
    tok_path = out_root / "XJ_tempderived_aux_labels" / "inputs" / "xj_aux_lic_token.json"
    tok = XJ.write_lic_token(lic, tok_path)
    assert tok["ref_sha256"] == XJ.ref_sha256(ref) and tok["plan_commit_t0"] == XB.PLAN_COMMIT
    nop = tmp_path / "no_plan.md"
    with pytest.raises(SystemExit):
        XJ.check_aux_lic_ref(ref, token_path=None, plan_path=nop)
    with pytest.raises(SystemExit):
        XJ.check_aux_lic_ref(ref, token_path=tmp_path / "none.json", plan_path=nop)
    with pytest.raises(SystemExit):
        XJ.check_aux_lic_ref(ref + " 다른 문구", token_path=tok_path, plan_path=nop)
    r = XJ.check_aux_lic_ref(ref, token_path=tok_path, plan_path=nop)
    assert r["ok"] and r["method"] == "token" and r["commit"] == lic["commit"]
    bad = json.loads(tok_path.read_text())
    bad["plan_commit_t0"] = "0000000"
    tok_path.write_text(json.dumps(bad, ensure_ascii=False))
    with pytest.raises(SystemExit):
        XJ.check_aux_lic_ref(ref, token_path=tok_path, plan_path=nop)
    with pytest.raises(SystemExit):
        XJ.write_lic_token(dict(ok=True, method="token", ref=ref), tok_path)
    with pytest.raises(SystemExit):
        XJ.parse_args(["--write-lic-token"])                                  # 문구 없이 표지를 쓸 수 없다
