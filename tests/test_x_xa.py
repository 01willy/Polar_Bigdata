"""scripts/2_evaluation/xa_c2_gain_decomposition.py 단위 시험(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.1절 '구현·시험').

계획 2.1 의 시험 목록(계획은 파일 이름을 tests/test_xa_c2.py 로 적었다. 작업 지시의 이름 tests/test_x_xa.py 로 두었고 구현 기록에 적었다)
(a) 합성 저장소에서 재표집마다 G_recal + G_W = G_tot(셀 가중·블록 등가중). G_ML2 = min(P1 − R1, P2 − R1)(b 마다), 수준 분포와 대비 분포의 일치.
(b) 동률이 있는 Spearman: scipy.stats.spearmanr, h54._spearman_rows 와 같고, 중복도(군집 재표집의 복제)는 복제한 표본과 같다.
(c) 편순위상관(순위 잔차 방식)이 손 계산 값과 명시적 잔차 회귀와 같고, 공통 원인만 있는 합성 자료에서 0 에 가깝다.
(d) 군집 재표집의 계열 수 계산: 행 합 = 계열 수, 3계열 미만 비율이 해석적 확률과 같고, 계열 배정(주 묶음 = WF4-c 28대상 8계열,
    NAtlantic~lic 단독 계열, main27 = 계획 7계열 27대상, Russia_C(v3)는 계열 없음)이 맞다.
(e) [HEAVY] 두 재현 관문(WF4-c ρ 0.38 [0.17, 0.55], WF0 ρ 0.66·−0.08). 환경 변수 XA_HEAVY=1 일 때만 돈다. 화면에는 통과 여부만 남고,
    실패해도 표·값이 나오지 않도록 검사를 bool 로 줄여 고정 문구로 assert 한다.
그 밖(작업 지시)
(f) --count-only 는 블록 SSE·셀 수 배열과 진단값·계수 값을 읽지 않고, 화면에 라벨 유래 통계가 없다.
(g) 합성 조각 스모크: 출력 제한에 걸리는 줄을 쓰지 않으며, 결과 표는 봉인 폴더에만, 관문 표·메타는 그 위 폴더에 쓴다.
(h) --shard K/N 조각 표를 --merge-shards N 으로 합친 결과가 한 번에 계산한 결과와 같다.
(i) 결과 범주(|A| ≥ 100 조건, 군집 부족, 약한 쪽), 척도·정의 강건 표지, 해석 조각(사후 분석, 수축 잔여 의존, 두 범주, 보정 전 유의), Holm m 3,
    main27 범주. (i2) '보정 전 유의'는 방향 단측 p(음이면 p_one_neg)의 Holm 값으로 정하고, 등록 Holm 열은 p_one 을 쓴다.
(j) 관문 (1)의 재계산이 동결 h54.tests_wf4 의 WF4-c 행과 같다(합성 자료).
(k) 실행 보호: 허용 표지 없는 본 실행 거부, --skip-gates 는 스모크 전용, 스모크가 아닌 실행은 재표집 10,000회만, --threads 1–4
    (--allow-local 이어도), 스레드 환경 변수 상한, nice 10, 기본 메모리 대기 양수, 관문 (1) CI 는 스모크가 아니면 늘 대조.
(m) 내부 일치 점검: 주 묶음 점 추정 4개가 기준과 1e-9 안에서 같아야 통과하고, 실패하면 결과 표 없이 종료 코드 2.
(l) 누설 점검(XA 는 선택·적합이 없어 h54 형식의 누설 시험 대상이 아니다): CE2 는 라벨 10개 진단값만 쓰고, 대상 전체 라벨 통계를 바꿔도
    CE2 와 모든 이득 분포가 같다.
실행: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c <코어 4개> python3 -m pytest -q -p no:cacheprovider tests/test_x_xa.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "3_deep_learning"))
if "xbatch_core" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("xbatch_core", ROOT / "scripts" / "3_deep_learning" / "xbatch_core.py")
    _m = importlib.util.module_from_spec(_spec)
    sys.modules["xbatch_core"] = _m
    _spec.loader.exec_module(_m)
XB = sys.modules["xbatch_core"]
_spec = importlib.util.spec_from_file_location("xa_c2_gain_decomposition", ROOT / "scripts" / "2_evaluation" / "xa_c2_gain_decomposition.py")
XA = importlib.util.module_from_spec(_spec)
sys.modules["xa_c2_gain_decomposition"] = XA
_spec.loader.exec_module(XA)
H, W, H4 = XB.H, XB.W, XB.H4
from scipy.stats import spearmanr                                          # noqa: E402

SENTINEL = 987654.321


# ---------------------------------------------------------------- 합성 저장소와 조각
def _add(st, rng, key, level):
    st.add_sse(key, st.ncell * (level + rng.gamma(2.0, 1.0, st.nb)) ** 2, st.ncell)


def synth_tm(name="Lena|x", splits=(1, 2, 3), grid=(10, -1), draws=3, nboot=300, seed=0, drop=None):
    """합성 WF4 형식 TMx. 물리식 P1·P2 는 seed −1, W·R1 은 seed 0·1. drop = (분할, 키 묶음 이름, n, 추출) 이면 그 키를 뺀다."""
    rng = np.random.RandomState(seed)
    by = {}
    for sp in splits:
        nb = 10
        st = H4.BlockStore(name, sp, np.repeat([f"s{sp}b{j:02d}" for j in range(nb)], rng.randint(3, 20, nb)))
        _add(st, rng, H.P0_KEY, 34.0)
        for n in grid:
            for d in range(1 if n == -1 else draws):
                for m, lr, seeds, lam, lev in (("P1", "none", (-1,), 0.0, 31.0), ("P2", "none", (-1,), 0.0, 31.5),
                                               ("W", "catboost_lo", (0, 1), 0.0, 30.0), ("R1", "catboost_lo", (0, 1), 0.25, 30.5)):
                    if drop is not None and drop == (sp, m, n, d):
                        continue
                    for s in seeds:
                        _add(st, rng, (m, lr, "1", "cell", int(n), d, int(s), float(lam)), lev + rng.rand())
        by[sp] = st
    tm = W.make_tm(name, by, [dict(split=sp, dup_of=-1, valid=True, expected_splits=list(splits)) for sp in splits], nboot, False)
    return tm


SYN_TARGETS = [("Alaska", "x"), ("AL-1", "i"), ("AL-1", "x"), ("Canada", "x"), ("CA-2", "x"), ("Lena", "x"), ("LE-1", "x"), ("Russia_W", "x"),
               ("Russia_E", "x"), ("Tibet_LGD", "x"), ("Russia_C~lgd", "x"), ("NAtlantic~lic", "x")]
SMALL = ("Russia_W", "Russia_E", "Russia_C~lgd", "NAtlantic~lic")


def write_synth_shards(d, root, splits=(1, 2, 3), seed=0, full_label_shift=0.0):
    """합성 WF4 조각(xbatch_core.write_shard, 이름 wf4__cpu__<대상>__<모드>__s<분할>). 진단값·계수 값에는 표지값을 섞지 않고,
    full_label_shift 는 대상 전체 라벨 통계(y_mean, y_sd, E_A, E_B)만 바꾼다(시험 (l))."""
    rng = np.random.RandomState(seed)
    cfg = XB.make_unit_cfg("wf4", data_sha="synthetic", grid=[10, 40, 160, -1])
    for t, m in SYN_TARGETS:
        nm = f"{t}|{m}"
        small = t in SMALL
        grid = (10, -1) if small else (10, 40, 160, -1)
        bias = rng.uniform(0.0, 20.0)
        for sp in splits:
            nb = 9
            st = H4.BlockStore(nm, sp, np.repeat([f"{t}{sp}b{j}" for j in range(nb)], rng.randint(3, 12, nb)))
            _add(st, rng, H.P0_KEY, 30.0 + bias)
            for n in grid:
                for dr in range(1 if n == -1 else 3):
                    _add(st, rng, ("P1", "none", "1", "cell", n, dr, -1, 0.0), 30.0 + 0.3 * bias + rng.rand())
                    _add(st, rng, ("P2", "none", "1", "cell", n, dr, -1, 0.0), 30.0 + 0.2 * bias + rng.rand())
                    for s in (0, 1):
                        _add(st, rng, ("W", "catboost_lo", "1", "cell", n, dr, s, 0.0), 29.0 + 0.2 * bias + rng.rand())
                        _add(st, rng, ("R1", "catboost_lo", "1", "cell", n, dr, s, 0.25), 29.0 + 0.25 * bias + rng.rand())
            b10 = bias + rng.randn(5)
            unit = dict(diag=[dict(n=10, draw=i, n_lab=10, bias=float(b), abs_bias=float(abs(b))) for i, b in enumerate(b10)],
                        n_A=15 if small else 300, E0=1.5, E_own=float(1.5 * np.exp(0.01 * bias)), logE_ratio_own=float(0.01 * bias),
                        y_mean=50.0 + full_label_shift, y_sd=10.0 + full_label_shift, E_A=1.4 + full_label_shift, E_B=1.6 + full_label_shift)
            XB.write_shard(d, "wf4", t, m, sp, [dict(method="P0", n=0, target=t, mode=m)], st, cfg, unit=unit, expected=list(splits),
                           allowed=[root])
    return d


@pytest.fixture()
def shards(tmp_path, monkeypatch):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    return write_synth_shards(tmp_path / "shards", tmp_path)


def _smoke_argv(shards_dir, out_root, *extra, nboot=60):
    return ["--smoke", "--skip-gates", "--wf4-shards", str(shards_dir), "--out-root", str(out_root), "--smoke-nboot", str(nboot),
            "--mem-min", "0", "--mem-max", "0", "--threads", "2", *extra]


# ---------------------------------------------------------------- (a) 가법성
@pytest.mark.parametrize("n", [10, -1])
def test_a_additivity_and_ml2(n):
    tm = synth_tm(nboot=400)
    g, chk = XA.gains(tm, n)
    assert set(g) == set(XA.GAIN_NAMES)
    for suf in ("", "_beq"):
        lhs = g["G_tot"]["dist" + suf]
        rhs = g["G_recal"]["dist" + suf] + g["G_W"]["dist" + suf]
        assert lhs.shape == (400,) and np.max(np.abs(lhs - rhs)) <= 1e-10, "재표집마다 G_tot = G_recal + G_W"
        assert abs(g["G_tot"]["pt" + suf] - g["G_recal"]["pt" + suf] - g["G_W"]["pt" + suf]) <= 1e-10
    assert chk["additive_checked"] and chk["additive_same_splits"] and chk["additive_max_abs"] <= 1e-10
    k = XA.keys_for(n)
    r2 = H.contrast(tm, k["P2"], k["R1"], return_dist=True)
    r1 = H.contrast(tm, k["P1"], k["R1"], return_dist=True)
    for suf in ("", "_beq"):
        assert np.array_equal(g["G_ML2"]["dist" + suf], np.minimum(r1["dist" + suf], r2["dist" + suf]))
    assert g["G_ML2"]["pt"] == min(r1["delta"], r2["delta"]) and np.all(g["G_ML2"]["dist"] <= g["G_ML"]["dist"] + 1e-12)
    # 수준 분포(척도 (ii)의 RMSE(P0)^b)가 h40.contrast 의 블록 가중과 같은 재표집을 쓴다
    l0, l1 = XA.level_dist(tm, H.P0_GRP), XA.level_dist(tm, k["P1"])
    for suf in ("", "_beq"):
        assert np.max(np.abs(g["G_recal"]["dist" + suf] - (l0["dist" + suf] - l1["dist" + suf]))) <= 1e-10
        assert np.max(np.abs(g["G_recal"]["p0_dist" + suf] - l0["dist" + suf])) <= 1e-12
    assert abs(g["G_recal"]["p0"] - l0["pt"]) <= 1e-12


def test_a2_missing_key_is_flagged():
    """W 의 한 추출이 빠지면 h40.contrast 가 공통 추출로 줄이므로 가법성이 깨질 수 있다. 점검 표가 그것을 드러낸다(조용히 통과하지 않는다)."""
    tm = synth_tm(nboot=200, drop=(2, "W", 10, 1))
    _, chk = XA.gains(tm, 10)
    assert chk["additive_checked"] and chk["additive_max_abs"] > 1e-9


# ---------------------------------------------------------------- (b) 동률 Spearman
def test_b_spearman_ties_and_multiplicity():
    rng = np.random.RandomState(3)
    x = rng.randint(0, 5, 25).astype(float); y = rng.randint(0, 4, 25).astype(float)
    r = XA.wspearman(x[None], y[None])[0]
    assert abs(r - spearmanr(x, y).correlation) < 1e-12
    assert abs(r - W._spearman_rows(x[None], y[None])[0]) < 1e-12
    m = rng.randint(0, 4, 25).astype(float)                                  # 군집 재표집의 중복도(0 = 뽑히지 않음)
    xe, ye = np.repeat(x, m.astype(int)), np.repeat(y, m.astype(int))
    assert abs(XA.wspearman(x[None], y[None], m[None])[0] - spearmanr(xe, ye).correlation) < 1e-12
    X2 = rng.randn(50, 25); Y2 = X2 + rng.randn(50, 25); M2 = rng.randint(0, 3, (50, 25)).astype(float)
    got = XA.wspearman(X2, Y2, M2, chunk=7)
    for b in (0, 17, 49):
        ok = M2[b] > 0
        if ok.sum() >= 3:
            ref = spearmanr(np.repeat(X2[b], M2[b].astype(int)), np.repeat(Y2[b], M2[b].astype(int))).correlation
            assert abs(got[b] - ref) < 1e-12
    xn = x.copy(); xn[[0, 3]] = np.nan
    keep = np.isfinite(xn)
    assert abs(XA.wspearman(xn[None], y[None])[0] - spearmanr(x[keep], y[keep]).correlation) < 1e-12
    assert np.isnan(XA.wspearman(np.array([[1.0, 2.0, np.nan]]), np.array([[2.0, 1.0, 3.0]]))[0]), "유효 항목 3 미만은 NaN"


# ---------------------------------------------------------------- (c) 편순위상관
def test_c_partial_spearman_known_and_residual():
    x = np.array([1, 2, 3, 4, 5.0]); y = np.array([2, 1, 4, 3, 5.0]); z = np.array([1, 3, 2, 5, 4.0])
    want = (0.8 - 0.8 * 0.3) / np.sqrt((1 - 0.64) * (1 - 0.09))               # ρ_xy 0.8, ρ_xz 0.8, ρ_yz 0.3(손 계산)
    assert abs(XA.partial_spearman(x, y, z) - want) < 1e-12

    def resid_method(x, y, z, w):
        from scipy.stats import rankdata
        xe, ye, ze = (np.repeat(v, w.astype(int)) for v in (x, y, z))
        rx, ry, rz = rankdata(xe), rankdata(ye), rankdata(ze)
        A = np.column_stack([np.ones_like(rz), rz])
        ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
        ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        return float(np.corrcoef(ex, ey)[0, 1])
    rng = np.random.RandomState(5)
    for _ in range(5):
        x, y, z = rng.randn(3, 30)
        y = y + 0.5 * x + 0.8 * z
        w = rng.randint(1, 4, 30).astype(float)
        assert abs(XA.partial_spearman(x, y, z) - resid_method(x, y, z, np.ones(30))) < 1e-10
        assert abs(XA.partial_spearman(x, y, z, w) - resid_method(x, y, z, w)) < 1e-10
    z = rng.randn(3000); x = z + 0.5 * rng.randn(3000); y = z + 0.5 * rng.randn(3000)
    assert XA.wspearman(x[None], y[None])[0] > 0.6 and abs(XA.partial_spearman(x, y, z)) < 0.06, "공통 원인 z 만 있으면 편순위상관은 0 근처"


# ---------------------------------------------------------------- (d) 군집 재표집
def test_d_cluster_counts_and_families():
    B = 20000
    c3 = XA.cluster_counts(("a", "b", "c"), B)
    assert np.all(c3.sum(1) == 3) and np.array_equal(c3, XA.cluster_counts(("a", "b", "c"), B))
    assert abs(np.mean((c3 > 0).sum(1) < 3) - (1 - 6 / 27)) < 0.01
    fams = ["알래스카"] * 3 + ["캐나다"] * 2 + ["레나", "러시아 W", "러시아 E", "티베트", "러시아 C"]
    cb = XA.cluster_boot(fams, B)
    p7 = (7 + 21 * (2 ** 7 - 2)) / 7 ** 7                                     # 서로 다른 계열 ≤ 2 의 확률
    assert len(cb["families"]) == 7 and abs(cb["frac_lt3"] - p7) < 0.002 and not cb["insufficient"]
    assert np.array_equal(cb["M"][:, 0], cb["M"][:, 1]) and np.array_equal(cb["M"][:, 3], cb["M"][:, 4]), "같은 계열의 대상은 같은 중복도"
    cnt = XA.cluster_counts(cb["families"], B)                               # 계열 순서: 알래스카(3), 캐나다(2), 나머지 1
    assert np.array_equal(cb["M"].sum(1), cnt @ np.array([3, 2, 1, 1, 1, 1, 1.0])) and np.all(cnt.sum(1) == 7)
    cb3 = XA.cluster_boot(["알래스카", "캐나다", "레나", "레나"], 5000)
    assert cb3["insufficient"] and len(cb3["families"]) == 3
    # 계열 배정(실제 이름): 주 묶음 = WF4-c 28대상 8계열(계획 7계열 + NAtlantic~lic 단독), main27 = 계획 7계열 27대상(등록 밖 민감도).
    # Russia_C(v3, 점 추정만인 LG 대상)와 Greenland 는 계열이 없다(진단값이 생겨도 주 묶음에 들어가지 않는다)
    names = ([f"AL-{i}|{m}" for i in range(1, 7) for m in "ix"] + ["Alaska|x"] + [f"{s}|{m}" for s in ("CA-2", "CA-3", "LE-1", "LE-2") for m in "ix"]
             + ["Canada|x", "Lena|x", "Russia_W|x", "Russia_E|x", "Tibet_LGD|x", "Russia_C~lgd|x", "NAtlantic~lic|x", "Russia_C|x", "Greenland|x"])
    T = {nm: dict(has_ce2=nm != "Greenland|x", nA_min=(20 if nm.split("|")[0] in SMALL + ("Tibet_LGD", "Russia_C") else 150)) for nm in names}
    assert XA.family_of("Russia_C|x") is None and XA.family_of("Russia_C~lgd|x") == "러시아 C"
    assert XA.family_of("NAtlantic~lic|x") == XA.EXTRA_FAMILY and XA.family_of("Greenland|x") is None
    main = XA.subset_members(T, "main")
    assert len(main) == 28 and {f for _, f in main} == set(XA.FAMILY_ORDER8) and any(nm == "NAtlantic~lic|x" for nm, _ in main)
    assert all(nm not in ("Russia_C|x", "Greenland|x") for nm, _ in main), "진단값이 있어도 계열 밖 대상은 주 묶음에 없다"
    m27 = XA.subset_members(T, "main27")
    assert len(m27) == 27 and {f for _, f in m27} == set(XA.FAMILY_ORDER) and all(nm != "NAtlantic~lic|x" for nm, _ in m27)
    assert len(XA.subset_members(T, "x_only")) == 18 and len(XA.subset_members(T, "macro7")) == 7
    assert {f for _, f in XA.subset_members(T, "a100")} == {"알래스카", "캐나다", "레나"} and len(XA.subset_members(T, "a100")) == 23
    assert all(f != "티베트" for _, f in XA.subset_members(T, "no_tibet")) and len(XA.subset_members(T, "no_tibet")) == 27
    cb8 = XA.cluster_boot([f for _, f in main], B)
    p8 = (8 + 28 * (2 ** 8 - 2)) / 8 ** 8                                     # 8계열에서 서로 다른 계열 ≤ 2 의 확률(0.042 %)
    assert len(cb8["families"]) == 8 and abs(cb8["frac_lt3"] - p8) < 0.001 and not cb8["insufficient"]
    sp = [XA.hyp_of(dict(gain="G_recal", x="CE2", scale="cm", n=-1, subset=s)) for s in ("main", "main27", "a100")]
    assert sp == ["XA-1", "등록 밖", "XA-5"] and XA.SUBSET_REGISTERED["main"] and not XA.SUBSET_REGISTERED["main27"]


# ---------------------------------------------------------------- (e) [HEAVY] 재현 관문
@pytest.mark.skipif(os.environ.get("XA_HEAVY", "") != "1", reason="[HEAVY] XA_HEAVY=1 일 때만(실제 WF4 조각, 재표집 10,000회)")
def test_e_heavy_gates(tmp_path, monkeypatch, capsys):
    if not XA.WF4_SHARDS.exists():
        pytest.skip("WF4 조각 없음")
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    rc = XA.main(["--gates-only", "--allow-local", "--out-root", str(tmp_path), "--mem-min", "0", "--mem-max", "0", "--threads", "2"])
    out = capsys.readouterr()
    # 계획 0.3: 실제 자료 시험은 실패해도 표·값을 화면에 내지 않는다. 검사마다 먼저 bool 로 줄이고 고정 문구로만 실패를 알린다
    ok_out = bool("[출력 제한]" not in out.out + out.err)
    del out
    assert ok_out, "출력 제한에 걸린 줄이 있다"
    meta = json.loads((tmp_path / XA.EXP_NAME / "xa_meta.json").read_text())
    ok_meta = bool(rc == 0 and meta["gates"]["gate1"] and meta["gates"]["gate2"] and int(meta["gates"]["gate1_n_targets"]) == 28
                   and int(meta["nboot"]) == 10000)
    del meta
    assert ok_meta, "관문 메타가 기대와 다르다(종료 코드, 관문 통과, 대상 수, 재표집 수)"
    g = pd.read_csv(tmp_path / XA.EXP_NAME / "xa_gates.csv")                  # 기존 등록 값의 재현 표(새 결과가 아니다). 값은 화면에 쓰지 않는다
    ok_rows = bool(len(g) == 5)
    ok_comp = bool(g.compared.astype(bool).all())
    ok_eq = bool(g.equal_2dp.astype(bool).all())
    del g
    assert ok_rows, "관문 표 행 수가 5 가 아니다"
    assert ok_comp, "관문 표에 대조하지 않은 행이 있다"
    assert ok_eq, "관문 표에 소수 둘째 자리까지 같지 않은 행이 있다"


# ---------------------------------------------------------------- (f) 세기: 라벨 유래 값을 읽지도 쓰지도 않는다
def test_f_count_only_reads_structure_only(shards, monkeypatch, capsys):
    for p in shards.glob("*_unit.json"):                                      # 진단값·계수 값에 표지값을 넣는다
        u = json.loads(p.read_text())
        for v in u["diag"]:
            v["bias"] = v["abs_bias"] = SENTINEL
        u.update(E0=SENTINEL, E_own=SENTINEL, logE_ratio_own=SENTINEL, y_mean=SENTINEL)
        p.write_text(json.dumps(u))
    seen = []
    real_load = np.load

    class Rec:
        def __init__(self, z):
            self.z = z

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.z.close()

        def __getitem__(self, k):
            seen.append(k)
            return self.z[k]
    monkeypatch.setattr(XA.np, "load", lambda *a, **k: Rec(real_load(*a, **k)))
    rc = XA.main(["--count-only", "--wf4-shards", str(shards)])
    out = capsys.readouterr()
    txt = out.out + out.err
    assert rc == 0 and "조각 36개" in txt
    line_main = next(ln for ln in txt.splitlines() if "묶음 main(" in ln)
    line_m27 = next(ln for ln in txt.splitlines() if "묶음 main27(" in ln)
    assert "대상 12, 계열 8" in line_main and "대상 11, 계열 7" in line_m27
    assert seen and not any(k.endswith(("_sse", "_cnt", "_ncell")) for k in seen), "블록 SSE·셀 수 배열을 열지 않는다"
    assert "987654" not in txt and XB.FORBIDDEN_OUT.search(txt) is None and "[출력 제한]" not in txt
    assert "묶음 main" in txt and "적합 0건" in txt


# ---------------------------------------------------------------- (g) 스모크: 출력 제한과 봉인
def test_g_smoke_outputs_sealed(shards, tmp_path, capsys):
    root = tmp_path / "out"
    rc = XA.main(_smoke_argv(shards, root))
    out = capsys.readouterr()
    txt = out.out + out.err
    assert rc == 0 and "[출력 제한]" not in txt and XB.FORBIDDEN_OUT.search(txt) is None, "걸러야 할 줄을 쓰지 않는다"
    sd = XB.sealed_dir(XA.EXP_NAME, root)
    want = {"xa_targets.csv", "xa_gains.csv", "xa_spearman.csv", "xa_categories.csv", "xa_hyp.csv", "xa_decomp.csv", "xa_pool_main4.csv",
            "xa_additivity.csv"}
    assert want <= {p.name for p in sd.iterdir()}
    man = json.loads((sd / "sealed_manifest.json").read_text())
    assert want <= {e["file"] for e in man["entries"]}
    meta = json.loads((root / XA.EXP_NAME / "xa_meta.json").read_text())
    assert meta["status"] == "완료" and meta["n_fits"] == 0 and meta["additivity"]["passed"] and meta["nboot"] == 60
    assert not any((root / XA.EXP_NAME).glob("xa_spearman*.csv")), "결과 표는 봉인 폴더 밖에 쓰지 않는다"
    sp = pd.read_csv(sd / "xa_spearman.csv")                                  # 합성 자료(시험 안에서만 연다)
    assert len(sp) == len(XA.spec_list()) == 864
    m = sp[(sp.subset == "main") & (sp.n == -1) & (sp.x == "CE2") & (sp.scale == "cm")]
    assert set(m.hyp) == {"XA-1", "XA-2", "XA-3", "XA-3a", "XA-4", "서술"} and (m.n_targets == 12).all() and (m.n_families == 8).all()
    assert m.targets.str.contains("NAtlantic~lic|x", regex=False).all() and (m.registered == True).all()   # noqa: E712
    m27 = sp[sp.subset == "main27"]
    assert (m27.registered == False).all() and (m27.hyp == "등록 밖").all() and (m27[m27.n == -1].n_targets == 11).all()   # noqa: E712
    assert {"cluster_p_one_neg_cell", "cluster_p_one_neg_beq", "cluster_p_one_neg_max", "fixed_p_one_neg_max"} <= set(sp.columns)
    hy = pd.read_csv(sd / "xa_hyp.csv")
    assert list(hy.hyp) == ["XA-1", "XA-2", "XA-3", "XA-3a", "XA-4", "C2 문장"]
    assert all("사후 분석" in s for s in hy.sentence.dropna() if s)
    assert (hy.deviation == XA.DEVIATION_MAIN).all() and {"cat_main27", "holm_p_dir", "p_one_neg_max", "direction"} <= set(hy.columns)
    ct = pd.read_csv(sd / "xa_categories.csv")
    assert "cat_main27" in ct.columns and len(ct) == 144
    tg = pd.read_csv(sd / "xa_targets.csv")
    assert int(tg.in_main.sum()) == 12 and int(tg.in_main27.sum()) == 11 and int(tg.in_a100.sum()) == 8
    assert int(tg.plan_family7.sum()) == 11 and set(tg.family) == set(XA.FAMILY_ORDER8)


# ---------------------------------------------------------------- (h) 조각 나눔과 합침
def test_h_shard_merge_equals_single(shards, tmp_path, capsys):
    r1, r2 = tmp_path / "one", tmp_path / "two"
    assert XA.main(_smoke_argv(shards, r1, nboot=40)) == 0
    for k in range(3):
        assert XA.main(_smoke_argv(shards, r2, "--shard", f"{k}/3", nboot=40)) == 0
    names = {p.name for p in XB.sealed_dir(XA.EXP_NAME, r2).iterdir()}
    assert {"xa_spearman_part0of3.csv", "xa_spearman_part2of3.json"} <= names
    assert not any("__cpu__" in nm for nm in names), "작업 나눔의 부분 표는 계획 1절의 적합 조각 이름을 쓰지 않는다"
    assert XA.main(_smoke_argv(shards, r2, "--merge-shards", "3", nboot=40)) == 0
    capsys.readouterr()
    for f in ("xa_spearman.csv", "xa_hyp.csv", "xa_categories.csv"):
        a = pd.read_csv(XB.sealed_dir(XA.EXP_NAME, r1) / f)
        b = pd.read_csv(XB.sealed_dir(XA.EXP_NAME, r2) / f)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    with pytest.raises(SystemExit):                                          # 설정이 다른 조각 표는 합치지 않는다
        XA.main(_smoke_argv(shards, r2, "--merge-shards", "3", nboot=41))


# ---------------------------------------------------------------- (i) 범주, 강건 표지, 해석 조각, Holm
def _row(gain, x, scale, subset, lo, hi, n=-1, insufficient=False, p_one=0.01, rho=0.4, p_one_neg=None):
    p_neg = (1.0 - p_one) if p_one_neg is None else p_one_neg
    r = dict(gain=gain, x=x, scale=scale, n=n, subset=subset, status="ok", cluster_insufficient=insufficient, n_targets=28, n_families=8,
             cluster_frac_lt3=0.0004, rho_cell=rho, rho_beq=rho)
    for k in ("cluster", "fixed"):
        for w in ("cell", "beq"):
            r.update({f"{k}_lo_{w}": lo, f"{k}_hi_{w}": hi, f"{k}_p_one_{w}": p_one, f"{k}_p_one_neg_{w}": p_neg})
        r[f"{k}_p_one_max"] = p_one
        r[f"{k}_p_one_neg_max"] = p_neg
    return r


def _grid(spec, pvals=None, main27=None):
    """spec = {(이득, x, 척도): (주 lo, hi, a100 lo, hi)} 로 전량 행을 만든다. 나머지는 확인하지 못함. pvals = {(이득, x, 척도): (p_one, p_one_neg, ρ)}
    는 주 묶음 행의 p 와 점 추정을 바꾼다. main27 = {(이득, x, 척도): (lo, hi)} 이면 main27 행을 더한다."""
    rows = []
    for g in XA.GAIN_NAMES:
        for x in XA.X_KINDS:
            for sc in XA.SCALES:
                lo, hi, blo, bhi = spec.get((g, x, sc), (-0.1, 0.5, -0.1, 0.5))
                p1, pn, rho = (pvals or {}).get((g, x, sc), (0.01, None, 0.4))
                rows += [_row(g, x, sc, "main", lo, hi, p_one=p1, p_one_neg=pn, rho=rho), _row(g, x, sc, "a100", blo, bhi)]
                if main27 and (g, x, sc) in main27:
                    rows.append(_row(g, x, sc, "main27", *main27[(g, x, sc)]))
    return pd.DataFrame(rows)


def test_i_categories_fragments_holm():
    pos, none = (0.1, 0.6, 0.05, 0.7), (-0.1, 0.5, -0.2, 0.6)
    spec = {("G_recal", "CE2", sc): pos for sc in XA.SCALES}
    spec[("G_recal", "CE1", "cm")] = pos                                     # XA-1: 양, 정의 강건
    spec[("G_W", "CE2", "cm")] = pos; spec[("G_W", "CE2", "rel")] = pos       # XA-2: 양, 척도 (iii) 다름 → 척도 강건 아님
    spec[("G_ML2", "CE2", "cm")] = pos                                        # XA-3: 양, 일부만 양 / XA-3a(G_ML): 확인하지 못함 → 수축 잔여 의존
    spec[("G_shr", "CE2", "cm")] = (0.1, 0.6, -0.05, 0.7)                     # XA-4: |A| ≥ 100 조건 불충족 → 확인하지 못함
    sp = _grid(spec)
    ct = XA.category_table(sp)
    vt = XA.verdict_table(sp, ct).set_index("hyp")
    assert vt.loc["XA-1", "category"] == XA.CAT_POS and vt.loc["XA-1", "definition_robust"] and vt.loc["XA-1", "scale_robust"]
    assert "원천 계수의 오차가 클수록" in vt.loc["XA-1", "sentence"] and "사후 분석" in vt.loc["XA-1", "sentence"]
    assert vt.loc["XA-2", "category"] == XA.CAT_POS and not vt.loc["XA-2", "scale_robust"]
    assert "척도 또는 정의를 바꾸면 확인하지 못했다" in vt.loc["XA-2", "sentence"]
    assert vt.loc["XA-3", "category"] == XA.CAT_POS and not vt.loc["XA-3", "all_variants_pos"]
    assert "정의에 따라 달랐다" in vt.loc["XA-3", "sentence"] and "수축 잔여 의존" in vt.loc["XA-3", "sentence"]
    assert vt.loc["XA-3a", "category"] == XA.CAT_NONE and vt.loc["XA-4", "category"] == XA.CAT_NONE
    c2 = vt.loc["C2 문장", "sentence"]
    assert c2.index("원천 계수") < c2.index("라벨 안 교차검증") < c2.index("수축 잔여를 뺀") and c2.count("사후 분석") >= 3
    assert vt.loc["XA-1", "holm_m"] == 3 and abs(vt.loc["XA-1", "holm_p"] - 0.03) < 1e-12 and np.isnan(vt.loc["XA-4", "holm_p"])
    # Holm 보정 p 0.03 ≥ 0.025 인 양 범주: '보정 전 유의'가 holm_note, 가설 문장, C2 문장에 모두 들어간다(1절 '다중성', '해석 문장의 다섯 갈래')
    for h in ("XA-1", "XA-2", "XA-3"):
        assert vt.loc[h, "holm_note"].startswith(XB.UNCORRECTED_TXT) and XB.UNCORRECTED_TXT in vt.loc[h, "sentence"]
        assert vt.loc[h, "direction"] == "양" and abs(vt.loc[h, "holm_p_dir"] - 0.03) < 1e-12
    assert c2.count(XB.UNCORRECTED_TXT) == 3 and XB.UNCORRECTED_TXT not in vt.loc["XA-4", "sentence"]
    assert (vt.deviation == XA.DEVIATION_MAIN).all()
    # Holm 보정 p 가 0.025 미만이면 표지가 없다
    spl = _grid(spec, pvals={("G_recal", "CE2", "cm"): (0.001, None, 0.4), ("G_W", "CE2", "cm"): (0.001, None, 0.4),
                             ("G_ML2", "CE2", "cm"): (0.001, None, 0.4)})
    vl = XA.verdict_table(spl, XA.category_table(spl)).set_index("hyp")
    assert all(vl.loc[h, "holm_note"] == "" and XB.UNCORRECTED_TXT not in vl.loc[h, "sentence"] for h in ("XA-1", "XA-2", "XA-3"))
    assert XB.UNCORRECTED_TXT not in vl.loc["C2 문장", "sentence"]
    # 음의 범주, 군집 부족, 주·보조 CI 의 범주 차이(약한 쪽)
    neg = (-0.6, -0.1, -0.7, -0.05)
    sp2 = _grid({("G_recal", "CE2", "cm"): neg})
    assert XA.verdict_table(sp2, XA.category_table(sp2)).set_index("hyp").loc["XA-1", "directive"] == "C2 를 철회한다"
    sp3 = _grid({("G_recal", "CE2", "cm"): pos})
    sp3.loc[(sp3.gain == "G_recal") & (sp3.x == "CE2") & (sp3.scale == "cm") & (sp3.subset == "main"), "cluster_insufficient"] = True
    v3 = XA.verdict_table(sp3, XA.category_table(sp3)).set_index("hyp")
    assert v3.loc["XA-1", "category"] == XA.CAT_NA_CLUSTER and "판정할 수 없었다(군집 부족)" in v3.loc["XA-1", "sentence"]
    sp4 = _grid({("G_recal", "CE2", "cm"): pos})
    sel = (sp4.gain == "G_recal") & (sp4.x == "CE2") & (sp4.scale == "cm") & (sp4.subset == "main")
    sp4.loc[sel, "fixed_lo_cell"] = -0.01
    v4 = XA.verdict_table(sp4, XA.category_table(sp4)).set_index("hyp")
    assert v4.loc["XA-1", "cat_cluster"] == XA.CAT_POS and v4.loc["XA-1", "cat_fixed"] == XA.CAT_NONE and v4.loc["XA-1", "category"] == XA.CAT_NONE
    assert "보조 CI(대상 고정) 확인하지 못함" in v4.loc["XA-1", "sentence"]
    assert XA.weaker(XA.CAT_POS, XA.CAT_NEG) == XA.CAT_NONE and XA.weaker(XA.CAT_NA_CLUSTER, XA.CAT_POS) == XA.CAT_NA_CLUSTER
    # main27(등록 밖 민감도)의 범주는 cat_main27 열에만 남고 주 범주를 바꾸지 않는다
    sp5 = _grid({("G_recal", "CE2", "cm"): pos}, main27={("G_recal", "CE2", "cm"): (-0.1, 0.5)})
    v5 = XA.verdict_table(sp5, XA.category_table(sp5)).set_index("hyp")
    assert v5.loc["XA-1", "category"] == XA.CAT_POS and v5.loc["XA-1", "cat_main27"] == XA.CAT_NONE
    assert v4.loc["XA-1", "cat_main27"] == XA.CAT_NA_ROWS, "main27 행이 없으면 판정 불가(행 없음)"


def test_i2_direction_of_uncorrected_note():
    """음 범주의 '보정 전 유의'는 p_one = P(ρ^b ≤ 0)(≈ 1)이 아니라 방향 단측 p_one_neg = P(ρ^b ≥ 0)로 정한다(1절 'p 값': 방향은 점 추정).
    등록 Holm 입력(holm_p)은 p_one 그대로다."""
    neg = (-0.6, -0.1, -0.7, -0.05)
    sp = _grid({("G_recal", "CE2", "cm"): neg}, pvals={("G_recal", "CE2", "cm"): (0.999, 0.001, -0.5)})
    vt = XA.verdict_table(sp, XA.category_table(sp)).set_index("hyp")
    assert vt.loc["XA-1", "category"] == XA.CAT_NEG and vt.loc["XA-1", "direction"] == "음"
    assert abs(vt.loc["XA-1", "p_dir"] - 0.001) < 1e-12 and abs(vt.loc["XA-1", "holm_p_dir"] - 0.003) < 1e-12
    assert vt.loc["XA-1", "holm_p"] > 0.9, "등록 Holm 열은 p_one 을 입력으로 쓴다"
    assert vt.loc["XA-1", "holm_note"] == "" and XB.UNCORRECTED_TXT not in vt.loc["XA-1", "sentence"]
    # 방향 단측 p 의 Holm 보정 값이 0.025 이상이면 음 범주에도 붙는다: (0.02, 0.01, 0.01) → XA-1 Holm 0.03
    sp2 = _grid({("G_recal", "CE2", "cm"): neg}, pvals={("G_recal", "CE2", "cm"): (0.98, 0.02, -0.5)})
    v2 = XA.verdict_table(sp2, XA.category_table(sp2)).set_index("hyp")
    assert abs(v2.loc["XA-1", "holm_p_dir"] - 0.03) < 1e-12 and v2.loc["XA-1", "holm_note"].startswith(XB.UNCORRECTED_TXT)
    assert XB.UNCORRECTED_TXT in v2.loc["XA-1", "sentence"] and XB.UNCORRECTED_TXT in v2.loc["C2 문장", "sentence"]
    # 확인하지 못함 범주의 방향은 셀 가중 점 추정의 부호, 표지는 붙지 않는다
    assert XA.direction_of(XA.CAT_NONE, dict(rho_cell=-0.2)) == -1 and XA.direction_of(XA.CAT_NONE, dict(rho_cell=0.0)) == 1
    assert XA.direction_of(XA.CAT_POS, dict(rho_cell=-0.2)) == 1 and XA.direction_of(XA.CAT_NEG, None) == -1
    assert XA._ci(np.array([-1.0, 0.0, 1.0, 2.0, np.nan]))[2:] == (0.5, 0.75, 4)


# ---------------------------------------------------------------- (j) 관문 (1)이 동결 h54 의 WF4-c 와 같다
def test_j_gate_wf4c_equals_frozen_h54(shards):
    tms, units = XA.load_wf4_tms(shards, "wf4", 300)
    g = XA.gate_wf4c(tms, units, 300)
    a = XB.h54_args(nboot=300)
    df = W.tests_wf4(a, tms, pd.DataFrame(), units)
    r = df[(df.test_id == "WF4-c") & (df.scope == "MEAN")].iloc[0]
    assert g["n_targets"] == int(r.n_targets) == 12
    for k in ("rho", "ci_lo", "ci_hi", "p_one"):
        assert abs(g[k] - float(r[k])) < 1e-12, k


# ---------------------------------------------------------------- (k) 실행 보호
def test_k_run_guard(shards, monkeypatch):
    monkeypatch.delenv("WF_RESCALE", raising=False); monkeypatch.delenv("LG_RESCALE", raising=False)
    with pytest.raises(SystemExit):
        XA.main(["--wf4-shards", str(shards), "--mem-min", "0", "--mem-max", "0"])
    with pytest.raises(SystemExit):
        XA.main(["--skip-gates", "--allow-local", "--wf4-shards", str(shards)])
    a = XA.parse_args(["--smoke", "--nboot", "10000"])
    assert XB.local_limits(a.threads, a.nboot, argv=["--smoke"])[1] == XB.LOCAL_NBOOT_MAX
    # 스모크가 아닌 실행(본 실행, 관문만, 합침)은 재표집 10,000회만 허용한다. 자료를 읽기 전에 멈춘다
    for extra in ([], ["--gates-only"], ["--merge-shards", "2"]):
        with pytest.raises(SystemExit):
            XA.main(["--allow-local", "--nboot", "500", "--wf4-shards", str(shards), "--mem-min", "0", "--mem-max", "0", "--threads", "2", *extra])
    # 스레드: --allow-local 이어도 4 를 넘지 못한다(계획 2.1). 기본 메모리 대기는 양수(30 GB 아래면 기다린다)
    for argv in (["--allow-local", "--threads", "8"], ["--threads", "5"], ["--smoke", "--threads", "0"]):
        with pytest.raises(SystemExit):
            XA.parse_args(argv)
    assert XA.parse_args([]).threads <= XA.XA_MAX_THREADS and XA.parse_args([]).mem_wait >= 600
    env = {"OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "x"}
    assert XA.clamp_thread_env(["prog", "--allow-local", "--threads", "8"], env) == 4
    assert env == {k: "4" for k in XA._THREAD_VARS}
    env = {"OMP_NUM_THREADS": "2"}
    assert XA.clamp_thread_env(["prog", "--threads=3"], env) == 3 and env["OMP_NUM_THREADS"] == "2" and env["OPENBLAS_NUM_THREADS"] == "3"
    assert XA.ensure_nice() >= XA.XA_NICE and os.nice(0) >= XA.XA_NICE
    assert XA.thread_cap(8) <= XA.XA_MAX_THREADS and XA.thread_cap(1) == 1
    # 관문 (1)의 CI 는 스모크가 아니면 재표집 수와 관계없이 대조한다(값은 쓰지 않고 대조 여부만 본다)
    monkeypatch.setattr(XA, "_file_refs", lambda: {})
    tms, units = XA.load_wf4_tms(shards, "wf4", 200)
    df_full, _ = XA.run_gates(tms, units, 200, smoke=False, skip_wf0=True)
    df_smoke, _ = XA.run_gates(tms, units, 200, smoke=True, skip_wf0=True)
    ok_full = bool(df_full.compared.astype(bool).all() and len(df_full) == 3)
    ok_smoke = bool(df_smoke.compared.astype(bool).tolist() == [True, False, False])
    assert ok_full and ok_smoke, "관문 (1) 대조 범위가 다르다"


# ---------------------------------------------------------------- (m) 내부 일치 점검(주 묶음 점 추정 = WF4c_28 기록)
def test_m_internal_check_wf4c28(shards, tmp_path, monkeypatch, capsys):
    root = tmp_path / "out"
    assert XA.main(_smoke_argv(shards, root, nboot=40)) == 0
    sp = pd.read_csv(XB.sealed_dir(XA.EXP_NAME, root) / "xa_spearman.csv")   # 합성 자료(시험 안에서만 연다)
    ref = {g: (float(XA._rec(sp, g, "CE2", "cm", -1, "main")["rho_cell"]), 12) for g in XA.C2_REF_ROWS}
    assert XA.check_wf4c28(sp, ref)["passed"] and XA.check_wf4c28(sp, ref)["n_compared"] == 4
    bad = dict(ref, G_W=(ref["G_W"][0] + 1e-6, 12))
    r = XA.check_wf4c28(sp, bad)
    assert not r["passed"] and r["gains_failed"] == ["G_W"]
    assert not XA.check_wf4c28(sp, dict(ref, G_ML=(ref["G_ML"][0], 11)))["passed"], "대상 수가 다르면 실패"
    assert not XA.check_wf4c28(sp, {})["passed"], "기준 행이 없으면 실패"
    part = sp[sp.gain != "G_W"]
    assert not XA.check_wf4c28(part, ref)["passed"] and XA.check_wf4c28(part, ref, require_all=False)["passed"]
    # 기준 파일 읽기(WF4c_28 블록, 이득 4종)
    f = tmp_path / "ref.csv"
    pd.DataFrame([dict(block=XA.C2_REF_BLOCK, x=XA.C2_REF_X, y=y, n_targets=12, rho=ref[g][0], p_two_sided=0.5, note="")
                  for g, y in XA.C2_REF_ROWS.items()]).to_csv(f, index=False)
    assert XA.c2_ref_values(f) == ref and XA.c2_ref_values(tmp_path / "없음.csv") == {}
    # 실행 경로: 관문을 통과한 것으로 두고(합성 자료) 기준이 맞으면 결과 표를 쓰고, 다르면 결과 표 없이 종료 코드 2
    gsum = dict(passed=True, gate1=True, gate2=True, gate1_n_targets=12, gate1_targets=[], skipped_wf0=False, smoke=True)
    monkeypatch.setattr(XA, "run_gates", lambda *a_, **k_: (pd.DataFrame([dict(gate="합성", compared=True, equal_2dp=True)]), dict(gsum)))
    base = [x for x in _smoke_argv(shards, tmp_path / "ok", nboot=40) if x != "--skip-gates"]
    monkeypatch.setattr(XA, "c2_ref_values", lambda *a_, **k_: ref)
    assert XA.main(base) == 0
    meta = json.loads((tmp_path / "ok" / XA.EXP_NAME / "xa_meta.json").read_text())
    assert meta["internal_check_wf4c28"]["passed"] and (XB.sealed_dir(XA.EXP_NAME, tmp_path / "ok") / "xa_hyp.csv").exists()
    base = [x for x in _smoke_argv(shards, tmp_path / "bad", nboot=40) if x != "--skip-gates"]
    monkeypatch.setattr(XA, "c2_ref_values", lambda *a_, **k_: bad)
    assert XA.main(base) == 2
    meta = json.loads((tmp_path / "bad" / XA.EXP_NAME / "xa_meta.json").read_text())
    assert meta["status"] == "내부 일치 점검 실패로 멈춤" and not meta["internal_check_wf4c28"]["passed"]
    sd = XB.sealed_dir(XA.EXP_NAME, tmp_path / "bad")
    assert not (sd.exists() and any(sd.glob("xa_*.csv"))), "점검 실패 때는 결과 표를 쓰지 않는다"
    txt = "".join(capsys.readouterr())
    assert "내부 일치 점검" in txt and XB.FORBIDDEN_OUT.search(txt) is None


# ---------------------------------------------------------------- (l) 누설 점검: CE2 는 라벨 10개 진단값만
def test_l_ce2_uses_only_ten_label_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setenv("XBATCH_OUT_ROOTS", str(tmp_path))
    d1 = write_synth_shards(tmp_path / "s1", tmp_path, seed=7)
    d2 = write_synth_shards(tmp_path / "s2", tmp_path, seed=7, full_label_shift=SENTINEL)
    t1, u1 = XA.load_wf4_tms(d1, "wf4", 100)
    t2, u2 = XA.load_wf4_tms(d2, "wf4", 100)
    T1, T2 = XA.target_table(t1, u1), XA.target_table(t2, u2)
    for nm in T1:
        assert T1[nm]["ce2"] == T2[nm]["ce2"] and np.array_equal(T1[nm]["abs_vals"], T2[nm]["abs_vals"])
        assert np.array_equal(XA.ce2_boot(T1[nm]["abs_vals"], nm, 50), XA.ce2_boot(T2[nm]["abs_vals"], nm, 50))
    G1, _ = XA.all_gains(t1, T1, (10, -1))
    G2, _ = XA.all_gains(t2, T2, (10, -1))
    for nm in G1:
        for n in (10, -1):
            for g in G1[nm][n]:
                a_, b_ = G1[nm][n][g], G2[nm][n][g]
                assert a_["pt"] == b_["pt"] and (a_["dist"] is None or np.array_equal(a_["dist"], b_["dist"]))
    # CE2 는 (분할, 추출) 진단값의 평균이고 재표집 seed 는 대상 이름에서만 정해진다
    nm = "Lena|x"
    assert abs(T1[nm]["ce2"] - float(np.mean(T1[nm]["abs_vals"]))) < 1e-12 and T1[nm]["n_diag"] == 15
    pick = np.random.RandomState(XB.seed_of("xa", nm)).randint(0, 15, size=(50, 15))
    assert np.array_equal(XA.ce2_boot(T1[nm]["abs_vals"], nm, 50), T1[nm]["abs_vals"][pick].mean(1))
