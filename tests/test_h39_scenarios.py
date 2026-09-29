"""scripts/2_evaluation/h39_scenarios.py 단위 시험(계획 docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 3.6절).

모든 시험은 합성 저장소나 합성 자료를 쓴다. LG·LGX·LGT 조각과 집계 표를 읽지 않는다. 학습기를 적합하지 않는다.
(a) 4분 판정, 비열등 판정, δ_rel 판정과 '한계 의존' 표지
(b) 짝지음: 같은 예측의 대비는 모든 재표집에서 0 이고, 주 분포의 seed 는 seed_of('lgw', 대상, 대비)다
(c) SC3w: 잭나이프 SE, 멈춤 규칙, 단계, 판정 분류('규칙 비작동' 포함), 합성 자료에서의 끝까지 경로
(d) units: A 행과 추출 재생성이 h40.build_ctx·draw_cells·draw_blocks 와 같고, 조각 대조가 불일치를 잡는다
(e) 쓰기 보호(심볼릭 링크 우회 포함)와 자원 제한(허용 표지 없는 실행은 스레드 1, 허용 표지가 있어도 스레드 4 이하)
(f) 행 없음은 지지로 세지 않는다(SC1w, SC2w, 층화 평균의 풀 지역 수)
(g) 묶음 Holm 은 행 없음을 p = 1 로 넣어 m = 10 을 유지한다
(h) L43 의 2단 재표집: 추출 재표집은 배치마다 따로, 채점 블록 가중은 공유
(i) AK1w: 교차 모드 저장소 합치기(B 블록이 다르면 거부), 최소 쌍 수·대상 수 규칙
(j) 교차 환경 대조: 수치 차, 문자열 불일치, 시간 열(명시 목록만) 제외, 없는 표, 두 환경 모두 빈 표
(k) S-a·S-b 도우미: 위치 평균, CALM 읽기(부등호·inactive 제외), 다중 대응 제외, 해석 규칙 문장
(l) bias-mae: 셀 파일에서 P0·P1·R1·D0 예측 복원
(m) 시나리오 표: 라벨 40개 단계의 풀 지역 표기
(n) 재채점 엔진과 L6 조합 선택
(p) 1.4 (b) 재분류: 두 가중·한 가중·행 CI·서술 형식, 없는 파일과 열의 '원자료 없음' 행, 메타의 재표집 횟수와 '확인하지 못함'
(q) 9.4 δ_rel 의 외부 표: 식별 열·CI 끝값·P0 RMSE 열만 읽고 판정을 CI 끝값으로 다시 계산, 대상 이름 대응과 대응 실패 수
(r) SC3w 산출 분리: 판정용 표에는 rule-cap 행만 있고 RMSE 열이 없으며, 봉인 표에 P0 대비와 lg_equiv 표지가 있다
실행: python3 -m pytest -q tests/test_h39_scenarios.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util                                                                                # noqa: E402
import json                                                                                          # noqa: E402
import sys                                                                                           # noqa: E402
from pathlib import Path                                                                             # noqa: E402

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402
import pytest                                                                                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("h39_scenarios", ROOT / "scripts" / "2_evaluation" / "h39_scenarios.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["h39_scenarios"] = S
_spec.loader.exec_module(S)
H, X = S.H, S.X
NB = 200


def K(method, n, lr=None, alpha="1", placement="cell", draw=0, seed=None, lam=None):
    g = S.G_lg(method, n, alpha=alpha, placement=placement, learner=lr, lam=lam)
    sd = -1 if g[1] == "none" else (0 if seed is None else seed)
    return (g[0], g[1], g[2], g[3], g[4], int(draw), int(sd), g[5])


def mk_tm(name, specs, nb=10, splits=(1, 2), nboot=NB, seed=0, p0_sd=6.0):
    """합성 저장소. specs = {키: (오차 SD, 잡음 묶음, 치우침)}. 같은 묶음·SD·치우침이면 예측이 같다."""
    rng = np.random.RandomState(seed)
    by = {}
    for sp in splits:
        blk = np.repeat(np.arange(nb), 5)
        st = H.BlockStore(name, sp, blk)
        y = rng.randn(len(blk)) * 5 + 50
        noise = {"p0": rng.randn(len(y))}
        st.add(H.P0_KEY, y, y + p0_sd * noise["p0"])
        for k, (sd, grp, bias) in specs.items():
            if grp not in noise:
                noise[grp] = rng.randn(len(y))
            st.add(k, y, y + bias + sd * noise[grp])
        by[sp] = st
    tm = H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in splits}, nboot)
    tm.base_target = tm.target
    tm.expected_splits = list(splits)
    return tm


# ---------------------------------------------------------------- (a)
def test_a_verdicts():
    assert X.verdict4(-2, -1, -2, -1, 0.5) == "우세" and X.verdict4(1, 2, 1, 2, 0.5) == "열세"
    assert X.verdict4(-0.4, 0.4, -0.3, 0.3, 0.5) == "동등" and X.verdict4(-0.4, 0.6, -0.3, 0.3, 0.5) == "미결정"
    assert S.noninf(0.49, 0.49) and not S.noninf(0.49, 0.5) and not S.noninf(np.nan, 0.1)
    assert S.noninf(0.9, 0.9, 1.0) and not S.noninf(0.9, 0.9, 0.5)
    assert S.verdict4_2(-0.3, 0.3, -0.9, 0.9, 0.5, 1.0) == "동등" and S.verdict4_2(-0.3, 0.3, -0.9, 0.9, 0.5, 0.5) == "미결정"
    r = dict(ci_lo=-0.6, ci_hi=0.6, ci_lo_beq=-0.6, ci_hi_beq=0.6, verdict4="미결정", verdict4_d10="동등")
    rc = S.rel_cols(r, 40.0, 40.0)                                    # δ_rel = 0.8
    assert rc["verdict4_rel"] == "동등" and rc["limit_dependence"] == "한계 의존" and abs(rc["delta_rel"] - 0.8) < 1e-12
    r2 = dict(ci_lo=-2, ci_hi=-1, ci_lo_beq=-2, ci_hi_beq=-1, verdict4="우세", verdict4_d10="우세")
    assert S.rel_cols(r2, 10.0, 10.0)["limit_dependence"] == ""


# ---------------------------------------------------------------- (b)
def test_b_pairing_and_seed():
    a, b = K("R1", 10), K("P1", 10)
    tm = mk_tm("Lena|x", {a: (3.0, "g", 0.0), b: (3.0, "g", 0.0)})
    s = S.region_stat(tm, S.G_lg("R1", 10), S.G_lg("P1", 10), "lab", NB)
    assert s["delta"] == 0.0 and np.all(s["dist"] == 0.0) and np.all(s["dist_beq"] == 0.0)
    tm2 = mk_tm("Lena|x", {a: (1.0, "g", 0.0), b: (3.0, "g", 0.0)})
    s1 = S.region_stat(tm2, S.G_lg("R1", 10), S.G_lg("P1", 10), "lab", NB)
    s2 = S.region_stat(tm2, S.G_lg("R1", 10), S.G_lg("P1", 10), "lab", NB)
    s3 = S.region_stat(tm2, S.G_lg("R1", 10), S.G_lg("P1", 10), "other", NB)
    assert np.array_equal(s1["dist"], s2["dist"]) and not np.array_equal(s1["dist"], s3["dist"])
    ref = S.boot_delta_blocks(tm2.used, [a], [b], nboot=NB, seed=S.seed_of("lgw", "Lena|x", "lab"), rep_fn=lambda k: k[5], return_dist=True)
    assert np.allclose(ref["dist"], s1["dist"]) and np.allclose(ref["dist_beq"], s1["dist_beq"])
    assert s1["delta"] < 0 and np.all(s1["dist"] < 0), "오차 SD 1 대 3 이면 모든 재표집에서 A 가 낫다"
    row = S.stat_row(s1, "Lena|x")
    assert row["verdict4"] == "우세" and row["p_two"] == max(row["p_cell"], row["p_beq"])


# ---------------------------------------------------------------- (c)
def test_c_sc3_rule():
    s = np.array([20.0, 25.0, 30.0, 35.0])
    assert S.jk_se(2.0 * s, s) == 0.0
    rng = np.random.RandomState(1)
    y = 2.0 * s + rng.randn(4) * 3
    E = np.array([H.ls_E(np.delete(y, i), np.delete(s, i)) for i in range(4)])
    man = np.sqrt(3 / 4 * np.sum((np.log(E) - np.log(E).mean()) ** 2))
    assert abs(S.jk_se(y, s) - man) < 1e-12
    assert S.jk_se(np.array([-5.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])) >= 0 and S.jk_se(np.array([1.0]), np.array([1.0])) == np.inf
    assert S.sc3_stages(100) == ([3, 10, 40], 40) and S.sc3_stages(15) == ([3, 10, 15], 15) and S.sc3_stages(8) == ([3, 8], 8)
    sA = np.linspace(20, 40, 50); yA = 1.7 * sA
    perm = np.arange(50)
    assert S.sc3_stop(yA, sA, perm, [3, 10, 40], 0.05) == 3
    yN = yA * np.exp(np.random.RandomState(2).randn(50) * 0.5)
    assert S.sc3_stop(yN, sA, perm, [3, 10, 40], 1e-6) == 40
    assert S.sc3_class(0.95, -0.1, 0.1, 0.1, False) == "규칙 비작동"
    assert S.sc3_class(0.3, 0.1, 0.4, 0.4, False) == "지지"
    assert S.sc3_class(0.3, 0.1, 0.6, 0.4, False) == "부분(정밀도 부족)"
    assert S.sc3_class(0.3, 0.1, 0.4, 0.4, True) == "기각" and S.sc3_class(0.6, 0.1, 0.4, 0.4, False) == "기각"


# ---------------------------------------------------------------- 합성 자료(h40.Data 형식)
def synth_data(nb_t=14, cells_per_block=12, seed=3):
    """대상 Canada(블록 nb_t 개)와 원천 Alaska·Lena. 좌표는 서로 멀다(버퍼 제외 없음)."""
    rng = np.random.RandomState(seed)
    rows = []
    for mac, nb, lat0, lon0 in (("Canada", nb_t, 62.0, -120.0), ("Alaska", 8, 66.0, -150.0), ("Lena", 6, 72.0, 126.0)):
        for b in range(nb):
            for c in range(cells_per_block):
                lat = lat0 + (b // 4) * 0.5 + rng.rand() * 0.02 + (c % 3) * 0.004
                lon = lon0 + (b % 4) * 0.5 + rng.rand() * 0.02
                rows.append(dict(macro=mac, block=f"{mac[:2]}{b:03d}", lat=lat, lon=lon))
    df = pd.DataFrame(rows)
    n = len(df)
    for c in H.FEATS:
        df[c] = rng.randn(n)
    df["e5_sqrt_tdd"] = rng.uniform(20, 40, n)
    df["cci_alt"] = rng.uniform(40, 90, n)
    df.loc[rng.rand(n) < 0.05, "cci_alt"] = np.nan
    df["e5_sqrt_tdd_soil"] = df.e5_sqrt_tdd + 1.0
    df[H.TARGET] = 1.6 * df.e5_sqrt_tdd * np.exp(rng.randn(n) * 0.2)
    df["s"] = df.e5_sqrt_tdd.values.astype(float); df["y"] = df[H.TARGET].values.astype(float)
    df["z"] = np.log(df.y) - np.log(df.s)
    df["loc_id"] = np.arange(n)
    df["sub"] = ""
    D = H.Data.__new__(H.Data)
    D.df = df
    D.args = H.parse_args(["--splits", "5", "--threads", "1"])
    D.subs = pd.DataFrame(dict(subregion=[], parent=[]))
    D.macros = set(df.macro.unique()); D.sub_parent = {}; D._src = {}; D._split = {}
    D.sub_src = "synthetic"; D.sub_kmeans_diff = 0
    return D


# ---------------------------------------------------------------- (d)
def test_d_units_regeneration(tmp_path):
    D = synth_data()
    codes = S.Codes(D.df)
    book = S.DrawBook(D, codes)
    sps = S.lg_splits(D, "Canada")
    assert sps
    for sp in sps[:2]:
        c = H.build_ctx(D, D.args, "Canada", "x", sp)
        A_idx, evB = S.ab_rows(D, "Canada", sp)
        assert np.array_equal(D.df.y.values[A_idx], c.yA) and np.array_equal(D.df.block.values[A_idx], c.blkA)
        assert np.array_equal(D.df.y.values[evB], c.yB)
        nA = len(c.yA)
        for n, d in H.cells_of([0, 3, 10, 40, -1], 2, nA):
            sel = H.draw_cells("Canada", "x", sp, n, d, nA)
            cnt = book.counts("Canada", "x", sp, n, d, "cell")
            assert cnt["n_rows"] == len(sel) and cnt["n_block"] == (len(np.unique(c.blkA[sel])) if len(sel) else 0)
        for n, d in H.cells_of([10, 40], 2, nA):
            assert np.array_equal(book.sel("Canada", "x", sp, n, d, "block"), H.draw_blocks("Canada", "x", sp, n, d, c.blkA))
    det, agg = S.units_draws(D, codes, targets=[("Canada", "x")])
    q = agg[(agg.n == 10) & (agg.placement == "cell")]
    assert len(q) == 1 and q.n_rows_min.iloc[0] == 10 and q.n_rows_max.iloc[0] == 10
    sp = sps[0]
    A_idx = book.A("Canada", sp)
    rows = []
    for n, d in ((0, 0), (3, 0), (10, 0), (-1, 0)):
        sel = H.draw_cells("Canada", "x", sp, n, d, len(A_idx))
        rows.append(dict(target="Canada", mode="x", split=sp, method="P1", learner="none", placement="cell", n=n, draw=d, n_lab=len(sel),
                         n_blocks_lab=len(np.unique(D.df.block.values[A_idx][sel])) if len(sel) else 0, rmse_cm=np.nan))
    f = tmp_path / f"lg_smoke__cpu__Canada__x__s{sp}_runs.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    chk, bad = S.units_check(D, codes, tmp_path, ["lg_smoke"])
    assert bool(chk.passed.iloc[0]) and not len(bad)
    rows[1]["n_lab"] += 1
    pd.DataFrame(rows).to_csv(f, index=False)
    chk, bad = S.units_check(D, codes, tmp_path, ["lg_smoke"])
    assert not bool(chk.passed.iloc[0]) and int(chk.n_mismatch_n_lab.iloc[0]) == 1 and len(bad) == 1


def test_d2_units_region_and_sa_sc3_paths():
    D = synth_data()
    codes = S.Codes(D.df)
    reg = S.units_region_table(D, codes)
    r = reg[reg.target == "Canada"].iloc[0]
    assert r.n_rows == (D.df.macro == "Canada").sum() and r.n_block == 14 and r.n_loc_1km <= r.n_rows
    E0, _ = S.source_E0(D, "Canada", "x")
    sa = S.sa_region(D, codes, "Canada", E0, (3, 10), 2)
    assert len(sa) and {"d_row_p1", "d_cell_p1"} <= set(sa.columns)
    stores, recs = S.sc3_target(D, "Canada", "x", E0, 3, taus=(0.05, 0.2))
    assert stores and set(recs.stop.unique()) <= {3, 10, 40}
    per = {"Canada|x": {t: S.sc3_contrasts("Canada|x", stores, 3, NB, t) for t in (0.05, 0.2)}}
    df, tests = S.sc3_rows(per, recs, taus=(0.05, 0.2), judge=("Canada|x",))
    assert len(df) and any(t["test_id"] == "SC3w" for t in tests)


# ---------------------------------------------------------------- (e)
def test_e_write_protection_and_local_limits(tmp_path):
    for bad in ("data/processed/lg", "data/processed/lgx/sub", "data/processed/lgt", "data/processed/lgu/x", "results/rescale_lgx_smoke2"):
        with pytest.raises(SystemExit):
            S.check_out_dir(S.ROOT / bad)
    assert S.check_out_dir(tmp_path) == Path(os.path.realpath(str(tmp_path)))
    S.check_out_dir(S.ROOT / "data" / "processed" / "lgw")
    link = tmp_path / "link_to_lgx"
    os.symlink(str(S.ROOT / "data" / "processed" / "lgx"), str(link))           # 링크만 만든다(보호 폴더에는 쓰지 않는다)
    with pytest.raises(SystemExit):
        S.check_out_dir(link / "sub")
    link2 = tmp_path / "link_to_results"
    os.symlink(str(S.ROOT / "results"), str(link2))
    with pytest.raises(SystemExit):
        S.check_out_dir(link2)
    with pytest.raises(SystemExit):
        S.parse_args(["--mode", "units", "--out-dir", "data/processed/lgd"])
    a = S.parse_args(["--mode", "sc3", "--out-dir", str(tmp_path), "--threads", "4"])
    assert a.threads == 1 and a.nboot == S.LOCAL_NBOOT_MAX and a.CAPPED
    b = S.parse_args(["--mode", "sc3", "--out-dir", str(tmp_path), "--threads", "2", "--allow-local"])
    assert b.threads == 2 and b.nboot == 10000 and not b.CAPPED
    c = S.parse_args(["--mode", "sc3", "--out-dir", str(tmp_path), "--threads", "16", "--allow-local"])
    assert c.threads == S.THREADS_MAX == 4 and c.threads_asked == 16
    assert S._peek_threads(["--allow-local", "--threads", "16"]) == "4" and S._peek_threads(["--threads", "16"]) == "1"
    assert S._peek_threads(["--allow-local", "--threads=3"]) == "3" and S._peek_threads(["--allow-local", "--threads", "x"]) == "1"


# ---------------------------------------------------------------- (f)
def _cells(noninf=True, drop=None, worse=None):
    rows = []
    for t in S.IND5:
        for n in (3, 10):
            if drop == (t, n):
                continue
            w = worse == (t, n)
            rows.append(dict(target=f"{t}|x", n=n, recipe="R1", reference="P0", verdict4="열세" if w else "동등", worse=w, noninf=noninf and not w,
                             noninf_d10=True, p_two=0.01 if w else 0.5, delta=0.8 if w else 0.1, ci_lo=0.2 if w else -0.3, ci_hi=1.2 if w else 0.4))
    return pd.DataFrame(rows)


def test_f_missing_not_support():
    assert S.sc1w_verdict(_cells())["verdict"].startswith("지지")
    v = S.sc1w_verdict(_cells(drop=("Russia_W", 3)))["verdict"]
    assert v.startswith("안전성 미확인(비열등 칸 9/10)") and "Russia_W n = 3" in v
    v2 = S.sc1w_verdict(_cells(worse=("Canada", 10)))["verdict"]
    assert v2.startswith("기각") and "Canada n = 10" in v2
    assert S.sc1w_verdict(_cells(noninf=False))["verdict"].startswith("안전성 미확인(비열등 칸 0/10)")
    c2 = pd.DataFrame([dict(target="Lena|x", n=40, recipe="R1", reference="P1", verdict4="우세")])
    assert S.sc2w_verdict(c2)["verdict"].startswith("판정 불가")
    c3 = pd.DataFrame([dict(target=f"{t}|x", n=40, recipe="R1", reference="P1", verdict4=v) for t, v in zip(S.M3, ("우세", "우세", "열세"))])
    v3 = S.sc2w_verdict(c3)["verdict"]
    assert "3지역 가운데 2곳" in v3 and "Alaska 에서는 오차를 키웠다" in v3
    a = K("R1", 10)
    tms = {"Lena|x": mk_tm("Lena|x", {a: (1.0, "g", 0.0)}), "Canada|x": mk_tm("Canada|x", {a: (1.0, "g", 0.0)}, nb=4)}
    rows = S.contrast_pool(tms, [f"{t}|x" for t in S.MAIN4], S.G_lg("R1", 10), H.P0_GRP, "x", NB)
    m = rows[-1]
    assert m["scope"] == "MEAN" and m["verdict4"] == "판정 불가" and m["pool"] == "판정 불가" and m["undetermined"]


# ---------------------------------------------------------------- (g)
def test_g_bundle_holm_m10():
    d0 = K("D0", 0, lam=1.0)
    tms = {f"{t}|x": mk_tm(f"{t}|x", {d0: (1.0, "g", 0.0)}, seed=i) for i, t in enumerate(S.MAIN4)}
    b = S.bundle_table(tms, {}, NB)
    m = b[b.scope == "MEAN"].set_index("ab")
    assert set(m.index) == {f"AB{i}" for i in range(1, 11)}
    assert m.loc["AB1", "verdict4"] == "우세" and (m.holm_m == 10).all()
    for k in ("AB2", "AB3", "AB4", "AB10"):
        assert m.loc[k, "verdict4"] == "행 없음" and m.loc[k, "holm_input_p"] == 1.0
    assert abs(m.loc["AB1", "holm_p"] - min(1.0, 10 * m.loc["AB1", "holm_input_p"])) < 1e-12
    assert str(m.loc["AB2", "abstract_rule"]).startswith("초록에서 뺀다")


# ---------------------------------------------------------------- (h)
def _l43_tm(draw_bias, nb=10):
    by = {}
    rng = np.random.RandomState(7)
    for sp in (1, 2):
        blk = np.repeat(np.arange(nb), 5)
        st = H.BlockStore("Lena|x", sp, blk)
        y = rng.randn(len(blk)) * 5 + 50
        base = rng.randn(len(y))
        st.add(H.P0_KEY, y, y + 6 * rng.randn(len(y)))
        for d in range(5):
            e = y + 2.0 * base + draw_bias(d) * rng.rand(nb).repeat(5)
            for pl in ("block", "cell"):
                st.add(K("P1", 10, placement=pl, draw=d), y, e)
        by[sp] = st
    return H.TMx("Lena|x", by, {sp: dict(dup_of=-1, valid=True) for sp in (1, 2)}, NB)


def test_h_l43_two_stage():
    tm = _l43_tm(lambda d: 3.0 * d)
    r = S.l43_region(tm, "P1", 10, NB, S.seed_of("lgw-l43", "Lena|x", "P1", 10))
    assert abs(r["delta"]) < 1e-12, "두 배치의 예측이 같으면 점 추정은 0"
    assert np.std(r["dist"]) > 0, "추출 재표집이 배치마다 따로 돌면 분포가 퍼진다"
    sc = S.region_stat(tm, S.G_lg("P1", 10, placement="block"), S.G_lg("P1", 10, placement="cell"), "L43", NB)
    assert np.all(sc["dist"] == 0.0), "추출 조건부 CI 는 0 이다"
    tm0 = _l43_tm(lambda d: 0.0)
    r0 = S.l43_region(tm0, "P1", 10, NB, 11)
    assert np.allclose(r0["dist"], 0.0) and np.allclose(r0["dist_beq"], 0.0), "추출 차이가 없으면 블록 가중 공유로 분포가 0"
    assert 0.0 <= r["win_rate"] <= 1.0


# ---------------------------------------------------------------- (i)
def test_i_ak1_merge_and_rules():
    ks = {K("R1", 10): (1.0, "g", 0.0)}
    tx, ti = mk_tm("AL-2|x", ks, seed=1), mk_tm("AL-2|i", ks, seed=1)
    m = S.merge_modes(tx.used[1], ti.used[1])
    assert m is not None and any(k[0] == "x" for k in m.keys) and any(k[0] == "i" for k in m.keys)
    ti_bad = mk_tm("AL-2|i", ks, nb=9, seed=1)
    assert S.merge_modes(tx.used[1], ti_bad.used[1]) is None
    res, note = S.ak1_rprime(tx, ti_bad, S.G_lg("R1", 10), NB, 5)
    assert res is None and note == "블록 불일치"
    res, note = S.ak1_rprime(tx, ti, S.G_lg("R1", 10), NB, 5)
    assert res is not None and "dist" in res and note == ""

    def curve(pass_i, pass_x):
        rows = []
        for (t, m_, n), pi in pass_i.items():
            for mode, ok in (("i", pi), ("x", pass_x.get((t, m_, n), False))):
                rows.append(dict(target=t, mode=mode, method=m_, learner="none" if m_ == "P1" else H.BASE_LEARNER, alpha="1", placement="cell",
                                 n=n, lam=H.base_lam(m_), sig_p0="improve" if ok else "ns", d_p0_hi=-1.0 if ok else 1.0))
        return pd.DataFrame(rows)
    three = {("AL-1", "P1", 10): True, ("AL-2", "P1", 10): True, ("AL-3", "P1", 10): True}
    _, t = S.ak1_tables(curve(three, {}), {}, NB)
    assert t["verdict"].startswith("판정 불가"), "통과 쌍 3개는 판정 불가"
    two_t = {("AL-1", "P1", 10): True, ("AL-1", "R1", 10): True, ("AL-2", "P1", 10): True, ("AL-2", "R1", 10): True}
    _, t = S.ak1_tables(curve(two_t, {}), {}, NB)
    assert t["verdict"].startswith("판정 불가"), "대상 2개는 판정 불가"
    four = dict(three); four[("AL-1", "R1", 10)] = True
    _, t = S.ak1_tables(curve(four, {k: True for k in list(four)[:3]}), {}, NB)
    assert t["verdict"].startswith("알래스카 하위 지역에서") and "r 0.75" in t["verdict"]
    _, t = S.ak1_tables(curve(four, {list(four)[0]: True}), {}, NB)
    assert t["verdict"].startswith("이 통과는 원천에 같은 지역 자료가 있는 조건에 의존했다")


# ---------------------------------------------------------------- (j)
def test_j_xenv_compare(tmp_path):
    ref, loc = tmp_path / "ref", tmp_path / "loc"
    ref.mkdir(); loc.mkdir()
    base = pd.DataFrame(dict(target=["A", "B", "C"], delta=[0.1, -0.2, np.nan], verdict4=["우세", "동등", "미결정"], elapsed_s=[1.0, 2.0, 3.0],
                             hk_s=[1.0, 1.0, 1.0], time_scale=[2.0, 2.0, 2.0]))
    base.to_csv(ref / "lgx_precheck_curve.csv", index=False)
    loc_df = base.copy(); loc_df.loc[0, "delta"] = 0.101; loc_df.loc[1, "verdict4"] = "미결정"; loc_df["elapsed_s"] = 9.0
    loc_df["hk_s"] = 1.5                                                       # 이름이 '_s' 로 끝나도 시간 열이 아니면 대조한다
    loc_df.to_csv(loc / "lgx_precheck_curve.csv", index=False)
    (ref / "lgx_precheck_lg_aux.csv").write_text("\n"); (loc / "lgx_precheck_lg_aux.csv").write_text("\n")
    base.to_csv(ref / "lgx_precheck_timing.csv", index=False)
    base.to_csv(ref / "lgx_precheck_tests.csv", index=False)
    d = S.compare_tables(loc, ref, ["lgx_precheck"])
    num = d[(d.table == "lgx_precheck_curve.csv") & (d.column == "delta")].iloc[0]
    assert abs(num.max_abs_diff - 0.001) < 1e-9 and not num.passed and num.n_nan_mismatch == 0
    st = d[(d.table == "lgx_precheck_curve.csv") & (d.column == "verdict4")].iloc[0]
    assert st.n_str_mismatch == 1 and not st.passed
    assert d[(d.column == "elapsed_s")].kind.iloc[0] == "skipped_time"
    hk = d[(d.table == "lgx_precheck_curve.csv") & (d.column == "hk_s")].iloc[0]
    assert hk.kind == "numeric" and not hk.passed and abs(hk.max_abs_diff - 0.5) < 1e-12
    assert d[(d.table == "lgx_precheck_curve.csv") & (d.column == "time_scale")].kind.iloc[0] == "numeric"
    sh = d[(d.table == "lgx_precheck_lg_aux.csv") & (d.kind == "shape")].iloc[0]
    assert bool(sh.empty_both) and bool(sh.passed)
    assert "lgx_precheck_timing.csv" not in set(d.table)
    assert (d[d.table == "lgx_precheck_tests.csv"].kind == "missing_local").all()


# ---------------------------------------------------------------- (k)
def test_k_sa_sb_helpers(tmp_path):
    y, m = S.loc_means(np.array([2, 0, 2, 1]), np.array([10.0, 20.0, 30.0, 40.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert list(y) == [20.0, 40.0, 20.0] and list(m) == [2.0, 4.0, 2.0]
    hdr = "/* DATA DESCRIPTION:\nx\n*/\n"
    body = "Event\tSite\tName\tID\tCountry\tArea\tLatitude\tLongitude\tDate/Time\tALD [cm]\tSample comment\tComment\n"
    body += "E1\tS\tN\t1\tR\tA\t65.0000\t100.0000\t2001\t50\t\t\n"
    body += "E1\tS\tN\t1\tR\tA\t65.0000\t100.0000\t2002\t>60\t\t\n"
    body += "E1\tS\tN\t1\tR\tA\t65.0000\t100.0000\t2003\t70\tinactive\t\n"
    body += "E2\tS\tN\t2\tR\tA\t66.0000\t101.0000\t2001\t40\t\t\n"
    body += "E3\tS\tN\t3\tR\tA\t66.0020\t101.0010\t2001\t45\t\t\n"
    f = tmp_path / "calm.tab"
    f.write_text(hdr + body, encoding="utf-8")
    allr, ok, st = S.read_calm(f)
    assert st["n_rows"] == 5 and st["n_ineq_gt"] == 1 and st["n_inactive"] == 1 and st["n_valid"] == 3
    cells = pd.DataFrame(dict(loc_id=[1, 2], lat=[65.0005, 66.001], lon=[100.0, 101.0005]))
    vals, qa, agg = S.calm_match(cells, allr, ok)
    assert agg["n_single"] == 1 and agg["n_multi"] == 1 and list(vals[1]["years"]) == [2001]
    summ = pd.DataFrame([dict(target=t, n=n, quantity=q, count=10, p10=-0.1, p50=0.1, p90=0.2) for t in ("Lena", "Canada", "Alaska")
                         for n in (3, 10) for q in ("unit_p1", "cell_p1", "row_p1")]
                        + [dict(target="Russia_W", n=n, quantity="ref_row_p1", count=10, p10=-9.0, p50=-8.0, p90=-7.0) for n in (3, 10)])
    assert S.sa_verdict(summ)["verdict"].startswith("라벨 단위(점 대 1 km 위치 평균)는")
    summ.loc[(summ.target == "Canada") & (summ.n == 10) & (summ.quantity == "unit_p1"), "p50"] = -0.9
    v = S.sa_verdict(summ)
    assert v["verdict"].startswith("점 라벨 조건에서는") and "Canada n = 10" in v["verdict"] and "지역별" in v["flags"]
    sb = pd.DataFrame([dict(target=t, n=n, p50=0.1) for t in ("Russia_W", "Russia_E") for n in (3, 10)])
    assert S.sb_verdict(sb)["verdict"].startswith("단년 관측 라벨에서도")
    sb.loc[0, "p50"] = 0.7
    assert S.sb_verdict(sb)["verdict"].startswith("단년 라벨에서는 이득이 줄었다")


# ---------------------------------------------------------------- (l)
def test_l_bias_mae_cells(tmp_path):
    rng = np.random.RandomState(0)
    nB = 30
    y = rng.uniform(30, 90, nB); s = rng.uniform(20, 40, nB)
    keys, P, E = [], [], []
    for d in (0,):
        for sd in (0, 1):
            keys.append(["R0", H.BASE_LEARNER, "1", "cell", 0, 0, sd, "g"]); P.append(np.full(nB, 4.0)); E.append(1.5)
            keys.append(["R1", H.BASE_LEARNER, "1", "cell", 10, d, sd, "g"]); P.append(np.zeros(nB)); E.append(1.8)
            keys.append(["D0", H.BASE_LEARNER, "1", "cell", 0, 0, sd, "pred"]); P.append(y + 2.0); E.append(np.nan)
    f = tmp_path / "u_cells.npz"
    np.savez_compressed(f, keys=np.array([json.dumps(k) for k in keys]), P=np.vstack(P).astype(np.float32), E=np.array(E), y=y.astype(np.float32),
                        s=s.astype(np.float32))
    u = S.bias_mae_unit(f)
    y32, s32 = y.astype(np.float32).astype(float), s.astype(np.float32).astype(float)
    p0 = u[u.method == "P0"].iloc[0]
    assert abs(p0.bias - np.mean(1.5 * s32 - y32)) < 1e-9
    r0 = u[(u.method == "R0")].bias.mean()
    assert abs(r0 - np.mean(1.5 * s32 + 0.25 * 4.0 - y32)) < 1e-5
    d0 = u[u.method == "D0"]
    assert np.allclose(d0.bias, 2.0, atol=1e-4) and np.allclose(d0.mae, 2.0, atol=1e-4)
    assert len(u[u.method == "P1"]) == 1 and abs(u[u.method == "P1"].bias.iloc[0] - np.mean(1.8 * s32 - y32)) < 1e-9
    assert np.allclose(u.mse, u.bias2 + u.var_err)


# ---------------------------------------------------------------- (m)
def test_m_scenario_pool_label():
    tms = {}
    for i, t in enumerate(S.IND5):
        specs = {}
        for n in (3, 10) + ((40,) if t in ("Lena", "Canada") else ()):
            specs[K("P1", n)] = (2.0, "p1", 0.0)
            specs[K("R1", n)] = (1.5, "r1", 0.0)
        specs[K("R1", 0)] = (4.0, "r0", 0.0); specs[K("R0", 0)] = (4.0, "r0", 0.0)
        tms[f"{t}|x"] = mk_tm(f"{t}|x", specs, seed=10 + i, nboot=100)
    cells, summ = S.scenario_tables(tms, 100)
    q = summ[(summ.stage == "T40") & (summ.recipe == "R1") & (summ.reference == "P0")].iloc[0]
    assert q.main4_pool == "부분(지역 2/4)" and q.n_independent == 2
    assert set(cells[cells.stage == "T3"].target) == {f"{t}|x" for t in S.IND5}
    t3 = cells[(cells.stage == "T3") & (cells.recipe == "R1")]
    assert t3.net_value.notna().all()


# ---------------------------------------------------------------- (n)
def test_n_rescore_and_l6():
    rng = np.random.RandomState(4)
    per = {}
    for t in ("Lena", "Canada"):
        units = []
        for sp in (0, 1):
            blk = np.repeat(np.arange(10), 4); y = rng.randn(40) * 5 + 50
            units.append((sp, y, blk, (y + rng.randn(40))[None, :], (y + 3 * rng.randn(40))[None, :]))
        per[t] = S._rescore_stat(units, NB, 1)
    rows = S._rescore_pool("Hx", "noinfo", per, NB)
    assert rows[-1]["target"].startswith("MEAN[") and rows[-1]["verdict4"] == "우세"
    specs = {K("R1", 10): (1, "a", 0), K("R1", 10, alpha="nested"): (1, "a", 0), K("R1", 10, placement="block"): (1, "a", 0),
             K("R1", 10, lam=0.5): (1, "a", 0), K("P1", 40): (1, "a", 0), K("V1", 160): (1, "a", 0), K("D0", 10, alpha="10"): (1, "a", 0),
             K("R1", 3): (1, "a", 0)}
    tm = mk_tm("AL-1|i", specs)
    got = S.l6_groups(tm)
    assert set(got) == {S.G_lg("R1", 10), S.G_lg("P1", 40), S.G_lg("V1", 160), S.G_lg("D0", 10, alpha="10")}


# ---------------------------------------------------------------- (o) 집계 경로(합성 조각, 임시 폴더)
def _write_shard(sdir, tag, st, target, mode, split, cfg_common="c0", runs=None):
    base = sdir / f"{tag}__cpu__{target}__{mode}__s{split}"
    S.H4.save_stores([st], Path(str(base) + "_blocksse.npz"))
    (runs if runs is not None else pd.DataFrame([dict(target=target, mode=mode, split=split, method="P1", placement="cell", n=10,
                                                     n_blocks_lab=3)])).to_csv(str(base) + "_runs.csv", index=False)
    Path(str(base) + "_unit.json").write_text(json.dumps(dict(target=target, mode=mode, split=split, part="cpu", cfg_common=cfg_common,
                                                              code_sha="x", status="ok")))


def test_o_summarize_end_to_end(tmp_path):
    D = synth_data()
    sdir = tmp_path / "lg" / "shards"; sdir.mkdir(parents=True)
    xdir = tmp_path / "lgx" / "shards"; xdir.mkdir(parents=True)
    rng = np.random.RandomState(5)
    for sp in S.lg_splits(D, "Canada")[:2]:
        A_idx, evB = S.ab_rows(D, "Canada", sp)
        y = D.df.y.values[evB]
        st = H.BlockStore("Canada|x", sp, D.df.block.values[evB])
        st.add(H.P0_KEY, y, y + 8 * rng.randn(len(y)))
        for n in (0, 3, 10, 40, -1):
            for d in ((0,) if n in (0, -1) else range(2)):
                st.add(K("P1", n, draw=d), y, y + 5 * rng.randn(len(y)))
                st.add(K("R1", n, draw=d), y, y + 4 * rng.randn(len(y)))
                st.add(K("D0", n, draw=d, lam=1.0), y, y + 9 * rng.randn(len(y)))
                if n in (10, 40):
                    st.add(K("P1", n, draw=d, placement="block"), y, y + 5 * rng.randn(len(y)))
        st.add(K("R0", 0), y, y + 4 * rng.randn(len(y)))
        _write_shard(sdir, "lg", st, "Canada", "x", sp)
        stx = H.BlockStore("Canada|x", sp, D.df.block.values[evB])
        stx.add(H.P0_KEY, y, y + 8 * rng.randn(len(y)))
        stx.add(("B:ens", "none", "1", "cell", 0, 0, -1, 0.0), y, y + 7 * rng.randn(len(y)))
        _write_shard(xdir, "lgx9", stx, "Canada", "x", sp)
    out = tmp_path / "out"
    a = S.parse_args(["--mode", "summarize", "--out-dir", str(out), "--lg-shards", str(sdir), "--lgx-shards", str(xdir),
                      "--lg-dir", str(tmp_path / "lg"), "--lgx-dir", str(tmp_path / "lgx"), "--data-dir", str(tmp_path), "--nboot", "100"])
    out.mkdir()
    res = S.run_summarize(a, D=D)
    for f in ("lgw_bundle.csv", "lgw_aux4.csv", "lgw_splitratio.csv", "lgw_scenarios.csv", "lgw_scenarios_summary.csv", "lgw_l43.csv",
              "lgw_tests.csv", "lgw_meta.json"):
        assert (out / f).exists(), f
    t = pd.read_csv(out / "lgw_tests.csv")
    v = dict(zip(t.test_id, t.verdict))
    assert v["SC1w"].startswith("(재표집 제한, 판정에 쓰지 않음) 안전성 미확인"), "한 지역뿐이면 지지가 아니다"
    assert "판정 불가" in v["AK1w"] and "판정 불가" in v["SC2w"]
    b = res["bundle"]
    assert (b[b.scope == "MEAN"].holm_m == 10).all() and (b.nboot == 100).all() and b.nboot_capped.all()
    assert b[(b.scope == "MEAN") & (b.ab == "AB2")].verdict4.iloc[0] == "판정 불가", "CI 풀 지역 1개는 판정 불가"
    _write_shard(sdir, "lg", st, "Canada", "x", 99, cfg_common="other")
    with pytest.raises(SystemExit):
        S.run_summarize(a, D=D)


# ---------------------------------------------------------------- (p) 1.4 (b) 재분류
def test_p_reclassify(tmp_path):
    (tmp_path / "h4").mkdir(); (tmp_path / "h2").mkdir()
    pd.DataFrame(dict(test=["a", "b", "c", "d"], target=["Lena", "Lena", "Canada", "Canada"], delta=[-1.0, 0.1, 0.2, 0.3],
                      ci_lo=[-2.0, -0.3, 0.1, np.nan], ci_hi=[-0.5, 0.4, 0.4, np.nan], delta_blockeq=[-1.0, 0.1, 0.2, 0.3],
                      ci_lo_beq=[-2.5, -0.2, np.nan, np.nan], ci_hi_beq=[-0.4, 0.3, np.nan, np.nan])).to_csv(tmp_path / "h4" / "x_tests.csv", index=False)
    (tmp_path / "h4" / "x_meta.json").write_text(json.dumps(dict(nboot=1000)))
    (tmp_path / "h4" / "y_meta.json").write_text(json.dumps(dict(args=dict(nboot=500))))
    pd.DataFrame(dict(target=["Lena"], method=["m"], lam=[0.25], n=[3], delta_vs_shrink=[-0.5], ci_lo=[-0.9], ci_hi=[-0.1])).to_csv(
        tmp_path / "h2" / "r_tests.csv", index=False)
    pd.DataFrame(dict(rho=[0.3])).to_csv(tmp_path / "h2" / "d_rule.csv", index=False)
    spec = [dict(hyp="HX", src="h4/x_tests.csv", meta="h4/x_meta.json", fmt="delta2", ids=("test", "target"), contrasts=(S._C2,)),
            dict(hyp="HY", src="h4/x_tests.csv", meta="h4/y_meta.json", fmt="delta2", ids=("test",),
                 contrasts=(("A-B", "delta", "ci_lo", "ci_hi", "d_missing", "lo_missing", "hi_missing"),)),
            dict(hyp="HR", src="h2/r_tests.csv", meta="h2/none_meta.json", fmt="rowci", ids=("target", "method"),
                 contrasts=(("방법-수축", "delta_vs_shrink", "ci_lo", "ci_hi", None, None, None),)),
            dict(hyp="HD", src="h2/d_rule.csv", meta="", fmt="desc", note="상관"),
            dict(hyp="HM", src="h3/no_such.csv", meta="", fmt="delta2", ids=("test",), contrasts=(S._C2,)),
            dict(hyp="HA", src="", meta="", fmt="absent", note="미산출")]
    r = S.reclassify(tmp_path, spec)
    x = r[r.hyp == "HX"].set_index("test_id")
    assert list(x.verdict4) == ["우세", "동등", "4분 판정 불가(CI 규약 다름)", "판정 불가(CI 없음)"]
    assert list(x.ci_kind) == ["두 가중", "두 가중", "셀 가중만", "CI 없음"] and (x.nboot_original == "1000").all()
    assert (r.kind == "재분류(비맹검)").all()
    y = r[r.hyp == "HY"].iloc[0]
    assert y.verdict4 == "원자료 없음" and "열 없음" in y.note and y.nboot_original == "500"
    hr = r[r.hyp == "HR"].iloc[0]
    assert hr.verdict4 == "규약 다름, 서술" and hr.ci_kind == "행 재표집 CI" and hr.nboot_original == "확인하지 못함"
    assert r[r.hyp == "HD"].iloc[0].verdict4 == "규약 다름, 서술"
    hm = r[r.hyp == "HM"].iloc[0]
    assert hm.verdict4 == "원자료 없음" and "파일 없음" in hm.note, "없는 파일은 건너뛰지 않고 기록한다"
    assert r[r.hyp == "HA"].iloc[0].verdict4 == "원자료 없음"
    empty = S.reclassify(tmp_path / "nothing")
    assert len(empty) == len(S.RECLASS_SPEC) and (empty.verdict4 == "원자료 없음").all()
    assert {"H23", "H24", "H26", "H27", "H29"} <= set(empty.hyp)
    fm = {sp["hyp"]: sp["fmt"] for sp in S.RECLASS_SPEC if sp["hyp"] in ("H23", "H24", "H26")}
    assert fm == {"H23": "rowci", "H24": "rowci", "H26": "absent"}


# ---------------------------------------------------------------- (q) 9.4 δ_rel 의 외부 표
def test_q_delta_rel_inputs(tmp_path):
    t = pd.DataFrame(dict(test_id=["L32"] * 5, contrast=["c"] * 5, scope=["region", "region", "MEAN", "region", "verdict"],
                          target=["Lena|x", "Lena", "MEAN[Lena|x,Canada|x]", "AL-2", ""], mode=["", "x", "", "", ""],
                          ci_lo=[-0.3, -2.0, -0.2, -0.3, np.nan], ci_hi=[0.3, -1.0, 0.2, 0.3, np.nan],
                          ci_lo_b=[-0.9, -2.0, -0.3, -0.3, np.nan], ci_hi_b=[0.9, -1.0, 0.3, 0.3, np.nan],
                          delta=[99.0] * 5, verdict4=["거짓"] * 5, verdict4_d10=["거짓"] * 5, rmse_A=[1.0] * 5))
    t.to_csv(tmp_path / "lgf_tests.csv", index=False)
    df = S.read_ci_table(tmp_path / "lgf_tests.csv")
    assert not ({"delta", "verdict4", "verdict4_d10", "rmse_A", "ci_lo_b"} & set(df.columns)), "판정·Δ·RMSE 열은 읽지 않는다"
    assert {"ci_lo_beq", "ci_hi_beq", "target", "mode", "scope"} <= set(df.columns)
    p0map = {"Lena|x": (40.0, 50.0), "Canada|x": (20.0, 20.0), "AL-2|i": (30.0, 30.0), "AL-2|x": (31.0, 31.0)}      # δ_rel 0.8·1.0 cm
    out, un = S.delta_rel_external(df, p0map, "lgf:lgf_tests.csv")
    assert len(out) == 4 and un == 1, "판정 행은 빼고, 모드가 여럿인데 모드가 없는 대상(AL-2)은 대응시키지 않는다"
    r0 = out.iloc[0]
    assert r0.verdict4 == "미결정" and r0.verdict4_d10 == "동등" and r0.verdict4_rel == "동등" and r0.limit_dependence == "한계 의존"
    r1 = out.iloc[1]
    assert r1.lg_targets == "Lena|x" and r1.verdict4 == "우세", "대상|모드 대응과 CI 끝값으로 다시 계산한 판정"
    r2 = out.iloc[2]
    assert bool(r2.pool_row) and abs(r2.rmse_p0 - 30.0) < 1e-12 and r2.pool_note
    r3 = out.iloc[3]
    assert r3.verdict4_rel == "판정 불가(P0 RMSE 없음)" and r3.limit_dependence == "" and r3.lg_targets == "?"
    t2 = t.iloc[:1].assign(rmse_p0=[10.0], rmse_p0_beq=[12.0])
    t2.to_csv(tmp_path / "lgt_tests.csv", index=False)
    o2, _ = S.delta_rel_external(S.read_ci_table(tmp_path / "lgt_tests.csv"), p0map, "lgt")
    assert o2.iloc[0].p0_source == "셀 가중 표; 블록 등가중 표" and abs(o2.iloc[0].delta_rel - 0.2) < 1e-12
    assert S.REL_SKIP_FILE.search("lgf_pre_tests.csv") and S.REL_SKIP_FILE.search("lgt_smoke_tests.csv") and not S.REL_SKIP_FILE.search("lgt_tests.csv")


# ---------------------------------------------------------------- (r) SC3w 산출 분리
def test_r_sc3_sealed_split(tmp_path):
    D = synth_data()
    E0, _ = S.source_E0(D, "Canada", "x")
    stores, recs = S.sc3_target(D, "Canada", "x", E0, 2, taus=(0.05,))
    per = {"Canada|x": {0.05: S.sc3_contrasts("Canada|x", stores, 2, NB, 0.05)}}
    df, _ = S.sc3_rows(per, recs, taus=(0.05,), judge=("Canada|x",))
    full, main, capA = S.sc3_write(tmp_path, df, recs)
    m = pd.read_csv(tmp_path / "lgw_sc3.csv")
    assert set(m.contrast) == {"rule-cap"} and not ({"rmse_A", "rmse_B", "lg_equiv"} & set(m.columns))
    f = pd.read_csv(tmp_path / "sealed" / "lgw_sc3_full.csv")
    assert set(f.contrast) == {"rule-cap", "rule-P0", "cap-P0"} and {"rmse_A", "rmse_B", "lg_equiv"} <= set(f.columns)
    assert (tmp_path / "sealed" / "README_SEALED.txt").exists()
    assert not capA["Canada"][1] and f[f.contrast == "cap-P0"].lg_equiv.str.contains("n = 40").all()
    recs2 = recs.assign(cap=recs.nA)
    full2, _, _ = S.sc3_write(tmp_path / "b", df, recs2)
    assert full2[full2.contrast == "cap-P0"].lg_equiv.str.contains("전량과 같은 값").all()
