"""지도 과제(MAP) 시험: scripts/2_evaluation/h49_transfer_map.py, scripts/1_data_prep/build_map_grid_lena_v1.py,
scripts/4_visualization/paper/fig_map_lena.py. 계획 docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 6.3–6.8.

합성 자료만 쓴다(LG·LGX·LGT·LGU·LGF 의 판정 표와 조각을 읽지 않는다). CatBoost 적합은 작은 합성 자료에서 몇 건뿐이다.
(a) 대비 이름 해석, (b) 층화 평균 우세 판정(행 없음, 표마다 다른 판정), (c) 레나 행 열세 아님 판정, (d) L29 P*·P1* 재현과 목록 안 대체,
(e) 6.7 규칙의 시나리오(기본, ML 우세, P* 대체, 두 후보의 점 추정 비교, L8 캡션 표지, GPU 학습기 사실), (f) LGU-B2 '전이'에서 σ 파일 없이 중단,
(g) P1·R1 이 h40 의 식(수축 계수, 잔차 학습 행렬, seed 0·1 평균)과 같다, (h) 앵커 기준선이 h42 의 식과 같다, (i) 구간 분위가 수계산과 같다,
(j) 외삽 열 수와 범주, (k) 실행 보호(허용 표지, 스레드 상한), (l) 격자 색인 왕복과 셀 수, 대조 함수의 NaN·허용 차 규칙,
(m) 그림의 래스터 → 셀 대응과 영문 캡션 규칙,
(n) 본 실행 입력 확인(규칙 대비 행 누락, 재표집 제한, 라벨 형식 변경이면 선택을 기록하지 않고 중단), MEAN 행의 주 4지역 확인,
(o) GPU 학습기 사실(L5, LGT L32, LGF-F2, LGF-N3의 라벨 해석, 기준 λ, LGT 표지), (p) 출력 폴더 보호, (q) 회색 육지 통계,
(r) h49_map_contrasts 의 대비 표(h39 엔진, 행 없음 자리 표시)와 h49 해석의 결합, (s) real_ctx 와 h40.build_ctx·run_ctx 의 직접 대조
(E0, 원천·A 행 순서, 추출, P1·R1 의 채점 RMSE), (t) 본문 3패널의 패널 문자와 캡션 문자.
실행(CPU 2스레드, nice 10):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 python3 -m pytest -q tests/test_h49_transfer_map.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

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


h49 = _load("h49_transfer_map", "scripts/2_evaluation/h49_transfer_map.py")
grid = _load("build_map_grid_lena_v1", "scripts/1_data_prep/build_map_grid_lena_v1.py")
mapc = _load("h49_map_contrasts", "scripts/2_evaluation/h49_map_contrasts.py")
M4 = "MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"


def _row(label, scope, target, v, d, n=0, test="L29"):
    return dict(test_id=test, contrast=label, scope=scope, target=target, verdict4=v, delta=d, n=n)


def _book(**tabs):
    return h49.VerdictBook({k: pd.DataFrame(v) for k, v in tabs.items()})


# ---------------------------------------------------------------- (a)
def test_parse_label():
    assert h49.parse_label("R1-P1|n10") == ("R1", "P1", 10)
    assert h49.parse_label("R1-P1|n-1") == ("R1", "P1", -1)
    assert h49.parse_label("R1-P0|all") == ("R1", "P0", -1)
    assert h49.parse_label("B:s_aff-P0|n0") == ("B:s_aff", "P0", 0)
    assert h49.parse_label("R1[mlp]-R1[catboost_lo]|n10") == ("R1[mlp]", "R1[catboost_lo]", 10)
    assert h49.parse_label("B:ku_raw-P0|n0|kuok") is None
    assert h49.parse_label("nan") is None


# ---------------------------------------------------------------- (b)
def test_dominant_rules():
    b = _book(lgx_lg_aux=[_row("R1-P1|n10", "MEAN", M4, "우세", -1.0, 10)], lgw_bundle=[_row("R1-P1|n10", "MEAN4", M4, "우세", -1.0, 10)])
    r = h49.dominant(b, "R1", "P1", 10)
    assert r["ok"] and r["note"] == ""
    b = _book(lgx_lg_aux=[_row("R1-P1|n10", "MEAN", M4, "우세", -1.0, 10)], lgw_bundle=[_row("R1-P1|n10", "MEAN4", M4, "미결정", -1.0, 10)])
    r = h49.dominant(b, "R1", "P1", 10)
    assert not r["ok"] and "재표집 의존" in r["note"]
    r = h49.dominant(b, "R1", "P1", 40)
    assert not r["ok"] and r["note"] == "행 없음"
    b = _book(lgx_lg_aux=[_row("R1-P1|n-1", "MEAN", M4, "우세", -1.0, -1)], lgw_bundle=[_row("R1-P1|all", "MEAN", "MEAN[]", "행 없음", np.nan, -1)])
    assert h49.dominant(b, "R1", "P1", -1)["ok"]                          # h39 의 자리 표시 행은 판정으로 세지 않는다
    b = _book(lgx_lg_aux=[_row("R1-P1|n-1", "MEAN", M4, "우세", -1.0, -1)], lgw_bundle=[_row("R1-P1|all", "MEAN", M4, "판정 불가", np.nan, -1)])
    assert not h49.dominant(b, "R1", "P1", -1)["ok"]
    b = _book(lgx_lg_aux=[_row("R1-P1|n10", "MEAN3", "MEAN[Lena|x,Canada|x,Alaska|x]", "우세", -1.0, 10)])
    assert not h49.dominant(b, "R1", "P1", 10)["ok"]                      # 3지역 보조 평균은 쓰지 않는다
    with pytest.raises(ValueError):
        h49.VerdictBook({"lgw_bundle": pd.DataFrame([dict(id="AB5", verdict="우세")])})


# ---------------------------------------------------------------- (c)
@pytest.mark.parametrize("v, ok", [("미결정", True), ("동등", True), ("우세", True), ("열세", False), ("판정 불가", False)])
def test_lena_not_inferior(v, ok):
    b = _book(lgx_tests=[_row("P0@soil-P0|n0", "region", "Lena|x", v, 0.1)])
    assert h49.lena_not_inferior(b, "P0@soil", "P0", 0)["ok"] is ok
    assert not h49.lena_not_inferior(_book(lgx_tests=[]), "P0@soil", "P0", 0)["ok"]


# ---------------------------------------------------------------- (d)
def test_l29_pick_and_replacement():
    rows = [_row(f"{m}-P0|n0", "MEAN", M4, "미결정", 0.3) for m in h49.L29_N0_H42 if m not in ("P0@tddm", "P0@soil", "B:s_aff")]
    rows += [_row("P0@tddm-P0|n0", "MEAN", M4, "우세", -3.0), _row("P0@soil-P0|n0", "MEAN", M4, "우세", -1.0),
             _row("B:s_aff-P0|n0", "MEAN", M4, "우세", -2.0),
             _row("P0@soil-P0|n0", "region", "Lena|x", "미결정", 0.2), _row("B:s_aff-P0|n0", "region", "Lena|x", "열세", 1.5)]
    b = _book(lgx_tests=rows)
    r = h49.l29_pick(b, 0, "P0")
    assert r["h42_pick"] == "P0@tddm" and r["replaced"] and r["pick"] == "P0@soil"     # B:s_aff 는 레나 행 열세라 뺀다
    rows2 = [q for q in rows if not q["contrast"].startswith("P0@soil")]
    r2 = h49.l29_pick(_book(lgx_tests=rows2), 0, "P0")
    assert r2["replaced"] and r2["pick"] is None                                         # 대체할 기준선이 없으면 P0
    rows3 = [_row("P1@ku-P1|n10", "MEAN", M4, "우세", -0.5, 10), _row("B:ens-P1|n10", "MEAN", M4, "우세", -0.7, 10)]
    r3 = h49.l29_pick(_book(lgx_tests=rows3), 10, "P1")
    assert r3["pick"] == "B:ens" and not r3["replaced"]
    r4 = h49.l29_pick(_book(lgx_aux=rows3), 10, "P1")                                    # L29 는 lgx_tests 의 행만 본다
    assert r4["pick"] is None


# ---------------------------------------------------------------- (e)
def _select(scenario, t=None):
    t = t if t is not None else h49.synthetic_tables(scenario)
    book = h49.VerdictBook({k: t.get(k) for k in h49.BOOK_SOURCES})
    b2, l8 = h49.lgu_b2(t.get("lgu_b_tests")), h49.l8_verdict(t.get("lg_tests"))
    sel = h49.rule_select(book, b2, l8, h49.gpu_learner_notes(t))
    return sel, h49.check_rule_inputs(book, sel, t, b2, l8)


def test_rule_scenarios():
    s, pr = _select("baseline")
    assert s["panels"] == dict(a="P0", b="P1", c="P1", d="P0") and s["c"]["caption_L8"] and s["d"]["normalizer"] == "const" and pr == []
    s, pr = _select("all_ml")
    assert s["panels"] == dict(a="R0", b="R1", c="R1", d="R0") and not s["c"]["caption_L8"] and pr == []
    assert [g["learner"] for g in s["gpu_notes"]] == ["tabm", "T", "mlp*"]
    s, pr = _select("pstar")
    assert s["panels"]["a"] == "P0@soil" and s["a"]["l29"]["h42_pick"] == "P0@tddm" and s["a"]["l29"]["replaced"] and pr == []
    assert s["b"]["Pr"] == "P1@ku" and s["panels"]["b"] == "R1" and s["panels"]["c"] == "R1"
    s, pr = _select("replace1")                                            # 목록 밖 P1@tddm → P1@soil, R1 − P1@soil 은 lgw_map_aux 에서
    assert s["b"]["l29"]["h42_pick"] == "P1@tddm" and s["b"]["l29"]["replaced"] and s["b"]["Pr"] == "P1@soil" and s["panels"]["b"] == "R1"
    assert {r["source"] for r in s["b"]["R1_minus_P1star"]["dominant"]["rows"]} == {"lgw_map_aux"} and pr == []
    s, pr = _select("b2")
    assert s["d"]["normalizer"] == "nflow" and s["placement"].startswith("main Fig 6c") and pr == []
    assert all(r["pool"] == "지역 4/4" for r in _select("all_ml")[0]["b"]["R1_minus_P1"]["dominant"]["rows"])


def test_rule_two_candidates_and_lena_block():
    t = h49.synthetic_tables("pstar")
    aux4 = pd.DataFrame([_row("R0-P0|n0", "MEAN4", M4, "우세", -0.9, 0, "aux"), _row("R0-P0|n0", "region", "Lena|x", "미결정", 0.1, 0, "aux")])
    book = h49.VerdictBook(dict(lgx_lg_aux=t["lgx_lg_aux"], lgx_tests=t["lgx_tests"], lgw_bundle=t["lgw_bundle"], lgw_map_aux=aux4))
    s = h49.rule_select(book, h49.lgu_b2(None), h49.l8_verdict(None))
    assert s["panels"]["a"] == "P0@soil"                                  # P0@soil(−1.5) 가 R0(−0.9) 보다 낮다
    aux4.loc[0, "delta"] = -2.0
    book = h49.VerdictBook(dict(lgx_lg_aux=t["lgx_lg_aux"], lgx_tests=t["lgx_tests"], lgw_bundle=t["lgw_bundle"], lgw_map_aux=aux4))
    assert h49.rule_select(book, h49.lgu_b2(None), h49.l8_verdict(None))["panels"]["a"] == "R0"
    lgx = t["lgx_lg_aux"].copy()
    lgx.loc[(lgx.contrast == "R1-P1|n10") & (lgx.scope == "region"), "verdict4"] = "열세"
    book = h49.VerdictBook(dict(lgx_lg_aux=lgx, lgx_tests=t["lgx_tests"], lgw_bundle=t["lgw_bundle"], lgw_map_aux=aux4))
    s = h49.rule_select(book, h49.lgu_b2(None), h49.l8_verdict(None))
    assert s["panels"]["b"] == "P1@ku"                                    # 레나 행이 열세면 R1 을 고르지 않는다(Pr = P1*)


# ---------------------------------------------------------------- (f)
def test_b2_requires_sigma(tmp_path):
    ctx, _ = h49.synthetic_ctx(n_grid_side=8)
    with pytest.raises(SystemExit):
        h49.run(ctx, h49.synthetic_tables("b2"), tmp_path, 1, dict(synthetic=True))
    assert (tmp_path / "lena_pred_v1_selection.json").exists()            # 선택 기록은 예측 전에 쓴다


# ---------------------------------------------------------------- (g)
def test_p1_r1_match_h40():
    H = h49.h40()
    ctx, _ = h49.synthetic_ctx(n_grid_side=6)
    F = h49.Fitter(threads=1)
    sel = ctx.draws_by_split[1][0]
    lab = ctx.lena_A.sub(sel)
    (p1,), info = h49.fit_predict("P1", ctx.src, lab, [ctx.grid], F)
    E0 = H.ls_E(ctx.src.y, ctx.src.s)
    Els = H.ls_E(lab.y, lab.s)
    En = (len(sel) * Els + 10.0 * E0) / (len(sel) + 10.0)
    assert np.isclose(info["E_n"], En) and np.allclose(p1, En * ctx.grid.s, equal_nan=True)
    (r1,), _ = h49.fit_predict("R1", ctx.src, lab, [ctx.grid], F)
    from types import SimpleNamespace
    args = SimpleNamespace(cb_iters=200, threads=1)
    X = np.vstack([ctx.src.X, lab.X]).astype(np.float32)
    y = np.concatenate([ctx.src.y - E0 * ctx.src.s, lab.y - En * lab.s])
    g = np.mean([H.cb_fit(args, X, y, sd).predict(ctx.grid.X.astype(np.float32), thread_count=1) for sd in (0, 1)], axis=0)
    assert np.allclose(r1, En * ctx.grid.s + 0.25 * g, equal_nan=True)
    assert F.n_fit == 2


# ---------------------------------------------------------------- (h)
def test_anchor_baselines_match_h42():
    H2 = h49.h42()
    ctx, _ = h49.synthetic_ctx(n_grid_side=6)
    F = h49.Fitter(threads=1)
    none = h49.empty_rows(ctx.grid.X.shape[1])
    (p,), info = h49.fit_predict("P0@ku", ctx.src, none, [ctx.grid], F)
    b, rho, _ = H2.anchor_fill(ctx.src.anc["ku"], np.zeros(0), ctx.grid.anc["ku"], ctx.src.s, np.zeros(0), ctx.grid.s)
    c0 = h49.ls_E(ctx.src.y, b["src"])
    assert np.isclose(info["rho"], rho) and np.allclose(p, c0 * b["B"], equal_nan=True)
    (e,), _ = h49.fit_predict("B:ens", ctx.src, none, [ctx.grid], F)
    bc, _, _ = H2.anchor_fill(ctx.src.anc["cci"], np.zeros(0), ctx.grid.anc["cci"], ctx.src.s, np.zeros(0), ctx.grid.s)
    E0 = h49.ls_E(ctx.src.y, ctx.src.s)
    ref = (E0 * ctx.grid.s + c0 * b["B"] + h49.ls_E(ctx.src.y, bc["src"]) * bc["B"]) / 3.0
    assert np.allclose(e, ref, equal_nan=True)
    (sa,), _ = h49.fit_predict("B:s_aff", ctx.src, none, [ctx.grid], F)
    a_, b_ = H2.affine_ls(ctx.src.s, ctx.src.y)
    assert np.allclose(sa, a_ + b_ * ctx.grid.s)
    (ca,), _ = h49.fit_predict("B:cci_aff", ctx.src, none, [ctx.grid], F)
    a2, b2 = H2.affine_ls(ctx.src.anc["cci"], ctx.src.y)
    assert np.allclose(ca, a2 + b2 * bc["B"], equal_nan=True)
    assert F.n_fit == 0


# ---------------------------------------------------------------- (i)
def test_calibration_matches_manual():
    import polar.lgu_common as LC
    ctx, _ = h49.synthetic_ctx(n_grid_side=6)
    cal = h49.calibrate("P0", ctx.src, h49.Fitter(threads=1))
    src = ctx.src
    ok = np.isfinite(src.y) & (src.y > 0) & (src.s > 0)
    G = sorted(r for r in np.unique(src.macro[ok]) if (src.macro[ok] == r).sum() >= 30)
    assert cal["groups"] == G and "Tibet" not in G and "Tibet" in cal["groups_lgu20"]
    sc, gr = [], []
    for k in G:
        E = h49.ls_E(src.y[src.macro != k], src.s[src.macro != k])
        m = ok & (src.macro == k)
        sc.append(np.abs(np.log(src.y[m] / np.maximum(E * src.s[m], 1.0)))); gr.append(np.full(m.sum(), k))
    q = float(LC.hier_quantiles(np.concatenate(sc), np.concatenate(gr), [0.9])[0])
    assert np.isclose(cal["q"], q)
    lo, hi, w = h49.interval(np.array([50.0, 0.2]), q)
    assert np.allclose(lo, [50 * np.exp(-q), np.exp(-q)]) and np.allclose(w, hi - lo)


# ---------------------------------------------------------------- (j)
def test_extrap_flags():
    Xs = np.tile(np.arange(1000.0)[:, None], (1, 3))
    Xg = np.array([[500.0, 500.0, 500.0], [-5.0, 500.0, np.nan], [2000.0, -1.0, np.nan], [2000.0, 2000.0, 2000.0]])
    df, info = h49.extrap_flags(Xs, Xg, ["a", "b", "c"], np.ones(4, bool))
    assert df.n_extrap_out.tolist() == [0, 1, 2, 3] and df.n_x25_missing.tolist() == [0, 1, 1, 0]
    assert df.extrap_cat.tolist() == [0, 2, 3, 3] and info["top3"][0]["column"] == "a"


# ---------------------------------------------------------------- (k)
def test_run_guards():
    with pytest.raises(SystemExit):
        h49.main([])                                                      # --allow-run 없음
    with pytest.raises(SystemExit):
        h49.main(["--threads", "5", "--synthetic-test", "/nonexistent"])


# ---------------------------------------------------------------- (l)
def test_grid_index_and_compare():
    g = grid.build_grid()
    assert len(g) == 53011 and g.cell_id.is_unique
    assert g.lat.min() >= 71.5 and g.lat.max() <= 73.6 and g.lon.min() >= 123.3 and g.lon.max() <= 130.1
    r = grid.compare_cols(np.array([1.0, np.nan, np.nan, 2.0]), np.array([1.0, np.nan, 3.0, 2.0 + 1e-7]), "abs")
    assert r["n_both_nan"] == 1 and r["n_nan_mismatch"] == 1 and r["n_over_tol"] == 0 and np.isclose(r["frac_mismatch"], 0.25)
    r = grid.compare_cols(np.array([100.0, 0.0, 50.0]), np.array([100.005, 0.0, 50.01]), "rel")
    assert r["n_over_tol"] == 1                                            # 50.01 대 50.0 은 상대 차 2e-4
    with pytest.raises(SystemExit):
        grid.main(["--out-dir", "data/processed/lgx/map", "--stage", "grid"])


# ---------------------------------------------------------------- (m)
def test_figure_raster_and_caption():
    fig = _load("fig_map_lena", "scripts/4_visualization/paper/fig_map_lena.py")
    C = _load("paper_common", "scripts/4_visualization/paper/_common.py")
    g = grid.build_grid().iloc[::97].reset_index(drop=True)
    R = fig.Raster(g, fig.laea(), res=2000.0)
    m = R.idx >= 0
    assert m.any()
    import cartopy.crs as ccrs
    x0, x1, y0, y1 = R.ext
    iy, ix = np.where(m)
    xs = x0 + (ix + 0.5) * 2000.0; ys = y0 + (iy + 0.5) * 2000.0
    ll = ccrs.PlateCarree().transform_points(fig.laea(), xs, ys)
    ky, kx = grid.cells_module().cell_index(ll[:, 1], ll[:, 0])
    j = R.idx[iy, ix]
    assert np.array_equal(ky, g.ky.values[j]) and np.array_equal(kx, g.kx.values[j])
    ctx_meta = dict(selection=dict(panels=dict(a="P0", b="R1", c="P1@ku", d="P0"), c=dict(caption_L8=True),
                                   gpu_notes=[dict(learner="tabm", n=10, delta=-0.7)]),
                    caption=dict(panels=dict(a=dict(method="P0"), b=dict(method="R1"), c=dict(method="P1@ku")),
                                 interval=dict(q90=0.61, groups=["Alaska", "Canada"], normalizer="const"),
                                 panel_b=dict(split=1, E_n_same_split=[3.1, 3.3], mean_alt_same_split=[80.0, 86.0], n_splits=5, E_n_all=[3.0, 3.5]),
                                 gray=dict(n_gray=10, n_cells=100), extrap_top3=[dict(column="dem_slope")],
                                 replaced=[dict(h42_pick="P0@tddm", pick="P0@soil")]))
    for layout in ("all6", "main3"):
        cap = fig.build_caption(ctx_meta, layout, None)
        assert C.caption_problems(cap) == [], C.caption_problems(cap)


# ---------------------------------------------------------------- (n) 본 실행 입력 확인
def _drop(t, key, cond):
    t = dict(t)
    df = t[key]
    t[key] = df[~cond(df)].reset_index(drop=True)
    return t


def test_strict_input_check(tmp_path):
    base = h49.synthetic_tables("replace1")
    cases = [
        (lambda t: _drop(t, "lgw_map_aux", lambda d: d.contrast == "R0-P0|n0"), "R0 − P0(n 0) 층화 평균 행 없음"),
        (lambda t: _drop(t, "lgw_map_aux", lambda d: (d.contrast == "R0-P0|n0") & (d.scope == "region")), "R0 − P0(n 0) 레나 행 없음"),
        (lambda t: _drop(t, "lgw_bundle", lambda d: d.ab == "AB5"), "R1 − P1(n 10) 층화 평균 행 없음 (lgw_bundle)"),
        (lambda t: _drop(t, "lgx_lg_aux", lambda d: d.contrast == "R1-P1|n-1"), "R1 − P1(n -1) 층화 평균 행 없음 (lgx_lg_aux)"),
        (lambda t: _drop(t, "lgw_map_aux", lambda d: d.contrast == "R1-P1@soil|n10"), "R1 − P1@soil(n 10) 층화 평균 행 없음"),
        (lambda t: _drop(t, "lgx_tests", lambda d: d.contrast.str.endswith("-P1|n-1")), "L29 기준선 − P1(n -1)"),
        (lambda t: dict(t, lgw_map_aux=t["lgw_map_aux"].assign(nboot_capped=True)), "재표집 제한"),
        (lambda t: dict(t, lgw_bundle=t["lgw_bundle"].assign(nboot=1000)), "nboot < 10,000"),
        (lambda t: dict(t, lgx_lg_aux=t["lgx_lg_aux"].assign(contrast=t["lgx_lg_aux"].contrast + "|v2")), "lgx_lg_aux.csv: 해석한 대비 행 0"),
        (lambda t: dict(t, lgw_map_aux=None), "lgw_map_aux.csv 없음"),
        (lambda t: dict(t, lgu_b_tests=None), "LGU-B2 행 없음"),
    ]
    for f, msg in cases:
        _, pr = _select("replace1", f(base))
        assert any(msg in p for p in pr), (msg, pr)
    ctx, _ = h49.synthetic_ctx(n_grid_side=6)
    bad = cases[0][0](base)
    with pytest.raises(SystemExit):
        h49.run(ctx, bad, tmp_path, 1, dict(synthetic=True), strict=True)
    assert not (tmp_path / "lena_pred_v1_selection.json").exists()        # 입력이 불완전하면 선택을 기록하지 않는다
    b = h49.VerdictBook(dict(lgx_lg_aux=base["lgx_lg_aux"]))
    assert b.stats["lgx_lg_aux"]["n_parsed"] == len(base["lgx_lg_aux"]) and b.stats["lgx_lg_aux"]["n_unparsed_label"] == 0


def test_mean_target_main4_and_pool():
    rows = [_row("R0-P0|n0", "MEAN", "MEAN[Lena|x,Canada|x,Alaska|x]", "우세", -1.0), _row("R0-P0|n0", "MEAN", "MEAN[Lena|x,Canada|x,Russia_W|x]",
                                                                                          "우세", -1.0)]
    rows[1].update(pool="부분(지역 3/4)")
    b = _book(lgw_map_aux=rows)
    assert b.stats["lgw_map_aux"]["n_mean_not_main4"] == 1 and b.stats["lgw_map_aux"]["n_mean_main4"] == 1
    r = h49.dominant(b, "R0", "P0", 0)
    assert r["ok"] and len(r["rows"]) == 1 and r["rows"][0]["pool"] == "부분(지역 3/4)"   # 주 4지역 밖 지역이 든 평균 행은 쓰지 않는다
    assert h49.mean_regions("MEAN[]") == [] and h49.mean_regions("Lena|x") == []


# ---------------------------------------------------------------- (o) GPU 학습기 사실
def test_gpu_notes():
    p = h49.parse_learner_label
    assert p("R1[T]-R1[C]|n10|lam0.25") == dict(ma="R1", la="T", mb="R1", lb="C", n=10, lam=0.25)
    assert p("R1[mlp*]-R1[CB]|nall")["n"] == -1 and p("R1[mlp]-R1[catboost_lo]|n-1")["n"] == -1
    assert p("R1[mlp*]-R1[mlp]|n10|lam1|nondefault") is None and p("R1-P1|n10") is None
    t = h49.synthetic_tables("all_ml")
    g = h49.gpu_learner_notes(t)
    assert [(x["learner"], x["experiment"]) for x in g["notes"]] == [("tabm", "LG L5"), ("T", "LGT L32"), ("mlp*", "LGF-N3")]
    assert g["notes"][1]["mark"] == "등록 시점에 결과 존재(미열람)" and g["notes"][0]["mark"] == ""       # λ 1.0 행과 지역 행은 쓰지 않는다
    g = h49.gpu_learner_notes(dict(t, lgt_tests=None))
    assert g["status"]["lgt_tests"] == "없음" and "LGT L32" not in [x["experiment"] for x in g["notes"]]
    assert h49.learner_en("mlp*") == "MLP (tuned)" and h49.learner_en("T") == "TabPFN"


# ---------------------------------------------------------------- (p) 출력 폴더 보호
def test_output_protection():
    for bad in ("data/processed/map_lena/syn", str(ROOT / "data/processed/lgx/syn"), str(ROOT / "results/syn")):
        with pytest.raises(SystemExit):
            h49.main(["--synthetic-test", bad, "--threads", "1"])
    fig = _load("fig_map_lena", "scripts/4_visualization/paper/fig_map_lena.py")
    for bad in ("outputs/maps/transfer_lena/demo", "data/processed/map_lena/demo", "data/processed/lgu/demo"):
        with pytest.raises(SystemExit):
            fig.main(["--demo-grid", bad])
    with pytest.raises(SystemExit):
        h49.check_out(ROOT / "data/processed/ext_labels/x")
    assert h49.check_out(ROOT / "data/processed/map_lena") == ROOT / "data/processed/map_lena"


# ---------------------------------------------------------------- (q) 회색 육지
def test_land_gray_stats():
    g = pd.DataFrame(dict(land=[1, 1, 1, 0, 0], gray=[0, 1, 1, 1, 0], gray_reason=["", "cci_missing", "pfr_missing", "water", ""]))
    r = h49.land_gray_stats(g)
    assert r["n_land"] == 3 and r["n_land_gray"] == 2 and np.isclose(r["frac_land_gray"], 2 / 3)
    assert r["land_by_reason_first"] == {"cci_missing": 1, "pfr_missing": 1}
    assert h49.land_gray_stats(g.drop(columns="land")) == {}


# ---------------------------------------------------------------- (r) h49_map_contrasts
def _tm(name, specs, nb=10, splits=(1, 2), nboot=200, seed=0):
    """h39 시험의 mk_tm 과 같은 합성 저장소. specs = {키: (오차 SD, 잡음 묶음, 치우침)}."""
    H = h49.h40()
    rng = np.random.RandomState(seed)
    by = {}
    for sp in splits:
        blk = np.repeat(np.arange(nb), 5)
        st = H.BlockStore(name, sp, blk)
        y = rng.randn(len(blk)) * 5 + 50
        noise = {"p0": rng.randn(len(y))}
        st.add(H.P0_KEY, y, y + 6.0 * noise["p0"])
        for k, (sd, grp, bias) in specs.items():
            if grp not in noise:
                noise[grp] = rng.randn(len(y))
            st.add(k, y, y + bias + sd * noise[grp])
        by[sp] = st
    tm = H.TMx(name, by, {sp: dict(dup_of=-1, valid=True) for sp in splits}, nboot)
    tm.base_target = tm.target
    tm.expected_splits = list(splits)
    return tm


def test_map_contrasts_table():
    S = mapc.h39()
    k8 = lambda g, d=0, sd=0: (g[0], g[1], g[2], g[3], g[4], d, sd, g[5])         # noqa: E731
    r0 = k8(S.G_lg("R0", 0))
    r1x = {n: k8(S.X.G("R1", n)) for n in (10, -1)}
    p1s = {n: k8(S.X.G("P1@soil", n), sd=-1) for n in (10, -1)}
    tms_lg = {f"{t}|x": _tm(f"{t}|x", {r0: (1.0, "g", 0.0)}, seed=i) for i, t in enumerate(S.MAIN4)}
    tms_lgx = {f"{t}|x": _tm(f"{t}|x", {**{r1x[n]: (1.0, "r", 0.0) for n in r1x}, **{p1s[n]: (6.0, "p0", 0.0) for n in p1s}}, seed=10 + i)
               for i, t in enumerate(S.MAIN4)}
    df = mapc.map_aux_table(tms_lg, tms_lgx, 200)
    m = df[df.scope == "MEAN"].set_index("contrast")
    assert set(m.index) == {c for _, c, _, _, _ in mapc.map_contrasts()}
    assert m.loc["R0-P0|n0", "verdict4"] == "우세" and m.loc["R1-P1@soil|n10", "verdict4"] == "우세"
    assert m.loc["R1-B:ens|n10", "verdict4"] == "행 없음" and m.loc["R1-P1@ku|n-1", "verdict4"] == "행 없음"
    reg = df[(df.scope == "region") & (df.contrast == "R0-P0|n0")]
    assert sorted(reg.target) == sorted(f"{t}|x" for t in S.MAIN4)
    b = h49.VerdictBook(dict(lgw_map_aux=df.assign(nboot=10000, nboot_capped=False)))
    assert b.stats["lgw_map_aux"]["n_unparsed_label"] == 0
    assert h49.dominant(b, "R0", "P0", 0)["ok"] and h49.lena_not_inferior(b, "R0", "P0", 0)["ok"]
    assert h49.dominant(b, "R1", "P1@soil", -1)["ok"] and not h49.dominant(b, "R1", "B:ens", 10)["ok"]
    assert S.region_stat(tms_lg["Lena|x"], S.G_lg("R0", 0), S.H.P0_GRP, "MAP-a|R0-P0|n0", 200)["delta"] == \
        pytest.approx(float(df[(df.scope == "region") & (df.target == "Lena|x") & (df.contrast == "R0-P0|n0")].delta.iloc[0]))
    with pytest.raises(SystemExit):
        mapc.parse_args(["--out-dir", "data/processed/lgx/map"])
    a = mapc.parse_args(["--out-dir", "data/processed/wrapup", "--threads", "3"])
    assert a.threads == 1 and a.nboot == 1000 and a.CAPPED                   # 허용 표지 없으면 스레드 1, 1,000회로 낮춘다
    with pytest.raises(SystemExit):
        mapc.main(["--out-dir", "data/processed/wrapup"])                    # --allow-run 없으면 조각을 열기 전에 거부한다


# ---------------------------------------------------------------- (s) real_ctx 와 h40 의 직접 대조
def _synth_D(seed=3):
    H = h49.h40()
    rng = np.random.RandomState(seed)
    rows = []
    for mac, nb, lat0, lon0, E in (("Lena", 8, 72.0, 126.0, 3.2), ("Alaska", 8, 66.0, -150.0, 4.8), ("Canada", 6, 62.0, -120.0, 4.2),
                                   ("Russia_W", 4, 67.0, 70.0, 3.6)):
        for b in range(nb):
            for c in range(10):
                rows.append(dict(macro=mac, block=f"{mac[:2]}{b:03d}", lat=lat0 + (b // 4) * 0.5 + rng.rand() * 0.02,
                                 lon=lon0 + (b % 4) * 0.5 + rng.rand() * 0.02, E=E))
    df = pd.DataFrame(rows)
    n = len(df)
    for c in H.FEATS:
        df[c] = rng.randn(n)
    df["e5_sqrt_tdd"] = rng.uniform(20, 40, n)
    df["cci_alt"] = rng.uniform(40, 90, n)
    df.loc[rng.rand(n) < 0.05, "cci_alt"] = np.nan
    df["e5_sqrt_tdd_soil"] = df.e5_sqrt_tdd + 1.0
    df["p4_ku"] = rng.uniform(40, 90, n); df["p2_edaphic"] = rng.uniform(40, 90, n)
    df[H.TARGET] = df.E * df.e5_sqrt_tdd * np.exp(rng.randn(n) * 0.2 + 0.1 * df[H.FEATS[0]])
    df["s"] = df.e5_sqrt_tdd.values.astype(float); df["y"] = df[H.TARGET].values.astype(float)
    df["z"] = np.log(df.y) - np.log(df.s)
    df["loc_id"] = np.arange(n); df["sub"] = ""
    D = H.Data.__new__(H.Data)
    D.df = df
    D.args = H.parse_args(["--splits", "5", "--threads", "1"])
    D.subs = pd.DataFrame(dict(subregion=[], parent=[]))
    D.macros = set(df.macro.unique()); D.sub_parent = {}; D._src = {}; D._split = {}
    D.sub_src = "synthetic"; D.sub_kmeans_diff = 0
    return D


def test_real_ctx_matches_h40(monkeypatch):
    H = h49.h40()
    D = _synth_D()
    gdf = D.df[D.df.macro == "Lena"].head(20).assign(cell_id=[f"c{i}" for i in range(20)], ky=0, kx=0).reset_index(drop=True)
    ctx = h49.real_ctx(None, 1, D=D, grid_df=gdf)
    sp = ctx.split
    args = H.parse_args(["--methods", "P1,R1", "--n-grid", "10", "--draws", "1", "--seeds", "1", "--learners", "catboost_lo", "--lams", "0.25",
                         "--alphas", "1", "--place-n-grid", "40", "--threads", "1", "--splits", "5"])
    c = H.build_ctx(D, args, "Lena", "x", sp)
    assert np.isclose(h49.ls_E(ctx.src.y, ctx.src.s), c.E0)
    assert np.array_equal(ctx.src.X, c.X_src, equal_nan=True) and np.array_equal(ctx.src.y, c.y_src)       # x25 의 cci_alt 결측은 NaN
    assert np.array_equal(ctx.lena_A.X, c.XA, equal_nan=True) and np.array_equal(ctx.lena_A.y, c.yA)
    sel = ctx.draws_by_split[sp][0]
    assert np.array_equal(sel, H.draw_cells("Lena", "x", sp, 10, 0, len(c.yA)))
    rows, st, _ = H.run_ctx(c, "cpu", args, learner_axis=False)
    ref = {r["method"]: r["rmse_cm"] for r in rows if r["n"] == 10 and r["method"] in ("P1", "R1") and r["draw"] == 0 and r["seed"] in (-1, 0)}
    monkeypatch.setattr(h49, "SEEDS", (0,))                                  # h40 행은 seed 하나(0)의 예측이다
    tgt = h49.Rows(c.XB, c.sB, {k: np.full(len(c.sB), np.nan) for k in h49.ANCHOR_COL})
    F = h49.Fitter(threads=1)
    lab = ctx.lena_A.sub(sel)
    for m in ("P1", "R1"):
        (p,), _ = h49.fit_predict(m, ctx.src, lab, [tgt], F)
        e = p - c.yB
        assert np.isclose(np.sqrt(np.mean(e[np.isfinite(e)] ** 2)), ref[m], rtol=0, atol=1e-9), m


# ---------------------------------------------------------------- (t) 패널 문자와 캡션
def _cap_meta(gpu=None, missing=None):
    return dict(selection=dict(panels=dict(a="P0", b="R1", c="P1@ku", d="P0"), c=dict(caption_L8=True), gpu_notes=gpu or [],
                               gpu_missing=missing or []),
                caption=dict(panels=dict(a=dict(method="P0"), b=dict(method="R1"), c=dict(method="P1@ku")),
                             interval=dict(q90=0.61, groups=["Alaska", "Canada"], normalizer="const"),
                             panel_b=dict(split=1, E_n_same_split=[3.1, 3.3], mean_alt_same_split=[80.0, 86.0], n_splits=5, E_n_all=[3.0, 3.5],
                                          mean_alt_all=[78.0, 92.0]),
                             gray=dict(n_gray=10, n_cells=100), extrap_top3=[dict(column="dem_slope")], gpu_missing=missing or [],
                             gray_land=dict(n_land=38173, n_land_gray=87, frac_land_gray=87 / 38173, land_by_reason_first=dict(pfr_missing=10,
                                                                                                                          cci_missing=77)),
                             replaced=[dict(h42_pick="P0@tddm", pick="P0@soil")]))


def test_caption_facts_and_letters():
    fig = _load("fig_map_lena", "scripts/4_visualization/paper/fig_map_lena.py")
    C = _load("paper_common", "scripts/4_visualization/paper/_common.py")
    gpu = h49.gpu_learner_notes(h49.synthetic_tables("all_ml"))["notes"]
    for layout in ("all6", "main3"):
        for meta in (_cap_meta(gpu), _cap_meta(missing=["lgt_tests.csv"])):
            cap = fig.build_caption(meta, layout, None)
            assert C.caption_problems(cap) == [], C.caption_problems(cap)
            assert fig.caption_letters(cap) == fig.LAYOUT_LETTERS[layout]
            assert "Of 38,173 land cells, 87 (0.2%) are grey (PFR missing 10, CCI ALT missing 77)." in cap["data"]
            assert "not inferior in Lena, otherwise P0" in cap["statistics"]
    cap = fig.build_caption(_cap_meta(gpu), "all6", None)
    assert "(78–92 cm)" in cap["panels"] and "TabPFN (LGT L32, n = 10" in cap["panels"] and "unviewed" in cap["panels"]
    assert "b and c are Supplementary panels d and e" in fig.build_caption(_cap_meta(), "main3", None)["panels"]
    assert "LGT/LGF learner comparisons not included" in fig.build_caption(_cap_meta(missing=["lgt_tests.csv"]), "all6", None)["panels"]
    gm = dict(masks=dict(summary=dict(n_land=10, n_land_gray=1, frac_land_gray=0.1, land_by_reason_first=dict(water=0, cci_missing=1))))
    m2 = _cap_meta(); m2["caption"].pop("gray_land")
    assert "Of 10 land cells, 1 (10.0%) are grey (CCI ALT missing 1)." in fig.build_caption(m2, "all6", gm)["data"]
    assert "PFR < 10% or missing" in fig.MASK_LEGEND


def test_main3_figure_letters():
    import matplotlib.pyplot as plt
    fig = _load("fig_map_lena", "scripts/4_visualization/paper/fig_map_lena.py")
    g = grid.build_grid().iloc[::41].reset_index(drop=True)
    p = g[["cell_id", "ky", "kx", "lat", "lon"]].copy()
    p["gray"] = 0; p["gray_reason"] = ""
    for c in ("pred_a", "pred_b", "pred_c"):
        p[c] = 50.0 + 10 * np.sin(p.lat)
    p["width90_a"] = 30.0; p["diff_c_minus_a"] = 0.5; p["extrap_cat"] = 1
    for layout in ("main3", "all6"):
        f, _ = fig.build_figure(p, _cap_meta(), layout)
        cap = fig.build_caption(_cap_meta(), layout, None)
        r = fig.panel_letters_check(f, cap, layout)
        assert r["panel_letters_ok"], r
        plt.close(f)
