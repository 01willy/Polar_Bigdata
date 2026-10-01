"""F3 2부 그림 값 대조(스펙 QA 6단계). 그림에 찍힌 값(source_data/v2)을 원천 표에서 다시 읽어 비교하고
_qa/v2_<이름>_values.txt 에 쓴다. 파생 층(data/processed/paper_figs)을 거치지 않고 원천 표를 직접 읽는다.
대상: Fig 1, Fig 5, Fig 7, Table 1 과 Fig 2–4 의 2부 추가분(AB 표지, L10 모든 n, SD/SE 표지, L3 보조).
      3부: Fig 6(라벨 0 포레스트 21선, 정규화기 30점과 C2 참조, 구간 점수 % 환산, L25 6칸, 캡션 수치). --only fig6 로 따로 돈다.
실행: CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/4_visualization/paper/v2_values_check.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "outputs" / "figures" / "paper" / "source_data" / "v2"
QA = ROOT / "outputs" / "figures" / "paper" / "_qa"
LGW = ROOT / "data" / "processed" / "lgw"
LGX = ROOT / "data" / "processed" / "lgx"
LGS = ROOT / "results" / "rescale_lg" / "data" / "processed" / "lg"
TOL = 1e-9


def rd(p):
    return pd.read_csv(p, low_memory=False)


class Log:
    def __init__(self):
        self.rows, self.ok = [], True

    def num(self, label, fig, src, tol=TOL):
        ok = bool(np.isclose(float(fig), float(src), atol=tol, rtol=0))
        self.ok &= ok
        self.rows.append(f"{'OK ' if ok else 'BAD'} {label}: figure {float(fig):.6f} | source {float(src):.6f}")

    def eq(self, label, fig, src):
        ok = str(fig) == str(src)
        self.ok &= ok
        self.rows.append(f"{'OK ' if ok else 'BAD'} {label}: figure {fig} | source {src}")

    def write(self, name):
        (QA / f"v2_{name}_values.txt").write_text("\n".join(self.rows) + "\n")
        print(f"[values] {name}: {'PASS' if self.ok else 'FAIL'} ({len(self.rows)} checks)")
        return self.ok


def fig7():
    L = Log()
    D = rd(SRC / "Fig7_d.csv"); B = rd(LGW / "lgw_bundle.csv"); B = B[B.scope == "MEAN"]
    for ab in ("AB1", "AB4", "AB5", "AB6", "AB8", "AB9"):
        f, s = D[D.ab == ab].iloc[0], B[B.ab == ab].iloc[0]
        L.num(f"d {ab} delta", f.delta, s.delta); L.num(f"d {ab} ci_hi", f.ci_hi, s.ci_hi); L.eq(f"d {ab} verdict4", f.verdict4, s.verdict4)
    L.num("d AB5 holm_p", D[D.ab == "AB5"].holm_p.iloc[0], 0.136, tol=5e-4)
    A10 = rd(SRC / "Fig7_d_ab10.csv").iloc[0]
    U = rd(ROOT / "data/processed/lgu/lgu_a_tests.csv"); u = U[(U.test == "LGU-A1") & (U.n.astype(str) == "10")].iloc[0]
    L.num("d AB10 delta (cm)", A10.delta_cm, u.delta); L.num("d AB10 ci_lo (cm)", A10.ci_lo_cm, u.ci_lo); L.num("d AB10 ci_hi (cm)", A10.ci_hi_cm, u.ci_hi)
    L.eq("d AB10 nboot", int(A10.nboot), 1000)
    C = rd(SRC / "Fig7_b_cells.csv"); S = rd(LGW / "lgw_scenarios.csv")
    for st, rec, ref, tg in (("T10", "P1", "P0", "Canada|x"), ("T3", "R1", "P0", "Russia_W|x"), ("T40", "R1", "P1", "Lena|x")):
        f = C[(C.stage == st) & (C.recipe == rec) & (C.reference == ref) & (C.target == tg)].iloc[0]
        s = S[(S.stage == st) & (S.recipe == rec) & (S.reference == ref) & (S.target == tg)].iloc[0]
        L.num(f"b {st} {rec}-{ref} {tg} delta", f.delta, s.delta); L.eq(f"b {st} {rec}-{ref} {tg} noninf", f.noninf, s.noninf)
    K = rd(SRC / "Fig7_c.csv"); Kr = rd(LGW / "lgw_sc3.csv")
    for tau in (0.05, 0.2):
        f = K[(K.scope == "MEAN3") & np.isclose(K.tau, tau)].iloc[0]; s = Kr[(Kr.scope == "MEAN3") & np.isclose(Kr.tau, tau)].iloc[0]
        L.num(f"c tau {tau} label_ratio", f.label_ratio, s.label_ratio); L.num(f"c tau {tau} delta", f.delta, s.delta)
    T = rd(SRC / "Fig7_a_texts.csv").set_index("key").text
    L.eq("a SC1w text has 4/10", "4/10" in T["SC1w"], True)
    L.eq("a SC3w text has 0.91", "0.91" in T["SC3w"], True)
    L.eq("a SC1w-P text has +0.97 and +2.26", ("+0.97" in T["SC1w-P"]) and ("+2.26" in T["SC1w-P"]), True)
    return L.write("Fig7_deployment")


def fig5():
    L = Log()
    A = rd(SRC / "Fig5_a.csv"); R = rd(LGW / "lgw_l43.csv")
    for n, kind, tg in ((10, "2단 재표집", "MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"), (40, "추출 조건부", "MEAN[Lena|x,Canada|x]"),
                        (40, "2단 재표집", "Canada|x")):
        f = A[(A.n == n) & (A.ci_kind == kind) & (A.target == tg)].iloc[0]
        s = R[(R.role == "주") & (R.method == "P1") & (R.n == n) & (R.ci_kind == kind) & (R.target == tg)].iloc[0]
        L.num(f"a n{n} {kind} {tg} delta", f.delta, s.delta); L.num(f"a n{n} {kind} {tg} ci_lo", f.ci_lo, s.ci_lo); L.eq(f"a {tg} verdict", f.verdict4, s.verdict4)
    B = rd(SRC / "Fig5_b.csv"); T = rd(LGX / "lgx_tests.csv")
    for c in ("D0[S_rand5]-D0[S_block]", "[R1-P1](inblk)-[R1-P1](A)|n10"):
        f = B[(B.contrast == c) & (B.kind == "mean")].iloc[0]
        s = T[(T.test_id == "L23") & (T.scope == "MEAN") & (T.contrast == c)].iloc[0]
        L.num(f"b {c} delta", f.delta, s.delta); L.eq(f"b {c} verdict", f.verdict4, s.verdict4)
    Cc = rd(SRC / "Fig5_c.csv")
    f = Cc[(Cc.target == "Alaska|x") & (Cc.n == 160) & (Cc.which == "R1 − P1")].iloc[0]
    s = T[(T.test_id == "L24") & (T.target == "Alaska|x") & (T.contrast == "[R1-P1](근)-[R1-P1](원)|n160")].iloc[0]
    L.num("c Alaska n160 R1-P1 near-far", f.delta, s.delta); L.eq("c Alaska n160 verdict", f.verdict4, s.verdict4)
    Dd = rd(SRC / "Fig5_d.csv"); X = rd(LGX / "lgx_distance.csv")
    f = Dd[(Dd.target == "Canada|x") & (Dd.n == 40) & (Dd.contrast == "P1-P0") & (Dd.stratum == "mid")].iloc[0]
    s = X[(X.target == "Canada|x") & (X.n == 40) & (X.contrast == "P1-P0") & (X.stratum == "mid")].iloc[0]
    L.num("d Canada n40 P1-P0 mid", f.delta, s.delta)
    return L.write("Fig5_placement")


def fig1():
    L = Log()
    Dd = rd(SRC / "Fig1_d.csv").set_index("target")
    m3 = {"Lena": 94.3, "Canada": 81.4, "Russia_W": 78.0, "Russia_E": 78.0, "Russia_C": 83.9, "Greenland": 77.9, "Alaska": 0.0}
    for t, v in m3.items():
        L.num(f"d {t} Alaskan share % vs M3 Table M3-1", round(100 * Dd.loc[t, "share_alaska"], 1), v, tol=1e-9)
    Tg = rd(LGS / "lg_targets.csv")
    for t in ("Lena", "Russia_W", "Alaska"):
        s = Tg[(Tg.target == t) & (Tg["mode"] == "x") & (Tg.part == "cpu") & Tg.E0.notna()].E0.iloc[0]
        L.num(f"d {t} E0 vs lg_targets", Dd.loc[t, "E0"], s)
        L.num(f"d {t} n_src vs lg_targets", Dd.loc[t, "n_src"], Tg[(Tg.target == t) & (Tg["mode"] == "x") & (Tg.part == "cpu") & Tg.n_src.notna()].n_src.iloc[0])
    Sb = rd(SRC / "Fig1_b_summary.csv").set_index("row")
    U = rd(LGW / "lgw_label_units.csv").set_index("target")
    for t in ("Lena", "Canada", "Russia_W", "Alaska"):
        L.num(f"b {t} n_cells vs lgw_label_units n_rows", Sb.loc[t, "n_cells"], U.loc[t, "n_rows"])
    N = rd(SRC / "Fig1_a_not_drawn.csv").set_index("region").n_cells_not_drawn
    Lb = rd(ROOT / "data/processed/fidelity_base_v4_labels.csv")
    q = Lb[(Lb.part == "new") & (Lb.lgd_role == "target") & (Lb.lic_unverified == 1)].macro_v4.value_counts()
    for r in N.index:
        L.num(f"a not drawn (licence) {r}", N[r], q[r])
    Bk = rd(SRC / "Fig1_a_blocks.csv")
    L.num("a v3 cells drawn = 17,467", Bk[Bk.kind == "v3"].n_rows.sum(), 17467)
    L.eq("a NAtlantic not drawn as LGD target", int((Bk.kind == "lgd_added").sum() > 0 and (Bk[(Bk.kind == "lgd_added")].region == "NAtlantic").sum()), 0)
    return L.write("Fig1_problem")


def table1():
    L = Log()
    D = rd(SRC / "Table1_display.csv").set_index("Target")
    U = rd(LGW / "lgw_label_units.csv").set_index("target")
    for t, name in (("Lena", "Lena"), ("Canada", "Canada"), ("AL-5", "AL-5"), ("LE-2", "LE-2")):
        L.eq(f"{name} rows", D.loc[name, "Rows"], f"{int(U.loc[t, 'n_rows']):,}")
        L.eq(f"{name} 1 km loc.", D.loc[name, "1 km loc."], f"{int(U.loc[t, 'n_loc_1km']):,}")
        L.eq(f"{name} blocks", D.loc[name, "Blocks"], f"{int(U.loc[t, 'n_block']):,}")
    E = rd(ROOT / "data/processed/lgd_eligibility_v1.csv").set_index("spec")
    L.eq("Russia C (expanded) rows vs eligibility n_cells", D.loc["Russia C (expanded)", "Rows"], f"{int(E.loc['Russia_C', 'n_cells'])}")
    L.eq("Tibet rows vs eligibility n_cells", D.loc["Tibet", "Rows"], f"{int(E.loc['Tibet', 'n_cells'])}")
    L.eq("27 targets", int(rd(SRC / "Table1_rows.csv").n_targets.sum()), 27)
    return L.write("Table1_data")


def part2_fig2_4():
    L = Log()
    T = rd(LGX / "lgx_tests.csv")
    B = rd(SRC / "Fig3_b_L10.csv")
    for c in ("R1-F1k|n40", "R1-F1n|n160", "R0-F1k|n0", "R1-F1n|n-1"):
        f = B[B.contrast == c].iloc[0]; s = T[(T.test_id == "L10") & (T.scope == "MEAN") & (T.contrast == c)].iloc[0]
        L.num(f"Fig3b L10 {c} delta", f.delta, s.delta); L.eq(f"Fig3b L10 {c} verdict", f.verdict4, s.verdict4)
        L.eq(f"Fig3b L10 {c} registered", str(f.registered).lower(), str(s.primary).lower())
    S = rd(SRC / "Fig3_c_sdse.csv"); R = rd(LGW / "lgw_splitratio.csv")
    for tg, c in (("Russia_E|x", "R1-P0|all"), ("Canada|x", "R1-P1|all"), ("Lena|x", "R1-P0|all")):
        f = S[(S.target == tg) & (S.contrast == c)].iloc[0]; s = R[(R.target == tg) & (R.contrast == c)].iloc[0]
        L.num(f"Fig3c R/R0 {tg} {c}", f.R_over_R0, s.R_over_R0); L.eq(f"Fig3c flag {tg} {c}", str(f.flag), str(s.flag))
    X = rd(SRC / "Fig4_c_L3aux.csv"); A = rd(LGW / "lgw_aux4.csv"); A = A[A.test_id == "L3"]
    for tg, c, n in (("Canada|x", "R0@nested-P0|n10", 10), ("AL-2|i", "R0@nested-R0@a1|n10", 10), ("AL-5|i", "R0@nested-R0@a1|n10", 10)):
        f = X[(X.target == tg) & (X.contrast == c)].iloc[0]; s = A[(A.target == tg) & (A.contrast == c)].iloc[0]
        L.num(f"Fig4c L3aux {tg} {c}", f.delta, s.delta); L.eq(f"Fig4c L3aux {tg} verdict", f.verdict4, s.verdict4)
    Bd = rd(LGW / "lgw_bundle.csv"); Bd = Bd[Bd.scope == "MEAN"].set_index("ab")
    for fig in ("Fig3", "Fig4"):
        t = rd(SRC / f"{fig}_ab_tags.csv")
        for r in t.itertuples():
            L.eq(f"{fig} tag {r.ab} verdict vs lgw_bundle", r.verdict4_bundle, Bd.loc[r.ab, "verdict4"])
    return L.write("Fig2_4_part2")


# ---------------------------------------------------------------- Fig 6(3부, LGF 창 마감 뒤)
V4 = {"우세": "superior", "동등": "equivalent", "미결정": "undecided", "열세": "inferior", "판정 불가": "not determinable"}
# 그림 선 → 원천 표 행(그림 모듈·v2_data 와 독립으로 다시 적는다). (묶음, 행 키, 선종): (파일, 필터)
FIG6_SRC = {
    ("baselines", "B:ens", "solid"): ("lgw/lgw_bundle.csv", dict(scope="MEAN", ab="AB2")),
    ("baselines", "P*", "solid"): ("lgx/lgx_tests.csv", dict(scope="MEAN", test_id="L29", contrast="P0@tddm-P0|n0")),
    ("direct_rescale", "catboost_lo", "solid"): ("lgw/lgw_bundle.csv", dict(scope="MEAN", ab="AB1")),
    ("direct_rescale", "catboost", "solid"): ("lgx/lgx_tests.csv", dict(scope="MEAN", test_id="L30", contrast="D0[catboost]-P0|n0")),
    ("direct_rescale", "rf", "solid"): ("lgx/lgx_tests.csv", dict(scope="MEAN", test_id="L30", contrast="D0[rf]-P0|n0")),
    ("direct_rescale", "catboost_tuned", "solid"): ("lgx/lgx_tests.csv", dict(scope="MEAN", test_id="L30", contrast="D0[catboost_tuned]-P0|n0")),
    ("direct_rescale", "F1k", "solid"): ("lgx/lgx_tests.csv", dict(scope="MEAN", test_id="L11", contrast="F1k-P0|n0")),
    ("lgt", "tabpfn", "solid"): ("lgt/lgt_tests.csv", dict(scope="MEAN", test_id="L34", contrast="D0[T]-P0|n0")),
    ("lgt", "catboost_ctx", "solid"): ("lgt/lgt_tests.csv", dict(scope="MEAN", test_id="L34", contrast="D0[C]-P0|n0")),
    ("lgf_f", "tabicl", "solid"): ("lgf/lgf_tests.csv", dict(scope="MEAN", test_id="LGF-F1", contrast="D0[I]-P0|n0")),
    ("lgf_f", "tabicl", "dashed"): ("lgf/lgf_tests.csv", dict(scope="MEAN", test_id="LGF-F1", contrast="D0@full[I]-P0|n0")),
    ("lgf_f", "catboost_ctx", "solid"): ("lgf/lgf_tests.csv", dict(scope="MEAN", test_id="LGF-F1", contrast="D0[C]-P0|n0")),
    ("lgf_f", "catboost_ctx", "dashed"): ("lgf/lgf_tests.csv", dict(scope="MEAN", test_id="LGF-F1", contrast="D0@full[C]-P0|n0")),
}
for _lr in ("mlp", "tabm", "ftt", "realmlp"):
    FIG6_SRC[("lgf_n", _lr, "solid")] = ("lgf/lgfn_tests.csv", dict(scope="MEAN", test_id="LGF-N1", contrast=f"D0[{_lr}*]-P0|n0"))
    FIG6_SRC[("lgf_n", _lr, "dashed")] = ("lgf/lgfn_tests.csv", dict(scope="MEAN", test_id="LGF-N1", contrast=f"D0[{_lr}]-P0|n0"))
FIG6_P4 = "MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"


def _row(df, **flt):
    q = df
    for k, v in flt.items():
        q = q[q[k].astype(str) == str(v)]
    if len(q) != 1:
        raise ValueError(f"{len(q)} rows for {flt}")
    return q.iloc[0]


def fig6():
    L = Log()
    P = ROOT / "data" / "processed"
    cache = {}
    tab = lambda f: cache.setdefault(f, rd(P / f))                       # noqa: E731
    # a: 그려진 선(위치·CI·기호) 21개 전부
    Dw = rd(SRC / "Fig6_a_drawn.csv")
    L.eq("a lines drawn = fixed rows", len(Dw), len(FIG6_SRC))
    for (blk, key, var), (f, flt) in FIG6_SRC.items():
        q = Dw[(Dw.block == blk) & (Dw.key == key) & (Dw.variant == var)]
        L.eq(f"a {blk}/{key}/{var} drawn once", len(q), 1)
        if len(q) != 1:
            continue
        g, s_ = q.iloc[0], _row(tab(f), **flt)
        lab = f"a {blk}/{key}/{var} [{f.split('/')[-1]} {flt.get('contrast', flt.get('ab'))}]"
        L.num(lab + " delta", g.delta, s_.delta); L.num(lab + " ci_lo", g.ci_lo, s_.ci_lo); L.num(lab + " ci_hi", g.ci_hi, s_.ci_hi)
        L.eq(lab + " verdict", g.verdict, V4[str(s_.verdict4)])
        L.eq(lab + " P4 target", s_.target, FIG6_P4)
    # 같은 대비 두 원천(B:ens: lgw_bundle h39 와 lgx_tests h42)의 점 추정
    L.num("a B:ens bundle vs L29 point", _row(tab("lgw/lgw_bundle.csv"), scope="MEAN", ab="AB2").delta,
          _row(tab("lgx/lgx_tests.csv"), scope="MEAN", test_id="L29", contrast="B:ens-P0|n0").delta)
    # a: 등록 판정 글자와 지지 갈래
    Rg = rd(SRC / "Fig6_a_registered.csv").set_index("hypothesis")
    X = tab("lgx/lgx_tests.csv")
    L.eq("a L29 registered starts 'P0 보다 우세인 기준선이 있다' and P*", str(_row(X, test_id="L29", scope="verdict").verdict).startswith(
        "P0 보다 우세인 기준선이 있다. P* = P0@tddm") and Rg.loc["L29", "verdict_en"] == "lower-error baseline found", True)
    Lg = rd(LGS / "lg_tests.csv")
    for h, v in (("L1", _row(Lg, test_id="L1", scope="verdict").verdict), ("L30", _row(X, test_id="L30", scope="verdict").verdict),
                 ("L11", _row(X, test_id="L11", scope="verdict_aux").verdict),
                 ("L34(a)", _row(tab("lgt/lgt_tests.csv"), test_id="L34", scope="verdict_aux", clause="(a) 직접 예측").verdict)):
        L.eq(f"a {h} registered 지지 → 'supported'", str(v).startswith("지지") and Rg.loc[h, "verdict_en"] == "supported", True)
    F = _row(tab("lgf/lgf_tests.csv"), test_id="LGF-F1", scope="verdict")
    L.eq("a LGF-F1 support_class (figure 'A')", F.support_class, json.loads(Rg.loc["LGF-F1", "support_detail"])["tabicl"])
    L.eq("a LGF-F1 verdict text first class", str(F.verdict).startswith("지지(물리식보다 오차가 크거나 구별되지 않음)"), True)
    N = _row(tab("lgf/lgfn_tests.csv"), test_id="LGF-N1", scope="verdict")
    code = {"지지(물리식보다 오차가 크거나 구별되지 않음)": "A", "지지(우세 근거 없음)": "B"}
    src_cls = {k: code.get(v, v) for k, v in json.loads(N.support_class).items()}
    L.eq("a LGF-N1 support classes per learner", json.dumps(src_cls, sort_keys=True), json.dumps(json.loads(Rg.loc["LGF-N1", "support_detail"]), sort_keys=True))
    B = tab("lgw/lgw_bundle.csv")
    for ab, rule in (("AB1", "(a)"), ("AB2", "(d)")):
        L.eq(f"a {ab} abstract rule {rule}", str(_row(B, scope="MEAN", ab=ab).abstract_rule).startswith(rule), True)
    # b: 정규화기 6종 × (P4 평균 + 지역 4) 30점과 C2 참조
    Fb = rd(SRC / "Fig6_b.csv")
    I = tab("lgu/lgu_b_intervals.csv")
    L.eq("b points = 30", len(Fb), 30)
    for r in Fb.itertuples():
        s_ = _row(I[(I.n == 0) & (I.scope == "all")], target=r.target, method=r.method)
        L.num(f"b {r.method} {r.target} width", r.wid10, s_.wid10); L.num(f"b {r.method} {r.target} coverage", r.cov10, s_.cov10)
    C2 = tab("h4/c2_coverage.csv")
    c2 = C2[(C2.test == "label0") & (C2.method == "hier2_cdf") & C2.target.astype(str).str.startswith("MEAN")].iloc[0]
    rf = rd(SRC / "Fig6_b_ref.csv"); rf = rf[rf.kind == "mean4"].iloc[0]
    L.num("b C2 hier2_cdf MEAN coverage", rf.coverage, c2.coverage); L.num("b C2 hier2_cdf MEAN width", rf.width_cm, c2.width_cm)
    # c: % 환산과 판정(2단, 1단), AB10
    Fc = rd(SRC / "Fig6_c.csv")
    Ub, Ua = tab("lgu/lgu_b_tests.csv"), tab("lgu/lgu_a_tests.csv")
    for r in Fc.itertuples():
        s_ = _row(Ub, test="LGU-B2") if r.test == "LGU-B2" else _row(Ua, test="LGU-A1", n=str(int(r.n)))
        ref = float(s_.ref_score)
        for c_fig, c_src in (("delta_pct", "delta"), ("ci_lo_pct", "ci_lo"), ("ci_hi_pct", "ci_hi")):
            L.num(f"c {r.test} n{int(r.n)} {c_fig} = 100·{c_src}/ref_score", getattr(r, c_fig), 100.0 * float(s_[c_src]) / ref)
        L.eq(f"c {r.test} n{int(r.n)} verdict (two-stage)", r.verdict, V4[str(s_.verdict)])
        L.eq(f"c {r.test} n{int(r.n)} ‡ flag = 보정 불확실성 의존", bool(r.calib_dependent), str(s_.note_cal) == "보정 불확실성 의존")
        L.eq(f"c {r.test} n{int(r.n)} nboot", int(r.nboot), int(s_.nboot))
    ab10 = _row(B, scope="MEAN", ab="AB10")
    a10 = Fc[Fc.ab == "AB10"].iloc[0]
    L.num("c AB10 tag row delta (cm) = lgw_bundle AB10", a10.delta_cm, ab10.delta)
    L.eq("c AB10 verdict = lgw_bundle", a10.verdict, V4[str(ab10.verdict4)])
    # d: 칸 6개(폭 비, 커버리지, CI)와 c1·c2
    Fd = rd(SRC / "Fig6_d.csv")
    Cf = tab("lgx/lgx_conformal.csv")
    base = Cf[(Cf["mode"] == "x") & (Cf.level == 90) & (Cf.learner == "catboost_lo")]
    for r in Fd.itertuples():
        s1 = _row(base, target=r.target, method="R1", n=int(r.n))
        s0 = _row(base, target=r.target, method="R0", n=0)
        L.num(f"d {r.target} n{int(r.n)} width ratio = R1 width / R0(n 0) width", r.width_ratio, float(s1.width_cm) / float(s0.width_cm))
        L.num(f"d {r.target} n{int(r.n)} coverage", r.coverage, s1.coverage)
        L.num(f"d {r.target} n{int(r.n)} cov_lo", r.cov_lo, s1.cov_lo); L.num(f"d {r.target} n{int(r.n)} cov_hi", r.cov_hi, s1.cov_hi)
    lv = _row(X, test_id="L25", scope="verdict_aux")
    m = re.search(r"c1 = (\d+)\(판정한 칸 \d+\), c2 = (\d+)", str(lv.stat))
    Dr = rd(SRC / "Fig6_d_registered.csv").iloc[0]
    L.eq("d c1 (L25 stat)", int(Dr.c1), int(m.group(1))); L.eq("d c2 (L25 stat)", int(Dr.c2), int(m.group(2)))
    # 캡션 수치(CAPTIONS.md 의 v2/Fig6 절)
    cap = (ROOT / "outputs/figures/paper/CAPTIONS.md").read_text()
    sec = cap[cap.index("## v2/Fig6_label0_uncertainty"):]
    sec = sec[:sec.index("<!-- words")]
    be = _row(B, scope="MEAN", ab="AB2")
    fm = lambda v: (f"{v:.2f}".replace("-", "−"))                         # noqa: E731
    L.eq("caption B:ens CI", f"({fm(be.delta)} [{fm(be.ci_lo)}, {fm(be.ci_hi)}] cm)" in sec, True)
    m4 = I[(I.n == 0) & (I.scope == "all") & (I.target == "MEAN4") & I.method.isin(["const", "phys", "cfm", "nflow", "nflow#placebo", "cbq"])]
    L.eq("caption P4 coverage range", f"P4 coverage {m4.cov10.min():.2f}–{m4.cov10.max():.2f}" in sec, True)
    L.eq("caption C2 interval values", f"C2 interval {c2.coverage:.2f}, {c2.width_cm:.0f} cm" in sec, True)
    L.eq("caption L25 counts", f"in {int(m.group(1))} of 6 cells, all {int(m.group(2))} narrower" in sec, True)
    L.eq("caption has no registered L25 sentence", "narrow the interval" not in sec, True)
    return L.write("Fig6_label0_uncertainty")


def main():
    if "--only" in sys.argv[1:]:                                   # 한 그림만(다른 그림의 _values.txt 는 다시 쓰지 않는다)
        k = sys.argv[sys.argv.index("--only") + 1]
        ok = {"fig6": fig6}[k]()
    else:
        ok = all([fig1(), fig5(), fig6(), fig7(), table1(), part2_fig2_4()])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
