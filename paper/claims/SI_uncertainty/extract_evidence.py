"""SI 불확실성(예측 구간) 근거 수치 추출과 원천 표 사본 작성.

원천 집계 표를 읽기만 하고 고치지 않는다. 학습·적합·재표집은 하지 않는다.
실행: OMP_NUM_THREADS=1 python paper/claims/SI_uncertainty/extract_evidence.py
출력:
  paper/claims/SI_uncertainty/evidence_values.csv  (README 2절 표의 수치 원본)
  paper/claims/SI_uncertainty/tables/*              (원천 집계 표 사본)
  paper/claims/SI_uncertainty/tables/MANIFEST.csv   (orig_path, copy_path, sha256, rows, bytes)
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence_values.csv"
TAB = HERE / "tables"

S11 = "data/processed/s11_conformal_results.csv"
S11_META = "data/processed/s11_conformal_meta.json"
E4 = "data/processed/e4_interval_score.csv"
E4_META = "data/processed/e4_interval_score_meta.json"
E4C = "data/processed/e4_conditional_coverage.csv"
TUQ = "data/processed/m1/transfer_uq_summary.csv"
TUQ_META = "data/processed/m1/transfer_uq_meta.json"
SCUQ = "data/processed/m1/m1_sc_uq.csv"
C2 = "data/processed/h4/c2_coverage.csv"
C2_META = "data/processed/h4/c2_meta.json"
LGUA = "data/processed/lgu/lgu_a_tests.csv"
LGUB = "data/processed/lgu/lgu_b_tests.csv"
AB10 = "data/processed/lgu/lgu_a_ab10_tests.csv"
AB10_META = "data/processed/lgu/lgu_a_ab10_meta.json"
LGUA_META = "data/processed/lgu/lgu_a_meta.json"
LGUB_META = "data/processed/lgu/lgu_b_meta.json"
LGUB_INT = "data/processed/lgu/lgu_b_intervals.csv"
LGW = "data/processed/lgw/lgw_bundle.csv"
LGX = "data/processed/lgx/lgx_tests.csv"
LGX_CONF = "data/processed/lgx/lgx_conformal.csv"
F6B = "data/processed/paper_figs/fig6_b.csv"
F6B_REF = "data/processed/paper_figs/fig6_b_ref.csv"
F6C = "data/processed/paper_figs/fig6_c.csv"
F6C_REG = "data/processed/paper_figs/fig6_c_registered.csv"
F6D = "data/processed/paper_figs/fig6_d.csv"
F6D_REG = "data/processed/paper_figs/fig6_d_registered.csv"
F7D_AB10 = "data/processed/paper_figs/fig7_d_ab10.csv"
F6_META = "data/processed/paper_figs/v2_data_fig6_meta.json"

# 판정 문서 절
V_S11 = "docs/EXPERIMENT_LOG.md 2026-07-27 S6~S11 절(S11 '채택, 헤드라인')"
V_E4 = "docs/EXPERIMENT_LOG.md 'E4.3 구간 점수·조건부 커버리지 (W12)' 절; docs/RESULTS_RECONCILIATION_2026-09-14.md 3.7"
V_H17 = "docs/EXPERIMENT_LOG.md '보조 분석 결과 (09-21)' 절 H17 행; 기준 docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md H17 행"
V_H15 = "docs/EXPERIMENT_LOG.md H15 행(대체: LGU 4.6·11.1 의 LGU-B1)"
V_F10 = "docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md 결과 표 F10 행, 10.4 (k)"
V_LGU = "docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 11.1(J3)"
V_J7 = "docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 결과 판정 기록(J7) 1.1"
V_L25 = "docs/EXPERIMENT_PLAN_LG_2026-09-29.md 7.2(J2) L25; WRAPUP 지도 잠정판 절 'L25 와 LGU 10절의 문장 충돌(14:20)'"

rows: list[dict] = []


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def add(eid, group, src, filt, col, val, verdict="", vsec="", note=""):
    f = num(val)
    if f is None:
        v2 = v4 = val
        raw = val
    elif math.isinf(f) or math.isnan(f):
        v2 = v4 = str(f)
        raw = str(f)
    else:
        v2 = round(f, 2)
        v4 = round(f, 4)  # 비율(커버리지)은 2자리로 0.4456 과 0.4500 을 구별하지 못해 4자리를 함께 둔다
        raw = repr(f)
    rows.append(dict(id=eid, group=group, source=src, row_filter=filt, column=col,
                     value=v2, value_4dp=v4, value_raw=raw, verdict=verdict, verdict_doc=vsec, note=note))


def one(df, mask, what):
    sub = df[mask]
    if len(sub) != 1:
        raise SystemExit(f"{what}: 행 {len(sub)}개(1개여야 한다)")
    return sub.iloc[0]


def rd(p):
    return pd.read_csv(ROOT / p)


# ---------------- U1 알래스카 지역 내 보정 구간(S11, E4.3) ----------------
s11 = rd(S11)
for setting, tag in [("raw_seedmean", "보정 전 분위 구간"), ("cqr_seedmean", "CQR")]:
    r = one(s11, (s11.setting == setting) & (s11.level == 0.9), f"S11 {setting}")
    f = f"setting=='{setting}' & level==0.9"
    for c in ["coverage", "cov_ci_lo", "cov_ci_hi", "width_cm"]:
        add(f"U1-{tag}", "U1", S11, f, c, r[c], "서술(대회 S11 채택)", V_S11,
            "seed 3 평균, 알래스카 13,606셀 6-fold 블록 OOF, x34 입력")

e4 = rd(E4)
for itv in ["raw", "cqr", "const_width"]:
    r = one(e4, e4.interval == itv, f"E4 {itv}")
    for c in ["coverage", "cov_ci_lo", "cov_ci_hi", "width_mean", "interval_score", "is_ci_lo", "is_ci_hi",
              "is_width_part", "is_under_part", "is_over_part"]:
        add(f"U1-E4-{itv}", "U1", E4, f"interval=='{itv}'", c, r[c], "서술(원고 조치: 서술 완화)", V_E4,
            "S11 OOF 단일 세트, 블록 부트스트랩 1,000회, 비짝지음 CI")
r_c = one(e4, e4.interval == "cqr", "cqr")
r_k = one(e4, e4.interval == "const_width", "const")
add("U1-E4-diff", "U1", E4, "interval=='cqr' 행 − interval=='const_width' 행", "interval_score(산술 차)",
    r_c.interval_score - r_k.interval_score, "서술(짝지음 CI 원천 표에 없음)", V_E4,
    "산술 차. 문서의 3.5 [0.6, 6.5] 는 원천 표에서 확인하지 못함")
e4m = json.loads((ROOT / E4_META).read_text())
add("U1-E4-half", "U1", E4_META, "const_width_reference", "half_width_cm",
    e4m["const_width_reference"]["half_width_cm"], "", V_E4,
    "채점 셀의 |y − pred_med| 90 % 분위(사후 참조)")

e4c = rd(E4C)
for b in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
    r = one(e4c, (e4c.interval == "cqr") & (e4c.axis == "alt_quintile") & (e4c.bin == b), f"E4C {b}")
    add(f"U1-E4C-alt-{b}", "U1", E4C, f"interval=='cqr' & axis=='alt_quintile' & bin=='{b}'", "coverage",
        r.coverage, "서술", V_E4)
for b in ["W1", "W5"]:
    r = one(e4c, (e4c.interval == "cqr") & (e4c.axis == "width_quintile") & (e4c.bin == b), f"E4C {b}")
    for c in ["coverage", "interval_score"]:
        add(f"U1-E4C-wid-{b}", "U1", E4C, f"interval=='cqr' & axis=='width_quintile' & bin=='{b}'", c,
            r[c], "서술", V_E4)
r = one(e4c, (e4c.interval == "cqr") & (e4c.axis == "block_coverage_dist"), "E4C block")
for c, lab in [("coverage", "블록 커버리지 중앙값"), ("lo", "10 백분위"), ("hi", "90 백분위"),
               ("interval_score", "커버리지 0.8 미만 블록 비율(열 이름은 interval_score)")]:
    add("U1-E4C-blk", "U1", E4C, "interval=='cqr' & axis=='block_coverage_dist'", c, r[c], "서술", V_E4, lab)

tuq = rd(TUQ)
r = one(tuq, (tuq.region == "Alaska") & (tuq.cond == "in_domain") & (tuq.method == "cqr_ak"), "TUQ ak")
for c in ["coverage", "cov_ci_lo", "cov_ci_hi", "width_mean", "interval_score"]:
    add("U1-H17-ak", "U1", TUQ, "region=='Alaska' & cond=='in_domain' & method=='cqr_ak'", c, r[c],
        "서술(참조)", V_H17, "x25 입력, 블록 25 % 보정")

# ---------------- U2 전이 CQR(H17) ----------------
REG4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
for meth in ["cqr_ak", "cqr_iw", "anchor_ak", "anchor_iw"]:
    for cond in ["noinfo", "covonly"]:
        for reg in REG4:
            r = one(tuq, (tuq.region == reg) & (tuq.cond == cond) & (tuq.method == meth), f"TUQ {meth}")
            below = (r.cov_ci_hi < 0.90)
            for c in ["coverage", "cov_ci_lo", "cov_ci_hi"]:
                add(f"U2-{meth}-{cond}-{reg}", "U2", TUQ,
                    f"region=='{reg}' & cond=='{cond}' & method=='{meth}'", c, r[c],
                    ("H17 확인(CI 상한 < 0.90)" if (meth == "cqr_ak" and below) else
                     ("H17 확인 아님(CI 상한 ≥ 0.90)" if meth == "cqr_ak" else "서술")), V_H17)
for meth in ["cqr_ak", "cqr_iw"]:
    sub = tuq[(tuq.method == meth) & tuq.cond.isin(["noinfo", "covonly"]) & tuq.region.isin(REG4)]
    assert len(sub) == 8 and sub.ci_flag.isna().all()
    lo = sub.loc[sub.coverage.idxmin()]
    hi = sub.loc[sub.coverage.idxmax()]
    add(f"U2-{meth}-min", "U2", TUQ, f"method=='{meth}' & cond∈{{noinfo,covonly}} & region∈주4지역(8행) 최솟값",
        "coverage", lo.coverage, "서술", V_H17, f"{lo.region}/{lo.cond}")
    add(f"U2-{meth}-max", "U2", TUQ, f"method=='{meth}' & cond∈{{noinfo,covonly}} & region∈주4지역(8행) 최댓값",
        "coverage", hi.coverage, "서술", V_H17, f"{hi.region}/{hi.cond}")
sub = tuq[(tuq.method == "cqr_ak") & (tuq.cond == "noinfo") & tuq.region.isin(REG4)]
add("U2-cqr_ak-noinfo-mean4", "U2", TUQ, "method=='cqr_ak' & cond=='noinfo' & region∈주4지역 단순 평균(산술)",
    "coverage", sub.coverage.mean(), "서술", V_F10, "F10 행의 '기존 CQR AB4 0.55'(09-26 문서의 AB4 = 주 4지역 평균)")
sub = tuq[(tuq.method == "cqr_real") & (tuq.cond == "covonly") & tuq.region.isin(REG4)]
add("U2-cqr_real-min", "U2", TUQ, "method=='cqr_real' & cond=='covonly' & region∈주4지역 최솟값", "coverage",
    sub.coverage.min(), "서술(대상 실측 보정 상한)", V_H17)
add("U2-cqr_real-max", "U2", TUQ, "method=='cqr_real' & cond=='covonly' & region∈주4지역 최댓값", "coverage",
    sub.coverage.max(), "서술(대상 실측 보정 상한)", V_H17)

c2 = rd(C2)
for meth in ["ref_cqr_ak", "ref_cqr_iw"]:
    r = one(c2, (c2.test == "ref_h17") & (c2.method == meth) & c2.target.str.startswith("MEAN6"), f"C2 {meth}")
    add(f"U2-c2ref-{meth}", "U2", C2, f"test=='ref_h17' & method=='{meth}' & target 'MEAN6[...6지역]'",
        "coverage", r.coverage, "서술(점 추정, H17 표에서 복사)", V_F10, str(r.target))

# ---------------- U3 계층 conformal(09-26 h37, F10) ----------------
M4 = "MEAN6[Russia_W,Russia_E,Canada,Lena]"
for meth in ["hier2_cdf", "pooled", "hier2_sub", "hier2_blk"]:
    r = one(c2, (c2.test == "label0") & (c2.method == meth) & (c2.target == M4), f"C2 {meth}")
    for c in ["coverage", "coverage_lo", "coverage_hi", "coverage_beq", "width_cm", "interval_score", "n_eval",
              "n_blocks"]:
        add(f"U3-{meth}-mean4", "U3", C2, f"test=='label0' & method=='{meth}' & target=='{M4}'", c, r[c],
            "부분 지지(F10)" if meth == "hier2_cdf" else "서술", V_F10)
for reg in REG4 + ["Alaska"]:
    r = one(c2, (c2.test == "label0") & (c2.method == "hier2_cdf") & (c2.target == reg), f"C2 {reg}")
    for c in ["coverage", "coverage_lo", "coverage_hi", "coverage_beq", "width_cm", "K_groups"]:
        add(f"U3-hier2_cdf-{reg}", "U3", C2, f"test=='label0' & method=='hier2_cdf' & target=='{reg}'", c, r[c],
            "서술", V_F10)
r = one(c2, c2.test == "F10", "C2 F10")
add("U3-F10", "U3", C2, "test=='F10'", "in_band", r.in_band, "부분 지지", V_F10, str(r.ci_flag))
add("U3-F10", "U3", C2, "test=='F10'", "coverage(MAIN6)", r.coverage, "부분 지지", V_F10)

# ---------------- U4 LGU-B(라벨 0 정규화기, 생성 분위 구간) ----------------
lb = rd(LGUB)
for meth in ["nflow", "cfm"]:
    r = one(lb, (lb.test == "LGU-B1") & lb.contrast.str.contains(f"cqr_pool:{meth}@lam1.0", regex=False),
            f"B1 {meth}")
    for c in ["delta", "ci_lo", "ci_hi", "delta_beq", "ci_lo_beq", "ci_hi_beq"]:
        add(f"U4-B1-{meth}", "U4", LGUB, f"test=='LGU-B1' & contrast ∋ 'cqr_pool:{meth}@lam1.0' (pool MEAN4)",
            c, r[c], str(r.verdict), V_LGU, "delta 열 = 90 % 커버리지 수준(대비 아님), 2단 CI")
for tid, key in [("LGU-B2", "nflow − 위약"), ("LGU-B3", "nflow − cbq")]:
    r = one(lb, (lb.test == tid) & lb.contrast.str.contains(key, regex=False), tid)
    for c in ["delta", "ci_lo", "ci_hi", "delta_beq", "ci_lo_beq", "ci_hi_beq", "coverage_pool"]:
        add(f"U4-{tid}", "U4", LGUB, f"test=='{tid}' (n 0, pool MEAN4, metric is10)", c, r[c], str(r.verdict),
            V_LGU)
    if tid == "LGU-B2":
        for c in ["verdict_3region", "n_neg_regions", "statement"]:
            add("U4-LGU-B2", "U4", LGUB, "test=='LGU-B2'", c, r[c], str(r.verdict), V_LGU)
b4 = lb[lb.test == "LGU-B4"]
for _, r in b4.iterrows():
    for c in ["delta", "ci_lo", "ci_hi", "delta_beq", "ci_lo_beq", "ci_hi_beq"]:
        add("U4-LGU-B4", "U4", LGUB, f"test=='LGU-B4' & contrast=='{r.contrast}'", c, r[c], str(r.verdict), V_LGU)
b6 = lb[lb.test == "LGU-B6"]
for _, r in b6.iterrows():
    for c in ["delta", "ci_lo", "ci_hi", "delta_beq", "ci_lo_beq", "ci_hi_beq"]:
        add("U4-LGU-B6", "U4", LGUB, f"test=='LGU-B6' & contrast=='{r.contrast}' & n=={int(r.n)}", c, r[c],
            str(r.verdict), V_LGU)
f6b = rd(F6B)
for meth in ["const", "nflow", "nflow#placebo", "cbq", "cfm", "phys"]:
    r = one(f6b, (f6b.target == "MEAN4") & (f6b.method == meth), f"F6B {meth}")
    for c in ["cov10", "cov10_lo2", "cov10_hi2", "cov10_beq", "wid10", "is10"]:
        add(f"U4-F6B-{meth}", "U4", F6B, f"target=='MEAN4' & method=='{meth}'", c, r[c], "서술(수준 값)", V_LGU,
            "lgu_b_intervals.csv 에서 옮긴 그림 표, n 0, 범위 all")
for reg in REG4:
    r = one(f6b, (f6b.target == reg) & (f6b.method == "const"), f"F6B const {reg}")
    add(f"U4-F6B-const-{reg}", "U4", F6B, f"target=='{reg}' & method=='const'", "cov10", r.cov10, "서술(수준 값)",
        V_LGU)

# ---------------- U5 LGU-A(라벨 있는 계층 예측 분포), AB10 ----------------
la = rd(LGUA)
for n in ["10", "40"]:
    r = one(la, (la.test == "LGU-A1") & (la.n.astype(str) == n), f"A1 {n}")
    for c in ["delta", "ci_lo", "ci_hi", "delta_beq", "ci_lo_beq", "ci_hi_beq", "verdict_1stage", "ci_lo_1stage",
              "ci_hi_1stage", "eq_state", "half_width", "ref_score", "coverage_pool"]:
        add(f"U5-A1-n{n}", "U5", LGUA, f"test=='LGU-A1' & n=='{n}' (contrast 'iii − B4', metric is10)", c, r[c],
            str(r.verdict), V_LGU)
r = one(la, (la.test == "LGU-A1") & (la.n.astype(str) == "10,40"), "A1 overall")
add("U5-A1-all", "U5", LGUA, "test=='LGU-A1' & n=='10,40'", "verdict", r.verdict, str(r.verdict), V_LGU)
for pool in ["주 4지역", "러시아 W 제외 3지역"]:
    for n in ["3", "10"]:
        r = one(la, (la.test == "LGU-A3") & (la.pool == pool) & (la.n.astype(str) == n), f"A3 {pool} {n}")
        for c in ["delta", "ci_lo", "ci_hi", "delta_beq", "ci_lo_beq", "ci_hi_beq"]:
            add(f"U5-A3-{pool}-n{n}", "U5", LGUA, f"test=='LGU-A3' & pool=='{pool}' & n=='{n}' (iii − P1 RMSE)",
                c, r[c], str(r.verdict), V_LGU)
r = one(la, (la.test == "LGU-A7") & (la.contrast == "iv − iii") & (la.n.astype(str) == "10") & (la.metric == "is10"),
        "A7")
for c in ["delta", "ci_lo", "ci_hi"]:
    add("U5-A7-iv-iii-n10", "U5", LGUA, "test=='LGU-A7' & contrast=='iv − iii' & n=='10' & metric=='is10'", c, r[c],
        str(r.verdict), V_LGU)
a2 = la[la.test == "LGU-A2"]
add("U5-A2", "U5", LGUA, "test=='LGU-A2'(5행) delta 최댓값", "delta(inf10)", a2.delta.max(),
    "; ".join(sorted(set(a2.verdict.astype(str)))), V_LGU)

ab = rd(AB10)
r = one(ab, ab.test_id == "LGU-A1", "AB10")
for c in ["delta", "p_cell", "p_beq", "p_eq", "verdict4"]:
    add("U5-AB10-lgu", "U5", AB10, "test_id=='LGU-A1' (n 10, is10)", c, r[c], str(r.verdict4), V_J7)
lgw = rd(LGW)
r = one(lgw, lgw.ab == "AB10", "LGW AB10")
for c in ["delta", "holm_p", "holm_p_eq", "holm_m", "verdict4", "abstract_rule"]:
    add("U5-AB10-bundle", "U5", LGW, "ab=='AB10'", c, r[c], str(r.verdict4), V_J7)
f6c = rd(F6C)
for tid, n in [("LGU-B2", 0), ("LGU-A1", 10), ("LGU-A1", 40)]:
    r = one(f6c, (f6c.test == tid) & (f6c.n == n), f"F6C {tid} {n}")
    for c in ["delta_pct", "ci_lo_pct", "ci_hi_pct"]:
        add(f"U5-F6C-{tid}-n{n}", "U5", F6C, f"test=='{tid}' & n=={n}", c, r[c], str(r.verdict4), V_LGU,
            "Δ/기준 점수(산술, v2_data)")

# ---------------- U6 라벨 있는 R1 구간(LGX L25) ----------------
lgx = rd(LGX)
l25 = lgx[lgx.test_id == "L25"]
for _, r in l25[l25.scope == "region"].iterrows():
    for c in ["coverage", "width_cm", "width_n0_r0", "in_band", "narrower"]:
        add(f"U6-L25-{r.target}-n{int(r.n)}", "U6", LGX, f"test_id=='L25' & scope=='region' & target=='{r.target}' & n=={int(r.n)}",
            c, r[c], "라벨이 구간을 좁힌다(원고에 쓰지 않음)", V_L25)
r = one(l25, l25.scope == "verdict_aux", "L25 verdict")
add("U6-L25-verdict", "U6", LGX, "test_id=='L25' & scope=='verdict_aux'", "verdict", r.verdict, r.verdict, V_L25)

# ---------------- H15 참고(대체됨) ----------------
sc = rd(SCUQ)
for res in ["cfm", "ddpm", "nflow"]:
    r = one(sc, (sc.cond == "labels") & (sc.target == "Alaska") & (sc.anchor == "none") & (sc.resid == res)
            & (sc.seed.astype(str) == "mean"), f"H15 {res}")
    for c in ["coverage", "cov_ci_lo", "cov_ci_hi", "width_mean", "interval_score"]:
        add(f"H15-direct-{res}", "H15", SCUQ,
            f"cond=='labels' & target=='Alaska' & anchor=='none' & resid=='{res}' & seed=='mean'", c, r[c],
            "기각(H15, LGU-B1 로 대체)", V_H15, "문서 값(0.70·0.62·0.67)과 일치하는 단일 행을 찾지 못함")

ev = pd.DataFrame(rows)
ev.to_csv(OUT, index=False)
print(f"evidence rows: {len(ev)} -> {OUT.relative_to(ROOT)}")

# ---------------- 원천 표 사본과 MANIFEST ----------------
COPY = [S11, S11_META, E4, E4_META, E4C, TUQ, TUQ_META, SCUQ, C2, C2_META, LGUA, LGUA_META, LGUB, LGUB_META,
        LGUB_INT, AB10, AB10_META, LGW, LGX, LGX_CONF, F6B, F6B_REF, F6C, F6C_REG, F6D, F6D_REG, F7D_AB10, F6_META]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


TAB.mkdir(parents=True, exist_ok=True)
man = []
names = set()
for src in COPY:
    sp = ROOT / src
    if sp.stat().st_size > 5 * 1024 * 1024:
        raise SystemExit(f"5 MB 초과: {src}")
    parts = Path(src).parts
    # 이름 충돌을 피하려고 상위 폴더 이름을 접두어로 붙인다(data/processed 바로 아래 파일은 그대로)
    name = sp.name if parts[-2] == "processed" else f"{parts[-2]}__{sp.name}"
    assert name not in names, name
    names.add(name)
    dp = TAB / name
    shutil.copy2(sp, dp)
    h_src, h_dst = sha(sp), sha(dp)
    assert h_src == h_dst, src
    nrows = "NA"
    if sp.suffix == ".csv":
        nrows = len(pd.read_csv(sp))
    man.append(dict(orig_path=src, copy_path=str(dp.relative_to(ROOT)), sha256=h_dst, rows=nrows,
                    bytes=dp.stat().st_size))
pd.DataFrame(man).to_csv(TAB / "MANIFEST.csv", index=False)
print(f"copied {len(man)} files -> {TAB.relative_to(ROOT)}")
