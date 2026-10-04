"""C4(충분 라벨 지역 안의 이득) 근거 수치 추출.

tables/ 의 사본만 읽고 evidence_values.csv 를 쓴다. 새 적합·재표집·판정은 하지 않는다.
파생값(줄일 수 있는 오차 비율, RMSE 개선률, 순위상관)은 원천 열의 산술이고 'derived' 로 표시한다.
실행: OMP_NUM_THREADS=1 python paper/claims/C4_sufficient_labels/extract_evidence.py  (수 초)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
TB = HERE / "tables"
ORIG = {
    "wf2b_tests.csv": "results/rescale_wf2/data/processed/wf/wf2b_tests.csv",
    "wf2b_curve.csv": "results/rescale_wf2/data/processed/wf/wf2b_curve.csv",
    "wf_tests.csv": "results/rescale_wf/data/processed/wf/wf_tests.csv",
    "wf_curve.csv": "results/rescale_wf/data/processed/wf/wf_curve.csv",
    "lgx_floor.csv": "data/processed/lgx/lgx_floor.csv",
    "s4_residual_results.csv": "data/processed/s4_residual_results.csv",
    "cv_scheme_comparison.csv": "data/processed/m1/cv_scheme_comparison.csv",
    "wf0_misspec.csv": "data/processed/wf/wf0_misspec.csv",
}
T6 = pd.read_csv(TB / "wf2b_tests.csv")
C6 = pd.read_csv(TB / "wf2b_curve.csv")
T1 = pd.read_csv(TB / "wf_tests.csv")
C1 = pd.read_csv(TB / "wf_curve.csv")
FL = pd.read_csv(TB / "lgx_floor.csv")
S4 = pd.read_csv(TB / "s4_residual_results.csv")
CV = pd.read_csv(TB / "cv_scheme_comparison.csv")
MS = pd.read_csv(TB / "wf0_misspec.csv")

rows: list[dict] = []


def one(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    r = df[m]
    if len(r) != 1:
        raise SystemExit(f"행 수 {len(r)} != 1: {kw}")
    return r.iloc[0]


def add(eid, src, filt, col, val, verdict="", kind="source"):
    rows.append(dict(id=eid, kind=kind, source=ORIG.get(src, src), filter=filt, column=col,
                     value=(round(float(val), 2) if isinstance(val, (int, float, np.floating, np.integer)) and pd.notna(val) else val),
                     verdict=verdict))


def test_row(eid, T, src, test_id, item, target, contrast):
    r = one(T, test_id=test_id, item=item, target=target, contrast=contrast) if item else one(T, test_id=test_id, target=target, contrast=contrast)
    f = f"test_id={test_id}" + (f", item={item}" if item else "") + f", target={target}, contrast={contrast}"
    for col in ("delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "rmse_A", "rmse_B", "n_splits"):
        add(eid, src, f, col, r[col], r["verdict4"])
    for col in ("verdict4_common", "ci_dependence", "small_note"):
        if col in r.index and pd.notna(r[col]):
            add(eid, src, f, col, r[col], r["verdict4"])
    return r


# E1 WF6-a 알래스카 주 대비(R1, R2, Re, D0 − P1)
for n in ("200", "500", "1000", "전량"):
    for m in ("R1(λ cv)", "R2(λ cv)", "Re(λ cv)", "D0(catboost)"):
        test_row(f"E1.{m.split('(')[0]}.n{n}", T6, "wf2b_tests.csv", "WF6-a", "WF6-a", "Alaska|r", f"{m}-P1|n{n}")
r = one(T6, test_id="WF6-a", item="WF6-a", target="Alaska|r", contrast="R1(λ cv)-P1|n1000")
add("E1.R1.n1000", "wf2b_tests.csv", "test_id=WF6-a, item=WF6-a, target=Alaska|r, contrast=R1(λ cv)-P1|n1000", "delta_pct_p0", r.delta_pct_p0, r.verdict4)
add("E1.R1.n1000", "wf2b_tests.csv", "같은 행", "holm_p", r.holm_p, r.verdict4)
add("E1.R1.n1000", "wf2b_tests.csv", "같은 행", "design_note", r.design_note, r.verdict4)
# E2 레나·캐나다 주 대비
for tgt, ns in (("Lena|r", ("200", "500", "1000", "전량")), ("Canada|r", ("200", "전량"))):
    for n in ns:
        for m in ("R1(λ cv)", "R2(λ cv)", "Re(λ cv)", "D0(catboost)"):
            test_row(f"E2.{tgt.split('|')[0]}.{m.split('(')[0]}.n{n}", T6, "wf2b_tests.csv", "WF6-a", "WF6-a", tgt, f"{m}-P1|n{n}")
# E3 대상별 판정 행(contrast 결측)
for tgt in ("Alaska|r", "Lena|r", "Canada|r"):
    v = T6[(T6.test_id == "WF6-a") & (T6["item"] == "WF6-a") & (T6.target == tgt) & T6.contrast.isna()].verdict.iloc[0]
    add(f"E3.{tgt.split('|')[0]}", "wf2b_tests.csv", f"test_id=WF6-a, item=WF6-a, target={tgt}, contrast 결측", "verdict", v)
v = T6[(T6.test_id == "WF6-b") & T6.contrast.isna()].verdict.iloc[0]
add("E3.WF6-b", "wf2b_tests.csv", "test_id=WF6-b, contrast 결측", "verdict", v)
for n in ("500", "1000", "전량"):
    tgt = "MEAN[Alaska|r,Lena|r]" if n != "전량" else "MEAN[Alaska|r,Lena|r,Canada|r]"
    test_row(f"E3.WF6-b.n{n}", T6, "wf2b_tests.csv", "WF6-b", "WF6-b", tgt, f"R2(λ cv)-P1|n{n}")
# E4 Pbest 보조, 고정 λ 0.25 보조
test_row("E4.Pbest.R1.n1000", T6, "wf2b_tests.csv", "WF6-a", "WF6-a(Pbest)", "Alaska|r", "R1(λ cv)-Pbest|n1000")
for tgt, ns in (("Lena|r", ("200", "500", "1000", "전량")), ("Canada|r", ("200", "전량"))):
    for n in ns:
        test_row(f"E4.lam025.{tgt.split('|')[0]}.n{n}", T6, "wf2b_tests.csv", "WF6-a", "WF6-a(λ 0.25)", tgt, f"R1(λ 0.25)-P1|n{n}")
# E5 WF6 곡선: P0 대비(곡선 열, 판정 대상 아님), 분할 승률, 라벨 수
for tgt, m, n, lam in (("Alaska", "R1", 1000, -1.0), ("Alaska", "R1", -1, -1.0), ("Alaska", "P1", 1000, 0.0),
                       ("Canada", "P1", 200, 0.0), ("Canada", "R1", 200, -1.0), ("Canada", "Re", 200, -1.0),
                       ("Canada", "D0", 200, 1.0), ("Canada", "R1", -1, -1.0), ("Canada", "Re", -1, -1.0),
                       ("Lena", "P1", -1, 0.0), ("Lena", "R1", 1000, -1.0)):
    r = one(C6, exp="wf6", target=tgt, method=m, n=n, lam=lam)
    f = f"exp=wf6, target={tgt}, method={m}, n={n}, lam={lam}"
    for col in ("rmse", "rmse_p0", "d_p0", "d_p0_lo", "d_p0_hi", "d_p0_beq", "d_p0_beq_lo", "d_p0_beq_hi", "split_win_p1", "n_lab", "n_splits_valid"):
        add(f"E5.{tgt}.{m}.n{n}", "wf2b_curve.csv", f, col, r[col], r["verdict4_p0"])
# E6 WF1-a(알래스카 주, 하위 지역 AL-2 서술)
for tgt, ns in (("Alaska|r", ("1000", "2000", "5000", "전량")), ("AL-2|r", ("1000", "전량"))):
    for n in ns:
        for m in ("R1(λ cv)", "R2(λ cv)"):
            test_row(f"E6.{tgt.split('|')[0]}.{m.split('(')[0]}.n{n}", T1, "wf_tests.csv", "WF1-a", None, tgt, f"{m}-Pbest|n{n}")
for tgt in ("Alaska|r", "AL-2|r", "Lena|r", "Canada|r", "AL-1|r", "AL-6|r"):
    v = T1[(T1.test_id == "WF1-a") & (T1.target == tgt) & T1.contrast.isna()].verdict.iloc[0]
    add(f"E6.verdict.{tgt.split('|')[0]}", "wf_tests.csv", f"test_id=WF1-a, target={tgt}, contrast 결측", "verdict", v)
for m in ("P1", "R1", "R2"):
    for n in (1000, -1):
        lam = 0.0 if m == "P1" else -1.0
        lr = "none" if m == "P1" else "catboost_lo"
        r = one(C1, exp="wf1", target="AL-2", mode="r", method=m, learner=lr, n=n, lam=lam)
        f = f"exp=wf1, target=AL-2, mode=r, method={m}, learner={lr}, n={n}, lam={lam}"
        for col in ("rmse", "d_p1", "d_p1_lo", "d_p1_hi", "d_p1_beq", "d_p1_beq_lo", "d_p1_beq_hi"):
            add(f"E6.AL-2.curve.{m}.n{n}", "wf_curve.csv", f, col, r[col], r["verdict4_p1"])
x = C1[(C1.exp == "wf1") & (C1.target == "Alaska") & (C1["mode"] == "r")]
for m, lr, lam in (("P1", "none", 0.0), ("R1", "catboost_lo", -1.0), ("D0", "catboost", 1.0)):
    for _, r in x[(x.method == m) & (x.learner == lr) & (x.lam == lam)].iterrows():
        add(f"E6.Alaska.curve.{m}.n{int(r.n)}", "wf_curve.csv", f"exp=wf1, target=Alaska, mode=r, method={m}, learner={lr}, lam={lam}, n={int(r.n)}", "rmse", r.rmse)
# E7 WF1-b(SAR), WF1-c
for n in ("500", "1000", "2000", "5000", "전량"):
    test_row(f"E7.R1x34.n{n}", T1, "wf_tests.csv", "WF1-b", None, "Alaska|r", f"R1@x34-R1|n{n}|λ cv")
    test_row(f"E7.D1x34.n{n}", T1, "wf_tests.csv", "WF1-b", None, "Alaska|r", f"D1@x34-D1|n{n}")
v = T1[(T1.test_id == "WF1-b") & T1.contrast.isna()].verdict.iloc[0]
add("E7.verdict", "wf_tests.csv", "test_id=WF1-b, contrast 결측", "verdict", v)
r = T1[(T1.test_id == "WF1-c") & (T1.target == "Alaska|r") & T1.contrast.isna()].iloc[0]
add("E7.WF1-c", "wf_tests.csv", "test_id=WF1-c, target=Alaska|r, contrast 결측", "verdict", r.verdict)
add("E7.WF1-c", "wf_tests.csv", "test_id=WF1-c, target=Alaska|r, contrast 결측", "stat", r.stat)
# E8 오차 하한, 대회 수치, 검증 방식 표
fa = one(FL, region="Alaska", scope="eval")
add("E8.floor", "lgx_floor.csv", "region=Alaska, scope=eval", "floor_rmse_cm", fa.floor_rmse_cm)
add("E8.floor", "lgx_floor.csv", "region=Alaska, scope=eval", "n_cells_used", fa.n_cells_used)
for reg in ("Lena", "Canada"):
    add(f"E8.floor.{reg}", "lgx_floor.csv", f"region={reg}, scope=eval", "floor_rmse_cm", one(FL, region=reg, scope="eval").floor_rmse_cm)
st = one(S4, part="D_indomain", cv="spatial_block_AK", model="ridge", featset="shared25", lam=0.0, seed=0.0).rmse_cm
rs = one(S4, part="D_indomain", cv="spatial_block_AK", model="ridge", featset="shared25", lam=0.75, seed=0.0).rmse_cm
add("E8.contest.stefan", "s4_residual_results.csv", "part=D_indomain, cv=spatial_block_AK, model=ridge, featset=shared25, lam=0.0, seed=0", "rmse_cm", st)
add("E8.contest.resid", "s4_residual_results.csv", "part=D_indomain, cv=spatial_block_AK, model=ridge, featset=shared25, lam=0.75, seed=0", "rmse_cm", rs)
d = S4[S4.part == "D_indomain"]
add("E8.contest.min", "s4_residual_results.csv", "part=D_indomain(70행) 의 최솟값", "rmse_cm", d.rmse_cm.min(), kind="derived")
add("E8.contest.nrows_lam_pos", "s4_residual_results.csv", "part=D_indomain, lam>0 행 수", "count", int((d.lam > 0).sum()), kind="derived")
cvb = CV[CV.scheme == "block_0.5_canonical"].groupby("model")
for mdl, g in cvb:
    add(f"E8.cv.block_canonical.{mdl}", "cv_scheme_comparison.csv", f"scheme=block_0.5_canonical, model={mdl}, seed 0–2 평균", "rmse_cm", g.rmse_cm.mean(), kind="derived")
add("E8.cv.block_canonical.nnd", "cv_scheme_comparison.csv", "scheme=block_0.5_canonical", "nnd_median_km", CV[CV.scheme == "block_0.5_canonical"].nnd_median_km.iloc[0])
# D 파생: 줄일 수 있는 오차 제곱과 개선률(산술)
fl = fa.floor_rmse_cm
r = one(T6, test_id="WF6-a", item="WF6-a", target="Alaska|r", contrast="R1(λ cv)-P1|n1000")
p1, r1 = r.rmse_B, r.rmse_A
add("D1.wf6.reducible_P1", "derived", "rmse_B² − floor²(WF6 R1−P1 n1000, Alaska eval floor)", "cm²", p1 ** 2 - fl ** 2, kind="derived")
add("D1.wf6.reducible_R1", "derived", "rmse_A² − floor²", "cm²", r1 ** 2 - fl ** 2, kind="derived")
add("D1.wf6.reducible_cut_pct", "derived", "(P1 몫 − R1 몫)/P1 몫 × 100", "%", ((p1 ** 2 - fl ** 2) - (r1 ** 2 - fl ** 2)) / (p1 ** 2 - fl ** 2) * 100, kind="derived")
add("D1.wf6.rmse_rel_P1_pct", "derived", "(rmse_B − rmse_A)/rmse_B × 100", "%", (p1 - r1) / p1 * 100, kind="derived")
add("D2.contest.reducible_stefan", "derived", "Stefan² − floor²", "cm²", st ** 2 - fl ** 2, kind="derived")
add("D2.contest.reducible_resid", "derived", "residual² − floor²", "cm²", rs ** 2 - fl ** 2, kind="derived")
add("D2.contest.reducible_cut_pct", "derived", "(Stefan 몫 − residual 몫)/Stefan 몫 × 100", "%", ((st ** 2 - fl ** 2) - (rs ** 2 - fl ** 2)) / (st ** 2 - fl ** 2) * 100, kind="derived")
add("D2.contest.rmse_rel_pct", "derived", "(Stefan − residual)/Stefan × 100", "%", (st - rs) / st * 100, kind="derived")
# D3 C2 좁힘 점검(재보정을 넘어선 ML 몫과 재보정 이득의 순위상관)
rho, p = spearmanr(MS.recal_gain, MS.ml_vs_p1)
add("D3.c2.rho_recal_vs_mlvsp1", "wf0_misspec.csv", "30행, Spearman(recal_gain, ml_vs_p1)", "rho", rho, kind="derived")
add("D3.c2.rho_recal_vs_mlvsp1", "wf0_misspec.csv", "같음", "p", p, kind="derived")
rho2, p2 = spearmanr(MS.recal_gain, MS.ml_best_delta)
add("D3.c2.rho_recal_vs_mlbest", "wf0_misspec.csv", "30행, Spearman(recal_gain, ml_best_delta)", "rho", rho2, kind="derived")

# O7 전이 조건 알래스카(LG 곡선, 7.1 MB 라 사본 없음. 저장소 원천을 직접 읽는다. 곡선 열, 판정 대상 아님)
LGC = HERE.parents[2] / "results/rescale_lg/data/processed/lg/lg_curve.csv"
if LGC.exists():
    G = pd.read_csv(LGC, usecols=["target", "mode", "axis", "method", "learner", "alpha", "n", "lam", "rmse", "rmse_p0", "rmse_p1",
                                  "d_p1", "d_p1_lo", "d_p1_hi", "d_p1_beq", "d_p1_beq_lo", "d_p1_beq_hi", "sig_p1"],
                    dtype={"alpha": str})
    for axis, alpha in (("method", "1"), ("alpha", "10"), ("alpha", "100"), ("alpha", "cont")):
        r = one(G, target="Alaska", mode="x", axis=axis, method="R1", learner="catboost_lo", alpha=alpha, n=-1, lam=0.25)
        f = f"target=Alaska, mode=x, axis={axis}, method=R1, learner=catboost_lo, alpha={alpha}, n=-1, lam=0.25"
        for col in ("rmse", "rmse_p0", "rmse_p1", "d_p1", "d_p1_lo", "d_p1_hi", "d_p1_beq", "d_p1_beq_lo", "d_p1_beq_hi"):
            rows.append(dict(id=f"O7.transfer.alpha{alpha}", kind="source", source="results/rescale_lg/data/processed/lg/lg_curve.csv",
                             filter=f, column=col, value=round(float(r[col]), 2), verdict=f"sig_p1={r['sig_p1']}(곡선 열, 4분 판정 아님)"))
    r = one(G, target="Alaska", mode="x", axis="method", method="P1", learner="none", alpha="1", n=-1, lam=0.0)
    rows.append(dict(id="O7.transfer.P1", kind="source", source="results/rescale_lg/data/processed/lg/lg_curve.csv",
                     filter="target=Alaska, mode=x, axis=method, method=P1, learner=none, alpha=1, n=-1, lam=0.0",
                     column="rmse", value=round(float(r.rmse), 2), verdict=""))

out = pd.DataFrame(rows)
out.to_csv(HERE / "evidence_values.csv", index=False)
print(f"{len(out)} 행 → {HERE / 'evidence_values.csv'}")
