"""C7(평가 설계) 근거 수치 추출. 원천 집계 표를 읽기만 하고 고치지 않는다.

실행: OMP_NUM_THREADS=1 python paper/claims/C7_evaluation_design/extract_evidence.py
출력: paper/claims/C7_evaluation_design/evidence_values.csv (README 2절 표의 수치 원본)
가벼운 pandas 읽기만 한다(학습·적합 없음).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "evidence_values.csv"

LGV_T = "data/processed/lgx/ladder/lgv_tests.csv"
LGV_M = "data/processed/lgx/ladder/lgv_metrics.csv"
CVS = "data/processed/m1/cv_scheme_comparison.csv"
CVS_META = "data/processed/m1/cv_scheme_comparison_meta.json"
M1T = "data/processed/m1/m1_sc_tests.csv"
L2F = "data/processed/m1/l2_fold_sensitivity_summary.csv"
LGFN = "data/processed/lgf/lgfn_tests.csv"

rows: list[dict] = []


def add(eid, src, filt, col, val, verdict="", vsec="", note=""):
    rows.append(dict(id=eid, source=src, row_filter=filt, column=col,
                     value=(round(float(val), 2) if isinstance(val, (int, float)) else val),
                     value_raw=val, verdict=verdict, verdict_section=vsec, note=note))


# ---- L19, L20, L21 (lgv_tests.csv) -------------------------------------------------------------
t = pd.read_csv(ROOT / LGV_T)


def one(df, **kw):
    s = df
    for k, v in kw.items():
        s = s[s[k] == v]
    assert len(s) == 1, (kw, len(s))
    return s.iloc[0]


for scope in ["MEAN5", "MEAN4", "POOL"]:
    r = one(t, test_id="L19", contrast="D0[catboost_lo]:V-G-V-R", scope=scope)
    f = f"test_id=L19, contrast=D0[catboost_lo]:V-G-V-R, scope={scope}"
    add(f"L19_degr_{scope}", LGV_T, f, "delta", r.delta, r.verdict4, "LG 7.2 L19",
        f"CI [{r.ci_lo:.2f}, {r.ci_hi:.2f}]; 블록 등가중 {r.delta_blockeq:.2f} [{r.ci_lo_beq:.2f}, {r.ci_hi_beq:.2f}]; "
        f"rmse_A(V-G) {r.rmse_A:.2f}, rmse_B(V-R) {r.rmse_B:.2f}; n_cells {int(r.n_cells)}, n_blocks {int(r.n_blocks)}")
for reg in ["Alaska", "Lena", "Canada", "Russia_W", "Russia_E"]:
    r = one(t, test_id="L19", contrast="D0[catboost_lo]:V-G-V-R", scope="region", target=reg)
    add(f"L19_degr_{reg}", LGV_T, f"test_id=L19, contrast=D0[catboost_lo]:V-G-V-R, scope=region, target={reg}", "delta",
        r.delta, r.verdict4, "LG 7.2 L19(지역 행)",
        f"CI [{r.ci_lo:.2f}, {r.ci_hi:.2f}]; 블록 등가중 {r.delta_blockeq:.2f} [{r.ci_lo_beq:.2f}, {r.ci_hi_beq:.2f}]; "
        f"n_cells {int(r.n_cells)}, n_blocks {int(r.n_blocks)}")
for c in ["D0[catboost_lo]-PS@V-R", "D0[catboost_lo]-PS@V-G"]:
    r = one(t, test_id="L19", contrast=c, scope="MEAN5")
    add(f"L19_{c.split('@')[1]}", LGV_T, f"test_id=L19, contrast={c}, scope=MEAN5", "delta", r.delta, r.verdict4,
        "LG 7.2 L19", f"CI [{r.ci_lo:.2f}, {r.ci_hi:.2f}]; 블록 등가중 {r.delta_blockeq:.2f} [{r.ci_lo_beq:.2f}, {r.ci_hi_beq:.2f}]")
for lr in ["catboost", "rf"]:
    r = one(t, test_id="L19", contrast=f"D0[{lr}]:V-G-V-R", scope="MEAN5")
    add(f"L19_degr_{lr}", LGV_T, f"test_id=L19, contrast=D0[{lr}]:V-G-V-R, scope=MEAN5", "delta", r.delta,
        r.verdict4, "LG 7.2 L19(보조 학습기)",
        f"CI [{r.ci_lo:.2f}, {r.ci_hi:.2f}]; 블록 등가중 {r.delta_blockeq:.2f} [{r.ci_lo_beq:.2f}, {r.ci_hi_beq:.2f}]")
v = t[(t.test_id == "L19") & (t.scope == "verdict")].iloc[0]
add("L19_verdict", LGV_T, "test_id=L19, scope=verdict", "verdict", v.verdict, v.verdict, "LG 7.2 L19", "")
for lr in ["catboost_lo", "catboost", "rf"]:
    vv = t[(t.test_id == "L20") & t.scope.str.startswith("verdict") & t.stat.astype(str).str.startswith(lr + " |")]
    assert len(vv) == 1
    add(f"L20_verdict_{lr}", LGV_T, f"test_id=L20, scope=verdict*, stat 가 '{lr} |' 로 시작", "verdict",
        vv.iloc[0].verdict, vv.iloc[0].verdict, "LG 7.2 L20", vv.iloc[0].stat)
for sch in ["V-R", "V-P", "V-S", "V-B", "V-C0", "V-C100", "V-C500", "V-G"]:
    r = one(t, test_id="L20", item=f"D0[catboost_lo] RMSE@{sch}", scope="MEAN5")
    add(f"L20_rmse_{sch}", LGV_T, f"test_id=L20, item='D0[catboost_lo] RMSE@{sch}', scope=MEAN5", "rmse_A",
        r.rmse_A, "", "LG 7.2 L20", f"role={r.role}")
r = one(t, test_id="L20", contrast="D0[catboost_lo]:V-C500-V-R", scope="MEAN5")
add("L20_C500_minus_R", LGV_T, "test_id=L20, contrast=D0[catboost_lo]:V-C500-V-R, scope=MEAN5", "delta", r.delta,
    r.verdict4, "LG 7.2 L20",
    f"CI [{r.ci_lo:.2f}, {r.ci_hi:.2f}]; 블록 등가중 {r.delta_blockeq:.2f} [{r.ci_lo_beq:.2f}, {r.ci_hi_beq:.2f}]; holm_p {r.holm_p}")
for lam in ["0.25", "1.0"]:
    r = one(t, test_id="L21", contrast=f"[RS({lam})-D0][catboost_lo]:V-G-V-R", scope="MEAN5")
    add(f"L21_lam{lam}", LGV_T, f"test_id=L21, contrast=[RS({lam})-D0][catboost_lo]:V-G-V-R, scope=MEAN5", "delta",
        r.delta, r.verdict4, "LG 7.2 L21",
        f"CI [{r.ci_lo:.2f}, {r.ci_hi:.2f}]; 블록 등가중 {r.delta_blockeq:.2f} [{r.ci_lo_beq:.2f}, {r.ci_hi_beq:.2f}]")

# ---- 사다리 단별 Stefan 최소제곱(PS) RMSE (lgv_metrics.csv, 서술) ------------------------------------
m = pd.read_csv(ROOT / LGV_M)
ps = m[(m.method == "PS") & (m.scope == "MEAN5") & m.scheme.isin(["V-R", "V-P", "V-S", "V-B", "V-C0", "V-C100", "V-C500", "V-G"])]
assert len(ps) == 8
for _, r in ps.iterrows():
    add(f"PS_rmse_{r.scheme}", LGV_M, f"method=PS, learner=none, scope=MEAN5, scheme={r.scheme}", "rmse", r.rmse, "",
        "서술(판정 없음)", f"rmse_beq {r.rmse_beq:.2f}")

# ---- 알래스카 검증 방식 비교 (cv_scheme_comparison.csv, seed 0-2 평균, 판정 없음) ----------------------
cv = pd.read_csv(ROOT / CVS)
g = cv.groupby(["scheme", "model"]).agg(rmse=("rmse_cm", "mean"), nnd=("nnd_median_km", "mean"), W=("W_km", "mean"),
                                        n_seed=("seed", "nunique")).reset_index()
for sch in ["random_cell", "site_0.05", "block_0.5", "block_0.5_canonical", "knndm"]:
    for mdl in ["catboost_lo", "stefan", "stefan_ridge_l075", "ridge"]:
        r = one(g, scheme=sch, model=mdl)
        add(f"CV_{sch}_{mdl}", CVS, f"scheme={sch}, model={mdl}, seed 0-2 평균", "rmse_cm", r.rmse, "",
            "서술(EXPERIMENT_LOG 218행, CI 없음)", f"n_seed {r.n_seed}")
    r = one(g, scheme=sch, model="stefan")
    add(f"CV_{sch}_nnd", CVS, f"scheme={sch}, seed 0-2 평균(모델 무관)", "nnd_median_km", r.nnd, "", "서술", "")
    add(f"CV_{sch}_W", CVS, f"scheme={sch}, seed 0-2 평균(모델 무관)", "W_km", r.W, "", "서술",
        "예측 영역 거리 분포와의 Wasserstein 거리")
kn = cv[(cv.scheme == "knndm") & (cv.model == "stefan")]
add("CV_knndm_nnd_by_seed", CVS, "scheme=knndm, model=stefan, seed 0/1/2", "nnd_median_km",
    "/".join(f"{x:.2f}" for x in kn.nnd_median_km), "", "서술", "seed 별 값")
meta = json.loads((ROOT / CVS_META).read_text())
add("CV_gij_permafrost_median", CVS_META, "gij_summary.permafrost_median_km", "permafrost_median_km",
    meta["gij_summary"]["permafrost_median_km"], "", "서술", "예측 격자(영구동토) 표본에서 가장 가까운 라벨까지 거리 중앙값")
add("CV_n_cells", CVS_META, "data.n_cells / n_blocks_0p5", "n_cells", meta["data"]["n_cells"], "", "",
    f"0.5° 블록 {meta['data']['n_blocks_0p5']}개, 0.05° 사이트 {meta['data']['n_sites_0p05']}개")

# ---- 알래스카 0.5° 블록 짝지음 대비(M1 X-ak, 두 가중) ----------------------------------------------------
mt = pd.read_csv(ROOT / M1T, low_memory=False)
for lab in ["알래스카: catboost_lo 직접 − Stefan", "알래스카: Stefan+ridge(λ=.75) − Stefan", "알래스카: ridge 직접 − Stefan"]:
    r = one(mt, H="X-ak", label=lab, target="Alaska")
    both = (r.ci_hi < 0 and r.ci_hi_blockeq < 0) or (r.ci_lo > 0 and r.ci_lo_blockeq > 0)
    add(f"Xak_{lab.split(': ')[1]}", M1T, f"H=X-ak, label='{lab}', target=Alaska", "delta_rmse", r.delta_rmse,
        "두 가중 일치" if both else "가중 규약 의존", "COVERAGE_MATRIX 8절, NOVELTY 5절 N8",
        f"CI [{r.ci_lo:.2f}, {r.ci_hi:.2f}]; 블록 등가중 {r.delta_blockeq:.2f} [{r.ci_lo_blockeq:.2f}, {r.ci_hi_blockeq:.2f}]; seed {int(r.n_seed)}")

# ---- 알래스카 fold 배정 민감도 (l2_fold_sensitivity_summary.csv) -------------------------------------
fs = pd.read_csv(ROOT / L2F)
for mdl in ["stefan", "stefan_ridge075", "catboost_lo"]:
    r = one(fs, method=mdl)
    add(f"L2F_{mdl}", L2F, f"method={mdl}", "current_pooled / random_median", f"{r.current_pooled:.2f} / {r.random_median:.2f}",
        "", "서술(EXPERIMENT_LOG 219행)",
        f"random p05-p95 {r.random_p05:.2f}-{r.random_p95:.2f}; Δ vs Stefan 무작위 배정 중앙값 {r.delta_vs_stefan_random_median}; "
        f"음수 배정 {r.n_random_delta_negative}/{int(r.n_random)}")

# ---- LGF-N2, N4 (lgfn_tests.csv) ---------------------------------------------------------------
ln = pd.read_csv(ROOT / LGFN, low_memory=False)
n2v = ln[(ln.test_id == "LGF-N2") & (ln.scope == "verdict_aux")]
for _, r in n2v.iterrows():
    add(f"N2_verdict_{r.learner if isinstance(r.learner, str) else 'all'}", LGFN,
        f"test_id=LGF-N2, scope=verdict_aux, learner={r.learner}", "verdict", r.verdict, "", "LGF 10절(LGF-N2)", "")
for c in ["D0[mlp*]-D0[mlp]|n0|lam1", "D0[tabm*]-D0[tabm]|n0|lam1", "R1[ftt*]-R1[ftt]|nall|lam0.25"]:
    r = one(ln, test_id="LGF-N2", contrast=c, scope="MEAN", role="주")
    add(f"N2_{c}", LGFN, f"test_id=LGF-N2, contrast={c}, scope=MEAN, role=주", "delta", r.delta, r.verdict4,
        "LGF 10절(LGF-N2)", f"CI [{r.ci_lo:.2f}, {r.ci_hi:.2f}]; 블록 등가중 {r.delta_blockeq:.2f} [{r.ci_lo_beq:.2f}, {r.ci_hi_beq:.2f}]")
r = ln[(ln.test_id == "LGF-N4") & ln.spearman.notna()].iloc[0]
pts = pd.DataFrame(json.loads(r.points))
nz = int(((pts.gain == 0) & (pts.delta == 0)).sum())
add("N4_spearman", LGFN, "test_id=LGF-N4, spearman 열이 있는 행", "spearman", r.spearman, "서술(검정 없음)",
    "LGF 10절(LGF-N4)", f"n_points {int(r.n_points)}; (gain, delta) = (0, 0) 점 {nz}개")

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out[["id", "column", "value", "verdict"]].to_string())
