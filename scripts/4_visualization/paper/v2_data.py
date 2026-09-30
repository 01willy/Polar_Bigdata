"""F3 1부(본문 Fig 2–4, v2) 그림용 파생 표. 원천 표 → data/processed/paper_figs/*.csv.

원칙
  - 새 재표집이나 판정 계산을 하지 않는다. 값·CI·4분 판정(verdict4)·small_note 는 원천 표의 열을 그대로 옮긴다.
    예외는 두 점 추정뿐이다: Fig 2f·3d 의 레나·캐나다 2지역 P* 선(P0@tddm − P0)과 P1* 점(P1@ed − P0, n 40·160).
    층화 평균의 점 추정은 지역 점 추정의 등가중 평균이므로(L8·L29 MEAN 행으로 대조) 이 두 값은 lgx_curve 지역 값의
    산술 평균으로 낸다. CI 는 내지 않는다. 등록 대비 R1 − P1@ed 와의 가법 대조를 checks 에 기록한다.
  - 그림 모듈(fig2_label_curve.py, fig3_physics_use.py, fig4_min_labels.py)은 이 모듈의 산출 CSV 만 읽는다.
  - 열지 않는 표: LGD(lgd_*), LGT(lgt_*), LGF(lgf_*, lgfn_*), h39(lgw_*). 판정 기록 순서가 정해져 있다.

원천(읽기 전용)
  $LGS = results/rescale_lg/data/processed/lg: lg_curve.csv, lg_tests.csv, lg_minn.csv(Rescale ZovWo, 곡선 CI 1,000회)
  data/processed/lgx: lgx_curve.csv, lgx_tests.csv, lgx_lg_aux.csv, lgx_splitdist.csv, lgx_floor.csv(Rescale 조각의 로컬 집계, 10,000회)
  data/processed/paper_figs: pool_fixed_curve.csv, n1_target_summary.csv(커밋 390db15 모듈, 이 폴더로 실행)

실행: CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/4_visualization/paper/v2_data.py
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LGS = ROOT / "results" / "rescale_lg" / "data" / "processed" / "lg"
LGX = ROOT / "data" / "processed" / "lgx"
OUTD = ROOT / "data" / "processed" / "paper_figs"

MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
SUBS = ["AL-1", "AL-2", "AL-3", "AL-4", "AL-5", "AL-6", "CA-2", "CA-3", "LE-1", "LE-2"]
BASE_LAM = {"P1": 0.0, "P2": 0.0, "P3": 0.0, "V1": 0.0, "D0": 1.0, "D1": 1.0, "R0": 0.25, "R1": 0.25, "R2": 0.25, "R3": 0.25}
N_ALL = -1
NBOOT_CURVE = 1000            # lg_curve·lg_minn(h40 --nboot)
NBOOT_AUX = 10000             # lgx_lg_aux·lgx_tests·pool_fixed_curve
V4_EN = {"우세": "superior", "동등": "equivalent", "미결정": "undecided", "열세": "inferior", "판정 불가": "not determinable",
         "행 없음": "no row"}
MEAN4_T = "MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"
MEAN3_T = "MEAN[Lena|x,Canada|x,Alaska|x]"


# ================================================================ 읽기
def _read(p: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(p, low_memory=False, **kw)


def lg_curve() -> pd.DataFrame:
    return _read(LGS / "lg_curve.csv", dtype=dict(alpha=str, learner=str, placement=str, method=str, target=str, mode=str))


def lgx_curve() -> pd.DataFrame:
    return _read(LGX / "lgx_curve.csv", dtype=dict(alpha=str, learner=str, placement=str, method=str, target=str, mode=str))


def base_rows(C: pd.DataFrame, methods, targets=None, mode="x") -> pd.DataFrame:
    """기준 구성(셀 무작위, α 1, 방법별 기준 λ, 학습기 catboost_lo 또는 none)."""
    q = C[C.method.isin(methods) & C.learner.isin(["catboost_lo", "none"]) & (C.alpha == "1") & (C.placement == "cell")]
    if "axis" in q:
        q = q[q.axis.isin(["method", "base", "x9"])]
    if mode is not None:
        q = q[q["mode"] == mode]
    if targets is not None:
        q = q[q.target.isin(targets)]
    q = q[np.isclose(q.lam.astype(float), q.method.map(BASE_LAM).fillna(0.0))]
    return q.copy()


def v4(s) -> str:
    return V4_EN.get(str(s), "no row" if pd.isna(s) else str(s))


def small(s) -> bool:
    return isinstance(s, str) and "0.5 cm" in s


def n_label(n) -> str:
    return "all" if int(n) == N_ALL else str(int(n))


def pick(T: pd.DataFrame, contrast: str, scope: str = "MEAN", target: str | None = None, test_id: str | None = None) -> pd.Series | None:
    q = T[(T.contrast == contrast) & (T.scope == scope)]
    if test_id is not None:
        q = q[q.test_id == test_id]
    if target is not None:
        q = q[q.target == target]
    if "primary" in q and scope == "MEAN" and len(q) > 1:
        q = q[q.primary.astype(str).str.lower() == "true"]
    if len(q) != 1:
        return None
    return q.iloc[0]


def row_out(r: pd.Series | None, **extra) -> dict:
    """판정 표 한 행 → 그림 행(값·CI·두 가중·4분 판정 그대로)."""
    if r is None:
        return dict(extra, delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, delta_beq=np.nan, ci_lo_beq=np.nan, ci_hi_beq=np.nan,
                    verdict4="행 없음", verdict=v4("행 없음"), small_effect=False, pool="", ci_dependence="")
    return dict(extra, delta=r.get("delta"), ci_lo=r.get("ci_lo"), ci_hi=r.get("ci_hi"), delta_beq=r.get("delta_blockeq"),
                ci_lo_beq=r.get("ci_lo_beq"), ci_hi_beq=r.get("ci_hi_beq"), verdict4=r.get("verdict4"), verdict=v4(r.get("verdict4")),
                small_effect=small(r.get("small_note")), pool=r.get("pool", ""), ci_dependence=r.get("ci_dependence", ""))


# ================================================================ Fig 2
def fig2(checks: dict) -> dict:
    C = lg_curve()
    X = lgx_curve()
    Tx = _read(LGX / "lgx_tests.csv")
    A = _read(LGX / "lgx_lg_aux.csv")
    Tl = _read(LGS / "lg_tests.csv")
    P = _read(OUTD / "pool_fixed_curve.csv")

    # a–d 지역 곡선
    q = base_rows(C, ["P1", "R0", "R1", "R2", "D0"], MAIN4)
    reg = q[["target", "method", "n", "n_lab", "d_p0", "d_p0_lo", "d_p0_hi", "d_p0_beq", "d_p0_beq_lo", "d_p0_beq_hi",
             "rmse", "rmse_p0", "n_splits_used", "point_only", "ci_flag"]].copy()
    reg["n_label"] = [n_label(n) for n in reg.n]
    reg["nboot"] = NBOOT_CURVE
    reg["source"] = "lg_curve.csv"
    # P* 가로선(지역), P1* 점(레나·캐나다, n 40·160)
    xs = base_rows(X, ["P0@tddm"], MAIN4 + ["Alaska"])
    xs = xs[xs.n == 0][["target", "d_p0", "d_p0_lo", "d_p0_hi", "verdict4_p0"]].rename(columns={"d_p0": "pstar_d", "d_p0_lo": "pstar_lo", "d_p0_hi": "pstar_hi"})
    xp = base_rows(X, ["P1@ed"], ["Lena", "Canada"])
    xp = xp[xp.n.isin([40, 160])][["target", "n", "d_p0", "d_p0_lo", "d_p0_hi", "verdict4_p0"]]
    # 대조: lgx_curve 의 P1(같은 조각 계열) 과 lg_curve 의 P1 점 추정이 같은가
    xp1 = base_rows(X, ["P1"], MAIN4)[["target", "n", "d_p0"]].merge(reg[reg.method == "P1"][["target", "n", "d_p0"]], on=["target", "n"], suffixes=("_lgx", "_lg"))
    checks["fig2_lgx_vs_lg_P1_max_abs_diff_cm"] = float(np.nanmax(np.abs(xp1.d_p0_lgx - xp1.d_p0_lg))) if len(xp1) else None

    # e–f 풀 곡선(구성 고정)
    pm = P[(P.scope == "MEAN") & (P.baseline == "P0") & P.method.isin(["P1", "R0", "R1", "R2", "D0"])].copy()
    pool = pm[["edition", "method", "n", "n_label", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "verdict4",
               "status", "composition_complete", "n_ci_regions"]].copy()
    pool["nboot"] = NBOOT_AUX
    # P* 선: E1 = lgx_tests L29 MEAN 행, E2 = 레나·캐나다 지역 값 평균(점 추정)
    r = pick(Tx, "P0@tddm-P0|n0", "MEAN", test_id="L29")
    e1_pstar = float(r.delta)
    reg_pstar = dict(zip(xs.target, xs.pstar_d))
    e1_check = float(np.mean([reg_pstar[t] for t in MAIN4]))
    checks["fig2_pstar_E1_table_vs_region_mean_cm"] = abs(e1_pstar - e1_check)
    e2_pstar = float(np.mean([reg_pstar["Lena"], reg_pstar["Canada"]]))
    pstar_pool = pd.DataFrame([dict(edition="E1_P4_n_le_10", pstar=e1_pstar, pstar_lo=r.ci_lo, pstar_hi=r.ci_hi, source="lgx_tests L29 P0@tddm-P0|n0 MEAN"),
                               dict(edition="E2_LenaCanada_all_n", pstar=e2_pstar, pstar_lo=np.nan, pstar_hi=np.nan,
                                    source="lgx_curve P0@tddm d_p0 레나·캐나다 평균(점 추정)")])
    # P1* 점(E2): 레나·캐나다 P1@ed − P0 평균, 등록 대비 R1 − P1@ed 와 가법 대조
    p1s = []
    for n in (40, 160):
        v = float(xp[xp.n == n].d_p0.mean())
        r1 = float(pm[(pm.edition == "E2_LenaCanada_all_n") & (pm.method == "R1") & (pm.n == n)].delta.iloc[0])
        reg_c = pick(Tx, f"R1-P1@ed|n{n}", "MEAN", test_id="L29")
        checks[f"fig2_additivity_R1_P1ed_n{n}_cm"] = abs((r1 - v) - float(reg_c.delta))
        p1s.append(dict(edition="E2_LenaCanada_all_n", n=n, p1star=v, source="lgx_curve P1@ed d_p0 레나·캐나다 평균(점 추정)"))
    p1star_pool = pd.DataFrame(p1s)

    # 기호 표(판정은 원천 열 그대로)
    rows = []
    for n in (3, 10, 40, 160, 320, 1000, N_ALL):
        rows.append(row_out(pick(A, f"R1-P1|n{n}", "MEAN", test_id="L4"), row="R1-P1", hyp="L4", n=n, source="lgx_lg_aux"))
    for n in (10, N_ALL):
        rows.append(dict(row="R1-P1*", hyp="L29", n=n, verdict="same as P1", verdict4="P1* = P1", source="7.1a(P1 보다 우세한 재보정 기준선 없음)",
                         pool="", small_effect=False))
    for n in (40, 160):
        rows.append(row_out(pick(Tx, f"R1-P1@ed|n{n}", "MEAN", test_id="L29"), row="R1-P1*", hyp="L29", n=n, source="lgx_tests"))
    for n in (10, 40, 160):
        rows.append(row_out(pick(A, f"R2-R1|n{n}", "MEAN", test_id="L2"), row="R2-R1", hyp="L2", n=n, source="lgx_lg_aux"))
    rows.append(row_out(pick(A, "R1-P0|all", "MEAN", test_id="L8"), row="R1-P0", hyp="L8", n=N_ALL, source="lgx_lg_aux"))
    rows.append(row_out(pick(Tx, "R1-P0@tddm|all", "MEAN", test_id="L29"), row="R1-P*", hyp="L29", n=N_ALL, source="lgx_tests"))
    vm = pd.DataFrame(rows)
    vm["n_label"] = [n_label(n) for n in vm.n]
    # 등록 판정 이름(lg_tests verdict 행). 캡션용
    regv = Tl[Tl.scope == "verdict"][["test_id", "contrast", "verdict"]].copy()
    # L1 지역 4분 판정(보조 열). Fig 2 a–d 기호 줄
    l1 = A[(A.test_id == "L1") & (A.scope == "region")][["target", "n", "delta", "verdict4"]].copy()
    l1["verdict"] = l1.verdict4.map(v4)
    return dict(fig2_region_curves=reg, fig2_pstar_region=xs, fig2_p1star_region=xp, fig2_pool_curves=pool, fig2_pstar_pool=pstar_pool,
                fig2_p1star_pool=p1star_pool, fig2_verdicts=vm, fig2_registered=regv, fig2_L1_region=l1)


# ================================================================ Fig 3
def fig3(checks: dict) -> dict:
    Tx = _read(LGX / "lgx_tests.csv")
    A = _read(LGX / "lgx_lg_aux.csv")
    C = lg_curve()
    P = _read(OUTD / "pool_fixed_curve.csv")
    S = _read(LGX / "lgx_splitdist.csv")
    F = _read(LGX / "lgx_floor.csv")
    X = lgx_curve()

    # a: L15 위약
    ra = []
    for pl in ("shuffle", "const_t", "const_src", "tddlin"):
        for n in (0, 10):
            ra.append(row_out(pick(Tx, f"D1-D1@{pl}|n{n}", "MEAN", test_id="L15"), placebo=pl, n=n, contrast=f"D1-D1@{pl}|n{n}"))
    a = pd.DataFrame(ra)
    # b: L10, L12, L17
    spec_b = [("L10", "R0-F1k|n0", "R0 − F1k", 0, None), ("L10", "R0-F1a|n0", "R0 − F1a", 0, None),
              ("L10", "R1-F1k|n10", "R1 − F1k", 10, None), ("L10", "R1-F1n|n10", "R1 − F1n", 10, None)]
    for n in (10, N_ALL):
        for lam in ("0.25", "1.0"):
            spec_b.append(("L12", f"RM-R1|n{n}|lam{lam}", "RM − R1", n, lam))
    for n in (10, N_ALL):
        for lam in ("0.25", "1.0"):
            spec_b.append(("L17", f"R1-R1s|n{n}|lam{lam}", "R1 − R1s", n, lam))
    rb = [row_out(pick(Tx, c, "MEAN", test_id=h), hyp=h, contrast=c, label=lab, n=n, lam=lam) for h, c, lab, n, lam in spec_b]
    b = pd.DataFrame(rb)
    b["n_label"] = [n_label(n) for n in b.n]

    # c: 라벨 전량
    rc = []
    for t in MAIN4:
        rc.append(row_out(pick(A, "R1-P0|all", "region", target=f"{t}|x", test_id="L8"), block="R1 − P0", hyp="L8", target=t, kind="region"))
    rc.append(row_out(pick(A, "R1-P0|all", "MEAN", target=MEAN4_T, test_id="L8"), block="R1 − P0", hyp="L8", target="P4 mean", kind="mean4"))
    rc.append(row_out(pick(A, "R1-P0|all", "MEAN3", target=MEAN3_T, test_id="L8"), block="R1 − P0", hyp="L8", target="Lena, Canada, Alaska mean", kind="mean3"))
    rc.append(row_out(pick(A, "R1-P0|all", "region", target="Alaska|x", test_id="L8"), block="R1 − P0", hyp="L8", target="Alaska", kind="ref"))
    rc.append(row_out(pick(Tx, "R1-P0@tddm|all", "MEAN", test_id="L29"), block="R1 − P*", hyp="L29", target="P4 mean", kind="mean4"))
    for t in MAIN4:
        rc.append(row_out(pick(A, "R1-P1|n-1", "region", target=f"{t}|x", test_id="L4"), block="R1 − P1", hyp="L4", target=t, kind="region"))
    rc.append(row_out(pick(A, "R1-P1|n-1", "MEAN", target=MEAN4_T, test_id="L4"), block="R1 − P1", hyp="L4", target="P4 mean", kind="mean4"))
    c = pd.DataFrame(rc)
    # 대조: lgx_lg_aux 점 추정 = lg_tests 점 추정(L8)
    Tl = _read(LGS / "lg_tests.csv")
    l8 = Tl[(Tl.test_id == "L8") & (Tl.scope.isin(["region", "MEAN4"]))][["target", "delta"]]
    a8 = A[(A.test_id == "L8") & (A.scope.isin(["region", "MEAN"]))][["target", "delta"]]
    m8 = l8.merge(a8, on="target", suffixes=("_lg", "_aux"))
    checks["fig3c_L8_lg_tests_vs_lgx_lg_aux_max_abs_diff_cm"] = float(np.max(np.abs(m8.delta_lg - m8.delta_aux)))
    # 분할 50회 비율 띠(L31)
    s = S[(S.primary_source == True) & S.contrast.isin(["R1-P0|all", "R1-P1|all"])]  # noqa: E712
    s = s[s.target.isin([f"{t}|x" for t in MAIN4 + ["Alaska"]])][["source", "target", "contrast", "n_splits", "frac_neg", "lg5_outside_10_90"]].copy()
    s["target"] = s.target.str.split("|").str[0]
    # 오차 하한 띠: floor − RMSE(P0)(전량 R1 행의 rmse_p0)
    cq = base_rows(C, ["R1"], ["Lena", "Canada", "Alaska"])
    cq = cq[cq.n == N_ALL][["target", "rmse_p0"]]
    fl = F[F.scope == "eval"][["region", "floor_rmse_cm", "n_cells_used"]].rename(columns={"region": "target"})
    floor = cq.merge(fl, on="target")
    floor["floor_minus_p0"] = floor.floor_rmse_cm - floor.rmse_p0
    l31 = Tx[(Tx.test_id == "L31") & (Tx.scope == "verdict_aux")][["verdict", "note"]]

    # d: 계수 부분 풀링(E1·E2, 기준선 P0)
    d = P[(P.scope == "MEAN") & (P.baseline == "P0") & P.method.isin(["P1", "P2", "P3", "V1"])][
        ["edition", "method", "n", "n_label", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "verdict4", "status"]].copy()
    xp = base_rows(X, ["P1@ed"], ["Lena", "Canada"])
    d_p1s = pd.DataFrame([dict(edition="E2_LenaCanada_all_n", n=n, p1star=float(xp[xp.n == n].d_p0.mean())) for n in (40, 160)])
    return dict(fig3_a=a, fig3_b=b, fig3_c=c, fig3_c_splits=s, fig3_c_floor=floor, fig3_L31=l31, fig3_d=d, fig3_d_p1star=d_p1s)


# ================================================================ Fig 4
def fig4(checks: dict) -> dict:
    M = _read(LGS / "lg_minn.csv", dtype=dict(alpha=str, learner=str, placement=str))
    Tl = _read(LGS / "lg_tests.csv")
    A = _read(LGX / "lgx_lg_aux.csv")
    Tx = _read(LGX / "lgx_tests.csv")
    P = _read(OUTD / "pool_fixed_curve.csv")
    N1 = _read(OUTD / "n1_target_summary.csv")

    # a: 대상별 n*
    q = M[M.method.isin(["P1", "R1"]) & M.learner.isin(["catboost_lo", "none"]) & (M.alpha == "1") & (M.placement == "cell")]
    q = q[np.isclose(q.lam.astype(float), q.method.map(BASE_LAM))]
    keep = [(t, "x") for t in MAIN4 + ["Alaska"]] + [(t, "i") for t in SUBS]
    q = q[[(t, m) in keep for t, m in zip(q.target, q["mode"])]]
    q = q[q.point_only.astype(str).str.lower() != "true"]
    rows = []
    for _, r in q.iterrows():
        base = dict(target=r.target, mode=r["mode"], n_max=int(r.n_max), n_tested=int(r.n_tested), n_splits_valid=int(r.n_splits_valid),
                    flag=r.flag if isinstance(r.flag, str) else "")
        cz = str(r.censored).lower() == "true"
        if r.method == "P1":
            rows.append(dict(base, contrast="P1 − P0", n_star=np.nan if cz else r.n_star, censored=cz))
        else:
            rows.append(dict(base, contrast="R1 − P0", n_star=np.nan if cz else r.n_star, censored=cz))
            cz1 = str(r.censored_p1).lower() == "true"
            rows.append(dict(base, contrast="R1 − P1", n_star=np.nan if cz1 else r.n_star_p1, censored=cz1))
    a = pd.DataFrame(rows)
    grid = [0, 3, 10, 40, 160, 320, 1000]
    a["n_lo"] = [grid[grid.index(int(v)) - 1] if np.isfinite(v) else np.nan for v in a.n_star]

    # b: L4(R1 − P1) 풀 곡선과 R1 − P1*
    pb = P[(P.scope == "MEAN") & (P.baseline == "P1") & (P.method == "R1")][
        ["edition", "n", "n_label", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "verdict4", "status"]].copy()
    r_all = pick(A, "R1-P1|n-1", "MEAN", target=MEAN4_T, test_id="L4")
    pb = pd.concat([pb, pd.DataFrame([dict(edition="P4_all(lgx_lg_aux L4)", n=N_ALL, n_label="all", delta=r_all.delta, ci_lo=r_all.ci_lo,
                                           ci_hi=r_all.ci_hi, delta_blockeq=r_all.delta_blockeq, ci_lo_beq=r_all.ci_lo_beq,
                                           ci_hi_beq=r_all.ci_hi_beq, verdict4=r_all.verdict4, status="ok")])], ignore_index=True)
    # 같은 대비의 두 표(pool_fixed_curve E1 과 lgx_lg_aux L4 MEAN)가 같은 값인가(같은 seed 규칙)
    e1 = P[(P.edition == "E1_P4_n_le_10") & (P.scope == "MEAN") & (P.baseline == "P1") & (P.method == "R1") & (P.n == 10)].iloc[0]
    a10 = pick(A, "R1-P1|n10", "MEAN", test_id="L4")
    checks["fig4b_pool_vs_lgx_lg_aux_L4_n10_ci_lo_abs_diff"] = abs(float(e1.ci_lo) - float(a10.ci_lo))
    ps_rows = [row_out(pick(Tx, f"R1-P1@ed|n{n}", "MEAN", test_id="L29"), n=n, contrast=f"R1-P1@ed|n{n}") for n in (40, 160)]
    pstar = pd.DataFrame(ps_rows)
    vb = [row_out(pick(A, f"R1-P1|n{n}", "MEAN", test_id="L4"), row="R1-P1", n=n) for n in (3, 10, 40, 160, 320, 1000, N_ALL)]
    vb += [dict(row="R1-P1*", n=n, verdict="same as P1", verdict4="P1* = P1", pool="", small_effect=False) for n in (10, N_ALL)]
    vb += [row_out(pick(Tx, f"R1-P1@ed|n{n}", "MEAN", test_id="L29"), row="R1-P1*", n=n) for n in (40, 160)]
    vbd = pd.DataFrame(vb)
    vbd["n_label"] = [n_label(n) for n in vbd.n]

    # c: L3 중첩 선택 α 대 α = 1(주 행)
    l3 = Tl[(Tl.test_id == "L3") & (Tl.role == "주") & (Tl.scope == "region")][["target", "n_star", "n_star_alpha1", "censored", "hit"]].copy()
    mr = M[(M.method == "R0") & (M.learner == "catboost_lo") & (M.placement == "cell") & np.isclose(M.lam.astype(float), 0.25)]
    out = []
    for _, r in l3.iterrows():
        t, md = r.target.split("|")
        for cond, al, ns in (("nested", "nested", r.n_star), ("alpha1", "1", r.n_star_alpha1)):
            mm = mr[(mr.target == t) & (mr["mode"] == md) & (mr.alpha == al)]
            nmax = int(mm.n_max.iloc[0]) if len(mm) else np.nan
            cz = not np.isfinite(ns)
            if len(mm):
                cz_m = str(mm.censored.iloc[0]).lower() == "true"
                nst_m = mm.n_star.iloc[0]
                if cond == "alpha1" and not cz and not cz_m and float(nst_m) != float(ns):
                    checks[f"fig4c_{t}_{md}_alpha1_mismatch"] = [float(ns), float(nst_m)]
            out.append(dict(target=t, mode=md, cond=cond, n_star=ns, censored=cz, n_max=nmax, hit=r.hit))
    c = pd.DataFrame(out)
    l3v = Tl[(Tl.test_id == "L3") & (Tl.scope.isin(["verdict", "verdict_aux"]))][["scope", "role", "verdict"]]

    # d: N1 대상별 4분 판정 수
    d = N1[(N1.source == "lg") & N1.group.isin(["P4", "sub_i"]) &
           (((N1.kind == "d_p0") & N1.method.isin(["R1", "P1"])) | ((N1.kind == "d_p1") & (N1.method == "R1")))][
        ["group", "kind", "method", "n", "n_label", "n_group", "n_rows", "n_missing", "composition_complete", "n_superior", "n_equivalent",
         "n_undecided", "n_inferior", "n_na", "missing_targets", "nboot"]].copy()
    d["contrast"] = [f"{m} − {'P0' if k == 'd_p0' else 'P1'}" for k, m in zip(d.kind, d.method)]
    return dict(fig4_a=a, fig4_b=pb, fig4_b_p1star=pstar, fig4_b_verdicts=vbd, fig4_c=c, fig4_c_verdict=l3v, fig4_d=d)


# ================================================================ 기록
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


INPUTS = [LGS / "lg_curve.csv", LGS / "lg_tests.csv", LGS / "lg_minn.csv", LGX / "lgx_curve.csv", LGX / "lgx_tests.csv",
          LGX / "lgx_lg_aux.csv", LGX / "lgx_splitdist.csv", LGX / "lgx_floor.csv", OUTD / "pool_fixed_curve.csv",
          OUTD / "n1_target_summary.csv"]


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    checks: dict = {}
    tables = {}
    for f in (fig2, fig3, fig4):
        tables.update(f(checks))
    for k, df in tables.items():
        df.to_csv(OUTD / f"{k}.csv", index=False)
    tol = 1e-6
    bad = {k: v for k, v in checks.items() if isinstance(v, float) and "ci_lo" not in k and v > tol}
    bad.update({k: v for k, v in checks.items() if isinstance(v, list)})
    meta = dict(module="scripts/4_visualization/paper/v2_data.py", created=_dt.datetime.now().isoformat(timespec="seconds"),
                inputs=[dict(path=str(p.relative_to(ROOT)), sha256=sha256(p)) for p in INPUTS],
                outputs=sorted(f"{k}.csv" for k in tables), checks=checks, checks_failed=bad,
                rules=dict(values="원천 열 그대로. 새 재표집·판정 없음", derived_points="E2(레나·캐나다) P* 선과 P1* 점은 lgx_curve 지역 값의 등가중 평균(점 추정)",
                           nboot=dict(lg_curve=NBOOT_CURVE, lgx=NBOOT_AUX, pool_fixed_curve=NBOOT_AUX),
                           not_opened="lgd_*, lgt_*, lgf_*, lgfn_*, lgw_*(판정 기록 순서)"))
    (OUTD / "v2_data_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    print(f"[v2_data] 표 {len(tables)}개 → {OUTD.relative_to(ROOT)}, 대조 실패 {len(bad)}")
    if bad:
        print(json.dumps(bad, ensure_ascii=False, indent=1))
        sys.exit(1)


if __name__ == "__main__":
    main()
