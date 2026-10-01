"""F3 1부(본문 Fig 2–4, v2) 그림용 파생 표. 원천 표 → data/processed/paper_figs/*.csv.

원칙
  - 새 재표집이나 판정 계산을 하지 않는다. 값·CI·4분 판정(verdict4)·small_note 는 원천 표의 열을 그대로 옮긴다.
    예외는 두 점 추정뿐이다: Fig 2f·3d 의 레나·캐나다 2지역 P* 선(P0@tddm − P0)과 P1* 점(P1@ed − P0, n 40·160).
    층화 평균의 점 추정은 지역 점 추정의 등가중 평균이므로(L8·L29 MEAN 행으로 대조) 이 두 값은 lgx_curve 지역 값의
    산술 평균으로 낸다. CI 는 내지 않는다. 등록 대비 R1 − P1@ed 와의 가법 대조를 checks 에 기록한다.
  - 그림 모듈(fig2_label_curve.py, fig3_physics_use.py, fig4_min_labels.py)은 이 모듈의 산출 CSV 만 읽는다.
  - 열지 않는 표: LGT(lgt_*), LGF(lgf_*, lgfn_*). 1부는 LGD·h39(lgw_*)도 열지 않았고, 2부는 판정 기록(LG 7.3 J5, WRAPUP J7)
    뒤에 lgw_*, lgd 적격 표·단위 JSON, lgu_a_tests(AB10 CI)를 읽는다. LGF 는 창 마감(2026-10-02 05:01) 전이라 열지 않는다.

원천(읽기 전용)
  $LGS = results/rescale_lg/data/processed/lg: lg_curve.csv, lg_tests.csv, lg_minn.csv(Rescale ZovWo, 곡선 CI 1,000회)
  data/processed/lgx: lgx_curve.csv, lgx_tests.csv, lgx_lg_aux.csv, lgx_splitdist.csv, lgx_floor.csv(Rescale 조각의 로컬 집계, 10,000회)
  data/processed/paper_figs: pool_fixed_curve.csv, n1_target_summary.csv(커밋 390db15 모듈, 이 폴더로 실행)

실행: CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/4_visualization/paper/v2_data.py
      3부(Fig 6, LGF 창 마감 뒤): ... v2_data.py --only fig6 → fig6_*.csv 와 v2_data_fig6_meta.json 만 쓴다(다른 표는 건드리지 않는다).
      3부 원천: lgw_bundle(AB1, AB2, AB10), lgx_tests(L29, L30, L11, L25), $LGS/lg_tests(L1 등록 판정), lgt_tests(L34), lgf_tests(LGF-F1),
      lgfn_tests(LGF-N1), lgu_b_intervals·lgu_b_tests(LGU-B2), lgu_a_tests(LGU-A1), h4/c2_coverage(hier2_cdf 참조), lgx_conformal.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
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


# ================================================================ 2부(F3 part 2): AB 묶음, Fig 1, 5, 7, Table 1, Fig 2–4 갱신
LGW = ROOT / "data" / "processed" / "lgw"
LGDD = ROOT / "data" / "processed" / "lgd"
LGU = ROOT / "data" / "processed" / "lgu"
PROC = ROOT / "data" / "processed"
MAIN4_W = "MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"
AB_RULE = {"(a)": "a", "(b)": "b", "(c)": "c", "(d)": "d"}


def _rule_code(s) -> str:
    s = str(s)
    for k, v in AB_RULE.items():
        if s.startswith(k):
            return v
    return ""


def bundle() -> dict:
    """lgw_bundle → 초록 묶음 표(MEAN)와 지역 행. 값·판정·규칙 열은 그대로 옮긴다."""
    B = _read(LGW / "lgw_bundle.csv")
    m = B[B.scope == "MEAN"].copy()
    out = pd.DataFrame(dict(ab=m.ab, hypothesis=m.hypothesis, contrast=m.contrast, source=m.source, pool=m.pool,
                            delta=m.delta, ci_lo=m.ci_lo, ci_hi=m.ci_hi, delta_beq=m.delta_blockeq, ci_lo_beq=m.ci_lo_beq,
                            ci_hi_beq=m.ci_hi_beq, verdict4=m.verdict4, verdict=m.verdict4.map(v4), small_effect=m.small_note.map(small),
                            holm_p=m.holm_p, holm_p_eq=m.holm_p_eq, holm_m=m.holm_m, abstract_rule=m.abstract_rule,
                            rule=m.abstract_rule.map(_rule_code), equiv_note=m.equiv_note, resample_dependence=m.resample_dependence,
                            h42_verdict4=m.h42_verdict4, verdict4_rel=m.verdict4_rel, delta_rel_margin=m.delta_rel, rmse_p0=m.rmse_p0,
                            nboot=m.nboot, note=m.note))
    r = B[B.scope == "region"][["ab", "contrast", "target", "delta", "ci_lo", "ci_hi", "verdict4"]].copy()
    r["region"] = r.target.str.split("|").str[0]
    r["verdict"] = r.verdict4.map(v4)
    return dict(ab_bundle=out.reset_index(drop=True), ab_bundle_regions=r.reset_index(drop=True))


def _ab_lookup(ab: pd.DataFrame, code: str) -> pd.Series:
    q = ab[ab.ab == code]
    assert len(q) == 1, code
    return q.iloc[0]


def part2_updates(checks: dict, ab: pd.DataFrame) -> dict:
    """Fig 2–4 갱신 자료: AB 칸 대응, Fig 3b L10 모든 n, Fig 3c SD/SE 표지, Fig 4c L3 4분 보조."""
    Tx = _read(LGX / "lgx_tests.csv")
    A = _read(LGX / "lgx_lg_aux.csv")
    P = _read(OUTD / "pool_fixed_curve.csv")
    # Fig 2 기호 표의 AB 칸(행, n) 과 Fig 4b, Fig 3 표지 위치
    cells = [("fig2", "R1-P1", 10, "AB5"), ("fig2", "R1-P1", N_ALL, "AB9"), ("fig2", "R2-R1", 10, "AB6"), ("fig2", "R1-P0", N_ALL, "AB8"),
             ("fig4b", "R1-P1", 10, "AB5"), ("fig4b", "R1-P1", N_ALL, "AB9"),
             ("fig3a", "shuffle", 0, "AB3"), ("fig3b", "F1k", 10, "AB7"), ("fig3c", "R1 − P0|P4 mean", N_ALL, "AB8"),
             ("fig3c", "R1 − P1|P4 mean", N_ALL, "AB9"), ("fig3d", "E1_P4_n_le_10|P1", 10, "AB4")]
    # 같은 대비의 원천 표 판정(그림 기호)과 lgw_bundle 판정을 대조한다
    same_src = {"AB5": pick(A, "R1-P1|n10", "MEAN", test_id="L4"), "AB9": pick(A, "R1-P1|n-1", "MEAN", target=MEAN4_T, test_id="L4"),
                "AB6": pick(A, "R2-R1|n10", "MEAN", test_id="L2"), "AB8": pick(A, "R1-P0|all", "MEAN", target=MEAN4_T, test_id="L8"),
                "AB3": pick(Tx, "D1-D1@shuffle|n0", "MEAN", test_id="L15"), "AB7": pick(Tx, "R1-F1k|n10", "MEAN", test_id="L10")}
    rows = []
    for fig, key, n, code in cells:
        b = _ab_lookup(ab, code)
        s = same_src.get(code)
        fv = str(s.verdict4) if s is not None else ""
        rows.append(dict(figure=fig, key=key, n=n, ab=code, verdict4_bundle=b.verdict4, verdict_bundle=b.verdict, rule=b.rule,
                         abstract_rule=b.abstract_rule, holm_p=b.holm_p, holm_p_eq=b.holm_p_eq, delta_bundle=b.delta,
                         verdict4_figure_source=fv, delta_figure_source=float(s.delta) if s is not None else np.nan,
                         verdict_agree=(fv == str(b.verdict4)) if s is not None else np.nan))
    abc = pd.DataFrame(rows)
    dis = abc[(abc.verdict_agree == False)]  # noqa: E712
    checks["info_ab_cells_verdict_disagreements"] = dis[["figure", "ab", "verdict4_bundle", "verdict4_figure_source"]].to_dict("records")
    # AB4 = pool_fixed_curve E1 P1 − P0 n 10 과 같은 대비(점 추정 대조)
    e1 = P[(P.edition == "E1_P4_n_le_10") & (P.scope == "MEAN") & (P.baseline == "P0") & (P.method == "P1") & (P.n == 10)]
    checks["fig3d_AB4_vs_pool_E1_P1_n10_abs_diff_cm"] = abs(float(e1.delta.iloc[0]) - float(_ab_lookup(ab, "AB4").delta))
    checks["fig3c_AB8_vs_L8_mean_abs_diff_cm"] = abs(float(same_src["AB8"].delta) - float(_ab_lookup(ab, "AB8").delta))
    checks["fig3c_AB9_vs_L4_all_mean_abs_diff_cm"] = abs(float(same_src["AB9"].delta) - float(_ab_lookup(ab, "AB9").delta))

    # Fig 3b(위): L10 모든 n(주 4지역 층화 평균 행; n 40·160 은 풀 지역 2/4)
    q = Tx[(Tx.test_id == "L10") & (Tx.scope == "MEAN")]
    q = q[q.contrast.str.match(r"^(R0|R1)-(F1k|F1n|F1a)\|n")].copy()
    q["series"] = q.contrast.str.extract(r"-(F1[kna])\|")[0]
    q["n"] = q.n.astype(int)
    l10 = pd.DataFrame(dict(series=q.series, contrast=q.contrast, n=q.n, registered=q.primary.astype(str).str.lower() == "true",
                            role=q.role, pool=q.pool, delta=q.delta, ci_lo=q.ci_lo, ci_hi=q.ci_hi, delta_beq=q.delta_blockeq,
                            ci_lo_beq=q.ci_lo_beq, ci_hi_beq=q.ci_hi_beq, verdict4=q.verdict4, verdict=q.verdict4.map(v4),
                            small_effect=q.small_note.map(small), nboot=NBOOT_AUX)).sort_values(["series", "n"])
    l10["n_label"] = [n_label(n) for n in l10.n]
    l10m3 = Tx[(Tx.test_id == "L10") & (Tx.scope == "MEAN3") & Tx.contrast.str.match(r"^(R0|R1)-(F1k|F1n|F1a)\|n")][
        ["contrast", "n", "delta", "ci_lo", "ci_hi", "verdict4"]].copy()
    checks["fig3b_L10_registered_rows_eq4"] = int(l10.registered.sum()) == 4
    # 1부 fig3_b(L10 주 네 행)와 값 대조
    b1 = _read(OUTD / "fig3_b.csv")
    mm = b1[b1.hyp == "L10"].merge(l10, on="contrast", suffixes=("_p1", "_p2"))
    checks["fig3b_L10_part1_vs_part2_max_abs_diff_cm"] = float(np.max(np.abs(mm.delta_p1 - mm.delta_p2)))

    # Fig 3c: SD/SE 비 표지(WRAPUP 1.6)
    S = _read(LGW / "lgw_splitratio.csv")
    sd = S[S.contrast.isin(["R1-P0|all", "R1-P1|all"])][["target", "contrast", "source", "sd_between", "se_within_mean", "R", "R0",
                                                        "R_over_R0", "flag", "lg5_outside_10_90", "main_switch"]].copy()
    sd["region"] = sd.target.str.split("|").str[0]
    sd["flagged"] = sd.flag.astype(str).str.startswith("SD/SE")
    checks["info_fig3c_sdse_main_switch_any"] = bool(sd.main_switch.astype(str).str.lower().eq("true").any())

    # Fig 4c: L3 4분 보조(lgw_aux4)
    X = _read(LGW / "lgw_aux4.csv")
    x3 = X[X.test_id == "L3"][["target", "contrast", "n", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq",
                               "verdict4", "small_note", "ci_dependence", "nboot"]].copy()
    x3["verdict"] = x3.verdict4.map(v4)
    x3["small_effect"] = x3.small_note.map(small)
    x3["kind"] = np.where(x3.contrast.str.contains("R0@a1"), "nested − α1", "nested − P0")
    return dict(ab_cells=abc, fig3_b_L10_alln=l10, fig3_b_L10_mean3=l10m3, fig3_c_sdse=sd, fig4_c_aux4=x3)


# ---------------------------------------------------------------- Fig 5
def fig5(checks: dict) -> dict:
    L = _read(LGW / "lgw_l43.csv")
    Tw = _read(LGW / "lgw_tests.csv")
    Tx = _read(LGX / "lgx_tests.csv")
    D = _read(LGX / "lgx_distance.csv")
    main_t = ["Lena|x", "Canada|x", "Russia_W|x", "Russia_E|x"]
    a = L[(L.role == "주") & (L.method == "P1") & L.n.isin([10, 40]) & (L.target.isin(main_t) | (L.scope == "MEAN"))].copy()
    a = a[["method", "n", "ci_kind", "scope", "target", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "verdict4",
           "verdict4_draw_conditional", "draw_dependence", "ci_dependence", "pool", "win_rate_mean", "nboot"]]
    a["verdict"] = a.verdict4.map(v4)
    a["region"] = [("P4 mean" if int(n) == 10 else "Lena + Canada mean") if s == "MEAN" else t.split("|")[0]
                   for s, t, n in zip(a.scope, a.target, a.n)]
    blk = L[L.scope == "n_blocks_lab"][["method", "n", "placement", "n_blocks_lab_mean"]].copy()
    ver = Tw[Tw.test_id == "L43"][["test_id", "verdict", "stat", "blind"]]
    # b: L23 주 행(층화 평균)과 지역 행
    q = Tx[(Tx.test_id == "L23")]
    prim = q[(q.scope == "MEAN") & (q.primary.astype(str).str.lower() == "true")]
    keys = list(prim.contrast)
    reg = q[(q.scope == "region") & q.contrast.isin(keys) & q.target.isin(main_t)]
    b = pd.concat([prim.assign(kind="mean"), reg.assign(kind="region")])[
        ["contrast", "kind", "target", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "verdict4", "small_note", "pool"]].copy()
    b["verdict"] = b.verdict4.map(v4); b["small_effect"] = b.small_note.map(small)
    bv = q[q.scope == "verdict_aux"][["verdict"]]
    # c: L24 대상별 근 − 원 차
    c = Tx[(Tx.test_id == "L24") & (Tx.scope == "region")][["contrast", "target", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq",
                                                           "ci_hi_beq", "verdict4", "small_note"]].copy()
    c["n"] = c.contrast.str.extract(r"\|n(\d+)")[0].astype(int)
    c["which"] = np.where(c.contrast.str.startswith("[R1-P1]"), "R1 − P1", "P1 − P0")
    c["verdict"] = c.verdict4.map(v4); c["small_effect"] = c.small_note.map(small)
    cv = Tx[(Tx.test_id == "L24") & (Tx.scope == "verdict_aux")][["verdict"]]
    # d: 거리 층 값(서술), c 와 같은 대상, n = 40·160
    tg = sorted(set(c.target))
    d = D[D.target.isin(tg) & D.n.isin([40, 160]) & D.contrast.isin(["R1-P1", "P1-P0"])][
        ["target", "n", "contrast", "stratum", "delta", "ci_lo", "ci_hi", "verdict4", "n_splits"]].copy()
    checks["info_fig5_L43_mean_rows"] = int((a.scope == "MEAN").sum())
    checks["info_fig5_L24_rows"] = int(len(c))
    return dict(fig5_a=a, fig5_a_blocks=blk, fig5_a_verdict=ver, fig5_b=b, fig5_b_verdict=bv, fig5_c=c, fig5_c_verdict=cv, fig5_d=d)


# ---------------------------------------------------------------- Fig 7
def fig7(checks: dict, ab: pd.DataFrame, abr: pd.DataFrame) -> dict:
    Tw = _read(LGW / "lgw_tests.csv")
    S = _read(LGW / "lgw_scenarios.csv")
    SS = _read(LGW / "lgw_scenarios_summary.csv")
    K = _read(LGW / "lgw_sc3.csv")
    U = _read(LGU / "lgu_a_tests.csv")
    tests = Tw[Tw.test_id.isin(["SC1w", "SC1w-d10", "SC1w-P", "SC2w", "SC2w-n160", "SC3w", "SC3w-tau0.025", "SC3w-tau0.1", "SC3w-tau0.2",
                                "L43", "AK1w", "S-a", "S-b"])][["test_id", "role", "verdict", "stat", "flags", "blind"]].copy()
    b = S[S.independent.astype(str).str.lower() == "true"][["stage", "n", "recipe", "reference", "contrast", "target", "delta", "ci_lo", "ci_hi",
                                                            "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "verdict4", "noninf", "noninf_d10",
                                                            "worse", "small_note", "n_splits"]].copy()
    b["region"] = b.target.str.split("|").str[0]
    b["verdict"] = b.verdict4.map(v4); b["small_effect"] = b.small_note.map(small)
    bs = SS[["stage", "n", "recipe", "reference", "contrast", "main4_delta", "main4_ci_lo", "main4_ci_hi", "main4_verdict4", "main4_noninf",
             "main4_pool", "n_independent", "n_noninf", "worst_delta", "worst_target"]].copy()
    bs["verdict"] = bs.main4_verdict4.map(v4)
    c = K[((K.scope == "MEAN3") | (K.role == "판정"))][["scope", "role", "target", "tau", "delta", "ci_lo", "ci_hi", "delta_blockeq",
                                                    "ci_lo_beq", "ci_hi_beq", "verdict4", "label_ratio", "labels_mean",
                                                    "frac_stop_3", "frac_stop_10", "frac_stop_cap"]].copy()
    c["region"] = [("3-region mean" if s == "MEAN3" else t.split("|")[0]) for s, t in zip(c.scope, c.target)]
    # AB10: LGU-A1 n 10 의 셀 가중 2단 CI 와 기준 점수(B4) → % 환산
    u = U[(U.test == "LGU-A1") & (U.n.astype(str) == "10")].iloc[0]
    ab10 = dict(ab="AB10", delta_cm=float(u.delta), ci_lo_cm=float(u.ci_lo), ci_hi_cm=float(u.ci_hi), ref_score=float(u.ref_score),
                delta_pct=100 * float(u.delta) / float(u.ref_score), ci_lo_pct=100 * float(u.ci_lo) / float(u.ref_score),
                ci_hi_pct=100 * float(u.ci_hi) / float(u.ref_score), verdict4_lgu=u.verdict, nboot=int(u.nboot))
    checks["fig7_AB10_bundle_vs_lgu_delta_abs_diff"] = abs(float(_ab_lookup(ab, "AB10").delta) - ab10["delta_cm"])
    checks["fig7_AB10_verdict_agree"] = str(_ab_lookup(ab, "AB10").verdict4) == str(u.verdict)
    # δ_rel 문장(9.4): AB1–AB9 의 verdict4_rel 과 verdict4 비교
    m = ab[ab.ab != "AB10"]
    rel = dict(n_same=int((m.verdict4 == m.verdict4_rel).sum()), n_rows=int(len(m)),
               margin_cm_P4=float(m.delta_rel_margin.iloc[0]), rmse_p0_P4=float(m.rmse_p0.iloc[0]))
    checks["info_fig7_delta_rel_all_same"] = rel["n_same"] == rel["n_rows"]
    # SC1w 칸(10) 과 lgw_tests stat 의 칸 상태 대조
    st = tests[tests.test_id == "SC1w"].stat.iloc[0]
    sc = b[(b.recipe == "R1") & (b.reference == "P0") & b.n.isin([3, 10])]
    mis = []
    for r in sc.itertuples():
        want = "비열등" if str(r.noninf).lower() == "true" else ("열세" if r.verdict4 == "열세" else "미확인")
        tok = f"{r.region} n{int(r.n)}: {want}"
        if tok not in st:
            mis.append(tok)
    checks["fig7_SC1w_cells_vs_stat_mismatch"] = mis
    return dict(fig7_tests=tests, fig7_b=b, fig7_b_summary=bs, fig7_c=c, fig7_d_ab10=pd.DataFrame([ab10]),
                fig7_delta_rel=pd.DataFrame([rel]), fig7_d=ab, fig7_d_regions=abr)


# ---------------------------------------------------------------- Fig 1 와 Table 1 공통
def loc1km(lat, lon):
    """6B.3 셀 색인(1 km 위치). ky = floor(lat/0.009), kx = floor(lon·cos φ/0.009), φ = (ky + 0.5)·0.009°."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    ky = np.floor(lat / 0.009).astype(np.int64)
    phi = np.deg2rad((ky + 0.5) * 0.009)
    kx = np.floor(lon * np.cos(phi) / 0.009).astype(np.int64)
    return ky * 10_000_000 + kx


def block05(lat, lon):
    return np.floor(np.asarray(lat) / 0.5).astype(int) * 100000 + np.floor(np.asarray(lon) / 0.5).astype(int)


def _v3_cells() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "src"))
    from polar.m1_core import load_base
    d = load_base(PROC)
    d = d[np.isfinite(d.alt_cm.astype(float))].copy()
    return pd.DataFrame(dict(loc_id=d.loc_id.values, region=d.macro.values, lat=d.lat.values, lon=d.lon.values,
                             y=d.alt_cm.astype(float).values, s=d.e5_sqrt_tdd.astype(float).values, part="v3"))


def _v4_new() -> pd.DataFrame:
    B = _read(PROC / "fidelity_base_v4.csv", usecols=["loc_id", "lat", "lon", "alt_cm", "e5_sqrt_tdd", "source_id"])
    L = _read(PROC / "fidelity_base_v4_labels.csv", usecols=["loc_id", "part", "macro_v4", "lgd_role", "lic_unverified", "method"])
    d = B.merge(L, on="loc_id", validate="one_to_one")
    d = d[d.part == "new"].copy()
    return pd.DataFrame(dict(loc_id=d.loc_id.values, region=d.macro_v4.values, lat=d.lat.values, lon=d.lon.values, y=d.alt_cm.astype(float).values,
                             s=d.e5_sqrt_tdd.astype(float).values, lgd_role=d.lgd_role.values, lic_unverified=d.lic_unverified.fillna(0).astype(int).values,
                             method=d.method.values, part="new"))


def _lgd_units(target_dir: str) -> pd.DataFrame:
    import glob
    rows = []
    for f in sorted(glob.glob(str(LGDD / target_dir / "shards" / "*__cpu__*_unit.json"))):
        u = json.loads(Path(f).read_text())
        rows.append({k: u.get(k) for k in ("target", "mode", "split", "valid", "n_A", "nb_A", "n_eval", "nb_eval", "n_src", "E0", "E_own",
                                           "n_cells", "n_blocks", "n_valid_splits", "n_buffer_excluded")})
    return pd.DataFrame(rows)


# 원천 구성 대조값: docs/MANUSCRIPT_DRAFT_SUPPORT_2026-09-30.md M3 Table M3-1(원천 셀, 버퍼 제외, 알래스카 셀)
M3_TABLE = {"Lena": (14429, 1, 13606), "Canada": (16697, 20, 13586), "Russia_W": (17436, 0, 13606), "Russia_E": (17437, 0, 13606),
            "Russia_C": (16208, 1252, 13606), "Greenland": (17464, 0, 13606), "Alaska": (3860, 1, 0)}


def source_pool(v3: pd.DataFrame, target: str, buffer_km: float = 100.0):
    """h40 source_idx(모드 x, 대상 = macro) 와 같은 규칙: 대상 셀 제외 → 대상 셀과 대권 거리 < buffer_km 인 셀 제외."""
    from sklearn.neighbors import BallTree
    t = v3.region.values == target
    cand = ~t
    tree = BallTree(np.deg2rad(v3.loc[t, ["lat", "lon"]].values), metric="haversine")
    dist, _ = tree.query(np.deg2rad(v3.loc[cand, ["lat", "lon"]].values), k=1)
    near = dist[:, 0] * 6371.0088 < buffer_km
    idx = np.where(cand)[0]
    return idx[~near], int(near.sum())


def fig1(checks: dict) -> dict:
    v3 = _v3_cells()
    new = _v4_new()
    lic = _read(LGDD / "lgd_eligibility_lic.csv")
    Tg = _read(LGS / "lg_targets.csv")
    # 1 km 위치 정의 대조(v3 macro 의 위치 수 = lgw_label_units n_loc_1km)
    U = _read(LGW / "lgw_label_units.csv")
    v3["loc1km"] = loc1km(v3.lat, v3.lon); v3["block"] = block05(v3.lat, v3.lon)
    new["loc1km"] = loc1km(new.lat, new.lon); new["block"] = block05(new.lat, new.lon)
    mine = v3.groupby("region").loc1km.nunique()
    ref = U[U.kind == "macro"].set_index("target").n_loc_1km
    checks["fig1_loc1km_v3_mismatch"] = {k: [int(mine.get(k, -1)), int(v)] for k, v in ref.items() if int(mine.get(k, -1)) != int(v)}

    # a: 블록 원(종류별). v3 = 모든 F4_direct 셀, LGD = 대상 행 중 약관 확인분, NAtlantic 약관 확인분은 SI 표시
    tgt = new[new.lgd_role == "target"].copy()
    natl_elig_lic = bool(lic[lic.spec == "NAtlantic_lic"].eligible.astype(str).str.lower().eq("true").any())
    checks["info_fig1_natlantic_eligible_in_lic_edition"] = natl_elig_lic
    tgt["kind"] = np.where(tgt.lic_unverified == 1, "not_drawn_licence",
                           np.where((tgt.region == "NAtlantic") & (not natl_elig_lic), "natl_si", "lgd_added"))
    v3["kind"] = "v3"
    cells = pd.concat([v3.assign(lgd_role="v3", lic_unverified=0), tgt], ignore_index=True)
    drawn = cells[cells.kind != "not_drawn_licence"]
    blocks = drawn.groupby(["kind", "region", "block"]).agg(lat=("lat", "mean"), lon=("lon", "mean"), n_rows=("y", "size"),
                                                           n_loc_1km=("loc1km", "nunique")).reset_index()
    nd = tgt[tgt.kind == "not_drawn_licence"].groupby("region").size().rename("n_cells_not_drawn").reset_index()
    checks["info_fig1_not_drawn_licence_cells"] = dict(zip(nd.region, nd.n_cells_not_drawn.astype(int)))

    # b: z = ln(ALT/√TDD) 행
    rows_b = [("Lena", "Lena", "v3"), ("Canada", "Canada", "v3"), ("Russia_W", "Russia W", "v3"), ("Russia_E", "Russia E", "v3"),
              ("Russia_C", "Russia C", "v3"), ("Greenland", "Greenland", "v3"), ("Alaska", "Alaska (reference)", "v3"),
              ("Russia_C_LGD", "Russia C, LGD", "lgd"), ("Tibet_LGD", "Tibet, LGD", "lgd")]
    rng = np.random.default_rng(20260930)
    zs, zsum = [], []
    rc_units = _lgd_units("Russia_C"); tb_units = _lgd_units("Tibet")
    E0_lgd = {"Russia_C_LGD": rc_units, "Tibet_LGD": tb_units}
    drop_rc = {17557}                                          # lgd_eligibility_v1_meta drop_v3(Russia_C)
    for key, lab, src in rows_b:
        if src == "v3":
            q = v3[v3.region == key]
            tq = Tg[(Tg.target == key) & (Tg["mode"] == "x") & (Tg.part == "cpu") & Tg.E0.notna()]
            e0 = tq.E0.unique()
            checks[f"fig1_E0_unique_{key}"] = int(len(np.unique(np.round(e0, 9)))) == 1
            E0 = float(e0[0]) if len(e0) else np.nan
            n_src = int(tq.n_src.iloc[0]) if len(tq) else np.nan
        else:
            reg = "Russia_C" if key == "Russia_C_LGD" else "Tibet_LGD"
            qn = tgt[(tgt.region == reg) & (tgt.kind == "lgd_added")]
            qv = v3[(v3.region == reg) & ~v3.loc_id.isin(drop_rc)] if reg == "Russia_C" else v3.iloc[0:0]
            q = pd.concat([qv, qn])
            un = E0_lgd[key]
            E0 = float(un.E0.dropna().unique()[0]); n_src = int(un.n_src.dropna().iloc[0])
            checks[f"fig1_E0_unique_{key}"] = int(len(np.unique(np.round(un.E0.dropna(), 6)))) == 1
            checks[f"fig1_lgd_cells_{key}_vs_unit"] = int(len(q)) == int(un.n_cells.dropna().iloc[0])
        z = np.log(q.y.values) - np.log(q.s.values)
        z = z[np.isfinite(z)]
        k = min(len(z), 400)
        sub = rng.choice(z, k, replace=False) if len(z) > k else z
        zs.append(pd.DataFrame(dict(row=key, label=lab, z=sub)))
        zsum.append(dict(row=key, label=lab, source=src, n_cells=int(len(z)), z_mean=float(np.mean(z)), z_q25=float(np.percentile(z, 25)),
                         z_q50=float(np.median(z)), z_q75=float(np.percentile(z, 75)), E_mean=float(np.exp(np.mean(z))), E0=E0,
                         lnE0=float(np.log(E0)) if np.isfinite(E0) else np.nan, n_src=n_src))
    zsum = pd.DataFrame(zsum)
    # d: 원천 구성(모드 x, v3)
    comp = []
    for t in ["Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland", "Alaska"]:
        src, nbuf = source_pool(v3, t)
        r = v3.region.values[src]
        cnt = pd.Series(r).value_counts()
        n = len(src)
        tq = Tg[(Tg.target == t) & (Tg["mode"] == "x") & (Tg.part == "cpu") & Tg.n_src.notna()]
        row = dict(target=t, n_target=int((v3.region == t).sum()), n_src=n, n_buffer_excluded=nbuf,
                   n_alaska=int(cnt.get("Alaska", 0)), n_lena=int(cnt.get("Lena", 0)), n_canada=int(cnt.get("Canada", 0)),
                   n_other=int(n - cnt.get("Alaska", 0) - cnt.get("Lena", 0) - cnt.get("Canada", 0)),
                   E0=float(tq.E0.iloc[0]), n_src_lg_targets=int(tq.n_src.iloc[0]))
        for k in ("alaska", "lena", "canada", "other"):
            row[f"share_{k}"] = row[f"n_{k}"] / n
        exp = M3_TABLE[t]
        row["m3_match"] = (n, nbuf, row["n_alaska"]) == exp
        row["lg_targets_match"] = n == row["n_src_lg_targets"]
        comp.append(row)
    comp = pd.DataFrame(comp)
    checks["fig1_source_vs_M3_all_match"] = bool(comp.m3_match.all())
    checks["fig1_source_vs_lg_targets_all_match"] = bool(comp.lg_targets_match.all())
    main_share = comp[comp.target.isin(MAIN4)].share_alaska
    checks["info_fig1_alaska_share_main4_pct"] = [round(100 * float(main_share.min()), 1), round(100 * float(main_share.max()), 1)]
    # 레나 지도 영역
    box = json.loads((PROC / "map_lena" / "lena_grid_x25_v1_meta.json").read_text())["grid"]["box"]
    meta = pd.DataFrame([dict(lena_lat0=box["lat0"], lena_lat1=box["lat1"], lena_lon0=box["lon0"], lena_lon1=box["lon1"],
                              n_v3_cells=int(len(v3)), n_lgd_added_drawn=int((tgt.kind == "lgd_added").sum()),
                              n_natl_si=int((tgt.kind == "natl_si").sum()), n_not_drawn_licence=int((tgt.kind == "not_drawn_licence").sum()),
                              natl_eligible_lic=natl_elig_lic)])
    return dict(fig1_blocks=blocks, fig1_not_drawn=nd, fig1_z_points=pd.concat(zs, ignore_index=True), fig1_z_summary=zsum,
                fig1_source=comp, fig1_meta=meta), dict(v3=v3, new=new, tgt=tgt, units=dict(Russia_C=rc_units, Tibet=tb_units))


# ---------------------------------------------------------------- Table 1
LABEL_TYPE_EN = {"점(ABoVE, GPR 다수), CALM 지점 평균 일부": "Point (ABoVE, mostly GPR); some CALM site means",
                 "점(ALLena, 대부분 단년 단일 방문)": "Point (ALLena, mostly single-year visits)",
                 "점(ABoVE), CALM 지점 평균 일부": "Point (ABoVE); some CALM site means",
                 "CALM 지점 다년 평균": "CALM site, multi-year mean"}
ROLE = {"Lena": "P4 (confirmatory)", "Canada": "P4 (confirmatory)", "Russia_W": "P4 (confirmatory)", "Russia_E": "P4 (confirmatory)",
        "Russia_C": "Point estimate only", "Greenland": "Point estimate only", "Alaska": "Reference (not pooled)"}


def table1(checks: dict, ctx: dict) -> dict:
    U = _read(LGW / "lgw_label_units.csv")
    Tg = _read(LGS / "lg_targets.csv")
    Tg = Tg[Tg.part == "cpu"]
    rows = []

    def split_info(t, md):
        q = Tg[(Tg.target == t) & (Tg["mode"] == md)]
        v = q[q.valid.astype(str).str.lower() == "true"]
        use = v if len(v) else q                              # 그린란드: 유효 분할 없음(무효 분할로 점 추정)
        rng_ = lambda c: (int(use[c].min()), int(use[c].max()))  # noqa: E731
        e0 = use.E0.dropna().unique()
        return dict(valid_splits=int(len(v)), A_cells=rng_("n_A"), eval_cells=rng_("n_eval"), eval_blocks=rng_("nb_eval"),
                    E0=float(e0[0]) if len(e0) else np.nan, n_src=int(use.n_src.dropna().iloc[0]) if use.n_src.notna().any() else np.nan)

    order = [("Lena", "main"), ("Canada", "main"), ("Russia_W", "main"), ("Russia_E", "main"), ("Russia_C", "main"), ("Greenland", "main"),
             ("Alaska", "ref")] + [(s, "sub") for s in SUBS]
    for t, grp in order:
        u = U[U.target == t].iloc[0]
        modes = ["x"] if grp != "sub" else ["i", "x"]
        info = {md: split_info(t, md) for md in modes}
        rr = dict(group={"main": "Main region", "ref": "Reference", "sub": "Sub-region"}[grp], target=t,
                  name=ps_name(t), modes=",".join(modes), n_targets=len(modes),
                  role=ROLE.get(t, f"Sub-region of {u.parent} (not independent)"), label_rows=int(u.n_rows), loc_1km=int(u.n_loc_1km),
                  blocks=int(u.n_block), label_type=LABEL_TYPE_EN.get(u.label_type, u.label_type), licence="v3 inventory", edition="v3")
        for md in modes:
            i = info[md]
            rr[f"valid_splits_{md}"] = i["valid_splits"]; rr[f"A_cells_{md}"] = f"{i['A_cells'][0]}–{i['A_cells'][1]}"
            rr[f"eval_cells_{md}"] = f"{i['eval_cells'][0]}–{i['eval_cells'][1]}"; rr[f"eval_blocks_{md}"] = f"{i['eval_blocks'][0]}–{i['eval_blocks'][1]}"
            rr[f"E0_{md}"] = i["E0"]; rr[f"n_src_{md}"] = i["n_src"]
        rows.append(rr)
    checks["table1_n_targets_eq27"] = int(sum(r["n_targets"] for r in rows)) == 27
    # LGD 새 지역(약관 확인분 판)
    new = ctx["tgt"]; v3 = ctx["v3"]
    lic = _read(LGDD / "lgd_eligibility_lic.csv")
    elig = _read(PROC / "lgd_eligibility_v1.csv")
    for key, reg, units, role in (("Russia_C_LGD", "Russia_C", ctx["units"]["Russia_C"], "PE1, PE2 (independent-region check)"),
                                  ("Tibet_LGD", "Tibet_LGD", ctx["units"]["Tibet"], "PE2 (deep regime)")):
        qn = new[(new.region == reg) & (new.kind == "lgd_added")]
        qv = v3[(v3.region == reg) & (v3.loc_id != 17557)] if reg == "Russia_C" else v3.iloc[0:0]
        q = pd.concat([qv, qn])
        v = units[units.valid.astype(bool)]
        meth = new[(new.region == reg) & (new.kind == "lgd_added")].method.value_counts()
        rows.append(dict(group="New region (LGD)", target=key, name=("Russia C (expanded)" if reg == "Russia_C" else "Tibet"), modes="x",
                         n_targets=0, role=role, label_rows=int(len(q)), loc_1km=int(pd.Series(loc1km(q.lat, q.lon)).nunique()),
                         blocks=int(pd.Series(block05(q.lat, q.lon)).nunique()),
                         label_type=("~1 km cell means (" + ", ".join(f"{k} {v}" for k, v in meth.items()) + ")" +
                                     (f"; {len(qv)} v3 CALM cells" if len(qv) else "")),
                         licence=f"verified {len(qn)}; unverified 0", edition="licence-verified (main)",
                         valid_splits_x=int(len(v)), A_cells_x=f"{int(v.n_A.min())}–{int(v.n_A.max())}",
                         eval_cells_x=f"{int(v.n_eval.min())}–{int(v.n_eval.max())}", eval_blocks_x=f"{int(v.nb_eval.min())}–{int(v.nb_eval.max())}",
                         E0_x=float(v.E0.iloc[0]), n_src_x=int(v.n_src.iloc[0])))
        e = elig[elig.spec == ("Russia_C" if reg == "Russia_C" else "Tibet")].iloc[0]
        checks[f"table1_{key}_cells_vs_eligibility"] = int(len(q)) == int(e.n_cells)
    # NAtlantic: 전체 판만(SI), 약관 확인분 판 부적격
    nl = lic[lic.spec == "NAtlantic_lic"].iloc[0]
    ne = elig[elig.spec == "NAtlantic"].iloc[0]
    qn_all = new[(new.region == "NAtlantic") & (new.lgd_role == "target")]
    rows.append(dict(group="New region (LGD)", target="NAtlantic", name="North Atlantic", modes="x", n_targets=0,
                     role="SI only: full edition; ineligible in the licence-verified edition", label_rows=int(ne.n_cells),
                     loc_1km=np.nan, blocks=int(ne.n_blocks), label_type="~1 km cell means; 3 v3 CALM cells",
                     licence=f"verified {int((qn_all.lic_unverified == 0).sum())}; unverified {int((qn_all.lic_unverified == 1).sum())} (licence check pending)",
                     edition="full edition only", valid_splits_x=int(ne.n_valid_splits), A_cells_x="", eval_cells_x=f"{int(ne.n_eval_cells)}",
                     eval_blocks_x=f"{int(ne.n_eval_blocks)}", E0_x=np.nan, n_src_x=int(ne.n_src)))
    checks["info_table1_natl_lic_cells"] = [int(nl.n_cells), int(nl.n_lic_dropped)]
    T = pd.DataFrame(rows)
    # 확충판(L40) 각주 값
    ex = elig[elig.kind == "expanded_L40"][["spec", "target", "n_cells", "n_new_cells", "n_v3_cells"]].copy()
    exl = lic[lic.kind == "expanded_L40"][["lic_of", "n_cells", "n_lic_dropped"]].rename(columns={"lic_of": "spec", "n_cells": "n_cells_lic"})
    ex = ex.merge(exl, on="spec", how="left")
    return dict(table1_rows=T, table1_expanded=ex)


def ps_name(t: str) -> str:
    return {"Russia_W": "Russia W", "Russia_E": "Russia E", "Russia_C": "Russia C", "Alaska": "Alaska (x)"}.get(t, t)


# ---------------------------------------------------------------- Fig 6 (F3 3부, LGF 창 마감 뒤)
LGT = ROOT / "data" / "processed" / "lgt"
LGFD = ROOT / "data" / "processed" / "lgf"
H4D = ROOT / "data" / "processed" / "h4"
FIG6_P4 = "MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"
SUP_CODE = {"지지(물리식보다 오차가 크거나 구별되지 않음)": "A", "지지(우세 근거 없음)": "B", "기각": "R", "판정 불가": "n.d."}
# Fig 6a 고정 행(figure_spec display_items 'Fig 6' panels a rows_fixed, 결과 전 고정). (묶음, 행 키, 선종, 원천 표, 필터, 가설)
FIG6A_LINES = [
    ("baselines", "B:ens", "solid", "lgw_bundle", dict(ab="AB2"), "L29"),
    ("baselines", "P*", "solid", "lgx_tests", dict(test_id="L29", contrast="P0@tddm-P0|n0"), "L29"),
    ("direct_rescale", "catboost_lo", "solid", "lgw_bundle", dict(ab="AB1"), "L1"),
    ("direct_rescale", "catboost", "solid", "lgx_tests", dict(test_id="L30", contrast="D0[catboost]-P0|n0"), "L30"),
    ("direct_rescale", "rf", "solid", "lgx_tests", dict(test_id="L30", contrast="D0[rf]-P0|n0"), "L30"),
    ("direct_rescale", "catboost_tuned", "solid", "lgx_tests", dict(test_id="L30", contrast="D0[catboost_tuned]-P0|n0"), "L30"),
    ("direct_rescale", "F1k", "solid", "lgx_tests", dict(test_id="L11", contrast="F1k-P0|n0"), "L11"),
    ("lgt", "tabpfn", "solid", "lgt_tests", dict(test_id="L34", contrast="D0[T]-P0|n0"), "L34(a)"),
    ("lgt", "catboost_ctx", "solid", "lgt_tests", dict(test_id="L34", contrast="D0[C]-P0|n0"), "L34(a)"),
    ("lgf_f", "tabicl", "solid", "lgf_tests", dict(test_id="LGF-F1", contrast="D0[I]-P0|n0"), "LGF-F1"),
    ("lgf_f", "tabicl", "dashed", "lgf_tests", dict(test_id="LGF-F1", contrast="D0@full[I]-P0|n0"), "LGF-F1"),
    ("lgf_f", "catboost_ctx", "solid", "lgf_tests", dict(test_id="LGF-F1", contrast="D0[C]-P0|n0"), "LGF-F1"),
    ("lgf_f", "catboost_ctx", "dashed", "lgf_tests", dict(test_id="LGF-F1", contrast="D0@full[C]-P0|n0"), "LGF-F1"),
] + [("lgf_n", lr, var, "lgfn_tests", dict(test_id="LGF-N1", contrast=f"D0[{lr}{'*' if var == 'solid' else ''}]-P0|n0"), "LGF-N1")
     for lr in ("mlp", "tabm", "ftt", "realmlp") for var in ("solid", "dashed")]
FIG6_PLATFORM = {"baselines": "Rescale", "direct_rescale": "Rescale", "lgt": "local GPU", "lgf_f": "local GPU", "lgf_n": "local GPU"}
FIG6_AGG = {"lgw_bundle": "h39", "lgx_tests": "h42", "lgt_tests": "h43", "lgf_tests": "h47", "lgfn_tests": "h48"}
FIG6_NORM = ["const", "phys", "cfm", "nflow", "nflow#placebo", "cbq"]          # 6b 정규화기(스펙 순서와 무관한 그리기 순서)
FIG6_REG_EN = [("지지(물리식보다 오차가 크거나 구별되지 않음)", "supported"), ("지지(우세 근거 없음", "supported"), ("지지", "supported"),
               ("P0 보다 우세인 기준선이 있다", "lower-error baseline found"), ("기각", "rejected"), ("판정 불가", "not determinable")]


def _one(T: pd.DataFrame, **flt) -> pd.Series:
    q = T
    for k, v in flt.items():
        q = q[q[k].astype(str) == str(v)]
    if len(q) != 1:
        raise ValueError(f"행 {len(q)}개: {flt}")
    return q.iloc[0]


def reg_en(v) -> str:
    """등록 판정 문구 → 그림 글자(접두 사전 한 곳). 사전에 없는 문구는 멈춘다."""
    for k, e in FIG6_REG_EN:
        if str(v).startswith(k):
            return e
    raise KeyError(f"등록 판정 영문 대응 없음: {v}")


def _lgu_row(r: pd.Series, test: str, n: int, ab: str) -> dict:
    """LGU 판정 행 → 6c 행. % = 100 × 값 / 기준 방법 점수(셀 가중, 같은 풀·n). 기준 점수는 고정값으로 나눈다(Fig 7 의 AB10 과 같은 환산)."""
    ref = float(r.ref_score)
    pct = (lambda v: 100.0 * float(v) / ref)
    g = (lambda c: float(r[c]) if c in r.index and pd.notna(r[c]) else np.nan)
    return dict(test=test, contrast=r.contrast, pool=r.pool, regions=r.get("regions", ""), n=n, metric=r.metric, ci_kind=r.ci_kind,
                delta_cm=g("delta"), ci_lo_cm=g("ci_lo"), ci_hi_cm=g("ci_hi"), delta_beq_cm=g("delta_beq"), ci_lo_beq_cm=g("ci_lo_beq"),
                ci_hi_beq_cm=g("ci_hi_beq"), ref_score=ref, ref_score_beq=g("ref_score_beq"),
                delta_pct=pct(r.delta), ci_lo_pct=pct(r.ci_lo), ci_hi_pct=pct(r.ci_hi),
                ci_lo_1stage_cm=g("ci_lo_1stage"), ci_hi_1stage_cm=g("ci_hi_1stage"),
                ci_lo_1stage_pct=pct(r.ci_lo_1stage), ci_hi_1stage_pct=pct(r.ci_hi_1stage),
                verdict4=r.verdict, verdict=v4(r.verdict), verdict4_1stage=r.verdict_1stage, verdict_1stage=v4(r.verdict_1stage),
                note_cal=r.note_cal if pd.notna(r.note_cal) else "", calib_dependent=str(r.note_cal) == "보정 불확실성 의존",
                eq_state=r.eq_state, eq_state_aux=r.eq_state_aux, half_width=g("half_width"), margin=str(r.margin),
                coverage_pool=g("coverage_pool"), nboot=int(r.nboot), blind=r.blind, ab=ab)


def fig6(checks: dict) -> dict:
    """Fig 6(layout_B, 지도 없음). 값·CI·4분 판정·등록 문구·지지 갈래는 원천 열을 그대로 옮긴다. 새 재표집과 판정 계산은 없다.
    6c 의 % 환산(Δ / 기준 점수)과 6d 의 폭 비(R1 폭 / R0 n = 0 폭)만 산술로 만든다(CI 는 환산만, 비의 CI 는 내지 않는다)."""
    Bd = _read(LGW / "lgw_bundle.csv")
    Tx = _read(LGX / "lgx_tests.csv")
    Tl = _read(LGS / "lg_tests.csv")
    Tt = _read(LGT / "lgt_tests.csv")
    Tf = _read(LGFD / "lgf_tests.csv")
    Tn = _read(LGFD / "lgfn_tests.csv")

    # a: 고정 행(선 단위)
    src = dict(lgw_bundle=Bd[Bd.scope == "MEAN"], lgx_tests=Tx[Tx.scope == "MEAN"], lgt_tests=Tt[Tt.scope == "MEAN"],
               lgf_tests=Tf[Tf.scope == "MEAN"], lgfn_tests=Tn[Tn.scope == "MEAN"])
    rows = []
    for blk, key, var, s, flt, hyp in FIG6A_LINES:
        r = _one(src[s], **flt)
        rows.append(dict(block=blk, key=key, variant=var, platform=FIG6_PLATFORM[blk], hypothesis=hyp, source=f"{s}.csv",
                         aggregator=FIG6_AGG[s], contrast=r.contrast, ab=r.get("ab", "") if s == "lgw_bundle" else "",
                         role=r.get("role", "") if s != "lgw_bundle" else "초록 묶음", target=r.target, pool=r.get("pool", ""),
                         delta=float(r.delta), ci_lo=float(r.ci_lo), ci_hi=float(r.ci_hi), delta_beq=float(r.delta_blockeq),
                         ci_lo_beq=float(r.ci_lo_beq), ci_hi_beq=float(r.ci_hi_beq), verdict4=r.verdict4, verdict=v4(r.verdict4),
                         small_effect=small(r.get("small_note")), ci_dependence=r.get("ci_dependence", "") if pd.notna(r.get("ci_dependence")) else "",
                         run_platform=r.get("platform", "") if pd.notna(r.get("platform", np.nan)) else "",
                         nboot=int(r.nboot) if "nboot" in r.index and pd.notna(r.get("nboot")) else NBOOT_AUX))
    a = pd.DataFrame(rows)
    checks["fig6a_rows_eq_fixed"] = len(a) == len(FIG6A_LINES)
    checks["fig6a_all_P4_mean"] = bool((a.target == FIG6_P4).all())
    checks["fig6a_local_platform_tag"] = bool(a[a.source.isin(["lgf_tests.csv", "lgfn_tests.csv"])].run_platform.str.startswith("local-3090").all())
    # 같은 대비의 두 원천(lgw_bundle h39 와 lgx_tests h42): 점 추정이 같은가(CI 는 재표집이 달라 다를 수 있다)
    l29e = _one(src["lgx_tests"], test_id="L29", contrast="B:ens-P0|n0")
    checks["fig6a_AB2_bundle_vs_L29_delta_abs_diff_cm"] = abs(float(a[a.key == "B:ens"].delta.iloc[0]) - float(l29e.delta))
    checks["info_fig6a_AB2_bundle_vs_L29_ci"] = [[float(a[a.key == "B:ens"].ci_lo.iloc[0]), float(a[a.key == "B:ens"].ci_hi.iloc[0])],
                                                 [float(l29e.ci_lo), float(l29e.ci_hi)]]
    # 로컬 조각 안의 같은 비교 행: LGT 의 D0[C] 와 LGF-F1 병기 D0[C](컨텍스트 10,000행)가 같은 값인가(서술)
    c_lgt = a[(a.block == "lgt") & (a.key == "catboost_ctx")].iloc[0]
    c_lgf = a[(a.block == "lgf_f") & (a.key == "catboost_ctx") & (a.variant == "solid")].iloc[0]
    checks["info_fig6a_catboost_ctx_LGT_vs_LGF_identical"] = bool(np.isclose(c_lgt.delta, c_lgf.delta, atol=1e-9) and np.isclose(c_lgt.ci_lo, c_lgf.ci_lo, atol=1e-9)
                                                                 and np.isclose(c_lgt.ci_hi, c_lgf.ci_hi, atol=1e-9))
    # 등록 판정(가설 수준). 문구는 원천 열, 영문은 reg_en 사전, 지지 갈래는 support_class 열(h47 'A'/'B', h48 학습기별 JSON)
    reg = []
    rv = _one(Tx[Tx.scope == "verdict"], test_id="L29")
    reg.append(dict(hypothesis="L29", block="baselines", source="lgx_tests.csv", verdict_ko=rv.verdict, verdict_en=reg_en(rv.verdict),
                     pstar=re.search(r"P\* = (\S+?)\.", rv.verdict).group(1), support="", support_detail=""))
    rv = _one(Tl[Tl.scope == "verdict"], test_id="L1")
    reg.append(dict(hypothesis="L1", block="direct_rescale", source="lg_tests.csv", verdict_ko=rv.verdict, verdict_en=reg_en(rv.verdict),
                     support="", support_detail=""))
    rv = _one(Tx[Tx.scope == "verdict"], test_id="L30")
    reg.append(dict(hypothesis="L30", block="direct_rescale", source="lgx_tests.csv", verdict_ko=rv.verdict, verdict_en=reg_en(rv.verdict),
                     support="", support_detail=""))
    rv = _one(Tx[Tx.scope == "verdict_aux"], test_id="L11")
    reg.append(dict(hypothesis="L11", block="direct_rescale", source="lgx_tests.csv", verdict_ko=rv.verdict, verdict_en=reg_en(rv.verdict),
                     support="", support_detail=""))
    rv = _one(Tt[Tt.scope == "verdict_aux"], test_id="L34", clause="(a) 직접 예측")
    reg.append(dict(hypothesis="L34(a)", block="lgt", source="lgt_tests.csv", verdict_ko=rv.verdict, verdict_en=reg_en(rv.verdict),
                     support="", support_detail=""))
    rv = _one(Tf[Tf.scope == "verdict"], test_id="LGF-F1")
    sc = str(rv.support_class)
    assert sc in ("A", "B", ""), sc
    reg.append(dict(hypothesis="LGF-F1", block="lgf_f", source="lgf_tests.csv", verdict_ko=rv.verdict, verdict_en=reg_en(rv.verdict),
                     support=sc, support_detail=json.dumps({"tabicl": sc})))
    rv = _one(Tn[Tn.scope == "verdict"], test_id="LGF-N1")
    cls = {k: SUP_CODE[v] for k, v in json.loads(rv.support_class).items()}
    reg.append(dict(hypothesis="LGF-N1", block="lgf_n", source="lgfn_tests.csv", verdict_ko=rv.verdict, verdict_en=reg_en(rv.verdict),
                     support=",".join(sorted(set(cls.values()))), support_detail=json.dumps(cls)))
    for code in ("AB1", "AB2"):
        b = _one(Bd[Bd.scope == "MEAN"], ab=code)
        reg.append(dict(hypothesis=code, block="ab", source="lgw_bundle.csv", verdict_ko=b.abstract_rule, verdict_en="", support="",
                         support_detail="", rule=_rule_code(b.abstract_rule), holm_p=float(b.holm_p)))
    a_reg = pd.DataFrame(reg)
    # 지지 갈래가 행의 4분 판정과 맞는가(A = 판정 대비 모두 열세 또는 동등). 판정은 다시 내지 않고 두 열의 정합만 본다
    f1 = a[(a.hypothesis == "LGF-F1") & (a.key == "tabicl")]
    checks["fig6a_F1_class_consistent_with_rows"] = (sc == "A") == bool(f1.verdict4.isin(["열세", "동등"]).all())
    n1 = a[(a.hypothesis == "LGF-N1") & (a.variant == "solid")]
    checks["fig6a_N1_class_consistent_with_rows"] = all((cls[k] == "A") == (v in ("열세", "동등")) for k, v in zip(n1.key, n1.verdict4))

    # b: 라벨 0 구간(LGU-B, n = 0, 범위 all)
    I = _read(LGU / "lgu_b_intervals.csv")
    q = I[(I.n == 0) & (I.scope == "all") & I.method.isin(FIG6_NORM) & I.target.isin(["MEAN4"] + MAIN4)]
    b = q[["target", "method", "cov10", "cov10_lo1", "cov10_hi1", "cov10_lo2", "cov10_hi2", "cov10_beq", "wid10", "wid10_lo1", "wid10_hi1",
           "wid10_lo2", "wid10_hi2", "wid10_beq", "is10", "in_band", "n_seeds_valid", "n_splits", "n_eval", "n_blocks", "regions", "ci_kind"]].copy()
    b["kind"] = np.where(b.target == "MEAN4", "mean4", "region")
    checks["fig6b_rows_eq_30"] = len(b) == len(FIG6_NORM) * 5
    for m in FIG6_NORM:
        mm = b[(b.method == m) & (b.kind == "mean4")].iloc[0]
        rr = b[(b.method == m) & (b.kind == "region")]
        checks[f"fig6b_{m}_MEAN4_cov_vs_region_mean"] = abs(float(mm.cov10) - float(rr.cov10.mean()))
        checks[f"fig6b_{m}_MEAN4_wid_vs_region_mean_cm"] = abs(float(mm.wid10) - float(rr.wid10.mean()))
    C2 = _read(H4D / "c2_coverage.csv")
    c2 = C2[(C2.test == "label0") & (C2.method == "hier2_cdf")]
    c2m = c2[c2.target.astype(str).str.startswith("MEAN")]
    c2r = c2[c2.target.isin(MAIN4)]
    checks["fig6b_C2_one_mean_row"] = len(c2m) == 1 and len(c2r) == 4
    checks["fig6b_C2_MEAN_cov_vs_region_mean"] = abs(float(c2m.coverage.iloc[0]) - float(c2r.coverage.mean()))
    checks["fig6b_C2_MEAN_wid_vs_region_mean_cm"] = abs(float(c2m.width_cm.iloc[0]) - float(c2r.width_cm.mean()))
    bref = pd.concat([c2m.assign(kind="mean4"), c2r.assign(kind="region")])[
        ["test", "target", "method", "scope", "kind", "coverage", "coverage_lo", "coverage_hi", "coverage_beq", "coverage_cellw", "width_cm",
         "width_cm_lo", "width_cm_hi", "width_cm_cellw", "n_eval", "n_blocks", "regions_all", "ci_flag"]].copy()
    bref["use"] = "참조(재현(비맹검), 규약 다름). LGU 와 CI 를 나란히 비교하지 않는다(LGU 2.7)"

    # c: 구간 점수 대비(%)
    Ub = _read(LGU / "lgu_b_tests.csv")
    Ua = _read(LGU / "lgu_a_tests.csv")
    rc = [_lgu_row(_one(Ub, test="LGU-B2"), "LGU-B2", 0, "")]
    for n in (10, 40):
        rc.append(_lgu_row(_one(Ua, test="LGU-A1", n=str(n)), "LGU-A1", n, "AB10" if n == 10 else ""))
    c = pd.DataFrame(rc)
    b2 = _one(Ub, test="LGU-B2")
    checks["fig6c_B2_margin_is_5pct"] = abs(float(b2.margin) / float(b2.ref_score) - 0.05)
    for n in (10, 40):
        ra = _one(Ua, test="LGU-A1", n=str(n))
        checks[f"fig6c_A1_n{n}_margin_is_5pct"] = abs(float(json.loads(ra.margin)[0]) / float(ra.ref_score) - 0.05)
    ab10 = _one(Bd[Bd.scope == "MEAN"], ab="AB10")
    checks["fig6c_AB10_bundle_vs_A1_n10_delta_abs_diff_cm"] = abs(float(ab10.delta) - float(c[c.ab == "AB10"].delta_cm.iloc[0]))
    checks["fig6c_AB10_verdict_agree"] = str(ab10.verdict4) == str(c[c.ab == "AB10"].verdict4.iloc[0])
    c_reg = pd.DataFrame([dict(test="LGU-B2", verdict4=b2.verdict, verdict=v4(b2.verdict), statement=b2.statement, verdict_3region=b2.verdict_3region),
                          dict(test="LGU-A1", verdict4=_one(Ua, test="LGU-A1", n="10,40").verdict, verdict=v4(_one(Ua, test="LGU-A1", n="10,40").verdict),
                               statement=_one(Ua, test="LGU-A1", n="10,40").statement, verdict_3region=""),
                          dict(test="AB10", verdict4=ab10.verdict4, verdict=v4(ab10.verdict4), statement=ab10.abstract_rule, verdict_3region="",
                               holm_p=float(ab10.holm_p))])

    # d: L25(R1 90 % 구간, 레나·캐나다·Alaska(x) × n 40·160)
    Cf = _read(LGX / "lgx_conformal.csv")
    tg = ["Lena", "Canada", "Alaska"]
    base = Cf[Cf.target.isin(tg) & (Cf["mode"] == "x") & (Cf.level == 90) & (Cf.learner == "catboost_lo") & np.isclose(Cf.lam.astype(float), 0.25)]
    d = base[(base.method == "R1") & base.n.isin([40, 160])][["target", "n", "coverage", "cov_lo", "cov_hi", "width_cm", "width_lo", "width_hi",
                                                             "n_splits", "n_splits_expected", "point_only"]].copy()
    r0 = base[(base.method == "R0") & (base.n == 0)].set_index("target")
    d["width_r0_n0"] = d.target.map(r0.width_cm)
    d["width_ratio"] = d.width_cm / d.width_r0_n0
    L25 = Tx[(Tx.test_id == "L25") & (Tx.scope == "region")].copy()
    L25["t"] = L25.target.str.split("|").str[0]
    L25["n"] = L25.n.astype(int)
    m = d.merge(L25[["t", "n", "coverage", "width_cm", "width_n0_r0", "in_band", "narrower"]].rename(columns={"t": "target"}),
                on=["target", "n"], suffixes=("", "_l25"), validate="one_to_one")
    checks["fig6d_rows_eq_6"] = len(m) == 6
    checks["fig6d_conformal_vs_L25_cov_max_abs_diff"] = float(np.max(np.abs(m.coverage - m.coverage_l25)))
    checks["fig6d_conformal_vs_L25_width_max_abs_diff_cm"] = float(np.max(np.abs(m.width_cm - m.width_cm_l25)))
    checks["fig6d_R0n0_vs_L25_width_n0_max_abs_diff_cm"] = float(np.max(np.abs(m.width_r0_n0 - m.width_n0_r0)))
    d = m.drop(columns=["coverage_l25", "width_cm_l25", "width_n0_r0"])
    d["in_band"] = d.in_band.astype(str).str.lower() == "true"
    d["narrower"] = d.narrower.astype(str).str.lower() == "true"
    lv = _one(Tx[Tx.scope == "verdict_aux"], test_id="L25")
    mt = re.search(r"c1 = (\d+)\(판정한 칸 (\d+)\), c2 = (\d+)\(판정한 칸 (\d+)\), 등록 칸 (\d+)", str(lv.stat))
    d_reg = pd.DataFrame([dict(test_id="L25", verdict_ko=lv.verdict, stat=lv.stat, c1=int(mt.group(1)), c2=int(mt.group(3)), n_cells=int(mt.group(5)),
                               wording="쓰지 않음(WRAPUP 14:20 결정: LGU 10절과 충돌, 커버리지와 폭 비교 값만 서술)")])
    checks["fig6d_c1_eq_in_band_count"] = int(d.in_band.sum()) == int(mt.group(1))
    checks["fig6d_c2_eq_narrower_count"] = int(d.narrower.sum()) == int(mt.group(3))
    return dict(fig6_a=a, fig6_a_registered=a_reg, fig6_b=b, fig6_b_ref=bref, fig6_c=c, fig6_c_registered=c_reg, fig6_d=d, fig6_d_registered=d_reg)


# ================================================================ 기록
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


INPUTS = [LGS / "lg_curve.csv", LGS / "lg_tests.csv", LGS / "lg_minn.csv", LGX / "lgx_curve.csv", LGX / "lgx_tests.csv",
          LGX / "lgx_lg_aux.csv", LGX / "lgx_splitdist.csv", LGX / "lgx_floor.csv", OUTD / "pool_fixed_curve.csv",
          OUTD / "n1_target_summary.csv",
          # 2부
          ROOT / "data/processed/lgw/lgw_bundle.csv", ROOT / "data/processed/lgw/lgw_tests.csv", ROOT / "data/processed/lgw/lgw_scenarios.csv",
          ROOT / "data/processed/lgw/lgw_scenarios_summary.csv", ROOT / "data/processed/lgw/lgw_sc3.csv", ROOT / "data/processed/lgw/lgw_l43.csv",
          ROOT / "data/processed/lgw/lgw_aux4.csv", ROOT / "data/processed/lgw/lgw_splitratio.csv", ROOT / "data/processed/lgw/lgw_label_units.csv",
          LGX / "lgx_distance.csv", ROOT / "data/processed/lgu/lgu_a_tests.csv", ROOT / "data/processed/lgd/lgd_eligibility_lic.csv",
          ROOT / "data/processed/lgd_eligibility_v1.csv", ROOT / "data/processed/fidelity_base_v3.csv", ROOT / "data/processed/fidelity_base_v4.csv",
          ROOT / "data/processed/fidelity_base_v4_labels.csv", LGS / "lg_targets.csv"]
# 3부(Fig 6, LGF 창 마감 2026-10-02 05:01 뒤)
FIG6_INPUTS = [ROOT / "data/processed/lgw/lgw_bundle.csv", LGX / "lgx_tests.csv", LGS / "lg_tests.csv", LGT / "lgt_tests.csv",
               LGFD / "lgf_tests.csv", LGFD / "lgfn_tests.csv", LGU / "lgu_b_intervals.csv", LGU / "lgu_b_tests.csv",
               LGU / "lgu_a_tests.csv", H4D / "c2_coverage.csv", LGX / "lgx_conformal.csv"]


def _failed(checks: dict, tol: float = 1e-6) -> dict:
    ck = {k: v for k, v in checks.items() if not k.startswith("info_")}
    bad = {k: v for k, v in ck.items() if isinstance(v, float) and not isinstance(v, bool) and "ci_lo" not in k and v > tol}
    bad.update({k: v for k, v in ck.items() if isinstance(v, (list, dict)) and len(v)})
    bad.update({k: v for k, v in ck.items() if isinstance(v, (bool, np.bool_)) and not bool(v)})
    return bad


def main_fig6():
    """3부: Fig 6 표만 만든다. 다른 그림의 표와 v2_data_meta.json 은 건드리지 않고 v2_data_fig6_meta.json 을 따로 쓴다."""
    OUTD.mkdir(parents=True, exist_ok=True)
    checks: dict = {}
    tables = fig6(checks)
    for k, df in tables.items():
        df.to_csv(OUTD / f"{k}.csv", index=False)
    bad = _failed(checks)
    meta = dict(module="scripts/4_visualization/paper/v2_data.py --only fig6", created=_dt.datetime.now().isoformat(timespec="seconds"),
                inputs=[dict(path=str(p.relative_to(ROOT)), sha256=sha256(p)) for p in FIG6_INPUTS],
                outputs=sorted(f"{k}.csv" for k in tables), checks=checks, checks_failed=bad,
                rules=dict(values="원천 열 그대로. 새 재표집·판정 없음. 6c 의 % 는 Δ/기준 점수, 6d 의 폭 비는 R1 폭/R0(n = 0) 폭(산술, CI 없음)",
                           rows_fixed="6a 행은 figure_spec display_items 'Fig 6' rows_fixed(결과 전 고정)를 그대로 쓴다. P* 행은 L29_pstar 조건(성립)",
                           platform="Rescale 행(lgw_bundle h39, lgx_tests h42)과 로컬 GPU 행(lgt h43, lgf h47, lgfn h48) 사이의 차는 계산하지 않는다(LGF 2.2)",
                           nboot=dict(a=NBOOT_AUX, b_c_lgu=1000, d=NBOOT_AUX),
                           not_opened="data/processed/wf/*, results/rescale_wf3/*(진행 중 실험)"))
    (OUTD / "v2_data_fig6_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    print(f"[v2_data --only fig6] 표 {len(tables)}개 → {OUTD.relative_to(ROOT)}, 대조 실패 {len(bad)}")
    if bad:
        print(json.dumps(bad, ensure_ascii=False, indent=1, default=str))
        sys.exit(1)


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    checks: dict = {}
    tables = {}
    for f in (fig2, fig3, fig4):
        tables.update(f(checks))
    # 2부(F3 part 2)
    ab = bundle()
    tables.update(ab)
    tables.update(part2_updates(checks, ab["ab_bundle"]))
    tables.update(fig5(checks))
    tables.update(fig7(checks, ab["ab_bundle"], ab["ab_bundle_regions"]))
    t1, ctx = fig1(checks)
    tables.update(t1)
    tables.update(table1(checks, ctx))
    tables.update(fig6(checks))                                # 3부(LGF 창 마감 뒤)
    for k, df in tables.items():
        df.to_csv(OUTD / f"{k}.csv", index=False)
    tol = 1e-6
    ck = {k: v for k, v in checks.items() if not k.startswith("info_")}
    bad = {k: v for k, v in ck.items() if isinstance(v, float) and not isinstance(v, bool) and "ci_lo" not in k and v > tol}
    bad.update({k: v for k, v in ck.items() if isinstance(v, (list, dict)) and len(v)})
    bad.update({k: v for k, v in ck.items() if isinstance(v, (bool, np.bool_)) and not bool(v)})
    meta = dict(module="scripts/4_visualization/paper/v2_data.py", created=_dt.datetime.now().isoformat(timespec="seconds"),
                inputs=[dict(path=str(p.relative_to(ROOT)), sha256=sha256(p)) for p in dict.fromkeys(INPUTS + FIG6_INPUTS)],
                outputs=sorted(f"{k}.csv" for k in tables), checks=checks, checks_failed=bad,
                rules=dict(values="원천 열 그대로. 새 재표집·판정 없음", derived_points="E2(레나·캐나다) P* 선과 P1* 점은 lgx_curve 지역 값의 등가중 평균(점 추정)",
                           nboot=dict(lg_curve=NBOOT_CURVE, lgx=NBOOT_AUX, pool_fixed_curve=NBOOT_AUX),
                           not_opened="1·2부: lgt_*, lgf_*, lgfn_* 를 열지 않았다(LGF 창 마감 전). lgw_*·lgd_* 는 판정 기록(J5, J7) 뒤 2부에서 연다. "
                                      "3부(Fig 6, 창 마감 뒤): lgt·lgf·lgfn·lgu_b·c2_coverage·lgx_conformal 을 연다"))
    (OUTD / "v2_data_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    print(f"[v2_data] 표 {len(tables)}개 → {OUTD.relative_to(ROOT)}, 대조 실패 {len(bad)}")
    if bad:
        print(json.dumps(bad, ensure_ascii=False, indent=1))
        sys.exit(1)


if __name__ == "__main__":
    if "--only" in sys.argv[1:]:
        _k = sys.argv[sys.argv.index("--only") + 1] if sys.argv.index("--only") + 1 < len(sys.argv) else ""
        if _k != "fig6":
            sys.exit(f"--only 는 fig6 만 받는다: {_k!r}")
        main_fig6()
    else:
        main()
