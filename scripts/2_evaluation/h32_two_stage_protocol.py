"""H32 (Track B1) 2단계 프로토콜 확인적 검정. 대표점 3개로 E비를 진단하고 분기한다. CPU.

계획 docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 Track B1, 가설 F4.
규칙(결과 열람 전 고정): 단계 1 = kmedoid 대표점 3개로 Ê3 = 최소제곱, r̂ = |log(Ê3/E0)|. 단계 2 = r̂ ≥ τ 이면 "E 수축(κ=10) 종료",
r̂ < τ 이면 "E0 고정 + 대상 n개 라벨 잔차 CatBoost(λ=0.25)". τ 후보 {0.10, 0.15, 0.20}.
τ 선택(계획서 §B1, 2026-09-26 검토 반영): 대상 t 의 τ 는 t 를 제외한 나머지 대상의 고정 τ 요약에서 같은 n 기준으로 고른다
  (기준 순서: 악화 대상 수(d_phys_mean > 0) 최소, 평균 Δ 최소, |τ − 0.15| 최소). 이렇게 모은 행이 protocol@loto 이며 F4 주 행이다.
  고정 τ 행(protocol@0.1 등)은 민감도(role = sensitivity)로 표기한다. τ_t 는 t 의 자료와 무관하므로 CI 는 해당 τ 행의 블록 CI 를 그대로 쓴다.
라벨 예산 규약(2026-09-26 검토 반영): 예산 n 안에서 진단한다. n=3 은 진단 추출 sel3 를 그대로 쓴다. n > 3 은 sel3 를 포함하는
  중첩 추출(select_nested: n-중심 KMeans 의 중심 중 sel3 각 점에 가장 가까운 중심 3개를 sel3 에 배정하고, 나머지 n−3 중심은 최근접 미선택 셀)로
  n 개를 뽑는다. 따라서 모든 방법(protocol·always_*·oracle_*)이 같은 n 개 라벨만 쓴다. n=10 추출은 h25 의 비중첩 kmedoid-10 추출과 다르다.
비교(같은 추출·같은 seed 로 짝지음): always_shrink_resid(h25 S3 와 같은 적합 정의, n=10 추출은 위 중첩 규약), always_E0_resid, shrink_only, refit_only, oracle_branch(두 분기 중 사후 최선), oracle_any(세 후보 중 최선, 참고), E0_resid_src(n=0 원천 잔차), physics(E0).
자료·분할·원천·버퍼·채점 셀·seed 규약은 h25_label_budget.py 와 동일(RULES 순서 random, kmedoid → kmedoid 인덱스 1).
탐색적(사전 등록 아님, 결과 열람 후 추가): r_cv3(3개 라벨 log 비율 표준편차)·r_agree3(3개 라벨의 E 방향 일치율), explore_gate_agree = r̂ ≥ τ 이고 방향 일치할 때만 수축.
  요약·프로토콜 표의 analysis 열에 prereg(사전 등록)·post_hoc_explore(사후 탐색)·reference(오라클·n=0 참조)를 표기한다.
CI 규약(h4_common, 2026-09-26 개정): 물리식 대비 Δ 의 95 % CI(d_phys_lo/hi)는 분할 안 채점 블록 부트스트랩(모든 방법·반복·seed 에 같은 재표집 인덱스)으로 구한다.
  기존 (분할, 반복) 행 재표집 CI 는 d_phys_lo_rep/hi_rep 로 보존한다. 실행 단위(대상, 분할, 방법, τ, n, 반복, seed)마다 블록 SSE·셀 수를 <tag>_blocksse.npz 에 저장한다.
  키 = (method, tau, n, rep, seed), τ 없음 = -1.0, 반복·seed 없음 = -1. 오라클 행은 사후 선택된 예측의 블록 SSE 를 저장한다(선택 기준은 풀링 RMSE).
  n_worse_ci(프로토콜 표) = 블록 CI 하한 > 0 인 대상 수(새 정의), n_worse_ci_rep = 기존 반복 CI 기준.
산출 data/processed/h4/b1_runs.csv, b1_summary.csv, b1_protocol.csv, b1_meta.json, b1_blocksse.npz
실행: python3 scripts/2_evaluation/h32_two_stage_protocol.py --workers 3 --tag b1 [--smoke]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET                                                                     # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks                        # noqa: E402
from polar.m1_ext import haversine_km                                                                # noqa: E402
from polar.h4_common import BlockStore, save_stores, load_stores, stores_for_target, boot_delta_blocks, rep_boot_ci  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--reps-cb", type=int, default=5)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--ns", default="3,10")
ap.add_argument("--taus", default="0.10,0.15,0.20")
ap.add_argument("--lam", type=float, default=0.25)
ap.add_argument("--kappa", type=float, default=10.0)
ap.add_argument("--buffer-km", type=float, default=100.0)
ap.add_argument("--k-sub", default="Alaska:6,Canada:3,Lena:2")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--tag", default="b1")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"; OUT = PROC / "h4"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
REPS = args.reps if not args.smoke else 3
REPS_CB = args.reps_cb if not args.smoke else 1
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
NS = [int(v) for v in args.ns.split(",")]
TAUS = [float(v) for v in args.taus.split(",")]
RULES = ["random", "kmedoid"]                      # h25 seed 규약 유지(kmedoid 인덱스 1)
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
POST_HOC = ("explore_gate_agree",)                  # 사후 탐색(사전 등록 아님)
REFERENCE = ("oracle_branch", "oracle_any", "E0_resid_src")
NO_TAU = -1.0


def analysis_label(method):
    m = method.split("@")[0]
    return "post_hoc_explore" if m in POST_HOC else ("reference" if m in REFERENCE else "prereg")

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)


def make_subregions(df):                            # h25 와 동일(결정적)
    from sklearn.cluster import KMeans
    sub = np.array(["" for _ in range(len(df))], dtype=object)
    rows = []
    for spec in args.k_sub.split(","):
        reg, k = spec.split(":"); k = int(k)
        idx = np.where(df.macro.values == reg)[0]
        bt = df.iloc[idx].groupby("block").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size")).reset_index()
        X = np.c_[bt.lat.values, bt.lon.values * np.cos(np.radians(bt.lat.values))]
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X, sample_weight=np.sqrt(bt.n.values))
        lab = km.labels_
        order = np.argsort([-bt.lat.values[lab == c].mean() for c in range(k)])
        name_of = {c: f"{reg[:2].upper()}-{r + 1}" for r, c in enumerate(order)}
        b2s = dict(zip(bt.block.values, [name_of[c] for c in lab]))
        for i in idx:
            sub[i] = b2s[df.block.values[i]]
        for c in range(k):
            rows.append(dict(subregion=name_of[c], parent=reg))
    return sub, pd.DataFrame(rows)


DF["sub"], SUBS = make_subregions(DF)
TARGETS = args.targets.split(",") if args.targets else MAIN4 + sorted(SUBS.subregion.tolist())
TARGETS = [t for t in TARGETS if t != "CA-1"]          # |A| < 6 (분석 제외, h25 와 동일)
if args.smoke:
    TARGETS = ["Russia_W", "Canada", "AL-1"]
print(f"[data] {len(DF):,}셀 · 대상 {len(TARGETS)}: {TARGETS}", flush=True)


def target_idx(t):
    return np.where(DF.macro.values == t)[0] if t in MAIN4 else np.where(DF["sub"].values == t)[0]


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def cb_fit_predict(Xtr, ytr, Xte, seed):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed, verbose=0, allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), float)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmse_beq(y, p, codes, nb):
    e2 = (y - p) ** 2; s = np.bincount(codes, e2, minlength=nb); c = np.bincount(codes, minlength=nb)
    return float(np.mean(np.sqrt(s[c > 0] / c[c > 0])))


def select_kmedoid(n, rng, Z):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.randint(1 << 30))).fit(Z)
    sel = []
    for c in km.cluster_centers_:
        d = ((Z - c) ** 2).sum(1); d[sel] = np.inf; sel.append(int(np.argmin(d)))
    return np.array(sel)


def select_nested(n, rng, Z, base_sel):
    """base_sel(진단 추출)을 포함하는 n 개 중첩 k-중심 추출. 예산 n 안에서 진단과 적합이 같은 라벨을 쓰게 한다."""
    from sklearn.cluster import KMeans
    base_sel = [int(i) for i in base_sel]
    if n <= len(base_sel):
        return np.array(base_sel[:n])
    C = KMeans(n_clusters=n, n_init=3, random_state=int(rng.randint(1 << 30))).fit(Z).cluster_centers_
    D = ((Z[base_sel][:, None, :] - C[None, :, :]) ** 2).sum(-1)          # (len(base), n) 거리
    used_c = set()
    for i in np.argsort(D.min(1)):                                        # 기존 점마다 가장 가까운 미배정 중심을 탐욕 배정
        order = np.argsort(D[i]); c = next(int(c) for c in order if int(c) not in used_c); used_c.add(c)
    sel = list(base_sel)
    for c in range(n):
        if c in used_c:
            continue
        d = ((Z - C[c]) ** 2).sum(1); d[sel] = np.inf; sel.append(int(np.argmin(d)))
    return np.array(sel)


def run_task(t, sp):
    t0 = time.time()
    t_idx = target_idx(t)
    parent = t if t in MAIN4 else str(SUBS.set_index("subregion").loc[t, "parent"])
    src_idx = np.where(np.isfinite(DF.y.values))[0]; src_idx = src_idx[~np.isin(src_idx, t_idx)]
    if args.buffer_km > 0:
        la, lo = DF.lat.values, DF.lon.values; keep = np.ones(len(src_idx), bool); tl, tn = la[t_idx], lo[t_idx]
        for j, i in enumerate(src_idx):
            if abs(la[i] - tl).min() < 1.0 and haversine_km(la[i], lo[i], tl, tn).min() < args.buffer_km:
                keep[j] = False
        src_idx = src_idx[keep]
    src = DF.iloc[src_idx]; E0 = ls_E(src.y.values, src.s.values)
    X_src = src[FEATS].values.astype(np.float32); y_src = src.y.values; s_src = src.s.values
    r_src = y_src - E0 * s_src; ok = np.isfinite(r_src); XR_src, R_src = X_src[ok], r_src[ok]
    A_idx, B_idx = half_split_blocks(DF, t_idx, sp); evB = B_idx[eval_mask(DF.iloc[B_idx])]
    A, B = DF.iloc[A_idx], DF.iloc[evB]; nA = len(A)
    yA, sA, XA = A.y.values, A.s.values, A[FEATS].values.astype(np.float32)
    yB, sB, XB = B.y.values, B.s.values, B[FEATS].values.astype(np.float32)
    _, codesB = np.unique(B.block.values, return_inverse=True); nbB = int(codesB.max()) + 1
    Xa = XA.astype(float); med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xa = np.where(np.isnan(Xa), med, Xa); Z = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
    base = dict(target=t, parent=parent, split=sp, n_A=nA, n_eval=len(evB), n_blocks_eval=nbB, n_src=int(len(src_idx)), E0=E0)
    rows = []; n_fit = 0
    st = BlockStore(t, sp, B.block.values, meta=dict(n_A=nA, n_eval=int(len(evB)), E0=E0))

    def put(method, tau, n, rep, seed, p):                   # 블록 SSE 저장(키 = method, tau, n, rep, seed)
        st.add((method, NO_TAU if tau is None or not np.isfinite(tau) else float(tau), int(n), int(rep), int(seed)), yB, p)
    phys = E0 * sB
    put("physics", None, 0, -1, -1, phys)
    rows.append(dict(**base, n=0, rep=-1, seed=-1, method="physics", tau=np.nan, r_hat=np.nan, E_used=E0, rmse_cm=rmse(yB, phys), rmse_beq_cm=rmse_beq(yB, phys, codesB, nbB)))
    g_src = {seed: cb_fit_predict(XR_src, R_src, XB, seed) for seed in SEEDS}; n_fit += len(SEEDS)     # 원천 잔차만(n=0 참조)
    for seed in SEEDS:
        p = E0 * sB + args.lam * g_src[seed]; put("E0_resid_src", None, 0, -1, seed, p)
        rows.append(dict(**base, n=0, rep=-1, seed=seed, method="E0_resid_src", tau=np.nan, r_hat=np.nan, E_used=E0, rmse_cm=rmse(yB, p), rmse_beq_cm=rmse_beq(yB, p, codesB, nbB)))
    for rep in range(REPS):
        # 단계 1: 진단(n=3 kmedoid, h25 seed 규약)
        rng3 = np.random.RandomState(7_919 * sp + 1_009 * 3 + 31 * RULES.index("kmedoid") + rep)
        sel3 = select_kmedoid(3, rng3, Z) if nA > 3 else np.arange(nA)
        E3 = ls_E(yA[sel3], sA[sel3]); r_hat = abs(np.log(E3 / E0)) if (np.isfinite(E3) and E3 > 0) else np.inf
        rat3 = yA[sel3] / np.where(sA[sel3] > 0, sA[sel3], np.nan)                                   # 탐색적: 3개 라벨의 개별 E 비율 산포(사전 등록 아님)
        r_cv3 = float(np.nanstd(np.log(rat3[rat3 > 0]))) if (rat3 > 0).sum() >= 2 else np.nan
        r_agree3 = float(np.mean(np.sign(np.log(rat3[rat3 > 0] / E0)) == np.sign(np.log(E3 / E0)))) if (rat3 > 0).sum() >= 2 else np.nan
        for n in NS:
            if n >= nA:
                continue
            if n == 3:
                sel = sel3
            else:
                rng = np.random.RandomState(7_919 * sp + 1_009 * n + 31 * RULES.index("kmedoid") + rep); sel = select_nested(n, rng, Z, sel3)
                assert len(set(sel.tolist())) == n and set(sel3.tolist()) <= set(sel.tolist())     # 예산 n 안 진단(sel3 ⊂ sel)
            E_ls = ls_E(yA[sel], sA[sel]); E_n = (n * E_ls + args.kappa * E0) / (n + args.kappa)
            p_sh = E_n * sB
            put("shrink_only", None, n, rep, -1, p_sh); put("refit_only", None, n, rep, -1, E_ls * sB)
            rows.append(dict(**base, n=n, rep=rep, seed=-1, method="shrink_only", tau=np.nan, r_hat=r_hat, r_cv3=r_cv3, r_agree3=r_agree3, E_used=E_n, rmse_cm=rmse(yB, p_sh), rmse_beq_cm=rmse_beq(yB, p_sh, codesB, nbB)))
            rows.append(dict(**base, n=n, rep=rep, seed=-1, method="refit_only", tau=np.nan, r_hat=r_hat, r_cv3=r_cv3, r_agree3=r_agree3, E_used=E_ls, rmse_cm=rmse(yB, E_ls * sB), rmse_beq_cm=rmse_beq(yB, E_ls * sB, codesB, nbB)))
            if rep >= REPS_CB:
                continue
            for seed in SEEDS:
                # E0 고정 + 대상 n 잔차(E0 앵커)
                r_n0 = yA[sel] - E0 * sA[sel]
                g0 = cb_fit_predict(np.vstack([XR_src, XA[sel]]), np.concatenate([R_src, r_n0]), XB, seed); n_fit += 1
                p_e0 = E0 * sB + args.lam * g0
                # 수축 앵커 + 대상 n 잔차(E_n 앵커) = h25 S3(α=1)
                r_n1 = yA[sel] - E_n * sA[sel]
                g1 = cb_fit_predict(np.vstack([XR_src, XA[sel]]), np.concatenate([R_src, r_n1]), XB, seed); n_fit += 1
                p_sr = E_n * sB + args.lam * g1
                cand = {"always_E0_resid": p_e0, "always_shrink_resid": p_sr}
                for name, p in cand.items():
                    put(name, None, n, rep, seed, p)
                    rows.append(dict(**base, n=n, rep=rep, seed=seed, method=name, tau=np.nan, r_hat=r_hat, r_cv3=r_cv3, r_agree3=r_agree3, E_used=E0 if name == "always_E0_resid" else E_n, rmse_cm=rmse(yB, p), rmse_beq_cm=rmse_beq(yB, p, codesB, nbB)))
                for tau in TAUS:
                    p = p_sh if r_hat >= tau else p_e0
                    pg = p_sh if (r_hat >= tau and r_agree3 == 1.0) else p_e0
                    put("explore_gate_agree", tau, n, rep, seed, pg); put("protocol", tau, n, rep, seed, p)
                    rows.append(dict(**base, n=n, rep=rep, seed=seed, method="explore_gate_agree", tau=tau, r_hat=r_hat, r_cv3=r_cv3, r_agree3=r_agree3, E_used=np.nan, rmse_cm=rmse(yB, pg), rmse_beq_cm=rmse_beq(yB, pg, codesB, nbB)))
                    rows.append(dict(**base, n=n, rep=rep, seed=seed, method="protocol", tau=tau, r_hat=r_hat, r_cv3=r_cv3, r_agree3=r_agree3, E_used=E_n if r_hat >= tau else E0, rmse_cm=rmse(yB, p), rmse_beq_cm=rmse_beq(yB, p, codesB, nbB)))
                ro = min(rmse(yB, p_sh), rmse(yB, p_e0))
                put("oracle_branch", None, n, rep, seed, p_sh if rmse(yB, p_sh) <= rmse(yB, p_e0) else p_e0)
                rows.append(dict(**base, n=n, rep=rep, seed=seed, method="oracle_branch", tau=np.nan, r_hat=r_hat, r_cv3=r_cv3, r_agree3=r_agree3, E_used=np.nan, rmse_cm=ro, rmse_beq_cm=np.nan))
                ro3 = min(ro, rmse(yB, p_sr))                     # 세 후보(수축 단독·E0+잔차·수축+잔차) 사후 최선(참고)
                put("oracle_any", None, n, rep, seed, min((p_sh, p_e0, p_sr), key=lambda q: rmse(yB, q)))
                rows.append(dict(**base, n=n, rep=rep, seed=seed, method="oracle_any", tau=np.nan, r_hat=r_hat, r_cv3=r_cv3, r_agree3=r_agree3, E_used=np.nan, rmse_cm=ro3, rmse_beq_cm=np.nan))
    return rows, dict(target=t, split=sp, n_A=nA, n_eval=len(evB), E0=E0, n_fit=n_fit, elapsed_s=round(time.time() - t0, 1)), st


def _key_fn(method, tau, n):
    tv = NO_TAU if not np.isfinite(tau) else float(tau)
    return lambda k: k[0] == method and abs(k[1] - tv) < 1e-9 and k[2] == int(n)


def summarize(runs, stores):
    """대상 × 방법(τ) × n 요약. d_phys_lo/hi = 블록 부트스트랩 CI(주), d_phys_lo_rep/hi_rep = (분할, 반복) 행 재표집 CI(보조)."""
    phys = runs[runs.method == "physics"].set_index(["target", "split"]).rmse_cm
    r = runs[runs.method != "physics"].copy(); r["d_phys"] = r.rmse_cm - [phys.loc[(t, s)] for t, s in zip(r.target, r.split)]
    r["key"] = r.method + np.where(r.tau.notna(), "@" + r.tau.round(2).astype(str), "")
    by_t = {t: stores_for_target(stores, t) for t in r.target.unique()}
    is_phys = lambda k: k[0] == "physics"                                                              # noqa: E731
    out = []
    for (t, key, n), s in r.groupby(["target", "key", "n"]):
        d = s.d_phys.values
        lo_r, hi_r = rep_boot_ci(d, nboot=1000, seed=1)
        tau = float(s.tau.iloc[0]) if s.tau.notna().any() else np.nan
        b = boot_delta_blocks(by_t[t], _key_fn(s.method.iloc[0], tau, n), is_phys, nboot=args.nboot, seed=0, rep_fn=lambda k: k[3])
        out.append(dict(target=t, parent=s.parent.iloc[0], method=key, analysis=analysis_label(key), n=n, d_phys_mean=float(d.mean()),
                        d_phys_lo=b["ci_lo"], d_phys_hi=b["ci_hi"], d_phys_lo_rep=lo_r, d_phys_hi_rep=hi_r,
                        d_phys_block=b["delta"], p_boot=b["p_boot"], split_win=b["split_win"], rep_win=b["rep_win"],
                        d_phys_beq=b["delta_beq"], d_phys_lo_beq=b["ci_lo_beq"], d_phys_hi_beq=b["ci_hi_beq"],
                        n_blocks_min=int(min(b["n_blocks"])) if b["n_blocks"] else 0, ci_flag=b["ci_flag"],
                        rmse_mean=float(s.rmse_cm.mean()), win_rate=float((d < 0).mean()), r_hat_mean=float(s.r_hat.mean()) if s.r_hat.notna().any() else np.nan,
                        frac_branch_shrink=float((s.r_hat >= float(key.split("@")[1])).mean()) if "@" in key else np.nan, n_runs=int(len(d)), n_splits=int(s.split.nunique())))
    S = pd.DataFrame(out)
    S["tau_loto"] = np.nan
    S["role"] = np.where(S.method.str.startswith("protocol@"), "sensitivity", "comparator")
    S = pd.concat([S, loto_rows(S)], ignore_index=True)
    # 프로토콜 표(F4): 방법별 악화 대상 수·평균·최악, 오라클 대비 차이. n_worse_ci = 블록 CI 하한 > 0(새 정의), n_worse_ci_rep = 기존 반복 CI
    prot = []
    for (key, n), s in S.groupby(["method", "n"]):
        orc = S[(S.method == "oracle_branch") & (S.n == n)].set_index("target").d_phys_mean
        prot.append(dict(method=key, analysis=analysis_label(key), role=str(s.role.iloc[0]), n=n, n_targets=len(s), n_worse=int((s.d_phys_mean > 0).sum()),
                         n_worse_ci=int((s.d_phys_lo > 0).sum()), n_worse_ci_rep=int((s.d_phys_lo_rep > 0).sum()),
                         n_better_ci=int((s.d_phys_hi < 0).sum()), n_flag_blocks=int((s.ci_flag != "").sum()), mean_d=float(s.d_phys_mean.mean()),
                         worst_d=float(s.d_phys_mean.max()), worst_target=str(s.loc[s.d_phys_mean.idxmax(), "target"]), gap_to_oracle=float((s.set_index("target").d_phys_mean - orc.reindex(s.target)).mean()),
                         tau_loto_counts=json.dumps({f"{v:.2f}": int(c) for v, c in s.tau_loto.value_counts().sort_index().items()}) if s.tau_loto.notna().any() else ""))
    return S, pd.DataFrame(prot)


def loto_tau(S, t, n):
    """대상 t 를 제외한 대상들의 고정 τ 요약으로 τ 를 고른다(기준: 악화 수, 평균 Δ, |τ − 0.15| 순)."""
    best = None
    for tau in TAUS:
        o = S[(S.method == f"protocol@{round(tau, 2)}") & (S.n == n) & (S.target != t)]
        if len(o) == 0:
            continue
        crit = (int((o.d_phys_mean > 0).sum()), float(o.d_phys_mean.mean()), abs(tau - 0.15))
        if best is None or crit < best[0]:
            best = (crit, tau)
    return np.nan if best is None else best[1]


def loto_rows(S):
    """protocol@loto 행(F4 주 행). 대상별 τ_t 의 고정 τ 요약 행을 복사하고 tau_loto 를 기록한다."""
    out = []
    for (t, n), _ in S[S.method.str.startswith("protocol@")].groupby(["target", "n"]):
        tau = loto_tau(S, t, n)
        if not np.isfinite(tau):
            continue
        row = S[(S.method == f"protocol@{round(tau, 2)}") & (S.n == n) & (S.target == t)].iloc[0].to_dict()
        row.update(method="protocol@loto", analysis="prereg", role="primary", tau_loto=float(tau))
        out.append(row)
    return pd.DataFrame(out)


def main():
    t0 = time.time(); tasks = [(t, sp) for t in TARGETS for sp in SPLITS]; rows, metas, sts = [], [], []
    def _done(r, m, st):
        rows.extend(r); metas.append(m); sts.append(st); print(f"  [{m['target']}|{m['split']}] 적합 {m['n_fit']} · A {m['n_A']} · E0 {m['E0']:.3f} · {m['elapsed_s']}s · 누적 {time.time()-t0:.0f}s", flush=True)
    if args.workers <= 1:
        for t, sp in tasks:
            _done(*run_task(t, sp))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = {ex.submit(run_task, t, sp): (t, sp) for t, sp in tasks}
            for f in as_completed(futs):
                _done(*f.result())
    runs = pd.DataFrame(rows); tag = args.tag + ("_smoke" if args.smoke else "")
    runs.to_csv(OUT / f"{tag}_runs.csv", index=False)
    bpath = save_stores(sts, OUT / f"{tag}_blocksse.npz"); stores = load_stores(bpath)             # 저장본으로 요약(왕복 검증 겸)
    S, P = summarize(runs, stores); S.to_csv(OUT / f"{tag}_summary.csv", index=False); P.to_csv(OUT / f"{tag}_protocol.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                                                 # noqa: BLE001
        commit = "NA"
    (OUT / f"{tag}_meta.json").write_text(json.dumps(dict(stage="B1/H32", plan="docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md", targets=TARGETS, splits=SPLITS, reps=REPS, reps_cb=REPS_CB,
        seeds=SEEDS, ns=NS, taus=TAUS, lam=args.lam, kappa=args.kappa, buffer_km=args.buffer_km, git_commit=commit,
        ci_rule="block bootstrap within split (h4_common.boot_delta_blocks), shared indices across methods/reps/seeds; d_phys_lo_rep/hi_rep = (split, rep) row bootstrap",
        nboot=args.nboot, blocksse=f"{tag}_blocksse.npz", key_format="(method, tau[-1=none], n, rep, seed)", post_hoc_methods=list(POST_HOC),
        tau_rule="protocol@loto = per-target tau chosen on other targets at same n: min n_worse(d_phys_mean>0), then min mean_d, then min |tau-0.15|; fixed-tau rows = sensitivity",
        label_budget="within-budget diagnosis: n=3 uses sel3; n>3 uses select_nested (sel3 subset of sel_n, n labels total) for all methods", reference_methods=list(REFERENCE), n_rows=int(len(runs)), elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1, default=float))
    pd.set_option("display.width", 250); print(P.round(2).to_string(index=False)); print(f"saved {tag}_* · 행 {len(runs)} · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
