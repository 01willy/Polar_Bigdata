"""H25·H26 라벨 예산 프로토콜 — 대상 13개(주 4지역 + 하위 지역 9) × 라벨 n × 선택 규칙 × 단계(E 수축 → 증강 → 잔차 ML, 대상 행 가중 α). CPU.

계획 docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md §2–§3. 전신 m1_sparse_label_curve.py(S-F)·h24_obs_design.py(H24).

대상: 주 4지역(레나·캐나다·러시아 W·E) + 하위 지역(알래스카 k=6·캐나다 k=3·레나 k=2; 블록 중심 k-means, 라벨 미사용) → data/processed/h3/subregions.csv
원천(LORO): 대상 셀을 뺀 전체 F4 라벨 셀에서 대상 경계 --buffer-km 이내 셀 제외. 앵커 E0 = 원천 최소제곱.
분할: 대상 블록 A/B 2분할(split 0..K-1). 풀 = A, 채점 = B(실측·CCI·토양 도일 유효).
규칙: random · kmedoid(표준화 x25 k-means 중심 최근접 셀).  n 격자: |A| 미만 값만.
단계(누적): S1 E 수축(κ=10, n=0 → E0) · S2 S1 + 유사라벨 증강 직접 CatBoost(A 풀에 E_n·√TDD 유사라벨 r=10, 라벨 셀은 실측 대체)
          · S3 S1 앵커 + 잔차 CatBoost(원천 잔차(E0 앵커) ∪ 대상 n개 잔차(E_n 앵커), 대상 행 가중 α, λ)
n=0: physics(E0), S2(순수 증강 = 공변량만 레시피), S3(원천 잔차만). 라벨 전량(allA): 같은 단계를 A 전체로.
산출 data/processed/h3/h25_{runs,curve,breakeven,targets}.csv, h25_meta.json
실행(ROOT, CPU): python3 scripts/3_deep_learning/h25_label_budget.py --workers 6 [--smoke]
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

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="", help="쉼표 목록(기본: 주 4지역 + 하위 지역 전부)")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--n-grid", default="3,5,10,20,40,80,160,320")
ap.add_argument("--reps", type=int, default=20, help="해석적 단계 반복")
ap.add_argument("--reps-cb", type=int, default=5, help="CatBoost 단계 반복(앞에서부터)")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--lams", default="0.25,0.5")
ap.add_argument("--alphas", default="1,10,100")
ap.add_argument("--r", type=float, default=10.0)
ap.add_argument("--kappa", type=float, default=10.0)
ap.add_argument("--buffer-km", type=float, default=100.0)
ap.add_argument("--k-sub", default="Alaska:6,Canada:3,Lena:2")
ap.add_argument("--rules", default="random,kmedoid")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tag", default="h25")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--summarize-only", action="store_true", help="저장된 runs 로 집계만 다시(대상 지표는 경량 재계산)")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"; OUT = PROC / "h3"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
N_GRID = [int(v) for v in args.n_grid.split(",")] if not args.smoke else [3, 20]
REPS = args.reps if not args.smoke else 2
REPS_CB = args.reps_cb if not args.smoke else 1
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
LAMS = [float(v) for v in args.lams.split(",")]
ALPHAS = [float(v) for v in args.alphas.split(",")]
RULES = args.rules.split(",")
MAIN4 = ["Lena", "Canada", "Russia_W", "Russia_E"]

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)


# ---------------------------------------------------------------- 하위 지역(라벨 미사용: 블록 중심 좌표 k-means)
def make_subregions(df):
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
        order = np.argsort([-bt.lat.values[lab == c].mean() for c in range(k)])           # 북→남 이름 순서
        name_of = {c: f"{reg[:2].upper()}-{r + 1}" for r, c in enumerate(order)}
        b2s = dict(zip(bt.block.values, [name_of[c] for c in lab]))
        for i in idx:
            sub[i] = b2s[df.block.values[i]]
        for c in range(k):
            m = lab == c
            rows.append(dict(subregion=name_of[c], parent=reg, n_blocks=int(m.sum()), n_cells=int(bt.n.values[m].sum()),
                             lat=float(bt.lat.values[m].mean()), lon=float(bt.lon.values[m].mean())))
    return sub, pd.DataFrame(rows)


DF["sub"], SUBS = make_subregions(DF)
SUBS.to_csv(OUT / "subregions.csv", index=False)
TARGETS = args.targets.split(",") if args.targets else MAIN4 + sorted(SUBS.subregion.tolist())
if args.smoke:
    TARGETS = ["Lena", "AL-1", "Russia_W"]
print(f"[data] {len(DF):,}셀 · 대상 {len(TARGETS)}: {TARGETS}", flush=True)


def target_idx(t):
    return np.where(DF.macro.values == t)[0] if t in MAIN4 else np.where(DF["sub"].values == t)[0]


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def cb_fit_predict(Xtr, ytr, Xte, seed, w=None):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr, sample_weight=w)
    return np.asarray(m.predict(Xte), float)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmse_beq(y, p, codes, nb):
    e2 = (y - p) ** 2; s = np.bincount(codes, e2, minlength=nb); c = np.bincount(codes, minlength=nb)
    return float(np.mean(np.sqrt(s[c > 0] / c[c > 0])))


def select(rule, n, rng, Z):
    if rule == "random":
        return rng.choice(len(Z), n, replace=False)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.randint(1 << 30))).fit(Z)
    sel = []
    for c in km.cluster_centers_:
        d = ((Z - c) ** 2).sum(1); d[sel] = np.inf; sel.append(int(np.argmin(d)))
    return np.array(sel)


def hetero_metrics(t_idx, src_idx):
    """라벨 미사용 이질성 지표 + 라벨 기반 진단 지표."""
    Xt = DF.iloc[t_idx][FEATS].values.astype(float); Xs = DF.iloc[src_idx][FEATS].values.astype(float)
    mt, ms = np.nanmean(Xt, 0), np.nanmean(Xs, 0); sd_s = np.nanstd(Xs, 0) + 1e-9
    smd = float(np.nanmean(np.abs(mt - ms) / sd_s))
    var_ratio = float(np.nanmean(np.nanvar(Xt, 0) / (np.nanvar(Xs, 0) + 1e-9)))
    d = DF.iloc[t_idx]; y, s = d.y.values, d.s.values
    ratio = y / np.where(s > 0, s, np.nan)
    g = pd.DataFrame(dict(b=d.block.values, r=ratio)).groupby("b").r
    Eb = g.mean(); nb = g.size()
    within = float(np.nanmean(g.std()[nb >= 3])) if (nb >= 3).any() else np.nan
    return dict(n_cells=int(len(t_idx)), n_blocks=int(d.block.nunique()), smd_x25=smd, var_ratio_x25=var_ratio,
                E_own=ls_E(y, s), E_block_cv=float(np.nanstd(Eb[nb >= 3]) / np.nanmean(Eb[nb >= 3])) if (nb >= 3).sum() >= 2 else np.nan,
                within_block_sd_ratio=within, y_sd=float(np.nanstd(y)), y_mean=float(np.nanmean(y)))


# ---------------------------------------------------------------- 작업 단위(대상 × 분할)
def run_task(t, sp):
    t0 = time.time()
    t_idx = target_idx(t)
    parent = t if t in MAIN4 else str(SUBS.set_index("subregion").loc[t, "parent"])
    src_idx = np.where(np.isfinite(DF.y.values))[0]
    src_idx = src_idx[~np.isin(src_idx, t_idx)]
    if args.buffer_km > 0:                                    # 대상 경계 버퍼 제외(좌표만)
        la, lo = DF.lat.values, DF.lon.values
        keep = np.ones(len(src_idx), bool)
        tl, tn = la[t_idx], lo[t_idx]
        for j, i in enumerate(src_idx):
            if abs(la[i] - tl).min() < 1.0:                   # 위도 1° 안에서만 거리 계산(속도)
                if haversine_km(la[i], lo[i], tl, tn).min() < args.buffer_km:
                    keep[j] = False
        src_idx = src_idx[keep]
    src = DF.iloc[src_idx]
    E0 = ls_E(src.y.values, src.s.values)
    X_src = src[FEATS].values.astype(np.float32); y_src = src.y.values; s_src = src.s.values
    r_src = y_src - E0 * s_src; ok = np.isfinite(r_src); XR_src, R_src = X_src[ok], r_src[ok]
    A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
    evB = B_idx[eval_mask(DF.iloc[B_idx])]
    A, B = DF.iloc[A_idx], DF.iloc[evB]
    nA = len(A)
    yA, sA, XA = A.y.values, A.s.values, A[FEATS].values.astype(np.float32)
    yB, sB, XB = B.y.values, B.s.values, B[FEATS].values.astype(np.float32)
    _, codesB = np.unique(B.block.values, return_inverse=True); nbB = int(codesB.max()) + 1
    Xa = XA.astype(float); med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xa = np.where(np.isnan(Xa), med, Xa); Z = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
    rows, n_fit = [], 0
    base = dict(target=t, parent=parent, split=sp, n_A=nA, n_blocks_A=int(A.block.nunique()), n_eval=len(evB), n_blocks_eval=nbB,
                n_src=int(len(src_idx)), E0=E0)

    def add(rule, stage, n, rep, seed, lam, alpha, pred, E_used, scope="n"):
        rows.append(dict(**base, rule=rule, stage=stage, scope=scope, n=int(n), rep=int(rep), seed=int(seed), lam=float(lam), alpha=float(alpha),
                         rmse_cm=rmse(yB, pred), rmse_beq_cm=rmse_beq(yB, pred, codesB, nbB), bias_cm=float(np.mean(pred - yB)), E_used=float(E_used)))

    n_ps = int(round(args.r * len(y_src)))
    PS_SEL = {seed: np.random.RandomState(seed).choice(nA, n_ps, replace=True) for seed in SEEDS}

    def stage2(sel, E_n, seed):
        ps = PS_SEL[seed]; yps = E_n * sA[ps]
        if sel is not None and len(sel):
            m = np.isin(ps, sel); yps = yps.copy(); yps[m] = yA[ps[m]]
        return cb_fit_predict(np.vstack([X_src, XA[ps]]), np.concatenate([y_src, yps]), XB, seed)

    def stage3(sel, E_n, seed, alpha):
        if sel is None or len(sel) == 0:
            return cb_fit_predict(XR_src, R_src, XB, seed)
        r_n = yA[sel] - E_n * sA[sel]
        Xtr = np.vstack([XR_src, XA[sel]]); ytr = np.concatenate([R_src, r_n])
        w = np.concatenate([np.ones(len(R_src)), np.full(len(sel), alpha)])
        return cb_fit_predict(Xtr, ytr, XB, seed, w=w)

    # n = 0
    add("none", "physics", 0, -1, -1, 0.0, 1.0, E0 * sB, E0)
    for seed in SEEDS:
        p2 = stage2(None, E0, seed); n_fit += 1; add("none", "S2", 0, -1, seed, 1.0, 1.0, p2, E0)
        g = stage3(None, E0, seed, 1.0); n_fit += 1
        for lam in LAMS:
            add("none", "S3", 0, -1, seed, lam, 1.0, E0 * sB + lam * g, E0)

    def evaluate(sel, n, rep, rule, scope, do_cb):
        nonlocal n_fit
        E_ls = ls_E(yA[sel], sA[sel]); E_n = (n * E_ls + args.kappa * E0) / (n + args.kappa)
        add(rule, "S1", n, rep, -1, 0.0, 1.0, E_n * sB, E_n, scope)
        add(rule, "S1refit", n, rep, -1, 0.0, 1.0, E_ls * sB, E_ls, scope)
        if not do_cb:
            return
        for seed in SEEDS:
            p2 = stage2(sel, E_n, seed); n_fit += 1; add(rule, "S2", n, rep, seed, 1.0, 1.0, p2, E_n, scope)
            for alpha in ALPHAS:
                g = stage3(sel, E_n, seed, alpha); n_fit += 1
                for lam in LAMS:
                    add(rule, "S3", n, rep, seed, lam, alpha, E_n * sB + lam * g, E_n, scope)

    evaluate(np.arange(nA), nA, -1, "all", "allA", do_cb=True)
    for n in N_GRID:
        if n >= nA:
            continue
        for rule in RULES:
            for rep in range(REPS):
                rng = np.random.RandomState(7_919 * sp + 1_009 * n + 31 * RULES.index(rule) + rep)
                sel = select(rule, n, rng, Z)
                evaluate(sel, n, rep, rule, "n", do_cb=rep < REPS_CB)
    het = hetero_metrics(t_idx, src_idx)
    return rows, dict(target=t, parent=parent, split=sp, n_A=nA, n_eval=len(evB), n_src=int(len(src_idx)), E0=E0, n_fit=n_fit,
                      elapsed_s=round(time.time() - t0, 1), **het)


# ---------------------------------------------------------------- 집계
def summarize(runs):
    r = runs[runs.stage != "physics"].copy()
    phys = runs[runs.stage == "physics"].set_index(["target", "split"]).rmse_cm
    r["phys"] = [phys.loc[(t, s)] for t, s in zip(r.target, r.split)]
    r["d_phys"] = r.rmse_cm - r.phys; r["win"] = (r.rmse_cm < r.phys).astype(float)
    g = r.groupby(["target", "parent", "split", "rule", "stage", "scope", "lam", "alpha", "n", "rep"], as_index=False).agg(
        rmse_cm=("rmse_cm", "mean"), rmse_beq_cm=("rmse_beq_cm", "mean"), d_phys=("d_phys", "mean"), win=("win", "mean"))
    curve = g.groupby(["target", "parent", "rule", "stage", "scope", "lam", "alpha", "n"], as_index=False).agg(
        rmse_mean=("rmse_cm", "mean"), rmse_sd=("rmse_cm", "std"), rmse_beq_mean=("rmse_beq_cm", "mean"),
        d_phys_mean=("d_phys", "mean"), win_rate=("win", "mean"), n_runs=("rmse_cm", "size"), n_splits=("split", "nunique"))
    # Δ 대 물리식의 (분할, 반복) 짝지음 부트스트랩 CI
    los, his = [], []
    for _, row in curve.iterrows():
        sub = g[(g.target == row.target) & (g.rule == row.rule) & (g.stage == row.stage) & (g.scope == row.scope) & (g.lam == row.lam) & (g.alpha == row.alpha) & (g.n == row.n)]
        d = sub.d_phys.values
        if len(d) >= 3:
            rng = np.random.RandomState(1); bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(1000)])
            los.append(float(np.percentile(bs, 2.5))); his.append(float(np.percentile(bs, 97.5)))
        else:
            los.append(np.nan); his.append(np.nan)
    curve["d_phys_lo"], curve["d_phys_hi"] = los, his
    # 회복률·손익분기: 같은 (rule 무관 allA) 단계의 라벨 전량 값을 기준으로
    allA = curve[curve.scope == "allA"].groupby(["target", "stage", "lam", "alpha"]).rmse_mean.mean()   # 분할마다 |A| 가 달라 n 으로 나뉜 행을 평균
    def rec(row):
        key = (row.target, row.stage if row.stage != "S1refit" else "S1", row.lam, row.alpha)
        if key not in allA.index:
            return np.nan
        full = allA.loc[key]; phys_v = row.rmse_mean - row.d_phys_mean
        den = phys_v - full
        return float((phys_v - row.rmse_mean) / den) if abs(den) > 1e-6 else np.nan
    curve["recovery"] = curve.apply(rec, axis=1)
    be = []
    for (t, rule, stage, lam, alpha), sub in curve[curve.scope == "n"].groupby(["target", "rule", "stage", "lam", "alpha"]):
        sub = sub[sub.n_splits == sub.n_splits.max()].sort_values("n")            # 분할 일부에만 있는 n(|A| 불균형 대상)은 제외
        ok = sub[(sub.recovery >= 0.5) & (sub.win_rate >= 0.75)]
        ok2 = sub[(sub.d_phys_hi < 0)]
        be.append(dict(target=t, parent=sub.parent.iloc[0], rule=rule, stage=stage, lam=lam, alpha=alpha,
                       breakeven_n_rec50=int(ok.n.iloc[0]) if len(ok) else -1, breakeven_n_ci=int(ok2.n.iloc[0]) if len(ok2) else -1,
                       n_max=int(sub.n.max()), best_n=int(sub.loc[sub.rmse_mean.idxmin(), "n"]), best_d_phys=float(sub.d_phys_mean.min())))
    return curve, pd.DataFrame(be)


def light_meta(t, sp):
    """CatBoost 없이 대상 지표만(요약 재계산용)."""
    t_idx = target_idx(t); parent = t if t in MAIN4 else str(SUBS.set_index("subregion").loc[t, "parent"])
    src_idx = np.where(np.isfinite(DF.y.values))[0]; src_idx = src_idx[~np.isin(src_idx, t_idx)]
    if args.buffer_km > 0:
        la, lo = DF.lat.values, DF.lon.values; keep = np.ones(len(src_idx), bool); tl, tn = la[t_idx], lo[t_idx]
        for j, i in enumerate(src_idx):
            if abs(la[i] - tl).min() < 1.0 and haversine_km(la[i], lo[i], tl, tn).min() < args.buffer_km:
                keep[j] = False
        src_idx = src_idx[keep]
    src = DF.iloc[src_idx]; A_idx, B_idx = half_split_blocks(DF, t_idx, sp); evB = B_idx[eval_mask(DF.iloc[B_idx])]
    return dict(target=t, parent=parent, split=sp, n_A=len(A_idx), n_eval=len(evB), n_src=int(len(src_idx)), E0=ls_E(src.y.values, src.s.values), n_fit=0, elapsed_s=0.0,
                **hetero_metrics(t_idx, src_idx))


def main():
    t0 = time.time()
    tasks = [(t, sp) for t in TARGETS for sp in SPLITS]
    rows, metas = [], []
    if args.summarize_only:
        tag = args.tag + ("_smoke" if args.smoke else "")
        runs = pd.read_csv(OUT / f"{tag}_runs.csv")
        metas = [light_meta(t, sp) for t, sp in tasks if ((runs.target == t) & (runs.split == sp)).any()]
        curve, be = summarize(runs)
        curve.to_csv(OUT / f"{tag}_curve.csv", index=False); be.to_csv(OUT / f"{tag}_breakeven.csv", index=False)
        pd.DataFrame(metas).to_csv(OUT / f"{tag}_targets.csv", index=False)
        (OUT / f"{tag}_meta.json").write_text(json.dumps(dict(stage="H25/H26", plan="docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md", targets=TARGETS, splits=SPLITS,
            n_grid=N_GRID, reps=REPS, reps_cb=REPS_CB, seeds=SEEDS, lams=LAMS, alphas=ALPHAS, r=args.r, kappa=args.kappa, buffer_km=args.buffer_km, k_sub=args.k_sub,
            rules=RULES, summarize_only=True, n_rows=int(len(runs))), ensure_ascii=False, indent=1, default=float))
        show = be[(be.rule == "kmedoid") & (be.lam.isin([0.0, 0.25])) & (be.alpha == 1.0)]
        print(show[["target", "stage", "breakeven_n_rec50", "breakeven_n_ci", "best_n", "best_d_phys"]].to_string(index=False)); print(f"summarized · 행 {len(runs)}")
        return
    def _done(r, m):
        rows.extend(r); metas.append(m)
        print(f"  [{m['target']}|{m['split']}] 적합 {m['n_fit']} · A {m['n_A']} · 원천 {m['n_src']} · E0 {m['E0']:.3f} · {m['elapsed_s']}s · 누적 {time.time()-t0:.0f}s", flush=True)
    if args.workers <= 1:
        for t, sp in tasks:
            _done(*run_task(t, sp))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:   # fork 후 OpenMP 충돌 회피
            futs = {ex.submit(run_task, t, sp): (t, sp) for t, sp in tasks}
            for f in as_completed(futs):
                _done(*f.result())
    runs = pd.DataFrame(rows); tag = args.tag + ("_smoke" if args.smoke else "")
    runs.to_csv(OUT / f"{tag}_runs.csv", index=False)
    curve, be = summarize(runs)
    curve.to_csv(OUT / f"{tag}_curve.csv", index=False); be.to_csv(OUT / f"{tag}_breakeven.csv", index=False)
    pd.DataFrame(metas).to_csv(OUT / f"{tag}_targets.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                                                 # noqa: BLE001
        commit = "NA"
    (OUT / f"{tag}_meta.json").write_text(json.dumps(dict(stage="H25/H26", plan="docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md", targets=TARGETS,
        splits=SPLITS, n_grid=N_GRID, reps=REPS, reps_cb=REPS_CB, seeds=SEEDS, lams=LAMS, alphas=ALPHAS, r=args.r, kappa=args.kappa,
        buffer_km=args.buffer_km, k_sub=args.k_sub, rules=RULES, git_commit=commit, elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1, default=float))
    show = be[(be.rule == "kmedoid") & (be.lam.isin([0.0, 0.25])) & (be.alpha == 1.0)]
    print(show[["target", "stage", "breakeven_n_rec50", "breakeven_n_ci", "best_n", "best_d_phys"]].to_string(index=False))
    print(f"saved {tag}_* · 행 {len(runs)} · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
