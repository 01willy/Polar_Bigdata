"""H24 관측 설계: 어떤 라벨 n개가 지역을 대표하는가 — 라벨 선택 규칙 × n × 추정량 (CPU, S-F 규약 계승).

설계 docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md §6. 전신 scripts/3_deep_learning/m1_sparse_label_curve.py(S-F, 무작위 선택만).

요인 = 라벨 선택 규칙(라벨 미사용: 공변량·좌표·물리 예측만). 추정량·n·지역·분할·앵커(stefan) 고정.
  실전: 대상 레나·캐나다·러시아 W·E. 학습 실측 = 알래스카(E_AK 앵커). 풀 = 대상 A블록, 채점 = B블록(분할 3).
  샌드박스: 대상 알래스카. 학습 실측 = 주 라벨 5지역+그린란드(앵커 E0 = 지역 등가중 E, 블록<8 지역은 κ=10 수축). 풀 = 알래스카 A블록.
  규칙: random · block_rr(블록 순환) · kmedoid(x25 표준화 k-means 중심 최근접 셀) · farthest(k-center 탐욕) · stefan_strat(앵커 예측 분위 층화)
        · geo_strat(좌표 k-means) · calm_sites(알래스카만: CALM 사이트 셀에서 무작위)
  추정량(S-F 와 동일): refit(E 최소제곱) · shrink_k10 · resid(앵커 E_AK 고정 + catboost_lo 잔차, λ) · both(shrink 앵커 + 잔차)
  채점: 셀 가중 RMSE(주)·블록 등가중(보조). Δ(규칙 − random) 은 같은 (분할, 반복) 짝지음. 승률 = 물리식(n=0) 대비.
산출 data/processed/h2/h24_{runs,curve,tests}.csv, h24_meta.json
실행(ROOT, CPU 전용): python3 scripts/3_deep_learning/h24_obs_design.py --workers 4 [--smoke]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, TRANSFER_MAIN                                                     # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks, fit_coefs             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="Lena,Canada,Russia_W,Russia_E,Alaska")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--n-grid", default="3,5,10,20,40,80")
ap.add_argument("--n-grid-ak", default="3,5,10,20,50,100,200")
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--reps-cb", type=int, default=10)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--lam", type=float, default=0.25)
ap.add_argument("--kappa", type=float, default=10.0)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--strategies", default="random,block_rr,kmedoid,farthest,stefan_strat,geo_strat,calm_sites")
ap.add_argument("--tag", default="h24")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"
OUT = PROC / "h2"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
TARGETS = args.targets.split(",")
SPLITS = list(range(args.splits)) if not args.smoke else [0]
N_GRID = [int(v) for v in args.n_grid.split(",")]
N_GRID_AK = [int(v) for v in args.n_grid_ak.split(",")]
REPS = args.reps if not args.smoke else 2
REPS_CB = args.reps_cb if not args.smoke else 1
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
STRATS = args.strategies.split(",")
if args.smoke:
    N_GRID, N_GRID_AK = [3, 10], [5, 50]

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float)
DF["y"] = DF[TARGET].values.astype(float)


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


# 학습 실측(원천)과 앵커 계수: 실전 = 알래스카, 샌드박스 = 나머지 주 라벨 지역
AK = DF[DF.macro == "Alaska"]
E_AK = float(fit_coefs(AK)["E"])
SRC_MAIN = [r for r in TRANSFER_MAIN]
E_reg = {}
for r in SRC_MAIN:
    d = DF[DF.macro == r]; E = ls_E(d.y.values, d.s.values)
    big = [rr for rr in SRC_MAIN if DF[DF.macro == rr].block.nunique() >= 8]
    E_reg[r] = E
E_big_mean = float(np.mean([E_reg[r] for r in SRC_MAIN if DF[DF.macro == r].block.nunique() >= 8]))
for r in SRC_MAIN:
    if DF[DF.macro == r].block.nunique() < 8:
        n = int((DF.macro == r).sum()); E_reg[r] = (n * E_reg[r] + args.kappa * E_big_mean) / (n + args.kappa)
E0_AK_TARGET = float(np.mean(list(E_reg.values())))                      # 샌드박스 앵커(지역 등가중)
print(f"[data] {len(DF):,}셀 · E_AK {E_AK:.4f} · 샌드박스 E0 {E0_AK_TARGET:.4f} ({ {k: round(v, 3) for k, v in E_reg.items()} }) · 규칙 {STRATS} · 분할 {SPLITS} · 반복 {REPS}/{REPS_CB}", flush=True)


def cb_fit_predict(Xtr, ytr, Xte, seed):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=args.threads)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte), float)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def rmse_beq(y, p, codes, nb):
    e2 = (y - p) ** 2
    s = np.bincount(codes, e2, minlength=nb); c = np.bincount(codes, minlength=nb)
    return float(np.mean(np.sqrt(s[c > 0] / c[c > 0])))


# ---------------------------------------------------------------- 선택 규칙(라벨 미사용)
def select(strategy, n, rng, Z, coords, anchor_pred, blocks, calm_mask):
    N = len(Z)
    if strategy == "random":
        return rng.choice(N, n, replace=False)
    if strategy == "block_rr":
        ub = np.unique(blocks); order = rng.permutation(ub)
        pools = {b: list(rng.permutation(np.where(blocks == b)[0])) for b in ub}
        sel = []
        while len(sel) < n:
            for b in order:
                if pools[b]:
                    sel.append(pools[b].pop())
                    if len(sel) >= n:
                        break
        return np.array(sel)
    if strategy in ("kmedoid", "geo_strat"):
        from sklearn.cluster import KMeans
        X = Z if strategy == "kmedoid" else coords
        km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.randint(1 << 30))).fit(X)
        sel = []
        for c in km.cluster_centers_:
            d = ((X - c) ** 2).sum(1); d[sel] = np.inf
            sel.append(int(np.argmin(d)))
        return np.array(sel)
    if strategy == "farthest":
        sel = [int(rng.randint(N))]
        dmin = ((Z - Z[sel[0]]) ** 2).sum(1)
        while len(sel) < n:
            j = int(np.argmax(dmin)); sel.append(j)
            dmin = np.minimum(dmin, ((Z - Z[j]) ** 2).sum(1))
        return np.array(sel)
    if strategy == "stefan_strat":
        order = np.argsort(anchor_pred); bins = np.array_split(order, n)
        return np.array([int(rng.choice(b)) for b in bins if len(b)])
    if strategy == "calm_sites":
        cand = np.where(calm_mask)[0]
        if len(cand) < n:
            return None
        return rng.choice(cand, n, replace=False)
    raise ValueError(strategy)


# ---------------------------------------------------------------- 작업 단위 (대상 × 분할)
def run_task(tg, sp):
    t0 = time.time()
    sandbox = tg == "Alaska"
    src = DF[DF.macro.isin(SRC_MAIN)] if sandbox else AK
    if sandbox:
        anc_src = np.concatenate([E_reg[r] * src[src.macro == r].s.values for r in SRC_MAIN])   # 지역별 E 로 잔차(원천 지역 라벨 사용)
        src = pd.concat([src[src.macro == r] for r in SRC_MAIN])
        E0 = E0_AK_TARGET
    else:
        E0 = E_AK
        anc_src = E0 * src.s.values
    X_src = src[FEATS].values.astype(np.float32); y_src = src.y.values
    r_src = y_src - anc_src; ok = np.isfinite(r_src)
    XR_src, R_src = X_src[ok], r_src[ok]
    t_idx = np.where(DF.macro.values == tg)[0]
    A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
    evB = B_idx[eval_mask(DF.iloc[B_idx])]
    A, B = DF.iloc[A_idx], DF.iloc[evB]
    nA = len(A)
    yA, sA, XA = A.y.values, A.s.values, A[FEATS].values.astype(np.float32)
    yB, sB, XB = B.y.values, B.s.values, B[FEATS].values.astype(np.float32)
    _, codesB = np.unique(B.block.values, return_inverse=True); nbB = int(codesB.max()) + 1
    # 규칙 입력(라벨 미사용): 표준화 x25(A 통계), 투영 좌표, 앵커 예측, 블록, CALM 마스크
    Xa = XA.astype(float); med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xa = np.where(np.isnan(Xa), med, Xa); mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Z = (Xa - mu) / sd
    coords = np.c_[A.lat.values, A.lon.values * np.cos(np.radians(A.lat.values))]
    anchor_A = E0 * sA
    calm_mask = (A.region.values == "United States (Alaska)")
    anc_B = E0 * sB
    grid = N_GRID_AK if sandbox else N_GRID
    rows, n_fit = [], 0
    base = dict(target=tg, split=sp, n_A=nA, n_blocks_A=int(A.block.nunique()), n_eval=len(evB), n_blocks_eval=nbB, E0=E0)

    def add(strategy, method, n, rep, seed, lam, pred, E_used):
        rows.append(dict(**base, strategy=strategy, method=method, n=int(n), rep=int(rep), seed=int(seed), lam=float(lam),
                         rmse_cm=rmse(yB, pred), rmse_beq_cm=rmse_beq(yB, pred, codesB, nbB), bias_cm=float(np.mean(pred - yB)),
                         E_used=float(E_used)))

    add("none", "physics", 0, -1, -1, 0.0, anc_B, E0)
    for seed in SEEDS:
        g = cb_fit_predict(XR_src, R_src, XB, seed); n_fit += 1
        add("none", "resid", 0, -1, seed, args.lam, anc_B + args.lam * g, E0)
    for n in grid:
        if n >= nA:
            continue
        for strategy in STRATS:
            if strategy == "calm_sites" and not sandbox:
                continue
            for rep in range(REPS):
                rng = np.random.RandomState(7_919 * sp + 1_009 * n + 31 * STRATS.index(strategy) + rep)
                sel = select(strategy, n, rng, Z, coords, anchor_A, A.block.values, calm_mask)
                if sel is None or len(sel) < n:
                    continue
                y_n, s_n, X_n = yA[sel], sA[sel], XA[sel]
                E_ls = ls_E(y_n, s_n)
                E_sh = (n * E_ls + args.kappa * E0) / (n + args.kappa)
                add(strategy, "refit", n, rep, -1, 0.0, E_ls * sB, E_ls)
                add(strategy, "shrink_k10", n, rep, -1, 0.0, E_sh * sB, E_sh)
                if rep < REPS_CB:
                    r_fix = y_n - E0 * s_n; r_ad = y_n - E_sh * s_n
                    Xtr = np.vstack([XR_src, X_n])
                    for seed in SEEDS:
                        g = cb_fit_predict(Xtr, np.concatenate([R_src, r_fix]), XB, seed)
                        g2 = cb_fit_predict(Xtr, np.concatenate([R_src, r_ad]), XB, seed); n_fit += 2
                        add(strategy, "resid", n, rep, seed, args.lam, anc_B + args.lam * g, E0)
                        add(strategy, "both", n, rep, seed, args.lam, E_sh * sB + args.lam * g2, E_sh)
    return rows, dict(target=tg, split=sp, n_A=nA, n_eval=len(evB), n_fit=n_fit, elapsed_s=round(time.time() - t0, 1))


# ---------------------------------------------------------------- 집계·검정
def summarize(runs):
    r = runs[runs.method != "physics"].copy()
    phys = runs[runs.method == "physics"].set_index(["target", "split"]).rmse_cm
    r["phys"] = [phys.loc[(t, s)] for t, s in zip(r.target, r.split)]
    r["win"] = (r.rmse_cm < r.phys).astype(float)
    r["d_phys"] = r.rmse_cm - r.phys
    # seed 평균(반복 단위)
    g = r.groupby(["target", "split", "strategy", "method", "lam", "n", "rep"], as_index=False).agg(
        rmse_cm=("rmse_cm", "mean"), rmse_beq_cm=("rmse_beq_cm", "mean"), win=("win", "mean"), d_phys=("d_phys", "mean"))
    curve = g.groupby(["target", "strategy", "method", "lam", "n"], as_index=False).agg(
        rmse_mean=("rmse_cm", "mean"), rmse_sd=("rmse_cm", "std"), rmse_p95=("rmse_cm", lambda x: float(np.percentile(x, 95))),
        rmse_beq_mean=("rmse_beq_cm", "mean"), win_rate=("win", "mean"), d_phys_mean=("d_phys", "mean"), n_runs=("rmse_cm", "size"))
    # Δ(규칙 − random), (분할, 반복) 짝지음, 반복 부트스트랩 CI
    tests = []
    rnd = g[g.strategy == "random"].set_index(["target", "split", "method", "lam", "n", "rep"]).rmse_cm
    for (t, st, m, lam, n), sub in g[g.strategy != "random"].groupby(["target", "strategy", "method", "lam", "n"]):
        key = list(zip(sub.target, sub.split, sub.method, sub.lam, sub.n, sub.rep))
        ok = [k in rnd.index for k in key]
        if sum(ok) < 3:
            continue
        d = sub.rmse_cm.values[ok] - rnd.loc[[k for k, o in zip(key, ok) if o]].values
        rng = np.random.RandomState(1)
        bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(1000)])
        tests.append(dict(target=t, strategy=st, method=m, lam=lam, n=n, n_pairs=len(d), delta_vs_random=float(d.mean()),
                          ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)), win_vs_random=float((d < 0).mean()),
                          d_phys_mean=float(sub.d_phys.values.mean()), d_phys_random=float((sub.d_phys.values[ok] - d).mean())))
    tests = pd.DataFrame(tests, columns=["target", "strategy", "method", "lam", "n", "n_pairs", "delta_vs_random", "ci_lo", "ci_hi",
                                         "win_vs_random", "d_phys_mean", "d_phys_random"])
    # AB4 지역 등가중 요약(같은 규칙·추정량·n 이 4지역 모두에 있을 때)
    ab4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
    agg = []
    for (st, m, lam, n), sub in tests[tests.target.isin(ab4)].groupby(["strategy", "method", "lam", "n"]):
        if sub.target.nunique() == len(ab4):
            agg.append(dict(target="MEAN[AB4]", strategy=st, method=m, lam=lam, n=n, n_pairs=int(sub.n_pairs.sum()),
                            delta_vs_random=float(sub.delta_vs_random.mean()), ci_lo=np.nan, ci_hi=np.nan,
                            win_vs_random=float(sub.win_vs_random.mean()), d_phys_mean=float(sub.d_phys_mean.mean()),
                            d_phys_random=float(sub.d_phys_random.mean()), neg_k=int((sub.delta_vs_random < 0).sum())))
    tests = pd.concat([tests, pd.DataFrame(agg)], ignore_index=True) if agg else tests
    return curve, tests


def main():
    t0 = time.time()
    tasks = [(tg, sp) for tg in TARGETS for sp in SPLITS]
    rows, metas = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_task, tg, sp): (tg, sp) for tg, sp in tasks}
        for f in as_completed(futs):
            r, m = f.result(); rows += r; metas.append(m)
            print(f"  [{m['target']}|{m['split']}] 적합 {m['n_fit']} · {m['elapsed_s']}s · 누적 {time.time()-t0:.0f}s", flush=True)
    runs = pd.DataFrame(rows)
    tag = args.tag + ("_smoke" if args.smoke else "")
    runs.to_csv(OUT / f"{tag}_runs.csv", index=False)
    curve, tests = summarize(runs)
    curve.to_csv(OUT / f"{tag}_curve.csv", index=False); tests.to_csv(OUT / f"{tag}_tests.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                                                 # noqa: BLE001
        commit = "NA"
    (OUT / f"{tag}_meta.json").write_text(json.dumps(dict(
        stage="H24", design="docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md", E_AK=E_AK, E0_sandbox=E0_AK_TARGET, E_reg=E_reg, kappa=args.kappa,
        strategies=STRATS, n_grid=N_GRID, n_grid_ak=N_GRID_AK, reps=REPS, reps_cb=REPS_CB, seeds=SEEDS, lam=args.lam,
        tasks=metas, git_commit=commit, elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1, default=float))
    show = tests[(tests.target == "MEAN[AB4]") & (tests.n.isin([3, 10]))].sort_values(["method", "n", "strategy"]) if len(tests) else tests
    if len(show):
        print(show[["strategy", "method", "n", "delta_vs_random", "win_vs_random", "d_phys_mean", "d_phys_random", "neg_k"]].to_string())
    print(f"saved {tag}_* · 행 {len(runs)} · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
