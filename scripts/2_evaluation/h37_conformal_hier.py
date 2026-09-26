"""H37 · Track C2 계층 conformal 구간(라벨 0 새 지역). CPU.

목적
  라벨이 없는 새 지역에 대해 물리 앵커(Stefan, E0 = 대상 외 원천 최소제곱) 예측의 90 % 구간을
  leave-one-region-out 으로 만들고, 커버리지·구간 폭·interval score 를 비교한다.
  계획 docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 Track C2, 가설 F10(§5).

점수
  s = |log y − log(E0·√TDD)|. 구간은 [ŷ·e^(−q), ŷ·e^(q)] (y 스케일 역변환, 폭 cm).
  E0 는 대상 셀과 대상 경계 --buffer-km(기본 100 km, h25 규약) 이내 셀을 제외한 원천 라벨 셀의 최소제곱.
  보정 집합 = 원천 안의 macro 지역 중 셀 수 ≥ --min-cal-cells(기본 20) 인 지역(K 그룹). 알래스카 포함.

방법(라벨 0)
  pooled      모든 보정 셀을 풀어 유한표본 conformal 분위(ceil((N+1)(1−α)) 번째 순서통계). 셀 교환성 가정.
  hier2_cdf   [주 방법] Dunn·Wasserman·Ramdas(2023) 2층 계층 conformal의 CDF 풀링: 지역별 경험 CDF 를
              등가중 평균한 F̄(x) = (1/K)ΣF̂_k(x) 의 (1−α) 분위(plug-in). 그룹 = macro 지역.
  hier2_exact 같은 CDF 풀링에 +∞ 질량 1/(K+1) 을 더한 유한표본 정확판(subsampling-once 의 기대 CDF).
              K+1 < 1/α(K ≤ 8)이면 구간이 무한이 된다. 이 자료는 K = 4–5 이므로 기록 목적으로만 둔다.
  hier2_sub   반복 서브샘플링: 각 지역에서 1셀씩 뽑은 K개 점수의 plug-in 분위(ceil(K(1−α)) 번째)를 B 회 평균.
              K = 4 이면 최대값의 평균이므로 hier2_cdf 보다 넓은 편향을 가진다. 보조 변형.
  hier2_blk   보조: 그룹을 0.5° 블록으로 둔 CDF 풀링(블록 교환성 가정, 지역 단위 아님).
  ref_*       현행 전이 UQ(H17, data/processed/m1/transfer_uq_summary.csv, cond = noinfo)의 cqr_ak·cqr_iw·
              anchor_ak·anchor_iw 커버리지·폭·CI 를 그대로 옮겨 적는다(재계산 없음).
  주 방법 선택 근거: hier2_cdf 는 결정적이고 hier2_sub 의 CDF 기대값과 일치한다. 유한표본 보장은 K ≥ 9 에서만
  성립하므로 여기서의 커버리지는 경험 평가(LORO)로 판정한다(F10).

라벨 있음(비교환성 가중 conformal, Barber·Candès·Ramdas·Tibshirani 2023)
  대상 A 블록에서 n ∈ --n-grid(규칙 --rules: kmedoid 주, random 참조; 분할 --splits, 반복 --reps) 라벨을 뽑아
  점수(앵커는 E0 고정, 라벨로 E 를 다시 맞추면 같은 라벨이 보정 점수와 겹쳐 낙관적이므로 고정)를 가중 1,
  검정점 가중 1, 원천 지역 k 의 셀은 지역 총 가중 ω(--src-weights: '1/K','1','2','4', 0 = 대상만)를 n_k 로 나눈
  값으로 둔 가중 분위 q = inf{x : Σ w̃_i 1{s_i ≤ x} ≥ 1−α}, w̃ 는 검정점 가중을 포함해 정규화. 검정점 질량
  w̃_test > α 이면 구간이 무한이다(n = 3, ω = 1, K = 4 이면 1/8 > 0.1 로 무한). 커버리지 손실 상한의 계수
  Σ_{원천} w̃_i(src_mass)를 함께 기록한다(d_TV 는 미지이므로 상한 자체는 계산하지 않는다).
  kmedoid 는 KMeans 중심 최근접이라 반복 간 선택이 거의 같다(반복 분산은 random 규칙에서 본다).

채점
  대상 전체 유효 셀(eval_mask, scope = all)과 분할별 B 블록 유효 셀(scope = B). 커버리지(셀 가중·블록 등가중),
  평균 폭(cm), interval score(α = 0.1), 무한 구간 비율. 지역 안 CI: 라벨 0 은 블록 부트스트랩(블록 ≥ 8),
  라벨 있음은 (분할, 반복) 짝지음 부트스트랩 1000회 percentile 95 %(Δ = 방법 − hier2_cdf, 같은 분할 B 셀).
  지역 평균: MAIN6 등가중(층화 블록 부트스트랩 결합) + 셀 가중 병기. F10 = MAIN6 중 hier2_cdf 커버리지가
  [0.85, 0.95] 에 드는 지역 수와 ref_cqr_ak < 0.85 지역 수.

입력  data/processed/fidelity_base_v3.csv, e5_soil_tdd_v3.csv(load_base), m1/transfer_uq_summary.csv(참조)
출력  data/processed/h4/<tag>_runs.csv, <tag>_coverage.csv, <tag>_meta.json (스모크는 <tag>_smoke_*)
실행  python3 scripts/2_evaluation/h37_conformal_hier.py --workers 6 [--smoke] [--tag c2]
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
from polar.fidelity import TARGET                                                        # noqa: E402
from polar.m1_core import INPUT_SETS, load_base, eval_mask, half_split_blocks           # noqa: E402
from polar.m1_stats import MIN_BLOCKS_CI, ci, strat                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", default="", help="쉼표 목록(기본: MAIN6 + Alaska)")
ap.add_argument("--splits", type=int, default=3)
ap.add_argument("--n-grid", default="3,10")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--rules", default="kmedoid,random", help="라벨 선택 규칙(주 = kmedoid; random 은 반복 분산 참조)")
ap.add_argument("--src-weights", default="0,1/K,1,2,4", help="원천 지역 총 가중 ω 목록(0 = 대상 라벨만)")
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--B", type=int, default=500, help="hier2_sub 반복 횟수")
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--min-cal-cells", type=int, default=20)
ap.add_argument("--buffer-km", type=float, default=100.0)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--tag", default="c2")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"; OUT = PROC / "h4"; OUT.mkdir(exist_ok=True)
FEATS = INPUT_SETS["x25"]
MAIN6 = ["Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"]
AB4 = MAIN6[:4]
ALPHA = args.alpha
SPLITS = list(range(args.splits)) if not args.smoke else [0]
N_GRID = [int(v) for v in args.n_grid.split(",")] if not args.smoke else [3]
REPS = args.reps if not args.smoke else 2
NB = args.B if not args.smoke else 100
NBOOT = args.nboot if not args.smoke else 200
SRC_W = args.src_weights.split(",")
RULES = args.rules.split(",")
TARGETS = args.targets.split(",") if args.targets else MAIN6 + ["Alaska"]
if args.smoke:
    TARGETS = ["Lena", "Russia_W"]
LABEL0 = ["pooled", "hier2_cdf", "hier2_exact", "hier2_sub", "hier2_blk"]
PRIMARY = "hier2_cdf"
REF_FILE = PROC / "m1" / "transfer_uq_summary.csv"

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float); DF["y"] = DF[TARGET].values.astype(float)
DF["ev"] = eval_mask(DF)
print(f"[data] {len(DF):,}셀 · 대상 {TARGETS} · 분할 {SPLITS} · n {N_GRID} · 반복 {REPS}", flush=True)


# ---------------------------------------------------------------- 분위·지표
def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


def wquantile(scores, w, level, w_test=0.0):
    """가중 경험분포(+∞ 질량 w_test 포함 정규화)의 level 분위. 누적 가중이 level 에 못 미치면 inf."""
    o = np.argsort(scores, kind="stable"); s, ww = np.asarray(scores, float)[o], np.asarray(w, float)[o]
    cum = np.cumsum(ww) / (ww.sum() + w_test)
    k = int(np.searchsorted(cum, level - 1e-12))
    return float(s[k]) if k < len(s) else np.inf


def group_weights(groups, total_each=1.0):
    """그룹별 총 가중 total_each 를 그룹 안 셀에 균등 분배."""
    ug, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    return total_each / cnt[inv], len(ug)


def label0_quantiles(cal_s, cal_g, cal_blk, rng):
    K = len(np.unique(cal_g)); N = len(cal_s)
    w_g, _ = group_weights(cal_g, 1.0 / K)                                     # 지역 등가중, 합 1
    w_b, Kb = group_weights(cal_blk, 1.0)
    q = dict(pooled=wquantile(cal_s, np.ones(N), 1 - ALPHA, w_test=1.0),
             hier2_cdf=wquantile(cal_s, w_g, 1 - ALPHA, w_test=0.0),
             hier2_exact=wquantile(cal_s, w_g, 1 - ALPHA, w_test=1.0 / K),
             hier2_blk=wquantile(cal_s, w_b / Kb, 1 - ALPHA, w_test=0.0))
    idx_by_g = [np.where(cal_g == g)[0] for g in np.unique(cal_g)]
    kk = int(np.ceil(K * (1 - ALPHA))) - 1
    sub = np.array([np.sort(cal_s[[rng.choice(ix) for ix in idx_by_g]])[kk] for _ in range(NB)])
    q["hier2_sub"] = float(sub.mean())
    return q, dict(K_groups=K, N_cal=N, K_blocks=Kb, hier2_sub_sd=float(sub.std()))


def interval_stats(y, yhat, q, blocks):
    """커버리지(셀·블록 등가중)·폭·interval score. q = inf 이면 폭·IS 는 inf."""
    r = np.abs(np.log(y) - np.log(yhat)); cov = (r <= q).astype(float)
    if np.isfinite(q):
        lo, hi = yhat * np.exp(-q), yhat * np.exp(q)
        width = hi - lo
        IS = width + (2 / ALPHA) * np.clip(lo - y, 0, None) + (2 / ALPHA) * np.clip(y - hi, 0, None)
    else:
        width = IS = np.full(len(y), np.inf)
    ub, inv = np.unique(blocks, return_inverse=True); nb = len(ub)
    cnt = np.bincount(inv, minlength=nb)
    cov_beq = float(np.mean(np.bincount(inv, cov, minlength=nb) / cnt))
    return dict(coverage=float(cov.mean()), coverage_beq=cov_beq, width_cm=float(width.mean()), interval_score=float(IS.mean()),
                is_inf=float(np.isinf(q)), n_eval=int(len(y)), n_blocks=int(nb)), (cov, width, IS, inv, nb, cnt)


def block_boot(parts, seed=0):
    """블록 재표집(nboot) 커버리지·폭·IS 분포. 블록 < MIN_BLOCKS_CI 또는 inf 면 None."""
    cov, width, IS, inv, nb, cnt = parts
    if nb < MIN_BLOCKS_CI or not np.all(np.isfinite(width)):
        return None
    S = np.stack([np.bincount(inv, v, minlength=nb) for v in (cov, width, IS)])       # (3, nb)
    pick = np.random.RandomState(seed).randint(0, nb, size=(NBOOT, nb))
    num = S[:, pick].sum(2); den = cnt[pick].sum(1)                                      # (3, nboot), (nboot,)
    return dict(coverage=num[0] / den, width_cm=num[1] / den, interval_score=num[2] / den)


def select(rule, n, rng, Z):
    if rule == "random":
        return rng.choice(len(Z), n, replace=False)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.randint(1 << 30))).fit(Z)
    sel = []
    for c in km.cluster_centers_:
        d = ((Z - c) ** 2).sum(1); d[sel] = np.inf; sel.append(int(np.argmin(d)))
    return np.array(sel)


def source_index(t_idx):
    src = np.where(np.isfinite(DF.y.values))[0]
    src = src[~np.isin(src, t_idx)]
    if args.buffer_km > 0:
        from sklearn.neighbors import BallTree
        ll = np.radians(DF[["lat", "lon"]].values.astype(float))
        d, _ = BallTree(ll[t_idx], metric="haversine").query(ll[src], k=1)
        src = src[d[:, 0] * 6371.0 >= args.buffer_km]
    return src


# ---------------------------------------------------------------- 작업 단위(대상)
def run_task(t):
    t0 = time.time()
    t_idx = np.where(DF.macro.values == t)[0]
    src_idx = source_index(t_idx)
    src = DF.iloc[src_idx]
    E0 = ls_E(src.y.values, src.s.values)
    s_src = np.abs(np.log(src.y.values) - np.log(E0 * src.s.values))
    g_src = src.macro.values.astype(str)
    ug, cnt = np.unique(g_src, return_counts=True)
    keep_g = ug[cnt >= args.min_cal_cells]
    m = np.isin(g_src, keep_g)
    cal_s, cal_g, cal_blk = s_src[m], g_src[m], src.block.values[m]
    rng = np.random.RandomState(1_000 + TARGETS.index(t) if t in TARGETS else 1_000)
    Q0, qinfo = label0_quantiles(cal_s, cal_g, cal_blk, rng)
    base = dict(target=t, E0=E0, n_src=int(len(src_idx)), cal_groups=",".join(keep_g), **qinfo)
    rows, boots = [], {}

    def add(scope, split, n, rep, src_w, method, q, cells, extra=None):
        d = DF.iloc[cells]
        st, parts = interval_stats(d.y.values, E0 * d.s.values, q, d.block.values)
        rows.append(dict(**base, scope=scope, split=split, n=n, rep=rep, src_w=src_w, method=method, q_log=q, **st, **(extra or {})))
        return parts

    ev_all = t_idx[DF.ev.values[t_idx]]
    if len(ev_all):
        for mth, q in Q0.items():
            parts = add("all", -1, 0, -1, "", mth, q, ev_all)
            boots[mth] = block_boot(parts, seed=1)
    for sp in SPLITS:
        A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
        evB = B_idx[DF.ev.values[B_idx]]
        if len(evB) == 0:
            continue
        for mth, q in Q0.items():
            add("B", sp, 0, -1, "", mth, q, evB)
        A = DF.iloc[A_idx]; nA = len(A)
        yA, sA = A.y.values, A.s.values
        Xa = A[FEATS].values.astype(float); med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
        Xa = np.where(np.isnan(Xa), med, Xa); Z = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
        K = qinfo["K_groups"]
        w_g1, _ = group_weights(cal_g, 1.0)                                              # 지역 총 가중 1
        for n in N_GRID:
            if n >= nA:
                continue
            for rule in RULES:
                for rep in range(REPS):
                    r2 = np.random.RandomState(7_919 * sp + 1_009 * n + 31 * rep + 101 * RULES.index(rule) + 17)
                    sel = select(rule, n, r2, Z)
                    s_t = np.abs(np.log(yA[sel]) - np.log(E0 * sA[sel]))
                    for sw in SRC_W:
                        omega = 1.0 / K if sw == "1/K" else float(sw)
                        if omega == 0:
                            sc, w = s_t, np.ones(n)
                        else:
                            sc, w = np.concatenate([s_t, cal_s]), np.concatenate([np.ones(n), omega * w_g1])
                        q = wquantile(sc, w, 1 - ALPHA, w_test=1.0)
                        tot = w.sum() + 1.0
                        add("B", sp, n, rep, sw, "wconf", q, evB,
                            dict(rule=rule, src_mass=float((tot - n - 1.0) / tot), test_mass=float(1.0 / tot), n_A=nA,
                                 s_t_max=float(s_t.max()), E_ls_n=ls_E(yA[sel], sA[sel])))
    meta = dict(n_cells=int(len(t_idx)), n_eval_all=int(len(ev_all)), elapsed_s=round(time.time() - t0, 1), **base,
                **{f"q_{k}": v for k, v in Q0.items()})
    return rows, boots, meta


# ---------------------------------------------------------------- 집계
def boot_pairs(d, seed=1):
    if len(d) < 3:
        return np.nan, np.nan
    rng = np.random.RandomState(seed); bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(NBOOT)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarize(runs, boots):
    out = []
    # 라벨 0 · scope all: 대상별 + 블록 부트스트랩 CI
    r0 = runs[(runs.n == 0) & (runs.scope == "all")]
    for _, row in r0.iterrows():
        b = boots.get(row.target, {}).get(row.method)
        rec = dict(test="label0", target=row.target, method=row.method, n=0, src_w="", scope="all", n_runs=1,
                   coverage=row.coverage, coverage_beq=row.coverage_beq, width_cm=row.width_cm, interval_score=row.interval_score,
                   is_inf=row.is_inf, q_log=row.q_log, n_eval=row.n_eval, n_blocks=row.n_blocks, K_groups=row.K_groups, N_cal=row.N_cal, E0=row.E0,
                   in_band=float(0.85 <= row.coverage <= 0.95), ci_flag="" if b is not None else ("blocks<8" if row.n_blocks < MIN_BLOCKS_CI else "inf"))
        for k in ("coverage", "width_cm", "interval_score"):
            lo, hi = ci(b[k]) if b is not None else (np.nan, np.nan); rec[f"{k}_lo"], rec[f"{k}_hi"] = lo, hi
        out.append(rec)
    # 라벨 0 · MAIN6 평균(등가중: 층화 블록 부트스트랩 결합, 셀 가중: n_eval 가중)
    for mth in LABEL0:
        sub = r0[(r0.method == mth) & r0.target.isin(MAIN6)]
        if not len(sub):
            continue
        have = list(sub.target); have_ci = [t for t in have if boots.get(t, {}).get(mth) is not None]
        pool = have_ci if have_ci else have
        sp = sub.set_index("target")
        rec = dict(test="label0", target=f"MEAN6[{','.join(pool)}]", method=mth, n=0, src_w="", scope="all", n_runs=len(pool),
                   n_eval=int(sp.loc[pool].n_eval.sum()), n_blocks=int(sp.loc[pool].n_blocks.sum()), q_log=float(sp.loc[pool].q_log.mean()),
                   is_inf=float(sp.loc[pool].is_inf.mean()), in_band=float(sum(0.85 <= sp.loc[t].coverage <= 0.95 for t in have)),
                   coverage_allregions=float(sp.loc[have].coverage.mean()), regions_all=",".join(have),
                   ci_flag=f"ci_regions {len(have_ci)}/{len(have)}")
        wcell = sp.loc[pool].n_eval.values.astype(float); wcell = wcell / wcell.sum()
        for k in ("coverage", "coverage_beq", "width_cm", "interval_score"):
            rec[k] = float(sp.loc[pool][k].mean())
            rec[f"{k}_cellw"] = float((sp.loc[pool][k].values * wcell).sum())
        for k in ("coverage", "width_cm", "interval_score"):
            dists = [boots[t][mth][k] for t in have_ci]
            lo, hi = ci(strat(dists)) if dists else (np.nan, np.nan); rec[f"{k}_lo"], rec[f"{k}_hi"] = lo, hi
            if dists:
                wc = np.array([sp.loc[t].n_eval for t in have_ci], float); wc = wc / wc.sum()
                L = min(len(d) for d in dists); dw = np.sum([w * d[:L] for w, d in zip(wc, dists)], 0)
                rec[f"{k}_cellw_lo"], rec[f"{k}_cellw_hi"] = ci(dw)
        out.append(rec)
    # 라벨 있음: (분할, 반복) 평균, Δ 대 hier2_cdf(같은 분할 B), 짝지음 부트스트랩 CI
    rB = runs[runs.scope == "B"]
    ref = rB[(rB.n == 0) & (rB.method == PRIMARY)].set_index(["target", "split"])
    rl = rB[rB.n > 0].copy()
    for k in ("coverage", "width_cm", "interval_score"):
        rl[f"d_{k}"] = rl[k].values - np.array([ref.loc[(t, s), k] for t, s in zip(rl.target, rl.split)])
    keys = ["target", "method", "rule", "n", "src_w"]
    lab_rows = []
    for (t, mth, rule, n, sw), sub in rl.groupby(keys):
        rec = dict(test="labeled", target=t, method=mth, rule=rule, n=int(n), src_w=sw, scope="B", n_runs=len(sub), n_splits=int(sub.split.nunique()),
                   n_eval=float(sub.n_eval.mean()), n_blocks=float(sub.n_blocks.mean()), q_log=float(sub.q_log.mean()), is_inf=float(sub.is_inf.mean()),
                   src_mass=float(sub.src_mass.mean()), test_mass=float(sub.test_mass.mean()), K_groups=int(sub.K_groups.iloc[0]), E0=float(sub.E0.iloc[0]),
                   in_band=float(0.85 <= sub.coverage.mean() <= 0.95), ci_flag="")
        for k in ("coverage", "coverage_beq", "width_cm", "interval_score"):
            rec[k] = float(sub[k].mean())
        for k in ("coverage", "width_cm", "interval_score"):
            rec[f"{k}_lo"], rec[f"{k}_hi"] = boot_pairs(sub[k].values)
            rec[f"d_{k}"] = float(sub[f"d_{k}"].mean()); rec[f"d_{k}_lo"], rec[f"d_{k}_hi"] = boot_pairs(sub[f"d_{k}"].values)
        lab_rows.append(rec)
    out.extend(lab_rows)
    lab = pd.DataFrame(lab_rows)
    if len(lab):
        for (mth, rule, n, sw), sub in lab[lab.target.isin(MAIN6)].groupby(["method", "rule", "n", "src_w"]):
            rec = dict(test="labeled", target=f"MEAN[{','.join(sub.target)}]", method=mth, rule=rule, n=int(n), src_w=sw, scope="B", n_runs=int(sub.n_runs.sum()),
                       n_eval=float(sub.n_eval.sum()), q_log=float(sub.q_log.mean()), is_inf=float(sub.is_inf.mean()), src_mass=float(sub.src_mass.mean()),
                       in_band=float((sub.coverage.between(0.85, 0.95)).sum()), regions_all=",".join(sub.target), ci_flag="point only")
            wc = sub.n_eval.values / sub.n_eval.sum()
            for k in ("coverage", "coverage_beq", "width_cm", "interval_score", "d_coverage", "d_width_cm", "d_interval_score"):
                rec[k] = float(sub[k].mean()); rec[f"{k}_cellw"] = float((sub[k].values * wc).sum())
            out.append(rec)
    # 참조 행: 현행 전이 UQ(H17) 옮겨 적기
    n_ref_lt85 = np.nan
    if REF_FILE.exists():
        rf = pd.read_csv(REF_FILE)
        rf = rf[(rf.cond == "noinfo") & rf.method.isin(["cqr_ak", "cqr_iw", "anchor_ak", "anchor_iw"]) & rf.region.isin(MAIN6)]
        for _, r in rf.iterrows():
            out.append(dict(test="ref_h17", target=r.region, method=f"ref_{r.method}", n=0, src_w="", scope="all(noinfo)", n_runs=int(r.n_seeds),
                            coverage=r.coverage, coverage_lo=r.cov_ci_lo, coverage_hi=r.cov_ci_hi, width_cm=r.width_mean, interval_score=r.interval_score,
                            interval_score_lo=r.is_ci_lo, interval_score_hi=r.is_ci_hi, is_inf=r.frac_inf, n_eval=int(r.n_cells), n_blocks=int(r.n_blocks),
                            in_band=float(0.85 <= r.coverage <= 0.95), ci_flag=f"copied from {REF_FILE.relative_to(ROOT)}; {r.ci_flag if isinstance(r.ci_flag, str) else ''}"))
        for mth, sub in rf.groupby("method"):
            wc = sub.n_cells.values / sub.n_cells.sum()
            out.append(dict(test="ref_h17", target=f"MEAN6[{','.join(sub.region)}]", method=f"ref_{mth}", n=0, src_w="", scope="all(noinfo)", n_runs=len(sub),
                            coverage=float(sub.coverage.mean()), coverage_cellw=float((sub.coverage.values * wc).sum()), width_cm=float(sub.width_mean.mean()),
                            interval_score=float(sub.interval_score.mean()), n_eval=int(sub.n_cells.sum()), in_band=float(sub.coverage.between(0.85, 0.95).sum()),
                            ci_flag="point only (copied)"))
        c = rf[rf.method == "cqr_ak"]; n_ref_lt85 = int((c.coverage < 0.85).sum())
    # F10
    prim = r0[(r0.method == PRIMARY) & r0.target.isin(MAIN6)]
    n_in = int(prim.coverage.between(0.85, 0.95).sum())
    f10 = dict(test="F10", target="MAIN6", method=PRIMARY, n=0, src_w="", scope="all", n_runs=len(prim), in_band=float(n_in),
               coverage=float(prim.coverage.mean()) if len(prim) else np.nan, n_eval=int(prim.n_eval.sum()) if len(prim) else 0,
               ci_flag=f"hier2_cdf in [0.85,0.95]: {n_in}/{len(prim)}; ref_cqr_ak<0.85: {n_ref_lt85}/{len(MAIN6)}; regions_in_band="
                       + ",".join(prim[prim.coverage.between(0.85, 0.95)].target))
    out.append(f10)
    cols = ["test", "target", "method", "rule", "n", "src_w", "scope", "n_runs", "n_splits", "coverage", "coverage_lo", "coverage_hi", "coverage_beq", "coverage_cellw",
            "coverage_cellw_lo", "coverage_cellw_hi", "width_cm", "width_cm_lo", "width_cm_hi", "width_cm_cellw", "interval_score", "interval_score_lo", "interval_score_hi",
            "interval_score_cellw", "d_coverage", "d_coverage_lo", "d_coverage_hi", "d_width_cm", "d_width_cm_lo", "d_width_cm_hi", "d_interval_score",
            "d_interval_score_lo", "d_interval_score_hi", "is_inf", "q_log", "src_mass", "test_mass", "in_band", "n_eval", "n_blocks", "K_groups", "N_cal", "E0",
            "coverage_allregions", "regions_all", "ci_flag"]
    df = pd.DataFrame(out)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df[cols], dict(F10_n_in_band=n_in, F10_n_regions=int(len(prim)), F10_regions_in_band=list(prim[prim.coverage.between(0.85, 0.95)].target),
                          ref_cqr_ak_n_below_085=None if n_ref_lt85 is np.nan else n_ref_lt85)


def main():
    t0 = time.time()
    rows, boots, metas = [], {}, []

    def _done(r, b, m):
        rows.extend(r); boots[m["target"]] = b; metas.append(m)
        print(f"  [{m['target']}] 셀 {m['n_cells']} · 평가 {m['n_eval_all']} · 원천 {m['n_src']} · K {m['K_groups']} · E0 {m['E0']:.3f} · "
              f"q pooled {m['q_pooled']:.3f} hier2_cdf {m['q_hier2_cdf']:.3f} sub {m['q_hier2_sub']:.3f} · {m['elapsed_s']}s", flush=True)

    if args.workers <= 1 or len(TARGETS) == 1:
        for t in TARGETS:
            _done(*run_task(t))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(TARGETS)), mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = {ex.submit(run_task, t): t for t in TARGETS}
            for f in as_completed(futs):
                _done(*f.result())
    runs = pd.DataFrame(rows); tag = args.tag + ("_smoke" if args.smoke else "")
    runs.to_csv(OUT / f"{tag}_runs.csv", index=False)
    cov, f10 = summarize(runs, boots)
    cov.to_csv(OUT / f"{tag}_coverage.csv", index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                                                       # noqa: BLE001
        commit = "NA"
    meta = dict(stage="H37/C2", plan="docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 C2, §5 F10", primary_method=PRIMARY,
                methods_label0=LABEL0, targets=TARGETS, splits=SPLITS, n_grid=N_GRID, reps=REPS, rules=RULES, src_weights=SRC_W, alpha=ALPHA,
                B_sub=NB, nboot=NBOOT, min_cal_cells=args.min_cal_cells, buffer_km=args.buffer_km, ref_file=str(REF_FILE.relative_to(ROOT)),
                score="abs(log y - log(E0*sqrt_tdd)); E0 = source least squares (target + buffer excluded)",
                note_exact="hier2_exact (subsampling-once expectation, +inf mass 1/(K+1)) is infinite when K+1 < 1/alpha; K here is 4-5",
                per_target=metas, F10=f10, git_commit=commit, smoke=args.smoke, n_rows_runs=int(len(runs)), n_rows_coverage=int(len(cov)),
                elapsed_s=round(time.time() - t0, 1))
    (OUT / f"{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    show = cov[(cov.test == "label0") & (cov.method.isin(["pooled", "hier2_cdf", "hier2_sub"]))]
    print(show[["target", "method", "coverage", "coverage_lo", "coverage_hi", "width_cm", "interval_score", "n_eval", "in_band"]].to_string(index=False))
    print(cov[cov.test == "F10"].ci_flag.iloc[0])
    print(f"saved {tag}_* · runs {len(runs)} 행 · coverage {len(cov)} 행 · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
