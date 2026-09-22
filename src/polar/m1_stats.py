"""개정 09-21 채점 규약의 순수 함수판 (scripts/3_deep_learning/m1_analysis.py 의 boot_delta 계열을 모듈화).

규약(docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md 개정 09-21 §A)
  주 채점 = 셀 가중 RMSE. 보조 = 블록 등가중 RMSE, 블록당 셀 상한 100 가중 RMSE.
  짝지음 = seed별 Δ 후 seed 평균. 부트스트랩 = 블록 재표집(1,000회), 통계량 = seed 평균 Δ.
  지역 수준 = 층화 결합(블록 ≥ 8 지역만 독립 재표집, 지역 비가중 평균 Δ). 가족별 Holm.
"""
from __future__ import annotations
import zlib
import numpy as np

MIN_BLOCKS_CI = 8
CAP_CELLS = 100


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


def block_sse(y, inv, nb, P):
    E2 = (P - y[None, :]) ** 2
    return np.stack([np.bincount(inv, weights=E2[s], minlength=nb) for s in range(P.shape[0])])


def scores(SSE, n_b, pick=None):
    w_b = np.minimum(1.0, CAP_CELLS / n_b)
    if pick is None:
        return (np.sqrt(SSE.sum(1) / n_b.sum()), np.sqrt(SSE / n_b).mean(1),
                np.sqrt((SSE * w_b).sum(1) / (w_b * n_b).sum()))
    S_, n_, w_ = SSE[:, pick], n_b[pick], w_b[pick]
    return (np.sqrt(S_.sum(2) / n_.sum(1)[None, :]), np.sqrt(S_ / n_[None]).mean(2),
            np.sqrt((S_ * w_[None]).sum(2) / (w_ * n_).sum(1)[None, :]))


def boot_delta(y, blocks, PA, PB, nboot=1000, seed=0):
    """PA·PB (S,n): seed 짝지은 예측. Δ = RMSE(A) − RMSE(B). 블록 < MIN_BLOCKS_CI 이면 분포 None."""
    y = np.asarray(y, float); PA = np.atleast_2d(np.asarray(PA, float)); PB = np.atleast_2d(np.asarray(PB, float))
    m = np.isfinite(y) & np.all(np.isfinite(PA), 0) & np.all(np.isfinite(PB), 0)
    y, PA, PB, blocks = y[m], PA[:, m], PB[:, m], np.asarray(blocks)[m]
    ub, inv = np.unique(blocks, return_inverse=True)
    nb = len(ub)
    n_b = np.bincount(inv, minlength=nb).astype(float)
    SA, SB = block_sse(y, inv, nb, PA), block_sse(y, inv, nb, PB)
    EA, EB = block_sse(y, inv, nb, PA.mean(0, keepdims=True)), block_sse(y, inv, nb, PB.mean(0, keepdims=True))
    cA, bA, kA = scores(SA, n_b)
    cB, bB, kB = scores(SB, n_b)
    d_blk = (np.sqrt(SA / n_b) - np.sqrt(SB / n_b)).mean(0)
    out = dict(n_cells=int(m.sum()), n_blocks=int(nb), n_seed=int(PA.shape[0]),
               rmse_A=float(cA.mean()), rmse_B=float(cB.mean()),
               delta_per_seed=float((cA - cB).mean()), delta_blockeq=float((bA - bB).mean()), delta_cap100=float((kA - kB).mean()),
               delta_ensemble=float(scores(EA, n_b)[0][0] - scores(EB, n_b)[0][0]),
               block_delta=d_blk, block_neg_k=int((d_blk < 0).sum()), block_majority=float((d_blk < 0).mean()) if nb else np.nan,
               dist_cell=None, dist_blockeq=None)
    if nb >= MIN_BLOCKS_CI and nboot > 0:
        pick = np.random.RandomState(seed).randint(0, nb, size=(nboot, nb))
        cA_, bA_, _ = scores(SA, n_b, pick)
        cB_, bB_, _ = scores(SB, n_b, pick)
        out["dist_cell"], out["dist_blockeq"] = (cA_ - cB_).mean(0), (bA_ - bB_).mean(0)
    return out


def ci(dist):
    return (float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))) if dist is not None else (np.nan, np.nan)


def boot_p(dist):
    if dist is None or not len(dist):
        return np.nan
    p = 2.0 * min(np.mean(dist <= 0), np.mean(dist >= 0))
    return float(min(1.0, max(p, 1.0 / len(dist))))


def strat(dists):
    """층화 결합: 분포가 있는 지역만 독립 결합 → 지역 비가중 평균 Δ의 분포."""
    ok = [d for d in dists if d is not None]
    if not ok:
        return None
    L = min(len(d) for d in ok)
    return np.mean([d[:L] for d in ok], axis=0)


def holm(pvals):
    """Holm 보정. NaN 은 그대로."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        out[i] = min(1.0, run)
    return out


def summarize_delta(name, per_region: dict, regions: list, seed: int):
    """지역별 boot_delta 결과 dict → 지역 층화 요약 행 + 지역 행."""
    rows = []
    dists, dists_b = [], []
    for r in regions:
        d = per_region.get(r)
        if d is None:
            continue
        lo, hi = ci(d["dist_cell"]); lob, hib = ci(d["dist_blockeq"])
        rows.append(dict(test=name, target=r, n_cells=d["n_cells"], n_blocks=d["n_blocks"], rmse_A=d["rmse_A"], rmse_B=d["rmse_B"],
                         delta=d["delta_per_seed"], ci_lo=lo, ci_hi=hi, p_boot=boot_p(d["dist_cell"]),
                         delta_blockeq=d["delta_blockeq"], ci_lo_beq=lob, ci_hi_beq=hib, delta_cap100=d["delta_cap100"],
                         block_majority=d["block_majority"], ci_flag="" if d["dist_cell"] is not None else "blocks<8"))
        dists.append(d["dist_cell"]); dists_b.append(d["dist_blockeq"])
    have = [r for r in regions if r in per_region]
    if have:
        have_ci = [r for r in have if per_region[r]["dist_cell"] is not None]
        pool = have_ci if have_ci else have                  # 점 추정치와 CI 의 지역 집합을 일치(감사 2026-09-22: 블록<8 지역이 점만 끌던 결함)
        dm = np.mean([per_region[r]["delta_per_seed"] for r in pool])
        db = np.mean([per_region[r]["delta_blockeq"] for r in pool])
        dm_all = np.mean([per_region[r]["delta_per_seed"] for r in have])
        sd, sdb = strat(dists), strat(dists_b)
        lo, hi = ci(sd); lob, hib = ci(sdb)
        assert not (np.isfinite(lo) and (dm < lo - 1e-9 or dm > hi + 1e-9)), f"{name}: 점 추정치 {dm:.3f} 가 CI [{lo:.3f}, {hi:.3f}] 밖"
        neg = sum(per_region[r]["delta_per_seed"] < 0 for r in have)
        rows.append(dict(test=name, target=f"MEAN[{','.join(pool)}]", n_cells=sum(per_region[r]["n_cells"] for r in pool),
                         n_blocks=sum(per_region[r]["n_blocks"] for r in pool),
                         rmse_A=np.mean([per_region[r]["rmse_A"] for r in pool]), rmse_B=np.mean([per_region[r]["rmse_B"] for r in pool]),
                         delta=dm, ci_lo=lo, ci_hi=hi, p_boot=boot_p(sd), delta_blockeq=db, ci_lo_beq=lob, ci_hi_beq=hib,
                         delta_cap100=np.mean([per_region[r]["delta_cap100"] for r in pool]),
                         block_majority=np.mean([per_region[r]["block_majority"] for r in pool]),
                         delta_allregions=dm_all, n_regions_all=len(have), regions_all=",".join(have),
                         ci_flag=f"neg {neg}/{len(have)}; ci_regions {len(have_ci)}"))
    return rows
