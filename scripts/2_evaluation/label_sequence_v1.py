"""지도 과제(MAP) · 라벨 수별 지도 수열(showpiece). 한 지역의 라벨 셀을 n = 0, 10, 40, 160, 전량으로 늘리며 워크플로가 권하는 방법을
그 n 에서 적합해 1 km 격자 전체를 예측한다. 모형 계산만 하고 그림은 scripts/4_visualization/paper_v3/label_sequence_fig.py 가 그린다.

추출  중첩 블록 분산 순서(seed 0): 0.5° 블록을 RandomState(0) 으로 순열하고 블록 안 셀도 순열한 뒤 블록을 돌아가며 한 셀씩 뽑는다
      (h40.draw_blocks 와 같은 규칙, 다만 순열을 한 번만 만들어 n 마다 앞 n 개를 쓴다. 그래서 n 이 커져도 앞 단계의 셀이 그대로 남는다).
방법  n = 0   P0 = E0·s, E0 = 원천 풀(다른 지역 전부, 100 km 버퍼) 최소제곱 계수(map_alaska_v1.region_context = h54.build_rctx 규칙)
      n = 10  P1 = E_n·s, E_n = (n·E_ls + κ·E0)/(n + κ), κ = 10(h54 RUnit.coefs)
      n ≥ 40  R1 = E_n·s + λ·ḡ, g = catboost_lo 잔차 모형(seed 0·1 평균), λ ∈ {0.25, 0.5, 1.0} 은 라벨 안 0.5° 블록 5겹 교차검증
              (map_alaska_v1.fit_r1 = h54 RUnit.lam_cv; 블록이 2개 미만이면 0.25 와 표지). 씨앗 키 (대상, r, split 0, n, draw 0), 전량은 n −1.
산출  data/processed/map_<region>/label_sequence_v1/pred_n<N>.csv.gz(cell_id, pred, p1, corr), label_sequence_v1_meta.json
실행  nice -n 10 python3 scripts/2_evaluation/label_sequence_v1.py --region lena --threads 4
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4" if __name__ == "__main__" else os.environ.get(_v, "4")

import argparse                                                                                         # noqa: E402
import json                                                                                             # noqa: E402
import resource                                                                                         # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "2_evaluation"))
import map_alaska_v1 as MA                                                                              # noqa: E402

REGIONS = {
    "lena": dict(target="Lena", grid="data/processed/map_lena/lena_grid_x25_v1.csv.gz", out="data/processed/map_lena/label_sequence_v1"),
    "alaska": dict(target="Alaska", grid="data/processed/map_alaska/alaska_grid_x25_v1.csv.gz", out="data/processed/map_alaska/label_sequence_v1"),
}
N_SEQ = (0, 10, 40, 160, -1)                 # −1 = 전량
SEED = 0


def n_tag(n: int) -> str:
    return "all" if n < 0 else str(int(n))


def dispersed_order(blk, seed: int = SEED) -> np.ndarray:
    """중첩 블록 분산 순서. 블록 순열 → 블록 안 셀 순열 → 돌아가며 한 셀씩(h40.draw_blocks 규칙, 순열은 한 번)."""
    blk = np.asarray(blk)
    rng = np.random.RandomState(seed)
    ub = np.unique(blk)
    order_b = [int(j) for j in rng.permutation(len(ub))]
    pools = {j: list(rng.permutation(np.where(blk == ub[j])[0])) for j in order_b}
    out, active = [], order_b
    while active:
        nxt = []
        for j in active:
            if pools[j]:
                out.append(int(pools[j].pop()))
                nxt.append(j)
        active = nxt
    out = np.array(out, int)
    assert len(out) == len(blk) and len(np.unique(out)) == len(blk)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="라벨 수별 지도 수열(P0 → P1 → R1)")
    ap.add_argument("--region", required=True, choices=list(REGIONS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    R = REGIONS[args.region]
    out = ROOT / R["out"]
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    H54, a = MA.load_h54(args.threads)
    F = list(H54.FEATS)
    df, t_idx, parent, E0, comp = MA.region_context(H54, a, R["target"])
    assert np.isfinite(df.y.values[t_idx]).all() and np.isfinite(df.s.values[t_idx]).all()
    blk = df.block.values[t_idx]
    order = dispersed_order(blk)
    t0 = time.time()
    grid = pd.read_csv(ROOT / R["grid"], dtype={"cell_id": str}, low_memory=False)
    t_grid = round(time.time() - t0, 1)
    XB, sB = grid[F].values.astype(np.float32), grid.e5_sqrt_tdd.values.astype(float)
    shown = grid.gray.values == 0
    u = MA.make_unit(H54, a, R["target"], parent, df, t_idx, XB, sB, grid.lat.values, grid.lon.values, E0)
    print(f"[data] {R['target']} 라벨 {len(t_idx):,} · 블록 {len(np.unique(blk))} · E0 {E0:.4f} · 격자 {len(grid):,}(표시 {int(shown.sum()):,}) · {t_grid}s",
          flush=True)

    preds, steps = {}, []
    for n in N_SEQ:
        t0 = time.time()
        sel = np.sort(order[:len(t_idx)] if n < 0 else order[:n])
        rec = dict(n=n_tag(n), n_used=int(len(sel)), n_blocks=int(len(np.unique(blk[sel]))) if len(sel) else 0,
                   cells_loc_id=[str(v) for v in df.loc_id.values[t_idx[sel]]], cells_A_index=[int(v) for v in sel])
        if n == 0:
            method, E_n, E_ls, lam, K, flag, gsd = "P0", float(E0), float("nan"), 0.0, 0, "", 0.0
            p1 = E0 * sB
            pred = p1.copy()
        elif n == 10:
            E_n, E_ls = u.coefs(sel)
            method, lam, K, flag, gsd = "P1", 0.0, 0, "", 0.0
            p1 = E_n * sB
            pred = p1.copy()
        else:
            r = MA.fit_r1(H54, u, sel, MA.N_ALL if n < 0 else int(n), MA.D_OUTER, preds=[u.c.XB])
            method, E_n, E_ls, lam, K, flag, gsd = "R1", r["E1"], r["E_ls"], r["lam"], r["K"], r["flag"], r["g_seed_sd"]
            p1 = E_n * sB
            pred = p1 + lam * r["g"]
        pred = np.where(np.isfinite(pred), pred, np.nan)
        tag = n_tag(n)
        path = out / f"pred_n{tag}.csv.gz"
        pd.DataFrame(dict(cell_id=grid.cell_id.values, pred=pred, p1=p1, corr=pred - p1)).to_csv(
            path, index=False, compression=dict(method="gzip", mtime=0), float_format="%.6g")
        v = pred[shown & np.isfinite(pred)]
        rec.update(method=method, E0=float(E0), E_ls=float(E_ls), E_n=float(E_n), lam=float(lam), lam_cv_folds=int(K), lam_cv_flag=str(flag),
                   g_seed_sd_mean=float(gsd), fit_s=round(time.time() - t0, 1), file=MA.file_info(path),
                   shown_stats=dict(n=int(len(v)), p01=float(np.percentile(v, 1)), p50=float(np.median(v)), p99=float(np.percentile(v, 99)),
                                    mean=float(v.mean())))
        preds[tag] = pred
        steps.append(rec)
        print(f"[n={tag}] {method} · 셀 {len(sel)} · 블록 {rec['n_blocks']} · E_ls {E_ls:.4f} · E_n {E_n:.4f} · λ {lam} (K {K} {flag or '-'}) · "
              f"중앙값 {rec['shown_stats']['p50']:.1f} cm · {rec['fit_s']}s", flush=True)

    ref = preds["all"]
    for rec in steps:
        d = (preds[rec["n"]] - ref)[shown]
        d = d[np.isfinite(d)]
        rec["change_vs_all_shown"] = dict(mean=float(d.mean()), mean_abs=float(np.abs(d).mean()), p01=float(np.percentile(d, 1)),
                                          p99=float(np.percentile(d, 99)), rmsd=float(np.sqrt(np.mean(d ** 2))))
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/2_evaluation/label_sequence_v1.py",
                script_sha256=MA.sha256_file(Path(__file__)), region=args.region, target=R["target"], seed=SEED,
                draw_rule="nested block-dispersed order (blocks permuted by RandomState(0), cells within block permuted, round-robin); first n of one order",
                order_A_index=[int(v) for v in order],
                reused=dict(map_alaska_v1=MA.file_info(ROOT / "scripts/2_evaluation/map_alaska_v1.py"),
                            h54=MA.file_info(ROOT / "scripts/3_deep_learning/h54_workflow.py")),
                labels=dict(n=int(len(t_idx)), n_blocks=int(len(np.unique(blk))), data=MA.file_info(MA.PROC / "fidelity_base_v3.csv"),
                            regions=pd.Series(df.region.values[t_idx]).value_counts().to_dict()),
                model=dict(E0=float(E0), E0_source=comp, kappa=float(H54.KAPPA), lam_grid=list(H54.LAMS), seeds=list(u.seeds),
                           learner="catboost_lo(h40.cb_fit: 200, depth 3, lr 0.05, l2 3)", features=F, threads=int(a.threads),
                           cv_seed_key="(target, r, split 0, n, draw 0); all → n −1"),
                grid=dict(file=MA.file_info(ROOT / R["grid"]), n_cells=int(len(grid)), n_shown=int(shown.sum())),
                steps=steps, elapsed_s=round(time.time() - t_start, 1),
                max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                fits=dict(n=dict(u.F.n), sec={k: round(v, 1) for k, v in u.F.sec.items()}, fail=dict(u.F.fail), errors=list(u.F.errors)))
    (out / "label_sequence_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[done] {out} · {meta['elapsed_s']}s · RSS {meta['max_rss_mb']:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
