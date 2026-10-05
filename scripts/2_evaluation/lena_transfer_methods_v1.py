"""showpiece · 레나 델타 전이(라벨 0개) 방법별 1 km 지도와 라벨 셀 예측. 등록된 전이 실험(LG, h40/h54)과 같은 함수·설정으로 원천 풀에만 적합한다.

설정  h54.build_tctx("Lena", "x", split) = h40.build_ctx: 원천 = 레나 제외 다른 지역 전부(100 km 버퍼), A = 라벨 반(0.5° 블록 절반), B = 채점 반(eval_mask).
방법  P0  E0·s(E0 = 원천 최소제곱 계수)
      D0  직접 ML: catboost_lo(200회, 깊이 3, 학습률 0.05)를 원천 행 (X_src, y_src)에 적합, seed 0·1 평균
      D1  물리 유사라벨 증강: 원천 행 + 유사라벨 행(대상 A 반 셀에서 원천 행 수 × 10 개 복원 추출, 값 E0·s; h42.ps_index, 씨앗 키 lg-ps), seed 0·1 평균
      라벨 0개이므로 레나 관측값은 어떤 적합에도 쓰이지 않는다. 라벨 셀 3,037개 모두가 검정점이다.
산출  data/processed/map_lena/transfer_methods_v1/pred_grid_split<S>.csv.gz(cell_id, P0, D0, D1), pred_labels_split<S>.csv(loc_id, lat, lon, block, half, y, P0, D0, D1),
      transfer_methods_v1_meta.json(분할별 RMSE: 등록 채점 셀(B 반), 전체 라벨 셀; seed 별·seed 평균)
실행  nice -n 10 python3 scripts/2_evaluation/lena_transfer_methods_v1.py --threads 4 [--fig-split 1] [--splits 1,2,3,4,5]
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

OUT = ROOT / "data" / "processed" / "map_lena" / "transfer_methods_v1"
GRID = ROOT / "data" / "processed" / "map_lena" / "lena_grid_x25_v1.csv.gz"
TARGET = "Lena"


def rmse(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fig-split", type=int, default=1)
    ap.add_argument("--splits", default="1,2,3,4,5")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    H54, a = MA.load_h54(args.threads)
    HA, D = H54.get_data(a)
    df = D.df
    F = list(H54.FEATS)
    t_idx = D.target_idx(TARGET)
    XL = df[F].values[t_idx].astype(np.float32)
    yL, sL = df.y.values[t_idx].astype(float), df.s.values[t_idx].astype(float)
    grid = pd.read_csv(GRID, dtype={"cell_id": str}, low_memory=False)
    XG, sG = grid[F].values.astype(np.float32), grid.e5_sqrt_tdd.values.astype(float)
    info = D.split_structure(TARGET)
    splits = [int(v) for v in args.splits.split(",")]
    per_split, timing = [], {}
    for sp in splits:
        st = info[sp]
        t0 = time.time()
        c = H54.build_tctx(a, TARGET, "x", sp)
        u = H54.TUnit(a, c, TARGET)
        sel = np.zeros(0, int)
        E0 = float(c.E0)
        A_loc = set(df.loc_id.values[H54.half_split_blocks(df, t_idx, sp)[0]].tolist())
        preds_g = {"P0": E0 * sG}
        preds_l = {"P0": E0 * sL}
        preds_B = {"P0": E0 * c.sB}
        seed_rmse = {}
        for m in ("D0", "D1"):
            pg, pl, pb = [], [], []
            for seed in u.seeds:
                if m == "D0":
                    out = u.fit(H54.LO, "D0", lambda: u.rows_D(sel), u.nsrc, seed, [XG, XL, c.XB], 0, 0, sel)
                else:
                    out = u.fit(H54.LO, "D1", lambda: u.rows_D(sel, seed, E0), u.nsrc + u.n_ps, seed, [XG, XL, c.XB], 0, 0, sel)
                g_, l_, b_ = [np.asarray(v, float) for v in out]
                pg.append(g_); pl.append(l_); pb.append(b_)
                seed_rmse[f"{m}|seed{seed}"] = dict(B=rmse(b_, c.yB), all_labels=rmse(l_, yL))
            preds_g[m] = np.mean(pg, axis=0); preds_l[m] = np.mean(pl, axis=0); preds_B[m] = np.mean(pb, axis=0)
        rec = dict(split=sp, valid=bool(st["valid"]), dup_of=int(st["dup_of"]), n_A=int(st["n_A"]), nb_A=int(st["nb_A"]), n_eval=int(st["n_eval"]),
                   nb_eval=int(st["nb_eval"]), n_src=int(u.nsrc), n_pseudo=int(u.n_ps), E0=E0,
                   rmse_B_seedmean={m: rmse(preds_B[m], c.yB) for m in preds_B}, rmse_all_labels_seedmean={m: rmse(preds_l[m], yL) for m in preds_l},
                   rmse_per_seed=seed_rmse,
                   rmse_A_half_seedmean={m: rmse(preds_l[m][[lid in A_loc for lid in df.loc_id.values[t_idx]]],
                                                 yL[[lid in A_loc for lid in df.loc_id.values[t_idx]]]) for m in preds_l},
                   bias_all_labels={m: float(np.mean(preds_l[m] - yL)) for m in preds_l},
                   seconds=round(time.time() - t0, 1))
        per_split.append(rec)
        half = np.array(["A" if lid in A_loc else "B" for lid in df.loc_id.values[t_idx]])
        lab = pd.DataFrame(dict(loc_id=df.loc_id.values[t_idx], lat=df.lat.values[t_idx], lon=df.lon.values[t_idx], block=df.block.values[t_idx],
                                half=half, y=yL, s=sL, P0=preds_l["P0"], D0=preds_l["D0"], D1=preds_l["D1"]))
        lab.to_csv(OUT / f"pred_labels_split{sp}.csv", index=False, float_format="%.6g")
        if sp == args.fig_split:
            pd.DataFrame(dict(cell_id=grid.cell_id.values, P0=preds_g["P0"], D0=preds_g["D0"], D1=preds_g["D1"])).to_csv(
                OUT / f"pred_grid_split{sp}.csv.gz", index=False, compression=dict(method="gzip", mtime=0), float_format="%.6g")
        print(f"[split {sp}] valid {st['valid']} · A {st['n_A']}셀/{st['nb_A']}블록 · B 채점 {st['n_eval']}셀/{st['nb_eval']}블록 · 유사라벨 {u.n_ps:,} · "
              f"RMSE(B, seed 평균) " + " ".join(f"{m} {v:.2f}" for m, v in rec["rmse_B_seedmean"].items()) + " · RMSE(라벨 전체) "
              + " ".join(f"{m} {v:.2f}" for m, v in rec["rmse_all_labels_seedmean"].items()) + f" · {rec['seconds']}s", flush=True)

    # 등록 곡선과 비교 가능한 합동 값: 유효 분할 × seed 의 SSE 합(세포 가중)
    pooled = {}
    for m in ("P0", "D0", "D1"):
        sse, cnt = 0.0, 0
        for rec, sp in zip(per_split, splits):
            if not rec["valid"]:
                continue
            keys = [k for k in rec["rmse_per_seed"] if k.startswith(m)] if m != "P0" else []
            vals = [rec["rmse_per_seed"][k]["B"] for k in keys] if keys else [rec["rmse_B_seedmean"]["P0"]]
            for v in vals:
                sse += v ** 2 * rec["n_eval"]; cnt += rec["n_eval"]
        pooled[m] = float(np.sqrt(sse / cnt)) if cnt else float("nan")
    reg = {}
    try:
        cur = pd.read_csv(ROOT / "results/rescale_lg/data/processed/lg/lg_curve.csv")
        q = cur[(cur.target == TARGET) & (cur["mode"] == "x") & (cur.n == 0) & (cur.learner.isin(["none", "catboost_lo"]))]
        reg = {f"{r.method}|{r.learner}|lam{r.lam}": dict(rmse=float(r.rmse), rmse_beq=float(r.rmse_beq), n_splits_valid=int(r.n_splits_valid))
               for r in q.itertuples() if r.method in ("P0", "D0", "D1")}
    except Exception as e:                                                   # noqa: BLE001
        reg = dict(error=str(e))
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/2_evaluation/lena_transfer_methods_v1.py",
                script_sha256=MA.sha256_file(Path(__file__)), target=TARGET, mode="x", fig_split=args.fig_split, splits=splits,
                recipe=dict(P0="E0·s", D0="catboost_lo on source rows (X_src, y_src), seeds 0·1 averaged",
                            D1="source rows + round(10 × n_src) pseudo rows drawn with replacement from the A-half cells of the split (h42.ps_index, seed key lg-ps), "
                               "value E0·s; seeds 0·1 averaged", learner="catboost_lo(h40.cb_fit: 200, depth 3, lr 0.05, l2 3)", r_pseudo=float(H54.R_PS)),
                labels=dict(n=int(len(t_idx)), n_blocks=int(len(np.unique(df.block.values[t_idx]))), data=MA.file_info(MA.PROC / "fidelity_base_v3.csv")),
                grid=dict(file=MA.file_info(GRID), n_cells=int(len(grid)), n_shown=int((grid.gray.values == 0).sum())),
                reused=dict(h54=MA.file_info(ROOT / "scripts/3_deep_learning/h54_workflow.py"), h40=MA.file_info(ROOT / "scripts/3_deep_learning/h40_label_grid.py"),
                            h42=MA.file_info(ROOT / "scripts/3_deep_learning/h42_label_grid_ext.py")),
                per_split=per_split, pooled_rmse_B_valid_splits_per_seed_keys=pooled, registered_lg_curve_n0=reg,
                elapsed_s=round(time.time() - t_start, 1), max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                threads=int(a.threads))
    (OUT / "transfer_methods_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[pooled B, valid splits × seeds] {pooled} · registered {reg} · {meta['elapsed_s']}s · RSS {meta['max_rss_mb']:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
