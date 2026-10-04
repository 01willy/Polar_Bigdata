"""지도 과제(MAP) · 레나 1 km 지도의 최종 방법(패널 (c), 라벨 전량 R1) 보조 산출. 기존 레나 산출(lena_pred_v1.*)은 고치지 않고 새 파일만 쓴다.

1) 외삽 범주(최종 방법의 학습 행 기준). h49.extrap_flags 를 그대로 쓰되 기준 행을 원천 셀(h49 의 n_extrap 기준) 대신 최종 방법 R1 의 잔차 모형
   학습 행(LG 원천 셀 ∪ 레나 라벨 전량)으로 둔다. h49 의 원천 기준으로는 표시 셀 전부가 범주 3 이다(e5_maat, e5_fdd, e5_tcold 가 모든 셀에서 밖).
2) 90 % 구간(상수 폭 분할 등각). 레나 라벨을 0.5° 블록 5겹(h42.cv_folds_of, 씨앗 키 (Lena, x, split 0, n −1, draw 0))으로 나누고 겹마다
   학습 겹 라벨 + 원천으로 h49.fit_predict('R1')(λ 0.25, seed 0·1, h40.cb_fit)를 적합해 블록 밖 예측을 얻는다. q = |y − ŷ_oof| 의
   ⌈(n + 1)·0.9⌉ 번째 작은 값. 구간 = [max(ŷ − q, 0), ŷ + q]. P1 의 블록 밖 예측도 함께 적는다.
산출  data/processed/map_lena/lena_final_extras_v1.csv.gz(cell_id, 외삽 열), lena_final_extras_v1_meta.json, lena_cv_oof_v1.csv
실행  nice -n 10 python3 scripts/2_evaluation/map_lena_final_extras_v1.py --threads 4
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

OUT = ROOT / "data" / "processed" / "map_lena"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    t_start = time.time()
    H54, a = MA.load_h54(args.threads)
    h49 = MA._load("h49_transfer_map", "scripts/2_evaluation/h49_transfer_map.py")
    _, D = H54.get_data(a)
    df = D.df
    t_idx, parent, src_idx, comp = D.source_idx("Lena", "x")
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    src_i = src_idx[ok]
    F = list(H54.FEATS)
    cont = [c for c in F if c != "cci_valid"]

    # 1) 외삽(최종 방법의 학습 행 기준)
    pred = pd.read_csv(OUT / "lena_pred_v1.csv.gz", dtype={"cell_id": str})
    gx = pd.read_csv(OUT / "lena_grid_x25_v1.csv.gz", dtype={"cell_id": str})
    assert (gx.cell_id.values == pred.cell_id.values).all()
    shown = pred.gray.values == 0
    Xg = gx[cont].values.astype(float)
    ex_f, info_f = h49.extrap_flags(df[cont].values[np.r_[src_i, t_idx]].astype(float), Xg, cont, shown)
    ex_s, info_s = h49.extrap_flags(df[cont].values[src_i].astype(float), Xg, cont, shown)
    assert (ex_s.n_extrap_out.values == pred.n_extrap_out.values).all(), "h49 원천 기준 재현 실패"
    ex_l, info_l = h49.extrap_flags(df[cont].values[t_idx].astype(float), Xg, cont, shown)
    o = pd.DataFrame(dict(cell_id=pred.cell_id.values, n_extrap_out_final=ex_f.n_extrap_out.values, n_x25_missing=ex_f.n_x25_missing.values,
                          n_extrap_total_final=ex_f.n_extrap_total.values, extrap_cat_final=ex_f.extrap_cat.values,
                          extrap_flag_final=(ex_f.n_extrap_out.values >= 1).astype(int),
                          n_extrap_out_lenalabels=ex_l.n_extrap_out.values, extrap_cat_lenalabels=ex_l.extrap_cat.values))
    o.to_csv(OUT / "lena_final_extras_v1.csv.gz", index=False, compression=dict(method="gzip", mtime=0))

    # 2) 블록 교차검증의 블록 밖 예측과 상수 폭 등각 q
    rows = lambda idx: h49.rows_from_df(df.iloc[idx], F, "y")                                                   # noqa: E731
    src = h49.Rows(df[F].values[src_i].astype(np.float32), df.s.values[src_i].astype(float),
                   {k: df[c].values[src_i].astype(float) for k, c in h49.ANCHOR_COL.items()}, df.y.values[src_i].astype(float),
                   df.macro.values[src_i].astype(str))
    blk = df.block.values[t_idx]
    fid, K, flag = H54.X.cv_folds_of(blk, "Lena", "x", 0, -1, 0)
    Fit = h49.Fitter(threads=int(a.threads))
    y = df.y.values[t_idx].astype(float)
    oof_r1 = np.full(len(t_idx), np.nan); oof_p1 = np.full(len(t_idx), np.nan)
    folds = []
    t0 = time.time()
    for j in range(int(K)):
        m = fid == j
        lab, te = rows(t_idx[~m]), rows(t_idx[m])
        (pr,), inf = h49.fit_predict("R1", src, lab, [te], Fit, tag=f"cv{j}")
        (pp,), infp = h49.fit_predict("P1", src, lab, [te], Fit, tag=f"cv{j}")
        oof_r1[m], oof_p1[m] = pr, pp
        folds.append(dict(fold=j, n_train=int((~m).sum()), n_test=int(m.sum()), nb_test=int(len(np.unique(blk[m]))), E_n=inf["E_n"]))
    t_cv = round(time.time() - t0, 1)
    q = MA.conformal_q(np.abs(oof_r1 - y))
    pd.DataFrame(dict(loc_id=df.loc_id.values[t_idx], lat=df.lat.values[t_idx], lon=df.lon.values[t_idx], block=blk, fold=fid, y=y,
                      s=df.s.values[t_idx], oof_p1=oof_p1, oof_r1=oof_r1)).to_csv(OUT / "lena_cv_oof_v1.csv", index=False, float_format="%.6g")
    r1 = pred.pred_c.values.astype(float)
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/2_evaluation/map_lena_final_extras_v1.py",
                script_sha256=MA.sha256_file(Path(__file__)), inputs=dict(pred=MA.file_info(OUT / "lena_pred_v1.csv.gz"),
                                                                        grid=MA.file_info(OUT / "lena_grid_x25_v1.csv.gz")),
                extrap=dict(rule="h49.extrap_flags(0.5–99.5 백분위, cci_valid 제외 24열), 기준 행 = 원천 셀 ∪ 레나 라벨 전량(최종 방법 R1 의 학습 행)",
                            n_ref_rows=int(len(src_i) + len(t_idx)), final=info_f, source_only=dict(cat_counts_shown=info_s["cat_counts_shown"], top3=info_s["top3"]),
                            lena_labels_only=dict(cat_counts_shown=info_l["cat_counts_shown"], top3=info_l["top3"]),
                            frac_flag_final_shown=float((ex_f.n_extrap_out.values[shown] >= 1).mean()),
                            frac_flag_source_shown=float((ex_s.n_extrap_out.values[shown] >= 1).mean())),
                interval=dict(kind="constant-width split-conformal on 0.5° block 5-fold CV residuals of the final method(h49 R1, λ 0.25, 라벨 전량 적합과 같은 절차)",
                              level=0.9, q=q, width=2 * q, K=int(K), flag=str(flag or ""), folds=folds, n=int(len(y)),
                              rmse_r1=float(np.sqrt(np.mean((oof_r1 - y) ** 2))), rmse_p1=float(np.sqrt(np.mean((oof_p1 - y) ** 2))),
                              bias_r1=float(np.mean(oof_r1 - y)), bias_p1=float(np.mean(oof_p1 - y)),
                              n_lo_clipped_shown=int((np.isfinite(r1) & shown & (r1 - q < 0)).sum()), cv_seconds=t_cv,
                              h49_interval_note="lena_pred_v1 의 lo90_a·hi90_a 는 패널 (a)(P0, 표지 없음)의 원천 지역 보정 구간이다(q 0.707, 로그 비, 폭 ∝ 예측)"),
                source=comp, fits=dict(n=int(Fit.n_fit), sec=round(Fit.sec, 1)), threads=int(a.threads), elapsed_s=round(time.time() - t_start, 1),
                max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                outputs=dict(extras=MA.file_info(OUT / "lena_final_extras_v1.csv.gz"), oof=MA.file_info(OUT / "lena_cv_oof_v1.csv")))
    (OUT / "lena_final_extras_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[lena] 외삽(최종 학습 행) 표시 셀 비율 {meta['extrap']['frac_flag_final_shown']:.3f} · 범주 {info_f['cat_counts_shown']} · "
          f"q90 {q:.2f} cm · RMSE R1 {meta['interval']['rmse_r1']:.2f} P1 {meta['interval']['rmse_p1']:.2f} · {meta['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
