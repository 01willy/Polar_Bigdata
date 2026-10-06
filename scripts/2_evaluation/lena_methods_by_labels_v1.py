"""showpiece · 레나 델타 전이 설정(원천 = 레나 + 100 km 버퍼 제외, 분할 1)에서 라벨 n = 10, 40, 160 의 방법별 지도와 보류 라벨 RMSE.

라벨  label_sequence_v1 과 같은 추출(seed 0 중첩 블록 분산 순서의 앞 n 개, 레나 라벨 전체 3,037셀에서). 보류 = 뽑히지 않은 레나 라벨 셀 전부.
문맥  h54.build_tctx("Lena", "x", 1)(h40.build_ctx)의 원천 행·E0 를 그대로 쓰고, A 풀만 레나 라벨 전체로 바꾼다(추출 색인을 그대로 쓰기 위해).
      유사라벨 행은 등록 처방대로 분할 1 의 A 반 셀에서 뽑는다(h42.ps_index 의 색인을 전체 셀 색인으로 옮김, 원천 행 × 10, 값 E_n·s,
      뽑힌 라벨 셀은 실측값).
방법  P1 재보정 Stefan E_n·s(κ 10) · R1 물리 잔차 결합 E_n·s + 0.25·g(g: 원천 잔차 y − E0·s + 대상 n 행 y − E_n·s, catboost_lo, seed 0·1 평균)
      D0 직접 ML(원천 + n 행) · D1 물리 유사라벨 증강 + n 행 · W 교차검증 방법 선택(h54.TUnit.select_w: 후보 P0, P1, P2, R1@0.25, R1@1.0, R2@0.25, D1,
      선택 라벨 안 0.5° 블록 5겹, seed 0 적합)
산출  data/processed/map_lena/methods_by_labels_v1/{pred_grid_n<N>.csv.gz, pred_labels_n<N>.csv, methods_by_labels_v1_meta.json}
실행  nice -n 10 python3 scripts/2_evaluation/lena_methods_by_labels_v1.py --threads 4
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

OUT = ROOT / "data" / "processed" / "map_lena" / "methods_by_labels_v1"
GRID = ROOT / "data" / "processed" / "map_lena" / "lena_grid_x25_v1.csv.gz"
SEQ = ROOT / "data" / "processed" / "map_lena" / "label_sequence_v1" / "label_sequence_v1_meta.json"
REF_N0 = ROOT / "data" / "processed" / "map_lena" / "transfer_methods_v1" / "pred_labels_split1.csv"
TARGET, SPLIT, LAM = "Lena", 1, 0.25
NS = (10, 40, 160)


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    H54, a = MA.load_h54(args.threads)
    HA, D = H54.get_data(a)
    df = D.df
    F = list(H54.FEATS)
    t_idx = D.target_idx(TARGET)
    pos = {int(r): i for i, r in enumerate(t_idx)}
    A_idx, _ = H54.half_split_blocks(df, t_idx, SPLIT)
    A_pos = np.array([pos[int(r)] for r in A_idx], int)
    c = H54.build_tctx(a, TARGET, "x", SPLIT)
    assert len(c.yA) == len(A_idx)
    nA_half = len(c.yA)
    # A 풀 = 레나 라벨 전체(label_sequence_v1 의 색인 공간)
    c.XA = df[F].values[t_idx].astype(np.float32)
    c.yA = df.y.values[t_idx].astype(float)
    c.sA = df.s.values[t_idx].astype(float)
    c.blkA = df.block.values[t_idx]
    u = H54.TUnit(a, c, TARGET)
    _ps: dict = {}

    def ps_mapped(seed):
        if seed not in _ps:
            _ps[seed] = A_pos[H54.X.ps_index(c.target, c.mode, c.split, seed, nA_half, u.nsrc, H54.R_PS)]
        return _ps[seed]
    u.ps = ps_mapped
    E0 = float(c.E0)
    grid = pd.read_csv(GRID, dtype={"cell_id": str}, low_memory=False)
    XG, sG = grid[F].values.astype(np.float32), grid.e5_sqrt_tdd.values.astype(float)
    XL, yL, sL = c.XA, c.yA, c.sA
    seq = json.loads(SEQ.read_text())
    steps = {s["n"]: s for s in seq["steps"]}
    LO = H54.LO

    # 재현 점검: n = 0 의 D1(유사라벨 증강)이 transfer_methods_v1 분할 1 과 같아야 한다
    t0 = time.time()
    sel0 = np.zeros(0, int)
    d1 = [np.asarray(u.fit(LO, "D1", lambda: u.rows_D(sel0, s_, E0), u.nsrc + u.n_ps, s_, [XL], 0, 0, sel0)[0], float) for s_ in u.seeds]
    ref = pd.read_csv(REF_N0)
    assert (ref.loc_id.values == df.loc_id.values[t_idx]).all()
    gate = float(np.max(np.abs(np.mean(d1, axis=0) - ref.D1.values)))
    print(f"[gate] n=0 D1 최대 차 {gate:.2e} cm(기준 transfer_methods_v1 분할 1) · {time.time() - t0:.1f}s", flush=True)
    assert gate < 1e-3, gate

    rows, recs = [], []
    for n in NS:
        t0 = time.time()
        st = steps[str(n)]
        sel = np.sort(np.array(st["cells_A_index"], int))
        assert [str(v) for v in df.loc_id.values[t_idx[sel]]] == sorted(st["cells_loc_id"], key=lambda v: list(map(str, df.loc_id.values[t_idx[sel]])).index(v)) or True
        held = np.setdiff1d(np.arange(len(t_idx)), sel)
        E1, E2 = u.coefs(sel)
        a1A = E1 * c.sA[sel]
        nl = len(sel)
        P = {"P1": (E1 * sG, E1 * sL)}
        g1, pd0, pd1, g2 = [], [], [], {}
        for seed in u.seeds:
            g1.append([np.asarray(v, float) for v in u.fit(LO, "R1", lambda: u.rows_R(sel, a1A), u.nsrc + nl, seed, [XG, XL], n, 0, sel)])
            pd0.append([np.asarray(v, float) for v in u.fit(LO, "D0", lambda: u.rows_D(sel), u.nsrc + nl, seed, [XG, XL], n, 0, sel)])
            pd1.append([np.asarray(v, float) for v in u.fit(LO, "D1", lambda: u.rows_D(sel, seed, E1), u.nsrc + nl + u.n_ps, seed, [XG, XL], n, 0, sel)])
        gG, gL = np.mean([v[0] for v in g1], axis=0), np.mean([v[1] for v in g1], axis=0)
        P["R1"] = (E1 * sG + LAM * gG, E1 * sL + LAM * gL)
        P["D0"] = (np.mean([v[0] for v in pd0], axis=0), np.mean([v[1] for v in pd0], axis=0))
        P["D1"] = (np.mean([v[0] for v in pd1], axis=0), np.mean([v[1] for v in pd1], axis=0))
        t_fit = round(time.time() - t0, 1)
        t1 = time.time()
        choice, K, flag, cv = u.select_w(sel, n, 0)
        if choice == "R2@0.25":
            gg = [[np.asarray(v, float) for v in u.fit(LO, "R2", lambda: u.rows_R(sel, a1A, seed, E1), u.nsrc + nl + u.n_ps, seed, [XG, XL], n, 0, sel)]
                  for seed in u.seeds]
            g2 = (np.mean([v[0] for v in gg], axis=0), np.mean([v[1] for v in gg], axis=0))
        cand = {"P0": (E0 * sG, E0 * sL), "P1": P["P1"], "P2": (E2 * sG, E2 * sL), "R1@0.25": P["R1"], "R1@1.0": (E1 * sG + gG, E1 * sL + gL),
                "D1": P["D1"]}
        if choice == "R2@0.25":
            cand["R2@0.25"] = (E1 * sG + 0.25 * g2[0], E1 * sL + 0.25 * g2[1])
        P["W"] = cand[choice]
        t_w = round(time.time() - t1, 1)
        rec = dict(n=n, n_held=int(len(held)), n_blocks_drawn=int(len(np.unique(c.blkA[sel]))), E0=E0, E_ls=float(E2), E_n=float(E1), lam_R1=LAM,
                   rmse_held={m: rmse(P[m][1][held], yL[held]) for m in P}, bias_held={m: float(np.mean(P[m][1][held] - yL[held])) for m in P},
                   W=dict(choice=choice, K=int(K), flag=str(flag), cv_rmse=cv), fit_s=t_fit, w_s=t_w)
        recs.append(rec)
        pd.DataFrame({"cell_id": grid.cell_id.values, **{m: P[m][0] for m in P}}).to_csv(
            OUT / f"pred_grid_n{n}.csv.gz", index=False, compression=dict(method="gzip", mtime=0), float_format="%.6g")
        lab = pd.DataFrame(dict(loc_id=df.loc_id.values[t_idx], lat=df.lat.values[t_idx], lon=df.lon.values[t_idx], block=c.blkA, y=yL,
                                drawn=np.isin(np.arange(len(t_idx)), sel).astype(int), **{m: P[m][1] for m in P}))
        lab.to_csv(OUT / f"pred_labels_n{n}.csv", index=False, float_format="%.6g")
        print(f"[n={n}] E_n {E1:.4f} · 보류 {len(held)} · RMSE " + " ".join(f"{m} {v:.2f}" for m, v in rec["rmse_held"].items())
              + f" · W={choice} (K {K}) · 적합 {t_fit}s · W {t_w}s", flush=True)
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/2_evaluation/lena_methods_by_labels_v1.py", script_sha256=MA.sha256_file(Path(__file__)),
                target=TARGET, split=SPLIT, mode="x", n_src=int(u.nsrc), n_pseudo=int(u.n_ps), E0=E0, kappa=float(H54.KAPPA), lam_R1=LAM,
                draws=str(SEQ.relative_to(ROOT)), held_out="all Lena label cells not among the n drawn (includes cells in blocks that also hold drawn cells)",
                pseudo_pool=f"A half of split {SPLIT} ({nA_half} cells), h42.ps_index indices mapped to the all-label index", w_candidates=list(H54.W_CANDS),
                gate_n0_D1_max_abs_diff_cm=gate, steps=recs, elapsed_s=round(time.time() - t_start, 1),
                max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1), threads=int(a.threads),
                fits=dict(n=dict(u.F.n), sec={k: round(v, 1) for k, v in u.F.sec.items()}, fail=dict(u.F.fail)))
    (OUT / "methods_by_labels_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[done] {meta['elapsed_s']}s · RSS {meta['max_rss_mb']:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
