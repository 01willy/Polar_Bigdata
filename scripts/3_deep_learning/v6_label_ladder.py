# -*- coding: utf-8 -*-
"""v6_label_ladder.py: 라벨 수 사다리(셀-연도 설계, 관측일 정합 강제력). 2026-10-15.

두 실험(사용자 10-15 요청 23: 결론·의의 강화용 추가 실험).
LL  지역 안 라벨 수 사다리. 블록 5겹에서 학습 겹의 라벨을 n개로 줄여 학습하고 시험 겹 전체를 채점한다.
    n ∈ {25, 50, 100, 250, 500, 1000, 2500, 전량}, 방법 P1(재보정 Stefan, n개로 E), D0(직접 학습), D1(물리 유사라벨 r=1), D1x4(r=4),
    F1(물리 입력), R1(물리 + 잔차 λ 0.5). 물리 유사라벨이 라벨이 적을 때 도움이 되는지, 잔차 결합의 이득이 라벨 수에 따라 어떻게 바뀌는지 본다.
TL  새 지역 전이 + 대상 라벨 n개. 대상 지역의 블록 5겹에서 시험 겹을 채점하되, 학습은 '다른 두 지역 전량(원천) + 시험 겹 밖 대상 블록에서 뽑은 n개'.
    n ∈ {0, 25, 50, 100, 250, 500}, 방법 P0(원천 E), P1n(대상 n개로 E; n=0 이면 P0), D0S(원천+n 직접 학습), D0T(n개만 직접 학습),
    R1S(앵커 P1n, 잔차는 원천+n 으로 학습; 원천 행의 잔차는 원천 E 앵커 기준), R1T(앵커 P1n, 잔차는 n개만), D1S(원천+n + 물리 유사라벨).
출력: <out>/<tag>/preds_<exp>_<region>.csv.gz (exp, region, fold, n, seed, method, cell_year_id, loc_id, year, block, y, pred), meta.
실행(Rescale): python3 scripts/3_deep_learning/v6_label_ladder.py --tag ladder_v6 --exp LL,TL --regions Alaska,Lena,Canada --threads 24
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--proc", default="data/processed/v5"); ap.add_argument("--out", default="data/processed/v5/runs"); ap.add_argument("--tag", required=True)
ap.add_argument("--exp", default="LL,TL"); ap.add_argument("--regions", default="Alaska,Lena,Canada"); ap.add_argument("--forcing", default="year_obs")
ap.add_argument("--featset", default="F8TS"); ap.add_argument("--ns-ll", default="25,50,100,250,500,1000,2500,all"); ap.add_argument("--ns-tl", default="0,25,50,100,250,500")
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--k", type=int, default=5); ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--nice", type=int, default=10); ap.add_argument("--learner", default="catboost")
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, str(a.threads))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.nice(a.nice)
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402

OUT = ROOT / a.out / a.tag; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
df = T.load_v5(ROOT / a.proc, products=False)
d_all = T.with_forcing(df, a.forcing); d_all = d_all[T.usable_mask(d_all, None)].reset_index(drop=True)
feats = T.FEATSETS[a.featset]
seeds = [int(s) for s in a.seeds.split(",")]
E_src = {r: T.ls_scale(d_all[d_all.region != r].e5_sqrt_tdd.values, d_all[d_all.region != r].alt_cm.values) for r in T.REGIONS}
print(f"[load] usable {len(d_all):,} E_src {json.dumps({k: round(v, 3) for k, v in E_src.items()})} {time.time()-t0:.0f}s", flush=True)
json.dump(dict(args=vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"), rows=int(len(d_all)), E_src=E_src), open(OUT / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1)


def rows_of(te, pred, **tags):
    o = te[["cell_year_id", "loc_id", "year", "block"]].copy(); o["y"] = te.alt_cm.values; o["pred"] = pred
    for k, v in tags.items():
        o[k] = v
    return o


def X(d):
    return d[feats].values.astype(float)


def residual_fit(tr, anchor_tr, te, anchor_te, seed):
    res = T.fit_learner(a.learner, X(tr), tr.alt_cm.values - anchor_tr, X(te), seed, a.threads)
    return anchor_te + a.lam * res


def run_ll(region):
    d = d_all[d_all.region == region].reset_index(drop=True)
    folds = T.folds_block(d, a.k, 0)
    ns = [None if x == "all" else int(x) for x in a.ns_ll.split(",")]
    out = []; t1 = time.time()
    for f in folds:
        tr_full, te = d.iloc[f.tr].reset_index(drop=True), d.iloc[f.te].reset_index(drop=True)
        for n in ns:
            for s in seeds:
                if n is not None and n < len(tr_full):
                    tr = tr_full.sample(n, random_state=s).reset_index(drop=True); nn = n
                else:
                    if n is not None:
                        continue                                  # n 이 전량보다 크면 건너뜀(전량은 'all' 로 한 번만)
                    tr = tr_full; nn = len(tr_full)
                    if s != seeds[0]:
                        pass                                      # 전량은 학습기 씨앗만 바뀐다
                k = T.fit_phys(tr)
                tags = dict(exp="LL", region=region, fold=f.name, n=nn, n_label=("all" if n is None else n), seed=s)
                out.append(rows_of(te, T.phys_pred("P1", te, k), method="P1", **tags))
                out.append(rows_of(te, T.phys_pred("P0", te, k, E0=E_src[region]), method="P0", **tags))
                for m, kw in (("D0", {}), ("F1", {}), ("D1", dict(r_pseudo=1.0)), ("D1x4", dict(r_pseudo=4.0)), ("R1", dict(lam=a.lam))):
                    mm = "D1" if m == "D1x4" else m
                    pred, _ = T.run_method(mm, tr, te, feats, learner=a.learner, seed=s, threads=a.threads, E0=E_src[region], **kw)
                    out.append(rows_of(te, pred, method=m, **tags))
        print(f"[LL {region} {f.name}] {time.time()-t1:.0f}s", flush=True)
    o = pd.concat(out, ignore_index=True); o.to_csv(OUT / f"preds_LL_{region}.csv.gz", index=False); print(f"[LL {region}] rows {len(o):,} {time.time()-t1:.0f}s", flush=True)


def run_tl(region):
    d = d_all[d_all.region == region].reset_index(drop=True)
    src = d_all[d_all.region != region].reset_index(drop=True)
    folds = T.folds_block(d, a.k, 0)
    ns = [int(x) for x in a.ns_tl.split(",")]
    k_src = T.fit_phys(src); E0 = k_src["E"]
    out = []; t1 = time.time()
    for f in folds:
        pool, te = d.iloc[f.tr].reset_index(drop=True), d.iloc[f.te].reset_index(drop=True)
        for n in ns:
            for s in seeds:
                if n == 0 and s != seeds[0]:
                    continue
                tgt = pool.sample(min(n, len(pool)), random_state=s).reset_index(drop=True) if n > 0 else pool.iloc[0:0]
                tags = dict(exp="TL", region=region, fold=f.name, n=n, n_label=n, seed=s)
                # 물리: 원천 E, 대상 n개로 재보정한 E
                En = T.fit_phys(tgt)["E"] if n >= 3 else E0
                if not np.isfinite(En):
                    En = E0
                out.append(rows_of(te, E0 * te.e5_sqrt_tdd.values, method="P0", **tags))
                out.append(rows_of(te, En * te.e5_sqrt_tdd.values, method="P1n", **tags))
                tr = pd.concat([src, tgt], ignore_index=True)
                # 직접 학습: 원천+n
                out.append(rows_of(te, T.fit_learner(a.learner, X(tr), tr.alt_cm.values, X(te), s, a.threads), method="D0S", **tags))
                # 물리 유사라벨 증강(원천+n + 시험 겹 셀의 물리 값 r=1, 라벨은 En 앵커)
                rng = np.random.RandomState(s); j = rng.choice(len(te), min(len(te), len(tr)), replace=False)
                Xp, yp = X(te)[j], En * te.e5_sqrt_tdd.values[j]
                out.append(rows_of(te, T.fit_learner(a.learner, np.r_[X(tr), Xp], np.r_[tr.alt_cm.values, yp], X(te), s, a.threads), method="D1S", **tags))
                # 잔차 결합(원천+n): 원천 행은 원천 E 앵커, 대상 행은 En 앵커, 시험은 En 앵커
                anc_tr = np.r_[E0 * src.e5_sqrt_tdd.values, En * tgt.e5_sqrt_tdd.values]
                out.append(rows_of(te, residual_fit(tr, anc_tr, te, En * te.e5_sqrt_tdd.values, s), method="R1S", **tags))
                if n >= 25:
                    out.append(rows_of(te, T.fit_learner(a.learner, X(tgt), tgt.alt_cm.values, X(te), s, a.threads), method="D0T", **tags))
                    out.append(rows_of(te, residual_fit(tgt, En * tgt.e5_sqrt_tdd.values, te, En * te.e5_sqrt_tdd.values, s), method="R1T", **tags))
        print(f"[TL {region} {f.name}] {time.time()-t1:.0f}s", flush=True)
    o = pd.concat(out, ignore_index=True); o.to_csv(OUT / f"preds_TL_{region}.csv.gz", index=False); print(f"[TL {region}] rows {len(o):,} {time.time()-t1:.0f}s", flush=True)


for region in a.regions.split(","):
    if "LL" in a.exp.split(","):
        run_ll(region)
    if "TL" in a.exp.split(","):
        run_tl(region)
print(f"[done] {time.time()-t0:.0f}s", flush=True)
