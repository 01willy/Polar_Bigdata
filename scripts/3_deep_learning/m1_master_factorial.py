"""M1: 마스터 요인 실험 하네스 — 세 조건 × 여섯 축을 단일 규약·단일 결과 DB로 (계획 docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md).

E1(e1_unified_factorial.py)을 일반화한다. 축 정의는 src/polar/m1_core.py.

조건별 학습·평가 집합 (대상 t의 셀을 0.5° 블록으로 A/B 2분할; --splits K 로 분할 반복)
  labels : 학습 = D집합 실측 ∪ A 실측              평가 B         (P=none)
  covonly: 학습 = D집합 실측 ∪ A 유사라벨(P≠none)  평가 B
  noinfo : 학습 = D집합 실측만                      평가 전체 셀   (P=none, 앵커는 공변량으로 산출)
  deploy : 학습 = D집합 실측 ∪ 전체 셀 유사라벨     평가 전체 셀   (transductive 배포형, S12 loro 규약)
  알래스카 지역 내: 6-fold 공간블록(labels 조건만, X=x34 허용).

구성 = (dset, xset, anchor, pseudo, r, resid[, iw]).  예측 = anchor + λ·g(x), anchor=none 이면 직접 회귀(λ=1).
축 스크린(--axis): 기본 레시피(★ covonly·alaska·x25·stefan·none·catboost_lo)에서 한 축만 바꾼다.
  --axis anchor : K 전 수준, λ=0(해석적) + resid=catboost_lo λ 스윕
  --axis pseudo : P 전 수준, anchor=none 직접(catboost_lo, r=10) + anchor=stefan 잔차
  --axis model  : G 전 수준 × {직접, stefan 잔차}
  --axis dset   : D 전 수준 (noinfo·covonly)
  --axis xset   : X 전 수준
  --axis iw     : T3 중요도 가중 on/off (catboost_lo·mlp)
  --configs f.json : 임의 구성 목록(상호작용 S-D·재현 게이트용)

산출 data/processed/m1/<tag>_shard{k}.csv, _preds.npz(g·앵커·q05·q95·평가 셀), _meta.json
실행: GPU=6 python3 scripts/3_deep_learning/m1_master_factorial.py --axis model --conds covonly,noinfo --shard 0 --nshard 4
"""
from __future__ import annotations
import os
GPU = os.environ.get("GPU", "6")
assert GPU in {"2", "3", "4", "5", "6", "7", "8", "9"}, f"GPU는 2-9만 (요청 {GPU})"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
# 공유 서버 CPU 보호(2026-09-21): torch/MKL/OMP 기본값이 전 코어(128)를 잡아 load average 200+ 발생 → 프로세스당 4스레드로 제한.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, TRANSFER_MAIN, TRANSFER_DEEP, spatial_block_splits   # noqa: E402
from polar.preprocessing import fold_prep                                                # noqa: E402
from polar.eval_metrics import all_metrics                                               # noqa: E402
from polar.m1_core import (INPUT_SETS, ANCHORS_ALL, PSEUDOS_ALL, MODELS_ALL, NAN_NATIVE, LAM_GRID,   # noqa: E402
                           load_base, eval_mask as _eval_mask, half_split_blocks, train_region_set, fit_coefs,
                           anchor_pred, pseudo_label, fit_model, importance_weights)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="fidelity_base_v3.csv")
ap.add_argument("--soil", default="e5_soil_tdd_v3.csv")
ap.add_argument("--sources", default="F4_direct")
ap.add_argument("--targets", default=",".join(TRANSFER_MAIN))
ap.add_argument("--conds", default="covonly,noinfo,labels,deploy")
ap.add_argument("--alaska", action="store_true", help="알래스카 지역 내 6-fold(labels) 포함")
ap.add_argument("--axis", default="", help="anchor|pseudo|model|dset|xset|iw|base")
ap.add_argument("--configs", default="", help="구성 목록 JSON 파일(list of dict)")
ap.add_argument("--splits", type=int, default=1, help="A/B 분할 반복 수(split_seed 0..K-1; 0=E1 규약)")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--r", type=float, default=10.0)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--min-cells", type=int, default=6)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--tag", default="m1")
ap.add_argument("--exclude-models", default="", help="모델 축에서 제외(쉼표)")
ap.add_argument("--only-models", default="", help="모델 축에서 이 모델만(쉼표)")
ap.add_argument("--eval-all", action="store_true", help="평가 셀 마스크를 y 유효만으로(CCI·토양 도일 결측 셀 포함). 옛 S3 규약 재현용")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"
OUT = PROC / "m1"; OUT.mkdir(exist_ok=True)


def eval_mask(d):
    return d[TARGET].notna().values if args.eval_all else _eval_mask(d)
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
R = args.r
CONDS = [c for c in args.conds.split(",") if c]
TARGETS = [t for t in args.targets.split(",") if t]
SPLITS = list(range(args.splits)) if not args.smoke else [0]
DEFAULT = dict(dset="alaska", xset="x25", anchor="stefan", pseudo="none", r=0.0, resid="catboost_lo", iw=0)


def cfg(**kw):
    c = dict(DEFAULT); c.update(kw)
    c["r"] = float(c.get("r", 0.0))                      # 키 문자열 일관성(분석 스크립트는 float 키를 쓴다)
    c["iw"] = int(c.get("iw", 0))
    if c["pseudo"] == "none":
        c["r"] = 0.0
    elif c["r"] == 0.0:
        c["r"] = R
    return c


def build_configs(axis: str):
    cs = []
    if axis == "base":
        cs = [cfg(), cfg(anchor="none"), cfg(anchor="stefan_cci"), cfg(pseudo="stefan", anchor="none")]
    elif axis == "anchor":
        for a in ANCHORS_ALL:
            if a != "none":
                cs.append(cfg(anchor=a))                       # 해석적(λ=0) + catboost_lo 잔차 λ 스윕
    elif axis == "pseudo":
        for p in PSEUDOS_ALL:
            cs.append(cfg(anchor="none", pseudo=p))            # 직접 회귀(H3 규약)
            if p != "none":
                cs.append(cfg(anchor="stefan", pseudo=p))      # 앵커 + 증강 잔차
    elif axis == "model":
        for m in MODELS_ALL:
            cs.append(cfg(anchor="none", resid=m))
            cs.append(cfg(anchor="stefan", resid=m))
            cs.append(cfg(anchor="stefan_cci", resid=m))
    elif axis == "dset":
        for d in ["alaska", "loro", "loro_main"]:
            cs.append(cfg(dset=d)); cs.append(cfg(dset=d, anchor="none"))
    elif axis == "xset":
        for x in ["x25", "x14", "x34"]:
            cs.append(cfg(xset=x)); cs.append(cfg(xset=x, anchor="none"))
    elif axis == "iw":
        for m in ["catboost_lo", "mlp"]:
            for w in [0, 1]:
                cs.append(cfg(resid=m, iw=w)); cs.append(cfg(resid=m, iw=w, anchor="none"))
    else:
        raise ValueError(axis)
    return cs


if args.configs:
    CONFIGS = [cfg(**c) for c in json.loads(Path(args.configs).read_text())]
else:
    CONFIGS = build_configs(args.axis or "base")
if args.exclude_models:
    CONFIGS = [c for c in CONFIGS if c["resid"] not in args.exclude_models.split(",")]
if args.only_models:
    CONFIGS = [c for c in CONFIGS if c["resid"] in args.only_models.split(",")]
seen, uniq = set(), []
for c in CONFIGS:                                             # 중복 제거(순서 유지)
    key = json.dumps(c, sort_keys=True)
    if key not in seen:
        seen.add(key); uniq.append(c)
CONFIGS = [c for i, c in enumerate(uniq) if i % args.nshard == args.shard]
if args.smoke:
    CONFIGS = CONFIGS[:3]

# ---------------------------------------------------------------- 자료
df = load_base(PROC, base=args.base, soil=args.soil, sources=tuple(args.sources.split(",")))
need_emap = any(c["anchor"].startswith("emap") for c in CONFIGS)
print(f"[data] {args.base} · {len(df):,}셀 · 지역 {df.macro.nunique()} · GPU {GPU} · 구성 {len(CONFIGS)} · 조건 {CONDS} · 분할 {SPLITS}", flush=True)

# ---------------------------------------------------------------- 작업 목록 (cond, target, split, dset)
TASKS = []
for tg in TARGETS:
    t_idx = np.where(df.macro.values == tg)[0]
    if len(t_idx) < args.min_cells:
        print(f"[skip] {tg}: 셀 {len(t_idx)} (< {args.min_cells})", flush=True)
        continue
    for sp in SPLITS:
        A_idx, B_idx = half_split_blocks(df, t_idx, sp)
        evB = B_idx[eval_mask(df.iloc[B_idx])]
        evAll = t_idx[eval_mask(df.iloc[t_idx])]
        for cond in CONDS:
            if cond == "labels":
                TASKS.append(dict(cond=cond, target=tg, split=sp, A=A_idx, ev=evB, pool=A_idx, extra_real=A_idx))
            elif cond == "covonly":
                TASKS.append(dict(cond=cond, target=tg, split=sp, A=A_idx, ev=evB, pool=A_idx, extra_real=None))
            elif cond == "noinfo":
                if sp > 0:
                    continue                                  # 분할 무관
                TASKS.append(dict(cond=cond, target=tg, split=0, A=None, ev=evAll, pool=None, extra_real=None))
            elif cond == "deploy":
                if sp > 0:
                    continue
                TASKS.append(dict(cond=cond, target=tg, split=0, A=None, ev=evAll, pool=t_idx, extra_real=None))
if args.alaska:
    ak_idx = np.where(df.macro.values == "Alaska")[0]
    for fi, (tr, te) in enumerate(spatial_block_splits(df, n_splits=6, sub_idx=ak_idx)):
        if args.smoke and fi >= 2:
            break
        TASKS.append(dict(cond="labels", target="Alaska", split=fi, A=None, ev=te[eval_mask(df.iloc[te])],
                          pool=None, extra_real=None, ak_train=tr))

# ---------------------------------------------------------------- 실행
rows, G, ANCH, Q, EVALS, COEFS = [], {}, {}, {}, {}, {}
t0, n_fit = time.time(), 0
for task in TASKS:
    cond, tg, sp, ev = task["cond"], task["target"], task["split"], task["ev"]
    tkey = f"{cond}|{tg}|{sp}"
    te = df.iloc[ev]
    yte = te[TARGET].values.astype(float)
    EVALS[tkey] = dict(loc_id=te.loc_id.values, y=yte, block=te.block.values)
    for c in CONFIGS:
        if c["pseudo"] != "none" and cond not in ("covonly", "deploy"):
            continue
        if cond in ("noinfo", "deploy") and c["dset"] != "alaska" and args.axis != "dset":
            pass
        if c["xset"] == "x34" and tg != "Alaska":
            continue                                          # SAR는 알래스카만
        if tg == "Alaska" and c["dset"] != "alaska":
            continue
        # 학습 실측 집합
        if tg == "Alaska":
            tr_real = task["ak_train"]
        else:
            tr_real = train_region_set(df, tg, c["dset"])
            if task["extra_real"] is not None:
                tr_real = np.concatenate([tr_real, task["extra_real"]])
        train_real = df.iloc[tr_real]
        ckey = f"{tkey}|{c['dset']}"
        if ckey not in COEFS:
            COEFS[ckey] = fit_coefs(train_real, emap=need_emap)
        k = COEFS[ckey]
        feats = INPUT_SETS[c["xset"]]
        Xte = te[feats].values.astype(np.float32)
        anc_te = anchor_pred(c["anchor"], te, k)
        akey = f"{tkey}|{c['dset']}|{c['anchor']}"
        if anc_te is not None and akey not in ANCH:
            ANCH[akey] = anc_te.astype(np.float32)
        Xreal = train_real[feats].values.astype(np.float32)
        yreal = train_real[TARGET].values.astype(float)
        anc_real = None if anc_te is None else anchor_pred(c["anchor"], train_real, k)
        n_src = int((train_real.macro.values == "Alaska").sum())
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            if c["pseudo"] != "none":
                pool_idx = task["pool"]
                P = df.iloc[pool_idx]
                if c["pseudo"] == "cci":
                    P = P[P.cci_alt.notna()]
                n_ps = int(c["r"] * n_src)
                sel = rng.choice(len(P), n_ps, replace=n_ps > len(P))
                ps = P.iloc[sel]
                yps = pseudo_label(c["pseudo"], ps, k, rng, pool_stefan_mean=float(np.nanmean(k["E"] * P.e5_sqrt_tdd.values)))
                Xtr = np.vstack([Xreal, ps[feats].values.astype(np.float32)])
                ytr = np.concatenate([yreal, yps])
                anc_tr = None if anc_te is None else np.concatenate([anc_real, anchor_pred(c["anchor"], ps, k)])
            else:
                Xtr, ytr, anc_tr = Xreal, yreal, anc_real
            sw = None
            if c.get("iw", 0):
                tgt_idx = np.where(df.macro.values == tg)[0] if tg != "Alaska" else ev
                sw = importance_weights(train_real[feats].values.astype(np.float32),
                                        df.iloc[tgt_idx][feats].values.astype(np.float32), seed=seed)
                if len(sw) < len(Xtr):
                    sw = np.concatenate([sw, np.ones(len(Xtr) - len(sw))])
            Xtr2, Xte2 = fold_prep(Xtr, Xte, nan_native=c["resid"] in NAN_NATIVE)
            pkey = f"{tkey}|{seed}|{c['dset']}|{c['xset']}|{c['anchor']}|{c['pseudo']}|{c['r']}|{c['resid']}|{c.get('iw', 0)}"
            base = dict(cond=cond, target=tg, split=sp, seed=seed, **c, n=len(yte), n_blocks=int(te.block.nunique()),
                        n_train=len(ytr), n_train_real=int(np.isfinite(yreal).sum()))
            try:
                tf = time.time()
                if anc_te is None:
                    ok = np.isfinite(ytr)
                    out = fit_model(c["resid"], Xtr2[ok], ytr[ok], Xte2, seed, epochs=args.epochs,
                                    sample_weight=None if sw is None else sw[ok])
                    p = out["pred"]; n_fit += 1
                    if not np.all(np.isfinite(p)) or np.nanmax(np.abs(p)) > 1e4:
                        raise FloatingPointError("diverged")
                    G[pkey] = p.astype(np.float32)
                    if "q05" in out:
                        Q[pkey] = np.stack([out["q05"], out["q95"]]).astype(np.float32)
                    m = all_metrics(yte, p)
                    rows.append(dict(**base, lam=1.0, rmse_cm=m["rmse_cm"], bias_cm=m["bias_cm"], mae_cm=m["mae_cm"],
                                     r2=m["r2"], fit_s=round(time.time() - tf, 1)))
                else:
                    rtr = ytr - anc_tr
                    ok = np.isfinite(rtr)
                    out = fit_model(c["resid"], Xtr2[ok], rtr[ok], Xte2, seed, epochs=args.epochs,
                                    sample_weight=None if sw is None else sw[ok])
                    g = out["pred"]; n_fit += 1
                    if not np.all(np.isfinite(g)) or np.nanmax(np.abs(g)) > 1e4:
                        raise FloatingPointError("diverged")
                    G[pkey] = g.astype(np.float32)
                    if "q05" in out:
                        Q[pkey] = np.stack([out["q05"], out["q95"]]).astype(np.float32)
                    for lam in LAM_GRID:
                        m = all_metrics(yte, anc_te + lam * g)
                        rows.append(dict(**base, lam=lam, rmse_cm=m["rmse_cm"], bias_cm=m["bias_cm"], mae_cm=m["mae_cm"],
                                         r2=m["r2"], fit_s=round(time.time() - tf, 1)))
            except Exception as e:                            # noqa: BLE001
                rows.append(dict(**base, lam=np.nan, rmse_cm=np.nan, bias_cm=np.nan, mae_cm=np.nan, r2=np.nan,
                                 fit_s=np.nan, error=str(e)[:120]))
    print(f"  [{tkey}] 누적 적합 {n_fit} · 행 {len(rows)} · {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- 저장
tag = args.tag + ("_smoke" if args.smoke else "")
SH = args.shard
pd.DataFrame(rows).to_csv(OUT / f"{tag}_shard{SH}.csv", index=False)
np.savez_compressed(OUT / f"{tag}_shard{SH}_preds.npz",
                    **{f"g::{k}": v for k, v in G.items()},
                    **{f"q::{k}": v for k, v in Q.items()},
                    **{f"anchor::{k}": v for k, v in ANCH.items()},
                    **{f"eval::{k}::{f}": np.asarray(v[f]) for k, v in EVALS.items() for f in ("loc_id", "y", "block")})
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:                                             # noqa: BLE001
    commit = "NA"
coef_out = {k: {kk: (vv if isinstance(vv, (int, float, str)) else None) for kk, vv in v.items() if kk != "emap"}
            for k, v in COEFS.items()}
for k, v in COEFS.items():
    if "emap" in v:
        coef_out[k]["emap_n_blocks"] = v["emap"]["n_blocks"]; coef_out[k]["emap_E_sd"] = v["emap"]["E_blocks_sd"]
(OUT / f"{tag}_shard{SH}_meta.json").write_text(json.dumps(dict(
    stage="M1", tag=args.tag, shard=args.shard, nshard=args.nshard, base=args.base, soil=args.soil, sources=args.sources,
    targets=TARGETS, conds=CONDS, splits=SPLITS, seeds=SEEDS, r=R, epochs=args.epochs, axis=args.axis,
    configs=CONFIGS, lam_grid=LAM_GRID, n_fit=n_fit, coefs=coef_out, git_commit=commit, gpu=GPU,
    n_cells=int(len(df)), regions={r: int(n) for r, n in df.macro.value_counts().items()},
    plan="docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md", elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1))
print(f"saved: {tag}_shard{SH}.csv ({len(rows)}행) · 적합 {n_fit} · {time.time()-t0:.0f}s", flush=True)
