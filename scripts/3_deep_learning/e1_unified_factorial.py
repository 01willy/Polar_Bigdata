"""E1: 통합 요인 설계 — 세 조건(라벨 있음·공변량만·정보 없음) × 앵커 × 증강 × 잔차 모델을 같은 셀에서 비교.

계획 `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` §1 (S13 결함 7건 해소 규약 §1.4), 사전 등록 H1·H2·H3·H6.

고정
----
- 공변량 25종(SHARED_CORE). 모델 ridge(α=10)·catboost_lo(depth3·200iter)·mlp. 3 seed.
- 계수(Stefan E, 토양 도일 Stefan E_soil, CCI 보정, TDD 선형)는 **학습 실측 라벨에서만** 적합.
  라벨 있음 조건에서는 대상 A블록 실측이 학습 라벨에 들어가므로 계수도 그것을 포함해 적합한다.
- 평가 셀 = 대상 B블록 ∩ CCI 유효 ∩ 토양 도일 유효 ∩ y 유효 (모든 방법 동일). 알래스카는 fold 셀.
- 유사라벨 n_ps = r·n_src(방법 절 정의). 실효 복제율 n_ps/|A| 를 메타에 기록.

조건별 학습 집합 (대상 t 를 0.5° 블록 2분할 A·B, B=평가)
  labels : 알래스카 실측 ∪ A 실측                (P=none)
  covonly: 알래스카 실측 ∪ A 유사라벨(P≠none) 또는 알래스카만(P=none)
  noinfo : 알래스카 실측만                       (P=none). 앵커는 B 공변량으로 산출(라벨 미사용)
  알래스카 지역 내: 6-fold 공간블록(labels 조건만).

요인
  anchor ∈ {none, stefan, cci_cal, stefan_cci, stefan_soil, stefan_soil_cci}
  pseudo ∈ {none, stefan, cci, const, shuffle, tddlin}  (covonly 조건, r=10)
  resid  ∈ {ridge, catboost_lo, mlp};  λ ∈ {0, .25, .5, .75, 1} 사후 스윕
예측 = anchor + λ·g(x).  anchor=none 이면 직접 회귀(λ=1).

산출
  data/processed/e1_factorial_shard{k}.csv          (조건·대상·fold·seed·구성·λ별 rmse·bias·n)
  data/processed/e1_factorial_shard{k}_preds.npz     (g 벡터·앵커 벡터·평가 셀 loc_id/y/block: 짝지은 부트스트랩용)
  data/processed/e1_factorial_meta_shard{k}.json
실행: GPU=2 python3 scripts/3_deep_learning/e1_unified_factorial.py --shard 0 --nshard 4 [--targets Lena,Canada] [--base fidelity_base.csv]
      SMOKE: GPU=2 python3 ... --smoke
"""
from __future__ import annotations
import os
GPU = os.environ.get("GPU", "2")
assert GPU in {"2", "3", "4", "5", "6", "7", "8", "9"}, f"GPU는 2-9만 (요청 {GPU})"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import (SHARED_CORE, TARGET, add_group_keys, macro_region, spatial_block_splits)  # noqa: E402
from polar.preprocessing import fold_prep   # noqa: E402
from polar.eval_metrics import all_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--targets", default="Lena,Canada")
ap.add_argument("--base", default="fidelity_base.csv")
ap.add_argument("--no-alaska", action="store_true", help="알래스카 지역 내 6-fold 생략(E3 신규 지역 실행용)")
ap.add_argument("--noinfo-only", action="store_true",
                help="정보 없음 조건만, 대상 지역 전체 셀을 평가(S12 loro 규약). 소표본 신규 지역(E3)용. 증강 없음.")
ap.add_argument("--min-cells", type=int, default=10)
ap.add_argument("--anchors", default="", help="앵커 수준 제한(쉼표). 예: ku_cal,stefan_ku,stefan_ku_cci")
ap.add_argument("--models", default="", help="잔차 모델 제한(쉼표)")
ap.add_argument("--out-shard", type=int, default=-1, help="산출 파일명의 shard 번호 덮어쓰기(같은 tag 결과에 병합용)")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--r", type=float, default=10.0)
ap.add_argument("--tag", default="")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

PROC = ROOT / "data" / "processed"
LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = list(range(args.seeds)) if not args.smoke else [0]
ANCHORS = ["none", "stefan", "cci_cal", "stefan_cci", "stefan_soil", "stefan_soil_cci"]
# 2026-09-08 추가(탐색적, E2 v2 결과 후): Kudryavtsev 보정 앵커·다중 물리 결합. p4 결측(Ku 비동토 판정) 셀은 Stefan으로 대체.
ANCHORS_EXTRA = ["ku_cal", "stefan_ku", "stefan_ku_cci"]
PSEUDOS = ["none", "stefan", "cci", "const", "shuffle", "tddlin"]
MODELS = ["ridge", "catboost_lo", "mlp"]
if args.smoke:
    ANCHORS, PSEUDOS, MODELS = ["stefan", "stefan_soil"], ["none", "stefan"], ["ridge", "catboost_lo"]
    args.r = 2.0
if args.anchors:
    ANCHORS = [a for a in args.anchors.split(",") if a]
if args.models:
    MODELS = [m for m in args.models.split(",") if m]
TARGETS = [t for t in args.targets.split(",") if t]
R = args.r

# ---------------------------------------------------------------- 자료
df = add_group_keys(pd.read_csv(PROC / args.base, low_memory=False))
soil = pd.read_csv(PROC / "e5_soil_tdd.csv")
s1 = soil[soil.loc_id >= 0][["loc_id", "e5_sqrt_tdd_soil"]]
df = df.merge(s1, on="loc_id", how="left")
if df.e5_sqrt_tdd_soil.isna().any():                       # 신규 CALM 셀: 좌표로 병합
    s2 = soil[soil.loc_id < 0][["lat", "lon", "e5_sqrt_tdd_soil"]].copy()
    s2["klat"], s2["klon"] = s2.lat.round(4), s2.lon.round(4)
    df["klat"], df["klon"] = df.lat.round(4), df.lon.round(4)
    df = df.merge(s2[["klat", "klon", "e5_sqrt_tdd_soil"]].rename(columns={"e5_sqrt_tdd_soil": "_s2"}),
                  on=["klat", "klon"], how="left")
    df["e5_sqrt_tdd_soil"] = df.e5_sqrt_tdd_soil.fillna(df["_s2"])
    df = df.drop(columns=["klat", "klon", "_s2"])
df["macro"] = macro_region(df)
df = df[df[TARGET].notna() & (df.source_id == "F4_direct")].reset_index(drop=True)
FEATS = list(SHARED_CORE)
from polar.physics import physics_ensemble  # noqa: E402
df["p4_ku"] = physics_ensemble(df, E=1.0)["p4_kudryavtsev"]          # 공변량 전용(라벨 미사용)
print(f"[data] {args.base} · {len(df):,}셀 · 토양도일 유효 {df.e5_sqrt_tdd_soil.notna().mean():.3%} · GPU {GPU}", flush=True)

src = df[df.macro == "Alaska"].reset_index(drop=True)
n_src = len(src)


def eval_mask(d: pd.DataFrame) -> np.ndarray:
    return (d[TARGET].notna() & d.cci_alt.notna() & d.e5_sqrt_tdd_soil.notna()).values


# ---------------------------------------------------------------- 계수(학습 실측 라벨에서만)
def fit_coefs(train: pd.DataFrame) -> dict:
    y = train[TARGET].values.astype(float)
    s = train["e5_sqrt_tdd"].values.astype(float)
    ss = train["e5_sqrt_tdd_soil"].values.astype(float)
    c = train["cci_alt"].values.astype(float)
    t = train["e5_tdd"].values.astype(float)
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    E = float((s[m] @ y[m]) / (s[m] @ s[m]))
    m2 = np.isfinite(y) & np.isfinite(ss) & (ss > 0)
    E_soil = float((ss[m2] @ y[m2]) / (ss[m2] @ ss[m2]))
    m3 = np.isfinite(y) & np.isfinite(c)
    cb, ca = np.polyfit(c[m3], y[m3], 1)
    m4 = np.isfinite(y) & np.isfinite(t)
    tb, ta = np.polyfit(t[m4], y[m4], 1)
    p4 = train["p4_ku"].values.astype(float)
    m5 = np.isfinite(y) & np.isfinite(p4)
    c_ku = float((p4[m5] @ y[m5]) / (p4[m5] @ p4[m5])) if m5.sum() >= 3 else np.nan
    return dict(E=E, E_soil=E_soil, cci_a=float(ca), cci_b=float(cb), tdd_a=float(ta), tdd_b=float(tb),
                c_ku=c_ku, ymean=float(np.nanmean(y)), n_train_real=int(m.sum()))


def anchor_pred(kind: str, d: pd.DataFrame, k: dict):
    s = k["E"] * d["e5_sqrt_tdd"].values.astype(float)
    ss = k["E_soil"] * d["e5_sqrt_tdd_soil"].values.astype(float)
    c = d["cci_alt"].values.astype(float)
    if kind == "none":
        return None
    if kind == "stefan":
        return s
    if kind == "cci_cal":
        return k["cci_a"] + k["cci_b"] * c
    if kind == "stefan_cci":
        return 0.5 * (s + c)
    if kind == "stefan_soil":
        return ss
    if kind == "stefan_soil_cci":
        return 0.5 * (ss + c)
    ku = k["c_ku"] * d["p4_ku"].values.astype(float)
    ku = np.where(np.isfinite(ku), ku, s)                 # Ku 비동토 판정 셀은 Stefan 대체
    if kind == "ku_cal":
        return ku
    if kind == "stefan_ku":
        return 0.5 * (s + ku)
    if kind == "stefan_ku_cci":
        return (s + ku + c) / 3.0
    raise ValueError(kind)


def pseudo_label(kind: str, d: pd.DataFrame, k: dict, rng: np.random.RandomState):
    s = k["E"] * d["e5_sqrt_tdd"].values.astype(float)
    if kind == "stefan":
        return s
    if kind == "cci":
        return d["cci_alt"].values.astype(float)
    if kind == "const":
        return np.full(len(d), k["ymean"])
    if kind == "shuffle":                                   # 값 분포 유지, 셀 대응 파괴
        return rng.permutation(s)
    if kind == "tddlin":
        return k["tdd_a"] + k["tdd_b"] * d["e5_tdd"].values.astype(float)
    raise ValueError(kind)


# ---------------------------------------------------------------- 모델
NAN_NATIVE = {"catboost_lo"}


def fit_model(name, Xtr, ytr, Xte, seed):
    if name == "ridge":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=10.0)
        m.fit(np.nan_to_num(Xtr), ytr)
        return m.predict(np.nan_to_num(Xte))
    if name == "catboost_lo":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0,
                              random_seed=seed, verbose=0, allow_writing_files=False, thread_count=8)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict(Xte))
    from polar.tab_models import fit_predict
    return fit_predict(name, Xtr, ytr, Xte, seed=seed, epochs=100)["pred"]


# ---------------------------------------------------------------- 작업 목록(조건·대상·fold 단위)
TASKS = []   # dict(cond, target, fold, train_real_idx(df 인덱스), A_idx, eval_idx)
for tg in TARGETS:
    t_idx = np.where(df.macro.values == tg)[0]
    if len(t_idx) < args.min_cells:
        print(f"[skip] {tg}: 셀 {len(t_idx)}개 (< {args.min_cells})", flush=True)
        continue
    if args.noinfo_only:                                            # 지역 전체 평가, 알래스카만 학습
        ev = t_idx[eval_mask(df.iloc[t_idx])]
        src_idx = np.where(df.macro.values == "Alaska")[0]
        TASKS.append(dict(cond="noinfo", target=tg, fold=0, train_real=src_idx, A=None, ev=ev))
        continue
    folds = spatial_block_splits(df, n_splits=2, sub_idx=t_idx)
    A_idx, B_idx = folds[0][1], folds[0][0]                    # S3·S12 규약과 동일(A=유사라벨/라벨, B=평가)
    ev = B_idx[eval_mask(df.iloc[B_idx])]
    src_idx = np.where(df.macro.values == "Alaska")[0]
    TASKS.append(dict(cond="labels", target=tg, fold=0, train_real=np.concatenate([src_idx, A_idx]), A=A_idx, ev=ev))
    TASKS.append(dict(cond="covonly", target=tg, fold=0, train_real=src_idx, A=A_idx, ev=ev))
    TASKS.append(dict(cond="noinfo", target=tg, fold=0, train_real=src_idx, A=None, ev=ev))
if not args.no_alaska:
    ak_idx = np.where(df.macro.values == "Alaska")[0]
    for fi, (tr, te) in enumerate(spatial_block_splits(df, n_splits=6, sub_idx=ak_idx)):
        if args.smoke and fi >= 2:
            break
        TASKS.append(dict(cond="labels", target="Alaska", fold=fi, train_real=tr, A=None, ev=te[eval_mask(df.iloc[te])]))

CONFIGS = []
for anc in ANCHORS:
    for mdl in MODELS:
        CONFIGS.append(dict(anchor=anc, pseudo="none", r=0.0, resid=mdl))
        for ps in PSEUDOS:
            if ps != "none":
                CONFIGS.append(dict(anchor=anc, pseudo=ps, r=R, resid=mdl))
CONFIGS = [c for i, c in enumerate(CONFIGS) if i % args.nshard == args.shard]
print(f"[shard {args.shard}/{args.nshard}] 작업 {len(TASKS)} × 구성 {len(CONFIGS)} × seed {len(SEEDS)}", flush=True)

# ---------------------------------------------------------------- 실행
rows, preds, anchors_store, evalsets = [], {}, {}, {}
meta_coefs = {}
t0 = time.time()
n_fit = 0
for task in TASKS:
    cond, tg, fold, ev = task["cond"], task["target"], task["fold"], task["ev"]
    train_real = df.iloc[task["train_real"]]
    k = fit_coefs(train_real)
    key_task = f"{cond}|{tg}|{fold}"
    meta_coefs[key_task] = dict(**k, n_eval=int(len(ev)),
                                n_A=int(len(task["A"])) if task["A"] is not None else 0,
                                eff_replication=(R * n_src / len(task["A"])) if task["A"] is not None else None)
    te = df.iloc[ev]
    yte = te[TARGET].values.astype(float)
    Xte = te[FEATS].values.astype(np.float32)
    evalsets[key_task] = dict(loc_id=te.loc_id.values, y=yte, block=te.block.values)
    Xreal = train_real[FEATS].values.astype(np.float32)
    yreal = train_real[TARGET].values.astype(float)

    for cfg in CONFIGS:
        if cfg["pseudo"] != "none" and cond != "covonly":
            continue                                                   # 증강은 공변량만 조건에서만 성립
        anc_te = anchor_pred(cfg["anchor"], te, k)
        akey = f"{key_task}|{cfg['anchor']}"
        if anc_te is not None and akey not in anchors_store:
            anchors_store[akey] = anc_te.astype(np.float32)
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            if cfg["pseudo"] != "none":
                A = df.iloc[task["A"]]
                pool = A[A.cci_alt.notna()] if cfg["pseudo"] == "cci" else A
                n_ps = int(R * n_src)
                sel = rng.choice(len(pool), n_ps, replace=n_ps > len(pool))
                ps_df = pool.iloc[sel]
                yps = pseudo_label(cfg["pseudo"], ps_df, k, rng)
                Xtr = np.vstack([Xreal, ps_df[FEATS].values.astype(np.float32)])
                ytr = np.concatenate([yreal, yps])
                anc_tr = (None if anc_te is None else
                          np.concatenate([anchor_pred(cfg["anchor"], train_real, k), anchor_pred(cfg["anchor"], ps_df, k)]))
            else:
                Xtr, ytr = Xreal, yreal
                anc_tr = None if anc_te is None else anchor_pred(cfg["anchor"], train_real, k)
            Xtr2, Xte2 = fold_prep(Xtr, Xte, nan_native=cfg["resid"] in NAN_NATIVE)
            pkey = f"{key_task}|{seed}|{cfg['anchor']}|{cfg['pseudo']}|{cfg['r']}|{cfg['resid']}"
            try:
                if anc_te is None:                                     # 직접 회귀
                    ok = np.isfinite(ytr)
                    p = fit_model(cfg["resid"], Xtr2[ok], ytr[ok], Xte2, seed)
                    n_fit += 1
                    if not np.all(np.isfinite(p)) or np.nanmax(np.abs(p)) > 1e4:
                        raise FloatingPointError("diverged")
                    preds[pkey] = p.astype(np.float32)
                    m = all_metrics(yte, p)
                    rows.append(dict(cond=cond, target=tg, fold=fold, seed=seed, **cfg, lam=1.0,
                                     rmse_cm=m["rmse_cm"], bias_cm=m["bias_cm"], n=len(yte)))
                else:
                    rtr = ytr - anc_tr
                    ok = np.isfinite(rtr)
                    g = fit_model(cfg["resid"], Xtr2[ok], rtr[ok], Xte2, seed)
                    n_fit += 1
                    if not np.all(np.isfinite(g)) or np.nanmax(np.abs(g)) > 1e4:
                        raise FloatingPointError("diverged")
                    preds[pkey] = g.astype(np.float32)
                    for lam in LAM_GRID:
                        m = all_metrics(yte, anc_te + lam * g)
                        rows.append(dict(cond=cond, target=tg, fold=fold, seed=seed, **cfg, lam=lam,
                                         rmse_cm=m["rmse_cm"], bias_cm=m["bias_cm"], n=len(yte)))
            except Exception as e:                                     # noqa: BLE001
                rows.append(dict(cond=cond, target=tg, fold=fold, seed=seed, **cfg, lam=np.nan,
                                 rmse_cm=np.nan, bias_cm=np.nan, n=len(yte), error=str(e)[:100]))
    print(f"  [{key_task}] 누적 적합 {n_fit} · 행 {len(rows)} · {time.time()-t0:.0f}s", flush=True)

tag = args.tag or ("_smoke" if args.smoke else "")
SH = args.out_shard if args.out_shard >= 0 else args.shard
out_csv = PROC / f"e1_factorial{tag}_shard{SH}.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
np.savez_compressed(PROC / f"e1_factorial{tag}_shard{SH}_preds.npz",
                    **{f"g::{k}": v for k, v in preds.items()},
                    **{f"anchor::{k}": v for k, v in anchors_store.items()},
                    **{f"eval::{k}::{f}": np.asarray(v[f]) for k, v in evalsets.items() for f in ("loc_id", "y", "block")})
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:                                                      # noqa: BLE001
    commit = "NA"
(PROC / f"e1_factorial{tag}_meta_shard{SH}.json").write_text(json.dumps(dict(
    stage="E1", shard=args.shard, nshard=args.nshard, base=args.base, targets=TARGETS, r=R, seeds=SEEDS,
    anchors=ANCHORS, pseudos=PSEUDOS, models=MODELS, lam_grid=LAM_GRID, feats=FEATS, n_src=n_src,
    n_configs=len(CONFIGS), n_fit=n_fit, coefs_by_task=meta_coefs, git_commit=commit, gpu=GPU,
    plan="docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md §1 (개정 2026-09-08: stefan_soil 앵커 추가)",
    elapsed_s=round(time.time() - t0, 1),
), ensure_ascii=False, indent=1))
print(f"saved: {out_csv.name} ({len(rows)}행) · 적합 {n_fit} · {time.time()-t0:.0f}s", flush=True)
