"""L2 공통 유틸 — Sci Rep 표준 보조 분석(l2_*.py) 6건의 공용 계층.

규약(원고·M1 하네스 m1_master_factorial.py와 동일)
  - 알래스카 지역 내 = 0.5° 블록 GroupKFold 6-fold. 계수(fit_coefs)·표준화(fold_prep)는 fold 학습 셀에서만.
  - 정보 없음 전이 = 알래스카 전 셀 학습 → 대상 지역 전체 셀. 평가 셀 = y·CCI·토양 도일 유효(eval_mask).
  - 모델: ridge(α=10, 학습 중앙값 대체 + 표준화), catboost_lo(200·0.05·depth 3·l2 3, thread 4). 예측 = 앵커 + λ·g.
  - CPU 전용(CUDA_VISIBLE_DEVICES="" · OMP 4). 기존 파일은 수정하지 않으므로 m1_core.fit_model(thread 6)을 여기서 재구현한다.
재현 확인(2026-09-21): 풀링 Stefan 14.457 · Stefan+ridge(λ=0.75) 13.330 · ridge 직접 13.620 · 정보 없음 레나 21.624/20.761,
캐나다 26.504/25.719 = data/processed/m1/m1_gate_gate.csv 와 일치.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import (SHARED_CORE, TERRAIN, CLIMATE, SOIL, CCI, TARGET, TRANSFER_MAIN, TRANSFER_DEEP,   # noqa: E402
                            spatial_block_splits)
from polar.m1_core import load_base, eval_mask, fit_coefs, anchor_pred                                        # noqa: E402
from polar.preprocessing import fold_prep                                                                     # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "m1"
FIG = ROOT / "outputs" / "figures" / "m1"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FEATS = list(SHARED_CORE)                       # x25
I_SQRT = FEATS.index("e5_sqrt_tdd")
I_CCI = FEATS.index("cci_alt")
GROUP_OF = {**{f: "지형" for f in TERRAIN}, **{f: "기후" for f in CLIMATE}, **{f: "토양" for f in SOIL}, **{f: "CCI" for f in CCI}}
GROUPS = ["지형", "기후", "토양", "CCI"]
THREADS = 4
SEEDS = [0, 1, 2]                               # catboost seed 앙상블(하네스 3 seed 규약)
MIN_BLOCKS_CI = 8

# 방법 정의: anchor(none|stefan|stefan_cci), 잔차/직접 모델(None|ridge|catboost_lo), λ
METHODS = {
    "stefan":         dict(anchor="stefan", model=None, lam=0.0),
    "stefan_cci":     dict(anchor="stefan_cci", model=None, lam=0.0),
    "stefan_ridge075": dict(anchor="stefan", model="ridge", lam=0.75),
    "ridge":          dict(anchor="none", model="ridge", lam=1.0),
    "catboost_lo":    dict(anchor="none", model="catboost_lo", lam=1.0),
    "stefan_cb1":     dict(anchor="stefan", model="catboost_lo", lam=1.0),
}
METHOD_LABEL = {
    "stefan": "Stefan (λ=0)", "stefan_cci": "Stefan+CCI (λ=0)", "stefan_ridge075": "Stefan+ridge (λ=0.75)",
    "ridge": "ridge 직접", "catboost_lo": "CatBoost 직접", "stefan_cb1": "Stefan+CatBoost 잔차 (λ=1)",
}
# 냉색 범주 팔레트(design/brand_tokens.json categorical) + 마커(색 단독 부호화 금지)
METHOD_COLOR = {"stefan": "#8a8f98", "stefan_cci": "#0b7285", "stefan_ridge075": "#1f4e79", "ridge": "#6a51a3",
                "catboost_lo": "#2e8b9e", "stefan_cb1": "#4a7c8c"}
METHOD_MARKER = {"stefan": "o", "stefan_cci": "v", "stefan_ridge075": "s", "ridge": "D", "catboost_lo": "^", "stefan_cb1": "P"}
GROUP_COLOR = {"지형": "#8296a8", "기후": "#1f4e79", "토양": "#4a7c8c", "CCI": "#6a51a3"}


# ---------------------------------------------------------------- 자료
def load(sources=("F4_direct",)) -> pd.DataFrame:
    return load_base(PROC, sources=tuple(sources))


def region_idx(df: pd.DataFrame, region: str) -> np.ndarray:
    return np.where(df.macro.values == region)[0]


def alaska_folds(df: pd.DataFrame, n_splits: int = 6):
    """현행 규약: sklearn GroupKFold(블록 크기 내림차순 탐욕 배정, 결정적). fold 0 = 단일 대블록 4,716셀."""
    return spatial_block_splits(df, n_splits=n_splits, sub_idx=region_idx(df, "Alaska"))


def random_block_folds(df: pd.DataFrame, idx: np.ndarray, n_splits: int, seed: int):
    """무작위 블록 fold 배정: 블록 순서를 seed로 순열한 뒤 GroupKFold와 같은 '셀 수 최소 fold' 탐욕 배정.
    (sklearn 1.3 GroupKFold는 블록 id를 섞어도 크기순 배정이라 결과가 바뀌지 않으므로 순서만 무작위화한다.)"""
    blocks = df["block"].values[idx]
    ub, cnt = np.unique(blocks, return_counts=True)
    order = np.random.RandomState(seed).permutation(len(ub))
    fold_of, load = {}, np.zeros(n_splits, int)
    for j in order:
        f = int(np.argmin(load)); fold_of[ub[j]] = f; load[f] += cnt[j]
    fid = np.array([fold_of[b] for b in blocks])
    return [(idx[fid != f], idx[fid == f]) for f in range(n_splits)]


# ---------------------------------------------------------------- 모델 (fold-safe)
class Prep:
    """fold_prep(nan_native=False)와 동일한 학습 통계 대체·표준화를 임의의 X에 적용(순열 중요도용)."""
    def __init__(self, Xtr):
        med = np.nanmedian(Xtr, axis=0); self.med = np.where(np.isfinite(med), med, 0.0)
        Xf = np.where(np.isnan(Xtr), self.med, Xtr)
        self.mu = Xf.mean(0); self.sd = Xf.std(0) + 1e-6

    def __call__(self, X):
        X = np.where(np.isnan(X), self.med, X)
        return ((X - self.mu) / self.sd).astype(np.float32)


def fit_catboost_lo(Xtr, ytr, seed: int):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=seed,
                          verbose=0, allow_writing_files=False, thread_count=THREADS)
    m.fit(Xtr, ytr)
    return m


class Composite:
    """원 공변량 행렬 X(FEATS 순, NaN 포함, float32)에서 최종 예측. 순열 중요도는 X 열을 섞어 다시 호출한다."""
    def __init__(self, anchor: str, E: float, lam: float, kind: str | None, models, prep: Prep | None):
        self.anchor, self.E, self.lam, self.kind, self.models, self.prep = anchor, E, lam, kind, models, prep

    def __call__(self, X):
        X = np.asarray(X, np.float32)
        p = np.zeros(len(X))
        if self.anchor == "stefan":
            p += self.E * X[:, I_SQRT].astype(float)
        elif self.anchor == "stefan_cci":
            p += 0.5 * (self.E * X[:, I_SQRT].astype(float) + X[:, I_CCI].astype(float))
        if self.kind == "ridge":
            p += self.lam * self.models[0].predict(np.nan_to_num(self.prep(X)))
        elif self.kind == "catboost_lo":
            p += self.lam * np.mean([m.predict(X) for m in self.models], axis=0)
        return p


def fit_methods(train: pd.DataFrame, test: pd.DataFrame, methods, seeds=SEEDS) -> dict:
    """학습 셀(train)만으로 계수·표준화·모델을 적합하고 방법별 Composite를 돌려준다. 예측은 comp(X_test)."""
    k = fit_coefs(train)
    Xtr = train[FEATS].values.astype(np.float32)
    ytr = train[TARGET].values.astype(float)
    out, cache = {}, {}
    for m in methods:
        spec = METHODS[m]
        anc_tr = np.zeros(len(train)) if spec["anchor"] == "none" else anchor_pred(spec["anchor"], train, k)
        if spec["model"] is None:
            out[m] = Composite(spec["anchor"], k["E"], 0.0, None, None, None)
            continue
        rtr = ytr - anc_tr
        ok = np.isfinite(rtr)
        key = (spec["anchor"], spec["model"])
        if key not in cache:
            if spec["model"] == "ridge":
                from sklearn.linear_model import Ridge
                prep = Prep(Xtr)
                Xz, _ = fold_prep(Xtr, Xtr[:1], nan_native=False)          # 하네스와 동일 경로
                rd = Ridge(alpha=10.0).fit(np.nan_to_num(Xz[ok]), rtr[ok])
                cache[key] = ("ridge", [rd], prep)
            else:
                ms = [fit_catboost_lo(Xtr[ok], rtr[ok], s) for s in seeds]
                cache[key] = ("catboost_lo", ms, None)
        kind, models, prep = cache[key]
        out[m] = Composite(spec["anchor"], k["E"], spec["lam"], kind, models, prep)
    out["_coefs"] = k
    return out


KEEP_COLS = ["loc_id", "block", "lat", "lon", "macro", TARGET, "dem_elev", "dem_slope", "e5_maat", "e5_tdd",
             "e5_sqrt_tdd", "sg_soc_0_5", "cci_alt"]


def run_oof(df: pd.DataFrame, folds, methods, seeds=SEEDS, verbose=True) -> pd.DataFrame:
    """fold별 학습→평가 셀 예측(OOF). 반환: 평가 셀 행 + pred_<method> 열 + fold."""
    parts = []
    for fi, (tr, te) in enumerate(folds):
        tev = te[eval_mask(df.iloc[te])]
        trn, tst = df.iloc[tr], df.iloc[tev]
        t0 = time.time()
        comps = fit_methods(trn, tst, methods, seeds)
        X = tst[FEATS].values.astype(np.float32)
        o = tst[KEEP_COLS].copy(); o["fold"] = fi; o["E_fold"] = comps["_coefs"]["E"]
        for m in methods:
            o[f"pred_{m}"] = comps[m](X)
        parts.append(o)
        if verbose:
            msg = " · ".join(f"{m} {rmse(o[TARGET].values, o[f'pred_{m}'].values):.2f}" for m in methods)
            print(f"  fold {fi}: 학습 {len(tr):,} 평가 {len(tev):,} 블록 {tst.block.nunique()} · {msg} · {time.time()-t0:.1f}s", flush=True)
    return pd.concat(parts, ignore_index=True)


def run_noinfo(df: pd.DataFrame, regions, methods, seeds=SEEDS, verbose=True) -> pd.DataFrame:
    """정보 없음 전이: 알래스카 전 셀 학습 → 각 대상 지역 전체 셀(평가 마스크)."""
    trn = df.iloc[region_idx(df, "Alaska")]
    parts = []
    comps = None
    for r in regions:
        idx = region_idx(df, r)
        tst = df.iloc[idx[eval_mask(df.iloc[idx])]]
        if len(tst) == 0:
            continue
        if comps is None:
            comps = fit_methods(trn, tst, methods, seeds)
        X = tst[FEATS].values.astype(np.float32)
        o = tst[KEEP_COLS].copy(); o["fold"] = -1; o["E_fold"] = comps["_coefs"]["E"]
        for m in methods:
            o[f"pred_{m}"] = comps[m](X)
        parts.append(o)
        if verbose:
            msg = " · ".join(f"{m} {rmse(o[TARGET].values, o[f'pred_{m}'].values):.2f}" for m in methods)
            print(f"  noinfo {r}: 평가 {len(tst):,} 블록 {tst.block.nunique()} · {msg}", flush=True)
    return pd.concat(parts, ignore_index=True), comps


# ---------------------------------------------------------------- 지표·부트스트랩·거리
def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.sum() else np.nan


def bias(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(p[m] - y[m])) if m.sum() else np.nan


def block_boot(y, preds: dict, blocks, nboot: int = 1000, seed: int = 0, min_blocks: int = MIN_BLOCKS_CI, ref: str | None = None):
    """0.5° 블록 재표집 부트스트랩. 같은 재표집으로 모든 방법을 채점하므로 방법 간 Δ는 짝지어진다.
    반환 dict[method] = dict(rmse, rmse_lo, rmse_hi, bias, bias_lo, bias_hi, [delta, delta_lo, delta_hi]); 블록 < min_blocks 이면 CI NaN."""
    y = np.asarray(y, float); blocks = np.asarray(blocks)
    ub = np.unique(blocks)
    res = {}
    for m, p in preds.items():
        res[m] = dict(rmse=rmse(y, p), bias=bias(y, p), rmse_lo=np.nan, rmse_hi=np.nan, bias_lo=np.nan, bias_hi=np.nan)
        if ref is not None:
            res[m]["delta"] = rmse(y, p) - rmse(y, preds[ref])
            res[m]["delta_lo"], res[m]["delta_hi"] = np.nan, np.nan
    if len(ub) < min_blocks or nboot <= 0:
        return res
    pos = {b: np.where(blocks == b)[0] for b in ub}
    rng = np.random.RandomState(seed)
    R = {m: np.empty(nboot) for m in preds}
    B = {m: np.empty(nboot) for m in preds}
    for i in range(nboot):
        pick = rng.choice(ub, len(ub), replace=True)
        idx = np.concatenate([pos[b] for b in pick])
        yy = y[idx]
        for m, p in preds.items():
            e = p[idx] - yy
            R[m][i] = np.sqrt(np.mean(e * e)); B[m][i] = np.mean(e)
    for m in preds:
        res[m]["rmse_lo"], res[m]["rmse_hi"] = np.percentile(R[m], [2.5, 97.5])
        res[m]["bias_lo"], res[m]["bias_hi"] = np.percentile(B[m], [2.5, 97.5])
        if ref is not None:
            d = R[m] - R[ref]
            res[m]["delta_lo"], res[m]["delta_hi"] = np.percentile(d, [2.5, 97.5])
    return res


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:                                        # noqa: BLE001
        return "NA"


def write_meta(path: Path, **kw):
    meta = dict(created=datetime.now().strftime("%Y-%m-%d %H:%M"), git_commit=git_commit(), cpu_only=True,
                omp_threads=os.environ.get("OMP_NUM_THREADS"), catboost_threads=THREADS, base="fidelity_base_v3.csv",
                soil="e5_soil_tdd_v3.csv", features="x25 SHARED_CORE", **kw)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=_json_default))
    return meta


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def savefig(fig, name: str):
    """PNG(260 dpi) + PDF(벡터, Type 42) 동시 저장. outputs/figures/m1/<name>.{png,pdf}"""
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    return [str(FIG / f"{name}.png"), str(FIG / f"{name}.pdf")]
