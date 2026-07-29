"""S12: 전이 조건에서 물리 앵커·독립 제품·증강·잔차 결합의 정밀 비교.

동기
----
S3까지의 결과는 "물리 유사라벨 증강이 ML을 Stefan 수준까지 끌어올리나 넘지는 못한다"였다.
원인은 유사라벨이 Stefan 값 자체여서 ML이 Stefan을 모사하도록 학습되기 때문이다.
Stefan을 넘으려면 Stefan이 갖지 못한 정보가 필요하다.

사전 진단(동일 test 셀):
  레나  Stefan bias +7.4 / CCI bias -7.0  → 평균 시 bias +0.2, RMSE 16.16 → 14.57
  캐나다 Stefan bias -8.5 / CCI bias +10.2 → 평균 시 bias +1.1, RMSE 31.29 → 28.79
두 정보원의 편향이 반대 방향이라 결합 시 상쇄된다. 이를 체계적으로 검증한다.

설계
----
평가 프로토콜 2종
  B (half): 대상 지역 셀을 0.5° 공간블록 2분할. 한쪽에 유사라벨 부여해 학습, 다른 쪽 실측으로 평가.
            대상 지역 공변량은 사용 가능, 실측 라벨은 미사용.
  C (loro): 대상 지역 전량 평가. 알래스카만 학습. 유사라벨은 대상 공변량으로 생성(라벨 미사용).

예측기 계열
  anchor    : stefan | cci | stefan_cci(동일가중) | stefan_cci_w(알래스카서 가중 학습) | none
  pseudo    : none | stefan | cci | stefan_cci   (증강 유사라벨 생성원)
  residual  : none | ridge | catboost | mlp | ftt
  r         : 유사라벨 비율, lam: 잔차 가중

모든 계수(E, CCI 보정, 결합 가중)는 알래스카 학습 표본에서만 추정한다(fold-safe).
3 seed, 0.5° 블록 부트스트랩 95% CI.

실행: CUDA_VISIBLE_DEVICES=<gpu> python3 s12_hybrid_transfer.py --shard k --nshard 4
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from polar.fidelity import macro_region, spatial_block_splits, SHARED_CORE, TARGET
from polar.physics import fit_E
from polar.preprocessing import fold_prep
from polar.eval_metrics import all_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

OUT = ROOT / "data" / "processed"
R_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] if not args.smoke else [0.0, 2.0]
LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0] if not args.smoke else [0.0, 0.5]
SEEDS = list(range(args.seeds))
TARGETS = ["Lena", "Canada"]

# ---------------------------------------------------------------- 데이터
df = pd.read_csv(OUT / "fidelity_base.csv")
df["macro"] = macro_region(df)
src = df[df.macro == "Alaska"].reset_index(drop=True)
n_src = len(src)

FEATS = list(SHARED_CORE)
Xsrc = src[FEATS].values.astype(np.float32)
ysrc = src[TARGET].values.astype(float)

# --- fold-safe 계수: 전부 알래스카(학습) 표본에서만 추정 ---
E_AK = fit_E(ysrc, src["e5_sqrt_tdd"].values)                     # Stefan 계수
_m = np.isfinite(src.cci_alt.values) & np.isfinite(ysrc)
_b, _a = np.polyfit(src.cci_alt.values[_m], ysrc[_m], 1)          # CCI 선형 보정
CCI_A, CCI_B = float(_a), float(_b)


def stefan_of(d: pd.DataFrame) -> np.ndarray:
    return E_AK * d["e5_sqrt_tdd"].values.astype(float)


def cci_of(d: pd.DataFrame) -> np.ndarray:
    return d["cci_alt"].values.astype(float)


def cci_cal_of(d: pd.DataFrame) -> np.ndarray:
    return CCI_A + CCI_B * cci_of(d)


# 결합 가중 w: y ≈ w*stefan + (1-w)*cci, 알래스카에서 최소제곱
_s_ak, _c_ak = stefan_of(src), cci_of(src)
_ok = np.isfinite(_s_ak) & np.isfinite(_c_ak) & np.isfinite(ysrc)
_d = _s_ak[_ok] - _c_ak[_ok]
W_SC = float(np.clip(np.dot(_d, ysrc[_ok] - _c_ak[_ok]) / max(np.dot(_d, _d), 1e-9), 0.0, 1.0))


def anchor_pred(kind: str, d: pd.DataFrame) -> np.ndarray | None:
    s, c = stefan_of(d), cci_of(d)
    if kind == "none":
        return None
    if kind == "stefan":
        return s
    if kind == "cci":
        return c
    if kind == "cci_cal":
        return cci_cal_of(d)
    if kind == "stefan_cci":
        return 0.5 * (s + c)
    if kind == "stefan_cci_w":
        return W_SC * s + (1.0 - W_SC) * c
    raise ValueError(kind)


def pseudo_label(kind: str, d: pd.DataFrame) -> np.ndarray:
    s, c = stefan_of(d), cci_of(d)
    if kind == "stefan":
        return s
    if kind == "cci":
        return c
    if kind == "stefan_cci":
        return 0.5 * (s + c)
    if kind == "const":
        return np.full(len(d), ysrc.mean())
    raise ValueError(kind)


# ---------------------------------------------------------------- 모델
def fit_residual(name, Xtr, rtr, Xte, seed):
    """잔차 g(x) 학습·예측."""
    if name == "ridge":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=10.0)
        m.fit(np.nan_to_num(Xtr), rtr)
        return m.predict(np.nan_to_num(Xte))
    from polar.tab_models import fit_predict
    return fit_predict(name, Xtr, rtr, Xte, seed=seed, epochs=100)["pred"]


def fit_direct(name, Xtr, ytr, Xte, seed):
    if name == "ridge":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=10.0)
        m.fit(np.nan_to_num(Xtr), ytr)
        return m.predict(np.nan_to_num(Xte))
    from polar.tab_models import fit_predict
    return fit_predict(name, Xtr, ytr, Xte, seed=seed, epochs=100)["pred"]


NAN_NATIVE = {"catboost"}


def block_bootstrap_ci(y, p, blocks, n_boot=400, seed=0):
    """0.5° 블록 단위 부트스트랩 RMSE 95% CI."""
    rng = np.random.RandomState(seed)
    ub = np.unique(blocks)
    if len(ub) < 3:
        return (np.nan, np.nan)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(ub, len(ub), replace=True)
        idx = np.concatenate([np.where(blocks == b)[0] for b in pick])
        out.append(np.sqrt(np.mean((y[idx] - p[idx]) ** 2)))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


# ---------------------------------------------------------------- 구성 목록
# 잔차 g(x)는 lam과 무관하므로 (anchor,resid,pseudo,r)당 1회만 학습하고 lam은 사후 스윕한다.
CONFIGS = []
# 1) 순수 물리·제품 기준선 (학습 없음)
for anc in ["stefan", "cci", "cci_cal", "stefan_cci", "stefan_cci_w"]:
    CONFIGS.append(dict(family="analytic", anchor=anc, pseudo="none", resid="none", r=0.0))
# 2) 앵커 없는 직접 ML (증강 유무)
for mdl in ["catboost", "mlp", "ftt", "ridge"]:
    for ps, rr in [("none", 0.0)] + [(p, r) for p in ["stefan", "cci", "stefan_cci", "const"] for r in R_GRID if r > 0]:
        CONFIGS.append(dict(family="direct", anchor="none", pseudo=ps, resid=mdl, r=rr))
# 3) 앵커 + 잔차 (증강 유무). lam은 평가 단계에서 스윕.
for anc in ["stefan", "stefan_cci"]:
    for mdl in ["ridge", "catboost", "mlp"]:
        for ps, rr in [("none", 0.0)] + [(p, r) for p in ["stefan", "cci", "stefan_cci"] for r in R_GRID if r > 0]:
            CONFIGS.append(dict(family="anchored", anchor=anc, pseudo=ps, resid=mdl, r=rr))

CONFIGS = [c for i, c in enumerate(CONFIGS) if i % args.nshard == args.shard]
print(f"[shard {args.shard}/{args.nshard}] 구성 {len(CONFIGS)}개 · E={E_AK:.4f} "
      f"CCI보정 y={CCI_A:.2f}+{CCI_B:.3f}c · w_stefan={W_SC:.3f}", flush=True)

# ---------------------------------------------------------------- 평가 루프
rows = []
t0 = time.time()

for proto in ["half", "loro"]:
    for tg in TARGETS:
        t = df[df.macro == tg].reset_index(drop=True)
        if proto == "half":
            folds = spatial_block_splits(t, n_splits=2)
            pseudo_idx, test_idx = folds[0][1], folds[0][0]   # S3와 동일 규약
        else:
            pseudo_idx = np.arange(len(t))                     # 전량을 유사라벨 후보로
            test_idx = np.arange(len(t))                       # 전량 평가
        te = t.iloc[test_idx]
        ps_df = t.iloc[pseudo_idx]
        yte = te[TARGET].values.astype(float)
        Xte = te[FEATS].values.astype(np.float32)
        blocks_te = te["block"].values
        Xps_all = ps_df[FEATS].values.astype(np.float32)

        for cfg in CONFIGS:
            anc_te = anchor_pred(cfg["anchor"], te)
            def emit(seed, lam, pred, err=None):
                if err is not None:
                    rows.append(dict(proto=proto, target=tg, seed=seed, lam=lam, **cfg,
                                     rmse_cm=np.nan, bias_cm=np.nan, n=len(yte),
                                     ci_lo=np.nan, ci_hi=np.nan, error=err))
                    return
                m = all_metrics(yte, pred)
                lo, hi = block_bootstrap_ci(yte, pred, blocks_te, seed=max(seed, 0))
                rows.append(dict(proto=proto, target=tg, seed=seed, lam=lam, **cfg,
                                 rmse_cm=m["rmse_cm"], bias_cm=m["bias_cm"], n=len(yte),
                                 ci_lo=lo, ci_hi=hi))

            # --- 해석적 예측(학습 없음)
            if cfg["family"] == "analytic":
                emit(-1, 0.0, anc_te)
                continue

            for seed in SEEDS:
                # --- 학습 집합 구성
                if cfg["r"] > 0 and cfg["pseudo"] != "none":
                    n_ps = int(cfg["r"] * n_src)
                    sel = np.random.RandomState(seed).choice(
                        len(pseudo_idx), n_ps, replace=n_ps > len(pseudo_idx))
                    Xps = Xps_all[sel]
                    yps = pseudo_label(cfg["pseudo"], ps_df.iloc[sel])
                    Xtr = np.vstack([Xsrc, Xps])
                    ytr_raw = np.concatenate([ysrc, yps])
                    anc_tr = anchor_pred(cfg["anchor"], pd.concat([src, ps_df.iloc[sel]], ignore_index=True))
                else:
                    Xtr, ytr_raw = Xsrc, ysrc
                    anc_tr = anchor_pred(cfg["anchor"], src)

                native = cfg["resid"] in NAN_NATIVE
                Xtr2, Xte2 = fold_prep(Xtr, Xte, nan_native=native)

                try:
                    if cfg["family"] == "anchored":
                        rtr = ytr_raw - anc_tr
                        ok = np.isfinite(rtr)
                        g = fit_residual(cfg["resid"], Xtr2[ok], rtr[ok], Xte2, seed)
                        if not np.all(np.isfinite(g)) or np.nanmax(np.abs(g)) > 1e4:
                            for lam in LAM_GRID:
                                emit(seed, lam, None, err="diverged")
                            continue
                        for lam in LAM_GRID:            # 잔차 1회 학습 → lam 사후 스윕
                            emit(seed, lam, anc_te + lam * g)
                    else:
                        pred = fit_direct(cfg["resid"], Xtr2, ytr_raw, Xte2, seed)
                        if not np.all(np.isfinite(pred)) or np.nanmax(np.abs(pred)) > 1e4:
                            emit(seed, 1.0, None, err="diverged")
                        else:
                            emit(seed, 1.0, pred)
                except Exception as e:                                    # noqa: BLE001
                    emit(seed, 1.0, None, err=str(e)[:120])

        print(f"  [{proto}/{tg}] 누적 {len(rows)}행 · {time.time()-t0:.0f}s", flush=True)

res = pd.DataFrame(rows)
outp = OUT / f"s12_hybrid_transfer_shard{args.shard}.csv"
res.to_csv(outp, index=False)
print(f"saved: {outp}  ({len(res)}행, {time.time()-t0:.0f}s)")

meta = dict(stage="S12", shard=args.shard, nshard=args.nshard,
            E_alaska=E_AK, cci_cal=dict(a=CCI_A, b=CCI_B), w_stefan_cci=W_SC,
            r_grid=R_GRID, lam_grid=LAM_GRID, seeds=SEEDS, feats=FEATS,
            n_src=n_src, n_configs=len(CONFIGS),
            note="모든 계수는 알래스카 학습 표본에서만 추정(fold-safe). "
                 "half=대상 셀 절반 유사라벨 학습·나머지 평가, loro=대상 전량 평가.")
(OUT / f"s12_hybrid_transfer_meta_shard{args.shard}.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=1))
