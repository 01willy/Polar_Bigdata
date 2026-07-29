"""S13: 세 상황을 하나의 축으로 정렬한 통합 비교.

문제
----
기존 실험은 상황마다 평가 셀·프로토콜이 달라 수치를 직접 비교할 수 없었다.
특히 완전 전이 설정에서 유사라벨을 붙이는 셀과 평가 셀이 겹쳐(S12 loro) 정의가 흐려졌다.

설계
----
모든 지역(알래스카·레나델타·캐나다)을 0.5° 공간블록으로 2분할한다.
  A 블록 = 유사라벨 후보(실측 라벨은 절대 사용하지 않음)
  B 블록 = 평가 (모든 상황에서 동일)

세 상황은 "대상 지역 정보가 학습에 얼마나 들어가는가" 하나로만 구분된다.
  S1 지역 내      : 같은 지역 A블록의 **실측 라벨**을 학습에 사용
  S2 실측 없음    : 같은 지역 A블록의 **유사라벨**(물리식·위성)을 학습에 사용
  S3 완전 전이    : 대상 지역 정보를 학습에 일절 사용하지 않음(알래스카 실측만)

세 상황 모두 예측 시점에는 B블록 공변량을 쓴다(예측에 필요하므로 불가피).
알래스카의 S3는 정의상 성립하지 않으므로(자기 자신이 학습원) 제외한다.

예측 형태는 공통이다.
  ŷ = 앵커 + λ·g(공변량),  앵커 ∈ {없음, Stefan, 위성제품, Stefan+위성}
잔차 g는 각 상황의 학습 집합으로 적합한다.

계수(Stefan E, 위성 보정, 결합 가중)는 전부 알래스카 학습 블록에서만 추정한다.

실행: python3 s13_unified_settings.py --shard k --nshard N
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import macro_region, spatial_block_splits, SHARED_CORE, TARGET   # noqa: E402
from polar.physics import fit_E                                                       # noqa: E402
from polar.preprocessing import fold_prep                                             # noqa: E402
from polar.eval_metrics import all_metrics                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

OUT = ROOT / "data" / "processed"
R_GRID = [1.0, 5.0, 10.0] if not args.smoke else [5.0]
LAM = [0.0, 0.25, 0.5, 0.75, 1.0] if not args.smoke else [0.0, 0.5]
SEEDS = list(range(args.seeds))
FEATS = list(SHARED_CORE)

# ---------------------------------------------------------------- 자료·계수
df = pd.read_csv(OUT / "fidelity_base.csv")
df["macro"] = macro_region(df)

REGIONS = ["Alaska", "Lena", "Canada"]
SPLIT = {}
for rg in REGIONS:
    t = df[df.macro == rg].reset_index(drop=True)
    folds = spatial_block_splits(t, n_splits=2)
    a_idx, b_idx = folds[0][1], folds[0][0]          # A=유사라벨 후보, B=평가
    SPLIT[rg] = (t, np.asarray(a_idx), np.asarray(b_idx))

ak_t, ak_a, ak_b = SPLIT["Alaska"]
src = ak_t.iloc[ak_a]                                 # 알래스카 A블록 = 공통 학습원
Xsrc = src[FEATS].values.astype(np.float32)
ysrc = src[TARGET].values.astype(float)
n_src = len(src)

E_AK = fit_E(ysrc, src["e5_sqrt_tdd"].values)         # 학습 블록에서만
m = np.isfinite(src.cci_alt.values) & np.isfinite(ysrc)
CCI_B, CCI_A = np.polyfit(src.cci_alt.values[m], ysrc[m], 1)
print(f"[계수] E={E_AK:.4f} · CCI보정 y={CCI_A:.2f}+{CCI_B:.3f}c · 학습원 {n_src}셀", flush=True)


def anchor_of(kind, d):
    s = E_AK * d["e5_sqrt_tdd"].values.astype(float)
    c = d["cci_alt"].values.astype(float)
    return {"none": None, "stefan": s, "cci": c,
            "stefan_cci": 0.5 * (s + c)}[kind]


def pseudo_of(kind, d):
    s = E_AK * d["e5_sqrt_tdd"].values.astype(float)
    c = d["cci_alt"].values.astype(float)
    return {"stefan": s, "cci": c, "stefan_cci": 0.5 * (s + c),
            "const": np.full(len(d), ysrc.mean())}[kind]


def fit_model(name, X, y, Xte, seed):
    if name == "ridge":
        from sklearn.linear_model import Ridge
        mm = Ridge(alpha=10.0); mm.fit(np.nan_to_num(X), y)
        return mm.predict(np.nan_to_num(Xte))
    from polar.tab_models import fit_predict
    return fit_predict(name, X, y, Xte, seed=seed, epochs=100)["pred"]


NAN_NATIVE = {"catboost"}
ANCHORS = ["none", "stefan", "cci", "stefan_cci"]
RESIDS = ["ridge", "catboost", "mlp"]
PSEUDOS = ["stefan", "cci", "stefan_cci", "const"]

# ---------------------------------------------------------------- 구성
CONFIGS = []
for a in ANCHORS[1:]:                                   # 앵커 단독(학습 없음)
    CONFIGS.append(dict(family="analytic", anchor=a, resid="none", pseudo="none", r=0.0))
for a in ANCHORS:
    for rs in RESIDS:
        CONFIGS.append(dict(family="model", anchor=a, resid=rs, pseudo="none", r=0.0))
        for ps in PSEUDOS:
            for r in R_GRID:
                CONFIGS.append(dict(family="model", anchor=a, resid=rs, pseudo=ps, r=r))
CONFIGS = [c for i, c in enumerate(CONFIGS) if i % args.nshard == args.shard]
print(f"[shard {args.shard}/{args.nshard}] 구성 {len(CONFIGS)}개", flush=True)

# ---------------------------------------------------------------- 상황 정의
# (상황, 지역) → 학습 집합 구성 함수
def build_train(setting, rg, cfg, seed):
    """(X, y, anchor_tr) 반환. 대상 지역 실측 라벨은 어떤 상황에서도 쓰지 않는다."""
    t, a_idx, _ = SPLIT[rg]
    A = t.iloc[a_idx]
    if setting == "indomain":
        # 같은 지역 A블록의 실측 라벨 사용(알래스카는 곧 학습원 자체)
        X, y, base = A[FEATS].values.astype(np.float32), A[TARGET].values.astype(float), A
    elif setting == "transfer":
        X, y, base = Xsrc, ysrc, src           # 알래스카 실측만
    elif setting == "pseudo":
        if cfg["pseudo"] == "none" or cfg["r"] == 0:
            X, y, base = Xsrc, ysrc, src
        else:
            n_ps = int(cfg["r"] * n_src)
            sel = np.random.RandomState(seed).choice(len(A), n_ps, replace=n_ps > len(A))
            Aps = A.iloc[sel]
            X = np.vstack([Xsrc, Aps[FEATS].values.astype(np.float32)])
            y = np.concatenate([ysrc, pseudo_of(cfg["pseudo"], Aps)])
            base = pd.concat([src, Aps], ignore_index=True)
    else:
        raise ValueError(setting)
    return X, y, anchor_of(cfg["anchor"], base)


def block_ci(y, p, blocks, seed=0, n=300):
    rng = np.random.RandomState(seed); ub = np.unique(blocks)
    if len(ub) < 3: return (np.nan, np.nan)
    o = [np.sqrt(np.mean((y[i] - p[i]) ** 2)) for i in
         (np.concatenate([np.where(blocks == b)[0] for b in rng.choice(ub, len(ub), True)])
          for _ in range(n))]
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


# ---------------------------------------------------------------- 실행
rows, t0 = [], time.time()
TASKS = [("indomain", "Alaska"),
         ("pseudo", "Lena"), ("pseudo", "Canada"),
         ("transfer", "Lena"), ("transfer", "Canada"),
         ("pseudo", "Alaska")]          # 알래스카에도 유사라벨 증강 검증

for setting, rg in TASKS:
    t, _, b_idx = SPLIT[rg]
    B = t.iloc[b_idx]
    yte = B[TARGET].values.astype(float)
    Xte_raw = B[FEATS].values.astype(np.float32)
    blk = B["block"].values

    for cfg in CONFIGS:
        # 유사라벨 상황에서 pseudo=none은 transfer와 동일하므로 중복 제외
        if setting == "pseudo" and cfg["pseudo"] == "none":
            continue
        if setting in ("indomain", "transfer") and cfg["pseudo"] != "none":
            continue
        anc_te = anchor_of(cfg["anchor"], B)

        def emit(seed, lam, pred, err=None):
            rec = dict(setting=setting, region=rg, seed=seed, lam=lam, n=len(yte), **cfg)
            if err:
                rows.append({**rec, "rmse_cm": np.nan, "bias_cm": np.nan,
                             "ci_lo": np.nan, "ci_hi": np.nan, "error": err}); return
            mt = all_metrics(yte, pred)
            lo, hi = block_ci(yte, pred, blk, seed=max(seed, 0))
            rows.append({**rec, "rmse_cm": mt["rmse_cm"], "bias_cm": mt["bias_cm"],
                         "ci_lo": lo, "ci_hi": hi})

        if cfg["family"] == "analytic":
            emit(-1, 0.0, anc_te); continue

        for seed in SEEDS:
            try:
                Xtr, ytr, anc_tr = build_train(setting, rg, cfg, seed)
                nat = cfg["resid"] in NAN_NATIVE
                Xtr2, Xte2 = fold_prep(Xtr, Xte_raw, nan_native=nat)
                if cfg["anchor"] == "none":
                    p = fit_model(cfg["resid"], Xtr2, ytr, Xte2, seed)
                    emit(seed, 1.0, p if np.all(np.isfinite(p)) and np.nanmax(np.abs(p)) < 1e4 else None,
                         None if np.all(np.isfinite(p)) and np.nanmax(np.abs(p)) < 1e4 else "diverged")
                else:
                    rtr = ytr - anc_tr
                    ok = np.isfinite(rtr)
                    g = fit_model(cfg["resid"], Xtr2[ok], rtr[ok], Xte2, seed)
                    if not np.all(np.isfinite(g)) or np.nanmax(np.abs(g)) > 1e4:
                        for lam in LAM: emit(seed, lam, None, "diverged")
                    else:
                        for lam in LAM: emit(seed, lam, anc_te + lam * g)
            except Exception as e:                                       # noqa: BLE001
                emit(seed, 1.0, None, str(e)[:110])
    print(f"  [{setting}/{rg}] 누적 {len(rows)}행 · {time.time()-t0:.0f}s", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / f"s13_unified_shard{args.shard}.csv", index=False)
print(f"saved {len(res)}행 · {time.time()-t0:.0f}s")
(OUT / f"s13_unified_meta_shard{args.shard}.json").write_text(json.dumps(dict(
    stage="S13", E=E_AK, cci=dict(a=float(CCI_A), b=float(CCI_B)), n_src=n_src,
    feats=FEATS, r_grid=R_GRID, lam=LAM, seeds=SEEDS, tasks=TASKS,
    split="지역별 0.5° 공간블록 2분할. A=유사라벨 후보(실측 미사용), B=평가(전 상황 공통)",
    settings=dict(indomain="같은 지역 A블록 실측 라벨로 학습",
                  pseudo="알래스카 실측 + 대상 A블록 유사라벨로 학습",
                  transfer="알래스카 실측만으로 학습(대상 정보 없음)")),
    ensure_ascii=False, indent=1))
