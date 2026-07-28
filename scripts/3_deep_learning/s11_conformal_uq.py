"""S11: 보정 UQ · quantile CatBoost + CQR conformal (알래스카 in-domain 공간블록).

`docs/RESEARCH_PLAN_multifidelity_2026-07-22.md` S11. 기존 경험(raw 56→CQR 86%,
alt_conformal_cell_results.csv)을 S1 프로토콜(fidelity_base·공변량34·0.5° 공간블록
6-fold)로 연장한다. 핵심 누설통제: conformal calibration 셀은 각 fold의 train 블록
내부에서만 분리한다(test 오염 금지, 기존 overnight 스크립트의 OOF 절반분할과 다름).

방법:
- CatBoost MultiQuantile(11개 분위)로 fold별 proper-train 학습 → calib/test 예측.
- 명목 {50,60,70,80,90}% 각각 CQR conformity E=max(lo-y, y-hi)의 calib 분위 Q로
  구간을 [lo-Q, hi+Q] 보정(Romano 2019). 분위 교차는 정렬(rearrangement)로 교정.
- 3 seed(모델 random_seed + calib 블록 분할), 헤드라인은 블록 부트스트랩 95% CI 병기.

산출: data/processed/s11_conformal_results.csv  (setting x level x seed 지표)
      data/processed/s11_conformal_oof.csv      (셀별 seed-mean 구간; 지도용)
      data/processed/s11_conformal_meta.json
실행: python scripts/3_deep_learning/s11_conformal_uq.py   (SMOKE=1 로 빠른 점검)
"""
import os
# CPU 전용(sklearn/CatBoost). 공유서버 타 사용자 GPU 오점유 방지를 위해 노출 자체 차단.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.fidelity import COVARIATE_CORE, TARGET, add_group_keys, spatial_block_splits
from polar.eval_metrics import all_metrics

SMOKE = os.environ.get("SMOKE", "0") == "1"
PROC = C.PROCESSED
ALASKA = ["ABoVE_AK", "United States (Alaska)"]
SEEDS = [0] if SMOKE else [0, 1, 2]
ITERS = 150 if SMOKE else 600
NBOOT = 100 if SMOKE else 1000
CAL_FRAC = 0.25          # train 블록 중 calibration 블록 비율
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95]
LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90]   # 명목 커버리지
PAIR = {0.50: (0.25, 0.75), 0.60: (0.20, 0.80), 0.70: (0.15, 0.85),
        0.80: (0.10, 0.90), 0.90: (0.05, 0.95)}
AIDX = {a: i for i, a in enumerate(ALPHAS)}

t_start = time.time()
df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
ak = df[df.region.isin(ALASKA)].reset_index(drop=True)
if SMOKE:
    ak = ak.sample(n=min(3000, len(ak)), random_state=0).reset_index(drop=True)
X = ak[COVARIATE_CORE].values.astype(np.float32)
y = ak[TARGET].values.astype(float)
blocks = ak["block"].values
folds = spatial_block_splits(ak, n_splits=6)
print(f"[S11] 알래스카 {len(ak):,}셀 · 공변량 {len(COVARIATE_CORE)} · 6 공간블록 fold · "
      f"seeds={SEEDS} · SMOKE={SMOKE}")

from catboost import CatBoostRegressor

ALPHA_STR = ",".join(str(a) for a in ALPHAS)


def fit_quantiles(Xtr, ytr, Xpr, seed):
    """MultiQuantile CatBoost 학습 후 (n, len(ALPHAS)) 분위 예측. 교차는 정렬로 교정."""
    m = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={ALPHA_STR}",
                          iterations=ITERS, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                          random_seed=seed, verbose=0, thread_count=8, allow_writing_files=False)
    m.fit(Xtr, ytr)
    q = np.asarray(m.predict(Xpr))
    return np.sort(q, axis=1)   # rearrangement: 분위 단조 강제


def cqr_q(y_cal, lo_cal, hi_cal, level):
    """CQR conformity 분위 Q (유한표본 보정 ceil((n+1)c)/n)."""
    E = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(E)
    qlev = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(E, qlev))


# ============ fold-safe split-conformal OOF ============
per_seed = {}   # seed -> dict of OOF arrays
n_cal_total = {}
for seed in SEEDS:
    rng = np.random.default_rng(1000 + seed)
    med = np.full(len(ak), np.nan)
    lo_raw = {lv: np.full(len(ak), np.nan) for lv in LEVELS}
    hi_raw = {lv: np.full(len(ak), np.nan) for lv in LEVELS}
    lo_cqr = {lv: np.full(len(ak), np.nan) for lv in LEVELS}
    hi_cqr = {lv: np.full(len(ak), np.nan) for lv in LEVELS}
    ncal = 0
    for k, (tr, te) in enumerate(folds):
        # calibration은 train "블록" 내부에서 분리(test 블록 무접촉)
        ub = np.unique(blocks[tr])
        rng.shuffle(ub)
        n_cb = max(1, int(round(CAL_FRAC * len(ub))))
        cal_b = set(ub[:n_cb].tolist())
        cal_m = np.array([b in cal_b for b in blocks[tr]])
        prop, cal = tr[~cal_m], tr[cal_m]
        assert not (set(cal.tolist()) & set(te.tolist())), "calib/test 교집합(누설)"
        assert not (set(prop.tolist()) & set(te.tolist())), "train/test 교집합(누설)"
        ncal += len(cal)
        q_cal = fit_quantiles(X[prop], y[prop], X[cal], seed)
        q_te = fit_quantiles(X[prop], y[prop], X[te], seed)
        med[te] = q_te[:, AIDX[0.50]]
        for lv in LEVELS:
            alo, ahi = PAIR[lv]
            lo_c, hi_c = q_cal[:, AIDX[alo]], q_cal[:, AIDX[ahi]]
            lo_t, hi_t = q_te[:, AIDX[alo]], q_te[:, AIDX[ahi]]
            Q = cqr_q(y[cal], lo_c, hi_c, lv)
            lo_raw[lv][te], hi_raw[lv][te] = lo_t, hi_t
            lo_cqr[lv][te], hi_cqr[lv][te] = lo_t - Q, hi_t + Q
        print(f"  seed{seed} fold{k}: proper {len(prop):,} / calib {len(cal):,} "
              f"(블록 {len(ub)-n_cb}/{n_cb}) / test {len(te):,}")
    per_seed[seed] = dict(med=med, lo_raw=lo_raw, hi_raw=hi_raw, lo_cqr=lo_cqr, hi_cqr=hi_cqr)
    n_cal_total[seed] = ncal


def cov_w(yv, lo, hi):
    m = np.isfinite(yv) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((yv[m] >= lo[m]) & (yv[m] <= hi[m]))), float(np.mean(hi[m] - lo[m]))


def block_boot_ci(yv, lo, hi, blk, nboot, seed=0):
    """0.5° 블록 재표집 부트스트랩 커버리지 95% CI."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blk)
    idx_by = {b: np.where(blk == b)[0] for b in ub}
    covs = []
    for _ in range(nboot):
        pick = rng.choice(ub, size=len(ub), replace=True)
        ii = np.concatenate([idx_by[b] for b in pick])
        covs.append(np.mean((yv[ii] >= lo[ii]) & (yv[ii] <= hi[ii])))
    return float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5))


# ============ 지표 집계 ============
rows = []
for seed in SEEDS:
    P = per_seed[seed]
    mm = all_metrics(y, P["med"])
    rows.append(dict(setting="median_point", level=np.nan, seed=seed, **mm))
    for lv in LEVELS:
        for tag, lo, hi in [("raw", P["lo_raw"][lv], P["hi_raw"][lv]),
                            ("cqr", P["lo_cqr"][lv], P["hi_cqr"][lv])]:
            cv, w = cov_w(y, lo, hi)
            rows.append(dict(setting=tag, level=lv, seed=seed, n=len(ak),
                             coverage=round(cv, 4), width_cm=round(w, 3)))

# seed-mean 구간(앙상블): 지도·헤드라인용
ens = {}
for key in ["med"]:
    ens[key] = np.mean([per_seed[s][key] for s in SEEDS], axis=0)
for lv in LEVELS:
    for part in ["lo_raw", "hi_raw", "lo_cqr", "hi_cqr"]:
        ens[(part, lv)] = np.mean([per_seed[s][part][lv] for s in SEEDS], axis=0)
for lv in LEVELS:
    for tag in ["raw", "cqr"]:
        lo, hi = ens[(f"lo_{tag}", lv)], ens[(f"hi_{tag}", lv)]
        cv, w = cov_w(y, lo, hi)
        clo, chi = block_boot_ci(y, lo, hi, blocks, NBOOT, seed=0)
        rows.append(dict(setting=f"{tag}_seedmean", level=lv, seed=-1, n=len(ak),
                         coverage=round(cv, 4), width_cm=round(w, 3),
                         cov_ci_lo=round(clo, 4), cov_ci_hi=round(chi, 4)))

res = pd.DataFrame(rows)
res.to_csv(PROC / "s11_conformal_results.csv", index=False)

# 셀별 OOF(seed-mean, level 0.90): 지도용
oof = ak[["loc_id", "lat", "lon", "region", "block", TARGET]].copy()
oof["pred_med"] = ens["med"]
for tag in ["raw", "cqr"]:
    oof[f"lo90_{tag}"] = ens[(f"lo_{tag}", 0.90)]
    oof[f"hi90_{tag}"] = ens[(f"hi_{tag}", 0.90)]
    oof[f"width90_{tag}"] = oof[f"hi90_{tag}"] - oof[f"lo90_{tag}"]
    oof[f"cov90_{tag}"] = ((oof[TARGET] >= oof[f"lo90_{tag}"]) &
                           (oof[TARGET] <= oof[f"hi90_{tag}"])).astype(int)
oof.to_csv(PROC / "s11_conformal_oof.csv", index=False)

# ============ 요약 출력 + meta ============
sm = res[(res.seed == -1) & (res.level == 0.90)]
raw90 = sm[sm.setting == "raw_seedmean"].iloc[0]
cqr90 = sm[sm.setting == "cqr_seedmean"].iloc[0]
print("\n=== cov90 (seed-mean 구간, 블록부트스트랩 95% CI) ===")
print(f"  raw quantile CatBoost : {raw90.coverage*100:.1f}% [{raw90.cov_ci_lo*100:.1f}, "
      f"{raw90.cov_ci_hi*100:.1f}]  폭 {raw90.width_cm:.1f}cm")
print(f"  CQR 보정              : {cqr90.coverage*100:.1f}% [{cqr90.cov_ci_lo*100:.1f}, "
      f"{cqr90.cov_ci_hi*100:.1f}]  폭 {cqr90.width_cm:.1f}cm  (목표 90%)")

try:
    import subprocess
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=str(Path(__file__).resolve().parents[2])).decode().strip()
except Exception:
    commit = "unknown"
meta = dict(
    stage="S11", purpose="quantile CatBoost + CQR conformal 보정 UQ (알래스카 in-domain)",
    protocol=dict(
        data="fidelity_base.csv 알래스카(ABoVE_AK+US Alaska)", n=int(len(ak)),
        features=len(COVARIATE_CORE), folds="0.5° 공간블록 GroupKFold 6",
        calibration="각 fold의 train 블록 25%를 calib로 내부 분리(test 블록 무접촉, assert 배선)",
        model=f"CatBoost MultiQuantile alpha={ALPHA_STR}, iters={ITERS}, lr=0.03, depth=6",
        crossing_fix="분위 정렬(rearrangement)", seeds=SEEDS, nboot_blocks=NBOOT),
    n_calib_per_seed={str(s): int(n_cal_total[s]) for s in SEEDS},
    headline=dict(
        cov90_raw=float(raw90.coverage), cov90_raw_ci=[float(raw90.cov_ci_lo), float(raw90.cov_ci_hi)],
        width90_raw_cm=float(raw90.width_cm),
        cov90_cqr=float(cqr90.coverage), cov90_cqr_ci=[float(cqr90.cov_ci_lo), float(cqr90.cov_ci_hi)],
        width90_cqr_cm=float(cqr90.width_cm),
        median_rmse_cm=float(res[res.setting == "median_point"].rmse_cm.mean())),
    prior_reference=dict(alt_conformal_cell="raw 56.1→CQR 85.9% (OOF 절반분할, 프로토콜 상이)",
                         alt_conformal_aoa="raw 71.2→CQR 89.2%"),
    caveats=[
        "in-domain(알래스카 공간블록) 보정. 전이(LORO) 커버리지 보장 아님(S6에서 전이는 전 방법 과신 확인).",
        "calibration은 train 블록 내부 분리라 기존 overnight(OOF 절반분할)보다 엄격. 수치 직접 비교 금지.",
        "CQR 보장은 exchangeability 가정 하 marginal coverage. 공간 자기상관 존재 시 근사.",
    ],
    smoke=SMOKE, gpu="미사용(CPU CatBoost, CUDA_VISIBLE_DEVICES='')",
    git_commit=commit, runtime_min=round((time.time() - t_start) / 60, 1),
)
with open(PROC / "s11_conformal_meta.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(f"\n저장: s11_conformal_{{results,oof}}.csv · s11_conformal_meta.json "
      f"({meta['runtime_min']}분)")
