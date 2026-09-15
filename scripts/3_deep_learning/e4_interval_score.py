"""E4.3: 보정 예측구간의 proper scoring rule과 조건부 커버리지 (W12).

배경
----
S11은 90% 구간의 marginal 커버리지(raw 44.6% → CQR 93.4%)와 평균 폭(14.7 → 53.6 cm)만 보고했다.
폭 53.6 cm의 정보량을 proper scoring rule로 평가하고, 커버리지가 ALT 수준·구간 폭·공간블록에 따라
어떻게 달라지는지(조건부 커버리지) 보고한다. 계획 `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` §4 E4.3.

지표
----
  interval score(Winkler, α=0.1): IS = (hi−lo) + (2/α)(lo−y)·1[y<lo] + (2/α)(y−hi)·1[y>hi]. 작을수록 좋다.
  분해: 폭 항 / 하방 미달 벌점 / 상방 미달 벌점.
  조건부 커버리지: 관측 ALT 5분위, 구간 폭 5분위, 0.5° 블록별 커버리지 분포(블록 수·사분위).
  참조: 상수 폭 구간(전 셀에 같은 폭을 주어 marginal 90%를 맞춘 구간)의 IS. 보정 구간이 이 참조보다
       낮아야 "셀별 불확실성이 정보를 담는다"고 말할 수 있다.

입력: data/processed/s11_conformal_oof.csv  (셀별 pred_med·lo/hi90_raw·lo/hi90_cqr)
산출: data/processed/e4_interval_score.csv, e4_conditional_coverage.csv, e4_interval_score_meta.json
실행(ROOT, CPU): python3 scripts/3_deep_learning/e4_interval_score.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
ALPHA = 0.10

d = pd.read_csv(PROC / "s11_conformal_oof.csv")
y = d.alt_cm.values.astype(float)
print(f"[load] {len(d):,}셀 · 블록 {d.block.nunique()} · 지역 {d.region.unique().tolist()}")


def interval_score(y, lo, hi, alpha=ALPHA):
    width = hi - lo
    under = (2 / alpha) * np.clip(lo - y, 0, None)
    over = (2 / alpha) * np.clip(y - hi, 0, None)
    return width + under + over, width, under, over


def block_boot_mean(v, blocks, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    ub = np.unique(blocks)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(ub, len(ub), replace=True)
        idx = np.concatenate([np.where(blocks == b)[0] for b in pick])
        out.append(np.mean(v[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


rows = []
sets = {"raw": ("lo90_raw", "hi90_raw"), "cqr": ("lo90_cqr", "hi90_cqr")}
# 참조: 중앙값 예측 ± 상수 반폭(marginal 90% 커버리지가 되도록 |y−pred_med|의 90% 분위)
half = np.quantile(np.abs(y - d.pred_med.values), 0.90)
d["lo90_const"] = d.pred_med - half
d["hi90_const"] = d.pred_med + half
sets["const_width"] = ("lo90_const", "hi90_const")

for name, (lo_c, hi_c) in sets.items():
    lo, hi = d[lo_c].values.astype(float), d[hi_c].values.astype(float)
    IS, W, U, O = interval_score(y, lo, hi)
    cov = ((y >= lo) & (y <= hi)).astype(float)
    ci_is = block_boot_mean(IS, d.block.values)
    ci_cov = block_boot_mean(cov, d.block.values)
    rows.append(dict(interval=name, n=len(d), coverage=cov.mean(), cov_ci_lo=ci_cov[0], cov_ci_hi=ci_cov[1],
                     width_mean=W.mean(), interval_score=IS.mean(), is_ci_lo=ci_is[0], is_ci_hi=ci_is[1],
                     is_width_part=W.mean(), is_under_part=U.mean(), is_over_part=O.mean(),
                     width_median=np.median(W), width_p10=np.quantile(W, 0.1), width_p90=np.quantile(W, 0.9)))
    d[f"cov_{name}"] = cov
    d[f"is_{name}"] = IS
    d[f"w_{name}"] = W
res = pd.DataFrame(rows)
res.to_csv(PROC / "e4_interval_score.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== 90% 구간 점수(작을수록 좋음) ===")
print(res[["interval", "coverage", "cov_ci_lo", "cov_ci_hi", "width_mean", "interval_score", "is_ci_lo", "is_ci_hi",
           "is_width_part", "is_under_part", "is_over_part"]].round(3).to_string(index=False))

# 조건부 커버리지
cond = []
d["alt_bin"] = pd.qcut(y, 5, labels=[f"Q{i}" for i in range(1, 6)])
for name in ["raw", "cqr"]:
    g = d.groupby("alt_bin", observed=True)
    for b, gg in g:
        cond.append(dict(interval=name, axis="alt_quintile", bin=str(b), lo=gg.alt_cm.min(), hi=gg.alt_cm.max(),
                         n=len(gg), coverage=gg[f"cov_{name}"].mean(), width_mean=gg[f"w_{name}"].mean(),
                         interval_score=gg[f"is_{name}"].mean()))
    d[f"wbin_{name}"] = pd.qcut(d[f"w_{name}"].rank(method="first"), 5, labels=[f"W{i}" for i in range(1, 6)])
    for b, gg in d.groupby(f"wbin_{name}", observed=True):
        cond.append(dict(interval=name, axis="width_quintile", bin=str(b), lo=gg[f"w_{name}"].min(),
                         hi=gg[f"w_{name}"].max(), n=len(gg), coverage=gg[f"cov_{name}"].mean(),
                         width_mean=gg[f"w_{name}"].mean(), interval_score=gg[f"is_{name}"].mean()))
    blk = d.groupby("block")[f"cov_{name}"].agg(["mean", "size"])
    blk = blk[blk["size"] >= 20]
    q = blk["mean"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    cond.append(dict(interval=name, axis="block_coverage_dist", bin=f"blocks_n>=20 ({len(blk)})", lo=q[0.1], hi=q[0.9],
                     n=int(blk["size"].sum()), coverage=q[0.5], width_mean=np.nan,
                     interval_score=float((blk["mean"] < 0.8).mean())))   # 커버리지 80% 미만 블록 비율
cc = pd.DataFrame(cond)
cc.to_csv(PROC / "e4_conditional_coverage.csv", index=False)
print("\n=== 조건부 커버리지 (관측 ALT 5분위) ===")
print(cc[cc.axis == "alt_quintile"].pivot_table(index="bin", columns="interval", values="coverage").round(3).to_string())
print("\n=== 조건부 커버리지 (구간 폭 5분위, 좁음→넓음) ===")
print(cc[cc.axis == "width_quintile"][["interval", "bin", "lo", "hi", "coverage", "interval_score"]].round(2).to_string(index=False))
print("\n=== 블록별 커버리지 분포(중앙값·10–90% 범위·80% 미만 블록 비율) ===")
print(cc[cc.axis == "block_coverage_dist"][["interval", "bin", "lo", "coverage", "hi", "interval_score"]].round(3).to_string(index=False))

(PROC / "e4_interval_score_meta.json").write_text(json.dumps(dict(
    stage="E4.3", input="s11_conformal_oof.csv", alpha=ALPHA, n=int(len(d)),
    const_width_reference=dict(half_width_cm=float(half), note="pred_med ± |y−pred_med|의 90% 분위(marginal 90% 참조)"),
    plan="docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md §4 E4.3",
    caveat="S11 OOF는 3 seed 중 파일에 저장된 단일 세트. 블록 부트스트랩 1000회 95% CI.",
), ensure_ascii=False, indent=1))
print("\nsaved: e4_interval_score.csv · e4_conditional_coverage.csv")
