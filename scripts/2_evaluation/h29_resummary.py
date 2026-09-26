"""H29 블록 라벨 가치 재요약(감사 반영): 고유 블록 단위로 해로운 블록 비율과 블록 지표 상관을 다시 계산한다. CPU, 재학습 없음.

배경: h29_block_value.csv 는 (대상, 분할, 블록) 행이다. 같은 블록이 여러 분할의 A 쪽에 반복 등장하므로(캐나다 61행 = 고유 33, 레나 33행 = 고유 18)
옛 h29_summary.csv 의 비율·상관은 표본 수를 부풀렸다. 여기서는 블록마다 분할 평균 가치를 먼저 구하고 고유 블록을 단위로 한다.
  가치 = 물리식(E0) RMSE − 방법 RMSE(B 셀). 음수 = 해로움.
  frac_harm      : 분할 평균 가치 < 0 인 고유 블록 비율, Clopper–Pearson 95 % 구간.
  frac_harm_maj  : 등장한 분할의 과반에서 가치 < 0 인 고유 블록 비율(민감도).
  상관           : Spearman(블록 내 y/√TDD 표준편차, 셀 수, SMD, |E_block − E0|) 대 분할 평균 가치, n = 고유 블록 수(표준편차는 셀 ≥ 2 블록만).
  k-중심 가중 가치: 분할 평균 선택 빈도로 가중한 평균 가치.
입력 data/processed/h3/h29_block_value.csv. 산출 data/processed/h3/h29b_summary.csv, h29b_blocks.csv(고유 블록 표).
실행: python3 scripts/2_evaluation/h29_resummary.py [--smoke]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, beta

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "h3"
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(OUT / "h29_block_value.csv"))
ap.add_argument("--smoke", action="store_true", help="레나만, 산출 파일명에 _smoke")
args = ap.parse_args()
TAG = "h29b_smoke" if args.smoke else "h29b"


def cp_ci(k, n, a=0.05):
    """Clopper–Pearson 구간."""
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(x[m], y[m]); return float(r), float(p), int(m.sum())


R = pd.read_csv(args.src)
if args.smoke:
    R = R[R.target == "Lena"]
R["harm_S1"] = (R.value_S1 < 0).astype(float); R["harm_S3"] = (R.value_S3 < 0).astype(float)
U = R.groupby(["target", "block"]).agg(n_splits=("split", "nunique"), n_cells=("n_cells", "first"), lat=("lat", "first"), lon=("lon", "first"),
                                       value_S1=("value_S1", "mean"), value_S3=("value_S3", "mean"), harm_S1_share=("harm_S1", "mean"), harm_S3_share=("harm_S3", "mean"),
                                       within_sd_ratio=("within_sd_ratio", "first"), smd_src=("smd_src", "first"), E_block_minus_E0=("E_block_minus_E0", "first"),
                                       kmedoid_freq=("kmedoid_freq", "mean")).reset_index()
U.to_csv(OUT / f"{TAG}_blocks.csv", index=False)

rows = []
old = pd.read_csv(OUT / "h29_summary.csv").set_index("target") if (OUT / "h29_summary.csv").exists() else None
for t, sub in U.groupby("target"):
    d = dict(target=t, n_rows_old=int((R.target == t).sum()), n_blocks_unique=len(sub))
    for m in ("S1", "S3"):
        v = sub[f"value_{m}"].values; k = int((v < 0).sum()); lo, hi = cp_ci(k, len(v))
        d.update({f"value_{m}_mean": float(v.mean()), f"frac_harm_{m}": k / len(v), f"frac_harm_{m}_lo": lo, f"frac_harm_{m}_hi": hi,
                  f"frac_harm_{m}_maj": float((sub[f"harm_{m}_share"] > 0.5).mean())})
        w = sub.kmedoid_freq / sub.kmedoid_freq.sum() if sub.kmedoid_freq.sum() > 0 else None
        d[f"value_{m}_kmedoid_weighted"] = float((sub[f"value_{m}"] * w).sum()) if w is not None else np.nan
    for name, x, m in (("withinSD_S3", sub.within_sd_ratio, "S3"), ("withinSD_S1", sub.within_sd_ratio, "S1"), ("ncells_S3", sub.n_cells, "S3"),
                       ("smd_S3", sub.smd_src, "S3"), ("absdE_S1", sub.E_block_minus_E0.abs(), "S1")):
        r, p, n = sp(x, sub[f"value_{m}"]); d.update({f"rho_{name}": r, f"p_{name}": p, f"n_{name}": n})
    if old is not None and t in old.index:
        for c in ("frac_harm_S1", "frac_harm_S3", "rho_withinSD_S3", "p_withinSD", "rho_ncells_S3", "rho_smd_S3", "rho_absdE_S1"):
            d[f"old_{c}"] = float(old.loc[t, c])
    rows.append(d)
S = pd.DataFrame(rows); S.to_csv(OUT / f"{TAG}_summary.csv", index=False)
pd.set_option("display.width", 250)
print(S.T.round(3).to_string())
print(f"[out] {OUT / f'{TAG}_summary.csv'} · {OUT / f'{TAG}_blocks.csv'}")
