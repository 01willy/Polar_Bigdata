"""E3 부수 실험: 지역 적응 계수 — 대상 지역 라벨 n개로 Stefan 계수만 재적합할 때 전이 오차 곡선.

계획 `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` §3.3. 실용 질문 "라벨 몇 개면 전이 오차가 얼마나 줄어드는가".

설계
----
대상 지역 t마다 셀을 0.5° 블록으로 2분할(A: 라벨 공급, B: 평가; E1과 같은 규약).
n ∈ {0, 3, 5, 10, 20, 40} 개의 A 셀을 무작위로 뽑아(30회 반복) 그 라벨로만 E를 재적합(최소제곱, 절편 없음).
n=0 은 알래스카 E. 앵커 3종: Stefan(√TDD_air), 토양 도일 Stefan(√TDD_stl1), Stefan+CCI 등가중(CCI는 고정, E만 적응).
B 셀(CCI 유효)에서 RMSE. 반복 평균·10–90% 범위. 셀 수가 적은 지역은 n 상한을 |A| 로 자른다.

입력: fidelity_base_v2.csv(없으면 fidelity_base.csv), e5_soil_tdd.csv
산출: data/processed/e3_adaptive_E.csv, e3_adaptive_E_meta.json
실행(ROOT, CPU): python3 scripts/3_deep_learning/e3_adaptive_E.py [--base fidelity_base_v2.csv] [--targets ...]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import add_group_keys, macro_region, spatial_block_splits, TARGET, TRANSFER_MAIN, TRANSFER_DEEP  # noqa: E402

PROC = ROOT / "data" / "processed"
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="fidelity_base_v2.csv" if (PROC / "fidelity_base_v2.csv").exists() else "fidelity_base.csv")
ap.add_argument("--targets", default=",".join(TRANSFER_MAIN + TRANSFER_DEEP))
ap.add_argument("--reps", type=int, default=30)
args = ap.parse_args()
N_GRID = [0, 3, 5, 10, 20, 40]

df = add_group_keys(pd.read_csv(PROC / args.base, low_memory=False))
soil = pd.read_csv(PROC / "e5_soil_tdd.csv")
s1 = soil[soil.loc_id >= 0][["loc_id", "e5_sqrt_tdd_soil"]]
df = df.merge(s1, on="loc_id", how="left")
if df.e5_sqrt_tdd_soil.isna().any():
    s2 = soil[soil.loc_id < 0][["lat", "lon", "e5_sqrt_tdd_soil"]].copy()
    s2["klat"], s2["klon"] = s2.lat.round(4), s2.lon.round(4)
    df["klat"], df["klon"] = df.lat.round(4), df.lon.round(4)
    df = df.merge(s2[["klat", "klon", "e5_sqrt_tdd_soil"]].rename(columns={"e5_sqrt_tdd_soil": "_a"}), on=["klat", "klon"], how="left")
    df["e5_sqrt_tdd_soil"] = df.e5_sqrt_tdd_soil.fillna(df["_a"])
    df = df.drop(columns=["klat", "klon", "_a"])
df["macro"] = macro_region(df)
df = df[df[TARGET].notna() & (df.source_id == "F4_direct")].reset_index(drop=True)
src = df[df.macro == "Alaska"]


def fit_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


E_AK = fit_E(src[TARGET].values, src.e5_sqrt_tdd.values)
E_AK_soil = fit_E(src[TARGET].values, src.e5_sqrt_tdd_soil.values)
print(f"[E_AK] air {E_AK:.4f} · soil {E_AK_soil:.4f} · base {args.base} · {len(df):,}셀")

rows = []
for tg in [t for t in args.targets.split(",") if t]:
    t_idx = np.where(df.macro.values == tg)[0]
    if len(t_idx) < 6:
        print(f"[skip] {tg}: 셀 {len(t_idx)}")
        continue
    folds = spatial_block_splits(df, n_splits=2, sub_idx=t_idx)
    A_idx, B_idx = folds[0][1], folds[0][0]
    A, B = df.iloc[A_idx], df.iloc[B_idx]
    B = B[B.cci_alt.notna() & B.e5_sqrt_tdd_soil.notna()]
    if len(B) < 3:
        print(f"[skip] {tg}: 평가 셀 {len(B)}")
        continue
    yB = B[TARGET].values.astype(float)
    sB, ssB, cB = B.e5_sqrt_tdd.values.astype(float), B.e5_sqrt_tdd_soil.values.astype(float), B.cci_alt.values.astype(float)
    for n in N_GRID:
        if n > len(A):
            continue
        rms = {"stefan": [], "stefan_soil": [], "stefan_cci": []}
        Es = []
        for rep in range(args.reps if n > 0 else 1):
            if n == 0:
                E, Es_ = E_AK, E_AK_soil
            else:
                sel = np.random.RandomState(rep).choice(len(A), n, replace=False)
                a = A.iloc[sel]
                E = fit_E(a[TARGET].values, a.e5_sqrt_tdd.values)
                Es_ = fit_E(a[TARGET].values, a.e5_sqrt_tdd_soil.values)
            Es.append(E)
            rms["stefan"].append(np.sqrt(np.mean((yB - E * sB) ** 2)))
            rms["stefan_soil"].append(np.sqrt(np.mean((yB - Es_ * ssB) ** 2)))
            rms["stefan_cci"].append(np.sqrt(np.mean((yB - 0.5 * (E * sB + cB)) ** 2)))
        for anc, v in rms.items():
            v = np.asarray(v)
            rows.append(dict(target=tg, regime="main" if tg in TRANSFER_MAIN else "deep", anchor=anc, n_labels=n,
                             n_A=len(A), n_eval=len(B), n_blocks_eval=B.block.nunique(), reps=len(v),
                             rmse_mean=v.mean(), rmse_p10=np.quantile(v, 0.1), rmse_p90=np.quantile(v, 0.9),
                             E_mean=float(np.nanmean(Es)), E_alaska=E_AK))
res = pd.DataFrame(rows)
res.to_csv(PROC / "e3_adaptive_E.csv", index=False)
pd.set_option("display.width", 200)
print(res[res.anchor == "stefan"].pivot_table(index="target", columns="n_labels", values="rmse_mean").round(1).to_string())
print("\n[stefan_soil]")
print(res[res.anchor == "stefan_soil"].pivot_table(index="target", columns="n_labels", values="rmse_mean").round(1).to_string())
(PROC / "e3_adaptive_E_meta.json").write_text(json.dumps(dict(
    stage="E3-adaptiveE", base=args.base, n_grid=N_GRID, reps=args.reps, E_alaska=E_AK, E_alaska_soil=E_AK_soil,
    protocol="대상 지역 0.5° 블록 2분할, A에서 n개 라벨로 E 재적합, B(CCI·토양도일 유효)에서 RMSE. 반복 평균.",
    plan="docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md §3.3"), ensure_ascii=False, indent=1))
print("saved: e3_adaptive_E.csv")
