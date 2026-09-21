"""A2-5 오라클 E 상한(C2-D) — 대상 전 라벨로 적합한 Stefan(수준 오차 상한) 대 알래스카 E Stefan·Stefan+CCI·Stefan+CCI(보정).

지역별(정보 없음 평가 셀)로
  · 알래스카 E Stefan, Stefan+CCI 원값(등가중), Stefan+CCI 보정(등가중) RMSE
  · 오라클 k=1: E* = Σ s·y / Σ s² (대상 라벨, 표본 내) → RMSE; k=2: a + E·√TDD OLS → RMSE
  · 참고 오라클 k=3: a + b·Stefan + c·CCI OLS(대상 라벨) → 결합의 정보 상한
  · 비율 = (RMSE_S − RMSE_combo)/(RMSE_S − RMSE_oracle1): 결합이 수준 오차 상한 대비 회수한 몫
오라클은 표본 내 적합이라 낙관적(특히 n<10 지역). 셀 수·블록 수를 기록.
산출: data/processed/m1/a2_oracle_E.csv
실행: python3 scripts/2_evaluation/a2_oracle_E.py
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a2_common import Base, OUT, rmse, bias, flag_small   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sources", default="F4_direct,F4_calm_temp")
args = ap.parse_args()
t0 = time.time()
base = Base(sources=tuple(args.sources.split(",")))
print(base.describe(), flush=True)

rows = []
for tg in base.regions:
    te = base.te(tg)
    y = base.y(tg)
    s = te.e5_sqrt_tdd.values.astype(float)
    S, C, Cc = base.anchor("stefan", tg), base.anchor("cci_raw", tg), base.anchor("cci_cal", tg)
    SC, SCc = 0.5 * (S + C), 0.5 * (S + Cc)
    n = len(y)
    m = np.isfinite(s) & (s > 0)
    E1 = float((s[m] @ y[m]) / (s[m] @ s[m]))
    or1 = E1 * s
    A = np.c_[np.ones(n), s]
    a2, E2 = np.linalg.lstsq(A, y, rcond=None)[0] if n >= 2 else (np.nan, np.nan)
    or2 = a2 + E2 * s
    A3 = np.c_[np.ones(n), S, C]
    c3 = np.linalg.lstsq(A3, y, rcond=None)[0] if n >= 3 else np.full(3, np.nan)
    or3 = A3 @ c3
    r = dict(target=tg, n_cells=n, n_blocks=int(te.block.nunique()), flag=flag_small(n),
             E_alaska=round(base.k["E"], 4), E_oracle_k1=round(E1, 4), a_oracle_k2=round(float(a2), 2), E_oracle_k2=round(float(E2), 4),
             y_mean=round(float(y.mean()), 2), y_sd=round(float(y.std()), 2),
             rmse_S=round(rmse(y, S), 3), bias_S=round(bias(y, S), 3),
             rmse_SC=round(rmse(y, SC), 3), rmse_SCcal=round(rmse(y, SCc), 3), rmse_C=round(rmse(y, C), 3),
             rmse_oracle_k1=round(rmse(y, or1), 3), rmse_oracle_k2=round(rmse(y, or2), 3), rmse_oracle_k3_S_C=round(rmse(y, or3), 3),
             rmse_const_mean=round(float(y.std()), 3))
    gS1 = r["rmse_S"] - r["rmse_oracle_k1"]
    r["gain_SC"] = round(r["rmse_S"] - r["rmse_SC"], 3)
    r["gain_SCcal"] = round(r["rmse_S"] - r["rmse_SCcal"], 3)
    r["gain_oracle_k1"] = round(gS1, 3)
    r["gain_oracle_k2"] = round(r["rmse_S"] - r["rmse_oracle_k2"], 3)
    r["ratio_SC_over_oracle_k1"] = round(r["gain_SC"] / gS1, 3) if gS1 > 1e-9 else np.nan
    r["ratio_SCcal_over_oracle_k1"] = round(r["gain_SCcal"] / gS1, 3) if gS1 > 1e-9 else np.nan
    r["residual_after_oracle_k1_pct"] = round(100 * r["rmse_oracle_k1"] / r["rmse_S"], 1)
    rows.append(r)
    print(f"  [{tg}] n={n} S={r['rmse_S']} SC={r['rmse_SC']} SCcal={r['rmse_SCcal']} oracle k1={r['rmse_oracle_k1']} (E*={E1:.3f} vs {base.k['E']:.3f}) "
          f"k2={r['rmse_oracle_k2']} k3={r['rmse_oracle_k3_S_C']} ratio={r['ratio_SC_over_oracle_k1']}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUT / "a2_oracle_E.csv", index=False)
(OUT / "a2_oracle_E_meta.json").write_text(json.dumps(dict(sources=args.sources, E_alaska=base.k["E"], regions=base.regions,
    note="오라클은 대상 전 라벨 표본 내 적합(낙관적 상한)", elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1))
print("\n=== A2-5 오라클 E 상한 (정보 없음 평가 셀, cm) ===")
print(out[["target", "n_cells", "flag", "E_alaska", "E_oracle_k1", "rmse_S", "rmse_SC", "rmse_SCcal", "rmse_oracle_k1", "rmse_oracle_k2",
           "rmse_oracle_k3_S_C", "gain_SC", "gain_oracle_k1", "ratio_SC_over_oracle_k1", "ratio_SCcal_over_oracle_k1", "bias_S"]].to_string(index=False))
print(f"\n저장: {OUT/'a2_oracle_E.csv'} · {time.time()-t0:.0f}s")
