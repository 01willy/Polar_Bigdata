"""A2-3 결합 앵커 분해(C2-A) — Stefan(알래스카 E)·CCI 원값·CCI 보정(알래스카 affine)의 오차 구조와 등가중 결합 이득의 분해.

지역별(정보 없음 평가 셀)로
  · 오차 상관 ρ(e_S, e_C), ρ(e_S, e_Ccal); 편향²·분산 분해(MSE = bias² + var) — 각 앵커와 등가중 결합
  · 이론 최적 가중 w* = (m_CC − m_SC)/(m_SS + m_CC − 2 m_SC) (m = 비중심 오차 2차 모멘트, 표본 내 최적)와 그때의 RMSE
  · 결합 이득의 예측치: 오차 독립(ρ=0) 가정의 RMSE_combo = sqrt(((b_S+b_C)/2)² + (v_S+v_C)/4) 대 관측 RMSE_combo
    (ρ를 넣으면 항등식이므로, ρ에 의해 잠식된 이득 비율 = Δ_obs/Δ_indep 를 보고)
  · 블록 등가중 RMSE(블록별 RMSE의 비가중 평균), 블록 다수결(결합 < Stefan 블록 비율), 블록 잭나이프(한 블록 제외 시 Δ의 범위·최대 영향 블록)
  · 참고: H1 Δ(결합 − Stefan)의 블록 부트스트랩 CI.
산출: data/processed/m1/a2_combo_decomp.csv (지역 1행), a2_combo_decomp_jackknife.csv (지역×블록)
실행: python3 scripts/2_evaluation/a2_combo_decomp.py
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
from a2_common import Base, OUT, paired_boot, rmse, bias, flag_small   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sources", default="F4_direct,F4_calm_temp")
ap.add_argument("--nboot", type=int, default=400)
args = ap.parse_args()
t0 = time.time()
base = Base(sources=tuple(args.sources.split(",")))
print(base.describe(), flush=True)


def moments(eA, eB):
    return float(np.mean(eA * eA)), float(np.mean(eB * eB)), float(np.mean(eA * eB))


def block_rmse(y, p, blocks):
    return pd.Series({b: rmse(y[blocks == b], p[blocks == b]) for b in np.unique(blocks)})


rows, jk_rows = [], []
for tg in base.regions:
    y, blocks = base.y(tg), base.blocks(tg)
    S, C, Cc = base.anchor("stefan", tg), base.anchor("cci_raw", tg), base.anchor("cci_cal", tg)
    SC, SCc = 0.5 * (S + C), 0.5 * (S + Cc)
    eS, eC, eCc = S - y, C - y, Cc - y
    n = len(y)
    r = dict(target=tg, n_cells=n, n_blocks=int(len(np.unique(blocks))), flag=flag_small(n))
    for name, e in [("S", eS), ("C", eC), ("Ccal", eCc), ("SC", SC - y), ("SCcal", SCc - y)]:
        r[f"rmse_{name}"] = round(float(np.sqrt(np.mean(e ** 2))), 3)
        r[f"bias_{name}"] = round(float(np.mean(e)), 3)
        r[f"sd_{name}"] = round(float(np.std(e)), 3)
    r["rho_S_C"] = round(float(np.corrcoef(eS, eC)[0, 1]), 3) if n > 2 else np.nan
    r["rho_S_Ccal"] = round(float(np.corrcoef(eS, eCc)[0, 1]), 3) if n > 2 else np.nan
    for suf, e2, rm_combo in [("SC", eC, r["rmse_SC"]), ("SCcal", eCc, r["rmse_SCcal"])]:
        mSS, mCC, mSC = moments(eS, e2)
        den = mSS + mCC - 2 * mSC
        w = (mCC - mSC) / den if den > 1e-9 else np.nan
        r[f"w_star_{suf}"] = round(float(w), 3) if np.isfinite(w) else np.nan
        r[f"rmse_wstar_{suf}"] = round(float(np.sqrt(np.mean((w * eS + (1 - w) * e2) ** 2))), 3) if np.isfinite(w) else np.nan
        bS, b2, vS, v2 = eS.mean(), e2.mean(), eS.var(), e2.var()
        rm_indep = float(np.sqrt(((bS + b2) / 2) ** 2 + (vS + v2) / 4))
        r[f"rmse_{suf}_if_indep"] = round(rm_indep, 3)
        d_obs, d_ind = rm_combo - r["rmse_S"], rm_indep - r["rmse_S"]
        r[f"delta_{suf}_obs"] = round(float(d_obs), 3)
        r[f"delta_{suf}_if_indep"] = round(float(d_ind), 3)
        r[f"gain_realized_{suf}"] = round(float(d_obs / d_ind), 3) if d_ind < -1e-9 else np.nan
    # 블록 수준
    bS_, bSC_ = block_rmse(y, S, blocks), block_rmse(y, SC, blocks)
    r["rmse_S_blockEW"] = round(float(bS_.mean()), 3)
    r["rmse_SC_blockEW"] = round(float(bSC_.mean()), 3)
    r["delta_SC_blockEW"] = round(float(bSC_.mean() - bS_.mean()), 3)
    r["blocks_SC_better"] = int((bSC_ < bS_).sum())
    r["frac_blocks_SC_better"] = round(float((bSC_ < bS_).mean()), 3)
    d_all = r["rmse_SC"] - r["rmse_S"]
    jk = {}
    for b in np.unique(blocks):
        m = blocks != b
        if m.sum() == 0:
            continue
        jk[b] = rmse(y[m], SC[m]) - rmse(y[m], S[m])
        jk_rows.append(dict(target=tg, block=int(b), n_block=int((~m).sum()), delta_SC_without_block=round(jk[b], 3),
                            rmse_S_block=round(float(bS_[b]), 3), rmse_SC_block=round(float(bSC_[b]), 3)))
    jkv = pd.Series(jk)
    if len(jkv):
        infl = (jkv - d_all).abs().idxmax()
        r["delta_SC_all"] = round(float(d_all), 3)
        r["jk_delta_min"] = round(float(jkv.min()), 3)
        r["jk_delta_max"] = round(float(jkv.max()), 3)
        r["jk_most_influential_block"] = int(infl)
        r["jk_block_n"] = int((blocks == infl).sum())
        r["jk_delta_without_it"] = round(float(jkv[infl]), 3)
        r["jk_sign_consistent"] = round(float(np.mean(np.sign(jkv.values) == np.sign(d_all))), 3) if d_all != 0 else np.nan
    d0, lo, hi, _, _ = paired_boot(y, SC, S, blocks, nboot=args.nboot)
    r["h1_delta"], r["h1_ci_lo"], r["h1_ci_hi"] = round(d0, 3), round(lo, 3) if np.isfinite(lo) else np.nan, round(hi, 3) if np.isfinite(hi) else np.nan
    d0c, loc_, hic, _, _ = paired_boot(y, SCc, S, blocks, nboot=args.nboot)
    r["h1cal_delta"], r["h1cal_ci_lo"], r["h1cal_ci_hi"] = round(d0c, 3), round(loc_, 3) if np.isfinite(loc_) else np.nan, round(hic, 3) if np.isfinite(hic) else np.nan
    rows.append(r)
    print(f"  [{tg}] n={n} rmse S={r['rmse_S']} C={r['rmse_C']} SC={r['rmse_SC']} ρ={r['rho_S_C']} w*={r['w_star_SC']} "
          f"Δobs={r['delta_SC_obs']} Δindep={r['delta_SC_if_indep']} blockEW Δ={r['delta_SC_blockEW']} majority={r['frac_blocks_SC_better']}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUT / "a2_combo_decomp.csv", index=False)
pd.DataFrame(jk_rows).to_csv(OUT / "a2_combo_decomp_jackknife.csv", index=False)
(OUT / "a2_combo_decomp_meta.json").write_text(json.dumps(dict(sources=args.sources, nboot=args.nboot, E_alaska=base.k["E"],
    cci_a=base.k["cci_a"], cci_b=base.k["cci_b"], regions=base.regions, elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1))
print("\n=== A2-3 결합 앵커 분해 (정보 없음, cm) ===")
cols = ["target", "n_cells", "n_blocks", "flag", "rmse_S", "rmse_C", "rmse_Ccal", "rmse_SC", "rmse_SCcal", "bias_S", "bias_C", "sd_S", "sd_C",
        "rho_S_C", "rho_S_Ccal", "w_star_SC", "rmse_wstar_SC", "delta_SC_obs", "delta_SC_if_indep", "gain_realized_SC",
        "delta_SC_blockEW", "frac_blocks_SC_better", "jk_delta_min", "jk_delta_max", "jk_most_influential_block", "jk_block_n",
        "jk_sign_consistent", "h1_delta", "h1_ci_lo", "h1_ci_hi"]
print(out[cols].to_string(index=False))
print(f"\n저장: {OUT/'a2_combo_decomp.csv'} · {time.time()-t0:.0f}s")
