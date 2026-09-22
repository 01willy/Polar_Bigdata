"""H21 지역 간 계수 대여 + H20 시뮬레이터 상대 계수(CCI) + 물리식 조건 엄밀화 — 해석적(CPU, 수 분).

설계 docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md §3·§4. 채점 규약 개정 09-21(polar.m1_stats).

요인 = Stefan 계수 E 의 출처. 나머지(강제력 e5_sqrt_tdd, 평가 셀, 채점) 고정. ML 없음.
  대상 t ∈ TRANSFER_MAIN(정보 없음: 전체 평가 셀). 공여 S = 라벨 지역 7 − {t}. 심부 지역 제외.
  공여 지역 E_s = 지역 최소제곱(블록 < 8 지역은 κ=10 으로 E_AK 쪽 수축).
  유사도(라벨 미사용): geo(중심 대권거리) · cov(지역 중앙값 x14 표준화 유클리드) · phys(MAAT·TDD·SWE·SOC).
  추정량: nearest · kernel(τ = 공여 간 거리 중앙값) · 수축 w·E_est + (1−w)·E_AK (w ∈ {.25,.5,.75,1}) · 대조 pooled_cell · pooled_region · E_AK · oracle(자체 E, 상한).
  중첩 선택: 대상 t 를 채점할 때 공여끼리 leave-one-donor-out(비알래스카 공여를 유사 대상으로) 지역 등가중 RMSE 최소 구성을 고른 뒤 t 에 적용.
H20: cci_cal(affine) · cci_scale(1-파라미터 c = E_AK/median_AK(CCI/√TDD)) · cci_blockE(c·median_block(CCI/√TDD)·√TDD) · +Stefan 등가중.
물리식 조건 엄밀화(기술 통계, 자체 라벨): 멱지수 b, 대기/토양 도일 자체 계수 RMSE, Ku 보정, n-factor 대리, 블록 E 변동계수; 강제력 중첩 선택 규칙.

산출 data/processed/h2/h21_{summary,tests,nested,physics_rules,region_desc}.csv, h21_preds.npz, h21_meta.json
실행(ROOT): python3 scripts/3_deep_learning/h21_coef_borrow.py
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import TARGET, TERRAIN, CLIMATE, TRANSFER_MAIN                                  # noqa: E402
from polar.m1_core import load_base, eval_mask, half_split_blocks, fit_coefs                        # noqa: E402
from polar.m1_ext import haversine_km                                                               # noqa: E402
from polar.m1_stats import boot_delta, summarize_delta, seed_of                                     # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "h2"; OUT.mkdir(exist_ok=True)
NBOOT = 1000
KAPPA_SMALL = 10.0
W_GRID = [0.25, 0.5, 0.75, 1.0]
SIMS = ["geo", "cov", "phys"]
X14 = list(TERRAIN + CLIMATE)
PHYS4 = ["e5_maat", "e5_tdd", "e5_swe", "sg_soc_5_15"]
LABEL_REGIONS = ["Alaska"] + list(TRANSFER_MAIN)
t0 = time.time()

DF = load_base(PROC)
DF["s"] = DF.e5_sqrt_tdd.values.astype(float)
DF["y"] = DF[TARGET].values.astype(float)
AK = DF[DF.macro == "Alaska"]
K_AK = fit_coefs(AK)
E_AK = float(K_AK["E"])


def ls_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m])) if m.sum() >= 1 else np.nan


# ---------------------------------------------------------------- 지역 기술자 (E 는 라벨, 기술자는 공변량만)
lab_cells = DF[DF.macro.isin(LABEL_REGIONS)]
SD14 = lab_cells[X14].std().values.astype(float) + 1e-9
SD4 = lab_cells[PHYS4].std().values.astype(float) + 1e-9
REG = {}
for r in LABEL_REGIONS:
    d = DF[DF.macro == r]
    E = ls_E(d.y.values, d.s.values)
    nb = int(d.block.nunique())
    E_use = E if nb >= 8 else (len(d) * E + KAPPA_SMALL * E_AK) / (len(d) + KAPPA_SMALL)
    REG[r] = dict(region=r, n=int(len(d)), n_blocks=nb, E_ls=E, E_used=float(E_use), shrunk=bool(nb < 8),
                  lat=float(d.lat.mean()), lon=float(d.lon.mean()),
                  m14=np.nanmedian(d[X14].values.astype(float), 0), m4=np.nanmedian(d[PHYS4].values.astype(float), 0),
                  E_soil=ls_E(d.y.values, d.e5_sqrt_tdd_soil.values.astype(float)))
pd.DataFrame([{k: v for k, v in x.items() if k not in ("m14", "m4")} for x in REG.values()]).to_csv(OUT / "h21_region_desc.csv", index=False)


def dist(t, s, sim):
    a, b = REG[t], REG[s]
    if sim == "geo":
        return float(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]))
    if sim == "cov":
        return float(np.sqrt(np.nanmean(((a["m14"] - b["m14"]) / SD14) ** 2)))
    if sim == "phys":
        return float(np.sqrt(np.nanmean(((a["m4"] - b["m4"]) / SD4) ** 2)))
    raise ValueError(sim)


def configs():
    cs = [dict(cfg="E_AK", kind="E_AK", sim="", w=0.0), dict(cfg="pooled_cell", kind="pooled_cell", sim="", w=0.0),
          dict(cfg="pooled_region", kind="pooled_region", sim="", w=0.0)]
    for sim in SIMS:
        for kind in ("nearest", "kernel"):
            for w in W_GRID:
                cs.append(dict(cfg=f"{kind}_{sim}_w{w:g}", kind=kind, sim=sim, w=w))
    return cs


CONFIGS = configs()


def estimate_E(t, donors, c):
    """대상 t, 공여 집합 donors(t 제외), 구성 c → E. 라벨은 공여 E 에만."""
    if c["kind"] == "E_AK":
        return REG["Alaska"]["E_used"] if "Alaska" in donors else float(np.mean([REG[s]["E_used"] for s in donors]))
    if c["kind"] == "pooled_cell":
        d = DF[DF.macro.isin(donors)]
        return ls_E(d.y.values, d.s.values)
    if c["kind"] == "pooled_region":
        return float(np.mean([REG[s]["E_used"] for s in donors]))
    ds = np.array([dist(t, s, c["sim"]) for s in donors]); Es = np.array([REG[s]["E_used"] for s in donors])
    if c["kind"] == "nearest":
        est = float(Es[np.argmin(ds)])
    else:
        dd = [dist(a, b, c["sim"]) for i, a in enumerate(donors) for b in donors[i + 1:]]
        tau = float(np.median(dd)) if dd else 1.0
        wgt = np.exp(-ds / max(tau, 1e-9)); wgt /= wgt.sum()
        est = float(wgt @ Es)
    base = REG["Alaska"]["E_used"] if "Alaska" in donors else float(np.mean(Es))
    return c["w"] * est + (1 - c["w"]) * base


def region_rmse(r, E):
    d = DF[(DF.macro == r)]
    m = eval_mask(d)
    y, s = d.y.values[m], d.s.values[m]
    return float(np.sqrt(np.mean((E * s - y) ** 2)))


# ---------------------------------------------------------------- 중첩 선택 + 전 구성 표
summary, nested_rows, PRED, EVAL = [], [], {}, {}
for t in TRANSFER_MAIN + ["Alaska"]:
    donors = [r for r in LABEL_REGIONS if r != t]
    pseudo = [r for r in donors if r != "Alaska"] if t != "Alaska" else donors
    # leave-one-donor-out 점수(지역 등가중·셀 가중 둘 다 기록, 선택은 등가중)
    lodo = {}
    for c in CONFIGS:
        sc = []
        for p in pseudo:
            inner = [r for r in donors if r != p]
            if c["kind"] == "E_AK" and "Alaska" not in inner:
                continue
            sc.append(region_rmse(p, estimate_E(p, inner, c)))
        lodo[c["cfg"]] = float(np.mean(sc)) if sc else np.inf
    best = min(CONFIGS, key=lambda c: (lodo[c["cfg"]], CONFIGS.index(c)))
    d = DF[DF.macro == t]; m = eval_mask(d)
    y, s, blk = d.y.values[m], d.s.values[m], d.block.values[m]
    EVAL[t] = dict(y=y, block=blk, loc_id=d.loc_id.values[m])
    nested_rows.append(dict(target=t, chosen=best["cfg"], lodo_rmse=lodo[best["cfg"]], E_chosen=estimate_E(t, donors, best),
                            E_AK=REG["Alaska"]["E_used"], E_oracle=REG[t]["E_ls"],
                            **{f"lodo_{k}": v for k, v in lodo.items()}))
    for c in CONFIGS + [dict(cfg="nested", kind="nested", sim="", w=0.0), dict(cfg="oracle", kind="oracle", sim="", w=0.0)]:
        if c["kind"] == "nested":
            E = estimate_E(t, donors, best)
        elif c["kind"] == "oracle":
            E = REG[t]["E_ls"]
        elif c["kind"] == "E_AK" and t == "Alaska":
            continue
        else:
            E = estimate_E(t, donors, c)
        p = E * s
        PRED[(t, c["cfg"])] = p
        summary.append(dict(target=t, cfg=c["cfg"], kind=c["kind"], sim=c["sim"], w=c["w"], E=E, n=len(y), n_blocks=int(len(np.unique(blk))),
                            rmse=float(np.sqrt(np.mean((p - y) ** 2))), bias=float(np.mean(p - y)), lodo_rmse=lodo.get(c["cfg"], np.nan)))

# ---------------------------------------------------------------- H20 (CCI 상대 계수) 수준
c_ak = AK.cci_alt.values.astype(float); s_ak = AK.s.values
mm = np.isfinite(c_ak) & (s_ak > 0)
C_SCALE = E_AK / float(np.median(c_ak[mm] / s_ak[mm]))
DF["E_cci"] = DF.cci_alt.values.astype(float) / np.where(DF.s.values > 0, DF.s.values, np.nan)
DF["E_cci_block"] = DF.groupby("block").E_cci.transform("median")
for t in TRANSFER_MAIN + ["Alaska"]:
    d = DF[DF.macro == t]; m = eval_mask(d)
    y, s, cci = d.y.values[m], d.s.values[m], d.cci_alt.values.astype(float)[m]
    eb = d.E_cci_block.values[m]
    st = E_AK * s
    levels = {"stefan": st, "cci_cal": K_AK["cci_a"] + K_AK["cci_b"] * cci, "cci_scale": C_SCALE * cci,
              "cci_blockE": C_SCALE * eb * s, "cci_blockE_stefan": 0.5 * (C_SCALE * eb * s + st),
              "stefan_cci": 0.5 * (st + cci)}
    if t != "Alaska":
        levels["nested_cci_cal"] = 0.5 * (PRED[(t, "nested")] + levels["cci_cal"])
    for k, p in levels.items():
        PRED[(t, k)] = p
        summary.append(dict(target=t, cfg=k, kind="h20" if "cci" in k else "anchor", sim="", w=0.0, E=np.nan, n=len(y),
                            n_blocks=int(len(np.unique(d.block.values[m]))), rmse=float(np.sqrt(np.nanmean((p - y) ** 2))),
                            bias=float(np.nanmean(p - y)), lodo_rmse=np.nan))

# ---------------------------------------------------------------- 공변량만(B 셀, 분할 3 풀링) 재채점 — 같은 예측의 부분집합
cov_rows = []
for t in TRANSFER_MAIN:
    t_idx = np.where(DF.macro.values == t)[0]
    for sp in range(3):
        A_idx, B_idx = half_split_blocks(DF, t_idx, sp)
        evB = B_idx[eval_mask(DF.iloc[B_idx])]
        sel = np.isin(EVAL[t]["loc_id"], DF.loc_id.values[evB])
        if sel.sum() < 3:
            continue
        for (tt, cfg), p in PRED.items():
            if tt != t:
                continue
            y = EVAL[t]["y"][sel]
            cov_rows.append(dict(target=t, split=sp, cfg=cfg, n=int(sel.sum()), rmse=float(np.sqrt(np.nanmean((p[sel] - y) ** 2)))))
cov = pd.DataFrame(cov_rows)
cov_mean = cov.groupby(["target", "cfg"]).rmse.mean().rename("rmse_covonlyB").reset_index()
summ = pd.DataFrame(summary).merge(cov_mean, on=["target", "cfg"], how="left")
summ.to_csv(OUT / "h21_summary.csv", index=False)
pd.DataFrame(nested_rows).to_csv(OUT / "h21_nested.csv", index=False)

# ---------------------------------------------------------------- 검정 (정보 없음 6지역, 층화 부트스트랩)
TESTS = [("H21_nested_vs_EAK", "nested", "E_AK"), ("H21ctrl_pooled_region_vs_EAK", "pooled_region", "E_AK"),
         ("H21ctrl_pooled_cell_vs_EAK", "pooled_cell", "E_AK"), ("H21ref_oracle_vs_EAK", "oracle", "E_AK"),
         ("H20_cci_blockE_vs_stefan", "cci_blockE", "stefan"), ("H20x_cci_scale_vs_cci_cal", "cci_scale", "cci_cal"),
         ("H20x_cci_blockE_vs_cci_cal", "cci_blockE", "cci_cal"), ("H20x_cci_blockE_stefan_vs_stefan_cci", "cci_blockE_stefan", "stefan_cci"),
         ("H21x_nested_cci_cal_vs_stefan_cci", "nested_cci_cal", "stefan_cci")]
for c in CONFIGS:
    if c["kind"] not in ("E_AK",):
        TESTS.append((f"X_{c['cfg']}_vs_EAK", c["cfg"], "E_AK"))
test_rows = []
for name, a, b in TESTS:
    per = {}
    for t in TRANSFER_MAIN:
        if (t, a) not in PRED or (t, b) not in PRED:
            continue
        per[t] = boot_delta(EVAL[t]["y"], EVAL[t]["block"], PRED[(t, a)][None, :], PRED[(t, b)][None, :], NBOOT, seed_of(name, t))
    test_rows += summarize_delta(name, per, TRANSFER_MAIN, 0)
    # 4지역(AB4) 층화 요약도 병기
    test_rows += [dict(r, test=name + "|AB4") for r in summarize_delta(name, {k: v for k, v in per.items() if k in ["Lena", "Canada", "Russia_W", "Russia_E"]},
                                                                      ["Lena", "Canada", "Russia_W", "Russia_E"], 0) if str(r["target"]).startswith("MEAN")]
pd.DataFrame(test_rows).to_csv(OUT / "h21_tests.csv", index=False)

# ---------------------------------------------------------------- 물리식 조건 엄밀화(자체 라벨, 기술 통계)
phys_rows = []
for r in LABEL_REGIONS:
    d = DF[DF.macro == r]; m = eval_mask(d)
    y, s, tdd = d.y.values[m], d.s.values[m], d.e5_tdd.values.astype(float)[m]
    ss = d.e5_sqrt_tdd_soil.values.astype(float)[m]
    p4 = d.p4_ku.values.astype(float)[m]
    ok = (y > 0) & (tdd > 0)
    b, lnE = np.polyfit(np.log(tdd[ok]), np.log(y[ok]), 1) if ok.sum() >= 5 else (np.nan, np.nan)
    E_a, E_s = ls_E(y, s), ls_E(y, ss)
    okk = np.isfinite(p4) & (p4 > 0)
    c_ku = float((p4[okk] @ y[okk]) / (p4[okk] @ p4[okk])) if okk.sum() >= 3 else np.nan
    blocks = d.block.values[m]
    Eb = [ls_E(y[blocks == bb], s[blocks == bb]) for bb in np.unique(blocks) if (blocks == bb).sum() >= 3]
    nf = d.e5_tdd_soil.values.astype(float)[m] / np.where(tdd > 0, tdd, np.nan)
    phys_rows.append(dict(region=r, n=int(m.sum()), n_blocks=int(len(np.unique(blocks))), b_exponent=float(b), E_air_own=E_a, E_soil_own=E_s,
                          rmse_air_own=float(np.sqrt(np.mean((E_a * s - y) ** 2))), rmse_soil_own=float(np.sqrt(np.nanmean((E_s * ss - y) ** 2))),
                          rmse_powerlaw_own=float(np.sqrt(np.mean((np.exp(lnE) * tdd ** b - y) ** 2))) if np.isfinite(b) else np.nan,
                          rmse_ku_own=float(np.sqrt(np.nanmean((c_ku * p4 - y) ** 2))) if np.isfinite(c_ku) else np.nan, c_ku_own=c_ku,
                          rmse_air_EAK=float(np.sqrt(np.mean((E_AK * s - y) ** 2))), rmse_soil_EAK=float(np.sqrt(np.nanmean((K_AK["E_soil"] * ss - y) ** 2))),
                          nfactor_soil_air_median=float(np.nanmedian(nf)), block_E_cv=float(np.std(Eb) / np.mean(Eb)) if len(Eb) >= 2 else np.nan,
                          n_blocks_E=len(Eb), E_ratio_vs_AK=E_a / E_AK))
# 강제력 중첩 선택 규칙: 대상 t 의 공여(비알래스카)에서 알래스카 계수의 air vs soil 이 더 좋은 쪽을 골라 t 에 적용
rule_rows = []
for t in TRANSFER_MAIN:
    donors = [r for r in TRANSFER_MAIN if r != t]
    pr = pd.DataFrame(phys_rows).set_index("region")
    air = pr.loc[donors, "rmse_air_EAK"].mean(); soil = pr.loc[donors, "rmse_soil_EAK"].mean()
    pick = "air" if air <= soil else "soil"
    rule_rows.append(dict(target=t, donors_air=air, donors_soil=soil, pick=pick, target_air=pr.loc[t, "rmse_air_EAK"], target_soil=pr.loc[t, "rmse_soil_EAK"],
                          target_applied=pr.loc[t, f"rmse_{pick}_EAK"]))
pd.DataFrame(phys_rows).to_csv(OUT / "h21_physics_rules.csv", index=False)
pd.DataFrame(rule_rows).to_csv(OUT / "h21_forcing_rule.csv", index=False)

np.savez_compressed(OUT / "h21_preds.npz", **{f"pred::{t}::{c}": v for (t, c), v in PRED.items()},
                    **{f"eval::{t}::{f}": np.asarray(v[f]) for t, v in EVAL.items() for f in ("y", "block", "loc_id")})
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:                                                                                     # noqa: BLE001
    commit = "NA"
(OUT / "h21_meta.json").write_text(json.dumps(dict(
    stage="H21/H20", design="docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md", E_AK=E_AK, C_SCALE=C_SCALE, kappa_small=KAPPA_SMALL,
    w_grid=W_GRID, sims=SIMS, configs=[c["cfg"] for c in CONFIGS], nboot=NBOOT, regions={r: {k: v for k, v in x.items() if k not in ("m14", "m4")} for r, x in REG.items()},
    git_commit=commit, elapsed_s=round(time.time() - t0, 1)), ensure_ascii=False, indent=1, default=float))
tt = pd.DataFrame(test_rows)
print(tt[tt.target.str.startswith("MEAN") & ~tt.test.str.startswith("X_")][["test", "target", "rmse_A", "rmse_B", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_flag"]].to_string())
print(pd.DataFrame(nested_rows)[["target", "chosen", "E_chosen", "E_AK", "E_oracle", "lodo_rmse"]].to_string())
print(f"done {time.time()-t0:.0f}s")
