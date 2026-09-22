"""H30 배포 규칙 확인적 검정(C3) + 해로운 조합 표(A3) — 기존 예측·결과 재사용, CPU.

계획 docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md §3 H30·§4 A3·C3.
C3: M1 모델 축 예측(정보 없음, x25, seed 3)과 셀별 DI(a2_shift_cells)로 규칙별 예측을 만들고 seed 짝지음·층화 블록 부트스트랩으로
    (aoa_resid − stefan), (aoa_direct − ml_direct), (aoa_direct − stefan), (cci_agree20 − stefan_cci) 를 검정. 지역별 최악 오차 표 병기.
A3: 방법 × n 의 Δ(대 물리식) 통합표 — h25(S1·S1refit·S2·S3 α, 규칙 random/kmedoid), h24(block_rr·farthest·stefan_strat·geo_strat), h23(maml·finetune), M1 정보 없음 직접 회귀.
산출 data/processed/h3/h30_deploy_tests.csv, h30_deploy_worst.csv, h3_hurts.csv
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.m1_stats import boot_delta, summarize_delta, seed_of                                     # noqa: E402

PROC = ROOT / "data" / "processed"; OUT = PROC / "h3"; OUT.mkdir(exist_ok=True); H2 = PROC / "h2"
MAIN6 = ["Lena", "Canada", "Russia_W", "Russia_C", "Russia_E", "Greenland"]; AB4 = ["Lena", "Canada", "Russia_W", "Russia_E"]

# ---------------------------------------------------------------- C3 배포 규칙
P, EV = {}, {}
for f in sorted(glob.glob(str(PROC / "m1" / "m1_model_shard*_preds.npz"))):
    z = np.load(f, allow_pickle=True)
    for k in z.files:
        if k.startswith("g::noinfo") or k.startswith("anchor::noinfo"):
            P[k] = z[k]
        elif k.startswith("eval::noinfo"):
            _, tk, fld = k.split("::"); EV.setdefault(tk, {})[fld] = z[k]
di = pd.read_csv(PROC / "m1" / "a2_shift_cells.csv").set_index("loc_id")
base = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=["loc_id", "cci_alt"]).set_index("loc_id")


def g_of(tk, model, anchor):
    ks = sorted(k for k in P if k.startswith(f"g::{tk}|") and k.endswith(f"|alaska|x25|{anchor}|none|0.0|{model}|0"))
    return np.stack([P[k] for k in ks]) if ks else None


RULES, worst = {}, []
for tg in MAIN6:
    tk = f"noinfo|{tg}|0"
    if tk not in EV:
        continue
    y, loc, blk = EV[tk]["y"], EV[tk]["loc_id"], EV[tk]["block"]
    st = P[f"anchor::{tk}|alaska|stefan"]; cci = base.loc[loc, "cci_alt"].values
    inaoa = di.reindex(loc).in_aoa_x25.values.astype(bool)
    gd, gr = g_of(tk, "catboost_lo", "none"), g_of(tk, "catboost_lo", "stefan")
    S = gd.shape[0]
    R = {"stefan": np.repeat(st[None, :], S, 0), "ml_direct": gd, "ml_resid": st[None, :] + 0.25 * gr,
         "aoa_direct": np.where(inaoa[None, :], gd, st[None, :]), "aoa_resid": np.where(inaoa[None, :], st[None, :] + 0.25 * gr, st[None, :]),
         "stefan_cci": np.repeat((0.5 * (st + cci))[None, :], S, 0)}
    agree = np.abs(st - cci) <= 20
    R["cci_agree20"] = np.repeat(np.where(agree, 0.5 * (st + cci), st)[None, :], S, 0)
    RULES[tg] = (R, y, blk, float(inaoa.mean()))
    for name, p in R.items():
        ok = np.all(np.isfinite(p), 0) & np.isfinite(y)
        worst.append(dict(target=tg, rule=name, rmse=float(np.mean([np.sqrt(np.mean((p[s][ok] - y[ok]) ** 2)) for s in range(S)])), aoa_frac=float(inaoa.mean()), n=int(ok.sum())))
W = pd.DataFrame(worst); W.to_csv(OUT / "h30_deploy_worst.csv", index=False)
tests = []
for name, a, b in [("H30_aoa_resid_vs_stefan", "aoa_resid", "stefan"), ("H30_aoa_resid_vs_ml_direct", "aoa_resid", "ml_direct"), ("H30_aoa_direct_vs_ml_direct", "aoa_direct", "ml_direct"), ("H30x_aoa_direct_vs_stefan", "aoa_direct", "stefan"),
                   ("H30x_ml_resid_vs_stefan", "ml_resid", "stefan"), ("H30x_cci_agree20_vs_stefan_cci", "cci_agree20", "stefan_cci"), ("H30x_stefan_cci_vs_stefan", "stefan_cci", "stefan")]:
    per = {tg: boot_delta(y, blk, R[a], R[b], 1000, seed_of(name, tg)) for tg, (R, y, blk, _) in RULES.items()}
    tests += [dict(family="deploy", cond="noinfo", **r) for r in summarize_delta(name, per, MAIN6, 0)]
Tt = pd.DataFrame(tests); Tt.to_csv(OUT / "h30_deploy_tests.csv", index=False)
pd.set_option("display.width", 240)
print(W.pivot_table(index="rule", columns="target", values="rmse").round(2).assign(worst_AB4=lambda d: d[AB4].max(1), mean_AB4=lambda d: d[AB4].mean(1)).round(2).to_string())
print(Tt[Tt.target.astype(str).str.startswith("MEAN")][["test", "target", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_flag"]].round(2).to_string(index=False))

# ---------------------------------------------------------------- A3 해로운 조합 표
rows = []
f25 = OUT / "h25_curve.csv"
if f25.exists():
    c = pd.read_csv(f25); c = c[(c.scope == "n") & (c.target.isin(AB4))]
    for _, r in c.iterrows():
        if r.stage == "S3" and r.lam != 0.25:
            continue
        lab = {"S1": "E 수축(κ=10)", "S1refit": "E 재적합", "S2": "E 수축+유사라벨 증강 ML", "S3": f"E 수축 앵커+잔차 ML(α={r.alpha:g})"}[r.stage]
        rows.append(dict(source="h25", method=f"{lab} · {r.rule}", stage=r.stage, rule=r.rule, alpha=r.alpha, target=r.target, n=int(r.n), d_phys=r.d_phys_mean,
                         ci_lo=r.d_phys_lo, ci_hi=r.d_phys_hi, win=r.win_rate))
f24 = H2 / "h24_curve.csv"
if f24.exists():
    c = pd.read_csv(f24); c = c[(c.target.isin(AB4)) & (c.method == "shrink_k10") & (c.strategy.isin(["block_rr", "farthest", "stefan_strat", "geo_strat"]))]
    lab = {"block_rr": "블록 순환 선택", "farthest": "최원점 선택", "stefan_strat": "Stefan 분위 층화 선택", "geo_strat": "좌표 층화 선택"}
    for _, r in c.iterrows():
        rows.append(dict(source="h24", method=f"E 수축 · {lab[r.strategy]}", stage="S1", rule=r.strategy, alpha=1.0, target=r.target, n=int(r.n), d_phys=r.d_phys_mean, ci_lo=np.nan, ci_hi=np.nan, win=r.win_rate))
f23 = H2 / "h23_summary.csv"
if f23.exists():
    c = pd.read_csv(f23); c = c[(c.target.isin(AB4)) & (c.method.isin(["maml", "finetune"])) & (c.lam == 0.25)]
    lab = {"maml": "MAML 잔차 적응(λ=.25)", "finetune": "ERM 미세조정(λ=.25)"}
    for _, r in c.iterrows():
        rows.append(dict(source="h23", method=lab[r.method], stage="S3", rule="random", alpha=1.0, target=r.target, n=int(r.n), d_phys=r.d_phys, ci_lo=np.nan, ci_hi=np.nan, win=np.nan))
sc = pd.read_csv(PROC / "m1" / "m1_sc_summary.csv")
q = sc[(sc.cond == "noinfo") & (sc.dset == "alaska") & (sc.xset == "x25") & (sc.pseudo == "none") & (sc.anchor == "none") & (sc.resid.isin(["catboost_lo", "mlp"])) & (sc.iw == 0) & (sc.target.isin(AB4))]
st = sc[(sc.cond == "noinfo") & (sc.dset == "alaska") & (sc.xset == "x25") & (sc.pseudo == "none") & (sc.anchor == "stefan") & (sc.lam == 0) & (sc.resid == "catboost_lo") & (sc.iw == 0)].set_index("target").rmse
for _, r in q.iterrows():
    rows.append(dict(source="m1", method=f"직접 회귀 {r.resid}(증강 없음)", stage="direct", rule="none", alpha=1.0, target=r.target, n=0, d_phys=r.rmse - st[r.target], ci_lo=np.nan, ci_hi=np.nan, win=np.nan))
H = pd.DataFrame(rows); H.to_csv(OUT / "h3_hurts.csv", index=False)
if len(H):
    piv = H[H.n.isin([0, 3, 10])].pivot_table(index="method", columns=["target", "n"], values="d_phys").round(2)
    print(piv.to_string())
print("done")
