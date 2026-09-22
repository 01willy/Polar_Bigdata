"""설계 §7 배포 규칙(탐색적): AOA 안 ML / 밖 물리, 앵커 불일치 게이팅 — 기존 M1 모델 축 예측(npz) 재사용, GPU 없음.

규칙
  stefan        Stefan(알래스카 E) 전 셀
  ml_direct     직접 CatBoost 전 셀 / ml_resid: Stefan + .25·CatBoost 잔차 전 셀
  aoa_direct    AOA 안(di_x25 ≤ 임계) 직접 ML, 밖 Stefan  / aoa_resid: 안 잔차 ML, 밖 Stefan
  di_q{q}       DI 하위 q 분위(알래스카 학습 DI 분포 기준이 아니라 대상 셀 DI 순위) 안 ML, 밖 Stefan (q ∈ .25,.5,.75)
  cci_agree{τ}  |Stefan − CCI| ≤ τ cm 이면 Stefan+CCI 등가중, 아니면 Stefan (τ ∈ 10, 20, 30)
지표: 지역별 RMSE(정보 없음 6지역), 지역 평균, 최악 지역, ML 적용 셀 비율. seed 3 평균.
산출 data/processed/h2/h_deploy_gating.csv
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"; OUT = PROC / "h2"
MAIN6 = ["Lena", "Canada", "Russia_W", "Russia_C", "Russia_E", "Greenland"]
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
rows = []
def g_of(tk, model, anchor):
    ks = sorted(k for k in P if k.startswith(f"g::{tk}|") and k.endswith(f"|alaska|x25|{anchor}|none|0.0|{model}|0"))
    return np.stack([P[k] for k in ks]) if ks else None
res = {}
for tg in MAIN6:
    tk = f"noinfo|{tg}|0"
    if tk not in EV:
        continue
    y, loc = EV[tk]["y"], EV[tk]["loc_id"]
    st = P[f"anchor::{tk}|alaska|stefan"]
    cci = base.loc[loc, "cci_alt"].values
    d = di.reindex(loc); dix = d.di_x25.values; inaoa = d.in_aoa_x25.values.astype(bool)
    gd = g_of(tk, "catboost_lo", "none"); gr = g_of(tk, "catboost_lo", "stefan")
    if gd is None or gr is None:
        continue
    ml_d = gd.mean(0); ml_r = st + 0.25 * gr.mean(0)
    rules = {"stefan": (st, 0.0), "ml_direct": (ml_d, 1.0), "ml_resid": (ml_r, 1.0),
             "aoa_direct": (np.where(inaoa, ml_d, st), inaoa.mean()), "aoa_resid": (np.where(inaoa, ml_r, st), inaoa.mean())}
    for q in (0.25, 0.5, 0.75):
        thr = np.nanquantile(dix, q); m = dix <= thr
        rules[f"di_q{q:g}_direct"] = (np.where(m, ml_d, st), m.mean()); rules[f"di_q{q:g}_resid"] = (np.where(m, ml_r, st), m.mean())
    for tau in (10, 20, 30):
        agree = np.abs(st - cci) <= tau
        rules[f"cci_agree{tau}"] = (np.where(agree, 0.5 * (st + cci), st), agree.mean())
    rules["stefan_cci"] = (0.5 * (st + cci), 1.0)
    for name, (p, frac) in rules.items():
        ok = np.isfinite(p) & np.isfinite(y)
        res.setdefault(name, {})[tg] = (float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))), float(frac))
for name, d in res.items():
    r = {tg: d[tg][0] for tg in d}; f = {tg: d[tg][1] for tg in d}
    rows.append(dict(rule=name, **{f"rmse_{tg}": r[tg] for tg in r}, mean6=np.mean(list(r.values())), worst6=max(r.values()),
                     mean_AB4=np.mean([r[t] for t in ["Lena", "Canada", "Russia_W", "Russia_E"]]), worst_AB4=max(r[t] for t in ["Lena", "Canada", "Russia_W", "Russia_E"]),
                     ml_frac_mean=np.mean(list(f.values()))))
R = pd.DataFrame(rows); R.to_csv(OUT / "h_deploy_gating.csv", index=False)
pd.set_option("display.width", 250); print(R.round(2).to_string())
