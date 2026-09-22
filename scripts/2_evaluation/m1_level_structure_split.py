"""수준·구조 분리 가설(T0b) 사후 검정 — M1 모델 축 예측 벡터로 GPU 없이 채점.

가설: 전이 오차의 주성분은 지역 수준(계수 E) 편향이며, ML이 배운 '지역 안 공간 구조'(공변량에 따른 편차)는
지역을 넘어 옮겨질 수 있다. 검정: ML 출력에서 대상 지역 평균을 빼고(공변량만 사용, 라벨 미사용) 물리 앵커에 얹는다.
  (i)  잔차 중심화: anchor(x) + λ·(g(x) − mean_target g)
  (ii) 직접 구조:   mean_target anchor + (p(x) − mean_target p)     (p = 직접 회귀 예측)
비교 기준: anchor 단독, anchor + λ·g(현행). 짝지은 블록 부트스트랩 400회.
입력 data/processed/m1/m1_model_shard*_preds.npz  산출 data/processed/m1/m1_level_structure.csv
"""
import glob, sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; M1 = ROOT / "data/processed/m1"
G, ANCH, EVAL = {}, {}, {}
for f in sorted(glob.glob(str(M1 / "m1_model_shard*_preds.npz"))):
    z = np.load(f)
    for k in z.files:
        kind, rest = k.split("::", 1)
        if kind == "g": G[rest] = z[k]
        elif kind == "anchor": ANCH[rest] = z[k]
        elif kind == "eval":
            tk, fld = rest.rsplit("::", 1); EVAL.setdefault(tk, {})[fld] = z[k]
AB4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
MODELS = ["ridge", "catboost_lo", "catboost", "mlp", "ftt", "tabm", "cfm", "ddpm", "nflow"]
rng = np.random.RandomState(0)
def rmse(y, p): m = np.isfinite(y) & np.isfinite(p); return float(np.sqrt(np.mean((y[m]-p[m])**2)))
def boot(y, b, pA, pB, n=400):
    ub = np.unique(b); pos = {u: np.where(b == u)[0] for u in ub}; out = []
    for _ in range(n):
        idx = np.concatenate([pos[u] for u in rng.choice(ub, len(ub), replace=True)])
        out.append(rmse(y[idx], pA[idx]) - rmse(y[idx], pB[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
rows = []
for cond in ["noinfo", "covonly", "labels"]:
    for tg in AB4:
        tk = f"{cond}|{tg}|0"
        if tk not in EVAL: continue
        y, b = EVAL[tk]["y"].astype(float), EVAL[tk]["block"]
        for anc in ["stefan", "stefan_cci"]:
            a = ANCH.get(f"{tk}|alaska|{anc}")
            if a is None: continue
            a = a.astype(float); r_anc = rmse(y, a)
            for m in MODELS:
                gs = [G[k].astype(float) for k in G if k.startswith(f"{tk}|") and k.endswith(f"|alaska|x25|{anc}|none|0.0|{m}|0")]
                ps = [G[k].astype(float) for k in G if k.startswith(f"{tk}|") and k.endswith(f"|alaska|x25|none|none|0.0|{m}|0")]
                if not gs: continue
                g = np.mean(gs, 0)
                for lam in [0.25, 0.5, 1.0]:
                    p_std = a + lam * g
                    p_cen = a + lam * (g - np.nanmean(g))
                    lo, hi = boot(y, b, p_cen, a); lo2, hi2 = boot(y, b, p_cen, p_std)
                    rows.append(dict(cond=cond, target=tg, anchor=anc, model=m, variant="resid_centered", lam=lam, n=len(y), n_blocks=len(np.unique(b)),
                                     rmse_anchor=r_anc, rmse_standard=rmse(y, p_std), rmse_centered=rmse(y, p_cen),
                                     d_vs_anchor=rmse(y, p_cen) - r_anc, ci_lo=lo, ci_hi=hi, d_vs_standard=rmse(y, p_cen) - rmse(y, p_std), ci2_lo=lo2, ci2_hi=hi2))
                if ps:
                    p = np.mean(ps, 0); p_cen = np.nanmean(a) + (p - np.nanmean(p))
                    lo, hi = boot(y, b, p_cen, a)
                    rows.append(dict(cond=cond, target=tg, anchor=anc, model=m, variant="direct_structure", lam=1.0, n=len(y), n_blocks=len(np.unique(b)),
                                     rmse_anchor=r_anc, rmse_standard=rmse(y, p), rmse_centered=rmse(y, p_cen),
                                     d_vs_anchor=rmse(y, p_cen) - r_anc, ci_lo=lo, ci_hi=hi, d_vs_standard=rmse(y, p_cen) - rmse(y, p), ci2_lo=np.nan, ci2_hi=np.nan))
df = pd.DataFrame(rows); df.to_csv(M1 / "m1_level_structure.csv", index=False)
pd.set_option("display.width", 250)
for cond in ["noinfo", "covonly"]:
    d = df[(df.cond == cond) & (df.anchor == "stefan")]
    print(f"\n=== {cond} · Stefan 앵커 · Δ = 중심화 − 앵커 단독 (음수 = 개선), 4지역 평균 ===")
    print(d.groupby(["variant", "model", "lam"]).d_vs_anchor.mean().unstack("lam").round(2).to_string())
    print(f"--- {cond} 지역별, 잔차 중심화 λ=0.5 catboost_lo·mlp / 직접 구조 catboost_lo·mlp ---")
    q = d[((d.variant == "resid_centered") & (d.lam == 0.5) | (d.variant == "direct_structure")) & (d.model.isin(["catboost_lo", "mlp", "catboost"]))]
    print(q[["target", "variant", "model", "rmse_anchor", "rmse_standard", "rmse_centered", "d_vs_anchor", "ci_lo", "ci_hi"]].round(2).to_string(index=False))
print("saved m1_level_structure.csv", len(df))
