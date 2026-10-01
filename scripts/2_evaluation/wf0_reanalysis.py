"""WF0 재분석(계산 없음, 사후 서술). 계획 docs/EXPERIMENT_PLAN_WF_2026-10-01.md §2 WF0.

LG 본 실행 곡선(results/rescale_lg/data/processed/lg/lg_curve.csv)과 LGD 약관 확인분 곡선(data/processed/lgd/lgd_curve_lic.csv)에서
30대상(LG 의 (대상, 모드) 27 가운데 그린란드 제외 + LGD 새 지역·확충판 약관 확인분 6)을 골라 다음을 계산한다.
  1. 위험표: 라벨 n ∈ {0, 10, 전량} 마다 방법별 Δ(방법 − P0) 의 중앙값, 90 백분위, 최댓값, 최솟값, 2 cm 넘게 나빠진 대상 수.
  2. 물리 계수 오차와 ML 이득: 라벨 전량에서 재보정 이득(P0 − P1)과 최선 ML 이득(다섯 레시피 가운데 최소 Δ)의 Spearman 순위상관.
     최선 ML 은 사후 선택이라 낙관적이다(WF4 가 규칙 선택으로 다시 시험한다).
조건: 셀 무작위 추출, 대상 가중 1, 학습기 catboost_lo(물리식은 none), 점 추정만인 대상 제외. 판정어를 쓰지 않는다.
산출: data/processed/wf/wf0_risk.csv, wf0_misspec.csv, wf0_meta.json. 화면에는 요약만 쓴다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
LG = ROOT / "results/rescale_lg/data/processed/lg/lg_curve.csv"
LGD = ROOT / "data/processed/lgd/lgd_curve_lic.csv"
OUT = ROOT / "data/processed/wf"
KEEP_LGD = ["Tibet_LGD", "NAtlantic~lic", "Russia_C", "Russia_W~exp~lic", "Canada~exp~lic", "Russia_E~exp"]
RECIPES = [("P1", 0.0), ("R1", 0.25), ("R1", 1.0), ("R2", 1.0), ("D0", 1.0), ("D1", 1.0)]
ML = ["R1_0.25", "R1_1.0", "R2_1.0", "D0_1.0", "D1_1.0"]


def load() -> pd.DataFrame:
    lg = pd.read_csv(LG).assign(src="LG")
    ld = pd.read_csv(LGD).assign(src="LGD")
    c = pd.concat([lg, ld], ignore_index=True)
    c = c[(c.placement == "cell") & (c.alpha.astype(str) == "1") & (c.learner.isin(["catboost_lo", "none"])) & (c.point_only != True)]  # noqa: E712
    c = c[(c.src == "LG") | (c.target.isin(KEEP_LGD))]
    c = c[~((c.src == "LGD") & (c.target.isin(["Lena", "Canada", "Russia_W", "Russia_E"])))]
    c = c[c.target != "Greenland"]
    return c.assign(lamf=c.lam.astype(float))


def deltas(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, m, src), x in c.groupby(["target", "mode", "src"]):
        for n in (0, 10, -1):
            y = x[x.n == n]
            if not len(y):
                continue
            d = dict(target=f"{t}|{m}", src=src, n=n, P0=float(y.rmse_p0.iloc[0]))
            for meth, lam in RECIPES:
                z = y[(y.method == meth) & np.isclose(y.lamf, lam)]
                if len(z):
                    d[f"{meth}_{lam}" if meth != "P1" else "P1"] = float(z.rmse.iloc[0] - d["P0"])
                    d[f"sig_{meth}_{lam}" if meth != "P1" else "sig_P1"] = str(z.sig_p0.iloc[0])
            rows.append(d)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = deltas(load())
    risk = []
    for n in (0, 10, -1):
        y = R[R.n == n]
        for col in ["P1"] + ML:
            v = y[col].dropna()
            risk.append(dict(n=n, method=col, n_targets=len(v), median=v.median(), p90=v.quantile(0.9), max=v.max(), min=v.min(),
                             n_worse_2cm=int((v > 2).sum())))
    risk = pd.DataFrame(risk)
    risk.to_csv(OUT / "wf0_risk.csv", index=False)
    y = R[R.n == -1].copy()
    y["recal_gain"] = -y["P1"]
    y["ml_best_delta"] = y[ML].min(axis=1)
    y["ml_best_recipe"] = y[ML].idxmin(axis=1)
    y["ml_vs_p1"] = y["ml_best_delta"] - y["P1"]
    y.sort_values("recal_gain").to_csv(OUT / "wf0_misspec.csv", index=False)
    s1 = spearmanr(y.recal_gain, -y.ml_best_delta)
    s2 = spearmanr(y.P0, -y.ml_best_delta)
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M"), plan="docs/EXPERIMENT_PLAN_WF_2026-10-01.md WF0", label="사후 서술(판정어 없음)",
                inputs=[str(LG.relative_to(ROOT)), str(LGD.relative_to(ROOT))], n_targets=int(len(y)),
                spearman_recal_gain_vs_ml_gain=dict(rho=float(s1.correlation), p=float(s1.pvalue)),
                spearman_p0_vs_ml_gain=dict(rho=float(s2.correlation), p=float(s2.pvalue)),
                note="최선 ML 은 다섯 레시피 가운데 사후 선택이라 낙관적이다")
    (OUT / "wf0_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[wf0] 대상 {len(y)} · 위험표 {len(risk)}행 · Spearman(재보정 이득, ML 이득) {s1.correlation:.2f}(p {s1.pvalue:.4f}) → {OUT}")


if __name__ == "__main__":
    main()
