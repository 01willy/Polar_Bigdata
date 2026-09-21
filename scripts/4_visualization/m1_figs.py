"""M1 주효과 그림 — 분석 산출(m1_analysis.py: <out>_summary.csv, <out>_tests.csv, <out>_uq.csv)에서 생성.

패널
  (a) 앵커 축: 각 앵커 − Stefan(λ=0) ΔRMSE, 조건(공변량만·정보 없음), 4지역(AB4) + 지역 평균, 블록 CI
  (b) 유사라벨 대조군 사다리: Stefan 유사라벨 − {없음, 상수, 대상 상수, 셔플, TDD 선형}, 공변량만, 4지역
  (c) 모델 축: 정보 없음에서 직접 회귀 대 Stefan 잔차(λ=.25) 대 앵커 단독, 4지역 평균
  (d) 생성 모델 UQ: 90% 구간 커버리지 대 interval score (지역 내·전이), CQR 참조
실행: python3 scripts/4_visualization/m1_figs.py --out m1_sc
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.plotstyle import use_polar  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="m1_sc")
args = ap.parse_args()
M1 = ROOT / "data" / "processed" / "m1"
FIG = ROOT / "outputs" / "figures" / "m1"; FIG.mkdir(parents=True, exist_ok=True)
use_polar()
AB4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
KR = {"Lena": "레나델타", "Canada": "캐나다", "Russia_W": "러시아 서부", "Russia_E": "러시아 동부",
      "REGION_SUMMARY_AB4": "4지역 평균", "REGION_SUMMARY_MAIN": "주 6지역 평균"}
BLUE, TEAL, GREY, NAVY, SAND = "#3f6fa8", "#4c9a8f", "#7a7a7a", "#17365d", "#b58a3c"

tests = pd.read_csv(M1 / f"{args.out}_tests.csv")
summ = pd.read_csv(M1 / f"{args.out}_summary.csv")
dcol = "delta_per_seed" if "delta_per_seed" in tests.columns else "delta_rmse"


def rows(H, cond, label_sub=None):
    t = tests[(tests.H == H) & (tests.cond == cond)]
    if label_sub:
        t = t[t.label.str.contains(label_sub, regex=False)]
    return t


fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))

# ---------------------------------------------------------------- (a) 앵커 축
ax = axes[0, 0]
anc_order = ["stefan_k2", "stefan_soil", "cci_raw", "cci_cal", "stefan_cci", "stefan_cci_cal", "stefan_soil_cci",
             "ku_cal", "stefan_ku", "stefan_ku_cci", "emap_ridge", "emap_cb"]
anc_kr = {"stefan_k2": "Stefan k=2", "stefan_soil": "토양 도일 Stefan", "cci_raw": "CCI 원값", "cci_cal": "CCI 보정",
          "stefan_cci": "Stefan+CCI", "stefan_cci_cal": "Stefan+CCI(보정)", "stefan_soil_cci": "토양 Stefan+CCI",
          "ku_cal": "Kudryavtsev 보정", "stefan_ku": "Stefan+Ku", "stefan_ku_cci": "Stefan+Ku+CCI",
          "emap_ridge": "계수 지도(ridge)", "emap_cb": "계수 지도(CatBoost)"}
y = np.arange(len(anc_order))
for j, (cond, col, off) in enumerate([("covonly", BLUE, -0.18), ("noinfo", NAVY, 0.18)]):
    t = pd.concat([rows("X-anchor", cond), rows("H1", cond), rows("H10", cond)])
    for i, a in enumerate(anc_order):
        if a == "stefan_cci":
            r = t[(t.H == "H1") & (t.target == "REGION_SUMMARY_AB4")]
        else:
            r = t[(t.label.str.startswith(f"{cond}: {a} − stefan") | t.label.str.startswith(f"{cond}: {a} − Stefan")) & (t.target == "REGION_SUMMARY_AB4")]
        if not len(r):
            r = t[(t.label.str.startswith(f"{cond}: {a} − stefan")) & (t.target == "REGION_SUMMARY_MAIN")]
        if not len(r):
            continue
        d, lo, hi = float(r[dcol].iloc[0]), float(r.ci_lo.iloc[0]), float(r.ci_hi.iloc[0])
        ax.errorbar(d, y[i] + off, xerr=[[d - lo] if np.isfinite(lo) else [0], [hi - d] if np.isfinite(hi) else [0]],
                    fmt="o", color=col, ms=4.5, capsize=2, lw=1.1, label=("공변량만" if cond == "covonly" else "정보 없음") if i == 0 else None)
ax.axvline(0, color=GREY, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([anc_kr[a] for a in anc_order]); ax.invert_yaxis()
ax.set_xlabel("ΔRMSE 대 Stefan 앵커 (cm), 4지역 비가중 평균, 층화 블록 부트스트랩 95% CI")
ax.legend(loc="upper right", frameon=False); ax.set_title("(a) 앵커 축", loc="left")

# ---------------------------------------------------------------- (b) 유사라벨 사다리
ax = axes[0, 1]
ctrls = [("none", "증강 없음"), ("const", "상수(알래스카 평균)"), ("const_t", "대상 수준 상수"), ("shuffle", "행 셔플"), ("tddlin", "TDD 선형")]
x = np.arange(len(AB4))
w = 0.16
for k, (c, name) in enumerate(ctrls):
    t = rows("H3", "covonly", f"− {c} (")
    vals, los, his = [], [], []
    for tg in AB4:
        r = t[t.target == tg]
        vals.append(float(r[dcol].iloc[0]) if len(r) else np.nan)
        los.append(float(r.ci_lo.iloc[0]) if len(r) else np.nan); his.append(float(r.ci_hi.iloc[0]) if len(r) else np.nan)
    vals, los, his = map(np.array, (vals, los, his))
    col = [NAVY, GREY, SAND, TEAL, BLUE][k]
    ax.bar(x + (k - 2) * w, vals, w, color=col, label=name)
    ax.errorbar(x + (k - 2) * w, vals, yerr=[np.nan_to_num(vals - los), np.nan_to_num(his - vals)], fmt="none", ecolor="black", lw=0.8, capsize=1.5)
ax.axhline(0, color=GREY, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([KR[t] for t in AB4])
ax.set_ylabel("ΔRMSE (cm): Stefan 유사라벨 − 대조군")
ax.legend(frameon=False, fontsize=8, ncol=2); ax.set_title("(b) 대조군 사다리 (공변량만, 직접 CatBoost, r=10, 분할 3회)", loc="left")

# ---------------------------------------------------------------- (c) 모델 축
ax = axes[1, 0]
models = ["ridge", "catboost_lo", "catboost", "mlp", "ftt", "tabm", "realmlp", "cfm", "ddpm", "nflow"]
m_kr = {"ridge": "능형", "catboost_lo": "CatBoost(저용량)", "catboost": "CatBoost", "mlp": "MLP", "ftt": "FT-Transformer", "tabm": "TabM",
        "realmlp": "RealMLP", "cfm": "플로 매칭", "ddpm": "확산", "nflow": "정규화 플로"}
s = summ[(summ.cond == "noinfo") & (summ.target.isin(AB4)) & (summ.dset == "alaska") & (summ.xset == "x25") & (summ.pseudo == "none") & (summ.iw == 0)]
ref = s[(s.anchor == "stefan") & (s.lam == 0.0)].groupby("target").rmse.first()
direct = s[(s.anchor == "none")].groupby("resid").rmse.mean()
resid = s[(s.anchor == "stefan") & (s.lam == 0.25)].groupby("resid").rmse.mean()
xx = np.arange(len(models))
ax.bar(xx - 0.2, [direct.get(m, np.nan) for m in models], 0.38, color=GREY, label="직접 회귀")
ax.bar(xx + 0.2, [resid.get(m, np.nan) for m in models], 0.38, color=BLUE, label="Stefan 앵커 + 잔차(λ=0.25)")
ax.axhline(float(ref.mean()), color=NAVY, lw=1.2, ls="--", label=f"Stefan 앵커 단독 {ref.mean():.1f}")
ax.set_xticks(xx); ax.set_xticklabels([m_kr[m] for m in models], rotation=30, ha="right")
ax.set_ylabel("RMSE (cm), 정보 없음, 4지역 비가중 평균")
ax.set_ylim(0, max(45, float(np.nanmax(direct.values)) * 1.05))
ax.legend(frameon=False, fontsize=8); ax.set_title("(c) 모델 축", loc="left")

# ---------------------------------------------------------------- (d) UQ
ax = axes[1, 1]
uqf = M1 / f"{args.out}_uq.csv"
if uqf.exists():
    u = pd.read_csv(uqf)
    u = u[(u.get("pseudo", "none") == "none") if "pseudo" in u.columns else np.ones(len(u), bool)]
    for m, col, mk in [("cfm", BLUE, "o"), ("ddpm", TEAL, "s"), ("nflow", SAND, "^")]:
        for cond, alpha in [("labels", 1.0), ("noinfo", 0.45)]:
            q = u[(u.resid == m) & (u.cond == cond) & (u.lam == 1.0) & (u.anchor.isin(["none", "stefan"]))]
            if cond == "labels":
                q = q[q.target == "Alaska"]
            else:
                q = q[q.target.isin(AB4)]
            if len(q):
                g = q.groupby("anchor")[["interval_score", "coverage"]].mean()      # 지역·seed 평균(직접 / Stefan 잔차 λ=1)
                ax.scatter(g.interval_score, g.coverage, color=col, marker=mk, alpha=alpha, s=60, edgecolor="black", lw=0.5,
                           label=f"{m_kr[m]} ({'알래스카 지역 내' if cond == 'labels' else '전이 4지역'})")
                for a, rr in g.iterrows():
                    ax.annotate("직접" if a == "none" else "잔차", (rr.interval_score, rr.coverage), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0.9, color=GREY, lw=0.8, ls=":")
    ax.axvline(68.9, color=NAVY, lw=1.0, ls="--"); ax.text(72, 0.97, "CQR 68.9 (알래스카 지역 내, 커버리지 0.93)", fontsize=8, color=NAVY)
    ax.set_xlabel("interval score (cm, α=0.1, 낮을수록 좋음)"); ax.set_ylabel("90% 구간 경험 커버리지")
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
else:
    ax.text(0.5, 0.5, "uq.csv 없음", ha="center", va="center", transform=ax.transAxes)
ax.set_title("(d) 생성 모델 구간 (H15)", loc="left")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"{args.out}_main_effects.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
print("saved:", FIG / f"{args.out}_main_effects.png")
spec = ROOT / "figures" / "figure_spec.json"
try:
    js = json.loads(spec.read_text()) if spec.exists() else {}
    js[f"m1/{args.out}_main_effects"] = dict(script="scripts/4_visualization/m1_figs.py", inputs=[f"data/processed/m1/{args.out}_tests.csv", f"data/processed/m1/{args.out}_summary.csv", f"data/processed/m1/{args.out}_uq.csv"],
                                            message="M1 주효과: 앵커·유사라벨 대조·모델·생성 모델 UQ", exports=["png", "pdf"], units="cm", generated="2026-09-21")
    spec.write_text(json.dumps(js, ensure_ascii=False, indent=1))
except Exception as e:  # noqa: BLE001
    print("figure_spec 갱신 실패:", e)
