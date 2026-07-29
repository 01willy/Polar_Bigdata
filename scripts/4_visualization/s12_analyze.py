"""S12 결과 집계·표·그림. 샤드 CSV를 합쳐 조건별 최고 구성과 Stefan 대비 개선을 정리한다."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "processed"
FIGD = ROOT / "outputs" / "figures" / "s12_hybrid"
FIGD.mkdir(parents=True, exist_ok=True)

shards = sorted(OUT.glob("s12_hybrid_transfer_shard*.csv"))
if not shards:
    sys.exit("샤드 CSV 없음")
df = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
df = df[np.isfinite(df.rmse_cm)]
print(f"합계 {len(df)}행 · 샤드 {len(shards)}개")

# seed 평균
key = ["proto", "target", "family", "anchor", "pseudo", "resid", "r", "lam"]
agg = (df.groupby(key, dropna=False)
         .agg(rmse=("rmse_cm", "mean"), bias=("bias_cm", "mean"),
              sd=("rmse_cm", "std"), n=("n", "first"), nseed=("seed", "nunique"))
         .reset_index())

# 지역 비가중평균(프로토콜별)
gate = (agg.groupby(["proto", "family", "anchor", "pseudo", "resid", "r", "lam"], dropna=False)
           .agg(rmse_mean=("rmse", "mean"), n_reg=("target", "size"))
           .reset_index())
gate = gate[gate.n_reg == 2]

def label(row):
    """외부 독자용 구성 이름."""
    anc = {"stefan": "Stefan", "cci": "위성제품", "cci_cal": "위성제품(보정)",
           "stefan_cci": "Stefan+위성제품", "stefan_cci_w": "Stefan+위성제품(가중학습)",
           "none": ""}[row.anchor]
    md = {"ridge": "선형", "catboost": "부스팅", "mlp": "다층퍼셉트론",
          "ftt": "트랜스포머", "none": ""}[row.resid]
    ps = {"none": "", "stefan": "Stefan 증강", "cci": "위성제품 증강",
          "stefan_cci": "Stefan+위성 증강", "const": "상수 증강"}[row.pseudo]
    if row.family == "analytic":
        return anc
    if row.family == "direct":
        return f"{md}" + (f" · {ps}(r={row.r:g})" if ps else " · 증강없음")
    s = f"{anc} 앵커 + {md} 잔차(λ={row.lam:g})"
    return s + (f" · {ps}(r={row.r:g})" if ps else "")

for tbl in (agg, gate):
    tbl["name"] = tbl.apply(label, axis=1)

print("\n" + "=" * 100)
for proto, pn in [("half", "설정 B · 대상 지역 공변량 사용 가능"),
                  ("loro", "설정 C · 대상 지역 정보 없음")]:
    g = gate[gate.proto == proto].sort_values("rmse_mean")
    ref = g[(g.family == "analytic") & (g.anchor == "stefan")].rmse_mean
    ref = float(ref.iloc[0]) if len(ref) else np.nan
    print(f"\n### {pn}   (Stefan 단독 = {ref:.2f} cm)")
    print(f"{'순위':<4}{'구성':<52}{'평균':>8}{'Stefan대비':>11}")
    for i, (_, r) in enumerate(g.head(15).iterrows(), 1):
        print(f"{i:<4}{r['name'][:50]:<52}{r.rmse_mean:>8.2f}{ref - r.rmse_mean:>+11.2f}")
    # 지역별 상세(상위 8)
    print(f"\n  [지역별] {'구성':<46}{'레나':>8}{'캐나다':>9}")
    top = g.head(8)
    for _, r in top.iterrows():
        sub = agg[(agg.proto == proto) & (agg.family == r.family) & (agg.anchor == r.anchor)
                  & (agg.pseudo == r.pseudo) & (agg.resid == r.resid)
                  & (agg.r == r.r) & (agg.lam == r.lam)]
        le = sub[sub.target == "Lena"].rmse
        ca = sub[sub.target == "Canada"].rmse
        print(f"  {r['name'][:44]:<48}{(le.iloc[0] if len(le) else np.nan):>8.2f}"
              f"{(ca.iloc[0] if len(ca) else np.nan):>9.2f}")

# 저장
agg.to_csv(OUT / "s12_hybrid_by_region.csv", index=False)
gate.to_csv(OUT / "s12_hybrid_gate.csv", index=False)
print(f"\nsaved: s12_hybrid_by_region.csv · s12_hybrid_gate.csv")

# ---------------------------------------------------------------- 그림
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from polar.plotstyle import use_polar
    use_polar()
except Exception:
    pass

C = {"analytic": "#8296a8", "direct": "#b8c4d0", "anchored": "#2f4b6e"}
fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))
for ax, (proto, pn) in zip(axes, [("half", "대상 지역 공변량 사용 가능"),
                                  ("loro", "대상 지역 정보 없음")]):
    g = gate[gate.proto == proto].sort_values("rmse_mean").head(12).iloc[::-1]
    ref = gate[(gate.proto == proto) & (gate.family == "analytic")
               & (gate.anchor == "stefan")].rmse_mean
    ref = float(ref.iloc[0]) if len(ref) else np.nan
    y = np.arange(len(g))
    ax.barh(y, g.rmse_mean, color=[C[f] for f in g.family], height=0.7,
            edgecolor="none", zorder=3)
    ax.axvline(ref, ls=(0, (5, 3)), lw=1.2, color="#3a4a5a", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([n[:44] for n in g.name], fontsize=8.5)
    for yi, v in zip(y, g.rmse_mean):
        ax.text(v + 0.25, yi, f"{v:.2f}", va="center", fontsize=8.5, color="#444")
    ax.set_xlabel("두 지역 평균 RMSE (cm)", fontsize=10.5)
    ax.set_title(pn, fontsize=11, pad=8)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(0, max(g.rmse_mean) * 1.18)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc=C["analytic"], label="물리식·위성제품(학습 없음)"),
                    Patch(fc=C["direct"], label="기계학습 단독"),
                    Patch(fc=C["anchored"], label="앵커 + 잔차 결합")],
           loc="lower center", ncol=3, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.05, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(FIGD / f"s12_transfer_ranking.{ext}", dpi=300, bbox_inches="tight")
print(f"saved: {FIGD}/s12_transfer_ranking.png")
