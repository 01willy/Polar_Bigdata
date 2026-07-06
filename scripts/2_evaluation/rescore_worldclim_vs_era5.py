"""Stage 2 재채점: 동일 CALM 사이트에서 공변량만 바꿔 ALT 예측 성능 비교(통제 실험).
WorldClim(18km 평년값) vs ERA5-Land(9km 실측 월기온 도일/적설/토양온도).
헤드라인 = 공간블록 + LORO(지역전이) RMSE. → "더 좋은 데이터"의 효과 정량화.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from polar.plotstyle import use_korean
plt = use_korean()
from polar.alt_model import (load_table, add_engineered, folds_block, folds_loro,
                             variogram_range_km, _eval_featureset, FEATS, ENG)

os.makedirs("outputs/figures", exist_ok=True)

# ---------- 데이터 준비: WorldClim 피처 + 사인도일 + ERA5-Land 공변량 병합 ----------
df = add_engineered(load_table())
e5 = pd.read_csv("data/processed/alt_era5_covariates.csv")
E5F = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]

for d in (df, e5):
    d["_k"] = d["lat"].round(4).astype(str) + "_" + d["lon"].round(4).astype(str)
df = df.merge(e5[["_k"] + E5F], on="_k", how="left")
n_before = len(df)
df = df.dropna(subset=["e5_maat"]).reset_index(drop=True)     # 동일 사이트 집합 보장
print(f"통제 실험 대상: {len(df)} site-year (ERA5 결측 {n_before-len(df)} 제외), "
      f"{df.site.nunique()} site, {df.country.nunique()} 국가")

block_km, _ = variogram_range_km(df)
fb, nblk = folds_block(df, block_km)
fl = folds_loro(df)
print(f"공간블록 {block_km:.0f}km ({nblk}블록), LORO {len(fl)}개 지역")

# ---------- 피처셋 정의 (동일 사이트·동일 CV, 피처만 변경) ----------
configs = [
    ("WorldClim 5종 (기존 baseline)", FEATS),
    ("WC + 사인곡선 도일 (Stage1)", FEATS + ENG),
    ("ERA5-Land 실측 (+고도)", ["wc_elev"] + E5F),
    ("WC + ERA5-Land + 도일 (전체)", FEATS + ENG + E5F),
]
rows = []
for cvname, folds in [("공간블록", fb), ("LORO(전이)", fl)]:
    for label, feats in configs:
        r, m = _eval_featureset(df, folds, feats, "gbm")
        rows.append(dict(cv=cvname, config=label, rmse=round(r, 1), mae=round(m, 1)))
        print(f"  [{cvname:8s}] {label:32s} RMSE={r:5.1f}  MAE={m:5.1f} cm")
res = pd.DataFrame(rows)
res.to_csv("data/processed/stage2_era5_rescore.csv", index=False)

# ---------- 시각화: 공변량별 RMSE (공간블록 vs LORO) ----------
labels = [c[0] for c in configs]
short = ["WorldClim\n(기존)", "WC+사인도일\n(Stage1)", "ERA5-Land\n(실측)", "WC+ERA5\n(전체)"]
blk = [res[(res.cv == "공간블록") & (res.config == l)].rmse.values[0] for l in labels]
lor = [res[(res.cv == "LORO(전이)") & (res.config == l)].rmse.values[0] for l in labels]

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(labels)); w = 0.38
b1 = ax.bar(x - w/2, blk, w, label="공간블록 CV", color="#4c72b0")
b2 = ax.bar(x + w/2, lor, w, label="LORO 전이 CV (헤드라인)", color="#c44e52")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.1f}",
                ha="center", fontsize=9, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=10)
ax.set_ylabel("ALT RMSE (cm) — 낮을수록 좋음")
base_lor = lor[0]; best_lor = min(lor)
ax.set_title(f"공변량 업그레이드 효과 — 동일 {df.site.nunique()} 사이트 통제 실험\n"
             f"LORO(전이) {base_lor:.1f} → {best_lor:.1f} cm "
             f"({(base_lor-best_lor)/base_lor*100:+.0f}%)", fontsize=12, weight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.25, axis="y")
ax.axhline(base_lor, color="#c44e52", ls=":", lw=1, alpha=0.6)
fig.tight_layout(); fig.savefig("outputs/figures/18_era5_rescore.png", dpi=130)
plt.close(fig)
print("saved outputs/figures/18_era5_rescore.png")
print(f"\n핵심: LORO(전이) {base_lor:.1f} → {best_lor:.1f} cm — ERA5-Land 실측 공변량 효과")
