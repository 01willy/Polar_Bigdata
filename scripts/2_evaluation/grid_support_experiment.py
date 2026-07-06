"""B(격자 지지 재프레이밍) 핵심 검증: 예측·검증 support(지지 크기)를 키우면 정확도가 오르는가?
점 대표성 잡음(사이트내 SD 9~23cm)이 support 평균으로 상쇄되는지 정량화.
- 관측을 support 격자(점→1→2.5→5→10→25km)로 집계(셀평균 ALT + 셀평균 공변량).
- 동일 계열 공간블록 6-fold CV로 GBM이 '셀평균 ALT'를 얼마나 맞히나 → RMSE vs support 곡선.
- support가 커질수록 RMSE 급감하면: (i)floor=대표성 확증, (ii)격자 지지에선 훨씬 정확 → 3D/고해상 산출의 올바른 지지.
산출: data/processed/grid_support_results.csv, figures/06_deep_learning/grid_support_curve.png
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, FROZEN, THAWED
plt = use_polar()

FEAT = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
CLIP = (np.log1p(1), np.log1p(600))
obs = pd.read_csv("data/processed/dl_dataset.csv").dropna(subset=FEAT + ["alt_cm"]).reset_index(drop=True)
print(f"점 관측 {len(obs):,}")


def cv_rmse(df, latc, lonc):
    """df: 집계 단위(셀). 셀평균 ALT를 셀평균 공변량으로 예측, 공간블록 6-fold."""
    blk = (np.floor(df[latc] / 0.5).astype(int).astype(str) + "_" + np.floor(df[lonc] / 0.5).astype(int).astype(str))
    if blk.nunique() < 6:
        return np.nan
    X = df[FEAT].values.astype(np.float32); y = np.log1p(df["alt_cm"].values)
    res = []
    for tr, te in GroupKFold(6).split(X, groups=blk):
        g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X[tr], y[tr])
        p = np.expm1(np.clip(g.predict(X[te]), *CLIP))
        res.append(p - df["alt_cm"].values[te])
    return float(np.sqrt(np.mean(np.concatenate(res) ** 2)))


rows = []
for s_km in [0, 1, 2.5, 5, 10, 25]:
    if s_km == 0:
        rmse = cv_rmse(obs, "lat", "lon"); n = len(obs); within = np.nan
    else:
        r = s_km / 111.0
        obs["_cy"] = np.round(obs.lat / r).astype(int); obs["_cx"] = np.round(obs.lon / r).astype(int)
        agg = obs.groupby(["_cy", "_cx"]).agg(
            **{f: (f, "mean") for f in FEAT}, alt_cm=("alt_cm", "mean"),
            lat=("lat", "mean"), lon=("lon", "mean"), n=("alt_cm", "size"),
            alt_std=("alt_cm", "std")).reset_index()
        agg = agg[agg.n >= 1]
        rmse = cv_rmse(agg, "lat", "lon"); n = len(agg)
        within = float(np.nanmean(agg.alt_std[agg.n >= 3]))  # 셀 내부 ALT 표준편차(대표성 잡음)
    rows.append(dict(support_km=s_km if s_km else "점(point)", n_units=n, rmse_cm=round(rmse, 2),
                     within_cell_sd=round(within, 1) if within == within else None))
    print(f"support {s_km}km: n={n:,}  RMSE={rmse:.2f}cm  셀내SD={within if within==within else 'NA'}")

out = pd.DataFrame(rows)
out.to_csv("data/processed/grid_support_results.csv", index=False)
print("\n=== 격자 지지별 정확도 ===")
print(out.to_string(index=False))

# ---- 시각화: support vs RMSE + 셀내 대표성 잡음 ----
xs = [0.4 if r["support_km"] == "점(point)" else r["support_km"] for r in rows]
rm = [r["rmse_cm"] for r in rows]
sd = [r["within_cell_sd"] for r in rows]
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.plot(xs, rm, "-o", color=FROZEN, lw=2, ms=8, label="공간블록 CV RMSE (셀평균 ALT)")
ax.plot([x for x, s in zip(xs, sd) if s], [s for s in sd if s], "--s", color=THAWED, lw=1.5, ms=6,
        label="셀 내부 ALT 표준편차(대표성 잡음)")
for x, v in zip(xs, rm):
    ax.text(x, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=FROZEN)
ax.set_xscale("symlog", linthresh=0.5)
ax.set_xticks([0.4, 1, 2.5, 5, 10, 25]); ax.set_xticklabels(["점", "1", "2.5", "5", "10", "25"])
ax.set_xlabel("예측·검증 지지(support) 크기 (km)")
ax.set_ylabel("RMSE / 셀내 SD (cm)")
ax.set_title("격자 지지를 키워도 RMSE는 안 줄어든다 — 표본 급감·외삽 심화가 지배\n"
             "(단 셀내 SD≈11cm는 점 대표성 floor 확증: 점 오차의 상당분은 sub-km 미세변동)", weight="bold")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
sys.path.insert(0, "src")
from polar.outputs import figpath
fig.savefig(figpath("deep_learning", "grid_support_curve")); plt.close(fig)
print("saved", figpath("deep_learning", "grid_support_curve"))
