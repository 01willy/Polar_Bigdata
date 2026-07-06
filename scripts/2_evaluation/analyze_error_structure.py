#!/usr/bin/env python3
"""모델 토너먼트 예측의 공간오차 구조 정량 분석.

입력:
  data/processed/model_tournament_predictions.csv  (fold,lat,lon,alt_cm,pred_*)
  data/processed/dl_dataset.csv                    (lat,lon,alt_cm,region,loc_id,공변량 dem_*/e5_*)

분석:
  1) 실측 ALT 구간별 RMSE·평균편향(pred-obs) — 고ALT 과소예측(평균회귀) 정량화
  2) region·fold별 RMSE + 총RMSE의 지역간(between)·지역내(within) 분해
  3) 오차(|pred-obs|) vs 공변량(dem_elev,e5_maat,관측밀도) 상관
  4) GBM vs Diffusion 오차 공간분포 차이

산출:
  outputs/figures/06_deep_learning/error_structure_summary.png
  outputs/maps/error_spatial_gbm_vs_diffusion.png
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = "/home/willy010313/Polar_Bigdata"
sys.path.insert(0, os.path.join(ROOT, "src"))
from polar.plotstyle import use_polar, CMAP, tnorm, add_cbar, style_geo  # noqa: E402

plt = use_polar()

PRED_CSV = os.path.join(ROOT, "data/processed/model_tournament_predictions.csv")
DL_CSV = os.path.join(ROOT, "data/processed/dl_dataset.csv")
FIGDIR = os.path.join(ROOT, "outputs/figures/06_deep_learning")
MAPDIR = os.path.join(ROOT, "outputs/maps")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(MAPDIR, exist_ok=True)

# 주력 비교 모델(존재하는 것만 사용)
MAIN_MODEL = "pred_앙상블(GBM+FT-T)"
MODELS_OF_INTEREST = [MAIN_MODEL, "pred_GBM", "pred_Diffusion"]


def rmse(err):
    return float(np.sqrt(np.mean(np.square(err))))


# ---------------------------------------------------------------------------
# 1. 로드 + 병합 (lat,lon 반올림 + alt_cm 키)
# ---------------------------------------------------------------------------
pred = pd.read_csv(PRED_CSV)
dl = pd.read_csv(DL_CSV)
ND = 4  # 반올림 소수자릿수

def add_keys(df):
    return df.assign(
        klat=df.lat.round(ND), klon=df.lon.round(ND), kalt=df.alt_cm.round(2)
    )

pred = add_keys(pred)
dlk = add_keys(dl)

# 키 내부에서 region/공변량은 단일값(검증 완료)이므로 키 단위 dedup 무손실
cov_cols = ["region", "loc_id", "dem_elev", "e5_maat"]
dl_key = dlk.drop_duplicates(subset=["klat", "klon", "kalt"])[
    ["klat", "klon", "kalt"] + cov_cols
]

df = pred.merge(dl_key, on=["klat", "klon", "kalt"], how="left")
matched = df["region"].notna().mean()
print(f"[merge] rows={len(df)}  region 매칭률={matched*100:.2f}%")
df = df[df["region"].notna()].copy()

# 관측밀도: loc_id 당 관측 개수(전체 dl_dataset 기준) — 위치 표본 밀도 대리변수
dens = dl.groupby("loc_id").size().rename("obs_density")
df = df.merge(dens, on="loc_id", how="left")

obs = df["alt_cm"].values
present = [m for m in MODELS_OF_INTEREST if m in df.columns]
print(f"[models] 분석 대상: {present}")

# 각 모델 오차 컬럼
for m in present:
    df[m + "__err"] = df[m].values - obs           # 편향 부호 포함
    df[m + "__abs"] = np.abs(df[m + "__err"])

# ---------------------------------------------------------------------------
# 2. ALT 구간별 RMSE·평균편향 (평균회귀 정량화)
# ---------------------------------------------------------------------------
bins = [-np.inf, 30, 45, 60, 80, np.inf]
labels = ["<30", "30-45", "45-60", "60-80", ">80"]
df["alt_bin"] = pd.cut(df["alt_cm"], bins=bins, labels=labels)

print("\n===== [1] 실측 ALT 구간별 오차·편향 (주력=앙상블) =====")
alt_rows = []
mm = MAIN_MODEL if MAIN_MODEL in df.columns else present[0]
for lab in labels:
    sub = df[df.alt_bin == lab]
    if len(sub) == 0:
        continue
    e = sub[mm + "__err"].values
    alt_rows.append(
        dict(alt_bin=lab, n=len(sub),
             mean_obs=float(sub.alt_cm.mean()),
             rmse=rmse(e), bias=float(np.mean(e)),
             mean_pred=float(sub[mm].mean()))
    )
alt_tbl = pd.DataFrame(alt_rows)
print(alt_tbl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# 평균회귀 기울기: obs 대비 pred 회귀 (기울기<1 이면 축소=평균회귀)
slope, intercept = np.polyfit(df.alt_cm.values, df[mm].values, 1)
corr = np.corrcoef(df.alt_cm.values, df[mm].values)[0, 1]
print(f"\n[평균회귀] pred = {slope:.3f}*obs + {intercept:.2f}  (r={corr:.3f})")
print(f"  기울기 {slope:.3f} < 1  =>  고ALT 과소예측·저ALT 과대예측(평균으로 수축)")

# ---------------------------------------------------------------------------
# 3. region·fold별 RMSE + between/within 분해
# ---------------------------------------------------------------------------
print("\n===== [2a] region별 RMSE (주력 모델) =====")
reg_rows = []
for reg, sub in df.groupby("region"):
    reg_rows.append(dict(region=reg, n=len(sub),
                         rmse=rmse(sub[mm + "__err"].values),
                         mean_obs=float(sub.alt_cm.mean())))
reg_tbl = pd.DataFrame(reg_rows).sort_values("n", ascending=False)
print(reg_tbl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\n===== [2b] fold별 RMSE (주력 모델) =====")
fold_rows = []
for fo, sub in df.groupby("fold"):
    fold_rows.append(dict(fold=int(fo), n=len(sub),
                          rmse=rmse(sub[mm + "__err"].values)))
fold_tbl = pd.DataFrame(fold_rows).sort_values("fold")
print(fold_tbl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# between/within 분해:
#   총 SSE = Σ(pred-obs)^2. 이를 region 관점에서 분해.
#   within: 각 region 내부에서 '지역평균 예측(= region별 pred 평균)' 대신
#           실제 예측이 남긴 편차 → 공변량이 지역내 변동을 설명하는 스킬.
#   여기서는 '오차'를 region평균오차(between)와 잔차(within)로 분해.
#   err = e_i, region 평균 ebar_r.
#   between_SSE = Σ n_r * ebar_r^2   (지역별 계통편향)
#   within_SSE  = Σ (e_i - ebar_r)^2 (지역내 산포)
print("\n===== [2c] 오차 분산 between/within 분해 =====")
e = df[mm + "__err"].values
df["_e"] = e
ebar_r = df.groupby("region")["_e"].transform("mean").values
between_sse = float(np.sum((ebar_r) ** 2))
within_sse = float(np.sum((e - ebar_r) ** 2))
total_sse = float(np.sum(e ** 2))
n = len(e)
print(f"총 RMSE       = {np.sqrt(total_sse/n):.3f} cm")
print(f"between RMSE  = {np.sqrt(between_sse/n):.3f} cm  (지역별 계통편향 성분)")
print(f"within RMSE   = {np.sqrt(within_sse/n):.3f} cm  (지역내 잔차 성분)")
print(f"between 비중  = {between_sse/total_sse*100:.1f}%   within 비중 = {within_sse/total_sse*100:.1f}%")

# 지역내 스킬: 각 region 내부에서 '지역평균 예측' 기준선 대비 개선
#   baseline_within = region 내 obs를 region평균pred로 상수예측했을 때의 오차분산
#   즉 obs의 지역내 분산 대비, 실제 예측이 지역내 변동을 얼마나 잡았는가.
print("\n===== [2d] 지역내 스킬 (공변량이 지역내 ALT 변동 설명하는가) =====")
within_skill_rows = []
for reg, sub in df.groupby("region"):
    if len(sub) < 30:
        continue
    o = sub.alt_cm.values
    pr = sub[mm].values
    # 지역내 기준선: 지역 obs 평균을 상수 예측 → 지역내 obs 분산
    base_rmse = rmse(o - o.mean())
    model_rmse = rmse(pr - o)
    skill = 1 - (model_rmse / base_rmse) if base_rmse > 0 else np.nan
    within_skill_rows.append(dict(region=reg, n=len(sub),
                                  within_obs_std=float(o.std()),
                                  base_rmse=base_rmse,
                                  model_rmse=model_rmse,
                                  within_skill=skill))
ws_tbl = pd.DataFrame(within_skill_rows).sort_values("n", ascending=False)
print(ws_tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("  within_skill = 1 - RMSE_model/RMSE_지역평균기준선.  >0 이면 공변량이 지역내 변동 설명.")

# ---------------------------------------------------------------------------
# 4. |오차| vs 공변량 상관
# ---------------------------------------------------------------------------
print("\n===== [3] |오차| vs 공변량 상관 (주력 모델) =====")
abscol = mm + "__abs"
cov_targets = {"dem_elev": "dem_elev", "e5_maat": "e5_maat",
               "obs_density": "obs_density", "alt_cm(실측)": "alt_cm"}
cor_rows = []
for name, col in cov_targets.items():
    x = df[col].values
    y = df[abscol].values
    ok = np.isfinite(x) & np.isfinite(y)
    r_p = np.corrcoef(x[ok], y[ok])[0, 1]
    # Spearman
    from scipy.stats import spearmanr
    r_s = spearmanr(x[ok], y[ok]).correlation
    cor_rows.append(dict(covariate=name, pearson_r=r_p, spearman_r=r_s))
cor_tbl = pd.DataFrame(cor_rows)
print(cor_tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# 공변량 구간별 평균 |오차| (어디서 큰가)
print("\n[dem_elev 구간별 평균 |오차|]")
df["elev_bin"] = pd.cut(df.dem_elev, [-1, 50, 200, 500, 1000, 1e4],
                        labels=["0-50", "50-200", "200-500", "500-1000", ">1000"])
print(df.groupby("elev_bin", observed=True)[abscol].agg(["mean", "count"]).round(2).to_string())
print("\n[e5_maat 구간별 평균 |오차|]  (더 따뜻할수록)")
df["maat_bin"] = pd.cut(df.e5_maat, [-15, -8, -6, -4, -2, 5],
                        labels=["<-8", "-8~-6", "-6~-4", "-4~-2", ">-2"])
print(df.groupby("maat_bin", observed=True)[abscol].agg(["mean", "count"]).round(2).to_string())

# ---------------------------------------------------------------------------
# 5. GBM vs Diffusion 오차 공간분포 차이
# ---------------------------------------------------------------------------
gbm_c, dif_c = "pred_GBM", "pred_Diffusion"
have_pair = gbm_c in df.columns and dif_c in df.columns
if have_pair:
    print("\n===== [4] GBM vs Diffusion 오차 공간분포 =====")
    print(f"GBM       전체RMSE={rmse(df[gbm_c+'__err'].values):.2f}  MAE={df[gbm_c+'__abs'].mean():.2f}  bias={df[gbm_c+'__err'].mean():+.2f}")
    print(f"Diffusion 전체RMSE={rmse(df[dif_c+'__err'].values):.2f}  MAE={df[dif_c+'__abs'].mean():.2f}  bias={df[dif_c+'__err'].mean():+.2f}")
    # region별 어느 모델이 우세?
    print("\n[region별 RMSE: GBM vs Diffusion]")
    rrows = []
    for reg, sub in df.groupby("region"):
        rrows.append(dict(region=reg, n=len(sub),
                          gbm=rmse(sub[gbm_c+'__err'].values),
                          diff=rmse(sub[dif_c+'__err'].values)))
    rr = pd.DataFrame(rrows)
    rr["diff_minus_gbm"] = rr["diff"] - rr["gbm"]
    print(rr.sort_values("n", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    # 공간 격자화(0.5도) 후 |오차| 평균 차이
    df["glat"] = (df.lat / 0.5).round() * 0.5
    df["glon"] = (df.lon / 0.5).round() * 0.5
    grid = df.groupby(["glat", "glon"]).agg(
        gbm_abs=(gbm_c + "__abs", "mean"),
        dif_abs=(dif_c + "__abs", "mean"),
        n=("alt_cm", "size"),
    ).reset_index()
    grid = grid[grid.n >= 5]
    grid["dif_minus_gbm"] = grid.dif_abs - grid.gbm_abs
    print(f"\n[격자(0.5°) 단위 |오차| 차이]  n격자={len(grid)}")
    print(f"  Diffusion가 더 정확한 격자 비율: {(grid.dif_minus_gbm<0).mean()*100:.1f}%")
    print(f"  격자별 |오차| 차이 상관(GBM vs Diffusion 오차 공간패턴): "
          f"{np.corrcoef(grid.gbm_abs, grid.dif_abs)[0,1]:.3f}")

# ---------------------------------------------------------------------------
# 그림 1: 오차 구조 요약(4패널)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) ALT 구간별 RMSE + 편향
ax = axes[0, 0]
x = np.arange(len(alt_tbl))
ax.bar(x - 0.2, alt_tbl.rmse, width=0.4, color="#5a4fcf", label="RMSE")
ax.bar(x + 0.2, alt_tbl.bias, width=0.4, color="#cf5a4f", label="평균편향(pred-obs)")
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(alt_tbl.alt_bin)
ax.set_xlabel("실측 ALT 구간 (cm)")
ax.set_ylabel("cm")
ax.set_title("(a) ALT 구간별 RMSE·편향 — 고ALT 과소예측")
for xi, (b, nn) in enumerate(zip(alt_tbl.bias, alt_tbl.n)):
    ax.annotate(f"n={nn}", (xi, max(alt_tbl.rmse) * 0.92), ha="center", fontsize=8, color="#555")
ax.legend()

# (b) obs vs pred 산점(평균회귀)
ax = axes[0, 1]
samp = df.sample(min(8000, len(df)), random_state=0)
ax.scatter(samp.alt_cm, samp[mm], s=4, alpha=0.15, color="#3a7ca5")
lim = [0, min(200, df.alt_cm.quantile(0.999))]
ax.plot(lim, lim, "--", color="#444", lw=1, label="1:1 (완벽)")
xs = np.linspace(lim[0], lim[1], 50)
ax.plot(xs, slope * xs + intercept, "-", color="#cf5a4f", lw=1.8,
        label=f"회귀 기울기={slope:.2f}")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("실측 ALT (cm)")
ax.set_ylabel("예측 ALT (cm)")
ax.set_title("(b) 평균회귀: 기울기<1 → 극단값 수축")
ax.legend()

# (c) region별 within/between + 스킬
ax = axes[1, 0]
rlabels = ws_tbl.region.tolist()
xr = np.arange(len(rlabels))
ax.bar(xr, ws_tbl.model_rmse, width=0.55, color="#5a4fcf", label="모델 RMSE(지역내)")
ax.bar(xr, ws_tbl.base_rmse, width=0.55, color="none", edgecolor="#cf5a4f",
       lw=1.5, label="지역평균 기준선 RMSE")
ax.set_xticks(xr)
ax.set_xticklabels([r[:12] for r in rlabels], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("RMSE (cm)")
ax.set_title(f"(c) 지역내 스킬 — between {between_sse/total_sse*100:.0f}% / within {within_sse/total_sse*100:.0f}%")
for xi, sk in enumerate(ws_tbl.within_skill):
    ax.annotate(f"skill={sk:.2f}", (xi, ws_tbl.model_rmse.iloc[xi] + 1),
                ha="center", fontsize=7.5, color="#333")
ax.legend(fontsize=8)

# (d) |오차| vs 공변량 상관 막대 + 밀도구간
ax = axes[1, 1]
xc = np.arange(len(cor_tbl))
ax.bar(xc - 0.2, cor_tbl.pearson_r, width=0.4, color="#3a7ca5", label="Pearson")
ax.bar(xc + 0.2, cor_tbl.spearman_r, width=0.4, color="#7ba05b", label="Spearman")
ax.axhline(0, color="#444", lw=0.8)
ax.set_xticks(xc)
ax.set_xticklabels(cor_tbl.covariate, rotation=15, ha="right", fontsize=8.5)
ax.set_ylabel("상관계수 (vs |오차|)")
ax.set_title("(d) |오차|–공변량 상관")
ax.legend(fontsize=8)

fig.suptitle("모델 토너먼트 예측 오차 구조 (주력=앙상블 GBM+FT-T)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
f1 = os.path.join(FIGDIR, "error_structure_summary.png")
fig.savefig(f1)
plt.close(fig)
print(f"\n[fig] 저장: {f1}")

# ---------------------------------------------------------------------------
# 그림 2: GBM vs Diffusion 오차 공간지도
# ---------------------------------------------------------------------------
f2 = None
if have_pair:
    # 주 데이터 밀집영역(알래스카)로 지도 초점 — loc 단위 평균 |오차|
    loc = df.groupby("loc_id").agg(
        lat=("lat", "mean"), lon=("lon", "mean"),
        gbm_abs=(gbm_c + "__abs", "mean"),
        dif_abs=(dif_c + "__abs", "mean"),
        n=("alt_cm", "size"),
    ).reset_index()
    loc["dif_minus_gbm"] = loc.dif_abs - loc.gbm_abs
    # 알래스카 주 영역
    m = (loc.lon > -170) & (loc.lon < -140) & (loc.lat > 55) & (loc.lat < 72)
    L = loc[m]
    vmax = np.nanpercentile(np.r_[L.gbm_abs, L.dif_abs], 97)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, col, tit in [
        (axes2[0], "gbm_abs", "(a) GBM 평균 |오차|"),
        (axes2[1], "dif_abs", "(b) Diffusion 평균 |오차|"),
    ]:
        sc = ax.scatter(L.lon, L.lat, c=L[col], s=14, cmap=CMAP.err,
                        vmin=0, vmax=vmax, edgecolors="none")
        style_geo(ax, title=tit)
        add_cbar(fig2, sc, ax, "평균 |오차| (cm)")
    # (c) 차이맵 발산
    dmax = np.nanpercentile(np.abs(L.dif_minus_gbm), 96)
    sc = axes2[2].scatter(L.lon, L.lat, c=L.dif_minus_gbm, s=14, cmap=CMAP.diff,
                          norm=tnorm(-dmax, dmax, 0.0), edgecolors="none")
    style_geo(axes2[2], title="(c) Diffusion - GBM  (음=Diffusion 우세)")
    add_cbar(fig2, sc, axes2[2], "|오차| 차이 (cm)")
    fig2.suptitle("GBM vs Diffusion 오차 공간분포 (알래스카, loc 평균 |오차|)",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    f2 = os.path.join(MAPDIR, "error_spatial_gbm_vs_diffusion.png")
    fig2.savefig(f2)
    plt.close(fig2)
    print(f"[fig] 저장: {f2}")

print("\n=== 완료 ===")
