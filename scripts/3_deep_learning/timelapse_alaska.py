"""트랙 2 — 알래스카 연도별(2010-2024) ALT timelapse 산출과 정직 평가.

목표: 정적 ALT 지도를 넘어 연도별 ALT를 산출하고, 시간 변동을 정직하게 평가한다.
정적 공간 신호가 대부분을 차지하나, 위치 내부(within-site) 시간 신호가 실재하며
예측 가능한지 정량화한다.

데이터
- alt_era5_temporal.csv : 연도별 ERA5-Land forcing(e5t_*, 2010-2024). ALT 관측 좌표에
  정렬된 그리드(lat/lon 5dp가 관측점과 정확 일치).
- alt_above_pointlevel.csv : 알래스카 실측 ALT(연도 보유). (위치, 연도)로 집계.
  연도별 forcing과 (key=lat_lon 5dp, year)로 정확 병합.

모델(모두 그해 forcing 사용)
- STEFAN   : ALT = a + E·sqrt(TDD_that_year). E는 train에서만 적합(절편 포함/미포함 중
             train RMSE 낮은 쪽). 누설 방지 위해 fold마다 재적합.
- GBM_ANN  : 그해 공변량(e5t_*) 전체로 log1p(ALT) 직접 예측(HistGBM). T-lite에서 우위였던
             연도별 forcing 정적 GBM에 해당.
- MLP      : 경량 MLP(GPU 9). 그해 forcing 표준화 입력, log1p(ALT) 회귀.

정직 평가 2종
1) temporal holdout : leave-one-year-out. 각 연도를 test로 빼고 나머지 연도로 학습 →
   해당 연도 예측. 전 연도 OOF 취합 RMSE/R²/skill. "다른 해 정보로 미관측 연도 예측" 능력.
2) within-site 시간 skill 분리 : 각 위치의 연도 anomaly(위치평균 제거) 예측.
   - 관측 anomaly a_iy = ALT_iy - mean_i(ALT). 예측 anomaly = pred_iy - mean_i(pred).
   - 다년(>=2yr) 위치만. temporal-holdout OOF 예측에서 anomaly 상관·RMSE 산출.
   - "공간 지배분 vs 시간 예측 가능분"을 분리. 공간이 대부분이나 시간 신호 실재 여부.

추세
- 2010-2024 알래스카 관측 ALT 연평균 추세(OLS 기울기, cm/yr)와 forcing(√TDD·MAAT) 추세.
- within-site: 위치별 여름 온난(고 TDD) 해 → 심융해 상관(위치 내 anomaly 상관).

산출
- data/processed/timelapse_alaska_results.csv : (model, eval_type, n, rmse_cm, mae_cm,
  bias_cm, r2, skill_over_mean) + within-site anomaly 행 + 추세 행.
- data/processed/timelapse_alaska_meta.json : E값·추세·상관 등 메타.
- 연도별 알래스카 ALT 지도(대표 4개 연도, 공통 색축) + 전 연도 프레임(애니메이션용).
- within-site anomaly 예측 산점.
- 2010-2024 알래스카 평균 ALT 추세선 + TDD 상관 산점.

실행: CUDA_VISIBLE_DEVICES=9 python3 scripts/3_deep_learning/timelapse_alaska.py
"""
import sys, os, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn

from polar.eval_metrics import all_metrics
from polar.plotstyle import use_polar, CMAP, tnorm, add_cbar, style_geo, BAD
from polar.outputs import figpath, mappath, animpath

PROC = "data/processed"
CLIP = (np.log1p(1.0), np.log1p(600.0))
t_start = time.time()
rng = np.random.default_rng(0)
torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[env] device={DEVICE}  torch={torch.__version__}")

# forcing 컬럼(그해)
FEATS = ["e5t_maat", "e5t_tdd", "e5t_fdd", "e5t_sqrt_tdd", "e5t_twarm",
         "e5t_tcold", "e5t_stl1", "e5t_swe", "e5t_swe_prevwinter"]

# =====================================================================
# 1. 패널 구축: (위치, 연도) ALT + 그해 forcing
# =====================================================================
pt = pd.read_csv(os.path.join(PROC, "alt_above_pointlevel.csv"))
pt = pt.dropna(subset=["lat", "lon", "year", "alt_cm"]).copy()
pt["year"] = pt["year"].astype(int)
pt["key"] = pt["lat"].round(5).astype(str) + "_" + pt["lon"].round(5).astype(str)

# (key, year)로 집계 — 같은 위치·해의 다중 관측 평균
panel = (pt.groupby(["key", "year"])
         .agg(alt_cm=("alt_cm", "mean"), lat=("lat", "mean"), lon=("lon", "mean"),
              n_pts=("alt_cm", "size"), site=("site_name", "first"))
         .reset_index())

era = pd.read_csv(os.path.join(PROC, "alt_era5_temporal.csv"))
era["year"] = era["year"].astype(int)
era["key"] = era["lat"].round(5).astype(str) + "_" + era["lon"].round(5).astype(str)
era = era.drop_duplicates(["key", "year"])[["key", "year"] + FEATS]

df = panel.merge(era, on=["key", "year"], how="inner")  # forcing 있는 연도만
for c in FEATS + ["alt_cm"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["e5t_sqrt_tdd", "alt_cm"]).reset_index(drop=True)

YEARS = sorted(df["year"].unique())
print(f"[panel] rows={len(df)}  locs={df['key'].nunique()}  years={YEARS[0]}-{YEARS[-1]}")
print(f"[panel] rows/year: {df.groupby('year').size().to_dict()}")

y_cm = df["alt_cm"].values.astype(float)
ylog = np.log1p(y_cm)
Xall = df[FEATS].values.astype(float)
s_all = df["e5t_sqrt_tdd"].values.astype(float)  # 그해 √TDD
years = df["year"].values
keys = df["key"].values
to_cm = lambda p: np.expm1(np.clip(p, *CLIP))


# =====================================================================
# 2. 모델 정의
# =====================================================================
def gbm():
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, random_state=0)


def fit_E(y, s):
    """Stefan E 적합. 절편 포함/미포함 중 train RMSE 낮은 쪽."""
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    denom = float((s * s).sum())
    E0 = float((s * y).sum() / denom) if denom > 0 else 0.0
    rmse0 = float(np.sqrt(np.mean((y - E0 * s) ** 2)))
    A = np.c_[np.ones_like(s), s]
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a1, E1 = float(coef[0]), float(coef[1])
    rmse1 = float(np.sqrt(np.mean((y - (a1 + E1 * s)) ** 2)))
    if rmse1 < rmse0:
        return {"mode": "intercept", "E": E1, "a": a1}
    return {"mode": "origin", "E": E0, "a": 0.0}


def stefan_pred(fit, s):
    return fit["a"] + fit["E"] * s


class MLP(nn.Module):
    def __init__(self, n_feat, h=64, drop=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, h), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr_log, Xte, epochs=250):
    """경량 MLP. train 통계로 표준화(누설 방지). log1p 타깃."""
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Ztr = np.nan_to_num((Xtr - mu) / sd)
    Zte = np.nan_to_num((Xte - mu) / sd)
    Xt = torch.tensor(Ztr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(ytr_log, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(Zte, dtype=torch.float32, device=DEVICE)
    model = MLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss()
    model.train()
    n = len(Xt)
    bs = min(512, n)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xv).cpu().numpy()
    return pred


# =====================================================================
# 3. 정직 평가 1 — temporal holdout (leave-one-year-out)
# =====================================================================
print("\n[eval-1] temporal holdout (leave-one-year-out)")
MODELS = ["STEFAN", "GBM_ANN", "MLP"]
oof = {m: np.full(len(df), np.nan) for m in MODELS}
E_fits = []

for yr in YEARS:
    te = np.where(years == yr)[0]
    tr = np.where(years != yr)[0]
    if len(te) < 20:  # 표본 부족 연도는 예측만(평가엔 그대로 취합)
        pass
    t0 = time.time()
    # STEFAN — train에서 E 적합, test에 그해 √TDD 적용
    pf = fit_E(y_cm[tr], s_all[tr])
    oof["STEFAN"][te] = np.clip(stefan_pred(pf, s_all[te]), 1.0, 600.0)
    # GBM_ANN
    oof["GBM_ANN"][te] = np.clip(to_cm(gbm().fit(Xall[tr], ylog[tr]).predict(Xall[te])), 1.0, 600.0)
    # MLP
    oof["MLP"][te] = np.clip(to_cm(train_mlp(Xall[tr], ylog[tr], Xall[te])), 1.0, 600.0)
    E_fits.append({"holdout_year": int(yr), "E": round(pf["E"], 4), "a": round(pf["a"], 4),
                   "mode": pf["mode"], "n_te": int(len(te)), "n_tr": int(len(tr))})
    print(f"   year {yr}: n_te={len(te):5d}  E={pf['E']:.3f}({pf['mode']})  ({time.time()-t0:.0f}s)")

rows = []
for m in MODELS:
    met = all_metrics(y_cm, oof[m])
    rows.append({"model": m, "eval_type": "temporal_holdout", "n": met["n"],
                 "rmse_cm": met["rmse_cm"], "mae_cm": met["mae_cm"], "bias_cm": met["bias_cm"],
                 "r2": met["r2"], "skill_over_mean": met["skill_over_mean"]})
    print(f"   [holdout] {m:8s} RMSE={met['rmse_cm']:.2f}  R2={met['r2']:+.3f}  skill={met['skill_over_mean']:+.3f}")

# 헤드라인 모델(예측·지도용): temporal holdout RMSE 최소
best_model = min(MODELS, key=lambda m: rows[MODELS.index(m)]["rmse_cm"])
print(f"   → 헤드라인 모델: {best_model}")

# =====================================================================
# 4. 정직 평가 2 — within-site anomaly (시간 예측 가능분)
# =====================================================================
# 다년(>=2yr) 위치만. 관측/예측 anomaly = 값 - 위치평균. temporal-holdout OOF 사용.
print("\n[eval-2] within-site anomaly (시간 예측 가능분 분리)")
df_oof = df[["key", "year", "lat", "lon", "alt_cm"]].copy()
for m in MODELS:
    df_oof[f"pred_{m}"] = oof[m]

yc = df_oof.groupby("key")["year"].transform("nunique")
multi = df_oof[yc >= 2].copy()
print(f"   다년(>=2yr) 위치: {multi['key'].nunique()}  관측치: {len(multi)}")

# 위치평균 제거
multi["alt_mean_i"] = multi.groupby("key")["alt_cm"].transform("mean")
multi["alt_anom"] = multi["alt_cm"] - multi["alt_mean_i"]
anom_rows = []
anom_scatter = {}
for m in MODELS:
    multi[f"pred_{m}_mean_i"] = multi.groupby("key")[f"pred_{m}"].transform("mean")
    multi[f"anom_{m}"] = multi[f"pred_{m}"] - multi[f"pred_{m}_mean_i"]
    obs_a = multi["alt_anom"].values
    prd_a = multi[f"anom_{m}"].values
    mask = np.isfinite(obs_a) & np.isfinite(prd_a)
    oa, pa = obs_a[mask], prd_a[mask]
    # anomaly 예측 skill: RMSE·상관·anomaly SD 대비 skill
    a_rmse = float(np.sqrt(np.mean((oa - pa) ** 2)))
    a_sd = float(np.std(oa))
    a_corr = float(np.corrcoef(oa, pa)[0, 1]) if len(oa) > 2 and np.std(pa) > 0 else np.nan
    # anomaly R2(예측이 위치평균 대비 시간변동을 설명하는가; 음수면 무익)
    sse = np.sum((oa - pa) ** 2); sst = np.sum(oa ** 2)  # anomaly는 평균0 → sst=sum(oa^2)
    a_r2 = float(1 - sse / sst) if sst > 0 else np.nan
    a_skill = float(1 - a_rmse / a_sd) if a_sd > 0 else np.nan
    anom_rows.append({"model": m, "eval_type": "within_site_anomaly", "n": int(mask.sum()),
                      "rmse_cm": round(a_rmse, 3), "mae_cm": round(float(np.mean(np.abs(oa - pa))), 3),
                      "bias_cm": round(float(np.mean(pa - oa)), 3),
                      "r2": round(a_r2, 4), "skill_over_mean": round(a_skill, 4)})
    anom_scatter[m] = (oa, pa, a_corr)
    print(f"   [anom] {m:8s} anomSD={a_sd:.2f}  RMSE={a_rmse:.2f}  corr={a_corr:+.3f}  R2={a_r2:+.3f}")

# 공간 지배분: within-site anomaly SD vs 전체 ALT SD
tot_sd = float(np.std(y_cm))
anom_sd = float(np.std(multi["alt_anom"].values))
frac_temporal_var = anom_sd ** 2 / tot_sd ** 2
print(f"   공간 지배: 전체 ALT SD={tot_sd:.2f}cm, within-site anomaly SD={anom_sd:.2f}cm "
      f"→ 시간분산비율={frac_temporal_var:.3f}")

rows += anom_rows

# =====================================================================
# 5. 추세 — 2010-2024 알래스카 평균 ALT·forcing + within-site TDD 상관
# =====================================================================
print("\n[trend] 2010-2024 알래스카 평균 ALT 추세")
# 연평균(관측 위치 편향 완화 위해 위치평균 후 연평균은 표본 불균형; 여기선 관측 연평균 보고)
yr_agg = df.groupby("year").agg(alt_mean=("alt_cm", "mean"), n=("alt_cm", "size"),
                                tdd_mean=("e5t_tdd", "mean"),
                                maat_mean=("e5t_maat", "mean")).reset_index()
yv = yr_agg["year"].values.astype(float)


def ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x[m], y[m], 1)
    yhat = a + b * x[m]
    ss_res = np.sum((y[m] - yhat) ** 2); ss_tot = np.sum((y[m] - y[m].mean()) ** 2)
    r2v = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(b), float(a), float(r2v)

alt_slope, alt_int, alt_r2 = ols_slope(yv, yr_agg["alt_mean"].values)
tdd_slope, _, _ = ols_slope(yv, yr_agg["tdd_mean"].values)
maat_slope, _, _ = ols_slope(yv, yr_agg["maat_mean"].values)
print(f"   ALT 연평균 기울기={alt_slope:+.3f} cm/yr (R2={alt_r2:.3f})")
print(f"   TDD 기울기={tdd_slope:+.2f} °C·day/yr, MAAT 기울기={maat_slope:+.4f} °C/yr")

# within-site TDD-ALT anomaly 상관(온난 여름 → 심융해)
tdd_mean_i = multi.groupby("key")["e5t_tdd"].transform("mean") if "e5t_tdd" in multi.columns else None
# multi에 e5t_tdd 없으니 재병합
multi = multi.merge(df[["key", "year", "e5t_tdd", "e5t_sqrt_tdd"]], on=["key", "year"], how="left")
multi["tdd_mean_i"] = multi.groupby("key")["e5t_tdd"].transform("mean")
multi["tdd_anom"] = multi["e5t_tdd"] - multi["tdd_mean_i"]
mA = np.isfinite(multi["alt_anom"]) & np.isfinite(multi["tdd_anom"])
tdd_alt_corr = float(np.corrcoef(multi.loc[mA, "tdd_anom"], multi.loc[mA, "alt_anom"])[0, 1]) \
    if mA.sum() > 2 else np.nan
print(f"   within-site 상관 corr(ΔTDD, ΔALT)={tdd_alt_corr:+.3f}  (n={int(mA.sum())})")

# 추세 행 추가(결과 CSV)
rows.append({"model": "OBS_annual_mean", "eval_type": "trend_alt", "n": int(len(yr_agg)),
             "rmse_cm": round(alt_slope, 4), "mae_cm": round(alt_r2, 4), "bias_cm": round(alt_int, 3),
             "r2": round(alt_r2, 4), "skill_over_mean": np.nan})
rows.append({"model": "OBS_annual_mean", "eval_type": "trend_tdd", "n": int(len(yr_agg)),
             "rmse_cm": round(tdd_slope, 4), "mae_cm": np.nan, "bias_cm": np.nan,
             "r2": np.nan, "skill_over_mean": np.nan})
rows.append({"model": "within_site", "eval_type": "corr_tdd_alt_anom", "n": int(mA.sum()),
             "rmse_cm": np.nan, "mae_cm": np.nan, "bias_cm": np.nan,
             "r2": round(tdd_alt_corr, 4), "skill_over_mean": np.nan})

res = pd.DataFrame(rows)
res.to_csv(os.path.join(PROC, "timelapse_alaska_results.csv"), index=False)
print("\n=== timelapse_alaska_results.csv ===")
print(res.to_string(index=False))

# OOF 예측(지도·프레임용) 저장 — 헤드라인 모델
df_oof["pred_headline"] = oof[best_model]
df_oof.to_csv(os.path.join(PROC, "timelapse_alaska_oof.csv"), index=False)

# =====================================================================
# 6. 시각화
# =====================================================================
plt = use_polar()
import matplotlib.ticker as mticker

FIGCAT = "13_timelapse"
# 공통 색축(전 연도 예측 분위)
pred_all = df_oof["pred_headline"].values
vlo, vhi = np.nanpercentile(pred_all, [2, 98])
vlo, vhi = float(np.floor(vlo / 5) * 5), float(np.ceil(vhi / 5) * 5)
EXTENT = [df["lon"].min() - 0.5, df["lon"].max() + 0.5, df["lat"].min() - 0.3, df["lat"].max() + 0.3]

# 대표 4개 연도(표본 많은 순, 시간 분산 확보)
yr_counts = df.groupby("year").size()
rep_years = sorted(yr_counts.sort_values(ascending=False).head(6).index)
rep_years = sorted([rep_years[0], rep_years[len(rep_years) // 3],
                    rep_years[2 * len(rep_years) // 3], rep_years[-1]])
print(f"\n[viz] 대표 연도(패널): {rep_years}")


def draw_year_map(ax, yr, s=14):
    sub = df_oof[df_oof["year"] == yr]
    ax.set_facecolor("#f5f7fa")
    sc = ax.scatter(sub["lon"], sub["lat"], c=sub["pred_headline"], cmap=CMAP.alt,
                    vmin=vlo, vmax=vhi, s=s, edgecolors="#33333340", linewidths=0.2)
    ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(df["lat"].mean())))
    return sc


# --- (1) 대표 4개 연도 패널 (공통 색축) ---
fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
sc = None
for ax, yr in zip(axes.ravel(), rep_years):
    sc = draw_year_map(ax, yr)
    n_y = int((df_oof["year"] == yr).sum())
    style_geo(ax, title=f"{yr}년  (관측 위치 n={n_y})")
cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02, location="right")
cb.set_label(f"예측 ALT (cm)  [{best_model}, 그해 forcing]", fontsize=10)
cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor("#444444")
fig.suptitle("알래스카 연도별 활성층 두께(ALT) — 대표 4개 연도, 공통 색축", fontsize=13)
for ext in ("png", "pdf"):
    fig.savefig(figpath(FIGCAT, "alt_year_panels", ext), dpi=170, bbox_inches="tight")
plt.close(fig)

# --- (2) 애니메이션 프레임: 전 연도(공통 색축) ---
frames = []
for yr in YEARS:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    sc = draw_year_map(ax, yr, s=16)
    style_geo(ax, title=f"알래스카 ALT — {yr}년")
    cb = add_cbar(fig, sc, ax, f"예측 ALT (cm) [{best_model}]")
    n_y = int((df_oof["year"] == yr).sum())
    ax.text(0.02, 0.02, f"관측 위치 n={n_y}", transform=ax.transAxes, fontsize=9,
            va="bottom", ha="left", bbox=dict(fc="white", ec="#999", alpha=0.8, pad=2))
    fpath = animpath(f"timelapse_alt_{yr}", "png")
    fig.savefig(fpath, dpi=140, bbox_inches="tight")
    frames.append(fpath)
    plt.close(fig)
print(f"[viz] 애니메이션 프레임 {len(frames)}장 저장: outputs/animations/timelapse_alt_YYYY.png")

# GIF(선택) — imageio 있으면 생성
try:
    import imageio.v2 as imageio
    imgs = [imageio.imread(f) for f in frames]
    gif = animpath("timelapse_alt_alaska", "gif")
    imageio.mimsave(gif, imgs, duration=0.9, loop=0)
    print(f"[viz] GIF 저장: {gif}")
except Exception as e:
    print(f"[viz] GIF 생략({e}); 프레임 PNG는 저장됨")

# --- (3) within-site anomaly 예측 산점(헤드라인 모델) ---
oa, pa, a_corr = anom_scatter[best_model]
fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
lim = np.nanpercentile(np.abs(np.concatenate([oa, pa])), 99)
ax.axhline(0, color="#999", lw=0.6); ax.axvline(0, color="#999", lw=0.6)
ax.plot([-lim, lim], [-lim, lim], "--", color="#444", lw=0.8, label="완전 예측(y=x)")
hb = ax.hexbin(oa, pa, gridsize=40, cmap=CMAP.count, mincnt=1,
               extent=[-lim, lim, -lim, lim])
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel("관측 ALT anomaly (위치평균 제거, cm)")
ax.set_ylabel("예측 ALT anomaly (cm)")
a_row = next(r for r in anom_rows if r["model"] == best_model)
ax.set_title(f"within-site 시간 anomaly 예측 [{best_model}]\n"
             f"corr={a_corr:+.3f}  RMSE={a_row['rmse_cm']:.1f}cm  anomR2={a_row['r2']:+.3f}", fontsize=11)
cb = add_cbar(fig, hb, ax, "관측치 밀도 (개)")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
for ext in ("png", "pdf"):
    fig.savefig(figpath(FIGCAT, "within_site_anomaly_scatter", ext), dpi=170, bbox_inches="tight")
plt.close(fig)

# --- (4) 2010-2024 평균 ALT 추세선 + TDD 상관 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
# 좌: ALT 연평균 추세
ax1.scatter(yr_agg["year"], yr_agg["alt_mean"], s=np.clip(yr_agg["n"] / 200, 12, 90),
            color=CMAP.alt(0.6), edgecolors="#222", linewidths=0.4, zorder=3, label="연평균 관측 ALT")
xr = np.array([YEARS[0], YEARS[-1]], float)
ax1.plot(xr, alt_int + alt_slope * xr, "-", color="#b2182b", lw=1.6,
         label=f"OLS: {alt_slope:+.2f} cm/yr (R²={alt_r2:.2f})")
ax1.set_xlabel("연도"); ax1.set_ylabel("알래스카 평균 관측 ALT (cm)")
ax1.set_title("2010-2024 연평균 ALT 추세")
ax1.legend(fontsize=8, loc="best"); ax1.grid(alpha=0.3, lw=0.4)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax1.text(0.02, 0.02, "표시 크기 ∝ 그해 관측 수(공간 표본 불균형 주의)",
         transform=ax1.transAxes, fontsize=7.5, color="#555", va="bottom")
# 우: within-site ΔTDD vs ΔALT
dtdd = multi.loc[mA, "tdd_anom"].values
dalt = multi.loc[mA, "alt_anom"].values
hb = ax2.hexbin(dtdd, dalt, gridsize=40, cmap=CMAP.count, mincnt=1)
# 추세선
b2, a2 = np.polyfit(dtdd, dalt, 1)
xr2 = np.array([np.percentile(dtdd, 1), np.percentile(dtdd, 99)])
ax2.plot(xr2, a2 + b2 * xr2, "-", color="#b2182b", lw=1.5,
         label=f"기울기 {b2:+.4f} cm/(°C·day)\ncorr={tdd_alt_corr:+.3f}")
ax2.axhline(0, color="#999", lw=0.5); ax2.axvline(0, color="#999", lw=0.5)
ax2.set_xlabel("within-site ΔTDD (그해 - 위치평균, °C·day)")
ax2.set_ylabel("within-site ΔALT (cm)")
ax2.set_title("온난 여름(고 TDD) → 심융해 상관")
ax2.legend(fontsize=8, loc="upper left")
add_cbar(fig, hb, ax2, "관측치 밀도 (개)")
fig.suptitle("알래스카 ALT 시간 변동: 추세와 within-site TDD 반응", fontsize=13)
for ext in ("png", "pdf"):
    fig.savefig(figpath(FIGCAT, "trend_and_tdd_response", ext), dpi=170, bbox_inches="tight")
plt.close(fig)

# 대표 연도 지도도 outputs/maps에 1장(공통 색축, 최근년)
fig, ax = plt.subplots(figsize=(7, 5.6), constrained_layout=True)
sc = draw_year_map(ax, YEARS[-1], s=18)
style_geo(ax, title=f"알래스카 예측 ALT — {YEARS[-1]}년 ({best_model}, 그해 forcing)")
add_cbar(fig, sc, ax, "예측 ALT (cm)")
for ext in ("png", "pdf"):
    fig.savefig(mappath(f"timelapse_alt_{YEARS[-1]}", ext), dpi=170, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 7. 메타 저장
# =====================================================================
meta = {
    "n_panel_rows": int(len(df)), "n_locs": int(df["key"].nunique()),
    "years": [int(y) for y in YEARS], "rows_per_year": {int(k): int(v) for k, v in df.groupby("year").size().items()},
    "feats": FEATS, "headline_model": best_model,
    "temporal_holdout": {r["model"]: {"rmse_cm": r["rmse_cm"], "r2": r["r2"], "skill": r["skill_over_mean"]}
                         for r in rows if r["eval_type"] == "temporal_holdout"},
    "within_site_anomaly": {r["model"]: {"rmse_cm": r["rmse_cm"], "r2": r["r2"], "skill": r["skill_over_mean"]}
                            for r in rows if r["eval_type"] == "within_site_anomaly"},
    "anom_corr": {m: round(float(anom_scatter[m][2]), 4) for m in MODELS},
    "n_multi_year_locs": int(multi["key"].nunique()), "n_multi_year_obs": int(len(multi)),
    "total_alt_sd_cm": round(tot_sd, 3), "within_site_anom_sd_cm": round(anom_sd, 3),
    "frac_temporal_variance": round(float(frac_temporal_var), 4),
    "trend_alt_slope_cm_per_yr": round(alt_slope, 4), "trend_alt_r2": round(alt_r2, 4),
    "trend_tdd_slope_per_yr": round(tdd_slope, 4), "trend_maat_slope_per_yr": round(maat_slope, 5),
    "within_site_corr_tdd_alt": round(tdd_alt_corr, 4),
    "E_fits_by_holdout_year": E_fits,
    "rep_years_panel": [int(y) for y in rep_years],
    "elapsed_s": round(time.time() - t_start, 1),
}
with open(os.path.join(PROC, "timelapse_alaska_meta.json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n[done] {time.time()-t_start:.0f}s")
print(f"결과: timelapse_alaska_results.csv / timelapse_alaska_oof.csv / timelapse_alaska_meta.json")
print(f"그림: outputs/figures/{FIGCAT}/  outputs/maps/  outputs/animations/")
