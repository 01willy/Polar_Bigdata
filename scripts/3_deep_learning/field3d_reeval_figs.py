"""실험 2 재평가 시각화 — outputs/figures/08_field3d/ 및 outputs/maps/.

1. 깊이밴드 RMSE: base vs full × (site-GKF / 공간블록 / LORO) 그룹막대.
2. 사이트 최근접거리 히스토그램 + 0.5°블록 공유 다이어그램.
3. 깊이별 온도장 지도(알래스카, 2m·20m) + 0°C 등온선(냉색 vik+tnorm).
4. field-ALT 정합 scatter(예측 vs 관측 ALT, 1:1선, 클립 표시).
전부 냉색 규약·단위·PNG+PDF.
"""
import os, sys, json, glob, calendar, warnings
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")

ROOT = "/home/willy010313/Polar_Bigdata"
os.chdir(ROOT)
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, tnorm, add_cbar, style_geo, BAD, FROZEN, THAWED
from polar.outputs import figpath, mappath
plt = use_polar()
import matplotlib.ticker as mticker

FIGDIR = "outputs/figures/08_field3d"
os.makedirs(FIGDIR, exist_ok=True)


def savefig(fig, path_png):
    fig.savefig(path_png, dpi=260, bbox_inches="tight")
    fig.savefig(path_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


bands = pd.read_csv("data/processed/field3d_reeval_bands.csv")
leak = pd.read_csv("data/processed/field3d_reeval_leakage.csv")
altm = pd.read_csv("data/processed/field3d_reeval_altmatch.csv")
comp = pd.read_csv("data/processed/field3d_reeval_composition.csv")
meta = json.load(open("data/processed/field3d_reeval_meta.json"))

CV_LABEL = {"site_gkf6": "site-GKF6\n(기존, 근접누설)", "spatial_block6": "공간블록6\n(0.5°분리)", "loro": "LORO\n(지역분리)"}
CV_ORDER = ["site_gkf6", "spatial_block6", "loro"]
BAND_ORDER = ["0-2m", "2-5m", "5-10m", "10-20m", "all"]

# ================================================================ 그림 1: 깊이밴드 RMSE 그룹막대
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
col_base = CMAP.count(0.45); col_full = CMAP.count(0.82)
for ax, cv in zip(axes, CV_ORDER):
    sub = bands[bands.cv == cv]
    piv = sub.pivot(index="depth_band", columns="model", values="rmse_c").reindex(BAND_ORDER)
    x = np.arange(len(BAND_ORDER)); w = 0.38
    ax.bar(x - w / 2, piv["FIELD_base"], w, label="base(기후8+깊이)", color=col_base, edgecolor="#333", lw=0.5)
    ax.bar(x + w / 2, piv["FIELD_full"], w, label="full(+지형6+CCI)", color=col_full, edgecolor="#333", lw=0.5)
    # 심부 개선 여부 표기(음수=full 개선)
    for i, b in enumerate(BAND_ORDER):
        try:
            d = float(piv.loc[b, "FIELD_full"] - piv.loc[b, "FIELD_base"])
            y = max(piv.loc[b, "FIELD_base"], piv.loc[b, "FIELD_full"]) + 0.03
            lbl = ("-" if d < 0 else "+") + f"{abs(d):.2f}"
            ax.text(i, y, lbl, ha="center", va="bottom", fontsize=7.5,
                    color=(FROZEN if d < 0 else THAWED))
        except Exception:
            pass
    ax.set_xticks(x); ax.set_xticklabels(BAND_ORDER, fontsize=9)
    ax.set_title(CV_LABEL[cv], fontsize=11)
    ax.set_xlabel("깊이 밴드")
    ax.grid(axis="y", alpha=0.35)
    ax.axvspan(1.5, 3.5, color="#4477aa", alpha=0.06, zorder=0)  # 심부 강조(5-20m)
axes[0].set_ylabel("OOF RMSE (°C)")
axes[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle("지중온도장 깊이밴드 RMSE: 공변량 추가(full) 이득의 CV 의존성 (파랑=심부 5-20m)",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.text(0.5, -0.05, "숫자 = full-base RMSE차(음수·청=full 개선, 양수·적=악화). site-GKF에서만 심부 이득이 나타남. "
         "공간블록 델타는 pooled OOF(불균형 fold·스위스 가중)이라 정밀 효과크기 아님.",
         ha="center", fontsize=8.5, color="#555")
savefig(fig, os.path.join(FIGDIR, "band_rmse_by_cv.png"))
print("[fig1] band_rmse_by_cv")

# ================================================================ 그림 2: 누설 진단(최근접거리 + 블록공유)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
# (좌) 최근접거리 히스토그램(log-x)
nn = leak.nn_km.values.copy()
nn = np.clip(nn, 1e-3, None)
bins = np.logspace(np.log10(1e-3), np.log10(nn.max() * 1.1), 34)
ax1.hist(nn, bins=bins, color=CMAP.count(0.7), edgecolor="#333", lw=0.4)
ax1.set_xscale("log")
# 로그축 눈금: mathtext U+2212(폰트 미포함) 회피 위해 일반 문자열 포매터로 교체
def _logfmt(x, _pos):
    if x <= 0:
        return ""
    e = np.log10(x)
    if abs(e - round(e)) < 1e-6:
        e = int(round(e))
        return {-3: "0.001", -2: "0.01", -1: "0.1", 0: "1", 1: "10", 2: "100", 3: "1000"}.get(e, f"{x:g}")
    return ""
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(_logfmt))
ax1.xaxis.set_minor_formatter(mticker.NullFormatter())
ax1.axvline(1.0, color=THAWED, ls="--", lw=1.4, label="1 km")
med = float(np.median(leak.nn_km))
ax1.axvline(med, color=FROZEN, ls="-", lw=1.4, label=f"중앙 {med:.2f} km")
ax1.set_xlabel("사이트 간 최근접거리 (km, 로그축)")
ax1.set_ylabel("사이트 수")
frac1 = 100 * (leak.nn_km < 1).mean()
ax1.set_title(f"시추공 근접도: {frac1:.0f}%가 1km 이내 이웃 보유", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
# (우) 0.5°블록 공유 사이트 수 분포
bc = leak.groupby("block").size()
sizes = np.arange(1, int(bc.max()) + 1)
counts = [int((bc == s).sum()) for s in sizes]
ax2.bar(sizes, counts, color=CMAP.count(0.6), edgecolor="#333", lw=0.4)
ax2.set_xlabel("한 0.5°블록에 속한 사이트 수")
ax2.set_ylabel("블록 수")
share = 100 * (leak.block.map(leak.block.value_counts()) > 1).mean()
ax2.set_title(f"0.5°블록 공유: 사이트 {share:.0f}%가 다중공유 블록 소속", fontsize=11)
ax2.set_xticks(sizes[::max(1, len(sizes) // 12)])
ax2.grid(axis="y", alpha=0.3)
fig.suptitle("site-GroupKFold 낙관 원인: 근접 시추공 누설 정량화", fontsize=12.5, fontweight="bold", y=1.02)
savefig(fig, os.path.join(FIGDIR, "leakage_diagnosis.png"))
print("[fig2] leakage_diagnosis")

# ================================================================ 그림 3: 깊이별 온도장 지도(알래스카, 2m·20m)
# 알래스카 격자에 full 모델(전체 데이터 적합)로 2m·20m 온도 예측 후 지도화 + 0°C 등온선.
E5F = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]


def fourier(dm):
    dn = (np.asarray(dm) / 30.0).astype(np.float32)
    out = []
    for k in range(5):
        out += [np.sin(2 ** k * np.pi * dn), np.cos(2 ** k * np.pi * dn)]
    return np.column_stack(out)


# 라벨·기후8 재구성(재평가 스크립트와 동일 파이프라인)
g = pd.read_csv("data/processed/ground_temp_all.csv")
g = g[(g.depth_m > 0) & (g.depth_m <= 30) & (g.temp_c > -25) & (g.temp_c < 25)].reset_index(drop=True)
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim0 = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
elat, elon = clim0["latitude"].values, clim0["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]


def derive_grid(c):
    t = c["t2m"].values - 273.15; stl = c["stl1"].values - 273.15; sdp = c["sd"].values
    tdd = np.nansum(np.clip(t, 0, None) * days, 0); fdd = np.nansum(np.clip(-t, 0, None) * days, 0)
    return dict(e5_maat=np.nanmean(t, 0), e5_tdd=tdd, e5_fdd=fdd, e5_sqrt_tdd=np.sqrt(tdd),
                e5_twarm=np.nanmax(t, 0), e5_tcold=np.nanmin(t, 0),
                e5_stl1=np.nanmean(stl, 0), e5_swe=np.nanmean(sdp, 0))


E5 = derive_grid(clim0)
iy = np.clip(np.searchsorted(-elat, -g.lat.values), 0, len(elat) - 1)
ix = np.clip(np.searchsorted(elon, g.lon.values), 0, len(elon) - 1)
for k, gr in E5.items():
    g[k] = gr[iy, ix].astype(np.float32)
g = g.dropna(subset=["e5_maat"]).reset_index(drop=True)
g["logd"] = np.log1p(g.depth_m)
FF = fourier(g.depth_m.values)
FFn = [f"ff{i}" for i in range(FF.shape[1])]
for i, n in enumerate(FFn):
    g[n] = FF[:, i].astype(np.float32)
DEPTHF = ["depth_m", "logd"] + FFn
FEAT_BASE = E5F + DEPTHF

# 알래스카 격자(ERA5-Land subset)
ALA = dict(lat=(58, 72), lon=(-168, -140))
lam = (elat >= ALA["lat"][0]) & (elat <= ALA["lat"][1])
lom = (elon >= ALA["lon"][0]) & (elon <= ALA["lon"][1])
gla = elat[lam]; glo = elon[lom]
LO, LA = np.meshgrid(glo, gla)
grid_e5 = {k: E5[k][np.ix_(lam, lom)] for k in E5F}
valid = np.isfinite(grid_e5["e5_maat"])

# base 모델(전체 데이터 적합) — 지도는 base로(누설 없는 결론에 맞춰 기후+깊이만; 지형/CCI 알래스카 미부착)
for c in FEAT_BASE:
    med = g[c].median()
    g[c] = g[c].fillna(med if np.isfinite(med) else 0.0)
mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
mdl.fit(g[FEAT_BASE].values, g.temp_c.values)


def predict_depth(depth_m):
    n = valid.sum()
    Xd = np.zeros((n, len(FEAT_BASE)), np.float32)
    ff = fourier(np.full(n, depth_m))
    feat_map = {}
    for k in E5F:
        feat_map[k] = grid_e5[k][valid]
    feat_map["depth_m"] = np.full(n, depth_m, np.float32)
    feat_map["logd"] = np.full(n, np.log1p(depth_m), np.float32)
    for i in range(10):
        feat_map[f"ff{i}"] = ff[:, i]
    for j, c in enumerate(FEAT_BASE):
        Xd[:, j] = feat_map[c]
    p = mdl.predict(Xd)
    out = np.full(valid.shape, np.nan)
    out[valid] = p
    return out


t2 = predict_depth(2.0)
t20 = predict_depth(20.0)
vmax = np.nanmax(np.abs(np.concatenate([t2[np.isfinite(t2)], t20[np.isfinite(t20)]])))
vmax = float(np.ceil(vmax))
norm = tnorm(-vmax, vmax, 0.0)

# 심부 밴드 도메인 편중 비율(경고 캡션용)
try:
    deep_sw = float(comp[comp.depth_band.isin(["5-10m", "10-20m"])].frac_switzerland.mean())
except Exception:
    deep_sw = float(meta.get("sample_composition", {}).get("deep_5_20m_frac_switzerland", 0.82))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, arr, dz in zip(axes, [t2, t20], [2, 20]):
    pm = ax.pcolormesh(LO, LA, arr, cmap=CMAP.temp, norm=norm, shading="auto")
    cs = ax.contour(LO, LA, arr, levels=[0.0], colors=[FROZEN], linewidths=1.8)
    ax.clabel(cs, fmt="0°C", fontsize=8)
    # 심부 지도에 도메인 밖 외삽 경고 배지
    if dz >= 5:
        ax.text(0.02, 0.02,
                f"도메인 밖 외삽\n(심부 학습표본 {100 * deep_sw:.0f}% 스위스 알프스)",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                color="#7a1f1f",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fbeaea", ec="#c98a8a", lw=0.7))
    style_geo(ax, title=f"깊이 {dz} m 연평균 지중온도")
    ax.set_aspect(1.0 / np.cos(np.radians(65)))
    add_cbar(fig, pm, ax, "온도 (°C)")
fig.suptitle("알래스카 지중온도장(기후8+깊이) + 0°C 등온선 · 심부는 도메인 밖 외삽",
             fontsize=12.5, fontweight="bold", y=1.0)
fig.text(0.5, -0.03,
         f"20m 지도는 심부 학습표본이 스위스 알프스(암빙하)에 {100 * deep_sw:.0f}% 편중된 모델의 외삽이다. "
         "알래스카 심부 절대값으로 해석하지 말 것.",
         ha="center", fontsize=8.5, color="#555")
savefig(fig, mappath("field3d_reeval_temp_depth"))
print("[fig3] field3d_reeval_temp_depth (maps)")

# ================================================================ 그림 3b: 밴드별 도메인 구성 막대
fig, ax = plt.subplots(figsize=(7.2, 4.2))
BO = ["0-2m", "2-5m", "5-10m", "10-20m"]
cc = comp.set_index("depth_band").reindex(BO)
x = np.arange(len(BO))
ax.bar(x, 100 * cc.frac_switzerland, 0.6, color=CMAP.count(0.72),
       edgecolor="#333", lw=0.5, label="스위스(알프스 암빙하)")
ax.bar(x, 100 * (1 - cc.frac_switzerland), 0.6, bottom=100 * cc.frac_switzerland,
       color=CMAP.count(0.32), edgecolor="#333", lw=0.5, label="기타 지역")
for i, b in enumerate(BO):
    ax.text(i, 100 * cc.frac_switzerland.iloc[i] - 4, f"{100 * cc.frac_switzerland.iloc[i]:.0f}%",
            ha="center", va="top", fontsize=8.5, color="white", fontweight="bold")
    ax.text(i, 101, f"n={int(cc.n.iloc[i])}", ha="center", va="bottom", fontsize=8, color="#555")
ax.set_xticks(x); ax.set_xticklabels(BO)
ax.set_ylabel("표본 구성 (%)"); ax.set_xlabel("깊이 밴드")
ax.set_ylim(0, 112)
ax.set_title("깊이 밴드별 도메인 구성: 심부일수록 스위스 알프스 집중", fontsize=11.5)
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(axis="y", alpha=0.3)
fig.text(0.5, -0.04, "심부(5-20m) 표본의 80%+가 스위스 알프스 암빙하 시추공이다. "
         "심부 field 결과·알래스카 심부 지도는 도메인 밖 외삽으로 해석해야 한다.",
         ha="center", fontsize=8.5, color="#555")
savefig(fig, os.path.join(FIGDIR, "band_domain_composition.png"))
print("[fig3b] band_domain_composition")

# ================================================================ 그림 4: field-ALT 정합 scatter
fig, ax = plt.subplots(figsize=(5.8, 5.6))
m = altm.field_alt_seasmax_cm.notna() & altm.obs_alt_cm.notna()
pr = altm.loc[m, "field_alt_seasmax_cm"].values
ob = altm.loc[m, "obs_alt_cm"].values
# 클립 경계(0, 400) 표시
clip_hi = 400.0
at_clip = pr >= clip_hi - 1e-6
sc = ax.scatter(ob[~at_clip], pr[~at_clip], s=26, c=CMAP.count(0.68),
                edgecolor="#333", lw=0.4, alpha=0.85, label="유도 ALT")
if at_clip.any():
    ax.scatter(ob[at_clip], pr[at_clip], s=34, marker="^", c=THAWED,
               edgecolor="#333", lw=0.4, label="상한(400cm) 클립")
lim = max(np.nanmax(pr), np.nanmax(ob)) * 1.05
ax.plot([0, lim], [0, lim], color="#666", ls="--", lw=1.2, label="1:1 선")
ax.axhline(clip_hi, color=THAWED, ls=":", lw=1.0, alpha=0.7)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("관측 ALT (cm)")
ax.set_ylabel("유도 ALT (연최대 포락선, held-out δ) (cm)")
acg = meta["alt_consistency_honest"]
st = acg["seasonal_max_envelope"]
n_matched = acg.get("n_matched", len(altm))
ax.set_title(f"field-ALT 정합: 귀무결과 (r={st['corr']}, 1:1 무추종)", fontsize=11.5)
# 귀무결과 배지 + 표본 감소 명시
ax.text(0.03, 0.97,
        f"매칭 {n_matched} → 평가 n={st['n']}\n(포락선 미교차 {acg.get('n_nan_envelope','?')}, "
        f"400cm클립 {acg.get('n_clip_sat','?')})\nRMSE={st['rmse']}cm  r={st['corr']}",
        transform=ax.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fbeaea", ec="#c98a8a", lw=0.7))
ax.legend(fontsize=8.5, loc="lower right")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
fig.text(0.5, -0.02, "유도 ALT가 관측 ALT를 추종하지 못한다(귀무·음성). "
         "ALT 교차 경로는 field 모델의 독립 검증 근거가 아니다.",
         ha="center", fontsize=8.5, color="#555")
savefig(fig, os.path.join(FIGDIR, "alt_match_scatter.png"))
print("[fig4] alt_match_scatter (귀무결과)")
print("\n[완료] 그림 5종 저장(PNG+PDF)")
