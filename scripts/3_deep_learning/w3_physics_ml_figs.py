"""W3 시각화: 물리결합 엔진 전이 검증 그림 5종.

입력: data/processed/w3_physics_ml_{results,perregion,Ex_diag,council,oof}.csv
출력: outputs/figures/11_physics_ml/*.png|pdf

그림:
  1 모델별 전이(LORO) 지역별 RMSE·skill 막대(알래스카·CA·레나 그룹). PHYS_const 기준선.
  2 PHYS_nn vs ML_pure_nn 전이 차이(물리 제약 순효과) 막대. 레나 seed 요동 주석.
  3 E(x) 지역별 분포(박스) + 물리 타당범위(1-5 cm/√degday) 음영.
  4 콘슬 44셀(in-domain, 전이 아님) 산점. 상관·mean-collapse 노출. '약 1.7배(68%)' 문구.
  5 예측-실측 1:1(레나 LORO): 상관 약 0·debiased RMSE 병기, 모두 평균예측 이하 skill 명시.

정정 요약(적대적 검증 반영):
  - 콘슬은 held-out 전이가 아니라 완전 in-domain(44셀 전부 학습 포함). 전이 주장 불가.
  - PHYS_const 콘슬 과대예측은 1.68배(68%)이지 '2배'가 아니다.
  - PHYS_nn 콘슬 bias 감소는 셀별 물리보정이 아니라 평균회귀(좁은 밴드·상관 약 0.2, seed별 요동).
  - PHYS_soil 의 레나 RMSE 개선은 mean-shift 뿐, debiased RMSE·상관은 PHYS_const 와 동일.
  - 레나 PHYS_nn 헤드라인 RMSE 는 seed별 34-70cm 요동. 구간 병기(단일값 과장 금지).
전부 냉색·단위·PNG+PDF.
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from polar.plotstyle import use_polar, CMAP, tnorm, BAD, FROZEN, THAWED
from polar.outputs import figpath
plt = use_polar()

PROC = "data/processed"
CAT = "11_physics_ml"
res = pd.read_csv(os.path.join(PROC, "w3_physics_ml_results.csv"))
per = pd.read_csv(os.path.join(PROC, "w3_physics_ml_perregion.csv"))
exd = pd.read_csv(os.path.join(PROC, "w3_physics_ml_Ex_diag.csv"))
council = pd.read_csv(os.path.join(PROC, "w3_physics_ml_council.csv"))
oof = pd.read_csv(os.path.join(PROC, "w3_physics_ml_oof.csv"))

REGIONS = ["ABoVE_AK", "ABoVE_CA", "Lena_RU"]
REG_LABEL = {"ABoVE_AK": "알래스카", "ABoVE_CA": "캐나다", "Lena_RU": "레나(전이 시험)"}
MODELS = ["PHYS_const", "PHYS_clim", "PHYS_soil", "PHYS_nn", "ML_pure_nn", "ML_pure_gbm", "RESIDUAL", "HYBRID_aoa"]
# 냉색 팔레트(붉은계열 회피): 물리계열=청록, ML계열=회청/자주
COL = {"PHYS_const": "#08519c", "PHYS_clim": "#3182bd", "PHYS_soil": "#6baed6",
       "PHYS_nn": "#54278f", "ML_pure_nn": "#9e9ac8", "ML_pure_gbm": "#807dba",
       "RESIDUAL": "#525252", "HYBRID_aoa": "#238b45"}


def save(fig, name):
    fig.savefig(figpath(CAT, name, "png"), dpi=200, bbox_inches="tight")
    fig.savefig(figpath(CAT, name, "pdf"), bbox_inches="tight")
    plt.close(fig)


# ============ 1. 모델별 LORO 지역별 RMSE·skill 막대 ============
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
x = np.arange(len(REGIONS)); w = 0.10
for i, m in enumerate(MODELS):
    vals = [float(per[(per.model == m) & (per.region == r)]["rmse_cm"].values[0]) for r in REGIONS]
    axes[0].bar(x + (i - 3.5) * w, vals, w, color=COL[m], label=m, edgecolor="white", linewidth=0.4)
axes[0].axhline(0, color="#333", lw=0.6)
# PHYS_const 기준선(지역별 점선)
for j, r in enumerate(REGIONS):
    c = float(per[(per.model == "PHYS_const") & (per.region == r)]["rmse_cm"].values[0])
    axes[0].plot([x[j] - 0.45, x[j] + 0.45], [c, c], ls="--", color="#08519c", lw=1.0, zorder=5)
axes[0].set_xticks(x); axes[0].set_xticklabels([REG_LABEL[r] for r in REGIONS])
axes[0].set_ylabel("LORO RMSE (cm)")
axes[0].set_title("(a) 전이 RMSE: 지역별(파선=PHYS_const 기준)")
axes[0].legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)

for i, m in enumerate(MODELS):
    vals = [float(per[(per.model == m) & (per.region == r)]["skill_over_mean"].values[0]) for r in REGIONS]
    axes[1].bar(x + (i - 3.5) * w, vals, w, color=COL[m], label=m, edgecolor="white", linewidth=0.4)
axes[1].axhline(0, color="#333", lw=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels([REG_LABEL[r] for r in REGIONS])
axes[1].set_ylabel("skill over mean")
axes[1].set_title("(b) 전이 skill: 0 미만은 평균예측보다 나쁨")
fig.suptitle("모델별 전이(LORO) 성능: 레나가 핵심 시험", y=1.02, fontsize=13)
save(fig, "01_loro_rmse_skill_byregion")


# ============ 2. 물리 제약 순효과: ML_pure_nn 빼기 PHYS_nn (양수=물리가 도움) ============
fig, ax = plt.subplots(figsize=(7.2, 4.6))
diff = []
for r in REGIONS:
    pnn = float(per[(per.model == "PHYS_nn") & (per.region == r)]["rmse_cm"].values[0])
    pur = float(per[(per.model == "ML_pure_nn") & (per.region == r)]["rmse_cm"].values[0])
    diff.append(pur - pnn)  # 양수면 물리층이 RMSE를 낮춤
colors = [FROZEN if d > 0 else THAWED for d in diff]
bars = ax.bar(x, diff, 0.55, color=colors, edgecolor="white")
ax.axhline(0, color="#333", lw=0.8)
for j, d in enumerate(diff):
    ax.annotate(f"{d:+.1f}", (x[j], d), ha="center",
                va="bottom" if d >= 0 else "top", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([REG_LABEL[r] for r in REGIONS])
ax.set_ylabel("RMSE 차이 (ML_pure_nn - PHYS_nn) (cm)")
ax.set_title("물리 제약 순효과: 동일 입력·백본, 물리층 유무만 차이\n"
             "음수(적)=물리층이 전이를 오히려 악화(레나서 특히)")
# 레나 PHYS_nn 은 seed 요동이 큼(단일 seed RMSE 34-70cm). 정성 결론(악화)은 모든 seed 성립.
fig.subplots_adjust(bottom=0.26)
fig.text(0.5, 0.015, "주의: 레나 PHYS_nn 은 seed별 RMSE 34-70cm 요동. 부호(악화)는 모든 seed 성립,\n"
         "수치는 seed 앙상블 값이며 단일 seed 헤드라인은 과장 위험",
         ha="center", va="bottom", fontsize=7.4, color="#555")
save(fig, "02_physics_constraint_net_effect")


# ============ 3. E(x) 지역별 분포 + 물리 타당범위 ============
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
lo = oof[oof["Ex_soil_LORO"].notna()]
for ax, col, title in [(axes[0], "Ex_soil_LORO", "PHYS_soil"), (axes[1], "Ex_nn_LORO", "PHYS_nn")]:
    data = [oof[(oof.region == r) & oof[col].notna()][col].values for r in REGIONS]
    bp = ax.boxplot(data, positions=x, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#111", lw=1.4))
    for patch, r in zip(bp["boxes"], REGIONS):
        patch.set_facecolor(CMAP.count(0.55)); patch.set_alpha(0.85)
    ax.axhspan(1.0, 5.0, color="#c6dbef", alpha=0.35, zorder=0)
    ax.text(0.02, 4.7, "물리 타당범위 1 to 5 cm/√degday", fontsize=8, color="#2166ac", va="top")
    ax.set_xticks(x); ax.set_xticklabels([REG_LABEL[r] for r in REGIONS], fontsize=9)
    ax.set_title(f"{title}: E(x) 지역별")
axes[0].set_ylabel("적합된 E(x) (cm / √degday)")
fig.suptitle("E(x) 지역별 분포: 지역 더미처럼 튀는지 진단(전이 fold 값)", y=1.02, fontsize=12)
save(fig, "03_Ex_byregion_boxplot")


# ============ 4. 콘슬 44셀(in-domain) 산점: mean-collapse 노출 ============
# 주의(leakage): 콘슬 44셀 전부가 최종 모델 학습에 포함된 in-domain 이다. 전이 주장 불가.
# 콘슬 44셀은 √TDD 동일값이라 상수 E 모델(const·soil)은 수평선(단일점)으로 나타난다.
# PHYS_nn 은 좁은 밴드로 뭉쳐 obs 와 상관 낮음(셀별 물리보정 아닌 평균회귀).
fig, ax = plt.subplots(figsize=(6.6, 6.4))
lim = [0, 90]
ax.plot(lim, lim, ls="--", color="#333", lw=1.0, label="1:1")
obs_c = council["alt_cm_obs"].values
for m, c, mk in [("pred_PHYS_const", "#08519c", "o"), ("pred_PHYS_soil", "#6baed6", "s"),
                 ("pred_PHYS_nn", "#238b45", "^")]:
    p = council[m].values
    msk = np.isfinite(p) & np.isfinite(obs_c)
    bias = float(np.mean(p[msk] - obs_c[msk]))
    ratio = float(p[msk].mean() / obs_c[msk].mean())
    corr = float(np.corrcoef(p[msk], obs_c[msk])[0, 1]) if p[msk].std() > 1e-9 else np.nan
    ax.scatter(obs_c, p, s=44, color=c, marker=mk, edgecolor="white", linewidth=0.4, alpha=0.85,
               label=f"{m.replace('pred_','')}  bias{bias:+.1f}  r={corr:+.2f}")
ax.axhline(float(np.nanmean(obs_c)), color="#999", lw=0.8, ls=":")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("관측 ALT (cm)"); ax.set_ylabel("예측 ALT (cm)")
# 실측 과대예측 비율은 1.68배(약 68%)이지 2배가 아니다. PHYS_nn 의 bias 감소는 평균회귀.
ax.set_title("KPDC 콘슬 44셀(in-domain, 전이 아님)\n"
             "PHYS_const 약 1.7배(68%) 과대예측 · PHYS_nn 은 평균회귀(좁은 밴드·상관 낮음)")
ax.text(0.03, 0.03, "회색점선=관측 평균\n상수 E 모델은 √TDD 동일값이라 수평선",
        transform=ax.transAxes, fontsize=7.5, va="bottom", color="#555")
ax.legend(fontsize=8, loc="upper left")
save(fig, "04_council_calibration_scatter")


# ============ 5. 레나 LORO 예측-실측 1:1 ============
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), sharex=True, sharey=True)
lena = oof[oof.region == "Lena_RU"]
lim = [0, 130]
panel = [("LORO__PHYS_const", "PHYS_const"), ("LORO__PHYS_soil", "PHYS_soil"),
         ("LORO__ML_pure_gbm", "ML_pure_gbm (raw 토양)")]
obs_std = float(lena.alt_cm.std())
for ax, (col, title) in zip(axes, panel):
    ax.plot(lim, lim, ls="--", color="#333", lw=1.0)
    ax.axhline(lena.alt_cm.mean(), color="#999", lw=0.8, ls=":")
    ax.scatter(lena.alt_cm, lena[col], s=8, color=CMAP.count(0.6), alpha=0.35, edgecolor="none")
    p = lena[col].values; o = lena.alt_cm.values
    m = np.isfinite(p) & np.isfinite(o)
    rmse = float(np.sqrt(np.mean((p[m] - o[m]) ** 2)))
    bias = float(np.mean(p[m] - o[m]))
    corr = float(np.corrcoef(p[m], o[m])[0, 1]) if p[m].std() > 1e-9 else np.nan
    # debiased RMSE = 잔차 표준편차. obs_std 근처면 셀별 skill 없음(평균예측 수준).
    drmse = float(np.sqrt(np.mean(((p[m] - bias) - o[m]) ** 2)))
    ax.set_title(f"{title}\nRMSE {rmse:.1f}  bias {bias:+.1f}  r={corr:+.2f}  drmse {drmse:.1f}", fontsize=9.5)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("관측 ALT (cm)")
axes[0].set_ylabel("예측 ALT (cm)")
axes[0].text(0.04, 0.94, "점선=1:1, 회색점선=관측 평균", transform=axes[0].transAxes,
             fontsize=7.5, va="top", color="#555")
# 세 모델 모두 상관 약 0·debiased RMSE 가 obs 표준편차 근처라 셀별 skill 이 없다.
fig.text(0.5, -0.02, f"세 모델 모두 셀별 상관 약 0, debiased RMSE 가 관측 표준편차({obs_std:.1f}cm) 근처다. "
         "즉 RMSE 차이는 대부분 bias(평균이동)이며 어느 모델도 레나서 평균예측 이상의 skill 이 없다.",
         ha="center", va="top", fontsize=7.8, color="#555")
fig.suptitle("레나 전이(LORO) 예측-실측: 모두 평균예측 이하 skill, 차이는 대부분 bias", y=1.03, fontsize=12)
save(fig, "05_lena_transfer_scatter")

print("[figs] 저장 완료:", figpath(CAT, "", "").rsplit("/", 1)[0])
for n in ["01_loro_rmse_skill_byregion", "02_physics_constraint_net_effect",
          "03_Ex_byregion_boxplot", "04_council_calibration_scatter", "05_lena_transfer_scatter"]:
    print("  ", figpath(CAT, n, "png"))
