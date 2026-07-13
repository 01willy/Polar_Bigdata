# -*- coding: utf-8 -*-
"""신규 실험 그림(cell ablation·conformal·AOA·T-lite·pred-obs)을 논문 관례로 생성.
원칙: 도판에 결론형 굵은 제목을 넣지 않는다(해석은 슬라이드 캡션). 패널 라벨 (a)/(b),
중립적 축 제목, 냉색 규약(붉은 계열 금지). NanumGothic. 산출: deck/assets/figs/*.png/pdf (200dpi).
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"]:
    fm.fontManager.addfont(p)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'svg.fonttype':'none',
                     'pdf.fonttype':42,'axes.edgecolor':'#3a4047','axes.linewidth':0.9,
                     'xtick.color':'#3a4047','ytick.color':'#3a4047','text.color':'#1b1e23',
                     'font.size':10.5})
PROC = "data/processed"; OUT = "deck/assets/figs"; os.makedirs(OUT, exist_ok=True)
NAVY, TEAL, TEAL2, SLATE, MUTE, GOLD = "#1f3a52", "#0e5a61", "#2c7d83", "#5c666e", "#a7adb3", "#8a5a12"
INK = "#1b1e23"
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", name)
def despine(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
def panel(ax, tag, x=-0.02, y=1.03):
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right', color=INK)

# ============ Fig A: cell-level 다중모달 ablation ============
ab = pd.read_csv(f"{PROC}/alt_ablation_cell_results.csv")
order = ["M0 지역평균","M1 기후","M2 지형","M3 기후+지형","M4 +InSAR","M5 +PolSAR","M8 +CCI prior","M9 전체","Mloc 위치만(대조)"]
lab = {"M0 지역평균":"M0\n지역평균","M1 기후":"M1\n기후","M2 지형":"M2\n지형","M3 기후+지형":"M3\n기후+지형",
       "M4 +InSAR":"M4\n+InSAR","M5 +PolSAR":"M5\n+PolSAR","M8 +CCI prior":"M8\n+CCI","M9 전체":"M9\n전체","Mloc 위치만(대조)":"위치만\n(대조)"}
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
ymax = ab.rmse_cm.max()*1.20
for ax, cv, ttl, tag in zip(axes, ["spatial_block","LORO"], ["(a) 공간블록 CV","(b) LORO (지역 전이)"], ["(a)","(b)"]):
    sub = ab[ab.cv_type==cv].set_index("config").reindex(order).reset_index()
    x = np.arange(len(sub)); best = int(sub.rmse_cm.idxmin())
    cols = []
    for i,(cfg,sk) in enumerate(zip(sub.config, sub.skill_over_mean)):
        if "위치만" in cfg: cols.append(GOLD)
        elif i==best: cols.append(TEAL)
        elif sk<0: cols.append(MUTE)
        else: cols.append(NAVY)
    ax.bar(x, sub.rmse_cm, color=cols, width=0.70, zorder=3)
    for xi,(r,sk) in enumerate(zip(sub.rmse_cm, sub.skill_over_mean)):
        ax.text(xi, r+0.3, f"{r:.1f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi, 0.7, f"{sk*100:+.0f}%", ha="center", fontsize=7.5, color="white" if r>3 else SLATE)
    ax.set_xticks(x); ax.set_xticklabels([lab[c] for c in sub.config], fontsize=8)
    ax.set_xlabel(ttl, fontsize=11, color=INK, labelpad=6)
    ax.set_ylim(0, ymax); ax.grid(axis="y", alpha=0.20, zorder=0); despine(ax)
axes[0].set_ylabel("ALT RMSE (cm)", fontsize=11)
# 범례(막대 색 의미)
from matplotlib.patches import Patch
leg = [Patch(fc=TEAL, label="최저 RMSE"), Patch(fc=NAVY, label="skill>0"),
       Patch(fc=MUTE, label="skill<0(평균 이하)"), Patch(fc=GOLD, label="위치 대조군")]
axes[1].legend(handles=leg, fontsize=8, frameon=False, loc="upper right", ncol=1, handlelength=1.1)
axes[0].text(0.0, 1.06, "막대 위: RMSE(cm) · 막대 안: skill-over-mean", transform=axes[0].transAxes,
             fontsize=8.2, color=SLATE, va="bottom")
fig.tight_layout(); save(fig, "cell_ablation")

# ============ Fig B: conformal 보정 + AOA DI-bin ============
cf = pd.read_csv(f"{PROC}/alt_conformal_cell_results.csv")
di = pd.read_csv(f"{PROC}/alt_aoa_cell_transfer.csv")
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
ax = axes[0]
vals = [cf[cf.setting.str.contains("raw")].coverage_pct.iloc[0], cf[cf.setting.str.contains("CQR")].coverage_pct.iloc[0]]
w = [cf[cf.setting.str.contains("raw")].width_cm.iloc[0], cf[cf.setting.str.contains("CQR")].width_cm.iloc[0]]
ax.bar([0,1], vals, color=[MUTE, TEAL], width=0.5, zorder=3)
ax.axhline(90, color=GOLD, ls="--", lw=1.5, label="목표 90%")
for i,(v,wi) in enumerate(zip(vals,w)):
    ax.text(i, v-6 if v>80 else v+2, f"{v:.0f}%", ha="center", fontsize=13, fontweight="bold", color="white" if v>80 else INK)
    ax.text(i, 4, f"폭 {wi:.0f}cm", ha="center", fontsize=8.5, color=SLATE if v<80 else "white")
ax.set_xticks([0,1]); ax.set_xticklabels(["raw 분위 GBM","CQR 보정"], fontsize=10)
ax.set_ylabel("관측 커버리지 (%)", fontsize=11); ax.set_ylim(0,100); ax.grid(axis="y", alpha=0.20, zorder=0)
ax.legend(fontsize=8.5, loc="lower right", frameon=False); despine(ax); panel(ax, "(a)")
ax = axes[1]
x = np.arange(len(di))
ax.plot(x, di.rmse_cm, "o-", color=NAVY, lw=1.9, ms=6, label="RMSE (cm)", zorder=3)
ax.set_ylabel("LORO RMSE (cm)", color=NAVY, fontsize=11); ax.tick_params(axis="y", labelcolor=NAVY)
ax.set_xticks(x); ax.set_xticklabels([f"B{d}\nDI≈{m:.0f}" for d,m in zip(di.di_decile, di.di_mean)], fontsize=7.8)
ax.set_xlabel("환경 비유사도(DI) 구간", fontsize=10)
ax2 = ax.twinx()
ax2.plot(x, di.coverage_pct, "s--", color=TEAL2, lw=1.7, ms=5, label="커버리지 (%)")
ax2.axhline(90, color=GOLD, ls=":", lw=1.2); ax2.set_ylabel("커버리지 (%)", color=TEAL2, fontsize=11)
ax2.tick_params(axis="y", labelcolor=TEAL2); ax2.set_ylim(30,100)
despine(ax); ax2.spines['top'].set_visible(False)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8, loc="upper left", frameon=False); panel(ax, "(b)")
fig.tight_layout(); save(fig, "conformal_aoa")

# ============ Fig C: T-lite 게이트 ============
tl = pd.read_csv(f"{PROC}/tlite_sequence_gate_results.csv")
mods = ["persistence","climatology","gbm_annual","gru","tcn"]
mlab = {"persistence":"persistence","climatology":"climatology","gbm_annual":"GBM-annual","gru":"GRU","tcn":"TCN"}
mcol = {"persistence":MUTE,"climatology":"#c3c8cc","gbm_annual":NAVY,"gru":TEAL,"tcn":TEAL2}
fig, ax = plt.subplots(figsize=(8.6, 4.6))
x = np.arange(2); wbar = 0.15
for i,m in enumerate(mods):
    r = [tl[(tl.split==sp)&(tl.model==m)].rmse_cm.iloc[0] for sp in ["site_disjoint","temporal_holdout"]]
    ax.bar(x+(i-2)*wbar, r, wbar, label=mlab[m], color=mcol[m], zorder=3, edgecolor="white", linewidth=0.5)
    for xi,rv in zip(x+(i-2)*wbar, r):
        ax.text(xi, rv+0.15, f"{rv:.1f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(x); ax.set_xticklabels(["site-disjoint 5-fold","temporal holdout\n(train≤2014, test≥2015)"], fontsize=10)
ax.set_ylabel("ALT RMSE (cm)", fontsize=11); ax.set_ylim(0, 27)
ax.grid(axis="y", alpha=0.20, zorder=0)
ax.legend(fontsize=8.5, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5,1.10)); despine(ax)
gb_t = tl[(tl.split=="temporal_holdout")&(tl.model=="gbm_annual")].rmse_cm.iloc[0]
ax.annotate("temporal holdout 기준:\nGBM-annual 15.9 < GRU 19.2 < TCN 23.8",
            xy=(1.0, gb_t), xytext=(0.32, 24.0), fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f2f0e9", ec="#bebaae", lw=0.8),
            arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.0))
fig.tight_layout(); save(fig, "tlite_gate")

# ============ Fig D: 예측 vs 관측 (cell best OOF) ============
oof = pd.read_csv(f"{PROC}/alt_cell_best_oof.csv")
m = np.isfinite(oof.pred) & np.isfinite(oof.alt_cm)
yv, pv = oof.alt_cm.values[m], oof.pred.values[m]
rmse = float(np.sqrt(np.mean((yv-pv)**2))); r2 = 1-np.sum((yv-pv)**2)/np.sum((yv-yv.mean())**2)
fig, ax = plt.subplots(figsize=(5.2, 5.0))
hb = ax.hexbin(yv, pv, gridsize=42, cmap="Blues", mincnt=1, linewidths=0.1)
lim = [0, np.percentile(yv, 99.3)]
ax.plot(lim, lim, "--", color=SLATE, lw=1.3, label="1:1")
# 회귀선(수축 시각화)
b1, b0 = np.polyfit(yv, pv, 1)
xs = np.array(lim); ax.plot(xs, b0+b1*xs, "-", color=NAVY, lw=1.6, label=f"회귀 기울기 {b1:.2f}")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("관측 ALT (cm, 셀평균)", fontsize=11); ax.set_ylabel("예측 ALT (cm)", fontsize=11)
cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04); cb.set_label("셀 수", fontsize=9)
ax.text(0.04, 0.96, f"공간블록 CV (M1 기후)\nRMSE {rmse:.1f} cm\nR² {r2:.2f}\nn = {m.sum():,}", transform=ax.transAxes,
        fontsize=9.5, va="top", color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d9d6cc"))
ax.legend(fontsize=9, loc="lower right", frameon=False); despine(ax)
fig.tight_layout(); save(fig, "pred_obs_cell")
print("ALL FIGS DONE")
