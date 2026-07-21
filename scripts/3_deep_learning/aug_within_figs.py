# -*- coding: utf-8 -*-
"""알래스카 내부 라벨 증강 결과 그림(세션한도로 미생성분 보완).
- 모델별 real_only vs 최적증강 RMSE·skill 막대(SOTA 대역 표시)
- 물리유도 비율-성능 곡선(모델별)
- 최적 모델 예측 vs 실측 산점(bias 교정 진단)
출력 outputs/figures/12_aug_alaska/
"""
import os, sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for _f in ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold"]:
    try: fm.fontManager.addfont(f"/home/willy010313/.fonts/Pretendard-{_f}.otf")
    except Exception: pass
FP = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-SemiBold.otf")
FPB = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-ExtraBold.otf")
FPM = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-Medium.otf")
plt.rcParams.update({"axes.unicode_minus": False})
OUT = "outputs/figures/12_aug_alaska"; os.makedirs(OUT, exist_ok=True)
NAVY="#12324B"; BLUE="#2E86AB"; TEAL="#0E7C86"; GRAY="#9AA0A6"; GREEN="#2E8B57"; AMBER="#E08A1E"
INK="#18181B"; SLATE="#4A5158"; MUTE="#8A9096"; PAPER="#FFFFFF"; GRID="#E6E8EB"

def _style(ax, ts=12):
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("left","bottom"): ax.spines[sp].set_color("#C9CDD2")
    ax.tick_params(length=0, labelsize=ts)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontproperties(FPM)

res = pd.read_csv("data/processed/aug_within_alaska_results.csv")
sweep = pd.read_csv("data/processed/aug_within_alaska_sweep.csv")
models = ["GBM","MLP","FTTransformer","TabM"]
mlabel = {"GBM":"GBM","MLP":"MLP","FTTransformer":"FT-Transformer","TabM":"TabM"}

# ── 그림 1: 모델별 real_only vs 최적증강 RMSE (SOTA 대역) ──
fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
x = np.arange(len(models)); w = 0.38
ro = [res[res.model==m].real_only_rmse_cm.iloc[0] for m in models]
be = [res[res.model==m].best_aug_rmse_cm.iloc[0] for m in models]
ax.axhspan(14, 18, color="#EAF0F5", zorder=0)
ax.text(3.35, 17.6, "정직검증 SOTA 대역 14-18cm", fontsize=10, color=SLATE, ha="right", fontproperties=FP)
b1=ax.bar(x-w/2, ro, w, color=GRAY, label="실측만", zorder=3)
b2=ax.bar(x+w/2, be, w, color=NAVY, label="실측+물리유도 증강", zorder=3)
for xi,(r,b) in enumerate(zip(ro,be)):
    ax.text(xi-w/2, r+0.06, f"{r:.2f}", ha="center", fontsize=10.5, fontproperties=FPM, color=INK)
    ax.text(xi+w/2, b+0.06, f"{b:.2f}", ha="center", fontsize=10.5, fontproperties=FPB, color=NAVY)
ax.set_xticks(x); ax.set_xticklabels([mlabel[m] for m in models], fontsize=12)
for t in ax.get_xticklabels(): t.set_fontproperties(FP)
ax.set_ylabel("공간블록 CV RMSE (cm) · 실측 held-out", fontsize=13, fontproperties=FP)
ax.set_ylim(12, 18.6)
ax.legend(loc="upper left", frameon=False, prop=FP, fontsize=11)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax, ts=12)
ax.set_title("알래스카 내부: DL이 GBM 상회, 물리유도 증강이 소폭 개선", fontsize=15, fontproperties=FPB, color=INK, pad=12, loc="left")
fig.text(0.01,-0.01,"자료: aug_within_alaska_results.csv · 유사라벨은 훈련에만, 평가는 실측 13,606셀 held-out", fontsize=8.5, color=MUTE, fontproperties=FPM)
fig.tight_layout(); fig.savefig(f"{OUT}/aug_model_rmse.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
fig.savefig(f"{OUT}/aug_model_rmse.pdf", facecolor=PAPER, bbox_inches="tight"); plt.close(fig)
print("saved aug_model_rmse")

# ── 그림 2: 물리유도 비율-성능 곡선(const, w별) ──
fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)
cols = {"GBM":GRAY,"MLP":NAVY,"FTTransformer":BLUE,"TabM":GREEN}
for m in models:
    sm = sweep[(sweep.model==m)&(sweep.use_bh==False)&(sweep.stefan=="const")&(sweep.pseudo_w==0.3)]
    pts = [(0.0, sweep[(sweep.model==m)&(sweep.config=="real_only")].rmse_cm.iloc[0])]
    for r in [0.5,1.0,2.0]:
        row = sm[sm.p_ratio==r]
        if len(row): pts.append((r, row.rmse_cm.iloc[0]))
    pts=sorted(pts); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs, ys, "o-", color=cols[m], lw=2.0, ms=6, label=mlabel[m])
ax.set_xlabel("물리유도 유사라벨 비율 (실측 수 대비, weight 0.3)", fontsize=13, fontproperties=FP)
ax.set_ylabel("RMSE (cm)", fontsize=13, fontproperties=FP)
ax.set_xticks([0,0.5,1.0,2.0])
ax.legend(loc="upper right", frameon=False, prop=FP, fontsize=11, ncol=2)
ax.grid(color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax, ts=12)
ax.set_title("물리유도 증강 비율에 따른 성능", fontsize=15, fontproperties=FPB, color=INK, pad=12, loc="left")
fig.text(0.01,-0.01,"자료: aug_within_alaska_sweep.csv · Stefan 상수E 유사라벨", fontsize=8.5, color=MUTE, fontproperties=FPM)
fig.tight_layout(); fig.savefig(f"{OUT}/aug_ratio_curve.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
fig.savefig(f"{OUT}/aug_ratio_curve.pdf", facecolor=PAPER, bbox_inches="tight"); plt.close(fig)
print("saved aug_ratio_curve")

# ── 그림 3: 최적 모델 예측 vs 실측 (bias 교정 진단) ──
o = pd.read_csv("data/processed/aug_within_alaska_oof.csv")
y=o.alt_cm.values; a=o.yhat_real_only.values; b=o.yhat_best.values
fig, axs = plt.subplots(1, 2, figsize=(11, 5.0), dpi=200)
for ax,(p,ttl,bias) in zip(axs, [(a,"실측만", (a-y).mean()),(b,"실측+물리유도 증강",(b-y).mean())]):
    ax.hexbin(y, p, gridsize=45, cmap="Blues", mincnt=1)
    lim=[0,140]; ax.plot(lim,lim,"--",color="#444",lw=1.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("관측 ALT (cm)", fontsize=12.5, fontproperties=FP)
    ax.set_ylabel("예측 ALT (cm)", fontsize=12.5, fontproperties=FP)
    r=np.corrcoef(p,y)[0,1]
    ax.set_title(f"{ttl}\nRMSE {np.sqrt(np.mean((p-y)**2)):.2f}cm · bias {bias:+.2f} · r {r:.3f}", fontsize=12.5, fontproperties=FP, color=INK)
    _style(ax, ts=11)
fig.suptitle("증강 이득은 대부분 bias 교정(상관 r은 거의 불변)", fontsize=14.5, fontproperties=FPB, color=INK, y=1.02)
fig.text(0.01,-0.01,"자료: aug_within_alaska_oof.csv · MLP 최적증강 vs 실측만 · 평균회귀 아닌 bias 이동", fontsize=8.5, color=MUTE, fontproperties=FPM)
fig.tight_layout(); fig.savefig(f"{OUT}/aug_bias_diagnosis.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
fig.savefig(f"{OUT}/aug_bias_diagnosis.pdf", facecolor=PAPER, bbox_inches="tight"); plt.close(fig)
print("saved aug_bias_diagnosis")
