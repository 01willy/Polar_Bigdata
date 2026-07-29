"""S6 시각화: source-aware multi-fidelity(A5) 비교군 성능 + b_s·σ_s 진단.

그림 3종 → outputs/figures/s6_source_aware/
 1) s6_method_bars      : LORO 게이트·in-domain RMSE 막대(seed 산점) + cov90 보정 비교
 2) s6_bias_sigma_scatter: 학습된 b_s·σ_s vs 실제 소스 bias·잔차 SD (LORO fold별)
 3) s6_loro_regions     : 지역별 LORO RMSE 분해 + sa 블록부트스트랩 ΔRMSE CI

실행: python scripts/4_visualization/s6_source_aware_figs.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar

plt = use_polar()
plt.rcParams["axes.unicode_minus"] = False   # 한글 폰트에 U+2212 글리프 없음 → ASCII '-'
PROC = C.PROCESSED
OUT = C.FIGURES / "s6_source_aware"
OUT.mkdir(parents=True, exist_ok=True)

res = pd.read_csv(PROC / "s6_source_aware_results.csv", low_memory=False)
meta = json.loads((PROC / "s6_source_aware_meta.json").read_text())
S2_GATE = meta["gates_ref"]["s2_stefan"]
S4_GATE = meta["gates_ref"]["s4_anchor"]

ORDER = ["stefan_anchor", "direct_catboost", "direct_mlp", "physfeat_catboost",
         "pool_catboost", "pool_mlp", "fixaug_catboost", "sa_z", "sa_fusion"]
LABEL = {"stefan_anchor": "Stefan 앵커", "direct_catboost": "실측-only CatBoost",
         "direct_mlp": "실측-only MLP", "physfeat_catboost": "+Stefan feat. CatBoost",
         "pool_catboost": "naive pooling CatBoost", "pool_mlp": "naive pooling MLP",
         "fixaug_catboost": "고정가중 증강 r=10", "sa_z": "source-aware z(x)",
         "sa_fusion": "source-aware 융합"}
COLOR = {m: ("#1f6f8b" if m.startswith("sa_") else
             ("#8a8fa3" if m == "stefan_anchor" else "#a9b4c2")) for m in ORDER}
COLOR["sa_fusion"] = "#144d61"
EDGE = "#39404d"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


metric = res[res.method != "sa_diag"]
loro = metric[(metric.axis == "LORO")]
gate = metric[metric.axis == "LORO_gate"].set_index("method")
ind_seed = metric[metric.axis == "spatial_block_AK"]
ind_mean = metric[metric.axis == "spatial_block_AK_seedmean"].set_index("method")
diag = res[res.method == "sa_diag"]
boot = metric[metric.axis.astype(str).str.startswith("boot_")]

# ---------------- 1. 비교군 성능 막대 + cov90 ----------------
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
# (1) LORO 게이트
ax = axes[0]
ms = [m for m in ORDER if m in gate.index]
xs = np.arange(len(ms))
vals = [gate.loc[m, "rmse_cm"] for m in ms]
ax.bar(xs, vals, color=[COLOR[m] for m in ms], edgecolor=EDGE, lw=0.6, width=0.62)
# 지역 seed별 게이트 산점(방법별 seed 게이트 = 지역 rmse 비가중평균)
for i, m in enumerate(ms):
    g = loro[loro.method == m]
    per_seed = g.groupby("seed").apply(lambda s: s.groupby("region").rmse_cm.mean().mean())
    ax.plot(np.full(len(per_seed), i), per_seed.values, "o", color="#2b2f38", ms=3.5, alpha=0.75)
ax.axhline(S2_GATE, color="#1a3a4a", lw=1.2, ls="--", label=f"S2 Stefan 게이트 {S2_GATE:.2f}")
ax.axhline(S4_GATE, color="#1a3a4a", lw=1.2, ls=":", label=f"S4 최고 앵커 {S4_GATE:.2f}")
ax.set_xticks(xs, [LABEL[m] for m in ms], rotation=38, ha="right", fontsize=8)
ax.set_ylabel("LORO 게이트 RMSE (cm)")
ax.set_ylim(0, max(vals) + 3)   # y축 0 시작: 막대 높이로 RMSE 절대크기를 정직하게 표시
ax.set_title("(a)", fontsize=10)
ax.legend(fontsize=7.5)
# (2) in-domain
ax = axes[1]
ms2 = [m for m in ORDER if m in ind_mean.index]
xs2 = np.arange(len(ms2))
vals2 = [ind_mean.loc[m, "rmse_cm"] for m in ms2]
ax.bar(xs2, vals2, color=[COLOR[m] for m in ms2], edgecolor=EDGE, lw=0.6, width=0.62)
for i, m in enumerate(ms2):
    g = ind_seed[ind_seed.method == m]
    ax.plot(np.full(len(g), i), g.rmse_cm.values, "o", color="#2b2f38", ms=3.5, alpha=0.75)
ax.set_xticks(xs2, [LABEL[m] for m in ms2], rotation=38, ha="right", fontsize=8)
ax.set_ylabel("AK 공간블록 OOF RMSE (cm)")
ax.set_ylim(0, max(vals2) + 1)   # y축 0 시작
ax.set_title("(b)", fontsize=10)
# (3) cov90
ax = axes[2]
rows_c = []
for m in ORDER:
    c_l = gate.loc[m, "coverage_90"] if m in gate.index else np.nan
    c_i = ind_mean.loc[m, "coverage_90"] if m in ind_mean.index else np.nan
    if not np.isfinite(c_i) and m in set(ind_seed.method):
        c_i = ind_seed[ind_seed.method == m].coverage_90.mean()
    if np.isfinite(c_l) or np.isfinite(c_i):
        rows_c.append((m, c_l, c_i))
xs3 = np.arange(len(rows_c))
w = 0.36
ax.bar(xs3 - w / 2, [r[1] for r in rows_c], w, color="#5b7f95", edgecolor=EDGE, lw=0.5,
       label="LORO(3지역 평균)")
ax.bar(xs3 + w / 2, [r[2] for r in rows_c], w, color="#9dbcCB".lower(), edgecolor=EDGE, lw=0.5,
       label="in-domain AK")
ax.axhline(0.90, color="#1a3a4a", lw=1.2, ls="--", label="목표 90%")
ax.set_xticks(xs3, [LABEL[r[0]] for r in rows_c], rotation=38, ha="right", fontsize=8)
ax.set_ylabel("90% 구간 실제 커버리지")
ax.set_ylim(0, 1.14)
ax.set_title("(c)", fontsize=10)
# sa_z 과대커버리지는 구간폭 발산(무정보)임을 명기
i_saz = next((i for i, r in enumerate(rows_c) if r[0] == "sa_z"), None)
if i_saz is not None:
    ax.annotate("폭 발산\n(LORO 473cm)", (i_saz, 1.01), xytext=(i_saz - 1.6, 1.06),
                fontsize=7, color="#39404d",
                arrowprops=dict(arrowstyle="-", color="#39404d", lw=0.8))
ax.legend(fontsize=7.5, loc="center right")
save(fig, "s6_method_bars")

# ---------------- 2. 학습 b_s·σ_s vs 실제 소스 bias ----------------
dl = diag[(diag.axis == "LORO") & (diag.source != "direct")]
SRC_COLOR = {"stefan": "#1f6f8b", "cci": "#8a6f9b"}
REG_MARK = {"Alaska": "o", "Lena": "s", "Canada": "^"}
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
ax = axes[0]
for (src, reg), g in dl.groupby(["source", "region"]):
    ax.plot(g.b_emp_train_cm, g.b_learned_cm, REG_MARK[reg], color=SRC_COLOR[src], ms=7,
            mec=EDGE, mew=0.5, alpha=0.85)
lim = np.nanmax(np.abs(np.concatenate([dl.b_emp_train_cm.values, dl.b_learned_cm.values]))) * 1.2 + 1
ax.plot([-lim, lim], [-lim, lim], "-", color="0.6", lw=1.0, zorder=0)
ax.axhline(0, color="0.85", lw=0.8, zorder=0); ax.axvline(0, color="0.85", lw=0.8, zorder=0)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("실제 소스 bias (train 지역, y_s - 실측, cm)")
ax.set_ylabel("학습된 b_s (cm)")
ax.set_title("(a)", fontsize=10)
ax = axes[1]
for (src, reg), g in dl.groupby(["source", "region"]):
    ax.plot(g.sig_emp_train_cm, g.sig_learned_te_cm, REG_MARK[reg], color=SRC_COLOR[src], ms=7,
            mec=EDGE, mew=0.5, alpha=0.85)
hi = np.nanmax(np.concatenate([dl.sig_emp_train_cm.values, dl.sig_learned_te_cm.values])) * 1.15 + 1
ax.plot([0, hi], [0, hi], "-", color="0.6", lw=1.0, zorder=0)
ax.set_xlim(0, hi); ax.set_ylim(0, hi)
ax.set_xlabel("실제 소스 잔차 SD (train 지역, cm)")
ax.set_ylabel("학습된 σ_s 평균 (test 셀, cm)")
ax.set_title("(b)", fontsize=10)
from matplotlib.lines import Line2D
hnd = ([Line2D([], [], color=c, marker="o", ls="", ms=7, label=f"{s} 소스")
        for s, c in SRC_COLOR.items()]
       + [Line2D([], [], color="0.4", marker=m, ls="", ms=7, label=f"{r} fold(제외 지역)")
          for r, m in REG_MARK.items()])
fig.legend(handles=hnd, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.02))
save(fig, "s6_bias_sigma_scatter")

# ---------------- 3. LORO 지역별 분해 + 부트스트랩 CI ----------------
regions = ["Alaska", "Lena", "Canada"]
fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.4))
for ax, reg in zip(axes[:3], regions):
    sub = loro[loro.region == reg]
    ms = [m for m in ORDER if m in set(sub.method)]
    vals = [sub[sub.method == m].rmse_cm.mean() for m in ms]
    ax.bar(np.arange(len(ms)), vals, color=[COLOR[m] for m in ms], edgecolor=EDGE, lw=0.6,
           width=0.62)
    for i, m in enumerate(ms):
        g = sub[sub.method == m]
        ax.plot(np.full(len(g), i), g.rmse_cm.values, "o", color="#2b2f38", ms=3, alpha=0.7)
    ax.set_xticks(np.arange(len(ms)), [LABEL[m] for m in ms], rotation=42, ha="right", fontsize=7.5)
    ax.set_ylabel("LORO RMSE (cm)")
    ax.set_title(reg, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.08)   # y축 0 시작: 방법 간 RMSE 차이 과장 방지
# ΔRMSE CI 패널 (sa_fusion vs 경쟁 상위 baseline만. 붕괴 baseline(실측-only 22~28cm 차)은
# 자명 유의라 제외하고 본문 수치로 보고)
ax = axes[3]
pairs = [("stefan_anchor", "sa_fusion"), ("pool_catboost", "sa_fusion"),
         ("fixaug_catboost", "sa_fusion")]
SHORT = {"stefan_anchor": "vs Stefan 앵커", "pool_catboost": "vs pooling CB",
         "fixaug_catboost": "vs 고정가중 r=10"}
yt, ylab = [], []
k = 0
REG_C = {"Alaska": "#5b7f95", "Lena": "#1f6f8b", "Canada": "#8a6f9b"}
bl = boot[boot.axis == "boot_LORO"]
for base, comp in pairs:
    tag = f"{comp}_vs_{base}"
    for reg in regions:
        g = bl[(bl.method == tag) & (bl.region == reg)]
        if not len(g):
            continue
        d, lo, hi = g.iloc[0][["delta_rmse", "ci_lo", "ci_hi"]]
        ax.errorbar(d, k, xerr=[[d - lo], [hi - d]], fmt="o", color=REG_C[reg], ms=5,
                    capsize=3, lw=1.4)
        k += 1
    yt.append(k - 2)
    ylab.append(SHORT[base])
    k += 1
ax.axvline(0, color="0.4", lw=1.0)
ax.set_yticks(yt, ylab, fontsize=8)
ax.set_xlabel("ΔRMSE = baseline - sa_fusion (cm)\n양수=sa 개선 · 95% 블록부트스트랩 CI")
ax.set_title("ΔRMSE", fontsize=10)
hnd = [Line2D([], [], color=c, marker="o", ls="", ms=6, label=r) for r, c in REG_C.items()]
ax.legend(handles=hnd, fontsize=8, loc="best")
fig.subplots_adjust(wspace=0.42)
save(fig, "s6_loro_regions")

print("[done] 그림 3종 저장 완료")
