"""S4 시각화: shrinkage λ 곡선 (Stefan 앵커 + 저용량 잔차) — LORO 게이트 대비.

핵심 그림: 지역별(Alaska·Lena·Canada) + 게이트(비가중평균) RMSE = f(λ).
λ=0 = Stefan-only 앵커. S2 게이트(22.24cm) 수평선. 자동선택 λ*(inner CV) 마커.
실행: python scripts/4_visualization/s4_residual_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar

use_polar()
plt.rcParams["pdf.fonttype"] = 42   # PDF 텍스트를 TrueType로 임베딩(Type 3 금지)
PROC = C.PROCESSED
OUT = C.FIGURES / "s4_residual"
OUT.mkdir(parents=True, exist_ok=True)


ONLY = sys.argv[1] if len(sys.argv) > 1 else None  # 인자 주면 해당 그림만 저장


def save(fig, name):
    if ONLY is not None and name != ONLY:
        plt.close(fig)
        return
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


import json
res = pd.read_csv(PROC / "s4_residual_results.csv")
meta = json.loads((PROC / "s4_residual_meta.json").read_text())
GATE = meta["s2_gate_cm"]

M_COLOR = {"ridge": "#8296a8", "catboost_lo": "#4a7c8c", "catboost": "#2f4b6e"}
# 라벨은 본문 방법론 절의 표현(능형 회귀·저용량 부스팅)과 일치시킨다
M_LABEL = {"ridge": "능형 회귀 잔차", "catboost_lo": "CatBoost 잔차(저용량)", "catboost": "CatBoost 잔차(표준)"}
# 계열 식별은 색에만 의존하지 않도록 모델별 표식 모양을 분리(흑백·색약 대비)
M_MARKER = {"ridge": "o", "catboost_lo": "^", "catboost": "D"}
# 값이 거의 겹치는 구간에서 뒤에 그리는 계열이 앞 계열을 가리지 않도록 표식 크기를 단계적으로 축소
M_MS = {"ridge": 5.2, "catboost_lo": 4.3, "catboost": 3.4}
PROTO_LABEL = {"indomain": "지역 내", "transfer": "지역 간 전이"}
FS_STYLE = {"shift14": "-", "shared25": "--"}
FS_LABEL = {"shift14": "지형+기후 14", "shared25": "+토양·CCI 25"}

# ---------------- λ 곡선: 지역 3 + 게이트 ----------------
loro = res[(res.cv == "LORO") & (res.seed != "SUMMARY")].copy()
gate = res[res.cv == "LORO_gate"].copy()
auto = res[res.cv == "LORO_autolam"].copy()
regions = ["Alaska", "Lena", "Canada"]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
for ax, reg in zip(axes[:3], regions):
    sub = loro[loro.region == reg]
    for model in M_COLOR:
        for fs in FS_STYLE:
            g = sub[(sub.model == model) & (sub.featset == fs)].groupby("lam").rmse_cm.mean()
            ax.plot(g.index, g.values, FS_STYLE[fs], color=M_COLOR[model], lw=1.8, ms=5,
                    marker="o", label=f"{M_LABEL[model]} · {FS_LABEL[fs]}")
    a = auto[auto.region == reg]
    for _, r in a.iterrows():
        ax.plot(r.lam, r.rmse_cm, "*", color=M_COLOR[r.model], ms=13, mec="k", mew=0.5, zorder=5)
    stefan_only = sub[sub.lam == 0].rmse_cm.mean()
    ax.axhline(stefan_only, color="0.35", lw=1.0, ls=":", label="Stefan-only(λ=0)")
    ax.set_xlabel("shrinkage λ", fontsize=9)
    ax.set_ylabel("LORO RMSE (cm)", fontsize=9)
    ax.set_title(f"{reg} 전이", fontsize=10)
ax = axes[3]
for model in M_COLOR:
    for fs in FS_STYLE:
        g = gate[(gate.model == model) & (gate.featset == fs)].sort_values("lam")
        ax.plot(g.lam, g.rmse_cm, FS_STYLE[fs], color=M_COLOR[model], lw=1.8, ms=5, marker="o")
ax.axhline(GATE, color="#3a4a5a", lw=1.4, ls="--", label=f"S2 게이트 {GATE:.2f}cm")
ag = res[res.cv == "LORO_autolam_gate"]
for _, r in ag.iterrows():
    ax.plot(r.lam, r.rmse_cm, "*", color=M_COLOR[r.model], ms=13, mec="k", mew=0.5, zorder=5)
ax.set_xlabel("shrinkage λ", fontsize=9)
ax.set_ylabel("비가중평균 RMSE (cm)", fontsize=9)
ax.set_title("게이트(Alaska·Lena·Canada 비가중평균)", fontsize=10)
ax.legend(fontsize=8, loc="best")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.02))
save(fig, "s4_lambda_curves")

# ---------------- 부트스트랩 CI: 지역별 ΔRMSE(λ) ----------------
boot = res[res.cv == "LORO_boot"].copy()
if len(boot):
    fig, axes = plt.subplots(1, len(regions), figsize=(4.6 * len(regions), 3.8), sharey=False)
    for ax, reg in zip(axes, regions):
        b = boot[boot.region == reg].sort_values("lam")
        ax.errorbar(b.lam, b.delta_rmse,
                    yerr=[b.delta_rmse - b.ci_lo, b.ci_hi - b.delta_rmse],
                    fmt="o-", color="#1f6f8b", lw=1.6, ms=5, capsize=3)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.set_xlabel("shrinkage λ", fontsize=9)
        ax.set_ylabel("ΔRMSE vs Stefan-only (cm, 양수=개선)", fontsize=9)
        ax.set_title(f"{reg}", fontsize=10)
    save(fig, "s4_bootstrap_ci")

# ---------------- 지역 내 vs 지역 간 전이 대비 ----------------
ind = res[res.cv == "spatial_block_AK"].copy()
# 전이 곡선(3개 지역 비가중평균)의 지역 간 표준오차: 지역별 시드평균 RMSE의 SD/sqrt(3)
loro_sh = loro[loro.featset == "shift14"].copy()
reg_mean = loro_sh.groupby(["model", "lam", "region"]).rmse_cm.mean().unstack()
SE_REG = (reg_mean.std(axis=1, ddof=1) / np.sqrt(reg_mean.shape[1]))

# 종횡비: 보고서 2열 배치(폭 0.5\textwidth) 기준. 짝 그림과 높이 정합
fig, ax = plt.subplots(figsize=(4.9, 4.2))
for model in M_COLOR:
    c = M_COLOR[model]
    mk, ms = M_MARKER[model], M_MS[model]
    g = ind[(ind.model == model) & (ind.featset == "shift14")].groupby("lam").rmse_cm.mean()
    ax.plot(g.index, g.values, ls="-", marker=mk, color=c, lw=1.5, ms=ms, mew=0.0,
            label=f"{M_LABEL[model]} · {PROTO_LABEL['indomain']}", zorder=3)
    gg = gate[(gate.model == model) & (gate.featset == "shift14")].sort_values("lam")
    se = SE_REG.loc[model].reindex(gg.lam.values).values
    ax.errorbar(gg.lam, gg.rmse_cm, yerr=se, fmt="none", ecolor=c, elinewidth=0.8,
                capsize=2.0, capthick=0.7, alpha=0.55, zorder=2)
    ax.plot(gg.lam, gg.rmse_cm, ls=(0, (5, 2.5)), marker=mk, color=c, lw=1.2, ms=ms + 0.6,
            mfc="white", mec=c, mew=1.0, zorder=3,
            label=f"{M_LABEL[model]} · {PROTO_LABEL['transfer']}")
ax.set_xlabel("잔차 가중 λ", fontsize=10.5)
ax.set_ylabel("RMSE (cm)", fontsize=10.5)
ax.set_xlim(-0.03, 1.03)
ax.tick_params(labelsize=9.5, length=3, color="#888888", labelcolor="#333333")
ax.grid(alpha=0.25, lw=0.5, color="#9aa4ad")
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#888888")
    ax.spines[s].set_linewidth(0.7)
leg = ax.legend(fontsize=9, frameon=False, loc="upper left", labelspacing=0.38,
                handlelength=2.4, handletextpad=0.6, borderaxespad=0.2)
for t in leg.get_texts():
    t.set_color("#333333")

# 각주: 오차막대의 정의(표준편차가 아닌 지역 간 표준오차)와 반복 수. 캡션 없이 그림만으로 확인된다.
N_REG = int(reg_mean.shape[1])
REG_NAMES = "·".join({"Alaska": "알래스카", "Lena": "레나", "Canada": "캐나다"}.get(r, r)
                     for r in regions)
_seed_n = loro_sh.groupby(["model", "lam", "region"]).seed.nunique()
N_SEED = int(_seed_n.max())
DET = [M_LABEL[m] for m in M_COLOR if int(_seed_n.loc[m].max()) == 1]
note2 = f"지역 = {REG_NAMES}. 지역별 RMSE는 난수 {N_SEED}회 평균"
note2 += f", {'·'.join(DET)}는 결정적(1회)." if DET else "."
fig.text(0.0, -0.030,
         f"오차막대 = 지역 간 전이의 ±1 표준오차({N_REG}개 지역 RMSE의 표준편차/√{N_REG})."
         " 지역 내 곡선은 미표시.",
         fontsize=7.2, color="#666666", ha="left", va="bottom")
fig.text(0.0, -0.075, note2, fontsize=7.2, color="#666666", ha="left", va="bottom")
save(fig, "s4_indomain_vs_transfer")

print("[done] S4 시각화 완료")
