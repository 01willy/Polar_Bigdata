"""S3 시각화: 유사라벨 증강비율 반응곡선(물리 라벨 vs 대조 라벨) + 물리 라벨의 순가치.

핵심 그림: 개선폭 ΔRMSE=f(r) 곡선에 Stefan·Kudryavtsev·대조(상수 라벨)를 병기한다.
물리 라벨과 대조 라벨의 간격이 물리 정보의 순가치다. 냉색 규약.
실행: python scripts/4_visualization/s3_aug_curve_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar

use_polar()
# PDF에 TrueType(Type42) 서브셋 임베딩 — Type3 비트맵 글리프(축소 인쇄 시 한글 흐림) 방지
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.titleweight"] = "normal"
PROC = C.PROCESSED
OUT = C.FIGURES / "s3_aug"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


def paperize(ax, grid_axis="y", labelsize=8.5):
    """논문용 축 마무리: 좌·하 spine만, 옅은 단축 격자, 규칙적 눈금 크기."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.7)
        ax.spines[s].set_color("#7a7a7a")
    ax.grid(False)
    ax.grid(True, axis=grid_axis, color="#aab3bd", lw=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=labelsize, length=2.5, width=0.7,
                   color="#8a8a8a", labelcolor="#333333")


res = pd.read_csv(PROC / "s3_aug_curve_results.csv")
ftt_path = PROC / "s3_aug_curve_results_ftt.csv"  # S3 보강(FT-T) 결과 병합
if ftt_path.exists():
    res = pd.concat([res, pd.read_csv(ftt_path)], ignore_index=True)
seed_rows = res[res.seed != "SUMMARY"].copy()
seed_rows["r"] = pd.to_numeric(seed_rows["r"])
# 발산 모델 제외(전이 covariate shift 하 신경망 발산; base RMSE>100 = 비물리)
div = seed_rows[seed_rows.rmse_cm > 100][["target", "model"]].drop_duplicates()
for _, row in div.iterrows():
    seed_rows = seed_rows[~((seed_rows.target == row.target) & (seed_rows.model == row.model))]
if len(div):
    print(f"[warn] 발산 제외(전이 발산): {div.to_dict('records')}")
targets = sorted(seed_rows.target.unique())
models = sorted(seed_rows.model.unique())
PHYS_ORDER = ["stefan", "kudryavtsev", "placebo"]
PHYS_LABEL = {"stefan": "Stefan 라벨", "kudryavtsev": "Kudryavtsev 라벨", "placebo": "대조(상수 라벨)"}
# 냉색 저채도 규약: 물리 두 선은 진청·청록, 대조는 중립 연청회(붉은/주황 계열 배제)
PHYS_COLOR = {"stefan": "#2f4b6e", "kudryavtsev": "#4a7c8c", "placebo": "#a7b3c0"}
MODEL_LABEL = {"catboost": "CatBoost", "ftt": "FT-Transformer", "mlp": "MLP"}
MODEL_LS = {"catboost": "-", "ftt": "--", "mlp": ":"}
MODEL_MARK = {"catboost": "o", "ftt": "s", "mlp": "^"}
# 순가치 그림용: 모델=색(냉색), 전이 대상=선/마커 스타일 (색만으로 정보 인코딩 금지 규약)
MODEL_COLOR = {"catboost": "#2f4b6e", "ftt": "#4a7c8c", "mlp": "#8296a8"}
TARGET_STYLE = {"Canada": dict(ls="-", marker="o"), "Lena": dict(ls="--", marker="s")}
# 지역명은 본문 표기와 맞춘다. 이 실험 조건은 "대상 지역 공변량은 있고 실측 라벨만 없음"
# 이므로 패널 제목에 '전이'를 쓰지 않는다(본문 §지역 간 전이의 정보 전무 조건과 구분).
REGION_KO = {"Canada": "캐나다", "Lena": "레나델타", "Alaska": "알래스카"}
# 증강비율 격자값 전부를 주 눈금으로 고정한다. symlog 기본 로케이터에 맡기면 0.5·2가
# 보조 눈금으로 분류되어 라벨 색·크기가 나머지와 달라진다.
XTICKS = [0, 0.25, 0.5, 1, 2, 5, 10]
XLABEL = "증강 비율 r (유사라벨/실측)\n가로축 대칭로그(0.25 이하 선형)"


def style_xaxis(ax, show_labels=True):
    """symlog 가로축 공통 마무리: 격자값을 모두 주 눈금으로 두고 축 척도를 명시한다."""
    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xlim(-0.05, 12)
    ax.set_xticks(XTICKS)
    ax.set_xticks([], minor=True)                       # 보조 눈금 제거
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())  # 보조 라벨 차단
    if show_labels:
        # 라벨 색·크기는 paperize()의 주 눈금 설정을 그대로 따른다(전 눈금 동일 스타일).
        ax.set_xticklabels([str(t) for t in XTICKS])
        ax.set_xlabel(XLABEL, fontsize=8.5, linespacing=1.35)
    else:
        ax.set_xticklabels([])

# ---------------- 반응곡선: 전이 대상별 r vs ΔRMSE(개선) ----------------
# 좁은 단(0.32\textwidth) 배치 대비: 2행 1열. 색=유사라벨 종류, 선종류=모델.
# 두 패널의 y축을 공유한다. 척도가 다르면 Canada(절대 변화폭 약 2배)와 Lena의 기울기가
# 비슷한 크기로 보여 개선/악화 규모를 오독하게 된다.
fig, axes = plt.subplots(len(targets), 1, figsize=(2.95, 2.85), squeeze=False,
                         sharey=True, layout="constrained")
for ti, target in enumerate(targets):
    ax = axes[ti][0]
    for phys in PHYS_ORDER:
        for model in models:
            sub = seed_rows[(seed_rows.model == model) & (seed_rows.target == target)]
            if not len(sub):
                continue
            base = sub[sub.r == 0].groupby("seed").rmse_cm.mean().mean()  # r=0 기준 RMSE
            s = sub[sub.phys == phys]
            if not len(s):
                continue
            g = s.groupby("r").rmse_cm.agg(["mean", "std"])
            delta = base - g["mean"]           # 양수=개선
            ax.plot(g.index, delta, color=PHYS_COLOR[phys], ls=MODEL_LS[model],
                    marker=MODEL_MARK[model], lw=1.1, ms=2.4, mew=0)
            ax.fill_between(g.index, delta - g["std"], delta + g["std"],
                            color=PHYS_COLOR[phys], alpha=0.12, lw=0)
    ax.axhline(0, color="#9aa2ab", lw=0.7, ls="--", zorder=1)
    if ti == 0:
        ax.text(0.60, 0.74, "기준 0 = 증강 없음(r = 0)", transform=ax.transAxes,
                ha="center", va="top", fontsize=6.8, color="#6a7280",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    paperize(ax, labelsize=8)
    style_xaxis(ax, show_labels=(ti == len(targets) - 1))
    # 이 그림이 보고서에서 하위그림 (b)로 들어가므로 내부에 다시 (a)·(b)를 쓰지 않는다.
    ax.set_title(REGION_KO.get(target, target),
                 fontsize=9, loc="left", color="#222222", pad=2.5)
fig.supylabel("ΔRMSE (cm, 양수=개선)", fontsize=9)
handles = [Line2D([], [], color=PHYS_COLOR[p], lw=1.4, label=PHYS_LABEL[p]) for p in PHYS_ORDER]
handles += [Line2D([], [], color="#7a8593", lw=1.1, ls=MODEL_LS[m], marker=MODEL_MARK[m],
                   ms=2.6, mew=0, label=MODEL_LABEL.get(m, m)) for m in models]
# 음영대 의미(난수 시드 간 ±1σ)를 범례에 명시 — 신뢰구간과의 혼동 방지
handles += [Patch(facecolor="#8a8f99", alpha=0.25, lw=0, label="평균 ± 1σ (난수 3회)")]
fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=7.5,
           frameon=False, handlelength=1.6, columnspacing=0.9, handletextpad=0.4,
           labelspacing=0.35)
save(fig, "aug_response_curves")

# ---------------- 물리 라벨의 순가치: 물리 라벨 − 대조 라벨 (같은 r에서) ----------------
fig, ax = plt.subplots(figsize=(2.95, 2.85), layout="constrained")
_net_ends = {}
for model in models:
    for target in targets:
        sub = seed_rows[(seed_rows.model == model) & (seed_rows.target == target)]
        # 순가치 = (기준−Stefan RMSE) − (기준−대조 RMSE) = 대조 RMSE − Stefan RMSE.
        # 기준항이 상쇄되므로 같은 시드끼리 짝지어 계산할 수 있고, 시드 간 산포를 전파할 수 있다.
        p_st = sub[sub.phys == "stefan"].pivot_table(index="r", columns="seed",
                                                     values="rmse_cm", aggfunc="mean")
        p_pl = sub[sub.phys == "placebo"].pivot_table(index="r", columns="seed",
                                                      values="rmse_cm", aggfunc="mean")
        seeds = p_st.columns.intersection(p_pl.columns)
        if not len(seeds):
            continue
        per_seed = (p_pl[seeds] - p_st[seeds]).sort_index()
        net = per_seed.mean(axis=1)
        sd = per_seed.std(axis=1, ddof=1).fillna(0.0)
        stl = TARGET_STYLE.get(target, dict(ls="-", marker="o"))
        col = MODEL_COLOR.get(model, "#2f4b6e")
        ax.fill_between(net.index, net - sd, net + sd, color=col, alpha=0.14, lw=0, zorder=2)
        ax.plot(net.index, net.values, lw=1.1, ms=2.6, mew=0, zorder=3,
                color=col, ls=stl["ls"], marker=stl["marker"],
                label=f"{MODEL_LABEL.get(model, model)} · {REGION_KO.get(target, target)}")
        if model == "catboost":
            _net_ends.setdefault(REGION_KO.get(target, target), []).append(
                (float(net.index[-1]), float(net.values[-1])))
ax.axhline(0, color="#9aa2ab", lw=0.7, ls="--", zorder=1)
if "캐나다" in _net_ends:
    _xe = _net_ends["캐나다"][0][0]
    _ye = sum(v for _, v in _net_ends["캐나다"]) / len(_net_ends["캐나다"])
    ax.annotate(f"캐나다 순가치 +{_ye:.1f} cm (CatBoost)", xy=(_xe, _ye), xytext=(-4, -32),
                textcoords="offset points", ha="right", fontsize=7.2, color="#1f3a52",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0),
                arrowprops=dict(arrowstyle="-", color="#9aa2ab", lw=0.6))
style_xaxis(ax)                             # r=0(기준점) 포함, 보조 눈금 라벨 표기
# 수식 마이너스는 mathtext($-$)로 렌더 — NanumGothic에 U+2212 글리프 부재
ax.set_ylabel("Stefan $-$ 대조 라벨 개선폭 (cm)", fontsize=9)
paperize(ax, labelsize=8)
handles, labels = ax.get_legend_handles_labels()
# 음영대 의미(난수 시드 간 ±1σ)를 반응곡선 그림과 동일하게 명시
handles += [Patch(facecolor="#8a8f99", alpha=0.28, lw=0)]
labels += ["평균 ± 1σ (난수 3회)"]
ax.legend(handles, labels, fontsize=7.5, loc="center right", frameon=False,
          handlelength=1.6, handletextpad=0.4, labelspacing=0.35)
save(fig, "physics_net_value")

print("[done] S3 시각화 완료")
