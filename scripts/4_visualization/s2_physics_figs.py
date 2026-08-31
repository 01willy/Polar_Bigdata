"""S2 시각화: 물리 5종 지도(실제 배경) + phys_std 스프레드 + 물리별 LORO + physics feature 효과.

`RESEARCH_PLAN_...` §11.5. "어느 물리가 어느 지역서 강한가" 지도가 핵심.
실행: python scripts/4_visualization/s2_physics_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from cmcrameri import cm as cmc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar, CMAP, tnorm
from polar.geomap import make_ax, scatter_map, _proj, ALASKA, PANARCTIC

use_polar()
# PDF에 TrueType(Type42) 서브셋 임베딩 — Type3 비트맵 글리프(축소 인쇄 시 한글 흐림) 방지.
# s3_aug_curve_figs.py와 동일 설정으로 같은 절 그림들의 글꼴 임베딩 방식을 통일한다.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
PROC = C.PROCESSED
OUT = C.FIGURES / "s2_physics"
OUT.mkdir(parents=True, exist_ok=True)


ONLY = set(sys.argv[1:])   # 인자 지정 시 해당 그림만 저장(부분 재렌더)


def save(fig, name):
    if ONLY and name not in ONLY:
        plt.close(fig)
        return
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


oof = pd.read_csv(PROC / "s2_physics_oof.csv")
res = pd.read_csv(PROC / "s2_physics_results.csv")
MEMBERS = ["p1_stefan", "p2_edaphic", "p3_ttop", "p4_kudryavtsev", "p5_lambda"]
LABELS = {"p1_stefan": "Stefan", "p2_edaphic": "edaphic(토양물성)", "p3_ttop": "edaphic + TTOP 판정",
          "p4_kudryavtsev": "Kudryavtsev", "p5_lambda": "열전도도 보정",
          "phys_mean": "5종 평균"}
ak = oof[oof.region.isin(["ABoVE_AK", "United States (Alaska)"])]
proj = _proj(ALASKA)


def tune_gridlines(ax, size=10, left_labels=True):
    """지도 경위도 틱 라벨 확대 + 눈금 밀도 축소 + 회전 해제(축소 인쇄 판독성).

    cartopy 0.23: Gridliner는 ax.artists에 있다(ax._gridliners 제거됨).
    """
    from cartopy.mpl.gridliner import Gridliner
    for gl in [a for a in ax.artists if isinstance(a, Gridliner)]:
        gl.xlabel_style = {"size": size, "color": "0.25"}
        gl.ylabel_style = {"size": size, "color": "0.25"}
        gl.xlocator = mticker.FixedLocator([-165, -155, -145])
        gl.ylocator = mticker.FixedLocator([60, 64, 68, 72])
        gl.rotate_labels = False
        gl.left_labels = left_labels


# ---------------- 1. 물리 5종 ALT 지도 (알래스카, 실제 배경) ----------------
# 2행 3열 배치: 5패널 수평 배치 시 인쇄 축소로 경위도 틱이 판독 불가해지는 문제 회피.
fig = plt.figure(figsize=(10.8, 6.6))
for i, m in enumerate(MEMBERS):
    ax = fig.add_subplot(2, 3, i + 1, projection=proj)
    make_ax(ALASKA, ax=ax, fig=fig,
            title=f"({'abcde'[i]}) {LABELS[m]}\n(median {np.nanmedian(ak[m]):.0f} cm)",
            title_fontsize=11)
    sc = scatter_map(ax, ak.lon, ak.lat, ak[m], cmap=CMAP.alt, vmin=20, vmax=130, s=6)
    tune_gridlines(ax)
cax = fig.add_axes([0.70, 0.24, 0.24, 0.03])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.set_label("ALT (cm)", fontsize=11)
cb.ax.tick_params(labelsize=10)
save(fig, "physics_members_maps_alaska")


# ---------------- 2. phys_std 불확실성 스프레드 지도 ----------------
# phys_std 컬러맵: acton(자주·warm-magenta 경계) 대신 회색 순차형(냉색 규약 엄격 적용).
STD_CMAP = LinearSegmentedColormap.from_list(
    "phys_std_gray", cmc.grayC_r(np.linspace(0.15, 0.88, 256)))
fig = plt.figure(figsize=(11, 4.6))
for i, (title, col, cmap, kw) in enumerate([
        ("(a) 물리 앙상블 평균 phys_mean", "phys_mean", CMAP.alt, dict(vmin=20, vmax=130)),
        ("(b) 물리 간 불일치 phys_std", "phys_std", STD_CMAP, dict(vmin=0, vmax=40))]):
    ax = fig.add_subplot(1, 2, i + 1, projection=proj)
    make_ax(ALASKA, ax=ax, fig=fig, title=title)
    sc = scatter_map(ax, ak.lon, ak.lat, ak[col], cmap=cmap, s=7, **kw)
    tune_gridlines(ax, left_labels=(i == 0))
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(col + " (cm)", fontsize=11); cb.ax.tick_params(labelsize=10)
fig.subplots_adjust(wspace=0.18)
save(fig, "physics_ensemble_spread")


# ---------------- 3. 동토 마스크(TTOP<0) 지도 ----------------
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(6.5, 5.4))
ax = fig.add_subplot(111, projection=proj)
make_ax(ALASKA, ax=ax, fig=fig)
_mnorm = tnorm(0, 1, 0.5)
sc = scatter_map(ax, ak.lon, ak.lat, ak.perm_mask, cmap=CMAP.diff, norm=_mnorm, s=8)
tune_gridlines(ax)
handles = [
    Line2D([], [], marker="o", ls="", ms=7, color=CMAP.diff(float(_mnorm(1.0))), label="영구동토 (TTOP<0°C)"),
    Line2D([], [], marker="o", ls="", ms=7, color=CMAP.diff(float(_mnorm(0.0))), label="탈릭 (TTOP≥0°C)"),
]
ax.legend(handles=handles, loc="lower left", fontsize=10, framealpha=0.9)
save(fig, "ttop_permafrost_mask")


# ---------------- 4. 물리식별 지역 간 전이 오차 ----------------
# 좁은 단(0.32\textwidth) 배치 대비: 가로 막대 + 좌측 항목명(회전 라벨 제거).
REGIONS = ["Alaska", "Canada", "Lena"]
REG_COLOR = {"Alaska": "#2f4b6e", "Canada": "#7d97ab", "Lena": "#c3ccd5"}
REG_KO = {"Alaska": "알래스카", "Canada": "캐나다", "Lena": "레나델타"}
pa = res[(res.part == "A_physics") & (res.cv == "LORO")].copy()
piv = (pa.pivot_table(index="model", columns="region", values="rmse_cm", aggfunc="mean")
         .reindex(MEMBERS + ["phys_mean"])[REGIONS])
piv = piv.iloc[::-1]                       # 가로막대: 첫 항목(Stefan)을 위쪽에
fig, ax = plt.subplots(figsize=(2.95, 2.85), layout="constrained")
# 개별 물리식 5종과 그 평균은 성격이 다르다. 평균 막대를 아래로 분리(간격 + 얇은 구분선)해
# 평균이 6번째 독립 모델로 읽히지 않도록 한다.
GAP = 0.5
is_mean = np.array([m == "phys_mean" for m in piv.index])
y = np.arange(len(piv), dtype=float)
y[~is_mean] += GAP
sep = (y[is_mean].max() + y[~is_mean].min()) / 2
h = 0.26
for k, reg in enumerate(REGIONS):
    ax.barh(y + (1 - k) * h, piv[reg].to_numpy(dtype=float), height=h,
            color=REG_COLOR[reg], edgecolor="none", label=REG_KO.get(reg, reg), zorder=3)
ax.axhline(sep, color="#aab3bd", lw=0.6, ls=(0, (3, 3)), zorder=2)
ax.set_yticks(y)
ax.set_yticklabels([LABELS.get(m, m) for m in piv.index], fontsize=8.5)
for tl, m in zip(ax.get_yticklabels(), is_mean):
    if m:
        tl.set_color("#666666")     # 평균 항목은 옅은 회색 라벨로 개별 모델과 구분
ax.set_ylim(y.min() - 0.55, y.max() + 0.45)
ax.set_xlim(0, 100)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("지역 간 전이 RMSE (cm)", fontsize=9.5)
paperize(ax, grid_axis="x")
ax.legend(fontsize=8.5, ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
          frameon=False, columnspacing=1.0, handletextpad=0.45, handlelength=1.1)
save(fig, "physics_loro_bars")


# ---------------- 5. physics feature 효과 (Part B) ----------------
# 값이 14.5~16.8 cm 구간에 몰려 0-기준 막대로는 차이(0.1~0.3 cm)가 식별 불가.
# dot plot(덤벨) + 실제 값·차이 수치 주석으로 교체. y축은 데이터 구간으로 한정.
pb = res[(res.part == "B_feature") & (res.cv == "spatial_block_AK")].copy()
pb["base_model"] = pb.model.str.replace("_BASE", "", regex=False).str.replace("_+physics", "", regex=False)
pb["tag"] = np.where(pb.model.str.contains("physics"), "+physics", "BASE")
g = pb.groupby(["base_model", "tag"]).rmse_cm.mean().unstack()
C_BASE, C_PHYS = "#2f5d8a", "#8fa8c0"
fig, ax = plt.subplots(figsize=(7, 4.2))
dx = 0.14
for i, m in enumerate(g.index):
    b, p = g.loc[m, "BASE"], g.loc[m, "+physics"]
    ax.plot([i - dx, i + dx], [b, p], color="0.55", lw=1.3, zorder=2)
    ax.plot(i - dx, b, "o", ms=10, color=C_BASE, zorder=3,
            label="BASE" if i == 0 else None)
    ax.plot(i + dx, p, "s", ms=9, color=C_PHYS, mec="0.3", mew=0.6, zorder=3,
            label="+physics" if i == 0 else None)
    ax.annotate(f"{b:.2f}", (i - dx, b), xytext=(-7, -4), textcoords="offset points",
                ha="right", va="center", fontsize=10.5, color=C_BASE)
    ax.annotate(f"{p:.2f}", (i + dx, p), xytext=(7, -4), textcoords="offset points",
                ha="left", va="center", fontsize=10.5, color="#41617f")
    ax.annotate(f"Δ {p - b:+.2f} cm", (i, max(b, p)), xytext=(0, 16),
                textcoords="offset points", ha="center", fontsize=11, color="0.2")
ax.axhline(14, color="0.4", ls="--", lw=1.2, label="대표성 하한 ~14 cm")
ax.set_xticks(range(len(g.index)))
ax.set_xticklabels(g.index, fontsize=12)
ax.set_xlim(-0.6, len(g.index) - 0.4)
ax.set_ylim(13.6, 17.6)
ax.set_ylabel("in-domain RMSE (cm)", fontsize=12)
ax.tick_params(axis="y", labelsize=11)
ax.grid(axis="y"); ax.grid(axis="x", visible=False)
ax.legend(fontsize=10.5, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01),
          frameon=False, columnspacing=1.2, handletextpad=0.5)
save(fig, "physics_feature_effect")

print("[done] S2 시각화 5종 완료")
