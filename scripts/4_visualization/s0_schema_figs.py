"""S0 시각화: source overlap 히트맵 + 0.5° 공간블록 GroupKFold 지도.

`docs/RESEARCH_PLAN_...` §11.5 규약: 냉색(cmcrameri), 지도 우선, PNG300+PDF.
누설통제(블록이 폴드로 통째 배정)를 눈으로 확인 가능하게 한다.

실행: python scripts/4_visualization/s0_schema_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar, CMAP
from polar.fidelity import add_group_keys, spatial_block_splits

use_polar()
PROC = C.PROCESSED
OUT = C.FIGURES / "s0_schema"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


# ---------------- 1. source overlap 히트맵 ----------------
ov = pd.read_csv(PROC / "source_overlap_matrix.csv", index_col=0)
fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(ov.values, cmap=CMAP.count, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(ov.columns))); ax.set_xticklabels(ov.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(ov.index))); ax.set_yticklabels(ov.index, fontsize=8)
for i in range(len(ov.index)):
    for j in range(len(ov.columns)):
        v = ov.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                color="white" if v < 55 else "black")
ax.set_title("자료원 쌍 셀 overlap (%)\nStefan·CCI만 clean → full source-aware 가능", fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("overlap %", fontsize=8)
save(fig, "source_overlap_heatmap")


# ---------------- 2. 0.5° 공간블록 GroupKFold 지도 ----------------
df = add_group_keys(pd.read_csv(PROC / "fidelity_base.csv", low_memory=False))
folds = spatial_block_splits(df)
fold_of = np.full(len(df), -1)
for k, (_, te) in enumerate(folds):
    fold_of[te] = k

# 알래스카 창(주 무대)과 전지역 두 패널
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
cmap6 = plt.get_cmap("cmc.batlow", len(folds))
for ax, (title, bbox) in zip(axes, [
        ("전 지역 (블록=폴드 통째 배정)", None),
        ("알래스카 창 (in-domain 주 무대)", dict(lon=(-170, -140), lat=(60, 72)))]):
    m = np.ones(len(df), dtype=bool)
    if bbox:
        m = (df.lon.between(*bbox["lon"]) & df.lat.between(*bbox["lat"])).values
    sc = ax.scatter(df.lon[m], df.lat[m], c=fold_of[m], cmap=cmap6, s=6,
                    vmin=-0.5, vmax=len(folds) - 0.5, alpha=0.8, linewidths=0)
    ax.set_xlabel("경도 (°)", fontsize=9); ax.set_ylabel("위도 (°)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(df.lat[m].mean() if m.any() else 65)))
cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, ticks=range(len(folds)))
cb.set_label("공간블록 폴드 #", fontsize=8)
fig.suptitle("0.5° 공간블록 6-fold GroupKFold — 같은 블록은 같은 폴드(site-GKF 누설 차단)", fontsize=11)
save(fig, "spatial_block_folds_map")


# ---------------- 3. 지역별 공변량 가용성 막대(트랙 결정 근거) ----------------
cp = pd.read_csv(PROC / "covariate_availability_by_region.csv")
main = cp[cp.region.isin(["ABoVE_AK", "Lena_RU", "ABoVE_CA"])]
piv = main.pivot_table(index="group", columns="region", values="valid_pct", aggfunc="first")
piv = piv.reindex(["terrain", "climate", "soil", "cci", "insar", "polsar"])
fig, ax = plt.subplots(figsize=(7, 4))
piv.plot(kind="bar", ax=ax, colormap="cmc.batlow", width=0.78)
ax.axhline(50, color="0.5", lw=0.8, ls="--")
ax.set_ylabel("유효율 (%)", fontsize=9); ax.set_xlabel("공변량 그룹", fontsize=9)
ax.set_title("지역×공변량 가용성 — SAR(InSAR/PolSAR)는 알래스카에만 존재\n→ 정확도=AK in-domain, 전이=공유피처+물리", fontsize=10)
ax.legend(title="", fontsize=8); ax.tick_params(axis="x", rotation=0)
save(fig, "covariate_availability_bars")

print("[done] S0 시각화 3종 완료")
