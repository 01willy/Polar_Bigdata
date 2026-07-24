"""S2 시각화: 물리 5종 지도(실제 배경) + phys_std 스프레드 + 물리별 LORO + physics feature 효과.

`RESEARCH_PLAN_...` §11.5. "어느 물리가 어느 지역서 강한가" 지도가 핵심.
실행: python scripts/4_visualization/s2_physics_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar, CMAP, tnorm
from polar.geomap import make_ax, scatter_map, _proj, ALASKA, PANARCTIC

use_polar()
PROC = C.PROCESSED
OUT = C.FIGURES / "s2_physics"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


oof = pd.read_csv(PROC / "s2_physics_oof.csv")
res = pd.read_csv(PROC / "s2_physics_results.csv")
MEMBERS = ["p1_stefan", "p2_edaphic", "p3_ttop", "p4_kudryavtsev", "p5_lambda"]
LABELS = {"p1_stefan": "Stefan 기본", "p2_edaphic": "Stefan edaphic", "p3_ttop": "TTOP-Stefan",
          "p4_kudryavtsev": "Kudryavtsev", "p5_lambda": "λ보정(불포화)"}
ak = oof[oof.region.isin(["ABoVE_AK", "United States (Alaska)"])]
proj = _proj(ALASKA)


# ---------------- 1. 물리 5종 ALT 지도 (알래스카, 실제 배경) ----------------
fig = plt.figure(figsize=(15, 3.6))
for i, m in enumerate(MEMBERS):
    ax = fig.add_subplot(1, 5, i + 1, projection=proj)
    make_ax(ALASKA, ax=ax, fig=fig, title=f"{LABELS[m]}\n(중앙 {np.nanmedian(ak[m]):.0f}cm)")
    sc = scatter_map(ax, ak.lon, ak.lat, ak[m], cmap=CMAP.alt, vmin=20, vmax=130, s=6)
cb = fig.colorbar(sc, ax=fig.axes, fraction=0.011, pad=0.02); cb.set_label("ALT (cm)", fontsize=9)
fig.suptitle("S2 물리식 5종 ALT 예측 (알래스카) — 기본 Stefan이 관측대와 정합, 정교화는 상방편향", fontsize=12, y=1.04)
save(fig, "physics_members_maps_alaska")


# ---------------- 2. phys_std 불확실성 스프레드 지도 ----------------
fig = plt.figure(figsize=(11, 4.6))
for i, (title, col, cmap, kw) in enumerate([
        ("물리 앙상블 평균 phys_mean", "phys_mean", CMAP.alt, dict(vmin=20, vmax=130)),
        ("물리 불일치 phys_std (상대 불확실성)", "phys_std", CMAP.err, dict(vmin=0, vmax=40))]):
    ax = fig.add_subplot(1, 2, i + 1, projection=proj)
    make_ax(ALASKA, ax=ax, fig=fig, title=title)
    sc = scatter_map(ax, ak.lon, ak.lat, ak[col], cmap=cmap, s=7, **kw)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label(col + " (cm)", fontsize=8)
fig.suptitle("S2 물리 앙상블 평균과 불일치(불확실성 프록시)", fontsize=12)
save(fig, "physics_ensemble_spread")


# ---------------- 3. 동토 마스크(TTOP<0) 지도 ----------------
fig = plt.figure(figsize=(6.5, 5.4))
ax = fig.add_subplot(111, projection=proj)
make_ax(ALASKA, ax=ax, fig=fig, title="TTOP 동토 존재 마스크 (파랑=영구동토, 갈=탈릭)")
sc = scatter_map(ax, ak.lon, ak.lat, ak.perm_mask, cmap=CMAP.diff, norm=tnorm(0, 1, 0.5), s=8)
save(fig, "ttop_permafrost_mask")


# ---------------- 4. 물리별 LORO 막대 (어느 물리가 어느 지역서 강한가) ----------------
pa = res[(res.part == "A_physics") & (res.cv == "LORO")].copy()
piv = pa.pivot_table(index="model", columns="region", values="rmse_cm", aggfunc="mean").reindex(MEMBERS + ["phys_mean"])
fig, ax = plt.subplots(figsize=(9, 4.4))
piv.plot(kind="bar", ax=ax, colormap="cmc.batlow", width=0.8)
ax.axhline(21.86, color="0.35", ls="--", lw=1, label="p1 Stefan 게이트 21.9cm")
ax.set_ylabel("LORO RMSE (cm)"); ax.set_xlabel("")
ax.set_xticklabels([LABELS.get(m, m) for m in piv.index], rotation=30, ha="right", fontsize=9)
ax.set_title("물리식별 전이(LORO) 성능 — 기본 Stefan이 전 지역 최선, 정교화 물리는 악화", fontsize=11)
ax.legend(title="test 지역", fontsize=8)
save(fig, "physics_loro_bars")


# ---------------- 5. physics feature 효과 (Part B) ----------------
pb = res[(res.part == "B_feature") & (res.cv == "spatial_block_AK")].copy()
pb["base_model"] = pb.model.str.replace("_BASE", "", regex=False).str.replace("_+physics", "", regex=False)
pb["tag"] = np.where(pb.model.str.contains("physics"), "+physics", "BASE")
g = pb.groupby(["base_model", "tag"]).rmse_cm.mean().unstack()
fig, ax = plt.subplots(figsize=(7, 4.2))
g.plot(kind="bar", ax=ax, color=["#5a7fa5", "#c98a3a"], width=0.7)
ax.axhline(14, color="0.4", ls="--", lw=1, label="대표성 하한 ~14cm")
ax.set_ylabel("in-domain RMSE (cm)"); ax.set_xlabel("")
ax.set_title("S2 physics-as-feature 효과 (A2) — 물리 예측을 ML 입력에 추가", fontsize=11)
ax.tick_params(axis="x", rotation=0); ax.legend(fontsize=8)
save(fig, "physics_feature_effect")

print("[done] S2 시각화 5종 완료")
