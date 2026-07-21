"""라벨 증강 백본 재해부 — 시각화(냉색 규약, 단위, PNG+PDF). v2 정정판.

v1 그림의 과장·오독 소지를 정정한다.
  - fig1: '+Lena 단독 무해' 문구 철회. tautology(학습셋=BASE 동일) 셀을 해치·주석으로 표시.
          Lena 증강 순효과는 GTNPenv_OUT에서만 관측 가능함을 명시.
  - fig3: 예측 분포. 근퇴화 기저선(상수 평균예측 RMSE)을 병기.
  - fig4: '붕괴' 표현 유지하되 '결측 채널로만 전파'로 정확화, 근퇴화 기저선 표시.
  - fig5(신규): 붕괴 기제 2x2(플래그 x 심부18행). 플래그 단독은 부분완화, 심부제거가 핵심.
  - fig6: 심부 처리별 전이. cap만 회복(평균예측 수준), gate/downweight는 실패임을 명시.
  - fig7(신규): seed 견고성. GBM 붕괴는 견고, MLP BASE는 seed 변동 큼.

입력: data/processed/aug_backbone_{sourceregion,controls,mechanism,dl,predstats,oof}.csv
산출: outputs/figures/07_augmentation/*, outputs/maps/aug_lena_pred_maps.*
실행: (ROOT) /home/anaconda3/bin/python scripts/3_deep_learning/aug_backbone_figs.py
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from polar.plotstyle import use_polar, CMAP, tnorm, add_cbar, style_geo, BAD
from polar.outputs import figpath, mappath
plt = use_polar()

PROC = "data/processed"
A = pd.read_csv(f"{PROC}/aug_backbone_sourceregion.csv")
B = pd.read_csv(f"{PROC}/aug_backbone_controls.csv")
M = pd.read_csv(f"{PROC}/aug_backbone_mechanism.csv")
Cdl = pd.read_csv(f"{PROC}/aug_backbone_dl.csv")
P = pd.read_csv(f"{PROC}/aug_backbone_predstats.csv")
OOF = pd.read_csv(f"{PROC}/aug_backbone_oof.csv")

BLUE = "#2166ac"; TEAL = "#1a9988"; GREY = "#8a8f98"
IMP = "#2166ac"    # 개선(음의 Δrmse) 청색
WOR = "#7a5195"    # 악화 자주(붉은계열 금지)
TAUT = "#c9ccd1"   # tautology 회색(비교 불가)

# 근퇴화 기저선(상수 평균예측 RMSE = std(y))
lena_obs = OOF[(OOF.target == "Lena") & (OOF.variant == "BASE")].alt_cm.values
LENA_CONST_RMSE = float(np.std(lena_obs))
LENA_OBS_MEAN = float(lena_obs.mean())


def save(fig, path_png):
    fig.savefig(path_png, dpi=260, bbox_inches="tight")
    fig.savefig(path_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def apath(name):
    d = "outputs/figures/07_augmentation"
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name + ".png")


def rmse_of(variant, region, cv):
    r = A[(A.variant == variant) & (A.eval_region == region) & (A.cv_type == cv)]
    return float(r.iloc[0].rmse_cm) if len(r) else np.nan

def taut_of(variant, region, cv):
    r = A[(A.variant == variant) & (A.eval_region == region) & (A.cv_type == cv)]
    if not len(r):
        return ""
    v = r.iloc[0].tautology
    return "" if (pd.isna(v) or v == "") else str(v)


# ============ Fig 1: 소스·지역별 Δrmse (BASE 기준, tautology 표시) ============
# 지역 x 변형. Δrmse = rmse(variant) - rmse(BASE). 음수=개선(청), 양수=악화(자주),
# tautology(학습셋=BASE 동일)=회색+빗금(비교 불가).
regions = ["Alaska_all", "Lena_RU", "GTNPenv_US", "GTNPenv_OUT"]
variants = ["+Lena", "+GTNPenv", "+Lena+GTNPenv"]
cvmap = {"Alaska_all": "spatial_block_AK"}
delta, taut = {}, {}
for reg in regions:
    cv = cvmap.get(reg, "LORO")
    base = rmse_of("BASE", reg, cv)
    delta[reg] = {v: rmse_of(v, reg, cv) - base for v in variants}
    taut[reg] = {v: taut_of(v, reg, cv) for v in variants}

HATCH = {"+Lena": "", "+GTNPenv": "///", "+Lena+GTNPenv": "xx"}
fig, ax = plt.subplots(figsize=(9.8, 5.4))
x = np.arange(len(regions)); w = 0.26
for i, v in enumerate(variants):
    for xi, reg in zip(x + (i - 1) * w, regions):
        val = delta[reg][v]
        is_taut = taut[reg][v] != ""
        col = TAUT if is_taut else (IMP if (np.isfinite(val) and val < 0) else WOR)
        ax.bar(xi, val, w, color=col, edgecolor="#222", linewidth=0.6, hatch=HATCH[v])
        if np.isfinite(val):
            lbl = "=BASE" if is_taut else f"{val:+.0f}"
            ax.annotate(lbl, (xi, val), ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=7.4, xytext=(0, 2 if val >= 0 else -3),
                        textcoords="offset points",
                        color="#555" if is_taut else "#222")
ax.axhline(0, color="#222", lw=0.9)
ax.set_xticks(x)
labels = {"Alaska_all": "알래스카\nin-domain", "Lena_RU": "Lena_RU\n전이(순수 out)",
          "GTNPenv_US": "GTNPenv_US\n(AK내부·집계착시)", "GTNPenv_OUT": "GTNPenv_OUT\n(순수 out)"}
ax.set_xticklabels([labels[r] for r in regions], fontsize=9)
ax.set_ylabel("Δrmse (cm), BASE 기준\n음수=개선 · 양수=악화")
ax.set_title("증강 소스·지역별 전이 RMSE 변화 (BASE 대비)", fontsize=12.5)
h_color = [Patch(fc=IMP, ec="#222", label="개선(Δ<0)"),
           Patch(fc=WOR, ec="#222", label="악화(Δ>0)"),
           Patch(fc=TAUT, ec="#222", label="tautology(학습셋=BASE)")]
h_hatch = [Patch(fc="#dddddd", ec="#222", hatch=HATCH[v], label=v) for v in variants]
leg1 = ax.legend(handles=h_color, loc="upper left", fontsize=8.4, title="부호(색)")
ax.add_artist(leg1)
ax.legend(handles=h_hatch, loc="upper right", fontsize=8.4, title="변형(해치)")
ax.text(0.5, -0.26,
        "Lena가 test인 LORO에서 '+Lena'는 학습셋이 BASE와 비트 동일해져 비교 불가(tautology). "
        "Lena 증강 순효과는 GTNPenv_OUT서 관측: +Lena는 오히려 소폭 악화(+2cm). "
        "붕괴는 +GTNPenv·순수 out-region에 집중.",
        transform=ax.transAxes, ha="center", fontsize=7.8, color="#444")
save(fig, apath("fig1_sourceregion_delta"))
print("fig1 저장")


# ============ Fig 3: 예측 분포(실측 vs BASE vs AUG) + 근퇴화 기저선 ============
lena_base = OOF[(OOF.target == "Lena") & (OOF.variant == "BASE")]
lena_aug = OOF[(OOF.target == "Lena") & (OOF.variant == "+Lena+GTNPenv")]
lena_l = OOF[(OOF.target == "Lena") & (OOF.variant == "+Lena")]
obs = lena_base.alt_cm.values

fig, ax = plt.subplots(figsize=(8.8, 4.8))
bins = np.linspace(0, 160, 55)
ax.hist(obs, bins=bins, color=GREY, alpha=0.55, label=f"실측 (μ={obs.mean():.0f}cm, σ={obs.std():.0f})", density=True)
ax.hist(lena_base.pred.values, bins=bins, histtype="step", lw=2.0, color=BLUE,
        label=f"BASE 예측 (μ={lena_base.pred.mean():.0f}cm)", density=True)
ax.hist(lena_aug.pred.values, bins=bins, histtype="step", lw=2.0, color=WOR,
        label=f"AUG 예측 (μ={lena_aug.pred.mean():.0f}cm)", density=True)
ax.axvline(obs.mean(), color=GREY, ls="--", lw=1.0)
ax.axvline(lena_aug.pred.mean(), color=WOR, ls="--", lw=1.0)
ax.set_xlabel("ALT (cm)")
ax.set_ylabel("밀도")
ax.set_title("Lena_RU 예측 분포: 증강이 예측을 심부로 이동", fontsize=12)
ax.legend(fontsize=8.8, loc="upper right")
ax.annotate(f"AUG 예측이 실측 {obs.mean():.0f}cm에서\n{lena_aug.pred.mean():.0f}cm로 이동 (+85cm 편향)",
            xy=(lena_aug.pred.mean(), ax.get_ylim()[1] * 0.5),
            xytext=(92, ax.get_ylim()[1] * 0.72),
            fontsize=8.3, color=WOR,
            arrowprops=dict(arrowstyle="->", color=WOR, lw=1.0))
ax.text(0.5, -0.20,
        f"BASE 예측은 폭이 좁다(σ={lena_base.pred.std():.0f}cm). BASE 전이 RMSE {rmse_of('BASE','Lena_RU','LORO'):.0f}cm는 "
        f"상수 평균예측 RMSE {LENA_CONST_RMSE:.0f}cm(=실측 σ)보다 나쁨(skill<0). BASE도 근퇴화 상태다.",
        transform=ax.transAxes, ha="center", fontsize=8.0, color="#444")
save(fig, apath("fig3_pred_hist"))
print("fig3 저장")


# ============ Fig 4: 통제군 — 물리 vs ML × 증강 Δrmse (결측 채널 가설) ============
def get_ctrl(control, model, feats, setn, reg="Lena_RU"):
    r = B[(B.control == control) & (B.model == model) & (B.feats == feats)
          & (B["set"] == setn) & (B.eval_region == reg)]
    return float(r.iloc[0].rmse_cm) if len(r) else np.nan

models = [
    ("물리(Stefan)\n기후만·결측無", "physics_vs_ml", "physics_stefan", "climate_sqrt_tdd"),
    ("ML 기후8\n결측無", "climate_vs_full", "gbm", "climate8"),
    ("ML 지형+기후\n결측無", "climate_vs_full", "gbm", "terr+clim"),
    ("ML 전공변량\n(InSAR/CCI 결측)", "climate_vs_full", "gbm", "full"),
]
base_r = [get_ctrl(c, m, f, "BASE") for _, c, m, f in models]
aug_r = [get_ctrl(c, m, f, "AUG") for _, c, m, f in models]

fig, ax = plt.subplots(figsize=(9.2, 5.0))
x = np.arange(len(models)); w = 0.38
ax.bar(x - w / 2, base_r, w, color=BLUE, edgecolor="#333", lw=0.5, label="BASE(증강 전)")
ax.bar(x + w / 2, aug_r, w, color=WOR, edgecolor="#333", lw=0.5, label="AUG(+Lena+GTNPenv)")
for xi, (bv, av) in zip(x, zip(base_r, aug_r)):
    ax.annotate(f"{bv:.0f}", (xi - w / 2, bv), ha="center", va="bottom", fontsize=8)
    ax.annotate(f"{av:.0f}", (xi + w / 2, av), ha="center", va="bottom", fontsize=8)
    d = av - bv
    ax.annotate(f"Δ{d:+.0f}", (xi, max(bv, av)), ha="center", va="bottom",
                fontsize=8, color="#333", xytext=(0, 12), textcoords="offset points",
                fontweight="bold")
ax.axhline(LENA_CONST_RMSE, color=GREY, ls=":", lw=1.0,
           label=f"상수 평균예측 RMSE={LENA_CONST_RMSE:.0f}cm")
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in models], fontsize=8.5)
ax.set_ylabel("Lena_RU 전이 RMSE (cm)")
ax.set_title("통제군: 결측 공변량을 가진 전공변량 ML만 증강에 악화\n(붕괴는 결측 채널로만 전파)",
             fontsize=11.5)
ax.legend(fontsize=8.6, loc="upper left")
ax.text(0.5, -0.22, "결측 없는 물리·기후·지형기후 모델은 증강에 무해(Δ≈0). 결측(InSAR/PolSAR/CCI)을 "
        "가진 전공변량 ML만 +66cm 악화. 단 BASE 자체가 평균예측선 부근(회색 점선)이다.",
        transform=ax.transAxes, ha="center", fontsize=8.1, color="#444")
save(fig, apath("fig4_control_routing"))
print("fig4 저장")


# ============ Fig 5(신규): 붕괴 기제 2x2(플래그 x 심부18행) ============
def mrm(cell):
    r = M[M.cell == cell]
    return float(r.iloc[0].rmse_cm) if len(r) else np.nan
n_deep = int(M.iloc[0].n_deep_removed)
cells = [("AUG_flags", "플래그有\n심부18행有"), ("AUG_noflags", "플래그無\n심부18행有"),
         ("AUG_nodeep_flags", "플래그有\n심부18행無"), ("AUG_nodeep_noflags", "플래그無\n심부18행無")]
vals = [mrm(c) for c, _ in cells]
base_lena = rmse_of("BASE", "Lena_RU", "LORO")
fig, ax = plt.subplots(figsize=(8.6, 4.8))
cols = [IMP if v < base_lena + 10 else WOR for v in vals]
ax.bar(range(4), vals, color=cols, edgecolor="#333", lw=0.5, width=0.62)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=9)
ax.axhline(base_lena, color=BLUE, ls="--", lw=1.2, label=f"BASE 전이 = {base_lena:.0f}cm")
ax.set_xticks(range(4)); ax.set_xticklabels([lab for _, lab in cells], fontsize=8.6)
ax.set_ylabel("Lena_RU 전이 RMSE (cm)")
ax.set_title(f"붕괴 기제 분해: _isnan 플래그 단독이 아니라 심부 GTNPenv {n_deep}행이 핵심", fontsize=11.5)
ax.legend(fontsize=9, loc="upper right")
ax.text(0.5, -0.22,
        f"플래그 제거만으로는 {mrm('AUG_flags'):.0f}→{mrm('AUG_noflags'):.0f}cm 부분완화에 그친다. "
        f"심부 {n_deep}행 제거 시 플래그 유무와 무관하게 회복(플래그有 {mrm('AUG_nodeep_flags'):.0f}, "
        f"플래그無 {mrm('AUG_nodeep_noflags'):.0f}cm). 기제는 '플래그 라우팅'이 아니라 심부 라벨이다.",
        transform=ax.transAxes, ha="center", fontsize=8.0, color="#444")
save(fig, apath("fig5_mechanism_2x2"))
print("fig5 저장")


# ============ Fig 6: 심부 라벨 처리별 전이 RMSE(cap만 회복) ============
dt = B[B.control == "deep_treatment"].set_index("set")
order_t = ["as_is", "cap_gt150", "aoa_gate", "deep_downweight"]
names = {"as_is": "무처리\n(그대로)", "cap_gt150": "ALT>150 제외\n(캡)",
         "aoa_gate": "AOA 게이팅\n(환경 비유사 배제)", "deep_downweight": "심부 가중축소\n(w=0.1)"}
vals = [dt.loc[o, "rmse_cm"] for o in order_t]
fig, ax = plt.subplots(figsize=(8.6, 4.8))
cols = [IMP if v < base_lena + 10 else WOR for v in vals]
ax.bar(range(len(order_t)), vals, color=cols, edgecolor="#333", lw=0.5, width=0.6)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=9)
ax.axhline(base_lena, color=BLUE, ls="--", lw=1.2, label=f"BASE 전이 = {base_lena:.0f}cm")
ax.axhline(LENA_CONST_RMSE, color=GREY, ls=":", lw=1.0, label=f"상수 평균예측 = {LENA_CONST_RMSE:.0f}cm")
ax.set_xticks(range(len(order_t))); ax.set_xticklabels([names[o] for o in order_t], fontsize=8.6)
ax.set_ylabel("Lena_RU 전이 RMSE (cm)")
ax.set_title("심부 라벨 처리별 전이: 캡(ALT>150 제외)만 회복, 게이팅·가중은 실패", fontsize=11.5)
ax.legend(fontsize=8.8, loc="upper right")
ax.text(0.5, -0.22, "무처리 88cm에서 캡 25cm로 복귀(BASE·평균예측 수준). 게이팅·가중은 48cm로 회복 실패. "
        "'심부처리로 회복'은 cap 한정이며, 그 회복도 전이 기량이 아니라 평균예측 수준으로의 복귀다.",
        transform=ax.transAxes, ha="center", fontsize=8.0, color="#444")
save(fig, apath("fig6_deep_treatment"))
print("fig6 저장")


# ============ Fig 7(신규): seed 견고성(GBM 붕괴 견고 vs MLP BASE 변동) ============
def seed_vals(model, setn):
    return Cdl[(Cdl.model == model) & (Cdl["set"] == setn)].rmse_cm.values
groups = [("GBM\nBASE", "gbm", "BASE"), ("GBM\nAUG", "gbm", "+Lena+GTNPenv"),
          ("MLP\nBASE", "mlp", "BASE"), ("MLP\nAUG", "mlp", "+Lena+GTNPenv")]
fig, ax = plt.subplots(figsize=(8.4, 4.8))
for i, (lab, model, setn) in enumerate(groups):
    v = seed_vals(model, setn)
    col = WOR if "AUG" in lab else BLUE
    ax.bar(i, v.mean(), 0.6, color=col, edgecolor="#333", lw=0.5, alpha=0.75)
    ax.errorbar(i, v.mean(), yerr=v.std(), color="#222", capsize=5, lw=1.2)
    ax.scatter(np.full(len(v), i), v, color="#222", s=22, zorder=5)
    ax.annotate(f"{v.mean():.0f}±{v.std():.0f}", (i, v.max()), ha="center", va="bottom",
                fontsize=8, xytext=(0, 4), textcoords="offset points")
ax.axhline(base_lena, color=BLUE, ls="--", lw=1.0)
ax.axhline(LENA_CONST_RMSE, color=GREY, ls=":", lw=1.0)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylabel("Lena_RU 전이 RMSE (cm)")
ax.set_title("seed 견고성: GBM 붕괴는 견고, MLP BASE는 seed 변동 큼", fontsize=11.5)
ax.text(0.5, -0.20,
        "GBM AUG 붕괴는 모든 seed에서 견고(74-88cm). MLP는 BASE 자체가 seed에 따라 24-44cm로 변동해 "
        "'DL은 증강 하에서만 붕괴'라는 인과 서술이 단일 seed에서 불안정하다.",
        transform=ax.transAxes, ha="center", fontsize=8.0, color="#444")
save(fig, apath("fig7_seed_robustness"))
print("fig7 저장")


# ============ Map: Lena 예측 지도 BASE vs +GTNPenv vs +Lena ============
lena_g = OOF[(OOF.target == "Lena") & (OOF.variant == "+GTNPenv")]
panels = [("BASE (알래스카만)", lena_base), ("+GTNPenv (심부 라벨)", lena_g),
          ("+Lena (=BASE, tautology)", lena_l)]
alt_lo, alt_hi = 0, 130
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
for ax, (title, dfp) in zip(axes, panels):
    sc = ax.scatter(dfp.lon, dfp.lat, c=dfp.pred, cmap=CMAP.alt, vmin=alt_lo, vmax=alt_hi,
                    s=9, edgecolor="none")
    style_geo(ax, title=f"{title}\n예측μ={dfp.pred.mean():.0f}cm (실측μ={LENA_OBS_MEAN:.0f}cm)")
    ax.set_aspect("auto")
cb = add_cbar(fig, sc, list(axes), "예측 ALT (cm)")
fig.suptitle("Lena_RU 예측 지도: +GTNPenv 증강이 예측을 심부로 과대추정 (실측 42cm)",
             fontsize=12.5, fontweight="bold")
fig.savefig(mappath("aug_lena_pred_maps"), dpi=260, bbox_inches="tight")
fig.savefig(mappath("aug_lena_pred_maps", ext="pdf"), bbox_inches="tight")
plt.close(fig)
print("map 저장: aug_lena_pred_maps")

print("\n전부 저장 완료(PNG+PDF).")
