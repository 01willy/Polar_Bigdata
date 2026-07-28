"""S11 시각화: conformal UQ 지도 + calibration 곡선 + 비교표·다축 검증표 렌더.

입력: data/processed/s11_conformal_{results,oof}.csv · s11_conformal_meta.json
      data/processed/s11_comparison_table.csv · s11_multiaxis_validation.csv
출력: outputs/figures/s11_uq/*.png+pdf (규약: use_polar·hexbin_map·mask_ocean·
      ALT=oslo_r·불확실성=acton·축/컬러바 단위 필수)
실행: python scripts/4_visualization/s11_uq_figs.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar, CMAP
from polar.geomap import (ALASKA, make_ax, hexbin_map, mask_ocean, add_colorbar,
                          add_scalebar, add_inset_locator)

plt = use_polar()
PROC = C.PROCESSED
OUT = Path(C.ROOT) / "outputs/figures/s11_uq"
OUT.mkdir(parents=True, exist_ok=True)

res = pd.read_csv(PROC / "s11_conformal_results.csv")
oof = pd.read_csv(PROC / "s11_conformal_oof.csv")
meta = json.load(open(PROC / "s11_conformal_meta.json"))
comp = pd.read_csv(PROC / "s11_comparison_table.csv")
multi = pd.read_csv(PROC / "s11_multiaxis_validation.csv")
H = meta["headline"]


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  저장 {name}.png/.pdf")


# ============ 1. calibration 곡선 (명목 vs 실제) + cov90 막대 ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.6),
                               gridspec_kw=dict(width_ratios=[1.25, 1]))
sm = res[res.seed == -1]
per = res[(res.seed >= 0) & (res.setting.isin(["raw", "cqr"]))]
col = {"raw": "#8e9bad", "cqr": "#27536b"}
lab = {"raw": "raw quantile CatBoost", "cqr": "CQR conformal 보정"}
ax1.plot([40, 100], [40, 100], ls="--", c="0.6", lw=1, zorder=1, label="이상적 보정(y=x)")
for tag in ["raw", "cqr"]:
    g = sm[sm.setting == f"{tag}_seedmean"].sort_values("level")
    ax1.plot(g.level * 100, g.coverage * 100, "-o", c=col[tag], lw=1.8, ms=5, label=lab[tag])
    ax1.fill_between(g.level * 100, g.cov_ci_lo * 100, g.cov_ci_hi * 100,
                     color=col[tag], alpha=0.18, lw=0)
    p = per[per.setting == tag]
    ax1.scatter(p.level * 100, p.coverage * 100, c=col[tag], s=10, alpha=0.45, zorder=2)
ax1.set_xlabel("명목 커버리지 (%)"); ax1.set_ylabel("실제 커버리지 (%)")
ax1.set_xlim(45, 95); ax1.set_ylim(10, 100)
ax1.set_title("calibration 곡선 (음영=블록 부트스트랩 95% CI, 점=seed)")
ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
ax1.grid(alpha=0.3, lw=0.4)

r90 = sm[(sm.setting == "raw_seedmean") & (sm.level == 0.90)].iloc[0]
c90 = sm[(sm.setting == "cqr_seedmean") & (sm.level == 0.90)].iloc[0]
bars = ax2.bar([0, 1], [r90.coverage * 100, c90.coverage * 100], width=0.55,
               color=[col["raw"], col["cqr"]])
for i, r in enumerate([r90, c90]):
    ax2.errorbar(i, r.coverage * 100,
                 yerr=[[100 * (r.coverage - r.cov_ci_lo)], [100 * (r.cov_ci_hi - r.coverage)]],
                 c="0.2", capsize=4, lw=1.2)
    ax2.text(i, r.coverage * 100 + 3.5, f"{r.coverage*100:.1f}%\n폭 {r.width_cm:.0f}cm",
             ha="center", fontsize=9)
ax2.axhline(90, ls=":", c="0.35", lw=1)
ax2.text(-0.42, 91.5, "목표 90%", fontsize=8, color="0.35", ha="left")
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["raw\nquantile", "CQR\nconformal"])
ax2.set_ylabel("90% 구간 실제 커버리지 (%)"); ax2.set_ylim(0, 104)
ax2.set_title("cov90: 보정 전후")
fig.suptitle("S11 보정 UQ · 알래스카 in-domain 공간블록 (calibration=train 블록 내부 분리)",
             fontsize=11, y=1.02)
save(fig, "s11_calibration_curve")

# ============ 2. UQ 지도: 중앙값 예측(oslo_r) + CQR 구간폭(acton) ============
# make_ax는 단일 axes 전용이라 지역 표준 투영으로 subplot 두 개를 직접 구성한다.
from polar.geomap import _proj
proj = _proj(ALASKA)
fig = plt.figure(figsize=(13.2, 5.6))
panels = []
for i in range(2):
    ax = fig.add_subplot(1, 2, i + 1, projection=proj)
    _, ax = make_ax(ALASKA, ax=ax, fig=fig)
    panels.append(ax)
axA, axB = panels

v = oof.pred_med.values
hb1 = hexbin_map(axA, oof.lon.values, oof.lat.values, v, gridsize=46, cmap=CMAP.alt,
                 vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 98))
mask_ocean(axA)
add_colorbar(fig, hb1, axA, "예측 ALT 중앙값 (cm)")
add_scalebar(axA); add_inset_locator(fig, axA, ALASKA)
axA.set_title("(a) OOF 중앙값 예측 (quantile CatBoost, 셀 median)", fontsize=10)

w = oof.width90_cqr.values
hb2 = hexbin_map(axB, oof.lon.values, oof.lat.values, w, gridsize=46, cmap=CMAP.err,
                 vmin=np.nanpercentile(w, 2), vmax=np.nanpercentile(w, 98))
mask_ocean(axB)
add_colorbar(fig, hb2, axB, "CQR 90% 구간폭 (cm)")
add_scalebar(axB)
axB.set_title(f"(b) 보정 90% 구간폭 (cov90 {H['cov90_cqr']*100:.1f}%)", fontsize=10)
fig.suptitle("S11 보정 불확실성 지도 · 알래스카 (0.5° 공간블록 OOF, seed-mean)",
             fontsize=11.5, y=0.99)
save(fig, "s11_uq_maps")

# ============ 3. 비교표 렌더 ============
tab = comp.copy()
tab["LORO 지역별 (AK/Lena/CA, cm)"] = tab.apply(
    lambda r: "-" if pd.isna(r.loro_alaska_cm) else
    f"{r.loro_alaska_cm:.1f} / {r.loro_lena_cm:.1f} / {r.loro_canada_cm:.1f}", axis=1)
# regime: transductive 행은 게이트 수치 옆에 표식(‡)을 붙여 inductive 게이트와 혼동 방지
tab["is_trans"] = tab["regime"].astype(str).str.startswith("transductive")
tab["게이트표시"] = tab.apply(
    lambda r: "-" if pd.isna(r.loro_gate_rmse_cm) else
    (f"{r.loro_gate_rmse_cm:.2f} ‡" if r.is_trans else f"{r.loro_gate_rmse_cm:.2f}"), axis=1)
tab["유형표시"] = np.where(tab["is_trans"], "transductive ‡", "inductive")
show = tab[["stage", "method", "유형표시", "indomain_rmse_cm", "게이트표시",
            "LORO 지역별 (AK/Lena/CA, cm)", "indomain_bias_cm", "adopted"]].copy()
show.columns = ["단계", "방법", "회귀 유형", "in-domain\nRMSE(cm)", "LORO 게이트\nRMSE(cm)",
                "LORO 지역별\nAK/Lena/CA (cm)", "in-domain\nbias(cm)", "판정"]
show = show.fillna("-")
# 상단=제목·부제(2줄), 중앙=표, 하단=범례·‡각주(3줄). 표에 세로 여백을 넉넉히 줘 가독 개선.
fig, ax = plt.subplots(figsize=(15.2, 0.50 * len(show) + 3.0))
ax.axis("off")
t = ax.table(cellText=show.values, colLabels=show.columns, loc="center",
             cellLoc="center", bbox=[0.0, 0.11, 1.0, 0.80])
t.auto_set_font_size(False); t.set_fontsize(8.6)
t.auto_set_column_width(list(range(len(show.columns))))
regime_col = list(show.columns).index("회귀 유형")
for (r0, c0), cell in t.get_celld().items():
    cell.set_edgecolor("0.75"); cell.set_linewidth(0.5)
    cell.set_height(cell.get_height() * 1.18)  # 행 높이 확대(밀도 완화)
    if r0 == 0:
        cell.set_facecolor("#2b4a5e"); cell.set_text_props(color="white", fontweight="bold")
    else:
        verdict = str(show.iloc[r0 - 1, -1])
        trans = str(show.iloc[r0 - 1, regime_col]).startswith("transductive")
        if verdict.startswith("채택"):
            cell.set_facecolor("#eaf2ec")
        elif "negative" in verdict or verdict.startswith("미채택"):
            cell.set_facecolor("#f5eded")
        else:
            cell.set_facecolor("#f7f7f5")
        # transductive 행의 유형·게이트 셀은 냉색 톤 배경+테두리 강조로 표시(규약: 온색·주황 금지,
        # 색만으로 정보 인코딩하지 않음: ‡ 병기)
        if trans and c0 in (regime_col, list(show.columns).index("LORO 게이트\nRMSE(cm)")):
            cell.set_facecolor("#dbe6ee")
            cell.set_edgecolor("#27536b"); cell.set_linewidth(1.6)
            cell.set_text_props(color="#1c3c4d", fontweight="bold")
# --- 상단 제목·부제 ---
ax.text(0.5, 0.995, "증강방식 종합 비교 · 동일 데이터(fidelity_base)·동일 fold 규약, CSV 재계산",
        transform=ax.transAxes, ha="center", va="top", fontsize=11.5, fontweight="bold")
ax.text(0.5, 0.945, "in-domain=알래스카 0.5° 공간블록 · LORO 게이트=매크로 3지역 비가중평균 "
        "(S1 FT-T LORO·S3는 프로토콜 상이, caveat 열: s11_comparison_table.csv)",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.0, color="0.3")
# --- 하단 범례(판정 색 안내) + ‡ 각주: 표 아래로 이동해 가독 확보 ---
import matplotlib.patches as mpatches
handles = [mpatches.Patch(fc="#eaf2ec", ec="0.7", label="채택"),
           mpatches.Patch(fc="#f5eded", ec="0.7", label="미채택 · negative"),
           mpatches.Patch(fc="#dbe6ee", ec="#27536b", label="transductive ‡ (보조관측 사용)")]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075),
          ncol=3, frameon=False, fontsize=9.0, handlelength=1.4, columnspacing=1.6)
ax.text(0.5, 0.045, "‡ transductive(target 셀 보조관측 사용): test 지역 셀의 보조관측"
        "(Stefan pseudo·CCI)을 거리 버퍼 없이 학습에 넣음. LORO 게이트가 inductive 전이 아님"
        "(S5 아티팩트 기각과 동일 기준).",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.6, color="#1c3c4d")
ax.text(0.5, 0.010, "pool_mlp 21.11 등 transductive 게이트는 inductive 게이트와 직접 비교 금지. "
        "핵심 결론: 어떤 증강 방식도 Stefan 물리 앵커(전이 최선)를 넘지 못함.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.6, color="0.25")
save(fig, "s11_comparison_table")

# ============ 4. 다축 검증표 렌더 ============
mt = multi.copy()


def _fmt_val(r):
    if pd.isna(r.value):
        return "-"
    ci = str(r.ci95) if isinstance(r.ci95, str) and r.ci95 and r.ci95 != "nan" else ""
    return f"{r.value} {ci}".strip()


mt["value"] = mt.apply(_fmt_val, axis=1)
# 미실시 축(source leave-out)은 심사자 오해 방지 위해 상태를 명시적으로 표기(각주에 사유·대체진단).
notdone = mt["best_method"].astype(str).str.strip() == "미실시"
mt.loc[notdone, "best_method"] = "미실시 (future work) †"
showm = mt[["axis", "protocol", "best_method", "metric", "value"]].copy()
showm.columns = ["검증 축", "프로토콜", "대표 방법", "지표", "값 [95% CI]"]
showm = showm.fillna("-")
notdone_rows = set(np.where(notdone.values)[0].tolist())  # showm(=mt) 행 인덱스
fig, ax = plt.subplots(figsize=(13.6, 0.52 * len(showm) + 2.1))
ax.axis("off")
t = ax.table(cellText=showm.values, colLabels=showm.columns, loc="center",
             cellLoc="center", bbox=[0.0, 0.10, 1.0, 0.80])
t.auto_set_font_size(False); t.set_fontsize(8.2)
t.auto_set_column_width(list(range(len(showm.columns))))
for (r0, c0), cell in t.get_celld().items():
    cell.set_edgecolor("0.75"); cell.set_linewidth(0.5)
    if r0 == 0:
        cell.set_facecolor("#2b4a5e"); cell.set_text_props(color="white", fontweight="bold")
    elif (r0 - 1) in notdone_rows:
        # 미실시 축은 완료 축과 구분되게 흐린 냉색 배경+기울임(심사자 오해 방지)
        cell.set_facecolor("#e6ebee"); cell.set_text_props(color="0.35", fontstyle="italic")
    else:
        cell.set_facecolor("#f7f7f5" if r0 % 2 else "#efefec")
ax.text(0.5, 0.985, "다축 검증 요약 · 공간블록·LORO·temporal·독립 소스·누설통제·UQ",
        transform=ax.transAxes, ha="center", va="top", fontsize=10.5, fontweight="bold")
ax.text(0.5, 0.945, "각주 전문: s11_multiaxis_validation.csv verdict_footnote 열",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="0.3")
ax.text(0.5, 0.065, "† source leave-out: clean 보조소스가 Stefan·CCI 2종뿐이고 둘 다 이미 공변량이라 "
        "별도 검증 축이 성립하지 않아 미실시(future work). 대체 진단: S6 소스별 편향(b_stefan≈0·b_cci 과소추정)·"
        "S7 독립 KPDC 점검증.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.0, color="0.35")
save(fig, "s11_multiaxis_table")

print("완료: outputs/figures/s11_uq/")
