"""H18–H24 결과 그림 (전이 붕괴 완화 실험) — 논문급 정적 그림. 스펙 figures/figure_spec.json (id h2/*).

원칙(design/layout_rules.md): 그림 하나 = 결론형 제목 하나, 격자 배치, 단위 필수, 냉색 팔레트(검증 통과 4색 + 중립 참조선),
색으로만 정보 전달 금지(마커·라벨 병행), 벡터(PDF)+PNG 300 dpi, pdf.fonttype 42.
입력: data/processed/h2/*.csv (h21·h22·h23·h24·h19·h_tests), data/processed/m1/*.csv, covariates_ext_v1.csv
산출: outputs/figures/h2/<id>.{png,pdf}
실행: python3 scripts/4_visualization/h2_figs.py [--only fig01,fig04,map01]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.plotstyle import use_polar, despine, lon_formatter, lat_formatter, CMAP, tnorm   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--dpi", type=int, default=300)
args = ap.parse_args()
plt = use_polar()
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.fontsize": 9})
PROC = ROOT / "data" / "processed"; H2 = PROC / "h2"; M1 = PROC / "m1"
FIG = ROOT / "outputs" / "figures" / "h2"; FIG.mkdir(parents=True, exist_ok=True)

PAL = dict(blue="#3a6ea5", teal="#2a9d8f", purple="#8e6bbf", ochre="#b8791f", ref="#6b7280", navy="#17365d", light="#c9d3df")
KR = {"Lena": "레나델타", "Canada": "캐나다", "Russia_W": "러시아 서부", "Russia_C": "러시아 중부", "Russia_E": "러시아 동부",
      "Greenland": "그린란드", "Alaska": "알래스카"}
MAIN6 = ["Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"]
AB4 = ["Lena", "Canada", "Russia_W", "Russia_E"]
EXPORTS = []


def save(fig, name, note=None):
    for ext in ("png", "pdf"):
        p = FIG / f"{name}.{ext}"
        fig.savefig(p, dpi=args.dpi if ext == "png" else None, bbox_inches="tight")
        EXPORTS.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    print(f"[fig] {name}" + (f" — {note}" if note else ""), flush=True)


def corner_free(ax, x, y, frac=0.25):
    """데이터 점이 가장 적은 축 모서리(axes fraction 좌표)를 고른다: 텍스트 박스 충돌 회피."""
    xl, xh = ax.get_xlim() if ax.get_xlim() != (0.0, 1.0) else (np.nanmin(x), np.nanmax(x)); yl, yh = ax.get_ylim() if ax.get_ylim() != (0.0, 1.0) else (np.nanmin(y), np.nanmax(y))
    fx = (x - xl) / (xh - xl + 1e-12); fy = (y - yl) / (yh - yl + 1e-12)
    cands = {(0.02, 0.04): ((fx < frac) & (fy < frac)).sum(), (0.98, 0.04): ((fx > 1 - frac) & (fy < frac)).sum(),
             (0.02, 0.96): ((fx < frac) & (fy > 1 - frac)).sum(), (0.98, 0.96): ((fx > 1 - frac) & (fy > 1 - frac)).sum()}
    return min(cands, key=lambda k: cands[k])


def panel_label(ax, s, x=-0.08, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")


def ci_bar(ax, y, lo, hi, color, lw=1.4):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=2)


# ====================================================================== fig01 붕괴의 구조
def fig01_collapse():
    s21 = pd.read_csv(H2 / "h21_summary.csv"); l2 = pd.read_csv(M1 / "l2_regional_E.csv").set_index("region")
    sc = pd.read_csv(M1 / "m1_sc_summary.csv")
    ml = sc[(sc.cond == "noinfo") & (sc.dset == "alaska") & (sc.xset == "x25") & (sc.pseudo == "none") & (sc.anchor == "none") & (sc.resid == "catboost_lo") & (sc.lam == 1.0) & (sc.iw == 0)]
    ml = ml.groupby("target").rmse.mean()
    val = {r: dict(stefan=float(s21[(s21.target == r) & (s21.cfg == "E_AK")].rmse.iloc[0]), oracle=float(s21[(s21.target == r) & (s21.cfg == "oracle")].rmse.iloc[0]),
                   ml=float(ml.get(r, np.nan)), E=float(l2.loc[r, "E_LS"]), E_lo=l2.loc[r, "E_LS_lo"], E_hi=l2.loc[r, "E_LS_hi"], n=int(l2.loc[r, "n_cells"]),
                   nb=int(l2.loc[r, "n_blocks"])) for r in MAIN6}
    E_AK = float(l2.loc["Alaska", "E_LS"])
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), gridspec_kw=dict(width_ratios=[1.25, 1.0, 0.9]))
    # (a) 지역별 RMSE: Stefan(알래스카 E) · 직접 ML · 자체 E 오라클
    ax = axes[0]; y = np.arange(len(MAIN6))[::-1]; h = 0.26
    al = [1.0 if val[r]["nb"] >= 8 else 0.45 for r in MAIN6]
    ax.barh(y + h, [val[r]["stefan"] for r in MAIN6], h, color=PAL["ref"], alpha=None, label="Stefan(알래스카 E), 물리식")
    ax.barh(y, [val[r]["ml"] for r in MAIN6], h, color=PAL["teal"], label="직접 ML(CatBoost, x25)")
    ax.barh(y - h, [val[r]["oracle"] for r in MAIN6], h, color=PAL["blue"], label="자체 E Stefan(오라클 상한)")
    for bars in ax.containers:
        for b, a in zip(bars, al):
            b.set_alpha(a)
    for i, r in enumerate(MAIN6):
        ax.text(max(val[r]["stefan"], val[r]["ml"]) + 0.8, y[i], f"n={val[r]['n']:,}", va="center", fontsize=9, color="#444")
    ax.set_yticks(y); ax.set_yticklabels([KR[r] for r in MAIN6]); ax.set_xlabel("RMSE (cm)")
    ax.set_title("(a) 라벨 없는 전이에서 ML은 물리식을 넘지 못한다(블록≥8 인 4지역)", loc="left"); ax.legend(loc="lower right", fontsize=8.5); despine(ax)
    ax.text(0.99, 0.98, "흐린 막대 = 블록<8(참고)", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#666")
    ax.set_xlim(0, max(v["ml"] for v in val.values()) * 1.18)
    # (b) 수준 오차 대 구조 오차 분해: Stefan(E_AK) RMSE = 자체 E 잔여(구조) + 알래스카 E 초과(수준)
    ax = axes[1]
    struct = np.array([val[r]["oracle"] for r in MAIN6]); level = np.array([val[r]["stefan"] - val[r]["oracle"] for r in MAIN6])
    ax.barh(y, struct, 0.55, color=PAL["blue"], label="구조 오차(자체 E 로도 남는 오차)")
    ax.barh(y, level, 0.55, left=struct, color=PAL["ochre"], hatch="//", edgecolor="white", label="수준 오차(알래스카 E 사용으로 추가)")
    for i, r in enumerate(MAIN6):
        ax.text(struct[i] + level[i] + 0.8, y[i], f"E 비 {val[r]['E'] / E_AK:.2f}", va="center", fontsize=9, color="#333")
    ax.set_yticks(y); ax.set_yticklabels([KR[r] for r in MAIN6]); ax.set_xlabel("RMSE (cm)")
    ax.set_title("(b) 붕괴의 성격: 수준(러시아 W·C) 대 구조(레나·캐나다)", loc="left"); ax.legend(loc="lower right", fontsize=8.5); despine(ax)
    ax.set_xlim(0, (struct + level).max() * 1.28)
    # (c) 지역별 Stefan 계수 E (95% CI) 대 알래스카 E
    ax = axes[2]
    for i, r in enumerate(MAIN6):
        c = PAL["blue"] if val[r]["nb"] >= 8 else PAL["light"]
        ci_bar(ax, y[i], val[r]["E_lo"], val[r]["E_hi"], c)
        ax.plot(val[r]["E"], y[i], "o", color=PAL["blue"] if val[r]["nb"] >= 8 else PAL["ref"], ms=6, zorder=3)
    ax.axvline(E_AK, color=PAL["ref"], ls="--", lw=1.2); ax.text(E_AK + 0.05, y[-1] - 0.45, f"알래스카 E = {E_AK:.2f}", fontsize=8.5, color=PAL["ref"], va="top")
    ax.set_yticks(y); ax.set_yticklabels([KR[r] for r in MAIN6]); ax.set_xlabel("Stefan 계수 E (cm·(°C·day)$^{-1/2}$)")
    ax.set_title("(c) 지역 계수 E와 알래스카 E의 격차", loc="left"); despine(ax); ax.set_xlim(1.0, 4.3); ax.set_ylim(y[-1] - 1.0, y[0] + 0.8)
    ax.legend(handles=[Line2D([], [], marker="o", color=PAL["blue"], ls="-", label="E (블록 부트스트랩 95% CI)"),
                       Line2D([], [], marker="o", color=PAL["ref"], ls="none", label="블록<8: CI 미산출")], loc="upper right", fontsize=8)
    fig.suptitle("전이 붕괴의 구조: 정보 없음 조건(알래스카 학습, 대상 라벨 0), 6개 주 전이 지역", fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.01, -0.03, "근거: data/processed/h2/h21_summary.csv · m1/l2_regional_E.csv · m1/m1_sc_summary.csv(iw=0). 평가 셀 = 실측·CCI·토양 도일 유효. 러시아 중부(7셀)·그린란드(1셀)는 참고. n 은 지역 전체 셀(캐나다 752 = F4_direct 750 + 지온 유도 2).", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "h2_fig01_collapse_structure")


# ====================================================================== fig04 계수 대여(H21)·CCI 상대 계수(H20)
def fig04_coef_borrow():
    s21 = pd.read_csv(H2 / "h21_summary.csv"); t21 = pd.read_csv(H2 / "h21_tests.csv"); n21 = pd.read_csv(H2 / "h21_nested.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), gridspec_kw=dict(width_ratios=[1.15, 1.0]))
    # (a) 대상별 E 추정치: 구성 계열별 점 + 오라클·알래스카
    ax = axes[0]; y = np.arange(len(MAIN6))[::-1]
    fam = {"E_AK": ("알래스카 E(기준)", PAL["ref"], "s"), "pooled_region": ("공여 지역 등가중 E", PAL["ochre"], "D"),
           "nearest": ("최근접 공여 E(지리·공변량·물리)", PAL["teal"], "o"), "kernel": ("커널 가중 E", PAL["purple"], "^"),
           "nested": ("중첩 선택", PAL["blue"], "*"), "oracle": ("자체 E(오라클)", PAL["navy"], "x")}
    for i, r in enumerate(MAIN6):
        sub = s21[s21.target == r]
        for k, (lab, col, mk) in fam.items():
            if k in ("nearest", "kernel"):
                v = sub[(sub.kind == k) & (sub.w == 1.0)].E.values
            else:
                v = sub[sub.cfg == k].E.values
            jit = np.linspace(-0.18, 0.18, len(v)) if len(v) > 1 else [0]
            ax.plot(v, y[i] + np.array(jit), mk, color=col, ms=9 if k == "nested" else 6, mfc="none" if k in ("nearest", "kernel") else col, mew=1.4, zorder=3 if k in ("nested", "oracle") else 2)
    ax.set_yticks(y); ax.set_yticklabels([KR[r] for r in MAIN6]); ax.set_xlabel("Stefan 계수 E (cm·(°C·day)$^{-1/2}$)")
    ax.set_title("(a) 유사 지역에서 빌린 E는 자체 E(×)를 맞히지 못한다", loc="left"); despine(ax)
    ax.legend(handles=[Line2D([], [], marker=mk, color=col, ls="none", mfc="none" if k in ("nearest", "kernel") else col, label=lab) for k, (lab, col, mk) in fam.items()],
              loc="upper right", fontsize=8); ax.set_xlim(1.2, 4.2)
    # (b) ΔRMSE 대 알래스카 E Stefan: 정보 없음 6지역 + 지역 평균, 부트스트랩 CI
    ax = axes[1]
    tests = [("H21_nested_vs_EAK", "중첩 선택 계수 대여", PAL["blue"], "*"), ("H21ctrl_pooled_region_vs_EAK", "공여 등가중 E(대조)", PAL["ochre"], "D"),
             ("H21ctrl_pooled_cell_vs_EAK", "공여 전 셀 E(대조)", PAL["purple"], "s"), ("H20_cci_blockE_vs_stefan", "CCI 블록 계수(H20)", PAL["teal"], "^"),
             ("H21ref_oracle_vs_EAK", "자체 E(상한)", PAL["navy"], "x")]
    rows_y = MAIN6 + ["MEAN"]
    yy = np.arange(len(rows_y))[::-1]
    for j, (tid, lab, col, mk) in enumerate(tests):
        off = (j - (len(tests) - 1) / 2) * 0.14
        sub = t21[t21.test == tid]
        for i, r in enumerate(rows_y):
            q = sub[sub.target.astype(str).str.startswith("MEAN")] if r == "MEAN" else sub[sub.target == r]
            if not len(q):
                continue
            q = q.iloc[0]
            ci_bar(ax, yy[i] - off, q.ci_lo, q.ci_hi, col)
            ax.plot(q.delta, yy[i] - off, mk, color=col, ms=8 if mk in ("*", "x") else 5.5, mew=1.6, zorder=3, label=lab if i == 0 else None)
    ax.axvline(0, color=PAL["ref"], lw=1)
    ax.set_yticks(yy); ax.set_yticklabels([KR.get(r, "4지역 평균(블록≥8, 층화 CI)") for r in rows_y]); ax.set_xlabel("ΔRMSE 대 Stefan(알래스카 E) (cm)  ← 개선 | 악화 →")
    ax.set_title("(b) 계수 대여·CCI 계수 모두 확정 이득 없음(CI가 0 포함)", loc="left"); despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
    ax.set_xlim(-23, 17)
    nested_txt = " · ".join(f"{KR[r]}: {c}" for r, c in zip(n21.target, n21.chosen) if r in MAIN6)
    fig.suptitle("H21 지역 간 계수 대여와 H20 시뮬레이터 상대 계수 (정보 없음 조건, 해석적)", fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.01, -0.10, f"중첩 선택 결과(공여 leave-one-out): {nested_txt}.\n근거: data/processed/h2/h21_{{summary,tests,nested}}.csv. 블록<8 지역(러시아 중부·그린란드)은 CI 미산출(점만)이며 평균 행에서도 제외.", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "h2_fig04_coef_borrow")


# ====================================================================== map01 신규 공변량 지도
def map01_new_covariates():
    b = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=["loc_id", "region", "lat", "lon", "source_id"])
    e = pd.read_csv(PROC / "covariates_ext_v1.csv").drop(columns=["lat", "lon"]).merge(b, on="loc_id")
    e = e[e.source_id == "F4_direct"]
    regs = {"알래스카": e.region.isin(["ABoVE_AK", "United States (Alaska)"]), "레나델타": e.region == "Lena_RU", "캐나다": e.region.isin(["ABoVE_CA", "Canada", "CALM_Canada"])}
    cov = [("lst_nfactor", "MODIS LST n-factor (지표 도일/기온 도일, 무차원)", "div", 1.0, (0.6, 1.4)),
           ("twi", "지형 습윤 지수 TWI (무차원)", "seq", None, (7, 14)),
           ("water_occ_5km", "수체 점유율 5 km (%)", "seq", None, (0, 60))]
    fig, axes = plt.subplots(len(cov), len(regs), figsize=(13.5, 10.5), constrained_layout=True)
    for i, (c, lab, kind, center, vr) in enumerate(cov):
        from cmcrameri import cm as _cmc
        cmap = CMAP.diff if kind == "div" else (CMAP.count if c == "twi" else matplotlib.colors.LinearSegmentedColormap.from_list("water", _cmc.lapaz_r(np.linspace(0.05, 0.95, 256))))
        if kind != "div":
            cmap.set_bad("#e9ecef")
        norm = tnorm(vr[0], vr[1], center) if kind == "div" else matplotlib.colors.Normalize(vr[0], vr[1])
        for j, (rn, m) in enumerate(regs.items()):
            ax = axes[i, j]; d = e[m]
            sc = ax.scatter(d.lon, d.lat, c=d[c], s=4 if len(d) > 2000 else 14, cmap=cmap, norm=norm, linewidths=0, rasterized=True)
            ax.set_aspect(1 / np.cos(np.radians(d.lat.mean())))
            ax.xaxis.set_major_formatter(lon_formatter())
            from matplotlib.ticker import FuncFormatter, MaxNLocator
            dec = 1 if (d.lat.max() - d.lat.min()) < 5 else 0
            ax.yaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p, dec=dec: f"{v:.{dec}f}°N"))
            ax.tick_params(labelsize=8); despine(ax)
            if i == 0:
                ax.set_title(f"{rn} (셀 {len(d):,})", loc="left")
            if j == len(regs) - 1:
                cb = fig.colorbar(sc, ax=axes[i, :].tolist(), fraction=0.02, pad=0.02, extend="both" if kind == "div" else "max"); cb.set_label(lab, fontsize=9.5)
            med = float(np.nanmedian(d[c]))
            x0, y0 = corner_free(ax, d.lon.values, d.lat.values)
            ax.text(x0, y0, f"중앙값 {med:.2f}", transform=ax.transAxes, fontsize=9, ha="left" if x0 < 0.5 else "right", va="bottom" if y0 < 0.5 else "top",
                    bbox=dict(fc="white", ec="none", alpha=0.85))
    fig.suptitle("신규 공변량의 공간 분포: 관측 지표 강제력(LST)·지형 수문(TWI)·수체(GSW), 라벨 셀 위치", fontsize=13.5, fontweight="bold")
    fig.text(0.01, -0.015, "출처: MOD11C3 v061(2015–2020 월 기후값) · Copernicus DEM 90 m + pysheds · JRC GSW 2021. n-factor 는 1 중심 발산(청=지표가 기온보다 차가움). 색 범위는 고정(초과는 컬러바 화살표), 회색 = 결측. 셀 = F4_direct 라벨 셀.", fontsize=8, color="#555")
    save(fig, "h2_map01_new_covariates")


# ====================================================================== fig03 H19a 블록 E 설명력
def fig03_blockE():
    f = H2 / "h19_blockE.csv"
    if not f.exists():
        print("[skip] fig03: h19_blockE.csv 없음"); return
    R = pd.read_csv(f); C = pd.read_csv(H2 / "h19_blockE_cells.csv")
    order = ["base17", "base17+LST", "base17+T", "base17+W", "base17+V", "base17+H", "base17+A", "base17+A+LST", "A_only"]
    lab = {"base17": "기준 17(토양+기후)", "base17+LST": "+LST", "base17+T": "+지형수문", "base17+W": "+수체", "base17+V": "+식생", "base17+H": "+ERA5 확장",
           "base17+A": "+전부(A)", "base17+A+LST": "+A+LST", "A_only": "신규만(A)"}
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    x = np.arange(len(order)); w = 0.38
    for k, (metric, ttl) in enumerate([("r2_lobo_AK_w", "(a) 알래스카 블록 내 설명력(LOBO, n 가중 R²)"), ("r2_loro_nonAK_w", "(b) 지역 간 설명력(LORO, 비알래스카 블록, n 가중 R²)")]):
        ax = axes[k]
        for j, (model, col) in enumerate([("ridge", PAL["blue"]), ("catboost", PAL["teal"])]):
            v = [float(R[(R.set == s) & (R.model == model)][metric].iloc[0]) if ((R.set == s) & (R.model == model)).any() else np.nan for s in order]
            ax.bar(x + (j - 0.5) * w, np.clip(v, -1.0, None), w, color=col, label={"ridge": "ridge(α=10)", "catboost": "CatBoost(얕음)"}[model])
            for xi, vi in zip(x + (j - 0.5) * w, v):
                if np.isfinite(vi) and vi < -1.0:
                    ax.text(xi, -1.02, f"{vi:.1f}", ha="center", va="top", fontsize=7.5, color="#333333", rotation=90)
        ax.axhline(0, color=PAL["ref"], lw=1); ax.set_xticks(x); ax.set_xticklabels([lab[s] for s in order], rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel("R² (무차원)"); ax.set_title(ttl, loc="left"); despine(ax); ax.set_ylim(-1.2, 1.0)
        if k == 0:
            ax.legend(fontsize=8.5, loc="upper left")
    ax = axes[2]
    piv = C[C.model == "catboost"].pivot_table(index="set", columns="target", values="rmse_emap").reindex(order)
    st = C.groupby("target").rmse_stefan.first()
    for j, r in enumerate(AB4):
        ax.plot(x, piv[r].values - st[r], marker="osD^"[j], color=[PAL["blue"], PAL["teal"], PAL["purple"], PAL["ochre"]][j], lw=1.6, ms=5, label=KR[r])
    ax.axhline(0, color=PAL["ref"], lw=1); ax.set_xticks(x); ax.set_xticklabels([lab[s] for s in order], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("ΔRMSE 대 Stefan(알래스카 E) (cm)"); ax.set_title("(c) 계수 지도 앵커(CatBoost)의 셀 RMSE 변화", loc="left"); despine(ax)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))          # 범례 자리를 데이터 위쪽 빈 공간에 확보
    ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.95)
    fig.suptitle("H19a 신규 공변량이 0.5° 블록 Stefan 계수 E의 변동을 설명하는가", fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.01, -0.06, "블록 E = 블록 셀(≥2)의 최소제곱 E, 입력 = 블록 평균 공변량, √n 가중. LORO = 지역 hold-out(알래스카·레나·캐나다·러시아 W/E); (b)는 비알래스카 블록만의 n 가중 R²(알래스카 포함 r2_loro_all 과 다름). R² < -1 은 막대 아래 수치. 근거: data/processed/h2/h19_blockE{,_cells}.csv", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "h2_fig03_blockE_explained")



# ====================================================================== 공용: 포레스트(지역 점 + 평균 CI)
def forest(ax, T, specs, conds=(("covonly", PAL["blue"], "o"), ("noinfo", PAL["teal"], "s")), regions=AB4, xlab="ΔRMSE (cm)", xlim=None):
    """specs: [(test_id, label)] 행. 각 행에 조건별 평균 Δ(채움 마커 + CI)와 지역 점(작은 빈 마커)."""
    y = np.arange(len(specs))[::-1]
    for i, (tid, lab) in enumerate(specs):
        for j, (cond, col, mk) in enumerate(conds):
            off = (j - (len(conds) - 1) / 2) * 0.22
            sub = T[(T.test == tid) & (T.cond == cond)]
            if not len(sub):
                continue
            m = sub[sub.target.astype(str).str.startswith("MEAN")]
            for ri, r in enumerate(regions):
                q = sub[sub.target == r]
                if len(q):
                    ax.plot(q.delta.iloc[0], y[i] - off + (ri - (len(regions) - 1) / 2) * 0.035, mk, color=col, ms=4, mfc="none", mew=0.9, alpha=0.8, zorder=2)
            if len(m):
                m = m.iloc[0]; ci_bar(ax, y[i] - off, m.ci_lo, m.ci_hi, col, lw=1.6)
                ax.plot(m.delta, y[i] - off, mk, color=col, ms=6.5, zorder=4)
                sig_ci = np.isfinite(m.ci_lo) and (m.ci_hi < 0 or m.ci_lo > 0)
                if bool(m.get("confirmatory", False)) and np.isfinite(m.get("p_holm", np.nan)):
                    mark = "*" if m.p_holm < 0.05 else ("†" if sig_ci else "")
                else:
                    mark = "†" if sig_ci else ""
                if mark:
                    ax.text(m.delta, y[i] - off + 0.13, mark, ha="center", va="bottom", fontsize=11 if mark == "*" else 9, color="#222222", fontweight="bold")
    ax.axvline(0, color=PAL["ref"], lw=1); ax.set_yticks(y); ax.set_yticklabels([l for _, l in specs]); ax.set_xlabel(xlab); despine(ax)
    if xlim:
        ax.set_xlim(*xlim)
    return y


COND_LEG = [Line2D([], [], marker="o", color=PAL["blue"], ls="-", label="공변량만(AB4, 분할 3) 평균·CI"), Line2D([], [], marker="s", color=PAL["teal"], ls="-", label="정보 없음(블록≥8 주 지역 층화) 평균·CI"),
            Line2D([], [], marker="o", color="#888", ls="none", mfc="none", label="지역별 점(레나·캐나다·러시아 W·E)"),
            Line2D([], [], marker="$*$", color="#333", ls="none", label="* 확인적 가설 Holm p<.05"), Line2D([], [], marker="$†$", color="#333", ls="none", label="† CI가 0 제외(탐색 행·미보정)")]


def load_tests():
    f = H2 / "h_tests_all.csv"
    return pd.read_csv(f) if f.exists() else None


# ====================================================================== fig02 H18·H19 정보 축
def fig02_info_axis():
    T = load_tests()
    if T is None or not (T.source == "h1819").any():
        print("[skip] fig02: h_tests_all(h1819) 없음"); return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.0), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
    ax = axes[0]
    specs = [("H18a_stefan_lst_vs_stefan", "Stefan(LST 도일) - Stefan(기온 도일)  [H18a]"), ("H18x_stefan_soil_vs_stefan", "Stefan(토양 도일) - Stefan(기온)"),
             ("H18K1_stefan_lst_resid25_vs_stefan", "LST 앵커 + CatBoost 잔차 λ=.25 - Stefan(기온)"), ("H18Q1_pseudo_stefan_lst_vs_pseudo_stefan", "유사라벨 공급원 LST - 기온 (직접 ML)"),
             ("H18C_anc_lst_ps_lst_resid25_vs_stefan", "LST 앵커+LST 유사라벨+잔차 - Stefan(기온)")]
    forest(ax, T, specs, xlab="ΔRMSE (cm)  ← 개선 | 악화 →")
    ax.set_title("(a) H18 관측 지표 강제력(MODIS LST): 물리식 사다리", loc="left")
    ax = axes[1]
    xs = [("x25_lst", "x25 + LST 5종"), ("x25_T", "x25 + 지형 수문(TWI 등 6)"), ("x25_W", "x25 + 수체(GSW 3)"), ("x25_V", "x25 + 식생(수목·NDVI 4)"),
          ("x25_H", "x25 + ERA5 확장(토양수분·skt 등 10)"), ("x25_A", "x25 + 전부(A, 23)  [H19b]"), ("x14_A", "x14 + A (SoilGrids 제거)"), ("x16", "x25 - SoilGrids 9 (x16)"), ("x25_A_lst", "x25 + A + LST (47)")]
    specs = [(f"H19p_{x}_vs_x25_pseudo", l) for x, l in xs]
    forest(ax, T, specs, conds=(("covonly", PAL["blue"], "o"),), xlab="ΔRMSE 대 x25 입력 (cm)  ← 개선 | 악화 →", xlim=(-3.2, 6.6))
    ax.set_title("(b) H19 입력 집합: 공변량만(직접 CatBoost + Stefan 유사라벨)", loc="left")
    ax = axes[2]
    specs = [(f"H19d_{x}_vs_x25_direct", l) for x, l in xs]
    forest(ax, T, specs, conds=(("noinfo", PAL["teal"], "s"),), xlab="ΔRMSE 대 x25 입력 (cm)  ← 개선 | 악화 →", xlim=(-3.2, 6.6))
    ax.set_title("(c) H19 입력 집합: 정보 없음(직접 CatBoost)", loc="left")
    fig.legend(handles=COND_LEG, loc="lower center", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("H18·H19 정보 축: 관측 강제력과 신규 공변량이 전이 오차를 줄이는가 (M1 기본 레시피에서 한 요인만 변경)", fontsize=13.5, fontweight="bold", y=1.01)
    fig.text(0.01, -0.07, "평가 셀 = 실측·CCI·토양 도일·신규 공변량 전부 유효(모든 수준 동일). 정보 없음 = 레나·캐나다·러시아 W·E(CI)+러시아 C(점만); 그린란드는 LST 결측으로 제외. 채점 = seed 짝지음·지역 층화 블록 부트스트랩 95% CI(개정 09-21). 근거: data/processed/h2/h_tests_all.csv (source h1819).", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "h2_fig02_info_axis")


# ====================================================================== fig05 H22 학습 목표
def fig05_objective():
    T = load_tests()
    if T is None or not (T.source == "h22").any():
        print("[skip] fig05: h_tests_all(h22) 없음"); return
    meths = [("erm", "ERM(환경 풀링, 22-E0)"), ("irm", "IRMv1 [H22]"), ("vrex", "V-REx [H22]"), ("groupdro", "GroupDRO"), ("dann", "DANN(대상 A 공변량) [H22]"),
             ("regemb", "지역 임베딩 다중 과제"), ("stable", "안정 특징 부분집합(CatBoost)")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    ax = axes[0]; specs = [(f"H22_{m}_resid_lam0.25_vs_stefan", l) for m, l in meths]
    forest(ax, T, specs, xlab="ΔRMSE 대 Stefan (cm)  ← 개선 | 악화 →"); ax.set_title("(a) 잔차 학습(앵커 Stefan, λ=.25) - Stefan", loc="left")
    ax = axes[1]; specs = [(f"H22_{m}_direct_lam1_vs_stefan", l) for m, l in meths]
    forest(ax, T, specs, xlab="ΔRMSE 대 Stefan (cm)  ← 개선 | 악화 →"); ax.set_title("(b) 직접 회귀 - Stefan", loc="left")
    ax = axes[2]; specs = [(f"H22_{m}_resid_lam0.25_vs_erm", l) for m, l in meths if m != "erm"]
    forest(ax, T, specs, xlab="ΔRMSE 대 ERM (cm)"); ax.set_title("(c) 불변 학습 - ERM (같은 MLP·같은 환경, 잔차 λ=.25)", loc="left")
    fig.legend(handles=COND_LEG, loc="lower center", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("H22 학습 목표 함수: 지역 불변 관계를 강제해도 물리식을 넘지 못한다 (입력 x25·M1 MLP 고정, 초모수 중첩 선택)", fontsize=13.5, fontweight="bold", y=1.01)
    fig.text(0.01, -0.07, "환경 = 알래스카 6 공간 fold ∪ {레나·캐나다·러시아 W·E} - 대상. DANN 은 공변량만 조건에서만(대상 A블록 공변량 사용). 정보 없음 = 4지역 CI + 러시아 C 점(그린란드 3셀은 제외). 근거: data/processed/h2/h_tests_all.csv (source h22), h22_rows.csv.", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "h2_fig05_objective")


# ====================================================================== fig06 H23 MAML
def fig06_maml():
    f = H2 / "h23_summary.csv"
    if not f.exists():
        print("[skip] fig06: h23_summary.csv 없음"); return
    S = pd.read_csv(f); Tt = pd.read_csv(H2 / "h23_tests.csv")
    lines = [("shrink_k10", "E 수축 적합(κ=10)", PAL["ref"], "o", "-"), ("finetune", "ERM 사전학습 + 미세조정(λ=.25)", PAL["teal"], "^", "-"),
             ("maml", "MAML 초기화 + 적응(λ=.25)", PAL["blue"], "D", "-"), ("shrink+maml", "E 수축 + MAML 잔차(λ=.25)", PAL["purple"], "v", "-")]
    fig, axes = plt.subplots(1, 5, figsize=(17.0, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.25]))
    for j, r in enumerate(AB4):
        ax = axes[j]; sub = S[S.target == r]
        phys = float(sub.phys.iloc[0]) if len(sub) else np.nan
        ax.axhline(phys, color=PAL["ref"], ls="--", lw=1.2)
        ax.text(0.98, 0.97, f"Stefan(알래스카 E) {phys:.1f} cm", transform=ax.transAxes, fontsize=8.5, va="top", ha="right", color=PAL["ref"])
        for m, lab, col, mk, ls in lines:
            q = sub[(sub.method == m) & ((sub.lam == 0.25) | (sub.lam == 0.0))].sort_values("n")
            if m in ("maml", "finetune", "shrink+maml"):
                n0 = sub[(sub.method == ("maml0" if "maml" in m else "erm0")) & (sub.lam == 0.25)]
                q = pd.concat([n0.assign(n=0), q]) if len(n0) else q
            else:
                q = pd.concat([pd.DataFrame([dict(n=0, rmse_mean=phys)]), q])
            ax.plot(q.n, q.rmse_mean, marker=mk, color=col, ls=ls, lw=1.6, ms=5, label=lab)
        ax.set_title(f"{KR[r]}", loc="left"); ax.set_xlabel("대상 라벨 수 n"); ax.set_xticks([0, 3, 5, 10]); despine(ax)
        if j == 0:
            ax.set_ylabel("RMSE (cm), B블록 평가")
    ax = axes[4]
    mt = Tt[(Tt.target == "MEAN[AB4]") & (Tt.lam == 0.25)]
    for m, lab, col, mk, ls in lines:
        if m in ("shrink_k10",):
            continue
        q = mt[mt.method == m].sort_values("n")
        if len(q):
            ax.plot(q.n, q.delta_vs_shrink, marker=mk, color=col, ls=ls, lw=1.6, ms=5, label=lab)
    ax.axhline(0, color=PAL["ref"], lw=1); ax.set_xlabel("대상 라벨 수 n"); ax.set_ylabel("ΔRMSE 대 E 수축 적합 (cm)"); ax.set_xticks([3, 5, 10]); despine(ax)
    ax.set_title("4지역 평균 Δ(방법 - E 수축) [H23]", loc="left")
    fig.legend(handles=[Line2D([], [], marker=mk, color=col, ls=ls, label=lab) for _, lab, col, mk, ls in lines] + [Line2D([], [], color=PAL["ref"], ls="--", label="Stefan(알래스카 E), n=0")],
               loc="lower center", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("H23 희소 라벨 적응: 메타학습 초기화(MAML)는 라벨 3–10개에서 계수 수축 적합을 넘지 못한다(수준 오차 지역에서 특히 열세)", fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.01, -0.13, "분할 3 × 반복 10 × seed 3, 같은 라벨 추출(S-F 규약). MAML: FOMAML 3,000 에피소드(support 3–10·query 128, 내부 5 step), 적응 10 step. E 재적합(n=3 에서 +2~+5 악화)은 S-F 그림 참조. 근거: data/processed/h2/h23_{summary,tests}.csv", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "h2_fig06_maml_sparse")


# ====================================================================== fig07 H24 관측 설계
STRAT = [("random", "무작위", PAL["ref"], "o"), ("block_rr", "블록 순환", PAL["blue"], "s"), ("kmedoid", "공변량 k-중심(대표점)", PAL["teal"], "D"),
         ("farthest", "최원점(k-center)", PAL["purple"], "^"), ("stefan_strat", "Stefan 예측 분위 층화", PAL["ochre"], "v"), ("geo_strat", "좌표 k-means 층화", PAL["navy"], "P"),
         ("calm_sites", "기존 CALM 사이트만(알래스카)", "#999999", "X")]


def fig07_obs_design():
    f = H2 / "h24_curve.csv"
    if not f.exists():
        print("[skip] fig07: h24_curve.csv 없음"); return
    C = pd.read_csv(f); Tt = pd.read_csv(H2 / "h24_tests.csv")
    regs = AB4 + ["Alaska"]
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.6), sharex=False)
    for i, (method, lam, mlab) in enumerate([("shrink_k10", 0.0, "E 수축 적합(κ=10)"), ("resid", 0.25, "앵커 고정 + CatBoost 잔차(λ=.25)")]):
        for j, r in enumerate(regs):
            ax = axes[i, j]; sub = C[(C.target == r) & (C.method == method) & (C.lam == lam)]
            phys = None
            for st, lab, col, mk in STRAT:
                q = sub[sub.strategy == st].sort_values("n")
                if len(q):
                    ax.plot(q.n, q.rmse_mean, marker=mk, color=col, ls="-", lw=1.5, ms=4.5, label=lab)
                    phys = float((q.rmse_mean - q.d_phys_mean).iloc[0])
            if phys is not None:
                ax.axhline(phys, color=PAL["ref"], ls="--", lw=1.1)
            nmax = int(sub.n.max()) if len(sub) else 80
            ax.set_xscale("log"); ax.set_xticks([v for v in ([3, 5, 10, 20, 40, 80] if r != "Alaska" else [3, 10, 20, 50, 100, 200]) if v <= nmax]); ax.set_xlim(2.6, nmax * 1.15)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.tick_params(labelsize=8.5); despine(ax)
            if i == 0:
                ax.set_title(f"{KR[r]}" + (" (샌드박스: 5지역→알래스카)" if r == "Alaska" else ""), loc="left", fontsize=10.5)
            if j == 0:
                ax.set_ylabel(f"RMSE (cm)\n{mlab}")
            if i == 1:
                ax.set_xlabel("대상 라벨 수 n (로그)")
    fig.legend(handles=[Line2D([], [], marker=mk, color=col, ls="-", label=lab) for _, lab, col, mk in STRAT] + [Line2D([], [], color=PAL["ref"], ls="--", label="물리식(n=0)")],
               loc="lower center", ncol=8, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("H24 관측 설계: 어떤 라벨 n개가 지역을 대표하는가 (선택 규칙 × n, 추정량 고정)", fontsize=13.5, fontweight="bold", y=1.01)
    fig.text(0.01, -0.09, "4지역 실전: 풀 = A블록, 채점 = B블록(분할 3 × 반복 20, 잔차는 반복 10 × seed 3). 알래스카 샌드박스: 공여 5지역 등가중 E 앵커. 근거: data/processed/h2/h24_{curve,tests}.csv", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "h2_fig07_obs_design_curves")
    # 요약 포레스트: Δ(규칙 - 무작위) n ∈ {3, 10, 20}, AB4 평균 + 지역 점
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for k, (method, lam, mlab) in enumerate([("shrink_k10", 0.0, "E 수축 적합(κ=10)"), ("resid", 0.25, "앵커 + 잔차(λ=.25)")]):
        ax = axes[k]; rows = [(st, lab, col, mk) for st, lab, col, mk in STRAT if st not in ("random", "calm_sites")]
        y = np.arange(len(rows))[::-1]
        for j, n in enumerate([3, 10, 20]):
            off = (j - 1) * 0.24; alpha = [0.5, 0.8, 1.0][j]
            for i, (st, lab, col, mk) in enumerate(rows):
                q = Tt[(Tt.method == method) & (Tt.lam == lam) & (Tt.strategy == st) & (Tt.n == n)]
                m = q[q.target == "MEAN[AB4]"]
                for r in AB4:
                    qr = q[q.target == r]
                    if len(qr):
                        ax.plot(qr.delta_vs_random.iloc[0], y[i] - off, mk, color=col, ms=3.5, mfc="none", alpha=alpha)
                if len(m):
                    ax.plot(m.delta_vs_random.iloc[0], y[i] - off, mk, color=col, ms=6.5, alpha=alpha, zorder=3)
                    ax.text(m.delta_vs_random.iloc[0], y[i] - off + 0.1, f"n={n}", fontsize=6.5, ha="center", va="bottom", color=col, alpha=alpha)
        ax.axvline(0, color=PAL["ref"], lw=1); ax.set_yticks(y); ax.set_yticklabels([lab for _, lab, _, _ in rows]); despine(ax)
        ax.set_xlabel("ΔRMSE 대 무작위 선택 (cm)  ← 규칙이 낫다 | 무작위가 낫다 →"); ax.set_title(f"({'ab'[k]}) {mlab}", loc="left")
    fig.suptitle("H24 요약: 선택 규칙 - 무작위 (같은 n·추정량, 4지역 평균 채움·지역 점 빈 마커)", fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.01, -0.05, "행별 세 줄 = n 3(흐림)·10·20(진함, 레나·캐나다만: 러시아 W·E 는 |A|≈15 라 n ≤ 10). 채움 = 4지역 평균(러시아 제외 시 2지역), 빈 마커 = 지역별 Δ(분할 3 × 반복 짝지음). 근거: data/processed/h2/h24_tests.csv", fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "h2_fig07b_obs_design_delta")


# ====================================================================== map02 정보 추가에 따른 셀별 오차 변화
def map02_error_change():
    import glob
    files = sorted(glob.glob(str(M1 / "h1819_shard*_preds.npz")))
    if not files:
        print("[skip] map02: h1819 npz 없음"); return
    P, EV = {}, {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            if k.startswith("g::") and "|noinfo|" not in k and not k.startswith("g::noinfo"):
                pass
            if k.startswith("g::noinfo") or k.startswith("anchor::noinfo"):
                P[k] = z[k]
            elif k.startswith("eval::noinfo"):
                _, tk, fld = k.split("::"); EV.setdefault(tk, {})[fld] = z[k]
    base = pd.read_csv(PROC / "fidelity_base_v3.csv", usecols=["loc_id", "lat", "lon"]).set_index("loc_id")

    def direct(tg, xset):
        tk = f"noinfo|{tg}|0"
        ks = sorted([k for k in P if k.startswith(f"g::{tk}|") and k.endswith(f"|alaska|{xset}|none|none|0.0|catboost_lo|0")])
        return np.mean([P[k] for k in ks], 0) if ks else None
    levels = [("x25_lst", "x25 + LST"), ("x25_A", "x25 + 전부(A)")]
    regs = ["Lena", "Canada"]
    fig, axes = plt.subplots(len(regs), len(levels), figsize=(12.5, 9.5), constrained_layout=True, gridspec_kw=dict(height_ratios=[1.0, 0.55]))
    VMAX = {}                                                   # 지역별 고정 색 범위(수준 간 비교 가능)
    for tg in regs:
        tk = f"noinfo|{tg}|0"; p0 = direct(tg, "x25")
        if p0 is None or tk not in EV:
            continue
        ds = [np.abs(EV[tk]["y"] - direct(tg, x)) - np.abs(EV[tk]["y"] - p0) for x, _ in levels if direct(tg, x) is not None]
        VMAX[tg] = float(max(np.nanpercentile(np.abs(d), 95) for d in ds))
    for i, tg in enumerate(regs):                               # 행 = 지역(같은 종횡비), 열 = 정보 수준
        for j, (x, lab) in enumerate(levels):
            ax = axes[i, j]; tk = f"noinfo|{tg}|0"
            p0, p1 = direct(tg, "x25"), direct(tg, x)
            if p0 is None or p1 is None or tk not in EV:
                ax.set_visible(False); continue
            y = EV[tk]["y"]; loc = EV[tk]["loc_id"]
            d = np.abs(y - p1) - np.abs(y - p0)                # 음수 = 오차 감소
            la, lo = base.loc[loc, "lat"].values, base.loc[loc, "lon"].values
            vmax = VMAX[tg]
            order = np.argsort(np.abs(d))
            sc = ax.scatter(lo[order], la[order], c=d[order], s=8 if len(d) > 1000 else 22, cmap=CMAP.diff, norm=tnorm(-vmax, vmax, 0.0), linewidths=0, rasterized=True)
            ax.set_aspect(1 / np.cos(np.radians(np.nanmean(la)))); ax.set_anchor("N"); ax.xaxis.set_major_formatter(lon_formatter()); despine(ax); ax.tick_params(labelsize=8.5)
            dec = 1 if (np.nanmax(la) - np.nanmin(la)) < 5 else 0
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p, dec=dec: f"{v:.{dec}f}°N"))
            r0, r1 = np.sqrt(np.mean((y - p0) ** 2)), np.sqrt(np.mean((y - p1) ** 2))
            ax.set_title(f"{KR[tg]}: {lab} (RMSE {r0:.1f} → {r1:.1f} cm, 감소 셀 {np.mean(d < 0) * 100:.0f}%)", loc="left", fontsize=10.5)
            if j == len(levels) - 1:
                cb = fig.colorbar(sc, ax=axes[i, :].tolist(), fraction=0.03, pad=0.02, extend="both", shrink=0.9); cb.set_label("|오차| 변화 (cm), 청 = 감소 · 갈 = 증가", fontsize=9)
    fig.suptitle("정보 추가에 따른 셀별 절대 오차 변화(정보 없음, 직접 CatBoost, x25 대비, seed 평균)", fontsize=13.5, fontweight="bold")
    fig.text(0.01, -0.015, "근거: data/processed/m1/h1819_shard*_preds.npz (noinfo, 분할 0). 0 중심 발산(broc), 색 범위 = 지역별 |Δ오차| 95 백분위(초과는 화살표; 지역 간 색 비교 불가). 개선이 특정 블록에 몰리는지, 지역 전체에 퍼지는지를 본다.", fontsize=8, color="#555")
    save(fig, "h2_map02_error_change")


# ====================================================================== map03 관측 설계 지도(선택된 라벨 위치)
def map03_obs_design_map():
    """H24 의 선택 규칙을 같은 난수식으로 재현해 n=20 선택 위치를 지도에 표시(알래스카 샌드박스·레나, 분할 0, 반복 0)."""
    sys.path.insert(0, str(ROOT / "scripts" / "3_deep_learning"))
    from polar.m1_core import load_base, half_split_blocks, INPUT_SETS
    DF = load_base(PROC); FE = INPUT_SETS["x25"]
    strategies = ["random", "block_rr", "kmedoid", "geo_strat"]
    regs = [("Alaska", 20), ("Lena", 20)]
    fig, axes = plt.subplots(len(regs), len(strategies), figsize=(16.0, 7.6), constrained_layout=True)
    from sklearn.cluster import KMeans
    for i, (tg, n) in enumerate(regs):
        t_idx = np.where(DF.macro.values == tg)[0]; A_idx, B_idx = half_split_blocks(DF, t_idx, 0)
        A = DF.iloc[A_idx]; B = DF.iloc[B_idx]
        Xa = A[FE].values.astype(float); med = np.nanmedian(Xa, 0); med = np.where(np.isfinite(med), med, 0.0)
        Xa = np.where(np.isnan(Xa), med, Xa); Z = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-6)
        coords = np.c_[A.lat.values, A.lon.values * np.cos(np.radians(A.lat.values))]
        blocks = A.block.values
        for j, st in enumerate(strategies):
            rng = np.random.RandomState(7_919 * 0 + 1_009 * n + 31 * ["random", "block_rr", "kmedoid", "farthest", "stefan_strat", "geo_strat", "calm_sites"].index(st) + 0)
            if st == "random":
                sel = rng.choice(len(A), n, replace=False)
            elif st == "block_rr":
                ub = np.unique(blocks); order = rng.permutation(ub); pools = {b: list(rng.permutation(np.where(blocks == b)[0])) for b in ub}; sel = []
                while len(sel) < n:
                    for b in order:
                        if pools[b]:
                            sel.append(pools[b].pop())
                            if len(sel) >= n:
                                break
                sel = np.array(sel)
            else:
                X = Z if st == "kmedoid" else coords
                km = KMeans(n_clusters=n, n_init=3, random_state=int(rng.randint(1 << 30))).fit(X); sel = []
                for c in km.cluster_centers_:
                    dd = ((X - c) ** 2).sum(1); dd[sel] = np.inf; sel.append(int(np.argmin(dd)))
                sel = np.array(sel)
            ax = axes[i, j]
            ax.scatter(B.lon, B.lat, s=3, color="#c9d3df", linewidths=0, rasterized=True, label="B블록(채점)")
            ax.scatter(A.lon, A.lat, s=3, color="#8fa8c8", linewidths=0, rasterized=True, label="A블록(후보 풀)")
            ax.scatter(A.lon.values[sel], A.lat.values[sel], s=42, marker="*", color=PAL["ochre"], edgecolor="black", linewidths=0.4, zorder=5, label=f"선택 라벨 n={n}")
            ax.set_aspect(1 / np.cos(np.radians(A.lat.mean()))); ax.xaxis.set_major_formatter(lon_formatter()); despine(ax); ax.tick_params(labelsize=8.5)
            dec = 1 if (A.lat.max() - A.lat.min()) < 5 else 0
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p, dec=dec: f"{v:.{dec}f}°N"))
            nb_sel = len(np.unique(blocks[sel]))
            ax.set_title(f"{KR[tg]} · {dict(STRAT_LAB)[st]} (선택 블록 {nb_sel}/{len(np.unique(blocks))})", loc="left", fontsize=10.5)
            if i == 0 and j == 0:
                ax.legend(loc="lower left", fontsize=8, markerscale=1.5)
    fig.suptitle("H24 관측 설계 지도: 선택 규칙별 라벨 위치(n=20, 분할 0·반복 0). 블록 순환·k-중심은 공간·공변량 대표성을 확보한다", fontsize=13.5, fontweight="bold")
    fig.text(0.01, -0.015, "A/B = 0.5° 블록 2분할(E1 규약). 별 = 선택된 라벨 셀. 근거: scripts/3_deep_learning/h24_obs_design.py 의 선택 규칙과 같은 난수식.", fontsize=8, color="#555")
    save(fig, "h2_map03_obs_design_map")


STRAT_LAB = [(st, lab) for st, lab, _, _ in STRAT]


# ====================================================================== fig08 종합 포레스트(확인적 가설)
def fig08_summary():
    T = load_tests()
    if T is None:
        print("[skip] fig08: h_tests_all 없음"); return
    M = T[T.confirmatory & T.target.astype(str).str.startswith("MEAN")].copy()
    rows = []
    label = {"H18a_stefan_lst_vs_stefan": "H18a Stefan(LST) - Stefan(기온)", "H19p_x25_lst_vs_x25_pseudo": "H18b ML(x25+LST) - ML(x25), 공변량만",
             "H19p_x25_A_vs_x25_pseudo": "H19b ML(x25+A) - ML(x25), 공변량만", "H19d_x25_A_vs_x25_direct": "H19b ML(x25+A) - ML(x25), 정보 없음",
             "H20_cci_blockE_vs_stefan": "H20 CCI 블록 계수 - Stefan (AB4)", "H21_nested_vs_EAK": "H21 계수 대여(중첩) - Stefan (AB4)",
             "H22_irm_resid_lam0.25_vs_stefan": "H22 IRM 잔차 - Stefan", "H22_vrex_resid_lam0.25_vs_stefan": "H22 V-REx 잔차 - Stefan",
             "H22_dann_resid_lam0.25_vs_stefan": "H22 DANN 잔차 - Stefan"}
    for tid, lab in label.items():
        for cond in ("covonly", "noinfo"):
            q = M[(M.test == tid + "|AB4")] if tid.startswith(("H20", "H21")) else M[(M.test == tid) & (M.cond == cond)]
            if tid.startswith(("H20", "H21")) and cond == "covonly":
                continue
            if len(q):
                q = q.iloc[0]; rows.append(dict(lab=lab, cond=cond, delta=q.delta, lo=q.ci_lo, hi=q.ci_hi, bm=q.block_majority))
    f23 = H2 / "h23_tests.csv"; f24 = H2 / "h24_tests.csv"
    if f23.exists():
        t = pd.read_csv(f23); q = t[(t.target == "MEAN[AB4]") & (t.method == "maml") & (t.lam == 0.25) & (t.n == 3)]
        if len(q):
            rows.append(dict(lab="H23 MAML 적응(n=3) - E 수축 적합 [기준: 수축 적합]", cond="covonly", delta=float(q.delta_vs_shrink.iloc[0]), lo=np.nan, hi=np.nan, bm=np.nan))
    if f24.exists():
        t = pd.read_csv(f24)
        for st, lab in (("kmedoid", "k-중심"), ("block_rr", "블록 순환")):
            q = t[(t.target == "MEAN[AB4]") & (t.method == "shrink_k10") & (t.strategy == st) & (t.n == 3)]
            if len(q):
                rows.append(dict(lab=f"H24 {lab} 선택(n=3, E 수축) - 무작위 [기준: 무작위]", cond="covonly", delta=float(q.delta_vs_random.iloc[0]), lo=np.nan, hi=np.nan, bm=np.nan))
    R = pd.DataFrame(rows)
    if not len(R):
        print("[skip] fig08: 행 없음"); return
    labs = list(dict.fromkeys(R.lab)); y = np.arange(len(labs))[::-1]
    fig, ax = plt.subplots(figsize=(12.0, 0.46 * len(labs) + 2.0))
    xl, xr = -4.2, 3.2
    for j, (cond, col, mk) in enumerate([("covonly", PAL["blue"], "o"), ("noinfo", PAL["teal"], "s")]):
        off = (j - 0.5) * 0.26
        for i, lab in enumerate(labs):
            q = R[(R.lab == lab) & (R.cond == cond)]
            if not len(q):
                continue
            q = q.iloc[0]
            ci_bar(ax, y[i] - off, max(q.lo, xl) if np.isfinite(q.lo) else q.lo, min(q.hi, xr) if np.isfinite(q.hi) else q.hi, col, lw=1.6)
            ax.plot(q.delta, y[i] - off, mk, color=col, ms=6.5, zorder=3)
            txt = f"{q.delta:+.2f} [{q.lo:+.2f}, {q.hi:+.2f}]" if np.isfinite(q.lo) else f"{q.delta:+.2f}"
            ax.text(xr + 0.1, y[i] - off, txt, fontsize=8, color=col, va="center", ha="left", clip_on=False)
    ax.axvline(0, color=PAL["ref"], lw=1); ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9); despine(ax); ax.set_xlim(xl, xr)
    ax.set_xlabel("ΔRMSE (cm)  ← 기준보다 개선 | 악화 →")
    ax.legend(handles=COND_LEG[:2], loc="upper left", fontsize=8.5)
    ax.set_title("확인적 가설 H18–H24 종합: 라벨 없는 전이에서 물리식을 넘은 요인은 없고, 라벨 3개의 선택 규칙만 유의한 이득", loc="left", fontsize=12)
    fig.text(0.01, -0.05, "채움 마커 = 지역 평균 Δ, 선 = 층화 블록 부트스트랩 95% CI(개정 09-21), 오른쪽 수치 = Δ [CI]. H20·H21 은 AB4 평균(CI 산출 지역과 일치). H23·H24 는 기준이 물리식이 아니므로 CI 생략(지역별 CI 는 h23_tests·h24_tests).",
             fontsize=8, color="#555")
    fig.tight_layout()
    save(fig, "h2_fig08_summary_forest")


REG = {"fig01": fig01_collapse, "fig02": fig02_info_axis, "fig03": fig03_blockE, "fig04": fig04_coef_borrow, "fig05": fig05_objective, "fig06": fig06_maml,
       "fig07": fig07_obs_design, "fig08": fig08_summary, "map01": map01_new_covariates, "map02": map02_error_change, "map03": map03_obs_design_map}
only = [s for s in args.only.split(",") if s]
for k, fn in REG.items():
    if only and k not in only:
        continue
    try:
        fn()
    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"[skip] {k}: {type(e).__name__} {e}")
print("exports:", json.dumps(EXPORTS, ensure_ascii=False))
