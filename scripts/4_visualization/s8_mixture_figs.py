"""S8 시각화: mixture-of-physics experts 진단 4종.

1) s8_dominant_expert_map : 셀별 지배 전문가(범주형, 색+마커 병기, mask_ocean)
2) s8_weight_error       : 게이트 확신(w_max)·전문가 가중치와 |오차|의 관계
3) s8_env_selection      : 환경변수 구간별 전문가 선택 분포(스택 막대, 해치 병기)
4) s8_gate_compare       : mixture vs Stefan vs 균등평균 vs oracle(진단 하한) + 부트스트랩 CI

실행: python scripts/4_visualization/s8_mixture_figs.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar
from polar.geomap import (make_ax, mask_ocean, to_proj, add_scalebar, add_inset_locator,
                          circular_boundary, _proj, ALASKA, PANARCTIC)

plt = use_polar()
PROC = C.PROCESSED
OUT = C.FIGURES / "s8_mixture"
OUT.mkdir(parents=True, exist_ok=True)

res = pd.read_csv(PROC / "s8_mixture_results.csv")
oof = pd.read_csv(PROC / "s8_mixture_oof.csv")
meta = json.loads((PROC / "s8_mixture_meta.json").read_text())

EXPERTS = ["p1_stefan", "p2_edaphic", "p3_ttop", "p4_kudryavtsev", "p5_lambda"]
LBL = {"p1_stefan": "p1 Stefan 기본", "p2_edaphic": "p2 edaphic", "p3_ttop": "p3 TTOP",
       "p4_kudryavtsev": "p4 Kudryavtsev", "p5_lambda": "p5 λ보정"}
COL = {"p1_stefan": "#14508c", "p2_edaphic": "#3e8f6b", "p3_ttop": "#8571b8",
       "p4_kudryavtsev": "#46a3b4", "p5_lambda": "#7c7f36"}
MRK = {"p1_stefan": "o", "p2_edaphic": "s", "p3_ttop": "^",
       "p4_kudryavtsev": "D", "p5_lambda": "v"}
HAT = {"p1_stefan": "", "p2_edaphic": "///", "p3_ttop": "xx",
       "p4_kudryavtsev": "..", "p5_lambda": "\\\\\\"}
M_LBL = {"stefan": "Stefan 단일(p1)", "uniform": "균등평균(5종)", "mix_logit": "mixture(logit)",
         "mix_mlp": "mixture(MLP)", "mix_mlp_seedmean": "mixture(MLP seed평균)",
         "oracle": "oracle(진단 하한)"}
M_COL = {"stefan": "#14508c", "uniform": "#8a8fa3", "mix_logit": "#3e8f6b",
         "mix_mlp": "#46a3b4", "mix_mlp_seedmean": "#46a3b4", "oracle": "#b5b5b5"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


def cat_scatter(ax, sub, s=5):
    """지배 전문가 범주 산점(색+마커 병기). mask_ocean 위 zorder로 그린다.

    overplotting 완화: (i) 마커를 축소·반투명(alpha) 처리해 밀집 군집에서도 하부
    범주가 비쳐 보이게 하고, (ii) 개수 오름차순으로 그려 최다 범주를 상단(나중)에
    배치한다. 이로써 최다 전문가(예 LORO의 p5 λ)가 소수 파란 p1에 가려지던 문제를
    해소한다. 범례 표본 수는 범주별 실제 개수로 표기한다.
    """
    counts = {e: int((sub.dominant == e).sum()) for e in EXPERTS}
    draw_order = sorted([e for e in EXPERTS if counts[e]], key=lambda e: counts[e])
    for zi, e in enumerate(draw_order):
        g = sub[sub.dominant == e]
        x, yy = to_proj(ax, g.lon.values, g.lat.values)
        ax.scatter(x, yy, s=s * 0.8, c=COL[e], marker=MRK[e], linewidths=0,
                   alpha=0.5, zorder=4 + zi)
    return counts


def expert_legend(ax, counts, **kw):
    """전문가 순서(p1→p5) 불투명 프록시 범례. 산점 alpha와 무관하게 색 선명."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=MRK[e], color="none", markerfacecolor=COL[e],
                      markeredgecolor="none", markersize=7, linestyle="none",
                      label=f"{LBL[e]} ({counts[e]:,})")
               for e in EXPERTS if counts.get(e)]
    ax.legend(handles=handles, **kw)


# ============ 1) 지배 전문가 지도 ============
loro = oof[oof["eval"] == "loro"]
ak = oof[oof["eval"] == "indomain_ak"]
fig = plt.figure(figsize=(13.5, 6.2))
ax = fig.add_subplot(1, 2, 1, projection=_proj(PANARCTIC))
make_ax(PANARCTIC, ax=ax, fig=fig, title="(a) LORO 전이: 게이트 지배 전문가 (argmax w)")
circular_boundary(ax)
mask_ocean(ax)
c1 = cat_scatter(ax, loro, s=5)
expert_legend(ax, c1, loc="lower left", fontsize=8, framealpha=0.9)
ax2 = fig.add_subplot(1, 2, 2, projection=_proj(ALASKA))
make_ax(ALASKA, ax=ax2, fig=fig, title="(b) 알래스카 in-domain(공간블록 OOF): 지배 전문가")
mask_ocean(ax2)
c2 = cat_scatter(ax2, ak, s=7)
add_scalebar(ax2)
add_inset_locator(fig, ax2, ALASKA)
expert_legend(ax2, c2, loc="lower right", fontsize=8, framealpha=0.9)
fig.suptitle("S8 셀별 지배 물리 전문가 (다항로지스틱 게이트, 색·마커 병기)", fontsize=13, y=0.99)
save(fig, "s8_dominant_expert_map")

# ============ 2) 가중치-|오차| 관계 ============
loro_v = loro[np.isfinite(loro.mix_logit_pred) & np.isfinite(loro.alt_cm)].copy()
loro_v["abserr"] = (loro_v.alt_cm - loro_v.mix_logit_pred).abs()
loro_v["wmax"] = loro_v[[f"w_{e}" for e in EXPERTS]].max(axis=1)
corr = res[res.part == "D_corr"]
rho_pool = corr[(corr.metric == "wmax_vs_abserr") & (corr.region == "POOLED")].spearman.iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
axl = axes[0]
edges = np.nanquantile(loro_v.wmax, np.linspace(0, 1, 11))
edges = np.unique(edges)
bid = np.clip(np.digitize(loro_v.wmax, edges[1:-1]), 0, len(edges) - 2)
REG_COL = {"전체(LORO)": "#14508c", "Alaska": "#46a3b4", "Canada": "#8a8fa3", "Lena": "#3e8f6b"}
for name, g in [("전체(LORO)", loro_v)] + list(loro_v.groupby("macro")):
    gb = g.groupby(np.clip(np.digitize(g.wmax, edges[1:-1]), 0, len(edges) - 2))
    xs = gb.wmax.mean(); ys = gb.abserr.mean()
    kw = dict(lw=2.2, zorder=5) if name == "전체(LORO)" else dict(lw=1.2, alpha=0.8)
    axl.plot(xs, ys, "-o", ms=4, color=REG_COL.get(name, "0.5"), label=f"{name}", **kw)
axl.set_xlabel("게이트 최대 가중 $w_{max}$ (무차원)")
axl.set_ylabel("|오차| 평균 (cm)")
axl.set_title(f"(a) 게이트 확신 vs |오차| · 십분위 (ρ={rho_pool:+.2f})")
axl.legend(fontsize=8)
axr = axes[1]
ck = corr[corr.metric == "w_k_vs_abserr_k"].set_index("expert").reindex(EXPERTS)
xs = np.arange(len(EXPERTS))
for i, e in enumerate(EXPERTS):
    v = ck.loc[e, "spearman"]
    if np.isfinite(v):
        axr.bar(i, v, color=COL[e], hatch=HAT[e], edgecolor="white", width=0.62)
        axr.text(i, v + (0.02 if v >= 0 else -0.05), f"{v:+.2f}", ha="center", fontsize=8.5)
    else:
        axr.text(i, 0.02, "정의불가\n(가중 상수)", ha="center", fontsize=7.5, color="0.4")
axr.axhline(0, color="0.4", lw=0.8)
axr.set_xticks(xs)
axr.set_xticklabels([LBL[e].replace(" ", "\n", 1) for e in EXPERTS], fontsize=8)
axr.set_ylabel("스피어만 ρ (무차원)")
axr.set_title("(b) 전문가별 corr(가중 $w_k$, |해당 전문가 오차|)")
fig.suptitle("S8 게이트 가중치와 |오차|의 상관 (LORO OOF, logit 게이트)", fontsize=12.5, y=1.02)
save(fig, "s8_weight_error")

# ============ 3) 환경 구간별 선택 분포 ============
env = res[res.part == "E_envbins"].copy()
vars_ = list(dict.fromkeys(env["var"]))


def _bin_decimals(edges):
    """변수별 최소 소수 자릿수(0-3): 모든 구간 경계가 서로 구분되도록 선택.

    음수·소수 변수(예 SWE 0.02-0.19 m)에서 자릿수 부족으로 라벨이 겹치는 것을 막는다.
    """
    for nd in (1, 2, 3):
        if len({round(float(x), nd) for x in edges}) == len(edges):
            return nd
    return 3


def _merge_degenerate(g, nd):
    """1자리 이상 반올림 후에도 폭 0으로 보이는(라벨 lo==hi) 오분위 경계 중복 구간을
    이웃 구간에 병합한다(표본수 n 가중 평균). 재실험 없이 표시 단계에서만 정정한다.
    """
    rows = g.sort_values("bin").to_dict("records")
    out = []
    for r in rows:
        if out and round(r["bin_lo"], nd) == round(r["bin_hi"], nd):
            p = out[-1]                                    # 폭 0 구간 → 직전 구간에 병합
            ntot = p["n"] + r["n"]
            for e in EXPERTS:
                p[f"w_{e}"] = (p[f"w_{e}"] * p["n"] + r[f"w_{e}"] * r["n"]) / ntot
            p["bin_hi"], p["n"] = r["bin_hi"], ntot
        else:
            out.append(dict(r))
    return pd.DataFrame(out)


fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6))
for axi, var in zip(axes.ravel(), vars_):
    g0 = env[env["var"] == var].sort_values("bin")
    nd = _bin_decimals(list(g0.bin_lo) + [g0.bin_hi.iloc[-1]])
    g = _merge_degenerate(g0, nd)
    bottom = np.zeros(len(g))
    for e in EXPERTS:
        vals = g[f"w_{e}"].values
        axi.bar(np.arange(len(g)), vals, bottom=bottom, color=COL[e], hatch=HAT[e],
                edgecolor="white", linewidth=0.5, width=0.8, label=LBL[e])
        bottom += vals

    def _fmt(v, nd=nd):
        """축 눈금 수치 포맷: 지수표기 회피(예 7.76e+03 → 7,760), 소수 자릿수 통일."""
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.{nd}f}"
    # 범위 구분자는 '~'(음수·연결자 대시 오독 방지). 예: -11.7~-8.5
    axi.set_xticks(np.arange(len(g)))
    axi.set_xticklabels([f"{_fmt(lo)}\n~{_fmt(hi)}" for lo, hi in zip(g.bin_lo, g.bin_hi)],
                        fontsize=7.5)
    axi.set_xlabel(f"{g.var_label.iloc[0]} ({g.unit.iloc[0]}) 오분위", fontsize=9)
    axi.set_ylabel("평균 게이트 가중 (무차원)", fontsize=9)
    axi.set_ylim(0, 1)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.015), fontsize=9)
fig.suptitle("S8 환경변수 구간별 전문가 선택 분포 (LORO OOF · logit 게이트 평균 가중)",
             fontsize=12.5, y=1.0)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
save(fig, "s8_env_selection")

# ============ 4) 게이트 성능 비교 + 부트스트랩 CI ============
gate = res[res.cv == "LORO_gate"].set_index("model").rmse_cm
loro_r = res[(res.part == "B_loro") & (res.cv == "LORO")]
boot = res[res.part == "C_boot"]
order = ["stefan", "uniform", "mix_logit", "mix_mlp", "oracle"]
regions = ["Alaska", "Lena", "Canada"]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
axa = axes[0]
for i, m in enumerate(order):
    hatch = "//" if m == "oracle" else ""
    axa.bar(i, gate[m], color=M_COL[m], hatch=hatch, edgecolor="white", width=0.62)
    axa.text(i, gate[m] + 0.6, f"{gate[m]:.1f}", ha="center", fontsize=9)
seed_g = loro_r[(loro_r.model == "mix_mlp") & (loro_r.seed >= 0)].groupby("seed") \
    .apply(lambda g: g.groupby("region").rmse_cm.mean().mean())
axa.scatter([order.index("mix_mlp") + 0.24] * len(seed_g), seed_g.values, s=14, c="k", zorder=5,
            label="MLP seed별")
axa.axhline(meta["s2_reference_gate_cm"], color="0.35", lw=1.1, ls="--",
            label=f"S2 참조 {meta['s2_reference_gate_cm']:.2f}cm")
axa.set_xticks(range(len(order)))
axa.set_xticklabels([M_LBL[m].replace("(", "\n(") for m in order], fontsize=8)
axa.set_ylabel("LORO 게이트 RMSE (cm, 지역 비가중평균)")
axa.set_title("(a) LORO 게이트: mixture는 Stefan 미달")
axa.legend(fontsize=8)
axb = axes[1]
w = 0.16
for j, m in enumerate(order):
    sub = loro_r[loro_r.model == m]
    vals = [sub[(sub.region == r) & (sub.seed.isin([0, -1]))].rmse_cm.mean() if m != "mix_mlp"
            else sub[sub.region == r].rmse_cm.mean() for r in regions]
    axb.bar(np.arange(len(regions)) + (j - 2) * w, vals, width=w, color=M_COL[m],
            hatch="//" if m == "oracle" else "", edgecolor="white", label=M_LBL[m])
axb.set_xticks(range(len(regions)))
axb.set_xticklabels(regions)
axb.set_ylabel("LORO RMSE (cm)")
axb.set_title("(b) 지역별 전이 RMSE")
axb.legend(fontsize=7.5)
axc = axes[2]
off = {"mix_logit": -0.18, "mix_mlp_seedmean": 0.0, "uniform": 0.18}
for m, o in off.items():
    b = boot[(boot.model == m) & (boot.cv == "LORO")].set_index("region").reindex(regions)
    axc.errorbar(np.arange(len(regions)) + o, b.delta_rmse,
                 yerr=[b.delta_rmse - b.ci_lo, b.ci_hi - b.delta_rmse],
                 fmt="o", ms=5, capsize=3, lw=1.4, color=M_COL[m], label=M_LBL[m])
bak = boot[boot.region == "Alaska_indomain"]
if len(bak):
    axc.errorbar([len(regions)], bak.delta_rmse, ms=5, capsize=3, lw=1.4, fmt="s",
                 yerr=[bak.delta_rmse - bak.ci_lo, bak.ci_hi - bak.delta_rmse],
                 color=M_COL["mix_logit"], label="AK in-domain(logit)")
axc.axhline(0, color="0.4", lw=0.8, ls="--")
axc.set_xticks(range(len(regions) + 1))
axc.set_xticklabels(regions + ["AK\nin-domain"], fontsize=9)
axc.set_ylabel("ΔRMSE = Stefan - 모델 (cm, 양수=개선)")
axc.set_title("(c) 블록부트스트랩 95% CI")
axc.legend(fontsize=7.5)
fig.suptitle("S8 mixture-of-physics 성능: 단일 Stefan 대비 (oracle=test 라벨 셀별 최적, 모델 아님)",
             fontsize=12.5, y=1.02)
save(fig, "s8_gate_compare")

print("[done] S8 시각화 4종 완료")
