"""G6 결론 · G7 의의와 활용 (덱 전용 쪽, 12 × 3.70 in = 3600 × 1110 px, 300 dpi, RGB 흰 바탕). 설계 지침 v9 3절(항 25–27).

v9 재설계(2026-10-15): v8 의 세 열(원이 많은 물리 잔차 그래프 · 열 머리 · 제품 막대)을 폐기하고, v7 보고 20–21쪽처럼 넓게 둔다.
그림 안에는 문장·제목이 없다(덱이 그림 아래 주황 상자를 얹는다). 라벨은 영문(FreeSans; F 계열 결과 그림과 같은 방법 이름).

G6_conclusion: 패널 두 개
  a 관측 연도 기후 입력으로 만든 해마다 다른 지도: 알래스카 ALT 편차(그 해 r1 − 2011–2024 평균 r1)를 2011–2024 가운데 지역 평균 편차가
    가장 대비되는 두 해(최소·최대)에 대해 그린다. 연도는 자료에서 고른다(pred_<year>.csv.gz − mean_2011_2024.csv.gz 의 셀 평균).
    색 범위 = F7 편차 지도와 같은 대칭 ±MV.sym_max(모든 연도 |편차| 99 백분위, 5 cm 단위; F7 캐시 래스터가 있으면 그것으로) = ±15 cm, broc(MV.CM_DIFF),
    250 m 음영 격자 위 1 km 셀 중심값의 표시용 쌍선형 보간(v7_f6_maps.render smooth, 틈 메우기 FILL_MAX), 지도 없음 육지 = 토큰 map_base.no_data_diverging.
    적용 범위 밖 표시는 하지 않는다(라벨은 연도만; 적용 범위는 덱 R08·R10 쪽 몫). 축척 막대는 첫 지도 틀 안.
  b 새 지역 전이 안전성: 지역 홀드아웃에서 직접 ML(D0) · 물리 + 잔차 ML(R1, λ 0.5)의 Stefan 대비 ΔRMSE(점 + 95 % 블록 구간), 세 지역.
    runs/core_v6/deltas_vs_ref.csv (scheme region, forcing year_obs, featset F8TS, learner catboost, ref P1).
    지역 홀드아웃에서는 대상 지역 라벨이 없으므로 재보정 Stefan(P1) = 원천 계수 Stefan(P0): summary_metrics.csv 의 P0·P1 RMSE 가 세 지역 모두 같음을 실행 때 확인하고
    (다르면 중단), y축을 'ΔRMSE vs Stefan, region hold-out (cm)' 로 적는다.
G7_significance: 그림 하나(지역별 패널 셋)
  새 지역 전이 + 대상 지역 라벨 n 개: RMSE 대 n(0, 25, 50, 100, 250, 500). Stefan(원천 계수) P0 · Stefan(n 개로 재보정) P1n · 직접 ML(원천 + n) D0S ·
  물리 + 잔차 ML(원천 + n) R1S. runs/ladder_v6/ladder_metrics.csv (exp TL). x 는 로그 꼴, n = 0 은 왼쪽(x = 10 자리), n = 25 에 옅은 세로선.
  y 범위는 패널마다 자료에서(5 cm 눈금). 범위를 크게 넘는 값(레나 n = 0 직접 ML)은 위 가장자리의 빈 표식과 값 글자로 적는다.
글자(토큰 font.slide): 축 제목 14 pt, 눈금 12 pt, 범례·직접 라벨 13 pt, 패널 문자 14 pt 굵게. 색: 지역 S.REGION_HEX, 방법 S.METHOD_HEX(P0·P1·D0·R1; D0 파선).
산출: deck/assets/v8/G6_conclusion.png · G7_significance.png(RGB, 3600 × 1110) + pdf + <stem>_spec.json(배치·수치·출처·QA).
실행(CPU 만): CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v8_g6_g7_conclusion.py [--only G6|G7]
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU 사용 금지(CPU 전용)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.colors import TwoSlopeNorm, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization"))
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3"))
sys.path.insert(0, str(ROOT / "src"))
import method_figs_v4 as M  # noqa: E402  (canvas, T, sub_axes, qa_and_save, setup_fonts)
import polar.style_v5 as S  # noqa: E402  (REGION_HEX, REGION_EN, METHOD_HEX, METHOD_LS, TOK)
import maps_alt_v3 as MV  # noqa: E402  (CM_DIFF, sym_max, extend_of)
import v7_f6_maps as F6  # noqa: E402  (cell_table, cell_values, Base, render, frame, coast, graticule, region_limits, Med, CACHE)

M.OUT = ROOT / "deck" / "assets" / "v8"
PROC = ROOT / "data" / "processed" / "v5"
RUN = PROC / "runs" / "core_v6"
LADDER = PROC / "runs" / "ladder_v6" / "ladder_metrics.csv"
MAPS = PROC / "maps" / "Alaska"
F7_CACHE = ROOT / "outputs" / "figures" / "paper" / "v7" / "_cache" / "f7_alaska_anom_1500x1344.npz"

# ================================================================ 크기 · 글자 · 색
W, H = 12.0, 3.70
FS_T, FS_AX, FS_K, FS_L = 13.0, 14.0, 12.0, 14.0        # 범례·직접 라벨 · 축 제목 · 눈금 · 패널 문자(pt)
REGIONS = ["Alaska", "Lena", "Canada"]
REG_HEX = S.REGION_HEX
REG_EN = S.REGION_EN                                     # Alaska · Lena Delta · Canada (NW)
MH = S.METHOD_HEX; DASH = S.METHOD_LS["D0"]              # D0 파선 (0, (3.5, 1.8))
NAME = {"D0": "Direct ML", "R1": "Physics + residual ML",
        "P0": "Stefan (source coeff.)", "P1n": "Stefan (recalibrated on n)", "D0S": "Direct ML (source + n)", "R1S": "Physics + residual ML (source + n)"}
STYLE = {"P0": (MH["P0"], "-"), "P1n": (MH["P1"], "-"), "D0S": (MH["D0"], DASH), "R1S": (MH["R1"], "-")}
NO_DATA_DIV = S.TOK["color"]["map_base"]["slide"]["no_data_diverging"]      # 발산 색표 지도의 '지도 없음' 육지색(#D0D0D0)
ZERO = S.NEUTRAL["zero_line"]
N25_LINE = "#C8C8C8"                                     # n = 25 표시선(옅은 회색, 가는 선)


def minus(s: str) -> str:
    return s.replace("-", "−")


def bbox_in(fig, art):
    bb = art.get_window_extent(fig.canvas.get_renderer())
    return (bb.x0 / fig.dpi, bb.y0 / fig.dpi, bb.x1 / fig.dpi, bb.y1 / fig.dpi)


def chart_axes(fig, x, y, w, h):
    a = M.sub_axes(fig, x, y, w, h)
    for k in ("top", "right"):
        a.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        a.spines[k].set_linewidth(0.9); a.spines[k].set_edgecolor(ZERO)
    a.tick_params(labelsize=FS_K, length=3.5, width=0.9, pad=2.5, colors=M.INK)
    return a


def letter(ax, x, y, s):
    """패널 문자(14 pt 굵게), (x, y) = 글자 왼쪽 위(인치)."""
    return M.T(ax, x, y, s, size=FS_L, weight="bold", ha="left", va="top")


def finish(fig, stem, spec, extra_ok=True):
    """QA(겹침·밖·작은 글자·글자–선·글자–축) → 저장(300 dpi) → RGB 평탄화 → 화소 크기·잉크 여백 → spec.json."""
    fig.canvas.draw()
    small = [[t.get_text()[:30], float(t.get_fontsize())] for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < FS_K - 1e-6]
    rec = M.qa_and_save(fig, stem, allow_overlap=())
    from PIL import Image
    png_s, pdf_s = M.OUT / f"{stem}_slide.png", M.OUT / f"{stem}_slide.pdf"
    png, pdf = M.OUT / f"{stem}.png", M.OUT / f"{stem}.pdf"
    with Image.open(png_s) as im:
        bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        bg.save(png, dpi=(M.DPI, M.DPI)); px = list(im.size)
    png_s.unlink(); pdf_s.replace(pdf)
    with Image.open(png) as im:
        a = np.asarray(im.convert("L")); ys, xs = np.where(a < 250)
        ink = dict(top=int(ys.min()), bottom=int(a.shape[0] - 1 - ys.max()), left=int(xs.min()), right=int(a.shape[1] - 1 - xs.max()))
    spec["qa"] = {k: rec[k] for k in ("min_font_pt", "texts", "outside", "overlaps", "small", "text_line", "text_axes", "text_patch")}
    spec["qa"]["png_px"] = px; spec["qa"]["png_px_ok"] = bool(px == [int(W * M.DPI), int(H * M.DPI)])
    spec["qa"]["ink_margin_px"] = ink
    spec["qa"]["text_below_12pt"] = small
    spec["qa"]["bare_sqrt_glyph_texts"] = [t.get_text()[:40] for t in fig.findobj(Text) if "√" in t.get_text()]
    ok = not (rec["outside"] or rec["overlaps"] or small or rec["text_line"] or rec["text_axes"] or rec["text_patch"]) and spec["qa"]["png_px_ok"] and extra_ok
    print(f"[{stem}]", "QA OK" if ok else "QA CHECK", "· 화소", px, "· 잉크 여백 px", ink, flush=True)
    (M.OUT / f"{stem}_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1, default=float))
    print("출력:", png, pdf, M.OUT / f"{stem}_spec.json", flush=True)
    return rec


# ================================================================ 자료
def anomaly_years(spec):
    """2011–2024 연도별 지역 평균 편차(셀마다 그 해 r1 − 평균 r1 의 평균)와 가장 대비되는 두 해(최소·최대). 편차 셀 값(셀 표 순서)을 함께 돌려준다."""
    mean = pd.read_csv(MAPS / "mean_2011_2024.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1
    spec["sources"]["map_mean"] = str((MAPS / "mean_2011_2024.csv.gz").relative_to(ROOT))
    years = list(range(2011, 2025)); m = {}; preds = {}
    for y in years:
        p = pd.read_csv(MAPS / f"pred_{y}.csv.gz", dtype={"cell_id": str}, usecols=["cell_id", "r1"]).set_index("cell_id").r1
        d = p - mean.reindex(p.index)
        m[y] = float(np.nanmean(d.values)); preds[y] = d
    y_lo = min(m, key=m.get); y_hi = max(m, key=m.get)
    spec["sources"]["map_years"] = f"{MAPS.relative_to(ROOT)}/pred_<year>.csv.gz (r1)"
    spec["numbers"]["regional_mean_anomaly_cm"] = {int(k): round(v, 2) for k, v in m.items()}
    spec["choices"]["years"] = dict(rule="argmin / argmax of the regional mean anomaly 2011–2024", low=int(y_lo), high=int(y_hi),
                                    low_mean_cm=round(m[y_lo], 2), high_mean_cm=round(m[y_hi], 2))
    return [y_lo, y_hi], {y: preds[y] for y in (y_lo, y_hi)}, preds


def sym_range(preds_all, spec):
    """F7 편차 지도와 같은 대칭 범위: F7 캐시 래스터(모든 연도 편차, 원 격자)가 있으면 그것으로, 없으면 셀 값으로 MV.sym_max(99 백분위, 5 cm 단위)."""
    if F7_CACHE.exists():
        z = np.load(F7_CACHE)
        v = np.concatenate([z[k].astype(np.float32)[np.isfinite(z[k])] for k in z.files if k.startswith("img_")]); how = str(F7_CACHE.relative_to(ROOT))
    else:
        v = np.concatenate([d.values[np.isfinite(d.values)] for d in preds_all.values()]); how = "cell anomalies (all years)"
    vmax = MV.sym_max(v, step=5.0); ext = MV.extend_of(v, -vmax, vmax)
    spec["choices"]["colour_range"] = dict(rule="±MV.sym_max(|anomaly| 99th pct of all years, 5 cm step), same as F7", source=how, vmax_cm=vmax, extend=ext,
                                           p99_abs_cm=round(float(np.percentile(np.abs(v), 99)), 2))
    return vmax, ext


def transfer_deltas(spec):
    """지역 홀드아웃 ΔRMSE(방법 − 재보정 Stefan, 95 % 블록 구간). 지역 홀드아웃에서 P1 = P0 인지 summary_metrics 로 확인한다."""
    sm = pd.read_csv(RUN / "summary_metrics.csv")
    q = sm[(sm.scheme == "region") & (sm.forcing == "year_obs") & (sm.featset == "F8TS") & sm.method.isin(["P0", "P1"])]
    chk = {}
    for r in REGIONS:
        a = q[(q.region == r) & (q.method == "P0")].rmse.values; b = q[(q.region == r) & (q.method == "P1")].rmse.values
        assert len(a) == 1 and len(b) == 1, (r, a, b)
        chk[r] = dict(rmse_P0=round(float(a[0]), 4), rmse_P1=round(float(b[0]), 4), equal=bool(abs(a[0] - b[0]) < 1e-9))
    assert all(v["equal"] for v in chk.values()), f"지역 홀드아웃에서 P0 ≠ P1: {chk}"
    spec["checks"]["region_holdout_P1_equals_P0"] = chk
    d = pd.read_csv(RUN / "deltas_vs_ref.csv")
    d = d[(d.scheme == "region") & (d.forcing == "year_obs") & (d.featset == "F8TS") & (d.learner == "catboost") & (d.ref == "P1")]
    out = {}
    for r in REGIONS:
        for mth, lam in (("D0", -1.0), ("R1", 0.5)):
            row = d[(d.region == r) & (d.method == mth) & (np.isclose(d.lam, lam))]
            assert len(row) == 1, (r, mth, len(row))
            row = row.iloc[0]
            out[(r, mth)] = dict(delta=float(row.delta), lo=float(row.lo), hi=float(row.hi), n_blocks=int(row.n_blocks), n=int(row.n))
    spec["sources"]["deltas"] = str((RUN / "deltas_vs_ref.csv").relative_to(ROOT)); spec["sources"]["summary"] = str((RUN / "summary_metrics.csv").relative_to(ROOT))
    spec["numbers"]["region_holdout_delta_vs_stefan_cm"] = {f"{r}:{m}": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in vv.items()} for (r, m), vv in out.items()}
    return out


def ladder(spec):
    d = pd.read_csv(LADDER, low_memory=False)
    d = d[(d.exp == "TL") & d.method.isin(list(STYLE))].copy()
    d["n_label"] = pd.to_numeric(d.n_label, errors="coerce"); d = d[d.n_label.notna()]; d["n_label"] = d.n_label.astype(int)
    ns = [0, 25, 50, 100, 250, 500]
    out = {}
    for r in REGIONS:
        for m in STYLE:
            s = d[(d.region == r) & (d.method == m)].set_index("n_label").rmse
            assert all(n in s.index for n in ns), (r, m, sorted(s.index))
            out[(r, m)] = [float(s[n]) for n in ns]
    spec["sources"]["ladder"] = str(LADDER.relative_to(ROOT))
    spec["numbers"]["TL_rmse_cm"] = {f"{r}:{m}": [round(v, 2) for v in vs] for (r, m), vs in out.items()}
    spec["numbers"]["n_label"] = ns
    spec["numbers"]["n_test"] = {r: int(d[d.region == r].n_test.iloc[0]) for r in REGIONS}
    return ns, out


# ================================================================ 지도(편차 드레이프)
def scalebar_in(axm, lim, km, M6, fx=0.06, fy=0.07):
    """지도 틀 안 왼쪽 아래 축척 막대: 막대 + 양 끝 눈금, 글자는 막대 위 왼맞춤, 흰 테(틀 밖으로 나가지 않는다)."""
    x0, x1, y0, y1 = lim; L = km * 1000.0; tick = 0.014 * (y1 - y0)
    xs = x0 + fx * (x1 - x0); ys = y0 + fy * (y1 - y0)
    halo = [pe.withStroke(linewidth=M6.halo + 0.4, foreground="white")]
    axm.plot([xs, xs + L], [ys, ys], color="black", lw=M6.lw_scale, zorder=8, solid_capstyle="butt", path_effects=halo)
    for xx in (xs, xs + L):
        axm.plot([xx, xx], [ys, ys + tick], color="black", lw=0.8 * M6.lw_scale, zorder=8, solid_capstyle="butt", path_effects=halo)
    return axm.text(xs, ys + 1.7 * tick, f"{km} km", ha="left", va="bottom", fontsize=FS_K, zorder=9, path_effects=M6.halo_fx)


# ================================================================ G6 결론
def g6():
    t0 = time.time()
    stem = "G6_conclusion"
    spec: dict = {"stem": stem, "size_in": [W, H], "sources": {}, "numbers": {}, "choices": {}, "checks": {}}
    M.setup_fonts()
    fig, ax = M.canvas(W, H)

    # ---- a: 두 해의 편차 지도
    years, dsel, dall = anomaly_years(spec)
    vmax, ext = sym_range(dall, spec)
    key = "alaska"; cells, lut, ky0, kx0 = F6.cell_table(key)
    base = F6.Base(key, cells, lut, ky0, kx0, step=1, tag="full")                 # 250 m 음영 격자(표시용 보간 격자)
    lim = F6.region_limits(key, base.tf); asp = (lim[3] - lim[2]) / (lim[1] - lim[0])
    M6 = F6.Med("deck")
    land0 = F6.LAND.copy(); F6.LAND = np.array(to_rgb(NO_DATA_DIV), np.float32)      # 발산 색표 지도: '지도 없음' 육지 = 토큰 no_data_diverging
    norm = TwoSlopeNorm(0.0, -vmax, vmax)
    mh = 2.80; mw = mh / asp; gap = 0.10
    x_a = 0.40; y_a = 0.52                                                            # 지도 바닥 = b 축 바닥
    map_axes = []; yr_txt = []
    for k, yr in enumerate(years):
        val = F6.cell_values(key, cells, dsel[yr])
        rgb = F6.render(base, val, MV.CM_DIFF, norm, feather=2.0 * 2.0 / base.step, hatch=base.hatch_px(mw * 25.4, lim, M6.hatch_mm),
                        smooth=True, fill_max=F6.FILL_MAX)
        axm = M.sub_axes(fig, x_a + k * (mw + gap), y_a, mw, mh)
        F6.frame(axm, rgb, base.extent, lim, M6); F6.coast(axm, key, base.tf, M6); F6.graticule(axm, key, base.tf, lim, M6)
        yr_txt.append(axm.text(0.035, 0.965, str(yr), transform=axm.transAxes, ha="left", va="top", fontsize=FS_AX, color=M.INK, zorder=9,
                               path_effects=[pe.withStroke(linewidth=3.0, foreground="white")]))
        map_axes.append(axm)
    F6.LAND = land0
    sb_t = scalebar_in(map_axes[0], lim, 200, M6)
    x_cb = x_a + 2 * mw + gap + 0.12; cb_h = 0.78 * mh
    cax = M.sub_axes(fig, x_cb, y_a + (mh - cb_h) / 2, 0.11, cb_h)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=MV.CM_DIFF), cax=cax, orientation="vertical", extend=ext, extendfrac=0.05)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor(ZERO)
    tk = np.arange(-vmax, vmax + 1e-9, 5.0); cb.set_ticks(tk); cb.set_ticklabels([minus(f"{v:+.0f}") if v else "0" for v in tk])
    cax.tick_params(labelsize=FS_K, length=3, width=0.8, pad=2)
    cb.set_label("ALT anomaly (cm)", fontsize=FS_AX, labelpad=4)
    letter(ax, 0.10, y_a + mh + 0.02, "a")
    spec["choices"]["maps"] = dict(years=years, size_in=[round(mw, 3), mh], display="1 km cell centres → 250 m hillshade grid, bilinear (v7_f6_maps.render smooth, fill_max "
                                   f"{F6.FILL_MAX})", cmap="broc (MV.CM_DIFF)", no_data_land=NO_DATA_DIV, aoa="not marked (labels = year only)",
                                   scalebar="200 km inside map 1", labels="year only")

    # ---- b: 지역 홀드아웃 Stefan 대비 ΔRMSE(점 + 95 % 구간)
    dd = transfer_deltas(spec)
    xb0 = x_cb + 0.11 + 0.95                                                          # 색 막대 눈금·라벨 오른쪽에서 띄운 자리(y축 제목 두 줄 + 눈금)
    x_ax = xb0 + 0.86; ab = chart_axes(fig, x_ax, y_a, W - 0.08 - x_ax, H - 0.30 - y_a)
    off = {"D0": -0.13, "R1": 0.13}
    for i, r in enumerate(REGIONS):
        for mth in ("D0", "R1"):
            v = dd[(r, mth)]; x = i + off[mth]; c = MH[mth]
            ab.plot([x, x], [v["lo"], v["hi"]], color=c, lw=1.8, ls=(DASH if mth == "D0" else "-"), solid_capstyle="butt", dash_capstyle="butt", zorder=3)
            ab.plot([x], [v["delta"]], "o", ms=9.5, mfc=("white" if mth == "D0" else c), mec=c, mew=1.8, zorder=4)
    ab.axhline(0, color=ZERO, lw=0.9, zorder=1)
    lo = min(v["lo"] for v in dd.values()); hi = max(v["hi"] for v in dd.values())
    y0 = float(np.floor((lo - 1.0) / 5.0) * 5.0); y1 = float(np.ceil((hi + 1.0) / 5.0) * 5.0)
    ab.set_ylim(y0, y1); ab.set_yticks(np.arange(y0, y1 + 1e-9, 5.0)); ab.set_yticklabels([minus(f"{v:.0f}") for v in ab.get_yticks()])
    ab.set_xlim(-0.42, 2.42); ab.set_xticks(range(3)); ab.set_xticklabels([REG_EN[r] for r in REGIONS], fontsize=FS_T)
    for t_, r in zip(ab.get_xticklabels(), REGIONS):
        t_.set_color(REG_HEX[r])
    ab.tick_params(axis="x", length=0, pad=5)
    ab.set_ylabel("ΔRMSE vs Stefan,\nregion hold-out (cm)", fontsize=FS_AX, labelpad=4, linespacing=1.15)
    # 방법 이름은 범례 대신 레나 델타 표식 옆 직접 라벨(방법 색): 직접 ML = 표식 왼쪽, 물리 + 잔차 ML = 구간 위
    fig.canvas.draw()
    u = (y1 - y0) / (ab.get_position().height * H)                                   # cm / 인치
    xu = 2.84 / (ab.get_position().width * W)                                          # x 단위 / 인치
    vD = dd[("Lena", "D0")]; vR = dd[("Lena", "R1")]
    t_d0 = ab.text(1 + off["D0"] - 0.14 * xu, vD["delta"], NAME["D0"], ha="right", va="center", fontsize=FS_T, color=MH["D0"], zorder=6)
    t_r1 = ab.text(1 + off["R1"], vR["hi"] + 0.12 * u, NAME["R1"], ha="center", va="bottom", fontsize=FS_T, color=MH["R1"], zorder=6)
    lg = None
    fig.canvas.draw()
    yl = bbox_in(fig, ab.yaxis.get_label())
    letter(ax, min(yl[0], x_ax - 0.6) - 0.02, H - 0.08, "b")
    # ---- 점검: 직접 라벨 상자와 자료(점·구간 선)의 겹침, 색 막대 라벨 ↔ b 의 y축 제목 간격, 연도 글자·축척 글자가 지도 안, x 눈금 글자 사이
    inv = ab.transData.inverted(); inside = 0; ms_u = 9.5 / 72.0 / 2 * u               # 표식 반지름(cm 단위)
    for t in (t_d0, t_r1):
        b = bbox_in(fig, t)
        (lx0, ly0), (lx1, ly1) = inv.transform([(b[0] * fig.dpi, b[1] * fig.dpi), (b[2] * fig.dpi, b[3] * fig.dpi)])
        for (r, mth), v in dd.items():
            x = REGIONS.index(r) + off[mth]; mr_x = 9.5 / 72.0 / 2 * xu
            if lx0 - mr_x <= x <= lx1 + mr_x and not (v["hi"] + ms_u < ly0 or v["lo"] - ms_u > ly1):
                inside += 1
    xt = sorted(bbox_in(fig, t) for t in ab.get_xticklabels() if t.get_text())
    spec["qa_extra_xtick_gaps_in"] = [round(xt[i + 1][0] - xt[i][2], 3) for i in range(len(xt) - 1)]
    cbl = bbox_in(fig, cax.yaxis.get_label())
    gap_cb_b = round(yl[0] - cbl[2], 3)
    in_map = []
    for t, a_ in [(t, m) for t, m in zip(yr_txt, map_axes)] + [(sb_t, map_axes[0])]:
        tb = bbox_in(fig, t); mb = bbox_in(fig, a_); in_map.append(bool(tb[0] >= mb[0] and tb[2] <= mb[2] and tb[1] >= mb[1] and tb[3] <= mb[3]))
    spec["qa_extra"] = dict(legend_data_inside=int(inside), cbar_label_to_b_ylabel_gap_in=gap_cb_b, labels_inside_maps=in_map,
                            panel_gap_maps_to_b_in=round(yl[0] - (x_cb + 0.11), 3), elapsed_s=round(time.time() - t0, 1))
    extra_ok = inside == 0 and gap_cb_b >= 0.35 and all(in_map) and min(spec["qa_extra_xtick_gaps_in"]) >= 0.10
    print("G6 점검:", spec["qa_extra"], flush=True)
    spec["layout_in"] = dict(maps=[x_a, y_a, round(mw, 3), mh, gap], cbar=[round(x_cb, 3), 0.11], b_axes=[round(x_ax, 3), y_a])
    return finish(fig, stem, spec, extra_ok)


# ================================================================ G7 의의와 활용
def g7():
    stem = "G7_significance"
    spec: dict = {"stem": stem, "size_in": [W, H], "sources": {}, "numbers": {}, "choices": {}, "checks": {}}
    M.setup_fonts()
    fig, ax = M.canvas(W, H)
    ns, L = ladder(spec)
    xpos = [10.0 if n == 0 else float(n) for n in ns]                                  # n = 0 은 로그 축의 x = 10 자리(왼쪽)
    x0, gap, top, bot = 0.86, 0.80, H - 0.50, 0.66
    pw = (W - 0.08 - x0 - 2 * gap) / 3
    axes = []; notes = []; clipped = []
    for i, r in enumerate(REGIONS):
        a = chart_axes(fig, x0 + i * (pw + gap), bot, pw, top - bot)
        a.set_xscale("log"); a.set_xlim(8.2, 640.0)
        a.axvline(25.0, color=N25_LINE, lw=0.9, zorder=0.5)                           # n = 25 표시(옅은 가는 선)
        # y 범위: 자료(가장자리 값 하나가 나머지보다 멀리 떨어지면 자르고 표식·글자로 적는다)
        vals = np.array([v for m in STYLE for v in L[(r, m)]])
        core = np.sort(vals)
        cap = None
        if core[-1] - core[-2] > 8.0:                                                  # 고립된 큰 값(레나 n = 0 직접 ML)
            cap = float(np.ceil((core[-2] + 2.0) / 2.0) * 2.0)
        y_lo = float(np.floor((vals.min() - 0.6) / 1.0)); y_hi = cap if cap is not None else float(np.ceil((vals.max() + 0.6) / 1.0))
        for m, (c, ls) in STYLE.items():
            ys = np.array(L[(r, m)], float)
            yd = np.minimum(ys, y_hi) if cap is not None else ys
            a.plot(xpos, yd, color=c, ls=ls, lw=2.0, dash_capstyle="butt", zorder=3 if m != "R1S" else 4, clip_on=True)
            for xv, yv, yt in zip(xpos, yd, ys):
                if cap is not None and yt > y_hi:
                    a.plot([xv], [y_hi], "o", ms=7.5, mfc="white", mec=c, mew=1.6, zorder=6, clip_on=False)
                    notes.append(a.text(xv * 1.22, y_hi, f"{yt:.1f}", ha="left", va="center", fontsize=FS_T, color=c, zorder=6))
                    clipped.append(dict(region=r, method=m, n=int(ns[xpos.index(xv)]), value=round(float(yt), 2), drawn_at=y_hi))
                else:
                    a.plot([xv], [yv], "o", ms=5.5, mfc=c, mec="white", mew=0.7, zorder=5)
        step = 2.0 if (y_hi - y_lo) <= 12 else 5.0
        a.set_ylim(y_lo, y_hi); a.set_yticks(np.arange(np.ceil(y_lo / step) * step, y_hi + 1e-9, step))
        a.set_xticks(xpos); a.set_xticklabels([str(n) for n in ns]); a.minorticks_off()
        a.text(0.03, 0.04, REG_EN[r], transform=a.transAxes, ha="left", va="bottom", fontsize=FS_T, color=REG_HEX[r], zorder=7,
               path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
        axes.append(a)
    axes[0].set_ylabel("RMSE (cm)", fontsize=FS_AX, labelpad=4)
    axes[1].set_xlabel("Target-region labels, n", fontsize=FS_AX, labelpad=4)
    hs = [Line2D([], [], color=STYLE[m][0], ls=STYLE[m][1], lw=2.0, marker="o", ms=5.5, mfc=STYLE[m][0], mec="white", mew=0.7) for m in STYLE]
    lg = ax.legend(hs, [NAME[m] for m in STYLE], loc="upper center", bbox_to_anchor=(W / 2 + 0.25, H - 0.09), bbox_transform=ax.transData, ncol=4,
                   fontsize=FS_T, frameon=False, handlelength=2.2, handletextpad=0.5, columnspacing=1.5, borderaxespad=0.0, borderpad=0.0)
    spec["choices"]["layout"] = dict(panels=[[round(x0 + i * (pw + gap), 3), bot, round(pw, 3), round(top - bot, 3)] for i in range(3)],
                                     x="log, n = 0 drawn at x = 10", n25="thin grey vertical line", clipped=clipped,
                                     region_label="inside, bottom-left (region colour)", legend="one row above the panels")
    # ---- 점검: 지역 글자·값 글자와 선·점의 겹침(자료 좌표로), 범례가 캔버스 안
    fig.canvas.draw()
    hits = []
    for a, r in zip(axes, REGIONS):
        inv = a.transData.inverted()
        for t in [t for t in a.texts]:
            b = bbox_in(fig, t)
            (bx0, by0), (bx1, by1) = inv.transform([(b[0] * fig.dpi, b[1] * fig.dpi), (b[2] * fig.dpi, b[3] * fig.dpi)])
            for m in STYLE:
                ys = np.minimum(np.array(L[(r, m)]), a.get_ylim()[1])
                xs_ = np.exp(np.linspace(np.log(xpos[0]), np.log(xpos[-1]), 400)); yy = np.interp(np.log(xs_), np.log(xpos), ys)
                sel = (xs_ >= bx0) & (xs_ <= bx1)
                if np.any((yy[sel] >= by0) & (yy[sel] <= by1)):
                    hits.append([r, t.get_text(), m])
    lb = bbox_in(fig, lg)
    spec["qa_extra"] = dict(text_line_hits=hits, legend_in_canvas=bool(lb[0] >= 0.05 and lb[2] <= W - 0.05 and lb[3] <= H - 0.02),
                            legend_to_panels_gap_in=round(lb[1] - top, 3))
    print("G7 점검:", spec["qa_extra"], flush=True)
    return finish(fig, stem, spec, extra_ok=not hits and spec["qa_extra"]["legend_in_canvas"] and spec["qa_extra"]["legend_to_panels_gap_in"] >= 0.08)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["G6", "G7"], default=None)
    ap.add_argument("--slide", action="store_true", help="호환용")
    a = ap.parse_args()
    M.OUT.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "G7"):
        g7()
    if a.only in (None, "G6"):
        g6()
