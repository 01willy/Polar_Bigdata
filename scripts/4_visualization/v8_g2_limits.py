"""G2 기존 연구의 한계 (덱 전용 쪽, 12 × 4.65 in = 3600 × 1395 px, 300 dpi, 영문 FreeSans).

v10(2026-10-16, 사용자 지시: 그림 안 글자는 모두 영어, 논문 어휘): 배치는 그대로, 글자만 영어로 바꿨다(열 머리 잉크색 M.HEAD).
저장 전 모든 글자 객체에서 한글을 찾으면 멈춘다(no_hangul).

v9(2026-10-15, 설계 지침 v9 항 4): 그림 안 각주('바이올린 몸통…', '5분위 평균…', '선 끝 = 제품 마지막 해…')·통계 글줄·'자료 상한' 직접 라벨·
'F2 b' 표기·선 끝 표식을 지우고, RMSE 는 열 1 오른쪽의 값 열(머리 'RMSE', 제품 색)로 옮겼다. 열 사이 0.60 in, 행 사이는 x축 제목과 설명 글줄 사이 0.14 in.
x축 제목은 '토양 유기탄소 구간 (g/kg)', '경사 구간 (°)'(등간격 5분위, 눈금 = 구간 중앙값; 이 설명은 덱 설명 줄 몫). 아래 v8 명세의 각주 항목은 폐기.

명세(docs/DESIGN_BRIEF_v8_2026-10-15.md 2절 3쪽, 항 5)
- 세 열, 주황 열 머리: 기존 제품의 불일치 | Stefan 물리식의 한계 | 관측·제품의 시간 불균형. 그림 안 문장 없음, 라벨은 1–3 단어.
- 열 1: 알래스카 공통 셀-연도(Wei 2026·Liu 2024·ESA CCI v4·관측이 모두 있는 행)의 (제품 − 관측) 분포, 수평 바이올린 + 사분위 선 + 중앙값.
        0 선, 축 라벨 cm. 아래 두 줄: 같은 행에서 계산한 제품별 RMSE, 공통 셀-연도 수와 기간.
        바이올린 몸통은 중심 95 %(2.5–97.5 백분위) 행, RMSE 는 전체 행 → 열 각주에 적는다(10-15 검토 1).
- 열 2: 관측 − 재보정 Stefan(블록 교차검증 OOF)을 토양 유기탄소(sg_soc_0_5, g/kg)·경사(dem_slope, °) 5분위(등간격 배치, 눈금 = 구간 중앙값)로
        묶은 평균 ± 95 % 신뢰구간(수염, 표식 없음 → 작은 CI 도 보인다), 0 선. 계수 하나로 담기지 않는 체계적 잔차.
        P1 재보정은 최소제곱 배율 하나(src/polar/testbed_v5.py fit_phys → ls_scale)라 씨앗과 무관하고, OOF 파일은 seed 0 하나뿐이다.
        토양 유기탄소 5번째 구간(405–419 g/kg)은 자료 상한의 이산 값 구간이라 직접 라벨로 표시한다(10-15 검토 2).
- 열 3: 위 알래스카 연도별 셀-연도 수 막대, 아래 레나 최장 관측 셀(F2 b 와 같은 셀)의 연도별 관측 ALT(원) 대 제품 연도 값
        (Wei 선·Liu 파선·CCI v5 선). 선 끝 표식 = 제품의 마지막 연도. 각주: 선 끝 = 제품 마지막 해(CCI v4 2021 · v5 2023).
- 자료: labels_cell_year_v5.csv, products_cell_year_v5.csv, static_v5.csv, site_series_year_obs.csv.gz,
        runs/core_v6/preds/Alaska__block__year_obs__F8TS__P1__-__lam-1.csv.gz. 수치는 모두 파일에서 계산한다.
- 글자(design/style_tokens_v4.json font.slide): 축 라벨 14 pt, 눈금 12 pt, 설명·범례·각주·직접 라벨 13 pt, 열 머리 15 pt 굵게(주황).
        색: 제품 S.PRODUCT_HEX, 지역 S.REGION_HEX, 재보정 Stefan S.METHOD_HEX['P1'](토큰 하나로 전 그림 통일).
- 산출: deck/assets/v8/G2_limits.png(RGB, 흰 바탕) + G2_limits.pdf + G2_limits_spec.json(배치·선택·수치 출처·QA).
- 실행(CPU 만): OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v8_g2_limits.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization"))
sys.path.insert(0, str(ROOT / "src"))
import method_figs_v4 as M  # noqa: E402  (M.canvas, M.T, M.RT, M.head, M.hline, M.vline, M.sub_axes, M.qa_and_save, M.setup_fonts)
import polar.style_v5 as S  # noqa: E402  (S.PRODUCT_HEX, S.REGION_HEX, S.METHOD_HEX: design/style_tokens_v4.json)

M.OUT = ROOT / "deck" / "assets" / "v8"
PROC = ROOT / "data" / "processed" / "v5"
RUN = PROC / "runs" / "core_v6"
PRED = RUN / "preds" / "Alaska__block__year_obs__F8TS__P1__-__lam-1.csv.gz"
STEM = "G2_limits"

# ================================================================ 배치(인치)
W, H = 12.0, 4.65
COLS = [(0.10, 3.60), (4.20, 7.70), (8.30, 11.90)]      # 열 안쪽 x 범위(열 사이 0.60 in)
SEPX = [3.90, 8.00]                                     # 열 구분 파선(열 사이 가운데)
Y_HEAD, Y_HEADRULE = H - 0.17, H - 0.31                 # 열 머리 · 머리 밑줄
Y_CAP1 = H - 0.51                                       # 1행 설명 글줄(세 열 같은 높이)
AX1_T = H - 0.77                                        # 1행 축 위(열 1 의 한 장짜리 차트도 같다)
AX2_B = 0.55                                            # 2행 축 아래(열 1 차트 아래도 같다): 아래로 눈금 글자·x축 제목 0.47 in
XLAB_ZONE = 0.47                                        # 축 아래 ~ x축 제목 바닥(눈금 3 pt + 간격 2 pt + 눈금 글자 + 간격 + 14 pt 제목)
G_XLAB_CAP, H_CAP, G_CAP_AX = 0.14, 0.18, 0.13          # x축 제목 바닥 ~ 2행 설명 글줄 위 · 설명 글줄 높이 · 설명 글줄 바닥 ~ 축 위(눈금 글자 반 높이 포함)
_ROW_GAP = XLAB_ZONE + G_XLAB_CAP + H_CAP + G_CAP_AX    # 1행 축 아래 ~ 2행 축 위
_ROW_H = (AX1_T - AX2_B - _ROW_GAP) / 2                 # 두 행 같은 높이
AX1_B = AX1_T - _ROW_H
AX2_T = AX2_B + _ROW_H
Y_CAP2 = AX1_B - XLAB_ZONE - G_XLAB_CAP - H_CAP / 2     # 2행 설명 글줄 가운데
X_OFF = 0.62                                            # 축 라벨·눈금 글자 폭
FS_T = 13.0                                             # 설명·범례·각주·직접 라벨 13 pt(토큰 font.slide.direct_label_pt)
FS_AX = 14.0                                            # 축 라벨 14 pt(토큰 font.slide.axis_label_pt)
FS_K = 12.0                                             # 눈금 12 pt(토큰 font.slide.tick_pt)
FS_HEAD = M.FS["head"]                                  # 열 머리 15 pt
DASH = (0, (3.5, 1.8))                                  # style_v5 METHOD_LS 'dashed'
P1 = S.METHOD_HEX["P1"]                                 # 재보정 Stefan #1F5A99
ALASKA = S.REGION_HEX["Alaska"]                         # #332288
PROD = {"wei": ("Wei 2026", S.PRODUCT_HEX["wei"], "-"), "liu": ("Liu 2024", S.PRODUCT_HEX["liu"], DASH),
        "cci4y": ("ESA CCI v4", S.PRODUCT_HEX["cci4y"], "-"), "cci5y": ("ESA CCI v5", S.PRODUCT_HEX["cci5y"], "-")}
SPEC: dict = {"stem": STEM, "size_in": [W, H], "sources": {}, "choices": {}}


def minus(s: str) -> str:
    return s.replace("-", "−")


def src(key, p: Path):
    SPEC["sources"][key] = str(p.relative_to(ROOT))


# ================================================================ 자료
def labels():
    p = PROC / "labels_cell_year_v5.csv"; src("labels", p)
    L = pd.read_csv(p)
    return L[L.in_forcing == 1].copy()


def common_products(L):
    """알래스카에서 Wei·Liu·CCI v4·관측이 모두 있는 셀-연도: 제품 − 관측(cm) 과 같은 행의 RMSE."""
    p = PROC / "products_cell_year_v5.csv"; src("products", p)
    P = pd.read_csv(p)
    ak = L[L.region == "Alaska"][["loc_id", "year", "alt_cm"]]
    keys = ["wei", "liu", "cci4y"]
    for k in keys:
        s = P[(P["product"] == k) & P.value_cm.notna()][["loc_id", "year", "value_cm"]].rename(columns={"value_cm": k})
        ak = ak.merge(s, on=["loc_id", "year"], how="left")
    c = ak.dropna(subset=keys)
    out = {"n": int(len(c)), "n_cells": int(c.loc_id.nunique()), "years": [int(c.year.min()), int(c.year.max())], "diff": {}, "stats": {}}
    for k in keys:
        d = (c[k] - c.alt_cm).values.astype(float)
        out["diff"][k] = d
        q = np.percentile(d, [1, 2.5, 5, 25, 50, 75, 95, 97.5, 99])
        out["stats"][k] = dict(rmse=round(float(np.sqrt(np.mean(d ** 2))), 2), bias=round(float(d.mean()), 2),
                               p1=round(float(q[0]), 1), p2_5=round(float(q[1]), 1), p5=round(float(q[2]), 1), q25=round(float(q[3]), 1),
                               median=round(float(q[4]), 1), q75=round(float(q[5]), 1), p95=round(float(q[6]), 1), p97_5=round(float(q[7]), 1),
                               p99=round(float(q[8]), 1), min=round(float(d.min()), 1), max=round(float(d.max()), 1))
    # 제품별 자료 기간(값이 있는 연도): 각주 'CCI v4 –2021 · v5 –2023' 의 출처
    yr = P[P.value_cm.notna()].groupby("product").year.agg(["min", "max"])
    out["product_years"] = {k: [int(yr.loc[k, "min"]), int(yr.loc[k, "max"])] for k in ("wei", "liu", "cci4y", "cci5y")}
    SPEC["choices"]["common_cells"] = {k: v for k, v in out.items() if k != "diff"}
    return out


def stefan_residual_bins(nbin=5):
    """관측 − 재보정 Stefan(블록 교차검증 OOF)을 정적 공변량 5분위로 묶는다. 반환 {cov: DataFrame(mid, lo, hi, mean, ci, n)}.
    OOF 파일의 씨앗이 여러 개면 평균하지만, P1(최소제곱 배율 하나)은 결정적이라 파일은 seed 0 하나뿐이다."""
    src("stefan_oof", PRED)
    p = pd.read_csv(PRED)
    seeds = sorted(int(s) for s in p.seed.unique())
    m = p.groupby(["cell_year_id", "loc_id", "year"]).agg(y=("y", "first"), pred=("pred", "mean"), n_seed=("seed", "nunique")).reset_index()
    st_p = PROC / "static_v5.csv"; src("static", st_p)
    st = pd.read_csv(st_p, usecols=["loc_id", "sg_soc_0_5", "dem_slope"])
    m = m.merge(st, on="loc_id", how="left")
    m["res"] = m.y - m.pred
    out = {}
    for cov in ("sg_soc_0_5", "dem_slope"):
        q = pd.qcut(m[cov], nbin, labels=False, duplicates="drop")
        g = m.groupby(q, observed=True).agg(mid=(cov, "median"), lo=(cov, "min"), hi=(cov, "max"), mean=("res", "mean"),
                                           sd=("res", "std"), n=("res", "size"))
        g["ci"] = 1.96 * g.sd / np.sqrt(g.n)
        out[cov] = g.reset_index(drop=True)
    soc = m.sg_soc_0_5
    SPEC["choices"]["stefan_residual"] = dict(
        file=str(PRED.relative_to(ROOT)), n_cell_years=int(len(m)), seeds=seeds, n_seeds=int(m.n_seed.max()),
        seed_note="P1 = E·√TDD, E 는 최소제곱 배율(testbed_v5.fit_phys/ls_scale) 하나라 씨앗과 무관(결정적). OOF 파일은 seed 0 하나. "
                  "대본·캡션에는 '블록 교차검증 OOF 잔차(seed 0, 결정적 재보정)' 로 적는다. '복수 씨앗 평균' 이라고 쓰지 않는다.",
        scheme="block (5-fold, OOF)", method="P1 recalibrated Stefan",
        rmse_cm=round(float(np.sqrt(np.mean(m.res ** 2))), 2), units=dict(sg_soc_0_5="g/kg (SoilGrids column already physical, src/polar/physics.py)", dem_slope="degrees"),
        x_axis="등간격 5분위 배치(값 비례 아님) · 눈금 = 구간 중앙값 · 각주 '눈금 = 중앙값'",
        soc_discrete=dict(n_unique=int(soc.nunique()), max=round(float(soc.max()), 1), n_rows_ge_405=int((soc >= 405).sum()),
                          unique_ge_405=[round(float(v), 1) for v in np.sort(soc[soc >= 405].unique())],
                          note="토양 유기탄소 열은 이산 값(소수의 고유값). 5번째 구간 405.0–419.2 g/kg 은 자료 상한의 값 묶음이라 그림에 '상한 405–419' 직접 라벨"),
        bins={cov: dict(mid=[round(float(v), 2) for v in g.mid], lo=[round(float(v), 2) for v in g.lo], hi=[round(float(v), 2) for v in g.hi],
                        mean=[round(float(v), 2) for v in g["mean"]], ci95=[round(float(v), 2) for v in g.ci], n=[int(v) for v in g.n])
              for cov, g in out.items()})
    return out


def alaska_counts(L):
    s = L[L.region == "Alaska"].groupby("year").size().reindex(range(2010, 2025), fill_value=0)
    SPEC["choices"]["alaska_cell_years_by_year"] = {int(k): int(v) for k, v in s.items()}
    return s


def lena_long_cell():
    """레나 최장 관측 셀(v7_f2.pick_lena_cell 과 같은 규칙 → F2 b 와 같은 셀): 관측 연도 수 최다, 연도별 제품 3종(wei, liu, cci5y)이 모두 값을 가진 셀."""
    sp = PROC / "site_series" / "site_series_year_obs.csv.gz"; src("site_series", sp)
    ss = pd.read_csv(sp); pr = pd.read_csv(PROC / "products_cell_year_v5.csv")
    obs = ss[(ss.region == "Lena") & (ss.observed == 1)]; n = obs.groupby("loc_id").year.nunique()
    p = pr[(pr.year >= 2010) & (pr.year <= 2024) & pr["product"].isin(["wei", "liu", "cci5y"])]
    ok = p.groupby(["loc_id", "product"]).value_cm.apply(lambda s: s.notna().any()).unstack().fillna(False)
    ok = ok.index[ok.all(axis=1)]
    n = n[n.index.isin(ok)].sort_index()
    lid = int(n.sort_values(ascending=False, kind="stable").index[0])
    o = obs[obs.loc_id == lid].sort_values("year")[["year", "y"]]
    series = {}
    for k in ("wei", "liu", "cci5y"):
        s = pr[(pr.loc_id == lid) & (pr["product"] == k) & pr.year.between(2010, 2024)].dropna(subset=["value_cm"]).sort_values("year")
        series[k] = s[["year", "value_cm"]].reset_index(drop=True)
    lat, lon = float(obs[obs.loc_id == lid].lat.iloc[0]) if "lat" in obs else None, float(obs[obs.loc_id == lid].lon.iloc[0]) if "lon" in obs else None
    SPEC["choices"]["lena_long_cell"] = dict(loc_id=lid, n_years=int(n.max()), lat=lat, lon=lon, same_cell_as="v7 F2 panel b (pick_lena_cell)",
                                            observed={int(y): round(float(v), 1) for y, v in zip(o.year, o.y)},
                                            products={k: {int(y): round(float(v), 1) for y, v in zip(s.year, s.value_cm)} for k, s in series.items()},
                                            series_last_year={k: int(s.year.iloc[-1]) for k, s in series.items()})
    return lid, o, series


# ================================================================ 축 양식
def chart_axes(fig, col, b, t, x_off=X_OFF, right_pad=0.05):
    x0, x1 = COLS[col]
    ax = M.sub_axes(fig, x0 + x_off, b, x1 - right_pad - (x0 + x_off), t - b)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_linewidth(0.8); ax.spines[k].set_edgecolor(M.ZERO)
    ax.tick_params(labelsize=FS_K, length=3, width=0.8, pad=2, colors=M.INK)
    return ax


def caption(ax, col, y, s):
    return M.T(ax, COLS[col][0], y, s, size=FS_T, color=M.AUX)


def tick_label_artists(fig):
    ids = set()
    for a in fig.axes:
        if not a.axison:
            continue
        for t in a.get_xticklabels(minor=False) + a.get_yticklabels(minor=False):
            ids.add(id(t))
    return ids


def no_hangul(fig):
    """그림 안 모든 글자 객체(눈금 글자 포함)에 한글이 있으면 멈춘다(v10: 그림 글자는 영어만)."""
    import re
    bad = [t.get_text() for t in fig.findobj(Text) if re.search("[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", t.get_text())]
    if bad:
        raise RuntimeError(f"한글 글자 남음: {bad[:8]}")


def width_in(fig, t: Text) -> float:
    return float(t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi)


# ================================================================ 그림
def g2():
    M.setup_fonts()
    fig, ax = M.canvas(W, H)
    L = labels()
    SPEC["choices"]["colors"] = dict(products={k: v[1] for k, v in PROD.items()}, stefan_P1=P1, alaska=ALASKA)
    SPEC["choices"]["fonts_pt"] = dict(head=FS_HEAD, axis_label=FS_AX, tick=FS_K, caption_legend=FS_T)
    SPEC["layout_in"] = dict(cols=COLS, sep_x=SEPX, y_head=Y_HEAD, y_headrule=Y_HEADRULE, y_cap1=Y_CAP1, y_cap2=round(Y_CAP2, 3),
                             ax1=[round(AX1_B, 3), AX1_T], ax2=[AX2_B, round(AX2_T, 3)], row_h=round(_ROW_H, 3),
                             note="v9: 각주·통계 글줄·직접 주석 삭제, 열 사이 0.60 in, 높이 4.65 in")
    col_texts: list[tuple[int, Text]] = []                                   # 열 폭 점검용(설명 글줄)

    # ---- 열 머리(주황, 굵게) · 헤어라인 · 열 구분선
    for (x0, x1), s in zip(COLS, ("Product disagreement", "Limits of the Stefan model", "Temporal imbalance")):
        M.head(ax, x0, Y_HEAD, s, size=FS_HEAD)
        M.hline(ax, x0, x1, Y_HEADRULE, color=M.HAIR, lw=0.8)
    for x in SEPX:
        M.vline(ax, x, 0.12, Y_HEADRULE - 0.06, color=M.SEP, lw=0.9, ls=(0, (3, 3)))

    # ================================ 열 1: 공통 셀-연도의 제품 − 관측 분포(수평 바이올린 + 사분위 선 + 중앙값), 오른쪽에 RMSE 열
    C = common_products(L)
    keys = ["wei", "liu", "cci4y"]; ypos = {"wei": 3, "liu": 2, "cci4y": 1}
    RMSE_COL = 0.62                                                                                # 오른쪽 RMSE 열 폭(인치)
    a1 = chart_axes(fig, 0, AX2_B, AX1_T, x_off=1.02, right_pad=RMSE_COL)
    lo_s = [C["stats"][k]["p2_5"] for k in keys]; hi_s = [C["stats"][k]["p97_5"] for k in keys]
    xlo = float(np.floor(min(lo_s) / 10) * 10); xhi = float(np.ceil(max(hi_s) / 10) * 10)
    trimmed = []                                                                                   # 바이올린 몸통 = 가운데 95 %(2.5–97.5 백분위 안의 행)
    for k, lo_, hi_ in zip(keys, lo_s, hi_s):
        d = C["diff"][k]; trimmed.append(d[(d >= lo_) & (d <= hi_)])
    SPEC["choices"]["violin"] = dict(kde="matplotlib GaussianKDE, bw 0.15, body = rows inside the 2.5–97.5 percentile (central 95 %)", xlim=[xlo, xhi],
                                    body_range_cm={k: [lo_, hi_] for k, lo_, hi_ in zip(keys, lo_s, hi_s)},
                                    body_rows={k: int(len(t)) for k, t in zip(keys, trimmed)},
                                    stats_rows="RMSE·사분위·중앙값은 절단 없는 전체 행(common_cells.n). 그림 안 각주는 v9 에서 뺐다(덱 설명 줄 몫)")
    vp = a1.violinplot(trimmed, positions=[ypos[k] for k in keys], vert=False, widths=0.82, showextrema=False, showmedians=False,
                       points=400, bw_method=0.15)
    for body, k in zip(vp["bodies"], keys):
        body.set_facecolor(PROD[k][1]); body.set_edgecolor("none"); body.set_alpha(0.45); body.set_zorder(2)
    a1.axvline(0, color=M.ZERO, lw=0.9, zorder=3)
    for k in keys:
        st = C["stats"][k]; y = ypos[k]; c = PROD[k][1]
        a1.plot([st["q25"], st["q75"]], [y, y], color=c, lw=3.2, solid_capstyle="butt", zorder=4)
        a1.plot([st["median"]], [y], "o", ms=7.0, mfc="white", mec=c, mew=1.4, zorder=5)
    a1.set_xlim(xlo, xhi); a1.set_ylim(0.45, 3.55)
    step = 25.0 if (xhi - xlo) <= 150 else 50.0
    a1.set_xticks(np.arange(np.ceil(xlo / step) * step, xhi + 1e-9, step))
    a1.set_xticklabels([minus(f"{v:+.0f}" if v != 0 else "0") for v in a1.get_xticks()])
    a1.set_yticks([ypos[k] for k in keys]); a1.set_yticklabels([PROD[k][0] for k in keys], fontsize=FS_T)
    for t, k in zip(a1.get_yticklabels(), keys):
        t.set_color(PROD[k][1])
    a1.tick_params(axis="y", length=0)
    a1.spines["left"].set_visible(False)
    a1.set_xlabel("Product − observed (cm)", fontsize=FS_AX, labelpad=2)
    col_texts.append((0, caption(ax, 0, Y_CAP1, "Same cell, same year (Alaska)")))
    # RMSE 열: 축 오른쪽 정돈된 칸(머리 'RMSE' + 행마다 값, 제품 색). 같은 행(전체 행)에서 계산한 값
    p1_ = a1.get_position(); x_rm = COLS[0][1]                                                     # 오른쪽 끝 맞춤
    y_in = lambda yy: p1_.y0 * H + (yy - 0.45) / (3.55 - 0.45) * p1_.height * H                     # noqa: E731  자료 y → 캔버스 인치
    rm_head = M.T(ax, x_rm, y_in(3.55) - 0.02, "RMSE\n(cm)", size=FS_T, color=M.AUX, ha="right", va="top", linespacing=1.1)
    rm_vals = []
    for k in keys:
        rm_vals.append(M.T(ax, x_rm, y_in(ypos[k]), f"{C['stats'][k]['rmse']:.1f}", size=FS_T, color=PROD[k][1], ha="right"))
    SPEC["choices"]["rmse_column"] = dict(width_in=RMSE_COL, header="RMSE (cm), 두 줄", unit="cm", values={k: C["stats"][k]["rmse"] for k in keys},
                                         n_common_cell_years=C["n"], years=C["years"])

    # ================================ 열 2: 관측 − 재보정 Stefan 의 공변량 5분위 평균 ± 95 % CI(수염, 표식 없음)
    R = stefan_residual_bins()
    allv = np.concatenate([np.r_[g["mean"] - g.ci, g["mean"] + g.ci] for g in R.values()])
    ylo = float(np.floor(allv.min() / 5) * 5 - 1.5); yhi = float(np.ceil(allv.max() / 5) * 5 + 1.5)      # 눈금 5 cm 간격, 여유 1.5 cm
    rows = [("sg_soc_0_5", AX1_B, AX1_T, Y_CAP1, "Soil organic carbon bin (g/kg)", "{:.0f}", "Stefan residual by soil carbon (Alaska)"),
            ("dem_slope", AX2_B, AX2_T, Y_CAP2, "Slope bin (°)", "{:.1f}", "Stefan residual by slope (Alaska)")]
    axes2 = []
    for cov, b, t, ycap, xl, fmt, cap in rows:
        g = R[cov]; x = np.arange(1, len(g) + 1)
        a = chart_axes(fig, 1, b, t)
        a.axhline(0, color=M.ZERO, lw=0.9, zorder=1)
        a.plot(x, g["mean"], color=P1, lw=1.3, zorder=3)
        a.errorbar(x, g["mean"], yerr=g.ci, fmt="none", ecolor=P1, elinewidth=1.0, capsize=3.2, capthick=1.0, zorder=4)   # 수염 + 캡, 표식 없음
        a.set_xlim(0.5, len(g) + 0.5); a.set_ylim(ylo, yhi)
        a.set_xticks(x); a.set_xticklabels([fmt.format(v) for v in g.mid])
        yt = [v for v in np.arange(np.ceil(ylo / 5) * 5, yhi + 1e-9, 5.0)]
        if len(yt) > 4:                                                                          # 낮은 축(≈ 1.1 in)에서는 눈금 10 cm 간격
            yt = [v for v in yt if v % 10 == 0]
        a.set_yticks(yt); a.set_yticklabels([minus(f"{v:.0f}") for v in yt])
        a.set_xlabel(xl, fontsize=FS_AX, labelpad=2); a.set_ylabel("Residual (cm)", fontsize=FS_AX, labelpad=3)   # 잔차 = 관측 − Stefan(설명 글줄 'Stefan residual'). 축 높이 1.2 in 에 맞는 길이
        col_texts.append((1, caption(ax, 1, ycap, cap)))
        axes2.append(a)
    ci_max = float(max(g_.ci.max() for g_ in R.values()))
    SPEC["choices"]["stefan_residual"]["ci95_max_cm"] = round(ci_max, 2)
    SPEC["choices"]["stefan_residual"]["ylim"] = [ylo, yhi]
    SPEC["choices"]["stefan_residual"]["x_axis"] = "등간격 5분위 배치(값 비례 아님) · 눈금 = 구간 중앙값 · x축 제목 '구간'(v9: 각주 삭제, 덱 설명 줄 몫)"

    # ================================ 열 3 · 위: 알래스카 연도별 셀-연도 수
    cnt = alaska_counts(L)
    a3 = chart_axes(fig, 2, AX1_B, AX1_T)
    a3.bar(cnt.index, cnt.values / 1000.0, width=0.75, color=ALASKA, lw=0)
    a3.set_xlim(2009.3, 2024.7); a3.set_xticks([2010, 2015, 2020]); a3.set_ylim(0, 8.5); a3.set_yticks([0, 4, 8])
    a3.set_xlabel("Year", fontsize=FS_AX, labelpad=2); a3.set_ylabel("Cell-years (10³)", fontsize=FS_AX, labelpad=3)
    col_texts.append((2, caption(ax, 2, Y_CAP1, "Observations per year (Alaska)")))

    # ================================ 열 3 · 아래: 레나 최장 관측 셀, 관측(원) 대 제품 연도 값(Wei 선 · Liu 파선 · CCI v5 선)
    lid, obs, series = lena_long_cell()
    a4 = chart_axes(fig, 2, AX2_B, AX2_T)
    hs, ls_ = [], []
    for k in ("wei", "liu", "cci5y"):
        s = series[k]; name, c, ls = PROD[k]
        a4.plot(s.year, s.value_cm, color=c, lw=1.8, ls=ls, zorder=2)
        hs.append(Line2D([], [], color=c, lw=1.8, ls=ls)); ls_.append(name)
    a4.plot(obs.year, obs.y, "o", ms=6.8, mfc="white", mec=M.INK, mew=1.2, zorder=5)
    hs.append(Line2D([], [], marker="o", ms=6.8, mfc="white", mec=M.INK, mew=1.2, ls="none")); ls_.append("Observed")
    vmax = max(float(obs.y.max()), max(float(s.value_cm.max()) for s in series.values()))
    a4.set_xlim(2009.3, 2024.7); a4.set_xticks([2010, 2015, 2020])
    a4.set_xlabel("Year", fontsize=FS_AX, labelpad=2); a4.set_ylabel("ALT (cm)", fontsize=FS_AX, labelpad=3)
    lg4 = a4.legend(hs, ls_, loc="upper left", bbox_to_anchor=(0.04, 1.02), fontsize=FS_T, frameon=False, ncol=2, handlelength=1.2, columnspacing=0.7,
                    handletextpad=0.4, borderaxespad=0.0, borderpad=0.0, labelspacing=0.25)
    # 범례 아래 끝이 범례 x 범위 안의 모든 자료(선·관측)보다 0.05 in 위에 오도록 y 상한을 25 cm 씩 올린다
    pts4 = [(float(x_), float(v_)) for s in series.values() for x_, v_ in zip(s.year, s.value_cm)] + [(float(x_), float(v_)) for x_, v_ in zip(obs.year, obs.y)]
    ytop = float(np.ceil(vmax * 1.1 / 25) * 25)
    for _ in range(12):
        a4.set_ylim(0, ytop); fig.canvas.draw()
        lb_ = lg4.get_window_extent(fig.canvas.get_renderer()); inv_ = a4.transData.inverted()
        (lx0_, ly0_), (lx1_, _) = inv_.transform([(lb_.x0, lb_.y0 - 0.05 * fig.dpi), (lb_.x1, lb_.y1)])
        if all(v_ < ly0_ for x_, v_ in pts4 if lx0_ - 0.4 <= x_ <= lx1_ + 0.4):
            break
        ytop += 25.0
    a4.set_yticks(np.arange(0, ytop + 1e-9, 50))
    SPEC["choices"]["lena_panel_ytop_cm"] = ytop
    col_texts.append((2, caption(ax, 2, Y_CAP2, "Long-term cell (Lena Delta)")))

    # ---- 점검: 글자 크기(눈금 12 pt · 그 밖 13 pt), 범례와 선의 겹침, 글줄 폭이 열 폭 안인지, RMSE 열이 축·바이올린과 떨어졌는지
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    ticks = tick_label_artists(fig)
    below13 = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < FS_T - 1e-6:
            if id(t) in ticks and t.get_fontsize() >= FS_K - 1e-6:
                continue
            below13.append([t.get_text()[:30], float(t.get_fontsize())])
    leg = a4.get_legend(); lb = leg.get_window_extent(r); inv = a4.transData.inverted()
    (lx0, ly0), (lx1, ly1) = inv.transform([(lb.x0, lb.y0), (lb.x1, lb.y1)])
    under = []
    for k, s in series.items():
        sel = s[(s.year >= lx0 - 0.3) & (s.year <= lx1 + 0.3)]
        if len(sel):
            under.append((k, float(sel.value_cm.max())))
    sel_o = obs[(obs.year >= lx0 - 0.3) & (obs.year <= lx1 + 0.3)]
    if len(sel_o):
        under.append(("observed", float(sel_o.y.max())))
    legend_clear = all(v < ly0 for _, v in under)
    leg_left_gap_in = round(float((lb.x0 - a4.get_window_extent(r).x0) / fig.dpi), 3)                 # 범례 왼쪽 ~ y축 선
    widths = [[t.get_text()[:34], round(width_in(fig, t), 2), round(COLS[c][1] - COLS[c][0], 2)] for c, t in col_texts]
    fits = all(w <= cw for _, w, cw in widths)
    ab1 = a1.get_window_extent(r)
    rm_gap = round(float((min(t.get_window_extent(r).x0 for t in rm_vals + [rm_head]) - ab1.x1) / fig.dpi), 3)
    SPEC["qa_extra"] = dict(non_tick_text_below_13pt=below13, legend_bottom_cm=round(float(ly0), 1), legend_x_years=[round(float(lx0), 1), round(float(lx1), 1)],
                            series_max_under_legend=[[k, round(v, 1)] for k, v in under], legend_clear_of_lines=bool(legend_clear),
                            legend_left_gap_to_yaxis_in=leg_left_gap_in, text_widths_in=widths, texts_fit_columns=bool(fits),
                            rmse_column_gap_to_axes_in=rm_gap,
                            ci_px_at_300dpi={cov: [round(float(2 * c * (t - b) / (yhi - ylo) * M.DPI), 1) for c in R[cov].ci]
                                             for cov, b, t in (("sg_soc_0_5", AX1_B, AX1_T), ("dem_slope", AX2_B, AX2_T))})
    no_hangul(fig)
    rec = M.qa_and_save(fig, STEM, allow_overlap=())
    # 저장 이름: G2_limits.png(RGB 흰 바탕) · G2_limits.pdf
    from PIL import Image
    png_s, pdf_s = M.OUT / f"{STEM}_slide.png", M.OUT / f"{STEM}_slide.pdf"
    png, pdf = M.OUT / f"{STEM}.png", M.OUT / f"{STEM}.pdf"
    with Image.open(png_s) as im:
        bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        bg.save(png, dpi=(M.DPI, M.DPI)); px = list(im.size)
    png_s.unlink(); pdf_s.replace(pdf)
    SPEC["qa"] = {k: rec[k] for k in ("min_font_pt", "texts", "outside", "overlaps", "small", "text_line", "text_axes", "text_patch")}
    SPEC["qa"]["png_px"] = px; SPEC["qa"]["png_px_ok"] = bool(px == [int(W * M.DPI), int(H * M.DPI)])
    SPEC["caption_note"] = ("열 2 잔차는 블록 교차검증 OOF(seed 0; P1 재보정은 최소제곱 배율 하나라 결정적) 기준, 구간 = 등간격 배치 5분위(눈금 = 구간 중앙값), 수염 = 95 % CI. "
                            f"열 1 바이올린 몸통은 중심 95 %, RMSE 는 전체 행(공통 셀-연도 {C['n']:,}개, {C['years'][0]}–{C['years'][1]}). "
                            "열 3 아래는 v7 F2 b 와 같은 레나 셀. 이 설명은 그림 안에 두지 않는다(덱 설명 줄 몫).")
    print("글자 크기 점검(눈금 외 13 pt 미만):", "OK" if not below13 else f"CHECK {below13}", "· 화소", px, flush=True)
    print("범례–선 겹침 점검:", "OK" if legend_clear else f"CHECK {under} vs legend bottom {ly0:.1f}", "· 범례–y축 간격", leg_left_gap_in, "in", flush=True)
    print("글줄 폭 점검(열 폭 안):", "OK" if fits else "CHECK", widths, flush=True)
    print("RMSE 열–축 간격:", rm_gap, "in", flush=True)
    (M.OUT / f"{STEM}_spec.json").write_text(json.dumps(SPEC, ensure_ascii=False, indent=1, default=float))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", action="store_true", help="호환용. 이 그림은 슬라이드판 하나만 만든다")
    ap.add_argument("--fig", default="G2", help="호환용")
    ap.parse_args()
    M.OUT.mkdir(parents=True, exist_ok=True)
    g2()
    print("출력:", M.OUT / f"{STEM}.png", M.OUT / f"{STEM}.pdf", M.OUT / f"{STEM}_spec.json")
