# -*- coding: utf-8 -*-
"""v6 방법 도식(국문 슬라이드판, 12.0 × 5.2 in, 300 dpi). 셀-연도 설계(2026-10-10 재설정) 기준.

그림
- V1_time_design   시간 정의: 기존 방식 대 이 연구, 관측일까지의 융해 도일(실자료), 셀-연도 표
- V2_workflow      작업 흐름 여섯 단계(실자료 축소 그림)
- V3_methods       비교한 방법의 구조(물리식, 직접 학습, 물리 입력, 유사라벨 증강, 물리 잔차 결합)
- V4_pseudo_residual  유사라벨 증강과 잔차 모델의 설계(실자료 잔차 분포)
- V5_evaluation    평가 설계: 다섯 분할, 블록 부트스트랩 채점, 제품 비교 조건, 적용 가능 영역

문법: scripts/4_visualization/method_figs_v4.py 의 도구(캔버스, 셰브런, 블록, 직선 화살표, 헤어라인, 글자 QA)를 그대로 쓴다.
출력: deck/assets/v6/method/<name>_slide.png, .pdf. 실행: OMP_NUM_THREADS=4 nice -n 10 python3 scripts/4_visualization/v6_method_figs.py [--only V1,V3]
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import argparse, calendar, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization")); sys.path.insert(0, str(ROOT / "src"))
import method_figs_v4 as M  # noqa: E402
from method_figs_v4 import T, RT, head, chevron, block, arrow, key_line, hline, vline, sub_axes, frame, canvas, INK, AUX, ZERO, HAIR, GRN, YEL, FS  # noqa: E402

PROC = ROOT / "data" / "processed" / "v5"
OUT = ROOT / "deck" / "assets" / "v6" / "method"; OUT.mkdir(parents=True, exist_ok=True); M.OUT = OUT
TOK = M.TOK; C = TOK["color"]
COL = {k: v["hex"] for k, v in C["methods"].items()}
REGC = {"Alaska": C["regions"]["Alaska"], "Lena": C["regions"]["Lena Delta"], "Canada": C["regions"]["Canada"]}
REGK = {"Alaska": "알래스카", "Lena": "레나 델타", "Canada": "캐나다 북서부"}
NAME = {"P0": "원천 계수 Stefan", "P1": "재보정 Stefan", "Ku": "Kudryavtsev", "D0": "직접 학습", "F1": "물리 입력 학습", "D1": "물리 유사라벨 증강", "R1": "물리 잔차 결합"}
RUN = PROC / "runs" / ("core_v6" if (PROC / "runs" / "core_v6" / "summary_metrics.csv").exists() else "core_cb")
FORCING = "year_obs" if RUN.name == "core_v6" else "year"
_C: dict = {}


# ================================================================ 실자료
def labels():
    if "lab" not in _C:
        d = pd.read_csv(PROC / "labels_cell_year_v5.csv"); _C["lab"] = d[d.in_forcing == 1].copy()
    return _C["lab"]


def forcing():
    if "fy" not in _C:
        _C["fy"] = pd.read_csv(PROC / "forcing_year.csv")
    return _C["fy"]


def daily_tdd(loc_id, year):
    """월 누적 TDD(tdd_cum_1..12)의 선형 보간으로 일 누적 곡선. 반환 (doy, cum)."""
    r = forcing(); r = r[(r.loc_id == loc_id) & (r.year == year)].iloc[0]
    days = np.array([calendar.monthrange(int(year), m)[1] for m in range(1, 13)]); starts = np.r_[0, np.cumsum(days)]
    cum = np.array([r[f"tdd_cum_{m}"] for m in range(1, 13)], float)
    doy = np.arange(1, starts[-1] + 1)
    return doy, np.interp(doy, np.r_[0, starts[1:]], np.r_[0, cum])


def preds(region, method, scheme="block", lam=None):
    import glob
    pat = str(RUN / "preds" / f"{region}__{scheme}__{FORCING}__F8TS__{method}__*__lam{'*' if lam is None else lam}.csv.gz")
    fs = sorted(glob.glob(pat))
    if lam is None and method == "R1":
        fs = [f for f in fs if "lam0.5" in f]
    p = pd.concat([pd.read_csv(f) for f in fs[:1]])
    return p.groupby("cell_year_id").agg(y=("y", "first"), pred=("pred", "mean"), block=("block", "first"), year=("year", "first"))


def mini_axes(fig, x, y, w, h, size=None):
    ax = sub_axes(fig, x, y, w, h)
    size = size or FS["small"]
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_linewidth(0.8); ax.spines[s_].set_color(ZERO)
    ax.tick_params(labelsize=size, width=0.8, length=3, colors=INK, pad=2)
    return ax


# ================================================================ V1 시간 정의
def v1_time_design():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    # ---- a 기존 방식과 이 연구
    head(ax, 0.0, 5.0, "a  예측 대상과 입력의 시간 정의")
    rows = [("기존 제품·선행 연구", "여러 해 평균 기후", "지점의 다년 평균", "연도 없는 지도 한 장", AUX),
            ("이 연구", "그 해 ERA5-Land,\n관측일까지 누적", "셀·연도의 관측값", "해마다 다른 지도\nALT(셀, 연도)", INK)]
    cols = [0.0, 2.0, 4.15, 5.95]
    ytop = 4.55
    for j, t in enumerate(("구분", "기후 입력", "라벨(정답)", "출력")):
        T(ax, cols[j], ytop, t, size=FS["small"], color=AUX)
    hline(ax, 0.0, 7.6, ytop - 0.17, color=HAIR)
    for i, (a, b, c, d, col) in enumerate(rows):
        yy = ytop - 0.50 - i * 0.78
        T(ax, cols[0], yy, a, size=FS["body"], color=col, linespacing=1.15)
        T(ax, cols[1], yy, b, size=FS["body"], color=col, linespacing=1.15)
        T(ax, cols[2], yy, c, size=FS["body"], color=col, linespacing=1.15)
        T(ax, cols[3], yy, d, size=FS["body"], color=col, linespacing=1.15)
    hline(ax, 0.0, 7.6, ytop - 0.50 - 0.39, color=HAIR)
    hline(ax, 0.0, 7.6, ytop - 0.50 - 1.17, color=HAIR)
    T(ax, 0.0, 2.70, "같은 셀의 다른 해는 다른 행. 입력·라벨·채점이 같은 해를 가리킨다", size=FS["small"], color=AUX)
    # ---- c 셀-연도 표(실자료 세 행)
    head(ax, 0.0, 2.25, "c  셀-연도 표(실제 행 세 개)")
    lab = labels(); fy = forcing()
    multi = lab[lab.region == "Alaska"].groupby("loc_id").year.nunique(); lid = multi[multi >= 3].index[0]
    ex = lab[(lab.loc_id == lid)].sort_values("year").iloc[[0, 1]]
    ex2 = lab[(lab.region == "Lena")].sort_values("year").iloc[[len(lab[lab.region == "Lena"]) // 2]]
    ex = pd.concat([ex, ex2]).merge(fy[["loc_id", "year", "e5_tdd"]], on=["loc_id", "year"], how="left")
    hdr = ["지역", "셀", "연도", "관측일", "ALT (cm)", "TDD (°C·일)", "정적 공변량", "제품 값"]
    cx = [0.0, 1.15, 1.75, 2.35, 3.05, 3.95, 5.05, 6.35]
    for j, t in enumerate(hdr):
        T(ax, cx[j], 1.85, t, size=FS["small"], color=AUX)
    hline(ax, 0.0, 7.6, 1.70, color=HAIR)
    for i, r in enumerate(ex.itertuples()):
        yy = 1.45 - i * 0.36
        doy = "" if not np.isfinite(r.obs_doy) else pd.Timestamp(int(r.year), 1, 1) + pd.Timedelta(days=int(r.obs_doy) - 1)
        vals = [REGK[r.region], "A" if r.region == "Alaska" else "B", str(int(r.year)), doy.strftime("%m-%d") if doy != "" else "8-31 가정",
                f"{r.alt_cm:.0f}", f"{r.e5_tdd:.0f}", "지형·토양·위도", f"그 해({int(r.year)}) 값"]
        for j, t in enumerate(vals):
            T(ax, cx[j], yy, t, size=FS["small"], color=INK)
    hline(ax, 0.0, 7.6, 0.20, color=HAIR)
    # ---- b 관측일까지의 융해 도일(실자료)
    head(ax, 8.0, 5.0, "b  관측일까지의 융해 도일(실자료)")
    g = lab[(lab.region == "Alaska") & lab.obs_doy.notna() & (lab.year == 2014)]
    r0 = g.iloc[len(g) // 2]
    doy, cum = daily_tdd(r0.loc_id, 2014)
    ydoy = int(r0.obs_doy); tdd_obs = float(np.interp(ydoy, doy, cum)); tdd_year = float(cum[-1])
    clim = forcing(); clim = clim[(clim.loc_id == r0.loc_id) & clim.year.between(2011, 2024)].e5_tdd.mean()
    axb = mini_axes(fig, 8.6, 1.55, 3.2, 3.0)
    axb.plot(doy, cum, color=COL["P1"], lw=2.0)
    axb.axhline(clim, color=AUX, lw=1.2, ls=(0, (4, 2)))
    axb.plot([ydoy, ydoy], [0, tdd_obs], color=ZERO, lw=0.9); axb.plot([0, ydoy], [tdd_obs, tdd_obs], color=ZERO, lw=0.9)
    axb.plot([0, 365], [tdd_year, tdd_year], color=COL["P1"], lw=0.9, ls=(0, (2, 2)))
    axb.plot([ydoy], [tdd_obs], "o", ms=7, color=COL["R1"], zorder=5)
    axb.set_xlim(0, 380); axb.set_ylim(0, max(tdd_year, clim) * 1.25); axb.set_yticks([200, 400])
    axb.set_xticks([91, 182, 274, 365]); axb.set_xticklabels(["4월", "7월", "10월", "12월"])
    axb.set_ylabel("누적 융해 도일 (°C·일)", fontsize=FS["small"]); axb.set_xlabel("2014년, 알래스카 셀 A", fontsize=FS["small"], labelpad=2)
    axb.text(ydoy + 6, tdd_obs * 0.55, f"관측일 {pd.Timestamp(2014, 1, 1) + pd.Timedelta(days=ydoy - 1):%m-%d}", fontsize=FS["small"], color=INK, ha="left", va="center")
    axb.text(8, tdd_obs + tdd_year * 0.035, f"관측일까지 {tdd_obs:.0f}", fontsize=FS["small"], color=COL["R1"], ha="left", va="bottom")
    axb.text(8, tdd_year + tdd_year * 0.03, f"그 해 전체 {tdd_year:.0f}", fontsize=FS["small"], color=COL["P1"], ha="left", va="bottom")
    axb.text(8, clim - tdd_year * 0.035, f"2011–2024 평균 {clim:.0f}", fontsize=FS["small"], color=AUX, ha="left", va="top")
    T(ax, 8.0, 0.85, "세 가지 기후 입력을 같은 셀·같은 해에 비교", size=FS["body"], color=INK)
    T(ax, 8.0, 0.50, "주 결과 = 관측일까지, 보조 = 그 해 전체, 기존 = 평균", size=FS["small"], color=AUX)
    T(ax, 8.0, 0.22, "관측일이 없는 CALM 지점은 8월 31일로 둔다", size=FS["small"], color=AUX)
    return M.qa_and_save(fig, "V1_time_design")


# ================================================================ V2 작업 흐름
def v2_workflow():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    n = 6; gap = 0.06
    cw = (W - 0.04 - gap * (n - 1)) / n; xs = [0.02 + i * (cw + gap) for i in range(n)]
    ych, hch = 4.42, 0.74
    steps = [("① 셀·연도 자료", GRN), ("② 그 해 강제력", GRN), ("③ 물리 기준선", GRN), ("④ 학습 결합 4종", YEL), ("⑤ 다섯 가지 분할", GRN), ("⑥ 제품 비교·\n연도별 지도", GRN)]
    for i, (s_, f) in enumerate(steps):
        chevron(ax, xs[i], ych, cw + (0.0 if i == n - 1 else 0.10), hch, f, s_, first=(i == 0), size=FS["chev"])
    ty0, th = 2.05, 2.15; pad = 0.12; tw = cw - 2 * pad
    lab = labels()
    # ① 세 지역 셀 위치(북극 투영)
    import cartopy.crs as ccrs, cartopy.feature as cfeature, matplotlib.path as mpath
    proj = ccrs.NorthPolarStereo(central_longitude=-100.0)
    d = min(tw, th)
    axm = sub_axes(fig, xs[0] + pad + (tw - d) / 2, ty0 + (th - d) / 2, d, d, projection=proj)
    axm.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
    thh = np.linspace(0, 2 * np.pi, 200); axm.set_boundary(mpath.Path(np.c_[np.sin(thh), np.cos(thh)] * 0.5 + 0.5), transform=axm.transAxes)
    axm.add_feature(cfeature.LAND.with_scale("110m"), facecolor=M.LAND, edgecolor="none", zorder=0)
    axm.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=M.COAST, linewidth=0.4, zorder=0.5)
    for r in ("Alaska", "Canada", "Lena"):
        q = lab[lab.region == r].drop_duplicates("block")
        axm.scatter(q.lon, q.lat, s=14, color=REGC[r], edgecolor="white", lw=0.3, transform=ccrs.PlateCarree(), zorder=3)
    axm.spines["geo"].set_edgecolor(HAIR); axm.spines["geo"].set_linewidth(0.7)
    # ② 일 누적 TDD 곡선(실자료)
    g = lab[(lab.region == "Alaska") & lab.obs_doy.notna() & (lab.year == 2014)]; r0 = g.iloc[len(g) // 2]
    doy, cum = daily_tdd(r0.loc_id, 2014); yd = int(r0.obs_doy); to = float(np.interp(yd, doy, cum))
    ax2 = mini_axes(fig, xs[1] + pad + 0.34, ty0 + 0.42, tw - 0.40, th - 0.55)
    ax2.plot(doy, cum, color=COL["P1"], lw=1.8); ax2.plot([yd], [to], "o", ms=6, color=COL["R1"]); ax2.plot([yd, yd], [0, to], color=ZERO, lw=0.8)
    ax2.set_xticks([1, 182, 365]); ax2.set_xticklabels(["1월", "7월", "12월"]); ax2.set_yticks([0, 1000]); ax2.set_ylabel("TDD", fontsize=FS["small"], labelpad=1)
    # ③ ALT 대 √TDD(레나, 실자료) 와 재보정 직선
    fy = forcing(); q = lab[lab.region == "Lena"].merge(fy[["loc_id", "year", "e5_sqrt_tdd"]], on=["loc_id", "year"])
    ax3 = mini_axes(fig, xs[2] + pad + 0.34, ty0 + 0.42, tw - 0.40, th - 0.55)
    ax3.scatter(q.e5_sqrt_tdd, q.alt_cm, s=3, color=INK, alpha=0.25, lw=0)
    E = float((q.e5_sqrt_tdd * q.alt_cm).sum() / (q.e5_sqrt_tdd ** 2).sum()); xx = np.linspace(q.e5_sqrt_tdd.min(), q.e5_sqrt_tdd.max(), 10)
    ax3.plot(xx, E * xx, color=COL["P1"], lw=2.0); ax3.set_ylim(0, 110); ax3.set_xlabel("√TDD", fontsize=FS["small"], labelpad=1); ax3.set_ylabel("ALT (cm)", fontsize=FS["small"], labelpad=1)
    # ④ 결합 구조 4종
    rows4 = [("D0", "직접 학습", "물리 정보 없음"), ("F1", "물리 입력 학습", "물리 출력을 입력에"), ("D1", "유사라벨 증강", "미관측 셀에 물리 값"), ("R1", "물리 잔차 결합", "물리 값 + 잔차")]
    y4 = ty0 + th - 0.12
    for k, (code, nm, where) in enumerate(rows4):
        yy = y4 - k * 0.54
        key_line(ax, xs[3] + pad, yy, COL[code], w=0.26, lw=3.0)
        T(ax, xs[3] + pad + 0.36, yy, nm, size=FS["body"], color=COL[code] if code == "R1" else INK)
        T(ax, xs[3] + pad + 0.36, yy - 0.24, where, size=FS["small"], color=AUX)
    # ⑤ 블록 분할(레나, 실자료): 블록 색 = 겹
    import polar.testbed_v5 as TB
    q = lab[lab.region == "Lena"]; fold = TB._balanced_group_folds(q.block.values, 5, 0)
    d5 = min(tw, th - 0.5)
    ax5 = sub_axes(fig, xs[4] + pad + (tw - d5) / 2, ty0 + 0.5 + (th - 0.5 - d5) / 2, d5, d5)
    cols5 = ["#EFD98A", "#9DC7A4", "#AABFD6", "#D9A7C7", "#C9C9C9"]
    for f_ in range(5):
        qq = q[fold == f_]; ax5.scatter(qq.lon, qq.lat, s=10, color=cols5[f_], lw=0)
    ax5.set_aspect(1 / np.cos(np.deg2rad(72.5))); ax5.set_xticks([]); ax5.set_yticks([]); frame(ax5)
    T(ax, xs[4] + pad, ty0 + 0.30, "색 = 채점 겹(블록 단위)", size=FS["small"], color=AUX)
    T(ax, xs[4] + pad, ty0 + 0.06, "레나 델타 2,206 셀", size=FS["small"], color=AUX)
    # ⑥ 연도별 1 km 지도(실자료, 알래스카 2014 예측)
    mp = PROC / "maps" / "Alaska" / "pred_2014.csv.gz"
    if mp.exists():
        m = pd.read_csv(mp, dtype={"cell_id": str}); cells = pd.read_csv(PROC.parent / "map_alaska" / "alaska_grid_cells_v1.csv", dtype={"cell_id": str})
        m = m.drop(columns=[c for c in ("lat", "lon") if c in m]).merge(cells[["cell_id", "lat", "lon"]], on="cell_id").iloc[::6]
        ax6 = sub_axes(fig, xs[5] + pad, ty0 + 0.25, tw, th - 0.35)
        from matplotlib.colors import Normalize
        from cmcrameri import cm as cmc
        lo, hi = np.nanpercentile(m.r1, [2, 98])
        ax6.scatter(m.lon, m.lat, c=m.r1, s=0.6, cmap=getattr(cmc, "oslo_r", cmc.oslo.reversed()), norm=Normalize(lo, hi), lw=0, rasterized=True)
        ax6.set_aspect(1 / np.cos(np.deg2rad(65))); ax6.set_xticks([]); ax6.set_yticks([]); frame(ax6)
        T(ax, xs[5] + pad, ty0 + 0.06, "알래스카 2014년 지도", size=FS["small"], color=AUX)
    # 설명 줄
    yl = [1.70 - k * 0.33 for k in range(4)]
    lines = {0: ["세 지역 19,172 셀-연도", "2010–2024, 1 km 셀", "지형·토양 정적 입력"],
             1: ["ERA5-Land 월별 기온", "관측일까지 누적 TDD", "겨울 FDD · 적설"],
             2: ["ALT = E √TDD", "E 는 학습 겹에서 재보정", "Kudryavtsev 식 병행"],
             3: ["같은 학습기 CatBoost", "물리 정보 위치만 다름", "신경망 4종은 보조"],
             4: ["무작위 · 블록 · 연도", "블록×연도 · 지역", "블록 부트스트랩 CI"],
             5: ["같은 셀·같은 해 채점", "학습 지점 25 km 제외", "적용 가능 영역 표시"]}
    for i, L in lines.items():
        for k, s_ in enumerate(L):
            T(ax, xs[i] + pad, yl[k], s_, size=FS["body"])
    return M.qa_and_save(fig, "V2_workflow")


# ================================================================ V3 방법 구조
def v3_methods():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    T(ax, 0.0, 5.0, "입력", size=FS["head"], color=M.HEAD, weight="bold")
    T(ax, 3.1, 5.0, "구조", size=FS["head"], color=M.HEAD, weight="bold")
    T(ax, 9.45, 5.0, "물리 정보의 위치 · 라벨 사용", size=FS["head"], color=M.HEAD, weight="bold")
    # 입력 열(공통)
    inputs = [("그 해 강제력", "관측일까지 TDD, FDD, 적설"), ("정적 공변량", "지형·토양·위도 등 8종"), ("라벨", "ALT(셀, 연도), 학습 겹만")]
    for k, (a, b) in enumerate(inputs):
        yy = 4.35 - k * 0.62
        T(ax, 0.0, yy, a, size=FS["body"], color=INK); T(ax, 0.0, yy - 0.24, b, size=FS["small"], color=AUX)
    vline(ax, 2.85, 0.15, 4.65, color=HAIR)
    rows = ["P1", "D0", "F1", "D1", "R1"]
    y0s = {"P1": 4.30, "D0": 3.42, "F1": 2.56, "D1": 1.70, "R1": 0.72}
    xL = 3.1
    desc = {"P1": ("물리식 자체", "라벨은 계수 E 하나에만"),
            "D0": ("물리 정보 없음", "라벨 전부를 직접 학습"),
            "F1": ("물리 출력을 입력에", "라벨 전부, 물리 값은 특징"),
            "D1": ("물리 값을 라벨로", "실제 라벨 + 유사라벨(가중 r)"),
            "R1": ("물리 값이 기준", "학습은 잔차만, λ = 0.5")}
    for code in rows:
        y = y0s[code]
        key_line(ax, xL, y, COL[code], w=0.26, lw=3.0)
        T(ax, xL + 0.36, y, NAME[code], size=FS["body"], color=COL["R1"] if code == "R1" else INK)
        T(ax, 9.45, y, desc[code][0], size=FS["body"], color=INK); T(ax, 9.45, y - 0.25, desc[code][1], size=FS["small"], color=AUX)
    bx = 6.35; bw = 1.25; bh = 0.40
    # P1: [TDD] → 물리 블록 → ŷ_P
    y = y0s["P1"]; T(ax, bx - 0.95, y, "TDD", size=FS["small"], ha="center"); arrow(ax, (bx - 0.70, y), (bx, y))
    block(ax, bx, y - bh / 2, bw, bh, "Stefan  E√TDD", fill=GRN, size=FS["small"]); arrow(ax, (bx + bw, y), (bx + bw + 0.55, y))
    RT(ax, bx + bw + 0.62, y, ["$\\hat{y}_P$"], size=FS["body"])
    # D0: [X, TDD] → CatBoost → ŷ
    y = y0s["D0"]; T(ax, bx - 0.95, y, "X, TDD", size=FS["small"], ha="center"); arrow(ax, (bx - 0.60, y), (bx, y))
    block(ax, bx, y - bh / 2, bw, bh, "CatBoost", fill=YEL, size=FS["small"]); arrow(ax, (bx + bw, y), (bx + bw + 0.55, y))
    RT(ax, bx + bw + 0.62, y, ["$\\hat{y}$"], size=FS["body"])
    # F1: [X, TDD, ŷ_P] → CatBoost → ŷ
    y = y0s["F1"]; RT(ax, bx - 0.95, y, ["X, TDD, ", "$\\hat{y}_P$"], size=FS["small"], ha="center"); arrow(ax, (bx - 0.45, y), (bx, y))
    block(ax, bx, y - bh / 2, bw, bh, "CatBoost", fill=YEL, size=FS["small"]); arrow(ax, (bx + bw, y), (bx + bw + 0.55, y))
    RT(ax, bx + bw + 0.62, y, ["$\\hat{y}$"], size=FS["body"])
    # D1: 라벨 ∪ 유사라벨 → CatBoost → ŷ
    y = y0s["D1"]; T(ax, bx - 0.95, y + 0.14, "라벨 셀-연도", size=FS["small"], ha="center"); RT(ax, bx - 0.95, y - 0.16, ["+ 미관측 셀의 ", "$\\hat{y}_P$"], size=FS["small"], ha="center", color=[COL["P1"], COL["P1"]])
    arrow(ax, (bx - 0.30, y), (bx, y))
    block(ax, bx, y - bh / 2, bw, bh, "CatBoost", fill=YEL, size=FS["small"]); arrow(ax, (bx + bw, y), (bx + bw + 0.55, y))
    RT(ax, bx + bw + 0.62, y, ["$\\hat{y}$"], size=FS["body"])
    # R1: ŷ_P + λ·CatBoost(X,TDD → y − ŷ_P)
    y = y0s["R1"]; T(ax, bx - 0.95, y, "X, TDD", size=FS["small"], ha="center"); arrow(ax, (bx - 0.60, y), (bx, y))
    block(ax, bx, y - bh / 2, bw, bh, "CatBoost(잔차)", fill=YEL, size=FS["small"])
    arrow(ax, (bx + bw, y), (bx + bw + 0.40, y)); RT(ax, bx + bw + 0.46, y, ["$\\lambda$"], size=FS["body"])
    M.plus_node(ax, bx + bw + 0.95, y); arrow(ax, (bx + bw + 0.66, y), (bx + bw + 0.85, y))
    RT(ax, bx + bw + 0.95, y + 0.42, ["$\\hat{y}_P$"], size=FS["body"], ha="center"); arrow(ax, (bx + bw + 0.95, y + 0.30), (bx + bw + 0.95, y + 0.10))
    arrow(ax, (bx + bw + 1.05, y), (bx + bw + 1.35, y)); RT(ax, bx + bw + 1.42, y, ["$\\hat{y}$"], size=FS["body"])
    T(ax, bx - 0.95, y - 0.36, "학습 목표: y − ŷ_P", size=FS["small"], color=M.TRAIN_RED, ha="center")
    hline(ax, 3.1, 12.0, 0.25, color=HAIR)
    T(ax, 3.1, 0.16, "X 정적 공변량 · TDD 그 해 융해 도일 · ŷ_P 재보정 Stefan 값 · r 잔차 예측", size=FS["small"], color=AUX)
    return M.qa_and_save(fig, "V3_methods")


# ================================================================ V4 유사라벨·잔차 설계
def v4_pseudo_residual():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    lab = labels()
    # ---- a 유사라벨 증강(레나 실자료): 라벨 셀 + 미관측 격자 셀(물리 값)
    head(ax, 0.0, 5.0, "a  물리 유사라벨 증강의 설계")
    q = lab[lab.region == "Lena"].drop_duplicates("loc_id")
    grid = PROC / "grid" / "Lena" / "forcing_year.csv"
    axa = sub_axes(fig, 0.15, 1.75, 2.75, 2.75)
    if (PROC / "maps" / "Lena" / "mean_2011_2024.csv.gz").exists():
        m = pd.read_csv(PROC / "maps" / "Lena" / "mean_2011_2024.csv.gz", dtype={"cell_id": str})
        cells = pd.read_csv(PROC.parent / "map_lena" / "lena_grid_cells_v1.csv", dtype={"cell_id": str})
        m = m.drop(columns=[c for c in ("lat", "lon") if c in m]).merge(cells[["cell_id", "lat", "lon"]], on="cell_id").iloc[::3]
        axa.scatter(m.lon, m.lat, s=1.2, color=COL["P1"], alpha=0.25, lw=0, rasterized=True)
    axa.scatter(q.lon, q.lat, s=6, color=INK, lw=0)
    axa.set_aspect(1 / np.cos(np.deg2rad(72.5))); axa.set_xticks([]); axa.set_yticks([]); frame(axa)
    key_line(ax, 0.15, 1.45, INK, w=0.26, lw=3.0); T(ax, 0.5, 1.45, "관측 셀(라벨)", size=FS["small"])
    key_line(ax, 0.15, 1.15, COL["P1"], w=0.26, lw=3.0); T(ax, 0.5, 1.15, "미관측 셀: 재보정 Stefan 값을 라벨로", size=FS["small"])
    items = [("학습 집합", "라벨 n개 + 유사라벨 n개(r = 1)"),
             ("유사라벨 값", "학습 겹 계수 E 의 그 해 Stefan 값"),
             ("유사라벨 위치", "채점 겹 밖 미관측 셀, 무작위"),
             ("기대 효과", "라벨 없는 조합에 물리 경향 전달")]
    for k, (a, b) in enumerate(items):
        yy = 4.45 - k * 0.72
        T(ax, 3.2, yy, a, size=FS["body"], color=INK); T(ax, 3.2, yy - 0.26, b, size=FS["small"], color=AUX)
    T(ax, 3.2, 1.45, "결과: 세 지역 모두 직접 학습과 차이 없음", size=FS["small"], color=AUX)
    T(ax, 3.2, 1.18, "(유사라벨은 물리식 정보를 되풀이할 뿐)", size=FS["small"], color=AUX)
    vline(ax, 6.05, 0.3, 5.1, color=HAIR)
    # ---- b 잔차 결합: 실자료 잔차 분포
    head(ax, 6.3, 5.0, "b  물리 잔차 결합의 설계와 잔차 분포(블록 교차검증)")
    axb = mini_axes(fig, 9.55, 1.75, 2.3, 2.65)
    rP = preds("Lena", "P1"); rR = preds("Lena", "R1")
    j = rP.join(rR, lsuffix="_p", rsuffix="_r", how="inner")
    bins = np.linspace(-60, 60, 41)
    axb.hist(j.y_p - j.pred_p, bins=bins, histtype="step", color=COL["P1"], lw=1.8, label=f"관측 − Stefan, s.d. {np.std(j.y_p - j.pred_p):.1f}")
    axb.hist(j.y_r - j.pred_r, bins=bins, histtype="step", color=COL["R1"], lw=1.8, label=f"관측 − 잔차 결합, s.d. {np.std(j.y_r - j.pred_r):.1f}")
    axb.axvline(0, color=ZERO, lw=0.8); axb.set_xlabel("오차 (cm), 레나 델타", fontsize=FS["small"]); axb.set_ylabel("셀-연도 수", fontsize=FS["small"])
    axb.legend(fontsize=FS["small"], frameon=False, loc="upper left", bbox_to_anchor=(-0.04, 1.02), handlelength=1.0, labels=[f"Stefan, s.d. {np.std(j.y_p - j.pred_p):.1f}", f"잔차 결합, s.d. {np.std(j.y_r - j.pred_r):.1f}"])
    items = [("기준", "재보정 Stefan 값 ŷ_P 를 둔다"),
             ("학습 목표", "잔차 y − ŷ_P 만 CatBoost 로"),
             ("수축 λ", "잔차 예측 × 0.5 를 더한다"),
             ("뜻", "물리가 못 맞춘 지형·토양 효과만")]
    for k, (a, b) in enumerate(items):
        yy = 4.45 - k * 0.72
        T(ax, 6.3, yy, a, size=FS["body"], color=INK); T(ax, 6.3, yy - 0.26, b, size=FS["small"], color=AUX)
    RT(ax, 6.3, 1.25, ["$\\hat{y} = \\hat{y}_P + \\lambda\\,\\hat{r}(X, \\mathrm{TDD})$", ",  λ = 0.5"], size=FS["body"])
    T(ax, 6.3, 0.85, "λ 0.25 · 1 · 내부 CV 선택도 비교", size=FS["small"], color=AUX)
    T(ax, 6.3, 0.55, "λ 작으면 물리, 크면 직접 학습에 가깝다", size=FS["small"], color=AUX)
    return M.qa_and_save(fig, "V4_pseudo_residual")


# ================================================================ V5 평가 설계
def v5_evaluation():
    W, H = 12.0, 5.2
    fig, ax = canvas(W, H)
    import polar.testbed_v5 as TB
    lab = labels(); q = lab[lab.region == "Lena"].copy().reset_index(drop=True)
    head(ax, 0.0, 5.0, "a  다섯 가지 분할(레나 델타, 색 = 채점 겹)")
    rng = np.random.default_rng(0)
    cells = q.drop_duplicates("loc_id").loc_id.values; rmap = dict(zip(cells, rng.integers(0, 5, len(cells))))
    f_rand = q.loc_id.map(rmap).values
    f_blk = TB._balanced_group_folds(q.block.values, 5, 0)
    yrs = sorted(q.year.unique()); ymap = {y: i % 5 for i, y in enumerate(yrs)}; f_year = q.year.map(ymap).values
    f_by = (f_blk + f_year) % 5
    schemes = [("무작위", f_rand, "셀을 무작위로 5겹"), ("공간 블록", f_blk, "0.5° 블록 단위 5겹"), ("연도", f_year, "관측 연도 단위"), ("블록 × 연도", f_by, "블록과 연도를 함께"), ("지역", None, "다른 두 지역으로 학습")]
    cols5 = ["#EFD98A", "#9DC7A4", "#AABFD6", "#D9A7C7", "#C9C9C9"]
    for i, (nm, f_, sub) in enumerate(schemes):
        x0 = 0.05 + i * 1.22
        axs = sub_axes(fig, x0, 3.35, 1.1, 1.1)
        if f_ is None:
            axs.scatter(q.lon, q.lat, s=2, color=cols5[1], lw=0)
            axs.text(0.5, 0.5, "전부 채점", transform=axs.transAxes, ha="center", va="center", fontsize=FS["small"], color=INK)
        else:
            for k in range(5):
                qq = q[f_ == k]; axs.scatter(qq.lon, qq.lat, s=2, color=cols5[k], lw=0)
        axs.set_aspect(1 / np.cos(np.deg2rad(72.5))); axs.set_xticks([]); axs.set_yticks([]); frame(axs)
        T(ax, x0 + 0.55, 3.12, nm, size=FS["body"], ha="center")
    qs = [("무작위", "셀을 무작위로 5겹. 같은 셀의 다른 해가 학습에 있어 과대평가 기준선"), ("공간 블록", "0.5° 블록 단위 5겹. 주 결과: 새 장소에서의 정확도"), ("연도", "관측 연도 단위. 새 해에서의 정확도"), ("블록 × 연도", "블록과 연도를 함께. 새 장소·새 해를 동시에"), ("지역", "다른 두 지역으로 학습. 새 지역으로 옮길 때")]
    for k, (a, b) in enumerate(qs):
        yy = 2.60 - k * 0.40
        T(ax, 0.05, yy, a, size=FS["small"], color=INK); T(ax, 1.35, yy, b, size=FS["small"], color=AUX)
    vline(ax, 6.35, 0.3, 5.1, color=HAIR)
    # ---- b 채점: 블록 부트스트랩 짝지은 ΔRMSE
    head(ax, 6.6, 5.0, "b  채점 규칙")
    rP = preds("Lena", "P1"); rR = preds("Lena", "R1"); j = rP.join(rR, lsuffix="_p", rsuffix="_r", how="inner")
    blocks = j.block_p.unique(); rng = np.random.default_rng(0)
    def rmse(a):
        return float(np.sqrt(np.mean(a ** 2)))
    deltas = []
    for _ in range(2000):
        bs = rng.choice(blocks, len(blocks), replace=True); s = pd.concat([j[j.block_p == b] for b in bs])
        deltas.append(rmse(s.y_r - s.pred_r) - rmse(s.y_p - s.pred_p))
    deltas = np.array(deltas); lo, hi = np.percentile(deltas, [2.5, 97.5]); d0 = rmse(j.y_r - j.pred_r) - rmse(j.y_p - j.pred_p)
    axb = mini_axes(fig, 7.0, 3.15, 2.1, 1.45)
    axb.hist(deltas, bins=30, color=COL["R1"], alpha=0.75, lw=0); axb.axvline(0, color=ZERO, lw=0.9)
    axb.set_yticks([]); axb.spines["left"].set_visible(False); axb.set_xlabel("RMSE 차, 잔차 결합 − Stefan (cm)", fontsize=FS["small"])
    axb.text(0.02, 0.95, f"레나 델타  {d0:+.1f} [{lo:+.1f}, {hi:+.1f}]", transform=axb.transAxes, fontsize=FS["small"], va="top")
    items = [("짝지은 차이", "같은 셀-연도, 두 방법의 RMSE 차"), ("블록 부트스트랩", "블록 2,000회 재추출, 95% 구간"), ("판정", "구간이 0 을 제외하면 차이 있음"), ("보조 지표", "편향 · 상관 · 연도 편차 상관")]
    for k, (a, b) in enumerate(items):
        yy = 2.20 - k * 0.55
        T(ax, 6.6, yy, a, size=FS["small"], color=INK); T(ax, 6.6, yy - 0.25, b, size=FS["small"], color=AUX)
    vline(ax, 9.65, 0.3, 5.1, color=HAIR)
    # ---- c 제품 비교 조건
    head(ax, 9.9, 5.0, "c  제품 비교 조건")
    axc = sub_axes(fig, 10.0, 3.1, 1.9, 1.5)
    rng = np.random.default_rng(1); px, py = rng.uniform(0, 10, 60), rng.uniform(0, 8, 60)
    sx, sy = np.array([2.5, 7.0]), np.array([5.5, 2.5])
    for x_, y_ in zip(sx, sy):
        axc.add_patch(Circle((x_, y_), 1.6, facecolor="#EFD98A", edgecolor="none", alpha=0.8))
    inside = np.min(np.hypot(px[:, None] - sx[None], py[:, None] - sy[None]), axis=1) < 1.6
    axc.scatter(px[~inside], py[~inside], s=10, color=INK, lw=0); axc.scatter(px[inside], py[inside], s=10, color=AUX, alpha=0.4, lw=0)
    axc.scatter(sx, sy, s=26, marker="^", color=M.TRAIN_RED, lw=0)
    axc.set_xlim(0, 10); axc.set_ylim(0, 8); axc.set_xticks([]); axc.set_yticks([]); frame(axc)
    items = [("같은 셀·같은 해", "제품의 그 해 값 대 OOF 예측"), ("누설 제외", "학습 지점(▲) 25 km 안은 뺌"), ("공통 셀", "모든 제품이 값을 가진 셀만"), ("적용 범위", "학습 자료와 먼 셀은 흐리게")]
    for k, (a, b) in enumerate(items):
        yy = 2.20 - k * 0.55
        T(ax, 9.9, yy, a, size=FS["small"], color=INK); T(ax, 9.9, yy - 0.25, b, size=FS["small"], color=AUX)
    return M.qa_and_save(fig, "V5_evaluation")


FIGS = {"V1": v1_time_design, "V2": v2_workflow, "V3": v3_methods, "V4": v4_pseudo_residual, "V5": v5_evaluation}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None); a = ap.parse_args()
    M.setup_fonts()
    for k in (a.only.split(",") if a.only else FIGS):
        FIGS[k]()
    json.dump(M.LOG, open(OUT / "qa_log.json", "w"), ensure_ascii=False, indent=1)
