"""Fig 7 v3: 라벨이 많은 지역과 워크플로(그림 명세 outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md 9절).

주장(설명문 첫 문장, C4·C8 축소판): 알래스카 지역 내에서 재보정 Stefan 대비 잔차 ML 의 이득은 1 cm 미만이었고
기후 격자 단위 편향 보정에서 왔다. 그림에는 주장을 쓰지 않고 근거만 그린다(지침 1.1).

패널(그림 명세 9.2, 9.3)
  a  지역 내 시험(0.5° 블록 절반 채점, 분할 25회)의 오차 변화(방법 − 재보정 Stefan), 알래스카·레나델타·캐나다.
     점 + 셀 가중 CI(굵게) + 블록 등가중 CI(가늘게), 0선, ±0.5 cm 띠, 오차 하한 − 재보정 Stefan RMSE 선.
  b  알래스카 라벨 전량, 채점 블록별 오차 변화 지도(2안, 그림 명세 D-11 기본값). 관측 위치 검정 점.
  c  총·격자 사이·격자 안 성분(라벨 전량, 교차검증 잔차 가중) 작은 다중 포레스트.
  d  워크플로 경로(원천 계수 Stefan 대비, 레나델타·캐나다 전이 풀, D-13 기본값)와 배치·진단 띠(별도 축).
  e  워크플로 대비(XC, 2026-10-05 봉인 해제): XC-1(무작위 배치·고정 잔차 레시피 대비, 라벨 40·160)과 XC-2(재보정 Stefan 대비,
     라벨 10·40·160)의 풀 값 포레스트(점 + 두 가중 CI, 0선, ±0.5 cm 띠). 재보정 Stefan 대비 0.5 cm 넘게 나빴던 지역은 지역 모양 작은 기호.
     XC-5(무작위 대비, 교차검증 선정)는 시험하지 않음(Algorithm P = S1)이라 묶음을 두지 않는다.

원천(읽기만 한다. 그림 명세 9.3 의 경로와 필터)
  a  results/rescale_wf2/data/processed/wf/wf2b_tests.csv, data/processed/lgx/lgx_floor.csv
  b  results/rescale_wf3/data/processed/wf/shards/wf9__cpu__Alaska__r__s*_blocksse.npz(단위 Alaska|r),
     data/processed/fidelity_base_v3.csv(관측 위치), Natural Earth 50 m land(cartopy 자료 폴더)
  c  results/rescale_wf3/data/processed/wf/wf3b_tests.csv, wf3b_decomp.csv(설명문 수치 대조만)
  d  data/processed/paper_figs/fig2_pool_curves.csv, fig2_region_curves.csv,
     results/rescale_wf/data/processed/wf/wf_curve.csv
  e  data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_hyp.csv(등록 주 가설), xc_contrasts_main.csv(PE1 지역 행),
     xc_contrasts_aux.csv(Tibet S-XC8|PE2, Alaska S-XC8|AUX3 지역 행), xc_meta.json(재표집 횟수)

산출(작업 지시가 정한 파일만)
  outputs/figures/paper/v3_restructure/Fig7.pdf(벡터, 글꼴 내장), Fig7.png(600 dpi),
  Fig7_source_data.csv(패널 값), Fig7_values.txt(값 대조, 점검 기록). 설명문 Fig7_legend.md 는 따로 쓴다.

실행: OMP_NUM_THREADS=2 nice -n 10 python scripts/4_visualization/paper_v3/fig7.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402  (공용 양식, 고치지 않고 가져다 쓴다)

import matplotlib  # noqa: E402
from matplotlib import transforms as mtrans  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

ROOT = S.ROOT
OUT = S.OUT
MM = 1.0 / 25.4
FS = S.FONT_PT
INK, INK2 = S.INK, S.INK_AUX

SRC = dict(
    wf2b_tests=ROOT / "results/rescale_wf2/data/processed/wf/wf2b_tests.csv",
    floor=ROOT / "data/processed/lgx/lgx_floor.csv",
    wf9_shards=ROOT / "results/rescale_wf3/data/processed/wf/shards",
    wf3b_tests=ROOT / "results/rescale_wf3/data/processed/wf/wf3b_tests.csv",
    wf3b_decomp=ROOT / "results/rescale_wf3/data/processed/wf/wf3b_decomp.csv",
    pool=S.PAPER_FIGS / "fig2_pool_curves.csv",
    region_curves=S.PAPER_FIGS / "fig2_region_curves.csv",
    wf_curve=ROOT / "results/rescale_wf/data/processed/wf/wf_curve.csv",
    base=ROOT / "data/processed/fidelity_base_v3.csv",
    xc_hyp=ROOT / "data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_hyp.csv",
    xc_main=ROOT / "data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_contrasts_main.csv",
    xc_aux=ROOT / "data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_contrasts_aux.csv",
    xc_meta=ROOT / "data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_meta.json",
    xc_sent=ROOT / "data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_sentences.json",
)

REGIONS_A = ["Alaska", "Lena", "Canada"]
COLOR_R = S.METHOD["anchor_residual"]["color"]           # #9a7bc9
COLOR_D = S.METHOD["direct_ml"]["color"]                 # #6b7280
COLOR_FLOOR = S.METHOD["error_floor"]["color"]           # #b0b0b0
LON0 = -152.0                                            # 알래스카 지도 중심 경도(블록 경도 평균 −152.7 의 정수)

# ---------------------------------------------------------------- 배치(mm, 그림 왼쪽 위 원점; 그림 명세 9.2 를 170 × 165 안에서 조정)
FIG_W, FIG_H = 170.0, 165.0
L = dict(
    a_letter=(0.0, 0.0), a_top=4.4, a_h=28.0, a_left=10.6, a_wm=41.0, a_gap_all=1.8, a_wa=6.0, a_gap=6.5,
    b_letter=(0.0, 46.0), b_map=(0.0, 48.6, 84.0, 61.4), b_cbar=(12.0, 112.0, 60.0, 2.5),
    c_letter=(94.0, 46.0), c_label_right=116.4, c_x0=117.6, c_wc=15.6, c_gap=2.6, c_top=56.5, c_h=41.0,
    c_key=(117.6, 118.7),
    d_letter=(0.0, 124.0), d_top=125.6, d_h=17.0, d_n0=(13.2, 6.0), d_main=(21.2, 77.8), d_all=(101.0, 6.0),
    d_band=(0.0, 149.6, 110.0, 13.8), d_head_right=11.8,
    e_letter=(118.0, 124.0), e_ax=(132.0, 128.8, 37.6, 27.4), e_name_right=131.0, e_head_x=118.3, e_key=(121.6, 125.5),
)

# ---------------------------------------------------------------- 패널 e(XC, 등록 문서 2.3 주 가설; 그림 명세 9.3 e)
# (묶음 머리, 대비, [(가설, 라벨 수)]). 묶음 머리 문구는 조정 담당 지시(2026-10-05 05:20)
E_GROUPS = [("vs random placement, fixed recipe", "A1-A2", [("XC-1b", 40), ("XC-1c", 160)]),
            ("vs recalibrated Stefan", "A1-A3", [("XC-2a", 10), ("XC-2b", 40), ("XC-2c", 160)])]
E_XLIM, E_XT = (-2.3, 2.7), [-2, -1, 0, 1, 2]
E_WORSE_CM = 0.5            # 재보정 Stefan 대비 지역 예외: 셀 가중 Δ > 0.5 cm(등록 비열등 문장의 지역 규칙)
E_REGION_FROM_TARGET = {"Lena|x": "Lena", "Canada|x": "Canada", "Russia_W|x": "Russia_W", "Russia_E|x": "Russia_E",
                        "Russia_C~lgd|x": "Russia_C_LGD", "Tibet_LGD|x": "Tibet_LGD", "Alaska|x": "Alaska"}
# 봉인 문장(xc_sentences.json)의 지역 예외(라벨 수: 지역)와 조정 담당 목록(러시아 E 10, 알래스카 40·160, 레나 160)
E_WORSE_EXPECT = {10: {"Russia_E"}, 40: {"Alaska"}, 160: {"Lena", "Alaska"}}
# 조정 담당이 준 값(셀 가중 delta [lo, hi] / 블록 등가중 delta [lo, hi]), 대조용
E_EXPECT = {"XC-1b": (-0.86, -1.03, -0.39, -0.36, -0.72, 0.003), "XC-1c": (-0.93, -1.25, -0.40, -0.54, -0.94, -0.13),
            "XC-2a": (0.08, -0.07, 0.20, -0.005, -0.15, 0.14), "XC-2b": (-1.35, -1.64, -0.80, -1.00, -1.42, -0.59),
            "XC-2c": (-1.51, -1.99, -0.88, -1.29, -1.78, -0.80)}
MAX_CHARS = 760             # 그림 명세 9.5 의 600 자에 e 의 묶음 머리 2, 행 이름 5, 지역 열쇠 3, 축 이름을 더한 값(Fig 6 의 800 자와 같은 사유 기록)


# ================================================================ 작은 도우미(이 그림 전용)
def use_rc():
    """공용 use_v3 에 이 그림이 쓰는 값(기울인 n, 눈금 간격)을 더한다(공용 모듈은 고치지 않는다)."""
    S.use_v3("paper")
    matplotlib.rcParams.update({
        "mathtext.it": f"{S.FONT_LATIN}:italic", "mathtext.bf": f"{S.FONT_LATIN}:bold",
        "mathtext.sf": S.FONT_LATIN,
        "axes.unicode_minus": True, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0,
        "axes.facecolor": "none", "savefig.pad_inches": 0.0, "lines.solid_capstyle": "butt",
    })


def fig_wh(fig):
    w, h = fig.get_size_inches()
    return w / MM, h / MM


def text_mm(fig, x, y, s, **kw):
    W, H = fig_wh(fig)
    kw.setdefault("fontsize", FS)
    return fig.text(x / W, 1 - y / H, s, **kw)


def ax_wh_mm(ax):
    W, H = fig_wh(ax.figure)
    p = ax.get_position()
    return p.width * W, p.height * H


def mm_shift(ax, dx_mm=0.0, dy_mm=0.0):
    return ax.transData + mtrans.ScaledTranslation(dx_mm * MM, dy_mm * MM, ax.figure.dpi_scale_trans)


def axmm_text(ax, x_mm, y, s, **kw):
    """x 는 축 왼쪽 끝에서의 mm, y 는 자료 좌표."""
    tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
    kw.setdefault("fontsize", FS)
    return ax.text(x_mm / ax_wh_mm(ax)[0], y, s, transform=tr, **kw)


def data_y_to_fig_mm(ax, y):
    """자료 y → 그림 mm(위 원점). 그리기 전 축 위치만으로 계산(선형 y 축 전용)."""
    W, H = fig_wh(ax.figure)
    p = ax.get_position()
    y0, y1 = ax.get_ylim()
    f = (np.asarray(y, float) - y0) / (y1 - y0)
    return (1 - (p.y0 + f * p.height)) * H


def set_fixed_ticks(axis, values, labels):
    axis.set_major_locator(FixedLocator(list(values)))
    axis.set_major_formatter(FixedFormatter(list(labels)))
    axis.set_minor_locator(NullLocator())


def num_tick(v):
    """눈금 문자열: U+2212, 네 자리 쉼표 없음."""
    s = f"{abs(v):g}" if abs(v - round(v)) > 1e-9 else str(int(round(abs(v))))
    return ("−" if v < -1e-12 else "") + s


def zero_line(ax, axis="y"):
    f = ax.axhline if axis == "y" else ax.axvline
    return f(0.0, color=S.ZERO_LINE["color"], lw=S.ZERO_LINE["lw"], ls="-", zorder=1.5)


def eq_band(ax, axis="y"):
    f = ax.axhspan if axis == "y" else ax.axvspan
    h = S.EQUIV_HALF_WIDTH_CM
    return f(-h, h, facecolor=S.EQUIV_BAND, edgecolor="none", lw=0, zorder=0.5)


def ci_pair(ax, pos, r, color, orient="v", off_mm=0.0, ms=None, zorder=3.0):
    """점추정 + 셀 가중 95% CI(2.0 pt, 둥근 끝) + 블록 등가중 95% CI(1.0 pt, 0.8 mm 어긋남). 그림 명세 1.2."""
    ms = ms or S.MS["main"]
    boff = S.FOREST_BLOCK_OFFSET_MM
    est, lo, hi, lob, hib = (float(r[k]) for k in ("delta", "ci_lo", "ci_hi", "delta_blockeq_lo", "delta_blockeq_hi"))
    if orient == "v":
        tc, tb = mm_shift(ax, off_mm, 0.0), mm_shift(ax, off_mm + boff, 0.0)
        ax.plot([pos, pos], [lo, hi], color=color, lw=S.LW["ci_forest_cell"], solid_capstyle="round", transform=tc, zorder=zorder)
        ax.plot([pos, pos], [lob, hib], color=color, lw=S.LW["ci_forest_block"], solid_capstyle="butt", transform=tb, zorder=zorder)
        ax.plot([pos], [est], ls="none", marker="o", ms=ms, mfc=color, mec="none", transform=tc, zorder=zorder + 0.1)
    else:
        tc, tb = mm_shift(ax, 0.0, off_mm), mm_shift(ax, 0.0, off_mm - boff)
        ax.plot([lo, hi], [pos, pos], color=color, lw=S.LW["ci_forest_cell"], solid_capstyle="round", transform=tc, zorder=zorder)
        ax.plot([lob, hib], [pos, pos], color=color, lw=S.LW["ci_forest_block"], solid_capstyle="butt", transform=tb, zorder=zorder)
        ax.plot([est], [pos], ls="none", marker="o", ms=ms, mfc=color, mec="none", transform=tc, zorder=zorder + 0.1)


def strip_axis(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(axis="y", left=False, labelleft=False)
    if not bottom:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)


# ================================================================ 자료
def load_a():
    """a: WF6-a 지역 내 대비(방법 − 재보정 Stefan)와 오차 하한."""
    t = pd.read_csv(SRC["wf2b_tests"], low_memory=False)
    t = t[(t.test_id == "WF6-a") & (t["item"] == "WF6-a") & t.target.isin([f"{r}|r" for r in REGIONS_A])]
    fl = pd.read_csv(SRC["floor"])
    fl = fl[fl.scope == "eval"].set_index("region").floor_rmse_cm
    rows = []
    for key, pref in (("anchor_residual", "R1(λ cv)-P1|n"), ("direct_ml", "D0(catboost)-P1|n")):
        q = t[t.contrast.fillna("").str.startswith(pref)]
        for _, r in q.iterrows():
            reg = r.target.split("|")[0]
            rows.append(dict(panel="a", region=reg, method=key, n=int(r.n), contrast=r.contrast,
                             delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, delta_blockeq=r.delta_blockeq,
                             delta_blockeq_lo=r.ci_lo_beq, delta_blockeq_hi=r.ci_hi_beq, rmse_method=r.rmse_A,
                             rmse_recal=r.rmse_B, n_splits=r.n_splits, verdict4=r.verdict4, verdict4_common=r.verdict4_common,
                             floor_cm=float(fl[reg]), floor_minus_recal=float(fl[reg]) - r.rmse_B))
    A = pd.DataFrame(rows).sort_values(["region", "method", "n"]).reset_index(drop=True)
    dup = A.duplicated(["region", "method", "n"]).sum()
    assert dup == 0, f"a: duplicated rows {dup}"
    # 같은 (지역, n) 에서 재보정 Stefan RMSE 는 두 방법이 같아야 한다
    chk = A.groupby(["region", "n"]).rmse_recal.nunique().max()
    assert chk == 1, "a: rmse_B differs between methods"
    return A, fl


def block_latlon(bid: int):
    """0.5° 블록 식별자 = floor(lat/0.5)·100000 + floor(lon/0.5)(src/polar/fidelity.py add_group_keys)."""
    ilat = int(np.round(bid / 100000.0))
    ilon = int(bid - ilat * 100000)
    return ilat * 0.5, ilon * 0.5


def load_b():
    """b: WF9 지역 내 알래스카 총 저장소(u0, Alaska|r)에서 교차검증 잔차 가중 잔차 ML(seed 0·1 평균)과
    재보정 Stefan 의 블록별 SSE·셀 수를 분할 25개에 걸쳐 합산해 블록별 RMSE 차를 낸다."""
    files = sorted(glob.glob(str(SRC["wf9_shards"] / "wf9__cpu__Alaska__r__s*_blocksse.npz")))
    KR = [["R1", "catboost_lo", "1", "cell", -1, 0, s, -1.0] for s in (0, 1)]
    KP = ["P1", "none", "1", "cell", -1, 0, -1, 0.0]
    acc, d_split, rA, rB, splits = {}, [], [], [], []
    for f in files:
        with np.load(f) as z:
            meta = json.loads(str(z["meta"]))
            i = [j for j, u in enumerate(meta["units"]) if u["target"] == "Alaska|r"]
            assert len(i) == 1, f
            i = i[0]
            splits.append(int(meta["units"][i]["split"]))
            keys = [json.loads(k) for k in z[f"u{i}_keys"]]
            sse, cnt, blocks, ncell = z[f"u{i}_sse"], z[f"u{i}_cnt"], z[f"u{i}_blocks"], z[f"u{i}_ncell"]
        ir = [keys.index(k) for k in KR]
        ip = keys.index(KP)
        assert all((cnt[j] == cnt[ip]).all() for j in ir)
        a = [np.sqrt(sse[j].sum() / cnt[j].sum()) for j in ir]
        b = np.sqrt(sse[ip].sum() / cnt[ip].sum())
        d_split.append(np.mean(a) - b); rA.append(np.mean(a)); rB.append(b)
        for k, blk in enumerate(blocks):
            s = acc.setdefault(int(blk), dict(sse_resid=0.0, sse_recal=0.0, cnt=0, n_splits_scored=0, ncell=int(ncell[k])))
            s["sse_resid"] += 0.5 * (sse[ir[0], k] + sse[ir[1], k])
            s["sse_recal"] += sse[ip, k]
            s["cnt"] += int(cnt[ip, k])
            s["n_splits_scored"] += 1
    rows = []
    for bid, s in sorted(acc.items()):
        lat0, lon0 = block_latlon(bid)
        rm = np.sqrt(s["sse_resid"] / s["cnt"]); rp = np.sqrt(s["sse_recal"] / s["cnt"])
        rows.append(dict(panel="b", block=bid, lat_s=lat0, lon_w=lon0, rmse_resid=rm, rmse_recal=rp, delta=rm - rp,
                         n_cells_scored_sum=s["cnt"], n_cells_block=s["ncell"], n_splits_scored=s["n_splits_scored"],
                         masked=bool(s["cnt"] < 10)))
    B = pd.DataFrame(rows)
    chk = dict(n_shards=len(files), splits=sorted(splits), delta_cell_weighted=float(np.mean(d_split)),
               rmse_resid=float(np.mean(rA)), rmse_recal=float(np.mean(rB)))
    return B, chk


def loc1km(lat, lon):
    """1 km 위치 색인(ky = floor(lat/0.009), kx = floor(lon·cos φ/0.009)). v2 Table 1 의 정의와 같다."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    ky = np.floor(lat / 0.009).astype(np.int64)
    phi = np.deg2rad((ky + 0.5) * 0.009)
    kx = np.floor(lon * np.cos(phi) / 0.009).astype(np.int64)
    return ky * 10_000_000 + kx


def load_obs():
    """알래스카 라벨의 1 km 위치(지역 내 시험이 쓴 라벨 표: F4_direct, ALT 유효)."""
    sys.path.insert(0, str(ROOT / "src"))
    from polar.fidelity import MACRO_REGION
    d = pd.read_csv(SRC["base"], usecols=["loc_id", "lat", "lon", "region", "block", "alt_cm", "source_id"], low_memory=False)
    d["macro"] = d.region.map(lambda r: MACRO_REGION.get(r, r))
    a = d[(d.macro == "Alaska") & (d.source_id == "F4_direct") & np.isfinite(d.alt_cm.astype(float))].copy()
    a["loc1km"] = loc1km(a.lat, a.lon)
    g = a.groupby("loc1km").agg(lat=("lat", "mean"), lon=("lon", "mean"), n_labels=("alt_cm", "size")).reset_index()
    return g, dict(n_label_rows=int(len(a)), n_loc_1km=int(len(g)), n_blocks=int(a.block.nunique()))


def load_c():
    """c: WF9 총·격자 사이·격자 안 대비(라벨 전량, 교차검증 λ). 층화 평균은 등록 격자 안 행에만 있다."""
    t = pd.read_csv(SRC["wf3b_tests"], low_memory=False)
    spec = [("total", "WF9-b", "R1(λ cv)-P1[총]|n전량", {"Alaska": "Alaska|r", "Lena": "Lena|r", "Canada": "Canada|r"}),
            ("between", "WF9-b", "R1(λ cv)-P1[격자 사이]|n전량", {"Alaska": "Alaska~b|r", "Lena": "Lena~b|r", "Canada": "Canada~b|r"}),
            ("within", "WF9-a", "R1(λ cv)-P1[격자 안]|n전량", {"Alaska": "Alaska~w|r", "Lena": "Lena~w|r", "Canada": "Canada~w|r",
                                                          "Mean3": "MEAN[Alaska~w|r,Lena~w|r,Canada~w|r]"})]
    rows = []
    for comp, tid, con, tg in spec:
        for reg, target in tg.items():
            q = t[(t.test_id == tid) & (t.contrast == con) & (t.target == target)]
            if tid == "WF9-a":
                q = q[q["item"] == "WF9-a"]
            assert len(q) == 1, (comp, reg, len(q))
            r = q.iloc[0]
            rows.append(dict(panel="c", component=comp, region=reg, test_id=tid, contrast=con, target=target,
                             delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi, delta_blockeq=r.delta_blockeq,
                             delta_blockeq_lo=r.ci_lo_beq, delta_blockeq_hi=r.ci_hi_beq, verdict4=r.verdict4,
                             n_splits=r.n_splits))
    return pd.DataFrame(rows)


def load_d():
    """d: 경로. 3–10 앵커 + 잔차 ML(λ 0.25, 풀 표), 40–160·전량 대상 라벨 교차검증 선정(지역 단순 평균)."""
    p = pd.read_csv(SRC["pool"])
    r1 = p[(p.edition == "E2_LenaCanada_all_n") & (p.method == "R1") & p.n.isin([3, 10])].sort_values("n")
    rg = pd.read_csv(SRC["region_curves"])
    r1r = rg[rg.target.isin(["Lena", "Canada"]) & (rg.method == "R1") & rg.n.isin([3, 10])]
    w = pd.read_csv(SRC["wf_curve"], low_memory=False)
    w = w[(w.exp == "wf4") & (w.method == "W") & w.target.isin(["Lena", "Canada"]) & (w["mode"] == "x") & w.n.isin([40, 160, -1])]
    assert len(w) == 6 and len(r1) == 2 and len(r1r) == 4
    rows = []
    for _, r in r1.iterrows():
        rows.append(dict(panel="d", element="path", method="anchor_residual", region="pool_Lena_Canada", n=int(r.n), delta=r.delta,
                         source="fig2_pool_curves.csv", note="E2_LenaCanada_all_n, R1 (lambda 0.25)"))
    for _, r in r1r.iterrows():
        rows.append(dict(panel="d", element="region_point", method="anchor_residual", region=r.target, n=int(r.n), delta=r.d_p0,
                         source="fig2_region_curves.csv", note="R1 (lambda 0.25), d_p0"))
    for n in (40, 160, -1):
        q = w[w.n == n].set_index("target").d_p0
        rows.append(dict(panel="d", element="path", method="cv_selection", region="mean_Lena_Canada", n=n,
                         delta=float(q.mean()), source="wf_curve.csv", note="wf4, W, mode x, simple mean of two regions (descriptive)"))
        for reg, v in q.items():
            rows.append(dict(panel="d", element="region_point", method="cv_selection", region=reg, n=n, delta=float(v),
                             source="wf_curve.csv", note="wf4, W, mode x, d_p0"))
    D = pd.DataFrame(rows)
    # 풀 점추정 = 지역 점추정의 단순 평균(h42.pool_rows 의 delta 와 같은 정의)인지 대조
    chk = {}
    for n in (3, 10):
        chk[f"R1 n{n} pool - mean(regions)"] = float(r1[r1.n == n].delta.iloc[0] - r1r[r1r.n == n].d_p0.mean())
    return D, chk


# ================================================================ 패널 a
def load_e():
    """e: XC 등록 주 가설의 풀 행(xc_hyp.csv)과 재보정 Stefan 대비 지역 행(독립 지역 7곳 = 주 표 PE1 5곳 + 보조 표 Tibet·Alaska).
    지역 행 가운데 셀 가중 Δ > 0.5 cm 인 행만 그린다(shown). 반환 (풀 행, 지역 행)."""
    h = pd.read_csv(SRC["xc_hyp"])
    nboot = int(json.loads(SRC["xc_meta"].read_text(encoding="utf-8"))["nboot"])
    mm_ = pd.read_csv(SRC["xc_main"])

    def pool_desc(con, n):
        """풀 평균 행(xc_contrasts_main.csv scope MEAN)의 지역 목록을 영문 서술로(Source Data 용)."""
        q = mm_[(mm_.contrast == con) & (mm_.n == n) & (mm_.scope == "MEAN")]
        assert len(q) == 1, (con, n, len(q))
        regs = [E_REGION_FROM_TARGET[t] for t in str(q.target.iloc[0])[5:-1].split(",")]
        return f"pooled mean of {len(regs)} regions: " + ", ".join(S.REGION_NAME[r] for r in regs)
    rows = []
    for head, con, items in E_GROUPS:
        for hid, n in items:
            q = h[h.hypothesis == hid]
            assert len(q) == 1, (hid, len(q))
            r = q.iloc[0]
            assert r.contrast == con and int(r.n) == n, (hid, r.contrast, r.n)
            rows.append(dict(panel="e", element="pooled", group=head, hypothesis=hid, contrast=con, n=n, region=pool_desc(con, n),
                             pool_label=str(r.pool_label),
                             delta=float(r.delta), ci_lo=float(r.ci_lo), ci_hi=float(r.ci_hi), delta_blockeq=float(r.delta_beq),
                             delta_blockeq_lo=float(r.ci_lo_beq), delta_blockeq_hi=float(r.ci_hi_beq), verdict4=str(r.verdict4),
                             p_holm=float(r.p_holm), kind=str(r.kind), design=str(r.design), blind=str(r.blind), nboot=nboot,
                             source="data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_hyp.csv",
                             filter=f"hypothesis == '{hid}' (contrast {con}, n {n})"))
    E = pd.DataFrame(rows)
    m = pd.read_csv(SRC["xc_main"])
    a = pd.read_csv(SRC["xc_aux"])
    reg = []
    for hid, n in E_GROUPS[1][2]:
        rm = m[(m.contrast == "A1-A3") & (m.n == n) & (m.scope == "region")].assign(
            source="data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_contrasts_main.csv", filter_label="main")
        ra = a[(a.contrast == "A1-A3") & (a.n == n) & (a.scope == "region") & a.label.isin(["S-XC8|PE2", "S-XC8|AUX3"])].assign(
            source="data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_contrasts_aux.csv")
        ra["filter_label"] = ra.label
        both = pd.concat([rm, ra], ignore_index=True)
        dup = both[both.duplicated("target", keep=False)]
        for t_, g in dup.groupby("target"):                          # 주 표와 보조 표에 같이 있는 지역(러시아 E)은 값이 같아야 한다
            assert np.allclose(g.delta.astype(float), float(g.delta.iloc[0]), atol=1e-12), t_
        both = both.drop_duplicates("target", keep="first")
        for _, r in both.iterrows():
            rg = E_REGION_FROM_TARGET[str(r.target)]
            reg.append(dict(panel="e", element="region_value", group=E_GROUPS[1][0], hypothesis=hid, contrast="A1-A3", n=n, region=rg,
                            delta=float(r.delta), ci_lo=float(r.ci_lo), ci_hi=float(r.ci_hi), delta_blockeq=float(r.delta_blockeq),
                            delta_blockeq_lo=float(r.ci_lo_beq), delta_blockeq_hi=float(r.ci_hi_beq), verdict4=str(r.verdict4),
                            shown=bool(float(r.delta) > E_WORSE_CM), source=r.source,
                            filter=f"contrast == 'A1-A3', n == {n}, scope == 'region', target == '{r.target}'"
                                   + ("" if r.filter_label == "main" else f", label == '{r.filter_label}'")))
    ER = pd.DataFrame(reg)
    return E, ER


def a_positions():
    out = []
    for i in range(3):
        xm = L["a_left"] + i * (L["a_wm"] + L["a_gap_all"] + L["a_wa"] + L["a_gap"])
        out.append((xm, xm + L["a_wm"] + L["a_gap_all"]))
    return out


def draw_a(fig, A):
    YLIM, YT = (-6.0, 4.0), [-6, -4, -2, 0, 2, 4]
    XLIM = (150.0, 1400.0)
    heads = {"Alaska": "Alaska", "Lena": "Lena Delta", "Canada": "Canada"}
    axes = {}
    for (xm, xa), reg in zip(a_positions(), REGIONS_A):
        axm = S.axes_mm(fig, xm, L["a_top"], L["a_wm"], L["a_h"])
        axa = S.axes_mm(fig, xa, L["a_top"], L["a_wa"], L["a_h"], sharey=axm)
        axes[reg] = (axm, axa)
        axm.set_xscale("log"); axm.set_xlim(*XLIM); axm.set_ylim(*YLIM)
        axa.set_xlim(-1.0, 1.0)
        ticks = [200] if reg == "Canada" else [200, 500, 1000]
        set_fixed_ticks(axm.xaxis, ticks, [str(v) for v in ticks])
        set_fixed_ticks(axa.xaxis, [0.0], ["All"])
        set_fixed_ticks(axm.yaxis, YT, [num_tick(v) for v in YT])
        strip_axis(axm, left=True, bottom=True)
        strip_axis(axa, left=False, bottom=True)
        if reg != REGIONS_A[0]:
            axm.tick_params(axis="y", labelleft=False)          # y 공유: 눈금 라벨은 첫 하위 축에만
        for ax in (axm, axa):
            eq_band(ax); zero_line(ax)
        q = A[A.region == reg]
        for key, off, col in (("anchor_residual", -1.0, COLOR_R), ("direct_ml", 1.0, COLOR_D)):
            for _, r in q[q.method == key].iterrows():
                if r.n > 0:
                    ci_pair(axm, r.n, r, col, orient="v", off_mm=off)
                else:
                    ci_pair(axa, 0.0, r, col, orient="v", off_mm=off)
        # 오차 하한 − 재보정 Stefan RMSE(방법 무관, 한 방법의 행에서 읽는다)
        f = q[q.method == "anchor_residual"].sort_values("n")
        fm = f[f.n > 0]; fa = f[f.n < 0]
        inside = (fm.floor_minus_recal > YLIM[0]).all()
        if inside:
            if len(fm) > 1:
                axm.plot(fm.n, fm.floor_minus_recal, color=COLOR_FLOOR, lw=S.LW["ref"] * 1.5, ls=S.METHOD["error_floor"]["ls"], zorder=1.2,
                         dash_capstyle="butt")
            axa.plot([-0.55, 0.55], [fa.floor_minus_recal.iloc[0]] * 2, color=COLOR_FLOOR, lw=S.LW["ref"] * 1.5,
                     ls=S.METHOD["error_floor"]["ls"], zorder=1.2, dash_capstyle="butt")
        else:                                            # 축 밖(v4: 삼각 표지 대신 축 끝에 값을 숫자로)
            y0 = axm.get_ylim()[0]
            v = float(fm.floor_minus_recal.iloc[0])
            t_ = axm.text(fm.n.iloc[0], y0, num_tick(round(v, 1)), ha="center", va="bottom", fontsize=FS, color=COLOR_FLOOR,
                          transform=mm_shift(axm, 0.0, 0.5), zorder=4)
            t_.set_gid("offaxis")
        # 하위 축 머리(범주 라벨)
        W, H = fig_wh(fig)
        t = text_mm(fig, xm + L["a_wm"] / 2, L["a_top"] - 0.9, heads[reg], ha="center", va="bottom")
        t.set_gid("category")
    # 축 이름: y 는 첫 하위 축에 한 번, x 는 가운데 하위 축 아래 한 번
    axA = axes["Alaska"][0]
    axA.set_ylabel("Error change vs\nrecalibrated Stefan (cm)", linespacing=1.0, labelpad=1.8)
    axes["Lena"][0].set_xlabel("Labels, $n$", labelpad=1.6)
    # 직접 라벨(첫 하위 축에 한 번)과 방향 표지(그림당 1개)
    axmm_text(axA, 8.6, 3.35, S.METHOD["direct_ml"]["short"], ha="left", va="center")
    axmm_text(axA, 1.0, -1.72, S.METHOD["anchor_residual"]["short"], ha="left", va="center")
    fa = A[(A.region == "Alaska") & (A.method == "anchor_residual") & (A.n == 1000)].floor_minus_recal.iloc[0]
    axmm_text(axA, 40.6, fa - 0.62, S.METHOD["error_floor"]["short"], ha="right", va="center")
    for t_ in axA.texts:                                                     # 직접 라벨은 계열 색(v4)
        if t_.get_text() == S.METHOD["direct_ml"]["short"]:
            t_.set_color(COLOR_D)
        elif t_.get_text() == S.METHOD["anchor_residual"]["short"]:
            t_.set_color(COLOR_R)
        elif t_.get_text() == S.METHOD["error_floor"]["short"]:
            t_.set_color(COLOR_FLOOR)
    return axes


# ================================================================ 패널 b
def map_extent(proj, B, aspect, pad=0.03, dx_frac=0.0):
    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    lons, lats = [], []
    for _, r in B.iterrows():
        for dy in (0.0, 0.5):
            for dx in (0.0, 0.25, 0.5):
                lons.append(r.lon_w + dx); lats.append(r.lat_s + dy)
    p = proj.transform_points(pc, np.array(lons), np.array(lats))
    x0, x1, y0, y1 = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    cx, cy = (x0 + x1) / 2 + dx_frac * w, (y0 + y1) / 2
    return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)


def block_poly(proj, lat0, lon0, k=6):
    import cartopy.crs as ccrs
    e = np.linspace(0, 0.5, k)
    lon = np.concatenate([lon0 + e, np.full(k, lon0 + 0.5), lon0 + 0.5 - e, np.full(k, lon0)])
    lat = np.concatenate([np.full(k, lat0), lat0 + e, np.full(k, lat0 + 0.5), lat0 + 0.5 - e])
    return proj.transform_points(ccrs.PlateCarree(), lon, lat)[:, :2]


def frame_crossings(proj, ext):
    """경위선과 지도 틀의 교점(오른쪽 틀: 위도, 위쪽 틀: 경도)."""
    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    x0, x1, y0, y1 = ext
    lat_ticks, lon_ticks = [], []
    for lat in (60, 65, 70):
        lo = np.linspace(-200, -100, 20001)
        p = proj.transform_points(pc, lo, np.full_like(lo, lat))
        s = np.sign(p[:, 0] - x1)
        i = np.where(np.diff(s) != 0)[0]
        for j in i:
            y = p[j, 1]
            if y0 < y < y1:
                lat_ticks.append((y, f"{lat}°N"))
    for lon in (-170, -150, -130):
        la = np.linspace(50, 89.9, 20001)
        p = proj.transform_points(pc, np.full_like(la, lon), la)
        s = np.sign(p[:, 1] - y1)
        i = np.where(np.diff(s) != 0)[0]
        for j in i:
            x = p[j, 0]
            if x0 < x < x1:
                lon_ticks.append((x, f"{abs(lon)}°W"))
    return lat_ticks, lon_ticks


def draw_b(fig, B, obs):
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from cmcrameri import cm as cmc
    pc = ccrs.PlateCarree()
    proj = ccrs.Stereographic(central_latitude=90, central_longitude=LON0, true_scale_latitude=70)
    x, y, w, h = L["b_map"]
    ax = S.axes_mm(fig, x, y, w, h, projection=proj)
    ext = map_extent(proj, B, w / h, pad=0.03, dx_frac=0.04)
    ax.set_extent(ext, crs=proj)
    ax.spines["geo"].set_linewidth(1.0); ax.spines["geo"].set_edgecolor(INK)
    land = list(shpreader.Reader(shpreader.natural_earth(resolution="50m", category="physical", name="land")).geometries())
    ax.add_geometries(land, crs=pc, facecolor=S.BASEMAP["land"], edgecolor=S.BASEMAP["coast"], linewidth=S.LW["coast"], zorder=0.4)
    for lat in (60, 65, 70):
        lo = np.linspace(-200, -100, 600)
        ax.plot(lo, np.full_like(lo, lat), color=S.BASEMAP["graticule"], lw=S.LW["grid_map"], transform=pc, zorder=0.6)
    for lon in (-170, -160, -150, -140, -130):
        la = np.linspace(50, 89, 400)
        ax.plot(np.full_like(la, lon), la, color=S.BASEMAP["graticule"], lw=S.LW["grid_map"], transform=pc, zorder=0.6)
    # 블록 채움: broc, 0 중심, ±vmax(이 지도 |값| 의 99 백분위를 5 cm 단위로 올림)
    show = B[~B.masked]
    vmax = float(np.ceil(np.nanpercentile(np.abs(show.delta), 99) / 5.0) * 5.0)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    cmap = cmc.broc
    polys = [block_poly(proj, r.lat_s, r.lon_w) for _, r in show.iterrows()]
    pcoll = PolyCollection(polys, facecolors=cmap(norm(show.delta.values)), edgecolors="none", linewidths=0, zorder=1.0)
    ax.add_collection(pcoll)
    msk = B[B.masked]
    if len(msk):
        mp = [block_poly(proj, r.lat_s, r.lon_w) for _, r in msk.iterrows()]
        mcoll = PolyCollection(mp, facecolors=S.BASEMAP["nodata"], edgecolors=S.BASEMAP["hatch"], linewidths=0, zorder=1.0)
        mcoll.set_hatch("//")
        ax.add_collection(mcoll)
    # 관측 위치(1 km 라벨 위치) 검정 점 1.5 pt
    ax.plot(obs.lon.values, obs.lat.values, ls="none", marker="o", ms=1.5, mfc=INK, mec="none", transform=pc, zorder=2.0)
    # 축척 막대(200 km): 왼쪽 위 축해(자료 없음), 틀 위 경도 라벨과 겹치지 않게 틀에서 5 mm 아래. 약 69° N(기준 위도 70° N 과 축척 계수 차 0.3%)
    x0, x1, y0, y1 = ext
    sx, sy = x0 + 0.04 * (x1 - x0), y1 - 5.0 * (x1 - x0) / w
    ax.plot([sx, sx + 200e3], [sy, sy], color=INK, lw=S.LW["scale_bar"], solid_capstyle="butt", transform=ax.transData, zorder=3)
    t = ax.text(sx + 100e3, sy + 0.012 * (y1 - y0), "200 km", ha="center", va="bottom", fontsize=FS, transform=ax.transData)
    t.set_gid("scale")
    # 위치 삽도(범북극 윤곽과 대상 사각형): 오른쪽 위(캐나다 쪽, 알래스카 블록 없음). 오른쪽 아래는 블록을 가린다
    iw = 17.0
    axi = S.axes_mm(fig, x + w - iw - 0.6, y + 0.6, iw, iw, projection=proj)
    axi.set_extent([-3.6e6, 3.6e6, -3.6e6, 3.6e6], crs=proj)
    land110 = list(shpreader.Reader(shpreader.natural_earth(resolution="110m", category="physical", name="land")).geometries())
    axi.add_geometries(land110, crs=pc, facecolor=S.BASEMAP["land"], edgecolor=S.BASEMAP["coast"], linewidth=S.LW["coast"], zorder=0.4)
    axi.set_facecolor("white")
    axi.spines["geo"].set_linewidth(1.0); axi.spines["geo"].set_edgecolor(INK2)
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=axi.transData, facecolor="none", edgecolor=INK, lw=1.0, zorder=3)
    rect.set_gid("allowed_frame")
    axi.add_patch(rect)
    # 경위선 라벨: 오른쪽 틀(위도), 위쪽 틀(경도). 겹친 일반 축의 눈금으로 둔다(그림 안 수치 = 눈금).
    lat_t, lon_t = frame_crossings(proj, ext)
    axg = S.axes_mm(fig, x, y, w, h, frameon=False)
    axg.set_xlim(x0, x1); axg.set_ylim(y0, y1); axg.set_navigate(False)
    axg.patch.set_visible(False)
    axg.yaxis.tick_right(); axg.xaxis.tick_top()
    set_fixed_ticks(axg.yaxis, [v for v, _ in lat_t], [s for _, s in lat_t])
    set_fixed_ticks(axg.xaxis, [v for v, _ in lon_t], [s for _, s in lon_t])
    axg.tick_params(axis="both", length=S.TICK_LEN_PT, width=S.LW["tick"], direction="out")
    # 컬러바(가로, 지도 아래)
    from matplotlib.cm import ScalarMappable
    cx, cy, cw, ch = L["b_cbar"]
    cax = S.axes_mm(fig, cx, cy, cw, ch)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    ct = [-vmax, 0.0, vmax]
    cb.set_ticks(ct)
    cb.set_ticklabels([num_tick(v) for v in ct])
    cb.outline.set_linewidth(1.0)
    cb.dividers.set_linewidth(1.0)                         # 빈 구분선 모음의 기본 0.5 pt 가 F-08 점검에 걸리지 않게
    cax.tick_params(width=S.LW["tick"], length=S.TICK_LEN_PT)
    cb.set_label("Error change vs recalibrated Stefan (cm)", labelpad=1.6)
    # 블록 경도 폭(최종 크기 mm): 블록마다 자기 위도에서
    m_per_mm = (x1 - x0) / w
    widths = []
    for _, r in B.iterrows():
        q = proj.transform_points(pc, np.array([r.lon_w, r.lon_w + 0.5]), np.array([r.lat_s + 0.25] * 2))
        widths.append(np.hypot(*(q[1, :2] - q[0, :2])) / m_per_mm)
    widths = np.array(widths)
    latc = float(B.lat_s.mean() + 0.25)
    qc = proj.transform_points(pc, np.array([LON0 - 0.25, LON0 + 0.25]), np.array([latc, latc]))
    info = dict(vmax=vmax, extent_km=((x1 - x0) / 1e3, (y1 - y0) / 1e3), km_per_mm=m_per_mm / 1e3,
                block_width_mm_center_lat=float(np.hypot(*(qc[1, :2] - qc[0, :2])) / m_per_mm), center_lat=latc,
                block_width_mm_min=float(widths.min()), block_width_mm_max=float(widths.max()),
                n_blocks_lt_1mm=int((widths < 1.0).sum()), lat_ticks=lat_t, lon_ticks=lon_t,
                scale_bar_lonlat=tuple(float(v) for v in pc.transform_point(sx, sy, proj)), n_masked=int(B.masked.sum()),
                n_shown=int((~B.masked).sum()))
    return ax, info


# ================================================================ 패널 c
def draw_c(fig, C):
    rows = [("Alaska", "Alaska", 0.0), ("Lena", "Lena Delta", 1.0), ("Canada", "Canada", 2.0), ("Mean3", "Three-region mean", 3.35)]
    cols = [("total", "Total"), ("between", "Between"), ("within", "Within")]
    XLIM, XT = (-2.0, 1.5), [-2, -1, 0, 1]
    axes = []
    for j, (comp, head) in enumerate(cols):
        x = L["c_x0"] + j * (L["c_wc"] + L["c_gap"])
        ax = S.axes_mm(fig, x, L["c_top"], L["c_wc"], L["c_h"])
        ax.set_xlim(*XLIM); ax.set_ylim(3.95, -0.6)
        set_fixed_ticks(ax.xaxis, XT, [num_tick(v) for v in XT])
        ax.yaxis.set_major_locator(NullLocator())
        strip_axis(ax, left=False, bottom=True)
        eq_band(ax, axis="x"); zero_line(ax, axis="x")
        for key, _, yp in rows:
            q = C[(C.component == comp) & (C.region == key)]
            if len(q):
                ci_pair(ax, yp, q.iloc[0], COLOR_R, orient="h")        # v4: 대비 = 앵커 + 잔차 − 재보정 Stefan, 제안 방법 색
        t = text_mm(fig, x + L["c_wc"] / 2, L["c_top"] - 1.0, head, ha="center", va="bottom", linespacing=1.0)
        t.set_gid("category")
        axes.append(ax)
    # 묶음 머리: "climate cells" 를 Between·Within 두 열 위에 한 번(두 열 머리의 공통 수식어)
    xg = L["c_x0"] + (L["c_wc"] + L["c_gap"]) + L["c_wc"] + L["c_gap"] / 2
    tg = text_mm(fig, xg, L["c_top"] - 1.0 - 3.2, "climate cells", ha="center", va="bottom")
    tg.set_gid("category")
    for key, name, yp in rows:
        ymm = float(data_y_to_fig_mm(axes[0], yp))
        t = text_mm(fig, L["c_label_right"], ymm, name, ha="right", va="center")
        t.set_gid("category")
    axes[1].set_xlabel("Error change vs recalibrated Stefan (cm)", labelpad=1.6)
    return axes


def key_line(fig, x_mm, y_mm, gap_mm=2.4, pad_mm=0.9, half_h=1.05):
    """열쇠 한 줄(그림 명세 1.2): 셀 가중 막대, 블록 등가중 막대, ±0.5 cm 띠 견본. 반환: 오른쪽 끝 x(mm)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig_wh(fig)
    x = x_mm
    for i, lab in enumerate(("Cell-weighted", "Block-equal", "±0.5 cm")):
        if i < 2:
            xs = x + 0.5
            fig.add_artist(Line2D([xs / W] * 2, [1 - (y_mm - half_h) / H, 1 - (y_mm + half_h) / H], transform=fig.transFigure,
                                  color=INK, lw=S.LW["ci_forest_cell"] if i == 0 else S.LW["ci_forest_block"],
                                  solid_capstyle="round" if i == 0 else "butt"))
            x = xs + 0.5 + pad_mm
        else:
            r = Rectangle((x / W, 1 - (y_mm + half_h) / H), 1.8 / W, 2 * half_h / H, transform=fig.transFigure,
                          facecolor=S.EQUIV_BAND, edgecolor="none", lw=0)
            r.set_gid("allowed_frame")
            fig.add_artist(r)
            x = x + 1.8 + pad_mm
        t = text_mm(fig, x, y_mm, lab, ha="left", va="center")
        if i == 2:
            t.set_gid("sizekey")                                # 열쇠 값(그림 명세 1.2 의 견본). 관찰 수치가 아니다
        x = x + t.get_window_extent(rend).width / fig.dpi / MM + gap_mm
    return x - gap_mm


# ================================================================ 패널 d
D_YLIM, D_YT = (-0.6, 1.7), [-0.5, 0.0, 0.5, 1.0, 1.5]
D_XLIM = (2.5, 1300.0)
D_BOUND = (20.0, float(np.sqrt(160 * 320)))             # 3–10 | 40–160 | 320–1000 경계(기하 중앙)


def draw_d(fig, D):
    top, h = L["d_top"], L["d_h"]
    ax0 = S.axes_mm(fig, L["d_n0"][0], top, L["d_n0"][1], h)
    axm = S.axes_mm(fig, L["d_main"][0], top, L["d_main"][1], h, sharey=ax0)
    axa = S.axes_mm(fig, L["d_all"][0], top, L["d_all"][1], h, sharey=ax0)
    ax0.set_xlim(-1, 1); axa.set_xlim(-1, 1)
    axm.set_xscale("log"); axm.set_xlim(*D_XLIM)
    ax0.set_ylim(*D_YLIM)
    set_fixed_ticks(ax0.yaxis, D_YT, [num_tick(v) if v != 0 else "0" for v in D_YT])
    ax0.yaxis.set_major_formatter(FixedFormatter(["−0.5", "0", "0.5", "1.0", "1.5"]))
    strip_axis(ax0, left=True, bottom=True)
    strip_axis(axm, left=False, bottom=True)
    strip_axis(axa, left=False, bottom=True)
    # 구간 이름 = 눈금 라벨(눈금 길이 0), 구간 경계 = x 축 아래 긴 눈금
    set_fixed_ticks(ax0.xaxis, [0.0], ["0"])
    cen = [float(np.sqrt(D_XLIM[0] * D_BOUND[0])), float(np.sqrt(D_BOUND[0] * D_BOUND[1])), float(np.sqrt(D_BOUND[1] * D_XLIM[1]))]
    set_fixed_ticks(axm.xaxis, cen, ["3–10", "40–160", "320–1000"])
    set_fixed_ticks(axa.xaxis, [0.0], ["All"])
    for ax in (ax0, axm, axa):
        ax.tick_params(axis="x", length=0, pad=2.2)
        zero_line(ax)
    trb = mtrans.blended_transform_factory(axm.transData, axm.transAxes)
    long_tick = 3.6 / L["d_h"]
    for b in D_BOUND:
        axm.add_line(Line2D([b, b], [0, -long_tick], transform=trb, color=INK, lw=S.LW["tick"], clip_on=False))
    # 경로: 구간마다 권고 방법의 선분(경계에서 잇지 않는다)
    r1 = D[(D.element == "path") & (D.method == "anchor_residual")].sort_values("n")
    w = D[(D.element == "path") & (D.method == "cv_selection")]
    wm = w[w.n > 0].sort_values("n"); wa = w[w.n < 0]
    axm.plot(r1.n, r1.delta, color=COLOR_R, lw=S.LW["ci_forest_cell"], solid_capstyle="butt", zorder=3)
    axm.plot(r1.n, r1.delta, ls="none", marker="o", ms=S.MS["main"], mfc=COLOR_R, mec="none", zorder=3.1)
    CW = S.METHOD["target_cv_selection"]["color"]
    axm.plot(wm.n, wm.delta, color=CW, lw=S.LW["ci_forest_cell"], solid_capstyle="butt", zorder=3)
    axm.plot(wm.n, wm.delta, ls="none", marker="o", ms=S.MS["main"], mfc=CW, mec="none", zorder=3.1)
    axa.plot([0.0], [wa.delta.iloc[0]], ls="none", marker="o", ms=S.MS["main"], mfc=CW, mec="none", zorder=3.1)
    # 지역 값(2.5 pt, alpha 0.5, 지역 모양, 해당 계열 색). 풀 점과 겹치지 않게 ±0.9 mm 어긋남
    off = {"Lena": -0.9, "Canada": 0.9}
    for _, r in D[D.element == "region_point"].iterrows():
        col = COLOR_R if r.method == "anchor_residual" else CW
        ax = axm if r.n > 0 else axa
        xv = r.n if r.n > 0 else 0.0
        ax.plot([xv], [r.delta], ls="none", marker=S.REGION_MARKER[r.region], ms=S.MS["region_point"], mfc=col, mec="none",
                alpha=0.5, transform=mm_shift(ax, off[r.region], 0.0), zorder=2.8)
    # 방법 직접 라벨(1–3 단어, 방법마다 한 번)
    t0 = axmm_text(ax0, 0.0, -0.21, S.METHOD["source_stefan"]["short"], ha="left", va="center", color=S.METHOD["source_stefan"]["color"])
    t0.set_clip_on(False)
    for ax in (ax0, axm, axa):                                               # 0선 = 원천 계수 Stefan(v4: 그 방법 색)
        for ln in ax.lines:
            if ln.get_xydata().shape[0] == 2 and np.allclose(ln.get_ydata(), 0.0) and ln.get_linewidth() == S.ZERO_LINE["lw"]:
                ln.set_color(S.METHOD["source_stefan"]["color"]); ln.set_linewidth(S.LW["main"])
    axmm_text(axm, 1.2, 1.06, S.METHOD["anchor_residual"]["short"], ha="left", va="center", color=COLOR_R)
    xw = (np.log10(40) - np.log10(D_XLIM[0])) / (np.log10(D_XLIM[1]) - np.log10(D_XLIM[0])) * L["d_main"][1]
    axmm_text(axm, xw - 1.0, 1.06, S.METHOD["target_cv_selection"]["short"], ha="left", va="center", color=CW)
    # 지역 모양 열쇠(한 줄, 오른쪽 위 빈 곳)
    W, H = fig_wh(fig)
    ky = float(data_y_to_fig_mm(axm, 1.52))
    kx = 80.0
    for reg, name in (("Lena", "Lena Delta"), ("Canada", "Canada")):
        fig.add_artist(Line2D([(kx + 0.9) / W], [1 - ky / H], transform=fig.transFigure, ls="none",
                              marker=S.REGION_MARKER[reg], ms=S.MS["region_point"], mfc=INK2, mec="none", alpha=0.6))
        tt = text_mm(fig, kx + 2.0, ky, name, ha="left", va="center")
        fig.canvas.draw()
        kx = kx + 2.0 + tt.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / MM + 2.6
    ax0.set_ylabel("Error change vs\nsource Stefan (cm)", linespacing=1.0, labelpad=1.8)
    axm.set_xlabel("Labels, $n$", labelpad=0.8)
    return (ax0, axm, axa)


def n_to_fig_mm(axes_d, n):
    """경로 축의 라벨 수 n(0, 양수, 'All' = −1)을 그림 x mm 로."""
    ax0, axm, axa = axes_d
    if n == 0:
        return L["d_n0"][0] + L["d_n0"][1] / 2
    if n < 0:
        return L["d_all"][0] + L["d_all"][1] / 2
    f = (np.log10(n) - np.log10(D_XLIM[0])) / (np.log10(D_XLIM[1]) - np.log10(D_XLIM[0]))
    return L["d_main"][0] + f * L["d_main"][1]


def draw_d_bands(fig, axes_d):
    """관측(배치)·진단 띠: 경로 축과 다른 축 객체(그림 명세 9.3 d, D-13). 같은 내용이 이어지면 선 하나, 라벨 하나.
    진단 두 항목은 구간이 다르고(0 / 3–All) 짧은 구간(6 mm)이 라벨(19 mm)보다 좁으므로 줄을 나눠 라벨이 다른 선 위에 걸치지 않게 한다."""
    bx, by, bw, bh = L["d_band"]
    axb = S.axes_mm(fig, bx, by, bw, bh, frameon=False)
    axb.set_xlim(bx, bx + bw); axb.set_ylim(by + bh, by)
    axb.set_xticks([]); axb.set_yticks([])
    axb.patch.set_visible(False)
    y_pl, y_d1, y_d2 = by + 3.6, by + 8.4, by + 13.2
    xl = lambda n: n_to_fig_mm(axes_d, n)                                   # noqa: E731
    left0, right0 = L["d_n0"][0], L["d_n0"][0] + L["d_n0"][1]
    x160_end = xl(D_BOUND[1])                                               # 40–160 구간의 오른쪽 경계
    x3_start = L["d_main"][0]
    x_all_end = L["d_all"][0] + L["d_all"][1]
    lw = S.LW["ci_forest_cell"]
    CB = S.METHOD["workflow"]["color"]                                       # v4: 워크플로 단계 = 워크플로 색
    # 배치: 0–160 한 선, 라벨은 선 가운데 위
    axb.plot([left0, x160_end], [y_pl, y_pl], color=CB, lw=lw, solid_capstyle="butt")
    axb.text((left0 + x160_end) / 2, y_pl - 1.0, "Spread placement", ha="center", va="bottom", fontsize=FS)
    # 진단 1: 0(외삽 비율) 짧은 선, 라벨은 선 왼쪽 끝에서 시작. 진단 2: 3–All(라벨 10개 편향) 긴 선, 라벨은 가운데 위
    axb.plot([left0, right0], [y_d1, y_d1], color=CB, lw=lw, solid_capstyle="butt")
    axb.text(left0, y_d1 - 1.0, "Extrapolation share", ha="left", va="bottom", fontsize=FS)
    axb.plot([x3_start, x_all_end], [y_d2, y_d2], color=CB, lw=lw, solid_capstyle="butt")
    axb.text((x3_start + x_all_end) / 2, y_d2 - 1.0, "Ten-label bias", ha="center", va="bottom", fontsize=FS)
    # 띠 머리(#4d4d4d): 배치 줄, 진단 두 줄의 가운데
    for yy, s_ in ((y_pl, "Placement"), ((y_d1 + y_d2) / 2, "Diagnosis")):
        axb.text(L["d_head_right"], yy, s_, ha="right", va="center", fontsize=FS, color=INK2)
    return axb


# ================================================================ 패널 e(XC 포레스트)
def e_rows():
    """묶음 머리와 행의 y(자료 좌표, 아래로 증가)와 묶음별 행 범위."""
    pos, heads, spans = {}, [], []
    yv = 0.0
    for gi, (head, _con, items) in enumerate(E_GROUPS):
        if gi:
            yv += 0.35
        heads.append((yv, head))
        yv += 1.0
        g0 = yv
        for hid, _n in items:
            pos[hid] = yv
            yv += 1.0
        spans.append((g0 - 0.5, yv - 0.5))
    return pos, heads, spans, yv


def draw_e(fig, E, ER, fs=None, key_xy=None, ms_region=None, name_right=None, head_x=None, key_gap_mm=None, key_sep_mm=2.6):
    """XC 포레스트. 0선과 ±0.5 cm 띠는 묶음 행에만 그린다(묶음 머리 줄을 비워 머리 글자가 축 위를 지나가게) [판단].
    지역 기호: 재보정 Stefan 대비 0.5 cm 넘게 나빴던 지역의 셀 가중 Δ(지역 모양, 2.5 pt, alpha 0.5; 패널 d 의 지역 값과 같은 부호화)."""
    fs = fs or FS
    x, y, w, h = L["e_ax"]
    ax = S.axes_mm(fig, x, y, w, h)
    pos, heads, spans, yv = e_rows()
    ax.set_xlim(*E_XLIM)
    ax.set_ylim(yv - 0.5, -0.6)
    strip_axis(ax, left=False, bottom=True)
    ax.yaxis.set_major_locator(NullLocator())
    set_fixed_ticks(ax.xaxis, E_XT, [num_tick(v) for v in E_XT])
    hb = S.EQUIV_HALF_WIDTH_CM
    for y0, y1 in spans:
        ax.fill_betweenx([y0, y1], -hb, hb, facecolor=S.EQUIV_BAND, edgecolor="none", lw=0, zorder=0.5)
        ax.plot([0.0, 0.0], [y0, y1], color=S.ZERO_LINE["color"], lw=S.ZERO_LINE["lw"], solid_capstyle="butt", zorder=1.5)
    cwf = S.METHOD["workflow"]["color"]
    for _, r in E.iterrows():
        ci_pair(ax, pos[r.hypothesis], r, cwf, orient="h")
    msr = ms_region or S.MS["region_point"]
    for _, r in ER[ER.shown].iterrows():
        ax.plot([r.delta], [pos[r.hypothesis]], ls="none", marker=S.REGION_MARKER[r.region], ms=msr, mfc=cwf, mec="none",
                alpha=0.5, zorder=2.8, gid=f"e_region|{r.hypothesis}|{r.region}")
    W, H = fig_wh(fig)
    nr = name_right if name_right is not None else L["e_name_right"]
    hx = head_x if head_x is not None else L["e_head_x"]
    for hid, yy in pos.items():
        n = int(E.set_index("hypothesis").loc[hid, "n"])
        t = text_mm(fig, nr, float(data_y_to_fig_mm(ax, yy)), f"{n} labels", ha="right", va="center", fontsize=fs)
        t.set_gid("rowname_n")                                   # 범주 이름(명세 9.3 e 의 행 이름). audit_v3 수치 검사에서 뺀다
    for yy, head in heads:
        t = text_mm(fig, hx, float(data_y_to_fig_mm(ax, yy)), head, ha="left", va="center", fontsize=fs)
        t.set_gid("category")
    ax.set_xlabel("Workflow error change (cm)", labelpad=1.6, fontsize=fs)
    # 지역 모양 열쇠(그린 지역만, 한 줄)
    kx, ky = key_xy or L["e_key"]
    shown = [rg for rg in ("Alaska", "Lena", "Russia_E", "Russia_W", "Canada", "Russia_C_LGD") if rg in set(ER[ER.shown].region)]
    rend = None
    r_mm = msr / 72.0 * 25.4 / 2
    cx_off, tx_off = (0.9, 2.0) if key_gap_mm is None else (r_mm, 2 * r_mm + key_gap_mm)   # 기본값 = 패널 d 열쇠와 같은 간격
    for rg in shown:
        fig.add_artist(Line2D([(kx + cx_off) / W], [1 - ky / H], transform=fig.transFigure, ls="none", marker=S.REGION_MARKER[rg], ms=msr,
                              mfc=cwf, mec="none", alpha=0.5))
        tt = text_mm(fig, kx + tx_off, ky, S.REGION_NAME[rg], ha="left", va="center", fontsize=fs)
        fig.canvas.draw()
        rend = rend or fig.canvas.get_renderer()
        kx = kx + tx_off + tt.get_window_extent(rend).width / fig.dpi / MM + key_sep_mm
    return ax


# ================================================================ 조립
def build(medium="paper"):
    if medium != "paper":
        raise NotImplementedError("slide medium: 덱 작업에서 같은 자료·색·범위로 더한다(지침 0.4 3)")
    use_rc()
    A, floor = load_a()
    B, bchk = load_b()
    obs, ochk = load_obs()
    C = load_c()
    D, dchk = load_d()
    E, ER = load_e()
    fig = S.fig_mm(FIG_W, FIG_H)
    axes_a = draw_a(fig, A)
    axb, binfo = draw_b(fig, B, obs)
    axes_c = draw_c(fig, C)
    axes_d = draw_d(fig, D)
    draw_d_bands(fig, axes_d)
    draw_e(fig, E, ER)
    key_right = key_line(fig, *L["c_key"])
    for k in ("a", "b", "c", "d", "e"):
        S.panel_letter(fig, *L[f"{k}_letter"], k)
    meta = dict(floor=floor, bchk=bchk, ochk=ochk, dchk=dchk, binfo=binfo, key_right=key_right)
    return fig, dict(A=A, B=B, obs=obs, C=C, D=D, E=E, ER=ER), meta


# ================================================================ 기록
def write_source_data(data, path):
    A, B, obs, C, D, E, ER = (data[k] for k in ("A", "B", "obs", "C", "D", "E", "ER"))
    parts = []
    a = A.copy()
    a["n_label"] = np.where(a.n < 0, "all", a.n.astype(str))
    a["source"] = "results/rescale_wf2/data/processed/wf/wf2b_tests.csv; data/processed/lgx/lgx_floor.csv (scope eval)"
    a["filter"] = "test_id == WF6-a, item == WF6-a, contrast as given"
    parts.append(a.rename(columns={"region": "region"}))
    b = B.copy()
    b["source"] = "results/rescale_wf3/data/processed/wf/shards/wf9__cpu__Alaska__r__s*_blocksse.npz (unit Alaska|r)"
    b["filter"] = "keys [R1, catboost_lo, 1, cell, -1, 0, seed 0/1, -1.0] (seed mean) and [P1, none, 1, cell, -1, 0, -1, 0.0]; SSE and cells summed over 25 splits"
    b["region"] = "Alaska"; b["element"] = "block"
    parts.append(b)
    o = obs.copy(); o["panel"] = "b"; o["element"] = "label_location_1km"; o["region"] = "Alaska"
    o["source"] = "data/processed/fidelity_base_v3.csv (macro Alaska, source_id F4_direct, finite alt_cm)"
    parts.append(o)
    c = C.copy(); c["source"] = "results/rescale_wf3/data/processed/wf/wf3b_tests.csv"
    parts.append(c)
    parts.append(D.copy())
    e = E.drop(columns=["design", "blind", "kind", "pool_label"]).copy()      # 국문 설계·맹검 서술은 Fig7_values.txt [4b] 에만
    e["n_label"] = e.n.astype(str)
    parts.append(e)
    er = ER.copy()
    er["n_label"] = er.n.astype(str)
    er["note"] = np.where(er.shown, "drawn: worse than recalibrated Stefan by more than 0.5 cm (cell-weighted)", "not drawn")
    parts.append(er)
    out = pd.concat(parts, ignore_index=True, sort=False)
    lead = ["panel", "element", "region", "method", "component", "n", "n_label", "delta", "ci_lo", "ci_hi", "delta_blockeq",
            "delta_blockeq_lo", "delta_blockeq_hi"]
    cols = [c for c in lead if c in out.columns] + [c for c in out.columns if c not in lead]
    out = out[cols]
    out.to_csv(path, index=False, float_format="%.6g")
    return out


EXPECT = [   # (설명, 계산값 함수 키, 기대값, 출처) 기대값은 paper/claims README 의 소수 둘째 자리 값
    ("a Alaska R1 n200 delta / cell CI / block CI", ("a", "Alaska", "anchor_residual", 200), (-0.18, -0.34, 0.01, -0.25, -0.42, -0.07), "C4 README 2.1 E1.R1.n200"),
    ("a Alaska R1 n500", ("a", "Alaska", "anchor_residual", 500), (-0.39, -0.54, -0.21, -0.39, -0.57, -0.21), "C4 README 2.1 E1.R1.n500"),
    ("a Alaska R1 n1000", ("a", "Alaska", "anchor_residual", 1000), (-0.54, -0.69, -0.32, -0.60, -0.80, -0.40), "C4 README 2.1 E1.R1.n1000"),
    ("a Alaska R1 all", ("a", "Alaska", "anchor_residual", -1), (-0.52, -0.70, -0.31, -0.68, -0.91, -0.44), "C4 README 2.1 E1.R1.n전량"),
    ("a Alaska D0 n200", ("a", "Alaska", "direct_ml", 200), (3.51, 1.71, 3.56, -0.21, -0.71, 0.30), "C4 README 2.1 E1.D0.n200"),
    ("a Alaska D0 n500", ("a", "Alaska", "direct_ml", 500), (3.16, 1.46, 3.20, -0.56, -1.08, -0.04), "C4 README 2.1 E1.D0.n500"),
    ("a Alaska D0 n1000", ("a", "Alaska", "direct_ml", 1000), (2.76, 1.16, 2.78, -0.65, -1.15, -0.14), "C4 README 2.1 E1.D0.n1000"),
    ("a Alaska D0 all", ("a", "Alaska", "direct_ml", -1), (1.27, 0.30, 1.37, -1.06, -1.56, -0.58), "C4 README 2.1 E1.D0.n전량"),
    ("a Lena R1 n200", ("a", "Lena", "anchor_residual", 200), (0.22, -0.79, 0.64, -0.20, -0.75, 0.34), "C4 README 2.2 E2.Lena.R1.n200"),
    ("a Lena R1 n500", ("a", "Lena", "anchor_residual", 500), (0.34, -0.76, 0.78, -0.36, -1.16, 0.38), "C4 README 2.2 E2.Lena.R1.n500"),
    ("a Lena R1 n1000", ("a", "Lena", "anchor_residual", 1000), (0.61, -0.57, 1.38, -0.08, -1.04, 0.81), "C4 README 2.2 E2.Lena.R1.n1000"),
    ("a Lena R1 all", ("a", "Lena", "anchor_residual", -1), (0.47, -0.56, 1.28, -0.14, -1.07, 0.72), "C4 README 2.2 E2.Lena.R1.n전량"),
    ("a Lena D0 n1000", ("a", "Lena", "direct_ml", 1000), (-0.78, -1.94, 0.17, -1.08, -2.00, -0.21), "C4 README 2.2 E2.Lena.D0.n1000"),
    ("a Canada R1 n200", ("a", "Canada", "anchor_residual", 200), (-0.40, -1.05, 0.19, -0.84, -1.36, -0.28), "C4 README 2.2 E2.Canada.R1.n200"),
    ("a Canada R1 all", ("a", "Canada", "anchor_residual", -1), (0.45, -0.73, 1.24, -0.23, -1.09, 0.65), "C4 README 2.2 E2.Canada.R1.n전량"),
    ("a Canada D0 n200", ("a", "Canada", "direct_ml", 200), (-1.50, -1.99, -0.24, -0.03, -1.03, 1.01), "C4 README 2.2 E2.Canada.D0.n200"),
    ("a Canada D0 all", ("a", "Canada", "direct_ml", -1), (-1.50, -2.19, -0.12, -0.30, -1.44, 0.86), "C4 README 2.2 E2.Canada.D0.n전량"),
    ("c Alaska total", ("c", "Alaska", "total"), (-0.52, -0.70, -0.31, -0.68, -0.91, -0.44), "C8 README 2.1 A1"),
    ("c Alaska between", ("c", "Alaska", "between"), (-1.03, -1.30, -0.59, -0.89, -1.16, -0.61), "C8 README 2.1 A2"),
    ("c Alaska within", ("c", "Alaska", "within"), (-0.01, -0.03, 0.04, 0.04, 0.01, 0.06), "C8 README 2.1 A3"),
    ("c three-region mean within", ("c", "Mean3", "within"), (0.09, 0.02, 0.40, 0.37, 0.25, 0.50), "C8 README 2.4 D3"),
    ("c Lena within", ("c", "Lena", "within"), (0.05, -0.18, 0.92, 0.87, 0.51, 1.24), "C8 README 2.4 D4"),
    ("c Canada within", ("c", "Canada", "within"), (0.23, 0.16, 0.34, 0.21, 0.12, 0.31), "C8 README 2.4 D5"),
    ("c Lena total (= C4 E2.Lena.R1.n전량)", ("c", "Lena", "total"), (0.47, -0.56, 1.28, -0.14, -1.07, 0.72), "C4 README 2.2"),
    ("c Canada total (= C4 E2.Canada.R1.n전량)", ("c", "Canada", "total"), (0.45, -0.73, 1.24, -0.23, -1.09, 0.65), "C4 README 2.2"),
]


def get6(data, key):
    if key[0] == "a":
        _, reg, meth, n = key
        q = data["A"]; r = q[(q.region == reg) & (q.method == meth) & (q.n == n)].iloc[0]
    else:
        _, reg, comp = key
        q = data["C"]; r = q[(q.region == reg) & (q.component == comp)].iloc[0]
    return tuple(float(r[k]) for k in ("delta", "ci_lo", "ci_hi", "delta_blockeq", "delta_blockeq_lo", "delta_blockeq_hi"))


def write_values(path, data, meta, audit, pdfa, fonts_txt, png_info):
    A, B, obs, C, D = (data[k] for k in ("A", "B", "obs", "C", "D"))
    L_ = []
    P = L_.append
    P("Fig 7 v3 값 대조와 점검 기록")
    P(f"생성: scripts/4_visualization/paper_v3/fig7.py, 그림 명세 FIGURE_SPEC_v3.md 9절, 2026-10-04")
    P("규칙: 기대값은 paper/claims/*/README.md 의 소수 둘째 자리 값이다. 계산값을 둘째 자리로 반올림해 같으면 OK.")
    P("")
    P("[1] 패널 a, c: 원천 값과 claims README 대조(delta, 셀 가중 CI, 블록 등가중 delta, 블록 등가중 CI)")
    nbad = 0
    for desc, key, exp, src in EXPECT:
        got = get6(data, key)
        # 블록 등가중 점추정은 README 에 '블록 등가중 delta [CI]' 로 적혀 있다. 대조 순서: delta, lo, hi, delta_beq, lo_beq, hi_beq
        got_cmp = (got[0], got[1], got[2], got[3], got[4], got[5])
        ok = all(abs(round(g, 2) - e) < 0.0051 for g, e in zip(got_cmp, exp))
        nbad += (not ok)
        P(f"  {'OK ' if ok else 'BAD'} {desc}: 계산 " + ", ".join(f"{g:+.4f}" for g in got_cmp) + f" | 기대 {exp} | {src}")
    P("")
    P("[1b] 그림의 모든 수치와 원천 경로·행 필터(패널 a, c: delta [ci_lo, ci_hi] / delta_blockeq [lo, hi] cm)")
    srcA = "results/rescale_wf2/data/processed/wf/wf2b_tests.csv"
    for _, r in A.sort_values(["region", "method", "n"]).iterrows():
        P(f"  a {r.region:7s} {r.method:16s} n {('all' if r.n < 0 else int(r.n)):>4}: {r.delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] / "
          f"{r.delta_blockeq:+.4f} [{r.delta_blockeq_lo:+.4f}, {r.delta_blockeq_hi:+.4f}]; floor − recal {r.floor_minus_recal:+.4f} "
          f"| {srcA} | test_id == 'WF6-a', item == 'WF6-a', target == '{r.region}|r', contrast == '{r.contrast}'; "
          f"data/processed/lgx/lgx_floor.csv | region == '{r.region}', scope == 'eval'")
    srcC = "results/rescale_wf3/data/processed/wf/wf3b_tests.csv"
    for _, r in C.iterrows():
        P(f"  c {r.region:7s} {r.component:8s}: {r.delta:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] / {r.delta_blockeq:+.4f} "
          f"[{r.delta_blockeq_lo:+.4f}, {r.delta_blockeq_hi:+.4f}] | {srcC} | test_id == '{r.test_id}', contrast == '{r.contrast}', "
          f"target == '{r.target}'" + (", item == 'WF9-a'" if r.test_id == "WF9-a" else ""))
    P("  b (블록별 RMSE 차, cm; 블록 id: 값; m = 자료 없음 마스크) | results/rescale_wf3/data/processed/wf/shards/wf9__cpu__Alaska__r__s*_blocksse.npz "
      "| unit target == 'Alaska|r'; keys [R1, catboost_lo, 1, cell, -1, 0, seed 0 and 1, -1.0] (seed mean) vs [P1, none, 1, cell, -1, 0, -1, 0.0]; "
      "SSE and cell counts summed over the 25 splits")
    for i in range(0, len(B), 6):
        P("    " + ", ".join(f"{int(r.block)}: {r.delta:+.2f}{'m' if r.masked else ''}" for _, r in B.iloc[i:i + 6].iterrows()))
    P(f"  b 관측 위치 {len(obs)}곳 | data/processed/fidelity_base_v3.csv | macro == 'Alaska', source_id == 'F4_direct', finite alt_cm; "
      "1 km 위치 = floor(lat/0.009), floor(lon·cos φ/0.009)")
    P("")
    P("[2] 패널 a: 오차 하한과 '하한 − 재보정 Stefan RMSE'(그림 명세 9.3 a, 설명문)")
    fl = meta["floor"]
    for reg, exp in (("Alaska", 11.31), ("Lena", 14.83), ("Canada", 17.90)):
        ok = abs(round(float(fl[reg]), 2) - exp) < 0.0051
        nbad += (not ok)
        P(f"  {'OK ' if ok else 'BAD'} floor {reg} = {float(fl[reg]):.4f} cm | 기대 {exp} | C4 README 2.6 E8.floor")
    for reg in REGIONS_A:
        q = A[(A.region == reg) & (A.method == "anchor_residual")].sort_values("n")
        P(f"  {reg}: " + ", ".join(f"n {('all' if n < 0 else n)}: recal RMSE {rb:.2f}, floor − recal {fm:+.2f}"
                                   for n, rb, fm in zip(q.n, q.rmse_recal, q.floor_minus_recal)))
    cq = A[(A.region == "Canada") & (A.method == "anchor_residual")].floor_minus_recal
    P(f"  캐나다 하한선 범위 {cq.max():+.2f} 에서 {cq.min():+.2f} cm(축 [−6, 4] 밖, 삼각 표지 1개, 설명문 '−10.6 to −10.8 cm')")
    r = A[(A.region == "Alaska") & (A.method == "anchor_residual") & (A.n == 1000)].iloc[0]
    F = float(fl["Alaska"])
    red_p1, red_r1 = r.rmse_recal ** 2 - F ** 2, r.rmse_method ** 2 - F ** 2
    pct = (red_p1 - red_r1) / red_p1 * 100
    P(f"  알래스카 n 1000: 재보정 Stefan {r.rmse_recal:.2f} → 잔차 ML {r.rmse_method:.2f} cm, 줄일 수 있는 오차 제곱 감소 {pct:.2f}% "
      f"(설명문 19%, C4 README D1.wf6.reducible_cut_pct 19.17%)")
    for n in (500, -1):
        rr = A[(A.region == "Alaska") & (A.method == "anchor_residual") & (A.n == n)].iloc[0]
        P(f"  알래스카 n {('all' if n < 0 else n)}: delta {rr.delta:+.4f}, verdict4 {rr.verdict4}, verdict4_common {rr.verdict4_common}")
    P(f"  알래스카 n 1000 verdict4 {r.verdict4}, verdict4_common {r.verdict4_common}")
    P("")
    P("[3] 패널 b: 블록 지도(2안)")
    bc = meta["bchk"]
    okb = abs(bc["delta_cell_weighted"] - (-0.515289)) < 5e-6
    nbad += (not okb)
    P(f"  저장소 조각 {bc['n_shards']}개, 분할 {bc['splits'][0]}–{bc['splits'][-1]}")
    P(f"  {'OK ' if okb else 'BAD'} 조각에서 다시 계산한 셀 가중 Δ(분할 평균) {bc['delta_cell_weighted']:+.6f} = wf3b/wf2b 시험 값 −0.515289 "
      f"(RMSE 잔차 ML {bc['rmse_resid']:.4f}, 재보정 Stefan {bc['rmse_recal']:.4f}; C8 README A1)")
    bi = meta["binfo"]
    P(f"  블록 {len(B)}개(표시 {bi['n_shown']}, 자료 없음 마스크 {bi['n_masked']}: 25분할 합산 채점 셀 < 10, 그림 명세 9.3 b [판단])")
    P(f"  블록 Δ 범위(표시 블록) {B[~B.masked].delta.min():+.2f} 에서 {B[~B.masked].delta.max():+.2f} cm, |Δ| 99 백분위 "
      f"{np.nanpercentile(np.abs(B[~B.masked].delta), 99):.2f} → vmax {bi['vmax']:.0f} cm(5 cm 단위 올림)")
    nuniq = int((B.n_cells_block < 10).sum())
    P(f"  참고: 블록의 고유 채점 셀 수(ncell) < 10 인 블록은 {nuniq}개다. 마스크 규칙은 명세대로 '분할 합산' 셀 수를 쓴다(같은 셀이 분할마다 다시 센다).")
    P(f"  관측 위치: 라벨 행 {meta['ochk']['n_label_rows']}, 1 km 위치 {meta['ochk']['n_loc_1km']}(그림 명세 343), 블록 {meta['ochk']['n_blocks']}(시험 저장소 블록 {len(B)})")
    P(f"  지도 범위 {bi['extent_km'][0]:.0f} × {bi['extent_km'][1]:.0f} km, {bi['km_per_mm']:.2f} km/mm")
    P(f"  블록 경도 폭(최종 크기): 중앙 위도 {bi['center_lat']:.1f}° N 에서 {bi['block_width_mm_center_lat']:.2f} mm, 블록별 "
      f"{bi['block_width_mm_min']:.2f}–{bi['block_width_mm_max']:.2f} mm, 1.0 mm 미만 블록 {bi['n_blocks_lt_1mm']}개(북쪽 사면)")
    P(f"  위도 라벨 {[s for _, s in bi['lat_ticks']]}, 경도 라벨 {[s for _, s in bi['lon_ticks']]}(경선은 10° 간격 5개, 라벨은 3개), "
      f"축척 막대 200 km(왼쪽 끝 {bi['scale_bar_lonlat'][1]:.1f}° N, {abs(bi['scale_bar_lonlat'][0]):.1f}° W; 투영 기준 위도 70° N)")
    P("")
    P("[4] 패널 c: 등록 행과 행 구성")
    P("  층화 평균 행은 격자 안 등록 대비(WF9-a 주 행)에만 있다. 총·격자 사이의 교차검증 λ 3지역 평균 행은 원천에 없어 그리지 않았다(새 집계 안 함).")
    for comp in ("total", "between", "within"):
        q = C[C.component == comp]
        P(f"  {comp}: " + "; ".join(f"{r.region} {r.delta:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] / {r.delta_blockeq:+.2f} "
                                    f"[{r.delta_blockeq_lo:+.2f}, {r.delta_blockeq_hi:+.2f}] {r.verdict4}" for _, r in q.iterrows()))
    dc = pd.read_csv(SRC["wf3b_decomp"])
    q = dc[(dc.exp == "wf9") & (dc.base == "Alaska|r") & (dc.method == "R1") & (dc.learner == "catboost_lo") & (dc.n == -1) & np.isclose(dc.lam, -1.0)]
    sgw = float(q.share_gain_within.iloc[0])
    ok99 = abs((1 - sgw) * 100 - 99) < 1.0
    nbad += (not ok99)
    P(f"  {'OK ' if ok99 else 'BAD'} SSE 이득의 격자 안 몫 {sgw*100:.2f}% → 격자 사이 약 {(1-sgw)*100:.1f}% (설명문 'about 99%', C8 README B4)")
    P("")
    P("[4b] 패널 e: XC 등록 주 가설(풀 행, 셀 가중 delta [lo, hi] / 블록 등가중 delta [lo, hi] cm)과 재보정 Stefan 대비 지역 예외")
    E, ER = data["E"], data["ER"]
    for _, r in E.iterrows():
        exp = E_EXPECT[r.hypothesis]
        got = (r.delta, r.ci_lo, r.ci_hi, r.delta_blockeq, r.delta_blockeq_lo, r.delta_blockeq_hi)
        ok = all(abs(round(g, 2 if abs(round(e_, 2) - e_) < 1e-12 else 3) - e_) < 1e-9 for g, e_ in zip(got, exp))
        nbad += (not ok)
        P(f"  {'OK ' if ok else 'BAD'} {r.hypothesis} {r.contrast} n {r.n}: " + ", ".join(f"{g:+.4f}" for g in got)
          + f" | 조정 담당 값 {exp} | {r['source']} | {r['filter']}; 4분 판정 {r.verdict4}(그림에 판정 기호 없음), Holm p {r.p_holm:.4f},"
          f" 풀 '{r.region}' ({r.pool_label}), 설계 '{r.design}', 맹검 '{r.blind}', 재표집 {r.nboot}")
    for n, grp in ER.groupby("n"):
        shown = set(grp[grp.shown].region)
        ok = shown == E_WORSE_EXPECT[int(n)]
        nbad += (not ok)
        P(f"  {'OK ' if ok else 'BAD'} 재보정 Stefan 대비 라벨 {int(n)}: 평가 지역 {len(grp)}곳, Δ > {E_WORSE_CM} cm 지역 {sorted(shown)} | 기대 "
          f"{sorted(E_WORSE_EXPECT[int(n)])}(xc_sentences.json, 조정 담당 목록); 지역 값 "
          + "; ".join(f"{r.region} {r.delta:+.4f} ({'그림' if r.shown else '안 그림'}; {r.source.split('/')[-1]})" for _, r in grp.iterrows()))
    sent = json.loads(SRC["xc_sent"].read_text(encoding="utf-8"))["hypotheses"]
    for hid, needles in (("XC-2a", ["Russia_E|x Δ +1.34"]), ("XC-2b", ["Alaska|x(선택 계열) Δ +1.51"]),
                         ("XC-2c", ["Lena|x Δ +0.53", "Alaska|x(선택 계열) Δ +2.34"])):
        ok = all(nd in sent[hid] for nd in needles)
        nbad += (not ok)
        P(f"  {'OK ' if ok else 'BAD'} 봉인 문장 {hid} 의 지역 예외 문자열 {needles}")
    P("  XC-5b·5c(무작위 대비, 교차검증 선정)와 XC-F3 은 '시험하지 않음(Algorithm P = S1)'이라 행이 없다. 그림 명세 9.3 e 의 셋째 묶음"
      " 'vs random, CV selection' 을 두지 않았다.")
    P("  [판단] 0선과 ±0.5 cm 띠는 묶음 행에만 그렸다. 묶음 머리(7 pt, 최대 약 37 mm)가 이름 열(약 13 mm)보다 길어 축 위를 지나가기 때문이다.")
    P("  [판단] 지역 기호는 재보정 Stefan 대비 묶음에만 둔다(등록 비열등 문장의 지역 규칙). 무작위 대비 묶음의 지역 값(레나 +0.21·+0.54,"
      " 알래스카 +1.26·+2.26 cm, 보조 표)은 Source Data 와 이 기록에만 둔다. 알래스카 값은 Algorithm P 를 고른 알래스카 계열(선택 계열)이다.")
    P("  묶음 머리 문구는 조정 담당 지시('vs random placement, fixed recipe', 'vs recalibrated Stefan')를 따랐다(명세의 4단어 한도보다 1단어 길다).")
    P("")
    P("[5] 패널 d: 경로 값(서술, 판정 아님)")
    for _, r in D.sort_values(["method", "element", "n"]).iterrows():
        P(f"  {r.method:16s} {r.element:12s} {r.region:18s} n {('all' if r.n < 0 else int(r.n)):>4}: {r.delta:+.4f}  ({r.source})")
    for k, v in meta["dchk"].items():
        okd = abs(v) < 1e-9
        nbad += (not okd)
        P(f"  {'OK ' if okd else 'BAD'} {k} = {v:.2e}(풀 점추정 = 지역 단순 평균, h42.pool_rows 의 delta 정의와 같다)")
    wm = D[(D.element == "path") & (D.method == "cv_selection")].set_index("n").delta
    for n, e in ((40, 0.55), (160, 0.18)):
        okw = abs(round(wm[n], 2) - e) < 0.0051
        nbad += (not okw)
        P(f"  {'OK ' if okw else 'BAD'} 규칙 선정 2지역 평균 n {n} = {wm[n]:+.4f} | 그림 명세 9.3 내용 위험 문단 {e:+.2f}")
    P(f"  규칙 선정 2지역 평균 전량 = {wm[-1]:+.4f}(레나델타 −0.32, 캐나다 +0.32)")
    P("  경로는 라벨 3–160 에서 0 위에 놓인다(그림 명세 9.3 내용 위험, D-13 결정 대기). 설명문 d 에 비열등 근거(Fig 4c) 문장을 둔다.")
    P("")
    P("[5b] F-17 패널 위계(그림 명세 2절 표의 슬롯 기준: 축 라벨 포함 슬롯 폭 × 높이)")
    slot_a = 3 * 52.0 * 42.0; slot_b = L["b_map"][2] * (L["b_cbar"][1] - L["b_map"][1])
    P(f"  a 슬롯 3 × 52 × 42 = {slot_a:.0f} mm2, b 슬롯 84 × {L['b_cbar'][1] - L['b_map'][1]:.1f} = {slot_b:.0f} mm2, 합 {slot_a + slot_b:.0f} / "
      f"{FIG_W * FIG_H:.0f} = {(slot_a + slot_b) / (FIG_W * FIG_H):.0%} (명세 41%, 기준 35% 이상); 축 자체 면적은 a 3 × {L['a_wm'] + L['a_gap_all'] + L['a_wa']:.1f} × {L['a_h']:.0f}, "
      f"b {L['b_map'][2]:.0f} × {L['b_map'][3]:.1f} mm. 가장 큰 패널 b")
    P("")
    P(f"[6] audit_v3(지침 부록 A.1) 결과. 문자 수 한도 {MAX_CHARS}: 그림 명세 9.5 의 600 자에 e(XC)의 묶음 머리 2, 행 이름 5, 지역 열쇠 3, 축 이름을"
      " 더했다(Fig 6 의 800 자와 같은 사유 기록)")
    for k in ("fails", "sizes", "chars", "thin_lines", "titles", "boxed_text", "codes", "comma4", "long_labels", "loose_numbers", "text_overlaps"):
        P(f"  {k}: {audit.get(k)}")
    P("  (수치 검사 제외 gid: scale, sizekey, rowname_n(e 의 행 이름 '10 labels' 등, 명세 9.3 e 가 지정한 범주 이름))")
    P("")
    P("[7] pdf_audit(지침 부록 A.2)와 pdffonts")
    for k, v in pdfa.items():
        P(f"  {k}: {v}")
    P("  pdffonts:")
    for line in fonts_txt.strip().splitlines():
        P("    " + line)
    P(f"  PNG: {png_info}")
    P("")
    P("[8] 설명문 Fig7_legend.md 점검")
    for line in legend_check():
        P("  " + line)
    P("")
    P(f"[9] 요약: 값 대조 불일치 {nbad}건")
    Path(path).write_text("\n".join(L_) + "\n", encoding="utf-8")
    return nbad


def drawn_text_overlaps(fig) -> dict:
    """style.text_overlaps 와 같되, 그려지지 않는 GeoAxes·축을 끈 축의 기본 눈금 라벨 객체(보이는 값으로 남는다)는 뺀다(fig6.drawn_texts 와 같은 규칙)."""
    import itertools
    from matplotlib.text import Text
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    hidden = set()
    for a in fig.axes:
        if (not a.axison) or hasattr(a, "projection"):
            for axis in (a.xaxis, a.yaxis):
                hidden.update(axis.get_ticklabels(which="both"))
                hidden.add(axis.label)
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t not in hidden]
    bbs = [(t.get_text().strip(), t.get_window_extent(rend)) for t in texts]
    pairs = [(a_[:25], b_[:25]) for (a_, ba), (b_, bb) in itertools.combinations(bbs, 2) if ba.overlaps(bb)]
    W, H = fig.get_size_inches() * fig.dpi
    outside = [s_[:25] for s_, b in bbs if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5]
    return dict(n_texts=len(bbs), overlap_pairs=pairs, outside=outside)


XC_PLACEHOLDER = "[XC: Fig 7e | 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) | 2.3 XC-1, XC-2 포레스트]"   # 그림 명세 12절


def legend_check():
    """Fig7_legend.md 의 단어 수(350 이하), 금지 표현, 색·모양 서술, 지도 기재 사항, XC 자리표시 문자열을 점검한다."""
    import re
    p = OUT / "Fig7_legend.md"
    if not p.exists():
        return ["Fig7_legend.md 없음"]
    body = "\n".join(ln for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#") and not ln.startswith("<!--"))
    body = re.sub(r"\*\*", "", body)
    words = len(body.split())
    words_wo = len(body.replace(XC_PLACEHOLDER, "").split())
    ta = S.text_audit(body.replace(XC_PLACEHOLDER, ""))
    colour = re.findall(r"\b(grey|gray|black|white|blue|red|green|open|filled|thick|thin|dashed|dotted|solid|circle)\b", body, re.I)
    m_e = re.search(r"\be, (.*?) Intervals", body)
    n_e = len(m_e.group(1).split()) if m_e else -1
    flag = "designed after results were inspected; re-test in reused regions"
    out = [f"words {words} (limit 350; without the XC placeholder {words_wo}); first sentence words {len(body.split('. ')[0].split())}",
           f"dash {ta['dash']}; banned {ta['banned']}; middot-number {ta['middot_num']}; comma4 {ta['comma4']}; internal codes {ta['codes']}; "
           f"colour/shape words {colour}; 'This figure' start: {body.lstrip().lower().startswith('this figure')}",
           f"XC placeholder removed: {XC_PLACEHOLDER not in body}; e sentence words {n_e} (limit 45); flag '{flag}' present: {flag in body}"]
    for must in ("Natural Earth", "Cartopy", "stereographic", "10,000", "25 splits"):
        out.append(f"contains '{must}': {must in body}")
    return out


def main():
    avail, load = None, None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024 / 1024
        load = os.getloadavg()[0]
    except OSError:
        pass
    print(f"resources: available {avail:.1f} GB, load {load:.1f}")
    if avail is not None and (avail < float(os.environ.get("PAPER_MIN_AVAIL_GB", "30")) or load > 40):   # 기본 30 GB(작업 지시), 조정 담당이 정한 값은 환경 변수로
        raise SystemExit("shared server busy: available < 30 GB or load > 40; wait and rerun")
    fig, data, meta = build("paper")
    audit = S.audit_v3(fig, max_chars=MAX_CHARS, allowed_num_gids=("scale", "sizekey", "rowname_n", "offaxis"))
    audit["text_overlaps"] = drawn_text_overlaps(fig)
    print("text_overlaps:", audit["text_overlaps"])
    print("audit_v3 fails:", audit["fails"], "chars", audit["chars"], "sizes", audit["sizes"])
    for k in ("thin_lines", "titles", "boxed_text", "codes", "comma4", "long_labels", "loose_numbers"):
        if audit[k]:
            print("  ", k, audit[k])
    paths = S.save_fig(fig, "Fig7", OUT, formats=("pdf", "png"))
    pdf = OUT / "Fig7.pdf"
    pdfa = S.pdf_audit(pdf)
    fonts_txt = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    from PIL import Image
    with Image.open(OUT / "Fig7.png") as im:
        png_info = dict(size_px=im.size, dpi=im.info.get("dpi"))
    write_source_data(data, OUT / "Fig7_source_data.csv")
    nbad = write_values(OUT / "Fig7_values.txt", data, meta, audit, pdfa, fonts_txt, png_info)
    print("pdf_audit:", pdfa)
    print("values mismatches:", nbad)
    print("written:", [str(p) for p in paths])


if __name__ == "__main__":
    main()
