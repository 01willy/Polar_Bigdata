"""v7 F4 핵심 결과(개정 v9, 2026-10-15, 설계 지침 v9 2.3 항 16–18): a 그 해 입력 효과, b 검증 사다리 경사 차트, c 공변량 사다리 이득,
d TDD 3분위 이득, e 물리 정보의 위치별 이득(모델 구조 4종 − 재보정 Stefan), f 학습기별 이득.
자료: data/processed/v5/runs/{core_v6,modis_v6,featsets_v6,rf_v6,nn_v6}. 수치는 모두 CSV 에서 읽는다.
실행: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v7_f4_results_core.py [--fig 4] [--slide | --deck]   (CPU 만)
배치: 모든 축을 mm 격자(ax_mm)로 둔다. 논문판은 고정 캔버스 180 × 199 mm, 세 행(a·b / c·d / e·f)로 저장한다(bbox tight 없음).
    1행(a, b)의 오른쪽 끝은 그린 뒤 측정해 2·3행의 오른쪽 끝(R)에 맞춘다. 2·3행은 범주 하나의 폭(u)을 같게 두고 행마다 폭을 채운다.
    범례는 지역 범례(a 안) 하나와 마커 범례(f 위, 채움 = 물리 + 잔차 ML, 빈 = 직접 ML) 하나뿐이다.
    b: 축을 끊지 않는다. 나머지 최대의 1.3 배를 넘는 값(레나 직접 ML 지역 홀드아웃)만 축 위 끝에서 빈 마커로 자르고 값을 적으며, 선은 그 마커까지 잇는다.
    c–f: 신뢰구간을 자르지 않고 y 범위를 행마다(c·d, e·f) 모든 구간이 들어가게 둔다.
슬라이드판(--slide): 같은 배치를 12 in 폭 PNG 와 패널 크롭(a, b, c, d, e, f, cd, ef)으로 저장한다(덱 명세 fallback_assets).
덱판(--deck): 덱 본문 크기(12.0 × 4.65 in, 300 dpi, 3600 × 1395 px, RGB) PNG 를 outputs/figures/paper/v7/deck/ 에 둔다(v9 지침 1.4).
    R01 = 지역별 패널, 방법(Stefan (recalibrated) · Physics + residual ML · Direct ML)마다 평년 기후 입력(빈 원)과 관측 연도 기후 입력(채운 원)의
          공간 블록 RMSE 를 가는 선으로 잇고, 쌍 아래에 Δ(관측 연도 − 평년)와 95% 블록 부트스트랩 구간(deltas_forcing.csv)을 적는다.
          세 패널의 y 폭(cm)은 같다(변화량을 패널끼리 비교할 수 있게).
    R02 = b 세 소패널 + 제품 기준선(products_eval.csv, mask 0, 정의 'year-matched' = 그 해 제품 값이 있는 라벨 셀-연도; Wei 2026 · Liu 2024 ·
          ESA CCI v4, 제품색 가는 파선, 소패널마다 오른쪽 끝 라벨). 방법은 위쪽 범례 한 줄.
    R03 = c·d(두 패널), R03b = e·f(두 패널), R03e = 공변량별 3분위 이득 열지도(scripts/4_visualization/v9_f4e_gain_conditions.py 가 계산·그림).
    R03m(2026-10-16, 쪽 병합) = a 공변량 사다리 가로판(행 = 공변량 집합, 왼쪽 약 35 %) + b 공변량별 3분위 이득 열지도(오른쪽 약 65 %).
          TDD 3분위 패널은 뺀다(TDD 는 열지도의 한 행). 자료 부족 칸(셀-연도 < 30 또는 블록 < 5)은 옅은 회색 빗금, 아래 열쇠 줄에
          '95% CI excludes 0' 과 'Insufficient data (< 30 cell-years or < 5 blocks)'. 값은 소수 한 자리(|Δ| < 0.05 cm 는 0.0).
    --only R03m 처럼 덱 쪽을 골라 낼 수 있다(R03 을 고르면 R03·R03b 를 함께 낸다).
    그리기 함수와 자료 적재는 논문판과 같고, 크기 묶음(K_PAPER·K_DECK)만 다르다. 덱판 글자·선 크기는 토큰 font.slide·lines.slide.
어휘(v9 1.1): 'forcing'·'climatological' 대신 'climate input', 방법 이름 Stefan (source coeff.) · Stefan (recalibrated) · Physics + residual ML ·
    Direct ML · Physics-input ML · Physics pseudo-labels.
"""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator, MultipleLocator

PROC = ROOT / "data" / "processed" / "v5"; RUNS = PROC / "runs"
OUT = ROOT / "outputs" / "figures" / "paper" / "v7"; DECK_OUT = OUT / "deck"
REG = ["Alaska", "Lena", "Canada"]
MAIN = dict(forcing="year_obs", featset="F8TS", learner="catboost", lam=0.5, scheme="block")
METH_EN = {"P0": "Stefan (source coeff.)", "P1": "Stefan (recalibrated)", "Ku": "Kudryavtsev", "D0": "Direct ML", "F1": "Physics-input ML",
           "D1": "Physics pseudo-labels", "R1": "Physics + residual ML"}           # v9 어휘(1.1)
METH_2L = {"P0": "Stefan\n(source coeff.)", "P1": "Stefan\n(recalibrated)", "Ku": "Kudryavtsev", "D0": "Direct ML", "F1": "Physics-\ninput ML",
           "D1": "Physics\npseudo-labels", "R1": "Physics +\nresidual ML"}   # 두 줄(덱판 가로 눈금 라벨)
SCHEMES = ["random", "block", "year", "block_year", "region"]
SCHEME_LB = ["random", "block", "year", "block × year", "region"]
SCHEME_2L = ["random", "block", "year", "block\n× year", "region"]          # 덱판 가로 눈금 라벨(× 앞에서 줄바꿈: 윗줄이 짧아 이웃 라벨과 띄운다)
FEATS = ["F8", "F8T", "F8TS", "F8TS_M", "F8TS_X", "F8TS_ALL"]
FEAT_LB = ["climate only", "+terrain", "+soil", "+MODIS", "+hi-res", "all"]
FEAT_2L = ["climate\nonly", "+terrain", "+soil", "+MODIS", "+hi-res", "all"]
STRUCT = ["D0", "F1", "D1", "R1"]                                          # e 모델 구조(물리 정보가 들어가는 위치)
LEARN = ["catboost", "rf", "mlp", "tabm", "ftt"]; LEARN_LB = ["CatBoost", "RF", "MLP", "TabM", "FT-T"]
OFF = 0.26                      # 지역 x 오프셋(범주 단위)
MS, WLW = 2.8, 0.8              # 논문판 마커 지름(pt), 신뢰구간 선 굵기(pt)
W, H = S.W2, 199.0              # 논문 캔버스(mm)
L, R = 16.0, 176.0              # 축 왼쪽 x, 세 행 공통 오른쪽 끝 x(mm)
ROW1, ROW2, ROW3, HH = 154.5, 88.5, 24.5, 38.0
XL1, XL2, XL3 = 18.5, 12.5, 19.5   # x 축 라벨을 축 아래 두는 거리(mm): 기울인 눈금 라벨(a·b), 가로 눈금 라벨(c·d), 3행(e 의 긴 기울인 라벨과 f)
GAP_B = 2.8                     # 슬라이드판 b: 가장 낮은 눈금 라벨 아래 → x 축 라벨 위 거리(mm). 논문판은 XL1(a 의 'Method' 와 같은 높이)
G23 = 14.3                      # 2·3행 패널 사이 간격(mm)
UCAT = dict(c=6.2, d=3.2, e=4.2, f=5.2)   # 패널별 범주 폭 합(범주 수 + 0.2; x 범위 −0.6 … n − 0.4)
YLAB = "ΔRMSE (cm)\nvs Stefan (recalibrated)"
OVER = 1.3                      # b: 나머지 최대의 OVER 배를 넘는 값은 축 위 끝에서 자른다(토큰 symbols.out_of_range)

# 크기 묶음(pt). 논문판은 토큰 font.paper·lines.paper, 덱판은 font.slide·lines.slide 에 맞춘다(마커 7, 자료선 2.2, 보조선 1.4, 직접 라벨 13, 눈금 12)
_FS, _LN = S.F["slide"], S.TOK["lines"]["slide"]
K_PAPER = dict(ms=MS, wlw=WLW, mew=0.8, lw=1.0, lw_thin=0.6, ms_b=2.4, ms_b_thin=1.8, ms_clip=2.6, lw_line=0.7, lw_d0=0.5, zlw=0.6,
               fs_num=6, fs_lab=6, fs_reg=6.5, fs_leg=6.5, off=OFF, off_e=0.28, sub_e=0.07, lab_h_in=5.4 / 25.4, lab_dx_pt=4, num_dy_pt=2.0,
               lw_prod=0.6, prod_lab_h_in=2.6 / 25.4)
K_DECK = dict(ms=_LN["marker_pt"], wlw=2.4, mew=1.6, lw=_LN["data_pt"], lw_thin=_LN["data_aux_pt"], ms_b=6.0, ms_b_thin=4.5, ms_clip=6.5,
              lw_line=1.6, lw_d0=_LN["ci_block_pt"], zlw=_LN["reference_pt"], fs_num=_FS["tick_pt"], fs_lab=_FS["direct_label_pt"],
              fs_reg=_FS["axis_label_pt"], fs_leg=_FS["direct_label_pt"], off=0.30, off_e=0.30, sub_e=0.08, lab_h_in=0.44, lab_dx_pt=7, num_dy_pt=3.0,
              lw_prod=1.4, prod_lab_h_in=0.21)
DECK_W, HD = 12.0, 4.65        # 덱 본문 그림 크기(in): v9 지침 1.4 일반 쪽(3600 × 1395 px)
DY0, DXL = 0.88, 0.60           # 덱판 축 아래 여백(in), x 축 라벨을 축 아래 두는 거리(in). 두 줄 눈금 라벨(12 pt) 기준
PROD_R02 = [("wei", "Wei"), ("liu", "Liu"), ("cci4y", "CCI v4")]   # R02 제품 기준선(제품 키, 짧은 오른쪽 끝 라벨: Wei 2026 · Liu 2024 · ESA CCI v4)
R01_METH = ["P1", "R1", "D0"]   # R01 방법 순서


# ---------------------------------------------------------------- 공통 도우미
def ax_mm(fig, x, y, w, h):
    """mm 좌표(왼쪽 아래 원점)로 축을 둔다. 여백·정렬을 격자로 통제하기 위한 것."""
    Wf, Hf = fig.get_size_inches() * 25.4
    return fig.add_axes([x / Wf, y / Hf, w / Wf, h / Hf])


def ax_in(fig, x, y, w, h):
    return ax_mm(fig, x * 25.4, y * 25.4, w * 25.4, h * 25.4)


def rect_in(fig, x, y, w, h):
    """inch 좌표 → 그림 비율 사각형."""
    fw, fh = fig.get_size_inches(); return [x / fw, y / fh, w / fw, h / fh]


def letter(ax, s, w_mm, h_mm, dx_mm=4.0, dy_mm=1.0):
    """패널 문자를 축 왼쪽 위에서 같은 절대 거리(mm)에 둔다(논문판 8 pt)."""
    S.panel_letter(ax, s, x=-dx_mm / w_mm, y=1 + dy_mm / h_mm)


def deck_letter(ax, s, dx_in=0.30, dy_in=0.06):
    """덱판 패널 문자(토큰 font.slide.panel_letter_pt, 굵게). 패널이 둘 이상인 슬라이드에서만 쓴다."""
    fw, fh = ax.figure.get_size_inches(); p = ax.get_position()
    ax.text(-dx_in / (p.width * fw), 1 + dy_in / (p.height * fh), s, transform=ax.transAxes, fontsize=_FS["panel_letter_pt"], fontweight="bold",
            va="bottom", ha="left")


def deck_figure(h_in):
    """덱 캔버스 12.0 × h in. mm 환산의 부동소수 오차로 300 dpi 픽셀 수가 1 모자라지 않도록 크기를 inch 로 다시 둔다(+1e-9 in)."""
    fig = S.figure(DECK_W * 25.4, h_in * 25.4); fig.set_size_inches(DECK_W + 1e-9, h_in + 1e-9)
    return fig


def deck_rc():
    """덱판 rcParams: 축 라벨 14, 눈금 12, 범례·직접 라벨 13 pt, 선·눈금 굵기는 lines.slide."""
    plt.rcParams.update({"font.size": _FS["direct_label_pt"], "axes.labelsize": _FS["axis_label_pt"], "xtick.labelsize": _FS["tick_pt"],
                         "ytick.labelsize": _FS["tick_pt"], "legend.fontsize": _FS["direct_label_pt"], "axes.linewidth": _LN["axis_pt"],
                         "xtick.major.width": _LN["tick_pt"], "ytick.major.width": _LN["tick_pt"], "xtick.major.size": _LN["tick_len_pt"],
                         "ytick.major.size": _LN["tick_len_pt"], "lines.linewidth": _LN["data_pt"], "lines.markersize": _LN["marker_pt"],
                         "xtick.major.pad": 4.0, "ytick.major.pad": 4.0, "axes.labelpad": 6.0})


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        if k in df.columns:
            m &= (df[k] == v).values
    return df[m]


def method_rows(df, m):
    """방법 m 의 행: R1 은 lam 0.5·catboost, 학습 방법(D0·F1·D1)은 catboost, 물리식은 learner '-'."""
    d = df[df.method == m]
    if m == "R1":
        d = d[(d.lam == MAIN["lam"]) & (d.learner == MAIN["learner"])]
    elif m in ("D0", "F1", "D1"):
        d = d[d.learner == MAIN["learner"]]
    return d


def _lum(hex_):
    def ch(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = to_rgb(hex_); return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(hex_, bg="#FFFFFF"):
    a, b = sorted([_lum(hex_), _lum(bg)], reverse=True); return (a + 0.05) / (b + 0.05)


def text_safe(hex_, target=4.5):
    """흰 바탕 대비 WCAG ≥ target 이 될 때까지 같은 색상을 어둡게 한다(글자 전용; 선·마커는 원색)."""
    rgb = np.array(to_rgb(hex_)); f = 1.0
    while contrast(to_hex(rgb * f)) < target and f > 0.05:
        f -= 0.01
    return to_hex(rgb * f)


def fmt(v):
    return f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}"


def errbars(ax, x, s, color, k, line=False, mfc=None, lw=None, zorder=3):
    """점 + 95 % 신뢰구간(자르지 않는다. y 범위는 호출 쪽이 모든 구간이 들어가게 둔다)."""
    y, lo, hi = s.delta.values.astype(float), s.lo.values.astype(float), s.hi.values.astype(float)
    mfc = color if mfc is None else mfc; lw = k["wlw"] if lw is None else lw
    ax.vlines(x, lo, hi, color=color, lw=lw, zorder=zorder - 1)
    if line:
        ax.plot(x, y, "-", color=color, lw=k["lw_line"], alpha=0.9, zorder=1)
    ax.plot(x, y, "o", ms=k["ms"], mfc=mfc, mec=color, mew=k["mew"] if mfc == "white" else 0, lw=0, zorder=zorder)


def zero(ax, k):
    ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=k["zlw"], zorder=0)


def repel(y, h, lo, hi):
    """직접 라벨 y 위치: 각 라벨 높이 h(자료 단위)만큼 떨어지도록 밀어낸다(정렬 순서 유지, 축 범위 안)."""
    o = np.argsort(y); y = np.array(y, float); h = np.array(h, float); p = y[o].copy(); hh = h[o]
    for _ in range(60):
        moved = False
        for i in range(1, len(p)):
            need = (hh[i - 1] + hh[i]) / 2
            if p[i] - p[i - 1] < need:
                d = (need - (p[i] - p[i - 1])) / 2; p[i - 1] -= d; p[i] += d; moved = True
        if p[0] - hh[0] / 2 < lo:
            p += lo - (p[0] - hh[0] / 2); moved = True
        if p[-1] + hh[-1] / 2 > hi:
            p -= (p[-1] + hh[-1] / 2) - hi; moved = True
        if not moved:
            break
    out = np.empty_like(p); out[o] = p
    return out


def legend_above(ax, handles=None, labels=None, **kw):
    a = dict(loc="lower left", bbox_to_anchor=(-0.02, 1.0), handlelength=0.8, handletextpad=0.4, columnspacing=1.0, borderaxespad=0, fontsize=6.5)
    a.update(kw)
    return ax.legend(handles, labels, **a) if handles is not None else ax.legend(**a)


def region_handles(k):
    return [plt.Line2D([], [], marker="o", ms=k["ms"], mfc=S.REGION_HEX[r], mec=S.REGION_HEX[r], lw=0) for r in REG]


def marker_handles(k):
    """채운 원 = 물리 + 잔차 ML, 빈 원 = 직접 ML(회색)."""
    gz = S.NEUTRAL["zero_line"]
    return [plt.Line2D([], [], marker="o", ms=k["ms"], mfc=gz, mec=gz, lw=0), plt.Line2D([], [], marker="o", ms=k["ms"], mfc="white", mec=gz, mew=k["mew"], lw=0)]


def region_legend(ax, k, deck=False):
    kw = dict(handlelength=1.0, handletextpad=0.5, columnspacing=1.2) if deck else {}
    return legend_above(ax, region_handles(k), [S.REGION_EN[r] for r in REG], ncol=3, fontsize=k["fs_leg"], **kw)


def marker_legend(ax, k, deck=False):
    kw = dict(handlelength=1.0, handletextpad=0.5, columnspacing=1.2) if deck else {}
    return legend_above(ax, marker_handles(k), [METH_EN["R1"], METH_EN["D0"]], ncol=2, fontsize=k["fs_leg"], **kw)


def yrange(*dfs, pad=0.05):
    """여러 표의 신뢰구간(lo, hi)이 모두 들어가는 y 범위(자르지 않는다)."""
    ci = pd.concat([d[["lo", "hi"]] for d in dfs]); lo, hi = float(ci.lo.min()), float(ci.hi.max()); p = (hi - lo) * pad
    return (lo - p, hi + p)


def export_paper(fig, stem, outdir, dpi=600):
    """논문판: 고정 캔버스(figsize 그대로) pdf·svg·png. 쪽 크기가 spec 의 size_mm 와 같아진다."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True); out = []
    for ext in ("pdf", "svg", "png"):
        p = outdir / f"{stem}.{ext}"; fig.savefig(p, dpi=dpi); out.append(p)
    plt.close(fig)
    return out


def save_deck(fig, slide, h_in):
    """덱판 PNG: 캔버스 그대로(bbox tight 없음) 300 dpi, 흰 바탕, RGB. 저장 뒤 픽셀 크기와 잉크가 캔버스를 채운 비율을 확인한다."""
    from PIL import Image
    DECK_OUT.mkdir(parents=True, exist_ok=True); p = DECK_OUT / f"F4_results_core__deck_{slide}.png"
    fig.canvas.draw(); bb = fig.get_tightbbox(fig.canvas.get_renderer()); fw, fh = fig.get_size_inches()
    fig.savefig(p, dpi=300, facecolor="white"); plt.close(fig)
    im = Image.open(p); im.load(); rgb = im.convert("RGB"); im.close(); rgb.save(p, dpi=(300, 300)); size = rgb.size
    want = (int(round(DECK_W * 300)), int(round(h_in * 300)))
    print(f"{slide}: {size[0]}x{size[1]} px (want {want[0]}x{want[1]}) | ink fill w {bb.width / fw:.0%} h {bb.height / fh:.0%} | "
          f"margins in: left {bb.x0:.2f} right {fw - bb.x1:.2f} bottom {bb.y0:.2f} top {fh - bb.y1:.2f}")
    assert size == want, (size, want)
    return p


def margins_mm(fig):
    """QA: 모든 잉크의 꽉 찬 상자와 캔버스 가장자리 사이 여백(mm)."""
    r = fig.canvas.get_renderer(); bb = fig.get_tightbbox(r); k = 25.4
    return dict(left=bb.x0 * k, right=W - bb.x1 * k, bottom=bb.y0 * k, top=H - bb.y1 * k)


# ---------------------------------------------------------------- 자료
def load():
    core = RUNS / "core_v6"
    df = pd.read_csv(core / "deltas_forcing.csv"); df = sel(df, scheme=MAIN["scheme"], featset=MAIN["featset"], pair="year_obs:clim")
    sm = pd.read_csv(core / "summary_metrics.csv"); sm = sel(sm, forcing=MAIN["forcing"], featset=MAIN["featset"])
    ct = pd.read_csv(core / "conditions_tdd.csv"); ct = sel(ct, **{k: MAIN[k] for k in ("scheme", "forcing", "featset")})
    ct = method_rows(ct, "R1")
    # 공변량 사다리: modis_v6 우선, 없으면 featsets_v6
    lad = []
    for tag in ("modis_v6", "featsets_v6"):
        d = pd.read_csv(RUNS / tag / "deltas_vs_ref.csv"); d = sel(d, scheme=MAIN["scheme"], forcing=MAIN["forcing"], ref="P1"); d = method_rows(d, "R1"); lad.append(d.assign(_tag=tag))
    lad = pd.concat(lad).drop_duplicates(["region", "featset"], keep="first")
    # 모델 구조(e): core_v6 의 D0·F1·D1·R1 − P1(블록 CV, 관측일 정합, F8TS, catboost)
    st = pd.read_csv(core / "deltas_vs_ref.csv"); st = sel(st, scheme=MAIN["scheme"], forcing=MAIN["forcing"], ref="P1", featset=MAIN["featset"])
    st = pd.concat([method_rows(st, m) for m in STRUCT])
    # 학습기(f): core_v6(catboost), rf_v6, nn_v6
    lr = []
    for tag in ("core_v6", "rf_v6", "nn_v6"):
        d = pd.read_csv(RUNS / tag / "deltas_vs_ref.csv"); d = sel(d, scheme=MAIN["scheme"], forcing=MAIN["forcing"], ref="P1", featset=MAIN["featset"])
        d = d[d.method.isin(["R1", "D0"]) & ((d.method != "R1") | (d.lam == MAIN["lam"])) & d.learner.isin(LEARN)]; lr.append(d.assign(_tag=tag))
    lr = pd.concat(lr).drop_duplicates(["region", "method", "learner"], keep="first")
    return df, sm, ct, lad, st, lr


# ---------------------------------------------------------------- 패널 그리기(논문판·덱판 공용)
def draw_a(ax, df, k):
    """a(논문판) 관측 연도 기후 입력의 효과: 방법 × 지역의 ΔRMSE(관측 연도 − 평년 기후 입력) 점·95 % 구간. 지역 범례는 이 패널에만 둔다.
    덱판은 deck_r01(지역별 패널의 RMSE 쌍 점)을 쓴다."""
    meths = ["P0", "P1", "Ku", "D0", "R1"]
    for i, r in enumerate(REG):
        d = df[df.region == r]; s = pd.concat([method_rows(d, m) for m in meths]).set_index("method").reindex(meths)
        errbars(ax, np.arange(len(meths)) + (i - 1) * k["off"], s, S.REGION_HEX[r], k)
    zero(ax, k); ax.set_xticks(range(len(meths)))
    ax.set_xticklabels([METH_EN[m] for m in meths], rotation=40, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.6, len(meths) - 0.4)
    ax.set_xlabel("Method"); ax.set_ylabel("ΔRMSE (cm), year-specific −\nmean climate input")
    region_legend(ax, k)


def draw_b(fig, sm, rects, k, deck=False, xl_frac=None, prods=None, method_labels=True):
    """b 검증 사다리 경사 차트: 지역별 소패널(rects: 그림 비율 [x, y, w, h] 셋). 돌려주는 값: 주 축 목록.
    축은 끊지 않는다. 나머지 최대의 OVER 배를 넘는 값(레나 직접 ML 지역 홀드아웃)만 축 위 끝에서 빈 마커로 자르고 값을 적으며,
    그 점까지의 선은 끊지 않고 마커까지 잇는다. 나머지 소패널은 모든 값이 들어가는 범위.
    prods(덱 R02): {지역: [(제품 키, 라벨, RMSE)]} 이면 제품색 가는 파선 수평 기준선과 소패널 오른쪽 끝 라벨(겹침 밀어내기)을 그리고 y 범위에 넣는다.
    method_labels: 마지막 소패널 오른쪽 끝의 방법 직접 라벨(논문판). 덱 R02 는 위쪽 범례를 쓴다.
    지역 홀드아웃에서는 재보정할 역내 자료가 없어 P1 이 원천 계수로 되돌아가 P0 와 같은 값이 된다(설명은 캡션)."""
    series = ["P0", "P1", "D0", "R1"]; main_axes = []; fh = fig.get_size_inches()[1]; nx = len(SCHEMES)
    for j, r in enumerate(REG):
        x0, y0, w, h = rects[j]; d = sm[sm.region == r]
        vals = {m: method_rows(d, m).set_index("scheme").reindex(SCHEMES).rmse.values.astype(float) for m in series}
        allv = np.array([vals[m] for m in series]); thr = OVER * np.nanmax(allv[:, :4]); over = allv > thr
        pv = np.array([v for _, _, v in prods[r]]) if prods is not None else np.array([])
        ylo = min(np.nanmin(allv), *pv) if len(pv) else np.nanmin(allv); yin = max(np.nanmax(allv[~over]), *pv) if len(pv) else np.nanmax(allv[~over])
        span = yin - ylo
        yhi = yin + (0.16 if over.any() else 0.08) * span; ylo = ylo - 0.08 * span   # 잘린 점은 축 위 끝(yhi)에 둔다
        ax = fig.add_axes([x0, y0, w, h]); main_axes.append(ax); ax.set_ylim(ylo, yhi); ends = {}
        for m in series:
            v = vals[m]; c = S.METHOD_HEX[m]; thin = m == "P0"; ov = v > thr; vc = np.where(ov, yhi, v)
            kw = dict(ls=S.METHOD_LS.get(m, "-"), color=c, lw=k["lw_thin"] if thin else k["lw"], zorder=2 if thin else 3)
            ax.plot(range(nx), vc, clip_on=False, **kw)                                   # 선: 잘린 점까지 잇는다
            ax.plot(np.arange(nx)[~ov], v[~ov], "o", color=c, ms=k["ms_b_thin"] if thin else k["ms_b"], mew=0, lw=0, zorder=kw["zorder"])
            for xi in np.where(ov)[0]:                                                   # 잘린 점: 빈 원 + 값
                ax.plot([xi], [yhi], "o", ms=k["ms_clip"], mfc="white", mec=c, mew=k["mew"], clip_on=False, zorder=4)
                ax.annotate(fmt(v[xi]), (xi, yhi), xytext=(0, k["num_dy_pt"] + k["ms_clip"] / 2), textcoords="offset points", ha="center", va="bottom",
                            fontsize=k["fs_num"], color=text_safe(c), annotation_clip=False)
            ends[m] = vc[-1]
        xlim = (-0.25, nx - 0.75)
        ax.set_xticks(range(nx)); ax.set_xlim(*xlim)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True, steps=[1, 2, 5, 10]) if deck else MaxNLocator(4))   # 덱판은 정수 눈금(2·5 단위)
        if deck:
            ax.set_xticklabels(SCHEME_2L)
        else:
            ax.set_xticklabels(SCHEME_LB, rotation=40, ha="right", rotation_mode="anchor")
        ax.text(0.0, 1.0 + (0.06 / (h * fh) if deck else 0.0), S.REGION_EN[r], transform=ax.transAxes, ha="left", va="bottom", fontsize=k["fs_reg"],
                color=S.REGION_HEX[r] if deck else S.NEUTRAL["text_aux"])
        if prods is not None:   # 제품 기준선: 제품색 가는 파선, 오른쪽 끝 라벨(대비 ≥ 4.5 로 어둡게 한 제품색)
            h_in = ax.get_position().height * fh; unit = (yhi - ylo) / h_in
            for key, _, v in prods[r]:
                ax.axhline(v, color=S.PRODUCT_HEX[key], lw=k["lw_prod"], ls=(0, (2.6, 1.8)), zorder=1)
            ys = repel([v for _, _, v in prods[r]], [k["prod_lab_h_in"] * unit] * len(prods[r]), ylo, yhi)
            for (key, lab, _), yy in zip(prods[r], ys):
                ax.annotate(lab, (nx - 0.75, yy), xytext=(k["lab_dx_pt"], 0), textcoords="offset points", ha="left", va="center", fontsize=k["fs_num"],
                            color=text_safe(S.PRODUCT_HEX[key]), annotation_clip=False)
        if j == 0:
            ax.set_ylabel("RMSE (cm)")
        if j == 1:
            ax.set_xlabel("Validation split"); ax.xaxis.set_label_coords(0.5, xl_frac)
        if method_labels and j == len(REG) - 1:   # 오른쪽 끝 직접 라벨(마지막 소패널에 한 번). 글자색은 대비 ≥ 4.5 로 어둡게 한 방법 색
            h_in = ax.get_position().height * fh; unit = (yhi - ylo) / h_in
            ys = repel([ends[m] for m in series], [k["lab_h_in"] * unit] * len(series), ylo, yhi)
            for m, yy in zip(series, ys):
                ax.annotate(METH_2L[m], (nx - 1, yy), xytext=(k["lab_dx_pt"], 0), textcoords="offset points", ha="left", va="center",
                            fontsize=k["fs_lab"], color=text_safe(S.METHOD_HEX[m]), linespacing=1.0, annotation_clip=False)
    return main_axes


def draw_c(ax, ladf, k, deck=False):
    """c 공변량 사다리: 집합별 R1 − P1 ΔRMSE(지역 색, 점·구간·연결선)."""
    for i, r in enumerate(REG):
        s = ladf[ladf.region == r].set_index("featset").reindex(FEATS)
        ok = s.delta.notna().values; x = (np.arange(len(FEATS)) + (i - 1) * k["off"])[ok]
        errbars(ax, x, s[ok], S.REGION_HEX[r], k, line=True)
    zero(ax, k); ax.set_xticks(range(len(FEATS)))
    if deck:
        ax.set_xticklabels(FEAT_2L)
    else:
        ax.set_xticklabels(FEAT_LB, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(FEATS) - 0.4); ax.set_xlabel("Covariate set"); ax.set_ylabel(YLAB)


def draw_d(ax, ct, k):
    """d TDD 3분위별 R1 − P1 ΔRMSE."""
    for i, r in enumerate(REG):
        s = ct[ct.region == r].sort_values("tdd_bin"); x = s.tdd_bin.values + (i - 1) * k["off"]
        errbars(ax, x, s, S.REGION_HEX[r], k, line=True)
    zero(ax, k); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["cold", "mid", "warm"]); ax.set_xlim(-0.6, 2.6); ax.set_xlabel("TDD tercile within region")


def draw_e(ax, st, k, deck=False):
    """e 모델 구조별 ΔRMSE(직접 ML · 물리 입력 ML · 물리 유사라벨 · 물리 + 잔차 ML − 재보정 Stefan). 유사라벨은 전량 라벨에서 이득 ≈ 0."""
    for i, r in enumerate(REG):
        s = st[st.region == r].set_index("method").reindex(STRUCT)
        errbars(ax, np.arange(len(STRUCT)) + (i - 1) * k["off"], s, S.REGION_HEX[r], k)
    zero(ax, k); ax.set_xticks(range(len(STRUCT)))
    if deck:
        ax.set_xticklabels([METH_2L[m] for m in STRUCT])
    else:
        ax.set_xticklabels([METH_EN[m] for m in STRUCT], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(STRUCT) - 0.4); ax.set_xlabel("Model structure"); ax.set_ylabel(YLAB)


def draw_f(ax, lr, k):
    """f 학습기별 ΔRMSE(채운 원 = 물리 + 잔차 ML, 빈 원 = 직접 ML). 마커 범례는 이 패널에만 둔다."""
    for i, r in enumerate(REG):
        for m, mfc, elw in (("R1", S.REGION_HEX[r], k["wlw"]), ("D0", "white", k["lw_d0"])):
            s = lr[(lr.region == r) & (lr.method == m)].set_index("learner").reindex(LEARN)
            ok = s.delta.notna().values; x = (np.arange(len(LEARN)) + (i - 1) * k["off_e"] + (k["sub_e"] if m == "D0" else -k["sub_e"]))[ok]
            errbars(ax, x, s[ok], S.REGION_HEX[r], k, mfc=mfc, lw=elw, zorder=4 if m == "R1" else 2)
    zero(ax, k); ax.set_xticks(range(len(LEARN))); ax.set_xticklabels(LEARN_LB); ax.set_xlim(-0.6, len(LEARN) - 0.4); ax.set_xlabel("Learner")


# ---------------------------------------------------------------- 그림(논문판·슬라이드 크롭)
def f4():
    df, sm, ct, lad, st, lr = load(); k = K_PAPER
    fig = S.figure(W, H)
    slide = S.SCALE != 1.0
    # ---- 1행: a 강제력 효과(지역 범례)
    wa = 45.0; ax = ax_mm(fig, L, ROW1, wa, HH)
    draw_a(ax, df, k); ax.xaxis.set_label_coords(0.5, -XL1 / HH); letter(ax, "a", wa, HH)
    # ---- b 검증 사다리 경사 차트(지역별 소패널). 오른쪽 끝은 그린 뒤 R 에 맞춘다
    xb0, wb, gb = 75.0, 25.0, 8.0
    rects = [[(xb0 + j * (wb + gb)) / W, ROW1 / H, wb / W, HH / H] for j in range(len(REG))]
    b_axes = draw_b(fig, sm, rects, k, xl_frac=-XL1 / HH)
    letter(b_axes[0], "b", wb, HH)
    for a in b_axes[1:]:
        S.register("b", a)
    # ---- 2행: c 공변량 사다리 · d TDD 3분위(공통 y 범위, 자르지 않음). 범주 폭 u 는 2·3행 공통
    ladf = lad[lad.featset.isin(FEATS)]
    u = (R - L - G23) / (UCAT["c"] + UCAT["d"]); wc, wd = UCAT["c"] * u, UCAT["d"] * u
    YL2 = yrange(ladf, ct)
    ax_c = ax_mm(fig, L, ROW2, wc, HH); draw_c(ax_c, ladf, k); letter(ax_c, "c", wc, HH)
    ax_d = ax_mm(fig, L + wc + G23, ROW2, wd, HH); draw_d(ax_d, ct, k); letter(ax_d, "d", wd, HH)
    for a in (ax_c, ax_d):
        a.set_ylim(*YL2); a.yaxis.set_major_locator(MultipleLocator(4)); a.xaxis.set_label_coords(0.5, -XL2 / HH)   # 축마다 새 locator(공유하면 마지막 축 범위를 쓴다)
    # ---- 3행: e 모델 구조 · f 학습기(공통 y 범위, 자르지 않음). 마커 범례는 f 위
    we, wf = UCAT["e"] * u, (R - L - G23) - UCAT["e"] * u
    YL3 = yrange(st, lr)
    ax_e = ax_mm(fig, L, ROW3, we, HH); draw_e(ax_e, st, k); letter(ax_e, "e", we, HH)
    ax_f = ax_mm(fig, L + we + G23, ROW3, wf, HH); draw_f(ax_f, lr, k); letter(ax_f, "f", wf, HH); marker_legend(ax_f, k)
    for a in (ax_e, ax_f):
        a.set_ylim(*YL3); a.yaxis.set_major_locator(MultipleLocator(4)); a.xaxis.set_label_coords(0.5, -XL3 / HH)
    if slide:   # 슬라이드 단독 크롭용 단위 표기
        ax_d.set_ylabel("ΔRMSE (cm)"); ax_f.set_ylabel("ΔRMSE (cm)")
    S.register("cd", ax_c, ax_d); S.register("ef", ax_e, ax_f)
    # ---- 1행 오른쪽 끝(b 의 직접 라벨 끝)을 측정해 R 에 맞춘다
    fig.canvas.draw(); rd = fig.canvas.get_renderer(); kk = 25.4 / fig.dpi
    x1 = max([a.get_window_extent(rd).x1 for a in b_axes] + [t.get_window_extent(rd).x1 for a in b_axes for t in a.texts]) * kk
    shift = (R - x1) / W
    for a in b_axes:
        p = a.get_position(); a.set_position([p.x0 + shift, p.y0, p.width, p.height])
    if slide:   # 슬라이드 크롭(b 단독)에서 눈금 라벨과 축 라벨 사이가 비지 않도록 b 의 x 라벨을 가장 낮은 눈금 라벨 아래 GAP_B 에 둔다(논문판은 a 의 'Method' 와 같은 높이)
        fig.canvas.draw(); rd = fig.canvas.get_renderer()
        y0 = min(t.get_window_extent(rd).y0 for a in b_axes for t in a.get_xticklabels() if t.get_text()) * kk
        b_axes[1].xaxis.set_label_coords(0.5, (y0 - GAP_B - ROW1) / HH)
        print(f"slide: b x-label top at {y0 - GAP_B:.1f} mm (lowest tick label bottom {y0:.1f} mm)")
    fig.canvas.draw()
    m = margins_mm(fig); print("margins mm:", {q: round(v, 2) for q, v in m.items()}, "| row1 shift mm:", round(shift * W, 2),
                               "| b x0 mm:", round(b_axes[0].get_position().x0 * W, 2), "| u mm:", round(u, 2),
                               "| w c/d/e/f mm:", [round(v, 1) for v in (wc, wd, we, wf)],
                               "| y range c–d:", [round(v, 2) for v in YL2], "e–f:", [round(v, 2) for v in YL3])
    return S.export(fig, "F4_results_core", outdir=OUT) if slide else export_paper(fig, "F4_results_core", OUT)


# ---------------------------------------------------------------- 덱판(덱 본문 크기 PNG, 12.0 × 4.65 in)
def sgn(v, nd=1):
    """부호 붙인 수(U+2212 빼기). 반올림 뒤 0 이면 부호 없음."""
    t = f"{abs(v):.{nd}f}"
    return t if float(t) == 0 else ("−" if v < 0 else "+") + t


def product_rmse():
    """R02 제품 기준선: products_eval.csv(mask 0) 의 'year-matched' 정의(그 해 제품 값이 있는 라벨 셀-연도) RMSE. {지역: [(키, 라벨, RMSE)]}."""
    pe = pd.read_csv(RUNS / "core_v6" / "products_eval.csv")
    pe = pe[(pe.mask_km == 0) & (pe.definition == "year-matched")]
    out = {}
    for r in REG:
        q = pe[pe.region == r].set_index("series")
        out[r] = [(key, lab, float(q.loc[key, "rmse"])) for key, lab in PROD_R02]
        print(f"R02 products {r}: " + ", ".join(f"{lab} {float(q.loc[key, 'rmse']):.1f} cm (n {int(q.loc[key, 'n']):,})" for key, lab in PROD_R02))
    return out


def top_legend(fig, handles, labels, x_in, k, ncol=None, y_in=0.04):
    """그림 위쪽 범례 한 줄(데이터 영역 밖): 왼쪽 위 모서리를 (x_in, 캔버스 위 − y_in) 에 둔다."""
    fw, fh = fig.get_size_inches()
    return fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x_in / fw, 1 - y_in / fh), ncol=ncol or len(handles), frameon=False,
                      fontsize=k["fs_leg"], handlelength=1.6, handletextpad=0.5, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)


def deck_r01(dfc, k):
    """R01: 지역별 패널. 방법마다 평년 기후 입력(빈 원, 왼쪽)과 관측 연도 기후 입력(채운 원, 오른쪽)의 공간 블록 RMSE 를 가는 선으로 잇는다.
    쌍 아래 Δ(관측 연도 − 평년)와 95 % 블록 부트스트랩 구간(deltas_forcing.csv). 세 패널의 y 폭(cm)을 같게 둔다."""
    smb = sel(pd.read_csv(RUNS / "core_v6" / "summary_metrics.csv"), scheme=MAIN["scheme"], featset=MAIN["featset"])
    vals = {(r, f, m): float(method_rows(smb[(smb.region == r) & (smb.forcing == f)], m).rmse.iloc[0])
            for r in REG for f in ("clim", "year_obs") for m in R01_METH}
    fig = deck_figure(HD); fw, fh = fig.get_size_inches()
    x0, gap, xr, yb, top = 1.02, 0.74, 0.06, 1.12, 0.80
    w = (DECK_W - x0 - xr - 2 * gap) / len(REG); h = HD - yb - top
    rng = {r: (min(v for (rr, _, _), v in vals.items() if rr == r), max(v for (rr, _, _), v in vals.items() if rr == r)) for r in REG}
    span = 2.0 * np.ceil(1.15 * max(hi - lo for lo, hi in rng.values()) / 2.0)             # 공통 y 폭(cm, 2 의 배수)
    DX, MS1, MEW1, DT = 0.15, 9.0, 1.8, 0.60                                               # 쌍 x 오프셋, 마커 지름·테 굵기(pt), Δ 글자 위치(축 아래 in)
    gz = S.NEUTRAL["text_aux"]; axes = []
    for j, r in enumerate(REG):
        ax = ax_in(fig, x0 + j * (w + gap), yb, w, h); axes.append(ax)
        lo, hi = rng[r]; c = 0.5 * (lo + hi); y_lo = np.floor((c - span / 2) * 2) / 2; ax.set_ylim(y_lo, y_lo + span)
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        for i, m in enumerate(R01_METH):
            col = S.METHOD_HEX[m]; a_, b_ = vals[(r, "clim", m)], vals[(r, "year_obs", m)]
            ax.plot([i - DX, i + DX], [a_, b_], "-", color=col, lw=k["lw_thin"], zorder=2, solid_capstyle="butt")
            ax.plot([i - DX], [a_], "o", ms=MS1, mfc="white", mec=col, mew=MEW1, lw=0, zorder=3)
            ax.plot([i + DX], [b_], "o", ms=MS1, mfc=col, mec=col, mew=MEW1, lw=0, zorder=3)
            q = method_rows(dfc[dfc.region == r], m).iloc[0]
            assert abs((b_ - a_) - float(q.delta)) < 1e-6, (r, m, b_ - a_, q.delta)      # summary_metrics 와 deltas_forcing 의 Δ 일치
            ax.text(i, -DT / h, f"Δ {sgn(q.delta)}\n[{sgn(q.lo)}, {sgn(q.hi)}]", transform=tr, ha="center", va="top", fontsize=k["fs_num"], color=gz,
                    linespacing=1.15)
        ax.set_xticks(range(len(R01_METH))); ax.set_xticklabels([METH_2L[m] for m in R01_METH]); ax.set_xlim(-0.5, len(R01_METH) - 0.5)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.text(0.0, 1.0 + 0.08 / h, S.REGION_EN[r], transform=ax.transAxes, ha="left", va="bottom", fontsize=k["fs_reg"], color=S.REGION_HEX[r])
        if j == 0:
            ax.set_ylabel("RMSE, spatial block (cm)")
    hs = [Line2D([], [], marker="o", ms=MS1, mfc="white", mec=gz, mew=MEW1, lw=0), Line2D([], [], marker="o", ms=MS1, mfc=gz, mec=gz, mew=MEW1, lw=0)]
    top_legend(fig, hs, ["Mean climate input (2011–2024)", "Year-specific climate input (to observation day)"], x0 - 0.10, k)
    print("R01 RMSE (cm) mean → year-specific:", "; ".join(f"{r} {m} {vals[(r, 'clim', m)]:.2f}→{vals[(r, 'year_obs', m)]:.2f}" for r in REG for m in R01_METH),
          f"| common y span {span:.0f} cm")
    return save_deck(fig, "R01", HD)


def deck_r02(sm, k):
    """R02: b 세 소패널 + 제품 기준선. 소패널마다 오른쪽에 제품 라벨 열(lab_w), 다음 소패널 눈금 라벨 몫(T)을 둔다. 방법은 위쪽 범례 한 줄."""
    prods = product_rmse()
    fig = deck_figure(HD); fw, fh = fig.get_size_inches(); rd = fig.canvas.get_renderer()
    def tw(txt):
        t = fig.text(0, 0, txt, fontsize=k["fs_num"]); fig.canvas.draw(); wv = t.get_window_extent(rd).width / fig.dpi; t.remove(); return wv
    lab_w = max(tw(lab) for _, lab in PROD_R02) + k["lab_dx_pt"] / 72
    x0, T, xr, yb, top = 0.92, 0.68, 0.04, DY0, 0.78
    w = (DECK_W - x0 - xr - len(REG) * lab_w - (len(REG) - 1) * T) / len(REG); h = HD - yb - top
    rects = [rect_in(fig, x0 + j * (w + lab_w + T), yb, w, h) for j in range(len(REG))]
    axes = draw_b(fig, sm, rects, k, deck=True, xl_frac=-DXL / h, prods=prods, method_labels=False)
    order = ["P0", "P1", "R1", "D0"]
    hs = [Line2D([], [], color=S.METHOD_HEX[m], ls=S.METHOD_LS.get(m, "-"), lw=k["lw_thin"] if m == "P0" else k["lw"], marker="o",
                 ms=k["ms_b_thin"] if m == "P0" else k["ms_b"], mew=0) for m in order]
    top_legend(fig, hs, [METH_EN[m] for m in order], x0 - 0.10, k)
    print(f"R02 layout: panel width {w:.2f} in, product label column {lab_w:.2f} in")
    return save_deck(fig, "R02", HD)


def deck_r03(ct, lad, st, lr, k):
    """R03 c·d, R03b e·f(두 패널). 범주 폭 u 공통. 지역 범례는 왼쪽 패널 위, 마커 범례는 f 위. y 범위는 슬라이드마다 모든 구간이 들어가게."""
    out = []; x0, xr, gap, top = 1.12, 0.10, 0.76, 0.42; hh = HD - DY0 - top
    ladf = lad[lad.featset.isin(FEATS)]
    u = (DECK_W - x0 - xr - gap) / (UCAT["c"] + UCAT["d"])
    for slide, (p1, p2), (d1, d2), yl in (("R03", ("c", "d"), (lambda a: draw_c(a, ladf, k, deck=True), lambda a: draw_d(a, ct, k)), yrange(ladf, ct)),
                                          ("R03b", ("e", "f"), (lambda a: draw_e(a, st, k, deck=True), lambda a: draw_f(a, lr, k)), yrange(st, lr))):
        fig = deck_figure(HD); w1 = UCAT[p1] * u; w2 = DECK_W - x0 - xr - gap - w1
        ax1 = ax_in(fig, x0, DY0, w1, hh); d1(ax1); region_legend(ax1, k, deck=True)
        ax2 = ax_in(fig, x0 + w1 + gap, DY0, w2, hh); d2(ax2)
        if p2 == "f":
            marker_legend(ax2, k, deck=True)
        for a, q in zip((ax1, ax2), "ab"):
            a.set_ylim(*yl); a.yaxis.set_major_locator(MultipleLocator(4)); a.xaxis.set_label_coords(0.5, -DXL / hh); deck_letter(a, q)
        out.append(save_deck(fig, slide, HD))
    return out


def draw_c_h(ax, ladf, k):
    """c 가로판(덱 R03m a): 행 = 공변량 집합(위 → 아래로 누적), x = ΔRMSE(Physics + residual ML − Stefan (recalibrated)).
    지역마다 점 · 95 % 구간(가로 막대) · 행 사이 연결선. 지역 순서는 행 안에서 위 → 아래 = 알래스카 · 레나 · 캐나다(범례 순서)."""
    n = len(FEATS); offy = {"Alaska": -0.22, "Lena": 0.0, "Canada": 0.22}
    for r in REG:
        q = ladf[ladf.region == r].set_index("featset").reindex(FEATS); ok = q.delta.notna().values
        y = (np.arange(n) + offy[r])[ok]; c = S.REGION_HEX[r]
        ax.hlines(y, q.lo.values[ok], q.hi.values[ok], color=c, lw=k["wlw"], zorder=2)
        ax.plot(q.delta.values[ok], y, "-", color=c, lw=k["lw_line"], alpha=0.9, zorder=1)
        ax.plot(q.delta.values[ok], y, "o", ms=k["ms"], color=c, mew=0, lw=0, zorder=3)
    ax.axvline(0, color=S.NEUTRAL["zero_line"], lw=k["zlw"], zorder=0)
    ax.set_yticks(range(n)); ax.set_yticklabels(FEAT_LB); ax.set_ylim(n - 0.5, -0.5); ax.tick_params(axis="y", length=0)
    lo, hi = yrange(ladf); ax.set_xlim(lo, hi); ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.set_xlabel("ΔRMSE (cm)\nvs Stefan (recalibrated)")


def deck_r03m(lad, k):
    """R03m: a 공변량 사다리 가로판(왼쪽) + b 공변량별 3분위 이득 열지도(오른쪽, 자료 부족 칸 빗금) + 아래 열쇠 줄 + 오른쪽 색 막대."""
    deck_rc()
    spec = importlib.util.spec_from_file_location("v9_f4e_gain_conditions", Path(__file__).with_name("v9_f4e_gain_conditions.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); t = m.load_table()
    ladf = lad[lad.featset.isin(FEATS)]; fs = _FS["tick_pt"]; fsv = _FS["direct_label_pt"]
    fig = deck_figure(HD); fw, fh = fig.get_size_inches(); rd = fig.canvas.get_renderer()
    def tw(txt, f):
        z = fig.text(0, 0, txt, fontsize=f); fig.canvas.draw(); w = z.get_window_extent(rd).width / fig.dpi; z.remove(); return w
    yb, top, gap_ab = 0.95, 0.45, 0.42                                   # 축 아래(눈금 + 두 줄 x 라벨, 3분위 이름 + 열쇠 줄), 위 띠, a·b 사이 잉크 간격(in)
    # ---- a 공변량 사다리(가로)
    xa = 0.12 + max(tw(l, fs) for l in FEAT_LB) + 0.10; wa = 2.70; ha = HD - yb - top
    ax_a = ax_in(fig, xa, yb, wa, ha); draw_c_h(ax_a, ladf, k)
    # ---- b 열지도: 행 이름 열(row_w) 오른쪽에 지역 블록 셋, 그 오른쪽 색 막대
    row_w = max(tw(lab, fs) for _, _, lab, _, _ in m.COVS)
    xb0 = xa + wa + gap_ab + row_w + 0.12; gap_reg, gap_grp, cb_room = 0.18, 0.08, 1.00
    w_cell = (DECK_W - xb0 - cb_room - 2 * gap_reg) / 9; h_cell = (ha - 3 * gap_grp) / len(m.COVS)
    sh = t[t.shown & (t.covariate != "all")]; vmax = float(np.ceil(np.nanmax(np.abs(sh.delta.values))))
    plt.rcParams["hatch.linewidth"] = 1.0
    axes_b, norm, H = m.draw_map(fig, t, xb0, yb, w_cell, h_cell, gap_grp, gap_reg, fs_val=fsv, fs_tick=fs, fs_reg=_FS["axis_label_pt"], lw_sig=2.4,
                                 vmax=vmax, na_style="hatch", fs_row=fs, tercile_header="Tercile")
    xr = xb0 + 9 * w_cell + 2 * gap_reg
    cax = ax_in(fig, xr + 0.22, yb, 0.14, H)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=m.cmap_diff()), cax=cax)
    st_ = 3 if vmax >= 6 else 2; tk = np.arange(-np.floor(vmax / st_) * st_, vmax + 1e-9, st_)
    cb.set_ticks(tk); cb.set_ticklabels([("−" if v < 0 else "+" if v > 0 else "") + f"{abs(v):.0f}" for v in tk]); cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=fs, length=_LN["tick_len_pt"], width=_LN["tick_pt"]); cb.outline.set_linewidth(_LN["axis_pt"])
    cb.set_label("ΔRMSE (cm)", fontsize=_FS["axis_label_pt"], labelpad=4)
    # ---- 열쇠 줄(열지도 아래): 외곽선 칸 견본 + 글자, 빗금 칸 견본 + 글자
    kx = xb0 - 0.12 - row_w; ky, kh, ks = 0.30, 0.22, 0.30
    l1, l2 = "95% CI excludes 0", "Insufficient data (< 30 cell-years or < 5 blocks)"
    kw_tot = ks + 0.08 + tw(l1, fs) + 0.36 + ks + 0.08 + tw(l2, fs)
    kax = ax_in(fig, kx, ky - kh / 2, kw_tot, kh); kax.set_xlim(0, kw_tot); kax.set_ylim(0, kh); kax.axis("off")
    from matplotlib.patches import Rectangle
    e = 2.4 / 72 / 2
    kax.add_patch(Rectangle((e, e), ks - 2 * e, kh - 2 * e, facecolor=m.cmap_diff()(norm(-0.45 * vmax)), edgecolor=S.NEUTRAL["text_slide"], lw=2.4, joinstyle="miter"))
    kax.text(ks + 0.08, kh / 2, l1, ha="left", va="center", fontsize=fs, color=S.NEUTRAL["text_slide"])
    x2 = ks + 0.08 + tw(l1, fs) + 0.36
    m.hatch_rect(kax, x2, 0.0, ks, kh); kax.add_patch(Rectangle((x2, 0.0), ks, kh, facecolor="none", edgecolor=S.NEUTRAL["coast"], lw=0.8))
    kax.text(x2 + ks + 0.08, kh / 2, l2, ha="left", va="center", fontsize=fs, color=S.NEUTRAL["text_slide"])
    # ---- 패널 문자 · a 지역 범례(위 띠)
    fig.text(0.10 / fw, 1 - 0.05 / fh, "a", ha="left", va="top", fontsize=_FS["panel_letter_pt"], fontweight="bold")
    fig.text(kx / fw, 1 - 0.05 / fh, "b", ha="left", va="top", fontsize=_FS["panel_letter_pt"], fontweight="bold")
    fig.legend(region_handles(k), [S.REGION_EN[r] for r in REG], loc="upper left", bbox_to_anchor=(0.40 / fw, 1 - 0.03 / fh), ncol=3, frameon=False,
               fontsize=k["fs_leg"], handlelength=0.8, handletextpad=0.45, columnspacing=1.0, borderaxespad=0.0, borderpad=0.0)
    na = t[(t.covariate != "all") & ~t.shown]
    print(f"R03m layout: a {xa:.2f}–{xa + wa:.2f} in, b row labels from {kx:.2f} in, cells {w_cell:.2f} × {h_cell:.2f} in, map right {xr:.2f} in, vmax {vmax:.0f}"
          f" | hatched: " + ", ".join(f"{z.region} {z.label} {z.tercile_name} (n {z.n}, blocks {z.n_blocks})" for _, z in na.iterrows()))
    return save_deck(fig, "R03m", HD)


def deck_r03e():
    """R03e: 공변량별 3분위 이득 열지도. 계산·그림은 v9_f4e_gain_conditions.py(표가 있으면 읽고, 없으면 계산)."""
    spec = importlib.util.spec_from_file_location("v9_f4e_gain_conditions", Path(__file__).with_name("v9_f4e_gain_conditions.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    t = m.load_table(); m.summary(t)
    return m.deck(t)


DECK_SLIDES = ["R01", "R02", "R03", "R03b", "R03e", "R03m"]


def f4_deck(only=None):
    """R01 · R02 · R03 · R03b · R03e · R03m(only 로 고를 수 있다). 캔버스 12.0 × 4.65 in, 축이 캔버스를 채우고 여백은 축 라벨 몫만 둔다."""
    deck_rc(); df, sm, ct, lad, st, lr = load(); k = K_DECK; want = set(only or DECK_SLIDES); out = []
    assert want <= set(DECK_SLIDES), want - set(DECK_SLIDES)
    if "R01" in want:
        out.append(deck_r01(df, k))
    if "R02" in want:
        out.append(deck_r02(sm, k))
    if want & {"R03", "R03b"}:
        out += deck_r03(ct, lad, st, lr, k)          # R03 · R03b 는 한 함수에서 함께 낸다
    if "R03m" in want:
        out.append(deck_r03m(lad, k))
    if "R03e" in want:
        out.append(deck_r03e())
    return out


FIGS = {4: f4}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", type=int, default=4); ap.add_argument("--slide", action="store_true")
    ap.add_argument("--deck", action="store_true", help="덱 본문 크기(12 × 4.65 in, 300 dpi) PNG R01·R02·R03·R03b·R03e·R03m")
    ap.add_argument("--only", default="", help="--deck 와 함께: 낼 덱 쪽(쉼표, 예: R03m)"); a = ap.parse_args()
    S.setup(scale=1.69 if (a.slide or a.deck) else 1.0)
    for p in (f4_deck([q.strip() for q in a.only.split(",") if q.strip()] or None) if a.deck else FIGS[a.fig]()):
        print(p)
