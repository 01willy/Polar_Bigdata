# -*- coding: utf-8 -*-
"""v8_f9_label_ladder.py: F9 라벨 수 사다리(LL)와 새 지역 전이 + 소수 라벨(TL). 입력 runs/ladder_v6/{ladder_metrics,ladder_deltas}.csv.
논문판 180 × 118 mm(a LL ΔRMSE vs Stefan(재보정), b TL RMSE 대 대상 라벨 수), 슬라이드 크롭 a·b, 덱판 LL·TL(12.0 × 4.65 in, 3600 × 1395 px).
x 축은 라벨 수의 순서 범주(같은 간격): 25 · 50 · 100 · 250 · 500 · 1,000 · 2,500 · all(지역의 학습 겹 전량, 실제 수는 source_data). 지역마다 자료가
있는 범주만 둔다(캐나다는 학습 겹이 1,000 보다 작아 500 다음이 all). 옛 판은 'all' 을 지역별 최대 n × 2.5 의 로그 위치에 찍어 레나·캐나다의 'all' 점이
'2,500' 눈금 근처에 놓였다(10-15 v9 수정).
개정(10-15, 설계 지침 v9 항 19): 덱 그래프를 본문 크기로 확대(12 × 4.65 in), 범례는 자료 밖(그림 위 2줄), 지역 제목색 = style_v5.REGION_HEX(토큰),
  방법 이름 = 지침 1.1 표기(Direct ML · Physics pseudo-labels · Physics-input ML · Physics + residual ML · Stefan (source coeff.) · Stefan (recalibrated)),
  LL 의 0 선 = Stefan (recalibrated) 를 범례에 둔다(축 라벨은 'ΔRMSE (cm)'), 띠 = 95 % 블록 부트스트랩 구간(범례 '95 % interval').
실행: OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v8_f9_label_ladder.py [--slide] [--deck]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S  # noqa: E402
RUN = ROOT / "data" / "processed" / "v5" / "runs" / "ladder_v6"
OUT = ROOT / "outputs" / "figures" / "paper" / "v7"; DECK = OUT / "deck"; STEM = "F9_label_ladder"
REG = ["Alaska", "Lena", "Canada"]
LL_M = {"D0": ("Direct ML", S.METHOD_HEX["D0"], "--"), "D1": ("Physics pseudo-labels", S.METHOD_HEX["D1"], ":"),
        "F1": ("Physics-input ML", S.METHOD_HEX["F1"], "-."), "R1": ("Physics + residual ML", S.METHOD_HEX["R1"], "-")}
TL_M = {"P0": ("Stefan (source coeff.)", S.METHOD_HEX["P0"], "-"), "P1n": ("Stefan (recalibrated)", S.METHOD_HEX["P1"], "-"),
        "D0S": ("Direct ML", S.METHOD_HEX["D0"], "--"), "D1S": ("Physics pseudo-labels", S.METHOD_HEX["D1"], ":"),
        "R1S": ("Physics + residual ML", S.METHOD_HEX["R1"], "-"), "R1T": ("Physics + residual ML (target only)", S.METHOD_HEX["R1"], "--")}
REF_NAME = "Stefan (recalibrated)"          # LL 의 0 선(같은 n 의 재보정 Stefan)
BAND_NAME = "95 % interval"                 # LL 띠: 물리 + 잔차 ML 의 95 % 블록 부트스트랩 구간
BAND_ALPHA = 0.15
SRC: dict = {}


def load():
    """LL 에서 일부 겹에만 있는 n(학습 겹 크기보다 큰 n)은 뺀다: n_test 가 'all' 의 95 % 미만인 행."""
    m = pd.read_csv(RUN / "ladder_metrics.csv"); d = pd.read_csv(RUN / "ladder_deltas.csv")
    for t in (m, d):
        t["n_label"] = t.n_label.astype(str)
    full = m[(m.exp == "LL") & (m.n_label == "all")].groupby("region").n_test.max()
    for t in (m, d):
        ll = t.exp == "LL"; keep = ~ll | (t.n_test >= 0.95 * t.region.map(full))
        t.drop(index=t.index[~keep], inplace=True)
    return m, d


def cats(labels):
    """라벨 수 범주(순서): 숫자 오름차순 + 'all' 맨 오른쪽. 반환 (범주 목록, 범주 → x 위치)."""
    num = sorted({int(v) for v in labels if v != "all"}); out = [str(v) for v in num] + (["all"] if "all" in set(labels) else [])
    return out, {c: i for i, c in enumerate(out)}


def cat_label(c):
    return "all" if c == "all" else f"{int(c):,}"


def draw_ll(axes, d, M):
    """지역별 패널: ΔRMSE(방법 − Stefan (recalibrated), 같은 n) 대 라벨 수(순서 범주), 물리 + 잔차 ML 의 95 % 블록 구간 띠."""
    q = d[(d.exp == "LL") & (d.ref == "P1")]; SRC["LL"] = {}
    for ax, r in zip(axes, REG):
        qr = q[q.region == r]; cs, pos = cats(qr.n_label.unique())
        for m, (name, col, ls) in LL_M.items():
            s = qr[qr.method == m].copy(); s["x"] = s.n_label.map(pos); s = s.sort_values("x")
            if not len(s):
                continue
            if m == "R1":
                ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=BAND_ALPHA, lw=0, zorder=1)
            ax.plot(s.x, s.delta, ls=ls, lw=M["lw"], color=col, marker="o", ms=M["ms"], zorder=3 + (m == "R1"))
        ax.axhline(0, color=S.NEUTRAL["zero_line"], lw=M["lw_ref"], zorder=2)
        ax.set_xticks(range(len(cs))); ax.set_xticklabels([cat_label(c) for c in cs]); ax.set_xlim(-0.4, len(cs) - 0.6)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_title(S.REGION_EN[r], loc="left", fontsize=M["title"], color=S.REGION_HEX[r], pad=M["title_pad"])
        n_all = qr[qr.n_label == "all"].n.max()
        SRC["LL"][r] = dict(categories=cs, n_all_training_labels=int(n_all) if np.isfinite(n_all) else None,
                            delta_cm={LL_M[m][0]: {c: round(float(v), 3) for c, v in qr[qr.method == m].set_index("n_label").delta.items()} for m in LL_M})
    axes[0].set_ylabel("ΔRMSE (cm)", fontsize=M["lab"])
    axes[len(axes) // 2].set_xlabel("Training labels (cell-years)", fontsize=M["lab"])
    hs = [Line2D([], [], color=S.NEUTRAL["zero_line"], lw=M["lw_ref"])]
    hs += [Line2D([], [], ls=ls, lw=M["lw"], color=col, marker="o", ms=M["ms"]) for _, (name, col, ls) in LL_M.items()]
    hs += [Patch(facecolor=LL_M["R1"][1], alpha=BAND_ALPHA, lw=0)]
    return hs, [REF_NAME] + [v[0] for v in LL_M.values()] + [BAND_NAME]


def draw_tl(axes, m, M):
    """지역별 패널: 지역 홀드아웃 RMSE 대 대상 지역 라벨 수 n(0 · 25 · 50 · 100 · 250 · 500, 순서 범주)."""
    q = m[m.exp == "TL"]; SRC["TL"] = {}
    for ax, r in zip(axes, REG):
        qr = q[q.region == r]; cs, pos = cats(qr.n_label.unique())
        for meth, (name, col, ls) in TL_M.items():
            s = qr[qr.method == meth].copy(); s["x"] = s.n_label.map(pos); s = s.sort_values("x")
            if not len(s):
                continue
            ax.plot(s.x, s.rmse, ls=ls, lw=M["lw"], color=col, marker="o", ms=M["ms"], zorder=3 + meth.startswith("R1"))
        ax.set_xticks(range(len(cs))); ax.set_xticklabels([cat_label(c) for c in cs]); ax.set_xlim(-0.4, len(cs) - 0.6)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_title(S.REGION_EN[r], loc="left", fontsize=M["title"], color=S.REGION_HEX[r], pad=M["title_pad"])
        SRC["TL"][r] = {TL_M[k][0]: {c: round(float(v), 3) for c, v in qr[qr.method == k].set_index("n_label").rmse.items()} for k in TL_M}
    axes[0].set_ylabel("RMSE (cm)", fontsize=M["lab"])
    axes[len(axes) // 2].set_xlabel("Target-region labels (cell-years)", fontsize=M["lab"])
    hs = [Line2D([], [], ls=ls, lw=M["lw"], color=col, marker="o", ms=M["ms"]) for _, (name, col, ls) in TL_M.items()]
    return hs, [v[0] for v in TL_M.values()]


def _legend(fig, hs, ls, M, y):
    """그림 위 범례(자료 밖), 3열 2줄, 가운데 맞춤."""
    return fig.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, y), ncol=3, fontsize=M["leg"], frameon=False, handlelength=2.2,
                      handletextpad=0.6, columnspacing=1.8, labelspacing=0.45, borderaxespad=0.0)


def paper(slide=False):
    S.setup(scale=1.69 if slide else 1.0)
    m, d = load(); M = dict(lw=1.0, lw_ref=0.6, ms=2.6, tick=6.5, lab=7, leg=6.0, title=7, title_pad=3)
    W, H = S.W2, 118.0; fig = S.figure(W, H)
    mm = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    x0, ga, xr = 12.0, 14.0, 3.5        # 오른쪽 끝 = 180 − 3.5 mm(마지막 눈금 라벨 ≈ 1.5 mm 가 넘어도 ≤ 180 mm)
    def row(key, y):                    # 패널 폭 = 범주 수에 비례(덱과 같은 규칙: 알래스카 LL 의 '1,000' · '2,500' 라벨이 붙지 않게)
        n = [len(cats((d[(d.exp == "LL") & (d.ref == "P1") & (d.region == r)] if key == "LL" else m[(m.exp == "TL") & (m.region == r)]).n_label.unique())[0])
             for r in REG]
        u = (W - x0 - 2 * ga - xr) / sum(k - 0.2 for k in n); xs = np.cumsum([x0] + [(k - 0.2) * u + ga for k in n[:-1]])
        return [mm(x, y, (k - 0.2) * u, 34.0) for x, k in zip(xs, n)]
    axa = row("LL", 70.0)
    hs, ls = draw_ll(axa, d, M); S.panel_letter(axa[0], "a", x=-0.22); S.register("a", *axa[1:])
    la = axa[1].legend(hs, ls, loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=M["leg"], frameon=False, handlelength=2.0,
                       handletextpad=0.5, columnspacing=1.4, labelspacing=0.35, borderaxespad=0.0)
    axb = row("TL", 11.0)
    hs, ls = draw_tl(axb, m, M); S.panel_letter(axb[0], "b", x=-0.22); S.register("b", *axb[1:])
    axb[1].legend(hs, ls, loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=M["leg"], frameon=False, handlelength=2.0,
                  handletextpad=0.5, columnspacing=1.4, labelspacing=0.35, borderaxespad=0.0)
    for ax in axa + axb:
        ax.tick_params(labelsize=M["tick"])
    if not slide:
        (OUT / "source_data").mkdir(parents=True, exist_ok=True)
        (OUT / "source_data" / f"{STEM}.json").write_text(json.dumps(SRC, indent=1, ensure_ascii=False))
    return S.export(fig, STEM, outdir=OUT)


def deck():
    """덱판 LL·TL: 12.0 × 4.65 in, 300 dpi(3600 × 1395 px), 흰 바탕 RGB. 글자: 축 라벨·지역 제목 14, 눈금·범례 12 pt(토큰 font.slide)."""
    from PIL import Image
    S.setup(scale=1.69); Fs = S.TOK["font"]["slide"]; Ls = S.TOK["lines"]["slide"]
    plt.rcParams.update({"font.size": Fs["tick_pt"], "axes.labelsize": Fs["axis_label_pt"], "xtick.labelsize": Fs["tick_pt"], "ytick.labelsize": Fs["tick_pt"],
                         "legend.fontsize": Fs["tick_pt"], "axes.linewidth": Ls["axis_pt"], "xtick.major.width": Ls["tick_pt"], "ytick.major.width": Ls["tick_pt"],
                         "xtick.major.size": Ls["tick_len_pt"], "ytick.major.size": Ls["tick_len_pt"], "xtick.major.pad": 4.0, "ytick.major.pad": 4.0,
                         "axes.labelpad": 6.0})
    m, d = load(); M = dict(lw=Ls["data_pt"], lw_ref=Ls["reference_pt"], ms=6.5, tick=Fs["tick_pt"], lab=Fs["axis_label_pt"], leg=Fs["tick_pt"],
                            title=Fs["axis_label_pt"], title_pad=5)
    DECK.mkdir(parents=True, exist_ok=True); out = []
    Wi, Hi = 12.0, 4.65; x0, gp, xr, y0, hp = 0.78, 0.73, 0.22, 0.60, 3.06       # 축(in): 왼쪽 여백 = y 라벨·눈금, 간격 = 다음 패널 눈금 + ≥ 0.35 in 여백
    ncat = {"LL": [len(cats(d[(d.exp == "LL") & (d.ref == "P1") & (d.region == r)].n_label.unique())[0]) for r in REG],
            "TL": [len(cats(m[(m.exp == "TL") & (m.region == r)].n_label.unique())[0]) for r in REG]}
    for key, fn, arg in (("LL", draw_ll, d), ("TL", draw_tl, m)):
        fig = plt.figure(figsize=(Wi, Hi)); fig.patch.set_facecolor("white")
        u = (Wi - x0 - 2 * gp - xr) / sum(n - 0.2 for n in ncat[key])               # 범주 간격(in): 패널 폭 = 범주 수에 비례(알래스카 8 · 레나 7 · 캐나다 6)
        xs = np.cumsum([x0] + [(n - 0.2) * u + gp for n in ncat[key][:-1]])
        axes = [fig.add_axes([x / Wi, y0 / Hi, (n - 0.2) * u / Wi, hp / Hi]) for x, n in zip(xs, ncat[key])]
        hs, ls = fn(axes, arg, M)
        _legend(fig, hs, ls, M, y=1.0 - 0.03 / Hi)
        p = DECK / f"{STEM}__deck_{key}.png"; fig.savefig(p, dpi=300, facecolor="white"); plt.close(fig)
        with Image.open(p) as im:
            rgb = im.convert("RGB")
        rgb.save(p, dpi=(300, 300)); assert rgb.size == (3600, 1395), rgb.size
        print(f"[deck {key}] {p} {rgb.size}"); out.append(p)
    (OUT / "source_data").mkdir(parents=True, exist_ok=True)
    (OUT / "source_data" / f"{STEM}__deck.json").write_text(json.dumps(SRC, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--slide", action="store_true"); ap.add_argument("--deck", action="store_true"); a = ap.parse_args()
    if a.deck:
        deck()
    else:
        paper(slide=a.slide); print("[F9] done")
