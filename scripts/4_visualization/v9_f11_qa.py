# -*- coding: utf-8 -*-
"""v9_f11_qa.py: 질의 답변서 근거 그림 F11(2026-10-15). 논문판 180 × 128 mm, PDF·PNG.
(a) Wei 2026 학습 지점(보충 표 ALT_variables.xlsx)과 세 지역 라벨 셀(극 지도)
(b) 라벨 셀에서 가장 가까운 Wei 학습 지점까지의 거리(지역별 누적 분포, 25 km 선)
(c) 레나 델타 같은 셀-연도 RMSE: Wei 2026 · Liu 2024 대 레나 라벨 없이 학습한 우리 방법(대상 라벨 0)과 대상 라벨 25개
(d) 알래스카 2014 지도 수준 제품 − 우리 지도 평균 차이: 위도대 × 고도(600 m) 구분
입력: data/processed/v9_qa/{transfer_vs_products.csv, map_band_alaska2014_by_elev.csv}, xg_leak_cells_v1.csv, labels_cell_year_v5.csv
실행: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/v9_f11_qa.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.style_v5 as S  # noqa: E402
Q = ROOT / "data" / "processed" / "v9_qa"; OUT = ROOT / "outputs" / "figures" / "paper" / "v7"; REG = ["Alaska", "Lena", "Canada"]
C_W, C_L = S.PRODUCT_HEX["wei"], S.PRODUCT_HEX["liu"]


def main():
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    S.setup(scale=1.0); PC = ccrs.PlateCarree(); fs = 6.5
    fig = S.figure(S.W2, 132)
    # (a) 지도
    ax = fig.add_axes([0.0, 0.47, 0.36, 0.52], projection=ccrs.NorthPolarStereo(central_longitude=-100))
    ax.set_extent([-180, 180, 50, 90], PC); ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2f2f2", edgecolor="none")
    ax.coastlines("50m", linewidth=0.3, color="#9a9a9a"); ax.gridlines(crs=PC, linewidth=0.3, color="#d0d0d0", xlocs=range(-180, 180, 60), ylocs=[60, 70, 80])
    lab = pd.read_csv(ROOT / "data" / "processed" / "v5" / "labels_cell_year_v5.csv", usecols=["region", "lat", "lon"]).drop_duplicates()
    for r in REG:
        q = lab[lab.region == r]; ax.scatter(q.lon, q.lat, s=2, color=S.REGION_HEX[r], transform=PC, lw=0, zorder=3)
    w = pd.read_excel(ROOT / "data" / "raw" / "qtp_alt_open_2026-09-29" / "essd-2026-447_supplement" / "ALT_variables.xlsx", usecols=["LAT", "LONG"]).drop_duplicates()
    ax.scatter(w.LONG, w.LAT, s=7, marker="x", color="#222222", lw=0.6, transform=PC, zorder=4)
    hs = [Line2D([], [], ls="none", marker="x", color="#222222", ms=4, label="Wei 2026 sites")] + \
         [Line2D([], [], ls="none", marker="o", ms=3, mfc=S.REGION_HEX[r], mec="none", label=S.REGION_EN[r]) for r in REG]
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=fs - 0.5, ncol=1, frameon=False, handletextpad=0.3)
    S.panel_letter(ax, "a", x=0.0, y=1.0)
    # (b) 거리 누적 분포
    axb = fig.add_axes([0.555, 0.60, 0.14, 0.36])
    lk = pd.read_csv(ROOT / "data" / "processed" / "xbatch" / "XG_product_comparison" / "xg_leak_cells_v1.csv")
    lk = lk[(lk["product"] == "wei") & lk.target.str.endswith("|r")]; lk["region"] = lk.target.str.split("|").str[0]
    for r in REG:
        d = np.sort(lk[lk.region == r].dist_km.values); axb.plot(np.maximum(d, 0.5), np.arange(1, len(d) + 1) / len(d), color=S.REGION_HEX[r], lw=1.0, label=S.REGION_EN[r])
    axb.axvline(25, color="#9a9a9a", lw=0.6, ls=(0, (3, 2))); axb.set_xscale("log"); axb.set_xlim(0.5, 2000)
    axb.set_xticks([1, 25, 100, 1000]); axb.set_xticklabels(["1", "25", "100", "1,000"]); axb.minorticks_off()
    axb.set_xlabel("Distance to nearest Wei 2026\ntraining site (km)", fontsize=fs); axb.set_ylabel("Share of label cells", fontsize=fs)
    S.panel_letter(axb, "b", x=-0.30)
    # (c) 레나 델타 무라벨 전이 대 제품
    axc = fig.add_axes([0.85, 0.60, 0.14, 0.36])
    t = pd.read_csv(Q / "transfer_vs_products.csv"); t = t[t.region == "Lena"]
    rows = [("Wei 2026", None, "wei", 0, C_W, "D"), ("Stefan, source coeff.", "P0", "wei", 0, S.METHOD_HEX["P0"], "s"),
            ("Direct ML, source only", "D0S", "wei", 0, S.METHOD_HEX["D0"], "o"), ("Physics + residual,\nsource only", "R1S", "wei", 0, S.METHOD_HEX["R1"], "o"),
            ("Physics + residual,\n+25 Lena labels", "R1S", "wei", 25, S.METHOD_HEX["R1"], "o")]
    for i, (name, m, prod, n, col, mk) in enumerate(rows):
        q = t[(t["product"] == prod) & (t.n_label == n) & (t.method == ("P0" if m is None else m))].iloc[0]
        v = q.rmse_product if m is None else q.rmse_method; y = len(rows) - 1 - i
        axc.plot([v], [y], ls="none", marker=mk, ms=4, mfc=col if n == 0 else "white", mec=col, mew=0.9)
    axc.set_yticks(range(len(rows))); axc.set_yticklabels([r[0] for r in rows][::-1], fontsize=fs - 0.5); axc.tick_params(axis="y", length=0)
    axc.set_xlabel("RMSE, Lena Delta,\nsame cell-years (cm)", fontsize=fs); axc.spines["left"].set_visible(False); axc.grid(axis="x", color="#e6e6e6", lw=0.5)
    axc.set_xlim(15, 45); S.panel_letter(axc, "c", x=-1.05)
    # (d) 알래스카 2014 지도 차이
    axd = fig.add_axes([0.08, 0.07, 0.90, 0.31])
    e = pd.read_csv(Q / "map_band_alaska2014_by_elev.csv"); bands = list(dict.fromkeys(e.band))
    lab_b = {"(59, 64]": "59–64°N", "(64, 66]": "64–66°N", "(66, 68]": "66–68°N", "(68, 70]": "68–70°N", "(70, 72]": "70–72°N"}
    for j, (mtn, hatch) in enumerate((("elev<=600m", None), ("elev>600m", "////"))):
        for k, (col, prod) in enumerate(((C_W, "wei_minus"), (C_L, "liu_minus"))):
            q = e[e.mtn == mtn].set_index("band").reindex(bands)
            x = np.arange(len(bands)) + (j * 2 + k - 1.5) * 0.18
            axd.bar(x, q[prod].values, width=0.17, color=col if hatch is None else "white", edgecolor=col, hatch=hatch, lw=0.6)
    axd.axhline(0, color="#9a9a9a", lw=0.6); axd.set_xticks(range(len(bands))); axd.set_xticklabels([lab_b.get(b, b) for b in bands], fontsize=fs)
    axd.set_ylabel("Product − our map,\nAlaska 2014 (cm)", fontsize=fs)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=C_W, ec=C_W, label="Wei 2026, elevation ≤ 600 m"), plt.Rectangle((0, 0), 1, 1, fc=C_L, ec=C_L, label="Liu 2024, elevation ≤ 600 m"),
          plt.Rectangle((0, 0), 1, 1, fc="white", ec=C_W, hatch="////", label="Wei 2026, > 600 m"), plt.Rectangle((0, 0), 1, 1, fc="white", ec=C_L, hatch="////", label="Liu 2024, > 600 m")]
    axd.legend(handles=hs, loc="upper right", fontsize=fs - 0.5, frameon=False, ncol=2); S.panel_letter(axd, "d", x=-0.075)
    print(S.export(fig, "F11_qa_evidence", outdir=OUT))


if __name__ == "__main__":
    main()
