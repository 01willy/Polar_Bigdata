# -*- coding: utf-8 -*-
"""중간보고 PPT용 P0·P1 결과 그림.
- P0: outputs/maps의 인벤토리·6모델 오차 지도를 덱 에셋 폴더로 복사(여백 패딩).
- P1: 통합 재학습(unified_tournament) 결과 차트 2종.
출력: deck/assets/mid/{inventory_world,tournament_errors,unified_transfer,missing_routing}.png
빌드: (ROOT) python3 deck/mk_p0p1_figs.py
"""
import os, sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

for _f in ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold"]:
    try: fm.fontManager.addfont(f"/home/willy010313/.fonts/Pretendard-{_f}.otf")
    except Exception: pass
FP  = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-SemiBold.otf")
FPB = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-ExtraBold.otf")
FPM = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-Medium.otf")
plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none"})

OUT = "deck/assets/mid"; os.makedirs(OUT, exist_ok=True)
GRAY="#9AA0A6"; BLUE="#2E86AB"; GREEN="#2E8B57"; NAVY="#12324B"; AMBER="#E08A1E"
TEAL="#0E7C86"; RED="#C0392B"; INK="#18181B"; SLATE="#4A5158"; MUTE="#8A9096"
PAPER="#FFFFFF"; GRID="#E6E8EB"

def _style(ax, ts=13):
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("left","bottom"): ax.spines[sp].set_color("#C9CDD2"); ax.spines[sp].set_linewidth(1.0)
    ax.tick_params(length=0, labelsize=ts)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontproperties(FPM); t.set_fontsize(ts)

def _src(fig, txt):
    fig.text(0.01, -0.015, txt, fontsize=8.5, color=MUTE, fontproperties=FPM, ha="left", va="top")

# ================================================================= P0: 지도 에셋 복사(여백 패딩)
def _pad(src, dst, right=0.0, bottom=0.0):
    im = Image.open(src).convert("RGB"); w,h = im.size
    cv = Image.new("RGB", (int(w*(1+right)), int(h*(1+bottom))), (255,255,255))
    cv.paste(im, (0,0)); cv.save(dst)
    print("saved", dst)

_pad("outputs/maps/data_inventory_world.png", f"{OUT}/inventory_world.png")
_pad("outputs/maps/tournament_error_maps.png", f"{OUT}/tournament_errors.png")

# 인벤토리 지도의 상단 지도 패널만 크롭(하단 매트릭스는 슬라이드에서 네이티브 표로 재구성 → 판독성 확보)
_iv = Image.open("outputs/maps/data_inventory_world.png").convert("RGB")
_w, _h = _iv.size
_iv.crop((0, 0, _w, int(_h * 0.605))).save(f"{OUT}/inventory_map.png")
print("saved inventory_map.png (지도 패널만)")

# ================================================================= P1-1: 다지역 LORO 전이 (지역×모델)
pr = pd.read_csv("data/processed/unified_tournament_perregion.csv")
regions = [("ABoVE_AK","알래스카\n(전이)", 13542), ("ABoVE_CA","서캐나다\n(전이)", 726),
           ("Lena_RU","레나델타\n(신규·전이)", 3037)]
models = ["GBM","FT-Transformer","앙상블(GBM+FT-T)","Diffusion"]
mlab = {"GBM":"GBM","FT-Transformer":"FT-Transformer","앙상블(GBM+FT-T)":"앙상블","Diffusion":"Diffusion"}
mcol = {"GBM":GRAY,"FT-Transformer":BLUE,"앙상블(GBM+FT-T)":NAVY,"Diffusion":GREEN}
fig, ax = plt.subplots(figsize=(9.4, 4.7), dpi=200)
x = np.arange(len(regions)); wbar = 0.20
for j, m in enumerate(models):
    vals = []
    for r,_,_ in regions:
        row = pr[(pr.region==r)&(pr.model==m)]
        vals.append(float(row.rmse_cm.iloc[0]) if len(row) else np.nan)
    xs = x + (j-1.5)*wbar
    ax.bar(xs, vals, wbar, color=mcol[m], label=mlab[m], zorder=3)
    for xx,v in zip(xs, vals):
        ax.text(xx, v+0.7, f"{v:.0f}", ha="center", fontsize=10.5,
                fontproperties=FPB if m=="앙상블(GBM+FT-T)" else FPM, color=INK, zorder=4)
ax.set_xticks(x); ax.set_xticklabels([r[1] for r in regions], fontsize=13)
for t in ax.get_xticklabels(): t.set_fontproperties(FP)
ax.set_ylabel("전이(LORO) RMSE (cm)  ·  낮을수록 정확", fontsize=13.5, fontproperties=FP)
ax.set_ylim(0, 38)
ax.legend(loc="upper left", frameon=False, prop=FP, fontsize=11.5, ncol=2)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax)
# 알래스카 전이 DL 우세 강조(빈 골짜기 영역에 배치, 막대 라벨과 비겹침)
ax.annotate("전이에서 DL(FT-T·앙상블)이\nGBM보다 정확", xy=(x[0]+0.5*wbar, 15.5), xytext=(0.42, 8.5),
            fontsize=11, color=NAVY, fontproperties=FP, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4), zorder=5)
ax.text(x[2], 35.0, "레나델타 전이는 전 모델 미달(병목 지속)", fontsize=11, color=AMBER,
        ha="center", fontproperties=FP, zorder=5)
ax.set_title("다지역 지역제외(LORO) 전이: 지역·모델별 오차", fontsize=16, fontproperties=FPB,
             color=INK, pad=12, loc="left")
fig.tight_layout()
_src(fig, "자료: unified_tournament_perregion.csv · 셀 단위 전 공변량(25) 통합 학습 · 알래스카·서캐나다·레나델타")
fig.savefig(f"{OUT}/unified_transfer.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved unified_transfer.png")

# ================================================================= P1-2: 결측 모달리티 라우팅 아티팩트
va = pd.read_csv("data/processed/unified_vs_alaska_results.csv")
def _rmse(test, train):
    r = va[(va.test==test)&(va.train==train)]
    return float(r.rmse_cm.iloc[0]) if len(r) else np.nan
cats = ["알래스카특화\nNaN 네이티브", "알래스카특화\n중앙값 대체",
        "통합 학습\n중앙값 대체", "통합 학습\nNaN 네이티브"]
vals = [_rmse("Lena_RU(전이)","알래스카특화/NaN네이티브"),
        _rmse("Lena_RU(전이)","알래스카특화/중앙값대체"),
        _rmse("Lena_RU(전이)","통합(레나 제외 전체)/중앙값대체"),
        _rmse("Lena_RU(전이)","통합(레나 제외 전체)/NaN네이티브")]
cols = [BLUE, NAVY, AMBER, RED]
fig, ax = plt.subplots(figsize=(8.2, 4.7), dpi=200)
ax.bar(range(4), vals, color=cols, width=0.62, zorder=3)
for i,v in enumerate(vals):
    ax.text(i, v+1.0, f"{v:.1f}", ha="center", fontsize=13.5, fontproperties=FPB, color=INK, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels(cats, fontsize=11.5)
for t in ax.get_xticklabels(): t.set_fontproperties(FPM)
ax.set_ylabel("레나델타 전이 RMSE (cm)", fontsize=13.5, fontproperties=FP)
ax.set_ylim(0, 58)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True); _style(ax)
ax.annotate("NaN 네이티브 GBM이\n'InSAR 결측 = 깊은 ALT'를\n오학습", xy=(3, vals[3]), xytext=(1.7, 50),
            fontsize=11, color=RED, fontproperties=FP, ha="left",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5), zorder=5)
ax.set_title("결측 모달리티 처리에 따라 전이 성능이 뒤집힌다", fontsize=15.5, fontproperties=FPB,
             color=INK, pad=12, loc="left")
fig.tight_layout()
_src(fig, "자료: unified_vs_alaska_results.csv · GBM · 신규 지역 InSAR/PolSAR 전면 결측 처리 진단")
fig.savefig(f"{OUT}/missing_routing.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved missing_routing.png")

# ================================================================= 슬라이드 5: ALT 관측 커버리지 재생성
# 기존 outputs/figures/01_data/global_alt_coverage.png는 em-dash·"활성층" 표기 위반 →
# 냉색 규약·보고서 문체(활동층)로 덱 전용 재작성.
from polar.plotstyle import CMAP as PCMAP
g = pd.read_csv("data/processed/alt_global.csv")
per = g.groupby(["site","lat","lon","country"]).agg(alt_mean=("alt_cm","mean"),
                                                     n=("year","nunique")).reset_index()
fig = plt.figure(figsize=(13.2, 5.2), dpi=200); fig.patch.set_facecolor(PAPER)
ax = fig.add_subplot(1, 3, (1, 2))
sc = ax.scatter(per.lon, per.lat, c=np.clip(per.alt_mean, 20, 200), s=14+per.n*2.0,
                cmap=PCMAP.alt, alpha=0.88, edgecolor="k", linewidth=0.3, zorder=3)
cb = plt.colorbar(sc, ax=ax, fraction=0.024, pad=0.01)
cb.set_label("평균 활동층 두께 ALT (cm)", fontsize=12.5, fontproperties=FP)
cb.ax.tick_params(labelsize=11)
for t in cb.ax.get_yticklabels(): t.set_fontproperties(FPM)
ax.axhline(66.56, color="#5B8AAE", ls=":", lw=1.2, zorder=2)
ax.text(-178, 67.6, "북극권 66.5°N", fontsize=10.5, color="#3E6E8E", fontproperties=FPM)
ax.set_xlim(-180, 180); ax.set_ylim(28, 82); ax.set_facecolor("#eef2f5")
ax.set_xlabel("경도 (°E)", fontsize=12.5, fontproperties=FP)
ax.set_ylabel("위도 (°N)", fontsize=12.5, fontproperties=FP)
ax.set_title(f"(a) 전 지구 CALM 활동층 관측망 · {len(per)}개 사이트 / {g.country.nunique()}개국",
             fontsize=14, fontproperties=FPB, color=INK, loc="left", pad=9)
ax.grid(alpha=0.25); _style(ax, ts=11.5)
ax2 = fig.add_subplot(1, 3, 3)
cc = per.groupby("country")["site"].nunique().sort_values().tail(10)
ax2.barh(range(len(cc)), cc.values, color=NAVY, zorder=3)
ax2.set_yticks(range(len(cc))); ax2.set_yticklabels(cc.index, fontsize=10)
for t in ax2.get_yticklabels(): t.set_fontproperties(FPM)
ax2.bar_label(ax2.containers[0], fontsize=10, fontproperties=FPM, padding=2)
ax2.set_xlabel("사이트 수", fontsize=12.5, fontproperties=FP)
ax2.set_title("(b) 국가별 사이트 수(상위 10)", fontsize=13.5, fontproperties=FPB, color=INK, loc="left", pad=9)
ax2.margins(x=0.14); _style(ax2, ts=11)
fig.suptitle("전 지구 활동층 학습 데이터의 분포와 지역 다양성", fontsize=16,
             fontproperties=FPB, color=INK, x=0.01, ha="left", y=1.02)
fig.tight_layout()
_src(fig, "자료: alt_global.csv(CALM) · 색=평균 ALT, 점 크기=관측 연수")
fig.savefig(f"{OUT}/alt_coverage.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved alt_coverage.png")
