"""데이터 인벤토리 세계지도 + 지역x자료 가용성 매트릭스.

(a) 25-84N 세계지도(ERA5-Land 육지 마스크 배경) 위에
    ALT 셀, 시추공 지중온도 사이트, ALLena(레나델타), QTEC(티베트고원),
    KPDC Council 현장관측, InSAR/PolSAR 커버리지 박스, 북극권 66.56N.
(b) 지역(8) x 자료(7) 가용성 매트릭스.

출력: outputs/maps/data_inventory_world.png / .pdf
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, to_rgba

from polar.plotstyle import use_polar, BAD, lon_formatter, lat_formatter
from polar.outputs import mappath

plt = use_polar()
plt.rcParams["pdf.fonttype"] = 42   # PDF 텍스트를 TrueType로 임베딩(검색·편집 가능)

# ---------------------------------------------------------------- 데이터 로드
# 1) 육지 마스크: ERA5-Land t2m 첫 timestep 유효픽셀 (25-84N, 0.1도)
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
t2m0 = ds["t2m"].isel(valid_time=0).load()
land = np.isfinite(t2m0.values)                     # (591, 3600), lat 84 -> 25
lat_e5 = ds["latitude"].values
lon_e5 = ds["longitude"].values
ds.close()

# 2) ALT 셀
cell = pd.read_csv("data/processed/dl_dataset_cell.csv", usecols=["lat", "lon", "region"])
n_alt = len(cell)
n_ak = (cell.region.isin(["ABoVE_AK", "United States (Alaska)"])).sum()
n_ca = (cell.region.isin(["ABoVE_CA", "Canada"])).sum()

# 3) 지중온도 사이트 (site 단위 중복 제거)
gt = pd.read_csv("data/processed/ground_temp_all.csv").drop_duplicates("site")
n_gt = len(gt)
gt_n = gt[gt.lat > 0]                               # 지도 범위(25-84N) 밖 남극 1 제외
reg_cnt = gt.groupby("region").size()

# 4) ALLena(레나델타): 헤더 438행, 'Latitude ('/'Longitude (' 시작 컬럼 사용
al = pd.read_csv("data/raw/allena/PANGAEA_973813_ALLena_main.txt", sep="\t", skiprows=437)
al_lat = al[[c for c in al.columns if c.startswith("Latitude (")][0]]
al_lon = al[[c for c in al.columns if c.startswith("Longitude (")][0]]
al_pos = pd.DataFrame({"lat": al_lat, "lon": al_lon}).dropna().drop_duplicates()
n_allena = len(al_pos)

# 5) QTEC(티베트고원) 2 사이트: WMO 지점좌표(관측지점 대표 좌표, 근사)
qtec = pd.DataFrame({
    "site": ["Wudaoliang", "Golmud"],
    "lat": [35.22, 36.42],
    "lon": [93.08, 94.90],
})

# 6) KPDC 현장관측: Council 주 사이트(Seward Peninsula, AK).
# 코어 길이 18개(72-88cm) = ALT 직접 실측, 일별 지중온도 프로파일 = 0°C 등온선 ALT 유도.
kpdc = pd.DataFrame({
    "site": ["Council"],
    "lat": [64.85],
    "lon": [-163.70],
})
try:
    kpdc_alt = pd.read_csv("data/processed/s7_council_alt_derived.csv")
    n_kpdc_season = len(kpdc_alt)
except Exception:
    n_kpdc_season = 0

# 레나델타 박스 내 기존 지온 사이트 수(매트릭스용, 러시아 계수에 포함)
lena_gt = gt[(gt.lat >= 71.0) & (gt.lat <= 74.0) & (gt.lon >= 122) & (gt.lon <= 131)]

# ---------------------------------------------------------------- 색/마커 규약
# 저채도·중명도 냉색 4단 팔레트. 강조는 진청 하나만 사용하고 나머지는 회청 계열.
C_NAVY = "#2f4b6e"      # 진청 (현장관측 지점: 단일 강조)
C_TEAL = "#4a7c8c"      # 청록
C_GRAY = "#8296a8"      # 청회
C_PALE = "#b8c4d0"      # 연청회
C_TXT = "#3b4652"       # 본문 텍스트
C_MUTE = "#6b7683"      # 보조 주석 텍스트

C_ALT = "#9aabb9"       # ALT 셀(밀집 점군): 연청회
C_GT = C_TEAL           # 시추공 지중온도
C_SITE = C_NAVY         # ALLena·QTEC·KPDC 현장 지점
C_INSAR = "#5f7d95"
C_POLSAR = C_GRAY

# ---------------------------------------------------------------- 그림
fig = plt.figure(figsize=(12.6, 11.2))
# 두 패널의 '라벨 포함' 좌측 경계를 맞추기 위해 gridspec을 분리한다.
# 매트릭스는 행 라벨(최대 8자)이 축 밖으로 나가므로 축 좌측을 그만큼 안쪽으로 넣는다.
gs_map = fig.add_gridspec(1, 1, left=0.105, right=0.985, top=0.975, bottom=0.475)
gs_mat = fig.add_gridspec(1, 1, left=0.150, right=0.985, top=0.400, bottom=0.130)

# ============================ (a) 세계지도 ============================
ax = fig.add_subplot(gs_map[0])
ax.set_facecolor("white")

# 육지 배경: cartopy 부재로 ERA5 유효픽셀 마스크로 해안선 대체.
# 해양/결측=옅은 회색, 육지=백색으로 육지·해양 경계를 구분한다.
land_img = np.where(land, 1.0, np.nan)
cm_land = ListedColormap(["#ffffff"])
cm_land.set_bad(BAD)
ax.imshow(land_img, cmap=cm_land, origin="upper",
          extent=(lon_e5.min(), lon_e5.max(), lat_e5.min(), lat_e5.max()),
          aspect="auto", interpolation="nearest", zorder=0)

# ALT 셀: 점 밀도를 드러내되 채도는 낮게
ax.scatter(cell.lon, cell.lat, s=7, c=[C_ALT], marker="o",
           linewidths=0, alpha=0.85, zorder=4, rasterized=True)
# 시추공 지중온도 사이트
ax.scatter(gt_n.lon, gt_n.lat, s=24, c=[C_GT], marker="^",
           edgecolors="white", linewidths=0.35, alpha=0.9, zorder=5, rasterized=True)
# ALLena(레나델타): 색이 아닌 마커 형태로도 구분(색 단독 구분 금지)
ax.scatter(al_pos.lon, al_pos.lat, s=8, c=[C_SITE], marker="x", linewidths=0.7,
           alpha=0.85, zorder=6, rasterized=True)
# 라벨은 시베리아 지온 사이트 점군과 겹치지 않도록 북극해(자료 없음) 위로 배치
ax.annotate("ALLena 레나델타", xy=(129.0, 73.4), xytext=(133, 78.6),
            fontsize=9.5, color=C_MUTE,
            arrowprops=dict(arrowstyle="-", color=C_MUTE, lw=0.6))
# QTEC(티베트고원)
ax.scatter(qtec.lon, qtec.lat, s=130, c=[C_SITE], marker="*",
           edgecolors="white", linewidths=0.5, zorder=7)
ax.annotate("QTEC Wudaoliang·Golmud", xy=(94.0, 35.4), xytext=(52, 29.0),
            fontsize=9.5, color=C_MUTE,
            arrowprops=dict(arrowstyle="-", color=C_MUTE, lw=0.6))
# KPDC Council 현장관측
ax.scatter(kpdc.lon, kpdc.lat, s=150, c=[C_SITE], marker="P",
           edgecolors="white", linewidths=0.6, zorder=9)
ax.annotate("KPDC Council\nALT 코어·지중온도 프로파일",
            xy=(-163.7, 64.85), xytext=(-152, 45.5),
            fontsize=9.5, color=C_MUTE, linespacing=1.35,
            arrowprops=dict(arrowstyle="-", color=C_MUTE, lw=0.6))

# 물리관측 커버리지 박스
ax.add_patch(mpatches.Rectangle((-166.7, 57.8), (-110.4) - (-166.7), 71.5 - 57.8,
             fill=False, ec=C_INSAR, ls=(0, (5, 3)), lw=1.1, zorder=8))
ax.text(-165.5, 54.8, "InSAR (ReSALT)", color=C_INSAR, fontsize=9.5)
ax.add_patch(mpatches.Rectangle((-166.0, 68.0), (-148.0) - (-166.0), 71.5 - 68.0,
             fill=False, ec=C_POLSAR, ls=(0, (2, 2)), lw=1.1, zorder=8))
ax.text(-146.0, 72.8, "PolSAR (ABoVE 북알래스카)", color="#66798c", fontsize=9.5)

# 북극권
ax.axhline(66.56, color=C_GRAY, ls="--", lw=0.7, alpha=0.9, zorder=3)
# 라벨 위치: InSAR 박스와 겹치지 않도록 그린란드 내부(데이터 없음)로 배치
ax.text(-52, 67.6, "북극권 66.6°N", color=C_MUTE, fontsize=9)

ax.set_xlim(-180, 180)
ax.set_ylim(25, 84)
ax.xaxis.set_major_formatter(lon_formatter())
ax.yaxis.set_major_formatter(lat_formatter())
ax.set_xlabel("경도", fontsize=10.5, color=C_TXT)
ax.set_ylabel("위도", fontsize=10.5, color=C_TXT)
ax.tick_params(labelsize=9.5, length=2.5, colors=C_TXT)
ax.grid(alpha=0.25, lw=0.5, color="#aab2bb")
for s in ax.spines.values():
    s.set_linewidth(0.6)
    s.set_edgecolor("#b4bcc4")
ax.text(0.0, 1.010, "(a)", transform=ax.transAxes, ha="left", va="bottom",
        fontsize=11, fontweight="bold", color="#26333f")

handles = [
    Line2D([], [], marker="o", ls="", ms=5, mfc=C_ALT, mec="none",
           label=f"ALT 셀 {n_alt:,} (알래스카 {n_ak:,}·캐나다 {n_ca:,})"),
    Line2D([], [], marker="^", ls="", ms=6, mfc=C_GT, mec="white", mew=0.4,
           label=f"시추공 지중온도 {n_gt} 사이트"),
    Line2D([], [], marker="P", ls="", ms=9, mfc=C_SITE, mec="white", mew=0.5,
           label="KPDC Council 현장관측"),
    Line2D([], [], marker="x", ls="", ms=6, mew=1.2, color=C_SITE,
           label=f"ALLena 레나델타 {n_allena:,} 위치"),
    Line2D([], [], marker="*", ls="", ms=11, mfc=C_SITE, mec="white", mew=0.4,
           label="QTEC 티베트고원 2 사이트"),
    Line2D([], [], ls=(0, (5, 3)), color=C_INSAR, lw=1.1, label="InSAR (ReSALT)"),
    Line2D([], [], ls=(0, (2, 2)), color=C_POLSAR, lw=1.1, label="PolSAR (ABoVE)"),
]
leg_map = ax.legend(handles=handles, loc="lower left", fontsize=9.5, ncol=2,
                    frameon=True, framealpha=0.92, edgecolor="#d5dae0",
                    borderpad=0.7, labelspacing=0.55, columnspacing=1.3,
                    handletextpad=0.6)
leg_map.get_frame().set_linewidth(0.5)
for t in leg_map.get_texts():
    t.set_color(C_TXT)

# ============================ (b) 가용성 매트릭스 ============================
axm = fig.add_subplot(gs_mat[0])

rows = ["KPDC Council", "알래스카", "서캐나다", "레나델타",
        "티베트고원", "스위스 알프스", "러시아·시베리아", "스발바르"]
cols = ["ALT 관측", "지중온도", "ERA5 기후", "DEM 지형", "InSAR", "PolSAR", "ESA CCI"]

# 상태 코드: 3=보유, 2=부분 보유, 1=전지구 취득 가능, 0=해당 없음
HOLD, PART, ACQ, NO = 3, 2, 1, 0
n_ru = int(reg_cnt.get("Russia", 0) + reg_cnt.get("Russia(W.Siberia)", 0))
status = np.array([
    # ALT      지온    ERA5   DEM    InSAR  PolSAR CCI
    [HOLD,     HOLD,   HOLD,  HOLD,  NO,    NO,    HOLD],   # KPDC Council
    [HOLD,     HOLD,   HOLD,  HOLD,  HOLD,  HOLD,  HOLD],   # 알래스카
    [HOLD,     PART,   HOLD,  HOLD,  NO,    NO,    HOLD],   # 서캐나다
    [HOLD,     PART,   HOLD,  ACQ,   NO,    NO,    HOLD],   # 레나델타
    [PART,     PART,   HOLD,  ACQ,   NO,    NO,    HOLD],   # 티베트고원
    [NO,       HOLD,   HOLD,  ACQ,   NO,    NO,    HOLD],   # 스위스 알프스
    [NO,       HOLD,   HOLD,  ACQ,   NO,    NO,    HOLD],   # 러시아·시베리아
    [NO,       HOLD,   HOLD,  ACQ,   NO,    NO,    HOLD],   # 스발바르
])
text = [
    ["18 코어", f"{n_kpdc_season} 프로파일", "보유", "보유", "—", "—", "보유"],
    [f"{n_ak:,} 셀", f"{int(reg_cnt.get('United States', 0))} 사이트", "보유", "보유",
     "ReSALT", "ABoVE", "보유"],
    [f"{n_ca:,} 셀", f"{int(reg_cnt.get('Canada', 0))} 사이트", "보유", "보유",
     "—", "—", "보유"],
    [f"{n_allena:,} 위치", f"{len(lena_gt)} 사이트", "보유", "취득 가능", "—", "—", "보유"],
    ["2 사이트", "2 사이트", "보유", "취득 가능", "—", "—", "보유"],
    ["—", f"{int(reg_cnt.get('Switzerland', 0))} 사이트", "보유", "취득 가능",
     "—", "—", "보유"],
    ["—", f"{n_ru} 사이트", "보유", "취득 가능", "—", "—", "보유"],
    ["—", f"{int(reg_cnt.get('Svalbard', 0))} 사이트", "보유", "취득 가능",
     "—", "—", "보유"],
]

# 저채도 4단(연청회~중청). 진한 남색 채움을 쓰지 않아 표가 과포화되지 않게 한다.
col_hold = "#7d94a8"
col_part = "#a9bbc8"
col_acq = "#d4dce2"
col_no = "#eef1f3"
lut = {HOLD: col_hold, PART: col_part, ACQ: col_acq, NO: col_no}

rgb = np.zeros((len(rows), len(cols), 4))
for i in range(len(rows)):
    for j in range(len(cols)):
        rgb[i, j] = to_rgba(lut[status[i, j]])
axm.imshow(rgb, aspect="auto", interpolation="nearest")

for i in range(len(rows)):
    for j in range(len(cols)):
        st = status[i, j]
        tcol = "#98a2ac" if st == NO else "#26333f"
        axm.text(j, i, text[i][j], ha="center", va="center",
                 fontsize=10, color=tcol)

axm.set_xticks(range(len(cols)))
axm.set_xticklabels(cols, fontsize=10, color=C_TXT)
axm.set_yticks(range(len(rows)))
axm.set_yticklabels(rows, fontsize=10, color=C_TXT)
axm.set_xticks(np.arange(-0.5, len(cols)), minor=True)
axm.set_yticks(np.arange(-0.5, len(rows)), minor=True)
axm.grid(which="minor", color="white", lw=1.4)
axm.grid(which="major", visible=False)
axm.tick_params(length=0)
for s in axm.spines.values():
    s.set_visible(False)
# (a)·(b) 패널 라벨의 도형상 x 위치를 일치시킨다(두 축의 left 차이 0.045를
# 매트릭스 축 폭 0.835로 나눈 값만큼 왼쪽으로 이동).
axm.text(-0.054, 1.015, "(b)", transform=axm.transAxes, ha="left", va="bottom",
         fontsize=11, fontweight="bold", color="#26333f")

mat_handles = [
    mpatches.Patch(fc=col_hold, ec="none", label="보유"),
    mpatches.Patch(fc=col_part, ec="none", label="부분 보유"),
    mpatches.Patch(fc=col_acq, ec="none", label="전지구 취득 가능"),
    mpatches.Patch(fc=col_no, ec="#ccd3d9", lw=0.6, label="해당 없음"),
]
leg = axm.legend(handles=mat_handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
                 ncol=4, fontsize=9.5, frameon=False, handlelength=1.5,
                 handleheight=1.0, columnspacing=2.0, handletextpad=0.6)
for t in leg.get_texts():
    t.set_color(C_TXT)

png = mappath("data_inventory_world")
pdf = mappath("data_inventory_world", ext="pdf")
fig.savefig(png, dpi=300)
# 벡터 PDF: ALT 14,348 셀·지온 260 삼각형 산점도는 rasterized=True로 의도적 래스터화
# (파일 크기·렌더 부하 절감). 래스터 요소 dpi 600, 텍스트·범례는 fonttype 42 벡터 유지.
fig.savefig(pdf, dpi=600)
plt.close(fig)
print(f"[saved] {png}")
print(f"[saved] {pdf}")
print(f"ALT={n_alt:,} (AK {n_ak:,}/CA {n_ca:,}), 지온={n_gt}, ALLena={n_allena:,}, QTEC=2")
