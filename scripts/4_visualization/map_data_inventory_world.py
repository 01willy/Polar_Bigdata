"""데이터 인벤토리 세계지도 + 지역x데이터 가용성 매트릭스.

상단: 25-84N 세계지도(ERA5-Land 육지 마스크 배경) 위에
  a) ALT 셀(14,348), b) 시추공 지중온도(260 사이트), c) 신규 ALLena(레나델타),
  d) 신규 QTEC(티베트 2 사이트), e) InSAR/PolSAR 커버리지 박스, f) 북극권 66.56N.
하단: 지역(7) x 데이터(7) 가용성 매트릭스(보유/부분/취득가능/불가).

출력: outputs/maps/data_inventory_world.png / .pdf
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from polar.plotstyle import use_polar, CMAP, BAD, lon_formatter, lat_formatter
from polar.outputs import mappath

plt = use_polar()
plt.rcParams["pdf.fonttype"] = 42   # PDF 텍스트를 TrueType로 임베딩(검색·편집 가능, 세 그림 공통)

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

# 4) 신규 ALLena(레나델타): 헤더 438행, 'Latitude ('/'Longitude (' 시작 컬럼 사용
al = pd.read_csv("data/raw/allena/PANGAEA_973813_ALLena_main.txt", sep="\t", skiprows=437)
al_lat = al[[c for c in al.columns if c.startswith("Latitude (")][0]]
al_lon = al[[c for c in al.columns if c.startswith("Longitude (")][0]]
al_pos = pd.DataFrame({"lat": al_lat, "lon": al_lon}).dropna().drop_duplicates()
n_allena = len(al_pos)

# 5) 신규 QTEC(티베트) 2 사이트: WMO 지점좌표(관측지점 대표 좌표, 근사)
qtec = pd.DataFrame({
    "site": ["Wudaoliang", "Golmud"],
    "lat": [35.22, 36.42],
    "lon": [93.08, 94.90],
})

# 레나델타 박스 내 기존 지온 사이트 수(매트릭스용, 러시아 계수에 포함)
lena_gt = gt[(gt.lat >= 71.0) & (gt.lat <= 74.0) & (gt.lon >= 122) & (gt.lon <= 131)]

# ---------------------------------------------------------------- 색/마커 규약
C_ALT = CMAP.alt(0.72)          # ALT 셀: 짙은 냉색
C_GT = CMAP.err(0.62)           # 지온: 자주 계열(acton)
C_ALLENA = "#e0851f"            # 신규 강조: 앰버
C_QTEC = "#0f766e"              # 신규 강조: 청록
C_INSAR = "#33518e"
C_POLSAR = "#5c7fbe"

# ---------------------------------------------------------------- 그림
fig = plt.figure(figsize=(16.5, 13.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.22,
                      left=0.055, right=0.985, top=0.945, bottom=0.045)

# ============================ 상단: 세계지도 ============================
ax = fig.add_subplot(gs[0])
ax.set_facecolor("white")

# 육지 배경: cartopy 부재로 ERA5 유효픽셀 마스크로 해안선 대체.
# 규약(plotstyle.BAD): 해양/결측=옅은 회색, 육지=백색으로 육지·해양 경계를 구분한다.
land_img = np.where(land, 1.0, np.nan)
from matplotlib.colors import ListedColormap
cm_land = ListedColormap(["#ffffff"])
cm_land.set_bad(BAD)
ax.imshow(land_img, cmap=cm_land, origin="upper",
          extent=(lon_e5.min(), lon_e5.max(), lat_e5.min(), lat_e5.max()),
          aspect="auto", interpolation="nearest", zorder=0)

# a) ALT 셀
ax.scatter(cell.lon, cell.lat, s=2.5, c=[C_ALT], marker="o", lw=0,
           alpha=0.85, zorder=4, rasterized=True)
# b) 지중온도 사이트
ax.scatter(gt_n.lon, gt_n.lat, s=34, c=[C_GT], marker="^",
           edgecolors="white", linewidths=0.5, alpha=0.95, zorder=5, rasterized=True)
# c) 신규 ALLena: 신규 데이터임을 색+마커 형태('x')로 이중 부호화(색 단독 구분 금지 규약)
ax.scatter(al_pos.lon, al_pos.lat, s=8, c=[C_ALLENA], marker="x", linewidths=0.8,
           alpha=0.9, zorder=6, rasterized=True)
ax.annotate("레나델타 ALLena (신규)", xy=(126.7, 72.5), xytext=(100, 60),
            fontsize=10, color="#8a4d0a", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#8a4d0a", lw=0.9))
# d) 신규 QTEC
ax.scatter(qtec.lon, qtec.lat, s=210, c=[C_QTEC], marker="*",
           edgecolors="white", linewidths=0.7, zorder=7)
ax.annotate("QTEC Wudaoliang·Golmud (신규)\nWMO 지점좌표", xy=(94.0, 35.8),
            xytext=(55, 28.5), fontsize=9.5, color=C_QTEC, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C_QTEC, lw=0.9))

# e) 물리관측 커버리지 박스
ax.add_patch(mpatches.Rectangle((-166.7, 57.8), (-110.4) - (-166.7), 71.5 - 57.8,
             fill=False, ec=C_INSAR, ls=(0, (5, 3)), lw=1.6, zorder=8))
ax.text(-165.5, 55.0, "InSAR (ReSALT)", color=C_INSAR, fontsize=10, fontweight="bold")
ax.add_patch(mpatches.Rectangle((-166.0, 68.0), (-148.0) - (-166.0), 71.5 - 68.0,
             fill=False, ec=C_POLSAR, ls=(0, (2, 2)), lw=1.6, zorder=8))
ax.text(-146.5, 72.6, "PolSAR (ABoVE 북알래스카)", color=C_POLSAR, fontsize=10,
        fontweight="bold")

# f) 북극권
ax.axhline(66.56, color="#2166ac", ls="--", lw=0.8, alpha=0.7, zorder=3)
ax.text(-178, 67.6, "북극권 66.6°N", color="#2166ac", fontsize=8.5, alpha=0.85)

ax.set_xlim(-180, 180)
ax.set_ylim(25, 84)
ax.xaxis.set_major_formatter(lon_formatter())
ax.yaxis.set_major_formatter(lat_formatter())
ax.set_xlabel("경도")
ax.set_ylabel("위도")
ax.set_title("관측 범위 25-84°N (ERA5-Land 육지 마스크 배경, 해양·결측 회색)")
ax.text(0.0, 1.012, "(a)", transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12, fontweight="bold", color="#1b2a41")
ax.grid(alpha=0.3, lw=0.4, color="#bbbbbb")

handles = [
    Line2D([], [], marker="o", ls="", ms=5, mfc=C_ALT, mec="none",
           label=f"ALT 측정 셀 ({n_alt:,}개: 알래스카 {n_ak:,}, 서캐나다 {n_ca:,})"),
    Line2D([], [], marker="^", ls="", ms=7, mfc=C_GT, mec="white",
           label=f"시추공 지중온도 ({n_gt} 사이트, 남극 1 표시범위 밖)"),
    Line2D([], [], marker="x", ls="", ms=6, mew=1.4, color=C_ALLENA,
           label=f"신규 ALT: ALLena 레나델타 ({n_allena:,} 위치)"),
    Line2D([], [], marker="*", ls="", ms=13, mfc=C_QTEC, mec="white",
           label="신규: QTEC 티베트 (2 사이트)"),
    Line2D([], [], ls=(0, (5, 3)), color=C_INSAR, lw=1.6, label="InSAR 커버리지 (ReSALT)"),
    Line2D([], [], ls=(0, (2, 2)), color=C_POLSAR, lw=1.6, label="PolSAR 커버리지 (ABoVE)"),
]
ax.legend(handles=handles, loc="lower left", fontsize=9.5, ncol=2,
          framealpha=0.95, borderpad=0.8)

# ============================ 하단: 매트릭스 ============================
axm = fig.add_subplot(gs[1])

rows = ["알래스카 ABoVE", "서캐나다", "레나델타 (신규)", "티베트 QTP (신규)",
        "스위스 알프스", "러시아·시베리아", "스발바르"]
cols = ["ALT 라벨", "지온 3D 라벨", "ERA5 기후", "DEM 지형", "InSAR", "PolSAR", "CCI prior"]

# 상태 코드: 3=보유, 2=부분, 1=전지구 취득가능, 0=불가
HOLD, PART, ACQ, NO = 3, 2, 1, 0
n_ru = int(reg_cnt.get("Russia", 0) + reg_cnt.get("Russia(W.Siberia)", 0))
status = np.array([
    # ALT      지온    ERA5   DEM    InSAR  PolSAR CCI
    [HOLD,     HOLD,   HOLD,  HOLD,  HOLD,  HOLD,  HOLD],   # 알래스카
    [HOLD,     PART,   HOLD,  HOLD,  NO,    NO,    HOLD],   # 서캐나다
    [HOLD,     PART,   HOLD,  ACQ,   NO,    NO,    HOLD],   # 레나델타(신규)
    [PART,     PART,   HOLD,  ACQ,   NO,    NO,    HOLD],   # 티베트 QTP(신규)
    [NO,       HOLD,   HOLD,  ACQ,   NO,    NO,    HOLD],   # 스위스 알프스
    [NO,       HOLD,   HOLD,  ACQ,   NO,    NO,    HOLD],   # 러시아·시베리아
    [NO,       HOLD,   HOLD,  ACQ,   NO,    NO,    HOLD],   # 스발바르
])
text = [
    [f"{n_ak:,} 셀", f"{int(reg_cnt.get('United States', 0))} 사이트", "O", "O", "ReSALT", "ABoVE", "O"],
    [f"{n_ca:,} 셀", f"{int(reg_cnt.get('Canada', 0))} 사이트", "O", "O", "X", "X", "O"],
    [f"{n_allena:,} 위치", f"{len(lena_gt)} 사이트*", "O", "취득", "X", "X", "O"],
    ["2 사이트", "2 사이트", "O", "취득", "X", "X", "O"],
    ["X", f"{int(reg_cnt.get('Switzerland', 0))} 사이트", "O", "취득", "X", "X", "O"],
    ["X", f"{n_ru} 사이트", "O", "취득", "X", "X", "O"],
    ["X", f"{int(reg_cnt.get('Svalbard', 0))} 사이트", "O", "취득", "X", "X", "O"],
]

col_hold = CMAP.count(0.82)
col_part = CMAP.count(0.48)
col_acq = CMAP.count(0.16)
col_no = "#d8dce1"
lut = {HOLD: col_hold, PART: col_part, ACQ: col_acq, NO: col_no}

from matplotlib.colors import to_rgba
rgb = np.zeros((len(rows), len(cols), 4))
for i in range(len(rows)):
    for j in range(len(cols)):
        rgb[i, j] = to_rgba(lut[status[i, j]])
axm.imshow(rgb, aspect="auto", interpolation="nearest")

for i in range(len(rows)):
    for j in range(len(cols)):
        st = status[i, j]
        tcol = "white" if st == HOLD else ("#1b2a41" if st in (PART, ACQ) else "#8a919b")
        fw = "bold" if st == HOLD else "normal"
        axm.text(j, i, text[i][j], ha="center", va="center",
                 fontsize=9.5, color=tcol, fontweight=fw)

axm.set_xticks(range(len(cols)))
axm.set_xticklabels(cols, fontsize=10.5)
axm.set_yticks(range(len(rows)))
axm.set_yticklabels(rows, fontsize=10.5)
axm.set_xticks(np.arange(-0.5, len(cols)), minor=True)
axm.set_yticks(np.arange(-0.5, len(rows)), minor=True)
axm.grid(which="minor", color="white", lw=1.6)
axm.grid(which="major", visible=False)
axm.tick_params(length=0)
axm.set_title("지역 × 데이터 가용성 매트릭스")
axm.text(0.0, 1.02, "(b)", transform=axm.transAxes, ha="left", va="bottom",
         fontsize=12, fontweight="bold", color="#1b2a41")

mat_handles = [
    mpatches.Patch(fc=col_hold, label="보유"),
    mpatches.Patch(fc=col_part, label="부분 보유"),
    mpatches.Patch(fc=col_acq, label="전지구 취득가능"),
    mpatches.Patch(fc=col_no, label="불가 (X)"),
]
axm.legend(handles=mat_handles, loc="upper left", bbox_to_anchor=(0.0, -0.14),
           ncol=4, fontsize=9.5, frameon=False)
axm.text(1.0, -0.16,
         "* 레나델타 지온 2 사이트(Samoylov·Tiksi)는 러시아·시베리아 계수에 포함. "
         "표 외 지온: 스웨덴 8·오스트리아 4·남극 1.",
         transform=axm.transAxes, ha="right", va="top", fontsize=8.5, color="#555555")

fig.suptitle("영구동토 관측 데이터 인벤토리: 공간 분포와 지역별 가용성", fontsize=16,
             fontweight="bold", y=0.985)

png = mappath("data_inventory_world")
pdf = mappath("data_inventory_world", ext="pdf")
fig.savefig(png)
fig.savefig(pdf)
plt.close(fig)
print(f"[saved] {png}")
print(f"[saved] {pdf}")
print(f"ALT={n_alt:,} (AK {n_ak:,}/CA {n_ca:,}), 지온={n_gt}, ALLena={n_allena:,}, QTEC=2")
