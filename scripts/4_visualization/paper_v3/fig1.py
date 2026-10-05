"""Fig 1 v3. 연구 지역, Stefan 계수, 평가 설계(결과 절 R1).

명세: outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md 3절(이하 '명세'), 지침 design/journal_grade_style_guide.md 2절, 3절, 6.1절.
v2 모듈(scripts/4_visualization/paper/fig1_problem.py)은 참고만 했고 가져오지 않는다. 공용 양식은 같은 폴더의 style.py 를 그대로 쓴다.

패널(170 × 155 mm, 위에서부터 y)
  a  범북극 지도(북극 평사, 진척 위도 70° N, 중심 경도는 레나델타 중앙 경도 126.7° E [판단, 명세의 −45° 대신: 확대 연결선 때문]).
     0.5° 블록별 라벨 위치 원(면적 ∝ 1 km 위치 수).
     v3 블록과 주 지역 확충 블록은 검정 채움 alpha 0.35, 새 지역(중부 러시아 확충판, 티베트)은 흰 채움.
     영구동토 바탕(연속 ≥ 90 %, 불연속 50–90 %), 경위선, 위도 라벨, 1000 km 축척, 티베트 삽도(Lambert 방위 등적),
     c·d 범위 사각형과 c 로 가는 연결선, 왼쪽 아래 열쇠 2줄(크기 열쇠 + 새 지역 견본, 영구동토 견본).
  b  지역별 Stefan 계수 E = ALT/√TDD 분포(로그 축). 개별 점(행당 최대 400개, v2_data 표본), 중앙값과 사분위, 원천 계수 눈금.
  c  레나델타 지역 홀드아웃: 대상 셀과 100 km 거리 등치선.
  d  같은 범위의 블록 홀드아웃: 분할 1 의 라벨 블록과 채점 블록, 첫 추출의 라벨 셀 40개.
  e  검증 사다리 자리(XH 결과 전): 빈 축과 축 이름만(명세 12절).

자료(읽기만)
  data/processed/paper_figs/fig1_blocks.csv, fig1_z_points.csv, fig1_z_summary.csv, fig1_not_drawn.csv, fig1_source.csv,
  fig1_meta.csv, table1_rows.csv, data/processed/cci_pfr_mean_1997_2021.nc,
  data/processed/fidelity_base_v3.csv(+ e5_soil_tdd_v3.csv, polar.m1_core.load_base 경유),
  results/rescale_lg/data/processed/lg/lg_targets.csv(분할 재현 대조).
  블록 분할은 polar.m1_core.half_split_blocks, 라벨 추출은 scripts/3_deep_learning/h40_label_grid.py 의 draw_cells 를 그대로 부른다.

산출(outputs/figures/paper/v3_restructure/)
  Fig1.pdf(정본, 글꼴 내장), Fig1.png(600 dpi), Fig1_source_data.csv, Fig1_values.txt. 설명문 Fig1_legend.md 는 손으로 쓰고,
  이 모듈이 단어 수(350 이하)·문장 점검·첫 문장을 Fig1_values.txt 에 기록한다.

실행(공유 서버: 렌더 전에 가용 메모리 30 GB 이상, 부하 40 이하를 확인한다)
  OMP_NUM_THREADS=2 nice -n 10 python3 scripts/4_visualization/paper_v3/fig1.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.path as mpath  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402

ROOT = S.ROOT
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
PF = S.PAPER_FIGS
STEM = "Fig1"

# ---------------------------------------------------------------- 고정값(명세 3절, [판단]은 Fig1_values.txt 에 기록)
FIG_W, FIG_H = S.W2_MM, 155.0
TRUE_LAT, LAT_MIN = 70.0, 50.0                         # 지침 2.11
# a 의 중심 경도: 명세·지침의 −45°(EPSG:3413 형식) 대신 c·d 와 같은 레나델타 중앙 경도(126.7° E)를 쓴다 [판단].
# −45° 에서는 레나델타 사각형이 지도 위끝에 놓여 c 로 가는 연결선(지침 2.11)이 극, 캐나다, 그린란드를 가로질렀다(1차 렌더 검토).
LON0_A = None                                          # None = 확대도 중심 경도
SIZE_K = 5.0                                           # 원 면적(pt², 지름²) = SIZE_K × 1 km 위치 수 [판단]
SIZE_KEY = (1, 10, 50)                                 # 명세 3.3 a 의 크기 열쇠 값
EDGE_LW = 1.0
FILL_OBS = (0.0, 0.0, 0.0, 0.35)                       # (v3 판) 관측 위치 = 검정 alpha 0.35(R-18). v4 지역 색 판에서는 쓰지 않는다
# v4 토큰 color.regions(2026-10-05 조정 지시): a 의 블록 원 = 지역 색. 이 패널은 지역이 주 부호화라 지역 색을 쓴다(덱 M3_data a 와 같은 대응).
# 테두리는 같은 색상의 어두운 선(채움 RGB × 0.6): 흰 테두리 판과 비교한 결과 작은 원이 줄어 보이지 않고, 밝은 지역 색(레나델타, 러시아 중부)도
# 회색 영구동토 바탕에서 경계를 갖는다[판단, scratchpad f1a_var/compare.png]. 새 지역(러시아 중부 확충판, 티베트 고원 추가분)은 흰 원 그대로.
# 토큰에 지역 색이 없는 블록(그린란드 2, 스발바르 1, 스칸디나비아 1)과 원 면적 열쇠는 중립 회색(지역을 뜻하지 않음).
BLOCK_REGION = {"Alaska": "Alaska", "Canada": "Canada", "Lena": "Lena Delta", "Russia_W": "W Russia", "Russia_E": "E Russia",
                "Russia_C": "Central Russia", "Tibet": "Tibetan Plateau", "Tibet_LGD": "Tibetan Plateau"}
REGION_FILL_ALPHA = 0.9
REGION_EDGE = "darker"
REGION_EDGE_LW = 0.5
REGION_EDGE_DARK = 0.6                                 # REGION_EDGE == "darker" 일 때 테두리 = 채움 RGB × 0.6
OTHER_FILL = "#8C8C8C"
KEY_FILL = OTHER_FILL
KEY_EDGE = tuple(v * REGION_EDGE_DARK for v in matplotlib.colors.to_rgb(OTHER_FILL))   # 열쇠 원도 같은 테두리 규칙
BUFFER_KM = 100.0
DEMO = dict(target="Lena", mode="x", split=1, n=40, draw=0)   # 명세 '분할 1, 추출 1' = 분할 seed 1, 첫 추출(h40 번호 0) [판단]
JITTER_SEED = 20261004                                 # b 의 세로 흩뿌림 seed
ZPOINT_SEED = 20260930                                 # fig1_z_points 의 400개 표본 seed(v2_data.py 621행)
E_TICKS = (0.1, 0.3, 1, 3, 10, 30)                     # [판단] 명세 1–20 은 최솟값 0.022(캐나다)를 담지 못한다
E_LIM = (0.018, 36.0)
B_ROWS = [("Lena", 0.0), ("Canada", 1.0), ("Russia_W", 2.0), ("Russia_E", 3.0),
          ("Alaska", 4.6), ("Russia_C_LGD", 6.2), ("Tibet_LGD", 7.2)]   # Table 1 순서, 묶음 사이 여백 0.6
TIBET_MARKER = "h"                                     # 티베트 모양은 지침 2.5 에 없다(D-15) [판단]
PFR_MIN_PPI = 450

# 배치(mm, 그림 왼쪽 위 원점). 명세 3.2 의 슬롯 안에 축을 둔다.
# 지도 원은 지름 84 mm(원 중심 45, 49.5). 지름 90 에서는 티베트 삽도 틀이 원을 덮어(위도 50–60° N 의 허드슨만 일대) 2차 렌더 검토에서
# 줄였다. 삽도(오른쪽 위)와 열쇠(아래 오른쪽)가 모두 원 밖에 놓인다 [판단].
A_MAP = (3.0, 7.5, 84.0)                               # x, y, 지름
B_AX = (127.0, 4.0, 42.0, 84.0)
C_AX = (3.0, 106.0, 44.0, 44.0)
D_AX = (54.0, 106.0, 44.0, 44.0)
E_AX = (118.0, 106.0, 51.0, 37.0)
TIBET_AX = (74.5, 4.0, 23.5, 14.5)                     # 오른쪽 위 구석(원 밖, 원 중심에서 틀 왼쪽 아래 모서리까지 42.8 mm > 반지름 42), 이름은 틀 위
LETTERS = [("a", 0.5, 0.5), ("b", 106.0, 0.5), ("c", 0.5, 103.5), ("d", 51.0, 103.5), ("e", 106.0, 103.5)]   # 캔버스 가장자리 여백 0.5 mm

# a 의 지역 이름: 글자 중심을 지도 원 중심 기준 반지름 비율(fx 오른쪽 +, fy 아래 +)로 둔다. 지시선은 그 지역의 가장 가까운 블록 원으로 간다
# (그린란드 2블록은 Table 1 처럼 SI 로, 이름 없음). 비율은 2차 렌더에서 블록 무게중심을 재어 정했다(캐나다 +0.57, −0.25 등).
REGION_LABELS = {
    "Alaska": ("Alaska", (0.82, 0.03)),
    "Canada": ("Canada", (0.76, -0.52)),
    "Lena": ("Lena Delta", (-0.04, 0.22)),
    "Russia_W": ("W Russia", (-0.78, 0.16)),
    "Russia_E": ("E Russia", (0.69, 0.54)),
    "Russia_C": ("Central Russia", (-0.47, 0.57)),
}
# 지시선 끝: 기본은 글자에서 가장 가까운 그 지역 블록. 캐나다는 가까운 외딴 블록(앨버타 1셀) 대신 1 km 위치가 가장 많은 블록(주 군집)으로 [판단]
LEADER_ANCHOR = {"Canada": "largest"}
LAT_LABEL_LON = -20.0                                  # 위도 라벨을 다는 경선(북대서양·그린란드해)
SCALE_A = dict(lon=5.0, lat=70.0, km=1000)             # 축척 막대 중심(노르웨이해, 70° N)
KEY_AX = (54.5, 90.5, 43.5, 14.5)                      # a 의 열쇠 3줄(크기, 영구동토 머리, 구역 견본): 원 아래 오른쪽 띠(왼쪽 아래는 c 로 가는 연결선 자리) [판단]
RECT_HALO_LW = 2.5                                     # 확대 사각형의 흰 바탕 선(밀집 원 위에서 보이게), 검정 1.0 pt 아래


# ================================================================ 공용
def check_resources(min_gb=None, max_load=40.0, wait_s=60, max_wait_s=3600):
    min_gb = float(os.environ.get("PAPER_MIN_AVAIL_GB", "30")) if min_gb is None else min_gb   # 기본 30 GB(작업 지시), 조정 담당이 정한 값은 환경 변수로
    """공유 서버 자원 확인(작업 지시). 가용 메모리 < 30 GB 또는 1분 부하 > 40 이면 기다린다."""
    t0 = time.time()
    while True:
        with open("/proc/meminfo") as f:
            mem = {ln.split(":")[0]: float(ln.split()[1]) for ln in f}
        avail = mem["MemAvailable"] / 1024 ** 2
        load1 = os.getloadavg()[0]
        if avail >= min_gb and load1 <= max_load:
            return dict(avail_gb=round(avail, 1), load1=round(load1, 2))
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"자원 부족 지속: 가용 {avail:.1f} GB, 부하 {load1:.1f}")
        print(f"  [wait] 가용 {avail:.1f} GB, 부하 {load1:.1f}", flush=True)
        time.sleep(wait_s)


def rd(name):
    return pd.read_csv(PF / f"{name}.csv")


def pt2mm(v):
    return v * 25.4 / 72.0


def size_pt2(n):
    """scatter 크기(pt², 지름²). 면적 ∝ n."""
    return SIZE_K * np.asarray(n, float)


def load_h40():
    p = ROOT / "scripts" / "3_deep_learning" / "h40_label_grid.py"
    spec = importlib.util.spec_from_file_location("h40_label_grid", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pfr_classes():
    import netCDF4 as nc
    with nc.Dataset(PROC / "cci_pfr_mean_1997_2021.nc") as f:
        lat, lon, M = f["lat"][:].data, f["lon"][:].data, f["pfr_mean"][:].filled(np.nan)
        title = f.getncattr("title")
    return lat, lon, M, title


def draw_pfr(ax, proj, w_mm, h_mm, P, zorder=0.3):
    """영구동토 2단계(연속 ≥ 90 %, 불연속 50–90 %)를 투영 격자에 최근접 표본화한 래스터. 내장 해상도 ≥ 450 ppi."""
    import cartopy.crs as ccrs
    lat, lon, M, _ = P
    nx = int(np.ceil(w_mm / 25.4 * PFR_MIN_PPI * 1.05)); ny = int(np.ceil(h_mm / 25.4 * PFR_MIN_PPI * 1.05))
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y1, y0, ny))
    ll = ccrs.PlateCarree().transform_points(proj, X, Y)
    LON, LAT = ll[..., 0], ll[..., 1]
    i = np.clip(np.round((LAT - lat[0]) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
    j = np.clip(np.round((LON - lon[0]) / (lon[1] - lon[0])).astype(int), 0, len(lon) - 1)
    v = M[i, j]
    v[(LAT < lat.min() - 0.05) | (LAT > lat.max() + 0.05) | ~np.isfinite(LAT)] = np.nan
    cls = np.full(v.shape, np.nan)
    cls[(v >= 50) & (v < 90)] = 0
    cls[v >= 90] = 1
    cm = ListedColormap([S.BASEMAP["discontinuous"], S.BASEMAP["continuous"]])
    ax.imshow(np.ma.masked_invalid(cls), extent=(x0, x1, y0, y1), origin="upper", cmap=cm, vmin=0, vmax=1,
              interpolation="nearest", transform=proj, zorder=zorder, rasterized=True)
    return dict(nx=nx, ny=ny, ppi=round(nx / (w_mm / 25.4), 0))


def scale_bar(ax, proj, lon, lat, km, label_gid="scale", above=True, align="center"):
    """축척 막대(1.5 pt 검정)와 길이 글자. 투영 좌표 길이 = km × 그 위도의 평사 투영 축척 계수."""
    import cartopy.crs as ccrs
    k = (1 + np.sin(np.radians(TRUE_LAT))) / (1 + np.sin(np.radians(lat))) if isinstance(proj, ccrs.NorthPolarStereo) else 1.0
    cx, cy = proj.transform_point(lon, lat, ccrs.PlateCarree())
    L = km * 1000.0 * k
    x0 = cx - L / 2 if align == "center" else cx
    ax.plot([x0, x0 + L], [cy, cy], color=S.INK, lw=S.LW["scale_bar"], solid_capstyle="butt", transform=proj, zorder=8)
    t = ax.text(x0 + L / 2, cy, f"{km:d} km", transform=proj, ha="center", va="bottom" if above else "top", zorder=8,
                fontsize=S.FONT_PT)
    t.set_gid(label_gid)
    # 그 위도에서 측지 길이 검산(구면 근사): 막대 양 끝을 경위도로 되돌려 대원 거리를 잰다
    ll = ccrs.PlateCarree().transform_points(proj, np.array([x0, x0 + L]), np.array([cy, cy]))
    return dict(km=km, k=float(k), length_proj_m=float(L), geodesic_km=float(haversine(ll[0, 1], ll[0, 0], ll[1, 1], ll[1, 0])))


def loc1km_index(lat, lon):
    """1 km 위치 색인(fig1_blocks 의 n_loc_1km 와 같은 정의, v2_data.loc1km 재현). ky = floor(lat/0.009), kx = floor(lon·cos φ/0.009)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    ky = np.floor(lat / 0.009).astype(np.int64)
    phi = np.deg2rad((ky + 0.5) * 0.009)
    kx = np.floor(lon * np.cos(phi) / 0.009).astype(np.int64)
    return ky * 10_000_000 + kx


def block05_index(lat, lon):
    """0.5° 블록 id(fig1_blocks 의 block 과 같은 정의)."""
    return np.floor(np.asarray(lat, float) / 0.5).astype(int) * 100000 + np.floor(np.asarray(lon, float) / 0.5).astype(int)


def haversine(la1, lo1, la2, lo2, R=6371.0):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ================================================================ 자료
def load_all():
    blocks = rd("fig1_blocks")
    zp, zs = rd("fig1_z_points"), rd("fig1_z_summary")
    t1 = rd("table1_rows")
    meta = rd("fig1_meta").iloc[0]
    nd = rd("fig1_not_drawn")
    src = rd("fig1_source")
    return dict(blocks=blocks, zp=zp, zs=zs, t1=t1, meta=meta, nd=nd, src=src)


def block_kind_fill(r):
    """명세 3.3 a: lgd_added 의 Russia_C, Tibet_LGD 만 흰 채움(새 지역). 나머지 lgd_added 와 v3 는 지역 색(v4 토큰 color.regions),
    토큰에 없는 지역은 중립 회색. natl_si 는 그리지 않는다."""
    if r.kind == "natl_si":
        return None
    if r.kind == "lgd_added" and r.region in ("Russia_C", "Tibet_LGD"):
        return "white"
    reg = BLOCK_REGION.get(r.region)
    return S.REGION_COLOR[reg] if reg else OTHER_FILL


def circle_style(fills):
    """채움 목록 → (면 색, 테두리 색, 테두리 굵기). 새 지역 = 흰 채움 + 검정 테두리(EDGE_LW), 그 밖 = 지역 색(alpha) + 흰 테두리."""
    fc, ec, lw = [], [], []
    for f in fills:
        if f == "white":
            fc.append((1.0, 1.0, 1.0, 1.0)); ec.append(S.INK); lw.append(EDGE_LW)
        else:
            fc.append(matplotlib.colors.to_rgba(f, REGION_FILL_ALPHA))
            if REGION_EDGE == "darker":                     # 같은 색상의 어두운 테두리(밝은 지역 색이 회색 바탕에서도 원 경계를 갖게)
                r_, g_, b_, _ = matplotlib.colors.to_rgba(f)
                ec.append((r_ * REGION_EDGE_DARK, g_ * REGION_EDGE_DARK, b_ * REGION_EDGE_DARK, 1.0))
            else:
                ec.append(REGION_EDGE)
            lw.append(REGION_EDGE_LW)
    return fc, ec, lw


def lena_design(vals):
    """c·d 의 자료: LG 와 같은 대상 셀(macro == Lena), 분할 1 의 A/B 블록, 첫 추출의 라벨 셀 40개."""
    from polar.m1_core import load_base, half_split_blocks, eval_mask
    h40 = load_h40()
    df = load_base(PROC)
    t_idx = np.where(df.macro.values == DEMO["target"])[0]
    A_idx, B_idx = half_split_blocks(df, t_idx, DEMO["split"])
    evB = B_idx[eval_mask(df.iloc[B_idx])]
    sel = h40.draw_cells(DEMO["target"], DEMO["mode"], DEMO["split"], DEMO["n"], DEMO["draw"], len(A_idx))
    seed = h40.seed_of(DEMO["target"], DEMO["mode"], DEMO["split"], DEMO["n"], DEMO["draw"])
    lab = df.iloc[A_idx[sel]]
    A_blocks = np.unique(df.block.values[A_idx]); B_blocks = np.unique(df.block.values[B_idx])
    score_blocks = np.unique(df.block.values[evB])
    # 원천 셀(대상 밖, y 유효) 가운데 대상 셀 100 km 안(버퍼 제외)과 c·d 범위 안의 셀
    src_idx = np.setdiff1d(np.where(np.isfinite(df[h40.TARGET].values))[0], t_idx)
    tl, tn = df.lat.values[t_idx], df.lon.values[t_idx]
    dmin = np.array([haversine(df.lat.values[i], df.lon.values[i], tl, tn).min() for i in src_idx])
    vals.update(lena_n_cells=int(len(t_idx)), lena_n_blocks=int(df.block.values[t_idx].size and np.unique(df.block.values[t_idx]).size),
                lena_n_loc_unique_3dp=int(df.iloc[t_idx][["lat", "lon"]].round(3).drop_duplicates().shape[0]),
                split1_nA=int(len(A_idx)), split1_nbA=int(len(A_blocks)), split1_nB=int(len(B_idx)), split1_nbB=int(len(B_blocks)),
                split1_n_eval=int(len(evB)), split1_nb_eval=int(len(score_blocks)),
                split1_B_blocks_without_eval=int(len(np.setdiff1d(B_blocks, score_blocks))),
                draw_seed=int(seed), n_label_cells=int(len(lab)), label_cells_in_A=bool(np.isin(lab.block.values, A_blocks).all()),
                label_blocks_hit=int(lab.block.nunique()),
                src_within_100km=int((dmin < BUFFER_KM).sum()),
                src_within_100km_regions=";".join(sorted(df.region.values[src_idx[dmin < BUFFER_KM]].astype(str))))
    return dict(df=df, t_idx=t_idx, A_idx=A_idx, B_idx=B_idx, evB=evB, lab=lab, A_blocks=A_blocks, B_blocks=B_blocks,
                score_blocks=score_blocks, src_idx=src_idx, src_dmin=dmin)


def zoom_projection(L):
    import cartopy.crs as ccrs
    d = L["df"].iloc[L["t_idx"]]
    lon_c = float((d.lon.min() + d.lon.max()) / 2)
    proj = ccrs.NorthPolarStereo(central_longitude=lon_c, true_scale_latitude=TRUE_LAT)
    P = proj.transform_points(ccrs.PlateCarree(), d.lon.values, d.lat.values)
    lat_c = float(d.lat.mean())
    k = (1 + np.sin(np.radians(TRUE_LAT))) / (1 + np.sin(np.radians(lat_c)))
    pad = (BUFFER_KM + 14.0) * 1000.0 * k
    x0, x1 = P[:, 0].min() - pad, P[:, 0].max() + pad
    y0, y1 = P[:, 1].min() - pad, P[:, 1].max() + pad
    half = max(x1 - x0, y1 - y0) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ext = (cx - half, cx + half, cy - half, cy + half)
    return proj, ext, lon_c, lat_c


# ================================================================ 패널
def leader_line(fig, ax, text, target_xy, shrink_end_pt, gap_pt=2.0, lw=1.0):
    """글자 상자 가장자리에서 목표점(원 가장자리)까지의 지시선(Line2D, 화살표 머리 없음, 1.0 pt #4d4d4d).
    annotate 의 화살표(FancyArrowPatch)를 쓰지 않아 digest 5.3 audit_fig 의 사선 화살표 검사에 걸리지 않는다."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bb = text.get_window_extent(rend)
    c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
    tgt = ax.transData.transform(np.asarray(target_xy, float))
    d = tgt - c
    n = np.hypot(*d)
    if n == 0:
        return None
    u = d / n
    # 글자 상자(여백 gap_pt)와 중심→목표 직선의 교점
    hw, hh = bb.width / 2 + gap_pt * fig.dpi / 72, bb.height / 2 + gap_pt * fig.dpi / 72
    tx = hw / abs(u[0]) if u[0] else np.inf
    ty = hh / abs(u[1]) if u[1] else np.inf
    start = c + u * min(tx, ty)
    end = tgt - u * shrink_end_pt * fig.dpi / 72
    if np.dot(end - start, u) <= 0:                         # 목표가 글자 상자 안: 선을 긋지 않는다
        return None
    inv = ax.transData.inverted()
    p0, p1 = inv.transform(start), inv.transform(end)
    ln = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=S.INK_AUX, lw=lw, solid_capstyle="butt", zorder=6.8, clip_on=False)[0]
    ln.set_gid("leader")
    return ln


def fig_mm_to_data(fig, ax, x_mm, y_mm):
    """그림 mm(왼쪽 위 원점)를 축의 자료 좌표로."""
    W, H = fig.get_size_inches() * S.MM_PER_IN
    disp = fig.transFigure.transform((x_mm / W, 1 - y_mm / H))
    return ax.transData.inverted().transform(disp)


def panel_a(fig, D, vals, P, lon0):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    proj = ccrs.NorthPolarStereo(central_longitude=lon0, true_scale_latitude=TRUE_LAT)
    x, y, dia = A_MAP
    ax = S.axes_mm(fig, x, y, dia, dia, projection=proj)
    rho = abs(proj.transform_point(lon0, LAT_MIN, ccrs.PlateCarree())[1])
    ax.set_xlim(-rho, rho); ax.set_ylim(-rho, rho)
    theta = np.linspace(0, 2 * np.pi, 361)
    ax.set_boundary(mpath.Path(np.c_[np.sin(theta), np.cos(theta)] * 0.5 + 0.5), transform=ax.transAxes)
    ax.spines["geo"].set_edgecolor(S.BASEMAP["coast"]); ax.spines["geo"].set_linewidth(1.0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=S.BASEMAP["land"], edgecolor="none", linewidth=0, zorder=0).set_rasterized(True)
    vals["a_pfr_raster"] = draw_pfr(ax, proj, dia, dia, P)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=S.BASEMAP["coast"], facecolor="none", linewidth=S.LW["coast"], zorder=0.4)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=S.LW["grid_map"], color=S.BASEMAP["graticule"],
                 xlocs=np.arange(-180, 180, 30), ylocs=[60, 70, 80], zorder=0.5)
    vals["a_km_per_mm"] = round(rho / 1000.0 / (dia / 2), 2)
    vals["a_central_lon"] = round(float(lon0), 3)
    # 위도 라벨(경선 LAT_LABEL_LON 과 위선의 교점 오른쪽)
    for la in (60, 70, 80):
        t = ax.annotate(f"{la}°N", xy=proj.transform_point(LAT_LABEL_LON, la, ccrs.PlateCarree()), xycoords="data",
                        xytext=(2.0, 1.0), textcoords="offset points", ha="left", va="bottom", fontsize=S.FONT_PT, zorder=6)
        t.set_gid("graticule")
    # 블록 원
    B = D["blocks"].copy()
    B["fill"] = [block_kind_fill(r) for r in B.itertuples()]
    main = B[B.fill.notna() & ~B.region.isin(["Tibet_LGD", "Tibet"])].copy()
    main = main.sort_values("n_loc_1km", ascending=False)
    fc, ec, lw = circle_style(main.fill)
    ax.scatter(main.lon.values, main.lat.values, s=size_pt2(main.n_loc_1km.values), facecolors=fc, edgecolors=ec,
               linewidths=lw, transform=ccrs.PlateCarree(), zorder=3)
    vals["a_blocks_drawn_main"] = int(len(main))
    vals["a_blocks_not_drawn_natl"] = int((B.kind == "natl_si").sum())
    # 지역 이름과 지시선(글자 위치는 원 중심 기준 반지름 비율 → 그림 mm)
    Pm = proj.transform_points(ccrs.PlateCarree(), main.lon.values, main.lat.values)
    main = main.assign(px=Pm[:, 0], py=Pm[:, 1])
    cx_mm, cy_mm, r_mm = x + dia / 2, y + dia / 2, dia / 2
    for reg, (txt, (fx_, fy_)) in REGION_LABELS.items():
        L0 = fig_mm_to_data(fig, ax, cx_mm + fx_ * r_mm, cy_mm + fy_ * r_mm)
        q = main[main.region == reg]
        if LEADER_ANCHOR.get(reg) == "largest":
            k = int(np.argmax(q.n_loc_1km.values))
        else:
            k = int(np.argmin(np.hypot(q.px.values - L0[0], q.py.values - L0[1])))
        r_pt = np.sqrt(size_pt2(q.n_loc_1km.values[k])) / 2 + EDGE_LW / 2
        t = ax.text(L0[0], L0[1], txt, ha="center", va="center", fontsize=S.FONT_PT, zorder=7, clip_on=False)
        t.set_gid("region_label")
        leader_line(fig, ax, t, (q.px.values[k], q.py.values[k]), r_pt + 0.5)
    # 축척 막대(70° N 근처)
    vals["a_scale"] = scale_bar(ax, proj, SCALE_A["lon"], SCALE_A["lat"], SCALE_A["km"])
    ax.set_gid("map_a")
    return ax, proj


def zoom_rectangle(ax_a, proj_a, proj_z, ext):
    """c·d 범위(확대 투영의 정사각형)를 a 의 투영으로 옮겨 그린다(1.0 pt 검정). 꼭짓점 4개를 a 투영 좌표로 돌려준다."""
    x0, x1, y0, y1 = ext
    n = 20
    xs = np.r_[np.linspace(x0, x1, n), np.full(n, x1), np.linspace(x1, x0, n), np.full(n, x0)]
    ys = np.r_[np.full(n, y0), np.linspace(y0, y1, n), np.full(n, y1), np.linspace(y1, y0, n)]
    Q = proj_a.transform_points(proj_z, xs, ys)
    import matplotlib.patheffects as pe
    ln = ax_a.plot(Q[:, 0], Q[:, 1], color=S.INK, lw=1.0, zorder=6, solid_joinstyle="miter",
                   path_effects=[pe.Stroke(linewidth=RECT_HALO_LW, foreground="white"), pe.Normal()])[0]
    ln.set_gid("zoom_rectangle")
    C = proj_a.transform_points(proj_z, np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1]))
    return C[:, :2]          # 아래 왼쪽, 아래 오른쪽, 위 오른쪽, 위 왼쪽(확대 투영 기준)


def connect(fig, ax_a, ax_c, corners_a):
    """a 의 사각형 아래 두 꼭짓점과 c 의 위 두 모서리를 1.0 pt 회색 직선으로 잇는다(지침 2.11).
    선은 a 의 자료 좌표로 그리고 잘라내지 않는다(clip_on=False). zorder 를 블록 원 아래로 두어 원을 가리지 않는다."""
    fig.canvas.draw()
    disp = ax_a.transData.transform(corners_a)
    order = np.argsort(-disp[:, 1])[2:]                     # 화면에서 아래쪽 두 꼭짓점(표시 좌표 y 가 작은 쪽)
    order = order[np.argsort(disp[order, 0])]               # 왼쪽, 오른쪽
    inv = ax_a.transData.inverted()
    for k, xb in zip(order, (0.0, 1.0)):
        end = inv.transform(ax_c.transAxes.transform((xb, 1.0)))
        ln = ax_a.plot([corners_a[k, 0], end[0]], [corners_a[k, 1], end[1]], color=S.BASEMAP["coast"], lw=1.0, zorder=2.5,
                       clip_on=False, solid_capstyle="butt")[0]
        ln.set_gid("zoom_connector")


def panel_tibet(fig, D, vals, P, ax_a=None, proj_a=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    B = D["blocks"]
    q = B[B.region.isin(["Tibet_LGD", "Tibet"])].copy()
    q["fill"] = [block_kind_fill(r) for r in q.itertuples()]
    lon_c, lat_c = float((q.lon.min() + q.lon.max()) / 2), float((q.lat.min() + q.lat.max()) / 2)
    proj = ccrs.LambertAzimuthalEqualArea(central_longitude=lon_c, central_latitude=lat_c)
    x, y, w, h = TIBET_AX
    ax = S.axes_mm(fig, x, y, w, h, projection=proj)
    Pq = proj.transform_points(ccrs.PlateCarree(), q.lon.values, q.lat.values)
    m_per_mm_guess = (Pq[:, 0].max() - Pq[:, 0].min() + 2 * 160e3) / w
    r_m = np.sqrt(size_pt2(q.n_loc_1km.values)) / 2 / 72 * 25.4 * m_per_mm_guess     # 원 반지름(투영 m)
    # 범위: 가로는 블록 범위 + 양쪽 160 km, 세로는 틀 비율로 정하고 블록을 위쪽에 두어 아래에 축척 막대 자리를 남긴다 [판단]
    cx = (Pq[:, 0].min() + Pq[:, 0].max()) / 2
    half_w = (Pq[:, 0].max() - Pq[:, 0].min()) / 2 + 160e3
    half_h = half_w * h / w
    y_top = (Pq[:, 1] + r_m).max() + 110e3
    y_bot = y_top - 2 * half_h
    if y_bot > (Pq[:, 1] - r_m).min() - 60e3:                    # 블록이 아래로 넘치면 가운데 맞춤으로 되돌린다
        cy = (Pq[:, 1].min() + Pq[:, 1].max()) / 2
        y_top, y_bot = cy + half_h, cy - half_h
    ax.set_xlim(cx - half_w, cx + half_w); ax.set_ylim(y_bot, y_top)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=S.BASEMAP["land"], edgecolor="none", linewidth=0, zorder=0).set_rasterized(True)
    vals["tibet_pfr_raster"] = draw_pfr(ax, proj, w, h, P)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=S.BASEMAP["coast"], facecolor="none", linewidth=S.LW["coast"], zorder=0.4)
    ax.spines["geo"].set_edgecolor(S.BASEMAP["coast"]); ax.spines["geo"].set_linewidth(1.0)
    q = q.sort_values("n_loc_1km", ascending=False)
    fc, ec, lw = circle_style(q.fill)
    ax.scatter(q.lon.values, q.lat.values, s=size_pt2(q.n_loc_1km.values), facecolors=fc, edgecolors=ec, linewidths=lw,
               transform=ccrs.PlateCarree(), zorder=3)
    t = ax.text(0.0, 1.0, "Tibetan Plateau", transform=ax.transAxes, ha="left", va="bottom", fontsize=S.FONT_PT, zorder=6)
    t.set_gid("region_label")
    # 자체 축척 막대. 막대와 길이 글자의 상자가 어떤 원(블록)과도 겹치지 않는 자리를 아래 구석부터 찾는다.
    # 등적 방위 투영 중심 부근이라 축척 계수 1 로 둔다. 글자 상자는 7 pt(높이 약 2.5 mm, "500 km" 폭 약 5.5 mm).
    km = 500
    W2, H2 = 2 * half_w, y_top - y_bot
    m_per_mm = W2 / w
    L = km * 1e3
    t = ax.text(cx, y_bot, f"{km} km", ha="center", va="bottom", fontsize=S.FONT_PT, zorder=8)   # 실측용, 자리는 아래에서 정한다
    fig.canvas.draw()
    bbt = t.get_window_extent(fig.canvas.get_renderer())
    txt_w, txt_h = bbt.width / fig.dpi * 25.4 * m_per_mm, bbt.height / fig.dpi * 25.4 * m_per_mm
    gap = 0.8 * m_per_mm
    Pq2 = proj.transform_points(ccrs.PlateCarree(), q.lon.values, q.lat.values)
    r_m = (np.sqrt(size_pt2(q.n_loc_1km.values)) / 2 + EDGE_LW / 2) / 72 * 25.4 * m_per_mm
    best = None
    for fy in (0.10, 0.16, 0.22, 0.28):
        for fx in np.arange(0.04, 0.72, 0.03):
            xb0 = cx - half_w + fx * W2; yb = y_bot + fy * H2
            box = (min(xb0, xb0 + L / 2 - txt_w / 2) - gap, max(xb0 + L, xb0 + L / 2 + txt_w / 2) + gap, yb - gap, yb + txt_h + gap)
            hit = int(((Pq2[:, 0] + r_m > box[0]) & (Pq2[:, 0] - r_m < box[1]) & (Pq2[:, 1] + r_m > box[2]) & (Pq2[:, 1] - r_m < box[3])).sum())
            cost = (hit, fy, fx)
            if best is None or cost < best[0]:
                best = (cost, xb0, yb)
            if hit == 0:
                break
        if best[0][0] == 0:
            break
    _, xb0, yb = best
    vals["tibet_scale_overlap_blocks"] = int(best[0][0])
    ax.plot([xb0, xb0 + L], [yb, yb], color=S.INK, lw=S.LW["scale_bar"], solid_capstyle="butt", zorder=8)
    t.set_position((xb0 + L / 2, yb))
    t.set_gid("scale")
    ll = ccrs.PlateCarree().transform_points(proj, np.array([xb0, xb0 + L]), np.array([yb, yb]))
    vals["tibet_scale"] = dict(km=km, geodesic_km=float(haversine(ll[0, 1], ll[0, 0], ll[1, 1], ll[1, 0])))
    vals["tibet_blocks_drawn"] = int(len(q))
    vals["tibet_blocks_below_frame"] = int(((Pq2[:, 1] - r_m) < y_bot).sum() + ((Pq2[:, 1] + r_m) > y_top).sum()
                                           + ((Pq2[:, 0] - r_m) < cx - half_w).sum() + ((Pq2[:, 0] + r_m) > cx + half_w).sum())
    vals["tibet_km_per_mm"] = round(m_per_mm / 1000.0, 1)
    # 삽도 틀이 본 지도(a)의 블록 원을 가리는지: a 의 주 지도 블록 가운데 삽도 틀(그림 mm 상자) 안에 드는 원의 수
    if ax_a is not None and proj_a is not None:
        fig.canvas.draw()
        mainb = B[~B.region.isin(["Tibet_LGD", "Tibet"]) & (B.kind != "natl_si")]
        Pa = proj_a.transform_points(ccrs.PlateCarree(), mainb.lon.values, mainb.lat.values)
        disp = ax_a.transData.transform(Pa[:, :2])
        Wf, Hf = fig.get_size_inches() * S.MM_PER_IN
        xm = disp[:, 0] / fig.dpi * 25.4; ym = Hf - disp[:, 1] / fig.dpi * 25.4
        ra = pt2mm(np.sqrt(size_pt2(mainb.n_loc_1km.values)) / 2 + EDGE_LW / 2)
        hid = (xm + ra > x) & (xm - ra < x + w) & (ym + ra > y - 2.8) & (ym - ra < y + h)      # 틀 + 위 이름 줄
        vals["tibet_inset_hides_main_blocks"] = int(hid.sum())
    ax.set_gid("inset_tibet")
    return ax


def key_a(fig):
    """a 의 열쇠 2줄(지침 2.8): 줄 1 = 크기 열쇠(1, 10, 50, 아래끝 맞춤의 나란한 원) + 새 지역 견본, 줄 2 = 영구동토 견본 2개.
    명세의 '3단계 중첩 원'은 원 지름(0.8, 2.5, 5.6 mm)이 작아 값 글자가 겹쳐 나란한 원으로 바꿨다 [판단]."""
    x0, y0, w, h = KEY_AX
    ax = S.axes_mm(fig, x0, y0, w, h)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_axis_off(); ax.patch.set_visible(False)   # y 는 아래로(mm)
    rend = fig.canvas.get_renderer()

    def put_text(x, y, s_, gid=None):
        t = ax.text(x, y, s_, ha="left", va="center", fontsize=S.FONT_PT)
        if gid:
            t.set_gid(gid)
        fig.canvas.draw()
        bb = t.get_window_extent(rend).transformed(ax.transData.inverted())
        return max(bb.x0, bb.x1)

    base = 6.2                                                     # 줄 1 의 원 아래끝(mm)
    x = 0.4
    for v in SIZE_KEY:
        r = pt2mm(np.sqrt(size_pt2(v)) / 2)
        ax.scatter([x + r], [base - r], s=size_pt2(v), facecolors=[KEY_FILL], edgecolors=KEY_EDGE, linewidths=REGION_EDGE_LW, zorder=3)
        x = put_text(x + 2 * r + 0.7, base - 1.2, f"{v}", gid="sizekey") + 1.8
    r = pt2mm(np.sqrt(size_pt2(SIZE_KEY[1])) / 2)
    x += 0.8
    ax.scatter([x + r], [base - r], s=size_pt2(SIZE_KEY[1]), facecolors="white", edgecolors=S.INK, linewidths=EDGE_LW, zorder=3)
    x_end1 = put_text(x + 2 * r + 0.7, base - 1.2, "New regions")
    # 줄 2 = 영구동토 머리, 줄 3 = 구역 견본 2개(v4 토큰 map_base.rule: 열쇠에 구역 이름을 적는다)
    y2, sw = 9.4, 2.4
    put_text(0.4, y2, "Permafrost zone", gid="key")
    y3 = 12.6
    x = 0.4
    for lab, col in (("Continuous", S.BASEMAP["continuous"]), ("Discontinuous", S.BASEMAP["discontinuous"])):
        ax.add_patch(Rectangle((x, y3 - sw / 2), sw, sw, facecolor=col, edgecolor="none", linewidth=0, zorder=2))
        x = put_text(x + sw + 0.8, y3, lab) + 2.2
    ax.set_gid("key_a")
    return ax, dict(row1_end_mm=round(x_end1, 1), row2_end_mm=round(x - 2.2, 1), width_mm=w)


def panel_b(fig, D, vals):
    zs = D["zs"].set_index("row"); zp = D["zp"]
    ax = S.axes_mm(fig, *B_AX)
    ax.set_xscale("log")
    rng = np.random.default_rng(JITTER_SEED)
    ytop = max(y for _, y in B_ROWS)
    rows_out = []
    for key, y0 in B_ROWS:
        yy = ytop - y0
        r = zs.loc[key]
        E = np.exp(zp[zp.row == key].z.values)
        jit = rng.uniform(0.14, 0.66, len(E))
        mk = TIBET_MARKER if key == "Tibet_LGD" else S.REGION_MARKER[key]
        _rn = S.REGION_NAME.get(key, key).split("\n")[0].replace("(expanded)", "").strip()
        rcol = S.REGION_COLOR.get(_rn, S.INK)                                # v4: 지역이 행(주 부호화)이라 지역 색
        ax.scatter(E, yy + jit, s=S.MS["region_point"] ** 2, marker=mk, color=rcol, alpha=0.55, linewidths=0, zorder=2,
                   rasterized=True)                                        # 밀집 점 층(최대 400 × 7): PDF 크기(600 dpi 래스터)
        q25, q50, q75 = np.exp(r.z_q25), np.exp(r.z_q50), np.exp(r.z_q75)
        ax.plot([q25, q75], [yy, yy], color=S.INK, lw=S.LW["ci_forest_cell"], solid_capstyle="round", zorder=4)
        ax.plot([q50], [yy], "o", ms=S.MS["main"], color=S.INK, mew=0, zorder=5)
        ax.plot([r.E0, r.E0], [yy - 0.22, yy + 0.72], color=S.METHOD["source_stefan"]["color"], lw=S.LW["main"], solid_capstyle="butt",
                zorder=3)                                            # 원천 계수 = 원천 계수 Stefan 의 색(v4)
        rows_out.append(dict(row=key, y=yy, n_cells=int(r.n_cells), n_points=int(len(E)), E_q25=q25, E_q50=q50, E_q75=q75,
                             E0=float(r.E0), E_min_shown=float(E.min()), E_max_shown=float(E.max())))
    ax.set_yticks([ytop - y0 for _, y0 in B_ROWS])
    ax.set_yticklabels([S.REGION_NAME[k] for k, _ in B_ROWS])
    for t in ax.get_yticklabels():
        t.set_gid("category")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.45, ytop + 1.45)
    ax.set_xlim(*E_LIM)
    ax.xaxis.set_major_locator(mticker.FixedLocator(E_TICKS))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_xlabel("Stefan coefficient, $E$ (cm per √(°C d))")
    # 원천 계수 직접 라벨(첫 행 위 한 번)
    r0 = zs.loc[B_ROWS[0][0]]
    t = ax.text(r0.E0, ytop + 0.80, "Source coefficient", ha="center", va="bottom", fontsize=S.FONT_PT, color=S.METHOD["source_stefan"]["color"])
    t.set_gid("direct_label")
    ax.plot([r0.E0, r0.E0], [ytop + 0.72, ytop + 0.80], color=S.METHOD["source_stefan"]["color"], lw=S.LW["main"], solid_capstyle="butt", zorder=3)
    out_lo = int((np.exp(zp[zp.row.isin([k for k, _ in B_ROWS])].z) < E_LIM[0]).sum())
    out_hi = int((np.exp(zp[zp.row.isin([k for k, _ in B_ROWS])].z) > E_LIM[1]).sum())
    vals["b_points_outside_axis"] = out_lo + out_hi
    vals["b_rows"] = rows_out
    return ax


def zoom_base(fig, rect, proj, ext):
    import cartopy.feature as cfeature
    ax = S.axes_mm(fig, *rect, projection=proj)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor=S.BASEMAP["land"], edgecolor="none", linewidth=0, zorder=0).set_rasterized(True)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor=S.BASEMAP["coast"], facecolor="none", linewidth=S.LW["coast"], zorder=0.4)
    ax.spines["geo"].set_edgecolor(S.BASEMAP["coast"]); ax.spines["geo"].set_linewidth(1.0)
    return ax


def panel_c(fig, L, proj, ext, vals):
    import cartopy.crs as ccrs
    from sklearn.neighbors import BallTree
    ax = zoom_base(fig, C_AX, proj, ext)
    d = L["df"].iloc[L["t_idx"]]
    ax.scatter(d.lon.values, d.lat.values, s=2.0 ** 2, color=S.INK, linewidths=0, transform=ccrs.PlateCarree(), zorder=3,
               rasterized=True)                                            # 밀집 점 층(대상 셀 3,037)
    # 100 km 거리 등치선(대상 셀 위치 합집합의 측지 거리, 구면 R 6371 km)
    ng = 360
    X, Y = np.meshgrid(np.linspace(ext[0], ext[1], ng), np.linspace(ext[2], ext[3], ng))
    ll = ccrs.PlateCarree().transform_points(proj, X, Y)
    uq = d[["lat", "lon"]].drop_duplicates().values
    tree = BallTree(np.radians(uq), metric="haversine")
    dist, _ = tree.query(np.radians(np.c_[ll[..., 1].ravel(), ll[..., 0].ravel()]), k=1)
    Dkm = dist[:, 0].reshape(X.shape) * 6371.0
    cs = ax.contour(X, Y, Dkm, levels=[BUFFER_KM], colors=[S.BASEMAP["buffer_line"]], linewidths=[1.0], linestyles=[(0, (3, 2))], zorder=4)
    vals["c_contour_paths"] = int(sum(len(p.vertices) > 0 for p in cs.get_paths()))
    # 라벨 "Buffer": 등치선 위 가장 북서쪽 점에서 안쪽(오른쪽 아래)으로. 패널 밖으로 나가지 않는다
    verts = np.vstack([p.vertices for p in cs.get_paths() if len(p.vertices)])
    k = int(np.argmax(-verts[:, 0] + verts[:, 1]))
    t = ax.annotate("Buffer", xy=verts[k], xycoords="data", xytext=(5, -5), textcoords="offset points", ha="left", va="top",
                    fontsize=S.FONT_PT, zorder=7)
    t.set_gid("direct_label")
    vals["c_scale"] = scale_bar_zoom(ax, proj, ext, 100)
    # 범위 안 원천 셀(버퍼 안이면 원천에서 빠진 셀): 그리지 않고 기록만 한다
    s = L["df"].iloc[L["src_idx"]]
    Ps = proj.transform_points(ccrs.PlateCarree(), s.lon.values, s.lat.values)
    inside = (Ps[:, 0] >= ext[0]) & (Ps[:, 0] <= ext[1]) & (Ps[:, 1] >= ext[2]) & (Ps[:, 1] <= ext[3])
    vals["c_source_cells_in_extent"] = int(inside.sum())
    vals["c_source_cells_in_extent_detail"] = ";".join(f"{r}@{la:.3f},{lo:.3f}(d={dm:.1f} km)" for r, la, lo, dm in
                                                       zip(s.region.values[inside], s.lat.values[inside], s.lon.values[inside],
                                                           L["src_dmin"][inside]))
    return ax


def scale_bar_zoom(ax, proj, ext, km):
    """확대도 왼쪽 아래 축척 막대. 막대 위도의 축척 계수를 적용한다."""
    import cartopy.crs as ccrs
    x0 = ext[0] + 0.07 * (ext[1] - ext[0]); y0 = ext[2] + 0.07 * (ext[3] - ext[2])
    lon, lat = ccrs.PlateCarree().transform_point(x0, y0, proj)
    k = (1 + np.sin(np.radians(TRUE_LAT))) / (1 + np.sin(np.radians(lat)))
    L = km * 1e3 * k
    ax.plot([x0, x0 + L], [y0, y0], color=S.INK, lw=S.LW["scale_bar"], solid_capstyle="butt", zorder=8)
    t = ax.text(x0 + L / 2, y0, f"{km} km", ha="center", va="bottom", fontsize=S.FONT_PT, zorder=8)
    t.set_gid("scale")
    ll = ccrs.PlateCarree().transform_points(proj, np.array([x0, x0 + L]), np.array([y0, y0]))
    return dict(km=km, k=float(k), geodesic_km=float(haversine(ll[0, 1], ll[0, 0], ll[1, 1], ll[1, 0])))


def block_polys(blocks, proj, n=12):
    """0.5° 블록(id = floor(lat/0.5)·100000 + floor(lon/0.5))의 경계를 확대 투영 다각형으로."""
    import cartopy.crs as ccrs
    polys, corners = [], []
    for b in blocks:
        a = int(np.round(b / 100000.0)); c = int(b - a * 100000)
        la0, lo0 = a * 0.5, c * 0.5
        lo = np.r_[np.linspace(lo0, lo0 + 0.5, n), np.full(n, lo0 + 0.5), np.linspace(lo0 + 0.5, lo0, n), np.full(n, lo0)]
        la = np.r_[np.full(n, la0), np.linspace(la0, la0 + 0.5, n), np.full(n, la0 + 0.5), np.linspace(la0 + 0.5, la0, n)]
        Q = proj.transform_points(ccrs.PlateCarree(), lo, la)
        polys.append(Q[:, :2]); corners.append((la0, lo0))
    return polys, corners


def panel_d(fig, L, proj, ext, vals):
    import cartopy.crs as ccrs
    ax = zoom_base(fig, D_AX, proj, ext)
    lab_b = [b for b in L["A_blocks"]]
    sco_b = [b for b in L["score_blocks"]]
    pa, ca = block_polys(lab_b, proj)
    pb, cb = block_polys(sco_b, proj)
    # 블록 테두리: 흰색(v4 map_base 로 육지가 #E6E6E6 이 되어 회색 경위선 색 테두리가 라벨 블록과 구분되지 않는다)
    ax.add_collection(PolyCollection(pa, facecolors=S.BASEMAP["label_block"], edgecolors="white", linewidths=0.8, zorder=2))
    ax.add_collection(PolyCollection(pb, facecolors=S.BASEMAP["score_block"], edgecolors="white", linewidths=0.8, zorder=2))
    lab = L["lab"]
    ax.scatter(lab.lon.values, lab.lat.values, s=3.0 ** 2, color=S.INK, linewidths=0, transform=ccrs.PlateCarree(), zorder=4)
    # 블록 경도 폭(최종 크기 mm), 지침 2.11 의 1.0 mm 조건
    lat_mid = float(L["df"].iloc[L["t_idx"]].lat.mean())
    km_lon = 0.5 * 111.32 * np.cos(np.radians(lat_mid))
    km_per_mm = (ext[1] - ext[0]) / 1000.0 / D_AX[2]
    vals["d_block_lon_width_mm"] = round(km_lon / km_per_mm, 2)
    vals["d_extent_km"] = round((ext[1] - ext[0]) / 1000.0, 1)
    # 직접 라벨 2개: 패널 구석의 글자에서 같은 종류 블록 안쪽 점으로 지시선(1.0 pt #4d4d4d). 지시선은 글자 상자의 실제 아래(위) 가운데에서
    # 시작하고, 다른 종류 블록을 가로지르지 않으며, 라벨 셀 점(검정 3.0 pt)에서 1.5 mm 안으로 지나지 않는다. 2차 렌더에서 annotate 화살표의
    # 실제 시작점과 교차 검사 시작점이 달라 지시선이 채점 블록을 가로질렀고, 채점 블록 지시선 끝이 라벨 셀 점과 붙어 보여 고쳤다.
    W = ext[1] - ext[0]
    m_per_mm = W / D_AX[2]
    allp = [mpath.Path(np.vstack([p_, p_[:1]])) for p_ in pa + pb]
    kinds = ["A"] * len(pa) + ["B"] * len(pb)
    Pl = proj.transform_points(ccrs.PlateCarree(), lab.lon.values, lab.lat.values)[:, :2]

    def seg_point_dist(p0, p1, Q):
        d = p1 - p0
        t = np.clip(((Q - p0) @ d) / (d @ d), 0, 1)
        return np.hypot(*(Q - (p0 + t[:, None] * d)).T)

    def leader(start, kind):
        """비용 최소 끝점: 다른 종류 블록 가로지름(1000), 같은 종류의 다른 블록 가로지름(300, 끝점 블록이 아닌 블록 위를 지나면
        '중간' 블록을 가리키는 것처럼 보인다), 라벨 셀 점 1.5 mm 안 통과(100), 끝점이 라벨 셀 점 3 mm 안(50), 길이(mm)."""
        best = None
        for j, (pp, kd) in enumerate(zip(allp, kinds)):
            if kd != kind:
                continue
            V = pp.vertices[:-1]
            cands = [V.mean(0)] + [V.mean(0) * (1 - f) + v * f for v in V[:: max(1, len(V) // 16)] for f in (0.35, 0.6)]
            for end in cands:
                seg = mpath.Path(np.array([start, end]))
                viol = 0
                if any(o.intersects_path(seg, filled=True) for i, o in enumerate(allp) if kinds[i] != kind):
                    viol += 1000
                if any(o.intersects_path(seg, filled=True) for i, o in enumerate(allp) if kinds[i] == kind and i != j):
                    viol += 300
                if seg_point_dist(start, end, Pl).min() < 1.5 * m_per_mm:
                    viol += 100
                if np.hypot(*(Pl - end).T).min() < 3.0 * m_per_mm:
                    viol += 50
                cost = viol + float(np.hypot(*(end - start))) / m_per_mm
                if best is None or cost < best[0]:
                    best = (cost, end, viol)
        return best

    def labelled_leader(text, positions, ha, kind, side):
        """글자 위치 후보(패널 비율)를 차례로 시험해 위반 0 인 첫 자리를 쓴다. 모두 위반이면 비용 최소 자리를 쓰고 위반을 기록한다."""
        trials = []
        for (fx_, fy_) in positions:
            pos = np.array([ext[0] + fx_ * W, ext[3] - fy_ * W])
            t = ax.text(pos[0], pos[1], text, ha=ha, va="center", fontsize=S.FONT_PT, zorder=7)
            fig.canvas.draw()
            bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
            cx = (bb.x0 + bb.x1) / 2
            start = np.array([cx, bb.y0 - 0.5 * m_per_mm]) if side == "below" else np.array([cx, bb.y1 + 0.5 * m_per_mm])
            cost, end, viol = leader(start, kind)
            g = 0.5 * m_per_mm                                                  # 글자 상자가 블록이나 라벨 셀 점을 덮으면 위반(500)
            rect = mpath.Path([(bb.x0 - g, bb.y0 - g), (bb.x1 + g, bb.y0 - g), (bb.x1 + g, bb.y1 + g), (bb.x0 - g, bb.y1 + g),
                               (bb.x0 - g, bb.y0 - g)])
            if any(o.intersects_path(rect, filled=True) for o in allp) or rect.contains_points(Pl).any():
                viol += 500; cost += 500
            trials.append((cost, viol, t, start, end))
            if viol == 0:
                break
            t.remove()
        if trials[-1][1] != 0:                                                  # 모두 위반: 비용 최소 자리를 다시 그린다
            cost, viol, t, start, end = min(trials, key=lambda z: z[0])
            if t.axes is None:
                t = ax.text(*t.get_position(), text, ha=ha, va="center", fontsize=S.FONT_PT, zorder=7)
        else:
            cost, viol, t, start, end = trials[-1]
        t.set_gid("direct_label")
        ln = ax.plot([start[0], end[0]], [start[1], end[1]], color=S.INK_AUX, lw=1.0, solid_capstyle="butt", zorder=6.5)[0]
        ln.set_gid("leader")
        vals[f"d_leader_{kind}"] = dict(violation=int(viol), length_mm=round(float(np.hypot(*(end - start))) / m_per_mm, 1),
                                        text_pos_tried=len(trials))
        return t

    labelled_leader("Label blocks", [(0.05, 0.07), (0.05, 0.22), (0.05, 0.40), (0.30, 0.07)], "left", "A", "below")
    labelled_leader("Scoring blocks", [(0.95, 0.93), (0.95, 0.80), (0.95, 0.66), (0.70, 0.93), (0.95, 0.55), (0.60, 0.93)], "right", "B", "above")
    return ax


XH_DIR = ROOT / "data" / "processed" / "xbatch" / "XH_validation_ladder" / "sealed"
XH_REGIONS = ("Alaska", "Lena", "Canada")
XH_STAGES = (("W1R", "Random"), ("W1S", "Site"), ("W1B", "Block"), ("W1K", "kNNDM"), ("V-G", "Region holdout"))
XH_METHODS = (("D0w", "catboost_lo", "direct_ml", "Direct ML"), ("PSw", "none", "recalibrated_stefan", "Recalibrated Stefan"))
XH_XOFF = {"D0w": 0.86, "PSw": 1.16}                                        # 지역 점의 방법별 가로 비율(로그 축, CI 막대가 겹치지 않게)
XH_XLIM, XH_YLIM = (0.002, 2000.0), (5.0, 50.0)
VG_LGV = ROOT / "data" / "processed" / "lgx" / "ladder" / "lgv_metrics.csv"          # 지역 홀드아웃(V-G) RMSE(h41, 계획 8.3)
VG_METHOD = {"D0w": ("D0", "catboost_lo", 1.0), "PSw": ("PS", "none", 0.0)}           # V-G 의 직접 ML, 학습 셀(원천) 최소제곱 계수 Stefan
REG7_H41 = ["Alaska", "Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland"]


def region_holdout_rows():
    """지역 홀드아웃(V-G) 행. 거리는 좌표만으로 h41 과 같은 정의로 계산한다(새 적합 없음):
    채점 셀 = 대상 지역 셀 ∩ eval_mask(h41 VData.score), 학습 셀 = h40.Data.source_idx(지역, 'x')(대상 제외, 100 km 버퍼) ∩ y·s 유한,
    거리 = 채점 셀마다 가장 가까운 학습 셀까지 대원 거리(cv_schemes.nnd_km, XH 와 같은 함수)의 중앙값. RMSE 는 lgv_metrics.csv 의 V-G 지역 행."""
    from polar.m1_core import eval_mask
    from polar import cv_schemes as CV
    h40 = load_h40()
    HA = h40.parse_args(["--threads", "1", "--splits", "5"])
    D = h40.get_data(HA)
    df = D.df
    score = np.asarray(eval_mask(df), bool) & np.isin(df.macro.values, REG7_H41)
    trainable = np.isfinite(df.y.values) & np.isfinite(df.s.values)
    lgv = pd.read_csv(VG_LGV)
    lgv = lgv[(lgv.scheme == "V-G") & (lgv.scope == "region")]
    rows, geo = [], {}
    for R in XH_REGIONS:
        t_idx, _, s_idx, comp = D.source_idx(R, "x")
        tr = np.asarray(s_idx, np.int64); tr = tr[trainable[tr]]
        te = np.asarray(t_idx, np.int64); te = te[score[te]]
        d = CV.nnd_km(CV.to_rad(df.lon.values[te], df.lat.values[te]), CV.to_rad(df.lon.values[tr], df.lat.values[tr]))
        geo[R] = dict(n_score=int(len(te)), n_train=int(len(tr)), n_buffer_excluded=int(comp["n_buffer_excluded"]), nnd_median_km=float(np.median(d)),
                      nnd_min_km=float(d.min()), buffer_km=float(HA.buffer_km))
        for mth, (m_lgv, lr, lam) in VG_METHOD.items():
            q = lgv[(lgv.target == R) & (lgv.method == m_lgv) & (lgv.learner == lr) & np.isclose(lgv.lam.astype(float), lam)]
            assert len(q) == 1, (R, mth, len(q))
            q = q.iloc[0]
            rows.append(dict(region=R, variant=np.nan, stage="V-G", method=mth, learner=lr, lam=lam, rmse=q.rmse, rmse_lo=q.rmse_lo,
                             rmse_hi=q.rmse_hi, rmse_beq=q.rmse_beq, rmse_beq_lo=q.rmse_beq_lo, rmse_beq_hi=q.rmse_beq_hi, n_rows=np.nan,
                             n_blocks=q.n_blocks, nnd_median_km=geo[R]["nnd_median_km"], source=f"lgv_metrics.csv V-G {m_lgv}|{lr}",
                             n_cells_lgv=int(q.n_cells)))
    return pd.DataFrame(rows), geo


def load_xh():
    """봉인 해제된 xh_ladder.csv(계획 8.3)에서 두 방법 × 네 단 × 세 지역 24행. variant 가 비어 있는 행만(캐나다 W1K 'pm2' 제외).
    지역 홀드아웃(V-G) 6행을 region_holdout_rows 로 더한다(30행)."""
    d = pd.read_csv(XH_DIR / "xh_ladder.csv")
    keep = d.variant.isna() & d.region.isin(XH_REGIONS) & d.stage.isin([s for s, _ in XH_STAGES if s != "V-G"])
    sel = ((d.method == "D0w") & (d.learner == "catboost_lo") & np.isclose(d.lam.astype(float), 1.0)) | ((d.method == "PSw") & (d.learner == "none"))
    x = d[keep & sel].copy()
    assert len(x) == 24, len(x)
    vg, geo = region_holdout_rows()
    x = pd.concat([x, vg], ignore_index=True)
    m = (x.groupby(["method", "stage"]).agg(rmse_mean=("rmse", "mean"), n_reg=("region", "nunique"),
                                            x_geo=("nnd_median_km", lambda v: float(np.exp(np.mean(np.log(np.asarray(v, float))))))).reset_index())
    assert (m.n_reg == 3).all()
    return x, m, dict(n_pm2_excluded=int((d.variant == "pm2").sum()), file=str((XH_DIR / "xh_ladder.csv").relative_to(ROOT)), vg_geo=geo,
                      vg_file=str(VG_LGV.relative_to(ROOT)))


def panel_e(fig, vals):
    """검증 사다리(XH, 계획 8.3). x = 채점 셀에서 가장 가까운 학습 셀까지 거리 중앙값(log10), y = 셀 가중 RMSE.
    선과 큰 점: 세 지역 등가중 평균(x 는 세 지역 거리의 기하 평균). 작은 기호: 지역 값(2.5 pt, alpha 0.5)과 셀 가중 95 % 블록 재표집 CI.
    표에 평균 행의 CI 는 없다. 블록 등가중 값과 CI 는 원천 자료에만 둔다(겹침). 지역 홀드아웃(V-G)은 lgv_metrics.csv 의 RMSE 와 좌표만으로
    다시 계산한 거리(region_holdout_rows)로 그린다. 그 단의 Stefan 은 원천 계수라 빈 기호와 점선으로 구분한다."""
    X, Mn, info = load_xh()
    ax = S.axes_mm(fig, *E_AX)
    ax.set_xscale("log")
    stages = [s_ for s_, _ in XH_STAGES]
    for mth, lr, key, name in XH_METHODS:
        st = S.METHOD[key]
        r = X[X.method == mth]
        src_open = mth == "PSw"                                             # 지역 홀드아웃의 Stefan 은 원천 계수: 빈 기호
        for reg in XH_REGIONS:
            q = r[r.region == reg].set_index("stage").reindex(stages)
            xx = q.nnd_median_km.values * XH_XOFF[mth]
            ax.vlines(xx, q.rmse_lo.values, q.rmse_hi.values, color=st["color"], lw=S.LW["aux"], alpha=0.4, zorder=2)   # 지역 CI: 평균 선이 앞서도록 가늘게
            for j, s_ in enumerate(stages):
                op = src_open and s_ == "V-G"
                ax.plot([xx[j]], [q.rmse.values[j]], ls="none", marker=S.REGION_MARKER[reg], ms=S.MS["region_point"],
                        mfc="white" if op else st["color"], mec=st["color"], mew=S.LW["aux"] if op else 0, alpha=0.5, zorder=3)
        mm = Mn[Mn.method == mth].set_index("stage").reindex(stages)
        xs_, ys_ = mm.x_geo.values, mm.rmse_mean.values
        if src_open:
            ax.plot(xs_[:-1], ys_[:-1], color=st["color"], ls=st["ls"], lw=S.LW["main"], zorder=4)
            ax.plot(xs_[-2:], ys_[-2:], color=st["color"], ls=(0, (1.2, 1.4)), lw=S.LW["main"], zorder=4)   # 원천 계수로 바뀌는 구간: 점선
            ax.plot(xs_[:-1], ys_[:-1], ls="none", marker="o", ms=S.MS["main"], color=st["color"], mew=0, zorder=5)
            ax.plot(xs_[-1:], ys_[-1:], ls="none", marker="o", ms=S.MS["main"], mfc="white", mec=st["color"], mew=S.LW["main"], zorder=5)
        else:
            ax.plot(xs_, ys_, color=st["color"], ls=st["ls"], lw=S.LW["main"], zorder=4)
            ax.plot(xs_, ys_, ls="none", marker="o", ms=S.MS["main"], color=st["color"], mew=0, zorder=5)
    ax.set_xlim(*XH_XLIM); ax.set_ylim(*XH_YLIM)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.01, 0.1, 1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["0.01", "0.1", "1", "10", "100", "1000"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.FixedLocator([10, 20, 30, 40, 50]))
    ax.set_xlabel("Distance to nearest training cell (km)")
    ax.set_ylabel("RMSE (cm)")
    # 단 이름(5개): 축 아래쪽 띠. 오른쪽 세 단(Block, kNNDM, Region holdout)은 간격이 좁아 두 높이로 나눈다
    xg = Mn[Mn.method == "D0w"].set_index("stage").x_geo
    pos = {"W1R": (1.0, 7.0, "center"), "W1S": (1.0, 7.0, "center"), "W1B": (1.12, 7.0, "right"), "W1K": (1.0, 11.6, "center"),
           "V-G": (None, 7.0, "right")}                                      # V-G: 축 오른쪽 끝에 맞춘 두 줄. kNNDM 은 점 아래 가운데(두 줄 사이 높이)
    for stg, lab in XH_STAGES:
        fx, y_, ha = pos[stg]
        x_ = XH_XLIM[1] * 0.97 if fx is None else xg[stg] * fx
        t = ax.text(x_, y_, lab.replace(" ", "\n") if stg == "V-G" else lab, ha=ha, va="center", fontsize=S.FONT_PT, zorder=6,
                    linespacing=1.0, multialignment="center")
        t.set_gid("direct_label")
    # 방법 이름: 무작위 단과 지점 단 사이 빈 곳(점이 없는 거리 0.02–0.6 km)
    t = ax.text(0.15, 12.0, "Direct ML", ha="center", va="center", fontsize=S.FONT_PT, color=S.METHOD["direct_ml"]["color"], zorder=6)
    t.set_gid("direct_label")
    t = ax.text(0.15, 29.0, "Recalibrated\nStefan", ha="center", va="center", fontsize=S.FONT_PT, color=S.METHOD["recalibrated_stefan"]["color"],
                zorder=6, linespacing=1.0)
    t.set_gid("direct_label")
    ax.set_gid("XH_ladder")
    vals["xh"] = dict(rows=X, means=Mn, info=info)
    return ax


# ================================================================ 조립
def build(medium="paper"):
    if medium != "paper":
        raise NotImplementedError("슬라이드판 Fig 1 배치는 아직 정하지 않았다(그림 명세 1.7). 같은 자료로 따로 짠다")
    S.use_v3(medium)
    if hasattr(S, "mathtext_liberation"):
        S.mathtext_liberation()                      # b 의 "$E$" 기울임을 Liberation Sans Italic 으로(findfont 경고 없이)
    vals = dict()
    D = load_all()
    P = pfr_classes()
    vals["pfr_title"] = P[3]
    L = lena_design(vals)
    proj_z, ext, lon_c, lat_c = zoom_projection(L)
    vals.update(zoom_center_lon=round(lon_c, 3), zoom_lat_mean=round(lat_c, 3))
    fig = S.fig_mm(FIG_W, FIG_H)
    ax_a, proj_a = panel_a(fig, D, vals, P, lon_c if LON0_A is None else LON0_A)
    ax_t = panel_tibet(fig, D, vals, P, ax_a, proj_a)
    _, vals["key_a"] = key_a(fig)
    ax_b = panel_b(fig, D, vals)
    ax_c = panel_c(fig, L, proj_z, ext, vals)
    ax_d = panel_d(fig, L, proj_z, ext, vals)
    ax_e = panel_e(fig, vals)
    corners = zoom_rectangle(ax_a, proj_a, proj_z, ext)
    connect(fig, ax_a, ax_c, corners)
    for let, x, y in LETTERS:
        S.panel_letter(fig, x, y, let)
    return fig, dict(D=D, L=L, vals=vals, axes=dict(a=ax_a, t=ax_t, b=ax_b, c=ax_c, d=ax_d, e=ax_e), ext=ext, proj_z=proj_z)


# ================================================================ 기록
def source_data(ctx):
    D, L = ctx["D"], ctx["L"]
    rows = []
    B = D["blocks"].copy()
    for r in B.itertuples():
        f = block_kind_fill(r)
        rows.append(dict(panel="a" if r.region not in ("Tibet_LGD", "Tibet") else "a_inset", element="block_circle", region=r.region,
                         kind=r.kind, block=int(r.block), lat=r.lat, lon=r.lon, value=int(r.n_loc_1km), unit="1 km label locations",
                         detail=("not drawn (SI only)" if f is None else f"fill={f}")))
    zs = D["zs"].set_index("row")
    for key, _ in B_ROWS:
        r = zs.loc[key]
        for nm, v in (("E_q25", np.exp(r.z_q25)), ("E_median", np.exp(r.z_q50)), ("E_q75", np.exp(r.z_q75)), ("E0_source", r.E0),
                      ("n_cells", r.n_cells)):
            rows.append(dict(panel="b", element=nm, region=key, value=float(v), unit="cm per sqrt(degC d)" if nm != "n_cells" else "cells",
                             detail="quartiles and median over all cells; E0 = source-pool least-squares coefficient"))
        for z in D["zp"][D["zp"].row == key].z.values:
            rows.append(dict(panel="b", element="cell_E_shown", region=key, value=float(np.exp(z)), unit="cm per sqrt(degC d)",
                             detail=f"subsample <= 400 per row, seed {ZPOINT_SEED} (v2_data.py)"))
    d = L["df"].iloc[L["t_idx"]]
    for la, lo in d[["lat", "lon"]].values:
        rows.append(dict(panel="c", element="target_cell", region="Lena", lat=la, lon=lo, value=np.nan, unit="",
                         detail="region holdout target cell"))
    rows.append(dict(panel="c", element="buffer_distance", region="Lena", value=BUFFER_KM, unit="km",
                     detail="geodesic distance to nearest target cell, spherical R = 6371 km"))
    for b in L["A_blocks"]:
        rows.append(dict(panel="d", element="label_block", region="Lena", block=int(b), value=np.nan, unit="",
                         detail=f"split seed {DEMO['split']}"))
    for b in L["score_blocks"]:
        rows.append(dict(panel="d", element="scoring_block", region="Lena", block=int(b), value=np.nan, unit="",
                         detail=f"split seed {DEMO['split']}"))
    for la, lo, b in L["lab"][["lat", "lon", "block"]].values:
        rows.append(dict(panel="d", element="label_cell", region="Lena", block=int(b), lat=la, lon=lo, value=np.nan, unit="",
                         detail=f"n = {DEMO['n']}, draw index {DEMO['draw']}, seed {ctx['vals']['draw_seed']}"))
    xh = ctx["vals"]["xh"]
    lab = dict(XH_STAGES)
    for r in xh["rows"].itertuples():
        base = dict(panel="e", region=r.region, kind=f"{r.method}|{r.learner}|{lab[r.stage]}")
        rows.append(dict(base, element="nnd_median_km", value=float(r.nnd_median_km), unit="km", detail="x; median distance scoring cell to nearest training cell"))
        for nm, v in (("rmse_cell_weighted", r.rmse), ("rmse_cw_lo95", r.rmse_lo), ("rmse_cw_hi95", r.rmse_hi), ("rmse_block_equal", r.rmse_beq),
                      ("rmse_be_lo95", r.rmse_beq_lo), ("rmse_be_hi95", r.rmse_beq_hi)):
            rows.append(dict(base, element=nm, value=float(v), unit="cm",
                             detail="drawn" if nm.startswith("rmse_c") else "not drawn (block-equal; source data only)"))
    for r in xh["means"].itertuples():
        base = dict(panel="e", region="MEAN3[Alaska,Lena,Canada]", kind=f"{r.method}|{lab[r.stage]}")
        rows.append(dict(base, element="x_geometric_mean_km", value=float(r.x_geo), unit="km", detail="geometric mean of the three regions"))
        rows.append(dict(base, element="rmse_mean_equal_weight", value=float(r.rmse_mean), unit="cm", detail="equal-weight mean of cell-weighted RMSE; no CI in table"))
    for reg, g in xh["info"]["vg_geo"].items():
        for nm, v, u in (("vg_n_scoring_cells", g["n_score"], "cells"), ("vg_n_training_cells", g["n_train"], "cells"),
                         ("vg_n_buffer_excluded", g["n_buffer_excluded"], "cells"), ("vg_nnd_min_km", g["nnd_min_km"], "km")):
            rows.append(dict(panel="e", element=nm, region=reg, kind="V-G", value=float(v), unit=u,
                             detail="region holdout distance from coordinates only (h41 V-G definition, cv_schemes.nnd_km), no refit"))
    out = pd.DataFrame(rows, columns=["panel", "element", "region", "kind", "block", "lat", "lon", "value", "unit", "detail"])
    return out


def value_checks(ctx, aud, pa, fonts_txt, res, paths):
    D, vals = ctx["D"], ctx["vals"]
    t1 = D["t1"].set_index("target"); zs = D["zs"].set_index("row"); src = D["src"].set_index("target")
    B = D["blocks"]
    lines = []

    def chk(name, got, exp, ok=None):
        ok = (got == exp) if ok is None else ok
        lines.append(f"[{'OK' if ok else 'FAIL'}] {name}: 그림 {got} / 원천 {exp}")
        return ok

    lines.append("Fig 1 v3 값 점검(자동 생성, fig1.py). 그림 값은 Fig1_source_data.csv, 원천은 각 줄의 표.")
    lines.append(f"생성 시각 {time.strftime('%Y-%m-%d %H:%M:%S')}, 자원 확인 {res}")
    lines.append("")
    lines.append("## a 지도")
    v3 = B[B.kind == "v3"]
    # 블록 원의 값(fig1_blocks n_loc_1km)은 블록 안 1 km 위치 수이고 Table 1 의 loc_1km 는 지역 안 1 km 위치 수다. 한 1 km 셀의 행이
    # 0.5° 블록 경계 양쪽에 걸치면 블록 합이 지역 수보다 크다. 같은 기준 표(fidelity_base_v3, F4_direct)에서 걸친 셀 수를 세어 대조한다.
    base = ctx["L"]["df"]
    straddle = {}
    for reg in ["Lena", "Canada", "Russia_W", "Russia_E", "Russia_C", "Greenland", "Alaska"]:
        q = base[base.macro.values == reg]
        lk = loc1km_index(q.lat.values, q.lon.values)
        straddle[reg] = int((pd.Series(block05_index(q.lat.values, q.lon.values)).groupby(lk).nunique() > 1).sum())
        chk(f"{reg} 블록 원 값의 합(fig1_blocks v3 n_loc_1km) 대 table1 loc_1km + 블록 경계에 걸친 1 km 셀 {straddle[reg]}개",
            int(v3[v3.region == reg].n_loc_1km.sum()), int(t1.loc[reg, "loc_1km"]) + straddle[reg])
    lines.append(f"[정보] 블록 경계에 걸친 1 km 셀(블록 합 − 지역 수): {straddle}. 원 면적은 블록 안 위치 수를 따른다(설명문과 같음)")
    for reg in ["Lena", "Canada", "Russia_W", "Russia_E", "Alaska"]:
        chk(f"{reg} v3 블록 수(fig1_blocks) 대 table1 blocks", int((v3.region == reg).sum()), int(t1.loc[reg, "blocks"]))
    chk("새 지역 Russia_C_LGD 1 km 위치(LGD 51 + v3 6, 흰 원 51 + 검정 원 v3 Russia_C 블록) 대 table1 loc_1km 57",
        int(B[(B.kind == "lgd_added") & (B.region == "Russia_C")].n_loc_1km.sum()) + 6, int(t1.loc["Russia_C_LGD", "loc_1km"]))
    chk("Tibet_LGD 1 km 위치 합(흰 원) 대 table1 loc_1km", int(B[B.region == "Tibet_LGD"].n_loc_1km.sum()), int(t1.loc["Tibet_LGD", "loc_1km"]))
    chk("Tibet_LGD 블록 수 대 table1 blocks", int((B.region == "Tibet_LGD").sum()), int(t1.loc["Tibet_LGD", "blocks"]))
    chk("그리지 않은 North Atlantic 블록(natl_si) 1 km 위치 합 대 fig1_meta n_natl_si", int(B[B.kind == "natl_si"].n_loc_1km.sum()),
        int(D["meta"].n_natl_si))
    nd = D["nd"].set_index("region").n_cells_not_drawn
    lines.append(f"[정보] 약관 미확인으로 그리지 않은 셀(fig1_not_drawn): {', '.join(f'{k} {int(v)}' for k, v in nd.items())} (설명문과 같음)")
    lines.append(f"[정보] a 에 그린 블록 {vals['a_blocks_drawn_main']}개 + 삽도 {vals['tibet_blocks_drawn']}개 = "
                 f"{vals['a_blocks_drawn_main'] + vals['tibet_blocks_drawn']}, fig1_blocks 전체 {len(B)} − natl_si {vals['a_blocks_not_drawn_natl']}")
    chk("a·삽도 블록 수 합", vals["a_blocks_drawn_main"] + vals["tibet_blocks_drawn"], len(B) - vals["a_blocks_not_drawn_natl"])
    lines.append(f"[정보] 1 km 위치 수 최댓값 {int(B.n_loc_1km.max())}, 중앙값 {B.n_loc_1km.median():.0f}, 99 백분위 {B.n_loc_1km.quantile(0.99):.1f}"
                 f" → 크기 열쇠 1, 10, 50(명세 값 유지). 원 면적 = {SIZE_K} pt² × 위치 수 [판단]")
    lines.append(f"[정보] 지도 축척: 지름 {A_MAP[2]} mm 가 위도 {LAT_MIN}° N 원, 1 mm = {vals['a_km_per_mm']} km(70° N 기준)")
    lines.append(f"[정보] 투영: 북극 평사, 진척 위도 70° N, 중심 경도 {vals['a_central_lon']}° (레나델타 중앙 경도) [판단]. 명세 3.3 의 −45°(EPSG:3413 형식)"
                 "에서는 레나델타 사각형이 지도 위끝에 놓여 c 로 가는 연결선(지침 2.11)이 극·캐나다·그린란드를 가로질렀다. 이 선택으로 사각형이"
                 " 지도 아래끝에 오고 연결선이 c 의 위 모서리로 곧게 내려간다. Fig 6a·b(−45°)와 방향이 다르다(사용자 확인 필요)")
    sa = vals["a_scale"]
    chk("a 축척 막대 1000 km 의 측지 길이(구면 근사, 70° N)", round(sa["geodesic_km"]), 1000, ok=abs(sa["geodesic_km"] - 1000) < 15)
    ts = vals["tibet_scale"]
    chk("삽도 축척 막대 500 km 의 측지 길이", round(ts["geodesic_km"]), 500, ok=abs(ts["geodesic_km"] - 500) < 10)
    chk("삽도 축척 막대·길이 글자와 겹치는 블록 원 수", vals["tibet_scale_overlap_blocks"], 0)
    chk("삽도 틀 밖으로 잘린 티베트 블록 원 수", vals["tibet_blocks_below_frame"], 0)
    chk("삽도 틀(이름 줄 포함)이 가리는 본 지도 블록 원 수", vals.get("tibet_inset_hides_main_blocks", -1), 0)
    lines.append(f"[정보] 삽도 축척 1 mm = {vals['tibet_km_per_mm']} km(Lambert 방위 등적, 중심 부근); 본 지도 1 mm = {vals['a_km_per_mm']} km")
    lines.append(f"[정보] 영구동토 바탕: {vals['pfr_title']}; 내장 래스터 a {vals['a_pfr_raster']}, 삽도 {vals['tibet_pfr_raster']}")
    lines.append(f"[정보] 열쇠 3줄(지도 아래 오른쪽, 줄 2 = 영구동토 머리): 줄 1 끝 {vals['key_a']['row1_end_mm']} mm, 줄 3 끝 {vals['key_a']['row2_end_mm']} mm, 칸 폭 {vals['key_a']['width_mm']} mm")
    lines.append("")
    lines.append("## b Stefan 계수 분포")
    for r in vals["b_rows"]:
        key = r["row"]
        chk(f"{key} 셀 수(fig1_z_summary n_cells) 대 table1 label_rows", r["n_cells"], int(t1.loc[key, "label_rows"]))
        e0_t1 = float(t1.loc[key, "E0_x"])
        chk(f"{key} 원천 계수 E0 (fig1_z_summary) 대 table1 E0_x", round(r["E0"], 4), round(e0_t1, 4))
        if key in src.index:
            chk(f"{key} 원천 계수 E0 대 fig1_source E0", round(r["E0"], 4), round(float(src.loc[key, "E0"]), 4))
        lines.append(f"    {key}: 표시 점 {r['n_points']}, 중앙값 {r['E_q50']:.2f}, 사분위 {r['E_q25']:.2f}–{r['E_q75']:.2f}, "
                     f"표시 점 범위 {r['E_min_shown']:.3f}–{r['E_max_shown']:.2f}")
    chk("b 축 범위 밖 점 수(삼각 표지 대상)", vals["b_points_outside_axis"], 0)
    lines.append(f"[정보] b 행: Table 1 의 7행(주 4지역, Alaska, 새 지역 2)만 그린다. v3 의 Russia_C(7셀)와 Greenland(3셀)는 "
                 "Table 1 처럼 Supplementary Table S1 로 옮긴다 [판단, 명세 10.3]")
    lines.append(f"[정보] b 눈금 {E_TICKS}, 범위 {E_LIM}: 명세의 1, 2, 5, 10, 20 은 표시 점 최솟값(캐나다 0.022)을 담지 못해 바꿨다 [판단]")
    lines.append("")
    lines.append("## c·d 레나델타 설계 확대도")
    lt = pd.read_csv(ROOT / "results/rescale_lg/data/processed/lg/lg_targets.csv")
    q = lt[(lt.target == "Lena") & (lt["mode"] == "x") & (lt.part == "cpu") & (lt.split == DEMO["split"])].iloc[0]
    chk("대상 셀 수(load_base macro == Lena) 대 table1 label_rows", vals["lena_n_cells"], int(t1.loc["Lena", "label_rows"]))
    chk("대상 0.5° 블록 수 대 table1 blocks", vals["lena_n_blocks"], int(t1.loc["Lena", "blocks"]))
    chk("분할 1 A 셀 수 대 lg_targets n_A", vals["split1_nA"], int(q.n_A))
    chk("분할 1 A 블록 수 대 lg_targets nb_A", vals["split1_nbA"], int(q.nb_A))
    chk("분할 1 채점 셀 수 대 lg_targets n_eval", vals["split1_n_eval"], int(q.n_eval))
    chk("분할 1 채점 블록 수 대 lg_targets nb_eval", vals["split1_nb_eval"], int(q.nb_eval))
    lines.append(f"[정보] 분할 1 B 블록 {vals['split1_nbB']}개 가운데 채점 셀이 없는 블록 {vals['split1_B_blocks_without_eval']}개")
    chk("라벨 셀 수", vals["n_label_cells"], DEMO["n"])
    chk("라벨 셀이 모두 라벨 블록 안", vals["label_cells_in_A"], True)
    lines.append(f"[정보] 라벨 추출 seed = seed_of('Lena', 'x', 1, 40, {DEMO['draw']}) = {vals['draw_seed']}; 라벨 셀이 든 블록 {vals['label_blocks_hit']}개")
    chk("원천 셀 가운데 대상 100 km 안(버퍼 제외) 수 대 fig1_source n_buffer_excluded", vals["src_within_100km"],
        int(src.loc["Lena", "n_buffer_excluded"]))
    lines.append(f"[정보] 그 셀: {vals['src_within_100km_regions']}. c·d 범위 안 원천 셀 {vals['c_source_cells_in_extent']}개 "
                 f"({vals['c_source_cells_in_extent_detail']}). 버퍼 안이라 원천에서 빠졌고 c 에는 그리지 않았다 [판단]")
    lines.append(f"[정보] 확대 투영: 북극 평사(중심 경도 {vals['zoom_center_lon']}°, 진척 위도 70° N), 범위 {vals['d_extent_km']} km 정사각형")
    chk("d 블록 경도 폭(mm, 지침 2.11 의 1.0 mm 이상)", vals["d_block_lon_width_mm"], ">= 1.0", ok=vals["d_block_lon_width_mm"] >= 1.0)
    for kd, nm in (("A", "Label blocks"), ("B", "Scoring blocks")):
        lv = vals[f"d_leader_{kd}"]
        chk(f"d 지시선 '{nm}' 위반(다른 종류 블록 가로지름·라벨 셀 점 근접) 0", lv["violation"], 0)
        lines.append(f"[정보] d 지시선 '{nm}': 길이 {lv['length_mm']} mm, 글자 자리 시도 {lv['text_pos_tried']}회")
    sc = vals["c_scale"]
    chk("c 축척 막대 100 km 의 측지 길이", round(sc["geodesic_km"], 1), 100, ok=abs(sc["geodesic_km"] - 100) < 2)
    lines.append("")
    lines.append("## e 검증 사다리(XH, 계획 8.3)")
    xh = vals["xh"]
    X = xh["rows"].set_index(["region", "method", "stage"])
    lines.append(f"[정보] 원천 {xh['info']['file']} (sha256 앞 16자 dedef96fd096d2e4 는 계획 8.3 기록), 그린 행 30(XH 24 = 두 방법 × 네 단 × 세 지역, V-G 6), "
                 f"variant 'pm2' 행 {xh['info']['n_pm2_excluded']}개 제외")
    plan = {("Alaska", "D0w", "W1R"): 11.54, ("Lena", "D0w", "W1R"): 13.94, ("Canada", "D0w", "W1R"): 18.96,
            ("Alaska", "PSw", "W1R"): 14.24, ("Lena", "PSw", "W1R"): 20.89, ("Canada", "PSw", "W1R"): 26.50,
            ("Alaska", "D0w", "W1K"): 17.30, ("Lena", "D0w", "W1K"): 22.10, ("Canada", "D0w", "W1K"): 33.16,
            ("Alaska", "PSw", "W1K"): 14.53, ("Lena", "PSw", "W1K"): 21.40, ("Canada", "PSw", "W1K"): 30.78}
    for k, v in plan.items():
        chk(f"{k[0]} {k[1]} {k[2]} 셀 가중 RMSE 대 계획 8.3 표", round(float(X.loc[k, "rmse"]), 2), v)
    for reg, dd, dp in (("Alaska", 5.76, 0.29), ("Lena", 8.17, 0.51), ("Canada", 14.20, 4.28)):
        g_d = float(X.loc[(reg, "D0w", "W1K"), "rmse"] - X.loc[(reg, "D0w", "W1R"), "rmse"])
        g_p = float(X.loc[(reg, "PSw", "W1K"), "rmse"] - X.loc[(reg, "PSw", "W1R"), "rmse"])
        chk(f"{reg} 무작위 → kNNDM 증가(직접 ML, 재보정 Stefan) 대 계획 8.3 문장", (round(g_d, 2), round(g_p, 2)), (dd, dp),
            ok=abs(g_d - dd) < 0.006 and abs(g_p - dp) < 0.006)
    for reg, v in (("Alaska", (0.00, 0.88, 26.06, 58.73)), ("Lena", (0.01, 1.02, 9.55, 26.42)), ("Canada", (0.01, 3.86, 25.33, 236.53))):
        got = tuple(round(float(X.loc[(reg, "D0w", s_), "nnd_median_km"]), 2) for s_ in ("W1R", "W1S", "W1B", "W1K"))
        chk(f"{reg} 거리 중앙값(km) 대 계획 8.3 표", got, v)
    Mn = xh["means"]
    lines.append("[정보] 세 지역 등가중 평균(선, 셀 가중 RMSE, cm): " + "; ".join(
        f"{m_} {dict(XH_STAGES)[s_]} {v:.2f} at {xg:.3g} km" for m_, s_, v, xg in Mn[["method", "stage", "rmse_mean", "x_geo"]].values))
    lo = float(xh["rows"][["rmse_lo"]].min().iloc[0]); hi = float(xh["rows"][["rmse_hi"]].max().iloc[0])
    chk("y 축 범위가 지역 CI 를 모두 담음", (XH_YLIM[0] <= lo, hi <= XH_YLIM[1]), (True, True))
    xmin = float(xh["rows"].nnd_median_km.min()) * min(XH_XOFF.values()); xmax = float(xh["rows"].nnd_median_km.max()) * max(XH_XOFF.values())
    chk("x 축 범위가 지역 점을 모두 담음", (XH_XLIM[0] <= xmin, xmax <= XH_XLIM[1]), (True, True))
    lines.append("[정보] 그리지 않음: 블록 등가중 값과 CI(원천 자료에 있음, 셀 가중 CI 와 겹쳐 지역 점에도 그리지 않았다), 평균의 CI(표에 없음)")
    lines.append(f"[정보] 지역 홀드아웃(V-G): RMSE 는 {xh['info']['vg_file']} 의 지역 행(D0 catboost_lo λ 1.0 = 직접 ML, PS = 학습 셀(원천) 최소제곱 "
                 "계수 Stefan, 곧 원천 계수. 빈 기호와 점선으로 구분). 거리는 좌표만으로 h41 정의(채점 = 대상 ∩ eval_mask, 학습 = source_idx(지역, x) "
                 "∩ y·s 유한, 100 km 버퍼)와 XH 의 cv_schemes.nnd_km 으로 계산했다(새 적합 없음)")
    lgv_n = {"Alaska": 13606, "Lena": 2958, "Canada": 747}
    for reg, g in xh["info"]["vg_geo"].items():
        chk(f"{reg} V-G 채점 셀 수(좌표 재현) 대 lgv_metrics n_cells", g["n_score"], int(X.loc[(reg, "D0w", "V-G"), "n_cells_lgv"]))
        chk(f"{reg} V-G 채점 셀 수 대 계획 기록(h41 채점 집합)", g["n_score"], lgv_n[reg])
        chk(f"{reg} V-G 최근접 학습 셀 최소 거리 > 버퍼 100 km", round(g["nnd_min_km"], 1), "> 100", ok=g["nnd_min_km"] > g["buffer_km"])
        lines.append(f"[정보] {reg} V-G: 학습 셀 {g['n_train']}, 버퍼 제외 {g['n_buffer_excluded']}, 거리 중앙값 {g['nnd_median_km']:.2f} km")
    for (reg, mth), v in {("Alaska", "D0w"): 35.56, ("Lena", "D0w"): 25.31, ("Canada", "D0w"): 25.24,
                          ("Alaska", "PSw"): 14.68, ("Lena", "PSw"): 21.64, ("Canada", "PSw"): 26.48}.items():
        chk(f"{reg} {mth} V-G 셀 가중 RMSE 대 계획 8.3 표(V-G 열)", round(float(X.loc[(reg, mth, "V-G"), "rmse"]), 2), v)
    lines.append("")
    lines.append("## 설명문(Fig1_legend.md, 손으로 작성)")
    leg = S.OUT / f"{STEM}_legend.md"
    if leg.exists():
        import re as _re
        txt = leg.read_text(encoding="utf-8")
        body = " ".join(ln for ln in txt.splitlines() if ln.strip() and not ln.startswith("#")).replace("**", "")
        ph = _re.findall(r"\[X[A-J]:[^\]]*\]", body)
        body_noph = _re.sub(r"\[X[A-J]:[^\]]*\]", "", body)
        nw, nw_noph = len(body.split()), len(body_noph.split())
        chk("설명문 350단어 이하(자리표시 포함)", nw, "<= 350", ok=nw <= 350)
        lines.append(f"[정보] 설명문 단어 수: 자리표시 포함 {nw}, 제외 {nw_noph}; 자리표시 {len(ph)}개(등록 문서 형식, 원고 조립 때 XH 수치 문장으로 치환)")
        ta = S.text_audit(body_noph)
        bad = {k: v for k, v in ta.items() if v}
        chk("설명문 문장 점검(연결어 대시·금지어·가운뎃점 숫자·네 자리 쉼표·내부 약호, 자리표시 제외)", bad or "0건", "0건", ok=not bad)
        chk("설명문 첫 문장이 'This figure' 로 시작하지 않음(명사구 제목)", body.split(".")[0][:60], "명사구", ok=not body.lower().startswith("this figure"))
        for s_ in ("central meridian", "true at 70", "Natural Earth", "Cartopy", "ESA CCI"):
            chk(f"설명문 지도 기재 사항 '{s_}'", s_ in body, True)
    else:
        chk("설명문 파일", str(leg), "있음", ok=False)
    lines.append("")
    lines.append("## 점검 함수(지침 부록 A)")
    lines.append(f"audit_v3: fails = {aud['fails']}, 글자 크기 {sorted(aud['sizes'])}, 문자 수 {aud['chars']}, 얇은 선 {aud['thin_lines']}, "
                 f"제목 {aud['titles']}, 글씨 상자 {aud['boxed_text']}, 내부 코드 {aud['codes']}, 네 자리 쉼표 {aud['comma4']}, "
                 f"긴 라벨 {aud['long_labels']}, 그림 안 수치 {aud['loose_numbers']}")
    lines.append("  (위도 라벨 3개는 gid 'graticule' 로 그림 안 수치 검사에서 뺐다. 명세 3.3 a 가 위도 라벨 3개를 허용한다)")
    if aud.get("audit_fig") is not None:
        af = aud["audit_fig"]
        chk("digest 5.3 audit_fig: 글씨 상자·곡선 화살표·사선 화살표 0(지시선은 Line2D)", (af["text_boxes"], af["curved"], af["diagonal"]), (0, 0, 0))
        lines.append(f"  audit_fig 긴 라벨(3단어 초과, 사람 판정): {af['long_labels'] or '0'}; 제목 {af['titles'] or '0'}")
    if aud.get("text_overlaps") is not None:
        to = aud["text_overlaps"]
        chk("글자끼리 겹침 쌍(style.text_overlaps)", to["overlap_pairs"] or "0", "0", ok=not to["overlap_pairs"])
        chk("캔버스 밖 글자", to["outside"] or "0", "0", ok=not to["outside"])
    lines.append(f"pdf_audit: {pa}")
    lines.append("pdffonts:")
    lines += ["  " + ln for ln in fonts_txt.strip().splitlines()]
    lines.append("산출: " + ", ".join(str(p.relative_to(ROOT)) for p in paths))
    nfail = sum(ln.startswith("[FAIL]") for ln in lines)
    lines.insert(2, f"값 점검 실패 {nfail}건")
    return "\n".join(lines) + "\n", nfail


def text_overlaps_drawn(fig) -> dict:
    """그려지는 글자끼리의 경계 상자 겹침 쌍과 캔버스 밖 글자. style.text_overlaps 와 같되, 지도 축(GeoAxes, 축 눈금 숨김)과
    축을 끈 열쇠 축의 눈금 라벨(글자는 있으나 그려지지 않음)은 뺀다."""
    import itertools
    from matplotlib.text import Text
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    hidden = set()
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            if not a.axison or not axis.get_visible():
                hidden.update(axis.get_ticklabels(which="both"))
                hidden.add(axis.label)
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t not in hidden]
    bbs = [(t.get_text().strip(), t.get_window_extent(rend)) for t in texts]
    pairs = [(a[:25], b[:25]) for (a, ba), (b, bb) in itertools.combinations(bbs, 2) if ba.overlaps(bb)]
    W, H = fig.get_size_inches() * fig.dpi
    m = 0.3 / 25.4 * fig.dpi
    outside = [s_[:25] for s_, b in bbs if b.x0 < m or b.y0 < m or b.x1 > W - m or b.y1 > H - m]
    return dict(n_texts=len(bbs), overlap_pairs=pairs, outside=outside)


def main():
    import subprocess
    res = check_resources()
    print("resources", res, flush=True)
    fig, ctx = build("paper")
    aud = S.audit_v3(fig, allowed_num_gids=("scale", "sizekey", "graticule"))
    aud["audit_fig"] = S.audit_fig(fig) if hasattr(S, "audit_fig") else None          # digest 5.3(상자·곡선·사선 화살표)
    aud["text_overlaps"] = text_overlaps_drawn(fig)                                     # 지침 2.2 글자 겹침 0곳
    print("audit_v3 fails:", aud["fails"], "chars", aud["chars"], "sizes", aud["sizes"], flush=True)
    for k in ("thin_lines", "titles", "boxed_text", "codes", "comma4", "long_labels", "loose_numbers"):
        if aud[k]:
            print("  ", k, aud[k])
    paths = S.save_fig(fig, STEM, formats=("pdf", "png"))
    pa = S.pdf_audit(S.OUT / f"{STEM}.pdf")
    fonts_txt = subprocess.run(["pdffonts", str(S.OUT / f"{STEM}.pdf")], capture_output=True, text=True).stdout
    print("pdf_audit", pa, flush=True)
    sd = source_data(ctx)
    sd.to_csv(S.OUT / f"{STEM}_source_data.csv", index=False)
    txt, nfail = value_checks(ctx, aud, pa, fonts_txt, res, paths + [S.OUT / f"{STEM}_source_data.csv"])
    (S.OUT / f"{STEM}_values.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if (not aud["fails"] and nfail == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
