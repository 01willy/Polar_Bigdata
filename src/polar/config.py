"""
프로젝트 전역 설정 — 경로, GTN-P API, Alaska 영역, 전처리 상수.

모든 스크립트/모듈은 여기의 경로 상수를 import 해서 사용한다(경로 하드코딩 금지).
data/raw = 원본(불변), data/interim = 중간, data/processed = 분석용, outputs = 산출물.
"""
from pathlib import Path

# --- 프로젝트 루트 (src/polar/config.py 기준 2단계 위) ---
ROOT = Path(__file__).resolve().parents[2]

# --- 데이터 디렉터리 ---
DATA = ROOT / "data"
RAW_GTNP = DATA / "raw" / "gtnp"          # GTN-P 원본 다운로드
INTERIM = DATA / "interim"                 # 중간 산출물(예: 통합 long table)
PROCESSED = DATA / "processed"             # 최종 분석용 테이블

# --- 산출물 디렉터리 ---
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"              # 시각화 PNG
MESHES = OUTPUTS / "meshes"                # tri-mesh (.vtp/.stl)
MODELS = OUTPUTS / "models"               # 학습된 모델

# --- 원본 하위 경로 ---
PT_CSV_DIR = RAW_GTNP / "pt_csv"           # borehole 온도 CSV
ALT_CSV_DIR = RAW_GTNP / "alt_csv"         # active layer CSV
PT_MANIFEST = RAW_GTNP / "pt_manifest.json"
ALT_MANIFEST = RAW_GTNP / "alt_manifest.json"
SITES_JSON = RAW_GTNP / "sites.json"
BOREHOLES_CSV = RAW_GTNP / "boreholes.csv"  # borehole_id,lat,lon,elev,country,site

# --- 전처리 산출물 ---
PROFILES_CSV = PROCESSED / "borehole_profiles.csv"   # (borehole,depth)별 MAGT/연진폭
SUMMARY_CSV = PROCESSED / "borehole_summary.csv"     # borehole별 요약(DZAA, 지온경사, base)
LONGTABLE_PARQUET = INTERIM / "alaska_long_table.parquet"  # 통합 QC long table

# --- GTN-P API ---
GTNP_BASE = "https://data.gtn-p.org/api"
# /api/data 배치 크기: 100개 이상 ID는 HTTP 500, 20개(~18MB/~76s)가 안전
BATCH_SIZE = 20
HTTP_TIMEOUT = 180
HTTP_RETRIES = 3

# --- Alaska 지역 필터 ---
ALASKA_BBOX = dict(lat_min=51.0, lat_max=72.0, lon_min=-170.0, lon_max=-129.0)
ALASKA_COUNTRY = "US"

# --- MAGT(평형) 산정 상수 ---
MIN_MONTHS_PER_YEAR = 10        # 비편향 연평균에 필요한 최소 월 수
DZAA_AMP_THRESHOLD = 0.1        # °C, 연진폭 < 이 값이면 DZAA(연진폭 0 깊이)로 간주
MIN_DEPTHS_FOR_GRADIENT = 3     # 지온경사 선형적합 최소 깊이 점 수


def ensure_dirs():
    """필요한 디렉터리를 모두 생성(존재하면 무시)."""
    for d in (RAW_GTNP, PT_CSV_DIR, ALT_CSV_DIR, INTERIM, PROCESSED,
              FIGURES, MESHES, MODELS):
        d.mkdir(parents=True, exist_ok=True)
