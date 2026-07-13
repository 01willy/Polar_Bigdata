# Polar_Bigdata

영구동토 활성층 두께(ALT) 2D 매핑과 얕은 3D 지중 열구조를 관측기반 GeoAI로 예측하는 프로젝트.
알래스카에서 학습해 타 영구동토 지대로 전이하며, 셀별 보정 불확실성(UQ)과 적용범위(AOA)를 함께 제시한다.

> **현행 상태는 이 README가 아니라 `SESSION_HANDOFF.md`가 기준이다(README는 초기 계획 서술로 일부 stale).**
> 3D 엔진은 크리깅·신경장이 아니라 **GBM 조건장**으로 확정되었고, 주 출력은 **셀 단위 ALT 매핑**이다.
> 상세 진행: `docs/EXPERIMENT_LOG.md`, 방향: `docs/PLAN_FORWARD.md`.

연구 계획 전문: [docs/PLAN.md](docs/PLAN.md)

## 폴더 구조

```
Polar_Bigdata/
├── README.md / requirements.txt
├── docs/PLAN.md              # 연구 계획(데이터·방법론·로드맵)
├── src/polar/               # 핵심 로직 패키지
│   ├── config.py            # 모든 경로·상수(Alaska bbox, epoch) — 여기만 고치면 됨
│   ├── geo.py               # 좌표 투영(EPSG:4326↔3413)
│   ├── acquire.py           # ① GTN-P 확보(list+bulk, 적응형 배치, combined 폴백)
│   ├── preprocess.py        # ② QC·MAGT·DZAA·지온경사
│   ├── covariates.py        # ③ 지점(고도·토양) / ④ 격자(WorldClim 무계정, ERA5 계정)
│   ├── interpolate.py       # ⑤ baseline 3D 크리깅(층별 OK → 체적·등온면·불확실성)
│   ├── model.py             # ⑤ 공변량 RandomForest + leave-one-borehole-out CV
│   ├── visualize.py         # ⑥ 개요 패널
│   ├── viz_suite.py         # ⑥ 깊이슬라이스·ALT·프로파일·불확실성·공변량·비교·개념도
│   └── viz3d.py             # ⑥ 위도-깊이 단면 + 3D hero
├── scripts/                 # 실행 진입점 (번호 = 실행 순서)
│   ├── 01_download_gtnp.py
│   ├── 02_preprocess.py
│   ├── 03_enrich_covariates.py   # 무인증(USGS 고도·SoilGrids)
│   ├── 04_download_gridded.py    # CCI/ERA5/MODIS (계정 필요)
│   ├── 05_interpolate_3d.py
│   └── 06_visualize.py
├── data/                    # (git 미추적 권장)
│   ├── raw/gtnp/            # 원본 다운로드 — 불변
│   │   ├── pt_csv/  alt_csv/
│   │   ├── pt_manifest.json  sites.json  boreholes.csv
│   ├── interim/             # 중간 산출물 (통합 long table)
│   └── processed/           # 분석용 테이블 (profiles, summary)
└── outputs/
    ├── figures/             # 시각화 PNG
    ├── meshes/              # tri-mesh (.vtp/.stl)
    └── models/              # 학습된 모델
```

**데이터 흐름**: `raw`(원본·불변) → `interim`(중간) → `processed`(분석용) → `outputs`(그림·메시·모델).

## 실행

```bash
python scripts/01_download_gtnp.py       # ① Alaska 지중온도 (--manifest-only / --region all)
python scripts/02_preprocess.py          # ② long table + MAGT + DZAA + 지온경사
python scripts/03_enrich_covariates.py   # ③ 지점 공변량(고도·토양) — 무인증
python scripts/04_download_gridded.py    # ④ 격자 공변량(ERA5/CCI/MODIS) — 계정 필요
python scripts/05_interpolate_3d.py      # ⑤ baseline 3D 체적 + 0°C 등온면 mesh
python scripts/06_visualize.py           # ⑥ 개요 + 위도-깊이 단면 + 3D hero
```

설정(영역/기간 등)은 [src/polar/config.py](src/polar/config.py) 한 곳에서 변경.

## 주요 산출물 (outputs/figures) — 10종
- `hero_3d_permafrost.png` — 3D 온도장(크리깅 보간)
- `cross_section_lat_depth.png` — 위도–깊이 온도 단면(영구동토 경계)
- `02_depth_slices.png` — 깊이별 크리깅 온도 지도 / `05_kriging_uncertainty.png` — 불확실성
- `07_covariates.png` — WorldClim 격자 공변량 / `08_kriging_vs_rf.png` — 거리 vs 공변량(CV)
- `03_active_layer.png` — 활성층(ALT) / `04_profiles_gallery.png` — borehole 프로파일
- `06_concept_alt_pt.png` — ALT↔PT 연결 개념도 / `01_alaska_overview.png` — 개요
- `outputs/meshes/*.vtp` — 0°C 등온면·frozen body tri-mesh (ParaView/Blender)

검증 결과(`data/processed/cv_results.csv`): leave-one-borehole-out RMSE — IDW 2.17 → **RF(공변량) 1.53 °C**.
