# 실험·데이터·산출물 마스터 인덱스

> 이 문서 하나로 프로젝트 전체를 이해한다. 스크립트는 실행 단계별 폴더(`scripts/0_download` → `4_visualization`).
> 모든 경로는 프로젝트 루트에서 `python3 scripts/<폴더>/<파일>.py` 로 실행.

## 0. 목표 (한 줄)
전 지구 borehole 지중온도 + 활성층(ALT) 관측 + 공변량 → 딥러닝으로 **ALT 지도 + 얕은 3D 지중열구조 + 불확실성**,
**알래스카 학습 → 타 영구동토 지대 전이(transfer)** 검증.

## 1. 데이터 (data/)
### 라벨(정답) — 우리가 맞히려는 것
| 파일 | 내용 | 단위/규모 |
|---|---|---|
| `processed/alt_global.csv` | CALM 전지구 활성층 두께 + WorldClim 공변량 | site-year, 3,604행 |
| `processed/alt_above_pointlevel.csv` | ABoVE ALT (알래스카/NW캐나다) | 점단위, 22.4만 |
| `processed/ground_temp.csv` | 신규 지중온도 프로파일(서시베리아 GGD200·스위스 PERMOS) | 깊이-온도 |
| `processed/borehole_profiles.csv` | GTN-P 알래스카 시추공 깊이별 연평균지중온도(MAGT) | 35공, 1,574점 |
### 공변량(입력 단서)
| 원천 | 파일/위치 | 해상도 |
|---|---|---|
| WorldClim 2.1 기후 | `alt_global.csv`의 wc_* | 18km 평년값 |
| ERA5-Land 기후(도일/적설/토양온도) | `raw/era5land/`, `alt_era5_covariates.csv` | 9km 월별 |
| Copernicus DEM 지형 | `raw/dem/` (80타일), `dl_locations.csv`의 dem_* | 30m |
| InSAR ReSALT(ALT+침하) | `raw/resalt/` (52 granule) | 30m *(활용 예정)* |
### 학습셋(딥러닝용, 조립본)
| 파일 | 내용 |
|---|---|
| `dl_dataset.csv` | 북미 ALT 225,421점 = ALT + ERA5 8종 + 지형 6종 + loc_id |
| `dl_locations.csv` / `dl_patches.npy` | 고유위치 14,354개 지형특징 / 33×33 DEM 패치 |
| `resalt_weaklabels.parquet` | InSAR ReSALT weak label 408만 점(51사이트, 30m) |
| `pretrain_weaklabels.parquet` | weak label 403.6만 점 + 공변량 14종(사전학습셋) |
| `ground_temp_all.csv` | 전지구 지중온도 3D 라벨 10,747점(260 사이트: GTNP+GGD200+PERMOS) |
| `raw/cci_alt/` | ESA CCI ALT 연도별 격자 25개(1997-2021, 1km, 단위 **m**) |

## 2. 파이프라인 (scripts/)
| 폴더 | 스크립트 | 하는 일 → 산출 |
|---|---|---|
| **0_download** | gtnp_boreholes, gridded_covariates, era5land_monthly, copernicus_dem, resalt_insar | 원격 데이터 다운로드 |
| **1_data_prep** | parse_ground_temp, parse_above_alt, parse_gtnp_global, extract_resalt_weaklabels, assemble_pretrain_dataset, era5land_covariates, terrain_features_dem, assemble_dl_dataset | 파싱·공변량 산출·학습셋 조립 |
| **2_evaluation** | rescore_worldclim_vs_era5 | 공변량 교체 통제실험(공간블록/LORO) |
| **3_deep_learning** | train_patch_cnn, train_pretrain_finetune(B0), train_b0b_cci_feature(B0b), train_neural_field_3d(B1), model_tournament | DL 학습(GPU) + GBM/CCI 비교. tournament=GBM/MLP/FT-Transformer/TabM/Flow/Diffusion 공정비교 |
| **4_visualization** | map_alt_alaska, map_alt_surface, map_alt_surface_b0, map_ground_temp_global, map_weaklabels, thermal_3d_gif, maps_spatial_eval, dl_charts_cnn_vs_gbm | 지도·3D·GIF |
| legacy_alaska_pilot | interpolate_3d_kriging, visualize_pilot | 초기 알래스카 3D 파일럿(참고) |

## 3. 지금까지의 핵심 결과
| 실험 | 결과 | 근거 |
|---|---|---|
| 누설 진단 | 무작위 CV가 전이를 4배 과대평가 | `cv_leakage_table.csv` |
| **공변량 업그레이드** | ERA5-Land 실측이 **전이(LORO) 108.5→87.3cm (−20%)** | `stage2_era5_rescore.csv`, `figures/03_covariate_upgrade/` |
| **공간 DL** | 패치 CNN 17.2 ≈ GBM 17.7cm (**공간맥락 이득 미미**) — 병목은 모델 아닌 공변량 정보량 | `dl_cnn_results.csv`, `figures/06_deep_learning/` |
| **B0 사전학습+미세조정** | weak 403.6만 사전학습 DL 18.1 ≈ GBM 17.7 — 둘 다 **ESA CCI(20.8) 격파**. 공간블록 CV | `b0_pretrain_results.csv`, `maps/alt_surface_b0_comparison` |
| **B0b CCI피처+앙상블** | CCI 피처는 무익(중복정보). **앙상블 (DL+GBM)/2 = 17.5cm 최고** | `b0b_results.csv` |
| **B1/B1b 3D 신경장 게이트** | NF 2.17→2.36°C vs **GBM 1.31/IDW 1.30** — 탈락, **3D 엔진=GBM 조건장**(킬스위치). 단 지역전이 GBM 1.40 < IDW 1.69(조건장이 전이 우세) | `b1_neural_field_results.csv`, `b1b_results.csv` |
| **모델 토너먼트(6종, 6fold+2seed)** | 앙상블 16.95 ≈ **Diffusion 17.09** ≈ GBM 17.24 > FT-T 17.95 > MLP 18.18 > Flow 18.31 > TabM 20.36. **부트스트랩: 전 모델 GBM과 동률**(유의차 없음). 1:1 진단서 전 모델 평균회귀 → **정보병목 시각 확증**. Diffusion MAE 최저+네이티브 UQ로 채택 | `model_tournament_{results,perfold,significance}.csv`, `figures/06_deep_learning/model_tournament*.png`, `maps/deploy_*` |
| **point-scale floor 4중 확증** | InSAR(ReSALT r=0.23~0.35)·PolSAR(우리점 r=0.31/38cm)·격자지지·areal 전부 17cm 못뚫음. 셀내 SD 11cm=대표성 하한 | `insar_ablation`·`polsar_residual`·`grid_support`·`areal_eval` |
| **정확도-범위 트레이드오프(첫 돌파)** | 평탄툰드라(PolSAR+InSAR 앙상블) **12.97cm=SOTA급** → 완만 16.6 → 전역 17.3. 큐레이션+물리관측이 floor 돌파 | `curated_scope_results.csv`, `figures/06_deep_learning/accuracy_vs_scope.png` |
| **고정밀 국소 데모** | 북사면 평탄툰드라 250m ALT 필드(PolSAR/모델/UQ 3패널) + Area-of-Applicability 마스크 | `maps/local_demo_alt_field.png` |

## 4. 산출물 (outputs/) — 규칙은 docs/VISUALIZATION.md
- `figures/00_concept … 06_deep_learning/` : 분석 그래프
- `maps/` : 2D 공간 레이어 — `alt_alaska_pred`, `alt_surface_northslope`, `alt_surface_b0_comparison`(DL/GBM/CCI 3-패널),
  `b0_fold_error_map`, `magt_alaska_2m_20m`(MAGT+영구동토경계), `ground_temp_global`, `weaklabels_overview`, `spatial_cv_blocks`
- `volumes_3d/` : `thermal_cube_alaska`(3D 층층 슬라이스) / `animations/` : `thermal_depth_slices.gif`
- `models/` : `b0_mlp_pretrained/finetuned_full.pt`, `b1b_neural_field.pt`, `patchcnn_fold*.pt`

## 5. 다음
- **모델 토너먼트(준비 완료, GPU 대기중)**: `scripts/3_deep_learning/model_tournament.py` — GBM·MLP·FT-Transformer·TabM·Flow matching·Diffusion을 동일 공간블록 CV로 비교(+생성모델 UQ 커버리지). torch 직접구현(외부 의존성 0). 실행: `CUDA_VISIBLE_DEVICES=<유휴> python3 scripts/3_deep_learning/model_tournament.py` → `charts_model_tournament.py`로 시각화. 스모크(SMOKE=1) 검증 완료: 6모델 정상, 소표본 미리보기서 FT-T/Flow가 GBM 상회·Flow/Diffusion UQ 91~93% 보정.
- 그다음: 정확도 실질 지렛대 = 새 공변량 모달리티(Sentinel-1 InSAR 시계열·Sentinel-2 영상·SoilGrids)를 먹는 영상 CNN/ViT.
- 3D(PyVista/VTK): 최고 모델 확정 후 — 복셀 볼륨 + marching-cubes 삼각메쉬 등온면 + DEM 드레이프 + 인터랙티브 HTML/고해상 렌더.
