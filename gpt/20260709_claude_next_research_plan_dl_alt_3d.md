# Polar_Bigdata — Claude Code용 다음 연구·실험 계획

작성일: 2026-07-09  
목적: 현재까지의 ALT/지중온도 예측 결과와 최근 permafrost/GeoAI 연구 흐름을 반영하여, Claude Code가 다음 실험을 일관되게 수행하도록 연구 목표, 딥러닝 고도화 방향, 실행 우선순위, 중단 조건을 명확히 지정한다.

---

## 0. 한 줄 결론

**이번 논문의 주축은 full 4D 예측이 아니라, 위치-동등 평가로 재구조화한 ALT 2D 지도 + 얕은 3D 지중 열구조 + feature ablation + AOA + calibrated UQ다.**

딥러닝 고도화는 “정적 tabular ALT에서 더 복잡한 모델을 계속 돌리는 것”이 아니라, 다음 두 경우에만 제한적으로 사용한다.

1. **고차원 입력이 들어올 때**: Sentinel-1/2, PolSAR, DEM patch, SAR time-series 등 이미지·시계열 모달리티를 feature extractor로 처리할 때.
2. **시간축 예측을 별도 파일럿으로 검증할 때**: monthly ERA5 + lagged ALT를 쓰는 T-lite GRU/TCN/Transformer가 persistence와 GBM을 실제로 이길 때.

정적 ALT mapping에서는 이미 GBM≈DL이므로, 모델 고도화보다 **데이터 구조·공변량·불확실성·전이 평가**가 논문의 핵심이다.

---

## 1. 현재까지의 확정 상태

### 1.1 초기 목표와 평가 규율

- 프로젝트 목표는 전 지구/다지역 borehole 온도 + CALM ALT + 공변량을 이용해 **ALT를 주 출력**으로 하고, 보조적으로 얕은 3D 열구조와 불확실성을 산출하는 것이다.
- 초기 Stage 0/1에서 random CV는 공간 자기상관 때문에 성능을 크게 과대평가했다. 따라서 논문 headline 지표는 **spatial block CV, LORO, kNNDM/지역전이 평가**만 사용한다.
- 무계정 WorldClim/도일 피처만으로는 LORO RMSE가 약 97 cm 수준이었고, 이는 데이터/공변량 한계를 보여주는 기준선이다.

### 1.2 모델 토너먼트와 해석 정정

기존 모델 토너먼트에서 다음 결과가 나왔다.

- GBM, MLP, FT-Transformer, TabM, Flow matching, Diffusion이 거의 동률.
- 상위 결과는 대략 앙상블 ≈ Diffusion ≈ GBM.
- 결론: **정적 ALT tabular 문제에서 병목은 모델 용량이 아니라 공변량 정보량**이다.

반드시 지켜야 할 정정 사항:

| 과거 표현 | 폐기 이유 | 올바른 표현 |
|---|---|---|
| `17 cm = 물리 하한` | 현재 R²/skill이 낮고, 새 모달리티를 아직 충분히 투입하지 않았음 | `현재 covariate set에서의 apparent floor / 정보 병목` |
| `12.97 cm = SOTA 돌파` | 평탄툰드라로 범위가 줄어 절대 RMSE가 낮아진 효과가 큼 | `레짐별 정보원 차이와 범위-정확도 trade-off` |
| `Diffusion이 정확도 우세` | GBM과 통계적으로 동률 | `UQ/샘플링 편의가 있는 보조 후보` |
| `0°C 등온면 = ALT` | ALT는 계절 최대 융해 깊이, MAGT 0°C crossing과 다름 | `ALT와 얕은 3D MAGT/thermal field는 구분` |

### 1.3 2026-07-08 데이터 재구조화 결론

가장 최신 기준선은 Thread R의 결론을 따른다.

- **1/n 위치 가중 채택**: 같은 위치/셀의 반복 점이 과도하게 모델과 metric을 지배하지 않게 한다.
- **셀 집계 채택**: 225k point를 약 14k 위치/셀 단위로 집계하고, 셀평균 ALT를 target으로 둔다.
- **셀내 SD를 UQ 라벨로 분리**: 셀 내부 대표성/미세변동을 예측 오차와 혼동하지 않는다.
- **그해 ERA5 시간정합은 mapping용으로 게이트 탈락**: 위치 고유성이 ALT mapping을 더 강하게 지배했고, annual temporal forcing은 전이/연도 holdout을 개선하지 못했다.

앞으로 모든 ALT mapping 실험은 기본적으로 다음 중 하나를 사용한다.

```text
Option A: point-level data + 1/n location weight
Option B: cell-level data + cell-mean ALT target
```

논문 headline은 **cell-level 또는 location-equal metric**만 사용한다.

---

## 2. 최근 연구 동향 기반 포지셔닝

### 2.1 CALM/PANGAEA 2025: T-lite는 가능하지만 full 4D 주제로 삼기엔 위험

- Streletskiy et al. 2025 PANGAEA/CALM dataset은 Northern Hemisphere 263개 ALT time series, 1990–2024, 22,587 points를 제공한다.
- 이 데이터는 T-lite 또는 monitoring prediction에는 좋지만, full 4D subsurface thermal volume을 robust하게 학습하기에는 site 수가 제한적이다.
- 따라서 `site-year ALT forecasting`은 보조 실험으로 두고, main paper는 static/epoch-level mapping + shallow 3D에 둔다.

### 2.2 ERA5-Land: dynamic forcing은 가능하지만, mapping 이득은 게이트 검증 필요

- ERA5-Land monthly는 1950년 이후 월별 자료를 제공하고, 2m 기온, snow, 4-layer soil temperature 등 ALT/지중열과 직접 관련된 forcing을 포함한다.
- 다만 이미 수행한 temporal gate에서 `그해 기후`는 mapping 전이 성능을 개선하지 못했다.
- 따라서 annual/monthly ERA5는 다음 목적에만 사용한다.

```text
1. 정적 climatology feature 고도화
2. lagged-ALT 기반 T-lite 예측
3. 얕은 3D thermal field의 surface forcing/prior
```

### 2.3 ESA Permafrost CCI v5: ground truth가 아니라 benchmark/prior

- ESA Permafrost CCI v5는 1997–2023년, 1 km, annual ALT와 MAGT 0/1/2/5/10 m를 제공한다.
- 이 product는 EO/reanalysis를 CryoGrid 계열 permafrost model scheme에 넣어 산출한 product다.
- 따라서 CCI는 다음처럼 사용한다.

| 사용 가능 | 사용 금지/주의 |
|---|---|
| benchmark | ground truth처럼 학습/평가 |
| prior feature | CCI를 이겼다고 단정 |
| shallow 3D GT reference | ALT와 MAGT를 혼동 |
| pseudo-label pretraining | 관측 검증 없이 SOTA claim |

### 2.4 Sequential DL 문헌: GRU/LSTM/Transformer bake-off는 신규성이 약함

최근 Alaska permafrost 관련 sequential DL 연구는 ERA5-Land dynamic features, static geology/lithology, sliding windows, scenario signal을 넣고 TCN, Transformer, Conv1DLSTM, GRU, BiLSTM을 비교했다. 이 계열에서 GRU가 sequential soil-temperature pattern에서 강한 결과를 보인 사례가 있다.

따라서 우리가 단순히 `GRU/LSTM/Transformer로 ALT 예측했다`고 하면 신규성이 약하다. T-lite를 하려면 반드시 아래 세 조건을 붙인다.

```text
1. target = soil temperature가 아니라 ALT(cm) 또는 ALT anomaly
2. baseline = persistence + GBM annual-summary
3. validation = site/region-disjoint + conformal UQ
```

GRU가 persistence와 GBM을 이기지 못하면 T-lite는 중단한다.

### 2.5 최근 GeoAI 흐름: validation, AOA, UQ가 핵심

최근 permafrost/GeoAI 연구의 공통 문제의식은 다음이다.

- random split은 spatial/temporal leakage로 성능을 과대평가한다.
- 미관측 지역에 적용할 때는 AOA 또는 covariate shift 진단이 필요하다.
- 단일 예측값이 아니라 calibrated prediction interval이 필요하다.
- 순수 ML은 extrapolation에서 물리적으로 불안정하므로, hybrid physics-ML 또는 physics consistency check가 필요하다.

따라서 본 연구의 핵심 차별성은 다음 조합이다.

```text
location-equal/cell-level ALT
+ spatial block / LORO / kNNDM
+ feature-group ablation
+ AOA mask
+ conformal calibrated UQ
+ shallow 3D thermal field
```

---

## 3. 연구 목표와 논문 프레이밍

### 3.1 권장 연구 목표

> 본 연구는 전지구/다지역 ALT 관측과 borehole 지중온도, ERA5-Land, 지형, 토양, 식생, SAR/EO 공변량을 결합하여, active layer thickness와 얕은 3D 지중 열구조를 공간 누설 없이 예측하고, 각 셀의 적용 가능 영역과 보정된 불확실성을 함께 제시하는 관측기반 GeoAI 프레임워크를 개발한다.

### 3.2 주요 연구 질문

1. **ALT mapping에서 random CV와 point-level pseudo-replication은 성능을 얼마나 왜곡하는가?**
2. **기후, snow, 지형, 토양, 식생, SAR, CCI prior 중 실제 ALT 예측에 기여하는 모달리티는 무엇인가?**
3. **모델의 예측이 어느 지역/환경에서 신뢰 가능한가?**
4. **prediction interval은 실제로 nominal coverage를 만족하는가?**
5. **ALT 2D와 borehole/CCI 기반 얕은 3D thermal field를 개념적으로 분리하면서도 하나의 프레임워크에서 연결할 수 있는가?**
6. **시계열 DL은 static GBM/persistence를 이기는가, 아니면 본 논문 범위를 넓히지 않는 것이 맞는가?**

### 3.3 논문 기여 문장 후보

- Contribution 1: `We show that naive point-level/random validation inflates ALT mapping skill and propose a cell-/location-equal evaluation protocol.`
- Contribution 2: `We quantify the contribution of climate, terrain, soil, vegetation, SAR, and CCI priors under identical spatial validation.`
- Contribution 3: `We provide cell-wise calibrated prediction intervals and AOA masks, not only deterministic ALT maps.`
- Contribution 4: `We connect ALT mapping with shallow 0–20 m ground thermal structure using borehole validation and CCI depth products as priors/benchmarks.`
- Contribution 5: `We define explicit gates for DL escalation, preventing architecture-driven overclaiming when GBM is sufficient.`

---

## 4. 실험 로드맵

## P0. 문서·metric 정정

### 목적

기존 과대표현을 논문용 표현으로 바꾸고, 모든 실험 결과가 같은 metric 체계를 따르도록 한다.

### 작업

- `SESSION_HANDOFF.md`, `docs/EXPERIMENTS.md`, `docs/EXPERIMENT_LOG.md`, `docs/PLAN_FORWARD.md`를 최신 판단으로 정리.
- `17 cm physical floor`, `12.97 SOTA breakthrough` 표현 금지.
- 모든 RMSE 옆에 다음 metric을 병기.

```text
RMSE
MAE
bias
R2
target_SD
skill_over_mean = 1 - RMSE / target_SD
coverage_90
interval_width_90
AOA_inside/outside metrics
```

### 산출물

```text
docs/PLAN_FORWARD.md
outputs/figures/02_evaluation/skill_reframing.png
data/processed/*_rescored.csv
```

---

## P1. canonical restructured baseline 확정

### 목적

앞으로 모든 실험의 기준선을 cell/location-equal setting으로 고정한다.

### 입력

```text
data/processed/dl_dataset_cell.csv
data/processed/restructure_gate_results.csv
data/processed/temporal_gate_results.csv
src/polar/eval_metrics.py
```

### 모델

- HistGradientBoostingRegressor 또는 LightGBM/XGBoost가 가능하면 LightGBM.
- 기본 모델은 GBM.
- DL 모델은 이 단계에서 사용하지 않는다.

### 비교

```text
B0: regional/cell mean baseline
B1: pooled point-level, no weight
B2: pooled point-level + 1/n location weight
B3: cell-trained, cell-mean target
B4: cell-trained + robust loss / Huber
B5: cell-trained quantile GBM
```

### 평가

- spatial block CV
- LORO
- optional: repeated spatial block or block bootstrap
- metric: RMSE, MAE, bias, R², target SD, skill-over-mean

### 통과 기준

- 이후 모든 ablation은 B2 또는 B3 중 더 정직하고 안정적인 기준선을 사용한다.
- point-level unweighted metric은 논문 headline에서 금지한다.

### 산출물

```text
data/processed/canonical_baseline_results.csv
outputs/figures/02_evaluation/canonical_baseline.png
```

---

## P2. feature-group ablation 재실행

### 목적

정확도 개선의 핵심은 모델이 아니라 데이터 모달리티다. 따라서 재구조화된 기준선 위에서 어떤 정보원이 실제로 ALT를 설명하는지 분해한다.

### feature group

| Group | Features | Rationale |
|---|---|---|
| Climate heat | MAAT, TDD, FDD, warm/cold month, LST | thaw energy |
| Snow | SWE, snow depth, snow duration, melt timing | winter insulation / spring timing |
| Terrain | elevation, slope, aspect, TPI, roughness, wetness | microclimate / drainage / snow redistribution |
| Soil | SOC, bulk density, clay/sand/silt, organic layer proxy | thermal conductivity / water retention / latent heat |
| Vegetation / surface | NDVI, landcover, shrub/forest/barren, albedo | surface energy / organic layer proxy |
| SAR / InSAR | seasonal subsidence, coherence, PolSAR proxy | thaw settlement / water-ice state |
| CCI prior | CCI ALT, CCI GT 0/1/2/5/10m | prior/benchmark, not truth |
| Location-only control | lat/lon/elevation-only or region embedding | detect spatial memorization |

### ablation matrix

```text
M0: mean / regional mean
M1: ERA5 climate only
M2: DEM terrain only
M3: ERA5 + DEM
M4: ERA5 + DEM + vegetation/landcover
M5: ERA5 + DEM + SoilGrids
M6: ERA5 + DEM + SoilGrids + vegetation
M7: M6 + Sentinel-1/InSAR time-series features
M8: M6 + PolSAR, North Slope / ABoVE only
M9: M6 + ESA CCI ALT/GT prior
M10: M9 + SAR, full feature set
```

### implementation note

Create or update:

```text
scripts/2_evaluation/alt_feature_ablation_cell_weighted.py
src/polar/eval_metrics.py
src/polar/feature_groups.py
```

Every ablation must use:

```text
same folds
same target
same weights
same metric function
same train-only scaling if scaling is needed
```

### 통과 기준

- 각 group의 Δskill, ΔRMSE, Δcoverage, ΔAOA-inside performance를 보고한다.
- feature가 RMSE를 낮춰도 transfer/LORO나 coverage를 악화시키면 “주력 feature”로 채택하지 않는다.

### 산출물

```text
data/processed/alt_feature_ablation_cell_weighted.csv
outputs/figures/06_deep_learning/alt_feature_ablation_cell_weighted.png
outputs/figures/06_deep_learning/feature_group_delta_skill.png
```

---

## P3. AOA + conformal UQ

### 목적

ALT map에서 중요한 것은 `값`뿐 아니라 `그 값을 믿을 수 있는 영역과 오차 범위`다.

### AOA

- predictor space에서 train data와의 dissimilarity index를 계산한다.
- model feature importance로 distance를 가중한다.
- CV fold의 train-test dissimilarity 분포를 기준으로 threshold를 정한다.
- AOA inside/outside metric을 분리한다.

### conformal UQ

기본 방식:

```text
1. base model: GBM or quantile GBM
2. split: spatial block / LORO-aware train-calibration-test
3. nonconformity score: absolute residual or quantile residual
4. target: nominal 90% prediction interval
5. report: observed coverage, interval width, inside/outside AOA coverage
```

### implementation note

Create or update:

```text
scripts/2_evaluation/alt_aoa_conformal_cell.py
src/polar/aoa.py
src/polar/conformal.py
```

### 통과 기준

- 전체 90% nominal interval의 observed coverage가 최소한 85–95% 범위에 들어와야 한다.
- 지역별 coverage가 붕괴하면 region-wise conformal 또는 AOA-conditioned conformal로 보정한다.
- coverage가 맞지 않으면 Diffusion/GBM의 native UQ claim 금지.

### 산출물

```text
data/processed/alt_conformal_aoa_cell_results.csv
outputs/maps/alt_aoa_mask.png
outputs/maps/alt_uncertainty_width_90.png
outputs/figures/02_evaluation/coverage_calibration_cell.png
```

---

## P4. local model과 general model 분리

### 목적

`전역/다지역 일반화 모델`과 `North Slope 평탄툰드라 국소 고정밀 모델`을 같은 성능표로 섞지 않는다.

### 실험 트랙

| Track | Domain | 목적 | 평가 |
|---|---|---|---|
| General | pan-Arctic / North America / multi-region | 관측 희소 지역 mapping | spatial block, LORO, AOA, conformal UQ |
| Local | North Slope flat tundra | 고해상도 local ALT field | local spatial block, holdout site, AOA |
| Regime | flat tundra / wet tundra / mountain / forest / barren | 지형·생태별 정보원 차이 | regime-wise ablation |

### 주의

- `local RMSE 12–13 cm`를 `global SOTA`라고 쓰지 않는다.
- local에서는 target variance가 줄어들기 때문에 반드시 `skill-over-mean`과 `R²`를 같이 보고한다.

### 산출물

```text
data/processed/regime_scope_results_cell.csv
outputs/figures/06_deep_learning/accuracy_vs_scope_rescored.png
outputs/maps/local_demo_alt_field_verified.png
```

---

## P5. 얕은 3D thermal field

### 목적

원래 연구 목표인 `ALT 및 지하 온도에 대한 3D 공간 예측`을 살리되, full 4D로 범위를 넓히지 않는다.

### 개념 분리

```text
ALT = 계절 최대 융해 깊이, cm
MAGT / annual ground temperature = 연평균 지중온도, °C
0°C annual isotherm = MAGT field의 crossing, ALT와 동일하지 않음
```

### 입력

```text
ground_temp_all.csv
GTN-P / UAF / USGS / PERMOS borehole temperature
ERA5-Land climate/snow/soil temperature
DEM / terrain
SoilGrids / ground ice / organic proxy
ESA CCI MAGT 0/1/2/5/10m as prior or benchmark
CCI ALT as benchmark, not truth
```

### 모델

주력:

```text
GBM conditional field
input = x, y, depth encoding, climate, snow, terrain, soil, vegetation
output = temp_c or MAGT at depth
```

비교 baseline:

```text
IDW-3D
nearest-borehole
CCI GT direct benchmark
GBM without CCI prior
GBM with CCI prior
```

DL 사용 조건:

- from-scratch neural field/SIREN은 재시도하지 않는다. 이미 GBM/IDW에 패배했다.
- DL을 쓰려면 다음 중 하나만 허용한다.

```text
1. CCI depth stack + covariates를 이용한 residual correction network
2. borehole profile sequence를 depth-wise encoder로 쓰는 small model
3. future T2용 DeepONet/FNO prototype, main paper 아님
```

### 평가

```text
leave-one-borehole-out
leave-one-region-out
depth-wise RMSE and bias
site-wise RMSE
vertical smoothness check
physical consistency check
UQ coverage by depth
```

### physical consistency checks

- 0–20 m profile이 과도하게 요동치지 않는가?
- 연평균 10 m 온도가 표층보다 더 계절 forcing에 민감하게 나오지 않는가?
- 지열구배가 비현실적으로 크거나 작은가?
- ALT와 MAGT 0°C crossing을 혼동하지 않았는가?
- borehole 없는 AOA outside 지역의 uncertainty가 충분히 넓어지는가?

### 산출물

```text
data/processed/thermal3d_gbm_results.csv
outputs/maps/thermal3d_depth_slices_0_1_2_5_10_20m.png
outputs/figures/07_thermal3d/borehole_profile_validation.png
outputs/volumes_3d/thermal_cube_alaska.vti
outputs/maps/thermal3d_uncertainty_width.png
```

---

## P6. 딥러닝 고도화 로드맵

## P6-A. 정적 tabular ALT: DL 고도화 중단

### 판단

정적 14-feature 또는 약간 확장된 tabular feature에서는 GBM이 강하고, FT-Transformer/MLP/Diffusion이 통계적으로 GBM을 압도하지 못했다. 따라서 정적 tabular ALT에서 아키텍처 경쟁을 계속하는 것은 논문 가치가 낮다.

### 허용 모델

```text
GBM / LightGBM / XGBoost
quantile GBM
GBM ensemble
conformal wrapper
```

### 금지

```text
새로운 tabular Transformer 추가
더 큰 MLP 추가
Diffusion을 정확도 모델로 주장
TabM 재튜닝 반복
```

---

## P6-B. 고차원 EO/SAR 입력: DL 허용

### 사용 조건

Sentinel-1/2, PolSAR, DEM patch, SAR time-series 등 고차원 feature가 들어올 때만 CNN/ViT/Temporal CNN을 사용한다.

### 후보

```text
Sentinel-2 patch encoder: small CNN / pretrained ViT feature extractor
Sentinel-1 seasonal stack encoder: Temporal CNN / lightweight Transformer
DEM patch encoder: small CNN only if scalar terrain feature보다 좋을 때
SAR + tabular fusion: late fusion preferred
```

### 실험 원칙

```text
1. 먼저 handcrafted feature ablation을 돌린다.
2. handcrafted SAR/vegetation feature가 효과를 보이면 DL encoder를 시도한다.
3. DL encoder가 GBM+handcrafted feature를 spatial CV/LORO에서 이기지 못하면 폐기한다.
```

### 산출물

```text
data/processed/multimodal_encoder_results.csv
outputs/figures/06_deep_learning/multimodal_encoder_vs_handcrafted.png
```

---

## P6-C. T-lite sequence forecast: 보조 게이트 실험

### 목적

full 4D가 아니라, `site-year ALT monitoring prediction`만 작은 파일럿으로 검증한다.

### target

```text
y(site, year+1) = ALT_cm(site, year+1)
or
y = ALT_anomaly relative to site mean
```

### 입력

```text
lagged ALT: ALT(t), ALT(t-1), trend over last k years
monthly ERA5-Land: T2m, SWE/snow depth, STL1-4, TDD/FDD
static: DEM, soil, vegetation, landcover
optional: previous-year CCI ALT/GT anomaly
```

### baselines

```text
B0: persistence, ALT(t+1)=ALT(t)
B1: site climatology / site mean
B2: GBM annual summary
B3: GRU monthly sequence
B4: TCN monthly sequence
B5: Transformer monthly sequence, only if data suffices
```

### validation

```text
site-disjoint split
region-disjoint split
temporal holdout: train past, test future
per-horizon conformal UQ
```

### success gate

T-lite는 다음을 모두 만족할 때만 논문 보조 결과로 유지한다.

```text
GRU/TCN RMSE < persistence RMSE
GRU/TCN RMSE < GBM annual-summary RMSE
coverage_90 within 85–95%
region-disjoint performance does not collapse
```

하나라도 실패하면 T-lite는 appendix 또는 future work로 내린다.

### 산출물

```text
scripts/3_deep_learning/tlite_alt_sequence_gate.py
data/processed/tlite_sequence_gate_results.csv
outputs/figures/06_deep_learning/tlite_sequence_gate.png
```

---

## P6-D. Operator learning / full 4D: 후속 논문 후보

### 판단

DeepONet/FNO/operator learning은 현재 main paper에는 넣지 않는다. 단, 다음 조건이 충족되면 후속 논문 또는 pilot로 이동한다.

```text
1. P5 shallow 3D thermal field가 안정적으로 검증됨
2. T-lite가 persistence/GBM을 이김
3. GIPL2/CryoGrid synthetic column corpus를 만들 수 있음
4. time-resolved label/prior가 충분히 정합됨
```

### possible future model

```text
DeepONet:
branch = location covariates + monthly forcing history
trunk = query coordinate (x, y, z, t)
output = T(x,y,z,t) or ALT(x,y,t)

FNO:
input field = annual/monthly climate and prior state grids
output field = ALT / shallow GT map
```

### warning

- `2100 projection`을 headline으로 내세우면 기존 CryoGrid/GIPL/CCI/projection 연구와 바로 경쟁한다.
- 우리의 빈 공간은 projection 자체가 아니라 **transfer-honest + calibrated UQ + shallow subsurface volume**이다.

---

## 5. 실행 순서

Claude Code는 아래 순서로 진행한다.

```text
Step 0. 최신 문서 확인
  - docs/PLAN_FORWARD.md
  - SESSION_HANDOFF.md
  - docs/EXPERIMENTS.md
  - docs/EXPERIMENT_LOG.md
  - gpt/handoff/20260708_1538-data-restructuring-thread-r.md

Step 1. metric canonicalization
  - eval_metrics.py 확인/확장
  - all result CSV에 RMSE/MAE/bias/R2/target_SD/skill 병기

Step 2. canonical baseline
  - cell-level + 1/n-weight baseline 고정
  - canonical_baseline_results.csv 생성

Step 3. feature-group ablation
  - M0–M10 실행
  - alt_feature_ablation_cell_weighted.csv 생성

Step 4. AOA + conformal UQ
  - AOA mask 생성
  - 90% conformal interval coverage 검증

Step 5. shallow 3D thermal field
  - GBM conditional field
  - CCI GT 0/1/2/5/10m benchmark/prior
  - borehole validation

Step 6. optional T-lite gate
  - persistence vs GBM vs GRU/TCN
  - 실패 시 future work로 이동
```

---

## 6. 중단 조건

아래 조건이 발생하면 해당 방향은 즉시 중단하거나 appendix/future work로 강등한다.

| 방향 | 중단 조건 |
|---|---|
| 정적 DL | GBM을 spatial block/LORO에서 유의하게 이기지 못함 |
| Diffusion UQ | conformal 보정 전 coverage가 낮고 보정 후에도 interval이 과도하게 넓음 |
| SAR/PolSAR | local에서는 좋아도 general/LORO를 악화시킴 |
| CCI prior | CCI를 넣었을 때 관측 독립성이 흐려지거나 leakage 의심 발생 |
| T-lite | persistence 또는 GBM annual-summary를 이기지 못함 |
| neural field | GBM/IDW보다 leave-one-borehole-out에서 나쁨 |
| full 4D | 데이터 정합·검증 protocol이 명확하지 않음 |

---

## 7. 논문 결과표 기본 형식

모든 주요 표는 아래 형식을 따른다.

```text
Model / Feature Set / Domain / CV Type / N cells / RMSE / MAE / Bias / R2 / target_SD / skill_over_mean / coverage_90 / interval_width_90 / AOA_inside_RMSE / AOA_outside_RMSE
```

단순 RMSE만 있는 표는 만들지 않는다.

---

## 8. 논문 그림 구성 후보

```text
Fig 1. Data and task diagram: CALM/ABoVE/GTN-P/ERA5/DEM/Soil/SAR/CCI
Fig 2. Leakage and pseudo-replication: random CV vs spatial block vs LORO; point vs cell
Fig 3. Canonical baseline and feature ablation
Fig 4. AOA map and AOA inside/outside performance
Fig 5. Conformal UQ: nominal vs observed coverage, interval width map
Fig 6. Local vs general/regime trade-off
Fig 7. Shallow 3D thermal field depth slices and borehole validation
Fig 8. Optional T-lite gate: persistence vs GBM vs GRU/TCN
```

---

## 9. 외부 문헌·데이터 근거 메모

Claude Code는 아래 근거를 참고하되, 논문에는 정확한 bibliographic metadata를 다시 확인해 인용한다.

1. Streletskiy et al. 2025, PANGAEA CALM ALT dataset 1990–2024, 263 time series, 22,587 data points.  
   URL: https://doi.pangaea.de/10.1594/PANGAEA.972777

2. ERA5-Land monthly averaged data from 1950 to present. 0.1° grid in CDS, native 9 km, monthly temporal resolution, 4 soil layers down to 289 cm.  
   URL: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means

3. ESA Permafrost CCI v5.0. ALT 1997–2023, 1 km annual maximum thaw depth; MAGT 0/1/2/5/10 m, 1997–2023.  
   URL: https://climate.esa.int/en/projects/permafrost/

4. Rahaman 2025 arXiv. Sequential DL for Alaskan permafrost soil temperature; TCN, Transformer, Conv1DLSTM, GRU, BiLSTM; GRU best for sequential temperature pattern detection.  
   URL: https://arxiv.org/abs/2510.06258

5. Kriuk 2025 arXiv. Hybrid physics-ML pan-Arctic permafrost infrastructure risk; emphasizes spatiotemporal validation, UQ, hybrid physics-ML.  
   URL: https://arxiv.org/abs/2510.02189

6. Meyer & Pebesma 2020/2021. Area of Applicability for spatial prediction models.  
   URL: https://arxiv.org/abs/2005.07939

7. Lou et al. 2024/2025. GeoConformal prediction for model-agnostic uncertainty in spatial prediction.  
   URL: https://arxiv.org/abs/2412.08661

---

## 10. 최종 메시지

Claude Code는 다음 원칙을 지킨다.

```text
- 모델을 더 복잡하게 만들기 전에 데이터 구조와 metric을 고정한다.
- random CV 성능은 headline에 쓰지 않는다.
- point-level unweighted metric은 headline에 쓰지 않는다.
- 12.97cm와 17cm를 과대해석하지 않는다.
- ALT와 MAGT/0°C annual isotherm을 혼동하지 않는다.
- DL은 정적 tabular가 아니라 고차원 EO/SAR 또는 시간축에서만 사용한다.
- 모든 지도 산출물에는 AOA와 calibrated uncertainty를 같이 낸다.
- full 4D는 현재 main paper가 아니라 후속 게이트 후보로 둔다.
```
