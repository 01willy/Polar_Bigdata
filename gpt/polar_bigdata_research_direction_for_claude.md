# Polar_Bigdata 연구 방향 검토 및 Claude Code 작업 지시

작성일: 2026-07-07  
목적: 현재 ALT/지중온도 예측 연구 결과를 정리하고, Claude Code가 다음 실험을 혼선 없이 이어갈 수 있도록 연구 방향과 구현 우선순위를 지정한다.

---

## 0. 핵심 결론

현재 연구의 가장 좋은 방향은 **무리하게 4D 전체 문제로 확장하는 것**이 아니라, 우선 다음 형태로 논문 축을 잡는 것이다.

> **ALT 2D + 얕은 3D 지중 열구조를 대상으로, 공간 누설을 통제한 평가, feature ablation, 적용 가능 영역(AOA), 보정된 불확실성(UQ)을 결합한 관측 기반 딥러닝/ML 프레임워크**

즉, 논문 프레이밍은 다음처럼 수정한다.

- 피해야 할 주장:
  - “Diffusion/LSTM/Transformer가 GBM보다 정확도를 크게 높였다.”
  - “17 cm가 물리적 floor다.”
  - “평탄툰드라 12.97 cm가 SOTA를 돌파했다.”
  - “ALT와 0°C 연평균 등온면은 같은 물리량이다.”

- 유지해야 할 주장:
  - “무작위 CV는 공간 누설로 성능을 과대평가한다.”
  - “정적 ALT 예측에서는 모델 아키텍처보다 covariate 정보량이 병목이다.”
  - “ERA5-Land 같은 더 물리적인 forcing은 transfer/LORO 성능을 실제로 개선했다.”
  - “3D from-scratch neural field는 현재 데이터에서 GBM/IDW보다 약하므로 폐기하는 것이 합리적이다.”
  - “차별성은 정확도 숫자 하나가 아니라, transfer-honest 평가 + ablation + calibrated UQ + shallow 3D 구조 결합이다.”

---

## 1. 현재 연구 결과 해석

### 1.1 현재까지 의미 있는 결과

현재까지 의미 있는 성과는 다음이다.

| 축 | 판단 |
|---|---|
| 데이터 구축 | CALM, ABoVE, GTN-P, ERA5-Land, DEM, InSAR/PolSAR weak labels까지 연구용 파이프라인이 상당히 구축됨 |
| 평가 규율 | random CV, spatial block, LORO 차이를 정량화했고, random CV 누설 위험을 확인함 |
| 모델 비교 | GBM, MLP, FT-Transformer, TabM, Flow, Diffusion을 동일 feature와 동일 spatial CV에서 비교함 |
| 결과 해석 | 모델 교체보다 covariate 정보량이 병목이라는 결론이 설득력 있음 |
| 3D 방향 | neural field가 GBM/IDW보다 못해 kill-switch를 작동시킨 점은 좋은 연구 판단 |
| 문헌상 위치 | 순수 시계열 DL보다는 transfer/UQ/3D 결합 쪽이 더 차별성이 있음 |

### 1.2 정정해야 할 결과 해석

기존 문서의 일부 표현은 과하다.

| 기존 표현 | 문제 | 수정 표현 |
|---|---|---|
| 17 cm = 물리하한 | 현재 feature set의 설명력이 낮아서 생긴 apparent floor에 가까움 | “현재 covariate set에서의 정보 병목” |
| 12.97 cm = SOTA급 돌파 | 평탄툰드라로 범위를 좁혀 target variance가 줄어든 효과가 큼 | “레짐별 정보원 차이 확인” |
| Diffusion 채택 | RMSE 우위가 아니라 GBM과 통계적으로 동률 | “UQ/샘플링 편의 때문에 보조 채택 가능” |
| 0°C 등온면 = ALT | ALT는 계절 최대 융해 깊이, MAGT 0°C 등온면과 다름 | ALT와 MAGT/thermal field를 분리 |

---

## 2. GRU 설명

GRU는 **Gated Recurrent Unit**의 약자다. LSTM과 같은 계열의 recurrent neural network이며, 시간 순서가 있는 자료에서 “과거 정보를 얼마나 기억하고, 얼마나 버릴지”를 gate로 조절한다.

간단히 말하면:

```text
월별 기온, 적설, 토양온도 sequence
        ↓
GRU
        ↓
다음 해 ALT 또는 지중온도 예측
```

LSTM과의 차이:

| 항목 | LSTM | GRU |
|---|---|---|
| gate 수 | 더 많음 | 더 적음 |
| 구조 | 복잡 | 단순 |
| 학습 속도 | 상대적으로 느림 | 상대적으로 빠름 |
| 데이터가 적을 때 | 과적합 위험 있음 | 더 안정적인 경우 많음 |
| 해석 | 둘 다 black-box 성격 | 둘 다 black-box 성격 |

이번 연구에 GRU를 쓰려면 현재 `dl_dataset.csv`처럼 정적 feature만 넣으면 안 된다. `site-year`별로 월별/계절별 ERA5-Land sequence를 만들어야 한다.

---

## 3. 현재 결과는 기존 논문 대비 의미 있는가?

의미는 있다. 다만 “정확도만으로 강하다”는 방향은 약하다.

현재 결과의 해석은 다음처럼 해야 한다.

```text
정적 ALT 예측에서 GBM ≈ DL이라는 결과는 실패가 아니다.
오히려 현재 공변량만으로는 모델 capacity가 병목이 아니라는 증거다.
따라서 다음 novelty는 모델을 더 복잡하게 만드는 것이 아니라,
어떤 데이터가 실제로 ALT/지중온도 예측력을 주는지 ablation하고,
그 예측을 어디까지 신뢰할 수 있는지 UQ/AOA로 보여주는 것이다.
```

논문에서 쓸 수 있는 포인트:

1. random CV가 얼마나 과대평가하는지 보여준다.
2. spatial block/LORO에서 성능을 정직하게 보고한다.
3. 기존 gridded product와 관측 기반 모델을 같은 protocol로 비교한다.
4. feature group ablation으로 “무엇이 ALT를 설명하는가”를 분해한다.
5. prediction interval coverage를 실제로 검증한다.
6. ALT 2D와 얕은 3D thermal field를 구분해서 함께 제시한다.

---

## 4. 4D로 갈지, 3D에 한정할지

### 4.1 권고: 지금은 full 4D로 가지 말 것

full 4D는 다음을 모두 요구한다.

```text
x, y, z, t → T 또는 ALT
```

필요한 데이터:

- 연별/monthly CALM ALT time series
- 연별/monthly ERA5-Land forcing
- borehole temperature time series
- CCI annual ground temperature 0/1/2/5/10 m
- CMIP forcing 또는 scenario forcing
- 시간축 validation protocol

문제는 다음이다.

| 위험 | 설명 |
|---|---|
| 연구 범위 폭발 | ALT 2D, 지중온도 3D, 시간예측, scenario projection이 한 논문에 모두 들어감 |
| 데이터 정합 부담 | CALM ALT, borehole temperature, CCI, ERA5의 기간·해상도·물리량이 다름 |
| 차별성 희석 | “우리가 정확히 뭘 새로 했는가”가 흐려질 수 있음 |
| 논문 방어 난이도 상승 | 4D projection은 기존 CryoGrid/GIPL/CCI 계열과 직접 비교해야 함 |

### 4.2 더 좋은 타협안: 3D+T-lite

추천 방향은 full 4D가 아니라 **3D+T-lite**다.

```text
1단계 논문:
ALT 2D + 얕은 3D thermal field, 정적/epoch 평균

2단계 확장:
동일 framework를 site-year 또는 annual forcing으로 확장하는 T-lite 파일럿
```

즉, 이번 논문에서는 다음까지만 목표로 한다.

```text
입력:
- ERA5-Land climate forcing
- DEM terrain
- soil/organic/ice proxies
- vegetation/surface state
- optional SAR features
- depth encoding

출력:
- ALT(x,y)
- shallow T(x,y,z ≤ 20 m)
- uncertainty interval
- AOA mask
```

시간축은 본 논문에서 전면 주제가 아니라, 보조 실험으로만 둔다.

```text
T-lite optional:
site-year ALT 예측에서
static model vs annual forcing model vs GRU 비교
```

이렇게 하면 원래 연구 목표인 **3D 공간 예측**을 유지하면서도 최신 연구 대비 차별성을 만들 수 있다.

---

## 5. 데이터와 feature ablation 계획

현재 가장 필요한 것은 “불필요한 feature를 줄이는 것”이 아니라, **feature group별로 실제 기여도를 정직하게 측정하는 것**이다.

### 5.1 ALT에 중요한 feature group

ALT에 직접 영향을 주는 feature는 다음 순서로 본다.

| 그룹 | 예시 | 물리적 의미 |
|---|---|---|
| Heat forcing | MAAT, TDD, FDD, warm-season temperature, LST | 여름에 얼마나 녹일 에너지가 있는가 |
| Snow | SWE, snow depth, snow duration, snow melt timing | 겨울 insulation과 봄 thaw timing |
| Soil/organic/ice | SOC, bulk density, clay/sand, ground ice class | 열전도도, 잠열, 수분보유 |
| Vegetation/surface | NDVI, landcover, shrub/forest, albedo | 표면 에너지와 유기층 proxy |
| Terrain/hydrology | elevation, slope, aspect, TPI, wetness index | 미세기후, 배수, snow redistribution |
| SAR/InSAR | seasonal subsidence, coherence, PolSAR ALT proxy | thaw settlement, 수분/얼음 관련 proxy |
| Lagged state | previous ALT, previous winter/summer anomaly | site memory, interannual persistence |

### 5.2 지중온도 3D에 중요한 feature group

지중온도는 ALT보다 더 직접적으로 열전달 문제다.

| 그룹 | 예시 |
|---|---|
| Depth | depth_m, log_depth, Fourier depth encoding |
| Surface forcing | surface/air temperature, LST, TDD/FDD |
| Snow insulation | SWE, snow depth, snow duration |
| Soil thermal properties | soil texture, organic carbon, bulk density, ground ice |
| Hydrology | soil moisture proxy, wetness, runoff, water mask |
| Geothermal/bedrock | geothermal flux, lithology proxy, elevation/region |
| Temporal epoch | year, climate normal period, warming anomaly |

### 5.3 Ablation 실험 설계

Claude Code가 구현할 ablation은 다음 순서로 진행한다.

```text
ALT 2D ablation:
M0: mean / regional mean
M1: DEM only
M2: ERA5 climate only
M3: ERA5 + DEM
M4: ERA5 + DEM + vegetation/landcover
M5: ERA5 + DEM + soil/organic/ice
M6: ERA5 + DEM + soil + vegetation
M7: M6 + Sentinel-1/InSAR time-series features
M8: M6 + PolSAR, 단 ABoVE/North Slope 한정
M9: M6 + CCI ALT prior, 단 benchmark/feature로만 사용
```

```text
3D temperature ablation:
T0: IDW / nearest / kriging baseline
T1: depth only + regional mean
T2: depth + ERA5
T3: depth + ERA5 + DEM
T4: depth + ERA5 + soil/ice
T5: T4 + CCI ground temperature 0/1/2/5/10m prior
T6: T5 + physics residual constraint / monotonicity check
```

평가 metric은 반드시 다음을 모두 출력한다.

```text
RMSE
MAE
bias
R²
skill-over-mean = 1 - RMSE / target_SD
coverage_90
interval_width_90
AOA inside/outside RMSE
region-wise RMSE
terrain-regime-wise RMSE
```

Random CV는 참고용으로만 유지하고, headline은 spatial block, LORO, kNNDM/AOA 기준으로만 쓴다.

---

## 6. 기존 논문 대비 정확도 해석

ALT 예측 정확도는 “몇 cm면 충분한가”를 단독으로 말할 수 없다. 반드시 다음 세 가지를 같이 봐야 한다.

```text
절대오차(cm)
상대오차(%)
대상 ALT 분산/범위 대비 설명력(R², skill-over-mean)
```

예를 들어 ALT가 40 cm인데 30 cm로 예측하면:

```text
절대오차 = 10 cm
상대오차 = 25%
```

이 값은 연구 목적에 따라 다르게 해석된다.

| 목적 | 10 cm 오차 의미 |
|---|---|
| 넓은 지역 경향 지도 | 꽤 괜찮을 수 있음 |
| grid-scale climate validation | 유용할 수 있음 |
| 공학 설계/구조물 기초 | 부족할 가능성이 큼 |
| 특정 지점 장기 모니터링 | 변화량이 5–10 cm라면 의미가 애매함 |
| 생태/수문 regime 분류 | boundary 근처에서는 중요할 수 있음 |

따라서 본 연구에서 해야 할 말은 다음이다.

```text
우리 모델은 특정 지점의 엔지니어링 설계값을 주는 모델이 아니라,
관측이 희소한 지역에서 ALT/얕은 지중 열구조의 공간 패턴과 불확실성을 제공하는 모델이다.
```

기존 연구와 비교할 때도 다음 원칙을 지킨다.

- 좁은 지역, 좁은 ALT 범위, 공동위치 센서 기반 연구와 전역/다지역 지도 RMSE를 직접 비교하지 않는다.
- SOTA 값이 10–12 cm라도 그것이 같은 문제인지 먼저 확인한다.
- 넓은 범위의 LORO/transfer RMSE는 수십 cm 이상이 나와도 이상하지 않다.
- Ran류의 넓은 범위 ALT mapping과 비교할 때는 spatial protocol, 범위, target SD를 같이 본다.

---

## 7. Transfer가 정확도보다 중요한가?

정답은 “목적에 따라 다르다”이다. 하지만 이번 연구 목표에서는 transfer를 버리면 논문이 약해진다.

### 7.1 순수 정확도가 중요한 경우

다음 목적이면 transfer보다 같은 지역 내 정확도가 우선이다.

- North Slope 특정 현장용 고해상도 ALT 지도
- ABoVE 관측 범위 내부 interpolation
- 특정 생태/공학 현장의 local decision support
- 같은 센서/같은 지형/같은 분포 내부 배포

이 경우에는 local spatial block CV와 hold-out site 검증을 주 metric으로 써도 된다.

### 7.2 Transfer가 중요한 경우

다음 목적이면 transfer가 핵심이다.

- Alaska에서 학습해 Canada/Siberia/QTP로 확장
- 관측 없는 지역에 ALT map 생성
- 논문에서 “generalizable model” 주장
- ESA CCI/Ran/GIPL류와 독립적으로 비교
- climate scenario나 future projection으로 확장

현재 프로젝트 목표는 원래 “다지역 ALT + 얕은 3D 열구조 + 전이 검증”이므로, transfer를 빼면 차별성이 줄어든다.

따라서 권장 프레임은 다음이다.

```text
1. Local accuracy를 최대화하는 모델도 만든다.
2. 하지만 논문 headline은 transfer-honest accuracy + UQ + AOA로 둔다.
3. 두 결과를 분리해서 제시한다.
```

---

## 8. Claude Code에게 줄 구현 우선순위

### Priority 0 — 문서 정정

다음 문서를 수정한다.

- `SESSION_HANDOFF.md`
- `docs/EXPERIMENTS.md`
- `docs/EXPERIMENT_LOG.md`
- `README.md`

수정 사항:

```text
- “17cm = 물리하한” 표현 삭제
- “12.97cm = SOTA 돌파” 표현 삭제
- 모든 RMSE 옆에 R², target SD, skill-over-mean 병기
- ALT와 MAGT/0°C 등온면의 개념 분리
- Diffusion은 정확도 우위가 아니라 UQ/샘플링 편의로 위치 조정
```

### Priority 1 — metric 표준화

새 utility를 만든다.

```text
src/polar/eval_metrics.py
```

필수 함수:

```python
rmse(y, yhat)
mae(y, yhat)
bias(y, yhat)
r2(y, yhat)
skill_over_mean(y, yhat)  # 1 - RMSE / std(y)
coverage(y, lower, upper)
interval_width(lower, upper)
grouped_metrics(df, group_cols)
```

모든 실험 결과 CSV에 다음 컬럼을 넣는다.

```text
rmse_cm
mae_cm
bias_cm
r2
target_sd_cm
skill_over_mean
n
cv_type
region
scope
feature_group
```

### Priority 2 — ALT feature ablation

새 스크립트:

```text
scripts/3_deep_learning/alt_feature_ablation.py
```

기능:

- feature group dictionary 정의
- GBM baseline 고정
- optional MLP/FT-T는 나중
- spatial block CV + LORO
- group별 ablation 결과 저장

출력:

```text
data/processed/alt_feature_ablation_results.csv
outputs/figures/06_deep_learning/alt_feature_ablation.png
```

### Priority 3 — AOA + conformal UQ

새 스크립트:

```text
scripts/2_evaluation/aoa_conformal_alt.py
```

기능:

- training feature space 기준 dissimilarity index 계산
- AOA inside/outside mask
- split conformal 또는 spatial-block conformal interval
- nominal 90% interval의 observed coverage 평가

출력:

```text
data/processed/alt_conformal_aoa_results.csv
outputs/maps/alt_aoa_mask.png
outputs/maps/alt_uncertainty_width.png
```

### Priority 4 — 3D 정적 조건장 강화

새 스크립트:

```text
scripts/3_deep_learning/thermal3d_conditioned_gbm.py
```

기능:

- `ground_temp_all.csv` 사용
- depth encoding + ERA5 + DEM + soil/CCI ground temperature feature
- GBM 조건장
- site-wise GroupKFold + LORO
- IDW/kriging/nearest와 비교
- 0–20m depth slice 생성

출력:

```text
data/processed/thermal3d_ablation_results.csv
outputs/volumes_3d/thermal_cube_static.nc
outputs/maps/thermal3d_depth_slices.png
```

### Priority 5 — T-lite 파일럿만 수행

full 4D는 보류한다. 대신 작은 파일럿만 만든다.

```text
scripts/1_data_prep/assemble_alt_site_year_sequence.py
scripts/3_deep_learning/train_alt_tlite_gru.py
```

입력:

```text
CALM site-year ALT
ERA5-Land monthly forcing for year and previous year
static DEM/soil/vegetation
```

모델:

```text
GBM annual-summary baseline
GRU monthly-sequence model
```

평가:

```text
leave-site-out
leave-region-out
per-year holdout
```

판정 기준:

```text
GRU가 GBM annual-summary보다 spatial/site transfer에서 의미 있게 개선될 때만 T1/T2 확장.
개선이 없으면 본 논문에서는 T-lite를 appendix/negative result로 처리.
```

---

## 9. 최종 추천 연구 제목

1순위 제목:

> Transfer-honest prediction of active layer thickness and shallow permafrost thermal structure with calibrated uncertainty

2순위 제목:

> A spatially calibrated framework for mapping active layer thickness and shallow ground temperature across permafrost regions

3순위 제목:

> From ALT maps to shallow 3D thermal fields: leakage-aware machine learning for permafrost monitoring

---

## 10. 논문 핵심 문장 후보

```text
We find that replacing gradient boosting with modern deep architectures does not substantially improve static ALT prediction under spatially blocked validation, indicating that the dominant bottleneck is covariate information rather than model capacity.
```

```text
Instead of claiming a single global accuracy number, we evaluate where the model is applicable, how prediction intervals are calibrated, and which environmental information sources control performance across terrain regimes.
```

```text
Our contribution is not a new neural architecture, but a transfer-honest and uncertainty-calibrated framework that links 2D active layer thickness mapping with shallow 3D permafrost thermal structure.
```

```text
Full 4D projection is left as a follow-up; in this study, temporal information is used only in a limited site-year pilot to test whether dynamic forcing provides predictive value beyond static climatological summaries.
```

---

## 11. Stop conditions

다음 조건이면 해당 방향은 중단한다.

| 방향 | 중단 조건 |
|---|---|
| 정적 DL | GBM 대비 spatial block/LORO 개선 없음 |
| Diffusion | conformal 보정 후에도 interval이 너무 넓거나 coverage 불안정 |
| InSAR 단일 feature | ablation에서 반복적으로 악화 |
| PolSAR 전역 사용 | 평탄툰드라 외 지역에서 skill 없음 |
| full 4D | T-lite에서 GRU/sequence forcing이 GBM annual-summary보다 개선 없음 |
| neural field | GBM/IDW보다 site-wise 또는 LORO에서 열세 |

---

## 12. Claude Code 작업 요약

바로 할 일:

```text
1. 문서 정정: 17cm floor / 12.97 SOTA 표현 제거
2. eval_metrics.py 추가
3. 기존 결과 CSV에 target_sd, R², skill-over-mean 재계산
4. ALT feature ablation 스크립트 작성
5. AOA + conformal UQ 스크립트 작성
6. 3D 정적 GBM 조건장 강화
7. T-lite GRU 파일럿은 마지막에 작은 규모로만 수행
```

최종 목표:

```text
정확도 숫자 하나가 아니라,
어떤 데이터가 ALT/지중온도를 설명하는지,
어디에서 예측이 신뢰 가능한지,
그 불확실성이 실제 coverage를 만족하는지를 보여주는 논문으로 정리한다.
```
