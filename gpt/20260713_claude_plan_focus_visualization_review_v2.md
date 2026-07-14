# Polar_Bigdata Claude Code 실행계획 검토 v2: 연구 집중도, 차별성, 시각화, 예상 문제점

작성일: 2026-07-13  
수정 의도: v1의 **예선 전 실험 제한** 중심 서술을 제거한다. 실험 시간은 충분하다고 가정하되, 연구 질문이 흐려지지 않도록 **핵심 주장, 실험 게이트, 시각화 품질, 과장 방지**를 고정한다.  
적용 대상: `EXPERIMENT_LOG.md`, `PLAN_FORWARD.md`, `CONTEST_PLAN_2026.md`, `DESIGN_BRIEF_MIDREPORT.md` 기반의 다음 Claude Code 작업 세션.

---

## 0. 먼저 읽을 결론

현재 연구는 더 넓게 실험할 수 있다. 다만 모든 실험은 다음 세 문장 중 하나를 강화해야 한다.

1. **ALT 예측에서 모델보다 데이터 구조와 공변량 정보량이 중요하다.**
2. **예측값만이 아니라 적용 가능 영역(AOA)과 보정된 불확실성까지 제시해야 한다.**
3. **2D ALT 지도와 얕은 3D 지중 열구조를 관측 기반으로 연결하는 것이 연구 차별성이다.**

따라서 Claude Code는 실험을 줄이는 것이 아니라, 실험을 다음 기준으로 분류해서 진행한다.

```text
Core experiment:
논문의 핵심 주장 1~3을 직접 강화하는 실험

Support experiment:
핵심 주장의 원인 해석, 반례 검증, 음성 결과를 제공하는 실험

Appendix / future experiment:
좋지만 본 연구의 중심을 흐릴 수 있는 확장 실험
```

**금지할 것은 실험 확장이 아니라, 주장과 연결되지 않는 실험 나열이다.**

---

## 1. 현재 연구 정체성

현재 연구의 가장 설득력 있는 정체성은 다음이다.

> 대규모 극지 다중모달 데이터와 관측자료를 이용해 ALT 2D 지도와 얕은 3D 지중 열구조를 만들고, 공간 누설을 통제한 평가, 적용 가능 영역, 보정된 불확실성을 함께 제시하는 관측기반 GeoAI 프레임워크.

이 프레이밍은 단순한 “딥러닝으로 ALT RMSE를 낮췄다”보다 강하다. 실제로 지금까지 결과는 다음 방향을 가리킨다.

- 정적 ALT mapping에서는 GBM과 DL이 거의 동률이다.
- 위치-동등 가중과 셀 집계가 중요한 개선을 만들었다.
- 시간정합 ERA5 snapshot은 mapping에서는 게이트를 통과하지 못했다.
- T-lite 시계열 DL은 temporal holdout에서 붕괴했으므로 헤드라인이 아니다.
- AOA와 conformal UQ는 모델 정확도보다 실제 지도 사용성에 더 직접적이다.
- 얕은 3D 지중 열구조는 기존 2D ALT mapping과 구분되는 중요한 차별점이다.

---

## 2. 연구 차별성: 무엇을 내세울 것인가

## 2.1 차별성 1: 누설통제와 위치-동등 평가

초기 전지구 ALT baseline에서 무작위 CV가 공간 자기상관 때문에 성능을 크게 과대평가한다는 점을 확인했다. 이후 연구는 공간블록, LORO, 위치-동등 평가로 이동했다.

이것은 단순한 평가 방식 변경이 아니라 연구의 핵심 기여다.

나쁜 주장:

> 우리 모델은 RMSE가 낮다.

좋은 주장:

> 무작위 CV가 ALT mapping 성능을 과대평가한다는 문제를 보이고, 위치-동등 spatial validation으로 실제 미관측 지역 적용 성능을 평가했다.

## 2.2 차별성 2: 정보병목 진단

모델 토너먼트 결과는 “Diffusion이 최고”가 아니라 “정적 tabular ALT에서는 GBM과 최신 DL이 큰 차이를 보이지 않는다”는 결론을 준다.

핵심 메시지는 다음이다.

> 지금 병목은 모델 용량이 아니라, 공변량이 ALT를 설명하는 정보량이다.

따라서 앞으로 실험은 모델을 바꾸는 순서가 아니라 feature group을 분해하는 순서로 가야 한다.

```text
기후 only
지형 only
기후 + 지형
기후 + 지형 + InSAR
기후 + 지형 + PolSAR
기후 + 지형 + SoilGrids
기후 + 지형 + vegetation / landcover
기후 + 지형 + SAR + soil + vegetation
CCI prior 추가
위경도 대조군
```

각 실험은 반드시 다음을 함께 보고한다.

```text
RMSE
MAE
bias
R²
target SD
skill-over-mean
AOA inside/outside RMSE
90% coverage
interval width
```

## 2.3 차별성 3: AOA + conformal UQ

ALT 지도는 예측값 하나만으로는 부족하다.

```text
예측 ALT = 58 cm
```

보다 다음이 연구적으로 더 강하다.

```text
예측 ALT = 58 cm
90% prediction interval = 42~81 cm
AOA = inside
```

또는:

```text
예측 ALT = 58 cm
90% prediction interval = 20~135 cm
AOA = outside
```

두 번째 경우는 예측값이 있어도 “모델이 잘 모르는 환경”임을 의미한다. 이 정보가 있어야 지도 사용자가 예측값을 오해하지 않는다.

## 2.4 차별성 4: 2D ALT와 얕은 3D 지중 열구조의 연결

기존 ALT mapping은 주로 2D 지도에 머문다. 본 연구는 다음을 함께 내야 한다.

```text
ALT(x, y)
T(x, y, z = 0/1/2/5/10/20 m)
AOA(x, y)
UQ(x, y)
```

여기서 중요한 점은 ALT와 MAGT/0°C 연평균 등온면을 혼동하지 않는 것이다.

```text
ALT = 계절 최대 융해 깊이
MAGT = 연평균 지중온도
0°C 연평균 등온면 = 지중 열상태의 경계 후보
```

이 셋은 서로 연결될 수 있지만 같은 물리량이 아니다.

---

## 3. 예상 문제점과 조치

## 3.1 연구 질문이 흐려질 위험

실험 시간이 충분해도 연구 질문이 흐려지면 논문/보고서가 약해진다.

예상되는 흐림:

```text
ALT 2D
3D 지중온도
T-lite GRU
4D projection
AlphaEarth
SoilGrids
Sentinel-1
Sentinel-2
KPDC 검증
PyVista 시각화
대회 보고서
```

이 항목들은 각각 의미가 있지만, 같은 수준으로 나열하면 중심이 사라진다.

### 조치

모든 작업을 아래 위계로 분류한다.

```text
Primary claim:
ALT 2D + shallow 3D + AOA + calibrated UQ

Mechanism claim:
무엇이 ALT 예측을 지배하는가? 기후, 지형, SAR, soil, vegetation, 위치성?

Reliability claim:
어디서 믿을 수 있고, 어디서 외삽인가?

Extension claim:
T-lite, 4D, future projection, foundation embedding
```

Claude Code는 새 실험을 시작할 때마다 아래 문장을 로그에 남긴다.

```text
This experiment supports: Primary / Mechanism / Reliability / Extension
Main expected contribution:
Stop condition:
Output CSV:
Output figure:
```

## 3.2 모델 고도화가 자기목적화될 위험

현재 결과상 정적 tabular ALT 문제에서는 DL이 GBM을 확실히 이기지 못했다. 따라서 MLP, FT-Transformer, TabM, Diffusion, Flow를 계속 바꾸는 것은 효과가 작을 가능성이 높다.

### 조치

정적 tabular ALT의 주력 모델은 다음으로 고정한다.

```text
GBM / HistGradientBoosting
Quantile GBM
Conformalized quantile regression
```

DL은 아래 조건에서만 확장한다.

```text
1. Sentinel-1/2, SAR patch, DEM patch 등 고차원 입력을 실제로 먹일 때
2. GRU/TCN이 persistence와 GBM annual-summary를 site/temporal holdout에서 이길 때
3. 3D/4D operator-learning이 GBM 조건장을 site-wise CV에서 이길 때
```

이 조건을 못 만족하면 DL은 방법 비교 또는 부록으로 둔다.

## 3.3 feature ablation이 불완전할 위험

데이터를 많이 넣었는데 ablation이 없으면 “왜 좋아졌는지”를 설명할 수 없다. 반대로 어떤 데이터가 악화시키는지 숨기면 연구 신뢰도가 떨어진다.

### 조치

feature group ablation을 canonical experiment로 둔다.

권장 표:

| ID | Feature group | 목적 |
|---|---|---|
| M0 | mean / regional mean | 기준선 |
| M1 | climate only | 열 forcing 효과 |
| M2 | terrain only | 지형 효과 |
| M3 | climate + terrain | 최소 물리 baseline |
| M4 | +InSAR | 침하/동결융해 proxy |
| M5 | +PolSAR | local high-resolution 물리관측 |
| M6 | +SoilGrids | 토양 열물성 proxy |
| M7 | +vegetation/landcover | 표면 에너지/유기층 proxy |
| M8 | +CCI prior | 기존 product prior 효과 |
| M9 | all physical covariates | 통합 모델 |
| Mloc | lat/lon only | 위치 대리정보 대조군 |

`Mloc`는 반드시 포함한다. 위경도 대조군이 물리 공변량보다 높게 나오면, 현재 feature set이 아직 위치성/기후구배를 충분히 설명하지 못한다는 강한 증거가 된다.

## 3.4 KPDC 활용이 형식적으로 보일 위험

대회 관점에서는 KPDC 데이터가 단순 부록처럼 보이면 약하다. 다만 KPDC 지점 자료를 무리하게 전체 학습셋에 넣는 것도 좋지 않다.

### 조치

KPDC는 다음 역할로 전면 배치한다.

```text
KPDC/KOPRI 현장 관측 = 독립 검증 및 보정 축
해외 위성/재분석/격자 자료 = 공간 확장 축
```

권장 분석:

```text
1. Council 지중온도 실측 vs shallow thermal prediction
2. Council AWS vs ERA5-Land 편향
3. Cambridge Bay 대조구 vs 전이 예측
4. KPDC 관측이 AOA 내부인지 외부인지 사례 분석
5. KPDC 실측이 conformal 90% interval에 포함되는지 확인
```

금지:

```text
- Council 6주 관측을 연평균 지중온도처럼 해석
- Cambridge Bay 처리구를 자연상태 검증으로 사용
- KPDC 소수 지점을 넣고 전체 모델 재학습 후 성능 claim
- KPDC 데이터 한계를 숨김
```

## 3.5 과장된 주장 리스크

현재 문서 일부에는 과거 표현이 남아 있을 수 있다. 반드시 수정한다.

| 금지 표현 | 문제 | 대체 표현 |
|---|---|---|
| 17cm는 물리하한 | 후속 분석에서 apparent floor로 정정됨 | 현재 공변량 정보병목에서 나타난 apparent floor |
| 12.97cm SOTA 돌파 | 범위 축소 아티팩트 가능성이 큼 | 평탄툰드라 제한 범위에서 낮은 절대 RMSE |
| DL이 GBM보다 우수 | 토너먼트상 거의 동률 | 정적 tabular ALT에서는 GBM이 충분하며 DL 이득은 제한적 |
| Diffusion이 최종 우수모델 | native UQ도 보정 없이는 과신 가능 | conformal 보정된 UQ가 핵심 |
| T-lite 성공 | temporal holdout에서 미통과 | T-lite는 음성 결과이며 예측 응용 후보 |
| CCI prior 무익 | v4 기준이며 v5 미검증 | CCI v4 prior는 현재 feature set에서 중복 가능성 |

## 3.6 3D 결과가 예쁘지만 과학적으로 약해질 위험

3D 시각화는 발표 임팩트가 크다. 하지만 3D 그림이 과학적으로 검증되지 않으면 장식으로 보인다.

### 조치

3D는 반드시 아래 세 가지를 함께 낸다.

```text
1. depth-wise validation
2. borehole/site-wise validation
3. physical consistency check
```

3D 그림만 만들지 말고, 최소한 다음 표를 같이 만든다.

```text
z = 0/1/2/5/10/20 m 별 RMSE
site-wise RMSE
region-wise RMSE
bias by depth
uncertainty width by depth
AOA inside/outside by depth
```

물리 체크:

```text
- depth profile이 비현실적으로 출렁이지 않는가?
- 0°C contour를 ALT로 오해하지 않는가?
- 5~10m 온도가 표층 기후보다 과도하게 변하지 않는가?
- borehole 없는 지역에서 UQ가 부당하게 좁지 않은가?
```

---

## 4. 앞으로의 권장 실행 순서

실험 시간은 충분하다고 가정하므로, 단기 제출 제한이 아니라 연구 완성도 기준으로 순서를 둔다.

## P1. canonical dataset과 metric 고정

목적: 이후 모든 실험이 비교 가능하도록 기준선을 고정한다.

필수 산출:

```text
dl_dataset_cell.csv
location_equal_weight
standard metrics table
fold definition file
AOA split definition
```

모든 결과 CSV는 다음 컬럼을 포함한다.

```text
rmse_cm
mae_cm
bias_cm
r2
target_sd_cm
skill_over_mean
n
cv_type
fold
region
scope
feature_group
coverage_90
width_90
aoa_status
```

## P2. cell-level feature ablation 재실행

목적: 어떤 데이터가 ALT를 설명하는지 정직하게 분해한다.

필수 비교:

```text
climate
terrain
InSAR
PolSAR
SoilGrids
vegetation / landcover
CCI prior
lat/lon control
all features
```

필수 그림:

```text
ablation waterfall
feature group bar chart
region-wise performance heatmap
AOA inside/outside performance plot
```

## P3. AOA + conformal UQ 고도화

목적: 예측값의 신뢰 가능 영역을 지도화한다.

필수 산출:

```text
AOA mask map
DI vs RMSE plot
DI vs coverage plot
90% interval coverage table
uncertainty width map
KPDC point coverage check
```

주의:

AOA coverage가 단조롭지 않게 나올 수 있다. 이 경우 결과를 숨기지 말고, marginal coverage의 한계와 spatial calibration 문제로 설명한다.

## P4. KPDC 독립 검증

목적: KPDC가 연구의 실질 축임을 보여준다.

권장 분석:

```text
Council AWS vs ERA5-Land bias
Council shallow ground temperature vs predicted shallow thermal field
Cambridge Bay control plot vs transfer prediction
KPDC points on AOA/UQ map
```

KPDC 데이터가 제한적이면 학습보다 검증 중심으로 사용한다.

## P5. shallow 3D thermal field

목적: 연구의 3D 차별성을 구현한다.

추천 모델:

```text
GBM conditional field
input = climate + terrain + soil + vegetation + CCI shallow GT prior + depth encoding
output = annual ground temperature at depth
```

비교군:

```text
nearest
IDW
kriging if feasible
GBM without CCI prior
GBM with CCI prior
```

필수 검증:

```text
leave-one-borehole-out
leave-one-region-out
depth-wise RMSE
physical consistency check
uncertainty by depth
```

DL neural field는 GBM/IDW를 이길 때만 본문에 올린다. 그렇지 않으면 실패모드와 kill-switch로 제시한다.

## P6. DL 고도화는 조건부로 수행

딥러닝은 다음 경우에만 본문 후보로 둔다.

### Case A. EO/SAR patch model

```text
input: Sentinel-1/2 or SAR patch + tabular covariates
model: CNN / ViT-lite / tabular-image fusion
success: GBM tabular보다 spatial CV와 LORO에서 모두 개선
```

### Case B. T-lite forecasting

```text
input: lagged ALT + monthly ERA5 sequence
model: GRU / TCN
baseline: persistence + GBM annual-summary
success: temporal holdout에서 persistence와 GBM 모두 개선
```

### Case C. 3D/4D operator learning

```text
input: forcing field + static covariates + depth/time query
model: DeepONet / FNO / physics-guided surrogate
baseline: GBM conditional field
success: leave-one-site/region에서 GBM보다 개선 + calibrated UQ 가능
```

성공 조건을 못 만족하면 본문 모델이 아니라 appendix 또는 future work로 둔다.

---

## 5. 시각화 계획

## 5.1 시각화의 목적

시각화는 예쁘게 보이는 것이 아니라, 연구 주장을 빠르게 이해시키는 장치여야 한다.

시각화의 핵심 질문:

```text
1. 무작위 CV가 왜 위험한가?
2. 현재 병목이 왜 모델이 아니라 정보량인가?
3. 어떤 데이터가 ALT 예측에 실제로 도움 되는가?
4. 모델이 어디서 믿을 수 있고 어디서 외삽인가?
5. 예측 불확실성은 실제로 보정되었는가?
6. 얕은 3D 지중 열구조가 어떤 형태로 산출되는가?
```

## 5.2 필수 figure set

### Figure 1. 연구 개념도

내용:

```text
KPDC/KOPRI field observations
+ CALM/GTN-P/ABoVE
+ ERA5-Land
+ SAR/InSAR/PolSAR
+ DEM/Soil/Vegetation
→ ALT 2D + shallow 3D thermal + AOA + UQ
```

목적: 연구 전체 구조를 한 장에 설명한다.

### Figure 2. CV leakage / 평가 프로토콜

내용:

```text
random CV vs spatial block vs LORO
```

목적: 왜 random CV 수치를 headline으로 쓰면 안 되는지 보여준다.

### Figure 3. 정보병목 진단

내용:

```text
GBM vs DL tournament
lat/lon control vs physical covariates
apparent floor decomposition
```

목적: 모델보다 데이터 정보량이 병목임을 설명한다.

### Figure 4. feature ablation

내용:

```text
M0~M9 waterfall
region-wise heatmap
```

목적: 어떤 데이터가 어느 조건에서 도움이 되는지 보여준다.

### Figure 5. AOA + uncertainty

내용:

```text
ALT prediction map
AOA mask
90% interval width map
coverage calibration plot
```

목적: 예측값을 어디까지 믿을 수 있는지 보여준다.

### Figure 6. KPDC independent validation

내용:

```text
Council/Cambridge Bay points over prediction map
observed vs predicted
ERA5 vs AWS bias
conformal interval inclusion
```

목적: KPDC가 단순 인용이 아니라 검증 축임을 보여준다.

### Figure 7. shallow 3D thermal structure

내용:

```text
depth slices
borehole profiles
0°C contour / isosurface
uncertainty by depth
```

목적: 2D ALT mapping을 넘어 얕은 지중 열구조까지 연결한다.

## 5.3 디자인 지침

`DESIGN_BRIEF_MIDREPORT.md`의 핵심을 따른다.

```text
- 오프화이트 배경 통일
- 냉색 계열 유지
- ALT = oslo_r
- temperature = vik, 0°C 중심
- uncertainty = acton
- difference = broc
- slide당 핵심 그림 1개 원칙
- figure 내부 제목은 과장하지 않음
- 파일 경로는 슬라이드에 표시하지 않음
- 모든 지도에 단위, 컬러바, 스케일바, 관측점 표시
```

3D 개선:

```text
- PyVista off-screen high-resolution render
- 회색 Axes3D pane 제거
- 0°C contour 또는 isosurface 명확히 표시
- borehole 위치를 함께 표시
- depth scale을 왜곡 없이 주석 처리
- 3D 그림 옆에 depth-wise validation table을 함께 제시
```

---

## 6. Claude Code 작업 규칙

## 6.1 실험 시작 전 기록

모든 새 실험 스크립트 상단 또는 로그에 다음을 명시한다.

```text
Research claim supported:
Input data:
Target:
Baseline:
Success condition:
Stop condition:
Expected output CSV:
Expected output figure:
```

## 6.2 실험 후 기록

모든 실험 후 다음을 문서화한다.

```text
Result:
Did it pass the success condition?
What changed compared with baseline?
Does this support Primary, Mechanism, Reliability, or Extension claim?
Should this be main text, appendix, or future work?
```

## 6.3 과장 방지

결과를 쓸 때 다음 문장 구조를 사용한다.

나쁜 구조:

```text
Model X achieved SOTA performance.
```

좋은 구조:

```text
Under location-equal spatial validation, Model X changed RMSE from A to B cm, with R² = C and skill-over-mean = D. The improvement was / was not consistent under LORO. Therefore, this result supports / does not support the claim that this feature group improves transfer.
```

## 6.4 문서 동기화

결과가 업데이트되면 최소한 다음 문서를 동기화한다.

```text
docs/EXPERIMENT_LOG.md
docs/PLAN_FORWARD.md
SESSION_HANDOFF.md
docs/EXPERIMENTS.md
relevant gpt/handoff/*.md
```

과거 handoff에 남은 정정 전 표현은 주석으로 명시한다.

```text
This statement was superseded by later analysis.
```

---

## 7. 최종 권장 방향

연구를 좁힐 필요는 없다. 다만 중심축은 고정해야 한다.

최종 권장 방향:

```text
Main paper / main report:
ALT 2D + shallow 3D thermal structure + AOA + conformal UQ

Scientific contribution:
leakage-aware validation, location-equal target restructuring,
feature-group ablation, information-bottleneck diagnosis,
KPDC independent validation, calibrated uncertainty

DL contribution:
negative but important for static tabular ALT,
conditional and gated for EO/SAR patches, T-lite forecasting, and 3D/4D operator learning

Visualization contribution:
not decorative 3D, but interpretable ALT/UQ/AOA/thermal-volume products with validation
```

Claude Code의 한 줄 지시:

> 실험은 충분히 해도 된다. 단, 모든 실험은 **정보병목, 신뢰가능성, 얕은 3D 열구조** 중 하나를 강화해야 한다. 정확도 숫자 하나보다, 어디서 왜 맞고 어디서 믿으면 안 되는지를 보여주는 연구로 만든다.
