# 논문 구조 v5 (2026-10-10 초안)

재설정 문서 `docs/RESET_2026-10-10_KOPRI_FEEDBACK.md` 8절을 절·문단 단위로 편 것이다. 각 절에 들어갈 주장(검정 대상), 근거 표, 그림·표를 적는다. 첫 실행 결과(10-10 저녁)는 `docs/RESULTS_v5_2026-10-10.md` 에 있고, 아래 [판정] 은 그 결과로 적었다.

## 제목(후보)

- EN: Year-matched comparison of physics-based machine learning, physical models and existing products for active-layer thickness in label-rich permafrost regions
- KO: 라벨이 풍부한 영구동토 지역에서 연도 정합 시험지로 비교한 물리 기반 기계학습, 물리식, 기존 활동층 두께 제품

## 1. Introduction (4문단)

1. ALT 의 의미와 관측 자료(CALM, ABoVE, ALLENA), ALT 의 연간 변동(연도 고정값이 아님).
2. 기존 접근 세 갈래: 물리식(Stefan, Kudryavtsev), 통계·ML 지도(Aalto 2018, Ran 2022, Wei 2026, Liu 2024), 위성 모형 제품(CCI, Yi·Kimball). 보고 정확도는 시험지(검증 설계·시간 정의)가 달라 비교가 안 된다.
3. 두 가지 빈 곳: (i) 라벨·강제력의 연도 정합을 명시한 비교가 없다. 대부분 다년 기후값 입력과 다년 평균 라벨을 섞는다. (ii) 물리식·ML·결합·제품을 같은 셀·같은 해·같은 분할에서 비교한 연구가 없다.
4. 이 연구: 세 지역 셀-연도 시험지, 검증 사다리 5단, 방법 9종, 제품 6종. 질문 Q1–Q4.

## 2. Data (Fig 1, Table 1)

2.1 Study regions and labels: 지역 3, 셀-연도 정의, 계절 말 규칙, 출처·약관, 제외 규칙(2010 이전, 다른 지역). 수치: 알래스카 15,540·레나 2,436·캐나다 1,196 셀-연도.
2.2 Yearly forcing: ERA5-Land 2010–2024, 변수 11, 창 정의(전겨울, 수문년), 격자 규칙, 기후값 판.
2.3 Static and satellite inputs: 지형·토양·SAR·MODIS(기간·대체 규칙).
2.4 Existing ALT products: 6종의 기간·해상도·학습 자료·누설 등급, 연도 정합 추출.

## 3. Methods (Fig S1 도식)

3.1 Physical models: Stefan(P0 원천, P1 재보정, P1a 아핀), Kudryavtsev(Ku).
3.2 Machine learning and combinations: D0, F1, D1, R1(λ), S. 학습기 CatBoost(주), RF·MLP·TabM·FT-T(학습기 축).
3.3 Validation ladder: V-R, V-B, V-Y, V-BY, V-L 정의와 각 단이 답하는 질문.
3.4 Forcing versions: 연도 정합 대 기후값(같은 정의의 2011–2024 평균), 짝지은 비교.
3.5 Metrics and statistics: RMSE·편향·r·블록 등가중, 연간 편차 지표, 블록 부트스트랩 CI, 씨앗 앙상블.
3.6 Maps: 연도별 강제력 격자, 모델 적용, 구간, AOA.

## 4. Results

4.1 Effect of year-matched forcing (Fig 2): Q1. [판정] A1 은 레나에서 지지(물리식 −1.6~−2.1 cm, CI 가 0 을 넘지 않음), 알래스카에서는 반대 방향이나 미결정(+1.4 [−0.2, +3.0]), 캐나다 미결정. A2 는 구조적으로 참(기후값 강제력은 연간 편차 0)이며 연도 정합 상관은 레나 P0 0.51, 그 밖에는 0.2 안팎. A3 은 '지역에 따라 방향이 다르다' 로 쓴다. 근거 `runs/core_cb/deltas_forcing.csv`.
4.2 Method comparison across validation designs (Fig 3, Table 2): Q2. [판정] B1 지지(직접 ML 의 무작위 대 블록 차이 알래스카 5.4, 레나 4.2, 캐나다 13.1 cm). B2 는 레나(−3.0 [−6.1, −0.6])·캐나다(−1.7 [−3.3, −0.2]) 지지, 알래스카 미결정(−0.4 [−1.1, +0.5]). B3 은 수정: 연도 홀드아웃은 같은 셀의 다른 해가 학습에 들어가 ML 에 유리하다(알래스카 D0 −3.2). B4 지지(지역 홀드아웃 D0 알래스카 +5.1, 레나 +22.9; R1 은 물리식과 동급). 근거 `summary_metrics.csv`, `deltas_vs_ref.csv`.
4.3 Conditions of the gain (Fig 4): Q3. [판정] C1 지지(알래스카·캐나다는 따뜻한 3분위에서 이득 소멸, 레나는 따뜻한 3분위에서 최대 −4.8). C2 지지(기후만으로는 이득 없음, 지형·토양이 필요, MODIS +0.3~0.4 cm 추가). C3 은 '학습기 차이는 지역 차이보다 작고 캐나다에서만 신경망이 2–3 cm 낫다'. 근거 `conditions_tdd.csv`, `runs/featsets`, `runs/core_nn`, `runs/core_rf`.
4.4 Comparison with existing products (Fig 5): Q4. [판정] D1 부분 지지: 레나에서는 6종 모두보다 낮고(−4.6~−12.2), 알래스카·캐나다에서는 Wei 2026·Liu 2024 와 동급(CI 가 0 포함)이며 CCI·Yi·Kimball·Aalto 보다 낮다. D2 지지(CCI v5 +33~+49, Wei 레나 +21, 연도별 편향 변동). 누설 주의(Wei·Liu 는 CALM 학습). 근거 `products_eval.csv`, `products_bias_by_year.csv`.
4.5 Year-explicit maps (Fig 6·7): 연도별 지도, 다년 평균, 편차, 구간 폭, AOA. 연간 편차 재현 한계 명시.

## 5. Discussion (4문단)

1. 왜 연도 정합이 중요한가: 블록과 연도의 교락(알래스카 22/40 블록 단일 연도), 선행 연구의 시간 정의 관행.
2. 조건부 권고: 어떤 검증 목적·레짐·입력에서 R1 을 쓰고 어디서는 P1 로 충분한가.
3. 제품 비교의 해석: 학습 자료 누설, 해상도, 시간 정의 차이.
4. 한계: 세 지역, 2010–2024, 관측 월 혼재, 연간 편차 미재현, 격자 안 상세도.

## 6. Conclusions (1문단)

## Display items

| 번호 | 내용 | 근거 |
|---|---|---|
| Fig 1 | 자료와 시간축 | `labels_cell_year_v5.csv`, 등록부 |
| Fig 2 | 강제력 효과 | `deltas_forcing.csv` |
| Fig 3 | 검증 사다리 × 방법 | `summary_metrics.csv` |
| Fig 4 | 조건 | `conditions_tdd.csv`, 학습기·공변량 축 |
| Fig 5 | 제품 비교 | 제품 평가 표 |
| Fig 6 | 알래스카 연도 지도 | 격자 예측 |
| Fig 7 | 불확실성·레나 | 격자 예측 |
| Table 1 | 입력 등록부 요약 | `docs/INPUT_REGISTRY_v5.md` |
| Table 2 | 방법 × 검증 단 RMSE | `summary_metrics.csv` |
| SI | 방법 도식, 강제력 정의 표, 학습기 축 전체 표, 제품별 산점, 라벨 월 분포 | |

## 쓰지 않을 문장

- '라벨 n 개로 …' 식의 라벨 수 워크플로 주장. 라벨 0·3·10 결과는 인용하지 않는다.
- '다음 관측 위치' 처방.
- 연도를 밝히지 않은 'ALT 지도' 표현. 모든 지도는 연도·강제력을 적는다.
