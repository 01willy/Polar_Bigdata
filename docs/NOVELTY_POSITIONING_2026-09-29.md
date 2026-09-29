# 신규성·의의·선행 연구 대비 차별성 (검증본, 2026-09-29)

**작성 경위**: 문헌 재확인 4묶음(18편), 누락 선례 탐색(29편), 내부 근거 정리, 사용자 제공 PDF(`outputs/ALT 예측 선행연구 리뷰 및 그림 분석.pdf`) 대조, 신규성 문장 후보 9개에 대한 심사 3건(Sci Rep 심사자, 영구동토 전문가, ML 방법론 전문가)을 종합했다. 후보 9개 가운데 7개가 한정어와 함께 유지되고 2개는 폐기되었다.
**쓰임**: 원고의 서론 기여 문장과 관련 연구 절의 근거 문서다. 수치는 LG 통합 격자 결과가 나오면 교체한다.
**주의**: 확인 수준이 초록 이하인 문헌은 표에 표시되어 있다. 인용 전에 원문을 대조한다.

---

**확인 범위.** 심사 3건, 문헌 재확인 4묶음, 누락 선례 탐색, PDF 대조 결과를 종합했다. 이번 작업에서 읽기 전용으로 다시 대조한 자료는 `m1_sc_tests.csv`, `b1_protocol.csv`, `b2_curve.csv`, `fidelity_base_v3.csv`, LG 계획서(개정 5 포함)다. 학습, 실험, 파일 생성·수정, git 작업은 하지 않았다.

## 1. 결론

- 방법 구성 요소(물리 출력 입력, E의 ML 추정, 물리 유래 라벨, 잔차 학습, 계수 재보정)는 모두 선례가 있다. 기법의 신규성은 주장할 수 없다.
- 신규성은 두 곳에 있다. 첫째는 ALT에서 학습 제외 지역의 오차를 대상 라벨 수의 함수로, 두 Stefan 기준선 대비로 보고하는 평가 설계다. 둘째는 같은 절차로 얻은 음성 결과(계수 지도의 지역 간 실패, 라벨 0 직접 ML의 열세, 단일 계수 재보정의 비균일 효과)다. 조사 범위(2019–2026, 영문 위주)에서 이 조합의 ALT·MAGT 문헌은 확인되지 않았다.
- 이전 답변이 기여로 든 "성분 분해 72–95 %"와 "검증 방식에 따른 순위 역전"은 심사에서 기각되었다. 선행 연구가 아니라 연구 자체의 CSV와 맞지 않는다.
- 수준은 적용과 평가 중심의 점진적 기여다. 효과 크기가 작고(라벨 전량 4지역 평균 −0.68 cm, 러시아 W 제외 시 −0.10 cm), 독립 지역은 4개이며 그중 2개는 27–28셀이다.
- 현재 수치는 실험마다 원천 정의와 추출 규칙이 달라 같은 양의 부호까지 바뀐다. 문장은 LG 통합 격자 결과로 수치를 교체한 뒤 확정된다.

## 2. 사용자 인용문의 정확성

여섯 문헌의 분류는 모두 본문 기준으로 맞다. 고칠 것은 인용문이 아니라 이전 조사와 내부 문서다.

| 문헌 | 인용문의 분류 | 판정 | 보완 사항 | 확인 수준 |
|---|---|---|---|---|
| Pilyugina 2023 | 물리 출력을 입력 변수로 | 맞다 | 물리 모델은 Kudryavtsev다. 손실에 물리 항은 없다. 게재본으로 보이는 논문은 IEEE Access 13, 96423(2025) | 본문(게재본은 초록) |
| Wang G. 2025 | 물리 출력을 입력 변수로 | 맞다 | 블렌딩은 CatBoost와 Extra Trees 사이다. Stefan 대비 개선 수치는 타 문헌 값과의 비교다 | 본문 |
| Ran 2022 CEE | E를 ML로 추정 | 맞다 | 통계·ML 5종 앙상블이다. 기반시설 비용 평가의 중간 단계이고 ALT 오차는 보고하지 않았다. Ran 2022 ESSD와 별개 논문이다 | 본문 |
| Zhang C. 2024 | E를 ML로 추정 | 맞다 | 페어뱅크스 인근 실험지 2곳, SVM, 같은 횡단선 안 5-fold | 본문(요약 도구 경유, 인용 전 PDF 대조 필요) |
| Gautam 2025 | 두 방법을 따로 비교 | 맞다 | 무작위 70/30 1회, CI 없음. 결합은 향후 과제로만 권고했다 | 본문 |
| Gay 2026 | 물리 규칙으로 만든 라벨로 학습 | 맞다 | 물리 손실도 함께 쓴다. 대상은 zero-curtain이다. 96.4 %는 저자가 밝힌 자기 일관성 지표다 | 본문 |

인용문에는 세 범주가 빠져 있다.

- 물리 모의값 사전학습 후 관측 미세조정: Liu Y. 2023(초록 수준), Read 2019, Jia 2021.
- 물리 기준선 + 잔차: Tama 2025(WACV 2026 게재).
- 하이브리드 모델의 지역 홀드아웃 평가: Feng 2023.

## 3. 선행 연구 대비 차별성

| 문헌(확인 수준) | 물리 사용 방식 | 검증 방식 | 지역 홀드아웃 | 라벨 수 축 | 같은 라벨 재보정 기준선 | 본 연구와의 차이 |
|---|---|---|---|---|---|---|
| Pilyugina 2023(본문) | Kudryavtsev 출력을 CatBoost 입력으로 | 시간 분할 | 없음 | 없음 | 없음 | "Kudryavtsev 단독"은 물리 출력만 입력한 적합 모델이다 |
| Wang G. 2025(본문) | Stefan 출력을 입력 특징으로 | 무작위 10-fold | 없음 | 없음 | 없음 | 같은 자료의 Stefan 단독 오차가 없다 |
| Ran 2022 CEE(본문) | 공변량으로 E 회귀 후 E·√TDD | 지역 안 10-fold | 없음 | 없음 | 없음 | 본 연구는 같은 유형을 지역 제외 조건에서 시험했다 |
| Zhang C. 2024(본문, 요약 도구) | 원격탐사 특징으로 E 회귀 | 실험지 안 5-fold | 없음 | 없음 | 없음 | 국지 규모, 고해상도 입력 |
| Gautam 2025(본문) | 결합 없음, 병렬 비교 | 무작위 70/30 1회 | 없음 | 없음 | 없음(기준선 1개) | Stefan은 지점 역산 E의 크리깅이다 |
| Gay 2026(본문) | 규칙 라벨 + 물리 손실 + 물리 특징 | 지점 단위 분리(서술 엇갈림) | 없음 | 없음 | 없음 | 대상이 ALT가 아니다 |
| Karjalainen 2019(본문, 요약 도구) | 없음(GLM) | 500 km 거리 제외 | 없음 | 없음 | 없음 | ALT 공간 외삽 검증의 선례 |
| Read 2019, Jia 2021(본문) | 물리 모의값 사전학습 + 보존 손실 | 호수별 시간 홀드아웃 | 없음 | 있음 | 있음 | 호수 수온, 공간 전이 없음 |
| Willard 2021(본문) | 원천 모델 전이 + 메타모델 | 호수 단위 홀드아웃 | 지리 구역 아님 | 있음(1–50) | 없음 | 손익분기 비교 대상이 라벨 0 전이 모델이다 |
| Feng 2023(본문) | 신경망이 HBV 매개변수 예측 | 7개 지역 순환 제외 | 있음 | 없음 | 없음 | 유량, 대상 라벨 0 고정 |
| 본 연구 | 유사라벨, 앵커 + 잔차, 계수 재보정, 계수 회귀 | 지역 제외, 100 km 버퍼, 블록 절반 분할 | 있음(독립 4지역) | 있음(0–전량) | 있음 | 해당 없음 |

세 설계 요소를 함께 갖춘 ALT 문헌은 확인 범위에 없다. 요소별로는 모두 인접 분야에 선례가 있다.

## 4. 유지되는 신규성 문장

outcome이 keep_with_qualifier인 7개다. 심사 권고에 따른 배치는 서론 기여 문장(N1 + N2, N6 + N7, N5)과 보조 결과(N4, N9)다.

### N1 평가 축 (서론 기여 문장)

- **한국어**: 영구동토 ALT 또는 MAGT 매핑에서, 원천 학습 자료에서 제외한 지역의 예측 오차를 대상 지역 라벨 수의 함수로 보고한 연구는 조사 범위에서 확인되지 않았다. 본 연구는 대상 지역을 100 km 버퍼와 함께 원천에서 제외하고, 대상의 0.5° 블록 절반에서 고정된 규칙으로 라벨 n개를 뽑아 나머지 절반에서 채점한다. 각 n에서 평균과 중앙값, 오차가 커진 대상 수, 최대 증가를 보고한다. 지역 전체를 제외한 4지역과 상위 지역이 원천에 남는 하위 지역 10개는 따로 집계한다.
- **English**: To our knowledge, no study mapping permafrost active-layer thickness or mean annual ground temperature has reported prediction error for regions excluded from the source training data as a function of the number of labels drawn from the target region. Distance-blocked validation has been used for ALT (Aalto et al., 2018; Karjalainen et al., 2019; Ran et al., 2022), and label-count evaluations for unmonitored targets exist in other fields (Pool et al., 2019; Willard et al., 2021; Portes et al., 2026; O'Malley et al., 2026). We combine the two: the target region is excluded from the source data with a 100-km buffer, n labels are drawn under a fixed rule from half of its 0.5° blocks, and errors are scored on the other half. For each n we report the mean and median change relative to the source-coefficient Stefan model, the number of targets with increased error and the largest increase, separately for four regions excluded entirely (two with fewer than 30 labelled cells, n ≤ 10) and for ten sub-regional targets whose parent region remains in the source data.
- **근거 상태**: 설계는 확정이다. 수치를 채운 통합 곡선은 LG 실행 뒤 확정된다.
- **조건**:
  - "곡선"이라는 말은 LG 결과 뒤에만 쓴다. 그 전에는 출처 실험을 표기한 구간별 결과로 쓴다.
  - 라벨 단위를 명시한다. `fidelity_base_v3.csv`의 `spatial_support_m`은 전 행 100이다(이번에 확인). CALM 행에도 같은 값이므로 이 열이 실제 지지 규모인지 기본값인지는 원고 작성 전에 확인해야 한다.
  - "1 km 셀 단위 예측"이라는 표현은 자료와 맞는지 다시 정해야 한다.
- **남은 위험**: Ahajjam 2025와 Aalto 2018은 본문을 확인하지 못했다. O'Malley 2026의 관측 수별 표는 요약 도구 경유 확인이다.

### N2 두 기준선 (방법 절 서술, N1에 합침)

- **한국어**: 호수 수온 분야의 보고 형식을 따라 모든 방법을 두 Stefan 기준선과 비교한다. 하나는 원천 계수이고, 다른 하나는 같은 대상 라벨로 재보정한 계수(κ = 10 수축)다. 재보정 기준선이 원천 계수 기준선보다 나쁜 대상이 있으므로 두 기준선 대비 값을 모두 보고하고, 둘 중 오차가 낮은 쪽을 넘을 때만 학습기의 이득으로 쓴다.
- **English**: Following the reporting format used for lake temperature (Read et al., 2019; Jia et al., 2021), every method is compared with two Stefan baselines: the source-region coefficient and a coefficient recalibrated with the same target labels (shrinkage toward the source value, κ = 10; the unshrunk least-squares refit is reported as a sensitivity case). Because the recalibrated baseline is less accurate than the source-coefficient baseline in some targets, differences are reported against both, and a gain is attributed to the learner only when it holds against the better of the two.
- **근거 상태**: 형식은 확정이다. 재보정 기준선 대비 순가치의 4지역 층화 평균은 LG L4 뒤 확정된다.
- **조건**: 독립 기여로 세지 않는다. 현지 관측으로 E를 보정하는 것은 영구동토 분야의 관행이므로 그 출처를 함께 인용한다(Gautam 2025, Nelson 1997. Li C. 2022와 Peng 2018은 검색 요약 수준).

### N6 라벨 0 직접 ML과 증강 (서론 기여 문장)

- **한국어**: 원천이 알래스카 단독이고 대상 라벨이 없는 조건에서, 초모수를 고정한 학습기 10종은 제외 4지역의 등가중 평균으로 원천 계수 Stefan 식보다 오차가 크다(CatBoost +2.3 cm, 95 % CI 0.8–4.3). 입력에 √TDD와 CCI ALT가 들어 있어도 그렇다. 지역별로는 러시아 W에서만 두 가중 모두 유의하고 캐나다에서는 차이가 없다. 별도의 공변량만 조건 실험에서 Stefan 유사라벨은 직접 ML의 오차를 3.6 cm 줄여 물리식 수준으로 되돌리지만 그보다 낮추지는 못한다. 위약 유사라벨 대비 우위가 가중 규약에 강건하지 않으므로 이 감소를 유사라벨의 물리 내용에 귀속하지 않는다.
- **English**: With Alaska as the only source region and no target-region labels, ten learners with fixed hyperparameters had larger error than the source-coefficient Stefan model in the equally weighted mean of four excluded regions (gradient boosting +2.3 cm, 95% CI 0.8–4.3; exploratory contrast without multiplicity correction), although their inputs included the square root of thawing degree-days and a model-based ALT product. By region the difference held under both weighting conventions in Russia W only and was absent in Canada (+0.2 cm, −4.6 to 3.4). The direction agrees with random-split comparisons in Alaska (Gautam et al., 2025) and on the Tibetan Plateau (Shen et al., 2023). In a separate covariate-only experiment, Stefan-derived pseudo-labels for target cells reduced the error of direct learning by 3.6 cm (1.9–4.8), to but not below the Stefan error; because the advantage over placebo pseudo-labels was not robust to the weighting convention, we do not attribute the reduction to the physical content of the pseudo-labels.
- **근거 상태**: 확정(R01, R03, R04). 대조는 탐색 가족이라 Holm 보정이 없다.
- **이번에 확인한 점**: 4지역 평균에서는 10종 모두 셀 가중과 블록 등가중 CI가 0을 제외한다. 심사 두 건의 "CatBoost와 MLP만 확인"은 확인 범위의 차이였다.
- **조건**:
  - 직접 ML 수치와 증강 수치는 조건과 채점 셀 집합이 다르므로 문장을 나눈다.
  - 직접 ML은 "물리 앵커 없는 ML"로 정의한다.
  - 다지역 원천 조건은 LG L1, 용량 대조는 LGX L30 뒤 확정된다.

### N7 계수 지도와 라벨 0 우회 경로 (서론 기여 문장)

- **한국어**: 알래스카 한 지역에서 1 km 공변량으로 회귀한 E 계수는 그 지역 안 블록 검증에서는 설명력이 있으나(R² 0.39–0.52) 학습 제외 지역에서는 없다(R² −0.5에서 −3.1). 이를 앵커로 쓰면 제외 4지역 등가중 평균 RMSE가 단일 원천 계수 대비 3.0–6.7 cm 커진다. 지역별로는 러시아 W·E에서만 유의하다. 경험 매개변수가 지역을 넘어 옮겨지지 않는다는 기존 관찰을 ML로 회귀한 계수로 확장한 결과다.
- **English**: Edaphic-factor regressions on 1-km terrain, climate and soil covariates, fitted in a single source region (Alaska), had skill under block validation within that region (R² 0.39–0.52) but not in regions excluded from training (R² −0.5 to −3.1). Used as the anchor in four excluded regions they increased the equally weighted mean RMSE by 3.0–6.7 cm relative to a single source coefficient (Holm-adjusted p = 0.01); by region the increase was significant in Russia W and Russia E (27–28 cells each) and not in Lena or Canada. This extends to machine-learned coefficients earlier observations that empirically derived parameters do not transfer between regions (Oudin et al., 2008; Hasler et al., 2015; Garibaldi et al., 2026). Among the other zero-label routes, invariance learning increased the error, adding land surface temperature to the inputs left it unchanged (+0.03 cm, −0.12 to 0.12), a Stefan–Kudryavtsev anchor lowered cell-weighted but not block-weighted error, and product blending and coefficient borrowing gave intervals too wide to establish a benefit or its absence.
- **근거 상태**: 첫 문장은 확정(H10, 확인적 가족, 두 가중)이다. 9개 후보 가운데 통계 처리가 주장 수준에 가장 가깝다. 다지역 원천과 대상 라벨을 쓴 계수 회귀는 LG L7, LGX L13 뒤 확정된다.
- **조건**:
  - Ran 2022 CEE와 Zhang C. 2024의 재현이 아니라 같은 유형의 구현이다.
  - "관계가 지역마다 다르다"는 다지역 원천 결과 전에는 쓰지 않는다.
  - 지역 간 R² 음수는 수준 차이와 공간 패턴을 구분하지 못하므로 지역 평균을 뺀 R²나 순위상관을 병기한다.
  - Stefan–Kudryavtsev 앵커 행(셀 가중 −2.52 [−3.99, 0.39], −3.32 [−4.86, −0.05])을 빼면 선택 보고로 읽힌다.

### N5 단일 계수 재보정의 비균일 효과 (서론 기여 문장)

- **한국어**: 대상 라벨로 단일 Stefan 계수를 재보정하는 것(κ = 10 수축)은 모든 대상에서 이득이 아니다. 무작위 추출에서 캐나다는 n ≥ 10에서 두 가중 모두 악화한다(n = 10 +1.7 cm, n = 320 +3.5 cm). 같은 대상에서 원천 계수 앵커 + 잔차는 물리식보다 0.7–0.8 cm 낮게 유지된다. 어느 지역이 악화하는지는 추출 규칙에 따라 달라지며, 블록 분산 추출에서는 레나가 악화한다(n = 10 +3.7 cm). 시험한 풀링 추정기, 3라벨 진단, 선택 규칙은 이 설계에서 악화를 없애거나 미리 가려내지 못했다.
- **English**: Recalibrating a single Stefan coefficient with target-region labels (shrinkage, κ = 10) was not uniformly beneficial. Under random label draws the Canadian target, which combines four campaigns between 61°N and 69°N with coefficients from 1.06 to 2.16, deteriorated under both weightings at n ≥ 10 (+1.7 cm at n = 10 to +3.5 cm at n = 320), whereas a source-coefficient anchor with residual learning stayed 0.7–0.8 cm below the Stefan model; under block-spread draws the Lena region deteriorated instead (+3.7 cm at n = 10). None of the pooling estimators, three-label diagnostics or selection rules that we tested removed or anticipated the deterioration in this design.
- **근거 상태**: 캐나다 악화와 추정기·진단·선택 규칙의 실패는 확정이다. 원인 해석(캠페인별 계수 차이, A·B 블록 모집단 차이)은 탐색이다. 배치 축과 L7은 LG 실행 뒤 확정된다.
- **이번에 확인한 점**: "유의 개선 4, 유의 악화 4"는 B2 무작위 추출 n = 3의 셀 가중 CI 기준 값이다. 두 가중 기준으로 세면 다음과 같다.

| n | 개선 | 악화 |
|---|---|---|
| 3 | 2 (러시아 W, AL-3) | 4 (AL-2, AL-4, AL-5, AL-6) |
| 10 | 2 (러시아 W, AL-3) | 3 (캐나다, CA-3, AL-5) |

- 14대상 중 12대상에 유효 블록 5 미만 표시가 있다. 대상 수 집계는 가중 규약과 추출 규칙을 명시해 LG 값으로 다시 낸다.
- **조건**:
  - 배치 효과 문장은 뺀다(평균 행에 블록 등가중 CI가 없다).
  - n = 3의 +0.73 cm는 셀 가중 CI가 0을 포함하므로 시작점을 n = 10으로 쓴다.
  - "재보정은 통상 안전한 기준선"이라는 전제는 쓰지 않는다. E가 경관 단위마다 다르다는 것은 Nelson 1997 이래의 지식이다.
  - 가장 방어 가능한 대비는 Jia 2021의 단조 개선과의 차이다.

### N4 필요 라벨 수와 전량 효과 크기 (보조 결과)

- **한국어**: 대상 라벨을 원천 자료와 같은 가중(α = 1)으로 합치고 원천 계수 앵커에 CatBoost 잔차(λ 0.25)를 더한 조건에서, 잔차 학습은 레나와 하위 지역 5개에서 라벨 40개 이하로 물리식을 넘지 못했다. 레나는 셀 가중 기준 320개, 블록 등가중 기준 160개가 필요했고 모든 차이는 ±0.6 cm 안이다. 라벨 전량에서 4지역 등가중 평균은 −0.68 cm이고 러시아 W를 빼면 −0.10 cm다. 두 가중 모두 유의한 지역은 레나 하나다.
- **English**: With target labels pooled with the source data at equal weight (α = 1), a source-coefficient Stefan anchor and a gradient-boosted residual (λ = 0.25), residual learning did not reduce the error below that of the Stefan model with 40 or fewer labels in the Lena region or in five sub-regional targets; in Lena the cell-weighted interval excluded zero at n = 320 and the block-weighted interval at n = 160, and all differences were within ±0.6 cm. With all available labels the equally weighted mean over four regions was −0.68 cm (95% block-bootstrap CI −1.01 to −0.27; region-level test p = 0.35) and −0.10 cm without Russia W; the gain was significant under both weighting conventions in the Lena region only, and Russia E deteriorated under block weighting.
- **근거 상태**: 확정(F1, H13). −0.10 cm는 심사에서 지역별 점 추정으로 계산한 값이고 CI가 없다.
- **조건**:
  - 전반부는 α = 1 설계의 귀결일 수 있다. LG L3 결과에 따라 "대상 가중 없이는"으로 한정하거나 강화한다.
  - "한 지역만 유의"는 원천 정의에 의존한다. a2에서는 캐나다도 두 가중 유의다.
  - 효과 크기를 격자 안 변동과 병기한다. 재분석 격자 안 ALT 표준편차 10–17 cm는 심사 집계이므로 LGX N2의 하한 값으로 교체한다.

### N9 구간 커버리지 (보조 결과)

- **한국어**: 라벨 0 조건의 ALT 90 % 예측 구간에 대해 학습 제외 지역의 경험 커버리지를 보고한다. 이는 조사 범위(2026년 9월 기준)의 ALT 문헌에서 확인되지 않았다. 셀 풀링 보정은 0.73(평균 폭 56 cm), 지역을 집단으로 둔 근사 계층 conformal 절차는 0.86(81 cm)이다. 지역별로는 0.75(러시아 W)에서 0.93이고 알래스카를 제외 지역으로 두면 0.98이다.
- **English**: For label-free 90% prediction intervals of active-layer thickness we report empirical coverage in regions excluded from training, which to our knowledge (as of September 2026) has not been reported for ALT. Coverage was 0.73 for pooled-cell calibration (mean width 56 cm) and 0.86 (95% CI 0.80–0.92; 81 cm) for an approximate hierarchical conformal procedure with regions as groups (Dunn et al., 2023), ranging from 0.75 (Russia W, 28 cells) to 0.93 across four regions and 0.98 when Alaska was the excluded region. With four calibration groups no finite-sample guarantee holds at the 90% level, the exact procedure yields unbounded intervals, and interval width is proportional to the prediction (a factor of about two).
- **근거 상태**: 확정(F10 부분 지지). 라벨이 있는 조건의 구간은 LGX L25 뒤 확정된다.
- **조건**:
  - CQR 0.55는 다른 실험의 값이므로 같은 채점 셀로 다시 계산하기 전에는 같은 문장에 넣지 않는다.
  - 블록 등가중 커버리지(레나 0.81, 러시아 W 0.74)와 구간 점수(계층 132, 셀 풀링 153)를 병기한다.
  - 단순 기준 구간(Stefan ± 상수 분위, 심사 계산으로 0.78–0.80, 폭 65–69 cm)과 비교한다.

## 5. 폐기된 문장과 이유

| 문장 | 판정 | 이유 | 남길 수 있는 형태 |
|---|---|---|---|
| N3: 소수 라벨 이득의 72–95 %는 계수 재보정이고 잔차 순가치는 0과 구분되지 않는다 | 기각 2, 조건부 1 | 아래 참조 | 결과 절의 사례 서술 |
| N8: 검증 방식에 따라 ML과 물리식의 순위가 뒤집힌다 | 기각 2, 조건부 1 | 아래 참조 | 방법 절의 검증 방식 선택 근거 |
| N5의 배치 절(분산 대 집중의 부호 변화) | 삭제 | 평균 행에 블록 등가중 CI가 없다. n = 160 값은 대상 구성에 따라 부호가 바뀐다 | LG 배치 축이 두 가중에서 확인되면 별도 문장 |
| N2를 독립 기여로 세는 것 | 강등 | N1에 종속된 주장이고 형식은 선례가 있다 | 방법 절 |

**N3 기각 사유**

- 잔차 순가치의 CI가 0을 포함하는 대상은 러시아 W 하나다. AL-1, AL-3, 캐나다는 개선 쪽으로, 레나는 악화 쪽으로 0을 제외한다.
- 같은 분류의 CA-2는 재보정이 악화한다(+7.7 cm). 비율은 재보정이 통한 대상만 남긴 값이다.
- AL-1은 블록 등가중에서 재보정 이득이 없다.
- 분류가 채점 라벨을 쓴 사후 지표다.
- α = 1에서 잔차 성분은 대상 라벨에 반응하지 않으므로 결론이 설계의 귀결이다.
- 남길 형태: 러시아 W에서 재보정 −10.6 cm, 잔차 추가 −0.6 cm. 백분율은 본문에서 빼고 cm 값을 표로 싣는다.

**N8 기각 사유**

- 알래스카 블록 검증의 짝지음 대조에서 CatBoost − Stefan은 셀 가중 +1.66 [−1.32, 3.93], 블록 등가중 −1.87 [−3.15, −0.72]다. 자체 유의 규칙을 통과하지 못한다.
- 직접 ridge는 모든 방식에서 Stefan보다 낮으므로 "직접 ML이 뒤집힌다"도 틀렸다.
- seed 3개 점 추정이고 CI가 없다.
- 같은 패턴은 Gautam 2025에 이미 있다.
- 남길 형태: CatBoost 오차는 무작위 셀 분할 11.6 cm에서 공간 분리 방식 13.4–17.9 cm로 커지고 Stefan은 14.2–14.5 cm로 유지된다.

쓸 수 없는 표현은 다음과 같다.

| 표현 | 반례 문헌 |
|---|---|
| 최초의 물리·ML 결합, 최초의 E 추정 | 인용문 여섯 편 |
| 영구동토 ML의 지역 홀드아웃 최초 | Nitze 2021 |
| ALT의 공간 외삽 검증 최초 | Aalto 2018, Karjalainen 2019 |
| 미관측 지역에 현지 자료를 단계 투입하는 평가 최초 | Pool 2019, Willard 2021, Portes 2026 |
| ALT에서 ML이 Stefan보다 못함을 처음 보임 | Gautam 2025, Shen 2023 |
| 물리 유래 라벨 최초 | Gay 2026, Huang 2025, Li & Jia 2026 |
| 두 기준선 형식 최초 | Read 2019, Jia 2021, Tahvildari 2026 |
| 조건 없는 "ML이 물리식을 넘는다" | 자체 결과 |

## 6. 연구의 의의

**영구동토 과학.** 경험 계수가 지역을 넘어 옮겨지지 않는다는 정성적 지식(Hasler 2015, Garibaldi 2026)을 ALT 오차 단위로 수치화한다. ML로 회귀한 E 지도는 제외 지역에서 3.0–6.7 cm 악화하고, 단일 계수 재보정은 경관이 이질적인 캐나다에서 악화한다. 새로운 물리 사실은 아니다. 지역 간 계수 차이(알래스카 1.62 대 러시아 W 2.55)가 토양 물성, 라벨 정의(탐침, GPR, 지점 평균), ERA5-Land TDD 편향 가운데 어디서 오는지는 분리하지 못했다.

**ML 방법론.** 방법론의 신규성은 없고, 인접 분야의 알려진 결과(Steyerberg 2004, Oudin 2008, Ploton 2020, Staudinger 2025)를 ALT에서 재확인한 것이다. 다른 점은 세 가지다. 보정 대상이 물리 계수이고 원천 계수로 수축한다. 단순 보정도 해로운 대상이 있고 그 대상이 추출 규칙에 따라 바뀐다. 평균 이득과 실패 위험을 같은 표에서 보고한다. 기여는 "ALT 문헌에 없던 전이 평가와 음성 결과의 목록"으로 서술하는 것이 근거와 맞다.

**실용.** 새 지역에서 라벨 수에 따라 무엇을 쓸지의 근거를 준다. 라벨 0이면 원천 계수 Stefan 식과 경험 커버리지를 보고한 구간을 쓴다. 소수 라벨의 재보정은 이득일 수도 악화일 수도 있고 사전 진단 수단은 없다. 잔차 ML의 이득은 수백 개 이상에서 1 cm 미만이다. 전량 평균 0.68 cm는 물리식 RMSE의 2–4 %이고 GPR 유도 ALT의 평균 불확실성 0.14 m(Parsekian 2021)보다 작다.

**공통 한계.** 확인적 지역은 4개이고 지역 수준 검정 p는 0.35다. 러시아 W·E는 27–28셀이며 주요 평균의 유의성이 대부분 러시아 W에서 나온다. 블록 부트스트랩 CI는 지역을 고정한 조건부 구간이므로 새 지역으로의 일반화 근거가 아니다.

## 7. 신규성 강화에 필요한 것

### 7.1 LG 가설과 문장의 관계

| 가설 | 지지될 때 | 기각될 때 |
|---|---|---|
| L1 직접 ML, n ≤ 40 | N6이 다지역 원천으로 확장된다. N1의 직접 ML 행 확정 | N6을 "알래스카 단독 원천 한정"으로 쓴다 |
| L2 결합 | 증강 결합의 n > 0 기여를 새 문장으로 쓸 수 있다 | "증강은 라벨 0 전용" 확정 |
| L3 대상 가중 α | N4 전반부를 "대상 가중 없이는"으로 한정하고 가중의 효과를 양성 결과로 쓴다 | N4 전반부가 "가중을 주어도"로 강화된다 |
| L4 순가치(R1 − P1) | n ≤ 40에서 성립하면 "희소 라벨에서도 ML 순가치 있음". N3은 완전 폐기 | N3을 비율 없이 복원할 수 있다("재보정 Stefan 대비 추가 이득은 확인되지 않았다") |
| L5 학습기 | 결론이 CatBoost 한정이 아니다. 동등 판정은 LGX L26의 동등성 한계로 한다 | 우세 학습기를 표로 보고 |
| L6 알래스카 이전 | 알래스카 우선 검증 절차를 실용 문장으로 쓴다 | 지역 간 이전 불가를 음성 결과로 쓴다 |
| L7 공변량 의존 계수 | N5에 악화 해소 조건 추가, N7 둘째 문장 수정 | N7이 다지역 원천 조건까지 강화된다 |
| L8 전량 | N4 후반부가 통합 규약으로 재확인된다 | 전량 이득을 "M1 설정 한정"으로 쓴다 |

L4에서 n ≥ 40은 레나·캐나다 2지역 평균이므로 "부분(지역 2/4)"으로 표기된다. L7은 사전 점검이 분할 seed 1·2를 미리 보았으므로 seed 3–5 값을 병기한다.

### 7.2 LGX(개정 5) 가운데 문장에 직접 걸리는 항목

| LGX 항목 | 대응하는 위험 | 걸린 문장 |
|---|---|---|
| N4 분할 50회, 지역 수준 추론, 지역 하나 제외 평균 | 결론이 28셀 지역 하나와 분할에 의존 | N4, N5, N6 |
| X9 κ 민감도, 탐침만, 6–7월 제외 | 기준선 정의의 선택 편향, 라벨 정의 혼합 | N2, N5, N6 |
| N3 학습기 용량(L30) | 저용량 ML 기준선이라는 지적 | N6 |
| X3c 근접 단(L23) | 라벨 수 효과와 근접 효과의 미분리 | N1 |
| X1 물리 입력, 계수 회귀, 곱셈 보정(L10–L13) | "트리 모델이 잔차를 암묵적으로 배운다", 덧셈 대 곱셈 | N6, N7, N3 복원 |
| X2 위약의 n > 0, 비율(L15, L16) | 순환 논리, 증강 이득의 출처 | N6 |
| X3a 검증 사다리(L19–L21) | 무작위 분할의 낙관 편향 | N8 복원 |
| N1 물리 기준선 사다리(L29) | 단일 계수 Stefan이 약한 기준선이라는 지적 | N2, N4 |
| X5 셀 단위 저장(L24, L25) | 라벨 있는 조건의 구간, 거리별 성능 | N9 |

심사 문서의 "미실행, LG 범위 밖" 표기 가운데 위 항목은 "LGX 사전 등록, 미실행"으로 고쳐야 한다.

### 7.3 추가 자료

- **독립 지역 확충**(티베트, 러시아 W GTN-P 등): 세 심사가 공통으로 든 최대 위험(독립 지역 4개)은 계산 실험으로 풀리지 않는다. 라벨이 지온 유도 값이면 몽골과 같은 교락이 생기므로 직접 탐침 라벨인지 먼저 확인해야 한다.
- **기존 제품 격자**: Liu Z. 2024와 Wei 2026은 Zenodo 공개, Ran 2022 ESSD는 TPDC 계정이 필요하다.
- **문헌 원문**: Ahajjam 2025, Aalto 2018, Li C. 2022, Liu Y. 2023, Peng 2018은 출판사 차단으로 확인하지 못했다. 기관 구독으로 PDF를 확보하면 대조할 수 있다.

### 7.4 실행 선택지에 대한 함의

- 계획서 6.2절과 커밋 19f3f93 기준으로, 본 실행은 사전 등록 범위 전부(학습기 7종, RealMLP 포함)로 이미 제출되었다(작업 ZovWo). 이전 선택지로는 A와 B를 합친 범위다. 계획서 추정은 15–16 h, 약 88달러이고 상한 기준 최악은 125.31달러다.
- LGX는 계획서에 등록되었으나 하네스(h42, h41)와 보조 입력 표가 아직 없다. 계획서 추정은 약 110달러, 상한 기준 최악 약 172달러다.
- git 상태 기준으로 개정 5는 아직 커밋되지 않았다(`docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 수정 상태, 270행 추가). 계획서 6A.7은 이 커밋이 본 실행 결과 회수보다 앞서야 한다고 적고 있다. `results/`에는 사전 점검 결과만 있고 본 실행 결과는 회수되지 않았다.

LGX에 없는 새 선택지는 다음과 같다. 가치 순이다.

| 순위 | 선택지 | 내용 | 대응하는 문장 | 필요한 것 |
|---|---|---|---|---|
| 1 | 독립 지역 확충의 투고 전 실행 | 계획서는 투고 후 후보로 두었다. 완전성을 우선한다면 앞당긴다 | N1, N4, N6, N7 | 자료 취득 |
| 2 | 채점·라벨 단위 통일 | 라벨을 1 km 셀 또는 관측 지점 단위로 집계한 보조 격자 | N1(n 축의 지역 간 비교 가능성) | 채점 재집계는 X5 저장값으로 가능. 추출 단위 변경은 재적합 필요 |
| 3 | 20 cm 미만 ALT 값 제외 민감도 | 알래스카 138행, 레나 89행(심사 집계) | N5, N6 | X9 표지 표에 표지 추가 |
| 4 | 원천·대상 TDD 범위 중첩 표 | 범위 밖 셀 비율별 오차 | N6(단일 원천 외삽 반론) | 집계만 |
| 5 | 기존 제품 비교와 CALM 추세 재현 | LGX가 제외한 X7 | 의의(실용) | 격자 자료, 과거 ERA5-Land |
| 6 | 문헌 원문 확보 | 7.3절 목록 | N1의 선행 부재 서술 | PDF |

결정이 필요한 점은 선택지 2–4를 LGX에 넣을지 여부다. 넣는다면 본 실행 결과 열람 전에 계획서 개정으로 기록해야 한다.

## 8. 새로 찾은 위협 문헌과 대응

직접 겹치는 문헌(high)은 찾지 못했다. 아래는 주장 범위를 제한하는 문헌이다.

| 문헌 | 확인 수준 | 겹치는 요소 | 대응 |
|---|---|---|---|
| Ahajjam 2025, JGR MLC | 초록 | 환북극 CALM 115지점 ALT ML. 검증 방식 미확인 | 투고 전 본문 확인. 지점·지역 홀드아웃이면 N1 수정 |
| O'Malley 2026, arXiv | 초록(표는 요약 도구) | 지중 온도, 학습 제외 지역 × 현지 관측 1–40개 | N1 범위를 "영구동토 ALT 또는 MAGT"로 좁히고 선례로 인용 |
| Aalto 2018, GRL | 초록 | ALT·MAGT 거리 블록 검증 | N1에 인용. 조사 범위를 2019년 이전으로 확장 |
| Karjalainen 2019, Sci Data | 본문(요약 도구) | ALT 500 km 거리 제외 검증 | N1에 인용 |
| Wei 2026, ESSD Discussions | 초록 | 1 km ALT, leave-one-site-out | 경쟁 제품으로 인용. 심사에서 커버리지 요구를 받았으므로 N9에 확인 시점 표기 |
| Portes 2026, Spatial Statistics | 초록 | 미관측 지역에 현지 자료 단계 투입 | 설계의 일반적 신규성 포기, 선례로 인용 |
| Pool 2019, WRR | 초록 | 유역 leave-one-out + 관측 3–24개 | 같음 |
| Nitze 2021, Remote Sensing | 검색 요약 | 영구동토 6지역 교차검증(RTS 분할) | "ALT 또는 MAGT 매핑"으로 한정 |
| 암석 빙하 기반 분포 분류(Shaluli) | 검색 요약, 저자 미확인 | 타 산맥 검증 | 같음. 서지 확인 전 인용 보류 |
| Shen 2023, STOTEN | 검색 요약 | ALT에서 Stefan 우위 결론 | N6의 방향 선례로 인용. 권호 연도는 2023 |
| Pilyugina 2023 | 본문 | 물리 출력 입력, ALT | 분류 정정(10절) |
| Huang 2025, EGUsphere | 본문 | TTOP 출력을 학습 목표로 | 잔차 선례가 아니라 물리 출력 라벨 선례로 인용 |
| Herrington 2026 | 초록 | 재분석 토양 온도 + ML 보정, 단순 보정 기준선 | N2의 인접 선례 |
| Oudin 2008, WRR | 검색 요약 | 회귀 기반 매개변수 지역화의 실패 | N7에 인용 |
| Steyerberg 2004 | 검색 요약 | 소표본에서 재보정이 재추정보다 나음 | N2, N5의 일반 선례 |
| Hasler 2015, The Cryosphere | 요약 도구 | 경험 offset의 지역 간 비전이 | N7에 인용, 원문 대조 필요 |
| Adjei 2026, Kakhani 2024 | 초록, 검색 요약 | 환경 매핑의 conformal 커버리지 | N9의 인접 선례 |

Memiş 2025 리뷰의 "Stefan 정칙화 RF-LSTM, RMSE 약 15 % 감소" 문장은 1차 근거가 확인되지 않으므로 인용하지 않는다.

## 9. 사용자 제공 PDF 대조

### 9.1 PDF가 맞았고 이전 답변이 틀렸던 점

- Ran 2022 CEE를 포함했다. 이전 조사 25편에는 빠져 있었다.
- Pilyugina 2023을 물리 출력 입력으로 분류했다.
- Huang 2025를 TTOP 출력을 학습 목표로 쓴 구조로 분류했다.
- Karjalainen 2019의 500 km 거리 제외 검증과 Du Q. 2026을 포함했다.
- 신규성을 "물리가 들어가는 방식과 평가 설계"로 좁혀야 한다는 판단은 세 심사의 결론과 같다.

### 9.2 PDF의 서술 가운데 고쳐야 할 점

| PDF의 서술 | 근거가 말하는 것 |
|---|---|
| 핵심 기여는 유사라벨 증강과 덧셈 잔차 | 두 기법의 순가치는 작거나 조건부다. 증강은 물리식 수준까지만 되돌리고, 잔차 순가치는 n ≤ 40에서 ±1 cm 안이며 부호가 대상마다 다르다 |
| 덧셈 잔차가 곱셈 보정보다 낫다는 전제 | 큰 이득이 확인된 성분은 스칼라 재보정(곱셈)이다. 짝 비교는 LGX L12 |
| Pilyugina의 "Kudryavtsev 단독 32.4 cm" | 물리 출력만 입력한 CatBoost다 |
| Wei & Chen 2026은 연결 논문 없음 | ESSD Discussions 프리프린트가 있다 |
| Li X. 2022가 E 지도 공개 | 첫 저자는 Chuanhua Li다. E 지도 공개 여부와 산정 방식은 미확인 |
| AOA 마스크가 신뢰성 논의와 이어진다 | 비가중 AOA는 해로운 대상을 식별하지 못했다. 표기용으로만 쓴다 |
| 픽셀별 예측 구간 | 경험 커버리지를 보고한 구간이다. 보장은 없다 |
| 캐나다·시베리아를 희소 시나리오로 | 캐나다는 재보정이 해로운 사례이고, 러시아는 라벨 상한이 15셀 안팎이다 |
| 물리항이 기후 민감도를 맡는다 | 시험한 적이 없다. 지점 내 √DDT 계수(1.05 cm, Streletskiy 2026)는 공간 계수 E보다 작다 |
| SAR를 예측 변수로 | 전이 실험 입력에 SAR가 없다 |
| 라벨 희소성 곡선은 이 논문만의 그림 | 호수 수온과 수문 분야에 선례가 있다. "ALT의 지역 전이에서"로 한정 |
| 기준 모델 여섯 종에서 "ML 단독"과 "물리 입력" 구분 | 입력에 √TDD와 CCI ALT가 이미 들어 있다. 본 연구의 직접 ML은 "물리 앵커 없는 ML"이다 |
| Gay 2026은 지리 분리 지점 CV | 분할 서술이 세 가지로 엇갈린다 |
| Tama 2025는 arXiv 프리프린트 | WACV 2026 게재 논문이다. 초록의 정확도는 앵커와의 일치도다 |

### 9.3 PDF에서 채택할 시각

| 가치 | 시각 | 계획 반영 상태 | 뒷받침하는 문장 |
|---|---|---|---|
| 높음 | 물리 입력과 E 예측 방식을 같은 검증에서 재현 | LGX X1 | N6, N7 |
| 높음 | 덧셈 잔차 대 곱셈 보정 짝 비교 | LGX X1 | N3 복원, N2 |
| 높음 | 검증 사다리 | LGX X3a, X3b | N8 복원, N1의 동기 |
| 높음 | 라벨 이질성 민감도 | LGX X9. 캠페인 단위 분할은 제외, 20 cm 미만 제외와 단위 통일은 없음 | N5, N6 |
| 높음 | 셀 단위 예측 저장과 지표 확장 | LGX X5 | N9, 전 문장의 효과 크기 |
| 중간 | 유사라벨 비율 곡선과 n > 0 위약 | LGX X2 | N6 |
| 중간 | 라벨과의 거리별 성능 | LGX X5, X3c | N1 |
| 중간 | 기존 제품 비교 | LGX 제외. CCI만 N1 사다리에서 비교 | 의의(실용) |
| 중간 | 장기 CALM 추세 재현 | LGX 제외 | 보충 분석 |
| 낮음 | 잔차 모델 SHAP | LGX X5 서술 산출 | 보충 자료 |

## 10. 이전 답변의 정정 사항

**지정된 세 항목**

1. **Pilyugina 2023**: 이전 답변의 "열방정식 손실 정칙화"는 틀렸다. 초록의 'regularize' 표현을 따른 오분류다. 본문에는 물리 항을 넣은 손실이 없고, Kudryavtsev 출력 4쌍을 CatBoost 입력 특징으로 넣는다. "대상이 위험 지표이고 ALT가 아니다"도 틀렸다. ALT(cm)와 MAGT(°C)를 직접 예측한다.
2. **Gay 2026**: 이전 답변의 "물리 손실만 확인"은 불완전했다. 본문은 규칙 기반 라벨(345,033건)을 학습 목표로 쓰고, 물리 손실, 물리 특징, 도메인 전이학습을 함께 쓴다. 사용자 인용문의 분류가 사실에 부합한다.
3. **Ran 2022 CEE**: 이전 답변의 "조사에 없음"은 조사 누락이다. 논문은 실재하고(Commun. Earth Environ. 3, 238), Methods에 E를 5종 앙상블로 추정했다고 적혀 있다. 이전 조사는 Ran 2022 ESSD만 포함했다.

**그 밖의 정정**

- "성분 분해 72–95 %"를 주장할 수 있는 항목으로 든 것은 철회한다(5절).
- "평가 축" 문장의 표현을 고친다. n > 0이면 대상 라벨이 학습에 들어가므로 "지역 홀드아웃"을 "원천 학습 자료에서 제외한 지역"으로 쓰고, 독립 4지역과 하위 지역 10개를 분리한다.
- "두 기준선 형식은 Jia 2021에 선례"에는 Read 2019와 Tahvildari 2026을 함께 적는다.
- Willard 2021에는 라벨 수 축이 있다(대상 관측 1–50, 12수준).
- Huang 2025의 "물리 앵커 + 잔차" 분류는 미공개 개정 계획에 근거한 것이다.
- Tahvildari 2026은 실재한다(Sci Rep 16, 8925, 공극압). 영구동토 문헌은 아니다.
- Streletskiy 2026의 156지점은 원문에서 확인되었다.
- LG 계획서 3절의 D0 정의 "물리 없음"은 "물리 앵커 없음"으로 고쳐야 한다.

**내부 문서 정정 대상**(이번에 수정하지 않았다)

- `/home/willy010313/Polar_Bigdata/references/INDEX.md` 307–311행
- `/home/willy010313/Polar_Bigdata/docs/EXPERIMENT_PLAN_2026-07-21.md` 91행
- `/home/willy010313/Polar_Bigdata/docs/CONTEST_PLAN_2026.md` 126행
- `/home/willy010313/Polar_Bigdata/docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md` 52행, §1 주장 상태표
- `/home/willy010313/Polar_Bigdata/docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 3절 D0 정의

## 확인 수준과 한계

- 선행 부재 서술은 조사 범위 안의 판단이며 부재의 증명이 아니다. 중국어·러시아어 문헌은 거의 포함되지 않았다.
- 요약 도구를 거친 수치(Karjalainen 2019, Zhang C. 2024, O'Malley 2026, Hasler 2015, Du J. 2026, Li & Jia 2026)는 원고 인용 전에 원문 대조가 필요하다.
- 심사에서 계산한 값(러시아 W 제외 평균 −0.10 cm, 격자 안 ALT 표준편차 10–17 cm, 단순 기준 구간 커버리지)은 공식 산출물이 아니다.

**관련 파일**

- `/home/willy010313/Polar_Bigdata/docs/EXPERIMENT_PLAN_LG_2026-09-29.md`
- `/home/willy010313/Polar_Bigdata/docs/RESEARCH_FRAME_2026-09-29.md`
- `/home/willy010313/Polar_Bigdata/data/processed/m1/m1_sc_tests.csv`
- `/home/willy010313/Polar_Bigdata/data/processed/h4/b1_protocol.csv`
- `/home/willy010313/Polar_Bigdata/data/processed/h4/b2_curve.csv`
- `/home/willy010313/Polar_Bigdata/data/processed/fidelity_base_v3.csv`
- `/home/willy010313/Polar_Bigdata/outputs/ALT 예측 선행연구 리뷰 및 그림 분석.pdf`

**환경 안내.** claude.ai 커넥터 Gmail, Google Calendar, Google Drive는 인증이 필요해 이 세션에서 쓸 수 없다. 사용하려면 사용자가 claude.ai 커넥터 설정에서 직접 승인해야 한다. 이번 작업에는 필요하지 않았다.