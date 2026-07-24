# ALT 연구 개발 로드맵 및 Claude Code 인계 문서

## 0. 문서 목적

이 문서는 다음 연구를 실제 데이터·코드·실험 단위로 구현하기 위한 작업 명세다.

> **희소한 실제 ALT 관측을 보완하기 위해 실제 ALT, 지중온도장 유도 ALT, 물리경험식·수치모델 유도 ALT를 결합하되, 이들을 동일한 정답으로 취급하지 않고 자료원별 편향·불확실성·공간지지 규모를 반영하여 새로운 지역과 미래 시점의 ALT를 예측한다. 가능하면 계절별 해빙 깊이 \(D(t)\)와 얕은 지중온도장 \(T(z,t)\)까지 공동 예측한다.**

이 문서는 단순 아이디어 메모가 아니라 다음 용도로 사용한다.

- 연구 질문과 차별성 고정
- 데이터 구축 기준 정의
- 정보 누출을 막는 평가 프로토콜 정의
- 기준모델과 제안모델 구현 순서 결정
- Claude Code가 저장소를 생성하고 실험을 자동화할 수 있도록 작업 단위 제공
- 논문별 산출물과 중단·진행 기준 정의

---

# 1. 연구의 핵심 판단

## 1.1 연구 방향은 타당하다

ALT 직접 관측은 공간적·시간적으로 희소하다. 반면 다음 자료는 상대적으로 더 많이 확보할 수 있다.

- 현장 탐침·동결관·시추 기반 실제 ALT
- 보어홀·온도센서에서 관측한 \(T(z,t)\)
- 관측 온도장에서 계산한 temperature-derived ALT
- Stefan, modified Stefan, TTOP, Kudryavtsev 등의 경험·반경험 모델 ALT
- GIPL2, CryoGrid, 1차원 엔탈피 모델 등의 수치모델 ALT
- SAR, GPR, 기존 ML 지도 등에서 도출한 간접 ALT
- 재분석, 위성, 토양, 지형, 적설, 식생, 교란 자료

이들을 활용하여 예측 성능을 높이는 방향은 충분히 논리적이다.

## 1.2 단순 데이터 병합은 연구의 핵심이 될 수 없다

다음 방식은 위험하다.

```text
실제 ALT + 온도 유도 ALT + Stefan ALT
→ 모두 동일한 label로 병합
→ RF/LSTM/Transformer 학습
→ 랜덤 K-fold에서 최고 성능 모델 선정
```

이 방식에는 세 문제가 있다.

1. 물리식 산출값은 새로운 독립 관측이 아니라 입력변수의 결정론적 변환일 수 있다.
2. 저충실도 label이 많아지면 모델이 실제 ALT가 아니라 물리식의 편향을 학습할 수 있다.
3. 랜덤 K-fold는 공간·시간 자기상관과 같은 지점 반복관측으로 인해 성능을 과대평가할 수 있다.

따라서 연구의 중심은 **데이터 개수 증가**가 아니라 다음으로 전환한다.

\[
\boxed{
\text{어떤 자료원을}
\quad
\text{어떤 환경에서}
\quad
\text{얼마나 신뢰할 것인가}
}
\]

---

# 2. 선행연구와의 관계 및 차별성 포지셔닝

## 2.1 이미 선행연구가 있는 요소

다음 요소 자체는 신규성이 약하다.

- Stefan 식을 이용한 ALT 보정·확장
- 간접 추정 ALT를 대량 label로 사용한 공간 확장
- 물리모델 출력으로 신경망 사전학습 후 실측자료 미세조정
- 물리모델 출력을 ML 입력특징으로 사용
- Stefan과 ML 모델의 blending
- PINN을 이용한 희소 지중온도장의 보간·복원
- LSTM, XGBoost, LightGBM 등의 지중온도 예측
- 다년 ALT 예측
- 물리모델 잔차를 ML로 보정
- 다중 충실도 학습
- Mixture-of-Experts 자체

따라서 다음과 같은 주장은 피한다.

- “최초의 physics-informed ALT 예측”
- “최초의 Stefan–ML 하이브리드”
- “최초의 물리 기반 ALT 데이터 증강”
- “최초의 PINN 지중온도 증강”
- “잔차학습이라는 새로운 방법 제안”
- “Mixture-of-Experts를 ALT에 최초 적용”이라는 검증되지 않은 주장

## 2.2 2026년 PINN 지중온도 연구와의 차이

기존 PINN 기반 연구의 대표적인 흐름은 다음과 같다.

\[
\text{희소 } T_{\mathrm{obs}}(z,t)
\rightarrow
\text{PINN으로 연속 } T(z,t)\text{ 복원}
\rightarrow
\text{LSTM/XGBoost/LightGBM으로 미래 }T(z,t)\text{ 예측}
\]

현재 연구의 주된 흐름은 다음과 다르다.

\[
\begin{aligned}
&ALT_{\mathrm{obs}},\\
&ALT_T=\mathcal{F}[T(z,t)],\\
&ALT_{\mathrm{Stefan}},ALT_{\mathrm{TTOP}},ALT_{\mathrm{Kudryavtsev}},\ldots
\end{aligned}
\]

를 서로 다른 신뢰도의 관측으로 사용하여 잠재적 실제 ALT를 학습하고, 선택적으로 다음을 공동 예측한다.

\[
T(z,t), \quad D(t), \quad ALT=\max_tD(t)
\]

즉, 공통점은 물리 기반 저충실도 자료를 이용한다는 점이고, 차이는 다음과 같다.

- 최종 target이 지중온도만이 아니라 ALT 및 해빙 궤적임
- 여러 ALT 자료원을 동시에 사용함
- 자료원별 편향과 불확실성을 모델링함
- 최종 성능을 직접관측 ALT에서 평가함
- 공간·시간 외삽을 주된 검증문제로 둠

## 2.3 2025년 Scientific Reports Stefan–RF 연구와의 차이

해당 연구는 Stefan과 Random Forest를 하나의 하이브리드 모델로 결합한 것이 아니라 별도로 구축하여 비교한 연구에 가깝다.

간소화된 Stefan 모델은 다음 구조다.

\[
E_i = \frac{ALT_i^{obs}}{\sqrt{TI_i}}
\]

\[
\widehat{ALT}_{Stefan}(\mathbf{s},t)
=
E_{\mathrm{kriged}}(\mathbf{s})
\sqrt{TI(\mathbf{s},t)}
\]

RF는 기후·토양·지형·토지피복 등을 이용해 별도로 ALT를 예측한다.

현재 연구는 다음을 명확히 해야 한다.

- Stefan 계수 보정은 각 CV 학습 fold 내부에서만 수행
- 테스트 지점의 ALT는 물리식 보정·kriging·가중치 결정에 사용 금지
- Stefan과 ML의 단순 비교를 넘어서 잔차, 다중 충실도, 불확실성 모델을 평가
- 무작위 분할이 아닌 공간·시간 OOD 검증 수행

## 2.4 가장 방어력 있는 차별성

연구의 주된 차별성은 아래 조합에서 확보한다.

1. **Source-aware latent ALT**
   - 직접관측, 온도 유도, 물리식, 수치모델, 원격탐사 ALT를 서로 다른 자료원으로 취급
   - 자료원별 조건부 편향과 이분산 불확실성 추정

2. **Support-aware learning**
   - 점 탐침, CALM 격자 평균, 보어홀, GPR 선형자료, 위성 픽셀, 모델 격자의 공간지지 규모를 구분

3. **Censored/interval ALT 활용**
   - 탐침 한계 초과, rock hit, 경계 범위만 아는 자료를 삭제하지 않고 검열우도로 학습

4. **엄격한 OOD 평가**
   - 새로운 지점, 지역, 생태대, 측정방법, 미래 연도에 대한 평가
   - 모든 물리계수·pseudo-label 생성도 fold 내부에서 수행

5. **물리자료가 도움이 되는 조건과 해로운 조건의 규명**
   - pseudo-label 개수보다 편향, 분산, 환경적 거리, 물리모델 간 불일치가 성능에 미치는 영향 분석

6. **선택적 공동예측**
   - \(T(z,t)\), \(D(t)\), ALT를 물리적으로 일관되게 연결
   - 부분 해빙기 자료로 end-of-season ALT를 확률적으로 조기예측

---

# 3. 최종 연구 질문

## RQ1. 물리 기반 ALT pseudo-label은 실제로 일반화 성능을 높이는가?

\[
\Delta Skill
=
Skill(\text{실측+물리})
-
Skill(\text{실측만})
\]

랜덤 CV가 아니라 공간·시간 OOD에서 평가한다.

## RQ2. 단순 병합, 사전학습, 잔차학습, 다중 충실도 중 어떤 결합 방식이 가장 안정적인가?

비교대상:

- naive pooling
- physics feature
- pretrain–fine-tune
- residual correction
- source-aware multi-fidelity
- support-aware multi-fidelity

## RQ3. 물리자료가 도움이 되는 조건과 해로운 조건은 무엇인가?

분석변수:

- pseudo-label 비율
- 물리모델의 조건부 편향
- 예측분산
- 관측지점과의 환경적 거리
- 물리모델 간 불일치
- 생태대, 토성, 수분, 적설, 유기층, 교란상태
- 연속·불연속 영구동토 구분
- 공간해상도와 관측지지 규모

## RQ4. 온도장 유도 ALT는 어느 정도의 신뢰도를 가져야 하는가?

다음에 따른 오차를 정량화한다.

- 센서 수직 간격
- 시간해상도
- 최대 해빙시점 포함 여부
- 결측률
- 유효 동결점 가정
- zero-curtain
- 선형보간과 비선형보간
- PINN 또는 열모델 복원 여부

## RQ5. \(T(z,t)\), \(D(t)\), ALT 공동학습이 ALT 단독학습보다 유리한가?

다음 일관성을 평가한다.

\[
\widehat{D}(t)\approx \mathcal{F}[\widehat{T}(z,t)]
\]

\[
\widehat{ALT}\approx \max_t\widehat{D}(t)
\]

## RQ6. 제한된 현장조사 예산에서 어디를 추가 측정해야 하는가?

물리모델 간 불일치와 예측 불확실성을 이용한 active learning을 평가한다.

---

# 4. 연구 가설

## H1

단순 pseudo-label 병합은 랜덤 CV에서는 성능을 높일 수 있으나 공간 OOD에서는 효과가 감소하거나 악화될 수 있다.

## H2

물리모델 사전학습–실측 미세조정 또는 잔차학습은 naive pooling보다 안정적이다.

## H3

자료원별 편향과 이분산 불확실성을 학습하는 모델이 고정 증강비율 모델보다 실제 ALT 외부검증 성능과 calibration이 좋다.

## H4

온도장 유도 ALT는 Stefan ALT보다 평균적으로 높은 충실도를 가질 수 있으나, 센서 간격과 계절 커버리지를 무시하면 과신된다.

## H5

여러 물리식의 조건부 혼합은 단일 Stefan 식보다 이질적인 환경에서 평균편향을 줄인다.

## H6

\(T(z,t)\)–\(D(t)\)–ALT 공동학습은 ALT 단독학습보다 계절적 일관성과 미래시점 예측 안정성이 높다.

## H7

물리모델 간 불일치가 큰 지역을 우선 관측하는 active learning이 무작위 현장선정보다 효율적이다.

---

# 5. 연구 범위와 우선순위

## 5.1 필수 범위

1. 실제 ALT 자료 통합
2. 기후·지형·토양·적설·식생 입력 구축
3. Stefan 계열 저충실도 ALT 생성
4. 실측만 사용한 강한 tabular baseline 구축
5. naive pooling, physics feature, pretrain–fine-tune, residual 비교
6. source-aware multi-fidelity 모델 구현
7. 공간·시간 nested CV
8. 불확실성·calibration 평가
9. 결과 재현 파이프라인 구축

## 5.2 권장 확장

1. 온도장 유도 ALT 추가
2. 여러 물리식 비교
3. support-aware 관측연산자
4. censored ALT likelihood
5. \(D(t)\) 계절 해빙궤적 예측
6. end-of-season ALT 조기예측
7. active learning

## 5.3 후순위 또는 별도 논문

1. 완전한 \(T(z,t)\) 공동예측
2. PINN 또는 Neural Operator
3. 범북극 고해상도 지도 생성
4. 실시간 digital twin
5. 인과추론 기반 산불·적설·수문 교란효과
6. CMIP 미래시나리오 장기예측

---

# 6. 데이터 충실도 체계

## 6.1 Fidelity level

| 수준 | 자료 | 예시 |
|---|---|---|
| F4 | 직접 고충실도 | 탐침 격자, 동결관, 검증된 시추 |
| F3 | 온도관측 유도 | 충분한 \(T(z,t)\) 관측에서 산출한 ALT |
| F2 | 복원 온도장 유도 | 결측 온도장을 PINN·열모델로 복원 후 산출 |
| F1 | 물리모델 유도 | Stefan, TTOP, Kudryavtsev, GIPL2, CryoGrid |
| F0 | 간접·기존모델 | SAR retrieval, 기존 ML ALT 지도 |

Fidelity는 고정된 절대순위가 아니라 품질 메타데이터와 함께 사용한다. 예를 들어 센서 간격이 매우 큰 F3는 잘 보정된 수치모델 F1보다 불확실할 수 있다.

## 6.2 잠재 ALT와 자료원별 관측모델

잠재적 실제 ALT를 \(z_i\)로 둔다.

\[
z_i=ALT_i^\ast=f_\theta(\mathbf{x}_i)
\]

자료원 \(s\)의 관측값은 다음과 같이 본다.

\[
y_{i,s}
=
\mathcal{A}_s[z_i]
+
b_s(\mathbf{x}_i,\mathbf{q}_i)
+
\epsilon_{i,s}
\]

\[
\epsilon_{i,s}
\sim
\mathcal{N}
\left(
0,\sigma_s^2(\mathbf{x}_i,\mathbf{q}_i)
\right)
\]

- \(\mathcal{A}_s\): 공간·시간 관측연산자
- \(b_s\): 자료원별 조건부 편향
- \(\sigma_s\): 자료원별 조건부 불확실성
- \(\mathbf{q}\): 품질 메타데이터

직접관측 자료원을 기준으로 두어 식별성을 확보한다.

\[
b_{\mathrm{direct}}=0
\]

---

# 7. 데이터 스키마

## 7.1 기본 sample unit

기본 학습 단위는 다음 중 하나로 고정한다.

```text
site_id × plot_id × year
```

계절 시계열 연구에서는:

```text
site_id × plot_id × datetime × depth
```

모든 자료를 최종적으로 동일한 `site-year` 또는 `site-time-depth` 키에 연결한다.

## 7.2 필수 메타데이터

```yaml
sample_id:
site_id:
plot_id:
region_id:
country:
latitude:
longitude:
elevation_m:
year:
observation_date:
observation_start:
observation_end:

target_name:
target_value:
target_unit:
label_method:
source_id:
fidelity_level:
quality_flag:
estimated_uncertainty:
spatial_support_type:
spatial_support_scale_m:
temporal_support_type:

probe_limit_cm:
rock_hit:
right_censored:
interval_lower:
interval_upper:

sensor_depths_cm:
sensor_spacing_cm:
temperature_frequency:
temperature_missing_rate:
end_of_season_coverage:
freeze_point_assumption_c:

disturbance_type:
disturbance_year:
biome:
permafrost_zone:
soil_class:
organic_layer_cm:

split_group_site:
split_group_region:
split_group_campaign:
split_group_pixel:
```

## 7.3 target 구분

`target_name`은 최소한 다음을 구분한다.

- `alt_direct`
- `alt_temperature_derived`
- `alt_reconstructed_temperature`
- `alt_stefan`
- `alt_modified_stefan`
- `alt_ttop`
- `alt_kudryavtsev`
- `alt_numerical_model`
- `alt_remote_sensing`
- `thaw_depth`
- `ground_temperature`

## 7.4 QC 기준

- 단위 통일: ALT는 m, 온도는 °C, 시간은 UTC 또는 명시된 local time
- 지표면 기준 통일
- 중복 관측 식별
- 동일 campaign에서 생성된 파생자료에 동일 group ID 부여
- 탐침 한계와 rock hit 분리
- 측정일이 최대 해빙시기인지 명시
- 온도장 유도 ALT는 센서 간격과 계절 커버리지 기록
- 원자료와 파생자료 provenance 유지
- 모든 파생 label에 생성 함수·파라미터·버전 기록

---

# 8. 입력변수

## 8.1 정적 변수

- 위도·경도
- 고도
- 경사
- 사면방향
- 곡률
- 지형습윤지수
- 토성
- 토양분류
- 유기층 두께
- 기반암 깊이
- 공극률
- 열전도도
- 열용량
- 용적함수율
- 토양 얼음함량
- 식생·토지피복
- 수계와의 거리
- 영구동토 구분
- 산불·벌채·도로 등 교란
- 공간지지 규모

## 8.2 동적 변수

- 일·시간 기온
- 지표면온도
- 강수
- 적설 깊이
- SWE
- 단파·장파복사
- 바람
- 습도
- 토양수분
- 동결도일수
- 해빙도일수
- 적설 소멸일
- 해빙 시작일
- 극한 고온일수
- 강우·융설 사건
- 이전 겨울 냉각량
- 이전 해 ALT
- 이전 해 지중온도 상태

## 8.3 특징 생성 원칙

- 원시 시계열을 보존
- 누적량, 극한값, 지속시간을 별도 생성
- 예측시점 이후 정보 사용 금지
- 모든 전처리는 CV 학습 fold 내부에서 fitting
- 기후 정상값과 해당 연도 anomaly를 분리
- 입력해상도와 target 공간지지 규모의 불일치를 기록

---

# 9. 물리 기반 ALT 생성

## 9.1 Stefan baseline

\[
ALT_{\mathrm{Stefan}}
=
E\sqrt{TDD}
\]

또는 물성 기반 형태:

\[
ALT
=
\sqrt{
\frac{2k_t I_s}{L_v}
}
\]

## 9.2 modified Stefan

가능한 요소:

- thawing \(n\)-factor
- 토층별 열전도도
- 유기층 보정
- 함수율·얼음함량
- 지표온도 기반 TDD
- 적설·식생에 따른 지표경계 보정

## 9.3 물리 ensemble

최소 후보:

- Classical Stefan
- Modified Stefan
- TTOP
- Kudryavtsev
- 1-D enthalpy model

각 모델에 대해 parameter ensemble을 생성한다.

\[
ALT_{k,m}^{phys}
\]

- \(k\): 물리모델 종류
- \(m\): 파라미터 또는 forcing ensemble member

다음 통계를 저장한다.

\[
\mu_{phys,k},\quad
\sigma_{phys,k},\quad
Q_{0.05},Q_{0.50},Q_{0.95}
\]

물리모델 간 불일치:

\[
U_{\mathrm{model}}
=
\operatorname{Var}_k
\left(
\mu_{phys,k}
\right)
\]

## 9.4 fold-safe 생성

중요 원칙:

```text
외부 테스트 fold의 실제 ALT는
물리계수 보정, n-factor 추정, kriging,
pseudo-label 신뢰도 계산에 절대 사용하지 않는다.
```

각 outer fold에서 다음 순서를 강제한다.

1. train/validation/test site 분리
2. train direct ALT만 이용해 물리계수 보정
3. train 영역에 pseudo-label 생성
4. validation에서 증강비율·가중치 선택
5. test 입력에 대해서는 학습된 계수만 사용해 물리예측
6. test direct ALT로 최종 평가만 수행

---

# 10. 온도장 유도 ALT

## 10.1 기본 산출

두 센서 깊이 \(z_1,z_2\)에서 유효 동결점 \(T_f\)를 사이에 두면:

\[
z_f(t)
=
z_1+
\frac{T_f-T_1}
{T_2-T_1}
(z_2-z_1)
\]

\[
ALT_T=\max_t z_f(t)
\]

## 10.2 불확실성 산출

Monte Carlo 또는 bootstrap으로 다음을 변동시킨다.

- 센서 정확도
- 센서 깊이오차
- 유효 동결점
- 결측구간 보간
- 측정주기
- 최대 해빙시점 누락
- 보간법

결과:

\[
ALT_T
\sim
p(ALT\mid T_{\mathrm{obs}},q)
\]

저장값:

- 평균
- 표준편차
- 5/50/95 분위수
- 최대 해빙시점 불확실성
- sensor-spacing quality score

## 10.3 복원 온도장 ALT

PINN 또는 열모델로 복원한 \(T(z,t)\)에서 ALT를 산출한 경우 직접 온도관측 유도 ALT와 별도 source로 둔다.

```text
alt_temperature_derived
≠
alt_reconstructed_temperature
```

---

# 11. 기준모델 및 제안모델

## B0. 단순 기준

- 전체 평균
- site climatology
- biome 평균
- TDD 단변량 회귀
- Stefan

## B1. 실측 ALT만 사용한 tabular ML

- Linear/Ridge/Elastic Net
- Random Forest
- Extra Trees
- HistGradientBoosting
- XGBoost 또는 LightGBM
- CatBoost

직접관측이 적은 단계에서는 이 계열을 최우선 기준으로 둔다.

## B2. 실측 ALT만 사용한 시계열 모델

- Temporal CNN
- LSTM/GRU
- Temporal Fusion Transformer 또는 단순 Transformer
- static encoder + temporal encoder

DL은 원시 기후 시계열이 충분한 경우에만 주력으로 사용한다.

## A1. Naive pooling

\[
\mathcal{D}
=
\mathcal{D}_{obs}
\cup
\mathcal{D}_{temp}
\cup
\mathcal{D}_{phys}
\]

모든 label을 동일하게 취급한다. 제안모델이 아니라 실패 가능성을 확인하는 비교군이다.

## A2. Physics feature

\[
\widehat{ALT}
=
f_\theta
(
\mathbf{x},
ALT_{\mathrm{Stefan}},
ALT_{\mathrm{TTOP}},
\ldots
)
\]

## A3. Physics pretraining–observation fine-tuning

1. 대량 물리·온도 유도 ALT로 사전학습
2. 직접 ALT만으로 미세조정
3. backbone freeze 정도와 학습률 비교

## A4. Residual correction

\[
r_i
=
ALT_i^{obs}
-
ALT_i^{phys}
\]

\[
\widehat{r}_i
=
g_\theta
(
\mathbf{x}_i,
ALT_i^{phys}
)
\]

\[
\widehat{ALT}
=
ALT^{phys}
+
\widehat{r}
\]

잔차학습은 강한 baseline으로 사용하되 주된 신규성으로 과장하지 않는다.

## A5. Source-aware multi-fidelity latent ALT

\[
z_i=f_\theta(\mathbf{x}_i)
\]

\[
y_{i,s}
\sim
\mathcal{N}
\left(
\mathcal{A}_s[z_i]+b_s(\mathbf{x}_i,\mathbf{q}_i),
\sigma_s^2(\mathbf{x}_i,\mathbf{q}_i)
\right)
\]

가능한 손실:

\[
\mathcal{L}
=
\sum_{i,s}
\left[
\frac{
\left(
y_{i,s}
-
\mathcal{A}_s[f_\theta(\mathbf{x}_i)]
-
b_s
\right)^2
}{
2\sigma_s^2
}
+
\frac{1}{2}\log\sigma_s^2
\right]
\]

제약:

- 직접관측 source bias는 0으로 고정
- source bias의 과적합을 막기 위한 regularization
- source별 최소 overlap이 부족하면 단순화
- source uncertainty는 품질 메타데이터를 입력으로 사용

## A6. Mixture-of-physics experts

\[
\widehat{ALT}
=
\sum_{k=1}^{K}
\pi_k(\mathbf{x})
\left[
M_k(\mathbf{x})+r_k(\mathbf{x})
\right]
\]

\[
\sum_k\pi_k=1
\]

전문가:

- Stefan
- modified Stefan
- TTOP
- Kudryavtsev
- 1-D enthalpy model

게이트 입력:

- 토양수분
- 유기층
- 적설
- 토성
- 식생
- 산불
- 지형
- 영구동토 구분

단순 가중평균보다 다음을 분석하는 것이 목적이다.

- 어느 환경에서 어느 물리식이 선택되는가
- 전문가 가중치가 실제 오차와 일치하는가
- 물리모델 간 불일치가 불확실성을 설명하는가

## A7. Support-aware·censored model

관측연산자:

\[
y_{i,s}
=
\mathcal{A}_s[ALT^\ast(\mathbf{r},t)]
+\epsilon
\]

검열자료:

\[
ALT_i>L_i
\]

\[
\mathcal{L}_{cens,i}
=
-\log P(ALT_i>L_i)
\]

구간자료:

\[
ALT_i\in[L_i,U_i]
\]

\[
\mathcal{L}_{interval,i}
=
-\log
\left[
F(U_i)-F(L_i)
\right]
\]

## A8. Joint \(T(z,t)\)–\(D(t)\)–ALT model

공유 잠재 열상태:

```text
static encoder
+
dynamic climate encoder
+
optional site embedding
→ latent thermal state
```

출력:

\[
\widehat{T}(z,t)
\]

\[
\widehat{D}(t)
\]

\[
\widehat{ALT}
\]

일관성 손실:

\[
\mathcal{R}_{ALT}
=
\left|
\widehat{ALT}
-
\max_t\widehat{D}(t)
\right|
\]

\[
\mathcal{R}_{TD}
=
\left|
\widehat{D}(t)
-
\mathcal{F}[\widehat{T}(z,t)]
\right|
\]

선택적 열방정식 residual:

\[
\mathcal{R}_{heat}
=
\left\|
\frac{\partial H}{\partial t}
-
\frac{\partial}{\partial z}
\left(
k\frac{\partial T}{\partial z}
\right)
-Q
\right\|^2
\]

물리 손실은 hard constraint보다 uncertainty-weighted soft constraint로 적용한다.

---

# 12. time-lapse 문제 정의

“time-lapse ALT” 대신 다음을 명확히 분리한다.

## Task T1. 계절 thaw-depth trajectory

\[
\widehat{D}(t)
\]

일별 또는 주별 해빙 깊이를 예측한다.

## Task T2. End-of-season ALT 조기예측

forecast origin \(\tau\)까지의 자료만 사용한다.

\[
\widehat{ALT}_{EOS}
=
f(\mathbf{x}_{1:\tau})
\]

예측시점:

- 해빙 시작 시점
- 5월 말
- 6월 말
- 7월 말
- 30/60/90일 선행

## Task T3. 연간 multi-horizon 예측

\[
\widehat{ALT}_{y+h},
\qquad
h=1,\ldots,5
\]

이 과제는 후순위다. 우선 현재 연도 계절예측과 공간 OOD를 안정화한다.

---

# 13. 증강비율 실험

## 13.1 비율 정의

실제자료 수를 \(N_{obs}\), pseudo-label 수를 \(N_p\)라 하면:

\[
r=\frac{N_p}{N_{obs}}
\]

후보:

```text
r ∈ {0, 0.25, 0.5, 1, 2, 5, 10}
```

단, 단순 sample 수와 loss 영향력을 분리한다.

- sampling ratio
- source loss weight
- uncertainty weight
- source-balanced batch

## 13.2 증강 축

1. 단일 Stefan
2. 복수 물리식
3. 온도 유도 ALT
4. 복원 온도 유도 ALT
5. 관측 분포와 유사한 pseudo-label
6. 관측 분포에서 먼 pseudo-label
7. 낮은 물리모델 불일치
8. 높은 물리모델 불일치

## 13.3 핵심 분석

\[
\Delta Skill
=
f(
r,
bias,
variance,
domain\ distance,
model\ disagreement
)
\]

단순히 최적 \(r\) 하나를 찾는 것이 아니라 조건별 반응곡선을 제시한다.

---

# 14. 교차검증 및 정보 누출 방지

## 14.1 금지되는 주 검증

- 단순 random K-fold만 사용
- 같은 site의 서로 다른 연도를 train/test에 분리
- 같은 campaign의 파생자료를 서로 다른 fold에 배치
- 같은 1 km 픽셀의 중복자료를 train/test에 분리
- 전체자료로 Stefan 계수 보정 후 CV
- 전체자료로 결측대체·표준화·PCA 후 CV
- 전체 해빙기 온도를 사용해 중간시점 ALT 예측

## 14.2 필수 outer split

최소 세 종류를 수행한다.

### Spatial OOD

- leave-site-out
- spatial block CV
- leave-region-out

블록 크기는 공간 자기상관 분석 후 결정한다.

### Temporal OOD

- 과거 연도 학습 → 미래 연도 테스트
- rolling-origin evaluation

### Spatiotemporal OOD

- 새로운 지역의 미래 연도 테스트

## 14.3 추가 holdout

- leave-biome-out
- leave-permafrost-zone-out
- leave-disturbance-type-out
- leave-measurement-method-out
- leave-source-out

## 14.4 Nested CV

```text
Outer fold:
    최종 일반화 평가

Inner fold:
    모델 선택
    hyperparameter tuning
    물리계수 보정
    증강비율 선택
    source weight 선택
    feature selection
```

## 14.5 그룹 규칙

동일 그룹은 반드시 같은 fold에 둔다.

- `site_id`
- `plot_id`
- `campaign_id`
- `source_campaign`
- `pixel_id`
- `derived_from_observation_id`

---

# 15. 평가 지표

## 15.1 ALT 점예측

- MAE
- RMSE
- Median Absolute Error
- Bias
- \(R^2\)
- Spearman correlation
- 깊은 ALT 구간의 tail MAE
- biome/region별 group metric

주지표는 MAE와 bias로 둔다. \(R^2\) 단독 사용 금지.

## 15.2 확률예측

- CRPS
- Negative Log-Likelihood
- 50/80/90/95% interval coverage
- Mean prediction interval width
- Calibration error
- Region/biome별 reliability

## 15.3 thaw-depth trajectory

- 시점별 MAE
- integrated absolute error
- maximum thaw depth error
- maximum thaw date error
- thaw onset error
- freeze-up error

## 15.4 지중온도장

- depth별 RMSE
- season별 RMSE
- 0°C 인근 RMSE
- 위상오차
- 진폭오차
- zero-curtain duration error
- thaw/freeze timing error

## 15.5 물리적 일관성

- \(ALT-\max D(t)\) 불일치
- \(D(t)-\mathcal{F}[T]\) 불일치
- 열방정식 residual
- 비현실적 시간진동
- 깊이방향 비연속성
- 물리적으로 불가능한 ALT 확률

## 15.6 통계적 비교

- site 또는 region 단위 paired bootstrap
- fold별 신뢰구간
- 모델 간 paired error difference
- 다중비교 시 보정
- 평균뿐 아니라 분포와 실패지역 제시

---

# 16. Ablation 계획

| ID | 구성 | 목적 |
|---|---|---|
| B0 | 평균·기후평년·TDD | 최소 기준 |
| B1 | Stefan | 물리 기준 |
| B2 | 실측 ALT만 tabular ML | 강한 데이터 기준 |
| B3 | 실측 ALT만 temporal model | DL 기준 |
| A1 | 실측+물리 단순 pooling | naive augmentation |
| A2 | 물리 ALT를 입력특징으로 사용 | physics feature |
| A3 | physics pretrain → obs fine-tune | transfer |
| A4 | physics residual correction | bias correction |
| A5 | source-aware multi-fidelity | 주 제안 |
| A6 | source + uncertainty | 불확실성 효과 |
| A7 | support-aware | 공간지지 효과 |
| A8 | censored likelihood | 검열자료 효과 |
| A9 | mixture-of-physics experts | 조건부 물리모델 선택 |
| A10 | \(T-D-ALT\) 공동학습 | 물리 일관성 |
| A11 | active learning | 추가관측 효율 |

---

# 17. 단계별 연구 진행안

## Phase 0. 문헌·신규성 감사

### 목표

동일하거나 매우 유사한 연구의 존재 여부를 체계적으로 확인한다.

### 작업

- 검색식과 포함·제외기준 작성
- ALT physics-guided ML
- permafrost multi-fidelity
- pseudo-label ALT
- temperature-derived ALT learning
- residual correction ALT
- physics mixture of experts permafrost
- censored active layer thickness model
- spatial support ALT fusion
- thaw-depth trajectory forecasting
- joint ground temperature and ALT prediction

### 산출물

- `literature_matrix.csv`
- 연구별 데이터, target, 물리결합 방식, 검증방식, 한계
- 방어 가능한 novelty statement 3개
- 금지할 first-claim 목록

### 통과 기준

- 현재 제안과 동일한 6요소 조합의 선행연구가 있는지 확인
- 있으면 방법론이 아니라 데이터·검증·과학질문으로 차별화 수정

---

## Phase 1. 데이터 감사와 통합

### 목표

자료원별 overlap과 식별 가능성을 먼저 확인한다.

### 작업

- 직접 ALT 자료 수집
- 온도장 자료 수집
- 입력변수 소스 정리
- site-year 기준 통합
- source/fidelity/quality 메타데이터 생성
- 공간좌표·날짜·단위 QC
- 중복·파생관계 식별

### 핵심 산출물

`source_overlap_matrix.parquet`

예:

| source pair | 동일 site-year 수 | 동일 region-year 수 |
|---|---:|---:|
| direct–temperature |  |  |
| direct–Stefan |  |  |
| direct–Kudryavtsev |  |  |
| temperature–physics |  |  |

### 중단 기준

자료원 간 동일 site-year overlap이 거의 없으면 source bias와 지역효과를 분리하기 어렵다.

이 경우 우선순위를 다음으로 낮춘다.

1. residual correction
2. pretrain–fine-tune
3. coarse source-level fixed effect
4. 완전한 계층모델은 보류

---

## Phase 2. 누출 없는 benchmark 구축

### 목표

어떤 복잡한 모델보다 먼저 재현 가능한 기준선을 구축한다.

### 작업

- Grouped spatial split
- temporal split
- nested CV
- fold-safe preprocessing
- fold-safe physics calibration
- B0–B3 구현
- 결과 artifact 저장

### 통과 기준

- 동일 seed에서 재현
- split manifest 저장
- test label이 전처리·물리보정에 사용되지 않음을 unit test로 검증
- fold별 성능과 전체 성능 모두 출력

---

## Phase 3. 데이터 증강 방식 비교

### 목표

단순 병합이 실제로 유효한지 검증한다.

### 작업

- A1 naive pooling
- A2 physics feature
- A3 pretrain–fine-tune
- A4 residual correction
- 증강비율 실험
- source-balanced sampling
- OOD별 성능 비교

### 통과 기준

- 실제 direct ALT test에서 B2 대비 개선 여부
- 랜덤 CV 개선과 공간 OOD 개선을 분리 보고
- 물리 pseudo-label이 해로운 조건 식별

---

## Phase 4. Source-aware multi-fidelity

### 목표

자료원별 편향과 불확실성을 학습한다.

### 작업

- source embedding
- bias head
- heteroscedastic uncertainty head
- direct source anchor
- overlap-aware regularization
- calibration 평가
- source ablation

### 통과 기준

- naive pooling과 residual보다 direct ALT OOD 성능 개선
- 불확실성 interval이 calibration됨
- source bias가 실제 residual pattern과 일치
- 특정 source가 과도하게 지배하지 않음

---

## Phase 5. Support-aware·censored 확장

### 목표

ALT 관측의 공간지지 규모와 측정한계를 반영한다.

### 작업

- point/grid/pixel support feature
- 가능하면 observation operator 구현
- right-censored likelihood
- interval likelihood
- rock-hit 별도 처리
- deep ALT tail 평가

### 통과 기준

- censored 자료를 삭제한 모델보다 tail bias 감소
- support-aware 모델이 서로 다른 측정방법 holdout에서 안정적

---

## Phase 6. 계절 해빙 및 조기예측

### 목표

ALT 단일값을 넘어 \(D(t)\)와 end-of-season forecast를 구현한다.

### 작업

- thaw-depth time series 구축
- forecast origin mask
- TCN/LSTM/Transformer baseline
- physics consistency loss
- 30/60/90일 선행평가

### 통과 기준

- 미래정보 누출 없음
- 각 forecast origin별 calibration 보고
- 조기예측이 단순 TDD extrapolation보다 우수한지 확인

---

## Phase 7. 지중온도장 공동예측

### 목표

\(T(z,t)\), \(D(t)\), ALT를 공동학습한다.

### 작업

- depth-time decoder
- temperature observation mask
- ALT consistency head
- optional heat-equation residual
- sparse sensor experiment
- unseen depth interpolation 평가

### 통과 기준

- ALT 성능을 희생하지 않으면서 온도장 성능 향상
- 물리 일관성 지표 개선
- PINN 온도복원 연구와의 차별성 명확화

---

## Phase 8. Active learning

### 목표

추가 현장조사 후보를 선정한다.

### acquisition score 예시

\[
A(\mathbf{x})
=
\alpha U_{\mathrm{predictive}}
+
\beta U_{\mathrm{model\ disagreement}}
+
\gamma D_{\mathrm{domain}}
-
\delta C_{\mathrm{field}}
\]

### 평가

과거자료를 이용한 retrospective simulation:

1. 일부 direct ALT를 숨김
2. acquisition score로 순차 선택
3. 무작위, uncertainty-only, disagreement-only와 비교
4. 관측 수 대비 OOD error 감소곡선 작성

---

# 18. 저장소 구조 제안

```text
alt-multifidelity/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── data/
│   ├── physics/
│   ├── model/
│   ├── split/
│   └── experiment/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── docs/
│   ├── research_plan.md
│   ├── data_dictionary.md
│   ├── leakage_policy.md
│   ├── novelty_matrix.md
│   └── experiment_registry.md
├── src/altmf/
│   ├── data/
│   │   ├── ingest.py
│   │   ├── harmonize.py
│   │   ├── qc.py
│   │   ├── schema.py
│   │   └── provenance.py
│   ├── features/
│   │   ├── climate.py
│   │   ├── static.py
│   │   └── temporal.py
│   ├── physics/
│   │   ├── stefan.py
│   │   ├── modified_stefan.py
│   │   ├── ttop.py
│   │   ├── kudryavtsev.py
│   │   ├── enthalpy_1d.py
│   │   └── uncertainty.py
│   ├── targets/
│   │   ├── temperature_to_alt.py
│   │   ├── censoring.py
│   │   └── quality.py
│   ├── splits/
│   │   ├── spatial.py
│   │   ├── temporal.py
│   │   ├── nested.py
│   │   └── guards.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── residual.py
│   │   ├── pretrain_finetune.py
│   │   ├── multifidelity.py
│   │   ├── moe_physics.py
│   │   ├── censored.py
│   │   └── joint_thermal.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   ├── bootstrap.py
│   │   └── diagnostics.py
│   └── pipelines/
│       ├── build_dataset.py
│       ├── generate_physics.py
│       ├── train.py
│       └── evaluate.py
├── tests/
│   ├── test_units.py
│   ├── test_splits.py
│   ├── test_leakage.py
│   ├── test_physics.py
│   ├── test_censoring.py
│   └── test_reproducibility.py
├── scripts/
│   ├── run_benchmark.py
│   ├── run_augmentation_ablation.py
│   ├── run_multifidelity.py
│   └── make_report.py
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_overlap_analysis.ipynb
│   ├── 02_spatial_autocorrelation.ipynb
│   └── 03_error_diagnostics.ipynb
└── outputs/
    ├── models/
    ├── predictions/
    ├── metrics/
    ├── figures/
    └── reports/
```

---

# 19. 구현 원칙

## 19.1 재현성

- 모든 실험은 config 파일로 실행
- seed 고정
- split manifest 버전관리
- 원자료 checksum 저장
- 파생자료 provenance 저장
- 모델·데이터·코드 commit hash 기록
- 결과표 자동생성

## 19.2 누출 방지 unit test

최소 테스트:

1. outer test site가 물리계수 calibration에 포함되지 않는지
2. 동일 `derived_from_observation_id`가 여러 fold에 존재하지 않는지
3. forecast origin 이후 변수가 입력에 포함되지 않는지
4. scaler/imputer가 train에서만 fit되는지
5. test target이 feature 생성에 사용되지 않는지
6. pseudo-label과 그 원천 direct label이 서로 다른 fold로 분리되지 않는지

## 19.3 모델 복잡도 원칙

- 작은 direct dataset에서 대형 DL을 기본값으로 사용하지 않음
- tabular boosting을 반드시 강한 기준으로 유지
- DL은 원시 시계열, 대량 사전학습, 공동예측에서 사용
- 성능개선이 없는 복잡성은 제거
- 해석과 calibration을 성능과 함께 평가

---

# 20. 위험요인과 대응

## 위험 1. 자료원 overlap 부족

### 문제

source bias와 지역효과가 식별되지 않음.

### 대응

- 동일 site-year에서 물리모델 결과를 모두 계산해 overlap 생성
- direct–temperature overlap을 우선 확보
- source bias를 복잡한 함수가 아닌 fixed/random effect로 단순화
- pretrain–fine-tune 또는 residual을 주력으로 전환

## 위험 2. 물리 pseudo-label이 입력의 단순 변환

### 문제

새 정보가 아니라 물리식 복제만 발생.

### 대응

- pseudo-label을 정답으로만 넣지 않고 feature·prior·residual baseline과 비교
- 관측되지 않은 환경공간에서만 추가되는 정보인지 분석
- 물리모델의 파라미터 불확실성과 편향을 함께 저장

## 위험 3. 실제 ALT 정의 불일치

### 문제

측정일 thaw depth와 annual maximum ALT 혼합.

### 대응

- `thaw_depth`와 `alt` target 분리
- end-of-season coverage 기록
- 보정된 ALT는 별도 source로 관리
- 직접측정과 추정값을 혼합 표기하지 않음

## 위험 4. 공간해상도 불일치

### 문제

점 관측과 재분석·위성 픽셀 간 대표성 오차.

### 대응

- support scale feature
- multi-scale pooling
- pixel 내 관측분산 추정
- support-aware 모델을 확장실험으로 수행

## 위험 5. 과도한 연구범위

### 문제

ALT, 온도장, PINN, active learning, 미래시나리오를 동시에 수행하면 핵심이 흐려짐.

### 대응

논문 우선순위:

1. 데이터·benchmark
2. source-aware ALT
3. \(D(t)\)와 \(T(z,t)\) 공동예측
4. active learning 또는 장기전망

---

# 21. 예상 논문 구성

## Paper 1. 데이터 및 benchmark

### 핵심

- 이질적 ALT 자료 통합
- 누출 없는 spatial/temporal benchmark
- naive augmentation의 한계
- 실측자료만 사용한 강한 baseline

### 기여문장

> ALT 예측에서 데이터 증강 효과를 랜덤 분할이 아니라 새로운 지역과 미래 연도에 대한 직접관측 ALT로 평가하는 재현 가능한 benchmark를 제시한다.

## Paper 2. Source-aware multi-fidelity ALT

### 핵심

- 자료원별 편향
- 이분산 불확실성
- residual/pretraining/pooling 비교
- support·censoring 선택적 확장

### 기여문장

> 직접 ALT, 온도장 유도 ALT, 복수 물리모델 ALT를 동일한 정답으로 취급하지 않고 자료원별 조건부 편향과 불확실성을 학습하여 잠재적 실제 ALT를 추정한다.

## Paper 3. 계절 해빙·온도장 공동예측

### 핵심

- \(D(t)\), \(T(z,t)\), ALT 공동학습
- 부분계절 end-of-season ALT 예측
- sparse sensor assimilation
- 물리 일관성

### 기여문장

> 계절 해빙전선, 얕은 지중온도장, 연간 ALT를 하나의 잠재 열상태로 연결하고, 제한된 계절 관측만으로 최종 ALT를 확률적으로 예측한다.

---

# 22. 성공 기준

## 최소 성공

- B2 실측-only ML보다 residual 또는 pretrain–fine-tune이 공간 OOD에서 안정적으로 개선
- naive pooling이 언제 실패하는지 규명
- fold-safe benchmark 공개
- 직접 ALT test만으로 최종 결론 도출

## 중간 성공

- source-aware model이 residual보다 유의하게 개선
- interval calibration 개선
- source별 편향이 환경조건과 해석 가능하게 연결
- 검열자료 활용으로 깊은 ALT 편향 감소

## 높은 성공

- support-aware source model 구현
- \(T-D-ALT\) 공동예측에서 물리 일관성과 OOD 성능 동시 개선
- active learning이 현장관측 효율을 개선
- 범지역 데이터에서 일반화 재현

---

# 23. Claude Code가 우선 수행할 작업

## 첫 번째 구현 스프린트

1. 위 저장소 구조 생성
2. `CLAUDE.md`에 연구 원칙과 금지사항 기록
3. `schema.py`에 데이터 스키마 정의
4. `splits/`에 group spatial/temporal split 구현
5. 누출 방지 테스트 작성
6. Stefan baseline 구현
7. 실측-only tabular baseline 구현
8. 실험 config와 결과저장 형식 구현
9. synthetic toy dataset으로 전체 pipeline smoke test
10. 실제 데이터가 연결되면 overlap audit 실행

## 두 번째 구현 스프린트

1. temperature-to-ALT 변환
2. 온도 유도 ALT Monte Carlo uncertainty
3. naive pooling
4. physics feature
5. pretrain–fine-tune
6. residual correction
7. 증강비율 ablation
8. spatial/temporal OOD 보고서 자동생성

## 세 번째 구현 스프린트

1. source-aware latent ALT
2. source bias head
3. heteroscedastic uncertainty
4. calibration metrics
5. source ablation
6. overlap 부족 시 fallback 모델

## 네 번째 구현 스프린트

1. censored likelihood
2. support-aware feature
3. mixture-of-physics experts
4. joint \(T-D-ALT\) prototype
5. active-learning simulation

---

# 24. Claude Code용 작업 프롬프트

아래 내용을 Claude Code에 직접 전달할 수 있다.

```text
프로젝트명: alt-multifidelity

목표:
희소한 실제 ALT 관측을 보완하기 위해 direct ALT, temperature-derived ALT,
physics-derived ALT를 결합하되 동일한 정답으로 단순 병합하지 않고,
자료원별 편향·불확실성과 공간·시간 일반화를 반영하는 재현 가능한
multi-fidelity 학습 파이프라인을 구축한다.

가장 중요한 원칙:
1. 최종 평가는 학습에 사용하지 않은 direct ALT에서만 수행한다.
2. 모든 physics calibration과 pseudo-label 생성은 outer CV train fold 내부에서 수행한다.
3. 같은 site, campaign, pixel, 원천관측에서 파생된 샘플은 같은 fold에 둔다.
4. random K-fold 결과를 주결과로 사용하지 않는다.
5. 실측-only tabular boosting을 강한 baseline으로 유지한다.
6. naive pooling, physics feature, pretrain–fine-tune, residual, source-aware
   multi-fidelity를 순서대로 비교한다.
7. 모델 복잡도보다 leakage-free evaluation과 calibration을 우선한다.
8. 실제 데이터가 아직 없어도 synthetic data로 전체 pipeline과 test를 먼저 만든다.

우선 구현:
- Python 패키지 구조
- Pydantic 또는 dataclass 기반 데이터 스키마
- spatial/temporal/nested grouped split
- leakage guard와 unit test
- Stefan baseline
- direct-only sklearn boosting baseline
- config-driven experiment runner
- metric/calibration/report 모듈
- synthetic site-year dataset generator
- README와 실행 예시

필수 데이터 컬럼:
sample_id, site_id, plot_id, region_id, latitude, longitude, year,
target_name, target_value, label_method, source_id, fidelity_level,
estimated_uncertainty, spatial_support_type, spatial_support_scale_m,
probe_limit_cm, rock_hit, right_censored, interval_lower, interval_upper,
sensor_spacing_cm, end_of_season_coverage, biome, permafrost_zone,
disturbance_type, derived_from_observation_id.

필수 테스트:
- test site target이 physics calibration에 들어가지 않음
- 파생자료가 원천자료와 다른 fold에 들어가지 않음
- scaler/imputer가 train fold에서만 fit됨
- forecast origin 이후 feature가 제거됨
- 동일 seed에서 split과 metric 재현
- Stefan 계산 단위 검증

첫 결과물:
1. 저장소 구조
2. synthetic dataset을 이용한 benchmark 실행
3. B0 평균, B1 Stefan, B2 direct-only boosting 결과표
4. split manifest
5. leakage test 통과 로그
6. 다음 구현 이슈 목록

아직 구현하지 말 것:
- 대형 Transformer
- 복잡한 PINN
- 범북극 고해상도 추론
- CMIP 장기전망
- 실제 데이터 없이 과도한 딥러닝 구조 최적화
```

---

# 25. 최종 연구 방향 요약

현재 연구는 다음과 같이 정의한다.

\[
\boxed{
\text{물리 기반 ALT 데이터 증강}
\rightarrow
\text{불확실하고 편향된 다중 자료원의 신뢰도 학습}
}
\]

가장 현실적인 진행 순서는 다음이다.

\[
\boxed{
\begin{aligned}
&\text{데이터·overlap 감사}\\
&\rightarrow \text{누출 없는 benchmark}\\
&\rightarrow \text{잔차·사전학습 비교}\\
&\rightarrow \text{source-aware multi-fidelity}\\
&\rightarrow \text{support·censoring}\\
&\rightarrow D(t)\text{ 조기예측}\\
&\rightarrow T(z,t)\text{ 공동예측}
\end{aligned}
}
\]

연구의 차별성은 특정 딥러닝 모델 이름에서 나오지 않는다. 다음 네 요소의 결합에서 나온다.

1. 실제·온도유도·물리유도 ALT를 서로 다른 관측과정으로 모델링
2. 자료원별 조건부 편향과 불확실성 추정
3. 공간지지 규모와 검열정보 반영
4. 새로운 지역과 미래 연도에 대한 직접관측 ALT 검증

가장 우선할 모델은 다음 순서다.

```text
실측-only boosting
→ Stefan
→ residual correction
→ pretrain–fine-tune
→ source-aware multi-fidelity
→ support-aware/censored extension
→ joint T-D-ALT
```

대회형 모델 비교는 보조실험으로 수행한다. 주된 논문은 “어떤 모델이 가장 높은 점수를 냈는가”가 아니라 다음 질문에 답해야 한다.

> 물리 기반 저충실도 ALT는 실제 관측이 부족한 상황에서 언제 도움이 되고,
> 언제 편향을 증폭시키며, 그 신뢰도를 어떻게 학습해야 하는가?
