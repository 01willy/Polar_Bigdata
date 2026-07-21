# P2 실험 계획: ALT를 중심으로 3D 지중온도장·물리 결합 (2026-07-14)

확정 프레임: **ALT 예측이 main 산출.** 3D 지중온도장은 라벨 증강 도구, Stefan(및 개선 물리)은
광역 예측·물리 사전지식. 세 경로가 0°C 등온면을 공유 경계로 함. 근거 [connection 개념도](../deck/assets/mid/connection.png).

## 공통 원칙(모든 트랙에 강제)
1. **전 입력 활용**: 사용 가능한 모든 공변량(지형 6·기후 8·InSAR 5·PolSAR 3·CCI 2, 결측은 지시자)과 라벨(ALT 프로브·시추공 유도·물리)을 명시적으로 투입. 무엇을 뺐으면 사유 기록.
2. **데이터 관리**: 라벨 출처·유도 규칙·결측 처리·중복 제거를 메타 JSON에 기록. 기존 파일 불변, 신규 파일로 산출.
3. **산출·평가**: 공간블록 6-fold + LORO(지역 제외) 동일 프로토콜. skill_over_mean·R²·RMSE 병기. OOF 예측 저장(지도용). 게이트(개선 시 채택) 명시.

## 트랙 M (main) · 3D 지중온도장 라벨 증강
- 목적: 시추공 지중온도가 있으나 ALT 프로브가 없는 지역에서 0°C 등온면 깊이로 ALT 라벨을 유도해
  알래스카 편중(97%)을 완화, ALT 전이 성능을 높인다.
- 방법: GTN-P 전 시추공(ground_temp_gtnp_global.csv의 깊이별 t_max)에서 0°C 교차 깊이 = ALT 유도(P1의 GTNPenv 확장).
  증강 전/후 ALT 모델을 공간블록·LORO로 비교.
- 게이트: 증강이 LORO 전이 RMSE를 개선하면 채택. 알래스카 in-domain 저하 여부도 병기.
- 산출: `p2m_augment_results.csv`, OOF 지도.

## 트랙 α · 3D 지중온도장 기질 추정(전 공변량)
- 목적: 지중온도장 T(x,y,z)를 climate+depth만이 아니라 전 공변량(지형·CCI 등 전지구 커버 축 추가)으로 추정,
  깊이별 RMSE를 평가하고 0°C 등온면으로 ALT를 유도해 direct ALT와 비교.
- 방법: ground_temp_all + 공변량, GBM 조건장. 평가 = site-disjoint + 깊이별 RMSE. ALT = 0°C 등온면 깊이.
- 게이트: 전 공변량이 climate+depth 대비 깊이별 RMSE 개선 시 채택. field-유도 ALT가 direct ALT와 정합한지 확인.
- 산출: `p2a_field_results.csv`, 깊이별 RMSE 표, ALT 정합 산점도.

## 트랙 β · Stefan(및 개선 물리) 잔차학습
- 목적: Stefan 물리(ALT ≈ E·√TDD, E=edaphic factor)로 ALT 1차 추정 → ML이 잔차 보정.
  순수 ML·물리단독 대비 라벨 희소·전이에서 견고성 검증.
- 방법: (1) Stefan-only: E를 train fold에서 적합(ALT=E√TDD). (2) 잔차학습: 관측−Stefan을 전 공변량 GBM으로.
  개선 물리(예: 토양수분·유기물층 보정 계수, 최신 논문 기반)는 후속 확장 슬롯.
- 게이트: 잔차학습이 순수 ML 및 물리단독보다 공간블록·LORO에서 개선하면 채택. 특히 전이(LORO)에서 물리 결합 이점 확인.
- 산출: `p2b_stefan_results.csv`(physics-only / ML-only / residual × 공간블록·LORO).

## 평가 지표 공통
- RMSE(cm), MAE, bias, R², skill_over_mean(=1−RMSE/σ), (해당 시) 깊이별 RMSE.
- 결과는 `docs/EXPERIMENT_P2_RESULTS_2026-07-14.md`에 정리 후 덱 반영 여부 판단.
