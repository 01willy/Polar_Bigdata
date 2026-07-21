# P2 실험 결과 (2026-07-14) · ALT 중심 물리·데이터 결합

계획 [EXPERIMENT_PLAN_P2_2026-07-14.md](EXPERIMENT_PLAN_P2_2026-07-14.md). 3트랙 병렬 실행 + 적대적 검증.
스크립트 `scripts/3_deep_learning/p2_{augment,field,stefan}_experiment.py`.

## 핵심 결론

1. **물리 우선이 전이에 강하다(가장 중요).** Stefan 계열 물리(ALT = a + E·√TDD)가 지역 전이(LORO)에서
   RMSE 18.2cm로 순수 ML 40.6cm를 압도. 순수 ML은 알래스카 분포에 과적합해 시베리아 전이서 bias +86.6cm로 파탄.
   단일(사실상 2-파라미터 아핀) 물리는 분포 이동에 견고(Lena bias +7.0cm).
2. **잔차학습은 현 구성에서 무익.** GBM 잔차보정이 ML과 같은 전이 실패를 상속(LORO RMSE 48.8cm). 게이트 REJECT.
   함의: 물리를 base로만 쓰는 게 낫고, 잔차는 in-domain(캐나다 ABoVE_CA)에서만 소폭 우세.
3. **라벨 증강은 지역 선별이 필요하다.** Lena 대규모 라벨(3,037셀)은 자기일관성 유효(in-domain 17.84cm, skill +0.154).
   그러나 GTNPenv 심부 소수 라벨(37셀, 평균 154cm)이 Lena OOD 냉대륙 기후서 128cm 과대추정 교란 → 전이 파탄.
   게이트 미채택(현 구성). GTNPenv 심부 라벨 제외/AOA 게이팅 후 재평가 권고.
4. **3D 지중온도장에 전 공변량은 소폭 이득.** 지형+CCI 추가로 온도장 RMSE 1.23→1.18°C(심부 5-20m 개선).
   게이트 ADOPT(잠정). 단 CV 프로토콜 재검증 필요(아래 caveat).

## 트랙별 수치

### M · 라벨 증강 (`p2m_augment_results.csv`) · 게이트 미채택
| 평가 | BASE(알래스카만) | AUG(증강) |
|---|---|---|
| 알래스카 in-domain(공간블록) | 18.22cm (skill +1.7%) | 20.37cm (−9.9%) |
| Lena_RU 전이 | 21.78cm | 89.20cm |
| 진단: Lena 자기일관(AK+Lena, Lena CV) | · | **17.84cm (skill +0.154)** |
- BASE→Lena 21.78cm는 실력이 아니라 알래스카 평균(~50cm) 상수예측(pred_std 3.8, skill≈0). AUG는 GTNPenv 교란으로 상향 편향.

### α · 3D 기질 추정 (`p2a_field_results.csv`) · 게이트 ADOPT(잠정)
| 모델 | 전체 RMSE | R² | 5-10m | 10-20m |
|---|---|---|---|---|
| FIELD_base(기후+깊이) | 1.227°C | 0.594 | 1.199 | 1.217 |
| FIELD_full(+지형+CCI) | 1.181°C | 0.624 | 1.075 | 1.147 |
- field-ALT 정합: 연최대 포락선(δ=0.60m 보정) RMSE 158cm, bias +5.9cm, r=0.43(정성 정합, 절대오차 큼).

### β · Stefan 잔차학습 (`p2b_stefan_results.csv`) · 게이트 REJECT
| 방식 | 공간블록 RMSE(skill) | LORO RMSE(skill) |
|---|---|---|
| PHYS_only(a+E√TDD) | **17.86 (0.117)** | **18.24 (0.052)** |
| ML_only | 20.52 (−0.014) | 40.64 (−1.114) |
| RESIDUAL | 20.26 (−0.001) | 48.84 (−1.540) |
| PHYS_strat(E 층화) | 17.89 (0.116) | 19.04 (0.010) |

## 검증에서 나온 주의점(후속 반영 필요)
- **트랙 α CV 이탈**: 공통 원칙(공간블록+LORO) 대신 site-GroupKFold를 써서 근접 시추공(190/260이 같은 0.5°블록)
  누설 위험 → 심부 개선이 낙관 편향 가능. **공간블록+LORO로 재평가 필요.** field-ALT δ는 in-sample 적합.
  유도 라벨 일부 비물리(>400cm, 얕은 구간 미교차 사이트) → 유계 클립 필요.
- **트랙 M**: GTNPenv_ALL(38)에 알래스카 내 GTNPenv_US 9셀 포함(순수 out-region 아님). 심부 라벨 QC 미적용.
- **트랙 β**: PHYS_only는 실제 2-파라미터 아핀(a+E√TDD). '단일 파라미터' 표현은 부정확(결론은 유효).

## 다음 실험(권고)
1. 물리 base 고도화: Stefan 아핀을 토양수분·유기물층·적설로 확장한 물리(Kudryavtsev류)로 전이 우위 강화.
2. 트랙 M 재실행: GTNPenv 심부 라벨 AOA 게이팅 후 Lena 대규모 라벨만 증강 → 전이 재평가.
3. 트랙 α 공간블록+LORO 재평가 + field-ALT 유계 클립.
4. 물리-ML 결합 재설계: 잔차학습 대신 물리를 사전지식(제약/입력 피처)으로 넣는 방식 비교.
