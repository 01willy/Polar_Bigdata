# 세부 연구 계획 — Source-aware Multi-fidelity ALT (2026-07-22)

GPT 로드맵([gpt/ALT_multifidelity_research_roadmap_for_claude_code.md](../gpt/ALT_multifidelity_research_roadmap_for_claude_code.md))을
우리 실제 자산·기존 결과·대회 마감(2026-07-31)에 매핑한 세부 계획이다. 사용자 원래 뼈대(3종 ALT 증강 →
DL/ML 예측 + timelapse + 얕은 온도장, 물리식·증강비율·DL기법 변주 + 대회용 비교분석·교차검증 + 정확도·모델
고도화)를 중심축으로 유지한다. 초안을 실측 overlap 계산·문헌 novelty 검증·적대 검증으로 세 번 거른 확정본이다.

관련: [gpt/ALT_multifidelity_...](../gpt/ALT_multifidelity_research_roadmap_for_claude_code.md)(원 로드맵),
[EXPERIMENT_PLAN_2026-07-21.md](EXPERIMENT_PLAN_2026-07-21.md)(직전 계획), [CRITICAL_REVIEW_2026-07-20.md](CRITICAL_REVIEW_2026-07-20.md).

---

## 1. 연구 재정의 (사용자 비전과 기존 결과의 화해)

핵심 전환: **증강을 "라벨 개수 늘리기"가 아니라 "자료원별 신뢰 구조 모델링"으로 승격한다.**

사용자 비전의 3종 ALT 증강((a)직접 탐침 F4, (b)지중온도장 유도 F3, (c)Stefan 물리유도 F1)을 그대로
유지하되, 이들을 동일 정답으로 병합(내가 기존에 기각한 것)하지 않고 공유 잠재 ALT `z=f(x)`에 대한
자료원별 관측모델 `y_s = A_s[z] + b_s(x) + ε_s`, `ε_s ~ N(0, σ_s²(x))`로 분리 추정한다. 그러면:

- 사용자의 증강 비전이 **살아난다**(source-aware multi-fidelity로).
- 내가 확인한 "단순 병합 실패"가 오히려 이 연구의 **핵심 발견**이 된다(RQ: 물리 pseudo-label이 OOD에서 언제 돕고 언제 편향을 증폭하는가).
- 사용자가 원한 "정확도·모델 고도화"는 in-domain 점 RMSE(대표성 하한 14cm에 막힘)가 아니라 **LORO 전이·leave-source-out·calibration** 축에서 추구한다(그 축에서만 구조적 우위 여지가 있다).

## 2. 실측으로 확인한 관문 (GPT 중단기준 §21 위험1)

GPT 로드맵은 "자료원 간 동일 site 중첩이 없으면 source bias·지역효과 분리 불가"를 중단 기준으로 둔다.
우리 데이터로 실제 계산한 결과(0.01deg 셀 기준):

| source 쌍 | overlap | 판정 |
|---|---|---|
| direct × Stefan(F1) | **100%** (17,422/17,423) | b_Stefan·σ_Stefan 완전 식별 |
| direct × CCI(F0) | **99.5%** | 준전지구, 여러 지역서 식별 |
| direct × InSAR(F0) | 79.3% | 알래스카·캐나다만(Lena 0) → region 교락 |
| direct × PolSAR(F0) | 57.8% | 알래스카만 |
| direct × 온도유도(F3) 셀 | 9~17셀뿐 | **희소.** 단 매치파일에 107~130 paired 관측 존재 |

**판정(부분 통과)**: Stefan·CCI는 full source-aware(A5) 가능. 온도유도(F3)는 pooled 스칼라 bias까지만(지역
조건부 편향은 식별 불가). InSAR/PolSAR bias는 알래스카 밖 결측이라 region과 교락. GTNPenv(37셀)는 direct
overlap 부재라 Stefan 브리지로만.

**중대한 제약(적대검증 지적)**: 깨끗이 식별되는 b_s는 Stefan·CCI 둘뿐인데, **둘 다 이미 51 feature에
컬럼으로 들어 있다**(e5_sqrt_tdd, cci_alt). 따라서 source-aware 헤드가 "특징 추가" 대비 갖는 구조적 증분은
가정이 아니라 실증 대상이다. A5(S6)를 헤드라인 정확도로 걸지 않는다.

## 3. 기존 논문 대비 차별성 (novelty 검증 결과)

**6요소 조합(source-aware 관측모델 + 조건부 편향·이분산 + support-aware + 검열 ALT + fidelity 위계·증강비율
반응곡선 + 다축 OOD)을 ALT 도메인에서 결합·실증한 발표 연구는 2026-07 검색 기준 없다.** 단 요소별 first-claim은
전부 금지한다(아래는 인접 선점).

- Stefan+ML 하이브리드 ALT: SCE 2025(Stefan-CatBoost 블렌딩, R² 0.873), Gautam 2025(Stefan vs RF 비교).
- 다소스 ALT 융합: GeoCryoAI 2025, Ran 2022(표준화 후 병합).
- source-aware 잠재변수: LVGP 2024, NCAM 2026(방법론 골격 근접). 검열 공간회귀: Tobit GP 2023.
- 물리 pretrain: Read 2019 PGDL. 공간지지-불확실성: Du 2025.

**방어 가능한 3문장(전부 "우리가 아는 한 ALT에서 최초"로 한정)**:
1. 직접·온도유도·물리유도 ALT를 동일 정답으로 병합하지 않고 자료원별 조건부 편향·이분산으로 분리 추정.
2. 탐침 한계·rock-hit·센서 간격 구간검열을 ALT 관측 우도에 반영(방법론이 아니라 ALT 적용이 신규).
3. 물리 pseudo-label의 조건부 유효성을 증강비율 반응곡선 × 다축 OOD로 정량화(선행은 고정 결합만).

**금지**: "최초 물리+ML ALT", "최초 다소스 융합", "source-aware 방법론 최초 제안", "검열 회귀 최초".

## 4. Input 설계 (GPT §7 스키마를 우리 데이터에 매핑)

현재 51 feature(지형6 dem_ · 기후8 e5_ · PolSAR3 · InSAR5 · CCI2 · 토양9 sg_ · 라벨통계)를 공변량 코어로
유지하고, 라벨 측에 fidelity 메타를 붙여 long-format으로 재구성한다.

- **샘플키**: 셀 학습 `loc_id × year`, 온도장 `site × datetime × depth`.
- **source_id**: F4_direct_ABoVE / F4_direct_Lena / F3_temp_shallow3d / F3_temp_field3d / F2_gtnp_env / F1_stefan / F0_insar / F0_polsar / F0_cci / F0_resalt.
- **fidelity_level**: F4>F3>F2>F1>F0 (절대순위 아닌 품질메타 동반).
- **spatial_support_scale_m**: 탐침 point ~0.3m, 셀 라벨 100m, CCI/ReSALT gridded는 화소크기 → A_s 연산자.
- **estimated_uncertainty**: σ_s prior 초기값(탐침 alt_sd, ReSALT alt_unc_cm≈31, 온도유도는 센서간격 기반, Stefan은 앙상블 분산).
- **검열 스키마**: right_censored/interval_lower/interval_upper/probe_limit_cm(shallow3d max_obs_depth_cm·alt_obs_near_maxdepth, censor_flag).
- **group 키(누설통제 핵심)**: cell_id + **0.5° 블록**(field3d site-GKF 누설 방지) + derived_from_observation_id(파생-원천 동일 fold).

## 5. 전역 통제 규약 (과거 붕괴 재발 방지, 예외 없음)

과거 헤드라인이 누설·seed 운으로 여러 번 붕괴했다. 모든 결론에 강제한다.

- **누설방지 pytest 먼저**: S0에서 unit test 스위트 구축(test site가 물리계수 보정 배제 / 파생-원천 동일 fold /
  scaler·imputer train-only fit / forecast origin 이후 feature 제거 / 0.5°블록 GroupKFold). **하나라도 실패 시 결과 무효.**
- **≥3 seed + 74블록 부트스트랩 CI + 거리버퍼**로 유의성 판정(CI가 0 포함하면 "개선 없음").
- **placebo 대조**: 증강 실험은 셔플라벨 placebo와 donor 제거 버전을 항상 병기.

## 6. 세부 실험 단계 (비판 반영 확정본)

GPT 모델 순서(실측-only → Stefan → physics feature → residual → pretrain-finetune → source-aware → support/censored
→ mixture-of-physics → joint)를 우리 스크립트 재활용 가능하게 배치. **순서 정정**: S2(Stefan 하한 게이트)를 S3
앞으로(S3 판정 기준인 18.24cm 하한·fold-safe E보정이 S2에서 확정되므로).

| ID | 단계 | effort | GPT 매핑 | 핵심 게이트(정직) |
|---|---|---|---|---|
| **S0** | Fidelity 메타 스키마 + overlap gate + 누설 pytest | S | §6·§7·§11 | overlap matrix 산출, pytest 전체 통과 |
| **S1** | 실측-only 부스팅 baseline(재검증) | S | A1·B0 | in-domain 16.95-18.31·LORO baseline 재현(기준선) |
| **S2** | Stefan 물리 baseline + physics feature | S | A2·§9 | **Stefan LORO ≤18.24cm 하한 게이트 확정** |
| **S3** | 증강비율 r 조건부 반응곡선(핵심 재검증) | M | RQ1·RQ3·§13 | 블록부트스트랩 CI + placebo 대조로 조건부 효과(방향성) |
| **S6** | Source-aware multi-fidelity(A5, Stefan+CCI 한정) | L | A5·§6.2 | naive pooling 대비 LORO 또는 cov90 우위(구조 증분 실증) |
| **S11** | 불확실성 지도(σ_s+conformal) + 증강방식 비교표 | M | RQ2·§15 | cov90 85-90% 근접, 다축 OOD 비교표 |
| S4 | 잔차학습 fallback(게이트식) | S | A4·H2 | LORO ≤18.24, 미달 즉시 중단 |
| S5 | 물리 pretrain → 실측 finetune | M | A3·§9.4 | LORO에서 S1 대비 개선/동률 |
| S7 | Support-aware + 검열/구간 우도(KPDC 앵커) | M | A7·RQ4 | 방법론·KPDC 활용 근거(정량 기여 얇음 명시) |
| S8 | Mixture-of-physics experts | M | A6·RQ3 | 게이트 가중치-오차 상관(진단) |
| S9 | Timelapse D(t) 계절궤적 + EOS 조기예측 | M | A8·§12 | 계절 궤적·EOS 시점(연 anomaly는 목표 아님) |
| S10 | 얕은 3D 온도장 별도/공동 헤드 | L | A8·§10.3 | 온도장 R² 0.47 재현(공동학습은 조건부) |

### 단계 상세(핵심 4개)

**S2 — Stefan 물리 baseline + physics feature**: e5_sqrt_tdd 기반 Stefan(E 계수 fold 내부 보정), physics ALT를
(i)독립 예측 (ii)GBM 입력특징으로. LORO ≤18.24cm 하한 게이트 확정. Stefan E 보정 site-year와 test fold 교집합
공집합 assert. 스크립트: `p2_stefan_experiment.py`·`w3_physics_ml.py` 재활용.

**S3 — 증강비율 반응곡선(핵심 재검증, 엄격 통제)**: LORO × r∈{0,.25,.5,1,2,5,10} × 가중{고정 w, σ_s 역가중,
b_s 보정후} 스윕. Δskill = f(r, |b_s|, σ_s, 환경거리, 물리앙상블 불일치)를 생태대·동토구분·토성별 분해.
**⚠️ 중대 통제(적대검증 지적)**: 기존 aug_within_alaska_sweep의 "14.96→14.26 개선" 수치는 **이미 특징복제
누설로 기각된 헤드라인**이다(donor 제거 시 14.22→15.99 역전). 이 수치를 근거로 재동원하지 않는다. 반드시
거리버퍼(test 반경 0.25° pseudo 제거)·블록부트스트랩 CI·placebo 셔플라벨 대조를 게이트에 내장. "한 조건대역에서
r>0 유의"는 다중비교로 우연 대역이 반드시 나오므로 게이트로 부적합 → **CI + placebo를 넘는 방향성 있는 반응곡선**만
성공으로. 편향증폭 구간도 결과로 보고. 스크립트: `aug_within_alaska.py`·`aug_backbone_dissect.py` 확장.

**S6 — Source-aware multi-fidelity(A5, 주 제안이나 헤드라인 아님)**: 공유 인코더 + source별 (b_s, log σ_s) 헤드,
Gaussian NLL 학습. 직접 source b=0 고정. **Stefan·CCI 2소스만 full A5**(overlap 100%·99.5%). 단 둘 다 이미
feature이므로 **Stefan-feature+CCI-feature baseline 대비 구조적 증분을 실증**(가정 금지). 온도유도는 pooled b_temp
까지만. GTNPenv 브리지는 순환 위험(Stefan이 앵커이자 브리지)이라 진단용으로만. 비교군: 실측-only GBM, naive
pooling GBM, 고정가중 증강 GBM(동일 공변량·fold 강제). **게이트를 "정확도 OR cov90 우위"로 두되, 정확도 미달 시
UQ 산출물로만 편입("DL 구조적 우위"로 과대포장 금지)**. 신규 `source_aware_mf.py`(torch, GPU 6-9).

**S11 — 불확실성 지도 + 비교분석표(가장 방어적)**: σ_s(x) 이분산 + conformal(기존 56→86% 경험 연장)로 cov90 보정
지도. 전 단계를 동일 공변량·fold로 비교표(RMSE·MAE·bias·cov90·LORO·leave-source-out) 집계. 선행 SCE(랜덤 10-fold)·
Gautam(랜덤 70/30) 대비 방법론 우위를 정직하게 문장화. `alt_conformal_*` 재활용.

## 7. 9일 현실 경로 (마감 2026-07-31)

**핵심 경로**: S0 → S1 → S2 → S3(엄격 통제) → S6(Stefan+CCI 한정) → S11.
나머지(S4·S5·S7·S8·S9·S10)는 시간이 남으면 착수하고, 아니면 **방법론 서술 + 부분 결과**로 보고서에 편입.

- 신규 스크립트(build_fidelity_schema.py, source_aware_mf.py, 검열 우도 헤드, conformal 집계, 누설 pytest)가
  전부 미존재 → S0에서 pytest·스키마부터. 이게 이후 모든 게이트 무결성의 전제.
- **서사 축(적대검증 권고)**: 헤드라인을 S6 정확도가 아니라 **S11(보정 UQ) + S2(물리 앵커 18.24) + S7(검열
  방법론 프레이밍) + S3(반응곡선 규명)**에 둔다. S6은 야심이나 증분 실증 실패 시 가장 취약.
- GPU: 실행 전 nvidia-smi 필수(최근 8,9만 사용 이력).

## 8. 대회 예선 보고서 산출물

1. **증강방식 비교분석표**: 실측-only vs naive pooling vs physics feature vs residual vs pretrain-finetune vs
   source-aware를 동일 공변량·fold에서 RMSE·MAE·bias·cov90·LORO·leave-source-out(선행 대비 방법론 우위 표).
2. **증강비율 반응곡선**: Δskill = f(r, bias, variance, 도메인거리, 물리 불일치)를 조건별 분해(돕는/해치는 조건).
3. **다축 OOD 결과**: 공간블록·LORO·시간·생태대·측정방법·source leave-out 6축 + 누설 pytest 통과 표.
4. **불확실성 지도**: 이분산 σ_s + conformal cov90 지도 + calibration 곡선.
5. **KPDC 앵커**: 콘슬 온도유도 ALT(F3)의 direct 대비 overlap 진단 + 구간검열 우도 + 현장 정합(ERA5 √TDD bias~0.1).
6. **timelapse D(t)**: 계절 궤적 + EOS 조기예측(연 anomaly는 데이터 한계 명시).
7. **얕은 3D 온도장**: R² 0.47 재현 + 0°C 등온선.

## 9. 재검증으로 여는 것 (사용자 요청, 누설통제 하)

기존 "기각" 결론을 확정으로 받지 않고 재검증 대상으로 연다.

- **물리 pseudo-label 증강 무익(P2)**: naive pooling 조건의 판정. S3에서 r 반응곡선 × 가중방식으로 재검증.
  단 14.26cm 수치는 누설 기각본이므로 근거로 쓰지 않고, 거리버퍼·placebo로 새로 산출.
- **다지역 통합 미채택(P1)**: +GTNPenv가 Alaska 개선하나 Lena bias +85.5cm 파탄. S6 source-aware b_s가 흡수하도록
  재개(단 Stefan 브리지 순환 주의, 진단용).
- **잔차학습 무익(48cm)**: covariate shift 심한 토양 입력 조건이었음. S4에서 shift-robust 입력·저용량·shrinkage로 게이트식.
- **DL은 GBM 대비 무익**: in-domain 점 RMSE(대표성 하한) 척도였음. S6 source-aware 구조는 GBM이 표현 못 함 → LORO·calibration 축에서 재검증.

## 10. 정직한 경계 (범위 한계)

- **Overlap 제약**: source×region 조건부 편향은 온도유도에서 식별 불가(paired 107~130쌍은 클러스터·소수). 깨끗한
  b_s는 Stefan·CCI뿐이고 둘 다 이미 feature. A5는 프레이밍으로 방어되나 실증 기반은 얇다.
- **KPDC 정량 근거**: F3 앵커 식별력은 shallow3d 알래스카 18쌍(corr 0.28) + 우측검열 1개에 의존. field3d 풀링쌍은
  심부판(corr 0.11~0.16, MAE 142~175cm)이라 ALT 레짐 밖. KPDC는 방법론 앵커로만 방어 가능 → 보고서 "활용 데이터"
  첫머리 배치 + 정량표로 실질 편입 문장 확보. Council 6주 관측을 연 사이클로 확대해석 금지.
- **정확도**: in-domain 점 RMSE는 대표성 하한 14cm에 막혀 어떤 고도화로도 못 내림. 정확도 우위는 LORO·source-out
  축에서만 시도(게이트 정직 유지).
- **9일**: S6·S7·S10은 L/M 규모라 전부 완성은 낙관 아님. 핵심 경로 외는 부분 결과·방법론 서술.

---

## 11. 보강 (2026-07-23) — 다양한 DL·물리식·지역전략·시각화 (사용자 4개 요구 반영)

최신 DL·물리식·지역전략·시각화를 SOTA 지향으로 조사·적대검증한 결과를 반영한 보강이다. 마감은 실제 8일(07-31).

### 11.1 지역 전략 확정 — 하이브리드 (택일 아님, 계층)

**정확도 헤드라인 = 알래스카 in-domain 트랙**(전 51 feature·SAR 전량 활용), **전이·차별성 = 전 지역 source-aware
pooled 트랙**(공유 feature + 물리 앵커). 순수 전역 assemble(공유 feature-only)은 미채택.

- **데이터 근거(SAR 수치 정정)**: 핵심 값 `insar_alt`·`polsar_alt` 기준 **알래스카 InSAR 100%·PolSAR 74%, Lena
  InSAR 0%·PolSAR 0%**다. (직전 세션에 언급한 Lena 17%·33%는 보조 컬럼 insar_dist·insar_miss가 값이 있어 부풀려진
  수치이며, SAR ALT 값 자체는 Lena에 전무하다.) 따라서 SAR 이득은 **구조적으로 알래스카에만 실현**된다.
- **왜 하이브리드인가**: (a) SOTA 정확도는 SAR 전량 쓰는 알래스카 in-domain에서 나온다(문헌 P-band PolSAR 업스케일
  RMSE 7~12cm). (b) 순수 전역 assemble은 SAR를 못 써 정보가 줄고(순환극 pooled RMSE ~87cm), P1에서 Lena bias
  +85.5cm 파탄이 확인됐다. (c) 전 지역은 물리 앵커·source-aware b_s(지역편향 흡수)·선별 증강으로 전이 축에만 편입.
- 공변량 이질성은 **드롭이 아니라 마스킹 + source-aware 헤드**로 처리. 정확도 헤드라인(AK)과 전이 신뢰성(물리+source-aware)을 분리 보고.

### 11.2 다양한 DL (사용자 요구 1) — 비판 반영 스코핑

**중요 정정(적대검증)**: in-domain 점 RMSE는 7모델 전부 CI [14.9, 23.0]cm 동률이다. 신규 모델도 이 밴드 재착지가
거의 확실하므로 **"모델을 늘려 in-domain SOTA 돌파"는 하지 않는다**(동률 반복). 모델 추가는 정확도 추격이 아니라
아래 목적별로만.

| 모델 | 역할 | effort | 판정 |
|---|---|---|---|
| LightGBM/CatBoost/XGBoost | 앙상블 코어·대조군 | 낮음(보유) | 유지 |
| **RealMLP-TD** (pytabkit) | 자체 MLP 대체, "DL은 튜닝부족이라 졌다" 반박 차단 | 낮음(0.5일) | **채택(공정성 목적)** |
| **TabPFN-2.5** (불가 시 v2) | S11 UQ의 base 예측분포(정확도 아님) | 낮음(0.5일) | **채택(UQ용)** |
| **greedy 앙상블**(Caruana OOF) | 이종 혼합 앙상블 코어(TabArena 근거) | 낮음(자체 50줄) | **채택(앙상블)** |
| TabM / FT-Transformer | 앙상블 다양성 멤버 | 낮음(보유) | 유지(멤버만) |
| ModernNCA | 회귀특화이나 이웃검색=공간누설 통로 | 중간(1일) | **보류**(AK 공간블록 한정 옵션) |
| AutoGluon extreme | "도달 상한" 참조로 비교표 1회 | 낮음 | 참조만 |
| Flow / Diffusion | 정확도·전이·UQ 열세 | - | **강등**(대조 사례로만) |

source-aware 백본은 TabM(실측 최하위)이 아니라 **공유 MLP/GBM 계열**로.

### 11.3 다양한 물리식 (사용자 요구 1) — Stefan 스프레드 확장

단일 Stefan(LORO 18.24)을 **물리 앙상블 스프레드**로 확장. `src/polar/physics/`에 모듈화. 각 물리 예측을
source-aware의 별도 F1 소스로 투입하고, 앙상블 불일치 `U_model`(phys_std)을 DL 입력 feature 겸 conformal UQ 스케일러로.

| 물리식 | 우리 데이터로 | 비고 |
|---|---|---|
| Stefan 기본 `E√TDD` | ✅ 전지역(e5_sqrt_tdd) | LORO 18.24 하한 게이트 기준 |
| Modified Stefan (edaphic E, 열물성) | ✅ (sg_ 텍스처→kt Johansen) | 저비용 다양성 |
| Kurylyk-Hayashi λ 보정 | ✅ 단 **λ 계수 unverified**(원문 확인 전 플래그) | 열용량 보정 |
| TTOP | ✅ (출력은 동토 마스크/MAGT) | ALT 직접출력 아님, 마스크·feature로 |
| Kudryavtsev (Ku) | ✅ (permamodel 이식, 식생감쇠는 프록시) | 눈·물성 민감도 다양성 |
| n-factor Stefan | △ **부분**(land-cover 클래스 부재→프록시/1.0 스윕) | CCI는 land-cover 아님 |
| ASM (2심도 온도) | △ 시계열 서브셋만(GTNPenv 등) | F3 앵커 독립검증 |
| 1-D enthalpy 수치해 | ❌ 예선 범위 초과(일별 강제·층서 필요) | 제외 |

산출: 물리별 LORO 표, 물리 스프레드 지도("어느 물리가 어느 지역서 강한가"), phys_mean/phys_std 컬럼.

### 11.4 input 최대 활용 (사용자 요구 4)

- **전량 활용(공변량 코어 45종)**: 지형 dem_ 6 · 기후 e5_ 8 · 토양 sg_ 9 · InSAR 6 · PolSAR 3 · CCI 2(+SAR는
  pooled서 마스킹+source-aware 헤드로 유지, 드롭 금지). lat/lon·region은 group 키·source 식별자로만(직접 feature 투입은 누설·과적합 주의).
- **제외(타깃 파생 누설 6종, S0 pytest로 강제)**: alt_sd·alt_min·alt_max·n_obs·n_years·year_min/max. SHAP 입력에서도 제외.
- **파생 추가**: phys_mean/phys_std(물리 앙상블), 함수비 θ·공극률(sg_ 페드로트랜스퍼), 눈깊이(e5_swe), 열전도비 rk,
  시계열 정적 descriptor(전이용). 옵션: AlphaEarth 임베딩(전지역 100%, GEE 필요), fold-safe 크리깅 lag(in-domain 한정).

### 11.5 시각화 (사용자 요구 2) — 표<그래프<지도<GIF

공통: `src/polar/plotstyle.use_polar()`, 냉색(ALT=oslo_r · 온도=vik 0중심 · 오차/불확실성=acton · 개선=broc 0중심,
붉은계열 금지). PNG 300dpi + PDF 동시, `outputs/figures/<단계>/`, AK는 EPSG:3338 재투영·스케일바·물리 aspect. GIF는
프레임 PNG 선저장 후 ffmpeg palettegen, vmin/vmax 전 프레임 고정.

- **S1**: ALT 예측지도(oslo_r) + 잔차지도(acton) 모델별 나란히 + Taylor diagram.
- **S2**: 물리 5종 소패널 지도 + phys_std 불일치 지도 + 물리별 LORO 잔차맵.
- **S3**: 증강비율 r 스윕 예측지도 GIF(RMSE 곡선 인셋) + Δskill 곡선(placebo·CI 밴드).
- **S6**: source별 b_s 그래프 + naive vs source-aware LORO 바 + SHAP 공간지도 + ALE에 Stefan √TDD 이론곡선 중첩.
- **S11**: conformal 전후 커버리지 곡선(56→86%) + 예측지도에 구간폭 상위 해칭 오버레이 + 지역별 조건부 커버리지.
- 여유 시: ESA CCI ALT 연도별 타임랩스 GIF, AoA/DI 신뢰범위 지도.
- 우선순위: S1/S2 지도·잔차·Taylor, S6 SHAP, S11 커버리지 곡선을 필수. CD diagram·타임랩스는 선택.

### 11.6 SOTA 정직한 도달선 (과대약속 금지)

1. **in-domain 점 RMSE**: 대표성 하한 14cm에 막힘. 신규 모델+앙상블로 1~2cm(unverified) 여지뿐. "점 RMSE SOTA 돌파" 약속 안 함.
2. **전이(LORO)**: 물리 앵커 다양화·source-aware b_s가 유일한 구조적 우위 여지. 단 S6 증분은 실증 대상, 미달 시 UQ로만.
3. **UQ(가장 방어적)**: conformal 56→86% 재현·연장으로 cov90 85~90% 근접이 정직한 목표.
4. **앙상블**: 이종 greedy 앙상블이 단일 모델 상회(TabArena 근거)는 신뢰. 잔차 blending은 P2 무익이라 배제.

**서사 축**: 헤드라인을 S6 정확도가 아니라 **S11(보정 UQ) + S2(물리 앵커 18.24) + S3(반응곡선) + AK floor 근접**에.
novelty는 2024-2026 표형식 SOTA(TabPFN·greedy)와 물리 5종 스프레드를 ALT에 결합·실증한 것(요소별 first-claim 금지).

### 11.7 기록 정책 (사용자 요구 2·3)

- 수치: `data/processed/<단계>_results.csv`(모델·seed별 RMSE·MAE·bias·cov90·LORO·source-out) + `_meta.json`(입력해시·feature목록·제외컬럼·fold·seed·GPU·커밋SHA). OOF는 `_oof.csv`. `/collect` 수집 가능 스키마.
- 그림: `outputs/figures/<단계>/` PNG+PDF, `figures/figure_spec.json` 갱신, GIF 프레임 보존.
- 물리: `src/polar/physics/` 각 모듈에 계수 출처·unverified 플래그 주석.
- 로그: 단계마다 `docs/EXPERIMENT_LOG.md` 1블록(가설·통제·게이트 통과여부·결론). 게이트 실패 시 "무효" 명기.
- **단계별 사용자 설명**: 각 단계 착수·완료 시 쉬운 설명 제공(사용자 요구 3).

### 11.8 스코핑 (적대검증 반영, 8일 현실)

**핵심 경로(반드시)**: S0 → S1 → S2 → S3 → **S11**. S6(source-aware, 2~3일 최대 리스크)는 UQ fallback을 사전
확약하고 S11 이후 여유분으로. ModernNCA·AutoGluon·크리깅·타임랩스 GIF는 여유 시 옵션. 시각화도 필수(S1/S2/S11)와
선택(CD·타임랩스)을 분리해 과부하 방지.
