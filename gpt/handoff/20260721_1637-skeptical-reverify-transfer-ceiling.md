# Handoff: 회의적 재검증 다사이클 — 전이 상한·정보병목·연구목적 재확정

**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-07-21 16:37
**Session focus**: P0/P1/PPT 반영 후, 기존 결론들을 적대적으로 재검증해 다수 헤드라인을 정정하고, 라벨 없는 OOD 전이의 근본 한계를 확정하며, 연구목적을 "정확도·차별성"으로 재정렬.
**Author**: Claude Opus 4.8 (1M) + 01willy

---

## 1. TL;DR (≤5 bullets)
- **전이는 모델 구조로 못 뚫는다(확정)**: 물리 상수 E(a+E√TDD) LORO 18.24cm가 전이 최선. 순수 ML 40cm·잔차학습 48cm·토양 E(x)·미분물리층(구조 C) 모두 악화. 모든 모델의 레나 skill 음수(관측 평균보다 나쁨).
- **공변량 추가는 내삽 이득·전이 손실**: SoilGrids 토양은 전지구 커버라 결측 없는데도 전이 붕괴(레나 22→62.6cm) = 진짜 covariate shift. 정보병목이 전이 병목의 근본.
- **회의적 검증이 헤드라인을 반복 정정**: (a) 3D "지형+CCI 심부 개선"=site-GKF 누설 착시(기각), (b) 알래스카 내부 "증강 유의 개선"=test 인접 특징복제 누설+seed 운(기각), (c) 라벨 증강 "해가 된다"=심부라벨×결측 공선 결합만(부분기각).
- **연구목적 재확정(사용자 지시)**: "정직함"은 헤드라인 아님. 목적 = 기존 논문 대비 새롭고, 더 많고 적절한 데이터+증강+좋은 DL 비교로 **ALT(2D)·4D(timelapse)·얕은 3D를 정확하게** 예측. 불확실성은 도구.
- **다음 방향(사용자 제안)**: 전이 포기 → **전 지역 pooled 학습**(train/val/test 혼합) + **InSAR 30m 제대로 활용**(스칼라 증류 탈피, 면적 검증) + **물리+ML fine-tune**. 신규 KPDC(콘슬 8층 토양온도·Thaw depth 실측) 파싱.

---

## 2. Context
- 직전 핸드오프: [20260714_1139-ppt-rebuild-research-direction-audit.md](20260714_1139-ppt-rebuild-research-direction-audit.md) — 중간보고 PPT v3 + 연구방향 검증. 다음으로 P0→P1→P2 제시.
- 본 세션: 그 계획(P0/P1/P2)을 실행하고, **모든 헤드라인을 적대적으로 재검증**하는 원칙을 세워 다수 결론을 정정. P2에서 "물리 우선 전이"를 발견한 뒤, 물리+ML 결합·토양 E(x)·증강으로 전이를 개선하려는 여러 시도가 전부 실패함을 확인. 최종적으로 사용자가 연구목적을 정확도·차별성으로 재정렬.

## 3. What we did

### P0/P1 + PPT (`docs/EXPERIMENT_P0_P1_RESULTS_2026-07-14.md`)
- **Action**: 인벤토리 세계지도·6모델 오차지도, 다지역 셀 v2 조립(레나 3,037·GTNPenv 37·QTP 1), 통합 토너먼트.
- **Files**: `scripts/4_visualization/map_{data_inventory_world,tournament_error_maps}.py`, `scripts/1_data_prep/{parse_allena,parse_qtec,derive_alt_gtnp_envelope,assemble_cell_v2}.py`, `scripts/3_deep_learning/unified_tournament_cell.py`, `deck/build_{midreport,summary}.py`.
- **Result**: 위치가중 GBM 16.1≈Diffusion 16.2≈앙상블 16.2cm(동률). 통합학습 게이트 미채택(결측 라우팅). 중간보고 21p + 슬림 11p 덱.

### P2 3트랙 (`docs/EXPERIMENT_P2_RESULTS_2026-07-14.md`)
- **Files**: `scripts/3_deep_learning/p2_{augment,field,stefan}_experiment.py`.
- **Result**: Stefan 물리 LORO 18.2cm ≫ 순수 ML 40.6cm. 잔차학습 REJECT. → "물리 우선 전이" 발견.

### Phase 1 회의적 재검증 (`docs/EXPERIMENT_PHASE1_2026-07-20.md`)
- **Files**: `scripts/3_deep_learning/{aug_backbone_dissect,field3d_reeval}.py`.
- **Result**: 증강 "해가 된다" 부분기각(심부 GTNPenv 라벨×결측 모달리티 완전공선 결합만 붕괴, 레나 22→88cm; 물리·기후만 ML 면역). 3D "지형+CCI 심부 개선" 기각(site-GKF 누설, 72.6% 사이트가 같은 0.5°블록; 누설통제 시 LORO 1.60→1.73°C 악화).

### W2.1 SoilGrids + KPDC (`docs/EXPERIMENT_W21_KPDC_2026-07-20.md`)
- **Files**: `scripts/1_data_prep/enrich_soilgrids_wcs.py`, `scripts/3_deep_learning/soil_ablation_gate.py`, `scripts/1_data_prep/parse_kpdc_met.py`, `scripts/2_evaluation/kpdc_era5_validation.py`.
- **Result**: 토양 게이트 미채택(내삽 +5.6%·전이 −63.8%, 레나 62.6cm = covariate shift). KPDC 콘슬: ERA5 √TDD 실측 정합(bias ~0.1), 단일 E Stefan 콘슬 1.7배 과대예측(E(x) 동기, 단 in-domain·평균회귀). SoilGrids는 VRT 정체로 **WCS(ISRIC maps.isric.org) 우회** 취득.

### W3 물리결합 엔진 (`docs/EXPERIMENT_W3_2026-07-20.md`)
- **Files**: `scripts/3_deep_learning/w3_physics_ml.py`. **GPU**: 8, ~수분.
- **Result**: 가설 기각. PHYS_const 18.24cm 최선. PHYS_soil 19.99·PHYS_nn(미분물리층) 25.72 악화. 물리 형태강제 순효과 음수. 모든 모델 레나 skill 음수.

### 알래스카 내부 3트랙 + 적대적 검증 정정 (`docs/RESULTS_SUMMARY_2026-07-20.md`)
- **Files**: `scripts/3_deep_learning/{aug_within_alaska,timelapse_alaska,shallow3d_alaska}.py`. **GPU**: 8,9.
- **Result**:
  - 증강 1차 "4모델 유의 개선" → **적대적 검증에서 기각**(GBM=test 인접 특징복제 누설 donor 제거 시 14.2→16.0; MLP=seed 운, 블록부트스트랩 CI 0 포함). 살아남음: MLP>GBM ≈−0.7cm(3-seed), Stefan 유사라벨 placebo 대비 정보성.
  - timelapse: 연별 지도 물리 forcing 최선(연도홀드아웃 14.97cm), 연도 anomaly 예측 불가(corr 0.06). GIF `outputs/animations/timelapse_alt_alaska.gif`.
  - 얕은 3D: 알래스카 0-3m 실측 764행, 필드 2.66°C·R² 0.47, 0°C→ALT r 0.28(심부 0.16 대비 개선, 절대정합 미완).

### 비판적 검토·연구목적 교정 (`docs/CRITICAL_REVIEW_2026-07-20.md`)
- 점 검증 대표성 잡음(~12cm) 상한, InSAR 스칼라 증류·미활용, KPDC 검증만 사용. 연구목적을 정확도·차별성으로 재정렬(사용자).

## 4. Key numbers (this session)

| Method | Domain/Case | Metric | Value | Source artifact |
|---|---|---|---|---|
| PHYS_const (a+E√TDD) | LORO 전이 | RMSE / skill | 18.24 / +0.052 | `w3_physics_ml_results.csv` |
| PHYS_soil (토양 E(x)) | LORO 전이 | RMSE / skill | 19.99 / −0.039 | `w3_physics_ml_results.csv` |
| PHYS_nn (미분물리층) | LORO 전이 | RMSE / skill | 25.72 / −0.34 | `w3_physics_ml_results.csv` |
| ML_pure_gbm | 레나 전이 | RMSE / skill | 65.13 / −2.09 | `w3_physics_ml_perregion.csv` |
| GBM +토양 | 공간블록(내삽) / LORO | skill | +5.6% / −63.8% | `soil_ablation_gate.csv` |
| GBM +토양 | 레나 전이 | RMSE | 62.6 | `soil_ablation_gate_loro.csv` |
| MLP (real_only) | 알래스카 내부 공간블록 | RMSE | ~13.98 (3-seed 견고) | `aug_within_alaska_results.csv` |
| 증강 개선(GBM) | 알래스카 내부 | 판정 | 기각(누설·seed) | `aug_within_alaska_sweep.csv` + 세션검증 |
| STEFAN | 연도 홀드아웃(timelapse) | RMSE / R² | 14.97 / 0.34 | `timelapse_alaska_results.csv` |
| STEFAN | within-site anomaly | corr | +0.06 (예측 불가) | `timelapse_alaska_results.csv` |
| 얕은 3D GBM(mono) | 알래스카 0-3m 필드 | RMSE / R² | 2.66 / 0.47 | `shallow3d_alaska_results.csv` |
| 얕은 3D | 0°C→ALT 정합 | r / RMSE | 0.28 / 41cm | `shallow3d_alaska_altmatch.csv` |
| KPDC 콘슬/c1 | ERA5 √TDD 검증 | bias | ~0.1 | `kpdc_era5_validation.csv` |

주: 증강 적대검증 재실행 수치(donor 제거 15.99, 블록부트스트랩 CI [−0.53,+0.32], fresh seed 14.45)는 세션 내 계산으로만 존재, 저장 CSV 없음 → 재실험 시 산출물 저장 예정.

## 5. Decisions made
- **전이(OOD) 모델 구조 실험 종결**: 잔차·파라미터예측·미분물리층·토양 E(x) 전부 실패. 더 시도하지 않음(negative 확정).
- **연구목적 재정렬**: 정확한 ALT(2D)·4D·얕은3D 예측 + 데이터·증강·DL 비교 차별성. 불확실성/정직성은 도구이지 헤드라인 아님.
- **KPDC 위상 정정**: 과학 엔진은 자체 대형데이터(ABoVE 22만·InSAR 6.9GB·PolSAR 7GB). KPDC는 대회 규칙 충족 + 검증·소규모 라벨·물리 E 실측 보조.
- **SoilGrids 취득 경로**: VRT/vsicurl 정체 → WCS(GetCoverage, IGH 좌표, SCALESIZE ~5km) 확정.

## 6. Open questions / blockers
- **점 검증 상한 돌파 가능한가**: 단일프로브 대표성 잡음 ~12cm가 오차 바닥. 다중프로브 셀평균/InSAR 면적 검증으로 실제 내려가는지 미검증. → 면적 검증 실험이 해결 증거.
- **pooled 학습이 전이 실패를 우회하나**: 전 지역 혼합 train/val/test로 각 지역에 라벨을 주면 예측이 실제로 좋아지는지(범북극 제품 방식). P1 unified는 LORO로만 채점 → pooled-block 채점 재실행 필요.
- **InSAR 30m 제대로 쓰면 정보병목 뚫리나**: 스칼라 증류 대신 30m 레이어·면적 타깃으로 쓸 때 skill 상승 여부.
- **물리+ML fine-tune**: 알래스카 pretrain → 레나 소량 라벨(3,037)로 E 미세조정이 레나 in-domain을 개선하나.

## 7. Next steps (prioritized)
1. **신규 KPDC 파싱·통합** — owner: Claude, ~반나절. 콘슬 8층 토양온도(L1-L8)·VWC·CO2/CH4, 2016 Thaw depth 실측(ALT 직접 라벨), 쿠가록 화재/비화재. 대회 규칙 충족 + 얕은3D 검증 + 물리 E 실측. 위치 `kpdc/`(gitignore됨, 로컬만).
2. **전 지역 pooled 학습 실험** — owner: Claude, GPU. 알래스카+시베리아+티베트 혼합, 공간블록 train/val/test, 지역별 예측력. 지역 임베딩·MoE·multi-task 비교. (전이 아님 = 실용 배포 성능)
3. **면적 검증 실험** — owner: Claude. ABoVE 다중프로브 셀평균 + ReSALT InSAR를 타깃/검증으로. 점 vs 면적 오차 비교로 ~12cm 상한 돌파 확인.
4. **InSAR 30m 제대로 활용** — owner: Claude. 스칼라 증류 탈피, 30m 레이어·면적 타깃. 정보병목 헤드룸 검정.
5. **증강 재실험(엄밀판)** — owner: Claude. 거리버퍼(test 반경 0.25° pseudo 제거)·블록부트스트랩·multi-seed·nested 선택. 재실행 수치 CSV 저장.
6. **물리+ML fine-tune** — owner: Claude. 알래스카 pretrain → 레나 소량 라벨로 E 미세조정.

## 8. Pointers
- Authoritative reports: `docs/RESULTS_SUMMARY_2026-07-20.md`, `docs/EXPERIMENT_{P0_P1,P2,PHASE1,W21_KPDC,W3}_*.md`, `docs/CRITICAL_REVIEW_2026-07-20.md`, `docs/RESEARCH_PROGRAM_2026-07-17.md`, `docs/EVAL_FRAMING_NOTE.md`.
- 롤링 스냅샷: `SESSION_HANDOFF.md`(최신 섹션이 본 세션).
- Active jobs: none (GPU 8,9는 세션 중 사용, 현재 우리 프로세스 없음).
- 결과 CSV: `data/processed/{unified_tournament,soil_ablation_gate,w3_physics_ml,timelapse_alaska,shallow3d_alaska,aug_within_alaska,aug_backbone,field3d_reeval,kpdc_*}*.csv` (전부 git 추적).
- 신규 KPDC 원본: `kpdc/`(35파일 16MB, gitignore됨 = 로컬만, 커밋 안 됨).
- 커밋: `63fdc11`(scripts) `a1fbc53`(docs) `6c5880d`(results) `83a036f`(viz), pushed origin/main.

## 9. Caveats for GPT
- **"16.95cm 전이"는 오표기였다** → 실제는 알래스카 내 공간블록 CV. 진짜 다지역 전이는 물리 18cm·ML 40cm. `docs/EVAL_FRAMING_NOTE.md` 참조. 헤드라인은 항상 "공간블록(내삽)"과 "LORO(전이)"를 구분 표기.
- **"증강이 알래스카 내부 정확도 개선"은 기각됨**(누설·seed). 이전 세션 요약에 남아 있으면 무시. 재실험(엄밀판) 전까지 미확정.
- **3D 심부(5-20m) 지도는 스위스 편중 외삽**이라 알래스카 심부로 신뢰 못 함. 얕은(0-3m) 알래스카 실측판만 유효.
- **"정직함이 강점" 프레이밍 폐기**(사용자 지시). 목적은 정확도·차별성. UQ/AOA는 도구.
- Canonical: 셀 데이터 최신 = `dl_dataset_cell_v3_soil.csv`(v2+토양9, gitignore됨). 물리 = Stefan 아핀 `ALT=a+E√TDD`(단일 파라미터 아님, 2-파라미터 아핀). 전이 지표는 공간블록≠LORO.
- SoilGrids는 WCS로만 취득 가능(VRT 정체). Sentinel·AlphaEarth는 GEE 미설치로 미취득.
