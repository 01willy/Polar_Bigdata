# Handoff: 전면 재설정 전 상태 스냅샷(최종 묶음 XA–XM 판정, 원고·보고 덱·발표, 정리 감사)

**Project**: Polar Bigdata (영구동토 활동층 두께(ALT) 예측, Stefan 물리식과 기계학습의 결합)
**Date**: 2026-10-08 16:37
**Session focus**: 사전 등록한 최종 추가 실험 묶음 XA–XM 을 실행·판정하고, 원고·보고 덱·발표 대본을 만들어 10-06 에 발표했다. 10-08 극지연(KOPRI) 회의 뒤 다음 세션의 전면 재설정에 대비해 상태를 정리했다.
**Author**: Claude Opus 5.5 + 01willy

---

## 1. TL;DR

- **최종 묶음 완료**: 최종 추가 실험 묶음 XA–XM(사전 등록 10-04, 실행·열람 10-05)을 모두 실행하고 판정을 기록했다.
  - 남은 것은 마감 항목 둘이다. XE 2단계(위성·토양 고해상 입력)와 XF·R4(새 독립 지역, 마감 10-11 14:22)다.
  - XE 는 사용자 결정으로 마감 시점 상태만 기록했다. 특징 표 최종판 생성은 일정 이탈로 처리했다.
- **결론의 뼈대**:
  - 라벨 0개: 원천 계수 Stefan 이 기준이다. 직접 ML 은 30개 대상 가운데 18개에서 2 cm 넘게 나빠졌고, 물리 유사라벨 증강은 2개였다.
  - 라벨 10개: 수축 재보정 Stefan(κ 10)이 −2.45 cm 다.
  - 라벨 40·160개: 교차검증 방법 선택이 재보정 Stefan 보다 −1.35·−1.51 cm 다(XC, 레나·캐나다, 같은 지역 재분할).
  - 알래스카 지역 안: 이득은 약 0.5 cm 이고, 기후 격자 단위 보정에서 온다.
- **바뀐 주장(10-02 핸드오프 대비)**:
  - 'ML 이득은 물리 계수 오차에 비례': XA 에서 확인하지 못했다.
  - '규칙 W 라벨 40개 이득': XC-1b 에서 미결정이다.
  - '어느 대상에서도 악화 없음': 레나는 +1.9–2.2 cm 나빠졌다.
  - '기후 외삽 판정 불가': XI-c 지지(−2.44 cm, 같은 시기 공간 대용)로 바뀌었다.
  - '공변량 분산 배치': XD-1 기각이고, Algorithm P 는 무작위로 귀결되었다.
- **산출물과 감사**: 원고(영문 본문 4,447단어, SI, 국문), 보고 덱 85쪽, 발표 대본 PDF 를 만들어 10-06 에 발표했다. 10-08 정리 감사에서 덱·대본 문장 4종이 등록 판정보다 넓다는 것을 찾았다(`docs/AUDIT_2026-10-08_CLEANUP.md`).
  - 관측 위치 처방
  - '라벨 수가 어떻든 우리 방법이 낫다'
  - '손해 없음'
  - 덱과 원고의 제목 불일치
- **다음 세션 계획(사용자)**: 극지연 회의 피드백을 반영해 연구 목표와 논문 목표를 다시 정한다. 실험 입력 관리 체계를 정비하고, 필요하면 전체 실험을 다시 한다. 회의 피드백 내용은 아직 받지 않았다.

## 2. Context

- **직전 핸드오프**: `gpt/handoff/20261002_1033-workflow-framing-lgf-wf-verdicts.md`(10-02)
  - 연구 주제를 '라벨 수별 물리 기반 ML 워크플로'로 다시 세웠다.
  - LG 계열·LGF·WF 1–3차 판정을 마치고 원고 재구성을 다음 작업으로 두었다.
- **이번 세션의 동기**: 원고 재구성 중 사용자 질문과 QA 문서가 주장의 약점을 짚었다(`docs/QA_FOLLOWUP_2026-10-04.md`).
  - 이득이 계수 오차에 비례하는가
  - 워크플로 전체가 고정 레시피보다 나은가
  - 배치 규칙이 지역을 넘는가
  - 격자 안 상세도가 있는가
  - 기존 제품보다 나은가
  - 이에 답하려고 최종 추가 실험 묶음을 사전 등록하고 실행했다.
- **발표와 회의**:
  - 10-06: 연구실·지도교수 보고 발표(25분)
  - 10-08: 극지연(KOPRI) 측 회의
  - 회의 뒤 사용자가 전면 재설정을 결정했다.

## 3. What we did

**1. 최종 추가 실험 묶음 사전 등록**

- 내용:
  - XA–XJ 를 등록했다(개정 1 = T0).
  - 조사 보고서 6건과 질의응답 2건을 작성했다.
  - XK·XL(10-05 02:16)과 XM(10-05 13:23, 정정 14:09)을 추가 등록했다.
- 파일: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md`, `..._ADDENDUM_XK_XL_2026-10-05.md`, `..._ADDENDUM_XM_2026-10-05.md`
- 결과: 가설, 4분 판정 규칙(셀 가중·블록 등가중 CI, ±0.5 cm 동등), 사전 고정 해석 문장, 열람 순서, 마감을 고정했다.
- 커밋: 2678100, d721d24, ff7c97f, 31e19f7

**2. 하네스와 실행**

- 하네스: `scripts/3_deep_learning/xbatch_core.py`, `x_*.py`(실험 모듈 10개), `src/polar/cv_schemes`, 시험 `tests/test_x_*.py`
- 동결 모듈: h40, h41, h42, h54 는 고치지 않았다.
- 실행:
  - Rescale 작업 A(QoeTd, elm 96코어, 0.43 h, 약 1.40달러)에서 R0·R1a·R1b 를 돌렸다.
  - 나머지는 로컬 CPU(우리 작업 합계 32스레드 이하, nice 10)와 로컬 GPU 0·1 에서 돌렸다.
  - S9-DS 를 동결했다(f4a89ea).
- 재현 관문은 모두 통과했다.
  - XE 는 등록 수준 local_rescale 로 다시 채점해 35,442키가 통과했다.
  - XG g3 는 md5 4/4 를 다시 대조했다.

**3. 판정 기록**

- 위치: 계획서 8.1–8.12, 추가 등록 결과 절
- 판정은 4절 표와 1절 TL;DR 에 정리했다.
- 열람 사건: 10-05 17:20 에 XE xt2 표의 판정 수가 보조 에이전트 출력에 찍혔다. 값은 열람하지 않았고 '부분 비맹검' 표지를 달았다(2047e77).

**4. 원고**

- 위치: `paper/manuscript/{en,ko}`, 검토판 `main_review.pdf`
- 규모: 영문 본문 4,447단어(Sci Rep 한도 4,450), 영문 SI 76쪽, 국문 29쪽
- 그림: 그림 v3(Fig 1–7, SI S7–S14, 원고 번호 재배열), 1 km ALT 지도(알래스카, 레나)
- 남은 자리표시: [DECISION] 45, [PENDING] 12, [미확인] 8
- 커밋: 3390268, 0723491

**5. 보고 덱과 발표 대본**

- 덱: `deck/deck_spec_paper_report.json` → `deck/render/permafrost_paper_report.{pptx,pdf}`, 85쪽
  - 그림 양식: `design/style_tokens_v4.json`
  - 덱 빌더: `deck/build_paper_report.py`
  - 방법·결과 그림 생성: `scripts/4_visualization/method_figs_v4.py`
- 대본: `deck/script/` → `deck/render/발표대본_연구결과보고_2026-10-06.pdf`(34쪽)
- 산출물 허브: `deliverables/`(`tools/sync_deliverables.sh`)
- 커밋: 625d196 – 15bfcb0

**6. 정리 감사와 근거 표(10-08)**

- 감사 축: 수치, 프로토콜, 근거, 문서의 4축
- 결과 문서: `docs/AUDIT_2026-10-08_CLEANUP.md`
- 근거 표: `data/processed/xbatch/XG_product_comparison/alaska_label_cell_metrics_v1.csv`(`scripts/2_evaluation/xg_alaska_label_cell_metrics.py`)를 새로 만들었다. 셀 단위 파일에서 집계값만 냈다.
- 안내 문서 대체 표시와 정정을 했다.
- .gitignore 에 로컬 전용, 약관 보류, 미열람 봉인 표 규칙을 더했다.
- 커밋과 push: b3482cf, 1864799, 69bf0f2, a599be4. origin/main 과 일치함을 확인했다.

## 4. Key numbers (this session)

| Method | Domain/Case | Metric | Value | Source artifact path |
|---|---|---|---|---|
| 재보정 Stefan(P1, κ 10) − 원천 Stefan(P0) | 4지역 평균, 라벨 10개 | ΔRMSE (cm) | −2.45 | `data/processed/paper_figs/fig2_pool_curves.csv`(E1_P4_n_le_10, P1, n 10) |
| 워크플로(교차검증 방법 선택) − P1 | 레나·캐나다 풀, 같은 지역 재분할(seed 201–210), 라벨 10/40/160 | ΔRMSE (cm) | +0.08(동등) / −1.35 / −1.51 | `outputs/figures/paper/v3_restructure/Fig7_source_data.csv`(e), `data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_contrasts_main.csv` |
| 워크플로 − 무작위 배치·고정 레시피 | 같음, 라벨 40/160 | ΔRMSE (cm) | −0.86(미결정) / −0.93(우세, Holm p 0.040, 캐나다 −2.41) | `.../XC_workflow_end_to_end/sealed/xc_holm.csv`, 계획서 8.10 |
| 직접 ML(D0) 대 물리 유사라벨 증강(D1) | 30대상, 라벨 0개 | 2 cm 넘게 나빠진 대상 수 | 18/30 대 2/30 | `data/processed/wf/wf0_risk.csv` |
| 물리 유사라벨 − 섞은 유사라벨 | 라벨 0개 | ΔRMSE (cm) | −1.63 | `data/processed/paper_figs/fig3_a.csv` |
| CCI v5 − P0 | 라벨 0개(XG-1c) | ΔRMSE (cm) | +23.79 | `outputs/figures/paper/v3_restructure/Fig6_source_data.csv`(c) |
| R1 − 같은 라벨로 재보정한 CCI v5 | 라벨 10개(XG-2c) | ΔRMSE (cm) | −8.37 | `data/processed/xbatch/XG_product_comparison/sealed/xg_tests.csv` |
| 직접 ML / Stefan, 검증 사다리 | 3지역 평균(알래스카·레나·캐나다) | RMSE (cm), 무작위 → 지역 홀드아웃 | D0 14.81 → 28.71; Stefan 20.54–22.24 | `outputs/figures/paper/v3_restructure/Fig1_source_data.csv`(e) |
| R1 − P1 | 알래스카 지역 안, 라벨 1,000개/전량 | ΔRMSE (cm) | −0.54(RMSE 14.41 → 13.87, 두 가중 우세) / −0.52(RMSE 14.40 → 13.88) | `.../Fig7_source_data.csv`(a), `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`(WF6-a) |
| P1 / R1 블록 CV OOF | 알래스카 라벨 13,606셀 | r, R²(1 − MSE/Var), RMSE (cm) | 0.604, 0.323, 14.30 / 0.621, 0.370, 13.80(실측 SD 17.38) | `data/processed/xbatch/XG_product_comparison/alaska_label_cell_metrics_v1.csv` |
| 기존 제품(연도 정합) | 같은 셀 | r, RMSE (cm) | CCI v5 0.603, 74.91; Wei 2026 0.548, 17.19(13,605셀); Aalto 2018 −0.503, 43.60; Yi–Kimball 0.428, 61.04(10,606셀, 우리 0.508, 14.47) | 같은 파일(기간 평균 정의도 함께 있음: CCI v5 68.02, Wei 16.03, Yi–Kimball 60.56) |
| R1 − P0 | 독립 지역(약관 확인분, L8e PE1) | ΔRMSE (cm) | −3.19; 직접 ML 개선 0/5 지역 | `paper/claims/SI_learners_new_regions/tables/lgd_tests_lic.csv` |
| R1(xw) − R1(x25), XM-a | 알래스카·레나·캐나다, 격자 안, 전량 | ΔRMSE (cm) | −0.15 [−0.25, −0.05](부분 지지, 탐색); 레나 총 RMSE −1.50 | `data/processed/xbatch/XM_landcover_vegetation/sealed/xm_tests.csv` |
| 지온 유도 보조 라벨, XJ | 티베트, 라벨 3개, 재보정 Stefan 대비 | ΔRMSE (cm) | −83.17(201.05 → 117.89) | `data/processed/xbatch/XJ_tempderived_aux_labels/sealed/xj_tests.csv` |
| P0 / 직접 ML | 티베트(깊은 활동층 체계) | RMSE / ΔRMSE (cm) | 244.51 / −59.98 | `lgd_tests_lic.csv`(L38) |
| XI-c | 따뜻한 블록(같은 시기 공간 대용) | ΔRMSE (cm) | −2.44(지지) | `data/processed/xbatch/XI_climate_extrapolation_retest/sealed/xi_tests.csv` |
| XA-1·XA-2·XA-3 | 28대상 8계열, 이득 몫과 계수 오차의 순위상관 | ρ(셀/블록) | 0.37/−0.04, 0.18/0.01, 0.28/0.27, 모두 '확인하지 못함' | `data/processed/xbatch/XA_c2_gain_decomposition/sealed/xa_hyp.csv` |
| P1 − R1, XK | 알래스카, 격자 크기 1 km / 0.05° | ΔRMSE (cm) | 0.50 [0.18, 1.10] / 1.67 [0.60, 2.69] | `data/processed/xbatch/XK_support_scale/xk_support_scale_v1.csv` |
| 계층 conformal | 라벨 0개 90 % 구간 | q90, 포함률 | q 0.707(배수 2.03); 포함률 4대상 풀 0.86 [0.80, 0.92], 레나 단독 0.889 | `data/processed/map_lena/lena_pred_v1_meta.json`, `data/processed/h4/c2_coverage.csv` |
| 교차검증 방법 선택 비율 | WF4 선택 1,742건(라벨 10개 포함, 모드 x·i) | 비율 | P0 30.0 %, R1 λ1.0 26.6 %, P2 16.7 %, R1 λ0.25 11.7 %, D1 10.5 %, P1 2.8 %, R2 1.6 % | `results/rescale_wf/data/processed/wf/wf_meta.json`(choices "wf4\|W") |
| 지도 분산 | 알래스카 1 km 지도 | 0.1° 셀 사이 분산 비율 | 0.966 | `outputs/figures/paper/v3_restructure/maps/XL_figures_source_values.json` |

## 5. Decisions made

- **덱 제목(10-06, 사용자)**: '희소 관측 영구동토 지역의 활동층 두께 예측을 위한 물리 기반 기계학습 워크플로'. '라벨 수별', ':', 'Stefan' 을 넣지 않는다. 원고 제목은 아직 이전 판이다.
- **덱 구성(10-06, 사용자)**:
  - 결과 순서는 알래스카 성과 → 새 지역(적은 라벨) → 단계적 워크플로 → 지도 비교 → 결론이다.
  - 약점은 부록으로 옮겼다.
  - 대본은 사용자 세미나 대본 말투(짧은 평서문)로 썼다.
- **레나 라벨 수별 방법 지도는 쓰지 않음**: 1회 추출·1분할 결과가 등록 평균과 어긋난다. 파일은 커밋했고 설명문에 표시했다.
- **XE 2단계(10-08, 사용자)**: 마감 시점 상태만 기록했다(계획서 개정 이력 10-08 16:35).
  - 특징 표 최종판(마감 10-08 18:22)은 만들지 않았고 일정 이탈로 기록했다.
  - 동결 기준은 xt2 봉인 표를 열기 전에 정한다. 후보는 원자료 기준(O·V 포함)과 열 기준(H0·T2)이다.
- **git 정책(10-08)**:
  - 커밋한 것: 판정 표(`sealed/`)와 3 MB 이하 결과 표
  - git 밖에 둔 것:
    - 셀 단위 실측값·잔차 표(라벨 원천 약관 확인 전 배포 금지)
    - 대용량 재생성물
    - 미열람 봉인 표(`xe_r1b_xt2_*`)
- **다음 세션 방향(10-08, 사용자)**: 극지연 회의 피드백을 반영한다. 연구 목표와 논문 목표를 재설정하고, 실험 입력 관리를 정비하고, 필요하면 전체 실험을 다시 한다.

## 6. Open questions / blockers

- **극지연 회의 피드백 내용**: 다음 세션 시작 때 사용자에게 받는다. 재설정 방향이 이것에 달려 있다.
- **XE 처리**: 동결 기준, xt2 열람, 작업 R3(xh 적합), XE-a·XE-b·XE-a0 기록을 할지 정해야 한다. 재설정으로 묶음을 닫는다면 등록 이탈로 기록한다.
- **XF·R4(마감 10-11 14:22)**: 적격 새 지역은 0곳이다. 등록 규칙대로라면 XF-1–3 과 XC-F3 를 '시험하지 못함'으로 적고 R4 를 실행한다. 다만 R4 는 NAtlantic v5 민감도와 XB 확장이다.
- **원고**:
  - 제목을 덱과 맞출지 정해야 한다.
  - [DECISION] 45, [PENDING] 12, [미확인] 8 이 남아 있다.
  - 저자·사사·LLM 사용 고지가 필요하다.
  - 감사 3.2절의 정의·표기 항목을 반영해야 한다.
- **라벨 약관**: LGD 의 약관 미확인 원천(CUSP, GGD353, calm_web_subsites)은 제공자 확인이 필요하다. 셀 단위 표를 공개하려면 이 확인이 선행되어야 한다.
- **입력 관리**: 입력 표가 여러 판으로 흩어져 있다.
  - 라벨·공변량: `fidelity_base_v3.csv`, `fidelity_base_v4.csv`, `fidelity_base_v4_labels.csv`
  - 토양 도일: `e5_soil_tdd_v3.csv`, `e5_soil_tdd_v4.csv`
  - 확장 공변량·고해상 특징: `covariates_ext_v1.csv`, `xe_feat_v1.csv`
  - 지도 격자: `map_alaska`, `map_lena`
  - 재실험 전에 판·출처·약관·해시를 한 목록(manifest)으로 묶는 설계가 필요하다.

## 7. Next steps (prioritized)

1. **회의 피드백 문서화**
   - 담당: 사용자가 내용을 주고 Claude 가 정리한다.
   - 예상 시간: 0.5 h
   - 산출: `docs/` 에 회의 피드백 문서. 항목마다 현재 주장(C1–C8)·판정·그림과의 관계를 표시한다.
2. **연구·논문 목표 재설정안**
   - 담당: Claude 초안, GPT 비판, 사용자 결정
   - 예상 시간: 1–2 h
   - 산출: 유지·수정·폐기 주장 표(근거는 4절 표와 계획서 8절), 새 질문 목록, 새 목표에 맞는 그림 구성
   - 선행: 1번
3. **실험 입력 관리 체계**
   - 담당: Claude
   - 예상 시간: 2–4 h
   - 산출: 입력 표 인벤토리(경로, 판, 행 수, 지역, 원천, 약관, sha256, 생성 스크립트), 단일 입력 manifest, 입력 판이 바뀌면 하네스가 거부하는 점검
   - 선행: 2번의 범위 결정
4. **열린 마감 처리 결정**
   - 담당: 사용자 결정, Claude 기록
   - 예상 시간: 0.5 h
   - 내용: XE(동결 기준, xt2, R3)와 XF·R4(10-11)
   - 재설정으로 실험을 다시 하면 묶음 종료(등록 이탈)를 기록하는 쪽이 단순하다.
5. **재실험 사전 등록과 실행(필요 시)**
   - 담당: Claude 실행, GPT 설계 검토
   - 내용:
     - 새 목표로 가설·판정 규칙·분할·라벨 격자를 등록한다.
     - 하네스는 h40·h54·xbatch_core 를 재사용하고 수정 전에 동결 해시를 다시 적는다.
     - 계산 자원은 로컬 GPU(10-08 16:30 기준 5–9 비어 있음, 사용 전 `nvidia-smi`), 로컬 CPU 32스레드·nice 10 이하, 필요 시 Rescale elm 이다.
6. **덱·원고 문장 정정**
   - 담당: Claude
   - 내용: 감사 3.1절을 정정한다.
     - 관측 위치 처방 삭제
     - 라벨 수 구간별 판정 단어 표기
     - 제목 통일
     - 제품 값 정의 표기
   - 정정 전에는 10-06 덱을 외부에 돌리지 않는다.

## 8. Pointers

- **상태 정본**:
  - `SESSION_HANDOFF.md` 10-08 절
  - `docs/EXPERIMENT_LOG.md` 10-08 항목
  - `docs/AUDIT_2026-10-08_CLEANUP.md`
  - 안내 `00_START_HERE.md`
- **판정 기록**:
  - `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 8.1–8.12 와 개정 이력
  - `docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md`, `docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md` 결과 절
  - 이전 계열: LG `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 7절, WF `docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 5절
- **원고**: `paper/manuscript/en`(main.tex, si_main.tex), `paper/manuscript/ko`. 등록부 `paper/registry/experiments.csv`, 주장별 근거 `paper/claims/`
- **덱·대본**: `deck/deck_spec_paper_report.json`, `deck/build_paper_report.py`, `deck/script/`. 렌더는 `deck/render/`(git 밖)이고, 복사본은 `deliverables/01_발표자료/`
- **그림**: `outputs/figures/paper/v3_restructure/`(Fig1–7, si/, maps/, *_source_data.csv), 생성 코드 `scripts/4_visualization/method_figs_v4.py`, 양식 `design/style_tokens_v4.json`
- **실행 중인 작업**: 없음
- **체크포인트**: 해당 없음(트리 모형은 실행 때 다시 적합. `*.pt`·`*.cbm` 은 git 밖)
- **직전 핸드오프**: `gpt/handoff/20261002_1033-workflow-framing-lgf-wf-verdicts.md`, `gpt/handoff/20260926_2130-final-experiments-audit-figures.md`

## 9. Caveats for GPT

- **10-02 핸드오프에서 다시 쓰지 말 주장**: '이득은 계수 오차에 비례(0.66)', '규칙 W 가 라벨 40·160개에서 약 1 cm', '어느 대상에서도 악화 없음', '기후 외삽 판정 불가', '공변량 분산 배치 권고'. 최종 묶음 판정이 우선한다(1절).
- **관측 위치 문장**: '어디를 재야 하는지 제시', '관측 우선순위', '구간이 넓은 곳에 다음 관측' 같은 문장을 쓰지 않는다(WRAPUP 11.2). Algorithm P 는 무작위로 귀결되었고, 분산 기반·범위 밖 우선 배치는 무작위보다 나빴다. 불확실성 지도는 '예측을 덜 믿을 곳' 표시일 뿐이다.
- **포함률 0.86**: 4개 대상(러시아 W·E, 캐나다, 레나) 풀의 계층 conformal 값이다. 레나 단독은 0.889 다. 레나 지도의 90 % 구간 폭은 예측값에 비례하고(배수 2.03), 구간 폭이 오차 위험의 순위를 뜻하지는 않는다.
- **비교의 분할 집합**: 워크플로 −1.51 cm(XC, seed 201–210)와 고정 레시피 −0.64 cm(LG, seed 1–5)는 분할 집합이 다르다. 같은 분할 비교가 아니므로 나란히 두지 않는다. XC 안의 고정 레시피 대비는 −0.49·−0.58 이다.
- **제품 비교 정의**: 덱은 연도 정합 값, SI·XK 는 기간 평균 값이다. CCI v5 의 r(0.60–0.61)은 우리 값(0.62)과 가까워 '상관이 더 높다'고 쓰지 않는다. 8.4 cm 는 같은 라벨로 재보정한 CCI v5 대비이며, 다른 제품으로 일반화하지 않는다(Wei 는 미결정).
- **지역 수**: 검증 사다리는 그림·덱이 3지역 평균, 원고 본문이 5지역 평균이다. 홀드아웃 지역 수는 덱 7, 초록 5 다.
- **명칭**:
  - 방법 기호: P0(원천 계수 Stefan), P1(재보정 Stefan, 수축 κ 10), R1(물리 잔차 결합, λ), D0(직접 ML), D1(물리 유사라벨 증강), W(교차검증 방법 선택)
  - 최종 묶음의 배치 이름 S1(= 무작위)은 7월 단계 이름 S1(기준선 7모델)과 다르다.
  - 국문 용어는 '활동층'이다. README 의 '활성층'은 옛 표기다.
  - 덱 용어는 '단계적 예측 워크플로', '교차검증 방법 선택', '학습 지역'이다.
- **자료 위치**: 셀 단위 실측값·잔차 표와 대용량 자료는 git 에 없다.
  - 대용량 경로: `data/processed/map_alaska` 842 MB, `lgx` 344 MB, `lgu` 259 MB, `lgt` 138 MB, `shards/`, `results/rescale_*/data`
  - `xe_r1b_xt2_*` 봉인 표는 동결 기준을 정하기 전에 열지 않는다.
- **용어 정본**: 프로젝트 설정 `.claude/project.yaml` 의 naming_canonical 첫 블록(10-08 갱신)을 따른다.
