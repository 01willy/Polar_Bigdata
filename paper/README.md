# 논문 색인(paper/)

**성격**: Scientific Reports 투고 원고와 발표 자료를 만들기 위한 색인 폴더다. 실험 이름을 주장 중심으로 다시 붙이고, 주장 → 실험 → 근거 표 → 그림의 대응을 한곳에 둔다. 작성 2026-10-04. 이름 체계는 사용자가 2026-10-04 에 승인한 안(`docs/QA_FINAL_REVIEW_2026-10-02.md` Q10)을 따른다.

## 1. 원칙

1. **원래 경로는 바꾸지 않는다.** `scripts/`, `src/`, `data/processed/`, `results/`, `outputs/`, `docs/` 의 파일은 옮기거나 이름을 바꾸거나 고치지 않는다. 스크립트가 이 경로를 참조하고 재현성 점검이 이 경로를 기준으로 하기 때문이다.
2. **이 폴더는 가리키기만 한다.** 등록부(`registry/`)는 원래 경로를 적는다. 주장 폴더(`claims/`)에 두는 사본은 판정 근거가 되는 작은 표만이고, 사본마다 원래 경로와 sha256 을 `MANIFEST.csv` 에 남긴다.
3. **판정 정본은 계획 문서의 결과 절이다.** 이 폴더의 요약과 원고 문장이 판정 기록과 다르면 판정 기록을 따른다(4절 목록).
4. **문체**: 보고서·논문 문체, 명사형 제목, 과장 금지, 접속 기호로서의 줄표 금지. 확인하지 못한 값은 `[미확인]` 으로 표시한다.

## 2. 폴더 지도

| 경로 | 내용 | 상태(2026-10-04) |
|---|---|---|
| `paper/README.md` | 이 문서 | 작성 |
| `paper/registry/experiments.csv` | 실험 등록부. 47묶음을 64행(계획 X 10행 포함)으로 등록, 18열 | 작성 |
| `paper/registry/naming_map.csv` | 옛 id → 새 이름(80행) | 작성 |
| `paper/registry/README.md` | 열 정의, 상태 표기, 갱신 방법 | 작성 |
| `paper/claims/C1_label0_safety/` … `C8_gain_source/` | 주장별 README, 판정 근거 표 사본(`tables/`), `MANIFEST.csv` | 별도 작업에서 작성 중 |
| `paper/claims/D_data_and_design/` | 자료·시험지·라벨 단위·재현성 | 별도 작업에서 작성 중 |
| `paper/claims/SI_uncertainty/`, `paper/claims/SI_learners_new_regions/` | SI 주장(예측 구간, 학습기·새 지역) | 별도 작업에서 작성 중 |
| `paper/figures/` | 현행 v2 그림과 재구성안 그림의 비교 | 예정 |
| `paper/manuscript/en/` | Sci Rep 형식 영문 원고 | 예정 |
| `paper/manuscript/ko/` | 국문 원고 | 예정 |

## 3. 이름 체계

### 3.1 전이 실험(T)과 워크플로 실험(W)

| 새 이름 | 옛 id | 내용 |
|---|---|---|
| T1_transfer_label_grid | LG | 지역 홀드아웃 + 100 km 완충, 라벨 0–전량 격자, P0·P1 대비(주 실험) |
| T2_transfer_structure_placebo_baselines | LGX | 결합 구조, 위약, 물리 기준선 11종, 검증 사다리, 학습기 용량 |
| T3_learners_foundation_and_nn | LGT, LGF | TabPFN v2, TabICL v2, 원천 교차검증으로 조정한 신경망 4종 |
| T4_new_regions | LGD | 티베트, 러시아 C, 북대서양 |
| T5_prediction_intervals | LGU | 새 지역 예측 구간 |
| T6_abstract_contrasts_and_map | LGW(+WRAPUP) | 초록 대비 AB1–AB10, 배포 시나리오, 레나 지도, 보고 규칙 |
| W1_label0_risk_table | WF0 | 라벨 0 위험표(재분석) |
| W2_within_region_label_curve | WF1, WF6 | 지역 내 충분 라벨 곡선 |
| W3_label_placement | WF2, WF8 | 관측 위치 전략(지역 내, 전이) |
| W4_subcluster_coefficients | WF3 | 하위 군집 계수 |
| W5_method_selection_and_bias_diagnosis | WF4 | 규칙 W, 라벨 10개 편향 진단 |
| WF5_not_issued | WF5 | 다음 관측 우선순위 지도(등록 규칙상 미발행) |
| W6_stefan_cci_anchor | WF7 | Stefan·CCI 평균 앵커 |
| W7_gain_decomposition_grid | WF9(+토양 √TDD 민감도) | 격자 안·격자 사이 이득 분해 |
| W8_climate_extrapolation | WF10 | 기후 외삽(판정 불가) |

### 3.2 이력(H)

| 새 이름 | 옛 id |
|---|---|
| H0_contest | 06-30 – 08-31: PRE-A, PRE-B, P0/P1, P2, PH1, W2.1, W3, AK-AUG, 타임랩스(S9), 얕은 3D(S10), S0–S8, S11–S14, 07월 E1-K·E2-D |
| H1_prereg_E1-E4 | 09-08 사전 등록 E1–E4, 09-14 대조 감사, 09-15 전이 계획 |
| H2_master_M1 | 09-16 – 09-21 M1 |
| H3_transfer_fixes_H18-24 | 09-22 H18–H24 |
| H4_label_budget_H25-30 | 09-22 H25–H30(09-26 감사 정정) |
| H5_final_A2-D1 | 09-26 A2·B1–B4·C1·C2·D1, 09-28 NEXT(미실행), 09-29 FA |

### 3.3 새 묶음(X, 계획)

| 새 이름 | 내용 | 출처 |
|---|---|---|
| XA_c2_gain_decomposition | 재보정 몫과 재보정을 넘어선 ML 몫의 분리 | QA Q1·Q6 |
| XB_multisource_stacking | 여러 물리식·제품의 라벨 가중 적층(X1) | QA Q3·Q5·Q11 |
| XC_workflow_end_to_end | 라벨 수 구간별 워크플로의 순차 모의 검증 | 사용자 질문(2026-10-04) 2 |
| XD_placement_policy | 관측 위치 규칙의 알고리즘 명시, 샘플링 정책 학습 | QA Q7, 사용자 질문 1·2 |
| XE_hires_covariates | 격자 안 해상도 입력(토양 수분, 유기층, 식생, 적설, 미지형) | 사용자 질문 3, QA Q8 |
| XF_new_regions | 독립 지역 추가 | 사용자 질문 1, QA Q9·Q12 |
| XG_product_comparison | 기존 ALT 제품 비교(WRAPUP 10절 등록분) | QA Q9 |
| XH_validation_ladder | 무작위 분할부터 지역 홀드아웃까지의 검증 사다리 그림 | QA Q12·Q15 |
| XI_climate_extrapolation_retest | WF10 재시험(외삽 영역 블록 8개 이상) | 연구 개요 12절 |
| XJ_tempderived_aux_labels | 지온 유도 라벨의 보조 학습 | QA Q9 |

### 3.4 이름 충돌 주의

| 표기 | 뜻이 둘 이상인 경우 |
|---|---|
| W3 | 옛 실험 W3(07-20 물리 결합 엔진, H0) 와 새 이름 W3_label_placement(WF2·WF8) |
| C1, C2 | 주장 C1·C2 와 09-26 통합 실험 C1(CCI 다층)·C2(계층 conformal) |
| P0, P1, P2 | 기준선 P0(원천 계수)·P1(수축 재보정)·P2(현지 최소제곱) 와 07-14 실험 P0/P1·P2 |
| S1–S7 | WF2 관측 위치 전략 S1–S7 과 07월 실험 S1–S7 |
| E1, E2 | 09-08 사전 등록 E1·E2 와 07-27 E1-K(co-kriging)·E2-D(계절 안 D(t)) |

원고와 이 폴더에서는 새 이름, 또는 날짜를 붙인 옛 id 를 쓴다.

## 4. 판정 정본 문서

| 범위 | 문서와 절 |
|---|---|
| LG, LGX, LGD, LGT | `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 7.1, 7.1a, 7.2, 7.3, 7.4 |
| LGU | `docs/EXPERIMENT_PLAN_LGU_2026-09-29.md` 11.1 |
| LGF | `docs/EXPERIMENT_PLAN_LGF_2026-09-29.md` 10.1–10.4 |
| WRAPUP, LGW | `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 1절(보고 규칙), 결과 판정 기록(J7), 지도 잠정판 |
| WF0–WF10 | `docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 2절(WF0), 5.1, 5.2, 5.3, 개정 이력(WF9 토양 민감도) |
| 주장과 워크플로 | `docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md`, `docs/RESEARCH_OVERVIEW_2026-10-02.md` |
| 사용자 질의응답(Q1–Q15) | `docs/QA_FINAL_REVIEW_2026-10-02.md` |
| 09-26 이전 | `docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md` §10, `docs/AUDIT_2026-09-26.md`, `docs/EXPERIMENT_LOG.md` |

## 5. 주장 → 실험 → 근거 표 → 그림

근거 표는 원래 경로다. 주장 폴더의 사본은 이 경로에서 복사한 것이다. 그림 열의 'v2' 는 `outputs/figures/paper/v2/` 의 현행 본문 그림이고, '재구성안' 은 `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 5절의 제안이다(사용자 결정 전).

| 주장 | 요지 | 실험(새 이름) | 근거 표(원래 경로) | 그림 | 계획 보강 |
|---|---|---|---|---|---|
| C1_label0_safety | 라벨 0 에서 직접 ML 은 시험한 학습기 10종 모두 원천 계수 물리식(P0)보다 오차가 컸다. 물리 증강과 저가중 잔차는 이 손해를 막는다(2 cm 넘게 나빠진 대상: 직접 ML 18/30, 물리 증강 2/30) | T1(L1), T2(L11·L15·L30), T3(LGT L34, LGF F1·N1), T6(AB1·AB3), W1, W6(라벨 0 앵커, 서술) | `results/rescale_lg/data/processed/lg/lg_tests.csv`; `data/processed/lgx/lgx_tests.csv`; `data/processed/lgt/lgt_tests.csv`; `data/processed/lgf/lgf_tests.csv`; `data/processed/lgf/lgfn_tests.csv`; `data/processed/lgw/lgw_bundle.csv`; `data/processed/wf/wf0_risk.csv` | v2: Fig6_label0_uncertainty(6a), Fig3_physics_use, Fig2_label_curve. 재구성안: 그대로 | XC, XF |
| C2_bias_diagnosis | 라벨을 쓰는 보정(재보정 + 잔차)의 이득은 원천 계수 오차의 크기에 비례하고 라벨 10개로 진단할 수 있다(ρ 0.66 사후 서술, WF4-c ρ 0.38 [0.17, 0.55]). 재보정을 넘어선 ML 몫과 계수 오차의 상관은 사후 점검에서 −0.08 이다(QA Q1, XA 로 확정 예정) | W1, W5(WF4-c), T4(티베트), T1(L8 지역 행), H5(FA) | `data/processed/wf/wf0_misspec.csv`; `results/rescale_wf/data/processed/wf/wf_tests.csv`; `data/processed/lgd/lgd_tests_lic.csv`; `data/processed/h4/fa_gate.csv` | v2: 해당 그림 없음. 재구성안: Fig 4 'ML 이 언제 이득인가'(계수 오차 대 이득 산점도, 진단) | XA |
| C3_structure_by_label_count | 라벨 0–10개에서는 잔차 구조가 물리 입력 구조보다 2.5–3.7 cm 낫고, 40–160개에서는 물리 입력 구조가 1.9–2.4 cm 낫다. 잔차 순가치는 학습기 종류에 묶이지 않는다 | T2(L10·L12), T1(L2), T3(LGT L33, LGF F4), T6(AB6·AB7) | `data/processed/lgx/lgx_tests.csv`; `data/processed/lgx/lgx_lg_aux.csv`; `results/rescale_lg/data/processed/lg/lg_tests.csv`; `data/processed/lgt/lgt_tests.csv`; `data/processed/lgf/lgf_tests.csv` | v2: Fig3_physics_use, Fig2_label_curve. 재구성안: 그대로 | XB |
| C4_sufficient_labels | 라벨이 많은 지역 안에서 물리 잔차 ML 의 이득은 작다(알래스카 라벨 500·1,000개 −0.39·−0.54 cm, 레나·캐나다 구별 안 됨). SAR 은 이득을 더하지 않았다 | W2(WF1·WF6), W4(WF3), T2(오차 하한), H0(S4 대회 비교) | `results/rescale_wf/data/processed/wf/wf_tests.csv`; `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`; `data/processed/lgx/lgx_floor.csv`; `data/processed/s4_residual_results.csv` | v2: 해당 그림 없음. 재구성안: Fig 7(지역 내 라벨 수 곡선, 오차 하한 띠) | XB, XE |
| C5_method_selection | 라벨 안 교차검증 선택 규칙 W 가 고정 잔차 레시피보다 라벨 40·160개에서 0.87·1.32 cm 낫다(이득은 캐나다). 원천 교차검증 조정은 전이 오차를 줄이지 않았다 | W5(WF4-a·b), T3(LGF N2·N4) | `results/rescale_wf/data/processed/wf/wf_tests.csv`; `data/processed/lgf/lgfn_tests.csv` | v2: 해당 그림 없음. 재구성안: Fig 4 또는 Fig 7 패널 | XC, XB |
| C6_label_placement | 블록·공변량 공간 분산 추출은 전이 조건에서 무작위보다 나빠지지 않았고 캐나다에서 2.4–3.2 cm 이득이 있었다. 예측 분산 기반 능동 선택은 이득이 없었다 | W3(WF2·WF8), WF5_not_issued, T6(L43), T2(L23·L24) | `results/rescale_wf/data/processed/wf/wf_tests.csv`; `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`; `data/processed/lgw/lgw_l43.csv`; `data/processed/lgx/lgx_distance.csv` | v2: Fig5_placement(L43·L23·L24). 재구성안: Fig 5 교체(WF2 전략 7종 × 예산, WF8, 능동 선택) | XD |
| C7_evaluation_design(보조) | 무작위 분할 채점은 지역 홀드아웃보다 오차를 10.3 cm 작게 낸다. 알래스카 채점 방식 비교에서 kNNDM 은 0.5° 블록보다 조금 엄격했다 | T2(L19·L20, 검증 사다리), H2(M1 채점 방식 비교), T3(LGF N2·N4), H0(PRE-B, PH1, AK-AUG 이력) | `data/processed/lgx/lgx_tests.csv`; `data/processed/lgx/ladder/lgv_tests.csv`; `data/processed/m1/cv_scheme_comparison.csv` | v2: 해당 그림 없음(09-26 판 SI `outputs/figures/paper/FigS2_scoring_protocol.pdf`). 재구성안: 검증 사다리 그림(QA Q12, 본문 표시 항목 8개 상한 안에서 배치 결정 필요) | XH, XG |
| C8_gain_source | 물리식과 총 오차가 비슷한 지역에서 ML 의 이득은 ERA5 격자 단위 편향 보정에서 온다(알래스카 격자 사이 −1.03, 격자 안 −0.01 cm). 격자 안 상세도는 확인되지 않았고 기후 외삽은 판정할 수 없었다 | W7(WF9·WF9-soil), W8(WF10), W6(WF7), H3(H18–H24 입력 음성) | `results/rescale_wf3/data/processed/wf/wf3b_tests.csv`; `results/rescale_wf3/data/processed/wf/wf3b_decomp.csv`; `data/processed/wf_soil/wf3b_tests.csv`; `results/rescale_wf2/data/processed/wf/wf2b_tests.csv` | v2: 해당 그림 없음. 재구성안: Fig 7 분해 패널 또는 SI | XE, XI, XB |
| D_data_and_design | 자료와 대상, 시험지(지역 홀드아웃·지역 내 블록 홀드아웃), 라벨 단위, 오차 하한(알래스카 11.31, 레나 14.83, 캐나다 17.90 cm), 계산 환경 간 재현성 | H0(S0, S14), H1(E2·E4 라벨 정의), T1(설계), T4(v4 자료), T6(WRAPUP 라벨 단위·교차 환경) | `data/processed/fidelity_base_v3.csv`; `data/processed/fidelity_base_v4.csv`; `results/rescale_lg/data/processed/lg/lg_targets.csv`; `data/processed/lgw/lgw_label_units.csv`; `data/processed/lgw/lgw_xenv_gate_iii.csv`; `data/processed/lgx/lgx_floor.csv` | v2: Fig1_problem, Table1_data. 재구성안: Fig 1 패널 c 에 지역 내 블록 홀드아웃 설계 추가 | XF, XG, XJ |
| SI_uncertainty | 새 지역에서 보통의 ML 구간은 과소 커버되고 계층 conformal 은 라벨 0 커버리지 0.86 [0.80, 0.92]을 회복했다. 계층 모형 구간과 조건부 폭의 이득은 확인되지 않았다(LGU-A1·B2 미결정) | T5, H5(09-26 C2), T2(L25), T6(레나 지도), H0(S11) | `data/processed/lgu/lgu_a_tests.csv`; `data/processed/lgu/lgu_b_tests.csv`; `data/processed/h4/c2_coverage.csv`; `data/processed/lgx/lgx_conformal.csv` | v2: Fig6_label0_uncertainty(6b–d). SI 지도: `outputs/maps/transfer_lena/fig_map_lena.pdf` | - |
| SI_learners_new_regions | 학습기 비교(L5·L26·L32, LGF F2–F6·N2–N4)와 새 지역 세부(LGD 두 판) | T3, T1(L5), T2(L26·L30), T4 | `data/processed/lgt/lgt_tests.csv`; `data/processed/lgf/lgf_tests.csv`; `data/processed/lgf/lgfn_tests.csv`; `data/processed/lgd/lgd_tests_lic.csv`; `data/processed/lgd/lgd_tests.csv` | v2: Fig6_label0_uncertainty 6a 의 학습기 행. SI 그림은 예정 | XF, XJ |

수치 출처: C1·C3·C5·C6·C7 은 `docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md` 3절, C2 의 −0.08 과 C7 의 채점 방식 비교는 `docs/QA_FINAL_REVIEW_2026-10-02.md` Q1·Q15, C4·C8 은 `docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 5.2·5.3, 오차 하한은 `data/processed/lgx/lgx_floor.csv`(LG 계획서 7.2 인용), SI_uncertainty 는 LGU 계획서 11.1 과 `outputs/figures/paper/workflow_tags.json` 이다.

## 6. 그림 위치

| 구분 | 경로 | 비고 |
|---|---|---|
| 현행 본문 그림(v2) | `outputs/figures/paper/v2/` | Fig1_problem, Fig2_label_curve, Fig3_physics_use, Fig4_min_labels, Fig5_placement, Fig6_label0_uncertainty, Fig7_deployment, Table1_data(각 pdf·png·svg, 표는 csv·md·tex 포함) |
| 그림 값 표(v2) | `outputs/figures/paper/source_data/v2/`, `data/processed/paper_figs/` | 패널별 원천 값 |
| 그림 코드 | `scripts/4_visualization/paper/` | v2: `fig1_problem.py` … `fig7_deployment.py`, `fig6_label0_uncertainty.py`, `table1_data.py`, `v2_data.py`, `v2_style.py`, `v2_values_check.py`, 지도 `fig_map_lena.py` |
| 그림 명세 | `figures/figure_spec.json`, `docs/DISPLAY_ITEMS_2026-09-30.md` | 패널별 원천 표, 색표, 내보내기, QA 규칙 |
| 시각 QA 기록 | `outputs/figures/paper/_qa/` | 자동 QA json, 값 대조 |
| 09-26 판(구판) | `outputs/figures/paper/Fig*.pdf`, `outputs/figures/paper/FigS*.pdf`, `outputs/figures/paper/supp_tables/` | 09-26 통합 실험 기준. LG 계열 이후 원고에 쓰려면 수치를 다시 대조한다 |
| 레나 지도(SI) | `outputs/maps/transfer_lena/` | fig_map_lena.{pdf,png,svg}, 캡션 md, QA json |
| 재구성 비교 | `paper/figures/` | 예정. v2 와 재구성안(Fig 4·5·7 교체, 검증 사다리)을 나란히 둔다 |

Scientific Reports 본문 표시 항목 상한은 8개다(`docs/DISPLAY_ITEMS_2026-09-30.md` 1절). 검증 사다리 그림이나 새 Fig 4·5·7 은 이 상한 안에서 배치를 정한다.

## 7. 원고 위치

| 구분 | 경로 | 비고 |
|---|---|---|
| 기존 초안(2026-09-30) | `docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md`, `docs/MANUSCRIPT_DRAFT_RESULTS_ABSTRACT_2026-09-30.md`, `docs/MANUSCRIPT_DRAFT_SUPPORT_2026-09-30.md` | 본문은 영문, 머리말은 국문. `wc -w` 는 각각 10,952, 13,425, 11,329 단어이며 머리말·표·자리표시·참고문헌을 포함한 값이다. LG 계열 판정 기준이며 WF 실험과 10-02 주제 재설정은 반영되지 않았다 |
| 재구성안 | `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` | 결과 절 순서, 그림 배치, 제목 후보, 사용자 결정 3건 |
| 투고 요소 초안 | `docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md`, `docs/PAPER_PLAN_SCIREP.md` | 09-21·08-31 작성. 자료·코드 공개, 선언문 형식 참고용 |
| 대회 보고서 | `outputs/report/main.tex` | 국문 대회 예선 보고서. 논문 원고가 아니다 |
| 새 영문 원고 | `paper/manuscript/en/` | 예정. Sci Rep 형식 |
| 새 국문 원고 | `paper/manuscript/ko/` | 예정 |

## 8. 갱신 규칙

- 등록부 갱신 방법은 `paper/registry/README.md` 5절을 따른다.
- 새 실험(X 묶음)은 사전 등록 문서를 먼저 커밋하고 실행한다. 결과 파일은 각 하네스의 기본 산출 경로에 두고, 이 폴더에는 경로와 요약만 더한다.
- 주장 문장이 바뀌면 5절 표의 '요지' 와 해당 `claims/` README 를 함께 고친다. 판정 기록 자체는 고치지 않는다.
