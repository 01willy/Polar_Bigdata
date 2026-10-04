# SI 불확실성(예측 구간) 근거 색인

- 작성: 2026-10-04. 성격: 논문용 색인이다. 원본 스크립트, 자료, 문서는 옮기거나 고치지 않았다.
- 수치 원본: 아래 모든 수치는 `extract_evidence.py` 가 원천 집계 표를 pandas 로 읽어 만든 `evidence_values.csv` 에서 옮겼다(산문에서 옮긴 값 없음). 재실행: `OMP_NUM_THREADS=1 python paper/claims/SI_uncertainty/extract_evidence.py`(적합·재표집 없음, 수 초). 같은 스크립트가 `tables/` 사본과 `tables/MANIFEST.csv` 를 다시 만든다. 예외: 2026-10-04 정정(8절)에서 더한 값 일부는 사본 표(`tables/`)에서 직접 읽었거나 셀 단위 원천(`s11_conformal_oof.csv`)에서 다시 계산했다. 해당 위치에 출처와 계산 방법을 적었다.
- 값 표기: 표의 값은 파일 값을 소수 둘째 자리로 반올림한 것이다. 커버리지(비율)는 둘째 자리로 0.4456 과 0.45 를 구별하지 못하므로 `evidence_values.csv` 의 `value_4dp` 열(넷째 자리)을 괄호 없이 함께 적었다. 문서에만 있고 원천 표에서 찾지 못한 값은 [미확인]으로 표시했다.
- 판정어: 4분 판정(우세, 열세, 동등, 미결정)이다. 우세·열세는 셀 가중과 블록 등가중 두 95 % CI 가 모두 같은 쪽일 때다. 구간 점수 대비의 동등 한계는 기준 방법 점수의 5 %(보조 10 %)이고, CI 반폭이 한계를 넘으면 동등 판정은 '검정 불가(정밀도 미달)'다(LGU 계획서 2.7). 미결정은 '효과 없음'이 아니라 '구별하지 못함'이다.
- 부호: 구간 점수 Δ(A − B) 는 음수면 A 가 낫다. 커버리지 표는 대비가 아니라 수준 값이다.
- 이름 주의: 09-26 계획서의 실험 번호 'C2'(h37 계층 conformal)는 이 색인 체계의 주장 C2(편향 진단)와 다르다. 이 문서는 그 실험을 'h37 계층 conformal(09-26)'로 부른다. 09-26 문서와 그림 캡션의 'AB4'는 주 4지역 평균을 뜻하며 WRAPUP 초록 묶음의 AB4(P1 − P0, n 10)와 다르다.

## 1. 주장 문장

### 1.1 현행 문장

`docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md` 의 C1–C8 에는 불확실성 주장이 없다. 머리말(3행)이 판정 근거 목록에 LGU 11.1 을 넣었을 뿐이고, 6절 한계에도 예측 구간 항목이 없다. 현재 쓰이는 문장은 다음 다섯 곳에 흩어져 있다.

1. `docs/QA_FINAL_REVIEW_2026-10-02.md` Q7(185–189행, 원문). 이 SI 항목의 주장 목록이 여기서 왔다.

   > 지금 근거로는 '물리식보다 나은 불확실성'을 주장하기 어렵다.
   > - 알래스카 지역 내에서는 보정 구간의 커버리지를 목표 90 % 에 맞출 수 있었다(대회 44.6 → 93.4 %). 그러나 셀마다 폭이 달라지는 정보는 상수 폭 구간보다 낫지 않았다(구간 점수 68.9 대 65.4).
   > - 새 지역에서는 보통의 CQR 구간이 무너졌다(커버리지 0.32–0.73). 계층 conformal 구간은 0.86 [0.80, 0.92]을 회복했다.
   > - 생성 모델 구간은 과소 커버리지였다(76 %, 65 %). 라벨로 구간을 개선한다는 확인적 대비(AB10)는 미결정이었다.
   > - 쓸 수 있는 문장: '새 지역에서는 보통의 ML 구간이 과소 커버되므로 계층 conformal 같은 재보정이 필요하다.'

2. `docs/RESEARCH_OVERVIEW_2026-10-02.md` 51행: 'LGU | 예측 구간이 새 지역에서 맞는가 | 로컬 | 생성 모델 구간 과소 커버리지(76 %, 65 %), 계층 구간 이득 미확인'.
3. `docs/MANUSCRIPT_DRAFT_RESULTS_ABSTRACT_2026-09-30.md` 초록(15행) 'Hierarchical and constant-width prediction intervals (three regions) showed no established difference in interval score.', 결과 96–100행(LGU-B1, B2, B3, B4, A1, A3, L25), 한계 199행.
4. `docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md` 10.4 (k)(302행): 계층 conformal 0.86 문장을 '한정'으로 내리고 LGU 판정을 덧붙인 문장.
5. `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 4절 R8: '예측 구간 | 보조 | LGU-A1·B1·B2, L25 | Fig 6b–d'.

### 1.2 필요한 축소

`docs/QA_FINAL_REVIEW_2026-10-02.md` 의 공통 축소 항목(C2, '안전', WF 사후 설계)과 각 실험의 등록 규칙을 이 항목에 적용한 결과다.

| 항목 | 이 항목에 대한 적용 |
|---|---|
| C2 축소(재보정 몫과 ML 몫의 분리, QA Q1 C2 점검) | (1) 이름: 09-26 실험 'C2'(h37)는 주장 C2 와 무관하다. Fig 6 v2 범례 'C2 interval, earlier protocol', v2 캡션 'C2 interval 0.86, 81 cm', 09-26 SI Table 6c '(C2)' 를 원고에서 'hierarchical conformal (09-26 protocol)' 로 바꾼다. (2) 몫의 분리: 라벨이 있는 구간의 중심은 재보정 계수 E_n(P1) 이거나 R1 이다. 계층 예측 분포(단 iii)의 중앙값은 P1 보다 RMSE 가 컸다(LGU-A3 열세, 2.6절). L25 의 '폭이 좁아졌다'는 R1(n 40·160) 구간과 R0(n 0) 구간의 비교라 재보정과 잔차의 몫이 섞여 있다. 따라서 구간 결과를 ML 이득의 근거로 쓰지 않는다 |
| '안전' 표현(등록된 비열등 판정이 있을 때만) | 예측 구간에는 비열등 판정이 등록되지 않았다. '안전한 구간', '믿을 수 있는 구간', '커버리지가 보장된 구간·지도'를 쓰지 않는다(LGU 10절, WRAPUP 11.2). 보정 집단이 4개라 정확형 계층 conformal(hier2_exact)은 α 0.1 에서 폭이 무한이고(`h4/c2_meta.json` note_exact: K + 1 < 1/α), 계층 구간의 커버리지는 유한표본 보장이 없는 경험값이다 |
| WF 사후 표지 | 이 항목에 WF 결과는 없다. 대신 근거마다 등록 상태가 다르다(3절 표의 '등록' 열). S11 은 대회 단계 산출(등록 없음), E4.3 은 S11 OOF 의 재분석(결과를 본 뒤 설계, 서술), H17 은 09-21 개정 F 에 적혔으나 커밋은 결과와 같은 커밋(a5970f4, 2026-09-21 15:14:45)이라 커밋 기준으로는 사전 등록을 확인할 수 없다(작업 트리 기록. 실행 시점 HEAD 는 `transfer_uq_meta.json` git_commit 291e2c7 이고 그 판의 마스터 계획서에 H17 이 없다. `transfer_uq_summary.csv` 기록 시각 2026-09-21 15:04:05 가 커밋보다 앞선다. `docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md` 197행도 이 개정을 '미커밋'으로 적었다. 회복 방향은 탐색적), h37 계층 conformal 은 09-26 사전 등록(F10), LGU-A1·B2 는 09-29 사전 등록(맹검), LGU-B1 은 '재현(비맹검)', AB10 은 WRAPUP 1.1 묶음(Holm m = 10)이다. 확인적 문장은 LGU-A1, LGU-B2(둘 다 미결정)와 한계 확인 LGU-B1(지지)뿐이다(LGU 11.1 '원고 문장') |
| '대회 44.6 → 93.4 %' | S11 은 알래스카 지역 내 6-fold 0.5° 블록 OOF 의 직접 CatBoost 분위 모형(x34 입력)이다. 이 논문의 물리 잔차 모형도 전이 조건도 아니다. 지역 내 보정이 가능하다는 사례로만 쓰고 주 방법의 성능으로 쓰지 않는다 |
| '68.9 대 65.4' | 두 값은 비짝지음 CI 의 점 추정이다. 원천 표에는 차이의 CI 가 없다. 문서(RESULTS_RECONCILIATION 3.7)의 3.5 [0.6, 6.5] 는 정확히 재현되지 않으며 부트스트랩 추출에 따라 달라진다(2.2절 읽는 법, 5.1 단서 11). 다시 계산한 짝지은 CI 는 모두 0 을 넘는다. 상수 폭 참조는 채점 셀 자신의 잔차 90 % 분위로 반폭(21.73 cm)을 정한 사후 참조라 상수 폭에 유리하다. '낫지 않았다(점 추정)'까지만 쓰고 '유의하게 나빴다'를 쓰지 않는다 |
| '0.32–0.73' | H17 의 알래스카 보정 CQR(cqr_ak), 주 4지역 × 두 조건(정보 없음, 공변량만) 8행의 범위다. 양 끝은 모두 정보 없음 조건(러시아 W 0.3214, 러시아 E 0.7284)이다. 8행 모두 커버리지 CI 상한이 0.90 미만이다 |
| '0.86 [0.80, 0.92]' | h37 hier2_cdf 의 주 4지역 지역 등가중 평균(지역 안은 셀 가중)이다(0.8628 [0.8021, 0.9172], 4지역 값 (0.7500 + 0.9259 + 0.8862 + 0.8891)/4). 셀 수(n_eval) 가중 평균은 0.89 [0.82, 0.95](`coverage_cellw` 0.8878 [0.8214, 0.9468])이고, 블록 등가중(지역 평균)은 0.85(`coverage_beq` 0.8531)이다. 출처: `h4/c2_coverage.csv` MEAN6 행, `h37_conformal_hier.py` 283행 주석('등가중: 층화 블록 부트스트랩 결합, 셀 가중: n_eval 가중'), `paper_figs/v2_data_fig6_meta.json` 점검값 `fig6b_C2_MEAN_cov_vs_region_mean` 0.0, CAPTIONS.md 11행('unweighted mean'). 러시아 W 는 0.75 [0.59, 0.88]로 목표 띠 밖이고, 레나는 셀 가중 0.89 이나 블록 등가중 0.81 이다. 판정은 '부분 지지'(F10)다. '회복했다'보다 '평균 0.86, 지역 범위 0.75–0.93'으로 쓴다 |
| '76 %, 65 %' | LGU-B1 의 cqr_pool(λ 1.0, 원천 셀 교환성 보정) 주 4지역 커버리지다(블록 등가중 0.79, 0.68). 같은 nflow 를 계층 conformal 의 폭 정규화기로 쓰면 커버리지는 0.89 다(Fig 6b). '생성 모델 구간'이 아니라 '원천 셀 교환성으로 보정한 생성 분위 구간'으로 쓴다 |
| 'AB10 미결정' | 2단 CI 판정은 미결정이고 1단 CI 판정은 n 10 열세, n 40 우세라 '보정 불확실성 의존'이다. 두 n 모두 동등 검정 불가(정밀도 미달)다. 초록 규칙은 '(d) 차이를 확인하지 못했다'다. AB10 은 '라벨로 구간을 개선한다'의 대비가 아니라 '계층 예측 분포 구간 대 같은 라벨로 보정한 상수 폭 구간'의 대비다(LGU 1.3 표: '라벨이 구간을 좁힌다'는 LGU 가 판정하지 않는다) |

### 1.3 권고 문장

한국어(SI 본문, 본문에는 1–2문장):

> 예측 구간(보조). 알래스카 지역 내 0.5° 블록 교차검증에서 분위 CatBoost 의 90 % 구간 커버리지는 보정 전 0.45 [0.32, 0.62], CQR 보정 뒤 0.93 [0.89, 0.97]이었다. 같은 예측에서 CQR 의 구간 점수(68.85 cm)는 채점 셀 잔차로 폭을 정한 상수 폭 참조(65.39 cm)보다 낮지 않았다. 알래스카에서 보정한 CQR 구간을 학습에서 뺀 주 4지역에 쓰면 커버리지는 0.32–0.73 이었고, 원천 셀을 교환 가능하다고 보고 보정한 생성 분위 구간은 0.76(정규화 흐름), 0.65(흐름 정합)였다. 지역을 집단으로 보정한 계층 conformal 구간은 라벨 0 에서 주 4지역 평균 0.86 [0.80, 0.92]이었으나 러시아 서부는 0.75 였다. 보정 집단이 4개여서 유한표본 보장은 없다. 셀마다 폭을 바꾸는 nflow 정규화기는 지역 안에서 σ 를 섞은 위약 대비 구간 점수 차이가 확인되지 않았고(LGU-B2, +3.29 [−4.17, 8.16]), 어느 정규화기도 상수 폭 정규화기와 차이가 확인되지 않았다(LGU-B4). 라벨 10·40개의 계층 예측 분포도 보정한 상수 폭 구간(B4)과 구간 점수 차이가 확인되지 않았다(LGU-A1).

QA Q7 의 '쓸 수 있는 문장'('… 계층 conformal 같은 재보정이 필요하다')은 위 문장으로 바꾼다. 계층 conformal 과 CQR 의 짝지음 우세 대비는 등록되지 않았고(F10 은 띠 판정), 러시아 W 에서 계층 구간도 목표에 못 미쳤기 때문이다.

영어(원고 SI 초안):

> Prediction intervals (supplementary). Within Alaska, under 0.5° block cross-validation, the 90 % coverage of quantile CatBoost intervals was 0.45 [0.32, 0.62] before and 0.93 [0.89, 0.97] after conformalized quantile regression (CQR). On the same predictions, the interval score of CQR (68.85 cm) was not lower than that of a constant-width reference whose half-width was set from the scoring residuals (65.39 cm). Applied to the four held-out main regions, Alaska-calibrated CQR intervals covered 0.32–0.73, and generative quantile intervals calibrated on exchangeable source cells covered 0.76 (normalizing flow) and 0.65 (flow matching). Hierarchical conformal intervals, calibrated with regions as groups, covered 0.86 [0.80, 0.92] on average without labels, but 0.75 in Russia W; with four calibration regions there is no finite-sample guarantee. The interval score of the cell-specific flow normalizer did not differ by an established amount from its placebo with σ permuted within regions (LGU-B2, +3.29 cm [−4.17, 8.16]), and no normalizer differed from the constant-width normalizer (LGU-B4). Hierarchical predictive distributions with 10 or 40 labels did not differ in interval score by an established amount from calibrated constant-width intervals (B4) either (LGU-A1).

## 2. 근거 표

### 2.1 원천 약호

| 약호 | 원천 경로 | 사본(`tables/`) | 판정 문서 절 |
|---|---|---|---|
| S11 | `data/processed/s11_conformal_results.csv`(+ `s11_conformal_meta.json`) | `s11_conformal_results.csv` | `docs/EXPERIMENT_LOG.md` 2026-07-27 S6~S11 절(S11 '채택, 헤드라인') |
| E4 | `data/processed/e4_interval_score.csv`(+ `_meta.json`) | `e4_interval_score.csv` | `docs/EXPERIMENT_LOG.md` 'E4.3 구간 점수·조건부 커버리지 (W12)' 절, `docs/RESULTS_RECONCILIATION_2026-09-14.md` 3.7 |
| E4C | `data/processed/e4_conditional_coverage.csv` | `e4_conditional_coverage.csv` | 같음 |
| TUQ | `data/processed/m1/transfer_uq_summary.csv`(+ `transfer_uq_meta.json`) | `m1__transfer_uq_summary.csv` | `docs/EXPERIMENT_LOG.md` '보조 분석 결과 (09-21)' 절 H17 행. 판정 기준은 `docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md` H17 행(커버리지 CI 가 0.90 미만인지) |
| C2T | `data/processed/h4/c2_coverage.csv`(+ `c2_meta.json`) | `h4__c2_coverage.csv` | `docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md` 결과 표 F10 행(258행), 10.4 (k) |
| LB | `data/processed/lgu/lgu_b_tests.csv` | `lgu__lgu_b_tests.csv` | `docs/EXPERIMENT_PLAN_LGU_2026-09-29.md` 11.1(J3) |
| LA | `data/processed/lgu/lgu_a_tests.csv` | `lgu__lgu_a_tests.csv` | 같음 |
| AB | `data/processed/lgu/lgu_a_ab10_tests.csv`, `data/processed/lgw/lgw_bundle.csv` | `lgu__lgu_a_ab10_tests.csv`, `lgw__lgw_bundle.csv` | `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 결과 판정 기록(J7) 1.1 |
| F6B, F6C | `data/processed/paper_figs/fig6_b.csv`, `fig6_c.csv`(LGU 표를 옮긴 그림 표, 새 재표집 없음) | `paper_figs__fig6_b.csv`, `paper_figs__fig6_c.csv` | LGU 11.1 |
| X | `data/processed/lgx/lgx_tests.csv` | `lgx__lgx_tests.csv` | `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 7.2(J2) L25, WRAPUP '지도 잠정판' 절의 'L25 와 LGU 10절의 문장 충돌(14:20)' |
| SC | `data/processed/m1/m1_sc_uq.csv` | `m1__m1_sc_uq.csv` | `docs/EXPERIMENT_LOG.md` H15 행(LGU 4.6·11.1 에서 LGU-B1 로 대체) |

재표집: S11·E4·h37·LGU 는 1,000회, TUQ 는 400회(`transfer_uq_meta.json` nboot), 초록 묶음은 10,000회(AB10 만 LGU 1,000회).

### 2.2 U1 알래스카 지역 내 보정 구간(대회 S11, E4.3 재분석)

| 항목 | 원천 | 행 필터 | 열 | 값 | 4자리 | 판정어 |
|---|---|---|---|---|---|---|
| 보정 전 분위 구간(3 seed 앙상블 구간, 분위 경계 평균) | S11 | `setting=='raw_seedmean' & level==0.9` | coverage / cov_ci_lo / cov_ci_hi / width_cm | 0.45 / 0.32 / 0.62 / 14.65 | 0.4456 / 0.3227 / 0.6167 | 서술(대회 '채택') |
| CQR(3 seed 앙상블 구간, 분위 경계 평균) | S11 | `setting=='cqr_seedmean' & level==0.9` | coverage / cov_ci_lo / cov_ci_hi / width_cm | 0.93 / 0.89 / 0.97 / 53.64 | 0.9344 / 0.8894 / 0.9659 | 서술(대회 '채택') |
| 보정 전 구간 점수 | E4 | `interval=='raw'` | interval_score / is_ci_lo / is_ci_hi | 123.91 / 88.04 / 153.78 | | 서술 |
| CQR 구간 점수 | E4 | `interval=='cqr'` | interval_score / is_ci_lo / is_ci_hi | 68.85 / 58.64 / 94.92 | | 서술(원고 조치: 서술 완화) |
| CQR 분해 | E4 | `interval=='cqr'` | is_width_part / is_under_part / is_over_part | 53.64 / 2.66 / 12.55 | | 서술 |
| CQR 커버리지(E4 재채점, S11 과 같은 앙상블 구간) | E4 | `interval=='cqr'` | coverage / cov_ci_lo / cov_ci_hi | 0.93 / 0.89 / 0.96 | 0.9344 / 0.8942 / 0.9632 | 서술 |
| 상수 폭 참조 구간 점수 | E4 | `interval=='const_width'` | interval_score / is_ci_lo / is_ci_hi | 65.39 / 55.61 / 92.19 | | 서술 |
| 상수 폭 참조 분해 | E4 | `interval=='const_width'` | is_width_part / is_under_part / is_over_part | 43.46 / 3.84 / 18.09 | | 서술 |
| 상수 폭 참조 커버리지 | E4 | `interval=='const_width'` | coverage / cov_ci_lo / cov_ci_hi | 0.90 / 0.85 / 0.92 | 0.9000 / 0.8498 / 0.9250 | 서술 |
| 상수 폭 반폭(사후 참조) | E4 meta | `const_width_reference` | half_width_cm | 21.73 | | 해당 없음 |
| CQR − 상수 폭(산술 차) | E4 | 위 두 행 | interval_score | +3.46 | | 서술. 짝지음 CI 는 원천 표에 없음. 문서의 [0.6, 6.5] 는 정확히 재현되지 않음(추출 의존, 아래 읽는 법) |
| CQR 조건부 커버리지(관측 ALT 5분위) | E4C | `interval=='cqr' & axis=='alt_quintile' & bin==Q1…Q5` | coverage | 0.85 / 1.00 / 1.00 / 1.00 / 0.83 | 0.8479 / 0.9960 / 1.0000 / 0.9996 / 0.8284 | 서술 |
| CQR 폭 5분위 양 끝 | E4C | `interval=='cqr' & axis=='width_quintile' & bin=='W1'`, `'W5'` | coverage / interval_score | W1 0.86 / 75.52, W5 0.97 / 76.59 | W1 0.8593, W5 0.9680 | 서술 |
| CQR 블록별 커버리지(셀 20개 이상 33블록) | E4C | `interval=='cqr' & axis=='block_coverage_dist'` | coverage(중앙값) / lo(10 백분위) / hi(90 백분위) / interval_score(커버리지 0.8 미만 블록 비율) | 0.98 / 0.70 / 1.00 / 0.15 | 0.9800 / 0.7024 / 1.0000 / 0.1515 | 서술 |
| 참조: M1 판 알래스카 지역 내 CQR(x25) | TUQ | `region=='Alaska' & cond=='in_domain' & method=='cqr_ak'` | coverage / cov_ci_lo / cov_ci_hi / width_mean / interval_score | 0.91 / 0.83 / 0.94 / 48.94 / 70.35 | 0.9086 / 0.8336 / 0.9426 | 서술 |

읽는 법: CQR 은 지역 내 평균 커버리지를 목표 0.90 에 맞췄다. 구간 점수에서 CQR 의 이점은 상수 폭 참조 대비 보이지 않았고, 폭 5분위 양 끝의 구간 점수(75.52, 76.59)가 비슷해 폭이 오차 크기를 가려내는 정보가 작다. 조건부 커버리지는 ALT 양 끝 5분위에서 0.83–0.85 로 낮고 가운데에서 과다하다.

S11 의 앙상블 행과 E4: S11 의 `raw_seedmean`·`cqr_seedmean` 은 seed 별 커버리지의 평균이 아니다. 세 seed 의 분위 경계를 평균한 앙상블 구간의 커버리지다(`s11_conformal_uq.py` 148–162행). seed 별 커버리지(`level==0.9 & seed∈{0,1,2}`, 사본 `tables/s11_conformal_results.csv`)를 평균하면 CQR 0.9042((0.8240 + 0.9613 + 0.9272)/3), 보정 전 0.4145((0.4562 + 0.4458 + 0.3415)/3)로 표의 0.9344·0.4456 과 다르다. `s11_conformal_oof.csv` 는 이 앙상블 구간을 저장하므로(같은 스크립트 15행, 170–178행) E4 는 S11 앙상블 OOF 를 다시 채점한 것이다. E4 의 cqr 커버리지·폭(0.9344·53.64)과 raw(0.4456·14.65)는 S11 앙상블 행과 같다. 같은 예측의 커버리지 CI 가 S11 [0.89, 0.97](0.8894, 0.9659)과 E4 [0.89, 0.96](0.8942, 0.9632)으로 다른 것은 부트스트랩 추출 차이다. 원고에는 CI 를 하나만 쓴다.

[미확인] 해소(2026-10-04 점검, 셀 단위 원천 `data/processed/s11_conformal_oof.csv` 13,606행에서 스레드 1개로 재계산, 파일로 저장하지 않음):
- RESULTS_RECONCILIATION 3.7 의 '셀별 폭과 절대 오차의 순위 상관 −0.07'은 Spearman(`width90_cqr`, |`alt_cm` − `pred_med`|) = −0.0722 로 재현된다.
- 같은 문서의 'CQR − 상수 폭 3.5 [0.6, 6.5]'는 점 추정(+3.46)만 재현된다. 셀별 구간 점수 차 IS(CQR) − IS(상수 폭, 반폭 21.7285, `e4_interval_score_meta.json`)의 짝지은 0.5° 블록 부트스트랩(74블록, 1,000회) 95 % CI 는 E4 스크립트와 같은 절차(`np.random.RandomState(0)`)에서 [0.78, 6.50], `RandomState(1)` 에서 [0.50, 6.62], `np.random.default_rng(0)` 에서 [0.39, 6.64] 이다. 추출에 따라 값이 달라지며 모두 0 을 넘는다. 원고에 쓰려면 고정 seed 스크립트로 다시 내고 값을 파일로 기록한다. 상수 폭 참조가 채점 셀 잔차로 정한 사후 oracle 이라는 단서는 그대로다.

### 2.3 U2 전이 조건 CQR(H17, M1 보조 분석 09-21)

알래스카에서 보정한 CQR(cqr_ak), 밀도비 가중 CQR(cqr_iw), Stefan·CCI 앵커 ± 알래스카 보정 분위(anchor_ak, anchor_iw)의 90 % 커버리지다. 원천 TUQ, 열 `coverage / cov_ci_lo / cov_ci_hi`. 판정 기준(H17 확인적 부분): 커버리지 CI 상한 < 0.90. H17 의 등록 시점은 커밋으로 확인되지 않는다(1.2 'WF 사후 표지' 행).

| 방법 | 조건 | 레나 | 캐나다 | 러시아 W | 러시아 E | 판정어 |
|---|---|---|---|---|---|---|
| cqr_ak | noinfo | 0.57 [0.48, 0.66] (0.5700) | 0.60 [0.43, 0.74] (0.5953) | 0.32 [0.20, 0.43] (0.3214) | 0.73 [0.56, 0.89] (0.7284) | H17 확인(4/4 CI 상한 < 0.90) |
| cqr_ak | covonly | 0.57 [0.46, 0.64] (0.5748) | 0.53 [0.35, 0.74] (0.5292) | 0.34 [0.17, 0.45] (0.3415) | 0.66 [0.48, 0.90] (0.6597, 상한 0.8964) | H17 확인(4/4) |
| cqr_iw | noinfo | 0.35 [0.26, 0.41] | 0.52 [0.38, 0.68] | 0.30 [0.17, 0.43] | 0.69 [0.53, 0.87] | 서술(회복 없음) |
| cqr_iw | covonly | 0.33 [0.21, 0.40] | 0.47 [0.30, 0.66] | 0.34 [0.15, 0.46] | 0.62 [0.44, 0.87] | 서술 |
| anchor_ak | noinfo | 0.92 [0.79, 0.95] | 0.77 [0.68, 0.87] | 0.57 [0.44, 0.71] | 0.84 [0.71, 1.00] | 서술 |
| anchor_iw | noinfo | 0.93 [0.79, 0.96] | 0.88 [0.79, 0.94] | 0.56 [0.41, 0.70] | 0.84 [0.71, 1.00] | 서술 |

행 필터는 모두 `region=='<지역>' & cond=='<조건>' & method=='<방법>'` 이다. covonly 의 anchor 행(anchor_ak, anchor_iw)과 cqr_real 은 `evidence_values.csv` 에 있다. cqr_pseudo·anchor_pseudo·anchor_real 은 `evidence_values.csv` 에 없고 사본 표(`tables/m1__transfer_uq_summary.csv`)에만 있다(세 방법 모두 covonly 조건만 있다). 원천 값(covonly, 레나/캐나다/러시아 W/러시아 E, coverage): cqr_pseudo 0.2963/0.4593/0.2364/0.4179, anchor_pseudo 0.5369/0.3356/0.3676/0.5735, anchor_real 0.9563/0.7867/0.9722/0.8918.

| 요약 항목 | 원천 | 행 필터 | 열 | 값 | 4자리 | 판정어 |
|---|---|---|---|---|---|---|
| cqr_ak 범위(8행) | TUQ | `method=='cqr_ak' & cond∈{noinfo, covonly} & region∈주 4지역` 최솟값·최댓값 | coverage | 0.32–0.73 | 0.3214(러시아 W noinfo)–0.7284(러시아 E noinfo) | H17 확인 |
| cqr_iw 범위(8행) | TUQ | 같은 조건, `method=='cqr_iw'` | coverage | 0.30–0.69 | 0.2976–0.6914 | 서술 |
| cqr_ak noinfo 4지역 단순 평균(산술) | TUQ | `method=='cqr_ak' & cond=='noinfo' & region∈주 4지역` | coverage 평균 | 0.55 | 0.5538 | 서술(F10 행의 '기존 CQR AB4 0.55') |
| 대상 실측 보정(상한 참조) | TUQ | `method=='cqr_real' & cond=='covonly' & region∈주 4지역` 최솟값·최댓값 | coverage | 0.71–0.97 | 0.7127–0.9722 | 서술 |
| 6지역 평균 참조(점 추정, H17 표에서 복사) | C2T | `test=='ref_h17' & method=='ref_cqr_ak'`, `'ref_cqr_iw'` (target `MEAN6[Canada,Greenland,Lena,Russia_C,Russia_E,Russia_W]`) | coverage | 0.64, 0.55 | 0.6390, 0.5475 | 서술 |

### 2.4 U3 라벨 0 계층 conformal(h37, 09-26 사전 등록 F10)

원천 C2T. 대상 지역을 빼고 나머지 지역을 집단으로 보정한 90 % 구간이다. 주 4지역 평균 행의 target 은 `MEAN6[Russia_W,Russia_E,Canada,Lena]` 다(CI 풀 4/6, 채점 3,760셀·93블록). 이 행의 `coverage` 는 4지역 값의 단순 평균(지역 등가중, 지역 안은 셀 가중)이고 CI 는 층화 블록 부트스트랩 결합이다. `coverage_beq` 는 지역별 블록 등가중 값의 평균이다. 셀 수(n_eval) 가중 평균은 `coverage_cellw` 0.89 [0.82, 0.95](0.8878 [0.8214, 0.9468])로 따로 있다(`h37_conformal_hier.py` 283–303행. 이 열은 `evidence_values.csv` 에 없고 사본 `tables/h4__c2_coverage.csv` 에서 읽었다).

| 항목 | 행 필터 | 열 | 값 | 4자리 | 판정어 |
|---|---|---|---|---|---|
| hier2_cdf(주 방법) 주 4지역 | `test=='label0' & method=='hier2_cdf' & target=='MEAN6[Russia_W,Russia_E,Canada,Lena]'` | coverage / coverage_lo / coverage_hi / coverage_beq / width_cm / interval_score | 0.86 / 0.80 / 0.92 / 0.85 / 80.54 / 131.98 | 0.8628 / 0.8021 / 0.9172 / 0.8531 | 부분 지지(F10) |
| 셀 풀링(pooled) | 같은 target, `method=='pooled'` | coverage / coverage_lo / coverage_hi / coverage_beq / width_cm / interval_score | 0.73 / 0.65 / 0.81 / 0.74 / 56.34 / 152.86 | 0.7276 / 0.6479 / 0.8050 / 0.7391 | 서술 |
| hier2_sub | 같은 target, `method=='hier2_sub'` | coverage / coverage_lo / coverage_hi / width_cm | 0.83 / 0.77 / 0.89 / 72.86 | 0.8334 / 0.7709 / 0.8908 | 서술 |
| hier2_blk | 같은 target, `method=='hier2_blk'` | coverage / coverage_lo / coverage_hi / width_cm | 0.87 / 0.81 / 0.93 / 83.30 | 0.8730 / 0.8140 / 0.9278 | 서술 |
| hier2_cdf 레나 | `test=='label0' & method=='hier2_cdf' & target=='Lena'` | coverage / coverage_lo / coverage_hi / coverage_beq / width_cm / K_groups | 0.89 / 0.80 / 0.96 / 0.81 / 73.03 / 4 | 0.8891 / 0.8042 / 0.9600 / 0.8147 | 서술 |
| hier2_cdf 캐나다 | `… target=='Canada'` | 같음 | 0.89 / 0.80 / 0.97 / 0.88 / 95.35 / 4 | 0.8862 / 0.8000 / 0.9665 / 0.8850 | 서술 |
| hier2_cdf 러시아 W | `… target=='Russia_W'` | 같음 | 0.75 / 0.59 / 0.88 / 0.74 / 71.13 / 4 | 0.7500 / 0.5909 / 0.8800 / 0.7389 | 서술(목표 띠 밖) |
| hier2_cdf 러시아 E | `… target=='Russia_E'` | 같음 | 0.93 / 0.80 / 1.00 / 0.97 / 82.64 / 4 | 0.9259 / 0.8000 / 1.0000 / 0.9737 | 서술 |
| hier2_cdf 알래스카(참조) | `… target=='Alaska'` | 같음 | 0.98 / 0.96 / 0.99 / 0.97 / 78.54 / 4 | 0.9825 / 0.9566 / 0.9919 / 0.9656 | 서술 |
| F10 판정 행 | `test=='F10'` | in_band / coverage(MAIN6) | 4 / 0.88 | 0.8847 | 부분 지지. ci_flag: 'hier2_cdf in [0.85,0.95]: 4/6; ref_cqr_ak<0.85: 5/6; regions_in_band=Russia_C,Russia_E,Canada,Lena' |

### 2.5 U4 라벨 0 정규화기 비교와 생성 분위 구간(LGU-B, 09-29 사전 등록)

원천 LB. 열 `delta / ci_lo / ci_hi`(셀 가중 2단 CI), `delta_beq / ci_lo_beq / ci_hi_beq`(블록 등가중). 풀은 주 4지역(MEAN4), n 0, 범위 all. LGU-B1 의 delta 열은 대비가 아니라 90 % 커버리지 수준이다.

| 가설 | 행 필터 | 셀 가중 | 블록 등가중 | 판정어(맹검 표지) |
|---|---|---|---|---|
| LGU-B1 nflow, cqr_pool λ 1.0 | `test=='LGU-B1' & contrast∋'cqr_pool:nflow@lam1.0'` | 0.76 [0.67, 0.83] (0.7568 [0.6715, 0.8262]) | 0.79 [0.70, 0.84] (0.7851) | 지지(재현(비맹검)) |
| LGU-B1 cfm, cqr_pool λ 1.0 | `test=='LGU-B1' & contrast∋'cqr_pool:cfm@lam1.0'` | 0.65 [0.56, 0.76] (0.6529 [0.5636, 0.7645]) | 0.68 [0.60, 0.80] (0.6844) | 지지(재현(비맹검)) |
| LGU-B2 구간 점수 nflow − 위약(nflow), 확인적 | `test=='LGU-B2'` | +3.29 [−4.17, 8.16] | +0.03 [−6.49, 5.52] | 미결정(맹검). verdict_3region 미결정, n_neg_regions 1, coverage_pool 0.89(0.8919) |
| LGU-B3 nflow − cbq | `test=='LGU-B3'` | −7.41 [−52.55, 10.00] | −1.46 [−61.11, 23.49] | 미결정(동등 검정 불가) |
| LGU-B4 nflow − const(보정 그대로) | `test=='LGU-B4' & contrast=='is10: nflow − const(보정 그대로)'` | +3.75 [−10.73, 12.91] | +4.06 [−7.02, 15.18] | 미결정 |
| LGU-B4 nflow − const(평균 로그 폭 정합) | `… contrast=='is10: nflow@mw − const(평균 로그 폭 정합)'` | +4.96 [−0.81, 9.00] | +0.72 [−3.80, 6.65] | 미결정 |
| LGU-B4 cbq − const(보정 그대로) | `… 'is10: cbq − const(보정 그대로)'` | +11.16 [−11.20, 53.46] | +5.52 [−24.01, 65.10] | 미결정 |
| LGU-B4 cbq − const(폭 정합) | `… 'is10: cbq@mw − const(평균 로그 폭 정합)'` | +5.24 [−5.34, 18.66] | +0.18 [−9.62, 12.90] | 미결정 |
| LGU-B4 cfm − const(보정 그대로) | `… 'is10: cfm − const(보정 그대로)'` | −1.70 [−17.91, 10.03] | −0.58 [−11.86, 10.63] | 미결정 |
| LGU-B4 cfm − const(폭 정합) | `… 'is10: cfm@mw − const(평균 로그 폭 정합)'` | +2.72 [−1.38, 6.88] | −1.00 [−4.62, 3.66] | 미결정 |
| LGU-B4 phys − const(보정 그대로) | `… 'is10: phys − const(보정 그대로)'` | −0.83 [−8.74, 2.54] | −4.64 [−12.13, 0.82] | 미결정 |
| LGU-B4 phys − const(폭 정합) | `… 'is10: phys@mw − const(평균 로그 폭 정합)'` | +0.78 [−4.02, 3.46] | −2.20 [−7.42, 1.80] | 미결정 |
| LGU-B6 nflow − const, n 10 / 40 / 160(범위 B, 레나·캐나다·알래스카) | `test=='LGU-B6' & contrast=='is10: nflow − const(범위 B, 중심 E_n)' & n==10/40/160` | +10.71 [4.45, 15.57] / +10.81 [2.97, 18.57] / +10.93 [2.83, 18.93] | +3.50 [−3.52, 10.42] / +3.25 [−3.01, 13.03] / +2.55 [−3.04, 12.06] | 미결정(셀 가중만 열세 쪽) |
| LGU-B6 cbq − const, n 10 / 40 / 160 | `… contrast=='is10: cbq − const(범위 B, 중심 E_n)'` | +8.37 [−8.95, 135.07] / +38.34 [1.30, 272.33] / +43.00 [3.33, 279.22] | −8.86 [−34.22, 135.12] / +18.05 [−24.13, 298.48] / +23.93 [−22.35, 300.43] | 미결정 |

정규화기별 수준 값(F6B, `target=='MEAN4' & method==<정규화기>`, 열 `cov10 / cov10_lo2 / cov10_hi2 / cov10_beq / wid10 / is10`; 2단 CI, 판정 없음):

| 정규화기 | cov10 | 2단 CI | cov10_beq | wid10(cm) | is10(cm) |
|---|---|---|---|---|---|
| const | 0.86 (0.8640) | [0.83, 0.90] | 0.85 | 81.29 | 131.21 |
| phys | 0.87 (0.8724) | [0.84, 0.90] | 0.87 | 82.98 | 130.38 |
| nflow | 0.89 (0.8919) | [0.86, 0.90] | 0.89 | 96.01 | 134.96 |
| nflow 위약 | 0.89 (0.8872) | [0.86, 0.90] | 0.88 | 94.95 | 131.67 |
| cfm | 0.88 (0.8850) | [0.86, 0.90] | 0.88 | 94.34 | 129.50 |
| cbq | 0.95 (0.9533) | [0.91, 0.97] | 0.95 | 132.44 | 142.36 |

const 의 지역 값(F6B `target==<지역> & method=='const'`, cov10): 레나 0.89(0.8898), 캐나다 0.89(0.8902), 러시아 W 0.75(0.7500), 러시아 E 0.93(0.9259). LGU 의 const 는 h37 hier2_cdf 와 같은 구조를 표본 밖 보정 점수(E0^(−k))로 다시 계산한 것이다. 두 표의 CI 를 나란히 비교하지 않는다(LGU 2.7, F6B_ref 의 use 열).

### 2.6 U5 라벨 있는 계층 예측 분포와 AB10(LGU-A, WRAPUP 1.1)

원천 LA. 풀은 레나·캐나다·알래스카(x), 구간 점수 α 0.1(cm), 2단 CI(주)와 1단 CI(보조).

| 가설 | 행 필터 | 열 | 값 | 판정어 |
|---|---|---|---|---|
| LGU-A1 (iii) − B4, n 10(확인적, AB10) | `test=='LGU-A1' & n=='10'` | delta / ci_lo / ci_hi | +9.51 / −0.75 / 22.11 | 미결정(맹검) |
| 〃 | 〃 | delta_beq / ci_lo_beq / ci_hi_beq | +8.64 / −2.55 / 22.04 | 〃 |
| 〃 | 〃 | verdict_1stage / ci_lo_1stage / ci_hi_1stage | 열세 / 4.55 / 13.02 | 보정 불확실성 의존 |
| 〃 | 〃 | eq_state / half_width / ref_score / coverage_pool | 검정 불가(정밀도 미달) / 12.29 / 99.03 / 0.90(0.9011) | 〃 |
| LGU-A1 n 40 | `test=='LGU-A1' & n=='40'` | delta / ci_lo / ci_hi | −2.80 / −12.50 / 7.89 | 미결정 |
| 〃 | 〃 | delta_beq / ci_lo_beq / ci_hi_beq | −2.48 / −13.71 / 8.58 | 〃 |
| 〃 | 〃 | verdict_1stage / ci_lo_1stage / ci_hi_1stage | 우세 / −4.95 / −0.56 | 보정 불확실성 의존 |
| 〃 | 〃 | eq_state / half_width / ref_score / coverage_pool | 검정 불가(정밀도 미달) / 11.15 / 105.46 / 0.90(0.9017) | 〃 |
| LGU-A1 종합 | `test=='LGU-A1' & n=='10,40'` | verdict | 미결정 | 미결정 |
| 같은 대비의 % 환산(그림 표) | F6C `test=='LGU-A1' & n==10`, `n==40`; `test=='LGU-B2' & n==0` | delta_pct / ci_lo_pct / ci_hi_pct | n 10: 9.60 / −0.76 / 22.33; n 40: −2.66 / −11.85 / 7.48; B2: 2.50 / −3.17 / 6.19 | 미결정 |
| AB10 의 두 가중 p | AB `lgu_a_ab10_tests.csv`: `test_id=='LGU-A1'` | delta / p_cell / p_beq / p_eq / verdict4 | 9.51 / 0.07 / 0.09 / 0.80 / 미결정 | 미결정 |
| AB10 묶음 행 | AB `lgw_bundle.csv`: `ab=='AB10'` | delta / holm_p / holm_p_eq / holm_m / verdict4 / abstract_rule | 9.51 / 0.28 / 1.00 / 10 / 미결정 / '(d) 차이를 확인하지 못했다' | 미결정(J7) |
| LGU-A3 (iii) 중앙값 − P1, RMSE, 주 4지역 n 3 | `test=='LGU-A3' & pool=='주 4지역' & n=='3'` | delta / ci_lo / ci_hi; delta_beq / ci_lo_beq / ci_hi_beq | +1.43 / 0.68 / 1.81; +1.73 / 0.96 / 2.16 | 열세 |
| 〃 n 10 | `… n=='10'` | 같음 | +1.60 / 0.57 / 1.88; +1.65 / 0.76 / 2.13 | 열세 |
| 〃 러시아 W 제외 3지역 n 3 / n 10 | `test=='LGU-A3' & pool=='러시아 W 제외 3지역'` | delta / ci_lo / ci_hi | +0.90 / 0.41 / 1.68; +1.08 / 0.32 / 1.80 | 열세 |
| LGU-A7 (iv) − (iii), 구간 점수, n 10 | `test=='LGU-A7' & contrast=='iv − iii' & n=='10' & metric=='is10'` | delta / ci_lo / ci_hi | +25.23 / 17.46 / 34.02 | 열세 |
| LGU-A2 무한 구간 비율, n 3(5단) | `test=='LGU-A2'` 5행 | delta(inf10) 최댓값 | 0.00 | 통과(무한 구간 0) |

p 값은 둘째 자리로 반올림했다. 넷째 자리는 p_cell 0.0721, p_beq 0.0922, p_eq 0.8046, holm_p 0.2766 이다.

### 2.7 U6 라벨 있는 R1 구간(LGX L25, 방향 중립 등록)

원천 X, 행 필터 `test_id=='L25' & scope=='region' & target==<대상> & n==<n>`, 열 `coverage / width_cm / width_n0_r0 / in_band / narrower`.

| 대상 | n | coverage | width_cm | width_n0_r0(R0, n 0) | in_band | narrower |
|---|---|---|---|---|---|---|
| Lena\|x | 40 | 0.89(0.8940) | 67.88 | 74.62 | 1 | 1 |
| Lena\|x | 160 | 0.90(0.8984) | 72.64 | 74.62 | 1 | 1 |
| Canada\|x | 40 | 0.85(0.8475) | 92.39 | 104.84 | 0 | 0 |
| Canada\|x | 160 | 0.84(0.8351) | 86.00 | 104.84 | 0 | 0 |
| Alaska\|x | 40 | 0.89(0.8914) | 56.60 | 93.20 | 1 | 1 |
| Alaska\|x | 160 | 0.90(0.8982) | 51.82 | 93.20 | 1 | 1 |

판정 행 `test_id=='L25' & scope=='verdict_aux'` 의 verdict 는 '라벨이 구간을 좁힌다'(c1 = 4, c2 = 4)다. 원고는 이 문장을 쓰지 않고 커버리지와 폭 비교 값만 쓴다(WRAPUP 14:20 기록, LGU 10절).

### 2.8 H15(대체된 기록, 참고)

`docs/EXPERIMENT_LOG.md` 의 H15 행('알래스카 지역 내 90% 구간 커버리지 cfm 0.70·ddpm 0.62·nflow 0.67 … 전이 4지역 커버리지 0.28–0.40')은 원천 SC 에서 단일 행으로 찾지 못했다([미확인]). 직접 모드 행(`cond=='labels' & target=='Alaska' & anchor=='none' & resid==<학습기> & seed=='mean'`, 열 coverage / cov_ci_lo / cov_ci_hi / width_mean / interval_score)은 cfm 0.77(0.7651) [0.59, 0.86] / 36.03 / 89.49, ddpm 0.65(0.6539) [0.56, 0.73] / 27.40 / 90.55, nflow 0.63(0.6308) [0.51, 0.78] / 33.22 / 111.20 이다. 판정은 '기각'이었고 LGU 4.6·11.1 에 따라 원고의 생성 모델 구간 서술은 LGU-B1 로 바꾼다. H15 수치를 원고에 쓰지 않는다.

## 3. 뒷받침 실험

| 새 이름 | 옛 id | 내용 | 스크립트 | 등록 | 실행 환경 |
|---|---|---|---|---|---|
| T5_prediction_intervals | LGU(A1–A7, B1–B6, C1–C4) | 계층 예측 분포 사다리, 정규화기 비교형 계층 conformal, 공간 정보량 진단 | `scripts/2_evaluation/h44_hier_predictive_ladder.py`, `h45_normalized_conformal.py`, `h46_spatial_diagnostics.py`, `src/polar/lgu_common.py` | 09-29 사전 등록(B1 은 재현(비맹검)) | 로컬 RTX 3090(신경망 정규화기 nflow·cfm 포함) |
| T2_transfer_structure_placebo_baselines | LGX L25(X5 구간) | 라벨 있는 R1 의 교차 적합 conformal 구간 | `scripts/3_deep_learning/h42_label_grid_ext.py`(산출 `lgx_conformal.csv`) | LG 6A, 방향 중립 | Rescale 조각, 로컬 재집계 |
| T6_abstract_contrasts_and_map | LGW AB10, 레나 지도(SI) | 초록 묶음의 구간 점수 대비, 지도 패널 (d) 구간 폭(const) | h39 집계(`lgw_bundle.csv`), `scripts/2_evaluation/h49_transfer_map.py` | WRAPUP 1.1, 6.6–6.8 | 로컬 |
| H5_final_A2-D1 | 09-26 실험 'C2'(F10) | 라벨 0 계층 conformal(hier2_cdf·sub·blk·exact), 라벨 3·10 의 비교환성 가중 conformal(wconf) | `scripts/2_evaluation/h37_conformal_hier.py` | 09-26 사전 등록 | CPU |
| H2_master_M1 | H17(전이 UQ), H15(생성 모델 UQ) | 알래스카 보정 CQR·밀도비 가중·앵커 구간의 전이 커버리지, 생성 모델 구간 | `scripts/3_deep_learning/m1_transfer_uq.py`, M1 S-C 하네스(`m1_sc_uq.csv`) | 09-21 개정 F(H17, 커밋 기준 사전 등록 확인 불가: 개정과 결과가 같은 커밋 a5970f4, 1.2 참조), 09-16(H15) | CPU(H17) |
| H1_prereg_E1-E4 | E4.3 | S11 OOF 의 구간 점수, 조건부 커버리지 | `scripts/3_deep_learning/e4_interval_score.py` | 09-08 계획(S11 결과 뒤 재분석) | CPU |
| H0_contest | S11 | 분위 CatBoost + CQR(알래스카 지역 내) | `scripts/3_deep_learning/s11_conformal_uq.py` | 없음(대회 단계) | CPU |

## 4. 그림

- **본문 Fig 6 v2**(`outputs/figures/paper/v2/Fig6_label0_uncertainty.{pdf,svg,png}`). 이 항목의 패널은 b, c, d 다(패널 a 는 C1).
  - b: 라벨 0 구간의 커버리지와 평균 폭, 정규화기 6종(const, phys, nflow, 위약 nflow, cfm, cbq)의 주 4지역 평균과 지역 점, h37 hier2_cdf 참조 원(범례 'C2 interval, earlier protocol'). 원천 `lgu_b_intervals.csv`, `h4/c2_coverage.csv`. CI 는 그리지 않는다.
  - c: 구간 점수 대비 포레스트(LGU-B2 n 0, LGU-A1 n 10·40, AB10 표지, ±5 % 띠, 2단·1단 판정이 다르면 ‡).
  - d: L25 의 6칸, R1 커버리지 대 R0(n 0) 폭 비.
  - 원천 값: `outputs/figures/paper/source_data/v2/Fig6_{b,b_ref,c,c_registered,d,d_registered}.csv`. 값 대조: `outputs/figures/paper/_qa/v2_Fig6_label0_uncertainty_values.txt`(252행, 원천 표 직접 대조). 그림 표 생성 기록 `data/processed/paper_figs/v2_data_fig6_meta.json` 의 입력 sha256 7건(lgw_bundle, lgx_tests, lgu_b_intervals, lgu_b_tests, lgu_a_tests, c2_coverage, lgx_conformal)은 `tables/MANIFEST.csv` 의 사본 sha256 과 같다(이번 점검).
  - 캡션: `outputs/figures/paper/CAPTIONS.md` 'v2/Fig6_label0_uncertainty' 절.
- **본문 Fig 7 v2 패널 d**(`outputs/figures/paper/v2/Fig7_deployment.*`): AB10 을 다른 축(구간 점수, cm)에 둔다. 원천 `data/processed/paper_figs/fig7_d_ab10.csv`.
- **재구성안**(`docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md`): 결과 R8 '예측 구간'은 보조 주장이고 그림은 Fig 6b–d 다. Fig 6 은 '그대로'(변경 없음)다. 6절은 'LGU 세부'와 레나 지도를 SI 로 옮긴다. `figures/figure_spec.json` 의 SI 항목은 SF_LGU_A(사다리 A2–A7, 분산 성분), SF_LGU_B(B1, B3–B6, 층별 진단), SF_LGU_C(공간 진단), SF_intervals_labelled(L25 칸별 세부)다. LGU-B2 가 미결정이므로 지도는 SI 한 장이고 캡션에 '구간 폭은 예측값에 비례한다'를 쓴다(LGU 11.1, figure_spec 조건부 규칙 fig6_map_merge).
- **v2 에 없는 근거**: 지역 내 S11·E4.3 과 전이 H17 은 v2 그림에 없다. 옛 그림은 `outputs/figures/s11_uq/s11_calibration_curve.*`, `s11_uq_maps.*`(대회), `outputs/figures/m1/transfer_uq_coverage.*`, `transfer_uq_width_coverage.*`(H17), `outputs/figures/paper/Fig6_label0.*`(09-26 설계, 패널 d 가 h37 계층 conformal 과 H17 참조), 09-26 SI Table 6c(h37)다. 권고: SI 표 하나에 '지역 내 CQR(S11) → 전이 CQR(H17) → 원천 셀 교환성 생성 구간(LGU-B1) → 계층 conformal(h37, LGU const)'의 커버리지를 한 줄씩 놓는다. 이 표는 새 계산 없이 2절 값으로 만들 수 있다.

## 5. 단서와 쓰지 않을 문장

### 5.1 단서

1. **세 모형 계열**: 지역 내 S11·E4.3 은 직접 CatBoost 분위 모형(x34), H17 은 같은 계열의 x25 판, h37·LGU-B 는 물리 앵커(E0·s) 중심, LGU-A 는 재보정 계수 중심의 계층 모형, L25 는 R1 중심이다. 수치를 한 문장에서 이어 쓸 때 중심 모형을 밝힌다.
2. **상수 폭 참조의 유리함**: E4 의 상수 폭 반폭 21.73 cm 는 채점 셀 자신의 잔차 분위다. 비교는 CQR 에 불리하다. E4 는 S11 의 3 seed 앙상블 OOF(seed-mean 구간, 분위 경계 평균)를 다시 채점한 것이다(`s11_conformal_uq.py` 15행, 170–178행). 원천 `e4_interval_score_meta.json` 의 caveat('3 seed 중 파일에 저장된 단일 세트')와 이를 옮긴 `extract_evidence.py` 117행 노트, `evidence_values.csv` 의 U1-E4 행 note 열은 이 점에서 부정확하다. 같은 예측의 커버리지 CI 가 S11 과 E4 에서 다른 것은 부트스트랩 추출 차이이므로 원고에는 CI 를 하나만 쓴다.
3. **보정 집단 수**: h37 의 주 4지역 대상은 보정 집단 K = 4 다. 정확형 hier2_exact 는 폭이 무한이었고(K + 1 < 1/α), hier2_cdf 의 커버리지는 경험값이다. LGU 도 보정 집단 2–5개다(LGU 2.3: n ≥ 40 에서 2개).
4. **지역 편차**: 계층 conformal 과 LGU const 모두 러시아 W 에서 0.75 였다. 러시아 W 는 원천 계수 대비 수준 편향이 큰 지역이고 채점 셀이 28개(18블록)다. 레나 hier2_cdf 의 블록 등가중 커버리지는 0.81 이다.
5. **규약 차이**: h37 은 보정 점수를 E0 기준(표본 안)으로, LGU 는 E0^(−k) 기준(표본 밖)으로 둔다. 두 표의 CI 를 나란히 비교하지 않는다(LGU 2.7).
6. **보정 불확실성 의존**: LGU-A1 은 2단(보정 쪽 재계산 포함)과 1단 CI 의 판정이 n 10 에서 열세, n 40 에서 우세로 갈린다. 확인적 판정은 2단(미결정)이다.
7. **LGU-B6 는 '동등'이 아니다**: nflow − const 는 셀 가중 CI 가 세 n 모두 0 을 넘어(나쁜 쪽) 블록 등가중 CI 만 0 을 포함한다. '차이가 없다'로 쓰지 않는다.
8. **점 예측**: 계층 모형 단 (iii)의 중앙값은 P1 보다 RMSE 가 1.43–1.60 cm 컸다(LGU-A3 열세). 구간 모형의 중앙값을 점 예측으로 쓰지 않는다.
9. **실행 환경**: LGU 의 nflow·cfm 정규화기는 로컬 GPU 산출이다. 신경망은 환경에 따라 값이 달라지므로(재현성 규칙) LGU 행과 Rescale 행 사이의 차를 계산하지 않는다. S11, E4, H17, h37 은 CPU 산출이다.
10. **표기 충돌**: 09-26 문서·캡션의 'AB4'(주 4지역 평균)와 'C2'(h37 실험)는 WRAPUP AB4, 주장 C2 와 다르다.
11. **[미확인] 값과 재계산 값**: 폭·오차 순위 상관 −0.07(RESULTS_RECONCILIATION 3.7)은 `s11_conformal_oof.csv` 에서 재계산한 값(−0.0722)으로 [미확인]을 푼다. E4 차이의 CI [0.6, 6.5] 는 정확히 재현되지 않는다. 같은 원천의 짝지은 블록 부트스트랩 CI 는 추출에 따라 [0.39, 6.64]–[0.78, 6.50] 범위로 달라지며 모두 0 을 넘는다(2.2절 읽는 법). 쓰려면 고정 seed 스크립트로 다시 내고 값을 기록한다. 상수 폭 참조가 사후 oracle 이라는 단서는 유지한다. H15 의 0.70·0.62·0.67 과 전이 0.28–0.40 은 원천 표에서 확인하지 못했다([미확인]). 원고에 쓰려면 원천을 먼저 찾거나 다시 계산해야 한다.

### 5.2 쓰지 않을 문장

| 문장 | 사유 |
|---|---|
| 커버리지가 보장된 구간, 커버리지가 보장된 지도 | LGU 10절, WRAPUP 11.2. 보정 집단 2–5개, hier2_exact 무한 |
| 안전한 구간, 믿을 수 있는 구간 | 구간에 등록된 비열등 판정이 없다(QA 공통 축소) |
| 물리식보다 나은 불확실성 | QA Q7. 구간 점수 우세 판정이 하나도 없다 |
| 계층 모형이 구간을 개선한다 | LGU 11.1 원고 문장(LGU-A1 미결정) |
| 조건부 폭이 지역 간에 전이된다 | LGU-B2 미결정(LGU 10절, 4.4) |
| 라벨이 구간을 좁힌다 | LGU 10절(폭 감소는 모형 구조에서 자동 발생), WRAPUP 14:20 기록. L25 판정 기록은 유지 |
| 라벨로 구간을 개선한다(AB10 의 해석으로) | AB10 은 계층 예측 분포 대 상수 폭 보정 구간의 대비이고 판정은 미결정이다 |
| 계층 모형이 라벨 0 에서 목표 커버리지를 준다 | LGU 10절. 라벨 0 예측 분포는 초사전분포 선택이 정한다 |
| 생성 모델이 점 예측 오차를 줄인다, 확산·흐름 정합이 분위 회귀보다 낫다, 생성 학습기와 분위 회귀는 동급이다 | LGU 10절. LGU-B3 은 동등 검정 불가 |
| 흐름이 잔차 분포를 개선한다 | LGU 10절. 게이트(LGU-A4)는 실행 여부 기준이다 |
| 구간 폭 지도로 어느 지역을 신뢰할지 판단한다 | RESULTS_RECONCILIATION 3.7, 지도 정규화기 const(폭은 예측값에 비례) |
| AOA 마스크가 오차 위험을 보인다 | WRAPUP 11.2 |
| CQR 이 상수 폭 구간보다 유의하게 나쁘다 | 짝지음 CI 가 원천 표에 없다. 재계산 CI 는 0 을 넘지만 추출 의존이고 기록 파일이 없으며, 상수 폭 참조는 채점 셀 잔차로 정한 사후 oracle 이다. 점 추정 68.85 대 65.39 만 쓴다 |
| 계층 conformal 이 커버리지를 회복했다(지역 단서 없이) | 러시아 W 0.75, 레나 블록 등가중 0.81, F10 부분 지지 |
| 공간 상관이 없다 | LGU 10절. 실험 C 는 CI 없는 진단이다 |
| C2 구간, C2 interval | 주장 C2 와 혼동된다. 'hierarchical conformal (09-26 protocol)'로 쓴다 |

## 6. 이 주장을 바꿀 수 있는 계획 실험(X*)

X* 의 등록 계획서는 아직 없다. 구현·설계 초안은 `docs/research/2026-10-04/` 에 있다(`harness_implementation_plan.md` 'XA–XJ 하네스 구현 계획', `placement_and_workflow_algorithm.md` XC·XD 사전 등록용 설계). 두 문서는 등록 계획서가 아니므로 아래는 이 항목에 미칠 영향의 범위다.

| 계획 실험 | 이 항목과의 관계 | 바뀔 수 있는 것 |
|---|---|---|
| XF_new_regions | 직접 해당. 계층 conformal 의 보정 집단은 독립 지역이다. 사용자 질문 1(공개 자료로 독립 지역 추가)과 같은 축이다 | 보정 집단이 9개 이상이면(K + 1 ≥ 1/α = 10) 정확형 계층 conformal 이 유한한 폭을 가져 유한표본 보장을 말할 수 있다(c2_meta note_exact 의 조건). LGU 9절의 GU3 재심 조건('독립 지역 9개 이상')도 같다. 주 4지역 평균 0.86 과 지역 범위(0.75–0.93), LGU-A1·B2 의 검정력이 바뀐다 |
| XJ_tempderived_aux_labels | 지온 유도 라벨(몽골 등)은 정의가 달라 보정 집단에 넣으면 잔차 분포가 섞인다 | 보조 라벨 지역을 보정 집단에 넣을지 규칙이 필요하다. 넣으면 F10·LGU const 의 커버리지가 바뀐다 |
| XE_hires_covariates | 사용자 질문 3(토양 수분, 유기층, 식생, 적설). LGU-B2·B4 는 셀별 폭의 이득을 보이지 못했고 E4 의 폭 5분위는 오차를 가려내지 못했다 | 격자 안 입력이 생기면 nflow·cbq 정규화기를 다시 학습해 LGU-B2 형식 대비를 새로 등록해야 한다. '조건부 폭이 전이된다'를 쓸 수 있는 유일한 경로다 |
| XB_multisource_stacking | 구간의 중심이 P0·E_n 에서 적층 예측으로 바뀌면 보정 점수가 바뀐다 | 적층을 권장 중심으로 쓰면 계층 conformal 을 그 중심으로 다시 보정해야 한다. 현재 수치는 그대로 남는다 |
| XC_workflow_end_to_end | 워크플로의 각 단계(라벨 0, 1–10, 10–160 …)에 구간 단계를 넣을지 | 라벨 수별 커버리지와 폭을 같은 시험지에서 보고하면 1.3 문장이 워크플로 표의 한 열이 된다. '안전' 표현은 비열등 판정을 따로 등록해야 쓸 수 있다 |
| XH_validation_ladder | 지역 내 0.93(S11)과 전이 0.32–0.73(H17)은 검증 설계가 다른 두 점이다 | 같은 모형의 커버리지를 무작위·블록·kNNDM·지역 홀드아웃 사다리로 내면 '검증 설계가 커버리지를 바꾼다'를 한 그림으로 보일 수 있다 |
| XI_climate_extrapolation_retest | 학습 범위보다 따뜻한 블록의 커버리지는 시험하지 않았다(LGU 9절: AOA 안팎 조건부 커버리지 제외, 최근접 원천 거리 3분위로 대신) | 외삽 블록의 조건부 커버리지 행이 생긴다. 외삽 영역 블록이 8개 이상이어야 CI 를 낼 수 있다(C8, WF10 의 설계 점검 누락) |
| XG_product_comparison | 공개 ALT 제품의 불확실성 층과의 비교 | 제품이 불확실성 층을 주는지는 [미확인]이다. 준다면 같은 채점 셀의 커버리지를 비교 행으로 넣을 수 있고, 누설 점검(제품 학습 셀)이 먼저다 |
| XD_placement_policy | 라벨 위치가 n > 0 구간의 보정에 주는 영향 | h37 wconf 는 k-medoid 와 무작위 배치를 모두 돌렸다(`c2_coverage.csv` rule 열). 정책 학습이 추가되면 배치별 커버리지 행을 더할 수 있다. 영향은 작을 것으로 본다(근거 없음, 추정) |
| XA_c2_gain_decomposition | 이 항목의 수치를 바꾸지 않는다 | 주장 C2 의 이름이 정해지면 Fig 6b 범례의 'C2' 표기를 함께 바꾼다 |

## 7. 이 폴더의 파일

| 파일 | 내용 |
|---|---|
| `README.md` | 이 문서 |
| `extract_evidence.py` | 원천 표에서 근거 수치를 읽어 `evidence_values.csv` 를 만들고 `tables/` 사본과 MANIFEST 를 쓴다(읽기 전용, 적합 없음) |
| `evidence_values.csv` | 근거 수치 501행: id, group(U1–U6, H15), source, row_filter, column, value(둘째 자리), value_4dp, value_raw, verdict, verdict_doc, note |
| `tables/` | 원천 집계 표 사본 28개(모두 5 MB 이하, 셀 단위 라벨 자료 없음). 사본 이름은 상위 폴더를 접두어로 붙였다(예: `lgu__lgu_b_tests.csv`) |
| `tables/MANIFEST.csv` | orig_path, copy_path, sha256, rows, bytes. 사본과 원본의 sha256 일치를 스크립트가 확인했다 |

복사하지 않은 것: `s11_conformal_oof.csv`, `m1/transfer_uq_cells.csv`, `lgu/lgu_b_cells.csv`(셀 단위 예측), `lgu/*.npz`(재표집 분포).

## 8. 정정 기록(2026-10-04)

검증 지적 8건을 원천과 다시 대조한 뒤 고쳤다. `extract_evidence.py`, `evidence_values.csv`, `tables/` 와 MANIFEST 는 바꾸지 않았다(사본 표의 sha256 은 원천과 같다).

| 위치 | 고친 내용 | 대조한 원천 |
|---|---|---|
| 1.2 '0.86 [0.80, 0.92]' 행, 2.4 머리 | '셀 가중 평균'을 '지역 등가중 평균(지역 안은 셀 가중)'으로 고치고 셀 수 가중 0.8878 [0.8214, 0.9468], 블록 등가중 0.8531 을 함께 적었다 | `h4/c2_coverage.csv` MEAN6 hier2_cdf 행(coverage, coverage_cellw, coverage_beq), `h37_conformal_hier.py` 283–303행, `v2_data_fig6_meta.json` checks, CAPTIONS.md 11행 |
| 2.2 표 'E4 단일 세트' 행, 5.1 단서 2 | E4 는 S11 의 3 seed 앙상블 OOF 를 다시 채점한 것이다. CI 차이는 부트스트랩 추출 차이다 | `s11_conformal_uq.py` 15행·148–178행, `s11_conformal_results.csv`, `e4_interval_score.csv`, RESULTS_RECONCILIATION 3.7 |
| 1.3 권고 문장(한국어·영어) | LGU-B2(nflow 대 위약), LGU-B4(정규화기 대 상수 폭), LGU-A1(계층 예측 분포 대 보정 상수 폭 B4)의 비교 대상을 구분했다 | `lgu/lgu_b_tests.csv` LGU-B2·B4 행, LGU 계획서 11.1, 원고 초안 결과 96행 |
| 1.2 'WF 사후 표지' 행, 2.3 머리, 3절 등록 열 | H17 '등록 뒤 실행'을 '커밋 기준으로 사전 등록 확인 불가(개정과 결과가 같은 커밋 a5970f4)'로 고쳤다 | `git show 291e2c7:docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md`(H17 0건), `git log a5970f4`, `transfer_uq_meta.json` git_commit, `transfer_uq_summary.csv` 기록 시각, FRONTMATTER_DRAFT 197행 |
| 2.3 본문 | cqr_pseudo·anchor_pseudo·anchor_real 은 `evidence_values.csv` 에 없고 사본 표에만 있음을 밝히고 원천 값을 적었다 | `evidence_values.csv`(해당 방법 0건), `m1/transfer_uq_summary.csv` |
| 2.2 표 'seed 3 평균' | '3 seed 앙상블 구간(분위 경계 평균)'으로 고쳤다. seed 별 커버리지 평균(0.9042, 0.4145)과 다름을 읽는 법에 적었다 | `s11_conformal_uq.py` 148–162행, `s11_conformal_results.csv` seed 0–2 행 |
| 1.2 '68.9 대 65.4' 행, 2.2 표 차이 행과 읽는 법, 5.1 단서 11, 5.2 표 | −0.07 은 재계산(−0.0722)으로 [미확인]을 풀었다. [0.6, 6.5] 는 정확히 재현되지 않고 추출 의존임을 적었다 | `s11_conformal_oof.csv`(13,606행, 스레드 1개 재계산), `e4_interval_score_meta.json` half_width_cm, `e4_interval_score.py` 44–52행(`RandomState(0)` 블록 부트스트랩) |
| 6절 머리 | 'X* 계획서 없음'을 '등록 계획서는 없고 구현·설계 초안은 `docs/research/2026-10-04/` 에 있다'로 고쳤다 | `docs/research/2026-10-04/harness_implementation_plan.md`(13:12:42), `placement_and_workflow_algorithm.md`(13:10:37) |
