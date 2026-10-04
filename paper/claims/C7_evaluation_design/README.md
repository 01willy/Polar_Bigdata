# C7 평가 설계(보조 주장) 근거 색인

- 작성: 2026-10-04. 성격: 논문용 색인이다. 원본 스크립트, 자료, 문서는 옮기거나 고치지 않았다.
- 수치 원본: 근거 표(2.2–2.9절)의 수치는 `extract_evidence.py` 가 원천 집계 표를 pandas 로 읽어 만든 `evidence_values.csv` 에서 옮겼다. `evidence_values.csv` 에 없는 값(메타 JSON 의 값, 계획서·로그 산문의 값, 이번 점검의 재계산, 문헌 서지)은 2.10절에 값마다 출처를 적었다. 재실행: `OMP_NUM_THREADS=1 python paper/claims/C7_evaluation_design/extract_evidence.py`(적합 없음, 수 초).
- 2026-10-04 점검 반영: 원천 대조로 확인된 11개 항목(단서 2 의 범위 하한, 1.3 권고 문장의 LGF-N2·'통합 자료'·'공간 분리 단조' 표현, 머리말 출처 선언, 단서 11 의 문헌 목록, 5.2 의 LGF-N2 대체 문장, 3절 X-ak 코드, 2.2 역할 표기, G3 와 E3 값 차이, 4절 Q15 인용, 단서 10 의 일반화)을 고쳤다. `tables/` 사본과 `MANIFEST.csv` 는 원천과 sha256 이 같아 바꾸지 않았다.
- 원천 표 사본: `tables/`(sha256 은 `tables/MANIFEST.csv`).
- 판정어: 4분 판정(우세, 열세, 동등, 미결정). 우세·열세는 셀 가중과 블록 등가중 두 95 % CI 가 모두 같은 쪽일 때, 동등은 두 CI 가 모두 ±0.5 cm 안일 때, 미결정은 그 밖이다(`docs/QA_FINAL_REVIEW_2026-10-02.md` 머리말). 검증 사다리의 대비 A − B 에서 '열세'는 A 단의 오차가 B 단보다 크다는 뜻이다.

## 1. 주장 문장

### 1.1 현행 문장

`docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md` 82행(원문):

> C7(보조). 평가 설계가 결론을 바꾼다. 원천 지역 하나 제외 교차검증으로 신경망 초모수를 고르면 전이 오차가 줄지 않았고(LGF-N2), 원천 교차검증 이득과 대상 오차 변화는 반대 방향이었다(순위 상관 −0.32, LGF-N4 서술). 방법 선택을 원천이 아니라 대상 라벨 안 교차검증으로 하는 이유다(C5). 무작위 분할의 오차는 지역 홀드아웃보다 10.3 cm 작다(L19·L20). 대회 보고서 같은 지역 내 무작위·공간 교차검증의 우위가 새 지역에서 그대로 나타나지 않는 이유 가운데 하나다.

`docs/RESEARCH_OVERVIEW_2026-10-02.md` 160–163행은 같은 주장에 '높은 정확도를 보고한 선행 연구 대부분은 무작위 교차검증이었다'를 더한다.

### 1.2 필요한 축소

`docs/QA_FINAL_REVIEW_2026-10-02.md` 의 공통 축소 항목과 등록 규칙을 C7 에 적용한 결과다.

| 항목 | C7 에 대한 적용 |
|---|---|
| C2 축소(재보정 몫과 ML 몫의 분리, QA Q1 C2 점검) | C7 의 수치는 같은 방법의 조정판 대 기본판 비교(LGF-N2, 라벨 0·10·전량)와 고정 자료의 검증 방식 비교라 재보정 혼입 문제가 없다. C7 을 C2 의 근거로 끌어 쓰지 않는다 |
| '안전' 표현(등록된 비열등 판정이 있을 때만) | C7 근거에는 비열등 판정이 없다. 'kNNDM 이 가장 정직한 추정이다', '우리 채점은 과하게 어렵지 않다'(QA Q15) 같은 평가어 대신 거리와 RMSE 수치를 쓴다 |
| WF 사후 설계 표지 | C7 근거에 WF 결과는 없다. L19 는 확인적 가설이지만 그 판정 대비는 D0 − PS@V-R·V-G(A10, A11, T 의 role=주)이고, C7 이 인용하는 열화 크기 V-G − V-R(A1–A9, A12, A13)은 role=보조 행이다(LG 7.2 결정 5 에 따른 보고). L20·L21·LGF-N2 는 보조, LGF-N4 는 서술(검정 없음), 알래스카 검증 방식 비교(M1 부록, 09-21)는 가설 없는 서술, X-ak 대비는 탐색(X, family=exploratory)이다. 대신 L19·L20 에는 '재현(비맹검)' 표지가 붙는다(LG 계획서 6A.7: 2026-06 분할 누설 표와 M1 알래스카 검증 방식 비교로 방향을 이미 보았다) |
| L19 등록 규칙 | L19 판정은 '지지하지 않음'이다(V-R 에서 직접 ML 의 우세가 블록 등가중 CI 로 확인되지 않음). 등록 규칙에 따라 '역전'을 쓰지 않고 열화 크기만 쓴다(LG 7.2 결정 5) |
| 10.3 cm 의 범위 | 직접 ML(`catboost_lo` D0) 의 셀 가중 5지역 층화 평균(MEAN5: 지역마다 셀 가중 RMSE 를 내고 다섯 지역을 등가중 평균한 값)이다. 셀 풀 RMSE 가 아니다. 셀 풀(POOL)로는 V-R 13.54, V-G 33.67, 차 +20.13 이다(A4). 블록 등가중으로는 +6.15, 알래스카 제외 4지역은 +7.12(블록 등가중 +2.98)이다. 같은 사다리에서 Stefan 최소제곱(PS)은 26.81–26.96 cm 로 거의 변하지 않는다. 따라서 '무작위 분할의 오차'가 아니라 '직접 ML 의 오차'로 쓰고, 'pooled'·'통합 자료' 대신 '지역 등가중 평균'으로 적는다 |
| LGF-N2 의 학습기 일반 문장 | 현행 C7 의 '원천 지역 하나 제외 교차검증으로 신경망 초모수를 고르면 전이 오차가 줄지 않았고(LGF-N2)'는 학습기 일반 문장이다. LGF 10.3 판정은 '학습기별로 다르다'이고 강건 문장 규칙을 채운 학습기가 없다(H4). MLP·다중 헤드 MLP 의 직접 ML 에서는 조정이 전이 오차를 키웠고(H1, H2), 축소 FT-T 의 라벨 전량 잔차에서는 조정판이 작게 나았다(H3, 0.5 cm 미만). 학습기와 대비(D0·R1, 라벨 수, λ)를 붙여 쓴다(5.2) |
| 초록 | WRAPUP 1.1 은 L19 를 초록 묶음에서 뺐다(평가 규약의 근거라 본문 방법·토의에서 쓴다). 초록에 판정어를 붙이지 않는다 |

### 1.3 권고 문장

한국어:

> C7(보조, 방법의 근거). 검증 설계는 직접 ML 의 오차를 크게 바꾸고 물리식의 오차는 거의 바꾸지 않는다. 다섯 지역 RMSE 의 지역 등가중 평균(셀 가중 5지역 층화 평균)으로 보면, 직접 ML(CatBoost)의 RMSE 점 추정은 설계상 분리 단계(여덟 단) 순서로 단조 증가해 셀 무작위 5-fold 22.91 cm 에서 지역 홀드아웃 33.25 cm 가 되었다(차 +10.34 cm [7.29, 12.84], 블록 등가중 +6.15 [4.70, 7.57]). 등록 대비인 500 km 완충 공간 군집 제외와 셀 무작위의 차는 +8.22 cm [5.65, 10.38](블록 등가중 +4.49 [3.15, 5.85])이다. 같은 사다리에서 Stefan 최소제곱의 RMSE 는 26.81–26.96 cm 였다. 알래스카 안에서도 무작위 셀 분할은 시험 셀 바로 옆(가장 가까운 학습 셀 중앙값 0.004 km)에 학습 셀을 둔다. 직접 CatBoost 의 RMSE 는 무작위 셀 11.55 cm 에서 예측 격자 거리 분포에 맞춘 kNNDM 17.91 cm 로 커졌고, Stefan 은 다섯 방식에서 14.24–14.46 cm 였다(seed 3개 점 추정). 원천 지역 하나 제외 교차검증으로 신경망 초모수를 고르면, MLP·다중 헤드 MLP 의 직접 ML 에서는 전이 오차가 커졌고(라벨 0: +1.51 [0.87, 2.25], +1.33 [0.89, 1.81] cm), 축소 FT-T 의 라벨 전량 잔차에서는 작게 줄었다(λ 0.25: −0.19 [−0.50, −0.06] cm, 0.5 cm 미만). 강건 문장 규칙을 채운 학습기는 없다(LGF-N2). 그래서 이 연구는 방법 선택을 대상 라벨 안의 공간 분리 교차검증으로 하고, 성능을 지역 홀드아웃으로 보고한다.

영어(원고 방법·토의용 초안):

> Validation design changed the error of direct ML far more than that of the physics model. Averaged over the five regions with equal weight (cell-weighted RMSE within each region), the point estimates of the direct-ML (CatBoost) RMSE increased monotonically along the designed separation steps, from 22.91 cm under random 5-fold cells to 33.25 cm under region holdout (difference +10.34 cm [7.29, 12.84]; block-equal +6.15 [4.70, 7.57]). The registered contrast between spatial-cluster holdout with a 500 km buffer and random cells was +8.22 cm [5.65, 10.38] (block-equal +4.49 [3.15, 5.85]). The least-squares Stefan model stayed at 26.81–26.96 cm on the same ladder. Selecting neural-network hyperparameters by leave-one-source-region-out cross-validation increased the transfer error of direct ML with the MLP and the multi-head MLP (no target labels: +1.51 cm [0.87, 2.25] and +1.33 cm [0.89, 1.81]) and slightly reduced that of the residual reduced FT-Transformer with all target labels (λ = 0.25: −0.19 cm [−0.50, −0.06], below 0.5 cm); no learner met the robustness rule (LGF-N2). We therefore select methods by spatially separated cross-validation within target labels and report region-holdout errors.

## 2. 근거 표

### 2.1 원천 약호

| 약호 | 원천 경로 | 사본 |
|---|---|---|
| T | `data/processed/lgx/ladder/lgv_tests.csv`(h41 집계, L19–L22) | `tables/lgv_tests.csv` |
| M | `data/processed/lgx/ladder/lgv_metrics.csv`(h41 단별 지표) | `tables/lgv_metrics.csv` |
| CV | `data/processed/m1/cv_scheme_comparison.csv`(알래스카 검증 방식 비교) | `tables/cv_scheme_comparison.csv` |
| CVm | `data/processed/m1/cv_scheme_comparison_meta.json` | `tables/cv_scheme_comparison_meta.json` |
| X | `data/processed/m1/m1_sc_tests.csv`(M1 X-ak 대비) | `tables/m1_sc_tests.csv` |
| F | `data/processed/m1/l2_fold_sensitivity_summary.csv` | `tables/l2_fold_sensitivity_summary.csv` |
| N | `data/processed/lgf/lgfn_tests.csv`(LGF-N1–N4) | `tables/lgfn_tests.csv` |

사다리 단의 정의(LG 계획서 6A.3 X3a): V-R 셀 무작위 5-fold(반복 3), V-P 거친 공변량 묶음 5-fold, V-S 0.05° 격자 묶음 5-fold, V-B 0.5° 블록 5-fold, V-C0·V-C100·V-C500 공간 군집 30개 중 하나 제외 + 0·100·500 km 완충, V-G 지역 홀드아웃(100 km 완충, 지역 전체 채점). MEAN5 는 알래스카·레나·캐나다·러시아 W·러시아 E 다섯 지역 RMSE 의 등가중 평균(층화 평균)이다. 예를 들어 V-R 22.91 은 지역 RMSE 12.31, 15.62, 20.74, 40.09, 25.79 의 단순 평균이다(2.10). MEAN4 는 알래스카 제외 네 지역의 같은 평균, POOL 은 셀 풀 RMSE 다. '셀 가중'과 '블록 등가중'은 지역 안의 가중 방식이다. CI 는 0.5° 블록 재표집 10,000회다.

### 2.2 L19 검증 설계(확인적 가설, 판정 '지지하지 않음'. 열화 크기 행은 보조)

| ID | 역할(T 의 role) | 값(cm) | 원천 | 행 필터 | 열 | 판정어 | 판정 절 |
|---|---|---|---|---|---|---|---|
| A1 | 보조(열화 크기) | +10.34 [7.29, 12.84]; 블록 등가중 +6.15 [4.70, 7.57] | T | test_id=L19, contrast=`D0[catboost_lo]:V-G-V-R`, scope=MEAN5 | delta, ci_lo, ci_hi, delta_blockeq, ci_lo_beq, ci_hi_beq | 열세 | LG 7.2 L19, 결정 5 |
| A2 | 보조(A1 의 지표) | V-G 33.25, V-R 22.91 | T | A1 과 같은 행 | rmse_A, rmse_B | (지표) | LG 7.2 L20 |
| A3 | 보조(열화 크기) | +7.12 [4.51, 9.17]; 블록 등가중 +2.98 [1.42, 4.54] | T | 같은 contrast, scope=MEAN4 | delta 외 | 열세 | LG 7.2 L19, 결정 5 |
| A4 | 보조(열화 크기) | +20.13 [11.24, 27.69]; 블록 등가중 +9.83 [8.11, 11.57] | T | 같은 contrast, scope=POOL | delta 외 | 열세 | LG 7.2 L19, 결정 5 |
| A5 | 보조(열화 크기) | 알래스카 +23.25 [12.81, 32.78]; 블록 등가중 +18.83 | T | 같은 contrast, scope=region, target=Alaska | delta 외 | 열세 | LG 7.2 L19(지역 행) |
| A6 | 보조(열화 크기) | 레나 +9.69 [7.02, 14.08]; 블록 등가중 +2.53 [−1.49, 6.71] | T | target=Lena | delta 외 | 미결정 | 같음 |
| A7 | 보조(열화 크기) | 캐나다 +4.51 [1.87, 6.43]; 블록 등가중 +1.81 [0.28, 3.37] | T | target=Canada | delta 외 | 열세 | 같음 |
| A8 | 보조(열화 크기) | 러시아 W +6.07 [3.28, 9.26]; 블록 등가중 +5.33 [1.92, 8.72] | T | target=Russia_W | delta 외 | 열세 | 같음 |
| A9 | 보조(열화 크기) | 러시아 E +8.20 [0.44, 14.49]; 블록 등가중 +2.23 [−0.49, 5.35] | T | target=Russia_E | delta 외 | 미결정 | 같음 |
| A10 | 주(확인적 L19 의 판정 대비) | D0 − PS @V-R −3.90 [−5.80, −1.16]; 블록 등가중 −1.06 [−2.48, 0.37] | T | test_id=L19, contrast=`D0[catboost_lo]-PS@V-R`, scope=MEAN5 | delta 외 | 미결정 | LG 7.2 L19 |
| A11 | 주(확인적 L19 의 판정 대비) | D0 − PS @V-G +6.31 [3.99, 8.93]; 블록 등가중 +5.24 [3.64, 6.84] | T | contrast=`D0[catboost_lo]-PS@V-G`, scope=MEAN5 | delta 외 | 열세 | LG 7.2 L19 |
| A12 | 보조(보조 학습기 열화 크기) | 보조 학습기 `catboost` 열화 +13.41 [10.05, 15.83]; 블록 등가중 +8.44 | T | contrast=`D0[catboost]:V-G-V-R`, scope=MEAN5 | delta 외 | 열세 | LG 7.2 L19(보조) |
| A13 | 보조(보조 학습기 열화 크기) | 보조 학습기 `rf` 열화 +12.40 [10.15, 14.07]; 블록 등가중 +8.60 | T | contrast=`D0[rf]:V-G-V-R`, scope=MEAN5 | delta 외 | 열세 | LG 7.2 L19(보조) |
| A14 | 주(판정 행) | 판정 문구: '지지하지 않음(V-R 에서 우세가 아니다. '역전' 표현을 쓰지 않고 열화 크기만 보고한다)' | T | test_id=L19, scope=verdict | verdict | 지지하지 않음 | LG 7.2 L19, 결정 5 |

역할 열은 T 의 `role` 열이다. L19 행은 모두 `confirmatory`=True 지만, 확인적 판정에 들어가는 대비는 D0 − PS@V-R·V-G(A10, A11) 둘이다. 열화 대비 V-G − V-R(A1–A9, A12, A13)은 `role`=보조이며, LG 계획서 7.2 결정 5(982행: '열화 크기(V-G − V-R +10.3 cm)와 단조 열화(L20)만 쓴다')에 따라 열화 크기로 인용한다. 따라서 +10.34 를 확인적 대비의 결과로 쓰지 않는다. 보조 학습기(`catboost`, `rf`)의 L19 보조 판정은 '지지'다(T, test_id=L19, scope=verdict_aux). 주 학습기 판정이 등록 판정이다.

### 2.3 L20 단조 열화(보조, 판정 '지지')

| ID | 값(cm) | 원천 | 행 필터 | 열 | 판정어 | 판정 절 |
|---|---|---|---|---|---|---|
| B1 | V-R 22.91 | T | test_id=L20, item=`D0[catboost_lo] RMSE@V-R`, scope=MEAN5 | rmse_A | 주 단 | LG 7.2 L20 |
| B2 | V-P 24.28 | T | item=`...RMSE@V-P` | rmse_A | 서술 단 | 같음 |
| B3 | V-S 25.61 | T | item=`...RMSE@V-S` | rmse_A | 서술 단 | 같음 |
| B4 | V-B 27.17 | T | item=`...RMSE@V-B` | rmse_A | 주 단 | 같음 |
| B5 | V-C0 30.31 | T | item=`...RMSE@V-C0` | rmse_A | 서술 단 | 같음 |
| B6 | V-C100 30.97 | T | item=`...RMSE@V-C100` | rmse_A | 주 단 | 같음 |
| B7 | V-C500 31.13 | T | item=`...RMSE@V-C500` | rmse_A | 주 단 | 같음 |
| B8 | V-G 33.25 | T | item=`...RMSE@V-G` | rmse_A | 서술 단 | 같음 |
| B9 | V-C500 − V-R +8.22 [5.65, 10.38]; 블록 등가중 +4.49 [3.15, 5.85]; holm_p 0.0004(보조 열) | T | test_id=L20, contrast=`D0[catboost_lo]:V-C500-V-R`, scope=MEAN5 | delta 외, holm_p | 열세 | LG 7.2 L20 |
| B10 | 판정 '지지'(`catboost_lo`). 보조 `catboost` V-C500 − V-R +10.30, `rf` +10.83, 둘 다 '지지' | T | test_id=L20, scope=verdict / verdict_aux | verdict, stat | 지지 | LG 7.2 L20 |

여덟 단 모두 점 추정이 V-R < V-P < V-S < V-B < V-C0 < V-C100 < V-C500 < V-G 순이다(B1–B8). 등록 순서(V-R < V-B < V-C100 < V-C500)는 주 단 넷으로 판정했다(LG 계획서 6A.5 L20 행, 355행). 이 순서는 설계상 분리 단계의 순서이며, 학습·채점 셀 사이 거리의 순서로 확인된 것은 아니다. h41 은 단별 학습·채점 셀 거리를 계산하지 않았고(`lgv_meta.json` 에는 완충 `buffer_km` 100 만 있고 거리 통계 항목이 없다), V-P 는 공간이 아니라 공변량 값으로 묶은 단이다. V-G 는 서술 단이고 V-C500 과의 대비 CI 가 없다(단서 4). 따라서 '공간 분리가 커지는 순서로 단조 증가'가 아니라 '설계상 분리 단계 순서로 점 추정이 단조 증가'로 쓰고, 등록 대비 B9 를 함께 적는다.

### 2.4 L21 구조별 열화(보조, 판정 'λ 0.25 지지')

| ID | 값(cm) | 원천 | 행 필터 | 열 | 판정어 | 판정 절 |
|---|---|---|---|---|---|---|
| C1 | [RS(λ 0.25) − D0] 의 V-G − V-R: −8.69 [−10.71, −6.15]; 블록 등가중 −5.20 [−6.35, −4.05] | T | test_id=L21, contrast=`[RS(0.25)-D0][catboost_lo]:V-G-V-R`, scope=MEAN5 | delta 외 | 우세 | LG 7.2 L21 |
| C2 | λ 1.0: −1.12 [−1.85, −0.31]; 블록 등가중 −0.24 [−0.88, 0.41] | T | contrast=`[RS(1.0)-D0][catboost_lo]:V-G-V-R`, scope=MEAN5 | delta 외 | 미결정 | LG 7.2 L21 |

### 2.5 사다리 단별 Stefan 최소제곱(PS) RMSE(서술, 판정 없음)

| ID | 값(cm) | 원천 | 행 필터 | 열 | 판정어 | 판정 절 |
|---|---|---|---|---|---|---|
| D1 | V-R 26.81, V-P 26.81, V-S 26.82, V-B 26.96, V-C0 26.92, V-C100 26.86, V-C500 26.86, V-G 26.94 | M | method=PS, learner=none, scope=MEAN5, scheme=각 단 | rmse | 없음(서술) | 이 색인의 추출(판정 기록 없음) |

PS 는 학습 집합 최소제곱 E 하나만 맞추므로 단 사이 차가 0.15 cm 안이다. 이 행은 판정 대비가 아니라 지표다. 단 사이 대비 CI 는 계산하지 않았다.

### 2.6 알래스카 검증 방식 비교(M1 부록, 서술, CI 없음)

원천 CV, 행 필터 `scheme=<방식>, model=<모델>`, 열 `rmse_cm` 의 seed 0–2 평균. 거리 열은 같은 방식의 `nnd_median_km`(시험 셀에서 가장 가까운 학습 셀까지 거리의 중앙값), `W_km`(예측 격자 거리 분포와의 Wasserstein 거리)의 seed 평균이다. 알래스카 13,606셀, 0.5° 블록 74개, 6-fold(CVm `data`). 판정 절: `docs/EXPERIMENT_LOG.md` 218행(서술), QA Q15.

| ID | 방식(scheme) | nnd 중앙값(km) | W(km) | 직접 CatBoost(`catboost_lo`) | ridge 직접 | Stefan(`stefan`) | 물리 잔차(`stefan_ridge_l075`) |
|---|---|---|---|---|---|---|---|
| E1 | 무작위 셀(`random_cell`) | 0.004 | 103.90 | 11.55 | 12.65 | 14.24 | 12.77 |
| E2 | 0.05° 사이트(`site_0.05`) | 1.11 | 100.37 | 13.44 | 13.33 | 14.32 | 13.30 |
| E3 | 0.5° 블록, 대회 규약(`block_0.5_canonical`) | 28.37 | 60.32 | 16.12 | 13.62 | 14.46 | 13.33 |
| E4 | 0.5° 블록, 균형 무작위 배정(`block_0.5`) | 25.78 | 63.56 | 16.10 | 13.72 | 14.45 | 13.37 |
| E5 | kNNDM(`knndm`) | 53.50 | 31.35 | 17.91 | 14.11 | 14.45 | 13.70 |

| ID | 값 | 원천 | 행 필터 | 열 | 판정어 |
|---|---|---|---|---|---|
| E6 | kNNDM nnd 중앙값 seed 0/1/2: 58.71, 43.05, 58.73 km | CV | scheme=knndm, model=stefan | nnd_median_km | 없음 |
| E7 | 예측 격자(영구동토 표본 20,000)에서 가장 가까운 라벨까지 중앙값 81.57 km | CVm | gij_summary | permafrost_median_km | 없음 |

E1 의 nnd 원값은 0.0041 km(seed 0, CVm `distance_summary`)다. 물리 잔차는 Stefan 앵커 + 0.75 × ridge 잔차다(CVm `model_notes`). 'CatBoost' 는 저용량 설정(반복 200, 깊이 3)이다.

### 2.7 알래스카 0.5° 블록 짝지음 대비(M1 X-ak, 두 가중)

| ID | 대비 | 값(cm) | 원천 | 행 필터 | 열 | 두 가중 판정 | 판정 절 |
|---|---|---|---|---|---|---|---|
| F1 | 직접 CatBoost − Stefan | 셀 +1.66 [−1.32, 3.93]; 블록 등가중 −1.87 [−3.15, −0.72] | X | H=X-ak, label=`알래스카: catboost_lo 직접 − Stefan`, target=Alaska | delta_rmse, ci_lo, ci_hi, delta_blockeq, ci_lo_blockeq, ci_hi_blockeq | 가중 규약 의존(부호 반대) | `docs/COVERAGE_MATRIX_BY_REGION_2026-09-29.md` 8절, `docs/NOVELTY_POSITIONING_2026-09-29.md` 5절 N8 |
| F2 | 물리 잔차 − Stefan | 셀 −1.13 [−2.03, −0.39]; 블록 등가중 −0.93 [−2.55, 0.87] | X | label=`알래스카: Stefan+ridge(λ=.75) − Stefan` | 같음 | 가중 규약 의존 | 같음 |
| F3 | ridge 직접 − Stefan | 셀 −0.84 [−2.07, −0.20]; 블록 등가중 −0.22 [−2.40, 2.19] | X | label=`알래스카: ridge 직접 − Stefan` | 같음 | 가중 규약 의존 | 같음 |

F 행은 대회 규약 0.5° 블록 6-fold, seed 3 이다. kNNDM 과 무작위 셀에는 짝지음 CI 가 없다.

### 2.8 알래스카 fold 배정 민감도(서술)

| ID | 방법 | 현행 배정 / 무작위 블록 배정 10회 중앙값(cm) | 무작위 배정 5–95 백분위 | Stefan 대비 차(무작위 배정 중앙값), 음수 배정 수 | 원천·행 필터 | 판정 절 |
|---|---|---|---|---|---|---|
| G1 | Stefan | 14.46 / 14.44 | 14.36–14.49 | (기준) | F, method=stefan | EXPERIMENT_LOG 219행 |
| G2 | 물리 잔차 | 13.33 / 13.51 | 13.33–13.64 | −0.89, 10/10 | F, method=stefan_ridge075 | 같음 |
| G3 | 직접 CatBoost | 16.04 / 16.01 | 15.62–16.52 | +1.56, 0/10 | F, method=catboost_lo | 같음 |

열: current_pooled, random_median, random_p05, random_p95, delta_vs_stefan_random_median, n_random_delta_negative. 모두 셀 풀 RMSE 이고 CI 는 없다. G3 의 `catboost_lo` 는 3 seed 앙상블 예측의 풀링 RMSE 다(`scripts/2_evaluation/l2_fold_sensitivity.py` 5행 'catboost_lo 직접(3 seed 앙상블)'). E 표는 seed 별 RMSE 의 평균이다. 그래서 같은 대회 규약 배정의 직접 CatBoost 가 G3 현행 16.04, E3 16.12 로 다르다(E3 의 seed 0–2 RMSE 16.22, 15.51, 16.63, 2.10).

### 2.9 원천 교차검증 조정(LGF-N2 보조, LGF-N4 서술)

| ID | 값 | 원천 | 행 필터 | 열 | 판정어 | 판정 절 |
|---|---|---|---|---|---|---|
| H1 | MLP 직접 ML, n 0: 조정판 − 기본판 +1.51 [0.87, 2.25]; 블록 등가중 +0.85 [0.21, 1.49] | N | test_id=LGF-N2, contrast=`D0[mlp*]-D0[mlp]\|n0\|lam1`, scope=MEAN, role=주 | delta 외 | 열세 | LGF 계획서 10.3 |
| H2 | 다중 헤드 MLP 직접 ML, n 0: +1.33 [0.89, 1.81]; 블록 등가중 +1.33 [0.77, 1.90] | N | contrast=`D0[tabm*]-D0[tabm]\|n0\|lam1`, scope=MEAN, role=주 | delta 외 | 열세 | 같음 |
| H3 | 축소 FT-T 잔차, 라벨 전량 λ 0.25: −0.19 [−0.50, −0.06]; 블록 등가중 −0.39 [−0.63, −0.18] | N | contrast=`R1[ftt*]-R1[ftt]\|nall\|lam0.25`, scope=MEAN, role=주 | delta 외 | 우세(크기 0.5 cm 미만) | 같음 |
| H4 | 학습기별 판정: MLP·다중 헤드 MLP '조정이 전이 오차를 키웠다', FT-T '조정판의 오차가 작다', RealMLP '정밀도 미달'. 전체: '강건 문장 규칙을 채운 학습기 없음' | N | test_id=LGF-N2, scope=verdict_aux | verdict | 학습기별 | 같음 |
| H5 | 원천 교차검증 이득 대 대상 Δ(n 0) 순위 상관 −0.32(점 32개, 그 가운데 (0, 0) 점 12개) | N | test_id=LGF-N4, spearman 열이 있는 행 | spearman, n_points, points | 서술(검정 없음) | LGF 계획서 10.3 LGF-N4 |

LGF-N 은 모두 로컬 RTX 3090 한 플랫폼 조각이다(N, platform 열 `local-3090-torch2.6.0-numpy1.26.4`). 이 색인에서 H5 의 점 32개로 다시 계산한 순위 상관은 같은 값이고 p 는 0.078 이다(이번 점검, 사후, 판정 아님).

### 2.10 `evidence_values.csv` 밖의 값

아래 값은 `evidence_values.csv`(83행)에 없다. 값마다 원천을 적었고, 2026-10-04 점검에서 원천과 대조했다.

| 값 | 쓰인 곳 | 원천 | 비고 |
|---|---|---|---|
| 무작위 셀 nnd 중앙값 원값 0.0041 km(seed 0–2 모두) | 2.6 E1 아래 | CVm `distance_summary`(scheme=random_cell, nnd_median_km) | E1 의 0.004 는 이 값의 반올림 |
| 예측 격자 표본 20,000 | 2.6 E7 | CVm `prediction_domain.n_sample` | 같은 항목의 영구동토 격자 수 `n_grid_permafrost` 581,385, 주 영역 `main_domain`=permafrost |
| 6-fold, seed 0–2 | 2.6 머리 | CVm `k_folds`, `seeds` | 설계 상수 |
| CatBoost 저용량 설정(반복 200, 깊이 3) | 2.6 E 표 아래 | CVm `model_notes.catboost_lo` | iterations=200, depth=3 |
| 대회 규약 fold 0 단일 블록 4,716셀 | 단서 7 | `docs/EXPERIMENT_LOG.md` 219행, `scripts/2_evaluation/l2_fold_sensitivity.py` 3행, CVm `distance_summary`(block_0.5_canonical 의 fold_sizes 첫 값) | 세 곳이 같다 |
| E3 직접 CatBoost seed 0–2 RMSE 16.22, 15.51, 16.63 | 2.8 아래 | CV, scheme=block_0.5_canonical, model=catboost_lo, rmse_cm | 평균 16.12 = E3 |
| MEAN5 V-R 의 지역 RMSE 12.31, 15.62, 20.74, 40.09, 25.79 | 2.1 | T, test_id=L19, contrast=`D0[catboost_lo]:V-G-V-R`, scope=region, rmse_B | 단순 평균 22.91 = A2 |
| 순위 상관 p 0.078 | 2.9 아래 | N 의 LGF-N4 `points`(32점)로 scipy `spearmanr` 재계산(이번 점검, 사후) | 판정 아님 |
| FT-T 교차 환경 차 최대 5.61 cm, 0.5 cm 초과 키 43/288(14.9 %), 중앙값 0.059 cm | 단서 10 | `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 1155행(개정 15 (u)), 해석 1157행 | 원 표는 `data/processed/lgw/lgw_xenv_gate_iii.csv`(이 폴더에 사본 없음) |
| LGF 학습기 교차 환경 대조의 지역 행 \|Δ\| 최대 5.59 cm(RealMLP) | 단서 10 | `docs/EXPERIMENT_PLAN_LGF_2026-09-29.md` 568행, `data/processed/lgf/lgfn_cross.csv`(scope=region 행의 \|delta\| 최대 5.593) | 판정에 쓰지 않는 보조 표, 이 폴더에 사본 없음 |
| Gautam 2025 RF R² 0.24(시험), 0.84(학습), Liu 2024 RF R² 0.97 | 단서 11 | CVm `references.literature` | 원문 미확인 [미확인] |
| Wadoux et al. 2021, Ecol. Model. 457:109692 | 단서 8 | CVm `references.caveat` | 원문 미확인 [미확인] |
| Linnenbrink, Milà, Ludwig, Meyer (2024), Geosci. Model Dev. 17(15):5897–5912, doi:10.5194/gmd-17-5897-2024 | 단서 11 | `references/INDEX.md` 202행, `references/05_uq_transfer/linnenbrink2024_knndm.pdf` 1쪽(저자, 권, 쪽, doi 확인. 호수 15 는 INDEX.md 에만 있다) | CVm `references.knndm` 의 저자 목록에는 Ludwig 가 빠져 있다 |

## 3. 뒷받침 실험

| 새 이름 | 옛 id | 내용 | 코드 | 산출 |
|---|---|---|---|---|
| T2_transfer_structure_placebo_baselines | LGX X3a 검증 사다리(L19–L21), 확인적 L19 | 5지역 자료(지역별 채점, MEAN5·MEAN4·POOL 보고), 8단 사다리, D0·PS·RS, 학습기 3종 | `scripts/2_evaluation/h41_validation_ladder.py` | T, M, `lgv_meta.json`(사본 `tables/lgv_meta.json`) |
| H2_master_M1 | M1 부록 검증 방식 비교(09-21) | 알래스카 7방식 × 모델 4종 × seed 3 | `scripts/3_deep_learning/m1_cv_scheme_comparison.py` | CV, CVm |
| H2_master_M1 | M1 X-ak(지역 내 6-fold 짝지음 대비, 탐색: X 의 family=exploratory) | 알래스카 0.5° 블록, 두 가중 CI | 적합: `scripts/3_deep_learning/m1_master_factorial.py`, `src/polar/m1_core.py`. 대비·CI: `scripts/3_deep_learning/m1_analysis.py`(589–590행 `add_test("X-ak", …)`, 640행 `tests.to_csv(M1 / f"{OUT}_tests.csv")`) | X |
| H2_master_M1 | M1 부록 L2 fold 배정 민감도 | 현행 GroupKFold 대 무작위 블록 배정 10회 | `scripts/2_evaluation/l2_fold_sensitivity.py` | F |
| T3_learners_foundation_and_nn | LGF-N2, N2s, N4 | 원천 지역 하나 제외 교차검증 초모수 조정 | `scripts/3_deep_learning/h48_nn_tuning.py` | N |
| H0_contest | 2026-06 분할 누설 표, 대회 0.5° 블록 결과(13.33 대 14.46) | L19·L20 의 비맹검 원천 | [미확인: 2026-06 표의 경로를 이 작업에서 찾지 못함] | E3 이 같은 규약의 재계산 |

## 4. 그림

| 그림 | 파일 | 패널과 C7 관련 내용 | 상태 |
|---|---|---|---|
| 본문 Fig 1(v2) | `outputs/figures/paper/v2/Fig1_problem.{svg,pdf,png}` | 패널 c 가 설계 도식(지역 홀드아웃 + 100 km 완충, A/B 블록 절반, 라벨 추출, 채점). 검증 사다리 수치는 없다 | 완료. 재구성안(`docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 4·5절): R1 절이 C7(−10.3 cm)을 인용하고 Fig 1 패널 c 에 지역 내 블록 홀드아웃 설계 한 줄을 더한다 |
| v2 본문 Fig 2–7 | `outputs/figures/paper/v2/` | C7 수치를 쓰는 패널 없음 | 해당 없음 |
| SI 검증 사다리 | `figures/figure_spec.json` 의 `SF_validation_ladder`(1920행 부근, L19–L21, 원천 T·M) | 무작위 분할에서 지역 홀드아웃까지 | 산출 파일을 찾지 못했다(`outputs/figures` 검색, 2026-10-04). 원고 초안 R8.5 가 SI 배치로 둔다(`docs/MANUSCRIPT_DRAFT_RESULTS_ABSTRACT_2026-09-30.md` 299행) |
| 알래스카 검증 방식 비교 | `outputs/figures/m1/cv_scheme_comparison.{png,pdf}`, 스펙 `figures/figure_spec_cv_scheme.json` | (a) 모델 4종 × 방식 6종 막대(오차 막대 = fold RMSE SD), (b) 가장 가까운 학습 셀 거리 ECDF 와 W | 내부 검토용. 한국어 글자, '공간블록 0.5°(현행)' 막대가 실제로는 `block_0.5`(균형 무작위 배정, W 63.5 km)이고 대회 규약 `block_0.5_canonical` 이 아니다(스크립트 395–396행 `SCHEME_FIG`). 원고에 쓰려면 다시 그린다 |

배치 결정 사항: QA Q12 강화 방법 4는 검증 사다리를 본문 그림 하나로 보이자고 한다(무작위·지점·블록·kNNDM·지역 홀드아웃을 함께). Q15 권고는 무작위 분할 수치를 검증 사다리 그림으로 함께 싣자고 하며 본문·SI 배치는 정하지 않았다. 재구성안 5절은 표시 항목 8개를 유지하고 사다리를 본문 그림으로 두지 않는다. Q12 강화 방법 4와 재구성안 5절은 아직 조정되지 않았다(사용자 결정). 본문으로 올린다면 권고 구성은 (a) 5지역 사다리 8단의 D0(학습기 3종), PS, RS(λ 0.25) RMSE 와 (b) 알래스카 방식별 RMSE 대 가장 가까운 학습 셀 거리(E1–E5, 대회 규약 블록 사용)다. 단위(cm, km)와 셀 가중·블록 등가중 표기를 넣는다.

## 5. 단서와 쓰지 않을 문장

### 5.1 단서

1. **비맹검**: L19·L20 은 방향을 이미 본 '재현(비맹검)' 가설이다(LG 6A.7).
2. **집계 방식 의존**: 열화 크기는 집계에 따라 +2.98(알래스카 제외 4지역 블록 등가중, A3)에서 +20.13(셀 풀, A4) cm 까지 달라진다. 5지역 층화 평균은 +10.34(블록 등가중 +6.15, A1)다. 지역 행까지 넣으면 캐나다 블록 등가중 +1.81 까지 내려간다(A7). 지역 행에서 레나와 러시아 E 는 미결정이다(A6, A9). 러시아 W·E 는 각 28·27셀이지만 층화 평균에서 5분의 1 가중을 받는다(T, n_cells).
3. **무엇이 커지는가**: 열화는 직접 ML 의 성질이다. PS 는 단 사이 0.15 cm 안(D1)이고, Stefan 앵커 + 잔차(λ 0.25)의 열화는 D0 보다 8.69 cm 작다(C1). 다만 λ 0.25 는 앵커 비중이 커서 열화가 구조적으로 작다(LG 계획서 6A.5 L21 문구). λ 1.0 은 미결정이다(C2).
4. **두 효과의 혼합**: V-G − V-R 은 학습·채점 셀 사이 거리와 지역 간 영역 이동을 함께 담는다. 중간 단(V-C0–V-C500)은 같은 지역의 먼 라벨을 학습에 남기므로 두 효과를 일부 나눈다. V-C500 과 V-G 의 점 추정 차(31.13 대 33.25 cm, B7·B8)에는 대비 CI 가 없으므로 성분 크기로 해석하지 않는다.
5. **알래스카 비교는 점 추정**: E 표는 seed 3개 평균이고 CI 가 없다. 짝지음 CI 가 있는 0.5° 블록 대비(F1–F3)는 세 대비 모두 두 가중 규칙을 통과하지 못했다. kNNDM 의 nnd 는 seed 에 따라 43.05–58.73 km 로 흔들린다(E6).
6. **두 가지 '0.5° 블록' 수치**: QA Q15 표는 대회 규약(`block_0.5_canonical`, E3: 16.12, 13.62, 14.46, 13.33)을, EXPERIMENT_LOG 218행과 cv_scheme 그림은 균형 무작위 배정(`block_0.5`, E4: 16.10, 13.72)을 쓴다. 원고는 E3 으로 통일하고 E4 를 배정 민감도(G 행과 함께)로 둔다.
7. **13.33 cm 의 배정 의존**: 대회 규약의 fold 0 은 단일 블록 4,716셀이다(EXPERIMENT_LOG 219행). 무작위 블록 배정 10회 중앙값은 물리 잔차 13.51, Stefan 14.44 cm 다(G1, G2).
8. **kNNDM 의 성격**: kNNDM 은 예측 격자 거리 분포에 맞춘 교차검증이다. 설계 기반 확률 표본이 없으면 공간 교차검증도 지도 정확도의 편향 없는 추정이 아니라는 반론이 CVm 의 `references.caveat` 에 기록되어 있다(Wadoux et al. 2021, Ecol. Model. 457:109692. 원문 미확인 [미확인]).
9. **LGF-N4 상관**: 서술 행이고 검정이 등록되지 않았다. 점 32개 가운데 12개가 '선택 결과 = 기본값'의 (0, 0) 점이다(H5).
10. **신경망 플랫폼**: LGF-N 은 로컬 GPU 한 플랫폼 결과다. 교차 환경 점검 (iii)에서 FT-T 는 같은 코드·설정에서 로컬과 Rescale 사이 키 단위 최대 5.61 cm(0.5 cm 초과 키 14.9 %, 288개 중 43개) 달랐다(LG 계획서 개정 15 (u), 1155행). 이 값은 FT-T 키에서 잰 것이고 LGF-N 의 다른 학습기(MLP, 다중 헤드 MLP, RealMLP)에서 잰 값이 아니다. LGF 의 보조 교차 환경 대조(로컬 기본판 대 LG Rescale 조각, 판정에 쓰지 않음)에서는 지역 행 |Δ| 최대가 5.59 cm(RealMLP)였다(LGF 계획서 568행, 2.10). 그래서 같은 대비 안에 Rescale 결과를 섞지 않는다.
11. **문헌 수치**: CVm `references.literature` 에 Gautam 2025(Sci Rep, 무작위 70/30 사이트 분할, RF 시험 R² 0.24, 학습 0.84)와 Liu 2024(ERL, 무작위 10-fold, RF R² 0.97)가 기록되어 있다. 이 작업에서 원문을 확인하지 않았다 [미확인]. Ploton 2020, Meyer & Pebesma 2022, Wadoux 2021, Linnenbrink 2024 는 원고 참고문헌 목록에 아직 없다(QA Q15 '문헌의 흐름' 마지막 항목). Meyer & Pebesma 2021(Methods Ecol. Evol. 12, 1620–1633)은 원고 참고문헌에 이미 있다(`outputs/report/main.tex` 676행). Linnenbrink 2024 의 서지는 Linnenbrink, Milà, Ludwig, Meyer, Geosci. Model Dev. 17(15):5897–5912, doi:10.5194/gmd-17-5897-2024 다. `references/INDEX.md` 202행과 로컬 PDF 1쪽(`references/05_uq_transfer/linnenbrink2024_knndm.pdf`)으로 대조했고, 호수 15 는 INDEX.md 에만 있다. CVm `references.knndm` 의 저자 목록에는 Ludwig 가 빠져 있으므로 CVm 에서 서지를 옮기지 않는다.

### 5.2 쓰지 않을 문장

| 쓰지 않을 문장 | 이유와 근거 | 대신 쓸 형태 |
|---|---|---|
| '검증 방식에 따라 ML 과 물리식의 순위가 뒤집힌다', '역전', 'ranking reverses' | L19 지지하지 않음, 등록 규칙(A10, A14, LG 7.2 결정 5). 원고 금지 목록(`docs/MANUSCRIPT_DRAFT_RESULTS_ABSTRACT_2026-09-30.md` 215행), NOVELTY 5절 N8 폐기 | 열화 크기(A1)와 단조 열화(B 행)만 쓴다 |
| '무작위 분할의 오차는 지역 홀드아웃보다 10.3 cm 작다'(한정어 없이) | 직접 ML·셀 가중·5지역 층화 평균 값이다. PS 는 거의 변하지 않는다(D1) | '직접 ML 의 RMSE 는 … +10.34 cm(블록 등가중 +6.15)' |
| '5지역 통합 자료에서', 'on the pooled data of five regions'(MEAN5 값에) | 22.91, 33.25, +10.34 는 다섯 지역 RMSE 의 등가중 평균(MEAN5)이다. 이 표에서 'pooled'는 셀 풀(POOL: 13.54, 33.67, +20.13, A4)을 뜻한다 | '다섯 지역 RMSE 의 지역 등가중 평균(층화 평균)으로 보면', 'averaged over the five regions with equal weight' |
| '학습·채점 셀의 공간 분리가 커지는 순서로 단조 증가' | h41 은 단별 학습·채점 셀 거리를 계산하지 않았다. V-P 는 공변량 묶음 단이다. 등록 단조성(L20)은 주 단 넷이 대상이다(2.3 아래) | '설계상 분리 단계 순서로 점 추정이 단조 증가했다. 등록 대비 V-C500 − V-R 은 +8.22 [5.65, 10.38](블록 등가중 +4.49 [3.15, 5.85])이다'(B9) |
| '무작위 분할에서 1등인 CatBoost 가 지도 상황에서는 Stefan 보다 나쁘다'(검정 결과로) | 0.5° 블록 짝지음 대비가 가중에 따라 부호가 반대다(F1). kNNDM 은 CI 가 없다 | '점 추정으로 직접 CatBoost 는 블록·kNNDM 채점에서 Stefan 보다 오차가 컸다(16.12, 17.91 대 14.46, 14.45 cm). 0.5° 블록 짝지음 대비는 가중 방식에 따라 부호가 달랐다' |
| '물리 잔차는 모든 채점 방식에서 Stefan 보다 유의하게 낫다' | 점 추정만 모든 방식에서 낮다(E1–E5). 짝지음 대비는 셀 가중에서만 CI 가 0 을 제외한다(F2) | '점 추정으로 모든 방식에서 낮았고(12.77–13.70 대 14.24–14.46 cm), 무작위 블록 배정 10회 모두에서 낮았다(G2)' |
| 'kNNDM 이 지도 정확도의 가장 정직한 추정이다', '우리 채점은 과하게 어렵지 않다' | 평가어. 단서 8 | 거리 수치(E3, E5, E7)를 그대로 쓴다 |
| '높은 정확도를 보고한 선행 연구 대부분은 무작위 교차검증이었다'(OVERVIEW C7) | 문헌 집계가 없다. QA Q15 는 ALT 문헌에서 무작위 분할과 공간 분리가 함께 쓰인다고 적었다 [미확인] | '무작위 분할을 쓴 ALT 연구(예: Gautam 2025)의 수치는 이 연구의 지역 홀드아웃 수치와 직접 비교할 수 없다'(원문 확인 뒤) |
| '대회 보고서 같은 지역 내 무작위·공간 교차검증의 우위가 새 지역에서 나타나지 않는 이유다'(현행 C7 마지막 문장) | 대회는 무작위가 아니라 0.5° 블록 교차검증이었다. L19 열화는 직접 ML 의 값이고 대회 최선 구성(물리 잔차)의 값이 아니다. 인과 문장의 근거가 없다 | '지역 안 교차검증의 수치와 지역 홀드아웃의 수치는 같은 척도로 비교할 수 없다' |
| '원천 교차검증 조정은 해롭다', '원천 교차검증 조정은 전이 오차를 줄이지 않았다'(학습기 일반) | 학습기마다 다르다(H4). MLP 의 열세는 D0 대비(라벨 0·10·전량)에만 있고 R1 대비는 동등·미결정이다. 다중 헤드 MLP 는 R1 라벨 0 에서도 열세였다(λ 0.25 +0.24, 0.5 cm 미만. 보조 λ 1.0 +0.75). 축소 FT-T 의 우세는 R1 라벨 전량 두 대비(주 λ 0.25 −0.19, 보조 λ 1.0 −0.76)뿐이고 나머지 7대비는 동등·미결정이다(N, test_id=LGF-N2, scope=verdict_aux) | 'MLP·다중 헤드 MLP 의 직접 ML(D0)에서는 조정이 전이 오차를 키웠고(라벨 0 +1.51, +1.33 cm), 축소 FT-T 는 라벨 전량 잔차에서 조정판이 작게 나았다(λ 0.25 −0.19 cm, 0.5 cm 미만. 보조 λ 1.0 −0.76 cm)' |
| '신경망의 결론은 초모수 선택에 강건하다' | 강건 문장 규칙을 채운 학습기가 없다(H4) | LGF-N2s 문구('MLP 와 RealMLP 에서 핵심 판정이 바뀌지 않았다', LGF 10.3) |
| 'LGF-N4 의 상관 −0.32 는 원천 교차검증이 대상 오차를 키운다는 증거다' | 서술, 검정 없음, (0, 0) 점 12개(H5) | '방향은 반대였다(순위 상관 −0.32, 검정 없음)' |
| 초록에 L19 판정어 | WRAPUP 1.1 이 초록 묶음에서 L19 를 뺐다 | 본문 방법·토의에서만 쓴다 |
| 공간 교차검증을 '안전하다' 또는 '편향 없다'로 쓰는 것 | 비열등·편향 검정이 없다. 단서 8 | 조건과 수치만 쓴다 |

## 6. 이 주장을 바꿀 수 있는 계획 실험(X*)

설계는 아직 등록되지 않았다 [미확인]. 아래는 C7 과의 관계와 바뀔 수 있는 문장이다.

| 계획 실험 | C7 과의 관계 | 바뀔 수 있는 것 |
|---|---|---|
| XH_validation_ladder | 직접 해당. 알래스카 방식 비교(E 표)에 두 가중 짝지음 블록 CI 를 붙이고, 5지역 사다리와 한 그림으로 묶는 작업 | F1–F3 의 '가중 규약 의존'이 kNNDM 이나 다른 단에서도 유지되면 5.2 의 대체 문장을 그대로 쓴다. kNNDM 에서 두 가중 모두 열세가 나오면 'CatBoost 는 지도 상황 채점에서 Stefan 보다 나쁘다'를 판정 문장으로 쓸 수 있다. seed 를 늘리면 E5 의 17.91, 53.50 km 가 바뀔 수 있다 |
| XF_new_regions | V-G 의 평균 지역 수(현재 5) | 독립 지역이 늘면 A1 의 10.34 와 지역 행 판정이 바뀐다. 사용자 질문 1(공개 자료로 독립 지역 추가)과 같은 축이다 |
| XC_workflow_end_to_end | 방법 선택을 대상 라벨 안 교차검증으로 하는 근거(C5 와 C7 의 LGF-N2 부분) | 원천 교차검증 선택 팔을 CatBoost·해석식에도 두면, 학습기별로 갈린 LGF-N2 결과(MLP·다중 헤드 MLP 의 D0 열세, 축소 FT-T 의 R1 전량 작은 우세)를 신경망 밖에서도 확인할 수 있다 |
| XD_placement_policy | 샘플링 정책 학습의 평가 설계. 지역 하나 제외 평가가 필요하다는 C7 의 근거를 그대로 쓴다 | C7 수치는 바뀌지 않는다. 지역 안 평가로만 검증한 정책은 C7 기준에 따라 낙관 표지를 단다 |
| XG_product_comparison | 공개 ALT 제품의 보고 정확도가 무작위 교차검증인지 확인하고, 같은 채점 셀에서 비교 | 제품의 학습 셀과 우리 채점 셀 사이 거리(nnd)를 C7 의 거리 진단(E 표)으로 보고하면 누설 점검과 연결된다. '선행 연구 대부분은 무작위 교차검증'(5.2) 문장의 근거가 생길 수 있다 |
| XI_climate_extrapolation_retest | 학습 범위 밖 기후 블록이라는 별도 검증 단 | 사다리에 '외삽 단'을 더할 수 있다. 판정은 C8 소관이다 |
| XE_hires_covariates | 고해상 공변량은 무작위 분할에서 이웃 셀 기억의 이득을 키울 수 있다 | V-R 과 V-G 의 차가 커지거나 줄 수 있다. 공변량을 바꾸면 사다리를 다시 돌려야 C7 수치와 비교할 수 있다 |
| XA, XB, XJ | C2, 다중 소스 적층, 지온 유도 보조 라벨 | C7 수치에 직접 영향이 없다 |

## 7. 이 폴더의 파일

| 파일 | 내용 |
|---|---|
| `README.md` | 이 문서 |
| `extract_evidence.py` | 원천 표에서 근거 수치를 읽는 스크립트(읽기 전용) |
| `evidence_values.csv` | 추출 결과 83행(id, source, row_filter, column, value, value_raw, verdict, verdict_section, note) |
| `tables/` | 원천 집계 표 사본 8개와 `MANIFEST.csv`(orig_path, copy_path, sha256, rows, bytes). 셀 단위 라벨 자료와 거리 표본(`cv_scheme_distances.csv`, 7.4 MB)은 넣지 않았다 |
