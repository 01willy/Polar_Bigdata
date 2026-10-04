# C1 라벨 0 전이에서 물리 정보의 손해 감소: 근거 색인

**성격**: 주장 C1 의 근거 색인이다(2026-10-04 작성). 기존 파일은 옮기거나 고치지 않았고, 판정 근거 표의 사본을 `tables/` 에 두었다(`tables/MANIFEST.csv`, sha256 원본 일치 확인). 새 실험이나 새 판정은 없다. 이번에 새로 계산한 값은 2.8절의 세 가지뿐이며 '이번 점검(사후)'으로 표시했다.

**정정(2026-10-04 점검 반영)**: (1) 공통 재표집 보조 CI 의 판정(`verdict4_common`)과 `ci_dependence` 를 1.3·2.1·2.2·2.4·2.7·5.1 에 병기했다. (2) 직접 ML 대 P* 행(L28 앵커 tddm 변형)을 2.7절에 더하고 5.1(9)를 고쳤다. (3) 2.2절 3지역 표에 빠진 학습기 행을 더하고 5.1(3)을 고쳤다. (4) 1.3 의 '줄였다'를 사후 서술로 한정하고 유의 악화 대상 수(2.8절 3)를 병기했다. (5) WF0 위험표의 R2 가중(λ 1.0)을 밝혔다. (6) 새 지역 행에 원천 표지(`flags`, `cross_env`)를 옮겼다. (7) 1.2절의 절 참조를 고쳤다. 수치는 모두 원천 CSV 에서 다시 읽었다.

**수치 규칙**
- 모든 수치는 원천 CSV 를 pandas 로 읽어 옮겼다(OMP_NUM_THREADS=1, 읽기 전용). 문서 본문에서 옮긴 수치는 없다.
- Δ 는 RMSE 차(cm, 앞 방법 − 뒤 방법)다. 음수면 앞 방법의 오차가 작다. Δ 와 CI 는 소수 둘째 자리로 반올림했고, p 는 파일 값을 넷째 자리까지 적었다.
- '셀 가중'은 `delta`, `ci_lo`, `ci_hi` 열이고 '블록 등가중'은 `delta_blockeq`, `ci_lo_beq`, `ci_hi_beq` 열이다. 아래 표에서 '표준 8열'은 이 여섯 열과 `verdict4`, `holm_p` 를 뜻한다.
- 판정어는 4분 판정이다. 우세·열세는 두 가중 CI 가 모두 같은 쪽, 동등은 두 CI 가 모두 ±0.5 cm 안, 그 밖은 미결정이다. LG 곡선의 대상별 `sig_p0`(worse, ns, improve)는 두 가중 CI 가 모두 0 을 제외하는지를 나타낸다.
- 'MEAN' 은 주 4지역(레나, 캐나다, 러시아 서부, 러시아 동부, 모드 x) 층화 평균, 'MEAN3' 은 레나·캐나다·알래스카 3지역 보조 열이다.
- '공통 CI' 는 지역 블록 공통 재표집 보조 CI(`ci_lo_c`, `ci_hi_c` 셀 가중, `ci_lo_beq_c`, `ci_hi_beq_c` 블록 등가중)이고 그 4분 판정이 `verdict4_common` 이다. 주 판정(`verdict4`)과 다르면 원천 표의 `ci_dependence` 가 '분할 독립 가정 의존'이다. `pool_fixed_curve.csv` 는 `ci_dependence` 열이 없어 두 판정 열을 비교해 적었고, `lgd_tests_lic.csv` 는 `sig` 와 `sig_common` 을 쓴다. 표의 '공통 판정' 열은 `verdict4_common`, '의존' 열은 `ci_dependence` 이며 빈칸은 두 판정이 같다는 뜻이다.
- 단정형 규칙: LG 계획서 6A.2 '보조 CI' 행(`docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 202행)은 '본문은 두 CI 의 판정이 같은 대비만 단정형으로 쓴다'이고, WRAPUP 0.5 '보조 CI' 행(`docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 77행)이 같은 표지 규칙을 쓴다. 이 색인의 제안 문장은 이 규칙을 따른다.
- 반올림하면 0.00 이 되는 끝값은 셋째 자리까지 적었다.

**기호**: P0 원천 계수 Stefan, P* 연도 정합 도일 Stefan(P0@tddm), D0 직접 ML, D1 물리 유사라벨 증강, R0 원천 계수 앵커 + 잔차(λ 는 잔차 가중), R2 증강 + 앵커 + 잔차, F1k 물리 출력을 입력으로 준 직접 ML. 라벨 0 에서는 P1 = P0, R1 = R0 이다(`tables/wf0_risk.csv` 의 `n=0, method=P1` 행이 모두 0). R2 의 잔차 가중은 표마다 다르다. 2.4·2.6절의 R2 는 λ 0.25 이고, WF0 위험표(2.5절, 2.8절)의 `R2_1.0` 은 λ 1.0 이다(`scripts/2_evaluation/wf0_reanalysis.py` 의 `RECIPES` 에 `("R2", 1.0)` 만 있다).

---

## 1. 주장 문장

### 1.1 현재 문장

`docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md` 3절(31행):

> C1. 물리 증강과 저가중 잔차는 희소 라벨 지역에서 ML 을 안전하게 쓰는 방법이다.

같은 문서의 C8 문단(89행)은 이 주장을 '손해 없는 선택(C1)'으로 부른다. `docs/RESEARCH_OVERVIEW_2026-10-02.md` 5절 C1 과 9절 '안전성' 행도 같은 표현을 쓴다.

### 1.2 축소가 필요한 근거

| 항목 | 근거 문서 | C1 에 주는 영향 |
|---|---|---|
| '안전' 표현 | `docs/QA_FINAL_REVIEW_2026-10-02.md` 'Sci Rep 리뷰어 관점 평가' 표의 '결과의 지지' 행('안전' 표현은 지적 대상), Q12 강화 1(a) '물리 앵커가 ML 의 실패를 막는다' | 라벨 0 에서 D1·R2 의 P0 대비 비열등을 등록 가설로 시험한 적이 없다. 등록된 비열등 가설 SC1w(라벨 3·10, R1 − P0)는 '안전성 미확인(비열등 칸 4/10)'이다(`tables/lgw_tests.csv`, `test_id=SC1w`). 라벨 0 의 저가중 잔차는 4지역 평균에서 동등(±0.5 cm)이 등록 판정으로 나왔으나(2.4절) 이 동등은 공통 CI 에서 미결정이고(분할 독립 가정 의존), 알래스카를 넣은 3지역 열에서는 미결정 또는 열세다. 따라서 '안전'을 쓰지 않고 '손해를 줄인다'로 쓴다 |
| WF0 사후 표지 | `docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 머리말('결과 열람 뒤 설계'), 2절 WF0('판정어를 쓰지 않고 사후 서술 표지'), 3절('WF0 의 수치는 사후 서술로만 쓴다'); `tables/wf0_meta.json` 의 `label` = '사후 서술(판정어 없음)' | 위험표(18/30 대 2/30)는 판정어 없이 '사후 서술'로만 쓴다. 2 cm 문턱도 WF0 에서 정했다. 초록의 판정어 대상(WRAPUP 1.1 의 AB1–AB10)이 아니다 |
| C2 축소 | `docs/QA_FINAL_REVIEW_2026-10-02.md` Q1 'C2 점검' | 라벨 0 에서는 재보정이 없으므로(P1 = P0) 재보정 몫과 ML 몫을 나누는 C2 의 축소는 C1 의 라벨 0 문장을 바꾸지 않는다. 다만 워크플로 표의 '라벨 1–10' 행은 이득 대부분이 재보정(AB4)에서 오므로 C1 의 근거로 쓰지 않는다. 확인 값은 2.8절 |
| 집계 방식 | `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 1.2·1.3 | 주 4지역 평균의 판정을 지역 단위 판정처럼 쓰지 않는다. 30대상 위험표는 하위 지역을 포함하므로 대상 수로만 쓴다 |

### 1.3 제안 문장(축소판)

**국문**: 대상 라벨이 없는 지역 홀드아웃 전이에서, 랜덤 포레스트와 원천 지역 하나 제외 교차검증으로 고른 신경망 4종의 직접 ML 은 주 4지역 층화 평균으로 원천 계수 Stefan 식(P0)보다 오차가 컸다(+2.44 – +5.21 cm). 주 학습기 CatBoost(+2.25 cm [1.10, 3.42], Holm 보정 p 0.023)를 포함한 나머지 5종(CatBoost 세 설정, TabPFN v2, TabICL v2)도 주 CI 의 판정은 같았으나, 이 판정은 분할 독립 가정에 의존한다(지역 블록 공통 재표집 CI 에서는 미결정). 주 학습기의 직접 ML 은 연도 정합 도일 Stefan 식(P*)보다도 오차가 3.24 cm [1.86, 4.29] 컸다(L28 보조 가설, Holm 보정 p 0.0050, 앵커 변형은 비맹검). 30개 대상에서 P0 보다 2 cm 넘게 나빠진 대상은 직접 ML 18개, 물리 유사라벨 증강 2개, 증강 + 앵커 + 잔차(λ 1.0) 1개, 저가중(λ 0.25) 잔차 5개로, 물리 정보를 쓴 쪽이 적었다(사후 서술). 두 가중 CI 가 모두 0 을 넘은 대상 수로 세면 각각 14개, 8개, 5개, 6개다(사후 계산). 물리 유사라벨은 순서를 섞은 유사라벨보다 오차가 1.63 cm 작았다(Holm 보정 p 0.001).

**English draft**: In region-held-out transfer without target labels, direct ML with a random forest and with four neural networks configured by leave-one-source-region-out cross-validation had a larger four-region stratified-mean error than the source-coefficient Stefan model (P0) (+2.44 to +5.21 cm). For the other five learners (three CatBoost settings, TabPFN v2 and TabICL v2), including the main CatBoost learner (+2.25 cm [1.10, 3.42], Holm-adjusted p = 0.023), the primary intervals gave the same verdict, but this verdict depends on the split-independence assumption (undetermined under the common regional-block resampling interval). Direct ML with the main learner also had a 3.24 cm [1.86, 4.29] larger error than the year-matched thaw-degree-day Stefan model (P*) (auxiliary hypothesis L28, Holm-adjusted p = 0.0050; the anchor variant was unblinded). Among 30 targets, the error exceeded that of P0 by more than 2 cm for 18 targets with direct ML, 2 with physics pseudo-label augmentation, 1 with augmentation plus anchored residual (λ = 1.0) and 5 with a low-weight (λ = 0.25) residual (post hoc description). Counted by intervals above zero under both weightings, the numbers were 14, 8, 5 and 6 (post hoc computation). Physics pseudo-labels gave 1.63 cm lower error than shuffled pseudo-labels (Holm-adjusted p = 0.001).

**문장 요소별 두 CI 판정**(수치와 원천은 2절 각 행)

| 문장 요소 | 근거 행(절) | 주 판정(`verdict4`) | 공통 판정(`verdict4_common`) | `ci_dependence` | 본문 표기 |
|---|---|---|---|---|---|
| rf, mlp*, tabm*, ftt*, realmlp* 의 D0 − P0(MEAN) | L30, LGF-N1(2.1절 4·7–10 번) | 열세 | 열세 | 없음 | 단정형 |
| F1k − P0(MEAN, 보조) | L11(2.1절 보조 행) | 열세 | 열세 | 없음 | 단정형 |
| `catboost_lo` 의 D0 − P0(MEAN) | AB1(2.1절 1 번) | 열세 | 미결정 | 분할 독립 가정 의존 | 표지 병기 |
| `catboost`, `catboost_tuned` 의 D0 − P0(MEAN) | L30(2.1절 2·3 번) | 열세 | 미결정 | 분할 독립 가정 의존 | 표지 병기 |
| TabPFN v2 의 D0 − P0(MEAN) | L34(a)(2.1절 5 번) | 열세(보정 전 유의) | 미결정 | 분할 독립 가정 의존 | 표지 병기 |
| TabICL v2 의 D0 − P0(MEAN, 두 컨텍스트 조건) | LGF-F1(2.1절 6a·6b 번) | 열세 | 미결정 | 분할 독립 가정 의존 | 표지 병기 |
| `catboost_lo` 의 D0 − P*(MEAN) | L28 앵커 tddm 변형(2.7절) | 열세 | 열세 | 없음 | 단정형(보조 가설, 비맹검 표지) |
| 위험표의 대상 수 | WF0(2.5절, 2.8절) | 판정어 없음 | 해당 없음 | 해당 없음 | 사후 서술 |
| D1 − D1@shuffle(MEAN) | AB3(2.3절) | 우세 | 우세 | 없음 | 단정형 |
| 저가중 잔차 R0[T]·R0[C]·R0[I] − P0 동등(MEAN) | L34(b), 병기, LGF-F5(2.4절) | 동등 | 미결정 | 분할 독립 가정 의존 | 표지 병기 |

**함께 쓸 한정어**(5절 상세): 4지역 평균 기준이며 주 CI 의 지역 단위 열세는 러시아 동부 1/4 다. 이 행도 분할 독립 가정 의존이고 공통 CI 에서는 4지역 모두 미결정이다. 원천 계수가 크게 틀린 티베트에서는 직접 ML 이 물리식보다 훨씬 낫다(예측된 결과(비맹검), 교차 환경 표지). 알래스카에서는 증강도 물리식보다 1.79 cm 나쁘다. TabPFN v2 의 열세는 LGT 묶음 Holm 보정 전 유의이고 분할 독립 가정 의존이다. 저가중 잔차의 등록 근거는 'TabPFN·CatBoost·TabICL 잔차(λ 0.25)는 4지역 평균에서 P0 와 동등(분할 독립 가정 의존)'으로 쓴다. 라벨 0 에서 D1·R2 의 P0 대비와 D1 − D0·R0 − D0 대비를 판정한 등록 가설은 없다(2.4절 끝).

---

## 2. 근거 표

### 2.1 라벨 0 직접 ML 대 원천 계수 Stefan(학습기 10종, 주 4지역 MEAN, 대비 = 학습기의 D0 − P0)

| # | 학습기 | 원천 | 행 필터 | 셀 가중 Δ [CI] | 블록 등가중 Δ [CI] | 판정어 | Holm p | 공통 CI 셀 가중 / 블록 등가중, 공통 판정 | 의존 | 판정 문서 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CatBoost 주 학습기(`catboost_lo`) | `tables/lgw_bundle.csv` | `ab=AB1, scope=MEAN` (`contrast=D0-P0\|n0`), 표준 8열 | +2.25 [1.10, 3.42] | +1.62 [0.52, 2.69] | 열세 | 0.023(초록 묶음 m = 10) | [0.31, 4.19] / [−0.21, 3.43], 미결정 | 분할 독립 가정 의존 | WRAPUP 결과 판정 기록(J7) 1.1 표. 가설 L1 '지지'는 `tables/lg_tests.csv` `test_id=L1, scope=verdict`, `verdict` 열, LG 계획서 7.1 |
| 2 | CatBoost 용량 확대(`catboost`) | `tables/lgx_tests.csv` | `test_id=L30, contrast=D0[catboost]-P0\|n0, scope=MEAN`, 표준 8열 | +1.94 [0.88, 3.43] | +1.56 [0.30, 2.76] | 열세 | 0.0032 | [0.24, 4.32] / [−0.45, 3.57], 미결정 | 분할 독립 가정 의존 | LG 계획서 7.2 L30(지지) |
| 3 | CatBoost 원천 교차검증 조정(`catboost_tuned`) | `tables/lgx_tests.csv` | `test_id=L30, contrast=D0[catboost_tuned]-P0\|n0, scope=MEAN` | +1.59 [0.65, 2.69] | +1.63 [0.56, 2.64] | 열세 | 0.0032 | [0.07, 3.37] / [−0.08, 3.33], 미결정 | 분할 독립 가정 의존 | LG 7.2 L30 |
| 4 | 랜덤 포레스트(`rf`) | `tables/lgx_tests.csv` | `test_id=L30, contrast=D0[rf]-P0\|n0, scope=MEAN` | +2.63 [1.47, 3.96] | +2.03 [0.99, 3.06] | 열세 | 0.0003 | [0.72, 4.75] / [0.31, 3.75], 열세 | | LG 7.2 L30 |
| 5 | TabPFN v2 | `tables/lgt_tests.csv` | `test_id=L34, contrast=D0[T]-P0\|n0, scope=MEAN`, 표준 8열과 `holm_note` | +1.02 [0.23, 2.35] | +1.22 [0.27, 2.15] | 열세(`holm_note` = '보정 전 유의') | 0.1168 | [−0.27, 3.13] / [−0.34, 2.75], 미결정 | 분할 독립 가정 의존 | LG 계획서 7.4 L34(a)(지지) |
| 6a | TabICL v2, 컨텍스트 상한 10,000행 | `tables/lgf_tests.csv` | `test_id=LGF-F1, contrast=D0[I]-P0\|n0, scope=MEAN` | +0.93 [0.28, 3.12] | +2.43 [1.35, 3.55] | 열세 | 0.0232 | [−0.36, 4.20] / [0.72, 4.20], 미결정 | 분할 독립 가정 의존 | LGF 계획서 10.2 F1(지지) |
| 6b | TabICL v2, 원천 전체 컨텍스트 | `tables/lgf_tests.csv` | `test_id=LGF-F1, contrast=D0@full[I]-P0\|n0, scope=MEAN` | +0.85 [0.07, 3.19] | +2.22 [1.09, 3.45] | 열세 | 0.0374 | [−0.60, 4.26] / [0.53, 4.10], 미결정 | 분할 독립 가정 의존 | LGF 10.2 F1 |
| 7 | MLP(원천 지역 하나 제외 교차검증 조정) | `tables/lgfn_tests.csv` | `test_id=LGF-N1, contrast=D0[mlp*]-P0\|n0, scope=MEAN` | +2.45 [1.62, 4.11] | +2.05 [0.95, 3.19] | 열세 | 0.0004 | [0.95, 4.90] / [0.25, 3.94], 열세 | | LGF 계획서 10.3 N1(지지) |
| 8 | 다중 헤드 MLP(`tabm*`, 공개 TabM 과 다름) | `tables/lgfn_tests.csv` | `test_id=LGF-N1, contrast=D0[tabm*]-P0\|n0, scope=MEAN` | +2.44 [1.58, 4.73] | +3.94 [2.68, 5.29] | 열세 | 0.0004 | [0.87, 5.98] / [1.92, 6.33], 열세 | | LGF 10.3 N1 |
| 9 | 축소 FT-Transformer(`ftt*`) | `tables/lgfn_tests.csv` | `test_id=LGF-N1, contrast=D0[ftt*]-P0\|n0, scope=MEAN` | +4.13 [3.28, 5.63] | +4.58 [3.44, 5.76] | 열세 | 0.0004 | [2.62, 6.52] / [2.65, 6.59], 열세 | | LGF 10.3 N1 |
| 10 | RealMLP(`realmlp*`) | `tables/lgfn_tests.csv` | `test_id=LGF-N1, contrast=D0[realmlp*]-P0\|n0, scope=MEAN` | +5.21 [4.07, 7.17] | +5.91 [4.60, 7.25] | 열세 | 0.0004 | [3.18, 8.31] / [3.73, 8.19], 열세 | | LGF 10.3 N1 |
| 보조 | 물리 출력 입력 직접 ML(F1k) | `tables/lgx_tests.csv` | `test_id=L11, contrast=F1k-P0\|n0, scope=MEAN` | +2.62 [1.38, 4.36] | +2.73 [1.57, 3.85] | 열세 | 0.0023 | [0.40, 5.53] / [0.82, 4.62], 열세 | | LG 7.2 L11(보조, 지지) |
| 병기 | 같은 컨텍스트 CatBoost(로컬, `D0[C]`) | `tables/lgt_tests.csv` | `test_id=L34, contrast=D0[C]-P0\|n0, scope=MEAN` | +2.64 [1.36, 3.72] | +1.79 [0.69, 2.81] | 열세 | 없음 | [0.59, 4.39] / [−0.005, 3.51], 미결정 | 분할 독립 가정 의존 | LG 7.4(병기 행) |

- 학습기 10종은 1–10 번이다(TabICL 은 두 컨텍스트 조건을 한 학습기로 센다). 1–4 번은 Rescale 조각, 5–10 번과 병기 행은 로컬 GPU(RTX 3090) 조각이다. 플랫폼 사이의 차는 계산하지 않는다.
- Holm p 는 묶음마다 다르다. 1 번은 초록 묶음 10개(WRAPUP 1.1), 2–4 번과 보조 행은 LGX 묶음, 5 번은 LGT 묶음, 6–10 번은 LGF 확인적 묶음이다.
- 공통 CI 열은 각 원천 표의 `ci_lo_c`, `ci_hi_c`, `ci_lo_beq_c`, `ci_hi_beq_c`, `verdict4_common`, 의존 열은 `ci_dependence` 다. 두 CI 판정이 같은 학습기는 4·7·8·9·10 번(5종)과 보조 F1k 다. 1·2·3·5·6a·6b 번과 병기 행은 분할 독립 가정 의존이다(`tables/fig6_a.csv` 의 `ci_dependence` 열과 같다). 따라서 '10종 모두 열세'는 주 CI 기준이며, 본문 단정형은 5종으로 한정하고 나머지에는 표지를 병기한다(LG 계획서 202행).

### 2.2 같은 대비의 지역 행과 3지역 보조 열

**AB1 지역 행**(`tables/lgw_bundle.csv`, `ab=AB1, scope=region`, 표준 8열 가운데 앞 7열과 공통 CI 열)

| 지역 | 셀 가중 Δ [CI] | 블록 등가중 Δ [CI] | 판정어 | 공통 CI 셀 가중 / 블록 등가중, 공통 판정 | 의존 |
|---|---|---|---|---|---|
| 레나 | +4.02 [1.98, 5.62] | +0.20 [−1.66, 1.97] | 미결정 | [0.61, 6.58] / [−3.00, 3.18], 미결정 | |
| 캐나다 | −1.56 [−2.70, 1.41] | −0.01 [−1.61, 1.62] | 미결정 | [−3.27, 3.23] / [−2.85, 2.63], 미결정 | |
| 러시아 서부 | +2.05 [−0.94, 4.49] | +2.78 [−0.03, 5.28] | 미결정 | [−3.28, 6.35] / [−2.11, 7.13], 미결정 | |
| 러시아 동부 | +4.48 [1.41, 6.51] | +3.51 [1.14, 5.91] | 열세 | [−0.32, 7.50] / [−0.10, 7.20], 미결정 | 분할 독립 가정 의존 |

- 주 CI 의 지역 단위 열세는 러시아 동부 1/4 이고, 공통 CI 에서는 4지역 모두 미결정이다.

**새 독립 지역(LGD 약관 확인분 판)**(`tables/lgd_tests_lic.csv`, `test_id=L1e, pool=PE2, scope=region, n=0`, 열 `delta`·CI·`sig`, `sig_common`, `ci_dependence`, `flags`, `cross_env`)

| 지역 | 셀 가중 Δ [CI] | 블록 등가중 Δ [CI] | `sig` | `sig_common` | `ci_dependence` | 표지(`flags`) | 표지(`cross_env`) |
|---|---|---|---|---|---|---|---|
| 러시아 중부(`Russia_C\|x`) | +2.75 [2.11, 4.09] | +4.83 [3.17, 6.60] | worse | worse | 없음 | 없음 | 교차 환경; (i) CatBoost 불통과 |
| 티베트(`Tibet_LGD\|x`) | −59.98 [−62.90, −57.15] | −62.91 [−65.38, −60.32] | improve | improve | 없음 | 예측된 결과(비맹검); 비맹검(라벨 통계 열람) | 교차 환경; (i) CatBoost 불통과 |

- L1e 판정(`scope=verdict`, `verdict` 열)은 P4·PE1·PE2 모두 '지지'이고, `scope=compare` 행은 '4지역의 판정이 확장 풀(지역 6개)에서 유지된다'이다. 판정 문서는 LG 계획서 7.3 이다. PE2 의 지지는 기각 기준(⌊6/4⌋ = 1 지역까지 허용) 안에서 티베트 1지역의 반대 방향을 허용한 결과다(LG 7.3 L1e 근거 열).
- 표지: LG 계획서 7.3 '표지' 항(988행)에 따라 PE1·PE2 행(PE1·PE2 의 `verdict`·`compare` 행 포함)에는 '교차 환경; (i) CatBoost 불통과'를 단다. 새 지역 조각은 로컬, P4 는 Rescale 조각이기 때문이다. 같은 항은 티베트 대비를 비맹검으로 두며, 티베트 행의 `flags` 는 '예측된 결과(비맹검);비맹검(라벨 통계 열람)'이다. 이 표의 두 행과 5.1(2)의 티베트 수치는 이 표지와 함께 쓴다.

**3지역 보조 열(레나·캐나다·알래스카, `scope=MEAN3`)**

| 학습기 | 원천과 필터 | 셀 가중 Δ [CI] | 블록 등가중 Δ [CI] | 판정어 | 공통 CI 셀 가중 / 블록 등가중, 공통 판정 | 의존 | MEAN(4지역) 셀 가중 Δ |
|---|---|---|---|---|---|---|---|
| `catboost_lo` | `tables/lgfn_tests.csv`, `test_id=LGF-N1, contrast=D0[catboost_lo]-P0\|n0`(로컬 병기 행, `platform` = local-3090) | +7.75 [5.49, 10.11] | +6.23 [5.04, 7.40] | 열세 | [4.11, 11.45] / [4.23, 8.17], 열세 | | +2.25 |
| `catboost` | `tables/lgx_tests.csv`, `test_id=L30, contrast=D0[catboost]-P0\|n0` | +8.09 [5.89, 9.44] | +5.69 [4.63, 6.74] | 열세 | [4.95, 10.28] / [3.87, 7.46], 열세 | | +1.94 |
| `rf` | 같은 표, `contrast=D0[rf]-P0\|n0` | +6.91 [5.37, 8.07] | +5.17 [4.16, 6.16] | 열세 | [4.51, 8.85] / [3.39, 6.88], 열세 | | +2.63 |
| `catboost_tuned` | 같은 표, `contrast=D0[catboost_tuned]-P0\|n0` | +4.06 [2.62, 4.85] | +2.61 [1.79, 3.41] | 열세 | [2.09, 5.30] / [1.22, 3.94], 열세 | | +1.59 |
| TabPFN v2 | `tables/lgt_tests.csv`, `test_id=L34, contrast=D0[T]-P0\|n0` | +7.83 [5.40, 8.66] | +4.96 [3.94, 5.98] | 열세 | [4.67, 9.05] / [3.23, 6.62], 열세 | | +1.02 |
| TabICL v2, 컨텍스트 상한 10,000행 | `tables/lgf_tests.csv`, `test_id=LGF-F1, contrast=D0[I]-P0\|n0` | +12.25 [7.55, 14.21] | +7.91 [6.55, 9.26] | 열세 | [5.53, 15.38] / [5.72, 10.15], 열세 | | +0.93 |
| TabICL v2, 원천 전체 컨텍스트 | 같은 표, `contrast=D0@full[I]-P0\|n0` | +0.31 [−0.68, 2.37] | +0.70 [−0.35, 1.78] | 미결정 | [−1.41, 3.45] / [−1.01, 2.58], 미결정 | | +0.85 |
| `mlp*` | `tables/lgfn_tests.csv`, `test_id=LGF-N1, contrast=D0[mlp*]-P0\|n0` | +9.28 [6.30, 10.46] | +6.82 [5.75, 7.90] | 열세 | [5.45, 11.12] / [5.01, 8.63], 열세 | | +2.45 |
| `tabm*` | 같은 표, `contrast=D0[tabm*]-P0\|n0` | +9.26 [6.64, 10.77] | +7.91 [6.61, 9.18] | 열세 | [5.29, 11.77] / [5.78, 10.06], 열세 | | +2.44 |
| `ftt*` | 같은 표, `contrast=D0[ftt*]-P0\|n0` | +1.17 [0.48, 2.14] | +0.94 [0.32, 1.55] | 열세 | [0.03, 2.76] / [−0.14, 2.04], 미결정 | 분할 독립 가정 의존 | +4.13 |
| `realmlp*` | 같은 표, `contrast=D0[realmlp*]-P0\|n0` | +6.54 [4.67, 7.37] | +6.10 [5.07, 7.14] | 열세 | [3.78, 8.01] / [4.36, 7.82], 열세 | | +5.21 |
| 보조 F1k | `tables/lgx_tests.csv`, `test_id=L11, contrast=F1k-P0\|n0` | +7.69 [5.48, 10.33] | +7.05 [5.78, 8.33] | 열세 | [3.96, 11.97] / [4.92, 9.12], 열세 | | +2.62 |
| 병기 `D0[C]` | `tables/lgt_tests.csv`, `test_id=L34, contrast=D0[C]-P0\|n0` | +8.00 [5.55, 10.36] | +6.01 [4.88, 7.15] | 열세 | [4.28, 11.50] / [4.12, 7.82], 열세 | | +2.64 |

- 3지역 보조 열에서 시험한 학습기의 D0 − P0 는 +0.31 – +12.25 cm 다. TabICL v2 원천 전체 컨텍스트(+0.31)는 미결정이고, 축소 FT-T(`ftt*`)의 열세는 분할 독립 가정 의존이다.
- 3지역 값이 4지역 값보다 큰 학습기가 대부분이지만, 축소 FT-T(+1.17 대 +4.13)와 TabICL v2 원천 전체 컨텍스트(+0.31 대 +0.85)는 3지역 값이 더 작다. 블록 등가중 Δ 도 같은 방향이다(`ftt*` +0.94 대 +4.58, TabICL 원천 전체 +0.70 대 +2.22).
- `catboost_lo` 의 3지역 행은 LGF-N1 의 로컬 병기 행이다. AB1(`lgw_bundle.csv`)에는 `scope=MEAN3` 행이 없다. 이 행의 4지역 값(+2.25)은 AB1 과 같다.

**한계 행: TabICL v2 의 라벨 10개**(`tables/lgf_tests.csv`, `test_id=LGF-F1`, 판정 문서 LGF 10.2 F1 보조 (a))

| 필터 | 셀 가중 Δ [CI] | 블록 등가중 Δ [CI] | 판정어 | 공통 CI 셀 가중 / 블록 등가중, 공통 판정 | 의존 |
|---|---|---|---|---|---|
| `contrast=D0[I]-P0\|n10, scope=MEAN` | −2.91 [−3.58, −1.11] | −1.49 [−2.65, −0.21] | 우세 | [−4.07, −0.31] / [−3.15, 0.39], 미결정 | 분할 독립 가정 의존 |
| `contrast=D0[I]-P0\|n10, scope=MEAN3` | +5.98 [2.26, 7.22] | +3.46 [2.52, 4.43] | 열세 | [1.85, 7.53] / [2.02, 4.99], 열세 | |

### 2.3 물리 유사라벨의 정보 출처(위약 대조)

| 대비 | 원천과 필터 | 셀 가중 Δ [CI] | 블록 등가중 Δ [CI] | 판정어 | Holm p | 판정 문서 |
|---|---|---|---|---|---|---|
| AB3 D1 − 섞은 유사라벨, n 0 | `tables/lgw_bundle.csv`, `ab=AB3, scope=MEAN` (`contrast=D1-D1@shuffle\|n0`), 표준 8열 | −1.63 [−1.98, −1.04] | −1.05 [−1.51, −0.57] | 우세 | 0.001(초록 묶음 m = 10) | WRAPUP J7 1.1 |
| L15 shuffle, n 0 | `tables/lgx_tests.csv`, `test_id=L15, contrast=D1-D1@shuffle\|n0, scope=MEAN` | −1.63 [−1.99, −1.04] | −1.05 [−1.52, −0.58] | 우세 | 0.003 | LG 7.2 L15 |
| L15 shuffle, n 10 | 같은 표, `contrast=D1-D1@shuffle\|n10` | −1.15 [−1.43, −0.73] | −0.68 [−1.06, −0.29] | 우세 | 0.003 | LG 7.2 L15 |
| L15 const_t, n 0 | `contrast=D1-D1@const_t\|n0` | −1.68 [−2.06, −1.12] | −1.06 [−1.55, −0.56] | 우세 | 0.003 | LG 7.2 L15 |
| L15 const_t, n 10 | `contrast=D1-D1@const_t\|n10` | −1.16 [−1.46, −0.76] | −0.74 [−1.11, −0.37] | 우세 | 0.003 | LG 7.2 L15 |
| L15 const_src, n 0 | `contrast=D1-D1@const_src\|n0` | −3.63 [−4.33, −2.44] | −2.81 [−3.49, −2.11] | 우세 | 0.003 | LG 7.2 L15 |
| L15 const_src, n 10 | `contrast=D1-D1@const_src\|n10` | −2.21 [−2.89, −1.27] | −1.54 [−2.23, −0.86] | 우세 | 0.003 | LG 7.2 L15 |
| L15 tddlin, n 0 | `contrast=D1-D1@tddlin\|n0` | −0.48 [−0.62, −0.27] | −0.41 [−0.61, −0.21] | 우세(크기 0.5 cm 미만) | 0.0034 | LG 7.2 L15 |
| L15 tddlin, n 10 | `contrast=D1-D1@tddlin\|n10` | −0.08 [−0.18, 0.05] | −0.03 [−0.15, 0.09] | 동등 | 1 | LG 7.2 L15 |
| L15 shuffle, n 0, 3지역 | `contrast=D1-D1@shuffle\|n0, scope=MEAN3` | −3.78 [−4.31, −1.97] | −1.19 [−1.78, −0.61] | 우세 | 없음 | LG 7.2 L15(보조) |

- L15 등록 판정(`scope=verdict`, `verdict` 열): '혼재: n 별로 서술한다 [통계적으로 구별되나 크기는 0.5 cm 미만: tddlin n=0]'.
- AB3 과 L15 shuffle n 0 의 CI 차이는 재표집 구현 차이다(AB3 은 h39, L15 는 h42 재집계. WRAPUP 1.1 'p 값' 항).
- 이 표의 모든 행은 공통 CI 판정(`verdict4_common`)이 주 판정과 같다(`ci_dependence` 빈칸). AB3 MEAN 의 공통 CI 는 셀 가중 [−2.15, −0.80], 블록 등가중 [−1.71, −0.35] 로 우세다. 따라서 위약 대조 문장은 단정형으로 쓸 수 있다.

### 2.4 저가중 잔차와 결합 구조(라벨 0)

| 대비 | 원천과 필터 | 셀 가중 Δ [CI] | 블록 등가중 Δ [CI] | 판정어 | 공통 CI 셀 가중 / 블록 등가중, 공통 판정 | 의존 | 비고 | 판정 문서 |
|---|---|---|---|---|---|---|---|---|
| R0[T] − P0(TabPFN 잔차, λ 0.25) | `tables/lgt_tests.csv`, `test_id=L34, contrast=R0[T]-P0\|n0, scope=MEAN` | −0.31 [−0.48, 0.05] | −0.21 [−0.43, 0.02] | 동등 | [−0.59, 0.23] / [−0.58, 0.15], 미결정 | 분할 독립 가정 의존 | `p_eq` 0.0163, `holm_p_eq` 없음(보정 전) | LG 7.4 L34(b) |
| 같은 대비, 3지역 | 같은 필터, `scope=MEAN3` | −0.13 [−0.31, 0.27] | +0.38 [0.10, 0.66] | 미결정 | [−0.44, 0.47] / [−0.07, 0.84], 미결정 | | | LG 7.4(보조) |
| R0[C] − P0(같은 컨텍스트 CatBoost 잔차) | `test_id=L34, contrast=R0[C]-P0\|n0, scope=MEAN` | −0.05 [−0.42, 0.16] | −0.21 [−0.48, 0.04] | 동등 | [−0.62, 0.31] / [−0.64, 0.20], 미결정 | 분할 독립 가정 의존 | `p_eq` 0.0145 | LG 7.4(병기) |
| 같은 대비, 3지역 | `scope=MEAN3` | +0.02 [−0.29, 0.53] | +0.39 [0.04, 0.73] | 미결정 | [−0.46, 0.76] / [−0.22, 0.98], 미결정 | | | |
| R0[I] − P0(TabICL 잔차) | `tables/lgf_tests.csv`, `test_id=LGF-F5, contrast=R0[I]-P0\|n0, scope=MEAN` | −0.27 [−0.48, 0.25] | +0.01 [−0.28, 0.30] | 동등 | [−0.67, 0.54] / [−0.48, 0.49], 미결정 | 분할 독립 가정 의존 | `eq_p` 0.0191, `holm_p` 1 | LGF 10.2 F5 |
| 같은 대비, 3지역 | `scope=MEAN3` | +0.64 [0.17, 1.12] | +0.71 [0.25, 1.18] | 열세 | [−0.11, 1.43] / [−0.08, 1.49], 미결정 | 분할 독립 가정 의존 | | LGF 10.2 F5(보조) |
| R0 − P0(Rescale `catboost_lo`, λ 0.25), 단계 T0 | `tables/lgw_scenarios_summary.csv`, `stage=T0, contrast=R0-P0\|n0`, 열 `main4_delta`, `main4_ci_lo`, `main4_ci_hi`, `main4_ci_lo_beq`, `main4_ci_hi_beq`, `main4_verdict4`, `main4_noninf` | −0.10 [−0.44, 0.14] | CI [−0.48, 0.03] | 동등 | 요약표에 공통 CI 열 없음. 같은 대비의 구성 고정 풀 곡선 행(아래)은 공통 판정 미결정 | 알 수 없음(요약표에 열 없음) | `main4_noninf` True. `m3_delta` −0.08(`m3_verdict4` 미결정). 독립 5지역: 우세 1, 동등 1, 미결정 3, 열세 0, 비열등 2. 최악 `Russia_E\|x` +0.51 [−0.587, 1.098]. '동등 1'(레나)은 `tables/lgw_scenarios.csv` 에서 공통 판정 미결정(분할 독립 가정 의존), '우세 1'(캐나다)은 공통 판정도 우세 | WRAPUP 3.1–3.2(등록 요약값, 가설 판정 아님) |
| 같은 대비, 알래스카 칸 | `tables/lgw_scenarios.csv`, `stage=T0, target=Alaska\|x` | +0.42 [−0.30, 1.92] | +2.23 [1.29, 3.13] | 미결정 | [−0.72, 2.67] / [0.56, 3.82], 미결정 | | `noninf` False | WRAPUP 3.2 |
| R0 − P0, λ 0.25(구성 고정 풀 곡선) | `tables/pool_fixed_curve.csv`, `edition=E1_P4_n_le_10, contrast=R0-P0\|n0, scope=MEAN` | −0.10 [−0.45, 0.13] | −0.22 [−0.48, 0.03] | 동등 | [−0.65, 0.29] / [−0.66, 0.20], 미결정 | 분할 독립 가정 의존(두 판정 열 비교) | `role` = '서술(판정에 쓰지 않는다)' | DISPLAY_ITEMS 6절 1번 |
| R2 − P0, λ 0.25(구성 고정 풀 곡선) | 같은 표, `contrast=R2-P0\|n0` | −0.08 [−0.26, 0.01] | −0.29 [−0.43, −0.14] | 동등 | [−0.37, 0.10] / [−0.53, −0.06], 미결정 | 분할 독립 가정 의존(두 판정 열 비교) | 서술 | DISPLAY_ITEMS 6절 1번 |
| R0 − F1k(잔차 구조 대 물리 입력 구조) | `tables/lgx_tests.csv`, `test_id=L10, contrast=R0-F1k\|n0, scope=MEAN` | −2.72 [−4.34, −1.73] | −2.95 [−3.93, −1.97] | 우세 | [−5.37, −0.92] / [−4.59, −1.34], 우세 | | Holm 0.0023 | LG 7.2 L10(지지) |

- 라벨 0 에서 D1(증강 단독)과 R2 의 P0 대비를 4지역 층화 평균으로 판정한 등록 가설은 없다. D1 − D0, R0 − D0 대비를 판정한 등록 가설도 없다. 증강의 P0 대비 근거는 2.5절(사후 서술)과 2.6절(대상별 곡선)뿐이다. 가장 가까운 등록 행은 L16(X2, 유사라벨 비율, `role` = 보조)의 `D1@r30-P0|n0` MEAN(비율 30 변형)이며 −0.18 [−0.76, 0.18], 블록 등가중 −0.65 [−1.16, −0.18], 미결정(공통 판정도 미결정)이다(`tables/lgx_tests.csv`, `test_id=L16`). 기본 비율 D1 의 P0 대비가 아니다.
- 라벨 0 의 저가중 잔차 동등(R0[T], R0[C], R0[I] 의 MEAN 행과 구성 고정 풀 곡선의 R0·R2 행)은 모두 공통 CI 에서 미결정이다(분할 독립 가정 의존). 본문에서는 '동등(분할 독립 가정 의존)'으로 병기하고 단정형으로 쓰지 않는다. 등록 근거 문장은 'TabPFN·CatBoost·TabICL 잔차(λ 0.25)는 4지역 평균에서 P0 와 동등(분할 독립 가정 의존)'이다.
- 이 표의 R2 는 λ 0.25 다. WF0 위험표(2.5절)의 `R2_1.0` 은 λ 1.0 이므로 두 결과를 같은 R2 로 묶지 않는다.

### 2.5 라벨 0 위험표(WF0, 사후 서술, 판정어 없음)

원천: `tables/wf0_risk.csv`. 열 `n_targets`, `n_worse_2cm`(Δ = 방법 − P0 가 +2 cm 를 넘은 대상 수), `median`, `max`. 판정어 없음(`tables/wf0_meta.json` 의 `label`). 계산 코드 `scripts/2_evaluation/wf0_reanalysis.py`(셀 무작위 추출, α 1, `catboost_lo`, 점 추정만인 대상 제외). 문서 근거: WF 계획서 2절 WF0, 3절.

| 필터 `n` | 방법(`method`) | `n_targets` | `n_worse_2cm` | `median` | `max` |
|---|---|---|---|---|---|
| 0 | P1(= P0) | 30 | 0 | +0.00 | +0.00 |
| 0 | D0_1.0(직접 ML) | 30 | 18 | +2.69 | +38.46 |
| 0 | R1_1.0(잔차 가중 1.0) | 30 | 13 | +1.55 | +38.54 |
| 0 | R1_0.25(저가중 잔차) | 30 | 5 | +0.15 | +6.29 |
| 0 | D1_1.0(물리 증강) | 30 | 2 | +0.13 | +3.89 |
| 0 | R2_1.0(증강 + 앵커 + 잔차) | 30 | 1 | −0.02 | +6.03 |
| 10 | P1 | 30 | 2 | +0.06 | +2.27 |
| 10 | D0_1.0 | 30 | 12 | +1.49 | +12.89 |
| 10 | R1_1.0 | 30 | 11 | +1.34 | +15.19 |
| 10 | R1_0.25 | 30 | 2 | +0.49 | +2.71 |
| 10 | D1_1.0 | 30 | 5 | +0.43 | +5.00 |
| 10 | R2_1.0 | 30 | 2 | +0.31 | +7.50 |
| −1(전량) | P1 | 30 | 4 | +0.13 | +4.64 |
| −1 | D0_1.0 | 30 | 7 | +0.27 | +7.35 |
| −1 | R1_1.0 | 30 | 5 | +0.16 | +9.14 |
| −1 | R1_0.25 | 30 | 4 | −0.13 | +3.38 |
| −1 | D1_1.0 | 30 | 5 | −0.17 | +12.72 |
| −1 | R2_1.0 | 30 | 7 | −0.36 | +10.70 |

- `R2_1.0` 은 증강 + 앵커 + 잔차의 잔차 가중 λ 1.0 이다(`wf0_reanalysis.py` 의 `RECIPES`). 2.4·2.6절의 R2(λ 0.25)와 다르다. 위험표의 '저가중 잔차'는 `R1_0.25` 이고 라벨 0 에서 R1 = R0 이다.
- `n_worse_2cm` 은 점 추정 문턱이라 CI 를 보지 않는다. 같은 입력의 대상별 `sig_p0`(두 가중 CI 가 모두 0 을 넘으면 worse)로 센 유의 악화 대상 수는 2.8절 3 에 있다(사후 계산). 라벨 0 에서 증강(`D1_1.0`)의 유의 악화 대상은 8개로 2 cm 문턱 수(2개)보다 많다.
- 30대상은 LG 25대상(27 가운데 그린란드와 점 추정 전용 러시아 C 제외)과 LGD 약관 확인분 5대상이다(`tables/wf0_misspec.csv` 의 `target`, `src` 열). 이 가운데 독립 지역은 7개(레나, 캐나다, 러시아 서부·동부, 알래스카, 러시아 중부, 티베트)이고 나머지는 하위 지역 20개(AL-1–AL-6, CA-2, CA-3, LE-1, LE-2 의 모드 x·i)와 확충판 3개다.

### 2.6 대상별 곡선(LG 본 실행, 원천 크기 때문에 사본 없음)

원천: `results/rescale_lg/data/processed/lg/lg_curve.csv`(7,072,630 bytes, 5 MB 상한을 넘어 `tables/` 에 복사하지 않았다). 필터: `mode=x, n=0, axis=method, learner=catboost_lo, placement=cell, alpha=1`, D0·D1 은 `lam=1.0`, R0·R2 는 `lam=0.25`. 열 `rmse`, `rmse_p0`, `d_p0`(+CI `d_p0_lo`, `d_p0_hi`), `d_p0_beq`(+CI), `sig_p0`. 재표집 1,000회(LG 7.1).

| 대상 | 방법 | `rmse` | `rmse_p0` | 셀 가중 Δ [CI] | 블록 등가중 Δ [CI] | `sig_p0` |
|---|---|---|---|---|---|---|
| Alaska | D0 | 35.33 | 14.54 | +20.79 [14.14, 26.92] | +18.50 [16.09, 20.91] | worse |
| Alaska | D1 | 16.33 | 14.54 | +1.79 [0.37, 2.57] | +1.04 [0.34, 1.85] | worse |
| Alaska | R0 λ 0.25 | 14.96 | 14.54 | +0.42 [−0.31, 1.90] | +2.23 [1.30, 3.12] | ns |
| Alaska | R2 λ 0.25 | 14.13 | 14.54 | −0.41 [−0.60, 0.10] | +0.19 [0.01, 0.40] | ns |
| Lena | D1 | 21.88 | 21.69 | +0.18 [−0.21, 0.42] | −0.45 [−0.99, 0.01] | ns |
| Canada | D1 | 27.94 | 28.16 | −0.22 [−1.13, 0.40] | −1.53 [−2.52, −0.58] | ns |
| Russia_W | D1 | 41.16 | 42.98 | −1.82 [−3.75, −0.27] | −1.45 [−3.09, 0.09] | ns |
| Russia_E | D1 | 30.43 | 29.72 | +0.70 [0.17, 1.36] | +0.69 [−0.02, 1.43] | ns |

- `docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md` C1 의 '35.3 cm 에서 16.3 cm, 물리식 14.5 cm' 는 위 Alaska 행과 일치한다.

### 2.7 라벨 0 의 물리 기준선(비교 상대의 강도)

| 대비 | 원천과 필터 | 셀 가중 Δ [CI] | 블록 등가중 Δ [CI] | 판정어 | Holm p | 공통 CI 셀 가중 / 블록 등가중, 공통 판정 | 의존 | 판정 문서 |
|---|---|---|---|---|---|---|---|---|
| P* − P0(연도 정합 도일 Stefan) | `tables/lgx_tests.csv`, `test_id=L29, contrast=P0@tddm-P0\|n0, scope=MEAN` | −1.00 [−1.35, −0.32] | −0.41 [−0.74, −0.08] | 우세 | 0.0056 | [−1.61, 0.06] / [−0.98, 0.19], 미결정 | 분할 독립 가정 의존 | LG 7.2 L29('P0 보다 우세인 기준선이 있다. P* = P0@tddm') |
| B:ens − P0(물리식·제품 앙상블) | `tables/lgw_bundle.csv`, `ab=AB2, scope=MEAN` | −2.73 [−3.33, −0.91] | +0.05 [−0.94, 1.01] | 미결정 | 1.0(초록 묶음) | [−4.07, 0.15] / [−1.64, 1.63], 미결정 | | WRAPUP J7 1.1 AB2 |
| D0 − P*(직접 ML `catboost_lo` 대 연도 정합 도일 Stefan) | `tables/lgx_tests.csv`, `test_id=L28, item=X9, contrast=앵커 tddm\|D0-P0\|n0, scope=MEAN`, 표준 8열과 `rmse_A`, `rmse_B`, `stability`, `note` | +3.24 [1.86, 4.29] | +2.03 [0.90, 3.12] | 열세 | 0.0050(X9 묶음) | [0.99, 4.99] / [0.17, 3.88], 열세 | | LG 계획서 6A.5 L28(363행), 6A.5a 판정 세부 규칙 L28(399행), 7.2 L28(973행) |

- D0 − P* 행: L28 의 앵커 교체 변형은 P0 를 P0@b 로 바꿔 만든다(LG 계획서 6A.5a, 399행). 따라서 앵커 tddm 변형의 `D0-P0` 는 D0 − P0@tddm(= P*)이다. `rmse_A` 32.886240 은 AB1 의 D0 RMSE(`tables/lgw_bundle.csv`, `ab=AB1, scope=MEAN` 의 `rmse_A`)와 같고, `rmse_B` 29.643470 은 L29 `P0@tddm-P0|n0` MEAN 의 `rmse_A`(P* 의 RMSE)와 같다. `stability` 는 '강건', `note` 는 'κ, λ, 앵커는 비맹검'이다. 지역 행 가운데 열세는 러시아 동부 1/4(두 CI 판정 일치)다.
- L28 은 확인적 가설이 아니다(LG 계획서 6A.2 '확인적 가설' 행(207행): L10, L12, L15, L19, L29, L30). 앵커 변형은 비맹검이다(6A.5a '맹검 표지' 행, 386행). 이 행은 LG 7.2 '원고 문장에 주는 결정' 1(978행, '라벨 0 의 기준 물리식은 P* 로 병기')에 따라 1.3 문장의 비교 상대를 P* 로 병기하는 근거다.
- P* − P0 의 우세는 분할 독립 가정 의존이다. 'P0 보다 강한 물리 기준선 P*' 는 주 CI 판정이며, 본문에서는 표지를 병기한다.

### 2.8 이번 점검(사후, 판정 아님)

1. **독립 지역만 센 위험표**: `scripts/2_evaluation/wf0_reanalysis.py` 의 `load()`, `deltas()` 를 읽기 전용으로 불러 `n=0` 의 대상별 Δ 를 다시 계산했다(파일 저장 없음. 입력 `lg_curve.csv`, `tables/lgd_curve_lic.csv`). 30대상의 `n_worse_2cm` 는 `tables/wf0_risk.csv` 와 같다(18, 13, 5, 2, 1). 독립 7지역만 세면 +2 cm 초과 대상은 D0 5/7, R1_1.0 2/7(Alaska|x +19.94, Russia_E|x +5.01), D1 1/7(Russia_C|x +2.49), R1_0.25 0/7, R2_1.0 0/7 이다. 알래스카 계열(Alaska|x 와 AL-1–AL-6) 비중은 D0 9/18, R1_1.0 8/13, R1_0.25 5/5, D1 1/2, R2_1.0 1/1 이다.
2. **C2 축소 확인**: `tables/wf0_misspec.csv` 에서 `recal_gain`(P0 − P1)과 재보정을 넘어선 ML 몫(P1 − 최선 ML = −`ml_vs_p1`)의 Spearman 순위상관은 −0.08(p 0.67, 30대상)이다. `recal_gain` 과 P0 대비 최선 ML 이득(−`ml_best_delta`)의 순위상관은 0.66(p 8.2e-05)이며 `tables/wf0_meta.json` 의 값과 같다. QA 문서 Q1 의 값과 일치한다.
3. **유의 악화 대상 수(2 cm 문턱 대신 CI 기준)**: 1 과 같은 방식으로 `load()`, `deltas()` 를 읽기 전용으로 불러(파일 저장 없음) `n=0` 의 대상별 `sig_<방법>` 열(원천 곡선의 `sig_p0`)을 셌다. 30대상에서 두 가중 CI 가 모두 0 을 넘은 worse 대상 수는 D0_1.0 14, R1_1.0 10, D1_1.0 8, R1_0.25 6, R2_1.0 5 이다(improve 는 각각 4, 7, 2, 7, 3). 2 cm 문턱 수(18, 13, 2, 5, 1)와 비교하면 증강(D1_1.0 2 → 8)과 R2_1.0(1 → 5)의 차이가 크다. 2 cm 를 넘지 않으면서 worse 인 대상은 D1_1.0 6, R2_1.0 4, R1_0.25 2, D0_1.0 1, R1_1.0 0 이다. 독립 7지역만 세면 worse 는 D0_1.0 3/7(Alaska|x, Russia_C|x, Russia_E|x), R1_1.0 2/7(Alaska|x, Russia_E|x), D1_1.0 2/7(Alaska|x, Russia_C|x), R2_1.0 1/7(Russia_C|x), R1_0.25 0/7 이다. worse 대상 가운데 알래스카 계열(Alaska|x 와 AL-1–AL-6)은 D0_1.0 10/14, R1_1.0 8/10, R1_0.25 6/6, D1_1.0 5/8, R2_1.0 4/5 이다. 이 값은 사후 계산이며 판정어를 붙이지 않는다. `sig_p0` 는 주 CI 기준이고 공통 CI 는 반영하지 않는다.

---

## 3. 근거 실험

| 새 이름 | 옛 id | 이 주장에서 쓰는 가설·대비 | 계산 환경 | 판정 문서 |
|---|---|---|---|---|
| T1_transfer_label_grid | LG | L1(직접 ML, n ≤ 40), AB1 의 원천 조각, 대상별 곡선(2.6절) | Rescale(작업 ZovWo) | LG 계획서 7.1 |
| T2_transfer_structure_placebo_baselines | LGX | L10, L11, L15, L29, L30, L28(앵커 tddm 변형의 D0 − P*), L16(`D1@r30` − P0, 참고) | Rescale 조각, 로컬 재집계(h42) | LG 계획서 7.2 |
| T3_learners_foundation_and_nn | LGT, LGF | LGT L34(a)(b), LGF-F1, F1 보조 (a), F5, N1 | 로컬 GPU(RTX 3090) | LG 계획서 7.4, LGF 계획서 10.2·10.3 |
| T4_new_regions | LGD | L1e(P4, PE1, PE2) | 새 지역 로컬, P4 Rescale | LG 계획서 7.3 |
| T6_abstract_contrasts_and_map | LGW(+WRAPUP) | AB1, AB2, AB3, 단계 T0 요약, SC1w | Rescale 조각, 로컬 재집계(h39) | WRAPUP 결과 판정 기록(J7) |
| W1_label0_risk_table | WF0 | 위험표, 계수 오차 순위상관 | 계산 없음(곡선 재분석) | WF 계획서 2절(사후 서술) |
| H2_master_M1, H0_contest(이력) | M1, P2·W3 | AB1·AB3 의 비맹검 사전 정보(방향을 이미 보았음) | 해당 없음 | WRAPUP 1.1 표의 맹검 열 |

---

## 4. 그림

| 파일 | 패널 | 내용 | 그림 값 원천 |
|---|---|---|---|
| `outputs/figures/paper/v2/Fig6_label0_uncertainty.{pdf,svg,png}` | a | 라벨 0 포레스트. 기준선(B:ens, P*), Rescale 직접 ML(catboost_lo, catboost, rf, catboost_tuned, F1k), 로컬 LGT(TabPFN, 병기 CatBoost), 로컬 LGF-F(TabICL 두 조건), 로컬 LGF-N(조정 신경망 4종과 기본판). 플랫폼별 블록을 나누고 차를 계산하지 않는다 | `tables/fig6_a.csv`, `tables/fig6_a_registered.csv`, `outputs/figures/paper/source_data/v2/Fig6_a*.csv`. 생성 기록 `data/processed/paper_figs/v2_data_fig6_meta.json` 의 입력 sha256 6건(lgw_bundle, lgx_tests, lg_tests, lgt_tests, lgf_tests, lgfn_tests)은 `tables/MANIFEST.csv` 의 사본 sha256 과 같다. `tables/fig6_a.csv` 의 `ci_dependence` 열은 P*, catboost_lo, catboost, catboost_tuned, TabPFN, LGT 병기 CatBoost, TabICL 두 조건, LGF-F 병기 CatBoost 두 조건, LGF-N 기본판 `mlp`·`tabm` 행(21행 가운데 12행)이 '분할 독립 가정 의존'이다. D0 − P*(L28) 행은 패널에 없다 |
| `outputs/figures/paper/v2/Fig3_physics_use.{pdf,svg,png}` | a, b | a: 위약 대조(L15, AB3 표지). b: 결합 구조(L10 의 R0 − F1k n 0 포함) | `tables/fig3_a.csv`, `outputs/figures/paper/source_data/v2/Fig3_a.csv`, `Fig3_b_L10.csv` |
| `outputs/figures/paper/v2/Fig7_deployment.{pdf,svg,png}` | b, d | b: 단계 T0 의 R0 − P0 지역 칸. d: 초록 대비 AB1, AB3 | `outputs/figures/paper/source_data/v2/Fig7_b_cells.csv`, `Fig7_b_summary.csv`, `Fig7_d.csv` |
| `outputs/figures/paper/v2/Fig2_label_curve.{pdf,svg,png}` | 곡선의 n = 0 끝점 | D0·R0·R2 의 P0 대비 곡선 시작점(source data 파일 이름 기준 a–d 지역, e–f 풀) | `tables/pool_fixed_curve.csv`, `outputs/figures/paper/source_data/v2/Fig2_a_d.csv`, `Fig2_e_f.csv` |
| 없음 | | WF0 위험표는 현재 어떤 그림에도 없다 | `tables/wf0_risk.csv` |

**재구성안 메모**(`docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md`)
- 4절 표의 R2 행이 C1 의 결과 절이다. 근거는 AB1, L30, LGT L34(a), LGF-F1·N1, AB3, WF0 위험표이고 그림은 Fig 6a, Fig 3a 다. 절 제목 '물리 정보가 ML 을 안전하게 만든다'는 1.2절에 따라 '라벨 0 전이에서 물리 정보가 줄이는 손해' 같은 명사형 제목으로 바꿔야 한다.
- 5절: Fig 6 은 그대로 두고, Fig 7 은 '충분 라벨 지역과 워크플로'로 바꾸며 AB 묶음 표는 SI 로 옮긴다. 이 안을 따르면 Fig 7 b(T0)·d(AB1, AB3)는 SI 로 간다.
- WF0 위험표의 위치는 재구성안에 정해져 있지 않다. Fig 6 에 작은 패널로 넣거나 SI 표로 두는 선택은 그림 배치와 함께 사용자 결정 사항이다(8절).
- 옛 판 `outputs/figures/paper/Fig6_label0.{pdf,png}`(09-26 판)는 v2 로 대체되었다.

---

## 5. 단서와 쓰지 않을 문장

### 5.1 단서

1. **4지역 평균의 판정이다.** AB1 의 지역 행에서 주 CI 의 열세는 러시아 동부 1/4 이고, 이 열세는 분할 독립 가정 의존이다(공통 CI 에서는 4지역 모두 미결정). 캐나다는 점 추정이 −1.56 cm(물리식보다 작은 오차)다(2.2절). AB1 MEAN 의 열세도 분할 독립 가정 의존이다(공통 CI 셀 가중 [0.31, 4.19], 블록 등가중 [−0.21, 3.43], 미결정).
2. **원천 계수가 크게 틀린 지역은 반대다.** 티베트에서 직접 ML 은 P0 보다 59.98 cm 낫다(L1e, PE2). 이 행의 원천 표지는 '예측된 결과(비맹검); 비맹검(라벨 통계 열람)'이고, PE1·PE2 행은 '교차 환경; (i) CatBoost 불통과'다(`tables/lgd_tests_lic.csv` 의 `flags`, `cross_env`, LG 7.3 '표지' 항). L1e 의 지지는 이 1지역을 허용 범위 안에서 받아들인 결과다.
3. **3지역 보조 열의 크기는 학습기에 따라 다르다.** 3지역 보조 열에서 시험한 학습기의 D0 − P0 는 +0.31 – +12.25 cm 이고, TabICL v2 원천 전체 컨텍스트(+0.31)는 미결정, 축소 FT-T(+1.17)의 열세는 분할 독립 가정 의존이다. 축소 FT-T 와 TabICL 원천 전체는 3지역 값이 4지역 값보다 작다(+1.17 대 +4.13, +0.31 대 +0.85). 나머지 학습기는 3지역 값이 더 크다(2.2절).
4. **증강의 손해 감소는 사후 서술이며 없애지 못한 곳이 있다.** 알래스카에서 D1 은 P0 보다 +1.79 cm [0.37, 2.57] 나쁘다(2.6절). 증강의 P0 대비는 등록 가설이 아니다(2.4절 끝). 30대상에서 증강의 유의 악화(worse) 대상은 8개로 2 cm 문턱 수(2개)보다 많다(2.8절 3, 사후 계산).
5. **저가중 잔차의 동등은 4지역 평균의 주 CI 에서만 확인되었다.** 4지역 평균의 동등(R0[T], R0[C], R0[I], 구성 고정 풀 곡선 R0·R2)은 모두 분할 독립 가정 의존이다(공통 CI 에서는 미결정). 3지역 열은 미결정 또는 열세(TabICL +0.64, 이 열세도 분할 독립 가정 의존)이고, 알래스카 칸의 블록 등가중 Δ 는 +2.23 [1.29, 3.13] 이다(2.4절). 동등 판정의 Holm 보정은 없다.
6. **위험표는 사후 서술이다.** 문턱 2 cm 는 WF0 에서 정했고 점 추정 문턱이라 CI 를 보지 않는다. 30대상은 독립이 아니다(독립 7지역). 직접 ML 의 18대상 가운데 9대상, 저가중 잔차의 5대상 모두가 알래스카 계열이다(2.8절 1). 두 가중 CI 가 모두 0 을 넘은 대상 수는 D0 14, R1_1.0 10, D1 8, R1_0.25 6, R2_1.0 5 이다(2.8절 3, 사후 계산). 위험표의 R2 는 λ 1.0 이다.
7. **학습기 수의 한정.** '10종'은 주 CI 4분 판정 기준의 수다. 두 CI 판정이 같은 학습기는 rf, mlp*, tabm*, ftt*, realmlp* 5종(그리고 보조 F1k)이다. catboost_lo, catboost, catboost_tuned, TabPFN v2, TabICL v2(두 조건)의 열세는 분할 독립 가정 의존이다(2.1절). TabPFN v2 의 열세는 LGT 묶음 Holm p 0.1168 로 보정 전 유의이기도 하다.
8. **증강 이득의 정보 출처는 부분적이다.** L15 는 혼재다. √TDD 선형 변환 위약(tddlin)과의 차이는 n 0 에서 0.48 cm(크기 0.5 cm 미만), n 10 에서 동등이다(2.3절). 2.3절의 판정은 두 CI 에서 같다.
9. **비교 상대는 P0 다.** 라벨 0 에서 P0 보다 강한 P*(−1.00 cm, 분할 독립 가정 의존)가 있다(L29). 직접 ML(catboost_lo) 대 P* 는 L28 앵커 tddm 변형 행에서 +3.24 cm [1.86, 4.29](블록 등가중 +2.03 [0.90, 3.12])이고, 열세(두 CI 일치), Holm p 0.0050(X9 묶음)이며, 앵커 변형은 비맹검이다(2.7절). L28 은 확인적 가설이 아니다. Fig 6a 는 P* − P0 행만 두고 D0 − P* 행은 두지 않는다.
10. **라벨 0 전용이다.** 증강은 라벨이 생기면 추가 이득이 없다(AB6 R2 − R1 n 10 동등, LG 7.1 L2 '증강은 라벨 0 전용'). TabICL 직접 ML 은 라벨 10개에서 4지역 평균으로 P0 보다 2.91 cm 낫다(2.2절 한계 행). 이 우세는 분할 독립 가정 의존이다(공통 CI 셀 가중 [−4.07, −0.31], 블록 등가중 [−3.15, 0.39], 미결정).
11. **플랫폼.** Rescale 행과 로컬 GPU 행은 서로 빼지 않는다. 같은 컨텍스트 CatBoost 값은 LGT 와 LGF-F 에서 같다(`v2_data_fig6_meta.json` 의 `info_fig6a_catboost_ctx_LGT_vs_LGF_identical`).
12. **맹검.** AB1 은 비맹검 부분 포함(M1, P2·W3 에서 방향을 보았다), AB3 은 재현(비맹검)이다(WRAPUP 1.1 맹검 열).

### 5.2 쓰지 않을 문장

| 쓰지 않을 문장 | 사유와 근거 |
|---|---|
| '물리 증강과 저가중 잔차는 ML 을 안전하게 쓰는 방법이다', '안전하다' | 라벨 0 의 D1·R2 비열등 등록 가설이 없고, SC1w 는 '안전성 미확인(4/10)'이다(QA Sci Rep 평가, WRAPUP 3.3 문장 규칙) |
| '손해 없는 선택' | 위험표에서도 D1 2/30, R2(λ 1.0) 1/30, 저가중 잔차 5/30 이 2 cm 넘게 나빠졌다. 두 가중 CI 가 모두 0 을 넘은 대상은 D1 8, R2(λ 1.0) 5, 저가중 잔차 6 이다(2.8절 3, 사후 계산) |
| '물리 증강과 저가중 잔차는 손해를 줄였다'(단정형, 한정어 없이) | 라벨 0 에서 D1·R2 의 P0 대비, D1 − D0·R0 − D0 대비를 판정한 등록 가설이 없다(2.4절 끝). 근거는 사후 서술(WF0)과 대상별 곡선뿐이다. '사후 서술에서 2 cm 넘게 나빠진 대상 수가 적었다'로 쓴다 |
| '시험한 학습기 10종 모두 물리식보다 나쁘다'(단정형) | 두 CI 판정이 같은 학습기는 5종이다. 나머지 5종은 분할 독립 가정 의존이다(2.1절, LG 계획서 202행) |
| '저가중 잔차는 물리식과 동등하다'(단정형) | 4지역 평균의 동등은 모두 공통 CI 에서 미결정이다(2.4절) |
| '3지역(알래스카 포함)에서는 모든 학습기의 손해가 더 크다' | 축소 FT-T 와 TabICL 원천 전체 컨텍스트는 3지역 값이 더 작고, TabICL 원천 전체는 미결정이다(2.2절) |
| '직접 ML 은 모든 지역에서 물리식보다 나쁘다' | 주 CI 의 지역 단위 열세 1/4(분할 독립 가정 의존, 공통 CI 0/4), 티베트는 반대(2.2절) |
| '증강은 물리식 수준으로 되돌린다'(한정어 없이) | 알래스카 D1 +1.79 cm 열세(2.6절) |
| '증강 이득은 물리 정보에서만 온다' | L15 혼재, tddlin n 10 동등 |
| '표형 파운데이션 모델은 물리식을 넘지 못한다'(계열 일반화) | LGF 계획서 9.2. 'TabPFN v2, TabICL v2 두 모델'로 한정한다 |
| 'TabICL 은 라벨 0 에서 물리식을 넘지 못한다'(조건 없이) | LGF 9.2. 두 컨텍스트 조건을 밝힌다 |
| '신경망을 충분히 조정했다' | LGF 9.2. '원천 지역 하나 제외 교차검증으로 고른 설정'으로 쓴다 |
| '라벨 10개에서도 직접 ML 은 물리식을 넘지 못한다'(학습기 무관) | TabICL n 10 −2.91 cm 우세(LGF 10.2 F1 보조 (a), 분할 독립 가정 의존) |
| '원천 계수 Stefan 이 가장 강한 물리 기준선이다' | L29, P* = P0@tddm(NOVELTY 쓰지 않을 문장 표). P* − P0 의 우세는 분할 독립 가정 의존이다 |
| '직접 ML 대 P* 의 대비는 없다' | L28 앵커 tddm 변형 행이 D0 − P* 다(+3.24 cm, 열세, 두 CI 일치, 2.7절) |
| 초록에서 위험표에 판정어를 붙이는 문장 | WRAPUP 1.1(묶음 밖), WF 계획서 3절(사후 서술) |
| '30개 독립 지역' | 독립 지역은 7개다(2.5절) |
| 'ML 이 물리식을 넘는다'(조건 없이) | `docs/RESEARCH_OVERVIEW_2026-10-02.md` 10절 |

---

## 6. 이 주장을 바꿀 수 있는 예정 실험(X 묶음)

2026-10-04 기준 X 묶음의 설계 문서는 저장소에서 찾지 못했다. 아래 '설계'는 이름과 이번 요청의 질문에서 추론한 범위이며 [미확인]이다.

| 새 이름 | C1 에 주는 영향 | 바뀔 수 있는 문장 |
|---|---|---|
| XF_new_regions | 독립 지역이 늘면 L1 형식 판정(직접 ML 열세)과 위험표의 독립 지역 분모가 바뀐다. 티베트처럼 원천 계수가 크게 틀린 지역이 늘면 예외가 허용 범위(⌊k/4⌋)를 넘을 수 있다. 결과를 보기 전에 '라벨 0 에서 D1·R2·R0(λ 0.25)는 P0 대비 비열등(δ 0.5 cm)'을 등록해 새 지역에서 시험하면 '안전' 계열 문장의 근거를 처음 얻는다. 기존 지역은 이미 열람했으므로 이 시험에 쓸 수 없다 | 2.1·2.2절의 지역 수, 5.2절의 '안전' 문장 |
| XG_product_comparison | 라벨 0 의 비교 상대가 바뀔 수 있다. WRAPUP 10절에 등록만 된 설계(채점 셀에서 공개 ALT 제품 값 추출, 학습 자료 중복 표시)가 있다. 제품이 P0·P* 보다 낫다면 라벨 0 기본값 문장('라벨이 없으면 물리식이 기준')에 제품을 병기해야 한다 | 1.3절 비교 상대, 워크플로 표 라벨 0 행 |
| XB_multisource_stacking | 라벨 0 에서는 대상 라벨로 가중치를 배울 수 없어 원천 가중 또는 고정 가중이 된다. 고정 가중 앙상블 B:ens 는 미결정이었다(2.7절). 더 강한 라벨 0 물리·제품 기준선이 생기면 직접 ML 의 손해 크기 기준이 바뀐다 | 2.1절의 기준선 |
| XC_workflow_end_to_end | 라벨 0 단계 레시피(P0 또는 증강·저가중 잔차)를 순차 워크플로 안에서 평가하면 누적 손해를 직접 잴 수 있다 | 워크플로 표 라벨 0 행의 근거 |
| XJ_tempderived_aux_labels | 지온 유도 라벨을 보조 라벨로 쓰면 '라벨 0' 지역 일부(몽골 등)가 약한 라벨 지역이 된다. 보조 라벨이 물리 유사라벨보다 나은지에 따라 증강의 역할이 바뀐다 | 1.3절의 '물리 유사라벨' 문장 |
| XI_climate_extrapolation_retest | 학습 범위 밖 기후에서 직접 ML 의 손실이 판정되면 C1 의 '라벨이 없을 때' 를 '학습 범위 밖일 때'로 넓힐 근거가 된다. WF10 은 판정 불가였다(RESEARCH_CLAIMS C8) | C1 의 적용 범위 |
| XE_hires_covariates | 지역 내 실험으로 예정되어 라벨 0 전이 판정에는 직접 영향이 작다. 고해상 입력이 전이 입력에 들어가면 직접 ML 대 P0 의 차가 바뀔 수 있다 [미확인] | 없음(전이 입력에 넣을 때만) |
| XH_validation_ladder | 무작위 분할부터 지역 홀드아웃까지의 순서를 보이는 그림(C7)이다. C1 의 라벨 0 판정은 바꾸지 않는다 | 없음 |
| XA_c2_gain_decomposition | 라벨 0 에서 P1 = P0 이므로 C1 을 바꾸지 않는다 | 없음 |
| XD_placement_policy | 첫 라벨의 배치 지침(워크플로 표 라벨 0 행의 '다음 관측')만 관련된다 | 워크플로 표 라벨 0 행의 관측 열 |

### 6.1 X 묶음 결과 근거(2026-10-05 열람)

판정 정본은 `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 8절(추가 등록 XK·XL 은 `docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md` 결과 절)이다. 수치는 cm 이고 셀 가중 / 블록 등가중, 대괄호는 95 % CI 다. Δ 는 음수면 앞 방법의 오차가 작다. 이 표는 근거 색인이며 1절의 주장 문장은 고치지 않았다(원고 작업에서 정한다).

| 실험·가설 | 판정 | 수치 | 근거 표(원래 경로) | 이 주장과의 관계 |
|---|---|---|---|---|
| XG-1c: B:cci5y_raw − P0, n 0, 주 4지역(계획 8.7) | 열세(Holm p 0.0006, 5·25 km 블록 마스크 같은 갈래, 등록 이탈 WRAPUP 10) | +23.79 [+15.34, +30.16] / +19.17 [+14.83, +23.72]. 지역 열세: 레나 +1.19, 캐나다 +82.38, 러시아 E +5.84 | `data/processed/xbatch/XG_product_comparison/sealed/xg_tests.csv`, `xg_confirm_rows.csv` | 라벨 0 에서 CCI v5 원값을 그대로 쓰면 원천 계수 Stefan 보다 오차가 컸다(워크플로 라벨 0 행의 기준선 비교) |
| XG-1w: B:wei_raw − P0, n 0, 레나·캐나다 | 미결정(Holm p 0.156 / 0.383) | 5 km 마스크 +1.44 [−0.02, +4.30] / +4.09 [+2.10, +6.12], 제품 결측 대체 47.4 % | 같음 | Wei v2 와 P0 의 차이를 확인하지 못했다. Wei 는 대상 지역 안 CALM 지점도 학습에 썼다 |
| XB-5: Stack0 − P0, n 0(보조, 계획 8.2) | 주 4지역 미결정, PE1 열세, PE2 우세(확인적으로 세지 않음) | 주 4지역 +0.51 [+0.09, +0.69] / +0.09 [−0.14, +0.31], PE1 +0.75 / +0.35, PE2 −1.32 / −1.64 | `data/processed/xbatch/XB_multisource_stacking/sealed/xb_tests.csv` | 라벨 0 적층(원천 지역 하나 제외 가중)의 P0 대비 이득은 풀에 따라 달랐다 |
| XJ-1·XJ-2: 원천 쪽 지온 유도 보조 행, n 0·10(계획 8.5) | 시험하지 않음 | 약관 근거 미기록 | `data/processed/xbatch/XJ_tempderived_aux_labels/sealed/xj_hyp.csv` | 라벨 0 직접 ML 에 보조 행을 더한 효과는 시험하지 않았다 |
| XJ-3: 티베트 P1_aux(w) − P1, n 3·10 | 우세 6행(Holm p ≤ 0.004) | w 1: n 3 −83.17 [−90.21, −73.41] / −66.86, n 10 −37.50 [−43.30, −28.88] / −20.57 | 같음, `xj_tests.csv` | 지온 유도 보조 라벨이 소수 라벨 계수 보정의 전이 오차를 줄였다. 정의가 달라(L39) 라벨 확충 근거로 일반화하지 않는다 |
| XC S-XC10: A1(워크플로) − P0, 비열등 0.5 cm(계획 8.10) | n 10 우세·비열등, n 40·160 미결정·비열등 | n 10 −1.87 [−2.24, −1.33] / −1.58; n 40 −0.22 / −0.14; n 160 −0.06 / −0.20. 러시아 E 의 n 10 행 열세 +2.30 | `data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_contrasts_aux.csv` | 워크플로의 오차 증가는 원천 계수 Stefan 대비 0.5 cm 안이었다(보조, 확인적으로 세지 않음) |
| XC S-XC4 (a): P0 보다 2 cm 넘게 나빠진 대상 수(A1 / A2 / A3, 독립 지역 7곳) | 사전 고정 보고 항목 | n 10 1 / 1 / 0(러시아 E), 전량 1(알래스카 x) / 2 / 1; 21대상 전량 1 / 4 / 3 | `data/processed/xbatch/XC_workflow_end_to_end/sealed/xc_harm_a.csv` | 손해 대상 수 서술 |

---

## 7. 파일 목록

- `tables/MANIFEST.csv`: 사본 18건의 원본 경로, 사본 경로, sha256(원본과 같음을 확인), 행 수(CSV 는 pandas 행 수, JSON 은 1), 바이트.
- 사본: `lgw_bundle.csv`, `lgw_scenarios_summary.csv`, `lgw_scenarios.csv`, `lgw_tests.csv`, `lgx_tests.csv`, `lgt_tests.csv`, `lgf_tests.csv`, `lgfn_tests.csv`, `lg_tests.csv`(LG 본 실행), `lgd_tests_lic.csv`, `lgd_curve_lic.csv`, `wf0_risk.csv`, `wf0_misspec.csv`, `wf0_meta.json`, `fig6_a.csv`, `fig6_a_registered.csv`, `fig3_a.csv`, `pool_fixed_curve.csv`. 모두 집계 표이며 셀 단위 라벨 자료는 없다.
- 복사하지 않은 원천: `results/rescale_lg/data/processed/lg/lg_curve.csv`(7,072,630 bytes, 5 MB 상한 초과). 2.6절과 2.8절에서 행 필터로 읽었다.
