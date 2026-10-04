# C2 편향 진단: 근거 색인

**성격**: 주장 C2(라벨 10개 편향 진단과 이득의 관계)의 문장, 근거 수치, 근거 실험, 그림, 단서를 한곳에 모은 색인이다. 2026-10-04 에 작성했다. 원래 파일은 옮기거나 고치지 않았다. 같은 날 점검 지적 16건을 원천과 다시 대조해 반영했다(1.1–1.3, 2.1 의 A6·A11–A14, 2.2, 2.3, 4.2, 4.3, 5.1, 5.2, 6절. 표 사본과 `tables/MANIFEST.csv` 는 바꾸지 않았다). 근거로 쓴 집계 표의 사본은 `tables/` 에, 원본 경로와 sha256 은 `tables/MANIFEST.csv` 에 있다.

**수치 규칙**: 2절의 수치는 표에 적은 CSV·JSON 을 pandas(`OMP_NUM_THREADS=1`)로 읽어 소수 둘째 자리로 반올림한 값이다. p 값은 파일 값을 유효숫자 세 자리로 적었다(파일 값이 그보다 짧으면 그대로). 재표집 횟수는 파일 값 그대로 적었다. 문서 서술에서 옮긴 값은 문서 경로와 절을 적었다. 2.3 절은 이번 작성 때 계산한 사후 점검이고 판정이 아니다(블록 재표집 CI 없음).

**부호 규약**: Δ(대비 'A − B')는 음수가 개선이다. '재보정 몫'과 'ML 몫'은 아래 기호 표의 정의대로 양수가 개선이다. 본문에서 감소량을 말할 때는 '오차를 x cm 줄였다'(x 양수) 또는 'Δ = −x cm' 로 쓴다.

**기호**(이 폴더 안에서만 쓴다)

| 기호 | 뜻 |
|---|---|
| P0 | 원천 지역에서 정한 계수 E0 를 그대로 쓴 Stefan 식(ALT = E × √TDD) |
| P1 | 대상 라벨로 E 를 다시 맞추되 원천 E0 쪽으로 수축한 Stefan 식(κ = 10) |
| P2 | 대상 라벨만으로 E 를 최소제곱 재적합한 Stefan 식 |
| R1 | P1 을 앵커로 두고 CatBoost 잔차를 더한 예측(앵커 + λ × 잔차). 기준 λ 0.25 |
| W | 규칙 W. 대상 라벨 안 5겹 블록 교차검증으로 {P0, P1, P2, R1(λ 0.25, 1.0), R2, D1} 가운데 오차가 가장 작은 방법을 고른다. 라벨 10개 미만이면 P1(`docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 2절 WF4) |
| 진단값 | 라벨 10개에서 계산한 \|평균(E0·√TDD − ALT)\|. 추출·분할 평균(같은 문서 개정 이력 9) |
| Δ | 두 방법의 RMSE 차(cm). 음수면 앞 방법의 오차가 작다 |
| 재보정 몫 | P0 − P1(cm, 양수면 재보정이 오차를 줄였다) |
| ML 몫 | P1 − ML(cm, 양수면 ML 이 재보정 물리식보다 오차를 줄였다) |
| 판정어 | 4분 판정(우세·열세·동등·미결정), `sig` 열의 improve·ns·worse(셀 가중과 블록 등가중 두 CI 가 모두 0 을 제외할 때 improve·worse) |

---

## 1. 주장 문장

### 1.1 현행 문장

`docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md` 3절 C2 의 문장은 다음과 같다. `docs/RESEARCH_OVERVIEW_2026-10-02.md` 5절 C2 제목(99행)은 같은 취지의 문장이고 문구가 일부 다르다('라벨 10개로 진단할 수 있다', '그 정도를' 없음). `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 4절(32행)은 결과 R3 절에 C2 를 배정할 뿐 C2 문장을 적지 않는다('C2, C3' 표지와 근거 목록 AB4, AB5, L10·AB7, WF0 ρ 0.66, WF4-c ρ 0.38, SC1w 만 있다).

> C2. ML 의 이득은 물리 계수가 틀린 정도에 비례하고, 라벨 10개로 그 정도를 진단할 수 있다.

현행 근거 항목은 넷이다. (1) 재보정 이득과 최선 ML 이득의 순위상관 0.66(WF0, 사후 서술). (2) 라벨 10개 편향 크기와 규칙 W 이득의 ρ 0.38 [0.17, 0.55](WF4-c 지지), 부호 있는 편향 ρ −0.01. (3) 티베트 라벨 10개 R1 − P1 −21 cm, 러시아 서부 −14 cm, 캐나다 지역 내 Stefan·CCI 앵커 + 잔차의 재보정 대비 점 추정 −2.9 에서 −6.6 cm. (4) 계수가 맞는 지역(알래스카, 레나, 러시아 동부)에서는 이득이 작다.

**(4)의 점검(이 폴더)**: 이 README 는 캐나다를 '지역 계수는 맞다'고 판단하는 데 `logE_ratio_own`(log(E_own/E0))을 썼다(CA6). 같은 지표(`results/rescale_lg/data/processed/lg/lg_targets.csv`, `part == 'cpu'`, 모드 x, 대상별 첫 행, 소수 셋째 자리)는 러시아 E 0.171(E_own 1.887, E0 1.591), 레나 −0.122, 알래스카 0.072, 캐나다 0.003 이다. 러시아 E 의 값은 캐나다보다 크고 CA-3(0.163)과 비슷하므로 러시아 E 를 '계수가 맞는 지역'으로 묶을 근거가 없다. 세 지역에 공통인 것은 라벨 전량 재보정 몫(P0 − P1)이 작다는 점이다(알래스카 −0.07, 레나 −0.02, 러시아 E +0.09 cm, `data/processed/wf/wf0_misspec.csv` 의 `recal_gain`). 러시아 E 의 재보정 이득이 작은 이유로 라벨 반쪽(A)과 채점 반쪽(B)의 E 차(`logE_ratio_AB` 분할별 −0.41 – +0.23)를 생각할 수 있으나 확인하지 않았다[미확인: 원인 분석 없음. 캐나다도 같은 열이 −0.34 – +0.35 다]. 원고에서는 (4)를 '라벨 전량 재보정 이득이 작은 지역(알래스카, 레나, 러시아 동부)'으로 바꾸거나 러시아 E 를 뺀다.

### 1.2 요구된 축소

`docs/QA_FINAL_REVIEW_2026-10-02.md` 가 요구한 축소는 1–3번이다. 4·5번은 QA 가 지적 대상으로만 적은 항목(400행 'Sci Rep 리뷰어 관점 평가' 표 '결과의 지지' 행: "C2(재보정 혼입), '안전' 표현, WF 의 사후 설계는 지적 대상")에 등록 문서의 규칙과 이 README 의 권고를 붙인 것이다.

1. **재보정 혼입**(Q1 '좁혀야 할 점' 2번, 'C2 점검(이번 점검, 사후)'): 0.66 과 0.38 은 모두 P0 대비 이득이다. R1 은 P1 을 앵커로 쓰므로 상관의 상당 부분은 재보정 몫이다. 재보정을 넘어선 ML 몫과 계수 오차의 순위상관은 −0.08(p 0.67, 30대상)이다(QA 본문은 이 몫을 '최선 ML − P1' 로 적었으나 계산은 이 README 의 부호 P1 − 최선 ML 과 같다. 2.3 의 재현 참조). 러시아 서부의 −14 cm 는 R1 − P0 이다. 티베트·캐나다는 P1 자체가 나쁜 곳이라 P1 대비 이득이 크게 나온다.
2. **권고 문장**(Q1): '라벨을 쓰는 보정(재보정 + 잔차)의 이득은 원천 계수 오차의 크기에 비례하고, 라벨 10개로 그 크기를 진단할 수 있다. 재보정을 넘어선 ML 의 추가 이득은 계수 오차 크기와 상관이 없었다.'
3. **강화 방법 3**(Q12): C2 문장을 재보정 몫과 ML 몫으로 나눠 고친다.
4. **'안전' 표현**(QA 는 지적 대상으로만 적었다): 규칙의 등록 출처는 `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 다. 3.3 SC1w 의 문장 규칙(305행: 열세 칸도 없고 비열등 칸도 적을 때 '오차를 키우지 않았다'는 쓰지 않는다), 11.2 쓰지 않을 문장 표(711행: '배포 레시피는 오차를 키우지 않았다(비열등 판정 없이 열세가 없다는 이유만으로)'), 결과 판정 기록(J7)의 틀 (2) 결론(846행: '라벨 3–10개로 재보정하면 안전하다' 를 쓰지 않는다)이다. C2 의 근거에는 비열등 판정이 없다. J7 SC1w 판정은 '안전성 미확인(비열등 칸 4/10)'이다.
5. **WF 사후 설계**(QA 같은 행, Q12 부족한 부분 3번 'WF 는 결과를 본 뒤 설계했다'): WF0 은 사후 서술이고 WF4-c 는 결과 열람 뒤 설계다(5.1 의 1번). 원고에 이 표지를 그대로 적는 것은 이 README 의 권고다.

### 1.3 이 폴더의 제안 문장

QA 권고 문장의 첫 문장은 유지하되 '비례' 를 '순위상관' 으로 바꾼다. Spearman 순위상관은 단조 관계를 보일 뿐 비례를 보이지 않는다. 둘째 문장은 2.3 절의 사후 점검에 맞춰 고친다. 재보정을 넘어선 ML 몫과 계수 오차의 순위상관은 계수 오차의 정의에 따라 −0.08 에서 0.39 사이였다. 따라서 '상관이 없었다' 대신 '확인하지 못했다' 로 쓴다.

AB8(R1 − P0, 라벨 전량)을 인용하는 문장은 등록 규칙의 한정어를 함께 쓴다. LG 계획서 7.2 '원고 문장에 주는 결정' 3번은 러시아 W 치우침과 알래스카를 넣은 3지역 평균의 열세를, WRAPUP 결과 판정 기록(J7) 1.1 표 아래 문단은 P* 병기(R1 − P* −1.64 우세)와 '평균'·'지역 수'를 요구한다. 분해 수치(R1 − P1)는 규칙 W 가 아니라 고정 R1(λ 0.25) 값이므로 규칙 W 문장과 분리한다. 재보정 몫이 감소의 대부분이라는 문장은 주 4지역 평균(러시아 W 지배), 러시아 W, 티베트에서만 성립하므로 지역 한정 없이 쓰지 않는다(2.3 비율 표).

**국문안**

> 라벨 10개로 잰 원천 계수 Stefan 식의 편향 크기는, 대상 라벨 안 교차검증 규칙으로 고른 보정 방법이 라벨 전량에서 원천 계수 Stefan 식보다 줄인 오차와 양의 순위상관을 보였다(ρ 0.38 [0.17, 0.55], 28대상, 결과 열람 뒤 설계). 편향의 부호는 이득을 예측하지 않았다(ρ −0.01). 라벨 전량에서 잔차 ML(λ 0.25)은 주 4지역 평균으로 원천 계수 Stefan 식보다 오차를 2.64 cm 줄였고(Δ = −2.64 [−3.48, −1.85] cm, 개선이 유의한 지역 1/4), 그 가운데 재보정 물리식 대비 감소는 0.40 cm 였다. 이 평균은 러시아 서부(Δ = −13.98 cm)가 지배한다. 캐나다에서는 오차가 3.38 cm 늘었고, 레나·캐나다·알래스카 3지역 평균에서도 0.77 cm 늘었다(Δ = +0.77 [0.02, 1.28] cm). 연도 정합 도일을 쓴 원천 계수 Stefan 식(P*) 대비로는 오차를 1.64 cm 줄였다(Δ = −1.64 [−2.95, −0.74] cm). 재보정을 넘어선 ML 몫과 계수 오차의 관계는 확인하지 못했다.

**영문안(원고 초안용)**

> The magnitude of the source-coefficient bias measured with ten target labels was rank-correlated with the error reduction that the method chosen by within-target cross-validation achieved over the source-coefficient Stefan model with all labels (Spearman ρ = 0.38, 95% CI 0.17 to 0.55; 28 targets; hypothesis registered after earlier results were seen). The sign of the bias did not predict the gain (ρ = −0.01). With all labels, residual learning (λ = 0.25) reduced RMSE relative to the source-coefficient Stefan model by 2.64 cm on average over four main regions (95% CI 1.85 to 3.48 cm; significant improvement in one of four regions), and by 0.40 cm relative to the recalibrated Stefan model. The four-region mean was dominated by western Russia (13.98 cm reduction). RMSE increased by 3.38 cm in Canada and by 0.77 cm on average over Lena, Canada and Alaska (95% CI 0.02 to 1.28 cm). Relative to the source-coefficient Stefan model driven by year-matched degree days (P*), residual learning reduced RMSE by 1.64 cm (95% CI 0.74 to 2.95 cm). We could not establish a relation between coefficient error and the gain of machine learning beyond recalibration.

근거: A3, A4(규칙 W 와 편향), A8, A9, A11(AB8 지역 행), A13(3지역 평균), A14(P*).

**쓰임**: 진단값은 '이 지역에서 라벨과 계수 보정에 투자할 가치가 큰가' 를 가늠하는 데 쓴다. 'ML 을 더할 가치' 를 진단한다고 쓰지 않는다(5.2).

---

## 2. 근거 표

경로는 원본 경로다. 같은 파일의 사본이 `tables/` 에 있다(`tables/MANIFEST.csv`). `n == -1` 은 라벨 전량이다. WF 표의 `lam == -1` 은 교차검증 λ 의 키다(`scripts/3_deep_learning/h54_workflow.py` 226행 `LAM_CV = -1.0`).

### 2.1 판정·서술 기록이 있는 근거

| 번호 | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| A1 | 재보정 이득(P0 − P1)과 최선 ML 이득(P0 − 최선 ML)의 순위상관, 라벨 전량 | `data/processed/wf/wf0_meta.json` | 키 `spearman_recal_gain_vs_ml_gain` | `rho`, `p`, `n_targets` | 0.66, p 8.15e-05, 30대상 | 없음(`label`: '사후 서술(판정어 없음)'). `note`: 최선 ML 은 다섯 레시피 가운데 사후 선택이라 낙관적 | WF 계획서 2절 WF0 |
| A2 | P0 RMSE 와 최선 ML 이득의 순위상관 | 같은 파일 | 키 `spearman_p0_vs_ml_gain` | `rho`, `p` | 0.51, p 0.00362 | 없음(사후 서술) | WF 계획서 2절 WF0 |
| A3 | 라벨 10개 진단값과 규칙 W 이득(P0 − W, 라벨 전량)의 순위상관 | `results/rescale_wf/data/processed/wf/wf_tests.csv` | `test_id == 'WF4-c'` & `scope == 'MEAN'`(`contrast` 'Spearman(\|편향\|, P0 − W)') | `rho`, `ci_lo`, `ci_hi`, `n_targets`, `nboot`, `p_one` | 0.38 [0.17, 0.55], 28대상, 재표집 10000회, 한쪽 p 0.0002 | 지지. `scope == 'verdict'` 행의 `verdict` 열: '지지: 양의 순위상관(ρ 0.38, CI [0.17, 0.55])'. 같은 행 `role` 주, `confirmatory` False, `blind` True, `design_note` '결과 열람 뒤 설계' | WF 계획서 5.1 |
| A4 | 부호 있는 편향과 이득의 순위상관 | A3 와 같은 행 | 같음 | `rho_signed`, `ci_lo_signed`, `ci_hi_signed` | −0.01 [−0.17, 0.18] | 보조(판정어 없음) | WF 계획서 5.1 |
| A5 | 진단값과 고정 R1(λ 0.25) 이득(P0 − R1)의 순위상관 | A3 와 같은 행 | 같음 | `rho_gain_r1` | 0.47(CI 열 없음) | 보조 | WF 계획서 개정 이력 9 |
| A6 | P1 − P0, 라벨 10개, 주 4지역 평균(AB4) | `data/processed/lgw/lgw_bundle.csv` | `ab == 'AB4'` & `target == 'MEAN[Lena\|x,Canada\|x,Russia_W\|x,Russia_E\|x]'` | `delta`, `ci_lo`, `ci_hi`, `delta_blockeq`, `verdict4`, `holm_p` | −2.45 [−3.04, −1.88], 블록 등가중 −2.62, Holm p 0.001 | 우세. 초록 규칙 (a). S-a 계산에서 같은 양(추출 seed 만 다름)의 해석식 P1 − P0 분할 분포를 보았다(2026-09-30 03:25) | WRAPUP 결과 판정 기록(J7) 1.1. S-a 열람 표지: WRAPUP 754행(S-a 열람 기록), 814행(J7 등록 표지) |
| A7 | R1 − P1, 라벨 10개, 주 4지역 평균(AB5) | 같은 파일 | `ab == 'AB5'` & 같은 MEAN 행 | 같음 | −0.18 [−0.53, −0.01], 블록 등가중 −0.29, Holm p 0.136 | 우세(보정 전). 초록 규칙 (b) '차이를 확인하지 못했다' | J7 1.1 |
| A8 | R1 − P0, 라벨 전량, 주 4지역 평균(AB8) | 같은 파일 | `ab == 'AB8'` & 같은 MEAN 행 | 같음 | −2.64 [−3.48, −1.85], 블록 등가중 −2.64, Holm p 0.001 | 우세. 초록 규칙 (a) | J7 1.1 |
| A9 | R1 − P1, 라벨 전량, 주 4지역 평균(AB9) | 같은 파일 | `ab == 'AB9'` & 같은 MEAN 행 | 같음 | −0.40 [−0.76, −0.22], 블록 등가중 −0.51, Holm p 0.001 | 우세. 크기 0.5 cm 미만 | J7 1.1 |
| A10 | A8 − A9 = P1 − P0(라벨 전량, 주 4지역 평균) | A8, A9 의 `delta` | 같음 | `delta` 차 | −2.23(A8 의 85 %) | 없음(산술) | 이 README |
| A11 | AB8 지역 행(R1 − P0, 라벨 전량) | `data/processed/lgw/lgw_bundle.csv` | `ab == 'AB8'` & `target` ∈ {Lena\|x, Canada\|x, Russia_W\|x, Russia_E\|x} | `delta`, `ci_lo`, `ci_hi`, `verdict4` | 레나 −0.78 [−1.38, 0.08], 캐나다 +3.38 [1.10, 4.40], 러시아 W −13.98 [−16.05, −12.06], 러시아 E +0.84 [−0.43, 3.19] | 러시아 W 우세, 캐나다 열세, 레나·러시아 E 미결정(개선이 유의한 지역 1/4) | J7 1.1(AB8 행 '러시아 W −13.98 이 지배하고 캐나다는 +3.38 열세') |
| A12 | AB9 지역 행(R1 − P1, 라벨 전량) | 같은 파일 | `ab == 'AB9'` & 같은 지역 행 | 같음 | 레나 −0.80 [−1.51, −0.44], 캐나다 −1.26 [−1.57, −0.90], 러시아 W −0.48 [−1.05, 0.01], 러시아 E +0.93 [0.00(넷째 자리 0.0002), 1.28] | 레나·캐나다 우세, 러시아 W 미결정, 러시아 E 열세. 같은 대비의 LGX 보조 열(`data/processed/paper_figs/fig3_c.csv`, L4 행)은 러시아 E [−0.02, 1.28] 미결정이다(재표집 의존) | J7 1.1 |
| A13 | R1 − P0, 라벨 전량, 레나·캐나다·알래스카(x) 3지역 평균 | `data/processed/paper_figs/fig3_c.csv`(사본 `tables/paper_figs_fig3_c.csv`) | `block == 'R1 − P0'` & `kind == 'mean3'` | `delta`, `ci_lo`, `ci_hi`, `delta_beq`, `ci_lo_beq`, `ci_hi_beq`, `verdict4`, `ci_dependence` | +0.77 [0.02, 1.28], 블록 등가중 +1.15 [0.66, 1.63] | 열세. `ci_dependence` '분할 독립 가정 의존' | LG 계획서 7.1a(L8), 7.2 '원고 문장에 주는 결정' 3 |
| A14 | R1 − P*, 라벨 전량, 주 4지역 평균(P* = P0@tddm) | 같은 파일 | `block == 'R1 − P*'` & `kind == 'mean4'` | 같음 | −1.64 [−2.95, −0.74], 블록 등가중 −2.23 [−3.17, −1.24] | 우세 | LG 계획서 7.1a(P* 병기), J7 1.1 표 아래 문단 |

### 2.2 사례 지역의 분해

**러시아 서부(Russia_W, 라벨 |A| 14–16셀)**

| 번호 | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| RW1 | R1 − P0, 라벨 전량(L8 지역 행) | `results/rescale_lg/data/processed/lg/lg_tests.csv` | `test_id == 'L8'` & `target == 'Russia_W\|x'` & `n == -1` & `lam == 0.25` | `delta`, `ci_lo`, `ci_hi`, `delta_blockeq`, `ci_lo_beq`, `ci_hi_beq`, `sig` | −13.98 [−16.10, −11.94], 블록 등가중 −15.24 [−18.23, −11.56] | improve. L8 은 4지역 평균으로 지지 | LG 계획서 7.1 |
| RW2 | R1 − P1, 라벨 전량(L4 지역 행) | 같은 파일 | `test_id == 'L4'` & `target == 'Russia_W\|x'` & `n == -1` & `lam == 0.25` | 같음 | −0.48 [−1.11, 0.02], 블록 등가중 −0.24 [−0.79, 0.25] | ns. L4 는 4지역 평균으로 '희소 라벨에서도 ML 순가치 있음'. 이 판정은 주 4지역 한정이다. 확장 풀 PE1(주 4지역 + 러시아 C)의 L4e 는 '부분(지역 2/5), n ≤ 10 불성립'이고, 새 지역 평균 R1 − P1 은 n 10 +0.06 [−0.40, 0.51] 미결정이다(`data/processed/lgd/lgd_tests_lic.csv` L4e `verdict`·`compare` 행) | LG 계획서 7.1, 7.3 표와 '원고 문장' |
| RW3 | R1 − P1, 라벨 10개 | 같은 파일 | 같고 `n == 10` | 같음 | −0.56 [−1.24, 0.02], 블록 등가중 −0.24 [−0.85, 0.33] | ns. L4 의 n = 10 성립은 주 4지역 평균이고, 확장 풀 PE1 에서는 n ≤ 10 불성립이다(L4e, RW2 참조) | LG 계획서 7.1, 7.3 |
| RW4 | P1 − P0, 라벨 10개(AB4 지역 행) | `data/processed/lgw/lgw_bundle.csv` | `ab == 'AB4'` & `target == 'Russia_W\|x'` | `delta`, `ci_lo`, `ci_hi`, `delta_blockeq`, `verdict4` | −11.96 [−13.62, −10.53], 블록 등가중 −13.30 | 우세. S-a 계산에서 같은 양(추출 seed 만 다름)의 해석식 P1 − P0 분할 분포를 보았다(2026-09-30 03:25) | J7 1.1. S-a 열람 표지: WRAPUP 754행, 814행 |
| RW5 | 라벨 전량 P1 − P0, R1(λ 0.25) − P0, 최선 ML − P1 | `data/processed/wf/wf0_misspec.csv` | `target == 'Russia_W\|x'` | `P1`, `R1_0.25`, `ml_best_recipe`, `ml_vs_p1` | −13.50, −13.98, R1_1.0, −1.04 | 없음(사후 서술) | WF 계획서 2절 WF0 |
| RW6 | 같은 값, 확충 약관 확인분 | 같은 파일 | `target == 'Russia_W~exp~lic\|x'` | `P1`, `ml_best_recipe`, `ml_vs_p1` | −16.34, R2_1.0, −0.46 | 없음(사후 서술) | WF0 |
| RW7 | 진단값과 규칙 W 이득 | `results/rescale_wf/data/processed/wf/wf_tests.csv` | `test_id == 'WF4-c'` & `scope == 'region'` & `target == 'Russia_W\|x'` | `diag_abs`, `gain_w` | 38.88, 13.49 | 진단 행(판정어 없음) | WF 계획서 5.1 |
| RW8 | W − P1, 라벨 전량 | 같은 파일 | `test_id == 'WF4-b'` & `contrast == 'W-P1\|n전량'` & `target == 'Russia_W\|x'` | `delta`, `ci_lo`, `ci_hi`, `verdict4` | 0.01 [−2.48, 2.99] | 미결정 | WF 계획서 5.1(WF4-b 기각) |
| RW9 | 계수 비 | `results/rescale_lg/data/processed/lg/lg_targets.csv` | `target == 'Russia_W'` & `mode == 'x'` & `part == 'cpu'`(첫 행) | `E0`, `E_own`, `logE_ratio_own`; `n_A` 범위 | 1.59, 2.64, 0.51; 14–16 | 없음 | 해당 없음 |

읽기: 원천 계수 대비 오차 감소 13.98 cm(RW1, Δ = −13.98) 가운데 13.50 cm(RW5, 97 %)가 재보정 몫이고, 재보정 물리식 대비 ML 몫 0.48 cm(RW2, Δ = −0.48)는 구별되지 않았다.

**티베트(Tibet_LGD, 심부 레짐, 약관 확인분 판)**

| 번호 | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| TB1 | R1 − P1, 라벨 10개 | `data/processed/lgd/lgd_tests_lic.csv` | `test_id == 'L4e'` & `pool == 'PE2'` & `item == 'R1-P1'` & `target == 'Tibet_LGD\|x'` & `n == 10` | `delta`, `ci_lo`, `ci_hi`, `delta_blockeq`, `ci_lo_beq`, `ci_hi_beq`, `sig`, `cross_env` | −20.90 [−23.41, −17.94], 블록 등가중 −15.82 [−19.47, −12.40] | improve. `test_id == 'L38'` & `item == '(c) R1-P1'` 행 `verdict4` 우세. `cross_env` '교차 환경; (i) CatBoost 불통과'. L4e PE2 의 n ≤ 10 성립은 티베트에서 온다. 티베트를 뺀 PE1 은 '부분(지역 2/5), n ≤ 10 불성립'이다 | LG 계획서 7.3 |
| TB2 | P2 − P1, 라벨 10개 | 같은 파일 | 같고 `item == 'P2-P1'` | 같음 | −40.02 [−53.74, −24.18], 블록 등가중 −18.19 [−35.99, −0.51] | improve. L38 `(c) P2-P1` 우세 | LG 7.3 |
| TB3 | R1 − P2, 라벨 10개 | 같은 파일 | 같고 `item == 'R1-P2'` | 같음 | +19.11 [6.07, 30.17], 블록 등가중 +2.37 [−12.06, 17.00] | ns. L38 `(c) R1-P2` 미결정 | LG 7.3 |
| TB4 | P1 − P0, 라벨 10개 | 같은 파일 | `test_id == 'L38'` & `item == '(c) P1-P0'` & `target == 'Tibet_LGD\|x'` | `delta`, `ci_lo`, `ci_hi`, `delta_blockeq`, `verdict4` | −96.37 [−100.70, −91.92], 블록 등가중 −97.75 | 우세 | LG 7.3 |
| TB5 | R1 − P1, R1 − P2, 라벨 전량 | 같은 파일 | `test_id == 'L4e'` & `pool == 'PE2'` & `target == 'Tibet_LGD\|x'` & `n == -1`, `item` 각각 | `delta`, `ci_lo`, `ci_hi`, `delta_blockeq`, `sig` | R1 − P1 −9.08 [−10.88, −6.86](블록 등가중 −6.51); R1 − P2 −5.89 [−9.67, −2.30](블록 등가중 −7.63) | 둘 다 improve | LG 7.3 |
| TB6 | R1 − P0, 라벨 전량 | 같은 파일 | `test_id == 'L8e'` & `pool == 'PE2'` & `target == 'Tibet_LGD\|x'` | `delta`, `ci_lo`, `ci_hi`, `delta_blockeq`, `rmse_A`, `rmse_B` | −144.73 [−157.63, −130.23], 블록 등가중 −125.70; R1 99.78, P0 244.51 | improve. L8e(PE2) 지지 | LG 7.3 |
| TB7 | 라벨 정의를 지온 유도로 바꾼 R1 − P1, 라벨 10개 | 같은 파일 | `test_id == 'L39'` & `item == 'R1-P1\|n10'` & `target == 'Tibet_LGD~L39\|x'` | `delta`, `ci_lo`, `ci_hi`, `verdict4` | −20.92 [−22.59, −18.23] | 우세. L39 판정 강건 | LG 7.3 |
| TB8 | 라벨 전량 P1 − P0, R1(λ 1.0) − P0, 최선 ML − P1 | `data/processed/wf/wf0_misspec.csv` | `target == 'Tibet_LGD\|x'` | `P0`, `P1`, `R1_1.0`, `ml_vs_p1` | 244.51, −135.65, −159.55, −23.90 | 없음(사후 서술) | WF0 |
| TB9 | 진단값, 규칙 W 이득, W − P1(라벨 전량) | `results/rescale_wf/data/processed/wf/wf_tests.csv` | `WF4-c` 지역 행; `WF4-b` & `contrast == 'W-P1\|n전량'`; `target == 'Tibet_LGD\|x'` | `diag_abs`, `gain_w`; `delta`, `ci_lo`, `ci_hi`, `verdict4` | 227.71, 159.55; −23.90 [−30.86, −15.01] | W − P1 우세 | WF 계획서 5.1 |

읽기: 라벨 10개의 R1 − P1 −20.90 cm(TB1)는 κ = 10 수축 재보정이 남긴 편향을 잔차가 메운 값이다. 같은 라벨로 현지 최소제곱 재보정(P2)을 하면 P1 보다 40.02 cm 낫고(TB2), R1 은 P2 와 구별되지 않았다(TB3, 셀 가중은 R1 이 나쁨). 라벨 전량에서는 R1 이 P2 보다도 5.89 cm 낫다(TB5). 원천 계수 대비 오차 감소 144.73 cm(TB6, Δ = −144.73) 가운데 135.65 cm(TB8, 94 %)가 재보정 몫이다.

**캐나다(Canada, 지역 계수는 맞고 지역 안 이질성이 큰 대상)**

| 번호 | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| CA1 | P1 − P0, 라벨 10개(AB4 지역 행) | `data/processed/lgw/lgw_bundle.csv` | `ab == 'AB4'` & `target == 'Canada\|x'` | `delta`, `ci_lo`, `ci_hi`, `verdict4` | +2.26 [0.71, 2.94] | 열세. S-a 계산에서 같은 양(추출 seed 만 다름)의 해석식 P1 − P0 분할 분포를 보았다(2026-09-30 03:25) | J7 1.1. S-a 열람 표지: WRAPUP 754행, 814행 |
| CA2 | R1 − P0, 라벨 전량(AB8 지역 행) | 같은 파일 | `ab == 'AB8'` & `target == 'Canada\|x'` | 같음 | +3.38 [1.10, 4.40] | 열세 | J7 1.1 |
| CA3 | R1 − P1, 라벨 전량(AB9 지역 행) | 같은 파일 | `ab == 'AB9'` & `target == 'Canada\|x'` | 같음 | −1.26 [−1.57, −0.90] | 우세 | J7 1.1 |
| CA4 | 라벨 전량 P1 − P0, 최선 ML − P0, 최선 ML − P1 | `data/processed/wf/wf0_misspec.csv` | `target == 'Canada\|x'` | `P1`, `ml_best_delta`, `ml_best_recipe`, `ml_vs_p1` | +4.64, −1.92, D0_1.0, −6.56 | 없음(사후 서술) | WF0 |
| CA5 | W − P1, 라벨 전량 | `results/rescale_wf/data/processed/wf/wf_tests.csv` | `test_id == 'WF4-b'` & `contrast == 'W-P1\|n전량'` & `target == 'Canada\|x'` | `delta`, `ci_lo`, `ci_hi`, `delta_blockeq`, `verdict4` | −4.32 [−5.35, −2.67], 블록 등가중 −3.67 | 우세 | WF 5.1 |
| CA6 | 계수 비와 블록 간 E 변동 | `results/rescale_lg/data/processed/lg/lg_targets.csv` | `target == 'Canada'` & `mode == 'x'` & `part == 'cpu'`(첫 행) | `E0`, `E_own`, `logE_ratio_own`, `E_block_cv` | 1.59, 1.60, 0.00(셋째 자리 0.003), 0.22 | 없음 | 해당 없음 |
| CA7 | 지역 내 Stefan·CCI 앵커 + 잔차(Re, 교차검증 λ) − P1, 분할 5회(WF1) | `results/rescale_wf/data/processed/wf/wf_curve.csv` | `exp == 'wf1'` & `target == 'Canada'` & `mode == 'r'` & `method == 'Re'` & `lam == -1` | `d_p1`, `d_p1_beq`, `verdict4_p1`, `n_splits_valid` | n 20·50·100·200·전량 순으로 셀 가중 −2.87, −3.83, −4.55, −6.08, −6.61, 블록 등가중 +2.05, +1.41, −0.10, −0.75, −1.31, 5분할. 두 가중의 부호가 다른 것은 n 20·50 뿐이다 | 5행 모두 미결정 | WF 5.1(WF1-a 서술 대상) |
| CA8 | 같은 대비, 분할 25회(WF6-a) | `results/rescale_wf2/data/processed/wf/wf2b_tests.csv` | `test_id == 'WF6-a'` & `target == 'Canada\|r'` & `contrast` ∈ {'Re(λ cv)-P1\|n200', 'Re(λ cv)-P1\|n전량'} | `delta`, `ci_lo`, `ci_hi`, `delta_blockeq`, `ci_lo_beq`, `ci_hi_beq`, `verdict4` | n 200: −2.13 [−2.72, −0.23], 블록 등가중 +2.17 [1.29, 3.06]. 전량: −1.73 [−2.45, 0.09], 블록 등가중 +1.86 [0.92, 2.83] | 미결정. WF6-a 캐나다 기각 | WF 계획서 5.2 |

읽기: 캐나다의 지역 평균 계수는 원천과 거의 같다(CA6). 그런데 라벨 위치에서 맞춘 P1 이 원천 계수 P0 보다 나쁘다(CA1, CA4). 고정 잔차(CA3, R1 − P1 −1.26)와 규칙 W(CA5, W − P1 −4.32)의 재보정 대비 감소는 계수 오차가 아니라 나빠진 P1 을 되돌리는 값이다. 되돌린 뒤에도 R1 은 P0 보다 3.38 cm 나쁘고(CA2, 열세), W 는 P0 보다 0.32 cm 나쁘다(`results/rescale_wf/data/processed/wf/wf_tests.csv` WF4-c `Canada|x` 의 `gain_w`(P0 − W) −0.32, 점 추정). CA4 는 다르다. 사후 선택한 최선 ML(D0_1.0, 직접 ML)은 P1 대비 6.56 cm 를 줄여 P1 의 손해(4.64 cm)를 넘었고, P0 보다 1.92 cm 낫다(`wf0_misspec.csv` 의 `ml_best_delta` −1.92, `sig_D0_1.0` improve). 이 대상에서는 직접 ML 이 원천 계수 물리식보다 낫다. 다만 다섯 레시피 가운데 사후 선택이라 낙관적이다(`wf0_meta.json` 의 `note`). 현행 C2 의 '−2.9 에서 −6.6 cm' 는 CA7 의 셀 가중 점 추정이다. 분할 5회(CA7)에서 블록 등가중 값의 부호가 다른 것은 n 20·50(+2.05, +1.41)이고, n 100·200·전량은 두 가중 모두 음수다(블록 등가중 −0.10, −0.75, −1.31). 분할 25회(CA8)에서는 n 200·전량 모두 블록 등가중 값의 부호가 다르고(+2.17, +1.86) 미결정이다.

### 2.3 이번 점검(사후, 2026-10-04): 재보정 몫과 ML 몫의 분리

**계산**: `posthoc_c2_split.py`(이 폴더). 입력은 `tables/` 의 사본만 쓴다. 산출은 `tables/derived_c2_wf0_split.csv`(대상별, 30행), `tables/derived_c2_wf4c_split.csv`(대상별, 28행), `tables/derived_c2_posthoc_rho.csv`(아래 표). 대상 단위 Spearman 과 해석적 양측 p 만 계산했다. 블록 재표집 CI 는 없다. 대상은 서로 독립이 아니다(5.1 의 3번). 판정이 아니다.

**분해의 성립 확인**: WF4-c 의 `gain_w`(P0 − W)에 WF4-b 의 W − P1 을 더한 재보정 몫과, `gain_r1`(P0 − R1)에 R1(λ 0.25) − P1 을 더한 값의 최대 차는 2.8e-14 cm 다. 이 재보정 몫은 `wf0_misspec.csv` 의 `recal_gain`(LG·LGD 곡선에서 계산)과 두 표에 함께 있는 27대상에서 최대 차 2.8e-14 cm 로 같다(WF0 의 `Russia_C|x` 와 WF4 의 `Russia_C~lgd|x` 를 같은 대상으로 맞췄다).

| 묶음 | x | y | 대상 수 | ρ | 양측 p | 비고 |
|---|---|---|---|---|---|---|
| WF0 | 재보정 이득 P0 − P1 | 최선 ML 이득 P0 − 최선 ML | 30 | 0.66 | 8.15e-05 | A1 재현 |
| WF0 | 재보정 이득 P0 − P1 | ML 몫 P1 − 최선 ML | 30 | −0.08 | 0.666 | QA Q1 C2 점검 −0.08 재현 |
| WF0 | \|P0 − P1\| | ML 몫 P1 − 최선 ML | 30 | 0.16 | 0.406 | 계수 오차를 크기로 정의 |
| WF0 | P0 RMSE | ML 몫 P1 − 최선 ML | 30 | 0.57 | 0.000917 | 총 오차 수준(계수 오차가 아니다) |
| WF0, LG 대상 | \|log(E_own/E0)\| | 재보정 이득 P0 − P1 | 25 | 0.76 | 9.17e-06 | 직접 계수 오차(`lg_targets.csv`) |
| WF0, LG 대상 | \|log(E_own/E0)\| | ML 몫 P1 − 최선 ML | 25 | 0.03 | 0.904 | 직접 계수 오차 |
| WF0, LG 대상 | 블록 간 E 변동 계수(`E_block_cv`) | ML 몫 P1 − 최선 ML | 25 | 0.16 | 0.443 | 지역 안 이질성 |
| WF4-c | 진단값 | 규칙 W 이득 P0 − W | 28 | 0.38 | 0.0455 | A3 의 점 추정 재현 |
| WF4-c | 진단값 | 재보정 몫 P0 − P1 | 28 | 0.37 | 0.0534 | |
| WF4-c | 진단값 | ML 몫 P1 − W | 28 | 0.18 | 0.358 | |
| WF4-c | 진단값 | 고정 R1 이득 P0 − R1(λ 0.25) | 28 | 0.47 | 0.0123 | A5 재현 |
| WF4-c | 진단값 | ML 몫 P1 − R1(λ 0.25) | 28 | 0.39 | 0.0398 | |
| WF4-c, 티베트 제외 | 진단값 | 규칙 W 이득 P0 − W | 27 | 0.31 | 0.116 | |
| WF4-c, 티베트 제외 | 진단값 | 재보정 몫 P0 − P1 | 27 | 0.30 | 0.134 | |
| WF4-c, 티베트 제외 | 진단값 | ML 몫 P1 − W | 27 | 0.09 | 0.669 | |
| WF4-c, 티베트 제외 | 진단값 | 고정 R1 이득 P0 − R1(λ 0.25) | 27 | 0.41 | 0.0359 | |
| WF4-c, 티베트 제외 | 진단값 | ML 몫 P1 − R1(λ 0.25) | 27 | 0.32 | 0.103 | |

**정리**
1. \|log(E_own/E0)\|(LG 25대상)는 재보정 몫(P0 − P1)과 ρ 0.76(p 9.17e-06)이다. ML 몫(P1 − 최선 ML)과는 계수 오차의 정의별로 −0.08(P0 − P1, p 0.666), 0.16(\|P0 − P1\|, p 0.406), 0.03(\|log(E_own/E0)\|, p 0.904)이고 모두 p > 0.4 다. P0 − P1 은 재보정 몫과 같은 양이므로 둘의 상관은 계산하지 않았다.
2. 라벨 10개 진단값과의 순위상관은 재보정 몫 0.37(p 0.0534), 규칙 W 의 ML 몫 0.18(p 0.358)이다. 둘 다 점 추정이고 두 값의 차이는 검정하지 않았다. 고정 R1(λ 0.25)의 ML 몫과는 0.39(p 0.0398)이고, 티베트를 빼면 0.32(p 0.103)다.
3. ML 몫은 계수 오차보다 P0 의 총 오차 수준과 함께 움직인다(0.57, p 0.000917). 티베트·캐나다·CA-3·러시아 C 처럼 P0 오차가 크거나 이질적인 대상에서 ML 몫이 크다(`derived_c2_wf0_split.csv` 의 `ml_beyond_p1`).
4. 따라서 'ML 몫은 계수 오차와 상관이 없었다' 는 정의에 따라 성립하지 않을 수 있다. 1.3 은 '확인하지 못했다' 로 쓴다. 등록 분석(XA, 6절)이 이 표를 대신한다.

**재보정 몫의 비율**(점 추정 산술, 같은 채점 셀과 가중이라 대비가 더해진다. 모든 행이 고정 R1(λ 0.25) 기준이고 규칙 W 의 값이 아니다. 열의 값은 Δ 이고 음수가 개선이다)

| 대상 | R1 − P0(라벨 전량) | P1 − P0 | R1 − P1 | 재보정 몫 비율 | 원천 |
|---|---|---|---|---|---|
| 주 4지역 평균(러시아 W 지배) | −2.64(A8) | −2.23(A10) | −0.40(A9) | 85 % | `lgw_bundle.csv` |
| 러시아 서부 | −13.98(RW1) | −13.50(RW5) | −0.48(RW2) | 97 % | `lg_tests.csv`, `wf0_misspec.csv` |
| 티베트 | −144.73(TB6) | −135.65(TB8) | −9.08(TB5) | 94 % | `lgd_tests_lic.csv`, `wf0_misspec.csv` |
| 레나 | −0.78(A11, 미결정) | +0.02 | −0.80(A12, 우세) | 해당 없음(감소분은 모두 ML 몫) | `lgw_bundle.csv`, `wf0_misspec.csv`(`Lena\|x` 의 `P1` 0.0228) |
| 러시아 동부 | +0.84(A11, 미결정) | −0.09 | +0.93(A12, 열세) | 해당 없음(R1 이 P1 보다 나쁨) | `lgw_bundle.csv`, `wf0_misspec.csv`(`Russia_E\|x` 의 `P1` −0.089) |
| 캐나다 | +3.38(CA2) | +4.64(CA4) | −1.26(CA3) | 해당 없음(재보정이 오차를 키움) | `lgw_bundle.csv`, `wf0_misspec.csv` |

'감소의 대부분이 재보정 몫'은 위 표의 앞 세 행(주 4지역 평균, 러시아 서부, 티베트)에서만 성립한다. 지역별로는 성립하지 않는다(레나, 러시아 동부, 캐나다).

---

## 3. 근거 실험

| 새 이름 | 옛 id | 이 주장에 쓰인 것 | 원천 표 | 판정 기록 |
|---|---|---|---|---|
| W1_label0_risk_table | WF0 | 재보정 이득과 최선 ML 이득의 ρ 0.66, 대상별 P1·ML Δ(라벨 전량) | `data/processed/wf/wf0_misspec.csv`, `wf0_meta.json`(스크립트 `scripts/2_evaluation/wf0_reanalysis.py`) | WF 계획서 2절 WF0(사후 서술) |
| W5_method_selection_and_bias_diagnosis | WF4(WF4-c 주, WF4-b 대상별 W − P1) | ρ 0.38 [0.17, 0.55], 대상별 진단값과 이득 | `results/rescale_wf/data/processed/wf/wf_tests.csv` | WF 계획서 5.1 |
| T1_transfer_label_grid | LG(L4, L8) | 러시아 서부 R1 − P0·R1 − P1, 대상별 E0·E_own·`logE_ratio_own`, L8 의 3지역 보조 열과 L29 P* 병기(A13, A14, `data/processed/paper_figs/fig3_c.csv`, LG 7.1a) | `results/rescale_lg/data/processed/lg/lg_tests.csv`, `lg_targets.csv` | LG 계획서 7.1, 7.1a |
| T6_abstract_contrasts_and_map | LGW(WRAPUP AB4, AB5, AB8, AB9) | 재보정 몫과 ML 몫의 주 4지역 평균, 지역 행 | `data/processed/lgw/lgw_bundle.csv` | WRAPUP 결과 판정 기록(J7) 1.1 |
| T4_new_regions | LGD(L4e, L8e, L38, L39) | 티베트 R1 − P1, P2 − P1, R1 − P2, 라벨 정의 강건성 | `data/processed/lgd/lgd_tests_lic.csv` | LG 계획서 7.3 |
| W2_within_region_label_curve | WF1 + WF6 | 캐나다 지역 내 Re − P1 | `results/rescale_wf/data/processed/wf/wf_curve.csv`, `results/rescale_wf2/data/processed/wf/wf2b_tests.csv` | WF 계획서 5.1, 5.2 |
| H4_label_budget_H25-30, H5_final_A2-D1 | h25b, a2 | L4 의 '비맹검 부분 포함' 표지의 출처(앞선 실험에서 방향을 보았다) | 해당 없음 | LG 계획서 7.1 맹검 열 |

계산 환경: WF4·WF6 은 Rescale elm 한 플랫폼에서 대비가 닫힌다(WF 계획서 5.1·5.2). LG 는 Rescale T4 조각만 쓴다(LG 7.1). LGD 의 새 지역 조각은 로컬이고 P4 는 Rescale 이라 PE1·PE2 행에 교차 환경 표지가 있다(LG 7.3). 모두 CatBoost 와 해석식이다.

---

## 4. 그림

### 4.1 현행 v2 그림에서 이 주장에 쓰는 패널

패널 정의는 `figures/figure_spec.json` 의 `display_items` 에서 옮겼다.

| 그림·패널 | 파일 | C2 에서의 쓰임 | 원천 표 |
|---|---|---|---|
| Fig 3c | `outputs/figures/paper/v2/Fig3_physics_use.{pdf,svg,png}` | R1 − P0(L8, AB8)와 R1 − P1(L4, AB9)의 지역 행과 주 4지역 평균. 러시아 서부 행 −13.98 대 −0.48 이 재보정 몫을 보인다 | `data/processed/paper_figs/fig3_c.csv`(사본 `tables/paper_figs_fig3_c.csv`) |
| Fig 3d | 같은 파일 | P1·P2·P3·V1 의 P0 대비(계수 부분 풀링), AB4 표지. 재보정 몫의 라벨 수 곡선 | `data/processed/paper_figs/fig3_d.csv` |
| Fig 4b | `outputs/figures/paper/v2/Fig4_min_labels.{pdf,svg,png}` | L4 R1 − P1 층화 평균 곡선. ML 몫의 라벨 수 곡선 | `data/processed/paper_figs/fig4_b.csv` |
| Fig 4d | 같은 파일 | n 마다 R1 − P0, P1 − P0, R1 − P1 의 대상별 4분 판정 수 | `data/processed/paper_figs/fig4_d.csv` |
| Fig 2c | `outputs/figures/paper/v2/Fig2_label_curve.{pdf,svg,png}` | 러시아 서부 라벨 수 곡선(P1, R0, R1, R2, D0) | `data/processed/paper_figs/fig2_region_curves.csv` |
| Fig 1b | `outputs/figures/paper/v2/Fig1_problem.{pdf,svg,png}` | 지역별 z = ln ALT − ln √TDD 분포. 지역마다 E 가 다르다는 배경 | `data/processed/paper_figs/fig1_z_points.csv`, `fig1_z_summary.csv` |

현행 v2 에는 WF0·WF4-c 값과 티베트 대비가 없다. `outputs/figures/paper/_qa/v2_*_values.txt` 에서 'tibet' 은 Table 1 값 파일에만 나온다(2026-10-04 검색).

### 4.2 재구성안의 계획

`docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 4절은 C2 를 결과 R3 절(라벨 1–10: 재보정과 잔차, 그리고 진단)에 두고 근거를 AB4, AB5, L10·AB7, WF0 ρ 0.66, WF4-c ρ 0.38, SC1w 로 적는다(R3 행은 C2 와 C3 를 함께 다루며, L10·AB7 은 C3 의 근거다). 5절은 Fig 4 를 'ML 이 언제 이득인가' 로 바꾸고 계수 오차와 ML 이득 산점도(WF0 ρ 0.66, WF4-c 진단 ρ 0.38)를 더하는 안이다(변경 정도 중간). 그림 배치의 확정은 사용자 결정 사항이다(같은 문서 8절).

### 4.3 축소한 C2 에 맞춘 패널 제안(미작성)

재구성안의 산점도 하나는 재보정 몫과 ML 몫을 섞는다. 두 패널로 나눈다.
- (가) x 진단값(cm, 로그 축), y 재보정 몫 P0 − P1(cm, 라벨 전량, symlog 축). 원천 `tables/derived_c2_wf4c_split.csv` 의 `diag_abs`, `recal_share`.
- (나) 같은 x, y ML 몫 P1 − W(cm). 같은 파일의 `ml_beyond_W`. 보조로 `ml_beyond_R1` 을 열린 표지로 겹친다.
- 표지: 상위 지역 7곳(알래스카, 캐나다, 레나, 러시아 W·E·C, 티베트)은 채운 표지, 하위 지역은 열린 표지로 구분하고 i·x 모드는 모양으로 나눈다. 색만으로 구분하지 않는다. 북대서양(`NAtlantic~lic|x`)은 WF 계획서 개정 이력 10 이 WF4 의 점 추정 대상으로 등록했으므로 패널에 넣는 것은 맞다. 다만 '점 추정·SI 대상' 표지를 따로 둔다(`ci_flag` '점 추정(분포 없음)'). LG 계획서 7.3 '원고 문장'은 독립 지역 검증 문장의 근거를 약관 확인분 판(새 얕은 지역 Russia_C 1곳)으로 하고 NAtlantic 은 SI 의 전체 판에만 두라고 정하므로, 북대서양을 채운 독립 지역 표지로 그리지 않는다.
- 캡션의 지역 수: 위 8곳(북대서양 포함)은 WF4-c 대상 구성(`wf_tests.csv` WF4-c 지역 행)의 상위 지역 기준이다. QA 의 '확인적 독립 지역 4–5개'(`docs/QA_FINAL_REVIEW_2026-10-02.md` Q1 '좁혀야 할 점' 3)와 정의가 다르므로 캡션에 기준을 적는다.
- 등록 ρ 0.38 [0.17, 0.55]는 P0 − W 에만 붙인다. 분해 패널의 ρ 는 XA 의 CI 가 나오기 전까지 '사후' 로 표시하거나 싣지 않는다.
- 대안: 독립 지역별 막대(R1 − P0 를 재보정 몫과 ML 몫으로 쌓기, 2.3 의 비율 표). 티베트는 cm 척도에서 다른 지역을 가리므로 별도 축이나 삽도로 둔다.

---

## 5. 단서와 쓰지 않을 문장

### 5.1 단서

1. **사후 설계**: WF0 은 사후 서술이다(`wf0_meta.json` 의 `label`). WF4-c 는 실행 전에 등록했으나 LG 결과를 본 뒤 설계했다(`wf_tests.csv` 의 `design_note` '결과 열람 뒤 설계', WF 계획서 5.1 의 표지). 같은 파일의 판정 행은 `role` 주, `confirmatory` False 다.
2. **사후 선택**: WF0 의 최선 ML 은 다섯 레시피 가운데 사후 선택이라 낙관적이다(`wf0_meta.json` 의 `note`).
3. **대상의 비독립**: WF4-c 의 28대상은 상위 지역 8곳(WF4-c 대상 구성 기준, 북대서양은 점 추정 대상)과 그 하위 지역(AL-1–AL-6, CA-2·CA-3, LE-1·LE-2), 원천 모드 i·x 중복으로 이루어진다(`wf_tests.csv` WF4-c 지역 행의 `target`). CI 는 대상 집합을 고정한 채 이득의 블록 재표집과 진단값의 (분할, 추출) 재표집을 묶은 것이다(WF 계획서 개정 이력 9). 대상 사이의 중첩은 반영하지 않는다. WF0 의 30대상(A1 의 ρ 0.66, 1.2 의 ρ −0.08, 2.3 의 WF0 행의 모집단)에는 이와 별도로 라벨 확충판 `Canada~exp~lic|x`, `Russia_E~exp|x`, `Russia_W~exp~lic|x` 가 `Canada|x`, `Russia_E|x`, `Russia_W|x` 와 함께 들어 있다(`wf0_misspec.csv` 의 `target`). 같은 지역이 두 번 들어간 중복 쌍이다. WF0 과 WF4-c 의 대상 집합은 27개만 겹친다(WF0 의 `Russia_C|x` 와 WF4-c 의 `Russia_C~lgd|x` 를 같은 대상으로 셀 때. 이름이 같은 대상은 26개). WF0 에만 있는 대상은 확충판 3개, WF4-c 에만 있는 대상은 `NAtlantic~lic|x` 다.
4. **티베트의 비중**: 티베트의 진단값은 227.71 cm 로 다른 대상(2.94–38.88 cm)의 최댓값보다 약 6배 크다(`wf_tests.csv` `diag_abs`). 순위상관이라 크기의 영향은 제한되지만, 티베트를 빼면 규칙 W 이득과의 사후 ρ 는 0.31(p 0.116)이다(2.3). 새 지역 풀의 cm 단위 평균도 티베트가 지배한다(L8e PE2 −26.78, `lgd_tests_lic.csv`).
5. **티베트 자료의 한계**(`docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B.9 서술): 대상 라벨은 GPR 과 인력 시추이고 계절 말 근거가 자료 설명(`dataset_statement`)뿐이다. 대상 셀의 고도는 전부 원천 셀 고도의 99 백분위(901.8 m) 위다. 티베트 대비는 비맹검이다(6B.7). LGD 의 PE1·PE2 행에는 교차 환경 표지가 있다(7.3).
6. **티베트 R1 − P1 의 기준**: −20.90 cm 는 κ = 10 수축 재보정 대비다. 현지 최소제곱 재보정 P2 대비로는 라벨 10개에서 구별되지 않았다(TB3).
7. **캐나다**: 재보정 물리식이 원천 계수보다 나쁜 대상이라 P1 대비 이득은 계수 오차가 아니라 라벨 위치 이질성을 반영한다(QA Q1 해석, 2.2 CA1–CA6). 예외로 사후 선택한 직접 ML(CA4, D0_1.0)은 P0 보다도 1.92 cm 낫다(사후 선택, 낙관적). 지역 내 Re − P1 의 셀 가중과 블록 등가중 부호는 분할 5회(CA7)에서 n 20·50 만 다르고(n 100 이상은 두 가중 모두 음수), 분할 25회(CA8)에서는 n 200·전량 모두 다르다.
8. **크기**: 주 4지역 평균 R1 − P1(A9, 라벨 전량)은 4분 판정 우세이고 `small_note` 는 '통계적으로 구별되나 크기는 0.5 cm 미만'이다(셀 가중 점 추정 −0.40 cm 기준). 셀 가중 CI [−0.76, −0.22]의 하한과 블록 등가중 점 추정 −0.51 은 절댓값이 0.5 cm 를 넘는다. 따라서 '동등성 한계 0.5 cm 안' 으로 쓰지 않는다. 라벨 10개 R1 − P1(A7, −0.18 cm)은 Holm 보정 뒤 초록 규칙 (b)에 해당한다.
9. **초록**: 초록의 판정어는 AB1–AB10 에만 붙인다(WRAPUP 1.1). WF4-c 를 초록에 판정어로 쓰려면 WRAPUP 1.1 의 개정이 필요하다(재구성안 7절, 사용자 결정).
10. **비례**: Spearman 순위상관은 비례를 뜻하지 않는다.

### 5.2 쓰지 않을 문장

| 쓰지 않을 문장 | 이유 | 대신 쓸 문장 |
|---|---|---|
| ML 의 이득은 물리 계수가 틀린 정도에 비례한다 | 고정 R1(λ 0.25) 기준으로 원천 계수 대비 감소의 85–97 % 가 재보정 몫인 것은 주 4지역 평균(러시아 서부 지배), 러시아 서부, 티베트에서다. 지역별로는 다르다. 레나는 감소분 0.78 cm 가 모두 ML 몫이고(P1 − P0 +0.02, R1 − P1 −0.80 우세), 러시아 동부는 R1 − P1 +0.93(열세)이다(2.3 비율 표, A11, A12). 순위상관은 비례가 아니다 | 라벨을 쓰는 보정의 원천 계수 대비 이득은 라벨 10개 편향 크기와 양의 순위상관을 보였다 |
| 재보정을 넘어선 ML 의 추가 이득은 계수 오차와 상관이 없었다 | 정의에 따라 사후 ρ −0.08 에서 0.39 사이, CI 없음(2.3) | 재보정을 넘어선 ML 몫과 계수 오차의 관계는 확인하지 못했다 |
| 라벨 10개로 ML 을 더할 가치를 진단할 수 있다 | 진단값이 예측하는 것은 재보정과 잔차를 합친 이득이고, 그 대부분이 재보정이다 | 라벨 10개로 그 지역에서 라벨과 계수 보정이 줄 이득의 크기를 가늠할 수 있다 |
| 러시아 서부에서 ML 이 오차를 14 cm 줄였다 | 원천 계수 대비 감소 13.98 cm 가운데 13.50 cm 가 재보정 몫이고 R1 − P1 은 ns 다(RW1, RW2, RW5) | 러시아 서부에서 원천 계수 대비 오차 감소 13.98 cm 가운데 13.50 cm 는 계수 재보정에서 왔다 |
| 티베트에서 ML 이 재보정 물리식보다 21 cm 낫다 | 수축 재보정(P1) 대비이고, 현지 최소제곱 재보정(P2)과는 라벨 10개에서 구별되지 않았다(TB1–TB3) | 티베트에서 잔차 ML 은 라벨 10개에서 수축 재보정(P1)보다 오차가 20.90 cm 작았으나, 같은 라벨의 현지 최소제곱 재보정(P2)과는 구별되지 않았다(R1 − P2 +19.11 [6.07, 30.17], 셀 가중은 열세, 블록 등가중 +2.37 [−12.06, 17.00], L38 4분 판정 미결정). 라벨 전량에서는 P2 보다 오차가 5.89 cm 작았다 |
| 캐나다에서 Stefan·CCI 앵커 + 잔차가 재보정 물리식보다 2.9–6.6 cm 낫다 | 분할 5회의 셀 가중 점 추정이다. 블록 등가중 값은 분할 5회에서 n 20·50 의 부호가 다르고(n 100·200·전량은 −0.10, −0.75, −1.31), 분할 25회에서는 n 200·전량 모두 부호가 다르며 미결정이다(CA7, CA8) | 캐나다 지역 내에서 Stefan·CCI 앵커 + 잔차와 재보정 물리식의 차이는 확인하지 못했다 |
| 편향 진단으로 ML 사용 여부를 안전하게 정할 수 있다 | 비열등 판정이 없다. 소수 라벨 재보정의 안전성은 미확인이다(J7 SC1w) | 편향 진단은 라벨 투자의 우선순위를 정하는 참고 지표다 |
| 사전 등록 확인 가설에서 진단의 예측력이 입증되었다 | WF4-c 는 결과 열람 뒤 설계이고 파일의 `confirmatory` 는 False 다 | 결과 열람 뒤 등록한 진단 가설(WF4-c)이 지지되었다 |
| 편향의 방향을 보면 이득을 알 수 있다 | 부호 있는 편향의 ρ 는 −0.01 이다(A4) | 편향의 크기가 이득의 순위와 관련되고 방향은 관련되지 않았다 |

---

## 6. 이 주장을 바꿀 수 있는 새 실험(X*)

X* 이름과 계획은 `docs/research/2026-10-04/harness_implementation_plan.md` 4절에 있다(구현 계획서. 파일 시각 2026-10-04 13:12 로, 이 README 초판 13:03 뒤에 생겼다. XA 는 4.1 `XA_c2_gain_decomposition`). 사전 등록 문서는 아직 없다(`paper/registry/experiments.csv` XA 행의 `verdict_doc` '[예정] 등록 문서 미정', 파일 시각 13:04)[미확인: 이후 작성 여부]. 아래 '예상 영향' 은 실행 전 판단이다.

**XA 의 열람 상태(비맹검 기록의 정합)**: 구현 계획서의 열람 상태(머리 'XA' 항목과 4.1 '위험')에는 ρ −0.08 과 0.07 만 적혀 있다. 0.07 은 `data/processed/wf/wf0_misspec.csv` 에서 ρ(P0 − P1, P1 − R1(λ 0.25)) = 0.0696(p 0.715, 30대상)으로 다시 확인했다. 이 README 2.3 에서 이미 계산한 사후 값도 열람한 값이다. WF0 의 0.16(\|P0 − P1\| 대 ML 몫), 0.57(P0 RMSE 대 ML 몫), LG 25대상의 0.76, 0.03, 0.16, WF4-c 28대상의 0.38, 0.37, 0.18, 0.47, 0.39, 티베트 제외 판(27대상)의 0.31, 0.30, 0.09, 0.41, 0.32 다. XA 등록 문서의 열람 상태에는 이 값들을 더해 적어야 한다(구현 계획서와 이 README 의 비맹검 기록을 맞춘다).

| 새 이름 | 내용(이 주장과 관련된 부분) | 예상 영향 |
|---|---|---|
| XA_c2_gain_decomposition | WF4-c 와 같은 재표집 번호로 ρ(진단값, P0 − P1), ρ(진단값, P1 − W), ρ(진단값, P1 − R1) 의 블록 재표집 CI 를 계산한다. 새 적합은 없다(QA Q6 추가 실험 2번 '기존 곡선 재분석, 새 적합 없음', Q9 보강안 표 '계산 수 분'). 판정 규칙은 계산 전에 고정하고, 2.3 의 사후 값을 이미 보았으므로 '비맹검' 표지를 단다 | 1.3 의 마지막 문장을 '관계 없음' 또는 '양의 관계' 로 바꿀 수 있다. 고정 R1 몫의 사후 ρ 0.39 가 CI 로 0 을 제외하면 '재보정을 넘어선 ML 몫도 편향 크기와 관련된다' 로 고친다 |
| XB_multisource_stacking | 여러 물리식·제품을 대상 라벨로 가중하는 적층(QA Q11 X1). 재보정 기준이 P1 에서 적층 물리식으로 바뀐다 | '재보정 몫' 의 정의가 넓어진다. 캐나다처럼 P1 이 나쁜 대상에서 적층이 이득을 흡수하면 ML 몫은 더 줄어든다 |
| XF_new_regions | 독립 지역 추가 | WF4-c 대상 구성의 상위 지역 8곳(북대서양은 점 추정 대상, 4.3)이 늘어난다. 티베트 한 곳의 영향(5.1 의 4번)을 줄이고 지역 의존을 점검할 수 있다 |
| XJ_tempderived_aux_labels | 지온 유도 라벨을 보조 라벨로 쓰는 다중 충실도 | 티베트 지온 유도 라벨에서 R1 − P1 판정이 같았다(TB7, L39 강건). 라벨 희소·심부 대상이 늘면 진단값 범위가 넓어진다 |
| XC_workflow_end_to_end | 워크플로 전체를 한 시험지에서 실행 | 진단 단계를 결정 규칙(예: 진단값이 클 때 라벨을 더 모은다)으로 넣었을 때의 오차·라벨 비용을 잴 수 있다. 상관 대신 결정 가치로 C2 를 다시 쓸 수 있다 |
| XE_hires_covariates | 격자 안 해상도 입력(토양 수분, 식생 등) | ML 몫이 격자 안 변동에서 커지면 ML 몫은 계수 오차와 더 무관해질 수 있다(C8 과 연결) |
| XG_product_comparison | 기존 ALT 제품과 같은 시험지 비교 | 직접 영향은 작다. 제품의 지역 편향을 같은 진단값으로 잴 수 있다(서술 후보) |
| XD_placement_policy, XH_validation_ladder, XI_climate_extrapolation_retest | 관측 위치 정책, 검증 사다리, 기후 외삽 재시험 | 직접 영향 없음 |

---

## 7. 파일

| 파일 | 내용 |
|---|---|
| `README.md` | 이 문서 |
| `posthoc_c2_split.py` | 2.3 의 사후 점검 계산. `OMP_NUM_THREADS=1 python paper/claims/C2_bias_diagnosis/posthoc_c2_split.py`(저장소 뿌리에서, 수 초) |
| `tables/MANIFEST.csv` | 사본과 산출의 원본 경로, sha256, 행 수(CSV 는 머리글 제외 데이터 행, JSON 은 1), 바이트. 사본 10개는 원본과 sha256 이 같음을 확인했다 |
| `tables/wf0_misspec.csv`, `wf0_meta.json` | WF0(A1, A2, RW5, RW6, TB8, CA4, 2.3 비율 표의 레나·러시아 동부 P1 − P0, 1.1 의 (4) 점검, 6절의 ρ 0.07 재확인) |
| `tables/wf_tests.csv` | WF1–WF4 판정 표(A3–A5, RW7, RW8, TB9, CA5, 2.2 캐나다 읽기의 `gain_w`) |
| `tables/wf_curve.csv` | WF1–WF4 곡선(CA7) |
| `tables/wf2b_tests.csv` | WF6–WF8 판정 표(CA8) |
| `tables/lg_tests.csv`, `lg_targets.csv` | LG 판정 표와 대상 표(RW1–RW3, RW9, CA6, 1.1 의 (4) 점검) |
| `tables/lgd_tests_lic.csv` | LGD 약관 확인분 판정 표(TB1–TB7, RW2·RW3 의 L4e PE1 판정) |
| `tables/lgw_bundle.csv` | 초록 주 대비 묶음 AB1–AB10(A6–A12, RW4, CA1–CA3) |
| `tables/paper_figs_fig3_c.csv` | v2 Fig 3c 원천 표(A13, A14, A12 의 LGX 보조 열) |
| `tables/derived_c2_*.csv` | 2.3 의 산출(사후, 판정 아님) |

복사한 표는 모두 대상·대비 단위의 집계 표이고 셀 단위 라벨 자료는 없다. 가장 큰 사본은 `wf_curve.csv`(1,950,113 바이트)다.
