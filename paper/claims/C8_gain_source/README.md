# C8 근거: 이득의 출처(격자 단위 편향 보정)

**성격**: 논문용 색인 폴더다(2026-10-04 작성). 원본 파일은 옮기거나 고치지 않았다. 근거 수치는 `tables/` 의 사본(원본과 sha256 일치, `tables/MANIFEST.csv`)에서 `extract_evidence.py` 로 읽었고, 같은 값을 `evidence_c8.csv` 에 남겼다. 다시 만들려면 `OMP_NUM_THREADS=1 python paper/claims/C8_gain_source/extract_evidence.py` 를 실행한다(pandas 읽기만 하고 모형 적합은 없다).

**기호(이 폴더 한정)**: P1 은 같은 라벨로 계수 E 를 수축 재보정한 Stefan 식, R1 은 재보정 앵커 위의 CatBoost 잔차(교차검증 λ 또는 고정 λ), D0 은 직접 CatBoost, Re 는 Stefan·CCI 평균 앵커 위의 잔차다. 격자 묶음은 채점 셀 가운데 (√TDD 값, 0.5° 블록)이 같은 셀의 묶음이며 ERA5-Land 기후 격자의 대용이다. 격자 안 저장소(`~w`)는 2셀 이상 묶음의 셀에서 묶음 평균을 뺀 값, 격자 사이 저장소(`~b`)는 묶음 평균이다(`docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 7절, 개정 이력 2026-10-02 00:20 항목 1).

## 1. 주장 문장

### 1.1 현행 문장

`docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md` 3절 C8:

> 물리식과 총 오차가 비슷한 지역에서 물리 기반 ML 의 이득은 격자 단위의 편향 보정에서 오고, ERA5 격자 안의 공간 상세도는 더하지 못했다.

같은 절의 근거 항목(85–90행)은 다음과 같다. 알래스카 지역 내 라벨 전량에서 총 이득 −0.52 cm 가운데 격자 사이 −1.03 cm(우세), 격자 안 −0.01 cm(동등)이다(사후 서술). 격자 안 설명 비율은 지역 내 알래스카 0.1–0.7 %, 레나 R1(λ 0.25) 5.6 %·직접 ML 6.2 %, 캐나다 음수, 전이 레나 3.2 %·캐나다 −0.4 %(87행). WF9-a 는 기각(라벨 전량에서 격자 안 오차가 오히려 0.09 cm 커짐, 토양 √TDD 묶음 민감도에서도 기각, +0.27 – +0.42 cm), WF9-c 는 토양 정의 민감도 뒤 기각, WF10 은 판정 불가. 이런 지역에서 주장할 장점으로 손해 없는 선택(C1), 계수 오차 진단(C2), 위성 제품 결합(WF7, 서술), 격자 단위 편향 보정을 든다.

'격자 안 설명 비율 0–6 %' 라는 요약 표기는 claims 문서 C8 근거에는 없다. 그 표기의 출처는 `docs/RESEARCH_OVERVIEW_2026-10-02.md` 60행, `docs/QA_FINAL_REVIEW_2026-10-02.md` 224행(Q8), `docs/EXPERIMENT_LOG.md` 14행, `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 36행(R7)이다. 이 표기의 성격은 1.2 의 3 에 적었다.

### 1.2 좁혀야 할 점

1. **범위(이 폴더 점검, 사후)**: '이득이 격자 사이 성분에서 온다'는 알래스카 지역 내 라벨 전량에서 성립한다. 총 SSE 이득 가운데 격자 안 성분은 교차검증 λ 에서 0.9 %(B4, 토양 정의 1.0 %, B10), λ 0.25 에서 6.7 %(B5)다. 레나 지역 내 λ 0.25 는 53.5 %(B6, 토양 정의 23.7 %, B11), 전이 라벨 전량은 레나 25.2 %(B8), 알래스카 35.6 %(B9)다. 따라서 '물리식과 총 오차가 비슷한 지역' 일반으로 쓰지 않고 '알래스카 지역 내'로 한정한다.
2. **격자 안 문장**: '더하지 못했다'는 등록 가설(교차검증 λ)의 판정이다. 등록 가설 밖의 고정 λ 0.25 서술 대비에서는 격자 안 3지역 층화 평균이 기온 정의 −0.16 cm(H1), 토양 정의 −0.08 cm(H3)로 우세였고, 대상 행으로는 레나가 우세였다(H2 −0.44, H4 −0.19 cm. 알래스카는 기온 정의 −0.04 cm 동등, WF 계획서 5.3 WF9-b 행). 크기는 모두 0.5 cm 미만이다. 등록 가설 WF9-a 는 두 격자 정의 모두 열세가 있는 기각이다(기온 정의 라벨 전량 +0.09 cm(D3), 토양 정의 n 500·1,000·전량 +0.27·+0.42·+0.40 cm(E1–E3)). 결과 전에 고정한 해석 규칙(WF 계획서 7절 163행)은 열세가 있는 기각이면 'ML 의 격자 안 변동은 오차를 늘렸다' 를 쓰라고 정했고, 5.3 해석 (1)(198행)과 개정 이력 2026-10-02 11:32(255, 258행)도 같은 문구를 쓴다. 따라서 지역 내 문장은 '지역 내 층화 평균에서 ML(R1, 교차검증 λ)의 격자 안 예측은 오차를 오히려 늘렸다(기온 정의 라벨 전량 +0.09 cm, 토양 정의 +0.27 – +0.42 cm, 모두 0.5 cm 미만)' 로 쓰고, 고정 λ 0.25 서술 대비의 작은 우세(0.5 cm 미만)는 SI 에 함께 적는다. '근거는 확인되지 않았다' 형식은 열세가 없는 기각인 전이(WF9-c 토양 정의, G1–G4 동등)에만 쓴다: '전이 조건에서도 ML 의 격자 안 예측이 실측 변동을 맞힌다는 근거는 확인되지 않았다'.
3. **설명 비율**: 판정 기록(WF 계획서 5.3 설명 비율 항목)에 적힌 키(R1 교차검증 λ·λ 0.25, D0, 라벨 전량, C1–C11)의 값은 지역 내 −8.3 %(캐나다 D0, C7)에서 6.2 %(레나 D0, C5)까지, 전이를 넣으면 −8.5 %(러시아 E R1 λ 0.25, C11)에서 6.2 % 까지다. 다른 문서의 '0–6 %' 는 이 값들의 범위가 아니라 음수(설명력 없음)를 0 으로 본 요약이다. 저장된 모든 ML 키를 넣으면 지역 내 최댓값은 기온 정의 7.8 %(C12, 레나 R1 λ 0.5, 라벨 1,000), 토양 정의 4.0 %(C13), 전이 7.2 %(C14, 러시아 E R2 λ 1.0)다. 알래스카 지역 내 ML 키 전체의 최댓값은 0.7 %(C2, 2.3 아래 단서)다. 본문에는 알래스카 '1 % 미만(최대 0.7 %)', 세 지역 전체 '8 % 미만(최대 레나 7.8 %)' 으로 나눠 쓴다. 등록 정의(SSE 비)로는 지역 내 최댓값이 7.99 % 라 '8 % 미만' 의 여유가 작다(2.3 아래 단서).
4. **C2 연결(`docs/QA_FINAL_REVIEW_2026-10-02.md` Q1 C2 점검)**: C8 의 장점 목록에 있는 '계수 오차 진단(C2)'은 재보정을 포함한 이득에 대한 진단이다. 재보정 이득과 P0 대비 최선 ML 이득의 순위상관은 0.66(K2)이지만, 재보정을 넘어선 ML 이득과의 순위상관은 −0.08(p 0.67, K1)이다. C8 의 대비(R1 − P1)는 이미 재보정을 넘어선 ML 몫이다. 장점 목록의 C2 문장은 QA 권고 문장으로 바꾼다: '라벨을 쓰는 보정(재보정 + 잔차)의 이득은 원천 계수 오차의 크기에 비례하고, 라벨 10개로 그 크기를 진단할 수 있다. 재보정을 넘어선 ML 의 추가 이득은 계수 오차 크기와 상관이 없었다.'
5. **'안전'·'손해 없는' 표현**: 장점 목록의 '손해 없는 선택(C1)'은 WF0 사후 서술(라벨 0 위험표)이며 등록된 비열등 시험이 아니다. C8 에 걸린 유일한 등록 비열등 가설 WF10-c 는 판정 불가였다(I3). C8 에서는 '안전', '손해 없는' 을 쓰지 않고, 필요하면 `paper/claims/C1_label0_safety/` 의 서술 수치를 출처와 함께 인용한다.
6. **사후 설계 표지**: WF9·WF10 은 WF1–WF8 판정을 연 뒤 등록했다(2026-10-02 00:05, 등록 커밋 8180632). WF9 지역 내 arm 의 총 RMSE 는 WF6 와 같은 추출이라 이미 열람한 값이다. '이득은 격자 사이 성분에서 왔다'는 WF 계획서 5.3 해석 (2)가 밝힌 사후 해석이다. 토양 √TDD 민감도는 이탈 기록(10:26) 뒤에 등록했다(10:51). 본문과 SI 에 '결과 열람 뒤 설계' 표지를 단다.

### 1.3 권고 문장

국문(결과 절):

> 알래스카 지역 내 라벨 전량에서 물리 잔차 ML(R1, 교차검증 λ)의 RMSE 는 재보정 Stefan 식(P1)보다 0.52 cm 작았고, 오차 제곱 감소의 약 99 % 는 ERA5 기후 격자 사이 성분, 곧 격자 단위의 편향 보정에서 왔다(격자 사이 −1.03 cm 우세, 격자 안 −0.01 cm 동등, 두 격자 정의에서 소수 둘째 자리까지 같음). 같은 기후 격자 안의 변동은 Stefan 오차 제곱의 72 % 를 차지했으나, 알래스카에서 ML 이 설명한 격자 안 변동은 1 % 미만이었다(최대 0.7 %). 세 지역의 저장된 모든 ML 키에서도 8 % 미만이었다(최대 레나 7.8 %). 지역 내 층화 평균에서 R1 의 격자 안 예측은 오차를 오히려 늘렸다(기온 정의 라벨 전량 +0.09 cm, 토양 정의 +0.27 – +0.42 cm, 모두 0.5 cm 미만). 전이 조건에서도 ML 의 격자 안 예측이 실측 변동을 맞힌다는 근거는 확인되지 않았다(토양 정의 민감도 뒤 기각). 학습 범위보다 따뜻한 블록으로의 외삽은 판정할 수 없었다. 이 분석은 앞선 결과를 연 뒤 설계했다.

영문(Results):

> Within Alaska, with all available labels, the physics-residual model (R1, cross-validated λ) had an RMSE 0.52 cm lower than the recalibrated Stefan model (P1). About 99% of the reduction in squared error came from the between-grid component, that is, from correcting bias at the scale of ERA5 climate grid cells (between-grid −1.03 cm, superior; within-grid −0.01 cm, equivalent; the same to two decimal places under both grid definitions). Variation within a climate grid cell accounted for 72% of the Stefan squared error, but in Alaska the machine-learning models explained less than 1% of it (maximum 0.7%). Across all stored machine-learning configurations in the three regions, the explained fraction remained below 8% (maximum 7.8%, Lena). Within regions, the within-grid predictions of R1 increased the error in the stratified mean (+0.09 cm with all labels under the air-temperature grid definition and +0.27 to +0.42 cm under the soil grid definition, all below 0.5 cm). In transfer, there was no evidence that the within-grid predictions of the machine-learning models matched the observed variation (rejected after the soil-grid sensitivity analysis). Extrapolation to blocks warmer than the training range could not be evaluated. These analyses were designed after earlier results had been inspected.

수치 출처: 0.52 = A1, 약 99 % = 1 − B4(0.9 %)와 1 − B10(1.0 %), −1.03 = A2·A5, −0.01 = A3·A6('소수 둘째 자리까지 같음': 격자 사이 −1.026090 대 −1.025766, 격자 안 −0.009052 대 −0.009623, R-T·S-T), 72 % = B1, 1 % 미만(최대 0.7 %) = C2(알래스카 ML 키 전체의 최댓값, 2.3 아래 단서), 8 % 미만(최대 7.8 %) = C12–C14, +0.09 = D3, +0.27 – +0.42 = E1–E3, 기각 = D6·E6·G6, 판정 불가 = I1–I3. R1 은 교차검증 λ 로 밝힌다. λ 0.25 에서는 총 대비가 −0.54 cm(R-T `test_id=WF9-b; contrast=R1(λ 0.25)-P1[총]|n전량; target=Alaska|r`, 셀 가중 −0.541382)이고 SSE 이득의 격자 안 몫이 6.7 %(B5, 격자 사이 약 93 %)라 값이 다르다.

## 2. 근거 표

**읽는 법**
- 원천 약어: R-T = `tables/rescale_wf3__wf3b_tests.csv`(원본 `results/rescale_wf3/data/processed/wf/wf3b_tests.csv`, Rescale 본 실행 Vppbeb, 기온 √TDD 묶음), S-T = `tables/wf_soil__wf3b_tests.csv`(원본 `data/processed/wf_soil/wf3b_tests.csv`, 로컬 CPU, 토양 √TDD 묶음), R-D = `tables/rescale_wf3__wf3b_decomp.csv`(원본 `results/rescale_wf3/data/processed/wf/wf3b_decomp.csv`), S-D = `tables/wf_soil__wf3b_decomp.csv`(원본 `data/processed/wf_soil/wf3b_decomp.csv`), R-G = `tables/rescale_wf3__wf3b_targets.csv`(원본 `results/rescale_wf3/data/processed/wf/wf3b_targets.csv`), RC = `tables/wf__wf9_repro_check.csv`(원본 `data/processed/wf/wf9_repro_check.csv`), MS = `tables/wf__wf0_misspec.csv`(원본 `data/processed/wf/wf0_misspec.csv`).
- 행 필터의 `n=-1` 은 라벨 전량, `lam=-1` 은 교차검증 λ 다. WF9-b 서술 묶음의 층화 평균 행과 대상 서술 행은 lam 열이 비어 있어 NA 로 적었다.
- 값은 파일 값을 소수 둘째 자리로 반올림했다. 대비 행은 `delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq]`(셀 가중 / 블록 등가중, 블록 재표집 95 % CI, 재표집 10,000회는 meta `nboot`)이고 단위는 cm 다. 대비는 대부분 M − P1 형식이라 음수면 M 의 오차가 작다(WF10-a·b 는 외삽 손실 EP 의 대비, WF10-b 의 기준은 D0 이다). 비율 행은 괄호에 백분율(파일 값 × 100, 소수 첫째 자리)을 덧붙였다.
- 판정어는 파일의 `verdict4` 열이다. 규칙(meta `rules.verdict4`): 우세는 두 가중 CI 상한 < 0, 열세는 두 가중 CI 하한 > 0, 동등은 네 끝값의 절댓값이 모두 0.5 cm 이하, 그 밖은 미결정이다. 우세·열세가 동등보다 앞선다(두 조건을 모두 만족하면 우세 또는 열세다. 이 폴더의 표에서는 우세 F3·H1–H4, 열세 D5·E1·E5 가 그렇다). 끝값이 비유한이면 판정 불가다(구현 `scripts/3_deep_learning/h42_label_grid_ext.py` 의 `verdict4`). 비율 행(decomp)에는 CI 가 없어 '서술'로 적었다.
- '판정 기록' 열의 'WF 계획서'는 `docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 다. '판정 문서 없음'은 판정 기록에 없는 값을 이 폴더에서 처음 인용한 것이며 판정이 아니라 서술이다.

### 2.1 알래스카 지역 내 이득의 분해(라벨 전량, 교차검증 λ)

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| A1 | 알래스카 지역 내 전량, R1(교차검증 λ) − P1, 총 | R-T | `test_id=WF9-b; contrast=R1(λ cv)-P1[총]\|n전량; target=Alaska\|r; n=-1; lam=-1.00` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.52 [-0.70, -0.31] / -0.68 [-0.91, -0.44] | 우세 | WF 계획서 5.3 해석 (2)(사후) |
| A2 | 같은 대비, 격자 사이 저장소 | R-T | `test_id=WF9-b; contrast=R1(λ cv)-P1[격자 사이]\|n전량; target=Alaska~b\|r; n=-1; lam=-1.00` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -1.03 [-1.30, -0.59] / -0.89 [-1.16, -0.61] | 우세 | WF 계획서 5.3 WF9-b 행, 해석 (2)(사후) |
| A3 | 같은 대비, 격자 안 저장소(WF9-a 대상 행) | R-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n전량; target=Alaska~w\|r; n=-1; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.01 [-0.03, 0.04] / 0.04 [0.01, 0.06] | 동등 | WF 계획서 5.3 WF9-a 행 |
| A4 | 토양 정의: 총 | S-T | `test_id=WF9-b; contrast=R1(λ cv)-P1[총]\|n전량; target=Alaska\|r; n=-1; lam=-1.00` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.52 [-0.70, -0.31] / -0.68 [-0.91, -0.44] | 우세 | WF 계획서 개정 이력 2026-10-02 11:32 |
| A5 | 토양 정의: 격자 사이 | S-T | `test_id=WF9-b; contrast=R1(λ cv)-P1[격자 사이]\|n전량; target=Alaska~b\|r; n=-1; lam=-1.00` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -1.03 [-1.30, -0.59] / -0.89 [-1.16, -0.61] | 우세 | WF 계획서 개정 이력 2026-10-02 11:32 대상 행 |
| A6 | 토양 정의: 격자 안 | S-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n전량; target=Alaska~w\|r; n=-1; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.01 [-0.03, 0.04] / 0.04 [0.01, 0.06] | 동등 | WF 계획서 개정 이력 2026-10-02 11:32 대상 행 |

### 2.2 SSE 기준 비율(서술, CI 없음)

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| B1 | P1 총 SSE 중 격자 안 비율, 알래스카 지역 내 전량 | R-D | `exp=wf9; base=Alaska\|r; method=P1; learner=none; n=-1; lam=0.0` | share_within_of_sse | 0.72 (71.9 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| B2 | 같음, 레나 | R-D | `exp=wf9; base=Lena\|r; method=P1; learner=none; n=-1; lam=0.0` | share_within_of_sse | 0.62 (61.7 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| B3 | 같음, 캐나다 | R-D | `exp=wf9; base=Canada\|r; method=P1; learner=none; n=-1; lam=0.0` | share_within_of_sse | 0.37 (37.4 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| B4 | R1(교차검증 λ) 총 SSE 이득 중 격자 안 성분 비율, 알래스카 지역 내 | R-D | `exp=wf9; base=Alaska\|r; method=R1; learner=catboost_lo; n=-1; lam=-1.0` | share_gain_within | 0.01 (0.9 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| B5 | R1(λ 0.25), 알래스카 지역 내 | R-D | `exp=wf9; base=Alaska\|r; method=R1; learner=catboost_lo; n=-1; lam=0.25` | share_gain_within | 0.07 (6.7 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| B6 | R1(λ 0.25), 레나 지역 내 | R-D | `exp=wf9; base=Lena\|r; method=R1; learner=catboost_lo; n=-1; lam=0.25` | share_gain_within | 0.53 (53.5 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| B7 | R1(λ 0.25), 캐나다 지역 내 | R-D | `exp=wf9; base=Canada\|r; method=R1; learner=catboost_lo; n=-1; lam=0.25` | share_gain_within | -0.00 (-0.1 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| B8 | R1(λ 0.25), 레나 전이 전량 | R-D | `exp=wf9x; base=Lena\|x; method=R1; learner=catboost_lo; n=-1; lam=0.25` | share_gain_within | 0.25 (25.2 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| B9 | R1(λ 0.25), 알래스카 전이 전량 | R-D | `exp=wf9x; base=Alaska\|x; method=R1; learner=catboost_lo; n=-1; lam=0.25` | share_gain_within | 0.36 (35.6 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| B10 | 토양 정의: R1(교차검증 λ), 알래스카 지역 내 | S-D | `exp=wf9; base=Alaska\|r; method=R1; learner=catboost_lo; n=-1; lam=-1.0` | share_gain_within | 0.01 (1.0 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| B11 | 토양 정의: R1(λ 0.25), 레나 지역 내 | S-D | `exp=wf9; base=Lena\|r; method=R1; learner=catboost_lo; n=-1; lam=0.25` | share_gain_within | 0.24 (23.7 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| B12 | P1 의 decomp RMSE 열(총; 격자 사이; 격자 안), 알래스카 지역 내 | R-D | `exp=wf9; base=Alaska\|r; method=P1; learner=none; n=-1; lam=0.0` | rmse_tot; rmse_b; rmse_w | 14.53; 7.68; 12.37 | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| B13 | R1(교차검증 λ) 의 decomp RMSE 열, 알래스카 지역 내 | R-D | `exp=wf9; base=Alaska\|r; method=R1; learner=catboost_lo; n=-1; lam=-1.0` | rmse_tot; rmse_b; rmse_w | 14.06; 6.76; 12.36 | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| B14 | R1(교차검증 λ) 의 총 SSE 이득과 격자 안 비율, 캐나다 지역 내 | R-D | `exp=wf9; base=Canada\|r; method=R1; learner=catboost_lo; n=-1; lam=-1.0` | gain_total_sse; share_gain_within | -12,133.57; 0.21 | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |

### 2.3 격자 안 설명 비율 expl_within = 1 − (rmse_w(M)/rmse_w(P1))²(서술, CI 없음, C1–C11 은 라벨 전량, C12–C14 는 모든 n)

파일 열 `expl_within` 은 분할 평균 격자 안 RMSE 의 비로 계산된 값이다(R-D 의 wf9·wf9x ML 키 238개에서 1 − (rmse_w(M)/rmse_w(P1))² 와의 차 최대 1e-15 미만). 등록 WF9-d 정의(WF 계획서 7절 162행)는 1 − SSE_w(M)/SSE_w(P1) 이고, 5.3 은 '1 − MSE_w(M)/MSE_w(P1)' 로 적었다. 차이와 단서는 표 아래에 적는다.

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| C1 | R1(교차검증 λ), 알래스카 지역 내 | R-D | `exp=wf9; base=Alaska\|r; method=R1; learner=catboost_lo; n=-1; lam=-1.0` | expl_within | 0.00 (0.1 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C2 | R1(λ 0.25), 알래스카 지역 내 | R-D | `exp=wf9; base=Alaska\|r; method=R1; learner=catboost_lo; n=-1; lam=0.25` | expl_within | 0.01 (0.7 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C3 | D0(catboost), 알래스카 지역 내 | R-D | `exp=wf9; base=Alaska\|r; method=D0; learner=catboost; n=-1; lam=1.0` | expl_within | -0.02 (-2.0 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C4 | R1(λ 0.25), 레나 지역 내 | R-D | `exp=wf9; base=Lena\|r; method=R1; learner=catboost_lo; n=-1; lam=0.25` | expl_within | 0.06 (5.6 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C5 | D0(catboost), 레나 지역 내 | R-D | `exp=wf9; base=Lena\|r; method=D0; learner=catboost; n=-1; lam=1.0` | expl_within | 0.06 (6.2 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C6 | R1(교차검증 λ), 캐나다 지역 내 | R-D | `exp=wf9; base=Canada\|r; method=R1; learner=catboost_lo; n=-1; lam=-1.0` | expl_within | -0.02 (-2.2 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C7 | D0(catboost), 캐나다 지역 내 | R-D | `exp=wf9; base=Canada\|r; method=D0; learner=catboost; n=-1; lam=1.0` | expl_within | -0.08 (-8.3 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C8 | R1(λ 0.25), 레나 전이 | R-D | `exp=wf9x; base=Lena\|x; method=R1; learner=catboost_lo; n=-1; lam=0.25` | expl_within | 0.03 (3.2 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C9 | R1(λ 0.25), 알래스카 전이 | R-D | `exp=wf9x; base=Alaska\|x; method=R1; learner=catboost_lo; n=-1; lam=0.25` | expl_within | 0.02 (2.0 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C10 | R1(λ 0.25), 캐나다 전이 | R-D | `exp=wf9x; base=Canada\|x; method=R1; learner=catboost_lo; n=-1; lam=0.25` | expl_within | -0.00 (-0.4 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C11 | R1(λ 0.25), 러시아 E 전이 | R-D | `exp=wf9x; base=Russia_E\|x; method=R1; learner=catboost_lo; n=-1; lam=0.25` | expl_within | -0.08 (-8.5 %) | 서술 | WF 계획서 5.3 설명 비율(서술) |
| C12 | 지역 내 ML 키 전체의 최댓값(기온 정의) | R-D | `exp=wf9; method in D0,D1,R1,R2,Re; 모든 n·lam 의 최댓값 → base=Lena\|r, method=R1, n=1000, lam=0.5` | expl_within (max) | 0.08 (7.8 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| C13 | 지역 내 ML 키 전체의 최댓값(토양 정의) | S-D | `exp=wf9; method in D0,D1,R1,R2,Re; 모든 n·lam 의 최댓값 → base=Lena\|r, method=Re, n=1000, lam=0.25` | expl_within (max) | 0.04 (4.0 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| C14 | 전이 ML 키 전체의 최댓값(기온 정의, 토양 정의도 같은 값) | R-D | `exp=wf9x; method in D0,D1,R1,R2,Re; 모든 n·lam 의 최댓값 → base=Russia_E\|x, method=R2, n=-1, lam=1.0` | expl_within (max) | 0.07 (7.2 %) | 서술 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |

2.3 의 단서(2026-10-04 정정 때 R-D·S-D 에서 계산, `evidence_c8.csv` 에는 없는 값):
- **알래스카 최댓값**: 알래스카 지역 내 ML 키 전체(D0, D1, R1, R2, Re, 모든 n·λ)의 최댓값은 C2 행(R1 λ 0.25, 라벨 전량)이다. 기온 정의 0.67 %(0.0067384), 토양 정의 0.68 %(0.0067866)다.
- **라벨 전량 최댓값**: C12·C13 은 모든 n 의 최댓값이다. 라벨 전량으로 한정하면 지역 내 최댓값은 기온 정의 7.37 %(레나 R1 λ 0.5, 0.073704), 토양 정의 3.91 %(레나 Re λ 0.25, 0.039132)다.
- **등록 정의(SSE 비)와의 차이**: decomp 의 `sse_mean_w` 로 1 − SSE_w(M)/SSE_w(P1) 을 계산하면 지역 내 최댓값은 기온 정의 7.99 %(레나 R1 λ 0.5, n 1,000. 파일 7.77 %), 토양 정의 3.93 %(레나 Re λ 0.25, n 1,000), 전이 7.22 %(러시아 E R2 λ 1.0, 파일과 같음)다. 레나 D0 라벨 전량(C5)은 SSE 비 정의로 6.65 %(파일 6.18 %)다. '8 % 미만' 은 두 정의에서 모두 성립하지만 SSE 비 정의에서는 여유가 0.01 %p 다.

### 2.4 등록 가설 WF9-a(지역 내 격자 안, R1(교차검증 λ) − P1), 기온 √TDD 묶음

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| D1 | 층화 평균 n 500(알래스카·레나) | R-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n500; target=MEAN[Alaska~w\|r,Lena~w\|r]; n=500; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.08 [-0.16, 0.15] / 0.22 [0.09, 0.35] | 동등 | WF 계획서 5.3 WF9-a 행 |
| D2 | 층화 평균 n 1,000(알래스카·레나) | R-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n1000; target=MEAN[Alaska~w\|r,Lena~w\|r]; n=1000; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.02 [-0.12, 0.35] / 0.36 [0.19, 0.54] | 미결정 | WF 계획서 5.3 WF9-a 행 |
| D3 | 층화 평균 전량(3지역) | R-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n전량; target=MEAN[Alaska~w\|r,Lena~w\|r,Canada~w\|r]; n=-1; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.09 [0.02, 0.40] / 0.37 [0.25, 0.50] | 열세 | WF 계획서 5.3 WF9-a 행 |
| D4 | 레나 전량 | R-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n전량; target=Lena~w\|r; n=-1; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.05 [-0.18, 0.92] / 0.87 [0.51, 1.24] | 미결정 | WF 계획서 5.3 WF9-a 행 |
| D5 | 캐나다 전량 | R-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n전량; target=Canada~w\|r; n=-1; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.23 [0.16, 0.34] / 0.21 [0.12, 0.31] | 열세 | WF 계획서 5.3 WF9-a 행 |
| D6 | 판정 문구 | R-T | `test_id=WF9-a; scope=verdict` | verdict | 부분(지역 2/3, 분할 23/24): 기각: 열세인 대비가 있다(n=전량) [통계적으로 구별되나 크기는 0.5 cm 미만: n=전량] |  | WF 계획서 5.3 WF9-a 행 |

### 2.5 등록 가설 WF9-a, 토양 √TDD 묶음(민감도)

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| E1 | 층화 평균 n 500(알래스카·레나) | S-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n500; target=MEAN[Alaska~w\|r,Lena~w\|r]; n=500; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.27 [0.07, 0.41] / 0.28 [0.15, 0.41] | 열세 | WF 계획서 개정 이력 2026-10-02 11:32 WF9-a |
| E2 | 층화 평균 n 1,000(알래스카·레나) | S-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n1000; target=MEAN[Alaska~w\|r,Lena~w\|r]; n=1000; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.42 [0.20, 0.64] / 0.44 [0.26, 0.61] | 열세 | WF 계획서 개정 이력 2026-10-02 11:32 WF9-a |
| E3 | 층화 평균 전량(3지역) | S-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n전량; target=MEAN[Alaska~w\|r,Lena~w\|r,Canada~w\|r]; n=-1; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.40 [0.24, 0.59] / 0.42 [0.30, 0.55] | 열세 | WF 계획서 개정 이력 2026-10-02 11:32 WF9-a |
| E4 | 레나 전량 | S-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n전량; target=Lena~w\|r; n=-1; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.99 [0.49, 1.52] / 1.02 [0.67, 1.39] | 열세 | WF 계획서 개정 이력 2026-10-02 11:32 WF9-a |
| E5 | 캐나다 전량 | S-T | `test_id=WF9-a; contrast=R1(λ cv)-P1[격자 안]\|n전량; target=Canada~w\|r; n=-1; lam=-1.00; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.23 [0.16, 0.33] / 0.21 [0.11, 0.31] | 열세 | WF 계획서 개정 이력 2026-10-02 11:32 WF9-a |
| E6 | 판정 문구 | S-T | `test_id=WF9-a; scope=verdict` | verdict | 부분(지역 2/3, 분할 23/24): 기각: 열세인 대비가 있다(n=500, n=1000, n=전량) [통계적으로 구별되나 크기는 0.5 cm 미만: n=500, n=1000, n=전량] |  | WF 계획서 개정 이력 2026-10-02 11:32 WF9-a |

### 2.6 등록 가설 WF9-c(전이 격자 안, R1(λ 0.25) − P1), 기온 √TDD 묶음

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| F1 | 층화 평균 n 0(레나·캐나다) | R-T | `test_id=WF9-c; contrast=R1(λ 0.25)-P1[격자 안]\|n0; target=MEAN[Lena~w\|x,Canada~w\|x]; n=0; lam=0.25; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.01 [-0.04, 0.01] / -0.02 [-0.06, 0.00] | 동등 | WF 계획서 5.3 WF9-c 행 |
| F2 | 층화 평균 n 10 | R-T | `test_id=WF9-c; contrast=R1(λ 0.25)-P1[격자 안]\|n10; target=MEAN[Lena~w\|x,Canada~w\|x]; n=10; lam=0.25; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.01 [-0.04, 0.01] / -0.04 [-0.07, -0.00] | 동등 | WF 계획서 5.3 WF9-c 행 |
| F3 | 층화 평균 전량 | R-T | `test_id=WF9-c; contrast=R1(λ 0.25)-P1[격자 안]\|n전량; target=MEAN[Lena~w\|x,Canada~w\|x]; n=-1; lam=0.25; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.11 [-0.25, -0.01] / -0.12 [-0.22, -0.03] | 우세 | WF 계획서 5.3 WF9-c 행 |
| F4 | 레나 전량 | R-T | `test_id=WF9-c; contrast=R1(λ 0.25)-P1[격자 안]\|n전량; target=Lena~w\|x; n=-1; lam=0.25; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.27 [-0.52, -0.05] / -0.22 [-0.42, -0.04] | 우세 | WF 계획서 5.3 WF9-c 행 |
| F5 | 러시아 E 전량(채점 블록이 적어 CI 없음) | R-T | `test_id=WF9-c; contrast=R1(λ 0.25)-P1[격자 안]\|n전량; target=Russia_E~w\|x; n=-1; lam=0.25; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 1.08 [NA, NA] / 1.08 [NA, NA] | 판정 불가 | WF 계획서 5.3 WF9-c 행 |
| F6 | 판정 문구 | R-T | `test_id=WF9-c; scope=verdict` | verdict | 부분(지역 2/4): 부분 지지: 우세인 대비 = n=전량 [통계적으로 구별되나 크기는 0.5 cm 미만: n=전량] |  | WF 계획서 5.3 WF9-c 행 |

### 2.7 등록 가설 WF9-c, 토양 √TDD 묶음(민감도)

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| G1 | 층화 평균 n 0(레나·캐나다) | S-T | `test_id=WF9-c; contrast=R1(λ 0.25)-P1[격자 안]\|n0; target=MEAN[Lena~w\|x,Canada~w\|x]; n=0; lam=0.25; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.00 [-0.04, 0.02] / -0.02 [-0.05, 0.01] | 동등 | WF 계획서 개정 이력 2026-10-02 11:32 WF9-c |
| G2 | 층화 평균 n 10 | S-T | `test_id=WF9-c; contrast=R1(λ 0.25)-P1[격자 안]\|n10; target=MEAN[Lena~w\|x,Canada~w\|x]; n=10; lam=0.25; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.01 [-0.03, 0.02] / -0.03 [-0.07, 0.00] | 동등 | WF 계획서 개정 이력 2026-10-02 11:32 WF9-c |
| G3 | 층화 평균 전량 | S-T | `test_id=WF9-c; contrast=R1(λ 0.25)-P1[격자 안]\|n전량; target=MEAN[Lena~w\|x,Canada~w\|x]; n=-1; lam=0.25; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.04 [-0.20, 0.04] / -0.10 [-0.21, -0.01] | 동등 | WF 계획서 개정 이력 2026-10-02 11:32 WF9-c |
| G4 | 레나 전량 | S-T | `test_id=WF9-c; contrast=R1(λ 0.25)-P1[격자 안]\|n전량; target=Lena~w\|x; n=-1; lam=0.25; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.13 [-0.42, 0.05] / -0.19 [-0.39, -0.01] | 동등 | WF 계획서 개정 이력 2026-10-02 11:32 WF9-c |
| G5 | 러시아 E 전량(CI 없음) | S-T | `test_id=WF9-c; contrast=R1(λ 0.25)-P1[격자 안]\|n전량; target=Russia_E~w\|x; n=-1; lam=0.25; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 1.08 [NA, NA] / 1.08 [NA, NA] | 판정 불가 | WF 계획서 개정 이력 2026-10-02 11:32 WF9-c |
| G6 | 판정 문구 | S-T | `test_id=WF9-c; scope=verdict` | verdict | 부분(지역 2/4): 기각: 우세인 대비가 없다 |  | WF 계획서 개정 이력 2026-10-02 11:32 WF9-c |

### 2.8 등록 가설 밖의 서술 대비(지역 내 라벨 전량)

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| H1 | R1(λ 0.25) − P1 격자 안, 3지역 층화 평균(기온) | R-T | `test_id=WF9-b; contrast=R1(λ 0.25)-P1[격자 안]\|n전량; target=MEAN[Alaska~w\|r,Lena~w\|r,Canada~w\|r]; n=-1; lam=NA` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.16 [-0.17, -0.07] / -0.09 [-0.13, -0.04] | 우세 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| H2 | 같음, 레나(기온) | R-T | `test_id=WF9-b; contrast=R1(λ 0.25)-P1[격자 안]\|n전량; target=Lena~w\|r; n=-1; lam=NA; role=서술` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.44 [-0.50, -0.19] / -0.26 [-0.40, -0.14] | 우세 | WF 계획서 5.3 WF9-b 행 |
| H3 | 같음, 3지역 층화 평균(토양) | S-T | `test_id=WF9-b; contrast=R1(λ 0.25)-P1[격자 안]\|n전량; target=MEAN[Alaska~w\|r,Lena~w\|r,Canada~w\|r]; n=-1; lam=NA` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.08 [-0.11, -0.02] / -0.07 [-0.12, -0.03] | 우세 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| H4 | 같음, 레나(토양) | S-T | `test_id=WF9-b; contrast=R1(λ 0.25)-P1[격자 안]\|n전량; target=Lena~w\|r; n=-1; lam=NA; role=서술` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.19 [-0.30, -0.04] / -0.22 [-0.35, -0.10] | 우세 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| H5 | R1(λ 0.25) − P1 격자 사이, 알래스카 | R-T | `test_id=WF9-b; contrast=R1(λ 0.25)-P1[격자 사이]\|n전량; target=Alaska~b\|r; n=-1; lam=0.25` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.99 [-1.11, -0.69] / -0.58 [-0.71, -0.45] | 우세 | WF 계획서 5.3 WF9-b 행 |
| H6 | 같음, 레나 | R-T | `test_id=WF9-b; contrast=R1(λ 0.25)-P1[격자 사이]\|n전량; target=Lena~b\|r; n=-1; lam=0.25` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.51 [-1.14, -0.35] / -1.27 [-1.65, -0.90] | 우세 | WF 계획서 5.3 WF9-b 행 |
| H7 | 같음, 캐나다 | R-T | `test_id=WF9-b; contrast=R1(λ 0.25)-P1[격자 사이]\|n전량; target=Canada~b\|r; n=-1; lam=0.25` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -0.93 [-1.39, -0.48] / -1.39 [-1.69, -1.08] | 우세 | WF 계획서 5.3 WF9-b 행 |
| H8 | D0(catboost) − P1 격자 안, 3지역 층화 평균 | R-T | `test_id=WF9-b; contrast=D0(catboost)-P1[격자 안]\|n전량; target=MEAN[Alaska~w\|r,Lena~w\|r,Canada~w\|r]; n=-1; lam=NA` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.16 [0.10, 0.42] / 0.46 [0.33, 0.59] | 열세 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |
| H9 | Re(Stefan·CCI 앵커, 교차검증 λ) − P1 격자 안, 3지역 층화 평균 | R-T | `test_id=WF9-b; contrast=Re(λ cv)-P1[격자 안]\|n전량; target=MEAN[Alaska~w\|r,Lena~w\|r,Canada~w\|r]; n=-1; lam=NA` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.31 [0.21, 0.59] / 0.42 [0.28, 0.56] | 열세 | 판정 문서 없음(이 폴더에서 처음 인용, 서술) |

### 2.9 WF10 기후 외삽(warm 변형)

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| I1 | WF10-a 판정 문구 | R-T | `test_id=WF10-a; scope=verdict` | verdict | 판정 불가(대비 0/2; 없는 대비: n=100, n=전량) |  | WF 계획서 5.3 WF10 행 |
| I2 | WF10-b 판정 문구 | R-T | `test_id=WF10-b; scope=verdict` | verdict | 판정 불가(대비 0/2; 없는 대비: n=100, n=전량) |  | WF 계획서 5.3 WF10 행 |
| I3 | WF10-c 판정 문구 | R-T | `test_id=WF10-c; scope=verdict` | verdict | 판정 불가(대비 0/2; 없는 대비: n=100, n=전량) |  | WF 계획서 5.3 WF10 행 |
| I4 | EP(D0 − P1), 알래스카 warm, n 100 | R-T | `test_id=WF10-a; contrast=EP[D0(catboost)-P1]\|warm\|n100; target=Alaska~warm\|r; n=100; lam=NA; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 7.02 [3.91, 9.51] / 0.75 [-1.55, 3.04] | 미결정 | WF 계획서 5.3 WF10 행 |
| I5 | 같음, n 500(서술 행) | R-T | `test_id=WF10-a; contrast=EP[D0(catboost)-P1]\|warm\|n500; target=Alaska~warm\|r; n=500; lam=NA; role=서술` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 4.55 [1.58, 7.09] / -1.31 [-3.54, 0.92] | 미결정 | WF 계획서 5.3 WF10 행(WF10-a 행 'n 500 +4.55') |
| I6 | 같음, 전량 | R-T | `test_id=WF10-a; contrast=EP[D0(catboost)-P1]\|warm\|n전량; target=Alaska~warm\|r; n=-1; lam=NA; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 3.39 [0.61, 5.95] / -1.33 [-3.41, 0.77] | 미결정 | WF 계획서 5.3 WF10 행 |
| I7 | EP(R1 − D0), 알래스카 warm, n 100 | R-T | `test_id=WF10-b; contrast=EP[R1(λ cv)-D0(catboost)]\|warm\|n100; target=Alaska~warm\|r; n=100; lam=NA; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -5.56 [-7.87, -3.02] / -1.38 [-3.29, 0.52] | 미결정 | WF 계획서 5.3 WF10 행 |
| I8 | 같음, 전량 | R-T | `test_id=WF10-b; contrast=EP[R1(λ cv)-D0(catboost)]\|warm\|n전량; target=Alaska~warm\|r; n=-1; lam=NA; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | -2.45 [-4.73, 0.09] / 1.59 [-0.25, 3.38] | 미결정 | WF 계획서 5.3 WF10 행 |
| I9 | W 의 R1 − P1, 알래스카 warm, n 100 | R-T | `test_id=WF10-c; contrast=R1(λ cv)-P1[W]\|warm\|n100; target=Alaska~warmW\|r; n=100; lam=NA; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.87 [-0.12, 1.16] / -1.32 [-1.83, -0.78] | 미결정 | WF 계획서 5.3 WF10 행 |
| I10 | 같음, 전량 | R-T | `test_id=WF10-c; contrast=R1(λ cv)-P1[W]\|warm\|n전량; target=Alaska~warmW\|r; n=-1; lam=NA; role=주` | delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq] | 0.13 [-0.54, 0.41] / -0.51 [-0.93, -0.06] | 미결정 | WF 계획서 5.3 WF10 행 |
| I11 | 캐나다 warm 의 외삽 채점(W) 블록 수 | R-G | `exp=wf10; target=Canada; wf10_variant=warm; 분할 10개` | nb_eval_W (min–max) | 3–3 |  | WF 계획서 개정 이력 2026-10-02 00:20 항목 4, WF 계획서 5.3 |
| I12 | 알래스카 warm 의 W 블록 수 | R-G | `exp=wf10; target=Alaska; wf10_variant=warm; 분할 10개` | nb_eval_W (min–max) | 17–17 |  | WF 계획서 개정 이력 2026-10-02 00:20 항목 4, WF 계획서 5.3 |
| I13 | 완료되지 않은(partial) 작업 단위 | R-G | `exp=wf10; status!=ok` | target, wf10_variant, split, status | Alaska warm 분할 10 partial |  | WF 계획서 5.3 원천 |

### 2.10 재현 점검

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| J1 | WF9 지역 내 총 저장소와 WF6 조각의 대조 | RC | `전체 행` | 행 수; ok 모두 True; n_common 합; max_abs_dsse 최대; n_count_mismatch 합 | 74; True; 11814; 0; 0 |  | WF 계획서 5.3 재현 점검 |

### 2.11 C2 좁힘에 쓰는 순위상관(사후)

| ID | 내용 | 원천 | 행 필터 | 열 | 값 | 판정어 | 판정 기록 |
|---|---|---|---|---|---|---|---|
| K1 | 재보정 이득과 재보정을 넘어선 ML 이득의 순위상관(30대상) | MS | `전체 30행` | Spearman(recal_gain, -ml_vs_p1) (p) | -0.08 (p 0.67) | 사후 | QA_FINAL_REVIEW Q1 C2 점검 |
| K2 | 재보정 이득과 P0 대비 최선 ML 이득의 순위상관(30대상) | MS | `전체 30행` | Spearman(recal_gain, -ml_best_delta) (p) | 0.66 (p 0.0001) | 사후 | RESEARCH_CLAIMS C2(WF0 0.66, 사후 서술) |

## 3. 뒷받침 실험

| 새 이름 | 옛 ID | 이 주장에서의 역할 | 실행 | 코드·설정 | 판정 기록 |
|---|---|---|---|---|---|
| W7_gain_decomposition_grid | WF9 지역 내(`--exp wf9`): 알래스카·레나·캐나다, 분할 1–25, n {200, 500, 1,000, 전량} | 주 근거(A1–A3, B, C1–C7, D, H) | Rescale Vppbeb(hematite 64코어). 3차 작업 단위 129개(meta `n_units`) 가운데 wf9 74개(5.3) | `scripts/3_deep_learning/h54_workflow.py`, `configs/rescale/wf3_full_hematite.yaml`, 등록 8180632, 구현 6d4c2dd | WF 계획서 7절, 5.3 |
| W7_gain_decomposition_grid(전이 arm) | WF9 전이(`--exp wf9x`): 주 4지역과 알래스카(모드 x), 분할 1–5, n {0, 10, 40, 160, 전량} | WF9-c(F), 전이 설명 비율(C8–C11, C14), 전이 SSE 몫(B8, B9) | 같은 Rescale 실행, wf9x 25단위(5.3) | 같음 | 5.3 |
| W7_gain_decomposition_grid(토양 민감도) | WF9 토양 √TDD 묶음(`--wf9-group soil`, wf9·wf9x 재적합) | 격자 정의 강건성(A4–A6, B10, B11, C13, E, G, H3, H4) | 로컬 CPU 워커 8 × 4스레드, 99단위(meta `n_units`) | h54 `--wf9-group soil`(커밋 7da0064), `tests/test_h54_workflow.py` 시험 (z) | 개정 이력 2026-10-02 10:51(등록), 11:32(결과) |
| W8_climate_extrapolation | WF10(`--exp wf10`): 알래스카·캐나다, warm·cold, 분할 1–10 | 판정 불가 근거(I1–I13) | 같은 Rescale 실행, wf10 30단위(5.3) | 같음 | 7절, 5.3 |
| W2_within_region_label_curve | WF6(와 WF1) | WF9 지역 내와 같은 라벨 추출. 총 저장소 재현 대조(J1, `scripts/2_evaluation/wf9_repro_check.py`) | Rescale elm | `configs/rescale/wf2_full.yaml` | 5.2 |

C8 장점 목록의 다른 항목은 해당 주장 폴더에서 근거를 찾는다.
- 손해 없는 선택: W1_label0_risk_table(WF0, 사후 서술) → `paper/claims/C1_label0_safety/`
- 계수 오차 진단: W5_method_selection_and_bias_diagnosis(WF4-c) → `paper/claims/C2_bias_diagnosis/`
- 위성 제품 결합: W6_stefan_cci_anchor(WF7, 서술). 이 폴더에는 수치를 옮기지 않았다.
- 관측 함의(격자·공변량 공간에 퍼뜨리기): W3_label_placement(WF2, WF8) → `paper/claims/C6_label_placement/`
- 이력: H3_transfer_fixes_H18-24 의 MODIS LST 강제력은 이득이 없었다는 서술이 QA Q8 에 있다. 원천 표는 이 폴더에서 열지 않았다.

## 4. 그림

- **현행 v2**: `outputs/figures/paper/v2/` 의 Fig1–Fig7, Table1 가운데 C8(WF9·WF10) 패널은 없다. `data/processed/paper_figs/v2_data_fig6_meta.json` 의 `not_opened` 항목에 `results/rescale_wf3/*(진행 중 실험)` 이 적혀 있어, v2 그림은 WF9·WF10 결과를 쓰지 않았다.
- **재구성안**(`docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md`): 4절의 결과 절 R7 '물리식과 총 오차가 비슷할 때의 장점'(C8)이 근거로 'WF9: 이득은 격자 단위 편향 보정, 격자 안 상세도 없음(설명 비율 0–6 %). WF10 판정 불가. WF7 위성 앵커(서술)' 를 들고, 그림은 'Fig 7(안)의 분해 패널 또는 SI' 로 둔다. 5절 Fig 7 행은 WF9·WF10 결과를 새 Fig 7 '충분 라벨 지역과 워크플로' 에 넣는 안이고, 대안은 C4·C8 을 SI 그림 하나로 두는 것이다. 배치 확정은 사용자 결정 사항이다(8절). R7 근거 문구의 '0–6 %' 와 '격자 안 상세도 없음' 은 1.2 의 2·3 에 따라 고친다.
- **패널 제안(미작성)**: 그림을 만들기 전에 그림 사양 파일을 먼저 쓴다.
  - (a) 대상별 P1 오차 제곱의 격자 안·격자 사이 비율 막대(B1–B3).
  - (b) R1 − P1 의 총·격자 사이·격자 안 대비와 두 CI(A1–A6, H5–H7), 기온·토양 정의를 나란히 둔다. RMSE 차는 저장소마다 정의가 달라 쌓지 않는다.
  - (c) SSE 이득의 격자 안·격자 사이 몫(B4–B11)을 쌓은 막대. 가산 분해는 SSE 기준에서만 성립한다. 총 이득이 음수인 행(B14)은 빼고 캡션에 적는다.
  - (d) 방법별 격자 안 설명 비율(C1–C14), 0 기준선과 8 % 참조선.
  - WF10 은 판정 불가이므로 본문 패널로 두지 않고 SI 표(I1–I13)로 둔다.
- **지도 표현**: QA Q8 권고대로 '고해상 지도' 대신 '1 km 셀로 표시한 기후 격자 단위 보정 지도' 로 적는다.

## 5. 단서와 쓰지 않을 문장

### 5.1 단서

1. **격자 묶음 정의의 이탈**: WF 계획서 7절의 등록 문구는 '√TDD(토양 도일)' 이나, 본 실행(Vppbeb)은 물리식이 쓰는 기온 √TDD(`e5_sqrt_tdd`)로 묶었다(개정 이력 2026-10-02 10:26 항목 1). 사본 `tables/rescale_wf3__wf3b_meta.json` 의 `implementation.wf9_grid` 설명에도 '토양 도일' 표기가 남아 있다. 두 정의의 묶음 일치도(조정 Rand 지수)는 알래스카 0.999, 캐나다 0.997, 레나 0.832 다(같은 기록).
2. **토양 정의의 근사**: 토양 정의에서는 물리식 예측이 묶음 안에서 근사로만 같아, P1 의 격자 안 RMSE 가 격자 안 실측 표준편차와 같지 않을 수 있다(개정 이력 10:51).
3. **RMSE 차는 더할 수 없다**: 총(A1 −0.52), 격자 사이(A2 −1.03), 격자 안(A3 −0.01)은 서로 다른 저장소의 RMSE 차다. 가산 분해는 SSE 기준(B4–B11)으로만 쓴다.
4. **두 표의 집계 차이**: decomp 표의 RMSE 열로 계산한 알래스카 R1 − P1 은 총 14.06 − 14.53, 격자 사이 6.76 − 7.68 이다(B12, B13). tests 의 짝 대비(A1, A2)와 값이 다르다. 두 표의 집계 방식이 달라 생기는 차이로 보이나 정확한 원인은 이 폴더에서 확인하지 않았다 [미확인]. 대비 수치는 tests 를 쓴다.
5. **SSE 몫의 불안정**: SSE 이득 몫은 총 이득이 작거나 음수이면 해석할 수 없다. 캐나다 R1(교차검증 λ)은 총 SSE 이득이 음수(B14)라 그 행의 격자 안 몫은 쓰지 않는다. 지역 내 라벨 전량의 R1 가운데 레나 R1(교차검증 λ, −22,058.61), 알래스카 R1(λ 1.0, −39,388.44), 레나 R1(λ 1.0, −45,838.06)도 총 SSE 이득이 음수라 몫을 쓰지 않는다(R-D·S-D `exp=wf9; method=R1; n=-1` 행 `gain_total_sse`). 예를 들어 레나 R1(교차검증 λ)의 격자 안 몫은 기온 정의 −9.8 %, 토양 정의 189.2 % 로 해석할 수 없다. 전이 라벨 전량의 러시아 E R1(λ 0.25)도 총 SSE 이득이 음수(−736.21, R-D `exp=wf9x`)다. decomp 표의 비율에는 CI 가 없다.
6. **고정 λ 0.25 의 격자 안 우세(H1–H4)**: 등록 가설 밖의 결과 열람 뒤 비교다. 판정 근거로 쓰지 않되 SI 에서 숨기지 않는다.
7. **직접 ML 과 위성 앵커**: 격자 안 3지역 층화 평균은 D0 +0.16 cm(H8), Re +0.31 cm(H9)로 열세였다(셀 가중). CCI 는 약 1 km 제품이지만(QA Q8 표) Stefan·CCI 평균 앵커는 격자 안 오차를 줄이지 않았다.
8. **전이의 CI 풀**: 레나·캐나다 2지역이다('지역 2/4', F6, G6). 러시아 W·E 는 채점 블록이 적어 CI 가 없고, 러시아 E 의 격자 안 점 추정은 +1.08 cm 다(F5, G5).
9. **WF9-a 의 부분 표기**: '부분(지역 2/3, 분할 23/24)' 의 '지역 2/3' 은 캐나다(|A| 315–441)에 n 500·1,000 이 없어서 생긴다(개정 이력 2026-10-01 14:32 항목 2). '분할 23/24' 는 레나 n 1,000 이 유효 분할 24개 가운데 23개에만 있어서 생긴다(R-T 의 n 1,000 층화 평균 행 `splits_short` 'Lena~w\|r 23/24', `splits_min` 23, `splits_expected` 24). 레나 분할 17 의 |A| 가 969 라(R-G `exp=wf9; target=Lena; split=17` 의 `n_A`) n < |A| 규칙(같은 항목 2)에 따라 n 1,000 이 빠졌다. 레나 분할 24 가 채점 블록 1개로 무효인 것(개정 이력 2026-10-01 14:50)은 레나의 기대 분할 수를 25에서 24로 줄인 원인이며 23/24 부족의 원인은 아니다.
10. **효과 크기**: 등록 가설의 층화 평균 대비는 열세·우세 모두 0.5 cm 미만이다(D6, E6, F6 판정 문구). 대상 행 가운데 0.5 cm 를 넘는 것은 다음과 같다. 레나 지역 내 라벨 전량은 토양 정의 +0.99 cm(E4, 열세), 기온 정의 블록 등가중 +0.87 cm [0.51, 1.24](D4, 셀 가중 +0.05, 미결정)다. 러시아 E 전이 격자 안 점 추정 +1.08 cm(F5, G5, CI 없음, 판정 불가)도 0.5 cm 를 넘는다.
11. **WF10**: 캐나다 warm 의 W 는 모든 분할에서 3블록(I11)이라 CI 조건(블록 합집합 8 이상, 5.3)을 채우지 못했고, 층화 평균의 풀 지역이 1개라 세 가설 모두 판정 불가다(I1–I3). 알래스카 cold 는 가장 추운 블록 하나가 대상 셀의 25 % 를 넘어 무효다(개정 이력 2026-10-02 00:20 항목 4). 알래스카 warm 행은 셀 가중과 블록 등가중의 부호가 다른 경우가 많다(I5, I6, I8–I10). claims 문서의 '직접 ML 의 외삽 손실 +3.4 – +7.0 cm' 는 셀 가중 점 추정이고 판정은 미결정이다. 알래스카 warm 분할 10 은 partial 단위다(I13, λ 교차검증 적합 1건 실패, 5.3).
12. **공변량 해상도**: 격자 안 설명력의 상한은 공변량 해상도와 관측 잡음이 정한다(WF 계획서 7절 해석 규칙). 격자 묶음 안에서 값이 달라지는 x25 공변량은 주로 DEM 변수다(7절 자료 서술: 묶음 안 x25 조합 수의 셀 가중 중앙값 알래스카 423, 레나 191, 캐나다 33. 토양 √TDD 묶음 기준, 개정 이력 2026-10-02 10:26 항목 1). 기후는 약 10 km, 토양은 약 5 km 로 추출했다(QA Q8 표).
13. **다른 양을 섞지 않는다**: 7절 자료 서술의 격자 안 분산 비율(알래스카 0.51)은 토양 열로 묶은 ALT 분산 비율이다. B1 의 0.72 는 기온 정의 묶음에서 P1 오차 제곱의 격자 안 비율이다.
14. **재현**: WF9 지역 내 총 저장소는 WF6 조각과 74/74 일치했다(공통 키 11,814개, 최대 |ΔSSE| 0, J1). 서로 다른 Rescale 노드(elm, hematite)에서 같은 값이 나왔다(5.3). 토양 민감도의 총 저장소는 Rescale 본 실행과 최대 |ΔSSE| 6.8e-6 cm² 차였다(개정 이력 11:32, 로컬과 Rescale 의 수치 라이브러리 차이).
15. **대상 수**: 지역 내 대상은 3개이고 알래스카가 중심이다. 독립 지역 수의 한계는 `paper/claims/D_data_and_design/` 에서 다룬다.

### 5.2 쓰지 않을 문장

| 쓰지 않을 문장 | 이유 | 대신 쓸 문장 |
|---|---|---|
| '물리 기반 ML 은 ERA5 격자 안의 공간 상세도를 더한다', '고해상도 ALT 지도' | WF9-a 열세가 있는 기각(D3, E1–E3, D6, E6), WF9-c 토양 정의 기각(G6), QA Q8 | 지역 내: '지역 내 층화 평균에서 ML(R1, 교차검증 λ)의 격자 안 예측은 오차를 오히려 늘렸다(0.5 cm 미만)'(WF 계획서 7절 해석 규칙, 개정 이력 11:32). 전이: '전이 조건에서도 ML 의 격자 안 예측이 실측 변동을 맞힌다는 근거는 확인되지 않았다'. 지도: '1 km 셀로 표시한 기후 격자 단위 보정 지도' |
| '전이 조건에서 ML 이 격자 안 변동의 일부를 맞혔다(레나 3.2 %)' | 5.3 해석 (1)의 문장이지만, 등록 해석 규칙(10:51)에 따라 토양 정의 민감도 뒤 더 약한 판정(기각)을 쓴다(11:32) | '전이 조건에서도 ML 의 격자 안 예측이 실측 변동을 맞힌다는 근거는 확인되지 않았다' |
| 'ML 은 격자 안 변동을 전혀 설명하지 못한다', '격자 안 상세도는 원리적으로 얻을 수 없다' | 서술 대비의 작은 우세(H1–H4), 최대 설명 비율 7.8 %(C12), 입력 해상도의 한계(5.1 의 12) | '알래스카에서 ML 이 설명한 격자 안 변동은 1 % 미만이었다(최대 0.7 %). 세 지역의 저장된 모든 ML 키에서도 8 % 미만이었다(최대 레나 7.8 %)' |
| '물리식과 비슷한 지역에서 이득의 99 % 가 격자 단위 보정이다' | 알래스카 지역 내 한정(B4 대 B6, B8, B9) | '알래스카 지역 내 라벨 전량에서 오차 제곱 감소의 약 99 %' |
| '격자 사이 −1.03 cm 와 격자 안 −0.01 cm 를 더하면 총 이득이다' | 저장소별 RMSE 차는 더할 수 없다(5.1 의 3) | SSE 몫(B4)으로 쓴다 |
| '학습 범위 밖(따뜻한) 기후에서 물리 잔차가 더 안전하다', '직접 ML 은 외삽에서 3.4–7.0 cm 손실을 본다' | WF10 판정 불가(I1–I3), 알래스카 행 미결정(I4–I10), 5.3 해석 (3) | '기후 외삽은 외삽 영역의 블록 수가 부족해 판정할 수 없었다' |
| '이런 지역에서 물리 기반 ML 은 손해가 없다', '안전하다' | 등록 비열등 시험 WF10-c 가 판정 불가(I3), C1 은 사후 서술 | C1 폴더의 서술 수치를 출처와 함께 인용 |
| 'ML 이득은 물리 계수 오차에 비례한다' | 재보정을 넘어선 ML 몫의 순위상관 −0.08(K1) | QA Q1 권고 문장(1.2 의 4) |
| '이 분해는 사전 등록된 확인적 결과다' | 격자 사이 서술은 사후 해석이고 WF9 전체가 결과 열람 뒤 설계 | '결과 열람 뒤 설계한 분석에서' |
| 'C8 은 격자 묶음 정의에 강건하다'(전체에 대해) | WF9-a 와 알래스카 분해만 두 정의에서 같고, WF9-c 는 정의에 따라 판정이 달랐다(F6 대 G6) | 'WF9-a 와 알래스카의 격자 사이 서술은 두 정의에서 같았다. WF9-c 는 정의에 따라 달라 더 약한 판정(기각)을 쓴다' |

## 6. 이 주장을 바꿀 수 있는 계획 실험(X*)

X 묶음은 2026-10-04 계획 단계다. 이 폴더 작성 시점(13:03)에는 `docs/` 에 X 묶음 계획서가 없었다. 그 뒤 2026-10-04 13:12 에 `docs/research/2026-10-04/harness_implementation_plan.md`('신규 실험 묶음(XA–XJ) 하네스 구현 계획')가 작성되었다. 이 문서는 구현 계획 초안이며 등록 계획서는 아니다. 아래 표는 그 문서를 반영하지 않은 C8 과의 관계와 등록 때 정할 것의 제안이다.

| 실험 | C8 에 미치는 영향 | 등록 때 정할 것(제안) |
|---|---|---|
| XE_hires_covariates | 가장 직접적인 시험이다. 격자 안 해상도 입력(토양 수분, 유기층, 식생·관목 피복, 적설 지속 기간, 미지형; QA Q8·Q9)으로 격자 안 설명 비율이 오르면 '격자 안 상세도 없음' 은 '현재 공변량(x25)에서' 로 한정되고, 결론이 '격자 안 상세도는 입력 해상도에 달려 있다' 로 바뀔 수 있다 | WF9 와 같은 분할·추출·seed·격자 묶음(기온·토양 두 정의)과 저장소(총, `~w`, `~b`)를 쓰고, 주 대비를 R1(x25 + 고해상 입력) − R1(x25)의 격자 안 RMSE 로 둔다. CatBoost CPU 한 플랫폼. 자료 약관과 시기 정합(라벨 1990–2024 대 기후 2015–2020, QA Q12)을 먼저 확인한다. `data/processed/covariates_ext_v1.csv`(git 미추적)의 내용은 이 폴더에서 열지 않았다 [미확인] |
| XI_climate_extrapolation_retest | WF10 의 '판정 불가' 를 판정으로 바꿀 수 있다. 결과에 따라 외삽의 물리 일관성을 장점으로 쓸지가 정해진다 | 외삽 영역 W 의 블록이 8개 이상인 두 번째 대상을 등록 전에 확인한다(claims 문서 6절). 대상별 `nb_eval_W` 를 등록 문서에 적는다. 알래스카만으로는 블록 사이 변동 때문에 미결정 가능성이 크다(`docs/RESEARCH_OVERVIEW_2026-10-02.md` 12절) |
| XB_multisource_stacking(QA Q11 의 X1) | 약 1 km 제품(CCI)과 여러 물리식을 대상 라벨로 가중하면 격자 안 정보가 들어올 수 있다. 고정 50:50 앵커(Re)는 격자 안에서 열세였다(H9) | WF9 저장소(총, `~w`, `~b`)를 함께 저장해 적층 이득의 격자 안·사이 몫을 같은 방식으로 보고한다. 이름 주의: LGX 의 X1(결합 방식, `docs/EXPERIMENT_PLAN_LG_2026-09-29.md`)과 다른 실험이다 |
| XG_product_comparison | 기존 ALT 제품(Ran 2022, Wei 2026, Liu Z. 2024; `docs/NOVELTY_POSITIONING_2026-09-29.md`, QA Q9)의 격자 안 설명 비율을 같은 채점 셀에서 재면 공개 자료로 얻을 수 있는 격자 안 설명력의 외부 기준이 생긴다 | 제품 학습 자료와 우리 채점 셀의 중복(누설)을 먼저 점검한다(QA Q11). 같은 격자 묶음으로 decomp 를 계산한다 |
| XF_new_regions | 지역 내 대상이 3개뿐이다. 레나처럼 격자 안 몫이 큰 대상(B6)이 늘면 C8 의 범위 문장이 바뀐다 | 새 대상마다 WF9 지역 내 arm 을 같은 규약으로 돌린다. 라벨 수와 블록 수의 하한을 등록 때 대조한다(WF10 의 설계 점검 누락을 되풀이하지 않는다) |
| XA_c2_gain_decomposition | C8 장점 목록의 C2 문장이 바뀐다(1.2 의 4). 재보정 몫과 ML 몫에 격자 사이·안 분해를 함께 보이면 재보정(지역 평균 편향)과 R1(격자 단위 편향)의 역할을 한 표로 보일 수 있다 | 기존 저장소의 재분석(새 적합 없음)으로 하고 사후 분석 표지를 단다 |
| XD_placement_policy | C8 의 관측 함의('같은 격자 안에 라벨을 더 모으는 것보다 격자와 공변량 공간에 라벨을 퍼뜨리는 것이 낫다', claims 문서 3절 C8)는 해석이며 직접 시험하지 않았다 | 격자당 라벨 상한을 둔 추출과 같은 격자에 모은 추출의 대비를 넣으면 이 함의를 시험할 수 있다 |
| XC_workflow_end_to_end | 워크플로 전 과정을 적용할 때 분해 저장소를 남기면 C8 을 워크플로 맥락에서 다시 확인할 수 있다 | 저장소 형식을 WF9 와 맞춘다 |
| XJ_tempderived_aux_labels | 지온 유도 보조 라벨이 새 격자에 라벨을 더하면 격자 사이 보정의 범위가 넓어진다. 라벨 정의 차이가 격자 단위 편향으로 들어갈 수 있다 | 정의 차이 보정을 격자 사이 성분과 분리해 보고한다 |
| XH_validation_ladder | C8 에 직접 영향은 없다 | 해당 없음 |

새 대비를 WF9 값과 견주려면 같은 CPU 결정적 경로(CatBoost·물리식)를 쓴다. Rescale 노드(elm, hematite) 사이는 같은 값이었고(J1, 5.3), 로컬과 Rescale 사이는 6.8e-6 cm² 차였다(개정 이력 11:32). 신경망 결과는 이 대비에 섞지 않는다.

### 6.1 X 묶음 결과 근거(2026-10-05 열람)

판정 정본은 `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 8절(추가 등록 XK·XL 은 `docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md` 결과 절)이다. 수치는 cm 이고 셀 가중 / 블록 등가중, 대괄호는 95 % CI 다. Δ 는 음수면 앞 방법의 오차가 작다. 이 표는 근거 색인이며 1절의 주장 문장은 고치지 않았다(원고 작업에서 정한다).

| 실험·가설 | 판정 | 수치 | 근거 표(원래 경로) | 이 주장과의 관계 |
|---|---|---|---|---|
| XI-c: 외삽 블록 R1(교차검증 λ) − P1, 캐나다 v4 warm_trim(계획 8.4) | 우세·비열등 충족, 지지(Holm p < 0.0001) | n 100 −2.44 [−2.83, −1.78] / −2.35 [−2.92, −1.79], 전량 −2.41 [−3.09, −2.03] / −2.88 [−3.40, −2.36] | `data/processed/xbatch/XI_climate_extrapolation_retest/sealed/xi_hyp.csv`, `xi_tests.csv` | 같은 시기의 따뜻한 블록을 쓴 공간 대용(외삽 폭 √TDD 차 0.97 (°C·일)^0.5)에서 재보정 앵커 잔차의 오차가 재보정 Stefan 보다 작았다 |
| XI-a·XI-b: 외삽 손실 EP(D0 − P1), EP(R1 − D0) | 미결정, 기각(기본 판 n 전량은 우세·열세로 달라 약한 쪽으로 씀) | XI-a 전량 −6.17 [−7.00, −3.02] / −1.12 [−3.36, +1.22], XI-b 전량 +3.99 [+1.90, +4.61] / +0.49 [−1.47, +2.38] | 같음 | 직접 ML 과 물리 잔차의 외삽 손실 차이를 확인하지 못했다 |
| XE-c(xh0): 격자 안 대비(계획 8.6) | R1(xh0) − R1(x25) 동등, R1(xh0) − P1 동등·미결정 | 격자 안 설명 비율(서술): 알래스카 0.07 → 0.69 %, 레나 −0.06 → 3.88 %, 캐나다 −2.18 → −2.15 % | `data/processed/xbatch/XE_hires_covariates/sealed/xe_r1b_xh0_tests.csv`, `xe_r1b_xh0_decomp.csv` | H0 10열로 격자 안 상세도가 생기지 않았다(보조). xt2·xh 판정 대기 |
| XK: 격자 크기별 P1 − R1(추가 등록, 서술) | 판정어 없음 | 알래스카 1 km 0.50, 0.05° 1.67, 0.1° 1.54, 0.25° 1.61(셀 가중) | `data/processed/xbatch/XK_support_scale/xk_support_scale_v1.csv` | 해석 규칙 2 는 두 가중이 다른 격자를 골라 갈래 문장을 쓰지 않고 사실만 기록했다 |
| XL: 지도 비교(추가 등록, 서술) | 판정 없음 | 0.1° 셀 평균이 1 km 지도 분산의 97 %(알래스카)·61 %(레나)를 설명 | `data/processed/xbatch/XL_map_products/xl_summary_v1.csv` | 지도 설명문: '1 km 지도는 표시 해상도이고 정보 해상도는 기후 입력(약 10 km)과 토양 입력(약 5 km)에 묶인다' |
| XB-6: 적층 가중(서술) | 판정어 없음 | 전량 평균 가중: 알래스카 Ss 0.82, 캐나다 Ku 0.57, 레나 P* 0.77, 티베트 Ca 0.92 | `data/processed/xbatch/XB_multisource_stacking/sealed/xb_weights_summary.csv` | 가중의 원인 문장은 쓰지 않는다(계획 2.2) |
| XE-e: 현장 VWC 상한 진단(계획 8.6, 진단(탐색)) | 판정을 바꾸지 않는다 | P1 격자 안 잔차 설명 비율(블록 교차검증): 알래스카 −3.54 %(1차 회귀)·−4.89 %(catboost_lo), 캐나다 −4.58 %·−4.48 %; 민감도 −6.59 에서 +2.55 %; 0.5 cm 등가 7.9 %·5.0 % | `data/processed/xbatch/XE_hires_covariates/sealed/xe_e_xe_e.csv` | 현장에서 잰 토양 수분은 격자 안 잔차를 설명하지 못했다 |

## 7. 파일

| 파일 | 내용 |
|---|---|
| `README.md` | 이 문서(2026-10-04 정정 반영, 8절) |
| `extract_evidence.py` | 사본에서 근거 수치를 읽어 `evidence_c8.csv` 와 `tables/MANIFEST.csv` 를 쓴다. 사본과 원본의 sha256 이 다르면 멈춘다 |
| `evidence_c8.csv` | 근거 표 83행(ID, 묶음, 원천, 행 필터, 열, 값, 판정어, 판정 기록) |
| `tables/MANIFEST.csv` | 사본 9개의 원본 경로, 사본 경로, sha256, 행 수(JSON 은 NA), 바이트 |
| `tables/*.csv`, `tables/*.json` | 집계 결과 표와 메타 사본(셀 단위 라벨 자료 없음, 각 1 MB 미만). `rescale_wf3__wf3b_targets.csv` 의 원본은 git 미추적이다 |

## 8. 정정 이력

- **2026-10-04(작성 뒤 점검)**: 아래 항목을 원천과 다시 대조해 이 문서만 고쳤다. 근거 사본(`tables/`)은 원본과 sha256 이 계속 일치해 `tables/MANIFEST.csv` 는 바꾸지 않았다. `extract_evidence.py` 와 `evidence_c8.csv` 는 다시 만들지 않았다.
  1. 1.1: claims 문서 C8 근거(85–90행)를 원문대로 옮기고, '0–6 %' 표기의 실제 출처 문서 네 곳을 적었다.
  2. 1.2 의 2, 1.3, 5.2 첫 행: WF9-a(열세가 있는 기각)의 문구를 등록 해석 규칙(WF 계획서 7절 163행, 5.3 해석 (1), 개정 이력 11:32)대로 '격자 안 예측은 오차를 오히려 늘렸다' 로 고쳤다. '근거는 확인되지 않았다' 형식은 전이(WF9-c 토양 정의)에만 쓴다.
  3. 1.2 의 3: '0–6 %' 는 판정 기록 키의 범위(−8.5 % – 6.2 %)가 아니라 음수를 0 으로 본 요약임을 적었다.
  4. 1.3: R1 을 '교차검증 λ' 로 밝히고, '8 % 미만' 을 알래스카(1 % 미만, 최대 0.7 %)와 세 지역 전체(8 % 미만, 최대 7.8 %)로 나눴다. 영문 'identical' 을 'the same to two decimal places' 로 고쳤다.
  5. 2 읽는 법: 판정 규칙에 우세·열세가 동등보다 앞선다는 순서와 비유한 끝값의 판정 불가를 더했다.
  6. 2.3: 제목의 정의와 n 범위를 고치고, 알래스카 최댓값, 라벨 전량 최댓값, 등록 정의(SSE 비)와의 차이를 단서로 더했다.
  7. 2.9 I5: 판정 기록 열을 'WF 계획서 5.3 WF10 행' 으로 고쳤다.
  8. 5.1 의 5·9·10·12: 총 SSE 이득이 음수인 행 추가, '분할 23/24' 의 원인(레나 분할 17 |A| 969), 0.5 cm 를 넘는 대상 행(D4, F5·G5 추가), 조합 수의 토양 묶음 표지를 고쳤다.
  9. 6절: 이 문서 작성 뒤 생긴 `docs/research/2026-10-04/harness_implementation_plan.md`(13:12, 등록 전 구현 계획)를 적었다.
- **남은 불일치**: `evidence_c8.csv` 의 I5 행 `verdict_doc` 열과 `extract_evidence.py` 의 I5 인자는 아직 '판정 문서 없음(이 폴더에서 처음 인용, 서술)' 이고, 스크립트의 C 묶음 주석은 '1 − MSE_w(M)/MSE_w(P1)' 이다. 값(value 열)은 바뀌지 않았다. 두 파일을 다시 만들 때 이 문서의 2.3 제목과 2.9 I5 판정 기록을 따른다.
