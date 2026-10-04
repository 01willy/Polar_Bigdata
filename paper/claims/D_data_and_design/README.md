# D. 자료와 평가 설계

**성격**: 논문 색인 폴더 `paper/claims/` 의 근거 묶음이다. 자료원, 표의 판(v1–v4), 지역·블록·셀, 라벨 수 격자, A/B 절반, 100 km 버퍼, 두 물리 기준선(P0, P1)과 P*, 오차 하한, 원천의 알래스카 편중을 한곳에 모은다. 기존 파일은 옮기거나 고치지 않았다. 원천 표의 사본은 `tables/` 에 있고 해시는 `tables/MANIFEST.csv` 에 있다.
**작성** 2026-10-04. **수정** 2026-10-04(검증 지적 11건 반영, 8절 수정 기록).
**읽은 문서**: `docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md`(2절, 6절), `docs/QA_FINAL_REVIEW_2026-10-02.md`(Q1, Q7, Q8, Q9, Q12, Q15, 리뷰어 평가), `docs/EXPERIMENT_PLAN_LG_2026-09-29.md`(§1–§3, 6A.3 N2, 6B.3–6B.6, 7.1a, 7.2), `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md`(1.1–1.3, 2.1–2.5, 결과 전 계산, J7), `docs/EXPERIMENT_PLAN_WF_2026-10-01.md`(머리말, 1절), `docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md`(Methods), `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md`, `docs/RESEARCH_OVERVIEW_2026-10-02.md`(4절, 10절), `docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md`(§2, §9). 수정 때 더 읽은 것: LG 계획서 6B.1, 7.1a, 7.3, 개정 15 (p); WRAPUP 1.2, 3.3, 9.2, 9.4, 10절, J7; `paper/claims/C2_bias_diagnosis/README.md` 2.3; `paper/registry/experiments.csv` X 행; `docs/EXPERIMENT_LOG.md` 155행.
**수치 규칙**: 2절 근거 표의 값은 모두 원천 파일을 pandas 로 읽어(`OMP_NUM_THREADS=1`, 읽기만) 표에 적은 행 필터와 열에서 꺼냈다. 문서 산문에서 옮긴 수치는 없다. 값은 소수 2자리로 반올림했고 부호는 파일 그대로다(음수 Δ 는 앞 방법의 오차가 작다는 뜻이다). 이 폴더에서 새로 계산한 값은 '이번 계산'으로 표시했고 판정이 아니다.

---

## 1. 주장 문장

### 1.1 현행 문구

`docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md` 에는 자료·설계에 대한 번호 붙은 주장이 없다. 이 폴더의 주장에 해당하는 현행 문구는 2절 표와 끝 문장, 6절 한계 항목이다.

- 2절 표 '평가 지역': "지역을 통째로 빼고 100 km 완충(전이), 그리고 지역 안 블록 홀드아웃(충분 라벨)".
- 2절 표 '라벨 수': "0개부터 수천 개까지, 모든 방법이 같은 라벨을 쓴다".
- 2절 표 '물리 비교 상대': "원천 계수와 같은 라벨로 재보정한 계수, 현지 최소제곱, 계수 공간 보간, 군집 계수, 더 강한 물리 기준선 11종".
- 2절 끝: "방법 하나하나는 새롭지 않다. 새로운 것은 물리 기반 ML 의 능력을 라벨 수, 물리 계수의 오차, 관측 위치에 따라 같은 시험지에서 잰 평가 설계와, 그 결과로 만든 워크플로다."
- 6절 '원천의 편중': "원천은 알래스카 셀이 78–94 % 다. 확인적 비교는 주 4지역 중심이다."
- 6절 '새 지역': "약관이 확인된 새 얕은 지역은 러시아 중부 하나다."

### 1.2 QA 점검(2026-10-02)이 요구한 축소

`docs/QA_FINAL_REVIEW_2026-10-02.md` 의 지적 가운데 이 폴더의 문장에 걸리는 것은 네 가지다.

1. **C2 축소와 두 기준선의 역할**: QA 는 'ML 이득이 계수 오차에 비례한다'의 근거(WF0 순위상관 0.66, WF4-c ρ 0.38)가 모두 P0 대비 이득이고, R1 이 재보정(P1)을 포함하므로 상관의 상당 부분이 재보정 몫이라고 지적했다. 재보정을 넘어선 ML 몫(P1 − 최선 ML, 양수가 ML 이득)과 계수 오차(재보정 이득 P0 − P1)의 순위상관은 −0.08(p 0.67, 30대상)이다. QA 문서는 이 괄호를 '최선 ML − P1'로 적었으나, 그 차를 그대로 쓰면 부호가 반대(+0.08)다(2절 '이번 계산'). C2 폴더(`paper/claims/C2_bias_diagnosis/README.md` 2.3)는 −0.08 을 재현했고, 계수 오차와 ML 몫의 정의에 따라 순위상관이 −0.08 에서 0.39(라벨 10개 진단값 대 P1 − R1(λ 0.25), 해석적 p 0.04)까지 달라진다고 정리했다. 따라서 이 관계는 '상관이 없었다'가 아니라 '확인하지 못했다'로 쓴다. 설계 문장에 주는 함의는 그대로 하나다. P0 대비 Δ 는 '라벨을 쓰는 보정 전체'의 이득이고, ML 의 순가치는 같은 라벨로 재보정한 P1 대비 Δ 로만 쓴다.
2. **'안전' 표현**: 등록된 비열등 검정(WRAPUP 3.3 SC1w, 두 CI 상한 < +0.5 cm)을 통과한 경우가 아니면 '안전'을 쓰지 않는다. 설계 절에서는 '안전한 평가', '보수적이라 안전' 같은 표현을 쓰지 않는다.
3. **WF 의 사후 설계 표지**: 지역 내 시험(충분 라벨) 설계는 LG·LGX·LGD·LGT·LGU·LGW 결과를 연 뒤 등록했다(WF 계획서 머리말). WF 표의 `design_note` 열이 모두 '결과 열람 뒤 설계'다(근거 E48). 설계 절에서 지역 홀드아웃(사전 등록)과 지역 내 블록 홀드아웃(결과 열람 뒤 설계)을 구분해 적는다.
4. **일반화 범위**: 확인적 독립 지역은 4–5개이고 원천의 78–94 % 가 알래스카 셀이다(근거 E35). QA·연구 개요 문서는 '4–5'의 구성을 적지 않았다. 이 폴더는 이를 주 4지역(주 판정 4개)과 약관 확인 새 얕은 지역 러시아 C 확충판을 더한 PE1(5개)로 정의한다(LG 7.3 '주 판정 판', 연구 주장 문서 6절 '새 얕은 지역은 러시아 중부 하나'). 등록 문서의 '독립 지역'은 더 넓다. WRAPUP 1.2 는 주 4지역, Alaska(x)(참조), LGD 적격 새 지역(NAtlantic, Russia_C, Tibet)을 모두 독립 지역으로 두고, WRAPUP 3.3 SC1w 의 '독립 지역 5곳'은 주 4지역과 Alaska(x)다. LG 7.3 의 결론 문구는 '확장 풀(지역 6개)'(PE2 = PE1 + 티베트)을 쓴다. 문장마다 어느 묶음인지 밝혀 적는다. 알래스카도 13,606행이 1 km 위치 343곳, 0.5° 블록 74개에 몰려 있어 공간 일반화에 쓰이는 독립 단위는 74–343 수준이다(QA Q1, 근거 E10).

### 1.3 이 폴더의 주장(권고안)

**국문**: 모든 방법은 같은 시험지에서 비교했다. 대상 지역 전체와 그 주변 100 km 를 학습 원천에서 빼고, 대상의 0.5° 블록을 A·B 절반으로 나눠(분할 5회) A 에서 라벨 n 개(0, 3, 10, 40, 160, 320, 1,000, 전량)를 뽑고 B 의 채점 셀에서 오차를 잰다. 기준선은 원천 계수 Stefan 식(P0)과, 같은 n 개 라벨로 계수를 원천 계수 쪽으로 κ = 10 수축 재보정한 Stefan 식(P1)이다. 라벨 0 과 전량에서는 P0 보다 강한 것으로 판정된 연도 정합 도일 Stefan 식(P* = P0@tddm)을 병기하고(전량 R1 − P* −1.64 [−2.95, −0.74], E51), n = 40·160 에서는 더 강한 재보정 기준선 P1* = P1@ed(토양 물성 Stefan 앵커의 κ = 10 재보정)를 병기한다. R1 − P1* 는 n = 40·160 에서 미결정이므로(E52) 이 n 에서는 '재보정 물리식을 넘는다'를 쓰지 않는다. 라벨 n 은 행 수이고 행의 뜻은 지역마다 다르다(점 관측, CALM 지점 다년 평균, 1 km 셀 평균). 확인적 독립 지역은 주 4지역(재사용 지역의 재검정)과 약관이 확인된 새 얕은 지역 1곳(러시아 C 확충판)이다(4–5개). 심부 레짐인 티베트는 확장 풀 PE2 에만 들어간다. Alaska(x)는 독립 참조 대상이며 주 4지역 평균에 넣지 않는다. 원천 셀의 78–94 % 가 알래스카다. 효과 크기는 공변량 조건부 오차 하한(알래스카 11.31, 레나 14.83, 캐나다 17.90 cm)과 함께 보고한다. 지역 내 블록 홀드아웃(충분 라벨) 시험은 전이 결과를 본 뒤 설계했다.

**영문(원고용 초안)**: All methods were compared on one test protocol. Each target region and a 100 km buffer were removed from the source data; the target's 0.5° blocks were split into halves A and B five times; n labels (0, 3, 10, 40, 160, 320, 1,000 or all) were drawn from A and errors were scored on B. Every method is reported against two Stefan baselines: the source coefficient (P0) and the coefficient recalibrated with the same n labels by shrinkage towards the source coefficient (κ = 10; P1). At n = 0 and with all labels, the year-matched degree-day Stefan model (P*), which outperformed P0, is reported as well. At n = 40 and 160, the stronger recalibrated baseline P1* (the edaphic Stefan anchor recalibrated with κ = 10) is also reported, and the residual model is not described as outperforming recalibrated physics at these n. The label count n is a row count, and a row is a point observation, a multi-year CALM site mean or a 1 km cell mean depending on the region. Confirmatory independent targets are the four re-used main regions and one licence-verified new shallow region (central Russia, expanded edition); the deep-regime Tibetan Plateau enters only the extended pool, and Alaska is an independent reference target that is not averaged with the main regions. Of the source cells, 78–94 % are Alaskan. Effect sizes are reported together with a covariate-conditional error floor. The within-region block-holdout tests were designed after the transfer results had been seen.

---

## 2. 근거 표

| id | 항목 | 원천 경로 | 행 필터 | 열 | 값(소수 2자리) | 판정어 | 판정 근거 절 |
|---|---|---|---|---|---|---|---|
| E01 | v1 표(`fidelity_base.csv`) 행 수 | `data/processed/fidelity_base_v2_meta.json` | 최상위 키 | `n_old` | 17,423 | 서술 | meta `plan`: `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` §3 |
| E02 | v2 표 행 수, 추가 행, 중복 제거 | `data/processed/fidelity_base_v2_meta.json` | 최상위 키 | `n_total / n_new / dedup_dropped` | 17,572 / 149 / 8 | 서술 | 같은 문서 §3(E3-2 단계) |
| E03 | v3 표 행 수, 출처 표지 변경 행, 지역 변경 행 | `data/processed/fidelity_base_v3_meta.json` | 최상위 키 | `n_rows / n_source_changed / n_region_changed` | 17,572 / 68 / 3 | 서술 | meta `plan`: `docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md` §3 S-A |
| E04 | v3 직접 라벨(F4_direct) 셀 합계(macro_table 합) | `data/processed/fidelity_base_v3_meta.json` | `macro_table` 중 source_id = F4_direct 행의 합 | `n_cells` | 17,467 | 서술 | 같은 문서 §3 S-A |
| E05 | v3 직접 라벨 셀(그림 1 기준) | `data/processed/paper_figs/fig1_meta.csv` | 유일 행 | `n_v3_cells` | 17,467 | 서술 | `docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md` Observational data |
| E06 | v4 표 행 수 = v3 + 새 행 | `data/processed/fidelity_base_v4_meta.json` | 키 `n_rows` | `v3 / new / total` | 17,572 / 516 / 18,088 | 서술 | `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 6B.6 |
| E07 | v4 새 행의 라벨 등급 | `data/processed/fidelity_base_v4_meta.json` | 키 `n_rows.new_by_source_id` | `F4_ext_direct / F3_ext_temp / F2_ext_unknown` | 329 / 39 / 148 | 서술 | LG 계획서 6B.3(라벨 정의 규칙) |
| E08 | v3 F4_direct 안의 방법 표지(PANGAEA 972777 이벤트 좌표 대조. 대조 범위는 region 'Canada'·'United States (Alaska)' 행뿐이고, 대조된 행은 모두 F4_direct 다) | `data/processed/fidelity_base_v4_meta.json` | 키 `v3_method_check.counts` | `United States (Alaska)\|temperature / Canada\|temperature / United States (Alaska)\|thaw_tube / Canada\|thaw_tube / Canada\|ambiguous:probe\|thaw_tube` | 15 / 4 / 1 / 1 / 7 | 서술(본 실행에서 F4_direct 유지) | LG 계획서 6B.3 'v3 라벨 표시의 불일치', 개정 15 (p)(3) |
| E09 | 확장 공변량표 행 수와 군별 열 수(LST·T·W·V·H) | `data/processed/covariates_ext_v1_meta.json` | 키 `n`, `groups` | `n / len(groups[*])` | 17,572 / LST 6·T 6·W 3·V 5·H 10 | 서술 | `docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md` §2.1 |
| E10 | Alaska (x): 라벨 행 / 1 km 위치 / 0.5° 블록 | `data/processed/paper_figs/table1_rows.csv` | target = Alaska | `label_rows / loc_1km / blocks` | 13,606 / 343 / 74 | 서술 | Table 1(`outputs/figures/paper/v2/Table1_data.csv`) |
| E11 | Lena: 라벨 행 / 1 km 위치 / 0.5° 블록 | `data/processed/paper_figs/table1_rows.csv` | target = Lena | `label_rows / loc_1km / blocks` | 3,037 / 201 / 20 | 서술 | Table 1(`outputs/figures/paper/v2/Table1_data.csv`) |
| E12 | Canada: 라벨 행 / 1 km 위치 / 0.5° 블록 | `data/processed/paper_figs/table1_rows.csv` | target = Canada | `label_rows / loc_1km / blocks` | 750 / 86 / 39 | 서술 | Table 1(`outputs/figures/paper/v2/Table1_data.csv`) |
| E13 | Russia W: 라벨 행 / 1 km 위치 / 0.5° 블록 | `data/processed/paper_figs/table1_rows.csv` | target = Russia_W | `label_rows / loc_1km / blocks` | 31 / 29 / 21 | 서술 | Table 1(`outputs/figures/paper/v2/Table1_data.csv`) |
| E14 | Russia E: 라벨 행 / 1 km 위치 / 0.5° 블록 | `data/processed/paper_figs/table1_rows.csv` | target = Russia_E | `label_rows / loc_1km / blocks` | 30 / 30 / 21 | 서술 | Table 1(`outputs/figures/paper/v2/Table1_data.csv`) |
| E15 | Russia C: 라벨 행 / 1 km 위치 / 0.5° 블록 | `data/processed/paper_figs/table1_rows.csv` | target = Russia_C | `label_rows / loc_1km / blocks` | 7 / 7 / 6 | 서술 | Table 1(`outputs/figures/paper/v2/Table1_data.csv`) |
| E16 | Greenland: 라벨 행 / 1 km 위치 / 0.5° 블록 | `data/processed/paper_figs/table1_rows.csv` | target = Greenland | `label_rows / loc_1km / blocks` | 3 / 2 / 2 | 서술 | Table 1(`outputs/figures/paper/v2/Table1_data.csv`) |
| E17 | Russia C (expanded)(LGD): 라벨 행 / 1 km 위치 / 블록, 판 | `data/processed/paper_figs/table1_rows.csv` | target = Russia_C_LGD | `label_rows / loc_1km / blocks / edition` | 57 / 57 / 13 / licence-verified (main) | 서술(적격은 LG 6B.4 규칙) | LG 계획서 6B.4, 7.3 |
| E18 | Tibet(LGD): 라벨 행 / 1 km 위치 / 블록, 판 | `data/processed/paper_figs/table1_rows.csv` | target = Tibet_LGD | `label_rows / loc_1km / blocks / edition` | 132 / 132 / 34 / licence-verified (main) | 서술(적격은 LG 6B.4 규칙) | LG 계획서 6B.4, 7.3 |
| E19 | North Atlantic(LGD): 라벨 행 / 1 km 위치 / 블록, 판 | `data/processed/paper_figs/table1_rows.csv` | target = NAtlantic | `label_rows / loc_1km / blocks / edition` | 41 / 결측 / 13 / full edition only | 서술(적격은 LG 6B.4 규칙) | LG 계획서 6B.4, 7.3 |
| E20 | 약관 미확인으로 그림·주 판정에서 뺀 셀 | `data/processed/paper_figs/fig1_not_drawn.csv` | region = Canada, NAtlantic, Russia_W | `n_cells_not_drawn` | Canada 38 / NAtlantic 19 / Russia_W 1 | 서술 | Table 1 각주 e, LG 계획서 개정 15 (m) |
| E21 | Alaska: ERA5 기후값 조합 수 / 1 km 위치당 행 / 1 km 위치 안 ALT SD(위치 수) | `data/processed/lgw/lgw_label_units.csv` | target = Alaska, kind = macro | `n_era5 / rows_per_loc / sd_in_1km_cm (n_loc_sd)` | 164 / 39.67 / 11.35 (178) | 서술(결과 전 계산) | `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 2.2 |
| E22 | Lena: ERA5 기후값 조합 수 / 1 km 위치당 행 / 1 km 위치 안 ALT SD(위치 수) | `data/processed/lgw/lgw_label_units.csv` | target = Lena, kind = macro | `n_era5 / rows_per_loc / sd_in_1km_cm (n_loc_sd)` | 52 / 15.11 / 12.98 (82) | 서술(결과 전 계산) | `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 2.2 |
| E23 | Canada: ERA5 기후값 조합 수 / 1 km 위치당 행 / 1 km 위치 안 ALT SD(위치 수) | `data/processed/lgw/lgw_label_units.csv` | target = Canada, kind = macro | `n_era5 / rows_per_loc / sd_in_1km_cm (n_loc_sd)` | 56 / 8.72 / 17.70 (45) | 서술(결과 전 계산) | `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 2.2 |
| E24 | 라벨 수 격자(−1 = 전량) | `results/rescale_lg/data/processed/lg/lg_meta.json` | 키 | `n_grid` | 0, 3, 10, 40, 160, 320, 1000, -1 | 서술(사전 등록) | LG 계획서 §2 |
| E25 | 대상 수 / 분할 / 추출 / 학습기 seed / 방법 수 | `results/rescale_lg/data/processed/lg/lg_meta.json` | 키 | `len(targets) / splits / draws / seeds / len(methods)` | 27 / 5 / 5 / 2 / 12 | 서술(사전 등록) | LG 계획서 §1–§3 |
| E26 | LG 본 실행 적합 수 | `results/rescale_lg/data/processed/lg/lg_meta.json` | 키 | `n_fit_total` | 129,887 | 서술 | LG 계획서 §6.2 |
| E27 | Lena(x): 유효 분할 / A 셀 / 채점 셀 / 채점 블록(유효 분할 범위) | `results/rescale_lg/data/processed/lg/lg_targets.csv` | part = cpu, target = Lena, mode = x, valid = True | `split(고유 수) / n_A / n_eval / nb_eval` | 5 / 1120–1737 / 1278–1860 / 5–16 | 서술 | LG 계획서 §2, Table 1 |
| E28 | Canada(x): 유효 분할 / A 셀 / 채점 셀 / 채점 블록(유효 분할 범위) | `results/rescale_lg/data/processed/lg/lg_targets.csv` | part = cpu, target = Canada, mode = x, valid = True | `split(고유 수) / n_A / n_eval / nb_eval` | 5 / 325–406 / 342–423 / 12–25 | 서술 | LG 계획서 §2, Table 1 |
| E29 | Russia_W(x): 유효 분할 / A 셀 / 채점 셀 / 채점 블록(유효 분할 범위) | `results/rescale_lg/data/processed/lg/lg_targets.csv` | part = cpu, target = Russia_W, mode = x, valid = True | `split(고유 수) / n_A / n_eval / nb_eval` | 5 / 14–16 / 12–17 / 7–12 | 서술 | LG 계획서 §2, Table 1 |
| E30 | Russia_E(x): 유효 분할 / A 셀 / 채점 셀 / 채점 블록(유효 분할 범위) | `results/rescale_lg/data/processed/lg/lg_targets.csv` | part = cpu, target = Russia_E, mode = x, valid = True | `split(고유 수) / n_A / n_eval / nb_eval` | 5 / 15–17 / 11–14 / 8–11 | 서술 | LG 계획서 §2, Table 1 |
| E31 | Alaska(x): 유효 분할 / A 셀 / 채점 셀 / 채점 블록(유효 분할 범위) | `results/rescale_lg/data/processed/lg/lg_targets.csv` | part = cpu, target = Alaska, mode = x, valid = True | `split(고유 수) / n_A / n_eval / nb_eval` | 5 / 6322–7436 / 6170–7284 / 21–63 | 서술 | LG 계획서 §2, Table 1 |
| E32 | Greenland(x): 유효 분할 / A 셀 / 채점 셀 / 채점 블록(유효 분할 범위) | `results/rescale_lg/data/processed/lg/lg_targets.csv` | part = cpu, target = Greenland, mode = x, valid = True | `split(고유 수) / n_A / n_eval / nb_eval` | 0 / 해당 없음(유효 분할 없음) | 서술 | LG 계획서 §2, Table 1 |
| E33 | 100 km 버퍼로 원천에서 뺀 셀 | `data/processed/paper_figs/fig1_source.csv` | target = Lena, Canada, Russia_W, Russia_E, Russia_C, Greenland, Alaska | `n_buffer_excluded` | Lena 1 / Canada 20 / Russia_W 0 / Russia_E 0 / Russia_C 1252 / Greenland 0 / Alaska 1 | 서술 | LG 계획서 §1, 6B.6 |
| E34 | 원천 셀 수 | `data/processed/paper_figs/fig1_source.csv` | 같음 | `n_src` | Lena 14,429 / Canada 16,697 / Russia_W 17,436 / Russia_E 17,437 / Russia_C 16,208 / Greenland 17,464 / Alaska 3,860 | 서술 | LG 계획서 §1 |
| E35 | 원천 가운데 알래스카 셀 비율(주 6지역 대상) | `data/processed/paper_figs/fig1_source.csv` | target = Lena, Canada, Russia_W, Russia_E, Russia_C, Greenland | `share_alaska` | Lena 0.94 / Canada 0.81 / Russia_W 0.78 / Russia_E 0.78 / Russia_C 0.84 / Greenland 0.78 (최소 0.78, 최대 0.94) | 서술 | `docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md` 6절 |
| E36 | Alaska(x) 대상의 원천 구성 | `data/processed/paper_figs/fig1_source.csv` | target = Alaska | `share_lena / share_canada / share_other` | 0.79 / 0.19 / 0.02 | 서술 | `docs/RESEARCH_OVERVIEW_2026-10-02.md` 5절 C4 |
| E37 | 원천 계수 E0(cm (°C·d)^−1/2) | `data/processed/paper_figs/fig1_source.csv` | target = Lena, Canada, Russia_W, Russia_E, Russia_C, Greenland, Alaska | `E0` | Lena 1.62 / Canada 1.59 / Russia_W 1.59 / Russia_E 1.59 / Russia_C 1.60 / Greenland 1.59 / Alaska 1.50 | 서술 | LG 계획서 §3, 식 E0 = Σsy/Σs² |
| E38 | 지역 자체의 E 기하 평균(셀 E = ALT/√TDD 의 기하 평균 exp(mean ln(ALT/√TDD)), `scripts/4_visualization/paper/v2_data.py` 649행) | `data/processed/paper_figs/fig1_z_summary.csv` | row = Lena, Canada, Russia_W, Russia_E, Russia_C, Greenland, Alaska, Russia_C_LGD, Tibet_LGD | `E_mean` | Lena 1.31 / Canada 1.41 / Russia_W 2.51 / Russia_E 1.73 / Russia_C 1.81 / Greenland 4.25 / Alaska 1.62 / Russia_C_LGD 1.80 / Tibet_LGD 11.03 | 서술 | 그림 1b |
| E39 | P0 RMSE, 라벨 0(셀 가중, cm) | `results/rescale_lg/data/processed/lg/lg_curve.csv` | axis = method, method = P0, n = 0, mode = x, target = Lena, Canada, Russia_W, Russia_E, Alaska | `rmse` | Lena 21.69 / Canada 28.16 / Russia_W 42.98 / Russia_E 29.72 / Alaska 14.54 | 서술(기준선 값) | LG 계획서 §3 P0 |
| E40 | P0 RMSE, 라벨 0(블록 등가중, cm) | `results/rescale_lg/data/processed/lg/lg_curve.csv` | 같음 | `rmse_beq` | Lena 23.81 / Canada 21.77 / Russia_W 38.43 / Russia_E 12.45 / Alaska 14.09 | 서술(기준선 값) | LG 계획서 §3 P0 |
| E41 | P*(연도 정합 도일 Stefan) − P0, 라벨 0, 주 4지역 층화 평균 | `data/processed/lgx/lgx_tests.csv` | test_id = L29, contrast = P0@tddm-P0\|n0, scope = MEAN | `delta [ci_lo, ci_hi]; delta_blockeq [ci_lo_beq, ci_hi_beq]; holm_p` | -1.00 [-1.35, -0.32]; -0.41 [-0.74, -0.08]; 0.01(원값 0.0056) | 우세 | LG 계획서 7.2 L29, 7.1a(P* 병기) |
| E42 | P* − P0, 라벨 0, 3지역 보조 평균(레나·캐나다·Alaska) / Alaska(x) | `data/processed/lgx/lgx_tests.csv` | test_id = L29, contrast = P0@tddm-P0\|n0, scope = MEAN3 / target = Alaska\|x | `delta [ci_lo, ci_hi]` | -0.25 [-0.94, 0.50] / 1.65 [0.30, 2.08] | 미결정 / 미결정 | LG 계획서 7.2 L29 |
| E43 | P1@tddm − P1, 주 4지역 평균, 라벨 10 / 전량 | `data/processed/lgx/lgx_tests.csv` | test_id = L29, contrast = P1@tddm-P1\|n10 / P1@tddm-P1\|n-1, scope = MEAN | `delta [ci_lo, ci_hi]` | -0.73 [-0.98, -0.27] / -0.55 [-0.78, -0.17] | 미결정 / 미결정 | LG 계획서 7.1a(P1* 규칙), 7.2 L29 |
| E44 | 오차 하한 RMSE(cm) | `data/processed/lgx/lgx_floor.csv` | scope = eval, region = Alaska, Lena, Canada | `floor_rmse_cm` | Alaska 11.31 / Lena 14.83 / Canada 17.90 | 서술(보고 규칙 N2, 판정 없음) | LG 계획서 6A.3 N2, 6A.5a 효과 크기 표기 |
| E45 | 오차 하한 계산에 쓴 셀 / 묶음 | `data/processed/lgx/lgx_floor.csv` | 같음 | `n_cells_used / n_bundles_used` | Alaska 13,333, 187 / Lena 2,766, 76 / Canada 673, 49 | 서술 | LG 계획서 6A.3 N2 |
| E46 | S-a: 라벨 단위(1 km 위치 평균 − 점)가 P1 − P0 에 준 차, n = 10, 분할 분포 중앙값 [p10, p90] | `data/processed/lgw/lgw_sens_unit.csv` | quantity = unit_p1, n = 10, target = Canada, Lena, Alaska | `p50 [p10, p90]` | Canada -0.86 [-3.48, 1.05] / Lena 0.34 [-0.26, 1.99] / Alaska 0.09 [-0.36, 0.58] | 서술(해석 규칙 적용: 캐나다 문장) | WRAPUP 결과 전 계산 2.4 (S-a), J7 표 |
| E47 | S-b: 단년 라벨 − 다년 평균 라벨의 P1 − P0 차, n = 10, 중앙값 | `data/processed/lgw/lgw_sens_year.csv` | n = 10, target = Russia_W, Russia_E | `p50` | Russia_W 0.05 / Russia_E 0.04 | 서술(해석 규칙 적용: 이득 유지) | WRAPUP 결과 전 계산 2.4 (S-b), J7 표 |
| E48 | 지역 내 시험 WF6 의 기대 분할 수, 설계 표지 | `results/rescale_wf2/data/processed/wf/wf2b_tests.csv` | test_id = WF6-a, target = Alaska\|r, Canada\|r, Lena\|r | `max(n_splits_expected) / design_note` | Alaska\|r 25 / Canada\|r 25 / Lena\|r 24; '결과 열람 뒤 설계' | 표지(결과 열람 뒤 설계) | `docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 머리말 '성격', 6절 WF6, 5.2 |
| E49 | H19b: 확장 공변량 A군(T·W·V·H) 추가 − x25, 공변량만 조건, 주 4지역 평균(M1 규약) | `data/processed/h2/h_tests_all.csv` | test = H19p_x25_A_vs_x25_pseudo, cond = covonly, is_mean = True | `delta [ci_lo, ci_hi]; delta_blockeq [ci_lo_beq, ci_hi_beq]; p_holm` | 0.64 [0.05, 1.12]; 0.55 [-0.02, 1.24]; 0.17(원값 0.168) | 기각(이득 없음. 셀 가중 +0.64 [0.05, 1.12], 블록 등가중 +0.55 [−0.02, 1.24], Holm p 0.17) | `docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md` §9 결과 요약 H19b(원문 판정어 '기각(추가 시 악화)') |
| E50 | v3 F4_direct 안의 티베트 지온 유도 셀(QTP_CN, E08 의 대조 범위 밖) | `data/processed/fidelity_base_v3_meta.json` / `data/processed/fidelity_base_v4_labels.csv` | `macro_table` 중 macro = Tibet, source_id = F4_direct / loc_id = 17389 | `n_cells` / `source_id, lgd_role` | 1 / F4_direct, 'v3_f4_temp_derived(Tibet_LGD 대상 아님)' | 서술(본 실행에서 F4_direct 유지, Tibet 대상 셀에서 제외) | LG 계획서 6B.1(703행), 6B.3 'v3 라벨 표시의 불일치', 개정 15 (p)(3) |
| E51 | R1 − P*(P0@tddm), 라벨 전량, 주 4지역 층화 평균 | `data/processed/lgx/lgx_tests.csv` | test_id = L29, contrast = R1-P0@tddm\|all, scope = MEAN | `delta [ci_lo, ci_hi]; delta_blockeq [ci_lo_beq, ci_hi_beq]; verdict4` | -1.64 [-2.95, -0.74]; -2.23 [-3.17, -1.24]; 우세 | 우세 | LG 계획서 7.1a 'P* 병기', 7.2 '원고 문장에 주는 결정' 3 |
| E52 | R1 − P1*(P1@ed), n = 40 / 160, 층화 평균(풀 지역 2/4: 레나·캐나다) | `data/processed/lgx/lgx_tests.csv` | test_id = L29, contrast = R1-P1@ed\|n40 / R1-P1@ed\|n160, scope = MEAN | `delta [ci_lo, ci_hi]; delta_blockeq [ci_lo_beq, ci_hi_beq]; verdict4` | 1.31 [-0.13, 1.96]; 0.02 [-0.54, 0.57] / 1.30 [0.01, 1.86]; -0.13 [-0.68, 0.41] | 미결정 / 미결정 | LG 계획서 7.1a, 7.2 '원고 문장에 주는 결정' 2, WRAPUP J7 SC2w 행 |
| E53 | H19a: 알래스카 블록 E 의 LOBO R²(n 가중), 기본 17열 대 기본 17열 + A군(24열) | `data/processed/h2/h19_blockE.csv` | set = base17 / base17+A, model = catboost / ridge | `r2_lobo_AK_w` | CatBoost 0.48 → 0.52 / ridge 0.14 → 0.38 | 서술(H19a 해석) | `docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md` 2.2, `docs/EXPERIMENT_LOG.md` 155행 |

**표 읽는 법**
- 판정어 '서술'은 판정 대상이 아닌 자료 기술값이다. 판정어가 있는 행은 E41–E43, E51, E52(L29, 4분 판정), E49(H19b, 9월 규약의 확인적 가설)뿐이다. 4분 판정의 정의는 LG 계획서 6A.5a 와 QA 문서 머리말(우세·열세는 셀 가중과 블록 등가중 두 95 % CI 가 모두 같은 쪽, 동등은 두 CI 가 ±0.5 cm 안, 그 밖은 미결정)을 따른다.
- E42 의 Alaska(x) 행은 셀 가중 CI [0.30, 2.08] 이 0 을 넘지만 블록 등가중 CI 가 0 을 포함해 미결정이다(파일 `ci_lo_beq` −0.01, `ci_hi_beq` 0.52 를 같은 행에서 읽음).
- E43 은 셀 가중 CI 가 0 을 제외하나 블록 등가중 CI 가 0 을 포함해 미결정이다. LG 7.1a 의 P1* 규칙에 따라 라벨 10개와 전량의 재보정 기준선은 P1 그대로다.
- E38 의 `E_mean` 은 셀 E(= ALT/√TDD)의 기하 평균 exp(mean(z)), z = ln(ALT/√TDD) 이다(`v2_data.py` 649행. 예: 레나 `z_mean` 0.2722, exp(0.2722) = 1.3128). E37 의 E0 는 Σsy/Σs² 최소제곱 추정이다. 두 값은 정의가 달라 직접 빼지 않는다.
- E49 는 9월 M1 하네스 규약(공변량만 조건, 주 4지역 평균)의 결과다. LG 시험지와 규약이 달라 LG 수치와 섞지 않는다. 판정어 '기각'은 H18–H24 문서 §9 의 확인적 가설 판정(이득 없음)을 옮긴 것이다. 원문의 '추가 시 악화'는 확인되지 않았다. 셀 가중 CI 만 0 을 넘고 블록 등가중 CI 는 0 을 포함하며(같은 행의 `ci_lo_beq` −0.02, `ci_hi_beq` 1.24), Holm p 는 0.17 이다. 이 표의 4분 규칙으로 읽으면 미결정이다.
- E08 과 E50 을 합한 지온 유도 셀은 20개(알래스카 15, 캐나다 4, 티베트 1)로 LG 개정 15 (p)(3)의 기록과 같다. E08 의 대조는 region 'Canada'·'United States (Alaska)' 행에만 적용되었으므로 그 밖의 v3 F4_direct 행에 같은 표지가 없다는 뜻은 아니다.

**이번 계산(판정 아님)**
- 동등 한계의 P0 RMSE 대비 비율: E39(셀 가중, 분할 평균, 주 4지역 21.69–42.98 cm) 기준으로 δ 0.5 cm 는 1.2–2.3 %, δ 1.0 cm 는 2.3–4.6 % 다. P0 RMSE 의 정의는 WRAPUP 9.4 를 따른다(지역 행은 그 지역의 셀 가중 RMSE 의 분할 평균. `lg_curve.csv` 의 `rmse` 는 `h4_common.boot_delta_blocks` 의 `rmse_A`, 곧 분할별 RMSE 의 평균이다). 원고 초안(`docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md` 128행)의 'about 1–5 %'는 이 기준에서 유지된다. 블록 등가중(E40, 12.45–38.43 cm) 기준이면 δ 0.5 cm 는 1.3–4.0 %, δ 1.0 cm 는 2.6–8.0 % 이므로 원고 문장에 '셀 가중'을 밝힌다. 원고 초안의 `[old result, recheck: P0 RMSE 21.6–42.5 cm]` 자리(WRAPUP 9.2 의 M1 규약 값)는 E39 값으로 바꾼다.
- C2 점검의 부호(1.2 의 1): `data/processed/wf/wf0_misspec.csv`(사본 `tables/wf0_misspec.csv`, 30행)에서 `recal_gain`(재보정 이득 P0 − P1)과 `ml_vs_p1`(최선 ML − P1. 예: `Canada|x` 행 `ml_best_delta` −1.92 와 `P1` 열 4.64 의 차 −6.56)의 Spearman ρ 는 +0.082(p 0.666, 30대상, scipy `spearmanr`)다. ML 몫을 이득(P1 − 최선 ML)으로 정의하면 −0.082 다. 대상은 서로 독립이 아니고 블록 재표집 CI 는 계산하지 않았다.
- E35 의 최소·최대(0.78, 0.94)가 '78–94 %' 의 출처다. 확인적 평균 4지역(레나, 캐나다, 러시아 W, 러시아 E)만 보아도 0.78–0.94 로 같다.

---

## 3. 이 주장을 받치는 실험

| 새 이름 | 옛 id | 이 폴더에서 쓰는 내용 | 계획·판정 문서 |
|---|---|---|---|
| T1_transfer_label_grid | LG | 대상 27개, 분할 5, 라벨 수 격자, 추출 5, P0·P1 정의, 100 km 버퍼, A/B 셀 수(E24–E32, E39–E40) | LG 계획서 §1–§3, 7.1 |
| T2_transfer_structure_placebo_baselines | LGX | P*·P1*(L29, E41–E43, E51–E52), 오차 하한(N2), 무작위 분할 대 지역 홀드아웃(L19·L20, C7 폴더) | LG 계획서 6A.3, 6A.5a, 7.1a, 7.2 |
| T4_new_regions | LGD | v4 표, 새 지역 적격·약관 판(E06, E07, E17–E20) | LG 계획서 6B.3–6B.6, 7.3 |
| T6_abstract_contrasts_and_map | LGW(+WRAPUP) | 라벨 단위 표, S-a·S-b 민감도, 보고 규칙 1.1–1.3(E21–E23, E46–E47) | WRAPUP 1–2절, 결과 전 계산, J7 |
| W2_within_region_label_curve | WF1 + WF6 | 지역 내 블록 홀드아웃의 분할 수와 사후 설계 표지(E48) | WF 계획서 1절, 6절, 5.1–5.2 |
| W7_gain_decomposition_grid | WF9 | 같은 지역 내 설계(분할 25회)를 ERA5 격자 단위로 나눠 씀. 수치는 C8 폴더 | WF 계획서 7절, 5.3 |
| H0_contest | 대회(S0–S14 등) | v1 표 `data/processed/fidelity_base.csv`(파일 날짜 7월 23일, 17,423행은 E01) | `docs/CONTEST_REPORT_2026.md` |
| H1_prereg_E1-E4 | E3-2(09-08) | v2 표(CALM 149행 추가, E02) | `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` §3 |
| H2_master_M1 | M1 S-A | v3 표(출처 표지 68행 변경, E03), 검증 사다리 자료 `data/processed/m1/cv_scheme_comparison.csv`(C7 폴더) | `docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md` §3 |
| H3_transfer_fixes_H18-24 | H18·H19 | 확장 공변량표 `covariates_ext_v1`(E09), H19a 알래스카 블록 E 회귀(E53), H19b 기각(E49) | `docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md` §2, §9 |

**원고 대응 위치**: `docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md` 의 Methods 가운데 Observational data, Additional independent regions (LGD), Physics baselines, Targets, splits and label draws, Scoring and statistical inference 다. 재구성안의 결과 절 R1(평가 설계와 자료)도 이 폴더를 쓴다.

---

## 4. 그림과 표

| 항목 | 파일 | 패널·내용 | 자료 |
|---|---|---|---|
| Fig 1 | `outputs/figures/paper/v2/Fig1_problem.{pdf,svg,png}` | a 범북극 지도(0.5° 블록의 1 km 위치 수에 비례하는 원, v3 셀은 채운 원, LGD 약관 확인 셀은 윤곽 원, NAtlantic 은 'SI only' 표지, 레나 지도 영역 사각형, 티베트 삽도, CCI PFR 배경). b 지역별 z = ln(ALT/√TDD) 분포와 대상별 ln E0 눈금. c 설계 도식(지역 홀드아웃 + 100 km 버퍼, A/B 블록 절반, 라벨 추출, 채점, 방법 계열). d 원천 풀의 지역 구성(누적 막대)과 E0 | `data/processed/paper_figs/fig1_*.csv`(만든 모듈 `scripts/4_visualization/paper/v2_data.py`), 그림 모듈 `scripts/4_visualization/paper/fig1_problem.py`, 스펙 `figures/figure_spec.json` 'Fig 1' |
| Table 1 | `outputs/figures/paper/v2/Table1_data.{csv,md,tex,pdf,svg,png}` | 대상 27개(주 6, Alaska(x), 하위 지역 10)와 LGD 2지역의 행 수, 1 km 위치, 블록, 유효 분할, A 셀, 채점 셀(블록), E0, 라벨 유형, 판. 각주 a–f(모드, 셀 색인, 그린란드의 무효 분할, E0 단위, 약관 판, 확충판) | `data/processed/paper_figs/table1_rows.csv`, 모듈 `scripts/4_visualization/paper/table1_data.py` |

**재구성안 메모**(`docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md`)
- 4절 R1 '평가 설계와 자료'의 근거는 Table 1, LG §1–3, LGX N2, C7(무작위 분할 −10.3 cm)이고 그림은 Fig 1 이다.
- 5절: Fig 1 은 유지하고 패널 c 에 지역 내 블록 홀드아웃(충분 라벨) 설계를 한 줄 더한다. Table 1 은 유지한다. 패널 c 의 추가 줄에는 '결과 열람 뒤 설계' 표지를 함께 단다(1.2 의 3).

**QA 권고에서 이 폴더와 이어지는 표시 항목**
- 검증 사다리(무작위, 지점, 블록, kNNDM, 지역 홀드아웃)를 본문 그림 하나로 보인다(Q12 강화 4, Q15). 자료는 `data/processed/m1/cv_scheme_comparison.csv` 와 LGX L19·L20 이고 C7 폴더가 맡는다.
- 요인 × 실험 범위표를 SI 표로 낸다(Q9).
- 라벨 단위 표(WRAPUP 2.2, 근거 E21–E23 의 원표 `lgw_label_units.csv`)를 Table 1 또는 SI 에 병기한다(WRAPUP 2.5).

---

## 5. 단서와 쓰지 않을 문장

**단서**
1. **라벨 단위**: n 은 행 수다(WRAPUP 2.1). 알래스카는 1 km 위치당 평균 39.67행, 레나 15.11행, 캐나다 8.72행이다(E21–E23). 러시아 W·E 는 CALM 지점 다년 평균 1행이 지점 하나다. 같은 n 의 지역 비교 문장에는 단위를 붙인다(WRAPUP 2.5).
2. **라벨 단위 민감도**: 캐나다 n = 10 에서 점 라벨 조건의 재보정 이득은 1 km 위치 평균 조건보다 작았다(E46, 중앙값 −0.86 cm). WRAPUP 의 해석 규칙에 따라 배포 지침의 라벨 단위는 '1 km 위치 평균'으로 적는다. 이 해석 규칙의 n 범위(n ∈ {3, 10})는 첫 계산 뒤에 코드 해석으로 확정했으므로 '값을 본 뒤 정한 해석' 표지가 붙어 있다(WRAPUP 결과 전 계산, 수정 1).
3. **라벨 정의**: v3 의 F4_direct 에는 지온 유도 셀 20개가 남아 있다(알래스카 15, 캐나다 4, E08. 티베트 QTP_CN 1, E50). 알래스카·캐나다 셀은 PANGAEA 이벤트 좌표 대조로, 티베트 셀은 QTEC 지온 자료에서 만든 셀이라는 기록(LG 6B.1)으로 확인되었다. 표가 고정된 뒤 발견되었고 '탐침만' 변형과 같은 방식으로 다룬다(LG 6B.3). 원고 초안 Labels 절의 '15 in Alaska, 4 in Canada, 1 on the Tibetan Plateau'와 같다. 융해관 규칙도 판마다 다르다. v3 작성기(`scripts/1_data_prep/build_fidelity_base_v3.py` 51–52행, 119–121행)는 v2 에서 더한 CALM 149셀(`alt_calm_global_cell.csv`) 가운데 이벤트 방법이 지온 또는 융해관인 셀만 `F4_calm_temp` 로 내렸다. 같은 이벤트 좌표 대조에서 그 밖의 v3 F4_direct 셀 가운데 융해관 이벤트와 대응하는 셀이 알래스카 1, 캐나다 1 이고, 탐침·융해관이 모호한 셀이 캐나다 7 이다(E08). 따라서 'v3 는 융해관 지점을 뺐다'는 v2 에서 더한 CALM 지점 행에만 맞다. LGD 는 융해관을 직접 측정으로 센다(LG 6B.3, 개정 15 (p)(1)). 원고 초안의 `[verify: harmonize the thaw-tube rule]` 자리를 이 기록으로 해소한다.
4. **값 범위 규칙(자료원별)**: v3 ABoVE 행의 범위는 0 < ALT < 300 cm 다(`scripts/1_data_prep/parse_above_alt.py` 24–25행 `d = d[(d.ALT > 0) & (d.ALT < 300)]`). LGD 새 행은 등록대로 0 < ALT ≤ 600 cm 로 실행되었다. LG 6B.3 의 괄호 '(v3 의 QC 와 같다)'는 오기다(LG 개정 15 (p)(2)). 원고 자료 절에는 자료원별 범위 규칙을 적는다. v3 의 CALM·ALLena 행에 적용된 범위 규칙은 이 폴더에서 파서로 확인하지 않았다 [미확인](원고 초안의 `[verify: range rules of the CALM and ALLena parsers]`).
5. **시기 불일치**: 라벨 연도는 1990–2024 이고 공변량 기후값은 ERA5-Land 2015–2020, CCI 는 1997–2021 평균이다(원고 초안 Observational data, LG 6B.3). 지도와 예측은 2015–2020 기후값 기준의 정적 산출이다.
6. **공변량 해상도**: 기후(ERA5-Land 0.1°)와 토양(SoilGrids 를 약 5 km 로 요청해 추출)이 가장 거칠다. 'SoilGrids 250 m 사용' 표기는 정정 대상이다(QA Q8). 1 km 출력이어도 정보 해상도는 이 입력에 묶인다.
7. **새 지역과 약관**: 주 판정은 약관 확인 판이다. North Atlantic 은 전체 판 41셀(13블록)이고 약관 확인 판에서는 부적격이라 SI 에만 둔다(E19, Table 1 각주 e). 약관 미확인 셀(캐나다 38, North Atlantic 19, 러시아 W 1)은 그림과 주 판정에 넣지 않는다(E20). 원고 초안 서론의 "Three regions ... were added" 문장은 약관 판 규칙에 맞게 고친다.
8. **러시아 C 의 두 정의**: LG 본 실행의 러시아 C(v3, 7셀, 점 추정)는 100 km 버퍼가 레나 셀 1,252개를 원천에서 뺐다(E33). LGD 는 레나 델타 상자 안의 CALM R8 Tiksi 행(v3 loc 17557)을 Russia_C 대상에서 뺐다(`data/processed/fidelity_base_v4_meta.json` 의 `conventions.wrapper_design`, 개정 13). 두 러시아 C 결과를 한 문장으로 묶지 않는다.
9. **알래스카(x)의 원천**: 알래스카를 새 지역으로 둔 시험의 원천은 3,860셀이고 그 0.79 가 레나다(E34, E36). 주 4지역 평균과 합치지 않고 3지역 보조 열(레나·캐나다·Alaska(x))만 허용한다(WRAPUP 1.2).
10. **E0 의 편중**: 주 6지역 대상의 E0 는 1.59–1.62 로 거의 같다(E37). 원천이 알래스카 셀로 채워지기 때문이다. 지역 자체의 E 는 주 4지역 기하 평균 1.31–2.51(레나 1.31, 러시아 W 2.51)로 다르고, 러시아 C 는 1.81, 셀이 3개인 그린란드는 4.25 다(E38). P0 의 오차 상당 부분은 이 차이에서 온다는 해석이 가능하나 이 폴더에서 분해하지는 않았다.
11. **작은 대상**: 러시아 W·E 는 A 셀이 14–17개라 n ≥ 40 행이 없다(E29, E30). n = 40·160 의 층화 평균은 레나·캐나다 2지역이다(LG 판정 세부 규칙). 그린란드는 유효 분할이 없어 점 추정만 있다(E32). 러시아 C 와 그린란드는 블록 CI 와 n* 를 계산하지 않는다(LG §1).
12. **오차 하한의 뜻**: N2 의 하한은 DEM 계열을 뺀 거친 공변량 묶음(5셀 이상) 안의 관측 분산을 합동한 값이며 효과 크기 표기의 분모로만 쓴다(LG 6A.3). 측정 오차나 물리적 한계가 아니다. 레나는 범위(scope)에 따라 14.83(eval)과 14.91(all)로 다르다(`lgx_floor.csv`, Lena all 행). 대회 교차검증 수치에 알래스카 하한을 적용한 것은 같은 라벨 집합을 가정한 근사다(`docs/RESEARCH_OVERVIEW_2026-10-02.md` 5절).
13. **재사용 지역**: 주 4지역은 7–9월 실험에서 여러 차례 쓴 지역이다. 판정은 '재사용 지역의 재검정'으로, LGD 새 지역 판정은 '독립 지역 확인'으로 적는다(WRAPUP 1.2).
14. **P* 의 범위**: P* 가 P0 보다 우세라는 판정은 주 4지역 층화 평균에서만 성립한다. 알래스카를 넣은 3지역 평균과 Alaska(x) 단독은 미결정이다(E41, E42).
15. **실행 환경**: CatBoost 와 물리식은 CPU 결정적 실행이고 신경망은 환경에 따라 결과가 다르다. 한 대비 안에서 두 플랫폼을 섞지 않는다(WF 계획서 1절, LG 개정 15 (u)).

**쓰지 않을 문장**
- '알래스카 라벨 13,606개'(단위 없이). 대신 '13,606행(1 km 위치 343곳, 0.5° 블록 74개)'.
- '100 m 지지 라벨'(v3 `spatial_support_m` 은 상수 자리값이다, WRAPUP 2.1).
- 'SoilGrids 250 m 공변량을 썼다'.
- '확인적 독립 지역 6–7개'처럼 확인적 지역 수를 6 이상으로 적는 문장. 확인적 독립 지역은 주 4지역과 약관 확인 새 얕은 지역 1곳(러시아 C 확충판)이다(4–5개). 풀 크기를 적는 LG 7.3 의 등록 문구 '확장 풀(지역 6개)'(PE2, 티베트 포함)는 허용하되 확인적 지역 수와 섞지 않는다.
- '오차 하한은 측정 오차다', '물리적 한계다'.
- '안전한 설계', '보수적이므로 안전하다'.
- 'P0 대비 이득이 ML 의 기여다'(재보정 몫 포함).
- '지역 내 블록 홀드아웃 시험은 사전 등록했다'(사후 설계 표지 없이).
- '우리 채점은 지나치게 엄격하다' 또는 '무작위 분할은 틀렸다'(검증 사다리는 C7 폴더의 몫이고, QA Q15 는 질문에 따라 맞는 채점이 다르다고 정리했다).
- 'v3 의 직접 라벨은 모두 탐침·GPR 이다'(E08 의 지온 유도 셀이 남아 있다).

---

## 6. 이 주장을 바꿀 수 있는 계획 실험(X)

X 묶음은 2026-10-04 기준 계획 단계다. XG 는 WRAPUP 10절 '기존 제품 비교(등록만)'에 등록만 되어 있고(실행 조건, 제품, 방법, 중복 표시, 문장 규칙), 실행 조건(LGU-B2 를 충족해 지도를 본문 Fig 6c 에 둘 때)이 걸려 있다. 나머지 X 는 등록 문서가 없다(`paper/registry/experiments.csv` 의 X 행, `docs/` 검색). 아래는 이 폴더의 문장이 바뀔 수 있는 지점만 적는다.

| X | 내용(계획) | 이 폴더에 주는 영향 | 관련 사용자 질문(2026-10-04) |
|---|---|---|---|
| XF_new_regions | 공개 자료로 독립 지역 추가, 약관 회신(CUSP, GGD353, CALM 웹 하위 지점) | Table 1 행, v5 표, 확인적 독립 지역 수(4–5개), 원천 알래스카 비율(78–94 %), 6절 '새 지역' 문장 | 질문 1(독립 지역 추가, 알래스카 시험 뒤 다른 지역 시험) |
| XD_placement_policy | 샘플링 정책 학습과 지역 하나 제외 검증 | 검증 단위가 지역이므로 XF 의 지역 수에 의존한다. 이 폴더의 독립 지역 정의를 그대로 쓴다 | 질문 1, 질문 2 |
| XH_validation_ladder | 무작위·지점·블록·kNNDM·지역 홀드아웃을 한 그림으로 | 1.3 의 '같은 시험지' 문장에 검증 사다리의 위치를 더한다. 수치는 C7 | QA Q12·Q15 |
| XE_hires_covariates | 토양 수분, 유기층, 식생, 적설 등 격자 안 해상도 입력 | x25 공변량 정의와 단서 6. 확장표(`covariates_ext_v1`: ERA5-Land 토양 수분·적설 깊이·적설 밀도·LAI, MODIS NDVI 0.05°, Hansen 수관 피복, TWI 등)는 9월 전이 시험에서 이득이 없었다(E49, 기각). 알래스카 지역 내 셀 단위 ALT 예측 시험은 하지 않았다. 블록 E 의 LOBO 회귀(H19a 해석)는 있다(CatBoost 가중 R² 0.48 → 0.52, ridge 0.14 → 0.38, E53) | 질문 3(선행 연구의 격자 안 입력, 확보 가능 자료) |
| XJ_tempderived_aux_labels | 지온 유도 라벨을 보조 라벨로 쓰는 다중 충실도 | 라벨 정의 규칙(v3 F4_calm_temp 제외, v4 F3_ext_temp 39행), 단서 3 | QA Q9 |
| XG_product_comparison | 기존 ALT 제품과 같은 시험지 비교, 학습 자료 누설 점검 | 물리·제품 기준선 목록에 외부 제품이 들어간다. 누설 점검 규칙을 설계 절에 더한다 | QA Q9·Q11 |
| XB_multisource_stacking | P1, P*, 보정 Kudryavtsev, 토양형 Stefan, CCI 를 대상 라벨로 가중 적층 | '가장 강한 물리 기준선'(P*, P1*)의 정의가 바뀔 수 있다 | QA Q11 |
| XA_c2_gain_decomposition | 재보정 몫과 ML 몫 분리(기존 곡선 재분석) | C2 폴더(2.3)가 −0.08(재보정 이득 P0 − P1 대 ML 몫 P1 − 최선 ML)을 재현했고, 정의에 따라 순위상관이 −0.08 에서 0.39 사이라고 정리했다. 그래서 1.2 의 1 은 '상관이 없었다' 대신 '확인하지 못했다'로 쓴다. 두 기준선 원칙(ML 순가치는 P1 대비)은 유지된다. 등록 분석(블록 재표집 CI)이 나오면 이 문장을 다시 본다 | QA Q1 C2 점검 |
| XI_climate_extrapolation_retest | 외삽 영역 블록 8개 이상인 두 번째 대상으로 재시험 | 대상 적격 규칙(외삽 영역 블록 수)을 설계 절에 더한다 | 연구 주장 6절 |
| XC_workflow_end_to_end | 라벨 수별 워크플로 전체를 같은 시험지에서 실행 | A/B 분할·라벨 격자를 그대로 쓴다. 설계 변경은 없다 | 사용자 요청(추가 실험 가능한 한 모두) |

---

## 7. 사본 표와 복사하지 않은 원천

`tables/` 에는 위 근거에 쓴 집계 표 19개의 사본이 있다(각 5 MB 이하, 셀 단위 라벨 자료 없음). 수정 때 `wf0_misspec.csv`(대상 30행, 1.2 의 1과 '이번 계산')와 `h19_blockE.csv`(공변량 집합 × 모형 18행, E53)를 더했다. `tables/MANIFEST.csv` 의 열은 `orig_path, copy_path, sha256, rows, bytes` 다. `rows` 는 CSV 의 데이터 행 수(머리글 제외)이고 JSON 은 최상위 키 수다. 복사 뒤 원본과 사본의 sha256 이 같음을 확인했다.

**복사하지 않은 원천**
- `data/processed/fidelity_base_v3_meta.json`, `data/processed/fidelity_base_v4_meta.json`: 셀 단위 행(loc_id, 좌표, ALT 값 일부)을 포함하므로 복사하지 않았다. 근거 E03, E04, E06–E08 은 이 두 파일을 경로로 인용한다.
- `results/rescale_lg/data/processed/lg/lg_curve.csv`: 7,072,630 바이트로 5 MB 를 넘는다. 근거 E39–E40 은 이 파일을 경로와 행 필터로 인용한다.
- `data/processed/paper_figs/fig1_z_points.csv`: 셀 단위 z 값이라 복사하지 않았다.
- `data/processed/fidelity_base_v4_labels.csv`: 셀 단위 라벨 표라 복사하지 않았다. 근거 E50 은 loc_id 17389 행의 `source_id`, `lgd_role` 을 경로와 행 필터로 인용한다.
- 코드 인용(`scripts/1_data_prep/parse_above_alt.py`, `scripts/1_data_prep/build_fidelity_base_v3.py`, `scripts/4_visualization/paper/v2_data.py`, `src/polar/h4_common.py`)은 경로와 행 번호로만 적었다.

---

## 8. 수정 기록(2026-10-04)

검증 지적 11건을 원천과 다시 대조한 뒤 반영했다. 수치를 바꾼 기존 행은 없다(E49 는 블록 등가중 값과 원값을 더했다).

| 지적 | 확인한 원천 | 반영 위치 |
|---|---|---|
| E38 의 정의(산술 평균 → 기하 평균) | `v2_data.py` 649행 `E_mean=float(np.exp(np.mean(z)))`; `fig1_z_summary.csv` 레나 `z_mean` 0.2722, `E_mean` 1.3128 | E38 행, 표 읽는 법 4번째, 단서 10 |
| 단서 10 의 E 범위 | `fig1_z_summary.csv`(그린란드 `E_mean` 4.248) | 단서 10 |
| 단서 4 의 [미확인] | `parse_above_alt.py` 24–25행; LG 개정 15 (p)(2) | 단서 4 |
| E08·단서 3 의 지온 유도 셀과 융해관 | v4 meta `v3_method_check.counts`; v3 meta `macro_table`(Tibet F4_direct 1셀), `obs_method.per_cell` 149셀; `fidelity_base_v4_labels.csv` loc_id 17389; `build_fidelity_base_v3.py` 51–52행, 119–121행; LG 6B.1, 6B.3, 개정 15 (p)(1)(3) | E08 행, E50 행 신설, 표 읽는 법, 단서 3 |
| 1.3 의 기준선 문장 | LG 7.1a 'P* 병기', 7.2 '원고 문장에 주는 결정' 2–3; WRAPUP J7 SC2w 행; `lgx_tests.csv` L29 | 1.3 국문·영문, E51·E52 행 신설 |
| 확인적 독립 지역의 정의 | WRAPUP 1.2 표, 3.3 SC1w, J7 SC1w 행; LG 7.3 '주 판정 판'과 결론 문구; 연구 주장 문서 6절 | 1.2 의 4, 1.3 국문·영문, 쓰지 않을 문장 |
| C2 점검 −0.08 의 부호 | `wf0_misspec.csv` 재계산(ρ +0.082, p 0.666); C2 폴더 2.3 근거 표와 '정리' 1–4 | 1.2 의 1, 이번 계산, XA 행 |
| E49 판정어 | `h_tests_all.csv`(H19p_x25_A_vs_x25_pseudo, covonly, is_mean) `delta_blockeq` 0.55, `ci_lo_beq` −0.02, `ci_hi_beq` 1.24, `p_holm` 0.168; H18–H24 §9 | E49 행, 표 읽는 법 |
| XE 행의 '지역 내 시험은 하지 않았다' | `h19_blockE.csv`(`r2_lobo_AK_w`); `docs/EXPERIMENT_LOG.md` 155행 | XE 행, E53 행 신설 |
| 6절 머리말(등록 문서 없음) | WRAPUP 10절; registry XG 행 | 6절 머리말 |
| 동등 한계 비율 | `lg_curve.csv`(P0, n 0, x) `rmse`·`rmse_beq`; WRAPUP 9.2, 9.4; `h4_common.boot_delta_blocks`; 원고 초안 128행 | 이번 계산 |

**이 폴더 밖에 남은 불일치(고치지 않았다)**: `docs/QA_FINAL_REVIEW_2026-10-02.md` 43행과 `paper/registry/experiments.csv` XA 행은 ML 몫을 '최선 ML − P1'로 적고 ρ −0.08 을 붙였다. 부호 정의가 1.2 의 1과 반대다. LG 6B.3 의 '(v3 의 QC 와 같다)' 괄호는 개정 15 (p)(2)가 오기로 기록했으나 본문은 그대로다.
