# Handoff: 연구 주제 재설정(라벨 수별 물리 기반 ML 워크플로), LG 계열·LGF·WF 1–3차 판정 완료
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-10-02 10:33
**Session focus**: 사전 등록한 LG 계열 실험(라벨 격자 전이 시험)과 확장 묶음(LGX, LGT, LGD, LGU, LGF)을 끝내 판정하고, 사용자 지적(10-01)에 따라 연구 주제를 '물리 증강·물리 잔차 ML 의 라벨 수별 능력과 새 지역 워크플로'로 다시 세워 WF 1–3차 실험으로 근거를 보강했다.
**Author**: Claude Opus 5.5 + 백승원

---

## 1. TL;DR
- **주제 재설정(사용자 10-01)**: 논문의 중심은 '물리 증강·물리 잔차 ML 이 라벨 수에 따라 희소·충분 라벨 지역에서 보이는 능력과, 새 지역의 학습·평가·추가 관측 워크플로' 다. '무작위 분할은 과대평가한다', '라벨 0 에서 ML 은 실패한다' 는 보조 근거다.
- **라벨 0**: 직접 ML 은 시험한 학습기 10종 모두 원천 계수 Stefan(P0)보다 오차가 컸다(CatBoost +2.25 cm [1.10, 3.42], TabICL v2 +0.93, 원천 교차검증으로 조정한 신경망 4종 +2.4 – +5.2 cm). 물리 증강과 저가중 잔차가 손해를 막는다(30대상 위험표: 2 cm 넘게 악화한 대상 직접 ML 18, 물리 증강 2).
- **이득의 조건과 방법 선택**: ML 이득은 물리 계수 오차에 비례한다(순위상관 0.66, 사후). 라벨 10개 편향 진단이 이득을 예측한다(ρ 0.38 [0.17, 0.55], 등록 지지). 라벨 안 교차검증 규칙 W 가 고정 레시피보다 라벨 40·160개에서 0.87·1.32 cm 낫다(레나·캐나다 2지역, 이득은 캐나다).
- **충분 라벨과 이득의 출처**: 알래스카 지역 내 이득은 0.4–0.5 cm(분할 25회, 유의)이고 대부분 ERA5 격자 사이 성분(격자 단위 편향 보정)에서 온다. 격자 안 공간 상세도는 없다(설명 비율 0–6 %). 기후 외삽(WF10)은 판정 불가.
- **상태**: 등록 실험은 모두 끝났다. 남은 것은 원고 재구성(사용자 결정 3건), SI, 기탁이다. 모든 커밋은 origin/main 에 push 했다(b260754).

## 2. Context
- 이전 handoff: `gpt/handoff/20260926_2130-final-experiments-audit-figures.md`. 그 뒤 09-28–09-29 에 큰 틀(`docs/RESEARCH_FRAME_2026-09-29.md`)을 정하고 LG 계열을 사전 등록했다.
- 이번 세션의 흐름:
  - 09-29: LG(라벨 격자 통합 재실행)를 Rescale 에 제출.
  - 09-30: LG 계열 판정. 사용자가 Rescale 지출 중단을 지시해 남은 작업은 로컬 GPU 로 옮겼다.
  - 10-01: 사용자가 결과 설명을 듣고 '너무 뻔하다, 물리 기반 ML 의 장점과 충분 라벨 지역 시나리오, 다음 관측 위치를 결론으로 내라' 고 지적했다. 이에 따라 WF 실험을 설계해 Rescale 에서 돌렸다(사용자 지시: 주장 강화 실험은 Rescale).
  - 10-02: LGF 판정, WF 3차(물리식과 비슷한 지역에서 ML 의 장점 검증), Fig 6, 연구 개요를 마쳤다.
- legacy_handoff_dirs 는 없다.

## 3. What we did
- **LG 계열 판정(09-30)**
  - Action: 주 4지역(레나, 캐나다, 러시아 W·E)을 학습에서 빼고 라벨 0–전량에서 방법 12종·학습기 9종을 비교한 LG 와 확장 묶음을 판정했다.
  - Files: `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 7.1–7.4, `docs/EXPERIMENT_PLAN_LGU_2026-09-29.md` 11.1, `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 결과 판정 기록(J7).
  - Result: 라벨 0 물리식 우위, 더 강한 물리 기준선 P* = P0@tddm(−1.00 cm), 전량 잔차 ML 은 러시아 W 에 치우친 이득, 불확실성 확인적 가설 미결정.
  - Wallclock: Rescale 6작업 합 220.26달러(목록가 × 실행 시간).
- **WF 1·2차(10-01, Rescale elm)**
  - Action: 충분 라벨 지역 안의 라벨 수 곡선(20–5,000), 관측 위치 전략 7종, 하위 군집 계수, 라벨 안 선택 규칙과 편향 진단, 검정력 보강(분할 25회), Stefan·CCI 앵커 전이, 전이 조건 관측 위치.
  - Files: 계획 `docs/EXPERIMENT_PLAN_WF_2026-10-01.md`(판정 5.1·5.2), 하네스 `scripts/3_deep_learning/h54_workflow.py`, 시험 `tests/test_h54_workflow.py`.
  - Result: 4절 표.
  - Wallclock: xQyfeb 26분, OUZBT 6분(합 약 2.5달러).
- **LGF(로컬 RTX 3090)**
  - Action: TabICL v2 와 원천 지역 교차검증으로 조정한 신경망 4종(MLP, 다중 헤드 MLP, 축소 FT-T, RealMLP).
  - Files: `scripts/3_deep_learning/h47_foundation_models.py`, `h48_nn_tuning.py`, 판정 `docs/EXPERIMENT_PLAN_LGF_2026-09-29.md` 10절.
  - Result: F1·N1 지지(라벨 0 직접 ML 열세), F4 TabICL 잔차 순가치 n = 10 부터, N2 조정 이득 없음.
  - Wallclock: 1,730단위, 실패 0, 10-01 23:34 완료.
  - 집계 결함 2건을 결과 열람 전에 고쳤다: 결과 뒤 파이프라인의 `--allow-local` 누락, h48 조각 대조가 대상별 선택 설정까지 비교한 문제.
- **WF 3차(10-02, Rescale hematite)**
  - Action: WF9 는 오차를 ERA5 격자 안·격자 사이로 분해했고, WF10 은 따뜻한 블록으로 기후 외삽을 시험했다.
  - Files: 계획 7절, 판정 5.3, 표 `results/rescale_wf3/data/processed/wf/wf3b_{tests,decomp,rmse}.csv`, 재현 점검 `scripts/2_evaluation/wf9_repro_check.py`.
  - Result: 4절 표.
  - Wallclock: elm 작업 oCvpT 가 60분 동안 노드를 받지 못해 멈추고, Vppbeb(hematite 64코어)로 다시 돌렸다(실행 7분, 0.48달러).
- **그림·원고**
  - Fig 6 v2: `outputs/figures/paper/v2/Fig6_label0_uncertainty.{svg,png}`, 모듈 `scripts/4_visualization/paper/fig6_label0_uncertainty.py`. 값 대조 252항목이 통과했다.
  - 원고 초안의 LGF 자리를 채웠다: `docs/MANUSCRIPT_DRAFT_RESULTS_ABSTRACT_2026-09-30.md`.
  - 재구성안: `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md`.
- **종합 문서**
  - 연구 개요: `docs/RESEARCH_OVERVIEW_2026-10-02.md`(흐름, 주장 C1–C8, 워크플로, 선행 연구 대비 차이, 질문별 답).
  - 주장: `docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md`.

## 4. Key numbers (this session)

Δ 는 RMSE 차(cm)이고 음수가 개선이다. 괄호는 셀 가중 95 % CI 다. 판정은 두 가중(셀 가중, 블록 등가중) CI 를 모두 쓴 4분 판정이다.

| Method | Domain/Case | Metric | Value | Source artifact |
|---|---|---|---|---|
| 직접 ML(CatBoost) − P0 | 라벨 0, 주 4지역 평균 | Δ | +2.25 [1.10, 3.42], Holm 뒤 열세 | `data/processed/lgw/lgw_bundle.csv`(AB1) |
| TabICL v2 직접 − P0 | 라벨 0, 컨텍스트 1만 행 / 원천 전체 | Δ | +0.93 [0.28, 3.12] / +0.85, 열세 | `data/processed/lgf/lgf_tests.csv`(LGF-F1) |
| TabICL v2 직접 − P0 | 라벨 10, 4지역 / 3지역(알래스카 포함) | Δ | −2.91 / +5.98(L1 결론의 한계) | 같음 |
| 조정 신경망 직접 − P0 | 라벨 0, MLP / 다중 헤드 / FT-T / RealMLP | Δ | +2.45 / +2.44 / +4.13 / +5.21, 모두 열세 | `data/processed/lgf/lgfn_tests.csv`(LGF-N1) |
| TabICL 잔차 − 재보정 물리식(P1) | 라벨 10 / 전량 | Δ | −0.55 / −1.00, 우세 | `lgf_tests.csv`(LGF-F4) |
| 2 cm 넘게 악화한 대상 수 | 라벨 0, 30대상: 직접 ML / 잔차 1.0 / 물리 증강 / 증강+앵커+잔차 | 대상 수 | 18 / 13 / 2 / 1(사후 서술) | `data/processed/wf/wf0_risk.csv` |
| 재보정 이득과 최선 ML 이득 | 30대상, 라벨 전량 | Spearman | 0.66(사후 서술) | `data/processed/wf/wf0_meta.json` |
| 편향 크기(라벨 10) 대 규칙 W 이득 | 28대상 | Spearman | 0.38 [0.17, 0.55], 지지 | `results/rescale_wf/data/processed/wf/wf_tests.csv`(WF4-c) |
| 규칙 W − 고정 R1(λ 0.25) | 라벨 40 / 160(레나·캐나다) | Δ | −0.87 / −1.32, 우세 | 같음(WF4-a) |
| R1(교차검증 λ) − P1 | 알래스카 지역 내, 라벨 500 / 1,000 | Δ | −0.39 / −0.54, 우세(RMSE 14.41 → 13.87) | `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`(WF6-a) |
| 공변량 최대 최소 거리(S4) − 무작위(S1) | 캐나다 전이, 라벨 40 | Δ | −2.39, 우세 | 같음(WF8-a) |
| R1 − P1, 격자 사이 / 격자 안 / 총 | 알래스카 지역 내, 라벨 전량 | Δ | −1.03 우세 / −0.01 동등 / −0.52 우세 | `results/rescale_wf3/data/processed/wf/wf3b_tests.csv`(WF9-b) |
| R1 − P1 격자 안, 3대상 평균 | 지역 내, 라벨 전량 | Δ | +0.09 [0.02, 0.40], 열세(0.5 cm 미만) | 같음(WF9-a 기각) |
| 격자 안 설명 비율 | 알래스카 R1 / 레나 R1(λ 0.25) / 레나 전이 | 1 − MSE_w(M)/MSE_w(P1) | 0.1–0.7 % / 5.6 % / 3.2 % | `wf3b_decomp.csv` |
| 직접 ML 외삽 손실 EP | 알래스카 warm, 라벨 100 | Δ_W − Δ_I | +7.02 [3.91, 9.51], 블록 등가중 CI 0 포함 → 미결정. 층화 평균은 판정 불가 | `wf3b_tests.csv`(WF10-a) |
| 오차 하한 | 알래스카 / 레나 / 캐나다 | RMSE | 11.31 / 14.83 / 17.90 | `data/processed/lgx/lgx_floor.csv` |
| 지역 홀드아웃 − 무작위 분할(V-G − V-R) | 5지역, 직접 ML(catboost_lo) | ΔRMSE | +10.34 [7.29, 12.84](무작위 분할이 낙관) | `data/processed/lgx/ladder/lgv_tests.csv`(L19), 판정 기록 LG 7.2 |
| WF9 대 WF6 재현 | 74조각, 공통 키 11,814 | 최대 ΔSSE 절댓값 | 0(elm 대 hematite) | `data/processed/wf/wf9_repro_check.csv` |

대회 보고서와의 관계(`outputs/report/main.tex` 의 표와 일치 확인):
- 알래스카 지역 내: Stefan 14.46, 물리 잔차 최선 13.33 cm.
- 오차 하한 위의 줄일 수 있는 오차로 환산하면 대회 38.7 %, 이번 엄격한 설정 19.2 % 다(하한은 v3 셀 집합 기준 근사).

## 5. Decisions made
- **주제 재설정**: 사용자 지정(메모리 `research-main-framing`). 원고 재구성안이 이를 따른다.
- **실험 실행 위치**: 주장 강화 실험(WF)은 Rescale, 멈춘 GPU 실험(LGF)은 로컬 GPU 가 빌 때. 사용자 10-01 지시다.
- **실행하지 않은 것**
  - LGF 3순위 J12: 등록한 창 규칙에 따라 돌리지 않았다.
  - FT-T 로컬 이어 실행: 교차 환경 점검 (iii)에서 키의 14.9 % 가 0.5 cm 를 넘어 규칙대로 멈췄다.
  - WF10 재시험: 알래스카만으로는 블록 사이 변동 때문에 다시 미결정이 될 가능성이 커 권하지 않는다.
- **장점으로 쓰지 않는 것**
  - 공간 상세도: WF9 에서 격자 안 설명 비율이 0–6 % 였다.
  - 기후 외삽의 물리 일관성: WF10 판정 불가.
  - 이런 지역의 장점은 손해 없는 선택, 계수 오차 진단, 위성 제품 결합, 격자 단위 편향 보정으로 쓴다.
- **WF9·WF10 √TDD 열 이탈**
  - 등록 문구는 토양 √TDD 였고, 구현은 물리식이 쓰는 기온 √TDD(`e5_sqrt_tdd`)였다.
  - 등록 문구의 목적(묶음 안 물리식 예측 동일)은 구현 쪽이 충족한다.
  - 두 정의의 묶음 일치도(ARI)는 알래스카 0.999, 캐나다 0.997, 레나 0.832 다. WF10 판정 불가는 두 정의에서 같다.
  - 이탈 기록은 WF 계획서 개정 이력(2026-10-02 10:26)에 있다.
- **판정 근거 표 추적**: LGF·WF 판정 표를 git 에 넣었다(b260754). LG·LGX 표는 기존처럼 추적 밖이다.

## 6. Open questions / blockers
- **원고 재구성(사용자 결정 3건)**
  - Fig 5 를 관측 위치 전략(WF2·WF8)으로, Fig 7 을 충분 라벨·이득 분해·워크플로로 바꿀지
  - 초록에 WF 판정을 넣을지(WRAPUP 1.1 개정 필요)
  - 제목
  - 근거: `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 8절
- **독립 지역 수**: 확인적 독립 지역은 4–5개(러시아 W·E 15셀 안팎)이고, 원천의 78–94 % 가 알래스카다. 약관이 확인된 새 얕은 지역은 러시아 중부 하나다. LGD 약관 제공자 4곳(`docs/LGD_LICENSE_CONTACTS_2026-09-30.md`)의 답이 오면 늘릴 수 있다.
- **WF9 토양 √TDD 민감도**: 선택 사항이다. Rescale 약 0.5달러, 15분이면 된다.
- **문헌 원문 확인**: O'Malley 2026(arXiv, 지중 온도, 학습 제외 지역 × 현지 관측 1–40개)과 Ahajjam 2025(JGR MLC)는 초록 수준만 확인했다. 원문을 봐야 신규성 문장의 범위(ALT 한정)를 확정할 수 있다.
- **README**: 셀별 UQ·AOA 주장이 LGU·WF9 결과와 맞지 않는다. 공개 전에 갱신해야 한다.

## 7. Next steps (prioritized)
1. **재구성안 결정**: owner user. 5분. 선행 조건 없음.
2. **결과·토의·초록 재작성**: owner Claude. 1–2일. 선행 조건 1.
   - `docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md` 의 C1–C8 과 워크플로 표를 결과 절 순서로 옮긴다.
   - 기존 초안의 [MISSING]·[DECISION] 을 함께 처리한다.
3. **Fig 5·Fig 7 재설계**: owner Claude. 약 1일. 선행 조건 1.
   - Fig 5 원천: `wf_tests.csv` WF2, `wf2b_tests.csv` WF8.
   - Fig 7 원천: `wf2b_tests.csv` WF6, `wf3b_decomp.csv`.
   - 같은 v2 스타일, 값 대조, 시각 검토를 거친다.
4. **SI 조립, 투고 형식 점검, 자료·코드 기탁 준비**: owner Claude. 1일. 선행 조건 2·3. 약관 미확인 행은 제외한다.
5. **GPT 비평 요청**: owner GPT. 선행 조건 없음.
   - (a) C1–C8 문장이 4절 근거보다 강하게 쓰였는지
   - (b) `docs/RESEARCH_OVERVIEW_2026-10-02.md` 7절의 선행 연구 대비 차이가 정확한지, 빠진 최근 문헌이 있는지
   - (c) 제목 후보 3개(재구성안 2절) 평가
6. **선택 실험**: WF9 토양 √TDD 민감도. owner Claude. Rescale 15분. 선행 조건은 사용자 승인.

## 8. Pointers
- **연구 개요(먼저 읽을 것)**: `docs/RESEARCH_OVERVIEW_2026-10-02.md`.
- **판정 정본**:
  - LG 7.1–7.4
  - LGU 11.1
  - WRAPUP J7
  - LGF 10절
  - WF 5.1–5.3
- **남은 작업 표**: `docs/EXECUTION_PLAN_REMAINING_2026-09-30.md` 10.3. 상태 스냅샷은 `SESSION_HANDOFF.md`.
- **대회 보고서(authoritative)**: `outputs/report/main.tex`.
- **그림**:
  - v2 본문 그림: `outputs/figures/paper/v2/`
  - 사양: `figures/figure_spec.json`
  - 캡션: `outputs/figures/paper/CAPTIONS.md`
  - 레나 지도(SI): `outputs/maps/transfer_lena/`
- **Active jobs**: 없음.
- **Related prior handoffs**: `gpt/handoff/20260926_2130-final-experiments-audit-figures.md`, `20260926_1200-final-paper-plan.md`.

## 9. Caveats for GPT
- **사후 설계 표지**: WF 실험은 LG 결과를 본 뒤 설계했다. 결과 전에 커밋해 규칙을 고정했지만 '결과 열람 뒤 설계' 표지가 붙는다. WF0 와 WF9 이득 분해의 일부는 사후 서술이다.
- **판정 읽는 법**: 4분 판정의 '미결정' 은 효과가 없다는 뜻이 아니다. 두 가중 CI 가 엇갈리거나 0 을 포함한다는 뜻이다. '동등' 은 두 가중 CI 가 모두 ±0.5 cm 안이라는 뜻이다.
- **플랫폼 규칙**: Rescale 결과와 로컬 결과를 한 대비 안에서 섞지 않는다. CatBoost 와 물리식은 플랫폼 사이에서 같았다. 신경망은 최대 5.6 cm 달랐다.
- **이름 주의**:
  - 코드의 `tabm` 은 다중 헤드 MLP, `ftt` 는 축소판이다. 논문 모델(TabM, FT-Transformer)로 부르지 않는다.
  - P1 은 κ = 10 수축 재보정, P* 는 P0@tddm(연도 정합 도일), P1* 는 P1@ed 다.
- **폐기·대체된 주장**(인용하지 않는다):
  - '17 cm 물리 하한'
  - '12.97 cm SOTA'
  - '필요 라벨 수 ρ −0.84'
  - '소수 라벨 이득의 72–95 % 는 재보정'
  - 'ML 이 물리식을 넘는다'(조건 없이)
  - '공간 상세도 장점'
  - '기후 변화에서 더 안전'
  - '표형 파운데이션 모델은 물리식을 못 넘는다'(계열 일반화)
- **물리식 입력**: Stefan 식의 √TDD 는 ERA5-Land 기온 기반 `e5_sqrt_tdd` 다(h40 의 s). 토양 도일 열 `e5_sqrt_tdd_soil` 은 채점 셀 마스크에만 쓴다.
- **정본 용어**(project.yaml naming_canonical): 활동층(ALT), 물리 잔차 결합, 위성 제품(ESA CCI). 전이 공통 입력 25종 = 지형 6 + 기후 8 + 토양 9 + CCI 2.

## 10. 추가(2026-10-02 11:35, handoff 뒤)
- **WF9 토양 √TDD 민감도 완료**(사용자 지시, 로컬 CPU 99단위, 실패 0). 총 저장소는 본 실행과 같다.
  - WF9-a(지역 내)는 두 정의 모두 기각이다. 토양 정의에서는 격자 안 R1 − P1 이 +0.27 – +0.42 cm 로 열세다.
  - WF9-c(전이)는 토양 정의에서 기각(우세 없음)이다. 기온 정의의 부분 지지는 유지되지 않았다.
  - 등록 해석 규칙대로 C8 의 전이 격자 안 문장은 기각으로 쓴다. 알래스카 이득이 격자 사이 성분(−1.03 cm)에서 온다는 결과는 두 정의에서 같다.
  - 표 `data/processed/wf_soil/wf3b_tests.csv`, 기록은 WF 계획서 개정 이력이다.
- 7절 다음 단계 6(선택 실험)은 끝났다. 다음 세션 첫 작업은 원고 재구성과 그림 재구성(Fig 5·7)이다.

