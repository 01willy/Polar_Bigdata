# Handoff: 10-10 재설정(셀-연도 설계)부터 덱 v10 · 논문 초안 v11(Sci Rep)까지
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-10-18 19:04 (작업 기간 2026-10-10 ~ 10-16, 커밋 · push 10-18)
**Session focus**: 극지연구소 피드백으로 연구를 셀-연도 설계로 재설정하고, 그 결과를 발표 덱 v10(28쪽)과 Scientific Reports 양식 원고 v11(영문 · 국문 · 보충 자료)로 정리하였다.
**Author**: Claude Opus 5.5 + 01willy

---

## 1. TL;DR
- **설계**: 라벨 = 1 km 셀 × 관측 연도(2010–2024, 알래스카 · 레나 델타 · 캐나다 북서부 셀-연도 19,172개). 기후 입력은 ERA5-Land 융해 도일(TDD)을 관측일까지 누적한 값이다. 주 검증은 0.5° 공간 블록 교차검증, 전이 시험은 지역 홀드아웃이다. '판정됨(resolved)'은 짝지은 블록 부트스트랩 95% 구간이 0을 제외한다는 뜻이다.
- **주 결과**: 물리 잔차 모델 R1(재보정 Stefan + CatBoost 잔차, λ 0.5)은 재보정 Stefan P1 대비 RMSE 를 레나 델타 −3.6 [−7.8, −0.8], 캐나다 −3.7 [−4.8, −0.8] cm 줄였고, 알래스카에서는 −0.2 [−0.6, +0.1] cm 로 같은 수준이었다. 직접 ML D0 는 지역 안 전량 라벨에서 점추정이 R1 보다 낮지만(17.3 · 15.9 · 26.1 대 18.4 · 17.4 · 26.9 cm), 레나 델타 지역 홀드아웃에서 오차가 Stefan 보다 +17.8 [+8.0, +23.9] cm 컸다.
- **적은 라벨 · 새 지역**: 레나 델타를 빼고 학습한 R1 은 23.8 cm, 레나 라벨 25개를 더하면 19.9 cm 였다. 외부 CALM 7개 지역(공통 474 셀-연도)에서는 라벨 5개로 56.2 → 40.3 cm 가 되었고, ESA CCI v5 60.7 cm 보다 낮았다. CCI v5 값이 있는 700 셀-연도 기준 ΔRMSE 는 −19.5 [−31.3, −9.2] cm 이다.
- **제품 비교(같은 셀 · 같은 해)**: 제품이 학습하지 않은 레나 델타에서 R1 21.1 cm 는 공개 제품 5종(28.4–34.6 cm)보다 낮았다. 알래스카 · 캐나다에서는 Wei · Liu 와 같은 수준이었다(이 두 제품은 인근 관측으로 학습).
- **산출물**: 원고 v11(방법 → 결과 순서, 그림 7 + 표 1, 보충 그림 S1–S10, 규정 점검 통과)과 덱 v10 을 만들었다. 10-08 이후 미커밋분은 8개 커밋으로 main 에 반영하였다(fbcbb64 까지 push, dbcb552 와 이 인계 문서는 이번 push).

## 2. Context
- 직전 인계: `gpt/handoff/20261008_1637-pre-reset-snapshot.md`(최종 묶음 XA–XM 판정, 재설정 전 상태).
- 10-08 극지연구소 회의 뒤 피드백(10-10)은 세 가지였다. 입력 연도가 섞여 있고, 지도의 연도가 불분명하며, 희소 라벨 초점은 버린다. 정본은 `docs/RESET_2026-10-10_KOPRI_FEEDBACK.md` 이다. 이에 따라 라벨 · 입력을 같은 해로 맞추는 셀-연도 설계(v5)로 전면 재실행하였다.
- 10-14 관측일 정합(관측일까지 누적 TDD)을 채택하였다(core_v6). 10-14~10-16 사용자 반려가 네 차례 있었고, 덱은 v7 → v10, 원고는 v6 → v10 → v11 로 고쳤다.
- 9월까지의 결과(M1, H18–H30, LG · LGF · WF, XA–XM)는 다년 평균 라벨과 다른 입력을 썼으므로 v5 이후 수치와 직접 비교하지 않는다.

## 3. What we did
- **v5 셀-연도 체계(10-10)**
  - Files: `docs/RESET_2026-10-10_KOPRI_FEEDBACK.md`, 라벨 `data/processed/v5/labels_cell_year_v5.csv`(대용량, 로컬), 집계 `scripts/2_evaluation/v5_aggregate.py`.
  - Result: 검증 사다리 5단(무작위, 공간 블록, 연도, 블록 × 연도, 지역 홀드아웃)을 정의하였다.
- **관측일 정합 core_v6(10-14)**
  - Files: `data/processed/v5/runs/core_v6/{summary_metrics,deltas_vs_ref,deltas_forcing,common_cells_eval,products_eval*}.csv`, 수치 `scripts/2_evaluation/v6_numbers.py` → `docs/numbers_v6.json`.
  - Result: 관측 연도 입력이 평년 입력보다 RMSE 를 줄였다(레나 P1 −1.4 [−2.6, −0.8], 레나 R1 −1.6 [−2.4, −1.0], 캐나다 R1 −1.5 [−2.0, −0.4] cm). 알래스카는 +1.8 [−0.1, +3.9] cm 로 판정되지 않았다(라벨의 51% 가 2014년).
  - Wallclock / GPU: Rescale CPU(elm 96코어) 실행 CJsgU, 신경망 학습기(MLP · TabM · FT-T)는 A100 1장 노드 RkmTeb.
- **라벨 수 사다리 · 전이(10-15)**
  - Files: `data/processed/v5/runs/ladder_v6/`, `scripts/3_deep_learning/v6_label_ladder.py`, 집계 `scripts/2_evaluation/v6_label_ladder_agg.py`.
  - Result: Rescale nMsgU(약 10분)으로 실행하였다. 라벨 25개에서 R1 ≥ 같은 라벨의 재보정 Stefan, 라벨 100개에서 세 지역 모두 이득이 판정되었다.
- **추가 실험 v9(10-15)과 외부 적응 EXTFT(10-16)**
  - Files: `data/processed/v5/runs/extra_v9/{extft_deltas,extft_metrics,pa_deltas,shap_summary}.csv`, 행 표 `outputs/figures/paper/v7/source_data/F10_extft.csv`, 수치 `scripts/2_evaluation/v9_numbers.py`.
  - Result: 실행은 Rescale gxnyhb, dfuMW(elm, 각 10분 안팎)이다.
    - 제품 기준 잔차 결합: 레나에서 Wei 재보정 + 잔차가 −2.3 [−7.4, −0.4] cm, Stefan 기준이 제품 기준보다 −2.8 [−6.2, −0.1] cm 였다.
    - 레나 무라벨 전이: Wei 대비 −4.3 [−6.1, −0.8] cm.
    - 외부 CALM 소수 라벨 적응.
    - SHAP: 잔차 모델에서 레나는 지형 50%, 캐나다는 토양 39% 였다.
- **덱 v10(10-16)**
  - Files: `deck/deck_spec_v10.json`, 빌더 `deck/build_v9.py`, 납품 `deliverables/10_v10_2026-10-16/`(로컬).
  - Result: 28쪽(20분)이다. 그림 안 글자는 영문, 덱 문장은 국문 개조식으로 썼다.
- **원고 v10 → v11(10-16)**
  - Files: `paper/manuscript/en_v11/`, `ko_v11/`(main.tex, si.tex, sections/, build.sh, compliance.py, COMPLIANCE_v11.md), 매크로 `paper/manuscript/shared_v10/`(v6 · v9 · v10) + `shared_v11/numbers_v11.tex`(`scripts/2_evaluation/v11_numbers.py`), 계획 `docs/PAPER_PLAN_v11_2026-10-16.md`, 그림 `outputs/figures/paper/v11/`(Fig1–7) · `v11_si/`(S1–S10), 스크립트 `scripts/4_visualization/paper_v11/`.
  - 사용자 지적 10항을 반영하였다.
    - Methods 를 Results 앞에 두고, 소제목을 논문 범주로 바꾸었다.
    - LLM 진술을 제외하고, 수사 표현을 제거하였다.
    - Sci Rep 지침을 재확인하고, 그림을 본문 위치에 넣었다.
    - 무채색을 덱 색의 채도 낮춘 판(paper_v11)으로 바꾸었다.
    - 연구 흐름 그림 Fig. 2 를 신설하고, 제품 지도 비교 Fig. 6 을 본문으로 옮겼다.
    - 선행 연구 비교 절을 신설하고, 그래프와 Fig. 2 의 품질을 높였다.
  - Result:
    - 영문 22쪽 · SI 20쪽, 국문 20쪽 · 20쪽. 오류 0, 넘침 0.
    - 제목 14, 초록 199, 본문 3,921단어. 표시 항목 8개, 첫 제출 압축본 2.5 MB.
- **감사 · 커밋(10-16 ~ 10-18)**
  - Result:
    - 읽기 전용 감사 에이전트 2개(수치 · 근거, 구조 · 문서 정합)를 돌렸고, 차단 항목은 없었다.
    - 지적 사항은 모두 반영하였다(외부 비교 행 집합 명시, 채점 집합 805 셀-연도 · 72셀 · 49블록, 초록 · 결론의 실험 통일 등).
    - 커밋은 5bf8c9a, e1223d8, 44d3f95, 6098db0, dfcc94d, ebed13f, fbcbb64, dbcb552 이다.

## 4. Key numbers (this session)
| Method | Domain/Case | Metric | Value | Source artifact path |
|---|---|---|---|---|
| R1 − P1 | 레나 델타, 공간 블록, 관측 연도 입력 | ΔRMSE (cm) | −3.6 [−7.8, −0.8] (21.0 → 17.4) | `data/processed/v5/runs/core_v6/deltas_vs_ref.csv`, `summary_metrics.csv` |
| R1 − P1 | 캐나다 북서부, 공간 블록 | ΔRMSE (cm) | −3.7 [−4.8, −0.8] (30.6 → 26.9) | 같은 표 |
| R1 − P1 | 알래스카, 공간 블록 | ΔRMSE (cm) | −0.2 [−0.6, +0.1] (18.6 → 18.4) | 같은 표 |
| D0 | 알래스카 · 레나 · 캐나다, 공간 블록 | RMSE (cm) | 17.3 · 15.9 · 26.1 | `core_v6/summary_metrics.csv` |
| D0 − P0 | 레나 델타, 지역 홀드아웃 | ΔRMSE (cm) | +17.8 [+8.0, +23.9] (39.3 대 21.5) | `core_v6/deltas_vs_ref.csv` |
| R1 − P0 | 레나 · 캐나다, 지역 홀드아웃 | ΔRMSE (cm) | +2.0 [−0.8, +3.4] · −3.5 [−5.1, −0.1] | 같은 표 |
| 관측 연도 − 평년 입력 | 레나 P1 · 레나 R1 · 캐나다 R1 | ΔRMSE (cm) | −1.4 · −1.6 · −1.5(모두 판정됨) | `core_v6/deltas_forcing.csv` |
| R1S 전이 | 레나, 대상 라벨 0 → 25 | RMSE (cm) | 23.8 → 19.9(P0 21.5, D0S 39.5) | `data/processed/v5/runs/ladder_v6/ladder_metrics.csv` |
| R1S 전이 | 캐나다, 대상 라벨 0 → 25 | RMSE (cm) | 24.3 → 26.3(P1n 27.7 → 31.3, 악화) | 같은 표 |
| R1S 외부 적응 | 외부 CALM 7개 지역, 공통 474 셀-연도 | RMSE (cm) | 56.2(라벨 0) → 40.3(라벨 5), CCI v5 60.7, Wei 18.0, Liu 23.2 | `outputs/figures/paper/v7/source_data/F10_extft.csv` |
| R1S − CCI v5 | 외부, CCI v5 값 있는 700 셀-연도, 라벨 5 | ΔRMSE (cm) | −19.5 [−31.3, −9.2] | `data/processed/v5/runs/extra_v9/extft_deltas.csv` |
| R1 대 제품 | 레나 델타 공통 셀-연도(2010–2020) | RMSE (cm) | R1 21.1, CCI v5 28.4, Aalto 30.6, CCI v4 31.0, Liu 33.0, Wei 34.6 | `core_v6/common_cells_eval.csv` |
| R1 − Wei · R1 − CCI v5 | 레나, Wei 학습 지점 25 km 마스크 | ΔRMSE (cm) | −12.5 [−15.6, −5.8] · −5.0 [−9.5, −2.5] | `core_v6/products_eval_mask25.csv` |
| R1(OOF) · Wei · Liu | 알래스카 2014(95셀, 85셀이 Wei 학습 지점 25 km 안) · 레나 2016(14셀) | RMSE (cm) | 17.1 · 15.8 · 16.4 / 12.7 · 26.2 · 30.4 | `outputs/figures/paper/v11/source_data/Fig6_product_maps.json` |
| 지도 | 알래스카 · 레나, 2011–2024 | 적용 가능 영역 · 90% 예측 구간 반폭 | 56% · 62%, 25.5 · 24.5 cm | `docs/numbers_v6.json`(map 항목) |

## 5. Decisions made
- **셀-연도 설계와 관측일 누적 TDD 채택**: 극지연 피드백(입력 연도 혼재)의 직접 해결이다. 모든 예측이 명시된 해에 대응한다.
- **주 모델 R1(λ 0.5, CatBoost 600 · 0.03 · 6 · L2 3, 씨앗 3)**: 지역 홀드아웃에서 Stefan 수준을 유지하는 유일한 학습 구조이다. D0 · F1 · D1 은 대조로만 보고한다.
- **주 검증 = 0.5° 공간 블록, 전이 = 지역 홀드아웃(100 km 완충)**: 무작위 CV 는 D0 오차를 3.6–9.9 cm 과소 추정하므로 보조로만 쓴다.
- **원고 구성**: Introduction → Data and methods → Results → Discussion → Conclusions 이다. Sci Rep 은 구성 규정이 없고 첫 제출 그림을 본문에 넣을 수 있다(공식 지침 10-16 확인). LLM 진술은 사용자 요청으로 넣지 않았다.
- **그림 색**: 회색 위주(paper_v10)를 버리고 덱 색 체계의 채도 낮춘 판(paper_v11)을 쓴다. 토큰은 `design/style_tokens_v4.json`, 전환은 `POLAR_PALETTE=paper11` 이다.
- **저장소 정책(10-18)**: 원고 빌드, 전체 해상도 원고 PDF, SVG, 대체된 그림 판(v5–v7, 원천 표 제외), 1 MB 넘는 실행 표 47개는 .gitignore 에 두고 로컬에만 보관한다.

## 6. Open questions / blockers
- **LLM 사용 진술**: Sci Rep 지침은 Methods 기록을 요구하지만 사용자는 제외를 지시하였다. 제출 직전 사용자 결정이 필요하다.
- **저자 자리표시**: 교신 저자 전자우편, 극지연 공저자와 기여, 감사의 글, 자료 · 코드 저장소 DOI 가 비어 있다.
- **알래스카 관측 연도 입력 효과 미판정**: 라벨이 2014년에 몰려 있다(51%). 10-12 의 2014 제외 실행(`data/processed/v5/runs/ak_no2014`, 관측일 정합 이전 v5 입력)을 관측일 정합 입력으로 다시 돌릴지 정해야 한다.
- **캐나다 소수 라벨 재보정 악화**: 융해 깊이 범위가 넓은 지역에서 단일 계수가 맞지 않는다. 라벨 선택(공변량 범위 포괄) 규칙 시험이 필요한지 정해야 한다.
- **외부 소지역 악화**: 라벨 5개에서 추코트카 8.8 → 14.6, 캄차카 7.9 → 11.7, 북동 시베리아 32.6 → 34.8 cm 로 오차가 늘었다. 원고에는 명시하였다. 적응 규칙(예: 라벨 없는 오차가 작으면 재보정 생략)을 검토할 수 있다.
- **레나 2020년 관측 편차 +46.9 cm(143 셀-연도)**: 예측 편차 +10.0 cm 와 크게 다르다. 상관은 2020년을 빼도 0.72 로 유지된다. 관측 셀 구성 변화인지 확인이 필요하다.

## 7. Next steps (prioritized)
1. **원고 v11 검토와 극지연 공저자 회람**: owner user. 전제는 `deliverables/12_paper_v11_2026-10-16/`(로컬)의 PDF 이다. 1–2일.
2. **LLM 진술 · 저자 자리표시 결정**: owner user. 결정 뒤 Claude 가 front.tex 를 반영한다. 30분.
3. **GPT 비판 검토**: owner GPT. 초록 · 고찰 '선행 연구와의 비교' · 결론의 주장이 §4 수치와 맞는지, 과장 · 누락이 없는지 본다. 대상은 `paper/manuscript/en_v11/sections/{front,intro,discussion,conclusions}.tex` 이다.
4. **(선택) 보강 실험**: owner Claude. Rescale 에서만 실행한다(로컬 CPU · 타 사용자 GPU 금지). 각 1–2시간 이내.
   - (a) 알래스카 연도 균형 표본에서 관측 연도 입력 효과를 본다.
   - (b) 캐나다 라벨 선택 규칙을 시험한다.
   - (c) 외부 474 행 집합의 짝지은 블록 부트스트랩 구간을 구한다.
5. **제출 묶음**: owner Claude. 커버레터, 그림 별도 파일(180 mm, 600 dpi), SI 최종 번호 파일명, 원고 압축본(3 MB 이하)을 만든다. 1–3번 결정 뒤 진행한다.

## 8. Pointers
- Authoritative reports:
  - 영문 원고 `paper/manuscript/en_v11/main.tex`(빌드 `./build.sh`, 압축본 `main_submission.pdf` 추적), 보충 `en_v11/si.tex`, 국문 `paper/manuscript/ko_v11/`.
  - 규정 점검 `en_v11/COMPLIANCE_v11.md`.
  - 상태 `SESSION_HANDOFF.md`(10-16 저녁 절), 로그 `docs/EXPERIMENT_LOG.md`(맨 위 10-16 항목들).
- Active jobs: none(Rescale · 로컬 모두 종료).
- Ckpt paths: 해당 없음(CatBoost 는 실행마다 재적합, 모델 파일 미보관).
- Related prior handoffs: `gpt/handoff/20261008_1637-pre-reset-snapshot.md`, `20261002_1033-workflow-framing-lgf-wf-verdicts.md`.
- 수치 매크로 정본: `docs/numbers_v6.json`, `numbers_v9.json`, `numbers_v10.json`, `numbers_v11.json`(TeX 판은 `paper/manuscript/shared_v10/`, `shared_v11/`).

## 9. Caveats for GPT
- **방법 기호(v5 이후)**: P0 원천 계수 Stefan(다른 지역에서 적합한 Ê), P1 재보정 Stefan(대상 지역 라벨, 원점 통과 최소제곱, 원고 식 (4)), R1 물리 잔차, D0 직접 ML, F1 물리 입력 ML, D1 물리 유사라벨, R1S · D0S · P1n 은 전이 실험의 원천 + 대상 라벨 판이다. project.yaml 의 naming_canonical(10-08 작성, 'P1 수축 κ 10')은 재설정 전 정의이므로 v5 이후 수치에 쓰지 않는다.
- **용어**: '강제력 · forcing' 대신 '관측 연도 기후 입력(관측일까지 누적 TDD)', '평년 기후 입력(2011–2024 평균)'을 쓴다. 'significant' 는 P 값 없이 쓰지 않는다. 구간 판정은 'resolved' 로 쓴다.
- **직접 ML 은 '최선'이 아니다**: 지역 안 전량 라벨에서 R1 보다 점추정만 낮다. Stefan 대비 판정은 캐나다만 구간이 0 을 제외한다. 캐나다 지역 안 최저 점추정은 Kudryavtsev 25.4 cm 이다. 새 지역에서는 오차가 커진다.
- **두 실험을 섞지 않는다**: 지역 홀드아웃(100 km 완충, D0 39.3, R1 23.5)과 전이 실험 라벨 0개(D0S 39.5, R1S 23.8)는 다른 실행이다. 초록 · 결론은 전이 실험(D0S − P0 +18.0 [+6.9, +24.5]), 결과 · 고찰의 지역 홀드아웃 문장은 +17.8 을 쓴다.
- **행 집합**: 외부 CALM 수치는 474(세 제품 공통), 700(CCI v5 값 있음), 805(채점 전체, 6 m 스발바르 시추공 제외)행으로 갈린다. 원고 문장마다 어느 집합인지 적혀 있다.
- **그림 6 제품 값은 학습 지점 마스크가 없다**: 알래스카 95셀 중 85셀이 Wei 학습 지점 25 km 안이다. 마스크 비교는 그림 5와 보충 표 S6 에 있다.
- **보충 그림 번호**: 파일 이름(FigS4_external 등)은 작업 번호이고, 최종 번호는 si.tex 의 \sifig 순서(본문 첫 인용 순서)이다. 예를 들어 FigS4_external 은 S5, FigS10_label_ladder 는 S4 이다.
- **9월 이전 결과와 비교 금지**: XA–XM, LG, M1, H 계열은 다년 평균 라벨과 다른 입력을 썼다. 예를 들어 알래스카 13,606셀 · 74블록은 옛 집합이다.
