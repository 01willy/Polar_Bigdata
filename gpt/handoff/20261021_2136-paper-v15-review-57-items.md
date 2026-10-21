# Handoff: 논문 v15 — 사용자 전체 검토 57항 반영(본문 전면 개정, Fig 3 · 4 · 5 재구성, 외부 CALM 제거, 표 S7 · S8, 3차 제안)
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-10-21 21:36
**Session focus**: 사용자가 원고 전체를 검토해 남긴 57항(초록 · 서론 · 방법 · 결과 · 고찰 · 결론 · 표 · 그림 · 양식)을 사실 확인과 함께 반영해 v15 를 만들고, 커밋 · push · 인계까지 마쳤다.
**Author**: Claude Fable 5.1 + 01willy

---

## 1. TL;DR (≤5 bullets)
- 본문 전면 개정: 초록(관측 편중 · Stefan 한계 · ML 한계 · 제품 검증 공백 → 방법 → 결과 → 결론 문장, 200단어), 서론(두 경로와 제품; Liu 2024 · Wei 2026 은 연도별 예측 변수 · 지점-연도 라벨을 쓰므로 '다년 평균 학습' 서술 정정), 결과 5절(질문 순서, 정성 해석), 고찰 6문단, 결론(물리에서 출발 · 라벨이 가리키는 곳에 학습기 · 수십 개 라벨). 점검표 본문 4,498단어 통과.
- 그림 재구성: Fig 3 = 블록 CV 관측 대 예측 밀도 + 설계별 Δ; Fig 4 = 공변량 · 3분위 · 라벨 사다리 · 전이; Fig 5 = RMSE 점(직접 ML 제외) · 3 × 3 행렬 · 오차 분포; Fig 6 산점 분리 · RMSE 표 아래; Fig 1 Aalto 막대; Fig 2 글자 · 높이. 외부 CALM 지역은 본문 · 보충에서 전부 제거(사용자 결정).
- 사실 확인: 알래스카 이득 부재는 설계 · λ 와 무관(R1−P1 = −0.15 ~ +0.33)이고 10-06 발표의 1.1 cm 는 재설정 전 설계의 값; 전량 라벨 블록 CV 의 직접 ML 우위(≤ 1.6 cm)는 사실이나 셀-연도에서는 알래스카에서 역전; Liu 2024 는 ABoVE SMALT(이전 판) + CALM 으로 학습; 원천별 잔차(캐나다 CALM −23 cm); 지도 편차 ↔ 지역 TDD 상관 0.92–1.00.
- 보충: 표 S7(셀-연도 재채점), 표 S8(관측 프로그램별 잔차), S5 교체(연도별 입력 효과 + 연도별 RMSE), 보충 방법 S3 삭제 · 재번호. 방법 절 보강 11항(Stefan 선택 · E 관행 · λ · CatBoost · 부트스트랩 · AOA 50 % · 추세 · 계산 · 제품 영역 · Liu 학습 자료 · 같은 해 원칙), 표 1 영역 열.
- 3차 제안 3장(P3_G 학습 지점 거리별 오차, P7_E 선택 연도 편차, P1_B 범위 행렬). 커밋 7c52943(그림 · 스크립트 · 제안) · 2cad493(원고 v15) · de84ba3(문서), origin/main push 완료. 저자 메타(57항)는 다음 세션.

## 2. Context
- 직전 인계: `gpt/handoff/20261021_1238-paper-v14-1-fig6-years-cellyear-s7.md`(v14.1). 사용자가 v14.1 전체를 읽고 57항을 남겼고, GPU 가 전부 타 사용자 사용 중이라 무거운 작업은 Rescale 로 보내라고 지시했다(이번 세션은 로컬 경량 작업만).
- 점검표: `docs/REVIEW_RESPONSE_v15_2026-10-21.md`(항목별 결정 · 근거 · 상태).

## 3. What we did
- **Action**: 질문형 항목의 사실 확인(5 · 6 · 9 · 10 · 17 · 18 · 21 · 22 · 24 · 25 · 27 · 36 · 40 · 41). **Files**: `data/processed/v5/runs/core_v6/{summary_metrics,deltas_vs_ref}.csv`, `ladder_v6`, 제품 논문 PDF(Liu 2024 2.2절, Wei 2026 2.2–2.4절), `scripts/2_evaluation/v15_extra_numbers.py`(새 매크로: Ku · PE · λ · 학습기 범위 · 원천별 잔차 · TDD 편차 상관). **Result**: 4절 표.
- **Action**: 그림 v15. **Files**: `scripts/4_visualization/paper_v15/{fig1_data,fig2_workflow,fig3_accuracy,fig4_conditions,fig5_products_v15,fig6_product_maps,fig7_maps,si_s05_yearwise_input}.py`, `outputs/figures/paper/v15/`, `v15_si/`. **Result**: 본문 7장 QA 통과(Fig 5 는 회전 눈금 글자 경고만), SI 10장.
- **Action**: 원고 v15 영 · 국문. **Files**: `paper/manuscript/{en,ko}_v15/sections/*.tex`, `si.tex`, `table1.tex`, `references.bib`(Efron 1993, Grinsztajn 2022 추가), `shared_v15/`(numbers_v15, _cellyear, _extra, table_s7*, table_s8*). **Result**: 빌드 · 점검표는 8절 로그.
- **Action**: 3차 제안. **Files**: `scripts/4_visualization/paper_v15/proposals_v15_1.py`, `outputs/figures/paper/v15_proposals/`, `docs/FIGURE_PROPOSALS_v15_2026-10-21.md`.
- **Wallclock / GPU**: CPU 만(nice 10, 스레드 2). Fig 6 렌더 12분(해안선 재계산), Fig 7 약 15분, 나머지 수십 초.

## 4. Key numbers (this session)
| Method | Domain/Case | Metric | Value | Source artifact path |
|---|---|---|---|---|
| 물리 잔차 − 재보정 Stefan, 블록 CV | 알래스카, 기후 입력 평균 / 연도 / 관측일 | ΔRMSE cm | +0.33 / −0.40 / −0.15 | `data/processed/v5/runs/core_v6/deltas_vs_ref.csv` |
| 같은 비교, λ 0.25 / 0.5 / 1 | 알래스카 | ΔRMSE cm | −0.15 / −0.15 / +0.30 | 같은 파일 |
| 같은 비교, λ 1 | 레나 블록 / 레나 지역 홀드아웃 | ΔRMSE cm | −5.1 / +5.4 | 같은 파일 (`\ndlamOneLeblk`, `\ndlamOneLereg`) |
| Kudryavtsev(척도 보정) 대 재보정 Stefan, 블록 CV | 알래스카 / 레나 / 캐나다 | RMSE cm | 21.7 대 18.6 / 20.4 대 21.0 / 25.4 대 30.6 | `core_v6/summary_metrics.csv` |
| 물리 잔차, 원천별 블록 OOF | 알래스카 ABoVE 탐침 / CALM 격자; 캐나다 ABoVE / CALM; 레나 ALLena | RMSE (편향) cm | 18.0 (−3.5) / 27.2 (+0.1); 25.8 (+3.8) / 41.0 (−23.0); 17.4 (+0.5) | `outputs/figures/paper/v15/source_data/TableS8_residuals_by_source.csv` |
| 지도 지역 평균 편차 ↔ 지역 평균 TDD | 알래스카 / 레나 / 캐나다 | r | 0.92 / 0.96 / 1.00 | `outputs/figures/paper/v14/source_data/Fig7_tdd_anomaly_by_year.json` |
| 직접 ML − 물리 잔차, 블록 CV 전량 라벨 | 최대(레나) | RMSE 차 cm | 1.6 | `summary_metrics.csv` (`\ndrefRoDzblkmax`) |
| 학습 지점 거리별 블록 OOF RMSE(물리 잔차) | 알래스카 < 5 / 5–25 / 25–100 / > 100 km / 지역 홀드아웃 | cm | 10.0 / 14.6 / 21.5 / 22.6 / 17.5 | `outputs/figures/paper/v15_proposals/P3_G_error_vs_distance.json` |
| 원고 분량 | 영문 v15 | 초록 · 본문 | 200단어 · 4,498단어 | `paper/manuscript/en_v15/COMPLIANCE_v15.md` |

## 5. Decisions made
- 외부 CALM 지역(8개 소지역)은 자료가 적어 논문에서 제외(사용자). 자료 · 스크립트 · 결과는 보존.
- 그림 5 는 제품 비교에 집중하여 직접 ML 을 뺌; 직접 ML 과의 비교는 Fig 3 · Fig 4 · 표 S7 에서 다룸(사용자 40항).
- 초록에서 융해도일만 언급하던 것을 '기후 · 지형 · 토양 입력' 으로 바꿈(2항).
- 서론의 세 한계를 문헌에 맞게 정정(5항): Liu · Wei 는 연도별 예측 변수와 지점-연도 라벨을 쓰며, 공백은 '자기 지점 집합 안 검증 · 학습 지점 없는 지역의 특정 해 정확도 미평가'.
- 추론 문장은 '~로 추정된다/기대된다' 로 완화(33 · 44항). "처음이다" 표현 삭제(50항).
- Fig 7f(연도별 편차 14장)는 유지하되 선택 연도 4장(P7_E)을 대안으로 제시.
- 1 km 재집계 전면 재실행 없음; 보충 표 S7 로 대체(사용자 1항).

## 6. Open questions / blockers
- 3차 제안 선택: P3_G(거리별 오차)를 Fig 3b 대체 또는 추가 패널로 쓸지, P7_E 를 Fig 7f 대체로 쓸지, P1_B 를 Fig 1b–c 대체로 쓸지.
- 저자 메타(57항): 극지연 영문 소속, 저자 순서, 기여, 교신 전자우편, 감사의 글, 저장소 DOI.
- 본문이 한도 직전(4,498단어)이라 문장을 더하려면 다른 문장을 줄여야 함.
- 캐나다 CALM 지점(65 지점-연도, 평균 89 cm)의 −23 cm 편향: 규약 차이인지 모델 한계인지는 미확정(표 S8, 고찰 한계 문단에 서술).

## 7. Next steps (prioritized)
1. **사용자 검토** — v15 PDF(`deliverables/paper/manuscript_en.pdf`, 국문 `원고_국문.pdf`)와 점검표 `docs/REVIEW_RESPONSE_v15_2026-10-21.md` 대조.
2. **GPT 원고 비평** — owner: GPT. 집중: 초록 · 서론 2문단의 제품 서술이 문헌과 맞는지(Liu 2024 ERL, Wei 2026 ESSDD), 고찰 1 · 2문단 논리, 결론의 규칙 문장, 표 S8 해석.
3. **3차 제안 선택 반영** — owner: Claude, 반나절.
4. **저자 메타** — owner: user + 극지연.

## 8. Pointers
- Authoritative reports: `paper/manuscript/en_v15/`(main.tex, si.tex, sections/, COMPLIANCE_v15.md, main_submission.pdf), `paper/manuscript/ko_v15/`, 매크로 `paper/manuscript/shared_v15/`.
- 납품(서버 전용, 복사본): `deliverables/paper/`, `deliverables/figure_proposals/`(2차 · 3차), `deliverables/reports/`.
- 그림: `outputs/figures/paper/v15/`, `v15_si/`, `v15_proposals/`; 스크립트 `scripts/4_visualization/paper_v15/`.
- 기록: `docs/REVIEW_RESPONSE_v15_2026-10-21.md`, `docs/EXPERIMENT_LOG.md`(2026-10-21 저녁), `SESSION_HANDOFF.md` 최신 절, `docs/FIGURE_AUDIT_v12_2026-10-18.md` 15절.
- Active jobs: none. Ckpt paths: 해당 없음.
- Related prior handoffs: `gpt/handoff/20261021_1238-paper-v14-1-fig6-years-cellyear-s7.md`, `20261020_1740-paper-v14-truth-maps-scirep.md`.

## 9. Caveats for GPT
- 외부 CALM 지역 결과(v14 까지의 Fig 4c–e, S5)는 v15 원고에 없다. 관련 매크로(`\nft*`, `\next*`)는 정의돼 있지만 본문에서 쓰지 않는다.
- Fig 3 · 4 · 5 의 패널 글자가 v14 와 다르다(Fig 3 a · b, Fig 4 a–d, Fig 5 a–c). v14 인계 문서의 패널 참조는 v15 에 맞지 않는다.
- 직접 ML 이 전량 라벨 블록 CV 에서 물리 잔차보다 낮은 점 오차를 보이는 것은 사실이며 본문에 적혀 있다. 논문의 주장은 '물리 잔차가 항상 가장 정확' 이 아니라 '라벨이 드문 곳과 새 지역에서 안전하고, 라벨이 많아도 ≤ 1.6 cm 안에서 따라간다' 이다.
- 알래스카 2013 Fig 6 의 낮은 상관은 제품 학습 지점 25 km 안 셀에서만 나며(v14.1 인계 참조), v15 본문도 같은 서술이다.
- 방법 기호: P0 원천 계수 Stefan, P1 재보정 Stefan, R1 물리 잔차 결합(λ 0.5), D0 직접 ML, Ku Kudryavtsev 척도 보정, PE 물리 앙상블.
