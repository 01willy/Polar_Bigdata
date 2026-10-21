# Handoff: 논문 v14.1 — Fig 6 지역별 연도 · 학습 지점 거리 구분, 보충 표 S7(셀-연도 재채점), 저자 표기, 그림 2차 제안, 납품 폴더 재구성
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-10-21 12:38
**Session focus**: 사용자 질문 2건(알래스카 2013 상관, 1 km 재집계 필요성)에 근거 분석으로 답한 뒤, 7항 지시(보충 표, Fig 6 연도 기준, 지도 관행, 제안 판정, 저자, 2차 제안, 폴더 정리)를 반영해 v14.1 을 만들고 커밋 · push 까지 마쳤다.
**Author**: Claude Fable 5.1 + 01willy

---

## 1. TL;DR (≤5 bullets)
- 알래스카 2013 의 낮은 상관(0.34 대 0.53 · 0.57)은 2013 특유가 아니라 알래스카 전반의 양상이고, 제품 학습 지점 25 km 안 셀에서만 난다. 밖 67셀에서는 세 지도가 같다(9.1 · 10.9 · 8.5 cm). Fig 6 은 지역별 '관측 셀 최다 해'(알래스카 2013, 레나 2016, 캐나다 2013)와 25 km 안 · 밖 표지 구분으로 재설계했다.
- 1 km 셀 재집계 전면 재실행은 하지 않는다(사용자 판단). 대신 기존 겹 밖 예측을 셀-연도로 다시 채점한 보충 표 S7 을 넣었다. 레나 · 캐나다 블록 이득과 레나 전이 벌점은 셀 단위에서도 유지되고, 캐나다 지역 홀드아웃의 물리 잔차 이득만 셀 단위에서 미결이다.
- 선행 연구 조사(41편): 주 지도는 다년 평균, 추세는 별도 그림, 연도별 편차 지도는 선례 없음. Fig 6 · 7a–e 는 관행 안에 있고 7f(연도별 편차 14장)는 보충 이동을 권고(미적용, 사용자 선택).
- 그림 2차 제안 8장 렌더(`deliverables/figure_proposals/`): 권장 Fig 3 F(세 질문 배치), Fig 5 D(관측 대 예측 3 × 3 행렬). 사양 곡선 기각 · 표적 도식 보류는 기록.
- 저자 블록(백승원 · 김승희 · 정윤택 · 조용채 교신) 반영. 원고 v14.1: 영문 25쪽 · 국문 22쪽 · 보충 21쪽, 점검표 전 항목 통과(본문 4,442단어). 납품 폴더를 종류별로 재구성. 커밋 ce8e337(그림 · 재채점 · 제안) · 697e415(원고 v14.1) · 14a49bd(도구 · 문서), origin/main 에 push 완료.

## 2. Context
- 직전 인계: `gpt/handoff/20261020_1740-paper-v14-truth-maps-scirep.md`(v14 완성, 커밋 로컬, push 대기). 사용자가 10-21 아침에 두 질문을 했고(알래스카 2013 상관, 재집계 이유), 답변 뒤 7항 지시와 함께 커밋 · push 를 승인했다.
- 분석 근거: 연도별 · 거리별 공통 셀 재계산(스크래치 `fig6_yearwise_common_cells.csv`; 본문 미사용), 셀-연도 재채점(`scripts/2_evaluation/v14_cellyear_rescore.py`).

## 3. What we did
- **Action**: Fig 6 재설계. **Files**: `scripts/4_visualization/paper_v14/fig6_product_maps.py`(YEARS 사전, FAR_KM 25, 통계 글 모서리 자동 선택), `outputs/figures/paper/v14/Fig6_product_maps.png`, `source_data/Fig6_product_maps.json`; 본문 · 캡션 `paper/manuscript/{en,ko}_v14/sections/results.tex`, 보충 표 S2 행, 매크로 `\nfe*Far*`(`scripts/2_evaluation/v14_numbers.py`). **Result**: QA 통과(텍스트 104, 겹침 0), 산점 통계 글이 표지와 겹치는 수 알래스카 1 · 레나 0 · 캐나다 3.
- **Action**: 보충 표 S7. **Files**: `scripts/2_evaluation/v14_cellyear_rescore.py` → `outputs/figures/paper/v14/source_data/TableS7_cellyear_{rescore,deltas}.csv`, `TableS7_cellyear_meta.json`, `paper/manuscript/shared_v14/table_s7{a,b}_cellyear_{en,ko}.tex`, `numbers_v14_cellyear.tex`; `si.tex`(en · ko) 표 S7, `methods.tex` 한 문장, `front.tex` S1–S7. **Result**: 4절 표.
- **Action**: 저자 · 소속 · 기여. **Files**: `paper/manuscript/{en,ko}_v14/sections/front.tex`. **Result**: 교신 전자우편 · 감사의 글 · 저장소 DOI 는 자리표시로 남음. 극지연 영문 소속은 두 사람의 최근 논문 표기 'Center of Remote Sensing and GIS'(국문 원격탐사빙권정보센터)로 두었고 확인이 필요하다.
- **Action**: 선행 연구 조사(에이전트). **Files**: `docs/research/2026-10-21/map_and_figure_practice.md`. **Result**: 2절 · 3절 표, Fig 6 · 7 시사점.
- **Action**: 그림 2차 제안. **Files**: `scripts/4_visualization/paper_v14/proposals_v14_1.py`, `outputs/figures/paper/v14_proposals/P*_*.{png,pdf,svg,json}`, `docs/FIGURE_PROPOSALS_v14_1_2026-10-21.md`. **Result**: 8장(P3_D · E · F, P4_E · F, P5_D · F, P7_D), 유형별 권장안.
- **Action**: 납품 · 폴더 재구성. **Files**: `tools/sync_deliverables.sh`(v2), `tools/finish_paper_v14_1.sh`, `00_START_HERE.md`, `README.md`. **Result**: `deliverables/{paper,figure_proposals,deck,reports,references,archive}`.
- **Wallclock / GPU**: CPU 만(nice 10, 스레드 2). Fig 6 렌더 72 s × 3, 재채점 6 s, 빌드 약 5 분 × 2.

## 4. Key numbers (this session)
| Method | Domain/Case | Metric | Value | Source artifact path |
|---|---|---|---|---|
| 물리 잔차(블록 OOF) / Wei / Liu | 알래스카 2013, 공통 셀 126 (25 km 안 59 · 밖 67) | RMSE cm 전체 · 밖 | 14.9 · 9.1 / 13.4 · 10.9 / 12.9 · 8.5 | `outputs/figures/paper/v14/source_data/Fig6_product_maps.json` |
| 같은 셋 | 알래스카 2013 | r 전체 · 밖 | 0.34 · 0.30 / 0.53 · 0.27 / 0.57 · 0.44 | 같은 파일 |
| 같은 셋 | 레나 델타 2016, 공통 셀 14(전부 밖) | RMSE cm · r | 12.7 / 26.2 / 30.4 · 0.47 / 0.29 / −0.21 | 같은 파일 |
| 같은 셋 | 캐나다 2013, 공통 셀 33 (밖 27) | RMSE cm 전체 · 밖 | 23.7 · 17.5 / 29.9 · 31.4 / 20.2 · 21.0 | 같은 파일 |
| 물리 잔차 − 재보정 Stefan, 블록 CV | 레나 / 캐나다 / 알래스카 | ΔRMSE cm 지점-연도 → 셀-연도 [95 %] | −3.6 → −3.6 [−7.4, −0.8] / −3.7 → −2.5 [−3.9, −1.4] / −0.2 → 0.0 [−1.8, +1.2] | `outputs/figures/paper/v14/source_data/TableS7_cellyear_deltas.csv` |
| 직접 ML − 원천 Stefan, 지역 홀드아웃 | 레나 | 같은 | +17.8 → +17.7 [+12.4, +24.8] | 같은 파일 |
| 물리 잔차 − 원천 Stefan, 지역 홀드아웃 | 캐나다 | 같은 | −3.5 [−5.2, −0.2] → −0.2 [−2.3, +1.8] | 같은 파일 |
| 관측의 셀-연도 안 합동 SD | 알래스카 / 레나 / 캐나다 | cm | 13.0 / 12.7 / 17.5 | `TableS7_cellyear_meta.json` |
| 원고 분량 | 영문 v14.1 | 본문 · 캡션 · 보충 쪽 · 압축본 | 4,442단어 · 70–164단어 · 21쪽 · 1.0 MB | `paper/manuscript/en_v14/COMPLIANCE_v14.md` |

## 5. Decisions made
- Fig 6 연도 기준 = 지역별 '세 지도 모두 값이 있는 관측 셀 최다 해'. 우리가 유리한 해를 고르지 않고 관측 기반이 가장 넓은 해를 쓴다. 알래스카는 이 기준으로도 2013 이다.
- 공정성은 연도 선택이 아니라 '제품 학습 지점 25 km 안 · 밖' 구분으로 보인다(Fig 5a 와 같은 기호).
- 1 km 재집계 전면 재실행 안 함. 채점 단위의 차이이며 보충 표 S7 로 충분(사용자 판단). 캐나다 지역 홀드아웃 문장은 본문에 수치가 없어 손대지 않았고 표 S7 이 미결을 보여 준다.
- Fig 7f(연도별 편차 지도)는 선례가 없어 보충 이동을 권고하되 적용은 사용자 선택으로 남김.
- 표적 도식 · Taylor 도식은 본문에 쓰지 않는다(조사 결과와 사용자 보류).
- 납품 폴더는 종류별 · 현재 판만. 옛 묶음은 archive/.

## 6. Open questions / blockers
- 그림 2차 제안 선택(Fig 3 F, Fig 5 D, Fig 7 7-1 권장). 선택 시 본문 그림 규격으로 다시 그리고 캡션 · 매크로 · 빌드를 다시 돌린다(`tools/finish_paper_v14_1.sh`).
- 극지연 영문 소속 표기 · 저자 순서 · 기여 문장 · 교신 전자우편 · 감사의 글 · 저장소 DOI 확인.

## 7. Next steps (prioritized)
1. **그림 제안 선택** — owner: user. `deliverables/figure_proposals/` 와 `docs/FIGURE_PROPOSALS_v14_1_2026-10-21.md`.
2. **GPT 원고 비평** — owner: GPT, 1시간. `deliverables/paper/manuscript_en_submission_compressed.pdf`(저장소에는 `paper/manuscript/en_v14/main_submission.pdf`). 집중: 결과 절 Fig 6 문단(25 km 구분 서술), 방법 절 셀-연도 문장, 표 S7 캡션, 고찰 둘째 문단.
3. **선택 반영 → v15** — owner: Claude, 반나절.
4. **제출 메타 확정** — owner: user + 극지연.

## 8. Pointers
- Authoritative reports: `paper/manuscript/en_v14/`(main.tex, si.tex, sections/, COMPLIANCE_v14.md, main_submission.pdf), `paper/manuscript/ko_v14/`, 매크로 `paper/manuscript/shared_v14/`.
- 납품(서버 전용, 복사본): `deliverables/paper/`, `deliverables/figure_proposals/`, `deliverables/reports/`.
- 그림: `outputs/figures/paper/v14/`, `v14_si/`, `v14_proposals/`; 스크립트 `scripts/4_visualization/paper_v14/`.
- 기록: `docs/EXPERIMENT_LOG.md`(2026-10-21 13:30), `SESSION_HANDOFF.md` 최신 절, `docs/FIGURE_AUDIT_v12_2026-10-18.md` 14절.
- Active jobs: none. Ckpt paths: 해당 없음.
- Related prior handoffs: `gpt/handoff/20261020_1740-paper-v14-truth-maps-scirep.md`, `20261018_1904-reset-cellyear-paper-v11.md`.

## 9. Caveats for GPT
- Fig 6 의 해는 지역마다 다르다(알래스카 · 캐나다 2013, 레나 2016). 레나 2013(6셀) 수치는 더 이상 본문에 없다.
- 산점도의 RMSE 두 값은 '전체 | Wei 학습 지점 25 km 밖'이다. 제품 학습 지점은 Wei 2026 기준이며 Liu 2024 의 학습 지점은 별도 마스크가 없다(캐나다 관측의 이전 판이 Liu 학습에 포함됨을 본문에 적음).
- 보충 표 S7 의 지점-연도 Δ 점추정은 본문 값과 같고 구간은 같은 방법(B 2000)으로 다시 뽑은 값이라 소수점이 다를 수 있다.
- 라벨 단위는 지점-연도, 셀-연도는 채점 단위일 뿐이다(재학습 없음).
- 방법 기호: P0 원천 계수 Stefan, P1 재보정 Stefan, R1 물리 잔차 결합(λ 0.5), D0 직접 ML. 옛 단계 이름(S1–S14, M1, H18–H30, LG, WF)은 쓰지 않는다.
