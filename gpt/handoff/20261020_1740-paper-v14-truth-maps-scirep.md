# Handoff: 논문 v14 — 지도 참값 비교 강조 · Fig 3 재설계 · Sci Rep 양식 정비 · 자료 흐름 감사
**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-10-20 17:40
**Session focus**: 원고 v13.2 를 v14 로 올리며 사용자 지시 12항(지도의 참값 비교, 그림 대안, 수사 제거와 수식 보강, 연도별 입력 관리 재점검, Sci Rep 양식, 캡션 축약, 의의 서술, 고찰 소제목, 새 지역 절차, 결론 집중, 수치 일치)을 반영하고 /cleanup 까지 마쳤다.
**Author**: Claude Fable 5.1 + 01willy

---

## 1. TL;DR (≤5 bullets)
- 원고 v14 완성(`paper/manuscript/{en,ko}_v14/`): 영문 25쪽 · 국문 22쪽 · 보충 20쪽, LaTeX 오류 0, Sci Rep 점검표 전 항목 통과(초록 200단어, 본문 4,403단어, 캡션 70–137단어, 압축 제출본 0.98 MB). 근거 `paper/manuscript/en_v14/COMPLIANCE_v14.md`.
- 지도 그림을 참값 비교 중심으로 재설계: Fig 6 은 2013년 × 3지역에서 관측 셀을 관측 ALT 색으로 채우고 제품 지도를 우리 셀에 마스크하며 지역별 색 범위(적용 가능 영역 안 세 지도의 5–95 백분위수)를 쓴다. 관측 대 예측 산점(1:1 선, r · RMSE)을 오차 벌떼 대신 넣었다. Fig 7 도 관측 셀을 관측 평균 ALT 로 채운다.
- Fig 3 을 숲 그림(연도 입력 − 평균 입력 ΔRMSE, 95 % 구간) + 설계별 점과 'R1 − P1' Δ 띠로 바꿨고, 그림 유형 4종 × 3안의 대안을 실제 자료로 렌더해 사용자 선택을 기다린다(`docs/FIGURE_PROPOSALS_v14_2026-10-20.md`, `outputs/figures/paper/v14_proposals/`).
- 자료 흐름 감사에서 "2013 은 세 지역 모두 관측이 있는 유일한 해" 가 거짓(그런 해는 12개)으로 드러나 "그중 알래스카 · 캐나다 관측 셀이 가장 많은 해" 로 정정했고, S5 범위 정의 · S6c · S10 을 9월 1일 기준 지도와 시계열로 재계산했다. 연도 · 관측일 정합과 매크로 263개의 원천 일치는 확인됐다.
- 커밋 4건 + 인계 1건을 main 에 로컬 커밋했다(data 2ec53a2, figures 1736ce3, paper 074c5f0, docs 14b6561). push 는 사용자 승인 전이라 하지 않았다.

## 2. Context
- 직전 인계: `gpt/handoff/20261018_1904-reset-cellyear-paper-v11.md`(10-10 재설정부터 원고 v11 까지). 그 뒤 10-18 저녁 ~ 10-20 오후에 그림 양식 v12 → v13(글자 위계 · 가는 선 · 채도 낮춘 색) → v13.1(3차 지적 6항) → v13.2(4차 지적 8항, 지점-연도 용어, 9월 1일 TDD 지도, 캐나다 1 km 지도)가 진행됐고 전부 미커밋이었다. 이번 세션의 커밋이 그 전부를 담는다.
- 이번 세션의 동기: 사용자가 자리를 비우며 남긴 12항 지시(요약은 `SESSION_HANDOFF.md` 최신 절). 핵심은 (i) 우리 지도와 제품 지도가 색 눈금 때문에 실제보다 더 달라 보이는 문제, (ii) 변화가 적은 그림(Fig 3)의 의미 강화, (iii) 과한 수사 · 과한 상세 제거와 Methods 수식 보강, (iv) 논문 · 그림 · 실험 수치 일치의 전면 재점검, (v) Sci Rep 제출 양식 확인이다.

## 3. What we did
- **Action**: Fig 6 재설계(2013 × 알래스카 · 레나 · 캐나다). 관측 셀을 관측 ALT 색으로 채운 원(지도와 같은 눈금), Wei 2026 · Liu 2024 제품을 우리가 그린 셀에 마스크, 지역별 색 범위 = 적용 가능 영역 안 세 지도 픽셀의 5–95 백분위수(5 cm 단위), 패널 오른쪽에 관측 대 예측 산점(1:1 선, 계열별 r · RMSE).
  **Files**: `scripts/4_visualization/paper_v14/fig6_product_maps.py`, `outputs/figures/paper/v14/Fig6_product_maps.png`, `outputs/figures/paper/v14/source_data/Fig6_product_maps.json`.
  **Result**: 세 지도가 같은 영역 · 같은 눈금에 놓여 "제품이 더 어둡다" 는 인상이 산지 극값을 빼면 사라진다. 수치는 4절.
- **Action**: Fig 7(평균 지도 · 확대창)의 관측 셀을 2011–2024 관측 평균 ALT 로 채움. Fig 3 을 숲 그림 + 설계별 점(연결선 없음) + Δ 띠로 재설계.
  **Files**: `scripts/4_visualization/paper_v14/fig7_maps.py`, `fig3_accuracy.py`; `outputs/figures/paper/v14/Fig7_maps.png`, `Fig3_accuracy.png`.
  **Result**: QA(글꼴 FreeSans 6.5 / 7 / 8 pt, 겹침 0, 캔버스 안) 통과. Fig 3 의 모든 패널에 불확실성 구간이 생겼고 범주 축의 잘못된 연속선을 없앴다.
- **Action**: 그림 유형별 대안 3안 설계 · 렌더(Fig3_B 사양 곡선 61개 사양, Fig3_C 전이 벌점 산점, Fig4_C 절대 RMSE 소형 다중, Fig5_B 표적 도식; Fig 4_B · 5_C · 6_B · 6_C 는 설계만).
  **Files**: `scripts/4_visualization/paper_v14/fig3_proposals.py`, `fig45_proposals.py`; `outputs/figures/paper/v14_proposals/`; `docs/FIGURE_PROPOSALS_v14_2026-10-20.md`.
  **Result**: v14 본문에는 권장안(Fig 3 A, Fig 6 A, Fig 7 A, Fig 4 · 5 현행)을 적용. 다른 안은 사용자 선택 대기.
- **Action**: 원고 본문 전면 개정(영 · 국문 쌍둥이). 서론 · 결과 축약, 고찰을 소제목 없는 6문단(기제 · 제품 · 선행 연구 · 지도와 새 지역 · 한계 · 전망)으로, 결론은 이 연구 고유 결과 3가지 + 산출물로 축소, 새 지역 적용은 "고정 절차" 가 아니라 "기대할 수 있는 바" 로 완화, 캡션은 제목 문장 + 패널 요지만(70–137단어), Methods 수식 13개(새로 (6) 대안 구조, (9) 전이, (12) 추세, (13) 예측 구간), 초록 200단어.
  **Files**: `paper/manuscript/en_v14/sections/*.tex`, `ko_v14/sections/*.tex`, `si.tex`, `table1.tex`.
  **Result**: 빌드 오류 0 · 미정의 참조 0. Sci Rep 규칙(본문 4,500단어 이하, 초록 200 이하, 표시 항목 8 이하, 캡션 350 이하, 고찰 소제목 없음 권장, 첫 제출 단일 파일 3 MB 이하) 전부 통과.
- **Action**: 자료 흐름 · 수치 감사(에이전트 1건 + /cleanup 감사 2건)와 반영.
  **Files**: `docs/AUDIT_DATAFLOW_NUMBERS_v14_2026-10-20.md`, `docs/AUDIT_{EXPERIMENT_SETTINGS,FIGURE_CONTROL,TEXT_NUMBERS}_2026-10-20.md`; `scripts/2_evaluation/v5_site_series.py`(`--ref-doy 244` 추가), `v14_numbers.py`(편향을 셀 목록에서 계산, Fig 6 r · 색 범위 매크로), `si_s06_training_sites.py`, `si_s10_site_series.py`.
  **Result**: P0 · P1 · P2 지적 전부 반영. 정정 목록은 `SESSION_HANDOFF.md` 최신 절 "감사 반영" 항목.
- **Action**: 매크로 재생성 · 양 원고 재빌드 · 납품 묶음 · 압축 제출본(그림 600 dpi PNG → 220 dpi) · /cleanup 커밋 5건.
  **Files**: `tools/finish_paper_v14.sh`, `tools/build_submission_png.sh`(VER=v14), `tools/package_paper_v14.sh`, `deliverables/15_paper_v14_2026-10-20/`(gitignore 대상, 서버에만 있음), `paper/manuscript/en_v14/main_submission.pdf`(추적).
  **Wallclock / GPU**: CPU 만(nice 10, 스레드 2), GPU 미사용.

## 4. Key numbers (this session)
| Method | Domain/Case | Metric | Value | Source artifact path |
|---|---|---|---|---|
| 물리 잔차 R1(공간 블록 OOF) / Wei 2026 / Liu 2024 | 알래스카 2013, 공통 1 km 셀 126 | RMSE cm · r | 14.9 / 13.4 / 12.9 · 0.34 / 0.53 / 0.57 | `outputs/figures/paper/v14/source_data/Fig6_product_maps.json` |
| 같은 셋 | 레나 델타 2013, 공통 셀 6 | RMSE cm · r · 편향 cm | 16.6 / 32.3 / 29.6 · 0.86 / −0.80 / −0.77 · +0.2 / +19.8 / +20.7 | 같은 파일 |
| 같은 셋 | 캐나다 2013, 공통 셀 33 | RMSE cm · r · 편향 cm | 23.7 / 29.9 / 20.2 · 0.10 / 0.41 / 0.59 · −4.0 / +20.5 / +7.8 | 같은 파일 |
| Fig 6 색 범위(적용 가능 영역 안 5–95 백분위) | 알래스카 / 레나 / 캐나다 | cm | 45–80 / 35–70 / 55–110 | 같은 파일 `colour_range_cm` |
| 물리 잔차 R1 − 재보정 Stefan P1(공간 블록) | 레나 / 캐나다 | ΔRMSE cm | −3.6 / −3.7 | `paper/manuscript/shared_v10/numbers_v6.tex`(`\ndrefLeblkRoabs`, `\ndrefCablkRoabs`; 원천 `results/` v6 표) |
| 직접 ML D0 − Stefan(지역 홀드아웃) | 레나 | ΔRMSE cm | +17.8 | 같은 파일 `\ndrefLeregDzabs` |
| 제품 vs 관측(지점-연도) | 레나, Wei / Liu | r | 0.05 / 0.07 | `paper/manuscript/shared_v14/numbers_v14.tex`(`\nrLeWei`, `\nrLeLiu`; Fig 5 b JSON) |
| 연도별 지도 2011–2024 선형 추세(모델 응답) | 알래스카 / 레나 / 캐나다 | cm/10년 | +2.1 / +6.7 / +1.5 | `numbers_v14.tex`(`\ntrend*`; `outputs/figures/paper/v14/source_data/Fig7_maps.json`) |
| 원고 분량 | 영문 v14 | 초록 · 본문 · 캡션 · 압축본 | 200단어 · 4,403단어 · 70–137단어 · 0.98 MB | `paper/manuscript/en_v14/COMPLIANCE_v14.md` |

주의: 2013 레나는 공통 셀 6개라 r 의 부호가 셀 몇 개에 좌우된다. 본문에서는 레나 제품 비교를 Fig 5(지점-연도 전체)로 하고 Fig 6 은 지도 수준 예시로만 쓴다.

## 5. Decisions made
- 색 범위를 지역별 5–95 백분위수로 정함. 한 색 범위(v13.2)는 레나(35–70 cm)와 캐나다(55–110 cm)를 한 눈금에 넣어 지도가 평평해지거나 포화됐다. 대가로 지역 간 절대값 비교는 색 막대를 읽어야 하며, 캡션에 명시했다.
- 제품 지도를 우리 셀에 마스크함. 비교 대상 영역이 같아야 "제품이 더 어둡다" 는 인상이 영역 차이에서 오지 않는다.
- Fig 6 의 해 선택 근거를 "유일한 해" 에서 "세 지역 모두 관측이 있는 12개 해 가운데 알래스카 · 캐나다 관측 셀이 가장 많은 해" 로 정정(감사 P0).
- 고찰 소제목 제거. Sci Rep 은 고찰에 소제목을 권장하지 않고 "선행 연구와의 비교" 식 제목은 관행이 아니다. 비교는 고찰 셋째 문단으로 녹였다.
- 새 지역 적용은 절차가 아닌 기대치로 서술. 외부 CALM 8개 지역과 레나 결과가 n=3 이상 라벨에서만 뒷받침되므로 단계별 지침은 위험하다는 사용자 판단을 따랐다.
- 결론은 고유 결과 3가지(조건부 이득, 물리 기준값의 전이성 보존, 같은 셀 · 같은 해 채점에서 제품 정확도가 학습 지점 위치를 따름)와 산출물만 적는다. Stefan 의 일반적 타당성이나 공간 CV 과대평가 같은 기존 확립 사실은 고찰에 둔다.
- 저장소: v14 · v14_si · v14_proposals PNG 만 추적, v12 · v13 PNG 는 제외(스크립트 · source_data 로 재생성). `_cache` 링크 제외. 압축 제출본은 v14 만 추적.

## 6. Open questions / blockers
- 그림 대안 선택: Fig 3 B(사양 곡선)를 보충 S2 대체로 넣을지, Fig 5 B(표적 도식)를 a 패널 대체로 쓸지. 사용자 판단 사항. 근거 렌더는 `outputs/figures/paper/v14_proposals/`.
- 1 km 재집계 재실행 여부(감사 결정 10-20 오후): 라벨 loc_id 는 지점(0.0001°)이지 1 km 셀이 아니다. 본문은 '지점-연도' 로 고쳤고 표 1 에 1 km 셀 수(321 · 143 · 71)를 병기했다. 셀 단위로 라벨을 재집계해 전부 재실행할지는 사용자 결정. 재실행 시 모든 수치가 바뀐다.
- 알래스카 2013 공통 셀에서 우리 지도의 r(0.34)이 제품(0.53 · 0.57)보다 낮다. RMSE 는 동급이다. 원인 후보는 공간 블록 OOF 예측(제품은 학습 지점을 포함한 지도)과 산지 극값. 본문은 사실만 적었고 해석은 고찰 둘째 문단의 한 문장에 그친다. 리뷰어가 물을 수 있는 지점이다.
- 저자 자리표시 · LLM 사용 진술 · 자금 · 데이터 가용성 URL 은 미정(사용자 · 극지연 결정).
- push 미실행. main 이 origin/main 보다 5 커밋 앞서 있다.

## 7. Next steps (prioritized)
1. **그림 대안 결정** — owner: user, 10분. `docs/FIGURE_PROPOSALS_v14_2026-10-20.md` 표에서 유형별 안을 고르면 Claude 가 해당 스크립트 변형을 본문 그림으로 올리고 캡션 · 매크로 · 빌드 · 납품을 다시 돌린다(`tools/finish_paper_v14.sh`, 30분).
2. **GPT 원고 비평** — owner: GPT, 1시간. 읽을 파일: `paper/manuscript/en_v14/main_submission.pdf`(압축본, 추적됨) 또는 `en_v14/sections/*.tex`. 집중 요청: 고찰 둘째 · 넷째 문단(제품 비교와 새 지역 기대치)의 주장 강도, 결론의 고유성, Methods 수식 (6) · (9) 의 표기 일관성, 캡션 축약으로 빠진 정보가 본문에 있는지.
3. **1 km 재집계 재실행 결정** — owner: user. 결정 시 Claude 가 `scripts/3_deep_learning/xbatch_core.py` 계열로 전량 재실행(Rescale 또는 로컬 CPU, 수 시간)하고 매크로 체인 전체를 다시 돌린다.
4. **push 승인** — owner: user. 승인 시 `git push origin main` 후 `git rev-parse HEAD` 와 `@{u}` 일치 확인.
5. **제출 메타** — owner: user + 극지연. 저자 · 소속 · 기여 · 자금 · LLM 진술 · 데이터 URL 확정 뒤 `sections/front.tex` 반영.

## 8. Pointers
- Authoritative reports: `paper/manuscript/en_v14/`(main.tex, si.tex, sections/, COMPLIANCE_v14.md, main_submission.pdf), `paper/manuscript/ko_v14/`, 매크로 `paper/manuscript/shared_v14/numbers_v14.tex`(생성 `scripts/2_evaluation/v14_numbers.py`), 납품 `deliverables/15_paper_v14_2026-10-20/`(서버 전용).
- 그림: `outputs/figures/paper/v14/`(본문 7장 + source_data JSON), `v14_si/`(보충 10장), `v14_proposals/`(대안 4장), 스크립트 `scripts/4_visualization/paper_v14/`.
- 감사: `docs/AUDIT_DATAFLOW_NUMBERS_v14_2026-10-20.md`, `docs/AUDIT_{EXPERIMENT_SETTINGS,FIGURE_CONTROL,TEXT_NUMBERS}_2026-10-20.md`, 그림 감사 누적 `docs/FIGURE_AUDIT_v12_2026-10-18.md`(12–13절).
- 세션 기록: `docs/EXPERIMENT_LOG.md`(2026-10-20 16:58 항목), `SESSION_HANDOFF.md` 최신 절.
- Active jobs: none(이 프로젝트의 학습 · 추론 프로세스 없음).
- Ckpt paths: 해당 없음(gradient boosting 모델은 실행 시 재학습, 지도는 `data/processed/v5/maps/{Alaska,Lena,Canada}/`).
- Related prior handoffs: `gpt/handoff/20261018_1904-reset-cellyear-paper-v11.md`, `20261008_1637-pre-reset-snapshot.md`.

## 9. Caveats for GPT
- 라벨 단위는 '지점-연도'(site-year)다. 10-18 인계의 '셀-연도' 는 용어 오류였고(loc_id 는 0.0001° 지점), 셀 수는 표 1 에 병기만 한다. 수치(19,172 지점-연도 등)는 그대로다.
- 지도 입력 TDD 는 9월 1일(day 244)까지 누적이다. 10-16 이전 지도(전체 계절 누적)는 `data/processed/v5/maps_fullseason_2026-10-14/` 로 밀려났고 본문 지도 수치는 모두 day 244 기준이다. 라벨 쪽 TDD 는 관측일까지 누적(변경 없음).
- Fig 6 수치는 2013년 한 해, Fig 5 수치는 지점-연도 전체다. 같은 제품의 RMSE 가 두 그림에서 다른 것은 표본이 달라서다.
- 공개 제품은 '채점 5종 + Yi–Kimball 참고(미채점)' 로 표 1 에 적는다. 본문의 "6종" 은 제품 수, "5종 채점" 은 점수 비교 수다.
- 캡션은 의도적으로 짧다(70–137단어). 상세는 본문과 보충 S1–S10 에 있다. 캡션에 정보가 없다고 지적하기 전에 본문 해당 문단을 보라.
- 방법 기호는 project.yaml 의 naming_canonical 을 따른다: P0 원천 계수 Stefan, P1 재보정 Stefan(수축 κ 10), R1 물리 잔차 결합(λ 0.5), D0 직접 ML, D1 물리 유사라벨 증강. S1–S14 · M1 · H18–H30 · LG · WF 는 옛 단계 이름이다.
- 고찰에 소제목이 없는 것은 Sci Rep 관행에 맞춘 결정이다. 되돌리자는 제안은 근거(저널 사례)와 함께 해야 한다.
