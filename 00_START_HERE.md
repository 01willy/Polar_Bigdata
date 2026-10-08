# Polar_Bigdata 안내

영구동토 활동층 두께(ALT) 예측 연구. 관측이 드문 새 지역에서 Stefan 물리식과 기계학습을 라벨 수에 따라 어떻게 결합할지를 지역 홀드아웃으로 평가한다.

## 현재 상태(2026-10-08)

| 항목 | 위치 |
|---|---|
| 상태와 다음 세션 계획(극지연 회의 뒤 연구·논문 목표 재설정 예정) | `SESSION_HANDOFF.md` 10-08 절 |
| 정리 감사와 고칠 목록 | `docs/AUDIT_2026-10-08_CLEANUP.md` |
| 판정 기록 | `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 8절, 추가 등록 `docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md`·`..._XM_2026-10-05.md` 결과 절 |
| 실험 등록부 | `paper/registry/experiments.csv` |
| 발표 대본·배경 지식 원천 | `deck/script/` |
| 열린 마감 | XE 2단계(10-08), XF·R4(10-11). 처리는 `SESSION_HANDOFF.md` 10-08 절 |

## 결과물을 보려면: `deliverables/`

최종본만 모은 폴더이다(복사본, `bash tools/sync_deliverables.sh` 로 갱신).

| 폴더 | 내용 |
|---|---|
| `deliverables/01_발표자료/` | 보고 덱 PPTX·PDF |
| `deliverables/02_논문초안/` | 영문 본문(검토용: 그림이 본문 안 / 제출형식: 그림이 끝), 영문 SI, 국문 본문 |
| `deliverables/03_그림/논문그림/` | 원고 번호 Figure 1–7(처음 인용 순서), Table 1(PDF·PNG·설명문) |
| `deliverables/03_그림/ALT지도/` | 알래스카·레나 1 km ALT 지도, 격자 크기별 오차, 기존 제품 비교 |
| `deliverables/03_그림/보충그림/` | SI 보충 그림(파일 FigS14 = 원고 S13, FigS15 = 원고 S14) |
| `deliverables/03_그림/방법도식/` | 문제 정의, 자료, 연구 흐름, 증강 대조 설계, 모델 구조, 평가 설계, 단계적 예측 워크플로 |
| `deliverables/03_그림/슬라이드용/` | 덱에 넣은 슬라이드 크기 그림 |
| `deliverables/04_참고문헌/` | 문헌 색인(INDEX.md)과 PDF 전체 폴더 바로가기 |
| `deliverables/05_보고서/` | 아침 보고, 실험 계획과 판정 기록, 질의응답 문서 |

## 작업 폴더(원본)

| 폴더 | 역할 |
|---|---|
| `paper/` | 원고 원천: `manuscript/en`, `manuscript/ko`(LaTeX), `registry/`(실험 등록부), `claims/`(주장별 근거) |
| `outputs/figures/paper/v3_restructure/` | 논문 그림 원본(그림 스크립트의 출력 위치) |
| `deck/` | 덱 생성 코드(`build_paper_report.py`)와 그림 자산(`assets/paper_report/`), 렌더(`render/`) |
| `design/` | 그림·덱 디자인 규칙. 현행 색·글꼴 토큰은 `design/style_tokens_v4.json` |
| `docs/` | 실험 계획·결과 기록(날짜별), 판정 기록은 `EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 8절 |
| `references/` | 참고 논문 PDF(주제별 폴더), 색인 `INDEX.md` |
| `scripts/`, `src/`, `tests/` | 자료 처리·실험·그림 코드와 시험 |
| `data/`, `results/` | 자료와 실행 결과(대부분 git 밖) |
| `archive/` | 지난 대회 제출물·다른 프로젝트 파일 등 현행 작업에 쓰지 않는 것 |
