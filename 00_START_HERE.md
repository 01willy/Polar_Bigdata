# Polar_Bigdata 안내

영구동토 활동층 두께(ALT) 연구. 지점-연도 설계(관측 지점 × 관측 연도, ERA5-Land 기후 입력을 관측일까지 누적)로 Stefan 물리식과 잔차 기계학습을 결합하고, 공간 블록 · 지역 홀드아웃 검증과 공개 제품 비교, 2011--2024년 연도별 1 km 지도를 제공한다. Scientific Reports 투고 원고(v15) 단계이다.

## 현재 상태(2026-10-21)

| 항목 | 위치 |
|---|---|
| 상태 스냅샷과 남은 결정 | `SESSION_HANDOFF.md` 최신 절 |
| 실험 · 세션 기록 | `docs/EXPERIMENT_LOG.md` |
| GPT 인계 보고 | `gpt/handoff/INDEX.md`(최신이 위) |
| 원고 규정 점검표 | `paper/manuscript/en_v15/COMPLIANCE_v15.md` |
| 그림 설계 대안 | 4차 `docs/FIGURE_PROPOSALS_v15_2_2026-10-22.md`(3차 `..._v15_2026-10-21.md`, 2차 `..._v14_1_2026-10-21.md`, 1차 `..._v14_2026-10-20.md`), 조사 `docs/research/2026-10-21/`, 렌더 `deliverables/figure_proposals/` |
| 검토 57항 처리 점검표 | `docs/REVIEW_RESPONSE_v15_2026-10-21.md` |

## 결과물을 보려면: `deliverables/`

최종본만 모은 폴더이다(복사본, `VER=v15 bash tools/sync_deliverables.sh` 로 갱신). 종류별 폴더 하나씩, 현재 판만 둔다.

| 폴더 | 내용 |
|---|---|
| `deliverables/paper/` | 영문 · 국문 원고(전체 · 압축 제출본), 보충 자료, 규정 점검표, 본문 그림 `figures_main/`, 보충 그림 `figures_supplementary/`, 그림별 원천 자료 `source_data/`, 표 `tables/` |
| `deliverables/figure_proposals/` | 그림 설계 대안 렌더(PNG)와 제안 문서 |
| `deliverables/deck/` | 최신 발표 덱(PDF · PPTX) |
| `deliverables/reports/` | 감사 보고, 조사 노트, 세션 상태, 실험 기록, 최신 GPT 인계 |
| `deliverables/references/` | 문헌 색인과 PDF 폴더 바로가기 |
| `deliverables/archive/` | 이전 묶음(01_–15_, 날짜별 판) |

## 작업 폴더(원본). 폴더 이름이 역할을 정한다

| 폴더 | 역할 |
|---|---|
| `paper/` | 원고 원천. `manuscript/{en,ko}_v15/`(LaTeX, `build.sh`), `manuscript/shared_v*/`(수치 매크로 · 공용 표), `registry/`(실험 등록부), `claims/`(주장별 근거). 이전 판 `*_v6`–`*_v14` 는 기록용 |
| `outputs/` | 스크립트가 만든 산출물. `figures/paper/v15/`(본문 그림 + `source_data/`), `v15_si/`(보충), `v15_proposals/`(대안; 2차는 `v14_proposals/`), `maps/`, `report*/`, `models/` |
| `results/` | Rescale 클라우드 실행의 원시 결과(`rescale_*/`, 내려받은 그대로). 집계 · 그림은 `outputs/` |
| `data/` | `raw/`(원자료, git 밖) · `processed/`(라벨 · 공변량 · 지도 격자 · 예측) |
| `scripts/` | 파이프라인. `0_download` → `1_data_prep` → `2_evaluation`(채점 · 매크로) → `3_deep_learning`(모델 · 지도) → `4_visualization/paper_v15`(그림) |
| `src/polar/` | 공용 라이브러리(스타일 토큰 로더 `style_v5.py` 등) |
| `docs/` | 계획 · 감사 · 조사 노트 · 실험 기록(날짜별 파일) |
| `deck/` | 발표 덱 생성 코드 · 자산 · 렌더(`render/`) |
| `design/` | 그림 · 덱 디자인 토큰(`style_tokens_v4.json`, paper13 색표) |
| `figures/` | 그림 명세 JSON(전역 규칙: `figure_spec*.json`). 그림 파일이 아니다 |
| `references/` | 참고 논문 PDF(주제별 폴더, git 밖)와 색인 `INDEX.md`; 협업자 문헌 `10_kopri_collaborators/` |
| `gpt/handoff/` | GPT 인계 보고 |
| `tools/` | 빌드 · 납품 · 클라우드 보조 스크립트(`sync_deliverables.sh`, `build_submission_png.sh`, `finish_paper_v15.sh`, `rescale_client.py`) |
| `configs/`, `tests/`, `logs/` | 실행 설정, 시험, 실행 로그(`logs/fig_v14/` 등) |
| `work/` | Rescale 업로드 묶음(tar, git 밖) |
| `kpdc/` | KPDC 극지 데이터 센터 자료(git 밖) |
| `archive/` | 지난 대회 제출물 등 현행 작업에 쓰지 않는 것 |

## 다시 만들기

- 그림: `cd scripts/4_visualization/paper_v15 && POLAR_PALETTE=paper13 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MPLBACKEND=Agg nice -n 10 python3 figN_*.py`
- 수치 매크로: `python3 scripts/2_evaluation/v15_numbers.py`, `v15_cellyear_rescore.py`(표 S7), `v15_extra_numbers.py`(표 S8 등)
- 원고: `bash paper/manuscript/en_v15/build.sh main si`(국문은 `ko_v15`)
- 한 번에: `nohup bash tools/finish_paper_v15.sh > logs/fig_v15/finish_chain.log 2>&1 &`(매크로 → 빌드 → 점검표 → deliverables 동기화)
