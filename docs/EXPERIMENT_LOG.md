# Polar_Bigdata 실험 로그 (chronological)

> 세션별 작업 기록. 큐레이션된 마스터 인덱스는 [EXPERIMENTS.md](EXPERIMENTS.md),
> GPT 공유 핸드오프는 [gpt/handoff/](../gpt/handoff/) 참조.

## 2026-07-06 11:13 — GitHub 초기화 + cleanup/handoff (토너먼트·floor·큐레이션 아크 정리)

### 이 세션의 작업
- **git 저장소 초기화** (`git init`, main 브랜치). 이전까지 버전관리 없던 상태를 첫 커밋으로 확정.
- **`.gitignore` 작성**: 22GB `data/raw`·대용량 파생(*.parquet/*.npy/dl_dataset*.csv/*_predictions.csv)·모델 체크포인트(*.pt)·저작권 PDF 제외. 코드·문서·작은 결과 CSV·시각화 산출물(그림/지도 28M)은 추적. 스테이징 검증: 157파일 / 28M, 대용량 누출 0.
- **`.claude/project.yaml` 생성**: cleanup/handoff/status 스킬용 프로젝트 설정(gpu_default [4,5], experiment_log, gpt_handoff_dir 등). Polar_Bigdata를 스킬 대상 프로젝트로 등록.
- **문서화**: 이 로그, `SESSION_HANDOFF.md`(롤링 스냅샷), `gpt/handoff/20260706_1113-tournament-floor-curated-demo.md` 생성.

### 이 아크(직전 세션들)에서 확정된 연구 결과 — 커밋되는 상태
| 실험 | 산출 | 핵심 수치 | 근거 CSV |
|---|---|---|---|
| 모델 토너먼트(6종, 6fold+2seed) | GBM≈DL 정보병목 확증, Diffusion 채택 | 앙상블 16.95 ≈ Diffusion 17.09 ≈ GBM 17.24, 전부 부트스트랩 "동률" | `model_tournament_results.csv`, `_significance.csv` |
| point-scale floor 4중 확증 | 17cm=대표성 하한(셀내 SD 11cm) | InSAR +시 17.24→18.79(악화), PolSAR base 38.3/잔차 24(악화) | `insar_ablation_results.csv`, `polsar_residual_results.csv` |
| 정확도-범위 트레이드오프(첫 돌파) | 큐레이션+물리관측이 floor 돌파 | 평탄툰드라 **12.97cm**(SOTA급) → 완만 16.6 → 전역 17.3 | `curated_scope_results.csv` |
| 고정밀 국소 데모 | 북사면 250m ALT 필드 + AoA 마스크 | 3패널(PolSAR/모델/UQ) | `outputs/maps/local_demo_alt_field.png` |
| 시각화 규약 정비 | cmcrameri 냉색 계열 표준화 | oslo/vik/acton/broc | `src/polar/plotstyle.py`, `docs/VISUALIZATION.md` |

### 감사(cleanup Step 2) — ✅ 통과, 블록커 없음
- 헤드라인 수치 전부 CSV 근거와 일치(위 표). 12.97/16.95/17.09/17.24/108.5→87.3/17.24→18.79/38.3→24 교차검증 완료.
- 개념 주의(문서에 반영됨): 연평균 MAGT장 ≠ 계절 최대융해 ALT — "0°C 등온면=ALT" 비교는 오류(과거 1회 범함, 수정됨). SOTA 11-12cm은 좁은·평탄·P-band 직접관측 조건값.

### 재현 메모
- 제외된 대용량 데이터는 `scripts/0_download`·`1_data_prep`로 재생성. 학습셋 조립: `assemble_dl_dataset.py`(→dl_dataset.csv), 물리관측 부착: `insar_ablation.py`/`polsar_residual.py`.
- 큐레이션 실험 재실행: `CUDA_VISIBLE_DEVICES=4,5 python3 scripts/3_deep_learning/curated_local_model.py`.

### 미결(다음 세션)
- 확장 방향 3택 대기: ① 국소 데모 완성도(실측 검증수치+여러 창 일반성) ② 지형 계층 확장(툰드라/산지/삼림 각자 최적화) ③ 3D+전이(CCI 지중온도 1/2/5/10m).
- SoilGrids 다운로드 보류(ISRIC 서버 불가). 서버 복구 시 `scripts/0_download/soilgrids_alaska.py` 재실행.
