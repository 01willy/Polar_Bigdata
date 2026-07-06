# SESSION_HANDOFF — Polar_Bigdata (현재 상태 스냅샷)

**갱신**: 2026-07-06 11:13 · **다음 세션은 이 파일부터 읽으세요.**

## 목표 (한 줄)
전 지구 borehole 지중온도 + CALM ALT 관측 + 공변량 → 딥러닝으로 **ALT 2D 지도 + 얕은 3D 지중열구조 + 셀별 불확실성**,
**알래스카 학습 → 타 영구동토 지대 전이(transfer)** 검증.

## 현재 확정 상태 (검증됨, 근거 CSV 있음)
| 축 | 결론 | 수치 | 근거 |
|---|---|---|---|
| **모델 선택** | GBM≈DL, 정보병목이 지배 — 정확도로는 모델교체 무의미. Diffusion을 UQ/생성경로 이점으로 채택 | 앙상블 16.95 ≈ Diff 17.09 ≈ GBM 17.24 (부트스트랩 전부 "동률") | `data/processed/model_tournament_{results,significance}.csv` |
| **정확도 하한(floor)** | point-scale 17cm은 대표성 물리하한(셀내 SD 11cm). InSAR·PolSAR·격자·areal 4중 확증 | InSAR +시 17.24→18.79, PolSAR base 38.3/잔차 24 (둘 다 악화) | `insar_ablation_results.csv`, `polsar_residual_results.csv` |
| **정확도-범위 트레이드오프** | 좁고 평탄할수록 물리관측(PolSAR)이 통해 정확 — 큐레이션이 floor 돌파 | **평탄툰드라 12.97cm(SOTA급)** → 완만 16.6 → 전역 17.3 | `curated_scope_results.csv` |
| **전이(covariate)** | ERA5-Land 실측 공변량이 전이 20% 개선 | LORO 108.5→87.3cm | `stage2_era5_rescore.csv` |
| **3D 엔진** | 신경장 탈락(킬스위치), 3D=GBM 조건장. 단 전이는 조건장이 보간 17% 우세 | NF 2.36 vs GBM 1.31°C | (B1/B1b, 예측파일 gitignore) |
| **고정밀 데모** | 북사면 평탄툰드라 250m ALT 필드 + Area-of-Applicability 마스크 | 3패널 | `outputs/maps/local_demo_alt_field.png` |

## 시각화 규약
- **냉색 계열 표준**(cmcrameri, Crameri 2020): ALT=oslo_r, 온도=vik, 오차=acton, 차이=broc. 붉은 계열 금지.
- 중앙 스타일: `src/polar/plotstyle.py`의 `use_polar()`. 상세 `docs/VISUALIZATION.md`.

## 다음 갈림길 (사용자 결정 대기)
① **국소 데모 완성도** — 실측 대비 검증수치 명기 + 여러 툰드라 창으로 일반성 확인.
② **지형 계층 확장** — 툰드라/산지/삼림 각 유형별 모델을 floor에 최적화 후 통합.
③ **3D + 전이** — CCI 지중온도 1/2/5/10m(무계정) → 얕은 3D 열구조 + 알래스카→타지대 전이.

## 운영 메모
- **GPU**: 기본 [4,5]. 사용자가 매 세션 지정하므로 최신 지시 우선.
- **자격증명**: `~/.netrc`(Earthdata)는 사용자가 직접 관리. 비밀값 출력 금지.
- **언어**: 모든 설명 한글.
- **보류**: SoilGrids 다운로드(ISRIC 서버 불가) — 복구 시 `scripts/0_download/soilgrids_alaska.py`.
- 대용량 데이터(22GB)는 git 제외 — `scripts/0_download`·`1_data_prep`로 재생성.

## 문서 지도
- 마스터 인덱스: `docs/EXPERIMENTS.md` · 세션 로그: `docs/EXPERIMENT_LOG.md` · 시각화: `docs/VISUALIZATION.md`
- GPT 핸드오프: `gpt/handoff/`(최신 `20260706_1113-tournament-floor-curated-demo.md`)
