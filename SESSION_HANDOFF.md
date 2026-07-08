# SESSION_HANDOFF — Polar_Bigdata (현재 상태 스냅샷)

**갱신**: 2026-07-06 18:40 · **다음 세션은 이 파일부터 읽으세요.** · 방향/계획: `docs/PLAN_FORWARD.md`

## 목표 (한 줄)
전 지구 borehole 지중온도 + CALM ALT 관측 + 공변량 → 딥러닝으로 **ALT 2D 지도 + 얕은 3D 지중열구조 + 셀별 불확실성**,
**알래스카 학습 → 타 영구동토 지대 전이(transfer)** 검증.

## 현재 확정 상태 (검증됨, 근거 CSV 있음)
| 축 | 결론 | 수치 | 근거 |
|---|---|---|---|
| **모델 선택** | GBM≈DL, 정보병목이 지배 — 정확도로는 모델교체 무의미. Diffusion을 UQ/생성경로 이점으로 채택 | 앙상블 16.95 ≈ Diff 17.09 ≈ GBM 17.24 (부트스트랩 전부 "동률") | `data/processed/model_tournament_{results,significance}.csv` |
| **정확도 병목(정정)** | 17cm은 **물리하한 아님 — 현재 공변량의 정보병목**(비가역잡음 ~4cm, R²≈0.2). pseudo-replication(같은 공변량 셀 ALT 34–96cm 공존)+척도불일치가 apparent floor 생성. InSAR/PolSAR 편입 실패는 "쉬운 공변량 소진"이지 물리벽 아님 | skill-over-mean 전역 10.4% · 미투입 모달리티(SoilGrids/Sentinel) 헤드룸 존재 | `insar_ablation_results.csv`, `polsar_residual_results.csv` |
| **정확도-범위 트레이드오프(정정)** | 평탄지 절대 RMSE↓는 **범위축소 아티팩트**(skill-over-mean 7.4% < 전역 10.4%). **"SOTA 돌파" 아님** — 레짐별 지배 정보원 차이(평탄지=PolSAR만 유효) | 평탄툰드라 12.97 / 완만 16.6 / 전역 17.3 (절대RMSE, **R²·skill 병기 필수**) | `curated_scope_results.csv` |
| **전이(covariate)** | ERA5-Land 실측 공변량이 전이 20% 개선 | LORO 108.5→87.3cm | `stage2_era5_rescore.csv` |
| **3D 엔진** | 신경장 탈락(킬스위치), 3D=GBM 조건장. 단 전이는 조건장이 보간 17% 우세 | NF 2.36 vs GBM 1.31°C | (B1/B1b, 예측파일 gitignore) |
| **고정밀 데모** | 북사면 평탄툰드라 250m ALT 필드 + Area-of-Applicability 마스크 | 3패널 | `outputs/maps/local_demo_alt_field.png` |
| **다중모달 ablation** (신규, 스레드 A) | within-domain=기후 지배·지형은 공간과적합; transfer=InSAR 필수·지형만 파탄. 정보원이 보간 vs 전이서 다름 | 공간블록 M2(기후)16.3<M3(+지형)19.4 · LORO M4(+InSAR)16.4·M1(지형)31.9 | `alt_feature_ablation_results.csv` |
| **보정 UQ** (신규, 횡단) | quantile-GBM 과신을 conformal(CQR)이 90%로 보정 | coverage 71.2→**89.2%** | `alt_conformal_aoa_results.csv` |
| **transfer AOA** (신규, 횡단) | 환경 비유사도(DI)↑ → 오차↑·커버리지↓. 외삽영역 정직 표기 | RMSE 15.5→27.1cm, cov 69→51%; AOA안 16.9<밖 21.0 | `alt_aoa_transfer_results.csv` |
| **apparent-floor 진단** (신규, 스레드 C) | 17cm은 물리벽 아님 — 비가역 7.2cm ≪ 현재 | within 13.7%/between 86.3% | `apparent_floor_diagnosis.csv` |

## 시각화 규약
- **냉색 계열 표준**(cmcrameri, Crameri 2020): ALT=oslo_r, 온도=vik, 오차=acton, 차이=broc. 붉은 계열 금지.
- 중앙 스타일: `src/polar/plotstyle.py`의 `use_polar()`. 상세 `docs/VISUALIZATION.md`.

## 확정 방향 (2026-07-06) → `docs/PLAN_FORWARD.md`
대회용 연구축 확정: **다중모달 big-data 융합 + 정직한 평가(누설통제·AOA·보정 UQ) + 얕은 3D + 전이**. 스레드 A(ALT 다중모달 ablation)·B(3D 조건장)·C(apparent-floor 진단)·D(T-lite 게이트) + 횡단 AOA/UQ. 우선순위: 데이터 활용량·규모 + 기술 차별성 → 시각화.

## 다음 (우선순위)
1. **데이터 확장**(심사 1순위 지렛대): SoilGrids(서버복구)·Sentinel-1 InSAR 시계열·Sentinel-2 → ablation M7~M9 채우기. 22GB 보유분(PolSAR 7G·ReSALT 6.9G·CCI) 활용률↑.
2. **스레드 B(3D)**: GBM 조건장 + CCI 0/1/2/5/10m prior + 물리 단조성 → PyVista 인터랙티브 열큐브 + 0°C 등온면. (`thermal3d_conditioned_gbm.py` 예정)
3. **스레드 D(T-lite 게이트)**: 연별 ERA5 forcing + CALM site-year 시계열 → GRU vs GBM-annual, 전이 개선시만 확장.
4. **시각화 통합(polish 단계)**: 이관된 QA 항목 — basemap(coastline), dual-axis 분리, fold error-bar, SVG 폰트 감사. + 슬라이드/포스터/인터랙티브 조립.

## 운영 메모
- **GPU**: [6,7,8,9](2026-07-08 최신). 사용자가 매 세션 지정하므로 최신 지시 우선. GBM ablation/UQ·데이터 재구조화는 CPU(sklearn); GPU는 torch(GRU/3D)만.
- **자격증명**: `~/.netrc`(Earthdata)는 사용자가 직접 관리. 비밀값 출력 금지.
- **언어**: 모든 설명 한글.
- **보류**: SoilGrids 다운로드(ISRIC 서버 불가) — 복구 시 `scripts/0_download/soilgrids_alaska.py`.
- 대용량 데이터(22GB)는 git 제외 — `scripts/0_download`·`1_data_prep`로 재생성.

## 문서 지도
- 마스터 인덱스: `docs/EXPERIMENTS.md` · 세션 로그: `docs/EXPERIMENT_LOG.md` · 시각화: `docs/VISUALIZATION.md`
- GPT 핸드오프: `gpt/handoff/`(최신 `20260706_1717-lit-review-forecasting-4d-tracks.md`) · 방향계획 `docs/PLAN_FORWARD.md` · 문헌 `references/INDEX.md`(+`00_core10/`)
