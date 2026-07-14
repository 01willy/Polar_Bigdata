# SESSION_HANDOFF — Polar_Bigdata (현재 상태 스냅샷)

**갱신**: 2026-07-14 · **다음 세션은 이 파일부터 읽으세요.**

## ★ 다음 세션 즉시 착수(2026-07-14 확정) — `docs/EXPERIMENT_PLAN_2026-07-14.md`
GPU **6,7,8,9**. 순서: **P0** 데이터 인벤토리 세계지도 + 6모델별 ALT 예측·오차 지도(즉시) → **P1** 전 공변량(DEM+InSAR+PolSAR+CCI) + 전 지역(알래스카+시베리아 ALLena+티베트 QTEC) 통합 ALT 재학습·6모델 재비교 → **P2** Stefan 물리 base + DL 잔차 → **P3** 3D 전공변량+연속성DL(등온면 매끄럽게) → **P4** AlphaEarth 임베딩 → **P5(트랙)** 이미지 조건 diffusion/flow. 결과는 전문 mapping·시각화 후 PPT 반영.
- 핵심 확인 사실: 학습데이터 6.6MB tabular(관행·상위권, 병목=라벨희소+공변량정보). ALT 94% 알래스카. 3D=GBM 조건장·기후+깊이만. 페이지7 ALT지도=ERA5만. 시추공 지중온도 9개국 260사이트.
- 중간보고 PPT v3: `deck/build_midreport.py`(18슬라이드, Pretendard·EMP톤·2.5D단면·아키텍처그림), 렌더 `deck/render/permafrost_midreport.pdf`.

기존 방향/계획: `docs/PLAN_FORWARD.md` · `docs/EXPERIMENT_ROADMAP.md`(E1~E7) · `docs/CONTEST_PLAN_2026.md`(대회 v2) · 데이터확충: `docs/DATA_ACQUISITION_PLAN.md`
**발표덱 v2**(에디토리얼/학술): `deck/render/permafrost_report.{pptx,pdf}` (18슬라이드, 빌드 `deck/build_report.py`+`deck/report_lib.py`). v1(progress)는 `deck/render/permafrost_progress.*`.

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
| **셀 단위 재분석** (신규 2026-07-10) | location-equal 재평가에서 skill 하락(점-단위 착시 제거 확증). **위치 대조군(lat+lon)이 물리 공변량 이상** = 정보병목 직접증거. 기후 지배·+InSAR 전이 최고·CCI prior 중복(무익) | LORO M1 기후 16.45(10.8%)·M4 +InSAR 16.09(12.7%)·**Mloc 위경도 15.72(14.7%)**·M8 +CCI 악화 | `alt_ablation_cell_results.csv` |
| **보정 UQ·AOA (셀)** (신규 2026-07-10) | raw 90% 커버리지 56%(심한 과신)→CQR 86%. AOA DI-구간 RMSE 13→30cm, 커버리지 **비단조**(D3 88% 피크) | 56.1→85.9%(폭 50.6cm) | `alt_conformal_cell_results.csv`, `alt_aoa_cell_transfer.csv` |
| **T-lite 시계열 DL 게이트** (신규 2026-07-10) | site-disjoint는 GRU 소폭 최우수이나 **temporal holdout에서 GBM-annual에 미달 → 게이트 미통과**(부록). 정적 tabular는 GBM 충분 재확인 | temporal: GBM 15.86<pers 17.02<GRU 19.15<TCN 23.85 | `tlite_sequence_gate_results.csv` |
| **CCI prior 확충** (신규 2026-07-10) | ESA CCI ALT 25년 다년평균을 14,348셀 추출(전 셀 유효, 관측과 r=0.53). ablation M8은 무익(기후 중복) | r=0.53 · M8 개선 없음 | `scripts/1_data_prep/enrich_cci_cell.py` |

## 시각화 규약
- **냉색 계열 표준**(cmcrameri, Crameri 2020): ALT=oslo_r, 온도=vik, 오차=acton, 차이=broc. 붉은 계열 금지.
- 중앙 스타일: `src/polar/plotstyle.py`의 `use_polar()`. 상세 `docs/VISUALIZATION.md`.

## 확정 방향 (2026-07-06) → `docs/PLAN_FORWARD.md`
대회용 연구축 확정: **다중모달 big-data 융합 + 정직한 평가(누설통제·AOA·보정 UQ) + 얕은 3D + 전이**. 스레드 A(ALT 다중모달 ablation)·B(3D 조건장)·C(apparent-floor 진단)·D(T-lite 게이트) + 횡단 AOA/UQ. 우선순위: 데이터 활용량·규모 + 기술 차별성 → 시각화.

## 다음 (우선순위)
1. **데이터 확장**(심사 1순위 지렛대): SoilGrids(vsicurl 정체 → 사전 타일 캐시로 재시도)·Sentinel-1/2 → ablation M6/M7 채우기. CCI prior(M8)는 무익 확인. `docs/DATA_ACQUISITION_PLAN.md`.
2. **스레드 B(3D)**: GBM 조건장 + CCI 0/1/2/5/10m prior + 물리 단조성 → PyVista 인터랙티브 열큐브 + 0°C 등온면. (`thermal3d_conditioned_gbm.py` 예정)
3. **T-lite(부록)**: 게이트 미통과(temporal). 재시도 시 월별 ERA5 forcing + lagged ALT로 재검증 — 통과 못하면 future work 유지.
4. **발표덱 후속**: 데이터 확충 결과 반영해 M6/M7 슬라이드 갱신. 필요시 포스터/인터랙티브 조립.

## 운영 메모
- **GPU**: [6,7,8,9](2026-07-14 최신). 사용자가 매 세션 지정하므로 최신 지시 우선. **점유 변동 잦음**(세션 중 타 사용자 점유로 OOM 발생 이력) — 사용 전 `nvidia-smi` 필수, 소형 DL은 CPU도 충분. GBM ablation/UQ는 CPU(sklearn).
- **자격증명**: `~/.netrc`·`~/.cdsapirc`는 사용자가 직접 관리. 비밀값 출력 금지.
- **언어·문체**: 모든 설명 한글. **정돈된 보고서/논문 톤**(과장·수사·AI틱 금지) — 메모리 `report-tone-default`, 전역 규칙 `~/.claude/rules/writing-tone.md`.
- **보류**: SoilGrids(ISRIC vsicurl VRT 정체) — `scripts/0_download/soilgrids_alaska.py` 재시도.
- 대용량 데이터(22GB)·`dl_dataset*.csv`·`*_oof.csv`·`logs/`·`deck/render/*.png`는 git 제외 — 스크립트로 재생성.

## 문서 지도
- 마스터 인덱스: `docs/EXPERIMENTS.md` · 세션 로그: `docs/EXPERIMENT_LOG.md` · 시각화: `docs/VISUALIZATION.md`
- GPT 핸드오프: `gpt/handoff/`(최신 `20260706_1717-lit-review-forecasting-4d-tracks.md`) · 방향계획 `docs/PLAN_FORWARD.md` · 문헌 `references/INDEX.md`(+`00_core10/`)
