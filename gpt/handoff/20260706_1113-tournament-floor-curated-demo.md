# Handoff: 모델 토너먼트 · 정확도 floor 규명 · 큐레이션 SOTA급 국소 데모 · GitHub 초기화
**Project**: Polar_Bigdata (영구동토 borehole 온도 → 3D ALT/열구조 예측)
**Date**: 2026-07-06 11:13
**Session focus**: 모델 정확도의 물리적 한계(floor)를 4중으로 규명하고, 큐레이션+물리관측으로 그 한계를 처음 돌파(평탄툰드라 12.97cm=SOTA급)했으며, 전체 작업을 GitHub로 버전관리 + cleanup/handoff.
**Author**: Claude Fable 5 + willy010313

---

## 1. TL;DR
- **모델교체론은 끝났다**: 6모델 토너먼트(GBM/MLP/FT-Transformer/TabM/Flow/Diffusion, 6fold+2seed, 225k점) — 앙상블 16.95 ≈ Diffusion 17.09 ≈ GBM 17.24cm, **블록 부트스트랩상 전부 GBM과 "동률"**. 병목은 모델이 아니라 공변량 정보량(1:1 진단서 전 모델 평균회귀로 시각 확증).
- **17cm는 물리 하한(floor)**: InSAR·PolSAR·격자지지·areal 4중 확증. 셀내 ALT 표준편차 ~11cm = 대표성 한계라 어떤 모델도 못 뚫음.
- **하지만 floor는 조건부다 — 첫 돌파**: 좁은·평탄·직접관측(SOTA 조건)을 재현하면 물리관측이 통한다. **평탄툰드라(slope<2,elev<150) 앙상블 12.97cm = SOTA(11-12cm)급**. 범위를 넓히면 17cm로 수렴.
- **산출물**: 고정밀 국소 데모(북사면 250m ALT 필드 + Area-of-Applicability 마스크) + 냉색 계열 시각화 규약 표준화.
- **인프라**: 프로젝트를 **GitHub로 초기화**(157파일/28M, 대용량 데이터·모델·PDF 제외), `.claude/project.yaml` 등록으로 cleanup/handoff 스킬 사용 가능.

## 2. Context
- 직전 핸드오프: [2026-06-30 전 지구 ALT baseline](20260630_1505-global-alt-baseline-stage0-1.md) — 데이터 구축 + 누설검증 평가골격 + 도일피처 baseline(LORO ~97cm), "다음 지렛대=계정 데이터/공간DL"로 종료.
- 그 이후(요약된 세션들)에 공변량 업그레이드(ERA5-Land, 전이 −20%) → 공간 DL(패치CNN≈GBM) → B0/B0b 사전학습·앙상블 → 모델 토너먼트 → floor 규명 → 큐레이션 돌파 → 국소 데모까지 진행. 이번 세션은 그 아크를 **버전관리로 확정 + 문서화**.
- 동기: 사용자가 "17cm에 만족 못 함". SOTA는 왜 11-12cm이고 우리 데이터에선 왜 안 되는가를 규명하고, 사용자 제안(P-band 물리관측 + 큐레이션된 툰드라 목표범위로 고정밀 국소 데모)을 검증.

## 3. What we did
- **모델 토너먼트** (`scripts/3_deep_learning/model_tournament.py`)
  - 6모델 torch 직접구현(외부의존 0), 공간블록 6fold + 2seed 앙상블, 전량 225k점.
  - 결과: 앙상블(GBM+FT-T) 16.95 / Diffusion 17.09 / GBM 17.24 / FT-T 17.95 / MLP 18.18 / Flow 18.31 / TabM 20.36.
  - 블록 부트스트랩(400회): ΔRMSE 95%CI 전부 0 포함 → **전 모델 GBM과 동률**(공간블록 6개뿐 + fold 난이도 편차 극심 → 상위 모델 차이 통계적 무의미).
  - Diffusion 채택: RMSE 동률이나 MAE 12.11로 최저 + 네이티브 UQ. 산출: `data/processed/model_tournament_{results,perfold,significance}.csv`, `outputs/figures/deep_learning/model_tournament.png`.
- **floor 4중 확증**
  - InSAR (`insar_ablation.py`): ReSALT 부착(BallTree k=64/5km). GBM 17.24→**18.79 악화**, 앙상블만 16.82. `insar_ablation_results.csv`.
  - PolSAR (`polsar_residual.py`): ABoVE PolSAR ds2332(30m, 단위 m→cm ×100). base 그대로 38.29, 잔차보정 24.05(악화), 앙상블+PolSAR 16.24. `polsar_residual_results.csv`.
  - 격자지지/areal: 지지 키워도 RMSE 안 줄어듦(표본손실/외삽). 셀내 SD 11cm=대표성 하한.
- **정확도-범위 트레이드오프 = 첫 floor 돌파** (`curated_local_model.py`)
  - scope별 공간블록 CV: 평탄툰드라(n=64k) 12.97 / 완만 16.6 / PolSAR유효 16.3 / 전역 17.3.
  - 큐레이션이 신호를 되살림: PolSAR 상관 전역 0.07 → 평탄 0.57. 물리관측은 앙상블/Diffusion 경유해야 효과(GBM 직접 투입은 악화). `curated_scope_results.csv`, `outputs/figures/deep_learning/accuracy_vs_scope.png`.
- **고정밀 국소 데모** (`scripts/4_visualization/map_local_demo_northslope.py`)
  - 북사면 평탄툰드라 250m 격자, PolSAR bilinear 재투영, 평탄툰드라 GBM+Diffusion 학습, AoA 마스크(slope<2 & elev<150 & all-finite). 3패널: PolSAR raw / 우리 모델 / Diffusion UQ. `outputs/maps/local_demo_alt_field.png`.
- **시각화 규약** (`src/polar/plotstyle.py`, `docs/VISUALIZATION.md`): cmcrameri 냉색 표준(ALT=oslo_r, 온도=vik, 오차=acton, 차이=broc). 붉은 계열 폐기.
- **GitHub 초기화 + 스킬 등록**: `.gitignore`(대용량/모델/PDF 제외), `.claude/project.yaml`, `docs/EXPERIMENT_LOG.md`, `SESSION_HANDOFF.md` 생성. 157파일/28M 커밋.

## 4. Key numbers (this session)
| Method | Domain/Case | Metric | Value | Source artifact |
|---|---|---|---|---|
| 앙상블(GBM+FT-T) | 북미 225k, 공간블록 CV | RMSE | 16.95cm | `data/processed/model_tournament_results.csv` |
| Diffusion | 동 | RMSE / MAE | 17.09 / 12.11cm | 동 |
| GBM | 동 | RMSE | 17.24cm | 동 |
| 전 모델 vs GBM | 블록 부트스트랩 400회 | ΔRMSE 유의성 | 전부 동률(CI에 0 포함) | `model_tournament_significance.csv` |
| GBM +InSAR | 동 | RMSE | 17.24→18.79 (악화) | `insar_ablation_results.csv` |
| PolSAR 그대로 / 잔차보정 | PolSAR유효 | RMSE | 38.29 / 24.05 (악화) | `polsar_residual_results.csv` |
| 앙상블(+PolSAR·InSAR) | **평탄툰드라 n=64k** | RMSE | **12.97cm (SOTA급)** | `curated_scope_results.csv` |
| 앙상블 | 전역 다양지형 | RMSE | 17.30cm | 동 |
| ERA5-Land 실측 | LORO 전이 | RMSE | 108.5→87.3cm (−20%) | `stage2_era5_rescore.csv` |

## 5. Decisions made
- **Diffusion을 배포 모델로 채택** — 정확도 손해 없이(RMSE 동률) MAE 최저 + 네이티브 UQ/다운스케일 이점.
- **17cm를 다양지형의 물리 하한으로 인정** — 모델 고도화로는 못 뚫는다는 것을 4중 확증. 정확도 지렛대는 모델이 아니라 (a)새 공변량 모달리티 또는 (b)범위 큐레이션.
- **전략 확정(사용자)**: 고정밀 국소 데모(평탄툰드라 SOTA급) 먼저 → 넓은 범위/다양 지형으로 확장.
- **GitHub 버전관리 도입** — 대용량 데이터는 제외(재생성 가능), 코드·문서·결과 요약·그림만 추적.

## 6. Open questions / blockers
- **확장 시 floor를 어디까지 유지하나?** — 지형 유형별(툰드라/산지/삼림) 분할모델이 통합 시 각 floor를 지킬지 미검증. 증거: 유형별 spatial CV + 통합 vs 분할 비교.
- **국소 데모의 실측 검증수치 부재** — 데모 필드의 정량 검증(hold-out 실측 대비 RMSE, 커버리지) 아직 미기재.
- **3D 라벨 심도** — 얕은 3D 열구조엔 CCI 지중온도(1/2/5/10m) 필요. 무계정 확보 경로는 있으나 아직 미취득.
- **SoilGrids 보류** — ISRIC(files.isric.org) 서버 연결 실패, 0파일. 서버 복구 시 재시도.

## 7. Next steps (prioritized)
1. **국소 데모 완성도** — owner: Claude, ~반나절. 데모 필드 실측 hold-out 검증수치(RMSE/커버리지) 명기 + 여러 툰드라 창으로 일반성 확인. 선결: 없음.
2. **지형 계층 확장** — owner: Claude, ~1일. 툰드라/산지/삼림 유형별 모델 각자 floor 최적화 후 통합, scope별 RMSE 정량화. 선결: 없음(기존 dl_dataset).
3. **3D + 전이** — owner: Claude, ~1-2일. CCI 지중온도 1/2/5/10m 무계정 취득 → 얕은 3D 열구조(GBM 조건장) + 알래스카→타지대 전이. 선결: CCI GRD v4.0 다운로드.
4. **(병렬) 새 공변량 모달리티** — Sentinel-1 InSAR 시계열/Sentinel-2/SoilGrids로 정보량 확대(floor 자체를 낮출 유일 경로). 선결: SoilGrids 서버 복구 or 계정 데이터.

## 8. Pointers
- Authoritative reports: `docs/EXPERIMENTS.md`(마스터 인덱스), `docs/EXPERIMENT_LOG.md`, `SESSION_HANDOFF.md`.
- 결과 CSV: `data/processed/{model_tournament_*,curated_scope_results,insar_ablation_results,polsar_residual_results,stage2_era5_rescore}.csv`.
- 그림: `outputs/figures/deep_learning/{model_tournament,accuracy_vs_scope}.png`, `outputs/maps/{local_demo_alt_field,deploy_alt_gbm_vs_diffusion}.png`.
- Active jobs: 없음. Ckpt: `outputs/models/*.pt`(gitignore, 로컬만).
- 관련 이전 핸드오프: `gpt/handoff/20260630_1505-global-alt-baseline-stage0-1.md`.
- **제외된 대용량 데이터**(22GB)는 git에 없음 — `scripts/0_download`·`1_data_prep`로 재생성.

## 9. Caveats for GPT
- **MAGT ≠ ALT**: 연평균 지중온도장(MAGT, 0°C 등온면 심도)은 계절 최대융해깊이(ALT)와 다른 양. "0°C 등온면=ALT" 비교는 개념 오류(과거 1회 범했다 수정). ALT 지도는 B0/토너먼트 계열, MAGT/3D는 GBM 조건장.
- **CCI ALT 단위는 m** (문서 일부의 cm 아님). PolSAR ds2332도 m → cm 변환(×100) 후 사용.
- **SOTA 11-12cm의 정체**: 좁은 ALT범위(31-73cm) + P-band 직접관측 + 큐레이션된 평탄지 + 공동위치 검증 조건의 값. 우리 12.97cm(평탄툰드라)이 이를 재현. 다양지형 17cm와 직접 비교 금지(다른 과제).
- **부트스트랩 "동률"**의 의미: 상위 모델 우열이 통계적으로 무의미(공간블록 6개 한계)일 뿐, 모델이 나쁘다는 뜻 아님. 병목은 정보량.
- 명칭: B0=사전학습+미세조정, B0b=CCI피처+앙상블, B1/B1b=신경장(탈락), tournament=6모델비교, curated=범위 트레이드오프.
