# Polar_Bigdata 실험 로그 (chronological)

> 세션별 작업 기록. 큐레이션된 마스터 인덱스는 [EXPERIMENTS.md](EXPERIMENTS.md),
> GPT 공유 핸드오프는 [gpt/handoff/](../gpt/handoff/) 참조.

## 2026-07-10 — overnight: 셀 단위 재분석 + T-lite 게이트 + 데이터 확충 + 발표덱 v2

GPT 계획(`gpt/20260709_claude_next_research_plan_dl_alt_3d.md`) P2/P3/P6-C 실행. 코드: `scripts/2_evaluation/overnight_cell_experiments.py`, `scripts/3_deep_learning/tlite_sequence_gate.py`, `scripts/1_data_prep/enrich_cci_cell.py`.

### 셀 단위(location-equal) 다중모달 ablation — 정직한 재분석 (`alt_ablation_cell_results.csv`)
- 기준선 = `dl_dataset_cell.csv`(14,348 셀, 셀평균 ALT). 공간블록·LORO, 표준지표.
- **LORO**: M0 지역평균 21.8 · M1 기후 **16.45(skill 10.8%)** · M3 기후+지형 16.94(지형 추가 악화) · **M4 +InSAR 16.09(skill 12.7%, 물리 최고)** · M5 +PolSAR 16.98 · M9 전체 16.43.
- **위치 대조군(lat+lon 2피처)**: LORO **15.72cm skill 14.7%** — 물리 공변량 조합보다 높음. 위도가 기후 이상을 대리 = **정보 병목의 직접 증거**. (점-단위 옛 ablation의 15% skill는 pseudo-replication 착시였음이 셀 재평가로 확증.)

### 보정 UQ + AOA (셀, `alt_conformal_cell_results.csv`, `alt_aoa_cell_transfer.csv`)
- raw 분위-GBM 90% 커버리지 **56.1%(심한 과신)** → **CQR 보정 85.9%**(폭 50.6cm). 점-단위(71%)보다 raw 과신 심함.
- AOA DI-구간(qcut10→중복제거 6구간): RMSE 저DI 13 → 고DI 30cm. **커버리지는 비단조**(D1 61% → D3 88% 피크 → D6 50%) — 공간 calib/test 분리 + marginal 보장 특성. 정직히 표기.

### T-lite 시계열 DL 게이트 — 정직한 음성 (`tlite_sequence_gate_results.csv`, gate_meta)
- CALM site-year 251사이트·3,345 시퀀스. GRU/TCN vs persistence·climatology·GBM-annual. 검증: site-disjoint 5-fold + temporal holdout(≤2014/≥2015).
- **site-disjoint**: GRU 16.79 < persistence 16.98 < GBM 17.33 (GRU 소폭 최우수). **temporal holdout**: **GBM-annual 15.86 < persistence 17.02 < GRU 19.15 < TCN 23.85** (DL 붕괴).
- **게이트 미통과**(temporal 미충족) → 부록/future work 강등. **정적 tabular ALT는 GBM으로 충분** 재확인. DL은 고차원 EO/SAR·시간축에서 게이트 통과 시만.

### 데이터 확충
- **ESA CCI ALT prior**: 25년 다년평균을 14,348 셀에 추출(`enrich_cci_cell.py`), 전 셀 유효, 관측 셀평균과 **r=0.53**. ablation **M8 +CCI**: 개선 없음/악화(기후와 중복) — 정직한 음성. CCI는 prior/benchmark로만.
- **SoilGrids**: ISRIC VRT vsicurl 원격 읽기 정체(산출 0) → 중단, 다음 세션 재시도(사전 타일 캐시 권장). 계획: `docs/DATA_ACQUISITION_PLAN.md` 갱신.

### 발표덱 v2 (에디토리얼/학술 보고서)
- v1의 "AI틱 라운드카드·테크그라디언트" 결별: 종이 배경 + 세리프 제목(Noto Serif CJK KR) + Pretendard 본문, booktabs 표, 저널형 러닝헤더/푸터, 박스없는 figure-of-merit, 렌더된 수식(skill·Stefan·CQR·DI·분산분해), 번호 캡션. 코드 `deck/report_lib.py`·`deck/build_report.py`(18슬라이드). 배경·동기·선행연구·연구질문 슬라이드 추가.
- 시각+과학 리뷰 반복 반영(수식 여백·표 넘침·여백축소·GRU 게이트 정직 서술·Mloc=위경도·AOA 비단조·CCI 중복). 산출: `deck/render/permafrost_report.{pptx,pdf}`.
- **문체 규율**: 모든 프로젝트 기본 = 정돈된 보고서/논문 톤(메모리 `report-tone-default` 고정, 전역 규칙 `~/.claude/rules/writing-tone.md`).
- **덱 v2 개정(사용자 지적 반영)**: 렌더 텍스트에서 em-dash(—) 전량 제거(불릿 머리·러닝헤더 포함), 전역 규칙에 em-dash 금지 명시. 장식 위젯 절제(finding 컬러 세로바 제거, fom 컬러 규칙선 제거). 그림은 논문 관례로 재작성(도판에 박힌 결론형 굵은 제목 제거, 패널 라벨 (a)/(b), 회귀선 추가). 친절한 예시 추가(skill 계산 예, pseudo-replication 34/96cm 예, 누설 예). 수식 캐시 경로 버그 수정(`assets/eq/`). 시각·과학 리뷰 재반영.
- **미완/다음 세션**: 그림 전면 재구성(concept·지도류 포함) 및 PPT 전면 재구성 검토. 사용자가 GPT와 상의 예정. README·PLAN_FORWARD·EXPERIMENTS 구계획 서술 현행화 필요(교차문서 감사 지적).

## 2026-07-08 — 스레드 R 착수: 데이터 재구조화(㉡집계·㉢가중) + ERA5 다년 확보

### 전략 확정 (재구조화 먼저 vs 다운로드 먼저)
- 확인: ERA5 원본이 디스크에 **2015–2020만** 있는데 **라벨 70%가 2010–2014**(2014=59%). → ㉠ 시간정합은 ERA5 다년 다운로드가 전제.
- 결론: **㉡집계·㉢가중은 무료·즉시**(재구조화 먼저 맞음), **㉠은 targeted ERA5 다운로드 필요**(둘 병렬). 새 모달리티(SoilGrids/Sentinel)는 재구조화 기준선 후.

### ㉡㉢ 집계·가중 (`aggregate_alt_cell.py`, `restructure_gate.py`)
- `dl_dataset_cell.csv`: 225k → **14,348 위치당 1행**. 정답=셀평균 ALT, **alt_sd=셀내 SD(불확실성 라벨, 중앙 2.0cm)**, 위치 동등가중.
- **R3 게이트**(셀평균 ALT 공정채점, 공간블록+LORO): (1)pooled 현재 skill **−0.8%/0.04%** → (2)**+1/n가중 10.9%/7.3%** → (3)cell학습 −1.9%/**8.1%(전이 최고)**.
- **핵심 발견**: 현재 pooled는 위치-동등 채점 시 **거의 평균 수준**(밀집셀 편향으로 점-단위 R²0.2가 부풀려짐). **재가중만으로(무료) 보간 +11%·전이 +7~8% 회복.** 적대검증: 가중치 정합(0.005~1.0, loc합=1)·상위1%셀이 점의 11.9% 확인 → 버그 아님. 절대 skill은 fold 분산 큼(정성 결론). `restructure_gate_results.csv`, `figures/02_evaluation/restructure_gate.png`.

### ㉠ 시간정합 준비 완료
- **ERA5-Land 2010–2024 다운로드 성공**(`era5land_monthly_multiyear.py`, 816MB, 180 monthly steps, t2m/sd/stl1). → 라벨 99.4% 연도정합.

### ㉠ 시간정합 게이트 (`era5land_temporal_covariates.py`, `temporal_gate.py`) — 정직한 혼합/음성
- 연도별 도일/적설/토양온도 파생(`alt_era5_temporal.csv` 2.96M행, 같은 위치 연도별 TDD SD=145 → 연도신호 실재). (위치,연도) 17,800단위, 조인 100%.
- **static vs temporal(그해 기후) GBM**: 보간(공간블록) temporal 8.7%>static 5.7%(+3), **전이(LORO) temporal 5.1%<static 9.9%(−5)**, **per-year holdout temporal 20%<static 28%(−8)**.
- **결론**: 그해 기후를 GBM에 스냅샷으로 넣는 것은 **매핑엔 도움 안 됨**(전이·연도holdout 악화). **ALT 변동은 '그 해 날씨'보다 '위치 고유성질'이 지배**(static이 위치 기후평년 학습 → 강한 baseline). 시간축은 **매핑 지렛대 아님** → 시계열 신호는 **lagged ALT(사이트 지속성)** 로만 유효한데 그건 모니터링 사이트 **예측(T1)** 용(매핑엔 미가용). **T-lite/GRU는 매핑용으로 게이트 탈락**, 예측 응용으로만 별도.

### 스레드 R 종합 (재구조화)
- **채택**: ㉢ 1/n 가중(무료·큰 이득 +11%/+8%) · ㉡ 셀집계+셀내SD(불확실성 라벨·척도정합).
- **탈락(게이트)**: ㉠ 시간정합 climate 스냅샷(매핑엔 혼합/음성).
- **정직한 함의**: "데이터를 올바르게 넣기"의 실질 이득은 **가중/집계**였고, 정확도의 남은 지렛대는 (a)새 모달리티/세밀 해상도 (b)예측(T1)으로의 응용 전환. 모델(DL)이 아님 재확인.

## 2026-07-06 18:40 — P0 실행 + 스레드 A(다중모달 ablation) + 횡단 AOA/UQ

### P0/P1 (기반)
- **`src/polar/eval_metrics.py`** 표준지표(rmse/mae/bias/r2/target_sd/skill_over_mean/coverage/width) — 모든 결과가 RMSE 옆 R²·skill 병기.
- **재채점**(`rescore_results.py`): 토너먼트 R²(앙상블 0.23·GBM 0.20, 전부 skill~12%), 큐레이션 skill **전역 10.4% > 평탄툰드라 7.4%** → "12.97 SOTA 돌파"가 범위축소 아티팩트임 확증. 산출 `model_tournament_results_rescored.csv`, `curated_scope_results_rescored.csv`, `figures/02_evaluation/skill_reframing.png`.
- **apparent-floor 진단**(`diagnose_apparent_floor.py`): 분산분해 within 13.7%/between 86.3%, **비가역하한 ~7.2cm ≪ 현재 16.9cm** = covariate 병목(헤드룸). 산출 `apparent_floor_diagnosis.csv`, `figures/02_evaluation/apparent_floor_diagnosis.png`.
- **`design/`** 디자인 시스템(brand_tokens/layout_rules/visual_qa_checklist).

### 스레드 A — ALT 다중모달 feature ablation (`alt_feature_ablation.py`, GBM 고정, 공간블록+LORO)
- PolSAR/InSAR 데이터셋 행정렬 검증 후 결합(14+PolSAR3+InSAR5). HistGBM NaN 네이티브.
- **핵심(정직-평가 스토리)**: within-domain(공간블록)=**기후(ERA5) 지배**(M2 16.3cm, skill 15%), 지형 추가는 **공간 과적합**으로 악화(M3 19.4); transfer(LORO)=**InSAR 필수**(M4 16.4 최고), 지형만 31.9cm 파탄(skill −65%). **정보원이 보간 vs 전이에서 다름.** per-fold 검증(악화는 fold0 지형=지역대리 과적합)으로 버그 아님 확인.
- 산출 `alt_feature_ablation_results.csv`, `alt_ablation_M6_oof.csv`, `figures/06_deep_learning/alt_feature_ablation.png`. 미취득: SoilGrids/Sentinel/CCI(다음 데이터 확장).

### 횡단 — AOA + Conformal UQ (`aoa_conformal_alt.py`, `aoa_transfer.py`)
- **within-domain CQR**(spatial-block): raw quantile-GBM coverage **71.2% → CQR 보정 89.2%**(목표 90%, width 37.7→53.7cm). 생성모델 과신 교정 실증. `alt_conformal_aoa_results.csv`, `figures/02_evaluation/coverage_calibration.png`, `maps/alt_uncertainty_width.png`.
- **transfer AOA**(LORO, Meyer DI): DI↑ → **RMSE 15.5→27.1cm, coverage 69→51%**; AOA 안(16.9cm/68%) < 밖(21.0cm/62%). ⚠️pseudo-replication이 DI 정규화 깨뜨림 → 참조점 **고유위치 dedup**으로 수정. `alt_aoa_transfer_results.csv`, `maps/alt_aoa_mask.png`.

### 시각화 QA
- scientific-figure-reviewer + visual-reviewer 2에이전트 검토 → 냉색 규약·경도(°W)·주석겹침·음의skill 구분 등 수정. **이관(다음 viz-통합 단계)**: basemap(coastline), dual-axis 분리, fold error-bar, SVG 폰트 감사.

## 2026-07-06 17:20 — 논리 검증 + 문헌 재조사(49편) + 방향 확정(PLAN_FORWARD) + P0 정정

### 검증 결과 (과대표현 정정)
- **"17cm=물리하한" 폐기**: 비가역잡음 ~4cm(분산분해 within 12.3%), 현재 R²≈0.2 = **공변량 정보병목**. pseudo-replication 진단으로 확증(아래).
- **pseudo-replication 발견**: `dl_dataset.csv` 225,421행 → 고유위치 14,348개. 한 위치 내 연도 달라도 **피처 std=0**(정적 climatology), ALT만 SD~14.6cm. 예: 한 셀 2013년 동일피처 ~100점, ALT 34–96cm. → 같은 X→다른 y = 라벨잡음, 모델 평균회귀 강제, apparent floor 생성. CV는 위치그룹핑이라 누설은 없음.
- **"12.97cm=SOTA 돌파" 정정**: skill-over-mean(1−RMSE/자기SD) 전역 10.4% > 평탄툰드라 7.4% → 큐레이션은 설명력을 낮춤. 범위축소 아티팩트. "레짐별 지배 정보원 차이"로 재프레이밍.

### 문헌 재조사
- 워크플로 57에이전트·49편 web-검증, 오픈PDF 36편 `references/0X_*/` 정리 + `references/INDEX.md` + 핵심10편 `references/00_core10/`. 저자 오귀속 3건 적발(Ran2022 공저자·Rahaman↔Chance·Suzuki↔Ieki).
- T1(지점 ALT 예측)=붐빔(Rahaman2025/Luo2022), T2(4D)=열림. 차별성=transfer+UQ+shallow3D(Koven2025 리뷰가 open gap 인증).

### 방향 확정 + P0 착수
- `docs/PLAN_FORWARD.md` 작성: 스레드 A(다중모달 ablation)/B(3D 조건장)/C(진단)/D(T-lite) + 횡단 AOA/UQ. 우선순위=데이터 활용량·규모+기술차별성.
- **P0 정정 반영**: SESSION_HANDOFF·EXPERIMENTS 표현 수정(본 항목), GPT 핸드오프 `gpt/handoff/20260706_1717-lit-review-forecasting-4d-tracks.md`.

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
