# SESSION_HANDOFF — Polar_Bigdata (현재 상태 스냅샷)

**갱신**: 2026-07-24(S4·S5·S3정정) · **다음 세션은 이 파일부터 읽으세요.**

## ★ 최신 완료(2026-07-24 오후) — S4·S5 완주 + S3 증강비율 버그 수정

GPU **4,5**(사용자 RTM 잡이 6,7,8,9 점유 → 겹침 회피. 스크립트 가드 2-9 확장, 기본 GPU=4). 실험로그 `docs/EXPERIMENT_LOG.md` 최신 4항목.
- **S4 잔차학습 fallback → negative 확정(P2 재확인)**: 예측=Stefan앵커(E_train·√TDD)+λ·저용량잔차. shift-robust 입력·저용량으로도 λ>0 전 구간 LORO 게이트 악화(21.26→). Alaska fold 파탄(부트스트랩 −3.4~−35cm)이 게이트 지배, inner CV λ 자동선택 불가 실증. **부수 성과**: in-domain AK 13.33cm(ridge·λ0.75, 프로젝트 최저 갱신). 최소제곱 E가 중앙값비 E보다 앵커 우세(21.26<22.24). 스크립트 `s4_residual_learning.py`, 그림 `outputs/figures/s4_residual/`.
- **S5 dense Stefan pseudo 사전학습→finetune → 이득은 transductive 아티팩트**: 게이트 개선(FT-T 22.47→21.56) 전량이 Alaska(transductive) fold. 깨끗한 Lena Δ+0.05 무효, Canada Δ−1.29 악화. 물리 사전학습은 격자가 target 공변량 덮을 때만 유효(전이 지식 아님). mlp 전이 발산. 스크립트 `s5_pretrain_finetune.py`, 그림 `outputs/figures/s5_pretrain/`.
- **S3 증강비율 버그 발견·수정(헤드라인 정정)**: `take=min(n_ps,pool)` 상한이 r 스윕 무력화(r=0.25=r=10) → `take=n_ps`로 수정. **"r≥1 포화" 결론은 버그 아티팩트**. 수정 후 catboost·FT-T 일치: 포화 없이 r=10까지 개선, 물리 순가치(Stefan−placebo) r 따라 증가(Lena +0.8→+1.7·Canada +9.6→+10.4). Canada는 물리 필수(placebo 악화), Lena는 base 품질 의존, Ku(부정확)는 r 키울수록 해. `s3_aug_curve_results{,_ftt}.csv`.
- **다음**: S6(source-aware A5, Stefan+CCI) 또는 S7(KPDC 검증)·S9(timelapse)·S10(3D)·S11(UQ). GPT 로드맵 순서 `gpt/handoff/20260724_1327-*`. 누설 pytest 16개 통과 유지.

## ★ 이전 완료(2026-07-20 저녁) — 알래스카 내부 3트랙 + 적대적 검증 정정 → `docs/RESULTS_SUMMARY_2026-07-20.md`
- **증강(적대적 검증 후 기각)**: 1차 "4모델 유의 개선"은 착시. GBM 개선=test 인접 특징복제 누설(donor 제거 시 14.2→16.0 악화), MLP=seed 운(블록 부트스트랩 CI 0 포함). 살아남은 것: MLP>GBM ≈−0.7cm(3-seed), Stefan 라벨의 placebo 대비 정보성, TabM 안정화. 재실험 조건: 거리버퍼·블록부트스트랩·multi-seed·nested 선택(`aug_within_alaska.py` 수정 필요).
- **timelapse(완료)**: 연별 지도는 물리 forcing 최선(연도 홀드아웃 14.97cm, R² 0.34). **연도 간 anomaly는 예측 불가(corr 0.06)** = 시간 신호의 예측 불가능성 정량화가 정직 산출. GIF `outputs/animations/timelapse_alt_alaska.gif`.
- **얕은 3D(검증 통과)**: 알래스카 0-3m 실측 764행, 필드 2.66°C·R² 0.47, 0°C→ALT r 0.28(심부 0.16 대비 개선, 절대 정합 미완).
- **신규 KPDC(16:22 추가)**: 콘슬 8층 토양온도(L1-L8)+VWC+CO2/CH4(ID0-ID5, 2021-2022), 쿠가록 화재/비화재 토양온도·수분(2019-2022), ID01-13 일별 VWC(2023-2025), 2016 토양물성(Thaw depth 실측 포함!), AWS 2023·2025. → E 지점 실측·콘슬 ALT 유도·화재 교란 사례연구 재료. 파싱 미착수.
- **비판적 검토** `docs/CRITICAL_REVIEW_2026-07-20.md`: 점 검증 상한(~12cm 대표성 잡음)에 갇힘 → 다음 지렛대=면적 검증(다중프로브·InSAR), 서사=신뢰·물리·UQ/AOA·3D·timelapse.

## 이전 완료(2026-07-20) — W3 물리결합 엔진 → `docs/EXPERIMENT_W3_2026-07-20.md`
가설 "토양 E(x)·물리식 형태강제 ML(구조 C)이 전이 회복" **기각**(적대적 검증). 8모델 공간블록+LORO.
- PHYS_const(상수 E) LORO 18.24cm가 여전히 최선. PHYS_soil 19.99·PHYS_nn(미분물리층) 25.72로 악화. 물리 형태강제 순효과 음수. E(x)는 covariate shift 재수입.
- **결정적**: 모든 모델의 레나 skill 음수(−0.02~−2.09) → **라벨 없는 OOD 전이는 모델 구조로 못 뚫는다**. 병목은 레나 공변량-ALT가 알래스카서 학습 불가.
- **방향 조정**: (1) 물리 상수 앵커+AOA 게이팅 정직 배포, (2) 타깃 지역 라벨 확보(W1 백본)가 유일 실효 지렛대(레나 라벨 있으면 in-domain 17.8cm), (3) "전이 풀었다" 아닌 "AOA 표시+물리 앵커" 프레이밍. 추가 모델 구조 실험 소진 말 것. W2.2 우선순위 하향(결측은 증상, 근본은 covariate shift).
- HYBRID_aoa 실용 최선(공간블록 17.56)이나 전이서 PHYS_const로 수렴. 스크립트 `w3_physics_ml.py`.

## 이전 완료(2026-07-20) — W2.1 SoilGrids + KPDC → `docs/EXPERIMENT_W21_KPDC_2026-07-20.md`
- **SoilGrids 토양 공변량**(ISRIC WCS로 취득, VRT는 정체로 실패): 게이트 **미채택**. 내삽(공간블록)은 개선(skill +5.6%)이나 **전이(LORO)는 붕괴(−63.8%, 레나 62.6cm)**. 토양은 전지구 커버라 결측 없는데도 전이 실패 = **진짜 covariate shift**(토양-ALT 관계가 지역마다 다름). → 공변량 추가는 내삽 이득·전이 손실. **전이 escape는 물리뿐**. 토양은 물리 E(x) 입력으로 재활용(W3). 산출 `dl_dataset_cell_v3_soil.csv`, `soil_ablation_gate.csv`.
- **KPDC 콘슬 현장 검증**: ERA5 √TDD가 실측과 정합(bias ~0.1)=공변량 backbone 검증(대회 KPDC 활용). 단일 E Stefan은 콘슬 약 1.7배(ratio 1.68) 과대예측 → **E(x) 동기**(토양·W3로 수렴). 단 콘슬 44셀 교정은 **in-domain**(전이 아님)이고 PHYS_nn 의 bias 감소는 **평균회귀**이므로 전이 일반화 근거로 쓰지 않는다. 산출 `kpdc_station_climate.csv`, `kpdc_era5_validation.csv`. Sentinel·AlphaEarth는 GEE 미설치로 미취득.
- **순서 재조정**: covariate shift가 전이 근본 병목이므로 **W3(물리 base 고도화 + 토양 의존 E(x))가 W2.2보다 전이엔 우선**. 다음=W3.

## 이전 완료(2026-07-20) — Phase 1 회의적 재검증 → `docs/EXPERIMENT_PHASE1_2026-07-20.md`
연구 프로그램 `docs/RESEARCH_PROGRAM_2026-07-17.md`(증강 백본 W1 최상단 재배치) 착수. 기존 P2 결론 2건을 반증 설계로 재검증(적대적 검증 통과):
- **라벨 증강 "해가 된다" 부분기각**: 증강 자체가 아니라 "심부 GTNPenv 라벨 + 결측 모달리티(신규지역 InSAR/PolSAR 100% 결측=완전 공선)" 결합만 붕괴(레나 22→88cm). 물리·기후만 ML은 면역. → 백본 게이트 = 결측 처리(W2.2). 스크립트 `aug_backbone_dissect.py`.
- **3D "지형+CCI 심부 개선" 기각**: site-GKF 누설 착시(72.6% 사이트가 같은 0.5°블록 공유). 누설통제 시 악화(LORO 1.60→1.73°C). 정보병목 재확인. 스크립트 `field3d_reeval.py`.
- 평가 프레이밍 정정: `docs/EVAL_FRAMING_NOTE.md`(공간블록≠전이, 16.95는 알래스카 내 공간블록).
- 다음: W2.2 결측 재설계 → 증강 안전화, W3 물리 base 고도화(Kudryavtsev), W2.1 데이터 확충. 유도 라벨은 held-out 실측 게이트 통과 전 백본 제외.

## 이전 완료(2026-07-14 저녁) — P2 실험 3트랙 + 슬림 덱(11p)
- **P2 결과** → `docs/EXPERIMENT_P2_RESULTS_2026-07-14.md`. 스크립트 `scripts/3_deep_learning/p2_{augment,field,stefan}_experiment.py`.
  - **물리 우선이 전이에 강함(핵심)**: Stefan 아핀(a+E√TDD) LORO 18.2cm ≫ 순수 ML 40.6cm(알래스카 과적합, Lena bias +86cm). 잔차학습 무익(REJECT).
  - 라벨 증강 미채택(GTNPenv 심부 37셀이 Lena OOD서 128cm 과대추정 교란. 단 Lena 3,037셀 자체는 유효 skill +0.154). 3D 기질 전 공변량 소폭 이득(1.23→1.18°C, ADOPT 잠정).
  - **후속 필수**: 트랙 α 공간블록+LORO 재평가(site-GKF 누설 의심), 트랙 M GTNPenv AOA 게이팅 재실행, 물리 base 고도화(Kudryavtsev류).
- **슬림 발표덱 11p**: `deck/build_summary.py` → `deck/render/permafrost_summary.{pptx,pdf}`. ALT main·3D 증강·Stefan 프레임, 용어정의 상세, 완료/진행/예정+대회일정. 신규 그림 `deck/mk_summary_figs.py`(connection·uncertainty_map·magt_clean). 상세 21p는 `permafrost_midreport` 보존.

## 이전 완료(2026-07-14 오후) — P0·P1 실행 + PPT 반영 → `docs/EXPERIMENT_P0_P1_RESULTS_2026-07-14.md`
- **P0**: 데이터 인벤토리 세계지도(`outputs/maps/data_inventory_world.*`) + 6모델 예측·오차 지도(`tournament_{pred,error}_maps.*`). 위치가중 순위 GBM 16.1 ≈ Diffusion 16.2 ≈ 앙상블 16.2cm(동률 재확인).
- **P1**: 다지역 통합 셀 v2 `data/processed/dl_dataset_cell_v2.csv`(17,423행: +레나델타 3,037·GTNPenv 37·QTP 1). 하네스 `scripts/3_deep_learning/unified_tournament_cell.py`(전 공변량 25·공간블록+LORO·GPU). 결과 `unified_tournament_*.csv`.
  - LORO 전이서 **DL(FT-T·앙상블) 15.0cm가 GBM 17.6 상회**(알래스카, 이질 소표본 첫 DL 이점). 레나 전이는 25~30cm로 병목 지속.
  - **결측 라우팅 아티팩트**: NaN 네이티브 GBM이 "InSAR 결측=깊은 ALT" 오학습(레나 50.6cm). 다지역은 중앙값 대체+플래그 필수.
  - **통합 학습 게이트 미채택**(현 결측 처리서 알래스카 in-domain 저하 18.1→20.9). 전이 병목 정량화는 채택. 후속=모달리티 드롭아웃·Stefan.
- **PPT v3 → 21슬라이드**: `deck/build_midreport.py`(신규 5b 인벤토리·13b 6모델오차·15b 다지역전이 + 1차 품질개선). 그림 `deck/mk_p0p1_figs.py`. 렌더 `deck/render/permafrost_midreport.{pptx,pdf}`.

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
