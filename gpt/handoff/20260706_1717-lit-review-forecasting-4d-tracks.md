# Handoff: 논리 검증 + 문헌 재조사 + 예측(T1)·4D(T2) 트랙 신규성 판정

**Project**: Polar_Bigdata (영구동토 borehole/CALM → ALT 2D 지도 + 얕은 3D 열구조 + 전이 + 불확실성)
**Date**: 2026-07-06 17:17
**Session focus**: (1) 지금까지의 연구 논리 검증, (2) DL 입출력 정밀 감사, (3) 문헌 49편 재조사·검증·정리, (4) "ALT 시계열 예측(T1)"·"4D=3D+time(T2)" 신규성 판정, (5) 모델 로드맵.
**Author**: Claude Fable 5 + willy010313
**근거**: 결과 CSV(`data/processed/*.csv`), DL 스크립트 코드, 문헌 워크플로(57 에이전트·49편 web-검증, `references/INDEX.md`).

---

## 0. TL;DR (GPT가 먼저 읽을 것)

1. **정확도 서사에 두 가지 과대 프레이밍이 있다(수정 필요)**:
   - "평탄툰드라 12.97cm = SOTA급 floor 돌파"는 **대부분 범위 축소 아티팩트**다. skill-over-mean(=1−RMSE/자기SD)으로 보면 전역 10.4~12.3% > 평탄툰드라 7.4%. 즉 큐레이션은 절대 RMSE만 낮추고 설명력(R²)은 오히려 떨어뜨린다. "돌파"가 아니라 "레짐별 지배 정보원이 다름(평탄지=PolSAR만 유효)".
   - "17cm = 물리 하한(floor)"은 과대표현. 점-반복 비가역 잡음은 ~4cm(분산분해 within 12.3%). 우리는 R²≈0.2로 covariate 정보 병목 상태이지 물리 하한이 아니다. **아직 안 먹인 모달리티(SoilGrids·Sentinel-1 InSAR 시계열·Sentinel-2)가 남아 있어 헤드룸 존재.**
   - **잘한 것(유지)**: 누설인지 CV(공간블록+LORO), 신경장 킬스위치, 앙상블 게이트, 부트스트랩 동률판정, 분산분해. 방법론 규율은 상위권.
2. **신규성의 무게중심을 "정확도"에서 "transfer-honest + calibrated cell-wise UQ + shallow 3D"로 옮겨야 한다.** 이 세 축의 결합은 permafrost ALT 도메인에서 아무도 안 했고, 2025 리뷰(Koven, `s43503-025-00080-8`)가 (a)전이학습 미평가, (b)UQ 미탑재를 명시적 open gap으로 인증한다.
3. **T1(지점 ALT 시계열 예측) = 이미 붐빔(신규성 낮음~중간).** Rahaman2025(arXiv:2510.06258, Alaska soil-T sequential DL bake-off)·Luo/Zhang2022(QTP thaw depth LSTM/CNN/RF)가 거의 동일. 순수 아키텍처 비교는 derivative. **단, target을 soil-T가 아닌 ALT(cm)로 두고 + transfer + calibrated UQ를 붙이면 방어 가능.**
4. **T2(4D = subsurface 0–20m thermal volume + time) = 상대적으로 열림(신규성 중간~높음, 가장 유망).** 단 "2100 projection" 헤드라인은 Gautam2025/Ran2021이 이미 선점. **빈 공간 = subsurface volume + time + transfer + calibrated UQ 4중 결합.**
5. **모델 로드맵**: 정적 2D=GBM+quantile/conformal 유지(DL 무익), 3D 정적=GBM 조건장+물리/UQ 레이어(from-scratch neural field 금지, 이미 패배), 4D/예측=operator learning(FNO/DeepONet)+GIPL2 물리 사전학습+conformal UQ. **DL 투자는 오직 시간축·field-to-field 물리 대리모델에 집중.**

---

## 1. 지금까지 한 것 (현재 상태 — 논리 검증 반영)

| 축 | 결론(원래) | 검증 후 정정 | 근거 |
|---|---|---|---|
| 모델 선택 | GBM≈DL, 정보병목 | ✅ 견고. 단 부트스트랩 검정력≈0(공간블록 6개)이라 "동률"은 사실상 자명 | `model_tournament_{results,significance}.csv` |
| 정확도 floor | 17cm=물리하한 4중확증 | ⚠️ **과대**. 비가역 잡음 ~4cm, 현재 R²≈0.2 = covariate 병목. 집계실험은 RMSE 악화(17→23)라 대표성하한 논리 미지지 | `insar_ablation`·`polsar_residual`·`grid_support`·`areal_eval` |
| 큐레이션 12.97cm | SOTA급 첫 돌파 | ⚠️ **범위축소 아티팩트 지배**. skill-over-mean 전역 10.4% > 평탄툰드라 7.4%. 진짜 발견=평탄지에서 공변량 무력·PolSAR만 유효 | `curated_scope_results.csv` |
| 전이(covariate) | ERA5-Land가 전이 20%↑ | ✅ 견고·가장 경쟁력 있는 결과. Ran2022 SOTA 86.9cm와 동급(87.3cm) | `stage2_era5_rescore.csv` |
| 3D 엔진 | 신경장 탈락, GBM 조건장 | ✅ 견고. 조건장이 전이에서 IDW 이김(1.40<1.69) | `b1_neural_field_results.csv`, `b1b_results.csv` |

**skill-over-mean 검증표** (= 1 − RMSE/각scope SD; 높을수록 좋음):

| scope | ALT SD | 최고 RMSE | skill vs mean | R² |
|---|---|---|---|---|
| 전역(다양지형) | 19.3 | 16.95~17.3 | **10.4~12.3%** | **0.20** |
| 완만 | 18.2 | 16.58 | 8.9% | 0.17 |
| PolSAR유효 | 17.2 | 16.3 | 5.2% | 0.10 |
| 평탄툰드라 | 14.0 | 12.97 | **7.4%** | **0.14** |

→ 큐레이션이 **설명력을 낮춘다**는 것이 명확. 12.97은 "더 잘 맞힘"이 아니라 "덜 변하는 대상". SOTA 11–12cm와 직접비교 금지(우리가 이미 아는 caveat을 12.97에도 적용해야 함).

---

## 2. #3 — 현재 DL 모델 입출력 (코드 근거 확정)

### 2D ALT 모델 (`dl_dataset.csv`, 225,421점 / 고유위치 ~5,900, 공간블록 CV)
- **입력 14종(스칼라)** — 정의: `scripts/1_data_prep/assemble_dl_dataset.py:23`
  - 지형 6 (Copernicus DEM 30m): `dem_elev, dem_slope, dem_aspect_sin, dem_aspect_cos, dem_tpi, dem_rough`
  - 기후 8 (**ERA5-Land 2015–2020 월별 평균 = 정적, 시간축 없음**): `e5_maat, e5_tdd, e5_fdd, e5_sqrt_tdd, e5_twarm, e5_tcold, e5_stl1, e5_swe`
  - (선택, 큐레이션 실험만) 물리관측 4: `polsar_alt, polsar_std, insar_sub, insar_alt`
- **출력**: `alt_cm` (활성층 두께, cm). 학습은 `log1p(alt_cm)` 로그공간 → `expm1` 역변환.
- **모델(토너먼트 6+1)**: GBM(HistGBR), MLP(14→256→128→64→1), FT-Transformer(feature-tokenizer d=64, 8head×3block), TabM(공유trunk+8head), Flow matching(rectified flow), Diffusion(DDPM T=100), 앙상블(0.5·GBM+0.5·FT-T). 생성모델은 90% 커버리지/샤프니스도 산출.
- **파생 실험**: B0(InSAR weak 403.6만 사전학습 MLP→미세조정), B0b(CCI 피처 추가), PatchCNN(DEM 패치+스칼라), insar/polsar ablation.

### 3D 열구조 모델 (`ground_temp_all.csv`, 10,747점 / 260 시추공)
- **입력**: 기후 8종 + 깊이인코딩(`depth_m`, `logd`; v2는 Fourier 10종). **출력**: `temp_c` (지중온도 °C, 0–30m).
- **모델**: NF(neural field MLP)·GBM·IDW-3·최근접. **NF 탈락**(site 2.36 vs GBM 1.31, IDW 1.30) → 3D 엔진=GBM 조건장.
- **평가**: site-wise GroupKFold(5) + leave-one-region-out(전이).

**핵심 시사점(T1/T2 직결)**: 현재 파이프라인엔 **시간축이 전혀 없다.** 예측(T1)/4D(T2)로 가려면 시간분해 공변량(연/월별 ERA5-Land + CMIP6)과 시간분해 라벨(CALM 사이트별 연간 ALT 시계열, 수십 년)로 **데이터 스키마 자체를 재구성**해야 함. 기존 데이터로 실현 가능한 피벗.

---

## 3. #2 — 모델 로드맵 (지금 → 앞으로)

> 대전제: 정적 매핑에서 GBM≈DL인 이유는 covariate 정보병목이지 모델용량 부족이 아님. **DL은 covariate가 실제로 더 풍부한 신호를 담는 곳(=시간축, field-to-field 물리)에서만 가치.**

- **(a) 2D 정적 매핑 → GBM 유지, DL 쓰지 말 것.**
  - 주력: Gradient Boosting + **Quantile 목적함수(QRF/quantile-GBM)** = per-cell 구간 무료.
  - UQ 상층: **conformal(Singh2024) + spatial-aware conformal(Lou2025)** = covariate shift 하 coverage 보장.
  - 전이 정직성: **AOA masking(Meyer2021) + kNNDM CV(Linnenbrink2024)**.
  - 신규성은 모델이 아니라 **UQ+transfer 계층**.
- **(b) 3D 정적 → from-scratch neural field 금지(이미 패배). GBM 조건장 + 물리/UQ 레이어.**
  - 물리 정규화(Stefan/heat PDE)는 residual/ablation으로만. Ieki2025 neural kriging·InterPIGNN은 비교 baseline.
- **(c) 예측/4D → 여기서만 DL이 진짜 값어치. 단 아키텍처 신규성에 기대지 말 것.**
  - T1: **physics-guided LSTM/GRU(Read2019 PGDL 템플릿) = GIPL2 합성 사전학습 → 관측 ALT 미세조정 + energy/Stefan-loss**. re-bake-off 금지.
  - T2: **operator learning FNO(Li2020)/DeepONet(Koric2023)** — DeepONet의 branch/trunk가 불규칙 borehole+연속 (x,z,t) query에 유리. GIPL2 물리를 amortize하는 field→field 대리모델. + shallow→deep 시공간 GNN(Liu2024 PIGNN) 이식.
  - 공통: **conformal/deep-ensemble UQ** 부착(vanilla FNO/DeepONet은 UQ 없음). 시간 forcing이 실제 신호를 줄 때만 이득(정적-feature면 GBM과 tie 위험).

---

## 4. #1 — 문헌 랜드스케이프 (49편 web-검증, `references/INDEX.md`)

카테고리: benchmark 3 · alt_dl_mapping 5 · alt_forecasting 10 · spatiotemporal_4d 6 · uq_transfer 13 · physics_ml 11 · context 1. (오픈액세스 PDF 39편 다운로드 완료, 나머지 페이월은 링크.)

### 4.1 우리가 이미 가진 논문 중 OFF-DIRECTION (인용 축소·재배치)
| 논문 | 사유 | 처리 |
|---|---|---|
| **Biskaborn2019**(warming) | ALT지도·ML 아님, 온난화추세 종합 | intro/동기 + GTN-P 검증표준으로만 |
| **InterPIGNN(Aljubran2024)** | 우리 3D neural field 폐기됨 → 현 엔진 기준 borderline | physics-reg **ablation**으로 강등 |
| **SIREN(Sitzmann2020)** | 우리가 실패한 좌표 neural field 재현 위험 | 실패모드 경고 + 조건부 method로만 |
| **Groenke2023** | 1D per-borehole Bayesian(공간지도 아님) | UQ **precedent**로 유지(공간 cell-wise UQ로 차별화) |
| Yurtsever2023 / Wang2024 / Li2025(ViT RTS) | 농업 soil-T / 인공동토 / thaw slump 탐지 = 태스크 상이 | context 1문장 |

### 4.2 새로 추가·상향해야 할 핵심 (우리 방어축의 무기고)
- **전이 채점 백본**: kNNDM(Linnenbrink2024, GMD) · NNDM(Milà2022) · AOA(Meyer2021/2022).
- **UQ 스택**: QRF(Lu2022) baseline → conformal(Singh2024) → spatial-aware GeoConformal(Lou2025).
- **전이 설계 템플릿**: O'Malley2026 in-context continental subsurface T(재학습 없이 신규위치 적응).
- **물리-예측 청사진**: Read2019 PGDL + Liu2023 PI-LSTM(GIPL2 사전학습) + Willard2020 survey(전략 명명).
- **4D 이식용**: Liu2024 PIGNN(shallow→deep 시공간) · FNO/DeepONet · Ieki2025 neural kriging.
- **앵커 인용**: Koven2025 리뷰(`s43503-025-00080-8`) — 전이·UQ를 open gap으로 명시.
- **보강 권장(미보유, 검증 후 추가)**: CryoGrid 원논문(Westermann 2016/17, GMD) — CCI 벤치마크 근거; CALM/GTN-P 데이터 논문(Shiklomanov/Streletskiy) — 검증표준.

### 4.3 검증에서 잡은 메타데이터 오류(인용 전 수정)
- **Ran2022 공저자 일부 날조**: 정정 = Ran, Li, Cheng, Che, Aalto, Karjalainen, Hjort, Luoto, Jin, Obu, Hori, Yu, Chang.
- **"Rahaman2024 North Slope"는 오귀속** — 실제 저자 = **Chance, Ahajjam, Putkonen, Pasch**(Earth Sci Inf 2024, MERRA-2 기반, GBDT/RF/SVR). Rahaman2025(arXiv:2510.06258)와 **다른 그룹**. "같은 저자가 Alaska T1 점유" 서사는 폐기.
- **"Suzuki 2025 neural kriging"는 오귀속** — 실제 = **Ieki et al. 2025**(Geothermics).

---

## 5. #4·#5 — 새 트랙 T1(예측)·T2(4D) 신규성 판정 + 문헌

### T1 — 지점 ALT 시계열 예측 (LSTM/GRU/Transformer): **붐빔, 신규성 낮음~중간**
- 직접 경쟁자: **Luo/Zhang2022(STOTEN, QTP thaw depth LSTM/CNN/RF)**, **Rahaman2025(arXiv:2510.06258, Alaska soil-T TCN/Transformer/GRU/BiLSTM + CMIP5)**, **Read2019 PGDL**(physics-guided LSTM 아키텍처 선점).
- 판정: 순수 "Alaska에서 sequential DL 비교"는 derivative. **재-bake-off 금지.**
- 남은 차별화(구체): ① target = soil-T 아닌 **ALT(cm) from 과거 ALT**(Stefan free-boundary 위치, 물리적으로 다른 양) ② **Alaska→Siberia/QTP transfer(LORO/kNNDM)** — 경쟁자 전부 단일지역 ③ **per-horizon conformal 예측구간** ④ "정적일 땐 GBM≈DL, 시계열일 땐 DL 이득"을 실증 대조로 프레이밍.

### T2 — 4D (subsurface 0–20m thermal volume + time): **상대적으로 열림, 신규성 중간~높음(가장 유망)**
- 부분 선점: **Gautam2025·Ran2021(SSP-to-2100 projection)**, **Kriuk2025(arXiv:2510.02189, 대규모 projection+공간 ensemble-std UQ; 단 target=risk/fraction, calibrated 아님, transfer 아님)**, **CCI v4(연별 고정심도 맵; forward, UQ/transfer 없음)**, **McMillen2026(3D→2D 구조보존; 단 canopy·정적)**.
- 판정: **"ALT를 2100까지 project + spread" 헤드라인은 즉시 먹힘.** 진짜 빈 공간 = **subsurface volume(식생 아님) + time + inter-region transfer + calibrated cell-wise UQ 4중 결합** — 어느 논문도 안 함.
- 이식할 method: **Liu2024 PIGNN(shallow→deep 시공간 GNN)** + **DeepONet/FNO**(GIPL2 amortize) + **O'Malley2026**(transfer 설계). 단 우리 neural field 실패 이력상 **GBM 조건장 위 물리·UQ 레이어**로만 실현.

**권고 포지셔닝**: 신규성을 "새 아키텍처/새 맵"이 아니라 **"영구동토 ALT/shallow-3D에 대한 transfer-honest + calibration-guaranteed 평가·산출 프레임워크"**로. **T2를 flagship**, T1은 UQ·transfer가 붙은 보조 트랙.

---

## 6. 다음 스텝 (우선순위)

1. **정정 먼저(반나절)**: 문서(SESSION_HANDOFF/EXPERIMENTS/EXPERIMENT_LOG)에서 "12.97 floor 돌파"·"17cm 물리하한"을 skill-over-mean 병기로 재프레이밍. 모든 RMSE 옆에 skill/R² 병기 규약화. 12.97에 토너먼트와 동일한 블록 부트스트랩 적용.
2. **전이+UQ를 헤드라인으로 승격(1일)**: Alaska→Siberia(GGD200)/QTP LORO + kNNDM CV + AOA 마스크. Diffusion 90% 커버리지 74%(과신) → conformal 보정 → 셀별 분위수 UQ 완성.
3. **T2 파일럿(2–3일)**: 시간분해 데이터 스키마(연별 ERA5-Land + CALM 연간 ALT 시계열 + CCI 1/2/5/10m) 조립 → GBM 조건장 + DeepONet 대리모델 + conformal UQ. "2100 projection" 아닌 "transfer+UQ 붙은 shallow 4D thermal" 프레이밍.
4. **(정확도 지렛대, 병렬)**: SoilGrids(서버복구)·Sentinel-1 InSAR 시계열·Sentinel-2 = floor 자체를 낮출 유일 경로. 모델 교체·큐레이션 아님.

---

## 7. Pointers
- 논문 인벤토리: `references/INDEX.md` (49편, 카테고리별 폴더 `references/0X_*/`, 오픈액세스 PDF 다운로드됨).
- 결과 CSV: `data/processed/{model_tournament_*,curated_scope_results,insar_ablation_results,polsar_residual_results,grid_support_results,areal_eval_results,stage2_era5_rescore,b0*_results,b1*_results}.csv`.
- DL I/O 정밀맵: 본 문서 §2, 코드 `scripts/1_data_prep/assemble_dl_dataset.py`, `scripts/3_deep_learning/*.py`.
- 문헌 종합 원본: 워크플로 리포트(§4·§5 근거).
- 관련 이전 핸드오프: `gpt/handoff/20260706_1113-tournament-floor-curated-demo.md`.

## 8. Caveats for GPT
- **12.97cm를 "SOTA 돌파"로 인용 금지** — 범위축소 아티팩트. skill-over-mean은 전역이 더 높음.
- **"17cm=물리하한" 인용 금지** — 비가역 잡음 ~4cm, 현재는 covariate 병목(R²≈0.2), 헤드룸 존재.
- **MAGT ≠ ALT** (연평균 지중온도장 ≠ 계절 최대융해). "0°C 등온면=ALT" 비교는 개념오류.
- **T1은 붐빔** — 순수 LSTM/GRU/Transformer 비교로 논문화 금지. target(ALT)·transfer·UQ 조합 필수.
- **저자 오귀속 3건 수정**(§4.3): Ran2022 공저자 / Rahaman↔Chance North Slope / Suzuki↔Ieki.
- **from-scratch neural field/SIREN 재시도 금지** — 이미 GBM/IDW에 패배.
