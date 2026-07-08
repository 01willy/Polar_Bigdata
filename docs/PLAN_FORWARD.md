# Polar_Bigdata — 앞으로의 연구 세부 계획 (Forward Plan, 확정본)

작성 2026-07-06 · 근거: 세션 검증결과 + GPT 방향분석(`gpt/polar_bigdata_research_direction_for_claude.md`) + 문헌 49편(`references/INDEX.md`).
**우선순위(사용자 확정)**: ① 데이터 활용량·규모 + 기술적 차별성(연구 본체) → ② 임팩트·시각화(발표). 대회 형식(보고서/데모/PPT)은 후순위. 마감 여유 1개월+.

이 문서는 **"어떤 모델로 / 어떤 input→output을 / 어떤 데이터로 / 어떤 차별성을 내고 / 어떻게 시각화하나"**를 스레드별로 못박는다. `PLAN.md`·`PLAN_3D_ALT.md`를 대체하는 go-forward 기준.

---

## 0. 연구 정체성 (한 줄)

> **대규모 다중모달 극지 빅데이터(위성 SAR/InSAR/광학 · 재분석 기후 · 고해상 지형 · 전지구 시추공)를 정직하게(누설통제·AOA·보정 UQ) 융합해, 북극 활성층 두께(ALT) 2D 지도 + 얕은 3D 지중 열구조를 만들고 — 어떤 데이터가 동토 열상태를 지배하는지 분해하며, 알래스카→타 지대 전이를 검증한다.**

- **기술 차별성 4종**(문헌상 official gap): transfer-honest 평가(kNNDM/AOA) · calibrated cell-wise UQ(conformal) · 다중모달 big-data ablation · 관측기반 얕은 3D 보간(vs forward GIPL/CCI).
- **데이터 차별성**: 22GB+ 보유 데이터의 활용을 극대화 + 신규 모달리티 추가. "얼마나 많은 데이터를, 얼마나 정직하게 녹였는가"가 핵심.

---

## 1. 원칙 (정정된 사실 기반 — 발표에 그대로)
- 헤드라인은 **공간블록/LORO/kNNDM CV**만(무작위 CV는 4배 과대평가 → 참고용).
- 정적 ALT에서 **GBM≈DL**(정보병목=공변량). 모델 고도화가 아니라 **데이터·평가·불확실성**이 지렛대.
- **정정**: "17cm=물리하한" → "현재 공변량의 정보병목(비가역잡음 ~4cm)"; "12.97cm=SOTA돌파" → "레짐별 지배 정보원 차이". 모든 RMSE 옆에 **R²·target_SD·skill-over-mean** 병기.
- 3D는 **GBM 조건장**(from-scratch neural field 폐기). ALT(계절 최대융해) ≠ MAGT(연평균장) 개념 분리.

---

## 2. 데이터 전략 — 활용 극대화 + 추가 (심사 1순위 지렛대)

**현황: 보유 22GB, 그러나 2D 모델은 정적 스칼라 14개로 증류.** 활용률을 끌어올리는 게 최대 기회.

| 데이터 | 용량 | 지금 활용도 | 계획 |
|---|---|---|---|
| PolSAR ALT (P-band 30m) | 7.0G | 큐레이션 실험만 | **ablation 정식 편입**(ABoVE 한정 feature) |
| InSAR ReSALT(침하/ALT) | 6.9G | weak label 사전학습만 | **계절 침하 feature + weak-label 확대** |
| Copernicus DEM 30m | 4.3G | 지형 6개로 축약 | 미세지형 추가 파생(습윤지수·곡률·다중스케일 TPI) |
| GTN-P 시추공 | 933M | 부분 | **3D 라벨 전량 파싱**(현 35→전지구) |
| ESA CCI(1km, 0/1/2/5/10m, 25년) | 828M | 최근접 baseline만 | **3D prior feature + 연별 시간축(T-lite)** |
| ERA5-Land 월별 | 328M | **2015–2020 평균만** | **연별/월별 시계열로 확장**(T-lite 핵심) |
| 시추공 GGD200/G10015/PERMOS | 258M | ground_temp_all | 3D 라벨 확대 |
| ABoVE ALT 점 | 117M | 라벨 224k | 유지 |

**신규 취득(데이터 규모 +차별성)**:
- **SoilGrids**(ISRIC 서버복구): SOC·점토/모래·용적밀도·지빙 proxy = **열전도/잠열 물리 feature**.
- **Sentinel-1 InSAR 시계열**(ASF/Earthdata): 대규모 계절 침하 = ALT 직접신호(제품 아닌 원신호).
- **Sentinel-2 / MODIS**: NDVI·landcover·LST = 표면 에너지·유기층 proxy + 시간축.
- **TPDC QTP 시추공**(84공, 3/6/10/20m): 3D 라벨 + **전이 target(알래스카→QTP)**.

> **원칙(정직성)**: 데이터를 무작정 넣는 게 아니라 **feature-group ablation으로 각 모달리티의 실제 기여를 정량**하고, 도움 안 되는 것도 정직히 보고. "많이 썼다"가 아니라 "많이 쓰고 무엇이 왜 통하는지 분해했다".

---

## 3. 연구 스레드 (모델 / input→output / 데이터 / 차별성 / 시각화)

### 스레드 R — 데이터 재구조화  ★최우선 (2026-07-08 재정렬)
> **왜 최우선**: apparent-floor 진단으로 "17cm 병목은 모델이 아니라 **데이터를 잘못 넣어서**"임이 드러남. 새 데이터 취득보다 **이미 가진 데이터를 올바른 구조로 넣는 것**이 더 싸고 효과 큼. 비가역하한 7.2cm(현재 16.9)도 여기서 내려감.
- **㉠ 시간정합(temporal alignment)**: 지금은 (위치,연도) 라벨 전부에 **2015–2020 평균 기후**를 붙임 → 연도차가 잡음. 각 라벨에 **그 해(+전년 겨울) ERA5-Land 월별→연 파생**을 붙여 연도차를 **신호로 전환**. 데이터는 이미 보유(`raw/era5land` 328M, 지금은 평균내 버림).
- **㉡ 집계단위 정합(scale match)**: 라벨 30m 점 vs 기후 9km 셀 불일치 → 셀 단위로 **평균 ALT + 셀내 분산(불확실성 라벨)** 집계, 또는 pseudo-replication 제거. "같은 X, 다른 y"를 "같은 X, 같은 y(+분산)"로.
- **㉢ 가중/중복정리**: 관측 조밀 셀(예 200점)이 희소 셀(1점)을 200배 압도하는 편향 → 위치당 대표/가중치 1/n.
- **input→output**: 재구조화된 `dl_dataset_temporal.csv`(시변) / `dl_dataset_cell.csv`(집계) → 스레드 A/D 입력.
- **게이트(R3)**: 동일 GBM으로 **정적 vs 시간정합 vs 집계** 3버전을 공간블록+LORO+per-year holdout로 비교. **시간정합이 within/transfer skill을 올리고 floor를 낮추면** → 시계열 모델(스레드 D) 확장. 미개선이면 집계본만 채택.
- **시각화**: floor 이동(7.2cm↓?) 막대 · 정적 vs 시간정합 skill 비교 · 연도별 ALT 변동을 담아낸 예측 산점.

### 스레드 A — ALT 2D 매핑 + 다중모달 feature ablence  ★본체 (스레드 R 위에서 재실행)
- **모델**: 주력 **GBM(HistGradientBoosting)** + **Quantile-GBM**(구간). 비교군 FT-Transformer/TabM(capacity 병목 아님 증거, 헤드라인 아님). 영상 모달리티(Sentinel/SAR 패치) 추가 시 **tabular+CNN 멀티모달**을 ablation M7/M8에서만 게이트 검증.
- **input → output**: [지형6 + 기후8 + **토양(SoilGrids)** + **식생(NDVI/landcover)** + **SAR(PolSAR/InSAR)** + **CCI prior**] → **ALT(cm) + 90% 예측구간 + AOA 마스크**.
- **데이터**: dl_dataset(225k) 확장 + PolSAR/InSAR/SoilGrids/Sentinel 편입.
- **차별성**: 다중모달 **feature-group ablation**(M0 mean→M9 CCI prior)으로 "무엇이 ALT를 지배하나"를 누설통제 하에 분해 + transfer(LORO/kNNDM) + 보정 UQ + AOA. **pseudo-replication 교정**(연별/셀집계).
- **시각화**: ablation **waterfall** + 그룹별 기여 공간지도 · ALT 예측지도(냉색 oslo_r) · 관측 오버레이 · 오차지도.
- **선결 정리**: 현재 "같은 위치 여러 연도를 정적 피처로 쌓음" → 셀단위 집계 또는 연별 피처로 재구성(§스레드 D와 연결).

### 스레드 B — 얕은 3D 지중 열구조 (0–20m)  ★3D 산출
- **모델**: **GBM 조건장 + depth encoding**(Fourier) + 물리 단조성/residual 체크. (neural field 폐기 유지.) baseline IDW/kriging/nearest.
- **input → output**: [기후8 + 지형 + **토양(SoilGrids)** + **CCI 지중온도 0/1/2/5/10m prior** + depth] → **T(x,y,z≤20m) 볼륨 + 0°C 등온면(동토 table) + 불확실성**.
- **데이터**: ground_temp_all(10,747점/260공) + GTN-P 전량 파싱 + CCI + 시추공 3망 + TPDC(전이).
- **차별성**: **관측기반 보간**(vs forward GIPL/CCI) + 전이(알래스카→QTP/시베리아) + 셀별 UQ, **tri-mesh 등온면 출력**.
- **시각화**: **PyVista 인터랙티브 3D 열큐브 + 0°C 등온면** · 깊이 슬라이스 **GIF** · 위도-깊이 단면 · 시추공 프로파일 3D 배치.

### 스레드 C — "apparent floor" 진단 (데이터 마스터리 스토리)  ★차별화 특별전
- **모델 아님 — 분석**. pseudo-replication(같은 공변량 셀 ALT 34–96cm 공존) + 척도불일치(라벨 30m vs 기후 9km) + 분산분해 → "17cm는 물리벽이 아니라 정보병목"을 정량·시각 증명.
- **차별성**: 대부분 참가작이 무작위 CV로 과대낙관할 때, **우리는 우리 데이터의 한계를 진단하고 다음 지렛대(새 모달리티)를 제시** = 정직성·깊이의 차별.
- **시각화**: 동일-공변량-셀 ALT 산포 1장 · 분산분해(위치간/내) · support↑→RMSE↑ 곡선.

### 스레드 D — 시간 forcing 파일럿 (T-lite, 게이트)  ★스트레치·4D-lite wow
- **모델**: **GRU(월별 ERA5 시퀀스)** vs **GBM annual-summary** vs **static**(3자 비교).
- **input → output**: [CALM site-year ALT + 해당연/전년 **월별 ERA5-Land** + 정적 지형/토양] → **그 해 ALT**(확장 시 미래 ALT).
- **데이터**: CALM site-year 시계열 + **ERA5-Land 연별/월별**(현재 미활용 시간축) + CCI 연별.
- **차별성**: pseudo-replication을 **신호로 전환**("연도가 잡음→정보"). 시간 forcing이 **전이를 개선하는지** 검증. 이게 T1(예측)/T2(4D) 확장의 게이트.
- **게이트**: GRU가 GBM annual-summary를 **spatial/site transfer에서 유의 개선할 때만** 확장 + **연별 융해전선 4D-lite 애니메이션** 제작. 미개선 시 negative result(그 자체가 스레드 C 보강).

### 횡단 — AOA + Conformal UQ (모든 스레드 공통)  ★기술 차별성 핵심
- **AOA**(Meyer2021): 학습 피처공간 dissimilarity → 외삽영역 마스킹(지도에 회색).
- **Conformal**(Singh2024/Lou2025): split/spatial-block conformal 구간, nominal 90% vs **observed coverage 검증**. Diffusion 74% 과신 → 보정.
- **kNNDM CV**(Linnenbrink2024): 누수 없는 전이 오차.

---

## 4. 차별성 종합 (심사자 3줄)
1. **규모+다중모달**: 22GB 위성 SAR/InSAR/광학·재분석·지형·전지구 시추공을 융합하고, 각 모달리티 기여를 ablation으로 정직 분해.
2. **정직한 일반화**: 무작위 CV의 과대낙관을 드러내고, 공간블록/LORO/kNNDM + AOA + 보정 coverage로 "어디까지 믿을지"를 지도에 명시.
3. **2D→3D 관측기반**: 얕은 지중 열구조와 0°C 동토경계를 관측기반 보간+전이+UQ로, 인터랙티브·애니메이션으로 제시.

---

## 5. 시각화 시스템 (발표 — 연구 뒤 정비, spec-first)
- **디자인 시스템 신설**: `design/{brand_tokens.json,layout_rules.md,visual_qa_checklist.md}` — 냉색 규약(ALT=oslo_r, 온도=vik 0°C중심, UQ=acton, 차이=broc), 타이포·격자. 벡터(SVG/PDF)+고해상 PNG, 축·컬러바 단위·스케일바 필수.
- **Hero 시각화 5**: ①공간블록+무작위CV 과대평가 ②ablation waterfall+그룹지도 ③AOA마스크+UQ폭+coverage ④인터랙티브 3D 열큐브+등온면 ⑤apparent-floor 진단.
- **산출 스위트**(추후): 시그니처 figure 6–8 · **PyVista 인터랙티브 HTML(라이브 데모)** · 애니메이션 2–3 · (보고서/PPT는 이걸 조립). scientific-figure-reviewer + visual-reviewer QA 후 완료.
- 원칙: 매 실험 **그래프만 X, 공간 시각화 ≥2종**(VISUALIZATION.md).

---

## 6. 실행 순서 · 게이트 · 중단조건
**완료(2026-07-06)**: P0 정정·디자인·진단(C) → P1 metric → 스레드A ablation(정적) → 횡단 AOA/UQ.
**재정렬 순서(2026-07-08~)**:
1. **스레드 R 데이터 재구조화**(㉠시간정합 ㉡집계 ㉢가중) → **R3 게이트**(정적 vs 시간정합 vs 집계 비교).
2. **스레드 A 재실행**(재구조화 데이터 위에서 ablation 다시 — floor 얼마나 내려갔나).
3. **데이터 확장**(SoilGrids/Sentinel/시추공) → ablation M7~M9.
4. **스레드 B 3D**(GBM 조건장+CCI+PyVista) · **스레드 D**(R3 통과시 GRU 시계열).
5. **시각화 통합**(이관 QA + 덱/포스터/인터랙티브).

**metric 표준(모든 CSV)**: `rmse_cm, mae_cm, bias_cm, r2, target_sd_cm, skill_over_mean, n, cv_type, region, scope, feature_group, coverage_90, width_90, aoa_in/out_rmse`.

**중단조건**: 정적 DL이 GBM 미개선→GBM 유지 · InSAR/PolSAR ablation 반복악화→강등 · T-lite GRU가 GBM-annual 미개선→appendix · conformal 후 coverage 불안정→재보정 · neural field(폐기 유지).

---

## 7. 다음 착수 (스레드 R — 데이터 재구조화)
1. `scripts/1_data_prep/assemble_alt_temporal.py`: raw/era5land 월별을 **연도별**로 파생(그 해+전년 겨울 TDD/FDD/여름온도/적설) → 각 (위치,연도) 라벨에 정합 → `dl_dataset_temporal.csv`.
2. `scripts/1_data_prep/aggregate_alt_cell.py`: 위치(또는 셀) 단위 집계 = 평균 ALT + 셀내 SD → `dl_dataset_cell.csv`(pseudo-replication 제거).
3. `scripts/2_evaluation/restructure_gate.py`: 정적 vs 시간정합 vs 집계 GBM 비교(공간블록/LORO/per-year) → `restructure_ablation_results.csv` + floor-이동 그림. **R3 게이트 판정**.
4. (게이트 통과시) 스레드 A/D 재실행 · (미통과시) 집계본 채택 후 데이터 확장.
