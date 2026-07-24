# 실험 계획 (2026-07-21) — 정확도·차별성 재정렬, 면적검증·pooled·물리주입·KPDC

문헌 4개 렌즈(딥러닝 ALT SOTA / InSAR·면적검증 / 물리+ML·전이 / 프로파일→ALT), 지금까지의 확정 결과,
사용자 지적사항을 종합해 세운 세부 실험 계획이다. 초안을 적대적으로 검증해 성공 게이트를 프로젝트 자체
반증 아티팩트와 화해시키고 누설 통제를 실험별로 명문화했다. 예선 보고서 마감 2026-07-31 기준.

관련 문서: `docs/CRITICAL_REVIEW_2026-07-20.md`(비판 진단), `docs/EVAL_FRAMING_NOTE.md`(공간블록≠전이),
`gpt/handoff/20260721_1637-skeptical-reverify-transfer-ceiling.md`(직전 세션), `kpdc/README.md`(신규 데이터).

---

## 1. 이번 세션 목표

전이(OOD) 개선은 종결된 negative로 확정하고, 연구목적을 정확도·차별성으로 재정렬한다.
정직성·불확실성은 헤드라인이 아니라 도구다. 세 축으로 문헌 SOTA 대비 정량 입증한다.

1. **점검증 상한(14cm)의 재프레이밍**: 모델 한계가 아니라 대표성 하한임을 면적검증으로 귀속.
2. **실용 배포 성능**: 전 지역 pooled 학습으로 in-distribution 성능 보고(전이 아님).
3. **KPDC 주 활용**: 콘슬·코어 실측을 독립 검증·물리 forcing·ALT 라벨로 산출물화(대회 규칙).

## 2. 전역 통제 규약 (모든 실험 공통, 예외 없음)

과거 헤드라인 붕괴 원인이 seed 운·누설이었다. 모든 결론에 다음을 강제한다.

- **≥3 seed** 반복, **74블록 공간블록 부트스트랩 CI**로 유의성 판정(CI가 0 포함하면 "개선 없음").
- **거리버퍼**: test 셀 반경 내 train 표본 제거. InSAR/원격 특징은 footprint 이상 버퍼.
- **셀 통째 폴드 배정**: 같은 셀이 train·test에 쪼개지지 않게(site-GKF 블록 공유 재발 방지).
- **누설감사 산출물 의무**: 재발 위험 실험(E2·E3)은 통제 전/후 수치를 나란히 게시(donor 제거·버퍼 스윕 표).
- **집계 라벨에 시점 공변량 금지**: 기후평년 라벨에 연도 특징 넣지 않음(시점 누설 방지).

## 3. 확정 negative (더 돌리지 않음)

- OOD 전이의 모델 구조 개선 전량(순수 ML 40cm·잔차학습·토양 E(x)·미분물리층·SoilGrids 전이 공변량, 레나 skill 음수).
- 심부 3D(5-20m) 지도(스위스 편중 외삽, 알래스카 심부 신뢰 불가).
- 연도 간 ALT anomaly 예측(corr 0.06, 어떤 모델도 예측 불가).
- 모델 교체로 점검증 정확도 향상(6모델 부트스트랩 동률, 정보병목 지배).
- 알래스카 내부 라벨 증강의 정확도 개선 주장(특징복제 누설·seed 착시로 기각. 엄격 통제 재실행 전 사망).

## 4. 실험 목록 (우선순위 순: E1 → E5 → E4 → E3 → E7 → E2 → E6)

### E1. 면적(다중프로브 셀평균) 검증 프로토콜 — 점검증 상한 재프레이밍 [effort M, 최우선]

- **동기**: 점검증 14cm 바닥이 모델 한계가 아니라 대표성 하한임을 문헌(Parsekian 2021, Du 2025, Uxa 2026)이
  확증한다(방법 무관 14-18cm 밴드). 헤드라인 지표를 단일프로브에서 셀평균으로 바꿔 상한을 규명한다.
- **⚠️ 게이트 재조정(비판 반영)**: 프로젝트 자체 반증과 화해시킨다. `grid_support_results.csv`는 지지 확대 시
  RMSE 상승(점 17.04 → 1km 17.49 → 25km 23.05), `areal_eval_results.csv`도 1km 18.82cm다. 집계 라벨을 집계
  공변량으로 예측하면 셀간 계통오차가 지배해 "RMSE 감소"는 기대난망. 따라서:
  - **1순위 산출물(주게이트)**: Parsekian식 오차분해(측정오차 + 스케일링오차 + 대표성오차=셀내SD). 셀내SD가
    잔여RMSE의 ≥60%를 설명하면 "바닥은 대표성"을 정량 귀속(성공). 셀내SD는 이미 9.7-13.3cm로 관측됨.
  - **2순위(탐색)**: 원시 다중프로브(`alt_above_pointlevel.csv` 223,937점)에서 SEM<3cm인 셀만 재라벨해
    학습·평가. RMSE 감소는 부수 탐색으로 강등(감소 시 보너스, 미감소 시 상한 귀속으로 보고).
- **방법**: 점→30m→1km→9km 지지별 집계, 74블록 공간블록 CV, GBM(HistGBR)·MLP. grid_support 상승곡선을
  명시적으로 인용해 null을 정직 보고. 셀은 폴드 통째 배정 + 거리버퍼, 기후평년 라벨에 연도 공변량 제외.
- **데이터**: `alt_above_pointlevel.csv`, `dl_dataset_cell_v3_soil.csv`, `model_tournament_predictions.csv`.
- **성공 게이트**: 셀내SD가 잔여RMSE의 ≥60% 설명(대표성 귀속) **또는** SEM<3cm 셀평균 RMSE가 단일프로브 대비
  블록부트스트랩 CI로 유의 감소. 둘 중 하나면 성공. **스크립트**: `areal_evaluation.py`를 원시프로브 라벨로 확장.
- **위험·통제**: 감소가 계통오차로 상쇄돼 null 재현 가능 → 대안 게이트(대표성 귀속)로 산출물화. 누설: 셀 통째 배정.
- **문헌**: Parsekian 2021(오차=측정+스케일링+대표성, 전체 RMSE 0.176m), Du 2025(30m→1km +5-10%), Uxa 2026.

### E5. KPDC 콘슬 지중온도 → 얕은3D 검증 + AWS forcing 대조 [effort M, 대회 필수]

- **동기**: 대회 KPDC 주 활용 의무. 콘슬 활동층 지중온도로 0°C 등온선 ALT를 유도해 얕은3D 열구조 모델
  (field R² 0.47, 0°C→ALT r 0.28)을 학습셋 밖 독립 지점으로 검증. AWS 실측 √TDD로 ERA5 forcing 현장 검증.
- **⚠️ 정정(비판 반영)**: "센서 깊이 메타 부재" blocker는 과장이다. **ID01-24 일자료는 깊이 라벨(10-160cm)이
  헤더에 있다**(ID01-14 0.8m 8층, ID21-24 1.6m 16층). 깊이 미표기는 5분 프로파일(2022/2023, L0-L8)에 한정.
  깊이 있는 ID01-24를 1차로 쓰고, 5분 프로파일은 상대깊이 보조로.
- **⚠️ 게이트 재조정**: 원안 |bias|<15cm는 얕은3D 자체 정합오차(0°C→ALT r 0.28, RMSE 41cm, Happy Valley +54cm)와
  모순돼 5점에서 통과하면 체리피킹. **사례연구 프레임으로 전환**해 편차를 정직 보고(일반화 주장 금지).
- **방법**: 중복 제거(VWC 파일=온도 파일 md5 동일). 시각 문자열 파싱(`[2021-10-02]`), `_x001A_` EOF 행 제거.
  ID10-14 채널-깊이 인터리브 교정, ID21-24 라벨 반전 교정 후 층별 0°C 등온선 선형보간 → 지점별 thaw depth
  시계열 → 연최대 ALT. AK_core_sample 코어길이(72-88cm)를 ALT 약라벨로 병용. AWS √TDD vs ERA5 √TDD Stefan
  RMSE 비교. **스크립트**: `kpdc_era5_validation.py` 확장 + 신규 `parse_kpdc_soil_profile.py`.
- **데이터**: `kpdc/council/soil_temp/daily_profile_ID/`, `.../active_layer_profile_5min/`, `.../core_alt/`, `.../aws_met/`.
- **성공 게이트**: KPDC 독립 검증점 ≥5개 산출 + 유도 ALT vs 예측 편차 정직 보고. AWS forcing Stefan RMSE ≤ ERA5.
- **위험·통제**: 좌표 메타는 KPDC 데이터셋 페이지 요청. 미확보 시 콘슬 기지 좌표(~64.85N -163.70W) 잠정 + 단일지역 사례.
- **문헌**: Uxa 2026 ASM(2깊이 온도→ALT, RMSE 14-18cm), Nakata 2026(표면임베딩→지중온도), Zhao 2025(시추공 얕은3D).

### E4. 물리 특징 주입(Stefan-as-feature) + 경량 E 미세조정 [effort M]

- **동기**: 잔차학습·미분물리층·파라미터예측은 전이서 전부 무익(확정). 대안은 Stefan 출력을 **잔차가 아니라
  입력 특징**으로 주입(Stefan-CatBoost 2025 QTP R² 0.873). 사용자 방향4(알래스카 pretrain→레나 소량 라벨로 E만
  fine-tune)를 PI-LSTM 레시피(Liu 2023: 물리 pretrain→실측 fine-tune, 순수 대비 +27%)로 재현.
- **방법**: 셀별 Stefan ALT=a+E√TDD 예측치를 GBM/MLP 입력 특징으로 추가. 공간블록 CV로 BASE vs +Stefan-feature
  vs 잔차(기존 실패) 비교. **E는 폴드 train-only 적합(폴드별 재적합, test 포함 적합 금지=라벨 누설 방지)**.
  특징중요도·부분의존성으로 √TDD 다중공선성 점검. Fine-tune: 알래스카 site-out E 적합 → 레나 3,037셀 일부
  라벨로 E만 재적합(a 고정), Lena 공간블록 held-out에서 frozen-E Stefan(18.24cm)과 비교.
- **데이터**: `dl_dataset_cell_v3_soil.csv`(e5_sqrt_tdd·e5_tdd), `p2_stefan_experiment.py`, `train_pretrain_finetune.py`.
- **성공 게이트**: +Stefan-feature가 in-domain RMSE ≥0.5cm 감소(CI 0 제외) **또는** Lena E 미세조정이 frozen-E
  18.24cm를 ≥2cm 개선. 미달 시 negative로 확정 기록(사전확약).
- **위험·통제**: 다중공선성 무이득 가능 → 부분의존성 점검. 소표본 과적합 → Lena 셀 공간블록 분리.
- **문헌**: Stefan-CatBoost 2025(특징주입 R² 0.873), PI-LSTM Liu 2023(+27~69%), Pilyugina 2023(열방정식 정규화 20% 이득, in-domain 한정).

### E3. 전 지역 pooled 배포 모델 — 유사도 층화 + 정적 임베딩 [effort M]

- **동기**: 사용자 방향2. 전이 포기하고 전 지역 train/val/test 혼합 pooled 학습으로 실용 배포 성능 보고.
  Ohmer&Liesch 2026: blind pooling은 단일학습과 비슷(NSE 0.47 vs 0.49)하나 유사도 층화 시 global 0.72 > local 0.67.
  Kratzert 2019: entity-aware 정적 임베딩이 다지역 통합 핵심.
- **⚠️ 핵심 누설통제(비판 반영)**: "혼합 test 배포 성능"을 무작위 분할로 만들면 공간 자기상관으로 인접 셀이
  train에 들어가 site-GKF 착시 재현. **혼합 test는 반드시 공간블록으로 구성**하고, 무작위 분할 수치와 블록
  수치를 나란히 보고해 격차를 노출(배포 성능=블록 수치). region-balanced 폴드.
- **방법**: ABoVE_AK(13,542) + ABoVE_CA(726) + Lena(3,037) + KPDC 콘슬 셀 pool. Köppen 기후·지형(tpi/slope)
  클러스터 층화. 정적 공변량(DEM6+SoilGrids+ERA5) entity-aware 임베딩(MLP, 소표본 대비 차원 규제). 비교:
  (a) blind pooled (b) 층화 pooled (c) 지역별 단일. in-distribution 혼합 test(공간블록) RMSE = 배포 성능.
  동시에 Lena leave-out OOD가 pooled에서도 음의 skill 재현됨을 명시(전이 종결 일관성).
- **데이터**: `dl_dataset_cell_v3_soil.csv`(region 컬럼).
- **성공 게이트**: 층화 pooled in-domain(공간블록) 혼합 test RMSE ≤ 14cm(바닥 유지) + 지역별 단일 대비 열화
  ≤0.5cm. blind < 층화를 3-seed 블록부트스트랩으로 확인. 혼합 test를 OOD로 오표기 금지.
- **위험·통제**: 결측 라우팅 아티팩트(과거 통합 기각 전례) → 결측 플래그 명시. region 불균형 → balanced 폴드.
- **문헌**: Ohmer&Liesch 2026(유사도 층화), Kratzert 2019(entity-aware 임베딩), Pan-Arctic ViT 2025.

### E7. SOTA 정렬 벤치마크 + 포지셔닝 표 [effort S, 조기 착수]

- **동기**: 예선 보고서 헤드라인. 우리 수치를 문헌 SOTA와 표준 프로토콜(점+면적)로 나란히 배치해 동급임을 근거 제시.
- **방법**: **확정 수치(in-domain 14cm, Stefan LORO 18cm, field R² 0.47)로 골격 표를 즉시 착수**(마감 리스크 감소).
  E1-E4 결과를 채워넣음. Gautam 2025(RF 22/Stefan 18cm)·Uxa 2026(14-18cm)·Merchant 2024(R² 0.476)·Whitcomb
  2023(11-12cm)·ESA CCI 2025(글로벌 56cm)와 동일 표. 각 항목에 검증 프로토콜(점/면적, in/out, 표본, CI) 라벨.
  GeoCryoAI 1.3cm는 teacher-forced/in-sample임을 각주로 명시(과대낙관 비교 차단).
- **성공 게이트**: 우리 in-domain·면적 수치가 문헌 SOTA 밴드(11-18cm) 내에 CI 겹침으로 위치. 프로토콜 메타 전부 표기.
- **위험·통제**: 프로토콜 불일치 사과-오렌지 위험 → 각 셀에 (지지·in/out·표본·CI) 메타 강제, 우위 과장 금지.
- **문헌**: Gautam 2025, Uxa 2026, Merchant 2024, Whitcomb 2023, ESA CCI PVIR 2025, GeoCryoAI 2025(in-sample 각주).

### E2. InSAR/PolSAR 면적장 업스케일 회귀 — 스칼라 증류 폐기 [effort L, 마감 임계경로 밖]

- **동기**: 사용자 방향3. InSAR 30m를 스칼라 weak label로 증류하지 말고 면적장으로 활용. Merchant 2024(RF
  업스케일 field R² 0.476)·Whitcomb 2023(CALM RMSE 11.8-12.1cm)을 우리 자산(InSAR 6.9GB·PolSAR 7GB)으로 재현.
- **⚠️ 위험 최상위(비판 반영)**: `insar_ablation_results.csv`에서 +InSAR GBM 18.79 > BASE 17.24(악화), PolSAR
  raw 38.29cm로 국소 RMSE 무익이 이미 저장됨. 타깃 셀에 최근접 ReSALT 픽셀을 붙이면 test 인접 특징복제 누설
  재발(`field3d_reeval_leakage.csv`가 nn 0.05-0.2km 이웃·block_share 8-19 기록). R²는 공간분산으로 부풀 수 있음.
- **방법(통제 강화)**: 공간블록 CV 버퍼를 **InSAR footprint 이상**으로 잡아 test 최근접 픽셀이 train에 없게 강제.
  **field R²와 국소 면적 RMSE를 반드시 분리 보고**(R² 단독 헤드라인 금지). PolSAR raw 38cm를 사전등록 baseline,
  개선분을 블록부트스트랩 CI로 정량. 심도 라우팅: ≤65cm PolSAR 신뢰, 65-150cm InSAR 침하+Stefan 역산(P-band 포화).
  특징=ReSALT 침하진폭장 + PolSAR + ERA5 8 + DEM 6, 타깃=E1 셀평균, SHAP 기여. in-domain 한정 주장.
- **데이터**: `resalt_weaklabels.parquet`, `dl_dataset_cell_v3_soil.csv`(polsar_*·insar_*), `data/raw/polsar_alt/`.
- **성공 게이트**: 버퍼 통제 후 스칼라 baseline 대비 면적 RMSE 개선이 블록부트스트랩 CI로 유의 **또는** field R²
  ≥0.47을 국소 RMSE와 함께 정직 보고. 미달 시 "면적 프레임에서도 InSAR 무익"을 negative 자산으로.
- **위험·통제**: 마감 못 맞추면 방법제안+예비결과로 강등(우선순위 최하위권 유지).
- **문헌**: Merchant 2024(R² 0.476), Whitcomb 2023(11-12cm·65cm 포화), Chen 2023 PDO-2(침하+수분 공동역산), Schaefer 2015 ReSALT.

### E6. AlphaEarth Foundations 임베딩 A/B [effort L, deferred — GEE blocker]

- **동기**: AEF 64차원·10m 임베딩(Brown 2025, 이전 특징화 대비 오차 -23.9%)·Nakata 2026(임베딩→지중온도 성공)이
  라벨 희소 SOTA. SoilGrids covariate shift 실패(-63.8%)를 임베딩이 완화하는지 A/B로 검증(신규성 차별점).
- **방법**: Earth Engine에서 AEF 임베딩을 ABoVE·Lena·콘슬 셀 좌표에 추출. pooled BASE vs +AEF-64를 in-domain
  면적검증(E1)과 pseudo-OOD로 A/B. SoilGrids 전이 실패와 나란히 비교. 임베딩 연도-라벨 시점 분리 검사(누설).
- **성공 게이트**: +AEF가 in-domain 면적 RMSE ≥1cm 감소 + pseudo-OOD 열화 <20%(SoilGrids -63.8% 대비 완화).
  GEE 접근 불가 시 deferred로 기록, 마감 산출물 제외.
- **위험·통제**: GEE 계정·접근 blocker + 10일 마감 촉박 → 우선순위 최하위 deferred. 미접근이면 방법제안으로 한정.
- **문헌**: AlphaEarth Foundations 2025(arXiv:2507.22291), Nakata 2026(arXiv:2604.14756), Prithvi-EO 2024.

## 5. KPDC 주 활용 서사 (대회 규칙 충족, 별도 산출물)

E5 단일 검증만으로는 "KPDC 주된 분석 대상" 규칙 해석상 미흡할 수 있다. 다음을 명문화한다.

- **보고서 "활용 데이터" 절 첫 배치**: 콘슬 8층 지중온도·코어 ALT·AWS 기상을 주 데이터로 서술.
- **독립 검증 정량 기여**: KPDC 유도 ALT를 우리 모델 검증점으로(학습셋 밖).
- **물리 E 실측**: AWS √TDD로 Stefan forcing 현장 검증, 코어길이로 ALT 라벨.
- **사무국 해석 확정**: KPDC 활용 범위 해석을 대회 사무국에 문의(계획서 대기 항목).

## 6. 대회 예선 보고서 산출물 (deliverables)

1. 지지크기별 RMSE·셀내SD·SEM 곡선 + Parsekian 오차분해표(측정·스케일링·대표성) [E1].
2. InSAR/PolSAR 30m 면적 ALT 회귀 지도(물리 aspect ratio·컬러바·스케일바) + SHAP 상위예측자 [E2].
3. pooled 배포 성능표(blind vs 층화 vs 지역별, 무작위 vs 공간블록 병기) + OOD 재확인 각주 [E3].
4. 물리특징 주입 ablation 표(BASE/+Stefan/잔차) + 레나 E 미세조정 전후 곡선 [E4].
5. KPDC 콘슬 등온선 얕은3D 단면도 + 유도 ALT vs 예측 대조 + AWS/ERA5 forcing 비교표 [E5].
6. SOTA 벤치마크 포지셔닝표(우리 vs Gautam/Uxa/Merchant/Whitcomb/CCI, 프로토콜·CI 라벨) [E7].
7. 4D timelapse ALT 애니메이션(기존 완료본) + 시간신호 예측불가 정량 각주.
8. 재현 패키징 + AI 사용내역 + 출처표(계획서 요구).

## 7. 자원·일정

- **GPU**: 6,7,8,9(torch만 GPU, GBM/sklearn은 CPU 병렬). 사용 전 nvidia-smi 필수.
- **임계경로(10일)**: E1·E5·E4·E7을 우선 완결(전부 기존 데이터·스크립트 재사용, effort S/M). E3는 결측통제 후.
  E2(effort L, 14GB)·E6(GEE blocker)은 임계경로 밖 → 시간 남으면 착수, 아니면 방법제안으로 보고.
- **문헌 근거표**: 각 실험 lit_basis의 논문은 웹 검증 완료(일부 arXiv preprint는 unverified 표기 유지).
