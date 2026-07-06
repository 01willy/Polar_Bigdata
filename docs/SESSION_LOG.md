# Session Log — Polar_Bigdata

## 2026-06-24 ~ 06-30 — 프로젝트 착수 ~ ALT 예측 Stage 0/1

영구동토 well 온도 + 활성층 데이터로 3D 모델을 만드는 프로젝트를 처음부터 구축한 세션.

### 확정된 연구 목표 (사용자 동의)
전 지구 다지역 borehole 온도 + CALM 활성층 + 공변량 → **공간인식 딥러닝으로 ALT(주 출력)**
+ 얕은 3D 지중 열구조(0~20m) + 셀별 불확실성 예측, **알래스카 학습 → 타 영구동토 지대 전이 검증**.
차별성 = 대규모 다지역 + 공간 3D + 불확실성 + 전이 (vs Ran 2022 점단위 2D RF). 상세: `docs/PLAN.md §0`, `docs/PLAN_ALT_pipeline.md`.

### 데이터 확보 (전부 무계정 오픈)
- **GTN-P API**: list+bulk 엔드포인트 효율 다운로드(브루트포스 회피). 전 지구 PT 450 데이터셋/397 borehole, ALT 106. `combined=false` 서버버그(대용량 500) → `combined=true` 폴백 해결.
- **PANGAEA CALM**(Streletskiy 2025): 전 지구 ALT 259 사이트/12국/3,604 site-year → `data/processed/alt_global.csv` (+WorldClim 공변량).
- **WorldClim 2.1**(기온·고도·계절성·강수, tifffile, rasterio 불필요). 월별 tavg/prec는 서버 불통 → bio 사인곡선 근사로 도일 계산.
- **공변량 지점샘플**: USGS 고도, SoilGrids 토양. 참고문헌 PDF 6편 → `references/`.
- 미취득(계정 필요, 추후): InSAR(Sentinel-1/ReSALT), MODIS(GEE), ERA5(CDS), ESA CCI(CEDA).

### 모델링 진행
- **Stage 0** (`src/polar/alt_model.py:run`): 평가 골격 4종 CV(무작위/site-disjoint/공간블록500km/LORO) + baseline 사다리(지역평균/IDW/Kriging/GBM). **누설 입증**: IDW RMSE 무작위 28 → LORO 111cm(무작위 CV 4배 과대평가). → 헤드라인 지표는 항상 공간블록+LORO.
- **Stage 1** (`run_stage1`): 도일 피처(TDD/FDD/Stefan √TDD/대륙성/적설 proxy) 추가. 공간블록 GBM 83.9→77.6, RF 70.2; LORO GBM 103.9→**96.9**, RF 108.6. **GBM이 전이에 더 안정**(RF 과적합). 변수중요도 wc_elev=0.67 지배.

### 핵심 발견 / 결정
- 우리 데이터는 **영구동토를 직접 측정**(심부 ≥5m 99% 동결, 최심 90m). 활성층만 아는 게 아님.
- 깊은 base(200~600m)는 데이터 부족 → 얕은 0~20m 집중, base는 물리 외삽+큰 불확실성.
- 기존 연구 중복: GIPL2(알래스카 base)·CCI(다중깊이)·InterPIGNN(PDE 신경장)·Ran2022(데이터융합). 차별=관측기반 보간+전이+UQ+메시. "전면 능가" 목표 아님.
- PINN은 선택사항 — 순수 DL(원안)로 시작 가능.

### 현재 갈림길 (다음 세션 결정 필요)
무계정 거친 공변량의 정확도 한계 도달(LORO ~97cm; 전 지구 ALT 34~427cm로 광범위). 다음 지렛대:
- **A. 더 좋은 데이터(계정)** ★추천 — InSAR/ERA5월별/MODIS → 큰 향상
- B. 공간 DL(Stage 3) — 소폭 개선
- C. 지역 집중 — 오차↓ 그러나 전 지구 포기

### 인프라 메모
- git 미초기화, `.claude/project.yaml` 없음(상태스크립트/자동 cleanup 미적용).
- **디스크: /home 98% 사용(공유 13T, 272G 여유) — 일시적 ENOSPC spike 발생. 대용량 다운로드 시 주의.**
- GPU: 학습 시 6,7,8,9번(타 사용자 점유 시 보류). 현 단계는 CPU.
- 환경: py3.9, sklearn/pykrige/gstools/skgstat/torch+CUDA/pyvista. xgboost 미설치(HistGBM fallback).
