# Polar_Bigdata — 3D 영구동토층 Geology Model 초기 연구 계획

> 목표: 영구동토층(permafrost)의 **well별 온도 데이터(depth × time)** 와 **active layer 자료**를
> 수집·전처리하고, 이를 보간(interpolation)하여 **3차원 영구동토 geology model**
> (0 °C 등온면 = 영구동토 base를 tri-mesh로 추출한 체적 모델)을 생성한다.

본 문서의 데이터/엔드포인트 수치는 2026-06-24 GTN-P API 실측 + 다중 에이전트 조사(검증 포함)로 확정한 값이다.

---

## 0. 확정 연구 목표 (2026-06-29)

> **전 지구 다지역 borehole 온도 + CALM 활성층 + InSAR + 공변량을, 공간인식 딥러닝(+불확실성)으로
> 융합해 — 활성층 두께(ALT)와 얕은 3D 지중 열구조를 예측하고, 알래스카에서 학습해
> 타 영구동토 지대로의 전이(transfer)를 검증하는 모델.**

- **타깃 깊이**: (A) 잘 뒷받침되는 **얕은 구간 0~약 20 m** 집중. 깊은 base(200~600 m)는 "물리 외삽 + 큰 불확실성"으로 표시(데이터 부족).
- **주 출력**: ① **ALT(x,y) 활성층 두께 지도(1순위)** ② 얕은 3D 온도장 T(x,y,z≤~20 m) ③ 영구동토 table 표면(tri-mesh) ④ **셀별 불확실성**. (멀티태스크: ALT와 온도장 물리 일관)
- **데이터 규모**: ALT 정답은 희소(전 지구 ~수백 사이트)이나 **InSAR 침하(조밀 ALT 대리값) + 공변량 격자 + 전 지구 CALM/GTN-P 풀링**으로 데이터-풍부화.
- **모델**: GBM(XGBoost/LightGBM) baseline → **공간인식 DL(3D CNN/Transformer/neural field)** → (선택) 물리(PINN)·생성(diffusion, 합성코퍼스 기반).
- **차별성(vs Ran 2022 점단위 2D RF)**: 대규모 다지역 + **공간 3D 구조** + 불확실성 + **지역간 전이 검증**.
- **벤치마크**: ESA CCI(범극지)·GIPL2(알래스카) + 튜닝 크리깅/GBM. *목표는 "전면 능가"가 아니라 동등 정확도 + 불확실성·3D·전이·메시 추가*(의의는 관측기반·독립방법·전이지식·벤치마크에 있음).
- **참고문헌**: `references/`(Ran2022·GIPL2·Gautam2025·InterPIGNN·Groenke·GTN-P).

---

## 1. 데이터 수집 (검증 완료)

### 1.1 핵심 관측 — GTN-P (Global Terrestrial Network for Permafrost)
- **PT(Permafrost Temperature) 데이터셋: 693개** (Ground Temperature 595 / Air 56 / Surface 42), **518개 borehole**, 약 2,416만 행.
- **ALT(Active Layer Thickness) 데이터셋: 106개** (103개 site), grid/transect 방식, 단위 cm.
- 라이선스: 포털 정책상 **CC BY 4.0** (단, 일부 site는 출판 전 PI 연락 요청/엠바고 가능, 9개 Restricted, ~140개 정책 미표기).
- PT CSV 컬럼: `id, date, depth, temperature, flag, dataset_id, borehole_id, site_id`
  → **borehole별 depth × time 온도 행렬** (예: 한 borehole에 깊이 2.0~73.0 m, 86개 depth). 정확히 3D 모델에 필요한 형태.
- ALT CSV 컬럼: `id, date, offset_x, offset_y, alt, flag, dataset_id, activelayer_id, site_id` (site별 2D 공간 격자 샘플).
- 각 zip에 `*_metadata.json` 포함 → borehole별 **lat/lon/elevation**, site_name, country, gtnp_code, drilling 정보.

**다운로드 방식 (브루트포스 금지).** `download_gtnp.py`로 자동화:
1. `GET /api/list-pt-datasets`, `GET /api/list-alt-datasets` (2회 호출로 전체 ID 열거)
2. `GET /api/list-sites?boreholes=true&metadata=true` (borehole 좌표)
3. `GET /api/data?pt_data=<ID,...>&combined=false` 를 **배치당 ~20개**로 분할 (PT ~35배치 + ALT ~6배치 = ~41 요청).
   - ⚠️ 100개 ID 요청도 HTTP 500, 693개 한 번에 = 4초 만에 실패. 전체-DB 단일 아카이브 없음.
   - ⚠️ `data_access` 필드는 내부 호스트(`http://fastapi:8000`)를 가리킴 → 공개 호스트로 재작성 필요(스크립트가 처리).

### 1.2 보완 borehole / ALT 네트워크 (GTN-P보다 최신/심부 기록 다수)
| 소스 | 내용 | 접근 |
|---|---|---|
| **UAF Permafrost Lab** (permafrost.gi.alaska.edu) | Alaska 심부(60–80 m) borehole, site별 Excel(센서 깊이 명시) | 웹 다운로드(검증) |
| **USGS/Clow Arctic-Slope deep array** (NSIDC G10015) | depth-resolved T(z,t), ASCII, 공개(검증) | NSIDC |
| **Nordicana D** (CEN, Laval) | Canada, DOI별 ZIP | doi |
| **PERMOS** (스위스) | 산악 영구동토, DOI 버전 ZIP/ASCII | 포털 |
| **NORPERM** (NGU) | Norway 본토 + Svalbard | 지도 GUI |
| **PANGAEA GTN-P 2018 MAGT** (DOI 10.1594/PANGAEA.884711) | 2,379점 MAGT(연도별, 깊이 6.7–40 m), **단일 파일** | tab-delimited |
| **PANGAEA CALM ALT** (Streletskiy 2025, DOI 10.1594/PANGAEA.972777) | 263개 NH ALT 시계열 1990–2024, **단일 파일**, CC-BY (elevation 컬럼 없음, lat/lon만) | tab-delimited |

### 1.3 격자형 영구동토 제품 (ground truth / 검증 / 사전정보)
| 제품 | 변수 / 깊이 | 해상도·기간 | 접근·형식 |
|---|---|---|---|
| **ESA CCI Permafrost v5** | GT (surface/1/2/5/10 m), ALT, extent — **깊이축 backbone(표층~10 m)** | 1 km(0.01°), 1997–2023 | CEDA, NetCDF, open(인용 조건) |
| **Obu et al. 2019** | MAGT @ top of permafrost (단일 깊이) | 1 km, 2000–2016 | PANGAEA, GeoTIFF, CC-BY-3.0 |
| **Ran et al. 2022 (NIEER)** | MAGT @ DZAA(~3–25 m), ALT — **심부 anchor** | 1 km | ESSD/TPDC |
| **Brown et al. 2002 (NSIDC GGD318)** | extent 등급 + **ground-ice content**(잠열·열물성) | 범극지 | NSIDC |

> ⚠️ **CCI 주의**: 5개 깊이가 `depth` 차원이 아니라 **별도 변수**(`GST, T1m, T2m, T5m, T10m`)로 저장됨. depth 차원으로 슬라이싱하면 코드가 깨짐 — 5개 변수를 따로 읽을 것. GT의 정확한 DOI는 `10.5285/5675b0be944f45a8af0e7ddbeb47a011`.

### 1.4 공변량(covariates / 예측변수) — regression-kriging / ML 용
영향력 순: **① ERA5-Land 기온(MAAT, 동결/융해 도일) → ② MODIS LST → ③ 적설(SWE) → ④ 지형 → ⑤ 토양/식생/토지피복**
| 데이터 | 해상도 | 접근 |
|---|---|---|
| ERA5-Land 2 m 기온 + 토양온도 4층 + SWE | ~9 km | Copernicus CDS API |
| MODIS LST (MOD11A2/MYD11A2) | 1 km, 8일 | Google Earth Engine |
| ArcticDEM v4.1 (10/32 m) / Copernicus GLO-30 | 10–32 m | PGC, GEE |
| ESA WorldCover | 10 m | GEE/직접 |
| SoilGrids 2.0 (sand/clay/SOC/bulk density, depth별) | 250 m | ISRIC WCS |
| MODIS NDVI (MOD13Q1) | 250 m | GEE |

> 공통 좌표계 권장: **EPSG:3413 (NSIDC Polar Stereographic North)**, 모델 격자 250 m–1 km.

---

## 2. 모델링 방법론 (5단계 파이프라인)

> 현재 환경: pyvista 0.46.5, scipy, numpy, pandas, scikit-image, scikit-learn, torch+CUDA 보유. **geopandas 없음**.
> 추가 설치(경량): `pip install pykrige gstools scikit-gstat meshio`

- **Stage 0 — 데이터 준비 + 물리 사전정보.** 모든 관측을 `(x, y, z, T)` long table로(측정 깊이 1개 = 1행 → 수직 해상도 확보). 기온 도일 + n-factor + 전도도비로 TTOP/MAGT, 지온경사(~25–31 °C/km) 계산 → 가상관측/수직 이방성 설정. ALT·고도·적설·식생 = 공변량 컬럼.
- **Stage 1 — 변동도(variogram) 분석** (scikit-gstat). z를 재척도하여 수직:수평 이방성 인코딩, 모델·range 추정.
- **Stage 2 — 3D 보간 → temperature volume.**
  - (a) baseline: `scipy RBFInterpolator` (무설치)
  - (b) 주력 지구통계: **GSTools**(이방성 CovModel + kriging + PyVista export) 또는 **PyKrige** `OrdinaryKriging3D`
  - (c) 공변량 활용: PyKrige `RegressionKriging` / GSTools External-Drift, 비선형은 sklearn RF·GBM으로 trend 후 잔차 kriging
  - → **kriging variance를 불확실성 volume으로 함께 산출.**
- **Stage 3 (선택/고급)** — implicit base surface(GemPy v3 / LoopStructural) 또는 SIREN neural field(torch+GPU, 점온도+표면/ALT 경계+열전도 prior 융합). 단 Stage 2 kriging으로 먼저 검증(희소 데이터 과적합 위험).
- **Stage 4 — 등온면 + mesh** (PyVista + scikit-image). `grid.contour([0.0], method='flying_edges')` → **0 °C 등온면 = 영구동토 base를 tri-mesh로** 추출, `.threshold(0.0)`로 frozen body 분리. export: `.vtp/.stl/.ply` (mesh), `.vti/.vtr` (volume), meshio로 OBJ/VTU/MSH.

---

## 3. 문헌 근거 (재사용 가능 component)
- **Obu et al. 2019** — TTOP 모델(범극지 매핑 표준). 방정식 재사용.
- **Ran et al. 2022** — predictor set(MODIS FDD/TDD, snow duration, LAI, 강수, 복사, 토양) + ensemble 불확실성 설계.
- **Westermann et al. 2023 CryoGrid / permamodel GIPL(MIT, PyMT) / CryoGrid.jl(EUPL)** — 1D 열전도+상변화 컬럼 물리(수직 외삽·base 위치).
- **Cao et al. 2026 ASM** — active layer 2개 깊이만으로 permafrost-table 온도 폐형해(저렴한 수직 외삽).
- **Groenke et al. 2024** — borehole 기반 과거 기후 재구성.

> **핵심 전략**: 새 3D solver를 발명하지 말고 위 component(TTOP, Ran predictor, GIPL/CryoGrid 컬럼 물리, Cao ASM)를 재사용. **진짜 신규 기여 = 열장(thermal field)의 진정한 volumetric 3D 보간 + permafrost base 추출.**

---

## 4. 실행 로드맵
- **Phase 0 (이번 주)** — 환경 세팅 + GTN-P 다운로드: `python download_gtnp.py --gt-only --open-only`
- **Phase 1** — 전처리/QC: borehole 좌표 join, long table 구축, flag/결측 처리, EDA(깊이·시간·공간 분포).
- **Phase 2** — 단일 borehole 수직 프로파일 + 0 °C 등온면 깊이 추출(검증 기준선).
- **Phase 3** — 공변량 stack 구축(GEE/CDS) + 공통 격자 리샘플.
- **Phase 4** — 기준선: 3D regression-kriging volume(RF/GBM trend + 잔차 kriging) + kriging variance(불확실성).
- **Phase 5** — 주력: PINN(SIREN) + 정상상태 열전도로 3D 열장 학습, 심부 base 물리 외삽.
- **Phase 6** — 0 °C 등온면 tri-mesh 추출 + 검증(CCI/Obu/Ran 대비, leave-one-borehole-out CV) + 불확실성 정량화.

---

## 5. 확정된 연구 범위 (2026-06-24)
- **공간 범위: Alaska 파일럿.** 지구상 가장 심부까지 계측된 영구동토 지역 중 하나(아래 깊이 통계 참조). 검증 후 범극지 확장.
- **시간 차원: 정적 평형 모델(static equilibrium).** 시계열은 평형 통계로 축약(§7), 원시 시계열은 보존해 향후 transient 확장.
- **깊이 범위: 관측 전체 깊이(표층~약 90 m) + base는 물리 외삽.** "표층~10 m 캡" 대신 심부 borehole을 그대로 쓰고, 영구동토 base(0 °C 등온면)는 지온경사/열전도 물리로 하향 외삽. CCI 0–10 m는 상부 검증·공변량.

### 5.1 Alaska 심부 데이터 현황 (GTN-P 실측, Ground Temperature)
- Alaska: 63 borehole / 76 데이터셋. 깊이 분포가 이중구조 — `<2 m` 32개(천부/CALM성) vs **`≥20 m` 26개, `≥50 m` 15개**.
- 최심부: Toolik Lake 90 m, Kaktovik 74 m, Galbraith Lake 73 m, Fairbanks 68.7 m, Coldfoot/Old Man/Five Mile 60–62 m, Deadhorse/West Dock/Franklin Bluff(=USGS Arctic-Slope array) 54–58 m.
- ⚠️ 북부(연속대) 영구동토 두께 200–600 m → 60–90 m borehole로도 base 미도달 → **외삽 필수**. 내륙 불연속대(Fairbanks 등)는 수십 m라 base 관측 가능.
- 추가 심부 소스: **UAF Permafrost Lab**(원본 Excel), **USGS/Clow Arctic-Slope NSIDC G10015**(ASCII 공개).

## 6. 딥러닝 모델링 방향 (3D 열장 예측) — 데이터 충분성·생성모델 검증 반영

### 6.0 데이터 충분성 (검증됨)
- borehole **단독 불가**: Webster-Oliver 규칙(변동도 <50 무의미, ~100 최소, 150–225 신뢰). Alaska 63 borehole/~2,316 깊이점은 이 아래이고 연속대 base는 미관측.
- **격자+공변량+물리와 결합하면 가능**: 0–25 m는 조밀 격자(CCI 5깊이 ~860만 셀)·공변량이 주도하는 데이터 풍부 회귀, 심부 base는 물리 외삽.
- ⚠️ **필수 baseline 경고**: 희소 데이터에서 단순 IDW가 kriging·INR을 모두 이긴 사례(arXiv:2512.11832). 신경장은 IDW·kriging·RF를 spatial CV로 **실제로 이긴 뒤에만** 채택.
- **추가 데이터 우선순위**: ① 불연속대 base-도달 심부 borehole(UAF/USGS G10015), ② 열전도도+ground-ice(Brown 2002, SoilGrids)+지열플럭스 맵, ③ Ran 2022 DZAA. (얕은 CALM 추가는 심부에 도움 적음.)

### 6.1 권장 아키텍처: 다중 충실도(multi-fidelity) Physics-Informed Neural Field
| 단계 | 모델 | 역할 |
|---|---|---|
| 기준선(필수) | **3D Regression-Kriging** (RF/GBM trend + 잔차 kriging) + IDW | 벤치마크. DL이 이를 못 이기면 채택 안 함 |
| **주력** | **multi-fidelity PINN/SIREN**: ① 조밀 격자(CCI/Obu/Ran)+합성 GIPL/CryoGrid 컬럼으로 사전학습 → ② 26개 심부 borehole로 미세조정 → ③ 10⁴–10⁶ collocation점에서 ∇·(k∇T)=0 (상부 BC=MAGT/TTOP/CCI, 하부=지열플럭스) | 매끈·물리정합 3D 열장 + 심부 base 외삽 |
| 확장 | 공변량 조건부 신경장 `f_θ(x,y,z,covariates)` | 공간 일반화 |

**선택 근거**: 좌표 기반→메시 없이 임의 점 질의(marching cubes); PDE가 미관측 심부 base로 물리 외삽; collocation점이 희소 데이터를 정칙화. (torch+CUDA 보유.)

### 6.2 생성모델(diffusion / flow matching) — v1 불필요, v2 보류 (검증됨)
- **v1 불필요**: ① 성공한 지하 생성모델은 전부 대형 합성 corpus(3k~80k샘플)에서 prior 학습, borehole은 추론 조건일 뿐 → 63 borehole로 prior 학습 불가; ② 평형 열장은 매끈(heat eq.)해 생성모델의 이질구조 강점 무용; ③ "필드 1개+메시 1개"는 결정론 회귀 정석.
- **v2(UQ/앙상블)에서만 가치**, 그때 **flow matching > diffusion**(OT 직선경로·적은 스텝·안정·물리제약 변형; TorchCFM, arXiv:2506.08604). GAN(SGAN/GEM)은 prior art only.
- 연결: 6.1의 **합성 GIPL/CryoGrid corpus를 v2 flow-matching 학습데이터로 재사용** → v1→v2 자연 연결.

### 6.3 불확실성(UQ) 계층 (검증됨)
- **Tier 1(즉시)**: kriging/GP 분산 volume + LOBO conformal. ⚠️ 외삽 base에서 과신 → 플래그.
- **Tier 2(주력)**: SGS/조건부 GP 실현 100–500개 → 각자 0 °C 등온면 → (x,y)별 분위수 = 정직한 base 밴드(공간상관 필수).
- **Tier 3(고급)**: SIREN-PINN deep ensemble 5–10개(심부 외삽 UQ). SOTA는 HMC/NUTS.
- **Tier 4(미래)**: 조건부 diffusion — 보류.

### 6.4 시각화 (PyVista, 설치됨)
0 °C 등온면 tri-mesh + frozen body(threshold) + 온도 volume + 깊이 slice + borehole 막대 + ArcticDEM 지형 중첩 + fly-through + 불확실성 색/투명도. export `.vtp/.stl/.ply/.obj/.vtu` → ParaView·Blender·웹(vtk.js/three.js), headless PNG/MP4.

## 7. 시계열 포함 복합 데이터 전처리·활용
정적 평형 모델 → 시계열을 평형 통계로 축약하되 축약 과정에 시계열을 활용:
1. **QC**: flag(`no data entry` 등)·null 제거, 최소 기록 길이 필터.
2. **비편향 연평균**: 완전한 연주기 ≥N개 확보 후 깊이별 **MAGT** → (z,t)를 borehole별 **T(z) 평형 프로파일**로 축약.
3. **epoch 정렬**: Alaska 심부망 가용기간(≈2005–2012)·CCI/Obu epoch에 맞춰 공통 기간.
4. **DZAA**: 깊이별 연진폭→진폭≈0 깊이(10–20 m) 식별→그 아래 선형부에서 **지온경사** 적합→0 °C 깊이(base) 외삽.
5. **ALT 집계**: sporadic 격자/transect→site별 대표 ALT=활성층 하부(영구동토 table, 상부 경계).
6. **공변량 기후평균**: ERA5-Land/MODIS도 동일 epoch 기후값(MAAT, FDD/TDD, 적설일수, NDVI).

→ 최종 학습 테이블: `(x,y,z,MAGT)+공변량` + 경계 가상관측(표면 MAGST, 하부 지열플럭스, ALT top). 원시 시계열은 보존(향후 transient 확장용).

## 8. 결정 필요 / 추후 검토
- 기준 epoch 확정(2005–2012 vs CCI 1997–2021 정합), 지열플럭스/열전도 파라미터 출처, 범극지 확장 시점.
