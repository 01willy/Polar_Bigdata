# 멀티에이전트 조사: InSAR 외 데이터 + DL 모델 (2026-07-01)

> 20 에이전트, 데이터 32종·모델 7종. **모든 판정을 우리 실측 데이터로 검증.**
> 직접 재확인: **ALT 분산의 87.7%가 위치간(between)** — 공변량/GBM이 이미 포착. 위치내(within) 12.3%는 std 4.09cm의 **예측불가 잡음**.
> → **병목은 모델 용량이 아니라 공변량 정보량.** 전이를 움직인 유일한 지렛대는 데이터(ERA5 −20%)였음.

## 1. InSAR 외 우선 확보 데이터 (병목완화도 순)
| 순위 | 데이터 | 무엇 | InSAR이 못 담는 것 | 계정/용량 |
|---|---|---|---|---|
| 1 | **GGD200 서시베리아 심부시추공** ✅보유 | 20~2000m 다깊이 암석온도 | 지중온도 직접 라벨 + 서시베리아 전이 | 무계정 104KB |
| 2 | **TPDC 3/6/10/20m 시추공**(고원) | 목표층 깊이 정확일치 온도+ALT | 3D 열구조 라벨 + 고원 전이 도메인 | TPDC 수동 |
| 3 | **ABoVE ds2369** ✅보유 | ALT 22.4만 + 토양수분 | 라벨밀도 + 잠열/열용량(토양수분) | Earthdata |
| 4 | NGEE-Arctic 다심도 온도/DTS | 연속 다심도 토양온도 프로파일 | 0~20m 수직 라벨 | 일부 Earthdata |
| 5 | **SoilGrids/Dai2019 열물성** | 열전도도·열용량(동결/비동결) | 지중 열물성 수직축(PINN 계수) | 무계정 |
| 6 | **지하빙(GGD318)·이탄두께(Hugelius)** | 상부20m excess ice·peat | 융해 잠열·침하 지배 요인 | 일부 Earthdata |
| 7 | Sentinel-2 + peatland + Tundra PFT | 10~20m 식생·이탄·피복 | 단열층(이끼/관목/이탄) | Earthdata/CDSE |
| 8 | ABoVE SnowModel(적설 열저항) | SWE·bulk snow thermal resistance | 겨울 적설 단열(⚠️267GB) | Earthdata |

**핵심**: 1~2위(심부 borehole)가 유일하게 **3D 열구조·전이를 "라벨 자체"로** 공략, 용량 극소(<1GB). 3~5위는 위치간 변동 설명 신규 공변량(ERA5 −20%와 같은 축). 6~8위는 within 잡음 대상이라 상한 낮음(within=총분산 12.3%뿐).

## 2. DL 모델 판정 (정보병목에서 GBM 이길까)
| 모델 | GBM 이길까 | 판정 | 진짜 쓸모 |
|---|---|---|---|
| Conditioned SIREN/INR | **No** | **pilot** | 연속 3D 열구조·∂T/∂z·0°C 등온면 |
| PINN(열전도+Stefan) | No | deprioritize | 심부 라벨 확보 후 v2 |
| Spatial GNN | No | deprioritize | LORO에서 붕괴(IDW/kriging와 동형) |
| Transformer/TabPFN | No | pilot | TabPFN 소표본 UQ만 |
| 지리공간 FM(Prithvi 등) | No | deprioritize | crop-yield LOCO서 XGBoost 못이김+GEE 자격없음 |
| ConvGNP/Neural Process | No | **pilot** | 공간상관 예측공분산(UQ)·전이 |
| Diffusion/Flow-matching | No | pilot | v2 UQ만(flow-matching>diffusion) |

**총평**: 7계열 전부 정보병목 하에서 GBM 유의 우위 **없음**. 모델의 가치는 **정확도가 아니라 산출물 형태**(연속 3D장·보정 UQ·전이).

## 3. 결정적 통찰
- **모델 축**: 패치CNN이 GBM을 2.8%(fold 잡음 이내)만 이김. 미세지형 중요도≈0. → 모델 용량은 병목 아님(실측 확정).
- **데이터 축**: ERA5-Land 하나가 전이 −20%. 나쁜/중복 피처는 오히려 악화. → **좋은 관측=전이 움직임, 모델 교체=안 움직임.**
- **목표별**: (1)ALT 지도=**GBM 유지** / (2)3D 열구조=**SIREN(단, 심부 borehole 라벨 선결)** / (3)불확실성=**ConvGNP·TabPFN**.
- **필수 게이트**: 어떤 NN도 **spatial CV에서 IDW/kriging/GBM을 실제로 이긴 뒤에만** 채택.

## 4. 다음 액션 3 (우선순위)
1. **[즉시] 심부 borehole 라벨 확보** — TPDC 3/6/10/20m(고원, 수동 가입). GGD200은 이미 보유 → 파싱해 3D 라벨로 편입. 목표(2)3D·전이의 라벨 병목 직격, 용량 극소.
2. **[병렬] 위치간 공변량 3종 추가** — SoilGrids/Dai 열물성 + 지하빙(GGD318) + 이탄/단열(peatland·PFT). 알래스카 bbox만(<수GB). **추가마다 공간블록+LORO 재채점**해 나쁜 피처 즉시 제거.
3. **[게이트 조건부] Conditioned SIREN 3D 파일럿** — borehole 라벨 확보 후, IDW/kriging를 spatial CV로 이기면 착수. ALT 정확도용 모델 교체는 전부 보류.

*(주의: 일부 arXiv ID·세부 인용은 에이전트 생성값이라 채택 전 확인 필요. 정량 결론은 우리 데이터로 검증됨.)*
