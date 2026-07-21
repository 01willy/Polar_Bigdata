# W2.1 SoilGrids + KPDC 통합 결과 (2026-07-20)

회의적 재검증 원칙 유지. 누설통제(공간블록+LORO) 게이트. 적대적 검증(KPDC) 통과, SoilGrids 자체 검증.

## W2.1 · SoilGrids 토양 공변량 (게이트 미채택, 단 물리 입력으로 재활용)

취득: ISRIC WCS 2.0.1(IGH 좌표, ~5km). VRT/vsicurl은 정체로 실패 → WCS로 우회.
9층(clay·sand·silt·bdod·cfvo·phh2o 5-15cm + soc 0-5/5-15/15-30cm), 유효율 알래스카 100%·레나 89%·남극만 결측.
산출 `dl_dataset_cell_v3_soil.csv`, `soil_ablation_gate.csv`, 스크립트 `enrich_soilgrids_wcs.py`·`soil_ablation_gate.py`.

| 피처셋 | 공간블록(내삽) skill | LORO(전이) skill |
|---|---|---|
| 기후8 | −7.5% | −29.9% |
| 기후+지형(baseline) | −20.9% | −14.3% |
| **기후+지형+토양** | **+5.6%** | **−63.8%** |
| 기후+토양 | −13.6% | −37.1% |

**판정: 미채택**(공간블록 개선 +0.27, LORO 악화 −0.50). 게이트=둘 다 개선 필요.

**핵심 발견(정보병목의 심화)**: 토양은 전지구 커버라 **결측 라우팅이 없는데도 전이서 붕괴**한다(레나 22→62.6cm, bias +55). 즉 Phase 1의 결측 라우팅과 별개인 **진짜 공변량 이동(covariate shift)**이다. 토양-ALT 관계가 알래스카서 강하나(|Spearman| 0.55-0.58) 레나서 소멸(<0.18)하고 토양 분포도 다르다(레나 SOC=알래스카 1/3). 모델이 알래스카 특유의 토양-ALT 사상을 학습해 레나를 과대예측한다.

**함의**: 공변량 추가는 **내삽에는 이득, 지역 전이에는 손실**이다. 이는 물리 우선(P2)을 강하게 뒷받침한다. 물리(Stefan √TDD)는 지역 불변 보편 법칙이라 전이하지만, region-specific 공변량-ALT 사상은 전이서 깨진다. **토양은 raw ML 입력이 아니라 물리 파라미터 E(땅 열특성) 입력으로 재활용**(W3): SOC·용적밀도가 열전도·잠열과 직결.

## KPDC · 콘슬 현장 검증·물리 사례연구 (대회 KPDC 활용 실현)

데이터: KPDC 콘슬(2019·2021)·c1(2016·2018) 지표 기상(기온·바람·습도·복사). 좌표 콘슬 64.85N,-163.7W(우리 ALT 44셀 존재). 지중온도·ALT 직접 라벨은 없음(지표 기상만).
산출 `kpdc_station_climate.csv`, `kpdc_era5_validation.csv`, `kpdc_council_forcing.csv`, 스크립트 `parse_kpdc_met.py`·`kpdc_era5_validation.py`.

**1) ERA5-Land 공변량 현장 검증(핵심)**: 우리 공변량의 시변 물리 구동항 √TDD가 KPDC 실측과 매우 정합.
- 콘슬 2021: √TDD ERA5 34.33 vs 실측 34.45(bias −0.12). c1 2016/2018: bias +0.12/+0.14. **ERA5 √TDD는 신뢰**.
- MAAT는 ERA5가 −2~−3°C 냉편의(고위도 지표 재분석 한계). TDD 위주로 쓰는 물리에는 영향 작음.
- 콘슬 2019는 성장기(6-9월) 부분값이라 연값과 분리 표기.

**2) 콘슬 물리 forcing 사례연구**: 콘슬 44 ALT 셀에서 Stefan(단일 E).
- 관측 ALT 평균 35.0cm 대비 단일 E Stefan 예측 58.8cm로 **약 1.7배(bias +23.8cm, ratio 1.68) 과대예측**. 지점 실측 forcing(√TDD 35.17)과 ERA5 forcing(36.15) 차이는 미미 → forcing이 아니라 **E가 문제**.
- 콘슬은 습윤·유기질 툰드라라 실효 E가 알래스카 평균보다 낮다. **단일 E가 콘슬을 과대예측 → E를 토양 의존 E(x)로 만들 동기**(W2.1 토양·W3 물리 결합).
- **주의(누설·프레이밍)**: W3 의 콘슬 44셀 교정은 held-out 전이가 아니라 **완전 in-domain**(44셀 전부가 최종 모델 학습에 포함, 43셀 region=ABoVE_AK)이다. 콘슬 44셀은 √TDD 동일값이라 상수 E 모델의 예측은 사실상 단일점이고, PHYS_nn 의 bias 감소(ratio 1.10)는 셀별 물리보정이 아니라 **평균회귀**(예측 표준편차 0.9cm vs obs 13.4cm, 상관 약 0.2)다. 이 결과로 **전이 일반화 주장을 도출하면 안 된다**. E(x) 동기부여 근거로만 사용한다.

## 종합 함의 (다음 순서 재조정)
1. **공변량 추가는 전이에 손실**(covariate shift). 정보병목을 모달리티 추가로 못 뚫는다(2D·3D·토양 모두 확인).
2. **전이의 유일한 escape는 물리**(지역 불변). 따라서 **W3(물리 base 고도화 + 토양 의존 E(x))가 전이의 최우선 지렛대**. 순서상 W2.2(결측 재설계)보다 W3이 전이엔 더 결정적(결측은 한 증상일 뿐, 근본은 covariate shift).
3. 토양·KPDC 모두 물리로 수렴: 토양→E(x) 입력, KPDC→E 과대(단일 E 부적합) 실증.
4. KPDC로 공변량 backbone(√TDD) 현장 검증 완료 = 대회 KPDC 활용 근거 확보.

## 산출 그림
- `outputs/figures/09_soilgrids/{soil_skill_bars,soil_alt_scatter,soil_valid_by_region}.*`, `outputs/maps/soilgrids_ak_lena.*`
- `outputs/figures/10_kpdc/{kpdc_era5_scatter,kpdc_era5_bias_bar,kpdc_forcing_scatter}.*`
