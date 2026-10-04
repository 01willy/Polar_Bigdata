# SI 학습기·새 지역 근거 색인

- 작성: 2026-10-04. 성격: 논문 SI 용 색인이다. 원본 스크립트, 자료, 문서는 옮기거나 고치지 않았다.
- 범위: 표형 파운데이션 모델 TabPFN v2(LGT), TabICL v2 와 원천 교차검증으로 조정한 판별 신경망 4종(LGF), 새 독립 지역(LGD: 티베트, 러시아 중부, 북대서양), 계산 환경 사이의 재현성.
- 수치 원본: 아래 모든 수치는 `extract_evidence.py` 가 원천 집계 표를 pandas·json 으로 읽어 만든 `evidence_values.csv`(213행)에서 옮겼다. 산문에서 옮긴 값은 없다. 표의 ID 는 `evidence_values.csv` 의 `id` 열과 같다. 재실행: `OMP_NUM_THREADS=1 python3 paper/claims/SI_learners_new_regions/extract_evidence.py`(적합과 재표집 없음, 수 초). `--copies` 를 주면 `tables/` 사본을 읽으며, 2026-10-04 에 두 출력이 바이트 단위로 같음을 확인했다. 예외: 2026-10-04 정정(검증 지적 11건 반영)에서 더한 값은 `evidence_values.csv` 에 없다. 이 값은 2.7 절에 원천 경로, 필터, 열과 함께 따로 적었고 `extract_evidence.py` 와 `evidence_values.csv` 는 바꾸지 않았다.
- 원천 표 사본: `tables/`(19건, sha256 은 `tables/MANIFEST.csv`). 19번째 `lgd_eligibility_v1.csv` 는 2026-10-04 정정에서 L41 변형의 대상 셀 수 근거로 더했다(원본과 sha256 이 같음을 복사 때 확인).
- 판정어: 4분 판정(우세, 열세, 동등, 미결정). 우세·열세는 셀 가중과 블록 등가중 두 95 % CI 가 모두 같은 쪽일 때, 동등은 두 CI 가 모두 ±0.5 cm 안일 때, 미결정은 그 밖이다(`docs/QA_FINAL_REVIEW_2026-10-02.md` 머리말). 재표집은 10,000회다. 예외: LGD 의 L1e·L4e·L8e 층화 평균 행은 h40 형식의 이진 판정(`sig` 열: improve, ns, worse, 재표집 1,000회)이다.
- 표기: Δ(A − B) = RMSE(A) − RMSE(B), cm. 음수면 A 가 더 정확하다. 값 칸은 '셀 가중 Δ [95 % CI]; 블록 등가중 Δ [95 % CI]' 순서다. 대비 행의 열은 모두 `delta, ci_lo, ci_hi, delta_blockeq, ci_lo_beq, ci_hi_beq` 이고 판정어 열은 `verdict4`(LGD 층화 평균 행은 `sig`)다. 표의 n = 전량은 원천 표에서 `nall`, `all` 또는 n = −1 이다.
- 기호: T = TabPFN v2, I = TabICL v2, C = 같은 컨텍스트 행으로 적합한 CatBoost(`catboost_ctx`), l* = 원천 지역 하나 제외 교차검증으로 고른 조정판, @full = 원천 전체를 컨텍스트로 준 변형. P0 = 원천 계수 Stefan, P1 = 같은 라벨로 재보정한 Stefan(원천 계수 쪽 수축, κ = 10), P2 = 대상 라벨만의 최소제곱 Stefan(수축 없음), P3 = 오프셋 최대우도 Stefan. D0 = 직접 ML, R0·R1 = P0·P1 앵커 + 잔차(기준 λ 0.25). 주 4지역 = 레나, 캐나다, 러시아 W, 러시아 E(모드 x). n 40·160·320 의 평균은 레나·캐나다 2지역이다. '3지역 보조 열' 은 레나·캐나다·Alaska(x)다.

---

## 1. 주장 문장

### 1.1 현행 문장

이 폴더가 받치는 문장은 하나의 주장이 아니라 여러 주장의 학습기·지역 범위 단서다. 현행 문장은 다음과 같다.

`docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md`:

> 43행(C1): 라벨 0개에서 직접 ML 은 시험한 학습기 10종 모두에서 원천 계수 물리식보다 오차가 컸다. 학습기는 CatBoost 3설정(기본, 용량 확대, 원천 교차검증 조정), 랜덤 포레스트, 표형 파운데이션 모델 TabPFN v2 와 TabICL v2(두 컨텍스트 조건), 원천 교차검증으로 초모수를 고른 판별 신경망 4종이다(AB1, L30, LGT L34(a), LGF-F1, LGF-N1). 한계: TabICL v2 직접 ML 은 라벨 10개에서 주 4지역 평균으로 물리식보다 2.91 cm 낫고, 알래스카를 넣은 3지역 평균에서는 5.98 cm 나쁘다(LGF-F1 보조 (a)).

> 52행(C2 표): 티베트 | 물리식 245 cm | 라벨 10개에서 잔차 ML 이 재보정보다 21 cm 낫다

> 61행(C3): 잔차 구조의 이득은 학습기 종류에 묶이지 않는다. 같은 라벨로 재보정한 물리식 대비 순가치는 TabICL v2 잔차에서 라벨 10개부터(−0.55 cm, 전량 −1.00 cm), TabPFN v2 잔차에서 라벨 3개부터(−0.32 cm) 나타났다(LGF-F4, LGT L33). 크기는 학습기마다 다르다.

> 112행(한계): 새 지역: 약관이 확인된 새 얕은 지역은 러시아 중부 하나다.

> 116–117행(한계): 학습기(LGF 판정) … 원천 교차검증 조정은 전이 오차를 줄이지 않았다(N2). 재현성: 신경망은 계산 환경에 따라 결과가 달라진다(FT-T 키 단위 최대 5.6 cm). 워크플로는 CatBoost 와 해석식으로 구성한다.

`docs/RESEARCH_OVERVIEW_2026-10-02.md` 84행: "CatBoost 와 물리식은 Rescale 과 로컬 사이 최대 차가 3.6e-14 cm 였고, Rescale 노드 종류(elm, hematite) 사이에서는 비트 단위로 같았다(WF9 의 WF6 재현 74/74 조각). 신경망은 환경에 따라 최대 5.6 cm 달랐다."

등록된 원고 문장(판정 기록):
- LG 계획서 7.4: "표형 파운데이션 모델(TabPFN)을 써도 라벨 0 직접 예측은 물리식을 넘지 못한다(L34 a)", "TabPFN 과 같은 컨텍스트의 CatBoost 사이의 차이는 확인되지 않았다(L32)" 를 SI 학습기 절에 쓴다. 본문은 "CatBoost 를 넘는 학습기는 없었다" 로 쓴다.
- LG 계획서 7.3: L4 의 '희소 라벨에서도 ML 순가치 있음' 은 주 4지역 한정 문장이고, 확장 풀에서는 유지되지 않는다는 문장을 함께 쓴다. 독립 지역 검증 문장은 약관 확인분 판(새 얕은 지역 Russia_C 1곳)을 근거로 쓰고, NAtlantic 은 SI 의 전체 판에만 둔다.
- LGF 계획서 10.4: 라벨 0 직접 ML 의 근거 학습기 목록과 TabICL v2 n = 10 한계, F4 순가치, 초모수 조정(N2, N2s, N3) 문장.

### 1.2 필요한 축소

`docs/QA_FINAL_REVIEW_2026-10-02.md` 의 공통 축소 항목(Q1 C2 점검, '안전' 표현, WF 사후 설계)과 이 폴더의 원천 표에서 확인한 단서를 적용한 결과다.

| 항목 | 이 폴더에 대한 적용 | 근거 ID |
|---|---|---|
| C2 축소(재보정 몫과 ML 몫의 분리) | 티베트 '라벨 10개 잔차 ML 이 재보정보다 21 cm 낫다' 의 비교 상대는 수축 재보정 P1 이다. 원천 표가 이 행에 '수축 사전분포 오지정 의존' 표지를 달았다. 수축 없는 현지 최소제곱 P2 대비로는 R1 − P2 가 +19.11 cm [5.74, 30.14]; 블록 등가중 +2.37 [−11.81, 16.77] 로 미결정이다. 셀 가중 CI 가 전부 0 초과이므로 셀 가중으로는 P2 가 R1 보다 정확하고, R1 은 P2 보다 낫지 않았다. R1 − P3 도 +16.65 [9.53, 22.48]; 블록 등가중 +5.20 [−3.29, 13.58] 로 같은 형태다. P2 − P1 은 −40.02 cm(우세, 원천 표지 '분할 독립 가정 의존')다. 따라서 티베트 값은 'ML 이 재보정을 넘는다' 가 아니라 '원천 계수가 크게 틀린 곳에서 수축 사전분포가 편향을 남긴다' 의 예다. 러시아 W 의 −13.98 cm 는 R1 − P0 이고 같은 지역 R1 − P1(n = 10)은 −0.56 cm(미결정)다. C2 문장에는 QA 권고문('라벨을 쓰는 보정(재보정 + 잔차)의 이득은 원천 계수 오차의 크기에 비례 … 재보정을 넘어선 ML 의 추가 이득은 계수 오차 크기와 상관이 없었다')을 쓰고 티베트는 P0 대비 예로만 든다 | D-L38-Tibet-R1-P1-n10, -R1-P2-n10, -R1-P3-n10, -P2-P1-n10, D-L38-RussiaW-ref-* |
| 티베트의 해석 범위 | 티베트의 P0 기반 대비에는 '예측된 결과(비맹검); 비맹검(라벨 통계 열람)' 표지가 있다. 채점 셀의 원천 지지 밖 비율은 고도 1.000, √TDD 0.333 이고, L38 판정 행은 '티베트 결과는 NOVELTY-N7 문장의 근거로 쓰지 않는다' 고 적었다. 심부 레짐(PE2)의 cm 단위 평균은 티베트가 지배한다(L8e PE2 −26.78 cm) | D-L38-V-note, D-L8e-PE2 |
| '안전' 표현 | 이 폴더의 판정 가운데 등록된 비열등 판정은 없다. F1·N1·L34(a) 는 '우세가 아니면 지지' 형식이고 판정 대비가 모두 열세라 '물리식보다 오차가 컸다' 로 쓴다. L34(b)·F5 는 등록된 양측 동등 판정(한계 0.5 cm)이라 '구별되지 않는다(한계 0.5 cm)' 로만 쓴다. '안전하다', '손해가 없다' 는 쓰지 않는다 | T-L34a, T-L34b, F-F1-*, F-F5, N-N1-* |
| WF 사후 설계 표지 | 이 폴더의 학습기·지역 가설(L32–L42, F1–F6, N1–N4)은 결과 열람 전 등록이다. 다만 비맹검 부분이 있다. (1) L32·L33 의 R1 행 가운데 n ∈ {3, 10, 40} 은 '재현(비맹검)'(b4 에서 방향을 보았다)이고 n = 0, 160, 320, 전량 행은 맹검이다(`lgt_tests.csv` 의 `blind` 열, L32·L33 verdict_aux 의 `note`. LG 7.4 의 'n ≤ 40' 문구는 n = 0 까지 포함하는 것처럼 읽히므로 쓰지 않는다). (2) LGF-N1(확인적)은 표지 '비맹검 부분 포함(기본판의 방향)' 이다(확인적 행의 `prior_info` 'M1(사전 정보: 기본 초모수 신경망의 방향)'). (3) LGF 계열 문장(3.6)은 'T 쪽 비맹검(LGT 표 열람 뒤 판정)' 이다. (4) F1·F4·F5 의 C 병기 행(같은 컨텍스트 CatBoost)은 '재현(비맹검, M1·a2)', F4 의 TabPFN 병기 행 n ∈ {3, 10, 40} 은 '재현(비맹검, b4)' 이다. (5) F3 의 TabPFN 쪽은 비맹검, LGD 티베트 대비와 L39 는 비맹검, L40 은 재현(비맹검)이다. 1.3 절의 TabPFN 순이득(n 3 −0.32, n 10 −0.42 cm)은 (1)의 비맹검 행이다. WF 계열 수치는 WF9 재현 점검(WF6 조각 대조) 하나뿐이고 판정이 아닌 점검이다. WF9 자체는 WF 1·2차 결과 열람 뒤 설계한 3차 실험이다 | R-wf9, T-L33-n3, -n10, S-blind-*(2.7 절) |
| 플랫폼 규칙 | LGT·LGF·LGD 새 지역 조각은 로컬, LG·LGX 는 Rescale 조각이다. 지역별 대비(A − B)는 한 플랫폼 안에서 계산했다(LGF 2.2, LG 개정 14 (5)). 확장 풀(PE1·PE2)의 층화 평균은 플랫폼이 다른 지역을 합친 값이다(`lgd_pool_lic.csv` 의 `platforms` 'local,rescale'). 그래서 L1e·L4e·L8e 의 PE1·PE2 행(지역 행, MEAN, 판정 행), compare 행과 compare_pool 행에는 '교차 환경; (i) CatBoost 불통과' 표지가 있다(`lgd_tests_lic.csv` 의 `cross_env` 열). 1.3 절의 '5지역 전량 R1 − P0 −3.19' 와 '5지역 n = 10 −0.13' 이 이 교차 환경 값이다. L40 확충판 비교도 '교차 환경(보조)' 다. 플랫폼 묶음 사이의 차(예: TabICL − Rescale CatBoost)는 계산하거나 서술하지 않는다 | D-pool-PE1, D-pool-PE2, S-xenv-*(2.7 절) |
| 재현성 문장의 범위 | 'CatBoost 는 환경 사이에서 같다' 는 기준 방법(base 축) 키에 한정한다. ridge 앵커 위 CatBoost(V1r)는 점검 (i)에서 최대 0.46 cm, LGD 재현 점검의 병기 범위에서 최대 0.98 cm 달랐고 ridge 는 최대 0.0205 cm 달랐다. '5.6 cm' 는 두 단위가 다른 값이다: FT-T 키 단위 최대 5.61 cm(점검 (iii))와 RealMLP 지역 RMSE 차 최대 5.59 cm(`lgfn_cross`) | R-C14, R-gate-i, R-lgd-repro-all, R-gate-iii, R-lgfn-cross |
| '학습기 10종' 의 셈 | TabICL v2 의 두 컨텍스트 조건은 한 학습기로 센다. 이 폴더가 받치는 것은 그 가운데 6종(TabPFN v2, TabICL v2, 조정 신경망 4종)이고, CatBoost 3설정과 랜덤 포레스트는 C1 폴더(`paper/claims/C1_label0_safety/`)가 받친다 | Fig6a-* |
| n = 40·160 의 기준선 | LG 7.4: n = 40·160 에는 더 강한 재보정 기준선 P1*(P1@ed)이 있으나 LGT·LGF 집계는 P1* 대비를 만들지 않았다. 이 n 에서 'TabPFN·TabICL 잔차가 재보정 물리식을 넘는다' 는 P1 대비 보조 결과로만 쓰고 P1* 를 넘는다는 문장은 쓰지 않는다 | T-L33-n40, -n160, F-F4-n40, -n160 |
| L4 의 범위 | 희소 라벨(n ≤ 10) 순가치는 주 4지역 한정이다. 약관 확인분 확장 풀 PE1 에서는 n = 10 이 동등으로 약화되었다(L28 분류) | D-L4e-compare, D-L4e-pool-PE1-10 |
| 'CatBoost 를 넘는 학습기는 없었다' | 문장 범위를 판정 문장의 대비(R1, λ 0.25 의 R1[T] − R1[C], R1[I] − R1[C])로 한정한다. 이 범위의 예외는 F2 n = 160 −0.55 cm [−0.79, −0.19]; 블록 등가중 −0.54 [−0.77, −0.30](우세, 레나·캐나다 2지역)다. 같은 대비의 추가 n 행 n = 320 도 −0.55 cm(우세, 2지역, 판정 문장 밖의 '보조(추가 n)')다. 보조 대비에서는 두 파운데이션 모델의 직접 예측과 R0 가 같은 컨텍스트 CatBoost 보다 0.66–4.09 cm 작았다(우세): D0[T] − D0[C] n = 10 −3.08, 전량 −3.40. D0[I] − D0[C] n = 10 −4.09, 전량 −4.08. R0[T] − R0[C] n = 10 −0.66, 전량 −0.69. R0[I] − R0[C] n = 10 −0.96, 전량 −1.01. 원천 전체 컨텍스트 잔차 R1@full[I] − R1@full[C] 전량 −0.49(우세, 크기 0.5 cm 미만)도 있다. 모든 비교 상대는 컨텍스트 상한 10,000행(@full 은 원천 전체)으로 적합한 `catboost_ctx` 이고 원천 전체로 학습한 `catboost_lo` 가 아니다 | F-F2-n160, T-L32-aux, F-F2-aux-*, S-ctx-*(2.7 절) |

### 1.3 권고 문장

한국어(SI 학습기·새 지역 절):

> 라벨 0 전이에서 직접 ML 은 표형 파운데이션 모델 2종과 원천 지역 하나 제외 교차검증으로 초모수를 고른 판별 신경망 4종에서도 원천 계수 Stefan(P0)보다 오차가 컸다(주 4지역 셀 가중 Δ: TabPFN v2 1.02, TabICL v2 0.93(원천 전체 컨텍스트 0.85), MLP 2.45, 다중 헤드 MLP 2.44, 축소 FT-Transformer 4.13, RealMLP 5.21 cm, 두 가중의 CI 모두 0 초과). 예외로 TabICL v2 직접 ML 은 라벨 10개에서 주 4지역 평균으로 P0 보다 2.91 cm 작았다. 이 차이는 러시아 W(−13.43 cm)에서 왔고, 레나·캐나다·알래스카 3지역 보조 열에서는 반대로 5.98 cm 컸다. 잔차 구조에서 같은 라벨로 재보정한 Stefan(P1) 대비 순이득은 두 파운데이션 모델에서도 나타났으나 작았다(TabPFN v2 라벨 3개 −0.32, 10개 −0.42 cm, 두 값 모두 재현(비맹검) 행. TabICL v2 라벨 10개 −0.55, 전량 −1.00 cm). 라벨 0 에서 두 모델의 잔차 예측(R0)은 P0 와 구별되지 않았다(한계 0.5 cm, 주 4지역). 알래스카를 넣은 3지역 보조 열에서는 TabICL v2 의 R0 가 P0 보다 0.64 cm 컸다(열세. 블록 등가중 +0.71 [0.25, 1.18]). 같은 열에서 TabPFN v2 의 R0 는 −0.13 cm 로 미결정이었으나 블록 등가중 CI [0.10, 0.66] 은 0 초과였다. 원천 교차검증으로 고른 신경망 초모수는 전이 오차를 줄이지 않았다. MLP 와 다중 헤드 MLP 는 직접 ML 의 오차가 오히려 늘었다. 4분 판정에서 우세(조정판의 오차 감소)인 대비는 축소 FT-Transformer 의 전량 잔차(λ 0.25 에서 −0.19 cm, 크기 0.5 cm 미만. λ 1.0 에서 −0.76 cm)뿐이었다. 점 추정이 음수인 나머지 대비(예: 축소 FT-Transformer 직접 ML 라벨 0 −0.82 cm)는 모두 동등 또는 미결정이라 감소로 쓰지 않는다. 원천 교차검증 이득과 대상 오차 변화의 순위상관은 −0.32(점 32개, 서술)였다.

> 약관이 확인된 새 독립 지역은 얕은 레짐의 러시아 중부(57셀)와 심부 레짐의 티베트(132셀) 두 곳이다. 독립 지역 검증의 근거는 얕은 레짐 새 지역인 러시아 중부 하나다. 러시아 중부를 더한 얕은 레짐 5지역 풀(PE1)에서 '라벨 0–40 에서 직접 ML 이 P0 를 넘지 못한다'(유의 개선 지역 0/5)와 '라벨 전량에서 잔차 ML 이 P0 를 넘는다'(R1 − P0 −3.19 cm [−4.19, −2.22]; 블록 등가중 −2.62 [−3.71, −1.53])는 유지되었다. 이 풀 값은 로컬에서 돌린 새 지역 조각과 Rescale 에서 돌린 주 4지역 조각을 층화 평균으로 합친 값이라 교차 환경 표지가 붙는다. 라벨 10개 이하의 재보정 대비 순이득은 확장 풀에서 유지되지 않았다(5지역 n = 10 −0.13 cm, 동등, 교차 환경. 러시아 중부 단독 +0.06 cm, 미결정). 러시아 중부 단독으로는 라벨 전량의 잔차 ML 도 P0 를 넘는다고 판정되지 않았다(R1 − P0 −5.42 cm [−8.96, −1.64]; 블록 등가중 −2.55 cm [−6.83, 1.48], 미결정(두 가중의 CI 가 엇갈림)). L38 판정은 이 지역을 '어긋난 지역' 으로 분류했다. 티베트는 원천 계수 Stefan 의 RMSE 가 244.51 cm 인 심부 레짐이고 채점 셀이 모두 원천 고도 범위 밖이다. 심부 레짐 티베트에서는 직접 ML 이 라벨 0–40 의 모든 n 에서 P0 보다 유의하게 작았다(n 0, 3, 10, 40 에서 −59.98, −130.18, −150.92, −157.59 cm). 6지역 풀(PE2)의 유의 개선 지역은 이 1/6 뿐이어서 기각 기준(⌊6/4⌋ = 1 초과) 이하였고, 판정은 '지지' 로 남았다. 티베트 결과는 비맹검 표지가 있고 원천 지지 밖이므로 독립 지역 검증의 근거로 쓰지 않는다. 이 지역에서 라벨 10개 잔차 ML 은 수축 재보정(P1)보다 20.90 cm 작았다. 그러나 수축 없는 현지 최소제곱 재보정(P2)보다는 낫지 않았다(R1 − P2 +19.11 cm [5.74, 30.14]; 블록 등가중 +2.37 [−11.81, 16.77], 미결정. 셀 가중으로는 P2 가 더 정확하다). 북대서양은 약관 확인 전 자료를 포함해 SI 전체 판에만 싣는다.

> 물리식과 CatBoost 기준 방법의 결과는 계산 환경 사이에서 같았다(Rescale 과 로컬 사이 학습 키 34,360개의 최대 차 3.6e-14 cm. Rescale 의 서로 다른 노드에서 돌린 WF9 조각 74개와 WF6 조각의 공통 키 11,814개에서 SSE 차 0). 같은 코드와 자료에서 신경망은 플랫폼에 따라 달랐다(FT-Transformer 키 288개 가운데 43개가 0.5 cm 초과, 최대 5.61 cm. 한 플랫폼 안 재적합 차는 0). ridge 앵커 위 CatBoost 도 플랫폼 사이 최대 0.46 cm 달랐다. 그래서 지역별 대비는 한 플랫폼 안에서 계산했다. 확장 풀(PE1·PE2)의 층화 평균은 플랫폼이 다른 지역을 합친 값이므로 교차 환경 표지를 단다. 워크플로는 CPU 에서 결정적으로 재현되는 CatBoost 와 해석식으로 구성했다.

영어(SI 초안):

> At zero target labels, direct ML exceeded the error of the source-coefficient Stefan model (P0) also for two tabular foundation models and four discriminative neural networks whose hyperparameters were selected by leave-one-source-region-out cross-validation (mean over the four main regions, cell-weighted ΔRMSE: TabPFN v2 1.02, TabICL v2 0.93 (0.85 with the full source as context), MLP 2.45, multi-head MLP 2.44, reduced FT-Transformer 4.13, RealMLP 5.21 cm; both cell-weighted and block-equal 95 % CIs above zero). The exception was direct TabICL v2 with 10 labels, which was 2.91 cm better than P0 on the four-region mean; this came from Russia W (−13.43 cm), and on the auxiliary three-region mean including Alaska it was 5.98 cm worse. Net gains of the residual structure over the Stefan model recalibrated with the same labels (P1) also appeared with both foundation models but were small (TabPFN v2: −0.32 cm at 3 labels, −0.42 cm at 10, both from replicated, non-blind rows; TabICL v2: −0.55 cm at 10, −1.00 cm with all labels). At zero labels, the residual predictions (R0) of both models were indistinguishable from P0 within a 0.5 cm margin on the four main regions; on the auxiliary three-region mean including Alaska, TabICL v2 R0 was 0.64 cm worse than P0, and TabPFN v2 R0 (−0.13 cm) was undetermined with a block-equal CI above zero. Selecting network hyperparameters by source-region cross-validation did not reduce transfer error; it increased the error of direct ML for the MLP and the multi-head MLP. The only contrasts classified as a reduction in the four-way verdict were the all-label residual of the reduced FT-Transformer (−0.19 cm at λ = 0.25, below 0.5 cm in size; −0.76 cm at λ = 1.0); other negative point estimates were equivalent or undetermined. The rank correlation between the source cross-validation gain and the change in target error was −0.32 (32 points, descriptive, no test).

> Two licence-verified independent regions were added: central Russia (57 cells, shallow regime) and Tibet (132 cells, deep regime). Independent-region validation rests on the single new shallow region, central Russia. In the five-region shallow pool including central Russia, the verdicts that direct ML does not beat P0 at 0–40 labels (no region with significant improvement, 0/5) and that residual ML beats P0 with all labels (−3.19 cm [−4.19, −2.22]; block-equal −2.62 [−3.71, −1.53]) were preserved. These pooled values combine new-region units run locally with main-region units run on the cloud platform and are flagged as cross-environment. The net gain over P1 at ≤10 labels did not hold in the expanded pool (five-region mean at 10 labels −0.13 cm, equivalent, cross-environment; central Russia alone +0.06 cm, undetermined). In central Russia alone, residual ML with all labels was not shown to beat P0 (R1 − P0 −5.42 cm [−8.96, −1.64]; block-equal −2.55 cm [−6.83, 1.48]; undetermined because the two weightings disagree), and the regional test classified it as a non-replicating region. Tibet (P0 RMSE 244.51 cm; all scored cells outside the source elevation range) is reported separately: there, direct ML was significantly better than P0 at every label count from 0 to 40 (−59.98, −130.18, −150.92 and −157.59 cm at 0, 3, 10 and 40 labels); with one region of six significant, the six-region pool stayed below the rejection threshold. Tibetan results carry a non-blind flag, lie outside the source support, and are not used as independent-region validation. Residual ML with 10 labels was 20.90 cm better than the shrunk recalibration P1 but was not better than unshrunk local least squares P2 (+19.11 cm; cell-weighted CI above zero, block-equal CI including zero; undetermined). The North Atlantic region, which includes data whose licence is not yet confirmed, is reported only in the full-data edition of the Supplementary Information.

> Physics and CatBoost baseline methods were reproducible across computing environments (maximum difference 3.6e-14 cm over 34,360 keys between cloud and local runs; zero SSE difference over 11,814 common keys in 74 units run on two cloud node types), whereas neural networks were not (FT-Transformer: 43 of 288 keys differed by more than 0.5 cm, maximum 5.61 cm, with identical refits within one platform). CatBoost on a ridge anchor also differed by up to 0.46 cm between platforms. Region-level contrasts were therefore computed within one platform; pooled means over the expanded pools (PE1, PE2) combine regions run on different platforms and are flagged as cross-environment. The workflow was built from CatBoost and closed-form physics, which reproduce deterministically on CPU.

---

## 2. 근거 표

### 2.1 원천 약호

| 약호 | 원천 경로 | 사본 | 내용 |
|---|---|---|---|
| T | `data/processed/lgt/lgt_tests.csv` | `tables/lgt_tests.csv` | LGT L32–L37 대비와 판정(402행) |
| F | `data/processed/lgf/lgf_tests.csv` | `tables/lgf_tests.csv` | LGF-F1–F6 대비와 판정(406행) |
| N | `data/processed/lgf/lgfn_tests.csv` | `tables/lgfn_tests.csv` | LGF-N1–N4 대비와 판정(1,571행) |
| D | `data/processed/lgd/lgd_tests_lic.csv` | `tables/lgd_tests_lic.csv` | LGD 약관 확인분 판(주 판정, 546행) |
| Dfull | `data/processed/lgd/lgd_tests.csv` | `tables/lgd_tests.csv` | LGD 전체 판(SI, 약관 확인 전 자료 포함, 634행) |
| Dpool | `data/processed/lgd/lgd_pool_lic.csv` | `tables/lgd_pool_lic.csv` | 약관 확인분 판의 풀 구성 |
| Delig | `data/processed/lgd/lgd_eligibility_lic.csv` | `tables/lgd_eligibility_lic.csv` | 약관 확인분 판의 적격 표 |
| Drepro | `data/processed/lgd/lgd_repro_gate.csv` | `tables/lgd_repro_gate.csv` | LGD 재현 점검 (c)2 |
| Tgate | `data/processed/lgt/lgt_gate.csv` | `tables/lgt_gate.csv` | LGT 와 LG 조각의 P0·P1 대조 |
| Fgate | `data/processed/lgf/lgf_gate.csv` | `tables/lgf_gate.csv` | LGF 와 LGT 조각의 짝 점검 |
| Ncross | `data/processed/lgf/lgfn_cross.csv` | `tables/lgfn_cross.csv` | 신경망 로컬 기본판 대 LG(Rescale) 교차 환경 대비 |
| Ngate | `data/processed/lgf/lgfn_cross_gate.csv` | `tables/lgfn_cross_gate.csv` | 위 대비의 물리식 키 대조 |
| WF9 | `data/processed/wf/wf9_repro_check.csv` | `tables/wf9_repro_check.csv` | WF9(hematite) 대 WF6(elm) 조각 대조 |
| XC14 | `data/processed/lgw/lgw_xenv_c14_summary.json` | `tables/lgw_xenv_c14_summary.json` | C14: CatBoost 기준 방법 로컬 재적합 대조 |
| XI | `data/processed/lgw/lgw_xenv_gate_i_summary.json` | `tables/lgw_xenv_gate_i_summary.json` | 교차 환경 점검 (i) |
| XIII | `data/processed/lgw/lgw_xenv_gate_iii_summary.json` | `tables/lgw_xenv_gate_iii_summary.json` | 교차 환경 점검 (iii)(FT-T) |
| Fig6a | `outputs/figures/paper/source_data/v2/Fig6_a.csv` | `tables/Fig6_a.csv` | Fig 6a 원천 자료(대조용) |
| Tab1 | `outputs/figures/paper/source_data/v2/Table1_rows.csv` | `tables/Table1_rows.csv` | Table 1 원천 자료(대조용) |

복사하지 않은 원천: `data/processed/lgw/lgw_xenv_c14.csv`(5,559,432 bytes, 5 MB 상한 초과). 같은 값의 요약인 `lgw_xenv_c14_summary.json` 을 대신 복사했다.

### 2.2 TabPFN v2(LGT, 원천 T, 판정 절 LG 7.4, 가설 6C.6)

공통 필터: `test_id` 와 `scope=MEAN`(주 4지역 층화 평균, n 40·160·320 은 레나·캐나다 2지역). 3지역 보조 행은 `scope=MEAN3`. 실행: 로컬 RTX 3090, 172/172 단위, 실패 0(LG 7.4).

| ID | 대비(contrast) | 값(cm) | 판정어 | 비고 |
|---|---|---|---|---|
| T-L34a | `D0[T]-P0\|n0`, test_id=L34 | 1.02 [0.23, 2.35]; 블록 등가중 1.22 [0.27, 2.15] | 열세 | holm_p 0.12. L34(a) 판정 '지지' |
| T-L34a-C | `D0[C]-P0\|n0`, test_id=L34 | 2.64 [1.36, 3.72]; 블록 등가중 1.79 [0.69, 2.81] | 열세 | 병기 |
| T-L34b | `R0[T]-P0\|n0`, test_id=L34 | −0.31 [−0.48, 0.05]; 블록 등가중 −0.21 [−0.43, 0.02] | 동등 | L34(b) '구별되지 않는다(한계 0.5 cm)' |
| T-L34b-C | `R0[C]-P0\|n0`, test_id=L34 | −0.05 [−0.42, 0.16]; 블록 등가중 −0.21 [−0.48, 0.04] | 동등 | 병기, 비맹검(`blind` False) |
| T-M3-L34a | `D0[T]-P0\|n0`, scope=MEAN3 | 7.83 [5.40, 8.66]; 블록 등가중 4.96 [3.94, 5.98] | 열세 | 3지역 보조 열 |
| T-L32-n0 | `R1[T]-R1[C]\|n0\|lam0.25`, test_id=L32 | −0.26 [−0.37, 0.15]; 블록 등가중 0.00 [−0.19, 0.21] | 동등 | |
| T-L32-n10 | `R1[T]-R1[C]\|n10\|lam0.25` | −0.26 [−0.35, 0.13]; 블록 등가중 0.05 [−0.13, 0.23] | 동등 | 재현(비맹검) |
| T-L32-n40 | `R1[T]-R1[C]\|n40\|lam0.25` | −0.10 [−0.20, 0.13]; 블록 등가중 0.08 [−0.06, 0.22] | 동등 | 2지역, 재현(비맹검) |
| T-L32-n160 | `R1[T]-R1[C]\|n160\|lam0.25` | −0.30 [−0.47, −0.02]; 블록 등가중 −0.15 [−0.31, 0.01] | 동등 | 2지역 |
| T-L32-nall | `R1[T]-R1[C]\|nall\|lam0.25` | −0.28 [−0.38, 0.08]; 블록 등가중 0.01 [−0.16, 0.20] | 동등 | |
| T-L32-lam1-* | `R1[T]-R1[C]\|{n0,n10,n40,n160,nall}\|lam1.0` | −1.60, −1.51, −0.53, −1.08, −1.22(셀 가중 점 추정) | 다섯 모두 미결정 | CI 는 `evidence_values.csv` |
| T-L32-V | scope=verdict_aux, test_id=L32, 열 `verdict` | '부분(지역 2/4): 차이를 확인하지 못함' | | λ 1.0 의 미결정 때문에 '동등' 이 아니다 |
| T-L32-aux | `D0[T]-D0[C]\|n10`, `\|nall` | −3.08 [−3.70, −1.75]; −3.40 [−3.94, −1.84] | 우세, 우세 | 직접 예측끼리의 보조 대비. 같은 행의 R0[T] − R0[C](n 10 −0.66, 전량 −0.69, 우세)는 2.7 절 S-ctx-T-R0 |
| T-L33-n3 | `R1[T]-P1\|n3`, test_id=L33 | −0.32 [−0.46, −0.02]; 블록 등가중 −0.19 [−0.38, −0.00] | 우세 | 크기 0.5 cm 미만, holm_p 0.24, 재현(비맹검) |
| T-L33-n10 | `R1[T]-P1\|n10` | −0.42 [−0.54, −0.18]; 블록 등가중 −0.31 [−0.47, −0.14] | 우세 | 크기 0.5 cm 미만, 재현(비맹검) |
| T-L33-n40 | `R1[T]-P1\|n40` | −0.62 [−0.78, −0.43]; 블록 등가중 −0.70 [−0.84, −0.57] | 우세 | 2지역, P1* 대비 없음, 재현(비맹검) |
| T-L33-n160 | `R1[T]-P1\|n160` | −0.93 [−1.20, −0.64]; 블록 등가중 −1.02 [−1.18, −0.85] | 우세 | 2지역, P1* 대비 없음 |
| T-L33-n320 | `R1[T]-P1\|n320` | −1.10 [−1.46, −0.75]; 블록 등가중 −1.17 [−1.36, −0.99] | 우세 | 2지역 |
| T-L33-nall | `R1[T]-P1\|nall` | −0.80 [−1.02, −0.51]; 블록 등가중 −0.65 [−0.83, −0.47] | 우세 | |
| T-L33C-* | `R1[C]-P1\|{n3,n10,n40,nall}` | −0.10, −0.16, −0.51, −0.52 | 미결정, 우세(크기 0.5 cm 미만), 우세, 우세 | 병기. n 3·10·40 은 비맹검(`blind` False) |
| T-M3-L33-n10 | `R1[T]-P1\|n10`, scope=MEAN3 | −0.18 [−0.32, 0.09]; 블록 등가중 0.14 [−0.05, 0.33] | 동등 | 3지역 보조 열 |
| T-L33-V | scope=verdict_aux, test_id=L33 | '희소 라벨에서도 TabPFN 잔차의 순가치가 있다(우세인 최소 n: 4지역 평균 n=3, 2–3지역 평균 n=40)' | | |
| T-L35 | `R1[T]-P0\|all`, test_id=L35 | −3.03 [−3.79, −2.09]; 블록 등가중 −2.78 [−3.72, −1.76] | 우세 | 병기 R1[C] −2.76, D0[T] −3.63(모두 우세) |
| T-M3-L35 | `R1[T]-P0\|all`, scope=MEAN3 | 0.64 [−0.01, 1.08]; 블록 등가중 1.18 [0.69, 1.67] | 미결정 | 3지역 보조 열 |
| T-L36-* | `D0[T]-R1[T]\|{n0,n10,n40,n160,nall}`, test_id=L36 | 1.33, 0.97, −1.79, −2.45, −0.60 | 열세, 열세, 우세, 우세, 미결정 | n 40·160 은 2지역 |
| T-L37-V-* | scope=verdict_aux, test_id=L37, clause=안정성, variant ∈ {d3, d10, rid, tgt} | d3·d10·rid '부분(지역 2/4): 강건', tgt '부분(지역 2/4): 약화: R1[T]-P1 n=40' | | tgt 는 컨텍스트 40행이라 범주형 추론이 꺼진다(6C 개정 12) |

### 2.3 TabICL v2(LGF-F, 원천 F, 판정 절 LGF 10.2, 가설 3.6)

공통 필터: `test_id`, `scope=MEAN`(주 4지역; n 40·160·320 은 2지역), 기준 λ(D0 1.0, R0·R1 0.25). F3 의 TabPFN 짝 대비는 레나·러시아 W 2지역이다(2.6 R-lgf-gate).

| ID | 대비 | 값(cm) | 판정어 | 비고 |
|---|---|---|---|---|
| F-F1-D0[I]-P0\|n0 | `D0[I]-P0\|n0`, test_id=LGF-F1 | 0.93 [0.28, 3.12]; 블록 등가중 2.43 [1.35, 3.55] | 열세 | 확인적, holm_p 0.02 |
| F-F1-D0@full[I]-P0\|n0 | `D0@full[I]-P0\|n0` | 0.85 [0.07, 3.19]; 블록 등가중 2.22 [1.09, 3.45] | 열세 | 확인적, holm_p 0.04 |
| F-F1-D0[C]-P0\|n0, -D0@full[C] | `D0[C]-P0\|n0`, `D0@full[C]-P0\|n0` | 2.64; 2.26 | 열세, 열세 | 병기, 재현(비맹검, M1·a2) |
| F-F1-V | scope=verdict, 열 `verdict` | '지지(물리식보다 오차가 크거나 구별되지 않음): … TabICL v2 를 학습기로 써도(컨텍스트 상한 10,000행과 원천 전체의 두 조건) 직접 ML 은 라벨 0 전이에서 원천 계수 물리식을 넘지 못한다' | | |
| F-F1a-n3 | `D0[I]-P0\|n3` | −0.61 [−1.12, 1.50]; 블록 등가중 1.12 [0.08, 2.21] | 미결정 | 보조 (a) |
| F-F1a-n10 | `D0[I]-P0\|n10` | −2.91 [−3.58, −1.11]; 블록 등가중 −1.49 [−2.65, −0.21] | 우세 | 보조 (a). L1 결론의 한계 |
| F-F1a-n40 | `D0[I]-P0\|n40` | −0.54 [−1.13, 0.69]; 블록 등가중 0.30 [−0.65, 1.30] | 미결정 | 2지역 |
| F-F1-M3-* | `D0[I]-P0\|{n0,n10,n40}`, scope=MEAN3 | 12.25, 5.98, 4.27 | 열세 ×3 | 3지역 보조 열 |
| F-F1-RW-n10 | `D0[I]-P0\|n10`, scope=region, target=Russia_W\|x | −13.43 [−16.33, −10.10]; 블록 등가중 −13.08 [−16.77, −8.81] | 우세 | n = 10 우세의 출처 |
| F-F1-{Lena,Canada,Russia_E}-n10 | `D0[I]-P0\|n10`, scope=region, target ∈ {Lena, Canada, Russia_E}\|x | 0.51, 0.51, 0.76 | 미결정 ×3 | 나머지 세 지역 행 |
| F-F1-AK-n0 | `D0[I]-P0\|n0`, scope=region, target=Alaska\|x | 33.90 [19.31, 38.27]; 블록 등가중 20.87 [17.64, 24.12] | 열세 | |
| F-F1b-V | scope=verdict_aux, clause='(b) L1 형식 지역 수' | 'L1 형식 지역 수 1/4' | | 2개 미만이라 L1 예외 문구 없음 |
| F-posthoc-RW-D0-P1 | rmse_A(`D0[I]-P0\|n10`, Russia_W) − rmse_B(`R1[I]-P1\|n10`, Russia_W, test_id=LGF-F4) | D0[I] 29.54, P1 31.01, 차 −1.47 | 판정 아님 | 이번 점검(사후). 같은 LGF 조각의 점 추정이다 |
| F-F2-n0 | `R1[I]-R1[C]\|n0\|lam0.25`, test_id=LGF-F2 | −0.22 [−0.36, 0.36]; 블록 등가중 0.22 [−0.01, 0.46] | 동등 | |
| F-F2-n10 | `R1[I]-R1[C]\|n10\|lam0.25` | −0.39 [−0.53, 0.10]; 블록 등가중 −0.01 [−0.23, 0.22] | 미결정 | 보조 한계(1.0 cm) 동등 |
| F-F2-n40 | `R1[I]-R1[C]\|n40\|lam0.25` | −0.27 [−0.45, 0.01]; 블록 등가중 −0.29 [−0.50, −0.09] | 미결정 | 2지역, 보조 한계 동등 |
| F-F2-n160 | `R1[I]-R1[C]\|n160\|lam0.25` | −0.55 [−0.79, −0.19]; 블록 등가중 −0.54 [−0.77, −0.30] | 우세 | 2지역, holm_p 0.03. 추가 n 행 n = 320 도 우세(2.7 절 S-ctx-F2-n320) |
| F-F2-nall | `R1[I]-R1[C]\|nall\|lam0.25` | −0.48 [−0.63, −0.02]; 블록 등가중 −0.21 [−0.43, 0.01] | 미결정 | 보조 한계 동등 |
| F-F2-aux-* | `D0[I]-D0[C]\|n10`, `\|nall`; `R0[I]-R0[C]\|n10`, `\|nall` | −4.09, −4.08; −0.96, −1.01 | 우세 ×4 | 보조 |
| F-F2-V | scope=verdict_aux, test_id=LGF-F2 | '부분(지역 2/4): … n=160 TabICL 우세(Δ -0.55 cm)' | | |
| F-F3-D0-n0 | `D0[I]-D0[T]\|n0`, test_id=LGF-F3 | −1.50 [−1.98, −0.62]; 블록 등가중 −0.57 [−1.08, −0.05] | 우세 | 레나·러시아 W 2지역, holm_p 0.00 |
| F-F3-R1-* | `R1[I]-R1[T]\|{n0,n10,nall}\|lam0.25` | 0.07, −0.28, −0.10 | 동등 ×3 | 2지역 |
| F-F3-R1-lam1-n0 | `R1[I]-R1[T]\|n0\|lam1.0` | 0.84 [0.14, 1.44]; 블록 등가중 1.13 [0.53, 1.69] | 열세 | |
| F-F3-V | scope=verdict_aux, test_id=LGF-F3 | '부분(대비 4/6, 지역 2/4): … D0 n=0 TabICL 우세(Δ -1.50 cm)' | | 입력 처리 차이(결측 대치, 열 종류) 포함 |
| F-F4-n3 | `R1[I]-P1\|n3`, test_id=LGF-F4 | −0.42 [−0.59, 0.01]; 블록 등가중 −0.15 [−0.40, 0.11] | 미결정 | |
| F-F4-n10 | `R1[I]-P1\|n10` | −0.55 [−0.71, −0.21]; 블록 등가중 −0.36 [−0.63, −0.10] | 우세 | holm_p 0.02 |
| F-F4-n40 | `R1[I]-P1\|n40` | −0.79 [−1.05, −0.53]; 블록 등가중 −1.07 [−1.35, −0.81] | 우세 | 2지역, P1* 대비 없음 |
| F-F4-n160 | `R1[I]-P1\|n160` | −1.18 [−1.53, −0.79]; 블록 등가중 −1.40 [−1.71, −1.11] | 우세 | 2지역, P1* 대비 없음 |
| F-F4-n320 | `R1[I]-P1\|n320` | −1.43 [−1.85, −0.97]; 블록 등가중 −1.59 [−1.92, −1.29] | 우세 | 2지역 |
| F-F4-nall | `R1[I]-P1\|nall` | −1.00 [−1.23, −0.63]; 블록 등가중 −0.87 [−1.15, −0.60] | 우세 | |
| F-F4-P0-all | `R1[I]-P0\|all` | −3.24 [−3.99, −2.25]; 블록 등가중 −3.01 [−3.99, −1.94] | 우세 | |
| F-F4-M3-* | `R1[I]-P1\|n10`, `\|n40`, `R1[I]-P0\|all`, scope=MEAN3 | −0.02(동등), −0.32(우세, 크기 0.5 cm 미만), 0.46(미결정) | | 3지역 보조 열 |
| F-F4-V1, V2 | scope=verdict_aux, clause ∈ {순가치, 전량 R1[I] − P0(L35 형식)} | '희소 라벨에서도 TabICL 잔차의 순가치가 있다(우세인 최소 n: 4지역 평균 n=10, 2–3지역 평균 n=40)'; '라벨 전량에서 TabICL 잔차는 원천 계수 물리식을 넘는다' | | |
| F-F5 | `R0[I]-P0\|n0`, test_id=LGF-F5 | −0.27 [−0.48, 0.25]; 블록 등가중 0.01 [−0.28, 0.30] | 동등 | 병기 R0[C] −0.05(동등, 재현(비맹검, M1·a2)) |
| F-F5-M3 | `R0[I]-P0\|n0`, scope=MEAN3 | 0.64 [0.17, 1.12]; 블록 등가중 0.71 [0.25, 1.18] | 열세 | 3지역 보조 열 |
| F-F6-R1-* | `R1@full[I]-R1[I]\|{n0,n10,nall}\|lam0.25`, test_id=LGF-F6 | −0.03(동등), −0.03(동등), 0.07 [0.01, 0.13](열세) | | 전량 열세는 크기 0.5 cm 미만, holm_p 0.37 |
| F-F6-D0-nall | `D0@full[I]-D0[I]\|nall` | 0.35 [0.10, 0.64]; 블록 등가중 0.41 [0.13, 0.69] | 열세 | 크기 0.5 cm 미만, holm_p 0.11 |
| F-fam-1, -2 | test_id=LGF-family, scope=verdict_aux | '표형 파운데이션 모델 2종(TabPFN v2, TabICL v2)을 학습기로 써도 직접 ML 은 라벨 0 전이에서 원천 계수 물리식을 넘지 못한다'; '학습기 동등성은 모델별로 쓴다(F2 와 L32 가 모두 동등이 아니다)' | | 계열 문장 규칙(LGF 3.6) |

### 2.4 조정 판별 신경망(LGF-N, 원천 N, 판정 절 LGF 10.3, 가설 4.5)

조정 절차: 원천 지역 하나 제외 교차검증, 무작위 탐색 32회, 2단계 seed 1·2 재평가, 기본값 우선 규칙(LGF 10.3). 학습기 이름: mlp = MLP, tabm = 다중 헤드 MLP(LG 의 `tabm`, 논문 TabM 과 다르다), ftt = 축소 FT-Transformer, realmlp = RealMLP.

| ID | 필터 | 값(cm) | 판정어 | 비고 |
|---|---|---|---|---|
| N-N1-mlp | test_id=LGF-N1, scope=MEAN, contrast=`D0[mlp*]-P0\|n0` | 2.45 [1.62, 4.11]; 블록 등가중 2.05 [0.95, 3.19] | 열세 | 확인적, holm_p 0.00 |
| N-N1-tabm | contrast=`D0[tabm*]-P0\|n0` | 2.44 [1.58, 4.73]; 블록 등가중 3.94 [2.68, 5.29] | 열세 | 확인적 |
| N-N1-ftt | contrast=`D0[ftt*]-P0\|n0` | 4.13 [3.28, 5.63]; 블록 등가중 4.58 [3.44, 5.76] | 열세 | 확인적 |
| N-N1-realmlp | contrast=`D0[realmlp*]-P0\|n0` | 5.21 [4.07, 7.17]; 블록 등가중 5.91 [4.60, 7.25] | 열세 | 확인적 |
| N-N1aux-* | contrast=`D0[{mlp,tabm,ftt,realmlp}]-P0\|n0`(로컬 기본판) | 0.94, 1.11, 4.95, 4.70 | 열세 ×4 | N1 보조 (a) |
| N-N1aux-catboost_lo | contrast=`D0[catboost_lo]-P0\|n0`(로컬) | 2.25 [1.10, 3.40]; 블록 등가중 1.62 [0.50, 2.66] | 열세 | 병기 |
| N-N1-V | scope=verdict | '지지(물리식보다 오차가 크거나 구별되지 않음)(mlp, tabm, ftt, realmlp) … [Δ = 0(선택 결과) 지역(종류 D): mlp 1/4, tabm 1/4, ftt 2/4, realmlp 2/4]' | | 선택 결과가 기본값과 같은 지역 수를 함께 적는다 |
| N-N2-V-mlp | test_id=LGF-N2, scope=verdict_aux, learner=mlp | '원천 교차검증 조정이 전이 오차를 키웠다(D0 n=0 +1.51, n=10 +2.96, 전량 +1.10)(정밀도 미달 대비 6/9)' | | |
| N-N2-V-tabm | learner=tabm | '원천 교차검증 조정이 전이 오차를 키웠다(R1 n=0 λ0.25 +0.24, D0 n=0 +1.33, n=10 +2.56, 전량 +1.67, R1 n=0 λ1 +0.75)(정밀도 미달 대비 4/9)' | | |
| N-N2-V-ftt | learner=ftt | '조정판의 오차가 작다(R1 전량 λ0.25 −0.19, R1 전량 λ1 −0.76)(정밀도 미달 대비 6/9)' | | λ 0.25 는 크기 0.5 cm 미만 |
| N-N2-V-realmlp | learner=realmlp | '정밀도 미달(동등성 판정 불가, 대비 6/9)' | | |
| N-N2-V-all | clause=전체 | '전체 문장 조건 미충족(강건 문장 규칙을 채운 학습기: 없음)' | | '신경망 결론은 초모수 선택에 강건' 금지 |
| N-N2s-V-* | test_id=LGF-N2s, scope=verdict_aux | mlp '강건', tabm '약화: R1−P1 전량', ftt '약화: R1−P1 전량, R1−CBT 전량', realmlp '강건' | | '의존' 없음 |
| N-N3-V-* | test_id=LGF-N3, scope=verdict_aux | mlp·tabm '차이를 확인하지 못함(정밀도 미달 대비 3/6)'; ftt '조정 신경망의 오차가 크다(R1 n=0 λ0.25 +0.61, n=10 +0.54 …)'; realmlp '조정 신경망의 오차가 크다(R1 n=10 λ0.25 +0.45, 전량 +0.28 …)' | | 비교 상대 CBT = `catboost_tuned_loc` |
| N-N4-rho | test_id=LGF-N4, contrast='원천 CV 이득(seed 1·2) 대 대상 Δ(l* − l, n = 0)의 순위 상관', 열 `spearman`, `n_points` | −0.32(점 32개) | 서술(검정 없음) | |

### 2.5 새 독립 지역(LGD, 주 판정 = 약관 확인분 판, 원천 D, 판정 절 LG 7.3, 가설 6B.5)

허락 기록 파일(`data/processed/lgd/lgd_license_permission.json`)이 없어 약관 확인분 판이 주 판정이다(LG 개정 15 (m)2; 2026-10-04 에 파일이 없음을 다시 확인했다). 풀: P4 = 주 4지역(Rescale 조각), PE1 = P4 + Russia_C(5지역, 얕은 레짐), PE2 = PE1 + Tibet(6지역). PE1·PE2 는 '교차 환경; (i) CatBoost 불통과' 표지가 있다(D-pool-PE1, D-pool-PE2).

**풀 판정(L1e, L4e, L8e)**

| ID | 필터 | 값 | 판정어 |
|---|---|---|---|
| D-L1e-V-P4, -PE1, -PE2 | test_id=L1e, pool ∈ {P4, PE1, PE2}, scope=verdict, 열 `verdict`, `stat` | 유의 개선 지역 0/4, 0/5, 1/6(Tibet_LGD: n 0, 3, 10, 40) | 지지 ×3 |
| D-L1e-compare | test_id=L1e, pool=P4·PE1·PE2, scope=compare | '4지역의 판정이 확장 풀(지역 6개)에서 유지된다' | |
| D-L4e-V-P4 | test_id=L4e, pool=P4, scope=verdict | '희소 라벨에서도 ML 순가치 있음'(최소 n 10) | |
| D-L4e-V-PE1 | pool=PE1 | '부분(지역 2/5): 희소 라벨에서 순가치 있음'(풀 전체 최소 n 없음, n ≤ 10 불성립) | |
| D-L4e-V-PE2 | pool=PE2 | '희소 라벨에서도 ML 순가치 있음'(최소 n 3), 표지 '수축 사전분포 오지정 의존' | |
| D-L4e-PE1-n10 | test_id=L4e, pool=PE1, item=R1-P1, scope=MEAN, n=10 | −0.13 [−0.43, 0.01]; 블록 등가중 −0.19 [−0.42, 0.00] | ns |
| D-L4e-PE2-n10 | pool=PE2, item=R1-P1, scope=MEAN, n=10 | −3.59 [−4.12, −3.13]; 블록 등가중 −2.80 [−3.43, −2.19] | improve(표지 '수축 사전분포 오지정 의존') |
| D-L4e-PE2-R1P2-n10 | pool=PE2, item=R1-P2, scope=MEAN, n=10 | 2.61 [0.33, 4.44]; 블록 등가중 −0.74 [−3.14, 1.73] | ns |
| D-L4e-pool-P4-10 | test_id=L4e, pool=P4, item='(a)1 풀 대비 R1-P1\|n10', scope=compare_pool | −0.18 [−0.52, −0.02]; 블록 등가중 −0.29 [−0.51, −0.08] | 우세 |
| D-L4e-pool-PE1-10 | pool=PE1, 같은 item | −0.13 [−0.42, 0.01]; 블록 등가중 −0.19 [−0.41, 0.01] | 동등 |
| D-L4e-pool-PE1-new-10 | pool=PE1-new(Russia_C), 같은 item | 0.06 [−0.40, 0.51]; 블록 등가중 0.20 [−0.39, 0.81] | 미결정 |
| D-L4e-pool-PE1-new--1 | pool=PE1-new, item='(a)1 풀 대비 R1-P1\|nall' | −1.40 [−2.95, −0.19]; 블록 등가중 −1.54 [−3.12, −0.10] | 우세 |
| D-L4e-compare | test_id=L4e, pool=P4·PE1·PE2, scope=compare | '4지역의 판정은 확장 풀에서 유지되지 않는다. PE1: L28 약화' | |
| D-L8e-P4 | test_id=L8e, pool=P4, item=R1-P0, scope=MEAN | −2.64 [−3.47, −1.90]; 블록 등가중 −2.64 [−3.53, −1.67] | improve |
| D-L8e-PE1 | pool=PE1 | −3.19 [−4.19, −2.22]; 블록 등가중 −2.62 [−3.71, −1.53] | improve |
| D-L8e-PE2 | pool=PE2 | −26.78 [−29.09, −24.21]; 블록 등가중 −23.13 [−26.03, −20.20] | improve(표지 '비맹검(라벨 통계 열람): 티베트 포함') |
| D-L8e-PE2-rel | pool=PE2, scope=MEAN_rel | −0.16 [−0.19, −0.13](상대 단위) | improve |
| D-L8e-compare | test_id=L8e, pool=P4·PE1·PE2, scope=compare | '4지역의 판정이 확장 풀(지역 6개)에서 유지된다' | |

**새 지역 행(L38, pool=새 지역, scope=region, 4분 판정)**

| ID | 필터(item, target, n) | 값(cm) | 판정어 | 비고(RMSE A, B 는 `rmse_A`, `rmse_B` 열) |
|---|---|---|---|---|
| D-L38-Tibet-D0-P0-n0 | (a) D0-P0, Tibet_LGD\|x, 0 | −59.98 [−62.71, −57.10]; 블록 등가중 −62.91 [−65.29, −60.32] | 우세 | P0 RMSE 244.51(`rmse_B`). 비맹검 표지 |
| D-L38-Tibet-D0-P0-n3, -n10, -n40 | (a) D0-P0, Tibet_LGD\|x, 3·10·40 | −130.18, −150.92, −157.59 | 우세 ×3 | L1 형식 유의 개선(심부 레짐) |
| D-L38-Tibet-R1-P0-n-1 | (b) R1-P0, Tibet_LGD\|x, 전량 | −144.73 [−157.42, −129.90]; 블록 등가중 −125.70 [−142.62, −108.64] | 우세 | |
| D-L38-Tibet-P1-P0-n10 | (c) P1-P0, Tibet_LGD\|x, 10 | −96.37 [−100.70, −91.92]; 블록 등가중 −97.75 [−102.91, −92.22] | 우세 | P1 RMSE 148.14 |
| D-L38-Tibet-R1-P1-n10 | (c) R1-P1, Tibet_LGD\|x, 10 | −20.90 [−23.40, −17.89]; 블록 등가중 −15.82 [−19.40, −12.33] | 우세 | 표지 '수축 사전분포 오지정 의존'. R1 127.23, P1 148.14 |
| D-L38-Tibet-R1-P2-n10 | (c) R1-P2, Tibet_LGD\|x, 10 | 19.11 [5.74, 30.14]; 블록 등가중 2.37 [−11.81, 16.77] | 미결정 | (a)10 병기. P2 RMSE 108.12. 셀 가중 CI 가 전부 0 초과(P2 가 더 정확)이고 블록 등가중 CI 만 0 을 포함한다. '구별되지 않는다' 로 쓰지 않는다 |
| D-L38-Tibet-R1-P3-n10 | (c) R1-P3, Tibet_LGD\|x, 10 | 16.65 [9.53, 22.48]; 블록 등가중 5.20 [−3.29, 13.58] | 미결정 | (a)10 병기. R1 − P2 와 같은 형태(셀 가중 CI 0 초과, 블록 등가중 CI 0 포함) |
| D-L38-Tibet-P2-P1-n10 | (c) P2-P1, Tibet_LGD\|x, 10 | −40.02 [−53.21, −24.07]; 블록 등가중 −18.19 [−35.84, −0.69] | 우세 | 표지 '분할 독립 가정 의존' |
| D-L38-RussiaC-D0-P0-n0, -n3, -n10, -n-1 | (a) D0-P0, Russia_C\|x, 0·3·10·전량 | 2.75, 2.44, 1.81, −4.58 | 열세, 열세, 미결정, 미결정 | P0 RMSE 32.12 |
| D-L38-RussiaC-R1-P0-n-1 | (b) R1-P0, Russia_C\|x, 전량 | −5.42 [−8.96, −1.64]; 블록 등가중 −2.55 [−6.83, 1.48] | 미결정 | 재현 실패 지역(L38 (b)). 셀 가중 CI 는 전부 0 미만이고 블록 등가중 CI 는 0 을 포함한다(두 가중이 엇갈린 미결정). 셀 가중 점 추정과 블록 등가중 CI 를 한 괄호에 묶지 않는다 |
| D-L38-RussiaC-P1-P0-n10 | (c) P1-P0, Russia_C\|x, 10 | −2.11 [−3.37, −0.61]; 블록 등가중 −0.70 [−2.24, 0.76] | 미결정 | |
| D-L38-RussiaC-R1-P1-n10 | (c) R1-P1, Russia_C\|x, 10 | 0.06 [−0.40, 0.51]; 블록 등가중 0.20 [−0.39, 0.81] | 미결정 | |
| D-L38-RussiaW-ref-* | pool=P4 참조, Russia_W\|x: (b) R1-P0 전량; (c) P1-P0 10; (c) R1-P1 10 | −13.98; −11.96; −0.56 [−1.21, −0.02], 블록 등가중 −0.24 [−0.84, 0.31] | 우세, 우세, 미결정 | LG 본 실행 CatBoost 조각. C2 축소의 근거 |
| D-L38-V | test_id=L38, pool=새 지역, scope=verdict | '어긋난 지역: Russia_C(shallow; (a) 충족, (b) 미결정); Tibet_LGD(deep; (a) 불충족, (b) 우세)' | | `stat` '(a) 충족 1/2, (b) 우세 1/2. 대칭 참조(P4): (a) 4/4, (b) 1/4' |
| D-L38-V-note | 같은 행, 열 `note` | '티베트 채점 셀의 원천 지지 밖 비율: 고도 1.000, √TDD 0.333 …; 티베트 결과는 NOVELTY-N7 문장의 근거로 쓰지 않는다' | | |

**민감도(L39–L42)**

| ID | 필터 | 값 | 판정어 |
|---|---|---|---|
| D-L39-V | test_id=L39, scope=verdict | '강건'(티베트 라벨 정의: GPR 대 지온 유도) | 강건 |
| D-L39-tempderived-R1P1-n10 | test_id=L39, item=R1-P1\|n10, target=Tibet_LGD~L39\|x | −20.92 [−22.59, −18.23]; 블록 등가중 −21.64 [−23.90, −19.20] | 우세(위치가 다른 라벨 집합) |
| D-L40-V-* | test_id=L40, scope=verdict, target ∈ {Russia_W~exp~lic, Russia_E~exp, Canada~exp~lic, Russia_E~expnokyt}, 열 `verdict`, `stat` | 네 확충판 모두 '약화'. 네 판 모두 R1-P0\|nall 과 P1-P0\|n10 은 '강건', 약화는 D0-P0 대비에서 나왔다 | 약화 ×4 |
| D-L41-V-* | test_id=L41, scope=verdict, 열 `verdict`, `stat`, `note`, `same_as`; 대상 셀 수는 원천 `tables/lgd_eligibility_v1.csv` 의 `n_cells`, `eligible` | 티베트: (a)(c)(h) 는 대상 셀 0개라 '변형 불가(적격 표에서 부적격)' 다. (b)(d)(e)(f)(g) 는 '강건' 이지만 모두 주 설정과 같은 실행 표(same_as, 132셀)라 note 가 '주 설정의 TMx 를 그대로 써서 모든 행이 주 설정과 같다' 고 적었다. 따라서 티베트에는 라벨 집합 민감도가 없다. 러시아 중부: (a) 변형 불가(12셀, 부적격), (h) 점 추정만, (b) '약화'(D0-P0\|n3 열세 → 미결정), (e) '약화'(D0-P0\|n10 미결정 → 열세), (d)(f) '강건'(다시 적합한 변형), (c)(g) '강건' 은 same_as 라 민감도가 아니다 | (b)(e)(d)(f) 만 실질 변형 |
| D-L42-V-* | test_id=L42, scope=verdict, 열 `verdict`, `stat` | Tibet '강건', Russia_C '약화'(D0-P0\|n0·n3 열세 → 미결정, R1-P1\|n10 미결정 → 우세), NAtlantic '행 없음(부적격)' | |
| D-elig-* | 원천 Delig, spec ∈ {NAtlantic_lic, Tibet_L42_lic, Russia_C_L42_lic}, 열 `n_cells, nb_union, min_nb_eval_used, eligible, regime` | NAtlantic 22셀, 블록 합집합 6, 부적격; Tibet 132셀, 34, 적격, deep; Russia_C 57셀, 13, 적격, shallow | |

**북대서양(전체 판, SI 전용, 원천 Dfull, 판정 절 LG 7.3 전체 판 열)**

| ID | 필터(test_id=L38, pool=새 지역, target=NAtlantic\|x) | 값(cm) | 판정어 | 비고 |
|---|---|---|---|---|
| Dfull-L38-NAtl-D0-P0-n0 | (a) D0-P0, 0 | −9.91 [−15.95, −2.85]; 블록 등가중 −6.15 [−14.53, 2.40] | 미결정 | 표지 '소수 블록'. P0 RMSE 29.80 |
| Dfull-L38-NAtl-D0-P0-n10 | (a) D0-P0, 10 | −9.15 [−14.87, −1.62]; 블록 등가중 −4.37 [−12.73, 4.15] | 미결정 | |
| Dfull-L38-NAtl-R1-P0-n-1 | (b) R1-P0, 전량 | −4.65 [−8.50, 1.42]; 블록 등가중 −0.42 [−6.40, 5.86] | 미결정 | |
| Dfull-L38-NAtl-P1-P0-n10, R1-P1-n10 | (c) P1-P0, R1-P1, 10 | −4.22; −0.89 | 미결정 ×2 | |
| Dfull-L1e-V-PE1 | test_id=L1e, pool=PE1, scope=verdict | 유의 개선 지역 0/6(NAtlantic 포함) | 지지 | |
| Dfull-L8e-V-PE1 | test_id=L8e, pool=PE1, scope=verdict | Δ −3.44 [−4.45, −2.14], 블록 등가중 [−3.54, −0.87], 지역 6/6 | 지지 | |
| Dfull-L4e-compare | test_id=L4e, pool=P4·PE1·PE2, scope=compare | '4지역의 판정은 확장 풀에서 유지되지 않는다'(R1-P1\|n10: P4 우세 → PE1 미결정) | | |

### 2.6 계산 환경 사이의 재현성

| ID | 원천과 필터 | 열 | 값 | 판정·규칙 | 판정 절 |
|---|---|---|---|---|---|
| R-C14 | XC14, `by_pair.local1_vs_rescale` | ml.n_keys, ml.max_cm, phys.n_keys, phys.max_cm, ml_n_over_eq | 학습 키 34,360개 최대 3.6e-14 cm; 물리식 키 1,030개 최대 3.2e-14 cm; 0.5 cm 초과 0; 단위 20개(레나·캐나다·CA-2·CA-3 x, base 축) | 보충 대조(계속 규칙 없음) | LG 개정 15 (t); EXECUTION_PLAN 10(12:57 C14) |
| R-gate-i | XI, `verdict`, `by_class`, `catboost_excl_v1r` | max_diff_cm | CatBoost 437키 최대 0.46 cm(허용 차 초과 11키, 모두 V1r); V1r 제외 398키 최대 1.4e-14 cm; ridge 83키 최대 0.0205 cm; 물리식 46키 최대 2.8e-14 cm | '물리식 통과, CatBoost·ridge 불통과' | LG 개정 14(점검 (i)); WRAPUP 8.4 |
| R-gate-iii | XIII, `by_pair` | ml.median_cm, ml.max_cm, ml_n_over_eq | FT-T 288키: 로컬 − Rescale 중앙값 0.059 cm, 최대 5.61 cm, 0.5 cm 초과 43키(14.9 %); 로컬 반복 차 최대 0.0; 물리식 60키 최대 8.9e-15 cm | 계속 규칙 '멈춤' | LG 개정 15 (u) |
| R-lgfn-cross | Ncross, 전 168행 | \|delta\| 의 학습기별 중앙값과 최댓값 | 중앙값 ftt 0.08, mlp 0.15, tabm 0.19, realmlp 0.32 cm; 최대 5.59 cm(`D0[realmlp]-D0[realmlp@lg]\|n0`, Canada\|x, 지역 행) | 교차 환경(보조), 판정에 쓰지 않음 | LGF 10.3 교차 환경 행 |
| R-wf9 | WF9, 전 74행 | ok, max_abs_dsse, n_common, n_count_mismatch, same_blocks | ok 74/74; 최대 \|ΔSSE\| 0.0; 공통 키 11,814; 셀 수 불일치 0; 채점 블록 같음 | 점검(판정 아님). WF9 hematite 대 WF6 elm | WF 5.3 재현 점검(열람 전) |
| R-lgt-gate | Tgate, 전 117행 | passed, max_diff, n_missing_ref | 통과 117/117; P0·P1 최대 차 3.2e-14 cm; 누락 0 | 통과 | LG 6C.8; 7.4 |
| R-lgf-gate | Fgate, 전 55행 | passed, status, n_sha_mismatch/n_sha_keys, p01_max_diff, cb_max_diff | 통과 29/55; 불통과 대상 CA-2, CA-3, Canada, Greenland, Russia_C, Russia_E(ctx_sha 불일치 비율 0.667); P0·P1 최대 차 2.1e-14 cm; 통과 단위 catboost_ctx 최대 차 0.0 | 불통과 단위의 TabPFN 키 제외 | LGF 10.1 |
| R-lgfn-gate | Ngate, 전 60행 | passed, max_diff | 통과 60/60; 물리식 키 최대 차 1.4e-14 cm | 통과 | LGF 10.1; DISPLAY_ITEMS 6절 10번 |
| R-lgd-repro-base | Drepro, scope=base | status, n_keys_catboost, max_diff_catboost, max_diff_phys | ok; CatBoost 264키 최대 1.42e-14 cm; 물리식 최대 7.1e-15 cm | 통과(주 판정 범위) | LG 7.3 원천; 개정 14 (c)2 |
| R-lgd-repro-all | Drepro, scope=all | 같음 + n_ml_over_tol, methods_ml_over_tol | CatBoost 408키 최대 0.98 cm; 허용 차 초과 30키(V1r) | 병기, 상태를 정하지 않는다 | 같음 |

### 2.7 2026-10-04 정정에서 더한 근거(`evidence_values.csv` 밖)

아래 값은 검증 지적 11건을 반영하면서 원천 표를 pandas·json 으로 직접 읽어 옮겼다(OMP_NUM_THREADS=1, 적합·재표집 없음). `extract_evidence.py` 에는 없으므로 재실행으로 다시 나오지 않는다. 원천은 모두 `tables/` 사본이 있다. 반올림은 2자리다.

| ID | 원천과 필터 | 열 | 값 | 판정어·표지 |
|---|---|---|---|---|
| S-ctx-T-R0-n10 | T, test_id=L32, scope=MEAN, contrast=`R0[T]-R0[C]\|n10` | delta, ci_lo, ci_hi, delta_blockeq, ci_lo_beq, ci_hi_beq, verdict4, blind | −0.66 [−0.79, −0.27]; 블록 등가중 −0.46 [−0.67, −0.23] | 우세, 맹검 |
| S-ctx-T-R0-nall | 같음, contrast=`R0[T]-R0[C]\|nall` | 같음 | −0.69 [−0.85, −0.29]; 블록 등가중 −0.49 [−0.70, −0.27] | 우세, 맹검 |
| S-ctx-F2-n320 | F, test_id=LGF-F2, scope=MEAN, contrast=`R1[I]-R1[C]\|n320\|lam0.25` | 같음 + role, pool_regions | −0.55 [−0.80, −0.17]; 블록 등가중 −0.53 [−0.78, −0.29] | 우세. role '보조(추가 n)', 레나·캐나다 2지역 |
| S-ctx-F6-full-nall | F, test_id=LGF-F6, scope=MEAN, contrast=`R1@full[I]-R1@full[C]\|nall\|lam0.25` | 같음 | −0.49 [−0.66, −0.01]; 블록 등가중 −0.27 [−0.52, −0.00] | 우세(크기 0.5 cm 미만, F6 verdict_aux 의 병기 문구) |
| S-M3-T-R0 | T, test_id=L34, scope=MEAN3, contrast=`R0[T]-P0\|n0` | 같음 | −0.13 [−0.31, 0.27]; 블록 등가중 +0.38 [0.10, 0.66] | 미결정(블록 등가중 CI 0 초과). 3지역 보조 열 |
| S-blind-T | T, test_id ∈ {L32, L33}, scope=MEAN, R1 행의 `blind`; 같은 test_id 의 scope=verdict_aux `note` | blind, note | n ∈ {3, 10, 40} False, n ∈ {0, 160, 320, 전량} True. note: '재현(비맹검): n ∈ {3, 10, 40} 은 b4 에서 방향을 보았다. n = 0, 160, 전량의 대비 행은 맹검이다'(L32) | L32 의 D0·R0 보조 행은 모든 n 에서 맹검 |
| S-blind-F | F, scope=MEAN, `prior_info`, `blind` | prior_info | C 행(D0[C]·D0@full[C] − P0, R1[C] − P1, R0[C] − P0): '재현(비맹검, M1·a2)'. F4 의 R1[T] − P1 n 3·10·40: '재현(비맹검, b4)' | LGF 10.2 표지 열과 같다 |
| S-blind-N1 | N, test_id=LGF-N1, scope=MEAN, 확인적 행 | prior_info, blind | 'M1(사전 정보: 기본 초모수 신경망의 방향)', blind True. 기본판·catboost_lo 병기 행은 '재현(비맹검, M1)'·'재현(비맹검, M1·a2 의 CatBoost 방향)' | LGF 10.3 표지 '비맹검 부분 포함(기본판의 방향)' |
| S-xenv-pool | Dpool, pool ∈ {PE1, PE2} | platforms, cross_env | 'local,rescale'; '교차 환경; (i) CatBoost 불통과'. P4 는 'rescale', NEW1·NEW2 는 'local' | |
| S-xenv-tests | D, test_id ∈ {L1e, L4e, L8e}, pool ∈ {PE1, PE2, P4·PE1·PE2, PE1-new} | cross_env | 지역 행, MEAN, MEAN_rel, verdict, compare, compare_pool 행 모두 '교차 환경; (i) CatBoost 불통과'. P4 행은 표지 없음. L38 verdict 행에도 같은 표지(대칭 참조 P4). L40 은 '교차 환경(보조); (i) CatBoost 불통과' | |
| S-L1e-PE2 | D, test_id=L1e, pool=PE2, scope=verdict | stat | '두 가중 CI 상한 < 0 인 지역 1/6(기각 기준 > ⌊N/4⌋ = 1)'; Tibet_LGD 유의 [0, 3, 10, 40], 나머지 5지역 유의 없음 | 지지 |
| S-N2-stat | N, test_id=LGF-N2, scope=verdict_aux, 학습기별 | stat | 점 추정이 음수인 대비: mlp R1 n 0 λ1 −0.27, 전량 λ1 −0.01(미결정); tabm R1 전량 λ0.25 −0.02(동등), n 10 λ1 −0.22(미결정); ftt R1 전량 λ0.25 −0.19(우세, 크기 0.5 cm 미만), 전량 λ1 −0.76(우세), D0 n 0 −0.82, n 10 −0.36, 전량 −0.16(미결정); realmlp R1 n 0 λ0.25 −0.06, 전량 λ0.25 −0.08(동등), D0 n 10 −0.23, 전량 −0.13, R1 전량 λ1 −0.34(미결정) | 우세는 ftt 의 두 대비뿐 |
| S-L41-note | D, test_id=L41, scope=verdict | verdict, note, same_as | Tibet (b)(d)(e)(f)(g): '강건', note '주 설정과 같은 실행 표(same_as): 주 설정의 TMx 를 그대로 써서 모든 행이 주 설정과 같다', same_as 'Tibet'. Tibet (a)(c)(h): '변형 불가(적격 표에서 부적격)'. Russia_C (c)(g): 같은 same_as note. Russia_C (a): 변형 불가, (h): '점 추정만(WRAPUP 7.3 (a)9)' | |
| S-L41-elig | `tables/lgd_eligibility_v1.csv`, spec ∈ {Tibet_L41a–h, Russia_C_L41a–h} | n_cells, nb_union, eligible | Tibet_L41a·c·h 0셀(부적격); Tibet_L41b·d·e·f·g 132셀, 블록 합집합 34(적격). Russia_C_L41a 12셀(부적격); b 35, c 57, d 30, e 51, f 57, g 57, h 30셀(적격) | |

---

## 3. 근거 실험

| 새 이름 | 옛 id | 내용 | 하네스 | 실행 환경 | 판정 기록 |
|---|---|---|---|---|---|
| T3_learners_foundation_and_nn | LGT | TabPFN v2 라벨 격자, L32–L37(27대상, 분할 5, 추출 5, seed 2) | `scripts/3_deep_learning/h43_tabpfn_label_grid.py` | 로컬 RTX 3090, tabpfn 8.0.7(v2 회귀 가중치) | LG 7.4(J4) |
| T3_learners_foundation_and_nn | LGF-F | TabICL v2, F1–F6(컨텍스트 10,000행과 원천 전체) | `scripts/3_deep_learning/h47_foundation_models.py` | 로컬 RTX 3090, `.venv_lgf`, tabicl 2.0.2 | LGF 10.2 |
| T3_learners_foundation_and_nn | LGF-N | 원천 교차검증 조정 신경망 4종, N1–N4 | `scripts/3_deep_learning/h48_nn_tuning.py` | 로컬 RTX 3090 | LGF 10.3 |
| T4_new_regions | LGD | 새 지역(Tibet, Russia_C, NAtlantic)과 확충판, L1e·L4e·L8e, L38–L42 | `scripts/3_deep_learning/h51_lgd_run.py`, `scripts/2_evaluation/h52_lgd_pool.py`, 자료 `scripts/1_data_prep/build_fidelity_base_v4.py`, `lgd_eligibility_v1.py` | 로컬 CPU(CatBoost, 물리식) | LG 7.3(J5) |
| T6_abstract_contrasts_and_map(+WRAPUP) | LGW 교차 환경 점검 (i)·(iii), C14 | 로컬 재적합과 Rescale 조각의 키별 대조 | `scripts/2_evaluation/lgw_xenv_gate_i.py`, `lgw_xenv_gate_iii.py` | 로컬 대 Rescale | LG 개정 14, 개정 15 (t)(u) |
| W7_gain_decomposition_grid | WF9(재현 점검) | WF9 조각 74개와 WF6 조각의 SSE 대조 | `scripts/3_deep_learning/h54_workflow.py` | Rescale hematite 대 elm | WF 5.3 |
| T1_transfer_label_grid | LG | P4 의 기준 조각(Rescale), LGT·LGF 교차 비교의 참조 | `scripts/3_deep_learning/h40_label_grid.py` | Rescale | LG 7.1 |
| H5_final_A2-D1(이력) | B4 | 09-26 TabPFN 잔차 시험(k-중심 선택, 3배 중복, 지역 id 입력) | `data/processed/h4/b4_*.csv` | 로컬 | LG 6C 머리말: 본문 TabPFN 수치는 LGT 를 쓰고 B4 는 규약 차이와 함께 보조 인용 |

돌지 않았거나 멈춘 등록 실행: LGF J12(등록한 시간 창 규칙, LGF 10.1), FT-T 로컬 이어 실행 G4–G6(점검 (iii) 계속 규칙 '멈춤', LG 개정 15 (u)). LGD 의 학습기 축(GPU)은 6B.6 등록 범위 밖이라 새 지역에서 TabPFN·TabICL·신경망은 돌지 않았다.

---

## 4. 그림

| 항목 | 파일 | 이 폴더 관련 패널·행 | 상태 |
|---|---|---|---|
| Fig 6(v2) | `outputs/figures/paper/v2/Fig6_label0_uncertainty.{pdf,svg,png}` | 패널 a 의 로컬 GPU 묶음: 'Direct ML (LGT)'(TabPFN v2, CatBoost 10,000행 컨텍스트), 'Direct ML (LGF-F)'(TabICL v2 실선 = 컨텍스트 10,000행, 파선 = 원천 전체; 같은 컨텍스트 CatBoost), 'Direct ML (LGF-N)'(조정판 실선, 로컬 기본판 파선 × MLP·TabM·FT-T·RealMLP). 플랫폼 묶음 사이의 차는 그리지 않는다 | v2 완료(커밋 ccdd7cd). 원천 `tables/Fig6_a.csv` 의 값이 2.2–2.4 절 값과 같다(Fig6a-lgt, -lgf_f, -lgf_n) |
| Fig 1(v2) | `outputs/figures/paper/v2/Fig1_problem.{pdf,svg,png}` | 패널 a: LGD 약관 확인분 판으로 더한 셀(티베트는 삽도), 북대서양은 'SI only' 직접 표지, 약관 미확인 셀은 그리지 않는다 | v2 완료. 스펙 `figures/figure_spec.json` 의 edition 'licence-verified (LG 7.3); NAtlantic SI only' |
| Table 1(v2) | `outputs/figures/paper/v2/Table1_data.{csv,md,tex,pdf}` | 'New region (LGD)' 3행: Russia_C_LGD(57행, 블록 13, PE1·PE2), Tibet_LGD(132행, 블록 34, PE2 심부), NAtlantic(41행, 블록 13, SI 전체 판만) | v2 완료. 원천 `tables/Table1_rows.csv`(Tab1-LGD) |
| Fig 2(v2) | `outputs/figures/paper/v2/Fig2_label_curve.*` | L34(a)·LGF-F1·LGF-N1 의 조건부 곡선 | 조건이 서지 않아 더하지 않았다(DISPLAY_ITEMS 2026-10-02 기록, 커밋 eaad89e) |
| 옛 FigS7 | `outputs/figures/paper/FigS7_tfm.{pdf,png}` | B4 TabPFN 잔차(옛 규약) | LGT 로 대체. SI 에서 쓰려면 규약 차이를 캡션에 적는다 |

재구성안(`docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md`) 반영 사항:
- 5절: Fig 6 은 '그대로(LGF 행 포함, v2 완료 ccdd7cd)' 다.
- 6절: 학습기 비교 L5·L26·L32·LGF-F2–F6·N2–N4 와 LGD 새 지역 세부를 SI 로 옮긴다.
- 4절 R2: 라벨 0 절의 근거 목록에 LGT L34(a), LGF-F1·N1(학습기 10종)이 들어간다.
- 새 지역 결과(L38 지역 행)를 보여 주는 SI 그림은 아직 없다. 제안: 지역 행 포리스트(티베트·러시아 중부 실선, 북대서양 전체 판 빗금, 대조로 P4 참조 행), 티베트는 축이 달라 별도 패널. SI 표 제안: 학습기 × n 의 4분 판정표(2.2–2.4 절).

---

## 5. 단서와 쓰지 않을 문장

### 5.1 단서

1. **TabICL v2 의 n = 10 예외는 한 지역에서 온다.** 4지역 평균 −2.91 cm 는 러시아 W 의 −13.43 cm(채점 셀 12–17개, `tables/Table1_rows.csv` 의 `eval_cells_x`)에서 왔다. 나머지 세 지역(레나 0.51, 캐나다 0.51, 러시아 E 0.76 cm)은 미결정이다. 같은 n 에서 러시아 W 의 P1 RMSE 는 31.01, TabICL 직접 RMSE 는 29.54 cm 다(이번 점검, 사후, 판정 아님). 즉 이 예외는 '직접 ML 이 라벨 10개로 원천 계수 편향을 바로잡은 경우' 이고 재보정과 비교한 확인적 대비는 없다. 3지역 보조 열(알래스카 포함)에서는 +5.98 cm 다.
2. **TabPFN 짝 대비는 2지역뿐이다.** LGF 와 LGT 조각의 컨텍스트 해시가 26/55 단위에서 2/3 키씩 어긋나 등록 규칙대로 그 단위의 TabPFN 키를 모두 뺐다(R-lgf-gate). F3 의 TabICL 대 TabPFN 비교는 레나·러시아 W 2지역이다. P0·P1 값은 어긋나지 않았다(최대 차 2.1e-14 cm).
3. **F3 의 차이는 모델만의 차이가 아니다.** TabICL 은 결측을 컨텍스트 평균으로 채우고 TabPFN 은 NaN 을 그대로 받는다(LGF 2.4). 결측 행 비율은 레나 13.1 %, 러시아 W 9.7 %, 러시아 E 10.0 % 로 기록되어 있다(LGF 2.4 의 기록값, 이 폴더에서 다시 계산하지 않았다 [미확인]).
4. **'CatBoost 와 동급' 은 TabPFN 에 쓰지 않는다.** L32 는 λ 0.25 에서 다섯 n 모두 동등이지만 λ 1.0 에서 다섯 n 모두 미결정이라 판정이 '차이를 확인하지 못함' 이다.
5. **TabPFN 직접 예측은 라벨 40·160 에서 자기 잔차보다 낫다**(L36, 2지역 −1.79, −2.45 cm). C3 의 '라벨 40–160 에서는 물리 입력 구조가 낫다' 와 같은 방향이나, 비교 상대가 물리 입력 구조가 아니라 직접 예측이다.
6. **조정이 전이를 돕지 않는다는 결론의 범위.** N2 는 MLP·다중 헤드 MLP 에서 '오차를 키웠다', FT-T 에서 '조정판의 오차가 작다'(전량 R1, λ 0.25 −0.19 cm(크기 0.5 cm 미만), λ 1.0 −0.76 cm), RealMLP 는 정밀도 미달이다. 강건 문장 규칙을 채운 학습기는 없다(N-N2-V-all). N2s 의 '핵심 판정 불변' 은 MLP·RealMLP 에만 쓴다.
7. **N4 순위상관 −0.32 는 검정이 없는 서술이다**(점 32개).
8. **새 지역의 일반화 범위.** 약관 확인분 판에서 얕은 레짐의 새 지역은 러시아 중부 하나이고, 그 지역에서 L38 (b)(전량 잔차 ML 이 P0 를 넘는다)는 재현되지 않았다(미결정). 러시아 중부 새 셀 51개 가운데 27개가 단일 조사(Mamontov Klyk)라는 기록이 있다(LG 6B.9 개정 13 기록, 이 폴더에서 다시 세지 않았다).
9. **티베트는 원천 지지 밖이다.** 채점 셀 가운데 원천 0.5–99.5 백분위 밖인 비율이 고도 1.000, √TDD 0.333 이다(L38 note). 대상 셀 라벨은 GPR 131, 인력 시추 1 이고(`tables/Table1_rows.csv` 의 `label_type`), 기록 단위 날짜가 없다(LG 6B.9). L41 의 (a)(c)(h) 변형은 대상 셀이 0개라 '변형 불가' 다. (b)(d)–(g) 는 주 설정과 같은 실행 표(same_as)라 '강건' 이 자동으로 나오고 민감도를 낼 수 없다(LG 개정 이력의 h52 구현 세부 'same_as 변형은 … 분류는 강건이 된다'). 그래서 티베트에는 실질적인 라벨 집합 민감도가 없고, PE2 결론에 이 한계를 병기한다(LG 6B.9 '계절 말 확인' 행, 개정 13). 티베트 P0 기반 대비는 결과 전에 예상한 방향이다(비맹검 표지).
10. **러시아 중부의 결론은 변형에 따라 약해진다.** L41 (b)(e) 와 L42 가 '약화' 다. L41 의 '강건' 가운데 (d)(f) 는 다시 적합한 변형이고 (c)(g) 는 same_as 라 민감도 근거가 아니다. L42(다른 새 지역을 원천에 더함)에서 D0 − P0(n 0·3)은 열세에서 미결정으로, R1 − P1(n = 10)은 미결정에서 우세로 바뀌었다(D-L42-V-Russia_C~L42~lic 의 `stat`).
11. **확충판(L40)은 네 판 모두 '약화' 다.** 확충판(로컬)과 본 실행 v3 값(Rescale)의 비교는 교차 환경 보조다.
12. **북대서양 전체 판은 직접 ML 의 점 추정이 P0 보다 작다**(n = 0 −9.91 cm). 블록 등가중 CI 가 0 을 포함해 미결정이고 '소수 블록' 표지가 있다. 약관 허락이 오면 주 판정 풀(PE1)에 들어가므로 L1e 문장에 영향이 있을 수 있다(6절 XF).
13. **재현성 값의 단위.** 3.6e-14 cm 와 5.61 cm 는 키별 셀 가중 RMSE 의 절대 차이고, 5.59 cm 는 지역 행 RMSE 차(Δ)다. WF9 재현 점검의 0 은 SSE 차다.
14. **3.6e-14 cm 의 범위.** C14 는 네 대상(레나, 캐나다, CA-2, CA-3, 모드 x)의 base 축 20단위다. 다른 대상의 CatBoost 동일성은 점검 (i)의 네 단위(V1r 을 뺀 CatBoost 398키, 1.4e-14 cm)와 LGD 재현 점검(러시아 W 분할 1, CatBoost 264키, 1.42e-14 cm)으로만 확인했다.

### 5.2 쓰지 않을 문장

| 문장 | 사유 |
|---|---|
| '표형 파운데이션 모델은 물리식을 넘지 못한다'(계열 일반화) | 시험한 두 모델(TabPFN v2, TabICL v2)로 한정한다(LGF 9.2) |
| 'TabICL 은 라벨 0 에서 물리식을 넘지 못한다'(조건 없이) | 두 컨텍스트 조건(상한 10,000행, 원천 전체)을 밝힌다(LGF 9.2) |
| '어떤 학습기도 라벨이 적을 때 물리식을 넘지 못한다' | TabICL v2 직접 ML 의 n = 10 예외(−2.91 cm, 러시아 W)가 있다 |
| 'TabICL 과 TabPFN 은 동급이다', 'TabPFN 은 CatBoost 와 동급이다' | F3 은 부분 판정(2지역), L32 는 '차이를 확인하지 못함' 이다(LGF 9.2, NOVELTY 6절) |
| '신경망을 충분히 조정했다' | '원천 지역 하나 제외 교차검증으로 고른 설정(무작위 탐색 32회, 2단계 seed 1·2 재평가, 기본값 우선 규칙)' 으로 쓴다(LGF 9.2, 10.3) |
| '신경망의 결론은 초모수 선택에 강건하다' | 강건 문장 규칙을 채운 학습기가 없다(N2 전체 행) |
| 'TabM', 'FT-Transformer' 를 논문 모델과 같은 것으로 쓰기 | LG 의 `tabm` 은 다중 헤드 MLP, `ftt` 는 축소판이다(LGF 9.2) |
| '조정하면 신경망이 CatBoost 를 넘는다', '조정 신경망과 CatBoost 는 동등하다' | N3 에서 우세가 없고 동등도 아니다(LGF 9.2) |
| 'CatBoost 를 넘는 학습기는 없었다'(예외 없이, 범위 없이) | 판정 문장의 대비(R1, λ 0.25)로 범위를 밝히고 F2 의 n = 160 예외(−0.55 cm, 같은 컨텍스트 CatBoost 대비, 2지역)를 병기한다. 보조 대비에서는 D0[T]·D0[I]·R0[T]·R0[I] 와 R1@full[I] 가 같은 컨텍스트 CatBoost 보다 0.49–4.09 cm 작았다(1.2 절, 2.7 절) |
| LGF·LGT 곡선과 LG(Rescale) 곡선의 차를 학습기 차로 쓰기 | 같은 플랫폼 규칙(LGF 2.2, LG 개정 14 (5)) |
| 'TabPFN·TabICL 잔차가 라벨 40–160 에서 재보정 물리식을 넘는다' | P1* 대비가 없다(LG 7.4). P1 대비 보조 결과로만 쓴다 |
| '물리 잔차 ML 은 안전하다', '라벨 0 에서 잔차는 손해가 없다' | 등록된 비열등 판정이 없다. L34(b)·F5 는 '구별되지 않는다(한계 0.5 cm)' 로만 쓴다(QA 공통 축소) |
| '티베트에서 ML 이 재보정 물리식보다 21 cm 낫다' | 비교 상대가 수축 재보정이고 '수축 사전분포 오지정 의존' 표지가 있다. 현지 최소제곱(P2)보다 낫지 않았다(R1 − P2 +19.11 cm [5.74, 30.14]; 블록 등가중 +2.37 [−11.81, 16.77], 미결정. 셀 가중으로는 P2 가 더 정확하다) |
| '티베트에서 잔차 ML 과 현지 최소제곱(P2)은 구별되지 않았다', 'indistinguishable from P2' | 동등 판정처럼 읽힌다. 셀 가중 CI 가 전부 0 초과라 '낫지 않았다(미결정)' 로 쓴다 |
| '러시아 중부에서 라벨 전량의 잔차 ML 과 P0 는 구별되지 않았다' | '구별되지 않는다' 는 이 문서에서 동등(한계 0.5 cm)에 쓰는 표현이다. 이 대비는 셀 가중 CI [−8.96, −1.64] 가 0 미만, 블록 등가중 CI [−6.83, 1.48] 이 0 포함인 미결정이다 |
| '두 새 지역에서 라벨 0–40 의 직접 ML 이 P0 를 넘지 못한다는 판정이 유지되었다'(티베트 포함) | 티베트에서는 직접 ML 이 n 0·3·10·40 모두 P0 보다 유의하게 작았다. PE2 판정이 '지지' 인 것은 유의 개선 지역 1/6 이 기각 기준 이하이기 때문이다. 검증 문장은 러시아 중부를 근거로 쓴다(LG 7.3 원고 문장) |
| '티베트 결과는 라벨 집합 변형에 강건했다(L41)' | 티베트 L41 의 '강건' 은 모두 same_as 변형이고 나머지는 변형 불가다(5.1 절 9번) |
| '모든 대비를 한 플랫폼 안에서 닫았다' | 확장 풀(PE1·PE2)의 층화 평균은 로컬과 Rescale 조각을 합친 교차 환경 값이다. '지역별 대비는 한 플랫폼 안에서 계산했다' 로 쓴다 |
| '조정으로 오차가 준 것은 FT-T 전량 잔차뿐이다'(기준 없이) | 점 추정이 음수인 미결정 대비가 더 있다. '4분 판정에서 우세인 대비는 … 뿐' 으로 기준을 밝힌다 |
| '독립 지역 6–7곳에서 결론이 재현되었다' | 약관 확인분 판의 새 지역은 두 곳(얕은 1, 심부 1)이고 L38 판정은 '어긋난 지역: Russia_C, Tibet_LGD' 다. 북대서양은 SI 전체 판뿐이다 |
| '희소 라벨 순가치(L4)가 새 지역에서도 유지된다' | 확장 풀 PE1 에서 유지되지 않는다(LG 7.3 원고 문장) |
| '티베트 결과는 계수 지도·라벨 0 우회 경로(NOVELTY N7)를 지지한다' | L38 note 가 금지한다 |
| 'CatBoost 결과는 어느 환경에서도 비트 단위로 같다'(무조건) | V1r 은 최대 0.46 cm(점검 (i)), 0.98 cm(LGD 병기 범위) 달랐다. 기준 방법 키로 한정한다 |
| '신경망 결과는 재현되지 않는다' | 한 플랫폼 안의 재적합 차는 0 이다(점검 (iii)). '플랫폼 사이에서 달랐다' 로 쓴다 |

---

## 6. 이 주장을 바꿀 수 있는 예정 실험(X 묶음)

X 묶음의 설계는 `docs/research/2026-10-04/` 의 문서에 있다. 2026-10-04 13:39(KST) 확인 시 이 폴더에는 문서 9건이 있었다(파일 수정 시각 13:08–13:33). X 묶음을 다루는 문서는 6건이다: `harness_implementation_plan.md`(4.1–4.10, XA–XJ 하네스 계획), `scirep_format_and_drafts.md`(4.9 절 'XA–XJ 의 원고 자리' 표), `placement_and_workflow_algorithm.md`(5절 XC 등록안, 6절 XD 등록안), `open_regions.md`(XF·XD·XJ 의 자료 조사), `alt_products.md`(XG·XB 후보 조사), `within_grid_inputs.md`(5절 XE 등록안). 나머지 3건(`scirep_manuscript_exemplars.md`, `figure_slide_standards.md`, `alt_figure_exemplars.md`)은 원고·그림 형식 문서다. 이 README 의 13:17 판은 '문서를 찾지 못했다' 고 적었으나 그 시각 전에 위 6건이 이미 있었다. 아래 표는 그 6건과 대조해 고친 것이다. 문서들은 모두 등록 전 설계안이고 실행 결과가 아니다. '영향' 칸의 [추론] 은 설계에서 이 폴더의 판정으로 이어지는 경로를 이 폴더가 추론한 것이다. 사용자는 2026-10-04 에 비어 있는 로컬 GPU 전부와 Rescale 사용을 허락했다. 신경망·파운데이션 모델이 들어가는 대비는 한 플랫폼 안에서만 만든다(2.6 절 근거).

| 새 이름 | 설계 문서의 범위(원문 위치) | 이 폴더에 주는 영향 | 바뀔 수 있는 문장 |
|---|---|---|---|
| XF_new_regions | 하네스 계획 4.6: 새 지역에서 핵심 대비(LG L1·L4·L8 형식, WF4 W·진단, XB, XC)를 다시 계산한다. 독립성 규칙은 기존 대상 셀에서 100 km 초과, 약관 확인분만 주 판정이다. `open_regions.md` 1·6절: LGD 적격 규칙으로 1–2일 안에 더할 수 있는 새 독립 macro 지역은 0개로 추정했다. 조건부 후보는 북대서양 약관 확인분 판의 적격화 0–1개다(현재 채점 블록 합집합 6 < 8. CC BY 4.0 자료 Stordalen(Zenodo 10420396)과 Adventdalen 2023(Zenodo 11187360)이 새 채점 블록 2개를 줘야 8 에 닿는다). 약관 회신(calm_web_subsites, cusp_v1_1, GGD353, Walker 보고서 값)이 오면 북대서양은 공개 자료 추가 없이 적격이다(전체 판 합집합 10) | (1) 북대서양이 적격이 되어 PE1 에 들어가면 PE1 의 L1e 분모가 6 이 된다. 전체 판 PE1(6지역)은 '지지'(Dfull-L1e-V-PE1)였으나 북대서양 직접 ML 의 점 추정이 P0 보다 작았다(Dfull-L38-NAtl-D0-P0-n0, 미결정). (2) 얕은 레짐 새 지역이 늘면 'L4 가 확장 풀에서 유지되지 않는다', 'L38 재현 실패' 문장의 분모가 바뀐다. 문서 추정으로는 1–2일 안에 새 macro 지역이 늘지 않는다. (3) 설계에는 새 지역에서 TabPFN·TabICL·조정 신경망을 돌리는 arm 이 없다. 학습기 결론(라벨 0 열세, TabICL n = 10 예외)을 새 지역에서 시험하려면 별도 등록이 필요하고, LGT·LGF 조각과 짝을 지으려면 같은 로컬 환경에서 돌려야 한다 [추론] | 1.3 절 새 지역 문장, 5.2 절 '독립 지역 n곳' 문장 |
| XJ_tempderived_aux_labels | 하네스 계획 4.10: (1) 원천 쪽에 지온 유도 행(F4_calm_temp 68행, 이 가운데 몽골·중앙아시아 46; F3_ext_temp 39행(Tibet_LGD); F2_gtnp_env 37행)을 가중 w ∈ {0, 0.1, 0.3, 1} 로 더하고 대상은 주 4지역, n {0, 10, 40} 이다. (2) 대상 쪽은 Tibet_LGD 의 지온 유도 39행을 계수 보정의 보조 라벨로 쓰고 채점은 직접 라벨 셀만 한다. 확인적 가설 없이 서술이고 우선순위가 가장 낮다. `open_regions.md` 5절: 보유 지역 밖의 장기 관측은 지온 유도이고 LGD 는 이를 확인적 풀에서 뺐다(6B.1) | 설계는 새 대상 지역을 만들지 않는다. 몽골 등은 원천 쪽 보조 행으로만 들어가므로 독립 지역 수는 늘지 않는다. 티베트 대상 쪽 결과는 서술이고, 정의 차이(티베트 2 km 이내 30쌍의 GPR − 지온 유도 중앙값 −40 cm, LG 6B.7 기록값 [미확인: 이 폴더에서 다시 계산하지 않음])의 단서를 단다. L39(지온 유도 라벨로 바꾼 변형)는 이미 '강건' 이었다 | 없음(서술 추가만) |
| XA_c2_gain_decomposition | 하네스 계획 4.1, Sci Rep 문서 4.9: 재보정 몫(P0 − P1)과 재보정을 넘어선 ML 몫(P1 − R1, P1 − W)을 나누고 계수 오차와의 순위상관에 CI 를 붙인다. 새 적합 없는 사후 분석이고 판정어 없이 수치만 쓴다. 입력은 WF4 조각(30 (대상, 모드))이며 하네스 계획 4.3 의 WF4 unit.json 기록에 Tibet_LGD 와 Russia_C~lgd 가 있다. 이미 열람한 값: ρ −0.08(p 0.666), ρ(재보정 이득, P1 − R1) 0.07(p 0.715) | 분해 단계에 P2 는 없다. 티베트의 P2 대비는 이미 L38 (c)에 있다(R1 − P2 셀 가중 +19.11 [5.74, 30.14], 셀 가중으로는 P2 가 더 정확). 따라서 C2 표의 티베트 행은 XA 결과와 관계없이 'ML 이 재보정을 넘는 예' 로 쓰지 않는다(1.2 절). XA 는 이 판단을 바꾸지 않고 순위상관의 CI 를 더한다 [추론] | 1.1 절 52행(C2 표) |
| XB_multisource_stacking | 하네스 계획 3.3·4.2, `alt_products.md` 8절: 후보는 물리식·제품 앵커다. 하네스 계획은 P1, Kudryavtsev 보정, 토양형 Stefan, 토양 도일 Stefan, CCI 원값·선형 보정, Stefan·CCI 평균을 계산 가능 후보로 들었다. `alt_products.md` 는 누설 없는 기본 후보로 P1, P*(연도 정합 도일), 보정 Kudryavtsev, 토양형 Stefan, 선형 보정 CCI v4, Stefan·CCI 평균, CCI v5 연도 정합(민감도)을 들고 확장 후보로 Wei 2026(마스크 통과 대상만)과 Yi·Kimball 2018(알래스카만)을 두었다. 단체 제약 최소제곱으로 가중하고 P1 쪽 수축 강도 γ 를 2단 블록 교차검증으로 고른다. 전이 범위 xb_t 에 LGD 약관 확인분 3대상과 캐나다 확충 약관 확인분이 들어간다. CatBoost CPU 와 물리식만 쓴다. 위험 항목: P* 는 v4 새 셀에 없다 | 후보에 P2·P3·TabICL 잔차는 없다(이 README 13:17 판의 기술은 설계와 달랐다). γ 를 교차검증으로 고르므로 티베트에서 수축 사전분포 오지정이 줄어드는지 볼 수 있다 [추론]. 다만 새 지역에는 P* 가 없어 후보 집합이 주 4지역과 다르다 | 1.2 절 티베트 행 |
| XC_workflow_end_to_end | `placement_and_workflow_algorithm.md` 5절(확인적 등록안), 하네스 계획 4.3: 배치(Algorithm P 또는 H-prop) → 라벨 10개 R1(λ 0.25)과 편향 진단 → n 40·160 규칙 W(후보 P0, P1, P2, R1@0.25, R1@1.0, R2@0.25, D1. 하네스 계획의 W+ 는 Stack, StackR 를 더한다). 주 계열 F1 = 레나, 캐나다, 러시아 W, 러시아 E, 알래스카(x), Tibet_LGD, Russia_C~lgd. 분할 seed 6–15, Rescale CPU 한 작업 안 | 후보에 TabICL·신경망은 없으므로 '워크플로는 CatBoost 와 해석식으로 구성한다' 문장은 XC 로 바뀌지 않는다. F1 층화 평균에 티베트가 들어가므로 n 10·40 의 cm 단위 평균을 티베트가 지배할 수 있다(L8e PE2 −26.78 cm 와 같은 형태) [추론]. 러시아 중부는 A 풀 크기 때문에 n = 10 만 들어간다(하네스 계획 4.3 의 A 풀 26–35셀) | 1.1 절 117행(바뀌지 않음), 1.3 절 새 지역 문장(XC 결과 병기 여부) |
| XD_placement_policy | 세 문서의 검증 단위가 다르다. 하네스 계획 4.4: 지역 내 과제(알래스카, 레나, 캐나다, 하위 지역)로 학습하고 거시 지역 하나 제외(3개)로 검증한다. `open_regions.md` 7절: AL-1–AL-6 으로 개발하고 LE-1, LE-2, CA-2, CA-3, Tibet_LGD 5과제에서 한 번 시험하며(러시아 중부는 n = 10 까지), 5과제 모두 같은 방향일 때만 단측 부호 검정 p = 0.031 이다. `placement_and_workflow_algorithm.md` 6.3: 계열 하나 제외(알래스카, 캐나다, 레나, 러시아 W, 러시아 E, 티베트, 러시아 C, XF 계열, 약 7개)다. 학습형 정책은 SI 탐색 분석이고 DeepSets 는 로컬 GPU, 그 뒤 적합은 Rescale CPU 다 | 이 폴더의 판정 문장은 바뀌지 않는다. 지역 수 단서(약관 확인분 판의 독립 지역은 주 4지역과 새 지역 2곳, D-pool-PE2)와 티베트를 원천 지지 밖으로 따로 표시하는 단서(5.1 절 9번)를 공유한다. 알래스카 하위 지역(AL-1–AL-6)은 서로 독립이 아니다(Table 1 'not independent') | 없음 |
| XG_product_comparison | `alt_products.md` 4.4·6·7.6절: 티베트는 CCI v4·v5(CCI 스스로 티베트 지층을 모수화하지 않았다고 밝혔다)와 Wei 2026 으로 비교할 수 있고, 등록 대비 가운데 티베트·러시아 중부는 XG-6(대상별 서술)이다. 확인적 대비 XG-1–3 은 Wei(레나·캐나다, 블록 마스크 5 km)와 CCI v5(주 4지역)다. 티베트 새 셀 132개 가운데 Wei 학습 지점 5 km 안은 1개, 25 km 안은 14개, 중앙 거리 173.5 km 다. Liu·Ran 의 티베트 문헌 자료와의 중복은 [미확인]이다. LG 6B.1 은 Wei 2026 의 학습 표가 전부 CALM 지점이고 티베트는 3지점이며 v3 와 겹친다고 기록했다 | 티베트의 라벨 0 기준선 문장에 제품 비교가 서술로 붙을 수 있다. 판정어는 붙지 않는다(XG-6 서술) [추론]. 러시아 중부 새 셀은 Wei·CALM 과 5·25 km 안 중복이 0개라 러시아에서 CALM 학습 제품과 비교할 수 있는 유일한 셀 집합이다(같은 문서 4.4절) | 2.5 절 티베트 행의 비교 상대(서술) |
| XI_climate_extrapolation_retest | 하네스 계획 4.9·1절 (i): 풀은 알래스카(v3)와 캐나다 확충 약관 확인분(v4, 787셀, 58블록. warm W 11블록)이고 warm_trim 변형을 함께 등록한다. 티베트는 들어가지 않는다 | 이 폴더의 티베트 서술과 관계가 없다. 캐나다 확충 약관 확인분 표는 L40 의 Canada~exp~lic 와 같은 자료판이므로 L40 '약화' 서술과 함께 읽는다 [추론] | 없음 |
| XE_hires_covariates | `within_grid_inputs.md` 5절, 하네스 계획 4.5: 알래스카·레나·캐나다 지역 내(모드 r)만 시험한다. 고해상도 입력이 원천 지역에 없으면 전이 arm 은 하지 않는다. CatBoost·물리식만, Rescale CPU 한 플랫폼 | 이 폴더의 전이 판정에는 직접 영향이 없다. 새 입력을 전이 입력(x25)에 넣는 경우에만 LGT·LGF·LGD 를 다시 돌려야 기존 판정과 비교할 수 있다 | 없음 |
| XH_validation_ladder | 하네스 계획 4.8(3.4절): 무작위·지점·블록·kNNDM·지역 홀드아웃 단의 오차와 CI | 이 폴더의 판정은 바꾸지 않는다 | 없음 |

---

## 7. 파일 목록

- `README.md`: 이 문서.
- `extract_evidence.py`: 원천 표에서 수치를 읽어 `evidence_values.csv` 를 만든다(쓰기는 이 폴더에만 한다).
- `evidence_values.csv`: 213행. 열은 id, source, source_path, row_filter, column, value(2자리 반올림 문자열), value_raw(원값), verdict, verdict_section, note.
- `tables/MANIFEST.csv`: 사본 19건의 원본 경로, 사본 경로, sha256(원본과 같음을 복사 때 확인), 행 수(CSV 는 pandas 행 수, JSON 은 NA), 바이트.
- 사본 19건: `lgt_tests.csv`, `lgf_tests.csv`, `lgfn_tests.csv`, `lgd_tests_lic.csv`, `lgd_tests.csv`(전체 판, SI 전용), `lgd_pool_lic.csv`, `lgd_eligibility_lic.csv`, `lgd_eligibility_v1.csv`(2026-10-04 정정에서 추가, L41 변형의 대상 셀 수), `lgd_repro_gate.csv`, `lgt_gate.csv`, `lgf_gate.csv`, `lgfn_cross.csv`, `lgfn_cross_gate.csv`, `wf9_repro_check.csv`, `lgw_xenv_c14_summary.json`, `lgw_xenv_gate_i_summary.json`, `lgw_xenv_gate_iii_summary.json`, `Fig6_a.csv`, `Table1_rows.csv`. 모두 집계 표이며 셀 단위 라벨 자료는 없다.
- 복사하지 않은 원천: `data/processed/lgw/lgw_xenv_c14.csv`(5 MB 상한 초과, 요약 JSON 으로 대신함), LGT·LGF·LGD 조각(`shards/`, 셀 단위 예측 포함).
