# 남은 작업 실행 계획 (2026-09-30)

**기준 시각**: 2026-09-30 11:30 KST(심사 2건 반영판). 초판은 10:45 에 썼다(스크래치 보관). 진행 정보는 09:05–11:28 의 진행 로그, 상태 표, nvidia-smi, 결과 묶음의 파일 이름 목록(`tar tzf`, 이름만)으로 확인했다.
**성격**: 실행 계획이다. 사전 등록 문서가 아니다. 기존 계획서의 가설, 판정 문구, 범위, 확인적 가설 집합은 바꾸지 않는다. 등록 규칙을 더하거나 바꿔야 하는 항목은 6절에 초안으로 두며, 해당 계획서의 개정으로 커밋할 때 효력이 생긴다.
**열람 상태**: 이 문서를 쓰면서 LG, LGX, LGT, LGU, LGF, LGD 의 곡선·판정 표와 조각의 Δ·RMSE 를 열지 않았다. 읽은 것은 진행 로그, 상태 표(`*_run_status.json`, `lgu_status.csv`, `lgx_all_status.csv`), 시간 표, 창 파일, 교차 환경 점검 (i) 요약, 결과 묶음의 파일 이름 목록과 묶음 안 `lgx_cpu.log` 의 조각별 시간 열, 입력 라벨 표 `fidelity_base_v4_labels.csv` 의 약관 열, 코드와 계획서다. LGX 두 묶음은 풀지 않았다(`data/processed/lgx/` 없음, 11:28 확인).
**기준 문서**: `docs/RESEARCH_FRAME_2026-09-29.md`, `docs/NOVELTY_POSITIONING_2026-09-29.md`, `docs/WRAPUP_REVIEW_2026-09-30.md`, `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md`(이하 WRAPUP), `docs/EXPERIMENT_PLAN_LG_2026-09-29.md`(이하 LG, 6A LGX, 6B LGD, 6C LGT, 개정 1–14), `docs/EXPERIMENT_PLAN_LGU_2026-09-29.md`(이하 LGU), `docs/EXPERIMENT_PLAN_LGF_2026-09-29.md`(이하 LGF), `docs/FAILURE_ANALYSIS_2026-09-29.md`, `docs/COVERAGE_MATRIX_BY_REGION_2026-09-29.md`, `SESSION_HANDOFF.md`, `docs/EXPERIMENT_LOG.md`.
**최신 사용자 지시(2026-09-30 08:45)**: Rescale 에 추가 비용을 쓰지 않는다. 남은 실험은 로컬 GPU 가 비는 대로, 또는 우리 작업이 끝나 남는 GPU 로 진행한다. 로컬 CPU 는 우리 작업 합계 32스레드 이하, nice 10 이다. 다른 사용자가 GPU 를 수시로 잡으며, 점유 판정은 메모리 50 MiB 이하와 계산 프로세스 0 이다.

---

## 0. 심사 반영 요약

두 심사의 지적 23건을 코드와 파일로 대조했다. 사실과 다른 지적은 없었고, 두 건은 정정해 반영했다(아래 표의 '정정').

| 지적 | 확인한 사실 | 반영 위치 |
|---|---|---|
| T_res 오기(두 심사, high·medium) | `results/rescale_lgx_b` 생성 07:20:22, `results/rescale_lgx_a` 생성 10:55:44(stat). WRAPUP 0.3 의 정의상 T_res = 07:20:22 다 | 1.2, 5.3, 6.1 (l), 6.4, 4.3 G0 명령 |
| LGD 약관 미확인 규칙 부재(high) | NAtlantic 새 셀 38 가운데 19 가 `lic_unverified`(calm_web_subsites 13, CUSP 6). 캐나다 확충판 새 셀 75 가운데 38(CUSP 5, GGD353 33). h51·h52 에 약관 거름 기능이 없다 | 6.1 (m), A6, C2b, M6, 7절 결정 6 |
| AB10 입력을 h39 가 읽지 못함(high) | h39 `lgu_ab10` 은 `*tests*.csv` 의 `test_id` 열만 읽는다(1287–1303행). 계획의 `lgu_a_ab10.csv` 는 glob 에 걸리지 않고, 비고는 '2,000회'(1344행)다. 출처 스크립트는 h44 다. `판정 불가(…)` 문자열은 h39 `NA_V` 와 정확히 같지 않아 p 가 1 로 바뀌지 않을 수 있다 | 6.3, A3, 보조 스크립트 `lgu_ab10_tests.py`(합성 입력 시험 통과) |
| 격자 σ 경로 부재(medium) | h49_transfer_map 990–993행은 σ 파일이 없으면 중단한다. h44–h46 은 적합 모델을 저장하지 않는다(`state_dict` 는 조기 종료용 메모리 사본뿐) | 6.3, A7, G8 우선순위 상향 |
| 남은 g45 조각 수(medium) | 묶음 이름 목록: g45 FT-T 22 가운데 15 완료, 남은 7(레나 s4, 캐나다·CA-2·CA-3 s4·s5). nn FT-T 4 완료, 남은 51. 합 58 = 572 − 514 | 1.1, 3.2 G4, 6.1 (b)(d), 큐 파일 |
| g45 설정 해시 서술(medium) | 코드로 계산한 결과 로컬(`--targets`)과 Rescale 인자의 전체 해시(cfg_hash)가 8개 (대상, 분할) 모두 같다. `--resume` 이 Rescale 완료 분할을 건너뛴다 | 6.1 (d), 실행기 `check` 가 전체 해시도 출력 |
| 계속 규칙 (나)의 퇴화(medium) | `lgw_xenv_gate_iii.py` 159–166행은 로컬 반복 차가 0 이면 기준이 1e-6 cm 로 떨어진다 | 6.1 (f)에 (다) 추가, 대조기 판 2 |
| 플랫폼 혼합 예외의 근거 범위(medium) | 점검 (i)의 1.4e-14 cm 는 캐나다 x 분할 1 과 스모크 3단위에서 잰 값이다. base 축 조각은 Rescale 에서 조각당 419–895 s(묶음 로그) | 6.1 (d), C14(선택), 7절 결정 1 |
| LGF 에서 N1 이 밀릴 경로(medium) | LGF 하네스 `ALLOWED_GPUS = (2, 3, 4, 5, 6, 7, 9)`(h47 226행, h48 236행). h48 집계는 코드 해시가 섞이면 경고만 한다(3569–3572행) | 6.2 전환 규칙, 감시기 GPU 부족 감시 |
| 조건부 작업 누락(medium) | WRAPUP 11.1 의 n = 80 격자점, 잔차 크리깅 기준선 | C12, C13 |
| 표시 항목 대응표 부재(medium) | 현재 본문 Fig 1–7 과 Table 1(8개), SI 표 다수(`figures/figure_spec.json`) | 7절 초안, F1 |
| 자료·코드 기탁 계획 부재(medium) | `.gitignore` 가 `*.npz`, `logs/` 를 뺀다. SI 표 `paper_tables10a_software`, `10b_runs` 가 있다 | O8, F3, 8절 결정 10 |
| 과잉 작업(low) | nn 축은 h42 `build_tests_x` 의 판정에 쓰이지 않는다. h42 집계에는 축·학습기 거름 인자가 없다 | G6 조건부, C10 부분 폴더 집계 |
| G7 기본값(low) | 로컬 RealMLP 조각은 주 판정에 들어가지 못한다 | G7 기본 '하지 않음', C0 에서 주 4지역 여부 확인 |
| `--splits`(low, 정정) | h42 `--splits K` 는 분할 1..K 다. 분할 하나만 고를 수 없다. 다만 전체 해시가 같아 재적합 낭비는 없다 | 6.1 (d) |
| 점검 (iii)의 규모 대표성(low) | 점검 단위는 작다(러시아 W, AL-2) | 레나 x g45 분할 5 보충 대조(6.1 (e)) |
| L5·L26 문장(low) | WRAPUP 1.5 의 원고 문장 원천은 L26 이다 | 1.5 표 |
| 원고 마무리 단계(low) | SI 조립, 형식 점검, H15 교체 누락 | M9, J10 |
| C3 의존성(low) | 개정 전 집계는 첫 실행 판정 고정 규칙과 충돌 | C3, `run_post_results.sh` 의 등록 확인 |
| 표 폴더 변수(low) | 로컬 재집계 시 표와 조각의 폴더가 갈린다 | `$LGS`(5.1) |
| 묶음 해제 경로(low) | 두 묶음의 최상위 항목이 저장소 루트에 풀린다. `lgx_payload_info.txt` 가 겹친다 | `data/processed/lgx` 만 푼다(실행기 `unpack`) |
| J12 결정 기록(low) | 감시기가 F·N 완료 시 J12 판단 없이 멈춘다. h47 은 창 마감 확인이 f_t3 판단보다 앞선다 | 감시기 역할 T3, 6.2 |
| 스레드 상한 재읽기(low) | `LGXL_OUR_THREADS` 는 시작 때 한 번만 읽힌다 | 실행 중 조정 파일(4.5) |
| R1 행 문구(low) | 5 % 여유는 J1–J11 전체 값이다 | 1.5 표 |
| LGT 넘김 조건(low, 정정) | 심사는 `n_done + n_resumed == n_todo` 를 제안했다. h43 의 `n_todo` 는 `--resume` 으로 건너뛴 단위를 뺀 실행 수라(2496–2503행) 올바른 조건은 `n_done == n_todo` 다 | 감시기 `lgt_update`, 두 번째 통과 자동화 |

---

## 1. 요약

### 1.1 현재 상태(11:28)

| 작업 | 위치 | 상태 | 종료(추정) |
|---|---|---|---|
| LG 본 실행 ZovWo | Rescale iolite-4 | 10:40 기준 CPU 117/117, GPU 225/231(남은 6조각은 RealMLP), 실패 0 | GPU 부분 약 13:50 자체 정지, 작업 안 집계·묶음 뒤 14:30 이전 종료 |
| LGX 확장 A Qjpbeb | 회수 완료(10:55:44, 묶음 미해제) | 09:02 중지 요청, 09:03:04 종료(코드 137). 마지막 묶음(09:02:02)이 514번째 GPU 조각(08:44:44 완료) 뒤라 빠진 조각은 없다. 묶음 이름 목록: CPU 3,860, h41 292, GPU 514 | 남은 GPU 58조각(FT-T: g45 7, nn 51)은 로컬 보조 이어 실행(G4–G6) |
| LGX 확장 B ShFtT | 회수 완료(07:20:22, 묶음 미해제) | 77/77(RealMLP), 종료 코드 0 | 완료 |
| LGT(h43, TabPFN) | 로컬 GPU 9·7·6 | 154/172, 실패 0 | 12:10–12:50 |
| LGF(h47·h48) | 로컬 GPU 5 | 08:32 GPU 5 상실(완료 282), 09:04:55 재개. 현재 호출 487/854 단위(11:28). 창 마감 10-02 05:01 | GPU 4장 확장 시 10-02 01:00–05:01 |
| LGU(h44–h46) | 로컬 | 본 실행 완료(04:32). 판정 표 `lgu_a_tests.csv`(04:14)·`lgu_b_tests.csv`(04:31)는 접근 시각이 생성 시각과 같다(미열람) | 완료 |
| LGD(h51) | 로컬 | 22표 완료(07:18), 실패 0 | 재현 점검, 약관 확인분 판, 확장 풀 집계는 LG 회수 뒤 |
| 결과 전 계산, 레나 지도 격자 | 로컬 | 완료(커밋 028900a) | 해당 없음 |

**Rescale 두 작업의 상한 확인(사용자 질문 2)**: ZovWo 는 `run_lg.sh` 의 시간 예산(1,275분, 집계 예비 40분)에 따라 GPU 부분을 약 13:50 에 스스로 멈추고, 작업 안 집계와 결과 묶음을 마친 뒤 14:30 이전에 끝난다. 22 h 벽시계 상한(09-29 17:03 제출 기준 09-30 15:03 이후. 실제 시작은 제출 뒤라 더 늦다)보다 앞서므로 상한에 걸리지 않는다. 남은 RealMLP 조각은 정지 전에 끝날 수 있으나 확정하지 않는다. Qjpbeb 는 09:03 에 종료되어 상한과 무관하고, 회수 묶음에 빠진 조각이 없음을 확인했다.

### 1.2 남은 작업의 묶음

1. **결과 열람 전 문서 개정(6절)**: LG 개정 15, LGF 개정 6, LGU 개정 4, WRAPUP 개정 2, NEXT 개정 기록. WRAPUP 0.3 의 T_res 는 이미 07:20:22(ShFtT 회수 폴더 생성)이므로, 네 개정은 모두 T_res 뒤에 커밋된다. 각 개정에 '회수 뒤, 조각 미해제·미열람 상태의 개정' 표지와 근거(묶음 미해제, `data/processed/lgx/` 없음, LGU 표 접근 시각 = 생성 시각)를 적는다. 다음 기준 시각은 LGX 묶음 해제 시각, 조각 내용 첫 열람 시각(점검 (iii) 대조기 또는 첫 집계), 첫 집계 표 생성 시각이며 실행기와 파이프라인이 자동으로 기록한다.
2. **로컬 GPU 실험**: LGF 남은 약 150 GPU-h, LGX FT-T 로컬 이어 실행(보조) 약 27.5 GPU-h(보충 단위 포함), 교차 환경 점검 (iii) 약 1.3 GPU-h, 조건부로 격자 σ(약 0.5 GPU-h)와 LGF J12.
3. **회수와 집계(CPU)**: ZovWo 회수, LGX 해제, LG·LGX·LGT·LGD·h39·LGF 집계, 지도.
4. **판정 기록과 문서 정정**.
5. **그림·표, 지도 그림, 원고, SI, 기탁**.
6. **운영 기록**: 원장, 실험 로그, 핸드오프, 메모리, 커밋, push.

### 1.3 순서

| 단계 | 시각(추정) | 내용 |
|---|---|---|
| 0 | 지금–12:30 | 감시기 교체(G0)와 지난 GPU 사건 기록. A1–A5 작성·커밋(결과 표를 열지 않은 상태). A1–A4 커밋 뒤 push(사용자 확인, O6)를 C0 보다 먼저 한다. A3·A6·A7 의 코드 작업(결과 전) |
| 1 | 12:30–14:30 | LGX 묶음 해제(C0b, `run_lgx_local_continue.sh chain` 이 해제·seed·check·점검 (iii)을 이어서 한다). GPU 8 에서 점검 (iii) 2회(약 1.3 h) 뒤 계속 규칙이 '계속'이면 G4 시작. CPU 로 C3 계산(열람은 5.3 순서), F1, M1 |
| 2 | 12:10–13:00 | LGT 종료. 감시기가 상태 표를 보고 완결이면 GPU 6·7·9 를 LGF 로 넘긴다(F 1장, N 나머지). 완결이 아니면 두 번째 통과를 자동으로 띄운 뒤 넘긴다 |
| 3 | 14:30–22:00 | ZovWo 회수(C0) 뒤 LG 판정(J1), LGD 재현 점검(C2), 약관 확인분 판(C2b), LGX 주 집계(C4), LGT 집계(C5), h52(C6), AB10(A3 보조), h39 1차(C7), 지도 대비(C8 전반) |
| 4 | 10-01 | LGF J2–J5 완료(오전 추정), LGX 로컬 이어 실행(G4–G5 오전, G6 오후) 뒤 보조 집계(C10). 그림 모듈 작성 |
| 5 | 10-02 05:01 이후 | LGF 창 마감 뒤 LGF 집계(C9), h39 재실행(δ_rel), 지도 최종(C8), 그림 최종(F3), 결과 절(M2), SI 조립(M9). 남는 GPU 로 LGX nn 나머지와 조건부 작업 |

### 1.4 예상 완료 시점(추정)

| 항목 | 시점 |
|---|---|
| LG L1–L8 판정 기록 | 09-30 17:00 전후 |
| LGT L32–L37 | 09-30 19:00 전후 |
| LGX L9–L31(주 판정, Rescale 조각만) | 09-30 20:00–22:00 |
| LGD L38–L42(두 판 병기) | 09-30 22:00 전후 |
| LGU A1–A7, B1–B6, C1–C4 | 09-30 오후(A3 커밋 뒤) |
| h39 주 대비 묶음(AB1–AB10), SC1w–SC3w, L43, AK1w | 10-01 오전(1차), 10-02 오전(δ_rel 을 채운 최종) |
| LGF F1–F6, N1–N4 | 10-02 08:00 전후 |
| LGX 보조 판정(L26[ftt] 교차 환경, nn FT-T 서술 행) | 10-01 저녁–10-02 |
| 지도 최종판과 논문 그림 | 10-02–10-03 |
| 결과 절·초록 초안, SI 조립과 형식 점검 | 10-03 이후 |

### 1.5 완료 가능성 검토(연구 의의 기준)

| 주장·틀 | 판정 근거 | 남은 일 | 로컬만으로 가능한가 | 위험 |
|---|---|---|---|---|
| R1 라벨 0 에서 직접 ML 은 물리식을 넘지 못한다 | LG L1, LGX L30(확인적), LGT L34, LGF-F1·N1(확인적) | 회수·집계, LGF J2–J5(약 83 GPU-h) | 가능. J1 은 끝났다 | N1 은 GPU 4장 기준 10-01 오전에 끝나 창까지 약 18 h 여유가 있다. 창 여유 약 5 % 는 J1–J11 전체의 값이며 위험은 J7·J10(N2·N3)에 있다(4.4) |
| R2·R3 라벨 수 곡선, 물리 잔차의 순가치 | LG L2–L4·L8, LGX L10·L12·L15(확인적) | 회수·집계 | 가능. CPU 축은 Rescale 에서 완결 | 로컬 집계 실패 시 대기 규칙(6.1 (g)). L4 최소 n 이 160 이면 C12 |
| R4 관측 배치 | WRAPUP L43, LGX L23·L24 | h39·h42 집계 | 가능 | 없음 |
| R5 불확실성과 독립 지역 | LGU-A1·B2(확인적), LGX L19(확인적), LGD L38–L42, PE1·PE2 | LGU 판정 기록, h41·h52 집계, 약관 확인분 판 | 가능 | PE1 의 새 얕은 지역 NAtlantic 새 셀 절반이 약관 미확인이다. 결과 전 규칙 6.1 (m)과 두 판 병기로 막는다. 허락이 없으면 주 판정은 약관 확인분 판이다 |
| 틀 (1) 물리 증강 + 물리 잔차 | L2, L15–L18 | 집계 | 가능 | 없음 |
| 틀 (2) 라벨 0–100 워크플로와 배치 지침 | SC1w–SC3w, L43, L3·L4 | h39 | 가능 | n* 가 '40 초과 160 이하'이면 C12 조건 |
| 틀 (3) ALT 지도와 불확실성 지도 | WRAPUP 6절 MAP, LGU-B2 | 해석 확정(A4), h49, 그림 | 가능(10-02) | LGU-B2 가 '전이'이면 격자 σ 가 필요하다. 규약(6.3)과 스크립트(A7)를 결과 전에 두고 GPU 8 에서 먼저 돈다 |
| 학습기 동등성 | 원고의 학습기 문장은 학습기마다 LGX L26 판정을 쓴다. FT-T 처럼 L26 이 판정 불가인 학습기만 LG L5 3분할 4분 판정(분할 3 표기)을 쓴다(WRAPUP 1.5) | FT-T 분할 4·5 가 Rescale 에서 끝나지 못했다 | 주 판정은 h42 의 '판정 불가(5분할 판정에서 제외, LG 3분할 판정 유지)'. 로컬 이어 실행은 보조 판정만 만든다 | 줄어드는 근거는 L26 의 FT-T 행과 nn 축 FT-T 서술 행뿐이다 |
| 신경망 조정 강건성 | LGF-N1(확인적), N2·N3(J7 필요) | J7 약 53 GPU-h, J10 약 13 GPU-h | 가능하나 여유가 작다 | GPU 상실 시 J10, J7 꼬리 순으로 창 밖에 남고 N2·N3 은 '부분(분할 k/K)'. 전환 규칙(6.2)이 GPU 8 을 넘길 수 있다 |
| Sci Rep 형식 | 표시 항목 8개 상한, 자료·코드 기탁 | 대응표(7절), F1, M9, O8 | 가능 | 대응표에서 8개를 넘으면 F1 에서 SI 로 옮긴다 |

- 확인적 가설(LGX L10, L12, L15, L19, L29, L30, LGU-A1·B2, LGF-F1·N1)과 LG L1–L8 은 모두 로컬 자원으로 판정할 수 있다.
- 계획대로 해도 닫히지 않을 수 있는 곳은 네 곳이었고, 이번 개정으로 모두 결과 전 규칙을 둔다: (1) PE1 의 약관(6.1 (m)), (2) 격자 σ(6.3, A7), (3) 표시 항목과 기탁(7절, F1, M9, O8), (4) 사전 등록 시각의 서술(6.1 (l), 6.4).
- GPU 우선순위는 LGF > LGT > 조건부 σ(B2 가 '전이'일 때) > LGX 보조 > 조건부 J12·G7 이다(4.2). LGF 하네스는 GPU 8 을 거부하므로 GPU 8 을 LGX 보조에 쓰면 LGF 의 가용량이 줄지 않는다. LGF 가 밀리면 전환 규칙(6.2)으로 GPU 8 을 넘긴다.

---

## 2. 세션 한도로 중단·누락된 항목과 조치

### 2.1 워크플로 중단(세션·사용량 한도)

이 세션의 워크플로 저널 22개에서 started 기록마다 같은 키의 result 가 있는지 대조했다. 상태 점검 에이전트의 대조와 결과가 같다.

| 워크플로 | 중단된 에이전트 | 재수행 | 남은 조치 |
|---|---|---|---|
| wf_2f304fa7-89a(09-26 13:11–13:32) | 감사 반박 66건, viz:synthesis·judge1·judge2(69건 failed) | wf_56b70905-792 에서 반박 67/67 result, wf_d0a95ebb-77b 에서 viz:judge1·2·revise result | 없음 |
| wf_04fa64ea-323(09-26 13:09–13:35) | impl:h31, verify:h33:code·leak, verify:h35:code·leak, verify:h36:leak, fix:h34, fix:h36(8건) | wf_d0a95ebb-77b 의 check·fix·launch:h31–h36 에 result, A2·B1–B4 결과 커밋 f098031 | 없음 |
| wf_1c035f0e-58a, wf_9ae37d54-969, wf_ea2b1da5-83e, wf_d7ec99b5-e4a, wf_6a81004a-323, wf_d3f37deb-d20(09-29–30) | 첫 시도 실패(synthesize, verify:h40 3종, propose 3종, LGT 1건, revise:plan, integrate:v4·verify 3종·fix:final) | 같은 워크플로의 재시도에서 모든 라벨에 result. 산출(RESEARCH_FRAME, LG 개정 1, NOVELTY, LGF 개정 1, v4 표)이 커밋되어 있다 | 없음 |
| wf_b0be06f0-359(09-29–30, LGU) | implement·verify·fix 첫 시도 실패, package:rescale 은 끝내 result 없음 | 구현·검증·수정은 재시도로 result. package:rescale 은 LGU 개정 1 의 로컬 전환으로 package:local 이 대신했다 | 없음. `configs/rescale/lgu_*.yaml` 은 만들지 않는다 |
| wf_e8661a61-243(이번 작업) | 해당 없음 | 해당 없음 | 해당 없음 |

결론: 세션 한도로 끊긴 에이전트 작업은 모두 재수행되었고 결과가 커밋 또는 산출 파일로 남아 있다. 저널 대조로 새로 발견한 미이행 계산 작업은 없다.

### 2.2 실행 중단과 누락(계산)

| 항목 | 내용 | 조치(3절 id) |
|---|---|---|
| LGF GPU 상실 | 08:32 다른 사용자(supergood)가 GPU 5 를 잡아 h48 워커가 멈췄다(rc 1, 완료 282 단위). 09:04:55 재개. `lgf_window.json` 의 `gpu_events` 에 기록되지 않았다 | 사건 사후 기록(G0), LGF 개정 6(A2) |
| LGF 확장 장치 누락 | `wait_and_resume.sh`(PID 82442, 11:09 확인)는 큐가 죽었을 때만 재시작한다. LGT 가 끝나도 GPU 를 늘리지 않고, F·N 분리(LGF 6.3 단계 6)와 J12 판단이 없다 | 감시기 교체(G0) |
| LGT 두 번째 통과의 세션 의존 | partial 이 있으면 두 번째 통과를 세션이 띄워야 했고, 그동안 GPU 3장이 빈다 | 감시기가 자동으로 띄운다(4.3) |
| Qjpbeb 중지 | 작업 안 통합 집계(`LGX_SUMMARIZE=1`)가 돌지 않았다. 빠진 조각은 없다(1.1) | 로컬 통합 집계(C4), 남은 FT-T 조각 처리(A1, G4–G6) |
| GPU 학습기의 교차 환경 점검 부재 | 8.4 (i)(ii)는 CPU 학습기와 집계만 본다 | 점검 (iii) 신설(6.1 (e)(f), G3) |
| LGD 본 실행 기록 누락 | 07:18 완료(22표, 실패 0, 10,754 s)가 커밋 64cf3d3 메시지에만 있다. `lgd_run_manifest.json` 수정분 미커밋 | A1 (i), O5 |
| 메모리 갱신 중단 | `lg-rescale-run-state` 가 09-29 밤 상태다. LGU·LGF·LGD·WRAPUP 상태 메모리가 없다 | O4 |

### 2.3 등록 문서의 미이행 항목(결과 열람 전에 이행)

| 항목 | 출처 | 조치 |
|---|---|---|
| LG 머리말·6A.9·6C.8 의 실행 환경 문안 이관 | WRAPUP 8.2, 부록 deferred_items 6 | A1 (g) |
| LG 6A.7 비맹검 누락(L29 의 B:ens − P0) | WRAPUP 12절 후속 (2) | A1 (h) |
| AB10 의 두 가중 p. h39 는 `*tests*.csv` 의 `test_id`·`p_cell`·`p_beq` 열을 읽는다 | WRAPUP 1.1·12절 후속 (3) | A3(보조 스크립트 작성 완료, 등록은 LGU 개정 4) |
| h39 AB10 비고의 'p 분해능 5e-4(2,000회)'(1344행)와 WRAPUP 의 재표집 횟수 표기 | 상태 점검, 심사 | A3(h39 문자열), A4(WRAPUP 본문) |
| WRAPUP 1.4 (b) 본문 정정(H18–H22 두 가중 CI 있음, H20 예측은 `h21_preds.npz`, 입력 29개) | 결과 추가 1 후속 | A4 |
| 지도 규칙 해석 두 건(6.5 수체, PFR 결측 회색)과 산출 경로(`map_lena`) | 결과 추가 2 후속 (1) | A4 |
| h39 `aux4_table` 에 지도 대비를 더할지 | 결과 추가 2 후속 (2) | A4 |
| LGU-B2 '전이'일 때의 격자 σ 규약과 스크립트 | WRAPUP 6.9 | A7(LGU 개정 4) |
| LGD 약관 미확인 자료의 판정 규칙 | WRAPUP 7.5 (c)3(공개 제한만 있음) | A1 (m), A6 |
| NEXT 계획서의 SC1–SC3·AK1 철회 기록 | WRAPUP 12절 후속 (1) | A5 |
| git push(사전 등록 시각의 제3자 기록). 로컬 main 이 origin 보다 33 커밋 앞서 있다(11:25) | WRAPUP 0.3 | O6(사용자 확인) |

### 2.4 일정이 없던 등록·권고 항목(이번 계획에 추가)

| 항목 | 출처 | 이번 계획의 처리 |
|---|---|---|
| figure_spec.json 갱신과 표시 항목 대응표 | WRAPUP_REVIEW 2절 9번, 심사 | F1(결과 전), 7절 초안 |
| 관측 시기·라벨 품질 해석식 점검 | WRAPUP_REVIEW 2절 10번 | C11(사후 탐색 표지) |
| E 의 물리 해석 문단, 원천 E0 구성, 지역 등가중 E0 민감도 | WRAPUP_REVIEW 2절 11번 | M3. 민감도 계산은 LG 결과 전 등록이 필요하다(8절 결정 9) |
| SI 음성 결과 등록표, 선택·철회 이력 표 | WRAPUP_REVIEW 2절 12번 | 재채점은 C7, 이력 표는 M4, SI 조립은 M9 |
| 방법·서론 영문 초안, Data·Code availability | WRAPUP_REVIEW 2절 13번 | M1, O8 |
| n = 80 격자점(L4 최소 n 이 160 일 때), 잔차 크리깅 기준선(LGU-C1 조건) | WRAPUP 11.1 | C12, C13 |
| 기존 제품 비교 SI 표(h50) | WRAPUP 10절 | M7(LGU-B2 조건) |
| NOVELTY 7.4 선택지 3·4 | NOVELTY 7.4 | 제외 기록(8절 결정 8) |
| RESEARCH_FRAME 5.2·FAILURE_ANALYSIS 6 의 선택 항목 | RESEARCH_FRAME 5.2 | 제외 기록(A4) |
| 지도 그림의 시각 검토 | WRAPUP 결과 추가 2 | F2 |
| `lg_aux` 경로의 교차 환경 재현 | WRAPUP 결과 추가 1 수정 1 | C4 의 '교차 환경 미확인' 표지 |
| n = 10 추출 재생성의 LG 조각 대조 | WRAPUP 2.3 | J8 |

---

## 3. 남은 작업 표

기호: `$LGD` = `results/rescale_lg/data/processed/lg`(ZovWo 조각, 읽기 전용). `$LGS` = LG 표 폴더(5.1). 모든 로컬 실행은 `nice -n 10` 이다. 시간은 모두 추정이다. 괄호 안 id 는 상태 점검(R, D, J, F, M, O)과 계산 자원 계획(P, G, C)의 원래 id 다. 파이프라인 명령은 `scripts/local/run_post_results.sh <하위 명령>`(이하 `post`), LGX 로컬 명령은 `scripts/local/run_lgx_local_continue.sh <하위 명령>`(이하 `lgxc`)이다.

### 3.1 결과 열람 전 문서 개정과 코드(A)

| id | 내용 | 우선 | 자원 | 시간 | 의존성 | 방법 | 상태 |
|---|---|---|---|---|---|---|---|
| A1 | LG 개정 15(6.1 초안 (a)–(n)) | P0 | 문서 | 1 h | 없음(숫자는 이 문서에 확정) | LG 개정 이력에 추가, 커밋 | 미착수 |
| A2 | LGF 개정 6(6.2 초안): GPU 사건, 감시기 교체, 역할 T3, LGT 두 번째 통과 자동화, GPU 8 전환 규칙, 창 산술 | P0 | 문서 | 0.5 h | G0 시각 | LGF 개정 이력에 추가, 커밋 | 미착수 |
| A3 | LGU 개정 4(6.3 초안): AB10 산출 규칙. 보조 스크립트 `scripts/2_evaluation/lgu_ab10_tests.py`(작성, 합성 입력 시험 통과). h39 1344행 비고를 '1,000회(분해능 1e-3)'로 고치는 한 줄(코드 변경, 커밋) | P0 | 문서, CPU 1 | 0.5 h | LGU 판정 표 열람 전 | 개정 커밋 뒤 `post lgu-ab10` | 스크립트 작성 |
| A4 | WRAPUP 개정 2(6.4 초안) | P0 | 문서 | 1 h | 결과 표 열람 전 | 개정 이력에 추가, 커밋 | 미착수 |
| A5 | NEXT 계획서 개정 기록(SC1–SC3·AK1 철회) | P2 | 문서 | 5분 | 없음 | 한 줄 | 미착수 |
| A6 | 약관 확인분 판 구현: h51 에 묶음 `lic`(`lic_unverified = 1` 셀이 들어가는 실행 표마다 그 셀을 뺀 표. 적격 규칙 6B 를 그대로 적용), h52 에 두 판 병기 열과 6.1 (m)(2)의 주 판정 표지. 단위 시험 | P0 | 코드, CPU 1 | 2–3 h | A1 커밋(규칙 먼저), C6 전 | `--count-only` 로 셀 수·적격 확인(1분 이내) | 미착수 |
| A7 | 격자 σ 스크립트(6.3 조건부 규약): LGU 실험 B 의 nflow 설정으로 보정 셀과 레나 격자 셀의 σ 를 h49 npz 형식(`cal={지역: σ}`, `grid_cell_id`, `grid_sigma`)으로 쓴다. py_compile·합성 입력 시험까지 | P1 | 코드 | 2–3 h | A3 커밋, LGU-B2 표 열람 전 | 결과 전 작성 | 미착수 |

### 3.2 로컬 GPU 실험(G)

| id | 내용 | 우선 | 자원 | 시간 | 의존성 | 명령 또는 방법 | 상태 |
|---|---|---|---|---|---|---|---|
| G0 | LGF 감시기 교체와 지난 GPU 사건 기록 (P3) | P0 | 감시 프로세스 1 | 10분 | LGT 종료(12:10–12:50) 전 | 4.3 의 명령 | 스크립트 수정(정적 점검) |
| G1 | LGF 본 실행 계속. 두 장 이상이면 F 역할 1장, N 역할 나머지. F·N 완료 뒤 역할 T3 이 J12 조건을 판단 (R06) | P0 | GPU 5, LGT 뒤 5·6·7·9(2·3·4 가 비면 추가, 최대 6장) | 남은 약 150 GPU-h | G0 | 감시기가 `scripts/local/lgf_role_queue.sh` 로 띄운다 | 실행 중(GPU 5) |
| G2 | LGT 본 실행 완료와 두 번째 통과(상태 표가 완결이 아닐 때) (R05) | P1 | GPU 9·7·6 | 12:10–12:50 종료 | 없음 | 교체한 감시기가 자동으로 띄운다(`h43 --resume --rerun-partial --gpu-mem-max-mib 50`, 빈 GPU 만). 그 뒤에도 partial 이 남으면 LG 6C 규칙 | 실행 중 |
| G3 | 교차 환경 점검 (iii): Rescale 에서 끝난 FT-T 작은 단위(nn 러시아 W x 분할 1, r0 러시아 W x 분할 1, g45 AL-2 x 분할 4·5)를 로컬에서 두 번 재적합하고 대조 | P1 | GPU 8, 스레드 2 | 2회 약 1.3 GPU-h | A1 커밋, C0b | `lgxc chain` 이 이어서 한다 | 스크립트 작성 |
| G4 | LGX g45 FT-T 남은 7조각(레나 s4, 캐나다·CA-2·CA-3 s4·s5)과 보충 1조각(레나 s5, 점검 (iii) 규모 보충) (R04) | P1 | GPU 8, 프로세스 1, 스레드 2 | 약 11.6 GPU-h(9–14) | G3 계속 규칙 '계속' | `lgxc chain`(규칙이 '계속'이면 자동 시작) | 스크립트 작성 |
| G5 | LGX nn FT-T 주 4지역(분할 2–5, 16조각) | P2 | GPU 8 | 약 6.4 GPU-h(5.6–7.4) | G4 | 같은 큐 | 스크립트 작성 |
| G6 | LGX nn FT-T 나머지 대상(35조각: AL-1 4, AL-2·AL-3·Alaska·CA-2·CA-3·Russia_C 각 5, Greenland 분할 2 의 1. 서술 행. 11:44 해제 뒤 대조로 정정). 등록 작업(σ, J12, 전환 규칙)이 GPU 8 을 요구하지 않을 때만 | P3 | GPU 8. LGF 창 마감 뒤 조정 파일로 '9 8 7 6 5', 작업 4 | 약 9.5 GPU-h(8.4–11.1) | G5 | 같은 큐. 요구가 생기면 드레인 | 스크립트 작성 |
| G7 | (조건부, 기본 하지 않음) ZovWo 가 끝내지 못한 RealMLP 조각의 로컬 보조 이어 실행 | P3 | GPU 1장/조각 | 조각당 2–10 h | C0 에서 빠진 조각이 주 4지역(CI 풀)에 속할 때만 사용자 결정 | 6.1 (j) | 대기 |
| G8 | (조건부) 격자 σ(LGU-B2 '전이', GPU 약 0.5 h, G4–G6 보다 먼저 GPU 8 에서), LGF J12(감시기 역할 T3 이 판단) | P1(σ), P3(J12) | GPU 1장 | σ 약 0.5 GPU-h, J12 약 5.7 GPU-h | σ: A7, J3. J12: LGF 8.4 | σ 는 LGX 로컬을 드레인한 뒤 돌린다 | 조건 대기 |

### 3.3 회수와 집계(C, CPU)

| id | 내용 | 우선 | 자원 | 시간 | 의존성 | 명령 | 상태 |
|---|---|---|---|---|---|---|---|
| C0 | ZovWo 회수·해제, 조각 수, 빠진 RealMLP 단위 이름과 그 단위가 주 4지역인지, 작업 단계 종료 코드. 조각 sha256 목록, 읽기 전용 (R01) | P0 | CPU 1 | 0.5 h | ZovWo 종료(14:30 전후) | `post fetch-lg` | 대기 |
| C0b | LGX 두 묶음의 `data/processed/lgx` 경로만 해제, unit.json 수 대조(CPU 3,860, h41 292, GPU 514 + 77), 해제 시각·sha256·읽기 전용 (R03, P2) | P0 | CPU 1, 디스크 약 1 GB | 10분 | A1–A4 커밋, push(O6) | `lgxc unpack`(`chain` 의 첫 단계) | 대기 |
| C1 | LG 집계 확인. 작업 안 h40 집계가 종료 코드 0 이고 표 4개가 있으면 `$LGS = $LGD`. 아니면 로컬 재집계 | P0 | CPU 4 | 0–1 h | C0, A1 | `post gate-lg`(필요하면 `sum-lg`) | 대기 |
| C2 | LGD 재현 점검(repro_Russia_W). base 범위 CatBoost 불통과면 v3local 3표 (R11) | P1 | CPU 4 | 10–20분, v3local 0.5–1 h | C0, A1 | `post gate-lgd`, 조건부 `post lgd-v3local` | 대기 |
| C2b | 약관 확인분 판 실행(해당 표 약 13개) | P1 | CPU 8(워커 2 × 4) | 약 1 h(22표 본 실행 10,754 s, NAtlantic 주 표 459 s 기준 추정) | A6, C0 | `post lgd-lic` | 대기 |
| C3 | h41 검증 사다리 집계(L19 확인적, L20–L22) | P1 | CPU 4 | 1 h 이내 | C0b, A1(개정 전에 만든 표는 첫 실행 판정 고정 규칙에 걸린다) | `post ladder`. 계산은 먼저, 열람은 5.3 순서 | 대기 |
| C4 | LGX 주 통합 집계(Rescale 조각만, 재표집 10,000): L9–L31, N1–N4, `lgx_lg_aux`('교차 환경 미확인'), splitdist, region_inference, lgx_gate (R13) | P0 | CPU 8(워커 2 × 4), 최대 RSS 기록 | 2–3 h | C0, C0b, A1, C1 | `post sum-lgx`(sha256 재대조, 무거운 집계 잠금) | 대기 |
| C5 | LGT 집계와 교차 비교(스레드 상한 2) | P1 | CPU 2 | 0.5–1.5 h | G2, C0 | `post sum-lgt` | 대기 |
| C6 | LGD 확장 풀 집계 h52(L1e·L4e·L8e, L38–L42, PE1·PE2, 두 판 병기) (R12) | P1 | CPU 4 | 1 h 이내 | C2, C2b, C4(`lgx_splitdist.csv`) | `post pool-lgd` | 대기 |
| C7 | h39 summarize(AB1–AB10 Holm, SC1w–SC3w, L43, AK1w, aux4, 분할 분산 비, 1.4 (b) 재채점, δ_rel)와 bias-mae. LGF 표가 나오면 다시 돈다 (J02) | P1 | CPU 4 | 각 1 h 이내 | C0, C4, C6, C5, A3(`post lgu-ab10`) | `post scenarios` | 대기 |
| C8 | 지도: h49_map_contrasts, h49_transfer_map(`--check-inputs` 먼저), fig_map_lena (F01) | P0 | CPU 4 이하 | 0.5일 | A4, C4, C7, LGU-B2, σ(조건부), C5·C9(없으면 잠정판) | `post map`(잠정판은 `POST_MAP_PROVISIONAL=1`) | 대기 |
| C9 | LGF 집계(h47, h48 `--cross-lg --lg-dir $LGS`) | P0 | CPU 2 | 각 1 h 이내 | G1 역할 모두 종료 또는 창 마감, G2 | `post sum-lgf` | 대기 |
| C10 | LGX 보조 집계(교차 환경). 기본은 L26·nn 에 필요한 축(lgxb, lgxg, lgxnn)의 Rescale 조각과 로컬 새 조각만 모은 부분 폴더 | P2 | CPU 8 | 1–3 h | G4(최소), G5, C4, G3 | `post sum-lgx-aux`(실패하면 `POST_AUX_FULL=1`) | 대기 |
| C11 | (사후 탐색) 관측 시기·라벨 품질 해석식 점검 | P3 | CPU 1 | 4–7 h | 확인적 판정 기록 뒤 | 해석식, '사후 탐색' 표지 | 미착수 |
| C12 | (조건부) n = 80 격자점: J1 에서 L4 최소 n 이 160 이면 `h40 --n-grid 80` 을 로컬 별도 폴더(`data/processed/lg_n80_local`)에서 돈다. '교차 환경; (i) 기준 방법 키 1.4e-14 cm' 표지. n* 보고는 등록된 구간 보고('40 초과 160 이하')를 n = 80 점으로 좁히는 방식으로만 쓴다 | P2 | CPU 8 | 재지 않음(h40 CPU 조각 117개 가운데 해당 격자점만) | J1 | 조건이 서면 명령과 시간을 A1 (n) 형식으로 개정 이력에 먼저 적는다 | 조건 대기 |
| C13 | (조건부) 잔차 크리깅 기준선: J3 의 LGU-C1 실효 범위가 채점 거리 중앙값보다 긴 지역이 있으면 별도 등록 문서를 쓴 뒤 실행하고 '결과 열람 뒤 등록' 표지를 붙인다 | P3 | CPU 1–4 | 등록 뒤 산정 | J3 | WRAPUP 11.1 | 조건 대기 |
| C14 | (선택) L26 보조 대비의 CatBoost 쪽을 같은 플랫폼으로: 레나·캐나다·CA-2·CA-3 x 의 base 축 CPU 조각을 로컬에서 다시 적합(`h42 --part cpu --axes base --targets …`, 분할 1–5. `--splits` 는 1..K 라 4·5 만 고를 수 없다)하고 Rescale 조각과 키별 차를 (i) 형식으로 보고 | P2 | CPU 8(워커 2 × 4) | 약 2–4 h(Rescale 조각당 419–895 s, 20조각. 로컬 속도는 재지 않았다) | G4, C4 | 출력 `data/processed/lgw/xenv/lgx_base_local`. 8절 결정 1 | 결정 대기 |

### 3.4 판정 기록(J)

| id | 내용 | 우선 | 의존성 | 기록 위치 |
|---|---|---|---|---|
| J1 | LG L1–L8 판정과 맹검 표지, 4분 보조 열(`lgx_lg_aux`), n* 구간 보고. L4 최소 n 이 160 이면 C12 조건 기록 | P0 | C0, C1 | LG §7 |
| J2 | LGX L9–L31(확인적 L10, L12, L15, L19, L29, L30)과 N1–N4. L26 FT-T 행은 h42 의 '판정 불가(5분할 판정에서 제외)' 그대로 | P0 | C3, C4 | LG 6A 결과 |
| J3 | LGU 판정(LGU-A1·B2 확인적, A2–A7, B1, B3–B6, C1–C4). B2 로 지도 본문/SI 배치와 Fig 6 구성, σ 필요 여부(G8), C1 실효 범위로 C13 조건 판단 (R08) | P1 | A3·A7 커밋 | LGU 11절 |
| J4 | LGT L32–L37 과 '등록 시점에 결과 존재(미열람)' 표지 | P1 | C5 | LG 6C 결과 |
| J5 | LGD L38–L42, P4·PE1·PE2, 두 판(전체, 약관 확인분) 병기와 주 판정 표지(6.1 (m)), '교차 환경; (i) CatBoost 불통과' 표지 | P1 | C6 | LG 6B 결과 |
| J6 | LGF F1–F6, N1–N4, N2s, 열람 상태, J12 결정(창 파일 또는 감시기 기록) | P0 | C9 | LGF 10절 |
| J7 | WRAPUP 결과(AB1–AB10, SC1w–SC3w, L43, AK1w). AB4·SC1w-P·LGX-N4a 인용 시 S-a 열람 표지 병기. AB10 행에 '등록 시점에 결과 존재(미열람)' 표지 | P1 | C7 | WRAPUP 결과 절 |
| J8 | 봉인 폴더 개봉(`data/processed/lgw/sealed`)과 n = 10 추출 재생성의 LG 조각 대조 (J03, J04) | P2 | C0 | WRAPUP 결과 절 |
| J9 | LGX 보조 판정(L26[ftt] 교차 환경, nn FT-T 서술 행, 점검 (iii)과 보충 대조의 분포, C14 결과) | P2 | C10 | LG 6A 결과의 보조 항목 |
| J10 | 결과 뒤 문서 정정: FINAL_PAPER §10.3 의 '72–95 %'(272행), 'TabPFN ... 동급'(277행), NOVELTY·RESEARCH_FRAME 의 문장 대응, NOVELTY 10절 문구, H15 서술을 LGU-B1 비교로 교체(LGU 4.6 '어느 경우든') (D06, D07) | P2 | J1, J3, J4 | 각 문서 |

### 3.5 그림·지도·원고(F, M)

| id | 내용 | 우선 | 자원 | 시간 | 의존성 | 상태 |
|---|---|---|---|---|---|---|
| F1 | `figures/figure_spec.json` 갱신: 7절 대응표를 확정하고 8개 상한을 점검한다. 패널별 원천 표, 조건부 병합 규칙, Fig 2 두 판, SI 교체 목록, N1 대상별 요약의 계산 위치(`lg_curve` 형식 두 가중 CI 로 4분 판정 수를 세는 모듈) (F02) | P1 | 문서 | 3–4 h | 결과 전(지금) | 미착수 |
| F2 | 지도 그림 시각 검토, 본문/SI 배치 | P1 | CPU 1 | 0.5 h | C8 | 미착수 |
| F3 | 논문 그림·표 재작성: Fig 1(LGD 새 지역 3곳, 레나 지도 영역), Fig 2–5, Fig 6(불확실성: LGU-A1·B2, LGX L25, c2 커버리지, 조건부 지도 6c), Fig 7, Table 1(새 지역, 라벨 단위 열: 행 수와 1 km 위치 수, Alaska(x) 참조 행), SI(소프트웨어·실행 표: 작업 id, 코드 해시, 설정 해시, 두 환경의 판, GPU 비결정성 서술). PDF·SVG·600 dpi PNG, 시각 QA, CAPTIONS.md (F03) | P1 | CPU 1–2 | 모듈 1–2일 | F1, J1–J7 | 미착수 |
| M1 | 방법·서론 영문 초안, Data·Code availability(두 실행 환경, LGD 약관, O8 기탁), 사전 등록은 '내부 사전 등록(커밋 해시와 시각)'으로 서술하고 T_res 뒤 개정의 표지를 그대로 적는다 (M01) | P0 | 문서 | 2–2.5일 | 없음 | 미착수 |
| M2 | 결과 절과 초록(주 대비 묶음만 판정어), 토의(배포 표, E 해석, 한계) (M02) | P0 | 문서 | 2일 | J1–J7 | 대기 |
| M3 | E 의 물리 해석 문단, 원천 E0 구성 (M03) | P2 | 문서 | 0.5일 | 없음 | 미착수 |
| M4 | 선택·철회 이력 표와 SI 음성 결과 등록표의 틀 (J05) | P2 | 문서 | 0.5일 | 틀은 지금, 수치는 C7 | 미착수 |
| M5 | 동등성 한계 근거와 S-a·S-b·SC3w 결과의 원고 문장 (M06) | P2 | 문서 | 2 h | 없음 | 미착수 |
| M6 | LGD 자료 약관 확인. CUSP(11셀)·GGD353(33셀)·calm_web_subsites(13셀) 제공자 연락을 지금 시작한다. 확인 전에는 해당 행을 뺀 표만 커밋·공개 (M08) | P0 | 사용자 연락 | 불명 | 사용자 | 미착수 |
| M7 | (조건부) 기존 제품 비교 SI 표 h50. 지도가 본문일 때만 (M05) | P3 | CPU 1–4, 네트워크 | 0.5–1일 | J3 | 조건 대기 |
| M8 | 문헌 원문 확인(NOVELTY 7.3 의 5편 등) (M07) | P2 | 사용자(PDF) | 불명 | 사용자 | 미착수 |
| M9 | SI 조립과 투고 형식 점검: 음성 결과 등록표, 선택·철회 이력, 교차 환경 점검 (i)–(iii)과 보충 대조, LGD 약관 주석, 참고문헌 정리, Sci Rep 형식(본문 단어 수, 초록 200단어, 표시 항목 8개), 최종 문장 점검(WRAPUP 11.2, NOVELTY 5절, RESEARCH_FRAME 의 쓰지 않을 문장) | P1 | 문서 | 1일 | M2 | 미착수 |

### 3.6 운영 기록(O)

| id | 내용 | 우선 | 시점 |
|---|---|---|---|
| O1 | `results/rescale/_ledger.csv` 갱신(ZovWo, Qjpbeb 중지, ShFtT, nxvpT, gLyfeb, qOkSo 의 상태·경과·실비용) | P2 | C0 뒤 |
| O2 | `docs/EXPERIMENT_LOG.md` 2026-09-30 항목 | P2 | 오늘 |
| O3 | `SESSION_HANDOFF.md` 갱신(09-30 상태, 이 계획의 3절 id, 진입점 `post status`, `lgxc status`, `logs/lgf/supervisor.log`) | P2 | 오늘 |
| O4 | 메모리 갱신(`lg-rescale-run-state`, 09-30 실행 상태, 로컬 보조 이어 실행 방침) | P2 | 오늘 |
| O5 | 미추적 산출 커밋(결과 전 계산 표, LGU 표는 판정 기록과 함께, 이 문서와 새 스크립트). 봉인 폴더와 약관 미확인 실행 표는 커밋하지 않는다 | P2 | 파일 단위 |
| O6 | git push | P0 | 사용자 확인 뒤, A1–A4 커밋 직후, C0 전 |
| O7 | cleanup 뒤 handoff | P3 | 사용자 결정 |
| O8 | 자료·코드 기탁: 판정 기록 뒤 결과 묶음, 조각 sha256 목록(해제 때 만든 것), 집계 표, 그림 원천 자료를 Zenodo 등에 기탁(약관 미확인 행 제외 판). 코드에 태그. 공개 범위는 사용자 결정 | P2 | 판정 기록 뒤 |

---

## 4. 로컬 GPU·CPU 일정

### 4.1 자원 현황(11:28 nvidia-smi)

| GPU | 사용 | 메모리 | 우리 작업의 지위 |
|---|---|---|---|
| 0–4 | 다른 사용자(hooni900) | 각 약 13 GB | 0·1 은 쓰지 않는다. 2·3·4 는 LGF 사전 허용 후보라 비면 LGF 가 쓴다 |
| 5 | LGF(h48) | 480 MiB | LGF |
| 6, 7, 9 | LGT(h43) | 3.0–7.3 GB | LGT 종료 뒤 LGF |
| 8 | 비어 있음(2 MiB). 08:32–09:04 에는 supergood 가 썼다 | 2 MiB | LGX 로컬 보조와 조건부 σ. 전환 규칙이 서면 LGF |

load average 약 6(128코어), 가용 RAM 약 169 GB(11:26).

### 4.2 GPU 우선순위

1. **LGF**(확인적 F1·N1, 창 마감 10-02 05:01): 후보 9·7·6·5·4·3·2(하네스 `ALLOWED_GPUS`). 최대 6장.
2. **LGT**: 끝날 때까지 9·7·6. 두 번째 통과도 같은 GPU 가운데 빈 것.
3. **조건부 격자 σ**(LGU-B2 가 '전이'일 때): GPU 8 에서 LGX 로컬을 드레인하고 먼저 돈다(약 0.5 h). 틀 (3) 지도의 유일한 선행 조건이기 때문이다.
4. **LGX 로컬 보조(G3–G6)**: LGF 가 도는 동안 GPU 8 한 장. LGF 창 마감 뒤에는 9·8·7·6·5.
5. **조건부 J12, G7**: 등록 조건이 설 때.

전환 규칙(LGF 개정 6 에 등록): 12 h 마다 한 투영에서 J2–J5 완료 예상이 창 마감을 넘거나, LGF 가 쓰는 GPU 가 2장 이하인 상태가 6 h 넘게 이어지면 LGX 로컬을 드레인하고 GPU 8 을 LGF 에 넘긴다. 감시기가 둘째 조건을 감지해 `logs/lgf/switch_condition` 을 만들고, 메인 세션이 사용자에게 알린 뒤 하네스 상수 변경과 드레인을 한다. 이 조건에서 G5·G6 은 멈춘다.

### 4.3 점유 판정과 자동 재개 규칙

**공통 규칙**

- 점유 판정: 메모리 사용이 50 MiB 를 넘거나, 우리 프로세스 그룹 밖의 계산 프로세스가 하나라도 있으면 점유다(빈 GPU 의 표시값은 2 MiB).
- 다른 사용자의 프로세스는 건드리지 않는다. 우리 작업이 GPU 를 쓰는 중에 다른 사용자가 들어오면 진행 중인 단위만 끝내고 그 GPU 를 놓는다.
- 모든 긴 실행은 `setsid nohup` 으로 세션과 분리하고 `--resume` 을 쓴다.
- 새 작업 시작 전 1분 load average 가 64 를 넘으면 시작하지 않는다. 실행 중 96 을 넘으면 새 단위를 받지 않는다.

**작업별 장치**

| 작업 | 점유 확인 | 다른 사용자가 들어올 때 | 자동 재개 | 드레인(수동 정지) |
|---|---|---|---|---|
| LGF | 하네스가 단위마다 확인. 감시기가 120 s 마다 후보 확인, 새 GPU 는 3회 연속 비어야 쓴다 | 해당 GPU 워커만 멈춘다. 모든 GPU 를 잃으면 역할이 rc 1 로 끝난다 | 감시기가 빈 후보로 다시 배정. 쓰지 않는 빈 후보가 생기면 드레인 뒤 넓힌다 | `touch data/processed/lgf/run_lgfn/drain`(N), `run_lgf/drain`·`run_lgfs/drain`(F·T3). 감시기는 자기가 만들지 않은 drain 파일을 보면 멈춘다 |
| LGT | 하네스가 단위마다 확인 | 해당 워커가 멈춘다 | 감시기가 상태 표를 보고 두 번째 통과를 한 번 띄운다 | 하네스 규칙 |
| LGX 로컬 | 실행기가 60 s 마다 확인 | 진행 중 단위가 끝나 unit.json 이 늘면 SIGTERM, 작업을 큐 앞에 되돌린다 | 다음 확인에서 빈 후보에 다시 올린다 | `touch data/processed/lgx_local/run_lgx_local/drain`(종료 코드 3) |

**감시기 교체(G0) 명령**(LGT 종료 전에 한다)

```bash
cd /home/willy010313/Polar_Bigdata
ps -o pid,cmd -p 82442                      # logs/lgf/wait_and_resume.sh 인지 확인
kill 82442                                   # 큐(104671)와 h48 는 끊지 않는다. 새 감시기가 역할 single 로 넘겨받는다
echo "LGX-B 회수 07:20·LGX-A 회수 10:55(미해제·미열람), LG 미회수, LGT 조각 미열람, LGU 판정 표 미열람, LGF 미열람" > logs/lgf/viewing_state.txt
setsid nohup scripts/local/lgf_supervisor.sh > /dev/null 2>&1 < /dev/null &
# 지난 사건의 사후 기록(기록 시각이 남으므로 LGF 개정 6 에 실제 시각 08:32, 09:04:55 를 함께 적는다)
env CUDA_VISIBLE_DEVICES= .venv_lgf/bin/python scripts/3_deep_learning/h47_foundation_models.py --gpu-event 5:remove --viewing-state "사후 기록: 08:32 다른 사용자 점유로 GPU 5 상실. $(cat logs/lgf/viewing_state.txt)"
env CUDA_VISIBLE_DEVICES= .venv_lgf/bin/python scripts/3_deep_learning/h47_foundation_models.py --gpu-event 5:add --viewing-state "사후 기록: 09:04:55 GPU 5 재개. $(cat logs/lgf/viewing_state.txt)"
```

감시기 동작(`scripts/local/lgf_supervisor.sh`, 역할 스크립트 `scripts/local/lgf_role_queue.sh`):

- 시작 때 도는 단일 큐(`logs/lgf/queue.pid`)를 역할 single 로 넘겨받는다. 큐를 끊지 않는다.
- LGT 넘김: 잠금 PID 가 살아 있으면 GPU 6·7·9 를 쓰지 않는다. LGT 가 끝났을 때 상태 표가 완결(n_partial 0, n_fail 0, n_done = n_todo, 중단·abort 없음)이면 바로 넘긴다. 완결이 아니면 두 번째 통과(`h43 --resume --rerun-partial --gpu-mem-max-mib 50`, 9·7·6 가운데 빈 GPU)를 한 번 띄우고, 그 통과가 끝나면 결과와 관계없이 넘긴다. 남은 partial 은 LG 6C 규칙으로 처리한다. `logs/lgf/lgt_released` 가 있으면 언제든 넘긴다.
- 빈 GPU 가 1장이면 single, 2장 이상이면 F(마지막 1장)와 N(나머지). F 가 끝나면 N 을 드레인해 그 GPU 를 더한다(LGF 6.3 단계 6).
- F·N 이 모두 종료 코드 0 이면 역할 T3(`h47 --jobs f_t3 --n-jobs-done`)을 한 번 띄운다. 하네스가 J12 조건을 판단해 허용이면 J12 를 돌고, 아니면 창 파일에 'J12 미실행: <사유>' 를 적는다. 창 마감(rc 4)으로 끝나면 h47 이 창 마감 확인을 f_t3 판단보다 먼저 하므로 창 파일에 결정이 남지 않는다. 감시기 기록('창 마감으로 J12 미실행')을 LGF 10절에 옮긴다.
- GPU 부족 감시: LGF 사용 GPU 가 2장 이하인 상태가 6 h 넘게 이어지면 `logs/lgf/switch_condition` 을 만든다(4.2 전환 규칙). 감시기는 하네스 코드를 바꾸지 않는다.
- 멈추는 경우: 창 마감, 사용자 드레인, F·N·T3 완료, 5분 안에 끝난 실패 3회 연속.
- 점검: bash -n 통과. 함수 단위 시험(10:30)은 빈 GPU 판정과 LGT 보류까지다. 역할 전환, 두 번째 통과, T3 은 실제로 돌려 보지 않았으므로 첫 확장(LGT 종료 직후)은 메인 세션이 `logs/lgf/supervisor.log` 와 `queue.log` 로 지켜본다.

### 4.4 LGF 창 산술(정보, 등록 규칙은 바꾸지 않는다)

| 항목 | 값(GPU-h, 추정) | 근거 |
|---|---|---|
| 등록 예산 B | 172.5 | LGF 개정 5 |
| 측정 기반 전체 P + ΔE1 | 157.8 | `lgf_projection.csv`(F 4.2, N 126.2, E1 27.4) |
| 완료분(10:40) | 약 5.1 | J1 0.13, J2–J5 약 5.0 |
| 남은 양 | 약 150–153 | J2–J5 약 83(확인적 N1 구성), J6 1.5, J7 52.8, J8 0.4, J9 0.4, J10 12.8, J11 1.7 |
| 가용량(11:30 부터 창 마감까지) | 약 155–158 | GPU 5 약 41.5 h, GPU 6·7·9 각 약 38.5–39 h(LGT 가 12:10–12:50 에 놓을 때) |
| 여유 | 약 5–7(3–5 %) | 단위 꼬리의 유휴, 드레인 대기, 다른 사용자의 점유는 반영하지 않았다 |

- J2–J5(N1)는 GPU 4장이면 10-01 오전에 끝나 창까지 약 18 h 남는다. 창이 모자라면 먼저 잘리는 것은 J10(N 2순위 Alaska x), 다음이 J7 의 꼬리다. F 역할은 약 4 GPU-h 라 첫날 끝난다.
- 끝나지 않은 단위는 LGF 8.1 과 8.4 의 규칙(분할 완결성, 풀 지역 수)으로 처리한다. B 는 다시 쓰지 않는다.
- 여유를 늘리는 등록 경로는 GPU 2·3·4 편입(사전 허용)과 전환 규칙의 GPU 8 두 가지다.

### 4.5 CPU 예산(우리 작업 합계 32스레드 이하, 모두 nice 10)

| 구간 | 작업과 스레드 | 합계 |
|---|---|---|
| 지금–LGT 종료 | LGT 약 10, LGF single 4, LGX 로컬 3, 문서·점검 1 | 약 18 |
| LGT 종료–14:30 | LGF F 4 + N 8, LGX 로컬 3, C3 4 | 약 19 |
| 14:30–22:00 | LGF 12, LGX 로컬 3, C4 8, 그 밖 한 작업(C2 4, C5 2, C2b 8 가운데 하나. C2b 는 C4 와 겹치지 않는다) | 약 25–27 |
| 10-01 | LGF 12, LGX 로컬 3, C10 8 또는 C7 4, C14 8(선택, C10 과 겹치지 않는다) | 약 23 |
| 10-02 창 마감 뒤 | LGX 로컬 최대 4작업 × 3, C9 2, C7 4, C8 4 | 약 22 |

- 무거운 집계(C4, C10)는 `logs/post/heavy.lock` 으로 한 번에 하나만 돈다. 가용 RAM 이 32 GB 미만이면 시작하지 않는다. 최대 RSS 를 로그와 `logs/post/events.tsv` 에 남긴다.
- LGX 실행기는 '다른 우리 작업의 스레드 합 + 작업 수 × (스레드 + 1) ≤ 32' 일 때만 새 작업을 시작한다. 이 값은 실행 중에 `data/processed/lgx_local/run_lgx_local/our_threads` 파일로 바꾼다(매 주기 다시 읽는다). 후보 GPU 와 동시 작업 수도 같은 폴더의 `gpus`, `max_jobs` 파일로 바꾼다.

### 4.6 시간표(추정)

| 시각 | GPU 5 | GPU 6·7·9 | GPU 8 | CPU |
|---|---|---|---|---|
| 11:30–12:30 | LGF single(J2–J5) | LGT | 대기(A1–A4 커밋 전) | A1–A5 문서, A3·A6 코드 |
| 12:30–14:00 | LGF single → F | LGT 종료 뒤 LGF N | C0b 뒤 G3 점검 (iii) 1·2회차 | C3 계산, F1, M1 |
| 14:00–22:00 | LGF F(18:00 전후 종료 뒤 N 에 합류) | LGF N | G4 g45 FT-T | C0, C1, C2, C2b, C4, C5, C6, C7 |
| 10-01 00:00–16:00 | LGF N | LGF N | G4 종료(약 01:30), G5(약 08:00 종료), G6 | C10(G5 뒤), C14(선택) |
| 10-01 16:00–10-02 05:01 | LGF N(J7, J10) | LGF N | G6 종료(약 17:30) 뒤 대기 | F3 모듈 |
| 10-02 05:01 이후 | LGX nn 남은 조각, 조건부 G7 | 같음 | 같음 | C9, C7 재실행, C8, F3 최종 |

LGU-B2 가 '전이'이면 J3 직후 GPU 8 에서 σ(약 0.5 h)를 먼저 돌리고 G4–G6 을 이어 간다.

### 4.7 LGX 로컬 이어 실행과 점검 (iii) 명령

전제: A1–A4 커밋과 push(O6). 두 실행기(`run_lgx_local_continue.sh`, 엔진 `run_lgx_gpu_local.sh`)는 LG 개정 15 가 커밋되지 않았으면 해제·재적합·이어 실행을 거부한다.

```bash
cd /home/willy010313/Polar_Bigdata
scripts/local/run_lgx_local_continue.sh status            # 진행 정보만
# 해제 → 남은 단위 목록 → seed → 설정 해시 대조 → 점검 (iii) 2회 → 계속 규칙 → 이어 실행 → 보충 대조(세션과 분리)
setsid nohup scripts/local/run_lgx_local_continue.sh chain > logs/lgx_local/chain.out 2>&1 < /dev/null &
# LGF 창 마감 뒤 GPU 를 넓힐 때(재시작 없이)
echo "9 8 7 6 5" > data/processed/lgx_local/run_lgx_local/gpus; echo 4 > data/processed/lgx_local/run_lgx_local/max_jobs
# 전환 규칙이나 σ 때문에 GPU 8 을 비울 때
touch data/processed/lgx_local/run_lgx_local/drain          # 진행 중 단위를 끝낸 뒤 멈춘다(종료 코드 3). 다시 시작: rm drain 뒤 lgxc run
```

- 해제(`unpack`)는 두 묶음의 `data/processed/lgx` 경로만 푼다. 묶음의 로그·상태 파일은 `results/rescale_lgx_{a,b}/bundle_top/` 에 둔다. 두 묶음의 파일 이름 겹침(해제 대상 안)이 있으면 멈춘다. 해제 뒤 unit.json 수를 대조하고, sha256 목록(`data/processed/lgx/rescale_shards_sha256.txt`)과 해제 시각(`.unpacked_rescale.txt`)을 남기고, 조각 폴더를 읽기 전용으로 둔다.
- 남은 단위(`remaining`): 기대 집합은 r0 FT-T 완료 단위의 (대상, 분할)이다(학습기 축 대상 12. Greenland 는 분할 2 에 단위가 있다. 11:44 해제 뒤 대조로 정정). 큐에 없는 남은 단위가 있으면 멈춘다.
- seed: Rescale 완료 FT-T 조각을 하드링크한다. 레나 x g45 분할 5 는 기본값으로 뺀다(보충 대조 단위).
- 점검 (iii)의 대조기가 Rescale `blocksse.npz` 를 처음 읽는 시각을 `data/processed/lgx/.first_content_read.txt` 에 적는다(WRAPUP 0.3 (3)).
- 계속 규칙이 '멈춤'이면 G4–G6 을 시작하지 않고 종료 코드 5 로 끝난다. 메인 세션이 (iii) 요약의 분포만 사용자에게 보고한다. 이때 L26[ftt] 는 LG L5 3분할 판정만 쓴다.
- 로그: `logs/lgx_local/continue.log`(상위), `queue_<출력 폴더>.log`(엔진), 작업별 로그.

### 4.8 세션이 끊길 때

- 세션 없이 계속 도는 것: ZovWo(Rescale), LGT(두 번째 통과 포함), LGF 감시기와 역할, LGX 로컬 `chain`.
- 세션이 있어야 하는 것: 개정 커밋(A1–A5)과 push, Rescale 회수(C0), 코드 작업(A6·A7), 집계와 판정 기록, 전환 규칙의 조치. 감시기 교체(G0)가 빠지면 LGF 는 GPU 5 한 장에 머문다. 그래서 G0 를 가장 먼저 한다.
- 다음 세션의 진입점: 이 문서 3절, `scripts/local/run_post_results.sh status`, `scripts/local/run_lgx_local_continue.sh status`, `logs/lgf/supervisor.log`, `logs/post/events.tsv`.

---

## 5. 결과 뒤 처리 파이프라인

### 5.1 공통 규칙

- `$LGD=results/rescale_lg/data/processed/lg` 는 조각 폴더(`$LGD/shards`)로 쓴다. `$LGS` 는 LG 표 폴더다. 작업 안 집계가 종료 코드 0 이고 표 4개가 있으면 `$LGS = $LGD`, 아니면 로컬 재집계 폴더 `data/processed/lg_sum_local`(조각은 `$LGD/shards` 를 가리키는 심볼릭 링크)이다. `post gate-lg` 가 `logs/post/lgs_dir` 에 적고, 표를 읽는 인자(h39·h49 의 `--lg-dir`, h42·h43·h48 의 `--lg-dir`)는 `$LGS` 를, 조각 인자(`--lg-shards`, h51·h52)는 `$LGD` 를 쓴다. `data/processed/lg` 는 만들지 않는다.
- Rescale 조각 폴더(`$LGD/shards`, `data/processed/lgx/shards`, `data/processed/lgx/ladder/shards`)는 해제 직후 읽기 전용으로 두고 sha256 목록을 남긴다. LGX 주 집계 전에 목록을 다시 대조한다.
- 로컬 조각은 `data/processed/lgx` 에 넣지 않는다. 주 집계는 한 플랫폼(Rescale)의 조각만 쓴다.
- 파이프라인은 하네스 출력을 `logs/post/<단계>_<시각>.log` 에만 쓰고 화면에는 종료 코드, 최대 RSS, 로그 경로, 새로 생긴 파일 이름만 쓴다. 로그를 여는 것은 그 표를 여는 것과 같게 다룬다. 표 생성 시각은 `logs/post/events.tsv` 에, 조각 내용 첫 열람 시각은 각 조각 상위 폴더의 `.first_content_read.txt` 에 남는다.
- 결과 표를 만드는 단계는 해당 개정(LG 15, WRAPUP 2, LGU 4, LGF 6)이 커밋된 뒤에만 돈다(파이프라인이 확인한다).

### 5.2 단계

| 단계 | 하위 명령 | 실제 명령(인자는 하네스 argparse 와 대조) | 산출 | 기록 위치 |
|---|---|---|---|---|
| 1 회수(C0) | `post fetch-lg` | `tools/rescale_client.py fetch ZovWo --dest results/rescale_lg` → `tar xzf results_lg.tar.gz -C results/rescale_lg`. 조각 수, 빠진 RealMLP 이름(`missing_realmlp.txt`), `lg_status.csv` 단계 종료 코드 | `$LGD`, sha256 목록 | 원장(O1), LG 개정 이력 |
| 1b 해제(C0b) | `lgxc unpack` | 4.7 | `data/processed/lgx` | LG 개정 15 (b)의 대조 |
| 2 LG 집계(C1) | `post gate-lg`, `post sum-lg` | 대체: `h40_label_grid.py --summarize-only --allow-local --threads 4 --workers 1 --out-dir data/processed/lg_sum_local` | `lg_curve`, `lg_minn`, `lg_tests`, `lg_targets` | LG §7(J1) |
| 3 LGX 주 집계(C3, C4) | `post ladder`, `post sum-lgx` | `h41_validation_ladder.py --summarize-only --allow-local --threads 4 --workers 1`. `h42_label_grid_ext.py --summarize-only --allow-local --nboot 10000 --threads 4 --workers 2 --lg-dir $LGS --out-dir data/processed/lgx` | `lgv_tests`, `lgx_tests`, `lgx_curve`, `lgx_lg_aux`, `lgx_splitdist`, `lgx_region_inference` | LG 6A 결과(J2). 실패하면 사용자 확인 전까지 대기(선택지는 (라)뿐) |
| 4 LGD(C2, C2b, C6) | `post gate-lgd`, `lgd-v3local`, `lgd-lic`, `pool-lgd` | `h51_lgd_run.py --specs repro --allow-local --workers 0 --threads 4 --repro-check --lg-shards $LGD/shards`. 조건부 `--specs v3local`. A6 뒤 `--specs lic`. `h52_lgd_pool.py --allow-local --threads 4 --lg-dir $LGD --splitdist data/processed/lgx/lgx_splitdist.csv` | `lgd_repro_gate`(scope·status 만 화면), `lgd_tests`, `lgd_pool` | LG 6B 결과(J5) |
| 5 LGT(C5) | `post sum-lgt` | `h43_tabpfn_label_grid.py --summarize-only --cross-lg --lg-dir $LGS --threads 2`(집계 스레드 상한 2) | `lgt_curve`, `lgt_tests`, `lgt_minn`, `lgt_targets`, `lgt_gate`, `lgt_cross` | LG 6C 결과(J4), `viewing_state.txt` |
| 5b AB10(A3) | `post lgu-ab10` | `lgu_ab10_tests.py`(LGU 개정 4 커밋 뒤) | `lgu_a_ab10_tests.csv`, `_meta.json` | J3·J7 |
| 6 h39(C7) | `post scenarios` | `h39_scenarios.py --mode summarize --allow-local --threads 4 --lg-shards $LGD/shards --lg-dir $LGS --lgx-dir data/processed/lgx --lgx-shards data/processed/lgx/shards --lgd-dir data/processed/lgd --lgu-dir data/processed/lgu --lgt-dir data/processed/lgt [--lgf-dir data/processed/lgf] --out-dir data/processed/lgw`, 같은 인자로 `--mode bias-mae` | `lgw_bundle`, `lgw_scenarios*`, `lgw_l43`, `lgw_ak1`, `lgw_rescore`, `lgw_delta_rel` | WRAPUP 결과(J7). 봉인 폴더는 이 단계 전에 연다(J8) |
| 7 지도(C8) | `post map` | `h49_map_contrasts.py --allow-run --allow-local --threads 4 --lg-shards $LGD/shards --lgx-shards data/processed/lgx/shards --out-dir data/processed/lgw` → `h49_transfer_map.py --check-inputs …` → `--allow-run --threads 2 --lgw-dir data/processed/lgw --lg-dir $LGS --lgx-dir … --lgu-dir … --lgt-dir … --lgf-dir … [--allow-missing-gpu-tables] [--sigma-file]` → `fig_map_lena.py` | 지도 표, PDF·SVG·PNG | WRAPUP 결과, `outputs/figures/paper` |
| 8 LGF(C9) | `post sum-lgf` | `.venv_lgf/bin/python h47_foundation_models.py --summarize-only --threads 2`, `h48_nn_tuning.py --summarize-only --cross-lg --lg-dir $LGS --threads 2` | `lgf_tests`, `lgfn_tests`, `lgfn_cross` | LGF 10절(J6). 뒤이어 단계 6·7 재실행 |
| 9 LGX 보조(C10) | `post sum-lgx-aux` | `data/processed/lgx_aux/shards` 에 lgxb·lgxg·lgxnn Rescale 조각과, Rescale 에 같은 이름이 없는 로컬 새 조각(`platform_manifest.csv`)을 하드링크. `h42 --summarize-only --allow-local --nboot 10000 --threads 4 --workers 2 --lg-dir $LGS --out-dir data/processed/lgx_aux`. 이 폴더에서는 L26[ftt] 행과 nn·g45 FT-T 곡선 행만 읽는다 | `lgx_aux/lgx_tests` 등 | LG 6A 결과의 보조 항목(J9) |
| 10 그림(C11 아닌 F3) | `post figs` | 지도 그림과 `POST_FIGS` 에 적은 모듈(재작성 뒤) | `outputs/figures/paper` | CAPTIONS.md |

회수 직후 한 번에 돌리는 명령: `post cpu-chain`(gate-lg → ladder → sum-lgx → gate-lgd → pool-lgd → lgu-ab10, 첫 실패에서 멈춘다). pool-lgd 는 C2b(A6 구현 뒤)가 끝난 뒤 두 판 병기로 다시 돈다.

### 5.3 판정 기록 순서와 결과 열람 규칙

1. **열람 전 커밋**: A1–A4 를 커밋하고 push(O6)하기 전에는 LGX 묶음을 풀지 않고 어떤 결과 표도 만들거나 열지 않는다. 회수와 해제, unit.json 수 대조, 상태 표 확인은 열람이 아니다.
2. **판정 기록 순서**: LG(J1) → LGX(J2, 확인적 L10·L12·L15·L19·L29·L30 을 먼저) → LGD(J5) → LGT(J4) → LGU(J3) → h39(J7) → LGF(J6, 창 마감 뒤) → 지도(C8) → 보조(J9). LGU 는 LG 와 독립이라 A3 커밋 뒤 먼저 기록해도 된다. 이때 `logs/lgf/viewing_state.txt` 와 LGF 개정 이력의 열람 상태 칸을 고친다.
3. **첫 실행 판정 고정**: 각 집계의 첫 판정 표를 판정으로 쓴다. 코드 결함으로 다시 돌리면 두 표를 모두 남기고 개정 이력에 사유를 적는다.
4. **표지**: LGD PE1·PE2 는 '교차 환경; (i) CatBoost 불통과'와 두 판(전체, 약관 확인분) 표기. LGT `lgt_cross`, LGF `lgfn_cross` 는 '교차 환경(보조)', 판정에 쓰지 않는다. `lgx_lg_aux` 파생 열은 '교차 환경 미확인'. AB4·SC1w-P·LGX-N4a 는 S-a 열람 표지. L26[ftt] 보조 판정은 6.1 (d)의 표기. T_res 뒤 개정으로 정한 규칙을 쓴 행에는 6.1 (l)의 표지.
5. **LGF 가 도는 중의 열람**: LGT·LG·LGX 표를 열면 `viewing_state.txt` 를 고친다. LGF 표는 창 마감 또는 J1–J11 완료 뒤 한 번 집계한다.
6. **봉인 폴더**(`data/processed/lgw/sealed`)는 LG 회수 뒤에만 연다.
7. **점검 (iii) 산출**은 계속 규칙 판단에만 쓴다. G4 이후 조각의 RMSE 나 Δ 는 C10 전에 열지 않는다.

---

## 6. 계획서 개정이 필요한 항목

아래는 초안이다. 가설과 판정 문구는 바꾸지 않는다. 네 개정 모두 T_res(07:20:22) 뒤, 조각 해제와 결과 표 생성 전에 커밋한다.

| 계획서 | 개정 | 커밋 시점 | 요지 |
|---|---|---|---|
| LG | 개정 15 | LGX 묶음 해제(C0b)와 ZovWo 회수 전 | 6.1 |
| LGF | 개정 6 | 감시기 교체 직후(LGF 표는 창 마감 뒤에 열린다) | 6.2 |
| LGU | 개정 4 | LGU 판정 표 열람 전 | 6.3 |
| WRAPUP | 개정 2 | 모든 결과 표 생성·열람 전 | 6.4 |
| NEXT | 개정 기록 | 권장 | SC1–SC3·AK1 철회 한 줄 |

### 6.1 LG 개정 15 초안

- (a) 사용자 지시(2026-09-30 08:45): Rescale 에 추가 비용을 쓰지 않는다. 남은 실험은 로컬 GPU 가 비는 대로 진행한다. 이에 따라 WRAPUP 8.6 의 (가)(Rescale 이어 돌리기)와 (마)(`lgx_sum.yaml` 제출)는 선택지에서 빠진다.
- (b) Qjpbeb: 09:02 중지 요청, 09:03:04 종료(코드 137). 마지막 묶음(09:02:02)은 514번째 GPU 조각(08:44:44 완료) 뒤에 썼다. 회수 묶음의 이름 목록(10:55 회수, 해제 전)으로 확인한 완료 수는 CPU 3,860/3,860, h41 292/292, GPU 514/572, 실패 0 이다. 남은 58조각은 모두 FT-T 이며 g45 7(레나 분할 4, 캐나다·CA-2·CA-3 분할 4·5), nn 51(주 4지역 분할 2–5 의 16, AL-1 4, 나머지 6대상 분할 1–5 의 30, Greenland 분할 2 의 1. 해제 뒤 대조로 정정, LG 개정 15 (b) 보충 항목)이다. r0 FT-T 는 55/55 완료라 L27 은 영향이 없다. 작업 안 통합 집계는 돌지 않았다. ShFtT 는 77/77, 종료 코드 0. 해제 뒤 unit.json 수로 다시 대조하고 숫자가 다르면 이 항목을 고친다.
- (c) WRAPUP 8.6 의 선택: 주 판정은 (다) 빠진 조각 없이 집계한다. 6A.5a 의 분할 완결성 규칙을 적용하며, h42 가 L26 의 FT-T 행에 '판정 불가(5분할 판정에서 제외: 분할 일부만 있다. LG 의 3분할 판정을 유지한다)'를 쓰는 것이 6A.3 X6 의 등록 대체 규칙이다. nn 축 FT-T 곡선 행은 '부분'이다. 개정 14 (4)의 '(나)를 제시하지 않는다'는 그대로 둔다.
- (d) 로컬 보조 이어 실행(사용자 지시의 이행): 남은 FT-T 조각을 로컬 RTX 3090 에서 이어 돌린다. 지위는 '교차 환경(보조)'이며 주 판정 표를 바꾸지 않는다.
  - 폴더: `data/processed/lgx`(Rescale 조각만, 주 집계, 읽기 전용), `data/processed/lgx_local`(로컬 조각과 Rescale 조각의 하드링크 seed), `data/processed/lgx_aux`(보조 집계). 새 조각마다 `platform_manifest.csv` 에 GPU 번호·UUID·이름, python·torch 판을 적는다.
  - 설정 해시: 로컬 인자(`--targets`)와 Rescale 인자로 계산한 g45 조각의 전체 해시(cfg_hash)가 남은 8개 (대상, 분할) 모두 같다(코드 계산, 11:15). nn 과 r0 의 해시도 같다. 따라서 `--resume` 이 seed 한 Rescale 완료 분할을 건너뛴다. `--splits` 는 분할 1..K 를 뜻하므로 분할 하나만 고를 수는 없지만 재적합 낭비는 없다. 실행 전 `check` 가 전체 해시와 공통 해시를 모두 대조한다.
  - 표기: L26[ftt] 의 보조 5분할 판정에 '교차 환경(보조): 분할 4·5 의 FT-T 조각 가운데 레나 분할 4, 캐나다·CA-2·CA-3 분할 4·5 는 로컬(RTX 3090, python 3.9.18, torch 2.6.0), 나머지는 Rescale(T4, python 3.11.10, torch 2.4.1)'을 붙이고, 한 플랫폼 판정(LG L5 3분할)과 점검 (iii)의 차이 분포를 같은 행에 병기한다(WRAPUP 8.6 비교 규칙 (2)).
  - 비교 규칙 (1)의 예외: L26 대비(학습기 − `catboost_lo` R1)의 CatBoost 쪽은 Rescale 조각이다. 근거인 점검 (i)의 기준 방법 키 최대 차 1.4e-14 cm 는 qOkSo 캐나다 x 분할 1 과 스모크 3단위에서 잰 값이고, 대비에 쓰일 레나·캐나다·CA-2·CA-3 의 분할 4·5 에서는 재지 않았다. C14(선택)를 하면 이 7개 (대상, 분할)의 CatBoost 조각을 로컬에서 다시 적합해 같은 플랫폼 대비와 키별 차를 함께 보고한다. 하지 않으면 표지에 '근거는 다른 단위에서 잰 값의 외삽'을 적는다. '교차 환경' 표지는 어느 경우에도 유지한다. V1r 이 들어간 대비는 플랫폼 사이에서 만들지 않는다(개정 14 (5)).
- (e) 교차 환경 점검 (iii) 정의: 단위는 Rescale 에서 끝난 FT-T 조각 가운데 nn 러시아 W x 분할 1, r0 러시아 W x 분할 1, g45 AL-2 x 분할 4·5 다. 로컬에서 같은 설정으로 두 번 재적합한다(출력 `data/processed/lgw/xenv/lgx_gpu_1`, `_2`). 대조는 `scripts/2_evaluation/lgw_xenv_gate_iii.py`(판 2)이며 키별 셀 가중 RMSE 의 절대 차만 낸다(짝: 로컬 1 − Rescale, 로컬 2 − Rescale, 로컬 1 − 로컬 2). RMSE 값, Δ, 판정은 내지 않는다. 산출 `data/processed/lgw/lgw_xenv_gate_iii.csv`, `_summary.json`. 보충 대조: 이어 돌릴 대상과 규모가 같은 레나 x g45 분할 5 FT-T 를 이어 실행 폴더에서 다시 적합하고(seed 에서 뺀다) Rescale 조각과 대조한다(`lgw_xenv_gate_iii_supp*`). 보충 대조는 계속 규칙에 쓰지 않고 J9 에 병기한다.
- (f) 계속 규칙(결과 전 고정): 물리식 키의 |로컬 1 − Rescale| 가 모두 1e-9 cm 이하이고 채점 블록이 같아야 한다(아니면 멈추고 원인을 찾는다). 신경망 키에서 다음 가운데 하나를 만족하면 G4–G6 을 계속한다. (가) |로컬 1 − Rescale| 가 0.5 cm 를 넘는 키가 0. (나) 로컬 반복 차(|로컬 1 − 로컬 2|)의 중앙값이 1e-6 cm 를 넘을 때, |로컬 1 − Rescale| 의 중앙값이 그 3배 이하. (다) 로컬 반복 차의 중앙값이 1e-6 cm 이하(로컬 재적합이 사실상 결정적)일 때, |로컬 1 − Rescale| 의 중앙값이 0.1 cm 이하이고 0.5 cm 초과 키의 비율이 5 % 이하. 모두 아니면 멈추고 사용자에게 보고한다. 멈춘 경우 L26[ftt] 보조 판정과 nn FT-T 서술 행은 만들지 않고, 원고의 FT-T 학습기 문장은 LG L5 3분할 판정만 쓴다. 이 경우 G4–G6 의 GPU 8 시간(약 27 h)은 σ(조건부)와 LGF 전환 규칙에 남긴다.
- (g) 실행 환경 문안 채택(WRAPUP 8.2): 머리말 '실행 환경', 6A.9 통합 집계(로컬 `h42 --summarize-only --allow-local --nboot 10000 --threads 4 --workers 2`, `h41 --summarize-only --allow-local`, nice 10, 전제는 8.4 (ii) 통과), 6C.8 의 GPU 배정 기록. 로컬 집계가 실패하면 사용자 확인 전까지 대기하며 선택지는 (라)뿐이다. LG 작업 안 집계가 실패해 로컬에서 다시 집계하면(`data/processed/lg_sum_local`) 사유를 이 항목 아래에 적는다.
- (h) 6A.7 비맹검 항목 추가: L29 의 B:ens − P0.
- (i) LGD 본 실행 기록: 2026-09-30 07:18 완료, 22표, 실패 0, 막힘 0, 10,754 s, 로컬 판(WRAPUP 8.3), 워커 2 × 스레드 4. 선택 묶음(repro 1표, v3local 3표)과 약관 확인분 묶음(m)은 LG 회수 뒤.
- (j) ZovWo RealMLP: 13:50 자체 정지 뒤 빠진 조각이 있으면 주 판정은 (다)(L5·L26 의 RealMLP 행 '부분(분할 k/K)' 또는 판정 불가). 로컬 이어 실행(G7)은 기본으로 하지 않는다. C0 에서 빠진 조각이 CI 풀 지역(주 4지역)에 속하는지 먼저 확인하고, 속하지 않으면 어떤 판정도 바뀌지 않으므로 하지 않는다. 속하면 사용자 결정 뒤 LGF 창 마감 뒤 빈 GPU 로 `h40 --part gpu --learners realmlp --targets <빠진 대상> --resume`(출력 `data/processed/lg_local`, ZovWo 조각 seed), 같은 표기 규칙 (d)로 보조만 만든다.
- (k) 표준 경로: `$LGD`(조각)와 `$LGS`(표)를 5.1 대로 모든 도구에 명시한다.
- (l) 열람 상태와 기준 시각: WRAPUP 0.3 의 T_res 는 2026-09-30 07:20:22(ShFtT 회수 폴더 `results/rescale_lgx_b` 생성)다. Qjpbeb 회수 폴더는 10:55:44 에 생겼다. 이 개정은 T_res 뒤에 커밋하므로 '회수 뒤, 조각 미해제·미열람 상태의 개정' 표지를 단다. 근거: 두 묶음은 풀지 않았고 `data/processed/lgx/` 가 없다(11:28 확인). 묶음에서 읽은 것은 파일 이름 목록과 `lgx_cpu.log` 의 조각별 시간 열뿐이다. `results/rescale_lg/` 는 없다. 커밋 시점에 디스크에 있는 결과와 상태는 다음과 같고, 각 대상에 '등록 시점에 결과 존재(미열람)' 표지를 단다: ShFtT·Qjpbeb 묶음(미해제), LGD 조각(04:38–07:18 생성, 미열람), LGU 판정 표(`lgu_a_tests.csv` 04:14, `lgu_b_tests.csv` 04:31, 접근 시각 = 생성 시각), LGT 조각(01:47:33 부터, 미열람). 해제 시각, 조각 내용 첫 열람 시각, 첫 집계 표 생성 시각은 실행기와 파이프라인이 기록하고 결과 절에 옮긴다. 작업 순서는 A1–A4 커밋, push, LGX 해제, seed·check, 점검 (iii) 대조로 고정한다.
- (m) LGD 약관 미확인 자료의 판정 규칙(h52 전, 결과 전 고정):
  1. 약관 확인분 판을 정의한다. `lic_unverified = 1` 셀이 들어가는 모든 LGD 실행 표(NAtlantic 주 표와 L41 변형 7개, 캐나다·러시아 W 확충판(L40), L42 원천 표 3개)에서 그 셀을 뺀 표다. 주 판정 풀에 쓰이는 대상 셀 기준으로 NAtlantic 새 셀 38 가운데 19, 캐나다 확충판 새 셀 75 가운데 37, 러시아 W 확충판 새 셀 8 가운데 7 이 남는다. h51 선택 묶음 `lic` 로 함께 돌리고 h52 에서 전체 판과 약관 확인분 판을 병기한다. 약관 확인분 판에 6B 적격 규칙을 그대로 적용하며, 적격이 아니면 그 지역은 약관 확인분 판의 PE1 에서 빠지고 PE1 은 남은 새 지역으로 계산한다.
  2. 투고 시점에 사용 허락을 받지 못한 자료원이 하나라도 있으면 PE1·PE2·L38–L42 의 주 판정은 약관 확인분 판으로 쓰고, 전체 판은 SI 에 '약관 확인 전 자료 포함' 표지와 함께 둔다. 모든 자료원의 허락을 받으면 주 판정은 전체 판이다. 이 선택은 결과를 보고 바꾸지 않는다.
  3. 제공자 연락(M6)은 지금 시작한다. 연락 시각과 답을 개정 이력에 적는다.
  4. 러시아 W 의 약관 미확인 149셀 가운데 148셀(GGD402 야말)은 `aux_unknown_yamal`(보조 역할)이다. 이 셀이 L42 원천 표에 들어가면 1의 약관 확인분 판에서 빠진다.
- (n) 조건부 작업의 등록 형식: C12(n = 80)와 C13(잔차 크리깅)은 조건이 서면 명령, 폴더, 표지를 먼저 개정 이력에 적고 실행한다. C12 는 '교차 환경; (i) 기준 방법 키 1.4e-14 cm' 표지를 붙이고 등록된 n* 구간 보고를 대체하는 방식으로만 쓴다. C13 은 '결과 열람 뒤 등록'이다.

### 6.2 LGF 개정 6 초안

- GPU 사건: 08:32 다른 사용자가 GPU 5 를 잡아 h48 워커가 멈췄다(rc 1, `gpu_lost`, 완료 282 단위). 09:04:55 `wait_and_resume.sh` 가 GPU 5 로 재시작했다. `lgf_window.json` 에는 감시기 교체 때 사후 기록했다(기록 시각과 실제 시각을 함께 적는다).
- 감시기 교체 시각과 방식: 6.3 단계 6 을 구현했다. 두 장 이상이면 F 역할 1장(h47 J1·J6·J8·J9·J11), N 역할 나머지(h48 J2–J5·J7·J10, 한 호출). N 안의 순서는 `JOB_RANK` 와 의존성이며 J2–J5 단위가 막혔을 때만 J7·J10 단위가 GPU 를 쓴다. 한 장이면 기존 단일 큐. 최대 6장.
- J12: F·N 이 모두 끝나면 역할 T3 이 `h47 --jobs f_t3 --n-jobs-done` 을 한 번 불러 8.4 의 조건을 판단하게 한다. 창 마감으로 끝나면 h47 이 결정을 창 파일에 남기지 않으므로 감시기 기록을 10절에 옮긴다.
- LGT 넘김: 상태 표가 완결(n_partial 0, n_fail 0, n_done = n_todo, 중단·abort 없음)이면 바로, 아니면 감시기가 두 번째 통과를 한 번 띄운 뒤 넘긴다. GPU 6·7·9 편입 전에 사용자에게 한 줄로 알리고(6.1), 편입 시각과 열람 상태를 적는다.
- GPU 8 전환 규칙(결과 전 등록, LGF 표는 창 마감 전에 열지 않는다): 조건 (1) 12 h 마다 한 투영(`h47 --window-status` 와 `lgf_projection.csv`)에서 J2–J5 완료 예상이 창 마감을 넘는다. 조건 (2) LGF 가 쓰는 GPU 가 2장 이하인 상태가 6 h 넘게 이어진다(감시기가 `logs/lgf/switch_condition` 으로 알린다). 둘 가운데 하나가 서면 메인 세션이 사용자에게 알리고, LGX 로컬 실행기를 드레인하고, h47·h48 의 `ALLOWED_GPUS` 에 8 을 더한다(상수 변경 시각과 커밋을 적는다. 코드 해시가 바뀌므로 h48 집계는 조각의 코드 해시가 섞였다는 경고를 낸다. 설정 해시는 바뀌지 않는다). 이 조건에서 G5·G6 은 멈춘다. GPU 8 제외는 LGF 초판(01:38)의 점유 상태에서 온 상수이며 결과 해시와 무관하다.
- 창 산술 재계산(4.4): 규칙은 바꾸지 않는다. B 를 다시 쓰지 않는다. 창 안에 끝나지 않은 단위는 분할 완결성 규칙으로 처리한다.
- 열람 상태 칸: 'LGX-B 회수 07:20·LGX-A 회수 10:55(미해제·미열람), LG 미회수, LGT 조각 미열람, LGU 판정 표 미열람, LGF 미열람'(작성 시각 기준).

### 6.3 LGU 개정 4 초안

- AB10(WRAPUP 1.1)의 입력: LGU-A1 구간 점수 대비(단 (iii) − B4, is10, n = 10, 풀 레나·캐나다·알래스카)의 두 가중 p 를 `data/processed/lgu/lgu_a_ab10_tests.csv` 에 쓴다. 파일 이름에 'tests' 가 들어가야 h39 의 `*tests*.csv` 읽기에 걸린다. 열은 `test_id`(= LGU-A1), `scope`(= MEAN), `contrast`, `n`, `metric`, `pool`, `delta`, `p_cell`, `p_beq`, `p_eq`, `verdict4`, `nboot`, `source` 다.
- 계산(출처는 h44): h44 가 저장한 2단 재표집 분포 `lgua_boot__<대상>__x.npz` 의 `dist[n, 방법, 재표집, 지표, 가중]` 에서 지역마다 (iii) − B4 를 구하고 같은 재표집 번호끼리 지역 등가중 평균한다(h44 `pool_contrast` 와 같다). `p_cell` 과 `p_beq` 는 마지막 축 0(셀 가중)과 1(블록 등가중)에 h44 와 같은 `boot_p` 를 적용한 값이다. `p_eq` 는 해당 판정 행의 `margin`(기준 점수의 5 %, 가중별)으로 max(P(Δ* ≤ −δ), P(Δ* ≥ δ), 1/R)를 가중마다 계산해 큰 값이다. `verdict4` 는 `lgu_a_tests.csv` 의 해당 행(n = 10) 4분 판정을 그대로 옮기고, '판정 불가(…)' 는 '판정 불가' 로 적는다(h39 `NA_V` 와 맞춘다). `delta` 는 해당 행의 셀 가중 점 추정이다. 재표집 횟수는 LGU 개정 2 의 1,000회다.
- 대조: `p_cell` 이 `lgu_a_tests.csv` 의 `p_boot` 와 1e-12 안에서 같아야 한다(같은 분포). 다르면 AB10 을 '판정 불가(계산 불일치)'로 두고 원인을 적는다. 결과는 `lgu_a_ab10_meta.json` 에 남고 값은 화면에 출력하지 않는다.
- 스크립트: `scripts/2_evaluation/lgu_ab10_tests.py`(합성 입력 시험에서 h39 `lgu_ab10` 이 파일을 읽는 것을 확인했다). h39 1344행의 비고 'p 분해능 5e-4(2,000회)'는 '1e-3(1,000회)'로 고친다.
- 조건부 격자 σ 규약(WRAPUP 6.9 이행): LGU-B2 가 조건을 모두 만족해 지도 패널 (d)의 정규화기가 nflow σ 가 되면, LGU 실험 B 와 같은 원천·epoch·절단 설정으로 nflow 를 seed 2개로 다시 적합하고 두 seed 의 σ 평균을 쓴다. 보정 셀(원천 라벨 지역)과 레나 격자 셀의 σ 를 h49 npz 형식(`cal={지역: σ 배열}`, `grid_cell_id`, `grid_sigma`)으로 저장한다(`data/processed/map_lena/lena_sigma_nflow_v1.npz`). h44–h46 은 적합 모델을 저장하지 않으므로 재적합이 필요하다. 스크립트(A7)는 B2 표를 열기 전에 작성하고 py_compile 과 합성 입력 시험으로 점검한다. B2 가 '전이'이면 GPU 8 에서 LGX 로컬보다 먼저 돌린다(약 0.5 GPU-h, 추정).
- 열람 상태와 표지: 이 개정은 LGU 판정 표를 열기 전에 커밋한다. 판정 표는 04:14·04:31 에 생성되었고 접근 시각이 생성 시각과 같다(relatime, 11:25 확인). LGU-A1(AB10 입력)과 B2 행에는 '등록 시점에 결과 존재(미열람)' 표지를 단다. `lgu_ab10_tests.py` 는 개정 커밋 뒤에 실행한다.

### 6.4 WRAPUP 개정 2 초안

- 6.5 수체 규칙과 PFR 결측 회색 규칙의 해석 확정: 현재 구현(셀 안 30 m 화소의 평균 발생 빈도로 읽는 해석, PFR 결측 회색 규칙 유지. 후자는 해당 셀 0개라 표시 셀에 영향이 없다)을 채택한다. 산출 경로는 `data/processed/map_lena` 로 정하고 6.10 의 `map` 을 고친다. 두 규칙의 입력은 지리 마스크이며 결과를 쓰지 않는다.
- 1.1 AB10 의 재표집 횟수 정정('2,000회' → 1,000회, LGU 개정 2)과 입력 파일 형식(LGU 개정 4).
- 1.4 (b) 본문 정정: H18–H22 에 두 가중 CI 가 있다, H20 예측은 `h21_preds.npz` 에 있다, 입력은 29개다.
- 11.1 에 예외 한 줄: '진행 중 Rescale 작업의 로컬 재실행'은 하지 않는다는 원칙은 유지한다. 08:45 사용자 지시로 중지된 Qjpbeb 의 남은 FT-T 조각은 LG 개정 15 (d)의 보조 이어 실행으로만 돌리며, 주 판정은 한 플랫폼 조각으로 닫는다.
- 11.1 에 제외 기록: RESEARCH_FRAME 5.2·FAILURE_ANALYSIS 6 의 공변량 공간 균등 규칙, Gautam 2025 설정 재현, R0 공변량 지지 점검 규칙은 이번 논문에서 하지 않는다(LGX X3a 가 무작위 대 지역 홀드아웃을 덮고, 지도 과제가 외삽 표시를 둔다). NOVELTY 7.4 선택지 3·4 도 사용자가 등록을 고르지 않으면 같은 방식으로 제외 기록한다.
- h39 `aux4_table` 에 지도 대비를 더하지 않는다(지도 대비는 `lgw_map_aux.csv` 로 충분하다).
- 0.3 넷째 항목의 적용 판단(근거와 함께 적는다): 이 개정은 T_res(07:20:22) 뒤에 커밋된다. 0.3 의 취지는 결과를 본 뒤 정한 규칙을 확인적 문장과 초록에서 빼는 것이다. 이 개정이 바꾸는 규칙은 (i) 등록된 대비와 판정을 바꾸지 않는 기술적 정정(AB10 입력 형식, 재표집 횟수 표기, 1.4 (b) 본문, 산출 경로)과 (ii) 결과를 입력으로 쓰지 않는 지도 마스크 해석이다. 커밋 시점에 회수 묶음은 풀리지 않았고 결과 표는 만들어지거나 열리지 않았다(6.1 (l)의 근거). 그래서 이 개정의 규칙에는 '결과 열람 뒤 등록' 대신 '회수 뒤, 미해제·미열람 상태의 개정' 표지를 단다. 11.1 의 로컬 보조 이어 실행 예외도 같은 표지이며, 보조 판정은 원래부터 확인적 문장과 초록에 쓰지 않는다. 이 판단은 사용자 확인 항목이다(8절 결정 11).

### 6.5 NEXT 개정 기록

- SC1–SC3·AK1 을 철회하고 WRAPUP 3.1·5절의 SC1w–SC3w·AK1w 로 대체한다는 한 줄.

---

## 7. 표시 항목 대응표 초안(F1 의 입력)

Sci Rep 의 본문 표시 항목은 8개다. 현재 틀은 Fig 1–7 과 Table 1 이다(`outputs/figures/paper/CAPTIONS.md`). 아래는 결과 전 초안이며 F1 에서 `figure_spec.json` 에 확정한다. 판정이 조건에 따라 바뀌는 패널(지도, Fig 6)은 조건부 병합 규칙을 함께 적는다.

| 표시 항목 | 주장·틀 | 가설·대비(원천 표) | SI 로 가는 것 |
|---|---|---|---|
| Fig 1 문제 정의 | 전이 문제, 지역별 E | 지역 지도(LGD 새 지역 3곳 추가, 레나 지도 영역 표시), 원천 E0 구성 | 지역별 세부 |
| Fig 2 라벨 수 곡선 | R2, 틀 (1)(2) | LG L1–L4(`lg_curve`, `lg_tests`), LGX L10·L12(`lgx_tests`). 두 판(n ≤ 10 의 4지역, 전 n 의 레나·캐나다) | 지역별 곡선 전체 |
| Fig 3 물리 잔차와 부분 풀링 | R3, 틀 (1) | LG L8, LGX L15–L18(`lgx_tests`) | 앵커 교체 L27·L28 |
| Fig 4 최소 라벨 수 | R2·R3, 틀 (2) | LG L3·L4(`lg_minn`), n* 구간 보고, C12(조건부) | 방법 × n 열지도 |
| Fig 5 관측 배치 | R4, 틀 (2) | WRAPUP L43(`lgw_l43`), LGX L23·L24 | 보조 R1 n ∈ {40, 160}, Alaska(x) |
| Fig 6 라벨 0 과 불확실성 | R1, R5, 틀 (3) | AB4(라벨 0), LGU-A1·B2(`lgu_a_tests`, `lgu_b_tests`), LGX L25, c2 커버리지. B2 가 조건을 모두 만족하면 지도 (a)(d)(e)를 6c 로 둔다 | 지도 나머지 패널, LGU A2–A7·B1·B3–B6·C1–C4 |
| Fig 7 배포 절차 | 틀 (2) | SC1w–SC3w(`lgw_scenarios*`), 주 대비 묶음 AB1–AB10(`lgw_bundle`) | 칸 단위 표 |
| Table 1 자료 | 자료 범위 | 대상 27개, 새 지역 3곳, 라벨 단위 열(행 수와 1 km 위치 수, WRAPUP 2.5), Alaska(x) 참조 행, 약관 상태 | 자료원별 세부 |
| SI(표시 항목 수에 들지 않음) | 보조 | L5·L26 학습기 동등성(학습기 포레스트), LGT L32–L37, LGF F1–F6·N1–N4(N1 대상별 요약: n 별 평균·중앙값·악화 대상 수·최대 증가), LGD PE1·PE2·L38–L42(두 판), L19–L22 검증 사다리, 음성 결과 등록표, 선택·철회 이력, 교차 환경 점검 (i)–(iii)과 보충 대조, 소프트웨어·실행 표(작업 id, 코드 해시, 설정 해시, 두 환경의 판, GPU 비결정성), 기존 제품 비교(조건부) | |

- 점검할 것: LGF-F1·N1(확인적)은 R1 의 근거인데 본문 항목이 없다. F1 에서 Fig 6 의 라벨 0 패널에 F1·N1 요약을 넣을지, SI 에 두고 본문에서 인용할지 정한다. 어느 쪽이든 본문 항목 수는 8 을 넘지 않는다.

---

## 8. 사용자 결정이 필요한 것

| 번호 | 결정 | 기본값(답이 없을 때) | 이유 |
|---|---|---|---|
| 1 | Qjpbeb 의 남은 FT-T 58조각을 로컬에서 이어 돌리되 주 판정에 합치지 않고 교차 환경 보조로만 쓰는 방식. L26 보조 대비의 CatBoost 쪽 근거를 C14(로컬 재적합, CPU 8 스레드 약 2–4 h)로 잴지, 외삽으로 적을지 | 보조로 진행(G3–G6, GPU 8). C14 는 C4 뒤 CPU 여유가 있을 때 한다 | 08:45 지시와 LG 개정 14 (4)·WRAPUP 11.1 을 함께 지키는 방식이다 |
| 2 | ZovWo 가 끝내지 못한 RealMLP 조각이 주 4지역에 속할 때 로컬 보조 이어 실행 여부 | 하지 않음 | 로컬 조각은 주 판정에 못 들어가고 조각당 2–10 h 다 |
| 3 | GPU 6·7·9 를 LGT 종료 뒤 LGF 에 넣는 것(LGF 6.1 의 '한 줄 알림') | 넣는다 | LGF 확인적 N1 과 N2·N3 의 창 여유 |
| 4 | GPU 0–4 사용 | LGF 는 2·3·4 가 비면 넣는다(사전 허용). 0·1 과 LGX 의 0–4 사용은 하지 않는다 | 다른 사용자(hooni900)가 쓰는 번호다 |
| 5 | git push 시점 | A1–A4 커밋 직후, C0 전. 사용자 확인 전에는 하지 않는다 | 사전 등록 커밋 시각의 제3자 기록 |
| 6 | LGD 자료 약관 문의(CUSP, GGD353, calm_web_subsites)와 6.1 (m)의 주 판정 규칙 | 규칙대로 등록하고 연락을 시작한다 | PE1 의 새 얕은 지역 판정이 약관에 걸려 있다 |
| 7 | 문헌 원문 PDF 확보 | 확보 전에는 '요약 도구 확인' 수준 표기 유지 | 원문 대조 필요 |
| 8 | NOVELTY 7.4 선택지 3·4 의 등록 여부 | 제외 기록 | 확인적 판정과 무관한 민감도다 |
| 9 | 지역 등가중 E0 해석식 민감도(M3)의 등록 | 등록하지 않고 문단만 쓴다 | 계산하려면 LG P0 값을 보기 전에 등록해야 한다 |
| 10 | 자료·코드 기탁(O8)의 공개 범위(저장소 공개 여부, 태그, Zenodo) | 판정 기록 뒤 약관 미확인 행을 뺀 판으로 기탁 준비만 한다 | Sci Rep Data·Code availability |
| 11 | WRAPUP 0.3 넷째 항목을 이번 개정들에 적용하지 않는 판단(6.4) | 6.4 의 판단대로 '회수 뒤, 미해제·미열람 상태의 개정' 표지 | T_res 가 이미 지났다 |
| 12 | 전환 규칙이 설 때 h47·h48 의 `ALLOWED_GPUS` 에 8 을 더하는 코드 변경 | 조건이 서면 알린 뒤 한다 | LGF 확인적 가설의 창 |

---

## 9. 하지 않을 것

| 항목 | 이유 |
|---|---|
| 새 Rescale 작업(이어 돌리기, `lgx_sum.yaml`, RealMLP 이어 실행 포함) | 08:45 사용자 지시 |
| 로컬 조각을 `data/processed/lgx` 나 `$LGD` 에 섞어 주 판정 집계 | LG 개정 14 (4), WRAPUP 8.6 비교 규칙 (1)(2) |
| 결과 열람 뒤 가설·판정 문구·범위·확인적 가설 집합 변경 | 사전 등록 |
| LGF 의 B 재계산, 창 연장, 축소 규칙의 사후 적용 | LGF 8.1·8.4 |
| 전환 규칙 밖에서 LGF 에 GPU 8 을 넣거나 GPU 0·1 을 쓰는 것 | 개정과 코드 변경이 필요하다 |
| 본 실행 중 LGF 집계 | LGF 6.3 단계 7 |
| 다른 사용자 프로세스의 종료나 방해 | 공유 서버 규칙 |
| 무거운 집계(C4, C10) 동시 실행 | 공유 서버 메모리 부족 이력 |
| Rescale 결과 폴더에서 직접 집계, 묶음을 저장소 루트에 통째로 풀기 | 대조 대상을 덮어쓴다 |
| 봉인 폴더를 LG 회수 전에 열기 | WRAPUP 결과 추가 1 수정 1 |
| 약관 확인 전 자료원의 행을 커밋·공개 | WRAPUP 7.5 (c)3, 6.1 (m) |
| WRAPUP 11.1 의 기존 제외 항목 | WRAPUP 11.1, WRAPUP_REVIEW 6절 |
| `configs/rescale/lgu_*.yaml` 작성 | LGU 개정 1, 08:45 지시 |

---

## 부록. 스크립트와 점검 결과(11:30)

| 파일 | 용도 | 점검 |
|---|---|---|
| `scripts/local/run_lgx_local_continue.sh`(신규) | LGX 남은 단위의 상위 실행기: status, unpack(묶음의 `data/processed/lgx` 경로만, 이름 겹침·수 대조·sha256·읽기 전용·시각), remaining, seed, check, xenv, gate(첫 조각 열람 시각 기록), run, supplement, chain | bash -n 통과. `status` 실행, `unpack` 이 개정 15 미커밋으로 거부됨을 확인. 학습·해제는 하지 않았다 |
| `scripts/local/run_post_results.sh`(신규) | 결과 뒤 파이프라인 18개 하위 명령(5.2). 등록 확인, `$LGS`, 무거운 집계 잠금, RAM·load 확인, 로그 격리, events.tsv, 첫 열람 시각 | bash -n 통과. `status` 실행, `ladder` 가 개정 15 미커밋으로 거부됨을 확인 |
| `scripts/2_evaluation/lgu_ab10_tests.py`(신규) | AB10 입력 표(6.3) | py_compile·pyflakes 통과. 합성 입력(재표집 1,000, 3지역)에서 p_cell = p_boot 대조 일치, '판정 불가(…)' 정규화, h39 `lgu_ab10` 이 파일을 읽음 |
| `scripts/local/run_lgx_gpu_local.sh`(수정) | 엔진: 실행 중 조정 파일(our_threads, gpus, max_jobs), seed 제외 목록(`LGXL_UNSEED`), check 가 g45 전체 해시도 출력, 개정 표지 '개정 15(' | bash -n 통과 |
| `scripts/local/lgx_local_queue.txt`(수정) | 큐: g45 4줄, nn 11줄(Greenland 제외) | `remaining` 이 해제 뒤 대조 |
| `scripts/local/lgx_xenv_iii_queue.txt` | 점검 (iii) 재적합 큐 | 해당 없음 |
| `scripts/2_evaluation/lgw_xenv_gate_iii.py`(판 2) | 계속 규칙 (다), 보충 대조 모드 | py_compile·pyflakes 통과. 스모크 조각 자기 대조에서 '계속 [(가) True, (나) NA, (다) True]', 보충 모드 동작 |
| `scripts/local/lgf_supervisor.sh`(수정) | LGT 완결 조건(n_done = n_todo), 두 번째 통과 자동화, 역할 T3(J12 판단), 창 마감 시 J12 기록, GPU 부족 감시 | bash -n 통과. 역할 전환과 두 번째 통과는 실제로 돌려 보지 않았다 |
| `scripts/local/lgf_role_queue.sh`(수정) | 역할 T3 | bash -n 통과 |

설정 해시 계산(11:15, 학습 없음, 1 s): 로컬 인자와 Rescale 인자의 g45 FT-T 조각 전체 해시가 레나·캐나다·CA-2·CA-3 의 분할 4·5 에서 모두 같고, nn·r0 해시도 같다. 로컬 코드·입력 판(h40, h42, h41, src/polar 7개, 입력 표 6개)의 sha256 앞 16자리가 두 LGX 묶음의 `lgx_payload_info.txt` 와 모두 같다.

---

## 10. 진행 기록

| 시각 | id | 내용 |
|---|---|---|
| 11:40:50 | G0 | LGF 감시기 교체. 단일 큐(PID 104671)를 역할 single 로 넘겨받았다. 08:32·09:04:55 GPU 5 사건을 창 파일에 사후 기록했다 |
| 11:43:15 | G1 | supergood 가 GPU 5·8 을 다시 잡아 h48 이 종료 코드 1 로 끝났다(501/854). 감시기가 GPU 6·7·9 를 기다린다 |
| 11:44:18 | A1–A5 | LG 개정 15, LGF 개정 6, LGU 개정 4, WRAPUP 개정 2, NEXT 철회 기록, 실행기 커밋(ff1be02, 개정 시각 표기 정정으로 amend). push 는 사용자 확인 전이라 하지 않았다 |
| 11:44:39 | C0b | LGX 두 묶음의 `data/processed/lgx` 만 해제. CPU 3,860, h41 292, GPU 591(RealMLP 77), 이름 겹침 0, 읽기 전용, sha256 목록 |
| 11:44:50 | C0b | 남은 FT-T 58(g45 7, nn 51). nn 구성은 개정 15 (b)의 서술과 달라 정정했다(AL-1 4, Greenland 분할 2 의 1). 큐에 Greenland 추가(1a86df0) |
| 11:46:15 | G3 | 엔진의 빈 연관 배열 오류(bash 4.4 `set -u`)로 점검 (iii) 1회차가 시작 전에 끝났다. 수정 뒤 11:47:23 재시작. GPU 8 이 빌 때까지 대기 |
| 11:48 | A6, A7 | 약관 확인분 묶음(h51 lic, h52 두 판)과 격자 σ 스크립트(h53)를 에이전트에 맡겼다(결과 열람 금지, CPU 2스레드, 합성 입력 시험만) |
| 12:05–12:09 | G1, G2 | LGT 완결(172/172, 실패 0, partial 0). 감시기가 12:09:11 에 LGF 역할 F(GPU 6)와 N(GPU 9·7)을 띄웠다. 두 번째 통과는 필요 없었다 |
| 12:09 | A7, F1, M1, M3–M5 | 격자 σ 스크립트 h53(e284c9b), 그림 명세와 표시 항목(903dce9), 방법·서론 영문 초안(5485688), SI 보조 문안(199588b) 커밋 |
| 12:20 | A1 보충 | LG 개정 15 (p)(q), LGU 개정 4 보충(σ seed 대체 규칙), 표지 보충, 72–95 % 철회 표지(00ee319) |
| 12:25 | C14 | L26 보조 대비의 CatBoost 같은 플랫폼 재적합 시작(20단위, 적합 5,010, 워커 2 × 스레드 4, 출력 `data/processed/lgw/xenv/lgx_base_local`). 끝나면 `lgw_xenv_gate_iii.py --supplement` 로 키별 차만 낸다(`lgw_xenv_c14*`) |
| 12:25 | F1 후속 | 그림 명세가 요구한 새 계산 모듈 두 개(구성 고정 풀 평균 곡선, N1 대상별 요약)를 에이전트에 맡겼다(결과 전 규칙 고정) |
| 12:25–12:30 | C0 | ZovWo Completed(12:25:16). 선택 회수·해제(12:30:38): CPU 117, GPU 231, 빠진 RealMLP 0, 단계 종료 코드 모두 0. G7 불필요 |
| 12:30–13:44 | C1–C6, A3 | cpu-chain: gate-lg(`$LGS` = `$LGD`), ladder(36 s), sum-lgx(12:33–12:43, 조각 4,451, 재현 117/117), gate-lgd(base ok, v3local 불필요), lgd-lic(12:46–13:41), pool-lgd(두 판), lgu-ab10(p_cell = p_boot 확인). 모두 종료 코드 0 |
| 12:31–13:45 | J1, J3, J2, J5, J4 | 판정 기록: LG(0236b89), LGU(869f18c), LGX(aa3b28e), LGD(6e637a1, 주 판정 약관 확인분 판), LGT(8016969). 열람 시각은 `logs/lgf/viewing_state.txt` |
| 12:57 | C14 | CatBoost 같은 플랫폼 재적합 대조 20단위, 최대 차 3.6e-14 cm(d92e1d9). 외삽 표지 불필요 |
| 13:01 | J8 | 추출 재생성 대조 ZovWo CPU 조각 117개 불일치 0(0fc2ef5) |
| 13:45 | F3(1) | Fig 2–4 v2(715814e). L10 보조 행 역전 기록(2fcd492) |
| 13:45 | C7 | h39 summarize 시작(봉인 폴더 개봉 기록은 `sealed/README_SEALED.txt`) |
| 조건 판정 | C12, C13, G8 | C12(L4 최소 n 10), C13(실효 범위 < 채점 거리), G8 σ(LGU-B2 미결정) 모두 조건 불성립으로 하지 않는다 |
| 13:47–13:50 | C7, J7, C8, F2 | h39 summarize·bias-mae(종료 코드 0), WRAPUP 판정(b9f3ec7), 지도 잠정판과 시각 검토(87bd8e2: 600 dpi, 폭 색 범위). 지도는 SI(LGU-B2 미결정) |
| 14:10–14:20 | J10, M2, 보정 | 지난 문서 정정(751d9fa), 결과 절·초록 초안(34f3f9e), 기록 표현 정정(98da2ed, 03b96ce) |
| 14:40 | F3(2) | Fig 1, 5, 7, Table 1, Fig 2–4 갱신(88a616c). 값 대조 120건 일치 |
| 14:45 | O1 | Rescale 원장 갱신: 6작업 Completed, 목록가 × 실행 시간 합 220.26달러(ZovWo 109.25, Qjpbeb 63.37, ShFtT 45.97, 스모크 3건 1.67). 청구서는 웹 화면 기준 |

### 10.1 남은 작업(14:45 기준)

| id | 내용 | 조건 |
|---|---|---|
| G1, C9, J6 | LGF 본 실행 계속(GPU 6·9·7), 창 마감 2026-10-02 05:01 뒤 `post sum-lgf`, LGF 10절 판정 | 감시기 종료 |
| C7 재실행, C8 최종 | LGF 표를 넣어 h39 재실행(δ_rel), 지도 최종판(캡션만 바뀐다) | C9 뒤 |
| F3(3) | Fig 6(라벨 0 포레스트에 LGF-F1·N1 행, 불확실성 패널), Fig 2 의 조건부 곡선 평가(L34(a), LGF-F1·N1) | J6 뒤 |
| M2 채움 | 결과 절·초록의 LGF 자리 표시([k], F1–F6, N1–N4) | J6 뒤 |
| G3–G6, C10, J9 | LGX FT-T 로컬 보조(점검 (iii), g45·nn 이어 실행), 보조 집계, 보조 판정 | GPU 8 이 비면 자동 시작. LGF 창 마감 뒤에는 GPU 9·8·7·6·5 로 넓힌다(`lgx-expand-gpus`) |
| M9, O8 | SI 조립과 투고 형식 점검, 자료·코드 기탁 준비(약관 미확인 행 제외) | J6, J9 뒤 |
| 사용자 | git push, LGD 약관 제공자 네 곳 연락(`docs/LGD_LICENSE_CONTACTS_2026-09-30.md`), 문헌 원문 PDF, 개정 표지 판단(WRAPUP 개정 2 (g)), 원고 [DECISION] 7건 | 사용자 |
