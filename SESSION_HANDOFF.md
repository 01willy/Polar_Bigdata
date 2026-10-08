# SESSION_HANDOFF — Polar_Bigdata (현재 상태 스냅샷)

**갱신**: 2026-10-08(최종 묶음 XA–XM 판정, 원고·덱·발표 완료, 극지연 회의 뒤 전면 재설정 예정) · 2026-10-02(LGF 판정, WF 1–3차, Fig 6, 연구 개요) · 2026-09-30(로컬 전환·개정 네 건·LGX 해제) · 2026-09-29(LG 계획·하네스·Rescale 사전 점검 제출) · 2026-09-26 저녁(통합 실험 실행·감사·논문 그림) · 09-26 오전(통합 계획 확정) · 09-22(H18–H30) · **다음 세션은 이 파일부터 읽으세요.**

## ★ 최신(2026-10-08): 최종 묶음 XA–XM 판정, 원고·보고 덱·발표(10-06) 완료, 극지연 회의 뒤 전면 재설정 예정

정본: 판정 `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 8.1–8.12 와 추가 등록 `docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md`·`..._ADDENDUM_XM_2026-10-05.md` 결과 절, 요약 `docs/MORNING_REPORT_2026-10-05.md` 2절, 원고 `paper/manuscript/{en,ko}`(검토판 `main_review.pdf`), 덱 `deck/deck_spec_paper_report.json` → `deck/render/permafrost_paper_report.{pptx,pdf}`(85쪽), 대본 `deck/script/` → `deck/render/발표대본_연구결과보고_2026-10-06.pdf`, 산출물 허브 `deliverables/`(`bash tools/sync_deliverables.sh`), 안내 `00_START_HERE.md`, 로그 10-08 항목.
- **다음 세션(사용자 10-08 결정)**: 극지연(KOPRI) 회의 피드백을 반영해 (1) 연구 목표와 논문 목표를 다시 정하고, (2) 실험 입력 관리 체계(입력 표 버전·출처·약관·해시, 지역·라벨 원천 목록)를 정비하고, (3) 필요하면 전체 실험을 다시 설계·실행한다. 첫 작업은 사용자에게 회의 피드백 항목을 받아 문서로 남기고, 현재 주장과 판정 가운데 유지할 것과 바꿀 것을 대조하는 일이다. 피드백 내용은 아직 받지 않았다.
- **현재 주장(덱·원고 기준)**: 라벨 0개에서는 원천 계수 Stefan 이 기준이고 직접 ML 은 손해 위험이 크다(2 cm 넘게 나빠진 대상 18/30, 물리 증강 2/30, 물리 유사라벨은 섞은 유사라벨보다 −1.63 cm). 라벨 10개에서 수축 재보정(κ 10)이 −2.45 cm 이고, 같은 라벨로 재보정한 CCI v5 보다 8.4 cm 낮다(XG). 라벨 40·160개에서 교차검증 방법 선택이 재보정 Stefan 보다 −1.35·−1.51 cm 다(XC, 레나·캐나다 풀, 같은 지역 재분할, 160개 이득은 캐나다). 알래스카 지역 안 잔차 결합은 −0.52 cm(전량, RMSE 14.41 → 13.87)이고, 이득은 기후 격자 단위 보정에서 온다(격자 안 상세도 없음, WF9·XE·XM). 직접 ML 오차는 무작위 분할 14.8 cm, 지역 홀드아웃 28.7 cm 다(XH, 3지역 평균). 알래스카 라벨 셀에서 기존 제품과 비교하면 우리 결과가 r 0.62, RMSE 13.8 cm 다(사후 서술, 제품 값은 연도 정합 정의, 근거 표 `data/processed/xbatch/XG_product_comparison/alaska_label_cell_metrics_v1.csv`).
- **정리 감사(10-08)**: `docs/AUDIT_2026-10-08_CLEANUP.md`. 고정 변수와 동결 모듈 해시는 기록과 같다. 헤드라인 수치 15개 가운데 13개는 저장된 표와 맞고, 나머지 2개는 근거 표를 새로 만들거나 설명을 정정했다. 다음 판에서 반드시 고칠 문장은 네 가지다. (1) 불확실성 지도와 워크플로 그림의 '다음 관측 위치' 처방은 XD-1 기각, Algorithm P = 무작위와 충돌한다. (2) 덱 S04 의 '라벨 수가 어떻든 우리 방법이 낫다', '손해 없음' 같은 문장은 판정보다 넓다. (3) 덱과 원고의 제목이 다르다. (4) RESEARCH_OVERVIEW 의 낡은 주장은 머리에 대체 표시만 달았다. 10-06 발표 덱은 고치기 전에는 외부에 돌리지 않는다.
- **git 밖 로컬 자료**: 대용량 재생성물(`data/processed/map_alaska` 842 MB, `lgx` 344 MB, `lgu` 259 MB, `lgt` 138 MB, 각 실험의 `shards/`, `results/rescale_*/data`)과 셀 단위 실측값·잔차 표, 열람 조건 미충족 봉인 표(`xe_r1b_xt2_*`)는 .gitignore 로 뺐다. 판정 표(`sealed/` 등)와 작은 결과 표는 10-08 정리에서 커밋했다.
- **GPU(10-08 16:30)**: 0–4 는 다른 사용자 점유(100 %), 5–9 는 비어 있다. 쓰기 전에 `nvidia-smi` 로 다시 본다.
- **열린 마감 항목**: XE 2단계 취득(AppEEARS 87건, 10-05 12:38 기준 완료 2·처리 중 37·대기 48, 내려받기 0)은 마감 10-08 14:22 까지 끝나지 않았다. 등록 규칙(계획서 2.5절)은 해당 군 결측 처리와 개정 이력 기록이다(모든 군이 빠지면 xh = xt2 로 작업 R3). xt2 봉인 표는 열람하지 않았다. XF·R4 마감은 10-11 14:22 이다(적격 새 지역 0곳, 외부 라벨 2건은 북대서양 민감도). 사용자 결정(10-08): XE 는 상태만 기록했다(계획서 개정 이력 10-08 16:35, 최종판 생성은 일정 이탈로 기록). xe_feat 를 원자료 기준(O·V 포함)과 열 기준(H0·T2) 가운데 어느 쪽으로 고정할지는 xt2 표를 열기 전에 정한다. 동결, xt2 열람, R3, XF 마감 처리는 다음 세션 재설정에서 정한다.
- **재사용 자산**: 하네스(LG `scripts/3_deep_learning/h40_label_grid.py`, WF `h54_workflow.py`, 최종 묶음 `xbatch_core.py`·`x_*.py`), 입력 표 v3/v4(`fidelity_base_v3/v4.csv`, `e5_soil_tdd_v3/v4.csv`, `covariates_ext_v1.csv`), 지도 격자(`data/processed/map_alaska`, `map_lena`), 그림 양식 `design/style_tokens_v4.json`, 덱 빌더 `deck/build_paper_report.py`, 방법·결과 그림 `scripts/4_visualization/method_figs_v4.py`.
- **사용자 확인 대기**: 회의 피드백, XE·XF 마감 처리, 원고 [DECISION]·[PENDING](XE 3, XF 2, XC-F3 1), 원고 제목(덱 제목과 맞출지), 저자·사사·LLM 사용 고지, LGD 약관 제공자 연락, Gmail·Calendar·Drive 커넥터 인증.
- **Rescale**: 원장 합 약 224.6달러(10-02 의 223.24달러 + 작업 A 약 1.40달러, 목록가 × 실행 시간).

## ★ 이전(2026-10-02): 실험 전부 종료·판정 기록, 연구 주제 재설정(워크플로), Fig 6, 원고 재구성 대기

정본: 연구 개요 `docs/RESEARCH_OVERVIEW_2026-10-02.md`(흐름, 주장 C1–C8, 워크플로, 선행 연구 대비 차이, 질문별 답), 주장 `docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md`, 판정 WF 계획서 `docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 5.1–5.3, LGF 계획서 10절, 남은 작업 `docs/EXECUTION_PLAN_REMAINING_2026-09-30.md` 10.3.
- **연구 주제(사용자 10-01)**: 물리 증강·물리 잔차 ML 의 라벨 수별 능력과 새 지역 워크플로(학습·평가·추가 관측). 무작위 분할 과대평가는 보조.
- **핵심 결과**: 라벨 0 직접 ML 은 학습기 10종 모두 물리식보다 나쁨(LGF-F1·N1 포함), 물리 증강·저가중 잔차가 손해를 막음(위험표 18/30 대 2/30). ML 이득은 계수 오차에 비례하고 라벨 10개로 진단(ρ 0.38). 규칙 W 가 고정 레시피보다 라벨 40·160개에서 약 1 cm(레나·캐나다, 이득은 캐나다). 알래스카 지역 내 이득 0.4–0.5 cm(유의)이고 격자 단위 편향 보정에서 옴(격자 안 상세도 없음, WF9). 공변량 분산 추출이 캐나다에서 2.4–3.2 cm(전이 조건에서는 어느 대상에서도 악화 없음), 능동 선택 무익. 기후 외삽은 판정 불가(WF10).
- **실험 상태**: 등록 실험 모두 끝남. LGF 10-01 23:34 완료(1,730단위). WF 1–3차 Rescale(xQyfeb, OUZBT, Vppbeb). 남은 실험 없음.
- **다음 세션 첫 작업(사용자 10-02)**: 원고 재구성(결과·토의·초록 재작성)과 그림 재구성(Fig 5·7). 시작 전에 재구성안 `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 8절의 결정 3건(그림 배치, 초록에 WF 판정 포함 여부, 제목)을 사용자와 정한다.
- **WF9 토양 √TDD 민감도(10-02 로컬 완료)**: 지역 내 판정은 같고(기각), 전이 부분 지지는 유지되지 않음(기각). C8 은 더 약한 판정으로 고쳤다. 표 `data/processed/wf_soil/wf3b_*`, 기록 WF 계획서 개정 이력.
- **원고**: LGF 자리 채움 완료. Fig 6 v2(`outputs/figures/paper/v2/Fig6_label0_uncertainty.*`). 재구성안 `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 의 사용자 결정 3건(Fig 5·7 교체, 초록에 WF 판정, 제목) 뒤 결과 절 재작성.
- **사용자 확인 대기**: git push(로컬 main 이 origin 보다 117커밋 앞), LGD 약관 제공자 연락, 문헌 원문 PDF, 원고 [DECISION] 7건, 재구성안 결정 3건.
- **Rescale**: `tools/rescale_client.py cost` 기준 합 223.24달러(목록가 × 실행 시간, WF 5건 약 2.98달러). 원장 CSV 의 WF 행은 제출 기록만 있다.

## ★ 이전(2026-09-30 오후): 판정 기록 완료(LG·LGX·LGD·LGT·LGU·WRAPUP), 그림 v2, 원고 초안

정본: 판정은 `docs/EXPERIMENT_PLAN_LG_2026-09-29.md` 7.1–7.4, `docs/EXPERIMENT_PLAN_LGU_2026-09-29.md` 11.1, `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 결과 판정 기록(J7)·지도 잠정판. 남은 작업은 `docs/EXECUTION_PLAN_REMAINING_2026-09-30.md` 10.1.
- **핵심 결론**: 라벨 0 에서 직접 ML 은 물리식을 넘지 못한다(L1, L30, L34a). 더 강한 물리 기준선 P* = P0@tddm 이 있다(L29). 잔차 ML 은 라벨 전량에서 P0·P* 를 넘으나 러시아 W 에 치우친다. n = 10 순가치는 Holm 뒤 비유의, n = 40·160 은 P1* 를 넘지 못한다. 불확실성 확인적 가설은 미결정. 배포 안전성 미확인.
- **원고**: 방법·서론(5485688), 결과·초록(34f3f9e), SI 보조(199588b), 그림 v2 Fig 1–5, 7, Table 1(`outputs/figures/paper/v2/`), 지도 SI(`outputs/maps/transfer_lena/`).
- **남은 계산**: LGF(창 마감 10-02 05:01, 감시기 `logs/lgf/supervisor.log`), LGX FT-T 로컬 보조(GPU 8 대기, `scripts/local/run_lgx_local_continue.sh status`).
- **Rescale**: 끝. 원장 합 220.26달러(목록가 기준).

## ★ 이전(2026-09-30 오전): 로컬 전환, LGX 회수·해제, 결과 열람 전 개정 네 건, 결과 뒤 파이프라인

정본: 실행 계획 `docs/EXECUTION_PLAN_REMAINING_2026-09-30.md`(3절 작업 id, 10절 진행 기록), 개정 LG 15·LGF 6·LGU 4·WRAPUP 2(커밋 ff1be02).
- **Rescale**: 추가 지출 중단(08:45 지시). ZovWo 는 14:30 전에 끝난다. 회수는 `scripts/local/run_post_results.sh fetch-lg`, 이어서 `cpu-chain`(gate-lg → ladder → sum-lgx → gate-lgd → pool-lgd → lgu-ab10).
- **LGX**: 두 묶음 해제 완료(`data/processed/lgx`, 읽기 전용). 남은 FT-T 58조각은 로컬 GPU 8 보조 이어 실행(`scripts/local/run_lgx_local_continue.sh status`).
- **LGF**: 감시기 `logs/lgf/supervisor.log`. 창 마감 2026-10-02 05:01. GPU 5 를 두 번 잃어 창 여유가 작다.
- **LGT**: 172단위 가운데 165(11:50). 끝나면 감시기가 상태 표를 보고 두 번째 통과 또는 GPU 넘김을 한다.
- **판정 기록 순서**: LG(J1) → LGX(J2) → LGD(J5) → LGT(J4) → LGU(J3) → h39(J7) → LGF(J6, 창 마감 뒤) → 지도 → 보조(J9).
- **사용자 확인 대기**: git push(로컬 main 이 origin 보다 앞서 있다), LGD 약관 제공자 연락(CUSP, GGD353, calm_web_subsites), 문헌 원문 PDF, 자료·코드 기탁 범위.

## ★ 이전(2026-09-29): LG(라벨 격자 통합 재실행) 사전 등록·하네스·Rescale 포장, 사전 점검 작업 제출

정본: 계획 `docs/EXPERIMENT_PLAN_LG_2026-09-29.md`(개정 2), 큰 틀 `docs/RESEARCH_FRAME_2026-09-29.md`, 지역별 행렬 `docs/COVERAGE_MATRIX_BY_REGION_2026-09-29.md`, 실패 원인 `docs/FAILURE_ANALYSIS_2026-09-29.md`, 로그 09-29 항목. 커밋 40be64c(로컬).
- **실행 환경**: 실험은 CPU 부분까지 Rescale에서 한다. 로컬은 `--count-only`, `--summarize-only`, CSV 읽기만 한다(스레드 1).
- **Rescale 사전 점검(완료)**: 작업 `qOkSo`(iolite-1), 38분, 정산 0.44달러. 13단계 전부 종료 코드 0, 적합 실패 0건, 단위 시험 12건 통과. 결과 `results/rescale_lg_smoke/`. 실측 환산은 계획서 §6.1.
- **본 실행(제출, 실행 중)**: 작업 `ZovWo`, https://kr.rescale.com/jobs/ZovWo/ . `configs/rescale/lg_full.yaml`(iolite-4, 64코어·T4 4장, 사전 등록 범위 전부, 학습기 7종, 상한 22 h, 최악 125.31달러, 예상 15–16 h·약 88달러). 사용자 지시: 비용보다 범위의 완전성 우선.
- **확장 실험 LGX(제출)**: 계획서 6A 절(가설 L9–L31). 하네스 `h42_label_grid_ext.py`, 검증 사다리 `h41_validation_ladder.py`. 사전 점검 `nxvpT` 는 21단계 중 19단계 통과, 재현 점검 통과, 결함 2건 수정. 본 실행은 A(`lgx_a.yaml`, CPU 축·검증 사다리·학습기 6종, 상한 14 h)와 B(`lgx_b.yaml`, RealMLP, 상한 10 h)로 나눠 제출했다. 작업 번호는 `results/rescale/_ledger.csv`. 끝나면 본 실행과 A·B 의 결과 묶음을 올려 `lgx_sum.yaml` 로 통합 집계한다.
- **TabPFN 라벨 격자 LGT(구현 중)**: 로컬 GPU 전용(가중치 배포 조건). 계획서 6C 절, 하네스 `h43_tabpfn_label_grid.py`. 워크플로 wf_d7ec99b5-e4a.
- **독립 지역 자료 확보 LGD(진행 중)**: 계획서 6B 절, 새 입력 표 v4. 워크플로 wf_d3f37deb-d20.
- **예측 분포 실험 LGU(구현 중)**: 생성 모델 검토 `docs/GENERATIVE_MODEL_REVIEW_2026-09-29.md` 의 권고 3건(계층 예측 분포 사다리, 폭 정규화 conformal, 공간 정보량 진단). 계획서 `docs/EXPERIMENT_PLAN_LGU_2026-09-29.md`. 워크플로 wf_b0be06f0-359.
- **신규성 검증본**: `docs/NOVELTY_POSITIONING_2026-09-29.md`. 기법 신규성 없음, 평가 설계와 음성 결과가 기여.
- **실행 환경 변경(2026-09-30 01시 사용자 지시)**: 새 작업은 Rescale 이 아니라 로컬 GPU 서버에서 돌린다. 진행 중 Rescale 작업(ZovWo, Qjpbeb, ShFtT)은 끝까지 둔다. CPU 는 우리 작업 합계 32스레드 이하, nice 10. GPU 배정: 9·7·6·5 TabPFN(LGT), 4·3 LGF, 2(필요하면 1) LGU, 8 다른 사용자. LGX 통합 집계와 LGD 새 지역 실행도 로컬에서 한다.
- **LGX 확인용 사전 점검 `gLyfeb`**: 21단계 전부 통과(단위 시험 h42 28건, h41 23건, 재현 점검 차 0.0). 결과 `results/rescale_lgx_smoke2/`.
- **세션 한도 중단(2026-09-29 22:40, 2026-09-30 00:00경)**: 로컬 워크플로 네 개(LGT 수정·스모크, LGD 통합, LGU 구현, LGF 설계)가 중단되었고 01:10 에 끊긴 단계부터 다시 돌렸다. 중단된 에이전트가 남긴 부분 파일(`src/polar/lgu_common.py`, `h43` 수정분)은 다시 돈 에이전트가 읽고 이어서 쓴다.
- **신규성 검증**: 워크플로 wf_ea2b1da5-83e(선행 문헌 재확인, 누락 선례 탐색, 반박 심사). 결과는 메모리 `novelty-positioning`에 반영한다.

### ▶ 다음 세션
1. `python3 tools/rescale_client.py status ZovWo`로 본 실행 상태를 본다. 끝났으면 `fetch ZovWo --dest results/rescale_lg`로 회수하고 `lg_status.csv`, `lg_tests.csv`, `lg_failed.csv`를 확인한다.
2. 벽시계 상한에 걸렸으면 `results_lg.tar.gz`를 `results_lg_prev.tar.gz`로 올려 이어 실행한다(`lg_full.yaml`의 dataset_file_ids).
3. 실행 중 로그는 Rescale API의 `jobs/<id>/runs/1/tail/process_output.log`로 볼 수 있다.
4. 회수 뒤 로컬에서 `--summarize-only`로 집계하고 계획서 §7에 L1–L8 판정을 적는다. 그림은 `scripts/4_visualization/paper/`에 추가한다.

## ★ 이전(2026-09-26 저녁): 통합 실험 A2·B1–B4·C1·C2·D1 실행 + 감사 21건 확정 + H25 블록 CI 재실행 + 논문 그림 Fig 1–7·Table 1·부록

정본: 판정 `docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md` §10(§10.1 가설 판정, §10.3 원고 주장 갱신), 감사 `docs/AUDIT_2026-09-26.md`, 로그 09-26 저녁 항목, 핸드오프 `gpt/handoff/20260926_2130-final-experiments-audit-figures.md`, 그림 스펙 `figures/PAPER_FIGURE_REDESIGN_2026-09-26.md`, 그림 `outputs/figures/paper/`(Fig1–7, Table1, FigS*, supp_tables, CAPTIONS.md, source_data).
- **채점 변경**: 모든 새 CI 는 `src/polar/h4_common.py` 블록 부트스트랩. 옛 (분할, 반복) CI 는 좁았다(감사). 손익분기 주 정의 = 블록 CI 상한 < 0 · 분할 승률 ≥ 2/3 · 반복 승률 ≥ 0.75.
- **가설**: 지지 F1·F8·F11(교락), 부분 F2·F3·F10, 기각 F4·F5·F6·F7·F9.
- **절 2 새 축**: 소수 라벨의 가치는 E 추정(수준 지역 이득의 72–95 %). 잔차 ML 이 대상에 적응하려면 수백 개(레나 320, LE-1·LE-2·AL-2·AL-6 은 320 로도, AL-4 는 160 으로도 부족). 라벨은 흩어 모은다(E 추정 시 n=40 −1.1 ~ −1.7). 고정 κ=10 이 가장 견고, 적응형 풀링은 극단 수준 차이(몽골)에서만 필요. 3라벨 진단 분기는 이득 없음(음성). [2026-09-30 표지: 이 "72–95 %" 주장은 신규성 심사에서 기각·철회되었다(docs/NOVELTY_POSITIONING_2026-09-29.md 15·143·326행). 원고에 쓰지 않는다. 정정 본문은 결과 판정 뒤 J10 에서 쓴다]
- **폐기**: 'ρ −0.84 로 필요 라벨 수 예측', 'H24 k-중심 채택', '레나·캐나다 n ≤ 320 미달성'. 캐나다·AL-1·CA-2·AL-5·AL-3 은 라벨 0(원천 잔차)에서 이미 물리식을 넘는다.
- **라벨 0**: 예측은 물리식 수준(CCI 다층 결합도 기각). 계층 conformal 90 % 구간 커버리지 AB4 0.86 [0.80, 0.92].
- **외부 홀드아웃(몽골)**: 3라벨 절차 −53 cm(물리식 271 cm). 대상 라벨이 전부 지온 유도(F4_calm_temp)라 라벨 정의 교락.
- 모델: 세션 중 Fable 5.1 → Opus 5.5(사용량 한도). GPU 9 만 사용(8 은 타 사용자).

### ▶ 다음 세션(2026-09-29 갱신)
00. **큰 틀 확정**: `docs/RESEARCH_FRAME_2026-09-29.md`(단일 축 = 대상 라벨 수 n에 대한 Δ(n) 곡선, 두 기준선, 절 R1–R5, 시험 범위 행렬과 공백, 쓰지 않을 문장). 실행 순서는 `docs/EXPERIMENT_PLAN_NEXT_2026-09-28.md` §8(재집계 → α 가중 → S0·S4 → conformal 확장 → 시나리오 매트릭스 → 지도 시연, 전부 CPU 9–12 h). 알래스카 우선 검증 설계 §8.3.
0. 잔여 실험 계획 `docs/EXPERIMENT_PLAN_NEXT_2026-09-28.md`(P1 α 가중 잔차·P2 재집계·P3 배포 시나리오·P4 증강+잔차 결합·P5 독립 지역 확충·P6 선택). GPU 점유로 실행 보류, 전부 CPU 로 가능. 준비도 심사(5인 + 반박 2회) 결론: 사전 등록 실험 완료, C4 불필요, P1·P2 는 투고 전 필수.
1. 원고 초안(영문): 절 구조 = 계획서 §10.3 표. 헤드라인은 '언제·몇 개 라벨로 ML 이 물리식을 넘는가'로. 그림은 `python3 scripts/4_visualization/paper_figs.py --qa` 로 재생성 가능.
2. 미결 자료 결정: GTNP 러시아 W 신규 4지점 편입(D2, v3 수정 필요), 몽골 직접 측정 라벨 확보 여부(D1 교락 해소).
3. 선택 실험: 대상 행 가중 α 를 키운 잔차(대상 라벨이 원천에 희석되는 문제, A2 해석), 분할 5회 이상·규약 통일(감사 조치 8), C4 InSAR 다중 충실도(미수행).
4. `git push`(이번 세션 커밋은 로컬만).

## ★ 이전(2026-09-26 오전) — 논문 완성 통합 실험 계획 확정(사전 등록) + 조사 3건 + 저널 규격 그림 원형

정본: `docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md`(§1 주장별 상태, §2 조사 반영, §4 트랙 A–E, §5 가설 F1–F11, §6 실행 순서, §7 그림 설계). 로그 09-26 항목, 핸드오프 `gpt/handoff/20260926_1200-final-paper-plan.md`. 이번 세션은 실험을 돌리지 않았다.
- 사용자 질문 1 답: 잔차 학습이 물리식을 넘는 최소 라벨 수는 수준 오차 지역 3–10(러시아 W·AL-1·AL-3), CA-2 40, 구조 오차 지역(레나·캐나다 등 10 대상)은 n ≤ 320에서 미달성. 단 H25 S3는 E 수축과 교락되어 있어 **구조 지역의 최소 n은 아직 정해지지 않았다** → Track A2로 판정.
- 조사: LORO 전이 오차를 보고한 ALT 매핑 논문 없음, Gautam 2025(Sci Rep)가 유일한 ML 대 Stefan 비교(방향 일치), E 공변량 예측 성공 사례 없음, InSAR 프록시는 알래스카 밖 불안정 → 약라벨은 한계 서술. 채택 기법 = log E 오프셋 최대우도·혼합효과 부스팅·PPI++·계층 conformal·D-최적 설계·TFM(잔차 한정).
- 그림: `scripts/4_visualization/paper_figs.py` 신설, `outputs/figures/paper/Fig2_label_budget` 원형(180 mm, Arial 7.5 pt, 패널 문자, PDF+600 dpi PNG, 캡션 자동).

### ▶ 다음 세션(계획서 §6 순서대로, GPU 6–9 확인 후)
0. CCI GTD·PFR·불확실성 층 다운로드 시작(`fetch_cci_layers.py` 신규, CEDA) → 백그라운드.
1. A2 최소 라벨 수(`h31_min_labels_residual.py`, h25 골격, CPU 6워커 3 h) 스모크 → 본 실행.
2. B2 부분 풀링(`h33_partial_pooling.py`) 병렬, B4 TFM few-shot(`h35_tfm_fewshot.py`, GPU 6·7).
3. B3 선택 규칙 → B1 2단계 프로토콜 검정 → C1·C2 → D1 외부 홀드아웃(사전 예측 `h4/d1_prediction.json` 먼저 기록) → `h4_analysis.py` → `paper_figs.py` Fig1–7·표·부록 → 검토 에이전트 2회 → 로그·핸드오프·커밋.


## ★ 최신 완료(2026-09-22 저녁) — 라벨 예산 프로토콜 H25–H30 + 라벨 있음 레시피 중첩 선택 + 그림 8종

정본: `docs/EXPERIMENT_PLAN_LABEL_BUDGET_2026-09-22.md`(§7 결과), 로그 09-22 저녁 항목, 핸드오프 `gpt/handoff/20260922_1930-label-budget-protocol.md`. 결과 `data/processed/h3/`, 그림 `outputs/figures/h3/`.
- 손익분기 라벨 수(S3): 러시아 W 3·AL-1 3·AL-3 10·CA-2 40, 나머지 10 대상 미달성. **|log E비| 가 손익분기 n 을 ρ −0.84 로 예측**(라벨 미사용 지표는 실패) → 2단계 프로토콜(대표점 3개 → E비 → 필요 라벨 수).
- 라벨 전량 E 재적합이 캐나다 +3.5·CA-3 +5.9 악화(블록 간 E 이질성). k-중심 선택은 n=3 일부 지역만 유리, n≥10 무작위 우세 지역 다수 → H24 채택 범위 축소. 한 블록에 몰린 라벨은 58~62% 해로움(H29).
- 라벨 있음 레시피: 사전 지정 Stefan+CatBoost λ.25/.5 −0.68/−1.14 확정, 중첩 선택(Ku 보정 앵커+CatBoost) −3~−4 는 CI 0 포함·블록 등가중 0(집계 의존). CatBoost 대 신경망 차이 비유의.
- 배포 규칙: AOA 게이팅 잔차 ML − 직접 ML −2.52(최악 완화), − Stefan +0.11(동급); AOA 직접 ML 은 물리식보다 +1.6 열세.

### ▶ 다음 세션
1. E 적합 개선: 군집 크기 가중 최소제곱, 라벨 출처 블록 수에 따른 계층 수축(κ 자동) → 캐나다형 악화 제거 여부.
2. 2단계 프로토콜 확인적 검정(대표점 3개 → E비 → 분기) + 라벨 지역 확장.
3. 원고 재단: 절 1(라벨 있음 확정 레시피) · 절 2(라벨 예산 계단·손익분기·해로운 조합) · 절 3(라벨 0 물리 앵커·한계).

## ★ 최신 완료(2026-09-22 오후) — H18–H24 실행 결과: 라벨 없는 전이에서 물리식을 넘는 요인 없음, 관측 설계(라벨 3개 k-중심 선택)만 유의

정본: 설계(사전 등록) `docs/EXPERIMENT_DESIGN_H18-H24_2026-09-22.md`(§9 결과 요약), 로그 `docs/EXPERIMENT_LOG.md` 09-22 오후 항목, 핸드오프 `gpt/handoff/20260922_1300-h18-h24-transfer-collapse.md`. 결과 DB `data/processed/h2/`(h21/h22/h23/h24/h19/h_tests_*), 확장 하네스 `data/processed/m1/h1819_*`, 공변량 확장 `data/processed/covariates_ext_v1.csv`, 그림 `outputs/figures/h2/`(스펙 `figures/figure_spec.json` h2_*). GPU 4·5·6·7 사용(8·9 타 사용자).
- **H18 MODIS LST 강제력**: Stefan(LST) − Stefan(기온) +0.16 [−0.65, 1.24] / 공변량만 +0.10; ML 입력 추가 +0.03 → 기각. 레나의 자체 E 는 LST 로 알래스카와 일치(1.62 vs 1.63)하나 RMSE 이득 없음(구조 오차).
- **H19 공변량 23종(TWI·수체·식생·ERA5 확장)**: 블록 E LORO R² 전 집합 음수; ML(x25+A) − ML(x25) 공변량만 +0.64 [0.05, 1.12](Holm 0.22, 악화), SoilGrids 제거도 +0.51 악화 → 기각.
- **H20 CCI 블록 계수 / H21 계수 대여**: −0.43 [−2.56, 3.51](블록 등가중 +5.5) / +1.26 [−0.52, 2.61] → 기각. 유사도(지리·공변량·물리)가 E 를 예측하지 못함.
- **H22 불변 학습(IRM·V-REx·GroupDRO·DANN·지역 임베딩·안정 특징)**: 잔차 λ=.25 − Stefan +0.02~+0.39(IRM 유의 악화), V-REx − ERM −0.31 [−0.44, 0.00] → ERM 보다 나아도 물리식 아래로 못 내려감 → 기각.
- **H24 관측 설계**: k-중심(공변량 대표점) n=3 − 무작위 레나 −0.67·캐나다 −0.74·러시아 W −2.44(CI 0 제외)·러시아 E +0.23·알래스카 샌드박스 −1.50; 캐나다 재적합 해 +5.6→+0.9. n≥10 에서 소멸, 블록 순환은 레나·러시아 E 악화, 기존 CALM 사이트 위치는 무작위보다 나쁨 → A(라벨 3–5개 한정).
- **H23 MAML(n=3·5·10)**: MAML 적응 − E 수축 AB4 +1.09(캐나다 −0.51 우세, 러시아 W +4.03 열세), E 수축+MAML 잔차 +0.20 → 기각. 수준 오차 지역은 라벨을 E 재추정에 써야 함.
- **감사 정정**: `polar.m1_stats.summarize_delta` 점 추정치·CI 지역 집합 통일(assert 추가), fig01 iw 필터, 포레스트 별표 Holm 기준. 검토 에이전트(시각·과학) 지적 반영해 그림 재렌더.
- 탐색: 배포 게이팅(AOA·DI·CCI 일치) 어느 것도 Stefan+CCI(28.9)를 넘지 못함(최선 29.0).

### ▶ 다음 세션
1. 라벨 지역 확장(GTN-P/CALM 추가 지역·시추공 유도 라벨 재편입)이 선행 조건: CI 가능 지역이 4개뿐이라 계수 대여·불변 학습·관측 설계 규칙의 일반성을 검정할 수 없다.
2. 지표 아래 정보: InSAR 계절 침하(얼음 풍부도)·이탄 두께(PEATMAP/NCSCD)·IPA 구역으로 E 회귀 재시도; 8일 MOD11A2·MOD10 적설 지속기간 n-factor 재시험.
3. 원고: 결과 절 재단(정보·계수·학습 목표 축 기각 + 관측 설계 채택), "라벨 예산 대 오차"+"선택 규칙" 통합 그림, 알래스카 최소 관측망 지도. 논문 2단 폭 그림 판(주석 9 pt) 별도 산출.

## ★ 최신 완료(2026-09-18/21) — M1 마스터 요인 실험 + 회의적 감사 반영 + Sci Rep 격차 보완

정본: 계획 `docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md`(끝의 **개정 09-21** = 결과 열람 전 고정한 채점 규약·다중 비교·결정 규칙), 로그 `docs/EXPERIMENT_LOG.md` 2026-09-18/21 항목, 프런트매터 `docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md`. 결과 DB `data/processed/m1/`(하네스 `scripts/3_deep_learning/m1_master_factorial.py`, 분석 `m1_analysis.py`, 그림 `scripts/4_visualization/m1_figs.py`). 데이터 v3(`fidelity_base_v3.csv`, 스발바르 분리·CALM 지온 유도 68셀 강등·토양 도일 0 결측). GPU 5–9 사용.
- **재현 게이트 통과**: 옛 표 18개 수치 재현 또는 원인 규명(캐나다 차이 = 데이터 판·S3 채점 셀 정의).
- **확정 결과(AB4=레나·캐나다·러시아 W·E, 층화 블록 부트스트랩)**: C1 유사라벨 순가치 사다리 전 단 유의(증강 없음 −3.6, 상수 −4.0, 대상 수준 상수 −1.8, 셔플 −1.9, TDD 선형 −0.7) · Stefan+CCI 결합 −0.9~−1.4 비유의·블록 등가중 부호 반전(확정 효과 아님) · 잔차 학습·학습 지역 확장·중요도 가중·물리 계수 지도·생성 모델 전부 이득 없음 또는 악화 · 중첩 선택 레시피 27.9 vs Stefan 26.7(H12 기각) · **라벨 있음 조건 잔차 결합 −0.7~−1.1 유의(H13)** · 모델 10종 어느 것도 정보 없음에서 Stefan 못 넘음.
- **보조 분석**: 교차검증 방식(무작위 낙관 4.5 cm, kNNDM 13.70 vs 블록 13.33), fold 민감도, 변수 중요도, 잔차 공간자기상관, 오차 층화, 지역별 E·n-factor, 잡음 하한, 라벨 연도·연도 정합 TDD 민감도(결론 불변), 공변량 이동 진단(AUC 포화·MMD·AOA), ML 붕괴 층화, 결합 분해(ρ 0.67–0.99), CCI 페돈 민감도, 오라클 E 상한, **전이 UQ(알래스카 CQR 전이 커버리지 0.32–0.73, 앵커±Q 가중판 0.84–0.93, 러시아 W 회복 불가)**.
- **희소 라벨 곡선 F1–F3 완료**: 러시아 W 라벨 3개로 39.8→30.6, 레나 40–80개에 0.1–1 cm, 캐나다 무작위 라벨 해(블록 이질성), 안전 기본값 κ=10 수축 E. 미완: S-D 상호작용(자기훈련 대조 `self`·편향 물리 용량-반응·셀 셔플·거리 버퍼: `src/polar/m1_ext.py` 준비됨, 하네스 연결 필요), 영문 원고 재구성.
- 운영 교훈(메모리): 동시 프로세스 GPU당 1개·스레드 상한, 병렬 에이전트 남발 금지(사용량 한도).

### ▶ 다음 세션
1. 희소 라벨 곡선 그림(`outputs/figures/m1/sparse_label_curve.png`)을 결과 절 마지막 그림으로 편성. 후속 실험 후보: 지역 간 계수 대여(가장 유사한 라벨 지역의 E를 빌리는 LORO, 해석적·CPU 수 분).
2. S-D 최소 집합만: 자기훈련 대조(H16)·셀 셔플·편향 물리 용량-반응(m1_ext.py를 --configs 경로로 연결, GPU 1장 1시간).
3. 원고 재단: 결과 절 = (1) 물리식 사다리·검증 방식 (2) 요인 주효과 표(AB4, 3조건) (3) 대조군 사다리 (4) 전이 벤치마크·결합 분해·심부 한계 (5) 라벨 있음 곡선 (6) UQ(지역 내 CQR + 전이 앵커 구간). 프런트매터 초안 적용. `docs/RESULTS_RECONCILIATION_2026-09-14.md` §5 문장 교체.

## ★ 최신 완료(2026-09-08) — 논문화 실험 설계(사전 등록) + E1·E2·E3·E4 실행 완료

정본 `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md`(사전 등록 H1–H6 + 개정 이력 5건), 결과 로그 `docs/EXPERIMENT_LOG.md` 2026-09-08 항목, 진행표 `docs/PAPER_PLAN_SCIREP.md` §7. GPU 2·3·4·5 사용. 커밋 5488d1a·726927d·292f8ba·ff6504a(09-15 cleanup).
- **E4.1 중첩 선택**: S12 185조합 탐색값(22.92/21.32)은 leave-one-target-out에서 25.49/25.32로 물리식(24.11/22.91)보다 나쁨 → 주 추정치를 사전 지정 등가중 앵커(23.27/21.68)·λ=0.25 ridge(23.00/22.24)로 교체.
- **E2 물리식 사다리**(같은 보정 자유도 k): "Stefan만 정확"은 보정 비대칭의 교락. 비가중 7지역 평균은 보정 Kudryavtsev 30.7 < Stefan 33.7이나 그린란드 3셀 효과이고 셀 가중은 Stefan 우세(09-14 정정). 토양 도일 Stefan(`e5_sqrt_tdd_soil`)은 알래스카·레나 우세, 러시아 열세, 그린란드는 TDD_stl1=0 아티팩트. 멱지수 b≈0.35–0.42.
- **E3 CALM 확충**: `assemble_dl_dataset.py:45` 북미 필터로 비북미 157좌표 미편입이었음 → `data/processed/fidelity_base_v2.csv`(17,572행, 기존 불변, 신규 149셀). 주 전이 집합(ALT<150 cm) 6지역 = 레나·캐나다·러시아 W/C/E·그린란드, 심부 4지역 = 스발바르·몽골·알프스·티베트(전 식 RMSE 180–350). `fidelity.py` TRANSFER_MAIN/DEEP, pytest 18 통과.
- **E1 통합 요인 설계**(3조건×앵커×증강×잔차, 같은 셀): H1 Stefan+CCI 앵커 방향 일관·비유의(셀 가중 기준 10지역 8/10, 주 5/6, p=0.11/0.22; 블록 다수결은 혼재) · **H2 잔차 학습(ridge λ=0.25) 기각**(주 6/6 악화, 외삽 폭주) · **H3 √ 함수형 추가 기여는 캐나다 1.0·레나 0.1 cm**(도일 관계 자체 3.6/0.6, 상수 대조 10.8/1.7) · **H6 무보정 등가중 CCI 결합은 알래스카 지역 내 +2.02 악화**(보정 CCI 결합은 −0.24, 09-14 정정). 앵커 우세 지역이 갈림(레나·캐나다=Stefan+CCI, 시베리아=Kudryavtsev). 알래스카 지역 내 13.33(풀링)은 25종 ridge 단독 13.62와 −0.29 [−0.61, 0.08] 차이.
- **E4.3 UQ**: CQR interval score 68.9 vs 상수 폭 참조 65.4, 양극단 5분위 커버리지 0.83–0.85 → C3 서술 완화. **E4.4 라벨 정의**: 알래스카 라벨 86% GPR, 계수·전이 결론 불변(≤0.5 cm).
- 신규 스크립트: `scripts/3_deep_learning/e1_unified_factorial.py`·`e1_analysis.py`·`e2_physics_ladder.py`·`e3_adaptive_E.py`·`e4_nested_selection.py`·`e4_interval_score.py`·`e4_label_sensitivity.py`, `scripts/1_data_prep/expand_calm_regions.py`·`era5land_soil_tdd.py`, `scripts/4_visualization/e2_ladder_figs.py`·`e1_figs.py`. 그림 `outputs/figures/{e2_physics_ladder,e1_factorial}/`(figure_spec 등록 5건).

### ★ 2026-09-14/15 추가 — 기존 주장 대조·감사 + 다음 세션 계획 확정
- 대조 워크플로(8 주장 + 5 스크립트 감사): **뒤집힘 0**·강화 1·정교화 4·하향 3, 신규 스크립트 결론 무효화 결함 0. 종합 `docs/RESULTS_RECONCILIATION_2026-09-14.md`(§5 원고 변경 표, §6 09-08 서술 정정 5건·데이터 후속 수정).
- 사용자 입장(2026-09-15): 기존 "CCI 앵커+잔차 ML의 완전 전이 개선"이 선택 효과라는 판정을 **아직 신뢰하지 않음**. 다음 세션에서 (1) 판정 재확인(V1–V4), (2) 완전 전이 돌파 후보 총력 시험(T1–T6, 중첩 선택 필수), (3) 병행으로 희소 라벨 전이(F1–F3), (4) 사전 고정 결정 규칙 A/B/C로 두 번째 의의 확정. **정본 `docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md`.**
- 선행 데이터 정비: Svalbard 매크로 본토 3사이트 분리, CALM 지온 유도 68셀 등급 분리, e5_tdd_soil=0 결측, 블록<8 CI 미산출, GTN-P ALT 106 데이터셋(`data/raw/gtnp/alt_csv/`, 미편입) 편입 검토.

### ▶ 다음 세션 (2026-09-16 사용자 결정)
정본 = **`docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md`**(세 조건 × 여섯 축 마스터 요인 실험, 단일 하네스·단일 결과 DB). 하위 = `docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md`(V1–V4·T0–T6·F1–F3·결정 규칙).
순서: S-A 데이터 정비 → S-B 재현 게이트(기존 표 전 수치 0.1 cm 재현, 실패 시 중단) + V1–V4 → S-C 주효과 → S-D 상호작용(T0 물리 계수 지도·T0b 수준 구조 분리·T2·T3) → S-E 중첩 선택·H1–H14 → S-F 희소 라벨 → S-G 표·그림·원고.
주 축 = 공변량만(배포 조건), 정보 없음 = 비적응 기준선. 결정 규칙 A/B/C 사전 고정. 모델 축은 트리·MLP·FT-T·TabM에 RealMLP·TabPFN·조건부 확산·플로 매칭·정규화 플로·앙상블을 더해 13종 전부 주효과에서 비교(사용자 요청 09-16). 시작 시 `nvidia-smi`(09-16 기준 4·5 타 사용자 점유).

## ★ 최신 완료(2026-08-31) — 본선 발표덱 최종화 + 정합 감사 + 논문화 전략

**본선 제출물**: `deck/render/permafrost_final.{pptx,pdf}` **23장**(본문 19 + 보충 4). 빌드 `deck/build_final.py`(+`final_lib.py`, 스펙 `deck/deck_spec_final.json`), 신규 그림 `deck/mk_final_figs.py`. 발표 문서 2종: `deck/render/발표대본_ALT_ctrl.docx`(`deck/mk_script_docx.py`), `발표QA대비_ALT_ctrl.docx`(36문항, `deck/mk_qa_doc.py` + `deck/qa_answers.md`).
- **방법부 그림 3차 재설계 확정 문법**: 백색 배경 + 파선 구분선 + 주황 헤딩 + 텍스트 위계. 박스는 3종만(실데이터 이미지 액자·셰브런·네트워크 연산 블록). 화살표는 수평/수직 직각만. **금지**: 외곽 프레임, 색 구역 패널, 카드 박스 일반, 판정 태그 박스, 곡선 화살표, 번호 틱바, 그림 안 pNN 상호참조, "선+주황 라벨+큰 숫자" 통계 카드(→ key-value 헤어라인 표). 상세 `deck/deck_spec_final.json` revision_v5~v9.
- **정합 감사 4축 + 최종 검수**: BLOCKER 1건 해소(s12 그림 물리식 단독이 anchor="stefan" 전체 셀 → 표 8과 같은 `stefan_cci_w` CCI 유효 셀로 교체, 그림 23.72/24.60 → 22.91/24.11로 표와 일치). main.tex 표 8 반올림(27.94→27.93, 26.61→26.60), anomaly 범위 대상 한정, 용어 통일(물리 결합→**물리 잔차 결합** 9건, CCI 제품→**위성 제품**), tab:s12 열머리·fig:s12 캡션을 검증 3조건 정식명으로. 덱 p20 구간 폭 "약 15–54" → "14.7 → 53.6 cm(보정 전→후)".
- **확정 정정**: 알래스카 13,606셀 = **0.5° 블록 74개**(구 83은 오류, map_model_gate_meta.json n_blocks=74로 독립 확인).
- **KPDC 콘슬 재검증**: 유도 ALT 파싱 정당성 확정(독립 재파싱 소수점 재현, 깊이 라벨 3중 물리검사 통과). 값의 사다리 = 탐침 35.0(시즌중반) → 8월 41.6 → CALM U27 53–89 → 피트 58–73 → 코어 하단 104–120 → 심부 시추공 121–153. 136.4는 심부 부분표본이지 사이트 대표값 아님. **주의**: 지온 프로파일 KPDC 식별자는 00002707(2024)·00002955(2025)로 별도(덱·보고서는 코어 00002125만 표기 → 병기 검토).
- **논문화 전략**(메모리 `paper-journal-strategy`): 현 상태 TC/PPP는 무리. **1순위 CRST**(IF 4.9, 구독 경로 APC 0원, Liu 2023 PI-LSTM 선례지) → 2순위 Scientific Reports. TC 배제 근거: TC의 ML 게재는 전부 "ML=도구" 구조이고 방법 조합을 주장한 tc-2022-9는 리젝, 공개심사 리젝 기록 영구 잔존.

### ▶ 다음 세션 = 실험 보완(P0) + 논문화 착수 (사용자 확정 2026-08-31)

**투고지 확정: Scientific Reports** (CRST는 예비). 상세 작업 계획 = `docs/PAPER_PLAN_SCIREP.md` (P0 통계 보완 → 원고 작업 → Sci Rep 실무 → P1 선택 순).

### ▶ 다음 세션 최우선 (P0, 계산 수일)
1. **185조합 winner's curse 해소**: 중첩 선택 재평가 또는 등가중 앵커(24.11→23.27) 사전 지정을 주 추정치로 교체.
2. **대조군 사다리**: 셔플 Stefan·TDD 선형식 추가로 "물리 정보 vs 저복잡도" 분리.
3. **관련연구 재배치**: Read 2019(WRR PGDL)·Jia 2021·Willard 2022·Liu 2023(CRST PI-LSTM) 인용, 차별점을 "공간 표적 증강 + 순가치 대조 + 음성 물리"로 좁힘. 이후 영문화 + 코드·데이터 공개 성명.
4. 잔여 정합(비차단): `report_overview_figure.py:124` n_grid=892,865 하드코딩의 근거 산출물 부재 → report_alt_map_hires.py가 meta JSON에 격자 셀 수 기록하도록. `figures/figure_spec.json`에 이번 재생성 그림 4종 스펙 미등록. `submission/`(07-30 예선 동결본)의 83 잔존은 **수정 금지**, 본선 패키지는 현행 scripts/에서 재생성.

## ★ 최신 완료(2026-07-27 오후) — E1 co-kriging + E2 계절내 D(t) (사용자 피드백 반영)

GPU **7**(세션 중 여러 번 재배정). 사용자 피드백 2건 반영. 상세 `docs/EXPERIMENT_LOG.md` 최상단.
- **E1 co-kriging (채택)**: 정통 공간보간 vs 물리+ML 비교표. in-domain OK 15.77·IDW 15.80 ≈ GBM 17.73(경쟁), 전이 붕괴(OK 29.40·IDW 50.92·GBM 41.37), **Stefan 앵커만 양축 최선(14.46·21.26)**. kriging 전이붕괴=variogram 지역차(covariate shift 공간통계판). 피드백 "인터폴레이션 비교" 충족.
- **E2 계절내 D(t) (partial)**: "timelapse에 따라 ALT 계산"을 계절내 융해진행으로 재정의. KPDC 콘슬 일별온도 활용. **계절내 D(t)는 예측가능 구조(연도간 corr0.06과 대조)**이나 2년 소표본서 persistence(17.5) > Stefan(40.8)·GRU(46.6). EOS는 GRU만 내부peak 예측. 결론=시간성능은 문제정의·데이터밀도 의존(문헌 정합). 피드백 "시계열 예측" 답변 완결.
- **시계열 게이트 미통과**: 기존 T-lite 미통과는 방법 실패 아닌 "정적 공변량+연1회로 미래외삽=예측신호 없음" 규명(문헌 대조).

## ★ 최신 완료(2026-07-27 밤) — 보고서 전면 재작업(사용자 지적 9건 반영)

사용자 지적 반영해 그림·보고서 전면 재작업. 상세 그림 재작업·LaTeX 논문.
- **그림 전면 개선**: (8) ALT 관측이 몇십개처럼 보이던 문제=데이터 구조 규명(13,606셀이 74개 0.5° 블록에 초집중, 배로 3,429셀 등)+배로 250m 격자 확대 inset(4,381셀)으로 "수천 관측 실재+공간대표성 한계" 동시 표현. (3·7) 고해상 자산 편입(local_demo PolSAR 250m·alt_surface_northslope·magt·field3d·tournament, dpi300+PDF), deploy 2종(Diffusion 아티팩트) 제외. (6) 얕은3D를 3D warp 대신 논문형(깊이슬라이스 4패널·fence 단면·트럼펫 프로파일)으로 재작성.
- **LaTeX 논문형 보고서 (제출가능)**: `outputs/report/main.{tex,pdf}`(13쪽, xelatex+xeCJK). ※ 경로는 outputs/report/ (구 docs/report/ 아님). 논문 구조(초록·서론·데이터·방법·결과·논의·결론·참고문헌). **강점·차별성·의의·한계극복 중심**(논의 5.2 기존연구 대비 차별성·5.3 의의 3층위·5.4 한계 극복방안·5.5 포지셔닝). 자연스러운 논문체(과한 수사·정직강조·em-dash 없음). 고해상 그림 12개. 블라인드. **xeCJK 한글 공백 버그(CJKspace=false→true) 수정**.
- **Word 병행**: `outputs/CONTEST_REPORT_2026.docx`(pandoc, 이미지 임베드). md 원본 `docs/CONTEST_REPORT_2026.md`.
- **제출 직전 잔여**: 참고문헌 완전 서지, (규정이 물리쪽수 요구시) 단단 조판·그림 확대로 10쪽화, PDF 9.1MB 용량제한시 다운샘플. git 미커밋.

## 이전 완료(2026-07-27 저녁) — 시각화 통일 + 예선 제출물(발표덱·분석보고서 md)

- **시각화 통일**: 8서사축 대표그림 점검·재렌더(냉색 규약·mask_ocean·단위·em-dash·블라인드 전수점검). data_inventory에 KPDC Council 편입, 아키텍처/개념도 최신방법 갱신, 정보병목 신규그림(alt_info_bottleneck), E2·S7 PDF 벡터 추가, 주황·붉은계열 제거.
- **발표덱 10장 (예선 아카이브)**: `deck/render/permafrost_summary.{pptx,pdf}`. ※ 본선 산출물은 permafrost_final 23장(최상단 참조). 페이지번호 정상·제목 클리핑 제거·블라인드 푸터 익명화(PDF에 프로젝트코드명 텍스트 0)·s7 dpi300. 10장 스토리(문제→데이터→방법→ALT→병목→전이→증강→UQ→시계열/3D→KPDC). visual QA 통과.
- **분석보고서 (제출가능)**: `docs/CONTEST_REPORT_2026.md`(10섹션·2467단어). 데이터출처·해석기법 명시, KPDC 주활용 부각, 블라인드·em-dash 위반 0, **핵심 수치 전 항목 CSV 대조 정합**, SOTA 무과대주장·negative 정직 서술.
- **제출 직전 잔여**: (a) §2 데이터 출처표에 각 자료원 라이선스·기간·해상도 별첨표 추가(§10.4 이월), (b) 보고서 md→PDF 변환·페이지 확인. git 미커밋(사용자 지시 대기).
- **다음**: 제출물 최종 점검·PDF 변환. 필요시 /cleanup으로 커밋. 마감 2026-07-31.

## 이전 완료(2026-07-27) — 잔여 6단계 S6~S11 완주 + 적대검증 3렌즈 + 표적 수정

GPU **3,4,5**(사용자 2026-07-27 지시). 멀티충실도 로드맵 잔여 단계를 전부 실행. 상세 `docs/EXPERIMENT_LOG.md` 최상단(2026-07-27 항목). 누설 pytest 16개 통과 유지, git 미커밋(cleanup 담당).
- **채택 3**: **S7**(KPDC 콘슬 19프로파일 0°C 등온선 ALT 유도+구간검열 정량표, 대회 KPDC 주활용 충족, 알래스카 in-domain·전이근거 아님), **S9**(2010-2024 연별 timelapse 재렌더+anomaly GIF, 홀드아웃 14.97cm·연anomaly 예측불가 각주), **S11**(증강방식 16행 비교표+conformal cov90 44.6→93.4%+다축검증표, 헤드라인).
- **negative 2**: **S6**(source-aware A5, LORO 21.53 vs baseline 21.11 미달·cov90 무정보→ 비교표1행+소스신뢰도 진단으로만 편입), **S8**(mixture-of-physics, Stefan 못 이김 29.22 vs 22.24, 전이서 전문가 선택 불가능성 진단).
- **재현 1**: **S10**(얕은 3D 온도장 바이트단위 재현 R²0.4688·2.66°C, 0°C 등온면 3D 재렌더).
- **핵심 결론 정합**: 어떤 증강·구조·mixture도 전이서 Stefan 앵커(LORO 21~22cm)를 유의 초과 못함(S6·S8 negative). in-domain 최저 13.33cm(S4). "물리는 앵커로만 유효, 전이는 물리+앵커링뿐"이라는 서사 재확증.
- **적대검증**: 직접 누설 0·핵심수치 전수재현. 지적 반영 수정 완료(em-dash 제거, S8 라벨겹침·overplotting, S6 y축0, S7 컬러바, S10 갭색·masked contour, S9 경도라벨). **S2 physics-as-feature 로그 부호오류 CSV 대조로 정정**.
- **다음**: 예선 보고서 조립(§8 산출물 7종)·대표 시각화 통일 재렌더. 보고서 헤드라인=S11(UQ)+S2(물리앵커)+S7(KPDC)+S3(반응곡선). 마감 2026-07-31.

## 이전 완료(2026-07-24 오후) — S4·S5 완주 + S3 증강비율 버그 수정

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
