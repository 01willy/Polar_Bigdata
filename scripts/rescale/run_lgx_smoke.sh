#!/bin/bash
# LGX(H41·H42) Rescale 스모크·사전 점검 작업. 설정 configs/rescale/lgx_smoke.yaml (iolite-1: 4코어, GPU 1장, 벽시계 3 h).
# 로컬 공유 서버에서는 실행하지 않는다(사용자 지시 2026-09-29).
#
# 단계(앞 단계가 실패해도 뒤 단계를 진행하고, 단계마다 종료 코드를 lgx_smoke_status.csv 에 남긴다. 필수 = 1 인 단계가 실패하면 종료 코드 1)
#   묶음 smoke
#     1 env, inputs      환경 출력, 입력 확인(고정 입력 표 3종 포함. 없으면 종료 코드 1 로 끝낸다)
#     2 deps             필수 패키지 설치(사전 점검 qOkSo 와 같은 판). 선택 패키지: pytabkit(RealMLP), pytest
#     3 smoke_gpu        h42 --smoke --part gpu (축 r0, g45, nn. 학습기 7종, epochs 3). 백그라운드로 시작해 4·5 와 함께 돈다
#     4 smoke_cpu        h42 --smoke --part cpu (축 11개, 대상 3, 분할 1)
#     5 smoke_h41        h41 --smoke (단 V-R, V-B, V-C100, V-G, 지역 내 Lena. 학습기 설정은 본 실행과 같다)
#     6 smoke_sum_h42    h42 --smoke --summarize-only (재표집 10,000회, 재현 점검 기준 lg_smoke)
#       smoke_sum_h41    h41 --smoke --summarize-only
#       gate_smoke       재현 점검 요약(LG 사전 점검 작업의 스모크 조각과 대조. 같은 노드 종류다)
#     7 timing           적합 시간 표
#   묶음 pytest
#     8 pytest_h42       LG_RUN_HEAVY=1 python3 -m pytest tests/test_h42_ext.py (LG_RESCALE 는 지우고 실행한다. 시험이 실행 거부를 확인한다)
#       pytest_h41       LG_RUN_HEAVY=1 python3 -m pytest tests/test_h41_ladder.py
#   묶음 count
#     9 count_*          본 실행 범위의 적합 수 표(--count-only). 시간 환산의 분모로 쓴다
#   묶음 precheck(남은 시간 안에서만 돈다. 시간 측정용이라 한 번에 하나씩 돌린다)
#    10 precheck_cpu     h42 --precheck --part cpu (캐나다 x, 분할 1, n {0, 40}, 본 실행과 같은 설정: r = 30, 고용량 학습기, 교차 적합)
#       precheck_gpu     h42 --precheck --part gpu (축 r0, nn. RealMLP 를 뺀 6종. GPU 1장에 프로세스 3개)
#       precheck_realmlp h42 --precheck --part gpu --axes r0 --learners realmlp
#       precheck_sum     h42 --precheck --summarize-only (재현 점검 기준 lg_precheck)
#       gate_precheck    재현 점검 요약
#    11 timing_final     적합 시간 표와 본 실행 환산(lgx_smoke_full_projection.csv)
#   결과 묶음 results_lgx_smoke.tar.gz 는 묶음마다 다시 쓴다(벽시계 상한에 걸려도 앞 단계 결과가 남는다)
#
# 환경 변수(설정 파일의 command 에서 준다)
#   LGX_SMOKE_BUDGET_MIN   스크립트 시작부터 쓸 수 있는 시간(분). 기본 110. 설정 파일은 165 를 준다(벽시계 3 h 보다 15분 짧게)
#   LGX_SMOKE_DO           실행할 묶음의 쉼표 목록. 기본 smoke,pytest,count,precheck
#   LGX_SMOKE_GPU          require(기본. GPU 가 없으면 gpu 단계를 실패로 기록) | skip(gpu 단계를 생략. CPU 전용 노드에서 쓸 때)
#   LGX_SMOKE_LEARNERS     스모크 gpu 부분의 학습기. 기본 mlp,cfm,ddpm,tabm,nflow,ftt,realmlp
#   LGX_RM_EPOCHS          RealMLP 시간 측정의 epochs. 기본 256(본 실행과 같다). 줄이면 조각 tag 를 lgxrm 으로 따로 둔다(집계에 섞지 않는다)
#   LGX_KEEP_INPUTS        1 이면 끝난 뒤 푼 코드와 입력 자료를 지우지 않는다. 기본 0

main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -e .git ]; then                                                    # 끝에 푼 코드와 입력을 지우므로 저장소 안에서는 실행하지 않는다
    echo "[smoke] 저장소(.git)가 있는 디렉터리다. 이 스크립트는 Rescale 작업 디렉터리에서만 실행한다"; exit 1
  fi
  if [ -f lgx_payload.tar.gz ] && [ ! -f data/processed/fidelity_base_v3.csv ]; then
    tar xzf lgx_payload.tar.gz || { echo "[smoke] 묶음 해제 실패"; exit 1; }
  fi
  # shellcheck source=scripts/rescale/lg_common.sh
  source scripts/rescale/lg_common.sh || exit 1
  # shellcheck source=scripts/rescale/lgx_common.sh
  source scripts/rescale/lgx_common.sh || exit 1
  lg_base_env

  local T0 BUDGET_S LOGD STATUS THREADS LEARNERS RM_EPOCHS DO GPU_MODE
  local BG_PID="" BG_NAME="" BG_REQ=0 BG_LIM=0 BG_T0=0
  T0=$(date +%s)
  BUDGET_S=$(( ${LGX_SMOKE_BUDGET_MIN:-110} * 60 ))
  DO=",${LGX_SMOKE_DO:-smoke,pytest,count,precheck},"
  GPU_MODE="${LGX_SMOKE_GPU:-require}"
  LEARNERS="${LGX_SMOKE_LEARNERS:-mlp,cfm,ddpm,tabm,nflow,ftt,realmlp}"
  RM_EPOCHS="${LGX_RM_EPOCHS:-256}"
  LOGD=logs_lgx_smoke
  STATUS=lgx_smoke_status.csv
  THREADS=$(lg_ncores)
  [ "$THREADS" -gt 4 ] && THREADS=4
  mkdir -p "$LOGD" "$LGX_DIR"
  echo "stage,required,rc,sec,note" > "$STATUS"
  [ -f lgx_payload_info.txt ] && sed 's/^/[payload] /' lgx_payload_info.txt

  left() { echo $(( BUDGET_S - ( $(date +%s) - T0 ) )); }
  want() { case "$DO" in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
  # cap <단계 상한 s> <뒤에 남길 시간 s>: 남은 시간 안에서 이 단계가 쓸 수 있는 시간
  cap() { local v; v=$(( $(left) - $2 )); [ "$v" -gt "$1" ] && v=$1; echo "$v"; }

  # run_stage <이름> <필수 0|1> <제한 시간 s, 0 = 없음> <명령...>
  run_stage() {
    local name="$1" req="$2" lim="$3"; shift 3
    local log="$LOGD/${name}.log" t0 rc note=""
    t0=$(date +%s)
    lg_log "[stage] ${name} 시작(제한 ${lim} s): $*"
    if [ "$lim" -gt 0 ]; then
      timeout -k 30 "$lim" "$@" 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}
    else
      "$@" 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}
    fi
    [ "$rc" -eq 124 ] && note="시간 제한 ${lim} s 로 중단"
    [ "$rc" -eq 137 ] && note="강제 종료"
    echo "${name},${req},${rc},$(( $(date +%s) - t0 )),${note}" >> "$STATUS"
    lg_log "[stage] ${name} 끝: 종료 코드 ${rc} · $(( $(date +%s) - t0 )) s · 경과 $(( ( $(date +%s) - T0 ) / 60 )) min"
    return 0
  }
  skip_stage() { echo "$1,$2,NA,0,$3" >> "$STATUS"; lg_log "[stage] $1 생략: $3"; }
  # timed <이름> <필수> <단계 상한 s> <뒤에 남길 시간 s> <최소 시간 s> <명령...>: 남은 시간이 최소 시간보다 짧으면 생략한다
  timed() {
    local name="$1" req="$2" mx="$3" keep="$4" mn="$5" lim; shift 5
    lim=$(cap "$mx" "$keep")
    if [ "$lim" -ge "$mn" ]; then
      run_stage "$name" "$req" "$lim" "$@"
    else
      skip_stage "$name" 0 "남은 시간 부족(쓸 수 있는 시간 ${lim} s, 최소 ${mn} s)"
    fi
  }
  # start_bg <이름> <필수> <제한 시간 s> <명령...>, wait_bg: 백그라운드 단계 하나
  start_bg() {
    BG_NAME="$1"; BG_REQ="$2"; BG_LIM="$3"; shift 3
    BG_T0=$(date +%s)
    lg_log "[stage] ${BG_NAME} 시작(백그라운드, 제한 ${BG_LIM} s): $*"
    timeout -k 30 "$BG_LIM" "$@" > "$LOGD/${BG_NAME}.log" 2>&1 &
    BG_PID=$!
  }
  wait_bg() {
    [ -n "$BG_PID" ] || return 0
    local rc note=""
    wait "$BG_PID"; rc=$?
    [ "$rc" -eq 124 ] && note="시간 제한 ${BG_LIM} s 로 중단"
    [ "$rc" -eq 137 ] && note="강제 종료"
    echo "${BG_NAME},${BG_REQ},${rc},$(( $(date +%s) - BG_T0 )),${note}" >> "$STATUS"
    grep -a -E "\[FAIL\]|\[warn\]|\[pool\]|Traceback|^\[(data|plan|done)\]" "$LOGD/${BG_NAME}.log" | head -n 30 | sed "s/^/  [${BG_NAME}] /"
    tail -n 8 "$LOGD/${BG_NAME}.log" | sed "s/^/  [${BG_NAME}] /"
    lg_log "[stage] ${BG_NAME} 끝: 종료 코드 ${rc} · $(( $(date +%s) - BG_T0 )) s · 경과 $(( ( $(date +%s) - T0 ) / 60 )) min"
    BG_PID=""
  }

  pack() {
    lgx_pack results_lgx_smoke.tar.gz 1 "$LGX_DIR" "$LOGD" "$STATUS" lgx_smoke_timing_by_learner.csv lgx_smoke_full_projection.csv \
             lgx_smoke_gate_check.csv lgx_precheck_gate_check.csv lgx_payload_info.txt
  }
  timing() { run_stage "$1" "$2" 300 python3 -u scripts/rescale/lgx_timing_table.py --lgx-dir "$LGX_DIR" --lg-dir "$LGX_LG_DIR" --out-dir . --prefix lgx_smoke; }

  finish() {                                                              # 상태 요약, 묶음, 정리, 종료 코드
    wait_bg
    pack
    lg_log "[smoke] 단계별 상태"
    sed 's/^/  /' "$STATUS"
    if [ "${LGX_KEEP_INPUTS:-0}" != "1" ]; then                          # 푼 코드와 입력 자료는 출력이 아니다(출력 동기화 절약)
      rm -rf src scripts tests .pydeps .lg_pip_constraints.txt .lgx_pip_constraints.txt lgx_payload.tar.gz "$LGX_LG_DIR" \
             data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/lg_subregion_map_v1.csv "${LGX_TABLES[@]}"
    fi
    local bad
    bad=$(awk -F, 'NR > 1 && $2 == 1 && $3 != 0 {print $1}' "$STATUS" | tr '\n' ' ')
    if [ -n "$bad" ]; then
      lg_log "[smoke] 실패한 필수 단계: $bad"
      exit 1
    fi
    bad=$(awk -F, 'NR > 1 && $2 == 0 && $3 != 0 && $3 != "NA" {print $1}' "$STATUS" | tr '\n' ' ')
    if [ -n "$bad" ]; then
      lg_log "[smoke] 필수 단계는 모두 통과했다. 끝나지 않았거나 경고가 있는 선택 단계: $bad"
      exit 2
    fi
    lg_log "[smoke] 모든 단계 통과 · 경과 $(( ( $(date +%s) - T0 ) / 60 )) min"
    exit 0
  }

  # ---------------------------------------------------------------- 1 환경·입력
  lg_env_report 2>&1 | tee "$LOGD/env.log"
  lgx_check_inputs || { echo "inputs,1,1,0,입력 파일 또는 고정 입력 표 없음" >> "$STATUS"; finish; }
  echo "inputs,1,0,0,ok" >> "$STATUS"

  # ---------------------------------------------------------------- 2 의존성
  if lgx_install_deps 2>&1 | tee "$LOGD/deps.log"; [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "deps,1,1,$(( $(date +%s) - T0 )),필수 패키지 설치 실패" >> "$STATUS"
    lg_log "[smoke] 필수 패키지 설치 실패. 종료 코드 1"
    finish
  fi
  [ -d .pydeps ] && export PYTHONPATH="$PWD/.pydeps${PYTHONPATH:+:$PYTHONPATH}"     # 파이프 안에서 설치한 경우의 경로 반영
  echo "deps,1,0,$(( $(date +%s) - T0 )),ok" >> "$STATUS"
  if lgx_versions_check 2>&1 | tee -a "$LOGD/deps.log"; [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "deps_versions,0,2,0,설치된 판이 사전 점검(qOkSo)의 판과 다르다" >> "$STATUS"
  else
    echo "deps_versions,0,0,0,사전 점검과 같은 판" >> "$STATUS"
  fi
  local HAVE_RM=1 HAVE_PT=1
  if [ "$GPU_MODE" != "skip" ]; then
    lg_install_opt pytabkit "pytabkit==1.7.3" "pytabkit" 2>&1 | tee -a "$LOGD/deps.log"
  fi
  python3 -c "import pytabkit" >/dev/null 2>&1 || HAVE_RM=0
  lg_install_opt pytest "pytest>=7" 2>&1 | tee -a "$LOGD/deps.log"
  [ -d .pydeps ] && export PYTHONPATH="$PWD/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
  python3 -c "import pytest" >/dev/null 2>&1 || HAVE_PT=0
  if [ "$HAVE_RM" -eq 0 ]; then
    LEARNERS=$(echo "$LEARNERS" | tr ',' '\n' | grep -vx realmlp | paste -sd, -)
    [ "$GPU_MODE" = "skip" ] || echo "deps_pytabkit,0,1,0,pytabkit 없음. RealMLP 를 뺀다" >> "$STATUS"
  fi
  lg_env_report 2>&1 | grep -E "^\[env\] (numpy|pandas|scipy|scikit-learn|catboost|pytabkit|pytest|torch) " | tee "$LOGD/env_after_deps.log"

  local GPUS=""
  [ "$GPU_MODE" = "skip" ] || GPUS=$(lg_gpu_list)
  lg_log "[smoke] 보이는 GPU: '${GPUS}'(방식 ${GPU_MODE}) · 스레드 ${THREADS} · 스모크 학습기 ${LEARNERS} · 시간 한도 $(( BUDGET_S / 60 )) min · 묶음 '${DO}'"

  # ---------------------------------------------------------------- 3–7 스모크
  if want smoke; then
    if [ -n "$GPUS" ]; then
      # 실제 GPU 1장. --epochs 5 를 주지만 h42 finalize 가 스모크의 epochs 를 min(epochs, 3) 으로 자른다(실제 값은 unit.json 의 cfg.epochs).
      # cpu 단계와 함께 돈다(스레드 1). 스모크의 시간은 환산에 쓰지 않는다(사전 점검 실측이 있으면 그것을 쓴다).
      start_bg smoke_gpu 1 1800 env LG_RESCALE=1 python3 -u "$LGX_H42" --part gpu --smoke --gpus 0 --procs-per-gpu 1 --threads 1 \
               --epochs 5 --learners "$LEARNERS" --no-summarize
    elif [ "$GPU_MODE" = "skip" ]; then
      skip_stage smoke_gpu 0 "LGX_SMOKE_GPU=skip"
    else
      echo "smoke_gpu,1,1,0,GPU 가 보이지 않는다" >> "$STATUS"
      lg_log "[smoke] GPU 가 보이지 않는다. gpu 부분을 실행하지 않고 실패로 기록한다"
    fi
    run_stage smoke_cpu 1 2700 env LG_RESCALE=1 python3 -u "$LGX_H42" --part cpu --smoke --workers 1 --threads "$THREADS" --no-summarize
    run_stage smoke_h41 1 1500 env LG_RESCALE=1 python3 -u "$LGX_H41" --smoke --workers 1 --threads "$THREADS" --no-summarize
    wait_bg
    pack
    run_stage smoke_sum_h42 1 1200 env LG_RESCALE=1 python3 -u "$LGX_H42" --smoke --summarize-only --workers 2 --threads 2 --gate-tag lg_smoke
    run_stage smoke_sum_h41 1 600 env LG_RESCALE=1 python3 -u "$LGX_H41" --smoke --summarize-only --threads "$THREADS"
    run_stage gate_smoke 0 180 env OMP_NUM_THREADS=1 python3 -u scripts/rescale/lgx_gate_check.py --smoke --gate-tag lg_smoke \
              --out-csv lgx_smoke_gate_check.csv
    timing timing 1
    pack
  else
    skip_stage smoke 0 "LGX_SMOKE_DO 에 없음"
  fi

  # ---------------------------------------------------------------- 8 단위 시험
  if want pytest; then
    if [ "$HAVE_PT" -eq 1 ]; then
      # 시험은 LG_RESCALE 이 없는 환경을 전제로 한다(실행 거부와 자원 제한을 확인한다). 학습을 하는 시험은 --allow-local 을 직접 준다.
      timed pytest_h42 1 1800 600 120 env -u LG_RESCALE LG_RUN_HEAVY=1 CUDA_VISIBLE_DEVICES= python3 -m pytest -q -rfEs -p no:cacheprovider \
            tests/test_h42_ext.py
      timed pytest_h41 1 1200 300 120 env -u LG_RESCALE LG_RUN_HEAVY=1 CUDA_VISIBLE_DEVICES= python3 -m pytest -q -rfEs -p no:cacheprovider \
            tests/test_h41_ladder.py
    else
      skip_stage pytest 0 "pytest 없음"
    fi
    pack
  else
    skip_stage pytest 0 "LGX_SMOKE_DO 에 없음"
  fi

  # ---------------------------------------------------------------- 9 적합 수 표
  if want count; then
    timed count_cpu 0 420 240 60 env LG_RESCALE=1 python3 -u "$LGX_H42" --part cpu --count-only --threads "$THREADS"
    timed count_gpu 0 300 240 60 env LG_RESCALE=1 python3 -u "$LGX_H42" --part gpu --count-only --threads "$THREADS" \
          --learners mlp,cfm,ddpm,tabm,nflow,ftt,realmlp
    timed count_h41 0 300 240 60 env LG_RESCALE=1 python3 -u "$LGX_H41" --count-only --threads "$THREADS"
  else
    skip_stage count 0 "LGX_SMOKE_DO 에 없음"
  fi

  # ---------------------------------------------------------------- 10–11 사전 점검
  if want precheck; then
    # 뒤에 남길 시간: 마무리 240 s 에 뒤 단계의 최소 시간을 더한 값. 조각은 작업 단위(gpu 는 학습기 단위)라 중단돼도 끝난 단위는 남는다.
    timed precheck_cpu 0 1200 600 120 env LG_RESCALE=1 python3 -u "$LGX_H42" --part cpu --precheck --workers 1 --threads "$THREADS" \
          --gate-tag lg_precheck --no-summarize
    if [ -n "$GPUS" ]; then
      # 본 실행과 같은 GPU 공유 조건(GPU 1장에 프로세스 3개)으로 잰다. 코어가 4개라 스레드는 1개씩이다.
      timed precheck_gpu 0 1500 $(( 420 + HAVE_RM * 300 )) 180 env LG_RESCALE=1 python3 -u "$LGX_H42" --part gpu --precheck --gpus 0 \
            --procs-per-gpu 3 --threads 1 --learners mlp,cfm,ddpm,tabm,nflow,ftt --no-summarize
      if [ "$HAVE_RM" -eq 1 ]; then
        local RMARG=()
        [ "$RM_EPOCHS" = "256" ] || RMARG=(--tag lgxrm --realmlp-epochs "$RM_EPOCHS")
        timed precheck_realmlp 0 900 420 180 env LG_RESCALE=1 python3 -u "$LGX_H42" --part gpu --precheck --axes r0 --gpus 0 --procs-per-gpu 1 \
              --threads "$THREADS" --learners realmlp ${RMARG[@]+"${RMARG[@]}"} --no-summarize
      else
        skip_stage precheck_realmlp 0 "pytabkit 없음"
      fi
    else
      skip_stage precheck_gpu 0 "GPU 가 보이지 않거나 LGX_SMOKE_GPU=skip"
    fi
    if ls "$LGX_DIR"/shards/lgx*_precheck__*_unit.json >/dev/null 2>&1; then
      timed precheck_sum 0 600 120 120 env LG_RESCALE=1 python3 -u "$LGX_H42" --precheck --summarize-only --workers 2 --threads 2 \
            --gate-tag lg_precheck
      run_stage gate_precheck 0 180 env OMP_NUM_THREADS=1 python3 -u scripts/rescale/lgx_gate_check.py --precheck --gate-tag lg_precheck \
                --out-csv lgx_precheck_gate_check.csv
    fi
    timing timing_final 0
  else
    skip_stage precheck 0 "LGX_SMOKE_DO 에 없음"
  fi
  finish
}

main "$@"
