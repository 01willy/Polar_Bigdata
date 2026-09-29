#!/bin/bash
# LG(H40) Rescale 스모크·사전 점검 작업. 설정 configs/rescale/lg_smoke.yaml (iolite-1: 4코어, GPU 1장, 벽시계 1 h).
# 로컬 공유 서버에서는 실행하지 않는다(사용자 지시 2026-09-29).
#
# 단계(앞 단계가 실패해도 뒤 단계를 진행하고, 단계마다 종료 코드를 lg_smoke_status.csv 에 남긴다)
#   1 env           python·torch·CUDA·GPU 목록·코어 수 출력
#   2 deps          필수 패키지 설치(catboost, scikit-learn, pandas, scipy. 이미 있으면 건너뜀). 실패하면 종료 코드 1 로 끝낸다
#                   선택 패키지: pytabkit(RealMLP), pytest. 없으면 해당 단계만 뺀다
#   3 smoke_cpu     h40 --smoke --part cpu (제한 20분)
#   4 smoke_gpu     h40 --smoke --part gpu --gpus 0 (학습기 7종). --epochs 5 를 주지만 h40 의 --smoke 는 epochs 를 3 이하로 자른다
#   5 smoke_sum     h40 --smoke --summarize-only → data/processed/lg/lg_smoke_{curve,minn,tests,timing,failed}.csv
#   6 timing        학습기별 적합 시간 표 → lg_smoke_timing.csv(조각·키별), lg_smoke_timing_by_learner.csv(학습기별)
#   7 pytest        LG_RUN_HEAVY=1 python3 -m pytest tests/test_h40_smoke.py (pytest 가 있을 때만)
#   8 count         본 실행 범위의 적합 수 표(--count-only). 시간 환산의 분모로 쓴다
#   9 precheck      남은 시간 안에서 사전 점검(--precheck: Canada x, 분할 1, n {0, 40}, 본 실행과 같은 epochs)
#                   cpu 부분 → gpu 부분(RealMLP 제외 6종) → RealMLP 시간 측정(tag lgrm, epochs 축소)
#  10 묶음          results_lg_smoke.tar.gz. 단계 6·7·9 뒤에 각각 다시 묶는다(벽시계 상한에 걸려도 앞 단계 결과가 남는다)
#
# 환경 변수(설정 파일의 command 에서 준다)
#   LG_SMOKE_BUDGET_MIN   스크립트 시작부터 쓸 수 있는 시간(분). 기본 50. 사전 점검은 이 안에서만 돈다
#   LG_SMOKE_PRECHECK     0 이면 단계 8·9 를 생략한다. 기본 1
#   LG_SMOKE_LEARNERS     스모크 gpu 부분의 학습기. 기본 mlp,cfm,ddpm,tabm,nflow,ftt,realmlp
#   LG_RM_EPOCHS          RealMLP 시간 측정의 epochs. 기본 8 (본 실행 256 으로 환산한다)
#   LG_KEEP_INPUTS        1 이면 끝난 뒤 푼 코드·입력 자료를 지우지 않는다. 기본 0

main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -f lg_payload.tar.gz ] && [ ! -f data/processed/fidelity_base_v3.csv ]; then
    tar xzf lg_payload.tar.gz || { echo "[smoke] 묶음 해제 실패"; exit 1; }
  fi
  # shellcheck source=scripts/rescale/lg_common.sh
  source scripts/rescale/lg_common.sh || exit 1
  lg_base_env

  local T0 BUDGET_S H40 LOGD STATUS LGD THREADS LEARNERS RM_EPOCHS PRECHECK
  T0=$(date +%s)
  BUDGET_S=$(( ${LG_SMOKE_BUDGET_MIN:-50} * 60 ))
  PRECHECK="${LG_SMOKE_PRECHECK:-1}"
  LEARNERS="${LG_SMOKE_LEARNERS:-mlp,cfm,ddpm,tabm,nflow,ftt,realmlp}"
  RM_EPOCHS="${LG_RM_EPOCHS:-8}"
  H40=scripts/3_deep_learning/h40_label_grid.py
  LGD=data/processed/lg
  LOGD=logs_lg_smoke
  STATUS=lg_smoke_status.csv
  THREADS=$(lg_ncores)
  [ "$THREADS" -gt 4 ] && THREADS=4
  mkdir -p "$LOGD" "$LGD"
  echo "stage,required,rc,sec,note" > "$STATUS"
  [ -f lg_payload_info.txt ] && sed 's/^/[payload] /' lg_payload_info.txt

  left() { echo $(( BUDGET_S - ( $(date +%s) - T0 ) )); }

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

  pack() {
    lg_pack results_lg_smoke.tar.gz "$LGD" "$LOGD" "$STATUS" lg_smoke_timing.csv lg_smoke_timing_by_learner.csv \
            lg_smoke_full_projection.csv lg_payload_info.txt
  }

  # timing <단계 이름> <필수 0|1>. lg_smoke_timing.csv = h40 집계의 조각·키별 표, lg_smoke_timing_by_learner.csv = 학습기별 표
  timing() {
    [ -f "$LGD/lg_smoke_timing.csv" ] && cp -f "$LGD/lg_smoke_timing.csv" lg_smoke_timing.csv
    run_stage "$1" "$2" 300 python3 -u scripts/rescale/lg_timing_table.py --lg-dir "$LGD" --out-dir . --prefix lg_smoke
  }

  finish() {                                                              # 상태 요약, 묶음, 정리, 종료 코드
    pack
    lg_log "[smoke] 단계별 상태"
    sed 's/^/  /' "$STATUS"
    if [ "${LG_KEEP_INPUTS:-0}" != "1" ]; then                           # 푼 코드·입력 자료는 출력이 아니다(출력 동기화 절약)
      rm -rf src scripts tests .pydeps .lg_pip_constraints.txt lg_payload.tar.gz \
             data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/lg_subregion_map_v1.csv
    fi
    local bad
    bad=$(awk -F, 'NR > 1 && $2 == 1 && $3 != 0 {print $1}' "$STATUS" | tr '\n' ' ')
    if [ -n "$bad" ]; then
      lg_log "[smoke] 실패한 필수 단계: $bad"
      exit 1
    fi
    bad=$(awk -F, 'NR > 1 && $2 == 0 && $3 != 0 && $3 != "NA" {print $1}' "$STATUS" | tr '\n' ' ')
    if [ -n "$bad" ]; then
      lg_log "[smoke] 필수 단계는 모두 통과했다. 끝나지 않은 선택 단계: $bad"
      exit 2
    fi
    lg_log "[smoke] 모든 단계 통과 · 경과 $(( ( $(date +%s) - T0 ) / 60 )) min"
    exit 0
  }

  # ---------------------------------------------------------------- 1 환경
  lg_env_report 2>&1 | tee "$LOGD/env.log"
  lg_check_inputs || { echo "inputs,1,1,0,입력 파일 없음" >> "$STATUS"; finish; }

  # ---------------------------------------------------------------- 2 의존성
  if lg_install_deps 2>&1 | tee "$LOGD/deps.log"; [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "deps,1,1,$(( $(date +%s) - T0 )),필수 패키지 설치 실패" >> "$STATUS"
    lg_log "[smoke] 필수 패키지 설치 실패. 종료 코드 1"
    finish
  fi
  [ -d .pydeps ] && export PYTHONPATH="$PWD/.pydeps${PYTHONPATH:+:$PYTHONPATH}"     # 파이프 안에서 설치한 경우의 경로 반영
  echo "deps,1,0,$(( $(date +%s) - T0 )),ok" >> "$STATUS"
  local HAVE_RM=1 HAVE_PT=1
  lg_install_opt pytabkit "pytabkit==1.7.3" "pytabkit" 2>&1 | tee -a "$LOGD/deps.log"
  python3 -c "import pytabkit" >/dev/null 2>&1 || HAVE_RM=0
  lg_install_opt pytest "pytest>=7" 2>&1 | tee -a "$LOGD/deps.log"
  python3 -c "import pytest" >/dev/null 2>&1 || HAVE_PT=0
  [ -d .pydeps ] && export PYTHONPATH="$PWD/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
  if [ "$HAVE_RM" -eq 0 ]; then
    LEARNERS=$(echo "$LEARNERS" | tr ',' '\n' | grep -vx realmlp | paste -sd, -)
    echo "deps_pytabkit,0,1,0,pytabkit 없음. RealMLP 를 뺀다" >> "$STATUS"
  fi
  lg_env_report 2>&1 | grep -E "^\[env\] (numpy|pandas|scipy|scikit-learn|catboost|pytabkit|pytest|torch) " | tee "$LOGD/env_after_deps.log"

  local GPUS
  GPUS=$(lg_gpu_list)
  lg_log "[smoke] 보이는 GPU: '${GPUS}' · 스레드 ${THREADS} · 스모크 학습기 ${LEARNERS} · 시간 한도 $(( BUDGET_S / 60 )) min"

  # ---------------------------------------------------------------- 3–6 스모크
  run_stage smoke_cpu 1 1200 env LG_RESCALE=1 python3 -u "$H40" --part cpu --smoke --workers 1 --threads "$THREADS" --no-summarize
  if [ -n "$GPUS" ]; then
    # 실제 GPU 1장. --epochs 5 를 주지만 h40 finalize 가 스모크의 epochs 를 min(epochs, 3) 으로 자른다(실제 값은 unit.json 의 cfg.epochs).
    run_stage smoke_gpu 1 1200 env LG_RESCALE=1 python3 -u "$H40" --part gpu --smoke --gpus 0 --procs-per-gpu 1 --threads "$THREADS" \
              --epochs 5 --learners "$LEARNERS" --no-summarize
  else
    echo "smoke_gpu,1,1,0,GPU 가 보이지 않는다" >> "$STATUS"
    lg_log "[smoke] GPU 가 보이지 않는다. gpu 부분을 실행하지 않고 실패로 기록한다"
  fi
  run_stage smoke_sum 1 600 python3 -u "$H40" --smoke --summarize-only --threads "$THREADS"
  timing timing 1
  pack

  # ---------------------------------------------------------------- 7 단위 시험
  if [ "$HAVE_PT" -eq 1 ]; then
    local lim
    lim=$(left); [ "$lim" -gt 1200 ] && lim=1200
    if [ "$lim" -gt 120 ]; then
      run_stage pytest 1 "$lim" env LG_RUN_HEAVY=1 CUDA_VISIBLE_DEVICES= python3 -m pytest -q -p no:cacheprovider tests/test_h40_smoke.py
    else
      skip_stage pytest 0 "남은 시간 부족"
    fi
  else
    skip_stage pytest 0 "pytest 없음"
  fi
  pack

  # ---------------------------------------------------------------- 8–9 적합 수 표와 사전 점검
  if [ "$PRECHECK" = "1" ]; then
    local lim
    run_stage count_cpu 0 300 python3 -u "$H40" --part cpu --count-only --threads "$THREADS"
    run_stage count_gpu 0 300 python3 -u "$H40" --part gpu --count-only --threads "$THREADS"
    lim=$(( $(left) - 240 )); [ "$lim" -gt 600 ] && lim=600
    if [ "$lim" -gt 60 ]; then
      run_stage precheck_cpu 0 "$lim" env LG_RESCALE=1 python3 -u "$H40" --part cpu --precheck --workers 1 --threads "$THREADS" --no-summarize
    else
      skip_stage precheck_cpu 0 "남은 시간 부족"
    fi
    if [ -n "$GPUS" ]; then
      # RealMLP 시간 측정에 남길 시간(10분)과 마무리 시간(4분)을 뺀 만큼만 쓴다. 조각은 학습기 단위라 중단돼도 끝난 학습기는 남는다.
      # 본 실행과 같은 GPU 공유 조건(GPU 1장에 프로세스 3개)으로 잰다. 코어가 4개라 스레드는 1개씩이다.
      lim=$(( $(left) - 240 - ( HAVE_RM * 600 ) ))
      if [ "$lim" -gt 180 ]; then
        run_stage precheck_gpu 0 "$lim" env LG_RESCALE=1 python3 -u "$H40" --part gpu --precheck --gpus 0 --procs-per-gpu 3 --threads 1 \
                  --learners mlp,cfm,ddpm,tabm,nflow,ftt --no-summarize
      else
        skip_stage precheck_gpu 0 "남은 시간 부족"
      fi
      lim=$(( $(left) - 240 ))
      if [ "$HAVE_RM" -eq 1 ] && [ "$lim" -gt 180 ]; then
        # RealMLP 는 본 실행 epochs(256)로는 이 작업 안에 끝나지 않는다. epochs 를 줄여 재고 환산한다. tag 를 따로 둬 집계에 섞지 않는다.
        run_stage precheck_realmlp 0 "$lim" env LG_RESCALE=1 python3 -u "$H40" --part gpu --precheck --tag lgrm --gpus 0 --procs-per-gpu 1 \
                  --threads "$THREADS" --learners realmlp --realmlp-epochs "$RM_EPOCHS" --no-summarize
      else
        skip_stage precheck_realmlp 0 "pytabkit 없음 또는 남은 시간 부족"
      fi
    else
      skip_stage precheck_gpu 0 "GPU 가 보이지 않는다"
    fi
    if ls "$LGD"/shards/lg_precheck__*_unit.json >/dev/null 2>&1; then
      run_stage precheck_sum 0 240 python3 -u "$H40" --precheck --summarize-only --threads "$THREADS"
    fi
    timing timing_final 0
  else
    skip_stage precheck 0 "LG_SMOKE_PRECHECK=0"
  fi
  finish
}

main "$@"
