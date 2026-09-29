#!/bin/bash
# LGX(H41·H42) Rescale 본 실행. 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6A.
# 설정: configs/rescale/lgx_cpu.yaml(CPU 부분과 검증 사다리), lgx_gpu.yaml(GPU 부분), lgx_sum.yaml(통합 집계).
# 로컬 공유 서버에서는 실행하지 않는다(사용자 지시 2026-09-29).
#
# 부분(LGX_PARTS)
#   cpu   h42 --part cpu (축 base, x1, x2, x9, n3, prox, x5, x9f, n4, n4a, ridge)
#   h41   h41 --part all (검증 사다리 X3a, 지역 내 블록 교차검증 X3b)
#   gpu   h42 --part gpu (축 r0, g45, nn)
#   비어 있으면 실행 없이 복원과 집계만 한다(통합 집계 작업)
#
# 순서
#   1 묶음 해제, 환경 출력, 입력 확인(고정 입력 표 3종 포함), 필수 패키지 설치(사전 점검과 같은 판), GPU 확인(gpu 부분이 있을 때)
#   2 입력 묶음 복원: results_lg*.tar.gz(본 실행 조각, 읽기 전용 참조)와 results_lgx*.tar.gz(확장 조각. --resume 이 건너뛴다)
#   3 부분을 백그라운드로 함께 실행한다(각각 --resume --no-summarize, 시간 한도는 timeout 으로 건다)
#     실행 순서는 하네스가 정한다(h42 cpu: 확인적 가설의 구성 단위 먼저. h42 gpu: 빠른 학습기 먼저, RealMLP 가 맨 뒤. h41: V-G, V-R 먼저)
#   4 5분마다 진행 줄, 30분마다 결과 묶음 갱신(임시 파일에 쓴 뒤 교체. 벽시계 상한에 걸려도 마지막 묶음이 남는다)
#     기준 축 조각이 생기면 재현 점검을 한 번 먼저 돌려 로그에 남긴다(lgx_gate_early.csv. 실행을 멈추지는 않는다)
#   5 부분이 모두 끝나거나 시간 한도에 닿으면 집계(h42, h41 의 --summarize-only), 시간 표, 마지막 묶음, 정리
#
# 환경 변수(설정 파일의 command 에서 준다)
#   LGX_PARTS            실행할 부분. 기본 "cpu h41 gpu". 빈 문자열이면 집계만 한다
#   LGX_LABEL            결과 묶음 이름의 마디(results_lgx_<마디>.tar.gz). 기본: 부분에 따라 cpu, gpu, all, sum
#   LGX_BUDGET_MIN       스크립트 시작부터 쓸 시간(분). 기본 315. 벽시계 상한보다 45분 짧게 둔다
#   LGX_SUM_RESERVE_MIN  집계와 묶음에 남길 시간(분). 기본 60. 실행은 (시간 한도 − 이 값)에 멈춘다
#   LGX_SUMMARIZE        1 이면 끝에 집계한다. 기본 1. GPU 부분만 도는 작업은 0 으로 둔다(집계는 통합 집계 작업에서 한다)
#   LGX_PACK_SHARDS      1 이면 결과 묶음에 조각을 넣는다. 기본 1. 통합 집계 작업은 0 으로 둔다(조각은 입력 묶음에 있다)
#   LGX_GPU_LEARNERS     GPU 부분 학습기. 기본 all(7종: mlp, cfm, ddpm, tabm, nflow, ftt, realmlp)
#   LGX_CPU_AXES         h42 cpu 부분의 축(쉼표 목록). 기본은 전부      LGX_GPU_AXES   h42 gpu 부분의 축. 기본은 전부
#   LGX_GPUS             GPU 목록. 기본 0,1,2,3             LGX_GPU_PROCS    GPU 하나당 프로세스 수. 기본 3
#   LGX_GPU_THREADS      GPU 워커의 스레드 수. 기본 2        LGX_CPU_THREADS  CPU 워커의 스레드 수. 기본 4
#   LGX_CPU_WORKERS      h42 cpu 워커 수. 기본: (쓸 코어 / 스레드 수)에서 h41 워커 수를 뺀 값
#   LGX_H41_WORKERS      h41 워커 수. 기본 2(h42 cpu 부분과 함께 돌 때), h41 만 돌 때는 (쓸 코어 / 스레드 수)
#   LGX_SUM_WORKERS      집계의 곡선 계산 프로세스 수. 기본 min(14, 코어 수 / 4)
#   LGX_KEEP_TREE        1 이면 끝난 뒤 조각 디렉터리와 푼 입력을 지우지 않는다. 기본 0
#
# 종료 코드: 0 = 모든 부분과 집계가 0 으로 끝났다, 1 = 실패 또는 시간 한도로 중단된 부분이 있다, 2 = 경고만 있다(판 불일치, 빠진 GPU 등)

main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -e .git ]; then                                                    # 끝에 푼 코드와 입력을 지우므로 저장소 안에서는 실행하지 않는다
    echo "[lgx] 저장소(.git)가 있는 디렉터리다. 이 스크립트는 Rescale 작업 디렉터리에서만 실행한다"; exit 1
  fi
  if [ -f lgx_payload.tar.gz ] && [ ! -f data/processed/fidelity_base_v3.csv ]; then
    tar xzf lgx_payload.tar.gz || { echo "[lgx] 묶음 해제 실패"; exit 1; }
  fi
  # shellcheck source=scripts/rescale/lg_common.sh
  source scripts/rescale/lg_common.sh || exit 1
  # shellcheck source=scripts/rescale/lgx_common.sh
  source scripts/rescale/lgx_common.sh || exit 1
  lg_base_env

  local T0 BUDGET_S SUM_RESERVE PACK_EVERY PROG_EVERY NCORE PARTS LABEL OUTB LOGD STATUS
  local GPUS GPU_PROCS GPU_THREADS CPU_THREADS CPU_WORKERS H41_WORKERS SUM_WORKERS GPU_LEARNERS CPU_AXES GPU_AXES DO_SUM PACK_SHARDS
  local HAS_C=0 HAS_V=0 HAS_G=0 AVAIL SLOTS
  T0=$(date +%s)
  BUDGET_S=$(( ${LGX_BUDGET_MIN:-315} * 60 ))
  SUM_RESERVE=$(( ${LGX_SUM_RESERVE_MIN:-60} * 60 ))
  PACK_EVERY=$(( ${LGX_PACK_MIN:-30} * 60 ))
  PROG_EVERY=$(( ${LGX_PROGRESS_MIN:-5} * 60 ))
  NCORE=$(lg_ncores)
  PARTS="${LGX_PARTS-cpu h41 gpu}"
  case " $PARTS " in *" cpu "*) HAS_C=1 ;; esac
  case " $PARTS " in *" h41 "*) HAS_V=1 ;; esac
  case " $PARTS " in *" gpu "*) HAS_G=1 ;; esac
  if [ -n "${LGX_LABEL:-}" ]; then LABEL="$LGX_LABEL"
  elif [ "$HAS_G" -eq 1 ] && [ $(( HAS_C + HAS_V )) -gt 0 ]; then LABEL=all
  elif [ "$HAS_G" -eq 1 ]; then LABEL=gpu
  elif [ $(( HAS_C + HAS_V )) -gt 0 ]; then LABEL=cpu
  else LABEL=sum; fi
  OUTB="results_lgx_${LABEL}.tar.gz"
  LOGD="logs_lgx_${LABEL}"
  STATUS="lgx_${LABEL}_status.csv"
  GPUS="${LGX_GPUS:-0,1,2,3}"
  GPU_PROCS="${LGX_GPU_PROCS:-3}"
  GPU_THREADS="${LGX_GPU_THREADS:-2}"
  CPU_THREADS="${LGX_CPU_THREADS:-4}"
  AVAIL=$NCORE
  [ "$HAS_G" -eq 1 ] && [ $(( HAS_C + HAS_V )) -gt 0 ] && AVAIL=$(( NCORE - 8 ))      # GPU 워커와 함께 돌 때는 8코어를 남긴다(본 실행과 같다)
  SLOTS=$(( AVAIL / CPU_THREADS )); [ "$SLOTS" -ge 1 ] || SLOTS=1
  if [ "$HAS_C" -eq 1 ] && [ "$HAS_V" -eq 1 ]; then
    H41_WORKERS="${LGX_H41_WORKERS:-2}"
    CPU_WORKERS="${LGX_CPU_WORKERS:-$(( SLOTS - H41_WORKERS ))}"
  else
    H41_WORKERS="${LGX_H41_WORKERS:-$SLOTS}"
    CPU_WORKERS="${LGX_CPU_WORKERS:-$SLOTS}"
  fi
  [ "$CPU_WORKERS" -ge 1 ] || CPU_WORKERS=1
  [ "$H41_WORKERS" -ge 1 ] || H41_WORKERS=1
  SUM_WORKERS="${LGX_SUM_WORKERS:-$(( NCORE / 4 ))}"
  [ "$SUM_WORKERS" -le 14 ] || SUM_WORKERS=14
  [ "$SUM_WORKERS" -ge 1 ] || SUM_WORKERS=1
  GPU_LEARNERS="${LGX_GPU_LEARNERS:-all}"
  [ "$GPU_LEARNERS" = "all" ] && GPU_LEARNERS="mlp,cfm,ddpm,tabm,nflow,ftt,realmlp"
  CPU_AXES="${LGX_CPU_AXES:-}"
  GPU_AXES="${LGX_GPU_AXES:-}"
  DO_SUM="${LGX_SUMMARIZE:-1}"
  PACK_SHARDS="${LGX_PACK_SHARDS:-1}"
  mkdir -p "$LOGD" "$LGX_DIR"
  echo "stage,rc,sec,note" > "$STATUS"
  [ -f lgx_payload_info.txt ] && sed 's/^/[payload] /' lgx_payload_info.txt

  left() { echo $(( BUDGET_S - ( $(date +%s) - T0 ) )); }
  note_stage() { echo "$1,$2,$(( $(date +%s) - T0 )),$3" >> "$STATUS"; }
  pack() {
    lgx_pack "$OUTB" "$PACK_SHARDS" "$LGX_DIR" "$LOGD" "$STATUS" "lgx_${LABEL}_timing_by_learner.csv" "lgx_${LABEL}_full_projection.csv" \
             lgx_gate_early.csv lgx_payload_info.txt
  }
  last_done() { grep -ao "완료 [0-9]*/[0-9]*" "$1" 2>/dev/null | tail -1; }
  n_fail() { local n; n=$(grep -ac "\[FAIL\]" "$1" 2>/dev/null); echo "${n:-0}"; }
  progress() {
    local gu=""
    [ "$HAS_G" -eq 1 ] && gu=" · GPU 사용률 $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd/ -) %"
    lg_log "[progress] 경과 $(( ( $(date +%s) - T0 ) / 60 )) min" \
           "· h42 cpu 조각 $(lgx_count_units cpu)개($(last_done "$LOGD/lgx_cpu.log")) 실패 $(n_fail "$LOGD/lgx_cpu.log")" \
           "· h41 조각 $(lgx_count_units h41)개($(last_done "$LOGD/lgx_h41.log")) 실패 $(n_fail "$LOGD/lgx_h41.log")" \
           "· h42 gpu 조각 $(lgx_count_units gpu)개($(last_done "$LOGD/lgx_gpu.log")) 실패 $(n_fail "$LOGD/lgx_gpu.log")${gu}" \
           "· 부하 $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null) · 디스크 여유 $(df -h . 2>/dev/null | awk 'NR==2{print $4}')"
  }
  finish() {                                                              # finish <종료 코드>
    pack
    lg_log "[lgx] 단계별 상태"
    sed 's/^/  /' "$STATUS"
    if [ "${LGX_KEEP_TREE:-0}" != "1" ]; then                            # 푼 코드와 입력은 출력이 아니다(출력 동기화 절약)
      local f
      rm -rf src scripts tests .pydeps .lg_pip_constraints.txt .lgx_pip_constraints.txt lgx_payload.tar.gz "$LGX_LG_DIR" \
             data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/lg_subregion_map_v1.csv "${LGX_TABLES[@]}"
      for f in results_lg_prev.tar.gz results_lg.tar.gz results_lgx*.tar.gz; do
        [ -f "$f" ] && [ "$f" != "$OUTB" ] && rm -f "$f"
      done
      if [ "$PACK_SHARDS" != "1" ]; then
        rm -rf "$LGX_DIR/shards" "$LGX_LADDER_DIR/shards"                  # 조각은 입력 묶음에 있다
      elif [ -f "$OUTB" ] && gzip -t "$OUTB" 2>/dev/null \
           && [ "$(tar tzf "$OUTB" | grep -c '_unit\.json$')" -eq "$(find "$LGX_DIR" -name '*_unit.json' 2>/dev/null | wc -l)" ]; then
        rm -rf "$LGX_DIR/shards" "$LGX_LADDER_DIR/shards"                  # 조각은 묶음에 있다. 집계 표는 남긴다
      else
        lg_log "[lgx] 묶음 확인 실패. 조각 디렉터리를 남긴다"
      fi
    fi
    lg_log "[lgx] 끝 · 종료 코드 $1 · 경과 $(( ( $(date +%s) - T0 ) / 60 )) min"
    exit "$1"
  }

  # ---------------------------------------------------------------- 1 환경·입력·의존성·GPU
  local WARN=0
  lg_env_report 2>&1 | tee "$LOGD/env.log"
  lgx_check_inputs || { note_stage inputs 1 "입력 파일 또는 고정 입력 표 없음"; finish 1; }
  note_stage inputs 0 ok
  if lgx_install_deps 2>&1 | tee "$LOGD/deps.log"; [ "${PIPESTATUS[0]}" -ne 0 ]; then
    note_stage deps 1 "필수 패키지 설치 실패"
    finish 1
  fi
  [ -d .pydeps ] && export PYTHONPATH="$PWD/.pydeps${PYTHONPATH:+:$PYTHONPATH}"       # 파이프 안에서 설치한 경우의 경로 반영
  note_stage deps 0 ok
  if lgx_versions_check 2>&1 | tee -a "$LOGD/deps.log"; [ "${PIPESTATUS[0]}" -ne 0 ]; then
    note_stage deps_versions 2 "설치된 판이 사전 점검(qOkSo)의 판과 다르다. 재현 점검의 허용 차를 넘을 수 있다"
    WARN=1
  else
    note_stage deps_versions 0 "사전 점검과 같은 판"
  fi
  if [ "$HAS_G" -eq 1 ]; then
    case ",$GPU_LEARNERS," in *,realmlp,*)
      lg_install_opt pytabkit "pytabkit==1.7.3" "pytabkit" 2>&1 | tee -a "$LOGD/deps.log"
      [ -d .pydeps ] && export PYTHONPATH="$PWD/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
      if ! python3 -c "import pytabkit" >/dev/null 2>&1; then
        GPU_LEARNERS=$(echo "$GPU_LEARNERS" | tr ',' '\n' | grep -vx realmlp | paste -sd, -)
        note_stage deps_pytabkit 1 "pytabkit 없음. RealMLP 를 뺀다(사전 등록 범위가 채워지지 않는다. 다시 실행해야 한다)"
        WARN=1
      fi ;;
    esac
  fi
  lg_env_report 2>&1 | grep -E "^\[env\] (numpy|pandas|scipy|scikit-learn|catboost|pytabkit|torch) " | tee "$LOGD/env_after_deps.log"

  if [ "$HAS_G" -eq 1 ]; then
    local SEEN g keep=""
    SEEN=$(lg_gpu_list)
    if [ -z "$SEEN" ]; then
      lg_log "[lgx] GPU 가 보이지 않는다. 비용이 나가기 전에 멈춘다"
      note_stage gpu_check 1 "GPU 가 보이지 않는다"
      finish 1
    fi
    for g in ${GPUS//,/ }; do
      case ",$SEEN," in *",$g,"*) keep="${keep:+$keep,}$g" ;; *) lg_log "[lgx] 경고: GPU $g 가 보이지 않는다. 목록에서 뺀다"; WARN=1 ;; esac
    done
    GPUS="$keep"
    [ -n "$GPUS" ] || { note_stage gpu_check 1 "요청한 GPU 가 하나도 보이지 않는다"; finish 1; }
    [ -n "$GPU_LEARNERS" ] || { note_stage gpu_check 1 "GPU 부분에 돌릴 학습기가 없다"; finish 1; }
    note_stage gpu_check 0 "GPU $GPUS"
  fi

  # ---------------------------------------------------------------- 2 입력 묶음 복원
  lgx_restore 2>&1 | tee "$LOGD/restore.log"
  local GATE_TAG GATE_ARG=()
  GATE_TAG=$(lgx_gate_tag)
  [ -n "$GATE_TAG" ] && [ "$GATE_TAG" != "lg" ] && GATE_ARG=(--gate-tag "$GATE_TAG")
  case "$GATE_TAG" in
    lg) lg_log "[lgx] 재현 점검의 기준: 본 실행 조각(tag lg) $(find "$LGX_LG_DIR/shards" -name 'lg__cpu__*_unit.json' | wc -l)개" ;;
    "") lg_log "[lgx] 재현 점검의 기준 조각이 없다. lgx_gate.csv 는 '기준 조각 없음'이 된다" ;;
    *)  lg_log "[lgx] 재현 점검의 기준: LG 사전 점검 조각(tag $GATE_TAG, 캐나다 x 분할 1). 본 실행 조각과의 대조는 통합 집계 작업에서 한다" ;;
  esac
  note_stage restore 0 "확장 조각 $(lgx_count_units all)개 · 재현 점검 기준 '${GATE_TAG}'"
  if [ $(( HAS_C + HAS_V + HAS_G )) -eq 0 ] && [ "$(lgx_count_units all)" -eq 0 ]; then
    lg_log "[lgx] 실행할 부분이 없고 복원된 조각도 없다. 입력 묶음(dataset_file_ids)을 확인한다"
    note_stage shards 1 "집계할 조각이 없다"
    finish 1
  fi

  lg_log "[lgx] 코어 ${NCORE} · 부분 '${PARTS}' · 묶음 ${OUTB} · h42 cpu 워커 ${CPU_WORKERS} × 스레드 ${CPU_THREADS} · h41 워커 ${H41_WORKERS} × 스레드 ${CPU_THREADS}" \
         "· GPU '${GPUS}' × 프로세스 ${GPU_PROCS} × 스레드 ${GPU_THREADS} · 집계 프로세스 ${SUM_WORKERS}"
  lg_log "[lgx] GPU 학습기 '${GPU_LEARNERS}' · cpu 축 '${CPU_AXES:-전부}' · gpu 축 '${GPU_AXES:-전부}' · 시간 한도 $(( BUDGET_S / 60 )) min(집계 예비 $(( SUM_RESERVE / 60 )) min)" \
         "· 집계 ${DO_SUM} · 조각 묶음 ${PACK_SHARDS}"

  # ---------------------------------------------------------------- 3 실행
  local RUN_S PID_C="" PID_V="" PID_G="" RC_C=0 RC_V=0 RC_G=0 AX_C=() AX_G=()
  [ -n "$CPU_AXES" ] && AX_C=(--axes "$CPU_AXES")
  [ -n "$GPU_AXES" ] && AX_G=(--axes "$GPU_AXES")
  RUN_S=$(( $(left) - SUM_RESERVE ))
  if [ $(( HAS_C + HAS_V + HAS_G )) -gt 0 ]; then
    [ "$RUN_S" -gt 60 ] || { note_stage budget 1 "시간 한도가 집계 예비 시간보다 짧다"; finish 1; }
    # timeout 은 자기 프로세스 그룹 전체에 신호를 보낸다. 하네스의 워커 프로세스도 함께 끝난다.
    if [ "$HAS_G" -eq 1 ]; then
      timeout -k 60 "$RUN_S" env LG_RESCALE=1 python3 -u "$LGX_H42" --part gpu ${AX_G[@]+"${AX_G[@]}"} --gpus "$GPUS" --procs-per-gpu "$GPU_PROCS" \
        --threads "$GPU_THREADS" --learners "$GPU_LEARNERS" --resume --no-summarize > "$LOGD/lgx_gpu.log" 2>&1 &
      PID_G=$!
      lg_log "[lgx] h42 gpu 부분 시작(pid $PID_G) → $LOGD/lgx_gpu.log"
    fi
    if [ "$HAS_C" -eq 1 ]; then
      timeout -k 60 "$RUN_S" env LG_RESCALE=1 python3 -u "$LGX_H42" --part cpu ${AX_C[@]+"${AX_C[@]}"} --workers "$CPU_WORKERS" --threads "$CPU_THREADS" \
        --resume --no-summarize > "$LOGD/lgx_cpu.log" 2>&1 &
      PID_C=$!
      lg_log "[lgx] h42 cpu 부분 시작(pid $PID_C) → $LOGD/lgx_cpu.log"
    fi
    if [ "$HAS_V" -eq 1 ]; then
      timeout -k 60 "$RUN_S" env LG_RESCALE=1 python3 -u "$LGX_H41" --part all --workers "$H41_WORKERS" --threads "$CPU_THREADS" \
        --resume --no-summarize > "$LOGD/lgx_h41.log" 2>&1 &
      PID_V=$!
      lg_log "[lgx] h41 시작(pid $PID_V) → $LOGD/lgx_h41.log"
    fi
  fi

  # ---------------------------------------------------------------- 4 감시
  local now last_prog last_pack done_c=1 done_v=1 done_g=1 gate_done=0 gate_try=0 rcg
  last_prog=$(date +%s); last_pack=$last_prog
  [ -n "$PID_C" ] && done_c=0
  [ -n "$PID_V" ] && done_v=0
  [ -n "$PID_G" ] && done_g=0
  [ "$HAS_C" -eq 1 ] && [ -n "$GATE_TAG" ] || gate_done=1                 # 기준 축은 cpu 부분에 있다
  early_gate() {                                                          # 기준 축 조각이 생기면 재현 점검을 한 번 먼저 본다
    [ "$gate_done" -eq 0 ] || return 0
    ls "$LGX_DIR"/shards/lgxb__cpu__*_unit.json >/dev/null 2>&1 || return 0
    gate_try=$(( gate_try + 1 ))
    timeout 180 env OMP_NUM_THREADS=1 python3 -u scripts/rescale/lgx_gate_check.py --gate-tag "$GATE_TAG" --out-csv lgx_gate_early.csv \
      > "$LOGD/gate_early.log" 2>&1
    rcg=$?
    if [ "$rcg" -eq 0 ] || [ "$rcg" -eq 3 ]; then
      sed 's/^/  /' "$LOGD/gate_early.log"
      note_stage gate_early "$rcg" "$([ "$rcg" -eq 0 ] && echo '허용 차 안' || echo '허용 차를 넘은 단위가 있다(실행은 계속한다)') · 기준 ${GATE_TAG}"
      [ "$rcg" -eq 3 ] && WARN=1
      gate_done=1
    elif [ "$gate_try" -ge 12 ]; then
      tail -n 5 "$LOGD/gate_early.log" | sed 's/^/  /'
      note_stage gate_early "$rcg" "대조할 단위를 찾지 못했다(시도 ${gate_try}회)"
      gate_done=1
    fi
    return 0
  }
  while [ "$done_c" -eq 0 ] || [ "$done_v" -eq 0 ] || [ "$done_g" -eq 0 ]; do
    sleep 20
    if [ "$done_g" -eq 0 ] && ! kill -0 "$PID_G" 2>/dev/null; then
      wait "$PID_G"; RC_G=$?; done_g=1
      note_stage gpu "$RC_G" "$([ "$RC_G" -eq 124 ] && echo '시간 한도로 중단' || echo '')"
      lg_log "[lgx] h42 gpu 부분 끝: 종료 코드 $RC_G"; tail -n 5 "$LOGD/lgx_gpu.log" | sed 's/^/  [gpu] /'
      progress; pack; last_pack=$(date +%s)
    fi
    if [ "$done_c" -eq 0 ] && ! kill -0 "$PID_C" 2>/dev/null; then
      wait "$PID_C"; RC_C=$?; done_c=1
      note_stage cpu "$RC_C" "$([ "$RC_C" -eq 124 ] && echo '시간 한도로 중단' || echo '')"
      lg_log "[lgx] h42 cpu 부분 끝: 종료 코드 $RC_C"; tail -n 5 "$LOGD/lgx_cpu.log" | sed 's/^/  [cpu] /'
      progress; pack; last_pack=$(date +%s)
    fi
    if [ "$done_v" -eq 0 ] && ! kill -0 "$PID_V" 2>/dev/null; then
      wait "$PID_V"; RC_V=$?; done_v=1
      note_stage h41 "$RC_V" "$([ "$RC_V" -eq 124 ] && echo '시간 한도로 중단' || echo '')"
      lg_log "[lgx] h41 끝: 종료 코드 $RC_V"; tail -n 5 "$LOGD/lgx_h41.log" | sed 's/^/  [h41] /'
      progress; pack; last_pack=$(date +%s)
    fi
    now=$(date +%s)
    if [ $(( now - last_prog )) -ge "$PROG_EVERY" ]; then progress; early_gate; last_prog=$now; fi
    if [ $(( now - last_pack )) -ge "$PACK_EVERY" ]; then pack; last_pack=$(date +%s); fi
  done
  early_gate

  # ---------------------------------------------------------------- 5 집계·묶음
  local f
  for f in "$LOGD/lgx_cpu.log" "$LOGD/lgx_h41.log" "$LOGD/lgx_gpu.log"; do
    [ -f "$f" ] || continue
    lg_log "[lgx] $f 의 실패·경고 줄(앞 40줄)"
    grep -a -E "\[FAIL\]|\[warn\]|\[pool\]|\[ext\]|거부|Traceback" "$f" | head -n 40 | sed 's/^/  /'
    grep -a -E "^\[(data|plan|done)\]" "$f" | sed 's/^/  /'
  done
  local lim RC_S=0 RC_SV=0 n_x n_v
  n_x=$(( $(lgx_count_units cpu) + $(lgx_count_units gpu) ))
  n_v=$(lgx_count_units h41)
  if [ "$DO_SUM" = "1" ]; then
    if [ "$n_x" -gt 0 ]; then
      lim=$(( $(left) - 300 - ( n_v > 0 ? 900 : 0 ) )); [ "$lim" -gt 300 ] || lim=300
      lg_log "[lgx] h42 집계 시작(조각 ${n_x}개, 제한 ${lim} s, 프로세스 ${SUM_WORKERS})"
      timeout -k 30 "$lim" env LG_RESCALE=1 python3 -u "$LGX_H42" --summarize-only --workers "$SUM_WORKERS" --threads "$CPU_THREADS" \
        ${GATE_ARG[@]+"${GATE_ARG[@]}"} 2>&1 | tee "$LOGD/lgx_summarize.log"
      RC_S=${PIPESTATUS[0]}
      note_stage summarize_h42 "$RC_S" "$([ "$RC_S" -eq 124 ] && echo '시간 제한으로 중단' || echo '')"
      pack
    else
      note_stage summarize_h42 NA "h42 조각 없음"
    fi
    if [ "$n_v" -gt 0 ]; then
      lim=$(( $(left) - 300 )); [ "$lim" -gt 300 ] || lim=300
      lg_log "[lgx] h41 집계 시작(조각 ${n_v}개, 제한 ${lim} s)"
      timeout -k 30 "$lim" env LG_RESCALE=1 python3 -u "$LGX_H41" --summarize-only --threads 8 2>&1 | tee "$LOGD/lgv_summarize.log"
      RC_SV=${PIPESTATUS[0]}
      note_stage summarize_h41 "$RC_SV" "$([ "$RC_SV" -eq 124 ] && echo '시간 제한으로 중단' || echo '')"
    else
      note_stage summarize_h41 NA "h41 조각 없음"
    fi
  else
    note_stage summarize NA "LGX_SUMMARIZE=0. 집계는 통합 집계 작업에서 한다"
  fi
  timeout 300 python3 -u scripts/rescale/lgx_timing_table.py --lgx-dir "$LGX_DIR" --lg-dir "$LGX_LG_DIR" --out-dir . --prefix "lgx_${LABEL}" \
    --cpu-workers "$CPU_WORKERS" --h41-workers "$H41_WORKERS" 2>&1 | tee "$LOGD/lgx_timing.log"
  note_stage timing "${PIPESTATUS[0]}" ""
  progress

  local RC=0
  [ "$RC_G" -eq 0 ] && [ "$RC_C" -eq 0 ] && [ "$RC_V" -eq 0 ] && [ "$RC_S" -eq 0 ] && [ "$RC_SV" -eq 0 ] || RC=1
  [ "$RC" -eq 0 ] && [ "$WARN" -ne 0 ] && RC=2
  finish "$RC"
}

main "$@"
