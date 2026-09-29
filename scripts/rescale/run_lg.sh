#!/bin/bash
# LG(H40) Rescale 본 실행. 설정 configs/rescale/lg_full.yaml (iolite-4: 64코어, GPU 4장, 벽시계 14 h).
# CPU 부분과 GPU 부분을 한 작업 안에서 함께 실행한다. 로컬 공유 서버에서는 실행하지 않는다(사용자 지시 2026-09-29).
#
# 순서
#   1 묶음 해제, 환경 출력, 필수 패키지 설치(실패하면 종료 코드 1), GPU 확인(보이지 않으면 종료 코드 1)
#   2 이전 작업의 결과 묶음(results_lg_prev.tar.gz 또는 results_lg.tar.gz)이 입력으로 있으면 풀어 조각을 복원한다(--resume 이 건너뛴다)
#   3 GPU 부분을 백그라운드로 실행: --gpus 0,1,2,3 --procs-per-gpu 3 --resume --no-summarize
#   4 CPU 부분을 실행: --workers (코어 수 − 8) / 4 --threads 4 --resume --no-summarize
#     실행 순서는 h40 의 unit_priority 가 정한다(주 4지역 → Alaska → 알래스카 하위 i → 알래스카 하위 x → 나머지, 빠른 학습기 먼저)
#   5 5분마다 진행 줄, 30분마다 results_lg.tar.gz 갱신(임시 파일에 쓴 뒤 교체. 벽시계 상한에 걸려도 마지막 묶음이 남는다)
#   6 두 부분이 끝나거나 시간 한도에 닿으면 --summarize-only, 마지막 묶음, 정리
#
# 환경 변수(설정 파일의 command 에서 준다)
#   LG_GPU_LEARNERS   GPU 부분 학습기. 기본 mlp,cfm,ddpm,tabm,nflow,ftt (RealMLP 제외. 범위는 사전 점검 실측 뒤 정한다: 계획서 §6)
#                     all 을 주면 h40 기본값(7종, RealMLP 가 맨 뒤)이다
#   LG_R2_EXCLUDE     학습기 축에서 R2 를 뺄 학습기(h40 --learner-r2-exclude). 기본은 없음(사전 등록 범위)
#   LG_BUDGET_MIN     스크립트 시작부터 쓸 시간(분). 기본 795 (13.25 h). 벽시계 상한 14 h 보다 45분 짧게 둔다
#   LG_SUM_RESERVE_MIN  집계·묶음에 남길 시간(분). 기본 40. 실행은 (시간 한도 − 이 값)에 멈춘다(기본 12.58 h)
#   LG_GPUS           GPU 목록. 기본 0,1,2,3        LG_GPU_PROCS    GPU 하나당 프로세스 수. 기본 3
#   LG_GPU_THREADS    GPU 워커의 스레드 수. 기본 2   LG_CPU_THREADS  CPU 워커의 스레드 수. 기본 4
#   LG_CPU_WORKERS    CPU 워커 수. 기본 (코어 수 − 8) / 4
#   LG_PARTS          실행할 부분. 기본 "cpu gpu"
#   LG_KEEP_TREE      1 이면 끝난 뒤 조각 디렉터리와 푼 입력을 지우지 않는다. 기본 0

main() {
  set -u -o pipefail
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
  if [ -f lg_payload.tar.gz ] && [ ! -f data/processed/fidelity_base_v3.csv ]; then
    tar xzf lg_payload.tar.gz || { echo "[lg] 묶음 해제 실패"; exit 1; }
  fi
  # shellcheck source=scripts/rescale/lg_common.sh
  source scripts/rescale/lg_common.sh || exit 1
  lg_base_env

  local T0 BUDGET_S NCORE GPUS GPU_PROCS GPU_THREADS CPU_THREADS CPU_WORKERS GPU_LEARNERS R2_EXCLUDE PARTS
  local H40 LGD LOGD STATUS SUM_RESERVE PACK_EVERY PROG_EVERY
  T0=$(date +%s)
  BUDGET_S=$(( ${LG_BUDGET_MIN:-795} * 60 ))
  SUM_RESERVE=$(( ${LG_SUM_RESERVE_MIN:-40} * 60 ))                       # 집계·묶음에 남길 시간(s)
  PACK_EVERY=$(( ${LG_PACK_MIN:-30} * 60 ))
  PROG_EVERY=$(( ${LG_PROGRESS_MIN:-5} * 60 ))
  NCORE=$(lg_ncores)
  GPUS="${LG_GPUS:-0,1,2,3}"
  GPU_PROCS="${LG_GPU_PROCS:-3}"
  GPU_THREADS="${LG_GPU_THREADS:-2}"
  CPU_THREADS="${LG_CPU_THREADS:-4}"
  CPU_WORKERS="${LG_CPU_WORKERS:-$(( (NCORE - 8) / 4 ))}"
  [ "$CPU_WORKERS" -ge 1 ] || CPU_WORKERS=1
  GPU_LEARNERS="${LG_GPU_LEARNERS:-mlp,cfm,ddpm,tabm,nflow,ftt}"
  [ "$GPU_LEARNERS" = "all" ] && GPU_LEARNERS="mlp,cfm,ddpm,tabm,nflow,ftt,realmlp"
  R2_EXCLUDE="${LG_R2_EXCLUDE:-}"
  PARTS="${LG_PARTS:-cpu gpu}"
  H40=scripts/3_deep_learning/h40_label_grid.py
  LGD=data/processed/lg
  LOGD=logs_lg
  STATUS=lg_status.csv
  mkdir -p "$LOGD"

  # 이전 작업의 결과 묶음 복원(있을 때만). 묶음의 경로는 저장소 기준(data/processed/lg/...)이다.
  local prev
  for prev in results_lg_prev.tar.gz results_lg.tar.gz; do
    if [ -f "$prev" ]; then
      if tar xzf "$prev" data/processed/lg 2>/dev/null; then
        lg_log "[lg] 이전 결과 복원: $prev · 조각 $(find "$LGD/shards" -name '*_unit.json' 2>/dev/null | wc -l)개"
      else
        lg_log "[lg] 경고: $prev 를 풀지 못했다. 처음부터 실행한다"
      fi
      break
    fi
  done
  mkdir -p "$LGD"
  echo "stage,rc,sec,note" > "$STATUS"
  [ -f lg_payload_info.txt ] && sed 's/^/[payload] /' lg_payload_info.txt

  left() { echo $(( BUDGET_S - ( $(date +%s) - T0 ) )); }
  note_stage() { echo "$1,$2,$(( $(date +%s) - T0 )),$3" >> "$STATUS"; }
  pack() { lg_pack results_lg.tar.gz "$LGD" "$LOGD" "$STATUS" lg_timing_by_learner.csv lg_payload_info.txt; }
  n_units() { find "$LGD/shards" -maxdepth 1 -name "lg__$1__*_unit.json" 2>/dev/null | wc -l; }
  progress() {
    local lc lg_ fc fg gu
    lc=$(grep -ao "완료 [0-9]*/[0-9]*" "$LOGD/lg_cpu.log" 2>/dev/null | tail -1)
    lg_=$(grep -ao "완료 [0-9]*/[0-9]*" "$LOGD/lg_gpu.log" 2>/dev/null | tail -1)
    fc=$(grep -ac "\[FAIL\]" "$LOGD/lg_cpu.log" 2>/dev/null)
    fg=$(grep -ac "\[FAIL\]" "$LOGD/lg_gpu.log" 2>/dev/null)
    gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd/ -)
    lg_log "[progress] 경과 $(( ( $(date +%s) - T0 ) / 60 )) min · cpu 조각 $(n_units cpu)개(${lc:-대기}) 실패 ${fc:-0}" \
           "· gpu 조각 $(n_units gpu)개(${lg_:-대기}) 실패 ${fg:-0} · GPU 사용률 ${gu:-NA} % · 부하 $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)"
  }
  finish() {                                                              # finish <종료 코드>
    pack
    lg_log "[lg] 단계별 상태"
    sed 's/^/  /' "$STATUS"
    if [ "${LG_KEEP_TREE:-0}" != "1" ]; then                             # 푼 코드·입력은 출력이 아니다(출력 동기화 절약)
      rm -rf src scripts tests .pydeps .lg_pip_constraints.txt lg_payload.tar.gz results_lg_prev.tar.gz \
             data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/lg_subregion_map_v1.csv
      if [ -f results_lg.tar.gz ] && gzip -t results_lg.tar.gz 2>/dev/null \
         && [ "$(tar tzf results_lg.tar.gz | grep -c '_unit\.json$')" -eq "$(find "$LGD/shards" -name '*_unit.json' 2>/dev/null | wc -l)" ]; then
        rm -rf "$LGD/shards"                                              # 조각은 묶음에 있다. 집계 파일(lg_*.csv)은 남긴다
      else
        lg_log "[lg] 묶음 확인 실패. 조각 디렉터리를 남긴다"
      fi
    fi
    lg_log "[lg] 끝 · 종료 코드 $1 · 경과 $(( ( $(date +%s) - T0 ) / 60 )) min"
    exit "$1"
  }

  # ---------------------------------------------------------------- 1 환경·의존성·GPU
  lg_env_report 2>&1 | tee "$LOGD/env.log"
  lg_check_inputs || { note_stage inputs 1 "입력 파일 없음"; finish 1; }
  if lg_install_deps 2>&1 | tee "$LOGD/deps.log"; [ "${PIPESTATUS[0]}" -ne 0 ]; then
    note_stage deps 1 "필수 패키지 설치 실패"
    finish 1
  fi
  [ -d .pydeps ] && export PYTHONPATH="$PWD/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
  note_stage deps 0 ok
  local WARN=0
  case ",$GPU_LEARNERS," in *,realmlp,*)
    lg_install_opt pytabkit "pytabkit==1.7.3" "pytabkit" 2>&1 | tee -a "$LOGD/deps.log"
    [ -d .pydeps ] && export PYTHONPATH="$PWD/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
    if ! python3 -c "import pytabkit" >/dev/null 2>&1; then
      GPU_LEARNERS=$(echo "$GPU_LEARNERS" | tr ',' '\n' | grep -vx realmlp | paste -sd, -)
      note_stage deps_pytabkit 1 "pytabkit 없음. RealMLP 를 뺀다"
      WARN=1
    fi ;;
  esac
  lg_env_report 2>&1 | grep -E "^\[env\] (numpy|pandas|scipy|scikit-learn|catboost|pytabkit|torch) " | tee "$LOGD/env_after_deps.log"

  local SEEN
  SEEN=$(lg_gpu_list)
  case " $PARTS " in *" gpu "*)
    if [ -z "$SEEN" ]; then
      lg_log "[lg] GPU 가 보이지 않는다. 비용이 나가기 전에 멈춘다"
      note_stage gpu_check 1 "GPU 가 보이지 않는다"
      finish 1
    fi
    local g keep=""
    for g in ${GPUS//,/ }; do
      case ",$SEEN," in *",$g,"*) keep="${keep:+$keep,}$g" ;; *) lg_log "[lg] 경고: GPU $g 가 보이지 않는다. 목록에서 뺀다"; WARN=1 ;; esac
    done
    GPUS="$keep"
    [ -n "$GPUS" ] || { note_stage gpu_check 1 "요청한 GPU 가 하나도 보이지 않는다"; finish 1; } ;;
  esac
  lg_log "[lg] 코어 ${NCORE} · 부분 '${PARTS}' · GPU '${GPUS}' × 프로세스 ${GPU_PROCS} × 스레드 ${GPU_THREADS} · CPU 워커 ${CPU_WORKERS} × 스레드 ${CPU_THREADS}"
  lg_log "[lg] GPU 학습기 ${GPU_LEARNERS} · R2 제외 '${R2_EXCLUDE}' · 시간 한도 $(( BUDGET_S / 60 )) min(집계 예비 $(( SUM_RESERVE / 60 )) min)"

  # ---------------------------------------------------------------- 3–4 실행
  local RUN_S PID_G="" PID_C="" RC_G=0 RC_C=0 R2ARG=()
  [ -n "$R2_EXCLUDE" ] && R2ARG=(--learner-r2-exclude "$R2_EXCLUDE")
  RUN_S=$(( $(left) - SUM_RESERVE ))
  [ "$RUN_S" -gt 30 ] || { note_stage budget 1 "시간 한도가 집계 예비 시간보다 짧다"; finish 1; }
  # timeout 은 자기 프로세스 그룹 전체에 신호를 보낸다. h40 의 워커 프로세스도 함께 끝난다.
  case " $PARTS " in *" gpu "*)
    timeout -k 60 "$RUN_S" env LG_RESCALE=1 python3 -u "$H40" --part gpu --gpus "$GPUS" --procs-per-gpu "$GPU_PROCS" --threads "$GPU_THREADS" \
      --learners "$GPU_LEARNERS" ${R2ARG[@]+"${R2ARG[@]}"} --resume --no-summarize > "$LOGD/lg_gpu.log" 2>&1 &
    PID_G=$!
    lg_log "[lg] GPU 부분 시작(pid $PID_G) → $LOGD/lg_gpu.log" ;;
  esac
  case " $PARTS " in *" cpu "*)
    timeout -k 60 "$RUN_S" env LG_RESCALE=1 python3 -u "$H40" --part cpu --workers "$CPU_WORKERS" --threads "$CPU_THREADS" \
      ${R2ARG[@]+"${R2ARG[@]}"} --resume --no-summarize > "$LOGD/lg_cpu.log" 2>&1 &
    PID_C=$!
    lg_log "[lg] CPU 부분 시작(pid $PID_C) → $LOGD/lg_cpu.log" ;;
  esac

  # ---------------------------------------------------------------- 5 감시
  local now last_prog last_pack done_g=0 done_c=0
  last_prog=$(date +%s); last_pack=$last_prog
  [ -n "$PID_G" ] || done_g=1
  [ -n "$PID_C" ] || done_c=1
  while [ "$done_g" -eq 0 ] || [ "$done_c" -eq 0 ]; do
    sleep 20
    if [ "$done_g" -eq 0 ] && ! kill -0 "$PID_G" 2>/dev/null; then
      wait "$PID_G"; RC_G=$?; done_g=1
      note_stage gpu "$RC_G" "$([ "$RC_G" -eq 124 ] && echo '시간 한도로 중단' || echo '')"
      lg_log "[lg] GPU 부분 끝: 종료 코드 $RC_G"; tail -n 5 "$LOGD/lg_gpu.log" | sed 's/^/  [gpu] /'
      progress; pack
    fi
    if [ "$done_c" -eq 0 ] && ! kill -0 "$PID_C" 2>/dev/null; then
      wait "$PID_C"; RC_C=$?; done_c=1
      note_stage cpu "$RC_C" "$([ "$RC_C" -eq 124 ] && echo '시간 한도로 중단' || echo '')"
      lg_log "[lg] CPU 부분 끝: 종료 코드 $RC_C"; tail -n 5 "$LOGD/lg_cpu.log" | sed 's/^/  [cpu] /'
      progress; pack
    fi
    now=$(date +%s)
    if [ $(( now - last_prog )) -ge "$PROG_EVERY" ]; then progress; last_prog=$now; fi
    if [ $(( now - last_pack )) -ge "$PACK_EVERY" ]; then pack; last_pack=$(date +%s); fi
  done

  # ---------------------------------------------------------------- 6 집계·묶음
  local f
  for f in "$LOGD/lg_cpu.log" "$LOGD/lg_gpu.log"; do
    [ -f "$f" ] || continue
    lg_log "[lg] $f 의 실패·경고 줄(앞 40줄)"
    grep -a -E "\[FAIL\]|\[warn\]|\[pool\]|Traceback" "$f" | head -n 40 | sed 's/^/  /'
    grep -a -E "^\[(data|plan|done)\]" "$f" | sed 's/^/  /'
  done
  local lim RC_S=0
  lim=$(( $(left) - 300 )); [ "$lim" -gt 300 ] || lim=300
  lg_log "[lg] 집계 시작(제한 ${lim} s)"
  timeout -k 30 "$lim" python3 -u "$H40" --summarize-only --threads 8 ${R2ARG[@]+"${R2ARG[@]}"} 2>&1 | tee "$LOGD/lg_summarize.log"
  RC_S=${PIPESTATUS[0]}
  note_stage summarize "$RC_S" "$([ "$RC_S" -eq 124 ] && echo '시간 제한으로 중단' || echo '')"
  timeout 300 python3 -u scripts/rescale/lg_timing_table.py --lg-dir "$LGD" --out-dir . --prefix lg 2>&1 | tee "$LOGD/lg_timing.log"
  note_stage timing "${PIPESTATUS[0]}" ""
  progress

  local RC=0
  [ "$RC_G" -eq 0 ] && [ "$RC_C" -eq 0 ] && [ "$RC_S" -eq 0 ] || RC=1
  [ "$RC" -eq 0 ] && [ "$WARN" -ne 0 ] && RC=2
  finish "$RC"
}

main "$@"
