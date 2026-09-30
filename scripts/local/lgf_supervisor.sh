#!/usr/bin/env bash
# LGF 감시기(2026-09-30). logs/lgf/wait_and_resume.sh 를 대신한다.
# 근거: LGF 계획서 6.1('GPU 가 늘 때', 'GPU 가 줄 때'), 6.2(점유 판정 50 MiB·계산 프로세스 0, 드레인 종료 코드 3, 창 마감 4),
#       6.3 단계 6(두 장 이상이면 F 한 장, N 나머지. 한쪽이 끝나면 그 GPU 를 다른 쪽에 더한다), 8.4(단일 큐 J 순서).
#       docs/EXECUTION_PLAN_REMAINING_2026-09-30.md 4.3.
# 동작
#   - 빈 후보(LGF_CANDS, 기본 9 7 6 5 4 3 2 = 하네스 ALLOWED_GPUS)를 POLL 초마다 확인한다. 새로 빈 GPU 는 STABLE 회 연속 비어 있어야 쓴다.
#   - GPU 6·7·9 는 LGT 잠금 PID 가 살아 있으면 쓰지 않는다. LGT 가 끝났을 때 상태 표가 완결(n_partial 0, n_fail 0, n_done = n_todo,
#     중단·abort 없음. n_todo 는 --resume 으로 건너뛴 단위를 뺀 실행 단위 수다)이면 바로 쓴다. 완결이 아니면 두 번째 통과
#     (h43 --resume --rerun-partial, GPU 9·7·6 가운데 빈 것)를 감시기가 한 번 띄우고(LGT_AUTO_SECOND=1, 기본), 그 통과가 끝나면
#     결과와 관계없이 놓는다(남은 partial 은 LG 6C 규칙으로 처리한다). 자동 통과를 끄면(LGT_AUTO_SECOND=0) LGT_GRACE 초(기본 1800)
#     동안 비워 두고 수동 통과를 기다린다. logs/lgf/lgt_released 가 있으면 언제든 놓는다.
#   - 역할 배정: 빈 GPU 가 1장이면 single(run_queue.sh), 2장 이상이면 F(마지막 1장)와 N(나머지). F 가 끝났으면 N 이 전부, N 이 끝났으면 F 가 전부.
#   - 도는 역할이 있고 안 쓰는 빈 후보가 생기면 모든 역할을 드레인하고(진행 중 단위는 끝낸다) 다시 배정한다(--resume).
#   - F·N 이 모두 종료 코드 0 으로 끝나면 역할 T3(h47 --jobs f_t3 --n-jobs-done)를 한 번 띄운다. 하네스가 LGF 8.4 의 J12 조건을
#     판단하고 창 파일에 실행 또는 미실행 결정을 적는다. T3 이 끝나면 멈춘다.
#   - 창 마감(종료 코드 4)이면 멈춘다. h47 은 창 마감 확인이 f_t3 판단보다 앞서므로 이때는 J12 결정이 창 파일에 남지 않는다.
#     감시기 기록('창 마감으로 J12 미실행')을 LGF 10절에 옮겨 적는다.
#   - 사용자 드레인(감시기가 만들지 않은 drain 파일), 짧은 실패 3회 연속이면 멈춘다.
#   - GPU 부족 감시(LGF 개정 6 의 GPU 8 전환 규칙 조건 (2)): LGF 가 쓰는 GPU 가 2장 이하인 상태가 SWITCH_H 시간(기본 6) 넘게
#     이어지면 logs/lgf/switch_condition 을 만들고 기록한다. 전환 조치(LGX 로컬 드레인, 하네스 ALLOWED_GPUS 에 8 추가)는
#     메인 세션이 한다. 감시기는 하네스 코드를 바꾸지 않는다.
#   - GPU 풀의 변화는 h47 --gpu-event 로 lgf_window.json 에 적는다(열람 상태는 logs/lgf/viewing_state.txt 의 한 줄).
# 시작: kill <wait_and_resume.sh PID>; setsid nohup scripts/local/lgf_supervisor.sh > /dev/null 2>&1 < /dev/null &
set -u
cd /home/willy010313/Polar_Bigdata || exit 2
LOGD=logs/lgf; SD=$LOGD/sup; mkdir -p "$SD"
SLOG=$LOGD/supervisor.log; QLOG=$LOGD/queue.log
PY=.venv_lgf/bin/python; H47=scripts/3_deep_learning/h47_foundation_models.py
CANDS=${LGF_CANDS:-"9 7 6 5 4 3 2"}
STABLE=${LGF_STABLE:-3}
POLL=${LGF_POLL:-120}
MIN_ADD=${LGF_MIN_ADD:-1}
MAXG=${LGF_MAX_GPUS:-6}                         # CPU 예산: F (1+1)×2 + N (MAXG−1+1)×2 ≤ 16
LOAD_START_MAX=64
DRAINS="data/processed/lgf/run_lgf/drain data/processed/lgf/run_lgfs/drain data/processed/lgf/run_lgfn/drain"
EXP=$SD/expanding
ROLES="single F N T3"
SWITCH_H=${LGF_SWITCH_H:-6}
LGT_GPUS="9,7,6"

log() { echo "[$(date '+%F %T')] $*" >> "$SLOG"; }
in_list() { [[ ",$2," == *",$1,"* ]]; }
load1() { cut -d' ' -f1 /proc/loadavg; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }
uuid_of() { nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' '; }
mem_of() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
napps_of() { local u; u=$(uuid_of "$1"); [ -n "$u" ] || { echo 99; return; }; nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -c "^$u$"; }
raw_free() { local m n; m=$(mem_of "$1"); n=$(napps_of "$1"); [ -n "$m" ] && [ "$m" -le 50 ] && [ "$n" -eq 0 ]; }
LGT_HOLD=1
lgt_update() {                                   # 주기마다 한 번: LGT_HOLD(1 이면 GPU 6·7·9 를 LGT 몫으로 둔다)를 정하고 필요하면 두 번째 통과를 띄운다
  local p st age g fr=""
  if [ -f "$LOGD/lgt_released" ]; then LGT_HOLD=0; return; fi
  p=$(grep -o '"pid": *[0-9]*' data/processed/lgt/run_lgt/lock.json 2>/dev/null | grep -o '[0-9]*$')
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then LGT_HOLD=1; return; fi   # LGT 학습 실행 중(본 실행 또는 두 번째 통과)
  st=data/processed/lgt/lgt_run_status.json
  if [ ! -f "$st" ]; then LGT_HOLD=0; return; fi  # 실행 중이 아니고 상태 표도 없다(비정상 종료): 놓는다
  if python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); ok=int(d.get("n_partial",1))==0 and int(d.get("n_fail",1))==0 and int(d.get("n_done",-1))==int(d.get("n_todo",-2)) and not d.get("interrupted") and not d.get("abort"); sys.exit(0 if ok else 1)' "$st" 2>/dev/null; then
    [ "$LGT_HOLD" = 1 ] && log "LGT 완결(상태 표 $st). GPU 6·7·9 를 놓는다"
    LGT_HOLD=0; return                           # 완결: 놓는다
  fi
  if [ -f "$LOGD/lgt_second_pass.started" ]; then  # 두 번째 통과가 이미 돌았고 지금은 돌지 않는다: 결과와 관계없이 놓는다
    [ "$LGT_HOLD" = 1 ] && log "LGT 두 번째 통과 종료(완결 아님, LG 6C 규칙으로 처리). GPU 6·7·9 를 놓는다"
    LGT_HOLD=0; return
  fi
  if [ "${LGT_AUTO_SECOND:-1}" = 1 ]; then
    for g in ${LGT_GPUS//,/ }; do raw_free "$g" && fr="${fr:+$fr,}$g"; done
    if [ -n "$fr" ]; then
      date '+%F %T' > "$LOGD/lgt_second_pass.started"
      setsid nohup nice -n 10 python3 -u scripts/3_deep_learning/h43_tabpfn_label_grid.py --gpus "$fr" --threads 3 --gpu-mem-max-mib 50 \
        --resume --rerun-partial --no-summarize >> "$LOGD/lgt_second_pass.log" 2>&1 < /dev/null &
      log "LGT 상태 표가 완결이 아니다. 두 번째 통과를 띄웠다(GPU $fr, pid $!, 로그 $LOGD/lgt_second_pass.log)"
    fi
    LGT_HOLD=1; return
  fi
  age=$(( $(date +%s) - $(stat -c %Y "$st") ))   # 자동 통과를 끈 경우: 수동 두 번째 통과를 LGT_GRACE 초까지 기다린다
  if [ "$age" -lt "${LGT_GRACE:-1800}" ]; then LGT_HOLD=1; else LGT_HOLD=0; fi
}
is_free() {
  local g=$1
  case "$g" in 6|7|9) [ "$LGT_HOLD" = 1 ] && return 1;; esac
  raw_free "$g"
}
alive() { local f=$SD/$1.pid; [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; }
any_drain() { local f; for f in $DRAINS; do [ -f "$f" ] && return 0; done; return 1; }
vs() { head -1 "$LOGD/viewing_state.txt" 2>/dev/null || echo "감시기 자동 기록(logs/lgf/viewing_state.txt 없음)"; }
gpu_event() {                                    # gpu_event <GPU> <add|remove>
  env CUDA_VISIBLE_DEVICES= "$PY" "$H47" --gpu-event "$1:$2" --viewing-state "$(vs)" >> "$SLOG" 2>&1 || log "경고: GPU $1 $2 기록 실패"
}
rc_of() {                                        # 역할의 마지막 종료 코드. 넘겨받은 single 은 queue.log 로 정한다
  local r=$1
  if [ -f "$SD/$r.rc" ]; then cat "$SD/$r.rc"; return; fi
  if [ "$r" = single ] && [ -f "$SD/single.adopted" ]; then
    if grep -aq "큐 완료" "$QLOG"; then echo 0; return; fi
    grep -a "끝(rc=" "$QLOG" | tail -1 | grep -o 'rc=[0-9]*' | grep -o '[0-9]*' || echo 1
    return
  fi
  echo 1
}
start_role() {                                   # start_role <역할> <GPU 목록>
  rm -f "$SD/$1.rc" "$SD/$1.adopted"
  echo "$2" > "$SD/$1.gpus"
  setsid nohup bash scripts/local/lgf_role_queue.sh "$1" "$2" > /dev/null 2>&1 < /dev/null &
  log "시작: 역할 $1 · GPU $2"
}

if [ -f "$SD/supervisor.pid" ] && kill -0 "$(cat "$SD/supervisor.pid")" 2>/dev/null; then echo "감시기가 이미 돈다"; exit 2; fi
echo $$ > "$SD/supervisor.pid"
log "시작(후보 $CANDS · 연속 확인 $STABLE 회 · 간격 ${POLL}s · 최대 GPU $MAXG)"

# 기존 단일 큐(wait_and_resume.sh 가 띄운 run_queue.sh)를 역할 single 로 넘겨받는다
if [ -f "$LOGD/queue.pid" ] && kill -0 "$(cat "$LOGD/queue.pid")" 2>/dev/null && ! alive single && ! alive F && ! alive N; then
  g=$(grep -a "시작:" "$QLOG" | tail -1 | grep -o -- '--gpus [0-9,]*' | awk '{print $2}')
  cp "$LOGD/queue.pid" "$SD/single.pid"; echo "${g:-5}" > "$SD/single.gpus"; date +%s > "$SD/single.adopted"; rm -f "$SD/single.rc"
  echo single > "$SD/started"
  log "기존 큐(pid $(cat "$LOGD/queue.pid"), GPU ${g:-5})를 역할 single 로 넘겨받았다"
fi
PREV=$(for r in $ROLES; do alive "$r" && cat "$SD/$r.gpus"; done | paste -sd, -)
declare -A SEEN
FAILS=0; USERDRAIN=0; LOW_SINCE=0
while true; do
  lgt_update
  NUSE=0; for r in $ROLES; do alive "$r" && NUSE=$(( NUSE + $(tr ',' '\n' < "$SD/$r.gpus" | grep -c .) )); done
  if [ "$NUSE" -le 2 ]; then                     # GPU 부족 감시(전환 규칙 조건 (2))
    [ "$LOW_SINCE" = 0 ] && LOW_SINCE=$(date +%s)
    if [ $(( $(date +%s) - LOW_SINCE )) -ge $(( SWITCH_H * 3600 )) ] && [ ! -f "$LOGD/switch_condition" ]; then
      echo "$(date '+%F %T') LGF 사용 GPU ${NUSE}장 이하가 ${SWITCH_H} h 넘게 이어졌다" > "$LOGD/switch_condition"
      log "전환 규칙 조건 (2) 충족(LGF 사용 GPU ${NUSE}장, ${SWITCH_H} h 이상). logs/lgf/switch_condition 을 만들었다. 조치는 메인 세션"
    fi
  else
    LOW_SINCE=0
  fi
  for g in $CANDS; do if is_free "$g"; then SEEN[$g]=$(( ${SEEN[$g]:-0} + 1 )); else SEEN[$g]=0; fi; done
  ALIVE=""; USED=""
  for r in $ROLES; do if alive "$r"; then ALIVE="$ALIVE $r"; USED="$USED,$(cat "$SD/$r.gpus")"; fi; done
  USED=${USED#,}
  if [ ! -f "$EXP" ] && any_drain && [ "$USERDRAIN" = 0 ]; then USERDRAIN=1; log "감시기가 만들지 않은 drain 파일을 보았다(사용자 드레인). 역할이 끝나면 멈춘다"; fi
  if [ -n "$ALIVE" ]; then
    if [ ! -f "$EXP" ] && [ "$USERDRAIN" = 0 ]; then
      NEW=""; NU=$(tr ',' '\n' <<< "$USED" | grep -c .)
      for g in $CANDS; do
        [ "${SEEN[$g]:-0}" -ge "$STABLE" ] || continue
        in_list "$g" "$USED" && continue
        NEW="$NEW $g"
      done
      if [ "$(wc -w <<< "$NEW")" -ge "$MIN_ADD" ] && [ "$NU" -lt "$MAXG" ]; then
        log "안 쓰는 빈 후보 GPU$NEW(사용 중 $USED, 역할$ALIVE). 드레인 뒤 다시 배정한다"
        touch "$EXP"
        for f in $DRAINS; do mkdir -p "$(dirname "$f")"; touch "$f"; done
      fi
    fi
    sleep "$POLL"; continue
  fi
  # 도는 역할이 없다: 끝난 역할의 종료 코드를 처리한다
  CLOSED=0; SHORTFAIL=0
  for r in $(cat "$SD/started" 2>/dev/null); do
    rc=$(rc_of "$r"); st=$(cat "$SD/$r.start" 2>/dev/null || echo 0)
    log "역할 $r 종료 코드 $rc"
    case "$rc" in
      0) case "$r" in single) touch "$SD/F.done" "$SD/N.done";; F) touch "$SD/F.done";; N) touch "$SD/N.done";; T3) touch "$SD/T3.done";; esac ;;
      3) ;;
      4) CLOSED=1 ;;
      *) if [ "$r" = T3 ]; then touch "$SD/T3.done"; log "역할 T3 종료 코드 $rc(J12 판단 또는 실행의 실패). 다시 띄우지 않는다";
         else [ $(( $(date +%s) - st )) -lt 300 ] && SHORTFAIL=1; fi ;;
    esac
  done
  rm -f "$SD/started"
  if [ "$CLOSED" = 1 ]; then
    [ -f "$SD/T3.done" ] || log "창 마감으로 J12 미실행(h47 은 창 마감 확인이 f_t3 판단보다 앞서 창 파일에 결정을 남기지 않는다. LGF 10절에 이 줄을 옮긴다)"
    log "창 마감(종료 코드 4). 멈춘다"; rm -f "$EXP"; exit 0
  fi
  if [ "$USERDRAIN" = 1 ]; then log "사용자 드레인으로 멈춘다(drain 파일은 그대로 둔다)"; rm -f "$EXP"; exit 0; fi
  if [ -f "$SD/F.done" ] && [ -f "$SD/N.done" ] && [ -f "$SD/T3.done" ]; then log "F·N 완료와 J12 판단(T3) 끝. 멈춘다"; rm -f "$EXP"; exit 0; fi
  for f in $DRAINS; do rm -f "$f"; done
  rm -f "$EXP"
  if [ "$SHORTFAIL" = 1 ]; then FAILS=$((FAILS + 1)); else FAILS=0; fi
  if [ "$FAILS" -ge 3 ]; then log "5분 안에 끝난 실패가 3회 연속이다. 원인 확인 전까지 멈춘다(queue_steps_*.log)"; exit 1; fi
  if gt "$(load1)" "$LOAD_START_MAX"; then log "load average $(load1) > $LOAD_START_MAX: 시작을 미룬다"; sleep "$POLL"; continue; fi
  L=""
  for g in $CANDS; do
    is_free "$g" || continue
    if in_list "$g" "$PREV" || [ "${SEEN[$g]:-0}" -ge "$STABLE" ]; then L="${L:+$L,}$g"; fi
  done
  L=$(tr ',' '\n' <<< "$L" | grep . | head -n "$MAXG" | paste -sd, -)
  if [ -z "$L" ]; then sleep "$POLL"; continue; fi
  n=$(tr ',' '\n' <<< "$L" | grep -c .)
  if [ -f "$SD/F.done" ] && [ -f "$SD/N.done" ]; then
    start_role T3 "$L"; echo T3 > "$SD/started"   # J12 조건 판단(허용이면 J12 실행, 아니면 '미실행' 결정 기록)
  elif [ ! -f "$SD/F.done" ] && [ ! -f "$SD/N.done" ]; then
    if [ "$n" -ge 2 ]; then
      FG=$(tr ',' '\n' <<< "$L" | tail -1); NG=$(tr ',' '\n' <<< "$L" | head -n $((n - 1)) | paste -sd, -)
      start_role F "$FG"; start_role N "$NG"; echo "F N" > "$SD/started"
    else
      start_role single "$L"; echo single > "$SD/started"
    fi
  elif [ -f "$SD/F.done" ]; then
    start_role N "$L"; echo N > "$SD/started"
  else
    start_role F "$L"; echo F > "$SD/started"
  fi
  for g in ${L//,/ }; do in_list "$g" "$PREV" || gpu_event "$g" add; done
  for g in ${PREV//,/ }; do in_list "$g" "$L" || gpu_event "$g" remove; done
  PREV=$L
  sleep 120
done
