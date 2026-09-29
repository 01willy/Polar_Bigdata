#!/usr/bin/env bash
# LGU(예측 분포 실험 A·B·C) 로컬 GPU 서버 실행 스크립트
#
# 계획서: docs/EXPERIMENT_PLAN_LGU_2026-09-29.md 8.1절(로컬 실행 규칙과 실행 순서), 2.9절(사전 점검 뒤 확정 항목).
# 실행 환경: 사용자 지시(2026-09-30)로 LGU 는 Rescale 이 아니라 로컬 GPU 서버에서 실행한다. 이 스크립트는 Rescale 스크립트
# (scripts/rescale/*)와 설정(configs/rescale/*)을 쓰지 않고 고치지 않는다.
#
# 사용법(어느 디렉터리에서 불러도 저장소 루트로 옮겨 실행한다)
#   scripts/local/run_lgu_local.sh --smoke    [--gpu 2] [--threads 4] [--cpu-fallback]
#   scripts/local/run_lgu_local.sh --precheck [--gpu 2] [--threads 4] [--gpu-wait 360]
#   scripts/local/run_lgu_local.sh --full     [--gpu 2] [--threads 4] [--gpu-wait 360]
#   scripts/local/run_lgu_local.sh --full --summarize-only [--threads 4]
#   선택 인자
#     --gpu LIST        GPU 물리 번호(nvidia-smi 번호) 쉼표 목록. 1, 2 만 받는다. 앞 번호부터 비어 있는 첫 번호를 쓴다. 기본 2
#     --threads N       프로세스당 스레드(OMP·MKL·OPENBLAS·NUMEXPR, torch, CatBoost). 1–8. 기본 4
#     --workers N       CPU 단계의 프로세스 수. 기본 min(2, 8 / 스레드). 워커 × 스레드 ≤ 8 이어야 한다
#     --resume          완료 조각을 건너뛴다(하네스의 --resume). --precheck·--full 의 기본값이다
#     --no-resume       조각을 다시 계산한다. --smoke 의 기본값이다
#     --gpu-wait MIN    GPU 단계 직전에 배정 GPU 가 비기를 기다리는 최대 시간(분). 기본: 스모크 0, 그 밖 360
#     --cpu-fallback    (--smoke 전용) 기다린 뒤에도 GPU 가 비지 않으면 GPU 단계(epochs 3)를 CPU 로 돈다. b_fit 이 CPU 로 돌면
#                       cfm 셀당 표본 수를 64 로 줄인다(--s-gen 을 주면 그 값. 뒤의 b_score 도 같은 값을 써야 조각이 맞는다)
#     --only LIST       실행할 단계 목록(아래 단계 이름). 목록 밖의 선행 단계는 이미 끝났다고 본다
#     --budget-min MIN  전체 시간 상한(분). 넘으면 남은 단계를 실행하지 않는다. 기본: 스모크 30, 그 밖 0(상한 없음)
#     --nboot N, --s-gen N   본 실행의 재표집 횟수와 cfm 셀당 표본 수를 직접 준다(기본: 사전 점검 확정표, 스모크 확정표, 2000·512 순)
#     --max-load X      실행 전 1분 load average 상한. 기본 코어 수의 절반. 넘으면 --load-wait 분까지 기다린 뒤 중단한다
#     --load-wait MIN   기본 60
#     --min-free-gb N   출력 디스크의 최소 여유(GB). 기본 20
#     --min-mem-gb N    최소 가용 메모리(GB). 기본 16(공유 서버 OOM 이력)
#     --dry-run         점검과 명령 출력만 하고 하네스를 실행하지 않는다
#
# 단계(계획서 8.1절의 순서. 앞 단계가 끝나야 다음 단계를 시작한다. 모든 하네스에 --allow-local 을 준다)
#   a_cpu    h44 --part cpu  --workers W --threads T --no-summarize
#   a_gate   h44 --part gate --gpus G --threads T --no-summarize            GPU. a_cpu 뒤
#   a_flow   h44 --part flow --gpus G --threads T --no-summarize            GPU. a_gate 뒤
#   a_sum    h44 --summarize-only --workers W --threads T                   a_cpu·a_gate·a_flow 뒤(사전 점검은 정밀도 표만)
#   b_fit    h45 --part fit --gpus G --workers 1 --threads T                GPU(CPU 정규화기 cbq·phys 는 먼저 CPU 풀 1개)
#   b_score  h45 --part score --workers W --threads T                       b_fit 뒤. 집계 포함
#   c_diag   h46 --workers W --threads T                                    집계 포함
#   --summarize-only: a_sum, b_sum(h45 --summarize-only), c_sum(h46 --summarize-only)
#   --smoke·--precheck 는 모든 하네스에 같은 인자를 붙인다(하네스가 대상·분할·epochs 를 줄인다. 산출 이름에 _smoke·_precheck).
#
# 자원 규칙(사용자 지시 2026-09-30. 하네스도 같은 규칙을 강제한다)
#   GPU   배정 2번(필요하면 1번). GPU 9·7·6·5(TabPFN), 4·3(LGF), 8(다른 사용자)은 쓰지 않는다. GPU 단계 직전에 nvidia-smi 로 메모리
#         사용 0 MiB 이고 계산 프로세스가 없는 번호만 쓴다. 사용 중이면 1분 간격으로 --gpu-wait 분까지 기다리고, 그래도 비지 않으면
#         그 단계와 뒤따르는 단계를 blocked_gpu 로 두고 독립 단계만 계속한다(스모크의 --cpu-fallback 은 예외).
#   CPU   이 실험의 스레드 합계 8 이하. 한 단계 안에서 (프로세스 수 × 스레드) ≤ 8 이고 단계는 하나씩 돈다. 하네스 사이의 합은 실행 중
#         등록부(data/processed/lgu/.running)로 하네스가 확인한다. 이 스크립트는 nice 10 으로 자신을 다시 실행하고 자식이 물려받는다.
#   동시 실행  이 스크립트는 한 번에 하나만 돈다(data/processed/lgu/.running/runner.lock).
#
# 기록
#   logs/lgu/run.log                       단계 시작·끝, 점검, 5분마다 진행 줄(조각 수, GPU 메모리, load average, 가용 메모리)
#   logs/lgu/<실행 번호>_<모드>_<단계>.log   하네스 출력 전체(판정 표가 들어 있을 수 있다. 결과 비열람 규칙에 따라 열지 않는다)
#   data/processed/lgu/lgu_status.csv      단계별 상태, 종료 코드, 시간, 조각 status 개수(점수·판정 없음). 같은 모드·단계의 이전 행을 바꾼다
#   data/processed/lgu/lgu_<precheck|smoke>_decisions.csv   (--precheck, --smoke) 계획서 2.9절 확정 규칙의 적용 결과
#                                          (scripts/local/lgu_local_tables.py decide). --full 은 항목마다 사전 점검표, 스모크표, 기본값 순으로
#                                          --nboot, --s-gen 을 정한다(직접 주면 그 값). 계획서 개정 2 참조
#
# 종료 코드: 0 모든 단계 ok · 1 실패 단계 있음 · 2 인자·점검 거부 · 3 GPU 대기 초과, load 대기 초과, 시간 상한으로 남은 단계 있음

set -u -o pipefail

# ---------------------------------------------------------------- 0. nice 10 으로 다시 실행한다(자식 프로세스가 물려받는다)
if [ -z "${LGU_RUNNER_NICED:-}" ]; then
  _cur_nice=$(nice)
  export LGU_RUNNER_NICED=1
  if [ "$_cur_nice" -lt 10 ]; then
    exec nice -n $((10 - _cur_nice)) bash "${BASH_SOURCE[0]}" "$@"
  fi
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
PY="${PYTHON:-python3}"
H44="scripts/2_evaluation/h44_hier_predictive_ladder.py"
H45="scripts/2_evaluation/h45_normalized_conformal.py"
H46="scripts/2_evaluation/h46_spatial_diagnostics.py"
HELPER="scripts/local/lgu_local_tables.py"
OUT="data/processed/lgu"
SHARDS="$OUT/shards"
REGDIR="$OUT/.running"
LOGD="logs/lgu"
RUNLOG="$LOGD/run.log"
THREAD_CAP=8
GPU_ALLOWED="1 2 5"          # 사용자 지시 2026-09-30: 배정 2번(필요하면 1번). 01:39 부터 1·2 를 다른 사용자가 써서 5번을 더했다(LGU 개정 3)
GPU_MAX_MIB=50               # 빈 GPU 의 메모리 표시가 2 MiB 여서 0 MiB 기준으로는 빈 GPU 도 거부된다. LGT 와 같은 50 MiB 기준(LGU 개정 3). 계산 프로세스 0 조건은 유지
SMOKE_CPU_SGEN=64            # --cpu-fallback 스모크에서 b_fit 이 CPU 로 돌 때의 cfm 셀당 표본 수(--s-gen 을 주면 그 값)

MODE=""; SUMMARIZE_ONLY=0; GPU_LIST="2"; THREADS=4; WORKERS=""; RESUME=""; GPU_WAIT=""; CPU_FALLBACK=0; ONLY=""; BUDGET_MIN=""
DRY=0; NBOOT=""; SGEN=""; MAX_LOAD=""; LOAD_WAIT=60; MIN_FREE_GB=20; MIN_MEM_GB=16

usage() { sed -n '2,58p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "[run_lgu_local] $*" >&2; exit 2; }
is_int() { [[ "${1:-}" =~ ^[0-9]+$ ]]; }

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke|--precheck|--full)
      [ -n "$MODE" ] && die "--smoke, --precheck, --full 가운데 하나만 준다"
      MODE="${1#--}" ;;
    --summarize-only) SUMMARIZE_ONLY=1 ;;
    --gpu) GPU_LIST="${2:-}"; shift ;;
    --gpu=*) GPU_LIST="${1#*=}" ;;
    --threads) THREADS="${2:-}"; shift ;;
    --threads=*) THREADS="${1#*=}" ;;
    --workers) WORKERS="${2:-}"; shift ;;
    --workers=*) WORKERS="${1#*=}" ;;
    --resume) RESUME=1 ;;
    --no-resume) RESUME=0 ;;
    --gpu-wait) GPU_WAIT="${2:-}"; shift ;;
    --cpu-fallback) CPU_FALLBACK=1 ;;
    --only) ONLY="${2:-}"; shift ;;
    --budget-min) BUDGET_MIN="${2:-}"; shift ;;
    --nboot) NBOOT="${2:-}"; shift ;;
    --s-gen) SGEN="${2:-}"; shift ;;
    --max-load) MAX_LOAD="${2:-}"; shift ;;
    --load-wait) LOAD_WAIT="${2:-}"; shift ;;
    --min-free-gb) MIN_FREE_GB="${2:-}"; shift ;;
    --min-mem-gb) MIN_MEM_GB="${2:-}"; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "알 수 없는 인자: $1 (--help 참조)" ;;
  esac
  shift
done

# ---------------------------------------------------------------- 1. 인자 확인
[ -n "$MODE" ] || die "--smoke, --precheck, --full 가운데 하나를 준다"
is_int "$THREADS" && [ "$THREADS" -ge 1 ] && [ "$THREADS" -le "$THREAD_CAP" ] || die "--threads 는 1–$THREAD_CAP 이다: $THREADS"
if [ -z "$WORKERS" ]; then WORKERS=$(( THREAD_CAP / THREADS )); [ "$WORKERS" -gt 2 ] && WORKERS=2; fi
is_int "$WORKERS" && [ "$WORKERS" -ge 1 ] || die "--workers 는 1 이상이다: $WORKERS"
[ $(( WORKERS * THREADS )) -le "$THREAD_CAP" ] || die "워커 $WORKERS × 스레드 $THREADS 가 스레드 합계 상한 $THREAD_CAP 을 넘는다(사용자 지시 2026-09-30)"
GPUS=()
IFS=',' read -r -a _g <<< "$GPU_LIST"
for g in "${_g[@]}"; do
  g="${g// /}"; [ -z "$g" ] && continue
  case " $GPU_ALLOWED " in *" $g "*) GPUS+=("$g") ;; *) die "--gpu $g 는 이 실험에 배정된 번호(2, 필요하면 1)가 아니다(사용자 지시 2026-09-30)" ;; esac
done
[ "${#GPUS[@]}" -ge 1 ] || die "--gpu 가 비어 있다"
[ "$CPU_FALLBACK" = 1 ] && [ "$MODE" != smoke ] && die "--cpu-fallback 은 --smoke 전용이다(사전 점검의 시간 측정과 본 실행은 GPU 에서 한다)"
[ -n "$RESUME" ] || { [ "$MODE" = smoke ] && RESUME=0 || RESUME=1; }
[ -n "$GPU_WAIT" ] || { [ "$MODE" = smoke ] && GPU_WAIT=0 || GPU_WAIT=360; }
[ -n "$BUDGET_MIN" ] || { [ "$MODE" = smoke ] && BUDGET_MIN=30 || BUDGET_MIN=0; }
NCPU=$(nproc)
[ -n "$MAX_LOAD" ] || MAX_LOAD=$(( NCPU / 2 ))
for v in GPU_WAIT BUDGET_MIN LOAD_WAIT MIN_FREE_GB MIN_MEM_GB; do is_int "${!v}" || die "$v 는 음이 아닌 정수다: ${!v}"; done
[ -z "$NBOOT" ] || is_int "$NBOOT" || die "--nboot 는 정수다"
[ -z "$SGEN" ] || is_int "$SGEN" || die "--s-gen 은 정수다"
case "$MODE" in smoke) MFLAG="--smoke" ;; precheck) MFLAG="--precheck" ;; *) MFLAG="" ;; esac

if [ "$SUMMARIZE_ONLY" = 1 ]; then ALL_STEPS=(a_sum b_sum c_sum); else ALL_STEPS=(a_cpu a_gate a_flow a_sum b_fit b_score c_diag); fi
STEPS=()
if [ -n "$ONLY" ]; then
  IFS=',' read -r -a _o <<< "$ONLY"
  for s in "${ALL_STEPS[@]}"; do for o in "${_o[@]}"; do [ "$s" = "${o// /}" ] && STEPS+=("$s"); done; done
  for o in "${_o[@]}"; do case " ${ALL_STEPS[*]} " in *" ${o// /} "*) ;; *) die "--only 의 단계 ${o} 는 이 모드의 단계(${ALL_STEPS[*]})가 아니다" ;; esac; done
else
  STEPS=("${ALL_STEPS[@]}")
fi
[ "${#STEPS[@]}" -ge 1 ] || die "실행할 단계가 없다"

declare -A DEPS=([a_gate]="a_cpu" [a_flow]="a_gate" [a_sum]="a_cpu a_gate a_flow" [b_score]="b_fit")
declare -A NEEDS_GPU=([a_gate]=1 [a_flow]=1 [b_fit]=1)
declare -A HARNESS=([a_cpu]=h44 [a_gate]=h44 [a_flow]=h44 [a_sum]=h44 [b_fit]=h45 [b_score]=h45 [b_sum]=h45 [c_diag]=h46 [c_sum]=h46)
declare -A ST=()

mkdir -p "$LOGD" "$SHARDS" "$REGDIR" || die "디렉터리를 만들 수 없다"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
if [ "$DRY" = 1 ]; then                                  # dry-run 은 기록 파일을 남기지 않는다
  STEPS_TSV="$(mktemp)"; STATE="$(mktemp)"
else
  STEPS_TSV="$LOGD/${RUN_ID}_${MODE}_steps.tsv"; STATE="$LOGD/.${RUN_ID}_state"
fi
T0=$(date +%s)

log() { local line; line="[$(date '+%Y-%m-%d %H:%M:%S')] [$MODE] $*"; echo "$line"; [ "$DRY" = 1 ] || echo "$line" >> "$RUNLOG"; }

# ---------------------------------------------------------------- 2. 동시 실행 방지
if [ "$DRY" = 0 ]; then
  exec 9>"$REGDIR/runner.lock" || die "잠금 파일을 열 수 없다"
  flock -n 9 || die "다른 run_lgu_local.sh 가 실행 중이다($REGDIR/runner.lock)"
fi

# ---------------------------------------------------------------- 3. 점검 함수
gpu_mem() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
gpu_util() { nvidia-smi -i "$1" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
gpu_napps() { nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]'; }
gpu_is_free() {          # 메모리 사용 0 MiB(GPU_MAX_MIB 이하)이고 계산 프로세스가 없으면 참
  local m n; m=$(gpu_mem "$1"); n=$(gpu_napps "$1")
  is_int "$m" || return 1
  [ "$m" -le "$GPU_MAX_MIB" ] && [ "${n:-1}" -eq 0 ]
}
pick_gpu() {             # 비어 있는 첫 배정 GPU 번호를 출력한다. 없으면 빈 문자열
  local g; for g in "${GPUS[@]}"; do if gpu_is_free "$g"; then echo "$g"; return 0; fi; done; echo ""
}
gpu_desc() { local g out=""; for g in "${GPUS[@]}"; do out+="GPU$g $(gpu_mem "$g") MiB·사용률 $(gpu_util "$g") %·프로세스 $(gpu_napps "$g") "; done; echo "$out"; }
load1() { awk '{print $1}' /proc/loadavg; }
mem_avail_gb() { awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo; }
disk_free_gb() { df -Pk "$OUT" | awk 'NR==2 {printf "%d", $4/1048576}'; }
load_over() { awk -v l="$(load1)" -v m="$MAX_LOAD" 'BEGIN {exit !(l > m)}'; }
elapsed() { echo $(( $(date +%s) - T0 )); }
budget_left() { if [ "$BUDGET_MIN" -eq 0 ]; then echo 999999999; else echo $(( BUDGET_MIN * 60 - $(elapsed) )); fi; }

# ---------------------------------------------------------------- 4. 실행 전 점검
log "시작 · 실행 번호 $RUN_ID · 단계 ${STEPS[*]} · GPU 목록 ${GPUS[*]} · 스레드 $THREADS · 워커 $WORKERS · 재개 $RESUME · GPU 대기 ${GPU_WAIT}분 · CPU 대체 $CPU_FALLBACK · 시간 상한 ${BUDGET_MIN}분 · nice $(nice) · dry-run $DRY"
log "점검: 코드 $(git rev-parse --short HEAD 2>/dev/null || echo '?') · python $($PY -c 'import sys; print(sys.version.split()[0])') · 코어 $NCPU · load average $(cut -d' ' -f1-3 /proc/loadavg) (상한 $MAX_LOAD) · 가용 메모리 $(mem_avail_gb) GB(하한 $MIN_MEM_GB) · 디스크 여유 $(disk_free_gb) GB(하한 $MIN_FREE_GB)"
log "점검: 배정 GPU 상태 $(gpu_desc)· 기준 메모리 ${GPU_MAX_MIB} MiB 이하·계산 프로세스 0"
for f in data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/lg_subregion_map_v1.csv data/processed/lgx_label_flags_v1.csv "$H44" "$H45" "$H46" "$HELPER"; do
  [ -f "$f" ] || die "입력 또는 코드 파일이 없다: $f"
done
[ "$(disk_free_gb)" -ge "$MIN_FREE_GB" ] || die "디스크 여유 $(disk_free_gb) GB 가 하한 $MIN_FREE_GB GB 보다 작다"
[ "$(mem_avail_gb)" -ge "$MIN_MEM_GB" ] || die "가용 메모리 $(mem_avail_gb) GB 가 하한 $MIN_MEM_GB GB 보다 작다(공유 서버 OOM 이력)"
REG="$($PY "$HELPER" registry --dir "$REGDIR")"
log "점검: 실행 중 등록부 $REG"
REG_THR="${REG#THREADS=}"; REG_THR="${REG_THR%% *}"
if is_int "$REG_THR" && [ "$REG_THR" -gt 0 ] && [ "$DRY" = 0 ]; then
  die "실행 중인 LGU 하네스가 있다($REG). 끝난 뒤 다시 실행한다(스레드 합계 8 규칙)"
fi
if [ "$MODE" = full ] && { [ -z "$NBOOT" ] || [ -z "$SGEN" ]; }; then
  PAR="$($PY "$HELPER" params --out-dir "$OUT")"
  for kv in $PAR; do case "$kv" in NBOOT=*) [ -n "$NBOOT" ] || NBOOT="${kv#NBOOT=}" ;; SGEN=*) [ -n "$SGEN" ] || SGEN="${kv#SGEN=}" ;; esac; done
  log "본 실행 확정값: $PAR → --nboot $NBOOT --s-gen $SGEN"
fi

export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
unset LG_RESCALE

# ---------------------------------------------------------------- 5. 진행 줄(5분마다)
printf 'step\tharness\tdevice\tthreads\tworkers\tstatus\texit_code\tstarted\tended\twall_s\tlog\tcommand\n' > "$STEPS_TSV"
echo "init $(date +%s) -" > "$STATE"
MAIN_PID=$$
HB_SEC="${LGU_HB_SEC:-300}"                                                     # 진행 줄 간격(초). 시험용으로만 바꾼다
is_int "$HB_SEC" && [ "$HB_SEC" -ge 1 ] || HB_SEC=300
heartbeat() {
  local st t0 dev line
  while sleep "$HB_SEC"; do
    kill -0 "$MAIN_PID" 2>/dev/null || exit 0
    read -r st t0 dev < "$STATE" 2>/dev/null || { st="?"; t0=$(date +%s); dev="-"; }
    line="[hb] 단계 $st(경과 $(( $(date +%s) - t0 )) s, 전체 $(elapsed) s) · 조각 $($PY "$HELPER" heartbeat --shards "$SHARDS" --mode "$MODE" 2>/dev/null)"
    line+=" · $(gpu_desc)· load $(cut -d' ' -f1 /proc/loadavg) · 가용 메모리 $(mem_avail_gb) GB"
    log "$line"
  done
}
HB_PID=""
if [ "$DRY" = 0 ]; then heartbeat & HB_PID=$!; fi
cleanup() { [ -n "$HB_PID" ] && kill "$HB_PID" 2>/dev/null; rm -f "$STATE"; [ "$DRY" = 1 ] && rm -f "$STEPS_TSV"; }
trap cleanup EXIT
trap 'log "신호를 받아 중단한다"; exit 130' INT TERM

write_status() {
  [ "$DRY" = 1 ] && return 0
  $PY "$HELPER" status --tsv "$STEPS_TSV" --out "$OUT/lgu_status.csv" --shards "$SHARDS" --mode "$MODE" --run-id "$RUN_ID" \
    || log "[warn] 상태 표를 쓰지 못했다"
}

# ---------------------------------------------------------------- 6. 단계 명령
CMD=()
build_cmd() {            # 전역 CMD 를 만든다. $1 단계, $2 장치(GPU 번호 또는 cpu)
  local st="$1" dev="$2" common=(--allow-local --threads "$THREADS") res=() nb=() sg=()
  [ -n "$MFLAG" ] && common+=("$MFLAG")
  [ "$RESUME" = 1 ] && res=(--resume)
  [ -n "$NBOOT" ] && nb=(--nboot "$NBOOT")
  [ -n "$SGEN" ] && sg=(--s-gen "$SGEN")
  case "$st" in
    a_cpu)   CMD=("$H44" --part cpu --workers "$WORKERS" "${common[@]}" ${res[@]+"${res[@]}"} ${nb[@]+"${nb[@]}"} --no-summarize) ;;
    a_gate|a_flow)
      if [ "$dev" = cpu ]; then CMD=("$H44" --part "${st#a_}" --workers "$WORKERS" "${common[@]}")
      else CMD=("$H44" --part "${st#a_}" --gpus "$dev" --workers 1 "${common[@]}"); fi
      CMD+=(${res[@]+"${res[@]}"} ${nb[@]+"${nb[@]}"} --no-summarize) ;;
    a_sum)   CMD=("$H44" --summarize-only --workers "$WORKERS" "${common[@]}" ${res[@]+"${res[@]}"} ${nb[@]+"${nb[@]}"}) ;;
    b_fit)
      if [ "$dev" = cpu ]; then CMD=("$H45" --part fit --workers "$WORKERS" "${common[@]}")
      else CMD=("$H45" --part fit --gpus "$dev" --workers 1 "${common[@]}"); fi
      CMD+=(${res[@]+"${res[@]}"} ${nb[@]+"${nb[@]}"} ${sg[@]+"${sg[@]}"}) ;;
    b_score) CMD=("$H45" --part score --workers "$WORKERS" "${common[@]}" ${res[@]+"${res[@]}"} ${nb[@]+"${nb[@]}"} ${sg[@]+"${sg[@]}"}) ;;
    b_sum)   CMD=("$H45" --summarize-only --workers "$WORKERS" "${common[@]}" ${nb[@]+"${nb[@]}"} ${sg[@]+"${sg[@]}"}) ;;
    c_diag)  CMD=("$H46" --workers "$WORKERS" "${common[@]}" ${res[@]+"${res[@]}"}) ;;
    c_sum)   CMD=("$H46" --summarize-only --workers "$WORKERS" "${common[@]}") ;;
    *) die "알 수 없는 단계 $st" ;;
  esac
}

record() {               # $1 단계 $2 장치 $3 상태 $4 종료 코드 $5 시작 $6 끝 $7 로그
  local st="$1" dev="$2" status="$3" rc="$4" s0="$5" s1="$6" lg="$7" wall="" started="" ended="" w="$WORKERS"
  ST[$st]="$status"
  is_int "$s0" && started="$(date -d "@$s0" '+%Y-%m-%d %H:%M:%S')"
  is_int "$s1" && ended="$(date -d "@$s1" '+%Y-%m-%d %H:%M:%S')"
  is_int "$s0" && is_int "$s1" && wall=$(( s1 - s0 ))
  [ "$dev" != cpu ] && [ "$dev" != "-" ] && w=1
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$st" "${HARNESS[$st]}" "$dev" "$THREADS" "$w" "$status" "$rc" \
    "$started" "$ended" "$wall" "$lg" "${CMD[*]:-}" >> "$STEPS_TSV"
  log "단계 $st 끝 · 상태 $status · 종료 코드 $rc · 장치 $dev · ${wall:-0} s · 로그 $lg"
  write_status
}

classify() {             # $1 종료 코드 $2 로그 → 상태
  local rc="$1" lg="$2"
  if [ "$rc" -eq 0 ]; then echo ok
  elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then echo timeout
  elif grep -q -e "배정된 GPU 가 모두 사용 중" -e "요청한 GPU 가 모두 사용 중" "$lg" 2>/dev/null; then echo blocked_gpu
  elif grep -q "^\[거부\]" "$lg" 2>/dev/null; then echo refused
  elif grep -q "^Traceback" "$lg" 2>/dev/null; then echo failed
  elif grep -q -e "^\[done\]" -e "^\[summarize\]" -e "^\[summary\]" -e "^\[h46\]" "$lg" 2>/dev/null; then echo failed_units
  else echo failed; fi
}

wait_resources() {       # load average 와 가용 메모리가 기준 안이 될 때까지 1분 간격으로 기다린다(최대 LOAD_WAIT 분). 넘으면 1
  local waited=0
  while load_over || [ "$(mem_avail_gb)" -lt "$MIN_MEM_GB" ]; do
    if [ "$waited" -ge $(( LOAD_WAIT * 60 )) ]; then return 1; fi
    [ $(( waited % 300 )) -eq 0 ] && log "대기: load $(load1)(상한 $MAX_LOAD) · 가용 메모리 $(mem_avail_gb) GB(하한 $MIN_MEM_GB)"
    sleep 60; waited=$(( waited + 60 ))
  done
  return 0
}

wait_gpu() {             # 비어 있는 배정 GPU 번호를 출력한다(최대 GPU_WAIT 분). 없으면 빈 문자열
  local g waited=0
  while :; do
    g="$(pick_gpu)"
    if [ -n "$g" ]; then echo "$g"; return 0; fi
    if [ "$waited" -ge $(( GPU_WAIT * 60 )) ] || [ "$(budget_left)" -le 120 ]; then echo ""; return 0; fi
    [ $(( waited % 300 )) -eq 0 ] && log "대기: 배정 GPU 사용 중 · $(gpu_desc)" >&2
    echo "wait_gpu $(date +%s) -" > "$STATE"
    sleep 60; waited=$(( waited + 60 ))
  done
}

# ---------------------------------------------------------------- 7. 단계 실행
STOP=""
for st in "${STEPS[@]}"; do
  CMD=()
  lg="$LOGD/${RUN_ID}_${MODE}_${st}.log"
  if [ -n "$STOP" ]; then record "$st" "-" "not_run($STOP)" "" "" "" ""; continue; fi
  bad=""
  for d in ${DEPS[$st]:-}; do
    case " ${STEPS[*]} " in *" $d "*) ;; *) continue ;; esac                        # 목록 밖의 선행 단계는 이미 끝났다고 본다
    case "${ST[$d]:-}" in ok|failed_units) ;; *) bad+="$d(${ST[$d]:-없음}) " ;; esac
  done
  if [ -n "$bad" ]; then record "$st" "-" "skipped_dep" "" "" "" ""; log "단계 $st: 선행 단계 $bad 때문에 실행하지 않는다"; continue; fi
  if [ "$(budget_left)" -le 60 ]; then STOP="budget"; record "$st" "-" "not_run(budget)" "" "" "" ""; continue; fi
  if [ "$DRY" = 0 ] && ! wait_resources; then STOP="load"; record "$st" "-" "not_run(load)" "" "" "" ""; continue; fi
  dev=cpu
  if [ -n "${NEEDS_GPU[$st]:-}" ]; then
    if [ "$DRY" = 1 ]; then dev="${GPUS[0]}"
    else
      dev="$(wait_gpu)"
      if [ -z "$dev" ]; then
        if [ "$CPU_FALLBACK" = 1 ]; then dev=cpu; log "단계 $st: 배정 GPU 가 비지 않아 CPU 로 돈다(--cpu-fallback, 스모크 epochs 3) · $(gpu_desc)"
        else record "$st" "-" "blocked_gpu" "" "" "" ""; log "단계 $st: 배정 GPU 가 ${GPU_WAIT}분 안에 비지 않았다 · $(gpu_desc)"; continue; fi
      else
        log "단계 $st: GPU $dev 사용(메모리 $(gpu_mem "$dev") MiB, 계산 프로세스 $(gpu_napps "$dev"))"
      fi
    fi
  fi
  if [ "$st" = b_fit ] && [ "$dev" = cpu ] && [ -z "$SGEN" ]; then                # CPU 대체 스모크: cfm 표본 수를 줄여 시간 상한 안에서 경로만 확인한다
    SGEN="$SMOKE_CPU_SGEN"; log "단계 $st: CPU 대체 스모크이므로 --s-gen $SGEN 을 쓴다(뒤 단계 b_score 도 같은 값)"
  fi
  build_cmd "$st" "$dev"
  if [ "$DRY" = 1 ]; then log "[dry-run] $st: $PY ${CMD[*]}"; ST[$st]=ok; continue; fi
  echo "$st $(date +%s) $dev" > "$STATE"
  left=$(budget_left)
  log "단계 $st 시작 · 장치 $dev · 명령 $PY ${CMD[*]}"
  s0=$(date +%s)
  if [ "$dev" = cpu ]; then
    if [ "$BUDGET_MIN" -gt 0 ]; then CUDA_VISIBLE_DEVICES="" timeout --kill-after=60 "${left}s" "$PY" "${CMD[@]}" > "$lg" 2>&1; rc=$?
    else CUDA_VISIBLE_DEVICES="" "$PY" "${CMD[@]}" > "$lg" 2>&1; rc=$?; fi
  else                                            # 주 프로세스도 배정 GPU 하나만 보게 한다(워커는 하네스가 같은 번호로 다시 둔다)
    if [ "$BUDGET_MIN" -gt 0 ]; then CUDA_VISIBLE_DEVICES="$dev" timeout --kill-after=60 "${left}s" "$PY" "${CMD[@]}" > "$lg" 2>&1; rc=$?
    else CUDA_VISIBLE_DEVICES="$dev" "$PY" "${CMD[@]}" > "$lg" 2>&1; rc=$?; fi
  fi
  s1=$(date +%s)
  status="$(classify "$rc" "$lg")"
  record "$st" "$dev" "$status" "$rc" "$s0" "$s1" "$lg"
  if [ "$status" = timeout ]; then STOP="budget"; fi
  if [ "$status" != ok ]; then
    grep -E -e "^\[거부\]" -e "^\[FAIL\]" -e "^  \[failed\]" -e "Error" "$lg" 2>/dev/null | head -5 | while IFS= read -r l; do log "  $st: ${l:0:300}"; done
  fi
done

# ---------------------------------------------------------------- 8. 사전 점검 확정표와 마무리
if [ "$MODE" != full ] && [ "$DRY" = 0 ] && [ "$SUMMARIZE_ONLY" = 0 ]; then      # 계획서 2.9절 확정 규칙(스모크 확정표는 대체값)
  $PY "$HELPER" decide --out-dir "$OUT" --stage "$MODE" 2>&1 | while IFS= read -r l; do log "$l"; done
fi
write_status
n_ok=0; n_fail=0; n_stop=0
for st in "${STEPS[@]}"; do
  case "${ST[$st]:-}" in
    ok) n_ok=$(( n_ok + 1 )) ;;
    failed|failed_units|refused) n_fail=$(( n_fail + 1 )) ;;
    *) n_stop=$(( n_stop + 1 )) ;;
  esac
done
log "끝 · ok $n_ok · 실패 $n_fail · 실행하지 못함 $n_stop · 전체 $(elapsed) s · 상태 표 $OUT/lgu_status.csv"
if [ "$n_fail" -gt 0 ]; then exit 1; elif [ "$n_stop" -gt 0 ]; then exit 3; else exit 0; fi
