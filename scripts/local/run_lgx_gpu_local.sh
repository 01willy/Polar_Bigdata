#!/usr/bin/env bash
# LGX 확장 A(Qjpbeb, 2026-09-30 09:02 사용자 비용 지시로 중지)에서 남은 GPU 조각(FT-Transformer, 축 g45·nn)을 로컬 GPU 에서
# 이어 돌리는 실행기. 같은 실행기로 교차 환경 점검 (iii)의 재적합(별도 출력 폴더)도 돈다.
#
# 지위: 보조(교차 환경). WRAPUP 8.4 (i) 이 'CatBoost·ridge 불통과'였으므로 8.6 (나)의 합치기는 주 판정에 쓰지 않는다.
#       조각은 data/processed/lgx_local(또는 xenv 폴더)에만 쓰고, 주 집계 폴더(data/processed/lgx, Rescale 조각만)에는 쓰지 않는다.
#       근거와 표기 규칙: docs/EXECUTION_PLAN_REMAINING_2026-09-30.md 6.1(LG 개정 15 초안). 개정이 커밋되기 전에는 run 을 거부한다
#       (이어 실행과 점검 (iii) 모두. LGXL_SKIP_PREREG_CHECK=1 로 끌 수 있으나 쓰지 않는다).
#
# 사용법(저장소 루트에서)
#   scripts/local/run_lgx_gpu_local.sh seed        Rescale 완료 조각 가운데 (g45, nn) × ftt 조각을 하드링크로 둔다(목록 run_lgx_local/seeded.txt)
#   scripts/local/run_lgx_gpu_local.sh check       설정 해시 대조(학습 없음, 스레드 1)
#   LGXL_GPUS=8 setsid nohup scripts/local/run_lgx_gpu_local.sh run scripts/local/lgx_local_queue.txt > logs/lgx_local/nohup.out 2>&1 < /dev/null &
#   교차 환경 점검 (iii):
#   LGXL_MODE=xenv LGXL_OUT=data/processed/lgw/xenv/lgx_gpu_1 LGXL_GPUS=8 setsid nohup scripts/local/run_lgx_gpu_local.sh run \
#       scripts/local/lgx_xenv_iii_queue.txt > logs/lgx_local/nohup_xenv1.out 2>&1 < /dev/null &
#   touch <출력>/run_lgx_local/drain        전체 드레인(진행 중 단위를 끝낸 뒤 멈춘다, 종료 코드 3)
#
# 큐 파일 형식: 한 줄에 '<축> <대상:모드> <학습기> [h42 에 더할 인자 ...]'. '#' 로 시작하는 줄은 건너뛴다.
#
# 환경 변수
#   LGXL_MODE        cont(기본, 남은 조각 이어 실행) 또는 xenv(교차 환경 점검 재적합, seed·개정 확인 없음)
#   LGXL_OUT         출력 폴더. 기본 data/processed/lgx_local. data/processed/lgx 는 거부한다
#   LGXL_GPUS        후보 GPU(nvidia-smi 번호, 공백 구분, 앞 번호 우선). 기본 "8". LGF 가 끝나면 "9 8 7 6 5"
#   LGXL_ALLOW_0_4   1 이면 0–4 도 후보로 받는다(사용자 확인 뒤에만). 기본 0
#   LGXL_MAX_JOBS    동시 작업 수(GPU 하나에 작업 하나). 기본 1
#   LGXL_THREADS     작업당 스레드. 기본 2(주 프로세스 포함 약 3스레드)
#   LGXL_OUR_THREADS 다른 우리 작업의 스레드 합(LGT·LGF·집계). 기본 20. 이 값 + 작업 수 × (스레드 + 1) ≤ 32 일 때만 새 작업을 시작한다
#   LGXL_UNSEED      seed 에서 뺄 Rescale 조각 이름(공백 구분). 기본 'lgxg__gpu__Lena__x__s5__ftt'(점검 (iii) 의 규모 보충 단위.
#                    로컬에서 다시 적합해 Rescale 조각과 대조한다. 보조 집계는 같은 이름이면 Rescale 조각을 쓴다)
#
# 실행 중 조정(매 주기 다시 읽는다. 시작 때의 환경 변수보다 앞선다)
#   <출력>/run_lgx_local/our_threads   다른 우리 작업의 스레드 합(정수). 집계가 겹치는 구간에 값을 올린다
#   <출력>/run_lgx_local/gpus          후보 GPU(공백 구분). LGF 창 마감 뒤 '9 8 7 6 5' 로 넓힐 때 쓴다
#   <출력>/run_lgx_local/max_jobs      동시 작업 수(정수)
#
# 점유 판정(LGF·LGT 와 같다): 메모리 사용 ≤ 50 MiB 이고 계산 프로세스 0. 시작 뒤에는 자기 프로세스 그룹 밖의 계산 프로세스가
# 나타나면 그 작업을 드레인한다(진행 중 단위가 끝나 unit.json 이 늘면 SIGTERM, 큐 앞에 되돌린다. --resume 이라 끝난 단위는 건너뛴다).
set -u -o pipefail
cd /home/willy010313/Polar_Bigdata || exit 2
[ "${LGXL_NICED:-0}" = 1 ] || exec env LGXL_NICED=1 nice -n 10 "$0" "$@"

PY=${PY:-python3}
H42=scripts/3_deep_learning/h42_label_grid_ext.py
MODE=${LGXL_MODE:-cont}
SRC=${LGXL_SRC:-data/processed/lgx}           # Rescale 조각(주 집계, 읽기 전용)
OUT=${LGXL_OUT:-data/processed/lgx_local}      # 이어 실행 또는 점검 재적합의 출력
RUN=$OUT/run_lgx_local
LOG=logs/lgx_local
CANDS=${LGXL_GPUS:-8}
MAXJ=${LGXL_MAX_JOBS:-1}
THR=${LGXL_THREADS:-2}
OURT=${LGXL_OUR_THREADS:-20}
UNSEED=${LGXL_UNSEED-lgxg__gpu__Lena__x__s5__ftt}
MEM_MAX=50; LOAD_START_MAX=64; LOAD_RUN_MAX=96; POLL=60
MAN=$OUT/platform_manifest.csv
PLAN_LG=docs/EXPERIMENT_PLAN_LG_2026-09-29.md

case "$MODE" in cont|xenv) ;; *) echo "LGXL_MODE 는 cont 또는 xenv 다"; exit 2;; esac
if [ "$(realpath -m "$OUT")" = "$(realpath -m "$SRC")" ]; then echo "[거부] 출력 폴더가 주 집계 폴더($SRC)와 같다"; exit 2; fi
mkdir -p "$RUN" "$LOG" "$OUT/shards"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG/queue_$(basename "$OUT").log"; }
load1() { cut -d' ' -f1 /proc/loadavg; }
ctl_read() {                                     # 실행 중 조정 파일을 다시 읽는다(값이 바뀌면 기록한다)
  local v
  if [ -f "$RUN/our_threads" ]; then v=$(tr -dc '0-9' < "$RUN/our_threads"); [ -n "$v" ] && [ "$v" != "$OURT" ] && { log "조정: 다른 우리 작업 스레드 $OURT → $v"; OURT=$v; }; fi
  if [ -f "$RUN/gpus" ]; then v=$(tr -s ' ,\n' '  ' < "$RUN/gpus" | sed 's/^ *//; s/ *$//'); [ -n "$v" ] && [ "$v" != "$CANDS" ] && { log "조정: 후보 GPU '$CANDS' → '$v'"; CANDS=$v; }; fi
  if [ -f "$RUN/max_jobs" ]; then v=$(tr -dc '0-9' < "$RUN/max_jobs"); [ -n "$v" ] && [ "$v" != "$MAXJ" ] && { log "조정: 최대 작업 $MAXJ → $v"; MAXJ=$v; }; fi
  return 0
}
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }
uuid_of() { nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' '; }
mem_of() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
pids_on() { local u; u=$(uuid_of "$1"); [ -n "$u" ] || return 0; nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | tr -d ' ' | awk -F, -v u="$u" '$1==u{print $2}'; }
gpu_free() { local m n; m=$(mem_of "$1"); n=$(pids_on "$1" | grep -c .); [ -n "$m" ] && [ "$m" -le "$MEM_MAX" ] && [ "$n" -eq 0 ]; }
allowed_gpu() { case "$1" in 5|6|7|8|9) return 0;; 0|1|2|3|4) [ "${LGXL_ALLOW_0_4:-0}" = 1 ];; *) return 1;; esac; }
foreign_on() {                                   # foreign_on <gpu> <pgid>: 이 작업 그룹 밖의 계산 프로세스가 있으면 참
  local p g
  for p in $(pids_on "$1"); do
    g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    [ "$g" = "$2" ] || return 0
  done
  return 1
}
ours_on() { local p g; for p in $(pids_on "$1"); do g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '); [ "$g" = "$2" ] && return 0; done; return 1; }
axtag() { case "$1" in g45) echo g;; nn) echo nn;; r0) echo r0;; *) echo "$1";; esac; }
units_of() {                                     # units_of <축> <대상> <학습기>: 출력 폴더의 해당 unit.json 수
  ls "$OUT"/shards/lgx"$(axtag "$1")"__gpu__"$2"__*__"$3"_unit.json 2>/dev/null | grep -c .
}

prereg_ok() {                                    # LG 개정 15 가 계획서에 있고 커밋되었는지(결과 열람 전 등록의 확인)
  grep -q '개정 15(' "$PLAN_LG" 2>/dev/null && git diff --quiet HEAD -- "$PLAN_LG" 2>/dev/null
}

seed() {                                         # 같은 (축, 학습기)의 Rescale 완료 조각을 하드링크(없으면 복사)로 둔다
  local n=0 f b
  local u skip
  for f in "$SRC"/shards/lgxg__gpu__*__ftt_* "$SRC"/shards/lgxnn__gpu__*__ftt_*; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); [ -e "$OUT/shards/$b" ] && continue
    skip=0; for u in $UNSEED; do case "$b" in "${u}_"*) skip=1;; esac; done
    if [ "$skip" = 1 ]; then echo "$b" >> "$RUN/unseeded.txt"; continue; fi
    ln "$f" "$OUT/shards/$b" 2>/dev/null || cp -p "$f" "$OUT/shards/$b"
    n=$((n + 1)); echo "$b" >> "$RUN/seeded.txt"
  done
  touch "$RUN/seeded.txt"
  log "seed: Rescale 조각 파일 ${n}개를 $OUT/shards 에 두었다(목록 $RUN/seeded.txt). 뺀 단위: '${UNSEED}'"
  echo "seed: ${n}개(뺀 단위 '${UNSEED}')"
}

check() {                                        # 학습 없이 설정 해시를 계산해 Rescale 조각의 해시와 대조한다
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$PY" - "$SRC" <<'EOF'
import sys, json, glob, importlib.util
src = sys.argv[1]
spec = importlib.util.spec_from_file_location("h42x", "scripts/3_deep_learning/h42_label_grid_ext.py")
m = importlib.util.module_from_spec(spec); sys.argv = ["h42"]; spec.loader.exec_module(m)
tg = "Lena:x,Canada:x,CA-2:x,CA-3:x"
a = m.parse_args(["--part", "gpu", "--axes", "g45,nn", "--learners", "ftt", "--targets", tg, "--allow-local", "--out-dir", "data/processed/lgx_local"])
_HG = {}
def hg_for(target, mode):
    # 실제 이어 실행과 같은 인자(큐 한 줄 = --targets <대상:모드> 하나)로 h40 인자를 만든다. learner_axis 가 대상 목록에 따라 달라지기 때문이다
    k = f"{target}:{mode}"
    if k not in _HG:
        ak = m.parse_args(["--part", "gpu", "--axes", "g45", "--learners", "ftt", "--targets", k, "--allow-local", "--out-dir", "data/processed/lgx_local"])
        _HG[k] = m.H.parse_args(m.g45_argv(ak))
    return _HG[k]
bad = n = warn = 0
for f in sorted(glob.glob(f"{src}/shards/lgxg__gpu__*__ftt_unit.json")) + sorted(glob.glob(f"{src}/shards/lgxnn__gpu__*__ftt_unit.json")):
    u = json.load(open(f)); n += 1
    if "/lgxg__" in f:
        # g45: learner_axis 는 로컬 --targets 에 따라 달라지므로 공통 해시(cfg_common)로 대조한다(h40 CFG_UNIT_KEYS 제외).
        # 전체 해시가 달라 --resume 이 Rescale 완료 조각을 건너뛰지 못한다. 같은 대상의 완료 분할은 로컬에서 다시 돌고
        # 새 조각이 하드링크를 대체한다(주 폴더의 파일은 os.replace 라 바뀌지 않는다). platform_manifest 가 이를 local 로 적는다.
        cfg = m.H.unit_cfg(hg_for(u["target"], u["mode"]), "gpu", u["target"], u["mode"], int(u["split"]), "ftt")
        want, have = m.H.cfg_hash(cfg, common=True), u.get("cfg_common")
        # 전체 해시도 출력한다. 같으면 --resume 이 Rescale 완료 분할을 건너뛰고, 다르면 로컬에서 다시 적합한다(주 폴더 파일은 바뀌지 않는다)
        full_ok = m.H.cfg_hash(cfg) == u.get("cfg_hash")
        warn += (not full_ok)
        print(f"   g45 전체 해시 {'같음' if full_ok else '다름(이 분할은 로컬에서 다시 적합된다)'}: {f.split('/')[-1]}")
    else:
        # nn: unit_cfg_x 에는 대상·분할이 없으므로 전체 해시가 같아야 --resume 이 완료 조각을 건너뛴다
        want = m.H.cfg_hash(m.unit_cfg_x(a, "nn", "ftt"))
        have = u.get("cfg_hash")
    ok = want == have
    bad += (not ok)
    print(f"{'ok ' if ok else 'BAD'} {f.split('/')[-1]} rescale={have} local={want}")
print(f"[check] 대조 {n} · 불일치 {bad} · g45 전체 해시 다름 {warn}")
sys.exit(1 if (bad or n == 0) else 0)
EOF
}

manifest_new() {                                 # 새로 생긴(주 폴더 파일과 같은 inode 가 아닌) unit.json 을 플랫폼 표에 적는다
  local f b d py tv
  [ -f "$MAN" ] || echo "unit_json,platform,host,gpu_index,gpu_uuid,gpu_name,python,torch,recorded_at" > "$MAN"
  py=$("$PY" -c 'import platform;print(platform.python_version())' 2>/dev/null)
  tv=$("$PY" -c 'import importlib.metadata as m;print(m.version("torch"))' 2>/dev/null)
  for f in "$OUT"/shards/*_unit.json; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    [ "$f" -ef "$SRC/shards/$b" ] && continue    # Rescale 조각(하드링크) 그대로다
    grep -q "^$b," "$MAN" && continue
    d=$(grep -o '"device": *"[0-9]*"' "$f" | grep -o '[0-9]*' | tail -1)
    echo "$b,local,$(hostname),$d,$(uuid_of "${d:-x}"),$(nvidia-smi -i "${d:-x}" --query-gpu=name --format=csv,noheader 2>/dev/null),$py,$tv,$(date '+%F %T')" >> "$MAN"
  done
}

LASTPID=""
start_job() {                                    # start_job <gpu> <큐 줄>. PID 는 전역 LASTPID 에 둔다(명령 치환으로 부르지 않는다)
  local g="$1" line="$2" ax tg lr extra jl
  read -r ax tg lr extra <<< "$line"
  jl="$LOG/$(basename "$OUT")_${ax}_${tg%%:*}_${lr}_g${g}_$(date +%Y%m%dT%H%M%S).log"
  # shellcheck disable=SC2086
  setsid env CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS="$THR" MKL_NUM_THREADS="$THR" OPENBLAS_NUM_THREADS="$THR" \
    NUMEXPR_NUM_THREADS="$THR" "$PY" -u "$H42" --part gpu --axes "$ax" --learners "$lr" --targets "$tg" --gpus "$g" \
    --procs-per-gpu 1 --threads "$THR" --allow-local --resume --no-summarize --out-dir "$OUT" ${extra:-} > "$jl" 2>&1 < /dev/null &
  LASTPID=$!
  log "시작: $line · GPU $g · pid $LASTPID(프로세스 그룹) · 로그 $jl"
}

run() {
  local QUEUE="${1:-}"
  [ -f "$QUEUE" ] || { echo "큐 파일 없음: '$QUEUE'"; exit 2; }
  if [ "$MODE" = cont ] && [ ! -f "$RUN/seeded.txt" ]; then echo "[거부] seed 를 먼저 한다"; exit 2; fi
  if [ "${LGXL_SKIP_PREREG_CHECK:-0}" != 1 ] && ! prereg_ok; then     # 이어 실행과 점검 (iii) 모두 등록 뒤에 돈다
    echo "[거부] LG 개정 15(로컬 이어 실행의 지위, 표기 규칙, 점검 (iii)의 계속 규칙)가 커밋되지 않았다. 결과 열람 전에 커밋한다"; exit 2
  fi
  if [ -f "$RUN/lock" ] && kill -0 "$(cat "$RUN/lock")" 2>/dev/null; then echo "이미 실행 중"; exit 2; fi
  echo $$ > "$RUN/lock"; trap 'rm -f "$RUN/lock"' EXIT
  local -a Q
  mapfile -t Q < <(grep -v '^[[:space:]]*#' "$QUEUE" | awk 'NF>=3')
  declare -A PID=() GPU=() JOB=() DRN=() BASE=()
  local slot rc g k ax tg lr extra started drained=0
  log "실행 시작($MODE): 큐 ${#Q[@]}개 · 후보 GPU '$CANDS' · 최대 작업 $MAXJ · 스레드 $THR · 출력 $OUT"
  while :; do
    ctl_read
    [ -f "$RUN/drain" ] && drained=1
    for slot in "${!PID[@]}"; do                 # 끝난 작업 정리와 드레인 판정
      g=${GPU[$slot]}
      read -r ax tg lr extra <<< "${JOB[$slot]}"
      if ! kill -0 "${PID[$slot]}" 2>/dev/null; then
        wait "${PID[$slot]}" 2>/dev/null; rc=$?
        manifest_new
        log "끝: ${JOB[$slot]} · GPU $g · 종료 코드 $rc"
        if [ -n "${DRN[$slot]:-}" ] && [ "$drained" -eq 0 ]; then Q=("${JOB[$slot]}" "${Q[@]}"); fi   # 드레인한 작업은 큐 앞에 되돌린다
        unset "PID[$slot]" "GPU[$slot]" "JOB[$slot]" "DRN[$slot]" "BASE[$slot]"
        continue
      fi
      if [ -z "${DRN[$slot]:-}" ] && { [ "$drained" -eq 1 ] || foreign_on "$g" "${PID[$slot]}"; }; then
        DRN[$slot]=1; BASE[$slot]=$(units_of "$ax" "${tg%%:*}" "$lr")
        log "드레인 표시: ${JOB[$slot]} · GPU $g($([ "$drained" -eq 1 ] && echo '드레인 파일' || echo '다른 사용자의 계산 프로세스'))"
      fi
      if [ -n "${DRN[$slot]:-}" ]; then          # 진행 중 단위가 끝나면(unit.json 이 늘면) 멈춘다
        k=$(units_of "$ax" "${tg%%:*}" "$lr")
        if [ "$k" -gt "${BASE[$slot]}" ]; then kill -TERM -- "-${PID[$slot]}" 2>/dev/null; log "드레인: ${JOB[$slot]} 를 단위 경계에서 멈췄다"; fi
      fi
    done
    if [ "$drained" -eq 1 ] && [ "${#PID[@]}" -eq 0 ]; then log "드레인 완료. 멈춘다"; exit 3; fi
    if [ "${#Q[@]}" -eq 0 ] && [ "${#PID[@]}" -eq 0 ]; then log "큐 완료"; exit 0; fi
    if [ "$drained" -eq 0 ] && [ "${#Q[@]}" -gt 0 ] && [ "${#PID[@]}" -lt "$MAXJ" ] \
       && ! gt "$(load1)" "$LOAD_START_MAX" && [ $(( OURT + (${#PID[@]} + 1) * (THR + 1) )) -le 32 ]; then
      for g in $CANDS; do                        # 빈 후보 GPU 하나에 큐의 앞 작업을 올린다
        allowed_gpu "$g" || continue
        [[ " ${GPU[*]:-} " == *" $g "* ]] && continue
        gpu_free "$g" || continue
        slot="s$(date +%s%N)"
        start_job "$g" "${Q[0]}"; PID[$slot]=$LASTPID; GPU[$slot]=$g; JOB[$slot]="${Q[0]}"; Q=("${Q[@]:1}")
        started=$(date +%s)                      # 배치 확인: 3분 안에 이 그룹의 프로세스가 배정 GPU 에만 있어야 한다
        while [ $(( $(date +%s) - started )) -lt 180 ]; do ours_on "$g" "${PID[$slot]}" && break; kill -0 "${PID[$slot]}" 2>/dev/null || break; sleep 10; done
        for k in 0 1 2 3 4 5 6 7 8 9; do
          [ "$k" = "$g" ] && continue
          if ours_on "$k" "${PID[$slot]}"; then log "배치 오류: ${JOB[$slot]} 가 GPU $k 에 올라갔다. 전체를 멈춘다"; kill -TERM -- "-${PID[$slot]}"; exit 1; fi
        done
        break
      done
    fi
    gt "$(load1)" "$LOAD_RUN_MAX" && log "load average $(load1) > $LOAD_RUN_MAX: 새 작업을 시작하지 않는다"
    sleep "$POLL"
  done
}

case "${1:-}" in
  seed) seed ;;
  check) check ;;
  run) shift; run "$@" ;;
  *) echo "사용: $0 seed|check|run <큐 파일>"; exit 2 ;;
esac
