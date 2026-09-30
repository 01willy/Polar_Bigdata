#!/usr/bin/env bash
# LGX 확장 A(Rescale 작업 Qjpbeb, 2026-09-30 09:02 사용자 비용 지시로 중지)의 남은 GPU 단위를 로컬 빈 GPU 에서 이어 돌리는 상위 실행기.
#
# 지위: 보조(교차 환경). WRAPUP 8.4 (i) 이 'CatBoost·ridge 불통과'이므로 로컬 조각은 주 판정에 합치지 않는다(LG 개정 14 (4),
#       WRAPUP 8.6 비교 규칙 (1)(2)). 로컬 조각은 data/processed/lgx_local 에만 쓰고, 주 집계 폴더 data/processed/lgx(Rescale 조각만)는
#       해제 뒤 읽기 전용으로 둔다. 근거와 표기 규칙: docs/EXECUTION_PLAN_REMAINING_2026-09-30.md 3.2(G3–G6), 6.1(LG 개정 15 초안).
#       LG 개정 15 가 커밋되기 전에는 unpack·xenv·run 을 거부한다(LGXC_SKIP_PREREG_CHECK=1 로 끌 수 있으나 쓰지 않는다).
#
# 하위 명령(저장소 루트에서, 모두 nice 10)
#   status      진행 정보만 출력한다(해제 여부, unit.json 수, 남은 단위 수, 실행기 잠금, GPU 점유). 결과 표와 RMSE 는 열지 않는다
#   unpack      Qjpbeb·ShFtT 묶음에서 data/processed/lgx 경로만 푼다. 이미 풀려 있으면(표지와 unit.json 수가 맞으면) 건너뛴다.
#               두 묶음의 이름 겹침 확인, 수 대조(CPU 3,860, h41 292, GPU 514 + 77), 조각 sha256 목록, 읽기 전용 처리, 해제 시각 기록
#   remaining   남은 FT-T 단위(g45·nn)의 이름 목록을 만든다(기대 집합 = r0 FT-T 완료 단위의 (대상, 분할)). 큐와 대조한다
#   seed        실행기(run_lgx_gpu_local.sh) seed: Rescale 완료 FT-T 조각을 data/processed/lgx_local 에 하드링크
#   check       실행기 check: 설정 해시 대조(학습 없음). 불일치가 있으면 멈춘다
#   xenv        교차 환경 점검 (iii) 재적합 2회(출력 data/processed/lgw/xenv/lgx_gpu_1, _2). 끝난 회차는 건너뛴다
#   gate        점검 (iii) 대조(lgw_xenv_gate_iii.py)와 계속 규칙 판정. '계속'이면 종료 코드 0, 아니면 5
#   run         이어 실행 큐(scripts/local/lgx_local_queue.txt)를 돈다(--resume, 끝난 단위는 건너뛴다)
#   supplement  레나 x g45 분할 5 FT-T 의 로컬 재적합과 Rescale 조각의 보충 대조(계속 규칙에 쓰지 않는다)
#   chain       unpack → remaining → seed → check → xenv → gate → run → supplement. 세션과 분리해 띄운다:
#               setsid nohup scripts/local/run_lgx_local_continue.sh chain > logs/lgx_local/chain.out 2>&1 < /dev/null &
#
# 환경 변수
#   LGXC_GPUS       후보 GPU(공백 구분, 앞 번호 우선). 기본 "8". LGF 창 마감 뒤에는 실행 중 조정 파일
#                   data/processed/lgx_local/run_lgx_local/gpus 에 '9 8 7 6 5', max_jobs 에 4 를 쓴다(재시작 불필요)
#   LGXC_THREADS    작업당 스레드. 기본 2
#   LGXC_OUR_THREADS 다른 우리 작업의 스레드 합. 기본 20(실행 중에는 run_lgx_local/our_threads 파일로 조정)
#   LGXC_IGNORE_SWITCH 1 이면 logs/lgf/switch_condition(LGF 가 GPU 8 을 요구하는 조건)이 있어도 새로 시작한다. 기본 0
#
# 점유 판정(LGF·LGT 와 같다): 메모리 사용 ≤ 50 MiB 이고 계산 프로세스 0. 다른 사용자의 계산 프로세스가 배정 GPU 에 나타나면
# 실행기가 진행 중 단위를 끝낸 뒤(unit.json 이 늘면) 그 작업을 멈추고 큐 앞에 되돌린 다음, 빈 GPU 가 생길 때까지 60 초마다 기다렸다
# 다시 시작한다(--resume). 전체 드레인: touch data/processed/lgx_local/run_lgx_local/drain(종료 코드 3).
# 로그: logs/lgx_local/continue.log(이 스크립트), queue_<출력 폴더>.log(실행기), 작업별 로그.
set -u -o pipefail
SELF=$(readlink -f "$0")
cd /home/willy010313/Polar_Bigdata || exit 2
[ "${LGXC_NICED:-0}" = 1 ] || exec env LGXC_NICED=1 nice -n 10 "$SELF" "$@"

ENGINE=scripts/local/run_lgx_gpu_local.sh
GATE=scripts/2_evaluation/lgw_xenv_gate_iii.py
LOG=logs/lgx_local; mkdir -p "$LOG"
CLOG=$LOG/continue.log
SRC=data/processed/lgx
OUT=data/processed/lgx_local
RUN=$OUT/run_lgx_local
A_TAR=results/rescale_lgx_a/results_lgx_all.tar.gz
B_TAR=results/rescale_lgx_b/results_lgx_gpu_rm.tar.gz
MARK=$SRC/.unpacked_rescale.txt
SHA_LIST=$SRC/rescale_shards_sha256.txt
EXP_CPU=3860; EXP_LADDER=292; EXP_GPU_A=514; EXP_GPU_B=77
PLAN_LG=docs/EXPERIMENT_PLAN_LG_2026-09-29.md
QUEUE=scripts/local/lgx_local_queue.txt
XQUEUE=scripts/local/lgx_xenv_iii_queue.txt
XDIR=data/processed/lgw/xenv
GSUM=data/processed/lgw/lgw_xenv_gate_iii_summary.json
SUPP_UNIT=${LGXC_SUPP_UNIT:-lgxg__gpu__Lena__x__s5__ftt}
GPUS=${LGXC_GPUS:-8}
THR=${LGXC_THREADS:-2}
OURT=${LGXC_OUR_THREADS:-20}
PY=${PY:-python3}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CLOG"; }
die() { log "[중단] $1"; exit "${2:-1}"; }
cnt() { find "$1" -name "$2" 2>/dev/null | wc -l; }
prereg_ok() {
  [ "${LGXC_SKIP_PREREG_CHECK:-0}" = 1 ] && return 0
  grep -q '개정 15(' "$PLAN_LG" 2>/dev/null && git diff --quiet HEAD -- "$PLAN_LG" 2>/dev/null
}
need_prereg() { prereg_ok || die "LG 개정 15(로컬 이어 실행의 지위·폴더·표기, 점검 (iii)의 계속 규칙)가 커밋되지 않았다. 결과 열람 전에 커밋한다" 2; }
switch_ok() {
  if [ -f logs/lgf/switch_condition ] && [ "${LGXC_IGNORE_SWITCH:-0}" != 1 ]; then
    log "logs/lgf/switch_condition 이 있다(LGF 가 GPU 를 요구하는 조건). 새 작업을 시작하지 않는다. 메인 세션의 전환 조치를 기다린다"
    return 1
  fi
  return 0
}
engine() {                                       # engine <모드> <출력> <하위 명령> [인자]
  local mode=$1 out=$2; shift 2
  LGXL_MODE=$mode LGXL_OUT=$out LGXL_GPUS="$GPUS" LGXL_THREADS="$THR" LGXL_OUR_THREADS="$OURT" bash "$ENGINE" "$@"
}

counts() {                                       # 해제된 Rescale 조각 수(이름만 센다)
  N_CPU=$(cnt "$SRC/shards" 'lgx*__cpu__*_unit.json')
  N_LAD=$(cnt "$SRC/ladder" '*_unit.json')
  N_GPU=$(cnt "$SRC/shards" 'lgx*__gpu__*_unit.json')
  N_GPU_RM=$(cnt "$SRC/shards" 'lgx*__gpu__*__realmlp_unit.json')
}

unpack() {
  need_prereg
  counts
  if [ -f "$MARK" ] && [ "$N_CPU" -eq "$EXP_CPU" ] && [ "$N_LAD" -eq "$EXP_LADDER" ] && [ "$N_GPU" -eq $(( EXP_GPU_A + EXP_GPU_B )) ]; then
    log "unpack: 이미 해제되어 있다(표지 $MARK, CPU $N_CPU, h41 $N_LAD, GPU $N_GPU). 건너뛴다"
    return 0
  fi
  [ -f "$A_TAR" ] || die "묶음 없음: $A_TAR(Qjpbeb 회수를 먼저 한다)"
  [ -f "$B_TAR" ] || die "묶음 없음: $B_TAR"
  local fa fb ov
  fa=$(mktemp); fb=$(mktemp)
  tar tzf "$A_TAR" | grep '^data/processed/lgx/' | grep -v '/$' | sort > "$fa"
  tar tzf "$B_TAR" | grep '^data/processed/lgx/' | grep -v '/$' | sort > "$fb"
  ov=$(comm -12 "$fa" "$fb" | wc -l)
  log "unpack: 묶음 A $(wc -l < "$fa")개 파일, 묶음 B $(wc -l < "$fb")개 파일(data/processed/lgx 아래), 이름 겹침 $ov"
  if [ "$ov" -ne 0 ]; then comm -12 "$fa" "$fb" | head -5 >> "$CLOG"; rm -f "$fa" "$fb"; die "두 묶음의 파일 이름이 겹친다. 원인을 확인한다"; fi
  rm -f "$fa" "$fb"
  if [ -d "$SRC" ] && [ ! -f "$MARK" ]; then log "unpack: 표지 없이 $SRC 가 있다(중단된 해제로 본다). 쓰기 권한을 되돌리고 다시 푼다"; chmod -R u+w "$SRC" 2>/dev/null; fi
  local t0; t0=$(date '+%F %T')
  tar xzf "$A_TAR" -C . data/processed/lgx || die "묶음 A 해제 실패"
  tar xzf "$B_TAR" -C . data/processed/lgx || die "묶음 B 해제 실패"
  mkdir -p results/rescale_lgx_a/bundle_top results/rescale_lgx_b/bundle_top    # 묶음의 로그·상태 파일은 회수 폴더 아래에 둔다(저장소 루트에 풀지 않는다)
  tar xzf "$A_TAR" -C results/rescale_lgx_a/bundle_top --exclude='data/*' 2>/dev/null
  tar xzf "$B_TAR" -C results/rescale_lgx_b/bundle_top --exclude='data/*' 2>/dev/null
  counts
  log "unpack: 해제 시각 $t0 · CPU $N_CPU/$EXP_CPU · h41 $N_LAD/$EXP_LADDER · GPU $N_GPU(기대 $EXP_GPU_A + $EXP_GPU_B, 그 가운데 RealMLP $N_GPU_RM/$EXP_GPU_B)"
  [ "$N_CPU" -eq "$EXP_CPU" ] || die "CPU 조각 수가 다르다($N_CPU ≠ $EXP_CPU)"
  [ "$N_LAD" -eq "$EXP_LADDER" ] || die "h41 조각 수가 다르다($N_LAD ≠ $EXP_LADDER)"
  [ "$N_GPU" -eq $(( EXP_GPU_A + EXP_GPU_B )) ] || die "GPU 조각 수가 다르다($N_GPU ≠ $(( EXP_GPU_A + EXP_GPU_B )))"
  [ "$N_GPU_RM" -eq "$EXP_GPU_B" ] || die "RealMLP 조각 수가 다르다($N_GPU_RM ≠ $EXP_GPU_B)"
  find "$SRC/shards" "$SRC/ladder/shards" -type f -print0 | sort -z | xargs -0 sha256sum > "$SHA_LIST"
  chmod -R a-w "$SRC/shards" "$SRC/ladder/shards"
  { echo "unpacked_at=$t0"; echo "finished_at=$(date '+%F %T')"; echo "a_tar_sha256=$(sha256sum "$A_TAR" | cut -c1-64)";
    echo "b_tar_sha256=$(sha256sum "$B_TAR" | cut -c1-64)"; echo "cpu=$N_CPU ladder=$N_LAD gpu=$N_GPU realmlp=$N_GPU_RM";
    echo "sha_list=$SHA_LIST($(wc -l < "$SHA_LIST") 파일)"; echo "read_only=$SRC/shards,$SRC/ladder/shards";
    echo "note=해제는 결과 열람이 아니다(WRAPUP 0.3). 조각 내용(blocksse, cells, runs 의 RMSE 열)은 열지 않았다"; } > "$MARK"
  log "unpack: 완료. 표지 $MARK, sha256 목록 $SHA_LIST, 조각 폴더 읽기 전용"
}

remaining() {                                    # 기대 집합 = r0 FT-T 완료 단위의 (대상, 분할). g45 는 분할 4·5, nn 은 전부
  [ -d "$SRC/shards" ] || die "해제 전이다(unpack 을 먼저 한다)"
  mkdir -p "$RUN"
  local f t s out=$RUN/remaining.txt n=0 miss=0 ax tg lr
  : > "$out"
  for f in "$SRC"/shards/lgxr0__gpu__*__x__s*__ftt_unit.json; do
    [ -e "$f" ] || continue
    t=$(basename "$f" | sed -E 's/^lgxr0__gpu__(.*)__x__s([0-9]+)__ftt_unit\.json$/\1/')
    s=$(basename "$f" | sed -E 's/^lgxr0__gpu__(.*)__x__s([0-9]+)__ftt_unit\.json$/\2/')
    [ -e "$SRC/shards/lgxnn__gpu__${t}__x__s${s}__ftt_unit.json" ] || { echo "nn ${t}:x s${s}" >> "$out"; n=$((n + 1)); }
    case "$s" in 4|5) [ -e "$SRC/shards/lgxg__gpu__${t}__x__s${s}__ftt_unit.json" ] || { echo "g45 ${t}:x s${s}" >> "$out"; n=$((n + 1)); } ;; esac
  done
  log "remaining: 남은 FT-T 단위 ${n}개(g45 $(grep -c '^g45' "$out"), nn $(grep -c '^nn' "$out")) → $out"
  while read -r ax tg _; do                      # 큐 대조: 남은 단위의 (축, 대상)이 큐에 있어야 한다
    grep -v '^[[:space:]]*#' "$QUEUE" | awk '{print $1, $2, $3}' | grep -qx "$ax $tg ftt" || { log "remaining: 큐에 없는 단위 $ax $tg"; miss=$((miss + 1)); }
  done < "$out"
  while read -r ax tg lr _; do                   # 큐에 있으나 남은 단위가 없는 줄(--resume 으로 즉시 끝난다)
    grep -q "^$ax $tg " "$out" || log "remaining: 큐 줄 '$ax $tg $lr' 에 남은 단위가 없다(보충 단위가 아니면 즉시 끝난다)"
  done < <(grep -v '^[[:space:]]*#' "$QUEUE" | awk 'NF>=3')
  [ "$miss" -eq 0 ] || die "큐에 없는 남은 단위 ${miss}개. 큐를 고친다"
}

xenv() {
  need_prereg
  local k d want have
  for k in 1 2; do
    d=$XDIR/lgx_gpu_$k
    want=4                                       # nn·r0 러시아 W 분할 1, g45 AL-2 분할 4·5
    have=$(cnt "$d/shards" '*_unit.json')
    if [ "$have" -ge "$want" ]; then log "xenv: ${k}회차 이미 끝남($have 단위). 건너뛴다"; continue; fi
    switch_ok || return 3
    log "xenv: ${k}회차 시작(GPU '$GPUS', 출력 $d)"
    engine xenv "$d" run "$XQUEUE"; local rc=$?
    log "xenv: ${k}회차 끝(종료 코드 $rc, 단위 $(cnt "$d/shards" '*_unit.json'))"
    [ "$rc" -eq 0 ] || return "$rc"
  done
}

gate() {
  [ -f "$SRC/.first_content_read.txt" ] || { echo "first_content_read=$(date '+%F %T') by=lgw_xenv_gate_iii.py(점검 (iii) 대조기, Rescale blocksse.npz 4단위)" > "$SRC/.first_content_read.txt";
    log "gate: WRAPUP 0.3 (3) 의 '조각 내용 첫 열람' 시각을 $SRC/.first_content_read.txt 에 적었다(대조기는 키별 절대 차만 낸다)"; }
  "$PY" "$GATE" >> "$CLOG" 2>&1 || die "점검 (iii) 대조 실패(로그 $CLOG)"
  local rule
  rule=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rule"])' "$GSUM" 2>/dev/null)
  log "gate: 점검 (iii) 계속 규칙 = '$rule'(요약 $GSUM. 키별 절대 차의 분포만 있다)"
  case "$rule" in 계속*) return 0 ;; *) return 5 ;; esac
}

run_queue() {
  need_prereg
  switch_ok || return 3
  [ -f "$RUN/seeded.txt" ] || die "seed 를 먼저 한다"
  log "run: 이어 실행 큐 $QUEUE(GPU '$GPUS', 스레드 $THR). 진행은 $LOG/queue_$(basename "$OUT").log"
  engine cont "$OUT" run "$QUEUE"; local rc=$?
  log "run: 끝(종료 코드 $rc. 0 큐 완료, 3 드레인)"
  return "$rc"
}

supplement() {
  [ -e "$OUT/shards/${SUPP_UNIT}_unit.json" ] || { log "supplement: 로컬 조각 없음($SUPP_UNIT). g45 레나 작업이 끝난 뒤 한다"; return 0; }
  if [ "$OUT/shards/${SUPP_UNIT}_unit.json" -ef "$SRC/shards/${SUPP_UNIT}_unit.json" ]; then
    log "supplement: $SUPP_UNIT 가 Rescale 조각의 하드링크다(seed 에서 빠지지 않았다). 보충 대조를 하지 않는다"; return 0
  fi
  "$PY" "$GATE" --supplement --run1-dir "$OUT/shards" --units "$SUPP_UNIT" \
      --gate-csv data/processed/lgw/lgw_xenv_gate_iii_supp.csv --summary data/processed/lgw/lgw_xenv_gate_iii_supp_summary.json >> "$CLOG" 2>&1 \
    || die "보충 대조 실패"
  log "supplement: 완료(data/processed/lgw/lgw_xenv_gate_iii_supp_summary.json. 계속 규칙에 쓰지 않는다)"
}

status() {
  counts
  echo "[해제] 표지 $([ -f "$MARK" ] && echo 있음 || echo 없음) · CPU $N_CPU/$EXP_CPU · h41 $N_LAD/$EXP_LADDER · GPU $N_GPU/$(( EXP_GPU_A + EXP_GPU_B ))"
  [ -f "$RUN/remaining.txt" ] && echo "[남은 단위(해제 시점)] $(wc -l < "$RUN/remaining.txt")"
  echo "[로컬 새 조각] $(tail -n +2 "$OUT/platform_manifest.csv" 2>/dev/null | wc -l) · seed $([ -f "$RUN/seeded.txt" ] && wc -l < "$RUN/seeded.txt" || echo 0)"
  for d in "$XDIR/lgx_gpu_1" "$XDIR/lgx_gpu_2"; do echo "[점검 (iii)] $d: $(cnt "$d/shards" '*_unit.json')/4 단위"; done
  [ -f "$GSUM" ] && echo "[점검 (iii) 규칙] $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rule"])' "$GSUM" 2>/dev/null)"
  for d in "$OUT" "$XDIR/lgx_gpu_1" "$XDIR/lgx_gpu_2"; do
    [ -f "$d/run_lgx_local/lock" ] && kill -0 "$(cat "$d/run_lgx_local/lock")" 2>/dev/null && echo "[실행기] $d 실행 중(pid $(cat "$d/run_lgx_local/lock"))"
    [ -f "$d/run_lgx_local/drain" ] && echo "[실행기] $d 드레인 파일 있음"
  done
  [ -f logs/lgf/switch_condition ] && echo "[LGF] 전환 규칙 조건 표지 있음: $(cat logs/lgf/switch_condition)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' '; echo
}

chain() {
  log "chain: 시작(pid $$)"
  unpack || exit $?
  remaining || exit $?
  [ -f "$RUN/seeded.txt" ] || { engine cont "$OUT" seed >> "$CLOG" 2>&1 || die "seed 실패"; }
  engine cont "$OUT" check >> "$CLOG" 2>&1 || die "설정 해시 대조 실패(check). 로그 $CLOG"
  log "chain: 설정 해시 대조 통과"
  xenv; local rc=$?
  [ "$rc" -eq 0 ] || die "점검 (iii) 재적합이 끝나지 않았다(종료 코드 $rc)" "$rc"
  if ! gate; then
    log "chain: 계속 규칙 불충족. 이어 실행(G4–G6)을 시작하지 않는다. 사용자에게 (iii) 요약의 분포만 보고한다"
    exit 5
  fi
  run_queue; rc=$?
  supplement
  log "chain: 끝(이어 실행 종료 코드 $rc)"
  exit "$rc"
}

case "${1:-}" in
  status) status ;;
  unpack) unpack ;;
  remaining) remaining ;;
  seed) engine cont "$OUT" seed ;;
  check) engine cont "$OUT" check ;;
  xenv) xenv ;;
  gate) gate ;;
  run) run_queue ;;
  supplement) supplement ;;
  chain) chain ;;
  *) echo "사용: $0 status|unpack|remaining|seed|check|xenv|gate|run|supplement|chain"; exit 2 ;;
esac
