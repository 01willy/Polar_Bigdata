#!/usr/bin/env bash
# 결과 뒤 처리 파이프라인(회수, 해제, 집계, 지도, 그림). 판정 기록은 하지 않는다(메인 세션이 5.3 순서로 표를 열고 기록한다).
# 근거: docs/EXECUTION_PLAN_REMAINING_2026-09-30.md 5절(단계 표), 3.3(C0–C10), 5.3(열람 순서). 인자는 각 하네스의 argparse 와 대조했다.
#
# 원칙
#   - 판정 표·곡선 표의 내용을 화면에 출력하지 않는다. 하네스 출력은 모두 logs/post/<단계>_<시각>.log 에만 쓰고, 화면에는 단계 이름,
#     종료 코드, 로그 경로, 만들어지거나 바뀐 파일 이름만 쓴다. 로그를 여는 것은 그 표를 여는 것과 같게 다룬다(5.3).
#   - 모든 계산은 nice 10, GPU 없음(CUDA_VISIBLE_DEVICES=''). 우리 작업 합계 32스레드 이하를 지키도록 단계별 스레드를 고정한다.
#   - 무거운 집계(sum-lgx, sum-lgx-aux)는 잠금(logs/post/heavy.lock)으로 한 번에 하나만 돈다. 가용 RAM 32 GB 미만이거나 1분 load
#     average 64 초과면 시작하지 않는다. 최대 RSS 를 /usr/bin/time 으로 로그에 남긴다.
#   - 등록 확인: 결과 표를 만드는 단계는 해당 계획서 개정(LG 개정 15, WRAPUP 개정 2, LGU 개정 4, LGF 개정 6)이 커밋된 뒤에만 돈다.
#   - 표 생성은 WRAPUP 0.3 의 T_res (2)에 해당한다. 단계마다 생성 시각을 logs/post/events.tsv 에 남긴다.
#   - $LGD = results/rescale_lg/data/processed/lg(ZovWo 조각, 읽기 전용). $LGS = LG 표 폴더(작업 안 집계가 성공하면 $LGD,
#     실패해 로컬에서 다시 집계하면 data/processed/lg_sum_local). gate-lg 가 logs/post/lgs_dir 에 적는다.
#
# 하위 명령(순서는 5.2 와 같다)
#   status        각 단계의 입력·산출 파일 존재와 등록 상태만 출력한다
#   fetch-lg      (C0) ZovWo 결과 회수·해제, 조각 수(CPU 117, GPU ≤ 231), 빠진 RealMLP 단위 이름, 읽기 전용, sha256 목록
#   unpack-lgx    (C0b) scripts/local/run_lgx_local_continue.sh unpack 을 부른다
#   gate-lg       (C1) 작업 안 h40 집계의 종료 코드와 표 존재를 확인하고 $LGS 를 정한다. 실패면 sum-lg 를 부른다
#   sum-lg        (C1 대체) h40 --summarize-only(재표집 1,000, LG 규약)를 data/processed/lg_sum_local 에서 돈다
#   ladder        (C3) h41 검증 사다리 집계(L19–L22)
#   sum-lgx       (C4) h42 LGX 주 통합 집계(Rescale 조각만, 재표집 10,000). 무거운 집계
#   gate-lgd      (C2) h51 재현 점검(repro, 러시아 W x 분할 1). 결과 표의 scope·status 열만 출력한다
#   lgd-v3local   (C2 조건부) h51 --specs v3local(재현 점검이 base 범위 CatBoost 불통과일 때만)
#   lgd-lic       (C2b 조건부) 약관 확인분 실행 표(h51 묶음 lic, LG 개정 15 (m)). h51 에 묶음이 구현된 뒤에만 돈다
#   pool-lgd      (C6) h52 LGD 확장 풀 집계(L38–L42, PE1·PE2)
#   sum-lgt       (C5) h43 LGT 집계와 교차 비교(--cross-lg). LGT 가 끝난 뒤
#   lgu-ab10      (A3) AB10 입력 표(scripts/2_evaluation/lgu_ab10_tests.py). LGU 개정 4 커밋 뒤
#   scenarios     (C7) h39 summarize 와 bias-mae. LGF 표가 있으면 --lgf-dir 를 더한다(재실행 시 δ_rel)
#   map           (C8) h49_map_contrasts → h49_transfer_map(--check-inputs 먼저) → fig_map_lena
#   sum-lgf       (C9) h47·h48 집계(LGF 감시기와 역할이 모두 끝난 뒤)
#   sum-lgx-aux   (C10) LGX 보조 집계(Rescale 조각 + 로컬 새 조각, data/processed/lgx_aux). 무거운 집계
#   figs          (C11) fig_map_lena 와 POST_FIGS 에 적은 그림 모듈(예: POST_FIGS="fig2 fig3"). 모듈 재작성(F3) 전에는 지도만
#   cpu-chain     gate-lg → ladder → sum-lgx → gate-lgd → pool-lgd → lgu-ab10 순으로 돌고 첫 실패에서 멈춘다(회수 뒤 한 번에)
#
# 환경 변수
#   POST_LG_JOB           ZovWo(기본). Rescale 작업 id
#   POST_MAP_PROVISIONAL  1 이면 map 에 --allow-missing-gpu-tables(LGT·LGF 표 전 잠정판, 산출에 잠정 표지)
#   POST_SIGMA_FILE       LGU-B2 가 '전이'일 때 격자 σ npz(LGU 개정 4 의 조건부 규약으로 만든 파일)
#   POST_AUX_FULL         1 이면 sum-lgx-aux 가 Rescale 조각 전부를 모은다(기본은 L26·nn 에 필요한 축 lgxb·lgxg·lgxnn 만)
#   POST_SKIP_PREREG      1 이면 등록 확인을 건너뛴다(쓰지 않는다)
set -u -o pipefail
SELF=$(readlink -f "$0")
cd /home/willy010313/Polar_Bigdata || exit 2
[ "${POST_NICED:-0}" = 1 ] || exec env POST_NICED=1 nice -n 10 "$SELF" "$@"
export CUDA_VISIBLE_DEVICES=""

PY=${PY:-python3}
PYF=.venv_lgf/bin/python
LGD=results/rescale_lg/data/processed/lg
LGR=results/rescale_lg
LGX=data/processed/lgx
LGXL=data/processed/lgx_local
AUX=data/processed/lgx_aux
LGW=data/processed/lgw
PL=logs/post; mkdir -p "$PL"
EV=$PL/events.tsv
JOB=${POST_LG_JOB:-ZovWo}
TS=$(date +%Y%m%dT%H%M%S)
LGS=$(cat "$PL/lgs_dir" 2>/dev/null || echo "$LGD")

say() { echo "[$(date '+%F %T')] $*"; }
die() { say "[중단] $1"; exit "${2:-1}"; }
ev() { printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" "$3" >> "$EV"; }
load1() { cut -d' ' -f1 /proc/loadavg; }
avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
reg_ok() {                                       # reg_ok <계획서> <개정 표지>: 개정이 있고 커밋되었다
  [ "${POST_SKIP_PREREG:-0}" = 1 ] && return 0
  grep -q "$2" "$1" 2>/dev/null && git diff --quiet HEAD -- "$1" 2>/dev/null
}
need() {                                         # need <표지 목록...>: A1 LG15, A4 W2, A3 U4, A2 F6
  local k
  for k in "$@"; do
    case "$k" in
      LG15) reg_ok docs/EXPERIMENT_PLAN_LG_2026-09-29.md '개정 15(' || die "LG 개정 15 가 커밋되지 않았다(결과 표를 만들기 전 등록)" 2 ;;
      W2)   reg_ok docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md '개정 2(' || die "WRAPUP 개정 2 가 커밋되지 않았다" 2 ;;
      U4)   reg_ok docs/EXPERIMENT_PLAN_LGU_2026-09-29.md '개정 4(' || die "LGU 개정 4 가 커밋되지 않았다(LGU 판정 표를 열기 전 등록)" 2 ;;
      F6)   reg_ok docs/EXPERIMENT_PLAN_LGF_2026-09-29.md '개정 6(' || die "LGF 개정 6 이 커밋되지 않았다" 2 ;;
    esac
  done
}
need_file() { local f; for f in "$@"; do [ -e "$f" ] || die "입력 없음: $f"; done; }
resources() {                                    # resources <가용 RAM 하한 GB>
  local a; a=$(avail_gb)
  [ "$a" -ge "$1" ] || die "가용 RAM ${a} GB < ${1} GB. 다른 무거운 작업이 끝난 뒤 한다"
  awk -v l="$(load1)" 'BEGIN{exit !(l <= 64)}' || die "1분 load average $(load1) > 64. 뒤로 미룬다"
}
newer() { find "$@" -maxdepth 1 -type f -newer "$PL/.mark_$TS" 2>/dev/null | sed 's#^\./##' | sort | tr '\n' ' '; }
run_step() {                                     # run_step <단계> <산출 폴더(공백 구분)> -- <명령...>. 출력은 로그에만
  local step=$1 outs=$2; shift 3
  local lg=$PL/${step}_${TS}.log rc
  touch "$PL/.mark_$TS"
  { echo "# $(date '+%F %T') $step"; echo "# 명령: $*"; echo "# 판 목록: python $("$PY" -c 'import platform;print(platform.python_version())'), host $(hostname), nice $(nice), 스레드 제한 OMP=${OMP_NUM_THREADS:-}"; } > "$lg"
  /usr/bin/time -v "$@" >> "$lg" 2>&1; rc=$?
  local rss; rss=$(grep -a "Maximum resident set size" "$lg" | tail -1 | awk '{print $NF}')
  local made; made=$(newer $outs)
  ev "$step" "$rc" "log=$lg maxrss_kb=${rss:-NA} files=${made:-없음}"
  say "$step: 종료 코드 $rc · 최대 RSS ${rss:-NA} kB · 로그 $lg(표 내용이 들어 있을 수 있다. 5.3 순서로 연다)"
  say "$step: 새로 만들어지거나 바뀐 파일: ${made:-없음}"
  rm -f "$PL/.mark_$TS"
  return "$rc"
}
heavy() {                                        # heavy <단계> <산출> -- <명령>: 잠금과 자원 확인
  exec 9> "$PL/heavy.lock"
  flock -n 9 || die "다른 무거운 집계가 돌고 있다(logs/post/heavy.lock)"
  resources 32
  run_step "$@"
}
first_read() {                                   # first_read <조각 상위 폴더> <단계>: WRAPUP 0.3 (3) 의 조각 내용 첫 열람 시각(없을 때만 적는다)
  [ -d "$1" ] || return 0
  [ -f "$1/.first_content_read.txt" ] || { echo "first_content_read=$(date '+%F %T') by=$2" > "$1/.first_content_read.txt"; ev first-read 0 "$1 by $2"; }
}
th() { export OMP_NUM_THREADS=$1 MKL_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 NUMEXPR_NUM_THREADS=$1; }

# ------------------------------------------------------------------ 단계
fetch_lg() {
  mkdir -p "$LGR"
  if [ ! -f "$LGR/results_lg.tar.gz" ]; then
    say "fetch-lg: Rescale 작업 $JOB 회수(내려받기만, 새 작업 없음)"
    # 출력 파일이 많으면 전체 목록 조회가 시간 초과된다(09-30 Qjpbeb). 필요한 파일만 이름으로 받는다
    timeout 1800 "$PY" tools/rescale_fetch_selected.py "$JOB" "$LGR" results_lg.tar.gz > "$PL/fetch_lg_$TS.log" 2>&1 || die "묶음 회수 실패(로그 $PL/fetch_lg_$TS.log)"
    timeout 600 "$PY" tools/rescale_fetch_selected.py "$JOB" "$LGR" lg_status.csv process_output.log lg_payload_info.txt lg_timing_by_learner.csv >> "$PL/fetch_lg_$TS.log" 2>&1 || say "fetch-lg: 부가 파일 일부를 받지 못했다(로그 $PL/fetch_lg_$TS.log)"
    gzip -t "$LGR/results_lg.tar.gz" || die "묶음 무결성 실패"
    ev fetch-lg 0 "fetch $JOB → $LGR"
  fi
  need_file "$LGR/results_lg.tar.gz"
  if [ ! -f "$LGR/.unpacked.txt" ]; then
    local t0; t0=$(date '+%F %T')
    tar xzf "$LGR/results_lg.tar.gz" -C "$LGR" || die "해제 실패"
    local nc ng; nc=$(find "$LGD/shards" -name 'lg__cpu__*_unit.json' | wc -l); ng=$(find "$LGD/shards" -name 'lg__gpu__*_unit.json' | wc -l)
    find "$LGD/shards" -type f -print0 | sort -z | xargs -0 sha256sum > "$LGR/lg_shards_sha256.txt"
    chmod -R a-w "$LGD/shards"
    { echo "unpacked_at=$t0"; echo "tar_sha256=$(sha256sum "$LGR/results_lg.tar.gz" | cut -c1-64)"; echo "cpu=$nc gpu=$ng";
      echo "note=해제는 결과 열람이 아니다. 작업 안 집계 표(lg_curve 등)는 열지 않았다"; } > "$LGR/.unpacked.txt"
    ev fetch-lg 0 "unpacked cpu=$nc gpu=$ng"
  fi
  local nc ng t s miss=""
  nc=$(find "$LGD/shards" -name 'lg__cpu__*_unit.json' | wc -l); ng=$(find "$LGD/shards" -name 'lg__gpu__*_unit.json' | wc -l)
  for f in "$LGD"/shards/lg__gpu__*__ftt_unit.json; do                    # 빠진 RealMLP 단위 = FT-T 가 있는 (대상, 모드, 분할) 가운데 RealMLP 가 없는 것
    [ -e "$f" ] || continue
    s=$(basename "$f" __ftt_unit.json)
    [ -e "$LGD/shards/${s}__realmlp_unit.json" ] || miss="$miss ${s#lg__gpu__}"
  done
  say "fetch-lg: CPU 조각 $nc(기대 117) · GPU 조각 $ng(기대 231) · 빠진 RealMLP:${miss:- 없음}"
  if [ -f "$LGR/lg_status.csv" ]; then say "fetch-lg: 작업 단계 종료 코드: $(awk -F, 'NR>1{printf "%s=%s ", $1, $2}' "$LGR/lg_status.csv")"; fi
  echo "${miss# }" > "$LGR/missing_realmlp.txt"
}

gate_lg() {
  need_file "$LGD/shards"
  local rc_s ok=1 f
  rc_s=$(awk -F, '$1=="summarize"{print $2}' "$LGR/lg_status.csv" 2>/dev/null | tail -1)
  for f in lg_curve.csv lg_minn.csv lg_tests.csv lg_targets.csv; do [ -f "$LGD/$f" ] || ok=0; done
  if [ "${rc_s:-x}" = 0 ] && [ "$ok" = 1 ]; then
    echo "$LGD" > "$PL/lgs_dir"; LGS=$LGD
    say "gate-lg: 작업 안 집계 종료 코드 0, 표 4개 있음. \$LGS = $LGD"; ev gate-lg 0 "LGS=$LGD"
  else
    say "gate-lg: 작업 안 집계 종료 코드 '${rc_s:-없음}', 표 $( [ "$ok" = 1 ] && echo 있음 || echo 일부 없음). 로컬 재집계(sum-lg)로 넘어간다"
    sum_lg || return $?
  fi
}

sum_lg() {
  need LG15 W2
  need_file "$LGD/shards"
  local d=data/processed/lg_sum_local
  mkdir -p "$d"; [ -e "$d/shards" ] || ln -s "$(readlink -f "$LGD/shards")" "$d/shards"
  th 4; resources 16
  first_read "$LGD" sum-lg
  run_step sum-lg "$d" -- "$PY" -u scripts/3_deep_learning/h40_label_grid.py --summarize-only --allow-local --threads 4 --workers 1 --out-dir "$d" || return $?
  echo "$d" > "$PL/lgs_dir"; LGS=$d
  say "sum-lg: \$LGS = $d(작업 안 집계 실패 뒤 로컬 재집계. LG 개정 15 (g) 에 사유를 적는다)"
}

ladder() {
  need LG15 W2
  need_file "$LGX/.unpacked_rescale.txt"
  th 4; resources 16
  first_read "$LGX" ladder
  run_step ladder "$LGX/ladder" -- "$PY" -u scripts/2_evaluation/h41_validation_ladder.py --summarize-only --allow-local --threads 4 --workers 1
}

sum_lgx() {
  need LG15 W2
  need_file "$LGX/.unpacked_rescale.txt" "$LGX/rescale_shards_sha256.txt" "$LGS"
  sha256sum -c --quiet "$LGX/rescale_shards_sha256.txt" > "$PL/sha_check_$TS.log" 2>&1 || die "Rescale 조각의 sha256 가 해제 때와 다르다(로그 $PL/sha_check_$TS.log)"
  th 4
  first_read "$LGX" sum-lgx; first_read "$LGD" sum-lgx
  heavy sum-lgx "$LGX" -- "$PY" -u scripts/3_deep_learning/h42_label_grid_ext.py --summarize-only --allow-local --nboot 10000 --threads 4 --workers 2 \
    --lg-dir "$LGS" --out-dir "$LGX"
}

gate_lgd() {
  need LG15 W2
  need_file "$LGD/shards"
  th 4; resources 16
  first_read "$LGD" gate-lgd
  run_step gate-lgd data/processed/lgd -- "$PY" -u scripts/3_deep_learning/h51_lgd_run.py --specs repro --allow-local --workers 0 --threads 4 --repro-check \
    --lg-shards "$LGD/shards" || return $?
  [ -f data/processed/lgd/lgd_repro_gate.csv ] && say "gate-lgd: 재현 점검 상태: $("$PY" -c 'import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); print("; ".join(f"{r.scope}={r.status}" for r in d.itertuples()))' data/processed/lgd/lgd_repro_gate.csv)"
}

lgd_v3local() {
  need LG15 W2
  th 4; resources 16
  run_step lgd-v3local data/processed/lgd -- "$PY" -u scripts/3_deep_learning/h51_lgd_run.py --specs v3local --allow-local --workers 2 --threads 4 --resume
}

lgd_lic() {
  need LG15 W2
  grep -q '"lic"' scripts/3_deep_learning/h51_lgd_run.py || die "h51 에 약관 확인분 묶음(lic)이 없다. LG 개정 15 (m) 의 구현(A6) 뒤에 돈다" 2
  th 4; resources 16
  run_step lgd-lic data/processed/lgd -- "$PY" -u scripts/3_deep_learning/h51_lgd_run.py --specs lic --allow-local --workers 2 --threads 4 --resume
}

pool_lgd() {
  need LG15 W2
  need_file data/processed/lgd/lgd_repro_gate.csv "$LGX/lgx_splitdist.csv" "$LGD/shards"
  th 4; resources 16
  run_step pool-lgd data/processed/lgd -- "$PY" -u scripts/2_evaluation/h52_lgd_pool.py --allow-local --threads 4 --lg-dir "$LGD" \
    --splitdist "$LGX/lgx_splitdist.csv"
}

sum_lgt() {
  need LG15 W2
  need_file data/processed/lgt/lgt_run_status.json "$LGS"
  local p; p=$(grep -o '"pid": *[0-9]*' data/processed/lgt/run_lgt/lock.json 2>/dev/null | grep -o '[0-9]*$')
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then die "LGT 학습이 아직 돈다(pid $p)"; fi
  th 2; resources 16
  first_read "$LGD" sum-lgt
  run_step sum-lgt data/processed/lgt -- "$PY" -u scripts/3_deep_learning/h43_tabpfn_label_grid.py --summarize-only --cross-lg --lg-dir "$LGS" --threads 2
  say "sum-lgt: LGF 가 도는 중에 LGT 표를 열면 logs/lgf/viewing_state.txt 와 LGF 개정 이력의 열람 상태 칸을 고친다(5.3 (5))"
}

lgu_ab10() {
  need U4
  need_file data/processed/lgu/lgu_a_tests.csv
  th 1
  run_step lgu-ab10 data/processed/lgu -- "$PY" -u scripts/2_evaluation/lgu_ab10_tests.py
}

scenarios() {
  need LG15 W2 U4
  need_file "$LGX/lgx_tests.csv" "$LGS/lg_tests.csv" data/processed/lgu/lgu_a_ab10_tests.csv "$LGD/shards"
  [ -f data/processed/lgd/lgd_tests.csv ] || say "scenarios: 경고 lgd_tests.csv 없음(pool-lgd 전). LGD 행은 비어 나온다"
  local lgf=()
  if ls data/processed/lgf/lgf_tests.csv data/processed/lgf/lgfn_tests.csv > /dev/null 2>&1; then lgf=(--lgf-dir data/processed/lgf); fi
  local common=(--allow-local --threads 4 --lg-shards "$LGD/shards" --lg-dir "$LGS" --lgx-dir "$LGX" --lgx-shards "$LGX/shards"
                --lgd-dir data/processed/lgd --lgu-dir data/processed/lgu --lgt-dir data/processed/lgt ${lgf[@]+"${lgf[@]}"} --out-dir "$LGW")
  th 4; resources 24
  run_step scenarios "$LGW" -- "$PY" -u scripts/2_evaluation/h39_scenarios.py --mode summarize "${common[@]}" || return $?
  run_step scenarios-bias "$LGW" -- "$PY" -u scripts/2_evaluation/h39_scenarios.py --mode bias-mae "${common[@]}"
  say "scenarios: LGF 표 $( [ ${#lgf[@]} -gt 0 ] && echo '포함(δ_rel 채움)' || echo '없음(LGF 집계 뒤 다시 돈다)')"
}

map_step() {
  need LG15 W2 U4
  need_file "$LGD/shards" "$LGX/shards" data/processed/map_lena/lena_grid_x25_v1.csv.gz
  th 4; resources 24
  run_step map-contrasts "$LGW" -- "$PY" -u scripts/2_evaluation/h49_map_contrasts.py --allow-run --allow-local --threads 4 --lg-shards "$LGD/shards" \
    --lgx-shards "$LGX/shards" --out-dir "$LGW" || return $?
  local extra=()
  [ "${POST_MAP_PROVISIONAL:-0}" = 1 ] && extra+=(--allow-missing-gpu-tables)
  [ -n "${POST_SIGMA_FILE:-}" ] && extra+=(--sigma-file "$POST_SIGMA_FILE")
  local common=(--lgw-dir "$LGW" --lg-dir "$LGS" --lgx-dir "$LGX" --lgu-dir data/processed/lgu --lgt-dir data/processed/lgt --lgf-dir data/processed/lgf)
  th 2
  run_step map-check data/processed/map_lena -- "$PY" -u scripts/2_evaluation/h49_transfer_map.py --check-inputs "${common[@]}" ${extra[@]+"${extra[@]}"} || return $?
  run_step map-predict data/processed/map_lena -- "$PY" -u scripts/2_evaluation/h49_transfer_map.py --allow-run --threads 2 "${common[@]}" ${extra[@]+"${extra[@]}"} || return $?
  run_step map-figure outputs/figures/paper -- "$PY" -u scripts/4_visualization/paper/fig_map_lena.py
  [ "${POST_MAP_PROVISIONAL:-0}" = 1 ] && say "map: 잠정판(LGT·LGF 표 없음 허용). 최종판은 sum-lgt·sum-lgf 뒤 다시 돈다"
  say "map: 시각 검토(F2)는 PDF·PNG 를 열어 체크리스트로 한다"
}

sum_lgf() {
  need F6
  local r
  for r in single F N T3; do
    [ -f "logs/lgf/sup/$r.pid" ] && kill -0 "$(cat "logs/lgf/sup/$r.pid")" 2>/dev/null && die "LGF 역할 $r 가 아직 돈다(창 마감 또는 J1–J11 완료 뒤에 집계한다)"
  done
  need_file "$PYF" "$LGS"
  th 2; resources 16
  run_step sum-lgf data/processed/lgf -- "$PYF" -u scripts/3_deep_learning/h47_foundation_models.py --summarize-only --threads 2 || return $?
  run_step sum-lgfn data/processed/lgf -- "$PYF" -u scripts/3_deep_learning/h48_nn_tuning.py --summarize-only --cross-lg --lg-dir "$LGS" --threads 2
  say "sum-lgf: 이어서 scenarios(δ_rel)와 map(최종판)을 다시 돈다"
}

sum_lgx_aux() {
  need LG15 W2
  need_file "$LGX/.unpacked_rescale.txt" "$LGXL/platform_manifest.csv" "$LGS"
  local lock=$LGXL/run_lgx_local/lock
  if [ -f "$lock" ] && kill -0 "$(cat "$lock")" 2>/dev/null; then say "sum-lgx-aux: 경고 로컬 이어 실행기가 아직 돈다(지금까지의 조각으로 집계한다)"; fi
  rm -rf "$AUX/shards"; mkdir -p "$AUX/shards"
  local f b p n_r=0 n_l=0
  local -a pats
  if [ "${POST_AUX_FULL:-0}" = 1 ]; then pats=('lgx*'); else pats=('lgx[bg]__*' 'lgxnn__*'); fi
  for p in "${pats[@]}"; do
    for f in "$LGX"/shards/$p; do [ -e "$f" ] || continue; ln "$f" "$AUX/shards/" && n_r=$((n_r + 1)); done
  done
  while IFS=, read -r uj plat _; do              # 로컬 새 조각: Rescale 에 같은 이름이 없는 것만
    [ "$plat" = local ] || continue
    b=${uj%_unit.json}
    ls "$LGX"/shards/"${b}"_* > /dev/null 2>&1 && continue
    for f in "$LGXL"/shards/"${b}"_*; do [ -e "$f" ] && ln "$f" "$AUX/shards/" && n_l=$((n_l + 1)); done
  done < <(tail -n +2 "$LGXL/platform_manifest.csv")
  { echo "built_at=$(date '+%F %T')"; echo "rescale_files=$n_r local_files=$n_l subset=$([ "${POST_AUX_FULL:-0}" = 1 ] && echo full || echo 'lgxb,lgxg,lgxnn')";
    echo "note=교차 환경(보조). 이 폴더에서는 L26[ftt] 행과 nn·g45 FT-T 곡선 행만 읽는다"; } > "$AUX/aux_manifest.txt"
  say "sum-lgx-aux: 조각 파일 Rescale $n_r · 로컬 $n_l(하드링크)"
  th 4
  heavy sum-lgx-aux "$AUX" -- "$PY" -u scripts/3_deep_learning/h42_label_grid_ext.py --summarize-only --allow-local --nboot 10000 --threads 4 --workers 2 \
    --lg-dir "$LGS" --out-dir "$AUX" \
    || { say "sum-lgx-aux: 부분 폴더 집계 실패. POST_AUX_FULL=1 로 다시 한다(무거운 집계, 한 번에 하나)"; return 1; }
}

figs() {
  th 2
  if [ -f data/processed/map_lena/lena_pred_v1.csv.gz ]; then
    run_step figs-map outputs/figures/paper -- "$PY" -u scripts/4_visualization/paper/fig_map_lena.py || return $?
  fi
  local m
  for m in ${POST_FIGS:-}; do
    need_file "scripts/4_visualization/paper/$m.py"
    run_step "figs-$m" outputs/figures/paper -- "$PY" -u "scripts/4_visualization/paper/$m.py" || return $?
  done
  [ -n "${POST_FIGS:-}" ] || say "figs: 논문 그림 모듈은 figure_spec 갱신(F1)과 재작성(F3) 뒤 POST_FIGS 로 지정한다"
}

status() {
  local k
  echo "[등록] LG 개정 15: $(reg_ok docs/EXPERIMENT_PLAN_LG_2026-09-29.md '개정 15(' && echo 커밋됨 || echo 없음) · WRAPUP 개정 2: $(reg_ok docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md '개정 2(' && echo 커밋됨 || echo 없음)" \
       "· LGU 개정 4: $(reg_ok docs/EXPERIMENT_PLAN_LGU_2026-09-29.md '개정 4(' && echo 커밋됨 || echo 없음) · LGF 개정 6: $(reg_ok docs/EXPERIMENT_PLAN_LGF_2026-09-29.md '개정 6(' && echo 커밋됨 || echo 없음)"
  echo "[LG] 회수 묶음 $([ -f "$LGR/results_lg.tar.gz" ] && echo 있음 || echo 없음) · 해제 $([ -f "$LGR/.unpacked.txt" ] && echo 됨 || echo 안 됨) · \$LGS=$LGS"
  echo "[LGX] 해제 $([ -f "$LGX/.unpacked_rescale.txt" ] && echo 됨 || echo 안 됨) · 로컬 새 조각 $(tail -n +2 "$LGXL/platform_manifest.csv" 2>/dev/null | wc -l)"
  for k in "$LGS/lg_tests.csv" "$LGX/ladder/lgv_tests.csv" "$LGX/lgx_tests.csv" data/processed/lgd/lgd_repro_gate.csv data/processed/lgd/lgd_tests.csv \
           data/processed/lgt/lgt_run_status.json data/processed/lgt/lgt_tests.csv data/processed/lgu/lgu_a_ab10_tests.csv "$LGW/lgw_bundle.csv" \
           data/processed/map_lena/lena_pred_v1.csv.gz data/processed/lgf/lgf_tests.csv data/processed/lgf/lgfn_tests.csv "$AUX/lgx_tests.csv"; do
    printf '  %-48s %s\n' "$k" "$([ -e "$k" ] && stat -c '%y' "$k" | cut -c1-16 || echo 없음)"
  done
  echo "[무거운 집계 잠금] $(flock -n "$PL/heavy.lock" true 2>/dev/null && echo 비어 있음 || echo 사용 중) · 가용 RAM $(avail_gb) GB · load $(load1)"
}

cpu_chain() {
  gate_lg && ladder && sum_lgx && gate_lgd && pool_lgd && lgu_ab10 && say "cpu-chain: 끝. 다음은 sum-lgt(LGT 종료 뒤), scenarios, map"
}

case "${1:-}" in
  status) status ;;
  fetch-lg) fetch_lg ;;
  unpack-lgx) bash scripts/local/run_lgx_local_continue.sh unpack ;;
  gate-lg) gate_lg ;;
  sum-lg) sum_lg ;;
  ladder) ladder ;;
  sum-lgx) sum_lgx ;;
  gate-lgd) gate_lgd ;;
  lgd-v3local) lgd_v3local ;;
  lgd-lic) lgd_lic ;;
  pool-lgd) pool_lgd ;;
  sum-lgt) sum_lgt ;;
  lgu-ab10) lgu_ab10 ;;
  scenarios) scenarios ;;
  map) map_step ;;
  sum-lgf) sum_lgf ;;
  sum-lgx-aux) sum_lgx_aux ;;
  figs) figs ;;
  cpu-chain) cpu_chain ;;
  *) echo "사용: $0 status|fetch-lg|unpack-lgx|gate-lg|sum-lg|ladder|sum-lgx|gate-lgd|lgd-v3local|lgd-lic|pool-lgd|sum-lgt|lgu-ab10|scenarios|map|sum-lgf|sum-lgx-aux|figs|cpu-chain"; exit 2 ;;
esac
