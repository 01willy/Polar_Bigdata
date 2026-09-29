#!/bin/bash
# LGX(H41·H42) Rescale 작업 공통 함수. run_lgx.sh 와 run_lgx_smoke.sh 가 lg_common.sh 다음에 source 한다. 단독으로 실행하지 않는다.
# lg_common.sh 는 본 실행(LG, 작업 ZovWo)의 기록이므로 고치지 않는다. LGX 에만 필요한 함수를 이 파일에 둔다.
# 이 파일은 Rescale 노드에서만 쓴다. 로컬 공유 서버에서는 실행하지 않는다(사용자 지시 2026-09-29).
#
# 제공 함수
#   lgx_check_inputs    LG 입력(lg_check_inputs)에 더해 h41, h42, 고정 입력 표 3종을 확인한다(없으면 1)
#   lgx_install_deps    필수 패키지를 사전 점검(qOkSo)과 같은 판으로 작업 디렉터리(.pydeps)에 설치한다. 실패하면 lg_install_deps 로 넘긴다
#   lgx_versions_check  설치된 판이 고정 판과 같은지 출력한다(다르면 1. 실행은 막지 않는다)
#   lgx_restore         작업 디렉터리의 입력 묶음(results_lg*.tar.gz, results_lgx*.tar.gz)에서 조각을 복원한다
#   lgx_gate_tag        재현 점검에 쓸 기준 조각의 tag 를 출력한다(본 실행 조각 lg, 없으면 lg_precheck, 둘 다 없으면 빈 문자열)
#   lgx_pack            결과 묶음 작성(임시 파일에 쓴 뒤 교체. 조각 포함 여부를 고를 수 있다)
#   lgx_count_units     조각 완료 표지 수(부분별)

LGX_H42=scripts/3_deep_learning/h42_label_grid_ext.py
LGX_H41=scripts/2_evaluation/h41_validation_ladder.py
LGX_DIR=data/processed/lgx
LGX_LADDER_DIR=data/processed/lgx/ladder
LGX_LG_DIR=data/processed/lg
LGX_TABLES=(data/processed/lgx_label_flags_v1.csv data/processed/lgx_tdd_matched_v1.csv data/processed/lgx_cluster_map_v1.csv)
# 사전 점검(qOkSo, 2026-09-29)에서 설치된 판. 본 실행(ZovWo)은 같은 날 같은 설치 절차로 제출했다(본 실행의 판은 결과 회수 뒤 env 로그로 확인한다).
LGX_PIN_SPECS=("catboost==1.2.10" "scikit-learn==1.9.1" "pandas==2.3.3" "scipy==1.17.1")
LGX_PIN_NUMPY="2.4.6"

lgx_check_inputs() {
  local f miss=0
  lg_check_inputs || miss=1
  for f in "$LGX_H42" "$LGX_H41" src/polar/physics.py src/polar/fidelity.py src/polar/m1_ext.py src/polar/m1_stats.py \
           scripts/rescale/lgx_timing_table.py scripts/rescale/lgx_gate_check.py; do
    [ -f "$f" ] || { lg_log "[input] 없음: $f"; miss=1; }
  done
  # 고정 입력 표. 없으면 h42 의 x9·x9f 축과 h41 의 V-C·V-G 단이 실행 전에 중단하므로 여기서 먼저 멈춘다(비용이 들기 전).
  for f in "${LGX_TABLES[@]}"; do
    [ -s "$f" ] || { lg_log "[input] 없음: $f (로컬 선행 작업으로 만든다: bash scripts/rescale/make_payload_lgx.sh --make-tables)"; miss=1; }
  done
  return $miss
}

# 고정 판 설치. 이미지의 기본 설치 위치는 용량이 없으므로(qOkSo 의 deps 로그) 처음부터 작업 디렉터리(.pydeps)에 설치한다.
# 필수 패키지가 이미 모두 import 되면 설치하지 않는다. 고정 판 설치가 실패하면 lg_install_deps(범위 지정, 세 방식)로 넘긴다.
lgx_install_deps() {
  local tgt="$PWD/.pydeps" cons=".lgx_pip_constraints.txt" need
  mapfile -t need < <(_lg_missing "catboost:x" "sklearn:x" "pandas:x" "scipy:x")
  if [ ${#need[@]} -gt 0 ]; then
    python3 - "$LGX_PIN_NUMPY" > "$cons" <<'PY'
import importlib.metadata as md
import sys
try:
    print(f"torch=={md.version('torch')}")
except md.PackageNotFoundError:
    pass
print(f"numpy=={sys.argv[1]}")
PY
    mkdir -p "$tgt"
    lg_log "[deps] 고정 판 설치(--target $tgt): ${LGX_PIN_SPECS[*]} · numpy==${LGX_PIN_NUMPY}"
    if python3 -m pip install --quiet --target "$tgt" -c "$cons" "${LGX_PIN_SPECS[@]}"; then
      case ":${PYTHONPATH:-}:" in *":$tgt:"*) ;; *) export PYTHONPATH="$tgt${PYTHONPATH:+:$PYTHONPATH}" ;; esac
    else
      lg_log "[deps] 고정 판 설치 실패. 범위 지정 설치(lg_install_deps)로 넘긴다. 설치된 판이 사전 점검과 다를 수 있다"
    fi
  fi
  lg_install_deps
}

# 설치된 판과 고정 판의 대조. 다르면 1 을 돌려준다(재현 점검의 허용 차를 넘을 수 있다는 표지. 실행은 계속한다).
lgx_versions_check() {
  python3 - "$LGX_PIN_NUMPY" "${LGX_PIN_SPECS[@]}" <<'PY'
import importlib
import sys

want = {"numpy": sys.argv[1]}
for spec in sys.argv[2:]:
    name, ver = spec.split("==")
    want[name] = ver
mod = {"scikit-learn": "sklearn"}
bad = []
for name, ver in want.items():
    try:
        got = importlib.import_module(mod.get(name, name)).__version__
    except Exception as e:                                                # noqa: BLE001
        got = f"import 실패({e!r})"
    flag = "같다" if got == ver else "다르다"
    print(f"[deps] 판 대조: {name} 설치 {got} · 고정 {ver} · {flag}")
    if got != ver:
        bad.append(name)
sys.exit(1 if bad else 0)
PY
}

# 입력 묶음 복원. 묶음의 경로는 저장소 기준이다. 조각 디렉터리와 catboost_tuned 선택 표만 푼다(이전 작업의 집계 표는 풀지 않는다).
#   results_lg_prev.tar.gz, results_lg.tar.gz   본 실행(LG) 결과. data/processed/lg/shards 만 푼다(읽기 전용 참조)
#   results_lgx*.tar.gz                          확장(LGX) 결과. 이름 순서로 푼다(뒤의 묶음이 같은 조각을 덮어쓴다). 이름에 smoke 가 있으면 뺀다
lgx_restore() {
  local f n0 n1 any=0
  for f in results_lg_prev.tar.gz results_lg.tar.gz; do
    [ -f "$f" ] || continue
    n0=$(find "$LGX_LG_DIR/shards" -name 'lg__*_unit.json' 2>/dev/null | wc -l)
    tar xzf "$f" --wildcards "$LGX_LG_DIR/shards/*" 2>/dev/null
    n1=$(find "$LGX_LG_DIR/shards" -name 'lg__*_unit.json' 2>/dev/null | wc -l)
    lg_log "[restore] 본 실행 묶음 $f · 본 실행 조각 ${n0} → ${n1}개"
    any=1
  done
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    case "$f" in *smoke*) lg_log "[restore] 건너뜀(스모크 묶음): $f"; continue ;; esac
    n0=$(lgx_count_units all)
    tar xzf "$f" --wildcards "$LGX_DIR/shards/*" "$LGX_LADDER_DIR/shards/*" "$LGX_DIR/*_tuned_select.csv" 2>/dev/null
    n1=$(lgx_count_units all)
    lg_log "[restore] 확장 묶음 $f · 확장 조각 ${n0} → ${n1}개"
    any=1
  done < <(ls results_lgx*.tar.gz 2>/dev/null | LC_ALL=C sort)
  [ "$any" -eq 1 ] || lg_log "[restore] 입력 묶음 없음. 처음부터 실행한다"
  return 0
}

lgx_gate_tag() {
  if ls "$LGX_LG_DIR"/shards/lg__cpu__*_unit.json >/dev/null 2>&1; then
    echo lg
  elif ls "$LGX_LG_DIR"/shards/lg_precheck__cpu__*_unit.json >/dev/null 2>&1; then
    echo lg_precheck
  else
    echo ""
  fi
}

# 조각 완료 표지 수. 인자: cpu | gpu | h41 | all
lgx_count_units() {
  local c=0 g=0 v=0
  c=$(find "$LGX_DIR/shards" -maxdepth 1 -name 'lgx*__cpu__*_unit.json' 2>/dev/null | wc -l)
  g=$(find "$LGX_DIR/shards" -maxdepth 1 -name 'lgx*__gpu__*_unit.json' 2>/dev/null | wc -l)
  v=$(find "$LGX_LADDER_DIR/shards" -maxdepth 1 -name 'lgv*__*_unit.json' 2>/dev/null | wc -l)
  case "${1:-all}" in
    cpu) echo "$c" ;; gpu) echo "$g" ;; h41) echo "$v" ;; *) echo $(( c + g + v )) ;;
  esac
}

# 결과 묶음. 인자: 묶음 파일 이름, 조각 포함 여부(1 | 0), 묶을 경로들. 없는 경로는 뺀다.
# 조각은 임시 파일에 쓴 뒤 교체되므로 *.tmp* 는 넣지 않는다. 실행 중에 묶으면 tar 가 1(파일 변경 경고)을 돌려줄 수 있다. 1 까지는 정상으로 본다.
lgx_pack() {
  local out="$1" with_shards="$2"; shift 2
  local tmp="${out}.part" p have=() ex=() rc
  for p in "$@"; do [ -e "$p" ] && have+=("$p"); done
  [ ${#have[@]} -gt 0 ] || { lg_log "[pack] 묶을 경로가 없다"; return 1; }
  [ "$with_shards" = "1" ] || ex=(--exclude="$LGX_DIR/shards" --exclude="$LGX_LADDER_DIR/shards")
  tar czf "$tmp" --warning=no-file-changed --warning=no-file-removed --exclude='*.tmp*' --exclude='__pycache__' ${ex[@]+"${ex[@]}"} "${have[@]}" 2>/dev/null
  rc=$?
  if [ $rc -le 1 ] && gzip -t "$tmp" 2>/dev/null; then
    mv -f "$tmp" "$out"
    lg_log "[pack] $out $(du -h "$out" | cut -f1) · 파일 $(tar tzf "$out" | grep -vc '/$')개 · 조각 포함 ${with_shards}"
    return 0
  fi
  rm -f "$tmp"
  lg_log "[pack] 실패(tar 종료 코드 $rc)"
  return 1
}
