#!/bin/bash
# LG(H40) Rescale 작업 공통 함수. run_lg.sh 와 run_lg_smoke.sh 가 source 한다. 단독으로 실행하지 않는다.
# 이 파일은 Rescale 노드에서만 쓴다. 로컬 공유 서버에서는 실행하지 않는다(사용자 지시 2026-09-29).
#
# 제공 함수
#   lg_log            시각을 붙인 출력
#   lg_ncores         쓸 수 있는 코어 수
#   lg_base_env       파이썬 입출력 인코딩·버퍼링 설정
#   lg_check_inputs   입력 자료와 코드 존재 확인(없으면 1)
#   lg_env_report     python·torch·CUDA·GPU 목록·코어 수·메모리 출력
#   lg_install_deps   필수 패키지 설치(이미 있으면 건너뜀, 실패 시 1)
#   lg_install_opt    선택 패키지 설치(실패해도 0 이 아닌 값을 돌려줄 뿐 중단하지 않는다)
#   lg_gpu_list       보이는 GPU 번호를 쉼표 목록으로 출력
#   lg_pack           결과 묶음 작성(임시 파일에 쓴 뒤 교체)

lg_log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# 쓸 수 있는 코어 수. nproc 은 OMP_NUM_THREADS 가 있으면 그 값을 돌려주므로 그 변수를 빼고 부른다.
lg_ncores() { env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo "${RESCALE_CORES_PER_SLOT:-4}"; }

lg_base_env() {
  # h40 은 한글을 출력하고 UTF-8 파일을 읽는다. 노드 로캘이 POSIX 여도 깨지지 않게 한다.
  export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
  export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
  export MPLBACKEND=Agg
}

lg_check_inputs() {
  local f miss=0
  for f in src/polar/m1_core.py src/polar/tab_models.py src/polar/h4_common.py scripts/3_deep_learning/h40_label_grid.py \
           data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv; do
    [ -f "$f" ] || { lg_log "[input] 없음: $f"; miss=1; }
  done
  if [ ! -f data/processed/lg_subregion_map_v1.csv ]; then
    lg_log "[input] 경고: 하위 지역 대응표(lg_subregion_map_v1.csv)가 없다. h40 이 k-means 로 계산한다(subregion_src = kmeans)"
  fi
  return $miss
}

lg_env_report() {
  lg_log "[env] 작업 디렉터리 $PWD"
  lg_log "[env] 코어 수 $(lg_ncores) (RESCALE_CORES_PER_SLOT = ${RESCALE_CORES_PER_SLOT:-없음}, OMP_NUM_THREADS = ${OMP_NUM_THREADS:-없음}) · 메모리 $(free -g 2>/dev/null | awk '/^Mem:/{print $2" GB (여유 "$7" GB)"}')"
  lg_log "[env] 디스크 $(df -h . 2>/dev/null | awk 'NR==2{print $4" 여유 / "$2}')"
  lg_log "[env] python $(command -v python3) · pip $(command -v pip 2>/dev/null || echo 없음)"
  python3 - <<'PY'
import importlib.metadata as md
import os
import platform
import sys

print(f"[env] python {sys.version.split()[0]} · {platform.platform()}")
print(f"[env] 가상환경 {sys.prefix != getattr(sys, 'base_prefix', sys.prefix)} · prefix {sys.prefix}")
for p in ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "pytabkit", "pytest", "torch"):
    try:
        v = md.version(p)
    except md.PackageNotFoundError:
        v = "없음"
    print(f"[env] {p} {v}")
print(f"[env] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
try:
    import torch
    ok = torch.cuda.is_available()
    print(f"[env] torch {torch.__version__} · CUDA 빌드 {torch.version.cuda} · CUDA 사용 가능 {ok} · GPU 수 {torch.cuda.device_count() if ok else 0}")
    if ok:
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            print(f"[env] GPU {i}: {pr.name} · {pr.total_memory / 2 ** 30:.1f} GiB")
except Exception as e:                                                   # noqa: BLE001
    print(f"[env] torch 확인 실패: {e!r}")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv 2>&1 | sed 's/^/[env] nvidia-smi: /'
  else
    lg_log "[env] nvidia-smi 없음"
  fi
}

# 인자로 받은 "모듈:패키지지정" 가운데 import 되지 않는 것의 패키지 지정만 출력한다.
_lg_missing() {
  local it mod
  for it in "$@"; do
    mod="${it%%:*}"
    python3 -c "import ${mod}" >/dev/null 2>&1 || echo "${it#*:}"
  done
}

# torch 와 numpy 주 버전을 고정하는 제약 파일. pip 이 의존성 해소 과정에서 CUDA 빌드 torch 를 바꾸지 못하게 한다.
_lg_constraints() {
  local out="$1"
  python3 - > "$out" <<'PY'
import importlib.metadata as md
for p in ("torch",):
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        pass
try:
    mj = int(md.version("numpy").split(".")[0])
    print(f"numpy>={mj},<{mj + 1}")
except Exception:                                                        # noqa: BLE001
    pass
PY
}

# pip 설치. --user → 기본 위치(가상환경) → --target 순으로 시도한다. 가상환경에서는 --user 가 거부되므로 둘째 시도가 쓰인다.
_lg_pip() {
  local cons=".lg_pip_constraints.txt" tgt="$PWD/.pydeps"
  [ -s "$cons" ] || _lg_constraints "$cons"
  lg_log "[deps] pip install --user $*"
  python3 -m pip install --user --quiet -c "$cons" "$@" && return 0
  lg_log "[deps] --user 설치 실패(가상환경이면 정상이다). 기본 위치에 다시 시도한다"
  python3 -m pip install --quiet -c "$cons" "$@" && return 0
  lg_log "[deps] 기본 위치 설치 실패. --target $tgt 에 다시 시도한다"
  mkdir -p "$tgt"
  python3 -m pip install --quiet --target "$tgt" -c "$cons" "$@" || return 1
  case ":${PYTHONPATH:-}:" in *":$tgt:"*) ;; *) export PYTHONPATH="$tgt${PYTHONPATH:+:$PYTHONPATH}" ;; esac
  return 0
}

# 필수 패키지. 이미 import 되면 건너뛴다. 설치 뒤에도 import 되지 않으면 1 을 돌려준다.
# pandas 는 3 미만으로 둔다(h40 은 pandas 2.1 에서 작성·확인했다).
lg_install_deps() {
  local items=("catboost:catboost>=1.2" "sklearn:scikit-learn>=1.3" "pandas:pandas>=2.1,<3" "scipy:scipy>=1.11")
  local need tv0 tv1
  tv0=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo 없음)
  mapfile -t need < <(_lg_missing "${items[@]}")
  if [ ${#need[@]} -eq 0 ]; then
    lg_log "[deps] 필수 패키지가 모두 있다. 설치를 건너뛴다"
  else
    lg_log "[deps] 없는 필수 패키지: ${need[*]}"
    _lg_pip "${need[@]}" || { lg_log "[deps] 실패: pip 설치가 세 방식 모두 실패했다"; return 1; }
    mapfile -t need < <(_lg_missing "${items[@]}")
    if [ ${#need[@]} -ne 0 ]; then
      lg_log "[deps] 실패: 설치 뒤에도 import 되지 않는다: ${need[*]}"
      return 1
    fi
  fi
  tv1=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo 없음)
  if [ "$tv0" != "$tv1" ]; then
    lg_log "[deps] 실패: 설치 과정에서 torch 가 바뀌었다($tv0 → $tv1)"
    return 1
  fi
  python3 - <<'PY' || return 1
import importlib.metadata as md
mj = int(md.version("pandas").split(".")[0])
if mj != 2:
    print(f"[deps] 경고: pandas {md.version('pandas')} 이다. h40 은 pandas 2.1 에서 확인했다")
import catboost, sklearn, pandas, scipy, numpy                            # noqa: E401,F401
print(f"[deps] 확인: catboost {catboost.__version__} · scikit-learn {sklearn.__version__} · pandas {pandas.__version__} · "
      f"scipy {scipy.__version__} · numpy {numpy.__version__}")
PY
  return 0
}

# 선택 패키지 하나. 인자: 모듈 이름, pip 지정(여럿이면 앞에서부터 차례로 시도).
lg_install_opt() {
  local mod="$1"; shift
  local spec
  python3 -c "import ${mod}" >/dev/null 2>&1 && { lg_log "[deps] 선택 패키지 ${mod} 있음"; return 0; }
  for spec in "$@"; do
    _lg_pip "$spec" && python3 -c "import ${mod}" >/dev/null 2>&1 && { lg_log "[deps] 선택 패키지 ${mod} 설치함(${spec})"; return 0; }
  done
  lg_log "[deps] 선택 패키지 ${mod} 를 설치하지 못했다"
  return 1
}

# 보이는 GPU 번호 목록. 확인하지 못하면 빈 문자열.
lg_gpu_list() {
  python3 - <<'PY' 2>/dev/null
try:
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:                                                        # noqa: BLE001
    n = 0
print(",".join(str(i) for i in range(n)))
PY
}

# 결과 묶음. 인자: 묶음 파일 이름, 묶을 경로들. 없는 경로는 뺀다.
# 조각은 임시 파일에 쓴 뒤 교체되므로 *.tmp* 는 넣지 않는다. 실행 중에 묶으면 tar 가 1(파일 변경 경고)을 돌려줄 수 있다. 1 까지는 정상으로 본다.
lg_pack() {
  local out="$1"; shift
  local tmp="${out}.part" p have=() rc
  for p in "$@"; do [ -e "$p" ] && have+=("$p"); done
  [ ${#have[@]} -gt 0 ] || { lg_log "[pack] 묶을 경로가 없다"; return 1; }
  tar czf "$tmp" --warning=no-file-changed --warning=no-file-removed --exclude='*.tmp*' --exclude='__pycache__' "${have[@]}" 2>/dev/null
  rc=$?
  if [ $rc -le 1 ] && gzip -t "$tmp" 2>/dev/null; then
    mv -f "$tmp" "$out"
    lg_log "[pack] $out $(du -h "$out" | cut -f1) · 파일 $(tar tzf "$out" | grep -vc '/$')개"
    return 0
  fi
  rm -f "$tmp"
  lg_log "[pack] 실패(tar 종료 코드 $rc)"
  return 1
}
