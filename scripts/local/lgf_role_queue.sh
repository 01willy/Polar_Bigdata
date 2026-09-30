#!/usr/bin/env bash
# LGF 역할별 큐. 감시기 scripts/local/lgf_supervisor.sh 가 부른다. 사용: lgf_role_queue.sh <single|F|N> <GPU 목록(쉼표)>
#   single: GPU 1장. logs/lgf/run_queue.sh(LGF 8.4 의 J 순서 단일 큐)를 그대로 돈다.
#   F     : h47 의 F 작업(J1, J6, J8, J9, J11). 끝난 단위는 --resume 으로 건너뛴다. J12(f_t3)는 8.4 의 조건부라 넣지 않는다.
#   N     : h48 의 N 작업(J2, J3, J4, J5, J7, J10). 단위 순서는 h48 의 작업 순위(JOB_RANK)와 의존성이 정한다.
#   T3    : F·N 이 모두 끝난 뒤 한 번. h47 --jobs f_t3 --n-jobs-done. 하네스가 LGF 8.4 의 J12 조건(F 기본 범위 완료, 창 여유 > J12 추정)을
#           판단해 허용이면 J12 를 돌고, 아니면 창 파일 decisions 에 'J12 미실행: <사유>' 를 적는다.
# 근거: LGF 계획서 6.3 단계 6('두 장 이상이면 h47 은 한 장에서 F 작업을, h48 은 나머지에서 N 작업을 각자의 순서로 돈다').
# 종료 코드는 logs/lgf/sup/<역할>.rc 에 적는다(0 완료, 3 드레인, 4 창 마감, 그 밖 실패).
set -u
cd /home/willy010313/Polar_Bigdata || exit 2
role=${1:-}; G=${2:-}
case "$role" in single|F|N|T3) ;; *) echo "역할은 single, F, N, T3 가운데 하나다"; exit 2;; esac
[[ "$G" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "GPU 목록 형식 오류: '$G'"; exit 2; }
SD=logs/lgf/sup; mkdir -p "$SD"
echo $$ > "$SD/$role.pid"; echo "$G" > "$SD/$role.gpus"; rm -f "$SD/$role.rc"; date +%s > "$SD/$role.start"
PY=.venv_lgf/bin/python
QLOG=logs/lgf/queue.log
run() {
  echo "[$(date '+%F %T')] 시작[$role]: $*" >> "$QLOG"
  nice -n 10 "$PY" "$@" >> "logs/lgf/queue_steps_$role.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] 끝[$role](rc=$rc): $*" >> "$QLOG"
  return $rc
}
rc=0
case "$role" in
  single) echo $$ > logs/lgf/queue.pid; LGF_GPUS="$G" bash logs/lgf/run_queue.sh; rc=$? ;;
  F) run scripts/3_deep_learning/h47_foundation_models.py --jobs f_n0,f_t1,f_full,f_t2ak,f_t2 --allow-local --gpus "$G" --threads 2 --resume --no-summarize; rc=$? ;;
  N) run scripts/3_deep_learning/h48_nn_tuning.py --jobs n_sel_fast,n_p0_fast,n_sel_slow,n_p0_slow,n_p1,n_t2 --allow-local --gpus "$G" --threads 2 --e1 1 --resume --no-summarize; rc=$? ;;
  T3) VS=$(head -1 logs/lgf/viewing_state.txt 2>/dev/null || echo "감시기 자동 기록")
      run scripts/3_deep_learning/h47_foundation_models.py --jobs f_t3 --n-jobs-done --viewing-state "$VS" --allow-local --gpus "$G" --threads 2 --resume --no-summarize; rc=$? ;;
esac
echo "$rc" > "$SD/$role.rc"
exit "$rc"
