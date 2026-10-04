"""XA_c2_gain_decomposition: P0 대비 이득의 재보정 몫, 수축 잔여, 재보정을 넘어선 ML 몫의 분리와 계수 오차 순위상관(계획 2.1).

계획: docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 2.1절(개정 1, T0 = git 2678100). 설계 보고서: docs/research/2026-10-04/
harness_implementation_plan.md 3.2·4.1절. 계획 문구의 해석과 구현 결정은 docs/research/2026-10-04/impl_notes/xa_c2_gain_decomposition.md 에 적었다.
성격: 사후 분석(맹검 표지 '재현(비맹검)'). 새 적합 없음(적합 0건). 로컬 단계 1d(CPU 4스레드 이하, nice 10).

계산 순서
  1. WF4 조각(results/rescale_wf/data/processed/wf/shards/wf4__cpu__*, 30 (대상, 모드), 132 단위) → 저장소 이름별 h40.TMx
     (h54.summarize_round 와 같은 경로: xbatch_core.load_tms, h54.make_tm, 점 추정 대상은 h54.is_point_only('wf4', 이름)).
  2. 재현 관문 두 개. (1) WF4-c: h54.tests_wf4 2657–2700행 경로를 그대로 다시 계산해 ρ 0.38 [0.17, 0.55]. (2) WF0: wf0_reanalysis.py 의
     입력(LG 곡선, LGD 약관 확인분 곡선)으로 ρ(recal_gain, ml_gain_total) 0.66 과 ρ(recal_gain, ml_beyond_p1) −0.08. 하나라도 소수 둘째
     자리까지 다르면 본 계산을 하지 않고 멈춘다(종료 코드 2).
  3. 대상별 이득 분포: h40.contrast(return_dist=True). 같은 대상의 모든 대비가 재표집 번호(seed_of('lgboot', 이름), 분할별 블록 가중)를
     공유하므로 Δ(P0 − W)^b = Δ(P0 − P1)^b + Δ(P1 − W)^b 가 재표집마다 성립한다(대상·n 마다 점검해 메타에 적는다).
     G_recal = P0 − P1, G_shr = P1 − P2, G_W = P1 − W, G_ML = P1 − R1(λ 0.25), G_ML2 = min(P1, P2) − R1(λ 0.25)(b 마다 최솟값),
     G_tot = P0 − W(양수가 개선). 셀 가중과 블록 등가중을 따로 둔다. 척도 (ii)·(iii)의 RMSE(P0)^b 는 같은 블록 가중으로 계산한다.
  4. 계수 오차: CE2 = 라벨 10개 진단값 |편향| 의 (분할, 추출) 평균(unit.json diag, WF4-c 와 같은 정의), CE1 = |log(E_own/E0)|(오라클),
     CE3 = RMSE(P0) − RMSE(P1@전량)(서술).
  5. Spearman ρ 와 결합 재표집 CI. 주 묶음 = WF4-c 와 같은 28대상(계획 2.1 '대상'). 주 CI = 계열 군집 재표집 × 블록 재표집 × CE2 의
     (분할, 추출) 복원 재표집(seed_of('xa', 대상))을 같은 번호 b 로 묶는다. 계열은 계획의 7계열에 NAtlantic~lic 을 8번째 단독 계열로 더한
     8계열이다(계획 계열 목록에 NAtlantic~lic 이 없다. 구현 해석, xa_hyp.csv 의 deviation 열). 보조 = 대상 고정(WF4-c 와 같은 방식).
     10,000회, 2.5–97.5 백분위, p_one = P(ρ^b ≤ 0)(Holm 입력), p_one_neg = P(ρ^b ≥ 0)(방향 단측 p 용).
     척도 (i) cm, (ii) G / RMSE(P0), (iii) RMSE(P0) 를 통제한 편순위상관(순위 잔차). n 전량(주)과 10·40·160(서술).
     민감도(28대상의 부분 집합): 모드 x 만, 독립 거시 지역 7, 티베트 제외, |A| ≥ 100 대상만. 등록 밖 민감도: 계획 7계열 구성원 27대상(main27).
     내부 일치 점검(등록 관문 아님): 주 묶음·CE2·cm·n 전량 행의 셀 가중 ρ 점 추정(G_tot, G_recal, G_W, G_ML)이 이미 열람된
     derived_c2_posthoc_rho.csv 의 WF4c_28 값과 1e-9 안에서 같아야 한다. 다르면 결과 표를 쓰지 않고 멈춘다(종료 코드 2).
  6. 결과 범주(양(벗어남), 확인하지 못함, 음(벗어남), 판정 불가(군집 부족)), 척도·정의 강건 표지, Holm(m 3, 보조 열), 사전 고정 해석 조각.
     방향 단측 p(점 추정 또는 범주의 방향)의 Holm 보정 값이 0.025 이상이면 양·음 범주의 문장에 '보정 전 유의'를 붙인다(1절).
  7. 서술(XA-6): 대상별 G_recal / G_tot(|G_tot| ≥ 0.5 cm), G_shr / G_tot, 주 4지역 층화 평균의 G_recal, G_shr, G_W(h42.pool_rows).

출력(계획 0.3 출력 제한): 표준 출력에는 대상 수, 계열 수, 관문 통과 여부, 파일 이름·행 수·sha256 만 쓴다. 결과 표는 봉인 폴더
data/processed/xbatch/XA_c2_gain_decomposition/sealed/ 에(xbatch_core.write_sealed), 관문 표(기존 등록 값의 재현)와 메타는 그 위 폴더에 쓴다.

명령행(로컬 자원: 계획 2.1 'CPU 4스레드 이하, nice 10'. --threads 는 4 를 넘을 수 없고(--allow-local 이어도), 시작할 때 nice 를 10 으로 올린다.
가용 메모리가 30 GB 아래면 --mem-wait(기본 3600 s) 동안 60 s 간격으로 기다린다)
  세기(라벨 값 미사용):  CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/2_evaluation/xa_c2_gain_decomposition.py --count-only
  스모크(재표집 200, 출력 제한, 산출 data/processed/xbatch/_smoke/XA_c2_gain_decomposition/):
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 taskset -c <코어 4개> python3 scripts/2_evaluation/xa_c2_gain_decomposition.py --smoke --threads 2
  관문만:   nice -n 10 python3 scripts/2_evaluation/xa_c2_gain_decomposition.py --gates-only --allow-local --threads 4
  본 실행:  nice -n 10 python3 scripts/2_evaluation/xa_c2_gain_decomposition.py --nboot 10000 --threads 4 --allow-local
            (본 실행·관문만·합침은 재표집 10,000회로만 돈다. 다른 --nboot 는 거부한다)
  나눠 실행: 작업 나눔(계획 1절의 적합 조각이 아니다)이다. 동시에 돌리면 조각들이 4스레드 예산을 나눠 쓴다. 예: --shard 0/4 … --shard 3/4 를
            각각 --threads 1 로 동시에 4개, 또는 --threads 4 로 하나씩 차례로. 봉인 폴더에 xa_spearman_part{K}of{N}.csv·.json 을 쓰고,
            끝나면 --merge-shards 4 로 합친다.
  셸에서도 xbatch_core.SHELL_FILTER 를 붙인다: ... 2>&1 | grep -v -E '판정|verdict|Δ|delta|rmse|RMSE|우세|열세|동등|미결정|지지|기각'
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_CORE_DIR = ROOT / "scripts" / "3_deep_learning"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

XA_MAX_THREADS = 4                                                         # 계획 2.1 '로컬 CPU 4스레드 이하'. --allow-local 이어도 넘지 않는다
XA_NICE = 10                                                               # 계획 2.1 'nice 10'
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _peek_xa_threads(argv=None) -> int:
    """명령행의 --threads 를 미리 읽어 [1, XA_MAX_THREADS] 로 자른다(없거나 읽을 수 없으면 1, xbatch_core._peek_threads 와 같은 기본값)."""
    av = sys.argv if argv is None else list(argv)
    val = None
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            val = av[i + 1]
        elif v.startswith("--threads="):
            val = v.split("=", 1)[1]
    try:
        t = int(val) if val is not None else 1
    except ValueError:
        t = 1
    return min(max(t, 1), XA_MAX_THREADS)


def clamp_thread_env(argv=None, env=None) -> int:
    """xbatch_core(numpy)를 부르기 전에 스레드 환경 변수를 XA 상한 4 이하로 둔다. xbatch_core._peek_threads 는 --allow-local 이면 상한을
    풀므로 XA 는 그보다 먼저 값을 정한다. 없는 변수는 --threads 값(4 이하)으로 두고, 이미 4 보다 크거나 읽을 수 없는 값은 낮춘다. 반환: 정한 스레드 수."""
    env = os.environ if env is None else env
    t = _peek_xa_threads(argv)
    for k in _THREAD_VARS:
        cur = env.get(k)
        if cur is None:
            env[k] = str(t)
            continue
        try:
            ok = 1 <= int(cur) <= XA_MAX_THREADS
        except ValueError:
            ok = False
        if not ok:
            env[k] = str(t)
    return t


clamp_thread_env()
import xbatch_core as XB                                                   # noqa: E402  numpy 보다 먼저 불러 스레드 환경 변수를 정한다

import numpy as np                                                         # noqa: E402
import pandas as pd                                                        # noqa: E402

H, X, W, H4, MS = XB.H, XB.X, XB.W, XB.H4, XB.MS

# ================================================================ 1. 고정값(계획 2.1. 바꾸면 사전 등록에서 벗어난다)
EXP_ID = "XA"
EXP_NAME = XB.EXP_NAMES[EXP_ID]                                            # XA_c2_gain_decomposition
TAG = "xa"
WF4_SHARDS = ROOT / "results" / "rescale_wf" / "data" / "processed" / "wf" / "shards"
WF4_TAG = "wf4"
WF4_TESTS = ROOT / "results" / "rescale_wf" / "data" / "processed" / "wf" / "wf_tests.csv"
WF0_SCRIPT = ROOT / "scripts" / "2_evaluation" / "wf0_reanalysis.py"
WF0_META = ROOT / "data" / "processed" / "wf" / "wf0_meta.json"
C2_POSTHOC = ROOT / "paper" / "claims" / "C2_bias_diagnosis" / "tables" / "derived_c2_posthoc_rho.csv"

N_MAIN = -1                                                                # 주 = 라벨 전량
N_GRID = (-1, 10, 40, 160)                                                 # 서술 = 10·40·160
GAIN_NAMES = ("G_recal", "G_shr", "G_W", "G_ML", "G_ML2", "G_tot")
GAIN_PAIRS = {"G_recal": ("P0", "P1"), "G_shr": ("P1", "P2"), "G_W": ("P1", "W"), "G_ML": ("P1", "R1"), "G_tot": ("P0", "W")}
ML2_PAIRS = (("P1", "R1"), ("P2", "R1"))                                   # G_ML2 = min(P1 − R1, P2 − R1) = min(RMSE(P1), RMSE(P2)) − RMSE(R1)
X_KINDS = ("CE2", "CE1")
SCALES = ("cm", "rel", "partial")                                          # (i) cm, (ii) G / RMSE(P0), (iii) RMSE(P0) 통제 편순위상관
SUBSETS = ("main", "x_only", "macro7", "no_tibet", "a100", "main27")
SUBSET_LABEL = dict(main="주 묶음(WF4-c 28대상, 계획 7계열 + NAtlantic~lic 단독 계열 = 8계열)", x_only="민감도: 모드 x 만",
                    macro7="민감도: 독립 거시 지역 7", no_tibet="민감도: 티베트 제외", a100="민감도: |A| ≥ 100 대상만",
                    main27="등록 밖 민감도: 계획 7계열 구성원 27대상(NAtlantic~lic 제외)")
SUBSET_REGISTERED = dict(main=True, x_only=True, macro7=True, no_tibet=True, a100=True, main27=False)
A_MIN = 100                                                                # |A| ≥ 100 조건(대상의 모든 분할에서)
FAMILY_ORDER = ("알래스카", "캐나다", "레나", "러시아 W", "러시아 E", "티베트", "러시아 C")   # 계획 2.1 의 7계열
EXTRA_FAMILY = "북대서양(단독 계열)"                                         # NAtlantic~lic: WF4-c 28대상에 있으나 계획 계열 목록에 없다
FAMILY_ORDER8 = FAMILY_ORDER + (EXTRA_FAMILY,)                              # 주 묶음의 군집 단위
MACRO7 = ("Alaska|x", "Canada|x", "Lena|x", "Russia_W|x", "Russia_E|x", "Tibet_LGD|x", "Russia_C~lgd|x")
DEVIATION_MAIN = ("구현 해석(계획 2.1 '대상'): 주 묶음은 WF4-c 와 같은 28대상이다. NAtlantic~lic 은 계획의 7계열 목록에 없어 8번째 단독 계열로 "
                  "계열 군집 재표집에 넣었다('7계열 복원 추출'을 8계열로 적용). 계획 7계열 구성원 27대상 판은 등록 밖 민감도(main27, cat_main27 열)다")
CLUSTER_MIN_FAMILIES = 3                                                   # 서로 다른 계열이 3개 미만인 재표집
CLUSTER_MAX_FRAC = 0.05                                                    # 그 비율이 5 % 를 넘으면 판정 불가(군집 부족)
MIN_TARGETS = 3
HYPS = {"XA-1": "G_recal", "XA-2": "G_W", "XA-3": "G_ML2", "XA-3a": "G_ML", "XA-4": "G_shr"}
HOLM_FAMILY = ("XA-1", "XA-2", "XA-3")
HOLM_M = 3
BLIND = {"XA-1": "재현(비맹검)", "XA-2": "재현(비맹검)", "XA-3": "비맹검 부분 포함", "XA-3a": "재현(비맹검)", "XA-4": "맹검",
         "XA-5": "부분 재현", "XA-6": "", "서술": "", "등록 밖": ""}
DESIGN = "사후 분석"
GATE1_REF = dict(rho=0.38, ci_lo=0.17, ci_hi=0.55)                          # 계획 2.1 재현 관문 (1)
GATE2_REF = dict(rho_total=0.66, rho_beyond=-0.08)                         # 계획 2.1 재현 관문 (2)
# 내부 일치 점검(등록 관문이 아니다): 주 묶음 점 추정과 이미 열람된 WF4c_28 기록(posthoc_c2_split.py, wf_tests.csv 의 WF4-b·WF4-c 행)
C2_REF_BLOCK = "WF4c_28"
C2_REF_X = "|bias10|"
C2_REF_ROWS = {"G_tot": "gain_w(P0-W)", "G_recal": "recal_share(P0-P1)", "G_W": "ml_beyond_W(P1-W)", "G_ML": "ml_beyond_R1(P1-R1_0.25)"}
C2_REF_TOL = 1e-9
CAT_POS, CAT_NONE, CAT_NEG = "양(벗어남)", "확인하지 못함", "음(벗어남)"
CAT_NA_CLUSTER = "판정 불가(군집 부족)"
CAT_NA_ROWS = "판정 불가(대상 3 미만)"
_CAT_RANK = {CAT_POS: 2, CAT_NEG: 2, CAT_NONE: 1}
SMOKE_NBOOT = 200
CHUNK = 1000                                                               # 재표집 묶음(메모리: 1000 × 28 × 28 × 8 B ≈ 6 MB)


def nlab(n) -> str:
    return "전량" if int(n) == -1 else str(int(n))


def keys_for(n) -> dict:
    """곡선 키(method, learner, alpha, placement, n, lam). WF4 조각의 키와 같다(h54.gk)."""
    return dict(P0=H.P0_GRP, P1=W.gk("P1", n, "none"), P2=W.gk("P2", n, "none"), W=W.gk("W", n, XB.LO, 0.0),
                R1=W.gk("R1", n, XB.LO, XB.LAM_BASE))


# ================================================================ 2. 계열과 대상 묶음
def family_of(name):
    """계열(군집 단위, 계획 2.1 '대상'): 알래스카(알래스카 x, AL-1–AL-6 의 i·x), 캐나다(캐나다 x, CA-2·CA-3), 레나(레나 x, LE-1·LE-2),
    러시아 W, 러시아 E, 티베트, 러시아 C(Russia_C~lgd 만. v3 의 Russia_C 는 계획 목록에 없어 None). NAtlantic~lic 은 WF4-c 28대상에
    있으나 계획 7계열에 없으므로 8번째 단독 계열(EXTRA_FAMILY)로 둔다. 그 밖은 None."""
    base = str(name).split("|")[0]
    if base == "Alaska" or base.startswith("AL-"):
        return "알래스카"
    if base == "Canada" or base.startswith("CA-"):
        return "캐나다"
    if base == "Lena" or base.startswith("LE-"):
        return "레나"
    if base == "Russia_W":
        return "러시아 W"
    if base == "Russia_E":
        return "러시아 E"
    if base == "Tibet_LGD":
        return "티베트"
    if base == "Russia_C~lgd":
        return "러시아 C"
    if base == "NAtlantic~lic":
        return EXTRA_FAMILY
    return None


def _fam_key(f):
    return (FAMILY_ORDER8.index(f) if f in FAMILY_ORDER8 else len(FAMILY_ORDER8), str(f))


def subset_members(T, sub) -> list:
    """대상 묶음 → [(저장소 이름, 계열)]. 진단값(CE2)이 있고 계열이 정해진 대상만 넣는다(WF4-c 와 같다). main = WF4-c 28대상(8계열),
    민감도 4종은 main 의 부분 집합, main27 = 계획 7계열 구성원(등록 밖 민감도)."""
    main = [(nm, family_of(nm)) for nm in sorted(T) if T[nm]["has_ce2"] and family_of(nm)]
    if sub == "main":
        return main
    if sub == "main27":
        return [(nm, f) for nm, f in main if f in FAMILY_ORDER]
    if sub == "x_only":
        return [(nm, f) for nm, f in main if nm.endswith("|x")]
    if sub == "macro7":
        return [(nm, f) for nm, f in main if nm in MACRO7]
    if sub == "no_tibet":
        return [(nm, f) for nm, f in main if f != "티베트"]
    if sub == "a100":
        return [(nm, f) for nm, f in main if T[nm]["nA_min"] is not None and T[nm]["nA_min"] >= A_MIN]
    raise ValueError(f"알 수 없는 대상 묶음 {sub}")


# ================================================================ 3. 자료(WF4 조각)
def load_wf4_tms(shards_dir=WF4_SHARDS, tag=WF4_TAG, nboot=XB.NBOOT):
    """WF4 조각 → ({저장소 이름: h40.TMx}, unit 목록). h54.summarize_round 와 같은 경로(조각 이름 순, h54.make_tm)다."""
    tms, units, _ = XB.load_tms(shards_dir, tag, int(nboot), point_only=lambda nm: W.is_point_only("wf4", nm))
    if not tms:
        raise SystemExit(f"[xa] 조각 없음: {shards_dir}/{tag}__*")
    return tms, units


def ce_values(units) -> dict:
    """대상별 계수 오차 원값. CE2 의 (분할, 추출) 값은 unit 순서(조각 이름 순, WF4-c 와 같다)로 모은다. CE1 = |log(E_own/E0)|
    (unit.json 의 logE_ratio_own, 분할마다 같은 값이다). |A| 는 분할별 n_A 의 최소·최대."""
    out = {}
    for u in units:
        nm = f"{u['target']}|{u['mode']}"
        d = out.setdefault(nm, dict(abs_vals=[], sgn_vals=[], logr=[], nA=[]))
        d["abs_vals"] += [float(v["abs_bias"]) for v in (u.get("diag") or [])]
        d["sgn_vals"] += [float(v["bias"]) for v in (u.get("diag") or [])]
        lr = u.get("logE_ratio_own")
        d["logr"].append(float(lr) if lr is not None else np.nan)
        if u.get("n_A") is not None:
            d["nA"].append(int(u["n_A"]))
    res = {}
    for nm, d in out.items():
        av = np.asarray(d["abs_vals"], float)
        lr = np.asarray(d["logr"], float)
        res[nm] = dict(abs_vals=av, sgn_vals=np.asarray(d["sgn_vals"], float), n_diag=int(len(av)),
                       ce2=float(av.mean()) if len(av) else np.nan,
                       ce1=float(abs(np.nanmean(lr))) if np.isfinite(lr).any() else np.nan,
                       ce1_const=bool(np.isfinite(lr).any() and np.nanmax(lr) - np.nanmin(lr) < 1e-12),
                       nA_min=int(min(d["nA"])) if d["nA"] else None, nA_max=int(max(d["nA"])) if d["nA"] else None)
    return res


def ce2_boot(abs_vals, name, nboot) -> np.ndarray:
    """CE2 재표집: (분할, 추출) 값의 복원 재표집 평균(seed_of('xa', 대상)). 반환 (nboot,)."""
    av = np.asarray(abs_vals, float)
    k = len(av)
    pick = np.random.RandomState(XB.seed_of("xa", name)).randint(0, k, size=(int(nboot), k))
    return av[pick].mean(1)


def target_table(tms, units) -> dict:
    """대상 정보(계열, 모드, 점 추정 여부, |A|, CE1, CE2 원값)."""
    ce = ce_values(units)
    T = {}
    for nm, tm in sorted(tms.items()):
        c = ce.get(nm, {})
        T[nm] = dict(name=nm, family=family_of(nm), plan_family7=family_of(nm) in FAMILY_ORDER, mode=nm.split("|")[1], point_only=bool(tm.point_only),
                     has_ci=bool(tm.has_ci), n_splits=int(len(tm.used)), nb_union=int(tm.nb_union), nA_min=c.get("nA_min"),
                     nA_max=c.get("nA_max"), n_diag=int(c.get("n_diag", 0)), ce1=c.get("ce1", np.nan), ce1_const=c.get("ce1_const", False),
                     ce2=c.get("ce2", np.nan), abs_vals=c.get("abs_vals", np.zeros(0)), has_ce2=bool(c.get("n_diag", 0) > 0))
    return T


# ================================================================ 4. 이득 분포(h40.contrast, 같은 재표집 번호)
def _contrast(tm, gA, gB):
    """Δ = RMSE(A) − RMSE(B) 의 점 추정과 분포(h40.contrast). 양수 = B 가 낫다. 이득 G = RMSE(앞) − RMSE(뒤) 를 그대로 준다."""
    r = H.contrast(tm, gA, gB, return_dist=True)
    if r is None or not np.isfinite(r.get("delta", np.nan)):
        return None
    return dict(pt=float(r["delta"]), pt_beq=float(r["delta_beq"]), dist=r.get("dist"), dist_beq=r.get("dist_beq"),
                splits=tuple(sorted(int(s) for s in r["delta_split"])), n_splits=int(r["n_splits"]),
                n_blocks_min=int(min(r["n_blocks"])) if r.get("n_blocks") else 0)


def level_dist(tm, g, splits=None) -> dict | None:
    """곡선 키 g 의 대상 수준 RMSE 와 분포(셀 가중, 블록 등가중). h4_common.boot_delta_blocks 의 팔 하나 계산과 같다: 분할마다
    boot_weights(nb, nboot, seed_of(tm.seed, 분할)), 키 평균 뒤 분할 평균. splits 가 있으면 그 분할만 쓴다(이득과 같은 분할 집합)."""
    pts, ptb, dc, db, used = [], [], [], [], []
    for sp in sorted(tm.used):
        if splits is not None and int(sp) not in splits:
            continue
        ks = tm.idx[sp].get(g)
        if not ks:
            continue
        st = tm.used[sp]
        S, C = st.matrices(ks)
        with np.errstate(invalid="ignore", divide="ignore"):
            pts.append(float(np.nanmean(H4._rmse_rows(S, C)))); ptb.append(float(np.nanmean(H4._beq_rows(S, C))))
            if tm.nboot > 0:
                Wm = H4.boot_weights(st.nb, tm.nboot, XB.seed_of(tm.seed, sp))
                dc.append(np.nanmean(np.sqrt((S @ Wm.T) / (C @ Wm.T)), 0))
                v = np.where(C > 0, np.sqrt(S / np.where(C > 0, C, 1)), 0.0)
                m = (C > 0).astype(float)
                db.append(np.nanmean((v @ Wm.T) / (m @ Wm.T), 0))
        used.append(int(sp))
    if not pts:
        return None
    return dict(pt=float(np.mean(pts)), pt_beq=float(np.mean(ptb)), dist=np.mean(dc, 0) if dc else None, dist_beq=np.mean(db, 0) if db else None,
                splits=tuple(used))


def _dmax(*vals):
    """가법성 점검의 최대 절대 차(유한 값만)."""
    out = 0.0
    for v in vals:
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v):
            out = max(out, float(np.max(np.abs(v))))
    return out


def gains(tm, n) -> tuple:
    """대상 하나, 라벨 수 n 의 이득 분포 6종(G_recal, G_shr, G_W, G_ML, G_ML2, G_tot). 각 항목: pt, pt_beq(점 추정), dist, dist_beq
    (재표집, 점 추정만인 대상은 None), splits, p0·p0_beq·p0_dist·p0_dist_beq(같은 분할 집합의 RMSE(P0)). 반환 (이득 dict, 점검 dict).
    점검: Δ(P0 − W)^b − (Δ(P0 − P1)^b + Δ(P1 − W)^b) 의 최대 절대 차(두 가중, 점 추정 포함)와 분할 집합 일치 여부."""
    k = keys_for(n)
    pairs = set(GAIN_PAIRS.values()) | set(ML2_PAIRS)
    c = {p: _contrast(tm, k[p[0]], k[p[1]]) for p in sorted(pairs)}
    out = {g: dict(c[p]) for g, p in GAIN_PAIRS.items() if c[p] is not None}
    a_, b_ = c[ML2_PAIRS[0]], c[ML2_PAIRS[1]]
    ml2_mismatch = bool(a_ is not None and b_ is not None and a_["splits"] != b_["splits"])
    if a_ is not None and b_ is not None and not ml2_mismatch:
        both = a_["dist"] is not None and b_["dist"] is not None
        out["G_ML2"] = dict(pt=min(a_["pt"], b_["pt"]), pt_beq=min(a_["pt_beq"], b_["pt_beq"]),
                            dist=np.minimum(a_["dist"], b_["dist"]) if both else None,
                            dist_beq=np.minimum(a_["dist_beq"], b_["dist_beq"]) if both else None,
                            splits=a_["splits"], n_splits=a_["n_splits"], n_blocks_min=min(a_["n_blocks_min"], b_["n_blocks_min"]),
                            ml2_from_p2_frac=float(np.mean(b_["dist"] < a_["dist"])) if both else float(b_["pt"] < a_["pt"]))
    p0_cache = {}
    for g, s in out.items():
        sp = s["splits"]
        if sp not in p0_cache:
            p0_cache[sp] = level_dist(tm, H.P0_GRP, set(sp))
        lv = p0_cache[sp]
        s.update(p0=lv["pt"] if lv else np.nan, p0_beq=lv["pt_beq"] if lv else np.nan, p0_dist=lv["dist"] if lv else None,
                 p0_dist_beq=lv["dist_beq"] if lv else None)
    chk = dict(n=int(n), additive_checked=False, additive_same_splits=False, additive_max_abs=np.nan, ml2_split_mismatch=ml2_mismatch)
    a01, a1w, a0w = c[("P0", "P1")], c[("P1", "W")], c[("P0", "W")]
    if a01 is not None and a1w is not None and a0w is not None:
        same = a01["splits"] == a1w["splits"] == a0w["splits"]
        dev = [a0w["pt"] - (a01["pt"] + a1w["pt"]), a0w["pt_beq"] - (a01["pt_beq"] + a1w["pt_beq"])]
        if a0w["dist"] is not None:
            dev += [a0w["dist"] - (a01["dist"] + a1w["dist"]), a0w["dist_beq"] - (a01["dist_beq"] + a1w["dist_beq"])]
        chk.update(additive_checked=True, additive_same_splits=bool(same), additive_max_abs=_dmax(*dev))
    return out, chk


def all_gains(tms, T, n_grid=N_GRID) -> tuple:
    """{이름: {n: {이득: 통계}}}, 가법성 점검 표. 대상마다 boot_weights 캐시를 비운다(메모리)."""
    G, checks = {}, []
    for nm in sorted(tms):
        G[nm] = {}
        for n in n_grid:
            g, chk = gains(tms[nm], n)
            G[nm][int(n)] = g
            chk.update(name=nm)
            checks.append(chk)
        XB.boot_cache_clear()
    return G, pd.DataFrame(checks)


# ================================================================ 5. 순위상관(중복도 가중, 재표집 행 단위)
def _wranks(V, M):
    """행(재표집)마다 중복도 가중 평균 순위. V, M: (B, J). 값 v 인 항목의 순위 = (v 보다 작은 항목의 중복도 합) + (v 와 같은 항목의
    중복도 합 + 1)/2. 중복도가 정수이면 항목을 그 수만큼 복제한 표본의 평균 순위(동률 평균)와 같다. M = 0 인 항목은 다른 순위에 영향이 없다."""
    lt = (V[:, None, :] < V[:, :, None]).astype(float)
    eq = (V[:, None, :] == V[:, :, None]).astype(float)
    return np.einsum("bjk,bk->bj", lt, M) + 0.5 * (np.einsum("bjk,bk->bj", eq, M) + 1.0)


def _wcorr(A, Bv, M):
    """행마다 중복도 가중 Pearson 상관."""
    with np.errstate(invalid="ignore", divide="ignore"):
        sw = M.sum(1)
        ma = (M * A).sum(1) / sw
        mb = (M * Bv).sum(1) / sw
        da = A - ma[:, None]; db = Bv - mb[:, None]
        r = (M * da * db).sum(1) / np.sqrt((M * da * da).sum(1) * (M * db * db).sum(1))
    return np.clip(r, -1.0, 1.0)


def _partial(rxy, rxz, ryz):
    """순위 잔차 방식의 편순위상관: rank(x)·rank(y) 를 rank(z) 에 (가중) 최소제곱 회귀한 잔차의 상관 = 순위 상관의 편상관 공식."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rxy - rxz * ryz) / np.sqrt((1.0 - rxz ** 2) * (1.0 - ryz ** 2))
    r = np.where(np.isfinite(r), r, np.nan)
    return np.clip(r, -1.0, 1.0)


def _prep(Xv, Yv, M, Z):
    Yv = np.atleast_2d(np.asarray(Yv, float))
    Bn, J = Yv.shape
    Xv = np.broadcast_to(np.atleast_2d(np.asarray(Xv, float)), (Bn, J))
    M = np.ones((Bn, J)) if M is None else np.broadcast_to(np.atleast_2d(np.asarray(M, float)), (Bn, J))
    valid = (M > 0) & np.isfinite(Xv) & np.isfinite(Yv)
    if Z is not None:
        Z = np.broadcast_to(np.atleast_2d(np.asarray(Z, float)), (Bn, J))
        valid &= np.isfinite(Z)
        Z = np.where(valid, Z, 0.0)
    return np.where(valid, Xv, 0.0), np.where(valid, Yv, 0.0), np.where(valid, M, 0.0), Z


def wspearman(Xv, Yv, M=None, Z=None, chunk=CHUNK) -> np.ndarray:
    """행(재표집)마다 중복도 가중 Spearman ρ(동률 평균 순위). Z 가 있으면 Z 를 통제한 편순위상관(순위 잔차 방식). 서로 다른 유효 항목이
    3개 미만인 행은 NaN. 입력 (B, J) 또는 (J,)(행 하나), M 은 중복도(대상 고정이면 1)."""
    Xc, Yc, Mc, Zc = _prep(Xv, Yv, M, Z)
    Bn = Yc.shape[0]
    out = np.full(Bn, np.nan)
    for s in range(0, Bn, int(chunk)):
        sl = slice(s, s + int(chunk))
        m = Mc[sl]
        rx, ry = _wranks(Xc[sl], m), _wranks(Yc[sl], m)
        r = _wcorr(rx, ry, m)
        if Zc is not None:
            rz = _wranks(Zc[sl], m)
            r = _partial(r, _wcorr(rx, rz, m), _wcorr(ry, rz, m))
        out[sl] = np.where((m > 0).sum(1) >= MIN_TARGETS, r, np.nan)
    return out


def partial_spearman(x, y, z, w=None) -> float:
    """편순위상관 하나(점 추정). x, y, z: (J,), w: 중복도."""
    return float(wspearman(np.asarray(x, float)[None], np.asarray(y, float)[None], None if w is None else np.asarray(w, float)[None],
                           np.asarray(z, float)[None])[0])


def cluster_counts(F, nboot) -> np.ndarray:
    """계열 군집 복원 추출: 계열 목록 F 에서 len(F) 개를 복원 추출한 횟수 (nboot, len(F)). seed_of('xa', 'cluster', 계열…)라 같은 계열
    집합의 행은 같은 추출을 쓴다(가설 사이 공통 난수)."""
    F = tuple(F)
    I = np.random.RandomState(XB.seed_of("xa", "cluster", *F)).randint(0, len(F), size=(int(nboot), len(F)))
    cnt = np.zeros((int(nboot), len(F)))
    np.add.at(cnt, (np.repeat(np.arange(int(nboot)), len(F)), I.ravel()), 1.0)
    return cnt


def cluster_boot(fams, nboot, cache=None) -> dict:
    """대상별 계열 목록 → 대상 중복도 M (nboot, J), 서로 다른 계열 수, 3계열 미만 비율과 '군집 부족' 여부."""
    F = tuple(sorted(set(fams), key=_fam_key))
    key = (F, int(nboot))
    if cache is not None and key in cache:
        cnt = cache[key]
    else:
        cnt = cluster_counts(F, nboot)
        if cache is not None:
            cache[key] = cnt
    idx = np.array([F.index(f) for f in fams], int)
    distinct = (cnt > 0).sum(1)
    frac = float(np.mean(distinct < CLUSTER_MIN_FAMILIES))
    return dict(M=cnt[:, idx], families=F, distinct=distinct, frac_lt3=frac, insufficient=bool(frac > CLUSTER_MAX_FRAC))


def _ci(d):
    """유한 값의 2.5–97.5 백분위, p_one = P(ρ^b ≤ 0)(계획 2.1, Holm 입력), p_one_neg = P(ρ^b ≥ 0)(음 방향 단측 p), 유한 값 수."""
    d = np.asarray(d, float)
    f = d[np.isfinite(d)]
    if not len(f):
        return np.nan, np.nan, np.nan, np.nan, 0
    return (float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5)), float(np.mean(f <= 0)), float(np.mean(f >= 0)), int(len(f)))


def spec_list() -> list:
    """Spearman 행 명세(고정 순서): 이득 6 × 계수 오차 2 × 척도 3 × n 4 × 대상 묶음 6."""
    return [dict(spec_idx=i, gain=g, x=xk, scale=sc, n=int(n), subset=sub)
            for i, (g, xk, sc, n, sub) in enumerate(itertools.product(GAIN_NAMES, X_KINDS, SCALES, N_GRID, SUBSETS))]


def hyp_of(spec) -> str:
    """명세 → 가설 표지. 주 행(전량, CE2, cm, 주 묶음 28대상)은 XA-1·2·3·3a·4, G_tot 은 서술, 등록 밖 묶음(main27)은 '등록 밖', 나머지는 XA-5."""
    if not SUBSET_REGISTERED[spec["subset"]]:
        return "등록 밖"
    if spec["gain"] == "G_tot":
        return "서술"
    if spec["n"] == N_MAIN and spec["x"] == "CE2" and spec["scale"] == "cm" and spec["subset"] == "main":
        return next(h for h, g in HYPS.items() if g == spec["gain"])
    return "XA-5"


def _vals(s, w, scale):
    """이득 통계 s 의 가중 w 값: (점, 분포 또는 None, P0 점, P0 분포)."""
    suf = "" if w == "cell" else "_beq"
    return s["pt" + suf], s["dist" + suf], s["p0" + suf], s["p0_dist" + suf]


def joint_spearman(spec, T, G, nboot, cl_cache=None) -> dict:
    """Spearman 행 하나: 점 추정 ρ(대상 고정), 주 CI(계열 군집 결합 재표집), 보조 CI(대상 고정 결합 재표집), 두 가중."""
    g, xk, sc, n, sub = spec["gain"], spec["x"], spec["scale"], int(spec["n"]), spec["subset"]
    use = []
    for nm, fam in subset_members(T, sub):
        s = G.get(nm, {}).get(n, {}).get(g)
        if s is None or not (np.isfinite(s["pt"]) and np.isfinite(s["pt_beq"])):
            continue
        xv = T[nm]["ce2"] if xk == "CE2" else T[nm]["ce1"]
        if not np.isfinite(xv):
            continue
        if sc != "cm" and not (np.isfinite(s["p0"]) and np.isfinite(s["p0_beq"]) and s["p0"] > 0 and s["p0_beq"] > 0):
            continue
        use.append((nm, fam, s))
    row = dict(spec, hyp=hyp_of(spec), registered=SUBSET_REGISTERED[sub], subset_label=SUBSET_LABEL[sub], blind=BLIND.get(hyp_of(spec), ""),
               design=DESIGN, n_targets=len(use), n_families=len({f for _, f, _ in use}), targets=",".join(nm for nm, _, _ in use),
               n_point_only=int(sum(T[nm]["point_only"] for nm, _, _ in use)), nboot=int(nboot))
    if len(use) < MIN_TARGETS:
        row.update(status=f"대상 {len(use)} < {MIN_TARGETS}", cluster_insufficient=True, cluster_frac_lt3=np.nan)
        return row
    J, B = len(use), int(nboot)
    names = [nm for nm, _, _ in use]
    if xk == "CE2":
        xpt = np.array([T[nm]["ce2"] for nm in names])
        xb = np.column_stack([T[nm]["ce2_b"][:B] for nm in names])
    else:
        xpt = np.array([T[nm]["ce1"] for nm in names])
        xb = np.broadcast_to(xpt, (B, J))
    cb = cluster_boot([f for _, f, _ in use], B, cl_cache)
    kinds = (("cluster", cb["M"]), ("fixed", None))
    row.update(status="ok", families=",".join(cb["families"]), cluster_frac_lt3=cb["frac_lt3"], cluster_insufficient=cb["insufficient"])
    for w in ("cell", "beq"):
        ypt, yb, zpt, zb = [], [], [], []
        for _, _, s in use:
            p, d, p0, d0 = _vals(s, w, sc)
            ypt.append(p); zpt.append(p0)
            yb.append(d[:B] if d is not None else np.full(B, p))          # 점 추정만인 대상은 이득을 고정한다(WF4-c 와 같다)
            zb.append(d0[:B] if d0 is not None else np.full(B, p0))
        ypt, zpt = np.array(ypt), np.array(zpt)
        yb, zb = np.column_stack(yb), np.column_stack(zb)
        if sc == "rel":
            ypt, yb = ypt / zpt, yb / zb
        zz_pt, zz_b = (zpt, zb) if sc == "partial" else (None, None)
        row[f"rho_{w}"] = float(wspearman(xpt[None], ypt[None], None, None if zz_pt is None else zz_pt[None])[0])
        for kind, M in kinds:
            rb = wspearman(xb, yb, M, zz_b)
            lo, hi, p1, p1n, nf = _ci(rb)
            row.update({f"{kind}_lo_{w}": lo, f"{kind}_hi_{w}": hi, f"{kind}_p_one_{w}": p1, f"{kind}_p_one_neg_{w}": p1n,
                        f"{kind}_nfinite_{w}": nf})
    for kind in ("cluster", "fixed"):
        for suf in ("p_one", "p_one_neg"):                                  # 두 가중 가운데 큰 값(1절 'p 값'의 규칙)
            ps = [v for v in (row[f"{kind}_{suf}_cell"], row[f"{kind}_{suf}_beq"]) if np.isfinite(v)]
            row[f"{kind}_{suf}_max"] = float(max(ps)) if ps else np.nan
    return row


def spearman_table(T, G, nboot, specs=None) -> pd.DataFrame:
    cache = {}
    return pd.DataFrame([joint_spearman(sp, T, G, nboot, cache) for sp in (specs if specs is not None else spec_list())])


# ================================================================ 6. 결과 범주, 강건 표지, Holm, 해석 조각
def _rec(df, g, xk, sc, n, sub):
    q = df[(df.gain == g) & (df.x == xk) & (df.scale == sc) & (df.n == int(n)) & (df.subset == sub)]
    return q.iloc[0].to_dict() if len(q) else None


def _fin(v):
    return v is not None and np.isfinite(float(v))


def category(main, a100, ci="cluster") -> str:
    """결과 범주(계획 2.1). 양 = (a) 주 묶음의 CI(ci = cluster 이면 계열 군집, fixed 이면 대상 고정) 하한이 두 가중 모두 > 0 이고 (b) |A| ≥ 100
    대상만의 보조 CI(대상 고정) 하한도 두 가중 모두 > 0. 음 = 같은 조건을 상한 < 0 으로. 그 밖은 확인하지 못함. 계열 군집 CI 에서 3계열
    미만 재표집이 5 % 를 넘으면 판정 불가(군집 부족). (b) 를 계산할 수 없으면 (b) 불충족으로 본다."""
    if main is None or main.get("status") != "ok":
        return CAT_NA_ROWS
    if ci == "cluster" and bool(main.get("cluster_insufficient")):
        return CAT_NA_CLUSTER
    lo = [main.get(f"{ci}_lo_cell"), main.get(f"{ci}_lo_beq")]
    hi = [main.get(f"{ci}_hi_cell"), main.get(f"{ci}_hi_beq")]
    ok_b = a100 is not None and a100.get("status") == "ok"
    blo = [a100.get("fixed_lo_cell"), a100.get("fixed_lo_beq")] if ok_b else [np.nan, np.nan]
    bhi = [a100.get("fixed_hi_cell"), a100.get("fixed_hi_beq")] if ok_b else [np.nan, np.nan]
    if all(_fin(v) and float(v) > 0 for v in lo + blo):
        return CAT_POS
    if all(_fin(v) and float(v) < 0 for v in hi + bhi):
        return CAT_NEG
    return CAT_NONE


def weaker(a, b) -> str:
    """두 범주 가운데 약한 쪽(주 CI 와 보조 CI 가 다를 때). 판정 불가 < 확인하지 못함 < 양·음, 양과 음이 맞서면 확인하지 못함."""
    if a == b:
        return a
    ra, rb = _CAT_RANK.get(a, 0), _CAT_RANK.get(b, 0)
    if ra == rb == 2:
        return CAT_NONE
    if ra == rb:
        return a
    return a if ra < rb else b


def category_table(sp) -> pd.DataFrame:
    """(이득, 계수 오차, 척도, n)마다 주 묶음(28대상)과 |A| ≥ 100 묶음으로 범주를 낸다. cat_cluster(주 CI), cat_fixed(보조 CI), category(약한 쪽).
    cat_main27 = 같은 규칙을 계획 7계열 27대상(main27, 등록 밖 민감도)에 적용한 범주((b)는 같은 |A| ≥ 100 묶음)."""
    rows = []
    for g, xk, sc, n in itertools.product(GAIN_NAMES, X_KINDS, SCALES, N_GRID):
        m, b, m27 = _rec(sp, g, xk, sc, n, "main"), _rec(sp, g, xk, sc, n, "a100"), _rec(sp, g, xk, sc, n, "main27")
        cc, cf = category(m, b, "cluster"), category(m, b, "fixed")
        rows.append(dict(gain=g, x=xk, scale=sc, n=int(n), cat_cluster=cc, cat_fixed=cf, category=weaker(cc, cf), two_categories=cc != cf,
                         cat_main27=weaker(category(m27, b, "cluster"), category(m27, b, "fixed")),
                         a100_status=(b or {}).get("status", "행 없음"), main_status=(m or {}).get("status", "행 없음")))
    return pd.DataFrame(rows)


def _cat(ct, g, xk="CE2", sc="cm", n=N_MAIN):
    q = ct[(ct.gain == g) & (ct.x == xk) & (ct.scale == sc) & (ct.n == int(n))]
    return q.iloc[0].to_dict() if len(q) else dict(category=CAT_NA_ROWS, cat_cluster=CAT_NA_ROWS, cat_fixed=CAT_NA_ROWS, two_categories=False,
                                                    cat_main27=CAT_NA_ROWS)


def direction_of(cat, r) -> int:
    """방향 단측 p 의 방향(1절 'p 값': 방향은 점 추정으로 정한다). 범주가 양·음이면 그 방향, 아니면 셀 가중 ρ 점 추정의 부호(0 이상이면 양)."""
    if cat == CAT_POS:
        return 1
    if cat == CAT_NEG:
        return -1
    rho = (r or {}).get("rho_cell")
    return -1 if (_fin(rho) and float(rho) < 0) else 1


def p_direction(r, d) -> float:
    """주 CI(계열 군집) 분포의 방향 단측 p(두 가중 가운데 큰 값). d = 1 이면 p_one = P(ρ^b ≤ 0), d = −1 이면 p_one_neg = P(ρ^b ≥ 0)."""
    if not r or r.get("status") != "ok":
        return np.nan
    v = r.get("cluster_p_one_max" if d > 0 else "cluster_p_one_neg_max")
    return float(v) if _fin(v) else np.nan


def stat_text(r) -> str:
    """'(ρ, CI)' 자리의 수치(주 CI = 계열 군집, 두 가중)."""
    if r is None or r.get("status") != "ok":
        return "ρ 없음"
    return (f"ρ {r['rho_cell']:.2f}, CI [{r['cluster_lo_cell']:.2f}, {r['cluster_hi_cell']:.2f}], 셀 가중; "
            f"ρ {r['rho_beq']:.2f}, CI [{r['cluster_lo_beq']:.2f}, {r['cluster_hi_beq']:.2f}], 블록 등가중")


def _tag(sentences, notes=()) -> str:
    """문장마다 '사후 분석'을 붙인다(계획 2.1 공통 규칙). notes 는 마지막 문장의 괄호에 더한다."""
    out = []
    for i, s in enumerate(sentences):
        s = str(s).rstrip().rstrip(".")
        extra = [DESIGN] + (list(notes) if i == len(sentences) - 1 else [])
        out.append(f"{s}({', '.join(extra)}).")
    return " ".join(out)


def fragment(hyp, cat, robust, st, st_ce1="", by_def="", reason="") -> tuple:
    """사전 고정 해석 조각(계획 2.1 표). 반환 (문장 목록, 지시 문구). cat = 그 가설의 범주(약한 쪽 규칙 적용 뒤), robust = dict(scale, definition,
    all_pos(XA-3: CE1·CE2·척도 (ii)·(iii) 모두 양)). 판정 불가 범주는 1절 '해석 문장의 다섯 갈래'의 판정 불가 갈래로 쓴다."""
    if hyp == "XA-1":
        if cat == CAT_POS:
            if robust["definition"]:
                return [f"원천 계수의 오차가 클수록 라벨로 계수를 다시 맞춘 이득이 컸다(계수 오차 CE1·라벨 10개 진단값 CE2 모두, CE2 {st}; CE1 {st_ce1})"], ""
            return [f"라벨 10개로 잰 물리식 편향의 크기는 재보정 이득과 양의 순위상관을 보였다({st})",
                    "대상 전체 라벨로 잰 계수 오차(CE1)에서는 확인하지 못했다"], ""
        if cat == CAT_NONE:
            return (["라벨 10개로 잰 물리식 편향의 크기와 재보정 이득의 관계는 확인하지 못했다"],
                    "C2 의 비례·순위 문장을 쓰지 않는다. 진단 문장은 WF4-c 의 등록 판정(P0 − W, ρ 0.38 [0.17, 0.55], 셀 가중)만 쓴다")
        if cat == CAT_NEG:
            return [f"편향 크기와 재보정 이득은 음의 순위상관을 보였다({st})"], "C2 를 철회한다"
        return [f"라벨 10개로 잰 물리식 편향의 크기와 재보정 이득의 관계는 판정할 수 없었다({reason or cat})"], ""
    if hyp == "XA-2":
        subj = "라벨 안 교차검증 규칙이 재보정에 더한 몫"
        if cat == CAT_POS:
            if robust["scale"] and robust["definition"]:
                return [f"{subj}도 편향 크기와 함께 커졌다({st})"], ""
            return [f"{subj}은 편향 크기와 양의 순위상관을 보였으나({st}) 척도 또는 정의를 바꾸면 확인하지 못했다"], ""
        if cat == CAT_NONE:
            return [f"{subj}과 편향 크기의 관계는 확인하지 못했다"], ""
        if cat == CAT_NEG:
            return [f"{subj}은 편향 크기와 음의 순위상관을 보였다({st})"], ""
        return [f"{subj}과 편향 크기의 관계는 판정할 수 없었다({reason or cat})"], ""
    if hyp == "XA-3":
        if cat == CAT_POS:
            if robust["all_pos"]:
                return [f"ML 몫도 편향 크기와 함께 커졌다({st})"], ""
            return [f"수축 잔여를 뺀 ML 몫은 편향 크기와 양의 순위상관을 보였으나({st}) 정의에 따라 달랐다(정의별 범주: {by_def})"], ""
        if cat == CAT_NONE:
            return ["재보정을 넘어선 ML 몫과 편향 크기의 관계는 확인하지 못했다"], ""
        if cat == CAT_NEG:
            return [f"수축 잔여를 뺀 ML 몫은 편향 크기와 음의 순위상관을 보였다({st})"], ""
        return [f"재보정을 넘어선 ML 몫과 편향 크기의 관계는 판정할 수 없었다({reason or cat})"], ""
    return [], ""


def verdict_table(sp, ct) -> pd.DataFrame:
    """가설 행(XA-1, XA-2, XA-3, XA-3a, XA-4)과 C2 문장 행(XA-1·2·3 조각을 이 순서로 이어 붙인 것).
    Holm: 등록 입력은 {XA-1, XA-2, XA-3}의 p_one = P(ρ^b ≤ 0)(holm_p 열, 보조 열). '보정 전 유의' 표지는 1절 'p 값'(방향은 점 추정)에 따라
    방향 단측 p(양이면 p_one, 음이면 p_one_neg = P(ρ^b ≥ 0))의 Holm 보정 값(holm_p_dir)이 0.025 이상인 양·음 범주에 붙이고, 그 가설 문장과
    C2 문장에 함께 적는다(1절 '해석 문장의 다섯 갈래', '다중성')."""
    main = {h: _rec(sp, g, "CE2", "cm", N_MAIN, "main") for h, g in HYPS.items()}
    a100 = {h: _rec(sp, g, "CE2", "cm", N_MAIN, "a100") for h, g in HYPS.items()}
    c0s = {h: _cat(ct, g) for h, g in HYPS.items()}
    dirs = {h: direction_of(c0s[h]["category"], main[h]) for h in HYPS}
    holm_in = [main[h]["cluster_p_one_max"] if (main[h] and main[h].get("status") == "ok") else np.nan for h in HOLM_FAMILY]
    hp = dict(zip(HOLM_FAMILY, XB.holm(holm_in, HOLM_M)))
    pdir = {h: p_direction(main[h], dirs[h]) for h in HYPS}
    hpd = dict(zip(HOLM_FAMILY, XB.holm([pdir[h] for h in HOLM_FAMILY], HOLM_M)))
    rows, frags = [], {}
    for h, g in HYPS.items():
        c0 = c0s[h]
        cats = {lab: _cat(ct, g, xk, sc)["category"] for lab, xk, sc in (("CE2", "CE2", "cm"), ("CE1", "CE1", "cm"), ("rel", "CE2", "rel"),
                                                                          ("partial", "CE2", "partial"))}
        cat = c0["category"]
        robust = dict(scale=cats["rel"] == cat and cats["partial"] == cat, definition=cats["CE1"] == cat,
                      all_pos=all(v == CAT_POS for v in cats.values()))
        r, r1 = main[h], _rec(sp, g, "CE1", "cm", N_MAIN, "main")
        by_def = f"CE2 {cats['CE2']}, CE1 {cats['CE1']}, 척도 (ii) {cats['rel']}, 척도 (iii) {cats['partial']}"
        reason = "군집 부족" if cat == CAT_NA_CLUSTER else ("대상 3 미만" if cat == CAT_NA_ROWS else "")
        sents, directive = fragment(h, cat, robust, stat_text(r), stat_text(r1), by_def, reason)
        uncorrected = bool(h in hpd and cat in (CAT_POS, CAT_NEG) and np.isfinite(hpd[h]) and hpd[h] >= XB.ALPHA_ONE_SIDED)
        notes = []
        if c0.get("two_categories"):
            notes.append(f"주 CI {c0['cat_cluster']}, 보조 CI(대상 고정) {c0['cat_fixed']}, 약한 쪽으로 적었다")
        if uncorrected:
            notes.append(XB.UNCORRECTED_TXT)
        frags[h] = (sents, notes, directive)
        rr = r or {}
        rows.append(dict(hyp=h, gain=g, n=N_MAIN, x="CE2", scale="cm", subset="main", blind=BLIND[h], design=DESIGN, deviation=DEVIATION_MAIN,
                         category=cat, cat_cluster=c0.get("cat_cluster"), cat_fixed=c0.get("cat_fixed"), two_categories=bool(c0.get("two_categories")),
                         cat_main27=c0.get("cat_main27"), cat_CE1=cats["CE1"], cat_rel=cats["rel"], cat_partial=cats["partial"],
                         scale_robust=robust["scale"], definition_robust=robust["definition"], all_variants_pos=robust["all_pos"],
                         c2_coef_error_category=weaker(cats["CE2"], cats["CE1"]) if h == "XA-1" else "",
                         n_targets=rr.get("n_targets"), n_families=rr.get("n_families"), cluster_frac_lt3=rr.get("cluster_frac_lt3"),
                         rho_cell=rr.get("rho_cell"), rho_beq=rr.get("rho_beq"),
                         cluster_lo_cell=rr.get("cluster_lo_cell"), cluster_hi_cell=rr.get("cluster_hi_cell"),
                         cluster_lo_beq=rr.get("cluster_lo_beq"), cluster_hi_beq=rr.get("cluster_hi_beq"),
                         fixed_lo_cell=rr.get("fixed_lo_cell"), fixed_hi_cell=rr.get("fixed_hi_cell"),
                         fixed_lo_beq=rr.get("fixed_lo_beq"), fixed_hi_beq=rr.get("fixed_hi_beq"),
                         a100_n_targets=(a100[h] or {}).get("n_targets"), a100_fixed_lo_cell=(a100[h] or {}).get("fixed_lo_cell"),
                         a100_fixed_lo_beq=(a100[h] or {}).get("fixed_lo_beq"), a100_fixed_hi_cell=(a100[h] or {}).get("fixed_hi_cell"),
                         a100_fixed_hi_beq=(a100[h] or {}).get("fixed_hi_beq"),
                         p_one_cell=rr.get("cluster_p_one_cell"), p_one_beq=rr.get("cluster_p_one_beq"), p_one_max=rr.get("cluster_p_one_max"),
                         p_one_neg_cell=rr.get("cluster_p_one_neg_cell"), p_one_neg_beq=rr.get("cluster_p_one_neg_beq"),
                         p_one_neg_max=rr.get("cluster_p_one_neg_max"), direction="양" if dirs[h] > 0 else "음", p_dir=pdir[h],
                         holm_p=float(hp[h]) if h in hp else np.nan, holm_p_dir=float(hpd[h]) if h in hpd else np.nan, holm_m=HOLM_M if h in hp else np.nan,
                         holm_note=(f"{XB.UNCORRECTED_TXT}(방향 단측 p 의 Holm 보정 값 ≥ 0.025)" if uncorrected else ""),
                         sentence=_tag(sents, notes) if sents else "", directive=directive))
    # 공통 규칙: XA-3a(G_ML)와 XA-3(G_ML2)의 범주가 다르면 XA-3 에 '수축 잔여 의존'을 붙이고 XA-3 의 범주로 쓴다
    c3 = next(r for r in rows if r["hyp"] == "XA-3")
    c3a = next(r for r in rows if r["hyp"] == "XA-3a")
    dep = c3["category"] != c3a["category"]
    if dep:
        s3, n3, d3 = frags["XA-3"]
        frags["XA-3"] = (s3, n3 + ["수축 잔여 의존"], d3)
        c3["sentence"] = _tag(s3, n3 + ["수축 잔여 의존"]) if s3 else ""
    c3["shrinkage_dependence"] = "수축 잔여 의존" if dep else ""
    full = " ".join(_tag(frags[h][0], frags[h][1]) for h in ("XA-1", "XA-2", "XA-3") if frags[h][0])
    rows.append(dict(hyp="C2 문장", design=DESIGN, deviation=DEVIATION_MAIN, sentence=full,
                     directive=" / ".join(f"{h}: {frags[h][2]}" for h in ("XA-1", "XA-2", "XA-3") if frags[h][2]),
                     shrinkage_dependence="수축 잔여 의존" if dep else ""))
    return pd.DataFrame(rows)


# ================================================================ 7. 서술 표(XA-5·XA-6)
def gains_table(T, G) -> pd.DataFrame:
    """대상 × n × 이득의 점 추정과 대상 단위 CI(블록 재표집 2.5–97.5 백분위), RMSE(P0)(같은 분할)."""
    rows = []
    for nm in sorted(G):
        for n, gd in G[nm].items():
            for g in GAIN_NAMES:
                s = gd.get(g)
                if s is None:
                    continue
                lo, hi = _ci(s["dist"])[:2] if s["dist"] is not None else (np.nan, np.nan)
                lob, hib = _ci(s["dist_beq"])[:2] if s["dist_beq"] is not None else (np.nan, np.nan)
                rows.append(dict(target=nm, family=T[nm]["family"] or "", n=int(n), gain=g, g_cell=s["pt"], g_beq=s["pt_beq"],
                                 ci_lo_cell=lo, ci_hi_cell=hi, ci_lo_beq=lob, ci_hi_beq=hib, has_dist=s["dist"] is not None, n_splits=s["n_splits"],
                                 n_blocks_min=s.get("n_blocks_min"), p0_cell=s["p0"], p0_beq=s["p0_beq"],
                                 g_rel_cell=s["pt"] / s["p0"] if s["p0"] else np.nan, g_rel_beq=s["pt_beq"] / s["p0_beq"] if s["p0_beq"] else np.nan,
                                 ml2_from_p2_frac=s.get("ml2_from_p2_frac", np.nan)))
    return pd.DataFrame(rows)


def targets_table(T, G) -> pd.DataFrame:
    """대상별 계열, 모드, |A|, 진단값 수, CE1, CE2, CE3(= G_recal@전량, 서술), 묶음 소속."""
    rows = []
    for nm, t in sorted(T.items()):
        s = G.get(nm, {}).get(N_MAIN, {}).get("G_recal")
        memb = {sub: any(nm == m for m, _ in subset_members(T, sub)) for sub in SUBSETS}
        rows.append(dict(target=nm, family=t["family"] or "", plan_family7=bool(t["plan_family7"]), mode=t["mode"], point_only=t["point_only"],
                         has_ci=t["has_ci"], n_splits=t["n_splits"], nb_union=t["nb_union"], nA_min=t["nA_min"], nA_max=t["nA_max"],
                         n_diag=t["n_diag"], CE1=t["ce1"], CE1_const_over_splits=t["ce1_const"], CE2=t["ce2"],
                         CE3_cell=s["pt"] if s else np.nan, CE3_beq=s["pt_beq"] if s else np.nan, **{f"in_{k}": v for k, v in memb.items()}))
    return pd.DataFrame(rows)


def decomposition_table(T, G) -> pd.DataFrame:
    """XA-6 대상별 분해: G_recal, G_shr, G_W, G_ML, G_ML2, G_tot 의 점 추정과 G_recal / G_tot, G_shr / G_tot(|G_tot| ≥ 0.5 cm 인 가중만)."""
    rows = []
    for nm in sorted(G):
        for n, gd in G[nm].items():
            if "G_tot" not in gd:
                continue
            r = dict(target=nm, family=T[nm]["family"] or "", n=int(n))
            for w, suf in (("cell", ""), ("beq", "_beq")):
                tot = gd["G_tot"]["pt" + suf]
                for g in GAIN_NAMES:
                    r[f"{g}_{w}"] = gd[g]["pt" + suf] if g in gd else np.nan
                big = np.isfinite(tot) and abs(tot) >= XB.SMALL_EFFECT_CM
                r[f"recal_share_{w}"] = r[f"G_recal_{w}"] / tot if big else np.nan
                r[f"shr_share_{w}"] = r[f"G_shr_{w}"] / tot if big else np.nan
                r[f"tot_ge_0p5_{w}"] = bool(big)
            rows.append(r)
    return pd.DataFrame(rows)


def pool_table(tms, n_grid=N_GRID) -> pd.DataFrame:
    """XA-6 주 4지역 층화 평균(h42.pool_rows 감싸기, 같은 라벨 집합 대비): G_recal = Δ(P0 − P1), G_shr = Δ(P1 − P2), G_W = Δ(P1 − W)."""
    frames = []
    for n in n_grid:
        k = keys_for(n)
        for g in ("G_recal", "G_shr", "G_W"):
            a_, b_ = GAIN_PAIRS[g]
            rows, _ = XB.contrast_pool(tms, XB.MAIN4, k[a_], k[b_], label=f"{g}|n{nlab(n)}", kind="same", registered=len(XB.MAIN4))
            if rows:
                df = XB.clean_rows(rows)
                df.insert(0, "gain", g); df.insert(1, "n", int(n))
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ================================================================ 8. 재현 관문
def gate_wf4c(tms, units, nboot) -> dict:
    """관문 (1): h54.tests_wf4 의 WF4-c 계산(2657–2700행)을 그대로 다시 한다. 진단값(unit 순서), H.contrast(W@전량 − P0)의 셀 가중 분포,
    seed_of('wf4c', 대상 수)의 한 RandomState 를 대상 순서대로 쓴다. 반환 dict(rho, ci_lo, ci_hi, p_one, n_targets, names)."""
    gW = W.gk("W", -1, XB.LO, 0.0)
    diag = {}
    for u in units:
        diag.setdefault(f"{u['target']}|{u['mode']}", []).extend(u.get("diag") or [])
    rows = []
    for nm in sorted(tms):
        tm = tms[nm]
        dv = diag.get(nm, [])
        r = H.contrast(tm, gW, H.P0_GRP, return_dist=True)
        if not dv or r is None or not np.isfinite(r["delta"]):
            continue
        rows.append(dict(name=nm, abs_vals=np.array([v["abs_bias"] for v in dv], float), gain=-float(r["delta"]),
                         dist=(-np.asarray(r["dist"], float)) if "dist" in r else None))
    XB.boot_cache_clear()
    if len(rows) < 3:
        return dict(rho=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_one=np.nan, n_targets=len(rows), names=[q["name"] for q in rows])
    B = int(max(nboot, 1))
    rng = np.random.RandomState(XB.seed_of("wf4c", len(rows)))
    Dabs = np.zeros((B, len(rows))); Gm = np.zeros((B, len(rows)))
    for j, q in enumerate(rows):
        k = len(q["abs_vals"])
        pick = rng.randint(0, k, size=(B, k))
        Dabs[:, j] = q["abs_vals"][pick].mean(1)
        Gm[:, j] = q["dist"][:B] if (q["dist"] is not None and len(q["dist"]) >= B) else q["gain"]
    d0 = np.array([q["abs_vals"].mean() for q in rows]); g0 = np.array([q["gain"] for q in rows])
    rho = float(W._spearman_rows(d0[None], g0[None])[0])
    rho_b = W._spearman_rows(Dabs, Gm)
    return dict(rho=rho, ci_lo=float(np.nanpercentile(rho_b, 2.5)), ci_hi=float(np.nanpercentile(rho_b, 97.5)), p_one=float(np.mean(rho_b <= 0)),
                n_targets=len(rows), names=[q["name"] for q in rows])


def _load_wf0():
    name = "wf0_reanalysis"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, WF0_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def gate_wf0() -> dict:
    """관문 (2): wf0_reanalysis.py 의 입력과 표 구성(load, deltas)으로 라벨 전량 30대상 표를 다시 만들고 두 ρ 를 낸다.
    ρ(recal_gain, ml_gain_total) = Spearman(−P1, −최선 ML Δ), ρ(recal_gain, ml_beyond_p1) = Spearman(−P1, −(최선 ML Δ − P1))."""
    from scipy.stats import spearmanr
    mod = _load_wf0()
    R = mod.deltas(mod.load())
    y = R[R.n == -1].copy()
    recal = -y["P1"]
    best = y[mod.ML].min(axis=1)
    out = dict(n_targets=int(len(y)))
    for k, v in (("rho_total", -best), ("rho_beyond", -(best - y["P1"]))):
        z = pd.DataFrame(dict(a=recal.values, b=v.values)).dropna()
        out[k] = float(spearmanr(z.a, z.b).correlation)
        out[f"n_{k}"] = int(len(z))
    out["inputs"] = [str(Path(mod.LG).relative_to(ROOT)) if str(mod.LG).startswith(str(ROOT)) else str(mod.LG),
                     str(Path(mod.LGD).relative_to(ROOT)) if str(mod.LGD).startswith(str(ROOT)) else str(mod.LGD)]
    return out


def _eq2(v, ref) -> bool:
    return v is not None and np.isfinite(v) and abs(round(float(v), 2) - float(ref)) < 1e-9


def _file_refs() -> dict:
    """기존 기록 파일의 값(관문 표의 참고 열. 화면에 쓰지 않는다)."""
    ref = {}
    try:
        d = pd.read_csv(WF4_TESTS, low_memory=False)
        q = d[(d.test_id == "WF4-c") & (d.scope == "MEAN")]
        if len(q):
            ref.update(wf4c_rho=float(q.rho.iloc[0]), wf4c_ci_lo=float(q.ci_lo.iloc[0]), wf4c_ci_hi=float(q.ci_hi.iloc[0]))
    except (OSError, ValueError, KeyError):
        pass
    try:
        ref["wf0_rho_total"] = float(json.loads(WF0_META.read_text())["spearman_recal_gain_vs_ml_gain"]["rho"])
    except (OSError, ValueError, KeyError):
        pass
    try:
        d = pd.read_csv(C2_POSTHOC)
        q = d[(d.block == "WF0_30") & (d.x == "recal_gain(P0-P1)") & (d.y == "ml_beyond_p1(P1-bestML)")]
        if len(q):
            ref["wf0_rho_beyond"] = float(q.rho.iloc[0])
    except (OSError, ValueError, KeyError):
        pass
    return ref


def run_gates(tms, units, nboot, smoke=False, skip_wf0=False) -> tuple:
    """두 재현 관문. 반환 (관문 표, 요약 dict(passed, gate1, gate2)). 스모크에서만 WF4-c 의 CI 를 대조하지 않는다(ρ 점 추정만). 스모크가 아니면
    재표집 수와 관계없이 CI 를 대조한다(본 실행·관문만·합침은 run() 이 재표집 10,000회를 강제한다)."""
    rows = []
    ref = _file_refs()
    g1 = gate_wf4c(tms, units, nboot)
    full = not bool(smoke)
    for k in ("rho", "ci_lo", "ci_hi"):
        comp = (k == "rho") or full
        fk = f"wf4c_{k}"
        rows.append(dict(gate="(1) WF4-c", quantity=k, value=g1[k], ref_plan=GATE1_REF[k], compared=comp,
                         equal_2dp=_eq2(g1[k], GATE1_REF[k]) if comp else None, ref_file=ref.get(fk, np.nan),
                         file_abs_diff=abs(g1[k] - ref[fk]) if (fk in ref and np.isfinite(g1[k])) else np.nan,
                         n_targets=g1["n_targets"], nboot=int(nboot), note="" if comp else "스모크: CI 를 대조하지 않는다(ρ 점 추정만)"))
    if not skip_wf0:
        g2 = gate_wf0()
        for k in ("rho_total", "rho_beyond"):
            fk = f"wf0_{k}"
            rows.append(dict(gate="(2) WF0", quantity=k, value=g2[k], ref_plan=GATE2_REF[k], compared=True, equal_2dp=_eq2(g2[k], GATE2_REF[k]),
                             ref_file=ref.get(fk, np.nan), file_abs_diff=abs(g2[k] - ref[fk]) if (fk in ref and np.isfinite(g2[k])) else np.nan,
                             n_targets=g2[f"n_{k}"], nboot=0, note=";".join(g2["inputs"])))
    df = pd.DataFrame(rows)
    comp = df[df.compared.astype(bool)]
    s = dict(passed=bool(len(comp) and comp.equal_2dp.astype(bool).all()),
             gate1=bool(df[(df.gate == "(1) WF4-c") & df.compared.astype(bool)].equal_2dp.astype(bool).all()),
             gate2=None if skip_wf0 else bool(df[df.gate == "(2) WF0"].equal_2dp.astype(bool).all()),
             gate1_n_targets=int(g1["n_targets"]), gate1_targets=g1["names"], skipped_wf0=bool(skip_wf0), smoke=bool(smoke))
    return df, s


def c2_ref_values(path=C2_POSTHOC) -> dict:
    """내부 일치 점검의 기준: derived_c2_posthoc_rho.csv 의 WF4c_28 블록(이미 열람된 28대상 셀 가중 점 추정, 해석적 Spearman).
    반환 {이득: (ρ, 대상 수)}. 값은 코드 안에서만 쓰고 화면·메타에 쓰지 않는다. 읽을 수 없으면 빈 dict."""
    try:
        d = pd.read_csv(path)
    except (OSError, ValueError):
        return {}
    out = {}
    for g, y in C2_REF_ROWS.items():
        q = d[(d.block == C2_REF_BLOCK) & (d.x == C2_REF_X) & (d.y == y)]
        if len(q) == 1:
            out[g] = (float(q.rho.iloc[0]), int(q.n_targets.iloc[0]))
    return out


def check_wf4c28(sp, ref, require_all=True) -> dict:
    """내부 일치 점검(등록 관문이 아니다). 주 묶음(28대상)·CE2·cm·n 전량 행의 셀 가중 ρ 점 추정(G_tot, G_recal, G_W, G_ML)과 대상 수가
    WF4c_28 기록(ref)과 1e-9 안에서 같은지 본다. XA 자신의 경로(gains, h40.contrast 이득, ce_values 의 CE2, wspearman)를 실제 자료에서 시험한다.
    require_all = False(작업 나눔의 조각 하나)이면 그 조각에 없는 행은 건너뛴다. 반환 dict 에는 통과 여부와 개수, 실패한 이득 이름만 담는다(값 없음)."""
    if not ref or set(ref) != set(C2_REF_ROWS):
        return dict(passed=False, n_compared=0, n_absent=len(C2_REF_ROWS), gains_failed=[], reason="기준 행 없음")
    compared, absent, failed = 0, 0, []
    for g in C2_REF_ROWS:
        rv, nt = ref[g]
        r = _rec(sp, g, "CE2", "cm", N_MAIN, "main")
        if r is None:
            absent += 1
            continue
        compared += 1
        ok = (r.get("status") == "ok" and int(r.get("n_targets", -1)) == int(nt) and _fin(r.get("rho_cell"))
              and abs(float(r["rho_cell"]) - float(rv)) <= C2_REF_TOL)
        if not ok:
            failed.append(g)
    passed = (not failed) and (absent == 0 or not require_all)
    reason = "" if passed else ("행 불일치" if failed else "행 없음")
    return dict(passed=bool(passed), n_compared=int(compared), n_absent=int(absent), gains_failed=failed, reason=reason)


# ================================================================ 9. 세기(라벨 값 미사용)
def _npz_groups(path) -> set:
    """조각 npz 의 키 묶음(곡선 키)만 읽는다. 블록 SSE·셀 수 배열은 열지 않는다."""
    out = set()
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        for i, _u in enumerate(meta["units"]):
            for s in z[f"u{i}_keys"]:
                k = json.loads(str(s))
                out.add((k[0], k[1], str(k[2]), k[3], int(k[4]), float(k[7])))
    return out


def count_only(a) -> int:
    """구조만 센다: 조각 수, 대상, 계열, 분할, 진단값 수, |A|, (이득, n)별 두 팔의 키가 모두 있는 대상 수, Spearman 행 수와 예상 시간.
    unit.json 에서는 target, mode, split, n_A, diag 길이, valid, dup_of 만 읽는다. 라벨 값과 RMSE 는 읽지도 쓰지도 않는다."""
    sh = XB.find_shards(a.wf4_shards, a.wf4_tag)
    print(f"[xa 세기] 조각 {len(sh)}개(tag {a.wf4_tag}, {a.wf4_shards})", flush=True)
    if not sh:
        return 1
    info = {}
    for s_ in sh:
        u = json.loads(Path(s_["unit"]).read_text())
        nm = f"{u['target']}|{u['mode']}"
        d = info.setdefault(nm, dict(splits=[], nA=[], n_diag=0, groups=set(), valid=0))
        d["splits"].append(int(u["split"])); d["n_diag"] += len(u.get("diag") or [])
        if u.get("n_A") is not None:
            d["nA"].append(int(u["n_A"]))
        d["valid"] += int(bool(u.get("valid", True)) and int(u.get("dup_of", -1)) < 0)
        d["groups"] |= _npz_groups(s_["npz"])
    T = {nm: dict(has_ce2=d["n_diag"] > 0, nA_min=min(d["nA"]) if d["nA"] else None) for nm, d in info.items()}
    print(f"[xa 세기] 저장소(대상, 모드) {len(info)}개, 진단값이 있는 대상 {sum(t['has_ce2'] for t in T.values())}개", flush=True)
    for sub in SUBSETS:
        mem = subset_members(T, sub)
        fams = sorted({f for _, f in mem}, key=_fam_key)
        print(f"  묶음 {sub}({SUBSET_LABEL[sub]}): 대상 {len(mem)}, 계열 {len(fams)} [{', '.join(fams)}]", flush=True)
    for f in FAMILY_ORDER8:
        mem = [nm for nm in sorted(info) if family_of(nm) == f and T[nm]["has_ce2"]]
        if mem:
            nA = [info[nm]["nA"] for nm in mem]
            print(f"  계열 {f}: 대상 {len(mem)} · 분할 수 {min(len(info[nm]['splits']) for nm in mem)}–{max(len(info[nm]['splits']) for nm in mem)}"
                  f" · |A| {min(min(v) for v in nA if v)}–{max(max(v) for v in nA if v)} · 진단값 {min(info[nm]['n_diag'] for nm in mem)}–"
                  f"{max(info[nm]['n_diag'] for nm in mem)}개", flush=True)
    main = [nm for nm, _ in subset_members(T, "main")]
    for n in N_GRID:
        k = keys_for(n)
        parts = []
        for g in GAIN_NAMES:
            prs = [GAIN_PAIRS[g]] if g in GAIN_PAIRS else list(ML2_PAIRS)
            c = sum(all(k[p[0]] in info[nm]["groups"] and k[p[1]] in info[nm]["groups"] for p in prs) for nm in main)
            parts.append(f"{g} {c}")
        print(f"  n {nlab(n)}: 주 묶음에서 두 팔의 키가 있는 대상 수 · {' · '.join(parts)}", flush=True)
    nspec = len(spec_list())
    B = int(a.nboot)
    est = nspec * 4 * (B / 10000.0) * 0.2
    print(f"[xa 세기] 적합 0건. Spearman 행 {nspec}개(이득 {len(GAIN_NAMES)} × 계수 오차 {len(X_KINDS)} × 척도 {len(SCALES)} × n {len(N_GRID)} × 묶음 "
          f"{len(SUBSETS)}), 행마다 CI 2종 × 가중 2종, 재표집 {B}회. 예상 계산 시간 약 {est / 60:.0f}분(1코어, 추정)", flush=True)
    return 0


# ================================================================ 10. 실행
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="XA_c2_gain_decomposition(계획 2.1): 재보정 몫·수축 잔여·ML 몫과 계수 오차 순위상관. 적합 없음")
    ap.add_argument("--wf4-shards", "--shards", dest="wf4_shards", default=str(WF4_SHARDS), help="WF4 조각 폴더")
    ap.add_argument("--wf4-tag", default=WF4_TAG)
    ap.add_argument("--nboot", type=int, default=XB.NBOOT)
    ap.add_argument("--threads", type=int, default=XA_MAX_THREADS,
                    help=f"BLAS 스레드(1–{XA_MAX_THREADS}, 계획 2.1. --allow-local 이어도 상한은 같다. 동시 조각은 이 예산을 나눠 쓴다)")
    ap.add_argument("--out-root", default="", help="산출 뿌리(기본 data/processed/xbatch, 스모크는 data/processed/xbatch/_smoke)")
    ap.add_argument("--count-only", action="store_true", help="구조만 센다(라벨 값 미사용)")
    ap.add_argument("--smoke", action="store_true", help=f"재표집 {SMOKE_NBOOT}회, 출력 제한, _smoke 아래 봉인 폴더")
    ap.add_argument("--smoke-nboot", type=int, default=SMOKE_NBOOT)
    ap.add_argument("--skip-gates", action="store_true", help="관문 생략(--smoke 와 함께만, 합성 자료 시험용)")
    ap.add_argument("--gates-only", action="store_true", help="재현 관문만 계산한다")
    ap.add_argument("--shard", default="", help="K/N: Spearman 행 명세 가운데 색인 % N == K 만 계산해 봉인 폴더에 조각 표로 쓴다")
    ap.add_argument("--merge-shards", type=int, default=0, help="N: 조각 표 N개를 합쳐 판정 단계를 마친다")
    ap.add_argument("--allow-local", action="store_true", help="본 실행을 로컬에서 허용한다(WF_RESCALE=1 과 같은 효과)")
    ap.add_argument("--mem-min", type=float, default=30.0, help="시작 전 가용 메모리 하한(GB, 계획 1절 로컬 자원)")
    ap.add_argument("--mem-wait", type=float, default=3600.0, help="가용 메모리가 하한 아래일 때 기다릴 최대 시간(초, 60 s 간격으로 다시 본다)")
    ap.add_argument("--mem-max", type=float, default=10.0, help="프로세스 주소 공간 상한(GB). 0 이면 걸지 않는다(시험이 같은 프로세스에서 부를 때)")
    a = ap.parse_args(argv)
    if not (1 <= a.threads <= XA_MAX_THREADS):
        ap.error(f"--threads 는 1–{XA_MAX_THREADS} 이다(계획 2.1 'CPU 4스레드 이하', --allow-local 이어도 같다)")
    if a.skip_gates and not a.smoke:
        ap.error("--skip-gates 는 --smoke 와 함께만 쓴다")
    if a.shard:
        try:
            k, n = (int(v) for v in a.shard.split("/"))
        except ValueError:
            ap.error("--shard 는 K/N 형식이다")
        if not (0 <= k < n):
            ap.error("--shard K/N 에서 0 ≤ K < N 이어야 한다")
        a.shard_k, a.shard_n = k, n
    else:
        a.shard_k, a.shard_n = 0, 1
    if a.shard and a.merge_shards:
        ap.error("--shard 와 --merge-shards 는 함께 쓰지 않는다")
    return a


def _data_sha(shards_dir, tag) -> str:
    """WF4 조각 판 식별: 조각 파일 sha256 앞 16자 목록의 sha256 앞 16자."""
    import hashlib
    h = hashlib.sha256()
    for s_ in XB.find_shards(shards_dir, tag):
        for k in ("unit", "runs", "npz"):
            h.update(f"{Path(s_[k]).name}:{XB.sha256_file(s_[k], 16)}\n".encode())
    return h.hexdigest()[:16]


def run_cfg(a, nboot, data_sha) -> dict:
    return XB.make_unit_cfg(TAG, data_sha=data_sha, nboot=int(nboot), n_grid=list(N_GRID), gains=list(GAIN_NAMES), x_kinds=list(X_KINDS),
                            scales=list(SCALES), subsets=list(SUBSETS), a_min=A_MIN, cluster=[CLUSTER_MIN_FAMILIES, CLUSTER_MAX_FRAC],
                            families=list(FAMILY_ORDER8), n_specs=len(spec_list()), code_sha=XB.code_sha(__file__))


def cluster_structure(T, nboot) -> dict:
    """묶음별 계열 수와 3계열 미만 재표집 비율(계열 구조만 쓴다. 라벨 값과 무관하다)."""
    out = {}
    for sub in SUBSETS:
        fams = [f for _, f in subset_members(T, sub)]
        if fams:
            cb = cluster_boot(fams, nboot)
            out[sub] = dict(n_families=len(cb["families"]), frac_lt3=cb["frac_lt3"], insufficient=cb["insufficient"])
    return out


def _part_names(k, n) -> tuple:
    """작업 나눔(--shard K/N)의 부분 표 이름. 계획 1절의 적합 조각 이름(<tag>__cpu__<대상>__<모드>__s<분할>…)과 겹치지 않게 따로 짓는다."""
    base = f"xa_spearman_part{int(k)}of{int(n)}"
    return base + ".csv", base + ".json"


def ensure_nice(target=XA_NICE) -> int:
    """프로세스 nice 값을 target 이상으로 올린다(계획 2.1 'nice 10'). 이미 높으면 그대로 둔다(낮추지 않는다). 반환: 현재 nice 값."""
    try:
        cur = os.nice(0)
        if cur < int(target):
            cur = os.nice(int(target) - cur)
        return int(cur)
    except OSError:
        return int(os.nice(0))


def thread_cap(threads) -> int:
    """메타에 적는 스레드 상한 = min(--threads, 4, OMP_NUM_THREADS). 실제 상한은 모듈 첫머리의 clamp_thread_env 가 numpy 를 부르기 전에
    환경 변수로 건다(threadpoolctl 2.2 는 이 서버의 OpenBLAS 에서 판 조회 예외를 내므로 실행 중 조정은 쓰지 않는다)."""
    try:
        env_t = int(os.environ.get("OMP_NUM_THREADS", threads))
    except ValueError:
        env_t = int(threads)
    return max(1, min(int(threads), XA_MAX_THREADS, env_t))


def _read_parts(sdir, n, cfg) -> pd.DataFrame:
    """조각 표 N개를 읽어 합친다. 봉인 폴더의 파일을 코드가 읽기만 하고 화면에는 쓰지 않는다(열람이 아니다). 설정 해시와 명세 덮개를 확인한다."""
    frames = []
    for k in range(int(n)):
        fcsv, fjs = _part_names(k, n)
        pj, pc = Path(sdir) / fjs, Path(sdir) / fcsv
        if not (pj.exists() and pc.exists()):
            raise SystemExit(f"[xa] 조각 표 없음: {fcsv}")
        meta = json.loads(pj.read_text())
        if meta.get("cfg_hash") != XB.cfg_hash(cfg):
            raise SystemExit(f"[xa] 조각 표 {fcsv} 의 설정 해시가 다르다")
        frames.append(pd.read_csv(pc))
    df = pd.concat(frames, ignore_index=True).sort_values("spec_idx").reset_index(drop=True)
    want = [s["spec_idx"] for s in spec_list()]
    if df.spec_idx.tolist() != want:
        raise SystemExit(f"[xa] 조각 표가 명세 {len(want)}행을 정확히 한 번씩 덮지 않는다({len(df)}행)")
    return df


def _write_open(path, obj, allowed=None):
    """봉인 밖 파일 쓰기(관문 표, 메타). 허용 경로 안인지 확인한다."""
    p = Path(path)
    XB.check_out_dir(p.parent, allowed)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(tmp, index=False)
    else:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    os.replace(tmp, p)
    return p


def run(a, mode) -> int:
    t0 = time.time()
    if mode == "run" and int(a.nboot) != XB.NBOOT:                          # 본 실행·관문만·합침은 등록 재표집 수로만(계획 2.1 '10,000회')
        raise SystemExit(f"[xa] 스모크가 아닌 실행은 재표집 {XB.NBOOT}회로만 한다(계획 2.1). --nboot {int(a.nboot)} 을 거부한다")
    threads, nboot = XB.local_limits(a.threads, a.nboot, argv=a._argv)
    threads = min(int(threads), XA_MAX_THREADS)                             # --allow-local 이어도 4 이하(계획 2.1)
    if mode == "smoke":
        nboot = min(int(nboot), int(a.smoke_nboot))
    root = Path(a.out_root) if a.out_root else (XB.XBATCH_ROOT / "_smoke" if mode == "smoke" else XB.XBATCH_ROOT)
    if not root.is_absolute():
        root = ROOT / root
    exp_dir = root / EXP_NAME
    XB.check_out_dir(exp_dir)
    sdir = XB.sealed_dir(EXP_NAME, root)
    gb = XB.require_memory(a.mem_min, a.mem_wait) if a.mem_min > 0 else XB.mem_available_gb()
    if a.mem_max > 0:
        XB.limit_memory(a.mem_max)
    cap = getattr(a, "_thread_cap", threads)
    nice_now = int(os.nice(0))
    print(f"[xa] {mode} · 재표집 {nboot} · 스레드 {os.environ.get('OMP_NUM_THREADS')}(상한 {cap}) · nice {nice_now} · 가용 메모리 {gb:.0f} GB · "
          f"산출 {exp_dir}", flush=True)
    data_sha = _data_sha(a.wf4_shards, a.wf4_tag)
    cfg = run_cfg(a, nboot, data_sha)
    tms, units = load_wf4_tms(a.wf4_shards, a.wf4_tag, nboot)
    print(f"[xa] WF4 저장소 {len(tms)}개, unit {len(units)}개 적재", flush=True)
    meta = dict(exp=EXP_NAME, plan=XB.PLAN_DOC, plan_commit=XB.PLAN_COMMIT, mode=mode, nboot=int(nboot), threads=os.environ.get("OMP_NUM_THREADS"),
                threads_cap=int(cap), nice=nice_now, mem_available_gb_start=round(float(gb), 1) if np.isfinite(gb) else None,
                wf4_shards=str(a.wf4_shards), wf4_tag=a.wf4_tag, wf4_data_sha=data_sha, cfg=cfg, cfg_hash=XB.cfg_hash(cfg),
                code_sha=XB.code_sha(__file__), code_sha_core=XB.code_sha(XB.__file__), frozen=dict(XB.FROZEN_SHA16), n_fits=0,
                design=DESIGN, shard=a.shard or "", merge_shards=int(a.merge_shards))
    # ---- 재현 관문(계획 2.1: 하나라도 소수 둘째 자리까지 다르면 멈춘다)
    if a.skip_gates:
        gsum = dict(passed=None, skipped=True)
        print("[xa] 관문 생략(--skip-gates, 스모크 전용)", flush=True)
    else:
        gdf, gsum = run_gates(tms, units, nboot, smoke=(mode == "smoke"))
        _write_open(exp_dir / "xa_gates.csv", gdf)
        print(f"[xa] 관문 (1) WF4-c: {'통과' if gsum['gate1'] else '실패'}(대상 {gsum['gate1_n_targets']}"
              f"{', 스모크라 CI 는 대조하지 않음' if mode == 'smoke' else ''}) · 관문 (2) WF0: {'통과' if gsum['gate2'] else '실패'}", flush=True)
    meta["gates"] = gsum
    if not a.skip_gates and not gsum["passed"] and mode != "smoke":
        meta.update(status="관문 실패로 멈춤", elapsed_s=round(time.time() - t0, 1), max_rss_mb=XB.max_rss_mb())
        _write_open(exp_dir / "xa_meta.json", meta)
        print("[xa] 관문을 통과하지 못했다. 계획 2.1 에 따라 XA 를 멈춘다(관문 표: xa_gates.csv)", flush=True)
        return 2
    if a.gates_only:
        meta.update(status="관문만", elapsed_s=round(time.time() - t0, 1), max_rss_mb=XB.max_rss_mb())
        _write_open(exp_dir / "xa_meta.json", meta)
        return 0 if (gsum.get("passed") or mode == "smoke") else 2
    # ---- 이득 분포와 계수 오차
    T = target_table(tms, units)
    for nm, t in T.items():
        t["ce2_b"] = ce2_boot(t["abs_vals"], nm, nboot) if t["has_ce2"] else None
    G, chk = all_gains(tms, T)
    ok_add = chk[chk.additive_checked.astype(bool)]
    add_pass = bool(len(ok_add) and ok_add.additive_same_splits.astype(bool).all() and (ok_add.additive_max_abs <= 1e-9).all())
    meta.update(additivity=dict(n_checked=int(len(ok_add)), n_diff_splits=int((~ok_add.additive_same_splits.astype(bool)).sum()),
                                max_abs=float(ok_add.additive_max_abs.max()) if len(ok_add) else None, passed=add_pass))
    print(f"[xa] 가법성 점검(P0 − W = (P0 − P1) + (P1 − W), 재표집마다): 대상·n {len(ok_add)}개, {'통과' if add_pass else '불일치 있음'}", flush=True)
    specs = spec_list()
    meta["subsets"] = {sub: dict(n_targets=len(subset_members(T, sub)), families=sorted({f for _, f in subset_members(T, sub)}, key=_fam_key),
                                 registered=SUBSET_REGISTERED[sub]) for sub in SUBSETS}
    # ---- Spearman(조각 나눔·합침)
    if a.merge_shards:
        sp = _read_parts(sdir, a.merge_shards, cfg)
        print(f"[xa] 조각 표 {a.merge_shards}개 합침: 행 {len(sp)}", flush=True)
    else:
        mine = [s for s in specs if s["spec_idx"] % a.shard_n == a.shard_k]
        t1 = time.time()
        sp = spearman_table(T, G, nboot, mine)
        print(f"[xa] Spearman 행 {len(sp)}/{len(specs)}개 계산({time.time() - t1:.0f} s)", flush=True)
    meta_name = f"xa_meta_part{a.shard_k}of{a.shard_n}.json" if a.shard else "xa_meta.json"
    # ---- 내부 일치 점검(등록 관문 아님): 주 묶음 28대상의 점 추정이 이미 열람된 WF4c_28 기록과 같아야 한다. 화면·메타에는 통과 여부와 개수만
    if a.skip_gates:
        meta["internal_check_wf4c28"] = dict(skipped=True)
    else:
        ic = check_wf4c28(sp, c2_ref_values(), require_all=not a.shard)
        meta["internal_check_wf4c28"] = ic
        print(f"[xa] 내부 일치 점검(주 묶음 28대상 점 추정과 WF4c_28 기록, 등록 관문 아님): {'통과' if ic['passed'] else '실패'}"
              f"(비교 {ic['n_compared']}행)", flush=True)
        if not ic["passed"]:
            meta.update(status="내부 일치 점검 실패로 멈춤", elapsed_s=round(time.time() - t0, 1), max_rss_mb=XB.max_rss_mb())
            _write_open(exp_dir / meta_name, meta)
            print("[xa] 내부 일치 점검을 통과하지 못했다. 결과 표를 쓰지 않고 멈춘다", flush=True)
            return 2
    if a.shard:
        fcsv, fjs = _part_names(a.shard_k, a.shard_n)
        XB.write_sealed(EXP_NAME, {fcsv: sp, fjs: dict(cfg_hash=XB.cfg_hash(cfg), shard=a.shard, n_rows=int(len(sp)))}, root=root)
        meta.update(status=f"조각 {a.shard} 완료", elapsed_s=round(time.time() - t0, 1), max_rss_mb=XB.max_rss_mb())
        _write_open(exp_dir / meta_name, meta)
        return 0
    # ---- 범주, 해석 조각, 서술 표(봉인)
    ct = category_table(sp)
    vt = verdict_table(sp, ct)
    tables = {"xa_targets.csv": targets_table(T, G), "xa_gains.csv": gains_table(T, G), "xa_spearman.csv": sp, "xa_categories.csv": ct,
              "xa_hyp.csv": vt, "xa_decomp.csv": decomposition_table(T, G), "xa_pool_main4.csv": pool_table(tms), "xa_additivity.csv": chk}
    XB.boot_cache_clear()
    recs = XB.write_sealed(EXP_NAME, tables, root=root)
    meta.update(status="완료", sealed=[dict(file=r["file"], rows=r["rows"], sha256=r["sha256"]) for r in recs],
                cluster_frac_lt3_by_subset=cluster_structure(T, nboot), ml2_split_mismatch=int(chk.ml2_split_mismatch.astype(bool).sum()),
                elapsed_s=round(time.time() - t0, 1), max_rss_mb=XB.max_rss_mb())
    _write_open(exp_dir / "xa_meta.json", meta)
    print(f"[xa] 완료: 봉인 표 {len(recs)}개, 메타 {exp_dir / 'xa_meta.json'} · {time.time() - t0:.0f} s · 최대 RSS {XB.max_rss_mb()} MB", flush=True)
    return 0


def main(argv=None) -> int:
    a = parse_args(argv)
    a._argv = list(argv) if argv is not None else list(sys.argv)
    mode = "count" if a.count_only else ("smoke" if a.smoke else "run")
    XB.guard(mode, a._argv)
    ensure_nice(XA_NICE)                                                    # 계획 2.1 'nice 10'(낮추지 않는다)
    a._thread_cap = thread_cap(a.threads)
    with XB.restricted_output():
        if mode == "count":
            return count_only(a)
        return run(a, mode)


if __name__ == "__main__":
    sys.exit(main())
