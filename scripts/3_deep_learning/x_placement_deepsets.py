"""S9-DS: DeepSets 배치 정책(계획 2.4 XD-learn). 로컬 GPU 0–4 가운데 사용 직전 nvidia-smi 로 빈 것을 쓰고(CUDA_VISIBLE_DEVICES 에 그 번호 하나,
CUDA_DEVICE_ORDER=PCI_BUS_ID 로 nvidia-smi 번호 = CUDA 번호), 결정적 설정(CUBLAS_WORKSPACE_CONFIG=:4096:8 로 덮어쓰기,
torch.use_deterministic_algorithms(True), cuDNN 결정성, seed 고정)으로 학습·선택한다. 설정이 걸리지 않으면 거부한다.

계획: docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md(개정 1, T0 = git 2678100) 2.4절 'S9-DS', '조기 종료와 변형 선택', '정책(탐욕)', '동결'.
구현 결정: docs/research/2026-10-04/impl_notes/x_placement_policy.md(DeepSets 절). 학습 자료·특징·교차검증 접기·탐욕 정책·색인 파일은
x_placement_policy.py 의 함수를 그대로 쓴다(두 변형 S9-GBM, S9-DS 가 같은 자료와 같은 평가 규칙을 쓴다).

구성
  원소 특징 = [Z_c(25), s_c 표준화, 블록 셀 비율, DI_c, 같은 집합 안 최근접 라벨 거리](29), 문맥 = [후보 수, 블록 수, 유효 블록 수,
  공변량 블록 사이 비율, 원천 대비 공변량 차이, 집합 크기 n](6). 원소·문맥 특징은 학습 자료에서 맞춘 평균·표준편차로 표준화한다.
  φ = MLP(원소 특징 → 128 → 128), 합·평균 풀링 결합(256), ρ = MLP(256 + 문맥 → 128 → 1). 손실 = 같은 (과제, 분할, n) 묶음 안 모든 집합 쌍의
  RankNet 형 순위 손실(효용 U 가 작은 집합의 점수가 작도록). AdamW(학습률 1e-3, 가중 감쇠 1e-4), 한 걸음 = 묶음 8개, 최대 50 epoch, seed 0·1·2 평균.
  조기 종료 = 알래스카 하위 지역 하나 제외 교차검증(AL-k 와 알래스카 x 함께 제외, 검증 = AL-k)에서 세 seed 평균 점수의 개발 교차검증 효용이
  가장 작은 epoch. 최종 적합 = 개발 과제 전부, 그 epoch 수, seed 0·1·2.
  정책 = x_placement_policy.greedy_policy(첫 셀 = 정책 seed, 5개 단위, farthest-first 상위 300 ∪ 고정 seed 무작위 200)에서 L ∪ {c} 의 원소 특징을
  후보마다 다시 계산해 한 번에 평가한다.

명령(로컬 GPU. 학습 자료 = 작업 R1a 의 xd_learn 조각, 알래스카 계열만. 화면에는 효용 값을 쓰지 않는다)
  XBATCH_GPU=1 nice -n 10 python3 scripts/3_deep_learning/x_placement_deepsets.py --train --export
  XBATCH_GPU=1 python3 scripts/3_deep_learning/x_placement_deepsets.py --jaccard-gpu 3       # 동결 뒤 다른 GPU 에서 선택을 다시 만들어 겹침 서술
  XBATCH_GPU=1 python3 scripts/3_deep_learning/x_placement_deepsets.py --xd5                 # XD-4 표 열람 뒤: 계열 하나 제외(봉인 폴더)
  GPU 를 쓸 수 없으면 --device cpu(4스레드, nice 10, 계획 7.2-6 의 대체 경로)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XBATCH_GPU", "1")                                    # xbatch_core 가 CUDA_VISIBLE_DEVICES 를 비우지 않게 한다
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import x_placement_policy as XP                                             # noqa: E402  (xbatch_core 를 먼저 적재한다)

XB = XP.XB
import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402

HIDDEN = 128
EPOCHS_MAX = 50
DS_SEEDS = (0, 1, 2)
LR = 1e-3
WEIGHT_DECAY = 1e-4
GROUPS_PER_STEP = 8
GPU_ALLOWED = (0, 1, 2, 3, 4)
GPU_FREE_MIB = 500
CTX_NAMES = list(XP.CTX_FEATS) + ["n"]
PRED_CHUNK = 4096


# ================================================================ 장치
def gpu_table() -> list:
    """nvidia-smi 의 GPU 상태 [(번호, 사용 메모리 MiB, 사용률 %, 이름, 드라이버)]와 계산 프로세스가 있는 GPU uuid 집합."""
    q = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,name,driver_version,uuid",
                                 "--format=csv,noheader,nounits"], text=True, timeout=30)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True, timeout=30)
    busy = {v.strip() for v in apps.splitlines() if v.strip()}
    out = []
    for line in q.strip().splitlines():
        f = [v.strip() for v in line.split(",")]
        out.append(dict(index=int(f[0]), mem_mib=float(f[1]), util=float(f[2]), name=f[3], driver=f[4], busy=f[5] in busy))
    return out


def pick_gpu(allowed=GPU_ALLOWED, max_used_mib=GPU_FREE_MIB, exclude=()) -> dict | None:
    """허용 GPU(0–4) 가운데 사용 메모리가 작고 계산 프로세스가 없는 첫 GPU. 없으면 None."""
    try:
        tab = gpu_table()
    except Exception:                                                     # noqa: BLE001
        return None
    for g in tab:
        if g["index"] in allowed and g["index"] not in set(exclude) and g["mem_mib"] <= max_used_mib and not g["busy"]:
            return g
    return None


CUBLAS_CFG = ":4096:8"                                                      # 계획 1절 플랫폼: CUBLAS_WORKSPACE_CONFIG=:4096:8
DEVICE_ORDER = "PCI_BUS_ID"                                                 # nvidia-smi 의 GPU 번호와 CUDA 의 장치 번호를 같게 한다


def setup_device(device="auto", gpu=None, seed=0, threads=4, exclude=()) -> dict:
    """장치 결정과 결정적 설정(계획 1절 플랫폼: 로컬 GPU 0–4, CUBLAS_WORKSPACE_CONFIG=:4096:8, torch.use_deterministic_algorithms(True)).
    CUDA 문맥을 만들기 전에 CUDA_VISIBLE_DEVICES(허용 GPU 하나), CUDA_DEVICE_ORDER=PCI_BUS_ID(nvidia-smi 번호 = CUDA 번호),
    CUBLAS_WORKSPACE_CONFIG 을 정한다(환경에 다른 값이 있어도 덮어쓴다). 설정 뒤 결정적 알고리즘·보이는 장치 수를 확인하고 어긋나면 거부한다.
    반환 = 동결 기록용 판·장치 정보."""
    ginfo = None
    if device in ("auto", "cuda"):
        if gpu is not None:
            if int(gpu) not in GPU_ALLOWED:
                raise SystemExit(f"[거부] GPU {gpu} 는 허용 목록 {GPU_ALLOWED} 밖이다(계획 1절: 로컬 GPU 0–4)")
            ginfo = next((g for g in gpu_table() if g["index"] == int(gpu)), None)
            if ginfo is None:
                raise SystemExit(f"[거부] GPU {gpu} 가 nvidia-smi 목록에 없다")
        else:
            ginfo = pick_gpu(exclude=exclude)
        if ginfo is None and device == "cuda":
            raise SystemExit("[거부] 빈 GPU(0–4)가 없다. 대체 경로는 --device cpu(계획 7.2-6)")
    if ginfo is not None:
        if int(ginfo["index"]) not in GPU_ALLOWED:
            raise SystemExit(f"[거부] GPU {ginfo['index']} 는 허용 목록 {GPU_ALLOWED} 밖이다")
        os.environ["CUDA_DEVICE_ORDER"] = DEVICE_ORDER
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(ginfo["index"]))
        dev = "cuda"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        dev = "cpu"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_CFG                         # setdefault 가 아니다: 다른 값이 있어도 계획 값으로
    info = XB.torch_deterministic(seed)
    import torch
    torch.set_num_threads(int(threads))
    if not torch.are_deterministic_algorithms_enabled() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_CFG:
        raise SystemExit("[거부] 결정적 설정이 걸리지 않았다(use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG)")
    gpu_name_torch = None
    if dev == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("[거부] CUDA 를 쓸 수 없다")
        if torch.cuda.device_count() != 1:
            raise SystemExit(f"[거부] 보이는 CUDA 장치가 {torch.cuda.device_count()}개다(GPU 하나만 쓴다)")
        gpu_name_torch = torch.cuda.get_device_name(0)
    info.update(device=dev, gpu=None if ginfo is None else int(ginfo["index"]), gpu_name=None if ginfo is None else ginfo["name"],
                gpu_name_torch=gpu_name_torch, driver=None if ginfo is None else ginfo["driver"], threads=int(threads),
                cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""), deterministic=bool(torch.are_deterministic_algorithms_enabled()),
                device_order=os.environ.get("CUDA_DEVICE_ORDER", "") if dev == "cuda" else "", gpu_allowed=list(GPU_ALLOWED))
    return info


# ================================================================ 자료
def assemble(df, el, tasks=None) -> list:
    """학습 자료 표(load_learning)와 원소 자료 → 묶음 목록. 묶음 = (과제, 분할, n): X(S, n, 29), C(S, 6), U(S), j(S)."""
    groups = []
    for (t, sp, n), g in df.groupby(["task", "split", "n"], sort=True):
        if tasks is not None and t not in set(tasks):
            continue
        g = g.sort_values("j")
        idx = el["idx"][(t, int(sp), int(n))][g["j"].values]
        static = el["static"][(t, int(sp))]
        Xel = np.stack([XP.elements_of(static, row) for row in idx]).astype(np.float32)
        C = np.tile(np.r_[el["ctx"][(t, int(sp))], float(n)], (len(g), 1)).astype(np.float32)
        groups.append(dict(task=t, split=int(sp), n=int(n), X=Xel, C=C, U=g["U"].values.astype(np.float64), j=g["j"].values.astype(int)))
    return groups


def fit_scaler(groups) -> dict:
    """원소·문맥 특징의 평균·표준편차(학습 묶음에서만)."""
    E = np.concatenate([g["X"].reshape(-1, g["X"].shape[-1]) for g in groups]).astype(np.float64)
    C = np.concatenate([g["C"] for g in groups]).astype(np.float64)
    return dict(el_mu=E.mean(0), el_sd=E.std(0) + 1e-6, ctx_mu=C.mean(0), ctx_sd=C.std(0) + 1e-6)


def _scale(X, C, sc):
    return ((X - sc["el_mu"]) / sc["el_sd"]).astype(np.float32), ((C - sc["ctx_mu"]) / sc["ctx_sd"]).astype(np.float32)


# ================================================================ 모형
def make_model(d_el, d_ctx, seed, device):
    import torch
    from torch import nn

    class DeepSets(nn.Module):
        """φ = MLP(원소 → 128 → 128), 합·평균 풀링(256), ρ = MLP(256 + 문맥 → 128 → 1)."""

        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(d_el, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(2 * HIDDEN + d_ctx, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

        def forward(self, x, mask, ctx):
            h = self.phi(x) * mask.unsqueeze(-1)
            s = h.sum(1)
            cnt = mask.sum(1, keepdim=True).clamp(min=1.0)
            return self.rho(torch.cat([s, s / cnt, ctx], 1)).squeeze(1)

    torch.manual_seed(int(seed))
    return DeepSets().to(device)


def _pad(groups, sc, device):
    """묶음 목록 → 채운 텐서 (x, mask, ctx)와 묶음 경계."""
    import torch
    m = max(g["X"].shape[1] for g in groups)
    S = sum(len(g["U"]) for g in groups)
    d = groups[0]["X"].shape[2]
    x = np.zeros((S, m, d), np.float32)
    mask = np.zeros((S, m), np.float32)
    ctx = np.zeros((S, groups[0]["C"].shape[1]), np.float32)
    bounds, i = [], 0
    for g in groups:
        k, n = g["X"].shape[0], g["X"].shape[1]
        Xs, Cs = _scale(g["X"], g["C"], sc)
        x[i:i + k, :n] = Xs
        mask[i:i + k, :n] = 1.0
        ctx[i:i + k] = Cs
        bounds.append((i, i + k))
        i += k
    return torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device), torch.from_numpy(ctx).to(device), bounds


def ranknet_loss(f, U, bounds):
    """묶음마다 모든 집합 쌍(U_i < U_j)에 log(1 + exp(f_i − f_j)), 묶음 평균. 인덱스 누적 연산 없이 밀집 행렬로 계산한다(결정성)."""
    import torch
    losses = []
    for (a, b) in bounds:
        fg, ug = f[a:b], U[a:b]
        P = (ug.unsqueeze(1) < ug.unsqueeze(0) - 1e-12).to(f.dtype)
        if float(P.sum()) == 0:
            continue
        L = torch.nn.functional.softplus(fg.unsqueeze(1) - fg.unsqueeze(0))
        losses.append((L * P).sum() / P.sum())
    return torch.stack(losses).mean() if losses else f.sum() * 0.0


def predict(models, groups, sc, device) -> list:
    """묶음별 점수(모형 평균)."""
    import torch
    out = []
    with torch.no_grad():
        for g in groups:
            x, mask, ctx, _ = _pad([g], sc, device)
            p = np.mean([m(x, mask, ctx).double().cpu().numpy() for m in models], 0)
            out.append(p)
    return out


def group_cv_utility(groups, preds) -> float:
    """x_placement_policy.cv_utility 와 같은 정의(묶음마다 예측 최소 집합의 실제 U, 묶음 평균)."""
    rows = []
    for g, p in zip(groups, preds):
        rows.append(pd.DataFrame(dict(task=g["task"], split=g["split"], n=g["n"], j=g["j"], U=g["U"], pred=p)))
    d = pd.concat(rows, ignore_index=True)
    return XP.cv_utility(d, d["pred"].values)


def train_one(train_groups, sc, seed, epochs, device, val_groups=None, tag="final"):
    """seed 하나의 학습. val_groups 가 있으면 epoch 마다 검증 묶음 점수를 돌려준다. 반환 (모형, [epoch 별 검증 점수 목록])."""
    import torch
    d_el, d_ctx = train_groups[0]["X"].shape[2], train_groups[0]["C"].shape[1]
    model = make_model(d_el, d_ctx, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curve = []
    for ep in range(int(epochs)):
        model.train()
        order = np.random.RandomState(XB.seed_of("xd-ds", tag, int(seed), ep)).permutation(len(train_groups))
        for st in range(0, len(order), GROUPS_PER_STEP):
            batch = [train_groups[i] for i in order[st:st + GROUPS_PER_STEP]]
            x, mask, ctx, bounds = _pad(batch, sc, device)
            U = torch.from_numpy(np.concatenate([g["U"] for g in batch]).astype(np.float32)).to(device)
            opt.zero_grad()
            loss = ranknet_loss(model(x, mask, ctx), U, bounds)
            loss.backward()
            opt.step()
        if val_groups is not None:
            model.eval()
            curve.append(predict([model], val_groups, sc, device))
    model.eval()
    return model, curve


def cv_select_epoch(groups, seeds=DS_SEEDS, epochs=EPOCHS_MAX, device="cpu") -> dict:
    """조기 종료(계획 2.4): 접기(AL-k 와 알래스카 x 함께 제외)마다 seed 별로 학습하며 epoch 마다 검증 점수를 모으고, 세 seed 평균 점수의
    개발 교차검증 효용을 접기 평균한 곡선에서 가장 작은 epoch(동률은 앞 epoch)를 고른다."""
    tasks = sorted({g["task"] for g in groups})
    folds = XP.cv_folds(tasks)
    if not folds:
        raise SystemExit("[S9-DS] 교차검증 접기가 없다")
    per_fold, log = [], []
    for f in folds:
        tr = [g for g in groups if g["task"] in set(f["train"])]
        va = [g for g in groups if g["task"] == f["val"]]
        if not tr or not va:
            continue
        sc = fit_scaler(tr)
        curves = [train_one(tr, sc, s, epochs, device, va, tag=f"cv-{f['val']}")[1] for s in seeds]
        util = []
        for ep in range(int(epochs)):
            ens = [np.mean([curves[k][ep][gi] for k in range(len(seeds))], 0) for gi in range(len(va))]
            util.append(group_cv_utility(va, ens))
        per_fold.append(util)
        log.append(dict(val=f["val"], train=list(f["train"]), n_train_groups=len(tr), n_val_groups=len(va)))
    curve = np.mean(np.asarray(per_fold, float), 0)
    best = int(np.argmin(curve))
    return dict(best_epoch=best + 1, cv_utility=float(curve[best]), curve=[float(v) for v in curve], per_fold=[[float(v) for v in u] for u in per_fold],
                folds=log)


def cv_fixed_epoch(groups, folds, epochs, seeds=DS_SEEDS, device="cpu") -> pd.DataFrame:
    """XD-5: 정해진 epoch 수(동결 S9-DS 의 값)로 접기마다 다시 학습해 검증 과제별 개발 교차검증 효용을 낸다. folds = x_placement_policy 의
    lofo_folds(계열 하나 제외)·loto_folds(계열 안 과제 하나 제외)."""
    rows = []
    for f in folds:
        tr = [g for g in groups if g["task"] in set(f["train"])]
        if not tr:
            continue
        sc = fit_scaler(tr)
        models = [train_one(tr, sc, s, epochs, device, None, tag=f"xd5-{f['scheme']}-{f['fold']}")[0] for s in seeds]
        for t in f["val"]:
            va = [g for g in groups if g["task"] == t]
            if va:
                rows.append(dict(scheme=f["scheme"], fold=f["fold"], task=t, family=XP.family_of_task(t), n_train_groups=len(tr), n_val_groups=len(va),
                                 cv_utility=group_cv_utility(va, predict(models, va, sc, device))))
    return pd.DataFrame(rows)


def fit_final(groups, epochs, seeds=DS_SEEDS, device="cpu") -> dict:
    """최종 적합(개발 과제 전부, 고른 epoch 수, seed 0·1·2)."""
    sc = fit_scaler(groups)
    models = [train_one(groups, sc, s, epochs, device, None, tag="final")[0] for s in seeds]
    return dict(models=models, scaler=sc)


def save_bundle(bundle, path, cfg) -> str:
    import torch
    path = Path(path)
    XB.check_out_dir(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = dict(cfg=cfg, scaler={k: np.asarray(v, np.float64).tolist() for k, v in bundle["scaler"].items()},
               states=[{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in bundle["models"]])
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    sha = XB.sha256_file(path)
    XB._atomic_write(Path(str(path) + ".sha256"), f"{sha}  {path.name}\n")
    return sha


def load_bundle(path, device="cpu") -> dict:
    import torch
    XP.verify_sidecar(path)
    obj = torch.load(path, map_location=device, weights_only=False)
    cfg = obj["cfg"]
    models = []
    for st in obj["states"]:
        m = make_model(int(cfg["d_el"]), int(cfg["d_ctx"]), 0, device)
        m.load_state_dict(st)
        m.eval()
        models.append(m)
    return dict(models=models, scaler={k: np.asarray(v, np.float64) for k, v in obj["scaler"].items()}, cfg=cfg)


def ds_score_fn(bundle, device="cpu"):
    """탐욕 정책의 점수 함수: 후보 c 마다 L ∪ {c} 의 원소 특징(최근접 거리 갱신 포함)과 문맥을 만들어 세 모형의 평균 점수를 낸다."""
    import torch
    cache = {}

    def f(tf_, L, C):
        key = id(tf_)
        if key not in cache:
            cache[key] = tf_.element_static()
        static = cache[key]
        L = np.asarray(L, int); C = np.asarray(C, int)
        n1 = len(L) + 1
        d = static.shape[1] + 1
        Zs = static[:, :len(XB.FEATS)].astype(float)
        X = np.zeros((len(C), n1, d), np.float32)
        X[:, :n1 - 1, :d - 1] = static[L][None]
        X[:, n1 - 1, :d - 1] = static[C]
        if len(L):
            DLL = XP._cdist(Zs[L], Zs[L])
            np.fill_diagonal(DLL, np.inf)
            nnL = DLL.min(1) if len(L) > 1 else np.full(len(L), np.inf)
            DLC = XP._cdist(Zs[L], Zs[C])
            X[:, :n1 - 1, d - 1] = np.minimum(nnL[None, :], DLC.T)
            X[:, n1 - 1, d - 1] = DLC.min(0)
        Cx = np.tile(np.r_[tf_.ctx, float(n1)], (len(C), 1)).astype(np.float32)
        Xs, Cs = _scale(X, Cx, bundle["scaler"])
        out = np.zeros(len(C))
        with torch.no_grad():
            for st in range(0, len(C), PRED_CHUNK):
                x = torch.from_numpy(Xs[st:st + PRED_CHUNK]).to(device)
                mk = torch.ones(x.shape[0], n1, device=device)
                cx = torch.from_numpy(Cs[st:st + PRED_CHUNK]).to(device)
                out[st:st + PRED_CHUNK] = np.mean([m(x, mk, cx).double().cpu().numpy() for m in bundle["models"]], 0)
        return out
    return f


# ================================================================ 명령
def _data_args(passthrough=()):
    """x_placement_policy 의 인자 객체(자료 경로, 분할 1–5). 시험 과제 문맥을 만들 때만 쓴다(라벨 미사용 특징)."""
    return XP.parse(["--exp", "xd_t"] + list(passthrough))


def train_cli(o, dinfo):
    out = Path(o.out_dir) / "policy"
    suffix = "_smoke" if o.smoke else ""
    shards = XP.out_shard_dirs(o.out_dir, o.smoke)[0]                         # 본 실행 = <out>/shards(알래스카 계열 학습 자료), 스모크 = sealed/shards_smoke
    df, el = XP.load_learning(shards, XP.TAGS["xd_learn"] + suffix, XP._list(o.dev_tasks), with_elements=True)
    if not len(df):
        raise SystemExit("[S9-DS] 학습 자료가 없다")
    groups = assemble(df, el, XP._list(o.dev_tasks))
    seeds = tuple(int(v) for v in XP._list(o.seeds))
    t0 = time.time()
    cv = cv_select_epoch(groups, seeds, int(o.epochs), dinfo["device"])
    bundle = fit_final(groups, cv["best_epoch"], seeds, dinfo["device"])
    cfg = dict(d_el=int(groups[0]["X"].shape[2]), d_ctx=int(groups[0]["C"].shape[1]), hidden=HIDDEN, epochs=int(cv["best_epoch"]), seeds=list(seeds),
               lr=LR, weight_decay=WEIGHT_DECAY, groups_per_step=GROUPS_PER_STEP, el_feats=list(XP.EL_FEATS), ctx_feats=CTX_NAMES)
    mpath = out / f"s9_ds{suffix}.pt"
    msha = save_bundle(bundle, mpath, cfg)
    meta = dict(variant="S9-DS", model_path=str(mpath), model_sha256=msha, cv_utility=cv["cv_utility"], best_epoch=cv["best_epoch"], cv=cv,
                features=dict(element=list(XP.EL_FEATS), context=CTX_NAMES), seeds=list(seeds), torch=dinfo.get("torch"), cuda=dinfo.get("cuda"),
                cudnn=dinfo.get("cudnn"), cublas=dinfo.get("cublas"), device=dinfo.get("device"), gpu=dinfo.get("gpu"), gpu_name=dinfo.get("gpu_name"),
                gpu_name_torch=dinfo.get("gpu_name_torch"), device_order=dinfo.get("device_order"), gpu_allowed=dinfo.get("gpu_allowed"),
                driver=dinfo.get("driver"), threads=dinfo.get("threads"), deterministic=bool(dinfo.get("deterministic")), n_groups=len(groups),
                n_sets=int(len(df)),
                tasks=sorted(df.task.unique()), elapsed_s=round(time.time() - t0, 1), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                code_sha256=XB.sha256_file(__file__), policy_code_sha256=XB.sha256_file(XP.__file__),
                index_path=str(out / f"s9_ds_index{suffix}.json"))
    if o.smoke:
        XB.write_sealed(XP.EXP_NAME, {f"s9_ds_meta{suffix}.json": meta}, root=Path(o.out_dir).parent)
        print(f"[S9-DS] 스모크 기록은 봉인 폴더에 썼다 · 모형 sha256 {msha[:16]}", flush=True)
    else:
        sha = XP._write_json(out / "s9_ds_meta.json", meta)
        print(f"[S9-DS] 묶음 {len(groups)} · 집합 {len(df)} · 장치 {dinfo.get('device')}(GPU {dinfo.get('gpu')}) · 모형 sha256 {msha[:16]} · "
              f"기록 sha256 {sha[:16]}", flush=True)
    return mpath


def export_cli(o, dinfo, index_out=None, model_path=None):
    suffix = "_smoke" if o.smoke else ""
    out = Path(o.out_dir) / "policy"
    mpath = Path(model_path) if model_path else out / f"s9_ds{suffix}.pt"
    bundle = load_bundle(mpath, dinfo["device"])
    tasks = XP._list(o.test_tasks)
    if not o.smoke:
        XP.assert_dev_test_disjoint(XP._list(o.dev_tasks), tasks)
    a = _data_args(["--smoke"] if o.smoke else [])
    tfs = XP.task_feats_for(a, tasks)
    seeds = XP.POLICY_SEEDS[:1] if o.smoke else XP.POLICY_SEEDS
    n_max = 20 if o.smoke else XP.POLICY_NMAX
    path = Path(index_out) if index_out else out / f"s9_ds_index{suffix}.json"
    sha = XP.export_index(tfs, ds_score_fn(bundle, dinfo["device"]), "S9-DS", XB.sha256_file(mpath), path, seeds=seeds, n_max=n_max,
                          extra=dict(device=dinfo.get("device"), gpu=dinfo.get("gpu"), torch=dinfo.get("torch"), cuda=dinfo.get("cuda")))
    print(f"[색인] S9-DS · 과제·분할 {len(tfs)} · 정책 seed {len(seeds)} · {path.name} sha256 {sha[:16]}", flush=True)
    return path


def jaccard_cli(o, dinfo):
    """동결 뒤 다른 GPU 에서 선택을 다시 만들어 동결 색인과의 Jaccard 겹침을 서술한다(크기 n = 20, 40 의 앞부분)."""
    out = Path(o.out_dir) / "policy"
    ref = out / "s9_ds_index.json"
    XP.verify_sidecar(ref)
    alt = out / f"s9_ds_index_gpu{dinfo.get('gpu')}.json"
    export_cli(o, dinfo, index_out=alt)
    A, B = json.loads(ref.read_text()), json.loads(alt.read_text())
    kb = {(e["alias"], e["split"], e["policy_seed"]): e["order"] for e in B["entries"]}
    rows = []
    for e in A["entries"]:
        ob = kb.get((e["alias"], e["split"], e["policy_seed"]))
        if ob is None:
            continue
        for n in XP.TEST_N:
            sa, sb = set(e["order"][:n]), set(ob[:n])
            rows.append(dict(alias=e["alias"], split=e["split"], policy_seed=e["policy_seed"], n=n, jaccard=len(sa & sb) / max(len(sa | sb), 1)))
    df = pd.DataFrame(rows)
    p = out / f"s9_ds_jaccard_gpu{dinfo.get('gpu')}.csv"
    XP._write_csv(p, df)
    print(f"[Jaccard] 항목 {len(df)} · 평균 {df.jaccard.mean():.3f} · 최소 {df.jaccard.min():.3f} · 기준 GPU {A.get('gpu')} · 비교 GPU {dinfo.get('gpu')}",
          flush=True)
    return df


def xd5_cli(o, dinfo):
    """XD-5(XD-4 시험 표를 연 뒤): 개발 과제와 시험 과제의 학습 자료로 계열 하나 제외·계열 안 과제 하나 제외 교차검증 효용(S9-DS, 동결 epoch).
    봉인 폴더에 쓴다."""
    XP.require_xd4_sealed(o.out_dir)                                         # 계획 0.3 열람 순서 4: XD-4 시험 표가 있을 때만
    meta = json.loads((Path(o.out_dir) / "policy" / "s9_ds_meta.json").read_text())
    tasks = list(XP.DEV_TASKS) + list(XP.TEST_TASKS)
    df, el = XP.load_learning(XP.out_shard_dirs(o.out_dir)[0], XP.TAGS["xd_learn"], tasks, with_elements=True)
    groups = assemble(df, el, tasks)
    present = sorted({g["task"] for g in groups})
    tab = cv_fixed_epoch(groups, XP.lofo_folds(present) + XP.loto_folds(present), int(meta["best_epoch"]),
                         tuple(int(v) for v in meta.get("seeds", DS_SEEDS)), dinfo["device"])
    XB.write_sealed(XP.EXP_NAME, {"xd5_cv_ds.csv": tab, "xd5_cv_ds_meta.json": dict(tasks=present, epochs=int(meta["best_epoch"]),
                                                                                    device=dinfo.get("device"), gpu=dinfo.get("gpu"))},
                    root=Path(o.out_dir).parent)
    return tab


def build_parser():
    ap = argparse.ArgumentParser(description="S9-DS(DeepSets 배치 정책, 로컬 GPU, 결정적 설정)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--jaccard-gpu", type=int, default=None, help="동결 뒤 이 GPU 에서 선택을 다시 만들어 겹침을 쓴다")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS_MAX)
    ap.add_argument("--seeds", default=",".join(str(v) for v in DS_SEEDS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default=str(XP.DEFAULT_OUT))
    ap.add_argument("--dev-tasks", default=",".join(XP.DEV_TASKS))
    ap.add_argument("--test-tasks", default=",".join(XP.TEST_TASKS + XP.TEST_DESC))   # xd_test 의 기본 과제와 같다(S9-GBM 색인과 같은 과제 집합)
    ap.add_argument("--smoke", action="store_true", help="스모크 학습 자료(xdl_smoke), epoch 2, seed 0, 시험 과제 AL-5")
    ap.add_argument("--xd5", action="store_true", help="XD-5: 계열 하나 제외·과제 하나 제외(XD-4 시험 표를 연 뒤에만, 봉인 폴더)")
    ap.add_argument("--mem-wait", type=float, default=1800.0, help="가용 메모리가 30 GB 아래면 이 초만큼 기다린 뒤 거부(계획 1절, x_placement_policy 와 같다)")
    return ap


def main(argv=None):
    o = build_parser().parse_args(argv)
    if o.smoke:
        o.epochs = min(int(o.epochs), 2)
        o.seeds = "0"
        o.dev_tasks = ",".join(XP.SMOKE_TARGETS["xd_learn"])
        o.test_tasks = ",".join(XP.SMOKE_TARGETS["xd_test"])
    XB.guard("test" if o.smoke else "summarize")                               # 로컬 GPU 학습은 계획 1절 로컬 허용 범위(DeepSets 학습)
    XB.require_memory(XP.MEM_MIN_GB, wait_s=float(o.mem_wait))                # 1절 로컬 자원(스모크 포함): 30 GB 아래면 기다린 뒤 거부
    with XB.restricted_output():
        if o.jaccard_gpu is not None:
            dinfo = setup_device("cuda", o.jaccard_gpu, 0, o.threads)
            jaccard_cli(o, dinfo)
            return 0
        dinfo = setup_device(o.device, o.gpu, 0, o.threads)
        if dinfo["device"] == "cpu":
            XP.limit_data(10.0)
        print(f"[장치] {dinfo['device']} · GPU {dinfo.get('gpu')} · torch {dinfo.get('torch')} · CUDA {dinfo.get('cuda')} · 스레드 {dinfo.get('threads')}",
              flush=True)
        if o.xd5:
            xd5_cli(o, dinfo)
            return 0
        if o.train:
            train_cli(o, dinfo)
        if o.export:
            export_cli(o, dinfo)
        print(f"[S9-DS] 최대 RSS {XB.max_rss_mb()} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
