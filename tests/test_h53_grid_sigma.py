"""scripts/2_evaluation/h53_grid_sigma_nflow.py 시험(LGU 계획서 개정 4 조건부 격자 σ 규약, 실행 계획 3.1 A7).

합성 자료만 쓴다(LGU·LG·LGX 의 판정 표와 조각을 읽지 않는다). nflow 적합은 합성 자료에서 CPU, epochs 3, 8건(약 2 s)이다.
(a) GPU 후보 규칙: 0–4 번과 허용 목록 밖 거부, 메모리 50 MiB 초과·계산 프로세스 있음이면 점유, 후보 순서 8, 9, 7, 6, 5.
(b) 실행 보호: --allow-run 없는 실행 거부, --synthetic 의 --out 필수와 본 산출 폴더 거부, 스레드 상한, seed 규칙.
(c) npz 형식이 h49 가 읽는 형식(cal = {지역: σ}, grid_cell_id, grid_sigma)과 같고, cal 의 지역과 배열 길이가 h49 real_ctx·cal_groups 의
    보정 집단(셀 30개 이상)과 같다. σ 는 두 seed 의 평균이고 절단 범위 안이다.
(d) h49.run 이 LGU-B2 '전이' 합성 판정 표와 이 npz 로 끝까지 돈다(정규화기 nflow, 표시 셀의 구간 폭 유한, 폭이 예측값에 비례하지 않는다).
(e) --dry-run 은 아무것도 쓰지 않는다. (f) 한 seed 가 실패하면 npz 를 쓰지 않고 종료 코드 5. (g) 실험 B 조각과의 설정 대조 논리.
(h) 실행 중 GPU 상실 판정(다른 계산 프로세스, 우리 PID 의 GPU). (i) 빈 GPU 고르기(후보 순서, 없으면 종료 코드 3).
(j) 본 실행의 재현 대조(가짜 실험 B σ 조각, 같은 seed 의 최대 절대 차, 유효 seed 가 아니면 비교 불가).
(k) 본 실행 모드의 거부(등록 미커밋, 기존 npz, load average 초과). 자료 적재와 GPU 고정 전에 끝난다.
실행(CPU 2스레드, nice 10):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 python3 -m pytest -q tests/test_h53_grid_sigma.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # 시험은 GPU 를 쓰지 않는다
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


h53 = _load("h53_grid_sigma_nflow", "scripts/2_evaluation/h53_grid_sigma_nflow.py")
h45 = h53.h45()
h49 = h53.h49()


def _code(exc):
    return exc.value.code


# ---------------------------------------------------------------- (a) GPU 규칙
SMI_INFO = """0, GPU-a0, 13048
5, GPU-a5, 13188
6, GPU-a6, 2
7, GPU-a7, 2
8, GPU-a8, 60
9, GPU-a9, 2"""
SMI_APPS = """111, GPU-a0
222, GPU-a5
333, GPU-a7"""


def _info_procs():
    info = h53.parse_gpu_info(SMI_INFO)
    apps = h53.parse_gpu_apps(SMI_APPS)
    by = {v["uuid"]: g for g, v in info.items()}
    procs = {}
    for pid, u in apps:
        procs.setdefault(by[u], []).append(pid)
    return info, procs


def test_a_gpu_candidates_and_screen():
    assert h53.gpu_candidates("8,9,7,6,5") == [8, 9, 7, 6, 5]
    for bad in ("0", "8,4", "10", "3,9", "x"):
        with pytest.raises(SystemExit) as e:
            h53.gpu_candidates(bad)
        assert _code(e) == h53.EXIT_REFUSE
    info, procs = _info_procs()
    free, table = h53.screen_gpus([8, 9, 7, 6, 5], info, procs)
    assert free == [9, 6]                                                 # 8 은 60 MiB, 7 은 계산 프로세스, 5 는 메모리
    why = {r["gpu"]: r["why"] for r in table}
    assert "60 MiB" in why[8] and "계산 프로세스" in why[7] and why[9] == "" and "13188" in why[5]
    info[8]["used"] = 2.0
    assert h53.screen_gpus([8, 9], info, procs)[0] == [8, 9]              # 8 이 비면 8 이 먼저다
    info[8]["used"] = 50.0
    assert h53.screen_gpus([8], info, procs)[0] == [8]                    # 50 MiB 이하는 빈 것으로 본다


# ---------------------------------------------------------------- (b) 실행 보호
def test_b_guards(tmp_path):
    with pytest.raises(SystemExit) as e:
        h53.main([])
    assert _code(e) == h53.EXIT_REFUSE
    with pytest.raises(SystemExit) as e:
        h53.main(["--synthetic"])
    assert _code(e) == h53.EXIT_REFUSE
    with pytest.raises(SystemExit) as e:
        h53.main(["--synthetic", "--out", str(tmp_path / "s"), "--threads", "5"])
    assert _code(e) == h53.EXIT_REFUSE
    for sd in ("0", "0,0", "0,3", "1,2,0"):
        with pytest.raises(SystemExit) as e:
            h53.main(["--synthetic", "--out", str(tmp_path / "s"), "--seeds", sd])
        assert _code(e) == h53.EXIT_REFUSE
    with pytest.raises(SystemExit):                                         # 합성 시험은 본 산출 폴더에 쓰지 않는다
        h53.main(["--synthetic", "--out", str(ROOT / "data/processed/map_lena/syn_should_not_exist")])
    with pytest.raises(SystemExit):                                         # 실험 산출 폴더 보호(h49.check_out)
        h53.main(["--synthetic", "--out", str(ROOT / "data/processed/lgu/syn_should_not_exist")])
    assert not list((ROOT / "data/processed/map_lena").glob("syn_should_not_exist*"))
    assert h53.parse_args(["--allow-run"]).threads == 4 and h53.parse_args(["--dry-run"]).threads == 2
    st = h53.prereg_state()
    assert set(st) >= {"present", "committed"}


# ---------------------------------------------------------------- 합성 실행(모듈에서 한 번)
@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    out = tmp_path_factory.mktemp("h53syn") / "lena_sigma_nflow_v1"
    rc = h53.main(["--synthetic", "--out", str(out), "--threads", "2"])
    import torch
    assert torch.cuda.device_count() == 0                                   # 시험 프로세스가 GPU 를 보지 않는다
    p = h53.out_paths(out)
    return dict(rc=rc, npz=p["npz"], meta=json.loads(p["meta"].read_text()), dir=p["stem"].parent)


def _ctx():
    D, g = h53.synthetic_inputs(h45)
    return D, g, h49.real_ctx(None, 1, D=D, grid_df=g)


# ---------------------------------------------------------------- (c) npz 형식
def test_c_npz_schema_matches_h49(syn):
    assert syn["rc"] == 0 and syn["npz"].exists()
    z = np.load(syn["npz"], allow_pickle=True)
    assert {"cal", "grid_cell_id", "grid_sigma"} <= set(z.files)
    cal = {str(k): np.asarray(v, float) for k, v in z["cal"].item().items()}        # h49 994–995행과 같은 식
    grid = pd.Series(np.asarray(z["grid_sigma"], float), index=np.asarray(z["grid_cell_id"]).astype(str))
    D, g, mctx = _ctx()
    G = h49.cal_groups(mctx.src)
    assert sorted(cal) == G == ["Alaska", "Canada", "Russia_W"]                     # 30 셀 미만 Russia_E(24)와 Tibet 는 빠진다
    assert "Russia_E" in list(z["groups_lgu"])                                      # 실험 B 의 집단(20개 이상)에는 있다
    src = mctx.src
    ok = np.isfinite(src.y) & np.isfinite(src.s) & (src.y > 0) & (src.s > 0)
    for k in G:
        assert len(cal[k]) == int((ok & (src.macro.astype(str) == k)).sum())       # h49 calibrate 의 te 와 길이가 같다
    assert (~ok & (src.macro.astype(str) == "Canada")).sum() == 1                   # y = 0 셀은 보정 셀에서 빠진다
    assert grid.index.is_unique and list(grid.index) == list(g.cell_id.astype(str))
    lo, hi = z["sigma_clip"]
    for v in list(cal.values()) + [grid.values, z["lena_sigma"]]:
        assert np.all(np.isfinite(v)) and v.min() >= lo - 1e-12 and v.max() <= hi + 1e-12
    assert np.allclose(z["grid_sigma"], z["grid_sigma_seeds"].mean(0))
    assert np.allclose(z["cal_sigma"], z["cal_sigma_seeds"].mean(0))
    assert z["grid_sigma_seeds"].shape[0] == 2 and list(z["seeds"]) == [0, 1]
    for k in G:
        assert np.allclose(z["cal_sigma"][z["cal_region"] == k], cal[k])
    assert not np.allclose(z["grid_sigma_seeds"][0], z["grid_sigma_seeds"][1])     # 두 seed 는 다른 적합이다
    assert str(z["format"]) == h53.FORMAT


# ---------------------------------------------------------------- (d) h49 가 소비한다
def test_d_h49_run_consumes_sigma(syn, tmp_path):
    D, g, mctx = _ctx()
    meta = h49.run(mctx, h49.synthetic_tables("b2"), tmp_path, 1, dict(synthetic=True), sigma_file=str(syn["npz"]), strict=True)
    cal = meta["calibration"]
    assert cal["normalizer"] == "nflow" and cal["groups"] == ["Alaska", "Canada", "Russia_W"] and np.isfinite(cal["q"])
    pred = pd.read_csv(tmp_path / "lena_pred_v1.csv.gz", dtype={"cell_id": str})
    shown = pred.gray.values == 0
    w, p = pred.width90_a.values[shown], pred.pred_a.values[shown]
    assert np.all(np.isfinite(w))
    ratio = w / np.maximum(p, 1.0)
    assert np.ptp(ratio) > 1e-6                                             # σ 가 셀마다 달라 폭이 예측값에 비례하지 않는다
    assert syn["meta"]["h49_consumer_run"]["passed"] and syn["meta"]["h49_schema_check"]["passed"]


# ---------------------------------------------------------------- 메타
def test_meta_fields(syn):
    m = syn["meta"]
    assert m["status"] == "ok" and m["mode"] == "synthetic" and m["gpu"] is None
    assert len(m["script_sha256"]) == 64 and set(m["code"]) >= {"this", "h45", "lgu_common", "h40", "h49", "tab_models"}
    cfg = m["config"]["fit_cfg"]
    assert cfg["normalizer"] == "nflow" and cfg["seeds"] == [0, 1] and cfg["clamp"] == 2.0 and cfg["val"] == "block"
    assert cfg["epochs"] == h53.SYN_EPOCHS and cfg["sigma_clip"] == [0.02, 2.0] and cfg["cap_cells"] == 100
    assert m["fits"]["n_fit"] == 8 and m["fits"]["valid_seeds"] == [0, 1] and all(not r["failed"] for r in m["fits"]["rows"])
    assert {r["train_set"] for r in m["fits"]["rows"]} == {"Alaska", "Canada", "Russia_W", "full"}
    assert m["h49_alignment"]["aligned"] and m["outputs"]["n_grid"] == 150 and "wall_s" in m


# ---------------------------------------------------------------- (e) 계획만
def test_e_dry_run_writes_nothing(tmp_path, capsys):
    rc = h53.main(["--dry-run", "--synthetic", "--out", str(tmp_path / "d")])
    out = capsys.readouterr().out
    assert rc == 0 and "4 학습 집합 × seed [0, 1] = 8건" in out and "[fit]" not in out
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- (f) seed 실패
def test_f_failed_seed_blocks_output(tmp_path, monkeypatch):
    orig = h45._fit_one

    def flaky(a, nm, name, full, *rest):
        seed = rest[7]
        if name == "Canada" and int(seed) == 1:
            raise ValueError("시험용 실패")
        return orig(a, nm, name, full, *rest)
    monkeypatch.setattr(h45, "_fit_one", flaky)
    out = tmp_path / "f"
    rc = h53.main(["--synthetic", "--out", str(out)])
    p = h53.out_paths(out)
    assert rc == h53.EXIT_FIT_FAIL and not p["npz"].exists() and p["failed"].exists()
    fm = json.loads(p["failed"].read_text())
    assert fm["status"] == "fit_failed" and fm["fits"]["valid_seeds"] == [0]


# ---------------------------------------------------------------- (g) 설정 대조
def test_g_compare_ref_logic():
    a45 = h53.make_a45(h45, [0, 1], 2, synthetic=False)
    mine = h45.fit_cfg(a45, "nflow")
    ref_cfg = json.loads(json.dumps(dict(mine, seeds=[0, 1, 2])))
    ref = dict(cfg=ref_cfg, cfg_hash=h53.LC.cfg_hash(ref_cfg), elapsed_s=30.0, n_fit=15, inputs=h45.input_shas(a45), **h45.code_shas())
    c = h53.compare_ref(h45, a45, ref)
    assert c["cfg_equal_except_seeds"] and c["seeds_subset"] and c["cfg_hash_ref_ok"] and c["inputs_equal"] and c["code_equal"]
    bad = json.loads(json.dumps(dict(ref_cfg, epochs=50)))
    c2 = h53.compare_ref(h45, a45, dict(ref, cfg=bad))
    assert not c2["cfg_equal_except_seeds"] and c2["cfg_diff_keys"] == ["epochs"]
    c3 = h53.compare_ref(h45, a45, dict(ref, cfg=dict(ref_cfg, seeds=[0, 2])))
    assert c3["cfg_equal_except_seeds"] and not c3["seeds_subset"]
    assert not h53.compare_ref(h45, a45, None)["ref_found"]
    assert a45.epochs == 100 and a45.clamp == 2.0 and a45.min_cal_cells == 20 and a45.SEEDS == [0, 1]   # 실험 B 기본값


# ---------------------------------------------------------------- (h) 실행 중 GPU 상실
def test_h_gpu_guard(monkeypatch):
    me = os.getpid()
    gpu = dict(index=8)
    monkeypatch.setattr(h53, "query_gpus", lambda: ({}, {8: [me]}))
    assert h53.pid_check(gpu)["ok"]
    monkeypatch.setattr(h53, "query_gpus", lambda: ({}, {8: [me, 4242]}))
    with pytest.raises(h53.GpuLost):
        h53.gpu_guard(gpu)
    monkeypatch.setattr(h53, "query_gpus", lambda: ({}, {9: [me]}))
    with pytest.raises(h53.GpuLost):
        h53.pid_check(gpu)


# ---------------------------------------------------------------- (i) 본 실행 전용 경로(자료 적재·GPU 없이 시험할 수 있는 부분)
def test_i_pick_gpu(monkeypatch):
    monkeypatch.setattr(h53, "query_gpus", _info_procs)
    g, info, hist = h53.pick_gpu([8, 9, 7, 6, 5], wait_min=0)
    assert g == 9 and info["uuid"] == "GPU-a9" and hist[-1]["table"][0]["gpu"] == 8
    with pytest.raises(SystemExit) as e:
        h53.pick_gpu([8, 7, 5], wait_min=0)
    assert _code(e) == h53.EXIT_NO_GPU


def test_j_lgu_repro_against_fake_shard(syn, tmp_path, monkeypatch):
    a = h53.parse_args(["--synthetic", "--out", str(tmp_path / "x")])
    C = h53.build_context(a, h45, h49, [0, 1], synthetic=True)
    z = np.load(syn["npz"], allow_pickle=True)
    G = list(z["groups"])
    res = dict(seeds=[0, 1], valid=[0, 1], lena=np.array(z["lena_sigma_seeds"]),
               cal={k: np.array(z["cal_sigma_seeds"][:, z["cal_region"] == k]) for k in G})
    info = C["info"]; loc = C["D"].df.loc_id.values
    cal_all = np.random.RandomState(1).uniform(0.1, 0.5, (3, len(info["cal_idx"])))       # 실험 B 순서(20개 이상 집단 전부)
    for k in G:
        m = info["cal_region"] == k
        cal_all[0, m] = res["cal"][k][0]; cal_all[1, m] = res["cal"][k][1]
    tgt = np.vstack([res["lena"][0] + 1e-7, res["lena"][1], np.full(len(info["t_idx"]), np.nan)])
    fake = tmp_path / "lgub__fit__Lena__x__nflow_sigma.npz"
    np.savez_compressed(fake, seeds=np.array([0, 1, 2]), valid_seeds=np.array([0]), tgt_loc=loc[info["t_idx"]], tgt_sigma=tgt,
                        cal_loc=loc[info["cal_idx"]], cal_region=np.asarray(info["cal_region"], str), cal_sigma=cal_all)
    monkeypatch.setattr(h45, "fit_paths", lambda a_, t_, nm_: dict(sigma=fake, unit=tmp_path / "u.json", runs=tmp_path / "r.csv"))
    r = h53.lgu_repro(h45, C, res)
    d0 = r["per_seed"]["0"]
    assert r["status"] == "대조" and d0["lena_loc_same"] and abs(d0["lena_max_abs"] - 1e-7) < 1e-9
    assert all(d0[f"cal_{k}_loc_same"] and d0[f"cal_{k}_max_abs"] == 0.0 for k in G)
    assert r["per_seed"]["1"].startswith("비교 불가")                         # 원 조각에서 유효 seed 가 아니다


def test_k_real_mode_refusals(tmp_path, monkeypatch):
    out = tmp_path / "r"
    monkeypatch.setattr(h53, "prereg_state", lambda: dict(present=True, committed=False, head="x"))
    with pytest.raises(SystemExit) as e:
        h53.main(["--allow-run", "--out", str(out)])
    assert _code(e) == h53.EXIT_REFUSE
    monkeypatch.setattr(h53, "prereg_state", lambda: dict(present=True, committed=True, head="x"))
    (tmp_path / "r.npz").write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        h53.main(["--allow-run", "--out", str(out)])
    assert _code(e) == h53.EXIT_REFUSE
    (tmp_path / "r.npz").unlink()
    monkeypatch.setattr(h53.os, "getloadavg", lambda: (100.0, 0.0, 0.0))
    with pytest.raises(SystemExit) as e:
        h53.main(["--allow-run", "--out", str(out)])
    assert _code(e) == h53.EXIT_NO_GPU
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""                   # 거부 경로는 GPU 를 고정하지 않는다
