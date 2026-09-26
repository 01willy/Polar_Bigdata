"""ESA CCI Permafrost CRDPv4(v04.0) 지중온도(GTD)·영구동토 분율(PFR) 층 다운로드. 네트워크 전용, CPU.

목적: Track C1(계획 docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md §4 C1) 앵커 결합에 쓸 CCI 다층을 CEDA 공개 경로에서 받는다.
경로(2026-09-26 실제 조회로 확인, 공개 접근·HTTP Range 재개 가능):
  https://dap.ceda.ac.uk/neodc/esacci/permafrost/data/<product>/L4/area4/pp/v04.0/
    product = ground_temperature(GTD) | permafrost_extent(PFR)
    파일명 = ESACCI-PERMAFROST-L4-<VAR>-<SRC>-AREA4_PP-<year>-fv04.0.nc
      SRC = ERA5_MODISLST_BIASCORRECTED(1997–2002) | MODISLST_CRYOGRID(2003–2021)  (기존 ALT 파일과 같은 규칙)
  용량: GTD 약 367 MB/년(0.01°, 6000×36000, 깊이 다층), PFR 약 13 MB/년.
출력: data/raw/cci_gtd/, data/raw/cci_pfr/ (파일명 원본 유지). 완료 여부는 서버 Content-Length 와 로컬 크기 비교로 판정.
실행(ROOT): python3 scripts/1_data_prep/fetch_cci_layers.py --what pfr --years 1997-2021
            python3 scripts/1_data_prep/fetch_cci_layers.py --what gtd --years 2003-2021 [--years 1997-2002 추가]
  --dry-run 은 URL·크기만 출력. 중단 시 같은 명령으로 재개(curl -C -).
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://dap.ceda.ac.uk/neodc/esacci/permafrost/data"
PRODUCT = {"gtd": ("ground_temperature", "GTD", ROOT / "data" / "raw" / "cci_gtd"),
           "pfr": ("permafrost_extent", "PFR", ROOT / "data" / "raw" / "cci_pfr")}


def parse_years(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return sorted(set(out))


def fname(var: str, year: int) -> str:
    src = "ERA5_MODISLST_BIASCORRECTED" if year <= 2002 else "MODISLST_CRYOGRID"
    return f"ESACCI-PERMAFROST-L4-{var}-{src}-AREA4_PP-{year}-fv04.0.nc"


def remote_size(url: str) -> int | None:
    r = subprocess.run(["curl", "-sI", "-m", "60", url], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1].strip())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="pfr,gtd", help="쉼표 목록: pfr, gtd")
    ap.add_argument("--years", default="2003-2021", help="예: 1997-2021 또는 2003-2021,1999")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()
    years = parse_years(args.years)
    t0 = time.time(); n_ok = n_fail = n_skip = 0
    for what in [w.strip() for w in args.what.split(",") if w.strip()]:
        prod, var, outdir = PRODUCT[what]
        outdir.mkdir(parents=True, exist_ok=True)
        for y in years:
            fn = fname(var, y); url = f"{BASE}/{prod}/L4/area4/pp/v04.0/{fn}"; dst = outdir / fn
            rs = remote_size(url)
            if rs is None:
                print(f"[fail] HEAD 실패 {url}", flush=True); n_fail += 1
                continue
            if dst.exists() and dst.stat().st_size == rs:
                print(f"[skip] {fn} 완료({rs/1e6:.1f} MB)", flush=True); n_skip += 1
                continue
            if args.dry_run:
                print(f"[plan] {fn} {rs/1e6:.1f} MB -> {dst}", flush=True)
                continue
            ok = False
            for k in range(args.retries):
                r = subprocess.run(["curl", "-sS", "-L", "-C", "-", "--retry", "3", "--retry-delay", "10", "-o", str(dst), url])
                if r.returncode == 0 and dst.exists() and dst.stat().st_size == rs:
                    ok = True; break
                print(f"[retry {k + 1}] {fn} rc={r.returncode} size={dst.stat().st_size if dst.exists() else 0}/{rs}", flush=True)
                time.sleep(15)
            n_ok += ok; n_fail += (not ok)
            print(f"[{'ok' if ok else 'fail'}] {fn} {rs/1e6:.1f} MB  경과 {(time.time() - t0)/60:.1f} min", flush=True)
    print(f"[done] ok {n_ok} skip {n_skip} fail {n_fail}  경과 {(time.time() - t0)/60:.1f} min", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
