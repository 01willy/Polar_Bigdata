"""Copernicus DEM GLO-30 1° tiles for a lat/lon box (public AWS bucket, no account). Skips tiles already in data/raw/dem; records every tile's
status (exists / downloaded / absent_public = HTTP 404, i.e. open sea / error) in a CSV. Written for the Canada map grid (2026-10-20).
Run: python3 scripts/0_download/copernicus_dem_box.py --lat0 60 --lat1 70 --lon0 -141 --lon1 -112 --status data/processed/map_canada/dem_tiles_download_v1.csv
"""
import argparse, csv, os, sys, time, urllib.request, urllib.error
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "dem"
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
ap = argparse.ArgumentParser()
ap.add_argument("--lat0", type=int, required=True); ap.add_argument("--lat1", type=int, required=True)
ap.add_argument("--lon0", type=int, required=True); ap.add_argument("--lon1", type=int, required=True)
ap.add_argument("--status", required=True); ap.add_argument("--min-free-gb", type=float, default=20.0)
a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)


def tile_name(tlat, tlon):
    ns = f"N{abs(int(tlat)):02d}" if tlat >= 0 else f"S{abs(int(tlat)):02d}"
    ew = f"E{abs(int(tlon)):03d}" if tlon >= 0 else f"W{abs(int(tlon)):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def free_gb(path):
    st = os.statvfs(path); return st.f_bavail * st.f_frsize / 1e9


rows = []; t0 = time.time(); n_dl = n_ok = n_abs = n_err = 0; total = 0
for tlat in range(a.lat0, a.lat1):
    for tlon in range(a.lon0, a.lon1):
        name = tile_name(tlat, tlon); dst = OUT / (name + ".tif"); status = ""; size = 0
        if dst.exists() and dst.stat().st_size > 1000:
            status = "exists"; size = dst.stat().st_size; n_ok += 1
        else:
            if free_gb(OUT) < a.min_free_gb:
                status = "skipped_low_disk"; print("[stop] free disk below limit", flush=True); rows.append((name, tlat, tlon, status, 0)); break
            url = f"{BASE}/{name}/{name}.tif"; tmp = dst.with_suffix(".tif.part")
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "polar-bigdata/1.0"})
                    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                        while True:
                            b = r.read(1 << 20)
                            if not b:
                                break
                            f.write(b)
                    os.replace(tmp, dst); size = dst.stat().st_size; status = "downloaded"; n_dl += 1; total += size
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        status = "absent_public"; n_abs += 1; break
                    status = f"error:http{e.code}"; time.sleep(5)
                except Exception as e:
                    status = f"error:{type(e).__name__}"; time.sleep(5)
            if tmp.exists():
                tmp.unlink()
            if status.startswith("error"):
                n_err += 1
        rows.append((name, tlat, tlon, status, size))
        if len(rows) % 10 == 0:
            print(f"[{len(rows)}] exists {n_ok} · downloaded {n_dl} ({total / 1e9:.2f} GB) · absent {n_abs} · error {n_err} · {time.time() - t0:.0f} s", flush=True)
with open(a.status, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["tile", "tlat", "tlon", "status", "bytes"]); w.writerows(rows)
print(f"[done] tiles {len(rows)}: exists {n_ok} · downloaded {n_dl} ({total / 1e9:.2f} GB) · absent {n_abs} · error {n_err} · {time.time() - t0:.0f} s · free {free_gb(OUT):.1f} GB", flush=True)
