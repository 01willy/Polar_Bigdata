"""Liu Z. 등 2024 범북극 1 km 연별 ALT(2003–2020, Zenodo 10070610, CC BY 4.0) 가운데 2010–2020 을 내려받는다.
용도: v5 제품 비교(재설정 문서 4.4). 파일 md5 는 Zenodo API 의 checksum 과 대조한다. 산출: data/raw/liu2024_alt/ALT_<year>.tif, SOURCE.md
"""
import hashlib, json, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "liu2024_alt"; OUT.mkdir(exist_ok=True)
YEARS = range(2010, 2021)
rec = json.load(urllib.request.urlopen("https://zenodo.org/api/records/10070610", timeout=60))
files = {f["key"]: f for f in rec["files"]}
log = []
for y in YEARS:
    k = f"ALT_{y}.tif"; f = files[k]; dst = OUT / k
    want = f["checksum"].split(":")[1]
    if dst.exists() and hashlib.md5(dst.read_bytes()).hexdigest() == want:
        log.append((k, "exists", want)); continue
    for attempt in range(3):
        r = subprocess.run(["curl", "-sSL", "--retry", "3", "--max-time", "900", "-o", str(dst), f["links"]["self"]])
        got = hashlib.md5(dst.read_bytes()).hexdigest() if dst.exists() else ""
        if got == want:
            break
        time.sleep(10)
    log.append((k, "ok" if got == want else "md5_mismatch", want)); print(k, log[-1][1], flush=True)
md = ["# Liu 등 2024 ALT(Zenodo 10070610)", "", f"- 받은 때: {time.strftime('%Y-%m-%d %H:%M:%S')}, 스크립트 scripts/0_download/liu2024_alt.py",
      f"- 제목: {rec['metadata']['title']}", f"- 약관: {rec['metadata'].get('license', {}).get('id')}",
      "- 논문: https://doi.org/10.1088/1748-9326/ad0f73", "- 받은 파일(2010–2020)과 md5:", ""] + [f"  - {k}: {s}, md5 {m}" for k, s, m in log]
(OUT / "SOURCE.md").write_text("\n".join(md) + "\n")
print(json.dumps(log))
