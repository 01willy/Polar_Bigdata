"""v5 보강: 라벨 지점(17,385 고유 250 m 화소, 16,768 고유 500 m 화소)의 MODIS 식생·적설을 2010–2024 전 기간으로 AppEEARS 에 다시 요청한다.
기존 xe_stage2_acquire.py 의 로그인·과제 작성 함수를 그대로 쓴다. 산출은 data/raw/appeears_v5/{requests,results}/.
기간: MOD13Q1 2010–2024 의 06-01 – 08-31(반복), MOD10A2 2009-09-01 – 2024-09-30. 값 수 상한(약 3.5 × 10⁶)을 넘지 않도록 요청당 점 수를 정한다.
하위 명령: submit | status [--fetch]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "0_download"))
import xe_stage2_acquire as XE  # noqa: E402

D = ROOT / "data" / "raw" / "appeears_v5"; (D / "requests").mkdir(parents=True, exist_ok=True); (D / "results").mkdir(exist_ok=True)
DATES = {"MOD13Q1.061": [{"startDate": "06-01", "endDate": "08-31", "recurring": True, "yearRange": [2010, 2024]}],
         "MOD10A2.061": [{"startDate": "09-01-2009", "endDate": "09-30-2024"}]}
CHUNK = {"MOD13Q1.061": 1000, "MOD10A2.061": 500}        # MOD13Q1: 1000점 × 90일/16일≈6 × 15년 × 7층 ≈ 6.3e5. MOD10A2: 500점 × 690합성 × 2층 ≈ 6.9e5


def submit(a):
    import requests
    pix = pd.read_csv(ROOT / "data" / "raw" / "appeears_xe" / "points" / "xe_labels_pixels.csv")
    sub_p = D / "requests" / "submitted.json"
    subs = json.loads(sub_p.read_text()) if sub_p.exists() else {}
    token = XE.ae_login(); hdr = {"Authorization": f"Bearer {token}"}
    for prod, res in (("MOD13Q1.061", 250), ("MOD10A2.061", 500)):
        u = pix.drop_duplicates(f"pix{res}_id").sort_values(f"pix{res}_id").reset_index(drop=True)
        n = int(np.ceil(len(u) / CHUNK[prod]))
        for k in range(n):
            name = f"v5_{prod.split('.')[0].lower()}_labels_p{k + 1:02d}of{n:02d}"
            if name in subs and subs[name].get("task_id"):
                continue
            part = u.iloc[k * CHUNK[prod]:(k + 1) * CHUNK[prod]]
            coords = [{"id": pid, "category": tg, "latitude": float(la), "longitude": float(lo)}
                      for pid, tg, la, lo in zip(part[f"pix{res}_id"], part["target"], part[f"pix{res}_lat"], part[f"pix{res}_lon"])]
            task = XE.ae_task(name, coords, prod, DATES[prod])
            (D / "requests" / f"{name}.json").write_text(json.dumps(task))
            r = requests.post(f"{XE.AE}/task", json=task, headers=hdr, timeout=300)
            try:
                body = r.json()
            except Exception:
                body = {"text": r.text[:300]}
            subs[name] = dict(product=prod, n_points=len(coords), http=r.status_code, task_id=body.get("task_id"), response=body,
                              submitted=time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"[submit] {name} HTTP {r.status_code} {body.get('task_id', body)}", flush=True)
            sub_p.write_text(json.dumps(subs, ensure_ascii=False, indent=1))
            time.sleep(2)


def status(a):
    import requests
    sub_p = D / "requests" / "submitted.json"; subs = json.loads(sub_p.read_text())
    token = XE.ae_login(); hdr = {"Authorization": f"Bearer {token}"}
    st = {}
    for name, rec in subs.items():
        tid = rec.get("task_id")
        if not tid:
            st[name] = "not_submitted"; continue
        r = requests.get(f"{XE.AE}/task/{tid}", headers=hdr, timeout=60); j = r.json()
        st[name] = j.get("status", "?")
        if a.fetch and st[name] == "done" and not (D / "results" / name).exists():
            out = D / "results" / name; out.mkdir(parents=True, exist_ok=True)
            b = requests.get(f"{XE.AE}/bundle/{tid}", headers=hdr, timeout=120).json()
            for f in b.get("files", []):
                if f["file_name"].endswith((".csv", ".json", ".txt")):
                    rr = requests.get(f"{XE.AE}/bundle/{tid}/{f['file_id']}", headers=hdr, timeout=600, stream=True)
                    with open(out / Path(f["file_name"]).name, "wb") as fh:
                        for ch in rr.iter_content(1 << 20):
                            fh.write(ch)
            print(f"[fetch] {name} 받음", flush=True)
    from collections import Counter
    print(json.dumps(dict(Counter(st.values())), ensure_ascii=False))
    (D / "requests" / "status.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["submit", "status"]); ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args(); {"submit": submit, "status": status}[a.cmd](a)
