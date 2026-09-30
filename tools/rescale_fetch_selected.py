"""Rescale 작업의 출력 파일 가운데 이름으로 고른 것만 내려받는다(API 키는 출력하지 않는다).

작업 출력 파일이 수천 개이면 전체 목록 조회(tools/rescale_client.py fetch)가 시간 초과된다(2026-09-30 Qjpbeb).
이 도구는 jobs/<id>/files/?search=<이름> 로 파일마다 찾아 relativePath 가 정확히 같거나 basename 이 같은 것만 받는다.
사용: python3 tools/rescale_fetch_selected.py <job_id> <dest> <파일 이름> [<파일 이름> ...]
종료 코드: 0 모두 받음, 3 하나 이상 못 찾음.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescale_client as rc  # noqa: E402


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    job, dest, names = argv[0], Path(argv[1]), argv[2:]
    c = rc.RescaleClient()
    c.timeout = 120
    miss = 0
    for nm in names:
        r = c.session.get(f"{c.base}/jobs/{job}/files/", params={"search": nm, "page_size": 50}, timeout=120)
        if not r.ok:
            print(f"[실패] {nm}: HTTP {r.status_code}")
            miss += 1
            continue
        res = r.json().get("results", [])
        hit = [f for f in res if (f.get("relativePath") or f.get("name")) == nm] or \
              [f for f in res if Path(f.get("relativePath") or f.get("name") or "").name == nm]
        if not hit:
            print(f"[없음] {nm}(검색 결과 {len(res)}개)")
            miss += 1
            continue
        f = hit[0]
        rel = f.get("relativePath") or f.get("name")
        n = c.download(f["id"], dest / rel)
        print(f"  {rel}  ({n / 1e6:.1f} MB)")
    return 3 if miss else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
