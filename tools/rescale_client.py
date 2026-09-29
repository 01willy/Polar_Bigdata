#!/usr/bin/env python3
"""
Rescale REST API client — submit GPU jobs to Rescale without the web UI.

API base   : https://platform.rescale.com/api/v2/   (regional variants below)
Auth       : header `Authorization: Token <key>`
Key idea   : uploaded files get persistent IDs. Large datasets are uploaded once
             and referenced by ID across many jobs; only code+config is re-uploaded.

The API key is read from $RESCALE_API_KEY or ~/.config/rescale/apiconfig.
It is never written into this repo.

Endpoints used (all verified against Rescale API docs, all need trailing slash):
    POST /files/contents/            upload a file          -> {id, name, ...}
    GET  /files/{id}/contents/       download a file
    GET  /coretypes/                 available hardware     -> paginated
    GET  /analyses/                  available software     -> paginated
    POST /jobs/                      create a job           -> {id, ...}
    POST /jobs/{id}/submit/          start the job
    GET  /jobs/{id}/statuses/        status history         -> paginated
    GET  /jobs/{id}/files/           output files           -> paginated
    GET  /jobs/{id}/runs/            run instances          -> paginated
"""
from pathlib import Path
import argparse
import configparser
import json
import os
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]

# Regional platforms. Rescale accounts live on exactly one of these, and a key
# issued for one region is rejected ("Invalid token") by all the others.
# This project's account is on KR, verified 2026-08-03.
PLATFORMS = {
    "kr": "https://kr.rescale.com",
    "us": "https://platform.rescale.com",
    "eu": "https://eu.rescale.com",
    "jp": "https://platform.rescale.jp",
    "itar": "https://itar.rescale.com",
}
DEFAULT_PLATFORM = "kr"

# Job statuses that mean the job is over, one way or another.
TERMINAL_STATUSES = {"Completed", "Force Stopped", "Stopped", "Failed", "Cancelled"}


class RescaleError(RuntimeError):
    pass


def load_api_key(explicit=None, profile="default"):
    """Resolve the API key: explicit arg -> $RESCALE_API_KEY -> apiconfig file."""
    if explicit:
        return explicit
    env = os.environ.get("RESCALE_API_KEY")
    if env:
        return env.strip()

    cfg_path = Path.home() / ".config" / "rescale" / "apiconfig"
    if cfg_path.exists():
        # Raw: a key containing '%' would blow up ConfigParser's interpolation.
        parser = configparser.RawConfigParser()
        parser.read(cfg_path)
        for section in (profile, "default", "DEFAULT"):
            if parser.has_option(section, "apikey"):
                return parser.get(section, "apikey").strip()

    raise RescaleError(
        "No API key found. Set one of:\n"
        "  export RESCALE_API_KEY=<key>\n"
        f"  or write {cfg_path}:\n"
        "      [default]\n"
        "      apikey = <key>\n"
        "  (chmod 600, and never commit it)"
    )


class RescaleClient:
    def __init__(self, api_key=None, platform=DEFAULT_PLATFORM, timeout=60):
        self.api_key = load_api_key(api_key)
        base = PLATFORMS.get(platform)
        if base is None:
            raise RescaleError(f"Unknown platform {platform!r}. Pick one of {sorted(PLATFORMS)}")
        self.base = f"{base}/api/v2"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Token {self.api_key}"

    # --- low level ---------------------------------------------------------

    def _url(self, path):
        # every Rescale path needs a trailing slash
        path = path.strip("/")
        return f"{self.base}/{path}/"

    def _request(self, method, path, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        resp = self.session.request(method, self._url(path), **kwargs)
        if not resp.ok:
            raise RescaleError(
                f"{method} {self._url(path)} -> HTTP {resp.status_code}\n{resp.text[:800]}"
            )
        return resp

    def _paged(self, path, params=None, limit=None):
        """Follow `next` links, yielding items from `results`."""
        url = self._url(path)
        seen = 0
        while url:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            if not resp.ok:
                raise RescaleError(f"GET {url} -> HTTP {resp.status_code}\n{resp.text[:800]}")
            payload = resp.json()
            # some endpoints return a bare list rather than a paginated envelope
            items = payload.get("results", payload) if isinstance(payload, dict) else payload
            for item in items:
                yield item
                seen += 1
                if limit and seen >= limit:
                    return
            url = payload.get("next") if isinstance(payload, dict) else None
            params = None  # `next` already carries the query string

    # --- reference data ----------------------------------------------------

    def coretypes(self):
        return list(self._paged("coretypes"))

    def analyses(self):
        return list(self._paged("analyses"))

    # --- files -------------------------------------------------------------

    def upload(self, path):
        """Upload one file. Returns the file object (contains the reusable id)."""
        path = Path(path)
        if not path.is_file():
            raise RescaleError(f"Not a file: {path}")
        with path.open("rb") as fh:
            resp = self._request(
                "POST", "files/contents", files={"file": (path.name, fh)}, timeout=None
            )
        return resp.json()

    def download(self, file_id, dest):
        """Stream one file by id to dest. Returns bytes written."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = self.session.get(
            self._url(f"files/{file_id}/contents"), stream=True, timeout=self.timeout
        )
        if not resp.ok:
            raise RescaleError(f"download {file_id} -> HTTP {resp.status_code}")
        written = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
        return written

    # --- jobs --------------------------------------------------------------

    def create_job(self, spec):
        return self._request("POST", "jobs", json=spec).json()

    def submit_job(self, job_id):
        self._request("POST", f"jobs/{job_id}/submit")

    def statuses(self, job_id):
        """Status history, newest first as returned by the API."""
        return list(self._paged(f"jobs/{job_id}/statuses"))

    def latest_status(self, job_id):
        hist = self.statuses(job_id)
        return hist[0].get("status") if hist else None

    def job_files(self, job_id):
        return list(self._paged(f"jobs/{job_id}/files"))

    def runs(self, job_id):
        return list(self._paged(f"jobs/{job_id}/runs"))

    def wait(self, job_id, poll=30, quiet=False):
        """Poll until the job reaches a terminal status. Returns that status."""
        last = None
        while True:
            status = self.latest_status(job_id)
            if status != last and not quiet:
                stamp = time.strftime("%H:%M:%S")
                print(f"  [{stamp}] {status}")
                last = status
            if status in TERMINAL_STATUSES:
                return status
            time.sleep(poll)


def build_job_spec(cfg, input_file_ids):
    """Translate a job config dict into the Rescale job payload."""
    hw = cfg["hardware"]
    analysis = cfg.get("analysis", {"code": "user_included", "version": "0"})
    low_pri = bool(cfg.get("low_priority", False))
    return {
        "name": cfg["name"],
        # ON_DEMAND is the low-priority (preemptible) tier, INSTANT is full price.
        "billingPriorityValue": "ON_DEMAND" if low_pri else "INSTANT",
        "isLowPriority": low_pri,
        "jobanalyses": [
            {
                "command": cfg["command"],
                "analysis": {"code": analysis["code"], "version": str(analysis["version"])},
                "hardware": {
                    "coreType": hw["core_type"],
                    "coresPerSlot": int(hw["cores_per_slot"]),
                    "slots": int(hw.get("slots", 1)),
                    # walltime is the billing cap. The account also enforces
                    # maxJobHours=48 on top of whatever is set here.
                    "walltime": int(hw["walltime_hours"]),
                },
                "inputFiles": [{"id": fid} for fid in input_file_ids],
                "useRescaleLicense": bool(cfg.get("use_rescale_license", False)),
            }
        ],
    }


ACCOUNT_MAX_JOB_HOURS = 48  # from users/me.maxJobHours, verified 2026-08-03


def estimate_core_hours(cfg):
    hw = cfg["hardware"]
    return int(hw["cores_per_slot"]) * int(hw.get("slots", 1)) * int(hw["walltime_hours"])


def price_for(client, cfg):
    """(usd_per_hour, worst_case_usd) for a config. Price is PER CORE-HOUR."""
    hw = cfg["hardware"]
    ct = next((c for c in client.coretypes() if c.get("code") == hw["core_type"]), None)
    if ct is None:
        raise RescaleError(
            f"Unknown core_type {hw['core_type']!r}. "
            f"List valid codes with: rescale_client.py coretypes --gpu-only"
        )
    allowed = [int(x) for x in (ct.get("cores") or [])]
    if allowed and int(hw["cores_per_slot"]) not in allowed:
        raise RescaleError(
            f"cores_per_slot={hw['cores_per_slot']} is not allowed for "
            f"{hw['core_type']}; pick one of {allowed}"
        )
    field = "lowPriorityPrice" if cfg.get("low_priority") else "price"
    unit = ct.get(field) or ct.get("price")
    if unit is None:
        raise RescaleError(f"{hw['core_type']} has no {field}; low priority may be unavailable")
    per_hour = float(unit) * int(hw["cores_per_slot"]) * int(hw.get("slots", 1))
    return per_hour, per_hour * int(hw["walltime_hours"])


LEDGER = ROOT / "results" / "rescale" / "_ledger.csv"
LEDGER_COLS = [
    "submitted_at", "job_id", "name", "core_type", "cores", "slots",
    "walltime_hours", "low_priority", "usd_per_hour", "worst_case_usd",
    "status", "elapsed_hours", "actual_usd",
]


def ledger_append(row):
    """Append one job to the local spend ledger. There is no billing API, so this
    file is our only programmatic record of what we have spent."""
    import csv
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_COLS})


# --- CLI -------------------------------------------------------------------


def max_gpus(ct):
    """Real GPU count for a coretype. gpuCounts is a per-core-option list and is
    [0.0] on CPU-only types, so a non-empty list is NOT enough to call it a GPU."""
    return max([float(g) for g in (ct.get("gpuCounts") or [0])], default=0.0)


def cmd_coretypes(client, args):
    rows = client.coretypes()
    if args.gpu_only:
        rows = [c for c in rows if max_gpus(c) > 0]
    rows.sort(key=lambda c: float(c.get("price") or 0))

    print(f"{'code':<30}{'name':<24}{'GPU':>5}{'cores':<16}"
          f"{'$/core-hr':>10}{'low-pri':>9}  processor")
    print("-" * 120)
    for c in rows:
        cores = ",".join(str(int(x)) for x in (c.get("cores") or []))
        print(
            f"{str(c.get('code','')):<30}{str(c.get('name',''))[:23]:<24}"
            f"{max_gpus(c):>5.0f}{cores[:15]:<16}"
            f"{str(c.get('price') or '-'):>10}{str(c.get('lowPriorityPrice') or '-'):>9}"
            f"  {str(c.get('processorInfo') or '')[:40]}"
        )
    print(f"\n{len(rows)} coretype(s). Price is PER CORE-HOUR: "
          f"cost/hr = price x cores x slots.")


def cmd_analyses(client, args):
    for a in client.analyses():
        versions = [v.get("versionCode") or v.get("version") for v in a.get("versions", [])]
        print(f"{a.get('code',''):<24} {a.get('name','')[:40]:<42} {versions[:6]}")


def cmd_upload(client, args):
    for path in args.paths:
        info = client.upload(path)
        size = info.get("decryptedSize") or info.get("size") or 0
        print(f"{info['id']}  {info.get('name','')}  {size/1e6:.1f} MB")


def cmd_status(client, args):
    print(f"job {args.job_id}: {client.latest_status(args.job_id)}")
    for s in client.statuses(args.job_id):
        print(f"  {s.get('statusDate','')}  {s.get('status','')}")


def cmd_run(client, args):
    """Full flow: upload code -> create job -> confirm -> submit -> wait -> fetch."""
    import yaml

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    try:
        shown_path = cfg_path.relative_to(ROOT)
    except ValueError:  # config kept outside the repo
        shown_path = cfg_path
    print(f"config      : {shown_path}")
    print(f"job name    : {cfg['name']}")

    hw = cfg["hardware"]

    # 1. VALIDATE AND PRICE FIRST — before uploading anything. A bad core_type or
    #    an over-budget config should fail without having pushed bytes.
    wall = int(hw["walltime_hours"])
    if wall > ACCOUNT_MAX_JOB_HOURS:
        raise RescaleError(
            f"walltime_hours={wall} exceeds the account cap of "
            f"{ACCOUNT_MAX_JOB_HOURS}h (users/me.maxJobHours); the job would be rejected"
        )
    # Project policy (2026-08-03): low priority is preemptible, and a long
    # training run killed mid-flight costs more in wasted time than the tier
    # saves. Refuse it unless someone deliberately overrides on the CLI.
    if cfg.get("low_priority") and not getattr(args, "allow_low_priority", False):
        raise RescaleError(
            "low_priority: true is disabled by project policy — the preemptible "
            "tier can kill a running job. Set low_priority: false, or pass "
            "--allow-low-priority if you have accepted the preemption risk."
        )

    per_hour, worst = price_for(client, cfg)
    tier = "LOW PRIORITY (preemptible)" if cfg.get("low_priority") else "INSTANT (full price)"

    print(f"\nhardware    : {hw['core_type']}  "
          f"{hw['cores_per_slot']} core/slot x {hw.get('slots',1)} slot")
    print(f"billing     : {tier}")
    print(f"walltime cap: {wall} h   (account cap {ACCOUNT_MAX_JOB_HOURS} h)")
    print(f"rate        : ${per_hour:,.2f}/hour")
    print(f"WORST CASE  : ${worst:,.2f}  if it runs the full walltime")
    print(f"command     : {cfg['command'].strip().splitlines()[0]} ...")

    cap = cfg.get("max_cost_usd")
    if cap is not None and worst > float(cap):
        raise RescaleError(
            f"worst-case ${worst:,.2f} exceeds this config's max_cost_usd "
            f"(${float(cap):,.2f}). Lower walltime_hours/cores, enable low_priority, "
            f"or raise max_cost_usd deliberately."
        )

    if args.dry_run:
        print("\n--dry-run: nothing uploaded, nothing submitted.")
        print(json.dumps(build_job_spec(cfg, ["<pending-upload>"]), indent=2))
        return

    if not args.yes:
        reply = input(f"\nsubmit at up to ${worst:,.2f}? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("aborted.")
            return

    # 2. upload the small code/config payload fresh for this run
    file_ids = []
    for rel in cfg.get("code_paths", []):
        src = ROOT / rel
        if not src.is_file():
            raise RescaleError(f"code_paths entry is not a file: {src}")
        info = client.upload(src)
        file_ids.append(info["id"])
        size = info.get("decryptedSize") or info.get("size") or 0
        print(f"  uploaded  {rel}  ({size/1e3:.1f} kB)  -> {info['id']}")

    # 3. reuse previously-uploaded datasets by id (never re-upload the big ones)
    reused = cfg.get("dataset_file_ids") or []
    for fid in reused:
        print(f"  reusing   dataset file id {fid}")
    file_ids.extend(reused)

    job = client.create_job(build_job_spec(cfg, file_ids))
    job_id = job["id"]
    client.submit_job(job_id)
    print(f"\nsubmitted   : job {job_id}")
    print(f"  web       : https://kr.rescale.com/jobs/{job_id}/")

    ledger_append({
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "job_id": job_id, "name": cfg["name"], "core_type": hw["core_type"],
        "cores": hw["cores_per_slot"], "slots": hw.get("slots", 1),
        "walltime_hours": wall, "low_priority": bool(cfg.get("low_priority")),
        "usd_per_hour": f"{per_hour:.4f}", "worst_case_usd": f"{worst:.2f}",
        "status": "submitted",
    })
    print(f"  ledger    : {LEDGER.relative_to(ROOT)}")

    if args.no_wait:
        print(f"  poll with : python3 tools/rescale_client.py status {job_id}")
        return

    final = client.wait(job_id, poll=args.poll)
    print(f"finished    : {final}")

    dest = ROOT / (args.dest or cfg.get("results_dir") or f"results/rescale_{job_id}")
    args.job_id, args.dest = job_id, dest
    cmd_fetch(client, args)


def cmd_fetch(client, args):
    files = client.job_files(args.job_id)
    if not files:
        print("No output files (job may still be running).")
        return
    dest_dir = Path(args.dest)
    total = 0
    for f in files:
        rel = f.get("relativePath") or f.get("name")
        n = client.download(f["id"], dest_dir / rel)
        total += n
        print(f"  {rel}  ({n/1e6:.1f} MB)")
    print(f"\n{len(files)} file(s), {total/1e6:.1f} MB -> {dest_dir}")


def billed_hours(client, job_id):
    """Hours between Executing and the terminal status.

    Rescale's `elapsedWalltimeSeconds` came back null on a completed job, so we
    derive it from the status history instead. Returns None while still running.
    """
    from datetime import datetime

    def parse(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    hist = client.statuses(job_id)
    start = next((s for s in hist if s.get("status") == "Executing"), None)
    end = next((s for s in hist if s.get("status") in TERMINAL_STATUSES), None)
    if not start or not end:
        return None
    return max(0.0, (parse(end["statusDate"]) - parse(start["statusDate"])).total_seconds() / 3600)


def cmd_cost(client, args):
    """Reconcile the local ledger against actual runtimes. There is no billing
    API on Rescale, so this ledger is our only programmatic spend record."""
    import csv

    if not LEDGER.exists():
        print(f"No ledger yet at {LEDGER.relative_to(ROOT)} — nothing submitted through `run`.")
        return
    rows = list(csv.DictReader(LEDGER.open()))
    if not rows:
        print("Ledger is empty.")
        return

    print(f"{'submitted':<20}{'job':<10}{'name':<22}{'core_type':<20}"
          f"{'pri':<5}{'$/hr':>8}{'hrs':>7}{'actual$':>10}  status")
    print("-" * 112)
    total, pending = 0.0, 0
    for r in rows:
        jid = r["job_id"]
        try:
            status = client.latest_status(jid) or "?"
            hrs = billed_hours(client, jid)
        except RescaleError:
            status, hrs = "unreachable", None
        rate = float(r["usd_per_hour"])
        if hrs is None:
            actual, shown = None, "running"
            pending += 1
        else:
            actual = rate * hrs
            total += actual
            shown = f"{actual:,.2f}"
        print(f"{r['submitted_at'][:19]:<20}{jid:<10}{r['name'][:21]:<22}"
              f"{r['core_type'][:19]:<20}{'low' if r['low_priority']=='True' else 'inst':<5}"
              f"{rate:>8.2f}{(f'{hrs:.2f}' if hrs is not None else '-'):>7}"
              f"{shown:>10}  {status}")

    print("-" * 112)
    print(f"{len(rows)} job(s).  Settled spend: ${total:,.2f}"
          + (f"   ({pending} still running)" if pending else ""))
    print("\nEstimated from list price x measured runtime. Rescale exposes no billing\n"
          "API, so the authoritative invoice is the web UI. Treat this as a guardrail.")


def main():
    ap = argparse.ArgumentParser(description="Rescale REST API client")
    ap.add_argument("--platform", default=os.environ.get("RESCALE_PLATFORM", DEFAULT_PLATFORM),
                    choices=sorted(PLATFORMS))
    ap.add_argument("--api-key", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("coretypes", help="list available hardware (find your GPU code here)")
    p.add_argument("--gpu-only", action="store_true")
    p.set_defaults(func=cmd_coretypes)

    p = sub.add_parser("analyses", help="list available software codes")
    p.set_defaults(func=cmd_analyses)

    p = sub.add_parser("upload", help="upload file(s), print reusable file ids")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("status", help="show a job's status history")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("cost", help="spend ledger: submitted jobs, runtimes, estimated cost")
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("run", help="end-to-end: upload, create, submit, wait, fetch")
    p.add_argument("config", help="path to a job config yaml, e.g. configs/rescale/smoke_test.yaml")
    p.add_argument("--yes", "-y", action="store_true", help="skip the submit confirmation")
    p.add_argument("--dry-run", action="store_true", help="print the payload, submit nothing")
    p.add_argument("--no-wait", action="store_true", help="submit and return immediately")
    p.add_argument("--allow-low-priority", action="store_true",
                   help="override the project policy that blocks the preemptible tier")
    p.add_argument("--poll", type=int, default=30, help="status poll interval, seconds")
    p.add_argument("--dest", default=None, help="override results_dir from the config")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("fetch", help="download a job's output files")
    p.add_argument("job_id")
    p.add_argument("--dest", default="results/rescale")
    p.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    try:
        client = RescaleClient(api_key=args.api_key, platform=args.platform)
        args.func(client, args)
    except RescaleError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
