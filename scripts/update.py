#!/usr/bin/env python3
"""Live refresh: fetch results, scrape match stats, re-simulate, rebuild dashboard."""

import sys, io, json, argparse, subprocess, hashlib
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT       = Path(__file__).resolve().parent.parent
SCRIPTS    = ROOT / "scripts"
LIVE_JSON  = ROOT / "Data/simulated/live_scores.json"
STATE_FILE = ROOT / "Data/simulated/.update_state.json"


def run(script: str, args: list[str], quiet: bool) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=quiet, text=True, encoding="utf-8")
    if res.returncode != 0:
        if quiet and res.stdout:
            print(res.stdout)
        if quiet and res.stderr:
            print(res.stderr)
        sys.exit(f"{script} failed (exit {res.returncode})")


def results_fingerprint() -> str:
    if not LIVE_JSON.exists():
        return ""
    data = json.loads(LIVE_JSON.read_text(encoding="utf-8"))
    fin = sorted(
        f"{m['home']}|{m['away']}|{m['home_g']}-{m['away_g']}"
        for m in data.get("matches", []) if m.get("status") == "FINISHED"
    )
    return hashlib.sha1("\n".join(fin).encode()).hexdigest()


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(fp: str) -> None:
    STATE_FILE.write_text(json.dumps({"results_fingerprint": fp}), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Live refresh pipeline")
    ap.add_argument("--no-fetch",  action="store_true", help="skip the API fetch")
    ap.add_argument("--no-fotmob", action="store_true", help="skip the match-stats scrape")
    ap.add_argument("--force",     action="store_true", help="rebuild even with no new results")
    ap.add_argument("--quiet",     action="store_true", help="suppress sub-script output")
    args = ap.parse_args()

    before = results_fingerprint()

    if not args.no_fetch:
        run("fetch_results.py", [], args.quiet)

    after = results_fingerprint()
    prev  = load_state().get("results_fingerprint", "")

    if not args.force and after == prev and after == before:
        print(f"No new results (fingerprint {after[:8] or '-'}). Use --force to rebuild anyway.")
        return

    if not args.no_fotmob:
        try:
            run("scraper_fotmob.py", [], args.quiet)
        except SystemExit as e:
            print(f"  match-stats scrape failed ({e}); continuing.")

    run("simulate.py", [], args.quiet)
    run("build_dashboard.py", [], args.quiet)

    save_state(after)

    n = 0
    if LIVE_JSON.exists():
        n = json.loads(LIVE_JSON.read_text(encoding="utf-8")).get("group_finished", 0)
    print(f"Updated - dashboard rebuilt, conditioned on {n}/72 group results.")


if __name__ == "__main__":
    main()
