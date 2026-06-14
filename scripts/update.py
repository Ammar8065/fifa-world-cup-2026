#!/usr/bin/env python3
"""
One-command live refresh: fetch results → re-simulate → rebuild dashboard.
=========================================================================
Runs the full live-update pipeline so every projection (champion %, group
standings, knockout odds, awards) reflects the latest real results:

    1. fetch_results.py   — pull finished scores from football-data.org
    2. scraper_fotmob.py  — scrape FotMob xG + Player of the Match (headless)
    3. simulate.py        — Monte Carlo CONDITIONED on those results
    4. build_dashboard.py — regenerate dashboard.html (+ live strip/tab)

Usage
-----
    export FOOTBALL_DATA_API_KEY=<key>      # PowerShell: $env:FOOTBALL_DATA_API_KEY="<key>"
    python scripts/update.py                # full pipeline
    python scripts/update.py --no-fetch     # skip API call, re-sim on existing scores
    python scripts/update.py --no-fotmob    # skip the FotMob xG/MOTM scrape
    python scripts/update.py --quiet        # only print the summary line

Designed to be driven on an interval on match days (see the /loop note in
the README), or run by hand. Skips the expensive re-sim+rebuild when the
fetch finds no NEW finished results since the last run.
"""

import sys, io, os, json, argparse, subprocess, hashlib
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT       = Path(__file__).resolve().parent.parent
SCRIPTS    = ROOT / "scripts"
LIVE_JSON  = ROOT / "Data/simulated/live_scores.json"
STATE_FILE = ROOT / "Data/simulated/.update_state.json"   # gitignored


def run(script: str, args: list[str], quiet: bool) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=quiet, text=True, encoding="utf-8")
    if res.returncode != 0:
        if quiet and res.stdout:
            print(res.stdout)
        if quiet and res.stderr:
            print(res.stderr)
        sys.exit(f"✗ {script} failed (exit {res.returncode})")


def results_fingerprint() -> str:
    """Hash of the finished-match scores so we can detect 'anything new?'."""
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
    ap = argparse.ArgumentParser(description="Live WC2026 refresh pipeline")
    ap.add_argument("--no-fetch",  action="store_true", help="skip the API fetch")
    ap.add_argument("--no-fotmob", action="store_true", help="skip the FotMob xG/MOTM scrape")
    ap.add_argument("--force",     action="store_true", help="re-sim even if no new results")
    ap.add_argument("--quiet",    action="store_true", help="suppress sub-script output")
    args = ap.parse_args()

    before = results_fingerprint()

    # 1. Fetch live results (unless told not to)
    if not args.no_fetch:
        run("fetch_results.py", [], args.quiet)

    after = results_fingerprint()
    prev  = load_state().get("results_fingerprint", "")

    # Skip the expensive steps when nothing changed since last successful run
    if not args.force and after == prev and after == before:
        print(f"No new results since last update (fingerprint {after[:8] or '∅'}). "
              f"Skipping re-sim. Use --force to rebuild anyway.")
        return

    # 2. Scrape FotMob xG + Player of the Match (non-fatal — unofficial source)
    if not args.no_fotmob:
        try:
            run("scraper_fotmob.py", [], args.quiet)
        except SystemExit as e:
            print(f"  ⚠ FotMob scrape failed ({e}); continuing without xG/MOTM.")

    # 3. Re-simulate (conditioned on the updated scores)
    run("simulate.py", [], args.quiet)

    # 4. Rebuild the dashboard
    run("build_dashboard.py", [], args.quiet)

    save_state(after)

    n = 0
    if LIVE_JSON.exists():
        n = json.loads(LIVE_JSON.read_text(encoding="utf-8")).get("group_finished", 0)
    print(f"✓ Updated — dashboard rebuilt, conditioned on {n}/72 group results.")


if __name__ == "__main__":
    main()
