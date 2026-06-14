#!/usr/bin/env python3
"""Fetch live World Cup results from football-data.org into schedule + live_scores.json."""

import sys, io, os, json, argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("requests not installed — run: pip install requests")

API_BASE      = "https://api.football-data.org/v4"
COMP_CODE     = "WC"                       # FIFA World Cup (id 2000)
SCHEDULE_PATH = Path("Data/schedule_2026.csv")
LIVE_JSON     = Path("Data/simulated/live_scores.json")
ENV_KEY       = "FOOTBALL_DATA_API_KEY"

CANON = {
    "South Korea":            "Korea Republic",
    "Turkey":                 "Türkiye",
    "Türkiye":                "Türkiye",
    "Iran":                   "IR Iran",
    "Czech Republic":         "Czechia",
    "Ivory Coast":            "Côte d'Ivoire",
    "Cote d'Ivoire":          "Côte d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "DR Congo":               "Congo DR",
    "Cabo Verde":             "Cape Verde",
    "USA":                    "United States",
    "Curacao":                "Curaçao",
}

def canon(name: str) -> str:
    return CANON.get(str(name).strip(), str(name).strip())


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from the project-root .env into os.environ."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_api_key() -> str:
    _load_dotenv()
    key = os.environ.get(ENV_KEY, "").strip()
    if not key:
        sys.exit(
            f"No API key. Get a free one: https://www.football-data.org/client/register\n"
            f"  Easiest:  copy .env.example to .env and put your key in it\n"
            f'  Or env:   $env:{ENV_KEY} = "<key>"  (PowerShell)  /  export {ENV_KEY}=<key>  (bash)'
        )
    return key


def fetch_matches(key: str) -> list[dict]:
    """Return all WC matches (any status) from the API."""
    r = requests.get(
        f"{API_BASE}/competitions/{COMP_CODE}/matches",
        headers={"X-Auth-Token": key},
        timeout=30,
    )
    if r.status_code == 403:
        sys.exit("403 from API — key invalid or competition not on your plan.")
    if r.status_code == 429:
        sys.exit("429 rate-limited — wait a minute (free tier = 10 req/min).")
    r.raise_for_status()
    return r.json().get("matches", [])


def write_schedule_scores(matches: list[dict], dry_run: bool) -> int:
    """Inject FINISHED group-stage scores into schedule_2026.csv's Score column.
    Returns the number of fixtures filled."""
    if not SCHEDULE_PATH.exists():
        sys.exit(f"{SCHEDULE_PATH} not found.")
    df = pd.read_csv(SCHEDULE_PATH)
    if "Score" not in df.columns:
        df["Score"] = pd.NA
    df["Score"] = df["Score"].astype("object")

    # order-agnostic key
    gs = df[df["Round"] == "Group stage"]
    fixture_rows: dict[frozenset, int] = {}
    for idx, row in gs.iterrows():
        fixture_rows[frozenset((canon(row["home_team"]), canon(row["away_team"])))] = idx

    filled = 0
    unmatched: list[str] = []
    for m in matches:
        if m.get("stage") != "GROUP_STAGE" or m.get("status") != "FINISHED":
            continue
        h, a = canon(m["homeTeam"]["name"]), canon(m["awayTeam"]["name"])
        ft = m["score"]["fullTime"]
        if ft["home"] is None or ft["away"] is None:
            continue
        key = frozenset((h, a))
        idx = fixture_rows.get(key)
        if idx is None:
            unmatched.append(f"{h} vs {a}")
            continue
        srow = df.loc[idx]
        if canon(srow["home_team"]) == h:
            score = f"{ft['home']}-{ft['away']}"
        else:
            score = f"{ft['away']}-{ft['home']}"
        df.at[idx, "Score"] = score
        filled += 1

    if unmatched:
        print(f"  {len(unmatched)} finished match(es) didn't map to a fixture:")
        for u in unmatched:
            print(f"      {u}  (check CANON spelling)")

    if not dry_run:
        df.to_csv(SCHEDULE_PATH, index=False, encoding="utf-8")
    return filled


def write_live_json(matches: list[dict], dry_run: bool) -> dict:
    """Compact feed for the dashboard. Group stage only for the live strip."""
    feed = []
    n_finished = 0
    for m in matches:
        if m.get("stage") != "GROUP_STAGE":
            continue
        status = m.get("status")
        ft = m["score"]["fullTime"]
        if status == "FINISHED":
            n_finished += 1
        feed.append({
            "utc":     m["utcDate"],
            "group":   (m.get("group") or "").replace("GROUP_", ""),
            "home":    canon(m["homeTeam"]["name"]),
            "away":    canon(m["awayTeam"]["name"]),
            "home_tla": m["homeTeam"].get("tla"),
            "away_tla": m["awayTeam"].get("tla"),
            "home_g":  ft["home"],
            "away_g":  ft["away"],
            "ht_home": (m["score"].get("halfTime") or {}).get("home"),
            "ht_away": (m["score"].get("halfTime") or {}).get("away"),
            "status":  status,
            "minute":  m.get("minute"),
        })
    feed.sort(key=lambda x: x["utc"])
    out = {
        "updated_utc":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "group_finished": n_finished,
        "group_total":    72,
        "matches":        feed,
    }
    if not dry_run:
        LIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
        LIVE_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description="Fetch live WC2026 results")
    ap.add_argument("--dry-run", action="store_true", help="show, don't write files")
    args = ap.parse_args()

    key = get_api_key()
    print(f"Fetching {COMP_CODE} matches from football-data.org ...")
    matches = fetch_matches(key)
    print(f"  {len(matches)} matches returned by API")

    filled = write_schedule_scores(matches, args.dry_run)
    live   = write_live_json(matches, args.dry_run)

    verb = "(dry-run, nothing written)" if args.dry_run else ""
    print(f"  Group-stage results filled into schedule: {filled}/72 {verb}")
    print(f"  live_scores.json: {live['group_finished']} finished, "
          f"{len(live['matches'])} group fixtures total {verb}")
    if not args.dry_run:
        print(f"  -> {SCHEDULE_PATH}")
        print(f"  -> {LIVE_JSON}")
    print("\nNext: python scripts/simulate.py  ->  python scripts/build_dashboard.py")


if __name__ == "__main__":
    main()
