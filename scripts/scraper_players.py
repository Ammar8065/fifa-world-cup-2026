#!/usr/bin/env python3
"""Scrape per-player tournament xG/goals/shots from StatsBomb open data."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import time, requests, logging
from pathlib import Path
from collections import defaultdict
import pandas as pd

RAW_BASE   = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DELAY      = 0.5
OUTPUT_DIR = Path("Data/scraped")
HEADERS    = {"User-Agent": "Mozilla/5.0 (WC-Simulator/1.0)"}

COMPETITIONS = [
    (43,   106, "FIFA World Cup 2022"),
    (55,   282, "UEFA Euro 2024"),
    (223,  282, "Copa America 2024"),
    (1267, 107, "AFCON 2023"),
]

CANON = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    "USA": "United States",
    "Cabo Verde": "Cape Verde",
}
def canon(name): return CANON.get(str(name).strip(), str(name).strip())

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

def fetch_json(url, retries=3):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            log.warning(f"  HTTP {r.status_code} (attempt {attempt}): {url}")
        except Exception as exc:
            log.warning(f"  Error (attempt {attempt}): {exc}")
        time.sleep(DELAY * attempt * 2)
    return None

# Player stats aggregator: {player_name: {team, comp, matches, shots, xg, goals}}
player_stats = defaultdict(lambda: {"team": "", "matches_set": set(),
                                     "shots": 0, "xg": 0.0, "goals": 0,
                                     "comp_matches": defaultdict(set)})

for comp_id, season_id, label in COMPETITIONS:
    log.info(f"\n=== {label} ===")
    matches_url = f"{RAW_BASE}/matches/{comp_id}/{season_id}.json"
    matches = fetch_json(matches_url)
    if not matches:
        continue
    log.info(f"  {len(matches)} matches")
    time.sleep(DELAY)

    for i, match in enumerate(matches, 1):
        match_id = match["match_id"]
        events_url = f"{RAW_BASE}/events/{match_id}.json"
        events = fetch_json(events_url)
        time.sleep(DELAY)
        if not events:
            continue

        shots = [e for e in events if e.get("type", {}).get("name") == "Shot"]
        for shot in shots:
            player_name = shot.get("player", {}).get("name", "Unknown")
            team_name   = canon(shot.get("team", {}).get("name", "Unknown"))
            xg_val      = shot.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
            is_goal     = shot.get("shot", {}).get("outcome", {}).get("name") == "Goal"

            key = (player_name, team_name)
            player_stats[key]["team"] = team_name
            player_stats[key]["comp_matches"][label].add(match_id)
            player_stats[key]["shots"] += 1
            player_stats[key]["xg"]   += xg_val
            player_stats[key]["goals"] += int(is_goal)

        if i % 20 == 0:
            log.info(f"  Processed {i}/{len(matches)} matches...")

log.info(f"\nTotal player-team combos: {len(player_stats)}")

# Build output rows
rows = []
for (player_name, team_name), s in player_stats.items():
    total_matches = sum(len(v) for v in s["comp_matches"].values())
    comps = list(s["comp_matches"].keys())
    rows.append({
        "player":         player_name,
        "team":           team_name,
        "competitions":   "; ".join(comps),
        "matches":        total_matches,
        "shots":          s["shots"],
        "xg":             round(s["xg"], 4),
        "goals":          s["goals"],
        "xg_per_match":   round(s["xg"] / max(total_matches, 1), 4),
        "goals_per_match":round(s["goals"] / max(total_matches, 1), 4),
        "shot_accuracy":  round(s["goals"] / max(s["shots"], 1), 3),
    })

df = pd.DataFrame(rows).sort_values("xg", ascending=False).reset_index(drop=True)
out_path = OUTPUT_DIR / "player_xg.csv"
df.to_csv(out_path, index=False, encoding="utf-8")
log.info(f"\nSaved {len(df)} player records → {out_path}")
log.info("\nTop 20 by total xG:")
for _, r in df.head(20).iterrows():
    log.info(f"  {r['player']:<30s} [{r['team']:<25s}]  xG={r['xg']:6.2f}  Goals={int(r['goals']):3d}  Matches={int(r['matches']):3d}")
