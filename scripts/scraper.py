#!/usr/bin/env python3
"""
StatsBomb Open-Data Scraper — 2026 World Cup Simulation
=========================================================
Source : github.com/statsbomb/open-data  (free, no auth required)

Competitions scraped
--------------------
  FIFA World Cup 2022     — comp 43  season 106   64 matches
  UEFA Euro 2024          — comp 55  season 282   51 matches
  Copa America 2024       — comp 223 season 282   32 matches
  AFCON 2023              — comp 1267 season 107  52 matches

For every match we download the events JSON, extract all Shot events,
and compute per-team per-match:
  xg_for, xg_against, goals_for, goals_against, result (W/D/L)

Outputs
-------
  Data/scraped/raw_matches.csv        — one row per team per match
  Data/scraped/team_tournament_agg.csv — per-team aggregates by competition
  Data/scraped/team_overall_agg.csv   — per-team aggregates across all comps
  Data/scraped/scrape_log.txt         — full run log
"""

import re
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DELAY    = 1.0   # seconds between requests (GitHub is generous, still be polite)
OUTPUT_DIR = Path("Data/scraped")

HEADERS = {"User-Agent": "Mozilla/5.0 (World-Cup-Simulator/1.0)"}

# Competitions to scrape: (competition_id, season_id, label)
COMPETITIONS = [
    (43,   106, "FIFA World Cup 2022"),
    (55,   282, "UEFA Euro 2024"),
    (223,  282, "Copa America 2024"),
    (1267, 107, "AFCON 2023"),
]

# ── Name normaliser ───────────────────────────────────────────────────────────
# Map StatsBomb team names → schedule_2026.csv names where they differ
SB_NAME_MAP = {
    "United States":                "United States",
    "Republic of Ireland":          "Ireland",
    "Korea Republic":               "Korea Republic",
    "Republic of Korea":            "Korea Republic",
    "South Korea":                  "Korea Republic",
    "DR Congo":                     "Congo DR",
    "Congo DR":                     "Congo DR",
    "Democratic Republic of Congo": "Congo DR",
    "Ivory Coast":                  "Côte d'Ivoire",
    "Cote d'Ivoire":                "Côte d'Ivoire",
    "Iran":                         "IR Iran",
    "Turkey":                       "Türkiye",
    "Czech Republic":               "Czechia",
    "Cape Verde":                   "Cape Verde",
    "Curacao":                      "Curaçao",
    "Bosnia & Herzegovina":         "Bosnia-Herzegovina",
}

def normalise(name: str) -> str:
    return SB_NAME_MAP.get(name, name)


# ── Logging ───────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "scrape_log.txt", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── HTTP helper ───────────────────────────────────────────────────────────────
def fetch_json(url: str, retries: int = 3) -> Optional[list | dict]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            log.warning(f"  HTTP {r.status_code} (attempt {attempt}): {url}")
        except Exception as exc:
            log.warning(f"  Error (attempt {attempt}): {exc}")
        time.sleep(DELAY * attempt * 2)
    log.error(f"  FAILED after {retries} attempts: {url}")
    return None


# ── Process one competition ───────────────────────────────────────────────────
def process_competition(comp_id: int, season_id: int, label: str) -> list[dict]:
    """Fetch all matches + events for one competition. Returns list of row dicts."""
    log.info(f"\n{'='*55}")
    log.info(f"  {label}  (comp={comp_id}, season={season_id})")
    log.info(f"{'='*55}")

    # 1. Match list
    matches_url = f"{RAW_BASE}/matches/{comp_id}/{season_id}.json"
    matches = fetch_json(matches_url)
    if not matches:
        log.error(f"Could not fetch match list for {label}")
        return []
    log.info(f"  Matches: {len(matches)}")
    time.sleep(DELAY)

    rows = []
    for i, match in enumerate(matches, start=1):
        match_id   = match["match_id"]
        date       = match.get("match_date", "")
        stage      = match.get("competition_stage", {}).get("name", "")
        home_team  = normalise(match["home_team"]["home_team_name"])
        away_team  = normalise(match["away_team"]["away_team_name"])
        home_score = match.get("home_score", 0) or 0
        away_score = match.get("away_score", 0) or 0

        log.info(f"  [{i:02d}/{len(matches)}] {home_team} {home_score}-{away_score} {away_team}  ({stage})")

        # 2. Events for this match
        events_url = f"{RAW_BASE}/events/{match_id}.json"
        events = fetch_json(events_url)
        time.sleep(DELAY)

        if not events:
            log.warning(f"    No events — skipping xG for this match")
            home_xg = away_xg = None
        else:
            # Sum xG by team from Shot events
            shots = [e for e in events if e.get("type", {}).get("name") == "Shot"]
            xg_map: dict[str, float] = {}
            for shot in shots:
                team = normalise(shot["team"]["name"])
                xg   = shot.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
                xg_map[team] = xg_map.get(team, 0.0) + xg

            home_xg = round(xg_map.get(home_team, 0.0), 4)
            away_xg = round(xg_map.get(away_team, 0.0), 4)
            log.info(f"    xG: {home_team} {home_xg:.2f} — {away_team} {away_xg:.2f}  (shots: {len(shots)})")

        # 3. Emit two rows (one per team perspective)
        for side, team, opp, gf, ga, xgf, xga in [
            ("home", home_team, away_team, home_score, away_score, home_xg, away_xg),
            ("away", away_team, home_team, away_score, home_score, away_xg, home_xg),
        ]:
            result = "W" if gf > ga else ("D" if gf == ga else "L")
            rows.append({
                "competition": label,
                "stage":       stage,
                "date":        date,
                "match_id":    match_id,
                "team":        team,
                "opponent":    opp,
                "venue":       side,   # home/away (in this tournament, not country)
                "gf":          gf,
                "ga":          ga,
                "xg":          xgf,
                "xga":         xga,
                "xgd":         round((xgf or 0) - (xga or 0), 4) if xgf is not None else None,
                "gd":          gf - ga,
                "result":      result,
                "points":      3 if result == "W" else (1 if result == "D" else 0),
            })

    log.info(f"\n  → {len(rows)//2} matches processed for {label}")
    return rows


# ── Aggregate helpers ─────────────────────────────────────────────────────────
NUMERIC = ["gf", "ga", "xg", "xga", "xgd", "gd", "points"]

def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols).agg(
        matches   = ("result", "count"),
        wins      = ("result", lambda x: (x == "W").sum()),
        draws     = ("result", lambda x: (x == "D").sum()),
        losses    = ("result", lambda x: (x == "L").sum()),
        goals_for = ("gf",    "sum"),
        goals_against = ("ga", "sum"),
        xg_total  = ("xg",   lambda x: round(x.sum(), 3)),
        xga_total = ("xga",  lambda x: round(x.sum(), 3)),
        avg_gf    = ("gf",   lambda x: round(x.mean(), 3)),
        avg_ga    = ("ga",   lambda x: round(x.mean(), 3)),
        avg_xg    = ("xg",   lambda x: round(x.mean(), 3)),
        avg_xga   = ("xga",  lambda x: round(x.mean(), 3)),
        avg_xgd   = ("xgd",  lambda x: round(x.mean(), 3)),
        clean_sheets = ("ga", lambda x: (x == 0).sum()),
        total_points = ("points", "sum"),
    ).reset_index()

    agg["win_rate"]          = (agg["wins"]  / agg["matches"]).round(3)
    agg["clean_sheet_rate"]  = (agg["clean_sheets"] / agg["matches"]).round(3)
    agg["pts_per_game"]      = (agg["total_points"] / agg["matches"]).round(3)
    return agg


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 55)
    log.info("  StatsBomb Open-Data Scraper — 2026 WC Simulation")
    log.info(f"  Competitions: {len(COMPETITIONS)}")
    log.info(f"  Output: {OUTPUT_DIR.resolve()}")
    log.info("=" * 55)

    all_rows: list[dict] = []

    for comp_id, season_id, label in COMPETITIONS:
        rows = process_competition(comp_id, season_id, label)
        all_rows.extend(rows)

    if not all_rows:
        log.error("No data collected — check log for errors")
        return

    # ── Save raw match rows ───────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    raw_path = OUTPUT_DIR / "raw_matches.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8")
    log.info(f"\nRaw match rows → {raw_path}  ({len(df):,} rows)")

    # ── Per-team per-competition aggregates ───────────────────────────────────
    agg_by_comp = aggregate(df, ["team", "competition"])
    comp_path = OUTPUT_DIR / "team_tournament_agg.csv"
    agg_by_comp.to_csv(comp_path, index=False, encoding="utf-8")
    log.info(f"Tournament aggregates → {comp_path}  ({len(agg_by_comp)} rows)")

    # ── Per-team overall aggregates (across all comps) ────────────────────────
    agg_overall = aggregate(df, ["team"])
    overall_path = OUTPUT_DIR / "team_overall_agg.csv"
    agg_overall.to_csv(overall_path, index=False, encoding="utf-8")
    log.info(f"Overall aggregates   → {overall_path}  ({len(agg_overall)} teams)")

    # ── Quick summary table ───────────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("TOP 15 TEAMS BY AVG xG DIFFERENTIAL (across all comps)")
    log.info("=" * 55)
    top = agg_overall.sort_values("avg_xgd", ascending=False).head(15)
    for _, row in top.iterrows():
        log.info(
            f"  {row['team']:<30s}  matches={int(row['matches']):3d}  "
            f"avg_xg={row['avg_xg']:.2f}  avg_xga={row['avg_xga']:.2f}  "
            f"xgd={row['avg_xgd']:+.2f}  win%={row['win_rate']:.0%}"
        )

    log.info("\n" + "=" * 55)
    log.info("DONE")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
