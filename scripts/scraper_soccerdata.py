#!/usr/bin/env python3
"""
Current-season player xG scraper (Understat via soccerdata).

Pulls 2025-26 per-player expected-goals data from the big-5 European leagues
(+ RFPL) and writes a club-level table to Data/scraped/player_xg_current.csv.

This replaces the stale 2022-24 StatsBomb tournament data as the primary signal
for the Top Scorer / Player-of-the-Tournament / Young Player predictions. The
club rows are matched to real 2026 national-team squads (by player name) inside
build_dashboard.py, so a player's *current form* drives the awards.

Understat carries true xG / non-penalty xG / xA per player — the cleanest public
xG source — which is a big upgrade over event-derived tournament xG from years ago.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import logging
from pathlib import Path
import pandas as pd
import soccerdata as sd

SEASON      = "2526"          # current season (2025-26); falls back to 2425 per-league if empty
FALLBACK    = "2425"
LEAGUES     = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
    "RUS-Premier League",
]
OUTPUT_DIR  = Path("Data/scraped")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH    = OUTPUT_DIR / "player_xg_current.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("scraper_soccerdata")


def pull_league(league: str) -> pd.DataFrame:
    """Return the current-season player stats for one league (falls back a season if empty)."""
    for season in (SEASON, FALLBACK):
        try:
            us = sd.Understat(leagues=league, seasons=season)
            df = us.read_player_season_stats()
        except Exception as exc:
            log.warning(f"  {league} {season}: error {exc}")
            continue
        if df is not None and len(df):
            df = df.reset_index()
            df["season_used"] = season
            log.info(f"  {league:<22s} {season}  →  {len(df):4d} players  "
                     f"(Σxg={df['xg'].sum():.0f})")
            return df
        log.warning(f"  {league} {season}: empty, trying fallback")
    return pd.DataFrame()


def main():
    log.info(f"Pulling current-season ({SEASON}) player xG from Understat …")
    frames = [pull_league(lg) for lg in LEAGUES]
    frames = [f for f in frames if len(f)]
    if not frames:
        log.error("No data pulled — aborting (check network / TLS library).")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)

    # Per-match / per-90 rates used downstream for ranking.
    df["matches"]  = df["matches"].clip(lower=0)
    nineties       = (df["minutes"] / 90.0).replace(0, pd.NA)
    df["xg_per_match"]    = (df["xg"]    / df["matches"].replace(0, pd.NA)).round(4)
    df["goals_per_match"] = (df["goals"] / df["matches"].replace(0, pd.NA)).round(4)
    df["np_xg_per90"]     = (df["np_xg"] / nineties).round(4)
    df["shot_accuracy"]   = (df["goals"] / df["shots"].replace(0, pd.NA)).round(3)

    out = df.rename(columns={"team": "club", "league": "competition"})[[
        "player", "club", "competition", "season_used", "position",
        "matches", "minutes", "shots",
        "goals", "xg", "np_goals", "np_xg", "assists", "xa",
        "xg_chain", "xg_buildup",
        "xg_per_match", "goals_per_match", "np_xg_per90", "shot_accuracy",
    ]].copy()

    # Round float columns for a tidy CSV.
    for c in ["xg", "np_xg", "xa", "xg_chain", "xg_buildup"]:
        out[c] = out[c].round(3)

    out = out.sort_values("xg", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    log.info(f"\nSaved {len(out)} club-season player records → {OUT_PATH}")
    log.info("\nTop 15 by current-season xG:")
    for _, r in out.head(15).iterrows():
        log.info(f"  {r['player']:<26s} [{r['club']:<20s}]  "
                 f"xG={r['xg']:5.1f}  G={int(r['goals']):3d}  M={int(r['matches']):3d}")


if __name__ == "__main__":
    main()
