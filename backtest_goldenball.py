#!/usr/bin/env python3
"""
Golden Ball backtest — World Cup 2022
=====================================
Validates the Golden Ball scoring logic (attacking output × how deep the team
runs) by replaying it on the 2022 World Cup and comparing the predicted ranking
to the ACTUAL Golden Ball result (Messi 1st, Mbappé 2nd, Modrić 3rd).

Honest scope: the live 2026 model blends *pre-tournament* 2025-26 club form with
recent-tournament form and the team's projected run. A true pre-tournament backtest
would need each player's 2021-22 club-season xG, which isn't in this repo. So this
validates the **core hypothesis the model encodes** — that the award goes to a
high-output player on a deep-running team — using WC22 tournament output (StatsBomb)
weighted by each team's actual stage reached. If the logic is sound it should rank
the real winners at the top.

Source: StatsBomb open data (competition 43, season 106).
Output: Data/data/processed/gb_backtest_2022.json  (for the dashboard)
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
from collections import defaultdict
import requests
import pandas as pd

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
HEADERS = {"User-Agent": "Mozilla/5.0 (WC-Simulator/1.0)"}
WC22 = (43, 106)
CACHE = Path("Data/scraped/_wc22_events_cache"); CACHE.mkdir(parents=True, exist_ok=True)
OUT = Path("Data/data/processed/gb_backtest_2022.json")

# furthest stage → (expected-match credit already captured by matches played) +
# a "spotlight" multiplier voters give the business end of the tournament
STAGE_SPOTLIGHT = {
    "Final": 1.6, "Semi-finals": 1.3, "Quarter-finals": 1.1,
    "Round of 16": 1.0, "Group Stage": 0.9, "3rd Place Final": 1.4,
}
STAGE_ORDER = ["Group Stage", "Round of 16", "Quarter-finals",
               "Semi-finals", "3rd Place Final", "Final"]


def fetch(url, retries=3):
    for a in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"  retry {a}: {e}")
        time.sleep(0.6 * a)
    return None


def main():
    print("=" * 68)
    print("  Golden Ball backtest — World Cup 2022 (StatsBomb)")
    print("=" * 68)
    matches = fetch(f"{RAW}/matches/{WC22[0]}/{WC22[1]}.json")
    if not matches:
        print("Could not fetch WC22 matches — aborting."); return
    print(f"Fetched {len(matches)} matches")

    # furthest stage reached by each team
    team_stage_rank = defaultdict(int)
    team_matches = defaultdict(set)
    def srank(s): return STAGE_ORDER.index(s) if s in STAGE_ORDER else 0
    for m in matches:
        st = m.get("competition_stage", {}).get("name", "Group Stage")
        ht = m["home_team"]["home_team_name"]; at = m["away_team"]["away_team_name"]
        for t in (ht, at):
            team_matches[t].add(m["match_id"])
            team_stage_rank[t] = max(team_stage_rank[t], srank(st))

    team_furthest = {t: STAGE_ORDER[r] for t, r in team_stage_rank.items()}

    # per-player tournament output
    pstat = defaultdict(lambda: {"team": "", "xg": 0.0, "goals": 0,
                                 "assists": 0, "shots": 0, "matches": set()})
    for i, m in enumerate(matches, 1):
        mid = m["match_id"]
        cf = CACHE / f"{mid}.json"
        if cf.exists():
            events = json.loads(cf.read_text(encoding="utf-8"))
        else:
            events = fetch(f"{RAW}/events/{mid}.json")
            if events:
                cf.write_text(json.dumps(events), encoding="utf-8")
            time.sleep(0.4)
        if not events:
            continue
        for e in events:
            tn = e.get("type", {}).get("name")
            pl = e.get("player", {}).get("name")
            tm = e.get("team", {}).get("name")
            if not pl:
                continue
            if tn == "Shot":
                sh = e.get("shot", {})
                pstat[pl]["team"] = tm
                pstat[pl]["xg"] += sh.get("statsbomb_xg", 0.0) or 0.0
                pstat[pl]["goals"] += int(sh.get("outcome", {}).get("name") == "Goal")
                pstat[pl]["shots"] += 1
                pstat[pl]["matches"].add(mid)
            elif tn == "Pass" and e.get("pass", {}).get("goal_assist"):
                pstat[pl]["team"] = pstat[pl]["team"] or tm
                pstat[pl]["assists"] += 1
                pstat[pl]["matches"].add(mid)
        if i % 16 == 0:
            print(f"  processed {i}/{len(matches)} matches")

    # score each player with the Golden Ball logic
    rows = []
    for pl, s in pstat.items():
        n = len(s["matches"])
        if n < 2:
            continue
        team = s["team"]
        ga = s["goals"] + s["assists"]
        # output per match, blending end product with underlying xG (mirrors the
        # club_form blend in the live model, here on tournament data)
        out_rate = 0.55 * (ga / n) + 0.45 * (s["xg"] / n)
        spotlight = STAGE_SPOTLIGHT.get(team_furthest.get(team, "Group Stage"), 0.9)
        team_run = len(team_matches.get(team, []))   # actual matches the TEAM played
        # total tournament impact × spotlight  (team_run rewards a deep run, like
        # expected_matches in the live model)
        score = out_rate * team_run * spotlight
        rows.append({"player": pl, "team": team, "matches": n,
                     "goals": s["goals"], "assists": s["assists"],
                     "xg": round(s["xg"], 2), "ga": ga,
                     "stage": team_furthest.get(team, "?"),
                     "score": round(score, 3)})

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # actual 2022 Golden Ball podium + Golden Boot
    actual = {"Lionel Andrés Messi Cuccittini": "Golden Ball (1st)",
              "Kylian Mbappé Lottin": "Golden Boot / Ball 2nd",
              "Luka Modrić": "Golden Ball 3rd"}

    print("\nPredicted Golden Ball ranking (output × deep run):")
    print(f"  {'#':>2}  {'player':<34}{'team':<13}{'G+A':>4}{'xG':>6}{'stage':>14}{'score':>8}")
    for _, r in df.head(12).iterrows():
        tag = "  ←  " + actual[r["player"]] if r["player"] in actual else ""
        print(f"  {r['rank']:>2}  {r['player'][:33]:<34}{r['team'][:12]:<13}"
              f"{r['ga']:>4}{r['xg']:>6.1f}{r['stage']:>14}{r['score']:>8.2f}{tag}")

    podium = {p: int(df[df.player == p]["rank"].iloc[0]) if (df.player == p).any() else None
              for p in actual}
    print("\nActual Golden Ball podium → our predicted rank:")
    for p, label in actual.items():
        print(f"  {label:<24}{p:<34} predicted #{podium[p]}")

    payload = {
        "tournament": "FIFA World Cup 2022",
        "method": ("Golden Ball scoring logic (attacking output per match × team's "
                   "actual deep run × stage spotlight) replayed on WC22 StatsBomb data."),
        "actual_winner": "Lionel Messi (Argentina)",
        "predicted_top": df.head(10)[["rank", "player", "team", "ga", "xg",
                                      "stage", "score"]].to_dict("records"),
        "podium_recovery": [
            {"actual": label, "player": p, "predicted_rank": podium[p]}
            for p, label in actual.items()],
        "hit_at_1": podium.get("Lionel Andrés Messi Cuccittini") == 1,
        "n_players": int(len(df)),
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
