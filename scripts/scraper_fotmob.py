#!/usr/bin/env python3
"""
FotMob match-detail scraper — xG, Player of the Match, shotmaps
===============================================================
football-data.org (fetch_results.py) gives scores but NO advanced stats.
FotMob has per-match xG + player ratings + Player of the Match, but gates
its API behind a client-generated `x-mas` token and blocks plain HTTP.

This scraper drives a headless Chromium (Playwright) so FotMob's own JS
generates the token and calls the API; we intercept the JSON responses.
That's the robust way past the gate — no token reverse-engineering that
breaks on rotation.

For each FINISHED World Cup 2026 match it records:
  - home/away xG (summed from the shotmap)
  - Player of the Match (FotMob's official pick) + their rating
  - shot counts / on-target

Output → Data/simulated/match_details.json  (keyed by frozenset of canon
team names, so build_dashboard / the Live tab can join it to fixtures).

Usage
-----
    python scripts/scraper_fotmob.py            # all finished matches
    python scripts/scraper_fotmob.py --limit 3  # first 3 (quick test)
    python scripts/scraper_fotmob.py --match 4667751

Requires: playwright (already installed). Run once if needed:
    python -m playwright install chromium

FotMob is unofficial — keep polling gentle (this waits between matches).
Failures on a single match are skipped, not fatal, so a layout change on
one page can't break the whole run.
"""

import sys, io, json, time, argparse
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("playwright not installed — run: pip install playwright && python -m playwright install chromium")

WC_LEAGUE_ID = 77                                   # FotMob World Cup
LEAGUE_URL   = "https://www.fotmob.com/leagues/77/matches/world-cup"
MATCH_URL    = "https://www.fotmob.com{page_url}"   # pageUrl already starts with /matches/...
OUT_PATH     = Path("Data/simulated/match_details.json")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# FotMob → schedule_2026.csv canonical names (mirrors fetch_results.CANON)
CANON = {
    "South Korea": "Korea Republic", "Turkey": "Türkiye", "Türkiye": "Türkiye",
    "Turkiye": "Türkiye",
    "Iran": "IR Iran", "Czech Republic": "Czechia", "Ivory Coast": "Côte d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina", "DR Congo": "Congo DR",
    "Cabo Verde": "Cape Verde", "Cape Verde Islands": "Cape Verde", "USA": "United States",
    "Curacao": "Curaçao",
}
def canon(n: str) -> str:
    return CANON.get(str(n).strip(), str(n).strip())


def _capture_page(pg, url: str, needle: str, timeout_ms: int = 9000):
    """Navigate to url, return the first intercepted JSON whose URL holds `needle`."""
    box = {}
    def on_resp(resp):
        if needle in resp.url and resp.status == 200 and needle not in box:
            try:
                box[needle] = resp.json()
            except Exception:
                pass
    pg.on("response", on_resp)
    pg.goto(url, wait_until="domcontentloaded", timeout=60000)
    # poll until captured or timeout
    waited = 0
    while needle not in box and waited < timeout_ms:
        pg.wait_for_timeout(400); waited += 400
    pg.remove_listener("response", on_resp)
    return box.get(needle)


def get_fixtures(pg) -> list[dict]:
    lg = _capture_page(pg, LEAGUE_URL, "data/leagues?id=77", timeout_ms=12000)
    if not lg:
        sys.exit("Could not load FotMob league feed (data/leagues?id=77).")
    fx = lg.get("fixtures", {})
    return fx.get("allMatches", []) if isinstance(fx, dict) else (fx or [])


def parse_match(md: dict) -> dict | None:
    """Extract xG + Player of the Match from a matchDetails payload."""
    g = md.get("general", {})
    c = md.get("content", {})
    if not g.get("finished"):
        return None
    home = g.get("homeTeam", {}); away = g.get("awayTeam", {})
    h_name, a_name = canon(home.get("name", "")), canon(away.get("name", ""))
    h_id, a_id = home.get("id"), away.get("id")

    # xG: sum expectedGoals from the shotmap, bucketed by teamId
    xg = defaultdict(float); shots = defaultdict(int); ontgt = defaultdict(int)
    for s in (c.get("shotmap", {}) or {}).get("shots", []) or []:
        tid = s.get("teamId")
        xg[tid]    += float(s.get("expectedGoals") or 0)
        shots[tid] += 1
        ontgt[tid] += 1 if s.get("isOnTarget") else 0

    # Player of the Match — FotMob's official pick (matchFacts), with rating.
    # `name` is a dict {firstName,lastName,fullName}; rating under rating.num/value.
    pom = (c.get("matchFacts", {}) or {}).get("playerOfTheMatch") or {}
    nm = pom.get("name")
    if isinstance(nm, dict):
        pom_name = nm.get("fullName") or f"{nm.get('firstName','')} {nm.get('lastName','')}".strip()
    else:
        pom_name = nm or pom.get("fullName")
    pom_name = pom_name or None
    pom_rating = None
    if isinstance(pom.get("rating"), dict):
        pom_rating = pom["rating"].get("num") or pom["rating"].get("value")
    pom_team = canon(pom.get("teamName")) if pom.get("teamName") else None
    if not pom_team:
        tid = pom.get("teamId")
        if tid == h_id: pom_team = h_name
        elif tid == a_id: pom_team = a_name

    # Per-player FotMob ratings — for the "actual" Team of the Tournament once
    # enough matches accumulate. Each player's rating lives at
    #   playerStats[id].stats[].stats["FotMob rating"].stat.value
    players = []
    for pl in (c.get("playerStats", {}) or {}).values():
        rating = None
        for blk in pl.get("stats", []) or []:
            r = (blk.get("stats", {}) or {}).get("FotMob rating")
            if isinstance(r, dict):
                rating = (r.get("stat", {}) or {}).get("value")
                break
        if rating is None:
            continue
        tid = pl.get("teamId")
        players.append({
            "name": pl.get("name"),
            "team": canon(pl.get("teamName") or (h_name if tid == h_id else a_name)),
            "rating": round(float(rating), 2),
            "pos": pl.get("usualPosition") or ("GK" if pl.get("isGoalkeeper") else None),
            "min": pl.get("minutesPlayed"),
        })

    return {
        "home": h_name, "away": a_name,
        "home_xg": round(xg.get(h_id, 0.0), 2),
        "away_xg": round(xg.get(a_id, 0.0), 2),
        "home_shots": shots.get(h_id, 0), "away_shots": shots.get(a_id, 0),
        "home_sot": ontgt.get(h_id, 0),   "away_sot": ontgt.get(a_id, 0),
        "motm": pom_name, "motm_team": pom_team, "motm_rating": pom_rating,
        "player_ratings": players,
    }


def main():
    ap = argparse.ArgumentParser(description="Scrape FotMob WC2026 xG / MOTM")
    ap.add_argument("--limit", type=int, default=0, help="only first N finished matches")
    ap.add_argument("--match", type=str, default="", help="single match id (debug)")
    ap.add_argument("--headed", action="store_true", help="show the browser")
    args = ap.parse_args()

    results: dict[str, dict] = {}   # key "TeamA|TeamB" (sorted) → detail
    existing = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8")).get("matches", {})
        results.update(existing)

    with sync_playwright() as p:
        b = p.chromium.launch(headless=not args.headed)
        pg = b.new_page(user_agent=UA)

        fixtures = get_fixtures(pg)
        finished = [m for m in fixtures if (m.get("status", {}) or {}).get("finished")]
        if args.match:
            finished = [m for m in fixtures if str(m.get("id")) == args.match]
        if args.limit:
            finished = finished[: args.limit]
        print(f"FotMob: {len(fixtures)} fixtures, {len(finished)} to scrape")

        scraped = 0
        for i, m in enumerate(finished, 1):
            page_url = m.get("pageUrl")
            mid = m.get("id")
            if not page_url:
                continue
            url = MATCH_URL.format(page_url=page_url)
            try:
                md = _capture_page(pg, url, "matchDetails", timeout_ms=10000)
                if not md or "content" not in md:
                    print(f"  [{i}/{len(finished)}] {mid}: no detail (skipped)")
                    continue
                det = parse_match(md)
                if not det:
                    continue
                key = "|".join(sorted((det["home"], det["away"])))
                results[key] = det
                scraped += 1
                xgline = f"xG {det['home_xg']}-{det['away_xg']}"
                motm = f"MOTM {det['motm']}" if det["motm"] else "no MOTM"
                print(f"  [{i}/{len(finished)}] {det['home']} v {det['away']}  {xgline}  {motm}")
            except Exception as e:
                print(f"  [{i}/{len(finished)}] {mid}: ERROR {repr(e)[:90]}")
            time.sleep(1.2)   # gentle polling

        b.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "matches": results,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ {scraped} match(es) scraped, {len(results)} total → {OUT_PATH}")


if __name__ == "__main__":
    main()
