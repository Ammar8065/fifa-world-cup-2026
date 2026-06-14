#!/usr/bin/env python3
"""Scrape 2026 squads from The Guardian player guide into JSON + CSV."""
import sys, io, time, re, json, html
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import requests
from pathlib import Path
from datetime import date
import pandas as pd

MASTER = "https://interactive.guim.co.uk/docsdata/1_ZAfmUkTZ4BvDgvhEGaEruakfu4aWIIjjzXaMAiT1yc.json"
DOCS   = "https://interactive.guim.co.uk/docsdata/{sid}.json"
HEAD   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0 Safari/537.36"}
DELAY  = 0.35
TOURNAMENT_START = date(2026, 6, 11)
OUT_DIR = Path("Data/scraped")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical team names (Guardian spelling → project spelling)
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
def canon(n): return CANON.get(str(n).strip(), str(n).strip())

# Player-sheet column keys (verbatim from the source spreadsheet)
COL_SPECIAL = "special player? (eg. key player, promising talent, etc) OPTIONAL"
COL_GOALS   = "goals for country"
COL_DOB     = "date of birth"
COL_PHOTO_DONE = "photo done?"

def fetch_json(url, retries=3):
    for a in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEAD, timeout=40)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} (try {a}): {url}")
        except Exception as e:
            print(f"  err (try {a}): {e}")
        time.sleep(DELAY * a * 2)
    return None

def strip_html(s):
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", str(s))
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()

def to_int(v):
    try:
        return int(float(str(v).strip()))
    except (ValueError, TypeError):
        return None

def parse_age(dob):
    """dd/mm/yyyy → age at tournament start."""
    try:
        d, m, y = [int(x) for x in str(dob).strip().split("/")]
        born = date(y, m, d)
        a = TOURNAMENT_START.year - born.year
        if (TOURNAMENT_START.month, TOURNAMENT_START.day) < (born.month, born.day):
            a -= 1
        return a, born.isoformat()
    except Exception:
        return None, None

def pos_group(p):
    p = (p or "").lower()
    if "keep" in p or p.strip() == "gk":       return "GK"
    if "back" in p or "defen" in p:            return "DEF"
    if "mid" in p:                             return "MID"
    return "FWD"

def main():
    print("Fetching master Teams sheet...")
    master = fetch_json(MASTER)
    teams_rows = master["sheets"]["Teams"]
    print(f"  {len(teams_rows)} teams\n")

    squads, flat_rows = {}, []
    for i, tr in enumerate(teams_rows, 1):
        team = canon(tr["Team"])
        sid  = tr["spreadsheet"]
        star = strip_html(tr.get("player_pick", "")).strip()
        meta = {
            "team":        team,
            "guardian_name": tr["Team"],
            "fifa_ranking": to_int(tr.get("FIFA_ranking")),
            "group":       (tr.get("Group") or "").strip(),
            "coach":       strip_html(tr.get("Coach")),
            "bio":         strip_html(tr.get("Bio")),
            "strengths":   strip_html(tr.get("strengths")),
            "weaknesses":  strip_html(tr.get("weaknesses")),
            "star_player": star,
            "byline":      strip_html(tr.get("Byline")),
        }
        print(f"[{i:2d}/48] {team:<20} (rank {meta['fifa_ranking']}, group {meta['group']})  ← {sid[:10]}…")
        td = fetch_json(DOCS.format(sid=sid))
        time.sleep(DELAY)
        players = []
        if td and "Players" in td.get("sheets", {}):
            for p in td["sheets"]["Players"]:
                name = (p.get("name") or "").strip()
                if not name:
                    continue
                age, born_iso = parse_age(p.get(COL_DOB))
                special = (p.get(COL_SPECIAL) or "").strip()
                photo = (p.get("grid_image") or p.get("image_reference") or "").strip()
                rec = {
                    "team":       team,
                    "name":       name,
                    "position":   (p.get("position") or "").strip(),
                    "pos_group":  pos_group(p.get("position")),
                    "number":     to_int(p.get("number")),
                    "caps":       to_int(p.get("caps")),
                    "goals":      to_int(p.get(COL_GOALS)),
                    "club":       (p.get("club") or "").strip(),
                    "dob":        born_iso,
                    "age":        age,
                    "special":    special,
                    "is_wonderkid": special.lower() == "wonderkid",
                    "is_star":    bool(star) and (star.lower() in name.lower() or name.lower() in star.lower()),
                    "photo":      photo,
                    "bio":        strip_html(p.get("bio")),
                }
                players.append(rec)
                flat_rows.append(rec)
        else:
            print(f"        ⚠ no Players sheet for {team}")
        # keep squad order; mark captain heuristically (bio mentions 'captain')
        squads[team] = {"meta": meta, "players": players}

    # Save nested JSON
    json_path = OUT_DIR / "squads_2026.json"
    json_path.write_text(json.dumps(squads, ensure_ascii=False, indent=1), encoding="utf-8")
    # Save flat CSV
    csv_path = OUT_DIR / "squad_players.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8")

    n_players = sum(len(s["players"]) for s in squads.values())
    n_photos  = sum(1 for r in flat_rows if r["photo"])
    n_wonder  = sum(1 for r in flat_rows if r["is_wonderkid"])
    print(f"\n✓ {len(squads)} teams · {n_players} players · {n_photos} with photos · {n_wonder} wonderkids")
    print(f"  → {json_path}")
    print(f"  → {csv_path}")

if __name__ == "__main__":
    main()
