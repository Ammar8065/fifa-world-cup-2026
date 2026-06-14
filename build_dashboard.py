#!/usr/bin/env python3
"""
Generates dashboard.html — a fully self-contained premium web app for the 2026 WC simulation.
Open dashboard.html directly in any browser (no server required).
"""
import sys, io, json, re, unicodedata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
from pathlib import Path

# ── Load all data ──────────────────────────────────────────────────────────────
summary   = pd.read_csv("Data/simulated/wc2026_champion_probabilities.csv")
bracket   = pd.read_csv("Data/simulated/wc2026_bracket_predictions.csv")
bracket_full = pd.read_csv("Data/simulated/wc2026_bracket_full.csv")
slots_df  = pd.read_csv("Data/data/raw/knockout_slots.csv")
players   = pd.read_csv("Data/scraped/player_xg.csv")
groups_df = pd.read_csv("Data/data/raw/groups.csv")

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
groups_df["team"] = groups_df["team"].apply(canon)
players["team"]   = players["team"].apply(canon)

# ── Flags ──────────────────────────────────────────────────────────────────────
FLAGS = {
    "Mexico":"🇲🇽","South Africa":"🇿🇦","Korea Republic":"🇰🇷","Czechia":"🇨🇿",
    "Canada":"🇨🇦","Bosnia-Herzegovina":"🇧🇦","Qatar":"🇶🇦","Switzerland":"🇨🇭",
    "Brazil":"🇧🇷","Morocco":"🇲🇦","Haiti":"🇭🇹","Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States":"🇺🇸","Paraguay":"🇵🇾","Australia":"🇦🇺","Türkiye":"🇹🇷",
    "Germany":"🇩🇪","Curaçao":"🇨🇼","Côte d'Ivoire":"🇨🇮","Ecuador":"🇪🇨",
    "Netherlands":"🇳🇱","Japan":"🇯🇵","Sweden":"🇸🇪","Tunisia":"🇹🇳",
    "Belgium":"🇧🇪","Egypt":"🇪🇬","IR Iran":"🇮🇷","New Zealand":"🇳🇿",
    "Spain":"🇪🇸","Cape Verde":"🇨🇻","Saudi Arabia":"🇸🇦","Uruguay":"🇺🇾",
    "France":"🇫🇷","Senegal":"🇸🇳","Iraq":"🇮🇶","Norway":"🇳🇴",
    "Argentina":"🇦🇷","Algeria":"🇩🇿","Austria":"🇦🇹","Jordan":"🇯🇴",
    "Portugal":"🇵🇹","Congo DR":"🇨🇩","Uzbekistan":"🇺🇿","Colombia":"🇨🇴",
    "England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","Croatia":"🇭🇷","Ghana":"🇬🇭","Panama":"🇵🇦",
}

# ISO 3166-1 alpha-2 codes for flagcdn.com (home nations use gb-eng / gb-sct).
FLAG_ISO = {
    "Mexico":"mx","South Africa":"za","Korea Republic":"kr","Czechia":"cz",
    "Canada":"ca","Bosnia-Herzegovina":"ba","Qatar":"qa","Switzerland":"ch",
    "Brazil":"br","Morocco":"ma","Haiti":"ht","Scotland":"gb-sct",
    "United States":"us","Paraguay":"py","Australia":"au","Türkiye":"tr",
    "Germany":"de","Curaçao":"cw","Côte d'Ivoire":"ci","Ecuador":"ec",
    "Netherlands":"nl","Japan":"jp","Sweden":"se","Tunisia":"tn",
    "Belgium":"be","Egypt":"eg","IR Iran":"ir","New Zealand":"nz",
    "Spain":"es","Cape Verde":"cv","Saudi Arabia":"sa","Uruguay":"uy",
    "France":"fr","Senegal":"sn","Iraq":"iq","Norway":"no",
    "Argentina":"ar","Algeria":"dz","Austria":"at","Jordan":"jo",
    "Portugal":"pt","Congo DR":"cd","Uzbekistan":"uz","Colombia":"co",
    "England":"gb-eng","Croatia":"hr","Ghana":"gh","Panama":"pa",
}

# ── Young players curated list (born after Jun 2003 = under 23 at WC) ─────────
YOUNG_PLAYERS = [
    {"player":"Lamine Yamal",       "team":"Spain",       "birth_year":2007,"pos":"RW"},
    {"player":"Warren Zaïre-Emery", "team":"France",      "birth_year":2006,"pos":"CM"},
    {"player":"Endrick",            "team":"Brazil",      "birth_year":2006,"pos":"ST"},
    {"player":"Kobbie Mainoo",      "team":"England",     "birth_year":2005,"pos":"CM"},
    {"player":"Arda Güler",         "team":"Türkiye",     "birth_year":2005,"pos":"AM"},
    {"player":"Mathys Tel",         "team":"France",      "birth_year":2005,"pos":"FW"},
    {"player":"Savinho",            "team":"Brazil",      "birth_year":2004,"pos":"RW"},
    {"player":"Gavi",               "team":"Spain",       "birth_year":2004,"pos":"CM"},
    {"player":"Garnacho",           "team":"Argentina",   "birth_year":2004,"pos":"LW"},
    {"player":"Brajan Gruda",       "team":"Germany",     "birth_year":2004,"pos":"FW"},
    {"player":"Jude Bellingham",    "team":"England",     "birth_year":2003,"pos":"AM"},
    {"player":"Florian Wirtz",      "team":"Germany",     "birth_year":2003,"pos":"AM"},
    {"player":"Jamal Musiala",      "team":"Germany",     "birth_year":2003,"pos":"AM"},
    {"player":"Rayan Cherki",       "team":"France",      "birth_year":2003,"pos":"AM"},
    {"player":"Xavi Simons",        "team":"Netherlands", "birth_year":2003,"pos":"AM"},
    {"player":"Pedri",              "team":"Spain",       "birth_year":2002,"pos":"CM"},
    {"player":"Eduardo Camavinga",  "team":"France",      "birth_year":2002,"pos":"CM"},
    {"player":"Cole Palmer",        "team":"England",     "birth_year":2002,"pos":"AM"},
    {"player":"Takefusa Kubo",      "team":"Japan",       "birth_year":2001,"pos":"RW"},
]

# ── Real 2026 squads (scraped from The Guardian player guide) ──────────────────
squads_raw = json.loads(Path("Data/scraped/squads_2026.json").read_text(encoding="utf-8"))
team_win = dict(zip(summary["team"], summary["p_champion"]))
wc_teams = set(summary["team"].tolist())

# Per-team stage probabilities — drives Golden Ball exposure (how many matches a
# player's team is projected to play) and the deep-run "spotlight" voters reward.
_STAGE_COLS = ["p_qualify", "p_round_of_16", "p_quarter_final",
               "p_semi_final", "p_final", "p_champion"]
team_stage = {r["team"]: {c: float(r.get(c, 0.0) or 0.0) for c in _STAGE_COLS}
              for _, r in summary.iterrows()}

def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()

BIO_KEEP = 170
squads = {}
squad_match = {}      # (team, name-token) -> player record
for team, blk in squads_raw.items():
    meta, plist = blk["meta"], blk["players"]
    star_norm = _norm(meta.get("star_player", ""))
    star_idx = None
    for idx, p in enumerate(plist):
        fn = _norm(p["name"])
        if star_norm and (star_norm == fn or star_norm in fn or fn in star_norm):
            star_idx = idx; break
    if star_idx is None and star_norm:
        for idx, p in enumerate(plist):
            toks = _norm(p["name"]).split()
            if toks and toks[-1] in star_norm:
                star_idx = idx; break
    aged = [(idx, p) for idx, p in enumerate(plist) if p.get("age")]
    wonder_idx = min(aged, key=lambda ip: ip[1]["age"])[0] if aged else None
    if wonder_idx is not None and (plist[wonder_idx].get("age") or 99) > 21:
        wonder_idx = None
    out = []
    for idx, p in enumerate(plist):
        fn = _norm(p["name"]); toks = fn.split(); sur = toks[-1] if toks else ""
        is_star   = idx == star_idx
        is_wonder = (idx == wonder_idx) or (p.get("special", "").lower() == "wonderkid")
        highlight = is_star or is_wonder or bool(p.get("special", "").strip())
        bio = p.get("bio", "") or ""
        if not highlight and len(bio) > BIO_KEEP:
            bio = bio[:BIO_KEEP].rsplit(" ", 1)[0] + "…"
        rec = {**p, "is_star": is_star, "is_wonderkid": is_wonder, "bio": bio}
        out.append(rec)
        for t in set([sur] + toks):
            if len(t) >= 4:
                squad_match.setdefault((team, t), rec)
        squad_match[(team, fn)] = rec
    squads[team] = {
        "meta": meta, "players": out,
        "star_name":      out[star_idx]["name"]   if star_idx   is not None else None,
        "wonderkid_name": out[wonder_idx]["name"] if wonder_idx is not None else None,
    }

def squad_lookup(team, sb_name):
    fn = _norm(sb_name)
    if (team, fn) in squad_match:
        return squad_match[(team, fn)]
    for t in fn.split():
        if len(t) >= 4 and (team, t) in squad_match:
            return squad_match[(team, t)]
    return None

# ── Player awards — built from REAL 2026 squads, ranked by CURRENT-SEASON xG ────
# Primary signal: 2025-26 club xG (Understat via soccerdata) → reflects live form.
# Fallback: historical tournament xG (StatsBomb) for squad players outside Europe's
# big-5 leagues (Understat's coverage). Players with neither are not award candidates.
def _firstlast(n):
    t = n.split()
    return (t[0] + " " + t[-1]) if len(t) >= 2 else n

# Current-season club xG, keyed by normalized player name (best xG row wins).
cur_path = Path("Data/scraped/player_xg_current.csv")
USE_CURRENT = cur_path.exists()
cur_lookup = {}
if USE_CURRENT:
    cur_df = pd.read_csv(cur_path).sort_values("xg", ascending=False)
    cur_df["_n"] = cur_df["player"].map(_norm)
    for _, r in cur_df.iterrows():
        for key in (r["_n"], _firstlast(r["_n"])):
            cur_lookup.setdefault(key, r)

# Historical tournament xG fallback (StatsBomb), keyed by normalized name.
players["_n"] = players["player"].map(_norm)
tour_lookup = {}
for _, r in players.sort_values("xg", ascending=False).iterrows():
    for key in (r["_n"], _firstlast(r["_n"])):
        tour_lookup.setdefault(key, r)

award_rows = []
for team, blk in squads.items():
    if team not in wc_teams:
        continue
    twp = float(team_win.get(team, 0.0))
    for p in blk["players"]:
        n   = _norm(p["name"])
        cur = cur_lookup.get(n,  cur_lookup.get(_firstlast(n)))
        tor = tour_lookup.get(n, tour_lookup.get(_firstlast(n)))
        if cur is not None:
            xg, goals, matches = float(cur["xg"]), int(cur["goals"]), int(cur["matches"])
            shots, xgpm, src   = int(cur["shots"]), float(cur["xg_per_match"]), "current"
            club_disp          = cur["club"]
        elif tor is not None:
            xg, goals, matches = float(tor["xg"]), int(tor["goals"]), int(tor["matches"])
            shots, xgpm, src   = int(tor["shots"]), float(tor["xg_per_match"]), "tournament"
            club_disp          = p.get("club", "")
        else:
            continue
        award_rows.append({
            "player": p["name"], "team": team,
            "matches": matches, "shots": shots, "xg": round(xg, 3), "goals": goals,
            "xg_per_match": round(xgpm, 4), "composite": round(xgpm * twp * 100, 4),
            "flag": FLAGS.get(team, "🏳️"), "team_win_prob": twp,
            "photo": p.get("photo", ""), "position": p.get("position", ""),
            "club": club_disp, "age": p.get("age"), "xg_source": src,
        })

players_real = pd.DataFrame(award_rows)
n_current = int((players_real["xg_source"] == "current").sum())
n_matched = len(players_real)

# Player-of-the-Tournament impact = season xG weighted by a GENTLE deep-run factor.
# The old metric (xG/match × P(champion)) let one team's title odds dominate — with
# Spain ~27% it buried a 0.95-xG/match Harry Kane under Spanish role-players. We use
# (0.6 + P(reach semi-final)) instead: a soft multiplier (~0.6–1.1×) that still rewards
# deep runs but lets individual xG output drive the ranking. The list now SORTS BY and
# DISPLAYS this same impact score, so the visible numbers are always in order.
players_real["deep_run"]   = players_real["team"].map(
    lambda t: team_stage.get(t, {}).get("p_semi_final", 0.0))
players_real["poty_score"] = (players_real["xg"] * (0.6 + players_real["deep_run"])).round(2)

ACOLS = ["player","team","matches","shots","xg","goals","xg_per_match","composite",
         "poty_score","deep_run","flag","team_win_prob","photo","position","club",
         "age","xg_source"]

# Top Scorer ranks by total xG (raw volume), so a low match floor is fine.
top_scorers = players_real[players_real["matches"] >= 3].nlargest(20, "xg")[ACOLS].reset_index(drop=True)
# POTY / Young Player require a full sample (≥10 matches) — excludes noisy small-sample
# tournament fallbacks (a right-back with 0.8 xG/match over 3 games).
poty_candidates = players_real[players_real["matches"] >= 10].nlargest(10, "poty_score")[ACOLS].reset_index(drop=True)

# Young Player race: real U-23 squad players, same impact metric.
ypool = players_real[(players_real["age"].notna()) &
                     (players_real["age"] <= 23) & (players_real["matches"] >= 10)].copy()
ypool["score"] = ypool["poty_score"]
ypoty_df = ypool.nlargest(10, "score")[ACOLS + ["score"]].reset_index(drop=True)

# Rank defenses by expected goals conceded per game vs an average opponent
# (lower = better, real goal units) — far clearer than the abstract 0–1 score and
# free of the "clean sheets vs minnows" artifact since it runs through the model.
summary["goals_allowed_rank"] = summary["xga_per_game"].rank(ascending=True)
best_def_teams = summary.nsmallest(5, "xga_per_game")[["team","xga_per_game","def_score","elo","p_champion"]].reset_index(drop=True)
best_def_teams["flag"] = best_def_teams["team"].map(FLAGS).fillna("🏳️")

best_att_teams = summary.nlargest(5, "xgf_per_game")[["team","xgf_per_game","att_score","elo","p_champion"]].reset_index(drop=True)
best_att_teams["flag"] = best_att_teams["team"].map(FLAGS).fillna("🏳️")
print(f"  award pool: {n_matched} squad players with xG "
      f"({n_current} on current-season club xG, {n_matched - n_current} on tournament fallback)")

# ── Golden Ball: predicted best player of the tournament ─────────────────────
# Not a raw G+A sum — a forecast. Computed after players_db is built (needs the
# per-90 form fields). Populated by the model block below the players_db loop.
golden_ball_candidates = []  # populated after players_db loop

# ── Confederation map (for Overview confederation title-odds viz) ──────────────
CONFED = {
    # UEFA
    "Czechia":"UEFA","Bosnia-Herzegovina":"UEFA","Switzerland":"UEFA","Scotland":"UEFA",
    "Türkiye":"UEFA","Germany":"UEFA","Netherlands":"UEFA","Sweden":"UEFA","Belgium":"UEFA",
    "Spain":"UEFA","France":"UEFA","Norway":"UEFA","Austria":"UEFA","Portugal":"UEFA",
    "England":"UEFA","Croatia":"UEFA",
    # CONMEBOL
    "Brazil":"CONMEBOL","Paraguay":"CONMEBOL","Ecuador":"CONMEBOL","Uruguay":"CONMEBOL",
    "Argentina":"CONMEBOL","Colombia":"CONMEBOL",
    # CONCACAF
    "Mexico":"CONCACAF","Canada":"CONCACAF","Haiti":"CONCACAF","United States":"CONCACAF",
    "Curaçao":"CONCACAF","Panama":"CONCACAF",
    # CAF
    "South Africa":"CAF","Morocco":"CAF","Côte d'Ivoire":"CAF","Tunisia":"CAF","Egypt":"CAF",
    "Cape Verde":"CAF","Senegal":"CAF","Algeria":"CAF","Congo DR":"CAF","Ghana":"CAF",
    # AFC
    "Korea Republic":"AFC","Qatar":"AFC","Australia":"AFC","Japan":"AFC","IR Iran":"AFC",
    "Saudi Arabia":"AFC","Iraq":"AFC","Jordan":"AFC","Uzbekistan":"AFC",
    # OFC
    "New Zealand":"OFC",
}
summary["confederation"] = summary["team"].map(CONFED).fillna("—")

# ── Player explorer dataset — every squad player + current-season club stats ───
#    (G/A/xG/xA from Understat) and international tournament stats (StatsBomb),
#    plus a club-form rating and the full Guardian bio for the popup.
def _f(v, d=0.0):
    try:
        x = float(v)
        return d if (x != x) else x
    except (TypeError, ValueError):
        return d

players_db = []
team_firepower = {t: 0.0 for t in wc_teams}   # Σ current-season club xG per national team
for team, blk in squads.items():
    if team not in wc_teams:
        continue
    flag = FLAGS.get(team, "🏳️")
    for p in blk["players"]:
        n   = _norm(p["name"])
        cur = cur_lookup.get(n,  cur_lookup.get(_firstlast(n)))
        tor = tour_lookup.get(n, tour_lookup.get(_firstlast(n)))
        rec = {
            "name": p["name"], "team": team, "flag": flag,
            "club": p.get("club", ""), "position": p.get("position", ""),
            "pos_group": p.get("pos_group", ""), "age": p.get("age"),
            "number": p.get("number"), "photo": p.get("photo", ""),
            "caps": p.get("caps"), "intl_career_goals": p.get("goals"),
            "bio": p.get("bio", ""),
        }
        if cur is not None:
            mins   = _f(cur.get("minutes"))
            nineties = mins / 90.0 if mins else 0.0
            xg, xa = _f(cur.get("xg")), _f(cur.get("xa"))
            rec.update({
                "has_club": True, "league": cur.get("competition", ""),
                "club_team": cur.get("club", p.get("club", "")),
                "g": int(_f(cur.get("goals"))), "a": int(_f(cur.get("assists"))),
                "xg": round(xg, 2), "xa": round(xa, 2),
                "npxg": round(_f(cur.get("np_xg")), 2),
                "shots": int(_f(cur.get("shots"))), "matches": int(_f(cur.get("matches"))),
                "minutes": int(mins),
                "xg90": round(xg / nineties, 2) if nineties else 0.0,
                "xa90": round(xa / nineties, 2) if nineties else 0.0,
            })
            rec["form"] = round(min(10.0, (rec["xg90"] + rec["xa90"]) * 10), 1)
            team_firepower[team] += xg
        else:
            rec.update({"has_club": False, "league": "", "club_team": p.get("club", ""),
                        "g": None, "a": None, "xg": None, "xa": None, "npxg": None,
                        "shots": None, "matches": None, "minutes": None,
                        "xg90": None, "xa90": None, "form": None})
        if tor is not None:
            rec.update({
                "has_intl": True, "intl_comp": tor.get("competitions", ""),
                "intl_xg": round(_f(tor.get("xg")), 2), "intl_goals": int(_f(tor.get("goals"))),
                "intl_matches": int(_f(tor.get("matches"))),
                "intl_xgpm": round(_f(tor.get("xg_per_match")), 2),
            })
        else:
            rec.update({"has_intl": False, "intl_comp": "", "intl_xg": None,
                        "intl_goals": None, "intl_matches": None, "intl_xgpm": None})
        players_db.append(rec)

summary["firepower"] = summary["team"].map(team_firepower).fillna(0.0).round(1)
print(f"  player explorer: {len(players_db)} players "
      f"({sum(1 for r in players_db if r['has_club'])} with current club stats)")

# ── Golden Ball model ─────────────────────────────────────────────────────────
# The Golden Ball goes to the tournament's best player — in the modern era almost
# always an attacker/attacking-mid whose team runs deep. We forecast it by blending
# three form signals and weighting by how far the player's team is projected to go:
#   1. Club form   — 2025-26 goal involvements per 90 (xG+xA blended with actual G+A)
#   2. Recent intl — goal output per match in recent national-team tournaments
#   3. Stage run   — expected tournament matches × a deep-run "spotlight" multiplier
# Limitation: a goals/assists model can't rate deep-lying mids/keepers (Modrić '18,
# Kahn '02); it captures the attacking output that has decided most modern winners.
GB_INTL_W      = 0.6     # weight on recent-tournament form (smaller, noisier sample)
GB_SPOT_W      = 1.0     # extra reward for reaching the final four / final (voter bias)
GB_SHARPEN     = 2.2     # concentrates the win-probability share onto real favourites
GB_USAGE_HALF  = 900.0   # minutes at which the "nailed-on starter" weight hits 0.5
GB_MIN_MATCHES = 10      # club-match floor — kills small-sample per-90 flukes
GB_MIN_MINUTES = 600

def _intl_rate(rec):
    """Recent national-team tournament output per match (0 when no data)."""
    m = _f(rec.get("intl_matches"))
    if m <= 0:
        return 0.0
    return 0.5 * _f(rec.get("intl_xgpm")) + 0.5 * (_f(rec.get("intl_goals")) / m)

for rec in players_db:
    if not rec.get("has_club"):
        continue
    matches, minutes = _f(rec.get("matches")), _f(rec.get("minutes"))
    if matches < GB_MIN_MATCHES or minutes < GB_MIN_MINUTES:
        continue
    nineties = minutes / 90.0
    g, a = _f(rec.get("g")), _f(rec.get("a"))
    club_form   = 0.5 * (_f(rec.get("xg90")) + _f(rec.get("xa90"))) \
                + 0.5 * ((g + a) / nineties)          # goal involvements / 90
    # Usage weight: a saturating function of club minutes. A nailed-on starter
    # (~3000 min) reads ~0.77; a high-rate rotation player (~700 min) ~0.44 — so a
    # fluky per-90 over few minutes can't out-rank an undroppable star.
    usage       = minutes / (minutes + GB_USAGE_HALF)
    intl_form   = _intl_rate(rec)                     # recent WC/continental form
    form_rating = club_form * usage + GB_INTL_W * intl_form

    st = team_stage.get(rec["team"], {})
    # Expected matches: 3 group games + one per knockout round the team reaches.
    exp_matches = 3.0 + st.get("p_qualify", 0) + st.get("p_round_of_16", 0) \
                + st.get("p_quarter_final", 0) + st.get("p_semi_final", 0) \
                + st.get("p_final", 0)
    spotlight   = 1.0 + GB_SPOT_W * (st.get("p_semi_final", 0) + st.get("p_final", 0))
    gb_index    = form_rating * exp_matches * spotlight

    golden_ball_candidates.append({
        "player": rec["name"], "team": rec["team"],
        "photo": rec.get("photo", ""), "position": rec.get("position", ""),
        "club": rec.get("club", ""), "age": rec.get("age"),
        "g": int(g), "a": int(a), "ga": int(g + a),
        "xg": rec.get("xg"), "xa": rec.get("xa"), "matches": int(matches),
        "club_form": round(club_form, 3), "intl_form": round(intl_form, 3),
        "form_rating": round(form_rating, 3),
        "exp_matches": round(exp_matches, 2), "spotlight": round(spotlight, 3),
        "deep_run": round(st.get("p_semi_final", 0), 4),
        "title_odds": round(st.get("p_champion", 0), 4),
        "gb_index": round(gb_index, 4),
        "flag": FLAGS.get(rec["team"], "🏳️"),
    })

golden_ball_candidates.sort(key=lambda x: x["gb_index"], reverse=True)
# Convert the index to a win-probability share, sharpened so the favourite reads
# like a favourite rather than being diluted across the whole candidate pool.
_gb_tot = sum(c["gb_index"] ** GB_SHARPEN for c in golden_ball_candidates) or 1.0
for c in golden_ball_candidates:
    c["win_prob"] = round(c["gb_index"] ** GB_SHARPEN / _gb_tot, 4)
if golden_ball_candidates:
    _gb = golden_ball_candidates[0]
    print(f"  Golden Ball pick: {_gb['player']} ({_gb['team']}) — "
          f"index {_gb['gb_index']}, p={_gb['win_prob']:.1%} "
          f"[club_form {_gb['club_form']}, intl {_gb['intl_form']}, "
          f"exp_matches {_gb['exp_matches']}]")

# ── Group data ────────────────────────────────────────────────────────────────
group_data = {}
for g in sorted(summary["group"].unique()):
    teams_in_group = summary[summary["group"] == g].sort_values("p_qualify", ascending=False)
    group_data[g] = teams_in_group[["team","p_qualify","p_champion","att_score","def_score","elo"]].to_dict("records")
    for t in group_data[g]:
        t["flag"] = FLAGS.get(t["team"], "🏳️")

# ── Bracket data ──────────────────────────────────────────────────────────────
bracket_dict = {int(r["match_id"]): {"winner": r["pred_winner"], "p": r["p_win"],
                                      "round": r["round"],
                                      "flag": FLAGS.get(r["pred_winner"], "🏳️")}
                for _, r in bracket.iterrows()}

slot_map = {int(r["match_id"]): {"home": r["slot_home"], "away": r["slot_away"],
                                   "round": r["round"]}
            for _, r in slots_df.iterrows() if r["round"] != "Third-place playoff"}

# Full deterministic bracket — both teams in every match
def _na(v):
    return None if (v is None or (isinstance(v, float) and v != v)) else v
bracket_full_dict = {}
for _, r in bracket_full.iterrows():
    bracket_full_dict[int(r["match_id"])] = {
        "round":      r["round"],
        "date_utc":   _na(r["date_utc"]),
        "venue":      _na(r["venue"]),
        "home":       _na(r["home_team"]),
        "away":       _na(r["away_team"]),
        "winner":     _na(r["pred_winner"]),
        "p_home_win": _na(r["p_home_win"]),
        "p_winner":   _na(r["p_winner"]),
    }

# ── JSON payloads ─────────────────────────────────────────────────────────────
def _clean(obj):
    if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj

def to_js(obj):
    return json.dumps(_clean(obj), ensure_ascii=False, default=str)

teams_js   = to_js(summary.to_dict("records"))
groups_js  = to_js(group_data)
bracket_js = to_js(bracket_dict)
bracketfull_js = to_js(bracket_full_dict)
slots_js   = to_js(slot_map)
scorers_js = to_js(top_scorers.to_dict("records"))
poty_js    = to_js(poty_candidates.to_dict("records"))
ypoty_js   = to_js(ypoty_df.to_dict("records"))
bestdef_js = to_js(best_def_teams.to_dict("records"))
bestatt_js = to_js(best_att_teams.to_dict("records"))
goldenball_js = to_js(golden_ball_candidates[:12])
flags_js   = to_js(FLAGS)
flagiso_js = to_js(FLAG_ISO)
young_js   = to_js(YOUNG_PLAYERS)
squads_js  = to_js(squads)
playersdb_js = to_js(players_db)

print("Data prepared. Building HTML...")

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FIFA World Cup 2026 · Prediction Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
/* ── DESIGN TOKENS ── */
:root {{
  --bg:        #EDF0F6;
  --bg2:       #F4F6FA;
  --bg3:       #FFFFFF;
  --card:      #FFFFFF;
  --card2:     #F3F5F9;
  --card3:     #E9EDF4;
  --border:    rgba(15,23,42,0.09);
  --border2:   rgba(15,23,42,0.15);
  --gold:      #C2740B;
  --gold2:     #E8920C;
  --gold3:     #F4B43A;
  --gold-glow: rgba(217,119,6,0.22);
  --silver:    #7E8CA0;
  --bronze:    #B97333;
  --blue:      #2563EB;
  --blue2:     #3B82F6;
  --green:     #059669;
  --green2:    #10B981;
  --red:       #DC2626;
  --purple:    #7C3AED;
  --teal:      #0D9488;
  --txt:       #14213A;
  --txt2:      #4A5870;
  --txt3:      #6B7890;
  --txt4:      #94A3B8;
  --radius:    14px;
  --radius2:   9px;
  --radius3:   6px;
  --shadow:    0 12px 44px rgba(20,33,58,.10);
  --shadow2:   0 2px 14px rgba(20,33,58,.06);
  --glow-gold: 0 0 0 1px rgba(217,119,6,.14), 0 8px 28px rgba(217,119,6,.12);
  --ease:      cubic-bezier(.22,1,.36,1);
  --lift:      0 14px 38px rgba(20,33,58,.14), 0 0 0 1px rgba(217,119,6,.10);
  /* editorial / broadcast accents */
  --ink:       #0A1730;   /* deep navy — hero band + display headings */
  --ink2:      #112444;
  --pitch:     #10B981;   /* pitch-green secondary accent (pairs with gold) */
  --pitch2:    #34D399;
}}

/* ── RESET ── */
*, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{
  background:linear-gradient(180deg, #EFF2F7 0%, #E7EBF2 100%);
  background-attachment:fixed;
  color:var(--txt);
  font-family:'Inter', sans-serif;
  min-height:100vh;
  overflow-x:hidden;
  -webkit-font-smoothing:antialiased;
}}

/* Subtle noise texture overlay */
body::before {{
  content:'';
  position:fixed;
  inset:0;
  pointer-events:none;
  z-index:0;
  opacity:.015;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
  background-size:128px 128px;
}}

/* ── FLAG IMAGES ── */
.flag-img {{
  border-radius:3px;
  border:1px solid rgba(20,33,58,.16);
  box-shadow:0 1px 4px rgba(20,33,58,.10);
  object-fit:cover;
  display:inline-block;
  vertical-align:middle;
  flex-shrink:0;
}}

/* ── HERO ── */
.hero {{
  position:relative;
  overflow:hidden;
  padding:62px 24px 54px;
  text-align:center;
  border-bottom:none;
  color:#fff;
  background:
    radial-gradient(ellipse 95% 130% at 50% -25%, #18294a 0%, transparent 68%),
    linear-gradient(168deg, #0a1730 0%, #0e1d3a 46%, #08132a 100%);
}}
/* Gold→pitch-green broadcast rule along the bottom edge */
.hero::after {{
  content:''; position:absolute; left:0; right:0; bottom:0; height:3px;
  background:linear-gradient(90deg, var(--gold) 0%, var(--gold3) 28%, var(--pitch) 64%, var(--gold) 100%);
  opacity:.92;
}}
.hero-mesh {{
  position:absolute; inset:0; pointer-events:none;
  background:
    radial-gradient(ellipse 70% 80% at 50% -10%, rgba(244,180,58,.22) 0%, transparent 62%),
    radial-gradient(ellipse 46% 46% at 16% 86%, rgba(16,185,129,.16) 0%, transparent 56%),
    radial-gradient(ellipse 46% 46% at 84% 82%, rgba(59,130,246,.13) 0%, transparent 56%);
}}
/* Faint pitch-grid (light strokes on the dark band) */
.hero-grid {{
  position:absolute; inset:0; pointer-events:none; opacity:.06;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60'%3E%3Crect width='60' height='60' fill='none' stroke='%23ffffff' stroke-width='.6'/%3E%3C/svg%3E");
  background-size:60px 60px;
  -webkit-mask-image:radial-gradient(ellipse 80% 80% at 50% 40%, #000 30%, transparent 75%);
          mask-image:radial-gradient(ellipse 80% 80% at 50% 40%, #000 30%, transparent 75%);
}}
.hero-content {{ position:relative; z-index:1; }}
.hero-logo {{ display:flex; justify-content:center; margin-bottom:20px; }}
.hero-logo img {{
  height:128px; width:auto; display:block;
  filter:drop-shadow(0 14px 32px rgba(0,0,0,.45));
  transition:transform .4s var(--ease), filter .4s var(--ease);
}}
.hero-logo img:hover {{
  transform:translateY(-4px) scale(1.03);
  filter:drop-shadow(0 18px 40px rgba(244,180,58,.42));
}}
@media (max-width:560px) {{ .hero-logo img {{ height:98px; }} }}
.hero-badge {{
  display:inline-flex; align-items:center; gap:8px;
  background:rgba(245,158,11,.1);
  border:1px solid rgba(245,158,11,.28);
  border-radius:100px;
  padding:7px 18px;
  font-size:11px; font-weight:700;
  color:var(--gold); letter-spacing:.1em; text-transform:uppercase;
  margin-bottom:24px;
  font-family:'Space Grotesk', sans-serif;
}}
.hero h1 {{
  font-family:'Space Grotesk', sans-serif;
  font-size:clamp(36px, 6.6vw, 76px);
  font-weight:700;
  letter-spacing:-.035em;
  background:linear-gradient(135deg, #ffffff 16%, #ffe7b3 46%, var(--gold3) 70%, #ffffff 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  line-height:1.04;
  margin-bottom:16px;
  text-shadow:0 2px 40px rgba(244,180,58,.12);
}}
.hero-sub {{
  color:#aebbd2;
  font-size:15.5px;
  max-width:600px;
  margin:0 auto 34px;
  line-height:1.6;
}}
.hero-sub b {{ color:#e7eefb; font-weight:600; }}
.hero-stats {{
  display:flex;
  justify-content:center;
  gap:0;
  flex-wrap:wrap;
  border:1px solid rgba(255,255,255,.14);
  border-radius:var(--radius);
  overflow:hidden;
  max-width:500px;
  margin:0 auto;
  background:rgba(255,255,255,.05);
  box-shadow:0 16px 44px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.07);
  backdrop-filter:blur(10px);
}}
.hstat {{
  flex:1;
  min-width:100px;
  text-align:center;
  padding:17px 12px;
  border-right:1px solid rgba(255,255,255,.12);
}}
.hstat:last-child {{ border-right:none; }}
.hstat-val {{
  font-family:'Space Grotesk', sans-serif;
  font-size:24px; font-weight:700;
  color:var(--gold3);
  font-variant-numeric:tabular-nums;
  letter-spacing:-.01em;
  text-shadow:0 0 24px rgba(244,180,58,.25);
}}
.hstat-lbl {{
  font-size:10px; color:#93a3bf;
  text-transform:uppercase; letter-spacing:.1em;
  margin-top:3px;
}}

/* ── NAV TABS ── */
.nav-wrap {{
  background:rgba(248,250,252,.82);
  border-bottom:1px solid var(--border);
  position:sticky; top:0; z-index:100;
  backdrop-filter:blur(20px) saturate(160%);
  -webkit-backdrop-filter:blur(20px) saturate(160%);
}}
.nav {{
  display:flex; gap:0;
  overflow-x:auto;
  max-width:1280px; margin:0 auto;
  scrollbar-width:none; padding:0 20px;
}}
.nav::-webkit-scrollbar {{ display:none; }}
.nav-tab {{
  flex-shrink:0;
  padding:15px 22px;
  border:none; background:none;
  color:var(--txt3);
  font-family:'Space Grotesk', sans-serif;
  font-size:13px; font-weight:600;
  cursor:pointer;
  border-bottom:3px solid transparent;
  transition:color .2s, border-color .2s;
  display:flex; align-items:center; gap:8px;
  white-space:nowrap;
  letter-spacing:.01em;
}}
.nav-tab:hover {{ color:var(--ink); }}
.nav-tab.active {{ color:var(--ink); font-weight:700; border-bottom-color:var(--gold); }}
.nav-tab:focus-visible {{
  outline:2px solid var(--gold);
  outline-offset:-2px;
  border-radius:4px;
}}
.nav-credit {{
  margin-left:auto; flex-shrink:0;
  display:flex; align-items:center; gap:11px;
  padding-left:20px;
}}
.nav-credit .nc-by {{
  font-family:'Space Grotesk', sans-serif;
  font-size:12px; font-weight:500; color:var(--txt2);
  text-decoration:none; white-space:nowrap; transition:color .2s;
}}
.nav-credit .nc-by:hover {{ color:var(--txt); }}
.nav-credit .nc-by b {{ color:var(--txt); font-weight:700; }}
.nav-credit .nc-link {{
  color:var(--txt3); display:flex; align-items:center;
  transition:color .2s, transform .15s;
}}
.nav-credit .nc-link:hover {{ color:var(--gold); transform:translateY(-1px); }}
@media (max-width:900px) {{ .nav-credit .nc-by {{ display:none; }} }}

/* ── MAIN LAYOUT ── */
.main {{ max-width:1280px; margin:0 auto; padding:36px 20px 64px; }}
.panel {{ display:none; }}
.panel.active {{ display:block; }}

/* ── SECTION HEADERS (editorial / broadcast) ── */
.sec-header {{ margin-bottom:30px; }}
.sec-title {{
  font-family:'Space Grotesk', sans-serif;
  font-size:clamp(26px, 3.4vw, 34px); font-weight:700;
  color:var(--ink);
  letter-spacing:-.028em;
  line-height:1.05;
  margin-bottom:8px;
}}
.sec-sub {{ color:var(--txt2); font-size:13.5px; line-height:1.55; max-width:720px; }}
.sec-eyebrow {{
  font-size:10.5px; font-weight:700;
  color:var(--gold); letter-spacing:.16em;
  text-transform:uppercase;
  margin-bottom:10px;
  font-family:'Space Grotesk', sans-serif;
  display:inline-flex; align-items:center; gap:9px;
}}
.sec-eyebrow::before {{
  content:''; width:22px; height:2px; border-radius:2px;
  background:linear-gradient(90deg, var(--gold), var(--pitch));
}}

/* ── GLASS CARD ── */
.card {{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:24px;
  box-shadow:0 4px 22px rgba(10,23,48,.07), 0 1px 3px rgba(10,23,48,.05);
  position:relative;
  overflow:hidden;
}}
.card::before {{
  content:'';
  position:absolute; inset:0;
  background:linear-gradient(135deg, rgba(20,33,58,.022) 0%, transparent 60%);
  pointer-events:none;
  border-radius:inherit;
}}
.card-sm {{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius2);
  padding:16px;
  transition:background .2s, border-color .2s;
}}
.card-sm:hover {{
  background:var(--card2);
  border-color:var(--border2);
}}

/* ── GRIDS ── */
.grid-2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; }}
.grid-3 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; }}
.grid-4 {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px; }}

/* ── CHAMPION CARDS ── */
.champ-grid {{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(210px,1fr));
  gap:16px;
  margin-bottom:36px;
}}
.champ-card {{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:24px 20px 20px;
  text-align:center;
  transition:transform .25s, border-color .25s, box-shadow .25s;
  cursor:pointer;
  position:relative;
  overflow:hidden;
}}
.champ-card::after {{
  content:'';
  position:absolute;
  inset:0;
  background:linear-gradient(135deg, rgba(20,33,58,.022) 0%, transparent 50%);
  pointer-events:none;
  border-radius:inherit;
}}
.champ-card:hover {{
  transform:translateY(-4px);
  border-color:rgba(245,158,11,.4);
  box-shadow:0 12px 40px rgba(20,33,58,.16), 0 0 0 1px rgba(245,158,11,.1);
}}
.champ-card.gold-card {{
  border-color:rgba(217,119,6,.4);
  background:linear-gradient(160deg, #FFFFFF 0%, rgba(245,158,11,.16) 100%);
  box-shadow:0 10px 32px rgba(217,119,6,.16), var(--shadow2);
}}
.champ-card.gold-card::before {{
  content:'';
  position:absolute;
  top:-60px; left:-60px; right:-60px;
  height:120px;
  background:radial-gradient(ellipse, rgba(245,158,11,.2) 0%, transparent 70%);
  pointer-events:none;
  animation:goldPulse 3s ease-in-out infinite;
}}
.champ-card.silver-card {{
  border-color:rgba(126,140,160,.4);
  background:linear-gradient(160deg, #FFFFFF 0%, rgba(148,163,184,.16) 100%);
}}
.champ-card.bronze-card {{
  border-color:rgba(185,115,51,.4);
  background:linear-gradient(160deg, #FFFFFF 0%, rgba(205,127,50,.15) 100%);
}}
@keyframes goldPulse {{
  0%, 100% {{ opacity:.8; }}
  50% {{ opacity:1; }}
}}
/* Shine sweep on gold card */
.champ-card.gold-card::after {{
  background:linear-gradient(105deg,
    transparent 30%,
    rgba(245,158,11,.12) 50%,
    transparent 70%
  );
  background-size:200% 100%;
  animation:shine 4s linear infinite;
}}
@keyframes shine {{
  0% {{ background-position:200% 0; }}
  100% {{ background-position:-200% 0; }}
}}
.rank-badge {{
  position:absolute; top:14px; left:14px;
  width:26px; height:26px;
  border-radius:50%;
  font-family:'Space Grotesk', sans-serif;
  font-size:12px; font-weight:700;
  display:flex; align-items:center; justify-content:center;
  color:var(--txt3);
  background:var(--card2);
  border:1px solid var(--border);
}}
.rank-badge.gold {{
  background:linear-gradient(135deg, var(--gold), var(--gold2));
  color:#000; border:none;
  box-shadow:0 2px 8px rgba(245,158,11,.4);
}}
.rank-badge.silver {{
  background:linear-gradient(135deg, #94A3B8, #CBD5E1);
  color:#0f1424; border:none;
}}
.rank-badge.bronze {{
  background:linear-gradient(135deg, #CD7F32, #E8A060);
  color:#0f1424; border:none;
}}
.champ-flag-wrap {{
  margin:10px auto 12px;
  display:flex; align-items:center; justify-content:center;
}}
.champ-flag-wrap .flag-img {{
  width:64px; height:43px;
  border-radius:5px;
  border:1px solid rgba(20,33,58,.14);
  box-shadow:0 4px 16px rgba(20,33,58,.16);
}}
.champ-name {{
  font-family:'Space Grotesk', sans-serif;
  font-size:14px; font-weight:700;
  margin-bottom:4px; color:var(--txt);
}}
.champ-group {{
  font-size:10px; color:var(--txt3);
  text-transform:uppercase; letter-spacing:.07em;
  margin-bottom:14px;
}}
.champ-pct {{
  font-family:'Space Grotesk', sans-serif;
  font-size:30px; font-weight:700;
  color:var(--gold);
  line-height:1;
  font-variant-numeric:tabular-nums;
  letter-spacing:-.02em;
}}
.champ-lbl {{
  font-size:10px; color:var(--txt3);
  text-transform:uppercase; letter-spacing:.07em;
  margin-top:3px;
}}
.mini-bar {{
  height:3px;
  background:rgba(20,33,58,.08);
  border-radius:99px;
  margin-top:14px;
  overflow:hidden;
}}
.mini-bar-fill {{
  height:100%;
  border-radius:99px;
  background:linear-gradient(90deg, var(--gold), var(--gold3));
  transition:width 1s cubic-bezier(.22,1,.36,1);
}}

/* ── PROBABILITY BARS ── */
.prob-row {{
  display:flex; align-items:center; gap:10px;
  margin-bottom:8px;
}}
.prob-label {{
  width:170px; font-size:12px; color:var(--txt2);
  flex-shrink:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  display:flex; align-items:center; gap:8px;
}}
.prob-bar-wrap {{
  flex:1;
  background:rgba(20,33,58,.07);
  border-radius:99px; height:5px;
  overflow:hidden;
}}
.prob-bar-fill {{
  height:100%; border-radius:99px;
  transition:width .8s cubic-bezier(.22,1,.36,1);
}}
.prob-val {{
  width:48px; text-align:right;
  font-family:'Space Grotesk', sans-serif;
  font-size:12px; font-weight:600;
  color:var(--gold); flex-shrink:0;
}}

/* ── GROUP CARDS ── */
.groups-grid {{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(430px,1fr));
  gap:16px;
}}
.group-card {{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  overflow:hidden;
  transition:border-color .2s, box-shadow .2s;
}}
.group-card:hover {{
  border-color:var(--border2);
  box-shadow:0 8px 32px rgba(20,33,58,.10);
}}
.group-header {{
  background:linear-gradient(135deg, var(--card2), rgba(245,158,11,.08));
  padding:14px 18px;
  display:flex; align-items:center; gap:12px;
  border-bottom:1px solid var(--border);
}}
.group-letter {{
  font-family:'Space Grotesk', sans-serif;
  font-size:28px; font-weight:700;
  color:var(--gold);
  line-height:1;
}}
.group-subtitle {{ font-size:11px; color:var(--txt3); margin-top:2px; }}
.group-team-row {{
  padding:11px 18px;
  display:flex; align-items:center; gap:10px;
  border-bottom:1px solid var(--border);
  transition:background .15s;
}}
.group-team-row:last-child {{ border-bottom:none; }}
.group-team-row:hover {{ background:var(--card2); }}
.gteam-flag {{ flex-shrink:0; }}
.gteam-info {{ flex:1; min-width:0; }}
.gteam-name {{ font-size:13px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.gteam-qualify {{ font-size:11px; color:var(--txt2); margin-top:1px; }}
.gteam-pos {{
  width:21px; height:21px; border-radius:6px; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
  font-family:'Space Grotesk', sans-serif; font-size:11px; font-weight:700;
}}
.gteam-pos.pos-q     {{ background:rgba(5,150,105,.14);  color:var(--green); }}
.gteam-pos.pos-maybe {{ background:rgba(217,119,6,.15);  color:var(--gold); }}
.gteam-pos.pos-out   {{ background:rgba(20,33,58,.06);   color:var(--txt3); }}
.gteam-meter {{ width:66px; flex-shrink:0; text-align:right; }}
.gteam-pct {{
  font-family:'Space Grotesk', sans-serif;
  font-size:15px; font-weight:700; line-height:1;
  font-variant-numeric:tabular-nums; letter-spacing:-.01em;
}}
.gteam-pct-sym {{ font-size:10px; font-weight:600; opacity:.7; margin-left:1px; }}
.gteam-pct.qb-hot  {{ color:var(--green); }}
.gteam-pct.qb-mid  {{ color:var(--gold); }}
.gteam-pct.qb-cold {{ color:var(--txt3); }}
.gteam-bar {{ height:4px; background:rgba(20,33,58,.08); border-radius:99px; margin-top:5px; overflow:hidden; }}
.gteam-bar-fill {{ height:100%; width:0; border-radius:99px; transition:width 1s cubic-bezier(.22,1,.36,1); }}
.gteam-bar-fill.qb-hot  {{ background:linear-gradient(90deg, var(--green), var(--green2)); }}
.gteam-bar-fill.qb-mid  {{ background:linear-gradient(90deg, var(--gold), var(--gold2)); }}
.gteam-bar-fill.qb-cold {{ background:rgba(100,116,139,.55); }}
.gteam-meter-lbl {{ font-size:9px; font-weight:500; color:var(--txt4); text-transform:uppercase; letter-spacing:.05em; margin-top:3px; }}

/* ── GROUP HEAT TABLE (almonadavid-style colour grading) ── */
.gh-table {{ width:100%; border-collapse:collapse; font-family:'Space Grotesk', sans-serif; font-size:12px; }}
.gh-table th, .gh-table td {{ padding:7px 7px; text-align:center; white-space:nowrap; }}
.gh-table thead th {{ font-size:9.5px; font-weight:700; text-transform:uppercase; letter-spacing:.04em;
  color:var(--txt3); border-bottom:1px solid var(--border); }}
.gh-table thead th.gh-team {{ text-align:left; padding-left:8px; }}
.gh-team {{ text-align:left !important; }}
.gh-table td.gh-team {{ display:flex; align-items:center; gap:8px; font-weight:600; color:var(--txt); min-width:120px; }}
.gh-table td.gh-team .flag-img {{ width:18px; height:12px; flex-shrink:0; }}
.gh-pos {{ width:18px; color:var(--txt4); font-size:10px; font-weight:700; }}
.gh-cell {{ font-variant-numeric:tabular-nums; font-weight:600; border-radius:5px; min-width:42px; }}
.gh-table tbody tr {{ border-bottom:1px solid var(--border); }}
.gh-table tbody tr:last-child {{ border-bottom:none; }}
.gh-table tbody tr:hover td.gh-team {{ color:var(--gold); }}
.gh-sep {{ border-left:2px solid var(--border2); }}
.gh-wrap {{ padding:6px 14px 12px; }}

/* heat shading reused on the standings data-table */
.data-table td.heat {{ font-weight:700; font-variant-numeric:tabular-nums; border-radius:5px; }}

/* ── BRACKET ── */
.bracket-flow {{
  display:flex;
  flex-direction:column;
  gap:24px;
}}
.bracket-round-section {{
  position:relative;
}}
.bracket-round-label {{
  font-family:'Space Grotesk', sans-serif;
  font-size:10px; font-weight:700;
  color:var(--txt3);
  text-transform:uppercase; letter-spacing:.12em;
  margin-bottom:12px;
  display:flex; align-items:center; gap:10px;
}}
.bracket-round-label::after {{
  content:'';
  flex:1;
  height:1px;
  background:var(--border);
}}
.bracket-matches-grid {{
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(240px, 1fr));
  gap:10px;
}}
.bracket-matches-grid.sf-grid {{
  grid-template-columns:repeat(2, 1fr);
  max-width:600px;
  margin:0 auto;
}}
.bracket-matches-grid.final-grid {{
  grid-template-columns:1fr;
  max-width:420px;
  margin:0 auto;
}}
.b-match {{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius2);
  overflow:hidden;
  transition:border-color .2s, box-shadow .2s;
}}
.b-match:hover {{
  border-color:rgba(245,158,11,.35);
  box-shadow:0 4px 20px rgba(20,33,58,.10);
}}
.b-match-header {{
  padding:6px 10px;
  background:var(--card2);
  border-bottom:1px solid var(--border);
  font-size:9px; font-weight:700;
  color:var(--txt3); text-transform:uppercase; letter-spacing:.08em;
  display:flex; justify-content:space-between; align-items:center;
}}
.b-team {{
  display:flex; align-items:center; gap:9px;
  padding:9px 12px;
  font-size:12px;
  border-bottom:1px solid var(--border);
  transition:background .15s;
}}
.b-team:last-child {{ border-bottom:none; }}
.b-team.winner {{
  background:rgba(16,185,129,.07);
  font-weight:600;
  color:var(--txt);
}}
.b-team.loser {{ color:var(--txt3); }}
.b-name {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.b-prob {{
  font-family:'Space Grotesk', sans-serif;
  font-size:10px; color:var(--gold); font-weight:700; flex-shrink:0;
}}
.b-slot {{
  font-size:10px; color:var(--txt3);
  padding:5px 12px 6px;
  border-top:1px solid var(--border);
  background:rgba(20,33,58,.03);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}}

/* ── BRACKET TREE (two-sided, every slot filled) ── */
.bracket-scroll-hint {{
  text-align:center; font-size:11px; color:var(--txt3);
  margin-bottom:12px; font-family:'Space Grotesk', sans-serif; letter-spacing:.03em;
}}
.bracket-scroll {{ overflow-x:auto; overflow-y:hidden; padding:4px 2px 18px; }}
.bracket-scroll::-webkit-scrollbar {{ height:9px; }}
.bracket-scroll::-webkit-scrollbar-track {{ background:transparent; }}
.bracket-scroll::-webkit-scrollbar-thumb {{ background:var(--border2); border-radius:99px; }}
.bracket-tree {{ display:flex; align-items:stretch; min-width:1560px; min-height:800px; }}
.bwing {{ display:flex; flex:1; }}
.bcol {{ flex:1; display:flex; flex-direction:column; padding:0 16px; position:relative; --bln:rgba(20,33,58,.2); }}
.bm-slot {{ flex:1; display:flex; align-items:center; position:relative; }}
.bm-card {{
  width:100%; background:var(--card);
  border:1px solid var(--border); border-radius:9px;
  overflow:hidden; box-shadow:var(--shadow2);
  position:relative; z-index:1;
  transition:border-color .18s, box-shadow .18s, transform .18s;
}}
.bm-card:hover {{ border-color:rgba(217,119,6,.45); box-shadow:0 8px 24px rgba(20,33,58,.13); transform:translateY(-1px); z-index:3; }}
.bm-top {{
  display:flex; justify-content:space-between; align-items:center;
  padding:4px 9px; background:var(--card2);
  border-bottom:1px solid var(--border);
  font-family:'Space Grotesk', sans-serif;
  font-size:8.5px; font-weight:700; letter-spacing:.04em;
  color:var(--txt3); text-transform:uppercase;
}}
.bt {{
  display:flex; align-items:center; gap:7px;
  padding:6px 9px; border-bottom:1px solid var(--border); min-width:0;
}}
.bt:last-child {{ border-bottom:none; }}
.bt.win {{ background:rgba(5,150,105,.1); }}
.bt.lose {{ opacity:.82; }}
.bt .flag-img {{ width:21px; height:14px; }}
.bt.lose .flag-img {{ filter:grayscale(.35); }}
.bt-name {{
  flex:1; font-size:11px; font-weight:600; color:var(--txt2);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}}
.bt.win .bt-name {{ color:var(--txt); font-weight:700; }}
.bt-p {{
  font-family:'Space Grotesk', sans-serif;
  font-size:10px; font-weight:700; flex-shrink:0;
}}
.bt.win  .bt-p {{ color:var(--green); }}
.bt.lose .bt-p {{ color:var(--txt3); }}
.bt-flag-ph {{ width:21px; height:14px; border-radius:3px; background:rgba(20,33,58,.07); flex-shrink:0; }}

/* connector lines — LEFT wing flows right */
.bwing.left  .bcol:not(.sf) .bm-card::after  {{ content:''; position:absolute; left:100%;  top:50%; width:16px; height:2px; background:var(--bln); }}
.bwing.left  .bcol:not(.sf) .bm-slot:nth-child(odd)::after {{ content:''; position:absolute; left:calc(100% + 16px);  top:50%; width:2px; height:100%; background:var(--bln); }}
.bwing.left  .bcol:not(.r32) .bm-card::before {{ content:''; position:absolute; right:100%; top:50%; width:16px; height:2px; background:var(--bln); }}
/* connector lines — RIGHT wing flows left (mirrored) */
.bwing.right .bcol:not(.sf) .bm-card::after  {{ content:''; position:absolute; right:100%; top:50%; width:16px; height:2px; background:var(--bln); }}
.bwing.right .bcol:not(.sf) .bm-slot:nth-child(odd)::after {{ content:''; position:absolute; right:calc(100% + 16px); top:50%; width:2px; height:100%; background:var(--bln); }}
.bwing.right .bcol:not(.r32) .bm-card::before {{ content:''; position:absolute; left:100%;  top:50%; width:16px; height:2px; background:var(--bln); }}

/* center column: champion + final + bronze */
.bcenter {{ display:flex; flex-direction:column; justify-content:center; align-items:center; gap:16px; padding:0 12px; min-width:214px; }}
.bcenter .bm-slot {{ flex:none; display:block; }}
.bchamp {{
  text-align:center; width:100%;
  background:linear-gradient(160deg, #FFFFFF 0%, rgba(245,158,11,.16) 100%);
  border:1px solid rgba(217,119,6,.3); border-radius:16px;
  padding:16px 22px 18px; box-shadow:var(--shadow);
}}
.bchamp-trophy {{ font-size:28px; line-height:1; }}
.bchamp-label {{
  font-family:'Space Grotesk', sans-serif;
  font-size:9px; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
  color:var(--gold); margin:6px 0 8px;
}}
.bchamp-flag {{ display:flex; justify-content:center; }}
.bchamp-flag .flag-img {{ width:54px; height:36px; border-radius:6px; box-shadow:0 6px 18px rgba(20,33,58,.2); }}
.bchamp-name {{
  font-family:'Space Grotesk', sans-serif;
  font-size:18px; font-weight:700; color:var(--txt);
  margin-top:9px; letter-spacing:-.01em;
}}
.bchamp-sub {{ font-size:11px; color:var(--txt2); margin-top:2px; }}
.bcenter-final {{ width:100%; }}
.bcenter-cap {{
  font-family:'Space Grotesk', sans-serif;
  font-size:9px; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  color:var(--gold); text-align:center; margin-bottom:6px;
}}
.bcenter-cap.muted {{ color:var(--txt3); }}
.bcenter-final .bm-card {{ border-color:rgba(217,119,6,.38); box-shadow:0 6px 22px rgba(217,119,6,.12); }}
.bcenter-bronze {{ width:100%; opacity:.94; }}
@media (min-width:1620px) {{ .bracket-scroll-hint {{ display:none; }} }}

/* ── PREDICTED FINAL ── */
.final-hero {{
  position:relative;
  background:linear-gradient(160deg, #FFFFFF 0%, rgba(245,158,11,.14) 100%);
  border:1px solid rgba(217,119,6,.32);
  box-shadow:var(--shadow);
  border-radius:20px;
  padding:40px 32px 36px;
  text-align:center;
  overflow:hidden;
  margin-bottom:32px;
}}
.final-hero::before {{
  content:'';
  position:absolute; inset:0;
  background:radial-gradient(ellipse 80% 60% at 50% 0%, rgba(245,158,11,.12) 0%, transparent 60%);
  pointer-events:none;
}}
.final-hero-grid {{
  position:absolute; inset:0; pointer-events:none; opacity:.05;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='40' height='40'%3E%3Ccircle cx='20' cy='20' r='1' fill='%2314213A'/%3E%3C/svg%3E");
  background-size:40px 40px;
}}
.final-eyebrow {{
  font-family:'Space Grotesk', sans-serif;
  font-size:10px; font-weight:700;
  color:var(--gold); text-transform:uppercase; letter-spacing:.15em;
  margin-bottom:28px;
  display:flex; align-items:center; justify-content:center; gap:12px;
  position:relative; z-index:1;
}}
.final-eyebrow::before, .final-eyebrow::after {{
  content:''; flex:1; max-width:80px;
  height:1px; background:linear-gradient(90deg, transparent, rgba(245,158,11,.4));
}}
.final-eyebrow::after {{ background:linear-gradient(90deg, rgba(245,158,11,.4), transparent); }}
.final-teams {{
  display:flex; align-items:center; justify-content:center; gap:20px;
  position:relative; z-index:1;
  flex-wrap:wrap;
}}
.final-team {{ text-align:center; flex:1; min-width:140px; max-width:200px; }}
.final-flag-wrap {{
  margin:0 auto 12px;
  display:flex; align-items:center; justify-content:center;
}}
.final-flag-wrap .flag-img {{
  width:80px; height:53px;
  border-radius:6px;
  box-shadow:0 6px 24px rgba(20,33,58,.16);
  border:1px solid rgba(20,33,58,.14);
}}
.final-name {{
  font-family:'Space Grotesk', sans-serif;
  font-size:18px; font-weight:700;
  color:var(--txt); margin-bottom:4px;
}}
.final-pct {{ font-size:12px; color:var(--txt2); }}
.vs-divider {{
  display:flex; flex-direction:column; align-items:center; gap:6px; flex-shrink:0;
}}
.vs-text {{
  font-family:'Space Grotesk', sans-serif;
  font-size:13px; font-weight:700; color:var(--txt3);
  letter-spacing:.1em;
}}
.vs-line {{ width:1px; height:30px; background:var(--border); }}
.final-champion {{
  position:relative; z-index:1;
  margin-top:28px; padding-top:24px;
  border-top:1px solid rgba(245,158,11,.2);
}}
.final-champ-label {{
  font-size:10px; font-weight:700; color:var(--txt3);
  text-transform:uppercase; letter-spacing:.1em;
  margin-bottom:8px;
}}
.final-champ-name {{
  font-family:'Space Grotesk', sans-serif;
  font-size:28px; font-weight:700;
  color:var(--gold);
  display:flex; align-items:center; justify-content:center; gap:12px;
  letter-spacing:-.02em;
}}
.final-champ-sub {{ font-size:13px; color:var(--txt2); margin-top:6px; }}

/* ── PLAYER / AWARD CARDS ── */
.award-grid {{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
  gap:16px; margin-bottom:28px;
}}
.award-card {{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:22px;
  transition:border-color .3s var(--ease), box-shadow .3s var(--ease), transform .3s var(--ease);
  position:relative;
  overflow:hidden;
}}
.award-card::before {{
  content:''; position:absolute; inset:0;
  background:linear-gradient(135deg, rgba(20,33,58,.025) 0%, transparent 60%);
  pointer-events:none; border-radius:inherit;
}}
.award-card:hover {{
  transform:translateY(-5px);
  border-color:rgba(217,119,6,.34);
  box-shadow:0 16px 42px rgba(20,33,58,.15), 0 0 0 1px rgba(217,119,6,.10);
}}
.award-title {{
  font-size:10px; font-weight:700; color:var(--txt3);
  text-transform:uppercase; letter-spacing:.1em;
  margin-bottom:14px;
  font-family:'Space Grotesk', sans-serif;
}}
.award-player {{
  display:flex; align-items:center; gap:14px;
}}
.award-flag-wrap {{ flex-shrink:0; }}
.award-flag-wrap .flag-img {{
  width:48px; height:32px; border-radius:4px;
  box-shadow:0 3px 12px rgba(20,33,58,.10);
  border:1px solid rgba(20,33,58,.14);
}}
.award-info h3 {{
  font-family:'Space Grotesk', sans-serif;
  font-size:16px; font-weight:700; margin-bottom:2px;
}}
.award-info p {{ font-size:11px; color:var(--txt2); }}
.award-stat {{
  font-family:'Space Grotesk', sans-serif;
  font-size:22px; font-weight:700;
  color:var(--gold); margin-top:6px;
  font-variant-numeric:tabular-nums;
}}
.award-footer {{
  margin-top:14px; padding-top:12px;
  border-top:1px solid var(--border);
  font-size:11px; color:var(--txt3);
  line-height:1.5;
}}

/* ── PLAYER ROWS ── */
.player-row {{
  display:flex; align-items:center; gap:10px;
  padding:10px 0;
  border-bottom:1px solid var(--border);
  transition:background .15s;
}}
.player-row:last-child {{ border-bottom:none; }}
.player-row:hover {{ background:rgba(20,33,58,.03); border-radius:6px; padding-left:6px; }}
.player-rank {{
  width:22px; font-size:12px; font-weight:700;
  color:var(--txt3); flex-shrink:0; text-align:center;
  font-family:'Space Grotesk', sans-serif;
}}
.player-rank.top3 {{ color:var(--gold); }}
.player-flag {{ flex-shrink:0; }}
.player-info {{ flex:1; min-width:0; }}
.player-name {{
  font-size:13px; font-weight:600;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}}
.player-team {{ font-size:11px; color:var(--txt2); margin-top:1px; }}
.player-xg {{
  font-family:'Space Grotesk', sans-serif;
  font-size:13px; font-weight:700; color:var(--gold); flex-shrink:0;
  font-variant-numeric:tabular-nums;
}}
.player-goals {{ font-size:11px; color:var(--txt2); flex-shrink:0; }}

/* ── GOLDEN BALL RACE METER ── */
.gb-meter {{
  flex-shrink:0; width:120px; height:7px; border-radius:4px;
  background:rgba(234,179,8,.12); overflow:hidden;
}}
.gb-meter-fill {{
  height:100%; border-radius:4px;
  background:linear-gradient(90deg, rgba(234,179,8,.55), rgba(180,130,0,1));
}}
.gb-pct {{ color:rgba(180,130,0,1); width:46px; text-align:right; }}
@media (max-width:680px) {{ .gb-meter {{ display:none; }} }}

/* ── PLAYER PHOTO AVATARS ── */
.pphoto {{ width:46px; height:46px; border-radius:50%; object-fit:cover; object-position:top center;
  border:2px solid var(--card); box-shadow:0 2px 8px rgba(20,33,58,.14); background:var(--card2); }}
.award-avatar {{ position:relative; flex-shrink:0; width:48px; height:48px; display:flex; align-items:center; justify-content:center; }}
.award-avatar .flag-img {{ width:40px; height:27px; }}
.pphoto-flag {{ position:absolute; bottom:-2px; right:-4px; line-height:0; }}
.pphoto-flag .flag-img {{ width:18px; height:12px; border:1.5px solid var(--card); }}
.player-av {{ position:relative; width:30px; height:30px; flex-shrink:0; }}
.pphoto.sm {{ width:30px; height:30px; }}
.player-av-flag {{ position:absolute; bottom:-2px; right:-3px; line-height:0; }}
.player-av-flag .flag-img {{ width:14px; height:10px; border:1px solid var(--card); }}

/* ── TEAM GUIDE ── */
.pg-head {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }}
.pg-head-row {{ display:flex; flex-direction:column; gap:2px; background:var(--card2); border:1px solid var(--border); border-radius:var(--radius2); padding:8px 14px; }}
.pg-k {{ font-family:'Space Grotesk', sans-serif; font-size:9px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--txt3); }}
.pg-v {{ font-family:'Space Grotesk', sans-serif; font-size:14px; font-weight:600; color:var(--txt); }}
.pg-bio {{ font-size:13px; line-height:1.6; color:var(--txt2); margin-bottom:16px; }}
.pg-sw {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:18px; }}
.pg-sw-card {{ border-radius:var(--radius2); padding:13px 15px; border:1px solid var(--border); }}
.pg-str {{ background:rgba(5,150,105,.06); border-color:rgba(5,150,105,.22); }}
.pg-wk  {{ background:rgba(220,38,38,.05); border-color:rgba(220,38,38,.2); }}
.pg-sw-t {{ font-family:'Space Grotesk', sans-serif; font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; margin-bottom:6px; }}
.pg-str .pg-sw-t {{ color:var(--green); }}
.pg-wk  .pg-sw-t {{ color:var(--red); }}
.pg-sw-card p {{ font-size:12.5px; line-height:1.55; color:var(--txt2); }}
.pg-highlights {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:22px; }}
.pg-highlight {{ display:flex; gap:14px; padding:14px; border-radius:var(--radius); border:1px solid var(--border);
  background:linear-gradient(135deg, var(--card) 0%, color-mix(in srgb, var(--hl) 8%, var(--card)) 100%);
  position:relative; overflow:hidden; }}
.pg-hl-badge {{ position:absolute; top:0; left:0; font-family:'Space Grotesk', sans-serif; font-size:9px; font-weight:700;
  letter-spacing:.06em; text-transform:uppercase; color:var(--hl); background:color-mix(in srgb, var(--hl) 14%, transparent);
  padding:3px 10px; border-bottom-right-radius:8px; }}
.pg-hl-photo {{ flex-shrink:0; align-self:flex-end; line-height:0; }}
.pg-hl-photo img {{ width:70px; height:70px; border-radius:50%; object-fit:cover; object-position:top center;
  border:2px solid var(--hl); box-shadow:0 3px 12px rgba(20,33,58,.16); }}
.pg-hl-info {{ flex:1; min-width:0; padding-top:14px; }}
.pg-hl-name {{ font-family:'Space Grotesk', sans-serif; font-size:15px; font-weight:700; color:var(--txt); }}
.pg-hl-sub {{ font-size:11px; color:var(--txt3); margin-bottom:6px; }}
.pg-hl-bio {{ font-size:11.5px; line-height:1.5; color:var(--txt2);
  display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient:vertical; overflow:hidden; }}
.pg-squad-title {{ font-family:'Space Grotesk', sans-serif; font-size:13px; font-weight:700; color:var(--txt);
  margin:6px 0 12px; padding-top:14px; border-top:1px solid var(--border); }}
.pg-pos-label {{ font-family:'Space Grotesk', sans-serif; font-size:10px; font-weight:700; letter-spacing:.08em;
  text-transform:uppercase; color:var(--gold); margin:14px 0 10px; display:flex; align-items:center; gap:8px; }}
.pg-pos-label span {{ color:var(--txt4); }}
.pg-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(96px, 1fr)); gap:12px; }}
.pg-player {{ text-align:center; }}
.pg-ph {{ position:relative; width:64px; height:64px; margin:0 auto 7px; }}
.pg-ph img {{ width:64px; height:64px; border-radius:50%; object-fit:cover; object-position:top center;
  border:2px solid var(--border2); background:var(--card2); transition:border-color .15s, transform .15s; }}
.pg-player:hover .pg-ph img {{ border-color:var(--gold); transform:translateY(-2px); }}
.pg-player.is-star .pg-ph img {{ border-color:var(--gold); box-shadow:0 0 0 2px rgba(217,119,6,.25); }}
.pg-player.is-wonder .pg-ph img {{ border-color:var(--green); }}
.pg-ph-x {{ display:block; width:64px; height:64px; border-radius:50%; background:var(--card2); border:2px solid var(--border); }}
.pg-no {{ position:absolute; bottom:0; right:2px; min-width:18px; height:18px; padding:0 4px; border-radius:9px;
  background:var(--txt); color:var(--card); font-family:'Space Grotesk', sans-serif; font-size:10px; font-weight:700;
  display:flex; align-items:center; justify-content:center; }}
.pg-nm {{ font-size:11.5px; font-weight:600; color:var(--txt); line-height:1.25; }}
.pg-meta {{ font-size:9.5px; color:var(--txt3); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.pg-tag {{ display:inline-block; margin-top:4px; font-size:8.5px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
  color:var(--gold); background:rgba(217,119,6,.12); padding:2px 7px; border-radius:99px; font-family:'Space Grotesk', sans-serif; }}
.pg-player.is-wonder .pg-tag {{ color:var(--green); background:rgba(5,150,105,.12); }}
.pg-credit {{ margin-top:18px; padding-top:12px; border-top:1px solid var(--border); font-size:10px; color:var(--txt4); }}
@media (max-width:560px) {{ .pg-sw, .pg-highlights {{ grid-template-columns:1fr; }} }}

/* ── TEAM DEEP DIVE ── */
.team-header {{
  display:flex; align-items:center; gap:24px;
  margin-bottom:24px;
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:28px 24px;
  position:relative; overflow:hidden;
}}
.team-header::before {{
  content:''; position:absolute; inset:0;
  background:linear-gradient(135deg, rgba(20,33,58,.025), transparent);
  pointer-events:none;
}}
.team-flag-lg {{ flex-shrink:0; }}
.team-flag-lg .flag-img {{
  width:88px; height:59px;
  border-radius:7px;
  box-shadow:0 6px 24px rgba(20,33,58,.16);
  border:1px solid rgba(20,33,58,.14);
}}
.team-meta {{ position:relative; z-index:1; }}
.team-meta h2 {{
  font-family:'Space Grotesk', sans-serif;
  font-size:30px; font-weight:700;
  margin-bottom:4px; letter-spacing:-.02em;
}}
.team-meta p {{ color:var(--txt2); font-size:13px; }}
.team-stat-strip {{
  display:flex; gap:28px; margin-top:16px; flex-wrap:wrap;
}}
.ts {{ text-align:center; }}
.ts-val {{
  font-family:'Space Grotesk', sans-serif;
  font-size:22px; font-weight:700;
  color:var(--gold);
  font-variant-numeric:tabular-nums;
}}
.ts-lbl {{
  font-size:10px; color:var(--txt3);
  text-transform:uppercase; letter-spacing:.07em;
  margin-top:2px;
}}

.team-select-wrap {{ margin-bottom:28px; }}
select {{
  background:var(--card2);
  color:var(--txt);
  border:1px solid var(--border);
  border-radius:var(--radius2);
  padding:11px 40px 11px 16px;
  font-family:'Space Grotesk', sans-serif;
  font-size:14px; font-weight:500;
  cursor:pointer; width:100%; max-width:340px;
  outline:none; appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%2394A3B8' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat:no-repeat; background-position:right 14px center;
  transition:border-color .2s;
}}
select:focus {{ border-color:var(--gold); box-shadow:0 0 0 3px rgba(245,158,11,.1); }}

/* ── ROUND PROGRESS BARS ── */
.round-progress {{ margin-bottom:14px; }}
.rp-label {{
  font-size:12px; color:var(--txt2);
  margin-bottom:5px;
  display:flex; justify-content:space-between; align-items:center;
}}
.rp-label span:last-child {{
  font-family:'Space Grotesk', sans-serif;
  font-weight:600; font-size:12px;
}}
.rp-bar {{ height:7px; background:rgba(20,33,58,.07); border-radius:99px; overflow:hidden; }}
.rp-fill {{
  height:100%; border-radius:99px;
  transition:width .9s cubic-bezier(.22,1,.36,1);
}}

/* ── DATA TABLE ── */
table.data-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
table.data-table th {{
  padding:10px 14px;
  text-align:left;
  font-family:'Space Grotesk', sans-serif;
  font-size:10px; font-weight:700;
  color:var(--txt3); text-transform:uppercase; letter-spacing:.08em;
  border-bottom:1px solid var(--border);
}}
table.data-table td {{
  padding:11px 14px;
  border-bottom:1px solid var(--border);
  color:var(--txt2);
}}
table.data-table tr:last-child td {{ border-bottom:none; }}
table.data-table tr:hover td {{ background:rgba(20,33,58,.025); }}
table.data-table td.highlight {{ color:var(--txt); font-weight:600; }}
table.data-table td.gold {{
  color:var(--gold); font-weight:700;
  font-family:'Space Grotesk', sans-serif;
}}

/* ── PROJECTIONS HEATMAP TABLE ── */
.proj-wrap {{
  border:1px solid var(--border); border-radius:var(--radius);
  background:var(--card); box-shadow:var(--shadow2); overflow:visible;
}}
table.proj-table {{ border-collapse:collapse; width:100%; font-family:'Space Grotesk', sans-serif; font-size:12px; }}
.proj-table th, .proj-table td {{ padding:7px 9px; text-align:center; white-space:nowrap; }}
.proj-table thead th {{
  position:sticky; top:50px; z-index:2;
  background:var(--card2); color:var(--txt2);
  font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.04em;
  border-bottom:2px solid var(--border2); user-select:none;
}}
.proj-table thead th.proj-h {{ cursor:pointer; transition:color .15s, background .15s; }}
.proj-h:hover {{ color:var(--gold); }}
.proj-h.active {{ color:var(--gold); background:rgba(217,119,6,.1); }}
.proj-rk {{ width:28px; color:var(--txt4); font-size:10px; font-weight:600; }}
.proj-team {{ text-align:left !important; min-width:158px; }}
.proj-team {{ display:flex; align-items:center; gap:9px; font-weight:600; color:var(--txt); }}
.proj-team .flag-img {{ width:18px; height:12px; flex-shrink:0; }}
.proj-grp, .proj-grp-h {{ color:var(--txt3); font-weight:700; }}
.proj-cell {{ font-variant-numeric:tabular-nums; font-weight:600; min-width:42px; }}
.proj-table tbody tr:hover .proj-team {{ color:var(--gold); }}
.proj-grp-start td {{ border-top:2px solid var(--border2); }}
.proj-sep {{ border-left:2px solid var(--border2); }}
.proj-foot {{ padding:11px 15px; font-size:11px; color:var(--txt3); background:var(--card); border-top:1px solid var(--border); }}
.proj-foot a {{ color:var(--gold); cursor:pointer; font-weight:600; }}

/* ── UTILITIES ── */
.flex {{ display:flex; }}
.gap-2 {{ gap:8px; }} .gap-3 {{ gap:12px; }} .gap-4 {{ gap:16px; }}
.items-center {{ align-items:center; }}
.justify-between {{ justify-content:space-between; }}
.mb-4 {{ margin-bottom:16px; }} .mb-6 {{ margin-bottom:24px; }} .mb-8 {{ margin-bottom:32px; }}
.text-sm {{ font-size:13px; }} .text-xs {{ font-size:11px; }}
.font-bold {{ font-weight:700; }}
.text-gold {{ color:var(--gold); }} .text-green {{ color:var(--green); }}
.text-blue {{ color:var(--blue); }} .text-red {{ color:var(--red); }}
.text-muted {{ color:var(--txt2); }}
.w-full {{ width:100%; }}
.separator {{ height:1px; background:var(--border); margin:28px 0; }}
.tag {{
  display:inline-flex; align-items:center; gap:4px;
  padding:3px 10px; border-radius:99px;
  font-size:10px; font-weight:700; letter-spacing:.05em;
  font-family:'Space Grotesk', sans-serif;
}}
.tag-gold {{ background:rgba(245,158,11,.15); color:var(--gold); }}
.tag-blue {{ background:rgba(59,130,246,.15); color:var(--blue); }}
.tag-green {{ background:rgba(16,185,129,.15); color:var(--green); }}
.tag-red {{ background:rgba(239,68,68,.15); color:var(--red); }}
.tag-purple {{ background:rgba(139,92,246,.15); color:var(--purple); }}

/* ── SCATTER QUADRANT LABELS ── */
.quadrant-label {{
  position:absolute;
  font-size:9px; font-weight:700;
  color:rgba(20,33,58,.16);
  text-transform:uppercase; letter-spacing:.1em;
  pointer-events:none;
  font-family:'Space Grotesk', sans-serif;
}}

/* ── LOLLIPOP RANKING ── */
.lollipop-list {{ display:flex; flex-direction:column; gap:6px; }}
.lollipop-row {{
  display:flex; align-items:center; gap:10px;
}}
.lollipop-label {{
  width:160px; flex-shrink:0;
  display:flex; align-items:center; gap:8px;
  font-size:12px; color:var(--txt2);
  overflow:hidden; white-space:nowrap;
}}
.lollipop-track {{
  flex:1; height:2px;
  background:rgba(20,33,58,.08);
  border-radius:99px;
  position:relative;
}}
.lollipop-bar {{
  position:absolute; top:0; left:0;
  height:100%; border-radius:99px;
  background:linear-gradient(90deg, var(--gold), var(--gold2));
  transition:width .9s cubic-bezier(.22,1,.36,1);
}}
.lollipop-dot {{
  position:absolute; right:0; top:50%;
  transform:translate(50%,-50%);
  width:8px; height:8px;
  border-radius:50%;
  background:var(--gold);
  box-shadow:0 0 6px rgba(245,158,11,.6);
}}
.lollipop-val {{
  width:44px; text-align:right; flex-shrink:0;
  font-family:'Space Grotesk', sans-serif;
  font-size:11px; font-weight:700; color:var(--gold);
}}

/* ── GUARDIAN-STYLE BIO POPUP ── */
.bio-modal {{
  position:fixed; inset:0; z-index:200;
  display:none; align-items:flex-start; justify-content:center;
  background:rgba(20,33,58,.55); backdrop-filter:blur(4px);
  padding:5vh 16px; overflow-y:auto;
}}
.bio-modal.open {{ display:flex; }}
.bio-card {{
  position:relative; width:100%; max-width:468px;
  background:#fff; border-radius:14px;
  border-top:4px solid var(--blue);
  box-shadow:0 24px 70px rgba(20,33,58,.35);
  padding:26px 30px 24px; animation:bioIn .22s cubic-bezier(.22,1,.36,1);
}}
@keyframes bioIn {{ from {{ opacity:0; transform:translateY(14px) scale(.98); }} to {{ opacity:1; transform:none; }} }}
.bio-close {{
  position:absolute; top:14px; right:14px; width:30px; height:30px;
  border:none; border-radius:50%; background:#f1f3f5; color:#555;
  font-size:17px; line-height:1; cursor:pointer; transition:background .15s;
}}
.bio-close:hover {{ background:#e2451f; color:#fff; }}
.bio-top {{ display:flex; gap:16px; align-items:flex-start; }}
.bio-photo {{ width:72px; height:72px; border-radius:50%; object-fit:cover; flex-shrink:0;
  border:2px solid #e8eaed; background:#f1f3f5; }}
.bio-name {{ font-family:Georgia,'Times New Roman',serif; font-size:25px; font-weight:700;
  color:#121212; line-height:1.05; letter-spacing:-.01em; }}
.bio-role {{ font-family:Georgia,serif; font-style:italic; font-size:14.5px; color:var(--blue); margin-top:3px; }}
.bio-flagline {{ display:flex; align-items:center; gap:7px; margin-top:7px; font-size:12px; color:var(--txt2); font-weight:600; }}
.bio-flagline .flag-img {{ width:20px; height:14px; }}
.bio-kv-row {{ display:flex; gap:26px; margin:18px 0 4px; border-top:1px solid #eceef0; padding-top:14px; }}
.bio-kv .k {{ font-size:9.5px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#9aa3ad; }}
.bio-kv .v {{ font-family:Georgia,serif; font-size:14px; font-weight:700; color:#1a1a1a; margin-top:2px; }}
.bio-rating-lbl {{ font-size:9.5px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#9aa3ad; margin:16px 0 7px; }}
.bio-dots {{ display:flex; gap:9px; }}
.bio-dot {{ text-align:center; }}
.bio-dot .d {{ width:22px; height:22px; border-radius:50%; margin:0 auto; border:1px solid rgba(20,33,58,.12); }}
.bio-dot .dl {{ font-size:8.5px; font-weight:700; color:#9aa3ad; margin-top:3px; letter-spacing:.02em; }}
.bio-stats {{ display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 6px; }}
.bio-stat {{ flex:1; min-width:54px; text-align:center; background:#f7f8fa; border-radius:9px; padding:9px 4px; }}
.bio-stat .sv {{ font-family:'Space Grotesk',sans-serif; font-size:17px; font-weight:700; color:#14213a; font-variant-numeric:tabular-nums; }}
.bio-stat .sl {{ font-size:8.5px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:#9aa3ad; margin-top:2px; }}
.bio-statgroup-lbl {{ font-size:9.5px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
  color:var(--gold); margin:16px 0 -2px; }}
.bio-statgroup-lbl.intl {{ color:var(--blue); }}
.bio-text {{ font-family:Georgia,serif; font-size:13.5px; line-height:1.62; color:#333; margin:18px 0 0;
  max-height:230px; overflow-y:auto; }}
.bio-credit {{ font-size:10px; color:#aab2bb; margin-top:16px; border-top:1px solid #eceef0; padding-top:10px; }}

/* ── PLAYER EXPLORER ── */
.exp-controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:16px; }}
.exp-search {{
  flex:1; min-width:220px; padding:10px 14px; font-size:13px; font-family:inherit;
  border:1px solid var(--border); border-radius:10px; background:var(--card); color:var(--txt);
  outline:none; transition:border-color .15s, box-shadow .15s;
}}
.exp-search:focus {{ border-color:var(--gold); box-shadow:0 0 0 3px rgba(217,119,6,.12); }}
.exp-chips {{ display:flex; gap:6px; }}
.exp-chip {{
  padding:8px 13px; font-size:11px; font-weight:700; font-family:'Space Grotesk',sans-serif;
  border:1px solid var(--border); border-radius:99px; background:var(--card); color:var(--txt2);
  cursor:pointer; transition:all .15s; letter-spacing:.02em;
}}
.exp-chip:hover {{ border-color:var(--border2); color:var(--txt); }}
.exp-chip.active {{ background:var(--gold); border-color:var(--gold); color:#fff; }}
.exp-count {{ font-size:11px; color:var(--txt3); margin-left:auto; font-weight:600; }}
.exp-wrap {{ border:1px solid var(--border); border-radius:var(--radius); background:var(--card);
  box-shadow:var(--shadow2); overflow:clip; }}
table.exp-table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
.exp-table thead th {{
  position:sticky; top:50px; z-index:20; background:var(--card2); color:var(--txt2);
  font-size:9.5px; font-weight:700; text-transform:uppercase; letter-spacing:.04em;
  padding:11px 8px; text-align:center; white-space:nowrap; cursor:pointer; user-select:none;
  border-bottom:2px solid var(--border2); transition:color .15s;
}}
.exp-table thead th:hover {{ color:var(--gold); }}
.exp-table thead th.active {{ color:var(--gold); }}
.exp-table thead th.exp-l {{ text-align:left; }}
.exp-table tbody td {{ padding:8px; text-align:center; border-bottom:1px solid var(--border);
  white-space:nowrap; font-variant-numeric:tabular-nums; }}
.exp-table tbody tr {{ cursor:pointer; transition:background .12s; }}
.exp-table tbody tr:hover td {{ background:rgba(217,119,6,.06); }}
.exp-player {{ display:flex; align-items:center; gap:10px; text-align:left !important; min-width:170px; }}
.exp-ph {{ width:30px; height:30px; border-radius:50%; object-fit:cover; flex-shrink:0;
  background:#eef0f3; border:1px solid var(--border); }}
.exp-pname {{ font-weight:600; color:var(--txt); }}
.exp-nat {{ text-align:left !important; }}
.exp-nat span {{ display:inline-flex; align-items:center; gap:7px; color:var(--txt2); font-weight:500; }}
.exp-nat .flag-img {{ width:18px; height:12px; }}
.exp-club {{ text-align:left !important; color:var(--txt2); max-width:150px; overflow:hidden;
  text-overflow:ellipsis; }}
.exp-pos {{ font-size:10px; font-weight:700; color:var(--txt3); }}
.exp-form {{ font-weight:700; border-radius:5px; color:#fff; padding:3px 0; display:inline-block; min-width:34px; }}
.exp-na {{ color:var(--txt4); }}
.exp-foot {{ padding:11px 15px; font-size:11px; color:var(--txt3); background:var(--card); border-top:1px solid var(--border); }}

/* ── RESPONSIVE ── */
@media (max-width:768px) {{
  .hero {{ padding:40px 16px 32px; }}
  .main {{ padding:24px 12px 48px; }}
  .card {{ padding:16px; }}
  .champ-grid {{ grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:10px; }}
  .team-header {{ flex-direction:column; align-items:flex-start; gap:16px; }}
  .final-hero {{ padding:28px 20px; }}
  .final-champ-name {{ font-size:22px; }}
  .award-grid {{ grid-template-columns:1fr 1fr; }}
  .bracket-matches-grid.sf-grid {{ grid-template-columns:1fr; max-width:100%; }}
}}
@media (max-width:480px) {{
  .hero h1 {{ font-size:28px; }}
  .award-grid {{ grid-template-columns:1fr; }}
  .hstat-val {{ font-size:18px; }}
  .champ-grid {{ grid-template-columns:1fr 1fr; gap:8px; }}
  .champ-card {{ padding:16px 12px 14px; }}
}}

/* ── ANIMATIONS (respect prefers-reduced-motion) ── */
@keyframes fadeUp {{
  from {{ opacity:0; transform:translateY(16px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes countUp {{
  from {{ opacity:0; }}
  to   {{ opacity:1; }}
}}
.anim-fade-up {{
  animation:fadeUp .4s cubic-bezier(.22,1,.36,1) both;
}}

/* ════════════════ PREMIUM HOVER LAYER ════════════════ */
/* Generic content cards — elevate + faint gold border. No transform, so sticky
   table headers living inside cards never break. */
.card {{ transition:border-color .3s var(--ease), box-shadow .3s var(--ease); }}
.card:hover {{
  border-color:rgba(217,119,6,.22);
  box-shadow:0 14px 40px rgba(20,33,58,.11), 0 0 0 1px rgba(217,119,6,.06);
}}

/* Hero stat tiles — lift + glowing value */
.hstat {{ transition:transform .3s var(--ease); border-radius:10px; }}
.hstat:hover {{ transform:translateY(-3px); }}
.hstat:hover .hstat-val {{ text-shadow:0 0 18px var(--gold-glow); }}

/* Nav tabs — subtle rise */
.nav-tab {{ transition:color .2s, border-color .2s, transform .2s var(--ease); }}
.nav-tab:hover {{ transform:translateY(-1px); }}

/* Photos — gentle zoom when their card / row is hovered */
.pphoto {{ transition:transform .35s var(--ease), box-shadow .35s var(--ease); }}
.award-card:hover .pphoto,
.player-row:hover .pphoto,
.exp-table tbody tr:hover .pphoto {{ transform:scale(1.09); }}
.award-card:hover .award-avatar .pphoto {{ box-shadow:0 6px 18px rgba(217,119,6,.30); }}

/* Flags — micro-scale inside hovered rows / cards */
.flag-img {{ transition:transform .3s var(--ease); }}
.player-row:hover .flag-img,
.award-card:hover .flag-img,
.champ-card:hover .flag-img,
.bm-card:hover .flag-img,
.group-team-row:hover .flag-img {{ transform:scale(1.08); }}

/* List rows — a gold left-accent grows in */
.player-row {{ position:relative; }}
.player-row::before {{
  content:''; position:absolute; left:0; top:50%; transform:translateY(-50%);
  width:3px; height:0; border-radius:2px; background:var(--gold); opacity:.9;
  transition:height .25s var(--ease);
}}
.player-row:hover::before {{ height:60%; }}

/* Explorer + data-table rows — gold inset accent on the first cell */
.exp-table tbody tr td:first-child,
table.data-table tbody tr td:first-child {{ transition:box-shadow .2s var(--ease); }}
.exp-table tbody tr:hover td:first-child,
table.data-table tbody tr:hover td:first-child {{ box-shadow:inset 3px 0 0 var(--gold); }}

/* Filter chips — lift + shadow */
.exp-chip {{ transition:all .18s var(--ease); }}
.exp-chip:hover {{ transform:translateY(-2px); box-shadow:0 6px 16px rgba(20,33,58,.12); }}

/* Select dropdown — hover affordance (previously focus-only) */
select {{ transition:border-color .2s var(--ease), box-shadow .2s var(--ease); }}
select:hover {{ border-color:var(--border2); box-shadow:0 4px 14px rgba(20,33,58,.08); }}

/* Squad player + highlight cards — lift */
.pg-player {{ transition:transform .25s var(--ease); }}
.pg-player:hover {{ transform:translateY(-3px); }}
.pg-highlight {{ transition:transform .3s var(--ease), box-shadow .3s var(--ease), border-color .3s var(--ease); }}
.pg-highlight:hover {{ transform:translateY(-3px); box-shadow:var(--lift); }}

/* Sheen sweep across the award cards on hover (overflow:hidden clips it) */
.award-card::after {{
  content:''; position:absolute; top:0; left:-65%; width:45%; height:100%;
  background:linear-gradient(100deg, transparent, rgba(255,255,255,.5), transparent);
  transform:skewX(-18deg); pointer-events:none; opacity:0;
  transition:left .65s var(--ease), opacity .2s var(--ease);
}}
.award-card:hover::after {{ left:135%; opacity:1; }}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration:.01ms !important;
    transition-duration:.01ms !important;
  }}
}}
</style>
</head>
<body>

<!-- ── HERO ── -->
<div class="hero">
  <div class="hero-mesh"></div>
  <div class="hero-grid"></div>
  <div class="hero-content">
    <div class="hero-logo">
      <img src="Data/icons/tournaments_fifa-world-cup-2026--white.football-logos.cc.svg"
           alt="FIFA World Cup 2026 official emblem" width="227" height="351" loading="eager">
    </div>
    <h1>FIFA World Cup 2026</h1>
    <p class="hero-sub">A data-driven forecast built on <b>ELO ratings</b>, <b>StatsBomb xG</b>, <b>FIFA rankings</b> &amp; historical tournament form &mdash; across 10,000 simulations</p>
    <div class="hero-stats">
      <div class="hstat">
        <div class="hstat-val" id="stat-teams">48</div>
        <div class="hstat-lbl">Teams</div>
      </div>
      <div class="hstat">
        <div class="hstat-val" id="stat-sims">10K</div>
        <div class="hstat-lbl">Simulations</div>
      </div>
      <div class="hstat">
        <div class="hstat-val" id="stat-sources">5</div>
        <div class="hstat-lbl">Data Sources</div>
      </div>
      <div class="hstat">
        <div class="hstat-val" id="stat-matches">199</div>
        <div class="hstat-lbl">Matches Analysed</div>
      </div>
    </div>
  </div>
</div>

<!-- ── NAV ── -->
<div class="nav-wrap">
  <nav class="nav" role="tablist">
    <button class="nav-tab active" onclick="showTab('overview',this)" role="tab">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
      Overview
    </button>
    <button class="nav-tab" onclick="showTab('projections',this)" role="tab">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="3" x2="3" y2="21"/><line x1="3" y1="21" x2="21" y2="21"/><rect x="7" y="12" width="3" height="6"/><rect x="12" y="8" width="3" height="10"/><rect x="17" y="4" width="3" height="14"/></svg>
      Projections
    </button>
    <button class="nav-tab" onclick="showTab('groups',this)" role="tab">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
      Groups
    </button>
    <button class="nav-tab" onclick="showTab('bracket',this)" role="tab">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6H6a2 2 0 00-2 2v8a2 2 0 002 2h2M16 6h2a2 2 0 012 2v8a2 2 0 01-2 2h-2M8 12h8"/></svg>
      Knockout
    </button>
    <button class="nav-tab" onclick="showTab('explorer',this)" role="tab">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
      Players
    </button>
    <button class="nav-tab" onclick="showTab('players',this)" role="tab">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="6"/><path d="M15.477 12.89L17 22l-5-3-5 3 1.523-9.11"/></svg>
      Awards
    </button>
    <button class="nav-tab" onclick="showTab('team',this)" role="tab">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      My Team
    </button>
    <div class="nav-credit">
      <a class="nc-by" href="https://ammarshahid.netlify.app" target="_blank" rel="noopener">Built by <b>Ammar Shahid</b></a>
      <a class="nc-link" href="https://ammarshahid.netlify.app" target="_blank" rel="noopener" title="Website — ammarshahid.netlify.app" aria-label="Website">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
      </a>
      <a class="nc-link" href="https://github.com/Ammar8065" target="_blank" rel="noopener" title="GitHub — Ammar8065" aria-label="GitHub">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5C5.37.5 0 5.78 0 12.29c0 5.21 3.44 9.63 8.2 11.19.6.11.82-.25.82-.56v-2.2c-3.34.71-4.04-1.59-4.04-1.59-.55-1.37-1.34-1.74-1.34-1.74-1.09-.73.08-.72.08-.72 1.2.08 1.84 1.22 1.84 1.22 1.07 1.8 2.81 1.28 3.5.98.11-.76.42-1.28.76-1.58-2.67-.3-5.47-1.31-5.47-5.84 0-1.29.47-2.35 1.23-3.18-.12-.3-.53-1.51.12-3.15 0 0 1.01-.32 3.3 1.21a11.5 11.5 0 0 1 6 0c2.29-1.53 3.3-1.21 3.3-1.21.65 1.64.24 2.85.12 3.15.77.83 1.23 1.89 1.23 3.18 0 4.54-2.81 5.54-5.49 5.83.43.37.81 1.1.81 2.22v3.29c0 .31.22.68.83.56A12.05 12.05 0 0 0 24 12.29C24 5.78 18.63.5 12 .5z"/></svg>
      </a>
      <a class="nc-link" href="https://www.linkedin.com/in/ammar-shahid-087520263/" target="_blank" rel="noopener" title="LinkedIn — Ammar Shahid" aria-label="LinkedIn">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.42v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 2.06 0 0 1 0 4.13zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.22.79 24 1.77 24h20.45c.98 0 1.78-.78 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z"/></svg>
      </a>
    </div>
  </nav>
</div>

<div class="main">

<!-- ════════════════ OVERVIEW ════════════════ -->
<div id="panel-overview" class="panel active">
  <div class="sec-header">
    <div class="sec-eyebrow">Championship Predictions</div>
    <div class="sec-title">Who Wins the 2026 World Cup?</div>
    <div class="sec-sub">Ranked by championship probability across 10,000 Monte Carlo simulations &mdash; hover cards to explore</div>
  </div>

  <div id="champ-cards" class="champ-grid mb-8"></div>

  <div class="card mb-6">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:20px;flex-wrap:wrap">
      <div>
        <div class="sec-eyebrow">All 48 Teams</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:17px;font-weight:700;letter-spacing:-.01em">Championship Probability Rankings</div>
      </div>
      <div style="font-size:11px;color:var(--txt3);max-width:200px;text-align:right;line-height:1.5">Top 16 shown &mdash; see treemap for all 48</div>
    </div>
    <div id="chart-bar" style="height:520px"></div>
  </div>

  <div class="card mb-6">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:16px;flex-wrap:wrap">
      <div>
        <div class="sec-eyebrow">Top 12 Contenders</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:17px;font-weight:700;letter-spacing:-.01em">Deep-Run Probability</div>
      </div>
      <div style="font-size:11px;color:var(--txt3);max-width:220px;text-align:right;line-height:1.5">Chance of reaching each knockout stage &mdash; heat-graded</div>
    </div>
    <div class="gh-wrap" style="padding:0"><div id="overview-heat"></div></div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:6px">Team Quality</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:16px">Attack vs. Defense Quadrant</div>
      <div id="chart-scatter" style="height:340px;position:relative"></div>
    </div>
    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:6px">All 48 Teams</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:16px">Win Probability Map</div>
      <div id="chart-treemap" style="height:340px"></div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px">
    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:6px">Where the Cup Could Go</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:16px">Title Odds by Confederation</div>
      <div id="chart-confed" style="height:330px"></div>
    </div>
    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:6px">Current-Season Club xG</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:16px">Golden Boot Race</div>
      <div id="chart-goldenboot" style="height:330px"></div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px">
    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:6px">Squad Firepower</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:4px">Projected Attacking Output</div>
      <div style="font-size:11px;color:var(--txt3);margin-bottom:12px">Total current-season club xG across each squad (big-5 league players)</div>
      <div id="chart-firepower" style="height:330px"></div>
    </div>
    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:6px">Top 8 Contenders</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:4px">Stage-Survival Curves</div>
      <div style="font-size:11px;color:var(--txt3);margin-bottom:12px">How each favourite&rsquo;s probability decays from the group stage to lifting the trophy</div>
      <div id="chart-survival" style="height:330px"></div>
    </div>
  </div>
</div>

<!-- ════════════════ PROJECTIONS ════════════════ -->
<div id="panel-projections" class="panel">
  <div class="sec-header">
    <div class="sec-eyebrow">Tournament Projections</div>
    <div class="sec-title">Full Probability Table</div>
    <div class="sec-sub">Every team&rsquo;s chance at each stage across 10,000 simulations. Cells are colour-graded by probability &mdash; click any column header to rank by it.</div>
  </div>
  <div class="proj-wrap">
    <div id="proj-table"></div>
  </div>
</div>

<!-- ════════════════ GROUPS ════════════════ -->
<div id="panel-groups" class="panel">
  <div class="sec-header">
    <div class="sec-eyebrow">Group Stage</div>
    <div class="sec-title">Group Stage Predictions</div>
    <div class="sec-sub">Colour-graded finish &amp; advancement probabilities for all 12 groups &mdash; top 2 advance automatically, plus 8 best 3rd-place finishers. Greener = more likely.</div>
  </div>
  <div class="groups-grid" id="groups-container"></div>
</div>

<!-- ════════════════ BRACKET ════════════════ -->
<div id="panel-bracket" class="panel">
  <div class="sec-header">
    <div class="sec-eyebrow">Knockout Stage</div>
    <div class="sec-title">Predicted Knockout Bracket</div>
    <div class="sec-sub">Every slot filled by advancing the model&rsquo;s favourite from the predicted group standings &mdash; percentages show the winner&rsquo;s chance in that specific matchup</div>
  </div>

  <div class="bracket-scroll-hint">← scroll horizontally to explore the full bracket →</div>
  <div class="bracket-scroll">
    <div id="bracket-tree" class="bracket-tree"></div>
  </div>
</div>

<!-- ════════════════ PLAYER EXPLORER ════════════════ -->
<div id="panel-explorer" class="panel">
  <div class="sec-header">
    <div class="sec-eyebrow">Squad Database</div>
    <div class="sec-title">Player Explorer</div>
    <div class="sec-sub">All 1,248 players from the 48 squads. Current-season club stats (2025-26, Understat) &mdash; goals, assists, xG, xA &amp; a form rating &mdash; alongside international tournament xG. Search, filter by position, click any column to sort, click a row for the full profile.</div>
  </div>
  <div class="exp-controls">
    <input id="exp-search" class="exp-search" type="text" placeholder="Search player, nation or club…" oninput="renderExplorer()" aria-label="Search players">
    <div class="exp-chips" id="exp-chips"></div>
    <span class="exp-count" id="exp-count"></span>
  </div>
  <div class="exp-wrap"><div id="exp-table"></div></div>
</div>

<!-- ════════════════ AWARDS ════════════════ -->
<div id="panel-players" class="panel">
  <div class="sec-header">
    <div class="sec-eyebrow">Individual Awards</div>
    <div class="sec-title">Player &amp; Award Predictions</div>
    <div class="sec-sub">Ranked by <b>current-season club xG</b> (2025-26, Understat) for squad players in Europe&rsquo;s big-5 leagues, with historical tournament xG as fallback &mdash; weighted by team title probability</div>
  </div>

  <div class="award-grid mb-6" id="award-cards"></div>

  <div class="card mb-6">
    <div class="sec-eyebrow" style="margin-bottom:6px">Recent club form &times; recent tournament form &times; projected run</div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:4px">Golden Ball Race &mdash; Predicted Best Player</div>
    <div style="font-size:11.5px;color:var(--txt2);margin-bottom:14px;line-height:1.5">
      A forecast, not a goal tally. Blends each player&rsquo;s <b>2025-26 club goal involvements</b> (xG+xA &amp; actual G+A per 90),
      their <b>recent national-team output</b>, and how deep their team is projected to go &mdash; since the award almost always
      follows a deep run. <span style="color:rgba(180,130,0,1);font-weight:600">Win share</span> shown on the right.
    </div>
    <div id="goldenball-list"></div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:6px">Expected Goals</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:16px">Top Predicted Scorers</div>
      <div id="chart-scorers" style="height:420px"></div>
    </div>
    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:6px">Season xG &times; projected deep run</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:16px">Player of the Tournament Race</div>
      <div id="player-list"></div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:6px">Under 23 at Tournament</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:16px">Young Player of the Tournament</div>
      <div id="ypoty-list"></div>
    </div>
    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:6px">Defensive Ranking</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:16px">Best Defensive Teams</div>
      <div id="best-def-list"></div>
    </div>
  </div>
</div>

<!-- ════════════════ TEAM DEEP DIVE ════════════════ -->
<div id="panel-team" class="panel">
  <div class="sec-header">
    <div class="sec-eyebrow">Team Analysis</div>
    <div class="sec-title">My Team</div>
    <div class="sec-sub">Deep dive into any team's predicted performance and tournament journey</div>
  </div>

  <div class="team-select-wrap">
    <select id="team-select" onchange="renderTeam(this.value)" aria-label="Select a team"></select>
  </div>

  <div id="team-content"></div>
</div>

</div><!-- /main -->

<!-- ── PLAYER BIO POPUP ── -->
<div id="bio-modal" class="bio-modal" onclick="if(event.target===this)closeBio()">
  <div class="bio-card" id="bio-card"></div>
</div>

<script>
// ── DATA ──────────────────────────────────────────────────────────────────────
const TEAMS    = {teams_js};
const GROUPS   = {groups_js};
const BRACKET  = {bracket_js};
const BRACKETFULL = {bracketfull_js};
const SQUADS   = {squads_js};
const SLOTS    = {slots_js};
const SCORERS  = {scorers_js};
const POTY     = {poty_js};
const YPOTY    = {ypoty_js};
const BEST_DEF = {bestdef_js};
const BEST_ATT = {bestatt_js};
const GOLDEN_BALL = {goldenball_js};
const FLAGS    = {flags_js};
const FLAGCODE = {flagiso_js};
const YOUNG    = {young_js};
const PLAYERSDB = {playersdb_js};

// Real SVG flag via flagcdn — reliable on all platforms (emoji flags break on Windows Chrome)
function flagImg(team, h) {{
  const c = FLAGCODE[team];
  const px = h || 20;
  if (!c) return '<span class="flag-img" style="display:inline-block;width:' + Math.round(px*1.5) + 'px;height:' + px + 'px;background:rgba(20,33,58,.07);border-radius:3px"></span>';
  return '<img class="flag-img" src="https://flagcdn.com/' + c + '.svg" alt="' + team + '" loading="lazy" style="height:' + px + 'px;width:' + Math.round(px*1.5) + 'px">';
}}

const PLOTLY_LAYOUT = {{
  paper_bgcolor:'rgba(0,0,0,0)',
  plot_bgcolor:'rgba(0,0,0,0)',
  font:{{ family:"Space Grotesk, Inter, sans-serif", color:'#64748B', size:11 }},
  margin:{{ t:10, l:10, r:10, b:10 }},
  colorway:['#F59E0B','#3B82F6','#10B981','#EF4444','#8B5CF6','#EC4899'],
  hoverlabel:{{ bgcolor:'#161c2e', bordercolor:'rgba(255,255,255,.1)', font:{{ family:'Space Grotesk, Inter', color:'#F1F5F9', size:12 }} }},
}};
const PLOTLY_CFG = {{ displayModeBar:false, responsive:true }};

function pct(v)  {{ return (v*100).toFixed(1) + '%'; }}
function pct1(v) {{ return (v*100).toFixed(0) + '%'; }}

// ── ANIMATE COUNT-UP ──────────────────────────────────────────────────────────
function animateCount(el, target, suffix, duration) {{
  if (!el) return;
  const start = performance.now();
  const num = parseFloat(target.replace(/[^0-9.]/g,''));
  function step(now) {{
    const p = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1-p, 3);
    const val = Math.round(num * ease);
    el.textContent = val + suffix;
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = target;
  }}
  requestAnimationFrame(step);
}}

// ── STAGGER ANIMATION ─────────────────────────────────────────────────────────
function staggerIn(selector, delay) {{
  const els = document.querySelectorAll(selector);
  els.forEach((el, i) => {{
    el.style.opacity = '0';
    el.style.transform = 'translateY(12px)';
    setTimeout(() => {{
      el.style.transition = 'opacity .35s ease, transform .35s ease';
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }}, (delay||0) + i * 40);
  }});
}}

// ── TABS ──────────────────────────────────────────────────────────────────────
function showTab(name, btn) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'overview')    renderOverview();
  if (name === 'projections') renderProjections();
  if (name === 'groups')      renderGroups();
  if (name === 'bracket')     renderBracket();
  if (name === 'explorer')    renderExplorer();
  if (name === 'players')     renderPlayers();
  if (name === 'team')        initTeam();
  // Resize Plotly charts if they exist in this panel (handles 0-width render bug)
  setTimeout(() => {{
    const panel = document.getElementById('panel-' + name);
    const charts = panel ? panel.querySelectorAll('[id^="chart-"],[id="radar-chart"]') : [];
    charts.forEach(el => {{ try {{ Plotly.Plots.resize(el); }} catch(e) {{}} }});
  }}, 60);
}}
// Robustly switch tabs by name (nav order can change without breaking callers).
function gotoTab(name) {{
  const btn = [...document.querySelectorAll('.nav-tab')]
    .find(b => (b.getAttribute('onclick') || '').includes(`'${{name}}'`));
  if (btn) btn.click();
}}

// ── PLAYER BIO POPUP (Guardian-style) ─────────────────────────────────────────
const PLAYER_INDEX = {{}};
PLAYERSDB.forEach((p, i) => {{ PLAYER_INDEX[p.team + '|' + p.name] = i; }});
// grey → green shade for probability/form dots
function bioDot(x) {{
  x = Math.max(0, Math.min(1, x));
  const L = (a, b) => Math.round(a + (b - a) * x);
  return `rgb(${{L(233,30)}},${{L(236,140)}},${{L(240,80)}})`;
}}
function openBioByKey(team, name) {{ const i = PLAYER_INDEX[team + '|' + name]; if (i != null) openBio(i); }}
function openBio(idx) {{
  const p = PLAYERSDB[idx];
  if (!p) return;
  const t = TEAMS.find(x => x.team === p.team) || {{}};
  const stages = [['Group', 1], ['R32', t.p_qualify || 0], ['R16', t.p_round_of_16 || 0],
                  ['QF', t.p_quarter_final || 0], ['SF', t.p_semi_final || 0], ['F', t.p_final || 0]];
  const dots = stages.map(([lbl, pr]) =>
    `<div class="bio-dot"><div class="d" style="background:${{bioDot(pr)}}"></div><div class="dl">${{lbl}}</div></div>`).join('');
  const role = (p.special && p.special.trim()) ? p.special : (p.is_star ? 'Star player' : (p.position || ''));
  const clubStats = p.has_club ? `
    <div class="bio-statgroup-lbl">Club &middot; 2025-26${{p.league ? ' &middot; ' + p.league : ''}}</div>
    <div class="bio-stats">
      <div class="bio-stat"><div class="sv">${{p.g}}</div><div class="sl">Goals</div></div>
      <div class="bio-stat"><div class="sv">${{p.a}}</div><div class="sl">Assists</div></div>
      <div class="bio-stat"><div class="sv">${{(+p.xg).toFixed(1)}}</div><div class="sl">xG</div></div>
      <div class="bio-stat"><div class="sv">${{(+p.xa).toFixed(1)}}</div><div class="sl">xA</div></div>
      <div class="bio-stat"><div class="sv">${{p.matches}}</div><div class="sl">Matches</div></div>
    </div>` : '';
  const intlStats = p.has_intl ? `
    <div class="bio-statgroup-lbl intl">International &middot; recent tournaments</div>
    <div class="bio-stats">
      <div class="bio-stat"><div class="sv">${{p.intl_goals}}</div><div class="sl">Goals</div></div>
      <div class="bio-stat"><div class="sv">${{(+p.intl_xg).toFixed(1)}}</div><div class="sl">xG</div></div>
      <div class="bio-stat"><div class="sv">${{p.intl_matches}}</div><div class="sl">Matches</div></div>
    </div>` : '';
  const capLine = (p.caps != null)
    ? `${{p.caps}} caps` + (p.intl_career_goals != null ? ` &middot; ${{p.intl_career_goals}} intl goals` : '') : '';
  document.getElementById('bio-card').innerHTML = `
    <button class="bio-close" onclick="closeBio()" aria-label="Close">&times;</button>
    <div class="bio-top">
      ${{p.photo ? `<img class="bio-photo" src="${{p.photo}}" alt="${{p.name}}">` : ''}}
      <div style="flex:1;min-width:0">
        <div class="bio-name">${{p.name}}</div>
        ${{role ? `<div class="bio-role">${{role}}</div>` : ''}}
        <div class="bio-flagline">${{flagImg(p.team, 14)}}<span>${{p.team}}</span>${{p.number != null ? ' &middot; #' + p.number : ''}}</div>
      </div>
    </div>
    <div class="bio-kv-row">
      <div class="bio-kv"><div class="k">Club</div><div class="v">${{p.club || '—'}}</div></div>
      <div class="bio-kv"><div class="k">Age</div><div class="v">${{p.age != null ? p.age : '—'}}</div></div>
      <div class="bio-kv"><div class="k">Position</div><div class="v">${{p.position || '—'}}</div></div>
    </div>
    <div class="bio-rating-lbl">Team&rsquo;s run &middot; chance of reaching each stage</div>
    <div class="bio-dots">${{dots}}</div>
    ${{clubStats}}${{intlStats}}
    ${{p.bio ? `<div class="bio-text">${{p.bio}}</div>` : ''}}
    <div class="bio-credit">${{capLine ? capLine + ' &middot; ' : ''}}Profile &amp; photo: The Guardian player guide</div>`;
  document.getElementById('bio-modal').classList.add('open');
  document.body.style.overflow = 'hidden';
}}
function closeBio() {{
  document.getElementById('bio-modal').classList.remove('open');
  document.body.style.overflow = '';
}}
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeBio(); }});

// ── PLAYER EXPLORER ───────────────────────────────────────────────────────────
let expSort = {{ key:'xg', dir:-1 }};
let expPos = 'ALL';
const EXP_TEXT = new Set(['name', 'team', 'club', 'pos_group']);
const EXP_COLS = [
  {{ label:'Player',  cls:'exp-l', sort:'name' }},
  {{ label:'Nation',  cls:'exp-l', sort:'team' }},
  {{ label:'Club',    cls:'exp-l', sort:'club' }},
  {{ label:'Pos',  sort:'pos_group' }},
  {{ label:'M',    sort:'matches' }},
  {{ label:'G',    sort:'g' }},
  {{ label:'A',    sort:'a' }},
  {{ label:'xG',   sort:'xg' }},
  {{ label:'xA',   sort:'xa' }},
  {{ label:'Form', sort:'form' }},
  {{ label:'Intl xG', sort:'intl_xg' }},
];
function expSortBy(key) {{
  if (expSort.key === key) expSort.dir *= -1;
  else expSort = {{ key, dir: EXP_TEXT.has(key) ? 1 : -1 }};
  renderExplorer();
}}
function expSetPos(pos) {{ expPos = pos; renderExplorer(); }}
function renderExplorer() {{
  const chips = document.getElementById('exp-chips');
  if (chips && !chips.dataset.init) {{
    chips.innerHTML = [['ALL','All'],['GK','GK'],['DEF','DEF'],['MID','MID'],['FWD','FWD']]
      .map(([v, l]) => `<button class="exp-chip" data-pos="${{v}}" onclick="expSetPos('${{v}}')">${{l}}</button>`).join('');
    chips.dataset.init = '1';
  }}
  document.querySelectorAll('.exp-chip').forEach(c => c.classList.toggle('active', c.dataset.pos === expPos));

  const q = (document.getElementById('exp-search').value || '').toLowerCase().trim();
  let rows = PLAYERSDB.filter(p => {{
    if (expPos !== 'ALL' && p.pos_group !== expPos) return false;
    if (q && !(p.name + ' ' + p.team + ' ' + p.club).toLowerCase().includes(q)) return false;
    return true;
  }});
  const k = expSort.key, dir = expSort.dir, isText = EXP_TEXT.has(k);
  rows.sort((a, b) => {{
    let av = a[k], bv = b[k];
    if (isText) return dir * String(av || '').localeCompare(String(bv || ''));
    av = (av == null) ? -1 : av; bv = (bv == null) ? -1 : bv;
    return dir * (av - bv);
  }});
  const arrow = s => expSort.key === s ? (expSort.dir < 0 ? ' ↓' : ' ↑') : '';
  const head = `<tr>` + EXP_COLS.map(c =>
    `<th class="${{c.cls || ''}} ${{expSort.key === c.sort ? 'active' : ''}}" onclick="expSortBy('${{c.sort}}')">${{c.label}}${{arrow(c.sort)}}</th>`).join('') + `</tr>`;
  const num = (v) => (v == null || Number.isNaN(+v)) ? '<span class="exp-na">–</span>' : (+v).toFixed(1);
  const int = (v) => (v == null) ? '<span class="exp-na">–</span>' : v;
  const body = rows.map(p => {{
    const idx = PLAYER_INDEX[p.team + '|' + p.name];
    const fx = p.form == null ? null : Math.min(1, p.form / 10);
    const formCell = p.form == null ? '<span class="exp-na">–</span>'
      : `<span class="exp-form" style="background:${{bioDot(fx)}};color:${{fx > 0.45 ? '#fff' : '#14213a'}}">${{(+p.form).toFixed(1)}}</span>`;
    return `<tr onclick="openBio(${{idx}})">
      <td class="exp-player">${{p.photo ? `<img class="exp-ph" src="${{p.photo}}" loading="lazy" alt="">` : '<span class="exp-ph"></span>'}}<span class="exp-pname">${{p.name}}</span></td>
      <td class="exp-nat"><span>${{flagImg(p.team, 18)}}${{p.team}}</span></td>
      <td class="exp-club">${{p.club || '–'}}</td>
      <td><span class="exp-pos">${{p.pos_group || '–'}}</span></td>
      <td>${{int(p.matches)}}</td>
      <td>${{int(p.g)}}</td>
      <td>${{int(p.a)}}</td>
      <td>${{num(p.xg)}}</td>
      <td>${{num(p.xa)}}</td>
      <td>${{formCell}}</td>
      <td>${{num(p.intl_xg)}}</td>
    </tr>`;
  }}).join('');
  document.getElementById('exp-count').textContent = rows.length + ' players';
  document.getElementById('exp-table').innerHTML =
    `<table class="exp-table"><thead>${{head}}</thead><tbody>${{body}}</tbody></table>` +
    `<div class="exp-foot">Club stats: current season (2025-26, Understat, big-5 leagues) &middot; Intl xG: recent international tournaments (StatsBomb) &middot; Form = (xG+xA per 90)×10, capped at 10 &middot; <b>click a row</b> for the full profile.</div>`;
}}

// ── OVERVIEW EXTRA VIZ ─────────────────────────────────────────────────────────
function drawConfedChart() {{
  const agg = {{}};
  TEAMS.forEach(t => {{ const c = t.confederation || '—'; agg[c] = (agg[c] || 0) + (t.p_champion || 0); }});
  const order = ['UEFA','CONMEBOL','CONCACAF','CAF','AFC','OFC'].filter(c => agg[c]);
  const colors = {{UEFA:'#2563EB',CONMEBOL:'#059669',CONCACAF:'#D9770B',CAF:'#DC2626',AFC:'#7C3AED',OFC:'#0891B2'}};
  Plotly.newPlot('chart-confed', [{{
    type:'pie', hole:.58, sort:false,
    labels: order, values: order.map(c => +(agg[c] * 100).toFixed(1)),
    marker:{{ colors: order.map(c => colors[c]), line:{{ color:'#fff', width:2 }} }},
    textinfo:'label+percent', textfont:{{ family:'Space Grotesk', size:11 }},
    hovertemplate:'<b>%{{label}}</b><br>Combined title odds: %{{value:.1f}}%<extra></extra>',
  }}], {{ ...PLOTLY_LAYOUT, height:330, margin:{{ t:10, l:10, r:10, b:10 }}, showlegend:false }}, PLOTLY_CFG);
}}
function drawGoldenBoot() {{
  const top = SCORERS.slice(0, 12).slice().reverse();
  Plotly.newPlot('chart-goldenboot', [{{
    type:'bar', orientation:'h',
    x: top.map(p => +(+p.xg).toFixed(1)), y: top.map(p => p.player),
    text: top.map(p => (+p.xg).toFixed(1) + ' xG'),
    textposition:'outside', textfont:{{ family:'Space Grotesk', size:10, color:'#475569' }},
    marker:{{ color: top.map((p, i) => `rgba(217,119,6,${{(0.45 + 0.55 * i / top.length).toFixed(2)}})`) }},
    hovertemplate:'<b>%{{y}}</b><br>%{{x}} xG<extra></extra>', cliponaxis:false,
  }}], {{ ...PLOTLY_LAYOUT, height:330, margin:{{ t:10, l:128, r:48, b:24 }},
    xaxis:{{ gridcolor:'rgba(20,33,58,.07)', color:'#4B5872', tickfont:{{ size:9 }}, zeroline:false, showline:false }},
    yaxis:{{ tickfont:{{ family:'Space Grotesk', size:10, color:'#475569' }}, showline:false, gridcolor:'rgba(0,0,0,0)' }},
    bargap:.34,
  }}, PLOTLY_CFG);
}}
function drawFirepower() {{
  const top = [...TEAMS].sort((a, b) => (b.firepower || 0) - (a.firepower || 0)).slice(0, 14).reverse();
  const mx = Math.max(...top.map(t => t.firepower || 0)) || 1;
  Plotly.newPlot('chart-firepower', [{{
    type:'bar', orientation:'h',
    x: top.map(t => +(t.firepower || 0).toFixed(0)), y: top.map(t => t.team),
    text: top.map(t => (t.firepower || 0).toFixed(0)),
    textposition:'outside', textfont:{{ family:'Space Grotesk', size:10, color:'#475569' }},
    marker:{{ color: top.map(t => `rgba(5,150,105,${{(0.4 + 0.55 * (t.firepower || 0) / mx).toFixed(2)}})`) }},
    hovertemplate:'<b>%{{y}}</b><br>Squad club xG: %{{x}}<extra></extra>', cliponaxis:false,
  }}], {{ ...PLOTLY_LAYOUT, height:330, margin:{{ t:10, l:120, r:40, b:24 }},
    xaxis:{{ gridcolor:'rgba(20,33,58,.07)', color:'#4B5872', tickfont:{{ size:9 }}, zeroline:false, showline:false }},
    yaxis:{{ tickfont:{{ family:'Space Grotesk', size:10, color:'#475569' }}, showline:false, gridcolor:'rgba(0,0,0,0)' }},
    bargap:.34,
  }}, PLOTLY_CFG);
}}
function drawSurvival() {{
  const top = [...TEAMS].sort((a, b) => b.p_champion - a.p_champion).slice(0, 8);
  const stages = ['Group','R32','R16','QF','SF','Final','Win'];
  const keys = [null,'p_qualify','p_round_of_16','p_quarter_final','p_semi_final','p_final','p_champion'];
  const palette = ['#C2740B','#2563EB','#059669','#DC2626','#7C3AED','#0891B2','#D9770B','#475569'];
  const traces = top.map((t, i) => ({{
    type:'scatter', mode:'lines+markers', name:t.team,
    x: stages, y: keys.map(k => k ? +((t[k] || 0) * 100).toFixed(1) : 100),
    line:{{ color:palette[i % palette.length], width:2 }}, marker:{{ size:5 }},
    hovertemplate:'<b>' + t.team + '</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>',
  }}));
  Plotly.newPlot('chart-survival', traces, {{ ...PLOTLY_LAYOUT, height:330,
    margin:{{ t:10, l:42, r:14, b:24 }},
    xaxis:{{ color:'#4B5872', tickfont:{{ family:'Space Grotesk', size:10 }}, showline:false, gridcolor:'rgba(20,33,58,.05)' }},
    yaxis:{{ ticksuffix:'%', color:'#4B5872', tickfont:{{ size:9 }}, gridcolor:'rgba(20,33,58,.07)', zeroline:false }},
    showlegend:true, legend:{{ font:{{ family:'Space Grotesk', size:9 }}, orientation:'h', y:-0.16 }},
  }}, PLOTLY_CFG);
}}

// ── PROJECTIONS (colour-graded probability table) ─────────────────────────────
let projSortCol = null;
const PROJ_COLS = [
  {{ key:'p_group_1st',     label:'1st', scheme:'green' }},
  {{ key:'p_group_2nd',     label:'2nd', scheme:'green' }},
  {{ key:'p_group_3rd',     label:'3rd', scheme:'amber' }},
  {{ key:'p_group_4th',     label:'4th', scheme:'red'   }},
  {{ key:'p_qualify',       label:'R32', scheme:'green', sep:true }},
  {{ key:'p_round_of_16',   label:'R16', scheme:'green' }},
  {{ key:'p_quarter_final', label:'QF',  scheme:'green' }},
  {{ key:'p_semi_final',    label:'SF',  scheme:'green' }},
  {{ key:'p_final',         label:'F',   scheme:'green' }},
  {{ key:'p_champion',      label:'W',   scheme:'green' }},
];
function heatBg(x, scheme) {{
  x = Math.max(0, Math.min(1, x));
  const L = (a, b) => Math.round(a + (b - a) * x);
  if (scheme === 'red')   return `rgb(${{L(247,220)}},${{L(249,70)}},${{L(251,70)}})`;
  if (scheme === 'amber') return `rgb(${{L(247,234)}},${{L(249,159)}},${{L(251,38)}})`;
  return `rgb(${{L(247,30)}},${{L(249,140)}},${{L(251,80)}})`;
}}
function projSort(key) {{ projSortCol = (projSortCol === key) ? null : key; renderProjections(); }}
function renderProjections() {{
  const maxes = {{}};
  PROJ_COLS.forEach(c => maxes[c.key] = Math.max(...TEAMS.map(t => t[c.key] || 0)));
  let rows, grouped;
  if (projSortCol) {{
    rows = [...TEAMS].sort((a, b) => (b[projSortCol] || 0) - (a[projSortCol] || 0));
    grouped = false;
  }} else {{
    rows = [...TEAMS].sort((a, b) => a.group < b.group ? -1 : a.group > b.group ? 1 : (b.p_qualify - a.p_qualify));
    grouped = true;
  }}
  const head = `<tr>
    <th class="proj-rk"></th>
    <th class="proj-team-h">Team</th>
    <th class="proj-grp-h">Grp</th>
    ${{PROJ_COLS.map(c => `<th class="proj-h ${{c.sep?'proj-sep':''}} ${{projSortCol===c.key?'active':''}}" onclick="projSort('${{c.key}}')" title="Click to rank by ${{c.label}}">${{c.label}}</th>`).join('')}}
  </tr>`;
  let prevGrp = null;
  const body = rows.map((t, i) => {{
    const newGrp = grouped && t.group !== prevGrp; prevGrp = t.group;
    return `<tr class="${{newGrp?'proj-grp-start':''}}">
      <td class="proj-rk">${{projSortCol ? (i+1) : ''}}</td>
      <td class="proj-team">${{flagImg(t.team, 18)}}<span>${{t.team}}</span></td>
      <td class="proj-grp">${{t.group}}</td>
      ${{PROJ_COLS.map(c => {{
        const v = t[c.key] || 0;
        const x = maxes[c.key] > 0 ? v / maxes[c.key] : 0;
        return `<td class="proj-cell ${{c.sep?'proj-sep':''}}" style="background:${{heatBg(x,c.scheme)}};color:${{x>0.62?'#fff':'var(--txt)'}}">${{(v*100).toFixed(1)}}</td>`;
      }}).join('')}}
    </tr>`;
  }}).join('');
  document.getElementById('proj-table').innerHTML =
    `<table class="proj-table"><thead>${{head}}</thead><tbody>${{body}}</tbody></table>` +
    `<div class="proj-foot">Cell values are percentages &middot; <b>1st–4th</b> = group-stage finish, <b>R32→W</b> = stage reached. ` +
    `${{projSortCol ? '<a onclick="projSort(null)">↺ back to group order</a>' : 'Click a column header to rank the field by it.'}}</div>`;
}}

// ── OVERVIEW ──────────────────────────────────────────────────────────────────
let overviewDone = false;
function renderOverview() {{
  if (overviewDone) return; overviewDone = true;

  // Animate hero stats
  setTimeout(() => {{
    animateCount(document.getElementById('stat-teams'),   '48',  '', 800);
    animateCount(document.getElementById('stat-sims'),    '10',  'K', 1000);
    animateCount(document.getElementById('stat-sources'), '5',   '', 600);
    animateCount(document.getElementById('stat-matches'), '199', '', 1200);
  }}, 200);

  // Champion cards (top 10)
  const top10 = TEAMS.slice(0, 10);
  const container = document.getElementById('champ-cards');
  const cardClass   = ['gold-card','silver-card','bronze-card','','','','','','',''];
  const badgeClass  = ['gold','silver','bronze','','','','','','',''];
  const maxP = TEAMS[0].p_champion;

  container.innerHTML = top10.map((t, i) => `
    <div class="champ-card ${{cardClass[i]}}" onclick="gotoTab('team');setTimeout(()=>{{document.getElementById('team-select').value='${{t.team}}';renderTeam('${{t.team}}');}},80)">
      <div class="rank-badge ${{badgeClass[i]}}">${{i+1}}</div>
      <div class="champ-flag-wrap">${{flagImg(t.team, 43)}}</div>
      <div class="champ-name">${{t.team}}</div>
      <div class="champ-group">Group ${{t.group}} &middot; ELO ${{t.elo}}</div>
      <div class="champ-pct" data-target="${{(t.p_champion*100).toFixed(1)}}">${{pct(t.p_champion)}}</div>
      <div class="champ-lbl">Win Probability</div>
      <div class="mini-bar"><div class="mini-bar-fill" style="width:0%" data-width="${{(t.p_champion/maxP*100).toFixed(1)}}"></div></div>
    </div>
  `).join('');

  renderOverviewHeat();
  drawConfedChart();
  drawGoldenBoot();
  drawFirepower();
  drawSurvival();

  // Animate mini bars
  setTimeout(() => {{
    document.querySelectorAll('.mini-bar-fill').forEach(el => {{
      el.style.width = el.dataset.width + '%';
    }});
  }}, 100);

  staggerIn('.champ-card', 50);

  // ── TOP 16 LOLLIPOP CHART ──
  const top16 = [...TEAMS].sort((a,b) => b.p_champion - a.p_champion).slice(0,16);
  const sorted16 = [...top16].reverse();
  const maxVal = top16[0].p_champion;

  Plotly.newPlot('chart-bar', [
    // Base bars (muted gradient for rank-effect)
    {{
      type:'bar', orientation:'h',
      x: sorted16.map(t => +(t.p_champion*100).toFixed(2)),
      y: sorted16.map(t => t.team),
      text: sorted16.map(t => (t.p_champion*100).toFixed(1) + '%'),
      textposition:'outside',
      textfont:{{ family:'Space Grotesk, Inter', size:11, color:'#475569' }},
      marker:{{
        color: sorted16.map(t => {{
          const ratio = t.p_champion / maxVal;
          if (ratio > 0.85) return 'rgba(194,116,11,0.95)';
          if (ratio > 0.5)  return 'rgba(232,146,12,0.88)';
          if (ratio > 0.3)  return 'rgba(244,180,58,0.9)';
          return 'rgba(37,99,235,0.5)';
        }}),
        line:{{ color:'rgba(0,0,0,0)', width:0 }},
      }},
      hovertemplate:'<b>%{{y}}</b><br>Win Probability: %{{x:.1f}}%<extra></extra>',
      cliponaxis:false,
    }}
  ], {{
    ...PLOTLY_LAYOUT,
    height:520,
    margin:{{ t:10, l:150, r:80, b:30 }},
    xaxis:{{
      gridcolor:'rgba(20,33,58,.07)',
      ticksuffix:'%',
      color:'#4B5872',
      tickfont:{{ family:'Space Grotesk, Inter', size:10 }},
      zeroline:false,
      showline:false,
      range:[0, maxVal*100*1.25],
    }},
    yaxis:{{
      tickfont:{{ family:'Space Grotesk, Inter', size:12, color:'#475569' }},
      showline:false,
      gridcolor:'rgba(0,0,0,0)',
    }},
    bargap:0.35,
    shapes:sorted16.map((t,i) => ({{
      type:'line', xref:'x', yref:'y',
      x0:0, x1:t.p_champion*100, y0:t.team, y1:t.team,
      line:{{ color:'rgba(20,33,58,.08)', width:1 }},
    }})),
  }}, PLOTLY_CFG);

  // ── SCATTER: Attack vs Defense ──
  const quadrantAnnotations = [
    {{ x:0.85, y:0.85, text:'ELITE', showarrow:false, font:{{ color:'rgba(194,116,11,.5)', size:10, family:'Space Grotesk' }}, xref:'x', yref:'y' }},
    {{ x:0.2,  y:0.85, text:'DEFENSIVE', showarrow:false, font:{{ color:'rgba(37,99,235,.45)', size:10, family:'Space Grotesk' }}, xref:'x', yref:'y' }},
    {{ x:0.85, y:0.2,  text:'ATTACKING', showarrow:false, font:{{ color:'rgba(5,150,105,.45)', size:10, family:'Space Grotesk' }}, xref:'x', yref:'y' }},
    {{ x:0.2,  y:0.2,  text:'REBUILDING', showarrow:false, font:{{ color:'rgba(100,116,139,.4)', size:10, family:'Space Grotesk' }}, xref:'x', yref:'y' }},
  ];

  Plotly.newPlot('chart-scatter', [{{
    type:'scatter', mode:'markers+text',
    x: TEAMS.map(t => t.att_score),
    y: TEAMS.map(t => t.def_score),
    text: TEAMS.map(t => t.team.length > 12 ? t.team.slice(0,10)+'…' : t.team),
    textposition:'top center',
    textfont:{{ family:'Space Grotesk, Inter', size:8, color:'rgba(71,85,105,.75)' }},
    marker:{{
      size: TEAMS.map(t => Math.max(8, t.p_champion * 500)),
      color: TEAMS.map(t => t.p_champion),
      colorscale:[['0','#CBD5E1'],['0.35','#FCD9A0'],['0.7','#F0A92B'],['1','#C2740B']],
      showscale:false,
      line:{{ width:1, color:'rgba(20,33,58,.18)' }},
      opacity:0.9,
    }},
    hovertemplate:'<b>%{{customdata[0]}}</b><br>xG created/game: %{{customdata[2]}}<br>xG conceded/game: %{{customdata[3]}}<br>Win: %{{customdata[1]}}<extra></extra>',
    customdata: TEAMS.map(t => [t.team, pct(t.p_champion), (+t.xgf_per_game).toFixed(2), (+t.xga_per_game).toFixed(2)]),
  }}], {{
    ...PLOTLY_LAYOUT,
    height:340,
    margin:{{ t:20, l:54, r:20, b:48 }},
    xaxis:{{
      title:{{ text:'ATTACK SCORE', font:{{ family:'Space Grotesk', size:9, color:'#4B5872' }} }},
      gridcolor:'rgba(20,33,58,.07)', color:'#4B5872',
      zeroline:false, showline:false,
      tickfont:{{ size:9 }},
    }},
    yaxis:{{
      title:{{ text:'DEFENSE SCORE', font:{{ family:'Space Grotesk', size:9, color:'#4B5872' }} }},
      gridcolor:'rgba(20,33,58,.07)', color:'#4B5872',
      zeroline:false, showline:false,
      tickfont:{{ size:9 }},
    }},
    annotations: quadrantAnnotations,
    shapes:[
      {{ type:'line', x0:0.5, x1:0.5, y0:0, y1:1, xref:'x', yref:'paper', line:{{ color:'rgba(20,33,58,.07)', width:1, dash:'dot' }} }},
      {{ type:'line', x0:0, x1:1, y0:0.5, y1:0.5, xref:'paper', yref:'y', line:{{ color:'rgba(20,33,58,.07)', width:1, dash:'dot' }} }},
    ],
  }}, PLOTLY_CFG);

  // ── TREEMAP ──
  Plotly.newPlot('chart-treemap', [{{
    type:'treemap',
    ids: TEAMS.map(t => t.team),
    labels: TEAMS.map(t => (t.team.length > 13 ? t.team.slice(0,11)+'…' : t.team)),
    parents: TEAMS.map(() => ''),
    values: TEAMS.map(t => Math.max(t.p_champion, 0.001)),
    textinfo:'label',
    textfont:{{ family:'Space Grotesk, Inter', size:11 }},
    hovertemplate:'<b>%{{label}}</b><br>Win Probability: %{{customdata}}<extra></extra>',
    customdata: TEAMS.map(t => pct(t.p_champion)),
    marker:{{
      colors: TEAMS.map(t => t.p_champion),
      colorscale:[['0','#E2E8F0'],['0.25','#FCD9A0'],['0.55','#F4B43A'],['0.8','#E8920C'],['1','#C2740B']],
      line:{{ width:2, color:'#FFFFFF' }},
    }},
    tiling:{{ pad:2 }},
  }}], {{
    ...PLOTLY_LAYOUT,
    height:340,
    margin:{{ t:0, l:0, r:0, b:0 }},
  }}, PLOTLY_CFG);
}}

// ── GROUPS ────────────────────────────────────────────────────────────────────
let groupsDone = false;
// Overview heat table — top-12 contenders' knockout deep-run probabilities.
const OV_COLS = [
  {{ key:'p_qualify',       label:'R32' }},
  {{ key:'p_round_of_16',   label:'R16' }},
  {{ key:'p_quarter_final', label:'QF'  }},
  {{ key:'p_semi_final',    label:'SF'  }},
  {{ key:'p_final',         label:'F'   }},
  {{ key:'p_champion',      label:'Win', sep:true }},
];
function renderOverviewHeat() {{
  const el = document.getElementById('overview-heat');
  if (!el) return;
  const top = TEAMS.slice(0, 12);
  const maxes = {{}};
  OV_COLS.forEach(col => maxes[col.key] = Math.max(...TEAMS.map(t => t[col.key] || 0)));
  const head = `<tr><th class="gh-pos"></th><th class="gh-team">Team</th><th>Grp</th>` +
    OV_COLS.map(col => `<th class="${{col.sep ? 'gh-sep' : ''}}">${{col.label}}</th>`).join('') + `</tr>`;
  const body = top.map((t, i) => {{
    const cells = OV_COLS.map(col => {{
      const v = t[col.key] || 0, x = maxes[col.key] > 0 ? v / maxes[col.key] : 0;
      return `<td class="gh-cell ${{col.sep ? 'gh-sep' : ''}}" style="background:${{heatBg(x, 'green')}};color:${{x > 0.62 ? '#fff' : 'var(--txt)'}}">${{(v * 100).toFixed(1)}}</td>`;
    }}).join('');
    return `<tr>
      <td class="gh-pos">${{i + 1}}</td>
      <td class="gh-team">${{flagImg(t.team, 18)}}<span>${{t.team}}</span></td>
      <td style="color:var(--txt3);font-weight:700">${{t.group}}</td>${{cells}}
    </tr>`;
  }}).join('');
  el.innerHTML = `<table class="gh-table"><thead>${{head}}</thead><tbody>${{body}}</tbody></table>`;
}}

// Columns + colour schemes for the group heat tables (match the Projections tab).
const GROUP_COLS = [
  {{ key:'p_group_1st', label:'1st', scheme:'green' }},
  {{ key:'p_group_2nd', label:'2nd', scheme:'green' }},
  {{ key:'p_group_3rd', label:'3rd', scheme:'amber' }},
  {{ key:'p_group_4th', label:'4th', scheme:'red'   }},
  {{ key:'p_qualify',   label:'R32', scheme:'green', sep:true }},
  {{ key:'p_champion',  label:'Win', scheme:'green' }},
];
function renderGroups() {{
  if (groupsDone) return; groupsDone = true;
  // Per-column maxima across all 48 teams → colours are comparable group-to-group.
  const maxes = {{}};
  GROUP_COLS.forEach(col => maxes[col.key] = Math.max(...TEAMS.map(t => t[col.key] || 0)));
  const byGroup = {{}};
  TEAMS.forEach(t => {{ (byGroup[t.group] = byGroup[t.group] || []).push(t); }});

  const c = document.getElementById('groups-container');
  c.innerHTML = Object.keys(byGroup).sort().map(letter => {{
    const teams = byGroup[letter].slice().sort((a, b) => (b.p_qualify || 0) - (a.p_qualify || 0));
    const head = `<tr><th class="gh-pos"></th><th class="gh-team">Team</th>` +
      GROUP_COLS.map(col => `<th class="${{col.sep ? 'gh-sep' : ''}}">${{col.label}}</th>`).join('') + `</tr>`;
    const body = teams.map((t, i) => {{
      const posCls = i < 2 ? 'pos-q' : i === 2 ? 'pos-maybe' : 'pos-out';
      const cells = GROUP_COLS.map(col => {{
        const v = t[col.key] || 0;
        const x = maxes[col.key] > 0 ? v / maxes[col.key] : 0;
        return `<td class="gh-cell ${{col.sep ? 'gh-sep' : ''}}" style="background:${{heatBg(x, col.scheme)}};color:${{x > 0.62 ? '#fff' : 'var(--txt)'}}">${{(v * 100).toFixed(1)}}</td>`;
      }}).join('');
      return `<tr>
        <td class="gh-pos"><span class="gteam-pos ${{posCls}}" style="display:inline-flex">${{i + 1}}</span></td>
        <td class="gh-team">${{flagImg(t.team, 18)}}<span>${{t.team}}</span></td>${{cells}}
      </tr>`;
    }}).join('');
    return `<div class="group-card">
      <div class="group-header">
        <div class="group-letter">${{letter}}</div>
        <div>
          <div style="font-size:13px;font-weight:600;font-family:'Space Grotesk',sans-serif">Group ${{letter}}</div>
          <div class="group-subtitle">${{teams.length}} teams &middot; cells heat-graded by probability</div>
        </div>
      </div>
      <div class="gh-wrap"><table class="gh-table"><thead>${{head}}</thead><tbody>${{body}}</tbody></table></div>
    </div>`;
  }}).join('');
  staggerIn('.group-card', 30);
}}

// ── BRACKET (two-sided tree, every slot filled) ───────────────────────────────
let bracketDone = false;
const ROUND_ABBR = {{
  'Round of 32':'R32', 'Round of 16':'R16', 'Quarter-final':'QF',
  'Semi-final':'SF', 'Final':'FINAL', 'Third-place playoff':'3RD'
}};
function fmtDate(iso) {{
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d)) return '';
  return d.toLocaleDateString('en-US', {{ month:'short', day:'numeric' }});
}}
function btRow(team, isWin, p) {{
  if (!team) return `<div class="bt"><span class="bt-flag-ph"></span><span class="bt-name" style="color:var(--txt4)">TBD</span></div>`;
  return `<div class="bt ${{isWin ? 'win' : 'lose'}}">
    ${{flagImg(team, 15)}}
    <span class="bt-name">${{team}}</span>
    ${{(p != null) ? `<span class="bt-p">${{(p*100).toFixed(0)}}%</span>` : ''}}
  </div>`;
}}
function bMatch(mid) {{
  const m = BRACKETFULL[mid];
  if (!m) return `<div class="bm-slot"><div class="bm-card"></div></div>`;
  const homeWin = m.winner && m.winner === m.home;
  const awayWin = m.winner && m.winner === m.away;
  const pHome = (m.p_home_win != null) ? m.p_home_win : null;
  const pAway = (m.p_home_win != null) ? (1 - m.p_home_win) : null;
  const rl = ROUND_ABBR[m.round] || m.round;
  return `<div class="bm-slot">
    <div class="bm-card" title="${{m.venue || ''}}">
      <div class="bm-top"><span>${{rl}} &middot; M${{mid}}</span><span>${{fmtDate(m.date_utc)}}</span></div>
      ${{btRow(m.home, homeWin, pHome)}}
      ${{btRow(m.away, awayWin, pAway)}}
    </div>
  </div>`;
}}
function bCol(ids, cls) {{
  return `<div class="bcol ${{cls}}">${{ids.map(bMatch).join('')}}</div>`;
}}
function renderBracket() {{
  if (bracketDone) return; bracketDone = true;

  const leftWing = `<div class="bwing left">
    ${{bCol([73,75,74,77,83,84,81,82], 'r32')}}
    ${{bCol([89,90,93,94], 'r16')}}
    ${{bCol([97,98], 'qf')}}
    ${{bCol([101], 'sf')}}
  </div>`;

  const rightWing = `<div class="bwing right">
    ${{bCol([102], 'sf')}}
    ${{bCol([99,100], 'qf')}}
    ${{bCol([91,92,95,96], 'r16')}}
    ${{bCol([76,78,79,80,86,88,85,87], 'r32')}}
  </div>`;

  const champ  = BRACKETFULL[104];
  const bronze = BRACKETFULL[103];
  const center = `<div class="bcenter">
    <div class="bchamp">
      <div class="bchamp-trophy">🏆</div>
      <div class="bchamp-label">Predicted Champion</div>
      ${{(champ && champ.winner) ? `
        <div class="bchamp-flag">${{flagImg(champ.winner, 52)}}</div>
        <div class="bchamp-name">${{champ.winner}}</div>
        <div class="bchamp-sub">${{pct(champ.p_winner)}} to win the final</div>
      ` : ''}}
    </div>
    <div class="bcenter-final">
      <div class="bcenter-cap">Final &middot; ${{fmtDate(champ ? champ.date_utc : '')}} &middot; MetLife Stadium</div>
      ${{bMatch(104)}}
    </div>
    ${{bronze ? `<div class="bcenter-bronze">
      <div class="bcenter-cap muted">Third-place play-off</div>
      ${{bMatch(103)}}
    </div>` : ''}}
  </div>`;

  document.getElementById('bracket-tree').innerHTML = leftWing + center + rightWing;
}}

// ── PLAYERS ───────────────────────────────────────────────────────────────────
let playersDone = false;
function renderPlayers() {{
  if (playersDone) return; playersDone = true;

  const poty     = POTY[0];
  const ypoty    = YPOTY[0];
  const bestD    = BEST_DEF[0];
  const bestA    = BEST_ATT[0];
  const topS     = SCORERS[0];
  const gb       = GOLDEN_BALL[0];

  const awardDefs = [
    {{
      t:'Player of the Tournament', accent:'rgba(245,158,11,.4)', bg:'rgba(245,158,11,.05)',
      team: poty.team, name: poty.player, photo: poty.photo,
      sub: poty.position ? poty.team + ' &middot; ' + poty.position : poty.team,
      stat: (+poty.xg).toFixed(2) + ' xG', statColor:'var(--gold)',
      footer: (+poty.goals) + ' goals &middot; ' + (+poty.matches) + ' matches &middot; ' + pct(+poty.team_win_prob) + ' title odds',
    }},
    {{
      t:'Young Player of the Tournament', accent:'rgba(16,185,129,.3)', bg:'rgba(16,185,129,.04)',
      team: ypoty.team, name: ypoty.player, photo: ypoty.photo,
      sub: ypoty.team + ' &middot; Age ' + ypoty.age + (ypoty.position ? ' &middot; ' + ypoty.position : ''),
      stat: (ypoty.xg != null && !Number.isNaN(+ypoty.xg)) ? (+ypoty.xg).toFixed(2)+' xG' : 'Prospect',
      statColor:'var(--green)',
      footer: (ypoty.club || '') + (ypoty.goals != null ? ' &middot; ' + ypoty.goals + 'g in ' + ypoty.matches + ' matches' : ''),
    }},
    {{
      t:'Top Goal Scorer (xG)', accent:'rgba(239,68,68,.3)', bg:'rgba(239,68,68,.04)',
      team: topS.team, name: topS.player, photo: topS.photo,
      sub: topS.position ? topS.team + ' &middot; ' + topS.position : topS.team,
      stat: (+topS.xg).toFixed(2) + ' xG', statColor:'var(--red)',
      footer: (+topS.goals) + ' goals &middot; ' + (+topS.matches) + ' tournament matches',
    }},
    {{
      t:'Best Defensive Team', accent:'rgba(59,130,246,.3)', bg:'rgba(59,130,246,.04)',
      team: bestD.team, name: bestD.team, photo: null,
      sub: (+bestD.xga_per_game).toFixed(2) + ' xG conceded / game',
      stat: '#1 Defense', statColor:'var(--blue)',
      footer: 'ELO ' + bestD.elo + ' &middot; ' + pct(+bestD.p_champion) + ' title odds',
    }},
    {{
      t:'Most Dangerous Attack', accent:'rgba(139,92,246,.3)', bg:'rgba(139,92,246,.04)',
      team: bestA.team, name: bestA.team, photo: null,
      sub: (+bestA.xgf_per_game).toFixed(2) + ' xG created / game',
      stat: '#1 Attack', statColor:'var(--purple)',
      footer: 'ELO ' + bestA.elo + ' &middot; ' + pct(+bestA.p_champion) + ' title odds',
    }},
    {{
      t:'Golden Ball', accent:'rgba(234,179,8,.5)', bg:'rgba(234,179,8,.06)',
      team: gb.team, name: gb.player, photo: gb.photo,
      sub: gb.team + (gb.position ? ' &middot; ' + gb.position : ''),
      stat: pct(+gb.win_prob) + ' to win', statColor:'rgba(180,130,0,1)',
      footer: gb.g + 'G+' + gb.a + 'A club form &middot; ' + pct(+gb.deep_run) + ' to reach SF &middot; ' + pct(+gb.title_odds) + ' title odds',
    }},
  ];

  document.getElementById('award-cards').innerHTML = awardDefs.map(a => `
    <div class="award-card" style="border-color:${{a.accent}};background:linear-gradient(160deg,var(--card) 0%,${{a.bg}} 100%)">
      <div class="award-title">${{a.t}}</div>
      <div class="award-player">
        <div class="award-avatar">
          ${{a.photo
            ? `<img class="pphoto" src="${{a.photo}}" loading="lazy" alt="${{a.name}}"><span class="pphoto-flag">${{flagImg(a.team, 16)}}</span>`
            : flagImg(a.team, 40)}}
        </div>
        <div class="award-info">
          <h3>${{a.name}}</h3>
          <p>${{a.sub}}</p>
          <div class="award-stat" style="color:${{a.statColor}}">${{a.stat}}</div>
        </div>
      </div>
      <div class="award-footer">${{a.footer}}</div>
    </div>
  `).join('');

  // Scorer chart
  const sc = SCORERS.slice(0, 14);
  const scSorted = [...sc].sort((a,b) => +a.xg - +b.xg);
  const maxXG = Math.max(...scSorted.map(p => +p.xg));

  Plotly.newPlot('chart-scorers', [{{
    type:'bar', orientation:'h',
    x: scSorted.map(p => +(+p.xg).toFixed(2)),
    y: scSorted.map(p => p.player.split(' ').slice(-2).join(' ')),
    text: scSorted.map(p => (+(+p.xg).toFixed(1)) + ' xG'),
    textposition:'outside',
    textfont:{{ family:'Space Grotesk, Inter', size:10, color:'#475569' }},
    marker:{{
      color: scSorted.map(p => {{
        const ratio = +p.xg / maxXG;
        return `rgba(201,124,10,${{(0.5 + ratio*0.45).toFixed(2)}})`;
      }}),
      line:{{ width:0 }},
    }},
    hovertemplate:'<b>%{{y}}</b><br>xG: %{{x}}<br>Goals: %{{customdata[0]}}<br>Matches: %{{customdata[1]}}<extra></extra>',
    customdata: scSorted.map(p => [p.goals, p.matches]),
    cliponaxis:false,
  }}], {{
    ...PLOTLY_LAYOUT,
    height:420,
    margin:{{ t:10, l:140, r:70, b:30 }},
    xaxis:{{
      gridcolor:'rgba(20,33,58,.07)', color:'#4B5872',
      title:{{ text:'TOTAL xG', font:{{ family:'Space Grotesk', size:9, color:'#4B5872' }} }},
      zeroline:false, showline:false,
      tickfont:{{ size:9 }},
      range:[0, maxXG * 1.3],
    }},
    yaxis:{{
      tickfont:{{ family:'Space Grotesk, Inter', size:11, color:'#475569' }},
      showline:false, gridcolor:'rgba(0,0,0,0)',
    }},
    bargap:0.38,
  }}, PLOTLY_CFG);

  // Player POTY list
  const pAvatar = (p, sz) => `<div class="player-av">${{p.photo ? `<img class="pphoto sm" src="${{p.photo}}" loading="lazy" alt="">` : ''}}<span class="player-av-flag">${{flagImg(p.team, 14)}}</span></div>`;
  document.getElementById('player-list').innerHTML = POTY.slice(0, 10).map((p, i) => `
    <div class="player-row">
      <div class="player-rank ${{i<3?'top3':''}}">${{i+1}}</div>
      ${{pAvatar(p)}}
      <div class="player-info">
        <div class="player-name">${{p.player}}</div>
        <div class="player-team">${{p.team}}${{p.position ? ' &middot; ' + p.position : ''}} &middot; ${{(+p.xg).toFixed(1)}} xG</div>
      </div>
      <div class="player-xg" title="Impact = season xG × (0.6 + chance to reach the semi-final)">${{(+p.poty_score).toFixed(1)}}</div>
      <div class="player-goals" style="color:var(--txt3)">${{p.goals}}g</div>
    </div>
  `).join('');

  // YPOTY
  document.getElementById('ypoty-list').innerHTML = YPOTY.slice(0, 8).map((p, i) => `
    <div class="player-row">
      <div class="player-rank ${{i<3?'top3':''}}">${{i+1}}</div>
      ${{pAvatar(p)}}
      <div class="player-info">
        <div class="player-name">${{p.player}}</div>
        <div class="player-team">${{p.team}} &middot; Age ${{p.age}}${{p.position ? ' &middot; ' + p.position : ''}} &middot; ${{(+p.xg).toFixed(1)}} xG</div>
      </div>
      <div class="player-xg" style="color:var(--green)" title="Impact = season xG × (0.6 + chance to reach the semi-final)">${{(p.poty_score != null && !Number.isNaN(+p.poty_score)) ? (+p.poty_score).toFixed(1) : '—'}}</div>
    </div>
  `).join('');

  // Best defense
  document.getElementById('best-def-list').innerHTML = BEST_DEF.map((t, i) => `
    <div class="player-row">
      <div class="player-rank ${{i===0?'top3':''}}">${{i+1}}</div>
      <div class="player-flag">${{flagImg(t.team, 18)}}</div>
      <div class="player-info">
        <div class="player-name">${{t.team}}</div>
        <div class="player-team">${{(+t.xga_per_game).toFixed(2)}} xG conceded / game</div>
      </div>
      <div class="player-xg" style="color:var(--blue)">${{pct(+t.p_champion)}}</div>
    </div>
  `).join('');

  // Golden Ball race — predicted best player (form blend × projected run)
  const gbTop = GOLDEN_BALL.slice(0, 12);
  const gbMax = Math.max(...gbTop.map(p => +p.win_prob), 0.0001);
  document.getElementById('goldenball-list').innerHTML = gbTop.map((p, i) => `
    <div class="player-row gb-row">
      <div class="player-rank ${{i<3?'top3':''}}">${{i+1}}</div>
      ${{pAvatar(p)}}
      <div class="player-info">
        <div class="player-name">${{p.player}}</div>
        <div class="player-team">${{p.team}}${{p.position ? ' &middot; ' + p.position : ''}} &middot; ${{p.g}}G+${{p.a}}A club &middot; ${{(+p.exp_matches).toFixed(1)}} exp. matches</div>
      </div>
      <div class="gb-meter" title="Form rating ${{(+p.form_rating).toFixed(2)}} &middot; ${{pct(+p.deep_run)}} to reach SF">
        <div class="gb-meter-fill" style="width:${{(100*+p.win_prob/gbMax).toFixed(1)}}%"></div>
      </div>
      <div class="player-xg gb-pct">${{pct(+p.win_prob)}}</div>
    </div>
  `).join('');
}}

// ── TEAM DEEP DIVE ─────────────────────────────────────────────────────────────
function initTeam() {{
  const sel = document.getElementById('team-select');
  if (sel.options.length === 0) {{
    TEAMS.forEach(t => {{
      const o = document.createElement('option');
      o.value = t.team;
      o.text  = t.team;
      sel.appendChild(o);
    }});
  }}
  if (!sel.value || !TEAMS.find(x => x.team === sel.value)) {{
    renderTeam(TEAMS[0].team);
    sel.value = TEAMS[0].team;
  }}
}}

function pgPlayerCard(p) {{
  const bi = PLAYER_INDEX[p.team + '|' + p.name];
  const click = bi != null ? `onclick="openBio(${{bi}})" style="cursor:pointer"` : '';
  return `<div class="pg-player ${{p.is_star?'is-star':''}} ${{p.is_wonderkid?'is-wonder':''}}" ${{click}} title="View profile">
    <div class="pg-ph">
      ${{p.photo ? `<img src="${{p.photo}}" loading="lazy" alt="${{p.name}}">` : '<span class="pg-ph-x"></span>'}}
      ${{p.number != null ? `<span class="pg-no">${{p.number}}</span>` : ''}}
    </div>
    <div class="pg-nm">${{p.name}}</div>
    <div class="pg-meta">${{p.age != null ? p.age + 'y' : ''}}${{p.club ? ' &middot; ' + p.club : ''}}</div>
    ${{p.special ? `<div class="pg-tag">${{p.special}}</div>` : ''}}
  </div>`;
}}
function pgHighlight(p, label, color) {{
  if (!p) return '';
  const bi = PLAYER_INDEX[p.team + '|' + p.name];
  const click = bi != null ? `onclick="openBio(${{bi}})" style="--hl:${{color}};cursor:pointer"` : `style="--hl:${{color}}"`;
  return `<div class="pg-highlight" ${{click}}>
    <div class="pg-hl-badge">${{label}}</div>
    <div class="pg-hl-photo">${{p.photo ? `<img src="${{p.photo}}" loading="lazy" alt="${{p.name}}">` : ''}}</div>
    <div class="pg-hl-info">
      <div class="pg-hl-name">${{p.name}}</div>
      <div class="pg-hl-sub">${{[p.position, p.club, (p.age!=null?p.age+'y':'')].filter(Boolean).join(' · ')}}</div>
      ${{p.bio ? `<div class="pg-hl-bio">${{p.bio}}</div>` : ''}}
    </div>
  </div>`;
}}
function teamGuideHTML(team) {{
  const sq = SQUADS[team];
  if (!sq) return '';
  const m = sq.meta, players = sq.players;
  const star   = players.find(p => p.name === sq.star_name);
  const wonder = players.find(p => p.name === sq.wonderkid_name);
  const groups = [['Goalkeepers','GK'],['Defenders','DEF'],['Midfielders','MID'],['Forwards','FWD']];
  return `
    <div class="card" style="margin-top:16px">
      <div class="sec-eyebrow" style="margin-bottom:8px">Team Guide</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:14px">The Squad</div>
      <div class="pg-head">
        <div class="pg-head-row"><span class="pg-k">Coach</span><span class="pg-v">${{m.coach || '—'}}</span></div>
        <div class="pg-head-row"><span class="pg-k">FIFA Rank</span><span class="pg-v">#${{m.fifa_ranking || '—'}}</span></div>
        <div class="pg-head-row"><span class="pg-k">Squad size</span><span class="pg-v">${{players.length}}</span></div>
      </div>
      ${{m.bio ? `<p class="pg-bio">${{m.bio}}</p>` : ''}}
      <div class="pg-sw">
        <div class="pg-sw-card pg-str"><div class="pg-sw-t">Strengths</div><p>${{m.strengths || '—'}}</p></div>
        <div class="pg-sw-card pg-wk"><div class="pg-sw-t">Weaknesses</div><p>${{m.weaknesses || '—'}}</p></div>
      </div>
      <div class="pg-highlights">
        ${{pgHighlight(star, '★ Star Player', 'var(--gold)')}}
        ${{pgHighlight(wonder, '◆ Wonderkid', 'var(--green)')}}
      </div>
      <div class="pg-squad-title">Full Squad &middot; ${{players.length}} players</div>
      ${{groups.map(([label, code]) => {{
        const ps = players.filter(p => p.pos_group === code);
        if (!ps.length) return '';
        return `<div class="pg-pos-label">${{label}} <span>${{ps.length}}</span></div>
                <div class="pg-grid">${{ps.map(pgPlayerCard).join('')}}</div>`;
      }}).join('')}}
      ${{m.byline ? `<div class="pg-credit">Squad data &amp; photos: The Guardian &middot; ${{m.byline}}</div>` : ''}}
    </div>`;
}}
function renderTeam(teamName) {{
  const t = TEAMS.find(x => x.team === teamName);
  if (!t) return;

  const content = document.getElementById('team-content');
  content.innerHTML = `
    <div class="team-header">
      <div class="team-flag-lg">${{flagImg(teamName, 59)}}</div>
      <div class="team-meta">
        <h2>${{t.team}}</h2>
        <p>Group ${{t.group}} &middot; ELO Rank ${{TEAMS.findIndex(x => x.elo >= t.elo) + 1}} of 48</p>
        <div class="team-stat-strip">
          <div class="ts"><div class="ts-val">${{t.elo}}</div><div class="ts-lbl">ELO</div></div>
          <div class="ts"><div class="ts-val">${{(t.att_score*100).toFixed(0)}}</div><div class="ts-lbl">Attack</div></div>
          <div class="ts"><div class="ts-val">${{(t.def_score*100).toFixed(0)}}</div><div class="ts-lbl">Defense</div></div>
          <div class="ts"><div class="ts-val">#${{t.rank}}</div><div class="ts-lbl">Global Rank</div></div>
          <div class="ts"><div class="ts-val">${{pct(t.p_champion)}}</div><div class="ts-lbl">Win %</div></div>
        </div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
      <div class="card">
        <div class="sec-eyebrow" style="margin-bottom:10px">Group Stage</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:18px">Finishing Position Odds</div>
        ${{[
          ['Finish 1st',              t.p_group_1st,   'var(--gold)'],
          ['Finish 2nd',              t.p_group_2nd,   'var(--blue)'],
          ['Finish 3rd (may qualify)',t.p_group_3rd,   '#64748B'],
          ['Finish 4th (eliminated)', t.p_group_4th,   'var(--red)'],
        ].map(([label, val, color]) => `
          <div class="round-progress">
            <div class="rp-label"><span>${{label}}</span><span style="color:${{color}}">${{pct(val)}}</span></div>
            <div class="rp-bar"><div class="rp-fill" style="width:${{(val*100).toFixed(1)}}%;background:${{color}}"></div></div>
          </div>
        `).join('')}}
      </div>

      <div class="card">
        <div class="sec-eyebrow" style="margin-bottom:10px">Knockout Journey</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:18px">Progression Probabilities</div>
        ${{[
          ['Qualify (Round of 32)', t.p_qualify,        'var(--txt2)'],
          ['Round of 16',           t.p_round_of_16,    'var(--blue)'],
          ['Quarter-final',         t.p_quarter_final,  'var(--purple)'],
          ['Semi-final',            t.p_semi_final,     'var(--gold)'],
          ['Final',                 t.p_final,          'var(--red)'],
          ['Champion',              t.p_champion,       'var(--gold)'],
        ].map(([label, val, color]) => `
          <div class="round-progress">
            <div class="rp-label"><span>${{label}}</span><span style="color:${{color}}">${{pct(val)}}</span></div>
            <div class="rp-bar"><div class="rp-fill" style="width:${{Math.min(val*100,100).toFixed(1)}}%;background:${{color}}"></div></div>
          </div>
        `).join('')}}
      </div>
    </div>

    <div class="card" style="margin-bottom:16px">
      <div class="sec-eyebrow" style="margin-bottom:8px">Team Profile</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:4px">Strength Radar</div>
      <div id="radar-chart" style="height:340px"></div>
    </div>

    <div class="card">
      <div class="sec-eyebrow" style="margin-bottom:8px">Group ${{t.group}}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;margin-bottom:16px">Group Comparison</div>
      ${{renderGroupTable(t.group, teamName)}}
    </div>
    ${{teamGuideHTML(teamName)}}
  `;

  // Radar chart
  const maxElo = Math.max(...TEAMS.map(x => x.elo));
  const minElo = Math.min(...TEAMS.map(x => x.elo));
  const eloNorm = (t.elo - minElo) / (maxElo - minElo);
  const categories = ['Attack','Defense','ELO','Qualify %','Win %'];
  const values     = [t.att_score, t.def_score, eloNorm, t.p_qualify, t.p_champion * 5];

  Plotly.newPlot('radar-chart', [{{
    type:'scatterpolar',
    mode:'lines+markers',
    fill:'toself',
    r: [...values, values[0]],
    theta: [...categories, categories[0]],
    fillcolor:'rgba(245,158,11,0.1)',
    line:{{ color:'#F59E0B', width:2.5 }},
    marker:{{ color:'#F59E0B', size:7, line:{{ color:'rgba(245,158,11,.3)', width:3 }} }},
    name: teamName,
    hovertemplate:'<b>%{{theta}}</b><br>%{{r:.3f}}<extra></extra>',
  }}], {{
    ...PLOTLY_LAYOUT,
    height:340,
    margin:{{ t:30, l:60, r:60, b:30 }},
    polar:{{
      bgcolor:'rgba(0,0,0,0)',
      radialaxis:{{
        visible:true, range:[0,1],
        gridcolor:'rgba(20,33,58,.09)',
        color:'#4B5872',
        tickfont:{{ size:8, family:'Space Grotesk' }},
        tickvals:[0.25, 0.5, 0.75, 1.0],
        ticktext:['25%','50%','75%','100%'],
      }},
      angularaxis:{{
        color:'#64748B',
        tickfont:{{ size:12, family:'Space Grotesk' }},
        linecolor:'rgba(20,33,58,.08)',
      }},
      gridshape:'circular',
    }},
    showlegend:false,
  }}, PLOTLY_CFG);
}}

function renderGroupTable(groupLetter, currentTeam) {{
  const teams = GROUPS[groupLetter];
  // Per-column maxima within the group → heat-shade the numeric cells like the heat tables.
  const hcols = ['att_score', 'def_score', 'p_qualify', 'p_champion'];
  const hmax = {{}};
  hcols.forEach(k => hmax[k] = Math.max(...teams.map(t => t[k] || 0)));
  const heat = (t, k) => {{
    const v = t[k] || 0, x = hmax[k] > 0 ? v / hmax[k] : 0;
    return `background:${{heatBg(x, 'green')}};color:${{x > 0.62 ? '#fff' : 'var(--txt)'}}`;
  }};
  return `<table class="data-table">
    <thead><tr>
      <th>Flag</th><th>Team</th><th>ELO</th><th>Attack</th><th>Defense</th>
      <th>Qualify %</th><th>Win %</th>
    </tr></thead>
    <tbody>
      ${{teams.map(t => `<tr style="${{t.team===currentTeam?'background:rgba(245,158,11,.05)':''}}">
        <td style="padding:10px 8px 10px 14px">${{flagImg(t.team, 20)}}</td>
        <td class="${{t.team===currentTeam?'highlight':''}}">${{t.team}}</td>
        <td>${{t.elo}}</td>
        <td class="heat" style="${{heat(t,'att_score')}}">${{(t.att_score*100).toFixed(0)}}</td>
        <td class="heat" style="${{heat(t,'def_score')}}">${{(t.def_score*100).toFixed(0)}}</td>
        <td class="heat" style="${{heat(t,'p_qualify')}}">${{pct(t.p_qualify)}}</td>
        <td class="heat" style="${{heat(t,'p_champion')}}">${{pct(t.p_champion)}}</td>
      </tr>`).join('')}}
    </tbody>
  </table>`;
}}

// Boot — render overview immediately
renderOverview();
</script>
</body>
</html>"""

out_path = Path("dashboard.html")
out_path.write_text(HTML, encoding="utf-8")
print(f"dashboard.html written ({out_path.stat().st_size // 1024} KB)")
print("  Open dashboard.html in any browser — no server required.")
