#!/usr/bin/env python3
"""2026 World Cup Monte Carlo simulation."""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import re
import time
import random
import logging
import unicodedata
from math import exp, factorial
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import json
import xgboost as xgb

N_SIMS       = 10_000
ET_FACTOR    = 0.50

MODEL_PATH   = "Data/data/processed/poisson_goals_ours.pkl"
RATINGS_PATH = "Data/data/processed/team_ratings_2026.csv"
DC_PATH      = "Data/data/processed/dc_ratings_2026.csv"
XGB_PATH     = "Data/data/processed/xgb_goals.json"
ENS_PATH     = "Data/data/processed/ensemble.json"
MODEL_FEATURES = ["elo_diff", "off_rating_a", "def_rating_b", "is_home"]
XGB_FEATURES   = ["elo_home", "elo_away", "elo_diff", "off_rating_a", "def_rating_b", "is_home"]

AGE_TILT     = 0.010
AGE_PEAK     = 27.0

# per-player goal attribution (Golden Boot projection)
SQUADS_PATH        = "Data/scraped/squads_2026.json"
PXG_CUR_PATH       = "Data/scraped/player_xg_current.csv"   # 2025-26 club xG (Understat)
PXG_TOUR_PATH      = "Data/scraped/player_xg.csv"           # tournament xG (fallback)
MIN_SCORER_MATCHES = 10   # full club season — drops fringe/injury small samples
# Bayesian shrinkage of a player's per-match scoring rate toward a league prior,
# so 3-match samples don't masquerade as elite finishers.
SHRINK_PRIOR_RATE  = 0.12   # league-average attacker goals+xG per match
SHRINK_PRIOR_N     = 6.0    # prior strength, in "phantom matches"
# Each unmatched outfield squad player adds discard weight so a lone covered
# striker on a poorly-scraped squad can't inherit 100% of the team's goals.
DEPTH_WEIGHT       = 0.07    # per-match scoring weight of an unknown depth player

OUTPUT_DIR   = Path("Data/simulated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(OUTPUT_DIR / "simulation_log.txt", mode="w", encoding="utf-8")],
)
log = logging.getLogger(__name__)

CANON = {
    "Bosnia and Herzegovina":        "Bosnia-Herzegovina",
    "DR Congo":                       "Congo DR",
    "Ivory Coast":                    "Côte d'Ivoire",
    "Iran":                           "IR Iran",
    "South Korea":                    "Korea Republic",
    "Turkey":                         "Türkiye",
    "Czech Republic":                 "Czechia",
    "USA":                            "United States",
    "Cabo Verde":                     "Cape Verde",
}

def canon(name: str) -> str:
    return CANON.get(str(name).strip(), str(name).strip())

def load_all() -> dict:
    log.info("Loading data...")

    grp_df = pd.read_csv("Data/data/raw/groups.csv")
    grp_df["team"] = grp_df["team"].apply(canon)
    groups = dict(zip(grp_df["team"], grp_df["group"]))
    group_teams: dict[str, list[str]] = defaultdict(list)
    for team, grp in groups.items():
        group_teams[grp].append(team)

    slots_df = pd.read_csv("Data/data/raw/knockout_slots.csv")

    # completed group-stage results, order-agnostic
    played: dict[frozenset, dict[str, int]] = {}
    sched_path = Path("Data/schedule_2026.csv")
    if sched_path.exists():
        sch = pd.read_csv(sched_path)
        gs = sch[sch["Round"] == "Group stage"]
        for _, row in gs.iterrows():
            score = str(row.get("Score", "")).strip()
            if not score or score.lower() in ("nan", ""):
                continue
            try:
                gh, ga = (int(x) for x in score.replace("–", "-").split("-"))
            except (ValueError, AttributeError):
                continue
            h, a = canon(row["home_team"]), canon(row["away_team"])
            played[frozenset((h, a))] = {h: gh, a: ga}
        if played:
            log.info(f"  Conditioning on {len(played)} completed group match(es)")

    b3_df = pd.read_csv("Data/data/raw/fw26_best_third_placed_combinations.csv")
    slot_cols = [c for c in b3_df.columns if c != "Option"]
    best3_lookup: dict[frozenset, dict[str, str]] = {}
    for _, row in b3_df.iterrows():
        mapping = {col: row[col][1] for col in slot_cols}
        key = frozenset(mapping.values())
        best3_lookup[key] = mapping

    log.info(f"  Groups:          {len(groups)} teams")
    log.info(f"  Best-3rd lookup: {len(best3_lookup)} combinations")

    return dict(groups=groups, group_teams=dict(group_teams),
                slots_df=slots_df, best3_lookup=best3_lookup, played=played)

def build_strengths(data: dict) -> dict[str, dict]:
    """Build per-team ratings and pairwise expected-goals lookup from the ensemble."""
    teams = list(data["groups"].keys())

    pipe = joblib.load(MODEL_PATH)
    booster = xgb.Booster(); booster.load_model(XGB_PATH)
    ens = json.loads(Path(ENS_PATH).read_text(encoding="utf-8"))
    w_glm = float(ens["weights"]["GLM-Poisson"])
    w_xgb = float(ens["weights"]["XGBoost"])
    w_dc  = float(ens["weights"]["Dixon-Coles"])

    ratings = pd.read_csv(RATINGS_PATH)
    ratings["team"] = ratings["team"].apply(canon)
    rat = {r["team"]: r for _, r in ratings.iterrows()}

    dc_df = pd.read_csv(DC_PATH); dc_df["team"] = dc_df["team"].apply(canon)
    dcr = {r["team"]: r for _, r in dc_df.iterrows()}
    dc_att_med = dc_df["dc_attack"].median(); dc_def_med = dc_df["dc_defence"].median()

    med_elo = ratings["elo"].median()
    med_off = ratings["off_rating"].median()
    med_def = ratings["def_rating"].median()
    med_age = ratings["avg_age"].median()

    def get(t, col, fallback):
        return float(rat[t][col]) if t in rat and pd.notna(rat[t][col]) else float(fallback)

    elo = {t: get(t, "elo", med_elo) for t in teams}
    off = {t: get(t, "off_rating", med_off) for t in teams}
    dfn = {t: get(t, "def_rating", med_def) for t in teams}
    age = {t: get(t, "avg_age", med_age) for t in teams}
    dc_att = {t: float(dcr[t]["dc_attack"]) if t in dcr else dc_att_med for t in teams}
    dc_def = {t: float(dcr[t]["dc_defence"]) if t in dcr else dc_def_med for t in teams}

    age_factor = {t: float(np.exp(-AGE_TILT * abs(age[t] - AGE_PEAK))) for t in teams}

    def ens_lambda(rows_glm, rows_xgb, dc_lams):
        out = np.zeros(len(dc_lams))
        if w_glm: out += w_glm * pipe.predict(pd.DataFrame(rows_glm, columns=MODEL_FEATURES))
        if w_xgb: out += w_xgb * booster.predict(xgb.DMatrix(pd.DataFrame(rows_xgb, columns=XGB_FEATURES)))
        if w_dc:  out += w_dc  * np.asarray(dc_lams)
        return out

    pairs = [(a, b) for a in teams for b in teams if a != b]
    glm_rows = [[elo[a] - elo[b], off[a], dfn[b], 0] for a, b in pairs]
    xgb_rows = [[elo[a], elo[b], elo[a] - elo[b], off[a], dfn[b], 0] for a, b in pairs]
    dc_lams  = [float(np.exp(dc_att[a] - dc_def[b])) for a, b in pairs]
    preds = ens_lambda(glm_rows, xgb_rows, dc_lams)

    lam: dict[str, dict[str, float]] = {t: {} for t in teams}
    for (a, b), p in zip(pairs, preds):
        lam[a][b] = float(p) * age_factor[a]

    # expected goals for/against vs an average opponent
    avg_elo = float(np.mean(list(elo.values())))
    avg_off = float(np.mean(list(off.values())))
    avg_def = float(np.mean(list(dfn.values())))
    avg_dca = float(np.mean(list(dc_att.values())))
    avg_dcd = float(np.mean(list(dc_def.values())))
    xgf_glm = [[elo[t] - avg_elo, off[t], avg_def, 0] for t in teams]
    xgf_xgb = [[elo[t], avg_elo, elo[t] - avg_elo, off[t], avg_def, 0] for t in teams]
    xgf_dc  = [float(np.exp(dc_att[t] - avg_dcd)) for t in teams]
    xga_glm = [[avg_elo - elo[t], avg_off, dfn[t], 0] for t in teams]
    xga_xgb = [[avg_elo, elo[t], avg_elo - elo[t], avg_off, dfn[t], 0] for t in teams]
    xga_dc  = [float(np.exp(avg_dca - dc_def[t])) for t in teams]
    xgf = {t: float(v) * age_factor[t] for t, v in zip(teams, ens_lambda(xgf_glm, xgf_xgb, xgf_dc))}
    xga = {t: float(v) for t, v in zip(teams, ens_lambda(xga_glm, xga_xgb, xga_dc))}

    def norm01(d, invert=False):
        vals = list(d.values()); mn, mx = min(vals), max(vals)
        rng = (mx - mn) or 1.0
        return {t: (1 - (v - mn) / rng) if invert else (v - mn) / rng for t, v in d.items()}
    att_score = norm01(xgf)
    def_score = norm01(xga, invert=True)

    strengths = {}
    for t in teams:
        strengths[t] = {
            "att": off[t],
            "def": dfn[t],
            "xgf": xgf[t],
            "xga": xga[t],
            "att_score": att_score[t],
            "def_score": def_score[t],
            "elo": elo[t],
            "age": age[t],
            "lam": lam[t],
        }
    return strengths

def _pnorm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    # split hyphens / dots / apostrophes into separate tokens (Mbappe-Lottin → mbappe lottin)
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()

def _firstlast(n: str) -> str:
    t = n.split()
    return (t[0] + " " + t[-1]) if len(t) >= 2 else n

def build_player_shares(teams: list[str]) -> tuple[list[dict], dict]:
    """For every WC squad player, derive their share of the team's scoring from
    2025-26 club xG/goals (Understat), with tournament xG as fallback. The share
    becomes the per-goal attribution probability in the Monte Carlo.

    Returns (players_meta, team_share):
      players_meta : list of per-player dicts, indexed by global id
      team_share   : {team: (idx_array, prob_array)}  probs sum to 1 per team
    """
    squads_raw = json.loads(Path(SQUADS_PATH).read_text(encoding="utf-8"))

    def build_lookup(path: str) -> dict:
        lk: dict = {}
        if not Path(path).exists():
            return lk
        df = pd.read_csv(path).sort_values("xg", ascending=False)
        df["_n"] = df["player"].map(_pnorm)
        for _, r in df.iterrows():
            toks = r["_n"].split()
            keys = [r["_n"], _firstlast(r["_n"])]
            # also index "first + each other token" so compound surnames in the
            # data (Messi Cuccitini, Ronaldo dos Santos) reconcile to clean squad names
            keys += [toks[0] + " " + t for t in toks[1:]]
            for key in keys:
                lk.setdefault(key, r)   # higher-xG row wins (df sorted desc)
        return lk

    cur_lk  = build_lookup(PXG_CUR_PATH)
    tour_lk = build_lookup(PXG_TOUR_PATH)

    players_meta: list[dict] = []
    team_share: dict[str, tuple] = {}
    teamset = set(teams)

    for raw_team, blk in squads_raw.items():
        team = canon(raw_team)
        if team not in teamset:
            continue
        n_outfield = sum(1 for p in blk["players"] if p.get("pos_group") != "GK")
        idxs, rates = [], []
        for p in blk["players"]:
            n  = _pnorm(p["name"])
            toks = n.split()
            cand_keys = [n, _firstlast(n)]
            if len(toks) > 2:
                cand_keys.append(toks[0] + " " + toks[1])   # first + second token
            rec = None
            for lk in (cur_lk, tour_lk):       # prefer current-club xG over tournament
                for key in cand_keys:
                    if key in lk:
                        rec = lk[key]; break
                if rec is not None:
                    break
            if rec is None:
                continue
            matches = int(rec["matches"])
            if matches < MIN_SCORER_MATCHES:
                continue
            xg, goals = float(rec["xg"]), int(rec["goals"])
            # half xG (chance quality), half actual finishing
            raw_output = 0.5 * xg + 0.5 * goals
            # shrink toward the league prior so small samples regress to the mean
            rate = ((raw_output + SHRINK_PRIOR_RATE * SHRINK_PRIOR_N)
                    / (matches + SHRINK_PRIOR_N))
            if rate <= 0:
                continue
            players_meta.append({
                "player": p["name"], "team": team,
                "club": rec.get("club", p.get("club", "")),
                "position": p.get("position", ""),
                "club_rate": round(rate, 4), "club_matches": matches,
            })
            idxs.append(len(players_meta) - 1)
            rates.append(rate)
        if rates:
            # add a discard bucket for the squad's unmatched outfield depth, so
            # goals aren't fully absorbed by the few scraped players
            depth = max(n_outfield - len(rates), 0) * DEPTH_WEIGHT
            weights = np.asarray(rates + [depth], float)
            team_share[team] = (np.asarray(idxs, int), weights / weights.sum())

    log.info(f"  Player scorers:  {len(players_meta)} squad players across "
             f"{len(team_share)} teams")
    return players_meta, team_share

def expected_goals(team_a: str, team_b: str, strengths: dict) -> tuple[float, float]:
    lam_a = strengths[team_a]["lam"][team_b]
    lam_b = strengths[team_b]["lam"][team_a]
    return lam_a, lam_b

def sim_match(team_a: str, team_b: str, strengths: dict,
              knockout: bool = False) -> tuple[int, int, bool]:
    """Return (goals_a, goals_b, penalties_used)."""
    lam_a, lam_b = expected_goals(team_a, team_b, strengths)
    ga = np.random.poisson(lam_a)
    gb = np.random.poisson(lam_b)

    if not knockout or ga != gb:
        return ga, gb, False

    # extra time
    ga += np.random.poisson(lam_a * ET_FACTOR)
    gb += np.random.poisson(lam_b * ET_FACTOR)

    if ga != gb:
        return ga, gb, True

    # penalties
    elo_a = strengths[team_a]["elo"]
    elo_b = strengths[team_b]["elo"]
    p_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 800))
    if np.random.random() < p_a:
        return ga + 1, gb, True
    else:
        return ga, gb + 1, True

def sim_group(teams: list[str], strengths: dict,
              played: dict | None = None) -> list[dict]:
    """Simulate one group, returning team rows sorted by standing."""
    played = played or {}
    records = {t: {"team": t, "pts": 0, "gf": 0, "ga": 0, "gd": 0,
                   "w": 0, "d": 0, "l": 0} for t in teams}
    h2h_pts: dict[tuple[str, str], int] = {(a, b): 0 for a in teams for b in teams if a != b}
    h2h_gd:  dict[tuple[str, str], int] = {(a, b): 0 for a in teams for b in teams if a != b}

    for a, b in combinations(teams, 2):
        real = played.get(frozenset((a, b)))
        if real is not None:
            ga, gb = real[a], real[b]
        else:
            ga, gb, _ = sim_match(a, b, strengths, knockout=False)
        for t, gf, gag in [(a, ga, gb), (b, gb, ga)]:
            records[t]["gf"] += gf
            records[t]["ga"] += gag
            records[t]["gd"] += gf - gag
        if ga > gb:
            records[a]["pts"] += 3; records[a]["w"] += 1; records[b]["l"] += 1
            h2h_pts[(a, b)] += 3; h2h_gd[(a, b)] += ga - gb; h2h_gd[(b, a)] += gb - ga
        elif ga < gb:
            records[b]["pts"] += 3; records[b]["w"] += 1; records[a]["l"] += 1
            h2h_pts[(b, a)] += 3; h2h_gd[(b, a)] += gb - ga; h2h_gd[(a, b)] += ga - gb
        else:
            records[a]["pts"] += 1; records[a]["d"] += 1
            records[b]["pts"] += 1; records[b]["d"] += 1
            h2h_pts[(a, b)] += 1; h2h_pts[(b, a)] += 1

    # sort: pts, gd, gf, h2h pts, h2h gd, elo
    def sort_key(t):
        r = records[t]
        others = [o for o in teams if o != t]
        h2h_p = sum(h2h_pts[(t, o)] for o in others)
        h2h_g = sum(h2h_gd[(t, o)]  for o in others)
        return (r["pts"], r["gd"], r["gf"], h2h_p, h2h_g, strengths[t]["elo"])

    sorted_teams = sorted(teams, key=sort_key, reverse=True)
    return [{"pos": i + 1, **records[t]} for i, t in enumerate(sorted_teams)]

def pick_best_third(third_teams: dict[str, dict],
                    strengths: dict,
                    best3_lookup: dict) -> dict[str, str]:
    """Return {slot_col: team_name} for the 8 best 3rd-place teams."""
    ranked = sorted(
        third_teams.items(),
        key=lambda kv: (kv[1]["pts"], kv[1]["gd"], kv[1]["gf"],
                        strengths[kv[1]["team"]]["elo"]),
        reverse=True
    )
    advancing = ranked[:8]
    adv_groups = frozenset(g for g, _ in advancing)

    slot_map = best3_lookup.get(adv_groups)
    if slot_map is None:
        slot_map = next(iter(best3_lookup.values()))

    group_to_team = {g: r["team"] for g, r in advancing}
    return {slot: group_to_team[slot_map[slot]] for slot in slot_map
            if slot_map[slot] in group_to_team}

def sim_knockout(slots_df: pd.DataFrame,
                 group_results: dict[str, list[dict]],
                 best3_slots: dict[str, str],
                 strengths: dict) -> dict[str, str]:
    """Simulate the knockout bracket; return (match_winners, champion)."""
    def resolve(slot_str: str) -> str | None:
        s = slot_str.strip()
        if s.startswith("Winner Group "):
            g = s[-1]
            return group_results[g][0]["team"]
        if s.startswith("Runner-up Group "):
            g = s[-1]
            return group_results[g][1]["team"]
        if s.startswith("Best 3rd"):
            return None
        if s.startswith("Winner Match "):
            return None
        if s.startswith("Loser Match "):
            return None
        return None

    BEST3_MATCH_TO_SLOTCOL = {
        79: "1A",
        85: "1B",
        82: "1D",
        75: "1E",
        81: "1G",
        78: "1I",
        88: "1K",
        80: "1L",
    }

    match_winners: dict[int, str] = {}
    match_losers:  dict[int, str] = {}
    ko_goals:      dict[str, int] = defaultdict(int)

    def get_team(slot_str: str, match_id: int) -> str | None:
        s = slot_str.strip()
        if s.startswith("Winner Group "):
            return group_results[s[-1]][0]["team"]
        if s.startswith("Runner-up Group "):
            return group_results[s[-1]][1]["team"]
        if s.startswith("Best 3rd"):
            col = BEST3_MATCH_TO_SLOTCOL.get(match_id)
            return best3_slots.get(col) if col else None
        if s.startswith("Winner Match "):
            mid = int(s.split()[-1])
            return match_winners.get(mid)
        if s.startswith("Loser Match "):
            mid = int(s.split()[-1])
            return match_losers.get(mid)
        return None

    for _, row in slots_df.iterrows():
        mid   = int(row["match_id"])
        rnd   = row["round"]
        home  = get_team(row["slot_home"], mid)
        away  = get_team(row["slot_away"], mid)
        if home is None or away is None:
            continue
        if rnd == "Third-place playoff":
            continue   # skip 3rd-place match for simulation purposes

        gh, ga, pen = sim_match(home, away, strengths, knockout=True)
        winner = home if gh > ga else away
        loser  = away if gh > ga else home
        match_winners[mid] = winner
        match_losers[mid]  = loser
        # on-pitch goals only — drop the shootout goal so it isn't attributed to a scorer
        rgh, rga = gh, ga
        if pen:
            if gh > ga: rgh -= 1
            elif ga > gh: rga -= 1
        ko_goals[home] += rgh
        ko_goals[away] += rga

    # Final is the last match in the bracket (max match_id, round='Final')
    finals = slots_df[slots_df["round"] == "Final"]
    champion = match_winners.get(int(finals["match_id"].iloc[0])) if len(finals) > 0 else None

    return match_winners, champion, ko_goals

# full tournament simulation
def simulate_tournament(data: dict, strengths: dict) -> dict:
    group_teams  = data["group_teams"]
    slots_df     = data["slots_df"]
    best3_lookup = data["best3_lookup"]
    played       = data.get("played", {})

    # --- Group stage ---
    group_results: dict[str, list[dict]] = {}
    third_teams:   dict[str, dict] = {}

    for grp, teams in group_teams.items():
        standing = sim_group(teams, strengths, played)
        group_results[grp] = standing
        third_teams[grp]   = standing[2]  # 3rd place

    # --- Best 3rd ---
    best3_slots = pick_best_third(third_teams, strengths, best3_lookup)

    # --- Knockout ---
    match_winners, champion, ko_goals = sim_knockout(
        slots_df, group_results, best3_slots, strengths
    )

    # --- Goals scored per team this tournament (group + knockout) ---
    team_goals: dict[str, int] = defaultdict(int)
    for standing in group_results.values():
        for row in standing:
            team_goals[row["team"]] += row["gf"]
    for t, g in ko_goals.items():
        team_goals[t] += g

    return {
        "group_results": group_results,
        "best3_slots":   best3_slots,
        "match_winners": match_winners,
        "champion":      champion,
        "team_goals":    team_goals,
    }

# monte carlo
ROUND_ORDER = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]

def run_monte_carlo(data: dict, strengths: dict,
                    players_meta: list[dict], team_share: dict,
                    n_sims: int = N_SIMS) -> tuple:
    all_teams = list(data["groups"].keys())
    slots_df  = data["slots_df"]

    # Track counts per team per outcome
    counts = {t: defaultdict(int) for t in all_teams}
    # Track group position distribution {team: {1:n, 2:n, 3:n, 4:n}}
    pos_dist = {t: defaultdict(int) for t in all_teams}
    # Track per-match winner counts {match_id: {team: n}}
    match_win_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Per-player goal attribution accumulators (Golden Boot projection)
    n_players = len(players_meta)
    g_total   = np.zeros(n_players)   # Σ goals over sims  → mean
    g_sumsq   = np.zeros(n_players)   # Σ goals²          → variance
    gb_wins   = np.zeros(n_players)   # fractional Golden Boot wins (ties split)

    # Build match-id → round mapping
    match_round = dict(zip(slots_df["match_id"].astype(int), slots_df["round"]))

    log.info(f"Running {n_sims:,} simulations...")
    t0 = time.time()

    for i in range(n_sims):
        result = simulate_tournament(data, strengths)
        group_results = result["group_results"]
        match_winners = result["match_winners"]
        champion      = result["champion"]

        # Group stage position distribution
        for grp, standing in group_results.items():
            for row in standing:
                t = row["team"]
                counts[t]["group_played"] += 1
                pos_dist[t][row["pos"]] += 1
                if row["pos"] <= 2:
                    counts[t]["qualified_top2"] += 1
                elif row["pos"] == 3:
                    counts[t]["finished_3rd"] += 1

        # Count "best 3rd" qualifications
        for team in result["best3_slots"].values():
            if team:
                counts[team]["qualified_best3"] += 1

        # Knockout round appearances + per-match winner tracking
        for mid, winner in match_winners.items():
            rnd = match_round.get(mid)
            if rnd and rnd != "Third-place playoff":
                counts[winner][f"reached_{rnd}"] += 1
                match_win_counts[mid][winner] += 1

        # Champion
        if champion:
            counts[champion]["champion"] += 1

        # Attribute each team's goals to individual scorers (multinomial by share)
        team_goals = result["team_goals"]
        sim_best, sim_winners = 0, []
        for team, (idx, probs) in team_share.items():
            g = team_goals.get(team, 0)
            if g <= 0:
                continue
            # probs has a trailing discard bucket (unscraped depth) — drop it
            draw = np.random.multinomial(g, probs)[:-1]
            g_total[idx] += draw
            g_sumsq[idx] += draw * draw
            m = int(draw.max())
            if m > sim_best:
                sim_best, sim_winners = m, list(idx[draw == m])
            elif m == sim_best and m > 0:
                sim_winners.extend(idx[draw == m])
        if sim_winners:
            w = 1.0 / len(sim_winners)
            for gi in sim_winners:
                gb_wins[gi] += w

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            log.info(f"  Sim {i+1:,}/{n_sims:,}  ({elapsed:.1f}s)")

    log.info(f"Simulations done in {time.time()-t0:.1f}s")

    # --- Build summary DataFrame ---
    rows = []
    for t in all_teams:
        c = counts[t]
        q32 = c["qualified_top2"] + c["qualified_best3"]
        p1 = pos_dist[t][1] / n_sims
        p2 = pos_dist[t][2] / n_sims
        p3 = pos_dist[t][3] / n_sims
        p4 = pos_dist[t][4] / n_sims
        rows.append({
            "team":              t,
            "group":             data["groups"][t],
            "elo":               round(strengths[t]["elo"]),
            "att_score":         round(strengths[t]["att_score"], 3),  # 0–1, higher = better
            "def_score":         round(strengths[t]["def_score"], 3),  # 0–1, higher = better
            "xgf_per_game":      round(strengths[t]["xgf"], 2),        # exp goals FOR vs avg opp
            "xga_per_game":      round(strengths[t]["xga"], 2),        # exp goals AGAINST vs avg opp (lower better)
            "off_rating":        round(strengths[t]["att"], 3),        # raw form+firepower
            "def_rating":        round(strengths[t]["def"], 3),        # raw, lower = better
            "p_qualify":         round(q32              / n_sims, 4),
            "p_group_1st":       round(p1, 4),
            "p_group_2nd":       round(p2, 4),
            "p_group_3rd":       round(p3, 4),
            "p_group_4th":       round(p4, 4),
            "p_round_of_16":     round(c["reached_Round of 16"]   / n_sims, 4),
            "p_quarter_final":   round(c["reached_Quarter-final"] / n_sims, 4),
            "p_semi_final":      round(c["reached_Semi-final"]    / n_sims, 4),
            "p_final":           round(c["reached_Final"]         / n_sims, 4),
            "p_champion":        round(c["champion"]              / n_sims, 4),
        })

    summary_df = pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
    summary_df["rank"] = summary_df.index + 1

    # --- Build bracket predictions DataFrame ---
    bracket_rows = []
    for mid, win_counts in match_win_counts.items():
        rnd = match_round.get(mid, "Unknown")
        if not win_counts:
            continue
        total = sum(win_counts.values())
        best_team = max(win_counts, key=win_counts.get)
        best_prob = win_counts[best_team] / n_sims
        bracket_rows.append({
            "match_id":   mid,
            "round":      rnd,
            "pred_winner": best_team,
            "p_win":       round(best_prob, 4),
            "win_counts":  dict(win_counts),
        })
    bracket_df = pd.DataFrame(bracket_rows).sort_values("match_id")

    # --- Build per-player scoring projection ---
    prows = []
    for gi, meta in enumerate(players_meta):
        exp = g_total[gi] / n_sims
        var = g_sumsq[gi] / n_sims - exp * exp
        prows.append({
            **meta,
            "exp_goals":     round(float(exp), 3),
            "sd_goals":      round(float(np.sqrt(max(var, 0.0))), 3),
            "p_golden_boot": round(float(gb_wins[gi] / n_sims), 4),
        })
    player_df = (pd.DataFrame(prows)
                 .sort_values("exp_goals", ascending=False)
                 .reset_index(drop=True)) if prows else pd.DataFrame()

    return summary_df, bracket_df, player_df

# deterministic "favourite advances" bracket
def poisson_pmf(k: int, lam: float) -> float:
    return exp(-lam) * lam ** k / factorial(k)

def match_win_prob(home: str, away: str, strengths: dict, max_goals: int = 12) -> tuple[float, float]:
    """Closed-form probability (home_wins, away_wins) for a single knockout match.
    90-min Poisson → draws resolved by extra time → remaining draws by ELO-weighted pens.
    Deterministic (no RNG)."""
    lam_h, lam_a = expected_goals(home, away, strengths)
    ph = [poisson_pmf(k, lam_h) for k in range(max_goals + 1)]
    pa = [poisson_pmf(k, lam_a) for k in range(max_goals + 1)]

    def split(ph, pa):
        h = a = d = 0.0
        for i in range(len(ph)):
            for j in range(len(pa)):
                p = ph[i] * pa[j]
                if   i > j: h += p
                elif i < j: a += p
                else:       d += p
        return h, a, d

    h_reg, a_reg, d_reg = split(ph, pa)
    # Extra time at reduced scoring rate
    ph2 = [poisson_pmf(k, lam_h * ET_FACTOR) for k in range(max_goals + 1)]
    pa2 = [poisson_pmf(k, lam_a * ET_FACTOR) for k in range(max_goals + 1)]
    h_et, a_et, d_et = split(ph2, pa2)
    # Penalties — muted ELO weighting (matches sim_match)
    elo_h, elo_a = strengths[home]["elo"], strengths[away]["elo"]
    p_h_pen = 1 / (1 + 10 ** ((elo_a - elo_h) / 800))

    p_home = h_reg + d_reg * (h_et + d_et * p_h_pen)
    p_away = a_reg + d_reg * (a_et + d_et * (1 - p_h_pen))
    s = p_home + p_away
    return p_home / s, p_away / s

def predict_bracket(data: dict, strengths: dict, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build ONE deterministic predicted bracket where the model's favourite always
    advances, filling both teams of every knockout match through to the champion."""
    slots_df     = data["slots_df"]
    best3_lookup = data["best3_lookup"]
    group_teams  = data["group_teams"]
    srow = {r["team"]: r for _, r in summary_df.iterrows()}

    # 1. Predicted final standing of each group (greedy, guarantees distinct teams)
    standings: dict[str, list[str]] = {}
    for grp, teams in group_teams.items():
        remaining = set(teams)
        order = []
        for col in ["p_group_1st", "p_group_2nd", "p_group_3rd"]:
            pick = max(remaining, key=lambda t: srow[t][col])
            order.append(pick)
            remaining.discard(pick)
        order.append(remaining.pop())               # 4th
        standings[grp] = order

    # 2. Eight best 3rd-place teams (ranked by their probability of reaching R16)
    thirds = {grp: standings[grp][2] for grp in group_teams}
    ranked = sorted(thirds.items(), key=lambda kv: srow[kv[1]]["p_round_of_16"], reverse=True)
    adv_groups = frozenset(g for g, _ in ranked[:8])
    slot_map = best3_lookup.get(adv_groups) or next(iter(best3_lookup.values()))
    best3_slots = {slot: thirds[slot_map[slot]] for slot in slot_map if slot_map[slot] in thirds}

    BEST3_MATCH_TO_SLOTCOL = {79: "1A", 85: "1B", 82: "1D", 75: "1E",
                              81: "1G", 78: "1I", 88: "1K", 80: "1L"}

    winners: dict[int, str] = {}
    losers:  dict[int, str] = {}

    def resolve(slot_str: str, mid: int):
        s = slot_str.strip()
        if s.startswith("Winner Group "):    return standings[s[-1]][0]
        if s.startswith("Runner-up Group "): return standings[s[-1]][1]
        if s.startswith("Best 3rd"):         return best3_slots.get(BEST3_MATCH_TO_SLOTCOL.get(mid))
        if s.startswith("Winner Match "):    return winners.get(int(s.split()[-1]))
        if s.startswith("Loser Match "):     return losers.get(int(s.split()[-1]))
        return None

    rows = []
    for _, r in slots_df.iterrows():
        mid, rnd = int(r["match_id"]), r["round"]
        home = resolve(r["slot_home"], mid)
        away = resolve(r["slot_away"], mid)
        rec = {"match_id": mid, "round": rnd,
               "date_utc": r["date_utc"], "venue": r["venue"],
               "slot_home": r["slot_home"], "slot_away": r["slot_away"],
               "home_team": home, "away_team": away,
               "pred_winner": None, "p_home_win": None, "p_winner": None}
        if home and away:
            p_home, p_away = match_win_prob(home, away, strengths)
            winner = home if p_home >= p_away else away
            winners[mid] = winner
            losers[mid]  = away if winner == home else home
            rec.update(pred_winner=winner,
                       p_home_win=round(p_home, 4),
                       p_winner=round(max(p_home, p_away), 4))
        rows.append(rec)

    return pd.DataFrame(rows)

# main
def main():
    np.random.seed(42)
    random.seed(42)

    log.info("=" * 60)
    log.info("  2026 FIFA World Cup Monte Carlo Simulation")
    log.info(f"  {N_SIMS:,} iterations | trained Poisson model | AGE_TILT={AGE_TILT}")
    log.info("=" * 60)

    data      = load_all()
    strengths = build_strengths(data)
    players_meta, team_share = build_player_shares(list(data["groups"].keys()))

    # Show top / bottom strength rankings
    log.info("\nTop 10 teams by composite attack score:")
    sorted_teams = sorted(strengths.items(), key=lambda kv: kv[1]["att"], reverse=True)
    for i, (t, s) in enumerate(sorted_teams[:10], 1):
        log.info(f"  {i:2d}. {t:<25s}  att={s['att']:.3f}  def={s['def']:.3f}  elo={s['elo']:.0f}")

    log.info("\nBottom 10 teams by composite attack score:")
    for i, (t, s) in enumerate(sorted_teams[-10:], 1):
        log.info(f"  {i:2d}. {t:<25s}  att={s['att']:.3f}  def={s['def']:.3f}  elo={s['elo']:.0f}")

    # Run Monte Carlo
    summary, bracket, player_scoring = run_monte_carlo(
        data, strengths, players_meta, team_share, N_SIMS)

    # Save
    out_path = OUTPUT_DIR / "wc2026_champion_probabilities.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"\nResults saved → {out_path}")

    player_path = OUTPUT_DIR / "wc2026_player_scoring.csv"
    player_scoring.to_csv(player_path, index=False, encoding="utf-8")
    log.info(f"Player scoring → {player_path}")
    if len(player_scoring):
        log.info("\nGolden Boot projection (top 12 by expected goals):")
        log.info(f"  {'Player':<24} {'Team':<16} {'xGoals':>6} {'±sd':>5} {'Boot%':>6}")
        for _, r in player_scoring.head(12).iterrows():
            log.info(f"  {r['player']:<24} {r['team']:<16} {r['exp_goals']:>6.2f} "
                     f"{r['sd_goals']:>5.2f} {r['p_golden_boot']:>6.1%}")

    bracket_path = OUTPUT_DIR / "wc2026_bracket_predictions.csv"
    # Exclude win_counts dict column for clean CSV
    bracket[["match_id","round","pred_winner","p_win"]].to_csv(bracket_path, index=False, encoding="utf-8")
    log.info(f"Bracket saved  → {bracket_path}")

    # Deterministic full predicted bracket (both teams in every match → champion)
    full_bracket = predict_bracket(data, strengths, summary)
    full_path = OUTPUT_DIR / "wc2026_bracket_full.csv"
    full_bracket.to_csv(full_path, index=False, encoding="utf-8")
    log.info(f"Full bracket   → {full_path}")

    champ_row = full_bracket[full_bracket["round"] == "Final"].iloc[0]
    log.info("\n" + "=" * 70)
    log.info("PREDICTED BRACKET (favourite advances):")
    log.info("=" * 70)
    for _, r in full_bracket.iterrows():
        if r["home_team"] and r["away_team"]:
            mark_h = "►" if r["pred_winner"] == r["home_team"] else " "
            mark_a = "►" if r["pred_winner"] == r["away_team"] else " "
            log.info(f"  [{int(r['match_id'])}] {r['round']:<18} "
                     f"{mark_h}{r['home_team']:<22} vs {mark_a}{r['away_team']:<22} "
                     f"→ {r['pred_winner']} ({r['p_winner']:.0%})")
    log.info(f"\n  PREDICTED CHAMPION: {champ_row['pred_winner']}")

    # Print top 20
    log.info("\n" + "=" * 70)
    log.info(f"{'Rk':<4} {'Team':<25} {'Group':<6} {'ELO':<6} "
             f"{'Qualify':>8} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>6} {'Win%':>7}")
    log.info("-" * 70)
    for _, row in summary.head(20).iterrows():
        log.info(
            f"{int(row['rank']):<4} {row['team']:<25} {row['group']:<6} {int(row['elo']):<6} "
            f"{row['p_qualify']:>8.1%} {row['p_round_of_16']:>6.1%} "
            f"{row['p_quarter_final']:>6.1%} {row['p_semi_final']:>6.1%} "
            f"{row['p_final']:>6.1%} {row['p_champion']:>7.1%}"
        )

    log.info("\n" + "=" * 70)
    log.info("GROUP STAGE — Expected finishers by group:")
    log.info("=" * 70)
    # Show group predictions (most likely qualifier per group)
    for grp in sorted(data["group_teams"].keys()):
        grp_teams = [row for _, row in summary.iterrows() if row["group"] == grp]
        grp_teams_sorted = sorted(grp_teams, key=lambda r: r["p_qualify"], reverse=True)
        log.info(f"  Group {grp}:")
        for r in grp_teams_sorted:
            log.info(f"    {r['team']:<25s} qualify={r['p_qualify']:.1%}  win={r['p_champion']:.1%}")

    log.info("\nDone.")

if __name__ == "__main__":
    main()
