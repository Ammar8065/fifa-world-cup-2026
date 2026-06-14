#!/usr/bin/env python3
"""
2026 FIFA World Cup Monte Carlo Simulation
==========================================
Data sources used:
  - Data/data/processed/poisson_goals_ours.pkl — fitted Poisson goals pipeline (train_model.py)
  - Data/data/processed/team_ratings_2026.csv   — per-team elo / off / def / age (train_model.py)
  - Data/data/raw/groups.csv                     — 2026 WC group assignments
  - Data/data/raw/knockout_slots.csv             — Official 2026 bracket
  - Data/data/raw/fw26_best_third_placed_combinations.csv — 3rd-place slot rules

Model
-----
  A *trained* Poisson goals model (see train_model.py), not hand-tuned weights:
      Pipeline( StandardScaler -> PoissonRegressor )
      features = [elo_diff, off_rating_a, def_rating_b, is_home]
  off_rating / def_rating are 7-game Elo-adjusted form ("Elo-adjusted goals"):
      off = Σ_last7 goals_scored  × (opp_elo / 1500)
      def = Σ_last7 goals_against × (1500 / opp_elo)

  λ_A vs B = model.predict([elo_A-elo_B, off_A, def_B, is_home=0])
           × optional small squad-age tilt
  goals drawn from Poisson(λ)

  Knockout ties → 30 min extra time (0.5× base goals) → penalties (ELO-weighted coin flip)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import time
import random
import logging
from math import exp, factorial
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import json
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────
N_SIMS       = 10_000
ET_FACTOR    = 0.50     # extra-time reduces scoring rate to 50% of 90-min rate

# Trained model artifacts (produced by train_model.py + train_advanced.py)
MODEL_PATH   = "Data/data/processed/poisson_goals_ours.pkl"   # GLM-Poisson
RATINGS_PATH = "Data/data/processed/team_ratings_2026.csv"
DC_PATH      = "Data/data/processed/dc_ratings_2026.csv"      # Dixon-Coles attack/defence
XGB_PATH     = "Data/data/processed/xgb_goals.json"           # XGBoost (count:poisson)
ENS_PATH     = "Data/data/processed/ensemble.json"            # tuned blend weights + ρ
MODEL_FEATURES = ["elo_diff", "off_rating_a", "def_rating_b", "is_home"]
XGB_FEATURES   = ["elo_home", "elo_away", "elo_diff", "off_rating_a", "def_rating_b", "is_home"]

# Optional, small post-hoc squad-age tilt (peak ≈ 27). Set AGE_TILT=0 to disable.
AGE_TILT     = 0.010    # per year away from peak; multiplies a team's scoring λ
AGE_PEAK     = 27.0

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

# ── Name canonicalisation ─────────────────────────────────────────────────────
# All names converge to the schedule.csv spellings
CANON = {
    # ELO / results.csv names
    "Bosnia and Herzegovina":        "Bosnia-Herzegovina",
    "DR Congo":                       "Congo DR",
    "Ivory Coast":                    "Côte d'Ivoire",
    "Iran":                           "IR Iran",
    "South Korea":                    "Korea Republic",
    "Turkey":                         "Türkiye",
    "Czech Republic":                 "Czechia",
    # FIFA ranking names
    "USA":                            "United States",
    "Cabo Verde":                     "Cape Verde",
    # groups.csv (uses same alternatives as results)
}

def canon(name: str) -> str:
    return CANON.get(str(name).strip(), str(name).strip())


# ── Load data ─────────────────────────────────────────────────────────────────
def load_all() -> dict:
    log.info("Loading data...")

    # Groups
    grp_df = pd.read_csv("Data/data/raw/groups.csv")
    grp_df["team"] = grp_df["team"].apply(canon)
    groups = dict(zip(grp_df["team"], grp_df["group"]))
    group_teams: dict[str, list[str]] = defaultdict(list)
    for team, grp in groups.items():
        group_teams[grp].append(team)

    # Knockout slots
    slots_df = pd.read_csv("Data/data/raw/knockout_slots.csv")

    # Best-3rd combinations
    b3_df = pd.read_csv("Data/data/raw/fw26_best_third_placed_combinations.csv")
    slot_cols = [c for c in b3_df.columns if c != "Option"]
    best3_lookup: dict[frozenset, dict[str, str]] = {}
    for _, row in b3_df.iterrows():
        mapping = {col: row[col][1] for col in slot_cols}  # '3E' → 'E'
        key = frozenset(mapping.values())
        best3_lookup[key] = mapping

    log.info(f"  Groups:          {len(groups)} teams")
    log.info(f"  Best-3rd lookup: {len(best3_lookup)} combinations")

    return dict(groups=groups, group_teams=dict(group_teams),
                slots_df=slots_df, best3_lookup=best3_lookup)


# ── Build team strengths + λ lookup from the trained model ──────────────────────
def build_strengths(data: dict) -> dict[str, dict]:
    """Load the trained ENSEMBLE (GLM-Poisson + XGBoost + Dixon-Coles, with the
    holdout-tuned weights from train_advanced.py) + 2026 team ratings, precompute
    every pairwise expected-goals value, and store it on each team for fast lookup."""
    teams = list(data["groups"].keys())

    pipe = joblib.load(MODEL_PATH)                       # GLM-Poisson
    booster = xgb.Booster(); booster.load_model(XGB_PATH)  # XGBoost (count:poisson)
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

    # squad-age tilt: teams further from the peak age score a touch less (small effect)
    age_factor = {t: float(np.exp(-AGE_TILT * abs(age[t] - AGE_PEAK))) for t in teams}

    def ens_lambda(rows_glm, rows_xgb, dc_lams):
        """Blend the three models' λ predictions with the tuned weights."""
        out = np.zeros(len(dc_lams))
        if w_glm: out += w_glm * pipe.predict(pd.DataFrame(rows_glm, columns=MODEL_FEATURES))
        if w_xgb: out += w_xgb * booster.predict(xgb.DMatrix(pd.DataFrame(rows_xgb, columns=XGB_FEATURES)))
        if w_dc:  out += w_dc  * np.asarray(dc_lams)
        return out

    # Vectorised prediction for all ordered team pairs (a scoring vs b, neutral venue)
    pairs = [(a, b) for a in teams for b in teams if a != b]
    glm_rows = [[elo[a] - elo[b], off[a], dfn[b], 0] for a, b in pairs]
    xgb_rows = [[elo[a], elo[b], elo[a] - elo[b], off[a], dfn[b], 0] for a, b in pairs]
    dc_lams  = [float(np.exp(dc_att[a] - dc_def[b])) for a, b in pairs]   # neutral venue
    preds = ens_lambda(glm_rows, xgb_rows, dc_lams)

    lam: dict[str, dict[str, float]] = {t: {} for t in teams}
    for (a, b), p in zip(pairs, preds):
        lam[a][b] = float(p) * age_factor[a]

    # Interpretable ratings in REAL GOAL UNITS: expected goals for / against per
    # match versus a benchmark "average WC opponent" — now through the full ENSEMBLE.
    # xgf higher = better attack; xga LOWER = better defense.
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

    # Normalised 0–1 versions (higher = better for BOTH) for heat shading / radar.
    def norm01(d, invert=False):
        vals = list(d.values()); mn, mx = min(vals), max(vals)
        rng = (mx - mn) or 1.0
        return {t: (1 - (v - mn) / rng) if invert else (v - mn) / rng for t, v in d.items()}
    att_score = norm01(xgf)
    def_score = norm01(xga, invert=True)

    strengths = {}
    for t in teams:
        strengths[t] = {
            "att": off[t],          # raw offensive rating (7-game Elo-adj form + firepower)
            "def": dfn[t],          # raw defensive rating (lower = better)
            "xgf": xgf[t],          # expected goals FOR / match vs avg opponent
            "xga": xga[t],          # expected goals AGAINST / match vs avg opponent (lower better)
            "att_score": att_score[t],   # normalised 0–1, higher = better
            "def_score": def_score[t],   # normalised 0–1, higher = better
            "elo": elo[t],
            "age": age[t],
            "lam": lam[t],          # {opponent: expected goals vs that opponent}
        }
    return strengths


# ── Match simulation ──────────────────────────────────────────────────────────
def expected_goals(team_a: str, team_b: str, strengths: dict) -> tuple[float, float]:
    # Precomputed by the trained Poisson model in build_strengths()
    lam_a = strengths[team_a]["lam"][team_b]
    lam_b = strengths[team_b]["lam"][team_a]
    return lam_a, lam_b


def sim_match(team_a: str, team_b: str, strengths: dict,
              knockout: bool = False) -> tuple[int, int, bool]:
    """Return (goals_a, goals_b, penalties_used).
    If knockout and draw after 90 min, play 30-min ET then penalties.
    """
    lam_a, lam_b = expected_goals(team_a, team_b, strengths)
    ga = np.random.poisson(lam_a)
    gb = np.random.poisson(lam_b)

    if not knockout or ga != gb:
        return ga, gb, False

    # Extra time (30 min ≈ ET_FACTOR × 90-min rate)
    ga += np.random.poisson(lam_a * ET_FACTOR)
    gb += np.random.poisson(lam_b * ET_FACTOR)

    if ga != gb:
        return ga, gb, True

    # Penalties — slight ELO-weighted advantage
    elo_a = strengths[team_a]["elo"]
    elo_b = strengths[team_b]["elo"]
    p_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 800))  # muted for penalties
    if np.random.random() < p_a:
        return ga + 1, gb, True     # A wins on pens
    else:
        return ga, gb + 1, True     # B wins on pens


# ── Group stage ───────────────────────────────────────────────────────────────
def sim_group(teams: list[str], strengths: dict) -> list[dict]:
    """Simulate one group. Returns list of team-row dicts sorted by standing."""
    records = {t: {"team": t, "pts": 0, "gf": 0, "ga": 0, "gd": 0,
                   "w": 0, "d": 0, "l": 0} for t in teams}
    h2h_pts: dict[tuple[str, str], int] = {(a, b): 0 for a in teams for b in teams if a != b}
    h2h_gd:  dict[tuple[str, str], int] = {(a, b): 0 for a in teams for b in teams if a != b}

    for a, b in combinations(teams, 2):
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

    # Sort: pts → gd → gf → h2h pts → h2h gd → ELO
    def sort_key(t):
        r = records[t]
        others = [o for o in teams if o != t]
        h2h_p = sum(h2h_pts[(t, o)] for o in others)
        h2h_g = sum(h2h_gd[(t, o)]  for o in others)
        return (r["pts"], r["gd"], r["gf"], h2h_p, h2h_g, strengths[t]["elo"])

    sorted_teams = sorted(teams, key=sort_key, reverse=True)
    return [{"pos": i + 1, **records[t]} for i, t in enumerate(sorted_teams)]


# ── Best 3rd-place ─────────────────────────────────────────────────────────────
def pick_best_third(third_teams: dict[str, dict],
                    strengths: dict,
                    best3_lookup: dict) -> dict[str, str]:
    """
    third_teams: {group_letter: row_dict} (all 12 groups' 3rd-place teams)
    Returns: {slot_col: team_name} for the 8 best 3rd teams placed in bracket.
    """
    # Rank 12 third teams: pts → gd → gf → ELO
    ranked = sorted(
        third_teams.items(),
        key=lambda kv: (kv[1]["pts"], kv[1]["gd"], kv[1]["gf"],
                        strengths[kv[1]["team"]]["elo"]),
        reverse=True
    )
    # Top 8 advance
    advancing = ranked[:8]
    adv_groups = frozenset(g for g, _ in advancing)  # 8 group letters

    # Look up slot assignment
    slot_map = best3_lookup.get(adv_groups)
    if slot_map is None:
        # Fallback: pick the alphabetically closest key (shouldn't happen for valid combos)
        slot_map = next(iter(best3_lookup.values()))

    # Map slot → team name
    group_to_team = {g: r["team"] for g, r in advancing}
    return {slot: group_to_team[slot_map[slot]] for slot in slot_map
            if slot_map[slot] in group_to_team}


# ── Knockout stage ─────────────────────────────────────────────────────────────
def sim_knockout(slots_df: pd.DataFrame,
                 group_results: dict[str, list[dict]],  # {grp: sorted_rows}
                 best3_slots: dict[str, str],           # {slot_col: team}
                 strengths: dict) -> dict[str, str]:
    """
    Returns: {match_id_str: winner_name} for all knockout matches.
    Also returns 'champion' key.
    """
    # Build slot resolution
    def resolve(slot_str: str) -> str | None:
        s = slot_str.strip()
        if s.startswith("Winner Group "):
            g = s[-1]
            return group_results[g][0]["team"]
        if s.startswith("Runner-up Group "):
            g = s[-1]
            return group_results[g][1]["team"]
        if s.startswith("Best 3rd"):
            # e.g. "Best 3rd (Groups A/B/C/D/F)" → best3_slots lookup
            # The right team is already placed via best3_slots
            return None  # will be filled below
        if s.startswith("Winner Match "):
            return None  # depends on earlier match
        if s.startswith("Loser Match "):
            return None
        return None

    # Pre-build best-3rd slot mapping using column names (1A, 1B, etc.)
    # Slot column → team already computed in best3_slots
    # We need to reverse-map "Best 3rd (Groups X/Y/Z)" in knockout_slots to a team.
    # Approach: the 8 best-3rd slots correspond to matches 75,78,79,80,81,82,85,88
    # and their slot descriptions. We map via the slot_col → team from best3_slots.
    BEST3_MATCH_TO_SLOTCOL = {
        79: "1A",  # Winner A vs Best 3rd
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

        gh, ga, _ = sim_match(home, away, strengths, knockout=True)
        winner = home if gh > ga else away
        loser  = away if gh > ga else home
        match_winners[mid] = winner
        match_losers[mid]  = loser

    # Final is the last match in the bracket (max match_id, round='Final')
    finals = slots_df[slots_df["round"] == "Final"]
    champion = match_winners.get(int(finals["match_id"].iloc[0])) if len(finals) > 0 else None

    return match_winners, champion


# ── Full tournament simulation ────────────────────────────────────────────────
def simulate_tournament(data: dict, strengths: dict) -> dict:
    group_teams  = data["group_teams"]
    slots_df     = data["slots_df"]
    best3_lookup = data["best3_lookup"]

    # --- Group stage ---
    group_results: dict[str, list[dict]] = {}
    third_teams:   dict[str, dict] = {}

    for grp, teams in group_teams.items():
        standing = sim_group(teams, strengths)
        group_results[grp] = standing
        third_teams[grp]   = standing[2]  # 3rd place

    # --- Best 3rd ---
    best3_slots = pick_best_third(third_teams, strengths, best3_lookup)

    # --- Knockout ---
    match_winners, champion = sim_knockout(
        slots_df, group_results, best3_slots, strengths
    )

    return {
        "group_results": group_results,
        "best3_slots":   best3_slots,
        "match_winners": match_winners,
        "champion":      champion,
    }


# ── Monte Carlo ───────────────────────────────────────────────────────────────
ROUND_ORDER = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]

def run_monte_carlo(data: dict, strengths: dict, n_sims: int = N_SIMS) -> tuple:
    all_teams = list(data["groups"].keys())
    slots_df  = data["slots_df"]

    # Track counts per team per outcome
    counts = {t: defaultdict(int) for t in all_teams}
    # Track group position distribution {team: {1:n, 2:n, 3:n, 4:n}}
    pos_dist = {t: defaultdict(int) for t in all_teams}
    # Track per-match winner counts {match_id: {team: n}}
    match_win_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

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

    return summary_df, bracket_df


# ── Deterministic "favourite advances" bracket ────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    random.seed(42)

    log.info("=" * 60)
    log.info("  2026 FIFA World Cup Monte Carlo Simulation")
    log.info(f"  {N_SIMS:,} iterations | trained Poisson model | AGE_TILT={AGE_TILT}")
    log.info("=" * 60)

    data      = load_all()
    strengths = build_strengths(data)

    # Show top / bottom strength rankings
    log.info("\nTop 10 teams by composite attack score:")
    sorted_teams = sorted(strengths.items(), key=lambda kv: kv[1]["att"], reverse=True)
    for i, (t, s) in enumerate(sorted_teams[:10], 1):
        log.info(f"  {i:2d}. {t:<25s}  att={s['att']:.3f}  def={s['def']:.3f}  elo={s['elo']:.0f}")

    log.info("\nBottom 10 teams by composite attack score:")
    for i, (t, s) in enumerate(sorted_teams[-10:], 1):
        log.info(f"  {i:2d}. {t:<25s}  att={s['att']:.3f}  def={s['def']:.3f}  elo={s['elo']:.0f}")

    # Run Monte Carlo
    summary, bracket = run_monte_carlo(data, strengths, N_SIMS)

    # Save
    out_path = OUTPUT_DIR / "wc2026_champion_probabilities.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"\nResults saved → {out_path}")

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
