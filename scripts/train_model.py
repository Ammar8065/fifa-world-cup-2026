#!/usr/bin/env python3
"""Fit the Poisson goals GLM and 2026 team ratings from historical matches."""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

# config
ELO_AVG      = 1500.0          # constant from the "÷1500" formula (scaler absorbs it)
FORM_WINDOW  = 7               # last N games for the Elo-adjusted form ratings
TRAIN_CUTOFF = "1950-01-01"    # ignore very early, low-information football
HOLDOUT_FROM = "2022-01-01"    # evaluate generalisation on recent matches
RECENCY_HALFLIFE_YEARS = 8.0   # exponential down-weight of old matches
ALPHA        = 0.1             # L2 strength (same as reference pkl)

# Blend individual squad firepower into the 2026 offensive rating, so squads
# stacked with in-form attackers (e.g. France: Mbappé/Olise/Dembélé/Cherki/Doué)
# aren't underrated just because recent *team* results were modest.
#   off_rating = (1-INDIV_WEIGHT)·team_form  +  INDIV_WEIGHT·individual_firepower
# This is a deliberate MIX of whole-squad performance (off_form = the team's own
# international results) and individual output. Because this is an *international*
# tournament, individual firepower itself blends club xG with international xG
# rather than leaning on club form alone.
INDIV_WEIGHT  = 0.35       # weight on individual firepower vs whole-squad team form
INTL_SCALE    = 1.5        # international xG is fewer matches than a club season, so
                           # weight each intl-xG point a bit more; added on top of club
                           # xG (NOT z-scored separately, which would penalise the many
                           # teams with no recent-tournament data — e.g. Norway, Sweden)

FEATURES = ["elo_diff", "off_rating_a", "def_rating_b", "is_home"]

PROC = Path("Data/data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# name canonicalisation (kept in sync with simulate.py)
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
def canon(name: str) -> str:
    return CANON.get(str(name).strip(), str(name).strip())

# feature engineering
def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match) with the team's own goals + opponent context."""
    home = pd.DataFrame({
        "match_id": df["match_id"], "date": df["date"],
        "team": df["home_team"], "opp": df["away_team"],
        "gf": df["home_score"], "ga": df["away_score"],
        "team_elo": df["home_elo_before"], "opp_elo": df["away_elo_before"],
        "is_home": (~df["neutral"].astype(bool)).astype(int),
    })
    away = pd.DataFrame({
        "match_id": df["match_id"], "date": df["date"],
        "team": df["away_team"], "opp": df["home_team"],
        "gf": df["away_score"], "ga": df["home_score"],
        "team_elo": df["away_elo_before"], "opp_elo": df["home_elo_before"],
        "is_home": 0,                       # away side never gets home advantage
    })
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)
    return long

def add_form_ratings(long: pd.DataFrame) -> pd.DataFrame:
    """Rolling 7-game SUM of Elo-adjusted goals (offensive & defensive ratings)."""
    long["off_contrib"] = long["gf"] * (long["opp_elo"] / ELO_AVG)
    long["def_contrib"] = long["ga"] * (ELO_AVG / long["opp_elo"])

    grp = long.groupby("team", sort=False)
    # shift(1) so a match never uses its own result; require a full 7-game history
    long["off_rating"] = grp["off_contrib"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=FORM_WINDOW).sum())
    long["def_rating"] = grp["def_contrib"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=FORM_WINDOW).sum())
    return long

def assemble_training(long: pd.DataFrame) -> pd.DataFrame:
    """Attach the opponent's defensive rating and build the model frame."""
    opp = long[["match_id", "team", "def_rating"]].rename(
        columns={"team": "opp", "def_rating": "opp_def_rating"})
    m = long.merge(opp, on=["match_id", "opp"], how="left")

    m["elo_diff"]     = m["team_elo"] - m["opp_elo"]
    m["off_rating_a"] = m["off_rating"]
    m["def_rating_b"] = m["opp_def_rating"]
    m["goals"]        = m["gf"]

    cols = FEATURES + ["goals", "date"]
    m = m.dropna(subset=cols)
    m = m[m["date"] >= TRAIN_CUTOFF].reset_index(drop=True)
    return m

def recency_weights(dates: pd.Series) -> np.ndarray:
    ref = dates.max()
    years = (ref - dates).dt.days / 365.25
    return np.power(0.5, years / RECENCY_HALFLIFE_YEARS).to_numpy()

# evaluation helpers
def outcome_logloss(lam_h, lam_a, gh, ga, max_goals=12):
    """Average log-loss of the W/D/L outcome implied by the predicted lambdas."""
    from math import exp, factorial
    def pmf(k, lam):
        return exp(-lam) * lam ** k / factorial(k)
    ph_grid = np.array([[pmf(i, lh) for i in range(max_goals + 1)] for lh in lam_h])
    pa_grid = np.array([[pmf(j, la) for j in range(max_goals + 1)] for la in lam_a])
    tri = np.triu_indices(max_goals + 1)  # not used; explicit loop below for clarity
    losses, correct = [], 0
    for k in range(len(lam_h)):
        ph, pa = ph_grid[k], pa_grid[k]
        joint = np.outer(ph, pa)
        p_home = np.tril(joint, -1).sum()
        p_away = np.triu(joint, 1).sum()
        p_draw = np.trace(joint)
        s = p_home + p_away + p_draw
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s
        if gh[k] > ga[k]:   p = p_home; pred = "H"; act = "H"
        elif gh[k] < ga[k]: p = p_away; pred = "A"; act = "A"
        else:               p = p_draw; pred = "D"; act = "D"
        losses.append(-np.log(max(p, 1e-9)))
        guess = max([("H", p_home), ("D", p_draw), ("A", p_away)], key=lambda x: x[1])[0]
        correct += int(guess == act)
    return float(np.mean(losses)), correct / len(lam_h)

# squad firepower (current-season club xg)
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()

def _firstlast(n: str) -> str:
    t = n.split()
    return (t[0] + " " + t[-1]) if len(t) >= 2 else n

def _xg_lookup(path: str) -> dict[str, float]:
    """name → xG lookup (best xG row wins), keyed by normalized + first-last name."""
    df = pd.read_csv(path).sort_values("xg", ascending=False)
    df["_n"] = df["player"].map(_norm)
    lk: dict[str, float] = {}
    for _, r in df.iterrows():
        for key in (r["_n"], _firstlast(r["_n"])):
            lk.setdefault(key, float(r["xg"]))
    return lk

def squad_firepower(squads: pd.DataFrame, wc_teams: list[str]) -> pd.DataFrame:
    """Per-squad individual xG totals: current-season club xG AND international
    tournament xG. Mirrors build_dashboard.py's name matching."""
    club_lk = _xg_lookup("Data/scraped/player_xg_current.csv")  # club, current season
    intl_lk = _xg_lookup("Data/scraped/player_xg.csv")          # international tournaments

    club = {t: 0.0 for t in wc_teams}
    intl = {t: 0.0 for t in wc_teams}
    mc = mi = 0
    for _, p in squads.iterrows():
        if p["team"] not in club:
            continue
        n = _norm(p["name"])
        cx = club_lk.get(n, club_lk.get(_firstlast(n)))
        ix = intl_lk.get(n, intl_lk.get(_firstlast(n)))
        if cx is not None: club[p["team"]] += cx; mc += 1
        if ix is not None: intl[p["team"]] += ix; mi += 1
    print(f"  squad firepower: matched {mc} players (club xG), {mi} (international xG)")
    return pd.DataFrame({"team": wc_teams,
                         "club_xg":  [club[t] for t in wc_teams],
                         "intl_xg":  [intl[t] for t in wc_teams]})

# current-team ratings (for 2026 simulation)
def compute_2026_ratings(long: pd.DataFrame) -> pd.DataFrame:
    groups = pd.read_csv("Data/data/raw/groups.csv")
    groups["team"] = groups["team"].apply(canon)
    wc_teams = groups["team"].tolist()

    elo_df = pd.read_csv(str(PROC / "final_elo.csv"))
    elo_df["team"] = elo_df["team"].apply(canon)
    elo = dict(zip(elo_df["team"], elo_df["elo"]))

    squads = pd.read_csv("Data/scraped/squad_players.csv")
    squads["team"] = squads["team"].apply(canon)
    avg_age = squads.groupby("team")["age"].mean().to_dict()
    fire = squad_firepower(squads, wc_teams).set_index("team")

    rows = []
    for t in wc_teams:
        g = long[(long["team"] == t) & long["opp_elo"].notna() & (long["opp_elo"] > 0)]
        g = g.sort_values("date").tail(FORM_WINDOW)
        n = len(g)
        if n == 0:
            off = def_ = np.nan
        else:
            # scale to a full 7-game window if a team has fewer valid recent games
            off  = g["off_contrib"].sum() * FORM_WINDOW / n
            def_ = g["def_contrib"].sum() * FORM_WINDOW / n
        rows.append({
            "team": t, "group": groups.loc[groups.team == t, "group"].iloc[0],
            "elo": elo.get(t, np.nan),
            "off_form": off, "def_rating": def_,
            "club_xg": fire.loc[t, "club_xg"], "intl_xg": fire.loc[t, "intl_xg"],
            "avg_age": round(avg_age.get(t, np.nan), 2),
            "form_games": n,
        })
    out = pd.DataFrame(rows)
    for c in ["elo", "off_form", "def_rating", "avg_age", "club_xg", "intl_xg"]:
        out[c] = out[c].fillna(out[c].median())

    # Individual firepower = club xG + (scaled) international xG, summed BEFORE
    # standardising. A team with no recent-tournament data simply leans on its
    # club signal instead of being dragged below average for the missing intl xG.
    firepower = out["club_xg"] + INTL_SCALE * out["intl_xg"]
    # Map onto the team-form scale so the value the trained model sees stays
    # in-distribution, then mix whole-squad form with individual firepower.
    of_mean, of_std = out["off_form"].mean(), out["off_form"].std()
    fp_z = (firepower - firepower.mean()) / (firepower.std() or 1.0)
    mapped = fp_z * of_std + of_mean
    out["off_rating"] = ((1 - INDIV_WEIGHT) * out["off_form"]
                         + INDIV_WEIGHT * mapped).round(4)
    out["off_form"]   = out["off_form"].round(4)
    out["def_rating"] = out["def_rating"].round(4)
    out["club_xg"]    = out["club_xg"].round(1)
    out["intl_xg"]    = out["intl_xg"].round(1)

    cols = ["team", "group", "elo", "off_rating", "def_rating",
            "off_form", "club_xg", "intl_xg", "avg_age", "form_games"]
    return out[cols]

# main
def main():
    print("=" * 70)
    print("  Training Poisson goals model (Elo-adjusted offensive/defensive form)")
    print("=" * 70)

    df = pd.read_csv("Data/data/processed/training_df.csv", parse_dates=["date"])
    df["home_team"] = df["home_team"].apply(canon)
    df["away_team"] = df["away_team"].apply(canon)
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[(df["home_elo_before"] > 0) & (df["away_elo_before"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index
    print(f"Matches with valid scores + elo: {len(df):,}")

    long = add_form_ratings(build_long(df))
    frame = assemble_training(long)
    print(f"Training rows (full 7-game history): {len(frame):,}")
    print(f"  off_rating  mean={frame.off_rating_a.mean():.2f}  std={frame.off_rating_a.std():.2f}")
    print(f"  def_rating  mean={frame.def_rating_b.mean():.2f}  std={frame.def_rating_b.std():.2f}")

    train = frame[frame["date"] < HOLDOUT_FROM]
    test  = frame[frame["date"] >= HOLDOUT_FROM]
    print(f"Train: {len(train):,}  |  Holdout ({HOLDOUT_FROM}+): {len(test):,}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=ALPHA, max_iter=1000)),
    ])
    w = recency_weights(train["date"])
    pipe.fit(train[FEATURES], train["goals"], model__sample_weight=w)

    # ---- evaluation on holdout ----
    test = test.copy()
    test["pred"] = pipe.predict(test[FEATURES])
    dev = mean_poisson_deviance(test["goals"], np.clip(test["pred"], 1e-6, None))
    print(f"\nHoldout mean Poisson deviance: {dev:.4f}")
    print(f"Holdout predicted goals/team mean={test['pred'].mean():.3f}  actual={test['goals'].mean():.3f}")

    # outcome log-loss / accuracy: pair the two team-rows of each holdout match
    both = test.groupby("match_id").filter(lambda g: len(g) == 2)
    a_rows, b_rows = [], []
    for _, g in both.groupby("match_id"):
        r1, r2 = g.iloc[0], g.iloc[1]
        a_rows.append(r1); b_rows.append(r2)
    if a_rows:
        a = pd.DataFrame(a_rows); b = pd.DataFrame(b_rows)
        ll, acc = outcome_logloss(a["pred"].to_numpy(), b["pred"].to_numpy(),
                                  a["goals"].to_numpy(), b["goals"].to_numpy())
        print(f"Holdout W/D/L log-loss: {ll:.4f}   accuracy: {acc:.3f}  (n={len(a):,} matches)")

    # ---- refit on ALL data for deployment ----
    w_all = recency_weights(frame["date"])
    pipe.fit(frame[FEATURES], frame["goals"], model__sample_weight=w_all)

    scaler = pipe.named_steps["scaler"]
    model  = pipe.named_steps["model"]
    our = {
        "features": FEATURES,
        "scaler_mean":  [round(float(x), 5) for x in scaler.mean_],
        "scaler_scale": [round(float(x), 5) for x in scaler.scale_],
        "coef_standardized": [round(float(x), 5) for x in model.coef_],
        "intercept": round(float(model.intercept_), 5),
        "alpha": ALPHA, "form_window": FORM_WINDOW, "elo_avg": ELO_AVG,
        "n_train": int(len(frame)),
    }
    (PROC / "our_model.json").write_text(json.dumps(our, indent=2), encoding="utf-8")
    joblib.dump(pipe, PROC / "poisson_goals_ours.pkl")
    print(f"\nSaved → {PROC/'poisson_goals_ours.pkl'} and our_model.json")

    # ---- comparison with the reference pkl ----
    ref_path = PROC / "reference_model.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        print("\n" + "=" * 70)
        print("  OURS  vs  REFERENCE (poisson_goals.pkl)  — standardized coefficients")
        print("=" * 70)
        print(f"  {'feature':<18}{'ours':>12}{'reference':>14}")
        for feat in ["elo_diff", "off_rating_a", "def_rating_b"]:
            o = our["coef_standardized"][our["features"].index(feat)]
            r = ref["coef_standardized"][ref["features"].index(feat)] if feat in ref["features"] else None
            print(f"  {feat:<18}{o:>12.4f}{(f'{r:.4f}' if r is not None else 'n/a'):>14}")
        print(f"  {'is_home':<18}{our['coef_standardized'][our['features'].index('is_home')]:>12.4f}{'n/a (no venue)':>14}")
        print(f"  {'intercept':<18}{our['intercept']:>12.4f}{ref['intercept']:>14.4f}")
        print("  reference-only features: avg_age_diff, avg_caps_diff, europe_top5_diff")

    # ---- 2026 team ratings ----
    ratings = compute_2026_ratings(long)
    ratings.to_csv(PROC / "team_ratings_2026.csv", index=False, encoding="utf-8")
    print(f"\nSaved 2026 team ratings → {PROC/'team_ratings_2026.csv'}")
    show = ratings.sort_values("off_rating", ascending=False).head(12)
    extra = ratings[ratings.team.isin(["France", "Argentina"])]
    show = pd.concat([show, extra[~extra.team.isin(show.team)]])
    print(f"\n  (off_rating = {1-INDIV_WEIGHT:.0%} team form + {INDIV_WEIGHT:.0%} individual firepower;"
          f" firepower = club xG + {INTL_SCALE}× international xG)")
    print(f"\n{'team':<22}{'elo':>7}{'off_form':>10}{'club':>7}{'intl':>7}{'off_rtg':>9}{'def':>7}")
    for _, r in show.iterrows():
        print(f"{r.team:<22}{r.elo:>7.0f}{r.off_form:>10.2f}{r.club_xg:>7.0f}{r.intl_xg:>7.0f}{r.off_rating:>9.2f}{r.def_rating:>7.2f}")
    print("\nDone.")

if __name__ == "__main__":
    main()
