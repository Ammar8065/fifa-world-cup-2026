#!/usr/bin/env python3
"""Train XGBoost + Dixon-Coles, blend into an ensemble, and evaluate calibration."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
import xgboost as xgb

import train_model as tm   # reuse feature engineering (build_long, ratings, canon …)

PROC = Path("Data/data/processed")
HOLDOUT_FROM = "2022-01-01"
DC_WINDOW_FROM = "2014-01-01"      # recent window for current team strengths
HALFLIFE_YEARS = 8.0
MAXG = 10                          # truncate the scoreline grid at 10 goals/side
XGB_FEATURES = ["elo_home", "elo_away", "elo_diff", "off_rating_a", "def_rating_b", "is_home"]

rng = np.random.default_rng(42)

# ════════════════════════════════════════════════════════════════════════════
#  Scoreline → W/D/L probabilities
# ════════════════════════════════════════════════════════════════════════════
def _dc_tau(i, j, lh, la, rho):
    """Dixon-Coles low-score dependence correction (matrix form)."""
    t = np.ones_like(lh)
    t = np.where((i == 0) & (j == 0), 1 - lh * la * rho, t)
    t = np.where((i == 0) & (j == 1), 1 + lh * rho, t)
    t = np.where((i == 1) & (j == 0), 1 + la * rho, t)
    t = np.where((i == 1) & (j == 1), 1 - rho, t)
    return t

def wdl_from_lambda(lh, la, rho=0.0):
    """Vectorised W/D/L probabilities from home/away λ arrays (+ optional ρ)."""
    lh = np.clip(np.asarray(lh, float), 1e-6, None)
    la = np.clip(np.asarray(la, float), 1e-6, None)
    ph = np.empty(len(lh)); pd_ = np.empty(len(lh)); pa = np.empty(len(lh))
    ks = np.arange(MAXG + 1)
    pois_h = poisson.pmf(ks[None, :], lh[:, None])   # (n, MAXG+1)
    pois_a = poisson.pmf(ks[None, :], la[:, None])
    I, J = np.meshgrid(ks, ks, indexing="ij")        # (G+1, G+1)
    for n in range(len(lh)):
        joint = np.outer(pois_h[n], pois_a[n])
        if rho:
            joint = joint * _dc_tau(I, J, lh[n], la[n], rho)
        joint = np.clip(joint, 0, None)
        joint /= joint.sum()
        ph[n] = np.tril(joint, -1).sum()
        pa[n] = np.triu(joint, 1).sum()
        pd_[n] = np.trace(joint)
    return np.clip(np.c_[ph, pd_, pa], 1e-9, None)

# ════════════════════════════════════════════════════════════════════════════
#  Dixon-Coles maximum-likelihood fit
# ════════════════════════════════════════════════════════════════════════════
def fit_dixon_coles(matches, halflife=HALFLIFE_YEARS, l2=0.02):
    """
    matches: DataFrame with home_team, away_team, home_score, away_score, date.
    Returns dict: attack{team}, defence{team}, home_adv, rho, teams.
    """
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    idx = {t: k for k, t in enumerate(teams)}
    n = len(teams)
    hi = matches["home_team"].map(idx).to_numpy()
    ai = matches["away_team"].map(idx).to_numpy()
    hg = matches["home_score"].to_numpy(float)
    ag = matches["away_score"].to_numpy(float)
    ref = matches["date"].max()
    yrs = (ref - matches["date"]).dt.days.to_numpy() / 365.25
    w = np.power(0.5, yrs / halflife)

    def unpack(p):
        att = p[:n]; dfn = p[n:2 * n]; gamma = p[2 * n]; rho = p[2 * n + 1]
        return att, dfn, gamma, rho

    def nll(p):
        att, dfn, gamma, rho = unpack(p)
        lh = np.exp(att[hi] - dfn[ai] + gamma)
        la = np.exp(att[ai] - dfn[hi])
        # log Poisson for both sides
        ll = (hg * np.log(lh) - lh) + (ag * np.log(la) - la)
        # Dixon-Coles τ correction for the four low-score cells
        tau = _dc_tau(hg, ag, lh, la, rho)
        tau = np.clip(tau, 1e-6, None)
        ll = ll + np.log(tau)
        pen = l2 * (att @ att + dfn @ dfn)            # ridge → identifiability + shrinkage
        return -np.sum(w * ll) + pen

    p0 = np.concatenate([np.zeros(2 * n), [0.25, -0.05]])
    bnds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.2, 0.2)]
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bnds,
                   options={"maxiter": 400, "maxfun": 60000})
    att, dfn, gamma, rho = unpack(res.x)
    att = att - att.mean()                            # centre for interpretability
    dfn = dfn - dfn.mean()
    return {"attack": dict(zip(teams, att)), "defence": dict(zip(teams, dfn)),
            "home_adv": float(gamma), "rho": float(rho), "teams": teams,
            "converged": bool(res.success)}

def dc_lambda(dc, home, away, neutral=True):
    """λ_home, λ_away for a single matchup from a fitted DC model."""
    a, d = dc["attack"], dc["defence"]
    aa, ad = a.get(home, 0.0), d.get(home, 0.0)
    ba, bd = a.get(away, 0.0), d.get(away, 0.0)
    g = 0.0 if neutral else dc["home_adv"]
    return np.exp(aa - bd + g), np.exp(ba - ad)

# ════════════════════════════════════════════════════════════════════════════
#  Calibration metrics
# ════════════════════════════════════════════════════════════════════════════
def metrics(probs, actual):
    """probs: (n,3) [H,D,A]; actual: array of 0/1/2. Returns dict of metrics."""
    probs = np.clip(probs, 1e-9, 1); probs /= probs.sum(1, keepdims=True)
    n = len(actual)
    onehot = np.eye(3)[actual]
    acc = float((probs.argmax(1) == actual).mean())
    logloss = float(-np.log(probs[np.arange(n), actual]).mean())
    brier = float(((probs - onehot) ** 2).sum(1).mean())
    # ECE on the top-class confidence (15 bins)
    conf = probs.max(1); pred = probs.argmax(1); hit = (pred == actual).astype(float)
    bins = np.linspace(0, 1, 16); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return {"accuracy": round(acc, 4), "log_loss": round(logloss, 4),
            "brier": round(brier, 4), "ece": round(float(ece), 4), "n": int(n)}

def reliability_homewin(p_home, actual, nbins=10):
    """Reliability bins for the home-win probability (predicted vs observed)."""
    y = (actual == 0).astype(float)
    edges = np.linspace(0, 1, nbins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_home > lo) & (p_home <= hi) if hi < 1 else (p_home > lo) & (p_home <= hi + 1e-9)
        if m.sum() >= 5:
            out.append({"p_pred": round(float(p_home[m].mean()), 4),
                        "p_obs": round(float(y[m].mean()), 4), "n": int(m.sum())})
    return out

# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  Advanced model suite — Dixon-Coles · XGBoost · Ensemble · Calibration")
    print("=" * 72)

    # ---- data & shared feature frame (reuse train_model's engineering) --------
    df = pd.read_csv("Data/data/processed/training_df.csv", parse_dates=["date"])
    df["home_team"] = df["home_team"].apply(tm.canon)
    df["away_team"] = df["away_team"].apply(tm.canon)
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[(df["home_elo_before"] > 0) & (df["away_elo_before"] > 0)]
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index

    long = tm.add_form_ratings(tm.build_long(df))
    # assemble the per-(team,match) model frame, KEEPING match_id/team/Elo (unlike
    # tm.assemble_training, which trims to FEATURES+goals+date for the GLM)
    opp = long[["match_id", "team", "def_rating"]].rename(
        columns={"team": "opp", "def_rating": "opp_def_rating"})
    frame = long.merge(opp, on=["match_id", "opp"], how="left")
    frame["elo_diff"] = frame["team_elo"] - frame["opp_elo"]
    frame["off_rating_a"] = frame["off_rating"]
    frame["def_rating_b"] = frame["opp_def_rating"]
    frame["goals"] = frame["gf"]
    frame["elo_home"] = frame["team_elo"]
    frame["elo_away"] = frame["opp_elo"]
    frame = frame.dropna(subset=tm.FEATURES + ["goals", "date", "elo_home", "elo_away"])
    frame = frame[frame["date"] >= tm.TRAIN_CUTOFF].reset_index(drop=True)

    train = frame[frame["date"] < HOLDOUT_FROM].copy()
    test = frame[frame["date"] >= HOLDOUT_FROM].copy()
    print(f"Train rows: {len(train):,}  |  Holdout (2022+): {len(test):,}")

    # ---- Model 1: GLM-Poisson (reload the deployed pipeline) -------------------
    glm = tm.joblib.load(PROC / "poisson_goals_ours.pkl")
    frame["lam_glm"] = glm.predict(frame[tm.FEATURES])

    # ---- Model 2: XGBoost (Poisson objective) ---------------------------------
    wtr = np.power(0.5, ((train["date"].max() - train["date"]).dt.days / 365.25) / HALFLIFE_YEARS)
    dtrain = xgb.DMatrix(train[XGB_FEATURES], label=train["goals"], weight=wtr.to_numpy())
    xgb_params = {"objective": "count:poisson", "eta": 0.05, "max_depth": 4,
                  "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 5,
                  "lambda": 1.0, "seed": 42, "verbosity": 0}
    booster = xgb.train(xgb_params, dtrain, num_boost_round=350)
    frame["lam_xgb"] = booster.predict(xgb.DMatrix(frame[XGB_FEATURES]))

    # ---- Model 3: Dixon-Coles (fit on TRAIN-period recent window) -------------
    dc_train_matches = df[(df["date"] >= DC_WINDOW_FROM) & (df["date"] < HOLDOUT_FROM)]
    print(f"Fitting Dixon-Coles on {len(dc_train_matches):,} matches "
          f"({DC_WINDOW_FROM[:4]}–2021)…")
    dc = fit_dixon_coles(dc_train_matches)
    print(f"  DC home advantage γ={dc['home_adv']:.3f}  ρ={dc['rho']:.3f}  "
          f"converged={dc['converged']}  teams={len(dc['teams'])}")

    # ---- Pair the two team-rows of each match → per-match home/away λ ----------
    def pair_lambdas(sub, lam_col):
        """Return per-match arrays using the HOME row's λ for home, AWAY row's for away."""
        # the 'home' row is the one with is_home==1 OR (for neutral) first by team order;
        # simplest: rejoin to df to get home/away identity
        h = sub.merge(df[["match_id", "home_team"]], on="match_id")
        h = h[h["team"] == h["home_team"]][["match_id", lam_col]].rename(columns={lam_col: "lh"})
        a = sub.merge(df[["match_id", "away_team"]], on="match_id")
        a = a[a["team"] == a["away_team"]][["match_id", lam_col]].rename(columns={lam_col: "la"})
        return h.merge(a, on="match_id")

    test_ids = test["match_id"].unique()
    base = df[df["match_id"].isin(test_ids)][["match_id", "home_team", "away_team",
                                              "home_score", "away_score"]].copy()
    base["actual"] = np.where(base["home_score"] > base["away_score"], 0,
                              np.where(base["home_score"] < base["away_score"], 2, 1))

    # GLM & XGB λ per match
    for lam_col in ["lam_glm", "lam_xgb"]:
        pl = pair_lambdas(frame[frame["match_id"].isin(test_ids)], lam_col)
        base = base.merge(pl.rename(columns={"lh": f"{lam_col}_h", "la": f"{lam_col}_a"}),
                          on="match_id", how="left")
    # DC λ per match (from team identity; WC-style neutral for fairness on holdout
    # would drop home edge — but these are real fixtures, so keep venue via home_adv)
    dh, da = [], []
    for _, r in base.iterrows():
        lh, la = dc_lambda(dc, r["home_team"], r["away_team"], neutral=False)
        dh.append(lh); da.append(la)
    base["lam_dc_h"], base["lam_dc_a"] = dh, da
    base = base.dropna().reset_index(drop=True)
    actual = base["actual"].to_numpy()

    # ---- W/D/L probabilities per model ----------------------------------------
    P = {}
    P["GLM-Poisson"] = wdl_from_lambda(base["lam_glm_h"], base["lam_glm_a"])
    P["XGBoost"] = wdl_from_lambda(base["lam_xgb_h"], base["lam_xgb_a"])
    P["Dixon-Coles"] = wdl_from_lambda(base["lam_dc_h"], base["lam_dc_a"], rho=dc["rho"])

    # ---- Ensemble: tune λ weights on the TRAIN period (no holdout leakage) -----
    tr_ids = train["match_id"].unique()
    tb = df[df["match_id"].isin(tr_ids)][["match_id", "home_team", "away_team",
                                          "home_score", "away_score"]].copy()
    tb["actual"] = np.where(tb["home_score"] > tb["away_score"], 0,
                            np.where(tb["home_score"] < tb["away_score"], 2, 1))
    for lam_col in ["lam_glm", "lam_xgb"]:
        pl = pair_lambdas(frame[frame["match_id"].isin(tr_ids)], lam_col)
        tb = tb.merge(pl.rename(columns={"lh": f"{lam_col}_h", "la": f"{lam_col}_a"}),
                      on="match_id", how="left")
    tdh, tda = [], []
    for _, r in tb.iterrows():
        lh, la = dc_lambda(dc, r["home_team"], r["away_team"], neutral=False)
        tdh.append(lh); tda.append(la)
    tb["lam_dc_h"], tb["lam_dc_a"] = tdh, tda
    tb = tb.dropna().reset_index(drop=True)
    ta = tb["actual"].to_numpy()

    def ens_ll(wts, frm, act):
        wts = np.clip(wts, 0, None); wts = wts / wts.sum()
        lh = (wts[0] * frm["lam_glm_h"] + wts[1] * frm["lam_xgb_h"] + wts[2] * frm["lam_dc_h"]).to_numpy()
        la = (wts[0] * frm["lam_glm_a"] + wts[1] * frm["lam_xgb_a"] + wts[2] * frm["lam_dc_a"]).to_numpy()
        p = wdl_from_lambda(lh, la, rho=dc["rho"])
        return -np.log(np.clip(p[np.arange(len(act)), act], 1e-9, 1)).mean()

    from itertools import product
    best, best_w = 1e9, np.array([1 / 3, 1 / 3, 1 / 3])
    for w in product(np.linspace(0, 1, 11), repeat=3):
        if abs(sum(w) - 1) > 1e-6:
            continue
        v = ens_ll(np.array(w), tb, ta)
        if v < best:
            best, best_w = v, np.array(w)
    print(f"  ensemble weights (tuned on train): GLM={best_w[0]:.2f} "
          f"XGB={best_w[1]:.2f} DC={best_w[2]:.2f}")
    lh = (best_w[0] * base["lam_glm_h"] + best_w[1] * base["lam_xgb_h"] + best_w[2] * base["lam_dc_h"]).to_numpy()
    la = (best_w[0] * base["lam_glm_a"] + best_w[1] * base["lam_xgb_a"] + best_w[2] * base["lam_dc_a"]).to_numpy()
    P["Ensemble"] = wdl_from_lambda(lh, la, rho=dc["rho"])

    # ---- Metrics + reliability -------------------------------------------------
    print("\n" + "-" * 72)
    print(f"  HOLDOUT (2022+, n={len(actual):,} matches) — lower log-loss/Brier/ECE = better")
    print("-" * 72)
    print(f"  {'model':<14}{'acc':>8}{'logloss':>10}{'brier':>9}{'ece':>8}")
    eval_out = {"holdout_from": HOLDOUT_FROM, "n_matches": int(len(actual)), "models": {}}
    for name, p in P.items():
        m = metrics(p, actual)
        m["reliability"] = reliability_homewin(p[:, 0], actual)
        eval_out["models"][name] = m
        print(f"  {name:<14}{m['accuracy']:>8.3f}{m['log_loss']:>10.4f}"
              f"{m['brier']:>9.4f}{m['ece']:>8.4f}")
    eval_out["ensemble_weights"] = {"GLM-Poisson": round(float(best_w[0]), 3),
                                    "XGBoost": round(float(best_w[1]), 3),
                                    "Dixon-Coles": round(float(best_w[2]), 3)}
    eval_out["dc"] = {"home_adv": round(dc["home_adv"], 4), "rho": round(dc["rho"], 4)}

    # ════════════════════════════════════════════════════════════════════════
    #  Refit DC on ALL recent data and export everything for the 2026 sim
    # ════════════════════════════════════════════════════════════════════════
    print("\nRefitting Dixon-Coles on all recent data for deployment…")
    dc_full = fit_dixon_coles(df[df["date"] >= DC_WINDOW_FROM])
    booster_full = xgb.train(
        xgb_params,
        xgb.DMatrix(frame[XGB_FEATURES], label=frame["goals"],
                    weight=np.power(0.5, ((frame["date"].max() - frame["date"]).dt.days / 365.25) / HALFLIFE_YEARS).to_numpy()),
        num_boost_round=350)

    groups = pd.read_csv("Data/data/raw/groups.csv"); groups["team"] = groups["team"].apply(tm.canon)
    wc_teams = groups["team"].tolist()
    am, dm = np.mean(list(dc_full["attack"].values())), np.mean(list(dc_full["defence"].values()))
    dc_rows = [{"team": t,
                "dc_attack": round(dc_full["attack"].get(t, am), 4),
                "dc_defence": round(dc_full["defence"].get(t, dm), 4)} for t in wc_teams]
    pd.DataFrame(dc_rows).to_csv(PROC / "dc_ratings_2026.csv", index=False, encoding="utf-8")
    booster_full.save_model(str(PROC / "xgb_goals.json"))
    (PROC / "ensemble.json").write_text(json.dumps({
        "weights": eval_out["ensemble_weights"],
        "rho": round(dc_full["rho"], 4), "home_adv": round(dc_full["home_adv"], 4),
        "xgb_features": XGB_FEATURES,
    }, indent=2), encoding="utf-8")
    (PROC / "model_eval.json").write_text(json.dumps(eval_out, indent=2), encoding="utf-8")
    print(f"  deploy DC: γ={dc_full['home_adv']:.3f} ρ={dc_full['rho']:.3f}")
    print(f"\nSaved → dc_ratings_2026.csv · xgb_goals.json · ensemble.json · model_eval.json")
    print("Done.")

if __name__ == "__main__":
    main()
