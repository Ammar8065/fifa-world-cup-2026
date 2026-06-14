# FIFA World Cup 2026 — Simulation & Prediction Dashboard

A data-driven forecast of the 2026 FIFA World Cup (48 teams, USA/Canada/Mexico).
The pipeline trains a goals model on ~49,000 historical internationals, runs a
10,000-iteration Monte Carlo of the real 2026 bracket, and renders everything into
a single self-contained file — **`dashboard.html`** — that opens in any browser
with no server.

> **The deliverable is `dashboard.html`.** Everything else in this repo exists to
> build it: scripts that collect the data, train the model, run the simulation, and
> assemble the page.

---

## What the dashboard shows

- **Champion & deep-run odds** for all 48 teams (title %, reach R32 → Final).
- **Group tables** and a full **knockout bracket** prediction.
- **Attack / defense ratings** in real units — expected goals **for/against per game**
  vs an average opponent (run through the fitted model, not an abstract 0–1 score).
- **Player Explorer** — every one of the ~1,250 squad players, searchable/sortable,
  with current-season club output and a Guardian-style bio popup.
- **Awards** — Top Scorer, Player of the Tournament, Young Player, Best Attack/Defense,
  and a **predictive Golden Ball race** (see *Golden Ball model* below).

Current headline result (10,000 sims): **Spain 27%**, Argentina 17%, France 10%,
England 9%.

---

## How it works (the pipeline)

```
 scrapers ─►  Data/scraped/*          ┐
 historical ─► Data/data/processed/   ├─►  train_model.py ─►  simulate.py ─►  build_dashboard.py ─►  dashboard.html
 raw inputs ─► Data/data/raw/*        ┘        (model)          (10k MC)         (renders page)
```

### 1. Data collection — `scraper*.py`
| Script | Produces | Source |
|---|---|---|
| `scraper.py` | match/team aggregates | StatsBomb open data (WC22, Euro24, Copa24, AFCON23) |
| `scraper_players.py` | `player_xg.csv` | StatsBomb tournament player xG |
| `scraper_soccerdata.py` | `player_xg_current.csv` | **Understat** 2025-26 club xG (big-5 leagues) via the `soccerdata` package |
| `scraper_squads.py` | `squads_2026.json`, `squad_players.csv` | The Guardian 2026 World Cup squad guide + FIFA rankings |

### 2. The goals model — `train_model.py`
A **Poisson regression** fit on ~49k historical international matches (1950–2026),
predicting goals scored by a team in a match:

```
Pipeline( StandardScaler → PoissonRegressor(alpha=0.1) )
features = [ elo_diff, off_rating_a, def_rating_b, is_home ]
```

- **Elo-adjusted form ratings** (rolling 7-game window):
  - `off_rating = Σ_last7 [ goals_for     × (opponent_elo / 1500) ]`
  - `def_rating = Σ_last7 [ goals_against × (1500 / opponent_elo) ]`
- **Recency weighting:** matches down-weighted with an 8-year half-life.
- **2026 squad firepower blend:** because this is an *international* tournament, the
  2026 offensive rating mixes whole-squad team form with individual player output —
  `off_rating = 0.65·team_form + 0.35·firepower`, where firepower = club xG + 1.5·intl xG.
  This stops stacked squads (e.g. France's attack) from being underrated for modest
  recent team results.
- **Holdout (2022+):** W/D/L accuracy ≈ 60%, log-loss ≈ 0.88, predicted goals/team
  1.36 ≈ actual 1.355.

Outputs: `poisson_goals_ours.pkl` (fitted pipeline), `team_ratings_2026.csv`
(per-team elo/off/def/age), `our_model.json` (readable coefficients).
`poisson_goals.pkl` is a reference model used only for a coefficient sanity-check.

### 3. The tournament simulation — `simulate.py`
Monte Carlo over the **official 2026 bracket** (`knockout_slots.csv`, matches 73–104):

- For each match, the fitted model gives each team's scoring rate **λ**; goals are
  drawn from Poisson(λ). A small squad-age tilt (peak 27) nudges λ.
- **Group stage:** 12 groups of 4 → top 2 + 8 best third-placed teams advance
  (`fw26_best_third_placed_combinations.csv` handles the 495 best-third cases).
- **Knockouts:** R32 → Final, with extra time (50%) and Elo-weighted penalties.
- **N = 10,000 iterations.** Outputs per-team stage probabilities, a most-likely
  bracket, and the full deterministic bracket to `Data/simulated/`.

### 4. The dashboard — `build_dashboard.py`
Reads the simulated outputs + scraped player/squad data and writes one ~2.7 MB
self-contained `dashboard.html` (HTML + CSS + JS + Plotly, no build step, no server).

---

## Golden Ball model (predicted best player)

The Golden Ball card is a **forecast**, not a goals+assists tally. For every squad
player it blends three signals and weights them by how deep the team is projected to go:

```
gb_index   = form_rating × expected_matches × spotlight
form_rating = club_form · usage  +  0.6 · intl_form
```

- **club_form** — 2025-26 goal involvements per 90: `0.5·(xG90 + xA90) + 0.5·(G+A)/90`.
- **usage** — `minutes / (minutes + 900)`, a saturating "nailed-on starter" weight so a
  fluky per-90 over few minutes can't out-rank an undroppable star.
- **intl_form** — goal output per match in recent national-team tournaments (0 if none).
- **expected_matches** — `3 group games + Σ stage-reach probabilities` (more games = more
  chances + more voter exposure).
- **spotlight** — `1 + (P(semi-final) + P(final))`, because the award almost always
  follows a deep run.

The index is converted to a sharpened win-probability share. *Limitation:* it's an
attacking-output model, so it can't rate deep-lying midfielders or keepers
(a Modrić '18 / Kahn '02 type winner).

---

## Running it

```bash
pip install pandas numpy scikit-learn joblib soccerdata

# Rebuild only the dashboard from existing simulated data:
python build_dashboard.py            # → dashboard.html

# Or re-run the full model + simulation pipeline:
python train_model.py                # fit the Poisson goals model
python simulate.py                   # 10,000-iteration Monte Carlo  (~13s)
python build_dashboard.py            # assemble the dashboard
```

Then open `dashboard.html` in any browser.

> The scrapers (`scraper*.py`) require network access and are only needed to refresh
> the underlying data; the committed `Data/` snapshot already contains everything the
> train → simulate → build chain needs.

---

## Data sources

- **Elo / historical results** — international match history (1872–2026), processed into
  `training_df.csv` and `final_elo.csv` (ratings for 333 teams).
- **StatsBomb open data** — team & player xG from WC22, Euro 2024, Copa América 2024, AFCON 2023.
- **Understat (via `soccerdata`)** — 2025-26 club-season xG/xA for big-5-league players.
- **The Guardian 2026 squad guide** — real 2026 squads, bios, coaches, star players.
- **FIFA rankings** — June 2026 and October 2022 snapshots (for ranking + trajectory).
- **Official 2026 fixtures** — group draw and the 104-match bracket slots.

---

## Repository layout

```
build_dashboard.py     Renders dashboard.html from simulated + scraped data
simulate.py            10,000-iteration Monte Carlo of the 2026 bracket
train_model.py         Fits the Poisson goals model
scraper*.py            Data collection (StatsBomb / Understat / Guardian)
dashboard.html         ← the deliverable (open in a browser)
Data/
  data/raw/            Group draw, bracket slots, best-third combinations
  data/processed/      Trained model, team ratings, training data, Elo
  scraped/             Player xG (club + tournament), 2026 squads
  simulated/           Champion probabilities + bracket predictions
  fifa_ranking_*.csv   FIFA ranking snapshots
```

*Built as a portfolio project. Predictions are probabilistic and for entertainment.*
