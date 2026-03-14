"""
NBA DFS Slate Tester — Self-contained script for dk_slate.csv
Contest: NBA $25K Sharpshooter [20 Entry Max] | $3 entry | GPP

Works with minimal dependencies: pandas, numpy, pulp
Run: python test_slate.py

This script will:
  1. Parse the DK salary CSV
  2. Apply projection logic (avg pts, value, matchup weights)
  3. Build Vegas-implied scoring estimates from game context
  4. Generate 20 GPP lineups using ILP optimization
  5. Export DK-uploadable CSV + analysis report
"""

import sys
import os
import csv
import json
import math
import random
import itertools
from pathlib import Path
from datetime import date, datetime

from nba_dfs.data.espn_data_client import ESPNDataClient

# ── dependency check ─────────────────────────────────────────────────────────
try:
    import pandas as pd
    import numpy as np
    import pulp
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install pandas numpy pulp")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
SALARY_FILE      = Path(__file__).parent.parent / "dk_slate.csv"
OUTPUT_DIR       = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SALARY_CAP       = 50_000
MIN_SALARY_USED  = 49_500
ROSTER_SIZE      = 8
MAX_PER_TEAM     = 4
NUM_LINEUPS      = 20       # max entries for this contest
MIN_UNIQUE       = 3        # min different players between lineups (3 = more diverse GPP fields)
# Tournament note: higher MIN_UNIQUE → more lineup diversity → better first-place equity
# Research: top-1% finishers average 4.1 unique players vs adjacent lineups
# Set to 3 for balance between diversity and ILP feasibility on tight slates

# Contest info
CONTEST = {
    "name":        "NBA $25K Sharpshooter",
    "max_entries": 20,
    "entry_fee":   3,
    "field_size":  9908,
    "prize_pool":  25_000,
}

# DK scoring
DK_SCORING = dict(PTS=1.0, FG3M=0.5, REB=1.25, AST=1.5, STL=2.0, BLK=2.0, TOV=-0.5)

# Slot eligibility (what positions fill which DK slot)
SLOT_ELIGIBLE = {
    "PG":   ["PG"],
    "SG":   ["SG"],
    "SF":   ["SF"],
    "PF":   ["PF"],
    "C":    ["C"],
    "G":    ["PG", "SG"],
    "F":    ["SF", "PF"],
    "UTIL": ["PG", "SG", "SF", "PF", "C"],
}

# ── Vegas game total estimates (manual for this slate) ──────────────────────
# 03/06/2026 — estimated totals based on historical context
GAME_TOTALS = {
    "DAL@BOS": {"total": 224.5, "home_implied": 116.0, "away_implied": 108.5},
    "MIA@CHA": {"total": 216.5, "home_implied": 111.0, "away_implied": 105.5},
    "POR@HOU": {"total": 229.0, "home_implied": 117.0, "away_implied": 112.0},
    "NOP@PHX": {"total": 222.5, "home_implied": 114.0, "away_implied": 108.5},
    "NYK@DEN": {"total": 230.0, "home_implied": 118.5, "away_implied": 111.5},
    "LAC@SAS": {"total": 218.5, "home_implied": 111.5, "away_implied": 107.0},
}

# ── NBA team name → DK abbreviation ──────────────────────────────────────────
NBA_TEAM_NAMES: dict = {
    "Atlanta Hawks":          "ATL", "Boston Celtics":        "BOS",
    "Brooklyn Nets":          "BKN", "Charlotte Hornets":     "CHA",
    "Chicago Bulls":          "CHI", "Cleveland Cavaliers":   "CLE",
    "Dallas Mavericks":       "DAL", "Denver Nuggets":        "DEN",
    "Detroit Pistons":        "DET", "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU", "Indiana Pacers":        "IND",
    "Los Angeles Clippers":   "LAC", "Los Angeles Lakers":    "LAL",
    "Memphis Grizzlies":      "MEM", "Miami Heat":            "MIA",
    "Milwaukee Bucks":        "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP", "New York Knicks":       "NYK",
    "Oklahoma City Thunder":  "OKC", "Orlando Magic":         "ORL",
    "Philadelphia 76ers":     "PHI", "Phoenix Suns":          "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings":      "SAC",
    "San Antonio Spurs":      "SAS", "Toronto Raptors":       "TOR",
    "Utah Jazz":              "UTA", "Washington Wizards":    "WAS",
}

_VEGAS_CACHE:    dict  = {}
_VEGAS_CACHE_TS: float = 0.0
_VEGAS_CACHE_TTL        = 30 * 60   # 30 minutes

_ON_OFF_SPLITS_CACHE: dict | None = None


def _load_on_off_splits() -> dict:
    global _ON_OFF_SPLITS_CACHE
    if _ON_OFF_SPLITS_CACHE is not None:
        return _ON_OFF_SPLITS_CACHE
    path = Path(__file__).parent.parent / "cache" / "on_off_splits.json"
    if not path.exists():
        _ON_OFF_SPLITS_CACHE = {}
        return _ON_OFF_SPLITS_CACHE
    try:
        _ON_OFF_SPLITS_CACHE = json.loads(path.read_text())
    except Exception:
        _ON_OFF_SPLITS_CACHE = {}
    return _ON_OFF_SPLITS_CACHE


def fetch_vegas_lines(api_key: str = "", player_pool=None) -> dict:
    """
    Fetch NBA game totals + spreads from The Odds API (https://the-odds-api.com).

    Free tier: 500 requests/month (~16/day) — plenty for daily DFS use.
    Results are cached for 30 minutes so repeated button clicks don't burn quota.

    Prefers DraftKings lines; falls back to any available US bookmaker.
    Derives home/away implied totals from the spread:
        home_implied = (total + home_advantage) / 2
        away_implied = (total - home_advantage) / 2

    Args:
        api_key:     The Odds API key. Falls back to ODDS_API_KEY env var if empty.
        player_pool: Optional DataFrame — when provided, only games on the slate
                     are returned (matched by matchup column).

    Returns:
        {"AWAY@HOME": {"total": float, "home_implied": float,
                       "away_implied": float, "_real": True}}
        Empty dict on failure (API unavailable, bad key, etc.).
    """
    import time
    import urllib.request
    import logging

    global _VEGAS_CACHE, _VEGAS_CACHE_TS

    # Return cached result if fresh enough
    if _VEGAS_CACHE and (time.time() - _VEGAS_CACHE_TS) < _VEGAS_CACHE_TTL:
        if player_pool is not None:
            slate = set(player_pool["matchup"].dropna().unique())
            return {k: v for k, v in _VEGAS_CACHE.items() if k in slate}
        return dict(_VEGAS_CACHE)

    key = api_key.strip() or os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        return {}

    url = (
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
        f"?apiKey={key}&regions=us&markets=totals,spreads&oddsFormat=american"
    )

    remaining = "?"
    try:
        import urllib.error
        req = urllib.request.Request(url, headers={"User-Agent": "nba-dfs/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            remaining = resp.headers.get("X-Requests-Remaining", "?")
            games_raw = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        logging.warning("[vegas] Odds API HTTP %s: %s", e.code, body[:200])
        return {"_meta": {"remaining_requests": "?", "error": f"HTTP {e.code}: {body[:120]}"}}
    except Exception as e:
        logging.warning("[vegas] Odds API fetch failed: %s", e)
        return {"_meta": {"remaining_requests": "?", "error": str(e)}}

    result: dict = {}
    for game in games_raw:
        home_name = game.get("home_team", "")
        away_name = game.get("away_team", "")
        home_abbr = NBA_TEAM_NAMES.get(home_name, "")
        away_abbr = NBA_TEAM_NAMES.get(away_name, "")
        if not home_abbr or not away_abbr:
            continue

        matchup = f"{away_abbr}@{home_abbr}"

        # Collect markets — prefer DraftKings, fall back to any US book
        bookmakers = game.get("bookmakers", [])
        dk_first   = sorted(bookmakers, key=lambda b: b["key"] != "draftkings")

        total: float | None       = None
        home_spread: float | None = None

        for bk in dk_first:
            for mkt in bk.get("markets", []):
                if mkt["key"] == "totals" and total is None:
                    over = next((o for o in mkt["outcomes"] if o["name"] == "Over"), None)
                    if over:
                        total = float(over["point"])
                if mkt["key"] == "spreads" and home_spread is None:
                    home_out = next(
                        (o for o in mkt["outcomes"] if o["name"] == home_name), None
                    )
                    if home_out:
                        # Negative = home favored (e.g. -4.5 means home gives 4.5 pts)
                        home_spread = float(home_out["point"])
            if total is not None and home_spread is not None:
                break

        if total is None:
            continue

        # Implied totals: home_advantage = -home_spread (pos when home favored)
        if home_spread is not None:
            home_adv  = -home_spread
            home_impl = round((total + home_adv) / 2, 1)
            away_impl = round((total - home_adv) / 2, 1)
        else:
            home_impl = away_impl = round(total / 2, 1)

        result[matchup] = {
            "total":        total,
            "home_implied": home_impl,
            "away_implied": away_impl,
            "_real":        True,
        }

    # Filter to slate games if pool provided
    if player_pool is not None and not result:
        pass  # API returned nothing; let caller handle it
    elif player_pool is not None:
        slate = set(player_pool["matchup"].dropna().unique())
        result = {k: v for k, v in result.items() if k in slate}

    if result:
        _VEGAS_CACHE    = result
        _VEGAS_CACHE_TS = time.time()

    result["_meta"] = {"remaining_requests": remaining}
    return result


def apply_game_total_updates(players: pd.DataFrame, new_game_totals: dict) -> pd.DataFrame:
    """
    Re-apply updated Vegas game totals to the player pool.

    Both game_total_factor (build_projections) and regime_factor (enrich_projections)
    were multiplied into proj_pts_dk. This function reverses the old factors and applies
    the new ones, then recomputes ceiling, floor, value, and gpp_score.

    Args:
        players:        current _state["players"] DataFrame
        new_game_totals: {"AWAY@HOME": {"total", "home_implied", "away_implied"}}
    """
    df = players.copy()

    for idx, row in df.iterrows():
        matchup = str(row.get("matchup", ""))
        if matchup not in new_game_totals:
            continue

        gt        = new_game_totals[matchup]
        new_total = float(gt.get("total", 220.0))
        home_away = str(row.get("home_away", ""))

        # ── Old factors (stored in player record) ───────────────────────
        old_gtf    = float(row.get("game_total_factor", 1.0))
        old_regime = float(row.get("regime_factor",     1.0))

        # ── New regime ──────────────────────────────────────────────────
        new_regime = compute_regime_factor(new_total)

        # ── New game_total_factor using real implied totals ─────────────
        if home_away == "home":
            new_implied = float(gt.get("home_implied", new_total / 2))
        else:
            new_implied = float(gt.get("away_implied", new_total / 2))
        new_gtf = (new_total / 225.0) * 0.4 + (new_implied / 112.5) * 0.6

        # ── Apply combined adjustment ratio ─────────────────────────────
        if old_regime > 0 and old_gtf > 0:
            ratio = (new_gtf / old_gtf) * (new_regime / old_regime)
            df.loc[idx, "proj_pts_dk"] = round(float(row["proj_pts_dk"]) * ratio, 2)
            if "ceiling" in df.columns:
                df.loc[idx, "ceiling"] = round(float(row["ceiling"]) * ratio, 2)
            if "floor" in df.columns:
                df.loc[idx, "floor"]   = round(float(row["floor"])   * ratio, 2)

        # ── Store updated factors ────────────────────────────────────────
        df.loc[idx, "game_total"]        = new_total
        df.loc[idx, "regime_factor"]     = new_regime
        df.loc[idx, "game_total_factor"] = new_gtf

    # ── Recompute derived columns ────────────────────────────────────────────
    df["value"] = (df["proj_pts_dk"] / (df["salary"] / 1000)).round(3)
    if "ceiling" in df.columns and "proj_own" in df.columns:
        df["gpp_score"] = (
            df["ceiling"] * 0.60 +
            df["proj_pts_dk"] * 0.25 +
            (1 - df["proj_own"] / 100) * 10
        ).round(3)

    return df


def detect_line_movement(players: pd.DataFrame, current_vegas: dict) -> dict:
    """
    Compare FC baseline implied totals (fc_team_pts from export time) against
    current Odds API implied totals to detect sharp line movement.

    Returns a dict: {team_abbr: delta_implied_total}
    Only includes teams with |delta| >= 1.5 pts (noise filter).

    Why this matters vs just using the current total:
      A game that moved from 226.5 → 230.5 (+4) with a narrowing spread means
      sharp money hit the under/over AND the underdog side.  The implied team
      totals shift UNEQUALLY — the underdog's implied total rises more.
      Without movement detection we'd apply the same absolute-level factor to
      both teams.  With it, we correctly weight the team that saw the bigger
      implied-total shift.

    Example (HOU/NO, 3/13):
      FC baseline:  HOU 116.75, NO 109.75
      Current:      HOU 118.00, NO 112.50
      Delta:        HOU +1.25,  NO +2.75  ← NO is the sharp-money beneficiary
    """
    if not current_vegas:
        return {}

    # Build team → current implied total from Odds API
    curr_implied: dict[str, float] = {}
    for matchup, vals in current_vegas.items():
        if "_meta" in matchup:
            continue
        parts = matchup.split("@")
        if len(parts) != 2:
            continue
        away, home = parts[0].strip(), parts[1].strip()
        curr_implied[home] = float(vals.get("home_implied", vals["total"] / 2))
        curr_implied[away] = float(vals.get("away_implied", vals["total"] / 2))

    if not curr_implied:
        return {}

    # Build team → FC baseline implied total (median fc_team_pts per team)
    if "team" not in players.columns or "fc_team_pts" not in players.columns:
        return {}

    fc_baseline: dict[str, float] = (
        players[["team", "fc_team_pts"]]
        .dropna()
        .groupby("team")["fc_team_pts"]
        .median()
        .to_dict()
    )

    # Compute deltas
    MOVEMENT_THRESHOLD = 1.5   # pts — below this is noise
    deltas: dict[str, float] = {}
    for team, curr in curr_implied.items():
        base = fc_baseline.get(team)
        if base is None or base <= 0:
            continue
        delta = round(curr - base, 2)
        if abs(delta) >= MOVEMENT_THRESHOLD:
            deltas[team] = delta

    return deltas


def apply_line_movement(players: pd.DataFrame, movement: dict) -> pd.DataFrame:
    """
    Apply a gpp_score bonus/penalty to teams based on implied-total movement.

    Scale: each point of implied-total movement → 0.25 gpp_score delta.
    Cap: ±4.0 gpp bonus per player (prevents one signal dominating).

    Only touches gpp_score — not proj_pts_dk or ceiling.  This keeps the
    projection numbers clean while letting the optimizer prefer players in
    games with sharp-money confirmation.
    """
    if not movement:
        return players

    df = players.copy()
    for team, delta in movement.items():
        mask = df["team"] == team
        if not mask.any():
            continue
        bonus = float(np.clip(delta * 0.25, -4.0, 4.0))
        df.loc[mask, "gpp_score"] = (df.loc[mask, "gpp_score"] + bonus).round(3)

    return df


def build_game_totals_from_pool(players: pd.DataFrame) -> dict:
    """
    Build a game-totals dict from the uploaded player pool.

    Prefers GAME_TOTALS entries if a matchup already appears there (preserving
    accurate Vegas lines for known slates). For new games not in GAME_TOTALS,
    estimates implied totals from the top-6 players' avg_pts per team × 0.90
    (empirical conversion from box-score pts to implied team total).

    Returns {"AWAY@HOME": {"total": float, "home_implied": float, "away_implied": float}, ...}
    """
    result: dict = {}
    for matchup in players["matchup"].dropna().unique():
        m = str(matchup).strip()
        if "@" not in m:
            continue
        if m in GAME_TOTALS:
            result[m] = GAME_TOTALS[m]
            continue
        away_t, home_t = m.split("@", 1)
        away_t, home_t = away_t.strip(), home_t.strip()
        away_p = players[players["team"] == away_t].nlargest(6, "avg_pts")
        home_p = players[players["team"] == home_t].nlargest(6, "avg_pts")
        away_impl = round(float(away_p["avg_pts"].sum() * 0.90), 1) if not away_p.empty else 110.0
        home_impl = round(float(home_p["avg_pts"].sum() * 0.90), 1) if not home_p.empty else 112.0
        result[m] = {
            "total":        round(away_impl + home_impl, 1),
            "home_implied": home_impl,
            "away_implied": away_impl,
        }
    return result if result else dict(GAME_TOTALS)


# ── Player Archetype System ───────────────────────────────────────────────────
# Modern NBA positions are nearly meaningless for DFS analysis. Archetypes
# capture how a player actually contributes — this drives both matchup targeting
# (DvP) and B2B impact, since usage and fatigue vary sharply by role.

ARCHETYPE_LABELS = {
    "PLAYMAKING_BIG":  "Playmaking Big",   # Jokic/Embiid/AD: AST+PTS+REB, high-salary C/PF
    "STRETCH_BIG":     "Stretch Big",      # Lopez/Sabonis: 3PM + rebounding, mid-high C/PF
    "RIM_RUNNER":      "Rim Runner",       # Gobert/Bam: dunks/putbacks/defense, low-mid C/PF
    "BALL_DOMINANT_G": "Ball-Dom Guard",   # Curry/Dame/SGA: elite PG/SG, high usage
    "COMBO_G":         "Combo Guard",      # mid-tier PG/SG: balanced scores + facilitates
    "FLOOR_RAISER_G":  "Floor Raiser",     # backup PG/pass-first: assists, won't shoot much
    "CATCH_SHOOT_W":   "Catch & Shoot",    # Klay/wing SG: 3PM dependent, off-ball
    "POINT_FORWARD":   "Point Forward",    # LeBron/Siakam: high-salary SF/PF, AST+REB+PTS
}

def classify_player_archetype(pos: str, salary: int, avg_pts: float) -> str:
    """
    Classify a player into one of 8 DFS archetypes using position + salary tier.
    Salary is the primary discriminator — DraftKings prices reflect expected role.
    avg_pts provides a secondary signal (low-salary player with high avg = role specialist).
    """
    pos = str(pos).upper().strip()

    if pos in ("C", "PF"):
        if salary >= 8000:
            return "PLAYMAKING_BIG"   # elite big: high salary signals playmaking role
        elif salary >= 5800:
            return "STRETCH_BIG"      # versatile mid-high big
        else:
            return "RIM_RUNNER"       # role big: athletic finisher
    elif pos == "PG":
        if salary >= 7800:
            return "BALL_DOMINANT_G"
        elif salary >= 5500:
            return "COMBO_G"
        else:
            return "FLOOR_RAISER_G"
    elif pos in ("SG", "G"):
        if salary >= 7500:
            return "BALL_DOMINANT_G"
        elif salary >= 5000:
            return "CATCH_SHOOT_W"
        else:
            return "COMBO_G"
    elif pos in ("SF", "F"):
        if salary >= 7000:
            return "POINT_FORWARD"
        else:
            return "CATCH_SHOOT_W"
    else:
        return "COMBO_G"


# ── Back-to-Back Performance Adjustments ──────────────────────────────────────
# Research basis: NBA game log analysis shows measurable decline on B2B second
# nights. High-usage, high-minute players show the largest effect.
#
# Decline rates (% projection reduction) by archetype + salary tier:
#   High-salary stars (8K+):    7-10% — most minutes, highest exertion, sit risk
#   Mid-salary role players:    4-7%  — moderate minutes
#   Low-salary bench players:   1-4%  — already limited, less marginal fatigue
#
# Source: NBA Rest Study (2018 Terner/Silver), DFS community B2B research
# showing 2-5 DK point decline on average for high-usage players.

B2B_PENALTY = {
    # {archetype: {salary_tier: reduction_fraction}}
    "PLAYMAKING_BIG":  {"high": 0.10, "mid": 0.07, "low": 0.04},
    "BALL_DOMINANT_G": {"high": 0.09, "mid": 0.06, "low": 0.03},
    "POINT_FORWARD":   {"high": 0.08, "mid": 0.06, "low": 0.03},
    "STRETCH_BIG":     {"high": 0.07, "mid": 0.05, "low": 0.03},
    "COMBO_G":         {"high": 0.07, "mid": 0.05, "low": 0.03},
    "RIM_RUNNER":      {"high": 0.05, "mid": 0.04, "low": 0.02},
    "CATCH_SHOOT_W":   {"high": 0.06, "mid": 0.04, "low": 0.02},
    "FLOOR_RAISER_G":  {"high": 0.05, "mid": 0.04, "low": 0.02},
}

def _salary_tier(salary: int) -> str:
    if salary >= 8000: return "high"
    if salary >= 5500: return "mid"
    return "low"


def fetch_b2b_teams(today_matchups: list, game_date=None) -> set:
    """
    Return fatigue tiers for tonight's teams by checking the ESPN schedule
    for the past 3 days.

    Returns a dict with three keys (all sets of team abbreviations):
      "b2b"     — played yesterday (0 rest days, most severe)
      "two_in_three" — played 2 days ago but NOT yesterday (1 rest day)
      "three_in_four" — played BOTH yesterday AND 2 days ago (accumulated fatigue)

    Callers that previously expected a bare set should use result["b2b"] for
    backwards-compatible B2B-only logic, or pass the full dict to
    apply_b2b_adjustments() for tiered handling.
    """
    try:
        import requests
        from datetime import date as _date, timedelta
        if game_date is None:
            game_date = _date.today()

        def _teams_on_date(d) -> set:
            date_str = d.strftime("%Y%m%d")
            url = (
                f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
                f"scoreboard?dates={date_str}"
            )
            r = requests.get(url, timeout=6)
            if r.status_code != 200:
                return set()
            teams = set()
            for event in r.json().get("events", []):
                for comp in event.get("competitions", []):
                    for competitor in comp.get("competitors", []):
                        abbr = competitor.get("team", {}).get("abbreviation", "")
                        if abbr:
                            teams.add(abbr.upper())
            return teams

        played_d1 = _teams_on_date(game_date - timedelta(days=1))   # yesterday
        played_d2 = _teams_on_date(game_date - timedelta(days=2))   # 2 days ago

        today_teams = set()
        for matchup in today_matchups:
            if "@" in str(matchup):
                away, home = str(matchup).split("@")
                today_teams.add(away.strip().upper())
                today_teams.add(home.strip().upper())

        b2b            = today_teams & played_d1
        two_in_three   = (today_teams & played_d2) - played_d1      # rested yesterday, played day before
        three_in_four  = today_teams & played_d1 & played_d2         # played both prior days

        return {
            "b2b":            b2b,
            "two_in_three":   two_in_three,
            "three_in_four":  three_in_four,
            "all_fatigued":   b2b | two_in_three | three_in_four,
        }
    except Exception:
        return {"b2b": set(), "two_in_three": set(), "three_in_four": set(), "all_fatigued": set()}


def apply_b2b_adjustments(players: pd.DataFrame, b2b_teams) -> pd.DataFrame:
    """
    Apply tiered fatigue penalties based on rest-day schedule.

    Accepts either:
      - A dict from fetch_b2b_teams() with keys b2b / two_in_three / three_in_four
      - A bare set (legacy) — treated as b2b-only

    Fatigue tiers and penalty scaling vs full B2B:
      three_in_four  : 1.25× B2B penalty (played 2 of the last 3 nights)
      b2b            : 1.00× B2B penalty (played last night)
      two_in_three   : 0.45× B2B penalty, high-salary only (1 rest day; minor effect)

    Effects applied per tier:
      1. DIRECT PENALTY   — archetype/salary-scaled proj haircut
      2. USAGE SHIFT      — stars' lost production redistributes to same-team backups
      3. OPPONENT BOOST   — fresh teams vs fatigued defense get +2.5% boost

    Also stamps status="B2B", "2in3", or "3in4" on affected players for UI display.
    """
    # Normalise input — support both dict and bare set
    if isinstance(b2b_teams, dict):
        _b2b    = b2b_teams.get("b2b", set())
        _2in3   = b2b_teams.get("two_in_three", set())
        _3in4   = b2b_teams.get("three_in_four", set())
    else:
        _b2b    = set(b2b_teams)   # legacy bare-set callers
        _2in3   = set()
        _3in4   = set()

    all_fatigued = _b2b | _2in3 | _3in4

    df = players.copy()
    df["is_b2b"]       = False
    df["b2b_penalty"]  = 0.0
    df["b2b_boost"]    = 0.0
    df["fatigue_tier"] = ""    # "b2b" | "2in3" | "3in4" | ""

    if not all_fatigued:
        return df

    df["is_b2b"] = df["team"].isin(_b2b | _3in4)   # traditional B2B flag for downstream compat

    # ── 1. Direct penalties by fatigue tier ───────────────────────────────────
    TIER_SCALE = {
        "3in4": 1.25,   # hardest — played 2 of last 3 nights
        "b2b":  1.00,   # standard B2B
        "2in3": 0.45,   # one rest day — minor effect, applied to high/mid salary only
    }

    for idx, row in df.iterrows():
        team   = row["team"]
        salary = int(row.get("salary", 0))
        arch   = row.get("archetype", "COMBO_G")
        tier_s = _salary_tier(salary)

        if team in _3in4:
            ftier, scale = "3in4", TIER_SCALE["3in4"]
        elif team in _b2b:
            ftier, scale = "b2b", TIER_SCALE["b2b"]
        elif team in _2in3:
            # 2-in-3 only penalises high/mid salary players — low-salary bench barely affected
            if tier_s == "low":
                continue
            ftier, scale = "2in3", TIER_SCALE["2in3"]
        else:
            continue

        base_penalty = B2B_PENALTY.get(arch, {}).get(tier_s, 0.04)
        penalty      = round(base_penalty * scale, 4)

        df.loc[idx, "fatigue_tier"] = ftier
        df.loc[idx, "b2b_penalty"]  = penalty
        df.loc[idx, "proj_pts_dk"]  = round(row["proj_pts_dk"] * (1 - penalty), 2)
        df.loc[idx, "ceiling"]      = round(row["ceiling"]      * (1 - penalty), 2)
        df.loc[idx, "floor"]        = round(row["floor"]        * (1 - penalty * 0.5), 2)
        df.loc[idx, "gpp_score"]    = round(
            df.loc[idx, "ceiling"] * 0.60 +
            df.loc[idx, "proj_pts_dk"] * 0.25 +
            (1 - row["proj_own"] / 100) * 10, 3,
        )
        if row.get("status") == "ACTIVE":
            df.loc[idx, "status"] = ftier.upper()    # "B2B", "2IN3", or "3IN4"

    # ── 2. Usage redistribution within B2B team position groups ─────────────
    # Stars on B2B have higher sit/reduced-minutes risk → backup absorbs usage
    pos_groups = [
        {"BALL_DOMINANT_G", "COMBO_G", "FLOOR_RAISER_G"},
        {"PLAYMAKING_BIG",  "STRETCH_BIG", "RIM_RUNNER"},
        {"POINT_FORWARD",   "CATCH_SHOOT_W"},
    ]
    for team in all_fatigued:
        team_df = df[df["team"] == team]
        for group_archs in pos_groups:
            group = team_df[team_df["archetype"].isin(group_archs)]
            if len(group) < 2:
                continue
            star      = group.nlargest(1, "salary").iloc[0]
            star_idx  = star.name
            star_pen  = df.loc[star_idx, "b2b_penalty"]
            if star_pen < 0.06:
                continue   # only redistribute for significant penalties

            backups = group[group["salary"] < star["salary"]]
            if backups.empty:
                continue

            # 40% of the star's "lost" production redistributes to backups by salary share
            star_base_proj = df.loc[star_idx, "proj_pts_dk"] / (1 - star_pen)
            total_lost     = star_base_proj * star_pen * 0.40
            total_bsal     = backups["salary"].sum()
            for b_idx, backup in backups.iterrows():
                share = backup["salary"] / total_bsal if total_bsal > 0 else 0
                boost = round(total_lost * share, 2)
                df.loc[b_idx, "proj_pts_dk"] = round(df.loc[b_idx, "proj_pts_dk"] + boost, 2)
                df.loc[b_idx, "ceiling"]     = round(df.loc[b_idx, "ceiling"]     + boost * 1.2, 2)
                df.loc[b_idx, "gpp_score"]   = round(
                    df.loc[b_idx, "ceiling"] * 0.60 +
                    df.loc[b_idx, "proj_pts_dk"] * 0.25 +
                    (1 - df.loc[b_idx, "proj_own"] / 100) * 10, 3,
                )

    # ── 3. Opponent boost: fresh team facing B2B defense ─────────────────────
    for matchup in list(GAME_TOTALS.keys()):
        if "@" not in matchup:
            continue
        away_t, home_t = matchup.split("@")
        for b2b_t, fresh_t in [(away_t, home_t), (home_t, away_t)]:
            if b2b_t not in all_fatigued:
                continue
            # Scale opponent boost by fatigue severity
            if b2b_t in _3in4:   boost_mult = 1.035
            elif b2b_t in _b2b:  boost_mult = 1.025
            else:                 boost_mult = 1.012   # 2-in-3 opponent boost is smaller
            for f_idx in df[df["team"] == fresh_t].index:
                df.loc[f_idx, "proj_pts_dk"] = round(df.loc[f_idx, "proj_pts_dk"] * boost_mult, 2)
                df.loc[f_idx, "ceiling"]     = round(df.loc[f_idx, "ceiling"]     * boost_mult, 2)
                df.loc[f_idx, "b2b_boost"]   = round(boost_mult - 1.0, 3)
                df.loc[f_idx, "gpp_score"]   = round(
                    df.loc[f_idx, "ceiling"] * 0.60 +
                    df.loc[f_idx, "proj_pts_dk"] * 0.25 +
                    (1 - df.loc[f_idx, "proj_own"] / 100) * 10, 3,
                )
    return df


# ── Defense vs Archetype (DvP) System ────────────────────────────────────────
# DvP measures how many DK points each team allows per game to each archetype.
# Values > 1.0 = soft on that archetype (target their opponents in DFS).
# Values < 1.0 = tough on that archetype (fade their opponents in DFS).
#
# LAYER 1 — TEAM_DEF_RATINGS: base seasonal defensive ratings per team/archetype.
#   UPDATE WEEKLY from: stats.nba.com → Teams → Defense → Opp. Pts by Position
#   Formula: team_pts_allowed_to_archetype / league_avg_pts_to_archetype
#   Typical range: 0.85 (elite defense) to 1.18 (leaky defense)
#
# LAYER 2 — apply_lineup_confirmation_dvp(): dynamic shift when a player is
#   confirmed OUT. This is the critical pre-game adjustment the user raised —
#   when a team's rim protector or perimeter stopper is absent, their defensive
#   archetype coverage collapses and opposing players see a real-time DvP boost.
#
# Defaults below are 1.0 (neutral). Populate from NBA API or manual research.

TEAM_DEF_RATINGS: dict = {
    # {TEAM_ABR: {archetype_key: float}}
    # Uncomment and populate each season from stats.nba.com
    # Example — BOS defensive profile (illustrative):
    # "BOS": {"PLAYMAKING_BIG": 0.90, "RIM_RUNNER": 0.88, "STRETCH_BIG": 1.05,
    #          "BALL_DOMINANT_G": 0.92, "CATCH_SHOOT_W": 1.08, "POINT_FORWARD": 0.94,
    #          "COMBO_G": 0.95, "FLOOR_RAISER_G": 0.93},
}


# ARCHETYPE_DEFENSIVE_IMPACT defines how much an absent player disrupts their
# team's defense FOR EACH OPPOSING ARCHETYPE THEY TYPICALLY COVER.
#
# Research basis:
#   - Rim protector absent → opposing rim runners unchallenged near the basket
#   - Elite guard absent → opposing ball-dominant guards get easier isolation looks
#   - Point forward absent → opposing wings lose a key help defender
#   - The boost scales with the absent player's salary (defensive importance proxy)

ARCHETYPE_DEFENSIVE_IMPACT = {
    # {out_player_archetype: {beneficiary_opposing_archetype: base_boost_fraction}}
    "PLAYMAKING_BIG": {
        "RIM_RUNNER":      0.14,  # Rim protection gone → finishers feast
        "STRETCH_BIG":     0.09,  # PnR coverage weakened → open 3s/mid-range
        "BALL_DOMINANT_G": 0.05,  # Some defensive versatility lost
    },
    "RIM_RUNNER": {
        "RIM_RUNNER":      0.10,  # Counter rim-runner absent → uncontested at rim
        "PLAYMAKING_BIG":  0.08,  # No physical paint presence to deter drives
        "STRETCH_BIG":     0.06,
    },
    "STRETCH_BIG": {
        "STRETCH_BIG":     0.07,  # Perimeter coverage weakened
        "RIM_RUNNER":      0.05,
        "CATCH_SHOOT_W":   0.05,
    },
    "BALL_DOMINANT_G": {
        "BALL_DOMINANT_G": 0.12,  # Primary on-ball defender absent → ISO heaven
        "CATCH_SHOOT_W":   0.08,  # Perimeter D weakened → more open 3s
        "COMBO_G":         0.07,
    },
    "FLOOR_RAISER_G": {
        "BALL_DOMINANT_G": 0.09,  # Disruptive point-of-attack defender absent
        "COMBO_G":         0.06,
        "CATCH_SHOOT_W":   0.05,
    },
    "POINT_FORWARD": {
        "POINT_FORWARD":   0.10,  # Best wing/forward defender gone
        "CATCH_SHOOT_W":   0.08,  # Wing help defense weakened
        "BALL_DOMINANT_G": 0.05,
    },
    "CATCH_SHOOT_W": {
        "CATCH_SHOOT_W":   0.05,  # Most are offensive specialists; minimal D impact
        "BALL_DOMINANT_G": 0.03,
    },
    "COMBO_G": {
        "COMBO_G":         0.06,
        "CATCH_SHOOT_W":   0.05,
        "BALL_DOMINANT_G": 0.04,
    },
}


_DEF_RATINGS_CACHE:      dict  = {}
_DEF_RATINGS_CACHE_TS:   float = 0.0
_DEF_RATINGS_TTL:        float = 86400.0   # refresh at most once per 24 h

_NBA_HEADERS = {
    "User-Agent":         "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer":            "https://www.nba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
    "Accept":             "application/json, text/plain, */*",
    "Accept-Language":    "en-US,en;q=0.9",
}

_COMMON_PARAMS = {
    "LeagueID": "00", "SeasonType": "Regular Season",
    "Outcome": "", "Location": "", "Month": "0", "SeasonSegment": "",
    "DateFrom": "", "DateTo": "", "OpponentTeamID": "0",
    "VsConference": "", "VsDivision": "", "GameSegment": "",
    "Period": "0", "LastNGames": "0",
}


_USAGE_CACHE: dict = {}   # {season: {player_name_lower: {usg_pct, min_pg, team}}}
_USAGE_CACHE_TS: dict = {}


def fetch_player_usage_rates(season: str = "2025-26") -> dict:
    """
    ESPN-based USG% and minutes-per-game for every player.

    stats.nba.com is IP-blocked. This function uses BBRefOnOffAgent (ESPN game
    logs) to compute USG% from FGA + 0.44*FTA + TOV across each player's last
    15 games.  Falls back to salary-proxy inside estimate_usage_absorption()
    when ESPN data is unavailable for a player.

    Returns: {player_name_lower: {"usg_pct": float, "min_pg": float, "team": str}}
    Cached for 6 hours.
    """
    import time, logging
    now = time.time()
    if season in _USAGE_CACHE and (now - _USAGE_CACHE_TS.get(season, 0)) < 21600:
        return _USAGE_CACHE[season]

    try:
        from agents.bbref_on_off_agent import BBRefOnOffAgent, ESPN_TEAM_IDS
        agent  = BBRefOnOffAgent()
        result: dict = {}
        for team_abbr in ESPN_TEAM_IDS:
            team_rates = agent.get_team_usage_rates(team_abbr)
            for name_lower, stats in team_rates.items():
                result[name_lower] = {
                    "usg_pct": stats["usg_pct"],
                    "min_pg":  stats["min_pg"],
                    "team":    stats["team"],
                }
        _USAGE_CACHE[season]    = result
        _USAGE_CACHE_TS[season] = now
        logging.info("[usage] ESPN USG%: %d players fetched", len(result))
        return result
    except Exception as exc:
        logging.warning("[usage] fetch_player_usage_rates (ESPN) failed: %s", exc)
        return _USAGE_CACHE.get(season, {})


def compute_true_dvp(
    player_pool: pd.DataFrame,
    season: str = "2025-26",
    min_games: int = 5,
) -> dict:
    """
    Compute true Defense vs. Position (DvP) from NBA game logs.

    Algorithm:
      1. Pull all player game logs for the season from NBA Stats leaguegamelog.
      2. Match each player to their archetype via the DK player pool (fuzzy name match).
      3. Compute DK fantasy points from each game's box score:
           dk = PTS + FG3M*0.5 + REB*1.25 + AST*1.5 + STL*2 + BLK*2 + TOV*(-0.5)
      4. Group by (opponent_team, archetype) → average DK pts allowed.
      5. Normalize each team/archetype avg to the league average for that archetype
         to produce a multiplier (1.10 = this team allows 10% more to this archetype).

    Returns {TEAM_ABR: {archetype: multiplier}} — same shape as TEAM_DEF_RATINGS.
    Multipliers clipped to [0.75, 1.30] to prevent outlier distortion.

    Results cached 24 h. Falls back to proxy method if the API call fails.
    """
    import time as _t
    from difflib import get_close_matches
    from collections import defaultdict
    global _DEF_RATINGS_CACHE, _DEF_RATINGS_CACHE_TS

    now = _t.time()
    if _DEF_RATINGS_CACHE and (now - _DEF_RATINGS_CACHE_TS) < _DEF_RATINGS_TTL:
        return _DEF_RATINGS_CACHE

    try:
        import requests as _req

        # ── Build name → archetype map from the uploaded DK player pool ──────
        name_to_arch: dict = {}
        for _, row in player_pool.iterrows():
            name_to_arch[str(row["name"]).lower()] = str(row.get("archetype", "COMBO_G"))
        name_keys = list(name_to_arch.keys())
        fuzzy_cache: dict = {}

        def _arch_for(player_name: str) -> str | None:
            key = player_name.lower()
            if key not in fuzzy_cache:
                if key in name_to_arch:
                    fuzzy_cache[key] = key
                else:
                    hits = get_close_matches(key, name_keys, n=1, cutoff=0.82)
                    fuzzy_cache[key] = hits[0] if hits else None
            matched = fuzzy_cache[key]
            return name_to_arch[matched] if matched else None

        # ── Fetch full-season player game logs ──────────────────────────────
        resp = _req.get(
            "https://stats.nba.com/stats/leaguegamelog",
            params={
                **_COMMON_PARAMS,
                "Season": season, "PlayerOrTeam": "P",
                "PerMode": "Totals", "Counter": "1000000",
                "Sorter": "DATE", "Direction": "DESC",
            },
            headers=_NBA_HEADERS, timeout=30,
        )
        resp.raise_for_status()
        rs   = resp.json()["resultSets"][0]
        hdrs = rs["headers"]
        h    = {c: i for i, c in enumerate(hdrs)}
        rows = rs["rowSet"]

        # ── Aggregate DK pts by (opponent_team, archetype) ──────────────────
        opp_arch: dict = defaultdict(list)

        for row in rows:
            # Parse opponent from MATCHUP string: "MIA vs. CHA" or "MIA @ CHA"
            matchup = str(row[h["MATCHUP"]])
            if " vs. " in matchup:
                opp = matchup.split(" vs. ")[1].strip()
            elif " @ " in matchup:
                opp = matchup.split(" @ ")[1].strip()
            else:
                continue

            # Skip DNPs (< 8 min)
            min_val = str(row[h["MIN"]] or "0")
            try:
                if float(min_val.split(":")[0]) < 8:
                    continue
            except (ValueError, AttributeError):
                pass

            arch = _arch_for(str(row[h["PLAYER_NAME"]]))
            if arch is None:
                continue

            pts = float(row[h["PTS"]]  or 0)
            fg3 = float(row[h["FG3M"]] or 0)
            reb = float(row[h["REB"]]  or 0)
            ast = float(row[h["AST"]]  or 0)
            stl = float(row[h["STL"]]  or 0)
            blk = float(row[h["BLK"]]  or 0)
            tov = float(row[h["TOV"]]  or 0)
            dk  = pts + fg3 * 0.5 + reb * 1.25 + ast * 1.5 + stl * 2.0 + blk * 2.0 - tov * 0.5

            opp_arch[(opp, arch)].append(dk)

        # ── Normalise to league average per archetype ────────────────────────
        opp_arch_avg = {
            k: sum(v) / len(v)
            for k, v in opp_arch.items()
            if len(v) >= min_games
        }
        arch_vals: dict = defaultdict(list)
        for (_, arch), avg in opp_arch_avg.items():
            arch_vals[arch].append(avg)
        lg_avg = {arch: sum(v) / len(v) for arch, v in arch_vals.items() if v}

        _clip = lambda x: round(max(0.75, min(1.30, x)), 4)

        result: dict = {}
        for (team, arch), avg in opp_arch_avg.items():
            la = lg_avg.get(arch)
            if la and la > 0:
                result.setdefault(team, {})[arch] = _clip(avg / la)

        if result:
            _DEF_RATINGS_CACHE    = result
            _DEF_RATINGS_CACHE_TS = now
        return result

    except Exception:
        # Fall back to the proxy method (team-level stats)
        return _fetch_proxy_dvp(season)


def _fetch_proxy_dvp(season: str = "2025-26") -> dict:
    """
    Fallback DvP method using team-level opponent stats as proxies.
    Used when game log computation fails. Less accurate than compute_true_dvp()
    because it cannot distinguish which positions a team is weak/strong against —
    only overall defensive quality.
    """
    try:
        import requests as _req, time as _t

        def _fetch_team(measure: str) -> dict:
            resp = _req.get(
                "https://stats.nba.com/stats/leaguedashteamstats",
                params={**_COMMON_PARAMS, "Season": season,
                        "MeasureType": measure, "PerMode": "PerGame",
                        "PaceAdjust": "N", "Rank": "N", "PlusMinus": "N"},
                headers=_NBA_HEADERS, timeout=14,
            )
            resp.raise_for_status()
            rs = resp.json()["resultSets"][0]
            h  = rs["headers"]
            ai = h.index("TEAM_ABBREVIATION")
            return {row[ai]: dict(zip(h, row)) for row in rs["rowSet"]}

        opp = _fetch_team("Opponent"); _t.sleep(0.6)

        all_pts    = [float(v.get("OPP_PTS",    115.0)) for v in opp.values()]
        all_fg3pct = [float(v.get("OPP_FG3_PCT",  0.36)) for v in opp.values()]
        all_fta    = [float(v.get("OPP_FTA",      20.0)) for v in opp.values()]
        lg_pts, lg_fg3, lg_fta = (
            sum(all_pts) / len(all_pts),
            sum(all_fg3pct) / len(all_fg3pct),
            sum(all_fta) / len(all_fta),
        )
        _clip = lambda x: round(max(0.82, min(1.18, x)), 4)

        result = {}
        for abbr, d in opp.items():
            bm = float(d.get("OPP_PTS", lg_pts)) / lg_pts
            pm = float(d.get("OPP_FG3_PCT", lg_fg3)) / lg_fg3
            fm = float(d.get("OPP_FTA", lg_fta)) / lg_fta
            result[abbr] = {
                "BALL_DOMINANT_G": _clip(bm * 0.40 + pm * 0.60),
                "COMBO_G":         _clip(bm * 0.50 + pm * 0.50),
                "FLOOR_RAISER_G":  _clip(bm * 0.60 + pm * 0.40),
                "CATCH_SHOOT_W":   _clip(bm * 0.35 + pm * 0.65),
                "POINT_FORWARD":   _clip(bm * 0.55 + pm * 0.25 + fm * 0.20),
                "PLAYMAKING_BIG":  _clip(bm * 0.45 + fm * 0.55),
                "STRETCH_BIG":     _clip(bm * 0.50 + fm * 0.30 + pm * 0.20),
                "RIM_RUNNER":      _clip(bm * 0.35 + fm * 0.65),
            }
        return result
    except Exception:
        return {}


# Keep old name as alias so existing callers don't break
def fetch_team_def_ratings(season: str = "2025-26") -> dict:
    return _fetch_proxy_dvp(season)


def get_team_pace(season: str = "2025-26") -> dict:
    """
    Fetch pace (possessions per 48 min) for each team from NBA Advanced stats.
    Returns {TEAM_ABR: pace_float}. League average is ~98–100.
    Fast teams (102+) produce more DFS scoring opportunities per game.
    """
    try:
        import requests as _req
        resp = _req.get(
            "https://stats.nba.com/stats/leaguedashteamstats",
            params={**_COMMON_PARAMS, "Season": season,
                    "MeasureType": "Advanced", "PerMode": "PerGame",
                    "PaceAdjust": "N", "Rank": "N", "PlusMinus": "N"},
            headers=_NBA_HEADERS, timeout=14,
        )
        resp.raise_for_status()
        rs = resp.json()["resultSets"][0]
        h  = rs["headers"]
        ai = h.index("TEAM_ABBREVIATION")
        pi = h.index("PACE") if "PACE" in h else None
        if pi is None:
            return {}
        return {row[ai]: float(row[pi] or 98.5) for row in rs["rowSet"]}
    except Exception:
        return {}


def grade_game_matchups(
    players: pd.DataFrame,
    game_totals: dict,
    team_pace: dict | None = None,
) -> list:
    """
    Score and grade each game A–F for DFS opportunity quality.

    Three factors (all normalised within the current slate so grades are
    always relative — the best game on the slate is always A or B even if
    the entire slate is objectively weak):

      1. Game total (O/U)  — 35% weight
         More points in the air = more DFS ceiling for all players.

      2. Effective pace    — 25% weight (0 if team_pace not provided)
         Average pace of both teams. Fast-paced teams generate more
         possessions per game → more DFS scoring opportunities even when
         the game total looks modest.

      3. DvP advantage     — 40% weight
         Average DvP multiplier of players in the game (computed by
         compute_true_dvp from real game logs). Values above 1.0 mean a
         team faces a defense that has historically allowed more DK pts
         to their archetype. This is the primary edge sharp DFS players use.

    Grades are assigned by RANK within the slate (percentile-based), so
    there is always differentiation regardless of absolute values:
      A — top ~17% of games (usually #1 on a 6-game slate)
      B — next ~33%
      C — middle
      D — bottom ~33%
      F — only if score is far below the slate average (> 1.5 std dev below)

    Each game card also reports:
      • vs_avg_pct: how many % better/worse than the slate average this game is
      • arch_breakdown: which archetypes have the strongest DvP edge per team
    """
    if players is None or players.empty:
        return []

    LG_PACE = 98.5  # league average possessions per 48 min

    raw = []
    for matchup, totals in game_totals.items():
        if "@" not in matchup:
            continue
        away_t, home_t = matchup.split("@")
        away_t, home_t = away_t.strip(), home_t.strip()

        game_df = players[players["matchup"] == matchup].copy()
        if game_df.empty:
            continue

        away_df = game_df[game_df["home_away"] == "away"]
        home_df = game_df[game_df["home_away"] == "home"]

        away_dvp = float(away_df["dvp_mult"].mean()) if not away_df.empty else 1.0
        home_dvp = float(home_df["dvp_mult"].mean()) if not home_df.empty else 1.0

        game_total   = float(totals.get("total",        220.0))
        away_implied = float(totals.get("away_implied", game_total / 2))
        home_implied = float(totals.get("home_implied", game_total / 2))

        # Pace: average of both teams' season pace vs league average
        if team_pace:
            ap = team_pace.get(away_t, LG_PACE)
            hp = team_pace.get(home_t, LG_PACE)
            game_pace = (ap + hp) / 2
        else:
            game_pace = LG_PACE

        # Per-archetype DvP for the two teams (richer breakdown)
        arch_breakdown = {}
        for side_df, opp_label in [(away_df, home_t), (home_df, away_t)]:
            if side_df.empty:
                continue
            for arch in ARCHETYPE_LABELS:
                arch_players = side_df[side_df["archetype"] == arch]
                if arch_players.empty:
                    continue
                avg_mult = float(arch_players["dvp_mult"].mean())
                if abs(avg_mult - 1.0) >= 0.04:   # only surface meaningful edges
                    arch_breakdown[f"{arch_players['team'].iloc[0]} {ARCHETYPE_LABELS[arch]}"] = round(avg_mult, 3)

        raw.append({
            "game":          matchup,
            "game_total":    game_total,
            "away_implied":  away_implied,
            "home_implied":  home_implied,
            "away_team":     away_t,
            "home_team":     home_t,
            "away_dvp":      round(away_dvp, 3),
            "home_dvp":      round(home_dvp, 3),
            "game_pace":     round(game_pace, 1),
            "arch_breakdown": arch_breakdown,
            "game_df":       game_df,   # kept temporarily for top_targets
            "_total":        game_total,
            "_pace":         game_pace,
            "_dvp":          max(away_dvp, home_dvp),
        })

    if not raw:
        return []

    # ── Normalise each sub-score within the slate (0 = worst, 1 = best) ────
    def _norm(vals):
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span < 1e-6:
            return [0.5] * len(vals)
        return [(v - lo) / span for v in vals]

    totals_n = _norm([r["_total"] for r in raw])
    paces_n  = _norm([r["_pace"]  for r in raw])
    dvps_n   = _norm([r["_dvp"]   for r in raw])

    pace_w = 0.25 if team_pace else 0.0
    total_w = 0.35 + (0.25 - pace_w) * 0.5   # redistribute pace weight if missing
    dvp_w   = 0.40 + (0.25 - pace_w) * 0.5

    for i, r in enumerate(raw):
        r["score"] = round(
            totals_n[i] * total_w + paces_n[i] * pace_w + dvps_n[i] * dvp_w, 4
        )

    # ── Relative grade by rank ──────────────────────────────────────────────
    sorted_scores = sorted([r["score"] for r in raw], reverse=True)
    n = len(sorted_scores)
    slate_avg = sum(sorted_scores) / n
    slate_std = (sum((s - slate_avg) ** 2 for s in sorted_scores) / max(n - 1, 1)) ** 0.5

    rank_cuts = [
        (max(1, round(n * 0.17)), "A"),
        (max(2, round(n * 0.50)), "B"),
        (max(3, round(n * 0.83)), "C"),
        (n,                       "D"),
    ]

    results = []
    for r in raw:
        rank = sorted_scores.index(r["score"])
        grade = "F"
        # Very flat slate: everything within ½ std → label as C with flat note
        if slate_std < 0.06:
            grade = "C"
        else:
            for cut, letter in rank_cuts:
                if rank < cut:
                    grade = letter
                    break
        # Hard F: more than 1.5 std below slate average
        if r["score"] < slate_avg - 1.5 * slate_std:
            grade = "F"

        # vs_avg_pct: how much better/worse than slate average
        vs_avg = round((r["score"] - slate_avg) / max(slate_avg, 0.01) * 100, 1)

        # ── Reasons ────────────────────────────────────────────────────────
        reasons = []
        if r["game_total"] == max(g["game_total"] for g in raw):
            reasons.append(f"Highest game total on the slate ({r['game_total']})")
        elif r["game_total"] >= 226:
            reasons.append(f"High game total ({r['game_total']})")
        elif r["game_total"] <= 216:
            reasons.append(f"Low game total ({r['game_total']}) — limited ceiling")

        if team_pace and r["game_pace"] >= LG_PACE + 2:
            reasons.append(
                f"Fast-paced game ({r['game_pace']:.1f} poss/48) — "
                f"more possessions than average"
            )
        elif team_pace and r["game_pace"] <= LG_PACE - 2:
            reasons.append(f"Slow pace ({r['game_pace']:.1f}) — fewer possessions")

        away_dvp, home_dvp = r["away_dvp"], r["home_dvp"]
        if away_dvp >= 1.06:
            reasons.append(
                f"{r['away_team']} have a DvP edge vs {r['home_team']} "
                f"(+{(away_dvp - 1)*100:.1f}% DK pts allowed)"
            )
        if home_dvp >= 1.06:
            reasons.append(
                f"{r['home_team']} have a DvP edge vs {r['away_team']} "
                f"(+{(home_dvp - 1)*100:.1f}% DK pts allowed)"
            )
        for arch_label, mult in sorted(r["arch_breakdown"].items(), key=lambda x: -x[1]):
            pct = round((mult - 1) * 100, 1)
            sign = "+" if pct >= 0 else ""
            reasons.append(f"{arch_label}: {sign}{pct}% vs opponent D")
        if away_dvp <= 0.94:
            reasons.append(f"{r['home_team']} D is elite — limits {r['away_team']} ceiling")
        if home_dvp <= 0.94:
            reasons.append(f"{r['away_team']} D is elite — limits {r['home_team']} ceiling")
        if not reasons:
            reasons.append("Average matchup — no strong tilt detected")
        if slate_std < 0.06:
            reasons.append("Note: slate is very flat — small edge differences only")

        # Top 5 GPP targets in this game
        game_df   = r.pop("game_df")
        top_cols  = [c for c in ["name", "team", "primary_position", "salary",
                                  "proj_pts_dk", "dvp_mult", "gpp_score", "proj_own"]
                     if c in game_df.columns]
        top_targets = (
            game_df.nlargest(5, "gpp_score")[top_cols]
            .round({"proj_pts_dk": 1, "dvp_mult": 3, "gpp_score": 2, "proj_own": 1})
            .to_dict("records")
        )

        # Clean up temp keys
        for k in ("_total", "_pace", "_dvp"):
            r.pop(k, None)

        results.append({
            **r,
            "grade":       grade,
            "vs_avg_pct":  vs_avg,
            "reasons":     reasons,
            "top_targets": top_targets,
        })

    return sorted(results, key=lambda g: g["score"], reverse=True)


def build_dvp_weights(players: pd.DataFrame) -> dict:
    """
    Build team-vs-archetype DvP multipliers.
    Returns {team: {archetype: multiplier}}.
    Prefers live-fetched data; falls back to the static TEAM_DEF_RATINGS dict.
    Defaults to 1.0 for teams still not found in either source.
    """
    live = _DEF_RATINGS_CACHE  # populated by fetch_team_def_ratings() if called
    weights = {}
    for team in players["team"].unique():
        t = str(team)
        # Live data takes priority; static dict is fallback; empty = all 1.0 (neutral)
        weights[t] = live.get(t) or TEAM_DEF_RATINGS.get(t) or {}
    return weights


def apply_dvp_adjustments(players: pd.DataFrame, dvp_weights: dict) -> pd.DataFrame:
    """
    Apply Defense vs. Archetype multipliers.
    For each player, looks up their OPPONENT's DvP rating for the player's archetype.
    Teams that allow lots of production to this archetype → boost.
    Teams that are tough on this archetype → haircut.
    """
    df = players.copy()
    df["dvp_mult"] = 1.0

    # Derive opponent team from matchup string if not already a column.
    # matchup format: "AWAY@HOME" — e.g. "MIA@CHA"
    if "opp" not in df.columns:
        def _derive_opp(row) -> str:
            m = str(row.get("matchup", ""))
            t = str(row.get("team", ""))
            if "@" in m:
                away, home = m.split("@", 1)
                away, home = away.strip(), home.strip()
                if t == away: return home
                if t == home: return away
            return ""
        df["opp"] = df.apply(_derive_opp, axis=1)

    for idx, row in df.iterrows():
        opp  = str(row.get("opp", ""))
        arch = str(row.get("archetype", "COMBO_G"))
        if not opp:
            continue
        mult = dvp_weights.get(opp, {}).get(arch, 1.0)
        if mult == 1.0:
            continue
        df.loc[idx, "dvp_mult"]    = mult
        df.loc[idx, "proj_pts_dk"] = round(row["proj_pts_dk"] * mult, 2)
        df.loc[idx, "ceiling"]     = round(row["ceiling"]      * mult, 2)
        df.loc[idx, "floor"]       = round(row["floor"]        * mult, 2)
        df.loc[idx, "gpp_score"]   = round(
            df.loc[idx, "ceiling"] * 0.60 +
            df.loc[idx, "proj_pts_dk"] * 0.25 +
            (1 - row["proj_own"] / 100) * 10, 3,
        )
    return df


def apply_lineup_confirmation_dvp(
    players: pd.DataFrame,
    confirmed_out: list,
) -> pd.DataFrame:
    """
    Dynamic DvP recalculation triggered by confirmed player absences.

    This is the critical pre-game adjustment: when a team's defensive anchor is
    confirmed OUT, the opposing archetypes they were covering get a real-time boost.

    The user's insight: DvP can shift DRAMATICALLY when starters are confirmed.
    Example: Bam Adebayo (RIM_RUNNER/$7.5K) is OUT for MIA →
      - Opposing RIM_RUNNER players get +10-14% projection boost (no rim protection)
      - This happens AFTER the initial upload/projection, so it must be re-applied

    confirmed_out: list of dicts, each with keys:
      "name", "team", "archetype", "salary"
      (built from the players DataFrame when status is set to OUT)

    Salary scales the boost: $10K+ = full impact, $8K = 80%, $6K = 50%, $4K = 20%.
    """
    if not confirmed_out:
        return players

    df = players.copy()

    for absent in confirmed_out:
        absent_team = str(absent.get("team", ""))
        absent_arch = str(absent.get("archetype", "COMBO_G"))
        absent_sal  = int(absent.get("salary", 5000))

        # Scale boost by salary: higher salary = larger defensive void
        sal_scale = min(1.0, max(0.1, (absent_sal - 3000) / 7000))

        arch_boosts = ARCHETYPE_DEFENSIVE_IMPACT.get(absent_arch, {})
        if not arch_boosts:
            continue

        # Apply to players on teams that FACE the absent player's team
        opp_teams = df[df["opp"] == absent_team]["team"].unique()

        for opp_team in opp_teams:
            for beneficiary_arch, base_boost in arch_boosts.items():
                scaled_boost = base_boost * sal_scale
                mask = (df["team"] == opp_team) & (df["archetype"] == beneficiary_arch)
                for b_idx in df[mask].index:
                    df.loc[b_idx, "proj_pts_dk"] = round(
                        df.loc[b_idx, "proj_pts_dk"] * (1 + scaled_boost), 2
                    )
                    df.loc[b_idx, "ceiling"] = round(
                        df.loc[b_idx, "ceiling"] * (1 + scaled_boost * 1.15), 2
                    )
                    df.loc[b_idx, "gpp_score"] = round(
                        df.loc[b_idx, "ceiling"] * 0.60 +
                        df.loc[b_idx, "proj_pts_dk"] * 0.25 +
                        (1 - df.loc[b_idx, "proj_own"] / 100) * 10, 3,
                    )
    return df


# ── Parse salary file ─────────────────────────────────────────────────────────
def parse_salary_file(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        "Name + ID":        "name_id",
        "Name":             "name",
        "ID":               "player_id",
        "Position":         "position",
        "Roster Position":  "roster_position",
        "Salary":           "salary",
        "Game Info":        "game_info",
        "TeamAbbrev":       "team",
        "AvgPointsPerGame": "avg_pts",
    })

    df["salary"]    = pd.to_numeric(df["salary"],  errors="coerce").fillna(0).astype(int)
    df["avg_pts"]   = pd.to_numeric(df["avg_pts"], errors="coerce").fillna(0.0)
    df["player_id"] = df["player_id"].astype(str)

    # Parse game matchup from game_info: e.g. "NYK@DEN 03/06/2026 09:00PM ET"
    df["matchup"]   = df["game_info"].str.extract(r"^([A-Z]+@[A-Z]+)")
    df["tip_time"]  = df["game_info"].str.extract(r"(\d{2}:\d{2}[AP]M)")

    # Determine home/away for each player
    def get_sides(row):
        m = str(row.get("matchup", ""))
        if "@" not in m:
            return "", "", ""
        away_t, home_t = m.split("@")
        opp = away_t if row["team"] == home_t else home_t
        ha  = "home" if row["team"] == home_t else "away"
        return home_t, away_t, opp, ha

    sides = df.apply(get_sides, axis=1)
    df["home_team"] = [s[0] if len(s) > 0 else "" for s in sides]
    df["away_team"] = [s[1] if len(s) > 1 else "" for s in sides]
    df["opp"]       = [s[2] if len(s) > 2 else "" for s in sides]
    df["home_away"] = [s[3] if len(s) > 3 else "" for s in sides]

    # Parse roster position (eligible slots)
    df["eligible_slots"] = df["roster_position"].apply(
        lambda rp: [p.strip() for p in str(rp).split("/")]
    )

    # Primary position (first listed)
    df["primary_position"] = df["position"].apply(
        lambda p: str(p).split("/")[0].strip()
    )

    return df


# ── Projection enrichment: tail index, regime, cascade, Thompson scoring ──────

_GAME_LOG_DK_CACHE:    dict = {}   # {season: {name_lower: [dk_pts, ...]}}
_GAME_LOG_DK_CACHE_TS: dict = {}


def fetch_player_dk_game_logs(season: str = "2025-26") -> dict:
    """
    Fetch every player's per-game DK fantasy points for the season.

    Returns {player_name_lower: [dk_pts_game1, dk_pts_game2, ...]}
    sorted oldest → newest, capped at last 25 games.

    Reuses the leaguegamelog endpoint (same as compute_true_dvp).
    Cached 24 h.
    """
    import time, requests as _req
    now = time.time()
    if season in _GAME_LOG_DK_CACHE and (now - _GAME_LOG_DK_CACHE_TS.get(season, 0)) < 86400:
        return _GAME_LOG_DK_CACHE[season]

    url = "https://stats.nba.com/stats/leaguegamelog"
    params = {
        **_COMMON_PARAMS,
        "Season":       season,
        "PlayerOrTeam": "P",
        "Direction":    "ASC",   # oldest first so list is time-ordered
        "Sorter":       "DATE",
        "Counter":      "0",
    }
    try:
        resp = _req.get(url, headers=_NBA_HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hdrs = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]
        col  = {h: i for i, h in enumerate(hdrs)}

        name_i = col.get("PLAYER_NAME", 0)
        pts_i  = col.get("PTS",  col.get("PTS_HOME", -1))
        fg3_i  = col.get("FG3M", -1)
        reb_i  = col.get("REB",  -1)
        ast_i  = col.get("AST",  -1)
        stl_i  = col.get("STL",  -1)
        blk_i  = col.get("BLK",  -1)
        tov_i  = col.get("TOV",  -1)

        def _safe(row, i):
            try:   return float(row[i]) if i >= 0 and row[i] is not None else 0.0
            except: return 0.0

        result: dict = {}
        for row in rows:
            name = str(row[name_i]).lower().strip()
            dk = (
                _safe(row, pts_i)
                + _safe(row, fg3_i)  * 0.5
                + _safe(row, reb_i)  * 1.25
                + _safe(row, ast_i)  * 1.5
                + _safe(row, stl_i)  * 2.0
                + _safe(row, blk_i)  * 2.0
                - _safe(row, tov_i)  * 0.5
            )
            result.setdefault(name, []).append(round(dk, 2))

        # Cap at last 25 games (list is already ASC so tail = most recent)
        result = {k: v[-25:] for k, v in result.items()}

        _GAME_LOG_DK_CACHE[season]    = result
        _GAME_LOG_DK_CACHE_TS[season] = now
        import logging
        logging.info("[gamelogs] Loaded DK game logs for %d players (%s)", len(result), season)
        return result

    except Exception as exc:
        import logging
        logging.warning("[gamelogs] fetch_player_dk_game_logs failed: %s", exc)
        return _GAME_LOG_DK_CACHE.get(season, {})


def compute_player_tail_index(scores: list) -> float:
    """
    Estimate tail heaviness (ξ) of a player's DK score distribution using
    the Hill estimator — the standard extreme-value tail index estimator.

    Interpretation:
      ξ = 0.0  → thin tail (consistent scorer, safe cash play)
      ξ = 0.3  → moderate variance (typical mid-salary player)
      ξ = 0.6+ → heavy tail (boom-or-bust, ideal GPP target)

    Returns ξ ∈ [0.0, 1.50].  Defaults to 0.30 if insufficient data.
    """
    if len(scores) < 5:
        return 0.30
    arr = np.array(sorted(scores, reverse=True), dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 3:
        return 0.30
    k = max(3, round(len(arr) * 0.30))   # top 30% as tail
    threshold = arr[k - 1]
    if threshold <= 0:
        return 0.30
    xi = float(np.mean(np.log(arr[:k] / threshold)))
    return float(np.clip(xi, 0.0, 1.50))


def compute_regime_factor(game_total: float) -> float:
    """
    Game-level scoring regime multiplier derived from Vegas over/under total.

    High-total games produce more DFS points across both rosters.
    Without spread data this is total-only; extend to include spread
    when Vegas spread data is added to GAME_TOTALS.

    Returns multiplier in [0.93, 1.06].
    """
    if   game_total >= 235: return 1.06
    elif game_total >= 228: return 1.03
    elif game_total >= 220: return 1.00
    elif game_total >= 212: return 0.97
    else:                   return 0.93


def detect_cascade_players(players: pd.DataFrame) -> set:
    """
    Identify players at high risk of cascade-inflated ownership.

    A cascade occurs when early DFS players adopt a minimum-salary fill
    with no hard news, and subsequent entrants blindly copy the choice,
    driving ownership far above a player's true EV-implied level.

    Detection criteria (all must be met):
      - Salary ≤ $3,500   (minimum-salary filler tier)
      - proj_own ≥ 20%    (our model already estimates high field interest)
      - proj_pts_dk ≥ 8   (sufficient to be a valid lineup filler)

    Returns a set of player_ids flagged as cascade-risk.
    For these players, effective proj_own is inflated 2.5× in the
    Stackelberg denominator, making the optimizer treat them as
    more heavily-owned than our baseline model estimates.
    """
    mask = (
        (players["salary"] <= 3500) &
        (players["proj_own"] >= 20) &
        (players["proj_pts_dk"] >= 8)
    )
    return set(players.loc[mask, "player_id"].astype(str).tolist())


def enrich_projections(
    players: pd.DataFrame,
    season: str = "2025-26",
    game_log_cache: dict | None = None,
) -> pd.DataFrame:
    """
    Post-process build_projections() output with four enhancements:

    1. Tail Index (ξ) — fitted from each player's game-log DK score
       distribution using the Hill estimator. Heavy-tailed players
       receive wider sampling variance in Thompson mode.

    2. Regime Factor — game-level multiplier from Vegas over/under total.
       High-total games scale proj_pts_dk upward before optimization.

    3. Cascade Detection — minimum-salary players with high estimated
       ownership get their effective proj_own inflated 2.5× in the
       scoring denominator to reduce optimizer over-reliance on them.

    4. Recomputed ceiling — replaced with tail-adjusted formula:
         ceiling = proj_pts + (1.28 + 0.72 * ξ) * proj_std
       Normal (ξ=0): ceiling = proj_pts + 1.28σ  (unchanged)
       Heavy (ξ=1):  ceiling = proj_pts + 2.00σ  (20% wider upside)

    The static gpp_score column is also updated to reflect the new ceiling.
    Thompson-sampled Stackelberg gpp_scores are computed PER LINEUP ITERATION
    inside generate_gpp_lineups() — not here.
    """
    df = players.copy()

    # ── 1. Tail Index ─────────────────────────────────────────────────────────
    logs = game_log_cache if game_log_cache is not None else fetch_player_dk_game_logs(season)
    tail_indices = []
    for _, row in df.iterrows():
        name_l = str(row.get("name", "")).lower().strip()
        scores = logs.get(name_l, [])
        if not scores:
            # Fuzzy fallback: last-name match within same team
            team = str(row.get("team", ""))
            last = name_l.split()[-1] if name_l else ""
            for k, v in logs.items():
                if k.endswith(last) and len(v) >= 5:
                    scores = v
                    break
        tail_indices.append(compute_player_tail_index(scores))
    df["tail_index"] = tail_indices

    # ── 2. Regime Factor ──────────────────────────────────────────────────────
    df["regime_factor"] = df["game_total"].apply(compute_regime_factor)
    df["proj_pts_dk"]   = (df["proj_pts_dk"] * df["regime_factor"]).round(2)

    # ── 3. Cascade Detection ─────────────────────────────────────────────────
    cascade_ids = detect_cascade_players(df)
    if cascade_ids:
        cascade_mask = df["player_id"].astype(str).isin(cascade_ids)
        df.loc[cascade_mask, "proj_own"] = (
            df.loc[cascade_mask, "proj_own"] * 2.5
        ).clip(upper=65).round(1)

    # ── 4. Tail-adjusted ceiling ──────────────────────────────────────────────
    # ceiling = proj_pts + (1.28 + 0.72 * ξ) * σ
    # At ξ=0 this equals the existing normal ceiling; at ξ=1 it's 2.0σ above mean.
    tail_z       = 1.28 + 0.72 * df["tail_index"]
    df["ceiling"] = (df["proj_pts_dk"] + tail_z * df["proj_std"]).round(2)

    # ── Recompute static gpp_score with updated ceiling and proj_own ──────────
    # (The per-iteration Stackelberg score is computed separately in the loop)
    df["gpp_score"] = (
        df["ceiling"] * 0.60 +
        df["proj_pts_dk"] * 0.25 +
        (1 - df["proj_own"] / 100) * 10
    ).round(3)

    import logging
    n_cascade = len(cascade_ids)
    n_tail_gt_half = int((df["tail_index"] > 0.5).sum())
    logging.info(
        "[enrich] tail_index applied (%d heavy-tail players >0.5), "
        "%d cascade players flagged, regime range [%.2f, %.2f]",
        n_tail_gt_half, n_cascade,
        df["regime_factor"].min(), df["regime_factor"].max(),
    )
    return df


# ── Projection logic ──────────────────────────────────────────────────────────
def _load_espn_mismatch_signals(cutoff_date: str = "") -> dict:
    """
    Load ESPN game log cache and compute price-mismatch signals per player.

    Returns: {name_lower: {recent5, season, form_ratio, mismatch}}

    Key signals:
      - recent5: avg DK pts over last 5 games (before cutoff_date)
      - season: avg DK pts over full season (before cutoff_date)
      - form_ratio: recent5 / season — >1.2 = HOT, <0.8 = COLD
      - mismatch: recent5 / (salary/1000) — pts per $1K (value signal)

    Calibrated against 4-night backtest (3/6-3/9/2026):
      - Precious Achiuwa: recent5=40.2 vs season=20.7 (1.94x) = MASSIVE usage spike
      - Gui Santos: recent5=31.6 vs season=17.6 (1.80x) = HOT streak
      - Josh Giddey: recent5=37.0 vs season=42.7 (0.87x) = declining = wrong pick
    """
    _DK_PTS  = 1.0; _DK_3PM = 0.5; _DK_REB = 1.25; _DK_AST = 1.5
    _DK_STL  = 2.0; _DK_BLK = 2.0; _DK_TOV = -0.5
    _DK_DD   = 1.5; _DK_TD  = 3.0

    def _calc_dk(gl: dict, i: str) -> float:
        pts = float(gl.get("pts", {}).get(i, 0))
        fg3 = float(gl.get("fg3", {}).get(i, 0))
        reb = float(gl.get("trb", {}).get(i, 0))
        ast = float(gl.get("ast", {}).get(i, 0))
        stl = float(gl.get("stl", {}).get(i, 0))
        blk = float(gl.get("blk", {}).get(i, 0))
        tov = float(gl.get("tov", {}).get(i, 0))
        cats = [pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10]
        dd = _DK_DD if sum(cats) >= 2 else 0
        td = _DK_TD if sum(cats) >= 3 else 0
        return (pts * _DK_PTS + fg3 * _DK_3PM + reb * _DK_REB +
                ast * _DK_AST + stl * _DK_STL + blk * _DK_BLK +
                tov * _DK_TOV + dd + td)

    # Find ESPN cache directory
    cache_dir = Path(__file__).parent.parent / "cache" / "espn"
    if not cache_dir.exists():
        return {}

    # Build name -> ESPN ID mapping from roster files
    name_to_espn: dict[str, str] = {}
    for roster_file in cache_dir.glob("roster_*_2026.json"):
        try:
            with open(roster_file) as f:
                name_to_espn.update(json.load(f))
        except Exception:
            pass

    if not name_to_espn:
        return {}

    # Load all gamelogs (lazy: only the files we need below)
    espn_gl_cache: dict[str, dict] = {}

    def _get_gl(espn_id: str) -> dict:
        if espn_id not in espn_gl_cache:
            p = cache_dir / f"gamelog_{espn_id}_2026.json"
            if p.exists():
                try:
                    with open(p) as f:
                        espn_gl_cache[espn_id] = json.load(f)
                except Exception:
                    espn_gl_cache[espn_id] = {}
            else:
                espn_gl_cache[espn_id] = {}
        return espn_gl_cache[espn_id]

    signals: dict[str, dict] = {}
    for name_lower, espn_id in name_to_espn.items():
        gl = _get_gl(espn_id)
        if not gl or "date" not in gl:
            continue

        scores: list[tuple[str, float]] = []
        for i in gl["date"]:
            d = str(gl["date"][i])
            if cutoff_date and d >= cutoff_date:
                continue  # no future leakage
            mp = float(gl.get("mp", {}).get(i, 0))
            if mp < 5:
                continue  # garbage time — exclude
            dk = _calc_dk(gl, i)
            scores.append((d, dk))

        if not scores:
            continue

        scores.sort(key=lambda x: x[0], reverse=True)  # newest first
        recent5 = [s for _, s in scores[:5]]
        season  = [s for _, s in scores]

        r5  = float(sum(recent5) / len(recent5))
        sea = float(sum(season)  / len(season))

        signals[name_lower] = {
            "recent5":    round(r5, 2),
            "season":     round(sea, 2),
            "form_ratio": round(r5 / max(sea, 1.0), 3),
            "n_games":    len(season),
        }

    return signals


import logging as _logging_fc  # noqa: E402 — used by load_fc_data / _merge_fc


def load_fc_data(fc_path: "Path | str | None" = None) -> "pd.DataFrame | None":
    """
    Load a Fantasy Cruncher CSV export (header row 2) and return a clean DataFrame
    with columns: fc_name, fc_proj, fc_own, fc_mins, fc_floor, fc_ceiling.

    Fantasy Cruncher CSV has two header rows:
      Row 1: section labels (Game and Vegas, Player Averages - Season, etc.)
      Row 2: actual column names (Player, Salary, Proj Own%, Proj Mins, FC Proj, Floor, Ceiling...)

    Returns None if path not found or parsing fails.
    """
    import glob as _glob
    if fc_path is None:
        # Auto-discover: look for draftkings_NBA_*.csv in contest/ folder
        contest_dir = Path(__file__).parent.parent / "contest"
        matches = sorted(_glob.glob(str(contest_dir / "draftkings_NBA_*.csv")))
        if not matches:
            return None
        fc_path = matches[-1]   # most recent file

    try:
        fc_path = Path(fc_path)
        if not fc_path.exists():
            return None
        df = pd.read_csv(fc_path, header=1, encoding="utf-8-sig")
        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]
        required = {"Player", "FC Proj", "Proj Own%", "Proj Mins"}
        if not required.issubset(df.columns):
            return None
        out = pd.DataFrame()
        out["fc_name"]    = df["Player"].astype(str).str.strip()
        out["fc_proj"]    = pd.to_numeric(df["FC Proj"], errors="coerce")
        out["fc_own"]     = (
            df["Proj Own%"].astype(str).str.rstrip("%")
            .pipe(pd.to_numeric, errors="coerce")
        )
        out["fc_mins"]    = pd.to_numeric(df["Proj Mins"], errors="coerce")
        out["fc_floor"]   = pd.to_numeric(df.get("Floor",   pd.Series(dtype=float)), errors="coerce")
        out["fc_ceiling"] = pd.to_numeric(df.get("Ceiling", pd.Series(dtype=float)), errors="coerce")
        # Additional FC stats (corr with FPTS confirmed 3/13 backtest)
        out["fc_fppm"]    = pd.to_numeric(df.get("FPPM",    pd.Series(dtype=float)), errors="coerce")  # +0.428 corr
        out["fc_usg"]     = pd.to_numeric(df.get("USG",     pd.Series(dtype=float)), errors="coerce")  # +0.501 corr
        out["fc_stdv36"]  = pd.to_numeric(df.get("STDV/36", pd.Series(dtype=float)), errors="coerce")  # ceiling width
        out["fc_avg36"]   = pd.to_numeric(df.get("AVG/36",  pd.Series(dtype=float)), errors="coerce")  # +0.429 corr
        # Team/Opp implied totals + game total at FC export time (used as baseline
        # for line movement detection vs current Odds API fetch)
        out["fc_team_pts"] = pd.to_numeric(df.get("TeamPts", pd.Series(dtype=float)), errors="coerce")
        out["fc_opp_pts"]  = pd.to_numeric(df.get("OppPts",  pd.Series(dtype=float)), errors="coerce")
        out["fc_total"]    = (out["fc_team_pts"].fillna(0) + out["fc_opp_pts"].fillna(0)).where(
            out["fc_team_pts"].notna() & out["fc_opp_pts"].notna()
        ).round(1)
        # FPPM-based projection: proj_mins * FPPM = direct DK point estimate
        out["fc_fppm_proj"] = (out["fc_mins"] * out["fc_fppm"]).round(2)
        # DVP (Def v Pos): raw matchup advantage score from FC export
        # Positive = favorable matchup, negative = tough matchup
        _dvp_raw = df.get("Def v Pos", df.get("DVP", pd.Series(dtype=float)))
        out["fc_dvp"] = pd.to_numeric(_dvp_raw, errors="coerce")
        # Role / injury signals
        out["fc_season_min"]  = pd.to_numeric(df.get("Min",    pd.Series(dtype=float)), errors="coerce")  # season avg mins
        out["fc_consistency"] = pd.to_numeric(df.get("Con.",   pd.Series(dtype=float)), errors="coerce")  # game-to-game consistency %
        out["fc_proj_stdv"]   = pd.to_numeric(df.get("STDV.1", pd.Series(dtype=float)), errors="coerce")  # projection std dev
        _start_raw = df.get("Start", pd.Series(dtype=str))
        out["fc_starter"]     = (_start_raw.astype(str).str.strip().str.upper() == "Y").astype(float)
        # mins_expansion: proj_mins / season_min — >1.3x = expanded role (injury replacement signal)
        _season = out["fc_season_min"].fillna(10).clip(lower=5)
        out["fc_mins_expansion"] = (out["fc_mins"] / _season).round(3)
        # GPP score: ceiling per salary_k × ownership leverage × mins_expansion
        # This is what top-1% pros optimize for in GPP — not raw projection
        # Validated 3/6-3/12: surfaces Rupert(2.25x), Wolf, Fontecchio, Dru Smith,
        # Killian Hayes(1.96x) which all appeared at 25-80% in top-1% lineups
        _sal_k     = (out["fc_floor"].fillna(0) * 0).add(1)   # placeholder; real salary from merge
        out["fc_gpp_score"]   = float("nan")  # computed in _merge_fc after salary is available
        # Drop rows with no usable projection
        out = out[out["fc_proj"].notna() & (out["fc_proj"] > 0)].copy()
        return out
    except Exception as _exc:
        _logging_fc.warning("[fc] Failed to load FC data from %s: %s", fc_path, _exc)
        return None


def _merge_fc(players: pd.DataFrame, fc: "pd.DataFrame | None") -> pd.DataFrame:
    """
    Merge Fantasy Cruncher data into player pool by fuzzy name match.
    Adds columns: fc_proj, fc_own, fc_mins, fc_floor, fc_ceiling,
                  fc_fppm, fc_usg, fc_stdv36, fc_avg36, fc_team_pts, fc_fppm_proj, fc_dvp.
    Players with no FC match get NaN in those columns.
    """
    _ALL_FC_COLS = ("fc_proj", "fc_own", "fc_mins", "fc_floor", "fc_ceiling",
                    "fc_fppm", "fc_usg", "fc_stdv36", "fc_avg36", "fc_team_pts", "fc_fppm_proj",
                    "fc_dvp", "fc_season_min", "fc_consistency", "fc_proj_stdv",
                    "fc_starter", "fc_mins_expansion", "fc_gpp_score")
    if fc is None or fc.empty:
        for col in _ALL_FC_COLS:
            players[col] = float("nan")
        return players

    from difflib import get_close_matches as _gcm

    fc_name_lc = {n.lower(): i for i, n in enumerate(fc["fc_name"])}
    fc_cols = [c for c in _ALL_FC_COLS if c in fc.columns]
    for col in _ALL_FC_COLS:
        players[col] = float("nan")

    for pidx, row in players.iterrows():
        name_lc = str(row.get("name", "")).lower().strip()
        # Exact match first
        fi = fc_name_lc.get(name_lc)
        if fi is None:
            # Fuzzy match
            close = _gcm(name_lc, list(fc_name_lc.keys()), n=1, cutoff=0.82)
            if close:
                fi = fc_name_lc[close[0]]
        if fi is not None:
            fc_row = fc.iloc[fi]
            for col in fc_cols:
                players.at[pidx, col] = fc_row[col]

    matched = players["fc_proj"].notna().sum()
    _logging_fc.info("[fc] Merged FC data: %d / %d players matched", matched, len(players))

    # Compute fc_gpp_score now that salary is available
    # Formula: ceiling × (1 - proj_own%) × mins_expansion / salary_k
    # Validated 3/6-3/12: correctly ranks injury-replacement plays (Rupert, Wolf,
    # Fontecchio, Dru Smith, Killian Hayes) that appeared in 25-80% of top-1% lineups
    # but were missed by pure fc_proj ranking.
    has_ceil = "fc_ceiling" in players.columns and players["fc_ceiling"].notna().any()
    has_own  = "fc_own"     in players.columns and players["fc_own"].notna().any()
    has_exp  = "fc_mins_expansion" in players.columns and players["fc_mins_expansion"].notna().any()
    if has_ceil and "salary" in players.columns:
        _sal_k   = players["salary"].clip(lower=1000) / 1000
        _ceil    = players["fc_ceiling"].fillna(0)
        _own_lev = (1 - players["fc_own"].fillna(15) / 100).clip(lower=0.3) if has_own else 1.0
        _exp     = players["fc_mins_expansion"].fillna(1.0).clip(upper=2.5) if has_exp else 1.0
        players["fc_gpp_score"] = (_ceil / _sal_k * _own_lev * _exp).round(3)
    else:
        players["fc_gpp_score"] = float("nan")

    return players


def build_projections(df: pd.DataFrame, cutoff_date: str = "") -> pd.DataFrame:
    """
    Enhanced projections using:
    - AvgPointsPerGame as baseline
    - Game total adjustment (+/- from 225 baseline)
    - Home/away split
    - Team implied total weighting
    - Value tier identification
    - Injury/0-avg filtering
    - ESPN game log price-mismatch signal (new: form ratio + pts-per-$K)
    """
    out = df.copy()

    # ── Status classification ─────────────────────────────────────────────
    # DK prices players at $3,000 (minimum) when they are not expected to play.
    # A player with 0 avg but HIGHER salary is being priced to PLAY — they are
    # either a season-debut (e.g. Tatum returning from injury) or a recently-
    # called-up player.  We must NOT exclude them.
    out["status"] = "ACTIVE"

    # ── Archetype classification ───────────────────────────────────────────
    out["archetype"] = out.apply(
        lambda r: classify_player_archetype(r["primary_position"], r["salary"], r["avg_pts"]),
        axis=1,
    )

    # Stub columns populated later by apply_b2b_adjustments / apply_dvp_adjustments
    out["is_b2b"]      = False
    out["b2b_penalty"] = 0.0
    out["b2b_boost"]   = 0.0
    out["dvp_mult"]    = 1.0

    # Opponent team — derived from matchup string ("AWAY@HOME")
    # Available immediately so every downstream function (DvP, correlation,
    # apply_lineup_confirmation_dvp) can use row["opp"] without re-parsing.
    def _opp_from_matchup(row) -> str:
        m = str(row.get("matchup", ""))
        t = str(row.get("team", ""))
        if "@" in m:
            away, home = m.split("@", 1)
            away, home = away.strip(), home.strip()
            if t == away: return home
            if t == home: return away
        return ""
    out["opp"] = out.apply(_opp_from_matchup, axis=1)

    # Stub columns populated later by enrich_projections()
    # Defaults ensure the columns always exist even if the API is unreachable.
    # tail_index = 0.30 → moderate variance (safe default, renders gold ξ0.30)
    # regime_factor = 1.0 → neutral game-total multiplier
    out["tail_index"]    = 0.30
    out["regime_factor"] = 1.0

    # True OUTs: minimum DK salary ($3,000) AND 0 avg → genuinely inactive
    out.loc[(out["avg_pts"] == 0) & (out["salary"] <= 3000), "status"] = "OUT"

    # DEBUT: above-minimum salary AND 0 avg → season debut / returning from injury
    # Use salary-implied projection: salary / 1000 * 4.5
    # (conservative — slightly below the 5x typical value threshold)
    debut_mask = (out["avg_pts"] == 0) & (out["salary"] > 3000)
    out.loc[debut_mask, "status"]   = "DEBUT"
    out.loc[debut_mask, "avg_pts"]  = (out.loc[debut_mask, "salary"] / 1000 * 4.5).round(1)

    # Low-salary + low-avg players (bench depth, may not get minutes)
    out.loc[
        (out["salary"] < 4000) & (out["avg_pts"] > 0) & (out["avg_pts"] < 10) &
        (out["status"] == "ACTIVE"),
        "status"
    ] = "QUESTIONABLE"

    # Remove confirmed OUTs only
    out = out[out["status"] != "OUT"].copy()

    # GTD: 30% haircut applied after projection math (see below)
    # QUESTIONABLE: 15% haircut

    # Game total adjustment
    def game_total_factor(matchup: str, team: str, home_away: str) -> float:
        gt = GAME_TOTALS.get(matchup)
        if not gt:
            return 1.0
        baseline_total = 225.0
        # How much does this game's total deviate from average?
        total_factor = gt["total"] / baseline_total
        # Team implied total: home or away
        implied = gt["home_implied"] if home_away == "home" else gt["away_implied"]
        team_factor = implied / (baseline_total / 2)
        return (total_factor * 0.4 + team_factor * 0.6)

    out["game_total_factor"] = out.apply(
        lambda r: game_total_factor(r["matchup"], r["team"], r["home_away"]), axis=1
    )

    # Home court factor
    out["hca_factor"] = out["home_away"].map({"home": 1.025, "away": 0.985}).fillna(1.0)

    # Base projection = avg_pts * context factors
    out["proj_raw"] = (
        out["avg_pts"] *
        out["game_total_factor"] *
        out["hca_factor"]
    )

    # Regression to mean (don't let proj drift too far from avg)
    out["proj_pts_dk"] = (out["proj_raw"] * 0.70 + out["avg_pts"] * 0.30).round(2)

    # GTD haircut: 30% reduction (uncertain availability)
    out.loc[out["status"] == "GTD",          "proj_pts_dk"] *= 0.70
    # QUESTIONABLE haircut: 15% reduction
    out.loc[out["status"] == "QUESTIONABLE", "proj_pts_dk"] *= 0.85
    out["proj_pts_dk"] = out["proj_pts_dk"].round(2)

    # ESPN recency-weighted projections
    out["form_ratio"] = 1.0
    out["is_hot"] = False
    out["is_cold"] = False
    recency_upgrades = 0
    try:
        espn_client = ESPNDataClient()
        for idx, row in out.iterrows():
            player_name = str(row.get("name") or row.get("player_name") or "").strip()
            if not player_name:
                continue
            rec = espn_client.compute_recency_weighted_projection(player_name)
            if not rec:
                continue
            new_proj = rec.get("weighted_proj", out.at[idx, "proj_pts_dk"])
            out.at[idx, "proj_pts_dk"] = round(float(new_proj), 2)
            out.at[idx, "form_ratio"] = rec.get("form_ratio", 1.0)
            out.at[idx, "is_hot"] = bool(rec.get("is_hot", False))
            out.at[idx, "is_cold"] = bool(rec.get("is_cold", False))
            recency_upgrades += 1
        if recency_upgrades:
            print(f"[projections] ESPN recency weighting applied to {recency_upgrades} players")
        else:
            print("[projections] ESPN recency weighting applied to 0 players")
    except Exception as exc:
        print(f"[projections] ESPN recency weighting unavailable: {exc}")

    # Standard deviation estimate (Gamma-style: ~28% of projection)
    # DEBUT players get wider uncertainty (38%) — unknown minutes/usage
    out["proj_std"] = (out["proj_pts_dk"] * 0.28).round(2)
    out.loc[out["status"] == "DEBUT", "proj_std"] = (
        out.loc[out["status"] == "DEBUT", "proj_pts_dk"] * 0.38
    ).round(2)

    # NOTE: FC STDV/36 is NOT used for proj_std — 3/13 backtest showed -0.137
    # correlation with actual FPTS. High variance ≠ high score on typical slates.
    # FC Proj (+0.859) and Proj Mins (+0.791) are the signals that actually matter.

    # ── Explosion Profile Signals ─────────────────────────────────────────────
    # Four signals that predict 2-3x performance probability:
    #
    #   1. boom_rate       — historical P(dk >= 1.8x avg). Malik Monk at $4,500
    #                        has a measurably higher boom rate than a similar-priced
    #                        consistent player. Inflates ceiling for GPP targeting.
    #
    #   2. variance_ratio  — recent_std / season_std. > 1.3 = player is in a
    #                        "volatile" phase (could boom or bust). For GPP we
    #                        want this at the cheap tier.
    #
    #   3. game_env_mult   — pace × game_total interaction. High-pace + high-total
    #                        games generate more possessions → more counting stats
    #                        → higher ceiling for all players in that game.
    #
    #   4. salary_gap      — ceiling - (salary/1000 × 5). Positive = underpriced
    #                        relative to market expectation. Large gaps identify
    #                        players like Raynaud ($6,300, ceil=64, gap=+32.5).
    out["boom_rate"]      = 0.05   # baseline: 5% boom games (league average)
    out["variance_ratio"] = 1.0
    out["is_volatile"]    = False
    out["game_env_mult"]  = 1.0

    # 1 & 2: boom_rate + variance_ratio from ESPN game logs
    try:
        _exp_client = ESPNDataClient()
        boom_hits   = 0
        LG_PACE_EST = 100.0
        for idx, row in out.iterrows():
            player_name = str(row.get("name") or row.get("player_name") or "").strip()
            if not player_name:
                continue
            profile = _exp_client.compute_explosion_profile(player_name)
            if not profile:
                continue
            out.at[idx, "boom_rate"]      = profile["boom_rate"]
            out.at[idx, "variance_ratio"] = profile["variance_ratio"]
            out.at[idx, "is_volatile"]    = bool(profile["is_volatile"])
            boom_hits += 1
        print(f"[projections] Explosion profiles computed for {boom_hits} players")
    except Exception as _exc:
        print(f"[projections] Explosion profiles unavailable: {_exc}")

    # 3: game_env_mult — pace × game_total interaction per matchup
    # Normalised to 1.0 at league-average (pace=100, total=225).
    # Max boost ≈ 8% for a pace-110 / total-240 game.
    LG_TOTAL = 225.0
    LG_PACE  = 100.0
    for matchup in out["matchup"].unique():
        gt   = GAME_TOTALS.get(matchup, {})
        tot  = float(gt.get("total", LG_TOTAL))
        # Derive pace from away/home implied: higher implied totals → higher pace
        # (proxy: use implied ratio since we don't always have real pace here)
        pace = (tot / LG_TOTAL) * LG_PACE   # approximate; overridden by enrich_projections
        total_factor = (tot  / LG_TOTAL - 1.0) * 0.50   # 50% weight on game total
        pace_factor  = (pace / LG_PACE  - 1.0) * 0.30   # 30% weight on pace
        env_mult     = float(np.clip(1.0 + total_factor + pace_factor, 0.92, 1.10))
        out.loc[out["matchup"] == matchup, "game_env_mult"] = env_mult

    # ── Tournament ceiling: salary-tiered fat-tail multipliers ────────────────
    # Backtest (4 nights 3/6–3/9) calibration:
    #   Jaylin Williams  ($5700,  proj=20.4, actual=57.0) → 2.80x
    #   Maxime Raynaud   ($6300,  proj=23.0, actual=51.75) → 2.25x
    #   Malik Monk       ($4500,  proj=21.4, actual=46.0) → 2.15x
    #   Russell Westbrook($7000,  proj=34.2, actual=62.75) → 1.84x
    #   Tyler Herro      ($7400,  proj=32.2, actual=63.75) → 1.98x
    #   SGA              ($10700, proj=47.2, actual=75.75) → 1.60x
    # Formula: ceiling = min(proj * mult, proj + max_additional_pts)
    _CEIL_TIERS = [
        (10_000, 1.65, 30),   # Elite: $10K+ → 1.65x, max +30 pts
        ( 8_000, 1.85, 28),   # Star:  $8-10K → 1.85x, max +28 pts
        ( 6_500, 2.10, 32),   # Mid:   $6.5-8K → 2.10x, max +32 pts
        ( 5_000, 2.80, 40),   # Value: $5-6.5K → 2.80x, max +40 pts
        (     0, 3.20, 40),   # Cheap: <$5K → 3.20x, max +40 pts
    ]
    def _get_ceiling(proj, salary, boom_rate=0.05, game_env_mult=1.0):
        # Base ceiling from salary tier
        base = proj
        for sal_thresh, mult, max_add in _CEIL_TIERS:
            if salary >= sal_thresh:
                base = min(proj * mult, proj + max_add)
                break
        else:
            base = proj * 3.20

        # Boom-rate inflation: players with high explosion history get wider ceiling.
        # boom_rate=0.20 → +10% ceiling. Capped at +15% to avoid overcorrection.
        boom_inflation = float(np.clip(boom_rate * 0.50, 0.0, 0.15))

        # Game environment multiplier: high-pace / high-total games expand ceilings.
        # Already normalised to 1.0 at league average. Additional contribution
        # beyond 1.0 is applied at 60% weight (partial — game context is shared).
        env_inflation = float(np.clip((game_env_mult - 1.0) * 0.60, -0.05, 0.08))

        return round(base * (1.0 + boom_inflation + env_inflation), 2)

    out["ceiling"] = out.apply(
        lambda r: _get_ceiling(
            r["proj_pts_dk"], r["salary"],
            r.get("boom_rate", 0.05),
            r.get("game_env_mult", 1.0),
        ),
        axis=1,
    )
    out["floor"] = (out["proj_pts_dk"] - 1.28 * out["proj_std"]).clip(0).round(2)

    # Blend FC floor/ceiling when available — FC uses actual pace/matchup data
    if "fc_ceiling" in out.columns:
        has_fc_ceil = out["fc_ceiling"].notna() & (out["fc_ceiling"] > 0)
        out.loc[has_fc_ceil, "ceiling"] = (
            out.loc[has_fc_ceil, "fc_ceiling"] * 0.65 +
            out.loc[has_fc_ceil, "ceiling"]    * 0.35
        ).round(2)
    if "fc_floor" in out.columns and "floor" in out.columns:
        has_fc_floor = out["fc_floor"].notna() & (out["fc_floor"] > 0)
        out.loc[has_fc_floor, "floor"] = (
            out.loc[has_fc_floor, "fc_floor"] * 0.65 +
            out.loc[has_fc_floor, "floor"]    * 0.35
        ).round(2)

    # Value metric
    out["value"] = (out["proj_pts_dk"] / (out["salary"] / 1000)).round(3)

    # Salary inefficiency gap — positive = underpriced vs market expectation.
    # Market baseline: salary/1000 × 5 is the implied DK expectation at 5x value.
    # Large positive gaps (e.g. Raynaud $6,300 ceil=64 → gap=+32.5) flag players
    # the optimizer should weight more heavily in GPP builds.
    out["salary_gap"] = (out["ceiling"] - out["salary"] / 1000.0 * 5.0).round(2)

    # ── ON/OFF Injury Usage Absorption ───────────────────────────────────────
    # For every player the DK slate marks as OUT (or confirmed injured in injury
    # data), use BBRefOnOffAgent to compute empirical DK-point deltas for their
    # teammates and apply those boosts to projections.
    #
    # This is what Rotowire's "Starting Lineup Usage Rates" captures: when
    # Mitchell is OUT, Mobley's usage share jumps from 35% → 45%, translating
    # to a concrete DK-point boost.  Without this step, the optimizer ignores
    # the injury-replacement edge entirely.
    out["on_off_boost"] = 0.0  # tracks how much each player received
    try:
        from agents.bbref_on_off_agent import BBRefOnOffAgent
        _onoff_agent = BBRefOnOffAgent()
        _usage_data  = fetch_player_usage_rates()

        # Identify OUT players still present in the pool (status=OUT filtered
        # earlier, but GTD/QUESTIONABLE starters still here and may be ruled
        # out tonight)
        # Also look for players who were removed (status==OUT) by checking
        # if any team on the slate is missing a high-salary player
        injured_rows = []
        for team in out["team"].unique():
            team_df = df[df["team"] == team] if "team" in df.columns else pd.DataFrame()
            pool_df = out[out["team"] == team]
            if team_df.empty:
                continue
            # Any player in original slate not in active pool = confirmed OUT
            orig_ids = set(team_df["player_id"].astype(str))
            pool_ids = set(pool_df["player_id"].astype(str))
            for missing_id in orig_ids - pool_ids:
                row = team_df[team_df["player_id"].astype(str) == missing_id]
                if not row.empty and float(row.iloc[0].get("salary", 0)) >= 4500:
                    injured_rows.append(row.iloc[0])

        if injured_rows:
            on_off_map = _onoff_agent.compute(injured_rows, out)
            for out_row in injured_rows:
                out_pid  = str(out_row.get("player_id", ""))
                oo_entry = on_off_map.get(out_pid)
                before   = out["proj_pts_dk"].copy()
                out      = estimate_usage_absorption(
                    out_row, out,
                    usage_data=_usage_data,
                    on_off_data=oo_entry,
                )
                boost = (out["proj_pts_dk"] - before).clip(lower=0)
                out["on_off_boost"] = (out["on_off_boost"] + boost).round(2)
                n_boosted = int((boost > 0.1).sum())
                print(f"[on_off] {out_row.get('name','?')} OUT -> {n_boosted} teammates boosted")

    except Exception as _exc:
        import logging
        logging.debug("[on_off] Usage absorption skipped: %s", _exc)

    # Game total for stack prioritization — computed first so ownership can use it
    def get_game_total(matchup):
        gt = GAME_TOTALS.get(matchup, {})
        return gt.get("total", 225.0)
    out["game_total"] = out["matchup"].apply(get_game_total)

    # Ownership estimate — uses FC (Fantasy Cruncher) projected ownership when available,
    # falls back to salary+projection rank formula.
    # FC ownership is far more accurate: it reflects actual public behavior (DK avg pts,
    # name recognition, chalk patterns) rather than just salary percentile rank.
    sal_rank  = out["salary"].rank(pct=True)
    proj_rank = out["proj_pts_dk"].rank(pct=True)
    formula_own = sal_rank ** 2 * 20 + proj_rank * 10 + 2

    # Slate concentration adjustment (applied to formula fallback only)
    n_games = int(out["matchup"].nunique())
    if n_games <= 4:
        slate_mult = 1.12
    elif n_games >= 9:
        slate_mult = 0.90
    else:
        slate_mult = 1.0
    formula_own = formula_own * slate_mult

    # Game total attraction (formula fallback only)
    gt_q75 = out["game_total"].quantile(0.75)
    gt_q25 = out["game_total"].quantile(0.25)
    gt_mult = pd.Series(1.0, index=out.index)
    gt_mult[out["game_total"] >= gt_q75] = 1.08
    gt_mult[out["game_total"] <  gt_q25] = 0.93
    formula_own = formula_own * gt_mult

    # Use FC ownership where available; blend (70% FC / 30% formula) for players with FC data
    if "fc_own" in out.columns:
        has_fc = out["fc_own"].notna()
        out["proj_own"] = formula_own.copy()
        out.loc[has_fc, "proj_own"] = (
            out.loc[has_fc, "fc_own"] * 0.70 +
            formula_own.loc[has_fc]   * 0.30
        )
    else:
        out["proj_own"] = formula_own

    # 3. Injury haircut: uncertain players draw less public ownership
    inj_mult = pd.Series(1.0, index=out.index)
    inj_mult[out["status"] == "GTD"]          = 0.72
    inj_mult[out["status"] == "QUESTIONABLE"] = 0.60
    inj_mult[out["status"] == "DOUBTFUL"]     = 0.35

    out["proj_own"] = (out["proj_own"] * inj_mult).clip(1, 40).round(1)

    # Blend projection: 3-way FC Proj + FPPM*Mins + our model.
    # 3/13 backtest correlations: FC Proj +0.859, Proj Mins +0.791, USG +0.501
    # FPPM * Proj Mins = direct DK point estimate (e.g. Murray: 1.33 * 30.25 = 40.2)
    # Blending FPPM-based estimate reduces over-reliance on FC Proj for players
    # with unusual matchup pace or role changes not captured by season FPPM.
    if "fc_proj" in out.columns:
        has_fc = out["fc_proj"].notna() & (out["fc_proj"] > 0)
        has_fppm = (
            out.get("fc_fppm_proj", pd.Series(dtype=float)).notna() &
            (out.get("fc_fppm_proj", pd.Series(dtype=float)) > 0)
        ) if "fc_fppm_proj" in out.columns else pd.Series(False, index=out.index)

        fc_n = has_fc.sum()
        if fc_n > 0:
            # Where both FC Proj and FPPM*Mins are available: 3-way blend
            both = has_fc & has_fppm
            fc_only = has_fc & ~has_fppm
            out.loc[both, "proj_pts_dk"] = (
                out.loc[both, "fc_proj"]        * 0.55 +
                out.loc[both, "fc_fppm_proj"]   * 0.20 +
                out.loc[both, "proj_pts_dk"]    * 0.25
            ).round(2)
            # Where only FC Proj is available: 2-way blend
            out.loc[fc_only, "proj_pts_dk"] = (
                out.loc[fc_only, "fc_proj"]     * 0.60 +
                out.loc[fc_only, "proj_pts_dk"] * 0.40
            ).round(2)
            _logging_fc.info("[fc] Blended FC+FPPM projection for %d players (%d 3-way)", fc_n, int(both.sum()))

    # Use FC projected minutes for more accurate DNP risk when available.
    # Pros use proj_mins as a DNP proxy: < 15 mins projected = genuine DNP risk.
    if "fc_mins" in out.columns:
        has_fc_mins = out["fc_mins"].notna()
        # Reclassify: very low projected minutes = high DNP risk regardless of salary
        out.loc[has_fc_mins & (out["fc_mins"] < 15), "dnp_risk_fc"] = 0.50
        out.loc[has_fc_mins & (out["fc_mins"].between(15, 22)), "dnp_risk_fc"] = 0.20
        out.loc[has_fc_mins & (out["fc_mins"] > 22), "dnp_risk_fc"] = 0.05
        # Apply FC-based DNP risk override (take max of salary-tier vs FC-mins risk)

    # ── DNP Risk Assessment (tournament-critical) ─────────────────────────────
    # Postmortem March 8: Grant Nelson 0pt (30% exp), Mo Bamba 0pt (30% exp),
    # Malevy Leons 6pt (45% exp), Tyler Kolek 1.5pt (15% exp), Jalen Pickett 0pt.
    # These cheap "usage spike" picks DNP'd or played junk minutes — 14 lineup-slots
    # scored 0 pts. A single 0-point player tanks a lineup by 25-40 pts.
    #
    # DNP risk formula (calibrated from 5-slate analysis):
    #   < $3,500 salary              → 50% DNP risk (emergency call-ups)
    #   $3,500–$4,000 AND avg < 12   → 40% DNP risk (fringe rotation)
    #   $4,000–$4,500 AND avg < 16   → 25% DNP risk (spot minutes)
    #   $4,500–$5,000 AND avg < 18   → 12% DNP risk (borderline starter)
    #   All others                   → 5%  DNP risk (baseline)
    out["dnp_risk"] = 0.05  # baseline
    out.loc[out["salary"] < 3500, "dnp_risk"] = 0.50
    out.loc[(out["salary"] < 4000) & (out["avg_pts"] < 12), "dnp_risk"] = 0.40
    out.loc[(out["salary"] < 4500) & (out["avg_pts"] < 16), "dnp_risk"] = 0.25
    out.loc[(out["salary"] < 5000) & (out["avg_pts"] < 18), "dnp_risk"] = 0.12
    # Already-flagged QUESTIONABLE players carry elevated DNP risk on top of status haircut
    out.loc[out["status"] == "QUESTIONABLE", "dnp_risk"] = out.loc[
        out["status"] == "QUESTIONABLE", "dnp_risk"
    ].clip(lower=0.20)
    # FC minutes override: if FC projects < 15 mins, treat as high DNP risk regardless of salary tier
    if "fc_mins" in out.columns:
        fc_mins_mask = out["fc_mins"].notna()
        out.loc[fc_mins_mask & (out["fc_mins"] < 15), "dnp_risk"] = out.loc[
            fc_mins_mask & (out["fc_mins"] < 15), "dnp_risk"
        ].clip(lower=0.45)
        out.loc[fc_mins_mask & (out["fc_mins"].between(15, 22)), "dnp_risk"] = out.loc[
            fc_mins_mask & (out["fc_mins"].between(15, 22)), "dnp_risk"
        ].clip(lower=0.18)
        # DOWNWARD override: confirmed high-minute players must not be removed by salary-tier dnp_risk.
        # E.g. Mitchell Robinson ($4K, fc_mins=24) gets dnp_risk=0.40 from salary tier alone,
        # which removes him before the bench filter runs. Cap dnp_risk at 0.05 for fc_mins >= 24,
        # and at 0.12 for fc_mins 20-23. Salary is NOT a proxy for minutes.
        out.loc[fc_mins_mask & (out["fc_mins"] >= 24), "dnp_risk"] = out.loc[
            fc_mins_mask & (out["fc_mins"] >= 24), "dnp_risk"
        ].clip(upper=0.05)
        out.loc[fc_mins_mask & (out["fc_mins"].between(20, 23.9)), "dnp_risk"] = out.loc[
            fc_mins_mask & (out["fc_mins"].between(20, 23.9)), "dnp_risk"
        ].clip(upper=0.12)

    # ── Injury-replacement override: mins_expansion >1.5x + meaningful projection ──
    # Players whose FC-projected minutes are 50%+ above their season average are
    # filling in for an OUT starter. Their salary-tier DNP risk is wrong — they
    # ARE playing (FC already confirmed this by projecting them for starter minutes).
    # Validated: Rupert 2.25x (39%), Hayes 1.96x (81%), Gardner 2.55x (28%),
    # Fontecchio 1.44x (59%) — all appeared heavily in top-1% lineups.
    if "fc_mins_expansion" in out.columns:
        _exp_mask = (
            out["fc_mins_expansion"].fillna(1.0) >= 1.50
        ) & (
            out["fc_proj"].fillna(0) >= 15
        ) & (
            out["fc_own"].fillna(20) <= 25   # not already heavy chalk
        )
        out.loc[_exp_mask, "dnp_risk"] = out.loc[_exp_mask, "dnp_risk"].clip(upper=0.08)

    # ── Projected minutes (best estimate from all available signals) ─────────
    # IMPORTANT: avg_pts was removed as a fallback signal. avg_pts is a HISTORICAL
    # season average — Larry Nance Jr. might average 12 DK pts/game (→ 18 est. min)
    # but currently only plays 10 minutes per game. avg_pts is NOT a current-slate
    # minutes predictor for bench/rotation players. Only two signals are reliable:
    #   1. FC projected minutes  (authoritative — use when available)
    #   2. Salary proxy          (conservative floor: $3K≈12min, $4K≈15min, $5K≈19min)
    # Sub-$5K players without FC data get the conservative proxy, which puts them
    # near or under the 15-min threshold where they belong.
    def _compute_proj_mins(row):
        fc = row.get("fc_mins")
        if pd.notna(fc) and float(fc) > 0:
            return float(fc)
        sal = float(row.get("salary", 0) or 0)
        # Standard salary proxy: $9K starter ≈ 35 min
        return max(0.0, round((sal / 9000.0) * 35.0, 1))

    out["proj_mins"] = out.apply(_compute_proj_mins, axis=1)

    # DNP-adjusted expected value: E[score] = P(plays) * projection
    out["dnp_adj_proj"] = (out["proj_pts_dk"] * (1 - out["dnp_risk"])).round(2)

    # ── Tournament GPP Score (ceiling-first, value-adjusted, DNP-penalized) ──
    # Backtest finding: winning lineups consistently use cheap players who score
    # 2-3x projection. The OLD formula (ceiling * 0.65) undervalued them because
    # absolute ceiling = proj * 1.38 max. NEW formula adds value-per-dollar bonus
    # so a $5700 player with ceiling=57 scores comparably to a $10700 stud.
    #
    # Components:
    #   1. ceiling * 0.50 — tournament upside (was 0.65, reduced to make room for value)
    #   2. proj * 0.15    — projection floor (was 0.20)
    #   3. (1-own)*8      — leverage bonus (unchanged)
    #   4. value_bonus    — ceiling per $K salary bonus (NEW: rewards cheap explosions)
    #      = clip(ceiling_per_1K - 4.0, 0, 10) * 0.60
    #      High value: Jaylin Williams ($5700, ceil=57) → 57/5.7=10.0 → bonus=3.6
    #      Low value:  Jokic ($12400, ceil=92) → 92/12.4=7.4 → bonus=2.0
    play_prob = (1.0 - out["dnp_risk"])
    value_ceil_per_k = out["ceiling"] / (out["salary"] / 1000.0)
    value_bonus = (value_ceil_per_k - 4.0).clip(0, 10) * 0.60

    # Salary gap bonus: rewards structurally underpriced players.
    # salary_gap = ceiling - (salary/1000 * 5).  Normalised: cap at 15pts gap → +1.5 pts.
    gap_bonus = out["salary_gap"].clip(0, 30) * 0.05

    # Volatile bonus: players in variance-expansion phase get a small GPP lift.
    # Only applied at cheap/mid tiers where variance translates to GPP edge.
    vol_bonus = (
        (out["variance_ratio"] - 1.0).clip(0, 1.0) * 0.8 *
        (out["salary"] < 7500).astype(float)   # only cheap/mid tier
    )

    # USG bonus (fc_usg corr +0.501): each % above league avg (20%) → +0.25, capped +6.
    # Added here AND in recompute_proj_own_and_gpp() for consistency.
    if "fc_usg" in out.columns:
        _usg_bonus_build = ((out["fc_usg"].fillna(20.0) - 20.0) * 0.25).clip(0, 6)
    else:
        _usg_bonus_build = 0.0

    # Use calibrated field ownership (FC proj_own / 3 = realistic actual ownership)
    # Pro data (7-date, 800+ lineups): FC proj_own is 3x higher than actual field own.
    # FC 18% → actual ~5%, FC 28% → actual ~9%.
    _field_own = out["proj_own"] / 3.0

    out["gpp_score"] = (
        out["ceiling"]     * 0.50 * play_prob +
        out["proj_pts_dk"] * 0.15 * play_prob +
        (1 - _field_own / 100) * 8 +
        value_bonus        * play_prob +
        gap_bonus          * play_prob +
        vol_bonus          * play_prob +
        _usg_bonus_build   * play_prob
    ).round(3)
    # DNP penalty for very-high-risk plays (>30%)
    high_risk = out["dnp_risk"] >= 0.30
    out.loc[high_risk, "gpp_score"] = (out.loc[high_risk, "gpp_score"] * 0.85).round(3)

    # ── Field ownership calibration ──────────────────────────────────────────
    # Pro data (7-date, 800+ lineups): FC proj_own is 3x higher than actual.
    # FC 18% → actual ~5%, FC 28% → actual ~9%.
    # field_own_est is the realistic estimate of what % of the field will have
    # this player, used for leverage scoring and ownership penalty.
    out["field_own_est"] = (out["proj_own"] / 3.0).clip(upper=35.0)

    # ── FC vs season average ratio ───────────────────────────────────────────
    # Key pro signal: when FC projects a player 15%+ above their season avg,
    # it signals a good matchup/usage boost → gpp_score boost.
    # When 15%+ below, it signals bad matchup/health concern → gpp_score cut.
    _avg_valid = (
        out["avg_pts"].notna() & (out["avg_pts"] > 5.0)
        & out["fc_proj"].notna() & (out["fc_proj"] > 0)
    ) if "fc_proj" in out.columns else pd.Series(False, index=out.index)
    out["fc_vs_avg_ratio"] = 1.0
    if _avg_valid.any():
        out.loc[_avg_valid, "fc_vs_avg_ratio"] = (
            out.loc[_avg_valid, "fc_proj"] / out.loc[_avg_valid, "avg_pts"]
        ).clip(0.3, 3.0)
    # Apply ratio bonus/penalty directly to gpp_score:
    # +15% above avg → up to +4 gpp pts; -15% below avg → up to -3 gpp pts
    _ratio_bonus = ((out["fc_vs_avg_ratio"] - 1.0) * 10).clip(-3.0, 4.0)
    out["gpp_score"] = (out["gpp_score"] + _ratio_bonus).clip(lower=0)

    # ── ESPN Price-Mismatch Signal (new) ──────────────────────────────────────
    # Load real ESPN game log data to identify players who are:
    #   A) HOT: recent form significantly above season average (usage spike, hot streak)
    #   B) UNDERPRICED: priced below their recent DK output (value opportunity)
    #
    # Calibrated finding (4-night backtest 3/6-3/9):
    #   Precious Achiuwa: recent5=40.2 vs season=20.7 (form_ratio=1.94) → won 3/8!
    #   Gui Santos:       recent5=31.6 vs season=17.6 (form_ratio=1.80) → won 3/9!
    #   Josh Giddey:      recent5=37.0 vs season=42.7 (form_ratio=0.87) → wrong pick!
    #
    # Without this signal: ILP picks Giddey (highest gpp) over Achiuwa/Sexton/Westbrook.
    # With this signal: Achiuwa's gpp_score jumps above Giddey's → correct picks.
    mismatch_signals = _load_espn_mismatch_signals(cutoff_date=cutoff_date)

    if mismatch_signals:
        for idx, row in out.iterrows():
            name_key = str(row.get("name", "")).lower().strip()
            sig = mismatch_signals.get(name_key)
            if not sig or sig["n_games"] < 5:
                continue  # need enough history to trust the signal

            r5         = sig["recent5"]
            sea        = sig["season"]
            form_ratio = sig["form_ratio"]
            salary     = int(row.get("salary", 5000))
            mismatch   = r5 / (salary / 1000.0) if salary > 0 else 0

            # Form bonus: rewards players trending UP (>10% above season avg)
            # Max +10 pts for players at 2x their season average (usage spike)
            form_bonus = max(0.0, (form_ratio - 1.0) * 10.0)

            # Mismatch bonus: rewards underpriced players (>4 pts per $1K)
            # A player scoring 40 pts at $6,100 (6.56 pts/$K) gets +3.8 pts
            mismatch_bonus = max(0.0, (mismatch - 4.0) * 1.5)

            # Form penalty: reduces gpp_score for declining players
            # Prevents ILP from picking fading players despite high season avg
            form_penalty = min(0.0, (form_ratio - 0.85) * 6.0)

            # Projection adjustment: blend DK projection with recent form
            # Recent 5-game avg is often more predictive than DK's conservative estimate
            # (especially for usage-spike players like Achiuwa with doubled recent output)
            if r5 > 0 and sig["n_games"] >= 10:
                old_proj = float(row["proj_pts_dk"])
                # Weight toward recent form when it significantly deviates from DK proj
                blend_w  = min(0.4, max(0.1, abs(form_ratio - 1.0)))
                new_proj = old_proj * (1 - blend_w) + r5 * blend_w
                out.loc[idx, "proj_pts_dk"] = round(new_proj, 2)
                # Also adjust ceiling proportionally
                proj_ratio = new_proj / max(old_proj, 1.0)
                out.loc[idx, "ceiling"] = round(
                    float(row["ceiling"]) * min(proj_ratio, 1.5), 2
                )

            # Apply gpp_score adjustment
            adj = form_bonus + mismatch_bonus + form_penalty
            out.loc[idx, "gpp_score"] = round(
                float(out.loc[idx, "gpp_score"]) + adj * (1.0 - float(out.loc[idx, "dnp_risk"])),
                3,
            )

        print(f"[projections] ESPN price-mismatch signal applied to "
              f"{sum(1 for n in out['name'].str.lower() if n in mismatch_signals)} / {len(out)} players")

    return out.sort_values("proj_pts_dk", ascending=False).reset_index(drop=True)


# ── Ownership refresh ─────────────────────────────────────────────────────────
def refresh_ownership(players: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute proj_own and gpp_score after downstream signals have run
    (B2B adjustments, usage absorption, GT field model blend).

    Should be called after:
      - apply_b2b_adjustments()   — so is_b2b is set
      - estimate_usage_absorption() — so proj_pts_dk boosts are reflected
      - GT field_own_model blend  — done in app.py before lineup generation

    Changes applied:
      1. B2B fade: players on back-to-back teams draw ~15% less public ownership.
         Public fades B2B players even when the model still projects them well.
      2. Usage-gainer bump: players whose proj_pts_dk was boosted (injury absorbers)
         draw some additional public ownership as their role becomes obvious.
         Capped at a modest +4pts so leveraged plays don't get fully priced in.
    """
    df = players.copy()

    # 1. B2B fade
    if "is_b2b" in df.columns:
        b2b_mask = df["is_b2b"] == True
        df.loc[b2b_mask, "proj_own"] = (
            df.loc[b2b_mask, "proj_own"] * 0.85
        ).clip(1, 40).round(1)

    # 2. Usage-gainer bump — players whose B2B boost or injury absorption
    #    raised their projection above their salary-implied baseline get a
    #    small ownership bump (public follows obvious role expansions).
    #    Salary-implied DK pts = salary / 1000 * 5.0 (typical value threshold).
    if "salary" in df.columns:
        sal_implied = df["salary"] / 1000 * 5.0
        gainer_mask = df["proj_pts_dk"] > sal_implied * 1.10  # >10% above salary-implied
        df.loc[gainer_mask, "proj_own"] = (
            df.loc[gainer_mask, "proj_own"] + 4.0
        ).clip(1, 40).round(1)

    # Recompute gpp_score everywhere with updated proj_own.
    # USG bonus: 3/13 backtest showed fc_usg corr +0.501 with actual FPTS.
    # Added as additive term (NOT blended into proj_pts_dk) to avoid double-
    # counting FC Proj which already incorporates usage. Effect: differentiates
    # players with similar projections — 35% USG star > 20% USG role player
    # at the same projected score. Scale: each % above 20 → +0.25 bonus, capped
    # at +6. League avg 20% USG → 0 bonus; 35% USG → +3.75; 44% USG → +6.
    if "fc_usg" in df.columns:
        _usg_bonus = ((df["fc_usg"].fillna(20.0) - 20.0) * 0.25).clip(0, 6)
    else:
        _usg_bonus = 0.0

    df["gpp_score"] = (
        df["ceiling"] * 0.60
        + df["proj_pts_dk"] * 0.25
        + (1 - df["proj_own"] / 100) * 10
        + _usg_bonus
    ).round(3)

    return df


# ── ILP Lineup Optimizer ──────────────────────────────────────────────────────
_SLOT_ORDER = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

def build_lineup(
    players: pd.DataFrame,
    objective_col: str = "gpp_score",
    prev_lineups: list = None,
    min_unique: int = 2,
    locked_ids: list = None,
    excluded_ids: list = None,
    max_per_team: int = MAX_PER_TEAM,
    ownership_penalty: float = 0.04,
    stack_game: str = None,
    stack_bonus: float = 0.12,
    bringback_bonus: float = 0.06,
    force_stack: bool = True,        # add hard ILP constraint (>=3 from stack_game); set False in fallbacks
    barbell_params: dict = None,
    correlation_pairs: dict = None,
    correlation_bonus: float = 2.5,
    min_proj_total: float = None,
    base_proj_vals: "np.ndarray | None" = None,
    max_premium_players: int = 2,    # max players with salary >= $8K (postmortem: pros avg 1.57)
) -> dict | None:
    """
    ILP lineup optimizer.

    stack_game:    If set, apply a score multiplier bonus to players from this matchup.
                   Cycles across lineups in generate_gpp_lineups() to ensure diverse stacks.
    stack_bonus:   Multiplier bonus applied to same-stack-game players (12% by default).
    bringback_bonus: Bonus for opponent players in the stack game (6% — game stack).
    """
    prob = pulp.LpProblem("DFS_Lineup", pulp.LpMaximize)
    n    = len(players)
    idx  = list(range(n))

    # Player selection variables
    x = pulp.LpVariable.dicts("p", idx, cat="Binary")

    # Slot assignment variables: y[(i, slot)] = 1 means player i fills slot
    # Only created for (player, slot) pairs where the slot is in the player's eligible_slots
    y = {}
    for i in idx:
        elig = players["eligible_slots"].iloc[i]
        for slot in _SLOT_ORDER:
            if slot in elig:
                y[(i, slot)] = pulp.LpVariable(f"y_{i}_{slot}", cat="Binary")

    # Objective: maximize gpp_score - ownership penalty
    # With optional game-stack bonus (linear: multiplies a constant score value)
    obj_vals = players[objective_col].fillna(0).values.copy().astype(float)
    own_vals = players["proj_own"].fillna(20).values / 100.0

    if stack_game:
        # Identify which teams are in this game (both sides get a bonus —
        # primary stack team gets full bonus, opponent gets bring-back bonus)
        game_parts = stack_game.split("@") if "@" in stack_game else []
        for i in idx:
            matchup = players["matchup"].iloc[i]
            team_i  = players["team"].iloc[i]
            if matchup == stack_game:
                # Determine which team is primary vs bring-back
                # We rotate primary team by lineup number to avoid always picking same team
                if game_parts and team_i == game_parts[1]:  # home team = primary
                    obj_vals[i] *= (1.0 + stack_bonus)
                else:  # away team = bring-back
                    obj_vals[i] *= (1.0 + bringback_bonus)

    # ── Build objective ────────────────────────────────────────────────────────
    # Base: Stackelberg score minus ownership penalty
    base_obj = pulp.lpSum(
        x[i] * (obj_vals[i] - ownership_penalty * own_vals[i] * obj_vals[i])
        for i in idx
    )

    # Correlation stack bonus:
    # For each high-correlation pair (corr > 0.20), add an auxiliary binary
    # variable z[(i1,i2)] that equals 1 when BOTH players are selected.
    # In a maximisation problem, z <= x[i1] AND z <= x[i2] is sufficient —
    # the solver will set z=1 whenever both are selected and the bonus is positive.
    # Only meaningful positive pairs are added to keep the ILP tractable.
    corr_obj = pulp.lpSum([])   # empty sum; populated below
    if correlation_pairs:
        pid_to_idx: dict[str, int] = {
            str(players["player_id"].iloc[i]): i for i in idx
        }
        z_vars: dict[tuple, pulp.LpVariable] = {}
        for (pid1, pid2), corr_val in correlation_pairs.items():
            if -0.15 < corr_val < 0.20:
                continue    # near-zero: skip (noise)
            i1 = pid_to_idx.get(str(pid1))
            i2 = pid_to_idx.get(str(pid2))
            if i1 is None or i2 is None or i1 == i2:
                continue
            key = (min(i1, i2), max(i1, i2))
            if key in z_vars:
                continue    # already added this pair
            z = pulp.LpVariable(f"z_{key[0]}_{key[1]}", cat="Binary")
            z_vars[key] = z
            prob += z <= x[key[0]]
            prob += z <= x[key[1]]
            if corr_val >= 0.20:
                # Positive correlation: reward co-selection (stack bonus)
                corr_obj += z * corr_val * correlation_bonus
            else:
                # Negative correlation: penalise co-selection (usage competitors)
                # e.g. two PGs on same team competing for ball-handling possessions
                corr_obj += z * corr_val * correlation_bonus  # corr_val is negative → penalty

    prob += base_obj + corr_obj

    # Salary cap
    sals = players["salary"].values
    prob += pulp.lpSum(x[i] * sals[i] for i in idx) <= SALARY_CAP
    prob += pulp.lpSum(x[i] * sals[i] for i in idx) >= MIN_SALARY_USED

    # Roster size
    prob += pulp.lpSum(x[i] for i in idx) == ROSTER_SIZE

    # Each slot filled by exactly 1 player
    for slot in _SLOT_ORDER:
        slot_vars = [y[(i, slot)] for i in idx if (i, slot) in y]
        prob += pulp.lpSum(slot_vars) == 1

    # Each selected player fills exactly 1 slot; unselected players fill none
    for i in idx:
        player_slot_vars = [y[(i, slot)] for slot in _SLOT_ORDER if (i, slot) in y]
        if player_slot_vars:
            prob += pulp.lpSum(player_slot_vars) == x[i]
        else:
            prob += x[i] == 0  # player eligible for no slot — cannot be selected

    # Team cap
    for team in players["team"].unique():
        team_idx = [i for i in idx if players["team"].iloc[i] == team]
        prob += pulp.lpSum(x[i] for i in team_idx) <= max_per_team

    # Minimum 3-man game stack — backtest showed top 1% averaged 3.22-man stacks.
    # At least one game must contribute 3+ players (both teams combined).
    # We enforce this by requiring that at least one matchup has ≥3 players selected.
    # Use auxiliary binary z_game[matchup] = 1 if that game supplies ≥3 players.
    if "matchup" in players.columns:
        matchups = players["matchup"].dropna().unique()
        if len(matchups) > 1:  # only meaningful on multi-game slates
            z_game = {m: pulp.LpVariable(f"zg_{j}", cat="Binary")
                      for j, m in enumerate(matchups)}
            for m, zg in z_game.items():
                game_idx = [i for i in idx if players["matchup"].iloc[i] == m]
                if game_idx:
                    # zg=1 only if ≥3 players from this game are selected
                    # zg ≤ (Σx[i] for game) / 3  →  3*zg ≤ Σx[i]
                    prob += 3 * zg <= pulp.lpSum(x[i] for i in game_idx)
                    # zg ≥ (Σx[i] - 2) / 8  (big-M: if ≥3 selected, zg CAN be 1)
                    prob += zg >= (pulp.lpSum(x[i] for i in game_idx) - 2) / 8
            # At least one game must have zg=1
            prob += pulp.lpSum(z_game.values()) >= 1

    # Hard stack-game constraint: when a specific game is designated as the stack,
    # FORCE ≥3 players from that game. The soft 20% score bonus alone is not enough —
    # the ILP will freely pick a different game if it scores higher overall.
    # force_stack=False in fallbacks so the soft bonus still guides the solver
    # without making the problem infeasible when exposure caps thin out the game.
    if force_stack and stack_game and "matchup" in players.columns:
        _hard_sg_idx = [i for i in idx if players["matchup"].iloc[i] == stack_game]
        # Only add the hard constraint if >= 3 NON-excluded players exist in the game.
        # Counting excluded players causes infeasibility when caps thin the game out.
        _excl_set = set(str(p) for p in (excluded_ids or []))
        _non_excl_sg = [
            i for i in _hard_sg_idx
            if str(players["player_id"].iloc[i]) not in _excl_set
        ]
        if len(_non_excl_sg) >= 3:
            prob += pulp.lpSum(x[i] for i in _hard_sg_idx) >= 3

    # Salary barbell structure: driven by SlateConstructionAgent parameters.
    # Thresholds and minimums are adaptive to tonight's slate salary distribution.
    # barbell_params=None means no barbell constraint (unconstrained optimization).
    if barbell_params and barbell_params.get("barbell_enabled", False):
        _stud_floor  = int(barbell_params.get("stud_threshold",  9000))
        _stud_min    = int(barbell_params.get("stud_min",        2))
        _cheap_ceil  = int(barbell_params.get("cheap_threshold", 4500))
        _cheap_min   = int(barbell_params.get("cheap_min",       3))
        stud_idx  = [i for i in idx if sals[i] >= _stud_floor]
        cheap_idx = [i for i in idx if sals[i] <= _cheap_ceil]
        if len(stud_idx) >= _stud_min and _stud_min > 0:
            prob += pulp.lpSum(x[i] for i in stud_idx) >= _stud_min
        if len(cheap_idx) >= _cheap_min and _cheap_min > 0:
            prob += pulp.lpSum(x[i] for i in cheap_idx) >= _cheap_min

    # Salary-tier balance: cap number of premium-salary ($8K+) players.
    # Postmortem 3/6-3/12: pros averaged 1.57 players >=8K; we averaged 2.20 — over by 0.63.
    # Previous cap was $9K+ which left $8K-$9K players unchecked. Moved to $8K+ to correct.
    if max_premium_players is not None and max_premium_players > 0:
        premium_idx = [i for i in idx if sals[i] >= 8000]
        if len(premium_idx) > max_premium_players:
            prob += pulp.lpSum(x[i] for i in premium_idx) <= max_premium_players

    # Mid-value floor: require at least 3 players in the $5,000-$7,000 range.
    # Postmortem 3/6-3/12: pros averaged 2.5 players in $5K-$7K; we averaged 0.29.
    # THIS was our biggest structural gap. The pros' value plays (Jaylin Williams $5.7K,
    # Javon Small $5.4K, Danny Wolf $4.5K, Cam Spencer $5.4K) were all in this tier.
    # Previous floor was $5.5K-$7.5K >=2 — too loose. Tightened to the actual value zone.
    mid_value_idx = [i for i in idx if 5000 <= sals[i] <= 7000]
    if len(mid_value_idx) >= 3:
        prob += pulp.lpSum(x[i] for i in mid_value_idx) >= 3

    # Sub-$5K cap: at most 3 punt plays per lineup.
    # Analysis of 800 top lineups across 8 dates: pros averaged 2.66 players <$5K, NOT 1.5.
    # 80%+ of top lineups have at least one player under $4K (e.g. Clint Capela $3.1K, 28 min).
    # Cap at 2 was too restrictive — it forced budget UP into the $7K+ tier when the real
    # value is concentrated at $3K-$4.5K for confirmed high-minute cheap plays.
    sub5k_idx = [i for i in idx if sals[i] < 5000]
    if len(sub5k_idx) > 3:
        prob += pulp.lpSum(x[i] for i in sub5k_idx) <= 3

    # Locks
    if locked_ids:
        for pid in locked_ids:
            for i in idx:
                if players["player_id"].iloc[i] == str(pid):
                    prob += x[i] == 1

    # Exclusions
    if excluded_ids:
        for pid in excluded_ids:
            for i in idx:
                if players["player_id"].iloc[i] == str(pid):
                    prob += x[i] == 0

    # Uniqueness from previous lineups
    if prev_lineups:
        for prev_pids in prev_lineups:
            prev_idx = [i for i in idx if players["player_id"].iloc[i] in prev_pids]
            prob += pulp.lpSum(x[i] for i in prev_idx) <= len(prev_pids) - min_unique

    # Minimum viable lineup projection — uses BASE (non-sampled) projections so
    # the floor is grounded in real player value, not lucky Thompson draws.
    # base_proj_vals must come from the original players DataFrame, not the sample.
    if min_proj_total is not None and base_proj_vals is not None:
        prob += pulp.lpSum(x[i] * float(base_proj_vals[i]) for i in idx) >= min_proj_total

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    selected_idx = [i for i in idx if x[i].value() and x[i].value() > 0.5]
    selected = players.iloc[selected_idx].copy()

    # Extract the ILP-determined slot assignment.
    # Iterate over SLOTS (not players) so every slot is checked exactly once.
    # The old player-first loop could silently drop a slot when CBC's numerical
    # tolerances left two y-variables for the same slot both > 0.5 — the second
    # player overwrote the first, leaving the first player's real slot empty.
    slot_assignment = {}
    for slot in _SLOT_ORDER:
        best_i, best_val = None, 0.0
        for i in selected_idx:
            if (i, slot) not in y:
                continue
            val = y[(i, slot)].value()
            if val is not None and val > best_val:
                best_val = val
                best_i = i
        if best_i is not None and best_val > 0.5:
            slot_assignment[slot] = players["player_id"].iloc[best_i]

    return {
        "player_ids":      selected["player_id"].tolist(),
        "names":           selected["name"].tolist(),
        "positions":       selected["primary_position"].tolist(),
        "teams":           selected["team"].tolist(),
        "salaries":        selected["salary"].tolist(),
        "projections":     selected["proj_pts_dk"].round(2).tolist(),
        "total_salary":    int(selected["salary"].sum()),
        "proj_pts":        round(float(selected["proj_pts_dk"].sum()), 2),
        "ceiling":         round(float(selected["ceiling"].sum()), 2),
        "proj_own":        round(float(selected["proj_own"].mean()), 1),
        "slot_assignment": slot_assignment,
    }


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets. Returns 0.0 if both empty."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def jaccard_diversify(
    lineups: list,
    players: pd.DataFrame,
    max_overlap: float = 0.50,
    locked_ids: list | None = None,
    excluded_ids: list | None = None,
    max_replacements: int = 25,
) -> list:
    """
    Post-generation Jaccard diversity filter.

    For each pair of lineups whose player overlap exceeds `max_overlap`,
    the lower-projected lineup is flagged for replacement. A replacement
    lineup is generated with the shared players added to excluded_ids,
    forcing the ILP to choose a different composition.

    This acts as a hard floor guarantee on portfolio diversity.
    Thompson sampling already creates organic diversity; this catches
    any remaining clusters without adding ILP complexity.

    Parameters
    ----------
    lineups        : list of lineup dicts from generate_gpp_lineups()
    players        : original (non-sampled) player pool DataFrame
    max_overlap    : Jaccard threshold above which a lineup is replaced (0.50 = 4/8 shared)
    max_replacements: safety cap — stops after this many replacements to bound runtime
    """
    if len(lineups) < 2:
        return lineups

    replaced = 0
    changed  = True

    while changed and replaced < max_replacements:
        changed = False
        pid_sets = [set(str(p) for p in lu["player_ids"]) for lu in lineups]

        for i in range(len(lineups)):
            for j in range(i + 1, len(lineups)):
                if jaccard_similarity(pid_sets[i], pid_sets[j]) <= max_overlap:
                    continue

                # Replace the lower-projected lineup (index j)
                shared   = pid_sets[i] & pid_sets[j]
                excl_new = list(excluded_ids or []) + list(shared)

                # Fresh Thompson sample for the replacement attempt
                players_sample = _thompson_sample_players(players)

                replacement = build_lineup(
                    players_sample,
                    objective_col="gpp_score",
                    prev_lineups=[pid_sets[k] for k in range(len(lineups)) if k != j],
                    min_unique=2,
                    locked_ids=locked_ids,
                    excluded_ids=excl_new,
                    ownership_penalty=0.05,
                    stack_game=lineups[j].get("stack_game"),
                )

                if replacement is None:
                    continue   # can't diversify this pair — leave as-is

                replacement["lineup_num"]    = lineups[j]["lineup_num"]
                replacement["stack_game"]    = lineups[j].get("stack_game", "")
                lev = score_lineup_leverage(replacement, players)
                replacement["leverage"]      = lev["leverage"]
                replacement["avg_own"]       = lev["avg_own"]
                replacement["chalk_ct"]      = lev["chalk_ct"]
                replacement["low_own_ct"]    = lev["low_own_ct"]
                replacement["has_game_stack"]= lev["has_game_stack"]

                # Restore baseline projected pts / ceiling for display
                pid_index = players.set_index("player_id")
                pids = [str(p) for p in replacement["player_ids"]]
                replacement["proj_pts"] = round(
                    sum(float(pid_index.loc[p, "proj_pts_dk"]) for p in pids if p in pid_index.index), 1
                )
                replacement["ceiling"] = round(
                    sum(float(pid_index.loc[p, "ceiling"])     for p in pids if p in pid_index.index), 1
                )

                old_j = lineups[j]["lineup_num"]
                lineups[j] = replacement
                pid_sets[j] = set(str(p) for p in replacement["player_ids"])
                replaced += 1
                changed  = True
                print(f"  [Jaccard] Lineup #{old_j} diversified "
                      f"(shared {len(shared)} players with #{lineups[i]['lineup_num']})")
                break   # restart inner loop after any replacement
            if changed:
                break

    if replaced:
        print(f"  [Jaccard] {replaced} lineup(s) replaced for diversity (max_overlap={max_overlap:.0%})")

    return lineups


def _thompson_sample_players(players: pd.DataFrame) -> pd.DataFrame:
    """
    Draw one Thompson sample from each player's projected DK distribution
    and compute a per-sample Stackelberg gpp_score.

    Thompson sample:
      sampled_proj ~ N(proj_pts_dk, sigma_adj²)
      where sigma_adj = proj_std * (1 + 0.50 * tail_index)

    Stackelberg gpp_score:
      score = sampled_proj * tail_mult / (proj_own / 100)
      where tail_mult = 1 + 0.50 * tail_index

    Heavy-tailed, low-ownership players receive a compounding bonus:
    wider sampling σ makes their high draws more likely AND the
    Stackelberg denominator makes those draws worth more per lineup.

    Ownership is floored at 1% to prevent division blow-up for
    newly-added or very cheap players.
    """
    df = players.copy()

    tail_idx = df["tail_index"].values if "tail_index" in df.columns else np.full(len(df), 0.30)
    proj_std  = df["proj_std"].values  if "proj_std"  in df.columns else (df["proj_pts_dk"].values * 0.30)
    proj_pts  = df["proj_pts_dk"].values
    proj_own  = df["proj_own"].values

    # Wider σ for heavier tails; clip to minimum 0.5 to prevent scale<=0 error
    sigma_adj = np.maximum(proj_std * (1.0 + 0.50 * tail_idx), 0.5)

    # Sample — floor at 50% of base projection so leverage lineups stay viable.
    # Clipping at hard 0 lets deep negative draws crater GPP-build quality.
    floor   = proj_pts * 0.50
    sampled = np.random.normal(proj_pts, sigma_adj).clip(floor)

    # Tail multiplier for the Stackelberg denominator weight
    tail_mult = 1.0 + 0.50 * tail_idx

    # Stackelberg score: expected value per unit of ownership competition
    own_fraction = np.clip(proj_own / 100.0, 0.01, 1.0)
    stackelberg  = (sampled * tail_mult / own_fraction).round(3)

    df["proj_pts_dk"] = sampled.round(2)
    df["gpp_score"]   = stackelberg
    return df


def compute_optimal_split(
    n_lineups: int = 20,
    field_size: int = 9908,
    n_games: int = 7,
) -> dict:
    """
    Compute the optimal chalk/leverage lineup split for a GPP contest.

    Larger fields and smaller slates push toward more leverage lineups because:
      - Large field: top finishes require ownership differentiation
      - Small slate: chalk ownership concentrates on fewer players → more punishing

    Returns a dict with:
      n_chalk, n_leverage: lineup counts
      chalk_schedule, leverage_schedule: per-lineup ownership penalty lists
      description: human-readable summary
    """
    # ── Field size → leverage fraction ──────────────────────────────────────
    if   field_size < 500:   lev_frac = 0.30
    elif field_size < 2_000: lev_frac = 0.35
    elif field_size < 5_000: lev_frac = 0.40
    elif field_size < 15_000: lev_frac = 0.50
    elif field_size < 50_000: lev_frac = 0.55
    else:                    lev_frac = 0.60

    # ── Slate size adjustment ────────────────────────────────────────────────
    if   n_games <= 4:  lev_frac += 0.10   # very concentrated ownership
    elif n_games <= 6:  lev_frac += 0.05
    elif n_games <= 9:  pass               # neutral
    elif n_games <= 12: lev_frac -= 0.05
    else:               lev_frac -= 0.08   # large slate, ownership spreads

    lev_frac = max(0.25, min(0.65, lev_frac))

    n_leverage = max(1, round(n_lineups * lev_frac))
    n_chalk    = n_lineups - n_leverage

    # ── Chalk penalty schedule (low, slight variation for internal diversity) ─
    chalk_base = 0.02
    chalk_schedule = [
        round(chalk_base + (i % 3) * 0.005, 3)
        for i in range(n_chalk)
    ]

    # ── Leverage penalty schedule (scales with field + slate) ────────────────
    if   field_size < 500:    lev_base = 0.05
    elif field_size < 2_000:  lev_base = 0.06
    elif field_size < 5_000:  lev_base = 0.07
    elif field_size < 15_000: lev_base = 0.08
    elif field_size < 50_000: lev_base = 0.10
    else:                     lev_base = 0.12

    if n_games <= 4:   lev_base += 0.01   # small slates: softer bump (fewer contrarian options)
    elif n_games <= 6: lev_base += 0.005

    # On small slates the player pool is thin — cap penalty lower so the solver
    # doesn't exhaust viable contrarian options and fall into non-competitive picks.
    if   n_games <= 4: lev_cap = 0.07
    elif n_games <= 6: lev_cap = 0.09
    elif n_games <= 9: lev_cap = 0.11
    else:              lev_cap = 0.14

    lev_base = min(lev_cap, lev_base)

    # Ramp step: smaller on small slates to keep the schedule tighter
    ramp = 0.005 if n_games <= 6 else 0.01
    leverage_schedule = [
        round(min(lev_base + (i % 3) * ramp, lev_cap), 3)
        for i in range(n_leverage)
    ]

    # ── Human-readable description ───────────────────────────────────────────
    field_label = (
        "Small field"   if field_size < 2_000  else
        "Medium field"  if field_size < 10_000 else
        "Large field"   if field_size < 50_000 else
        "Massive field"
    )
    slate_label = (
        f"{n_games}-game slate (small)"  if n_games <= 5 else
        f"{n_games}-game slate (medium)" if n_games <= 9 else
        f"{n_games}-game slate (large)"
    )
    description = (
        f"{n_chalk} chalk (pen {chalk_schedule[0]}–{max(chalk_schedule)}) + "
        f"{n_leverage} leverage (pen {leverage_schedule[0]}–{max(leverage_schedule)}) · "
        f"{field_label} · {slate_label}"
    )

    return {
        "n_chalk":          n_chalk,
        "n_leverage":       n_leverage,
        "chalk_schedule":   chalk_schedule,
        "leverage_schedule": leverage_schedule,
        "chalk_pen_range":  (chalk_schedule[0], max(chalk_schedule)),
        "leverage_pen_range": (leverage_schedule[0], max(leverage_schedule)),
        "description":      description,
    }


def select_portfolio(
    pool: list,
    k: int,
    players: pd.DataFrame,
    max_exposure: float = 0.35,
    diversity_weight: float = 0.26,
    leverage_weight: float = 0.24,
    score_weight: float = 0.40,
    corr_weight: float = 0.10,
) -> list:
    """
    Greedy portfolio selection: from a large pool of lineups pick the best k
    that collectively maximise score, leverage, correlation quality, and diversity.

    Algorithm (greedy marginal value):
      1. Score each lineup individually on proj_pts + leverage + stack correlation.
      2. Greedily add the lineup with the highest marginal portfolio value,
         where marginal value penalises:
           - Jaccard overlap with already-selected lineups (diversity_weight)
           - Player exposure exceeding max_exposure fraction (hard drop)
      3. Repeat until k lineups are selected or pool is exhausted.

    Parameters
    ----------
    pool            : output of generate_gpp_lineups (large pool, e.g. 150)
    k               : target number of lineups to return (e.g. 20)
    players         : player pool DataFrame (for proj_pts normalisation)
    max_exposure    : maximum fraction of k lineups any single player may appear in
    diversity_weight: weight on Jaccard similarity penalty (0–1)
    leverage_weight : weight on lineup leverage score (0–1)
    score_weight    : weight on projected score (0–1)
    corr_weight     : weight on lineup stack correlation quality (0–1)
    """
    if not pool or k <= 0:
        return pool[:k]

    # Pre-compute correlation quality per lineup using CorrelationModel
    # Uses heuristic corr_pairs (already built) to score stack quality.
    # Higher = players in this lineup have more correlated upside.
    _pool_corr_pairs = build_player_correlation(players)
    pid_to_name = dict(zip(players["player_id"].astype(str), players["name"]))

    def _lineup_corr_score(lu: dict) -> float:
        pids = [str(p) for p in lu.get("player_ids", [])]
        scores = []
        for pa, pb in itertools.combinations(pids, 2):
            c = _pool_corr_pairs.get((pa, pb), _pool_corr_pairs.get((pb, pa), 0.0))
            scores.append(c)
        return float(np.mean(scores)) if scores else 0.0

    # Normalise proj_pts, leverage, and correlation to [0, 1] across the pool
    projs = [lu.get("proj_pts", 0) for lu in pool]
    levs  = [lu.get("leverage",  0) for lu in pool]
    corrs = [_lineup_corr_score(lu)  for lu in pool]

    p_min, p_max = min(projs), max(projs)
    l_min, l_max = min(levs),  max(levs)
    c_min, c_max = min(corrs), max(corrs)
    p_range = max(p_max - p_min, 1)
    l_range = max(l_max - l_min, 1)
    c_range = max(c_max - c_min, 0.001)

    # Cache correlation scores to avoid recomputing in the greedy loop
    _corr_cache = {id(lu): c for lu, c in zip(pool, corrs)}

    def _base_score(lu: dict) -> float:
        p_norm = (lu.get("proj_pts", 0) - p_min) / p_range
        l_norm = (lu.get("leverage",  0) - l_min) / l_range
        c_norm = (_corr_cache.get(id(lu), 0.0) - c_min) / c_range
        return score_weight * p_norm + leverage_weight * l_norm + corr_weight * c_norm

    def _jaccard(set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    selected: list = []
    selected_pids: list[set] = []
    exposure: dict[str, int] = {}
    hard_cap = max(1, int(k * max_exposure))

    remaining = list(pool)

    for _ in range(k):
        if not remaining:
            break

        best_score = -1e9
        best_idx   = 0

        for j, lu in enumerate(remaining):
            pids = set(str(p) for p in lu.get("player_ids", []))

            # Hard drop: player already at exposure cap
            if any(exposure.get(str(pid), 0) >= hard_cap for pid in pids):
                continue

            base = _base_score(lu)

            # Diversity penalty: avg Jaccard overlap with already-selected
            if selected_pids:
                avg_overlap = sum(_jaccard(pids, s) for s in selected_pids) / len(selected_pids)
                div_penalty = diversity_weight * avg_overlap
            else:
                div_penalty = 0.0

            marginal = base - div_penalty

            if marginal > best_score:
                best_score = marginal
                best_idx   = j

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        chosen_pids = set(str(p) for p in chosen.get("player_ids", []))
        selected_pids.append(chosen_pids)
        for pid in chosen_pids:
            exposure[pid] = exposure.get(pid, 0) + 1

    # Sort by projected score descending so lineup #1 is always the highest-value
    selected.sort(key=lambda lu: lu.get("proj_pts", 0), reverse=True)

    # Re-number lineup_num sequentially
    for i, lu in enumerate(selected, 1):
        lu["lineup_num"] = i

    return selected


def generate_gpp_lineups(
    players: pd.DataFrame,
    n: int = 20,
    locked_ids: list = None,
    excluded_ids: list = None,
    penalty_schedule: list = None,
    slate_profile: dict = None,
    ownership_penalty: float = 0.04,
    max_exposure: int = None,   # ignored — exposure is now template-driven
) -> list[dict]:
    """
    Pro-mirroring GPP portfolio builder.

    Architecture based on 7-day analysis of top 1% winning lineups (3/6-3/13):
    - Every winning day has 2-3 "ultra-chalk" plays at 80-100% exposure
    - Every winning lineup has a 4-player game stack (not 3)
    - 80% of lineups use the top game; 15% use a secondary game; 5% are wild

    Replaces the previous Thompson-sampling + exposure-cap approach which
    produced only 55% stack compliance vs the required 80%.
    """
    import itertools
    import pulp
    from collections import Counter as _Counter

    if players is None or len(players) < 8:
        return []

    _sp = slate_profile or {}
    n_lineups = n

    # ── Step 1: Determine stack game order ───────────────────────────────────────
    # Backtest 3/6-3/13 showed that game_total (O/U) alone is wrong: on 3/13
    # CLE@DAL had O/U=236.5 vs NOP@HOU O/U=230.5 but NOP@HOU was correct.
    # The missing signal is DVP (Def v Pos): NOP@HOU had avg DVP +21 to +28
    # for the key value plays, vs much lower DVP for CLE@DAL.
    #
    # Composite stack score = game_total * dvp_mult * injury_value_mult
    #   dvp_mult: average DVP of top-4 players in game, normalized 1.0-1.3
    #   injury_value_mult: bonus when an injury replacement play exists in game
    #     (cheap player with fc_proj >> salary-implied due to starter being OUT)
    #
    # Corr model reorders this when available — it's the most accurate signal.
    # DVP composite is the robust fallback that also handles the corr model case.

    _corr_pairs: dict = {}
    stack_games: list[str] = []

    if "matchup" in players.columns:
        # ── DVP-weighted composite game score ────────────────────────────────
        _matchup_composite: dict[str, float] = {}
        for _mg in players["matchup"].dropna().unique():
            _gp = players[players["matchup"] == _mg]
            _top4 = _gp.nlargest(4, "proj_pts_dk")

            # Component 1: game total (Vegas O/U)
            _gt = float(_gp["game_total"].iloc[0]) if "game_total" in _gp.columns and not _gp["game_total"].isna().all() else 225.0

            # Component 2: DVP multiplier — avg of top-4 players' DVP scores
            # Priority: fc_dvp (raw FC "Def v Pos" values, e.g. +28, -5)
            #           dvp_mult (computed game-log DVP, normalized around 1.0)
            if "fc_dvp" in players.columns and _top4["fc_dvp"].notna().any():
                # fc_dvp is raw: +30 → 1.15x bonus, -15 → 0.925x penalty
                _avg_dvp_raw = float(_top4["fc_dvp"].fillna(0).mean())
                _dvp_mult = 1.0 + min(0.20, max(-0.10, _avg_dvp_raw / 200.0))
            elif "dvp_mult" in players.columns:
                # dvp_mult is normalized around 1.0 (e.g. 1.06, 0.94)
                # Convert deviation from 1.0 to a composite bonus
                _avg_dvm = float(_top4["dvp_mult"].fillna(1.0).mean())
                _dvp_mult = 1.0 + min(0.10, max(-0.05, (_avg_dvm - 1.0) * 0.5))
            else:
                _dvp_mult = 1.0

            # Component 3: injury-value bonus — game gets +5-8% when a cheap
            # player (<$4.5K) has fc_proj significantly above their avg_pts
            # (signal that they're filling in for an OUT starter)
            _injury_bonus = 1.0
            if "fc_proj" in players.columns and "avg_pts" in players.columns:
                _cheap_gp = _gp[_gp["salary"] <= 4500]
                for _, _cp in _cheap_gp.iterrows():
                    _fc = float(_cp.get("fc_proj", 0) or 0)
                    _avg = float(_cp.get("avg_pts", 0) or 0)
                    if _avg > 5 and _fc > _avg * 1.40:  # 40%+ above average → injury play
                        _injury_bonus = max(_injury_bonus, 1.08)
                    elif _avg > 3 and _fc > _avg * 1.25:
                        _injury_bonus = max(_injury_bonus, 1.04)

            # Component 4: value density bonus — when a team has 3+ players
            # all with value_score (fc_proj / salary_k) > 5.0, their game is
            # a must-stack regardless of O/U. Each extra high-value player above
            # the 3rd adds +8%. This captures injury-replacement concentrated slates
            # (e.g. MEM 3/12 with 6 players at 6+ value, MIA 3/10 with 7 players).
            _val_density_bonus = 1.0
            if "fc_proj" in players.columns and "salary" in players.columns:
                _val_thresh = 5.0
                _min_hv = 3
                _max_hv_team = 0
                for _, _tgp in _gp.groupby("team"):
                    _val_scores = (_tgp["fc_proj"] / (_tgp["salary"] / 1000)).fillna(0)
                    _n_hv = int((_val_scores > _val_thresh).sum())
                    _max_hv_team = max(_max_hv_team, _n_hv)
                if _max_hv_team >= _min_hv:
                    _val_density_bonus = 1.0 + (_max_hv_team - _min_hv + 1) * 0.08

            _matchup_composite[_mg] = _gt * _dvp_mult * _injury_bonus * _val_density_bonus

        stack_games = sorted(_matchup_composite, key=lambda g: -_matchup_composite[g])

        print(f"\n  Stack game ranking (DVP-weighted composite):")
        for _mg in stack_games[:6]:
            _s = _matchup_composite[_mg]
            _gp2 = players[players["matchup"] == _mg]
            _gt_v = float(_gp2["game_total"].iloc[0]) if "game_total" in players.columns and not _gp2["game_total"].isna().all() else 0
            _top4_2 = _gp2.nlargest(4, "proj_pts_dk")
            if "fc_dvp" in players.columns and _top4_2["fc_dvp"].notna().any():
                _dvp_str = f"avg DVP={_top4_2['fc_dvp'].fillna(0).mean():.1f}"
            elif "dvp_mult" in players.columns:
                _dvp_str = f"dvp_mult={_top4_2['dvp_mult'].fillna(1.0).mean():.3f}"
            else:
                _dvp_str = "DVP=n/a"
            # Show value density count for this game
            _vd_count = 0
            if "fc_proj" in players.columns and "salary" in players.columns:
                for _, _tgp in _gp2.groupby("team"):
                    _vs = (_tgp["fc_proj"] / (_tgp["salary"] / 1000)).fillna(0)
                    _vd_count = max(_vd_count, int((_vs > 5.0).sum()))
            print(f"    {_mg:<22s} composite={_s:.1f}  (O/U={_gt_v:.0f}  {_dvp_str}  val_density={_vd_count})")

    # ── Corr model reordering (primary signal — overrides composite when available) ─
    try:
        from nba_dfs.models.correlation_model import CorrelationModel as _CorrModel
        _cm = _CorrModel()
        _cm_players = players.rename(columns={"proj_pts_dk": "projected_pts_dk"})
        _cm_stacks = _cm.get_teammate_stacks(_cm_players, min_stack=2, max_stack=4)
        if _cm_stacks:
            _seen_matchups: list = []
            for _s in _cm_stacks:
                _team = _s["team"]
                _matchup = next((m for m in stack_games if _team in m), None)
                if _matchup and _matchup not in _seen_matchups:
                    _seen_matchups.append(_matchup)
            for _m in stack_games:
                if _m not in _seen_matchups:
                    _seen_matchups.append(_m)
            stack_games = _seen_matchups
            for _s in _cm_stacks[:20]:
                _pids = [str(p) for p in _s.get("player_ids", [])]
                for _pa, _pb in itertools.combinations(_pids, 2):
                    _new_corr = max(0.20, float(_s.get("avg_corr", 0.25)))
                    if _corr_pairs.get((_pa, _pb), 0) < _new_corr:
                        _corr_pairs[(_pa, _pb)] = round(_new_corr, 3)
                        _corr_pairs[(_pb, _pa)] = round(_new_corr, 3)
            print(f"  Stack order (corr-weighted): {' | '.join(stack_games[:4])}")
            print(f"  Top stack: {_cm_stacks[0]['team']} score={_cm_stacks[0]['stack_score']:.1f}")
        else:
            print(f"  Stack order (DVP composite — corr model returned no stacks): {' | '.join(stack_games[:4])}")
    except Exception as _ce:
        print(f"  [WARN] Corr model failed ({_ce}) — using DVP composite ordering")
        print(f"  Stack order (DVP composite fallback): {' | '.join(stack_games[:4])}")

    if not stack_games:
        stack_games = players["matchup"].dropna().unique().tolist() if "matchup" in players.columns else []

    top_game = stack_games[0] if stack_games else None
    second_game = stack_games[1] if len(stack_games) > 1 else top_game

    # ── Slate regime detection ────────────────────────────────────────────────────
    # Count injury-replacement signals across the slate.
    # When >20% of active players show mins_expansion > 1.25, the slate is
    # dominated by injury cascades — cheap replacements are the core GPP play.
    _slate_regime = "normal"
    _extreme_expansion_ids: list[str] = []
    if "fc_mins_expansion" in players.columns:
        _n_active = len(players)
        _n_expanded = int((players["fc_mins_expansion"].fillna(1.0) > 1.25).sum())
        _exp_pct = _n_expanded / max(_n_active, 1)
        if _exp_pct > 0.22:
            _slate_regime = "high_injury"
        elif _exp_pct > 0.14:
            _slate_regime = "moderate_injury"
        else:
            _slate_regime = "normal"

        # Identify extreme expansion plays — injury replacements confirmed by FC
        # These should be included in lineups regardless of game stack assignment
        _exp_mask = (
            (players["fc_mins_expansion"].fillna(1.0) >= 1.50)
            & (players.get("fc_proj", players.get("proj_pts_dk", 0)).fillna(0) >= 15)
            & (players.get("fc_own", pd.Series(20, index=players.index)).fillna(20) <= 22)
            & (players["dnp_risk"].fillna(0.5) <= 0.15)
        )
        _extreme_expansion_ids = [
            str(r["player_id"])
            for _, r in players[_exp_mask].iterrows()
        ]

        print(f"\n  Slate regime: {_slate_regime.upper()} "
              f"({_n_expanded}/{_n_active} = {_exp_pct:.0%} players with mins_expansion>1.25)")
        if _extreme_expansion_ids:
            print(f"  Extreme expansion plays (injury replacements — target 20-40% exposure):")
            for _eid in _extreme_expansion_ids[:8]:
                _er = players[players["player_id"].astype(str) == _eid]
                if not _er.empty:
                    _er = _er.iloc[0]
                    print(f"    {_er.get('name', _eid):<28s} "
                          f"${_er.get('salary', 0):,}  "
                          f"proj={_er.get('proj_pts_dk', 0):.1f}  "
                          f"exp={_er.get('fc_mins_expansion', 1):.2f}x  "
                          f"own={_er.get('fc_own', '?'):.1f}%  "
                          f"gpp={_er.get('fc_gpp_score', 0):.1f}")
    else:
        _exp_pct = 0.0
        print(f"\n  Slate regime: UNKNOWN (fc_mins_expansion not available)")

    # ── Step 2: Find core plays (80-90% exposure targets) ────────────────────────
    # Based on 7-day analysis: each day has 2-3 ultra-chalk plays in 80-100%
    # of winning lineups. These are either top value plays (<$5.5K) or the
    # confirmed safe stud (highest proj player with >20 min).
    def find_core_plays(df: pd.DataFrame) -> list[str]:
        """Return player_ids that should appear in 80%+ of lineups.

        On normal slates: top value play + top stud.
        On injury slates: top fc_gpp_score plays (ceiling × own_leverage × mins_expansion / salary).
        Validated 3/6-3/12: gpp_score correctly ranks Rupert, Wolf, Fontecchio,
        Dru Smith, and Killian Hayes that appeared at 25-80% in top-1% lineups.
        """
        dnp_col = "dnp_risk" if "dnp_risk" in df.columns else None
        eligible = df.copy()
        if dnp_col:
            eligible = eligible[eligible[dnp_col] < 0.25]
        mins_col = "fc_mins" if "fc_mins" in df.columns else None
        if mins_col:
            eligible = eligible[eligible[mins_col].fillna(0) >= 15]
        if eligible.empty:
            return []

        core_ids: set[str] = set()

        # Category A: Best 2 cheap plays by fc_gpp_score (injury-aware GPP metric)
        # Falls back to proj/salary value score if gpp_score not available
        if "fc_gpp_score" in eligible.columns and eligible["fc_gpp_score"].notna().any():
            gpp_sort_col = "fc_gpp_score"
        else:
            eligible["_vs"] = eligible["proj_pts_dk"] / (eligible["salary"] / 1000)
            gpp_sort_col = "_vs"
        cheap_plays = eligible[eligible["salary"] <= 5500].nlargest(2, gpp_sort_col)
        for _, row in cheap_plays.iterrows():
            core_ids.add(str(row["player_id"]))

        # Category B: Top 1 overall stud (highest proj regardless of salary)
        stud = eligible.nlargest(1, "proj_pts_dk")
        for _, row in stud.iterrows():
            core_ids.add(str(row["player_id"]))

        return list(core_ids)

    core_play_ids = find_core_plays(players)
    print(f"\n  Core plays (80%+ exposure):")
    for pid in core_play_ids:
        row = players[players["player_id"].astype(str) == pid]
        if not row.empty:
            r = row.iloc[0]
            print(f"    {r.get('name', pid)}: ${r.get('salary',0):,} proj={r.get('proj_pts_dk',0):.1f}")

    # ── Step 3: Build stack pool for top game ─────────────────────────────────────
    # Take top 6 players from top game — we'll cycle through 4-player combos
    def get_stack_pool(df: pd.DataFrame, game: str, pool_size: int = 8,
                       slate_regime: str = "normal") -> list[str]:
        """
        Hybrid pool: top 4 by proj_pts_dk + top 4 by fc_gpp_score.
        Pool of 8 gives C(8,4)=70 combinations — the optimizer finds the
        salary-feasible combos from the full set.

        fc_gpp_score = ceiling × (1-proj_own%) × mins_expansion / salary_k
        Captures what top-1% pros optimize for on injury-replacement slates.
        Validated on 3/9, 3/10, 3/12 — surfaces cheap injury plays that
        appeared at 25-80% of top-1% lineups but were buried by proj ranking.

        On injury/high_injury slates: gpp_score pool gets 5 slots vs 3 for proj.
        On normal slates: 4+4 split.
        """
        if not game or "matchup" not in df.columns:
            return []
        game_players = df[df["matchup"] == game].copy()
        if "dnp_risk" in game_players.columns:
            game_players = game_players[game_players["dnp_risk"] < 0.50]
        if game_players.empty:
            return []

        # Slot allocation changes by slate regime
        if slate_regime in ("high_injury", "moderate_injury"):
            n_proj, n_gpp = 3, 5   # lean gpp on injury slates
        else:
            n_proj, n_gpp = 4, 4   # balanced on normal slates

        top_proj = game_players.nlargest(n_proj, "proj_pts_dk")

        # GPP sort column: fc_gpp_score preferred, fallback to value
        if "fc_gpp_score" in game_players.columns and game_players["fc_gpp_score"].notna().any():
            gpp_col = "fc_gpp_score"
        elif "salary" in game_players.columns and game_players["salary"].gt(0).any():
            game_players["_val"] = game_players["proj_pts_dk"] / (game_players["salary"] / 1000)
            gpp_col = "_val"
        else:
            gpp_col = "proj_pts_dk"
        # Filter gpp candidates: require fc_proj >= 18 to exclude ultra-cheap lottery
        # tickets ($3K players, proj=12) that inflate gpp_score via mins_expansion
        # but are unreliable. Fontecchio ($3.3K proj=24.6) and Dru Smith ($3.4K
        # proj=23.6) pass; Keshad Johnson ($3K proj=12.3) does not.
        _proj_col = "fc_proj" if "fc_proj" in game_players.columns else "proj_pts_dk"
        _gpp_cands = game_players[game_players[_proj_col].fillna(0) >= 18]
        if len(_gpp_cands) < n_gpp:
            _gpp_cands = game_players  # fallback if too few pass threshold
        top_gpp = _gpp_cands.nlargest(pool_size, gpp_col)  # oversample, stop at pool_size

        seen_ids: set[str] = set()
        pool: list[str] = []
        for _, r in top_proj.iterrows():
            pid = str(r["player_id"])
            if pid not in seen_ids:
                pool.append(pid)
                seen_ids.add(pid)
        for _, r in top_gpp.iterrows():
            if len(pool) >= pool_size:
                break
            pid = str(r["player_id"])
            if pid not in seen_ids:
                pool.append(pid)
                seen_ids.add(pid)
        return pool

    top_stack_pool    = get_stack_pool(players, top_game,    pool_size=8, slate_regime=_slate_regime)
    second_stack_pool = get_stack_pool(players, second_game, pool_size=6, slate_regime=_slate_regime)

    print(f"\n  Top game stack pool [{top_game}] (8 players → C(8,4)=70 combos):")
    for pid in top_stack_pool:
        row = players[players["player_id"].astype(str) == pid]
        if not row.empty:
            r = row.iloc[0]
            gpp = r.get("fc_gpp_score", float("nan"))
            exp = r.get("fc_mins_expansion", float("nan"))
            own = r.get("fc_own", float("nan"))
            gpp_str = f"  gpp={gpp:.1f}  exp={exp:.2f}x  own={own:.0f}%" if not pd.isna(gpp) else ""
            print(f"    {r.get('name', pid):<28s} ${r.get('salary',0):,}  proj={r.get('proj_pts_dk',0):.1f}{gpp_str}")

    # Generate all 4-man combinations from the top stack pool
    # Prefer combos that don't overlap with core plays (avoid salary concentration)
    # Sort combos by combined proj_pts_dk descending
    stack_combos_4 = list(itertools.combinations(top_stack_pool, 4))

    def combo_proj(combo):
        total = 0
        for pid in combo:
            row = players[players["player_id"].astype(str) == pid]
            if not row.empty:
                total += float(row.iloc[0]["proj_pts_dk"])
        return total

    stack_combos_4 = sorted(stack_combos_4, key=combo_proj, reverse=True)

    stack_combos_3 = list(itertools.combinations(second_stack_pool, 3))
    stack_combos_3 = sorted(stack_combos_3, key=combo_proj, reverse=True)

    # ── Step 4: Assign lineup templates ──────────────────────────────────────────
    # Template A (80% of lineups): core plays + 4-man top game stack
    # Template B (15%): 1 core play + 3-man second game stack
    # Template C (5%): no locks — ILP chooses freely with soft stack bonus

    n_template_a = round(n * 0.80)
    n_template_b = round(n * 0.15)
    n_template_c = n - n_template_a - n_template_b

    templates = []
    for i in range(n_template_a):
        combo = stack_combos_4[i % len(stack_combos_4)] if stack_combos_4 else ()
        locks = list(set(core_play_ids) | set(combo))
        templates.append({"type": "A", "locks": locks, "game": top_game, "force": True})
    for i in range(n_template_b):
        combo = stack_combos_3[i % max(1, len(stack_combos_3))] if stack_combos_3 else ()
        locks = list(set(core_play_ids[:1]) | set(combo))
        templates.append({"type": "B", "locks": locks, "game": second_game, "force": True})
    for i in range(n_template_c):
        templates.append({"type": "C", "locks": list(locked_ids or []), "game": top_game, "force": False})

    print(f"\n  Portfolio: {n_template_a}x Template-A (top game 4-stack) | "
          f"{n_template_b}x Template-B (secondary game) | "
          f"{n_template_c}x Template-C (wild)")

    # ── Step 5: Build each lineup ─────────────────────────────────────────────────
    lineups: list[dict] = []
    prev_pids: list[set] = []
    _base_proj = players["proj_pts_dk"].fillna(0).values.copy()

    # Barbell params from slate profile
    _barbell = _sp.get("barbell_params") if _sp.get("barbell_enabled", False) else None
    _min_proj_floor = float(_sp.get("min_proj_floor", 200))

    for lu_num, tmpl in enumerate(templates):
        t_locks = list(tmpl["locks"]) + list(locked_ids or [])
        t_excl  = list(excluded_ids or [])
        t_game  = tmpl["game"]
        t_force = tmpl["force"]

        # Remove excluded from locks
        t_locks = [p for p in t_locks if str(p) not in [str(e) for e in t_excl]]

        result = build_lineup(
            players,
            objective_col="gpp_score",
            prev_lineups=prev_pids,
            min_unique=2 if lu_num > 0 else 0,
            locked_ids=t_locks,
            excluded_ids=t_excl,
            ownership_penalty=0.03,
            stack_game=t_game,
            stack_bonus=0.20,
            bringback_bonus=0.10,
            force_stack=t_force,
            max_premium_players=3,
            correlation_pairs=_corr_pairs,
            min_proj_total=_min_proj_floor,
            base_proj_vals=_base_proj,
        )

        # Fallback 1: loosen diversity
        if result is None:
            print(f"  [fallback-1] LU{lu_num+1}: relax min_unique=1")
            result = build_lineup(
                players,
                objective_col="gpp_score",
                prev_lineups=prev_pids[-3:] if prev_pids else None,
                min_unique=1,
                locked_ids=t_locks,
                excluded_ids=t_excl,
                ownership_penalty=0.02,
                stack_game=t_game,
                stack_bonus=0.20,
                bringback_bonus=0.10,
                force_stack=t_force,
                max_premium_players=3,
                correlation_pairs=_corr_pairs,
                min_proj_total=_min_proj_floor,
                base_proj_vals=_base_proj,
            )

        # Fallback 2: drop locks, keep stack soft
        if result is None:
            print(f"  [fallback-2] LU{lu_num+1}: drop locks, soft stack only")
            result = build_lineup(
                players,
                objective_col="gpp_score",
                prev_lineups=None,
                min_unique=0,
                locked_ids=list(locked_ids or []),
                excluded_ids=t_excl,
                ownership_penalty=0.0,
                stack_game=t_game,
                stack_bonus=0.20,
                bringback_bonus=0.10,
                force_stack=False,
                max_premium_players=None,
                correlation_pairs=_corr_pairs,
                min_proj_total=None,
                base_proj_vals=_base_proj,
            )

        if result is None:
            print(f"  Lineup {lu_num+1}: INFEASIBLE -- skipping")
            continue

        # Metadata
        result["lineup_num"]  = lu_num + 1
        result["stack_game"]  = t_game
        result["template"]    = tmpl["type"]

        lev = score_lineup_leverage(result, players)
        result["leverage"]       = lev["leverage"]
        result["avg_own"]        = lev["avg_own"]
        result["chalk_ct"]       = lev["chalk_ct"]
        result["low_own_ct"]     = lev["low_own_ct"]
        result["has_game_stack"] = lev["has_game_stack"]

        pid_index = players.set_index("player_id")
        pids = [str(p) for p in result["player_ids"]]
        result["proj_pts"] = round(
            sum(float(pid_index.loc[p, "proj_pts_dk"]) for p in pids if p in pid_index.index), 1
        )
        result["ceiling"] = round(
            sum(float(pid_index.loc[p, "ceiling"]) for p in pids if p in pid_index.index), 1
        ) if "ceiling" in players.columns else result["proj_pts"] * 1.35

        prev_pids.append(set(result["player_ids"]))
        lineups.append(result)

        # Print lineup
        names = [pid_index.loc[p, "name"] if p in pid_index.index else p for p in pids]
        stack_ct = sum(1 for p in pids
                       if p in pid_index.index and pid_index.loc[p, "matchup"] == t_game) if "matchup" in players.columns else "?"
        print(f"  LU{lu_num+1} [{tmpl['type']}]: proj={result['proj_pts']} "
              f"stack={stack_ct}@{t_game.split()[0] if t_game else '?'} "
              f"| {', '.join(names)}")

    # Summary
    if lineups:
        print(f"\n  Generated {len(lineups)}/{n} lineups")
        # Stack compliance
        if top_game and "matchup" in players.columns:
            pid_idx = players.set_index("player_id")
            compliant = sum(
                1 for lu in lineups
                if sum(1 for p in lu["player_ids"]
                       if p in pid_idx.index and pid_idx.loc[p, "matchup"] == top_game) >= 4
            )
            print(f"  4-man stack compliance: {compliant}/{len(lineups)} ({100*compliant//max(1,len(lineups))}%)")
        # Exposure
        exp = _Counter(p for lu in lineups for p in lu["player_ids"])
        print(f"  Top exposure:")
        for pid, cnt in exp.most_common(8):
            row = players[players["player_id"].astype(str) == str(pid)]
            name = row.iloc[0]["name"] if not row.empty else pid
            print(f"    {name}: {cnt}/{len(lineups)} ({100*cnt//len(lineups)}%)")

    return lineups


# ── DK Upload CSV export ──────────────────────────────────────────────────────
def export_dk_csv(lineups: list, players: pd.DataFrame, out_path: Path):
    """Export lineups in DraftKings upload format using ILP slot assignments."""
    name_id_map = dict(zip(players["player_id"].astype(str), players["name_id"]))
    slot_order  = _SLOT_ORDER

    rows = []
    for lu in lineups:
        # Use the slot assignment computed by the ILP (guaranteed valid)
        slot_assignment = lu.get("slot_assignment", {})
        row_out = {
            slot: name_id_map.get(str(slot_assignment.get(slot, "")), "")
            for slot in slot_order
        }
        rows.append(row_out)

    df_out = pd.DataFrame(rows, columns=slot_order)
    df_out.to_csv(out_path, index=False)
    print(f"\nDK upload CSV saved: {out_path}")
    return df_out


# ── Tournament Postmortem Analyzer ───────────────────────────────────────────
import re as _re_pm  # noqa: E402  (module-level import for postmortem helper)


def run_postmortem(
    contest_csv: Path,
    entries_csv: Path = None,
    our_username: str = None,
    players_df: pd.DataFrame = None,
) -> dict:
    """
    Parse a DraftKings contest results CSV and produce a full postmortem diagnostic.

    Identifies:
      - Our lineup ranks and scores
      - Cash line
      - Winning lineup composition
      - Zero-scoring players we rostered (DNP/inactive)
      - Our over-exposed players vs actual scores
      - Players we missed who delivered big
      - Lineup diversity failures (shared player count between our lineups)

    Returns a diagnostic dict with all findings.
    """
    if not contest_csv.exists():
        print(f"[postmortem] File not found: {contest_csv}")
        return {}

    player_scores: dict[str, float] = {}
    player_own: dict[str, float] = {}
    all_entries: list[dict] = []
    our_entries: list[dict] = []

    with open(contest_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # DK contest CSV columns: Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS
        for row in reader:
            rank   = row.get("Rank", "").strip()
            name   = row.get("EntryName", "").strip()
            pts    = row.get("Points", "").strip()
            lineup = row.get("Lineup", "").strip()
            player = (row.get("Player") or "").strip()
            pct    = (row.get("%Drafted") or "").strip()
            fpts   = (row.get("FPTS") or "").strip()

            if player and fpts:
                try:
                    player_scores[player] = float(fpts)
                    pct_clean = pct.replace("%", "")
                    if pct_clean:
                        player_own[player] = float(pct_clean)
                except (ValueError, TypeError):
                    pass

            if rank and pts and lineup:
                try:
                    entry = {"rank": int(rank), "name": name, "pts": float(pts), "lineup": lineup}
                    all_entries.append(entry)
                    if our_username and our_username.lower() in name.lower():
                        our_entries.append(entry)
                except (ValueError, TypeError):
                    pass

    if not all_entries:
        print("[postmortem] No entries parsed from contest CSV")
        return {}

    all_entries.sort(key=lambda x: x["rank"])
    our_entries.sort(key=lambda x: x["rank"])
    total_entries = max(e["rank"] for e in all_entries)
    cash_cutoff   = int(total_entries * 0.25)  # top 25% cash

    # Find cash line score
    cash_entries = [e for e in all_entries if e["rank"] <= cash_cutoff and e["lineup"]]
    cash_score   = min(e["pts"] for e in cash_entries) if cash_entries else 0.0

    # Parse player names from lineup strings
    def _parse_names(lineup_str: str) -> list[str]:
        parts = _re_pm.split(r"\b(PG|SG|SF|PF|C|G|F|UTIL)\b", " " + lineup_str)
        names = []
        for j, part in enumerate(parts):
            if part in ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL") and j + 1 < len(parts):
                n = parts[j + 1].strip()
                if n:
                    names.append(n)
        return names

    # Winner analysis
    winner = all_entries[0]
    winner_names = _parse_names(winner["lineup"])

    # Our exposure (if username provided)
    our_exposure: dict[str, int] = {}
    for e in our_entries:
        for pname in _parse_names(e["lineup"]):
            our_exposure[pname] = our_exposure.get(pname, 0) + 1

    n_ours = len(our_entries)

    # DNP / zero scorers we rostered
    our_zeros = {p: cnt for p, cnt in our_exposure.items()
                 if player_scores.get(p, -1) == 0}
    our_near_zero = {p: (cnt, player_scores.get(p, 0))
                     for p, cnt in our_exposure.items()
                     if 0 < player_scores.get(p, 999) < 10}

    # Players we missed (scored 50+, we had 0 or 1 lineup)
    missed_stars = {
        p: sc for p, sc in player_scores.items()
        if sc >= 50 and our_exposure.get(p, 0) == 0
    }
    underexposed = {
        p: (our_exposure.get(p, 0), sc) for p, sc in player_scores.items()
        if sc >= 55 and our_exposure.get(p, 0) < max(1, n_ours // 5)
    }

    # Diversity: avg shared players between our lineups
    our_player_sets = [set(_parse_names(e["lineup"])) for e in our_entries]
    overlap_scores = []
    for i in range(len(our_player_sets)):
        for j in range(i + 1, len(our_player_sets)):
            shared = len(our_player_sets[i] & our_player_sets[j])
            overlap_scores.append(shared)
    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0

    # Print full postmortem
    print("\n" + "=" * 70)
    print("TOURNAMENT POSTMORTEM")
    print("=" * 70)
    print(f"  Contest:      {total_entries:,} entries")
    print(f"  Cash line:    {cash_score:.2f} pts (top 25% = rank <= {cash_cutoff:,})")
    print(f"  Winner:       {winner['pts']:.2f} pts ({winner['name']})")
    if our_entries:
        our_best  = our_entries[0]
        our_worst = our_entries[-1]
        our_avg   = sum(e["pts"] for e in our_entries) / n_ours
        print(f"\n  Our results ({n_ours} lineups):")
        print(f"    Best:  Rank {our_best['rank']:,}  ({our_best['pts']:.2f} pts)")
        print(f"    Worst: Rank {our_worst['rank']:,}  ({our_worst['pts']:.2f} pts)")
        _gap = our_avg - cash_score
        _gap_label = f"{_gap:+.2f} vs cash line" if _gap >= 0 else f"{abs(_gap):.2f} SHORT of cash line"
        print(f"    Avg:   {our_avg:.2f} pts  ({_gap_label})")
        print(f"    Lineups cashed: {sum(1 for e in our_entries if e['pts'] >= cash_score)}/{n_ours}")

    print(f"\n  Avg lineup overlap: {avg_overlap:.1f}/8 players shared between our lineups")
    if avg_overlap >= 5:
        print(f"    WARNING: HIGH overlap -- too many shared players, low diversity")

    if our_zeros:
        print(f"\n  [!] ZERO-SCORING PLAYERS WE ROSTERED (DNP/inactive):")
        for p, cnt in sorted(our_zeros.items(), key=lambda x: -x[1]):
            print(f"    {p}: {cnt}/{n_ours} lineups @ 0 pts each")

    if our_near_zero:
        print(f"\n  NEAR-ZERO (<10pt) PLAYERS WE ROSTERED:")
        for p, (cnt, sc) in sorted(our_near_zero.items(), key=lambda x: -x[1][0]):
            print(f"    {p}: {cnt}/{n_ours} lineups @ {sc:.1f} pts")

    if missed_stars:
        print(f"\n  ELITE SCORERS WE MISSED (scored 50+, 0 lineups):")
        for p, sc in sorted(missed_stars.items(), key=lambda x: -x[1]):
            own = player_own.get(p, 0)
            print(f"    {p}: {sc:.1f} pts  ({own:.1f}% field own)")

    if underexposed:
        print(f"\n  UNDER-EXPOSED STARS (scored 55+, <{max(1, n_ours//5)} lineups):")
        for p, (cnt, sc) in sorted(underexposed.items(), key=lambda x: -x[1][1]):
            own = player_own.get(p, 0)
            print(f"    {p}: {sc:.1f} pts  -- in {cnt}/{n_ours} lineups ({own:.1f}% field own)")

    print(f"\n  WINNING LINEUP ({winner['pts']:.2f} pts):")
    for pname in winner_names:
        sc = player_scores.get(pname, "?")
        print(f"    {pname}: {sc} pts")

    print("=" * 70)

    return {
        "total_entries":  total_entries,
        "cash_score":     cash_score,
        "winner_pts":     winner["pts"],
        "our_best_rank":  our_entries[0]["rank"] if our_entries else None,
        "our_avg_pts":    sum(e["pts"] for e in our_entries) / n_ours if our_entries else 0,
        "our_zeros":      our_zeros,
        "missed_stars":   missed_stars,
        "avg_overlap":    round(avg_overlap, 1),
        "player_scores":  player_scores,
    }


# ── DraftKings contest standings parser ──────────────────────────────────────
_DK_LINEUP_SLOTS = {"PG", "SG", "SF", "PF", "C", "F", "G", "UTIL"}


def _parse_dk_lineup_string(lineup_str: str) -> list[str]:
    """
    Parse a DK contest lineup string into a list of player names.
    Format: "PG LaMelo Ball SG Tyler Herro SF Brandon Miller PF ... UTIL ..."
    Returns player names in slot order, skipping "LOCKED" placeholders.
    """
    tokens = str(lineup_str).strip().split()
    names: list[str] = []
    current_parts: list[str] = []

    for token in tokens:
        if token in _DK_LINEUP_SLOTS:
            if current_parts:
                name = " ".join(current_parts)
                if name.upper() != "LOCKED":
                    names.append(name)
            current_parts = []
        else:
            current_parts.append(token)

    if current_parts:
        name = " ".join(current_parts)
        if name.upper() != "LOCKED":
            names.append(name)

    return names


def parse_contest_csv(csv_path, players: pd.DataFrame) -> dict:
    """
    Parse a DraftKings contest standings CSV into contest intelligence data.

    The DK contest export has two sections merged into one file:
      Left  (cols 0-5): Rank, EntryId, EntryName, TimeRemaining, Points, Lineup
      Right (cols 7-10): Player, Roster Position, %Drafted, FPTS
    These are independent — left = per-entry standings, right = per-player ownership.

    Returns:
      {
        "real_ownership":  {player_name: float pct},   # actual %Drafted in this contest
        "real_fpts":       {player_name: float},        # live DK points scored so far
        "field_lineups":   [set_of_names, ...],         # all opponent lineup sets
        "leader_lineups":  [set_of_names, ...],         # top-25% by current score
        "field_stacks":    {name_pair: int},            # how many lineups share each 2-player combo
        "entry_count":     int,                         # total entries parsed
        "leader_count":    int,
        "top_players":     [{"name","own_pct","fpts"}, ...],  # field-wide ownership report
      }
    """
    try:
        df = pd.read_csv(csv_path, header=0)
    except Exception as e:
        return {"error": str(e)}

    # ── Real ownership from right-side columns ────────────────────────────────
    real_ownership: dict[str, float] = {}
    real_fpts:      dict[str, float] = {}

    player_col = next((c for c in df.columns if "Player"    in str(c)), None)
    pct_col    = next((c for c in df.columns if "%Drafted"  in str(c) or "Drafted" in str(c)), None)
    fpts_col   = next((c for c in df.columns if "FPTS"      in str(c)), None)

    if player_col and pct_col:
        for _, row in df.iterrows():
            name = str(row.get(player_col, "")).strip()
            pct  = str(row.get(pct_col, "")).replace("%", "").strip()
            fpts = str(row.get(fpts_col, "0")).strip() if fpts_col else "0"
            if name and name not in ("nan", "Player"):
                try:
                    pct_val = float(pct)
                    # Keep the highest value — player appears multiple times in the
                    # file (once per entry they appear in); first occurrence = highest %
                    if name not in real_ownership or pct_val > real_ownership[name]:
                        real_ownership[name] = pct_val
                except ValueError:
                    pass
                try:
                    fpts_val = float(fpts)
                    if name not in real_fpts or fpts_val > real_fpts[name]:
                        real_fpts[name] = fpts_val
                except ValueError:
                    pass

    # ── Field lineups from left-side Lineup column ────────────────────────────
    lineup_col  = next((c for c in df.columns if c.strip().lower() == "lineup"), None)
    points_col  = next((c for c in df.columns if c.strip().lower() == "points"), None)
    field_lineups:  list[set] = []
    entry_points:   list[float] = []
    entries_parsed: int = 0

    if lineup_col:
        for _, row in df.iterrows():
            raw = str(row.get(lineup_col, "")).strip()
            if not raw or raw == "nan":
                continue
            names = _parse_dk_lineup_string(raw)
            if len(names) >= 6:
                field_lineups.append(set(names))
                pts = 0.0
                if points_col:
                    try:
                        pts = float(row.get(points_col, 0))
                    except (ValueError, TypeError):
                        pass
                entry_points.append(pts)
                entries_parsed += 1

    # Leader lineups = top 25% by current score
    if entry_points:
        cutoff = sorted(entry_points, reverse=True)[max(0, len(entry_points) // 4 - 1)]
        leader_lineups = [lu for lu, pts in zip(field_lineups, entry_points) if pts >= cutoff]
    else:
        leader_lineups = []

    # ── Field stack exposure: which 2-player combos are most common ───────────
    from itertools import combinations
    from collections import Counter
    pair_counts: Counter = Counter()
    for lu in field_lineups:
        for a, b in combinations(sorted(lu), 2):
            pair_counts[(a, b)] += 1

    top_pairs = {f"{a} / {b}": cnt for (a, b), cnt in pair_counts.most_common(30)}

    # ── Top players by ownership ──────────────────────────────────────────────
    top_players = sorted(
        [{"name": n, "own_pct": p, "fpts": real_fpts.get(n, 0.0)}
         for n, p in real_ownership.items()],
        key=lambda x: -x["own_pct"],
    )[:40]

    return {
        "real_ownership":  real_ownership,
        "real_fpts":       real_fpts,
        "field_lineups":   [list(lu) for lu in field_lineups],
        "leader_lineups":  [list(lu) for lu in leader_lineups],
        "field_stacks":    top_pairs,
        "entry_count":     entries_parsed,
        "leader_count":    len(leader_lineups),
        "top_players":     top_players,
    }


# ── DK Upload CSV re-import ───────────────────────────────────────────────────
def parse_lineup_csv(csv_path, players: pd.DataFrame) -> tuple[list, list]:
    """
    Re-import a previously exported DK upload CSV back into lineup state.

    Accepts both formats:
      1. Our own export  — columns PG/SG/SF/PF/C/G/F/UTIL, cells = "Name (ID)" name_id
      2. Standard DK CSV — same column names, same cell format (identical structure)

    Returns:
      (lineups, warnings)
        lineups  — list of lineup dicts compatible with late_swap_lineups() and renderLineups()
        warnings — list of strings describing any unmatched players

    Matching priority:
      1. Exact name_id match   (Name + ID string, e.g. "Jayson Tatum (10215718)")
      2. Player ID extracted from parentheses
      3. Fuzzy name match (cutoff 0.82) as last resort
    """
    from difflib import get_close_matches

    # Build lookup indexes from current player pool
    name_id_to_pid  = dict(zip(players["name_id"].astype(str),
                               players["player_id"].astype(str)))
    pid_to_row      = {str(row["player_id"]): row
                       for _, row in players.iterrows()}
    name_lower_map  = {row["name"].lower(): str(row["player_id"])
                       for _, row in players.iterrows()}

    def resolve(cell_value: str) -> str | None:
        """Return player_id for a cell value, or None if not matched."""
        v = str(cell_value).strip()
        if not v:
            return None
        # 1. Exact name_id match
        if v in name_id_to_pid:
            return name_id_to_pid[v]
        # 2. Extract numeric DK ID from parentheses — "Name (12345678)"
        import re
        m = re.search(r"\((\d{6,})\)", v)
        if m:
            dk_id = m.group(1)
            # Try matching against player_id directly
            if dk_id in pid_to_row:
                return dk_id
            # Try as suffix of any name_id
            for nid, pid in name_id_to_pid.items():
                if nid.endswith(f"({dk_id})"):
                    return pid
        # 3. Fuzzy name match on the text before the parenthesis
        raw_name = re.sub(r"\s*\(.*\)", "", v).strip().lower()
        close    = get_close_matches(raw_name, name_lower_map.keys(), n=1, cutoff=0.82)
        if close:
            return name_lower_map[close[0]]
        return None

    try:
        df_csv = pd.read_csv(csv_path)
    except Exception as e:
        return [], [f"Could not read file: {e}"]

    # Detect if the file has the expected slot columns
    slot_cols = [c for c in _SLOT_ORDER if c in df_csv.columns]
    if len(slot_cols) < 6:
        return [], [f"CSV does not look like a DK lineup export. Expected columns: {_SLOT_ORDER}"]

    lineups  : list[dict] = []
    warnings : list[str]  = []

    for row_idx, row in df_csv.iterrows():
        slot_assignment : dict[str, str] = {}
        player_ids      : list[str]      = []
        unmatched       : list[str]      = []

        for slot in _SLOT_ORDER:
            cell = row.get(slot, "")
            pid  = resolve(str(cell)) if pd.notna(cell) and str(cell).strip() else None
            if pid:
                slot_assignment[slot] = pid
                if pid not in player_ids:
                    player_ids.append(pid)
            else:
                if pd.notna(cell) and str(cell).strip():
                    unmatched.append(f"'{cell}' @ {slot}")

        if unmatched:
            warnings.append(f"Lineup {row_idx+1}: could not match {', '.join(unmatched)}")

        if len(player_ids) < 6:
            warnings.append(f"Lineup {row_idx+1}: only {len(player_ids)} players matched, skipping")
            continue

        # Reconstruct lineup fields from current player pool
        matched_rows = [pid_to_row[p] for p in player_ids if p in pid_to_row]
        names        = [r["name"]           for r in matched_rows]
        positions    = [r["primary_position"] for r in matched_rows]
        teams        = [r["team"]            for r in matched_rows]
        salaries     = [int(r["salary"])     for r in matched_rows]
        projections  = [round(float(r["proj_pts_dk"]), 2) for r in matched_rows]

        lineups.append({
            "lineup_num":      row_idx + 1,
            "player_ids":      player_ids,
            "names":           names,
            "positions":       positions,
            "teams":           teams,
            "salaries":        salaries,
            "projections":     projections,
            "slot_assignment": slot_assignment,
            "total_salary":    sum(salaries),
            "proj_pts":        round(sum(projections), 2),
            "ceiling":         round(sum(float(r.get("ceiling", r["proj_pts_dk"])) for r in matched_rows), 2),
            "proj_own":        round(float(sum(float(r.get("proj_own", 10)) for r in matched_rows) / len(matched_rows)), 1),
            "leverage":        0,
            "swapped":         False,
            "swap_method":     None,
            "imported":        True,   # flag so UI can show it was re-loaded
        })

    return lineups, warnings


# ── Exposure report ───────────────────────────────────────────────────────────
def exposure_report(lineups: list, n_lineups: int) -> pd.DataFrame:
    exp = {}
    for lu in lineups:
        for name in lu["names"]:
            exp[name] = exp.get(name, 0) + 1
    rows = [
        {"player": name, "count": cnt, "exposure_pct": round(cnt / n_lineups * 100, 1)}
        for name, cnt in sorted(exp.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows)


# ── Player correlation matrix ─────────────────────────────────────────────────
# DFS correlation is bidirectional:
#   Positive = players whose DFS scores rise together (stack them)
#   Negative = players competing for the same usage (avoid pairing)
#
# Methodology:
#   - Position-pair correlation approximates historical game-log DFS score correlation
#   - Same-team PG→C/PF: high positive (assist-to-bucket — the PG feeds the big)
#   - Same-team G+G: mildly negative (usage competition for ball-handling)
#   - Same-team PF+C: negative (paint/rebound competition)
#   - Cross-team (game stack): mild positive (high O/U pace benefits both rosters)
#
# On/Off Split impact:
#   When Player A is OUT, their usage redistributes — but NOT uniformly:
#   - Same-position teammates absorb shot volume (positive replacement value)
#   - Players A was FEEDING via assists LOSE value (negative downstream impact)
#   - Example: Star PG is OUT → backup PG gains usage, but C who relied on PG
#     assists may actually LOSE value because that assist source is gone.
#   We model this via estimate_usage_absorption() below.
#
# For full accuracy: NBA API playerdashlineups + playerdashptpass (assist-to-bucket
# pairs) would give real historical correlation. The approximation here is accurate
# enough for most slates and runs without API calls.

POSITION_PAIR_CORRELATION = {
    # Same-team pairs: (primary_pos_1, primary_pos_2) -> correlation
    # Higher = more positively correlated DFS outcomes
    ("PG", "C"):   +0.60,   # Classic PG→C assist stack
    ("PG", "PF"):  +0.45,   # PG feeds stretch PF
    ("PG", "SF"):  +0.30,   # PG feeds wing
    ("PG", "SG"):  -0.15,   # Usage competition
    ("PG", "PG"):  -0.35,   # Duplicate position — strong usage competition
    ("SG", "C"):   +0.40,   # SG can feed C from mid-range
    ("SG", "PF"):  +0.30,
    ("SG", "SF"):  -0.10,
    ("SG", "SG"):  -0.30,
    ("SF", "C"):   +0.25,
    ("SF", "PF"):  -0.05,
    ("SF", "SF"):  -0.20,
    ("PF", "C"):   -0.20,   # Paint competition
    ("PF", "PF"):  -0.25,
    ("C",  "C"):   -0.30,
}

# ── Game Script / Blowout Modeling ───────────────────────────────────────────
def apply_game_script_adjustments(
    players: pd.DataFrame,
    game_totals: dict,
) -> pd.DataFrame:
    """
    Reprice players based on expected game script derived from the point spread.

    Logic:
      - Spread = |home_implied - away_implied|
      - Blowout risk (spread > 11): bench/role players on FAVORITE get reduced
        minutes (pulled early); stars on UNDERDOG get reduced 4Q production.
        Role players on UNDERDOG get a garbage-time ceiling boost.
      - Competitive (spread < 6): no adjustment — game goes full rotation.

    Salary tiers used as minute-certainty proxy:
      Starter tier  : salary >= $7,000
      Rotation tier : $5,000 – $6,999
      Bench tier    : < $5,000
    """
    df = players.copy()
    if "matchup" not in df.columns:
        return df

    for matchup, totals in game_totals.items():
        home_imp = float(totals.get("home_implied", 0))
        away_imp = float(totals.get("away_implied", 0))
        if home_imp == 0 or away_imp == 0:
            continue

        spread = abs(home_imp - away_imp)
        if spread < 6:
            continue   # competitive game — no adjustment

        fav_team  = matchup.split("@")[1] if home_imp > away_imp else matchup.split("@")[0]
        dog_team  = matchup.split("@")[0] if home_imp > away_imp else matchup.split("@")[1]
        game_mask = df["matchup"] == matchup

        # ── Blowout risk tiers ────────────────────────────────────────────────
        if spread >= 12:
            # Heavy blowout — strong adjustments
            fav_bench_mult   = 0.88   # bench on favorite loses garbage-time minutes
            fav_star_mult    = 1.02   # starters on favorite play first 3Q comfortably
            dog_star_mult    = 0.94   # stars on underdog may be pulled late
            dog_bench_mult   = 1.08   # underdog bench gets garbage time ceiling boost
        elif spread >= 8:
            # Moderate blowout — softer adjustments
            fav_bench_mult   = 0.93
            fav_star_mult    = 1.01
            dog_star_mult    = 0.97
            dog_bench_mult   = 1.04
        else:
            # Slight mismatch (6-8 pts) — minimal
            fav_bench_mult   = 0.97
            fav_star_mult    = 1.00
            dog_star_mult    = 0.99
            dog_bench_mult   = 1.02

        def _apply_script(team: str, star_mult: float, bench_mult: float) -> None:
            tmask = game_mask & (df["team"] == team)
            if not tmask.any():
                return
            star_mask  = tmask & (df["salary"] >= 7000)
            bench_mask = tmask & (df["salary"] <  5000)
            rot_mask   = tmask & (df["salary"] >= 5000) & (df["salary"] < 7000)

            for mask_sub, mult in [(star_mask, star_mult),
                                   (rot_mask, (star_mult + bench_mult) / 2),
                                   (bench_mask, bench_mult)]:
                if not mask_sub.any():
                    continue
                df.loc[mask_sub, "proj_pts_dk"] = (
                    df.loc[mask_sub, "proj_pts_dk"] * mult
                ).round(2)
                df.loc[mask_sub, "ceiling"] = (
                    df.loc[mask_sub, "ceiling"] * mult
                ).round(2)

        _apply_script(fav_team, fav_star_mult, fav_bench_mult)
        _apply_script(dog_team, dog_star_mult, dog_bench_mult)

    # Recompute gpp_score everywhere
    if "gpp_score" in df.columns:
        df["gpp_score"] = (
            df["ceiling"] * 0.60
            + df["proj_pts_dk"] * 0.25
            + (1 - df["proj_own"] / 100) * 10
        ).round(3)

    return df


# ── Positional Scarcity ────────────────────────────────────────────────────────
def compute_positional_scarcity(players: pd.DataFrame) -> dict:
    """
    Count viable players per DK roster slot and identify scarce positions.

    Returns:
        {
          "PG": int, "SG": int, ..., "UTIL": int,
          "scarce_slots": [list of slot names with < 5 viable],
          "scarcity_score": float 0-1,
          "construction_notes": [str, ...],
        }
    """
    _SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    VIABLE_PROJ_FLOOR = 12.0    # minimum projection to be "viable"

    slot_counts: dict[str, int] = {}

    if "eligible_slots" in players.columns:
        for slot in _SLOTS:
            viable = players[
                (players["proj_pts_dk"] >= VIABLE_PROJ_FLOOR)
                & players["eligible_slots"].apply(
                    lambda s: slot in (s if isinstance(s, list) else [])
                )
            ]
            slot_counts[slot] = len(viable)
    else:
        # Fallback using primary_position
        pos_to_slots = {
            "PG": ["PG", "G", "UTIL"],
            "SG": ["SG", "G", "UTIL"],
            "SF": ["SF", "F", "UTIL"],
            "PF": ["PF", "F", "UTIL"],
            "C":  ["C", "UTIL"],
        }
        for slot in _SLOTS:
            eligible_pos = [p for p, sl in pos_to_slots.items() if slot in sl]
            count = len(players[
                (players["proj_pts_dk"] >= VIABLE_PROJ_FLOOR)
                & players["primary_position"].isin(eligible_pos)
            ])
            slot_counts[slot] = count

    scarce = [s for s, n in slot_counts.items() if n < 5]
    scarcity_score = round(len(scarce) / len(_SLOTS), 2)

    notes = []
    if scarce:
        notes.append(f"Scarce slots: {', '.join(scarce)} — ownership concentration expected.")
    for slot, n in slot_counts.items():
        if n < 3:
            notes.append(f"CRITICAL: {slot} has only {n} viable options — near-forced chalk.")
        elif n < 5:
            notes.append(f"WARNING: {slot} has only {n} viable options.")

    return {
        **slot_counts,
        "scarce_slots":       scarce,
        "scarcity_score":     scarcity_score,
        "construction_notes": notes,
    }


def build_player_correlation(players: pd.DataFrame) -> dict:
    """
    Build a pairwise DFS correlation estimate for all players in the pool.
    Returns {(pid1, pid2): float} where float in [-1.0, +1.0].

    Positive  = stack these players (correlated upside)
    Negative  = avoid pairing (usage competition or conflicting roles)
    Near-zero = independent players (different teams/games)

    Incorporates:
      1. Position-pair correlation (usage competition vs assist synergy)
      2. Same-team multiplier (stronger relationship when sharing court)
      3. Game-stack bonus for cross-team pairing in high O/U games
    """
    pid_list = players["player_id"].astype(str).tolist()
    n = len(players)
    corr_map = {}

    for i in range(n):
        for j in range(i + 1, n):
            pi = players.iloc[i]
            pj = players.iloc[j]

            pid_i = str(pi["player_id"])
            pid_j = str(pj["player_id"])

            if pi["team"] == pj["team"]:
                # Same team: use position-pair correlation
                pos_i = pi["primary_position"]
                pos_j = pj["primary_position"]
                key   = (pos_i, pos_j) if (pos_i, pos_j) in POSITION_PAIR_CORRELATION \
                        else (pos_j, pos_i) if (pos_j, pos_i) in POSITION_PAIR_CORRELATION \
                        else None
                base = POSITION_PAIR_CORRELATION.get(key, 0.0) if key else 0.10
                # Salary weight: both being high-salary raises correlation magnitude
                sal_factor = min(pi["salary"], pj["salary"]) / 8000
                corr = base * (0.7 + 0.3 * sal_factor)

            elif pi["matchup"] == pj["matchup"]:
                # Same game, opposing teams (game-stack / bring-back)
                gt = GAME_TOTALS.get(pi["matchup"], {}).get("total", 225.0)
                # Higher O/U = stronger cross-team correlation
                game_corr = (gt - 215) / 40  # 0.0 at 215, +0.375 at 230
                corr = max(0.05, game_corr)
            else:
                # Different games: effectively uncorrelated (slightly negative
                # because they compete for salary cap allocation)
                corr = -0.05

            corr_map[(pid_i, pid_j)] = round(corr, 3)
            corr_map[(pid_j, pid_i)] = round(corr, 3)

    return corr_map


def estimate_usage_absorption(
    out_player: pd.Series,
    players: pd.DataFrame,
    usage_data: dict | None = None,
    on_off_data: dict | None = None,
) -> pd.DataFrame:
    """
    When a player is OUT, redistribute their USG% to all remaining teammates.

    Algorithm:
      1. Look up the OUT player's real USG% from usage_data (NBA Stats API).
         Falls back to salary-proxy if data unavailable.
      2. ALL same-team players (not just same-position) receive a usage boost
         weighted 60% by position group affinity and 40% by minutes played.
         This correctly captures cross-position boosts like a big (Kel'el Ware)
         benefiting when multiple guards (Rozier, Wiggins) are absent.
      3. Convert delta_usg → delta_dk using each player's own efficiency ratio
         (proj_pts_dk / usg_pct) so high-efficiency players gain more DK value
         per absorbed usage point than inefficient ones.
      4. When on_off_data is provided (from OnOffAgent), use the empirically
         observed DK delta directly for teammates with sufficient sample size.
         Falls back to usage-quantity calculation when on_off_data is absent
         or the sample is too small.
      5. Retain PG playmaker-loss penalty for bigs when the primary ball-handler
         is out and no backup PG is rostered.

    Parameters
    ----------
    out_player  : row from the players DataFrame for the OUT player
    players     : full player pool DataFrame
    usage_data  : {name_lower: {usg_pct, min_pg, team}} from fetch_player_usage_rates()
                  Pass None to use salary-based fallback.
    on_off_data : {player_id_str: {"delta_dk", "n_with", "n_without", ...}}
                  Output of OnOffAgent.compute().  When present and reliable,
                  the empirical delta_dk overrides the usage-quantity estimate.
    """
    if out_player.empty:
        return players

    out_team = out_player["team"]
    out_pos  = out_player["primary_position"]
    out_sal  = out_player["salary"]
    out_name = str(out_player.get("name", "")).lower().strip()

    df = players.copy()
    team_mask = df["team"] == out_team

    guard_positions = {"PG", "SG", "G"}
    big_positions   = {"PF", "C", "F"}

    on_off_splits = _load_on_off_splits()
    on_off_logs: list[str] = []

    # ── Step 1: Determine the OUT player's USG% ───────────────────────────────
    out_usg: float = 0.0
    ud = usage_data or {}
    if ud:
        entry = ud.get(out_name)
        if entry is None:
            # Fuzzy: first-name + last-name token match
            for k, v in ud.items():
                if k and out_name and (k.split()[-1] == out_name.split()[-1]):
                    if v.get("team", "") == out_team:
                        entry = v
                        break
        if entry:
            out_usg = float(entry.get("usg_pct", 0.0))

    if out_usg <= 0:
        # Salary-proxy fallback: median NBA player earns ~$6 500 salary at ~20% USG
        out_usg = max(5.0, (out_sal / 6500) * 20.0)

    # ── Step 2: Build teammate candidate list ─────────────────────────────────
    teammates = df[team_mask & (df["player_id"] != out_player["player_id"])].copy()
    if teammates.empty:
        return df

    # Attach real minutes from usage_data (or salary proxy)
    def _min_pg(row) -> float:
        name_l = str(row.get("name", "")).lower().strip()
        if ud:
            e = ud.get(name_l)
            if e:
                return max(1.0, float(e.get("min_pg", 1.0)))
        # Salary proxy: $9k player ≈ 35 min, $3.5k ≈ 12 min
        return max(1.0, (float(row["salary"]) / 9000) * 35.0)

    teammates["_min_pg"] = teammates.apply(_min_pg, axis=1)

    # Position-group affinity weight (same group absorbs more)
    def _pos_affinity(pos: str) -> float:
        if out_pos in guard_positions and pos in guard_positions:
            return 2.0
        if out_pos in big_positions and pos in big_positions:
            return 2.0
        # Cross-position: bigs/wings pick up secondary usage
        return 1.0

    teammates["_affinity"] = teammates["primary_position"].apply(_pos_affinity)

    # Combined weight: 60% position affinity, 40% minutes
    min_total  = teammates["_min_pg"].sum()
    teammates["_min_norm"]   = teammates["_min_pg"] / min_total
    aff_total  = teammates["_affinity"].sum()
    teammates["_aff_norm"]   = teammates["_affinity"] / aff_total
    teammates["_weight"]     = 0.40 * teammates["_min_norm"] + 0.60 * teammates["_aff_norm"]
    # Re-normalise to sum to 1.0
    w_total = teammates["_weight"].sum()
    if w_total > 0:
        teammates["_weight"] /= w_total

    # ── Step 3: Compute delta_dk for each teammate ────────────────────────────
    # We distribute 85% of the OUT player's usage (empirical retention factor)
    freed_usg = out_usg * 0.85
    _ood = on_off_data or {}

    for idx_b, row_b in teammates.iterrows():
        tm_pid = str(row_b.get("player_id", ""))
        b_proj = float(df.loc[idx_b, "proj_pts_dk"])
        b_name_disp = str(row_b.get("name", row_b.get("player_name", tm_pid)))
        b_name_lower = b_name_disp.lower().strip()

        # Prefer cached ON/OFF uplift when star-specific sample exists
        split_entry = on_off_splits.get(b_name_lower, {}) if on_off_splits else {}
        if split_entry and str(split_entry.get("primary_star", "")).lower().strip() == out_name:
            uplift = float(split_entry.get("uplift_factor", 1.0))
            new_proj = round(b_proj * uplift, 2)
            df.loc[idx_b, "proj_pts_dk"] = new_proj
            df.loc[idx_b, "ceiling"] = round(float(df.loc[idx_b, "ceiling"]) * uplift, 2)
            df.loc[idx_b, "gpp_score"] = round(
                df.loc[idx_b, "ceiling"] * 0.60
                + df.loc[idx_b, "proj_pts_dk"] * 0.25
                + (1 - float(df.loc[idx_b, "proj_own"]) / 100) * 10,
                3,
            )
            on_off_logs.append(
                f"[usage] {b_name_disp} adjusted {uplift:.2f}x with {out_player.get('name', out_name.title())} OUT"
            )
            continue

        # Prefer empirical on/off delta when available and reliable
        if tm_pid and tm_pid in _ood:
            oo_entry = _ood[tm_pid]
            boost = round(float(oo_entry["delta_dk"]), 2)
        else:
            # Usage-quantity fallback
            delta_usg = freed_usg * row_b["_weight"]

            # Player efficiency ratio: how many DK pts per 1% of USG
            b_name = b_name_lower
            b_usg  = 20.0  # league avg fallback
            if ud:
                b_entry = ud.get(b_name)
                if b_entry:
                    b_usg = max(5.0, float(b_entry.get("usg_pct", 20.0)))

            # DK efficiency: avoid /0 and cap to reasonable range
            efficiency = b_proj / b_usg if b_usg > 0 else 1.5
            efficiency = min(efficiency, 3.0)

            boost = round(delta_usg * efficiency, 2)
        df.loc[idx_b, "proj_pts_dk"] = round(b_proj + boost, 2)
        df.loc[idx_b, "ceiling"]     = round(float(df.loc[idx_b, "ceiling"]) + boost * 1.3, 2)
        df.loc[idx_b, "gpp_score"]   = round(
            df.loc[idx_b, "ceiling"] * 0.60
            + df.loc[idx_b, "proj_pts_dk"] * 0.25
            + (1 - float(df.loc[idx_b, "proj_own"]) / 100) * 10,
            3,
        )

    if on_off_logs:
        for msg in on_off_logs:
            print(msg)

    # ── Step 4: PG playmaker-loss penalty ─────────────────────────────────────
    # When the primary ball-handler is OUT and no backup PG is in the pool,
    # bigs lose assist-derived scoring (fewer easy lob/putback opportunities)
    if out_pos == "PG" and out_sal >= 6000:
        backup_pgs = teammates[teammates["primary_position"] == "PG"]
        if backup_pgs.empty or backup_pgs["salary"].max() < 5000:
            # No credible backup PG → bigs take a mild penalty
            big_mask = team_mask & df["primary_position"].isin(big_positions)
            for idx_b in df[big_mask].index:
                penalty = df.loc[idx_b, "proj_pts_dk"] * 0.10
                df.loc[idx_b, "proj_pts_dk"] = round(float(df.loc[idx_b, "proj_pts_dk"]) - penalty, 2)
                df.loc[idx_b, "ceiling"]     = round(float(df.loc[idx_b, "ceiling"])     - penalty, 2)
                df.loc[idx_b, "gpp_score"]   = round(
                    df.loc[idx_b, "ceiling"] * 0.60
                    + df.loc[idx_b, "proj_pts_dk"] * 0.25
                    + (1 - float(df.loc[idx_b, "proj_own"]) / 100) * 10,
                    3,
                )

    return df


def compute_lineup_usage_impact(
    out_players: list,
    player_pool: pd.DataFrame,
    usage_data: dict | None = None,
    on_off_map: dict | None = None,
) -> list[dict]:
    """
    Return a human-readable summary of who benefits most when `out_players` are absent.

    Parameters
    ----------
    out_players  : list of player name strings (e.g. ["Terry Rozier", "Kel'el Ware"])
                   OR list of player_id values
    player_pool  : full DK player pool DataFrame
    usage_data   : output of fetch_player_usage_rates(); fetched here if None

    Returns
    -------
    List of dicts sorted by delta_dk descending:
      [{name, team, position, proj_pts_before, proj_pts_after, delta_dk,
        delta_usg_pct, usg_before, usg_after}, ...]

    Only players with delta_dk >= 0.3 are returned (noise filter).
    """
    if not out_players:
        return []

    ud = usage_data or fetch_player_usage_rates()

    # Normalise out_players to player_id set
    pool = player_pool.copy()
    pool["_name_lower"] = pool.get("name", pool.get("player_name", pd.Series(dtype=str))).str.lower().str.strip()

    out_ids: set = set()
    out_rows: list = []
    for item in out_players:
        # Try as name string first
        name_l = str(item).lower().strip()
        match  = pool[pool["_name_lower"] == name_l]
        if match.empty:
            # Try as player_id
            match = pool[pool["player_id"].astype(str) == str(item)]
        if not match.empty:
            out_ids.add(str(match.iloc[0]["player_id"]))
            out_rows.append(match.iloc[0])

    if not out_rows:
        return []

    # Snapshot projections before any adjustments
    before = pool.set_index("player_id")["proj_pts_dk"].to_dict()

    # Apply absorption for each OUT player sequentially
    _oom = on_off_map or {}
    adjusted = pool.copy()
    for row in out_rows:
        out_pid  = str(row["player_id"])
        oo_entry = _oom.get(out_pid)
        adjusted = estimate_usage_absorption(row, adjusted, usage_data=ud, on_off_data=oo_entry)

    after = adjusted.set_index("player_id")["proj_pts_dk"].to_dict()

    results = []
    for pid, proj_after in after.items():
        if str(pid) in out_ids:
            continue
        proj_before = before.get(pid, proj_after)
        delta = round(proj_after - proj_before, 2)
        if delta < 0.3:
            continue

        row = pool.set_index("player_id").loc[pid] if pid in pool.set_index("player_id").index else None
        if row is None:
            continue

        name_l   = str(row.get("_name_lower", ""))
        b_entry  = ud.get(name_l, {})
        usg_before = float(b_entry.get("usg_pct", 0.0))
        # Estimate post-adjustment USG based on freed_usg redistribution
        # (approximate: delta_dk / efficiency ≈ delta_usg)
        eff = proj_before / usg_before if usg_before > 0 else 1.5
        eff = min(eff, 3.0)
        delta_usg = round(delta / eff, 1) if eff > 0 else 0.0
        usg_after  = round(usg_before + delta_usg, 1)

        results.append({
            "name":            str(row.get("name", row.get("player_name", pid))),
            "team":            str(row.get("team", "")),
            "position":        str(row.get("primary_position", "")),
            "proj_pts_before": round(float(proj_before), 2),
            "proj_pts_after":  round(float(proj_after),  2),
            "delta_dk":        delta,
            "usg_before":      usg_before,
            "delta_usg_pct":   delta_usg,
            "usg_after":       usg_after,
        })

    results.sort(key=lambda x: x["delta_dk"], reverse=True)
    return results[:12]  # top 12 beneficiaries


def compute_starting_lineup_usage(
    team: str,
    player_pool: pd.DataFrame,
    n_starters: int = 5,
) -> dict:
    """
    Rotowire-style starting lineup usage breakdown.

    Identifies the likely starting 5 for a team (by salary tier + avg_pts),
    then computes each player's share of the starting lineup's possessions using
    ESPN game log data (FGA + 0.44*FTA + TOV).  Falls back to avg_pts proxy
    when ESPN data is unavailable.

    Example output:
        {
            "team": "CLE",
            "minutes_together": 28,
            "starters": [
                {"name": "Evan Mobley",      "usg_pct": 35.2, "min_pg": 33.1},
                {"name": "Donovan Mitchell",  "usg_pct": 24.1, "min_pg": 33.4},
                {"name": "James Harden",      "usg_pct": 18.3, "min_pg": 29.8},
                {"name": "Sam Merrill",       "usg_pct": 12.9, "min_pg": 27.0},
                {"name": "Dean Wade",         "usg_pct": 9.5,  "min_pg": 22.4},
            ],
        }

    Mirrors the Rotowire "Starting Lineup Usage Rates" display so it can be
    used to predict how possession share shifts when one starter is injured.
    """
    team_players = player_pool[player_pool["team"] == team].copy()
    if team_players.empty:
        return {"team": team, "starters": [], "minutes_together": 0}

    # Identify likely starters: top n_starters by salary (best proxy without
    # explicit starter/bench flag from the DK slate)
    top = team_players.nlargest(n_starters, "salary")
    starter_names = top["name"].tolist() if "name" in top.columns else top.get("player_name", pd.Series()).tolist()

    # Try ESPN game-log based USG% first
    try:
        from agents.bbref_on_off_agent import BBRefOnOffAgent
        agent  = BBRefOnOffAgent()
        result = agent.get_starting_lineup_usage(team, starter_names)
        if result.get("starters"):
            return result
    except Exception:
        pass

    # Fallback: use avg_pts as possession proxy, normalise to 100%
    lineup_stats = []
    for _, row in top.iterrows():
        name    = str(row.get("name") or row.get("player_name") or "")
        avg_pts = float(row.get("avg_pts", 0) or 0)
        min_pg  = float(row.get("avg_min", row.get("min_pg", 25.0)) or 25.0)
        lineup_stats.append({"name": name, "usg_pct": max(avg_pts, 1.0), "min_pg": min_pg})

    total = sum(s["usg_pct"] for s in lineup_stats) or 1.0
    for s in lineup_stats:
        s["usg_pct"] = round(s["usg_pct"] / total * 100.0, 1)
    lineup_stats.sort(key=lambda x: -x["usg_pct"])

    min_pgs = [s["min_pg"] for s in lineup_stats]
    return {
        "team":             team,
        "starters":         lineup_stats,
        "minutes_together": int(min(min_pgs)) if min_pgs else 0,
    }


def score_lineup_leverage(lineup: dict, players: pd.DataFrame) -> dict:
    """
    Score a lineup's contest leverage — how differentiated it is from the
    estimated DFS field.

    Metrics:
      - avg_own:    average projected ownership (lower = more leverage)
      - chalk_ct:   players over 30% ownership (chalk — everyone has them)
      - low_own_ct: players under 15% ownership (true differentiators)
      - leverage:   0-100 score (higher = more differentiated from field)
      - stack_quality: sum of pairwise correlation for same-team pairs
    """
    pids = [str(p) for p in lineup["player_ids"]]
    pool = players.set_index("player_id")

    # Use calibrated field ownership if available (field_own_est = proj_own / 3).
    # Pro data (7-date, 800+ lineups): FC proj_own is 3x higher than actual field own.
    owns = []
    for p in pids:
        if p not in pool.index:
            continue
        row = pool.loc[p]
        if hasattr(row, 'field_own_est') and row.field_own_est is not None and not (isinstance(row.field_own_est, float) and row.field_own_est != row.field_own_est):
            owns.append(float(row.field_own_est))
        elif hasattr(row, 'proj_own'):
            owns.append(float(row.proj_own) / 3.0)
        else:
            owns.append(15.0)
    sals  = [float(pool.loc[p, "salary"])   for p in pids if p in pool.index]
    teams = [str(pool.loc[p, "team"])       for p in pids if p in pool.index]

    avg_own    = round(sum(owns) / len(owns), 1) if owns else 0
    chalk_ct   = sum(1 for o in owns if o >= 9)    # calibrated: FC 27% = actual ~9%
    low_own_ct = sum(1 for o in owns if o <= 4)    # calibrated: FC 12% = actual ~4%

    # Leverage score: 100 = perfectly contrarian, 50 = balanced, 0 = all chalk
    # Calibrated for realistic ownership range (5–50%) from the power-curve proj_own model:
    #   avg_own=22%, 1 chalk, 2 low-own → 100 - 39.6 + 10 - 4 = 66  (contrarian)
    #   avg_own=28%, 3 chalk, 1 low-own → 100 - 50.4 + 5 - 12 = 43  (balanced)
    #   avg_own=38%, 5 chalk, 0 low-own → 100 - 68.4 + 0 - 20 = 12  (chalk-heavy)
    leverage = max(0, min(100, round(100 - avg_own * 1.8 + low_own_ct * 5 - chalk_ct * 4, 1)))

    # Stack quality: do we have a primary + secondary stack?
    from collections import Counter
    team_counts = Counter(teams)
    primary_stack   = max(team_counts.values()) if team_counts else 0
    secondary_stack = sorted(team_counts.values(), reverse=True)[1] if len(team_counts) > 1 else 0
    has_game_stack  = False
    # Check if primary and secondary stacks are from the same game (bring-back)
    if primary_stack >= 3 and secondary_stack >= 2:
        top_teams = [t for t, c in team_counts.most_common(2)]
        if len(top_teams) == 2:
            for matchup in GAME_TOTALS:
                teams_in_game = matchup.split("@")
                if top_teams[0] in teams_in_game and top_teams[1] in teams_in_game:
                    has_game_stack = True
                    break

    return {
        "avg_own":       avg_own,
        "chalk_ct":      chalk_ct,
        "low_own_ct":    low_own_ct,
        "leverage":      leverage,
        "primary_stack": primary_stack,
        "has_game_stack": has_game_stack,
    }


# ── Stack analysis ────────────────────────────────────────────────────────────
def find_top_stacks(players: pd.DataFrame) -> list:
    # Filter to only active, likely-playing players before computing stacks.
    # OUT players from FC (fc_mins=0) or high-DNP-risk players corrupt the
    # stack rankings by inflating team projections with players who won't play.
    active = players.copy()

    # 1. FC minutes = 0 → confirmed not playing (primary signal when FC is loaded)
    if "fc_mins" in active.columns:
        active = active[~(active["fc_mins"].notna() & (active["fc_mins"] < 1))]

    # 2. proj_mins < 5 → our computed estimate says essentially no minutes
    if "proj_mins" in active.columns:
        active = active[~(active["proj_mins"].notna() & (active["proj_mins"] < 5))]

    # 3. High DNP risk → likely out (covers cases before FC is uploaded)
    if "dnp_risk" in active.columns:
        active = active[active["dnp_risk"] < 0.60]

    stacks = []
    for team in active["team"].unique():
        tp = active[active["team"] == team].nlargest(4, "proj_pts_dk")
        if len(tp) >= 2:
            stacks.append({
                "team":    team,
                "players": tp["name"].tolist(),
                "proj":    round(tp["proj_pts_dk"].sum(), 1),
                "game":    tp["matchup"].iloc[0],
                "game_total": GAME_TOTALS.get(tp["matchup"].iloc[0], {}).get("total", 0),
            })
    return sorted(stacks, key=lambda x: -x["proj"])


# ── Slate analysis printout ───────────────────────────────────────────────────
def print_slate_analysis(players: pd.DataFrame):
    print("\n" + "="*70)
    print(f"SLATE ANALYSIS -- {date.today().isoformat()}")
    print(f"Contest: {CONTEST['name']} | ${CONTEST['entry_fee']} entry | "
          f"Max {CONTEST['max_entries']} entries")
    print("="*70)

    print(f"\nTotal players in pool: {len(players)}")

    # Flags
    zero_avg = players[players["avg_pts"] <= 5].copy()
    if not zero_avg.empty:
        print(f"\nPLAYERS WITH 0 OR VERY LOW AVG -- LIKELY OUT / INJURED:")
        for _, r in zero_avg.iterrows():
            flag = "LIKELY OUT" if r["avg_pts"] == 0 else "low avg"
            print(f"  {r['name']:<30s} ${r['salary']:,}  avg={r['avg_pts']}  [{flag}]")

    print(f"\nGAME TOTALS (tonight):")
    for matchup, data in sorted(GAME_TOTALS.items(), key=lambda x: -x[1]["total"]):
        print(f"  {matchup:<10s}  O/U: {data['total']}  "
              f"(Home: {data['home_implied']} | Away: {data['away_implied']})")

    print(f"\nTOP 20 PROJECTIONS:")
    print(f"  {'Name':<26s} {'Team':<5s} {'Pos':<6s} {'Salary':>7s}  "
          f"{'Proj':>6s}  {'Ceil':>6s}  {'Value':>6s}  {'Est Own':>7s}")
    print("  " + "-"*70)
    for _, r in players.head(20).iterrows():
        print(f"  {r['name']:<26s} {r['team']:<5s} {r['primary_position']:<6s} "
              f"${r['salary']:>6,}  {r['proj_pts_dk']:>6.1f}  {r['ceiling']:>6.1f}  "
              f"{r['value']:>6.2f}x  {r['proj_own']:>5.1f}%")

    print(f"\nTOP VALUE PLAYS (proj/salary ratio):")
    top_val = players[players["salary"] >= 4000].nlargest(10, "value")
    for _, r in top_val.iterrows():
        print(f"  {r['name']:<26s} ${r['salary']:>6,}  {r['proj_pts_dk']:>5.1f} pts  "
              f"{r['value']:>5.2f}x  {r['team']}")

    print(f"\nTOP STACKS BY GAME TOTAL:")
    for s in find_top_stacks(players)[:6]:
        print(f"  [{s['team']}] {s['game']} (O/U {s['game_total']}) -- "
              f"{', '.join(s['players'][:3])}  Combined: {s['proj']:.1f} pts")


# ── GPP strategy note ────────────────────────────────────────────────────────
def print_strategy():
    print("\n" + "="*70)
    print("GPP STRATEGY -- $25K Sharpshooter [20 Entry Max]")
    print("="*70)
    lines = [
        "",
        "Contest profile:",
        "  - Large field: ~9,908 entrants",
        "  - 20 max entries = we enter ALL 20 (maximize EV at $3/entry)",
        "  - Total investment: $60",
        "  - Need top ~15-20% to cash; top ~0.01% for major prize",
        "",
        "Key strategic principles applied:",
        "  1. STACK high-total games (NYK@DEN 230, POR@HOU 229) -- correlated upside",
        "  2. TARGET low-ownership plays -- win by being different from the field",
        "  3. FADE chalk (Jokic in every lineup is suboptimal in GPP)",
        "     -> Include Jokic in ~50-60% of lineups max",
        "  4. SEEK injury beneficiaries -- Tatum (0 avg/$8K) is likely OUT",
        "     -> BOS players (Brown, White, Vucevic, Pritchard) get usage boost",
        "  5. LATE SWAP -- check lineups at tip time for last-minute scratches",
        "  6. DIVERSIFY -- min 2 different players between each lineup pair",
        "",
    ]
    for line in lines:
        print(line)


# ── Live injury status (ESPN public API) ──────────────────────────────────────
def scrape_espn_injuries() -> dict:
    """
    Fetch current NBA injury/status report from ESPN's public API.
    Returns {display_name: status_str} e.g. {"LeBron James": "OUT"}.
    Falls back to {} on any error (never crashes lineup generation).
    """
    try:
        import requests
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return {}
        raw = resp.json()
        result = {}
        for team_entry in raw.get("injuries", []):
            for inj in team_entry.get("injuries", []):
                name   = inj.get("athlete", {}).get("displayName", "")
                status = inj.get("status", "Active").upper()
                if not name:
                    continue
                if status in ("OUT", "INACTIVE", "SUSPENSION", "SUSPENDED"):
                    result[name] = "OUT"
                elif status == "DOUBTFUL":
                    result[name] = "DOUBTFUL"
                elif status == "QUESTIONABLE":
                    result[name] = "QUESTIONABLE"
                elif "DAY" in status or status == "GTD":
                    result[name] = "GTD"
                elif status == "PROBABLE":
                    result[name] = "PROBABLE"
        return result
    except Exception:
        return {}


def scrape_rotowire_starters() -> dict:
    """
    Scrape Rotowire NBA lineups page for confirmed/projected starting lineups.
    Rotowire typically has lineups 60–90 min before tip — well ahead of ESPN's API.

    HTML structure (verified 2025):
      .lineup__box            → one per game
        .lineup__abbr         → team abbreviation (2 per box = visitor @ home)
        .lineup__status.is-confirmed → present when lineup is confirmed
        .lineup__list.is-visit / .lineup__list.is-home
          li.lineup__player   → first 5 = starters, rest = bench/inactive
            a[title]          → full player name (e.g. "Jayson Tatum")
            .lineup__pos      → position

    Returns {
        "players": {full_name: "STARTER" | "BENCH"},
        "games":   [{"matchup", "has_lineup", "starter_count", "confirmed", "source"}],
        "source":  "rotowire",
    }
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        url  = "https://www.rotowire.com/basketball/nba-lineups.php"
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code != 200:
            return {"players": {}, "games": [], "source": "rotowire"}

        soup       = BeautifulSoup(resp.text, "html.parser")
        player_map : dict[str, str] = {}
        games_info : list[dict]     = []

        for box in soup.select(".lineup__box"):
            # Team abbreviations (visitor first, home second)
            abbrs   = [el.get_text(strip=True) for el in box.select(".lineup__abbr")]
            matchup = " @ ".join(abbrs[:2]) if len(abbrs) >= 2 else "Unknown"

            # Confirmed indicator
            is_confirmed = bool(box.select_one(".lineup__status.is-confirmed"))

            starter_count = 0

            for team_list in box.select(".lineup__list"):
                players_in_list = team_list.select("li.lineup__player")
                for idx, li in enumerate(players_in_list):
                    # Full name lives in the <a title="Full Name"> attribute
                    name_el = li.select_one("a")
                    if not name_el:
                        continue
                    # Prefer the title attribute (has full name); fall back to text
                    name = name_el.get("title", "").strip() or name_el.get_text(strip=True)
                    if not name:
                        continue
                    # First 5 per list = starters; rest = bench
                    role = "STARTER" if idx < 5 else "BENCH"
                    player_map[name] = role
                    if role == "STARTER":
                        starter_count += 1

            # Skip non-game boxes (ads/widgets have no valid abbreviations)
            if len(abbrs) < 2 or not abbrs[0]:
                continue
            games_info.append({
                "matchup":       matchup,
                # has_lineup = officially confirmed (Rotowire green dot)
                # projected  = Rotowire has expected starters but NOT yet confirmed
                "has_lineup":    is_confirmed,
                "projected":     (starter_count >= 5) and not is_confirmed,
                "starter_count": starter_count,
                "confirmed":     is_confirmed,
                "source":        "rotowire",
            })

        # Build a separate map of ONLY confirmed-game players.
        # Used by /api/starters so DvP shifts are never triggered
        # on projected (unconfirmed) lineups.
        confirmed_player_map: dict[str, str] = {}
        for box in soup.select(".lineup__box"):
            abbrs_c = [el.get_text(strip=True) for el in box.select(".lineup__abbr")]
            if len(abbrs_c) < 2 or not abbrs_c[0]:
                continue
            if not box.select_one(".lineup__status.is-confirmed"):
                continue
            for team_list in box.select(".lineup__list"):
                for idx, li in enumerate(team_list.select("li.lineup__player")):
                    name_el = li.select_one("a")
                    if not name_el:
                        continue
                    name = name_el.get("title", "").strip() or name_el.get_text(strip=True)
                    if name:
                        confirmed_player_map[name] = "STARTER" if idx < 5 else "BENCH"

        return {
            "players":           player_map,           # all players (confirmed + projected)
            "confirmed_players": confirmed_player_map, # only from .is-confirmed games
            "games":             games_info,
            "source":            "rotowire",
        }
    except Exception:
        return {"players": {}, "games": [], "source": "rotowire"}


def get_confirmed_starters(game_date=None) -> dict:
    """
    Multi-source starting lineup fetcher.

    Priority:
      1. Rotowire  — fastest, typically 60-90 min pre-tip, HTML scrape
      2. ESPN      — fallback if Rotowire returns nothing (API-based, slower)

    Returns same shape as scrape_espn_starters():
        {"players": {name: role}, "games": [...], "source": "rotowire"|"espn"}
    """
    rw = scrape_rotowire_starters()
    if rw["players"]:
        return rw
    # Fall back to ESPN
    espn = scrape_espn_starters(game_date)
    espn["source"] = "espn"
    # ESPN has no confirmed/projected distinction — treat all returned players as confirmed
    espn.setdefault("confirmed_players", espn.get("players", {}))
    return espn


def scrape_espn_starters(game_date=None) -> dict:
    """
    Fetch today's NBA scoreboard from ESPN and extract confirmed/projected starters.

    Strategy (two-pass):
      1. Scoreboard endpoint → get event IDs + any embedded roster.entries
      2. Per-game summary endpoint → boxscore.players[*].athletes (more reliable starter data)

    Returns {
        "players": {player_display_name: "STARTER" | "BENCH"},
        "games":   [{"matchup": str, "event_id": str, "has_lineup": bool, "starter_count": int}]
    }
    Falls back to {"players": {}, "games": []} if data unavailable.
    """
    try:
        import requests
        import time as _time
        from datetime import date as _date
        if game_date is None:
            game_date = _date.today()
        date_str = game_date.strftime("%Y%m%d")

        # Pass 1 — scoreboard: get event list + any roster.entries already available
        sb_url = (
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
            f"scoreboard?dates={date_str}"
        )
        resp = requests.get(sb_url, timeout=8)
        if resp.status_code != 200:
            return {"players": {}, "games": []}

        sb_data = resp.json()
        player_map  = {}
        games_info  = []

        for event in sb_data.get("events", []):
            event_id = event.get("id", "")
            matchup  = event.get("shortName", event.get("name", "Unknown"))

            # Collect any roster entries embedded in scoreboard response
            sb_players: dict[str, str] = {}
            for comp in event.get("competitions", []):
                for competitor in comp.get("competitors", []):
                    for entry in competitor.get("roster", {}).get("entries", []):
                        ath  = entry.get("athlete", {})
                        name = ath.get("displayName", "")
                        if name:
                            sb_players[name] = "STARTER" if entry.get("starter", False) else "BENCH"

            # Pass 2 — game summary: boxscore.players is the most reliable source
            summary_players: dict[str, str] = {}
            if event_id:
                try:
                    sum_url  = (
                        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
                        f"summary?event={event_id}"
                    )
                    sum_resp = requests.get(sum_url, timeout=8)
                    if sum_resp.status_code == 200:
                        sum_data = sum_resp.json()
                        for team_entry in sum_data.get("boxscore", {}).get("players", []):
                            for ath_entry in team_entry.get("athletes", []):
                                ath  = ath_entry.get("athlete", {})
                                name = ath.get("displayName", "")
                                if name:
                                    summary_players[name] = (
                                        "STARTER" if ath_entry.get("starter", False) else "BENCH"
                                    )
                        # Also check header > competitions > competitors > roster if boxscore empty
                        if not summary_players:
                            for hcomp in sum_data.get("header", {}).get("competitions", []):
                                for hcomp_c in hcomp.get("competitors", []):
                                    for entry in hcomp_c.get("roster", {}).get("entries", []):
                                        ath  = entry.get("athlete", {})
                                        name = ath.get("displayName", "")
                                        if name:
                                            summary_players[name] = (
                                                "STARTER" if entry.get("starter", False) else "BENCH"
                                            )
                except Exception:
                    pass
                _time.sleep(0.3)  # light rate-limit between game requests

            # Merge: prefer summary data (more up-to-date) over scoreboard data
            merged        = {**sb_players, **summary_players}
            starter_count = sum(1 for v in merged.values() if v == "STARTER")
            player_map.update(merged)
            games_info.append({
                "matchup":       matchup,
                "event_id":      event_id,
                "has_lineup":    starter_count >= 5,   # at least one team's 5 confirmed
                "starter_count": starter_count,
            })

        return {"players": player_map, "games": games_info}
    except Exception:
        return {"players": {}, "games": []}


def apply_status_updates(players: pd.DataFrame, status_map: dict) -> pd.DataFrame:
    """
    Apply a live {player_name: status} dict to the player pool.
    - OUT       → drop from pool entirely
    - DOUBTFUL  → 50% projection haircut
    - GTD       → 30% haircut
    - QUESTIONABLE → 15% haircut
    - PROBABLE / ACTIVE → no change
    Keys are matched case-insensitively; fuzzy-matched within 82% similarity.
    Returns an updated copy of the DataFrame.
    """
    from difflib import get_close_matches

    df = players.copy()
    name_lower_map = {n.lower(): n for n in df["name"].tolist()}
    haircuts = {"DOUBTFUL": 0.50, "GTD": 0.30, "QUESTIONABLE": 0.15}
    to_drop  = []

    for raw_name, status in status_map.items():
        status = status.upper()
        key    = raw_name.lower()
        if key in name_lower_map:
            matched = name_lower_map[key]
        else:
            close = get_close_matches(key, name_lower_map.keys(), n=1, cutoff=0.82)
            if not close:
                continue
            matched = name_lower_map[close[0]]

        mask = df["name"] == matched
        if status == "OUT":
            to_drop.append(matched)
        elif status in haircuts:
            cut = haircuts[status]
            df.loc[mask, "status"]       = status
            df.loc[mask, "proj_pts_dk"]  = (df.loc[mask, "proj_pts_dk"]  * (1 - cut)).round(2)
            df.loc[mask, "ceiling"]      = (df.loc[mask, "ceiling"]      * (1 - cut)).round(2)
            df.loc[mask, "floor"]        = (df.loc[mask, "floor"]        * (1 - cut)).round(2)
            df.loc[mask, "gpp_score"]    = (df.loc[mask, "gpp_score"]    * (1 - cut)).round(3)

    if to_drop:
        df = df[~df["name"].isin(to_drop)].copy().reset_index(drop=True)

    return df


# ── Contest-aware Late Swap ───────────────────────────────────────────────────
def _late_swap_score_candidate(
    candidate: pd.Series,
    current_ids: list,
    players: pd.DataFrame,
    corr_map: dict,
) -> float:
    """
    Score a replacement candidate for GPP contest-aware late swap.

    composite_score = gpp_score
                      * stack_synergy_mult    (does replacement correlate with keepers?)
                      * leverage_mult         (low ownership = differentiation)
                      * salary_efficiency     (using more cap is better)

    Positive correlation with existing lineup players is rewarded —
    we want replacements who synergize with the players we're keeping.
    """
    cand_pid  = str(candidate["player_id"])
    cand_team = candidate["team"]

    # ── Stack synergy: how correlated is this candidate with existing lineup? ──
    # Sum pairwise correlations with all keepers (non-OUT players in lineup)
    pool_pids = players["player_id"].astype(str).tolist()
    synergy   = 0.0
    n_pairs   = 0
    for keeper_pid in current_ids:
        key = (cand_pid, str(keeper_pid))
        if key in corr_map:
            synergy  += corr_map[key]
            n_pairs  += 1
    avg_synergy   = synergy / max(n_pairs, 1)
    synergy_mult  = 1.0 + max(-0.20, min(0.30, avg_synergy * 0.5))

    # ── Same-team stack bonus: if 2+ keepers from same team as candidate ───────
    keeper_teams  = []
    pid_map = players.set_index("player_id")
    for kp in current_ids:
        if kp in pid_map.index:
            keeper_teams.append(str(pid_map.loc[kp, "team"]))
    same_team_ct = keeper_teams.count(cand_team)
    # 2 teammates already = strong stack → reward; 3+ = approaching cap, slight penalty
    if same_team_ct == 2:
        synergy_mult *= 1.08
    elif same_team_ct >= 3:
        synergy_mult *= 0.97  # near team cap — be cautious

    # ── Bring-back bonus: candidate from opponent of primary stack ───────────
    # Primary stack = team with most keepers
    from collections import Counter
    team_count = Counter(keeper_teams)
    if team_count:
        primary_team = team_count.most_common(1)[0][0]
        # Find the opponent of the primary team's game
        for matchup in GAME_TOTALS:
            parts = matchup.split("@")
            if primary_team in parts:
                opponent = parts[0] if primary_team == parts[1] else parts[1]
                if cand_team == opponent and same_team_ct == 0:
                    synergy_mult *= 1.05  # bring-back bonus
                break

    # ── Leverage multiplier: reward low-ownership picks ─────────────────────
    proj_own     = float(candidate.get("proj_own", 20))
    leverage_mult = 1.0 + max(0, (30 - proj_own)) / 100  # +30% bonus at 0% own

    return float(candidate["gpp_score"]) * synergy_mult * leverage_mult


def get_locked_teams(game_date=None) -> set:
    """
    Return set of team abbreviations whose game has already started or finished.
    Uses ESPN live scoreboard API.  Locked teams' players cannot be swapped.
    Returns empty set on any error (fail-open so late swap still runs).
    """
    import requests
    from datetime import date as _date
    if game_date is None:
        game_date = _date.today()
    date_str = game_date.strftime("%Y%m%d")
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
        f"scoreboard?dates={date_str}"
    )
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return set()
        data   = resp.json()
        locked: set = set()
        for event in data.get("events", []):
            state = (
                event.get("status", {})
                     .get("type", {})
                     .get("state", "pre")
            )
            if state in ("in", "post"):
                for comp in event.get("competitions", []):
                    for competitor in comp.get("competitors", []):
                        abbr = competitor.get("team", {}).get("abbreviation", "")
                        if abbr:
                            locked.add(abbr.upper())
        return locked
    except Exception:
        return set()


def contest_mode_late_swap(
    lineups: list,
    players: pd.DataFrame,
    locked_team_abbrevs: set,
    *,
    min_score_gain_pct: float = 0.08,
) -> list:
    """
    Re-optimize unlocked slots using contest-aware scoring.

    Unlike late_swap_lineups() (which only fires for OUT players), this
    upgrades any unlocked slot where a meaningfully better contest-aware
    option exists.  A player is "locked" if their team is in
    locked_team_abbrevs (game already started or finished).

    Contest score = proj_pts_dk * (1 - field_exposure * 0.012)
                                * (1 - leader_exposure * 0.018)

    A swap is made only when new_score > old_score * (1 + min_score_gain_pct).
    """
    def _contest_score(row: pd.Series) -> float:
        proj = float(row.get("proj_pts_dk", 0))
        fe   = float(row.get("field_exposure",  0))
        le   = float(row.get("leader_exposure", 0))
        return proj * max(0.2, 1.0 - fe * 0.012) * max(0.2, 1.0 - le * 0.018)

    pid_index = players.set_index("player_id")
    updated   = []

    for lu in lineups:
        slot_assignment = dict(lu.get("slot_assignment", {}))
        current_ids     = [str(p) for p in lu["player_ids"]]

        # Collect unlocked slots (game not yet started) plus any slots whose
        # player was dropped from the pool (OUT) — those must always be replaced.
        unlocked_slots = []
        for slot, pid in slot_assignment.items():
            pid = str(pid)
            if pid not in pid_index.index:
                # Player was removed from pool (OUT) — force replacement regardless
                # of locked-team rules, otherwise the slot stays broken in the export.
                unlocked_slots.append((slot, pid, True))  # True = forced replace
                continue
            team = str(pid_index.loc[pid, "team"]).upper()
            if team not in locked_team_abbrevs:
                unlocked_slots.append((slot, pid, False))

        if not unlocked_slots:
            updated.append(lu)
            continue

        slot_assign_new = dict(slot_assignment)
        current_ids_new = list(current_ids)
        changed         = False

        for slot, old_pid, forced in unlocked_slots:
            if old_pid not in pid_index.index and not forced:
                continue
            old_row   = pid_index.loc[old_pid] if old_pid in pid_index.index else None
            old_score = _contest_score(old_row) if old_row is not None else 0.0
            threshold = old_score * (1.0 + min_score_gain_pct) if not forced else -1.0

            keepers = [p for p in current_ids_new if p != old_pid]

            other_sal = sum(
                int(pid_index.loc[p, "salary"])
                for p in keepers
                if p in pid_index.index
            )
            budget  = SALARY_CAP - other_sal
            min_sal = max(0, MIN_SALARY_USED - other_sal)

            # Team counts among keepers — enforce MAX_PER_TEAM
            team_counts: dict = {}
            for p in keepers:
                if p in pid_index.index:
                    t = str(pid_index.loc[p, "team"])
                    team_counts[t] = team_counts.get(t, 0) + 1

            # Candidates: right slot, not already in lineup,
            # not in a locked game, within salary cap, above salary floor, team cap ok
            candidates = players[
                players["eligible_slots"].apply(lambda es: slot in es) &
                ~players["player_id"].isin(current_ids_new) &
                (players["salary"] >= min_sal) &
                (players["salary"] <= budget) &
                ~players["team"].str.upper().isin(locked_team_abbrevs) &
                players["team"].apply(
                    lambda t: team_counts.get(str(t), 0) < MAX_PER_TEAM
                )
            ].copy()

            if candidates.empty:
                continue

            candidates["_cscore"] = candidates.apply(_contest_score, axis=1)
            best = candidates.nlargest(1, "_cscore").iloc[0]

            if float(best["_cscore"]) <= threshold:
                continue  # not a meaningful improvement

            new_pid = str(best["player_id"])
            idx_in  = current_ids_new.index(old_pid) if old_pid in current_ids_new else -1
            if idx_in >= 0:
                current_ids_new[idx_in] = new_pid
            slot_assign_new[slot] = new_pid
            changed = True

        if changed:
            sel = players[players["player_id"].isin(current_ids_new)]
            lu  = dict(lu)
            lu["player_ids"]      = current_ids_new
            lu["names"]           = sel["name"].tolist()
            lu["teams"]           = sel["team"].tolist()
            lu["salaries"]        = sel["salary"].tolist()
            lu["projections"]     = sel["proj_pts_dk"].round(2).tolist()
            lu["total_salary"]    = int(sel["salary"].sum())
            lu["proj_pts"]        = round(float(sel["proj_pts_dk"].sum()), 2)
            lu["ceiling"]         = round(float(sel["ceiling"].sum()), 2)
            lu["proj_own"]        = round(float(sel["proj_own"].mean()), 1)
            lu["slot_assignment"] = slot_assign_new
            lu["swapped"]         = True
            lu["swap_method"]     = "contest-upgrade"

            lev = score_lineup_leverage(lu, players)
            lu.update({
                "leverage":       lev["leverage"],
                "avg_own":        lev["avg_own"],
                "chalk_ct":       lev["chalk_ct"],
                "low_own_ct":     lev["low_own_ct"],
                "has_game_stack": lev["has_game_stack"],
            })

        updated.append(lu)

    return updated


def late_swap_lineups(
    lineups: list,
    players: pd.DataFrame,
    out_player_ids: set,
    on_off_data: dict | None = None,
) -> list:
    """
    Contest-aware late swap using mini-ILP re-optimization.

    For each lineup containing an OUT player:
      1. Lock all active (non-OUT) players from that lineup.
      2. Run a mini-ILP to find the optimal replacement(s) for the open slot(s),
         constrained by salary cap, slot eligibility, and team limits.
      3. If the mini-ILP is infeasible (salary/constraint conflict), fall back
         to contest-aware greedy scoring:
            composite = gpp_score * stack_synergy * leverage_mult
         which rewards correlated + low-ownership replacements.
      4. Apply usage absorption: boost projections of players who benefit from
         the OUT player's usage redistribution before scoring candidates.

    This ensures late swaps maximize contest-winning potential, not just
    roster-validity compliance.
    """
    # Fetch real USG% data once; fall back gracefully if API is down
    usage_data = fetch_player_usage_rates()

    # Apply on/off usage absorption for each OUT player
    # (modifies player projections to reflect the absence of OUT players)
    _oo = on_off_data or {}
    adjusted_players = players.copy()
    for out_pid in out_player_ids:
        mask = adjusted_players["player_id"].astype(str) == str(out_pid)
        if mask.any():
            out_row  = adjusted_players[mask].iloc[0]
            oo_entry = _oo.get(str(out_pid))
            adjusted_players = estimate_usage_absorption(
                out_row, adjusted_players,
                usage_data=usage_data,
                on_off_data=oo_entry,
            )

    # Build correlation map once for all swaps
    corr_map = build_player_correlation(adjusted_players)

    pid_index = adjusted_players.set_index("player_id")
    updated   = []

    for lu in lineups:
        slot_assignment = dict(lu.get("slot_assignment", {}))
        current_ids     = [str(p) for p in lu["player_ids"]]
        out_in_lineup   = [p for p in current_ids if p in out_player_ids]

        if not out_in_lineup:
            updated.append(lu)
            continue

        # ── Mini-ILP re-optimization ─────────────────────────────────────────
        # Lock all active players in this lineup; exclude all OUT players.
        # The ILP will find the optimal replacement(s) within salary/slot constraints.
        active_ids   = [p for p in current_ids if p not in out_player_ids]
        locked_ids   = active_ids  # keep every active player
        excluded_ids = list(out_player_ids)

        # Try to preserve the stack game this lineup was built for
        lineup_stack_game = lu.get("stack_game")

        ilp_result = build_lineup(
            adjusted_players,
            objective_col="gpp_score",
            prev_lineups=None,     # no diversity constraint — we're repairing
            min_unique=0,
            locked_ids=locked_ids,
            excluded_ids=excluded_ids,
            max_per_team=MAX_PER_TEAM,
            ownership_penalty=0.05,  # GPP penalty still applied to new slots
            stack_game=lineup_stack_game,
        )

        if ilp_result is not None:
            # Mini-ILP succeeded — use the full re-optimized lineup
            ilp_result["lineup_num"]  = lu["lineup_num"]
            ilp_result["stack_game"]  = lineup_stack_game
            ilp_result["swapped"]     = True
            ilp_result["swap_method"] = "mini-ILP"

            lev = score_lineup_leverage(ilp_result, adjusted_players)
            ilp_result.update({
                "leverage": lev["leverage"], "avg_own": lev["avg_own"],
                "chalk_ct": lev["chalk_ct"], "low_own_ct": lev["low_own_ct"],
                "has_game_stack": lev["has_game_stack"],
            })
            updated.append(ilp_result)
            continue

        # ── Greedy fallback: contest-aware candidate scoring ─────────────────
        # Mini-ILP infeasible (usually salary lock-in issues) — use scored greedy
        slot_assign_new = dict(slot_assignment)
        current_ids_new = list(current_ids)
        changed         = False

        for slot in _SLOT_ORDER:
            pid = str(slot_assign_new.get(slot, ""))
            if pid not in out_player_ids:
                continue

            keepers = [p for p in current_ids_new if p != pid]

            other_sal = sum(
                int(pid_index.loc[p, "salary"])
                for p in keepers
                if p in pid_index.index
            )
            budget  = SALARY_CAP - other_sal          # upper salary bound
            min_sal = max(0, MIN_SALARY_USED - other_sal)  # lower salary bound

            # Team counts among keepers — used to enforce MAX_PER_TEAM
            team_counts: dict = {}
            for p in keepers:
                if p in pid_index.index:
                    t = str(pid_index.loc[p, "team"])
                    team_counts[t] = team_counts.get(t, 0) + 1

            candidates = adjusted_players[
                adjusted_players["eligible_slots"].apply(lambda es: slot in es) &
                ~adjusted_players["player_id"].isin(current_ids_new) &
                ~adjusted_players["player_id"].isin(out_player_ids) &
                (adjusted_players["salary"] >= min_sal) &
                (adjusted_players["salary"] <= budget) &
                adjusted_players["team"].apply(
                    lambda t: team_counts.get(str(t), 0) < MAX_PER_TEAM
                )
            ].copy()

            if candidates.empty:
                continue

            # Score each candidate with contest-aware composite
            candidates["_swap_score"] = candidates.apply(
                lambda row: _late_swap_score_candidate(row, keepers, adjusted_players, corr_map),
                axis=1,
            )
            best    = candidates.nlargest(1, "_swap_score").iloc[0]
            new_pid = str(best["player_id"])

            idx_in = current_ids_new.index(pid) if pid in current_ids_new else -1
            if idx_in >= 0:
                current_ids_new[idx_in] = new_pid
            slot_assign_new[slot] = new_pid
            changed = True

        if changed:
            sel = adjusted_players[adjusted_players["player_id"].isin(current_ids_new)]
            lu  = dict(lu)
            lu["player_ids"]      = current_ids_new
            lu["names"]           = sel["name"].tolist()
            lu["teams"]           = sel["team"].tolist()
            lu["salaries"]        = sel["salary"].tolist()
            lu["projections"]     = sel["proj_pts_dk"].round(2).tolist()
            lu["total_salary"]    = int(sel["salary"].sum())
            lu["proj_pts"]        = round(float(sel["proj_pts_dk"].sum()), 2)
            lu["ceiling"]         = round(float(sel["ceiling"].sum()), 2)
            lu["proj_own"]        = round(float(sel["proj_own"].mean()), 1)
            lu["slot_assignment"] = slot_assign_new
            lu["swapped"]         = True
            lu["swap_method"]     = "greedy-contest"

            lev = score_lineup_leverage(lu, adjusted_players)
            lu.update({
                "leverage": lev["leverage"], "avg_own": lev["avg_own"],
                "chalk_ct": lev["chalk_ct"], "low_own_ct": lev["low_own_ct"],
                "has_game_stack": lev["has_game_stack"],
            })

        updated.append(lu)

    return updated


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("NBA DFS Model -- Slate Test Runner")
    print(f"Loading: {SALARY_FILE}")
    print("-" * 60)

    if not SALARY_FILE.exists():
        print(f"ERROR: {SALARY_FILE} not found")
        print("Make sure dk_slate.csv is in: e:/Projects/Sports/nba_py/")
        sys.exit(1)

    # 1. Parse
    raw = parse_salary_file(SALARY_FILE)
    print(f"Loaded {len(raw)} players from {SALARY_FILE.name}")

    # 1b. Load Fantasy Cruncher data (FC Proj, Proj Own%, Proj Mins, Floor, Ceiling)
    # Auto-discovers most recent draftkings_NBA_*.csv in contest/ folder.
    _fc_data = load_fc_data()
    if _fc_data is not None:
        raw = _merge_fc(raw, _fc_data)
        fc_matched = raw["fc_proj"].notna().sum()
        print(f"FC data loaded: {fc_matched}/{len(raw)} players matched "
              f"(FC Proj, Proj Own%, Proj Mins)")
    else:
        print("[warn] No Fantasy Cruncher CSV found in contest/ -- using model-only projections")
        print("       Upload draftkings_NBA_YYYY-MM-DD_players.csv to contest/ for better accuracy")

    # 1c. Fetch live Vegas lines (Odds API) and apply to game_total column.
    # The FC file has Spread/Total/TeamPts but may be hours old by game time.
    # Odds API catches any line movement since FC was exported.
    try:
        _vegas = fetch_vegas_lines(player_pool=raw)
        if _vegas and not any(k == "_meta" for k in _vegas):
            raw = apply_game_total_updates(raw, _vegas)
            print(f"Vegas lines updated: {len(_vegas)} games from Odds API")
            for _m, _v in sorted(_vegas.items(), key=lambda x: -x[1].get("total", 0)):
                print(f"  {_m:<24s} O/U {_v['total']:.1f}  "
                      f"home {_v['home_implied']:.1f} / away {_v['away_implied']:.1f}")
        else:
            _err = (_vegas or {}).get("_meta", {}).get("error", "key not set")
            print(f"[warn] Odds API unavailable ({_err}) -- using FC/avg-pts game totals")
    except Exception as _ve:
        print(f"[warn] Vegas lines fetch failed: {_ve}")

    # 2. Fetch live injury data (ESPN scraper -- free, no API key needed)
    injury_status_map: dict = {}
    try:
        from data.injury_scraper import InjuryScraper
        _sc = InjuryScraper()
        _records = _sc.scrape_espn()
        _sc.close()
        for rec in _records:
            injury_status_map[rec["name"]] = rec["status"]
        out_ct = sum(1 for s in injury_status_map.values() if s == "OUT")
        gtd_ct = sum(1 for s in injury_status_map.values() if s in ("GTD", "QUESTIONABLE", "DOUBTFUL"))
        print(f"Injury report loaded: {out_ct} OUT, {gtd_ct} GTD/Questionable")
    except Exception as _e:
        print(f"[warn] Injury scraper failed: {_e} -- using salary-tier DNP estimates only")

    # 3. Project
    players = build_projections(raw)
    # Remove OUT players (avg=0 AND salary low) and players already flagged
    players = players[players["proj_pts_dk"] > 0].copy()

    # Apply live injury statuses: OUT players dropped, GTD/QUESTIONABLE get projection cuts
    if injury_status_map:
        pre_inj = len(players)
        players = apply_status_updates(players, injury_status_map)
        dropped = pre_inj - len(players)
        if dropped:
            print(f"Live injury update: {dropped} OUT players removed from pool")

    # Hard filter: remove remaining high-DNP-risk players (>= 35%) from optimizer pool.
    # These players hurt more than help -- a DNP destroys an entire lineup slot.
    # Players with 12-25% risk stay in (meaningful upside if they play).
    if "dnp_risk" in players.columns:
        pre_dnp = len(players)
        players = players[players["dnp_risk"] < 0.35].copy()
        removed = pre_dnp - len(players)
        if removed:
            print(f"Removed {removed} players with salary-tier DNP risk >= 35%")

    # ── Multi-layer projected minutes filter ─────────────────────────────────
    # The old single check (fc_mins.notna() & fc_mins < 15) was blind to players
    # without FC data (NaN). AJ Johnson, DeAndre Jordan, Zach Edey, etc. all passed
    # through because they had no FC match. We now use proj_mins which is computed
    # from fc_mins → avg_pts estimate → salary proxy in build_projections().
    #
    # Three layers applied in order:
    #   Layer 1: proj_mins < 12  → HARD remove (no exceptions — these players won't play)
    #   Layer 2: proj_mins < 15  → remove unless news signal gives them a role
    #   Layer 3: proj_mins < 20 + salary < $5K → remove unless news signal (bench filter below)
    #
    # This must run AFTER the dnp_risk filter so the fc_mins downward override has fired.

    _mins_col = "proj_mins" if "proj_mins" in players.columns else "fc_mins"

    if _mins_col in players.columns:
        # Layer 1: Hard cutoff — salary proxy puts them at < 16 min → almost certainly bench depth.
        # With salary proxy (sal/9000)*35: $4K → 15.6 min, $4.1K → 15.9 min.
        # Setting threshold at 16 catches all sub-$4.1K players without FC data.
        _hard_low = players[_mins_col].notna() & (players[_mins_col] < 16)
        if _hard_low.any():
            _hl_names = players.loc[_hard_low, "name"].tolist()
            players = players[~_hard_low].copy()
            print(f"[MinsFlt L1] Removed {len(_hl_names)} players proj_mins < 16: "
                  f"{', '.join(_hl_names[:12])}{'...' if len(_hl_names) > 12 else ''}")

    print(f"After filtering: {len(players)} eligible players")

    # 3b. News intel — NBC Sports (primary) with X beat writers as fallback.
    # Extracts STARTING_REPLACEMENT / SCRATCHED / USAGE_INCREASE signals and
    # applies projection multipliers before lineup generation.
    try:
        from agents.news_intel_agent import NewsIntelAgent
        _nia = NewsIntelAgent()
        _intel = _nia.analyze(players)
        players, _excluded_pids = _nia.apply_to_players(players, _intel["impacts"])
        if _excluded_pids:
            players = players[~players["player_id"].astype(str).isin(_excluded_pids)].copy()
        _nia.close()

        _sr   = _intel.get("source_report", {})
        _xst  = _intel.get("x_stats", {})
        _sigs = _intel.get("signals", [])

        print("\n═══ NEWS INTEL DEBUG ═══════════════════════════════════════")

        # ── Per-source fetch results ──────────────────────────────────────
        print("  SOURCE FETCH RESULTS:")
        _nbc = _sr.get("nbcsports", {})
        _nbc_count = _nbc.get("count", 0)
        _nbc_err   = _nbc.get("error")
        if _nbc_err:
            print(f"  NBC Sports   : FAILED — {_nbc_err}")
            print(f"                 (page may be JS-rendered; X fallback was used)")
        else:
            print(f"  NBC Sports   : {_nbc_count} items scraped")

        _xfb = _sr.get("x_fallback", {})
        if _xfb.get("error") == "skipped — NBC active":
            print(f"  X (fallback) : skipped — NBC Sports feed active")
        elif _xst.get("enabled"):
            print(f"  X (fallback) : {_xfb.get('count', 0)} items  "
                  f"({_xst.get('tweets_retrieved', 0)} tweets, "
                  f"{_xst.get('signals_from_x', 0)} signals)")
        else:
            print(f"  X (fallback) : DISABLED — {_xst.get('error', 'no token')}")

        for _sname in ("espn", "rotowire", "fantasypros", "rotogrinders"):
            _si = _sr.get(_sname, {})
            _cnt, _err = _si.get("count", 0), _si.get("error")
            _status = f"{_cnt} items" if not _err else f"FAILED — {_err[:60]}"
            print(f"  {_sname:<13s}: {_status}")

        # ── Raw NBC Sports samples (key diagnostic) ───────────────────────
        _samples = _nbc.get("samples", [])
        if _samples:
            print(f"\n  NBC SPORTS RAW SAMPLES (first {len(_samples)}):")
            for _s in _samples:
                print(f"    • {_s[:110]}")
        elif _nbc_count == 0 and not _nbc_err:
            print("\n  NBC SPORTS: 0 items — page likely JS-rendered (BeautifulSoup")
            print("              cannot execute JavaScript). Options:")
            print("              1. Check if page structure changed (inspect HTML)")
            print("              2. Use a headless browser (playwright/selenium)")

        # ── Signals that fired ────────────────────────────────────────────
        print(f"\n  SIGNALS ({len(_sigs)} total, {len(_excluded_pids)} players excluded):")
        if _sigs:
            _pid_to_name = players.set_index("player_id")["name"].to_dict() if "player_id" in players.columns else {}
            for _s in _sigs:
                _pid  = str(_s.get("player_id", ""))
                _pname = _pid_to_name.get(_pid, _pid_to_name.get(
                    int(_pid) if _pid.isdigit() else _pid, _pid))
                _sig  = _s.get("signal", "?")
                _src  = _s.get("source", "?")
                _conf = _s.get("confidence", 0)
                _txt  = _s.get("text", "")[:80]
                print(f"    {str(_pname):<26s} {_sig:<24s} [{_src}] conf={_conf:.2f}")
                print(f"      \"{_txt}\"")
        else:
            print("    (none — no player-matched signals found)")
            print("    This means either:")
            print("    a) NBC scraped 0 items (JS rendering issue)")
            print("    b) Items scraped but no player names matched the pool")
            print("    c) Items scraped but no regex patterns matched the text")

        print("═══════════════════════════════════════════════════════════\n")

    except Exception as _e:
        import traceback
        print(f"\n[warn] News intel agent failed: {_e}")
        print(traceback.format_exc())
        _intel = {}
        _excluded_pids = []

    # 3c. Block pure bench players (salary < $5,000) with no confirmed role signal.
    # Rationale: sub-$5K players only produce value when a starter is injured and they
    # inherit the starting role. Without a confirmed STARTING_REPLACEMENT or USAGE_INCREASE
    # signal, they are dead weight (0 FPTS) that destroys a lineup slot.
    # ON/OFF stats cannot rescue them -- usage only spikes when a specific starter is OUT.
    #
    # Exception: players with FC proj_mins >= 24 (genuinely starting or heavy rotation)
    # are allowed through even without a news signal.
    _role_signal_pids = set()
    for _pid, _impacts in (_intel.get("impacts") or {}).items():
        _sig = _impacts.get("signal_type", "")
        if _sig in ("STARTING_REPLACEMENT", "USAGE_INCREASE", "CLEARED_FULLY"):
            _role_signal_pids.add(str(_pid))

    def _has_role_signal(row):
        return str(row.get("player_id", "")) in _role_signal_pids

    def _proj_mins_ok(row):
        # Requires proj_mins >= 22 to be considered a confirmed minutes player without a news signal.
        # Threshold of 22 catches the $5K-$5.7K range on salary proxy alone:
        #   $5.0K → 19.4 min (removed) | $5.7K → 22.2 min (passes) | $6K → 23.3 min (passes)
        # FC-data players are protected: Robinson ($4K, fc_mins=24) → proj_mins=24 → passes.
        # Clint Capela ($3.1K, fc_mins=28) → proj_mins=28 → passes.
        mins = row.get("proj_mins", row.get("fc_mins"))
        return pd.notna(mins) and float(mins) >= 22

    # Extended bench filter — applies to ALL players under $6,500, not just sub-$5K.
    # This catches:
    #   Sub-$5K: AJ Johnson, McCullar Jr., Wendell Moore Jr., Zach Edey, DeAndre Jordan
    #   $5K-$6.5K: Larry Nance Jr. (~$5K, proj_mins≈19), Gradey Dick (~$5.2K, proj_mins≈20)
    # Players are only removed when ALL three conditions hold:
    #   1. salary < $6,500
    #   2. proj_mins < 22 (no FC data or FC says limited minutes)
    #   3. no news signal (STARTING_REPLACEMENT/USAGE_INCREASE keeps them in)
    # $6K+ players on salary proxy give proj_mins=23.3 → pass condition 2 → not removed.
    pre_bench = len(players)
    bench_mask = players["salary"] < 6500
    no_signal  = ~players.apply(_has_role_signal, axis=1)
    no_mins_ok = ~players.apply(_proj_mins_ok, axis=1)
    drop_mask  = bench_mask & no_signal & no_mins_ok
    if drop_mask.any():
        _bench_names = players.loc[drop_mask, "name"].tolist()
        players = players[~drop_mask].copy()
        bench_dropped = pre_bench - len(players)
        print(f"[MinsFlt L3] Removed {bench_dropped} players (salary <$6.5K, proj_mins <22, "
              f"no role signal): {', '.join(_bench_names[:15])}{'...' if len(_bench_names) > 15 else ''}")
    print(f"Final pool: {len(players)} players\n")

    # NOTE: Line movement gpp_score bonus was removed after 3/13 validation.
    # Spread narrowing (e.g. HOU/NO -7→-5.5) was interpreted as NOP implied
    # total rising more, but the actual winning stacks were HOU-heavy. Root
    # cause: spread movement predicts COVERING, not DFS scoring. In NBA, a
    # narrowing spread often reflects a key player being questionable —
    # which increases REMAINING teammates' usage, not the underdog's.
    # The absolute implied total (already in game_total_factor) is the correct
    # signal. detect_line_movement() / apply_line_movement() are kept in code
    # for reference but NOT called here.

    # 4. Slate analysis
    print_slate_analysis(players)
    print_strategy()

    # 5. Generate lineups
    lineups = generate_gpp_lineups(
        players,
        n=NUM_LINEUPS,
        locked_ids=None,
        excluded_ids=None,
    )

    if not lineups:
        print("ERROR: No lineups generated. Check player pool and constraints.")
        sys.exit(1)

    # 5. Export DK upload CSV
    today_str = date.today().strftime("%Y-%m-%d")
    upload_path = OUTPUT_DIR / f"dk_upload_{today_str}.csv"
    export_dk_csv(lineups, players, upload_path)

    # 6. Save lineup details as JSON
    lineups_json_path = OUTPUT_DIR / f"lineups_{today_str}.json"
    with open(lineups_json_path, "w") as f:
        json.dump(lineups, f, indent=2)

    # 7. Save projections CSV (include DNP risk for transparency)
    proj_path = OUTPUT_DIR / f"projections_{today_str}.csv"
    proj_cols = [
        "name", "team", "primary_position", "salary",
        "avg_pts", "proj_pts_dk", "ceiling", "floor",
        "value", "proj_own", "gpp_score", "matchup", "game_total",
    ]
    for _sig in ["boom_rate", "variance_ratio", "is_volatile", "game_env_mult", "salary_gap", "on_off_boost"]:
        if _sig in players.columns:
            proj_cols.append(_sig)
    if "dnp_risk" in players.columns:
        proj_cols.append("dnp_risk")
    players[proj_cols].to_csv(proj_path, index=False)
    print(f"Projections CSV saved: {proj_path}")

    # 7b. Print DNP risk warnings for the top pool players
    if "dnp_risk" in players.columns:
        risky = players[players["dnp_risk"] >= 0.25].sort_values("dnp_risk", ascending=False)
        if not risky.empty:
            print(f"\n[!] HIGH DNP RISK PLAYERS (penalized in lineup scoring):")
            for _, r in risky.head(10).iterrows():
                print(f"  {r['name']:<28s} ${r['salary']:,}  dnp={r['dnp_risk']*100:.0f}%  avg={r['avg_pts']:.1f}pts")

    # 8. Exposure report
    print(f"\nEXPOSURE REPORT ({len(lineups)} lineups):")
    print(f"  {'Player':<28s} {'Count':>5s}  {'Exposure':>8s}")
    print("  " + "-"*45)
    exp_df = exposure_report(lineups, len(lineups))
    for _, r in exp_df.iterrows():
        bar = "|" * int(r["exposure_pct"] / 5)
        print(f"  {r['player']:<28s} {r['count']:>4d}x  {r['exposure_pct']:>6.1f}%  {bar}")

    # 9. Summary
    avg_proj = sum(lu["proj_pts"] for lu in lineups) / len(lineups)
    avg_ceil = sum(lu["ceiling"] for lu in lineups) / len(lineups)
    avg_sal  = sum(lu["total_salary"] for lu in lineups) / len(lineups)

    # Salary tier construction report
    # Target from 7-day backtest: pros avg $8K+=1.5, $5K-$7K=2.5, <$5K=1.5
    _pid_index = players.set_index("player_id")
    _tier_premium, _tier_mid, _tier_cheap = [], [], []
    for lu in lineups:
        pids = [str(p) for p in lu["player_ids"]]
        pids_in_pool = [p for p in pids if p in _pid_index.index]
        _sals = [int(_pid_index.loc[p, "salary"]) for p in pids_in_pool]
        _tier_premium.append(sum(1 for s in _sals if s >= 8000))
        _tier_mid.append(sum(1 for s in _sals if 5000 <= s <= 7000))
        _tier_cheap.append(sum(1 for s in _sals if s < 5000))

    _avg_prem = sum(_tier_premium) / len(lineups)
    _avg_mid  = sum(_tier_mid)     / len(lineups)
    _avg_chp  = sum(_tier_cheap)   / len(lineups)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(lineups)} lineups generated")
    print(f"  Avg projection:    {avg_proj:.1f} DK pts")
    print(f"  Avg ceiling:       {avg_ceil:.1f} DK pts")
    print(f"  Avg salary used:   ${avg_sal:,.0f}")
    print(f"  Total investment:  ${len(lineups) * CONTEST['entry_fee']}")
    print(f"\n  SALARY CONSTRUCTION (target from 7-day pro backtest):")
    print(f"    $8K+ (studs):      {_avg_prem:.2f} avg  [pro target: 1.57]  {'OK' if _avg_prem <= 2.0 else 'HIGH'}")
    print(f"    $5K-$7K (value):   {_avg_mid:.2f}  avg  [pro target: 2.50]  {'OK' if _avg_mid >= 2.0 else 'LOW'}")
    print(f"    <$5K (cheap):      {_avg_chp:.2f}  avg  [pro target: 1.50]  {'OK' if _avg_chp <= 2.0 else 'HIGH'}")
    print(f"  Files written:")
    print(f"    DK Upload:       {upload_path}")
    print(f"    Lineups JSON:    {lineups_json_path}")
    print(f"    Projections:     {proj_path}")
    print("\nIMPORTANT: Check injury news before locking lineups!")
    print("Late swap tip: Re-run after 6:45PM ET to catch any last-minute scratches.")
    print("="*60)

    # 10. Auto-postmortem: check if yesterday's contest results are available
    contest_dir = Path(__file__).parent.parent / "contest"
    if contest_dir.exists():
        import datetime as _dt
        # Look for the most recent contest-results CSV
        _result_files = sorted(contest_dir.glob("contest-results_*.csv"), reverse=True)
        if _result_files:
            _latest = _result_files[0]
            print(f"\n[postmortem] Found results: {_latest.name} -- running diagnostic...")
            run_postmortem(
                contest_csv=_latest,
                our_username="Sandcobra",  # DK username
            )


if __name__ == "__main__":
    main()
