"""
Central configuration for the NBA DFS Model.
Set API keys and preferences via environment variables or a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_DIR       = BASE_DIR / "cache"
LINEUPS_DIR     = BASE_DIR / "lineups"
LOGS_DIR        = BASE_DIR / "logs"
DB_PATH         = BASE_DIR / "nba_dfs.db"

for _d in (CACHE_DIR, LINEUPS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── API Keys (set in .env) ─────────────────────────────────────────────────────
SPORTRADAR_API_KEY      = os.getenv("SPORTRADAR_API_KEY", "")
ROTOWIRE_API_KEY        = os.getenv("ROTOWIRE_API_KEY", "")
THE_ODDS_API_KEY        = os.getenv("THE_ODDS_API_KEY", "")     # Vegas lines (free tier available)
SPORTSLINE_API_KEY      = os.getenv("SPORTSLINE_API_KEY", "")
RG_PROJECTIONS_KEY      = os.getenv("RG_PROJECTIONS_KEY", "")   # RotoGrinders projections
ANTHROPIC_API_KEY       = os.getenv("ANTHROPIC_API_KEY", "")
X_BEARER_TOKEN          = os.getenv("X_BEARER_TOKEN", "")   # X / Twitter API v2 bearer token

# Beat writer X handles to monitor for injury/lineup intel.
# Add or remove handles here — or override via X_BEAT_WRITERS env var
# (comma-separated list of handles without @).
_x_env = os.getenv("X_BEAT_WRITERS", "")
X_BEAT_WRITER_HANDLES: list[str] = (
    [h.strip().lstrip("@") for h in _x_env.split(",") if h.strip()]
    if _x_env
    else [
        # National injury/lineup insiders
        "ShamsCharania",
        "ChrisBHaynes",
        "wojespn",
        "NBAInsider",
        # Team beat writers — add your list here
        # "sacbee_stern",       # Kings beat
        # "tribjazz",           # Jazz beat
        # "AlexKendall_NBA",    # etc.
    ]
)

# ── DFS Site Settings ──────────────────────────────────────────────────────────
DRAFTKINGS_SALARY_CAP   = 50_000
FANDUEL_SALARY_CAP      = 60_000

# DraftKings DFS scoring weights
DK_SCORING = {
    "PTS":          1.0,
    "FG3M":         0.5,
    "REB":          1.25,
    "AST":          1.5,
    "STL":          2.0,
    "BLK":          2.0,
    "TOV":         -0.5,
    "DOUBLE_DOUBLE": 1.5,   # bonus
    "TRIPLE_DOUBLE": 3.0,   # bonus (stacks on top of DD bonus)
}

# FanDuel DFS scoring weights
FD_SCORING = {
    "PTS":          1.0,
    "REB":          1.2,
    "AST":          1.5,
    "STL":          2.0,
    "BLK":          2.0,
    "TOV":         -1.0,
}

# DraftKings roster configuration (GPP main slate)
DK_ROSTER = {
    "PG":   1,
    "SG":   1,
    "SF":   1,
    "PF":   1,
    "C":    1,
    "G":    1,   # PG or SG
    "F":    1,   # SF or PF
    "UTIL": 1,   # any
}
DK_ROSTER_SIZE = 8

# FanDuel roster configuration
FD_ROSTER = {
    "PG": 2,
    "SG": 2,
    "SF": 2,
    "PF": 2,
    "C":  1,
}
FD_ROSTER_SIZE = 9

# ── Model Settings ─────────────────────────────────────────────────────────────
CURRENT_SEASON          = "2025-26"
MONTE_CARLO_SIMS        = 10_000
PROJECTION_LOOKBACK_GAMES = 20      # rolling window for recent form
OWNERSHIP_LOOKBACK      = 30        # days for ownership history
MIN_MINUTES_THRESHOLD   = 10.0      # ignore players < N avg minutes

# Ensemble model weights (must sum to 1.0)
ENSEMBLE_WEIGHTS = {
    "xgboost":       0.25,
    "lightgbm":      0.25,
    "neural_net":    0.20,
    "random_forest": 0.15,
    "bayesian_ridge":0.10,
    "catboost":      0.05,
}

# ── Optimizer Settings ─────────────────────────────────────────────────────────
NUM_LINEUPS_GPP          = 150   # number of GPP lineups to generate
NUM_LINEUPS_CASH         = 20    # number of cash game lineups
MAX_EXPOSURE_GPP         = 0.55  # max player exposure in GPP (55%)
MAX_EXPOSURE_CASH        = 0.80  # max player exposure in cash
MAX_PLAYERS_PER_TEAM_GPP = 4    # correlation stacking rule
MIN_SALARY_USED          = 49_000  # DraftKings - don't leave money on table
OWNERSHIP_PENALTY_WEIGHT = 0.03   # penalize high-ownership in GPP

# ── NBA API Headers ────────────────────────────────────────────────────────────
NBA_API_HEADERS = {
    "Connection":      "keep-alive",
    "Accept":          "application/json, text/plain, */*",
    "x-nba-stats-token": "true",
    "User-Agent":      (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "Sec-Fetch-Site":  "same-origin",
    "Sec-Fetch-Mode":  "cors",
    "Referer":         "https://stats.nba.com/",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Rate Limiting ──────────────────────────────────────────────────────────────
NBA_API_DELAY_SECS      = 0.6    # delay between NBA API calls
SCRAPER_DELAY_SECS      = 1.5   # delay between scraping calls
MAX_RETRIES             = 4
RETRY_BACKOFF_FACTOR    = 2.0

# ── Monitoring ─────────────────────────────────────────────────────────────────
LINEUP_LOCK_MINUTES_BEFORE_TIP  = 60   # alert if change < 60 min to tip
MONITOR_POLL_INTERVAL_SECS      = 120  # check for news every 2 minutes
