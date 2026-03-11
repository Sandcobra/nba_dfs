"""Shared utility functions."""

import re
import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())


def team_abbrev_to_full(abbrev: str) -> str:
    """Convert 3-letter team abbreviation to full team name."""
    MAP = {
        "ATL": "Atlanta Hawks",     "BOS": "Boston Celtics",
        "BKN": "Brooklyn Nets",     "CHA": "Charlotte Hornets",
        "CHI": "Chicago Bulls",     "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks",  "DEN": "Denver Nuggets",
        "DET": "Detroit Pistons",   "GSW": "Golden State Warriors",
        "HOU": "Houston Rockets",   "IND": "Indiana Pacers",
        "LAC": "LA Clippers",       "LAL": "Los Angeles Lakers",
        "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
        "MIL": "Milwaukee Bucks",   "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
        "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
        "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
        "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
        "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors",
        "UTA": "Utah Jazz",         "WAS": "Washington Wizards",
    }
    return MAP.get(abbrev.upper(), abbrev)


def compute_dk_fantasy_pts(row: dict) -> float:
    """Compute DraftKings fantasy points from a box score row."""
    pts  = row.get("pts", row.get("PTS", 0)) or 0
    reb  = row.get("reb", row.get("REB", 0)) or 0
    ast  = row.get("ast", row.get("AST", 0)) or 0
    stl  = row.get("stl", row.get("STL", 0)) or 0
    blk  = row.get("blk", row.get("BLK", 0)) or 0
    tov  = row.get("tov", row.get("TOV", 0)) or 0
    fg3m = row.get("fg3m", row.get("FG3M", 0)) or 0

    base = pts * 1.0 + fg3m * 0.5 + reb * 1.25 + ast * 1.5 + stl * 2.0 + blk * 2.0 - tov * 0.5

    cats = {
        "pts": pts >= 10,
        "reb": reb >= 10,
        "ast": ast >= 10,
        "stl": stl >= 10,
        "blk": blk >= 10,
    }
    dd_cnt = sum(cats.values())
    bonus  = 1.5 if dd_cnt >= 2 else 0
    bonus += 3.0 if dd_cnt >= 3 else 0
    return round(base + bonus, 2)


def compute_fd_fantasy_pts(row: dict) -> float:
    pts  = row.get("pts", row.get("PTS", 0)) or 0
    reb  = row.get("reb", row.get("REB", 0)) or 0
    ast  = row.get("ast", row.get("AST", 0)) or 0
    stl  = row.get("stl", row.get("STL", 0)) or 0
    blk  = row.get("blk", row.get("BLK", 0)) or 0
    tov  = row.get("tov", row.get("TOV", 0)) or 0
    return round(pts * 1.0 + reb * 1.2 + ast * 1.5 + stl * 2.0 + blk * 2.0 - tov * 1.0, 2)


def rolling_stats(df: pd.DataFrame, cols: list, windows: list = [5, 10, 20]) -> pd.DataFrame:
    """Add rolling average columns for given stat columns."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for w in windows:
            out[f"{col}_L{w}"] = out[col].rolling(w, min_periods=1).mean()
    return out


def salary_to_value(salary: int, projected_pts: float) -> float:
    """DK value = projected pts / (salary / 1000)."""
    return round(projected_pts / (salary / 1000), 2) if salary > 0 else 0.0


def cache_key(*args) -> str:
    """Generate a stable cache key from arguments."""
    return hashlib.md5(str(args).encode()).hexdigest()[:12]


def fmt_salary(salary: int) -> str:
    return f"${salary:,}"


def pct(val: float, decimals: int = 1) -> str:
    return f"{val:.{decimals}f}%"


def lineup_to_display_str(lineup: dict) -> str:
    """Format a lineup dict as a readable string."""
    lines = [
        f"Lineup #{lineup.get('lineup_num', '?')} | "
        f"Proj: {lineup.get('proj_pts', 0):.1f} | "
        f"Salary: ${lineup.get('total_salary', 0):,} | "
        f"Own: {lineup.get('proj_ownership', 0):.1f}%"
    ]
    for name, pos, sal, proj in zip(
        lineup.get("player_names", []),
        lineup.get("positions", []),
        lineup.get("salaries", []),
        lineup.get("projections", []),
    ):
        lines.append(f"  {pos:4s} {name:<22s} ${sal:,}  {proj:.1f}")
    return "\n".join(lines)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


class RollingWindow:
    """Compute rolling statistics for a stream of game values."""
    def __init__(self, maxsize: int = 20):
        from collections import deque
        self._q = deque(maxlen=maxsize)

    def push(self, val: float):
        self._q.append(val)

    def mean(self) -> float:
        return float(np.mean(self._q)) if self._q else 0.0

    def std(self) -> float:
        return float(np.std(self._q)) if len(self._q) > 1 else 0.0

    def max(self) -> float:
        return float(max(self._q)) if self._q else 0.0

    def last_n(self, n: int) -> list:
        return list(self._q)[-n:]
