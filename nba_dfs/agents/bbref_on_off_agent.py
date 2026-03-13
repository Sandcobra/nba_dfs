"""
ESPN On/Off Agent (game-log based).

stats.nba.com is IP-blocked; Basketball Reference returns 403.
ESPN's public APIs are accessible, return real box scores, and include
team schedules (so we can derive games a player MISSED this season).

Method
------
1. Get team schedule to find all completed game dates.
2. For each confirmed-OUT player, get their game log (only played games).
   Missed dates = all team game dates − player's played dates.
3. For each teammate, get their game log and split:
     WITH    = games where OUT player was present (not in missed set)
     WITHOUT = games where OUT player was absent (in missed set)
4. Compute per-36 DK pts for each split.
   delta_dk = dk_without − dk_with

Minimum thresholds
------------------
  MIN_GAMES_WITHOUT = 5   (need ≥5 games without the OUT player)
  MIN_GAMES_WITH    = 10  (need ≥10 games with the OUT player)

Cache (cache/espn/)
-------------------
  roster_{team}_{year}.json      — 7-day TTL
  schedule_{team}_{year}.json    — 12-hour TTL
  gamelog_{espn_id}_{year}.json  — 24-hour TTL
  on_off_{key}_{year}.json       — 24-hour TTL
"""

import io
import json
import logging
import re
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DELAY_SECS         = 1.0    # between ESPN HTTP requests (much gentler than BBRef)
ROSTER_TTL_DAYS    = 7
SCHEDULE_TTL_HRS   = 12
GAMELOG_TTL_HRS    = 24
ONOFF_TTL_HRS      = 24
SEASON_YEAR        = 2026   # end year of 2025-26 season
MIN_GAMES_WITHOUT  = 5
MIN_GAMES_WITH     = 10
MAX_DELTA_CAP      = 12.0

# ESPN team IDs keyed by DK abbreviation
ESPN_TEAM_IDS: dict[str, int] = {
    "ATL": 1,  "BOS": 2,  "NOP": 3,  "CHI": 4,  "CLE": 5,
    "DAL": 6,  "DEN": 7,  "DET": 8,  "GSW": 9,  "HOU": 10,
    "IND": 11, "LAC": 12, "LAL": 13, "MIA": 14, "MIL": 15,
    "MIN": 16, "BKN": 17, "NYK": 18, "ORL": 19, "PHI": 20,
    "PHX": 21, "POR": 22, "SAC": 23, "SAS": 24, "OKC": 25,
    "UTA": 26, "WAS": 27, "TOR": 28, "MEM": 29, "CHA": 30,
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_ESPN_BASE = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
_ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


class BBRefOnOffAgent:
    """
    Computes per-36 DK deltas for teammates using ESPN game log data.
    Named BBRefOnOffAgent for backward compatibility with existing imports.
    Same external interface as the old OnOffAgent.compute().
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        season_year: int = SEASON_YEAR,
    ):
        self._cache_dir = (cache_dir or Path("cache")) / "espn"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._year = season_year
        self._roster_mem:   dict[str, dict[str, str]] = {}  # team → {name_lower: espn_id}
        self._schedule_mem: dict[str, set[str]] = {}         # team → set of completed dates
        self._gamelog_mem:  dict[str, pd.DataFrame] = {}
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ── Public API ─────────────────────────────────────────────────────────────
    def compute(
        self,
        out_players: list,          # list of pd.Series rows for OUT players
        teammates:   pd.DataFrame,
    ) -> dict[str, dict]:
        """
        Returns {player_id_str: {delta_dk, dk_with, dk_without, min_with, min_without}}
        min_with / min_without here represent game counts, not minutes.
        Only players meeting the MIN_GAMES thresholds are included.
        """
        if not out_players:
            return {}

        out_names_sorted = sorted(
            str(p.get("name", "")).lower().strip()
            for p in out_players
            if str(p.get("name", "")).strip()
        )
        if not out_names_sorted:
            return {}

        cache_key = "__".join(n.replace(" ", "_") for n in out_names_sorted)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("[espn] Cache hit: %s", cache_key)
            return cached

        label = "+".join(n.split()[-1].title() for n in out_names_sorted)
        logger.info("[espn] Computing splits for OUT set: %s", label)

        # Fetch team roster (name → espn_id) for the OUT player's team
        team_dk = str(out_players[0].get("team", ""))
        espn_team_id = ESPN_TEAM_IDS.get(team_dk.upper())
        if espn_team_id is None:
            logger.warning("[espn] Unknown team '%s' — cannot compute splits", team_dk)
            return {}

        roster = self._get_roster(espn_team_id, team_dk)
        if not roster:
            logger.warning("[espn] No roster data for %s — cannot compute splits", team_dk)
            return {}

        # Get all completed team game dates this season
        team_game_dates = self._get_schedule(espn_team_id, team_dk)
        if not team_game_dates:
            logger.warning("[espn] No schedule data for %s", team_dk)
            return {}

        # Find dates each OUT player missed
        out_missed: set[str] = set()
        for out_p in out_players:
            name = str(out_p.get("name", "")).strip()
            espn_id = self._resolve_id(name, roster)
            if espn_id is None:
                logger.warning("[espn] OUT player '%s' not found in %s roster", name, team_dk)
                continue
            played_dates = self._get_played_dates(espn_id)
            missed = team_game_dates - played_dates
            logger.info("[espn] %s missed %d of %d team games", name, len(missed), len(team_game_dates))
            out_missed |= missed

        if not out_missed:
            logger.warning(
                "[espn] %s — no missed games found; "
                "player may not have missed any games this season. "
                "Cannot compute WITHOUT split.", label
            )
            return {}

        result: dict[str, dict] = {}
        n_no_id  = 0
        n_thresh = 0

        for _, tm_row in teammates.iterrows():
            tm_name = str(tm_row.get("name", "")).strip()
            tm_pid  = str(tm_row.get("player_id", ""))

            espn_id = self._resolve_id(tm_name, roster)
            if espn_id is None:
                logger.debug("[espn] Teammate '%s' not in roster map", tm_name)
                n_no_id += 1
                continue

            split = self._compute_game_split(espn_id, out_missed)
            if split is None:
                n_thresh += 1
                continue

            result[tm_pid] = split
            logger.info(
                "[espn] %s | WITH %s: %.1f DK/36 (%d g) | "
                "WITHOUT: %.1f DK/36 (%d g) | delta=%.2f",
                tm_name, label,
                split["dk_with"],    split["min_with"],
                split["dk_without"], split["min_without"],
                split["delta_dk"],
            )

        logger.info(
            "[espn] %s -> %d splits computed | %d ID misses | %d below threshold",
            label, len(result), n_no_id, n_thresh,
        )
        self._save_cache(cache_key, result)
        return result

    # ── Core computation ───────────────────────────────────────────────────────
    def _compute_game_split(
        self,
        espn_id:    str,
        out_missed: set[str],
    ) -> Optional[dict]:
        gl = self._get_game_log(espn_id)
        if gl is None or gl.empty:
            return None

        without = gl[gl["date"].isin(out_missed)]
        with_   = gl[~gl["date"].isin(out_missed)]

        if len(without) < MIN_GAMES_WITHOUT or len(with_) < MIN_GAMES_WITH:
            logger.debug(
                "[espn] %s: with=%d g, without=%d g — below threshold (%d/%d)",
                espn_id, len(with_), len(without),
                MIN_GAMES_WITH, MIN_GAMES_WITHOUT,
            )
            return None

        dk_with    = self._avg_dk_per36(with_)
        dk_without = self._avg_dk_per36(without)
        delta      = float(np.clip(dk_without - dk_with, -MAX_DELTA_CAP, MAX_DELTA_CAP))

        return {
            "delta_dk":    round(delta, 2),
            "dk_with":     round(dk_with, 2),
            "dk_without":  round(dk_without, 2),
            "min_with":    int(len(with_)),
            "min_without": int(len(without)),
        }

    @staticmethod
    def _avg_dk_per36(games: pd.DataFrame) -> float:
        """Average per-36-min DK pts across games, weighted by minutes played."""
        scores  = []
        weights = []
        for _, row in games.iterrows():
            mp = float(row.get("mp", 0) or 0)
            if mp < 1:
                continue
            pts = float(row.get("pts", 0) or 0)
            fg3 = float(row.get("fg3", 0) or 0)
            trb = float(row.get("trb", 0) or 0)
            ast = float(row.get("ast", 0) or 0)
            stl = float(row.get("stl", 0) or 0)
            blk = float(row.get("blk", 0) or 0)
            tov = float(row.get("tov", 0) or 0)

            dk = pts*1.0 + fg3*0.5 + trb*1.25 + ast*1.5 + stl*2.0 + blk*2.0 + tov*(-0.5)

            cats = sum([pts >= 10, trb >= 10, ast >= 10, stl >= 10, blk >= 10])
            if cats >= 2:
                dk += 1.5
            if cats >= 3:
                dk += 3.0

            scores.append(dk * 36.0 / mp)
            weights.append(mp)

        if not scores:
            return 0.0
        return round(float(np.average(scores, weights=np.array(weights))), 2)

    # ── ESPN fetching ──────────────────────────────────────────────────────────
    def _get_roster(self, espn_team_id: int, team_dk: str) -> dict[str, str]:
        """Fetch ESPN team roster → {player_name_lower: espn_id_str}."""
        key = team_dk.upper()
        if key in self._roster_mem:
            return self._roster_mem[key]

        p = self._cache_dir / f"roster_{key}_{self._year}.json"
        if p.exists() and (time.time() - p.stat().st_mtime) < ROSTER_TTL_DAYS * 86400:
            data = json.loads(p.read_text())
            self._roster_mem[key] = data
            return data

        url = f"{_ESPN_SITE}/teams/{espn_team_id}/roster"
        logger.info("[espn] Fetching roster %s (team %d)", key, espn_team_id)
        time.sleep(DELAY_SECS)

        try:
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            roster: dict[str, str] = {}
            for athlete in data.get("athletes", []):
                name     = str(athlete.get("fullName", "")).strip().lower()
                espn_id  = str(athlete.get("id", "")).strip()
                if name and espn_id:
                    roster[name] = espn_id

            logger.info("[espn] %s roster: %d players", key, len(roster))
            p.write_text(json.dumps(roster, indent=2))
            self._roster_mem[key] = roster
            return roster

        except Exception as exc:
            logger.warning("[espn] Roster fetch failed for %s: %s", key, exc)
            return {}

    def _get_schedule(self, espn_team_id: int, team_dk: str) -> set[str]:
        """Return set of completed game date strings (YYYY-MM-DD) for this team."""
        key = team_dk.upper()
        if key in self._schedule_mem:
            return self._schedule_mem[key]

        p = self._cache_dir / f"schedule_{key}_{self._year}.json"
        if p.exists() and (time.time() - p.stat().st_mtime) < SCHEDULE_TTL_HRS * 3600:
            data = json.loads(p.read_text())
            result = set(data)
            self._schedule_mem[key] = result
            return result

        url = f"{_ESPN_SITE}/teams/{espn_team_id}/schedule?season={self._year}"
        logger.info("[espn] Fetching schedule %s (team %d)", key, espn_team_id)
        time.sleep(DELAY_SECS)

        try:
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            game_dates: set[str] = set()
            for ev in data.get("events", []):
                comps = ev.get("competitions", [{}])
                status_name = comps[0].get("status", {}).get("type", {}).get("name", "")
                if status_name == "STATUS_FINAL":
                    date_str = ev.get("date", "")[:10]
                    if date_str:
                        game_dates.add(date_str)

            logger.info("[espn] %s schedule: %d completed games", key, len(game_dates))
            p.write_text(json.dumps(sorted(game_dates)))
            self._schedule_mem[key] = game_dates
            return game_dates

        except Exception as exc:
            logger.warning("[espn] Schedule fetch failed for %s: %s", key, exc)
            return set()

    def _get_played_dates(self, espn_id: str) -> set[str]:
        """Return dates where this player appeared in the box score."""
        gl = self._get_game_log(espn_id)
        if gl is None or gl.empty:
            return set()
        return set(gl["date"].dropna().tolist())

    def _get_game_log(self, espn_id: str) -> Optional[pd.DataFrame]:
        """Fetch full season game log for a player (only games played)."""
        if espn_id in self._gamelog_mem:
            return self._gamelog_mem[espn_id]

        p = self._cache_dir / f"gamelog_{espn_id}_{self._year}.json"
        if p.exists() and (time.time() - p.stat().st_mtime) < GAMELOG_TTL_HRS * 3600:
            df = pd.read_json(io.StringIO(p.read_text()))
            self._gamelog_mem[espn_id] = df
            return df

        url = f"{_ESPN_BASE}/athletes/{espn_id}/gamelog?season={self._year}"
        logger.info("[espn] Fetching game log %s", espn_id)
        time.sleep(DELAY_SECS)

        try:
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            labels     = data.get("labels", [])
            events_meta = data.get("events", {})
            season_types = data.get("seasonTypes", [])

            idx = {label: i for i, label in enumerate(labels)}

            def _stat(stats: list, label: str) -> float:
                i = idx.get(label)
                if i is None or i >= len(stats):
                    return 0.0
                raw = stats[i]
                # Handle "X-Y" format (e.g. "3-7" for 3PT)
                if isinstance(raw, str) and "-" in raw:
                    raw = raw.split("-")[0]
                try:
                    return float(raw)
                except (ValueError, TypeError):
                    return 0.0

            rows = []
            for st in season_types:
                for cat in st.get("categories", []):
                    for ev in cat.get("events", []):
                        eid   = ev.get("eventId", "")
                        meta  = events_meta.get(eid, {})
                        date  = meta.get("gameDate", "")[:10]
                        stats = ev.get("stats", [])
                        if not date or not stats:
                            continue
                        mp = _stat(stats, "MIN")
                        if mp < 1:
                            continue
                        rows.append({
                            "date": date,
                            "mp":   mp,
                            "pts":  _stat(stats, "PTS"),
                            "trb":  _stat(stats, "REB"),
                            "ast":  _stat(stats, "AST"),
                            "stl":  _stat(stats, "STL"),
                            "blk":  _stat(stats, "BLK"),
                            "tov":  _stat(stats, "TO"),
                            "fg3":  _stat(stats, "3PT"),
                            "fga":  _stat(stats, "FGA"),
                            "fta":  _stat(stats, "FTA"),
                        })

            if not rows:
                logger.warning("[espn] Empty game log for %s", espn_id)
                return None

            df = pd.DataFrame(rows)
            logger.info("[espn] Game log %s: %d games", espn_id, len(df))
            p.write_text(df.to_json())
            self._gamelog_mem[espn_id] = df
            return df

        except Exception as exc:
            logger.warning("[espn] Game log fetch failed for %s: %s", espn_id, exc)
            return None

    # ── Name resolution ────────────────────────────────────────────────────────
    def _resolve_id(self, name: str, roster: dict[str, str]) -> Optional[str]:
        key = name.lower().strip()

        # 1. Exact
        if key in roster:
            return roster[key]

        # 2. Strip punctuation  (P.J. -> PJ, Jr. -> Jr)
        stripped     = re.sub(r"[.\-']", "", key).strip()
        roster_strip = {re.sub(r"[.\-']", "", k).strip(): v for k, v in roster.items()}
        if stripped in roster_strip:
            return roster_strip[stripped]

        # 3. Unique last-name match
        parts = key.split()
        if parts:
            last       = parts[-1]
            candidates = {k: v for k, v in roster.items() if k.split()[-1] == last}
            if len(candidates) == 1:
                return next(iter(candidates.values()))

        # 4. Fuzzy
        close = get_close_matches(key, roster.keys(), n=1, cutoff=0.80)
        if close:
            logger.debug("[espn] Fuzzy match '%s' -> '%s'", key, close[0])
            return roster[close[0]]

        return None

    # ── Team usage rates ───────────────────────────────────────────────────────
    def get_team_usage_rates(self, team_dk: str) -> dict[str, dict]:
        """
        Compute ESPN-based USG% for every player on a team from their game logs.

        USG% formula (no play-by-play needed):
            poss_used = FGA + 0.44*FTA + TOV   (per game, last 15 games)
            USG% = poss_used / (pace_per_48 * MP/48) * 100
        where pace_per_48 ≈ 100 (league-average NBA pace).

        Returns {name_lower: {usg_pct, poss_pg, min_pg, espn_id}}
        """
        espn_team_id = ESPN_TEAM_IDS.get(team_dk.upper())
        if espn_team_id is None:
            return {}

        roster = self._get_roster(espn_team_id, team_dk)
        if not roster:
            return {}

        result: dict[str, dict] = {}
        PACE = 100.0  # league-average possessions per 48 minutes

        for name_lower, espn_id in roster.items():
            gl = self._get_game_log(espn_id)
            if gl is None or gl.empty:
                continue

            recent = gl.tail(15)  # weight toward recent form
            mp_avg  = float(recent["mp"].clip(lower=0).mean())
            if mp_avg < 1.0:
                continue

            fga_avg = float(recent.get("fga", 0).fillna(0).mean()) if "fga" in recent.columns else 0.0
            fta_avg = float(recent.get("fta", 0).fillna(0).mean()) if "fta" in recent.columns else 0.0
            tov_avg = float(recent["tov"].fillna(0).mean())

            # Fall back to pts-based proxy if FGA/FTA missing from cache
            if fga_avg == 0:
                pts_avg = float(recent["pts"].fillna(0).mean())
                fga_avg = pts_avg / 2.0  # rough: ~2 pts per attempt

            poss_pg = fga_avg + 0.44 * fta_avg + tov_avg
            # USG% = poss_pg / (PACE * mp_avg/48) * 100
            usg_pct = (poss_pg / (PACE * mp_avg / 48.0)) * 100.0
            usg_pct = float(np.clip(usg_pct, 2.0, 45.0))

            result[name_lower] = {
                "usg_pct":  round(usg_pct, 1),
                "poss_pg":  round(poss_pg, 2),
                "min_pg":   round(mp_avg, 1),
                "espn_id":  espn_id,
                "team":     team_dk.upper(),
            }

        return result

    def get_starting_lineup_usage(
        self,
        team_dk: str,
        starters: list[str],          # player name strings (display names)
        n_games_together: int = 0,     # if known; 0 = estimate
    ) -> dict:
        """
        Rotowire-style starting lineup usage breakdown.

        Returns:
        {
            "team": "CLE",
            "starters": [
                {"name": "Evan Mobley",     "usg_pct": 35.2},
                {"name": "Donovan Mitchell", "usg_pct": 24.1},
                ...
            ],
            "minutes_together": <int>,   # estimated or provided
        }

        Usage shares are normalised so they sum to 100%.
        """
        team_usage = self.get_team_usage_rates(team_dk)

        lineup_stats: list[dict] = []
        for display_name in starters:
            key = display_name.lower().strip()
            entry = team_usage.get(key)
            if entry is None:
                # fuzzy lookup
                close = get_close_matches(key, team_usage.keys(), n=1, cutoff=0.75)
                if close:
                    entry = team_usage[close[0]]
            if entry is None:
                entry = {"usg_pct": 15.0, "min_pg": 25.0}  # league-avg fallback
            lineup_stats.append({
                "name":    display_name,
                "usg_pct": entry["usg_pct"],
                "min_pg":  entry.get("min_pg", 25.0),
            })

        total_usg = sum(s["usg_pct"] for s in lineup_stats) or 1.0
        for s in lineup_stats:
            s["usg_pct"] = round(s["usg_pct"] / total_usg * 100.0, 1)

        lineup_stats.sort(key=lambda x: -x["usg_pct"])

        # Estimate minutes played together = min of all starters' min_pg
        if n_games_together == 0:
            min_pgs = [s["min_pg"] for s in lineup_stats]
            n_games_together = int(min(min_pgs)) if min_pgs else 25

        return {
            "team":             team_dk.upper(),
            "starters":         lineup_stats,
            "minutes_together": n_games_together,
        }

    # ── Cache helpers ──────────────────────────────────────────────────────────
    def _load_cache(self, key: str) -> Optional[dict]:
        p = self._cache_dir / f"on_off_{key[:100]}_{self._year}.json"
        if not p.exists():
            return None
        if (time.time() - p.stat().st_mtime) > ONOFF_TTL_HRS * 3600:
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _save_cache(self, key: str, data: dict) -> None:
        try:
            p = self._cache_dir / f"on_off_{key[:100]}_{self._year}.json"
            p.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.debug("[espn] Cache write failed: %s", exc)
