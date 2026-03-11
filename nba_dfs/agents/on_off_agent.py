"""
On/Off Split Agent (Lineup-Level).

Uses PlayerDashboardByLineups to compute per-minute DK scoring deltas
for each teammate when a **combined set** of OUT players are simultaneously
absent from the court.

Why lineup-level instead of game-level
---------------------------------------
Game-level: player misses ~10 games/season → 10 data points maximum.
            Combined absence of 3 players: maybe 2-3 games → below threshold.

Lineup-level: lineups shuffle every few minutes.  When LeBron sits, there
              are ~15 min/game × 67 games = ~1 000 minutes of "off" data.
              Combined absence occurs every game during rest rotations → 50+
              minutes available even for the tightest combinations.

Method
-------
For each teammate (e.g. Kleber):
  1. Fetch PlayerDashboardByLineups (Per-36 mode) for the teammate.
     Each row = one specific 5-man lineup + player's per-36 stats in it.
  2. Label rows:
       WITH    : all OUT players appear in GROUP_VALUE
       WITHOUT : none of the OUT players appear in GROUP_VALUE
  3. Compute minutes-weighted avg DK per-36 for each group.
  4. delta_dk = dk_without − dk_with
     Positive  = teammate scores more per minute when OUT players are absent.
     Negative  = teammate depends on OUT players' playmaking.

Reliability
-----------
Requires MIN_MINUTES_WITHOUT = 50 total minutes in the WITHOUT split.
Falls back to None → caller uses usage-quantity estimate.

Cache
-----
Results cached at cache/on_off/{cache_key}_{season}.json, 24-hour TTL.
cache_key = sorted out-player names joined by "__".

Integration
-----------
  on_off_data = OnOffAgent(client).compute(out_player_rows, teammate_rows)
  Pass on_off_data into estimate_usage_absorption(on_off_data=on_off_data).
  Format: {player_id_str: {"delta_dk": float, "dk_with": float,
                            "dk_without": float, "min_with": float,
                            "min_without": float}}
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_MINUTES_WITHOUT = 50    # min total minutes in "without" split to trust delta
MIN_MINUTES_WITH    = 30    # min total minutes in "with" split for valid baseline
MAX_DELTA_CAP       = 12.0  # cap delta at ±12 DK pts to prevent outlier blow-ups
CACHE_TTL_HOURS     = 24
CURRENT_SEASON      = "2025-26"

# DK NBA Classic scoring applied to Per-36 stats → DK pts per 36 min
_DK_SCORING = {
    "PTS": 1.0, "FG3M": 0.5, "REB": 1.25,
    "AST": 1.5, "STL": 2.0,  "BLK": 2.0, "TOV": -0.5,
}


class OnOffAgent:
    """
    Computes per-minute DK deltas for teammates relative to a combined set of
    OUT players, using 5-man lineup data instead of game-level presence.
    """

    def __init__(self, nba_client, season: str = CURRENT_SEASON,
                 cache_dir: Optional[Path] = None):
        self._client    = nba_client
        self._season    = season
        self._cache_dir = (cache_dir or Path("cache")) / "on_off"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────
    def compute(
        self,
        out_players: list,          # list of pd.Series rows (ALL players who are OUT)
        teammates:   pd.DataFrame,
    ) -> dict[str, dict]:
        """
        Compute per-36-minute DK deltas for all teammates when ALL out_players
        are simultaneously absent from the court.

        Parameters
        ----------
        out_players : list of player rows (pd.Series) for every OUT player
                      on this team.  Pass all OUT players at once so the
                      WITHOUT split = lineups where none of them are present.
        teammates   : remaining active players on the same team

        Returns
        -------
        {player_id_str: {"delta_dk", "dk_with", "dk_without",
                          "min_with", "min_without"}}
        Only players with sufficient minute samples are included.
        """
        if not out_players:
            return {}

        # Build lower-case name set for GROUP_VALUE matching
        out_names: set[str] = set()
        for out_p in out_players:
            name = str(out_p.get("name", "")).strip()
            if name:
                out_names.add(name.lower())

        if not out_names:
            return {}

        # Cache keyed on sorted out-player names
        cache_key = "__".join(sorted(out_names)).replace(" ", "_")
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("[on_off] Cache hit for OUT set: %s", sorted(out_names))
            return cached

        label = "+".join(n.split()[-1].title() for n in sorted(out_names))
        logger.info("[on_off] Computing lineup on/off for combined OUT: %s", label)

        result: dict[str, dict] = {}
        n_id_miss = 0
        n_threshold_miss = 0

        for _, tm_row in teammates.iterrows():
            tm_name = str(tm_row.get("name", "")).strip()
            tm_pid  = str(tm_row.get("player_id", ""))
            nba_id  = self._resolve_nba_id(tm_name)
            if nba_id is None:
                logger.warning("[on_off] Could not resolve NBA ID for '%s' — skipping", tm_name)
                n_id_miss += 1
                continue

            split = self._compute_split_lineup(nba_id, out_names)
            if split is None:
                logger.debug(
                    "[on_off] %s (id=%s): insufficient minutes for lineup split — skipping",
                    tm_name, nba_id,
                )
                n_threshold_miss += 1
                continue

            result[tm_pid] = split
            logger.info(
                "[on_off] %s | WITH %s: %.1f DK/36 (%d min) | "
                "WITHOUT: %.1f DK/36 (%d min) | Δ=%.1f",
                tm_name, label,
                split["dk_with"],    split["min_with"],
                split["dk_without"], split["min_without"],
                split["delta_dk"],
            )

        logger.info(
            "[on_off] %s → %d splits | %d ID misses | %d below minute threshold",
            label, len(result), n_id_miss, n_threshold_miss,
        )

        self._save_cache(cache_key, result)
        return result

    # ── Core computation ───────────────────────────────────────────────────────
    def _compute_split_lineup(
        self,
        nba_id:    int,
        out_names: set[str],    # lower-case display names of ALL OUT players
    ) -> Optional[dict]:
        """
        Fetch the teammate's PlayerDashboardByLineups, split rows into
        WITH / WITHOUT groups, and compute minutes-weighted DK per-36.

        WITH    = every OUT player appears in GROUP_VALUE (they're all on court)
        WITHOUT = none of the OUT players appear in GROUP_VALUE
        """
        df = self._fetch_lineup_dashboard(nba_id)
        if df is None or df.empty:
            return None

        df = df.copy()
        df.columns = [c.upper() for c in df.columns]

        required = {"GROUP_VALUE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV"}
        if not required.issubset(df.columns):
            logger.debug(
                "[on_off] Missing columns for player %s: %s",
                nba_id, required - set(df.columns),
            )
            return None

        df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0)

        n_out = len(out_names)

        # Count how many OUT players appear in each lineup row's GROUP_VALUE
        df["_n_out_present"] = df["GROUP_VALUE"].apply(
            lambda gv: sum(
                1 for name in out_names
                if self._name_in_group(name, str(gv))
            )
        )

        with_mask    = df["_n_out_present"] == n_out   # all OUT players on court
        without_mask = df["_n_out_present"] == 0        # no OUT players on court

        min_with    = float(df.loc[with_mask,    "MIN"].sum())
        min_without = float(df.loc[without_mask, "MIN"].sum())

        if min_with < MIN_MINUTES_WITH or min_without < MIN_MINUTES_WITHOUT:
            logger.debug(
                "[on_off] Insufficient minutes for %s: with=%.0f, without=%.0f",
                nba_id, min_with, min_without,
            )
            return None

        dk_with    = self._weighted_dk_per36(df[with_mask])
        dk_without = self._weighted_dk_per36(df[without_mask])
        delta      = float(np.clip(dk_without - dk_with, -MAX_DELTA_CAP, MAX_DELTA_CAP))

        return {
            "delta_dk":    round(delta, 2),
            "dk_with":     round(dk_with, 2),
            "dk_without":  round(dk_without, 2),
            "min_with":    round(min_with, 1),
            "min_without": round(min_without, 1),
        }

    def _weighted_dk_per36(self, df: pd.DataFrame) -> float:
        """
        Minutes-weighted average DK pts per 36 min across lineup rows.
        Stats are already Per-36 (from the Per36 API request).
        """
        total_min = float(df["MIN"].sum())
        if total_min <= 0:
            return 0.0

        dk_total = 0.0
        for _, row in df.iterrows():
            dk_row = sum(
                float(row.get(col, 0)) * mult
                for col, mult in _DK_SCORING.items()
                if col in df.columns
            )
            dk_total += dk_row * (float(row["MIN"]) / total_min)

        return round(dk_total, 2)

    def _name_in_group(self, name_lower: str, group_value: str) -> bool:
        """
        Check if a player name appears in a lineup GROUP_VALUE string.

        GROUP_VALUE format: "LeBron James - Anthony Davis - Austin Reaves - ..."
        Tries full name first, then last-name-only to handle abbreviated
        first names (e.g. "L. James" vs "LeBron James").
        """
        gv = group_value.lower()

        if name_lower in gv:
            return True

        # Last-name fallback
        parts = name_lower.split()
        if parts:
            last = parts[-1]
            if len(last) > 3 and last in gv:
                return True

        return False

    # ── NBA API helpers ────────────────────────────────────────────────────────
    def _resolve_nba_id(self, name: str) -> Optional[int]:
        try:
            return self._client.player_name_to_id(name)
        except Exception:
            return None

    def _fetch_lineup_dashboard(self, player_id: int) -> Optional[pd.DataFrame]:
        """Fetch PlayerDashboardByLineups (Per-36, full 5-man lineups)."""
        try:
            df = self._client.get_player_lineup_dashboard(
                player_id=player_id,
                season=self._season,
                group_quantity=5,
            )
            return df if not df.empty else None
        except Exception as exc:
            logger.warning(
                "[on_off] Lineup dashboard fetch failed for player_id=%s: %s", player_id, exc
            )
            return None

    # ── Cache ──────────────────────────────────────────────────────────────────
    def _cache_path(self, cache_key: str) -> Path:
        safe = cache_key[:120]   # keep filenames reasonable
        return self._cache_dir / f"{safe}_{self._season.replace('-', '_')}.json"

    def _load_cache(self, cache_key: str) -> Optional[dict]:
        p = self._cache_path(cache_key)
        if not p.exists():
            return None
        age_hours = (time.time() - p.stat().st_mtime) / 3600
        if age_hours > CACHE_TTL_HOURS:
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _save_cache(self, cache_key: str, data: dict) -> None:
        try:
            self._cache_path(cache_key).write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.debug("[on_off] Cache write failed: %s", exc)
