"""
Player Props & Line Movement Agent.

Two responsibilities:
  1. PLAYER PROPS — fetch Vegas player O/U lines (points, rebounds, assists)
     and use them as a projection anchor alongside the ML model.

  2. LINE MOVEMENT — detect significant opening-vs-current game total shifts
     (sharp-money signal) and apply a projection multiplier to affected players.

Data source priority:
  a. The Odds API (player_points, player_rebounds, player_assists markets)
     — requires paid plan; gracefully skipped if key absent or returns 404.
  b. Synthetic props — derived from team implied total + player usage rate.
     Always available; provides a reasonable prior without any API calls.

Integration:
  Call apply_props_to_players(players, game_totals, api_key) from
  enrich_projections() or the UI pipeline. Returns an enriched DataFrame
  with two new columns:
    - props_pts_line  : Vegas O/U points line (or synthetic estimate)
    - props_blend_pts : blended projection (60% ML + 40% props anchor)
  and updated proj_pts_dk / ceiling columns.
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
ODDS_API_BASE       = "https://api.the-odds-api.com/v4"
SPORT               = "basketball_nba"
PROPS_MARKETS       = "player_points,player_rebounds,player_assists"
CACHE_TTL_SECS      = 1800          # 30-min cache for props
BLEND_WEIGHT_PROPS  = 0.40          # 40% props anchor, 60% ML model
MOVEMENT_THRESHOLD  = 2.0           # total must move ≥2 pts to trigger signal
MOVEMENT_BOOST_PER_PT = 0.012       # 1.2% projection boost per point of movement

# ── Team implied-total lookup (falls back to game_totals dict passed in) ──────
_LEAGUE_AVG_TEAM_PTS = 113.5        # 2024-25 NBA league average


class PlayerPropsAgent:
    """
    Fetches / synthesises player prop lines and detects line movement.
    """

    def __init__(self, api_key: str = "", cache_dir: Optional[Path] = None):
        self._api_key   = api_key
        self._cache_dir = cache_dir or Path("cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._props_cache:    dict = {}   # {event_id: {player_name: {pts, reb, ast}}}
        self._baseline_totals: dict = {}  # stored at upload time for movement detection

    # ── Public API ─────────────────────────────────────────────────────────────
    def apply_to_players(
        self,
        players:     pd.DataFrame,
        game_totals: dict,
        baseline_totals: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Main entry point.  Returns enriched DataFrame with props columns and
        updated projections.

        Parameters
        ----------
        players         : build_projections() output DataFrame
        game_totals     : current {"AWAY@HOME": {"total","home_implied","away_implied"}}
        baseline_totals : totals stored at CSV upload time (for movement detection).
                          Pass None to skip line-movement signal.
        """
        df = players.copy()

        # 1. Fetch props (live API or synthetic)
        props_map = self._fetch_or_synthesise(df, game_totals)

        # 2. Apply props blend to projections
        df = self._apply_props_blend(df, props_map)

        # 3. Line movement signal
        if baseline_totals:
            df = self._apply_line_movement(df, game_totals, baseline_totals)

        logger.info(
            "[props] Applied props to %d players | movement check: %s",
            len(df), "yes" if baseline_totals else "no",
        )
        return df

    def store_baseline(self, game_totals: dict) -> None:
        """Call at CSV upload time to snapshot opening lines for movement detection."""
        self._baseline_totals = dict(game_totals)
        logger.info("[props] Baseline game totals stored: %s", list(game_totals.keys()))

    # ── Props fetching ─────────────────────────────────────────────────────────
    def _fetch_or_synthesise(
        self, players: pd.DataFrame, game_totals: dict
    ) -> dict:
        """Returns {player_name_lower: {"pts": float, "reb": float, "ast": float}}."""
        if self._api_key:
            try:
                live = self._fetch_odds_api_props(game_totals)
                if live:
                    logger.info("[props] Loaded %d player props from The Odds API", len(live))
                    return live
            except Exception as exc:
                logger.warning("[props] Odds API failed (%s) — using synthetic props", exc)

        return self._synthesise_props(players, game_totals)

    def _fetch_odds_api_props(self, game_totals: dict) -> dict:
        """Fetch player props from The Odds API."""
        import urllib.request, urllib.parse

        # market key → our internal stat abbreviation
        _MARKET_TO_STAT = {
            "player_points":   "pts",
            "player_rebounds":  "reb",
            "player_assists":   "ast",
        }

        # Get today's NBA events
        url = (f"{ODDS_API_BASE}/sports/{SPORT}/events"
               f"?apiKey={self._api_key}&dateFormat=iso")
        with urllib.request.urlopen(url, timeout=10) as r:
            events = json.loads(r.read().decode())

        props_map: dict = {}
        for event in events:
            event_id = event["id"]

            props_url = (
                f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds"
                f"?apiKey={self._api_key}&regions=us"
                f"&markets={PROPS_MARKETS}&oddsFormat=american"
            )
            try:
                with urllib.request.urlopen(props_url, timeout=10) as r:
                    event_odds = json.loads(r.read().decode())
                time.sleep(0.5)  # rate limit
            except Exception:
                continue

            for bookmaker in event_odds.get("bookmakers", [])[:1]:  # first book only
                for market in bookmaker.get("markets", []):
                    mkey = market["key"]
                    stat = _MARKET_TO_STAT.get(mkey)
                    if stat is None:
                        continue
                    for outcome in market.get("outcomes", []):
                        name_l = outcome["description"].lower().strip()
                        line   = float(outcome.get("point", 0))
                        if outcome["name"] == "Over":
                            props_map.setdefault(name_l, {})[stat] = line

        return props_map

    def _synthesise_props(self, players: pd.DataFrame, game_totals: dict) -> dict:
        """
        Derive synthetic prop lines from team implied totals + salary-proxy usage.

        Method:
          pts_prop = team_implied_total * (player_proj_dk / team_proj_sum) * scale_factor
          reb_prop = 0.14 * pts_prop  (league avg rebounds per point ratio)
          ast_prop = 0.10 * pts_prop  (league avg assists per point ratio)

        This gives a reasonable anchor without any external data.
        """
        # Build team implied total lookup
        team_implied: dict[str, float] = {}
        for matchup, vals in game_totals.items():
            parts = matchup.split("@")
            if len(parts) == 2:
                away, home = parts[0].strip(), parts[1].strip()
                team_implied[away] = float(vals.get("away_implied", vals["total"] / 2))
                team_implied[home] = float(vals.get("home_implied", vals["total"] / 2))

        props_map: dict = {}

        # Group by team and compute each player's share of the implied total
        for team, grp in players.groupby("team"):
            implied = team_implied.get(team, _LEAGUE_AVG_TEAM_PTS)
            team_proj_sum = float(grp["proj_pts_dk"].sum())
            if team_proj_sum <= 0:
                continue

            for _, row in grp.iterrows():
                name_l     = str(row.get("name", "")).lower().strip()
                player_proj = float(row["proj_pts_dk"])
                share       = player_proj / team_proj_sum

                # Scale factor: DK pts ≠ real pts (DK scoring inflates by ~1.15x)
                real_pts_est = player_proj / 1.15
                pts_line     = round(implied * share * 1.05, 1)  # small premium for variance
                reb_line     = round(real_pts_est * 0.30, 1)     # approx reb/pts ratio
                ast_line     = round(real_pts_est * 0.22, 1)     # approx ast/pts ratio

                props_map[name_l] = {
                    "pts": max(pts_line, 3.5),
                    "reb": max(reb_line, 1.0),
                    "ast": max(ast_line, 0.5),
                    "source": "synthetic",
                }

        return props_map

    # ── Props blend application ────────────────────────────────────────────────
    def _apply_props_blend(self, df: pd.DataFrame, props_map: dict) -> pd.DataFrame:
        """
        Blend ML projection with props line.
        New proj = (1 - BLEND_WEIGHT_PROPS) * ML_proj + BLEND_WEIGHT_PROPS * props_dk_equiv
        """
        pts_lines = []
        for _, row in df.iterrows():
            name_l = str(row.get("name", "")).lower().strip()
            entry  = props_map.get(name_l)
            if entry is None:
                # fuzzy: last-name match
                last = name_l.split()[-1] if name_l else ""
                for k, v in props_map.items():
                    if k.endswith(last):
                        entry = v
                        break

            if entry:
                # Convert real-stat props to DK points equivalent
                pts_dk = (entry["pts"] * 1.0
                          + entry.get("reb", 0) * 1.25
                          + entry.get("ast", 0) * 1.5)
                pts_lines.append(round(pts_dk, 1))
            else:
                pts_lines.append(None)

        df["props_pts_line"] = pts_lines

        # Blend where we have a props line
        has_props = df["props_pts_line"].notna()
        ml_proj   = df.loc[has_props, "proj_pts_dk"]
        pr_proj   = df.loc[has_props, "props_pts_line"]

        blended = (
            (1 - BLEND_WEIGHT_PROPS) * ml_proj
            + BLEND_WEIGHT_PROPS      * pr_proj
        ).round(2)

        df.loc[has_props, "proj_pts_dk"]   = blended
        df.loc[has_props, "props_blend_pts"] = blended

        # Recompute ceiling and gpp_score where changed
        if has_props.any():
            df.loc[has_props, "ceiling"] = (
                df.loc[has_props, "proj_pts_dk"]
                + 1.28 * df.loc[has_props, "proj_std"]
            ).round(2)
            if "gpp_score" in df.columns:
                df.loc[has_props, "gpp_score"] = (
                    df.loc[has_props, "ceiling"] * 0.60
                    + df.loc[has_props, "proj_pts_dk"] * 0.25
                    + (1 - df.loc[has_props, "proj_own"] / 100) * 10
                ).round(3)

        props_count = int(has_props.sum())
        synthetic   = sum(1 for p in props_map.values() if p.get("source") == "synthetic")
        logger.info(
            "[props] Blended %d players | %d synthetic props | %d live props",
            props_count, synthetic, len(props_map) - synthetic,
        )
        return df

    # ── Line movement ──────────────────────────────────────────────────────────
    def _apply_line_movement(
        self,
        df: pd.DataFrame,
        current_totals: dict,
        baseline_totals: dict,
    ) -> pd.DataFrame:
        """
        Detect sharp-money line movement and apply a team-level projection
        boost/cut proportional to the magnitude of movement.

        A closing total higher than the opening total = sharp action on the over
        = faster scoring environment = boost all players in that game.
        """
        movements: dict[str, float] = {}   # matchup → delta total

        for matchup, curr in current_totals.items():
            base = baseline_totals.get(matchup)
            if base is None:
                continue
            delta = float(curr.get("total", 0)) - float(base.get("total", 0))
            if abs(delta) >= MOVEMENT_THRESHOLD:
                movements[matchup] = delta
                direction = "↑" if delta > 0 else "↓"
                logger.info(
                    "[props] Line movement: %s %s%.1f (%.1f → %.1f)",
                    matchup, direction, delta, base["total"], curr["total"],
                )

        if not movements:
            return df

        df = df.copy()
        for matchup, delta in movements.items():
            mask = df["matchup"] == matchup
            if not mask.any():
                continue
            mult = 1.0 + (delta * MOVEMENT_BOOST_PER_PT)
            mult = max(0.90, min(1.12, mult))   # cap at ±10%

            df.loc[mask, "proj_pts_dk"] = (df.loc[mask, "proj_pts_dk"] * mult).round(2)
            df.loc[mask, "ceiling"]     = (df.loc[mask, "ceiling"]     * mult).round(2)
            if "gpp_score" in df.columns:
                df.loc[mask, "gpp_score"] = (
                    df.loc[mask, "ceiling"] * 0.60
                    + df.loc[mask, "proj_pts_dk"] * 0.25
                    + (1 - df.loc[mask, "proj_own"] / 100) * 10
                ).round(3)

        return df

    # ── Positional scarcity helper (surfaced to slate_agent) ──────────────────
    @staticmethod
    def compute_positional_scarcity(players: pd.DataFrame) -> dict:
        """
        Count viable players per DK roster slot.
        'Viable' = proj_pts_dk >= 12 AND salary in a usable range.

        Returns dict: {"PG": n, "SG": n, "SF": n, "PF": n, "C": n,
                       "G": n, "F": n, "UTIL": n,
                       "scarce_slots": [...], "scarcity_score": 0-1}
        """
        _SLOT_ORDER = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        _POS_TO_SLOTS = {
            "PG": ["PG", "G", "UTIL"],
            "SG": ["SG", "G", "UTIL"],
            "SF": ["SF", "F", "UTIL"],
            "PF": ["PF", "F", "UTIL"],
            "C":  ["C", "UTIL"],
        }

        slot_counts: dict[str, int] = {}
        for slot in _SLOT_ORDER:
            # Count players eligible for this slot with viable projection
            viable_mask = players["proj_pts_dk"] >= 12
            if "eligible_slots" in players.columns:
                slot_mask = players["eligible_slots"].apply(
                    lambda s: slot in (s if isinstance(s, list) else [])
                )
                count = int((viable_mask & slot_mask).sum())
            else:
                # Fallback: use primary_position
                eligible_pos = [p for p, slots in _POS_TO_SLOTS.items() if slot in slots]
                pos_mask = players["primary_position"].isin(eligible_pos)
                count = int((viable_mask & pos_mask).sum())
            slot_counts[slot] = count

        # Scarcity: slots with fewer than 5 viable players
        scarce = [slot for slot, n in slot_counts.items() if n < 5]
        # Score 0-1 where 1 = completely scarce
        scarcity_score = round(len(scarce) / len(_SLOT_ORDER), 2)

        return {
            **slot_counts,
            "scarce_slots":    scarce,
            "scarcity_score":  scarcity_score,
        }
