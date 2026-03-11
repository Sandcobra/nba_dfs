"""
Referee Crew Agent.

Fetches tonight's referee crew assignments and applies foul-rate adjustments
to player projections. Refs who call more fouls create more free-throw
opportunities, which inflates scoring — especially for guards who attack the
rim and players who draw fouls at a high rate.

Data sources (in priority order):
  1. Scrape https://official.nba.com/referee-assignments/ (public JSON endpoint)
  2. Cached historical ref foul-rate tiers (hardcoded from career data)

Adjustment logic:
  - Each ref crew is assigned a foul tier (HIGH / MEDIUM / LOW) based on
    average personal fouls called per 48 min across their career.
  - HIGH foul crew  → +4% boost to players with high historical FT rate
  - LOW foul crew   → −3% reduction to the same players
  - MEDIUM / unknown → no adjustment

Integration:
  Call apply_ref_adjustments(players, date_str) → enriched DataFrame.
  A new column "ref_adjustment" is added for UI display.
"""

import json
import logging
import re
import time
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Referee historical foul-tier lookup ───────────────────────────────────────
# Tier based on career personal fouls called per game (source: basketball-reference)
# HIGH  = typically calls >25 fouls/game total, HI foul rate
# LOW   = typically calls <21 fouls/game total
# MEDIUM = 21-25 (most refs fall here)
#
# This lookup covers active refs as of 2024-25. New refs default to MEDIUM.
REF_TIERS: dict[str, str] = {
    # HIGH foul callers
    "john goble":         "HIGH",
    "bill kennedy":       "HIGH",
    "james capers":       "HIGH",
    "tony brothers":      "HIGH",
    "marc davis":         "HIGH",
    "scott foster":       "HIGH",
    "eric lewis":         "HIGH",
    "ken mauer":          "HIGH",
    "bennie adams":       "HIGH",
    # LOW foul callers
    "derek richardson":   "LOW",
    "tyler ford":         "LOW",
    "tom washington":     "LOW",
    "sean corbin":        "LOW",
    "ed malloy":          "LOW",
    "kevin scott":        "LOW",
    "cheryl burnett":     "LOW",
    # MEDIUM (many refs; listed for completeness but default handles rest)
    "zach zarba":         "MEDIUM",
    "david guthrie":      "MEDIUM",
    "pat fraher":         "MEDIUM",
    "j.t. fields":        "MEDIUM",
    "nick buchert":       "MEDIUM",
    "courtney kirkland":  "MEDIUM",
    "matt boland":        "MEDIUM",
    "josh tiven":         "MEDIUM",
    "brendan randle":     "MEDIUM",
    "brian forte":        "MEDIUM",
}

# Foul-rate adjustment multipliers per tier
TIER_MULT: dict[str, float] = {
    "HIGH":   1.04,   # +4% for high-FT-rate players
    "MEDIUM": 1.00,
    "LOW":    0.97,   # -3% for high-FT-rate players
}

# Players with FT rate proxy: salary >= $7000 guards or wing scorers benefit most
# We use primary_position + salary as a proxy for "attacks the rim"
_GUARD_POSITIONS = {"PG", "SG", "G"}
_HIGH_FT_MIN_SALARY = 5500   # minimum salary to receive ref adjustment


class RefAgent:
    """
    Fetches referee crew assignments and adjusts player projections for
    tonight's foul-calling environment.
    """

    _ASSIGNMENTS_URL = "https://official.nba.com/referee-assignments/"

    def __init__(self, cache_ttl_secs: int = 3600):
        self._cache_ttl = cache_ttl_secs
        self._last_fetch: float = 0.0
        self._cached_crews: dict = {}   # {matchup_key: [ref_name, ...]}

    # ── Public API ─────────────────────────────────────────────────────────────
    def apply_ref_adjustments(
        self,
        players: pd.DataFrame,
        date_str: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch tonight's ref assignments and apply foul-rate projection adjustments.

        Returns enriched DataFrame with 'ref_crew' and 'ref_adjustment' columns.
        """
        today = date_str or date.today().isoformat()
        crews = self._fetch_crews(today)

        if not crews:
            logger.info("[refs] No crew data available — skipping ref adjustments")
            players = players.copy()
            players["ref_crew"]       = ""
            players["ref_adjustment"] = 1.0
            return players

        return self._apply_adjustments(players.copy(), crews)

    def get_crew_summary(self, date_str: Optional[str] = None) -> list[dict]:
        """Return list of {matchup, refs, tier, adjustment} for UI display."""
        today = date_str or date.today().isoformat()
        crews = self._fetch_crews(today)
        summary = []
        for matchup, refs in crews.items():
            tier = self._crew_tier(refs)
            summary.append({
                "matchup":    matchup,
                "refs":       refs,
                "tier":       tier,
                "adjustment": TIER_MULT[tier],
            })
        return summary

    # ── Crew fetching ──────────────────────────────────────────────────────────
    def _fetch_crews(self, date_str: str) -> dict:
        """
        Attempt to scrape NBA.com referee assignment page.
        Returns {matchup_str: [ref_name_lower, ...]} or {}.
        Falls back to empty dict (no adjustments applied) on failure.
        """
        import time as _time
        now = _time.time()
        if self._cached_crews and (now - self._last_fetch) < self._cache_ttl:
            return self._cached_crews

        try:
            result = self._scrape_nba_refs(date_str)
            if result:
                self._cached_crews = result
                self._last_fetch   = now
                logger.info("[refs] Fetched crews for %d games on %s", len(result), date_str)
                return result
        except Exception as exc:
            logger.warning("[refs] Scrape failed: %s — no ref adjustments", exc)

        return {}

    def _scrape_nba_refs(self, date_str: str) -> dict:
        """
        Scrape NBA.com referee assignments page.
        The page contains a JSON blob with today's assignments.
        """
        import urllib.request

        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"),
            "Accept": "text/html,application/xhtml+xml",
        }
        req = urllib.request.Request(self._ASSIGNMENTS_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            html = r.read().decode("utf-8", errors="ignore")

        # Extract JSON embedded in the page (NBA.com pattern)
        result: dict = {}
        # Look for patterns like "referee":["Name1","Name2","Name3"]
        # and "home":"TEAM" / "visitor":"TEAM" nearby
        game_blocks = re.findall(
            r'"visitor"\s*:\s*"([^"]+)".*?"home"\s*:\s*"([^"]+)".*?"officials"\s*:\s*(\[[^\]]+\])',
            html, re.DOTALL
        )
        for away, home, refs_json in game_blocks:
            try:
                refs = [r.lower().strip() for r in json.loads(refs_json)]
                matchup = f"{away.upper()}@{home.upper()}"
                result[matchup] = refs
            except Exception:
                continue

        # Alternative pattern: look for table rows with ref names
        if not result:
            rows = re.findall(
                r'<tr[^>]*>.*?</tr>', html, re.DOTALL
            )
            current_game = None
            for row in rows:
                game_m = re.search(r'([A-Z]{2,3})\s+@\s+([A-Z]{2,3})', row)
                if game_m:
                    current_game = f"{game_m.group(1)}@{game_m.group(2)}"
                    result.setdefault(current_game, [])
                ref_m = re.findall(r'<td[^>]*>([A-Z][a-z]+ [A-Z][a-z]+)</td>', row)
                if ref_m and current_game:
                    result[current_game].extend([n.lower() for n in ref_m])

        return result

    # ── Adjustment application ─────────────────────────────────────────────────
    def _apply_adjustments(self, df: pd.DataFrame, crews: dict) -> pd.DataFrame:
        """Apply per-game foul-rate multiplier to eligible players."""
        df["ref_crew"]       = ""
        df["ref_adjustment"] = 1.0

        for matchup, refs in crews.items():
            tier = self._crew_tier(refs)
            mult = TIER_MULT[tier]

            # Match players to this matchup
            mask = df["matchup"] == matchup
            if not mask.any():
                # Try partial match (team abbreviations may differ)
                parts = matchup.split("@")
                if len(parts) == 2:
                    mask = (
                        df["team"].str.upper().isin([parts[0].strip(), parts[1].strip()])
                    )

            if not mask.any():
                continue

            # Tag crew and tier
            crew_str = " / ".join(refs[:3]) + f" [{tier}]"
            df.loc[mask, "ref_crew"] = crew_str
            df.loc[mask, "ref_adjustment"] = mult

            if mult == 1.0:
                continue

            # Apply adjustment only to players who benefit from foul calls:
            # guards + wings with salary >= threshold (likely attack the rim)
            eligible = (
                mask
                & (df["salary"] >= _HIGH_FT_MIN_SALARY)
                & (df["primary_position"].isin(_GUARD_POSITIONS | {"SF", "F"}))
            )

            if not eligible.any():
                # If no guards/wings, apply a softer adjustment to everyone
                soft_mult = 1.0 + (mult - 1.0) * 0.4
                eligible = mask
                df.loc[eligible, "proj_pts_dk"] = (
                    df.loc[eligible, "proj_pts_dk"] * soft_mult
                ).round(2)
                df.loc[eligible, "ceiling"] = (
                    df.loc[eligible, "ceiling"] * soft_mult
                ).round(2)
            else:
                df.loc[eligible, "proj_pts_dk"] = (
                    df.loc[eligible, "proj_pts_dk"] * mult
                ).round(2)
                df.loc[eligible, "ceiling"] = (
                    df.loc[eligible, "ceiling"] * mult
                ).round(2)

            # Recompute gpp_score for adjusted players
            if "gpp_score" in df.columns:
                adj_mask = eligible if eligible.any() else mask
                df.loc[adj_mask, "gpp_score"] = (
                    df.loc[adj_mask, "ceiling"] * 0.60
                    + df.loc[adj_mask, "proj_pts_dk"] * 0.25
                    + (1 - df.loc[adj_mask, "proj_own"] / 100) * 10
                ).round(3)

            logger.info(
                "[refs] %s → crew tier %s (x%.3f) applied to %d players",
                matchup, tier, mult, int(eligible.sum()),
            )

        return df

    # ── Tier classification ────────────────────────────────────────────────────
    def _crew_tier(self, refs: list[str]) -> str:
        """
        Classify the crew's collective foul-calling tendency.
        Majority vote across the three refs; ties go to MEDIUM.
        """
        tier_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for ref in refs:
            tier = REF_TIERS.get(ref.lower().strip(), "MEDIUM")
            tier_counts[tier] += 1

        if tier_counts["HIGH"] >= 2:
            return "HIGH"
        if tier_counts["LOW"] >= 2:
            return "LOW"
        return "MEDIUM"
