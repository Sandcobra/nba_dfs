"""
Slate Construction Agent.

Uses Claude AI to analyze the nightly slate environment and recommend
optimal DFS lineup construction parameters. Synthesizes:
  - Salary distribution characteristics
  - Vegas game totals and implied team scores
  - Injury/status news (stars out → role player elevation)
  - Slate size and game count
  - Ownership concentration landscape
  - Historical contest result feedback (optional)

Returns a SlateProfile dict that drives adaptive parameters in
generate_gpp_lineups() and build_lineup() instead of hard-coded constants.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


# ── Default fallback profile (used when API is unavailable) ───────────────────
_DEFAULT_PROFILE = {
    "barbell_enabled":             True,
    "stud_threshold":              9000,
    "stud_min":                    2,
    "cheap_threshold":             4500,
    "cheap_min":                   3,
    "max_exposure_pct":            0.33,
    "min_ownership_penalty":       0.03,
    "max_ownership_penalty":       0.07,
    "role_player_std_mult":        0.42,
    "role_player_proj_threshold":  25,
    "role_player_sal_threshold":   6500,
    "stack_emphasis":              "medium",
    "slate_type":                  "main",
    "slate_size":                  5,
    "injury_chaos_level":          "low",
    "rationale":                   "Default profile — API unavailable.",
    "key_opportunities":           [],
    "key_risks":                   [],
    "source":                      "default",
}


class SlateConstructionAgent:
    """
    Analyzes the nightly DFS slate environment with Claude AI and returns
    adaptive construction parameters for the ILP lineup optimizer.
    """

    # Salary tier label boundaries used in prompt context
    _STUD_FLOOR   = 9000
    _MID_FLOOR    = 6000
    _VALUE_CEIL   = 5000

    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._client  = None
        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("[SlateAgent] anthropic package not installed — "
                               "pip install anthropic to enable AI slate analysis.")
            except Exception as exc:
                logger.warning("[SlateAgent] Failed to init Anthropic client: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────
    def analyze(
        self,
        players:         pd.DataFrame,
        game_totals:     dict,
        injuries:        Optional[list] = None,
        contest_history: Optional[list] = None,
    ) -> dict:
        """
        Analyze tonight's slate and return a SlateProfile dict.

        Parameters
        ----------
        players         : DataFrame from load_players() — must have salary,
                          proj_pts_dk, proj_own, status, team, matchup columns.
        game_totals     : {"AWAY@HOME": {"total", "home_implied", "away_implied"}}
        injuries        : List of injury dicts [{"player", "team", "status", "news"}]
        contest_history : Optional list of recent contest result dicts for context.

        Returns
        -------
        SlateProfile dict with optimizer construction parameters + rationale.
        """
        slate_stats = self._compute_slate_stats(players, game_totals, injuries or [])

        if self._client is None:
            logger.info("[SlateAgent] No API client — returning rule-based profile.")
            return self._rule_based_profile(slate_stats)

        try:
            prompt   = self._build_prompt(slate_stats, contest_history or [])
            raw      = self._call_claude(prompt)
            profile  = self._parse_response(raw, slate_stats)
            profile["source"] = "claude-ai"
            return profile
        except Exception as exc:
            logger.error("[SlateAgent] Claude analysis failed: %s — falling back to rules.", exc)
            profile = self._rule_based_profile(slate_stats)
            profile["source"] = "rule-based-fallback"
            return profile

    # ── Slate statistics computation ───────────────────────────────────────────
    def _compute_slate_stats(
        self,
        players:     pd.DataFrame,
        game_totals: dict,
        injuries:    list,
    ) -> dict:
        """Compute structured statistics that describe the slate environment."""
        sal  = players["salary"].dropna()
        proj = players["proj_pts_dk"].dropna()
        own  = players["proj_own"].dropna() if "proj_own" in players.columns else pd.Series([20.0])

        # Salary tier counts
        studs     = (sal >= 9000).sum()
        hi_mid    = ((sal >= 7000) & (sal < 9000)).sum()
        lo_mid    = ((sal >= 5500) & (sal < 7000)).sum()
        value     = ((sal >= 4500) & (sal < 5500)).sum()
        cheap     = (sal < 4500).sum()

        # Salary distribution shape
        sal_p25, sal_p50, sal_p75, sal_p90 = (
            float(sal.quantile(0.25)),
            float(sal.quantile(0.50)),
            float(sal.quantile(0.75)),
            float(sal.quantile(0.90)),
        )
        # "Compression" = gap between 75th and 25th pct relative to mean
        sal_compression = (sal_p75 - sal_p25) / float(sal.mean()) if sal.mean() > 0 else 0

        # Game environment
        game_count   = len(game_totals)
        totals       = [v["total"] for v in game_totals.values()]
        avg_total    = float(np.mean(totals)) if totals else 220.0
        max_total    = float(max(totals))     if totals else 220.0
        min_total    = float(min(totals))     if totals else 220.0
        high_total_games = sum(1 for t in totals if t >= 228)
        blowout_risk_games = sum(1 for v in game_totals.values()
                                 if abs(v.get("home_implied", 0) - v.get("away_implied", 0)) > 12)

        # Injury environment
        out_players  = [p for p in injuries if str(p.get("status", "")).upper() in ("OUT", "DNP")]
        gtd_players  = [p for p in injuries if str(p.get("status", "")).upper() in
                        ("GTD", "QUESTIONABLE", "PROBABLE")]
        # Star injuries = players with salary >= 8000 who are OUT
        star_out_names = [
            p["player"] for p in out_players
            if players[players["name"].str.lower() == str(p.get("player", "")).lower()]["salary"].max() >= 8000
        ] if "name" in players.columns else []

        # Ownership landscape
        chalk_count = (own >= 25).sum()  # >25% projected ownership
        chalk_pct   = float(chalk_count / max(len(own), 1))
        avg_own     = float(own.mean())

        # Value tier depth (cheap plays available with real projections)
        cheap_with_proj = players[
            (players["salary"] < 4500) & (players["proj_pts_dk"] >= 15)
        ]

        # Salary compression — how spread out is the talent?
        stud_proj_avg   = float(
            players[players["salary"] >= 9000]["proj_pts_dk"].mean()
        ) if studs > 0 else 0.0
        value_proj_avg  = float(
            players[players["salary"] < 5000]["proj_pts_dk"].mean()
        ) if cheap + value > 0 else 0.0

        return {
            # Pool composition
            "total_players":     len(players),
            "game_count":        game_count,
            "slate_type":        "showdown" if game_count == 1
                                 else "small" if game_count <= 3
                                 else "main",

            # Salary tiers
            "studs_9k_plus":     int(studs),
            "hi_mid_7k_9k":      int(hi_mid),
            "lo_mid_5500_7k":    int(lo_mid),
            "value_4500_5500":   int(value),
            "cheap_sub_4500":    int(cheap),
            "cheap_viable_ct":   int(len(cheap_with_proj)),  # cheap plays with proj >= 15

            # Salary distribution
            "sal_p25":           sal_p25,
            "sal_p50":           sal_p50,
            "sal_p75":           sal_p75,
            "sal_p90":           sal_p90,
            "sal_compression":   round(sal_compression, 3),

            # Game environment
            "avg_game_total":    round(avg_total, 1),
            "max_game_total":    round(max_total, 1),
            "min_game_total":    round(min_total, 1),
            "high_total_games":  high_total_games,
            "blowout_risk_games":blowout_risk_games,
            "game_totals":       game_totals,

            # Injuries
            "out_count":         len(out_players),
            "gtd_count":         len(gtd_players),
            "star_outs":         star_out_names,

            # Ownership
            "chalk_player_count":int(chalk_count),
            "chalk_pct":         round(chalk_pct, 3),
            "avg_proj_own":      round(avg_own, 1),

            # Value
            "stud_proj_avg":     round(stud_proj_avg, 1),
            "value_proj_avg":    round(value_proj_avg, 1),

            # Positional scarcity (computed inline for prompt context)
            "scarce_slots":      self._get_scarce_slots(players),
        }

    @staticmethod
    def _get_scarce_slots(players: pd.DataFrame) -> list[str]:
        """Quick positional scarcity check for prompt context."""
        slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        pos_to_slots = {
            "PG": ["PG", "G", "UTIL"], "SG": ["SG", "G", "UTIL"],
            "SF": ["SF", "F", "UTIL"], "PF": ["PF", "F", "UTIL"],
            "C":  ["C", "UTIL"],
        }
        scarce = []
        for slot in slots:
            eligible_pos = [p for p, sl in pos_to_slots.items() if slot in sl]
            count = len(players[
                (players["proj_pts_dk"] >= 12)
                & players["primary_position"].isin(eligible_pos)
            ])
            if count < 5:
                scarce.append(slot)
        return scarce

    # ── Prompt construction ────────────────────────────────────────────────────
    def _build_prompt(self, stats: dict, contest_history: list) -> str:
        gt_lines = "\n".join(
            f"  {matchup}: O/U {v['total']}, home {v.get('home_implied','?')}, away {v.get('away_implied','?')}"
            for matchup, v in stats.get("game_totals", {}).items()
        )
        history_block = ""
        if contest_history:
            history_block = "\n## Recent Contest Results (last 5 slates):\n"
            for r in contest_history[-5:]:
                history_block += (
                    f"  Date: {r.get('date','?')} | "
                    f"Cashed: {r.get('cashed',0)}/{r.get('entries',0)} | "
                    f"Top score: {r.get('top_score','?')} | "
                    f"Notes: {r.get('notes','')}\n"
                )

        return f"""You are an expert NBA DFS lineup construction strategist. Analyze tonight's slate and return optimal lineup construction parameters as JSON.

## Tonight's Slate Statistics:
- Slate type: {stats['slate_type']} ({stats['game_count']} games, {stats['total_players']} players)
- Salary tiers: $9k+ studs={stats['studs_9k_plus']}, $7k-$9k hi-mid={stats['hi_mid_7k_9k']}, $5.5k-$7k lo-mid={stats['lo_mid_5500_7k']}, $4.5k-$5.5k value={stats['value_4500_5500']}, <$4.5k cheap={stats['cheap_sub_4500']} (viable cheap with proj ≥ 15pts: {stats['cheap_viable_ct']})
- Salary percentiles: p25=${stats['sal_p25']:,.0f}, p50=${stats['sal_p50']:,.0f}, p75=${stats['sal_p75']:,.0f}, p90=${stats['sal_p90']:,.0f}
- Salary compression score: {stats['sal_compression']} (lower = more compressed/equal talent distribution)
- Avg stud projection: {stats['stud_proj_avg']} DK pts | Avg value-tier projection: {stats['value_proj_avg']} DK pts

## Game Environment:
{gt_lines}
- High-total games (≥228): {stats['high_total_games']}
- Blowout-risk games (spread >12): {stats['blowout_risk_games']}

## Positional Scarcity:
- Scarce slots (<5 viable players): {', '.join(stats.get('scarce_slots', [])) or 'None'}
- Note: Scarce positions force ownership concentration — the field has no choice but to play the same players there.

## Injury / Status Situation:
- OUT players: {stats['out_count']}
- GTD/Questionable players: {stats['gtd_count']}
- Star-level (salary ≥$8k) OUT: {', '.join(stats['star_outs']) if stats['star_outs'] else 'None'}

## Ownership Landscape:
- Players projected >25% owned: {stats['chalk_player_count']}
- Chalk concentration: {stats['chalk_pct']*100:.1f}% of pool is chalk
- Average projected ownership: {stats['avg_proj_own']}%
{history_block}

## Your Task:
Based on this slate environment, determine the optimal DFS GPP lineup construction parameters.
Consider:
1. Does the salary distribution support a barbell strategy (studs + cheap), or is value concentrated in the mid-range?
2. How many viable studs exist, and how many viable cheap plays with real upside?
3. Does heavy chalk warrant higher ownership penalties and lower exposure caps?
4. Do injuries create mid-salary value that overrides the cheap floor?
5. Do blowout-risk games make game stacking risky?

Return ONLY a JSON object with exactly these fields:
{{
  "barbell_enabled": true or false,
  "stud_threshold": integer (salary floor for "stud" tier, e.g. 8500-10000),
  "stud_min": integer (0, 1, or 2 — minimum studs required per lineup),
  "cheap_threshold": integer (salary ceiling for "cheap" tier, e.g. 4000-5000),
  "cheap_min": integer (0, 1, 2, or 3 — minimum cheap plays required per lineup),
  "max_exposure_pct": float (0.25-0.45 — max fraction of lineups any player appears in),
  "min_ownership_penalty": float (0.02-0.05),
  "max_ownership_penalty": float (0.05-0.12),
  "role_player_std_mult": float (0.30-0.55 — ceiling width for low-salary role players),
  "role_player_proj_threshold": integer (proj_pts cutoff for "role player" label, e.g. 20-28),
  "role_player_sal_threshold": integer (salary cutoff for "role player" label, e.g. 5500-7500),
  "stack_emphasis": "high", "medium", or "low",
  "slate_type": "{stats['slate_type']}",
  "slate_size": {stats['game_count']},
  "injury_chaos_level": "low", "medium", or "high",
  "rationale": "2-3 sentence explanation of the key construction insight for tonight",
  "key_opportunities": ["list", "of", "up", "to", "3", "specific", "player-level", "or", "game-level", "opportunities"],
  "key_risks": ["list", "of", "up", "to", "3", "construction", "risks", "to", "avoid"]
}}"""

    # ── Claude API call ────────────────────────────────────────────────────────
    def _call_claude(self, prompt: str) -> str:
        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    # ── Response parsing ───────────────────────────────────────────────────────
    def _parse_response(self, raw: str, slate_stats: dict) -> dict:
        """Extract JSON from Claude response and validate all required fields."""
        # Strip markdown code fences if present
        text = raw
        if "```" in text:
            start = text.find("{", text.find("```"))
            end   = text.rfind("}") + 1
            text  = text[start:end] if start >= 0 and end > start else text

        try:
            profile = json.loads(text)
        except json.JSONDecodeError:
            # Last-resort: find the outermost { } block
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start >= 0 and end > start:
                profile = json.loads(raw[start:end])
            else:
                raise ValueError(f"No valid JSON found in Claude response:\n{raw[:400]}")

        # Merge with defaults for any missing fields, then clamp to safe ranges
        profile = {**_DEFAULT_PROFILE, **profile}
        profile = self._clamp_profile(profile, slate_stats)
        return profile

    # ── Rule-based fallback ────────────────────────────────────────────────────
    def _rule_based_profile(self, stats: dict) -> dict:
        """
        Deterministic fallback that applies domain rules when Claude is unavailable.
        Produces reasonable adaptive parameters without LLM inference.
        """
        profile = dict(_DEFAULT_PROFILE)
        profile["slate_type"] = stats["slate_type"]
        profile["slate_size"] = stats["game_count"]

        # ── Barbell feasibility ───────────────────────────────────────────────
        # Require enough players at both extremes to make the constraint viable
        if stats["studs_9k_plus"] < 3 or stats["cheap_viable_ct"] < 4:
            profile["barbell_enabled"] = False
            profile["stud_min"]        = max(0, min(1, stats["studs_9k_plus"] - 1))
            profile["cheap_min"]       = max(0, min(2, stats["cheap_viable_ct"] - 1))
        else:
            profile["barbell_enabled"] = True
            profile["stud_min"]        = 2
            profile["cheap_min"]       = 3

        # ── Salary tier thresholds — adaptive to actual distribution ──────────
        # Use p85 as stud floor if very few players exist above $9k
        if stats["studs_9k_plus"] <= 2:
            profile["stud_threshold"] = int(stats["sal_p75"])
        else:
            profile["stud_threshold"] = 9000

        # Use p20 as cheap ceiling if the cheap pool is thin
        if stats["cheap_sub_4500"] < 5:
            profile["cheap_threshold"] = int(stats["sal_p25"])
        else:
            profile["cheap_threshold"] = 4500

        # ── Exposure — tighter when field is heavy chalk ──────────────────────
        if stats["chalk_pct"] > 0.30:          # >30% of pool is projected chalk
            profile["max_exposure_pct"] = 0.28
        elif stats["chalk_pct"] < 0.15:
            profile["max_exposure_pct"] = 0.38  # spread field → can repeat more
        else:
            profile["max_exposure_pct"] = 0.33

        # ── Ownership penalty — higher when chalk is concentrated ─────────────
        if stats["chalk_player_count"] <= 3:
            profile["min_ownership_penalty"] = 0.04
            profile["max_ownership_penalty"] = 0.09
        elif stats["chalk_player_count"] >= 8:
            profile["min_ownership_penalty"] = 0.02
            profile["max_ownership_penalty"] = 0.05
        else:
            profile["min_ownership_penalty"] = 0.03
            profile["max_ownership_penalty"] = 0.07

        # ── Role player ceiling width — wider with injury chaos ───────────────
        injury_chaos = "low"
        if stats["out_count"] >= 5 or stats["star_outs"]:
            injury_chaos = "high"
            profile["role_player_std_mult"]       = 0.52
            profile["role_player_proj_threshold"] = 28
            profile["role_player_sal_threshold"]  = 7000
        elif stats["out_count"] >= 2 or stats["gtd_count"] >= 4:
            injury_chaos = "medium"
            profile["role_player_std_mult"]       = 0.45
            profile["role_player_proj_threshold"] = 26
            profile["role_player_sal_threshold"]  = 6800
        else:
            profile["role_player_std_mult"]       = 0.42
            profile["role_player_proj_threshold"] = 25
            profile["role_player_sal_threshold"]  = 6500
        profile["injury_chaos_level"] = injury_chaos

        # ── Stack emphasis — high for high-total slates ───────────────────────
        if stats["high_total_games"] >= 2:
            profile["stack_emphasis"] = "high"
        elif stats["blowout_risk_games"] >= 2:
            profile["stack_emphasis"] = "low"
        else:
            profile["stack_emphasis"] = "medium"

        # ── Rationale ─────────────────────────────────────────────────────────
        notes = []
        if not profile["barbell_enabled"]:
            notes.append(f"Barbell disabled — only {stats['studs_9k_plus']} studs and "
                         f"{stats['cheap_viable_ct']} viable cheap plays available.")
        if injury_chaos == "high":
            notes.append(f"High injury chaos ({stats['out_count']} OUT, "
                         f"stars out: {stats['star_outs'] or 'none'}) — "
                         "widening role-player ceilings to capture replacements.")
        if stats["blowout_risk_games"] >= 2:
            notes.append(f"{stats['blowout_risk_games']} blowout-risk games — "
                         "de-emphasizing stacking in lopsided matchups.")
        profile["rationale"] = " ".join(notes) or (
            f"{stats['slate_type'].title()} slate with standard construction. "
            f"Avg O/U {stats['avg_game_total']}, "
            f"{stats['studs_9k_plus']} studs, {stats['cheap_viable_ct']} viable cheap plays."
        )

        return profile

    # ── Safety clamping ────────────────────────────────────────────────────────
    def _clamp_profile(self, p: dict, stats: dict) -> dict:
        """Clamp all numeric fields to safe ranges to prevent optimizer issues."""
        p["stud_threshold"]            = int(max(7000, min(11000, p["stud_threshold"])))
        p["stud_min"]                  = int(max(0, min(2, p["stud_min"])))
        p["cheap_threshold"]           = int(max(3500, min(5500, p["cheap_threshold"])))
        p["cheap_min"]                 = int(max(0, min(3, p["cheap_min"])))
        p["max_exposure_pct"]          = float(max(0.20, min(0.50, p["max_exposure_pct"])))
        p["min_ownership_penalty"]     = float(max(0.01, min(0.08, p["min_ownership_penalty"])))
        p["max_ownership_penalty"]     = float(max(p["min_ownership_penalty"] + 0.01,
                                                    min(0.15, p["max_ownership_penalty"])))
        p["role_player_std_mult"]      = float(max(0.28, min(0.60, p["role_player_std_mult"])))
        p["role_player_proj_threshold"]= int(max(15,  min(35,  p["role_player_proj_threshold"])))
        p["role_player_sal_threshold"] = int(max(4500, min(9000, p["role_player_sal_threshold"])))

        # Safety: if not enough players exist for the barbell, disable it
        if p["barbell_enabled"]:
            stud_count  = (stats.get("studs_9k_plus", 99))
            cheap_count = (stats.get("cheap_viable_ct", 99))
            if stud_count < p["stud_min"] + 1 or cheap_count < p["cheap_min"] + 1:
                p["barbell_enabled"] = False
                p["stud_min"]        = 0
                p["cheap_min"]       = 0

        return p
