"""
Field Behavior Agent.

Models what the DFS public field will actually build tonight —
not just who they'll own, but which *constructions* will dominate.

The DFS field is predictable:
  - They stack the highest-projection player with their highest-projection teammate
  - They run the best $/pts value at cheap slots
  - They avoid injury uncertainty (GTD/Q players get passed over)
  - They over-index on narrative plays (game of the night, easiest matchup)
  - They copy last-night's winning constructions when the slate is similar

This agent answers:
  1. What does the "median field lineup" look like tonight?
  2. Which player *combinations* will be over-concentrated in the field?
  3. Which high-owned players are chalk *traps* (high own + low ceiling)?
  4. Which game stacks / salary constructions is the field *under-indexing*?
  5. How do we build lineups the field won't have?

Outputs a FieldProfile dict consumed by app.py before lineup generation.
The profile influences:
  - proj_own adjustments (chalk stack correlation)
  - ILP ownership penalty weighting
  - UI warnings on chalk trap players
  - Differentiation targets surfaced in the log stream
"""

import itertools
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CHALK_OWN_THRESHOLD   = 20.0   # % — players above this are "chalk"
TRAP_UPSIDE_RATIO     = 1.22   # ceiling/proj below this = low upside (trap candidate)
TRAP_MIN_OWN          = 15.0   # must be at least this owned to be a trap
STACK_CONCENTRATION   = 0.50   # if field's top 2 players from a game sum to >50% own = over-concentrated
DIFF_OWN_CEILING      = 0.45   # game is "underowned" if top-2 combined own < 45%
SALARY_TIERS = {
    "stud":  (9000, 99999),
    "hi_mid":(7000, 8999),
    "lo_mid":(5500, 6999),
    "value": (4500, 5499),
    "cheap": (0,    4499),
}


class FieldBehaviorAgent:
    """
    Models aggregate DFS public field construction behavior for a given slate.
    Pure math — no API calls, no external data needed.
    """

    def model(
        self,
        players:      pd.DataFrame,
        game_totals:  dict,
        contest_size: int = 5000,
        n_lineups:    int = 20,
    ) -> dict:
        """
        Main entry point. Returns a FieldProfile dict.

        Parameters
        ----------
        players      : enriched player pool (must have proj_own, proj_pts_dk,
                       ceiling, salary, team, matchup, status)
        game_totals  : {"AWAY@HOME": {"total", "home_implied", "away_implied"}}
        contest_size : number of entries in the contest (affects chalk tolerance)
        n_lineups    : your lineup count (used for advice framing)
        """
        df = players.copy()

        chalk_core        = self._identify_chalk_core(df)
        chalk_traps       = self._identify_chalk_traps(df, chalk_core)
        stack_analysis    = self._analyze_game_stacks(df, game_totals)
        field_archetype   = self._build_field_archetype(df)
        diff_targets      = self._find_differentiation_targets(df, stack_analysis, chalk_core)
        own_correlation   = self._compute_chalk_stack_correlation(df, chalk_core)
        construction_tips = self._build_construction_tips(
            chalk_core, chalk_traps, stack_analysis, diff_targets, contest_size
        )

        profile = {
            "chalk_core":           chalk_core,
            "chalk_traps":          chalk_traps,
            "stack_analysis":       stack_analysis,
            "field_archetype":      field_archetype,
            "differentiation_targets": diff_targets,
            "own_correlation_boosts":  own_correlation,
            "construction_tips":    construction_tips,
        }

        logger.info(
            "[field] Chalk core: %d players | Traps: %d | Over-concentrated stacks: %d | Diff targets: %d",
            len(chalk_core), len(chalk_traps),
            sum(1 for s in stack_analysis if s.get("over_concentrated")),
            len(diff_targets),
        )
        return profile

    # ── 1. Chalk core identification ───────────────────────────────────────────
    def _identify_chalk_core(self, df: pd.DataFrame) -> list[dict]:
        """
        Identify the players the field will concentrate on.
        Chalk core = top-owned players weighted by salary gravity.

        Returns list of dicts sorted by field_lock_score descending.
        field_lock_score = proj_own * salary_rank  (highly owned + expensive = locked in)
        """
        df2 = df.copy()
        df2 = df2[df2["status"].isin(["ACTIVE", "B2B", "PROBABLE"]) | df2["status"].isna()]
        df2["sal_rank_pct"] = df2["salary"].rank(pct=True)
        df2["field_lock_score"] = df2["proj_own"] * df2["sal_rank_pct"]

        chalk = df2[df2["proj_own"] >= CHALK_OWN_THRESHOLD].copy()
        chalk = chalk.sort_values("field_lock_score", ascending=False)

        result = []
        for _, row in chalk.head(12).iterrows():
            result.append({
                "name":            row.get("name", ""),
                "team":            row.get("team", ""),
                "matchup":         row.get("matchup", ""),
                "salary":          int(row.get("salary", 0)),
                "proj_own":        round(float(row.get("proj_own", 0)), 1),
                "proj_pts_dk":     round(float(row.get("proj_pts_dk", 0)), 1),
                "ceiling":         round(float(row.get("ceiling", 0)), 1),
                "field_lock_score":round(float(row.get("field_lock_score", 0)), 2),
                "upside_ratio":    round(
                    float(row.get("ceiling", 0)) / max(float(row.get("proj_pts_dk", 1)), 0.1), 3
                ),
            })
        return result

    # ── 2. Chalk trap detection ────────────────────────────────────────────────
    def _identify_chalk_traps(
        self, df: pd.DataFrame, chalk_core: list[dict]
    ) -> list[dict]:
        """
        Chalk traps = highly owned players with LOW ceiling/projection ratio.
        These are "floor plays" the public over-rostered because they're safe —
        but they can't win you a tournament.

        Trap score = proj_own / upside_ratio
        High trap_score = high ownership + low ceiling upside = avoid in GPP.
        """
        chalk_names = {c["name"] for c in chalk_core}
        df2 = df[df["name"].isin(chalk_names)].copy()
        df2["upside_ratio"] = df2["ceiling"] / df2["proj_pts_dk"].clip(lower=0.1)
        df2["trap_score"]   = df2["proj_own"] / df2["upside_ratio"].clip(lower=0.1)

        traps = df2[
            (df2["proj_own"] >= TRAP_MIN_OWN) &
            (df2["upside_ratio"] < TRAP_UPSIDE_RATIO)
        ].sort_values("trap_score", ascending=False)

        result = []
        for _, row in traps.iterrows():
            result.append({
                "name":        row.get("name", ""),
                "team":        row.get("team", ""),
                "salary":      int(row.get("salary", 0)),
                "proj_own":    round(float(row.get("proj_own", 0)), 1),
                "proj_pts_dk": round(float(row.get("proj_pts_dk", 0)), 1),
                "ceiling":     round(float(row.get("ceiling", 0)), 1),
                "upside_ratio":round(float(row.get("upside_ratio", 0)), 3),
                "trap_score":  round(float(row.get("trap_score", 0)), 2),
                "reason":      (
                    f"{row.get('name','')} is {row.get('proj_own',0):.0f}% owned "
                    f"but ceiling ({row.get('ceiling',0):.1f}) is only "
                    f"{row.get('upside_ratio',0):.2f}x projection — low tournament upside."
                ),
            })
        return result

    # ── 3. Game stack analysis ─────────────────────────────────────────────────
    def _analyze_game_stacks(
        self, df: pd.DataFrame, game_totals: dict
    ) -> list[dict]:
        """
        For each game, compute:
          - top 2 players per team by proj_own (field's likely stack picks)
          - combined ownership of the field's natural 2-man stack
          - whether the game is over-concentrated (field stacking heavily)
          - stack differentiation score (value per unit of field ownership)
        """
        stacks = []
        for matchup, gt_data in game_totals.items():
            game_df = df[df["matchup"] == matchup].copy()
            if game_df.empty:
                continue

            total = float(gt_data.get("total", 220))
            home_imp = float(gt_data.get("home_implied", total / 2))
            away_imp = float(gt_data.get("away_implied", total / 2))

            teams = game_df["team"].unique()
            team_data = []
            for team in teams:
                team_df = game_df[game_df["team"] == team]
                top2    = team_df.nlargest(2, "proj_own")
                top2_own   = float(top2["proj_own"].sum())
                top2_proj  = float(top2["proj_pts_dk"].sum())
                top2_names = top2["name"].tolist()
                team_data.append({
                    "team":      team,
                    "top2_own":  round(top2_own, 1),
                    "top2_proj": round(top2_proj, 1),
                    "top2_names":top2_names,
                })

            # Total "field pull" = combined top-2 ownership across both teams
            total_top2_own  = sum(t["top2_own"] for t in team_data)
            total_top2_proj = sum(t["top2_proj"] for t in team_data)
            over_concentrated = total_top2_own > (STACK_CONCENTRATION * 100 * 2)

            # Differentiation score: proj pts per unit of combined ownership
            diff_score = total_top2_proj / max(total_top2_own, 1)

            stacks.append({
                "matchup":          matchup,
                "game_total":       total,
                "home_implied":     home_imp,
                "away_implied":     away_imp,
                "teams":            team_data,
                "total_top2_own":   round(total_top2_own, 1),
                "total_top2_proj":  round(total_top2_proj, 1),
                "over_concentrated":over_concentrated,
                "diff_score":       round(diff_score, 3),
            })

        stacks.sort(key=lambda s: s["game_total"], reverse=True)
        return stacks

    # ── 4. Field archetype lineup ──────────────────────────────────────────────
    def _build_field_archetype(self, df: pd.DataFrame) -> dict:
        """
        Construct the "median field lineup" — the archetypal roster the
        average contestant will submit tonight.

        Method: pick the highest proj_own player at each salary tier
        subject to DK position constraints (simplified).
        Returns the 8-player set and total projected salary.
        """
        if "status" in df.columns:
            df2 = df[df["status"].isin(["ACTIVE", "B2B", "PROBABLE"]) | df["status"].isna()].copy()
        else:
            df2 = df.copy()

        picks   = []
        used    = set()
        budget  = 0
        # Tier allocation: 2 studs, 2 hi_mid, 2 lo_mid, 2 cheap/value
        tier_alloc = [
            ("stud",   2),
            ("hi_mid", 2),
            ("lo_mid", 2),
            ("value",  1),
            ("cheap",  1),
        ]
        for tier, count in tier_alloc:
            lo, hi = SALARY_TIERS[tier]
            tier_df = df2[
                (df2["salary"] >= lo) & (df2["salary"] <= hi) &
                (~df2["name"].isin(used))
            ].sort_values("proj_own", ascending=False)
            for _, row in tier_df.head(count).iterrows():
                picks.append({
                    "name":    row.get("name", ""),
                    "team":    row.get("team", ""),
                    "salary":  int(row.get("salary", 0)),
                    "proj_own":round(float(row.get("proj_own", 0)), 1),
                    "tier":    tier,
                })
                used.add(row.get("name", ""))
                budget += int(row.get("salary", 0))

        return {
            "players":       picks,
            "total_salary":  budget,
            "avg_own":       round(np.mean([p["proj_own"] for p in picks]), 1) if picks else 0,
            "description":   (
                f"Field archetype: {', '.join(p['name'] for p in picks[:4])} "
                f"+ {len(picks)-4} more | avg own {round(np.mean([p['proj_own'] for p in picks]),1) if picks else 0:.1f}%"
            ),
        }

    # ── 5. Differentiation targets ─────────────────────────────────────────────
    def _find_differentiation_targets(
        self,
        df: pd.DataFrame,
        stack_analysis: list[dict],
        chalk_core: list[dict],
    ) -> list[dict]:
        """
        Find game stacks and individual players the field is under-indexing.

        Differentiation target criteria:
          - Game: high O/U + low combined top-2 ownership = field ignoring a good game
          - Player: proj_pts_dk in top 40% but proj_own in bottom 30% = underowned value
        """
        chalk_names = {c["name"] for c in chalk_core}
        targets = []

        # Under-owned games (field ignoring a high-scoring game)
        for stack in stack_analysis:
            if (stack["game_total"] >= 225 and
                    stack["total_top2_own"] < DIFF_OWN_CEILING * 100):
                targets.append({
                    "type":       "game_stack",
                    "matchup":    stack["matchup"],
                    "game_total": stack["game_total"],
                    "combined_own": stack["total_top2_own"],
                    "diff_score": stack["diff_score"],
                    "reason":     (
                        f"{stack['matchup']} O/U {stack['game_total']} but only "
                        f"{stack['total_top2_own']:.0f}% combined field ownership "
                        f"on top-2 players — field is ignoring this game."
                    ),
                })

        # Under-owned individual players
        proj_q60 = df["proj_pts_dk"].quantile(0.60)
        own_q30  = df["proj_own"].quantile(0.30)
        diff_players = df[
            (df["proj_pts_dk"] >= proj_q60) &
            (df["proj_own"] <= own_q30) &
            (~df["name"].isin(chalk_names)) &
            (df["status"].isin(["ACTIVE", "B2B", "PROBABLE"]) if "status" in df.columns else True)
        ].sort_values("proj_pts_dk", ascending=False)

        for _, row in diff_players.head(5).iterrows():
            targets.append({
                "type":        "player",
                "name":        row.get("name", ""),
                "team":        row.get("team", ""),
                "salary":      int(row.get("salary", 0)),
                "proj_pts_dk": round(float(row.get("proj_pts_dk", 0)), 1),
                "proj_own":    round(float(row.get("proj_own", 0)), 1),
                "ceiling":     round(float(row.get("ceiling", 0)), 1),
                "diff_score":  round(
                    float(row.get("proj_pts_dk", 0)) / max(float(row.get("proj_own", 1)), 1), 3
                ),
                "reason":      (
                    f"{row.get('name','')} projects {row.get('proj_pts_dk',0):.1f} pts "
                    f"at only {row.get('proj_own',0):.1f}% ownership — "
                    f"field is undervaluing this player."
                ),
            })

        targets.sort(key=lambda t: t.get("diff_score", 0), reverse=True)
        return targets

    # ── 6. Chalk stack correlation boosts ─────────────────────────────────────
    def _compute_chalk_stack_correlation(
        self, df: pd.DataFrame, chalk_core: list[dict]
    ) -> list[dict]:
        """
        When two chalk players are from the same team, the field stacks them
        together more than their individual ownerships suggest.

        Returns ownership correlation adjustments for co-owned pairs.
        These are applied to proj_own to simulate "the field doubles up here."
        """
        boosts = []
        chalk_names = {c["name"]: c for c in chalk_core}

        for (n1, c1), (n2, c2) in itertools.combinations(chalk_names.items(), 2):
            if c1["team"] != c2["team"]:
                continue  # only teammates compound
            # Joint probability assuming mild positive correlation (0.3)
            # P(A ∩ B) ≈ P(A) * P(B) * (1 + 0.3 * min(P(A), P(B)) / 0.5)
            p1 = c1["proj_own"] / 100
            p2 = c2["proj_own"] / 100
            joint_raw = p1 * p2
            corr_boost = joint_raw * (1 + 0.30 * min(p1, p2) / 0.50)
            corr_boost_pct = round(corr_boost * 100, 1)

            if corr_boost_pct >= 3.0:   # only meaningful combos
                boosts.append({
                    "players":   [n1, n2],
                    "team":      c1["team"],
                    "joint_own": corr_boost_pct,
                    "individual_own": [c1["proj_own"], c2["proj_own"]],
                    "note": (
                        f"{n1}+{n2} ({c1['team']}) will appear together in "
                        f"~{corr_boost_pct:.0f}% of field lineups."
                    ),
                })

        boosts.sort(key=lambda b: b["joint_own"], reverse=True)
        return boosts

    # ── 7. Construction tips ───────────────────────────────────────────────────
    def _build_construction_tips(
        self,
        chalk_core:    list[dict],
        chalk_traps:   list[dict],
        stack_analysis:list[dict],
        diff_targets:  list[dict],
        contest_size:  int,
    ) -> list[dict]:
        """
        Synthesize all signals into prioritized, actionable construction tips.
        """
        tips = []
        is_large_field = contest_size >= 3000

        # Chalk trap warnings
        for trap in chalk_traps[:3]:
            tips.append({
                "priority": "high",
                "category": "avoid",
                "player":   trap["name"],
                "message":  trap["reason"],
            })

        # Over-concentrated game stack warning
        for stack in stack_analysis:
            if stack["over_concentrated"]:
                team_names = " & ".join(
                    t["team"] for t in stack["teams"]
                )
                tips.append({
                    "priority": "medium",
                    "category": "fade_or_differentiate",
                    "matchup":  stack["matchup"],
                    "message":  (
                        f"{stack['matchup']} (O/U {stack['game_total']}) is the "
                        f"field's most concentrated stack ({stack['total_top2_own']:.0f}% "
                        f"combined top-2 ownership). Either fade it completely or "
                        f"differentiate within it using the 3rd/4th option."
                    ),
                })

        # Differentiation opportunities
        for target in diff_targets[:3]:
            if target["type"] == "game_stack":
                tips.append({
                    "priority": "high" if is_large_field else "medium",
                    "category": "target",
                    "matchup":  target.get("matchup"),
                    "message":  target["reason"],
                })
            elif target["type"] == "player":
                tips.append({
                    "priority": "medium",
                    "category": "target",
                    "player":   target.get("name"),
                    "message":  target["reason"],
                })

        # Large field: being different is more valuable
        if is_large_field and len(chalk_core) >= 4:
            top_chalk = chalk_core[0]
            tips.append({
                "priority": "medium",
                "category": "strategy",
                "message":  (
                    f"Large field ({contest_size:,} entries): "
                    f"{top_chalk['name']} ({top_chalk['proj_own']:.0f}% own) will be in "
                    f"~{int(contest_size * top_chalk['proj_own']/100):,} lineups. "
                    f"Fading chalk in 30-40% of your lineups maximizes top-finish probability."
                ),
            })

        return tips
