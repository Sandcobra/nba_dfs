"""
Game Theory Agent.
Applies game-theoretic principles to DFS contest strategy:
  - Nash equilibrium lineup diversification
  - Leverage plays (contrarian strategy)
  - Field composition modeling
  - Portfolio theory for lineup set construction
  - Roster construction based on contest type (GPP vs Cash)
"""

import itertools
import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional


class GameTheoryAgent:
    """
    Applies game-theoretic and portfolio optimization principles to DFS.
    """

    def __init__(self):
        self.field_model: Optional[dict] = None

    # ── Contest type strategy ──────────────────────────────────────────────────
    def get_contest_strategy(self, contest_type: str) -> dict:
        """
        Returns strategic parameters based on contest type.

        GPP (tournaments):
          - Maximize ceiling exposure
          - Target low-ownership plays for leverage
          - Stack correlated players
          - Accept more variance

        Cash (50/50, H2H, 3-max):
          - Maximize floor
          - Avoid GTD/risky plays
          - High-ownership "chalk" is fine
          - Minimize variance
        """
        strategies = {
            "gpp": {
                "optimize_for":         "ceiling",
                "max_ownership":        35.0,       # avoid >35% owned in GPP
                "target_stack_size":    3,
                "ownership_penalty":    0.04,        # penalize high ownership
                "variance_preference":  "high",
                "use_game_stack":       True,
                "min_game_total_stack": 220,         # only stack high-total games
                "chalk_pct_max":        0.40,        # max 40% lineup chalk
                "min_value_salary":     5000,        # avoid overpaying for chalk
                "diversify_captains":   True,
                "max_players_per_team": 4,
            },
            "cash": {
                "optimize_for":         "floor",
                "max_ownership":        80.0,
                "target_stack_size":    2,
                "ownership_penalty":    0.0,
                "variance_preference":  "low",
                "use_game_stack":       False,
                "min_game_total_stack": 0,
                "chalk_pct_max":        0.80,
                "min_value_salary":     4000,
                "diversify_captains":   False,
                "max_players_per_team": 4,
            },
            "double_up": {
                "optimize_for":         "floor",
                "max_ownership":        80.0,
                "target_stack_size":    2,
                "ownership_penalty":    0.0,
                "variance_preference":  "low",
                "use_game_stack":       False,
                "min_game_total_stack": 0,
                "chalk_pct_max":        0.80,
                "min_value_salary":     4000,
                "diversify_captains":   False,
                "max_players_per_team": 4,
            },
            "showdown": {
                "optimize_for":         "ceiling",
                "max_ownership":        30.0,
                "target_stack_size":    2,
                "ownership_penalty":    0.05,
                "variance_preference":  "high",
                "use_game_stack":       True,
                "min_game_total_stack": 0,
                "chalk_pct_max":        0.35,
                "min_value_salary":     5000,
                "diversify_captains":   True,
                "max_players_per_team": 5,           # showdown has one game
            },
        }
        return strategies.get(contest_type.lower(), strategies["gpp"])

    # ── Leverage score ─────────────────────────────────────────────────────────
    def compute_leverage_scores(
        self,
        projections: pd.DataFrame,
        strategy: str = "gpp",
    ) -> pd.DataFrame:
        """
        Leverage = (Projection - Salary Implied Projection) *
                   (1 - Projected Ownership %)

        High leverage = good projection relative to salary + low ownership.
        """
        df = projections.copy()
        strat = self.get_contest_strategy(strategy)

        # Salary-implied projection baseline (avg $ per point at position level)
        avg_ppk = df["projected_pts_dk"] / (df["salary"] / 1000)
        df["salary_implied_pts"] = df["salary"] / 1000 * avg_ppk.mean()

        # Over / under performance vs salary
        df["proj_vs_salary"] = df["projected_pts_dk"] - df["salary_implied_pts"]

        # Ownership leverage multiplier
        own = df.get("proj_ownership", pd.Series(20.0, index=df.index)).clip(1, 99)
        df["ownership_leverage"] = 1 - (own / 100)

        if strat["optimize_for"] == "ceiling":
            proj_col = "ceiling"
        else:
            proj_col = "floor"

        base_col = df.get(proj_col, df["projected_pts_dk"])

        df["leverage_score"] = (
            (df["proj_vs_salary"] + 1) *
            df["ownership_leverage"] *
            base_col
        ).round(3)

        df["is_leverage_play"] = (
            (df["proj_ownership"] < 15) &
            (df["projected_pts_dk"] > df["projected_pts_dk"].quantile(0.6))
        )

        df["is_chalk"] = df["proj_ownership"] > 30

        return df.sort_values("leverage_score", ascending=False)

    # ── Nash diversification ───────────────────────────────────────────────────
    def nash_lineup_diversification(
        self,
        base_lineup_scores: list[float],
        player_pool: pd.DataFrame,
        n_target: int = 150,
    ) -> dict:
        """
        Inspired by Nash equilibrium: in a large GPP, the optimal strategy
        is to not all choose the same lineup. Diversification across
        correlated groups maximizes collective EV.

        Returns diversification plan with player exposure targets.
        """
        n_players = len(player_pool)

        # Target exposures based on projection rank
        proj_rank = player_pool["projected_pts_dk"].rank(pct=True)

        # High-proj players: moderate exposure (not over-concentrated)
        # Mid-proj "upside" players: higher exposure (leverage)
        # Low-proj players: minimal exposure (only in speculative lineups)
        target_exp = (
            0.30 * proj_rank +              # base exposure from projection rank
            0.20 * (1 - player_pool.get("proj_ownership", 20) / 100) +  # low-own bonus
            0.10 * (player_pool.get("ceiling", 40) / 60).clip(0, 1)     # ceiling bonus
        ).clip(0.02, 0.70)

        player_pool = player_pool.copy()
        player_pool["target_exposure"] = (target_exp * n_target).round(0).astype(int)

        return {
            "n_lineups":         n_target,
            "exposure_targets":  player_pool[["name", "player_id", "target_exposure",
                                              "proj_ownership", "projected_pts_dk"]],
        }

    # ── Field composition modeling ─────────────────────────────────────────────
    def model_field_composition(
        self,
        player_pool: pd.DataFrame,
        contest_size: int = 10_000,
    ) -> pd.DataFrame:
        """
        Model how the DFS field will roster players.
        Returns player_pool with 'field_ownership_model' column.

        Key principles:
        - Top 5 salary players own ~30-45% of the field
        - Injured / GTD players own less than their projection suggests
        - Game-theory: some GPP players are "off-limits" due to risk aversion
        """
        df = player_pool.copy()
        base_own  = df.get("proj_ownership", pd.Series(10.0, index=df.index))

        # Salary gravity: higher salary → mass public attraction
        sal_pct = df["salary"].rank(pct=True)

        # Risk discount for uncertain players
        risk_map = {"GTD": 0.70, "QUESTIONABLE": 0.60, "DOUBTFUL": 0.35,
                    "OUT": 0.00, "ACTIVE": 1.0, "PROBABLE": 0.95}
        inj_multiplier = df.get("injury_status", "ACTIVE").map(risk_map).fillna(1.0)

        # Field model
        df["field_ownership_model"] = (
            base_own * 0.60 +           # base projection
            sal_pct  * 20               # salary attraction
        ) * inj_multiplier

        df["field_ownership_model"] = df["field_ownership_model"].clip(0.5, 70).round(1)
        df["leverage_vs_field"] = (
            df.get("proj_ownership", 10) - df["field_ownership_model"]
        ).round(2)

        return df

    # ── Optimal exposure targets ───────────────────────────────────────────────
    def compute_optimal_exposures(
        self,
        projections: pd.DataFrame,
        n_lineups: int,
        contest_type: str = "gpp",
        max_exp_top: float = 0.55,
        max_exp_mid: float = 0.40,
        max_exp_low: float = 0.20,
    ) -> pd.DataFrame:
        """
        Set max exposure % for each player bucket based on projection tier.
        GPP: reduce exposure on heavily-owned favorites
        Cash: increase exposure on high-floor plays
        """
        df = projections.copy()
        proj_q33 = df["projected_pts_dk"].quantile(0.33)
        proj_q66 = df["projected_pts_dk"].quantile(0.66)

        def _max_exp(row):
            own  = row.get("proj_ownership", 20)
            proj = row["projected_pts_dk"]
            if contest_type == "cash":
                if proj > proj_q66:
                    return min(max_exp_top * 1.5, 0.90)
                return max_exp_mid
            else:  # gpp
                if proj > proj_q66 and own > 25:
                    return max_exp_mid     # high proj + high own = moderate exposure
                elif proj > proj_q66 and own <= 25:
                    return max_exp_top     # high proj + low own = max exposure
                elif proj_q33 < proj <= proj_q66:
                    return max_exp_mid
                else:
                    return max_exp_low

        df["max_exposure"] = df.apply(_max_exp, axis=1)
        df["max_lineup_count"] = (df["max_exposure"] * n_lineups).round(0).astype(int)
        return df

    # ── Expected value of lineup portfolio ────────────────────────────────────
    def portfolio_ev(
        self,
        lineups: list[dict],
        sim_scores: np.ndarray,
        contest_prize_structure: dict,
    ) -> float:
        """
        Compute expected dollar value of a portfolio of lineups.
        sim_scores: shape (n_sims, n_lineups)
        contest_prize_structure: {rank_cutoff: prize_amount}
        """
        total_ev = 0.0
        n_sims = sim_scores.shape[0]
        n_lineups = sim_scores.shape[1]

        for sim_idx in range(n_sims):
            scores = sim_scores[sim_idx]
            sorted_scores = np.sort(scores)[::-1]
            for lu_idx in range(n_lineups):
                rank = int((scores[lu_idx] < sorted_scores).sum()) + 1
                for cutoff, prize in contest_prize_structure.items():
                    if rank <= cutoff:
                        total_ev += prize
                        break

        return total_ev / n_sims / n_lineups

    # ── Showdown / Captain mode helpers ───────────────────────────────────────
    def get_showdown_captain_candidates(
        self,
        player_pool: pd.DataFrame,
        n_candidates: int = 10,
    ) -> pd.DataFrame:
        """
        Captain slot = 1.5x salary + 1.5x points.
        Best captains: high upside, not all-chalk,
                       from high-implied teams.
        """
        df = player_pool.copy()
        df["captain_salary"] = (df["salary"] * 1.5).astype(int)
        df["captain_proj"]   = df["projected_pts_dk"] * 1.5

        # Captain score: ceiling-weighted, low-own bonus
        df["captain_score"] = (
            df.get("ceiling", df["projected_pts_dk"] * 1.3) * 1.5 *
            (1 - df.get("proj_ownership", 20) / 200)
        )

        return df.nlargest(n_candidates, "captain_score")[
            ["name", "team", "primary_position", "captain_salary",
             "captain_proj", "captain_score", "proj_ownership"]
        ]

    # ── Injury roster construction adjustment ─────────────────────────────────
    def adjust_for_injuries(
        self,
        player_pool: pd.DataFrame,
        injuries: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Adjust projections and exposure for injury news.
        - OUT players: remove from pool
        - GTD/Q: reduce projection by uncertainty factor
        - Beneficiary players: boost projection (usage absorption)
        """
        df = player_pool.copy()
        if injuries.empty:
            return df

        out_players = injuries[injuries["status"] == "OUT"]["name"].tolist()
        gtd_players = injuries[injuries["status"].isin(["GTD", "QUESTIONABLE"])]["name"].tolist()

        # Remove OUT players
        df = df[~df["name"].isin(out_players)].copy()

        # Reduce GTD player projections by 30%
        gtd_mask = df["name"].isin(gtd_players)
        df.loc[gtd_mask, "projected_pts_dk"] *= 0.70
        df.loc[gtd_mask, "injury_status"] = "GTD"

        # Find beneficiaries (same team as OUT player)
        if not injuries.empty and "team" in injuries.columns:
            out_teams = injuries[injuries["status"] == "OUT"]["team"].unique()
            for team in out_teams:
                team_mask = (df["team"] == team) & (~df["name"].isin(gtd_players))
                # Distribute usage: +8% to remaining starters
                df.loc[team_mask, "projected_pts_dk"] *= 1.08
                df.loc[team_mask, "injury_boosted"] = True
                logger.info(f"Boosted {team_mask.sum()} players on {team} for injury absence")

        return df.reset_index(drop=True)
