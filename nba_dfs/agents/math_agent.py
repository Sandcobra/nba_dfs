"""
Mathematical Agent.
Responsibilities:
  - Monte Carlo simulation of fantasy point distributions
  - Bayesian updating of projections with latest game data
  - Variance / ceiling / floor estimation
  - Poisson modeling of rare stats (blocks, steals, 3-pointers)
  - Kelly criterion for contest entry sizing
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import poisson, norm, gamma, beta
from loguru import logger
from typing import Optional

from core.config import MONTE_CARLO_SIMS


class MathAgent:
    """Handles all probabilistic and mathematical modeling."""

    def __init__(self, n_sims: int = MONTE_CARLO_SIMS):
        self.n_sims = n_sims
        self.rng    = np.random.default_rng(seed=42)

    # ── Monte Carlo simulation ─────────────────────────────────────────────────
    def monte_carlo_projections(
        self,
        projections: pd.DataFrame,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Simulate n_sims fantasy-point outcomes for each player.
        Uses Gamma distribution for non-negative, skewed fantasy points.
        If correlation_matrix provided, simulates correlated outcomes.

        Returns DataFrame with:
          player_id, name, mean, std, p10 (floor), p25, p50 (median),
          p75, p90 (ceiling), p_bust (< 15 DK pts), p_ceiling (> 50 DK pts)
        """
        logger.info(f"Running {self.n_sims:,} Monte Carlo simulations...")
        n_players = len(projections)
        mu  = projections["projected_pts_dk"].fillna(25).values.clip(1, 90)
        std = projections.get("projection_sd", pd.Series(mu * 0.28)).fillna(pd.Series(mu * 0.28)).values.clip(1, 40)

        if correlation_matrix is not None and correlation_matrix.shape == (n_players, n_players):
            sim_matrix = self._correlated_simulation(mu, std, correlation_matrix)
        else:
            sim_matrix = self._independent_simulation(mu, std)

        # sim_matrix shape: (n_sims, n_players)
        results = []
        for i, row in projections.iterrows():
            sims = sim_matrix[:, i]
            results.append({
                "player_id":   row.get("player_id"),
                "name":        row.get("name", ""),
                "mean_sim":    float(sims.mean()),
                "std_sim":     float(sims.std()),
                "floor":       float(np.percentile(sims, 10)),
                "p25":         float(np.percentile(sims, 25)),
                "median_sim":  float(np.percentile(sims, 50)),
                "p75":         float(np.percentile(sims, 75)),
                "ceiling":     float(np.percentile(sims, 90)),
                "p99_ceiling": float(np.percentile(sims, 99)),
                "p_bust":      float((sims < 15).mean()),     # < 15 DK pts = bust
                "p_ceiling":   float((sims > 50).mean()),     # > 50 DK pts = ceiling
                "p_double_double": float((sims > 40).mean()), # proxy for DD bonus range
            })

        logger.success("Monte Carlo complete")
        return pd.DataFrame(results)

    def _independent_simulation(self, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Gamma-distributed simulation for each player independently."""
        # Gamma: shape=k, scale=theta; mean=k*theta, var=k*theta^2
        # k = (mu/std)^2,  theta = std^2/mu
        epsilon = 1e-6
        k     = (mu / (std + epsilon)) ** 2
        theta = (std ** 2) / (mu + epsilon)
        sim_matrix = self.rng.gamma(shape=k[None, :], scale=theta[None, :],
                                    size=(self.n_sims, len(mu)))
        return sim_matrix.clip(0)

    def _correlated_simulation(
        self, mu: np.ndarray, std: np.ndarray, corr: np.ndarray
    ) -> np.ndarray:
        """
        Simulate correlated fantasy-point outcomes via Gaussian copula.
        1. Draw correlated normals
        2. Transform through Gamma marginals
        """
        n = len(mu)
        try:
            L = np.linalg.cholesky(corr + np.eye(n) * 1e-6)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed; falling back to independent simulation")
            return self._independent_simulation(mu, std)

        Z = self.rng.standard_normal((self.n_sims, n)) @ L.T   # correlated normals
        U = norm.cdf(Z)                                           # uniform marginals

        epsilon = 1e-6
        k     = (mu / (std + epsilon)) ** 2
        theta = (std ** 2) / (mu + epsilon)

        sim_matrix = np.zeros_like(Z)
        for i in range(n):
            sim_matrix[:, i] = gamma(a=k[i], scale=theta[i]).ppf(U[:, i])

        return sim_matrix.clip(0)

    # ── Bayesian updating ──────────────────────────────────────────────────────
    def bayesian_update(
        self,
        prior_mean: float,
        prior_std: float,
        observed_values: list[float],
        likelihood_std: float = 8.0,
    ) -> tuple[float, float]:
        """
        Gaussian conjugate prior update.
        Returns (posterior_mean, posterior_std).
        """
        if not observed_values:
            return prior_mean, prior_std

        n    = len(observed_values)
        obs_mean = np.mean(observed_values)

        prior_var = prior_std ** 2
        like_var  = (likelihood_std ** 2) / n

        post_var  = 1 / (1 / prior_var + 1 / like_var)
        post_mean = post_var * (prior_mean / prior_var + obs_mean / like_var)
        return float(post_mean), float(np.sqrt(post_var))

    def bayesian_update_projections(
        self,
        projections: pd.DataFrame,
        game_logs: dict[int, pd.DataFrame],
        n_recent_games: int = 5,
    ) -> pd.DataFrame:
        """
        Apply Bayesian updates to all projections using recent game data.
        """
        df = projections.copy()
        for idx, row in df.iterrows():
            pid  = row.get("player_id")
            logs = game_logs.get(pid)
            if logs is None or logs.empty:
                continue
            recent = logs["fantasy_pts_dk"].head(n_recent_games).tolist()
            prior_mean = row["projected_pts_dk"]
            prior_std  = row.get("projection_sd", prior_mean * 0.28)
            new_mean, new_std = self.bayesian_update(prior_mean, prior_std, recent)
            df.at[idx, "projected_pts_dk"] = round(new_mean, 2)
            df.at[idx, "projection_sd"]    = round(new_std, 2)
        return df

    # ── Poisson modeling for discrete stats ───────────────────────────────────
    def poisson_stat_distribution(
        self,
        lambda_: float,
        max_val: int = 15,
    ) -> dict[int, float]:
        """
        Returns P(X = k) for k in 0..max_val using Poisson distribution.
        Useful for modeling 3PM, blocks, steals (rare events).
        """
        dist = {}
        for k in range(max_val + 1):
            dist[k] = poisson.pmf(k, lambda_)
        return dist

    def estimate_triple_double_probability(
        self, pts: float, reb: float, ast: float, stl: float, blk: float
    ) -> float:
        """
        Estimate probability of a triple-double using independent Poisson models.
        A triple-double = 10+ in 3 of 5 categories.
        """
        lambdas = {"pts": pts, "reb": reb, "ast": ast, "stl": stl, "blk": blk}
        p_10plus = {
            cat: 1 - poisson.cdf(9, lam)
            for cat, lam in lambdas.items()
            if lam > 0
        }
        cats = list(p_10plus.values())
        n = len(cats)
        if n < 3:
            return 0.0

        # P(at least 3 categories hit 10+) using inclusion-exclusion approximation
        from itertools import combinations
        p_td = 0.0
        for r in range(3, n + 1):
            for combo in combinations(range(n), r):
                p = np.prod([cats[i] for i in combo])
                p_not = np.prod([1 - cats[i] for i in range(n) if i not in combo])
                sign = (-1) ** (r - 3)
                p_td += sign * p * p_not
        return max(0.0, min(1.0, p_td))

    # ── Expected value calculations ────────────────────────────────────────────
    def compute_dk_ev(
        self,
        projections: pd.DataFrame,
        n_lineups: int,
        salary_cap: int = 50_000,
        prize_pool_pct: float = 0.90,
    ) -> pd.DataFrame:
        """
        Compute EV contribution of each player across lineups.
        Higher ceiling + lower ownership = higher GPP EV.
        """
        df = projections.copy()
        df["ev_score"] = (
            df["ceiling"] * (1 - df.get("proj_ownership", 20) / 100) *
            df["projected_pts_dk"] / (df["salary"] / 1000)
        )
        df["ev_score"] = df["ev_score"].round(3)
        return df.sort_values("ev_score", ascending=False)

    # ── Kelly criterion for contest entry ──────────────────────────────────────
    def kelly_contest_sizing(
        self,
        win_probability: float,
        payout_multiplier: float,
        bankroll: float,
        kelly_fraction: float = 0.25,
    ) -> float:
        """
        Kelly criterion: optimal fraction of bankroll to wager.
        kelly_fraction: 0.25 = quarter-Kelly (recommended for DFS volatility)
        Returns: dollar amount to enter into this contest.
        """
        if win_probability <= 0 or payout_multiplier <= 1:
            return 0.0
        b = payout_multiplier - 1
        p = win_probability
        q = 1 - p
        kelly = (b * p - q) / b
        return max(0, kelly * kelly_fraction * bankroll)

    # ── Lineup score distribution ──────────────────────────────────────────────
    def simulate_lineup_scores(
        self,
        lineup_projections: pd.DataFrame,
        n_sims: int = 10_000,
    ) -> np.ndarray:
        """
        Simulate total lineup scores for a given 8-man DK lineup.
        Returns array of total lineup scores across n_sims.
        """
        mu  = lineup_projections["projected_pts_dk"].fillna(25).values.clip(1)
        std = lineup_projections.get(
            "projection_sd",
            pd.Series(mu * 0.28)
        ).fillna(pd.Series(mu * 0.28)).values.clip(1)

        sim = self._independent_simulation(mu, std)  # (n_sims, 8)
        lineup_totals = sim.sum(axis=1)               # (n_sims,)
        return lineup_totals

    def score_lineup_probability_of_winning(
        self,
        lineup_score_dist: np.ndarray,
        contest_size: int,
        pct_of_field_better: float = 0.15,
    ) -> float:
        """
        Rough estimate of probability a lineup wins a GPP contest.
        Assumes top pct_of_field_better of the field can beat our lineup.
        """
        # Model field scores as normal distribution
        field_mean = lineup_score_dist.mean() * 0.97
        field_std  = lineup_score_dist.std()  * 1.10
        # Win = our score > best opponent
        # For large fields, use extreme value theory approximation
        threshold = lineup_score_dist.mean()
        p_win = 1 - norm.cdf(threshold, field_mean, field_std) ** contest_size
        return min(max(p_win, 0), 1)

    # ── Game pace / implied stats ──────────────────────────────────────────────
    def compute_implied_team_stats(
        self,
        game_total: float,
        home_team_pace: float,
        away_team_pace: float,
        home_def_rating: float,
        away_def_rating: float,
    ) -> dict:
        """
        From Vegas total + pace/defense, estimate implied team-level stats.
        """
        avg_pace  = (home_team_pace + away_team_pace) / 2
        # Implied possessions
        poss = avg_pace

        # Implied team totals (split O/U)
        home_adj  = (100 / home_def_rating) * (100 / away_def_rating) * 100
        away_adj  = (100 / away_def_rating) * (100 / home_def_rating) * 100
        total_adj = home_adj + away_adj

        home_implied = game_total * (home_adj / total_adj)
        away_implied = game_total * (away_adj / total_adj)

        return {
            "poss":             round(poss, 1),
            "home_implied_pts": round(home_implied, 1),
            "away_implied_pts": round(away_implied, 1),
            "game_total":       round(game_total, 1),
        }

    # ── Regression to the mean ────────────────────────────────────────────────
    def regression_adjusted_projection(
        self,
        recent_avg: float,
        season_avg: float,
        n_games: int,
        regression_k: float = 20,
    ) -> float:
        """
        James-Stein shrinkage toward season mean.
        n_games: number of recent games observed
        regression_k: strength of regression (higher = more shrinkage)
        """
        w_recent = n_games / (n_games + regression_k)
        w_season = 1 - w_recent
        return w_recent * recent_avg + w_season * season_avg

    def apply_regression_to_projections(
        self, projections: pd.DataFrame, game_logs: dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        df = projections.copy()
        for idx, row in df.iterrows():
            pid  = row.get("player_id")
            logs = game_logs.get(pid)
            if logs is None or logs.empty:
                continue
            n        = min(len(logs), 10)
            recent   = float(logs["fantasy_pts_dk"].head(n).mean())
            season   = float(logs["fantasy_pts_dk"].mean())
            adjusted = self.regression_adjusted_projection(recent, season, n)
            df.at[idx, "projected_pts_dk"] = round(adjusted, 2)
        return df
