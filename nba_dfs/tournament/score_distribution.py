"""
Score Distribution Model
========================
For every player on the slate, models their full outcome distribution — not just
a single projection. Tournament play requires knowing the TAIL, not the mean.

Key outputs per player:
  - mean: expected DK points
  - std: standard deviation
  - p10 / p25 / p50 / p75 / p90: percentiles
  - explosion_rate: P(score >= 1.5x mean) — how often they go nuclear
  - bust_rate: P(score <= 0.4x mean OR DNP) — how often they tank a lineup
  - dnp_prob: probability of not playing at all
  - corr_team: intra-team correlation (if one star goes off, do teammates?)

These distributions drive the Monte Carlo simulation in ContestSimulator.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import erf as _scipy_erf, erfinv as _scipy_erfinv

# Default distribution parameters — calibrated from 4-night backtest (3/6–3/9).
# Key finding: cheap/value players have MUCH higher explosion rates than assumed.
#   Jaylin Williams ($5700): scored 2.80x projection (boom game)
#   Maxime Raynaud  ($6300): scored 2.25x projection
#   Malik Monk      ($4500): scored 2.15x projection
#   Gui Santos      ($6400): scored 2.89x projection
# std_mult drives how wide the log-normal distribution is — must be fat-tailed
# for value/cheap players to model these 2-3x outcomes.
# explosion_rate = P(score >= 1.5x mean). Empirically ~25-35% for value plays.
_SALARY_TIERS = {
    "elite":  {"min_sal": 10000, "std_mult": 0.28, "explosion_rate": 0.22, "bust_rate": 0.06, "dnp_prob": 0.02},
    "star":   {"min_sal":  8000, "std_mult": 0.34, "explosion_rate": 0.28, "bust_rate": 0.09, "dnp_prob": 0.03},
    "mid":    {"min_sal":  6500, "std_mult": 0.42, "explosion_rate": 0.32, "bust_rate": 0.14, "dnp_prob": 0.05},
    "value":  {"min_sal":  5000, "std_mult": 0.55, "explosion_rate": 0.35, "bust_rate": 0.20, "dnp_prob": 0.08},
    "cheap":  {"min_sal":  3500, "std_mult": 0.68, "explosion_rate": 0.28, "bust_rate": 0.35, "dnp_prob": 0.25},
    "floor":  {"min_sal":     0, "std_mult": 0.75, "explosion_rate": 0.15, "bust_rate": 0.55, "dnp_prob": 0.45},
}

# Game-log column names we look for (multiple naming conventions)
_FPTS_COLS = ["fantasy_pts_dk", "fpts_dk", "fpts", "dk_pts", "dkpts"]
_PTS_COLS  = ["pts", "points", "PTS"]


class ScoreDistribution:
    """
    Builds and maintains per-player score distributions.

    Usage:
        sd = ScoreDistribution()
        sd.fit(player_pool_df, game_logs_dict)
        dist = sd.get("SGA")  # returns distribution dict
        samples = sd.sample("SGA", n=10000)  # Monte Carlo samples
    """

    def __init__(self, history_dir: Optional[Path] = None):
        self.history_dir = history_dir or (Path(__file__).parent.parent / "cache" / "distributions")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._dists: dict[str, dict] = {}   # player_id -> distribution params
        self._game_logs: dict[str, list[float]] = {}  # player_id -> recent game scores

    # ── Fit distributions from player pool + game logs ────────────────────────
    def fit(
        self,
        players: pd.DataFrame,
        game_logs: Optional[dict] = None,
    ) -> "ScoreDistribution":
        """
        Compute distribution parameters for every player in the pool.

        players: DataFrame with at least [player_id, name, salary, avg_pts]
        game_logs: {player_id -> list of recent DK scores, newest first}
        """
        self._dists = {}
        for _, row in players.iterrows():
            pid    = str(row.get("player_id", row.get("name", "")))
            name   = str(row.get("name", pid))
            salary = int(row.get("salary", 5000))
            mean   = float(row.get("proj_pts_dk", row.get("avg_pts", 20)))
            dnp_risk = float(row.get("dnp_risk", 0.05))

            # Pull historical game scores for this player
            history = []
            if game_logs and pid in game_logs:
                logs = game_logs[pid]
                if isinstance(logs, pd.DataFrame):
                    for col in _FPTS_COLS:
                        if col in logs.columns:
                            history = logs[col].dropna().tolist()
                            break
                elif isinstance(logs, list):
                    history = [float(x) for x in logs if x is not None]

            # Use last 20 games (more recent = more relevant)
            history = [float(x) for x in history[:20] if float(x) >= 0]

            # Salary tier defaults
            tier_params = _SALARY_TIERS["cheap"]
            for tier_name, tp in _SALARY_TIERS.items():
                if salary >= tp["min_sal"]:
                    tier_params = tp
                    break

            if len(history) >= 8:
                # Use empirical distribution
                arr = np.array(history)
                emp_mean = float(np.mean(arr))
                emp_std  = float(np.std(arr))
                # Blend empirical with projection (projection may reflect tonight's context)
                blend_w = min(0.7, len(history) / 20)
                blended_mean = blend_w * emp_mean + (1 - blend_w) * mean
                blended_std  = max(emp_std, blended_mean * tier_params["std_mult"])
                explosion_rate = float(np.mean(arr >= 1.5 * emp_mean)) if emp_mean > 0 else tier_params["explosion_rate"]
                bust_rate      = float(np.mean(arr <= 0.4 * emp_mean)) if emp_mean > 0 else tier_params["bust_rate"]
                p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90]).tolist()
                has_empirical = True
            else:
                # Fallback: parametric log-normal approximation
                blended_mean   = mean
                blended_std    = mean * tier_params["std_mult"]
                explosion_rate = tier_params["explosion_rate"]
                bust_rate      = tier_params["bust_rate"]
                # Log-normal percentiles
                if mean > 0 and blended_std > 0:
                    sigma2 = math.log(1 + (blended_std / mean) ** 2)
                    mu_ln  = math.log(mean) - sigma2 / 2
                    sigma_ln = math.sqrt(sigma2)
                    p10, p25, p50, p75, p90 = [
                        math.exp(mu_ln + z * sigma_ln)
                        for z in [-1.282, -0.674, 0, 0.674, 1.282]
                    ]
                else:
                    p10 = p25 = p50 = p75 = p90 = mean
                has_empirical = False

            # DNP: if player doesn't play, they score exactly 0
            # Adjust burst metrics to account for DNP probability
            effective_explosion = explosion_rate * (1 - dnp_risk)
            effective_bust      = bust_rate * (1 - dnp_risk) + dnp_risk  # DNP = guaranteed bust

            # Salary-tiered ceiling — same formula as test_slate.build_projections()
            # Used as the anchor for explosion events in the mixture model.
            _CEIL_TIERS = [
                (10000, 1.65, 30), (8000, 1.85, 28), (6500, 2.10, 32),
                (5000, 2.80, 40), (0, 3.20, 40),
            ]
            ceiling = blended_mean  # fallback
            for sal_thresh, mult, max_add in _CEIL_TIERS:
                if salary >= sal_thresh:
                    ceiling = round(min(blended_mean * mult, blended_mean + max_add), 2)
                    break

            self._dists[pid] = {
                "player_id":       pid,
                "name":            name,
                "salary":          salary,
                "mean":            round(blended_mean, 2),
                "std":             round(blended_std, 2),
                "ceiling":         ceiling,
                "p10":             round(max(0, p10), 2),
                "p25":             round(max(0, p25), 2),
                "p50":             round(p50, 2),
                "p75":             round(p75, 2),
                "p90":             round(p90, 2),
                "explosion_rate":  round(effective_explosion, 4),
                "bust_rate":       round(effective_bust, 4),
                "dnp_prob":        round(dnp_risk, 4),
                "has_empirical":   has_empirical,
                "n_games":         len(history),
                "history":         history[:20],
            }
            self._game_logs[pid] = history

        return self

    # ── Sample from a player's distribution ───────────────────────────────────
    def sample(self, player_id: str, n: int = 10000, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Draw n Monte Carlo samples from a player's score distribution.
        Returns array of shape (n,) — each value is a possible DK score.

        Uses a mixture model:
          - With probability dnp_prob: score = 0
          - Otherwise: score ~ max(0, log-normal(mean, std))
        """
        if rng is None:
            rng = np.random.default_rng()

        dist = self._dists.get(player_id)
        if dist is None:
            return np.full(n, 20.0)  # default if unknown

        mean    = dist["mean"]
        std     = dist["std"]
        dnp_p   = dist["dnp_prob"]

        # DNP draw
        plays = rng.random(n) >= dnp_p

        # Score when playing: use empirical bootstrap if available, else log-normal
        history = dist.get("history", [])
        if len(history) >= 8:
            scores = rng.choice(history, size=n, replace=True).astype(float)
            # Add small gaussian noise to prevent discretisation artifacts
            scores += rng.normal(0, std * 0.15, size=n)
            scores = np.maximum(scores, 0)
        else:
            # Log-normal parametric
            if mean > 0 and std > 0:
                sigma2   = math.log(1 + (std / mean) ** 2)
                mu_ln    = math.log(mean) - sigma2 / 2
                sigma_ln = math.sqrt(sigma2)
                raw      = rng.lognormal(mu_ln, sigma_ln, size=n)
            else:
                raw = np.zeros(n)
            scores = np.maximum(raw, 0)

        # Zero out DNP events
        scores[~plays] = 0.0
        return scores

    # ── Correlated game-stack sampler ─────────────────────────────────────────
    def sample_correlated(
        self,
        player_ids: list[str],
        n: int = 10000,
        game_corr: float = 0.35,
        team_corr: float = 0.55,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample scores for multiple players with intra-game correlation.
        Returns array of shape (n, len(player_ids)).

        game_corr: correlation between same-game players from opposing teams
        team_corr: correlation between teammates (higher — same plays affect both)
        """
        if rng is None:
            rng = np.random.default_rng()

        k = len(player_ids)
        if k == 0:
            return np.empty((n, 0))

        # Build correlation matrix
        teams = [self._dists.get(pid, {}).get("team", pid) for pid in player_ids]
        matchups = [self._dists.get(pid, {}).get("matchup", "") for pid in player_ids]
        corr_mat = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                if teams[i] == teams[j]:
                    c = team_corr
                elif matchups[i] and matchups[i] == matchups[j]:
                    c = game_corr
                else:
                    c = 0.05  # minimal cross-game correlation
                corr_mat[i, j] = corr_mat[j, i] = c

        # Gaussian copula: generate correlated uniforms, then map to marginals
        try:
            L = np.linalg.cholesky(corr_mat)
        except np.linalg.LinAlgError:
            # Matrix not PSD — use diagonal (uncorrelated fallback)
            L = np.eye(k)

        z  = rng.standard_normal((k, n))
        zc = L @ z   # shape (k, n)
        u  = 0.5 * (1 + _scipy_erf(zc / math.sqrt(2)))  # CDF→uniform

        # Map uniform samples to each player's marginal distribution
        result = np.zeros((n, k))
        for i, pid in enumerate(player_ids):
            dist = self._dists.get(pid, {})
            history = dist.get("history", [])
            dnp_p = dist.get("dnp_prob", 0.05)

            if len(history) >= 8:
                sorted_hist = np.sort(history)
                quantiles   = np.interp(u[i], np.linspace(0, 1, len(sorted_hist)), sorted_hist)
                quantiles   = np.maximum(quantiles, 0)
            else:
                mean = dist.get("mean", 20)
                std  = dist.get("std", 6)
                if mean > 0 and std > 0:
                    sigma2   = math.log(1 + (std / mean) ** 2)
                    mu_ln    = math.log(mean) - sigma2 / 2
                    sigma_ln = math.sqrt(sigma2)
                    # Inverse log-normal CDF
                    z_ln = np.clip(u[i], 1e-6, 1 - 1e-6)
                    quantiles = np.exp(mu_ln + sigma_ln * math.sqrt(2) * _scipy_erfinv(2 * z_ln - 1))
                    quantiles = np.maximum(quantiles, 0)
                else:
                    quantiles = np.zeros(n)

            # Apply DNP
            dnp_mask = rng.random(n) < dnp_p
            quantiles[dnp_mask] = 0.0
            result[:, i] = quantiles

        return result

    # ── Accessors ────────────────────────────────────────────────────────────
    def get(self, player_id: str) -> Optional[dict]:
        return self._dists.get(player_id)

    def get_all(self) -> dict[str, dict]:
        return dict(self._dists)

    def set_team_matchup(self, player_id: str, team: str, matchup: str):
        """Add team/matchup info to a distribution (for correlated sampling)."""
        if player_id in self._dists:
            self._dists[player_id]["team"] = team
            self._dists[player_id]["matchup"] = matchup

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, date_str: str):
        path = self.history_dir / f"distributions_{date_str}.json"
        with open(path, "w") as f:
            json.dump(self._dists, f)

    def load(self, date_str: str) -> bool:
        path = self.history_dir / f"distributions_{date_str}.json"
        if path.exists():
            with open(path) as f:
                self._dists = json.load(f)
            return True
        return False
