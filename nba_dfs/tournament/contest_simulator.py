"""
Contest Simulator
=================
The core reason our system fails: we optimize expected value, not first-place equity.

This module answers the only question that matters in GPP tournaments:
  "Given the field's ownership distribution, what is the probability
   that THIS lineup finishes in the top 1% / top spot?"

Algorithm:
  1. Build a model of what the field is playing (ownership -> lineup distribution)
  2. For each of our lineups, run N_SIM game scenarios
  3. In each scenario, score every (sampled) field lineup and our lineup
  4. Count how often our lineup finishes top K% of the field
  5. Return First-Place Equity (FPE) and Top-1% Equity (T1E) per lineup

The FPE score replaces gpp_score as the optimization objective.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from tournament.score_distribution import ScoreDistribution


# Number of Monte Carlo simulations per lineup evaluation.
# 5,000 is fast enough (~0.5s per lineup) with good statistical stability.
# 10,000 gives tighter confidence intervals at ~1s per lineup.
N_SIM = 5000

# Field lineup sample size: how many field lineups to simulate per scenario.
# Real DraftKings contests have 10K-50K entries. We sample a representative set.
FIELD_SAMPLE = 3000


class ContestSimulator:
    """
    Monte Carlo contest simulator.

    Usage:
        sim = ContestSimulator(score_dist, field_ownership, n_sim=5000)
        result = sim.evaluate_lineup(lineup_player_ids)
        # result.fpe   = P(first place)
        # result.t1pct = P(top 1%)
        # result.t10pct = P(top 10%)

    For evaluating full portfolio of 20 lineups:
        results = sim.evaluate_portfolio(list_of_lineup_player_id_lists)
    """

    def __init__(
        self,
        score_dist: ScoreDistribution,
        players: pd.DataFrame,
        n_sim: int = N_SIM,
        field_sample: int = FIELD_SAMPLE,
        seed: int = 42,
    ):
        self.score_dist  = score_dist
        self.players     = players
        self.n_sim       = n_sim
        self.field_sample = field_sample
        self.rng          = np.random.default_rng(seed)

        # Build player index for fast lookup
        self._pid_to_idx = {
            str(row["player_id"]): i
            for i, row in players.iterrows()
        }
        self._pid_to_own = {
            str(row["player_id"]): float(row.get("proj_own", 15)) / 100.0
            for _, row in players.iterrows()
        }
        self._pid_to_name = {
            str(row["player_id"]): str(row["name"])
            for _, row in players.iterrows()
        }

        # Pre-generate the field lineup distribution
        # Each field lineup is a set of 8 player_ids sampled proportionally
        # to their ownership. This models what the average competitor plays.
        print(f"[sim] Pre-generating {field_sample} field lineups...")
        self._field_lineups = self._generate_field_lineups()
        print(f"[sim] Field lineups ready. Running simulations with {n_sim:,} scenarios.")

    # ── Field lineup generation ────────────────────────────────────────────────
    def _generate_field_lineups(self) -> list[list[str]]:
        """
        Generate a synthetic field of lineups based on ownership percentages.

        Method:
          - For each of the field_sample lineups, sample players proportionally
            to ownership (with position constraints) to build a valid 8-man lineup.
          - High-ownership players appear in more field lineups.
          - "Chalk" lineups cluster around the same popular cores.
        """
        pids   = [str(r["player_id"]) for _, r in self.players.iterrows()]
        owns   = np.array([self._pid_to_own.get(p, 0.10) for p in pids])
        sals   = np.array([float(r.get("salary", 5000)) for _, r in self.players.iterrows()])
        pos    = [str(r.get("primary_position", "G")) for _, r in self.players.iterrows()]

        SALARY_CAP  = 50_000
        ROSTER_SIZE = 8

        lineups = []
        attempts = 0
        while len(lineups) < self.field_sample and attempts < self.field_sample * 20:
            attempts += 1
            # Sample 8 players with probability proportional to ownership^0.7
            # (^0.7 = slight regression toward uniform to model uncertainty)
            probs = owns ** 0.7
            probs = probs / probs.sum()

            chosen_idx = self.rng.choice(len(pids), size=ROSTER_SIZE, replace=False, p=probs)
            chosen_pids = [pids[i] for i in chosen_idx]
            chosen_sal  = sals[chosen_idx].sum()

            # Simple salary cap check (field lineups don't need to be perfectly valid)
            if chosen_sal <= SALARY_CAP * 1.02 and chosen_sal >= SALARY_CAP * 0.96:
                lineups.append(chosen_pids)

        # Pad if we didn't generate enough valid lineups
        while len(lineups) < self.field_sample:
            top_own_idx = np.argsort(owns)[-ROSTER_SIZE:]
            lineups.append([pids[i] for i in top_own_idx])

        return lineups

    # ── Score simulation ───────────────────────────────────────────────────────
    def _simulate_scores(self, all_player_ids: list[str]) -> np.ndarray:
        """
        Run n_sim scenarios for the given player set.
        Returns array of shape (n_sim, len(all_player_ids)).

        Uses correlated sampling to respect intra-game correlations.
        """
        return self.score_dist.sample_correlated(
            all_player_ids,
            n=self.n_sim,
            rng=self.rng,
        )

    # ── Single lineup evaluation ───────────────────────────────────────────────
    def evaluate_lineup(self, lineup_pids: list[str]) -> dict:
        """
        Compute First-Place Equity and Top-1%/10% Equity for a single lineup.

        Returns:
            fpe:     P(this lineup beats every field lineup) — first place equity
            t1pct:   P(this lineup finishes top 1% of field)
            t10pct:  P(this lineup finishes top 10% of field)
            mean_pts: expected total DK points
            p90_pts:  90th percentile total DK points (ceiling)
            leverage: how differentiated this lineup is from field
        """
        # Collect all unique player IDs we need scores for
        all_pids = list(set(lineup_pids) | {p for lu in self._field_lineups for p in lu})
        all_pids = [p for p in all_pids if p in self._pid_to_idx or p in self.score_dist.get_all()]

        # Simulate scores for all players across all scenarios
        score_matrix = self._simulate_scores(all_pids)  # (n_sim, n_players)
        pid_col = {pid: i for i, pid in enumerate(all_pids)}

        def lineup_score(lu: list[str], scenario: int) -> float:
            return sum(
                score_matrix[scenario, pid_col[p]]
                for p in lu if p in pid_col
            )

        # Our lineup scores across all scenarios
        our_scores = np.array([
            sum(score_matrix[s, pid_col.get(p, 0)] if p in pid_col else 0
                for p in lineup_pids)
            for s in range(self.n_sim)
        ])

        # Field lineup scores — sample a subset of field lineups for speed
        n_field_sample = min(500, len(self._field_lineups))
        field_sample_idx = self.rng.integers(0, len(self._field_lineups), n_field_sample)
        field_scores = np.zeros((self.n_sim, n_field_sample))

        for j, fi in enumerate(field_sample_idx):
            field_lu = self._field_lineups[fi]
            for p in field_lu:
                if p in pid_col:
                    field_scores[:, j] += score_matrix[:, pid_col[p]]

        # Count how often our lineup beats X% of the field
        # beats_all[s] = 1 if our lineup beats every sampled field lineup in scenario s
        beats_all = np.all(our_scores[:, None] > field_scores, axis=1)

        # t1pct: P(our lineup is in top 1% of field) = P(our_score > 99th pct of field)
        field_p99 = np.percentile(field_scores, 99, axis=1)  # (n_sim,) — per-scenario threshold
        field_p90 = np.percentile(field_scores, 90, axis=1)  # (n_sim,)

        fpe    = float(np.mean(beats_all))
        t1pct  = float(np.mean(our_scores > field_p99))
        t10pct = float(np.mean(our_scores > field_p90))

        # Ownership-based leverage score
        field_own_sum = np.mean([
            sum(self._pid_to_own.get(p, 0.10) for p in lu)
            for lu in self._field_lineups[:200]
        ])
        our_own_sum = sum(self._pid_to_own.get(p, 0.10) for p in lineup_pids)
        leverage = field_own_sum - our_own_sum  # positive = less chalky than field

        return {
            "fpe":       round(fpe, 6),
            "t1pct":     round(t1pct, 4),
            "t10pct":    round(t10pct, 4),
            "mean_pts":  round(float(np.mean(our_scores)), 2),
            "p90_pts":   round(float(np.percentile(our_scores, 90)), 2),
            "p99_pts":   round(float(np.percentile(our_scores, 99)), 2),
            "leverage":  round(leverage, 3),
        }

    # ── Portfolio evaluation ───────────────────────────────────────────────────
    def evaluate_portfolio(self, portfolio: list[list[str]]) -> dict:
        """
        Evaluate a full portfolio of lineups.
        Returns per-lineup metrics plus portfolio-level stats.

        Portfolio FPE = P(at least one lineup finishes first) — this is what matters.
        """
        lineup_results = []
        for i, lineup_pids in enumerate(portfolio):
            result = self.evaluate_lineup(lineup_pids)
            result["lineup_num"] = i + 1
            lineup_results.append(result)
            print(f"  [sim] Lineup {i+1:2d}: FPE={result['fpe']*100:.3f}%  "
                  f"T1%={result['t1pct']*100:.2f}%  "
                  f"P90={result['p90_pts']:.0f}pts  "
                  f"Lev={result['leverage']:+.3f}")

        # Portfolio-level stats
        # P(portfolio wins) = 1 - P(all lineups lose) = 1 - product(1-FPE_i)
        all_fpe  = [r["fpe"] for r in lineup_results]
        port_fpe = 1.0 - math.prod(1 - f for f in all_fpe)

        all_t1   = [r["t1pct"] for r in lineup_results]
        port_t1  = 1.0 - math.prod(1 - t for t in all_t1)

        best_lu  = max(lineup_results, key=lambda x: x["fpe"])
        best_t1  = max(lineup_results, key=lambda x: x["t1pct"])
        avg_p90  = sum(r["p90_pts"] for r in lineup_results) / len(lineup_results)

        print(f"\n  [sim] PORTFOLIO: FPE={port_fpe*100:.3f}%  "
              f"T1%={port_t1*100:.2f}%  AvgP90={avg_p90:.0f}pts")
        print(f"  [sim] Best FPE lineup: #{best_lu['lineup_num']} "
              f"({best_lu['fpe']*100:.4f}%)")
        print(f"  [sim] Best T1% lineup: #{best_t1['lineup_num']} "
              f"({best_t1['t1pct']*100:.3f}%)")

        return {
            "lineup_results":   lineup_results,
            "portfolio_fpe":    round(port_fpe, 6),
            "portfolio_t1pct":  round(port_t1, 4),
            "best_fpe_lineup":  best_lu["lineup_num"],
            "avg_p90_pts":      round(avg_p90, 2),
        }

    # ── Marginal value of adding a player ─────────────────────────────────────
    def marginal_fpe(self, base_lineup: list[str], candidate_pid: str, slot_pid: str) -> float:
        """
        How much does swapping slot_pid -> candidate_pid improve FPE?
        Used during portfolio construction to evaluate candidate swaps.
        """
        new_lineup = [candidate_pid if p == slot_pid else p for p in base_lineup]
        base_result = self.evaluate_lineup(base_lineup)
        new_result  = self.evaluate_lineup(new_lineup)
        return new_result["fpe"] - base_result["fpe"]
