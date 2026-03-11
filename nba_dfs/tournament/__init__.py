"""
Tournament-First NBA DFS Engine
================================
Objective: Maximize P(first place) and P(top 1%) across a portfolio of lineups.

This module completely replaces median-projection optimization with:
  1. contest_simulator  — Monte Carlo contest simulation (10K scenarios per lineup)
  2. score_distribution — Per-player outcome distributions from historical data
  3. portfolio_optimizer — Simultaneous portfolio construction for max FPE
  4. self_improver       — Continuous learning from every contest result
  5. autonomous_runner   — Endless slate runner requiring zero human input
"""
from tournament.contest_simulator import ContestSimulator
from tournament.score_distribution import ScoreDistribution
from tournament.portfolio_optimizer import PortfolioOptimizer
from tournament.self_improver import SelfImprover
from tournament.autonomous_runner import AutonomousRunner

__all__ = [
    "ContestSimulator",
    "ScoreDistribution",
    "PortfolioOptimizer",
    "SelfImprover",
    "AutonomousRunner",
]
