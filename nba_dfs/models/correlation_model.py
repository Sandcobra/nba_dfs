"""
Player correlation / stacking model.
- Positive correlation: teammates (fast-break assists, second-chance points)
- Negative correlation: defend-heavy matchups (LeBron guarded by elite wing)
- Game-stack: players from both teams in high-total games
"""

import itertools
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None


class CorrelationModel:
    """
    Computes pairwise player correlations and identifies optimal stacks.
    """

    def __init__(self):
        self.corr_matrix: Optional[pd.DataFrame] = None
        self.stack_scores: Optional[pd.DataFrame] = None

    # ── Build correlation matrix ───────────────────────────────────────────────
    def build_correlation_matrix(
        self,
        game_logs: dict[int, pd.DataFrame],
        player_pool: pd.DataFrame,
        min_shared_games: int = 15,
    ) -> pd.DataFrame:
        """
        Compute pairwise Spearman correlations for DK fantasy points
        across shared game dates for teammates.
        """
        logger.info("Building player correlation matrix...")
        player_ids = player_pool["player_id"].tolist()
        n = len(player_ids)
        corr = np.zeros((n, n))

        for i, j in itertools.combinations(range(n), 2):
            pid_i = player_ids[i]
            pid_j = player_ids[j]
            logs_i = game_logs.get(pid_i)
            logs_j = game_logs.get(pid_j)

            if logs_i is None or logs_j is None:
                continue

            # Align on game dates
            merged = pd.merge(
                logs_i[["game_date", "fantasy_pts_dk"]].rename(columns={"fantasy_pts_dk": "fp_i"}),
                logs_j[["game_date", "fantasy_pts_dk"]].rename(columns={"fantasy_pts_dk": "fp_j"}),
                on="game_date",
            )

            if len(merged) < min_shared_games:
                continue

            if _scipy_stats is not None:
                r, _ = _scipy_stats.spearmanr(merged["fp_i"], merged["fp_j"])
            else:
                # Fallback: numpy Pearson correlation when scipy unavailable
                r = float(np.corrcoef(merged["fp_i"], merged["fp_j"])[0, 1])
            corr[i][j] = r
            corr[j][i] = r
            corr[i][i] = 1.0
            corr[j][j] = 1.0

        idx = player_pool.set_index("player_id")["name"]
        names = [idx.get(pid, str(pid)) for pid in player_ids]
        self.corr_matrix = pd.DataFrame(corr, index=names, columns=names)
        logger.success(f"Correlation matrix built: {n}x{n}")
        return self.corr_matrix

    # ── Identify stacks ────────────────────────────────────────────────────────
    def get_teammate_stacks(
        self,
        player_pool: pd.DataFrame,
        min_stack: int = 2,
        max_stack: int = 4,
    ) -> list[dict]:
        """
        Returns list of {players, team, stack_size, combined_proj, avg_corr, stack_score}.
        Sorts by stack_score descending.
        """
        stacks = []
        teams = player_pool["team"].unique()

        for team in teams:
            team_players = player_pool[player_pool["team"] == team].copy()
            if len(team_players) < min_stack:
                continue

            # Sort by projected points descending
            team_players = team_players.sort_values("projected_pts_dk", ascending=False)

            for size in range(min_stack, min(max_stack + 1, len(team_players) + 1)):
                for combo in itertools.combinations(team_players.itertuples(), size):
                    names  = [p.name for p in combo]
                    projs  = [p.projected_pts_dk for p in combo]
                    combined_proj = sum(projs)

                    # Average pairwise correlation
                    if self.corr_matrix is not None and all(n in self.corr_matrix.index for n in names):
                        pairs = list(itertools.combinations(names, 2))
                        avg_corr = np.mean([self.corr_matrix.loc[a, b] for a, b in pairs]) if pairs else 0
                    else:
                        avg_corr = 0.25  # default positive correlation for teammates

                    # Stack score: proj + correlation bonus
                    stack_score = combined_proj * (1 + 0.15 * avg_corr)

                    stacks.append({
                        "team":         team,
                        "players":      names,
                        "player_ids":   [p.player_id for p in combo],
                        "stack_size":   size,
                        "combined_proj": round(combined_proj, 2),
                        "avg_corr":     round(avg_corr, 3),
                        "stack_score":  round(stack_score, 2),
                        "game_total":   team_players.iloc[0].get("game_total", 225)
                            if hasattr(team_players.iloc[0], "game_total") else 225,
                    })

        stacks.sort(key=lambda x: x["stack_score"], reverse=True)
        logger.info(f"Generated {len(stacks)} team stacks")
        return stacks

    def get_game_stacks(
        self, player_pool: pd.DataFrame, top_n_games: int = 3
    ) -> list[dict]:
        """
        Game stacks: combine players from both teams in high-total games.
        NBA game stacks are effective because:
        - High-total games = fast pace = more possessions for both teams
        """
        if "game_total" not in player_pool.columns:
            return []

        game_totals = (
            player_pool.groupby("opp")["game_total"]
            .first()
            .nlargest(top_n_games)
        )

        stacks = []
        for opp_team, total in game_totals.items():
            # Get all players in this game
            home = player_pool[player_pool["team"] == opp_team]
            away = player_pool[
                (player_pool["opp"] == opp_team) |
                (player_pool["team"].isin(
                    player_pool[player_pool["opp"] == opp_team]["team"].unique()
                ))
            ]
            game_players = pd.concat([home, away]).drop_duplicates("player_id")

            top_from_each = game_players.groupby("team").apply(
                lambda g: g.nlargest(3, "projected_pts_dk")
            ).reset_index(drop=True)

            stacks.append({
                "game":        f"vs {opp_team}",
                "game_total":  total,
                "players":     top_from_each["name"].tolist(),
                "player_ids":  top_from_each["player_id"].tolist(),
                "combined_proj": top_from_each["projected_pts_dk"].sum(),
            })

        return stacks

    # ── Lineup correlation score ───────────────────────────────────────────────
    def score_lineup_correlation(self, lineup_names: list[str]) -> float:
        """
        Score a lineup by its internal correlation structure.
        Positive correlation = good for GPP (all go off together).
        """
        if self.corr_matrix is None or len(lineup_names) < 2:
            return 0.0

        valid = [n for n in lineup_names if n in self.corr_matrix.index]
        if len(valid) < 2:
            return 0.0

        pairs = list(itertools.combinations(valid, 2))
        corrs = [self.corr_matrix.loc[a, b] for a, b in pairs]
        return float(np.mean(corrs))

    # ── Negative correlations (avoid pairings) ─────────────────────────────────
    def get_negative_correlations(
        self,
        player_pool: pd.DataFrame,
        threshold: float = -0.15,
    ) -> list[tuple[str, str, float]]:
        """
        Return pairs with correlation below threshold to AVOID stacking.
        Common: star player vs. their primary defender.
        """
        if self.corr_matrix is None:
            return []
        names = [n for n in player_pool["name"] if n in self.corr_matrix.index]
        neg_pairs = []
        for a, b in itertools.combinations(names, 2):
            r = self.corr_matrix.loc[a, b]
            if r < threshold:
                neg_pairs.append((a, b, round(r, 3)))
        neg_pairs.sort(key=lambda x: x[2])
        return neg_pairs
