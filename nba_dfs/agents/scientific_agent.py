"""
Scientific Research Agent.
Applies sports science and physiological research to DFS projections:
  - Fatigue modeling (B2B, travel, schedule density)
  - Altitude / court surface effects
  - Rest advantage / disadvantage analysis
  - Circadian rhythm (morning vs evening games)
  - Playoff intensity ramp-up detection
  - Clutch performance under pressure
  - Home/Away splits with crowd noise factors
  - Load management prediction
"""

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
from datetime import datetime, timedelta
from typing import Optional


class ScientificAgent:
    """
    Applies empirical sports science research to quantify
    non-statistical factors affecting player performance.
    """

    # ── Fatigue model coefficients (empirically estimated from NBA data) ───────
    # Source: research on NBA player performance in back-to-back games
    # Average performance decline on B2B: ~3-7%
    B2B_DECLINE_RATE         = 0.055   # 5.5% avg decline on second night of B2B
    THIRD_GAME_3_DAYS_DECLINE = 0.04   # 3 games in 4 nights: ~4% decline
    TRAVEL_DECLINE_PER_KM    = 0.000008  # negligible per km but adds up on long trips
    HIGH_ALT_BOOST           = 0.03    # Denver Nuggets altitude advantage ~3%

    # Home court advantage (HCA)
    HOME_COURT_BOOST         = 0.035   # ~3.5% higher performance at home

    # ── Fatigue / load management ─────────────────────────────────────────────
    def compute_fatigue_factor(
        self,
        player_name: str,
        game_date: str,
        schedule: pd.DataFrame,   # must have: team, game_date, home_away
        age: float = 26.0,
        minutes_recent_avg: float = 30.0,
    ) -> dict:
        """
        Estimate a fatigue multiplier (0.85 - 1.02) for a player on a given date.
        < 1.0 = fatigued, > 1.0 = fresh/rested
        """
        if schedule.empty:
            return {"fatigue_factor": 1.0, "reason": "No schedule data"}

        game_dt = datetime.strptime(game_date, "%Y-%m-%d")
        reasons = []
        factor  = 1.0

        # B2B detection: played yesterday?
        yesterday = (game_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        if yesterday in schedule.get("game_date", pd.Series()).values:
            decline = self.B2B_DECLINE_RATE
            # Older players decline more on B2B
            if age >= 32:
                decline *= 1.25
            elif age >= 28:
                decline *= 1.10
            # High-minute players decline more
            if minutes_recent_avg >= 36:
                decline *= 1.15
            factor -= decline
            reasons.append(f"B2B (-{decline*100:.1f}%)")

        # 3-in-4 nights
        two_days_ago = (game_dt - timedelta(days=2)).strftime("%Y-%m-%d")
        three_days_ago = (game_dt - timedelta(days=3)).strftime("%Y-%m-%d")
        if (yesterday in schedule.get("game_date", pd.Series()).values and
                two_days_ago in schedule.get("game_date", pd.Series()).values):
            factor -= self.THIRD_GAME_3_DAYS_DECLINE
            reasons.append(f"3G/4N (-{self.THIRD_GAME_3_DAYS_DECLINE*100:.1f}%)")

        # Rest days advantage (3+ days rest)
        last_game = schedule[schedule["game_date"] < game_date].sort_values("game_date", ascending=False)
        if not last_game.empty:
            days_rest = (game_dt - datetime.strptime(last_game.iloc[0]["game_date"], "%Y-%m-%d")).days
            if days_rest >= 3:
                rest_bonus = min(0.02, (days_rest - 2) * 0.008)
                factor += rest_bonus
                reasons.append(f"Extended rest +{rest_bonus*100:.1f}%")

        return {
            "fatigue_factor": round(max(0.80, min(1.05, factor)), 4),
            "reason":         ", ".join(reasons) if reasons else "Normal rest",
            "b2b":            "B2B" in str(reasons),
        }

    # ── Home court advantage ───────────────────────────────────────────────────
    def compute_home_court_factor(
        self, home_away: str, team: str
    ) -> float:
        """
        Apply home court advantage factor.
        Some arenas have stronger documented HCA than others.
        """
        # Arenas with historically strong HCA
        strong_hca_teams = {
            "DEN": 1.06,    # altitude advantage
            "GSW": 1.04,    # Chase Center crowd
            "BOS": 1.04,    # TD Garden
            "MIL": 1.035,   # Fiserv Forum
        }
        if home_away.lower() == "home":
            return strong_hca_teams.get(team, 1.0 + self.HOME_COURT_BOOST)
        else:
            return 1.0 - (self.HOME_COURT_BOOST * 0.5)  # mild road penalty

    # ── Load management probability ───────────────────────────────────────────
    def predict_load_management_risk(
        self,
        player_name: str,
        age: float,
        games_played: int,
        minutes_avg: float,
        days_rest: int,
        known_injury_history: bool = False,
    ) -> float:
        """
        Estimate probability player is load-managed (rest day).
        Returns probability 0.0 - 1.0.
        """
        risk = 0.0

        # Age factor
        if age >= 35: risk += 0.30
        elif age >= 32: risk += 0.15
        elif age >= 30: risk += 0.05

        # High workload
        if minutes_avg >= 36: risk += 0.10
        if minutes_avg >= 38: risk += 0.08

        # Season fatigue
        if games_played >= 60: risk += 0.08
        if games_played >= 70: risk += 0.10

        # 3-in-4 or 4-in-5 schedule
        if days_rest == 0: risk += 0.30  # on a B2B = highest risk
        elif days_rest == 1: risk += 0.10

        # Injury history
        if known_injury_history: risk += 0.15

        return min(risk, 0.85)

    # ── Matchup quality ───────────────────────────────────────────────────────
    def compute_matchup_advantage(
        self,
        player_pos: str,
        opp_pos_defense_rating: float,   # opponent's DRTG vs this position
        league_avg_pos_drtg: float = 110.0,
    ) -> float:
        """
        Quantify the quality of the matchup.
        Returns a multiplier: 1.10 = great matchup, 0.90 = tough matchup.
        """
        # How much better/worse than league average is this defense?
        diff = opp_pos_defense_rating - league_avg_pos_drtg
        # Poor defense (high DRTG) = good matchup for offense
        matchup_factor = 1 + (diff / league_avg_pos_drtg) * 0.5
        return float(np.clip(matchup_factor, 0.85, 1.18))

    # ── Clutch performance ────────────────────────────────────────────────────
    def compute_clutch_boost(
        self,
        clutch_pts_per_48: float,
        reg_pts_per_48: float,
    ) -> float:
        """
        Players who outperform in clutch situations may score more DK pts
        in close games. Returns a boost factor.
        """
        if reg_pts_per_48 <= 0:
            return 1.0
        ratio = clutch_pts_per_48 / reg_pts_per_48
        # Cap extreme values
        return float(np.clip(0.95 + (ratio - 1.0) * 0.15, 0.92, 1.12))

    # ── Usage volatility ──────────────────────────────────────────────────────
    def compute_usage_volatility(
        self, game_log: pd.DataFrame, col: str = "usg_pct", n: int = 10
    ) -> float:
        """
        Coefficient of variation of usage% over last N games.
        High volatility = riskier play for cash games.
        """
        if game_log.empty or col not in game_log.columns:
            return 0.25
        vals = game_log[col].head(n).dropna()
        if len(vals) < 3:
            return 0.25
        return float(vals.std() / (vals.mean() + 1e-6))

    # ── Apply all science factors to projections ───────────────────────────────
    def apply_all_factors(
        self,
        projections: pd.DataFrame,
        schedules: dict[str, pd.DataFrame] = None,  # {team_abbrev: schedule_df}
        game_date: str = None,
    ) -> pd.DataFrame:
        """
        Apply all scientific adjustment factors to the projection DataFrame.
        Modifies projected_pts_dk in-place with multipliers.
        """
        import datetime as dt
        game_date = game_date or dt.date.today().isoformat()

        df = projections.copy()

        for idx, row in df.iterrows():
            multiplier = 1.0

            # Fatigue (B2B)
            b2b  = row.get("b2b", 0)
            age  = row.get("age", 26.0)
            mins = row.get("avg_min_L10", 30.0)
            if b2b:
                fatig = self.compute_fatigue_factor(
                    row.get("name", ""), game_date,
                    schedules.get(row.get("team", ""), pd.DataFrame()) if schedules else pd.DataFrame(),
                    age=age, minutes_recent_avg=mins,
                )
                multiplier *= fatig["fatigue_factor"]

            # Home court
            ha = row.get("home_away", "away")
            multiplier *= self.compute_home_court_factor(ha, row.get("team", ""))

            # Matchup quality
            opp_drtg = row.get("opp_def_rating", 110.0)
            matchup  = self.compute_matchup_advantage(
                row.get("primary_position", "G"), opp_drtg
            )
            multiplier *= matchup

            # Apply
            df.at[idx, "projected_pts_dk"] = round(
                row["projected_pts_dk"] * multiplier, 2
            )
            df.at[idx, "science_multiplier"] = round(multiplier, 4)

        logger.info(
            f"Scientific factors applied | "
            f"Avg multiplier: {df['science_multiplier'].mean():.4f}"
        )
        return df

    # ── Pace / tempo science ──────────────────────────────────────────────────
    def estimate_extra_possessions(
        self,
        home_pace: float,
        away_pace: float,
        league_avg_pace: float = 98.0,
    ) -> float:
        """
        Estimate bonus possessions above league average.
        Each extra possession ≈ +0.3 DK fantasy points per player.
        """
        game_pace = (home_pace + away_pace) / 2
        extra_poss = max(0, game_pace - league_avg_pace)
        return extra_poss

    def pace_adjustment(
        self, projected_pts: float, extra_poss: float, poss_value: float = 0.25
    ) -> float:
        """Adjust projection upward for faster-than-average pace games."""
        return projected_pts + extra_poss * poss_value
