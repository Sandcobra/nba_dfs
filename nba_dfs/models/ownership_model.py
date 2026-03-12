"""
Ownership projection model.
Predicts DFS public ownership % to identify:
  - Low-owned plays for leverage (GPP)
  - High-owned chalk for cash games
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


OWNERSHIP_FEATURES = [
    "salary_dk",
    "projected_pts_dk",
    "value_dk",
    "avg_pts",
    "injury_status_numeric",  # 0=active, 1=gtd, 2=q, 3=d, 4=out
    "game_total",
    "team_implied_total",
    "b2b",
    "days_rest",
    "home_away",
    "is_game_time_decision",
    "salary_percentile",      # where player ranks in salary
    "proj_percentile",        # where player ranks in projections
    "pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C",
]

INJURY_STATUS_MAP = {
    "ACTIVE":       0,
    "PROBABLE":     0,
    "GTD":          1,
    "QUESTIONABLE": 2,
    "DOUBTFUL":     3,
    "OUT":          4,
    "UNKNOWN":      0,
}


class OwnershipModel:
    """
    Gradient boosting model for projecting player ownership percentages.
    Trained on historical ownership data from DFS contests.
    """

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )),
        ])
        self.is_trained = False
        self._feature_cols: list[str] = []

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame()
        feat["salary_dk"]        = df.get("salary", 0)
        feat["projected_pts_dk"] = df.get("projected_pts_dk", 30)
        feat["value_dk"]         = df.get("value_dk", 5)
        feat["avg_pts"]          = df.get("avg_pts", 25)
        feat["game_total"]       = df.get("game_total", 225)
        feat["team_implied_total"] = df.get("team_implied_total", 112.5)
        feat["b2b"]              = df.get("b2b", 0).astype(int)
        feat["days_rest"]        = df.get("days_rest", 2)
        feat["home_away"]        = pd.Series(df.get("home_away", pd.Series(["home"]*len(df))), index=df.index).eq("home").astype(int)
        feat["is_game_time_decision"] = (
            df.get("injury_status", "ACTIVE").isin(["GTD", "QUESTIONABLE"])
        ).astype(int)
        feat["injury_status_numeric"] = (
            df.get("injury_status", "ACTIVE").map(INJURY_STATUS_MAP).fillna(0)
        )

        # Relative rank features
        feat["salary_percentile"] = (
            feat["salary_dk"].rank(pct=True)
        )
        feat["proj_percentile"] = (
            feat["projected_pts_dk"].rank(pct=True)
        )

        # Positional dummies
        pos = df.get("primary_position", pd.Series(["UTIL"] * len(df)))
        for p in ["PG", "SG", "SF", "PF", "C"]:
            feat[f"pos_{p}"] = (pos == p).astype(int)

        return feat.fillna(0)

    def train(self, df: pd.DataFrame):
        """
        df must have 'ownership_pct' column plus feature columns.
        """
        feat = self._build_features(df)
        y = df["ownership_pct"].values

        available = [c for c in OWNERSHIP_FEATURES if c in feat.columns]
        self._feature_cols = available
        X = feat[available].values

        self.model.fit(X, y)
        self.is_trained = True
        logger.success(f"Ownership model trained on {len(y)} samples")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Returns projected ownership % for each player in df."""
        feat = self._build_features(df)

        if not self.is_trained:
            # Heuristic fallback
            logger.warning("Ownership model not trained — using heuristic")
            return self._heuristic(df)

        available = [c for c in self._feature_cols if c in feat.columns]
        X = feat[available].fillna(0).values
        preds = self.model.predict(X)
        return pd.Series(np.clip(preds, 0.5, 80.0), index=df.index)

    @staticmethod
    def _heuristic(df: pd.DataFrame) -> pd.Series:
        """
        Simple heuristic based on value rank and salary.
        Players with high value + low salary → low ownership (GPP leverage).
        Elite players → high ownership.
        """
        sal_norm  = (df.get("salary", 7000) - 3500) / (12000 - 3500)
        val_norm  = df.get("value_dk", 5).clip(0, 10) / 10.0
        proj_norm = df.get("projected_pts_dk", 30).clip(0, 70) / 70.0
        # High salary + high projection = high ownership
        score = 0.4 * sal_norm + 0.4 * proj_norm + 0.2 * val_norm
        return (score * 40 + 2).clip(1, 60).round(1)

    def compute_leverage(
        self,
        projected_df: pd.DataFrame,
        actual_ownership: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Leverage = My Lineup Ownership - Field Ownership
        Positive leverage = contrarian play.
        """
        df = projected_df.copy()
        df["proj_ownership"] = self.predict(df).values

        if actual_ownership is not None:
            df["actual_ownership"] = actual_ownership.values
            df["leverage"] = df["proj_ownership"] - df["actual_ownership"]
        else:
            df["leverage"] = 0.0

        return df

    def gpp_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Composite GPP attractiveness score:
          - High projection
          - Low ownership
          - High ceiling
        """
        proj  = df.get("projected_pts_dk", 30).clip(0, 70)
        own   = df.get("proj_ownership", 20).clip(1, 80)
        ceil_ = df.get("ceiling", proj * 1.3).clip(0, 90)

        # Normalize each component
        proj_n  = (proj  - proj.min())  / (proj.max()  - proj.min()  + 1e-9)
        own_n   = 1 - (own   - own.min())  / (own.max()  - own.min()  + 1e-9)
        ceil_n  = (ceil_ - ceil_.min()) / (ceil_.max() - ceil_.min() + 1e-9)

        return (0.45 * proj_n + 0.30 * own_n + 0.25 * ceil_n).round(4)

    def cash_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Cash game attractiveness:
          - High floor
          - High minutes / consistency
          - Avoid GTD players
        """
        floor_ = df.get("floor", df.get("projected_pts_dk", 25) * 0.7)
        proj   = df.get("projected_pts_dk", 30)
        inj    = (df.get("injury_status", "ACTIVE") == "ACTIVE").astype(float)

        floor_n = (floor_ - floor_.min()) / (floor_.max() - floor_.min() + 1e-9)
        proj_n  = (proj   - proj.min())   / (proj.max()   - proj.min()   + 1e-9)

        return (0.55 * floor_n + 0.35 * proj_n + 0.10 * inj).round(4)
