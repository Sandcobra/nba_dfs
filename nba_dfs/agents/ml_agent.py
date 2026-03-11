"""
Machine Learning Agent.
Coordinates model training, prediction, uncertainty estimation,
feature importance analysis, and model evaluation.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

from models.projection_model import EnsembleProjectionModel, engineer_features
from models.ownership_model import OwnershipModel
from core.config import BASE_DIR, ENSEMBLE_WEIGHTS


MODEL_DIR = BASE_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class MLAgent:
    """
    Manages all ML models for DFS projection:
    - Ensemble projection (XGBoost, LightGBM, CatBoost, RF, NN, Bayesian)
    - Ownership model
    - Stat-specific models (minutes, pts, reb, ast, stl, blk, tov, fg3m)
    """

    def __init__(self):
        self.ensemble   = EnsembleProjectionModel(weights=ENSEMBLE_WEIGHTS)
        self.ownership  = OwnershipModel()
        self._is_ready  = False
        self._try_load()

    def _try_load(self):
        """Attempt to load pre-trained models."""
        ep = MODEL_DIR / "ensemble.pkl"
        op = MODEL_DIR / "ownership.pkl"
        if ep.exists():
            try:
                self.ensemble = EnsembleProjectionModel.load(ep)
                logger.success("Loaded pre-trained ensemble model")
                self._is_ready = True
            except Exception as e:
                logger.warning(f"Could not load ensemble model: {e}")
        if op.exists():
            try:
                with open(op, "rb") as f:
                    self.ownership = pickle.load(f)
                logger.success("Loaded pre-trained ownership model")
            except Exception as e:
                logger.warning(f"Could not load ownership model: {e}")

    # ── Training pipeline ──────────────────────────────────────────────────────
    def train(
        self,
        historical_df: pd.DataFrame,
        game_logs:     dict[int, pd.DataFrame],
        ownership_df:  Optional[pd.DataFrame] = None,
    ):
        """
        Train all ML models on historical data.

        historical_df: wide-format DataFrame with player features + targets
                       Must contain 'fantasy_pts_dk' and stat columns.
        game_logs:     {player_id -> recent game log DataFrame}
        ownership_df:  historical ownership data with 'ownership_pct' column
        """
        logger.info("=" * 60)
        logger.info("Starting ML training pipeline")
        logger.info("=" * 60)

        # Train projection ensemble
        self.ensemble.train(historical_df, game_logs)
        self.ensemble.save(MODEL_DIR / "ensemble.pkl")

        # Train ownership model
        if ownership_df is not None and "ownership_pct" in ownership_df.columns:
            self.ownership.train(ownership_df)
            with open(MODEL_DIR / "ownership.pkl", "wb") as f:
                pickle.dump(self.ownership, f)
            logger.success("Ownership model saved")

        self._is_ready = True
        logger.success("ML training complete")

    # ── Prediction pipeline ────────────────────────────────────────────────────
    def predict(
        self,
        player_pool: pd.DataFrame,
        game_logs:   dict[int, pd.DataFrame],
        apply_bayesian_update: bool = True,
        apply_regression:      bool = True,
    ) -> pd.DataFrame:
        """
        Generate full projections for all players in pool.
        Returns enriched player_pool with all projection columns.
        """
        from agents.math_agent import MathAgent
        math = MathAgent()

        df = player_pool.copy()

        # Step 1: Ensemble projections
        logger.info("Running ensemble projections...")
        df = self.ensemble.predict(df, game_logs)

        # Step 2: Bayesian update with recent games
        if apply_bayesian_update:
            logger.info("Applying Bayesian updates...")
            df = math.bayesian_update_projections(df, game_logs)

        # Step 3: Regression to mean
        if apply_regression:
            logger.info("Applying regression to mean...")
            df = math.apply_regression_to_projections(df, game_logs)

        # Step 4: Uncertainty estimates
        logger.info("Estimating projection uncertainty...")
        if self.ensemble.is_trained:
            uncertainty = self.ensemble.estimate_uncertainty(df, game_logs, n_bootstrap=100)
            df = df.merge(
                uncertainty[["player_id", "std_proj", "ceiling", "floor", "confidence"]],
                on="player_id", how="left"
            )
        else:
            df["std_proj"]  = df["projected_pts_dk"] * 0.28
            df["ceiling"]   = df["projected_pts_dk"] * 1.35
            df["floor"]     = df["projected_pts_dk"] * 0.65
            df["confidence"] = 0.50

        # Step 5: Ownership projections
        logger.info("Projecting ownership...")
        df["proj_ownership"] = self.ownership.predict(df).values

        # Step 6: Monte Carlo
        logger.info("Running Monte Carlo simulations...")
        mc_results = math.monte_carlo_projections(df)
        df = df.merge(
            mc_results[["player_id", "p_bust", "p_ceiling", "p_double_double",
                        "mean_sim", "median_sim"]].rename(
                columns={"mean_sim": "mc_mean", "median_sim": "mc_median"}
            ),
            on="player_id", how="left"
        )

        # Step 7: Compute scoring metrics
        df["gpp_score"]  = self.ownership.gpp_score(df)
        df["cash_score"] = self.ownership.cash_score(df)

        # Step 8: Value metrics
        df["value_dk"] = (
            df["projected_pts_dk"] / (df["salary"] / 1000)
        ).round(2).fillna(0)

        # Step 9: Triple-double probability
        df["p_triple_double"] = df.apply(
            lambda r: math.estimate_triple_double_probability(
                r.get("proj_pts",  r.get("projected_pts_dk", 10) * 0.5),
                r.get("proj_reb", 4),
                r.get("proj_ast", 3),
                r.get("proj_stl", 1),
                r.get("proj_blk", 0.5),
            ),
            axis=1,
        ).round(4)

        logger.success(f"Projections complete: {len(df)} players")
        return df.sort_values("projected_pts_dk", ascending=False).reset_index(drop=True)

    # ── Model evaluation ───────────────────────────────────────────────────────
    def evaluate(
        self,
        test_df:   pd.DataFrame,
        game_logs: dict[int, pd.DataFrame],
    ) -> dict:
        """Evaluate model on a held-out test set."""
        preds = self.predict(test_df, game_logs, apply_bayesian_update=False)
        y_true = test_df["fantasy_pts_dk"].values
        y_pred = preds["projected_pts_dk"].values

        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

        metrics = {
            "mae":  round(mae, 3),
            "rmse": round(rmse, 3),
            "r2":   round(r2, 3),
            "n":    len(y_true),
        }
        logger.info(f"Model evaluation: MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")
        return metrics

    # ── Feature importance ────────────────────────────────────────────────────
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return top N features by importance from XGBoost model."""
        target = "fantasy_pts_dk"
        models = self.ensemble.models.get(target, {})
        xgb_model = models.get("xgboost")

        if xgb_model is None:
            return pd.DataFrame()

        feat_names = self.ensemble.trained_feature_cols
        importances = xgb_model.feature_importances_

        df = pd.DataFrame({
            "feature":    feat_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_n)

        return df

    # ── Injury-adjusted re-projections ────────────────────────────────────────
    def reproject_after_injury(
        self,
        player_pool:   pd.DataFrame,
        game_logs:     dict[int, pd.DataFrame],
        scratched_name: str,
        scratched_team: str,
    ) -> pd.DataFrame:
        """
        After a player is scratched, reproject remaining teammates
        to account for usage absorption.
        """
        from agents.game_theory_agent import GameTheoryAgent
        gt = GameTheoryAgent()

        # Build a fake injury record for the game_theory agent
        injury_record = pd.DataFrame([{
            "name":   scratched_name,
            "team":   scratched_team,
            "status": "OUT",
        }])

        adjusted_pool = gt.adjust_for_injuries(player_pool, injury_record)
        return self.predict(adjusted_pool, game_logs, apply_bayesian_update=False)

    # ── Narrative / insight generation ───────────────────────────────────────
    def generate_insights(self, projections: pd.DataFrame) -> list[str]:
        """
        Generate human-readable insights about the slate.
        """
        insights = []

        if projections.empty:
            return ["No projections available."]

        # Top values
        top_vals = projections.nlargest(3, "value_dk")
        for _, r in top_vals.iterrows():
            insights.append(
                f"VALUE PLAY: {r['name']} ({r.get('team','?')}) "
                f"${r['salary']:,} | {r['projected_pts_dk']:.1f} proj | "
                f"{r['value_dk']:.2f}x value"
            )

        # Low-owned studs
        low_own = projections[
            (projections["proj_ownership"] < 10) &
            (projections["projected_pts_dk"] > projections["projected_pts_dk"].quantile(0.7))
        ].nlargest(3, "projected_pts_dk")
        for _, r in low_own.iterrows():
            insights.append(
                f"LEVERAGE PLAY: {r['name']} — only {r['proj_ownership']:.1f}% ownership "
                f"but {r['projected_pts_dk']:.1f} pts projected"
            )

        # Injury beneficiaries
        boosted = projections[projections.get("injury_boosted", False) == True] if "injury_boosted" in projections.columns else pd.DataFrame()
        for _, r in boosted.head(3).iterrows():
            insights.append(
                f"INJURY BENEFICIARY: {r['name']} ({r.get('team','?')}) "
                f"boosted to {r['projected_pts_dk']:.1f} pts due to teammate absence"
            )

        # High ceiling plays
        top_ceiling = projections.nlargest(3, "ceiling")
        for _, r in top_ceiling.iterrows():
            insights.append(
                f"CEILING PLAY: {r['name']} — {r.get('ceiling', 0):.1f} DK pt ceiling | "
                f"{r.get('p_ceiling', 0)*100:.1f}% chance of 50+ pts"
            )

        return insights
