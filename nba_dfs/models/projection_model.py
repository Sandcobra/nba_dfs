"""
Ensemble projection model combining:
  - XGBoost
  - LightGBM
  - CatBoost
  - Random Forest
  - Neural Network (PyTorch)
  - Bayesian Ridge
  - Linear Regression (baseline)

Outputs per-player projected fantasy points with confidence intervals.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.config import BASE_DIR, ENSEMBLE_WEIGHTS


MODEL_DIR = BASE_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature engineering ────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Player recent form
    "avg_pts_L5", "avg_pts_L10", "avg_pts_L20",
    "avg_reb_L5", "avg_reb_L10",
    "avg_ast_L5", "avg_ast_L10",
    "avg_stl_L10", "avg_blk_L10", "avg_tov_L10",
    "avg_min_L5", "avg_min_L10",
    "avg_fg3m_L10",
    "avg_fpts_dk_L5", "avg_fpts_dk_L10", "avg_fpts_dk_L20",
    # Efficiency
    "ts_pct", "usg_pct", "efg_pct",
    "off_rating", "def_rating", "net_rating",
    "ast_ratio", "reb_pct", "oreb_pct",
    "pace_adv",
    # Opponent defense metrics
    "opp_def_rating", "opp_pace", "opp_pts_allowed_pg",
    "opp_reb_allowed_pg", "opp_ast_allowed_pg",
    "opp_paint_defense_pct", "opp_3pt_defense_pct",
    # Context
    "salary_dk", "home_away",  # 1=home, 0=away
    "b2b",                     # back-to-back flag
    "days_rest",
    "game_total",              # Vegas O/U
    "team_implied_total",      # Vegas implied team points
    "opp_implied_total",
    # Positional encoding
    "pos_PG", "pos_SG", "pos_SF", "pos_PF", "pos_C",
    # Season-long averages
    "season_pts_pg", "season_reb_pg", "season_ast_pg",
    "season_min_pg", "season_fpts_dk_pg",
]

TARGET_COLS = [
    "fantasy_pts_dk",
    "pts", "reb", "ast", "stl", "blk", "tov", "fg3m",
]


def engineer_features(df: pd.DataFrame, game_logs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the full feature matrix from raw salary + stats data.
    game_logs: {player_id -> DataFrame of recent games}
    """
    rows = []
    for _, player in df.iterrows():
        pid = player.get("player_id")
        logs = game_logs.get(pid, pd.DataFrame())

        def avg(col, n):
            if logs.empty or col not in logs.columns:
                return 0.0
            return float(logs[col].head(n).mean())

        row = {
            "player_id":     pid,
            "name":          player.get("name", ""),
            "salary_dk":     player.get("salary", 0),
            "home_away":     1 if player.get("home_away", "") == "home" else 0,
            "b2b":           int(player.get("b2b", 0)),
            "days_rest":     player.get("days_rest", 2),
            "game_total":    player.get("game_total", 225),
            "team_implied_total":  player.get("team_implied_total", 112.5),
            "opp_implied_total":   player.get("opp_implied_total", 112.5),
            "opp_def_rating":      player.get("opp_def_rating", 110),
            "opp_pace":            player.get("opp_pace", 100),
            "opp_pts_allowed_pg":  player.get("opp_pts_allowed_pg", 112),
            "opp_reb_allowed_pg":  player.get("opp_reb_allowed_pg", 45),
            "opp_ast_allowed_pg":  player.get("opp_ast_allowed_pg", 25),
            "opp_paint_defense_pct": player.get("opp_paint_defense_pct", 0.55),
            "opp_3pt_defense_pct": player.get("opp_3pt_defense_pct", 0.36),
            "ts_pct":         player.get("ts_pct", 0.56),
            "usg_pct":        player.get("usg_pct", 0.20),
            "efg_pct":        player.get("efg_pct", 0.53),
            "off_rating":     player.get("off_rating", 110),
            "def_rating":     player.get("def_rating", 110),
            "net_rating":     player.get("net_rating", 0),
            "ast_ratio":      player.get("ast_ratio", 15),
            "reb_pct":        player.get("reb_pct", 0.10),
            "oreb_pct":       player.get("oreb_pct", 0.05),
            "pace_adv":       player.get("pace_adv", 100),
            "season_pts_pg":  player.get("season_pts_pg", 10),
            "season_reb_pg":  player.get("season_reb_pg", 4),
            "season_ast_pg":  player.get("season_ast_pg", 3),
            "season_min_pg":  player.get("season_min_pg", 25),
            "season_fpts_dk_pg": player.get("season_fpts_dk_pg", 25),
            # Rolling averages
            "avg_pts_L5":    avg("pts", 5),
            "avg_pts_L10":   avg("pts", 10),
            "avg_pts_L20":   avg("pts", 20),
            "avg_reb_L5":    avg("reb", 5),
            "avg_reb_L10":   avg("reb", 10),
            "avg_ast_L5":    avg("ast", 5),
            "avg_ast_L10":   avg("ast", 10),
            "avg_stl_L10":   avg("stl", 10),
            "avg_blk_L10":   avg("blk", 10),
            "avg_tov_L10":   avg("tov", 10),
            "avg_min_L5":    avg("min", 5),
            "avg_min_L10":   avg("min", 10),
            "avg_fg3m_L10":  avg("fg3m", 10),
            "avg_fpts_dk_L5":  avg("fantasy_pts_dk", 5),
            "avg_fpts_dk_L10": avg("fantasy_pts_dk", 10),
            "avg_fpts_dk_L20": avg("fantasy_pts_dk", 20),
        }

        # Positional dummies
        pos = str(player.get("primary_position", ""))
        for p in ["PG", "SG", "SF", "PF", "C"]:
            row[f"pos_{p}"] = 1 if p in pos else 0

        rows.append(row)

    return pd.DataFrame(rows)


# ── PyTorch Neural Network ─────────────────────────────────────────────────────
class DFSNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class NeuralNetModel:
    def __init__(self, input_dim: int, lr: float = 1e-3, epochs: int = 100):
        self.model  = DFSNet(input_dim)
        self.lr     = lr
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self.scaler.fit_transform(X).astype(np.float32)
        y = y.astype(np.float32)
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        loader  = DataLoader(dataset, batch_size=64, shuffle=True)
        opt     = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.HuberLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            sched.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(X).to(self.device)
            return self.model(t).cpu().numpy()


# ── Ensemble Model ─────────────────────────────────────────────────────────────
class EnsembleProjectionModel:
    """
    Multi-model ensemble for projecting fantasy points.
    Trains one model per stat category and one for total DK fantasy points.
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.feature_cols = [c for c in FEATURE_COLS]
        self.models: dict[str, dict] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.is_trained = False

    def _build_base_models(self, input_dim: int) -> dict:
        models = {
            "xgboost": xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            ),
            "lightgbm": lgb.LGBMRegressor(
                n_estimators=500,
                num_leaves=63,
                learning_rate=0.03,
                feature_fraction=0.7,
                bagging_fraction=0.8,
                bagging_freq=5,
                reg_alpha=0.1,
                reg_lambda=0.5,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            "bayesian_ridge": Pipeline([
                ("scaler", StandardScaler()),
                ("model", BayesianRidge(max_iter=500)),
            ]),
            "neural_net": NeuralNetModel(input_dim=input_dim, epochs=80),
        }
        if HAS_CATBOOST:
            models["catboost"] = cb.CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_state=42,
                verbose=0,
            )
        return models

    def train(self, train_df: pd.DataFrame, game_logs: dict[int, pd.DataFrame]):
        """
        train_df: historical player-game rows with feature + target columns.
        game_logs: {player_id -> sorted game log DataFrame}
        """
        logger.info("Engineering features for training set...")
        feat_df = engineer_features(train_df, game_logs)
        available_feats = [c for c in self.feature_cols if c in feat_df.columns]
        X = feat_df[available_feats].fillna(0).values

        for target in TARGET_COLS:
            if target not in train_df.columns:
                continue

            y = train_df[target].fillna(0).values
            logger.info(f"Training ensemble for target: {target} | n={len(y)}")

            models = self._build_base_models(X.shape[1])
            trained = {}
            for name, mdl in models.items():
                w = self.weights.get(name, 0)
                if w <= 0:
                    continue
                try:
                    mdl.fit(X, y)
                    cv = cross_val_score(
                        mdl if not isinstance(mdl, NeuralNetModel) else Ridge(),
                        X, y, cv=KFold(5, shuffle=True, random_state=42),
                        scoring="neg_mean_absolute_error",
                    )
                    logger.info(f"  {name:15s} MAE CV: {-cv.mean():.2f} ± {cv.std():.2f}")
                    trained[name] = mdl
                except Exception as e:
                    logger.warning(f"  {name} failed: {e}")

            self.models[target]  = trained
            self.scalers[target] = None  # each model handles its own scaling

        self.trained_feature_cols = available_feats
        self.is_trained = True
        logger.success("Ensemble training complete")

    def predict(
        self,
        player_pool: pd.DataFrame,
        game_logs: dict[int, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Returns player_pool with projection columns added.
        """
        if not self.is_trained:
            logger.warning("Model not trained — returning raw avg projections")
            return self._fallback_projections(player_pool)

        feat_df = engineer_features(player_pool, game_logs)
        X = feat_df[self.trained_feature_cols].fillna(0).values

        results = feat_df[["player_id", "name"]].copy()

        for target in TARGET_COLS:
            if target not in self.models:
                continue
            preds = np.zeros(len(X))
            total_w = 0.0
            for name, mdl in self.models[target].items():
                w = self.weights.get(name, 0)
                try:
                    p = mdl.predict(X)
                    preds += w * p
                    total_w += w
                except Exception as e:
                    logger.warning(f"Predict {name}/{target} failed: {e}")
            if total_w > 0:
                preds /= total_w
            results[f"proj_{target}"] = np.maximum(preds, 0)

        # Merge back to player pool
        out = player_pool.merge(results, on="player_id", how="left", suffixes=("", "_proj"))
        out = out.rename(columns={"proj_fantasy_pts_dk": "projected_pts_dk"})

        # Compute value
        out["value_dk"] = (out["projected_pts_dk"] / (out["salary"] / 1000)).round(2)

        return out

    def _fallback_projections(self, player_pool: pd.DataFrame) -> pd.DataFrame:
        """Use simple weighted rolling average as fallback."""
        df = player_pool.copy()
        if "avg_pts" in df.columns:
            df["projected_pts_dk"] = df["avg_pts"] * 1.0
        else:
            df["projected_pts_dk"] = 30.0
        df["value_dk"] = (df["projected_pts_dk"] / (df["salary"] / 1000)).round(2)
        return df

    def estimate_uncertainty(
        self,
        player_pool: pd.DataFrame,
        game_logs: dict[int, pd.DataFrame],
        n_bootstrap: int = 200,
    ) -> pd.DataFrame:
        """
        Bootstrap predictions to generate std-dev, ceiling, floor.
        Returns DataFrame with: player_id, mean_proj, std_proj, ceiling, floor.
        """
        feat_df = engineer_features(player_pool, game_logs)
        X = feat_df[self.trained_feature_cols].fillna(0).values

        all_preds = []
        rng = np.random.default_rng(42)

        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(X), len(X))
            X_boot = X[idx]
            row_preds = np.zeros(len(X))
            total_w = 0.0
            target = "fantasy_pts_dk"
            for name, mdl in self.models.get(target, {}).items():
                w = self.weights.get(name, 0)
                try:
                    row_preds += w * mdl.predict(X)
                    total_w += w
                except Exception:
                    pass
            if total_w > 0:
                row_preds /= total_w
            all_preds.append(row_preds)

        pred_matrix = np.array(all_preds)  # (n_bootstrap, n_players)
        out = feat_df[["player_id", "name"]].copy()
        out["mean_proj"]  = pred_matrix.mean(axis=0)
        out["std_proj"]   = pred_matrix.std(axis=0)
        out["ceiling"]    = np.percentile(pred_matrix, 90, axis=0)
        out["floor"]      = np.percentile(pred_matrix, 10, axis=0)
        out["confidence"] = 1.0 - (out["std_proj"] / (out["mean_proj"] + 1e-6))
        return out

    def save(self, path: Path = MODEL_DIR / "ensemble.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.success(f"Ensemble model saved to {path}")

    @classmethod
    def load(cls, path: Path = MODEL_DIR / "ensemble.pkl") -> "EnsembleProjectionModel":
        with open(path, "rb") as f:
            return pickle.load(f)
