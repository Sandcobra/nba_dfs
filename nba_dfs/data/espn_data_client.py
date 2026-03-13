"""Utilities for working with cached ESPN data.

Loads roster mappings, player game logs, and schedule data that
were previously scraped from ESPN endpoints so that the DFS pipeline
can operate without touching the NBA Stats API.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.config import CACHE_DIR, CURRENT_SEASON
from utils.helpers import normalize_name, compute_dk_fantasy_pts


class ESPNDataClient:
    def __init__(
        self,
        cache_dir: str | Path = Path(__file__).parent.parent.parent / "cache" / "espn",
        season: str = CURRENT_SEASON,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.season = season
        self._season_year = self._resolve_season_year(season)

        self._name_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._id_to_name: dict[str, str] = {}
        self._id_to_team: dict[str, str] = {}
        self._player_log_cache: dict[str, pd.DataFrame] = {}
        self._team_game_totals: Optional[pd.DataFrame] = None
        self._schedule_cache: dict[str, list[date]] = {}

        self.roster_mapping = self.load_roster_mapping()

    # ---------------------------------------------------------------------
    def _resolve_season_year(self, season: str) -> str:
        try:
            start_year = int(season.split("-")[0])
            return str(start_year + 1)
        except Exception:
            return season

    def load_roster_mapping(self) -> dict[str, str]:
        """Load {normalized_player_name: espn_id} from roster cache files."""
        mapping: dict[str, str] = {}
        pattern = f"roster_*_{self._season_year}.json"
        files = sorted(self.cache_dir.glob(pattern))
        if not files:
            logger.warning("No ESPN roster files found at %s", self.cache_dir)
            return mapping

        for roster_file in files:
            parts = roster_file.stem.split("_")
            team = parts[1].upper() if len(parts) > 2 else ""
            try:
                data = json.loads(roster_file.read_text())
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", roster_file.name, exc)
                continue

            for raw_name, espn_id in data.items():
                key = normalize_name(raw_name)
                entry = {
                    "espn_id": str(espn_id),
                    "team": team,
                    "display_name": raw_name,
                    "name_key": key,
                }
                mapping[key] = str(espn_id)
                self._name_index[key].append(entry)
                self._id_to_name[str(espn_id)] = raw_name
                self._id_to_team[str(espn_id)] = team

        logger.info("Loaded %d ESPN roster entries", len(mapping))
        return mapping

    # ------------------------------------------------------------------ logs
    def get_player_game_logs(
        self,
        player_name: str,
        team: Optional[str] = None,
        n_games: int = 20,
    ) -> pd.DataFrame:
        """Return the most recent n games for the requested player."""
        entry = self._resolve_player_entry(player_name, team)
        if not entry:
            logger.debug("No ESPN cache for %s (%s)", player_name, team)
            return pd.DataFrame()

        df = self._load_player_log_df(entry)
        if df.empty:
            return df
        return df.head(n_games).copy()

    def get_all_game_logs(self) -> dict[str, pd.DataFrame]:
        """Load every cached game log keyed by normalized player name."""
        result: dict[str, pd.DataFrame] = {}
        for name_key, entries in self._name_index.items():
            if not entries:
                continue
            df = self._load_player_log_df(entries[0])
            if not df.empty:
                result[name_key] = df.copy()
        return result

    def compute_dk_fantasy_pts(self, row: dict | pd.Series) -> float:
        return compute_dk_fantasy_pts(row)

    # ----------------------------------------------------------- aggregations
    def get_team_stats_from_gamelogs(self, last_n: int = 0) -> pd.DataFrame:
        team_games = self._get_team_game_totals()
        if team_games.empty:
            return pd.DataFrame()

        df = team_games.sort_values("game_date", ascending=False)
        if last_n > 0:
            df = df.groupby("team").head(last_n)

        agg = (
            df.groupby("team")
            .agg({
                "pts": "mean",
                "reb": "mean",
                "ast": "mean",
                "stl": "mean",
                "blk": "mean",
                "fantasy_pts_dk": "mean",
                "game_date": "count",
            })
            .reset_index()
        )
        agg = agg.rename(
            columns={
                "team": "TEAM_ABBREVIATION",
                "pts": "PTS",
                "reb": "REB",
                "ast": "AST",
                "stl": "STL",
                "blk": "BLK",
                "fantasy_pts_dk": "DK_PTS",
                "game_date": "GAMES_PLAYED",
            }
        )
        return agg

    def get_player_recent_form(
        self,
        player_name: str,
        team: Optional[str] = None,
        n_recent: int = 5,
    ) -> dict[str, float]:
        logs = self.get_player_game_logs(player_name, team=team, n_games=max(20, n_recent))
        if logs.empty:
            return {}
        recent = logs.head(n_recent)["fantasy_pts_dk"].mean()
        season = logs["fantasy_pts_dk"].mean()
        ratio = recent / season if season else 0
        return {
            "avg_recent": round(float(recent), 2),
            "avg_season": round(float(season), 2),
            "form_ratio": round(float(ratio), 3) if season else 0.0,
        }

    def compute_recency_weighted_projection(
        self,
        player_name: str,
        n_recent: int = 5,
        weight_recent: float = 3.0,
    ) -> dict:
        """Compute recency-weighted DK fantasy projection for a player."""
        df = self.get_player_game_logs(player_name, n_games=25)
        if df.empty or "fantasy_pts_dk" not in df.columns:
            return {}
        pts = df["fantasy_pts_dk"].dropna().values
        if len(pts) == 0:
            return {}
        n_recent = min(n_recent, len(pts))
        recent = pts[:n_recent]
        older = pts[n_recent:]
        recent_avg = float(np.mean(recent))
        season_avg = float(np.mean(pts))
        # Weighted average: recent games at weight_recent, older at 1.0
        weights = [weight_recent] * len(recent) + [1.0] * len(older)
        weighted_proj = float(np.average(pts[: len(recent) + len(older)], weights=weights[: len(pts)]))
        form_ratio = recent_avg / season_avg if season_avg > 0 else 1.0
        return {
            "weighted_proj": round(weighted_proj, 2),
            "form_ratio": round(form_ratio, 3),
            "recent5_avg": round(recent_avg, 2),
            "season_avg": round(season_avg, 2),
            "games_played": len(pts),
            "is_hot": form_ratio > 1.15,
            "is_cold": form_ratio < 0.85,
        }

    # --------------------------------------------------------- explosion profile
    def compute_explosion_profile(
        self,
        player_name: str,
        team: Optional[str] = None,
        boom_threshold: float = 1.8,   # game must be >= boom_threshold x season_avg
        recent_n: int = 10,            # games for variance window
        min_games: int = 8,            # minimum games needed for reliable stats
    ) -> dict:
        """
        Compute GPP ceiling signals from a player's full game log.

        Returns
        -------
        boom_rate       : float  — fraction of games where dk_score >= 1.8x season_avg
                          This is the historical probability of a "monster" game.
        variance_ratio  : float  — recent_std / season_std
                          > 1.3 means the player is in "volatile" mode (expanding variance)
        is_volatile     : bool   — variance_ratio > 1.3
        recent_std      : float  — std dev of last recent_n games
        season_std      : float  — std dev of full season
        season_avg      : float  — full-season DK average
        games_played    : int

        Usage in projections
        --------------------
        A player with boom_rate=0.20 and is_volatile=True at a cheap salary is
        a high-leverage GPP play even if their average projection is modest.
        The game environment multiplier (pace × game_total) further amplifies
        this signal when the environment favors counting-stat explosions.
        """
        df = self.get_player_game_logs(player_name, team=team, n_games=82)
        if df.empty or "fantasy_pts_dk" not in df.columns:
            return {}

        pts = df["fantasy_pts_dk"].dropna().values
        if len(pts) < min_games:
            return {}

        season_avg = float(np.mean(pts))
        season_std = float(np.std(pts)) if len(pts) > 1 else 0.0
        if season_avg <= 0:
            return {}

        # Boom rate: fraction of games >= boom_threshold × season_avg
        boom_floor = season_avg * boom_threshold
        boom_rate  = float(np.mean(pts >= boom_floor))

        # Variance expansion: recent vs season
        recent_pts = pts[:min(recent_n, len(pts))]
        recent_std = float(np.std(recent_pts)) if len(recent_pts) > 1 else season_std
        variance_ratio = (recent_std / season_std) if season_std > 0 else 1.0
        is_volatile    = variance_ratio > 1.3

        return {
            "boom_rate":      round(boom_rate, 3),
            "variance_ratio": round(variance_ratio, 3),
            "is_volatile":    is_volatile,
            "recent_std":     round(recent_std, 2),
            "season_std":     round(season_std, 2),
            "season_avg":     round(season_avg, 2),
            "games_played":   int(len(pts)),
        }

    # ------------------------------------------------------------- schedules
    def get_b2b_dataframe(self, slate_date: str) -> pd.DataFrame:
        target = datetime.strptime(slate_date, "%Y-%m-%d").date()
        rows: list[dict[str, Any]] = []
        teams = sorted(set(self._id_to_team.values()))
        for team in teams:
            dates = self._load_team_schedule(team)
            if target not in dates:
                continue
            idx = dates.index(target)
            prev = dates[idx - 1] if idx > 0 else None
            days_rest = (target - prev).days if prev else None
            rows.append(
                {
                    "TEAM_ABBREVIATION": team,
                    "GAME_DATE": target,
                    "B2B": bool(days_rest == 1) if days_rest is not None else False,
                    "DAYS_REST": days_rest if days_rest is not None else 3,
                }
            )
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- helpers
    def _resolve_player_entry(self, player_name: str, team: Optional[str]) -> Optional[dict[str, Any]]:
        key = normalize_name(player_name)
        entry = self._match_entry(key, team)
        if entry:
            return entry
        stripped = self._strip_suffix(key)
        if stripped != key:
            entry = self._match_entry(stripped, team)
            if entry:
                return entry
        return self._match_entry(key, None)

    def _match_entry(self, key: str, team: Optional[str]) -> Optional[dict[str, Any]]:
        entries = self._name_index.get(key)
        if not entries:
            return None
        if team:
            team = str(team).upper() if team else None
            for entry in entries:
                if entry.get("team") == team:
                    return entry
        return entries[0]

    def _strip_suffix(self, key: str) -> str:
        suffixes = (" jr", " sr", " ii", " iii", " iv")
        for suffix in suffixes:
            if key.endswith(suffix):
                return key[: -len(suffix)].strip()
        return key

    def _load_player_log_df(self, entry: dict[str, Any]) -> pd.DataFrame:
        espn_id = entry["espn_id"]
        if espn_id in self._player_log_cache:
            return self._player_log_cache[espn_id]

        path = self.cache_dir / f"gamelog_{espn_id}_{self._season_year}.json"
        if not path.exists():
            logger.debug("Missing ESPN gamelog cache for %s", espn_id)
            self._player_log_cache[espn_id] = pd.DataFrame()
            return self._player_log_cache[espn_id]

        try:
            raw = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path.name, exc)
            raw = {}

        df = self._gamelog_dict_to_df(raw, entry)
        self._player_log_cache[espn_id] = df
        return df

    def _gamelog_dict_to_df(self, raw: dict[str, Any], entry: dict[str, Any]) -> pd.DataFrame:
        if not raw or "date" not in raw:
            return pd.DataFrame()

        records: list[dict[str, Any]] = []
        for idx, date_str in raw.get("date", {}).items():
            game_dt = self._safe_parse_date(date_str)
            rec = {
                "game_date": game_dt,
                "date": date_str,
                "pts": float(raw.get("pts", {}).get(idx, 0) or 0),
                "reb": float(raw.get("trb", {}).get(idx, 0) or 0),
                "ast": float(raw.get("ast", {}).get(idx, 0) or 0),
                "stl": float(raw.get("stl", {}).get(idx, 0) or 0),
                "blk": float(raw.get("blk", {}).get(idx, 0) or 0),
                "tov": float(raw.get("tov", {}).get(idx, 0) or 0),
                "fg3m": float(raw.get("fg3", {}).get(idx, 0) or 0),
                "min": float(raw.get("mp", {}).get(idx, 0) or 0),
                "team": entry.get("team"),
                "espn_id": entry.get("espn_id"),
                "player_name": entry.get("display_name"),
            }
            rec["fantasy_pts_dk"] = self.compute_dk_fantasy_pts(rec)
            records.append(rec)

        df = pd.DataFrame(records)
        if df.empty:
            return df
        df = df.sort_values("game_date", ascending=False).reset_index(drop=True)
        return df

    def _safe_parse_date(self, value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return pd.to_datetime(value, errors="coerce")

    def _get_team_game_totals(self) -> pd.DataFrame:
        if self._team_game_totals is not None:
            return self._team_game_totals

        records: list[pd.DataFrame] = []
        for espn_id, team in self._id_to_team.items():
            df = self._player_log_cache.get(espn_id)
            if df is None:
                entry = {"espn_id": espn_id, "team": team, "display_name": self._id_to_name.get(espn_id, "")}
                df = self._load_player_log_df(entry)
            if df.empty:
                continue
            records.append(df[[
                "team", "game_date", "pts", "reb", "ast", "stl", "blk", "fantasy_pts_dk"
            ]].copy())

        if not records:
            self._team_game_totals = pd.DataFrame()
            return self._team_game_totals

        combined = pd.concat(records, ignore_index=True)
        combined = combined.dropna(subset=["game_date", "team"])
        combined["game_date"] = pd.to_datetime(combined["game_date"])
        grouped = (
            combined.groupby(["team", "game_date"])
            .sum(numeric_only=True)
            .reset_index()
        )
        self._team_game_totals = grouped
        return grouped

    def _load_team_schedule(self, team: str) -> list[date]:
        if team in self._schedule_cache:
            return self._schedule_cache[team]

        path = self.cache_dir / f"schedule_{team}_{self._season_year}.json"
        if not path.exists():
            self._schedule_cache[team] = []
            return self._schedule_cache[team]

        try:
            dates = json.loads(path.read_text())
            parsed = sorted(datetime.fromisoformat(d).date() for d in dates)
        except Exception as exc:
            logger.warning("Failed to parse schedule %s: %s", path.name, exc)
            parsed = []
        self._schedule_cache[team] = parsed
        return parsed
