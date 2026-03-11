"""
Data Aggregation Agent.
Orchestrates all data collection for a given slate:
  - NBA API (stats, rosters, game logs, shot charts, tracking)
  - Injury reports
  - Vegas lines
  - DraftKings / FanDuel salary files
  - Back-to-back detection
  - Implied team totals
"""

import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from data.nba_api_client import NBAApiClient
from data.injury_scraper import InjuryScraper
from core.database import get_db
from core.config import (
    THE_ODDS_API_KEY, CURRENT_SEASON, NBA_API_DELAY_SECS,
    CACHE_DIR, MIN_MINUTES_THRESHOLD
)


class DataAgent:
    """
    Fetches and normalizes all data required to build a DFS slate.
    """

    def __init__(self, season: str = CURRENT_SEASON):
        self.season   = season
        self.api      = NBAApiClient(season=season)
        self.scraper  = InjuryScraper()
        self.db       = get_db()
        self._cache   = {}

    # ── Full slate data pipeline ───────────────────────────────────────────────
    def build_slate_data(
        self,
        player_pool: pd.DataFrame,
        slate_date:  Optional[str] = None,
    ) -> dict:
        """
        Runs the full data pipeline for a given slate.
        Returns dict with all data components needed by downstream agents.
        """
        today = slate_date or date.today().isoformat()
        logger.info(f"Building slate data for {today}...")

        # 1. Get today's games
        games_df = self._get_todays_games(today)

        # 2. Vegas lines
        vegas_df = self._get_vegas_lines(today, games_df)

        # 3. Injury reports
        injuries_df = self._get_injuries()

        # 4. Team stats (season + last 10)
        team_stats    = self._get_team_stats()
        team_stats_L10 = self._get_team_stats(last_n=10)

        # 5. Pace stats
        pace_df = self._get_pace_stats()

        # 6. Player game logs (last N games for each player in pool)
        game_logs = self._get_player_game_logs(player_pool)

        # 7. Season stats
        season_stats_df = self._get_season_player_stats()

        # 8. Shot location data
        shot_loc_df    = self._get_shot_locations()
        team_shot_d_df = self._get_team_shot_defense()
        team_shot_d_L10 = self._get_team_shot_defense(last_n=10)

        # 9. Tracking stats (possessions / speed / rebounding)
        tracking_df = self._get_tracking_stats()

        # 10. B2B detection
        schedule_df = self._get_schedule_for_b2b(player_pool)

        # 11. Enrich player pool with all context
        enriched_pool = self._enrich_player_pool(
            player_pool=player_pool,
            games_df=games_df,
            vegas_df=vegas_df,
            injuries_df=injuries_df,
            team_stats=team_stats,
            team_stats_L10=team_stats_L10,
            pace_df=pace_df,
            season_stats_df=season_stats_df,
            shot_loc_df=shot_loc_df,
            team_shot_d_df=team_shot_d_df,
            team_shot_d_L10=team_shot_d_L10,
            tracking_df=tracking_df,
            schedule_df=schedule_df,
        )

        # 12. Save to DB
        self.db.save_vegas_lines(vegas_df) if not vegas_df.empty else None
        self.db.save_injury_report(injuries_df.to_dict("records")) if not injuries_df.empty else None

        logger.success("Slate data build complete")

        return {
            "slate_date":       today,
            "games":            games_df,
            "vegas_lines":      vegas_df,
            "injuries":         injuries_df,
            "team_stats":       team_stats,
            "team_stats_L10":   team_stats_L10,
            "pace":             pace_df,
            "game_logs":        game_logs,
            "season_stats":     season_stats_df,
            "shot_locations":   shot_loc_df,
            "team_shot_d":      team_shot_d_df,
            "team_shot_d_L10":  team_shot_d_L10,
            "tracking":         tracking_df,
            "enriched_pool":    enriched_pool,
        }

    # ── Individual fetch methods ───────────────────────────────────────────────
    def _get_todays_games(self, slate_date: str) -> pd.DataFrame:
        try:
            dk_date = datetime.strptime(slate_date, "%Y-%m-%d").strftime("%m/%d/%Y")
            df = self.api.get_todays_scoreboard(game_date=dk_date)
            logger.info(f"Today's games: {len(df)} found")
            return df
        except Exception as e:
            logger.warning(f"Could not fetch today's games: {e}")
            return pd.DataFrame()

    def _get_vegas_lines(self, slate_date: str, games_df: pd.DataFrame) -> pd.DataFrame:
        try:
            vdf = self.scraper.get_vegas_lines(api_key=THE_ODDS_API_KEY)
            if vdf.empty:
                logger.warning("No Vegas lines available — using defaults")
                return self._default_vegas_lines(games_df)
            logger.info(f"Vegas lines fetched: {len(vdf)} games")
            return vdf
        except Exception as e:
            logger.warning(f"Vegas lines error: {e}")
            return self._default_vegas_lines(games_df)

    def _default_vegas_lines(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Default game totals when Vegas data unavailable."""
        if games_df.empty:
            return pd.DataFrame()
        rows = []
        for _, row in games_df.iterrows():
            rows.append({
                "home_team":  row.get("HOME_TEAM_ABBREVIATION", ""),
                "away_team":  row.get("VISITOR_TEAM_ABBREVIATION", ""),
                "total":      225.5,
                "home_spread": -2.5,
                "bookmaker":  "default",
            })
        return pd.DataFrame(rows)

    def _get_injuries(self) -> pd.DataFrame:
        try:
            df = self.scraper.get_all_injury_data()
            logger.info(f"Injuries fetched: {len(df)} players affected")
            return df
        except Exception as e:
            logger.warning(f"Injury scrape failed: {e}")
            return pd.DataFrame()

    def _get_team_stats(self, last_n: int = 0) -> pd.DataFrame:
        try:
            df = self.api.get_team_stats(measure_type="Base", per_mode="PerGame", last_n=last_n)
            return df
        except Exception as e:
            logger.warning(f"Team stats ({last_n}G) failed: {e}")
            return pd.DataFrame()

    def _get_pace_stats(self) -> pd.DataFrame:
        try:
            return self.api.get_pace_stats()
        except Exception as e:
            logger.warning(f"Pace stats failed: {e}")
            return pd.DataFrame()

    def _get_player_game_logs(
        self, player_pool: pd.DataFrame, n_games: int = 20
    ) -> dict[int, pd.DataFrame]:
        """Fetch recent game logs for all players in the pool."""
        game_logs = {}
        total = len(player_pool)
        for i, (_, player) in enumerate(player_pool.iterrows(), 1):
            pid  = player.get("player_id") or player.get("ID")
            name = player.get("name", "")
            if not pid:
                continue
            pid = int(pid)
            # Check DB cache first
            cached = self.db.get_game_logs(pid, n_games)
            if not cached.empty:
                game_logs[pid] = cached
                continue
            # Fetch from NBA API
            try:
                df = self.api.get_player_game_log(player_id=pid, n_games=n_games)
                if not df.empty:
                    # Compute fantasy points
                    df["fantasy_pts_dk"] = self.api.compute_dk_fantasy_pts(df)
                    df["fantasy_pts_fd"] = self.api.compute_fd_fantasy_pts(df)
                    # Normalize columns
                    col_map = {
                        "PTS": "pts", "REB": "reb", "AST": "ast",
                        "STL": "stl", "BLK": "blk", "TOV": "tov",
                        "FG3M": "fg3m", "MIN": "min",
                        "GAME_DATE": "game_date",
                    }
                    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                    df["player_id"] = pid
                    game_logs[pid] = df
                    self.db.save_game_logs(df)
                if i % 10 == 0:
                    logger.info(f"  Game logs: {i}/{total} players fetched")
            except Exception as e:
                logger.debug(f"  Game log fetch failed for {name}: {e}")
        logger.info(f"Game logs loaded for {len(game_logs)} players")
        return game_logs

    def _get_season_player_stats(self) -> pd.DataFrame:
        try:
            base = self.api.get_league_player_stats(measure_type="Base", per_mode="PerGame")
            adv  = self.api.get_league_player_stats(measure_type="Advanced", per_mode="PerGame")
            if base.empty:
                return pd.DataFrame()
            if not adv.empty:
                df = base.merge(adv, on="PLAYER_ID", suffixes=("", "_adv"))
            else:
                df = base
            return df
        except Exception as e:
            logger.warning(f"Season player stats failed: {e}")
            return pd.DataFrame()

    def _get_shot_locations(self) -> pd.DataFrame:
        try:
            return self.api.get_player_shot_locations()
        except Exception as e:
            logger.warning(f"Shot locations failed: {e}")
            return pd.DataFrame()

    def _get_team_shot_defense(self, last_n: int = 0) -> pd.DataFrame:
        try:
            return self.api.get_team_shot_defense(last_n=last_n)
        except Exception as e:
            logger.warning(f"Team shot defense (last {last_n}) failed: {e}")
            return pd.DataFrame()

    def _get_tracking_stats(self) -> pd.DataFrame:
        try:
            return self.api.get_player_tracking_stats("Possessions")
        except Exception as e:
            logger.warning(f"Tracking stats failed: {e}")
            return pd.DataFrame()

    def _get_schedule_for_b2b(self, player_pool: pd.DataFrame) -> pd.DataFrame:
        try:
            league_log = self.api.get_league_game_log()
            if "TEAM_ABBREVIATION" in league_log.columns and "GAME_DATE" in league_log.columns:
                b2b = self.api.detect_back_to_back(league_log)
                return b2b
        except Exception as e:
            logger.warning(f"B2B detection failed: {e}")
        return pd.DataFrame()

    # ── Enrichment ─────────────────────────────────────────────────────────────
    def _enrich_player_pool(
        self,
        player_pool: pd.DataFrame,
        games_df: pd.DataFrame,
        vegas_df: pd.DataFrame,
        injuries_df: pd.DataFrame,
        team_stats: pd.DataFrame,
        team_stats_L10: pd.DataFrame,
        pace_df: pd.DataFrame,
        season_stats_df: pd.DataFrame,
        shot_loc_df: pd.DataFrame,
        team_shot_d_df: pd.DataFrame,
        team_shot_d_L10: pd.DataFrame,
        tracking_df: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = player_pool.copy()

        # Injury status
        if not injuries_df.empty and "name" in injuries_df.columns:
            inj_map = dict(zip(injuries_df["name"], injuries_df["status"]))
            df["injury_status"] = df["name"].map(inj_map).fillna("ACTIVE")
        else:
            df["injury_status"] = "ACTIVE"

        # Remove OUT players
        df = df[df["injury_status"] != "OUT"].copy()

        # Vegas lines (game totals, implied totals)
        if not vegas_df.empty:
            df = self._merge_vegas(df, vegas_df)

        # Team pace
        if not pace_df.empty and "TEAM_NAME" in pace_df.columns:
            df = df.merge(
                pace_df[["TEAM_NAME", "PACE"]].rename(
                    columns={"PACE": "team_pace", "TEAM_NAME": "_team_name"}
                ),
                left_on="team", right_on="_team_name", how="left"
            ).drop(columns=["_team_name"], errors="ignore")

        # Opponent defense stats
        if not team_stats.empty and "TEAM_ABBREVIATION" in team_stats.columns:
            opp_def = team_stats[["TEAM_ABBREVIATION", "PTS", "REB", "AST",
                                   "OPP_PTS" if "OPP_PTS" in team_stats.columns else "PTS"]].copy()
            opp_def = opp_def.rename(columns={
                "TEAM_ABBREVIATION": "_opp",
                "PTS": "opp_pts_allowed_pg",
                "REB": "opp_reb_allowed_pg",
                "AST": "opp_ast_allowed_pg",
            })
            df = df.merge(opp_def, left_on="opp", right_on="_opp", how="left").drop("_opp", axis=1, errors="ignore")

        # Season stats (TS%, USG%, etc.)
        if not season_stats_df.empty:
            merge_cols = ["PLAYER_ID", "TS_PCT", "USG_PCT", "EFG_PCT",
                          "OFF_RATING", "DEF_RATING", "NET_RATING",
                          "AST_RATIO", "REB_PCT", "OREB_PCT"]
            available  = [c for c in merge_cols if c in season_stats_df.columns]
            if "PLAYER_ID" in available:
                df = df.merge(
                    season_stats_df[available].rename(columns={
                        c.lower(): c.lower() for c in available
                    }),
                    left_on="player_id", right_on="PLAYER_ID", how="left"
                )
                for col in ["TS_PCT", "USG_PCT", "EFG_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING"]:
                    if col in df.columns:
                        df.rename(columns={col: col.lower()}, inplace=True)

        # B2B flag
        if not schedule_df.empty and "B2B" in schedule_df.columns:
            b2b_map = schedule_df.groupby("TEAM_ABBREVIATION")["B2B"].last().to_dict()
            df["b2b"] = df["team"].map(b2b_map).fillna(False).astype(int)
        else:
            df["b2b"] = 0

        df["days_rest"] = df.get("b2b", 0).apply(lambda x: 1 if x else 2)

        # Fill defaults
        df["game_total"]         = df.get("game_total", 225.5)
        df["team_implied_total"] = df.get("team_implied_total", 112.75)
        df["opp_implied_total"]  = df.get("opp_implied_total", 112.75)
        df["ts_pct"]             = df.get("ts_pct", 0.56)
        df["usg_pct"]            = df.get("usg_pct", 0.20)

        logger.info(f"Enriched pool: {len(df)} players after OUT removal")
        return df.reset_index(drop=True)

    def _merge_vegas(self, df: pd.DataFrame, vegas: pd.DataFrame) -> pd.DataFrame:
        if "home_team" not in vegas.columns or "total" not in vegas.columns:
            return df

        game_map = {}
        for _, vrow in vegas.iterrows():
            home = vrow.get("home_team", "")
            away = vrow.get("away_team", "")
            tot  = vrow.get("total", 225.5)
            spr  = vrow.get("home_spread", -2.0)
            if home and tot:
                home_impl = (tot / 2) - (spr / 2)
                away_impl = tot - home_impl
                game_map[home] = {"game_total": tot, "team_implied_total": home_impl, "opp_implied_total": away_impl}
                game_map[away] = {"game_total": tot, "team_implied_total": away_impl, "opp_implied_total": home_impl}

        for team, vals in game_map.items():
            mask = df["team"] == team
            for col, val in vals.items():
                df.loc[mask, col] = val
        return df

    # ── Historical data fetch for training ────────────────────────────────────
    def fetch_historical_data(self, n_seasons: int = 2) -> pd.DataFrame:
        """
        Fetch historical player game logs for ML model training.
        Combines multiple seasons of data.
        """
        seasons = []
        base_year = int(CURRENT_SEASON[:4])
        for i in range(n_seasons):
            yr   = base_year - i
            seas = f"{yr}-{str(yr+1)[2:]}"
            logger.info(f"Fetching historical data for {seas}...")
            try:
                self.api.season = seas
                gl = self.api.get_league_game_log()
                if not gl.empty:
                    gl["SEASON"] = seas
                    gl["fantasy_pts_dk"] = self.api.compute_dk_fantasy_pts(gl)
                    seasons.append(gl)
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Historical fetch for {seas} failed: {e}")

        self.api.season = CURRENT_SEASON
        if not seasons:
            return pd.DataFrame()
        combined = pd.concat(seasons, ignore_index=True)
        logger.success(f"Historical data: {len(combined)} player-game rows")
        return combined
