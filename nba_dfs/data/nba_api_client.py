"""
NBA API client - thin wrapper with rate-limiting, retries, and caching.
Pulls: game logs, player dashboards, shot locations, team defense, pace,
       play-by-play, lineups, hustle stats, tracking stats.
"""

import time
import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from nba_api.stats.endpoints import (
    leaguegamefinder,
    playerdashboardbyclutch,
    playergamelog,
    playervsplayer,
    leaguedashplayershotlocations,
    leaguedashteamshotlocations,
    leaguedashptteamdefend,
    leaguedashplayerstats,
    leaguedashteamstats,
    leaguedashptstats,
    boxscoreadvancedv2,
    boxscoresummaryv2,
    teamgamelog,
    commonteamroster,
    leaguegamelog,
    playbyplayv2,
    hustlestatsboxscore,
)

# These endpoints may be missing in newer nba_api versions
try:
    from nba_api.stats.endpoints import teamdashboardbygamesplits
except ImportError:
    teamdashboardbygamesplits = None

try:
    from nba_api.stats.endpoints import scoreboard
except ImportError:
    scoreboard = None
from nba_api.stats.static import players as static_players, teams as static_teams

from core.config import NBA_API_HEADERS, NBA_API_DELAY_SECS, CACHE_DIR, CURRENT_SEASON


def _delay():
    time.sleep(NBA_API_DELAY_SECS)


class NBAApiClient:
    """Stateless client; all methods return DataFrames."""

    def __init__(self, season: str = CURRENT_SEASON, cache_ttl_hours: int = 6):
        self.season = season
        self.cache_dir = CACHE_DIR / "nba_api"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = cache_ttl_hours
        self._player_map: dict[str, int] = {}
        self._team_map:   dict[str, int] = {}

    # ── Static lookups ─────────────────────────────────────────────────────────
    def get_all_active_players(self) -> pd.DataFrame:
        pl = static_players.get_active_players()
        df = pd.DataFrame(pl)
        df.columns = [c.upper() for c in df.columns]
        return df  # columns: ID, FULL_NAME, FIRST_NAME, LAST_NAME, IS_ACTIVE

    def get_all_teams(self) -> pd.DataFrame:
        tm = static_teams.get_teams()
        return pd.DataFrame(tm)

    def player_name_to_id(self, name: str) -> Optional[int]:
        """
        Resolve a player display name to their NBA Stats player ID.

        Tries in order:
          1. Exact lower-case match (fast path)
          2. Last-name + first-initial match  ("P.J. Washington" vs "PJ Washington")
          3. Punctuation-stripped match       ("Gary Trent Jr." vs "Gary Trent Jr")
          4. difflib fuzzy match at 0.82 cutoff
        """
        if not self._player_map:
            df = self.get_all_active_players()
            self._player_map = dict(zip(df["FULL_NAME"].str.lower(), df["ID"]))

        key = name.lower().strip()

        # 1. Exact
        if key in self._player_map:
            return self._player_map[key]

        # 2. Strip punctuation and try again (handles "P.J." → "PJ", "Jr." → "Jr")
        import re as _re
        stripped = _re.sub(r"[.\-']", "", key).strip()
        stripped_map = {_re.sub(r"[.\-']", "", k).strip(): v
                        for k, v in self._player_map.items()}
        if stripped in stripped_map:
            return stripped_map[stripped]

        # 3. Last-name token match scoped to same last name
        parts = key.split()
        if parts:
            last = parts[-1]
            candidates = {k: v for k, v in self._player_map.items()
                          if k.split()[-1] == last}
            if len(candidates) == 1:
                return next(iter(candidates.values()))

        # 4. Fuzzy
        from difflib import get_close_matches as _gcm
        close = _gcm(key, self._player_map.keys(), n=1, cutoff=0.82)
        if close:
            return self._player_map[close[0]]

        return None

    # ── Today's games ──────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_todays_scoreboard(self, game_date: Optional[str] = None) -> pd.DataFrame:
        if scoreboard is None:
            logger.warning("scoreboard endpoint not available in this nba_api version")
            return pd.DataFrame()
        _day = game_date or date.today().strftime("%m/%d/%Y")
        _delay()
        sb = scoreboard.Scoreboard(game_date=_day, headers=NBA_API_HEADERS)
        return sb.game_header.get_data_frame()

    def get_todays_game_ids(self, game_date: Optional[str] = None) -> list[str]:
        df = self.get_todays_scoreboard(game_date)
        return df["GAME_ID"].tolist() if not df.empty else []

    # ── Player game logs ───────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_player_game_log(
        self, player_id: int, season: Optional[str] = None, n_games: int = 50
    ) -> pd.DataFrame:
        _delay()
        gl = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season or self.season,
            season_type_all_star="Regular Season",
            headers=NBA_API_HEADERS,
            timeout=60,
        )
        df = gl.get_data_frames()[0]
        return df.head(n_games) if not df.empty else df

    # ── Season-level player stats ──────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_league_player_stats(
        self,
        measure_type: str = "Base",
        per_mode: str = "PerGame",
        last_n_games: int = 0,
    ) -> pd.DataFrame:
        _delay()
        ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=self.season,
            measure_type_detailed_defense=measure_type,
            per_mode_detailed=per_mode,
            last_n_games=last_n_games,
            headers=NBA_API_HEADERS,
            timeout=60,
        )
        return ep.get_data_frames()[0]

    # ── Advanced player stats ──────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_player_advanced_stats(
        self, player_id: int, season: Optional[str] = None
    ) -> dict[str, pd.DataFrame]:
        """Returns dict of {measure_type: DataFrame}."""
        out = {}
        for mtype in ("Base", "Advanced", "Misc", "Usage", "Scoring"):
            _delay()
            ep = playerdashboardbyclutch.PlayerDashboardByClutch(
                player_id=player_id,
                measure_type_detailed=mtype,
                per_mode_detailed="PerGame",
                season=season or self.season,
                headers=NBA_API_HEADERS,
                timeout=80,
            )
            df = ep.get_data_frames()[0]
            out[mtype] = df
        return out

    # ── Clutch stats ───────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_player_clutch_stats(self, player_id: int) -> pd.DataFrame:
        _delay()
        ep = playerdashboardbyclutch.PlayerDashboardByClutch(
            player_id=player_id,
            measure_type_detailed="Base",
            per_mode_detailed="PerGame",
            season=self.season,
            clutch_time="Last 5 Minutes",
            point_diff=5,
            headers=NBA_API_HEADERS,
            timeout=80,
        )
        return ep.get_data_frames()[1]  # clutch table is index 1

    # ── Shot locations ─────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_player_shot_locations(self, per_mode: str = "PerGame") -> pd.DataFrame:
        _delay()
        ep = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
            distance_range="By Zone",
            per_mode_detailed=per_mode,
            season=self.season,
            headers=NBA_API_HEADERS,
            timeout=80,
        )
        df = ep.get_data_frames()[0]
        # Flatten multi-index columns
        df.columns = [
            f"{a}_{b}" if a else b for a, b in df.columns
        ]
        return df

    # ── Opponent shot locations (defensive matchup) ───────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_team_shot_defense(
        self, last_n: int = 0, per_mode: str = "PerGame"
    ) -> pd.DataFrame:
        _delay()
        ep = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
            distance_range="By Zone",
            measure_type_simple="Opponent",
            per_mode_detailed=per_mode,
            season=self.season,
            last_n_games=last_n,
            headers=NBA_API_HEADERS,
            timeout=80,
        )
        df = ep.get_data_frames()[0]
        df.columns = [f"{a}_{b}" if a else b for a, b in df.columns]
        return df

    # ── Team paint / perimeter defense ────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_team_pt_defense(self, category: str = "Overall") -> pd.DataFrame:
        """category: Overall | 2 Pointers | 3 Pointers | Less Than 6Ft | ..."""
        _delay()
        ep = leaguedashptteamdefend.LeagueDashPtTeamDefend(
            defense_category=category,
            season=self.season,
            headers=NBA_API_HEADERS,
            timeout=80,
        )
        return ep.get_data_frames()[0]

    # ── Team aggregate stats ───────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_team_stats(
        self,
        measure_type: str = "Base",
        per_mode: str = "PerGame",
        last_n: int = 0,
    ) -> pd.DataFrame:
        _delay()
        ep = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            measure_type_detailed_defense=measure_type,
            per_mode_detailed=per_mode,
            last_n_games=last_n,
            headers=NBA_API_HEADERS,
            timeout=80,
        )
        return ep.get_data_frames()[0]

    # ── Pace & possessions ─────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_pace_stats(self) -> pd.DataFrame:
        _delay()
        ep = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            headers=NBA_API_HEADERS,
            timeout=80,
        )
        df = ep.get_data_frames()[0]
        return df[["TEAM_ID", "TEAM_NAME", "PACE", "POSS", "E_PACE"]]

    # ── Tracking: touches, speed, distance ────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_player_tracking_stats(self, pt_measure_type: str = "Possessions") -> pd.DataFrame:
        """pt_measure_type: Possessions | SpeedDistance | Rebounding | Passing | Defense | ..."""
        _delay()
        ep = leaguedashptstats.LeagueDashPtStats(
            player_or_team="Player",
            pt_measure_type=pt_measure_type,
            season=self.season,
            per_mode_detailed="PerGame",
            headers=NBA_API_HEADERS,
            timeout=80,
        )
        return ep.get_data_frames()[0]

    # ── Box score (live / post-game) ───────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_box_score_advanced(self, game_id: str) -> dict[str, pd.DataFrame]:
        _delay()
        ep = boxscoreadvancedv2.BoxScoreAdvancedV2(
            game_id=game_id, headers=NBA_API_HEADERS, timeout=60
        )
        return {
            "player": ep.player_stats.get_data_frame(),
            "team":   ep.team_stats.get_data_frame(),
        }

    # ── Game summary (starters/DNPs) ───────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_game_summary(self, game_id: str) -> dict[str, pd.DataFrame]:
        _delay()
        ep = boxscoresummaryv2.BoxScoreSummaryV2(
            game_id=game_id, headers=NBA_API_HEADERS, timeout=60
        )
        return {
            "line_score":    ep.line_score.get_data_frame(),
            "game_summary":  ep.game_summary.get_data_frame(),
            "inactive":      ep.inactive_players.get_data_frame(),
        }

    # ── Play-by-play ───────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_play_by_play(self, game_id: str) -> pd.DataFrame:
        _delay()
        ep = playbyplayv2.PlayByPlayV2(
            game_id=game_id, headers=NBA_API_HEADERS, timeout=60
        )
        return ep.get_data_frames()[0]

    # ── Team rosters ───────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_team_roster(self, team_id: int) -> pd.DataFrame:
        _delay()
        ep = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=self.season,
            headers=NBA_API_HEADERS,
            timeout=60,
        )
        return ep.get_data_frames()[0]

    # ── Hustle stats ───────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_hustle_stats(self, game_id: str) -> pd.DataFrame:
        _delay()
        ep = hustlestatsboxscore.HustleStatsBoxScore(
            game_id=game_id, headers=NBA_API_HEADERS, timeout=60
        )
        return ep.get_data_frames()[0]

    # ── Player lineup dashboard (per-minute on/off context) ───────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_player_lineup_dashboard(
        self,
        player_id: int,
        season: Optional[str] = None,
        group_quantity: int = 5,
    ) -> pd.DataFrame:
        """
        Returns the player's Per-36 stats broken down by 5-man lineup context.

        Each row is a unique lineup combination.
        GROUP_VALUE contains the 4 partner names separated by " - ".
        Counting stats (PTS, REB, AST, …) are Per-36 minutes.
        MIN is total minutes in that lineup context — use for weighting.

        Parameters
        ----------
        group_quantity : number of players in the lineup group (5 = full 5-man lineup).
                         Use 5 for the most granular on/off context.
        """
        _delay()
        try:
            from nba_api.stats.endpoints import playerdashboardbylineups
            ep = playerdashboardbylineups.PlayerDashboardByLineups(
                player_id=player_id,
                season=season or self.season,
                per_mode_simple="Per36",
                measure_type_detailed="Base",
                group_quantity=str(group_quantity),
                headers=NBA_API_HEADERS,
                timeout=80,
            )
            frames = ep.get_data_frames()
            # Frame 0 = overall season stats, Frame 1 = lineup breakdown
            if len(frames) > 1 and not frames[1].empty:
                return frames[1]
            # Fallback: scan all frames for one with GROUP_VALUE column
            for f in frames:
                if "GROUP_VALUE" in f.columns and not f.empty:
                    return f
        except (ImportError, AttributeError) as e:
            logger.warning(f"playerdashboardbylineups unavailable in nba_api: {e}")
        return pd.DataFrame()

    # ── All season game logs (bulk) ────────────────────────────────────────────
    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2))
    def get_league_game_log(self, season_type: str = "Regular Season") -> pd.DataFrame:
        _delay()
        ep = leaguegamelog.LeagueGameLog(
            season=self.season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="P",
            headers=NBA_API_HEADERS,
            timeout=120,
        )
        return ep.get_data_frames()[0]

    # ── Back-to-back detection ─────────────────────────────────────────────────
    def detect_back_to_back(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """
        schedule_df must have columns: TEAM_ABBREVIATION, GAME_DATE
        Returns schedule_df with 'B2B' bool column added.
        """
        df = schedule_df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
        df["PREV_GAME_DATE"] = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].shift(1)
        df["B2B"] = (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days == 1
        return df

    # ── Helper: compute DK fantasy points from box score ──────────────────────
    @staticmethod
    def compute_dk_fantasy_pts(df: pd.DataFrame) -> pd.Series:
        from core.config import DK_SCORING
        pts  = df.get("PTS", 0)  * DK_SCORING["PTS"]
        fg3m = df.get("FG3M", 0) * DK_SCORING["FG3M"]
        reb  = df.get("REB", 0)  * DK_SCORING["REB"]
        ast  = df.get("AST", 0)  * DK_SCORING["AST"]
        stl  = df.get("STL", 0)  * DK_SCORING["STL"]
        blk  = df.get("BLK", 0)  * DK_SCORING["BLK"]
        tov  = df.get("TOV", 0)  * DK_SCORING["TOV"]
        base = pts + fg3m + reb + ast + stl + blk + tov

        cats = {
            "pts":  df.get("PTS",  0) >= 10,
            "reb":  df.get("REB",  0) >= 10,
            "ast":  df.get("AST",  0) >= 10,
            "stl":  df.get("STL",  0) >= 10,
            "blk":  df.get("BLK",  0) >= 10,
        }
        dd_count = sum(cats.values())
        dd_bonus = DK_SCORING["DOUBLE_DOUBLE"] if dd_count >= 2 else 0
        td_bonus = DK_SCORING["TRIPLE_DOUBLE"] if dd_count >= 3 else 0
        return base + dd_bonus + td_bonus

    @staticmethod
    def compute_fd_fantasy_pts(df: pd.DataFrame) -> pd.Series:
        from core.config import FD_SCORING
        return (
            df.get("PTS",  0) * FD_SCORING["PTS"]  +
            df.get("REB",  0) * FD_SCORING["REB"]  +
            df.get("AST",  0) * FD_SCORING["AST"]  +
            df.get("STL",  0) * FD_SCORING["STL"]  +
            df.get("BLK",  0) * FD_SCORING["BLK"]  +
            df.get("TOV",  0) * FD_SCORING["TOV"]
        )
