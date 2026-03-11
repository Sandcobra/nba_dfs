"""
SQLite database layer for persistent storage of projections, lineups,
ownership history, and player/game metadata.
"""

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from core.config import DB_PATH


# ── Schema DDL ─────────────────────────────────────────────────────────────────
SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS players (
    player_id      INTEGER PRIMARY KEY,
    name           TEXT    NOT NULL,
    position       TEXT,
    team           TEXT,
    updated_at     TEXT
);

CREATE TABLE IF NOT EXISTS games (
    game_id        TEXT    PRIMARY KEY,
    game_date      TEXT,
    home_team      TEXT,
    away_team      TEXT,
    game_total     REAL,
    home_spread    REAL,
    tip_time       TEXT
);

CREATE TABLE IF NOT EXISTS projections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date       TEXT,
    slate_date     TEXT,
    player_id      INTEGER,
    name           TEXT,
    team           TEXT,
    opp            TEXT,
    salary_dk      INTEGER,
    salary_fd      INTEGER,
    position       TEXT,
    projected_pts_dk   REAL,
    projected_pts_fd   REAL,
    projection_sd  REAL,
    minutes_proj   REAL,
    pts_proj       REAL,
    reb_proj       REAL,
    ast_proj       REAL,
    stl_proj       REAL,
    blk_proj       REAL,
    tov_proj       REAL,
    fg3m_proj      REAL,
    ownership_proj REAL,
    value_dk       REAL,
    ceiling        REAL,
    floor          REAL,
    model_confidence REAL,
    injury_status  TEXT,
    is_confirmed   INTEGER DEFAULT 0,
    FOREIGN KEY(player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS lineups (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT,
    slate_date     TEXT,
    site           TEXT,
    contest_type   TEXT,
    lineup_num     INTEGER,
    player_ids     TEXT,   -- JSON array of player_ids
    player_names   TEXT,   -- JSON array of names (human readable)
    total_salary   INTEGER,
    proj_pts       REAL,
    proj_ownership REAL,
    leverage       REAL
);

CREATE TABLE IF NOT EXISTS ownership_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    slate_date     TEXT,
    player_id      INTEGER,
    name           TEXT,
    ownership_pct  REAL,
    site           TEXT,
    contest_type   TEXT,
    source         TEXT
);

CREATE TABLE IF NOT EXISTS player_game_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id        TEXT,
    player_id      INTEGER,
    game_date      TEXT,
    team           TEXT,
    opp            TEXT,
    min            REAL,
    pts            REAL,
    reb            REAL,
    ast            REAL,
    stl            REAL,
    blk            REAL,
    tov            REAL,
    fg3m           REAL,
    fg_pct         REAL,
    ts_pct         REAL,
    usg_pct        REAL,
    fantasy_pts_dk REAL,
    fantasy_pts_fd REAL,
    home_away      TEXT,
    b2b            INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS injury_reports (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id      INTEGER,
    name           TEXT,
    team           TEXT,
    status         TEXT,   -- OUT, QUESTIONABLE, DOUBTFUL, PROBABLE, GTD
    reason         TEXT,
    reported_at    TEXT,
    source         TEXT
);

CREATE TABLE IF NOT EXISTS vegas_lines (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id        TEXT,
    slate_date     TEXT,
    home_team      TEXT,
    away_team      TEXT,
    total          REAL,
    home_spread    REAL,
    home_ml        INTEGER,
    away_ml        INTEGER,
    bookmaker      TEXT,
    updated_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_proj_date      ON projections(slate_date);
CREATE INDEX IF NOT EXISTS idx_lineup_date    ON lineups(slate_date);
CREATE INDEX IF NOT EXISTS idx_gamelog_player ON player_game_log(player_id);
CREATE INDEX IF NOT EXISTS idx_gamelog_date   ON player_game_log(game_date);
CREATE INDEX IF NOT EXISTS idx_injury_player  ON injury_reports(player_id);
"""


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        logger.info(f"Database initialised at {self.db_path}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Upsert helpers ─────────────────────────────────────────────────────────
    def upsert_players(self, df: pd.DataFrame):
        rows = df[["player_id", "name", "position", "team"]].copy()
        rows["updated_at"] = datetime.utcnow().isoformat()
        with self._conn() as conn:
            rows.to_sql("players", conn, if_exists="replace", index=False)

    def save_projections(self, df: pd.DataFrame, slate_date: str):
        df = df.copy()
        df["run_date"]   = datetime.utcnow().isoformat()
        df["slate_date"] = slate_date
        with self._conn() as conn:
            df.to_sql("projections", conn, if_exists="append", index=False)
        logger.success(f"Saved {len(df)} projections for {slate_date}")

    def get_projections(self, slate_date: str) -> pd.DataFrame:
        sql = "SELECT * FROM projections WHERE slate_date = ? ORDER BY projected_pts_dk DESC"
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(slate_date,))

    def save_lineups(self, lineups: list[dict]):
        with self._conn() as conn:
            for lu in lineups:
                conn.execute(
                    """INSERT INTO lineups
                       (created_at, slate_date, site, contest_type, lineup_num,
                        player_ids, player_names, total_salary, proj_pts,
                        proj_ownership, leverage)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        datetime.utcnow().isoformat(),
                        lu["slate_date"], lu["site"], lu["contest_type"],
                        lu["lineup_num"],
                        str(lu["player_ids"]), str(lu["player_names"]),
                        lu["total_salary"], lu["proj_pts"],
                        lu.get("proj_ownership", 0), lu.get("leverage", 0),
                    ),
                )
        logger.success(f"Saved {len(lineups)} lineups to database")

    def save_game_logs(self, df: pd.DataFrame):
        with self._conn() as conn:
            df.to_sql("player_game_log", conn, if_exists="append", index=False)

    def get_game_logs(self, player_id: int, n_games: int = 20) -> pd.DataFrame:
        sql = """
            SELECT * FROM player_game_log
            WHERE player_id = ?
            ORDER BY game_date DESC
            LIMIT ?
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(player_id, n_games))

    def save_injury_report(self, records: list[dict]):
        with self._conn() as conn:
            for r in records:
                conn.execute(
                    """INSERT INTO injury_reports
                       (player_id, name, team, status, reason, reported_at, source)
                       VALUES (?,?,?,?,?,?,?)""",
                    (r["player_id"], r["name"], r["team"], r["status"],
                     r.get("reason", ""), datetime.utcnow().isoformat(),
                     r.get("source", "")),
                )

    def get_current_injuries(self) -> pd.DataFrame:
        today = date.today().isoformat()
        sql = """
            SELECT DISTINCT player_id, name, team, status, reason, source
            FROM injury_reports
            WHERE date(reported_at) = ?
            ORDER BY reported_at DESC
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(today,))

    def save_vegas_lines(self, df: pd.DataFrame):
        with self._conn() as conn:
            df.to_sql("vegas_lines", conn, if_exists="append", index=False)

    def get_vegas_lines(self, slate_date: str) -> pd.DataFrame:
        sql = "SELECT * FROM vegas_lines WHERE slate_date = ?"
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(slate_date,))

    def save_ownership_history(self, df: pd.DataFrame):
        with self._conn() as conn:
            df.to_sql("ownership_history", conn, if_exists="append", index=False)

    def get_ownership_history(self, player_id: int, days: int = 30) -> pd.DataFrame:
        sql = """
            SELECT * FROM ownership_history
            WHERE player_id = ?
            AND date(slate_date) >= date('now', ?)
            ORDER BY slate_date DESC
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(player_id, f"-{days} days"))

    def execute(self, sql: str, params: tuple = ()) -> list[Any]:
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall()


# Singleton
_db: Database | None = None

def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
