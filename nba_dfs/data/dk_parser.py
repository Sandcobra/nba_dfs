"""
DraftKings / FanDuel salary file parser.
Reads exported CSV from DK/FD contest lobby and prepares a clean player pool.
"""

import re
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger


SITE = Literal["dk", "fd"]

# DraftKings CSV column mapping
DK_COL_MAP = {
    "Name":               "name",
    "Name + ID":          "name_id",
    "ID":                 "player_id",
    "Position":           "position",
    "Roster Position":    "roster_position",
    "Salary":             "salary",
    "Game Info":          "game_info",
    "TeamAbbrev":         "team",
    "AvgPointsPerGame":   "avg_pts",
}

FD_COL_MAP = {
    "Nickname":           "name",
    "Id":                 "player_id",
    "Position":           "position",
    "Salary":             "salary",
    "Game":               "game_info",
    "Team":               "team",
    "Opponent":           "opp",
    "FPPG":               "avg_pts",
}

# Eligible position slots per site
DK_ELIGIBLE = {
    "PG":   ["PG", "G", "UTIL"],
    "SG":   ["SG", "G", "UTIL"],
    "SF":   ["SF", "F", "UTIL"],
    "PF":   ["PF", "F", "UTIL"],
    "C":    ["C", "UTIL"],
}

FD_ELIGIBLE = {
    "PG":   ["PG"],
    "SG":   ["SG"],
    "SF":   ["SF"],
    "PF":   ["PF"],
    "C":    ["C"],
}


def parse_dk_salary_csv(filepath: str | Path) -> pd.DataFrame:
    """Parse a DraftKings salary export CSV into a clean player pool DataFrame."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={k: v for k, v in DK_COL_MAP.items() if k in df.columns})

    required = ["name", "salary", "position", "team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DK CSV missing columns: {missing}")

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    df["avg_pts"] = pd.to_numeric(df.get("avg_pts", 0), errors="coerce").fillna(0)

    # Parse game info: e.g. "LAL@GSW 07:30PM ET"
    if "game_info" in df.columns:
        df[["away_team_raw", "home_team_raw"]] = (
            df["game_info"].str.extract(r"([A-Z]+)@([A-Z]+)")
        )
        df["tip_time"] = df["game_info"].str.extract(r"(\d{2}:\d{2}[AP]M)")
        df["opp"] = df.apply(
            lambda r: r["away_team_raw"]
            if r["team"] == r.get("home_team_raw", "")
            else r.get("home_team_raw", ""),
            axis=1,
        )

    # Build eligible slots list
    def eligible_slots(pos: str) -> list[str]:
        primary = [p.strip() for p in str(pos).split("/")]
        slots = set()
        for p in primary:
            slots.update(DK_ELIGIBLE.get(p, []))
        return sorted(slots)

    df["eligible_slots"] = df["position"].apply(eligible_slots)
    df["primary_position"] = df["position"].apply(
        lambda p: p.split("/")[0].strip() if "/" in str(p) else str(p).strip()
    )

    # Value metric
    df["value_dk"] = (df["avg_pts"] / (df["salary"] / 1000)).round(2)

    df = df.dropna(subset=["name"]).reset_index(drop=True)
    logger.info(f"Parsed {len(df)} players from DK salary file")
    return df


def parse_fd_salary_csv(filepath: str | Path) -> pd.DataFrame:
    """Parse a FanDuel salary export CSV into a clean player pool DataFrame."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={k: v for k, v in FD_COL_MAP.items() if k in df.columns})

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    df["avg_pts"] = pd.to_numeric(df.get("avg_pts", 0), errors="coerce").fillna(0)
    df["value_fd"] = (df["avg_pts"] / (df["salary"] / 1000)).round(2)

    def fd_eligible_slots(pos: str) -> list[str]:
        return [p.strip() for p in str(pos).split("/")]

    df["eligible_slots"] = df["position"].apply(fd_eligible_slots)
    df["primary_position"] = df["position"].apply(
        lambda p: p.split("/")[0].strip() if "/" in str(p) else str(p).strip()
    )

    df = df.dropna(subset=["name"]).reset_index(drop=True)
    logger.info(f"Parsed {len(df)} players from FD salary file")
    return df


def merge_salary_with_projections(
    salary_df: pd.DataFrame,
    projections_df: pd.DataFrame,
    name_col_proj: str = "name",
) -> pd.DataFrame:
    """
    Left-join projections onto salary data by player name (fuzzy).
    Returns merged DataFrame with projection columns appended.
    """
    from rapidfuzz import process, fuzz

    proj_names = projections_df[name_col_proj].tolist()

    def best_match(name: str) -> str | None:
        result = process.extractOne(name, proj_names, scorer=fuzz.token_sort_ratio, score_cutoff=82)
        return result[0] if result else None

    salary_df = salary_df.copy()
    salary_df["_matched_proj_name"] = salary_df["name"].apply(best_match)

    merged = salary_df.merge(
        projections_df.rename(columns={name_col_proj: "_matched_proj_name"}),
        on="_matched_proj_name",
        how="left",
        suffixes=("", "_proj"),
    )
    merged = merged.drop(columns=["_matched_proj_name"])

    unmatched = merged[merged["projected_pts_dk"].isna()]["name"].tolist()
    if unmatched:
        logger.warning(f"No projection found for {len(unmatched)} players: {unmatched[:10]}...")

    return merged
