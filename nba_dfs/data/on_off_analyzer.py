"""Compute ON/OFF usage splits from ESPN cached game logs."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"

# Key star/backup pairs to analyze
STAR_BACKUP_PAIRS = [
    ("de'aaron fox", "malik monk"),
    ("nikola jokic", "michael porter jr."),
    ("shai gilgeous-alexander", "jalen williams"),
    ("luka doncic", "kyrie irving"),
    ("giannis antetokounmpo", "damian lillard"),
    ("jayson tatum", "jaylen brown"),
    ("stephen curry", "klay thompson"),
    ("lebron james", "anthony davis"),
    ("kevin durant", "devin booker"),
    ("joel embiid", "tyrese maxey"),
]


def compute_on_off_splits(espn_client) -> dict:
    """Compute on/off splits for all star/backup pairs."""
    all_logs = espn_client.get_all_game_logs()
    splits = {}

    for star_name, backup_name in STAR_BACKUP_PAIRS:
        star_logs = all_logs.get(star_name)
        backup_logs = all_logs.get(backup_name)

        if star_logs is None or backup_logs is None:
            continue

        # Align by date
        star_logs = star_logs.copy()
        backup_logs = backup_logs.copy()

        if "game_date" not in star_logs.columns or "game_date" not in backup_logs.columns:
            continue

        star_dates = set(pd.to_datetime(star_logs["game_date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna())
        backup_dates = set(pd.to_datetime(backup_logs["game_date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna())

        shared_dates = star_dates & backup_dates
        star_played_dates = star_dates  # dates star appeared in log = played

        # Games where backup played WITH star
        with_star = backup_logs[
            pd.to_datetime(backup_logs["game_date"], errors="coerce").dt.strftime("%Y-%m-%d").isin(star_played_dates & backup_dates)
        ]["fantasy_pts_dk"].dropna()

        # Games where backup played WITHOUT star
        without_star = backup_logs[
            ~pd.to_datetime(backup_logs["game_date"], errors="coerce").dt.strftime("%Y-%m-%d").isin(star_played_dates)
        ]["fantasy_pts_dk"].dropna()

        if len(with_star) < 3 or len(without_star) < 1:
            continue

        with_avg = float(with_star.mean())
        without_avg = float(without_star.mean())
        uplift = without_avg / with_avg if with_avg > 0 else 1.0

        splits[backup_name] = {
            "primary_star": star_name,
            "with_star_avg_dk": round(with_avg, 2),
            "without_star_avg_dk": round(without_avg, 2),
            "uplift_factor": round(uplift, 3),
            "with_star_games": len(with_star),
            "without_star_games": len(without_star),
        }
        logger.info(f"{backup_name}: with {star_name}={with_avg:.1f}, without={without_avg:.1f} (uplift {uplift:.2f}x)")

    out = CACHE_DIR / "on_off_splits.json"
    with open(out, "w") as f:
        json.dump(splits, f, indent=2)
    logger.success(f"ON/OFF splits saved: {len(splits)} pairs")
    return splits
