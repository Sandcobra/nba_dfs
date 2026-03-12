"""
Ownership Calibrator — matches projected vs actual ownership from contest results
to find systematic biases and produce correction factors.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import process, fuzz
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONTEST_DIR = PROJECT_ROOT / "contest"
CACHE_DIR = PROJECT_ROOT / "cache"


def _salary_tier(salary):
    if salary < 5000:
        return "3k-5k"
    elif salary < 7000:
        return "5k-7k"
    elif salary < 9000:
        return "7k-9k"
    else:
        return "9k+"


def _perf_bucket(fpts):
    if fpts < 20:
        return "bust"
    elif fpts < 35:
        return "average"
    elif fpts < 50:
        return "value"
    else:
        return "ceiling"


def load_actual_ownership():
    """Load actual ownership and FPTS from all contest results files."""
    rows = []
    for f in sorted(CONTEST_DIR.glob("contest-results_3_*.csv")):
        date_str = f.stem.replace("contest-results_", "")
        df = pd.read_csv(f)
        sub = df[["Player", "%Drafted", "FPTS"]].dropna(subset=["Player"])
        sub = sub[sub["Player"].str.strip() != ""]
        sub = sub.drop_duplicates(subset=["Player"])
        sub["slate_date"] = date_str
        sub["%Drafted"] = sub["%Drafted"].astype(str).str.replace("%", "").astype(float)
        rows.append(sub)
    combined = pd.concat(rows, ignore_index=True)
    combined.columns = ["player", "actual_ownership", "actual_fpts", "slate_date"]
    return combined


def load_salary_data():
    """Load salary and position data from DK slate files."""
    rows = []
    for f in sorted(CONTEST_DIR.glob("dk_slate_3_*.csv")):
        date_str = f.stem.replace("dk_slate_", "")
        df = pd.read_csv(f)
        # Find name/salary/position columns
        name_col = next((c for c in df.columns if c in ["Name", "name"]), None)
        sal_col = next((c for c in df.columns if c in ["Salary", "salary"]), None)
        pos_col = next((c for c in df.columns if c in ["Position", "Roster Position", "position"]), None)
        avg_col = next((c for c in df.columns if "AvgPointsPerGame" in c or c == "avg_pts"), None)
        if not all([name_col, sal_col, pos_col]):
            continue
        sub = df[[name_col, sal_col, pos_col]].copy()
        sub.columns = ["player", "salary", "position"]
        if avg_col:
            sub["avg_pts"] = df[avg_col]
        sub["slate_date"] = date_str
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def match_players(ownership_df, salary_df):
    """Fuzzy match player names between ownership and salary data."""
    results = []
    salary_names = salary_df["player"].tolist()

    for _, row in ownership_df.iterrows():
        match = process.extractOne(row["player"], salary_names, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 80:
            sal_row = salary_df[salary_df["player"] == match[0]].iloc[0]
            results.append({
                "player": row["player"],
                "actual_ownership": row["actual_ownership"],
                "actual_fpts": row["actual_fpts"],
                "slate_date": row["slate_date"],
                "salary": sal_row["salary"],
                "position": sal_row["position"],
                "avg_pts": sal_row.get("avg_pts", np.nan),
            })
    return pd.DataFrame(results)


def compute_naive_projected_ownership(salary):
    """Naive salary-based ownership projection."""
    return max(1.0, min(50.0, (50000 - salary) / 50000 * 30))


def build_calibration():
    """Build ownership calibration from contest results."""
    logger.info("Loading contest results...")
    own_df = load_actual_ownership()
    sal_df = load_salary_data()

    logger.info(f"Loaded {len(own_df)} ownership records, {len(sal_df)} salary records")

    matched = match_players(own_df, sal_df)
    logger.info(f"Matched {len(matched)} players")

    matched["projected_ownership"] = matched["salary"].apply(compute_naive_projected_ownership)
    matched["bias"] = matched["projected_ownership"] - matched["actual_ownership"]
    matched["salary_tier"] = matched["salary"].apply(_salary_tier)
    matched["perf_bucket"] = matched["actual_fpts"].apply(_perf_bucket)

    # Per-position multipliers
    pos_multipliers = {}
    for pos in matched["position"].unique():
        sub = matched[matched["position"] == pos]
        if len(sub) < 5:
            continue
        ratio = (sub["actual_ownership"] / sub["projected_ownership"].replace(0, np.nan)).median()
        pos_multipliers[pos] = round(float(ratio), 3) if not np.isnan(ratio) else 1.0

    # Per-salary-tier bias
    tier_bias = {}
    for tier in ["3k-5k", "5k-7k", "7k-9k", "9k+"]:
        sub = matched[matched["salary_tier"] == tier]
        if len(sub) < 3:
            tier_bias[tier] = 0.0
            continue
        tier_bias[tier] = round(float(sub["bias"].median()), 2)

    # Top underowned (actual < 10% but FPTS > 35)
    underowned = matched[(matched["actual_ownership"] < 10) & (matched["actual_fpts"] > 35)]
    underowned = underowned.sort_values("actual_fpts", ascending=False).head(20)

    # Top chalk traps (actual > 40% but FPTS < 25)
    chalk_traps = matched[(matched["actual_ownership"] > 40) & (matched["actual_fpts"] < 25)]
    chalk_traps = chalk_traps.sort_values("actual_ownership", ascending=False).head(10)

    calibration = {
        "position_multipliers": pos_multipliers,
        "tier_bias": tier_bias,
        "underowned_gems": underowned[["player", "salary", "actual_ownership", "actual_fpts", "slate_date"]].to_dict("records"),
        "chalk_traps": chalk_traps[["player", "salary", "actual_ownership", "actual_fpts", "slate_date"]].to_dict("records"),
        "n_samples": len(matched),
        "mean_bias": round(float(matched["bias"].mean()), 2),
    }

    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / "ownership_calibration.json"
    with open(out, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.success(f"Calibration saved to {out} ({len(matched)} samples)")
    return calibration, matched


def apply_ownership_calibration(df: pd.DataFrame, calibration: dict) -> pd.DataFrame:
    """Apply ownership calibration corrections to a player pool DataFrame."""
    df = df.copy()
    if "proj_ownership" not in df.columns:
        df["proj_ownership"] = df["salary"].apply(compute_naive_projected_ownership)

    pos_mult = calibration.get("position_multipliers", {})
    tier_bias = calibration.get("tier_bias", {})

    for idx, row in df.iterrows():
        pos = str(row.get("position", row.get("primary_position", ""))).split("/")[0].strip()
        sal = row.get("salary", 7000)
        tier = _salary_tier(sal)

        proj = df.at[idx, "proj_ownership"]
        # Apply position multiplier
        mult = pos_mult.get(pos, 1.0)
        proj = proj * mult
        # Apply tier bias correction
        proj = proj - tier_bias.get(tier, 0.0)
        df.at[idx, "proj_ownership"] = max(0.5, min(70.0, proj))

    return df


if __name__ == "__main__":
    calibration, matched = build_calibration()
    print(f"\n=== Ownership Calibration Results ===")
    print(f"Samples: {calibration['n_samples']} | Mean bias: {calibration['mean_bias']:.1f}%")
    print(f"\nPosition multipliers: {calibration['position_multipliers']}")
    print(f"Tier bias: {calibration['tier_bias']}")
    print(f"\nTop underowned gems:")
    for g in calibration['underowned_gems'][:5]:
        print(f"  {g['player']}: {g['actual_ownership']:.1f}% owned, {g['actual_fpts']:.1f} FPTS (${g['salary']:,})")
    print(f"\nTop chalk traps:")
    for c in calibration['chalk_traps'][:5]:
        print(f"  {c['player']}: {c['actual_ownership']:.1f}% owned, {c['actual_fpts']:.1f} FPTS (${c['salary']:,})")
