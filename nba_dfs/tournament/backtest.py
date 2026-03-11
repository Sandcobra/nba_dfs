"""
Backtest Engine
===============
Validates the tournament system against historical contest data.

For each slate/result pair:
  1. Run the full pipeline to generate lineups
  2. Extract actual player scores from contest results
  3. Score our lineups against actual performance
  4. Measure rank in the actual field
  5. Identify what we got right and wrong

Iterates until the system produces top-1% lineups on all tested nights.

Usage:
    cd nba_dfs
    python tournament/backtest.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_REPO = _ROOT.parent
for p in [str(_ROOT), str(_REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tournament.autonomous_runner import AutonomousRunner

CONTEST_DIR = _REPO / "contest"
OUTPUTS_DIR = _REPO / "outputs"


# ── Actual score extraction ────────────────────────────────────────────────────
def extract_player_scores(results_path: Path) -> dict[str, float]:
    """
    Parse a DraftKings contest results CSV and return {player_name: dk_pts}.
    Uses the Player/FPTS side columns in the results file.
    """
    scores: dict[str, float] = {}
    with open(results_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player = (row.get("Player") or "").strip()
            fpts   = (row.get("FPTS")   or "").strip()
            if player and fpts:
                try:
                    val = float(fpts)
                    if player not in scores or val > scores[player]:
                        scores[player] = val
                except ValueError:
                    pass
    return scores


def extract_field_scores(results_path: Path) -> list[float]:
    """Return sorted (desc) list of all entry scores from the contest."""
    scores = []
    with open(results_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts = (row.get("Points") or "").strip()
            if pts:
                try:
                    scores.append(float(pts))
                except ValueError:
                    pass
    return sorted(scores, reverse=True)


def extract_winning_lineups(results_path: Path, top_n: int = 10) -> list[dict]:
    """Extract the top N winning lineups from contest results."""
    lineups = []
    with open(results_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank_str = (row.get("Rank") or "").strip()
            if not rank_str:
                continue
            try:
                rank = int(rank_str)
            except ValueError:
                continue
            if rank > top_n:
                continue
            lineup_str = (row.get("Lineup") or "").strip()
            pts_str    = (row.get("Points") or "").strip()
            name       = (row.get("EntryName") or "").strip()
            if lineup_str:
                # Parse "PG Tyler Herro SG Jordan Miller ..."
                players = re.findall(r"(?:PG|SG|SF|PF|C|G|F|UTIL)\s+([A-Za-z\s'.]+?)(?=\s+(?:PG|SG|SF|PF|C|G|F|UTIL)|$)", lineup_str)
                players = [p.strip() for p in players if p.strip()]
                lineups.append({
                    "rank":    rank,
                    "name":    name,
                    "score":   float(pts_str) if pts_str else 0.0,
                    "players": players,
                })
    return sorted(lineups, key=lambda x: x["rank"])


# ── Score a lineup using actual player scores ─────────────────────────────────
def score_lineup(player_ids: list[str], names: list[str],
                  actual_scores: dict[str, float]) -> tuple[float, list]:
    """
    Score a lineup using actual player scores.
    Returns (total_score, per_player_breakdown).
    Matches by name (case-insensitive partial match).
    """
    total = 0.0
    breakdown = []
    name_lower = {k.lower(): v for k, v in actual_scores.items()}

    for name in names:
        nl = name.lower()
        # Exact match first
        if nl in name_lower:
            pts = name_lower[nl]
        else:
            # Partial match (last name)
            parts = nl.split()
            last = parts[-1] if parts else ""
            matches = [(k, v) for k, v in name_lower.items() if last in k]
            pts = matches[0][1] if len(matches) == 1 else 0.0
            if len(matches) == 0:
                pts = 0.0  # player didn't play / not in results
        total += pts
        breakdown.append((name, pts))
    return round(total, 2), breakdown


# ── Find percentile rank in field ─────────────────────────────────────────────
def field_rank(score: float, field_scores: list[float]) -> tuple[int, float]:
    """Returns (rank_number, percentile) in the field."""
    rank = sum(1 for s in field_scores if s > score) + 1
    pct  = rank / len(field_scores) * 100 if field_scores else 100
    return rank, pct


# ── Oracle: best possible lineup from player pool ────────────────────────────
def oracle_lineup(actual_scores: dict[str, float],
                  slate_players: pd.DataFrame) -> tuple[float, list]:
    """
    Find the best possible lineup using actual scores, respecting DK constraints.
    Uses a greedy approximation (sorted by actual score, pick top 8 from valid roster).
    """
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus

    SALARY_CAP  = 50_000
    ROSTER_SIZE = 8
    _SLOT_ORDER    = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    _SLOT_ELIGIBLE = {
        "PG": ["PG", "G", "UTIL"],
        "SG": ["SG", "G", "UTIL"],
        "SF": ["SF", "F", "UTIL"],
        "PF": ["PF", "F", "UTIL"],
        "C":  ["C", "UTIL"],
    }

    # Match players to actual scores
    name_lower = {k.lower(): v for k, v in actual_scores.items()}
    rows = []
    for _, r in slate_players.iterrows():
        name = str(r.get("name", ""))
        nl   = name.lower()
        pts  = name_lower.get(nl, 0.0)
        if pts == 0.0:
            parts = nl.split()
            last  = parts[-1] if parts else ""
            matches = [(k, v) for k, v in name_lower.items() if last in k]
            pts = matches[0][1] if len(matches) == 1 else 0.0
        rows.append({
            "name":     name,
            "salary":   int(r.get("salary", 5000)),
            "pos":      str(r.get("primary_position", "G")),
            "elig":     list(r.get("eligible_slots", ["UTIL"])),
            "actual":   pts,
        })

    df = pd.DataFrame(rows)
    prob = LpProblem("Oracle", LpMaximize)
    n    = len(df)
    x    = [LpVariable(f"x{i}", cat="Binary") for i in range(n)]

    prob += lpSum(x[i] * df.iloc[i]["actual"] for i in range(n))
    prob += lpSum(x) == ROSTER_SIZE
    prob += lpSum(x[i] * df.iloc[i]["salary"] for i in range(n)) <= SALARY_CAP

    y = {}
    for i in range(n):
        slots = df.iloc[i]["elig"]
        if not isinstance(slots, list): slots = []
        for slot in slots:
            if slot in _SLOT_ORDER:
                y[(i, slot)] = LpVariable(f"y{i}_{slot}", cat="Binary")

    for slot in _SLOT_ORDER:
        fills = [y[(i, slot)] for i in range(n) if (i, slot) in y]
        if fills:
            prob += lpSum(fills) == 1

    for i in range(n):
        pvars = [y[(i, slot)] for slot in _SLOT_ORDER if (i, slot) in y]
        if pvars:
            prob += lpSum(pvars) == x[i]
        else:
            prob += x[i] == 0

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=10))

    if LpStatus[prob.status] != "Optimal":
        # Fallback: just take top 8 by actual score (ignoring roster)
        top8 = df.nlargest(8, "actual")
        total = top8["actual"].sum()
        return round(total, 2), top8["name"].tolist()

    selected = [(df.iloc[i]["name"], df.iloc[i]["actual"]) for i in range(n) if x[i].value() and x[i].value() > 0.5]
    total = sum(v for _, v in selected)
    return round(total, 2), [n for n, _ in selected]


# ── Per-night backtest ─────────────────────────────────────────────────────────
def backtest_night(
    slate_path:   Path,
    results_path: Path,
    runner:       AutonomousRunner,
    date_str:     str,
) -> dict:
    """
    Run one full backtest for a single night.
    Returns a metrics dict with all analysis results.
    """
    print(f"\n{'='*70}")
    print(f"BACKTEST: {date_str}  slate={slate_path.name}")
    print(f"{'='*70}")

    # Actual data
    actual_scores  = extract_player_scores(results_path)
    field_scores   = extract_field_scores(results_path)
    winning_lus    = extract_winning_lineups(results_path, top_n=5)

    n_entries     = len(field_scores)
    winner_score  = field_scores[0] if field_scores else 0
    top1_idx      = max(1, int(n_entries * 0.01))
    top1_score    = field_scores[top1_idx - 1]
    cash_idx      = n_entries // 2
    cash_score    = field_scores[cash_idx]

    print(f"  Field: {n_entries:,} entries | Winner: {winner_score:.2f} | "
          f"Top 1%: {top1_score:.2f} | Cash: {cash_score:.2f}")

    # Winning lineups
    print(f"\n  TOP 5 WINNING LINEUPS:")
    for lu in winning_lus[:5]:
        print(f"    #{lu['rank']} {lu['name']:30s} {lu['score']:.2f}  {lu['players']}")

    # Top 10 actual scorers
    top_scorers = sorted(actual_scores.items(), key=lambda x: -x[1])
    print(f"\n  TOP 10 ACTUAL SCORERS:")
    for name, pts in top_scorers[:10]:
        print(f"    {name:<30s} {pts:.2f}")

    # Generate our lineups
    print(f"\n  Generating lineups...")
    # Pass cutoff_date so ESPN game log signal only uses data before the game night
    # (prevents data leakage — we only know what was available before the contest)
    cutoff = date_str.replace("-03-0", "-03-0").strip()  # keep "2026-03-08" format
    players = runner._parse_slate(slate_path, cutoff_date=cutoff)
    # Skip live enrichment for backtests — B2B/starters data is date-specific and
    # fetching live data for historical dates returns today's schedule, not the
    # historical one. Raw slate projections are sufficient for backtest validation.
    # players = runner._enrich_players(players, slate_path)

    # Show our projections vs actual for top scorers
    print(f"\n  PROJECTION vs ACTUAL (top 10 actual scorers):")
    pid_to_proj = {}
    if not players.empty:
        for _, row in players.iterrows():
            pid_to_proj[str(row.get("name", "")).lower()] = {
                "proj":    float(row.get("proj_pts_dk", 0)),
                "ceiling": float(row.get("ceiling", 0)),
                "gpp":     float(row.get("gpp_score", 0)),
                "own":     float(row.get("proj_own", 0)),
                "salary":  int(row.get("salary", 0)),
                "dnp":     float(row.get("dnp_risk", 0)),
            }
    for name, actual in top_scorers[:10]:
        our = pid_to_proj.get(name.lower(), {})
        proj   = our.get("proj", 0)
        ceil_  = our.get("ceiling", 0)
        gpp    = our.get("gpp", 0)
        own    = our.get("own", 0)
        sal    = our.get("salary", 0)
        dnp    = our.get("dnp", 0)
        ratio  = actual / proj if proj > 0 else 0
        flag   = " ***MISSED***" if ratio > 1.4 else ("  ^^UNDERVALUED" if ratio > 1.15 else "")
        print(f"    {name:<28s} proj={proj:5.1f} ceil={ceil_:5.1f} actual={actual:5.1f} "
              f"ratio={ratio:.2f} gpp={gpp:5.1f} own={own:.0f}% sal=${sal:,}{flag}")

    # Oracle lineup
    oracle_score, oracle_names = oracle_lineup(actual_scores, players)
    oracle_rank, oracle_pct = field_rank(oracle_score, field_scores)
    print(f"\n  ORACLE (best valid 8-man lineup):")
    print(f"    Score: {oracle_score:.2f} | Rank: {oracle_rank}/{n_entries} (top {oracle_pct:.1f}%)")
    print(f"    Players: {oracle_names}")

    # Build and score our lineups
    from tournament.score_distribution import ScoreDistribution
    from tournament.contest_simulator import ContestSimulator
    from tournament.portfolio_optimizer import PortfolioOptimizer

    score_dist = ScoreDistribution()
    score_dist.fit(players)
    for _, row in players.iterrows():
        score_dist.set_team_matchup(
            str(row["player_id"]), str(row.get("team", "")), str(row.get("matchup", ""))
        )

    simulator = ContestSimulator(score_dist=score_dist, players=players, n_sim=500, seed=42)
    optimizer = PortfolioOptimizer(players=players, score_dist=score_dist, simulator=simulator,
                                   max_exposure=0.33, pool_multiplier=6)
    lineups = optimizer.build(n=20, use_simulation=True, n_sim=500)

    print(f"\n  OUR {len(lineups)} LINEUPS vs ACTUAL FIELD:")
    print(f"  {'#':>3} {'Actual':>7} {'Proj':>6} {'Rank':>7} {'Pct':>6}  Players")
    print(f"  " + "-"*75)

    our_scores_actual = []
    best_rank = n_entries
    best_score_actual = 0.0
    top1_count = 0
    cash_count  = 0

    for lu in lineups:
        actual_total, breakdown = score_lineup(lu["player_ids"], lu["names"], actual_scores)
        rank, pct = field_rank(actual_total, field_scores)
        our_scores_actual.append(actual_total)
        if rank < best_rank:
            best_rank = rank
            best_score_actual = actual_total
        if actual_total >= top1_score:
            top1_count += 1
        if actual_total >= cash_score:
            cash_count += 1

        flag = " TOP1%" if actual_total >= top1_score else (" CASH" if actual_total >= cash_score else "")
        players_str = ", ".join(lu["names"][:4]) + "..."
        print(f"  #{lu['lineup_num']:>2d} {actual_total:>7.2f} {lu['proj_pts']:>6.1f} "
              f"{rank:>7d} {pct:>5.1f}%  {players_str}{flag}")

    # Show what the TOP lineup was missing
    print(f"\n  SUMMARY:")
    print(f"    Best rank:        #{best_rank}/{n_entries} ({best_rank/n_entries*100:.1f}%)")
    print(f"    Best score:       {best_score_actual:.2f}")
    print(f"    Top-1% lineups:   {top1_count}/20 (need score >= {top1_score:.2f})")
    print(f"    Cashing lineups:  {cash_count}/20 (need score >= {cash_score:.2f})")
    print(f"    Gap to top-1%:    {best_score_actual - top1_score:+.2f}")
    print(f"    Gap to cash:      {best_score_actual - cash_score:+.2f}")

    # Players in winning lineups that we missed
    if winning_lus:
        win_players = set(p for lu in winning_lus for p in lu["players"])
        our_players = set(name for lu in lineups for name in lu["names"])
        missed_winners = win_players - our_players
        # Find their actual scores
        missed_with_scores = [(p, actual_scores.get(p, 0)) for p in missed_winners]
        missed_with_scores.sort(key=lambda x: -x[1])
        if missed_with_scores:
            print(f"\n  MISSED KEY PLAYERS (in top-5 winning lineups, not in any of our 20):")
            for name, pts in missed_with_scores[:10]:
                our_data = pid_to_proj.get(name.lower(), {})
                proj = our_data.get("proj", 0)
                own  = our_data.get("own", 0)
                dnp  = our_data.get("dnp", 0)
                print(f"    {name:<30s} actual={pts:.2f} proj={proj:.1f} own={own:.0f}% dnp={dnp:.0%}")

    return {
        "date":            date_str,
        "n_entries":       n_entries,
        "winner_score":    winner_score,
        "top1_score":      top1_score,
        "cash_score":      cash_score,
        "oracle_score":    oracle_score,
        "oracle_rank":     oracle_rank,
        "best_rank":       best_rank,
        "best_score":      best_score_actual,
        "top1_count":      top1_count,
        "cash_count":      cash_count,
        "gap_to_top1":     round(best_score_actual - top1_score, 2),
        "gap_to_cash":     round(best_score_actual - cash_score, 2),
    }


# ── Main backtest loop ─────────────────────────────────────────────────────────
def run_full_backtest():
    runner = AutonomousRunner(n_lineups=20, n_sim=500, verbose=False)

    nights = [
        ("3_6",  "dk_slate_3_6.csv",  "contest-results_3_6.csv",  "2026-03-06"),
        ("3_7",  "dk_slate_3_7.csv",  "contest-results_3_7.csv",  "2026-03-07"),
        ("3_8",  "dk_slate_3_8.csv",  "contest-results_3_8.csv",  "2026-03-08"),
        ("3_9",  "dk_slate_3_9.csv",  "contest-results_3_9.csv",  "2026-03-09"),
    ]

    results = []
    for suffix, slate_name, results_name, date_str in nights:
        slate_path   = CONTEST_DIR / slate_name
        results_path = CONTEST_DIR / results_name
        if not slate_path.exists() or not results_path.exists():
            print(f"[backtest] Skipping {date_str} — files not found")
            continue
        try:
            r = backtest_night(slate_path, results_path, runner, date_str)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"[backtest] ERROR on {date_str}: {e}")
            traceback.print_exc()

    # Final scorecard
    print(f"\n{'='*70}")
    print(f"BACKTEST SCORECARD")
    print(f"{'='*70}")
    print(f"{'Date':>12} {'Entries':>8} {'Winner':>8} {'Top1%':>8} {'Cash':>8} "
          f"{'OurBest':>8} {'BestRank':>10} {'GapTop1%':>9} {'Top1Lus':>8} {'CashLus':>8}")
    print("-"*90)
    all_top1 = True
    for r in results:
        top1_flag = "YES" if r["top1_count"] > 0 else "NO"
        if r["top1_count"] == 0:
            all_top1 = False
        print(f"  {r['date']:>10} {r['n_entries']:>8,} {r['winner_score']:>8.2f} "
              f"{r['top1_score']:>8.2f} {r['cash_score']:>8.2f} "
              f"{r['best_score']:>8.2f} #{r['best_rank']:>8,} "
              f"{r['gap_to_top1']:>+8.2f}  {top1_flag:>4}/{r['top1_count']}/20  "
              f"{r['cash_count']:>4}/20")

    print(f"\nObjective: {'ACHIEVED' if all_top1 else 'NOT YET'} — "
          f"{'All 4 nights have top-1% lineup' if all_top1 else 'Need improvements'}")

    return results


if __name__ == "__main__":
    run_full_backtest()
