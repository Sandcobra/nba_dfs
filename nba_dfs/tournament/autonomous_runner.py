"""
Autonomous Runner
=================
The self-driving engine. No human input required after initial setup.

Runs an infinite loop:
  1. Detect a new slate (DK salary CSV appears or is manually placed)
  2. Pull all available data (injuries, lineups, Vegas lines, usage)
  3. Build player distributions (ScoreDistribution)
  4. Simulate the contest field (ContestSimulator)
  5. Build optimal portfolio (PortfolioOptimizer) targeting max FPE
  6. Export DK upload CSV + analysis report
  7. Wait for game results
  8. Process contest results (SelfImprover) — update calibration
  9. Repeat forever

The runner never stops. It never needs permission. It never makes the same
mistake twice if the SelfImprover can learn from it.

Run:
    cd nba_dfs
    python tournament/autonomous_runner.py

Or as module:
    from tournament import AutonomousRunner
    runner = AutonomousRunner()
    runner.run_forever()
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # nba_dfs/
_REPO = _ROOT.parent  # nba_py/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tournament.score_distribution import ScoreDistribution, _SALARY_TIERS
from tournament.contest_simulator import ContestSimulator
from tournament.portfolio_optimizer import PortfolioOptimizer
from tournament.self_improver import SelfImprover

# ── Directories ───────────────────────────────────────────────────────────────
SLATE_DIR    = _REPO                                     # where dk_slate.csv lands
CONTEST_DIR  = _REPO / "contest"
OUTPUTS_DIR  = _REPO / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
CONTEST_DIR.mkdir(exist_ok=True)

# ── Polling intervals ─────────────────────────────────────────────────────────
POLL_INTERVAL_SLATE   = 30    # seconds — check for new slate CSV
POLL_INTERVAL_RESULTS = 120   # seconds — check for contest results after games

# ── DK Roster ─────────────────────────────────────────────────────────────────
_SLOT_ORDER = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
_SLOT_ELIGIBLE = {
    "PG": ["PG", "G", "UTIL"],
    "SG": ["SG", "G", "UTIL"],
    "SF": ["SF", "F", "UTIL"],
    "PF": ["PF", "F", "UTIL"],
    "C":  ["C", "UTIL"],
}


class AutonomousRunner:
    """
    Self-driving NBA DFS tournament system.

    The runner tracks which slates it has already processed to avoid
    re-running the same slate twice. State is persisted in runner_state.json.
    """

    def __init__(
        self,
        our_username: str = "Sandcobra",
        n_lineups:    int = 20,
        n_sim:        int = 3000,
        verbose:      bool = True,
    ):
        self.our_username = our_username
        self.n_lineups    = n_lineups
        self.n_sim        = n_sim
        self.verbose      = verbose

        self.improver  = SelfImprover()
        self._state    = self._load_state()

        print("[runner] Autonomous NBA DFS Tournament Engine initialized")
        perf = self.improver.get_performance_trend()
        if perf["slates"] > 0:
            print(f"[runner] History: {perf['slates']} slates | "
                  f"Avg best: {perf['avg_best']:.1f}pts | "
                  f"Gap to cash: {perf['avg_gap_to_cash']:+.1f}pts | "
                  f"Gap to win: {perf['avg_gap_to_win']:+.1f}pts")

    # ── State management ───────────────────────────────────────────────────────
    def _load_state(self) -> dict:
        state_file = _HERE / "runner_state.json"
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        return {
            "processed_slates":   [],
            "processed_results":  [],
            "last_slate_file":    None,
            "last_run":           None,
        }

    def _save_state(self):
        state_file = _HERE / "runner_state.json"
        with open(state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    # ── Slate detection ────────────────────────────────────────────────────────
    def _find_new_slate(self) -> Optional[Path]:
        """Look for an unprocessed dk_slate.csv."""
        # Primary path: nba_py/dk_slate.csv
        primary = SLATE_DIR / "dk_slate.csv"
        if primary.exists() and str(primary) not in self._state["processed_slates"]:
            return primary

        # Also check dated versions: dk_slate_YYYY-MM-DD.csv
        for path in sorted(SLATE_DIR.glob("dk_slate*.csv"), reverse=True):
            if str(path) not in self._state["processed_slates"]:
                return path

        return None

    def _find_new_results(self) -> Optional[Path]:
        """Look for unprocessed contest results CSV."""
        for path in sorted(CONTEST_DIR.glob("contest-results_*.csv"), reverse=True):
            if str(path) not in self._state["processed_results"]:
                return path
        return None

    # ── Slate parsing ──────────────────────────────────────────────────────────
    def _parse_slate(self, slate_path: Path, cutoff_date: str = "") -> pd.DataFrame:
        """
        Parse a DraftKings salary CSV into a clean player DataFrame.
        Handles multiple DK CSV formats.

        cutoff_date: "YYYY-MM-DD" — only use game log data before this date
                     (prevents data leakage when backtesting historical slates)
        """
        # Try to import the full test_slate parser first
        try:
            sys.path.insert(0, str(_ROOT))
            from test_slate import parse_salary_file, build_projections
            raw = parse_salary_file(slate_path)
            players = build_projections(raw, cutoff_date=cutoff_date)
            players = players[players["proj_pts_dk"] > 0].copy().reset_index(drop=True)
            print(f"[runner] Parsed {len(players)} players via test_slate")
            return players
        except Exception as e:
            print(f"[runner] test_slate parser failed ({e}), using built-in parser")

        # Built-in fallback parser
        try:
            df = pd.read_csv(slate_path, encoding="utf-8-sig")
        except Exception as e:
            print(f"[runner] Cannot read {slate_path}: {e}")
            return pd.DataFrame()

        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        col_map = {}
        for c in df.columns:
            cl = c.lower().replace(" ", "_")
            if "name" in cl and "id" not in cl:       col_map[c] = "name"
            elif "salary" in cl:                       col_map[c] = "salary"
            elif "position" in cl and "roster" not in cl: col_map[c] = "primary_position"
            elif "team" in cl or "teamabbrev" in cl:   col_map[c] = "team"
            elif "game_info" in cl or "gameinfo" in cl:col_map[c] = "game_info"
            elif "id" in cl and "name" not in cl:      col_map[c] = "player_id_raw"
            elif "avgpointsperga" in cl or "avg_pts" in cl: col_map[c] = "avg_pts"
        df = df.rename(columns=col_map)

        # Ensure required columns
        for col, default in [("avg_pts", 20.0), ("salary", 5000), ("primary_position", "G")]:
            if col not in df.columns:
                df[col] = default

        df["salary"]    = pd.to_numeric(df["salary"], errors="coerce").fillna(5000).astype(int)
        df["avg_pts"]   = pd.to_numeric(df["avg_pts"], errors="coerce").fillna(0)
        df["player_id"] = df.get("player_id_raw", df.index.astype(str)).astype(str)
        df["name_id"]   = df.apply(
            lambda r: f"{r.get('name', r.name)} ({r['player_id']})", axis=1
        )

        # Matchup from game_info
        if "game_info" in df.columns:
            df["matchup"] = df["game_info"].str.extract(r"([A-Z]{2,3}@[A-Z]{2,3})")[0].fillna("")
        else:
            df["matchup"] = ""

        # Home/away
        def _ha(row):
            m = str(row.get("matchup", ""))
            t = str(row.get("team", ""))
            if "@" in m:
                away, home = m.split("@", 1)
                return "home" if t == home.strip() else "away"
            return "home"
        df["home_away"] = df.apply(_ha, axis=1)

        # Eligible slots
        def _slots(pos):
            pos = str(pos)
            slots = []
            for canonical, eligible in _SLOT_ELIGIBLE.items():
                for ep in eligible:
                    if ep in pos:
                        slots.append(canonical)
                        break
            return list(set(slots)) or ["UTIL"]
        df["eligible_slots"] = df["primary_position"].apply(_slots)

        # Base projections
        df["proj_pts_dk"] = df["avg_pts"] * 1.05
        df["proj_std"]    = df["proj_pts_dk"] * 0.30

        # Salary-tiered ceiling multipliers (matches test_slate.build_projections)
        _CEIL_TIERS = [(10000,1.65,30),(8000,1.85,28),(6500,2.10,32),(5000,2.80,40),(0,3.20,40)]
        def _ceil(proj, sal):
            for t, mult, mx in _CEIL_TIERS:
                if sal >= t:
                    return round(min(proj * mult, proj + mx), 2)
            return round(proj * 3.20, 2)
        df["ceiling"] = df.apply(lambda r: _ceil(r["proj_pts_dk"], r["salary"]), axis=1)
        df["floor"]   = (df["proj_pts_dk"] - 1.28 * df["proj_std"]).clip(0)
        df["proj_own"]    = (df["salary"].rank(pct=True) ** 2 * 18 + df["proj_pts_dk"].rank(pct=True) * 8 + 3).clip(1, 40)
        df["game_total"]  = 225.0

        # DNP risk (from calibrated rates)
        def _dnp(sal):
            if sal < 3500:  return self.improver.get_dnp_risk(3000)
            if sal < 4000:  return self.improver.get_dnp_risk(3700)
            if sal < 4500:  return self.improver.get_dnp_risk(4200)
            if sal < 5000:  return self.improver.get_dnp_risk(4700)
            return self.improver.get_dnp_risk(6000)

        df["dnp_risk"] = df["salary"].apply(_dnp)

        # Apply calibration multipliers
        if "archetype" in df.columns:
            df["proj_pts_dk"] = df.apply(
                lambda r: self.improver.adjust_projection(
                    r["proj_pts_dk"], r["archetype"], r["salary"]
                ), axis=1
            )

        # Tournament GPP score
        play_prob = 1.0 - df["dnp_risk"]
        df["gpp_score"] = (
            df["ceiling"]     * 0.65 * play_prob +
            df["proj_pts_dk"] * 0.20 * play_prob +
            (1 - df["proj_own"] / 100) * 8
        ).round(3)
        high_risk = df["dnp_risk"] >= 0.30
        df.loc[high_risk, "gpp_score"] *= 0.80

        df = df[df["proj_pts_dk"] > 0].copy().reset_index(drop=True)
        print(f"[runner] Parsed {len(df)} players from {slate_path.name}")
        return df

    # ── Pipeline: one full slate ───────────────────────────────────────────────
    def run_slate(self, slate_path: Path) -> Optional[Path]:
        """
        Run the full tournament pipeline for a single slate.
        Returns path to the DK upload CSV, or None on failure.
        """
        print(f"\n{'='*70}")
        print(f"[runner] SLATE: {slate_path.name}")
        print(f"[runner] Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        today_str = date.today().strftime("%Y-%m-%d")

        # Step 1: Parse slate
        players = self._parse_slate(slate_path)
        if players.empty:
            print("[runner] ERROR: No players parsed. Skipping.")
            return None

        # Step 2: Apply any signals from test_slate pipeline (B2B, DvP, usage)
        players = self._enrich_players(players, slate_path)

        # Step 3: Build score distributions
        print(f"\n[runner] Building score distributions for {len(players)} players...")
        score_dist = ScoreDistribution()
        score_dist.fit(players)

        # Attach team/matchup to distributions for correlated sampling
        for _, row in players.iterrows():
            score_dist.set_team_matchup(
                str(row["player_id"]),
                str(row.get("team", "")),
                str(row.get("matchup", "")),
            )

        # Step 4: Build contest simulator (field model)
        print(f"[runner] Initializing ContestSimulator (n_sim={self.n_sim})...")
        simulator = ContestSimulator(
            score_dist=score_dist,
            players=players,
            n_sim=self.n_sim,
            seed=int(datetime.now().timestamp()) % (2**31),
        )

        # Step 5: Portfolio optimization
        print(f"\n[runner] Building {self.n_lineups}-lineup tournament portfolio...")
        optimizer = PortfolioOptimizer(
            players=players,
            score_dist=score_dist,
            simulator=simulator,
            max_exposure=0.33,
            pool_multiplier=4,
        )
        lineups = optimizer.build(
            n=self.n_lineups,
            use_simulation=True,
            n_sim=min(self.n_sim, 1000),  # faster eval for final portfolio
        )

        if not lineups:
            print("[runner] ERROR: No lineups generated. Skipping.")
            return None

        # Step 6: Export
        upload_path = OUTPUTS_DIR / f"dk_upload_{today_str}.csv"
        optimizer.export_dk_csv(lineups, upload_path)

        # Step 7: Save lineups JSON + projections
        lineups_path = OUTPUTS_DIR / f"lineups_{today_str}.json"
        # Convert DataFrames to serializable format
        lineups_serial = []
        for lu in lineups:
            lu_out = {k: v for k, v in lu.items() if k != "slot_assignment"}
            lu_out["slot_assignment"] = {k: str(v) for k, v in lu.get("slot_assignment", {}).items()}
            lineups_serial.append(lu_out)
        with open(lineups_path, "w") as f:
            json.dump(lineups_serial, f, indent=2)

        proj_path = OUTPUTS_DIR / f"projections_{today_str}.csv"
        save_cols = [c for c in ["name", "team", "primary_position", "salary",
                                  "avg_pts", "proj_pts_dk", "ceiling", "floor",
                                  "proj_own", "gpp_score", "dnp_risk", "matchup"]
                     if c in players.columns]
        players[save_cols].to_csv(proj_path, index=False)

        # Step 8: Print summary
        self._print_summary(lineups, players)

        # Mark slate as processed
        self._state["processed_slates"].append(str(slate_path))
        self._state["last_slate_file"] = str(slate_path)
        self._state["last_run"] = datetime.now().isoformat()
        self._save_state()

        print(f"\n[runner] FILES:")
        print(f"  DK Upload:   {upload_path}")
        print(f"  Lineups:     {lineups_path}")
        print(f"  Projections: {proj_path}")

        return upload_path

    # ── Enrichment: apply test_slate signals ──────────────────────────────────
    def _enrich_players(self, players: pd.DataFrame, slate_path: Path) -> pd.DataFrame:
        """
        Try to apply B2B adjustments, DvP, usage absorption from test_slate.
        Fails gracefully if test_slate pipeline isn't available.
        """
        try:
            from test_slate import (
                fetch_b2b_teams, apply_b2b_adjustments,
                build_dvp_weights, apply_dvp_adjustments,
                estimate_usage_absorption,
                get_confirmed_starters, apply_status_updates,
                GAME_TOTALS,
            )
            today_matchups = players["matchup"].dropna().unique().tolist()

            # B2B
            b2b_result = fetch_b2b_teams(today_matchups)
            players = apply_b2b_adjustments(players, b2b_result)

            # DvP
            dvp_weights = build_dvp_weights(players)
            players = apply_dvp_adjustments(players, dvp_weights)

            # Usage absorption: needs a specific out_player context — skip batch call

            # Starters — get_confirmed_starters returns {"players": {name: role}, "games": [...]}
            starters = get_confirmed_starters()
            starter_map = starters.get("players", {}) if isinstance(starters, dict) else {}
            if starter_map:
                players = apply_status_updates(players, starter_map)

            print(f"[runner] Enriched {len(players)} players with B2B/DvP/usage/starters")
        except Exception as e:
            print(f"[runner] Enrichment skipped ({e})")

        return players

    # ── Results processing ────────────────────────────────────────────────────
    def process_results(self, results_path: Path) -> dict:
        """Process contest results and update self-improver calibration."""
        today_str = date.today().strftime("%Y-%m-%d")
        proj_path = OUTPUTS_DIR / f"projections_{today_str}.csv"
        if not proj_path.exists():
            # Try yesterday
            yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            proj_path = OUTPUTS_DIR / f"projections_{yesterday}.csv"

        adjustments = self.improver.process_slate(
            contest_csv=results_path,
            projections_csv=proj_path if proj_path.exists() else None,
            our_username=self.our_username,
        )

        self._state["processed_results"].append(str(results_path))
        self._save_state()

        # Print trend
        trend = self.improver.get_performance_trend()
        print(f"\n[runner] Performance trend ({trend['slates']} slates):")
        print(f"  Avg best score:    {trend['avg_best']:.1f} pts")
        print(f"  Avg gap to cash:   {trend['avg_gap_to_cash']:+.1f} pts")
        print(f"  Avg gap to win:    {trend['avg_gap_to_win']:+.1f} pts")

        return adjustments

    # ── Summary printer ───────────────────────────────────────────────────────
    def _print_summary(self, lineups: list, players: pd.DataFrame):
        n = len(lineups)
        avg_proj = sum(lu["proj_pts"] for lu in lineups) / n
        avg_ceil = sum(lu["ceiling"] for lu in lineups) / n
        avg_fpe  = sum(lu.get("fpe", 0) for lu in lineups) / n
        avg_own  = sum(lu.get("avg_own", 0) for lu in lineups) / n
        best_fpe_lu = max(lineups, key=lambda x: x.get("fpe", 0))

        print(f"\n{'='*70}")
        print(f"PORTFOLIO SUMMARY: {n} LINEUPS")
        print(f"{'='*70}")
        print(f"  Avg projection:  {avg_proj:.1f} DK pts")
        print(f"  Avg ceiling:     {avg_ceil:.1f} DK pts")
        print(f"  Avg ownership:   {avg_own:.1f}%")
        print(f"  Avg FPE:         {avg_fpe*100:.4f}%")
        if best_fpe_lu.get("fpe", 0) > 0:
            print(f"  Best FPE lineup: #{best_fpe_lu['lineup_num']} "
                  f"({best_fpe_lu['fpe']*100:.4f}%  "
                  f"P90={best_fpe_lu.get('p90_pts', 0):.0f}pts)")

        print(f"\n  LINEUPS:")
        print(f"  {'#':>3} {'Proj':>6} {'Ceil':>6} {'Own%':>5} {'FPE%':>8} {'Stack'}")
        print(f"  " + "-"*55)
        for lu in lineups:
            fpe_str = f"{lu.get('fpe', 0)*100:.4f}" if lu.get('fpe', 0) > 0 else "  N/A "
            print(f"  #{lu['lineup_num']:>2d} "
                  f"{lu['proj_pts']:>6.1f} "
                  f"{lu['ceiling']:>6.1f} "
                  f"{lu.get('avg_own', 0):>5.1f}% "
                  f"{fpe_str:>8} "
                  f"{lu.get('stack_game', '')}")

        # Exposure report
        exposure: dict[str, int] = {}
        for lu in lineups:
            for pid in lu["player_ids"]:
                exposure[pid] = exposure.get(pid, 0) + 1

        pid_name = dict(zip(players["player_id"].astype(str), players["name"]))
        print(f"\n  EXPOSURE (top 10):")
        for pid, cnt in sorted(exposure.items(), key=lambda x: -x[1])[:10]:
            name = pid_name.get(str(pid), str(pid))
            bar  = "#" * int(cnt / n * 20)
            print(f"  {name:<28s} {cnt:2d}/{n}  {cnt/n*100:>5.1f}%  {bar}")

        print("="*70)

    # ── Endless loop ──────────────────────────────────────────────────────────
    def run_forever(self):
        """
        Run indefinitely: process slates as they appear, handle results as they arrive.
        No human intervention required.
        """
        print(f"\n[runner] Starting autonomous loop — Ctrl+C to stop")
        print(f"[runner] Watching: {SLATE_DIR}")
        print(f"[runner] Results:  {CONTEST_DIR}")

        while True:
            try:
                # Check for new contest results first (update calibration before next slate)
                results_path = self._find_new_results()
                if results_path:
                    print(f"\n[runner] Found new results: {results_path.name}")
                    self.process_results(results_path)

                # Check for new slate
                slate_path = self._find_new_slate()
                if slate_path:
                    print(f"\n[runner] Found new slate: {slate_path.name}")
                    self.run_slate(slate_path)
                else:
                    # No new slate — wait
                    if self.verbose:
                        print(f"[runner] Waiting for slate... ({datetime.now().strftime('%H:%M:%S')})", end="\r")
                    time.sleep(POLL_INTERVAL_SLATE)

            except KeyboardInterrupt:
                print(f"\n[runner] Stopped by user")
                break
            except Exception as e:
                print(f"\n[runner] ERROR: {e}")
                traceback.print_exc()
                print(f"[runner] Retrying in 60s...")
                time.sleep(60)

    # ── Single-run mode (called from main) ───────────────────────────────────
    def run_once(self, slate_path: Optional[Path] = None) -> Optional[Path]:
        """Process a specific slate once (for manual runs)."""
        if slate_path is None:
            slate_path = self._find_new_slate()
        if slate_path is None:
            print("[runner] No slate file found.")
            return None
        return self.run_slate(slate_path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous NBA DFS Tournament Runner")
    parser.add_argument("--once",      action="store_true", help="Run once then exit")
    parser.add_argument("--slate",     type=str,  default=None,  help="Path to dk_slate.csv")
    parser.add_argument("--results",   type=str,  default=None,  help="Path to contest results CSV")
    parser.add_argument("--username",  type=str,  default="Sandcobra", help="DK username")
    parser.add_argument("--lineups",   type=int,  default=20, help="Number of lineups to build")
    parser.add_argument("--sim",       type=int,  default=3000, help="Monte Carlo simulations")
    args = parser.parse_args()

    runner = AutonomousRunner(
        our_username=args.username,
        n_lineups=args.lineups,
        n_sim=args.sim,
    )

    if args.results:
        # Process contest results only
        runner.process_results(Path(args.results))

    elif args.once or args.slate:
        # Single-run mode
        slate = Path(args.slate) if args.slate else None
        runner.run_once(slate)

    else:
        # Endless autonomous mode
        runner.run_forever()
