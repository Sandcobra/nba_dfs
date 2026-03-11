"""
Self-Improver
=============
Continuously learns from every contest result. After each slate, it:

  1. Compares our projected distributions to actual scores
  2. Identifies systematic biases (we always overproject B2B players by X%)
  3. Updates calibration multipliers stored in a persistent JSON file
  4. Tracks which strategies produce top-1% finishes vs which tank
  5. Adjusts DNP risk thresholds based on actual DNP rates
  6. Updates ownership model accuracy (how far was our proj_own from actual own?)

Over time this creates a continuously improving system that corrects its own
mistakes without human intervention.

Calibration data stored in: nba_dfs/tournament/calibration.json
Slate history stored in:    nba_dfs/tournament/slate_history.json
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

CALIBRATION_FILE  = Path(__file__).parent / "calibration.json"
HISTORY_FILE      = Path(__file__).parent / "slate_history.json"


# Default calibration — no adjustments when starting fresh
_DEFAULT_CALIBRATION = {
    "version":            1,
    "slates_processed":   0,
    "last_updated":       "",
    # Projection multipliers by archetype (1.0 = no adjustment)
    "proj_mult": {
        "BALL_DOM_G":    1.0,
        "COMBO_G":       1.0,
        "CATCH_SHOOT_W": 1.0,
        "POINT_FORWARD": 1.0,
        "STRETCH_BIG":   1.0,
        "RIM_RUNNER":    1.0,
        "PLAYMAKING_BIG":1.0,
        "WING_STOPPER":  1.0,
    },
    # B2B penalty calibration (actual vs model)
    "b2b_proj_bias":     0.0,   # + = we underproject B2B players, - = overproject
    # Ownership model accuracy
    "own_bias":          0.0,   # mean(actual_own - proj_own) across all players
    "own_rmse":          5.0,   # RMSE of ownership predictions
    # DNP rate calibration
    "dnp_rate_by_salary": {
        "sub_3500": 0.45,
        "3500_4000": 0.38,
        "4000_4500": 0.22,
        "4500_5000": 0.10,
        "above_5000": 0.03,
    },
    # Strategy hit rates (top-1% finish rate by strategy)
    "strategy_hit_rates": {
        "ceiling":   0.0,
        "leverage":  0.0,
        "fpe":       0.0,
        "thompson":  0.0,
    },
    # Recent slate scores for momentum tracking
    "recent_best_scores":    [],  # last 10 slates: our best lineup score
    "recent_cash_lines":     [],  # last 10 slates: cash line scores
    "recent_win_scores":     [],  # last 10 slates: winner scores
}


class SelfImprover:
    """
    Reads contest results, compares to projections, and updates calibration.

    Usage:
        improver = SelfImprover()
        improver.process_slate(
            contest_csv=Path("contest/contest-results_3_9.csv"),
            projections_csv=Path("outputs/projections_2026-03-08.csv"),
            our_username="Sandcobra",
        )
        # Calibration is automatically updated and saved
        mult = improver.get_proj_mult("BALL_DOM_G")  # 0.94 if we've been overprojecting
    """

    def __init__(self):
        self.cal = self._load_calibration()
        self.history = self._load_history()

    # ── Calibration I/O ───────────────────────────────────────────────────────
    def _load_calibration(self) -> dict:
        if CALIBRATION_FILE.exists():
            with open(CALIBRATION_FILE) as f:
                cal = json.load(f)
            # Merge with defaults to pick up any new keys
            merged = dict(_DEFAULT_CALIBRATION)
            merged.update(cal)
            return merged
        return dict(_DEFAULT_CALIBRATION)

    def _save_calibration(self):
        self.cal["last_updated"] = datetime.now().isoformat()
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(self.cal, f, indent=2)

    def _load_history(self) -> list:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE) as f:
                return json.load(f)
        return []

    def _save_history(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history[-50:], f, indent=2)  # keep last 50 slates

    # ── Parse contest CSV ─────────────────────────────────────────────────────
    @staticmethod
    def _parse_contest(contest_csv: Path, our_username: str) -> dict:
        """Parse a DK contest results CSV into structured data."""
        player_scores: dict[str, float] = {}
        player_own:    dict[str, float] = {}
        our_entries:   list[dict]       = []
        all_scores:    list[float]      = []

        _slot_re = re.compile(r"\b(PG|SG|SF|PF|C|G|F|UTIL)\b")

        with open(contest_csv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rank   = (row.get("Rank")    or "").strip()
                name   = (row.get("EntryName") or "").strip()
                pts    = (row.get("Points")  or "").strip()
                lineup = (row.get("Lineup")  or "").strip()
                player = (row.get("Player")  or "").strip()
                pct    = (row.get("%Drafted") or "").strip()
                fpts   = (row.get("FPTS")    or "").strip()

                if player and fpts:
                    try:
                        player_scores[player] = float(fpts)
                        pc = pct.replace("%", "")
                        if pc:
                            player_own[player] = float(pc)
                    except (ValueError, TypeError):
                        pass

                if rank and pts and lineup:
                    try:
                        entry_pts = float(pts)
                        all_scores.append(entry_pts)
                        if our_username and our_username.lower() in name.lower():
                            our_entries.append({"rank": int(rank), "pts": entry_pts, "lineup": lineup})
                    except (ValueError, TypeError):
                        pass

        all_scores.sort(reverse=True)
        total = len(all_scores)
        cash_line = all_scores[int(total * 0.25)] if total > 4 else 0
        win_score = all_scores[0] if all_scores else 0

        our_entries.sort(key=lambda x: x["rank"])
        our_best = our_entries[0]["pts"] if our_entries else 0
        our_avg  = sum(e["pts"] for e in our_entries) / len(our_entries) if our_entries else 0
        cashed   = sum(1 for e in our_entries if e["pts"] >= cash_line)

        return {
            "total_entries":  total,
            "cash_line":      cash_line,
            "win_score":      win_score,
            "our_entries":    our_entries,
            "our_best":       our_best,
            "our_avg":        our_avg,
            "cashed":         cashed,
            "player_scores":  player_scores,
            "player_own":     player_own,
        }

    # ── Parse projections CSV ─────────────────────────────────────────────────
    @staticmethod
    def _parse_projections(projections_csv: Path) -> dict:
        """Load our pre-contest projections."""
        if not projections_csv or not projections_csv.exists():
            return {}
        proj = {}
        with open(projections_csv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("name") or "").strip()
                if name:
                    proj[name] = {
                        "salary":    float(row.get("salary", 0) or 0),
                        "avg_pts":   float(row.get("avg_pts", 0) or 0),
                        "proj_pts":  float(row.get("proj_pts_dk", 0) or 0),
                        "ceiling":   float(row.get("ceiling", 0) or 0),
                        "proj_own":  float(row.get("proj_own", 0) or 0),
                        "archetype": (row.get("archetype", "") or "").strip(),
                        "dnp_risk":  float(row.get("dnp_risk", 0) or 0),
                    }
        return proj

    # ── Core processing ───────────────────────────────────────────────────────
    def process_slate(
        self,
        contest_csv: Path,
        projections_csv: Optional[Path] = None,
        our_username: str = "Sandcobra",
        slate_date: str = "",
    ) -> dict:
        """
        Process one slate's results and update all calibration parameters.
        Returns a dict of all calibration adjustments made.
        """
        if not contest_csv.exists():
            print(f"[improver] Contest file not found: {contest_csv}")
            return {}

        print(f"[improver] Processing slate: {contest_csv.name}")
        contest = self._parse_contest(contest_csv, our_username)
        projections = self._parse_projections(projections_csv)

        adjustments = {}

        # ── 1. Update projection accuracy ─────────────────────────────────────
        if projections:
            errors_by_arch: dict[str, list] = {}
            dnp_by_salary: dict[str, list]  = {}
            own_errors: list[float]         = []

            for name, proj in projections.items():
                actual = contest["player_scores"].get(name)
                if actual is None:
                    continue

                proj_pts = proj["proj_pts"]
                salary   = proj["salary"]
                arch     = proj.get("archetype", "COMBO_G")
                proj_own = proj.get("proj_own", 0)
                act_own  = contest["player_own"].get(name, proj_own)

                # Projection bias
                if proj_pts > 0:
                    ratio = actual / proj_pts
                    if arch not in errors_by_arch:
                        errors_by_arch[arch] = []
                    errors_by_arch[arch].append(ratio)

                # Ownership bias
                own_errors.append(act_own - proj_own)

                # DNP detection (0 actual = likely DNP)
                sal_key = self._salary_bucket(salary)
                if sal_key not in dnp_by_salary:
                    dnp_by_salary[sal_key] = []
                dnp_by_salary[sal_key].append(1 if actual == 0 else 0)

            # Update projection multipliers (exponential smoothing, α=0.2)
            alpha = 0.2
            for arch, ratios in errors_by_arch.items():
                if len(ratios) < 2:
                    continue
                mean_ratio = float(np.median(ratios))
                current    = self.cal["proj_mult"].get(arch, 1.0)
                updated    = current * (1 - alpha) + mean_ratio * alpha
                updated    = max(0.70, min(1.40, updated))  # clamp to ±40%
                self.cal["proj_mult"][arch] = round(updated, 4)
                if abs(updated - 1.0) > 0.03:
                    adjustments[f"proj_mult_{arch}"] = round(updated, 4)

            # Update ownership bias
            if own_errors:
                self.cal["own_bias"] = round(float(np.mean(own_errors)), 2)
                self.cal["own_rmse"] = round(float(np.sqrt(np.mean(np.array(own_errors) ** 2))), 2)
                adjustments["own_bias"] = self.cal["own_bias"]

            # Update DNP rates
            for sal_key, dnps in dnp_by_salary.items():
                if len(dnps) >= 3:
                    actual_rate = float(np.mean(dnps))
                    current     = self.cal["dnp_rate_by_salary"].get(sal_key, 0.20)
                    updated     = current * 0.7 + actual_rate * 0.3  # slow update
                    self.cal["dnp_rate_by_salary"][sal_key] = round(updated, 3)
                    adjustments[f"dnp_{sal_key}"] = round(updated, 3)

        # ── 2. Update recent performance tracking ─────────────────────────────
        self.cal["recent_best_scores"].append(contest["our_best"])
        self.cal["recent_cash_lines"].append(contest["cash_line"])
        self.cal["recent_win_scores"].append(contest["win_score"])
        # Keep last 20 slates
        self.cal["recent_best_scores"] = self.cal["recent_best_scores"][-20:]
        self.cal["recent_cash_lines"]  = self.cal["recent_cash_lines"][-20:]
        self.cal["recent_win_scores"]  = self.cal["recent_win_scores"][-20:]

        self.cal["slates_processed"] += 1

        # ── 3. Save to history ────────────────────────────────────────────────
        slate_record = {
            "date":         slate_date or str(date.today()),
            "contest_file": contest_csv.name,
            "total_entries": contest["total_entries"],
            "cash_line":     contest["cash_line"],
            "win_score":     contest["win_score"],
            "our_best":      contest["our_best"],
            "our_avg":       contest["our_avg"],
            "cashed":        contest["cashed"],
            "n_lineups":     len(contest["our_entries"]),
            "adjustments":   adjustments,
        }
        self.history.append(slate_record)

        # ── 4. Persist ────────────────────────────────────────────────────────
        self._save_calibration()
        self._save_history()

        # ── 5. Print report ───────────────────────────────────────────────────
        print(f"\n[improver] Slate calibration updated ({self.cal['slates_processed']} slates total)")
        print(f"  Cash line: {contest['cash_line']:.2f}  |  Our best: {contest['our_best']:.2f}  "
              f"|  Cashed: {contest['cashed']}/{len(contest['our_entries'])}")
        if adjustments:
            print(f"  Adjustments ({len(adjustments)}):")
            for k, v in sorted(adjustments.items()):
                print(f"    {k}: {v}")
        else:
            print("  No significant adjustments needed")

        return adjustments

    # ── Calibrated projection ─────────────────────────────────────────────────
    def adjust_projection(self, raw_proj: float, archetype: str, salary: int, is_b2b: bool = False) -> float:
        """Apply learned calibration multiplier to a raw projection."""
        mult = self.cal["proj_mult"].get(archetype, 1.0)
        adjusted = raw_proj * mult

        # B2B adjustment correction
        if is_b2b:
            b2b_bias = self.cal.get("b2b_proj_bias", 0.0)
            adjusted += b2b_bias

        return max(0.0, round(adjusted, 2))

    def get_dnp_risk(self, salary: int) -> float:
        """Return calibrated DNP probability for a player's salary tier."""
        sal_key = self._salary_bucket(salary)
        return self.cal["dnp_rate_by_salary"].get(sal_key, 0.10)

    def get_proj_mult(self, archetype: str) -> float:
        return self.cal["proj_mult"].get(archetype, 1.0)

    def get_own_bias(self) -> float:
        """How much to correct projected ownership (+ = we underproject ownership)."""
        return self.cal.get("own_bias", 0.0)

    def get_performance_trend(self) -> dict:
        """Returns recent performance metrics for the runner's status display."""
        bests  = self.cal["recent_best_scores"]
        cashes = self.cal["recent_cash_lines"]
        wins   = self.cal["recent_win_scores"]
        n = len(bests)
        if n == 0:
            return {"avg_best": 0, "avg_gap_to_cash": 0, "avg_gap_to_win": 0, "slates": 0}
        avg_best  = sum(bests) / n
        avg_cash  = sum(cashes) / n if cashes else 0
        avg_win   = sum(wins) / n if wins else 0
        return {
            "slates":            self.cal["slates_processed"],
            "avg_best":          round(avg_best, 2),
            "avg_gap_to_cash":   round(avg_best - avg_cash, 2),
            "avg_gap_to_win":    round(avg_best - avg_win, 2),
            "recent_best":       bests,
            "cash_line_trend":   cashes,
        }

    @staticmethod
    def _salary_bucket(salary: int) -> str:
        if salary < 3500:  return "sub_3500"
        if salary < 4000:  return "3500_4000"
        if salary < 4500:  return "4000_4500"
        if salary < 5000:  return "4500_5000"
        return "above_5000"
