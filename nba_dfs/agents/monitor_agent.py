"""
Real-Time Lineup Monitor Agent.
Continuously monitors for:
  - Injury reports and lineup changes
  - Starting lineup confirmations
  - Scratch notifications
  - Late lineup swaps to update DFS entries

Runs as a background scheduler and alerts when action is needed.
"""

import asyncio
import time
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from data.injury_scraper import InjuryScraper
from core.config import (
    MONITOR_POLL_INTERVAL_SECS,
    LINEUP_LOCK_MINUTES_BEFORE_TIP,
    LOGS_DIR,
)


class LineupMonitorAgent:
    """
    Monitors injury news, lineup confirmations and fires callbacks
    when significant changes are detected.
    """

    def __init__(
        self,
        on_change_callback: Optional[Callable] = None,
        on_alert_callback:  Optional[Callable] = None,
    ):
        self.scraper             = InjuryScraper()
        self.scheduler           = BackgroundScheduler()
        self.on_change_callback  = on_change_callback or self._default_alert
        self.on_alert_callback   = on_alert_callback or self._default_alert
        self._previous_injuries: dict[str, str] = {}  # name -> status
        self._active_lineups:    list[dict]      = []
        self._confirmed_starters: dict[str, bool] = {}
        self._is_running         = False
        self._alert_log_path     = LOGS_DIR / f"alerts_{date.today().isoformat()}.json"
        self._alerts:            list[dict]      = []

    # ── Start / stop ───────────────────────────────────────────────────────────
    def start(self, player_pool: pd.DataFrame = None):
        """Start the background monitoring scheduler."""
        if self._is_running:
            logger.warning("Monitor already running")
            return

        if player_pool is not None:
            self._init_watchlist(player_pool)

        self.scheduler.add_job(
            self._poll_cycle,
            trigger=IntervalTrigger(seconds=MONITOR_POLL_INTERVAL_SECS),
            id="injury_poll",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._check_lineup_confirmations,
            trigger=IntervalTrigger(seconds=300),  # every 5 min
            id="lineup_confirm",
            replace_existing=True,
        )
        self.scheduler.start()
        self._is_running = True
        logger.success(
            f"Monitor started — polling every {MONITOR_POLL_INTERVAL_SECS}s"
        )

    def stop(self):
        if self._is_running:
            self.scheduler.shutdown(wait=False)
            self._is_running = False
            self.scraper.close()
            logger.info("Monitor stopped")

    def set_active_lineups(self, lineups: list[dict]):
        """Register current DFS lineups to watch for impacted players."""
        self._active_lineups = lineups
        logger.info(f"Monitor watching {len(lineups)} active lineups")

    def _init_watchlist(self, player_pool: pd.DataFrame):
        """Initialize prior injury state from player pool."""
        for _, row in player_pool.iterrows():
            name   = row.get("name", "")
            status = row.get("injury_status", "ACTIVE")
            self._previous_injuries[name] = status

    # ── Poll cycle ─────────────────────────────────────────────────────────────
    def _poll_cycle(self):
        """Main polling function called by scheduler."""
        logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] Polling injury data...")
        try:
            current_injuries = self.scraper.get_all_injury_data()
            if current_injuries.empty:
                return

            changes = self._detect_changes(current_injuries)
            if changes:
                self._process_changes(changes, current_injuries)

        except Exception as e:
            logger.error(f"Poll cycle error: {e}")

    def _detect_changes(self, current: pd.DataFrame) -> list[dict]:
        """Compare current vs previous injury statuses for changes."""
        changes = []
        for _, row in current.iterrows():
            name   = row.get("name", "")
            status = row.get("status", "UNKNOWN")
            prev   = self._previous_injuries.get(name)

            if prev is None:
                self._previous_injuries[name] = status
                continue

            if prev != status:
                change = {
                    "name":        name,
                    "team":        row.get("team", ""),
                    "prev_status": prev,
                    "new_status":  status,
                    "reason":      row.get("reason", ""),
                    "source":      row.get("source", ""),
                    "detected_at": datetime.now().isoformat(),
                    "severity":    self._change_severity(prev, status),
                }
                changes.append(change)
                self._previous_injuries[name] = status

        return changes

    def _change_severity(self, prev: str, new: str) -> str:
        """Categorize the severity of a status change."""
        escalation = {
            "ACTIVE":       0,
            "PROBABLE":     1,
            "GTD":          2,
            "QUESTIONABLE": 3,
            "DOUBTFUL":     4,
            "OUT":          5,
        }
        p = escalation.get(prev, 0)
        n = escalation.get(new, 0)
        if n == 5:     return "CRITICAL"   # newly OUT
        if n > p + 1:  return "HIGH"       # jumped multiple levels
        if n > p:      return "MEDIUM"
        if n < p:      return "RECOVERY"   # status improved
        return "LOW"

    def _process_changes(self, changes: list[dict], current: pd.DataFrame):
        """Handle detected changes — alert and trigger lineup adjustments."""
        for change in changes:
            logger.warning(
                f"INJURY CHANGE | {change['name']} ({change['team']}) | "
                f"{change['prev_status']} → {change['new_status']} | "
                f"{change['reason']} | {change['source']}"
            )
            self._log_alert(change)

            impacted = self._find_impacted_lineups(change["name"])
            if impacted:
                change["impacted_lineups"] = impacted
                logger.warning(
                    f"  !! IMPACTS {len(impacted)} LINEUPS — action may be required"
                )
                self.on_alert_callback(change, impacted)

            # Fire generic change callback
            self.on_change_callback(change)

    def _find_impacted_lineups(self, player_name: str) -> list[int]:
        """Find lineup numbers containing the changed player."""
        return [
            lu["lineup_num"]
            for lu in self._active_lineups
            if player_name in lu.get("player_names", [])
        ]

    # ── Lineup confirmation monitoring ────────────────────────────────────────
    def _check_lineup_confirmations(self):
        """Scrape for confirmed starters and update status."""
        try:
            confirmations = self.scraper.scrape_lineup_confirmations()
            for entry in confirmations:
                name = entry.get("name", "")
                confirmed = entry.get("confirmed", False)
                prev = self._confirmed_starters.get(name)
                if prev != confirmed and confirmed:
                    logger.info(f"CONFIRMED STARTER: {name} ({entry.get('team', '')})")
                    self._confirmed_starters[name] = confirmed
                    self._log_alert({
                        "type":        "CONFIRMATION",
                        "name":        name,
                        "team":        entry.get("team", ""),
                        "confirmed":   confirmed,
                        "detected_at": datetime.now().isoformat(),
                    })
        except Exception as e:
            logger.debug(f"Lineup confirmation check error: {e}")

    # ── Swap recommendations ───────────────────────────────────────────────────
    def recommend_swaps(
        self,
        impacted_lineup: dict,
        player_pool: pd.DataFrame,
        scratched_player: str,
    ) -> list[dict]:
        """
        When a player in our lineup is scratched, recommend the best
        replacement from the remaining player pool at same position.
        Returns top 5 replacement candidates.
        """
        scratched = impacted_lineup["player_names"].index(scratched_player) if scratched_player in impacted_lineup["player_names"] else -1

        if scratched == -1:
            return []

        # Get scratched player's position
        scratch_pos = impacted_lineup.get("positions", [])[scratched] if scratched < len(impacted_lineup.get("positions", [])) else None

        # Remaining salary budget
        lineup_salary = impacted_lineup.get("total_salary", 50000)
        scratch_salary = impacted_lineup.get("salaries", [])[scratched] if scratched < len(impacted_lineup.get("salaries", [])) else 0
        remaining_budget = 50000 - lineup_salary + scratch_salary

        # Available players not in lineup
        already_in = set(impacted_lineup.get("player_names", []))
        available = player_pool[
            (~player_pool["name"].isin(already_in)) &
            (player_pool["salary"] <= remaining_budget) &
            (player_pool.get("injury_status", "ACTIVE").isin(["ACTIVE", "PROBABLE"])) &
            (scratch_pos is None or player_pool["primary_position"].str.contains(scratch_pos, na=False))
        ].copy()

        if available.empty:
            return []

        available["swap_score"] = (
            available.get("projected_pts_dk", 0) * 0.6 +
            available.get("value_dk", 0) * 5 +
            (1 - available.get("proj_ownership", 20) / 100) * 10
        )

        top5 = available.nlargest(5, "swap_score")[
            ["name", "team", "primary_position", "salary",
             "projected_pts_dk", "proj_ownership", "swap_score"]
        ]

        return top5.to_dict("records")

    # ── Alert logging ─────────────────────────────────────────────────────────
    def _log_alert(self, alert: dict):
        self._alerts.append(alert)
        try:
            with open(self._alert_log_path, "w") as f:
                json.dump(self._alerts, f, indent=2)
        except Exception:
            pass

    def get_alerts(self) -> list[dict]:
        return list(self._alerts)

    def get_confirmed_starters(self) -> dict[str, bool]:
        return dict(self._confirmed_starters)

    # ── Default callbacks ─────────────────────────────────────────────────────
    @staticmethod
    def _default_alert(change: dict, impacted: list = None):
        severity = change.get("severity", "UNKNOWN")
        name     = change.get("name", "?")
        msg      = (
            f"\n{'='*60}\n"
            f"ALERT [{severity}] {name} → {change.get('new_status','?')}\n"
            f"  Reason: {change.get('reason','')}\n"
            f"  Source: {change.get('source','')}\n"
        )
        if impacted:
            msg += f"  Impacted lineups: {impacted}\n"
        msg += "=" * 60
        print(msg)

    # ── Time-to-lock warning ───────────────────────────────────────────────────
    def check_time_to_lock(self, tip_times: list[str]) -> list[dict]:
        """
        For each game tip time, warn if we're within LINEUP_LOCK_MINUTES_BEFORE_TIP.
        tip_times: list of ISO datetime strings
        """
        warnings = []
        now = datetime.now()
        for tip in tip_times:
            try:
                tip_dt = datetime.fromisoformat(tip)
                mins_remaining = (tip_dt - now).total_seconds() / 60
                if 0 < mins_remaining <= LINEUP_LOCK_MINUTES_BEFORE_TIP:
                    warnings.append({
                        "tip_time":       tip,
                        "minutes_to_lock": round(mins_remaining, 1),
                        "severity":        "CRITICAL" if mins_remaining < 15 else "HIGH",
                    })
            except Exception:
                pass
        return warnings
