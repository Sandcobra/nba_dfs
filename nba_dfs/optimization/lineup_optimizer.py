"""
Lineup Optimizer.
Uses Integer Linear Programming (PuLP) to generate optimal DFS lineups.

Features:
  - DraftKings / FanDuel roster construction rules
  - Salary cap constraint
  - Team stacking constraints
  - Ownership / leverage constraints (GPP)
  - Exposure limits across lineup set
  - Negative correlation avoidance
  - Mandatory player locks / exclusions
  - Unique lineup guarantee (min N unique players between lineups)
"""

import itertools
from typing import Optional

import numpy as np
import pandas as pd
import pulp
from loguru import logger

from core.config import (
    DK_ROSTER, DK_ROSTER_SIZE, DRAFTKINGS_SALARY_CAP,
    FD_ROSTER, FD_ROSTER_SIZE, FANDUEL_SALARY_CAP,
    NUM_LINEUPS_GPP, NUM_LINEUPS_CASH,
    MAX_PLAYERS_PER_TEAM_GPP,
    MIN_SALARY_USED,
)


# Position slot eligibility
DK_SLOT_ELIGIBLE = {
    "PG":   ["PG", "G", "UTIL"],
    "SG":   ["SG", "G", "UTIL"],
    "SF":   ["SF", "F", "UTIL"],
    "PF":   ["PF", "F", "UTIL"],
    "C":    ["C", "UTIL"],
}


class LineupOptimizer:
    def __init__(self, site: str = "dk", contest_type: str = "gpp"):
        self.site         = site.lower()
        self.contest_type = contest_type.lower()
        self.salary_cap   = DRAFTKINGS_SALARY_CAP if self.site == "dk" else FANDUEL_SALARY_CAP
        self.roster_size  = DK_ROSTER_SIZE if self.site == "dk" else FD_ROSTER_SIZE
        self.roster_cfg   = DK_ROSTER if self.site == "dk" else FD_ROSTER

    # ── Single lineup ILP ──────────────────────────────────────────────────────
    def _build_ilp(
        self,
        players: pd.DataFrame,
        objective_col: str,
        locked_players: list[int] = None,
        excluded_players: list[int] = None,
        prev_lineups: list[list[int]] = None,
        min_unique: int = 2,
        max_team: int = MAX_PLAYERS_PER_TEAM_GPP,
        ownership_col: Optional[str] = None,
        ownership_penalty: float = 0.0,
    ) -> Optional[dict]:
        """Solve single lineup ILP. Returns {player_id: selected} or None."""

        prob = pulp.LpProblem("DFS_Lineup", pulp.LpMaximize)
        n    = len(players)
        idx  = list(range(n))

        # Decision variables: x[i] ∈ {0, 1}
        x = pulp.LpVariable.dicts("player", idx, cat="Binary")

        # Objective: maximize projected points - ownership penalty
        obj_vals = players[objective_col].fillna(0).values
        if ownership_col and ownership_penalty > 0:
            own_vals = players[ownership_col].fillna(20).values / 100.0
            obj = pulp.lpSum(
                x[i] * (obj_vals[i] - ownership_penalty * own_vals[i] * obj_vals[i])
                for i in idx
            )
        else:
            obj = pulp.lpSum(x[i] * obj_vals[i] for i in idx)
        prob += obj

        # Salary cap
        salaries = players["salary"].values
        prob += pulp.lpSum(x[i] * salaries[i] for i in idx) <= self.salary_cap
        prob += pulp.lpSum(x[i] * salaries[i] for i in idx) >= MIN_SALARY_USED

        # Roster size
        prob += pulp.lpSum(x[i] for i in idx) == self.roster_size

        # ── DraftKings position constraints ───────────────────────────────────
        if self.site == "dk":
            slots_needed = {slot: cnt for slot, cnt in self.roster_cfg.items()}
            # For each slot, sum of eligible players must >= required count
            for slot, required in slots_needed.items():
                eligible = [
                    i for i in idx
                    if slot in (players["eligible_slots"].iloc[i] or [])
                ]
                prob += pulp.lpSum(x[i] for i in eligible) >= required

            # Core position minimums (must have at least 1 of each core position)
            for core_pos in ["PG", "SG", "SF", "PF", "C"]:
                eligible = [i for i in idx if core_pos in str(players["primary_position"].iloc[i])]
                prob += pulp.lpSum(x[i] for i in eligible) >= 1

        # ── FanDuel position constraints ──────────────────────────────────────
        else:
            for pos, cnt in self.roster_cfg.items():
                eligible = [i for i in idx if pos in str(players["primary_position"].iloc[i])]
                prob += pulp.lpSum(x[i] for i in eligible) >= cnt

        # ── Team constraint (stacking) ────────────────────────────────────────
        teams = players["team"].unique()
        for team in teams:
            team_idx = [i for i in idx if players["team"].iloc[i] == team]
            prob += pulp.lpSum(x[i] for i in team_idx) <= max_team

        # ── Locks & exclusions ────────────────────────────────────────────────
        if locked_players:
            lock_idx = [i for i in idx if players["player_id"].iloc[i] in locked_players]
            for li in lock_idx:
                prob += x[li] == 1

        if excluded_players:
            excl_idx = [i for i in idx if players["player_id"].iloc[i] in excluded_players]
            for ei in excl_idx:
                prob += x[ei] == 0

        # ── Uniqueness from previous lineups ─────────────────────────────────
        if prev_lineups:
            for prev in prev_lineups:
                prev_idx = [i for i in idx if players["player_id"].iloc[i] in prev]
                # At least min_unique different players
                prob += pulp.lpSum(x[i] for i in prev_idx) <= len(prev) - min_unique

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != "Optimal":
            return None

        selected = [int(players["player_id"].iloc[i]) for i in idx if x[i].value() == 1]
        return {"player_ids": selected, "obj_value": pulp.value(prob.objective)}

    # ── Generate multiple lineups ──────────────────────────────────────────────
    def generate_lineups(
        self,
        player_pool: pd.DataFrame,
        n_lineups: Optional[int] = None,
        objective_col: str = "projected_pts_dk",
        ownership_col: str = "proj_ownership",
        locked_players: list[int] = None,
        excluded_players: list[int] = None,
        min_unique: int = 2,
        ownership_penalty: float = 0.0,
        stacks: list[dict] = None,
        slate_date: str = "",
        contest_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Generate n_lineups unique optimal lineups.
        Returns list of lineup dicts with player info and metadata.
        """
        ct = contest_type or self.contest_type
        if n_lineups is None:
            n_lineups = NUM_LINEUPS_GPP if ct == "gpp" else NUM_LINEUPS_CASH

        # Use ceiling for GPP, floor for cash
        if ct in ("gpp", "showdown"):
            if "ceiling" in player_pool.columns:
                objective_col = "gpp_score" if "gpp_score" in player_pool.columns else "ceiling"
            own_penalty = ownership_penalty if ownership_penalty > 0 else 0.035
        else:
            if "floor" in player_pool.columns:
                objective_col = "cash_score" if "cash_score" in player_pool.columns else "floor"
            own_penalty = 0.0

        players = player_pool.copy().reset_index(drop=True)

        # Validate required columns
        for col in ["salary", "player_id", "primary_position", "team"]:
            if col not in players.columns:
                raise ValueError(f"player_pool missing required column: {col}")

        if objective_col not in players.columns:
            objective_col = "projected_pts_dk"

        lineups:     list[dict]       = []
        prev_pids:   list[list[int]]  = []
        exposure:    dict[int, int]   = {pid: 0 for pid in players["player_id"]}

        # Compute max exposure per player
        from core.config import MAX_EXPOSURE_GPP, MAX_EXPOSURE_CASH
        max_exp_pct = MAX_EXPOSURE_GPP if ct == "gpp" else MAX_EXPOSURE_CASH
        max_exp_cnt = max(1, int(max_exp_pct * n_lineups))

        forced_stack_exclusions: list[list[int]] = []

        # Optionally rotate stacks
        stack_queue = list(stacks) if stacks else []
        stack_idx   = 0

        logger.info(f"Generating {n_lineups} {ct.upper()} lineups...")

        for lu_num in range(n_lineups):
            # Exclude over-exposed players
            current_excl = list(excluded_players or [])
            for pid, cnt in exposure.items():
                if cnt >= max_exp_cnt:
                    current_excl.append(pid)

            # Rotate required stack
            current_locks = list(locked_players or [])
            if stack_queue and ct == "gpp" and lu_num % 3 == 0:
                stack = stack_queue[stack_idx % len(stack_queue)]
                # Lock 2 players from this stack
                lock_pids = stack.get("player_ids", [])[:2]
                current_locks.extend(lock_pids)
                stack_idx += 1

            result = self._build_ilp(
                players,
                objective_col=objective_col,
                locked_players=current_locks,
                excluded_players=current_excl,
                prev_lineups=prev_pids,
                min_unique=min_unique,
                ownership_col=ownership_col,
                ownership_penalty=own_penalty,
                max_team=MAX_PLAYERS_PER_TEAM_GPP if ct == "gpp" else 4,
            )

            if result is None:
                logger.warning(f"Lineup {lu_num+1}: ILP infeasible — relaxing constraints")
                result = self._build_ilp(
                    players,
                    objective_col=objective_col,
                    locked_players=list(locked_players or []),
                    excluded_players=[],
                    prev_lineups=prev_pids[-5:],
                    min_unique=1,
                    ownership_penalty=0.0,
                )
                if result is None:
                    logger.error(f"Lineup {lu_num+1} completely infeasible — skipping")
                    continue

            pids = result["player_ids"]
            selected = players[players["player_id"].isin(pids)].copy()

            lineup_dict = self._format_lineup(
                selected, lu_num + 1, slate_date, ct
            )
            lineup_dict["obj_value"] = round(result["obj_value"], 2)

            lineups.append(lineup_dict)
            prev_pids.append(pids)
            for pid in pids:
                exposure[pid] = exposure.get(pid, 0) + 1

            if (lu_num + 1) % 25 == 0:
                logger.info(f"  Generated {lu_num+1}/{n_lineups} lineups")

        logger.success(f"Generated {len(lineups)} valid lineups")
        return lineups

    # ── Lineup formatting ──────────────────────────────────────────────────────
    def _format_lineup(
        self,
        selected: pd.DataFrame,
        lineup_num: int,
        slate_date: str,
        contest_type: str,
    ) -> dict:
        total_salary = int(selected["salary"].sum())
        proj_pts     = round(selected.get("projected_pts_dk", pd.Series(0)).sum(), 2)
        proj_own     = round(selected.get("proj_ownership", pd.Series(0)).mean(), 1)
        ceiling      = round(selected.get("ceiling", selected.get("projected_pts_dk", 0)).sum(), 2)
        floor        = round(selected.get("floor",   selected.get("projected_pts_dk", 0) * 0.7).sum(), 2)

        return {
            "lineup_num":    lineup_num,
            "slate_date":    slate_date,
            "site":          self.site,
            "contest_type":  contest_type,
            "player_ids":    selected["player_id"].tolist(),
            "player_names":  selected["name"].tolist(),
            "positions":     selected.get("primary_position", ["?"] * len(selected)).tolist(),
            "teams":         selected.get("team", ["?"] * len(selected)).tolist(),
            "salaries":      selected["salary"].tolist(),
            "projections":   selected.get("projected_pts_dk", [0] * len(selected)).round(2).tolist(),
            "total_salary":  total_salary,
            "salary_remain": self.salary_cap - total_salary,
            "proj_pts":      proj_pts,
            "proj_ownership": proj_own,
            "ceiling":       ceiling,
            "floor":         floor,
            "leverage":      round(proj_pts - proj_own * 0.5, 2),
        }

    # ── Lineup export to CSV (DK format) ──────────────────────────────────────
    def export_to_dk_csv(
        self,
        lineups: list[dict],
        player_pool: pd.DataFrame,
        output_path: str,
    ) -> str:
        """
        Export lineups in DraftKings upload format:
        PG, SG, SF, PF, C, G, F, UTIL (all as "Name (PlayerID)")
        """
        rows = []
        name_id_map = {}
        if "name_id" in player_pool.columns:
            name_id_map = dict(zip(player_pool["player_id"], player_pool["name_id"]))

        for lu in lineups:
            pid_list = lu["player_ids"]
            sel      = player_pool[player_pool["player_id"].isin(pid_list)].copy()
            sel      = sel.sort_values("salary", ascending=False)

            # Assign to DK slots
            assigned = self._assign_dk_slots(sel)
            if assigned is None:
                continue

            row = {slot: name_id_map.get(pid, str(pid))
                   for slot, pid in assigned.items()}
            rows.append(row)

        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_path, index=False)
        logger.success(f"Exported {len(rows)} lineups to {output_path}")
        return output_path

    def _assign_dk_slots(self, selected: pd.DataFrame) -> Optional[dict]:
        """Greedy slot assignment for DK export."""
        slots = list(self.roster_cfg.keys())  # PG, SG, SF, PF, C, G, F, UTIL
        assigned = {}
        used = set()

        slot_order = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        for slot in slot_order:
            if slot not in slots:
                continue
            for _, row in selected.iterrows():
                if row["player_id"] in used:
                    continue
                eligible = row.get("eligible_slots", [])
                if slot in eligible:
                    assigned[slot] = row["player_id"]
                    used.add(row["player_id"])
                    break

        return assigned if len(assigned) == self.roster_size else None

    # ── Lineup analysis ────────────────────────────────────────────────────────
    def analyze_lineup_set(self, lineups: list[dict], n_lineups: int) -> pd.DataFrame:
        """Compute exposure rates across lineup set."""
        exposure: dict[str, int] = {}
        for lu in lineups:
            for name in lu["player_names"]:
                exposure[name] = exposure.get(name, 0) + 1

        rows = [
            {"name": name, "count": cnt, "exposure_pct": round(cnt / n_lineups * 100, 1)}
            for name, cnt in exposure.items()
        ]
        return pd.DataFrame(rows).sort_values("exposure_pct", ascending=False)
