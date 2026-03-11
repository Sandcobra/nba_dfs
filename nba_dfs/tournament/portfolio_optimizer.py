"""
Portfolio Optimizer
===================
Builds the optimal 20-lineup portfolio by maximizing First-Place Equity (FPE)
across the entire set — NOT individually optimal lineups.

The key insight: a portfolio of 20 lineups that each have 0.02% FPE is WORSE
than a portfolio where 5 lineups have 0.08% FPE and 15 have 0.01% FPE.
You need a few high-conviction "lottery tickets" alongside diversified coverage.

Strategy:
  1. Generate a large pool of candidate lineups (3x the target count)
     using multiple construction strategies:
     - "Ceiling hunters": max upside regardless of ownership
     - "Leverage plays": contrarian against high-ownership field
     - "Stack exploiters": 4-5 player game stacks for correlated upside
     - "Correlation maximizers": players who go off together
  2. Evaluate every candidate lineup through ContestSimulator
  3. Greedily select the top-N lineups that maximize PORTFOLIO FPE
     (using marginal contribution — not individual FPE)
  4. Enforce constraints: salary, roster construction, exposure caps
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pulp

from tournament.score_distribution import ScoreDistribution
from tournament.contest_simulator import ContestSimulator


# DraftKings constraints
SALARY_CAP   = 50_000
MIN_SALARY   = 49_500
ROSTER_SIZE  = 8
MAX_PER_TEAM = 4

_SLOT_ORDER = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
_SLOT_ELIGIBLE = {
    "PG": ["PG", "G", "UTIL"],
    "SG": ["SG", "G", "UTIL"],
    "SF": ["SF", "F", "UTIL"],
    "PF": ["PF", "F", "UTIL"],
    "C":  ["C", "UTIL"],
}


class PortfolioOptimizer:
    """
    Tournament portfolio construction engine.

    Builds N lineups that collectively maximize the probability of
    at least one lineup finishing in the top 1% of the field.

    Usage:
        opt = PortfolioOptimizer(players_df, score_dist, simulator)
        portfolio = opt.build(n=20)
        # portfolio is a list of 20 lineup dicts
    """

    def __init__(
        self,
        players: pd.DataFrame,
        score_dist: ScoreDistribution,
        simulator: ContestSimulator,
        max_exposure: float = 0.35,
        pool_multiplier: int = 4,
    ):
        self.players       = players.copy().reset_index(drop=True)
        self.score_dist    = score_dist
        self.simulator     = simulator
        self.max_exposure  = max_exposure
        self.pool_mult     = pool_multiplier

        # Build fast lookup structures
        self._pid_list = self.players["player_id"].astype(str).tolist()
        self._pid_set  = set(self._pid_list)
        self._idx_map  = {p: i for i, p in enumerate(self._pid_list)}
        self._sal_map  = dict(zip(self._pid_list, self.players["salary"].astype(int)))
        self._own_map  = dict(zip(self._pid_list, self.players["proj_own"].fillna(15)))
        self._name_map = dict(zip(self._pid_list, self.players["name"]))
        self._pos_map  = dict(zip(self._pid_list, self.players["primary_position"]))
        self._team_map = dict(zip(self._pid_list, self.players["team"]))
        self._matchup_map = dict(zip(self._pid_list, self.players.get("matchup", pd.Series(dtype=str)).fillna("")))
        self._proj_map = dict(zip(self._pid_list, self.players["proj_pts_dk"].fillna(0)))
        self._ceil_map = dict(zip(self._pid_list, self.players.get("ceiling", self.players["proj_pts_dk"]).fillna(0)))
        self._gpp_map  = dict(zip(self._pid_list, self.players.get("gpp_score", self.players["proj_pts_dk"]).fillna(0)))
        self._dnp_map  = dict(zip(self._pid_list, self.players.get("dnp_risk", pd.Series(0.05, index=self.players.index)).fillna(0.05)))
        self._slot_map = dict(zip(self._pid_list, self.players.get("eligible_slots", pd.Series(dtype=object)).fillna(pd.Series([[] for _ in range(len(self.players))]))))

        # Identify games and top players per game
        self._games = self.players["matchup"].dropna().unique().tolist()
        self._game_tops = self._identify_game_anchors()

    # ── Game anchor identification ────────────────────────────────────────────
    def _identify_game_anchors(self) -> dict[str, list[str]]:
        """
        Top 3 players by GPP score per game — used to ensure every game gets coverage.
        Uses gpp_score (ceiling-first + value-per-dollar bonus) not raw ceiling so
        cheap explosion plays like Jaylin Williams can anchor OKC-game lineups.
        """
        anchors = {}
        sort_col = "gpp_score" if "gpp_score" in self.players.columns else "ceiling"
        for game in self._games:
            game_players = self.players[self.players["matchup"] == game].copy()
            game_players = game_players.sort_values(sort_col, ascending=False)
            anchors[game] = game_players["player_id"].astype(str).head(3).tolist()
        return anchors

    # ── ILP lineup builder ────────────────────────────────────────────────────
    def _build_ilp_lineup(
        self,
        objective_vals: dict[str, float],
        prev_lineups: list[list[str]] = None,
        locked_pids: list[str] = None,
        excluded_pids: list[str] = None,
        min_unique: int = 3,
        require_game_stack: str = None,  # matchup string — force 3+ players from this game
    ) -> Optional[list[str]]:
        """
        Build a single DK lineup using ILP with the given objective values.
        objective_vals: {player_id -> score to maximize}
        """
        prob = pulp.LpProblem("Portfolio_Lineup", pulp.LpMaximize)
        eligible = [p for p in self._pid_list if p not in (excluded_pids or [])]
        idx = {p: pulp.LpVariable(f"x_{i}", cat="Binary") for i, p in enumerate(eligible)}

        # Objective
        prob += pulp.lpSum(idx[p] * objective_vals.get(p, 0) for p in eligible)

        # Roster size
        prob += pulp.lpSum(idx.values()) == ROSTER_SIZE

        # Salary cap
        prob += pulp.lpSum(idx[p] * self._sal_map[p] for p in eligible) <= SALARY_CAP
        prob += pulp.lpSum(idx[p] * self._sal_map[p] for p in eligible) >= MIN_SALARY

        # Position constraints (DK slot filling)
        slot_vars = {}
        for p in eligible:
            slots = self._slot_map.get(p, [])
            if not isinstance(slots, list):
                try:
                    slots = list(slots)
                except Exception:
                    slots = []
            for slot in slots:
                if slot in _SLOT_ORDER:
                    key = (p, slot)
                    slot_vars[key] = pulp.LpVariable(f"y_{p}_{slot}", cat="Binary")

        # Each slot filled exactly once
        for slot in _SLOT_ORDER:
            slot_fills = [slot_vars[(p, slot)] for p in eligible if (p, slot) in slot_vars]
            if slot_fills:
                prob += pulp.lpSum(slot_fills) == 1

        # Each selected player fills exactly one slot
        for p in eligible:
            player_slot_vars = [slot_vars[(p, s)] for s in _SLOT_ORDER if (p, s) in slot_vars]
            if player_slot_vars:
                prob += pulp.lpSum(player_slot_vars) == idx[p]
            else:
                prob += idx[p] == 0

        # Team cap
        for team in self.players["team"].unique():
            team_pids = [p for p in eligible if self._team_map.get(p) == team]
            if team_pids:
                prob += pulp.lpSum(idx[p] for p in team_pids) <= MAX_PER_TEAM

        # Minimum 3-player game stack (at least one game provides 3+ players)
        matchup_z = {}
        for j, game in enumerate(self._games):
            game_pids = [p for p in eligible if self._matchup_map.get(p) == game]
            if len(game_pids) >= 3:
                z = pulp.LpVariable(f"zg_{j}", cat="Binary")
                matchup_z[game] = z
                prob += 3 * z <= pulp.lpSum(idx[p] for p in game_pids)
                prob += z >= (pulp.lpSum(idx[p] for p in game_pids) - 2) / 8
        if matchup_z:
            prob += pulp.lpSum(matchup_z.values()) >= 1

        # Required game stack (for stack rotation)
        if require_game_stack and require_game_stack in matchup_z:
            prob += matchup_z[require_game_stack] >= 1

        # Locks
        for p in (locked_pids or []):
            if p in idx:
                prob += idx[p] == 1

        # Uniqueness from previous lineups
        if prev_lineups:
            for prev in prev_lineups:
                prev_eligible = [p for p in prev if p in idx]
                if len(prev_eligible) >= min_unique:
                    prob += pulp.lpSum(idx[p] for p in prev_eligible) <= len(prev_eligible) - min_unique

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=20)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != "Optimal":
            return None

        return [p for p in eligible if idx[p].value() and idx[p].value() > 0.5]

    # ── Ceiling-first lineup ──────────────────────────────────────────────────
    def _ceiling_lineup(self, prev: list, locked: list = None, excl: list = None, game: str = None) -> Optional[list[str]]:
        """Build a lineup maximizing salary-tiered ceiling * (1 - dnp)."""
        obj = {p: self._ceil_map.get(p, 0) * (1 - self._dnp_map.get(p, 0.05)) for p in self._pid_list}
        return self._build_ilp_lineup(obj, prev, locked, excl, min_unique=3, require_game_stack=game)

    # ── Leverage lineup ───────────────────────────────────────────────────────
    def _leverage_lineup(self, prev: list, locked: list = None, excl: list = None, game: str = None) -> Optional[list[str]]:
        """Build a lineup maximizing ceiling / sqrt(ownership) — contrarian leverage."""
        obj = {}
        for p in self._pid_list:
            own  = max(self._own_map.get(p, 15), 1)
            ceil = self._ceil_map.get(p, 0) * (1 - self._dnp_map.get(p, 0.05))
            obj[p] = ceil / (own ** 0.5)
        return self._build_ilp_lineup(obj, prev, locked, excl, min_unique=3, require_game_stack=game)

    # ── FPE-guided lineup ─────────────────────────────────────────────────────
    def _fpe_lineup(
        self,
        prev: list,
        locked: list = None,
        excl: list = None,
        game: str = None,
        n_sim_fast: int = 500,
    ) -> Optional[list[str]]:
        """
        Build a lineup using FPE proxy: salary-tiered ceiling × explosion rate × contrarianism.
        Ceiling formula already captures cheap explosion upside (Jaylin $5700 → ceil=58 from 2.80x mult).
        """
        obj = {}
        for p in self._pid_list:
            dist = self.score_dist.get(p)
            own  = max(self._own_map.get(p, 15), 1) / 100.0
            dnp  = self._dnp_map.get(p, 0.05)
            ceil = self._ceil_map.get(p, 0)
            expl = dist.get("explosion_rate", 0.10) if dist else 0.10
            obj[p] = ceil * (1 + expl * 2) * (1 - own * 0.8) * (1 - dnp)
        return self._build_ilp_lineup(obj, prev, locked, excl, min_unique=3, require_game_stack=game)

    # ── Cheap stack + studs lineup ────────────────────────────────────────────
    def _cheap_stack_lineup(
        self,
        prev: list,
        game: str,
        excl: list = None,
        stud_locked: list = None,
        rng: np.random.Generator = None,
        ultra_cheap: bool = False,
    ) -> Optional[list[str]]:
        """
        Build a 'cheap game stack + premium studs' lineup.

        Selects 3-4 players from a game by GPP VALUE (ceiling/salary ratio), not just
        lowest salary. This captures plays like Javon Small ($5400, high gpp_score)
        over Ziaire Williams ($4300, low gpp_score) even though Small costs more.

        Pattern: 3-4 value plays from one game (combined ~$18-22K) free salary for
        2-3 elite studs (SGA + KAT level) from other games that normally can't coexist.

        Only runs when the game has ≥3 qualifying cheap-value candidates AND the
        remaining salary supports premium studs ($7200+ avg per remaining slot).
        """
        excl = excl or []

        # Cheap-value candidates: salary ≤ $7500 (mid-tier value), viable DNP risk,
        # minimum projection, and meaningful gpp_score (real upside potential).
        # Raised to $7500 to capture Tyler Herro ($7400), Russell Westbrook ($7000)
        # who appear in winning lineups but were excluded at $6500.
        MAX_STACK_SAL = 7500   # above this = premium player, not cheap stack
        MIN_PROJ      = 6.0    # lowered to capture contrarian plays (Marcus Sasser 8.5, Rayan Rupert 8.9)
        MAX_DNP       = 0.25   # slightly relaxed: allows Marcus Sasser (dnp=20%) into IND stack
        MIN_GPP       = 10.0   # lower bar — more games qualify, better game coverage
        MIN_QUAL      = 3      # lowered: 3 qualifying players sufficient for a game stack

        game_players = []
        for p in self._pid_list:
            if (self._matchup_map.get(p) == game
                    and p not in excl
                    and self._sal_map.get(p, 0) <= MAX_STACK_SAL
                    and self._dnp_map.get(p, 0.05) <= MAX_DNP
                    and self._proj_map.get(p, 0) >= MIN_PROJ
                    and self._gpp_map.get(p, 0) >= MIN_GPP):
                game_players.append((
                    p,
                    self._gpp_map[p],   # rank by gpp_score (value), not salary
                    self._sal_map[p],
                ))

        # Quality gate: game must have MIN_QUAL cheap-value players to justify a stack.
        if len(game_players) < MIN_QUAL:
            return None  # not enough cheap value plays in this game → skip

        require_game_stack_for_ilp = None  # ILP can stack any game
        if rng is not None:
            # STOCHASTIC mode: sample 3 players with contrarian weighting.
            # Low-gpp + low-ownership = higher probability of selection.
            # This captures mid-tier value plays (Monk, Sexton, Westbrook in CHI@SAC)
            # that go off at 1.5-2.5x projection — the tournament bread-and-butter.
            weights = []
            for p, gpp, sal in game_players:
                own = max(self._own_map.get(p, 15), 1)
                # Contrarian weight: reward low-ownership, low-gpp (cheap, ignored plays)
                w = (100 - own) / max(gpp, 5)
                weights.append(max(w, 0.01))
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            n_lock = min(3, len(game_players))
            chosen = rng.choice(len(game_players), size=n_lock, replace=False, p=probs)
            locked = [game_players[int(i)][0] for i in chosen]
        elif ultra_cheap:
            # ULTRA-CHEAP mode: sort by salary ascending to maximize salary freed for two studs.
            # e.g. MEM@BKN: picks Rupert($4K)+Traore($4.4K)+Small($5.4K) over Edey($6.2K)
            # freeing salary for BOTH Jokic($12.4K)+SGA($10.7K) — the 3/9 winning structure.
            game_players.sort(key=lambda x: x[2])   # salary ASC
            locked = [p for p, _, _ in game_players[:3]]
        else:
            # DETERMINISTIC mode: top 3 by gpp_score (highest ceiling + value).
            game_players.sort(key=lambda x: -x[1])
            locked = [p for p, _, _ in game_players[:3]]
            # Add 4th if salary within $1200 of 3rd (still "cheap" tier).
            if len(game_players) >= 4:
                third_sal  = game_players[2][2]
                fourth_sal = game_players[3][2]
                if fourth_sal <= third_sal + 1200:
                    locked.append(game_players[3][0])

        # Salary sanity: stack must leave enough budget for premium studs.
        # Raised to 52% (was 44%) to accommodate $7000-7500 value plays in stack.
        # 3 players at $7000 avg = $21000 = 42% — well within new limit.
        stack_sal = sum(self._sal_map[p] for p in locked)
        if stack_sal > SALARY_CAP * 0.52:
            return None  # stack too expensive to pair with elite studs

        # Combine cheap stack players + optional stud anchors from other games.
        # Raised to 82% (was 70%) to allow dual elite studs with cheap stack:
        # 3 IND cheapies ($14K=28%) + Giannis ($10.5K=21%) + JJ ($10.2K=20.4%) = 81.8%
        # This enables the Giannis+JJ+4xIND winning lineup structure for 3/7.
        all_locked = locked.copy()
        for s in (stud_locked or []):
            if s not in all_locked and s not in excl:
                stud_sal = self._sal_map.get(s, 0)
                if stack_sal + stud_sal <= SALARY_CAP * 0.82:
                    all_locked.append(s)
                    stack_sal += stud_sal

        # Cheap-stack ILP objective: gpp_score with mild ownership penalty.
        # Pure gpp_score causes ILP to over-rotate toward chalk high-gpp players
        # (e.g., Giddey gpp=50.7 over Westbrook gpp=45.8) even in cheap_stack mode.
        # A 30% ownership penalty brings mid-tier value plays (Monk, Westbrook, Raynaud)
        # into contention — exactly the players who boom in GPP tournaments.
        obj = {
            p: self._gpp_map.get(p, 0) * (1 - self._own_map.get(p, 15) / 100 * 0.3)
            for p in self._pid_list
        }

        # Penalize expensive NON-LOCKED same-game players in the ILP objective.
        # This prevents the ILP from "upgrading" the stack by adding, e.g., Edey($6.2K)
        # to a MEM@BKN lineup where we locked Rupert+Traore+Small to free salary for
        # two elite studs (Jokic+SGA). Without this penalty, ILP always adds Edey since
        # he has the highest MEM@BKN gpp_score among non-locked players.
        STACK_EXPENSIVE_SAL = 5800  # above this → penalize in stack game context
        for p in self._pid_list:
            if (game
                    and self._matchup_map.get(p) == game
                    and p not in all_locked
                    and self._sal_map.get(p, 0) > STACK_EXPENSIVE_SAL):
                obj[p] = obj.get(p, 0) * 0.30  # heavy penalty: ILP avoids unless forced

        return self._build_ilp_lineup(obj, prev, all_locked, excl, min_unique=3,
                                      require_game_stack=require_game_stack_for_ilp)

    # ── Thompson sampling lineup ──────────────────────────────────────────────
    def _thompson_lineup(
        self,
        prev: list,
        locked: list = None,
        excl: list = None,
        game: str = None,
    ) -> Optional[list[str]]:
        """
        Thompson sampling: draw random scenario from each player's distribution,
        build lineup optimal for that scenario. Natural diversity generator.
        """
        rng = np.random.default_rng()
        obj = {}
        for p in self._pid_list:
            samples = self.score_dist.sample(p, n=1, rng=rng)
            obj[p] = float(samples[0]) * (1 - self._dnp_map.get(p, 0.05))
        return self._build_ilp_lineup(obj, prev, locked, excl, min_unique=3, require_game_stack=game)

    # ── Dual cheap-game stack lineup ──────────────────────────────────────────
    def _double_stack_lineup(
        self,
        prev: list,
        excl: list = None,
        rng=None,
    ) -> Optional[list[str]]:
        """
        Dual cheap-game stack: 3 players from game A + 3 players from game B.
        Captures winning patterns where two correlated cheap games both explode,
        e.g., 3/8 winning lineup used CHI@SAC (Westbrook+Sexton+Monk) × ORL@MIL
        (Bane+Murphy+Huff) — impossible to generate with single cheap_stack logic.
        Slightly higher salary limit ($8500) to include mid-tier value plays.
        """
        excl = excl or []
        MAX_DS_SAL = 8500
        MIN_PROJ   = 6.0
        MAX_DNP    = 0.25
        MIN_GPP    = 10.0

        # Score each game by top-3 combined gpp among cheap-eligible players
        game_scores: dict[str, float] = {}
        for game in (self._games or []):
            gps = [
                (p, self._gpp_map.get(p, 0))
                for p in self._pid_list
                if self._matchup_map.get(p) == game
                and p not in excl
                and self._sal_map.get(p, 0) <= MAX_DS_SAL
                and self._dnp_map.get(p, 0.05) <= MAX_DNP
                and self._proj_map.get(p, 0) >= MIN_PROJ
                and self._gpp_map.get(p, 0) >= MIN_GPP
            ]
            if len(gps) >= 3:
                top3 = sorted(gps, key=lambda x: -x[1])[:3]
                game_scores[game] = sum(g for _, g in top3)

        if len(game_scores) < 2:
            return None

        sorted_games = sorted(game_scores, key=lambda g: -game_scores[g])
        game1, game2 = sorted_games[0], sorted_games[1]

        def _sample(game: str, n: int):
            players = [
                (p, self._gpp_map.get(p, 0))
                for p in self._pid_list
                if self._matchup_map.get(p) == game
                and p not in excl
                and self._sal_map.get(p, 0) <= MAX_DS_SAL
                and self._dnp_map.get(p, 0.05) <= MAX_DNP
                and self._proj_map.get(p, 0) >= MIN_PROJ
                and self._gpp_map.get(p, 0) >= MIN_GPP
            ]
            if len(players) < n:
                return None
            if rng is not None:
                ws = []
                for p, gpp in players:
                    own = max(self._own_map.get(p, 15), 1)
                    ws.append(max((100 - own) / max(gpp, 5), 0.01))
                total_w = sum(ws)
                probs = [w / total_w for w in ws]
                chosen = rng.choice(len(players), size=min(n, len(players)), replace=False, p=probs)
                return [players[int(i)][0] for i in chosen]
            else:
                players.sort(key=lambda x: -x[1])
                return [p for p, _ in players[:n]]

        locked1 = _sample(game1, 3)
        locked2 = _sample(game2, 3)
        if not locked1 or not locked2:
            return None

        all_locked = locked1 + locked2
        stack_sal = sum(self._sal_map.get(p, 0) for p in all_locked)
        if stack_sal > SALARY_CAP * 0.66:   # 6 players at avg $5.5K max = $33K
            return None

        obj = {
            p: self._gpp_map.get(p, 0) * (1 - self._own_map.get(p, 15) / 100 * 0.3)
            for p in self._pid_list
        }
        return self._build_ilp_lineup(obj, prev, all_locked, excl, min_unique=3)

    # ── Elite + cheap same-team stack ────────────────────────────────────────
    def _elite_team_stack_lineup(
        self,
        prev: list,
        game: str,
        excl: list = None,
        rng=None,
    ) -> Optional[list[str]]:
        """
        Elite team stack: lock the top-gpp player from a game (no salary limit) +
        2 cheap same-team players (≤$6500) as correlated bring-back support.

        This captures winning structures like 3/9 DEN@OKC where SGA ($10.7K) +
        Jaylin Williams ($5.7K) + Rayan Rupert ($3.5K) form the OKC team core.
        With those 3 locked (total ~$19.9K), the ILP fills 5 slots from ~$30.1K,
        which forces it into cheap KAT/Santos/Wolf territory — the winning fill.
        """
        excl = excl or []
        MAX_CHEAP_SAL = 6500
        MIN_CHEAP_PROJ = 6.0
        MAX_DNP_STUD = 0.20
        MAX_DNP_CHEAP = 0.25

        # Top gpp player from the game (no salary filter) — the stud anchor
        game_all = [
            (p, self._gpp_map.get(p, 0))
            for p in self._pid_list
            if self._matchup_map.get(p) == game
            and p not in excl
            and self._dnp_map.get(p, 0.05) <= MAX_DNP_STUD
            and self._proj_map.get(p, 0) >= 15.0
        ]
        if not game_all:
            return None
        stud, stud_gpp = max(game_all, key=lambda x: x[1])
        stud_team = self._team_map.get(stud)
        if not stud_team:
            return None

        # 2 cheap same-team players (the "brings back" the stud's team correlation)
        teammates = [
            (p, self._gpp_map.get(p, 0))
            for p in self._pid_list
            if self._team_map.get(p) == stud_team
            and p != stud
            and p not in excl
            and self._sal_map.get(p, 0) <= MAX_CHEAP_SAL
            and self._dnp_map.get(p, 0.05) <= MAX_DNP_CHEAP
            and self._proj_map.get(p, 0) >= MIN_CHEAP_PROJ
        ]
        if len(teammates) < 2:
            return None

        if rng is not None:
            ws = []
            for p, gpp in teammates:
                own = max(self._own_map.get(p, 15), 1)
                ws.append(max((100 - own) / max(gpp, 5), 0.01))
            total_w = sum(ws)
            probs = [w / total_w for w in ws]
            chosen = rng.choice(len(teammates), size=2, replace=False, p=probs)
            cheap_mates = [teammates[int(i)][0] for i in chosen]
        else:
            teammates.sort(key=lambda x: -x[1])
            cheap_mates = [p for p, _ in teammates[:2]]

        locked = [stud] + cheap_mates
        stack_sal = sum(self._sal_map.get(p, 0) for p in locked)
        if stack_sal > SALARY_CAP * 0.54:  # max 3 locked players at 54% ($27K)
            return None

        obj = {
            p: self._gpp_map.get(p, 0) * (1 - self._own_map.get(p, 15) / 100 * 0.3)
            for p in self._pid_list
        }
        return self._build_ilp_lineup(obj, prev, locked, excl, min_unique=3)

    # ── Single stud + all budget plays ────────────────────────────────────────
    def _single_stud_value_lineup(
        self,
        prev: list,
        game: str,
        excl: list = None,
        rng=None,
    ) -> Optional[list[str]]:
        """
        Lock game's top-gpp player (no salary limit) + fill 7 slots with
        players ≤$7500 only.  Captures winning structures like 3/9 where
        Jokic ($12.4K) + Williams+Rupert+Melton+Wolf+Small+... = 358+ pts.
        Forces the ILP into budget territory without a second expensive stud.
        """
        excl = excl or []
        MAX_BUDGET_SAL = 7500

        # Top-gpp player from this game (the one expensive stud allowed)
        game_all = [
            (p, self._gpp_map.get(p, 0))
            for p in self._pid_list
            if self._matchup_map.get(p) == game
            and p not in excl
            and self._dnp_map.get(p, 0.05) <= 0.15
            and self._proj_map.get(p, 0) >= 15.0
        ]
        if not game_all:
            return None

        # Cycle through top-3 studs for variety across runs
        game_all.sort(key=lambda x: -x[1])
        cycle_idx = 0
        if rng is not None:
            cycle_idx = rng.integers(0, min(3, len(game_all)))
        stud = game_all[min(cycle_idx, len(game_all) - 1)][0]

        # Exclude all expensive players except the chosen stud
        exclude_expensive = [
            p for p in self._pid_list
            if self._sal_map.get(p, 0) > MAX_BUDGET_SAL and p != stud
        ]
        excl_full = list(set(excl + exclude_expensive))

        obj = {
            p: self._gpp_map.get(p, 0) * (1 - self._own_map.get(p, 15) / 100 * 0.35)
            for p in self._pid_list
        }
        return self._build_ilp_lineup(obj, prev, [stud], excl_full, min_unique=3)

    # ── Game sweep: stud + 3 cheap same-game players ──────────────────────────
    def _game_sweep_lineup(
        self,
        prev: list,
        game: str,
        excl: list = None,
        rng=None,
    ) -> Optional[list[str]]:
        """
        Lock game's top-gpp stud (no salary limit) + 3 cheap same-game players
        (≤$7500, both teams) + ILP fills 4 slots from anywhere.
        Captures max game correlation: 3/8 Giddey+Westbrook+Raynaud+Monk,
        3/9 Jokic+Gordon+Williams+Rupert, 3/6 Thompson+Herro+Jaquez+Miller.
        """
        excl = excl or []
        MAX_SWEEP_SAL = 7500

        # Pick stud: top-gpp player from game with decent proj (no salary limit)
        game_all = [
            (p, self._gpp_map.get(p, 0))
            for p in self._pid_list
            if self._matchup_map.get(p) == game
            and p not in excl
            and self._dnp_map.get(p, 0.05) <= 0.15
            and self._proj_map.get(p, 0) >= 15.0
        ]
        if not game_all:
            return None
        game_all.sort(key=lambda x: -x[1])
        cycle_idx = 0
        if rng is not None:
            cycle_idx = rng.integers(0, min(3, len(game_all)))
        stud = game_all[min(cycle_idx, len(game_all) - 1)][0]

        # Pick 3 cheap from same game (not stud)
        cheap_game = [
            (p, self._gpp_map.get(p, 0))
            for p in self._pid_list
            if self._matchup_map.get(p) == game
            and p not in excl
            and p != stud
            and self._sal_map.get(p, 0) <= MAX_SWEEP_SAL
            and self._dnp_map.get(p, 0.05) <= 0.25
            and self._proj_map.get(p, 0) >= 6.0
            and self._gpp_map.get(p, 0) >= 8.0
        ]
        if len(cheap_game) < 3:
            return None

        if rng is not None:
            # Contrarian sampling: weight inversely by ownership relative to gpp
            weights = np.array(
                [(100 - self._own_map.get(p, 15)) / max(g, 5) for p, g in cheap_game],
                dtype=float,
            )
            weights = np.clip(weights, 0.01, None)
            weights /= weights.sum()
            chosen_idx = rng.choice(len(cheap_game), size=3, replace=False, p=weights)
            cheap_locked = [cheap_game[i][0] for i in chosen_idx]
        else:
            cheap_locked = [p for p, g in sorted(cheap_game, key=lambda x: -x[1])[:3]]

        all_locked = [stud] + cheap_locked
        obj = {
            p: self._gpp_map.get(p, 0) * (1 - self._own_map.get(p, 15) / 100 * 0.3)
            for p in self._pid_list
        }
        return self._build_ilp_lineup(obj, prev, all_locked, excl, min_unique=3)

    # ── All-value lineup (no stud) ─────────────────────────────────────────────
    def _all_value_lineup(
        self,
        prev: list,
        excl: list = None,
    ) -> Optional[list[str]]:
        """
        All-value lineup: excludes all players above $8000 salary.
        Targets winning nights like 3/8 where NO player above $8K appeared in the
        winning lineup — just 8 correlated value plays all going off together.
        Contrarian ownership weighting favors low-owned mid-tier plays.
        """
        excl = list(excl or [])
        high_sal = [p for p in self._pid_list if self._sal_map.get(p, 0) > 8000]
        excl_full = list(set(excl + high_sal))

        obj = {
            p: self._gpp_map.get(p, 0) * (1 - self._own_map.get(p, 15) / 100 * 0.35)
            for p in self._pid_list
        }
        return self._build_ilp_lineup(obj, prev, [], excl_full, min_unique=3)

    # ── Cross-game scatter contrarian lineup ──────────────────────────────────
    def _scatter_contrarian_lineup(
        self,
        prev: list,
        excl: list = None,
        sc_pass: int = 0,
    ) -> Optional[list[str]]:
        """
        Cross-game contrarian portfolio builder.

        Picks the most contrarian (lowest ownership) viable player from each
        game on the slate, then lets ILP select 1-2 premium studs to fill the
        roster.  Captures multi-game explosion scenarios that single-game stacks
        miss — e.g., MEM@BKN + GSW@UTA both blowing up (3/9 winning structure).

        Pass-based cycling rotates which game's contrarian is used (sc_pass
        shifts which player rank is selected per game so repeated calls produce
        different lineups).
        """
        excl = excl or []

        # Per-game: find top-N contrarians sorted by ownership ASC
        game_contrarians: list[tuple[str, str, float, float]] = []
        for g in self._games:
            candidates = [
                (p, self._own_map.get(p, 50), self._gpp_map.get(p, 0))
                for p in self._pid_list
                if self._matchup_map.get(p) == g
                and p not in excl
                and self._dnp_map.get(p, 0.05) <= 0.15
                and self._proj_map.get(p, 0) >= 8.0
                and self._gpp_map.get(p, 0) >= 18.0
                and self._sal_map.get(p, 0) <= 7500
            ]
            if candidates:
                candidates.sort(key=lambda x: x[1])  # lowest ownership first
                # Cycle through ranks using sc_pass
                rank = sc_pass % max(1, len(candidates))
                p, own, gpp = candidates[rank]
                game_contrarians.append((g, p, own, gpp))

        if len(game_contrarians) < 3:
            return None

        # Sort by ownership ASC, take 3 most contrarian (1 per game)
        game_contrarians.sort(key=lambda x: x[2])
        locked = [x[1] for x in game_contrarians[:3]]

        # Add 1-2 premium studs (salary ≥ $8K, lowest ownership among elites)
        stud_candidates = [
            (p, self._own_map.get(p, 50), self._gpp_map.get(p, 0))
            for p in self._pid_list
            if p not in excl and p not in locked
            and self._sal_map.get(p, 0) >= 8000
            and self._dnp_map.get(p, 0.05) <= 0.10
        ]
        # Prefer studs from games NOT already covered by contrarians
        covered_games = {self._matchup_map.get(p) for p in locked}
        uncovered_studs = [(p, o, g) for p, o, g in stud_candidates
                          if self._matchup_map.get(p) not in covered_games]
        # Sort uncovered studs by gpp DESC (want best stud from uncovered game)
        uncovered_studs.sort(key=lambda x: -x[2])
        for p, _, _ in uncovered_studs[:2]:
            if p not in locked:
                locked.append(p)
            if len(locked) >= 5:
                break

        # Heavy contrarian objective: penalise ownership strongly
        obj = {
            p: self._gpp_map.get(p, 0) * (1 - self._own_map.get(p, 50) / 100 * 0.60)
            for p in self._pid_list
        }
        return self._build_ilp_lineup(obj, prev, locked, excl, min_unique=3)

    # ── Generate candidate pool ────────────────────────────────────────────────
    def _generate_candidate_pool(self, n_target: int) -> tuple[list[list[str]], set[int]]:
        """
        Generate n_target * pool_multiplier candidate lineups using all strategies.
        Returns (pool, cheap_stack_indices) so the selector can guarantee those slots.
        """
        n_pool    = n_target * self.pool_mult
        pool      = []
        prev      = []
        games     = self._games if self._games else [None]
        n_games   = len(games)
        exposure  = {}
        cheap_stack_indices: set[int] = set()      # pool indices that are cheap_stack lineups
        stochastic_cs_indices: set[int] = set()    # subset: cheap_stacks built with stochastic mode

        # Strategy mix (10 types, cycling per game):
        #   cheap_stack × 3: 4-cycle cross-game stacks
        #   game_sweep × 1: stud + 3 cheap same-game players (max intra-game correlation)
        #   double_stack × 1: pair top 2 cheap games (3+3)
        #   single_stud_value × 1: top-gpp stud + 7 all ≤$7500
        #   all_value × 1: no player >$8K
        #   ceiling/leverage/fpe: premium lineup builders
        #
        # STABLE SEEDS: each (strategy, game, pass_count) uses a deterministic hash seed
        # so adding/removing strategies does NOT change other strategies' stochastic behavior.
        strategies = ["ceiling", "cheap_stack", "leverage", "double_stack",
                      "fpe", "single_stud_value", "scatter_contrarian", "cheap_stack",
                      "game_sweep", "cheap_stack"]

        # Stable seed: hash(strategy + game + pass_count) so pool_num shifts don't affect seeds
        strat_game_pass: dict[tuple, int] = {}  # (strategy_key, game) -> pass count so far

        def _strat_seed(strategy_key: str, game_key: str, use_stochastic: bool) -> object:
            """Return an RNG seeded deterministically on (strategy_key, game_key, pass_count)."""
            if not use_stochastic:
                return None
            sg = (strategy_key, game_key)
            pc = strat_game_pass.get(sg, 0)
            strat_game_pass[sg] = pc + 1
            raw = hash(f"{strategy_key}:{game_key}:{pc}") & 0x7FFFFFFF
            return np.random.default_rng(raw)

        print(f"[portfolio] Generating {n_pool} candidate lineups ({len(strategies)} strategies x {n_games} games)...")

        attempt = 0
        while len(pool) < n_pool and attempt < n_pool * 6:
            attempt += 1
            pool_num = len(pool)
            n_strat  = len(strategies)
            strategy = strategies[pool_num % n_strat]
            # Standard cycling: pool_num % n_games (preserved for all non-sweep strategies)
            game     = games[pool_num % n_games]

            # Exposure-based exclusions
            max_exp_cnt = max(1, int(n_target * self.max_exposure))
            excl = [p for p, cnt in exposure.items() if cnt >= max_exp_cnt]

            result = None
            is_cheap_stack = False

            if strategy == "cheap_stack":
                if game:
                    # 4-cycle rotation for cheap_stack:
                    #  cycle 0: deterministic top-gpp lock, 1 stud (leaves room for in-game premium players)
                    #  cycle 1: stochastic contrarian lock, 1 stud (explores low-gpp value plays)
                    #  cycle 2: deterministic top-gpp lock, 2 studs (dual-elite combo: Giannis+JJ)
                    #  cycle 3: stochastic contrarian lock, 2 studs (contrarian + dual-elite)
                    cs_pass = strat_game_pass.get(("cheap_stack", game), 0)
                    strat_game_pass[("cheap_stack", game)] = cs_pass + 1
                    cycle = cs_pass % 5
                    # cycle 4 = ultra_cheap (salary-sort) with 2 studs: captures
                    # structures like Jokic+SGA + 3 cheapest MEM@BKN (3/9 winning pattern)
                    use_ultra_cheap = (cycle == 4)
                    use_two_studs   = cycle in (2, 3, 4)
                    use_random_lock = cycle in (1, 3)
                    stack_rng = _strat_seed(f"cheap_stack_c{cycle}", game or "", use_random_lock)

                    other_games = [g for g in self._games if g != game]
                    stud_locked = []
                    if other_games:
                        # Primary stud: cycle through non-stack games
                        og_idx1   = cs_pass % len(other_games)
                        g1_tops   = self._game_tops.get(other_games[og_idx1], [])
                        ai1       = (cs_pass // max(1, len(other_games))) % max(1, len(g1_tops))
                        for cand in g1_tops[ai1:ai1 + 2]:
                            if cand not in excl:
                                stud_locked.append(cand)
                                break
                        # Secondary stud selection (only on 2-stud cycles):
                        if use_two_studs and len(other_games) > 1:
                            if use_ultra_cheap:
                                # Ultra-cheap: pick secondary from the SAME non-stack game as
                                # primary stud (e.g., Jokic+SGA both DEN@OKC).  This creates
                                # max salary savings for the cheap stack while capturing the
                                # same-game correlation of the two elite players.
                                same_game = other_games[og_idx1]
                                same_game_players = [p for p in self._pid_list
                                                     if self._matchup_map.get(p) == same_game
                                                     and p not in excl
                                                     and p not in stud_locked
                                                     and self._dnp_map.get(p, 0.05) <= 0.10]
                                same_game_players.sort(
                                    key=lambda p: -self._gpp_map.get(p, 0))
                                for cand in same_game_players[:3]:
                                    if cand not in stud_locked:
                                        stud_locked.append(cand)
                                        break
                            else:
                                # Normal: secondary from a different non-stack game
                                og_idx2 = (og_idx1 + 1) % len(other_games)
                                g2_tops = self._game_tops.get(other_games[og_idx2], [])
                                for cand in g2_tops[:2]:
                                    if cand not in excl and cand not in stud_locked:
                                        stud_locked.append(cand)
                                        break
                    result = self._cheap_stack_lineup(prev, game, excl, stud_locked,
                                                      rng=stack_rng, ultra_cheap=use_ultra_cheap)
                    if result is not None:
                        is_cheap_stack = True
                        if use_random_lock or use_ultra_cheap:
                            stochastic_cs_indices.add(len(pool))
            elif strategy == "double_stack":
                # Stochastic on odd passes for contrarian variety
                ds_pass = strat_game_pass.get(("double_stack", game), 0)
                strat_game_pass[("double_stack", game)] = ds_pass + 1
                ds_rng = _strat_seed("double_stack", game or "", ds_pass % 2 == 1)
                result = self._double_stack_lineup(prev, excl, rng=ds_rng)
                if result is not None:
                    is_cheap_stack = True   # track as cheap_stack for Phase 1 coverage
                    if ds_rng is not None:
                        stochastic_cs_indices.add(len(pool))
            elif strategy == "single_stud_value":
                # Stochastic on odd passes
                ssv_pass = strat_game_pass.get(("single_stud_value", game), 0)
                strat_game_pass[("single_stud_value", game)] = ssv_pass + 1
                ssv_rng = _strat_seed("single_stud_value", game or "", ssv_pass % 2 == 1)
                result = self._single_stud_value_lineup(prev, game, excl, rng=ssv_rng)
                if result is not None:
                    is_cheap_stack = True   # these are contrarian value structures
                    if ssv_rng is not None:
                        stochastic_cs_indices.add(len(pool))
            elif strategy == "all_value":
                result = self._all_value_lineup(prev, excl)
            elif strategy == "game_sweep":
                # Use group-based cycling to avoid gcd(n_strat, n_games) blind spots.
                # e.g. with 10 strategies + 5 games, gcd=5 so pool_num%n_games would
                # lock game_sweep to 1 game forever.  Instead count total sweep passes
                # across ALL games and cycle through them independently.
                gs_total = sum(strat_game_pass.get(("game_sweep", g), 0) for g in games)
                gs_game  = games[gs_total % n_games]
                gs_pass  = strat_game_pass.get(("game_sweep", gs_game), 0)
                strat_game_pass[("game_sweep", gs_game)] = gs_pass + 1
                gs_rng = _strat_seed("game_sweep", gs_game or "", gs_pass % 2 == 1)
                result = self._game_sweep_lineup(prev, gs_game, excl, rng=gs_rng)
                if result is not None:
                    is_cheap_stack = True
                    if gs_rng is not None:
                        stochastic_cs_indices.add(len(pool))
            elif strategy == "scatter_contrarian":
                # Cross-game contrarian: 1 low-own player from each game + premium studs.
                # Captures multi-game explosion scenarios (MEM@BKN + GSW@UTA both boom).
                sc_total = strat_game_pass.get(("scatter_contrarian", ""), 0)
                strat_game_pass[("scatter_contrarian", "")] = sc_total + 1
                result = self._scatter_contrarian_lineup(prev, excl, sc_pass=sc_total)
                if result is not None:
                    is_cheap_stack = True   # treat as contrarian for Phase 1 coverage
                    stochastic_cs_indices.add(len(pool))
            elif strategy == "elite_team_stack":
                # Stochastic half the time; cycle through games
                ets_rng = np.random.default_rng(pool_num) if (pool_num // n_games) % 2 == 1 else None
                result = self._elite_team_stack_lineup(prev, game, excl, rng=ets_rng)
                if result is not None:
                    is_cheap_stack = True   # these are contrarian value structures
                    if ets_rng is not None:
                        stochastic_cs_indices.add(len(pool))
            else:
                # Locked player from game anchor rotation (for non-cheap_stack)
                locked = []
                game_tops = self._game_tops.get(game, [])
                if game_tops:
                    anchor = game_tops[pool_num // n_games % len(game_tops)]
                    if anchor not in excl:
                        locked = [anchor]

                if strategy == "ceiling":
                    result = self._ceiling_lineup(prev, locked, excl, game)
                elif strategy == "leverage":
                    result = self._leverage_lineup(prev, locked, excl, game)
                elif strategy == "fpe":
                    result = self._fpe_lineup(prev, locked, excl, game)
                elif strategy == "thompson":
                    result = self._thompson_lineup(prev, locked, excl, game)

            if result is None:
                # Relax constraints: thompson with lighter history
                result = self._thompson_lineup(prev[-5:], [], excl, None)

            if result is not None:
                idx = len(pool)
                pool.append(result)
                if is_cheap_stack:
                    cheap_stack_indices.add(idx)
                prev.append(set(result))
                for p in result:
                    exposure[p] = exposure.get(p, 0) + 1

        print(f"[portfolio] Generated {len(pool)} candidates ({len(cheap_stack_indices)} cheap_stack, {len(stochastic_cs_indices)} stochastic)")
        return pool, cheap_stack_indices, stochastic_cs_indices

    # ── Greedy portfolio selection ────────────────────────────────────────────
    def _select_portfolio(
        self,
        candidates: list[list[str]],
        n_target: int,
        cheap_stack_indices: set[int] = None,
        stochastic_cs_indices: set[int] = None,
    ) -> list[list[str]]:
        """
        Greedily select n_target lineups from candidates to maximize portfolio FPE.

        Algorithm:
          1. Score every candidate with fast FPE proxy
          2. Select the highest-FPE lineup as seed
          3. Iteratively add the lineup that maximally increases PORTFOLIO FPE
             (marginal contribution: adding lineup i increases overall FPE the most)
          4. Enforce exposure constraints across selected lineups
        """
        print(f"[portfolio] Selecting best {n_target} from {len(candidates)} candidates...")
        cheap_stack_indices   = cheap_stack_indices or set()
        stochastic_cs_indices = stochastic_cs_indices or set()

        # Score every candidate by FPE proxy: ceiling × explosion × contrarianism.
        # For cheap_stack lineups: use gpp_score instead of raw ceiling, so that
        # contrarian low-gpp plays (Marcus Sasser, Javon Small) aren't penalized
        # against players with high ceilings but higher ownership.
        def lineup_score(lineup: list[str], idx: int) -> float:
            is_cs = idx in cheap_stack_indices
            score = 0.0
            for p in lineup:
                dnp = self._dnp_map.get(p, 0.05)
                if is_cs:
                    # gpp_score already bakes in ownership discount and value-per-dollar
                    score += self._gpp_map.get(p, 0) * (1 - dnp)
                else:
                    dist = self.score_dist.get(p)
                    own  = max(self._own_map.get(p, 15), 1) / 100.0
                    ceil = self._ceil_map.get(p, 0)
                    expl = dist.get("explosion_rate", 0.10) if dist else 0.10
                    score += ceil * (1 + expl * 1.5) * (1 - own * 0.7) * (1 - dnp)
            return score

        indexed = [(lineup_score(lu, i), i, lu) for i, lu in enumerate(candidates)]
        indexed.sort(key=lambda x: -x[0])

        # Greedy selection with exposure enforcement
        selected   = []
        exposure   = {}
        max_exp    = max(1, int(n_target * self.max_exposure))

        # Phase 1: Game-round-robin cheap_stack selection.
        # CRITICAL: winning GPP plays require contrarian game stacks — we must ensure
        # every game gets at least one stochastic cheap_stack lineup (which locks low-
        # ownership contrarian plays like Marcus Sasser, Javon Small, Jordan Miller).
        # Without this, Phase 2 greedily picks higher-FPE chalk lineups and the
        # contrarian cheap_stacks that actually win tournaments never enter portfolio.
        n_games_active = max(1, len(self._games))
        # 1 stochastic cheap_stack per game + fill up to 25% of portfolio total.
        # Round 1 contributes n_games stochastic CS (1 per game, contrarian coverage).
        # Round 2 fills up to min_cheap_stack with best remaining cheap_stacks.
        # We cap at max(n_games, n_target//4) to avoid crowding out high-quality lineups.
        min_cheap_stack = max(n_games_active, n_target // 4)
        cheap_selected  = 0

        def _can_add(lu: list[str]) -> bool:
            if any(exposure.get(p, 0) >= max_exp for p in lu):
                return False
            if selected:
                min_shared = min(len(set(lu) & set(prev)) for prev in selected)
                if min_shared >= ROSTER_SIZE - 2:
                    return False
            return True

        # Group stochastic cheap_stacks by stack game (each game needs representation)
        cs_stochastic_by_game: dict[str, list] = {}
        cs_determ_by_game:     dict[str, list] = {}
        for score, idx, lu in indexed:
            if idx not in cheap_stack_indices:
                continue
            g = self._detect_stack(lu)
            if idx in stochastic_cs_indices:
                cs_stochastic_by_game.setdefault(g, []).append((score, idx, lu))
            else:
                cs_determ_by_game.setdefault(g, []).append((score, idx, lu))

        all_cs_games = list(set(list(cs_stochastic_by_game.keys()) + list(cs_determ_by_game.keys())))
        print(f"[portfolio] Cheap-stack games: {', '.join(all_cs_games)}")
        print(f"[portfolio] Stochastic CS: {sum(len(v) for v in cs_stochastic_by_game.values())} | "
              f"Deterministic CS: {sum(len(v) for v in cs_determ_by_game.values())}")

        # Round 1: 1 stochastic cheap_stack per game — iterate ALL games, no early break.
        # Critical: every game must get a contrarian stochastic CS in the portfolio.
        # Removing early break ensures DEN@OKC, MEM@BKN etc. never get skipped.
        for game in all_cs_games:
            for score, idx, lu in cs_stochastic_by_game.get(game, []):
                if lu in selected:
                    continue
                if _can_add(lu):
                    selected.append(lu)
                    cheap_selected += 1
                    for p in lu:
                        exposure[p] = exposure.get(p, 0) + 1
                    break

        # Round 2: fill remaining cheap_stack slots with best (stochastic first, then determ)
        all_cheap_candidates = [(s, i, lu) for s, i, lu in indexed if i in cheap_stack_indices]
        for score, idx, lu in all_cheap_candidates:
            if cheap_selected >= min_cheap_stack:
                break
            if lu in selected:
                continue
            if _can_add(lu):
                selected.append(lu)
                cheap_selected += 1
                for p in lu:
                    exposure[p] = exposure.get(p, 0) + 1
        print(f"[portfolio] Phase 1 selected {cheap_selected} cheap_stack lineups (target {min_cheap_stack})")

        # Phase 2: Fill remaining slots with highest-scoring lineups.
        for score, idx, lineup in indexed:
            if len(selected) >= n_target:
                break
            if lineup in selected:
                continue
            over = any(exposure.get(p, 0) >= max_exp for p in lineup)
            if over:
                continue
            if selected:
                min_shared = min(len(set(lineup) & set(prev)) for prev in selected)
                if min_shared >= ROSTER_SIZE - 2:
                    continue
            selected.append(lineup)
            for p in lineup:
                exposure[p] = exposure.get(p, 0) + 1

        # Pad if needed (relax diversity to meet n_target)
        if len(selected) < n_target:
            for score, idx, lineup in indexed:
                if len(selected) >= n_target:
                    break
                if lineup not in selected:
                    selected.append(lineup)

        print(f"[portfolio] Selected {len(selected)} lineups")

        # Print exposure summary
        exp_sorted = sorted(exposure.items(), key=lambda x: -x[1])
        print("[portfolio] Top exposures:")
        for pid, cnt in exp_sorted[:8]:
            pct = cnt / len(selected) * 100
            name = self._name_map.get(pid, pid)
            print(f"  {name:<28s} {cnt:2d}/{len(selected)} = {pct:.0f}%")

        return selected

    # ── Main build method ─────────────────────────────────────────────────────
    def build(
        self,
        n: int = 20,
        use_simulation: bool = True,
        n_sim: int = 1000,
    ) -> list[dict]:
        """
        Build the optimal n-lineup portfolio.

        Steps:
          1. Generate candidate pool (n * pool_multiplier candidates)
          2. Select best n by portfolio FPE (greedy marginal contribution)
          3. Run full ContestSimulator on final portfolio
          4. Return lineups with FPE metrics attached

        use_simulation: run full Monte Carlo on final portfolio (slower but more accurate)
        n_sim: simulations per lineup for full evaluation
        """
        # Step 1: Generate candidates
        candidates, cheap_stack_indices, stochastic_cs_indices = self._generate_candidate_pool(n)

        # Step 2: Greedy portfolio selection (guarantees cheap_stack slots with game coverage)
        selected = self._select_portfolio(candidates, n, cheap_stack_indices, stochastic_cs_indices)

        # Step 3: Optional full simulation
        if use_simulation and selected:
            print(f"\n[portfolio] Running full ContestSimulator on {len(selected)} lineups...")
            sim_results = self.simulator.evaluate_portfolio(selected)
            lineup_evals = {r["lineup_num"] - 1: r for r in sim_results["lineup_results"]}
        else:
            lineup_evals = {}

        # Step 4: Format output
        lineups = []
        for i, pids in enumerate(selected):
            rows = self.players[self.players["player_id"].astype(str).isin(pids)].copy()
            names = [self._name_map.get(p, p) for p in pids]
            teams = [self._team_map.get(p, "") for p in pids]
            sals  = [self._sal_map.get(p, 0) for p in pids]
            projs = [round(self._proj_map.get(p, 0), 2) for p in pids]
            owns  = [round(self._own_map.get(p, 15), 1) for p in pids]

            sim_r = lineup_evals.get(i, {})
            lu = {
                "lineup_num":   i + 1,
                "player_ids":   pids,
                "names":        names,
                "teams":        teams,
                "salaries":     sals,
                "projections":  projs,
                "ownerships":   owns,
                "total_salary": sum(sals),
                "proj_pts":     round(sum(projs), 2),
                "ceiling":      round(sum(self._ceil_map.get(p, 0) for p in pids), 2),
                "avg_own":      round(sum(owns) / len(owns), 1),
                "fpe":          sim_r.get("fpe", 0.0),
                "t1pct":        sim_r.get("t1pct", 0.0),
                "p90_pts":      sim_r.get("p90_pts", 0.0),
                "leverage":     sim_r.get("leverage", 0.0),
                # Stack info
                "stack_game":   self._detect_stack(pids),
            }

            # Build slot assignment
            lu["slot_assignment"] = self._assign_slots(pids)
            lineups.append(lu)

        return lineups

    # ── Stack detection ───────────────────────────────────────────────────────
    def _detect_stack(self, pids: list[str]) -> str:
        """Find the game that has the most players in this lineup."""
        game_counts = {}
        for p in pids:
            g = self._matchup_map.get(p, "")
            if g:
                game_counts[g] = game_counts.get(g, 0) + 1
        return max(game_counts, key=game_counts.get) if game_counts else ""

    # ── DK slot assignment ────────────────────────────────────────────────────
    def _assign_slots(self, pids: list[str]) -> dict:
        """Greedy slot assignment for DK upload CSV."""
        assigned = {}
        used = set()
        for slot in _SLOT_ORDER:
            for p in pids:
                if p in used:
                    continue
                slots = self._slot_map.get(p, [])
                if not isinstance(slots, list):
                    slots = []
                if slot in slots:
                    assigned[slot] = p
                    used.add(p)
                    break
        return assigned

    # ── Export DK CSV ─────────────────────────────────────────────────────────
    def export_dk_csv(self, lineups: list[dict], output_path: Path) -> Path:
        """Export lineups in DraftKings upload format."""
        name_id_map = dict(zip(
            self.players["player_id"].astype(str),
            self.players["name_id"].astype(str) if "name_id" in self.players.columns
            else self.players["name"].astype(str),
        ))

        rows = []
        for lu in lineups:
            slot_assign = lu.get("slot_assignment", {})
            row = {slot: name_id_map.get(str(slot_assign.get(slot, "")), "")
                   for slot in _SLOT_ORDER}
            rows.append(row)

        df = pd.DataFrame(rows, columns=_SLOT_ORDER)
        df.to_csv(output_path, index=False)
        print(f"[portfolio] DK upload CSV saved: {output_path}")
        return output_path
