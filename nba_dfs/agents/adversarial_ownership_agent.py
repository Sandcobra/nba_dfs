"""
Adversarial Ownership Distribution Agent.

Models the *structure* of the DFS ownership distribution — not just which
players are owned, but WHERE ownership clusters, where it's thin, and where
building differently from the field is cheapest.

Four core analyses:

  1. Salary bracket curve ($500 increments)
     Where does field ownership peak (chalk clusters) and bottom out (voids)?
     Voids = salary ranges where no player locks >12% of field lineups.

  2. Positional ownership cliffs
     Within each DK position (PG/SG/SF/PF/C), where does the ownership
     drop sharply? Players AFTER the cliff are differentiation territory —
     the field doesn't go there but viable projection exists.

  3. Adversarial salary slot targeting
     Per DK slot, identify the cheapest salary bracket with viable
     projection AND low ownership concentration.  Tells the optimizer
     "pay down HERE — the field won't follow."

  4. Field overlap estimation
     Expected number of shared players between two random field lineups,
     derived from the ownership distribution.  Lower overlap = higher
     variance field = contrarian plays have more top-finish equity.

Outputs a dict consumed by app.py:
  - salary_curve           : list of bracket dicts (salary range, owned, viable)
  - positional_cliffs      : {pos: {cliff_idx, post_cliff_players, cliff_magnitude}}
  - adversarial_slots      : list of {slot, bracket_lo, bracket_hi, reason, savings}
  - field_overlap_estimate : float (expected shared players between two field lineups)
  - gpp_score_boosts       : {player_id: bonus_pts} — players in void zones get a small
                             gpp_score uplift so the ILP routes more lineups through them
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BRACKET_SIZE      = 500          # salary histogram resolution
VOID_MAX_OWN      = 12.0         # bracket "void" if max single-player own < this %
VOID_MIN_VIABLE   = 2            # bracket must have ≥ this many proj >= MIN_VIABLE_PROJ
MIN_VIABLE_PROJ   = 20.0         # minimum DK projection to count as viable
CLIFF_MIN_DROP    = 5.0          # minimum % drop to classify as a real cliff
GPP_VOID_BONUS    = 1.5          # gpp_score pts added to players in void salary zones

# DK slot → eligible primary positions
SLOT_POSITIONS = {
    "PG":   ["PG"],
    "SG":   ["SG"],
    "SF":   ["SF"],
    "PF":   ["PF"],
    "C":    ["C"],
    "G":    ["PG", "SG"],
    "F":    ["SF", "PF"],
    "UTIL": ["PG", "SG", "SF", "PF", "C"],
}


class AdversarialOwnershipAgent:
    """
    Analyses the ownership distribution to find where leverage is cheapest
    and feeds gpp_score boosts back to the optimizer.
    """

    def analyze(
        self,
        players:      pd.DataFrame,
        n_lineups:    int  = 20,
        contest_size: int  = 5000,
    ) -> dict:
        """
        Main entry point.

        Parameters
        ----------
        players      : enriched pool with proj_own, proj_pts_dk, salary, team,
                       primary_position, player_id
        n_lineups    : your planned lineup count
        contest_size : field size (affects overlap estimate interpretation)
        """
        salary_curve   = self._build_salary_curve(players)
        void_brackets  = [b for b in salary_curve if b["is_void"]]
        pos_cliffs     = self._find_positional_cliffs(players)
        adv_slots      = self._adversarial_slot_targets(players, salary_curve, pos_cliffs)
        overlap_est    = self._estimate_field_overlap(players)
        gpp_boosts     = self._compute_gpp_boosts(players, void_brackets, pos_cliffs)
        summary        = self._build_summary(
            salary_curve, void_brackets, pos_cliffs, adv_slots,
            overlap_est, contest_size, n_lineups,
        )

        logger.info(
            "[adv] Salary voids: %d brackets | Positional cliffs: %d positions | "
            "Field overlap est: %.2f players | GPP boosts: %d players",
            len(void_brackets), len(pos_cliffs), overlap_est, len(gpp_boosts),
        )
        return {
            "salary_curve":           salary_curve,
            "void_brackets":          void_brackets,
            "positional_cliffs":      pos_cliffs,
            "adversarial_slots":      adv_slots,
            "field_overlap_estimate": round(overlap_est, 2),
            "gpp_score_boosts":       gpp_boosts,
            "summary":                summary,
        }

    # ── 1. Salary bracket curve ────────────────────────────────────────────────
    def _build_salary_curve(self, df: pd.DataFrame) -> list[dict]:
        """
        Build ownership histogram in $500 salary brackets.
        Returns list sorted by salary_lo ascending.
        """
        sal_min = int(df["salary"].min() // BRACKET_SIZE * BRACKET_SIZE)
        sal_max = int(df["salary"].max() // BRACKET_SIZE * BRACKET_SIZE) + BRACKET_SIZE

        curve = []
        for lo in range(sal_min, sal_max, BRACKET_SIZE):
            hi = lo + BRACKET_SIZE - 1
            bracket = df[(df["salary"] >= lo) & (df["salary"] <= hi)]
            if bracket.empty:
                continue

            viable = bracket[bracket["proj_pts_dk"] >= MIN_VIABLE_PROJ]
            total_own  = float(bracket["proj_own"].sum())
            max_own    = float(bracket["proj_own"].max())
            avg_own    = float(bracket["proj_own"].mean())
            n_players  = len(bracket)
            n_viable   = len(viable)
            top_player = bracket.loc[bracket["proj_own"].idxmax(), "name"] if n_players > 0 else ""

            is_void = (max_own < VOID_MAX_OWN) and (n_viable >= VOID_MIN_VIABLE)
            is_chalk_cluster = max_own >= 25.0

            curve.append({
                "salary_lo":        lo,
                "salary_hi":        hi,
                "label":            f"${lo//1000:.1f}k–${(hi+1)//1000:.1f}k",
                "n_players":        n_players,
                "n_viable":         n_viable,
                "total_own":        round(total_own, 1),
                "max_own":          round(max_own, 1),
                "avg_own":          round(avg_own, 1),
                "top_player":       top_player,
                "is_void":          is_void,
                "is_chalk_cluster": is_chalk_cluster,
            })

        return curve

    # ── 2. Positional ownership cliffs ────────────────────────────────────────
    def _find_positional_cliffs(self, df: pd.DataFrame) -> dict:
        """
        For each primary DK position, rank players by proj_own descending
        and find the biggest ownership drop.

        Returns {pos: {cliff_idx, cliff_magnitude, chalk_players, post_cliff_players}}
          cliff_idx      : 0-based index of first "post-cliff" player
          cliff_magnitude: ownership drop at the cliff (%)
          chalk_players  : players BEFORE the cliff (field consensus)
          post_cliff_players: players AFTER the cliff (differentiation zone)
        """
        cliffs = {}
        for pos in ["PG", "SG", "SF", "PF", "C"]:
            pos_df = df[df["primary_position"] == pos].sort_values(
                "proj_own", ascending=False
            ).reset_index(drop=True)

            if len(pos_df) < 3:
                continue

            ownerships = pos_df["proj_own"].tolist()

            # Find largest consecutive drop
            drops = [
                (ownerships[i] - ownerships[i + 1], i)
                for i in range(len(ownerships) - 1)
            ]
            max_drop, max_idx = max(drops, key=lambda d: d[0])

            if max_drop < CLIFF_MIN_DROP:
                # No meaningful cliff — all players roughly equal ownership
                continue

            cliff_idx = max_idx + 1   # first post-cliff player index

            chalk = pos_df.iloc[:cliff_idx][
                ["name", "team", "salary", "proj_own", "proj_pts_dk"]
            ].to_dict("records")
            post_cliff = pos_df.iloc[cliff_idx:cliff_idx + 5][
                ["name", "team", "salary", "proj_own", "proj_pts_dk"]
            ].to_dict("records")

            chalk_names = ", ".join(p["name"] for p in chalk)
            chalk_owns  = ", ".join(str(round(p["proj_own"])) + "%" for p in chalk)
            diff_names  = ", ".join(p["name"] for p in post_cliff[:3])
            cliffs[pos] = {
                "cliff_idx":          cliff_idx,
                "cliff_magnitude":    round(max_drop, 1),
                "chalk_players":      chalk,
                "post_cliff_players": post_cliff,
                "interpretation": (
                    f"{pos}: field locks {chalk_names} ({chalk_owns}). "
                    f"Cliff of {max_drop:.1f}pts. "
                    f"Post-cliff differentiators: {diff_names}."
                ),
            }

        return cliffs

    # ── 3. Adversarial slot targets ────────────────────────────────────────────
    def _adversarial_slot_targets(
        self,
        df: pd.DataFrame,
        salary_curve: list[dict],
        pos_cliffs: dict,
    ) -> list[dict]:
        """
        For each DK slot, identify the salary bracket that offers:
          - Viable projection (at least one player >= MIN_VIABLE_PROJ)
          - Low ownership concentration (void or near-void)
          - Maximum salary savings vs the field's chalk allocation

        Returns list of adversarial slot recommendations sorted by savings descending.
        """
        # Estimate field's "default" salary per slot from chalk players
        chalk_salary_by_pos: dict[str, float] = {}
        for pos, cliff_data in pos_cliffs.items():
            chalk_list = cliff_data.get("chalk_players", [])
            if chalk_list:
                chalk_salary_by_pos[pos] = np.mean([p["salary"] for p in chalk_list])

        targets = []
        for slot, eligible_positions in SLOT_POSITIONS.items():
            slot_df = df[df["primary_position"].isin(eligible_positions)].copy()
            if slot_df.empty:
                continue

            # Field's typical salary spend for this slot (from chalk anchors)
            field_sal_for_slot = np.mean([
                chalk_salary_by_pos.get(p, slot_df["salary"].median())
                for p in eligible_positions
            ])

            # Find best void bracket for this slot
            best_void = None
            best_savings = 0

            for bracket in salary_curve:
                if not bracket["is_void"] or bracket["n_viable"] < 1:
                    continue

                lo, hi = bracket["salary_lo"], bracket["salary_hi"]
                bracket_slot_df = slot_df[
                    (slot_df["salary"] >= lo) & (slot_df["salary"] <= hi) &
                    (slot_df["proj_pts_dk"] >= MIN_VIABLE_PROJ)
                ]
                if bracket_slot_df.empty:
                    continue

                # Best player in this void bracket for this slot
                best_player = bracket_slot_df.nlargest(1, "proj_pts_dk").iloc[0]
                savings = field_sal_for_slot - best_player["salary"]

                if savings > best_savings:
                    best_savings = savings
                    best_void = {
                        "slot":        slot,
                        "player":      best_player.get("name", ""),
                        "team":        best_player.get("team", ""),
                        "salary":      int(best_player["salary"]),
                        "proj_pts_dk": round(float(best_player["proj_pts_dk"]), 1),
                        "proj_own":    round(float(best_player["proj_own"]), 1),
                        "bracket":     bracket["label"],
                        "bracket_max_own": bracket["max_own"],
                        "field_avg_sal": int(field_sal_for_slot),
                        "savings":     int(savings),
                        "reason": (
                            f"{slot}: {best_player.get('name','')} "
                            f"(${int(best_player['salary']):,}, "
                            f"{best_player['proj_pts_dk']:.0f}pts, "
                            f"{best_player['proj_own']:.0f}%own) — "
                            f"saves ${int(savings):,} vs field avg ${int(field_sal_for_slot):,} "
                            f"in a void bracket (max {bracket['max_own']:.0f}% own)."
                        ),
                    }

            if best_void and best_savings >= 500:
                targets.append(best_void)

        targets.sort(key=lambda t: t["savings"], reverse=True)
        return targets

    # ── 4. Field overlap estimation ────────────────────────────────────────────
    def _estimate_field_overlap(self, df: pd.DataFrame) -> float:
        """
        Estimate expected number of shared players between two random field lineups.

        Formula: E[overlap] = Σ p_i²  × ROSTER_SIZE / ROSTER_SIZE
                            = Σ (proj_own_i / 100)²  × 8

        Interpretation:
          ≥ 4.0  : very chalk-heavy field — top finishes need big differentiation
          3.0–4.0: moderately chalky — 1-2 pivots needed
          < 3.0  : spread field — many viable constructions, GPP variance is natural
        """
        owns = df["proj_own"].clip(0, 100) / 100.0
        return float((owns ** 2).sum() * 8)

    # ── 5. GPP score boosts for void zones ────────────────────────────────────
    def _compute_gpp_boosts(
        self,
        df: pd.DataFrame,
        void_brackets: list[dict],
        pos_cliffs: dict,
    ) -> dict:
        """
        Compute per-player gpp_score bonus for:
          A. Players in ownership void salary brackets (underpriced leverage)
          B. Players in post-cliff zone for their position (differentiation plays)

        Returns {player_id_str: bonus_pts}.
        These are applied in app.py before lineup generation.
        """
        boosts: dict[str, float] = {}

        # A. Void bracket bonus
        void_ranges = [(b["salary_lo"], b["salary_hi"]) for b in void_brackets]
        for _, row in df.iterrows():
            sal = int(row.get("salary", 0))
            pid = str(row.get("player_id", ""))
            if not pid:
                continue
            in_void = any(lo <= sal <= hi for lo, hi in void_ranges)
            if in_void:
                boosts[pid] = boosts.get(pid, 0.0) + GPP_VOID_BONUS

        # B. Post-cliff positional bonus
        for pos, cliff_data in pos_cliffs.items():
            post_cliff = cliff_data.get("post_cliff_players", [])
            cliff_mag  = cliff_data.get("cliff_magnitude", 0)
            # Scale bonus by cliff magnitude: bigger cliff = more exclusive differentiation
            pos_bonus = round(min(GPP_VOID_BONUS, cliff_mag / 15.0), 2)
            for pc_player in post_cliff:
                pc_name = pc_player.get("name", "")
                pid_rows = df[df["name"] == pc_name]["player_id"]
                if not pid_rows.empty:
                    pid = str(pid_rows.iloc[0])
                    boosts[pid] = round(boosts.get(pid, 0.0) + pos_bonus, 2)

        return boosts

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    def _build_summary(
        self,
        salary_curve:   list[dict],
        void_brackets:  list[dict],
        pos_cliffs:     dict,
        adv_slots:      list[dict],
        overlap_est:    float,
        contest_size:   int,
        n_lineups:      int,
    ) -> dict:
        """Build human-readable summary of adversarial findings."""

        # Overlap interpretation
        if overlap_est >= 4.0:
            overlap_label = "very chalk-heavy"
            overlap_advice = "Field is highly concentrated — bold pivots required to differentiate."
        elif overlap_est >= 3.0:
            overlap_label = "moderately chalky"
            overlap_advice = "1-2 pivot plays per lineup will separate you from 80% of the field."
        else:
            overlap_label = "spread / diverse"
            overlap_advice = "Variance is natural — focus on projection quality over extreme fades."

        chalk_clusters = [b for b in salary_curve if b["is_chalk_cluster"]]
        total_savings  = sum(t["savings"] for t in adv_slots)

        return {
            "overlap_estimate":    round(overlap_est, 2),
            "overlap_label":       overlap_label,
            "overlap_advice":      overlap_advice,
            "void_count":          len(void_brackets),
            "chalk_cluster_count": len(chalk_clusters),
            "cliff_positions":     list(pos_cliffs.keys()),
            "total_adv_savings":   total_savings,
            "headline": (
                f"Field overlap ~{overlap_est:.1f} players ({overlap_label}). "
                f"{len(void_brackets)} ownership void brackets. "
                f"Adversarial salary plan saves ${total_savings:,} vs chalk allocation."
            ),
        }
