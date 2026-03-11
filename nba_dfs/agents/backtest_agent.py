"""
Backtest Agent — post-slate analysis and model improvement engine.

Given DraftKings contest files (slate CSV, entries CSV, contest results CSV),
this agent produces a structured report covering:

  1. ENTRY PERFORMANCE   — how each submitted lineup scored vs the full field
  2. INJURY RECONSTRUCTION — which players likely DNP'd based on 0/missing FPTS
  3. ON/OFF ACCURACY       — predicted delta_dk vs actual FPTS delta for injured
                             players' teammates
  4. OWNERSHIP CALIBRATION — proj_own vs actual %Drafted
  5. CONSTRUCTION PATTERNS — what the top 1% of the field built vs our lineups
  6. RECOMMENDATIONS       — actionable parameter adjustments

Usage:
  from nba_dfs.agents.backtest_agent import BacktestAgent
  agent = BacktestAgent(cache_dir=Path("cache/espn"))
  report = agent.run("contest/dk_slate_3_8.csv",
                     "contest/dk_entries_3_8.csv",
                     "contest/contest-results_3_8.csv")
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── DK scoring weights (same as test_slate.py) ────────────────────────────────
DK_PTS   = 1.0
DK_3PM   = 0.5
DK_REB   = 1.25
DK_AST   = 1.5
DK_STL   = 2.0
DK_BLK   = 2.0
DK_TOV   = -0.5
DK_DD    = 1.5
DK_TD    = 3.0

# Minimum season-avg DK pts to be considered an "active contributor" (not garbage time)
MIN_AVG_PTS = 8.0

# A player is considered likely DNP if their actual FPTS is 0 or missing
DNP_FPTS_THRESHOLD = 0.0


class BacktestAgent:
    """
    Loads contest files and produces a comprehensive backtest report.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir or Path("cache/espn")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        slate_path: str | Path,
        entries_path: str | Path,
        results_path: str | Path,
        slate_date: str = "",
    ) -> dict:
        """
        Run full backtest for one slate.

        Returns
        -------
        dict with keys:
          slate_date, entry_scores, field_stats, cash_summary,
          injury_reconstruction, on_off_accuracy, ownership_accuracy,
          construction_patterns, recommendations
        """
        slate   = self._load_slate(Path(slate_path))
        entries = self._load_entries(Path(entries_path))
        results = self._load_results(Path(results_path))

        # ── Build player FPTS lookup ──────────────────────────────────────────
        fpts_map   = self._build_fpts_map(results)
        actual_own = self._build_ownership_map(results)

        # ── Score our entries ─────────────────────────────────────────────────
        entry_scores = self._score_entries(entries, fpts_map)

        # ── Field stats ───────────────────────────────────────────────────────
        field_stats  = self._field_stats(results)

        # ── Cash summary ──────────────────────────────────────────────────────
        cash_summary = self._cash_summary(entry_scores, field_stats)

        # ── Reconstruct injuries ──────────────────────────────────────────────
        injuries = self._reconstruct_injuries(slate, fpts_map, actual_own)

        # ── On/off accuracy ───────────────────────────────────────────────────
        on_off_acc = self._on_off_accuracy(injuries, slate, fpts_map, slate_date)

        # ── Ownership calibration ─────────────────────────────────────────────
        own_accuracy = self._ownership_accuracy(slate, actual_own)

        # ── Construction patterns ─────────────────────────────────────────────
        construction = self._construction_patterns(results, slate, fpts_map, field_stats)

        # ── Winner deconstruction ─────────────────────────────────────────────
        winner_analysis = self._winner_deconstruction(results, slate, fpts_map, actual_own, field_stats)

        # ── Chalk trap analysis ───────────────────────────────────────────────
        chalk_traps = self._chalk_trap_analysis(slate, fpts_map, actual_own)

        # ── Leverage play finder ──────────────────────────────────────────────
        leverage_plays = self._leverage_play_finder(slate, fpts_map, actual_own)

        # ── Our lineup autopsy ────────────────────────────────────────────────
        autopsy = self._lineup_autopsy(entries, fpts_map, actual_own, slate, field_stats)

        # ── Recommendations ───────────────────────────────────────────────────
        recs = self._build_recommendations(
            cash_summary, on_off_acc, own_accuracy, construction,
            chalk_traps, leverage_plays, autopsy,
        )

        return {
            "slate_date":            slate_date,
            "entry_scores":          entry_scores,
            "field_stats":           field_stats,
            "cash_summary":          cash_summary,
            "injury_reconstruction": injuries,
            "on_off_accuracy":       on_off_acc,
            "ownership_accuracy":    own_accuracy,
            "construction_patterns": construction,
            "winner_analysis":       winner_analysis,
            "chalk_traps":           chalk_traps,
            "leverage_plays":        leverage_plays,
            "lineup_autopsy":        autopsy,
            "recommendations":       recs,
        }

    def run_multi(self, slates: list[dict]) -> dict:
        """
        Run backtest across multiple slates and aggregate findings.

        Each item in slates: {"slate": path, "entries": path, "results": path, "date": str}
        """
        daily_reports = []
        for s in slates:
            try:
                r = self.run(s["slate"], s["entries"], s["results"], s.get("date", ""))
                daily_reports.append(r)
                logger.info("[backtest] %s complete — cashed %d/%d",
                            s.get("date","?"),
                            r["cash_summary"]["entries_cashed"],
                            r["cash_summary"]["total_entries"])
            except Exception as exc:
                logger.warning("[backtest] %s failed: %s", s.get("date","?"), exc)

        return self._aggregate_reports(daily_reports)

    # ── Loaders ────────────────────────────────────────────────────────────────

    def _load_slate(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df["Name"]   = df["Name"].str.strip()
        df["ID"]     = df["ID"].astype(str).str.strip()
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int)
        df["AvgPointsPerGame"] = pd.to_numeric(df["AvgPointsPerGame"], errors="coerce").fillna(0)
        return df

    def _load_entries(self, path: Path) -> list[list[str]]:
        """
        Parse entries CSV — handles two DK formats:
          Simple:  PG,SG,SF,PF,C,G,F,UTIL
          Bulk:    Entry ID,Contest Name,Contest ID,Entry Fee,PG,SG,...
        Returns list of lineups, each lineup = list of 8 player names.
        """
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip() for c in df.columns]

        # Detect format by checking if first column is 'PG' (simple) or 'Entry ID' (bulk)
        slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        if df.columns[0].upper() in ("PG", "ENTRY ID", "ENTRYID"):
            # Find which columns are the lineup slots
            slot_cols = [c for c in df.columns if c.upper() in slots]
            if len(slot_cols) < 8:
                # fallback: first 8 cols that contain player-name-like content
                slot_cols = df.columns[:8].tolist()
        else:
            slot_cols = df.columns[:8].tolist()

        lineups = []
        for _, row in df.iterrows():
            names = []
            for col in slot_cols:
                cell = str(row.get(col, "")).strip()
                if not cell or cell.lower() == "nan":
                    continue
                name = self._extract_name(cell)
                if name:
                    names.append(name)
            if len(names) >= 7:  # allow 7+ in case of parse edge cases
                lineups.append(names)

        logger.info("[backtest] Loaded %d lineups from %s", len(lineups), path.name)
        return lineups

    def _load_results(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        # Remove BOM if present
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        df["Points"] = pd.to_numeric(df.get("Points", pd.Series(dtype=float)), errors="coerce")
        if "FPTS" in df.columns:
            df["FPTS"] = pd.to_numeric(df["FPTS"], errors="coerce")
        if "%Drafted" in df.columns:
            df["%Drafted"] = (
                df["%Drafted"].astype(str)
                .str.replace("%", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
        return df

    # ── FPTS / Ownership maps ──────────────────────────────────────────────────

    def _build_fpts_map(self, results: pd.DataFrame) -> dict[str, float]:
        """{ player_name: actual_fpts }"""
        if "Player" not in results.columns or "FPTS" not in results.columns:
            return {}
        sub = results[["Player", "FPTS"]].dropna(subset=["Player", "FPTS"])
        sub = sub.drop_duplicates("Player")
        return {str(r["Player"]).strip(): float(r["FPTS"]) for _, r in sub.iterrows()}

    def _build_ownership_map(self, results: pd.DataFrame) -> dict[str, float]:
        """{ player_name: actual_ownership_pct }"""
        if "Player" not in results.columns or "%Drafted" not in results.columns:
            return {}
        sub = results[["Player", "%Drafted"]].dropna(subset=["Player", "%Drafted"])
        sub = sub.drop_duplicates("Player")
        return {str(r["Player"]).strip(): float(r["%Drafted"]) for _, r in sub.iterrows()}

    # ── Entry scoring ──────────────────────────────────────────────────────────

    def _score_entries(
        self, lineups: list[list[str]], fpts_map: dict[str, float]
    ) -> list[dict]:
        scored = []
        for i, names in enumerate(lineups, 1):
            pts_list = [(n, fpts_map.get(n, 0.0)) for n in names]
            total    = round(sum(p for _, p in pts_list), 2)
            scored.append({
                "lineup_num": i,
                "score":      total,
                "players":    names,
                "player_scores": {n: p for n, p in pts_list},
            })
        return sorted(scored, key=lambda x: x["score"], reverse=True)

    # ── Field stats ────────────────────────────────────────────────────────────

    def _field_stats(self, results: pd.DataFrame) -> dict:
        scores = results["Points"].dropna().sort_values(ascending=False)
        total  = len(scores)
        return {
            "total_entries":  total,
            "winning_score":  round(float(scores.iloc[0]), 2) if total else 0,
            "p99":  round(float(scores.quantile(0.99)), 2),
            "p90":  round(float(scores.quantile(0.90)), 2),
            "p80":  round(float(scores.quantile(0.80)), 2),
            "p50":  round(float(scores.quantile(0.50)), 2),  # cash line
            "p25":  round(float(scores.quantile(0.25)), 2),
            "mean": round(float(scores.mean()), 2),
        }

    # ── Cash summary ───────────────────────────────────────────────────────────

    def _cash_summary(self, entry_scores: list[dict], field_stats: dict) -> dict:
        cash_line = field_stats["p50"]
        top20_line = field_stats["p80"]
        top10_line = field_stats["p90"]

        scores = [e["score"] for e in entry_scores]
        cashed    = sum(1 for s in scores if s >= cash_line)
        top20     = sum(1 for s in scores if s >= top20_line)
        top10     = sum(1 for s in scores if s >= top10_line)
        best      = max(scores) if scores else 0
        worst     = min(scores) if scores else 0
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0

        return {
            "total_entries":  len(scores),
            "avg_score":      avg_score,
            "best_score":     round(best, 2),
            "worst_score":    round(worst, 2),
            "entries_cashed": cashed,
            "cash_rate":      round(cashed / len(scores), 3) if scores else 0,
            "entries_top20":  top20,
            "entries_top10":  top10,
            "cash_line":      cash_line,
            "top20_line":     top20_line,
            "gap_to_cash":    round(cash_line - avg_score, 2),
        }

    # ── Injury reconstruction ──────────────────────────────────────────────────

    def _reconstruct_injuries(
        self,
        slate: pd.DataFrame,
        fpts_map: dict[str, float],
        actual_own: dict[str, float],
    ) -> list[dict]:
        """
        Players in the slate who scored 0 FPTS or don't appear in the ownership
        table at all are likely DNPs. Return a list of reconstructed injuries
        with impact metadata.
        """
        injuries = []
        for _, row in slate.iterrows():
            name  = str(row["Name"]).strip()
            avg   = float(row["AvgPointsPerGame"])
            sal   = int(row["Salary"])

            if avg < MIN_AVG_PTS or sal < 6000:
                continue  # ignore low-salary / garbage-time players

            actual_fpts = fpts_map.get(name)
            ownership   = actual_own.get(name, 0.0)

            # Likely DNP: missing from results OR scored exactly 0 AND low ownership
            is_dnp = (actual_fpts is None) or (actual_fpts <= DNP_FPTS_THRESHOLD and ownership < 1.0)

            if is_dnp:
                team = str(row.get("TeamAbbrev", "")).strip()
                pos  = str(row.get("Roster Position", row.get("Position", ""))).strip()
                injuries.append({
                    "name":      name,
                    "team":      team,
                    "position":  pos,
                    "salary":    sal,
                    "avg_pts":   avg,
                    "actual_fpts": 0.0 if actual_fpts is None else actual_fpts,
                    "ownership": ownership,
                })

        logger.info("[backtest] Reconstructed %d likely DNPs", len(injuries))
        return injuries

    # ── On/off accuracy ────────────────────────────────────────────────────────

    def _on_off_accuracy(
        self,
        injuries: list[dict],
        slate: pd.DataFrame,
        fpts_map: dict[str, float],
        slate_date: str,
    ) -> list[dict]:
        """
        For each injured player, find their teammates in the slate and measure:
          actual_delta = teammate actual FPTS - teammate season avg
          predicted_delta = what our on/off cache said (if available)
          error = predicted_delta - actual_delta
        """
        # Normalise date for game-log lookup  (e.g. "3_8" -> "2026-03-08")
        norm_date = self._normalise_date(slate_date)

        # Load all cached game logs once
        gamelog_cache = self._load_gamelog_cache()

        results = []
        for inj in injuries:
            team      = inj["team"]
            inj_name  = inj["name"]
            inj_sal   = inj["salary"]

            # Find teammates from the same team in slate
            teammates = slate[slate["TeamAbbrev"] == team]
            teammates = teammates[teammates["Name"] != inj_name]

            if teammates.empty:
                continue

            # Try to load our cached on/off prediction for this injury
            oo_predicted = self._load_onoff_prediction(inj_name, teammates["Name"].tolist())

            teammate_impacts = []
            for _, tm in teammates.iterrows():
                tm_name = str(tm["Name"]).strip()
                tm_avg  = float(tm["AvgPointsPerGame"])
                tm_sal  = int(tm["Salary"])

                if tm_avg < MIN_AVG_PTS:
                    continue

                actual_fpts   = fpts_map.get(tm_name)
                if actual_fpts is None:
                    continue  # teammate also didn't play

                actual_delta  = round(actual_fpts - tm_avg, 2)

                # Get predicted delta from on/off cache
                pred_delta = oo_predicted.get(tm_name.lower(), None)

                # Also check game log for this specific date
                gl_delta = self._gamelog_delta_on_date(
                    tm_name, norm_date, gamelog_cache, tm_avg
                )

                teammate_impacts.append({
                    "teammate":        tm_name,
                    "salary":          tm_sal,
                    "avg_pts":         tm_avg,
                    "actual_fpts":     round(actual_fpts, 2),
                    "actual_delta":    actual_delta,
                    "predicted_delta": round(pred_delta, 2) if pred_delta is not None else None,
                    "prediction_error": round(pred_delta - actual_delta, 2)
                                        if pred_delta is not None else None,
                    "gamelog_delta":   gl_delta,
                })

            if teammate_impacts:
                # Sort by actual delta descending — positive gainers first
                teammate_impacts.sort(key=lambda x: x["actual_delta"], reverse=True)
                biggest = teammate_impacts[0]
                results.append({
                    "injured_player": inj_name,
                    "team":           team,
                    "salary":         inj_sal,
                    "avg_pts":        inj["avg_pts"],
                    "teammates":      teammate_impacts,
                    "biggest_gainer": biggest["teammate"],
                    "biggest_gain":   biggest["actual_delta"],
                    "predicted_gain": biggest["predicted_delta"],
                })

        results.sort(key=lambda x: abs(x["biggest_gain"]), reverse=True)
        return results

    def _load_onoff_prediction(self, injured_name: str, teammate_names: list[str]) -> dict:
        """
        Load cached on/off deltas from the cache/espn/ directory.
        Returns {teammate_name_lower: delta_dk}.
        """
        if not self._cache_dir.exists():
            return {}

        inj_lower = injured_name.lower().replace(" ", "_")

        # Scan on_off cache files for ones that include this injured player
        for p in self._cache_dir.glob("on_off_*.json"):
            stem = p.stem  # e.g. on_off_jarred_vanderbilt_2026
            if inj_lower.replace(" ", "_") in stem:
                try:
                    data = json.loads(p.read_text())
                    # data is {player_id: {delta_dk, dk_with, dk_without, ...}}
                    # We need to map player_id -> name somehow
                    # For now return the raw delta map keyed by whatever is available
                    return {str(k).lower(): float(v.get("delta_dk", 0))
                            for k, v in data.items() if isinstance(v, dict)}
                except Exception:
                    pass

        return {}

    def _load_gamelog_cache(self) -> dict[str, pd.DataFrame]:
        """Load all cached game logs into memory. Returns {espn_id: DataFrame}."""
        cache: dict[str, pd.DataFrame] = {}
        if not self._cache_dir.exists():
            return cache

        for p in self._cache_dir.glob("gamelog_*_2026.json"):
            try:
                espn_id = p.stem.split("_")[1]
                df = pd.read_json(io.StringIO(p.read_text()))
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                cache[espn_id] = df
            except Exception:
                pass

        return cache

    def _gamelog_delta_on_date(
        self,
        player_name: str,
        date: str,
        gamelog_cache: dict[str, pd.DataFrame],
        season_avg: float,
    ) -> Optional[float]:
        """
        Find a player's DK pts on a specific date from cached game logs and
        compute delta vs season average.  Returns None if not found.
        """
        if not date:
            return None

        # We don't have a name->espn_id map in cache, so we match via roster files
        espn_id = self._resolve_espn_id(player_name)
        if espn_id is None:
            return None

        gl = gamelog_cache.get(espn_id)
        if gl is None or gl.empty:
            return None

        row = gl[gl["date"] == date]
        if row.empty:
            return None

        r = row.iloc[0]
        mp  = float(r.get("mp",  0))
        if mp < 5:
            return None  # garbage minutes

        pts = float(r.get("pts", 0))
        reb = float(r.get("trb", 0))
        ast = float(r.get("ast", 0))
        stl = float(r.get("stl", 0))
        blk = float(r.get("blk", 0))
        tov = float(r.get("tov", 0))
        fg3 = float(r.get("fg3", 0))

        # Approximate DD/TD
        stats = [pts >= 10, reb >= 10, ast >= 10, stl >= 10, blk >= 10]
        dd_bonus = DK_DD if sum(stats) >= 2 else 0
        td_bonus = DK_TD if sum(stats) >= 3 else 0

        dk_pts = (pts * DK_PTS + fg3 * DK_3PM + reb * DK_REB +
                  ast * DK_AST + stl * DK_STL + blk * DK_BLK +
                  tov * DK_TOV + dd_bonus + td_bonus)

        return round(dk_pts - season_avg, 2)

    def _resolve_espn_id(self, player_name: str) -> Optional[str]:
        """Look up ESPN athlete ID from roster cache files."""
        name_lower = player_name.lower().strip()
        for p in self._cache_dir.glob("roster_*_2026.json"):
            try:
                data = json.loads(p.read_text())
                # data is {name_lower: espn_id}
                if name_lower in data:
                    return str(data[name_lower])
                # fuzzy: last name
                last = name_lower.split()[-1] if name_lower else ""
                for k, v in data.items():
                    if k.split()[-1] == last:
                        return str(v)
            except Exception:
                pass
        return None

    # ── Ownership calibration ──────────────────────────────────────────────────

    def _ownership_accuracy(
        self,
        slate: pd.DataFrame,
        actual_own: dict[str, float],
    ) -> dict:
        """
        Compare our model's projected ownership (estimated from salary rank)
        to actual %Drafted from contest results.
        """
        if not actual_own:
            return {"error": "No ownership data available"}

        rows = []
        for _, row in slate.iterrows():
            name   = str(row["Name"]).strip()
            sal    = float(row["Salary"])
            avg    = float(row["AvgPointsPerGame"])
            actual = actual_own.get(name)
            if actual is None:
                continue

            # Re-compute our projected ownership model
            sal_rank  = sal / slate["Salary"].max()
            proj_rank = avg / slate["AvgPointsPerGame"].max()
            proj_own  = float(np.clip(sal_rank**2 * 30 + proj_rank * 15 + 2, 1, 55))

            rows.append({
                "name":      name,
                "salary":    int(sal),
                "proj_own":  round(proj_own, 1),
                "actual_own": round(actual, 1),
                "error":     round(proj_own - actual, 1),
            })

        if not rows:
            return {"error": "Could not match slate players to ownership data"}

        df = pd.DataFrame(rows)
        mae  = round(float(df["error"].abs().mean()), 2)
        bias = round(float(df["error"].mean()), 2)     # positive = we over-project ownership
        corr = round(float(df["proj_own"].corr(df["actual_own"])), 3)

        # Most overestimated (chalk we expected that wasn't)
        most_over  = df.nlargest(5, "error")[["name","salary","proj_own","actual_own","error"]].to_dict("records")
        # Most underestimated (low-owned blowups we didn't target)
        most_under = df.nsmallest(5, "error")[["name","salary","proj_own","actual_own","error"]].to_dict("records")

        return {
            "mae":          mae,
            "bias":         bias,
            "correlation":  corr,
            "most_overestimated":  most_over,
            "most_underestimated": most_under,
            "note": ("bias>0 means model over-projects ownership → "
                     "lineups are less contrarian than intended") if bias > 2 else
                    ("bias<0 means model under-projects ownership → "
                     "lineups are more contrarian than field recognises"),
        }

    # ── Construction patterns ──────────────────────────────────────────────────

    def _construction_patterns(
        self,
        results: pd.DataFrame,
        slate: pd.DataFrame,
        fpts_map: dict[str, float],
        field_stats: dict,
    ) -> dict:
        """
        Analyse top 10% and top 1% lineups from the contest to extract:
          - Average salary used
          - Stack patterns (2/3/4-man stacks from same game)
          - Ownership distribution
          - Which players appeared most in winning lineups
        """
        top10_line = field_stats["p90"]
        top1_line  = field_stats["p99"]

        # Parse lineup strings from contest results
        # Format: "C Jay Huff F Desmond Bane G ... PG ... UTIL ..."
        lineup_rows = results[results["Points"].notna()].copy()

        # Build team lookup from slate
        name_team = dict(zip(slate["Name"].str.strip(), slate["TeamAbbrev"].str.strip()))
        name_sal  = dict(zip(slate["Name"].str.strip(), slate["Salary"]))

        parse_lineup = self._parse_lineup_str

        top10_lineups = []
        top1_lineups  = []

        for _, row in lineup_rows.iterrows():
            score   = float(row["Points"])
            lineup  = parse_lineup(str(row.get("Lineup", "")))
            if len(lineup) < 7:
                continue
            if score >= top10_line:
                top10_lineups.append((score, lineup))
            if score >= top1_line:
                top1_lineups.append((score, lineup))

        def analyse_group(lineups: list[tuple]) -> dict:
            if not lineups:
                return {}

            # Player frequency in winning lineups
            from collections import Counter
            player_freq = Counter(p for _, lineup in lineups for p in lineup)
            top_players = [{"name": n, "appearances": c,
                            "pct": round(c / len(lineups) * 100, 1)}
                           for n, c in player_freq.most_common(10)]

            # Average salary
            salaries = []
            for _, lineup in lineups:
                sal = sum(name_sal.get(p, 0) for p in lineup)
                salaries.append(sal)
            avg_salary = round(sum(salaries) / len(salaries), 0) if salaries else 0

            # Stack analysis: count how many players from same team per lineup
            stack_sizes = []
            for _, lineup in lineups:
                from collections import Counter as C2
                teams = [name_team.get(p, "UNK") for p in lineup]
                team_counts = C2(t for t in teams if t != "UNK")
                max_stack = max(team_counts.values()) if team_counts else 1
                stack_sizes.append(max_stack)

            stack_dist = {}
            for s in range(2, 6):
                stack_dist[f"{s}-man"] = sum(1 for x in stack_sizes if x == s)

            return {
                "sample_size":  len(lineups),
                "avg_salary":   int(avg_salary),
                "top_players":  top_players,
                "stack_dist":   stack_dist,
                "avg_stack":    round(sum(stack_sizes) / len(stack_sizes), 2) if stack_sizes else 0,
            }

        return {
            "top_10pct": analyse_group(top10_lineups),
            "top_1pct":  analyse_group(top1_lineups),
            "winning_lineup": {
                "score":   field_stats["winning_score"],
                "players": top1_lineups[0][1] if top1_lineups else [],
            },
        }

    # ── Winner deconstruction ──────────────────────────────────────────────────

    def _winner_deconstruction(
        self,
        results: pd.DataFrame,
        slate: pd.DataFrame,
        fpts_map: dict,
        actual_own: dict,
        field_stats: dict,
    ) -> dict:
        """
        Dissect the top 10 lineups: who were the key leverage plays, what was
        the avg lineup ownership, and what salary construction did winners use.
        """
        avg_map  = dict(zip(slate["Name"].str.strip(), slate["AvgPointsPerGame"]))
        sal_map  = dict(zip(slate["Name"].str.strip(), slate["Salary"]))
        top10_line = field_stats["p90"]

        lineup_rows = results[results["Points"].notna() & results["Lineup"].notna()].copy()
        top_lineups = lineup_rows.nlargest(10, "Points")

        deconstructed = []
        for _, row in top_lineups.iterrows():
            score   = float(row["Points"])
            players = self._parse_lineup_str(str(row.get("Lineup", "")))
            if len(players) < 7:
                continue

            player_detail = []
            for p in players:
                fpts  = fpts_map.get(p, 0)
                own   = actual_own.get(p, 0)
                avg   = avg_map.get(p, 0)
                sal   = sal_map.get(p, 0)
                multiplier = round(fpts / avg, 2) if avg > 0 else None
                player_detail.append({
                    "name": p, "fpts": fpts, "ownership": own,
                    "avg": avg, "salary": sal, "multiplier": multiplier,
                })

            # Sort by leverage value: FPTS / max(ownership, 0.1)
            player_detail.sort(key=lambda x: x["fpts"] / max(x["ownership"], 0.1), reverse=True)
            avg_own = round(sum(p["ownership"] for p in player_detail) / len(player_detail), 1)
            total_sal = sum(p["salary"] for p in player_detail)

            # The "key play" = best FPTS/ownership ratio with meaningful FPTS (>20)
            key_plays = [p for p in player_detail if p["fpts"] > 20][:3]

            deconstructed.append({
                "score": score,
                "avg_ownership": avg_own,
                "total_salary": total_sal,
                "key_plays": key_plays,
                "all_players": player_detail,
            })

        # Aggregate: which players appeared in winning lineups most?
        from collections import Counter
        all_winning_players = [
            p["name"] for lu in deconstructed for p in lu["all_players"]
        ]
        freq = Counter(all_winning_players)
        core_plays = [
            {"name": n, "appearances": c,
             "fpts": fpts_map.get(n, 0),
             "ownership": actual_own.get(n, 0),
             "pct_of_winners": round(c / len(deconstructed) * 100, 0) if deconstructed else 0}
            for n, c in freq.most_common(10)
        ]

        avg_winner_own = round(
            sum(lu["avg_ownership"] for lu in deconstructed) / len(deconstructed), 1
        ) if deconstructed else 0

        return {
            "top_lineups":     deconstructed[:5],
            "core_plays":      core_plays,
            "avg_winner_ownership": avg_winner_own,
            "winning_score":   field_stats["winning_score"],
        }

    # ── Chalk trap analysis ────────────────────────────────────────────────────

    def _chalk_trap_analysis(
        self,
        slate: pd.DataFrame,
        fpts_map: dict,
        actual_own: dict,
        own_threshold: float = 15.0,
        bust_threshold: float = 0.70,
    ) -> list[dict]:
        """
        Find highly-owned players who underperformed — these are the plays that
        killed average lineups while winners pivoted away from them.
        """
        avg_map = dict(zip(slate["Name"].str.strip(), slate["AvgPointsPerGame"]))
        sal_map = dict(zip(slate["Name"].str.strip(), slate["Salary"]))

        traps = []
        for name, own in actual_own.items():
            if own < own_threshold:
                continue
            fpts = fpts_map.get(name, 0)
            avg  = avg_map.get(name, 0)
            if avg <= 0:
                continue
            ratio = fpts / avg
            if ratio < bust_threshold:
                traps.append({
                    "name":        name,
                    "ownership":   round(own, 1),
                    "actual_fpts": round(fpts, 2),
                    "avg_pts":     round(avg, 2),
                    "ratio":       round(ratio, 2),
                    "salary":      sal_map.get(name, 0),
                    "pts_lost":    round(avg - fpts, 2),  # how much below avg
                    "severity":    "CATASTROPHIC" if ratio < 0.40 else
                                   "SEVERE" if ratio < 0.55 else "MODERATE",
                })

        traps.sort(key=lambda x: (x["ownership"] * (1 - x["ratio"])), reverse=True)
        return traps

    # ── Leverage play finder ───────────────────────────────────────────────────

    def _leverage_play_finder(
        self,
        slate: pd.DataFrame,
        fpts_map: dict,
        actual_own: dict,
        own_ceiling: float = 10.0,
        multiplier_floor: float = 1.40,
        fpts_floor: float = 22.0,
    ) -> list[dict]:
        """
        Find low-owned players who scored well above their average — these are
        the plays that won contests. Categorise WHY they blew up (injury
        replacement, hot shooting, role expansion).
        """
        avg_map = dict(zip(slate["Name"].str.strip(), slate["AvgPointsPerGame"]))
        sal_map = dict(zip(slate["Name"].str.strip(), slate["Salary"]))
        team_map = dict(zip(slate["Name"].str.strip(), slate["TeamAbbrev"]))

        plays = []
        for name, fpts in fpts_map.items():
            if fpts < fpts_floor:
                continue
            own = actual_own.get(name, 0)
            if own > own_ceiling:
                continue
            avg = avg_map.get(name, 0)
            if avg <= 0:
                continue
            multiplier = fpts / avg
            if multiplier < multiplier_floor:
                continue

            # Classify the likely reason
            sal = sal_map.get(name, 0)
            if sal < 5000:
                category = "CHEAP_DART"
            elif avg < 20:
                category = "ROLE_EXPANSION"
            elif own < 3:
                category = "CONTRARIAN_STAR"
            else:
                category = "VALUE_PIVOT"

            plays.append({
                "name":       name,
                "team":       team_map.get(name, ""),
                "salary":     sal,
                "ownership":  round(own, 1),
                "actual_fpts": round(fpts, 2),
                "avg_pts":    round(avg, 2),
                "multiplier": round(multiplier, 2),
                "category":   category,
                "leverage_score": round((fpts / max(own, 0.1)) * (multiplier - 1), 1),
            })

        plays.sort(key=lambda x: x["leverage_score"], reverse=True)
        return plays

    # ── Lineup autopsy ─────────────────────────────────────────────────────────

    def _lineup_autopsy(
        self,
        entries: list[list[str]],
        fpts_map: dict,
        actual_own: dict,
        slate: pd.DataFrame,
        field_stats: dict,
    ) -> dict:
        """
        For each submitted lineup, find:
          - The "anchor bust": the player who most cost the lineup
          - Whether we were exposed to chalk traps
          - What the lineup would have scored with the best available swap
        """
        avg_map = dict(zip(slate["Name"].str.strip(), slate["AvgPointsPerGame"]))
        cash_line = field_stats["p50"]

        autopsies = []
        for i, names in enumerate(entries, 1):
            scored = [(n, fpts_map.get(n, 0), avg_map.get(n, 0),
                       actual_own.get(n, 0)) for n in names]
            total = sum(s[1] for s in scored)

            # Anchor bust: player whose actual vs avg gap was worst
            worst = min(scored, key=lambda x: x[1] - x[2] if x[2] > 0 else 0)
            best_actual = max(scored, key=lambda x: x[1])

            # Exposure to chalk traps (>15% owned, scored <70% of avg)
            chalk_exposures = [
                s[0] for s in scored
                if s[3] >= 15 and s[2] > 0 and s[1] < s[2] * 0.70
            ]

            autopsies.append({
                "lineup_num":      i,
                "score":           round(total, 2),
                "cashed":          total >= cash_line,
                "gap_to_cash":     round(cash_line - total, 2),
                "anchor_bust":     worst[0],
                "bust_actual":     round(worst[1], 2),
                "bust_avg":        round(worst[2], 2),
                "bust_shortfall":  round(worst[2] - worst[1], 2),
                "best_player":     best_actual[0],
                "best_fpts":       round(best_actual[1], 2),
                "chalk_exposures": chalk_exposures,
            })

        # Aggregate findings
        all_busts = [a["anchor_bust"] for a in autopsies]
        from collections import Counter
        bust_freq = Counter(all_busts)
        chalk_exp_freq = Counter(
            p for a in autopsies for p in a["chalk_exposures"]
        )
        avg_bust_shortfall = round(
            sum(a["bust_shortfall"] for a in autopsies) / len(autopsies), 2
        ) if autopsies else 0

        return {
            "per_lineup":          autopsies,
            "most_common_bust":    bust_freq.most_common(5),
            "chalk_trap_exposure": chalk_exp_freq.most_common(5),
            "avg_bust_shortfall":  avg_bust_shortfall,
            "lineups_with_chalk_bust": sum(1 for a in autopsies if a["chalk_exposures"]),
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_lineup_str(lineup_str: str) -> list[str]:
        """Extract player names from DK lineup string like 'C Jay Huff F Desmond Bane ...'"""
        slots  = {"PG","SG","SF","PF","C","G","F","UTIL"}
        tokens = lineup_str.split()
        names, buf = [], []
        for tok in tokens:
            if tok in slots:
                if buf: names.append(" ".join(buf))
                buf = []
            else:
                buf.append(tok)
        if buf: names.append(" ".join(buf))
        return [n.strip() for n in names if n.strip()]

    # ── Recommendations ────────────────────────────────────────────────────────

    def _build_recommendations(
        self,
        cash_summary: dict,
        on_off_acc: list[dict],
        own_accuracy: dict,
        construction: dict,
        chalk_traps: list[dict] = None,
        leverage_plays: list[dict] = None,
        autopsy: dict = None,
    ) -> list[dict]:
        recs = []

        # Cash rate
        cash_rate = cash_summary.get("cash_rate", 0)
        gap       = cash_summary.get("gap_to_cash", 0)
        if cash_rate < 0.25:
            recs.append({
                "category": "SCORING",
                "priority": "HIGH",
                "finding":  f"Cash rate {cash_rate:.0%} — averaging {gap:.1f} pts below cash line",
                "action":   "Increase min_proj_total floor in optimizer; "
                            "de-prioritise leverage in small slates",
            })
        elif cash_rate > 0.60:
            recs.append({
                "category": "SCORING",
                "priority": "LOW",
                "finding":  f"Cash rate {cash_rate:.0%} — strong performance",
                "action":   "Current projection accuracy is good; focus on ownership leverage for GPP",
            })

        # On/off accuracy
        errors = [
            t["prediction_error"]
            for inj in on_off_acc
            for t in inj["teammates"]
            if t["prediction_error"] is not None
        ]
        if errors:
            mean_err = round(sum(errors) / len(errors), 2)
            mae      = round(sum(abs(e) for e in errors) / len(errors), 2)
            if abs(mean_err) > 2.0:
                direction = "over-estimates" if mean_err > 0 else "under-estimates"
                recs.append({
                    "category": "ON/OFF",
                    "priority": "MEDIUM",
                    "finding":  f"On/off model {direction} injury boosts by {abs(mean_err):.1f} DK pts on average (MAE={mae:.1f})",
                    "action":   (f"Scale down delta_dk by ~{abs(mean_err)/max(mae,1)*100:.0f}% "
                                 "in estimate_usage_absorption") if mean_err > 0 else
                                (f"Scale up delta_dk — model is too conservative on injury impact"),
                })

        # Ownership bias
        bias = own_accuracy.get("bias", 0)
        corr = own_accuracy.get("correlation", 1)
        if abs(bias) > 3:
            recs.append({
                "category": "OWNERSHIP",
                "priority": "MEDIUM",
                "finding":  f"Ownership model bias = {bias:+.1f}% (correlation={corr:.2f})",
                "action":   ("Reduce sal_rank^2 weight in proj_own formula — model over-projects "
                             "chalk ownership, lineups appear less contrarian than intended")
                             if bias > 0 else
                            ("Increase sal_rank^2 weight — model under-projects chalk ownership, "
                             "leverage scores are inflated"),
            })

        # Construction
        top1 = construction.get("top_1pct", {})
        top10 = construction.get("top_10pct", {})
        if top1.get("avg_stack", 0) > 3.5:
            recs.append({
                "category": "CONSTRUCTION",
                "priority": "MEDIUM",
                "finding":  f"Top 1% lineups averaged {top1['avg_stack']:.1f}-man stacks",
                "action":   "Increase stack_bonus in generate_gpp_lineups; "
                            "ensure 3+ player game stacks in all lineups",
            })
        if top1.get("avg_salary", 0) and top1["avg_salary"] > 49500:
            recs.append({
                "category": "CONSTRUCTION",
                "priority": "LOW",
                "finding":  f"Top 1% avg salary: ${top1['avg_salary']:,} — nearly full spend",
                "action":   "Raise MIN_SALARY_USED threshold; "
                            "stop leaving salary on the table",
            })

        # On/off missed opportunities
        missed = [
            inj for inj in on_off_acc
            if inj["biggest_gain"] > 8 and
               (inj["predicted_gain"] is None or inj["predicted_gain"] < inj["biggest_gain"] * 0.5)
        ]
        if missed:
            names = [m["injured_player"] for m in missed[:3]]
            recs.append({
                "category": "ON/OFF",
                "priority": "HIGH",
                "finding":  f"Missed injury opportunities: {', '.join(names)} — "
                            f"teammates had large actual gains our model didn't predict",
                "action":   "Ensure BBRefOnOffAgent runs for ALL confirmed OUT players "
                            "before lineup lock; widen ESPN roster cache coverage",
            })

        # Chalk trap exposure
        if chalk_traps and autopsy:
            n_lineups_burned = autopsy.get("lineups_with_chalk_bust", 0)
            if n_lineups_burned > 0:
                worst = chalk_traps[0]
                recs.append({
                    "category": "CHALK TRAP",
                    "priority": "HIGH",
                    "finding":  f"{n_lineups_burned} of your lineups contained chalk busts. "
                                f"Worst: {worst['name']} — {worst['actual_fpts']:.0f} actual "
                                f"vs {worst['avg_pts']:.0f} avg @ {worst['ownership']:.0f}% owned ({worst['severity']})",
                    "action":   "Before lock, identify the top 3 most-owned players and ask: "
                                "is there a credible reason they underperform tonight? "
                                "Fade at least one chalk per lineup to reduce correlated bust risk.",
                })

        # Leverage plays we missed
        if leverage_plays:
            top_missed = leverage_plays[:3]
            names_missed = [f"{p['name']} ({p['actual_fpts']:.0f} pts @ {p['ownership']:.1f}%)"
                            for p in top_missed]
            avg_leverage = round(sum(p["leverage_score"] for p in top_missed) / len(top_missed), 1)
            recs.append({
                "category": "LEVERAGE",
                "priority": "HIGH",
                "finding":  f"Top missed leverage plays: {' | '.join(names_missed)}. "
                            f"Avg leverage score {avg_leverage} — these were the contest-winning pivots.",
                "action":   "Review low-owned players (<8%) with salary under $6K or role-expansion "
                            "context (injury news, pace, matchup) and include at least 1-2 per lineup.",
            })

        # Lineup autopsy
        if autopsy:
            common_bust = autopsy.get("most_common_bust", [])
            if common_bust:
                bust_name, bust_count = common_bust[0]
                recs.append({
                    "category": "AUTOPSY",
                    "priority": "MEDIUM",
                    "finding":  f"'{bust_name}' was your anchor bust in {bust_count} lineups, "
                                f"avg shortfall {autopsy['avg_bust_shortfall']:.1f} pts below expectation.",
                    "action":   "Reduce exposure to this player archetype when they're at peak salary "
                                "and high ownership — the field is already pricing in their upside.",
                })

        return recs

    # ── Multi-slate aggregation ────────────────────────────────────────────────

    def _aggregate_reports(self, reports: list[dict]) -> dict:
        if not reports:
            return {"error": "No reports to aggregate"}

        all_cash_rates = [r["cash_summary"]["cash_rate"] for r in reports]
        all_gaps       = [r["cash_summary"]["gap_to_cash"] for r in reports]
        all_biases     = [r["ownership_accuracy"].get("bias", 0) for r in reports
                          if isinstance(r.get("ownership_accuracy"), dict)]

        # Aggregate on/off errors across all slates
        all_errors = []
        for r in reports:
            for inj in r.get("on_off_accuracy", []):
                for t in inj.get("teammates", []):
                    if t.get("prediction_error") is not None:
                        all_errors.append(t["prediction_error"])

        # Collect all recommendations
        all_recs = []
        seen_recs = set()
        for r in reports:
            for rec in r.get("recommendations", []):
                key = (rec["category"], rec["action"][:40])
                if key not in seen_recs:
                    all_recs.append(rec)
                    seen_recs.add(key)

        return {
            "slates_analysed":   len(reports),
            "daily_reports":     reports,
            "aggregate": {
                "avg_cash_rate":   round(sum(all_cash_rates) / len(all_cash_rates), 3),
                "avg_gap_to_cash": round(sum(all_gaps) / len(all_gaps), 2),
                "ownership_bias":  round(sum(all_biases) / len(all_biases), 2) if all_biases else None,
                "onoff_mean_error": round(sum(all_errors) / len(all_errors), 2) if all_errors else None,
                "onoff_mae":        round(sum(abs(e) for e in all_errors) / len(all_errors), 2)
                                    if all_errors else None,
            },
            "recommendations": all_recs,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_name(cell: str) -> str:
        """Extract player name from 'Player Name (ID)' or plain 'Player Name'."""
        cell = str(cell).strip()
        m = re.match(r"^(.+?)\s*\(\d+\)$", cell)
        return m.group(1).strip() if m else cell

    @staticmethod
    def _normalise_date(date_str: str) -> str:
        """Convert '3_8' or '3/8' or '2026-03-08' to 'YYYY-MM-DD'."""
        if not date_str:
            return ""
        # Already ISO
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return date_str
        # m_d or m/d format
        sep = "_" if "_" in date_str else "/"
        parts = date_str.split(sep)
        if len(parts) == 2:
            m, d = int(parts[0]), int(parts[1])
            return f"2026-{m:02d}-{d:02d}"
        return ""
