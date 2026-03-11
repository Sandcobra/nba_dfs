"""
Master Orchestrator.
Coordinates all agents in the correct order to produce a complete
NBA DFS lineup set for any given slate.

Pipeline:
  DataAgent → MLAgent (projections) → MathAgent (MC/Bayesian) →
  GameTheoryAgent (ownership/leverage) → CorrelationModel (stacks) →
  LineupOptimizer (ILP) → MonitorAgent (real-time watch)
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from agents.data_agent       import DataAgent
from agents.ml_agent         import MLAgent
from agents.math_agent       import MathAgent
from agents.game_theory_agent import GameTheoryAgent
from agents.monitor_agent    import LineupMonitorAgent
from models.correlation_model import CorrelationModel
from optimization.lineup_optimizer import LineupOptimizer
from data.dk_parser          import parse_dk_salary_csv, parse_fd_salary_csv, merge_salary_with_projections
from core.database           import get_db
from core.config             import (
    LINEUPS_DIR, NUM_LINEUPS_GPP, NUM_LINEUPS_CASH,
    CURRENT_SEASON,
)


console = Console()


class DFSOrchestrator:
    """
    Top-level coordinator for the NBA DFS model.
    Call run_slate() with a salary file path to produce lineups.
    """

    def __init__(self, season: str = CURRENT_SEASON):
        self.season   = season
        self.db       = get_db()
        self.data_agent   = DataAgent(season=season)
        self.ml_agent     = MLAgent()
        self.math_agent   = MathAgent()
        self.gt_agent     = GameTheoryAgent()
        self.corr_model   = CorrelationModel()
        self.monitor      = None

    # ── Main entry point ───────────────────────────────────────────────────────
    def run_slate(
        self,
        salary_file:   str,
        site:          str = "dk",
        contest_type:  str = "gpp",
        n_lineups_gpp: int = NUM_LINEUPS_GPP,
        n_lineups_cash: int = NUM_LINEUPS_CASH,
        locked_players:   list[str] = None,
        excluded_players: list[str] = None,
        slate_date:    Optional[str] = None,
        train_first:   bool = False,
        enable_monitor: bool = True,
    ) -> dict:
        """
        Full end-to-end DFS pipeline.

        Args:
            salary_file:     Path to DK/FD salary CSV export
            site:            'dk' or 'fd'
            contest_type:    'gpp', 'cash', 'double_up', 'showdown'
            n_lineups_gpp:   Number of GPP lineups to generate
            n_lineups_cash:  Number of cash lineups to generate
            locked_players:  Player names to force into all lineups
            excluded_players: Player names to exclude from all lineups
            slate_date:      Override date (YYYY-MM-DD)
            train_first:     Re-train ML models before projecting
            enable_monitor:  Start real-time injury monitor

        Returns:
            dict with lineups, projections, insights, metadata
        """
        today = slate_date or date.today().isoformat()
        console.rule(f"[bold blue]NBA DFS Model | {today} | {site.upper()} {contest_type.upper()}")

        # ── Step 1: Parse salary file ─────────────────────────────────────────
        console.print("[1/8] Parsing salary file...", style="cyan")
        player_pool = self._parse_salary(salary_file, site)
        if player_pool.empty:
            raise ValueError("Empty player pool after parsing salary file")
        console.print(f"  {len(player_pool)} players in pool", style="green")

        # ── Step 2: Collect all data ──────────────────────────────────────────
        console.print("[2/8] Aggregating data (NBA API, injuries, Vegas)...", style="cyan")
        slate_data = self.data_agent.build_slate_data(player_pool, today)
        enriched   = slate_data["enriched_pool"]
        game_logs  = slate_data["game_logs"]
        injuries   = slate_data["injuries"]
        console.print(f"  {len(enriched)} players after enrichment", style="green")

        # ── Step 3: Train ML models (optional) ───────────────────────────────
        if train_first:
            console.print("[3/8] Training ML models on historical data...", style="cyan")
            hist = self.data_agent.fetch_historical_data(n_seasons=2)
            if not hist.empty:
                self.ml_agent.train(hist, game_logs)
        else:
            console.print("[3/8] Using pre-trained ML models", style="dim cyan")

        # ── Step 4: Generate projections ──────────────────────────────────────
        console.print("[4/8] Generating ensemble projections + Monte Carlo...", style="cyan")
        projections = self.ml_agent.predict(enriched, game_logs)
        console.print(
            f"  Top projection: {projections.iloc[0]['name']} "
            f"{projections.iloc[0]['projected_pts_dk']:.1f} DK pts",
            style="green"
        )

        # ── Step 5: Game theory (ownership, leverage) ─────────────────────────
        console.print("[5/8] Computing ownership & leverage scores...", style="cyan")
        strategy = self.gt_agent.get_contest_strategy(contest_type)
        projections = self.gt_agent.compute_leverage_scores(projections, contest_type)
        projections = self.gt_agent.compute_optimal_exposures(projections, n_lineups_gpp, contest_type)
        projections = self.gt_agent.adjust_for_injuries(projections, injuries)
        projections = self.gt_agent.model_field_composition(projections)

        # ── Step 6: Correlation / stacking analysis ───────────────────────────
        console.print("[6/8] Analyzing player correlations & stacks...", style="cyan")
        self.corr_model.build_correlation_matrix(game_logs, projections)
        stacks = self.corr_model.get_teammate_stacks(
            projections,
            min_stack=strategy.get("target_stack_size", 2),
            max_stack=strategy.get("target_stack_size", 3) + 1,
        )
        game_stacks = self.corr_model.get_game_stacks(projections) if strategy.get("use_game_stack") else []
        all_stacks  = stacks[:20] + game_stacks[:5]
        console.print(f"  {len(stacks)} team stacks, {len(game_stacks)} game stacks found", style="green")

        # ── Step 7: Resolve locked/excluded by name → ID ─────────────────────
        lock_ids = self._resolve_ids(projections, locked_players or [])
        excl_ids = self._resolve_ids(projections, excluded_players or [])

        # ── Step 8: Optimize lineups ──────────────────────────────────────────
        console.print("[7/8] Running ILP lineup optimizer...", style="cyan")
        optimizer = LineupOptimizer(site=site, contest_type=contest_type)

        if contest_type in ("gpp", "showdown"):
            n_lineups = n_lineups_gpp
        else:
            n_lineups = n_lineups_cash

        lineups = optimizer.generate_lineups(
            player_pool=projections,
            n_lineups=n_lineups,
            locked_players=lock_ids,
            excluded_players=excl_ids,
            stacks=all_stacks,
            slate_date=today,
            ownership_penalty=strategy.get("ownership_penalty", 0.03),
        )

        # ── Save results ──────────────────────────────────────────────────────
        console.print("[8/8] Saving results...", style="cyan")
        output_path = self._save_results(lineups, projections, today, site, contest_type)

        # Export DK upload CSV
        upload_csv = str(LINEUPS_DIR / f"dk_upload_{today}.csv")
        try:
            optimizer.export_to_dk_csv(lineups, projections, upload_csv)
        except Exception as e:
            logger.warning(f"DK CSV export issue: {e}")

        # Exposure analysis
        exposure_df = optimizer.analyze_lineup_set(lineups, len(lineups))

        # Insights
        insights = self.ml_agent.generate_insights(projections)

        # ── Start monitor ─────────────────────────────────────────────────────
        if enable_monitor:
            self.monitor = LineupMonitorAgent()
            self.monitor.set_active_lineups(lineups)
            self.monitor.start(player_pool=projections)
            console.print("  Real-time monitor started", style="green")

        # ── Print summary ─────────────────────────────────────────────────────
        self._print_summary(projections, lineups, insights, stacks[:5])

        console.rule("[bold green]Done")
        console.print(f"Output: [bold]{output_path}[/bold]")
        if upload_csv:
            console.print(f"DK Upload CSV: [bold]{upload_csv}[/bold]")

        return {
            "slate_date":    today,
            "site":          site,
            "contest_type":  contest_type,
            "lineups":       lineups,
            "projections":   projections,
            "exposure":      exposure_df,
            "stacks":        stacks[:10],
            "insights":      insights,
            "injuries":      injuries,
            "output_path":   output_path,
            "upload_csv":    upload_csv,
            "monitor":       self.monitor,
        }

    # ── Training-only mode ─────────────────────────────────────────────────────
    def train_models(self, n_seasons: int = 2):
        """
        Standalone model training. Fetches historical data and trains
        all ML models. Run this once before your first slate.
        """
        console.rule("[bold yellow]Training NBA DFS Models")
        logger.info(f"Fetching {n_seasons} seasons of historical data...")
        hist     = self.data_agent.fetch_historical_data(n_seasons=n_seasons)
        if hist.empty:
            logger.error("No historical data available for training")
            return

        # Need game logs dict for feature engineering
        pids = hist["PLAYER_ID"].unique() if "PLAYER_ID" in hist.columns else []
        game_logs = {}
        for pid in pids[:200]:  # limit for training speed
            try:
                gl = hist[hist["PLAYER_ID"] == pid].copy()
                col_map = {"PTS": "pts", "REB": "reb", "AST": "ast", "STL": "stl",
                           "BLK": "blk", "TOV": "tov", "FG3M": "fg3m", "MIN": "min",
                           "GAME_DATE": "game_date", "fantasy_pts_dk": "fantasy_pts_dk"}
                gl = gl.rename(columns={k: v for k, v in col_map.items() if k in gl.columns})
                game_logs[int(pid)] = gl
            except Exception:
                pass

        self.ml_agent.train(hist, game_logs)
        logger.success("Model training complete")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _parse_salary(self, salary_file: str, site: str) -> pd.DataFrame:
        try:
            if site.lower() == "dk":
                return parse_dk_salary_csv(salary_file)
            else:
                return parse_fd_salary_csv(salary_file)
        except Exception as e:
            logger.error(f"Salary parse failed: {e}")
            return pd.DataFrame()

    def _resolve_ids(self, pool: pd.DataFrame, names: list[str]) -> list[int]:
        ids = []
        for name in names:
            match = pool[pool["name"].str.lower() == name.lower()]
            if not match.empty:
                ids.append(int(match.iloc[0]["player_id"]))
            else:
                logger.warning(f"Could not find player ID for: {name}")
        return ids

    def _save_results(
        self,
        lineups: list[dict],
        projections: pd.DataFrame,
        slate_date: str,
        site: str,
        contest_type: str,
    ) -> str:
        # Save lineups to DB
        for lu in lineups:
            lu["slate_date"]   = slate_date
            lu["site"]         = site
            lu["contest_type"] = contest_type
        try:
            self.db.save_lineups(lineups)
        except Exception as e:
            logger.warning(f"DB lineup save failed: {e}")

        # Save projections to DB
        try:
            self.db.save_projections(projections, slate_date)
        except Exception as e:
            logger.warning(f"DB projection save failed: {e}")

        # Write lineups JSON
        out_path = LINEUPS_DIR / f"lineups_{slate_date}_{site}_{contest_type}.json"
        with open(out_path, "w") as f:
            json.dump(lineups, f, indent=2, default=str)

        # Write projections CSV
        proj_path = LINEUPS_DIR / f"projections_{slate_date}.csv"
        projections.to_csv(proj_path, index=False)

        return str(out_path)

    def _print_summary(
        self,
        projections: pd.DataFrame,
        lineups: list[dict],
        insights: list[str],
        top_stacks: list[dict],
    ):
        # Top 15 projections table
        table = Table(title="Top Player Projections", show_header=True, header_style="bold cyan")
        table.add_column("Name",    style="white",  min_width=20)
        table.add_column("Team",    style="dim",    min_width=5)
        table.add_column("Pos",     style="dim",    min_width=6)
        table.add_column("Salary",  style="yellow", min_width=8)
        table.add_column("Proj",    style="green",  min_width=7)
        table.add_column("Ceil",    style="cyan",   min_width=7)
        table.add_column("Floor",   style="red",    min_width=7)
        table.add_column("Own%",    style="magenta",min_width=7)
        table.add_column("Value",   style="green",  min_width=7)

        for _, row in projections.head(15).iterrows():
            table.add_row(
                str(row.get("name", ""))[:22],
                str(row.get("team", ""))[:4],
                str(row.get("primary_position", "?"))[:5],
                f"${row.get('salary', 0):,}",
                f"{row.get('projected_pts_dk', 0):.1f}",
                f"{row.get('ceiling', 0):.1f}",
                f"{row.get('floor', 0):.1f}",
                f"{row.get('proj_ownership', 0):.1f}%",
                f"{row.get('value_dk', 0):.2f}x",
            )
        console.print(table)

        # Top stacks
        if top_stacks:
            console.print("\n[bold cyan]Top Stacks:[/bold cyan]")
            for s in top_stacks[:5]:
                console.print(
                    f"  [{s['team']}] {', '.join(s['players'])} "
                    f"| Stack Score: {s['stack_score']:.1f}"
                )

        # Lineup summary
        if lineups:
            console.print(f"\n[bold green]{len(lineups)} lineups generated[/bold green]")
            avg_proj = sum(lu["proj_pts"] for lu in lineups) / len(lineups)
            avg_sal  = sum(lu["total_salary"] for lu in lineups) / len(lineups)
            console.print(f"  Avg projected: {avg_proj:.1f} DK pts | Avg salary used: ${avg_sal:,.0f}")

        # Insights
        if insights:
            console.print("\n[bold yellow]Key Insights:[/bold yellow]")
            for insight in insights[:8]:
                console.print(f"  • {insight}")

    def stop(self):
        """Gracefully shut down the monitor."""
        if self.monitor:
            self.monitor.stop()
