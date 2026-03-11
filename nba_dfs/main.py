"""
NBA DFS Model — Main Entry Point
=================================
Usage examples:

  # Generate GPP lineups from DK salary file:
  python main.py --salary DKSalaries.csv --site dk --contest gpp --lineups 150

  # Generate cash lineups:
  python main.py --salary DKSalaries.csv --site dk --contest cash --lineups 20

  # Lock specific players:
  python main.py --salary DKSalaries.csv --lock "Luka Doncic" "Nikola Jokic"

  # Train models on historical data (run once before first slate):
  python main.py --train

  # Interactive mode:
  python main.py --interactive
"""

import argparse
import sys
import signal
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.prompt import Prompt, Confirm

from core.orchestrator import DFSOrchestrator
from core.config import (
    LOGS_DIR, NUM_LINEUPS_GPP, NUM_LINEUPS_CASH, CURRENT_SEASON
)

console = Console()

# Configure logging
logger.remove()
logger.add(
    str(LOGS_DIR / "nba_dfs_{time:YYYY-MM-DD}.log"),
    rotation="1 day",
    retention="14 days",
    level="INFO",
    format="{time:HH:mm:ss} | {level:<8} | {message}",
)
logger.add(sys.stderr, level="WARNING", format="{time:HH:mm:ss} | {level:<8} | {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NBA DFS Model — World-Class Lineup Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--salary",     type=str, help="Path to DK/FD salary CSV file")
    parser.add_argument("--site",       type=str, default="dk",     choices=["dk", "fd"],
                        help="DFS site (dk or fd)")
    parser.add_argument("--contest",    type=str, default="gpp",
                        choices=["gpp", "cash", "double_up", "showdown"],
                        help="Contest type")
    parser.add_argument("--lineups",    type=int, default=None,
                        help="Number of lineups to generate")
    parser.add_argument("--lock",       nargs="+", default=[],
                        help="Player name(s) to lock into all lineups")
    parser.add_argument("--exclude",    nargs="+", default=[],
                        help="Player name(s) to exclude from all lineups")
    parser.add_argument("--date",       type=str, default=None,
                        help="Slate date YYYY-MM-DD (default: today)")
    parser.add_argument("--train",      action="store_true",
                        help="Train ML models before generating lineups")
    parser.add_argument("--train-only", action="store_true",
                        help="Train models only (no lineup generation)")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Disable real-time injury monitor")
    parser.add_argument("--seasons",    type=int, default=2,
                        help="Number of historical seasons for training")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--season",     type=str, default=CURRENT_SEASON,
                        help=f"NBA season (default: {CURRENT_SEASON})")
    return parser.parse_args()


def interactive_mode(orch: DFSOrchestrator):
    """Interactive CLI for running the model."""
    console.rule("[bold blue]NBA DFS Model — Interactive Mode")

    salary_file = Prompt.ask("Path to salary CSV file")
    if not Path(salary_file).exists():
        console.print(f"[red]File not found: {salary_file}[/red]")
        return

    site = Prompt.ask("Site", choices=["dk", "fd"], default="dk")
    contest = Prompt.ask(
        "Contest type", choices=["gpp", "cash", "double_up", "showdown"], default="gpp"
    )

    n_gpp  = int(Prompt.ask("Number of GPP lineups", default="150"))
    n_cash = int(Prompt.ask("Number of cash lineups", default="20"))

    lock_str   = Prompt.ask("Lock players (comma-separated, or blank)", default="")
    excl_str   = Prompt.ask("Exclude players (comma-separated, or blank)", default="")

    locked   = [p.strip() for p in lock_str.split(",") if p.strip()]
    excluded = [p.strip() for p in excl_str.split(",") if p.strip()]

    train_first = Confirm.ask("Re-train ML models?", default=False)
    monitor     = Confirm.ask("Enable real-time monitor?", default=True)

    result = orch.run_slate(
        salary_file=salary_file,
        site=site,
        contest_type=contest,
        n_lineups_gpp=n_gpp,
        n_lineups_cash=n_cash,
        locked_players=locked or None,
        excluded_players=excluded or None,
        train_first=train_first,
        enable_monitor=monitor,
    )

    if monitor and result.get("monitor"):
        console.print(
            "\n[bold yellow]Monitor is running. Press Ctrl+C to stop.[/bold yellow]"
        )
        try:
            import time
            while True:
                alerts = result["monitor"].get_alerts()
                if alerts:
                    console.print(f"\n[bold red]{len(alerts)} alerts detected[/bold red]")
                time.sleep(30)
        except KeyboardInterrupt:
            console.print("\nStopping monitor...")
            result["monitor"].stop()


def main():
    args = parse_args()
    orch = DFSOrchestrator(season=args.season)

    # Handle SIGINT for graceful shutdown
    def _shutdown(sig, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        orch.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)

    # Training-only mode
    if args.train_only:
        orch.train_models(n_seasons=args.seasons)
        return

    # Interactive mode
    if args.interactive:
        interactive_mode(orch)
        return

    # Salary file required for lineup generation
    if not args.salary:
        console.print(
            "[red]Error: --salary is required (unless using --train-only or --interactive)[/red]"
        )
        console.print("  Run [bold]python main.py --help[/bold] for usage")
        sys.exit(1)

    salary_path = Path(args.salary)
    if not salary_path.exists():
        console.print(f"[red]Salary file not found: {args.salary}[/red]")
        sys.exit(1)

    # Determine lineup counts
    n_gpp  = args.lineups if args.lineups else NUM_LINEUPS_GPP
    n_cash = args.lineups if args.lineups else NUM_LINEUPS_CASH
    if args.contest in ("cash", "double_up"):
        n_gpp  = 0
        n_cash = args.lineups or NUM_LINEUPS_CASH

    result = orch.run_slate(
        salary_file=str(salary_path),
        site=args.site,
        contest_type=args.contest,
        n_lineups_gpp=n_gpp,
        n_lineups_cash=n_cash,
        locked_players=args.lock or None,
        excluded_players=args.exclude or None,
        slate_date=args.date,
        train_first=args.train,
        enable_monitor=not args.no_monitor,
    )

    # If monitor is running, keep process alive
    if not args.no_monitor and result.get("monitor"):
        import time
        console.print(
            "\n[bold yellow]Monitor running. Press Ctrl+C to stop and exit.[/bold yellow]"
        )
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping...[/yellow]")
            orch.stop()


if __name__ == "__main__":
    main()
