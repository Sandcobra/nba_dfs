"""
GPP Tournament — Main Entry Point
===================================
Replaces test_slate.py as the primary lineup generation system.

Two modes:
  1. Manual mode:    python gpp_tournament.py [--slate path/to/dk_slate.csv]
  2. Autonomous mode: python gpp_tournament.py --auto

In autonomous mode the system runs forever, watching for new slates and
processing contest results without any human input.

The objective function is FIRST-PLACE EQUITY, not expected value.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tournament.autonomous_runner import AutonomousRunner

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NBA DFS Tournament Engine — First-Place Equity Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on today's slate (manual):
  python gpp_tournament.py

  # Run on a specific slate file:
  python gpp_tournament.py --slate ../dk_slate.csv

  # Process yesterday's contest results:
  python gpp_tournament.py --results ../contest/contest-results_3_9.csv

  # Autonomous mode (watch forever, auto-process all slates + results):
  python gpp_tournament.py --auto

  # Full pipeline with custom parameters:
  python gpp_tournament.py --slate ../dk_slate.csv --lineups 20 --sim 5000
        """,
    )
    parser.add_argument("--slate",    type=str, default=None,
                        help="Path to DraftKings salary CSV (default: auto-detect)")
    parser.add_argument("--results",  type=str, default=None,
                        help="Path to contest results CSV (triggers SelfImprover calibration)")
    parser.add_argument("--auto",     action="store_true",
                        help="Autonomous mode: run forever, process all slates + results")
    parser.add_argument("--lineups",  type=int, default=20,
                        help="Number of lineups to build (default: 20)")
    parser.add_argument("--sim",      type=int, default=3000,
                        help="Monte Carlo simulations per lineup (default: 3000)")
    parser.add_argument("--username", type=str, default="Sandcobra",
                        help="DraftKings username for postmortem analysis")
    args = parser.parse_args()

    runner = AutonomousRunner(
        our_username=args.username,
        n_lineups=args.lineups,
        n_sim=args.sim,
    )

    if args.results:
        # Just process contest results, update calibration
        runner.process_results(Path(args.results))

    elif args.auto:
        # Full autonomous mode — runs forever
        runner.run_forever()

    else:
        # Single slate run
        slate = Path(args.slate) if args.slate else None
        upload_path = runner.run_once(slate)
        if upload_path:
            print(f"\nDone. Upload: {upload_path}")
        else:
            print("\nNo lineups generated.")
            sys.exit(1)
