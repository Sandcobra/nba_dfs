"""NBA DFS Pro — FastAPI backend."""

import sys
import json
import asyncio
import threading
import tempfile
from pathlib import Path
from queue import Queue, Empty
from datetime import date

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — import from test_slate.py
# ---------------------------------------------------------------------------
UI_DIR = Path(__file__).parent
PROJECT_ROOT = UI_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nba_dfs.test_slate import (
    parse_salary_file,
    build_projections,
    load_fc_data,
    _merge_fc,
    enrich_projections,
    generate_gpp_lineups,
    select_portfolio,
    export_dk_csv,
    parse_lineup_csv,
    parse_contest_csv,
    find_top_stacks,
    apply_status_updates,
    scrape_espn_injuries,
    scrape_espn_starters,
    get_confirmed_starters,
    late_swap_lineups,
    get_locked_teams,
    contest_mode_late_swap,
    compute_true_dvp,
    get_team_pace,
    grade_game_matchups,
    build_player_correlation,
    estimate_usage_absorption,
    fetch_player_usage_rates,
    compute_lineup_usage_impact,
    score_lineup_leverage,
    # Archetype / B2B / DvP
    classify_player_archetype,
    fetch_b2b_teams,
    apply_b2b_adjustments,
    build_dvp_weights,
    apply_dvp_adjustments,
    apply_lineup_confirmation_dvp,
    ARCHETYPE_LABELS,
    build_game_totals_from_pool,
    apply_game_total_updates,
    fetch_vegas_lines,
    compute_optimal_split,
    apply_game_script_adjustments,
    compute_positional_scarcity,
    refresh_ownership,
    GAME_TOTALS,
    SALARY_CAP,
    OUTPUT_DIR,
    _SLOT_ORDER,
)

# ---------------------------------------------------------------------------
# Lazy-loaded agents (avoid import errors at startup if packages missing)
# ---------------------------------------------------------------------------
_slate_agent          = None
_props_agent          = None
_ref_agent            = None
_on_off_agent         = None
_backtest_agent       = None
_game_theory_agent    = None
_field_behavior_agent       = None
_adversarial_ownership_agent = None
_news_intel_agent            = None


def _get_news_intel_agent():
    global _news_intel_agent
    if _news_intel_agent is None:
        from nba_dfs.agents.news_intel_agent import NewsIntelAgent
        _news_intel_agent = NewsIntelAgent()
    return _news_intel_agent


def _get_backtest_agent():
    global _backtest_agent
    if _backtest_agent is None:
        from nba_dfs.agents.backtest_agent import BacktestAgent
        _backtest_agent = BacktestAgent(
            cache_dir=PROJECT_ROOT / "cache" / "espn"
        )
    return _backtest_agent


def _get_on_off_agent():
    global _on_off_agent
    if _on_off_agent is None:
        from nba_dfs.agents.bbref_on_off_agent import BBRefOnOffAgent
        _on_off_agent = BBRefOnOffAgent()
    return _on_off_agent


def _get_game_theory_agent():
    global _game_theory_agent
    if _game_theory_agent is None:
        from nba_dfs.agents.game_theory_agent import GameTheoryAgent
        _game_theory_agent = GameTheoryAgent()
    return _game_theory_agent


def _get_field_behavior_agent():
    global _field_behavior_agent
    if _field_behavior_agent is None:
        from nba_dfs.agents.field_behavior_agent import FieldBehaviorAgent
        _field_behavior_agent = FieldBehaviorAgent()
    return _field_behavior_agent


def _get_adversarial_ownership_agent():
    global _adversarial_ownership_agent
    if _adversarial_ownership_agent is None:
        from nba_dfs.agents.adversarial_ownership_agent import AdversarialOwnershipAgent
        _adversarial_ownership_agent = AdversarialOwnershipAgent()
    return _adversarial_ownership_agent


def _get_props_agent():
    global _props_agent
    if _props_agent is None:
        import os
        from nba_dfs.agents.props_agent import PlayerPropsAgent
        from nba_dfs.core.config import THE_ODDS_API_KEY
        _props_agent = PlayerPropsAgent(
            api_key=THE_ODDS_API_KEY or os.getenv("THE_ODDS_API_KEY", ""),
        )
    return _props_agent


def _get_ref_agent():
    global _ref_agent
    if _ref_agent is None:
        from nba_dfs.agents.ref_agent import RefAgent
        _ref_agent = RefAgent()
    return _ref_agent


# ---------------------------------------------------------------------------
# Slate Construction Agent — lazy-loaded so missing anthropic package doesn't
# break startup; only initialised on first use.
# ---------------------------------------------------------------------------

def _get_slate_agent():
    global _slate_agent
    if _slate_agent is None:
        import os, sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from nba_dfs.agents.slate_agent import SlateConstructionAgent
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        _slate_agent = SlateConstructionAgent(api_key=api_key)
    return _slate_agent

def _run_slate_analysis(players) -> dict:
    """Run SlateConstructionAgent against current player pool and game totals."""
    try:
        agent      = _get_slate_agent()
        game_totals = _state.get("game_totals") or GAME_TOTALS
        injuries   = []  # populated from _state if available in future
        return agent.analyze(players, game_totals, injuries=injuries)
    except Exception as exc:
        import logging
        logging.warning("[slate-agent] Analysis failed: %s", exc)
        from nba_dfs.agents.slate_agent import _DEFAULT_PROFILE
        return dict(_DEFAULT_PROFILE)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="NBA DFS Pro", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# In-memory state (single-session)
# ---------------------------------------------------------------------------
_state: dict = {
    "players": None,
    "lineups": [],
    "job_running": False,
    "job_queue": None,
    "dk_path": None,
    "_full_pool_snapshot": None,    # snapshot of full player pool at upload time (for DvP tracking)
    "not_rostered_ids": set(),      # player IDs removed after starters check — not on active roster
    "contest": None,                # parsed contest standings data (real ownership, field lineups)
    "game_totals": None,            # dynamic game totals built from uploaded salary CSV
    "slate_profile": None,          # SlateConstructionAgent output for tonight's slate
    "positional_scarcity": {},      # viable player counts per DK slot
    "on_off_map": {},               # {out_player_id: {teammate_id: on_off_delta}} from OnOffAgent
    "confirmed_out_ids": set(),     # player_id strings confirmed OUT — always excluded from optimizer
    "signal_health": None,          # set by injury_check; None means injury check has not been run
    "field_profile": None,          # FieldBehaviorAgent output for tonight's slate
    "adversarial_profile": None,    # AdversarialOwnershipAgent output for tonight's slate
    "news_intel": None,             # NewsIntelAgent output for tonight's slate
    "fc_data": None,               # Fantasy Cruncher player CSV (FC Proj, Proj Own%, Proj Mins)
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = UI_DIR / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Parse a DK/FD salary CSV and return the player pool + analysis."""
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        raw = parse_salary_file(tmp_path)
        # Merge FC data if previously uploaded via /api/upload-fc
        if _state["fc_data"] is not None:
            raw = _merge_fc(raw, _state["fc_data"])
        players = build_projections(raw)
        players = players[players["proj_pts_dk"] > 0].copy()

        # Enrich with tail index (ξ), regime factors, cascade detection,
        # and tail-adjusted ceilings. Fetches game logs from NBA Stats API;
        # falls back gracefully if the API is unreachable.
        try:
            players = enrich_projections(players)
        except Exception as _enrich_err:
            import logging as _log
            _log.warning("[upload] enrich_projections failed: %s — using base projections", _enrich_err)

        game_totals = build_game_totals_from_pool(players)

        # ── Game script adjustment ──────────────────────────────────────────
        try:
            players = apply_game_script_adjustments(players, game_totals)
        except Exception as _gs_err:
            import logging as _log
            _log.warning("[upload] game_script failed: %s", _gs_err)

        # ── Props blend + line movement baseline ────────────────────────────
        try:
            pa = _get_props_agent()
            pa.store_baseline(game_totals)          # snapshot opening lines
            players = pa.apply_to_players(players, game_totals)
        except Exception as _pa_err:
            import logging as _log
            _log.warning("[upload] props_agent failed: %s", _pa_err)

        # ── Ref crew adjustment ─────────────────────────────────────────────
        try:
            ra = _get_ref_agent()
            players = ra.apply_ref_adjustments(players)
        except Exception as _ra_err:
            import logging as _log
            _log.warning("[upload] ref_agent failed: %s", _ra_err)

        # ── Positional scarcity ─────────────────────────────────────────────
        try:
            scarcity = compute_positional_scarcity(players)
        except Exception:
            scarcity = {}

        _state["players"]             = players
        _state["lineups"]             = []
        _state["dk_path"]             = None
        _state["_full_pool_snapshot"] = players.copy()
        _state["game_totals"]         = game_totals
        _state["slate_profile"]       = None
        _state["positional_scarcity"] = scarcity

        # Serialisable player records
        cols = [
            "player_id", "name", "team", "primary_position", "archetype", "salary",
            "avg_pts", "proj_pts_dk", "ceiling", "floor",
            "value", "proj_own", "gpp_score", "matchup", "game_total", "eligible_slots",
            "is_b2b", "b2b_penalty", "b2b_boost", "dvp_mult",
            "tail_index", "regime_factor",
        ]
        # Only include columns that exist (guard against future changes)
        cols = [c for c in cols if c in players.columns]
        pool = players[cols].copy()
        pool["eligible_slots"] = pool["eligible_slots"].apply(lambda x: "/".join(x))

        stacks = find_top_stacks(players)
        # make stacks JSON-safe
        for s in stacks:
            s["players"] = s["players"][:4]

        # Per-game summary
        game_summary = []
        _game_totals_now = _state.get("game_totals") or GAME_TOTALS
        for matchup, gt in sorted(_game_totals_now.items(), key=lambda x: -x[1]["total"]):
            gp = players[players["matchup"] == matchup]
            game_summary.append({
                "matchup": matchup,
                "total": gt["total"],
                "home_implied": gt["home_implied"],
                "away_implied": gt["away_implied"],
                "player_count": len(gp),
                "top_proj": round(float(gp["proj_pts_dk"].max()), 1) if len(gp) else 0,
            })

        return {
            "status": "ok",
            "player_count": len(players),
            "players": pool.to_dict(orient="records"),
            "stacks": stacks[:8],
            "game_summary": game_summary,
            "positional_scarcity": scarcity,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/upload-fc")
async def upload_fc_csv(file: UploadFile = File(...)):
    """
    Accept a Fantasy Cruncher player CSV (draftkings_NBA_*.csv).
    Stores it in server state so the next /api/upload call will merge
    FC Proj, Proj Own%, Proj Mins, Floor, Ceiling into the player pool.
    """
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
        f.write(content)
        tmp_path = Path(f.name)

    try:
        fc = load_fc_data(tmp_path)
        if fc is None or fc.empty:
            raise HTTPException(
                status_code=400,
                detail="Could not parse FC data. Make sure this is a Fantasy Cruncher "
                       "DraftKings player CSV (draftkings_NBA_*.csv) with Proj Own%, "
                       "Proj Mins, and FC Proj columns."
            )
        _state["fc_data"] = fc
        return {
            "status": "ok",
            "players_loaded": len(fc),
            "message": f"FC data loaded: {len(fc)} players with FC Proj, Proj Own%, Proj Mins. "
                       f"Upload your DK salary CSV now to apply it."
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/run")
async def run_optimization(
    num_lineups: int = Form(20),
    max_per_team: int = Form(4),
    locked_ids: str = Form(""),
    excluded_ids: str = Form(""),
    ownership_penalty: float = Form(0.04),
    contest_size: int = Form(0),
    contest_type: str = Form("gpp"),
    pool_size: int = Form(0),
):
    """
    Launch optimization in a background thread; stream progress via /api/stream.

    When contest_size > 0, computes an optimal chalk/leverage split automatically
    using compute_optimal_split(). The slate size is derived from the uploaded player
    pool. A single generate_gpp_lineups() call covers both batches sequentially with
    a combined penalty schedule so prev_pids uniqueness tracking prevents cross-batch
    duplicates.
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")
    if _state["job_running"]:
        # Auto-clear if the queue is gone or empty — the thread died without resetting the flag
        q_alive = _state.get("job_queue")
        if q_alive is None or q_alive.empty():
            _state["job_running"] = False
        else:
            raise HTTPException(status_code=409, detail="Optimization already running.")

    # Safety filter: remove players not on active roster
    not_rostered_ids  = _state.get("not_rostered_ids", set())
    confirmed_out_ids = _state.get("confirmed_out_ids", set())
    opt_players = _state["players"]
    if not_rostered_ids:
        opt_players = opt_players[
            ~opt_players["player_id"].astype(str).isin(not_rostered_ids)
        ].copy().reset_index(drop=True)
    # Always exclude confirmed OUT players regardless of pool state
    if confirmed_out_ids:
        before = len(opt_players)
        opt_players = opt_players[
            ~opt_players["player_id"].astype(str).isin(confirmed_out_ids)
        ].copy().reset_index(drop=True)
        removed = before - len(opt_players)
        if removed:
            import logging
            logging.getLogger(__name__).info(
                "[run] Excluded %d confirmed-OUT player(s) from optimizer pool.", removed
            )

    lock_list = [x.strip() for x in locked_ids.split(",")  if x.strip()] or None
    excl_list = [x.strip() for x in excluded_ids.split(",") if x.strip()] or None

    # ── Auto-split computation ────────────────────────────────────────────────
    n_games   = int(opt_players["matchup"].nunique()) if "matchup" in opt_players.columns else 7
    split     = None
    pen_sched = None

    if contest_size > 0:
        split     = compute_optimal_split(num_lineups, contest_size, n_games)
        pen_sched = split["chalk_schedule"] + split["leverage_schedule"]

    q: Queue = Queue()
    _state["job_queue"]   = q
    _state["job_running"] = True
    _state["lineups"]     = []

    def _run():
        nonlocal opt_players   # reassigned in GT block; declare here to avoid UnboundLocalError
        class _Tee:
            def __init__(self, queue: Queue):
                self._q = queue
            def write(self, text: str):
                if text and text.strip():
                    self._q.put({"type": "log", "msg": text.strip()})
            def flush(self):
                pass

        old_stdout = sys.stdout
        sys.stdout = _Tee(q)
        try:
            # ── Slate construction analysis ───────────────────────────────────
            slate_profile = _state.get("slate_profile")
            if slate_profile is None:
                # Run inline if not already cached from /api/slate-profile
                slate_profile = _run_slate_analysis(opt_players)
                _state["slate_profile"] = slate_profile
            q.put({"type": "slate_profile", "profile": {
                "rationale":         slate_profile.get("rationale", ""),
                "key_opportunities": slate_profile.get("key_opportunities", []),
                "key_risks":         slate_profile.get("key_risks", []),
                "barbell_enabled":   slate_profile.get("barbell_enabled", False),
                "stud_threshold":    slate_profile.get("stud_threshold", 9000),
                "stud_min":          slate_profile.get("stud_min", 2),
                "cheap_threshold":   slate_profile.get("cheap_threshold", 4500),
                "cheap_min":         slate_profile.get("cheap_min", 3),
                "max_exposure_pct":  slate_profile.get("max_exposure_pct", 0.33),
                "stack_emphasis":    slate_profile.get("stack_emphasis", "medium"),
                "source":            slate_profile.get("source", "default"),
            }})

            # Broadcast split plan before lineups start generating
            if split:
                q.put({"type": "split", "split": {
                    "n_chalk":      split["n_chalk"],
                    "n_leverage":   split["n_leverage"],
                    "chalk_range":  split["chalk_pen_range"],
                    "leverage_range": split["leverage_pen_range"],
                    "description":  split["description"],
                }})

            # ── News Intel enrichment ─────────────────────────────────────────
            # Runs before GT/field analysis so projections already reflect
            # breaking news (bench demotions, starting replacements, etc.)
            try:
                _ni = _get_news_intel_agent()
                ni_result = _ni.analyze(opt_players, n_lineups=num_lineups)
                _state["news_intel"] = ni_result

                ni_impacts  = ni_result.get("impacts", {})
                ni_summary  = ni_result.get("summary", {})
                ni_signals  = ni_result.get("signals", [])

                if ni_impacts:
                    # Apply in-place to avoid rebinding opt_players (would cause
                    # UnboundLocalError for earlier references in this closure)
                    ni_excluded = []
                    for _pid_str, _impact in ni_impacts.items():
                        _mask = opt_players["player_id"].astype(str) == _pid_str
                        if not _mask.any():
                            continue
                        if _impact.get("exclude"):
                            ni_excluded.append(_pid_str)
                            continue
                        if "proj_pts_mult" in _impact:
                            opt_players.loc[_mask, "proj_pts_dk"] = (
                                opt_players.loc[_mask, "proj_pts_dk"] * _impact["proj_pts_mult"]
                            ).round(2)
                        if "own_mult" in _impact:
                            opt_players.loc[_mask, "proj_own"] = (
                                opt_players.loc[_mask, "proj_own"] * _impact["own_mult"]
                            ).clip(1, 40).round(1)
                        if "own_delta" in _impact:
                            opt_players.loc[_mask, "proj_own"] = (
                                opt_players.loc[_mask, "proj_own"] + _impact["own_delta"]
                            ).clip(1, 40).round(1)
                    # Recompute gpp_score after news adjustments
                    if "ceiling" in opt_players.columns:
                        opt_players["gpp_score"] = (
                            opt_players["ceiling"] * 0.60
                            + opt_players["proj_pts_dk"] * 0.25
                            + (1 - opt_players["proj_own"] / 100) * 10
                        ).round(3)
                    # Remove excluded players from pool NOW — before game theory,
                    # adversarial ownership, and any other downstream step.
                    # Previously they were only excluded at generate_gpp_lineups(),
                    # which meant game theory flagged SCRATCHED players as chalk
                    # (e.g. Siakam "OUT" still appeared as 30.5% chalk target).
                    if ni_excluded:
                        before = len(opt_players)
                        opt_players = opt_players[
                            ~opt_players["player_id"].astype(str).isin(ni_excluded)
                        ].copy()
                        print(f"[News] {len(ni_excluded)} players excluded by news signals "
                              f"({before - len(opt_players)} removed from pool)")

                    # Apply sub-$5K bench filter: remove players priced < $5,000
                    # who have no confirmed role signal AND no FC mins >= 20.
                    # Mirrors the same filter in test_slate.py main().
                    _role_signal_pids = {
                        pid for pid, imp in ni_impacts.items()
                        if imp.get("signal_type", imp.get("signal", ""))
                        in ("STARTING_REPLACEMENT", "USAGE_INCREASE", "CLEARED_FULLY")
                    }
                    if "salary" in opt_players.columns:
                        _bench_mask   = opt_players["salary"] < 5000
                        _no_signal    = ~opt_players["player_id"].astype(str).isin(_role_signal_pids)
                        _fc_mins_col  = opt_players.get("fc_mins") if hasattr(opt_players, "get") \
                                        else opt_players["fc_mins"] if "fc_mins" in opt_players.columns \
                                        else None
                        if _fc_mins_col is not None:
                            _no_fc_mins = ~(opt_players["fc_mins"].notna() & (opt_players["fc_mins"] >= 20))
                        else:
                            _no_fc_mins = pd.Series(True, index=opt_players.index)
                        _drop = _bench_mask & _no_signal & _no_fc_mins
                        if _drop.any():
                            opt_players = opt_players[~_drop].copy()
                            print(f"[News] Bench filter removed {_drop.sum()} sub-$5K players "
                                  f"with no confirmed role")

                # Emit SSE event
                q.put({"type": "news_intel", "data": {
                    "headline":        ni_summary.get("headline", ""),
                    "total_signals":   ni_summary.get("total_signals", 0),
                    "excluded_count":  ni_summary.get("excluded_count", 0),
                    "high_priority":   ni_summary.get("high_priority", [])[:8],
                    "signal_counts":   ni_summary.get("signal_counts", {}),
                    "x_stats":         ni_result.get("x_stats", {}),
                    "signals": [
                        {
                            "player_name": s["player_name"],
                            "signal_type": s["signal_type"],
                            "source":      s["source"],
                            "age_hrs":     s["age_hrs"],
                            "confidence":  s["confidence"],
                            "text":        s["text"][:120],
                        }
                        for s in ni_signals[:20]
                    ],
                }})
            except Exception as _nie:
                import logging as _nilog
                _nilog.warning("[News] News intel enrichment skipped: %s", _nie)

            # ── Game Theory enrichment ────────────────────────────────────────
            # Runs after all projection adjustments (injuries, B2B, DvP, props)
            # so it operates on final projections and ownership estimates.
            try:
                gt = _get_game_theory_agent()
                _ct = contest_type.lower() if contest_type else "gpp"

                # GT agent expects different column names — build a view
                gt_in = opt_players.rename(columns={
                    "proj_pts_dk":  "projected_pts_dk",
                    "proj_own":     "proj_ownership",
                    "status":       "injury_status",
                })

                # 1. Field composition: injury-risk-discount ownership estimates
                gt_in = gt.model_field_composition(gt_in, contest_size=contest_size or 5000)

                # 2. Player-level leverage scores
                gt_in = gt.compute_leverage_scores(gt_in, strategy=_ct)

                # 3. Per-player max exposure based on proj tier + ownership
                gt_in = gt.compute_optimal_exposures(gt_in, n_lineups=num_lineups, contest_type=_ct)

                # Write enriched columns back to opt_players
                opt_players = opt_players.copy()
                opt_players["field_own_model"]  = gt_in["field_ownership_model"].values
                opt_players["leverage_vs_field"]= gt_in["leverage_vs_field"].values
                opt_players["gt_leverage_score"]= gt_in["leverage_score"].values
                opt_players["is_leverage_play"] = gt_in["is_leverage_play"].values
                opt_players["is_chalk"]         = gt_in["is_chalk"].values
                opt_players["gt_max_exposure"]  = gt_in["max_lineup_count"].values

                # Blend GT field model into proj_own:
                # 65% formula (calibrated from backtest) + 35% GT field model
                # (injury-risk-discounted, salary-gravity adjusted).
                # This is the final ownership used by the ILP objective.
                opt_players["proj_own"] = (
                    0.65 * opt_players["proj_own"]
                    + 0.35 * opt_players["field_own_model"]
                ).clip(1, 40).round(1)

                # Recompute gpp_score with blended ownership
                opt_players["gpp_score"] = (
                    opt_players["ceiling"] * 0.60
                    + opt_players["proj_pts_dk"] * 0.25
                    + (1 - opt_players["proj_own"] / 100) * 10
                ).round(3)

                # Emit top leverage plays and chalk flags to the UI log
                top_lev = gt_in[gt_in["is_leverage_play"] == True].nlargest(5, "leverage_score")
                chalk   = gt_in[gt_in["is_chalk"] == True].nlargest(5, "leverage_score")
                lev_names   = ", ".join(top_lev["name"].tolist()) if len(top_lev) else "none"
                chalk_names = ", ".join(chalk["name"].tolist())   if len(chalk)   else "none"
                print(f"[GT] Contest strategy: {_ct.upper()} | field size: {contest_size or '?'}")
                print(f"[GT] Top leverage plays (low-own / over-proj vs salary): {lev_names}")
                print(f"[GT] Chalk flags (>30% ownership): {chalk_names}")
                print(f"[GT] Per-player exposure caps applied ({len(gt_in)} players)")

                q.put({"type": "gt_analysis", "data": {
                    "contest_type":    _ct,
                    "leverage_plays":  top_lev[["name", "team", "leverage_score",
                                                "proj_ownership", "projected_pts_dk"]]
                                            .round({"leverage_score": 2, "proj_ownership": 1,
                                                    "projected_pts_dk": 1})
                                            .to_dict("records"),
                    "chalk_flags":     chalk[["name", "team", "proj_ownership"]]
                                            .round({"proj_ownership": 1})
                                            .to_dict("records"),
                }})
            except Exception as _gte:
                import logging as _gtlog
                _gtlog.warning("[GT] Game theory enrichment skipped: %s", _gte)

            # ── Field Behavior modeling ───────────────────────────────────────
            # Models what the aggregate DFS public will build tonight:
            # chalk core, chalk traps, over-concentrated stacks,
            # differentiation targets, and construction tips.
            try:
                _fb = _get_field_behavior_agent()
                _gt_data = _state.get("game_totals") or GAME_TOTALS
                field_profile = _fb.model(
                    opt_players,
                    _gt_data,
                    contest_size=contest_size or 5000,
                    n_lineups=num_lineups,
                )
                _state["field_profile"] = field_profile

                # Apply chalk-stack correlation boosts to proj_own:
                # Chalk pairs that co-appear in the field get bumped up so
                # the ILP treats them as more "expensive" to own together.
                for boost in field_profile.get("own_correlation_boosts", []):
                    for _pname in boost["players"]:
                        _pmask = opt_players["name"] == _pname
                        if _pmask.any():
                            opt_players.loc[_pmask, "proj_own"] = (
                                opt_players.loc[_pmask, "proj_own"] + boost["joint_own"] * 0.15
                            ).clip(1, 40).round(1)

                # Recompute gpp_score after chalk-stack own adjustments
                opt_players["gpp_score"] = (
                    opt_players["ceiling"] * 0.60
                    + opt_players["proj_pts_dk"] * 0.25
                    + (1 - opt_players["proj_own"] / 100) * 10
                ).round(3)

                # Log key findings
                traps = field_profile.get("chalk_traps", [])
                diffs = [t for t in field_profile.get("differentiation_targets", [])
                         if t["type"] == "player"]
                tips  = [t for t in field_profile.get("construction_tips", [])
                         if t["priority"] == "high"]

                if traps:
                    trap_str = ", ".join(
                        f"{t['name']} ({t['proj_own']:.0f}%own {t['upside_ratio']:.2f}x ceil)"
                        for t in traps[:3]
                    )
                    print(f"[Field] CHALK TRAPS: {trap_str}")
                if diffs:
                    diff_str = ", ".join(
                        f"{d['name']} ({d['proj_own']:.0f}%own {d['proj_pts_dk']:.0f}pts)"
                        for d in diffs[:3]
                    )
                    print(f"[Field] DIFF TARGETS: {diff_str}")

                q.put({"type": "field_behavior", "data": {
                    "chalk_traps":    traps[:5],
                    "diff_targets":   field_profile.get("differentiation_targets", [])[:5],
                    "construction_tips": field_profile.get("construction_tips", [])[:6],
                    "field_archetype_desc": field_profile.get("field_archetype", {}).get("description", ""),
                    "over_concentrated_games": [
                        s["matchup"] for s in field_profile.get("stack_analysis", [])
                        if s.get("over_concentrated")
                    ],
                }})
            except Exception as _fbe:
                import logging as _fblog
                _fblog.warning("[Field] Field behavior modeling skipped: %s", _fbe)

            # ── Adversarial Ownership Distribution Analysis ────────────────────
            # Identifies salary voids, positional cliffs, adversarial slot
            # targets, and applies gpp_score boosts to void/post-cliff players.
            try:
                _adv = _get_adversarial_ownership_agent()
                adv_profile = _adv.analyze(
                    opt_players,
                    n_lineups=num_lineups,
                    contest_size=5000,
                )
                _state["adversarial_profile"] = adv_profile

                # Apply gpp_score boosts to void-zone / post-cliff players
                boosts: dict = adv_profile.get("gpp_score_boosts", {})
                if boosts:
                    for _pid_str, _bonus in boosts.items():
                        _mask = opt_players["player_id"].astype(str) == _pid_str
                        if _mask.any():
                            opt_players.loc[_mask, "gpp_score"] = (
                                opt_players.loc[_mask, "gpp_score"] + _bonus
                            ).round(3)
                    print(f"[Adv] Applied gpp_score boosts to {len(boosts)} players")

                adv_sum = adv_profile.get("summary", {})
                q.put({"type": "adversarial_analysis", "data": {
                    "headline":            adv_sum.get("headline", ""),
                    "overlap_estimate":    adv_sum.get("overlap_estimate", 0),
                    "overlap_label":       adv_sum.get("overlap_label", ""),
                    "overlap_advice":      adv_sum.get("overlap_advice", ""),
                    "void_count":          adv_sum.get("void_count", 0),
                    "chalk_cluster_count": adv_sum.get("chalk_cluster_count", 0),
                    "cliff_positions":     adv_sum.get("cliff_positions", []),
                    "total_adv_savings":   adv_sum.get("total_adv_savings", 0),
                    "adversarial_slots":   adv_profile.get("adversarial_slots", [])[:5],
                    "void_brackets":       [
                        {"label": b["label"], "max_own": b["max_own"], "n_viable": b["n_viable"]}
                        for b in adv_profile.get("void_brackets", [])[:6]
                    ],
                    "positional_cliffs": {
                        pos: {
                            "cliff_magnitude": cd["cliff_magnitude"],
                            "interpretation":  cd["interpretation"],
                        }
                        for pos, cd in adv_profile.get("positional_cliffs", {}).items()
                    },
                    "boost_count": len(boosts),
                }})
            except Exception as _adve:
                import logging as _advlog
                _advlog.warning("[Adv] Adversarial ownership analysis skipped: %s", _adve)

            _pool_size = int(pool_size) if pool_size and int(pool_size) > num_lineups else None
            # News-intel exclusions already removed from opt_players above.
            # Only pass user-specified exclusions here.
            _state.pop("_ni_excluded", None)   # clean up stale state if any
            _effective_excl = list(excl_list) if excl_list else None
            lineups = generate_gpp_lineups(
                opt_players,
                n=_pool_size or num_lineups,
                locked_ids=lock_list,
                excluded_ids=_effective_excl,
                penalty_schedule=pen_sched,
                slate_profile=slate_profile,
                pool_size=_pool_size,
            )
        except Exception as _exc:
            import traceback, logging as _log
            _log.error("[run] Optimizer error:\n%s", traceback.format_exc())
            q.put({"type": "error", "msg": str(_exc)})
            _state["job_running"] = False
            return
        finally:
            sys.stdout = old_stdout

        # Tag lineups with chalk/leverage batch label
        if split:
            n_chalk = split["n_chalk"]
            for i, lu in enumerate(lineups):
                lu["lineup_type"] = "chalk" if i < n_chalk else "leverage"
        else:
            for lu in lineups:
                lu.setdefault("lineup_type", "standard")

        _state["lineups"] = lineups

        # Salary tier construction report (printed to optimizer log)
        # Target from 7-day backtest: pros avg $8K+=1.57, $5K-$7K=2.50, <$5K=1.50
        _pid_idx = opt_players.set_index("player_id") if "player_id" in opt_players.columns else None
        if _pid_idx is not None and not _pid_idx.empty and "salary" in _pid_idx.columns:
            _t_prem, _t_mid, _t_chp = [], [], []
            for _lu in lineups:
                _pids = [str(p) for p in _lu.get("player_ids", [])]
                _pids_ok = [p for p in _pids if p in _pid_idx.index]
                _sals = [int(_pid_idx.loc[p, "salary"]) for p in _pids_ok]
                _t_prem.append(sum(1 for s in _sals if s >= 8000))
                _t_mid.append(sum(1 for s in _sals if 5000 <= s <= 7000))
                _t_chp.append(sum(1 for s in _sals if s < 5000))
            _n = len(lineups)
            _ap, _am, _ac = sum(_t_prem)/_n, sum(_t_mid)/_n, sum(_t_chp)/_n
            print(f"\nSALARY CONSTRUCTION CHECK:")
            print(f"  $8K+ studs:    {_ap:.2f} avg  (target ~1.57) {'OK' if _ap <= 2.0 else 'HIGH'}")
            print(f"  $5K-$7K value: {_am:.2f} avg  (target ~2.50) {'OK' if _am >= 2.0 else 'LOW - INVESTIGATE'}")
            print(f"  <$5K cheap:    {_ac:.2f} avg  (target ~1.50) {'OK' if _ac <= 2.0 else 'HIGH'}")

        # Export DK CSV
        today = date.today().strftime("%Y-%m-%d")
        OUTPUT_DIR.mkdir(exist_ok=True)
        dk_path = OUTPUT_DIR / f"dk_upload_{today}.csv"
        export_dk_csv(lineups, _state["players"], dk_path)
        _state["dk_path"] = dk_path

        q.put({"type": "done", "count": len(lineups)})
        _state["job_running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@app.get("/api/stream")
async def stream():
    """SSE endpoint — yields optimization log messages until done/error."""
    async def _gen():
        yield 'data: {"type":"connected"}\n\n'
        q: Queue | None = _state.get("job_queue")
        while True:
            if q:
                try:
                    msg = q.get_nowait()
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("type") in ("done", "error"):
                        break
                except Empty:
                    pass
            await asyncio.sleep(0.1)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/lineups")
async def get_lineups():
    lineups = _state.get("lineups", [])
    players = _state.get("players")
    enriched = []
    if players is not None:
        pid_map = players.set_index("player_id").to_dict(orient="index")
        for lu in lineups:
            slots = lu.get("slot_assignment", {})
            roster = []
            for slot in _SLOT_ORDER:
                pid = str(slots.get(slot, ""))
                info = pid_map.get(pid, {})
                roster.append({
                    "slot":   slot,
                    "pid":    pid,
                    "name":   info.get("name", pid),
                    "team":   info.get("team", ""),
                    "pos":    info.get("primary_position", ""),
                    "salary": info.get("salary", 0),
                    "proj":   round(info.get("proj_pts_dk", 0), 1),
                    "own":    round(info.get("proj_own", 0), 1),
                })
            # Include leverage + stack metadata computed during generation / late-swap
            enriched.append({
                **lu,
                "roster":       roster,
                "leverage":     lu.get("leverage", 0),
                "avg_own":      lu.get("avg_own", 0),
                "chalk_ct":     lu.get("chalk_ct", 0),
                "low_own_ct":   lu.get("low_own_ct", 0),
                "has_game_stack": lu.get("has_game_stack", False),
                "stack_game":   lu.get("stack_game", ""),
                "swap_method":  lu.get("swap_method", ""),
            })
    return {"lineups": enriched}


@app.get("/api/field-behavior")
async def get_field_behavior():
    """Return the latest FieldBehaviorAgent profile (chalk traps, diff targets, tips)."""
    profile = _state.get("field_profile")
    if profile is None:
        raise HTTPException(status_code=404, detail="Run the optimizer first to generate field behavior analysis.")
    return profile


@app.get("/api/news-intel")
async def get_news_intel():
    """Return latest NewsIntelAgent signals and impacts."""
    profile = _state.get("news_intel")
    if profile is None:
        raise HTTPException(status_code=404, detail="Run the optimizer first to generate news intel.")
    return {
        "summary":     profile.get("summary", {}),
        "signals":     profile.get("signals", []),
        "player_news": profile.get("player_news", {}),
    }


@app.get("/api/beat-writers")
async def get_beat_writers():
    """
    Return configured beat writer handles and any signals they fired
    during the last optimizer run.
    """
    from nba_dfs.core.config import X_BEAT_WRITER_HANDLES, X_BEARER_TOKEN
    profile    = _state.get("news_intel") or {}
    signals    = profile.get("signals", [])
    bw_signals = [s for s in signals if s.get("source") == "beat_writer"]
    x_stats    = profile.get("x_stats", {})
    return {
        "configured_handles":  X_BEAT_WRITER_HANDLES,
        "x_api_configured":    bool(X_BEARER_TOKEN),
        "beat_writer_signals": bw_signals,
        "count":               len(bw_signals),
        "fetch_stats":         x_stats,
    }


@app.get("/api/adversarial-ownership")
async def get_adversarial_ownership():
    """Return the latest AdversarialOwnershipAgent profile (voids, cliffs, slot targets)."""
    profile = _state.get("adversarial_profile")
    if profile is None:
        raise HTTPException(status_code=404, detail="Run the optimizer first to generate adversarial ownership analysis.")
    return profile


@app.get("/api/positional-scarcity")
async def get_positional_scarcity():
    """Return viable player counts per DK slot and scarce-position warnings."""
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")
    scarcity = _state.get("positional_scarcity") or compute_positional_scarcity(_state["players"])
    _state["positional_scarcity"] = scarcity
    return scarcity


@app.get("/api/ref-crews")
async def get_ref_crews():
    """Return tonight's referee crew assignments and their foul-rate tier."""
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")
    try:
        ra      = _get_ref_agent()
        summary = ra.get_crew_summary()
        return {"crews": summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/download/dk")
async def download_dk():
    path = _state.get("dk_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="No DK CSV generated yet.")
    return FileResponse(str(path), filename=Path(path).name, media_type="text/csv")


@app.get("/api/slate-profile")
async def get_slate_profile():
    """
    Analyze tonight's slate environment with the SlateConstructionAgent and
    return recommended lineup construction parameters.

    The result is cached in _state["slate_profile"] and reused automatically
    during /api/run. Call this endpoint before running lineups to preview the
    AI construction recommendation and its rationale.
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")

    try:
        players = _state["players"]
        profile = _run_slate_analysis(players)
        _state["slate_profile"] = profile
        return {
            "profile":           profile,
            "source":            profile.get("source", "unknown"),
            "rationale":         profile.get("rationale", ""),
            "key_opportunities": profile.get("key_opportunities", []),
            "key_risks":         profile.get("key_risks", []),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Slate analysis error: {exc}") from exc


@app.get("/api/b2b-check")
async def b2b_check():
    """
    Detect which teams are on B2B second night tonight using ESPN yesterday
    scoreboard. Applies three adjustments to the player pool:

      1. Direct B2B penalty: archetype/salary-scaled projection haircut
      2. Usage redistribution: backup at same position absorbs star's lost usage
      3. Opponent boost: fresh teams facing B2B defense get +2.5% projection boost

    Returns per-player changes so the UI can highlight B2B-affected players.
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")

    today_matchups = list((_state.get("game_totals") or GAME_TOTALS).keys())
    fatigue_map    = fetch_b2b_teams(today_matchups)   # dict with b2b/two_in_three/three_in_four

    players_before = _state["players"].copy()
    updated        = apply_b2b_adjustments(_state["players"], fatigue_map)
    updated        = refresh_ownership(updated)   # B2B fade + salary-implied gainer bump
    _state["players"] = updated

    _b2b_ct   = int(updated["fatigue_tier"].eq("b2b").sum())   if "fatigue_tier" in updated.columns else int(updated["is_b2b"].sum())
    _2in3_ct  = int(updated["fatigue_tier"].eq("2in3").sum())  if "fatigue_tier" in updated.columns else 0
    _3in4_ct  = int(updated["fatigue_tier"].eq("3in4").sum())  if "fatigue_tier" in updated.columns else 0

    import logging as _b2blog
    if fatigue_map.get("three_in_four"):
        _b2blog.info("B2B: 3-in-4 teams (heaviest penalty): %s", sorted(fatigue_map["three_in_four"]))
    if fatigue_map.get("b2b"):
        _b2blog.info("B2B: B2B teams: %s — %d players adjusted", sorted(fatigue_map["b2b"]), _b2b_ct)
    if fatigue_map.get("two_in_three"):
        _b2blog.info("B2B: 2-in-3 teams (mild): %s — %d high/mid salary players adjusted", sorted(fatigue_map["two_in_three"]), _2in3_ct)

    # Build diff: players whose projection changed
    changes = []
    for _, row in updated.iterrows():
        before = players_before[players_before["player_id"] == row["player_id"]]
        if before.empty:
            continue
        old_proj = float(before.iloc[0]["proj_pts_dk"])
        new_proj = float(row["proj_pts_dk"])
        if abs(new_proj - old_proj) >= 0.3:
            changes.append({
                "name":        row["name"],
                "team":        row["team"],
                "archetype":   row.get("archetype", ""),
                "salary":      int(row["salary"]),
                "fatigue_tier": row.get("fatigue_tier", ""),
                "is_b2b":      bool(row.get("is_b2b", False)),
                "b2b_penalty": round(float(row.get("b2b_penalty", 0)) * 100, 1),
                "proj_before": round(old_proj, 1),
                "proj_after":  round(new_proj, 1),
                "proj_delta":  round(new_proj - old_proj, 1),
            })

    # Also apply base DvP weights (seasonal ratings)
    dvp_weights = build_dvp_weights(updated)
    dvp_updated = apply_dvp_adjustments(updated, dvp_weights)
    _state["players"] = dvp_updated

    # Serialize updated pool for UI refresh
    pool_cols = ["player_id", "name", "team", "archetype", "salary",
                 "proj_pts_dk", "ceiling", "gpp_score", "is_b2b", "b2b_penalty", "b2b_boost", "dvp_mult"]
    pool_cols = [c for c in pool_cols if c in dvp_updated.columns]

    return {
        "b2b_teams":        sorted(fatigue_map.get("b2b", [])),
        "two_in_three":     sorted(fatigue_map.get("two_in_three", [])),
        "three_in_four":    sorted(fatigue_map.get("three_in_four", [])),
        "b2b_player_ct":    _b2b_ct,
        "two_in_three_ct":  _2in3_ct,
        "three_in_four_ct": _3in4_ct,
        "changes":          sorted(changes, key=lambda x: abs(x["proj_delta"]), reverse=True)[:40],
        "updated_players":  dvp_updated[pool_cols].to_dict(orient="records"),
    }


@app.get("/api/injury-check")
async def injury_check():
    """
    Fetch live ESPN injury data, cross-reference against the current player pool
    and generated lineups. After applying status updates, triggers a dynamic
    DvP recalculation: confirmed OUT players shift the defensive archetype
    matchup for opposing players (e.g., rim protector OUT → opposing rim runners
    get a projection boost).
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")

    status_map = scrape_espn_injuries()

    # Build breakdown summary before applying updates
    _haircut_pcts = {"DOUBTFUL": 50, "GTD": 30, "QUESTIONABLE": 15}
    _status_summary: dict[str, list[str]] = {"OUT": [], "DOUBTFUL": [], "GTD": [], "QUESTIONABLE": []}
    for _n, _s in status_map.items():
        _su = _s.upper()
        if _su in _status_summary:
            _status_summary[_su].append(_n)

    # ── Identify OUT players BEFORE apply_status_updates drops them from pool ──
    # apply_status_updates removes OUT players from the DataFrame entirely,
    # so out_ids and confirmed_out_dvp must be built from the snapshot first.
    _state.setdefault("confirmed_out_ids", set())
    _snap = _state.get("_full_pool_snapshot")
    full_pool_before = _snap if _snap is not None else _state["players"]
    pool_name_lower  = {str(n).lower(): i for i, n in
                        full_pool_before["name"].items()}

    from difflib import get_close_matches as _gcm
    out_ids:          set  = set()
    confirmed_out_dvp: list = []

    for raw_name, status in status_map.items():
        if status.upper() != "OUT":
            continue
        key   = raw_name.lower()
        match = pool_name_lower.get(key)
        if match is None:
            close = _gcm(key, pool_name_lower.keys(), n=1, cutoff=0.82)
            if not close:
                continue
            match = pool_name_lower[close[0]]
        row = full_pool_before.iloc[match]
        pid = str(row["player_id"])
        out_ids.add(pid)
        confirmed_out_dvp.append({
            "name":      str(row["name"]),
            "team":      str(row.get("team", "")),
            "archetype": str(row.get("archetype", "COMBO_G")),
            "salary":    int(row.get("salary", 0)),
        })

    # Accumulate into state so optimizer always excludes confirmed OUTs
    _state["confirmed_out_ids"] |= out_ids

    # Apply ESPN status updates (OUT removes from pool, others get haircuts)
    updated_players = apply_status_updates(_state["players"], status_map)
    _state["players"] = updated_players

    if confirmed_out_dvp:
        updated_players = apply_lineup_confirmation_dvp(updated_players, confirmed_out_dvp)
        _state["players"] = updated_players

    # Find affected lineups
    affected = []
    for lu in _state["lineups"]:
        scratched = [p for p in lu["player_ids"] if str(p) in out_ids]
        if scratched:
            affected.append({"lineup_num": lu["lineup_num"], "scratched_ids": scratched})

    status_records = updated_players[["player_id", "name", "team", "status",
                                       "proj_pts_dk", "archetype"]].to_dict(orient="records") \
        if "archetype" in updated_players.columns \
        else updated_players[["player_id", "name", "team", "status"]].to_dict(orient="records")

    changes = [{"name": n, "status": s} for n, s in status_map.items()]

    # DvP shift summary: which opposing players were boosted by confirmed OUTs
    dvp_shifts = []
    for absent in confirmed_out_dvp:
        from nba_dfs.test_slate import ARCHETYPE_DEFENSIVE_IMPACT
        arch_boosts = ARCHETYPE_DEFENSIVE_IMPACT.get(absent["archetype"], {})
        sal_scale   = min(1.0, max(0.1, (absent["salary"] - 3000) / 7000))
        for ben_arch, base_boost in arch_boosts.items():
            boost_pct = round(base_boost * sal_scale * 100, 1)
            if boost_pct >= 2.0:
                dvp_shifts.append({
                    "absent_name":     absent["name"],
                    "absent_team":     absent["team"],
                    "absent_archetype": absent["archetype"],
                    "beneficiary_arch": ARCHETYPE_LABELS.get(ben_arch, ben_arch),
                    "boost_pct":        boost_pct,
                    "note": f"Opposing {ARCHETYPE_LABELS.get(ben_arch, ben_arch)} players +"
                            f"{boost_pct}% proj (vs {absent['name']}'s team)",
                })

    # ── Usage Impact: who benefits when confirmed OUTs are absent ────────────
    usage_impact:  list = []
    on_off_impact: list = []
    out_name_list = [p["name"] for p in confirmed_out_dvp]
    if out_name_list and _state["players"] is not None:
        try:
            import logging as _log
            usage_data = fetch_player_usage_rates()
            # players_df = current pool (OUT players already dropped by apply_status_updates)
            # full_pool_before = snapshot with OUT players still present (needed to look up their rows)
            players_df = _state["players"]

            # Group OUT players by team using the SNAPSHOT (OUT players still present there)
            # then run OnOffAgent for every teammate on every affected team — no shortcuts.
            # Results are cached 24h so the wait only happens once per day per combination.
            on_off_map: dict[str, dict] = {}   # teammate player_id → split data
            try:
                from collections import defaultdict as _dd
                oo_agent    = _get_on_off_agent()
                out_by_team: dict = _dd(list)

                for out_info in confirmed_out_dvp:
                    out_name = out_info["name"]
                    mask = full_pool_before["name"].str.lower() == out_name.lower()
                    if mask.any():
                        out_by_team[full_pool_before[mask].iloc[0]["team"]].append(
                            full_pool_before[mask].iloc[0]
                        )

                n_teams = len(out_by_team)
                _log.info("[injuries] OnOffAgent: querying %d affected teams", n_teams)

                for team, out_rows in out_by_team.items():
                    team_mask     = players_df["team"] == team
                    teammate_rows = players_df[team_mask]
                    if teammate_rows.empty:
                        continue

                    splits = oo_agent.compute(out_rows, teammate_rows)
                    if splits:
                        on_off_map.update(splits)
                        _log.info(
                            "[injuries] on/off splits: %s OUT on %s → %d teammate deltas",
                            "+".join(r["name"] for r in out_rows), team, len(splits),
                        )

                _log.info(
                    "[injuries] OnOffAgent complete: %d total teammate splits computed",
                    len(on_off_map),
                )

            except Exception as _ooe:
                _log.warning("[injuries] OnOffAgent failed: %s — using usage fallback", _ooe)

            # Apply usage absorption for each OUT player.
            # OUT player rows come from the SNAPSHOT; boosts apply to the current pool.
            adjusted_pool = players_df.copy()
            for out_info in confirmed_out_dvp:
                out_name = out_info["name"]
                snap_mask = full_pool_before["name"].str.lower() == out_name.lower()
                if not snap_mask.any():
                    continue
                out_row = full_pool_before[snap_mask].iloc[0]
                adjusted_pool = estimate_usage_absorption(
                    out_row, adjusted_pool,
                    usage_data=usage_data,
                    on_off_data=on_off_map if on_off_map else None,
                )

            # Persist adjusted projections and on_off_map for lineup generation
            if on_off_map:
                _state["on_off_map"] = on_off_map
            adjusted_pool     = refresh_ownership(adjusted_pool)   # usage-gainer bump
            _state["players"] = adjusted_pool

            # Compute display impact summary using snapshot (has OUT player rows for comparison)
            usage_impact = compute_lineup_usage_impact(
                out_name_list, full_pool_before,
                usage_data=usage_data,
                on_off_map=on_off_map,
            )

            # Build on/off opportunity board.
            # Primary source: lineup-level splits from PlayerDashboardByLineups.
            # Fallback source: usage redistribution estimates when no lineup data.
            _oo_rows = []
            if on_off_map:
                pid_index = adjusted_pool.set_index(
                    adjusted_pool["player_id"].astype(str)
                )
                for tm_pid, split in on_off_map.items():
                    try:
                        row = pid_index.loc[str(tm_pid)]
                    except KeyError:
                        continue
                    _oo_rows.append({
                        "name":        str(row.get("name", tm_pid)),
                        "team":        str(row.get("team", "")),
                        "position":    str(row.get("primary_position", "")),
                        "salary":      int(row.get("salary", 0)),
                        "source":      "lineup",
                        "delta_dk":    round(float(split.get("delta_dk", 0)), 2),
                        "dk_with":     round(float(split.get("dk_with", 0)), 2),
                        "dk_without":  round(float(split.get("dk_without", 0)), 2),
                        "min_with":    round(float(split.get("min_with", 0)), 0),
                        "min_without": round(float(split.get("min_without", 0)), 0),
                    })

            if not _oo_rows and usage_impact:
                # No lineup data available — fall back to usage redistribution estimates
                for ui in usage_impact:
                    _oo_rows.append({
                        "name":        ui.get("name", ""),
                        "team":        ui.get("team", ""),
                        "position":    ui.get("position", ""),
                        "salary":      0,
                        "source":      "usage_est",
                        "delta_dk":    ui.get("delta_dk", 0),
                        "dk_with":     ui.get("proj_pts_before", 0),
                        "dk_without":  ui.get("proj_pts_after", 0),
                        "min_with":    0,
                        "min_without": 0,
                    })

            _oo_rows.sort(key=lambda x: abs(x["delta_dk"]), reverse=True)
            on_off_impact = _oo_rows[:15]

        except Exception as _ue:
            import logging as _log
            _log.warning("[injuries] usage impact computation failed: %s", _ue)

    # ── Signal health report — tells frontend exactly what fired ─────────────
    on_off_source = "lineup" if any(r.get("source") == "lineup" for r in on_off_impact) \
                    else "usage_est" if on_off_impact else "none"
    signal_health = {
        "espn_injuries":    bool(status_map),
        "confirmed_outs":   len(confirmed_out_dvp),
        "dvp_applied":      bool(dvp_shifts),
        "usage_absorption": bool(usage_impact),
        "on_off_source":    on_off_source,   # "lineup" | "usage_est" | "none"
        "on_off_count":     len([r for r in on_off_impact if r.get("source") == "lineup"]),
    }
    # Store for optimizer pre-run check
    _state["signal_health"] = signal_health

    return {
        "espn_updates":     changes,
        "player_statuses":  status_records,
        "affected_lineups": affected,
        "out_ids":          list(out_ids),
        "out_names":        [p["name"] for p in confirmed_out_dvp],
        "dvp_shifts":       dvp_shifts,
        "usage_impact":     usage_impact,
        "on_off_impact":    on_off_impact,
        "signal_health":    signal_health,
        "status_summary": {
            "removed":          _status_summary["OUT"],
            "doubtful":         _status_summary["DOUBTFUL"],
            "gtd":              _status_summary["GTD"],
            "questionable":     _status_summary["QUESTIONABLE"],
            "dvp_boosts":       len(dvp_shifts),
            "usage_beneficiaries": len(usage_impact),
        },
    }


@app.get("/api/test-onoff")
async def test_onoff(name: str = "LeBron James", team: str = "LAL"):
    """
    Diagnostic endpoint — tests the ESPN on/off pipeline for a single OUT player.
    Usage: /api/test-onoff?name=Jarred+Vanderbilt&team=LAL
    Steps tested:
      1. ESPN roster fetch for team (name -> ESPN ID mapping)
      2. OUT player ID resolution
      3. Team schedule fetch (all completed game dates)
      4. Player game log fetch → missed dates = schedule - played dates
      5. Per-36 DK split for each teammate meeting thresholds
    """
    import pandas as pd
    from nba_dfs.agents.bbref_on_off_agent import BBRefOnOffAgent, ESPN_TEAM_IDS

    agent = BBRefOnOffAgent()

    espn_team_id = ESPN_TEAM_IDS.get(team.upper())
    if espn_team_id is None:
        return {"status": "UNKNOWN_TEAM", "team": team, "known_teams": list(ESPN_TEAM_IDS.keys())}

    # Step 1: roster
    roster = agent._get_roster(espn_team_id, team)
    if not roster:
        return {
            "status":      "ROSTER_FETCH_FAILED",
            "team_dk":     team,
            "espn_team_id": espn_team_id,
            "hint":        "Check ESPN API connectivity",
        }

    # Step 2: resolve OUT player ID
    espn_id = agent._resolve_id(name, roster)
    if espn_id is None:
        from difflib import get_close_matches
        close = get_close_matches(name.lower(), roster.keys(), n=5, cutoff=0.5)
        return {
            "status":            "PLAYER_NOT_IN_ROSTER",
            "name_searched":     name,
            "team":              team,
            "roster_size":       len(roster),
            "closest_in_roster": close,
        }

    # Step 3: schedule + missed dates
    schedule = agent._get_schedule(espn_team_id, team)
    played_dates = agent._get_played_dates(espn_id)
    missed = schedule - played_dates

    # Step 4: game log info
    gl = agent._get_game_log(espn_id)
    games_played = len(gl) if gl is not None else 0

    # Step 5: run compute() against teammates currently in pool
    players_df = _state.get("players")
    splits_computed = 0
    sample_splits = []
    if players_df is not None:
        mask = players_df["team"] == team.upper()
        teammates = players_df[mask]
        out_series = pd.Series({"name": name, "team": team.upper(), "player_id": "0"})
        splits = agent.compute([out_series], teammates)
        splits_computed = len(splits)
        sample_splits = [
            {"player_id": pid, **v}
            for pid, v in list(splits.items())[:5]
        ]

    return {
        "status":              "OK",
        "source":              "ESPN",
        "name":                name,
        "espn_id":             espn_id,
        "team":                team,
        "roster_size":         len(roster),
        "team_games_played":   len(schedule),
        "player_games_played": games_played,
        "missed_dates_count":  len(missed),
        "missed_dates_sample": sorted(missed)[:5],
        "splits_computed":     splits_computed,
        "sample_splits":       sample_splits,
        "thresholds":          {"MIN_GAMES_WITHOUT": 5, "MIN_GAMES_WITH": 10},
    }


@app.get("/api/correlation")
async def get_correlation():
    """
    Build and return the player correlation matrix for the current pool.

    Each entry describes how correlated two players' DFS outcomes are:
      +1.0  = perfectly correlated (stack them — they score together)
       0.0  = independent
      -1.0  = perfectly inverse (usage competition — avoid pairing)

    Top positive pairs = recommended same-lineup stacks.
    Top negative pairs = players who fight for the same usage (avoid doubling up).
    On/off impact: when Player A is OUT, players negatively correlated with A
    are usage beneficiaries — these are the best late-swap targets.
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")

    players  = _state["players"]
    corr_map = build_player_correlation(players)

    pid_to_name = dict(zip(players["player_id"].astype(str), players["name"]))
    pid_to_team = dict(zip(players["player_id"].astype(str), players["team"]))

    # Top 30 positive pairs (best stacks)
    positive = sorted(
        [(k, v) for k, v in corr_map.items() if k[0] < k[1] and v > 0.15],
        key=lambda x: -x[1],
    )[:30]

    # Top 20 negative pairs (usage conflicts — don't double-stack)
    negative = sorted(
        [(k, v) for k, v in corr_map.items() if k[0] < k[1] and v < -0.10],
        key=lambda x: x[1],
    )[:20]

    def pair_to_dict(k, v):
        return {
            "pid1":  k[0], "name1": pid_to_name.get(k[0], k[0]), "team1": pid_to_team.get(k[0], ""),
            "pid2":  k[1], "name2": pid_to_name.get(k[1], k[1]), "team2": pid_to_team.get(k[1], ""),
            "corr":  v,
        }

    return {
        "positive_stacks": [pair_to_dict(k, v) for k, v in positive],
        "negative_pairs":  [pair_to_dict(k, v) for k, v in negative],
        "note": (
            "Positive correlation = stack these players (correlated upside). "
            "Negative correlation = usage competition — avoid in same lineup. "
            "When a player is OUT, players with negative correlation to them "
            "are the strongest late-swap targets (they absorb usage)."
        ),
    }


@app.get("/api/matchup-analysis")
async def get_matchup_analysis():
    """
    Compute true DvP from NBA game logs, fetch pace, grade each game A–F.

    Steps:
      1. compute_true_dvp() — aggregates real DK pts allowed per archetype per
         opponent from the full season's player game logs. Much more accurate
         than team-level proxies. Falls back to proxy method if API fails.
      2. get_team_pace() — possessions per 48 min per team.
      3. Rebuild DvP weights and re-apply projections to the player pool.
      4. grade_game_matchups() — rank games by total, pace, and DvP (relative
         grading so the best game on the slate always earns A or B).

    Note: game log fetch is ~5–15 s on first call; cached for 24 h after that.
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")

    players = _state["players"].copy()

    # ── Step 1: True DvP from game logs ────────────────────────────────────
    true_dvp   = compute_true_dvp(players)
    data_source = "game-log DvP (true)" if true_dvp else "proxy metrics (fallback)"

    # ── Step 2: Pace ────────────────────────────────────────────────────────
    team_pace = get_team_pace()

    # ── Step 3: Rebuild DvP weights and re-apply to player pool ────────────
    if true_dvp:
        dvp_weights = build_dvp_weights(players)
        players     = apply_dvp_adjustments(players, dvp_weights)
        _state["players"] = players   # persist updated projections

    # ── Step 4: Grade games ─────────────────────────────────────────────────
    _game_totals = _state.get("game_totals") or GAME_TOTALS
    grades = grade_game_matchups(players, _game_totals, team_pace=team_pace or None)

    # Per-player DvP summary (top movers)
    dvp_cols = [c for c in ["name", "team", "opp", "archetype", "dvp_mult", "proj_pts_dk"]
                if c in players.columns]
    dvp_summary = (
        players[dvp_cols]
        .assign(dvp_pct=lambda df: ((df["dvp_mult"] - 1.0) * 100).round(1))
        .query("dvp_pct.abs() >= 3")
        .sort_values("dvp_pct", ascending=False)
        .head(30)
        .to_dict("records")
    )

    # Serialize updated player projections so the client can refresh without a
    # separate API call (replaces the broken `loadPlayers()` call in the UI).
    upd_cols = [c for c in ["player_id", "dvp_mult", "proj_pts_dk", "ceiling", "gpp_score"]
                if c in players.columns]
    updated_players = players[upd_cols].to_dict(orient="records")

    return {
        "status":          "ok",
        "data_source":     data_source,
        "teams_fetched":   len(true_dvp) if true_dvp else 0,
        "pace_fetched":    bool(team_pace),
        "games":           grades,
        "top_dvp":         dvp_summary,
        "updated_players": updated_players,
    }


@app.get("/api/game-totals")
async def get_game_totals_endpoint():
    """Return the current game totals (estimated or user-entered Vegas lines)."""
    gt = _state.get("game_totals") or {}
    return {"game_totals": gt}


@app.post("/api/game-totals")
async def update_game_totals_endpoint(request: Request):
    """
    Accept user-entered Vegas lines and re-apply them to the player pool.

    Payload: {"game_totals": {"AWAY@HOME": {"total": float, "home_implied": float, "away_implied": float}}}

    For each game in the payload:
      - Merges into _state["game_totals"] (games not included are unchanged)
      - Calls apply_game_total_updates() to adjust proj_pts_dk, ceiling, floor,
        value, and gpp_score by the ratio of new/old game_total_factor × regime_factor
      - Returns updated game_summary and per-player updated fields for the UI
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")

    body = await request.json()
    new_gt = body.get("game_totals", {})
    if not new_gt:
        raise HTTPException(status_code=400, detail="No game totals provided.")

    # Merge incoming lines into existing (only override games that were sent)
    current = dict(_state.get("game_totals") or GAME_TOTALS)
    current.update(new_gt)
    _state["game_totals"] = current

    # Re-apply to player pool
    updated = apply_game_total_updates(_state["players"], current)
    _state["players"] = updated

    # Build updated game summary
    game_summary = []
    for matchup, gt in sorted(current.items(), key=lambda x: -x[1]["total"]):
        gp = updated[updated["matchup"] == matchup]
        game_summary.append({
            "matchup":      matchup,
            "total":        gt["total"],
            "home_implied": gt["home_implied"],
            "away_implied": gt["away_implied"],
            "player_count": len(gp),
            "top_proj":     round(float(gp["proj_pts_dk"].max()), 1) if len(gp) else 0,
        })

    # Serialize updated player projections for UI refresh
    upd_cols = [c for c in ["player_id", "game_total", "proj_pts_dk", "ceiling",
                             "floor", "value", "gpp_score"]
                if c in updated.columns]
    updated_players = updated[upd_cols].to_dict(orient="records")

    return {
        "status":          "ok",
        "game_summary":    game_summary,
        "updated_players": updated_players,
    }


@app.post("/api/fetch-vegas-lines")
async def fetch_vegas_lines_endpoint(request: Request):
    """
    Auto-fetch NBA game totals and spreads from The Odds API.

    Body: {"api_key": "<your-key>"}  (optional if ODDS_API_KEY env var is set)

    Calls The Odds API, matches games to the current slate, applies the real
    lines to the player pool via apply_game_total_updates(), and returns the
    updated game summary + per-player projection changes.

    The Odds API free tier gives 500 requests/month. Results are cached 30 min
    server-side so repeated calls don't burn quota.
    """
    body    = await request.json()
    api_key = body.get("api_key", "")

    players = _state.get("players")  # may be None before upload

    raw = fetch_vegas_lines(api_key=api_key, player_pool=players)

    # Separate metadata from game data
    meta     = raw.pop("_meta", {})
    lines    = {k: v for k, v in raw.items() if not k.startswith("_")}

    if not lines:
        api_err = meta.get("error", "")
        detail  = f"No lines returned from The Odds API. {api_err}".strip()
        raise HTTPException(status_code=502, detail=detail)

    # Merge into existing game_totals and apply
    current = dict(_state.get("game_totals") or GAME_TOTALS)
    current.update(lines)
    _state["game_totals"] = current

    game_summary    = []
    updated_players = []

    if players is not None:
        updated = apply_game_total_updates(players, current)
        _state["players"] = updated

        for matchup, gt in sorted(current.items(), key=lambda x: -x[1]["total"]):
            gp = updated[updated["matchup"] == matchup]
            game_summary.append({
                "matchup":      matchup,
                "total":        gt["total"],
                "home_implied": gt["home_implied"],
                "away_implied": gt["away_implied"],
                "player_count": len(gp),
                "top_proj":     round(float(gp["proj_pts_dk"].max()), 1) if len(gp) else 0,
                "source":       "real" if gt.get("_real") else "estimated",
            })

        upd_cols = [c for c in ["player_id", "game_total", "proj_pts_dk",
                                 "ceiling", "floor", "value", "gpp_score"]
                    if c in updated.columns]
        updated_players = updated[upd_cols].to_dict(orient="records")
    else:
        # No player pool yet — just return the raw lines for display
        for matchup, gt in sorted(lines.items(), key=lambda x: -x[1]["total"]):
            game_summary.append({
                "matchup":      matchup,
                "total":        gt["total"],
                "home_implied": gt["home_implied"],
                "away_implied": gt["away_implied"],
                "player_count": 0,
                "top_proj":     0,
                "source":       "real",
            })

    return {
        "status":            "ok",
        "games_fetched":     len(lines),
        "remaining_requests": meta.get("remaining_requests", "?"),
        "game_summary":      game_summary,
        "updated_players":   updated_players,
    }


@app.post("/api/late-swap")
async def do_late_swap(request: Request):
    """
    Contest-aware late swap using mini-ILP re-optimization.

    Accepts optional JSON body:
      { "manual_out": ["Player Name", ...] }   — treat these players as OUT
        even if not confirmed scratched (useful for B2B risk management).

    For each lineup with an OUT player:
      1. Locks active players, re-runs the ILP to find the optimal replacement.
      2. Applies on/off usage absorption: boosts projections of players who
         benefit from the OUT player's absence before scoring replacements.
      3. Falls back to contest-aware greedy (correlation + leverage scoring)
         if ILP is infeasible due to salary constraints.
      4. Re-exports the DK CSV with all swapped lineups.
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")
    if not _state["lineups"]:
        raise HTTPException(status_code=400, detail="Generate lineups first.")

    # Parse optional manual_out list from request body
    manual_out_names: list[str] = []
    try:
        body = await request.json()
        manual_out_names = [n.strip() for n in body.get("manual_out", []) if n.strip()]
    except Exception:
        pass   # no body or not JSON — fine

    players = _state["players"]

    # Detect OUT players by comparing current pool against the original snapshot.
    # apply_status_updates() *removes* OUT players from the pool entirely rather than
    # marking them, so checking status=="OUT" in the current pool always returns nothing.
    full_pool = _state.get("_full_pool_snapshot")
    if full_pool is not None:
        current_pids = set(players["player_id"].astype(str))
        all_pids     = set(full_pool["player_id"].astype(str))
        out_ids      = all_pids - current_pids   # players dropped from pool = OUT
    else:
        # Fallback for sessions where no snapshot exists (legacy / direct API usage)
        out_ids = set(players[players["status"] == "OUT"]["player_id"].astype(str).tolist())

    # Merge manually flagged OUT players (B2B risk, personal decisions, etc.)
    if manual_out_names:
        from difflib import get_close_matches as _gcm
        name_to_pid = dict(zip(players["name"].str.lower(), players["player_id"].astype(str)))
        for mn in manual_out_names:
            key   = mn.lower()
            match = name_to_pid.get(key)
            if not match:
                close = _gcm(key, name_to_pid.keys(), n=1, cutoff=0.75)
                match = name_to_pid[close[0]] if close else None
            if match:
                out_ids.add(match)
                print(f"[late-swap] Manual OUT: {mn} (pid={match})")
            else:
                print(f"[late-swap] Manual OUT: '{mn}' not found in player pool — skipped")

    # If contest data was uploaded, enrich players with field/leader exposure
    # and real ownership — used by both late_swap_lineups and contest_mode_late_swap.
    contest = _state.get("contest")
    if contest and contest.get("real_ownership"):
        from difflib import get_close_matches
        field_sets   = [set(lu) for lu in contest.get("field_lineups", [])]
        leader_sets  = [set(lu) for lu in contest.get("leader_lineups", [])]
        real_own     = contest["real_ownership"]
        name_lower   = {n.lower(): n for n in players["name"].tolist()}

        players = players.copy()
        players["field_exposure"]  = 0.0
        players["leader_exposure"] = 0.0

        for _, row in players.iterrows():
            pname = row["name"]
            idx   = row.name
            n_field  = sum(1 for lu in field_sets  if pname in lu)
            n_leader = sum(1 for lu in leader_sets if pname in lu)
            players.at[idx, "field_exposure"]  = round(n_field  / max(len(field_sets),  1) * 100, 1)
            players.at[idx, "leader_exposure"] = round(n_leader / max(len(leader_sets), 1) * 100, 1)

            key   = pname.lower()
            match = name_lower.get(key)
            if not match:
                close = get_close_matches(key, name_lower.keys(), n=1, cutoff=0.80)
                match = name_lower[close[0]] if close else None
            if match and match in real_own:
                players.at[idx, "proj_own"] = real_own[match]

    # ── Choose swap strategy ───────────────────────────────────────────────────
    # A) OUT players present → standard injury-driven swap
    # B) No OUTs + contest data loaded → live-contest upgrade of unlocked slots
    # C) No OUTs + no contest data → nothing to do; return helpful diagnostic
    contest_mode_used = False
    locked_teams: set = set()

    try:
        if out_ids:
            swapped = late_swap_lineups(_state["lineups"], players, out_ids)
        elif contest and contest.get("real_ownership"):
            # Live-contest mode: identify which games have already started
            locked_teams      = get_locked_teams()
            swapped           = contest_mode_late_swap(_state["lineups"], players, locked_teams)
            contest_mode_used = True
        else:
            # Nothing actionable
            raise HTTPException(
                status_code=400,
                detail=(
                    "No OUT players detected and no contest standings uploaded. "
                    "Run Injury Check first, or upload the DraftKings contest standings "
                    "CSV to enable live-contest slot upgrades."
                ),
            )
    except HTTPException:
        raise
    except Exception as exc:
        import traceback, logging as _log
        _log.error("[late-swap] Unhandled error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Late swap error: {exc}")

    # Annotate each swapped lineup with field/leader overlap metrics
    if contest and contest.get("field_lineups"):
        field_sets  = [set(lu) for lu in contest["field_lineups"]]
        leader_sets = [set(lu) for lu in contest.get("leader_lineups", [])]
        for lu in swapped:
            lu_names = set(lu.get("names", []))
            n_match_field  = sum(1 for fs in field_sets  if len(lu_names & fs) >= 5)
            n_match_leader = sum(1 for ls in leader_sets if len(lu_names & ls) >= 5)
            lu["field_overlap_pct"]  = round(n_match_field  / max(len(field_sets),  1) * 100, 1)
            lu["leader_overlap_pct"] = round(n_match_leader / max(len(leader_sets), 1) * 100, 1)

    _state["lineups"] = swapped

    # Re-export DK CSV
    today = date.today().strftime("%Y-%m-%d")
    OUTPUT_DIR.mkdir(exist_ok=True)
    dk_path = OUTPUT_DIR / f"dk_upload_{today}.csv"
    export_dk_csv(swapped, players, dk_path)
    _state["dk_path"] = dk_path

    n_swapped  = sum(1 for lu in swapped if lu.get("swapped"))
    n_ilp      = sum(1 for lu in swapped if lu.get("swap_method") == "mini-ILP")
    n_greedy   = sum(1 for lu in swapped if lu.get("swap_method") == "greedy-contest")
    n_upgrade  = sum(1 for lu in swapped if lu.get("swap_method") == "contest-upgrade")
    avg_lev    = round(
        sum(lu.get("leverage", 0) for lu in swapped) / max(len(swapped), 1), 1
    )

    contest_used = bool(_state.get("contest") and _state["contest"].get("real_ownership"))
    avg_field_overlap  = round(
        sum(lu.get("field_overlap_pct",  0) for lu in swapped) / max(len(swapped), 1), 1
    ) if contest_used else None
    avg_leader_overlap = round(
        sum(lu.get("leader_overlap_pct", 0) for lu in swapped) / max(len(swapped), 1), 1
    ) if contest_used else None

    # Build a human-readable mode label for the UI
    if contest_mode_used:
        n_locked   = len(locked_teams)
        swap_label = (
            f"Live-contest upgrade mode — {n_locked} team(s) locked "
            f"(games started). Upgraded {n_upgrade} lineup(s) by replacing "
            f"underperforming unlocked slots with lower-owned options."
        )
    elif out_ids:
        swap_label = (
            f"Injury swap mode — {len(out_ids)} OUT player(s) replaced. "
            f"{n_ilp} via ILP, {n_greedy} via greedy fallback."
        )
    else:
        swap_label = "No swaps performed."

    return {
        "status":              "ok",
        "lineups_swapped":     n_swapped,
        "ilp_swaps":           n_ilp,
        "greedy_swaps":        n_greedy,
        "upgrade_swaps":       n_upgrade,
        "avg_leverage":        avg_lev,
        "contest_mode":        contest_mode_used,
        "locked_teams":        sorted(locked_teams),
        "swap_label":          swap_label,
        "avg_field_overlap":   avg_field_overlap,
        "avg_leader_overlap":  avg_leader_overlap,
    }


@app.post("/api/upload-lineups")
async def upload_lineups(file: UploadFile = File(...)):
    """
    Re-import a previously exported DK lineup CSV into app state.

    Accepts the same CSV format this app exports (PG/SG/SF/PF/C/G/F/UTIL columns,
    cells = "Name (DK_ID)" strings). This lets you run Late Swap on lineups from
    a previous session after a server restart.

    Requires a salary CSV to already be uploaded (so player data is available for
    projections, salary, and slot eligibility lookups).
    """
    if _state["players"] is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a salary CSV first so player data is available for matching.",
        )

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
        f.write(content)
        tmp_path = f.name

    try:
        lineups, warnings = parse_lineup_csv(tmp_path, _state["players"])
    finally:
        import os
        os.unlink(tmp_path)

    if not lineups:
        raise HTTPException(
            status_code=400,
            detail=("No valid lineups parsed. " + " | ".join(warnings[:5])) if warnings
                   else "No valid lineups found in the uploaded CSV.",
        )

    # Merge with or replace existing lineups depending on what the user has
    existing = _state["lineups"]
    if existing:
        # Renumber imported lineups to continue from existing
        offset = max(lu["lineup_num"] for lu in existing)
        for lu in lineups:
            lu["lineup_num"] += offset
        _state["lineups"] = existing + lineups
        mode = "merged"
    else:
        _state["lineups"] = lineups
        mode = "replaced"

    # Recompute aggregate state
    out_ids = set(
        _state["players"][_state["players"]["status"] == "OUT"]["player_id"].astype(str).tolist()
    )
    affected = [
        {"lineup_num": lu["lineup_num"], "scratched_ids": [
            p for p in lu["player_ids"] if str(p) in out_ids
        ]}
        for lu in lineups
        if any(str(p) in out_ids for p in lu["player_ids"])
    ]

    # Serialize for UI — same shape as /api/lineups
    serialized = []
    pid_to_name = dict(zip(_state["players"]["player_id"].astype(str), _state["players"]["name"]))
    for lu in _state["lineups"]:
        serialized.append({
            "lineup_num":      lu["lineup_num"],
            "player_ids":      [str(p) for p in lu["player_ids"]],
            "names":           lu["names"],
            "positions":       lu.get("positions", []),
            "teams":           lu.get("teams", []),
            "salaries":        lu.get("salaries", []),
            "projections":     lu.get("projections", []),
            "total_salary":    lu.get("total_salary", 0),
            "proj_pts":        lu.get("proj_pts", 0),
            "ceiling":         lu.get("ceiling", 0),
            "proj_own":        lu.get("proj_own", 0),
            "leverage":        lu.get("leverage", 0),
            "swapped":         lu.get("swapped", False),
            "swap_method":     lu.get("swap_method"),
            "imported":        lu.get("imported", False),
        })

    return {
        "status":           "ok",
        "mode":             mode,
        "lineups_imported": len(lineups),
        "total_lineups":    len(_state["lineups"]),
        "warnings":         warnings,
        "affected_lineups": affected,
        "lineups":          serialized,
    }


@app.post("/api/upload-contest")
async def upload_contest(file: UploadFile = File(...)):
    """
    Parse a DraftKings contest standings CSV to unlock contest-aware late swap.

    The contest file (downloadable from DK contest page → Export) contains:
      - Real player ownership (%Drafted) — replaces heuristic proj_own model
      - All field lineups — used to score replacement uniqueness vs the field
      - Current scores — identifies leaders whose stacks to avoid/target

    After upload:
      - proj_own for every player is updated to actual contest ownership
      - Late swap replacement scoring uses field-lineup overlap + leader avoidance
      - A field ownership panel is shown in the UI
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
        f.write(content)
        tmp_path = f.name

    try:
        contest_data = parse_contest_csv(tmp_path, _state["players"])
    finally:
        import os
        os.unlink(tmp_path)

    if "error" in contest_data:
        raise HTTPException(status_code=400, detail=f"Could not parse contest file: {contest_data['error']}")

    if not contest_data.get("real_ownership"):
        raise HTTPException(status_code=400,
            detail="No ownership data found. Make sure this is a DraftKings contest standings export.")

    # ── Update player pool proj_own with real contest ownership ───────────────
    from difflib import get_close_matches
    players   = _state["players"].copy()
    real_own  = contest_data["real_ownership"]
    name_lower_map = {n.lower(): n for n in players["name"].tolist()}
    updated_own: list[dict] = []

    for raw_name, pct in real_own.items():
        key   = raw_name.lower()
        match = name_lower_map.get(key)
        if not match:
            close = get_close_matches(key, name_lower_map.keys(), n=1, cutoff=0.80)
            match = name_lower_map[close[0]] if close else None
        if match:
            mask = players["name"] == match
            old_own = float(players.loc[mask, "proj_own"].iloc[0])
            players.loc[mask, "proj_own"] = round(pct, 1)
            # Recalculate gpp_score with real ownership
            for idx in players[mask].index:
                ceil  = float(players.loc[idx, "ceiling"])
                proj  = float(players.loc[idx, "proj_pts_dk"])
                players.loc[idx, "gpp_score"] = round(
                    ceil * 0.60 + proj * 0.25 + (1 - pct / 100) * 10, 3
                )
            updated_own.append({"name": match, "old_own": old_own, "new_own": pct})

    _state["players"] = players
    _state["contest"] = contest_data

    # Summarise field stacks (most common 2-player combos)
    top_stacks = list(contest_data.get("field_stacks", {}).items())[:10]

    return {
        "status":          "ok",
        "entry_count":     contest_data["entry_count"],
        "leader_count":    contest_data["leader_count"],
        "ownership_updated": len(updated_own),
        "top_players":     contest_data["top_players"][:20],
        "top_stacks":      [{"pair": k, "count": v} for k, v in top_stacks],
        "updated_players": players[["player_id", "name", "proj_own", "gpp_score"]].to_dict(orient="records"),
    }


@app.get("/api/starters")
async def get_starters():
    """
    Check ESPN for confirmed/projected starting lineups for tonight.

    After confirming starters, triggers a dynamic DvP recalculation for any
    player confirmed as NOT IN THE LINEUP (bench/absent). This is the critical
    pre-game shift the user raised: when a defensive anchor is confirmed sitting,
    the opposing archetype matchup shifts dramatically before tip-off.
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")

    from difflib import get_close_matches
    starter_result    = get_confirmed_starters()
    starter_map       = starter_result.get("players", {})         # all (confirmed + projected)
    confirmed_map     = starter_result.get("confirmed_players", {}) # confirmed games only
    games_info        = starter_result.get("games", [])
    data_source       = starter_result.get("source", "unknown")

    players  = _state["players"].copy()

    confirmed_starters = []
    confirmed_bench    = []
    debut_confirmed    = []

    # DvP shifts ONLY fire for players from officially confirmed lineups.
    # Using projected (unconfirmed) lineup data here would incorrectly flag
    # stars like Jokic as "out/bench" when their game hasn't confirmed yet.
    confirmed_not_starting = []

    for name, role in starter_map.items():
        names_lower = {n.lower(): n for n in players["name"].tolist()}
        key   = name.lower()
        match = names_lower.get(key)
        if not match:
            close = get_close_matches(key, names_lower.keys(), n=1, cutoff=0.82)
            match = names_lower[close[0]] if close else None
        if not match:
            continue

        row    = players[players["name"] == match].iloc[0]
        status = row["status"]

        if role == "STARTER":
            confirmed_starters.append({
                "name": match, "team": row["team"],
                "salary": int(row["salary"]), "status": status,
                "archetype": str(row.get("archetype", "")),
            })
            if status == "DEBUT":
                mask = players["name"] == match
                players.loc[mask, "proj_pts_dk"] = (players.loc[mask, "proj_pts_dk"] * 1.20).round(2)
                players.loc[mask, "ceiling"]     = (players.loc[mask, "ceiling"]     * 1.20).round(2)
                players.loc[mask, "gpp_score"]   = (players.loc[mask, "gpp_score"]   * 1.20).round(3)
                debut_confirmed.append(match)
        elif role == "BENCH":
            confirmed_bench.append({
                "name": match, "team": row["team"],
                "salary": int(row["salary"]), "status": status,
                "archetype": str(row.get("archetype", "")),
            })
            # High-salary bench player = likely resting or reduced role.
            # ONLY trigger DvP if this player is from a CONFIRMED game
            # (confirmed_map contains only players whose game has the Rotowire
            #  green "Confirmed Lineup" dot). This prevents false DvP shifts from
            # projected/expected lineups where a star may appear in any position.
            if int(row["salary"]) >= 7000 and name in confirmed_map:
                confirmed_not_starting.append({
                    "name":      match,
                    "team":      row["team"],
                    "archetype": str(row.get("archetype", "COMBO_G")),
                    "salary":    int(row["salary"]),
                })

    # ── Dynamic DvP shift for confirmed bench/absent high-salary players ──────
    dvp_shifts = []
    if confirmed_not_starting:
        players = apply_lineup_confirmation_dvp(players, confirmed_not_starting)
        # Build shift summary for UI
        from nba_dfs.test_slate import ARCHETYPE_DEFENSIVE_IMPACT
        for absent in confirmed_not_starting:
            arch_boosts = ARCHETYPE_DEFENSIVE_IMPACT.get(absent["archetype"], {})
            sal_scale   = min(1.0, max(0.1, (absent["salary"] - 3000) / 7000))
            for ben_arch, base_boost in arch_boosts.items():
                boost_pct = round(base_boost * sal_scale * 100, 1)
                if boost_pct >= 3.0:
                    dvp_shifts.append({
                        "absent_name":      absent["name"],
                        "absent_team":      absent["team"],
                        "absent_archetype": absent["archetype"],
                        "beneficiary_arch": ARCHETYPE_LABELS.get(ben_arch, ben_arch),
                        "boost_pct":        boost_pct,
                    })

    # ── Inactive / not-rostered detection ────────────────────────────────────
    # Only remove a player when ALL of these are true:
    #   1. Their team has ≥ 8 Rotowire-confirmed players (full roster coverage,
    #      not just the 5 starters announced early in the day)
    #   2. The player is NOT in the confirmed list
    #   3. The player's salary is ≥ $4,000 (ignore ultra-cheap streamers)
    #
    # This prevents the common false-positive where Rotowire only has starters
    # confirmed at noon and the filter wipes out 80+ legitimate bench players.

    # Names of pool players Rotowire confirmed exist on a roster
    rotowire_rostered_pool_names: set[str] = set(
        p["name"] for p in confirmed_starters + confirmed_bench
    )

    # Count confirmed players per team abbreviation
    team_confirmed_count: dict[str, int] = {}
    for p in confirmed_starters + confirmed_bench:
        t = str(p.get("team", "")).upper().strip()
        if t:
            team_confirmed_count[t] = team_confirmed_count.get(t, 0) + 1

    # Only treat a team as "fully covered" when ≥ 8 players confirmed
    ROSTER_COVERAGE_MIN = 8
    teams_fully_covered: set[str] = {
        t for t, cnt in team_confirmed_count.items() if cnt >= ROSTER_COVERAGE_MIN
    }

    NOT_ROSTERED_SALARY_FLOOR = 4000   # ignore players below this salary

    not_rostered: list[dict] = []
    not_rostered_ids: set[str] = set()

    for _, row in players.iterrows():
        team   = str(row["team"]).upper()
        name   = row["name"]
        pid    = str(row["player_id"])
        salary = int(row.get("salary", 0))
        # Only flag when team is fully covered, player is unlisted, and salary is meaningful
        if (
            team in teams_fully_covered
            and name not in rotowire_rostered_pool_names
            and salary >= NOT_ROSTERED_SALARY_FLOOR
        ):
            not_rostered.append({
                "name":      name,
                "team":      team,
                "salary":    salary,
                "player_id": pid,
            })
            not_rostered_ids.add(pid)

    # Remove not-rostered players from the live pool immediately
    if not_rostered_ids:
        players = players[
            ~players["player_id"].astype(str).isin(not_rostered_ids)
        ].copy().reset_index(drop=True)

    _state["players"]          = players
    _state["not_rostered_ids"] = not_rostered_ids  # optimizer checks this too

    pool_cols = ["player_id", "name", "team", "status", "salary",
                 "proj_pts_dk", "ceiling", "gpp_score", "archetype", "dvp_mult"]
    pool_cols = [c for c in pool_cols if c in players.columns]

    # has_lineup = confirmed (Rotowire green dot); projected games are NOT counted
    games_with_lineups  = sum(1 for g in games_info if g.get("confirmed"))
    games_without       = [g for g in games_info if not g.get("confirmed")]

    return {
        "starter_count":          len(confirmed_starters),
        "starters":               confirmed_starters[:30],
        "bench":                  confirmed_bench[:20],
        "debut_confirmed":        debut_confirmed,
        "dvp_shifts":             dvp_shifts,
        "updated_players":        players[pool_cols].to_dict(orient="records"),
        "games":                  games_info,
        "games_with_lineups":     games_with_lineups,
        "games_without_lineups":  [g["matchup"] for g in games_without],
        "data_source":            data_source,
        "not_rostered":           not_rostered,
    }


@app.post("/api/set-projection")
async def set_projection(body: dict):
    """
    Manually override a player's projection.
    Body: {"player_id": "...", "proj_pts_dk": 42.0}
    Recalculates ceiling, floor, std, gpp_score from the new projection.
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")

    pid  = str(body.get("player_id", ""))
    proj = float(body.get("proj_pts_dk", 0))
    if not pid or proj <= 0:
        raise HTTPException(status_code=400, detail="player_id and proj_pts_dk required.")

    players = _state["players"]
    mask    = players["player_id"].astype(str) == pid
    if not mask.any():
        raise HTTPException(status_code=404, detail="Player not found.")

    status = players.loc[mask, "status"].iloc[0]
    std_factor = 0.38 if status == "DEBUT" else 0.28
    proj_std   = round(proj * std_factor, 2)

    players.loc[mask, "proj_pts_dk"] = round(proj, 2)
    players.loc[mask, "proj_std"]    = proj_std
    players.loc[mask, "ceiling"]     = round(proj + 1.28 * proj_std, 2)
    players.loc[mask, "floor"]       = max(0, round(proj - 1.28 * proj_std, 2))
    players.loc[mask, "value"]       = round(proj / (players.loc[mask, "salary"].iloc[0] / 1000), 3)
    # Recalculate gpp_score
    ceil_new = players.loc[mask, "ceiling"].iloc[0]
    own_new  = players.loc[mask, "proj_own"].iloc[0]
    players.loc[mask, "gpp_score"] = round(
        ceil_new * 0.60 + proj * 0.25 + (1 - own_new / 100) * 10, 3
    )
    players.loc[mask, "status"] = "MANUAL"
    _state["players"] = players

    row = players[mask].iloc[0]
    return {
        "name":        row["name"],
        "proj_pts_dk": float(row["proj_pts_dk"]),
        "ceiling":     float(row["ceiling"]),
        "floor":       float(row["floor"]),
        "gpp_score":   float(row["gpp_score"]),
    }


@app.post("/api/manual-status")
async def manual_status(updates: dict):
    """
    Apply manual status overrides from the UI.
    Body: {"statuses": {"Player Name": "OUT"|"GTD"|"QUESTIONABLE"|"ACTIVE"}}
    """
    if _state["players"] is None:
        raise HTTPException(status_code=400, detail="Upload a salary CSV first.")

    status_map = updates.get("statuses", {})
    _state["players"] = apply_status_updates(_state["players"], status_map)
    return {"status": "ok", "applied": len(status_map)}


@app.get("/api/status")
async def status():
    return {
        "has_players": _state["players"] is not None,
        "player_count": len(_state["players"]) if _state["players"] is not None else 0,
        "lineup_count": len(_state["lineups"]),
        "job_running":  _state["job_running"],
    }


@app.post("/api/cancel")
async def cancel_job():
    """Force-reset a stuck job_running flag (e.g. after a crashed optimizer thread)."""
    _state["job_running"] = False
    _state["job_queue"]   = None
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Backtest API
# ---------------------------------------------------------------------------

@app.post("/api/backtest")
async def run_backtest(
    slate:   UploadFile = File(...),
    entries: UploadFile = File(...),
    results: UploadFile = File(...),
    date:    str        = Form(""),
):
    """
    Run a single-slate backtest.

    Accepts three CSV uploads:
      slate   — DraftKings salary export (Name,ID,Salary,AvgPointsPerGame,...)
      entries — Your submitted lineups (simple or DK bulk-upload format)
      results — DK contest results export (Rank,Points,Lineup,Player,%Drafted,FPTS)

    Returns a JSON report with entry scores, injury reconstruction,
    on/off accuracy, ownership calibration, construction patterns,
    and actionable recommendations.
    """
    import tempfile, os

    # Write uploads to temp files
    tmp_files = []
    try:
        paths = {}
        for name, upload in [("slate", slate), ("entries", entries), ("results", results)]:
            suffix = ".csv"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            content = await upload.read()
            Path(tmp_path).write_bytes(content)
            paths[name] = tmp_path
            tmp_files.append(tmp_path)

        agent  = _get_backtest_agent()
        report = agent.run(
            slate_path   = paths["slate"],
            entries_path = paths["entries"],
            results_path = paths["results"],
            slate_date   = date,
        )
        return JSONResponse(content=report)

    except Exception as exc:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "traceback": traceback.format_exc()},
        )
    finally:
        for f in tmp_files:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/api/backtest/multi")
async def run_backtest_multi(request: Request):
    """
    Run backtest across multiple slates using files already in the contest/ folder.

    POST body (JSON):
      {"slates": [
        {"slate": "contest/dk_slate_3_6.csv",
         "entries": "contest/dk_entries_3_6.csv",
         "results": "contest/contest-results_3_6.csv",
         "date": "3_6"},
        ...
      ]}

    Paths are relative to the project root.
    """
    try:
        body = await request.json()
        slates_spec = body.get("slates", [])
        if not slates_spec:
            raise ValueError("No slates provided")

        resolved = []
        for s in slates_spec:
            resolved.append({
                "slate":   str(PROJECT_ROOT / s["slate"]),
                "entries": str(PROJECT_ROOT / s["entries"]),
                "results": str(PROJECT_ROOT / s["results"]),
                "date":    s.get("date", ""),
            })

        agent  = _get_backtest_agent()
        report = agent.run_multi(resolved)
        return JSONResponse(content=report)

    except Exception as exc:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "traceback": traceback.format_exc()},
        )
