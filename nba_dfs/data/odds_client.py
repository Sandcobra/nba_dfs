"""Thin client for The Odds API with on-disk caching."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib import request, error

import pandas as pd
from loguru import logger

from core.config import CACHE_DIR, THE_ODDS_API_KEY


ESPN_TO_DK = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "los angeles clippers": "LAC",
    "la clippers": "LAC",
    "los angeles lakers": "LAL",
    "la lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}


class OddsClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_path: str | Path | None = None,
        cache_ttl_seconds: int = 3600,
    ):
        self.api_key = (api_key or THE_ODDS_API_KEY or "").strip()
        self.cache_path = Path(cache_path) if cache_path else Path(__file__).parent.parent.parent / "cache" / "odds_cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl_seconds

    def get_nba_odds(self) -> pd.DataFrame:
        cached = self._load_cache(fresh_only=True)
        if cached is not None:
            logger.info("Using cached Odds API response (%d games)", len(cached))
            return cached

        if not self.api_key:
            logger.warning("No THE_ODDS_API_KEY configured; skipping odds fetch")
            return cached if cached is not None else pd.DataFrame()

        df = self._fetch_odds()
        if df.empty:
            stale = self._load_cache(fresh_only=False)
            if stale is not None:
                logger.warning("Using stale odds cache (API unavailable)")
                return stale
            return df

        self._write_cache(df)
        return df

    # ------------------------------------------------------------------ internals
    def _fetch_odds(self) -> pd.DataFrame:
        url = (
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
            f"?apiKey={self.api_key}&regions=us&markets=totals,spreads&oddsFormat=american"
        )
        req = request.Request(url, headers={"User-Agent": "nba-dfs-data-agent/1.0"})
        try:
            with request.urlopen(req, timeout=20) as resp:
                remaining = resp.headers.get("X-Requests-Remaining", "?")
                payload = json.loads(resp.read().decode())
                logger.info("Odds API remaining requests: %s", remaining)
        except error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:240]
            logger.warning("Odds API HTTP %s: %s", exc.code, body)
            return pd.DataFrame()
        except Exception as exc:
            logger.warning("Odds API fetch failed: %s", exc)
            return pd.DataFrame()

        rows = []
        for game in payload:
            home_full = (game.get("home_team") or "").strip()
            away_full = (game.get("away_team") or "").strip()
            home = self._to_abbrev(home_full)
            away = self._to_abbrev(away_full)
            if not home or not away:
                continue

            total, home_spread = self._extract_markets(game.get("bookmakers", []), home_full)
            if total is None:
                continue

            home_implied, away_implied = self._compute_implied_totals(total, home_spread)
            commence = self._parse_commence(game.get("commence_time"))

            rows.append(
                {
                    "home_team_full": home_full,
                    "away_team_full": away_full,
                    "home_team": home,
                    "away_team": away,
                    "matchup": f"{away}@{home}",
                    "total": float(total),
                    "home_spread": float(home_spread) if home_spread is not None else 0.0,
                    "home_implied": home_implied,
                    "away_implied": away_implied,
                    "game_datetime": commence,
                    "game_date": commence.date() if commence else None,
                }
            )

        df = pd.DataFrame(rows)
        return df

    def _extract_markets(self, bookmakers: list, home_full: str) -> tuple[Optional[float], Optional[float]]:
        if not bookmakers:
            return None, None
        # Prefer DraftKings numbers
        ordered = sorted(bookmakers, key=lambda b: b.get("key") != "draftkings")
        total_val: Optional[float] = None
        home_spread: Optional[float] = None
        for book in ordered:
            for market in book.get("markets", []):
                if market.get("key") == "totals" and total_val is None:
                    over = next((o for o in market.get("outcomes", []) if o.get("name") == "Over"), None)
                    if over and over.get("point") is not None:
                        total_val = float(over["point"])
                if market.get("key") == "spreads" and home_spread is None:
                    home_out = next(
                        (o for o in market.get("outcomes", []) if o.get("name") == home_full),
                        None,
                    )
                    if home_out and home_out.get("point") is not None:
                        home_spread = float(home_out["point"])
            if total_val is not None and home_spread is not None:
                break
        return total_val, home_spread

    def _compute_implied_totals(self, total: float, home_spread: Optional[float]) -> tuple[float, float]:
        spread = home_spread or 0.0
        # Negative spread = home favored (give points)
        home_advantage = -spread
        home = round((total + home_advantage) / 2, 1)
        away = round((total - home_advantage) / 2, 1)
        return home, away

    def _parse_commence(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if not dt.tzinfo:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            return None

    def _to_abbrev(self, name: str) -> str:
        key = name.lower()
        return ESPN_TO_DK.get(key, "")

    # --------------------------------------------------------------- caching
    def _load_cache(self, fresh_only: bool) -> Optional[pd.DataFrame]:
        if not self.cache_path.exists():
            return None
        try:
            payload = json.loads(self.cache_path.read_text())
            timestamp = payload.get("timestamp")
            rows = payload.get("rows", [])
        except Exception:
            return None

        if not rows:
            return None

        if timestamp:
            try:
                cached_at = datetime.fromisoformat(timestamp)
            except ValueError:
                cached_at = datetime.utcnow()
            age = datetime.utcnow() - cached_at
            if fresh_only and age.total_seconds() > self.cache_ttl:
                return None

        return pd.DataFrame(rows)

    def _write_cache(self, df: pd.DataFrame):
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "rows": df.to_dict(orient="records"),
        }
        try:
            self.cache_path.write_text(json.dumps(payload))
        except Exception as exc:
            logger.debug("Failed to write odds cache: %s", exc)
