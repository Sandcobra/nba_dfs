"""
News Intel Agent.

Fetches unstructured news blurbs from multiple sources, extracts typed DFS
signals via regex NLP, quantifies projection/ownership modifiers, and returns
a per-player impact map that the optimizer applies before lineup generation.

Signal taxonomy
───────────────
  STARTER_CONFIRMED    : coach/team confirms player is in starting lineup
  BENCH_DEMOTION       : player moved to bench role
  MINUTES_RESTRICTION  : player is playing with a minutes cap
  STARTING_REPLACEMENT : player is starting IN PLACE OF an injured teammate
  LOAD_MANAGEMENT      : rest/load management — treat as OUT equivalent
  USAGE_INCREASE       : primary ball-handler, featured usage increase
  LIMITED_RETURN       : returning from injury, expected to be limited
  CLEARED_FULLY        : player cleared from injury, no restrictions
  GAME_TIME_DECISION   : status still uncertain at game time
  SCRATCHED            : player unexpectedly out tonight

Projection modifiers (applied multiplicatively to proj_pts_dk)
──────────────────────────────────────────────────────────────
  STARTER_CONFIRMED    : × 1.05  (proj_own +4%)
  BENCH_DEMOTION       : × 0.62  (proj_own ×0.70)
  MINUTES_RESTRICTION  : × 0.78  (proj_own ×0.82)
  STARTING_REPLACEMENT : × 1.28  (proj_own +12%)
  LOAD_MANAGEMENT      : × 0.00  (excluded — treat as OUT)
  USAGE_INCREASE       : × 1.12  (proj_own +6%)
  LIMITED_RETURN       : × 0.82  (proj_own ×0.85)
  CLEARED_FULLY        : × 1.08  (proj_own +5%) — reverses GTD haircut
  GAME_TIME_DECISION   : × 0.88  (proj_own ×0.72)
  SCRATCHED            : × 0.00  (excluded)

Source credibility weights
──────────────────────────
  beat_writer (X) : 1.15  — highest; beat writers break news before aggregators
  rotowire / espn : 1.00
  fantasypros     : 0.90
  rotogrinders    : 0.85
  cbssports       : 0.85
  unknown         : 0.70

Age decay
─────────
  Signal confidence = source_weight × exp(-age_hrs / 6)
  Signals older than 20 hours are dropped entirely.
"""

import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_AGE_HOURS   = 20.0    # discard signals older than this
DECAY_HALFLIFE  = 6.0     # hours; confidence halves every 6 hrs
SCRAPE_DELAY    = 0.8     # seconds between HTTP calls

SOURCE_WEIGHTS = {
    "beat_writer": 1.15,   # X / Twitter beat writers — highest trust
    "rotowire":    1.00,
    "espn":        1.00,
    "fantasypros":  0.90,
    "rotogrinders": 0.85,
    "cbssports":    0.85,
}

# X API v2: tweets older than this (hours) are ignored even from beat writers
X_MAX_AGE_HOURS = 6.0

# ── Signal definitions ─────────────────────────────────────────────────────────

SIGNAL_IMPACTS = {
    "STARTER_CONFIRMED":    {"proj_mult": 1.05, "own_delta": +4.0},
    "BENCH_DEMOTION":       {"proj_mult": 0.62, "own_mult":  0.70},
    "MINUTES_RESTRICTION":  {"proj_mult": 0.78, "own_mult":  0.82},
    "STARTING_REPLACEMENT": {"proj_mult": 1.28, "own_delta": +12.0},
    "LOAD_MANAGEMENT":      {"proj_mult": 0.00, "own_mult":  0.00, "exclude": True},
    "USAGE_INCREASE":       {"proj_mult": 1.12, "own_delta": +6.0},
    "LIMITED_RETURN":       {"proj_mult": 0.82, "own_mult":  0.85},
    "CLEARED_FULLY":        {"proj_mult": 1.08, "own_delta": +5.0},
    "GAME_TIME_DECISION":   {"proj_mult": 0.88, "own_mult":  0.72},
    "SCRATCHED":            {"proj_mult": 0.00, "own_mult":  0.00, "exclude": True},
}

# ── Regex pattern bank ────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, list[str]]] = [
    ("SCRATCHED", [
        r"\bscratched\b",
        r"\bwill not play\b",
        r"\bhas been ruled out\b",
        r"\bwon.t play\b",
        r"\bout tonight\b",
        r"\bout for (?:tonight|the game)\b",
    ]),
    ("LOAD_MANAGEMENT", [
        r"\bload management\b",
        r"\brest(?:ing)? tonight\b",
        r"\bsitting out for rest\b",
        r"\bmanagement rest\b",
        r"\bwill rest\b",
    ]),
    ("BENCH_DEMOTION", [
        r"\bcom(?:es?|ing) off the bench\b",
        r"\bmoved? to the bench\b",
        r"\bwill start on the bench\b",
        r"\bbench role\b",
        r"\bstarting (?:\w+ ){0,3}instead of \w+",   # someone else starting
        r"\bnot in the starting lineup\b",
        r"\bwon.t start\b",
    ]),
    ("MINUTES_RESTRICTION", [
        r"\bminutes (?:restriction|limit|cap|ceiling)\b",
        r"\blimited to \d+ minutes\b",
        r"\bwon.t play more than \d+",
        r"\bminute restriction\b",
        r"\bon a minutes limit\b",
        r"\bexpect(?:ed)? (?:around |roughly |approximately )?\d{1,2}[\-–]\d{1,2} minutes\b",
        r"\bplay(?:ing)? through it\b.*\blimited\b",
    ]),
    ("STARTING_REPLACEMENT", [
        r"\bstart(?:ing|s)? in place of\b",
        r"\bstart(?:ing|s)? (?:\w+ ){0,2}(?:while|with|as) \w+ (?:is )?out\b",
        r"\bfill(?:ing|s)? in (?:the )?start(?:ing)?\b",
        r"\bslid(?:es?|ing)? into the starting lineup\b",
        r"\binsert(?:ed|s)? into the starting five\b",
        r"\bexpected to start with .{3,40} out\b",
        r"\bwill draw the start\b",
    ]),
    ("LIMITED_RETURN", [
        r"\breturn(?:ing|s|ed)? from (?:an? )?(?:\w+ )?injury\b",
        r"\bback from \w+ injury\b",
        r"\bon a minutes restriction (?:upon|after) return\b",
        r"\bexpect(?:ed)? to be limited\b",
        r"\bcoming back (?:slowly|carefully|cautiously)\b",
        r"\bplaying through (?:a )?(?:sore|stiff|bruised|tender)\b",
    ]),
    ("CLEARED_FULLY", [
        r"\bcleared (?:to play|for (?:game |full )activity|all activities)\b",
        r"\bno restrictions\b",
        r"\bfull (?:go|participant)\b",
        r"\bavailable without restriction\b",
        r"\bfully healthy\b",
        r"\bremoved from the injury report\b",
        r"\boff the injury report\b",
    ]),
    ("USAGE_INCREASE", [
        r"\bprimary ball.?handler\b",
        r"\bfeatured heavily\b",
        r"\bexpect(?:ed)? (?:a )?(?:increased|elevated|major|large) (?:usage|role)\b",
        r"\bshould see (?:a )?(?:big|large|major|heavy) workload\b",
        r"\bwill (?:handle|run) the offense\b",
        r"\bpoint (?:forward|guard duties|guard role)\b",
        r"\bcreating (?:off|more) (?:the dribble|screens)\b.*\bwith \w+ out\b",
    ]),
    ("STARTER_CONFIRMED", [
        r"\bconfirmed (?:as )?(?:a )?starter\b",
        r"\bin the starting lineup\b",
        r"\bwill start\b",
        r"\bexpected to start\b",
        r"\bslated to start\b",
        r"\bset to start\b",
        r"\bstarting (?:at|tonight)\b",
    ]),
    ("GAME_TIME_DECISION", [
        r"\bgame.?time decision\b",
        r"\bGTD\b",
        r"\bwait(?:ing)? and see\b",
        r"\bwill be evaluated\b",
        r"\bdecision expected\b",
        r"\bstill being evaluated\b",
    ]),
]

# Compile patterns once
_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (sig, [re.compile(p, re.IGNORECASE) for p in pats])
    for sig, pats in _PATTERNS
]

# Priority order: higher index wins if multiple signals fire on same player
_SIGNAL_PRIORITY = [
    "GAME_TIME_DECISION",
    "STARTER_CONFIRMED",
    "USAGE_INCREASE",
    "LIMITED_RETURN",
    "CLEARED_FULLY",
    "MINUTES_RESTRICTION",
    "STARTING_REPLACEMENT",
    "BENCH_DEMOTION",
    "LOAD_MANAGEMENT",
    "SCRATCHED",
]
_PRIORITY_MAP = {s: i for i, s in enumerate(_SIGNAL_PRIORITY)}


# ── HTTP helpers ──────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _age_hours(ts_str: str) -> float:
    """Return age in hours from an ISO timestamp string."""
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        return (now - ts).total_seconds() / 3600.0
    except Exception:
        return 0.0


def _confidence(source: str, age_hrs: float) -> float:
    import math
    w = SOURCE_WEIGHTS.get(source, 0.70)
    return round(w * math.exp(-age_hrs / DECAY_HALFLIFE), 3)


# ── NewsIntelAgent ─────────────────────────────────────────────────────────────

class NewsIntelAgent:
    """
    Fetches news blurbs, extracts typed DFS signals, and returns projection
    + ownership modifiers per player.
    """

    def __init__(self, timeout: int = 20):
        self._client = httpx.Client(
            headers=_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )
        self._x_client = None          # lazy-loaded tweepy.Client
        self._x_user_id_cache: dict[str, str] = {}   # handle → user_id
        # Beat writer fetch stats — populated each call to _fetch_x_beat_writers()
        self._x_stats: dict = {
            "handles_queried": 0,
            "tweets_retrieved": 0,
            "signals_from_x": 0,
            "enabled": False,
            "error": None,
        }

    def close(self):
        self._client.close()

    def _get_x_client(self):
        """Lazy-load tweepy Client using bearer token from config."""
        if self._x_client is not None:
            return self._x_client
        try:
            import tweepy
        except ImportError:
            self._x_stats["error"] = "tweepy not installed on this server -- run: pip install tweepy"
            logger.warning("[news] tweepy not installed")
            return None
        try:
            try:
                from core.config import X_BEARER_TOKEN
            except ImportError:
                from nba_dfs.core.config import X_BEARER_TOKEN
            if not X_BEARER_TOKEN:
                self._x_stats["error"] = "X_BEARER_TOKEN missing -- add it to .env on the server"
                return None
            self._x_client = tweepy.Client(
                bearer_token=X_BEARER_TOKEN,
                wait_on_rate_limit=False,
            )
            return self._x_client
        except Exception as exc:
            self._x_stats["error"] = f"X client init failed: {exc}"
            logger.warning("[news] X client init failed: %s", exc)
            return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        players: pd.DataFrame,
        n_lineups: int = 20,
    ) -> dict:
        """
        Main entry point.

        Parameters
        ----------
        players  : enriched player pool with name, player_id, proj_pts_dk,
                   proj_own, salary, primary_position, team
        n_lineups: planned lineup count (unused currently, reserved)

        Returns
        -------
        {
          "signals"    : list of raw signal dicts,
          "impacts"    : {player_id_str: impact_dict},
          "summary"    : {headline, signal_counts, high_priority},
          "player_news": {player_id_str: [news_item_dicts]},
        }
        """
        raw_news = self._fetch_all()
        signals  = self._extract_signals(raw_news, players)
        impacts  = self._build_impacts(signals, players)

        # Count how many final signals came from beat writers (X)
        self._x_stats["signals_from_x"] = sum(
            1 for s in signals if s.get("source") == "beat_writer"
        )

        summary  = self._build_summary(signals, impacts)

        logger.info(
            "[news] %d news items → %d signals → %d players impacted",
            len(raw_news), len(signals), len(impacts),
        )
        return {
            "signals":     signals,
            "impacts":     impacts,
            "summary":     summary,
            "player_news": self._index_by_player(signals),
            "x_stats":     dict(self._x_stats),   # beat writer fetch stats for UI confirmation
        }

    def apply_to_players(
        self,
        players: pd.DataFrame,
        impacts: dict,
    ) -> pd.DataFrame:
        """
        Apply the impact dict returned by analyze() directly to a players
        DataFrame.  Modifies proj_pts_dk, proj_own, and marks exclusions.

        Returns modified copy.
        """
        df = players.copy()
        excluded_pids: list = []

        for pid_str, impact in impacts.items():
            mask = df["player_id"].astype(str) == pid_str
            if not mask.any():
                continue

            if impact.get("exclude"):
                excluded_pids.append(pid_str)
                continue

            if "proj_pts_mult" in impact:
                df.loc[mask, "proj_pts_dk"] = (
                    df.loc[mask, "proj_pts_dk"] * impact["proj_pts_mult"]
                ).round(2)

            if "own_mult" in impact:
                df.loc[mask, "proj_own"] = (
                    df.loc[mask, "proj_own"] * impact["own_mult"]
                ).clip(1, 40).round(1)
            if "own_delta" in impact:
                df.loc[mask, "proj_own"] = (
                    df.loc[mask, "proj_own"] + impact["own_delta"]
                ).clip(1, 40).round(1)

        # Recompute gpp_score after adjustments
        if "gpp_score" in df.columns and "ceiling" in df.columns:
            df["gpp_score"] = (
                df["ceiling"] * 0.60
                + df["proj_pts_dk"] * 0.25
                + (1 - df["proj_own"] / 100) * 10
            ).round(3)

        return df, excluded_pids

    # ── Fetchers ───────────────────────────────────────────────────────────────

    def _fetch_all(self) -> list[dict]:
        """Fetch from all sources and return combined list of news items."""
        items: list[dict] = []

        # X beat writers first — highest credibility, most real-time
        try:
            x_items = self._fetch_x_beat_writers()
            items.extend(x_items)
            if x_items:
                logger.info("[news] X beat writers: %d items", len(x_items))
        except Exception as exc:
            logger.warning("[news] X beat writer fetch failed: %s", exc)

        for fn in (
            self._fetch_rotowire_news,
            self._fetch_espn_news,
            self._fetch_fantasypros_news,
            self._fetch_rotogrinders_news,
        ):
            try:
                batch = fn()
                items.extend(batch)
                time.sleep(SCRAPE_DELAY)
            except Exception as exc:
                logger.warning("[news] %s failed: %s", fn.__name__, exc)
        logger.debug("[news] Fetched %d total news items", len(items))
        return items

    def _fetch_x_beat_writers(self) -> list[dict]:
        """
        Fetch recent tweets from NBA beat writers using X API v2.

        Uses a single search_recent_tweets call combining all handles with
        OR operators — one API hit regardless of how many writers you track.

        Requires:
          pip install tweepy
          X_BEARER_TOKEN set in .env
          X_BEAT_WRITER_HANDLES list in config.py
        """
        # Reset stats for this run
        self._x_stats = {
            "handles_queried": 0,
            "tweets_retrieved": 0,
            "signals_from_x": 0,
            "enabled": False,
            "error": None,
        }

        client = self._get_x_client()
        if client is None:
            self._x_stats["error"] = "X bearer token not configured or tweepy not installed"
            return []

        try:
            try:
                from core.config import X_BEAT_WRITER_HANDLES
            except ImportError:
                from nba_dfs.core.config import X_BEAT_WRITER_HANDLES
        except ImportError:
            self._x_stats["error"] = "config import failed"
            return []

        if not X_BEAT_WRITER_HANDLES:
            self._x_stats["error"] = "no handles configured"
            return []

        self._x_stats["enabled"] = True
        self._x_stats["handles_queried"] = len(X_BEAT_WRITER_HANDLES)

        # Build query: (from:writer1 OR from:writer2 ...) -is:retweet lang:en
        # X API allows up to 512 chars in query; chunk if needed
        chunks = self._chunk_handles(X_BEAT_WRITER_HANDLES, max_per_chunk=12)
        items: list[dict] = []
        total_tweets = 0

        for chunk in chunks:
            query = (
                "(" + " OR ".join(f"from:{h}" for h in chunk) + ")"
                + " -is:retweet lang:en"
            )
            try:
                import tweepy
                response = client.search_recent_tweets(
                    query=query,
                    max_results=min(100, 10 * len(chunk)),
                    tweet_fields=["created_at", "author_id", "text", "entities"],
                    expansions=["author_id"],
                    user_fields=["username", "name"],
                )
            except tweepy.errors.Unauthorized:
                # 401 = invalid/expired token OR Free tier (no search access)
                # X API Basic tier ($100/mo) required for search_recent_tweets
                self._x_stats["error"] = (
                    "401 Unauthorized — check bearer token or upgrade to X API Basic tier "
                    "(search_recent_tweets requires Basic or higher)"
                )
                self._x_stats["enabled"] = False
                logger.warning("[news] X API 401: bearer token invalid or Free tier account. "
                               "search_recent_tweets requires X API Basic ($100/mo) or higher.")
                break   # no point retrying other chunks
            except Exception as exc:
                logger.warning("[news] X search failed for chunk: %s", exc)
                continue

            if not response or not response.data:
                continue

            total_tweets += len(response.data)

            # Build author_id → username map from includes
            author_map: dict[str, str] = {}
            if response.includes and "users" in response.includes:
                for u in response.includes["users"]:
                    author_map[str(u.id)] = u.username

            for tweet in response.data:
                ts = tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat()
                age = _age_hours(ts)
                if age > X_MAX_AGE_HOURS:
                    continue
                handle = author_map.get(str(tweet.author_id), "unknown")
                items.append({
                    "name":        self._extract_player_name_from_tweet(tweet.text),
                    "text":        tweet.text,
                    "source":      "beat_writer",
                    "handle":      handle,
                    "reported_at": ts,
                    "tweet_id":    str(tweet.id),
                })

        self._x_stats["tweets_retrieved"] = total_tweets
        matched = [i for i in items if i["name"]]   # drop tweets with no player name found
        logger.info(
            "[news] X beat writers: %d handles, %d tweets retrieved, %d with player match",
            len(X_BEAT_WRITER_HANDLES), total_tweets, len(matched),
        )
        return matched

    @staticmethod
    def _chunk_handles(handles: list[str], max_per_chunk: int) -> list[list[str]]:
        """Split handles list into chunks to stay under query length limit."""
        return [
            handles[i: i + max_per_chunk]
            for i in range(0, len(handles), max_per_chunk)
        ]

    @staticmethod
    def _extract_player_name_from_tweet(text: str) -> str:
        """
        Heuristic: the first capitalized two-word phrase in a tweet is likely
        the player name. Falls back to empty string (filtered out later if
        no pool match is found).

        E.g. "LeBron James will not play tonight per source" → "LeBron James"
        """
        # Try to find "First Last" pattern at start or after common prefixes
        prefixes = r"(?:per|source:|injury:|update:|lineup:|news:)?\s*"
        m = re.search(
            prefixes + r"([A-Z][a-z]+(?:\s+[A-Z][a-z\']+){1,2})",
            text,
        )
        if m:
            return m.group(1).strip()
        return ""

    def _fetch_rotowire_news(self) -> list[dict]:
        resp = self._client.get("https://www.rotowire.com/basketball/news.php")
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        items = []
        for card in soup.select("div.news-item, li.news-item, div.player-news-item"):
            name_el  = card.select_one(".news-player, .player-name, .name, a[href*='/player/']")
            text_el  = card.select_one(".news-analysis, .news-text, p, .content")
            time_el  = card.select_one("time, .news-time, .timestamp")
            if not name_el or not text_el:
                continue
            ts = time_el.get("datetime", "") or time_el.get_text(strip=True) if time_el else ""
            items.append({
                "name":        name_el.get_text(strip=True),
                "text":        text_el.get_text(strip=True),
                "source":      "rotowire",
                "reported_at": ts or datetime.utcnow().isoformat(),
            })
        return items

    def _fetch_espn_news(self) -> list[dict]:
        # ESPN injury table columns: NAME | POS | EST.RETURN | STATUS | COMMENT
        resp = self._client.get("https://www.espn.com/nba/injuries")
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        items = []
        for table in soup.select("div.ResponsiveTable"):
            for row in table.select("tr.Table__TR")[1:]:
                cols = row.select("td")
                if len(cols) < 4:
                    continue
                name    = cols[0].get_text(strip=True)
                status  = cols[3].get_text(strip=True) if len(cols) > 3 else ""
                comment = cols[4].get_text(strip=True) if len(cols) > 4 else ""
                # Use comment as primary text; fall back to status so regex has something
                text = comment if comment else status
                if not name or not text:
                    continue
                items.append({
                    "name":        name,
                    "text":        text,
                    "status":      status,   # raw ESPN status for direct signal mapping
                    "source":      "espn",
                    "reported_at": datetime.utcnow().isoformat(),
                })
        return items

    def _fetch_fantasypros_news(self) -> list[dict]:
        resp = self._client.get(
            "https://www.fantasypros.com/nba/news/",
            headers={**_HEADERS, "Accept": "text/html,application/xhtml+xml"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        items = []
        for article in soup.select("div.player-news-item, article.news-item, li.news"):
            name_el = article.select_one("a.player-name, .player, h3, h4")
            text_el = article.select_one("p, .body, .summary, .content")
            time_el = article.select_one("time, .date, .timestamp")
            if not name_el or not text_el:
                continue
            ts = time_el.get("datetime", "") if time_el else ""
            items.append({
                "name":        name_el.get_text(strip=True),
                "text":        text_el.get_text(strip=True),
                "source":      "fantasypros",
                "reported_at": ts or datetime.utcnow().isoformat(),
            })
        return items

    def _fetch_rotogrinders_news(self) -> list[dict]:
        resp = self._client.get("https://rotogrinders.com/lineups/nba")
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        items = []
        for card in soup.select(".player-news-card, .news-item, .player-card"):
            name_el   = card.select_one(".player-name, .name, a.player")
            text_el   = card.select_one(".news-text, p, .content, .blurb")
            time_el   = card.select_one("time, .time, .timestamp")
            if not name_el or not text_el:
                continue
            ts = time_el.get("datetime", "") if time_el else ""
            items.append({
                "name":        name_el.get_text(strip=True),
                "text":        text_el.get_text(strip=True),
                "source":      "rotogrinders",
                "reported_at": ts or datetime.utcnow().isoformat(),
            })
        return items

    # ── Signal extraction ─────────────────────────────────────────────────────

    def _extract_signals(
        self,
        news_items: list[dict],
        players: pd.DataFrame,
    ) -> list[dict]:
        """
        Match each news item to a player in the pool, classify the signal,
        compute confidence, and return the enriched signal list.
        """
        # Build name → player_id lookup (lowercase for fuzzy matching)
        name_map: dict[str, str] = {}
        for _, row in players.iterrows():
            name_lc = str(row.get("name", "")).lower().strip()
            if name_lc:
                name_map[name_lc] = str(row.get("player_id", ""))

        signals: list[dict] = []

        for item in news_items:
            raw_name = item.get("name", "").strip()
            text     = item.get("text", "").strip()
            source   = item.get("source", "unknown")
            ts       = item.get("reported_at", "")

            if not raw_name or not text:
                continue

            age_hrs = _age_hours(ts) if ts else 0.0
            if age_hrs > MAX_AGE_HOURS:
                continue

            # Match to player pool
            pid = self._match_player(raw_name, name_map)
            if not pid:
                continue

            # Direct status → signal mapping (ESPN/injury scrapers provide raw status field)
            raw_status = item.get("status", "").upper().strip()
            _STATUS_MAP = {
                "OUT": "SCRATCHED",
                "DOUBTFUL": "SCRATCHED",
                "QUESTIONABLE": "GAME_TIME_DECISION",
                "GTD": "GAME_TIME_DECISION",
                "DAY-TO-DAY": "GAME_TIME_DECISION",
                "PROBABLE": "GAME_TIME_DECISION",
            }
            signal_type = _STATUS_MAP.get(raw_status) or self._classify(text)
            if signal_type is None:
                continue

            conf = _confidence(source, age_hrs)
            if conf < 0.05:
                continue

            signals.append({
                "player_id":   pid,
                "player_name": raw_name,
                "signal_type": signal_type,
                "text":        text[:300],
                "source":      source,
                "reported_at": ts,
                "age_hrs":     round(age_hrs, 2),
                "confidence":  conf,
            })

        # Deduplicate: per player, keep highest-priority + highest-confidence signal
        return self._deduplicate(signals)

    def _match_player(self, raw_name: str, name_map: dict[str, str]) -> Optional[str]:
        """Fuzzy-ish name matching: exact → last-name → first+last fragment."""
        lc = raw_name.lower().strip()

        # Exact
        if lc in name_map:
            return name_map[lc]

        # Last name match (only if unique)
        parts = lc.split()
        if len(parts) >= 2:
            last = parts[-1]
            first = parts[0]
            matches = [
                pid for name, pid in name_map.items()
                if name.endswith(last) and name.startswith(first[:2])
            ]
            if len(matches) == 1:
                return matches[0]

        return None

    def _classify(self, text: str) -> Optional[str]:
        """
        Return the highest-priority matching signal type for a news text.
        Returns None if no pattern fires.
        """
        fired: list[str] = []
        for sig, patterns in _COMPILED:
            for pat in patterns:
                if pat.search(text):
                    fired.append(sig)
                    break  # one match per signal type sufficient

        if not fired:
            return None

        # Return highest-priority signal
        return max(fired, key=lambda s: _PRIORITY_MAP.get(s, 0))

    def _deduplicate(self, signals: list[dict]) -> list[dict]:
        """
        Per player: if multiple signals, keep highest-priority one.
        If same priority, keep highest-confidence.
        """
        best: dict[str, dict] = {}
        for sig in signals:
            pid  = sig["player_id"]
            prio = _PRIORITY_MAP.get(sig["signal_type"], 0)
            conf = sig["confidence"]

            existing = best.get(pid)
            if existing is None:
                best[pid] = sig
            else:
                ex_prio = _PRIORITY_MAP.get(existing["signal_type"], 0)
                if prio > ex_prio or (prio == ex_prio and conf > existing["confidence"]):
                    best[pid] = sig

        return list(best.values())

    # ── Impact quantification ─────────────────────────────────────────────────

    def _build_impacts(
        self,
        signals: list[dict],
        players: pd.DataFrame,
    ) -> dict:
        """
        Convert signals → per-player impact dict.
        Scales multipliers by confidence (partial effect when conf < 1).

        Returns {player_id_str: {proj_pts_mult?, own_mult?, own_delta?,
                                  exclude?, signal_type, confidence, text}}.
        """
        impacts: dict[str, dict] = {}

        for sig in signals:
            pid   = sig["player_id"]
            stype = sig["signal_type"]
            conf  = sig["confidence"]
            base  = SIGNAL_IMPACTS.get(stype, {})

            impact: dict = {
                "signal_type": stype,
                "confidence":  conf,
                "source":      sig["source"],
                "text":        sig["text"],
                "age_hrs":     sig["age_hrs"],
            }

            if base.get("exclude"):
                impact["exclude"] = True
                impacts[pid] = impact
                continue

            # Scale multiplier toward 1.0 by (1 - conf): e.g. conf=0.5 → half effect
            if "proj_mult" in base:
                raw_mult = base["proj_mult"]
                # Blend: conf*raw_mult + (1-conf)*1.0
                blended = conf * raw_mult + (1.0 - conf) * 1.0
                impact["proj_pts_mult"] = round(blended, 4)

            if "own_mult" in base:
                raw_mult = base["own_mult"]
                blended  = conf * raw_mult + (1.0 - conf) * 1.0
                impact["own_mult"] = round(blended, 4)

            if "own_delta" in base:
                impact["own_delta"] = round(base["own_delta"] * conf, 2)

            impacts[pid] = impact

        return impacts

    # ── Summary ────────────────────────────────────────────────────────────────

    def _build_summary(self, signals: list[dict], impacts: dict) -> dict:
        """Build human-readable summary of news findings."""
        from collections import Counter
        type_counts = Counter(s["signal_type"] for s in signals)
        exclusions  = [pid for pid, imp in impacts.items() if imp.get("exclude")]
        high_conf   = [s for s in signals if s["confidence"] >= 0.70]

        high_priority = [
            f"{s['player_name']} [{s['signal_type']}] ({s['source']}, {s['age_hrs']:.1f}h ago)"
            for s in sorted(signals, key=lambda x: (
                _PRIORITY_MAP.get(x["signal_type"], 0),
                x["confidence"]
            ), reverse=True)[:8]
        ]

        total_excl = len(exclusions)
        return {
            "total_signals":  len(signals),
            "signal_counts":  dict(type_counts),
            "excluded_count": total_excl,
            "high_conf_count": len(high_conf),
            "high_priority":  high_priority,
            "headline": (
                f"{len(signals)} news signals found: "
                f"{total_excl} exclusions (OUT/REST), "
                f"{type_counts.get('STARTING_REPLACEMENT', 0)} starting replacements, "
                f"{type_counts.get('BENCH_DEMOTION', 0)} bench demotions, "
                f"{type_counts.get('MINUTES_RESTRICTION', 0)} minutes restrictions."
            ),
        }

    def _index_by_player(self, signals: list[dict]) -> dict:
        """Return {player_id: [signal_dicts]} for UI display."""
        idx: dict[str, list] = {}
        for s in signals:
            idx.setdefault(s["player_id"], []).append(s)
        return idx
