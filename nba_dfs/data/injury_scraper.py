"""
Injury & news scraper.
Sources (free):
  - NBA.com official injury report PDF
  - Rotowire injury page
  - ESPN injury page
  - CBSSports
  - RotoGrinders news feed
"""

import re
import time
import asyncio
from datetime import datetime, date
from typing import Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import SCRAPER_DELAY_SECS


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

STATUS_MAP = {
    "out":          "OUT",
    "questionable": "QUESTIONABLE",
    "doubtful":     "DOUBTFUL",
    "probable":     "PROBABLE",
    "gtd":          "GTD",
    "day-to-day":   "GTD",
    "active":       "ACTIVE",
    "available":    "ACTIVE",
}

def _normalize_status(raw: str) -> str:
    return STATUS_MAP.get(raw.strip().lower(), raw.upper())


class InjuryScraper:
    def __init__(self):
        self.client = httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True)

    def close(self):
        self.client.close()

    # ── Rotowire ───────────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def scrape_rotowire(self) -> list[dict]:
        url = "https://www.rotowire.com/basketball/injury-report.php"
        resp = self.client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        records = []
        for row in soup.select("div.injury-report-player"):
            try:
                name   = row.select_one(".injury-report-player__name")
                team   = row.select_one(".injury-report-player__team")
                status = row.select_one(".injury-report-player__status")
                reason = row.select_one(".injury-report-player__injury")

                if not name:
                    continue
                records.append({
                    "name":        name.get_text(strip=True),
                    "team":        team.get_text(strip=True) if team else "",
                    "status":      _normalize_status(status.get_text(strip=True)) if status else "UNKNOWN",
                    "reason":      reason.get_text(strip=True) if reason else "",
                    "source":      "rotowire",
                    "reported_at": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.warning(f"Rotowire row parse error: {e}")

        time.sleep(SCRAPER_DELAY_SECS)
        logger.info(f"Rotowire: {len(records)} injury records")
        return records

    # ── ESPN ───────────────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def scrape_espn(self) -> list[dict]:
        url = "https://www.espn.com/nba/injuries"
        resp = self.client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        records = []
        for table in soup.select("div.ResponsiveTable"):
            team_el = table.select_one("div.Table__Title")
            team_name = team_el.get_text(strip=True) if team_el else "Unknown"
            for row in table.select("tr.Table__TR")[1:]:
                cols = row.select("td")
                if len(cols) < 4:
                    continue
                try:
                    records.append({
                        "name":        cols[0].get_text(strip=True),
                        "team":        team_name,
                        "status":      _normalize_status(cols[2].get_text(strip=True)),
                        "reason":      cols[1].get_text(strip=True),
                        "source":      "espn",
                        "reported_at": datetime.utcnow().isoformat(),
                    })
                except Exception as e:
                    logger.warning(f"ESPN row parse error: {e}")

        time.sleep(SCRAPER_DELAY_SECS)
        logger.info(f"ESPN: {len(records)} injury records")
        return records

    # ── RotoGrinders news feed ─────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def scrape_rotogrinders_news(self) -> list[dict]:
        url = "https://rotogrinders.com/lineups/nba"
        resp = self.client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        records = []
        for card in soup.select(".player-news-card, .news-item"):
            try:
                name_el   = card.select_one(".player-name, .name")
                status_el = card.select_one(".status-tag, .status")
                text_el   = card.select_one(".news-text, p")
                if not name_el:
                    continue
                records.append({
                    "name":        name_el.get_text(strip=True),
                    "team":        "",
                    "status":      _normalize_status(status_el.get_text(strip=True)) if status_el else "UNKNOWN",
                    "reason":      text_el.get_text(strip=True) if text_el else "",
                    "source":      "rotogrinders",
                    "reported_at": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.warning(f"RG news parse error: {e}")

        time.sleep(SCRAPER_DELAY_SECS)
        logger.info(f"RotoGrinders: {len(records)} news records")
        return records

    # ── CBSSports ─────────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def scrape_cbssports(self) -> list[dict]:
        url = "https://www.cbssports.com/nba/injuries/"
        resp = self.client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        records = []
        for section in soup.select("div.TableBaseWrapper"):
            team_el = section.select_one("span.TeamName")
            team_name = team_el.get_text(strip=True) if team_el else ""
            for row in section.select("tbody tr"):
                cols = row.select("td")
                if len(cols) < 3:
                    continue
                try:
                    records.append({
                        "name":        cols[0].get_text(strip=True),
                        "team":        team_name,
                        "status":      _normalize_status(cols[2].get_text(strip=True)),
                        "reason":      cols[1].get_text(strip=True),
                        "source":      "cbssports",
                        "reported_at": datetime.utcnow().isoformat(),
                    })
                except Exception as e:
                    logger.warning(f"CBS Sports row parse: {e}")

        time.sleep(SCRAPER_DELAY_SECS)
        logger.info(f"CBSSports: {len(records)} injury records")
        return records

    # ── Aggregate all sources ─────────────────────────────────────────────────
    def get_all_injury_data(self) -> pd.DataFrame:
        """Fetch from all free sources and de-duplicate by player name."""
        all_records: list[dict] = []

        for fn in (self.scrape_espn, self.scrape_rotowire, self.scrape_cbssports):
            try:
                all_records.extend(fn())
            except Exception as e:
                logger.error(f"{fn.__name__} failed: {e}")

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # Priority: OUT > DOUBTFUL > QUESTIONABLE > GTD > PROBABLE > ACTIVE
        priority = {"OUT": 0, "DOUBTFUL": 1, "QUESTIONABLE": 2, "GTD": 3, "PROBABLE": 4, "ACTIVE": 5, "UNKNOWN": 6}
        df["priority"] = df["status"].map(priority).fillna(6)
        df = df.sort_values("priority")
        df = df.drop_duplicates(subset=["name"], keep="first")
        df = df.drop(columns=["priority"])

        logger.success(f"Total unique injury records: {len(df)}")
        return df.reset_index(drop=True)

    # ── Lineup confirmation scraper ────────────────────────────────────────────
    def scrape_lineup_confirmations(self) -> list[dict]:
        """
        Scrape RotoGrinders projected lineups page for confirmed starters.
        Returns list of {name, team, position, confirmed, batting_order_proxy}.
        """
        url = "https://rotogrinders.com/lineups/nba"
        try:
            resp = self.client.get(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            lineups = []
            for team_block in soup.select(".lineup"):
                team_el = team_block.select_one(".team-name, .abbrev")
                team = team_el.get_text(strip=True) if team_el else ""
                for slot in team_block.select(".player-slot, .lineup-player"):
                    name_el = slot.select_one(".player-name, .name")
                    pos_el  = slot.select_one(".position, .pos")
                    confirmed = "confirmed" in slot.get("class", [])
                    if name_el:
                        lineups.append({
                            "name":      name_el.get_text(strip=True),
                            "team":      team,
                            "position":  pos_el.get_text(strip=True) if pos_el else "",
                            "confirmed": confirmed,
                        })
            time.sleep(SCRAPER_DELAY_SECS)
            return lineups
        except Exception as e:
            logger.error(f"Lineup scrape failed: {e}")
            return []

    # ── Vegas lines ────────────────────────────────────────────────────────────
    def get_vegas_lines(self, api_key: str = "") -> pd.DataFrame:
        """
        Pull NBA game totals and spreads from The-Odds-API (free tier: 500 req/month).
        Get a key at https://the-odds-api.com
        """
        if not api_key:
            logger.warning("No Odds API key - skipping Vegas lines")
            return pd.DataFrame()

        url = (
            f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
            f"?apiKey={api_key}&regions=us&markets=h2h,spreads,totals&oddsFormat=american"
        )
        try:
            resp = self.client.get(url)
            resp.raise_for_status()
            data = resp.json()
            records = []
            for game in data:
                for bk in game.get("bookmakers", [])[:1]:  # use first bookmaker
                    row = {
                        "game_id":    game["id"],
                        "slate_date": game["commence_time"][:10],
                        "home_team":  game["home_team"],
                        "away_team":  game["away_team"],
                        "bookmaker":  bk["key"],
                        "updated_at": datetime.utcnow().isoformat(),
                        "total":      None,
                        "home_spread": None,
                        "home_ml":    None,
                        "away_ml":    None,
                    }
                    for mkt in bk.get("markets", []):
                        if mkt["key"] == "totals":
                            for o in mkt["outcomes"]:
                                if o["name"] == "Over":
                                    row["total"] = o["point"]
                        elif mkt["key"] == "spreads":
                            for o in mkt["outcomes"]:
                                if o["name"] == game["home_team"]:
                                    row["home_spread"] = o["point"]
                        elif mkt["key"] == "h2h":
                            for o in mkt["outcomes"]:
                                if o["name"] == game["home_team"]:
                                    row["home_ml"] = o["price"]
                                else:
                                    row["away_ml"] = o["price"]
                    records.append(row)
            return pd.DataFrame(records)
        except Exception as e:
            logger.error(f"Vegas lines fetch failed: {e}")
            return pd.DataFrame()
