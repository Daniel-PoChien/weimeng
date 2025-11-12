
from __future__ import annotations
from datetime import date
import pandas as pd

from nba_api.stats.endpoints import (
    scoreboardv3,
    boxscoretraditionalv2,
    playbyplayv2,
)

def games_by_date(date_str: str | None) -> pd.DataFrame:
    """Return a DataFrame of games for YYYY-MM-DD (or today if None)."""
    gd = date_str or date.today().isoformat()
    sb = scoreboardv3.ScoreboardV3(game_date=gd, league_id="00", timeout=30)
    j = sb.get_normalized_dict() or {}
    games = j.get("scoreboard", {}).get("games", []) or []
    if not games:
        return pd.DataFrame()
    df = pd.DataFrame(games)
    # Normalize a few consistent columns your app can use
    keep = {
        "gameId": "GAME_ID",
        "homeTeam.teamId": "HOME_TEAM_ID",
        "awayTeam.teamId": "VISITOR_TEAM_ID",
        "gameStatusText": "GAME_STATUS_TEXT",
        "gameStatus": "GAME_STATUS",
        "gameCode": "GAME_CODE",
    }
    # flatten nested fields
    def pick(row, key):
        if "." in key:
            a, b = key.split(".", 1)
            return (row.get(a) or {}).get(b)
        return row.get(key)

    rows = []
    for g in games:
        out = {}
        for src, dst in keep.items():
            out[dst] = pick(g, src)
        rows.append(out)
    return pd.DataFrame(rows)

def boxscore(game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (players_df, teams_df) for a given game_id using BoxScoreTraditionalV2.
    players_df: one row per player
    teams_df: one row per team (totals)
    """
    bs = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=30)
    data = bs.get_normalized_dict() or {}
    players = pd.DataFrame(data.get("PlayerStats", []))
    teams = pd.DataFrame(data.get("TeamStats", []))
    return players, teams

def clutch_events_last_two_minutes(game_id: str) -> pd.DataFrame:
    """
    Filters play-by-play to last 2 minutes of 4th quarter and any OT periods.
    """
    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=30)
    data = pbp.get_normalized_dict() or {}
    df = pd.DataFrame(data.get("PlayByPlay", []))
    if df.empty:
        return df

    # Keep needed columns
    cols = [c for c in [
        "GAME_ID", "EVENTNUM", "PERIOD", "PCTIMESTRING", "HOMEDESCRIPTION",
        "NEUTRALDESCRIPTION", "VISITORDESCRIPTION", "SCORE", "SCOREMARGIN",
        "PERSON1TYPE", "PLAYER1_NAME", "PLAYER1_TEAM_ID"
    ] if c in df.columns]
    df = df[cols].copy()

    # Parse PCTIMESTRING "MM:SS" into seconds remaining
    def to_secs(s: str) -> int:
        try:
            m, s = str(s).split(":")
            return int(m) * 60 + int(s)
        except Exception:
            return 9999

    df["SECS_LEFT"] = df["PCTIMESTRING"].map(to_secs)

    # 4th (period==4) last 2 minutes OR any OT (period >=5)
    mask = ((df["PERIOD"] == 4) & (df["SECS_LEFT"] <= 120)) | (df["PERIOD"] >= 5)
    return df.loc[mask].reset_index(drop=True)
