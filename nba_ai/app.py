# app.py  (merged + hardened)
from __future__ import annotations
from typing import Optional
from datetime import date, datetime
import math
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ---- your helpers (new version) ----
from fetch import games_by_date, boxscore, clutch_events_last_two_minutes
from summarize import make_summary

# ---- nba_api imports (for legacy endpoints) ----
from nba_api.stats.static import players as static_players
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import scoreboardv3, leaguedashteamstats, playergamelog

# some builds have LeagueGameLog, some don't
try:
    from nba_api.stats.endpoints import leaguegamelog
    HAS_LEAGUEGAMELOG = True
except Exception:
    leaguegamelog = None
    HAS_LEAGUEGAMELOG = False

# ----------------- FastAPI -----------------
app = FastAPI(title="NBA Agent", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- small utils -----------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def get_team_dict() -> dict[int, dict]:
    return {t["id"]: {"full_name": t["full_name"], "abbreviation": t["abbreviation"]}
            for t in static_teams.get_teams()}

def season_net_rating_map(season: str) -> dict[int, float]:
    """TEAM_ID -> season NET_RATING (Advanced)"""
    stat = leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Advanced", timeout=30
    ).get_normalized_dict()
    df = pd.DataFrame(stat.get("LeagueDashTeamStats", []))
    if df.empty:
        return {}
    df = df[["TEAM_ID", "NET_RATING"]].copy()
    df["NET_RATING"] = pd.to_numeric(df["NET_RATING"], errors="coerce").fillna(0.0)
    return {int(r.TEAM_ID): float(r.NET_RATING) for r in df.itertuples()}

def last10_form_via_leaguegamelog(team_id: int, season: str) -> tuple[float, float]:
    """
    Compute (win_pct_last10, mov_last10) using LeagueGameLog if available.
    If not available, return neutral values (0.5, 0.0).
    """
    if not HAS_LEAGUEGAMELOG:
        return 0.5, 0.0

    resp = leaguegamelog.LeagueGameLog(
        season=season, season_type_all_star="Regular Season", timeout=30
    ).get_normalized_dict()
    df = pd.DataFrame(resp.get("LeagueGameLog", []))
    if df.empty or "TEAM_ID" not in df.columns:
        return 0.5, 0.0

    tdf = df[df["TEAM_ID"] == team_id].copy()
    if "GAME_DATE" in tdf.columns:
        try:
            tdf["GAME_DATE_PARSED"] = pd.to_datetime(tdf["GAME_DATE"])
            tdf = tdf.sort_values("GAME_DATE_PARSED", ascending=False)
        except Exception:
            pass

    tdf = tdf.head(10)
    if tdf.empty:
        return 0.5, 0.0

    wins = (tdf["WL"] == "W").sum() if "WL" in tdf.columns else 0
    wp = wins / max(len(tdf), 1)
    mov = safe_float(tdf.get("PLUS_MINUS", pd.Series([0]*len(tdf))).mean(), 0.0)
    return wp, mov

def _yyyymmdd(iso_str: str) -> str:
    d = datetime.strptime(iso_str, "%Y-%m-%d").date()
    return f"{d.year}{d.month:02d}{d.day:02d}"

# ----------------- basic health -----------------
@app.get("/health")
def health():
    return {"ok": True}

# ----------------- YOUR NEW ROUTES (kept) -----------------
@app.get("/games")
def list_games(date_str: Optional[str] = Query(None, description="YYYY-MM-DD")):
    try:
        df = games_by_date(date_str)
    except Exception as e:
        raise HTTPException(502, f"Failed to load games: {e}")
    if df is None or df.empty:
        raise HTTPException(404, "No games found for that date.")
    return df.to_dict(orient="records")

@app.get("/game-summary")
def game_summary(game_id: str = Query(..., description="NBA game id, e.g., 0022400001")):
    try:
        players, teams = boxscore(game_id)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch box score: {e}")
    if players.empty or teams.empty:
        raise HTTPException(503, "Box score not available yet. Try again later.")
    return {"game_id": game_id, "summary": make_summary(players, teams)}

@app.get("/pbp-clutch")
def pbp_clutch(game_id: str = Query(..., description="NBA game id")):
    """Last 2 minutes (4th + any OT) clutch events."""
    try:
        df = clutch_events_last_two_minutes(game_id)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch play-by-play: {e}")
    if df is None or df.empty:
        raise HTTPException(404, "No clutch events found (check game_id or game not finished).")
    return df.to_dict(orient="records")

@app.get("/daily-summaries")
def daily_summaries(date_str: str = Query(..., description="YYYY-MM-DD")):
    """Return quick summaries for all games on a date."""
    try:
        games = games_by_date(date_str)
    except Exception as e:
        raise HTTPException(502, f"Failed to load games: {e}")
    if games.empty:
        raise HTTPException(404, f"No games on {date_str}.")

    out = []
    for _, row in games.iterrows():
        gid = row["GAME_ID"]
        try:
            players, teams = boxscore(gid)
            if not players.empty and not teams.empty:
                out.append({"game_id": gid, "summary": make_summary(players, teams)})
        except Exception:
            continue
    if not out:
        raise HTTPException(503, "Box scores not available yet for these games.")
    return out

@app.get("/scoreboard")
def scoreboard(game_date: str | None = None):
    """
    Tries stats.nba.com (ScoreboardV3), then falls back to data.nba.net by date.
    Returns a consistent shape: {"source", "game_date", "games": [...]}
    """
    gd = game_date or date.today().isoformat()

    # Try V3 first
    try:
        sb = scoreboardv3.ScoreboardV3(game_date=gd, league_id="00", timeout=30)
        v3 = sb.get_normalized_dict() or {}
        games_v3 = v3.get("scoreboard", {}).get("games") or []
        if isinstance(games_v3, list) and games_v3:
            return {"source": "ScoreboardV3", "game_date": gd, "games": games_v3}
    except Exception:
        pass

    # Fallback to public data.nba.net (reliable by date)
    try:
        ymd = _yyyymmdd(gd)
        url = f"https://data.nba.net/prod/v2/{ymd}/scoreboard.json"
        r = requests.get(url, timeout=20)
        if r.ok:
            j = r.json()
            games = j.get("games", [])
            norm_games = [{
                "GAME_ID": g.get("gameId"),
                "GAME_DATE_EST": gd,
                "HOME_TEAM_ID": g.get("hTeam", {}).get("teamId"),
                "VISITOR_TEAM_ID": g.get("vTeam", {}).get("teamId"),
                "HOME_SCORE": g.get("hTeam", {}).get("score"),
                "VISITOR_SCORE": g.get("vTeam", {}).get("score"),
                "GAME_STATUS_TEXT": g.get("statusNum"),
            } for g in games]
            if norm_games:
                return {"source": "data.nba.net", "game_date": gd, "games": norm_games}
    except Exception:
        pass

    return {"source": "none", "game_date": gd, "games": [], "note": "No scoreboard data returned."}

@app.get("/teams")
def teams():
    return [
        {"team_id": t["id"], "full_name": t["full_name"], "abbreviation": t["abbreviation"]}
        for t in static_teams.get_teams()
    ]

@app.get("/team_matchup")
def team_matchup(
    home_team_id: int = Query(..., description="e.g., 1610612744"),
    away_team_id: int = Query(..., description="e.g., 1610612747"),
    season: str = Query("2024-25")
):
    teams_by_id = get_team_dict()
    if home_team_id not in teams_by_id or away_team_id not in teams_by_id:
        raise HTTPException(400, "Unknown team id. Use /teams to look up valid IDs.")

    season = (season or "2024-25").strip().strip('"').strip("'")
    season_nr = season_net_rating_map(season)
    home_season_nr = season_nr.get(home_team_id, 0.0)
    away_season_nr = season_nr.get(away_team_id, 0.0)

    home_wp10, home_mov10 = last10_form_via_leaguegamelog(home_team_id, season)
    away_wp10, away_mov10 = last10_form_via_leaguegamelog(away_team_id, season)

    x = (
        0.35 * (home_season_nr - away_season_nr) +
        0.40 * (home_mov10 - away_mov10) +
        0.25 * (home_wp10 - away_wp10) +
        0.15 * 1.0  # small home bump
    )
    p_home = sigmoid(x)

    return {
        "home": {
            "team_id": home_team_id,
            "name": teams_by_id[home_team_id]["full_name"],
            "abbr": teams_by_id[home_team_id]["abbreviation"],
            "season_net_rating": round(home_season_nr, 3),
            "last10_win_pct": round(home_wp10, 3),
            "last10_mov": round(home_mov10, 3),
        },
        "away": {
            "team_id": away_team_id,
            "name": teams_by_id[away_team_id]["full_name"],
            "abbr": teams_by_id[away_team_id]["abbreviation"],
            "season_net_rating": round(away_season_nr, 3),
            "last10_win_pct": round(away_wp10, 3),
            "last10_mov": round(away_mov10, 3),
        },
        "win_probability_home": round(p_home, 3),
        "season": season,
        "last10_source": "LeagueGameLog" if HAS_LEAGUEGAMELOG else "fallback-neutral"
    }

@app.get("/player_compare")
def player_compare(
    player1_name: str = Query(..., description="e.g., Stephen Curry"),
    player2_name: str = Query(..., description="e.g., Damian Lillard"),
    season: str = Query("2024-25")
):
    plist = static_players.get_players()

    def find_id_by_name(name: str):
        if not name:
            return None
        exact = [p for p in plist if p.get("full_name", "").lower() == name.lower()]
        if exact:
            return exact[0]["id"]
        partial = [p for p in plist if name.lower() in p.get("full_name", "").lower()]
        return partial[0]["id"] if partial else None

    p1 = find_id_by_name(player1_name)
    p2 = find_id_by_name(player2_name)
    if not p1 or not p2:
        raise HTTPException(400, "Player name not found. Try full name, e.g., 'Stephen Curry'.")

    resp1 = playergamelog.PlayerGameLog(player_id=p1, season=season, timeout=30).get_normalized_dict()
    resp2 = playergamelog.PlayerGameLog(player_id=p2, season=season, timeout=30).get_normalized_dict()
    d1 = pd.DataFrame(resp1.get("PlayerGameLog", []))
    d2 = pd.DataFrame(resp2.get("PlayerGameLog", []))

    def last5(df: pd.DataFrame):
        if df.empty:
            return {"PPG": 0.0, "RPG": 0.0, "APG": 0.0}
        df = df.head(5)
        return {
            "PPG": round(safe_float(df["PTS"].mean(), 0.0), 3),
            "RPG": round(safe_float(df["REB"].mean(), 0.0), 3),
            "APG": round(safe_float(df["AST"].mean(), 0.0), 3),
        }

    p1s, p2s = last5(d1), last5(d2)
    score = (p1s["PPG"] - p2s["PPG"]) + 0.3 * (p1s["APG"] - p2s["APG"])
    favored = player1_name if score > 0 else player2_name

    return {
        "player1": {"name": player1_name, "id": p1, **p1s},
        "player2": {"name": player2_name, "id": p2, **p2s},
        "favored_by_simple_form": favored,
        "season": season
    }
