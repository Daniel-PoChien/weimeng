"""
live_stats.py

Helper functions for fetching *current* NBA stats from the BallDontLie API.

IMPORTANT:
- BallDontLie now uses /nba/v1/... endpoints for the modern API.
- An API key is required. Set BALDONTLIE_API_KEY in your environment, e.g.:

    export BALDONTLIE_API_KEY="your_api_key_here"

- On the free tier, detailed live season averages may not be available.
  This file handles that gracefully and explains what's going on in plain language.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import requests


# NOTE: Base is the root domain, we include /nba/v1/... in each endpoint.
BALDONTLIE_BASE = "https://api.balldontlie.io"


class LiveStatsError(Exception):
    """Custom error type for live stats failures."""
    pass


def _get_api_key() -> Optional[str]:
    """Read the BallDontLie API key from the environment."""
    return os.getenv("BALDONTLIE_API_KEY")


def _get_headers() -> Dict[str, str]:
    """
    Build the Authorization header in the format expected by BallDontLie.

    Docs: it should be
        Authorization: YOUR_API_KEY
    (NO 'Bearer ' prefix).
    """
    api_key = _get_api_key()
    if not api_key:
        return {}
    return {"Authorization": api_key}


def _request_json(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Helper to call the BallDontLie API and return JSON or raise LiveStatsError.

    endpoint examples:
        "nba/v1/players"
        "nba/v1/games"
    """
    url = f"{BALDONTLIE_BASE}/{endpoint.lstrip('/')}"
    headers = _get_headers()

    try:
        resp = requests.get(url, params=params or {}, headers=headers, timeout=10)

        if resp.status_code == 401:
            if not headers:
                raise LiveStatsError(
                    "401 Unauthorized from BallDontLie – no API key found. "
                    "Set BALDONTLIE_API_KEY in your environment."
                )
            else:
                raise LiveStatsError(
                    "401 Unauthorized from BallDontLie – your API key may be invalid, "
                    "or your account tier does not include this endpoint."
                )

        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise LiveStatsError(f"Error calling BallDontLie API: {e}") from e


def search_player_by_name(name: str) -> Optional[dict]:
    """
    Look up a player by name using the /nba/v1/players endpoint.

    Returns the first matching player dict or None.

    We try:
      1) full name (e.g. "Stephen Curry")
      2) last word only (e.g. "Curry") if full name fails
    """
    # 1) Try full name
    data = _request_json("nba/v1/players", params={"search": name})
    players = data.get("data", [])
    if players:
        return players[0]

    # 2) Try last token only (often last name works best)
    parts = name.strip().split()
    if len(parts) > 1:
        last = parts[-1]
        data2 = _request_json("nba/v1/players", params={"search": last})
        players2 = data2.get("data", [])
        if players2:
            return players2[0]

    return None


def get_player_season_average(
    player_name: str,
    season: int,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Try to fetch season average points for a player.

    NOTE:
    - On a free BallDontLie plan, the live season averages endpoint may not
      be available and might return 401 even with a valid key.
    - In that case, we return a friendly explanation instead of crashing.

    Returns:
        (stats_dict, error_message)

        - stats_dict: {
              "player_name": str,
              "team": str,
              "season": int,
              "pts": float or None,
              "games_played": int or None,
              "raw": dict (full payload),
          } or None

        - error_message: str or None
    """
    # 1) Find player id (this should work on free tier)
    player = None
    try:
        player = search_player_by_name(player_name)
    except LiveStatsError as e:
        # If even /players fails, surface that up directly
        return None, str(e)

    if not player:
        return None, f"Could not find a player named '{player_name}'."

    player_id = player.get("id")
    if not player_id:
        return None, f"Player '{player_name}' has no id in API response."

    team_name = player.get("team", {}).get("full_name", "Unknown Team")

    # 2) Make sure we actually have a key
    headers = _get_headers()
    if not headers:
        return None, (
            "BallDontLie API key not set. "
            "Set BALDONTLIE_API_KEY in your environment to enable live season stats."
        )

    # 3) Call season averages endpoint (modern nba/v1 route)
    url = f"{BALDONTLIE_BASE}/nba/v1/season_averages/general"
    params = {
        "season": season,
        "season_type": "regular",
        "type": "base",
        "category": "general",
        "player_ids[]": player_id,
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)

        if resp.status_code == 401:
            # Most likely: free tier key trying to hit a GOAT-only endpoint
            return None, (
                "Your BallDontLie API key is set, but your account tier does not "
                "include live season averages. Those endpoints require a paid GOAT plan.\n\n"
                "For this demo, the app will still work using the CSV data (Pandas/RAG) "
                "and general NBA knowledge for most questions."
            )

        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return None, f"Error calling BallDontLie season averages endpoint: {e}"

    rows = data.get("data", [])
    if not rows:
        return None, f"No live season average data found for {player_name} in {season}."

    row = rows[0]
    stats = {
        "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
        "team": team_name,
        "season": row.get("season", season),
        "pts": row.get("pts"),
        "games_played": row.get("games_played"),
        "raw": row,
    }
    return stats, None


def format_season_average_human(stats: Dict[str, Any]) -> str:
    """
    Turn the raw stats dict from get_player_season_average into a nice sentence.

    This is what app.py calls to turn the live API result into a chat reply.
    """
    name = stats.get("player_name", "This player")
    team = stats.get("team", "his team")
    season = stats.get("season", "this")
    pts = stats.get("pts")
    games = stats.get("games_played")

    def fmt_num(x):
        return f"{x:.1f}" if isinstance(x, (int, float)) else "N/A"

    pts_str = fmt_num(pts)

    base = (
        f"Based on live stats from the BallDontLie API, "
        f"{name} of the {team} is averaging {pts_str} points per game"
    )

    if isinstance(games, int):
        base += f" over {games} games"

    base += f" in the {season} season."

    return base

