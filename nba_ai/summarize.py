# summarize.py
from __future__ import annotations
import pandas as pd


def top_line(teams: pd.DataFrame) -> str:
    """
    Build 'Winner beat Loser 110–102.' string from team totals.
    """
    req = ["TEAM_NAME", "PTS"]
    if not set(req).issubset(teams.columns):
        return "Summary unavailable."
    t = teams[req].sort_values("PTS", ascending=False).reset_index(drop=True)
    if len(t) < 2:
        return "Summary unavailable."
    w, l = t.iloc[0], t.iloc[1]
    return f"{w.TEAM_NAME} beat {l.TEAM_NAME} {int(w.PTS)}–{int(l.PTS)}."


def leaders(players: pd.DataFrame, n: int = 3) -> str:
    """
    Top N player lines by points with PTS/REB/AST.
    """
    req = ["PLAYER_NAME", "PTS", "REB", "AST"]
    if not set(req).issubset(players.columns):
        return ""
    pts = players.sort_values("PTS", ascending=False).head(n)
    bullets = [
        f"{r['PLAYER_NAME']}: {int(r['PTS'])} PTS, {int(r['REB'])} REB, {int(r['AST'])} AST"
        for _, r in pts.iterrows()
    ]
    return "Top performers: " + "; ".join(bullets) + "." if bullets else ""


def make_summary(players: pd.DataFrame, teams: pd.DataFrame) -> str:
    """
    Overall one-liner + leaders.
    """
    parts = [top_line(teams), leaders(players)]
    return " ".join(p for p in parts if p)
