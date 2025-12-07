import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster, leaguegamefinder

def get_data():
    print("--- 1. Fetching Teams ---")
    nba_teams = teams.get_teams()
    df_teams = pd.DataFrame(nba_teams)
    df_teams.to_csv("nba_teams.csv", index=False)
    print("Saved nba_teams.csv")

    print("--- 2. Fetching Rosters (Please wait ~1 min) ---")
    all_rosters = []
    # We will just fetch the first 10 teams for the demo to save time. 
    # Remove [:10] to get ALL 30 teams for the final project.
    for team in nba_teams[:10]: 
        try:
            time.sleep(0.6) # Sleep to prevent API timeout
            roster = commonteamroster.CommonTeamRoster(team_id=team['id'], season='2024-25')
            df = roster.get_data_frames()[0]
            all_rosters.append(df)
            print(f"Got roster: {team['full_name']}")
        except Exception as e:
            print(f"Skipped {team['full_name']}: {e}")
    
    if all_rosters:
        pd.concat(all_rosters).to_csv("nba_rosters.csv", index=False)
        print("Saved nba_rosters.csv")

    print("--- 3. Fetching 2024-25 Game Scores ---")
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2024-25', league_id_nullable='00')
    games = gamefinder.get_data_frames()[0]
    # Filter for key columns to keep data clean
    games = games[['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL', 'PTS', 'PLUS_MINUS']]
    games.to_csv("nba_scores.csv", index=False)
    print("Saved nba_scores.csv")
    print("Data Collection Complete!")

if __name__ == "__main__":
    get_data()