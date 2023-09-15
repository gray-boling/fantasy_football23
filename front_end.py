import pandas as pd

from utils import *

stl.title("NFL Fantasy Predictor 2023")


here = os.path.dirname(os.path.abspath(__file__))

sched_url = os.path.join(here, 'full_season.csv')
weekly_url = os.path.join(here, 'weekly_calc_stats.csv')

sched = pd.read_csv(sched_url)
weekly = pd.read_csv(weekly_url)

gb_pipe = load(os.path.join(here, 'GradientBoostingRegressor.joblib'))
lin_reg_pipe = load(os.path.join(here, 'LinearRegression.joblib'))
ran_for_pipe = load(os.path.join(here, 'RandomForestRegressor.joblib'))

infer_df = get_infer_df(sched, weekly)

weekly['week'] = weekly['week'] + 1
drop_cols = [col for col in sched.columns if 'season' not in col and 'week' not in col]
weekly = weekly.drop(drop_cols, axis=1)
weekly = player_sched_join(weekly, infer_df)

infer_df = player_sched_join(weekly, infer_df)

feature_list = ['player_id', 'position', 'week', 'team_year', 'player_year', 'opp_year',
                'weekday', 'away_team', 'home_team', 'spread_line', 'away_spread_odds', 'home_spread_odds', 'player_is_home',
                'total_line', 'under_odds', 'over_odds', 'div_game', 'roof'
                ] + \
                [col for col in weekly.columns if '_avg' in col] + \
                [col for col in weekly.columns if '_std' in col] + \
                [col for col in weekly.columns if '_last' in col] + \
                [col for col in weekly.columns if '_opp' in col]


infer_df['gb_pipe_preds'] = gb_pipe.predict(infer_df[feature_list])
infer_df['lin_reg_pipe_preds'] = lin_reg_pipe.predict(infer_df[feature_list])
infer_df['ran_for_pipe_preds'] = ran_for_pipe.predict(infer_df[feature_list])

infer_df['Projected_PPR_Points'] = (infer_df['gb_pipe_preds'] + infer_df['lin_reg_pipe_preds'] + infer_df['ran_for_pipe_preds'])
infer_df['Projected_PPR_Points'] = (infer_df['Projected_PPR_Points'] / float(3)).round(1)
mse = mean_squared_error(infer_df['fantasy_points_ppr'], infer_df['Projected_PPR_Points'], squared=False)
infer_df['Lowest_Projected_Points'] = (infer_df['Projected_PPR_Points'] - mse).round(1)
infer_df['Lowest_Projected_Points'] = np.where(infer_df['Lowest_Projected_Points'] < 0, 0, infer_df['Lowest_Projected_Points'])
infer_df['Highest_Projected_Points'] = (infer_df['Projected_PPR_Points'] + mse).round(1)


user_input_player = stl.text_input("Search team by city/name in the field below")
if user_input_player:
    per_team = pd.DataFrame(infer_df[(infer_df['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                     (infer_df['recent_team'].str.contains(str(user_input_player.title().upper())))] \
                            [['recent_team', 'player_name', 'Projected_PPR_Points', 'Lowest_Projected_Points', 'Highest_Projected_Points']])
    per_team.reset_index(drop=True, inplace=True).rename(columns={"recent_team": "Team", "player_name": "Player"})
    stl.dataframe(per_team)
else:
    infer_df.sort_values('Projected_PPR_Points', ascending=False) \
        [['player_name', 'Projected_PPR_Points', 'Lowest_Projected_Points', 'Highest_Projected_Points']]
stl.text("")
