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
infer_df = infer_df.rename(columns={"recent_team": "Team", "player_name": "Player"})

user_input_player = stl.text_input("Search team by city/name in the field below")
if user_input_player:
    per_team = pd.DataFrame(infer_df[(infer_df['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                     (infer_df['Team'].str.contains(str(user_input_player.title().upper())))] \
                            [['Team', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points', 'Highest_Projected_Points']])
    per_team = per_team.reset_index()
    stl.dataframe(per_team)

checks = stl.columns(4)
with checks[0]:
    stl.checkbox('Sort RBs', key='RB')
with checks[1]:
    stl.checkbox('Sort QBs', key='QB')
with checks[2]:
    stl.checkbox('Sort WRs', key='WR')
with checks[3]:
    stl.checkbox('Sort TEs', key='TE')

# rb = stl.checkbox('Sort RBs')
if stl.session_state['RB']:
    per_rb = pd.DataFrame(infer_df[(infer_df['position'] == 'RB')])
    per_rb = per_rb.sort_values('Projected_PPR_Points', ascending=False)
    per_rb = pd.DataFrame(per_rb[(per_rb['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                     (per_rb['Team'].str.contains(str(user_input_player.title().upper())))] \
                                [['Team', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points',
                                  'Highest_Projected_Points']])
    per_rb['rank'] = range(len(per_rb))
    per_rb.set_index('rank', inplace=True)
    stl.dataframe(per_rb)

# qb = stl.checkbox('Sort QBs', key='qb')
if stl.session_state['QB']:
    per_qb = pd.DataFrame(infer_df[(infer_df['position'] == 'QB')])
    per_qb = per_qb.sort_values('Projected_PPR_Points', ascending=False)
    per_qb = pd.DataFrame(per_qb[(per_qb['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                     (per_qb['Team'].str.contains(str(user_input_player.title().upper())))] \
                                [['Team', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points',
                                  'Highest_Projected_Points']])
    per_qb['rank'] = range(len(per_qb))
    per_qb.set_index('rank', inplace=True)
    stl.dataframe(per_qb)

# wr = stl.checkbox('Sort WRs', key='wr')
if stl.session_state['WR']:
    per_wr = pd.DataFrame(infer_df[(infer_df['position'] == 'WR')])
    per_wr = per_wr.sort_values('Projected_PPR_Points', ascending=False)
    per_wr = pd.DataFrame(per_wr[(per_wr['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                     (per_wr['Team'].str.contains(str(user_input_player.title().upper())))] \
                                [['Team', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points',
                                  'Highest_Projected_Points']])
    per_wr['rank'] = range(len(per_wr))
    per_wr.set_index('rank', inplace=True)
    stl.dataframe(per_wr)

# te = stl.checkbox('Sort TEs', key='te')
if stl.session_state['TE']:
    per_te = pd.DataFrame(infer_df[(infer_df['position'] == 'TE')])
    per_te = per_te.sort_values('Projected_PPR_Points', ascending=False)
    per_te = pd.DataFrame(per_te[(per_te['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                     (per_te['Team'].str.contains(str(user_input_player.title().upper())))] \
                                [['Team', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points',
                                  'Highest_Projected_Points']])
    per_te['rank'] = range(len(per_te))
    per_te.set_index('rank', inplace=True)
    stl.dataframe(per_te)
if not ((stl.session_state['RB']) | (stl.session_state['QB']) | (stl.session_state['WR']) | (stl.session_state['TE'])):
    infer_df.sort_values('Projected_PPR_Points', ascending=False) \
        [['Team', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points', 'Highest_Projected_Points']]
stl.text("")
