# import pandas as pd

from Home import *


#
stl.title("Prediction History")


weekly_dfs = []
directory = os.path.join(here, 'Weekly')
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
      preds_df = pd.read_csv(f)
      weekly_dfs.append(preds_df)
weekly_dfs = pd.concat(weekly_dfs)

weekly_url = os.path.join(here, 'weekly_calc_stats.csv')
weekly = pd.read_csv(weekly_url, encoding='utf-8')


weekly_dfs = weekly_dfs.loc[:, ~weekly_dfs.columns.duplicated()]
weekly_dfs = weekly_dfs.sort_values(['season', 'week'], ascending=True)
weekly_dfs.drop('fantasy_points_ppr', axis=1, inplace=True)
weekly_dfs = pd.merge(weekly_dfs, weekly[['season', 'week', 'player_id', 'fantasy_points_ppr']], on=['season', 'week', 'player_id'], how='left')

weekly_dfs['Actual_PPR_Points_Scored'] = weekly_dfs['fantasy_points_ppr']
weekly_dfs['diff'] =  weekly_dfs['Actual_PPR_Points_Scored'] - weekly_dfs['Projected_PPR_Points']
weekly_dfs['AVG_Difference_in_Points_Scored_vs_Predicted'] = weekly_dfs.groupby('player_id')['diff'].transform('mean')
weekly_dfs['AVG_Difference_in_Points_Scored_vs_Predicted'] = weekly_dfs['AVG_Difference_in_Points_Scored_vs_Predicted'].round(1)

# weekly_dfs.sort_values(['AVG_Difference_in_Points_Scored_vs_Predicted'], ascending=False)[['Team', 'week', 'Opponent',  'Player', 'Projected_PPR_Points', 'Actual_PPR_Points_Scored', 'AVG_Difference_in_Points_Scored_vs_Predicted']]
weekly_dfs = weekly_dfs.rename(columns={"week": "Week"})
weekly_dfs = weekly_dfs.sort_values(['AVG_Difference_in_Points_Scored_vs_Predicted'], ascending=False)

tmp_dfs = weekly_dfs.drop_duplicates('Player')
tmp_dfs['rank'] = range(len(tmp_dfs))
tmp_dfs['rank'] = tmp_dfs['rank'] + 1
rank_dict = dict(zip(tmp_dfs['Player'], tmp_dfs['rank']))
weekly_dfs['rank'] = weekly_dfs['Player'].map(rank_dict)
# weekly_dfs['rank'] = weekly_dfs.groupby('player_id')['rank'].transform('min')
weekly_dfs.set_index('rank', inplace=True)

stl.write('Negative numbers represent a player under-performing his predicted points.'
          ' Positive numbers represent an over-performance.')
week_option = stl.selectbox('Look at specific week', (None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
if week_option:
    stl.dataframe(weekly_dfs[weekly_dfs['Week'] == week_option].sort_values(['AVG_Difference_in_Points_Scored_vs_Predicted'], ascending=False)[
                      ['Team', 'Week', 'Player', 'Projected_PPR_Points', 'Actual_PPR_Points_Scored',
                       'AVG_Difference_in_Points_Scored_vs_Predicted']]
                  )
else:
    stl.dataframe(weekly_dfs.sort_values(['AVG_Difference_in_Points_Scored_vs_Predicted'], ascending=False)[
                      ['Team', 'Week', 'Player', 'Projected_PPR_Points', 'Actual_PPR_Points_Scored',
                       'AVG_Difference_in_Points_Scored_vs_Predicted']]
                  )