# import pandas as pd

from front_end import *

# Specify what pages should be shown in the sidebar, and what their titles
# and icons should be
show_pages(
    [
        Page("front_end.py", "Main Page"),
        Page(os.path.join(here, 'pages/Historical Predictions.py'), "Historical Predictions")
    ]
)
#
# stl.title("NFL Fantasy Predictor 2023")

#
# here = os.path.dirname(os.path.abspath(__file__))
# #
# # SECRET_KEY = os.environ["client_secret"]
# #
# # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SECRET_KEY
#
# # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(here, "fantasy-nfl-2023-051c047c2915.json")
# #
# # client = storage.Client()
# bucket_name = 'cloud-ai-platform-cf9cca39-5f3b-4465-b28a-64ee11959e55'
# # bucket = client.get_bucket(bucket_name)
# #
# infer_df_url = os.path.join(here, 'infer_df.csv')
# #
# infer_df = pd.read_csv(infer_df_url, encoding='utf-8')

weekly_dfs = []
directory = os.path.join(here, 'Weekly')
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
      preds_df = pd.read_csv(f)
      weekly_dfs.append(preds_df)
weekly_dfs = pd.concat(weekly_dfs)

weekly_dfs = weekly_dfs.loc[:, ~weekly_dfs.columns.duplicated()]
weekly_dfs = weekly_dfs.sort_values(['season', 'week'], ascending=True)
weekly_dfs['Actual_PPR_Points_Scored'] = weekly_dfs.groupby('player_id')['fantasy_points_ppr'].shift(-1)
weekly_dfs['diff'] = weekly_dfs['Actual_PPR_Points_Scored'] - weekly_dfs['Projected_PPR_Points']
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
week_option = stl.selectbox('Look at specific week', (None, 2, 3, 4, 5, 6, 7, 8, 9, 10))
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