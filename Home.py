# import pandas as pd

from utils import *
#
# stl.title("NFL Fantasy Predictor 2023")

stl.set_page_config(
    page_title="NFL Fantasy Predictor 2023",
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "This is a NFL DFS projection tool. Make your own lineup with this system's help!"
    }
)
stl.title("NFL Fantasy Predictor 2023")
here = os.path.dirname(os.path.abspath(__file__))

# SECRET_KEY = os.environ["client_secret"]
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SECRET_KEY
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(here, "fantasy-nfl-2023-051c047c2915.json")
# client = storage.Client()
# bucket_name = 'cloud-ai-platform-cf9cca39-5f3b-4465-b28a-64ee11959e55'
# bucket = client.get_bucket(bucket_name)
workload_identity_provider = stl.secrets["GCS_PROVIDER"]
service_account = stl.secrets["GCS_SERVICE"]


infer_df_url = 'gs://cloud-ai-platform-cf9cca39-5f3b-4465-b28a-64ee11959e55/datasets/infer_df.csv'
# infer_df_url = os.path.join(here, 'infer_df.csv')
#
infer_df = pd.read_csv(infer_df_url, encoding='utf-8')

# #get week from schedule
week = infer_df['week'].values[0]

infer_df.dropna(subset=['opp'], inplace=True)
infer_df = infer_df.rename(columns={"recent_team": "Team", "player_name": "Player", 'opp': 'Opponent',
                                    'away_team': 'Away Team', 'home_team': 'Home Team'})

weekly_folder = os.path.join(here, 'Weekly')
if not os.path.exists(weekly_folder):
    os.mkdir(weekly_folder)
if not os.path.isfile(os.path.join(weekly_folder, f'week{week}_df.csv')):
    infer_df.to_csv(os.path.join(weekly_folder, f'week{week}_df.csv'))

weekly_files = [os.path.join(here, csv) for csv in weekly_folder]

#add weekly comparision df here

checks = stl.columns(4)
with checks[0]:
    stl.checkbox('Sort RBs', key='RB')
with checks[1]:
    stl.checkbox('Sort QBs', key='QB')
with checks[2]:
    stl.checkbox('Sort WRs', key='WR')
with checks[3]:
    stl.checkbox('Sort TEs', key='TE')

user_input_player = stl.text_input("Filter team by city/name in the field below")
if user_input_player:
    per_team = pd.DataFrame(infer_df[(infer_df['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                     (infer_df['Team'].str.contains(str(user_input_player.title().upper())))] \
                            [['Team', 'Opponent', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points', 'Highest_Projected_Points']])
    # per_team = pd.DataFrame(infer_df[(infer_df['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
    #                                  (infer_df['recent_team'].str.contains(str(user_input_player.title().upper())))] \
    #                         [feature_list])
    per_team = per_team.reset_index(drop=True)
    stl.dataframe(per_team)
else:

    # rb = stl.checkbox('Sort RBs')
    if stl.session_state['RB']:
        per_rb = pd.DataFrame(infer_df[(infer_df['position'] == 'RB')])
        per_rb = per_rb.sort_values('Projected_PPR_Points', ascending=False)
        per_rb = pd.DataFrame(per_rb[(per_rb['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                         (per_rb['Team'].str.contains(str(user_input_player.title().upper())))] \
                                    [['Team', 'Opponent', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points',
                                      'Highest_Projected_Points']])
        per_rb['rank'] = range(len(per_rb))
        per_rb['rank'] = per_rb['rank'] + 1
        per_rb.set_index('rank', inplace=True)
        stl.dataframe(per_rb)

    # qb = stl.checkbox('Sort QBs', key='qb')
    if stl.session_state['QB']:
        per_qb = pd.DataFrame(infer_df[(infer_df['position'] == 'QB')])
        per_qb = per_qb.sort_values('Projected_PPR_Points', ascending=False)
        per_qb = pd.DataFrame(per_qb[(per_qb['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                         (per_qb['Team'].str.contains(str(user_input_player.title().upper())))] \
                                    [['Team', 'Opponent', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points',
                                      'Highest_Projected_Points']])
        per_qb['rank'] = range(len(per_qb))
        per_qb['rank'] = per_qb['rank'] + 1
        per_qb.set_index('rank', inplace=True)
        stl.dataframe(per_qb)

    # wr = stl.checkbox('Sort WRs', key='wr')
    if stl.session_state['WR']:
        per_wr = pd.DataFrame(infer_df[(infer_df['position'] == 'WR')])
        per_wr = per_wr.sort_values('Projected_PPR_Points', ascending=False)
        per_wr = pd.DataFrame(per_wr[(per_wr['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                         (per_wr['Team'].str.contains(str(user_input_player.title().upper())))] \
                                    [['Team', 'Opponent', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points',
                                      'Highest_Projected_Points']])
        per_wr['rank'] = range(len(per_wr))
        per_wr['rank'] = per_wr['rank'] + 1
        per_wr.set_index('rank', inplace=True)
        stl.dataframe(per_wr)

    # te = stl.checkbox('Sort TEs', key='te')
    if stl.session_state['TE']:
        per_te = pd.DataFrame(infer_df[(infer_df['position'] == 'TE')])
        per_te = per_te.sort_values('Projected_PPR_Points', ascending=False)
        per_te = pd.DataFrame(per_te[(per_te['team_full_name'].str.contains(str(user_input_player.title().upper()))) | \
                                         (per_te['Team'].str.contains(str(user_input_player.title().upper())))] \
                                    [['Team', 'Opponent', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points',
                                      'Highest_Projected_Points']])
        per_te['rank'] = range(len(per_te))
        per_te['rank'] = per_te['rank'] + 1
        per_te.set_index('rank', inplace=True)
        stl.dataframe(per_te)
    if not ((stl.session_state['RB']) | (stl.session_state['QB']) | (stl.session_state['WR']) | (stl.session_state['TE'])):
        infer_df = infer_df.sort_values('Projected_PPR_Points', ascending=False) \
            [['Team', 'Opponent', 'Player', 'Projected_PPR_Points', 'Lowest_Projected_Points', 'Highest_Projected_Points']]
        infer_df['rank'] = range(len(infer_df))
        infer_df['rank'] = infer_df['rank'] + 1
        infer_df.set_index('rank', inplace=True)
        stl.dataframe(infer_df)
    stl.text("")
stl.text("")