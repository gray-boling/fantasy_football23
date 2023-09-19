import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from zoneinfo import ZoneInfo
from category_encoders import TargetEncoder, CatBoostEncoder
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import streamlit as stl
import os

def date_getter():
  today = datetime.now(tz=ZoneInfo("America/Denver"))
  today = pd.to_datetime(today, format='%Y-%m-%d').normalize()
  year = today.strftime("%Y")
  return (today, int(year))

def player_sched_join(weekly, sched):
  weekly = weekly \
  .merge(sched,
          left_on=["season", 'week',  "recent_team"],
          right_on=["season", 'week',  "home_team"],
          how="left",
          # suffixes=('_weekly', '_sched')
          ) \
  .merge(sched,
          left_on=["season", 'week', "recent_team"],
          right_on=["season", 'week', "away_team"],
          how="left",
          suffixes=('_home_player', '_away_player')
          )
  for col in sched.columns:
    if col not in weekly.columns:
      weekly[f'{col}_home_player'] = weekly[f'{col}_home_player'].fillna(weekly[f'{col}_away_player'])
      weekly[f'{col}_away_player'] = weekly[f'{col}_away_player'].fillna(weekly[f'{col}_home_player'])
  weekly = weekly.rename(columns={c: c.replace('_home_player', '') for c in weekly.columns}) \
          .rename(columns={c: c.replace('_away_player', '') for c in weekly.columns})

  drop_cols = ['game_id', 'game_type', 'gameday', 'weekday', 'gametime', 'away_team', 'away_score',
              'home_team', 'home_score', 'location', 'result', 'total', 'overtime', 'old_game_id', 'gsis', 'nfl_detail_id',
              'pfr', 'pff', 'espn', 'ftn', 'away_rest', 'home_rest', 'away_moneyline', 'home_moneyline', 'spread_line', 'away_spread_odds',
              'home_spread_odds', 'total_line', 'under_odds', 'over_odds', 'div_game', 'roof', 'surface', 'temp', 'wind',
              'away_qb_id', 'home_qb_id', 'away_qb_name', 'home_qb_name', 'away_coach', 'home_coach', 'referee', 'stadium_id', 'stadium']
  # weekly[drop_cols] = weekly[drop_cols].T.drop_duplicates().T
  weekly = weekly.loc[:,~weekly.columns.duplicated()]

  return weekly

def get_infer_df(sched, weekly):
  today, year = date_getter()[0], date_getter()[1]
  sched['gameday'] = pd.to_datetime(sched['gameday'], utc=True)
  weekly['gameday'] = pd.to_datetime(weekly['gameday'], utc=True)
  if today.strftime("%A") == 'Tuesday':
      date_setter = sched[sched['gameday'] >= pd.to_datetime(today + pd.Timedelta(1, unit="d"))]
      first_game_of_week = date_setter['gameday'].min().normalize()
      last_game_of_week = first_game_of_week + pd.Timedelta(4, unit="d")
  if today.strftime("%A") == 'Wednesday':
    date_setter = sched[sched['gameday'] >= pd.to_datetime(today + pd.Timedelta(1, unit="d"))]
    first_game_of_week = date_setter['gameday'].min().normalize()
    last_game_of_week = first_game_of_week + pd.Timedelta(4, unit="d")
  if today.strftime("%A") == 'Thursday':
    date_setter = sched[sched['gameday'] >= today]
    first_game_of_week = date_setter['gameday'].min().normalize()
    last_game_of_week = first_game_of_week + pd.Timedelta(4, unit="d")
  if today.strftime("%A") == 'Friday':
        date_setter = sched[sched['gameday'] >= pd.to_datetime(today - pd.Timedelta(1, unit="d"))]
        first_game_of_week = date_setter['gameday'].min().normalize()
        last_game_of_week = first_game_of_week + pd.Timedelta(4, unit="d")
  if today.strftime("%A") == 'Saturday':
        date_setter = sched[sched['gameday'] >= pd.to_datetime(today - pd.Timedelta(2, unit="d"))]
        first_game_of_week = date_setter['gameday'].min().normalize()
        last_game_of_week = first_game_of_week + pd.Timedelta(4, unit="d")
  if today.strftime("%A") == 'Sunday':
        date_setter = sched[sched['gameday'] >= pd.to_datetime(today - pd.Timedelta(3, unit="d"))]
        first_game_of_week = date_setter['gameday'].min().normalize()
        last_game_of_week = first_game_of_week + pd.Timedelta(4, unit="d")
  if today.strftime("%A") == 'Monday':
        date_setter = sched[sched['gameday'] >= pd.to_datetime(today - pd.Timedelta(5, unit="d"))]
        first_game_of_week = date_setter['gameday'].min().normalize()
        last_game_of_week = first_game_of_week + pd.Timedelta(4, unit="d")

  infer_df = sched[(sched['season'] == year) & ((sched['gameday'] >= first_game_of_week) & (sched['gameday'] <= last_game_of_week))].copy()
  # infer_df['week'] = infer_df['week'] + 1
  # infer_df = player_sched_join(infer_df,sched)
  return infer_df

def path_to_image_html(path):
    return '<img src="'+ path + '" width="65" >'