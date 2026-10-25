"""Reproducible AIS analysis. Run with python -m src.pipeline --input data/ais.csv."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

KNOT_KM_H = 1.852

def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0088
    a1, a2 = np.radians(lat1), np.radians(lat2)
    dl = a2-a1
    dn = np.radians(lon2-lon1)
    h = np.sin(dl/2)**2 + np.cos(a1)*np.cos(a2)*np.sin(dn/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(h,0,1)))

def load_and_clean(path):
    df = pd.read_csv(path, dtype={'mmsi':'string','imo':'string','ship_id':'string'},low_memory=False)
    required = ['mmsi','timestamp','latitude','longitude','speed','course','heading']
    absent = sorted(set(required)-set(df.columns))
    if absent: raise ValueError(f'Missing required columns: {absent}')
    df['timestamp'] = pd.to_datetime(df.timestamp,utc=True,errors='coerce')
    for c in ['latitude','longitude','speed','course','heading','length','width','draught']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    before=len(df)
    df=df.drop_duplicates(subset=['mmsi','timestamp','latitude','longitude']).dropna(subset=['mmsi','timestamp','latitude','longitude'])
    df=df.loc[df.latitude.between(-90,90)&df.longitude.between(-180,180)].copy()
    # Raw AIS units: SOG in tenths of knots, draught in tenths of metres.
    df['speed_knots']=df.speed/10
    df.loc[~df.speed_knots.between(0,102.2),'speed_knots']=np.nan
    df['heading_deg']=df.heading.where(df.heading.between(0,359))
    df['course_deg']=df.course.where(df.course.between(0,359.9))
    if 'draught' in df: df['draught_m']=df.draught/10
    df=df.sort_values(['mmsi','timestamp']).reset_index(drop=True)
    return df, {'raw_rows':int(before),'clean_rows':int(len(df)),'removed_rows':int(before-len(df))}

def trajectories(df):
    x=df.copy(); g=x.groupby('mmsi',sort=False)
    x['gap_minutes']=g.timestamp.diff().dt.total_seconds()/60
    x['prev_lat']=g.latitude.shift(1);x['prev_lon']=g.longitude.shift(1)
    x['distance_km']=haversine(x.prev_lat,x.prev_lon,x.latitude,x.longitude)
    x['implied_knots']=x.distance_km/(x.gap_minutes/60)/KNOT_KM_H
    x.loc[x.gap_minutes.le(0),'implied_knots']=np.nan
    x['valid_segment']=x.gap_minutes.between(1,120)&x.implied_knots.between(0,45)
    x['prev_speed']=g.speed_knots.shift(1)
    x['speed_delta']=x.speed_knots-x.prev_speed
    x['turn_deg']=((x.course_deg-g.course_deg.shift(1)+180)%360-180).abs()
    x['hour_utc']=x.timestamp.dt.hour
    x['day_of_week_utc']=x.timestamp.dt.dayofweek
    return x

def write_fig(fig,path):
    fig.update_layout(template='plotly_white',font_family='Arial')
    fig.write_html(str(path),include_plotlyjs='cdn')

