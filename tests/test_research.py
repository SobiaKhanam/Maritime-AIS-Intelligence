import numpy as np
import pandas as pd
from src.pipeline import haversine, trajectories
from src.research import position_at, valid_mmsi, ci_by_vessel

def test_due_north_60_knots_one_hour():
    lat,lon=position_at(0.,60.,60.,0.,60.)
    assert abs(float(haversine(0.,60.,lat,lon))-111.12)<0.05
    assert abs(float(lon)-60.)<1e-8

def test_east_heading_and_wrap():
    lat,lon=position_at(0.,179.9,10.,90.,60.)
    assert -180 <= lon < 180
    assert abs(float(haversine(0.,179.9,lat,lon))-18.52)<0.01

def test_mmsi_filter():
    assert valid_mmsi(pd.Series(['356540000','000000000','123','999999999'])).tolist()==[True,False,False,False]

def test_trajectory_group_boundaries():
    d=pd.DataFrame({'mmsi':['200000001','200000001','300000001'],'timestamp':pd.to_datetime(['2023-01-01T00:00Z','2023-01-01T01:00Z','2023-01-01T01:00Z']), 'latitude':[0,0,0],'longitude':[0,1,2], 'speed_knots':[0,0,0],'course_deg':[0,0,0]})
    x=trajectories(d)
    assert np.isnan(x.loc[2,'gap_minutes'])
    assert abs(x.loc[1,'implied_knots']-60.04)<.2

def test_vessel_bootstrap_zero_difference():
    f=pd.DataFrame({'mmsi':['200000001']*3+['300000001']*3,'target_speed':[1,2,3,4,5,6]})
    p=np.array([1,2,3,4,5,6]);assert ci_by_vessel(f,p,p)==[0.,0.]
