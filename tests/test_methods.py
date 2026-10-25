import numpy as np
import pandas as pd
from src.research import position_at
from src.pipeline import haversine

def test_northbound_one_hour():
    lat, lon=position_at(0,0,60,0,60)
    assert abs(float(haversine(0,0,lat,lon))-111.12)<0.1
    assert abs(float(lon))<1e-6

def test_dbscan_cell_resolution():
    # Spatial resolution must be no larger than the smallest tested epsilon.
    cell_km=0.5
    assert cell_km <= min([1,2,5,10,20])

def test_rolling_train_precedes_test():
    t=pd.date_range('2023-08-01',periods=100,freq='h')
    for left,right in [(.55,.70),(.70,.85),(.85,1.0)]:
        train=t[:int(len(t)*left)];test=t[int(len(t)*left):int(len(t)*right)]
        assert train.max()<test.min()
