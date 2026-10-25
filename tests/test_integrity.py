import json
import pandas as pd
import pytest
from src.research import spatial, time_safe_fold

def test_train_labels_available_before_test():
    times=pd.date_range('2023-01-01',periods=20,freq='h',tz='UTC')
    z=pd.DataFrame({'timestamp':times,'target_timestamp':times+pd.Timedelta(hours=3)})
    train,test=time_safe_fold(z,.5,.75)
    assert train.target_timestamp.max()<test.timestamp.min()
    assert len(train)==7  # three boundary-crossing labels removed

def test_invalid_spatial_resolution(tmp_path):
    with pytest.raises(ValueError,match='cell width'):
        spatial(pd.DataFrame(),tmp_path,cell_km=2)

def test_strict_json_and_synthetic_data():
    with open('outputs/results_summary.json') as f:
        json.load(f,parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    df=pd.read_csv('data/ais_sample_100.csv')
    assert (df.dsrc=='SYNTHETIC').all()
    assert df.mmsi.nunique()>0
