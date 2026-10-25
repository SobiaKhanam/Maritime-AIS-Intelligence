"""AIS exploratory research: transparent preprocessing, spatial sensitivity and forecasts."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .pipeline import load_and_clean, trajectories, haversine, write_fig, KNOT_KM_H
R=6371.0088

def valid_mmsi(s):
    return s.astype('string').str.fullmatch(r'[2-7]\d{8}').fillna(False)

def clean_vessels(df):
    # MMSI MID ranges are approximate; excludes common non-ship formats, not a definitive registry check.
    keep=valid_mmsi(df.mmsi)
    if 'ais_type_summary' in df:
        keep &= ~df.ais_type_summary.fillna('').str.lower().str.contains('navigation aid|vts|search and rescue',regex=True)
    return df.loc[keep].copy(),int((~keep).sum())

def position_at(lat,lon,speed_knots,course_deg,minutes):
    d=np.asarray(speed_knots)*KNOT_KM_H*np.asarray(minutes)/60/R
    a=np.radians(lat);b=np.radians(lon);bearing=np.radians(course_deg)
    lat2=np.arcsin(np.sin(a)*np.cos(d)+np.cos(a)*np.sin(d)*np.cos(bearing))
    lon2=b+np.arctan2(np.sin(bearing)*np.sin(d)*np.cos(a),np.cos(d)-np.sin(a)*np.sin(lat2))
    return np.degrees(lat2),(np.degrees(lon2)+180)%360-180

def spatial(x, out, cell_km=0.5):
    """Equal-area local grid for KDE; exact geodesic DBSCAN for sensitivity."""
    from scipy.ndimage import gaussian_filter
    if cell_km > 1.0 or cell_km <= 0:
        raise ValueError('Spatial cell width must be > 0 and <= the minimum 1 km DBSCAN radius')
    z=x.dropna(subset=['latitude', 'longitude']).copy()
    z['hour_bin'] = z.timestamp.dt.floor('h')
    z=z.drop_duplicates(['mmsi','hour_bin'])
    lat0=float(z.latitude.median()); lon0=float(z.longitude.median())
    # Weighted 0.5 km cell-centre approximation to point-level DBSCAN.
    xy_lat=np.floor(z.latitude.to_numpy()*111.195/cell_km).astype(int)
    xy_lon=np.floor(z.longitude.to_numpy()*111.195*np.cos(np.radians(lat0))/cell_km).astype(int)
    cells=pd.DataFrame({'iy':xy_lat,'ix':xy_lon}).value_counts().rename('count').reset_index()
    coords=np.radians(np.column_stack(((cells.iy.to_numpy()+.5)*cell_km/111.195,(cells.ix.to_numpy()+.5)*cell_km/(111.195*np.cos(np.radians(lat0))))))
    weights=cells['count'].to_numpy()
    sweep=[]; selected=None
    for eps in [1,2,5,10,20]:
        labels=DBSCAN(eps=eps/R,min_samples=5,metric='haversine',algorithm='ball_tree',n_jobs=1).fit_predict(coords,sample_weight=weights)
        counts=pd.DataFrame({'label':labels[labels>=0],'weight':weights[labels>=0]}).groupby('label').weight.sum()
        sweep.append({'epsilon_km':eps,'clusters':int(len(counts)),'noise_fraction':round(float(weights[labels<0].sum()/weights.sum()),4),'largest_cluster_fraction':round(float(counts.max()/weights.sum()) if len(counts) else 0,4)})
        if eps==2:selected=labels
    pd.DataFrame(sweep).to_csv(out/'dbscan_sensitivity.csv',index=False)
    cells['cluster']=selected
    cells['latitude']=np.degrees(coords[:,0]);cells['longitude']=np.degrees(coords[:,1])
    centers=cells[cells.cluster>=0].groupby('cluster').apply(lambda d:pd.Series({'reports':int(d['count'].sum()),'mean_lat':np.average(d.latitude,weights=d['count']),'mean_lon':np.average(d.longitude,weights=d['count'])}),include_groups=False).sort_values('reports',ascending=False)
    centers.to_csv(out/'cluster_summary.csv')
    if len(centers):
        fig=px.scatter_map(centers.head(100).reset_index(),lat='mean_lat',lon='mean_lon',size='reports',map_style='carto-positron',zoom=4,title='AIS report clusters (2 km DBSCAN; haversine distance)')
        write_fig(fig,out/'cluster_centres.html')
    # 2 km raster for KDE; 0.5 km weighted cells for geodesic DBSCAN.
    cell_km=2.0
    xx=(z.longitude.to_numpy()-lon0)*111.195*np.cos(np.radians(lat0)); yy=(z.latitude.to_numpy()-lat0)*111.195
    ix=np.floor(xx/cell_km).astype(int);iy=np.floor(yy/cell_km).astype(int)
    minx,maxx=ix.min(),ix.max();miny,maxy=iy.min(),iy.max()
    hist=np.zeros((maxy-miny+1,maxx-minx+1),dtype=np.float32)
    np.add.at(hist,(iy-miny,ix-minx),1)
    density=gaussian_filter(hist,sigma=10/cell_km,mode='constant')
    # Thin the display grid if necessary; density was already smoothed at 2 km.
    step=1;ny,nx=density.shape
    ys=np.arange(0,ny,step);xs=np.arange(0,nx,step)
    image=density[np.ix_(ys,xs)]
    lat_axis=lat0+(ys+miny+0.5)*cell_km/111.195
    lon_axis=lon0+(xs+minx+0.5)*cell_km/(111.195*np.cos(np.radians(lat0)))
    # Geographic basemap with an actual coastline, instead of longitude/latitude heatmap axes.
    lons,lats=np.meshgrid(lon_axis,lat_axis)
    flat=image.ravel(); keep=flat>max(float(flat.max())*.015,0.005)
    mapdf=pd.DataFrame({'latitude':lats.ravel()[keep],'longitude':lons.ravel()[keep],'density':flat[keep]})
    fig=px.scatter_map(mapdf,lat='latitude',lon='longitude',color='density',size='density',size_max=13,opacity=.48,
        color_continuous_scale='Turbo',map_style='carto-positron',zoom=5,center={'lat':24.5,'lon':65.9},
        title='AIS reporting density near Pakistan and the Arabian Sea (Gaussian bandwidth: 10 km)')
    fig.update_layout(map=dict(bearing=0),coloraxis_colorbar_title='Smoothed reports / grid cell')
    write_fig(fig,out/'kde_density.html')
    pd.DataFrame(sweep)[['epsilon_km','clusters','noise_fraction']].to_csv(out/'cluster_sweep_plot_data.csv',index=False)
    return {'vessel_hour_records':int(len(z)),'occupied_grid_cells':int(np.count_nonzero(hist)),
        'epsilon_sweep':sweep,'kde_bandwidth_km':10,'method':'Haversine DBSCAN on weighted 0.5 km cell centres using all vessel-hour records; Gaussian KDE on a 2 km raster with geographic basemap.',
        'caveat':'AIS reporting coverage varies; hotspots show reporting concentration, not unique vessel counts.'}

def anomalies(x,out):
    z=x.copy();z['rule_speed_jump']=z.valid_segment & z.speed_delta.abs().gt(15)
    z['rule_improbable_displacement']=z.gap_minutes.gt(0)&z.implied_knots.gt(45)
    z['if_score']=np.nan;z['if_flag']=False
    cats=z.ais_type_summary.fillna('Unknown') if 'ais_type_summary' in z else pd.Series('Unknown',index=z.index)
    z['category']=cats
    summary=[];examples=[]
    for cat,idx in z[z.valid_segment].groupby(cats).groups.items():
        if len(idx)<100:
            summary.append({'category':cat,'scored':0,'unscored':len(idx),'flagged':0,'note':'fewer than 100 segments'});continue
        feat=['speed_knots','implied_knots','speed_delta']
        d=z.loc[idx,feat].replace([np.inf,-np.inf],np.nan);d=d.fillna(d.median()).fillna(0)
        model=IsolationForest(n_estimators=120,contamination='auto',random_state=42,n_jobs=-1).fit(d)
        score=model.decision_function(d);flag=model.predict(d)==-1
        z.loc[idx,'if_score']=score;z.loc[idx,'if_flag']=flag
        summary.append({'category':cat,'scored':len(idx),'unscored':0,'flagged':int(flag.sum()),'note':'unlabelled screening'})
        for rowid in pd.Series(score,index=idx).nsmallest(min(10,len(idx))).index:
            r=z.loc[rowid];examples.append({'category':cat,'speed_knots':round(float(r.speed_knots),2) if pd.notna(r.speed_knots) else None,'implied_knots':round(float(r.implied_knots),2) if pd.notna(r.implied_knots) else None,'speed_change_knots':round(float(r.speed_delta),2) if pd.notna(r.speed_delta) else None,'score_within_category':round(float(r.if_score),4),'rule_speed_jump':bool(r.rule_speed_jump),'rule_improbable_displacement':bool(r.rule_improbable_displacement)})
    pd.DataFrame(summary).to_csv(out/'anomaly_by_category.csv',index=False)
    pd.DataFrame(examples,columns=['category','speed_knots','implied_knots','speed_change_knots','score_within_category','rule_speed_jump','rule_improbable_displacement']).sort_values('score_within_category').head(10).to_csv(out/'anomaly_examples_anonymized.csv',index=False)
    cols=['timestamp','mmsi','category','latitude','longitude','speed_knots','implied_knots','gap_minutes','speed_delta','rule_speed_jump','rule_improbable_displacement','if_score','if_flag']
    z.loc[z.rule_speed_jump|z.rule_improbable_displacement|z.if_flag,cols].to_csv(out/'anomaly_review_private.csv',index=False)
    return {'rule_speed_jump':int(z.rule_speed_jump.sum()),'rule_improbable_displacement':int(z.rule_improbable_displacement.sum()),'category_counts':summary,'note':'Unlabelled screening; no validated anomalies inferred.'}

def ci_by_vessel(frame, pred_a, pred_b, n_boot=300,seed=42):
    # Cluster bootstrap: resample vessels, not individual correlated AIS reports.
    t=frame[['mmsi']].copy();t['a']=np.abs(frame.target_speed.to_numpy()-pred_a);t['b']=np.abs(frame.target_speed.to_numpy()-pred_b)
    agg=t.groupby('mmsi').agg(a=('a','sum'),b=('b','sum'),n=('a','size'))
    rng=np.random.default_rng(seed);arr=agg[['a','b','n']].to_numpy();n=len(arr)
    if n<2:return [None,None]
    samples=[]
    for _ in range(n_boot):
        sub=arr[rng.integers(0,n,n)]
        samples.append(float((sub[:,1].sum()-sub[:,0].sum())/sub[:,2].sum()))
    return np.quantile(samples,[.025,.975]).round(4).tolist()

def time_safe_fold(z, left, right):
    """Only train on labels observed strictly before the test cutoff."""
    n=len(z)
    boundary=int(n*left)
    cutoff=z.iloc[boundary].timestamp
    train=z.iloc[:boundary].loc[lambda a: a.target_timestamp < cutoff].copy()
    test=z.iloc[boundary:int(n*right)].copy()
    if len(train) and len(test):
        assert train.target_timestamp.max() < test.timestamp.min()
    return train,test

def forecast(x,out):
    z=x.copy();g=z.groupby('mmsi',sort=False)
    z['target_speed']=g.speed_knots.shift(-1)
    z['target_timestamp']=g.timestamp.shift(-1)
    z['next_gap_min']=(z.target_timestamp-z.timestamp).dt.total_seconds()/60
    z=z[z.speed_knots.notna() & z.target_speed.notna() & z.next_gap_min.gt(0)].sort_values('timestamp').copy()
    # No future-gap filter. Next observation is a variable-horizon retrospective task.
    features=['speed_knots','latitude','longitude','course_deg','hour_utc','day_of_week_utc']
    if 'length' in z:features.append('length')
    z['category']=z.ais_type_summary.fillna('Unknown') if 'ais_type_summary' in z else 'Unknown'
    if len(z)<300:return {'status':'insufficient data'}
    # Rolling-origin with label availability enforced at each cutoff.
    cuts=[(.55,.70),(.70,.85),(.85,1.0)];rows=[];detail=[]
    for fold,(left,right) in enumerate(cuts,1):
        train,test=time_safe_fold(z,left,right)
        if train.empty or test.empty: continue
        med=train[features].median().fillna(0)
        model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=16,l2_regularization=2,random_state=42)
        model.fit(train[features].fillna(med),train.target_speed)
        ml=model.predict(test[features].fillna(med));base=test.speed_knots.to_numpy();y=test.target_speed.to_numpy()
        delta_model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,max_leaf_nodes=16,l2_regularization=2,random_state=42)
        delta_model.fit(train[features].fillna(med),train.target_speed-train.speed_knots)
        delta=np.maximum(0,base+delta_model.predict(test[features].fillna(med)))
        hybrid=np.where(base<1,base,delta)
        cis={name:ci_by_vessel(test,base,p) for name,p in [('Gradient boosting',ml),('Absolute-error delta',delta),('Stationary hybrid',hybrid)]}
        rows.extend([{'fold':fold,'model':name,'n':len(test),'MAE_knots':round(float(mean_absolute_error(y,p)),4),'RMSE_knots':round(float(np.sqrt(mean_squared_error(y,p))),4),'bootstrap_difference_ci_low':cis[name][0] if name in cis else None,'bootstrap_difference_ci_high':cis[name][1] if name in cis else None} for name,p in [('Persistence',base),('Gradient boosting',ml),('Absolute-error delta',delta),('Stationary hybrid',hybrid)]])
        temp=test[['mmsi','category','speed_knots','next_gap_min','target_speed']].copy();temp['baseline_abs_error']=np.abs(y-base);temp['ml_abs_error']=np.abs(y-ml);temp['delta_abs_error']=np.abs(y-delta);temp['hybrid_abs_error']=np.abs(y-hybrid);temp['fold']=fold
        detail.append(temp)
    pd.DataFrame(rows).to_csv(out/'rolling_forecast_metrics.csv',index=False)
    d=pd.concat(detail)
    d['movement_group']=np.where(d.speed_knots<1,'stationary (<1 knot)','moving (>=1 knot)')
    by=d.groupby(['fold','movement_group','category'],as_index=False).agg(n=('mmsi','size'),persistence_mae=('baseline_abs_error','mean'),ml_mae=('ml_abs_error','mean'),delta_mae=('delta_abs_error','mean'),hybrid_mae=('hybrid_abs_error','mean'))
    by.to_csv(out/'forecast_by_vessel_type_and_movement.csv',index=False)
    d['horizon_group']=pd.cut(d.next_gap_min,[0,30,60,90,120,np.inf],labels=['0-30','30-60','60-90','90-120','120+'])
    d.groupby(['fold','horizon_group'],observed=True).agg(n=('mmsi','size'),persistence_mae=('baseline_abs_error','mean'),ml_mae=('ml_abs_error','mean'),delta_mae=('delta_abs_error','mean'),hybrid_mae=('hybrid_abs_error','mean')).to_csv(out/'forecast_by_observed_gap.csv')
    return {'task':'Next observed speed; variable future observation interval, never a fixed-horizon prediction','rolling_folds':rows,'caveat':'Same vessels can recur in successive periods; report timing is not known in advance. Future gap used only for retrospective stratification.'}

def run(input_path,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    df,q=load_and_clean(input_path)
    df.loc[~df.course.between(0,359.9),'course_deg']=np.nan
    df,removed=clean_vessels(df)
    x=trajectories(df)
    x.loc[~x.course_deg.between(0,359.9),'course_deg']=np.nan
    daily=x.set_index('timestamp').resample('D').size()
    full_dates=(x.timestamp.min().ceil('D'),x.timestamp.max().floor('D'))
    daily=daily[(daily.index>=full_dates[0]) & (daily.index<full_dates[1])]
    daily.rename('reports').to_csv(out/'daily_coverage.csv')
    write_fig(px.line(x=daily.index,y=daily.values,labels={'x':'UTC date','y':'Received reports'},title='Daily AIS reception coverage'),out/'daily_coverage.html')
    gap=x.gap_minutes.dropna();gap_summary={'median_min':round(float(gap.median()),2),'p10_min':round(float(gap.quantile(.1)),2),'p90_min':round(float(gap.quantile(.9)),2),'fraction_30_to_90_min':round(float(gap.between(30,90).mean()),4)}
    summary={'raw_rows':q['raw_rows'],'after_dedup_and_coordinates':q['clean_rows'],'excluded_nonship_or_invalid_mmsi_rows':removed,'analysed_rows':len(x),'unique_vessels':int(x.mmsi.nunique()),'start_utc':str(x.timestamp.min()),'end_utc':str(x.timestamp.max()),'report_gaps':gap_summary,'daily_min':int(daily.min()),'daily_max':int(daily.max()),'study_area':'Arabian Sea / Pakistan coastal approaches, as represented in this export; no EEZ jurisdiction assertion'}
    summary['spatial']=spatial(x,out)
    summary['anomaly_screening']=anomalies(x,out)
    summary['speed_forecasting']=forecast(x,out)
    (out/'results_summary.json').write_text(json.dumps(summary,indent=2,default=str,allow_nan=False))
    print(json.dumps(summary,indent=2,default=str))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/ais.csv');p.add_argument('--output',default='outputs');a=p.parse_args();run(a.input,a.output)
