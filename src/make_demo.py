"""Create a reproducible, wholly synthetic AIS-like dataset for public demonstrations."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def generate_demo(path, vessels=120, days=35, seed=2026):
    rng=np.random.default_rng(seed)
    start=pd.Timestamp('2023-08-18',tz='UTC')
    rows=[]
    kinds=['Cargo','Tanker','Tug','Special Craft','Fishing']
    for v in range(vessels):
        mmsi=str(200000000+v+1)
        kind=kinds[v%len(kinds)]
        lat=float(rng.uniform(23.7,25.0));lon=float(rng.uniform(64.0,67.3))
        speed=float(rng.uniform(1,12))
        course=float(rng.uniform(0,360))
        for h in range(days*24):
            if rng.random()<0.25: continue
            speed=max(0,float(speed+rng.normal(0,0.8)))
            if rng.random()<0.13:speed=float(rng.uniform(0,0.8))
            course=(course+rng.normal(0,12))%360
            dt=1.0
            lat+=speed*1.852*dt*np.cos(np.radians(course))/111.195
            lon+=speed*1.852*dt*np.sin(np.radians(course))/(111.195*np.cos(np.radians(lat)))
            if not (23.0<lat<26.0 and 60.0<lon<68.5):
                lat=float(rng.uniform(23.7,25.0));lon=float(rng.uniform(64.0,67.3))
            rows.append(dict(mmsi=mmsi,timestamp=(start+pd.Timedelta(hours=h)).isoformat(),latitude=round(lat,6),longitude=round(lon,6),speed=round(speed*10,1),course=round(course,1),heading=round(course),ais_type_summary=kind,length=round(float(rng.uniform(20,200)),1),draught=round(float(rng.uniform(10,90)),1),dsrc='SYNTHETIC'))
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(target,index=False)
    return len(rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='data/ais_demo.csv');ap.add_argument('--vessels',type=int,default=120);ap.add_argument('--days',type=int,default=35);args=ap.parse_args()
    print(f'Wrote {generate_demo(args.output,args.vessels,args.days):,} synthetic AIS-like rows to {args.output}')
