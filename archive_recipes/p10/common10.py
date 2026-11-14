from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/'scripts/p9'))
from train import donor,sp,metrics,solar_mask,cr
from src.models.p10.donor import fit_suite
P9=ROOT/'results/p9'
OUT=ROOT/'results/p10'
CK=ROOT/'checkpoints/p10'
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf8')
def context(z,mode,weather='history'):
    x=z['x'];n=len(x)
    # Each daily block contains four six-hour intervals, four sites, five measurements and their coverage.
    history=x[:,34:].reshape(n,7,4,2,4,5)
    power=history[:,:,:,0,:,0].mean(3)
    irradiance=history[:,:,:,0,:,1].mean(3)
    environment=history[:,:,:,0,:,2:4].mean(3)
    features=[x[:,:2],x[:,2:34].reshape(n,8,4).mean(2),power[:,:2].reshape(n,-1),power[:,2:].mean(1),irradiance[:,:2].reshape(n,-1),irradiance[:,2:].mean(1),environment[:,0].reshape(n,-1),history[:,:,:,1].mean((1,2,3))]
    pm=z['pm'].copy();direction=z['w']
    if mode=='none':pm[:]=0
    elif mode=='region':
        pm[:]=pm.mean(1,keepdims=True);pm[:,:,[2,3,6,7]]=0
    elif mode=='shuffle':pm=np.roll(pm,7,axis=1)
    features.extend([pm.mean(1),pm.std(1),(pm*direction[:,:,None]).mean(1),pm[:,:,1],pm[:,:,5]])
    if weather.startswith('gfs'):
        table=pd.read_csv(OUT/'gfs_features.csv',dtype={'station':str})
        vars_=['rad','cloud'] if weather=='gfs' else [weather.split('-')[1]]
        for var in vars_:
            tab=table[table.field==var].pivot(index='day',columns='station',values='value').reindex(z['days'])
            features.append(tab.notna().mean(1).values[:,None])
            # Missing archive or provenance falls back to a previously matured covariate median.
            prior=tab.shift(1).expanding().median()
            features.append(tab.fillna(prior).fillna(300 if var=='rad' else 50).to_numpy())
    elif weather!='history':
        frame=pd.read_csv(OUT/'weather_coverage.csv',dtype={'station':str})
        used=['dswrf_sfc','tcdc_eatm'] if weather in ['both','allwx'] else [] if weather=='wind' else [weather]
        for var in used:
            tab=frame[frame['var']==var].pivot(index='day',columns='station',values='value').reindex(z['days'])
            if tab.isna().any().any():raise ValueError(f'{var} missing on {tab.index[tab.isna().any(axis=1)].tolist()}')
            features.append(tab.values)
        if weather in ['wind','allwx']:
            wind=pd.read_csv(OUT/'weather_wind.csv',dtype={'station':str})
            components=[wind[wind['var']==v].pivot(index='day',columns='station',values='value').reindex(z['days']).to_numpy() for v in ['ugrd_hgt','vgrd_hgt']]
            u,v=components;assert np.isfinite([u,v]).all()
            features.extend(components)
            geom=pd.read_csv(ROOT/'data/metadata/air_pv_monitor_distances.csv',dtype={'pv_station':str})
            align=[]
            for j,station in enumerate(['00','04','07','08']):
                g=geom[geom.pv_station==station].set_index('site_code').loc[z['sites']]
                angle=np.arctan2((g.longitude-g.pv_longitude)*np.cos(np.deg2rad(g.pv_latitude)),g.latitude-g.pv_latitude).to_numpy()
                speed=np.hypot(u[:,j],v[:,j]);upwind=np.maximum(0,-(u[:,j,None]*np.sin(angle)+v[:,j,None]*np.cos(angle))/np.maximum(speed[:,None],.01))
                align.append(upwind*(speed[:,None]>=2)*np.exp(-g.distance_km.to_numpy()/200))
            features.append((pm*np.mean(align,axis=0)[:,:,None]).mean(1))
    return np.column_stack(features).astype('f4')
def basepaths(z):
    path=OUT/'forward_D0.npz'
    if not path.exists():
        ds=z['days'];ids=np.arange(3,len(ds));paths=np.zeros((len(ds),32,96,4),'f4')
        paths[ids]=donor(z,ids)
        np.savez_compressed(path,days=ds,paths=paths)
    return np.load(path)['paths']
def tag(fold,mode,weather,seed,stage):
    return f'{fold}_{mode}_{weather}_{seed}_r{stage}'

