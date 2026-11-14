"""Frozen donor and ensemble routines. A mounted historical project supplies the full-data tables."""
from pathlib import Path
from functools import lru_cache
import os
import numpy as np
import pandas as pd
ROOT=Path(os.environ.get("DUST_ARCHIVE","."))
P=ROOT/"results/p11"
P10=ROOT/"results/p10"
OUT=P10
Q=np.array([.005,.025,.05,.1,.2,.35,.5,.65,.8,.9,.95,.975,.995])
def solar_mask(sun):
 # include any interval bordering sunrise or sunset; keep twilight generation, raw labels unchanged.
 return (np.maximum(sun,np.concatenate([sun[:,:1],sun[:,:-1]],1))>=-1.)

@lru_cache(1)
def near():
 z=np.load(ROOT/'results/p5/dataset.npz');p=np.load(ROOT/'results/p6/forecasts/F1_members.npz');ii=p['ids'];ts=pd.to_datetime(z['t'][ii],utc=True).tz_convert('Asia/Shanghai');ix=np.flatnonzero((ts.hour==0)&(ts.minute==0));v=p['paths'][ix].astype('f4')*z['cap'][None,None,:,None]
 return dict(zip(ts[ix].strftime('%Y-%m-%d'),v.transpose(0,1,3,2)))

@lru_cache(1)
def daylight_history():
 df=pd.read_parquet(ROOT/'data/processed/pv_four_stations_original.parquet',columns=['timestamp_local','station_id','value_valid','power_mw']);df.loc[~df.value_valid,'power_mw']=np.nan;t=df.pivot(index='timestamp_local',columns='station_id',values='power_mw').sum(1,min_count=4);i=t.index-pd.Timedelta('15min');s=(i.hour>=6)&(i.hour<19);return (t[s].groupby(i[s].strftime('%Y-%m-%d')).sum(min_count=52)*.25).to_dict()

def donor(z,idx,hour=0):
 out=[];e=daylight_history();days=z['days'];y=z['y'];prev=lambda d:str((pd.Timestamp(str(d))-pd.Timedelta('1d')).date())
 for i in idx:
  ds=np.flatnonzero(days<prev(days[i]))[-90:];target=e.get(prev(days[i]),np.nan)
  dist=[(abs(e.get(prev(days[j]),np.nan)-target) if np.isfinite(e.get(prev(days[j]),np.nan)+target) else 1000)+.5*(len(ds)-k) for k,j in enumerate(ds)]
  ids=np.resize(ds[np.argsort(dist,kind='stable')[:32]],32);v=y[ids].copy()
  if not hour and days[i] in near():v[:,:24]=near()[days[i]]
  out.append(v[:,hour*4:])
 return np.stack(out)
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
def qualified(z,wx):
 if wx=='W1':return np.ones(len(z['days']),bool)
 table=pd.read_csv(P/'weather_all.csv',usecols=['day','eligible']);count=table[table.eligible].groupby('day').size()
 return np.isin(z['days'],count[count==32].index)

def design(z,wx,pm):
 x=context(z,pm,'history')
 if wx in ['W1','W1common']:
  t=pd.read_csv(P10/'gfs_features.csv',dtype={'station':str});t['h']=78;leads=[78]
 else:
  t=pd.read_csv(P/'weather_all.csv',dtype={'station':str});t=t[t.eligible];leads=[72,78,84,90]
 wt=[];flat=[]
 for h in leads:
  a=np.stack([t[(t.h==h)&(t.field==v)].pivot(index='day',columns='station',values='value').reindex(z['days']).to_numpy() for v in ['rad','cloud']],1)
  ok=qualified(z,wx);assert np.isfinite(a[ok]).all()
  a[~ok]=0 # unused placeholder; rejected rows removed from fit, retrieval and output
  flat.append(a.reshape(len(x),-1))
  # Four site fields, time geometry and issue-relative lead. Average sun over valid part of target day.
  # Local valid end for h=72 is 08:00, not 02:00; convert UTC to local explicitly.
  end=(h-72+8)*4;begin=end-24;elev=z['sun'][:,max(0,begin):min(96,end)].mean((1,2))/60
  wt.append(np.c_[a[:,0]/700,a[:,1]/100,np.full(len(x),(h-78)/12),np.full(len(x),((h-72)+5)/24),elev])
 return np.column_stack([x]+flat).astype('f4'),np.stack(wt,1).astype('f4')

def geometry(z,wx):
 days=z['days'];x,w=design(z,wx,'none');y=z['y'];mask=solar_mask(z['sun']);ids=np.zeros((len(y),32),int);ok=qualified(z,wx)
 for i,d in enumerate(days):
  pool=np.flatnonzero((np.array(days,dtype='datetime64[D]')<=np.datetime64(d)-np.timedelta64(2,'D'))&ok)[-90:]
  if not len(pool):continue
  distance=np.abs(w[i,:,:8]-w[pool,:,:8]).mean((1,2))+.0015*np.arange(len(pool))[::-1]
  chosen=pool[np.resize(np.argsort(distance,kind='stable')[:32],32)]
  masked=y[chosen]*mask[i];ids[i]=chosen[np.argsort(masked.sum((1,2)),kind='stable')]
 bank=y[ids]*mask[:,None]
 E=bank.sum((2,3))*.25
 donor=np.quantile(E,Q,axis=1).T.astype('f4')
 den=E.clip(.001);block=(bank.sum(3).reshape(len(y),32,8,12).sum(3)*.25/den[:,:,None]).astype('f4')
 return ids,donor,block

def sample(z,p,i,n=32,structure=True):
 u=(np.arange(n)+.5)/n
 tq=np.interp(u,Q,p['q'][i]);rank=np.minimum(31,(u*32).astype(int));bank=z['y'][p['ids'][i,rank]]*solar_mask(z['sun'][p['idx'][i:i+1]])[0]
 if structure:bank=bank*np.repeat(np.exp(p['tilt'][i]),12)[None,:,None]
 den=bank.sum((1,2))*.25
 assert (den>.00001).all()
 return (bank*(tq/den)[:,None,None]).astype('f4')
def quantcal(v,old):
 E=v.sum(1)*.25;u=(np.arange(len(v))+.5)/len(v);rank=np.argsort(np.argsort(E,kind='stable'),kind='stable')
 if len(old)<4:return v
 residual=np.stack([r[1] for r in old[-28:]])
 offset=np.array([np.quantile(residual[:,j],q) for j,q in enumerate(Q)])*(len(residual)/(len(residual)+14))
 target=np.sort(np.maximum(.01,np.quantile(E,Q)+offset))
 new=np.interp(u,Q,target)[rank]
 return v*(new/E.clip(.001))[:,None]
