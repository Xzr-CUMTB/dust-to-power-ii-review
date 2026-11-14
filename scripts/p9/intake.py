from pathlib import Path
import numpy as np,pandas as pd,sys,json,warnings
ROOT=Path(__file__).absolute().parents[2]; OUT=ROOT/'results/p9'; CK=ROOT/'checkpoints/p9'
for folder in [OUT,CK,OUT/'actions',OUT/'scenarios',OUT/'external',OUT/'sensitivity',OUT/'jit']:folder.mkdir(parents=True,exist_ok=True)
def js(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=str),encoding='utf8')
def build(delay=3):
 s=pd.read_parquet(ROOT/'data/processed/pv_four_stations_original.parquet')
 s.loc[~s.value_valid,'power_mw']=np.nan
 cols=['power_mw','lmd_totalirrad','lmd_temperature','lmd_windspeed','lmd_winddirection']
 tabs=[s.pivot(index='timestamp_local',columns='station_id',values=c).sort_index() for c in cols+['solar_elevation_degrees']]
 ix=tabs[0].index; raw=np.stack([t.values for t in tabs[:-1]],2);geo=tabs[-1].values
 cap=s.groupby('station_id').metadata_capacity_mw.first().sort_index().values
 aq=pd.read_parquet(ROOT/'data/processed/air_hourly_monitors.parquet');sites=sorted(aq.site_code.unique());assert len(sites)==31
 hours=pd.date_range(aq.observation_time_utc.min(),aq.observation_time_utc.max(),freq='h');pms=[]
 for pol in ['PM10','PM2.5']:
  a=aq[aq.type==pol].pivot(index='observation_time_utc',columns='site_code',values='value').reindex(index=hours,columns=sites)
  a=a.where(a>=0);last=np.maximum.accumulate(np.where(a.notna(),np.arange(len(a))[:,None],-99999),0)
  age=np.arange(len(a))[:,None]-last; lv=np.log1p(a.ffill(limit=24));bg=np.log1p(a.shift().rolling(24,min_periods=6).median())
  pms.append(np.stack([np.nan_to_num(lv)/5,np.nan_to_num(lv-bg),(age<=24)*1.,np.minimum(age+delay,24+delay)/(24+delay)],2))
 pm=np.concatenate(pms,2);dist=pd.read_csv(ROOT/'data/metadata/air_pv_monitor_distances.csv',dtype={'pv_station':str})
 ds=[]; rows=[];X=[];Y=[];G=[];M=[];W=[];X6=[];M6=[];W6=[]
 daylist=pd.date_range(ix[0].normalize()+pd.Timedelta(days=7),ix[-1].normalize()-pd.Timedelta(days=1),freq='D')
 for day in daylist:
  start=ix.get_indexer([day])[0];r=raw[start+1:start+97,:,0]
  if r.shape!=(96,4) or not np.isfinite(r).all():continue
  sun=geo[start+1:start+97]; assert np.isfinite(sun).all()
  for kh,xa,ma,wa in [(0,X,M,W),(24,X6,M6,W6)]:
   cutoff=start+kh;last=cutoff-1;parts=[]
   for lag in range(1,8):
    # 24h slices ending at issue-15min or earlier; retain six-hour summaries and observation coverage.
    stop=last+1-(lag-1)*96;g=raw[stop-96:stop].reshape(4,24,4,5)
    with warnings.catch_warnings():
     warnings.simplefilter('ignore');z=np.nanmean(g,1);frac=np.isfinite(g).mean(1)
    parts.extend([np.nan_to_num(z).ravel(),frac.ravel()])
   ang=2*np.pi*day.dayofyear/365.25
   xa.append(np.r_[np.sin(ang),np.cos(ang),sun.reshape(8,12,4).mean(1).ravel()/90,np.concatenate(parts)])
   h=np.searchsorted(hours.asi8,ix[cutoff].value-delay*3600*10**9,'right')-1
   token=pm[h] if h>=0 else np.zeros((31,8));ma.append(token)
   wind=np.nan_to_num(raw[last,:,3:5]);ws=[]
   for j,st in enumerate(tabs[0].columns):
    d=dist[dist.pv_station==st].set_index('site_code').loc[sites]
    dx=(d.longitude-d.pv_longitude)*np.cos(np.deg2rad(d.pv_latitude));dy=d.latitude-d.pv_latitude
    ws.append(np.exp(-d.distance_km.values/200)*np.maximum(0,np.cos(np.arctan2(dx,dy)-np.deg2rad(wind[j,1])))*(wind[j,0]>=2))
   wa.append(np.mean(ws,0))
   rows.append(dict(day=str(day.date()),issue=ix[cutoff],pv_weather_latest_end=ix[last],latest_available=ix[last]+pd.Timedelta(minutes=15),pm_latest_possible_hour=hours[h] if h>=0 else None,first_target=ix[start+kh+1],last_target=ix[start+96],n=96-kh))
  ds.append(str(day.date())); Y.append(r); G.append(sun)
 kw=dict(days=ds,x=X,y=Y,sun=G,pm=M,w=W,x6=X6,pm6=M6,w6=W6,cap=cap,sites=sites)
 suffix='' if delay==3 else '_PM'+str(delay)
 np.savez_compressed(OUT/f'day_interface{suffix}.npz',**{k:np.asarray(v,dtype='f4' if k not in ('days','sites') else str) for k,v in kw.items()})
 pd.DataFrame(rows).to_csv(OUT/f'input_clock{suffix}.csv',index=False)
 if delay!=3:return
 df=pd.DataFrame(np.array(Y).sum(1)*.25,columns=['E00','E04','E07','E08']);df['day']=ds;df['cluster_MWh']=df.iloc[:,:4].sum(1);df.to_csv(OUT/'day_labels.csv',index=False);print(len(ds),ds[0],ds[-1],cap)
def inputs(z,pm='full',hour=0,history=3):
 suf='6' if hour else '';p=z['pm'+suf].copy();w=z['w'+suf]
 if pm=='none':p*=0
 if pm=='region':p[:]=p.mean(1,keepdims=True);p[:,:,[2,3,6,7]]=0
 if pm=='shuffle':p=np.roll(p,7,axis=1)
 f=z['x'+suf][:,:34+history*160]
 # Geometry binding makes a fixed permutation non-equivalent to renaming learned node weights.
 return np.column_stack([f,p.mean(1),p.std(1),(p*w[:,:,None]).mean(1),p[:,:,1],p[:,:,5]])
if __name__=='__main__':build()
