from pathlib import Path
import sys,json,time,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'scripts/p10'));sys.path.insert(0,str(ROOT))
from common10 import context,basepaths,sp,solar_mask,cr,metrics
P=ROOT/'results/p11';CK=ROOT/'checkpoints/p11';P10=ROOT/'results/p10';P9=ROOT/'results/p9'
for d in [P,CK,P/'forecasts',P/'scores',P/'logs',P/'actions',ROOT/'configs/p11']:d.mkdir(parents=True,exist_ok=True)
def dump(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=str),encoding='utf8')
def zload():return dict(np.load(P9/'day_interface.npz'))
Q=np.array([.005,.025,.05,.1,.2,.35,.5,.65,.8,.9,.95,.975,.995])
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
def splits(z,f):
 ds=z['days'];ti,vi,ca,ei=sp(z,f)
 if f in ['early','anchor','later']:return ti,vi,np.unique(np.r_[ca,ei]),ei
def getfold(z,f):
 ds=z['days']
 if f in ['early','anchor','later']:return splits(z,f)
 if f=='pilot':
  prior=np.flatnonzero(ds<'2019-02-28');ix=np.flatnonzero((ds>='2019-03-01')&(ds<='2019-03-15'));return prior[:-15],prior[-14:],ix,np.array([],int)
 if f.startswith('oof'):
  n=int(f[3:]);start=pd.Timestamp('2019-03-25')+pd.Timedelta(days=14*n);end=start+pd.Timedelta('14d')
  ix=np.flatnonzero((ds>=str(start.date()))&(ds<str(end.date())));prior=np.flatnonzero(ds<str((start-pd.Timedelta('1d')).date()))
  return prior[:-15],prior[-14:],ix,np.array([],int)
 g=pd.read_csv(P9/'retained_exposure_sources.csv');g=g[g.independent_block_id==f]
 ev=sorted({str(d.date()) for a,b in zip(g.official_start_bjt,g.official_end_bjt_exclusive) for d in pd.date_range(pd.Timestamp(a),pd.Timestamp(b)-pd.Timedelta('1d'))})
 ix=np.flatnonzero(np.isin(ds,ev));prior=np.flatnonzero(ds<str((pd.Timestamp(ev[0])-pd.Timedelta('1d')).date()))
 return prior[:-13],prior[-12:],ix,ix

