from intake import *
sys.path.insert(0,str(ROOT))
from src.models.p9.dayflow import fit,sample,Encoder
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import joblib,time,torch
from functools import lru_cache
def cr(v,y):
 v=np.asarray(v,dtype=float);n=v.shape[1];a=(2*np.arange(n)+1-n)/n**2
 return abs(v-np.expand_dims(y,1)).mean(1)-np.tensordot(np.sort(v,1),a,axes=(1,0))
def solar_mask(sun):
 # include any interval bordering sunrise or sunset; keep twilight generation, raw labels unchanged.
 return (np.maximum(sun,np.concatenate([sun[:,:1],sun[:,:-1]],1))>=-1.)
def fold_days(fold):
 f=pd.read_csv(ROOT/f'results/p8/daily_{fold}.csv');ev=np.sort(f.day.unique())
 v=pd.read_csv(ROOT/f'results/p8/training/tune_{fold}_1.csv');v=np.sort(v.day.unique())
 cfg=json.loads((ROOT/'configs/p7/protocol.json').read_text())['folds'][fold]
 return cfg[0],cfg[1],ev,v
def sp(z,fold):
 days=z['days'];cut,_,ev,va=fold_days(fold);fitall=np.flatnonzero(days<cut)
 # holdout for forecast selection before policy margin fitting. -1 day embargo between each role.
 ti,vi=fitall[:-19],fitall[-18:];cal=np.flatnonzero((days>cut)&(days<ev[0])&np.isin(days,all_dispatch_days()))
 ei=np.flatnonzero(np.isin(days,ev));return ti,vi,cal,ei
@lru_cache(1)
def all_dispatch_days():
 f=pd.read_csv(ROOT/'results/p6/forecasts/F1_maturity.parquet') if False else pd.read_csv(ROOT/'results/p7/baseline_days.csv') if (ROOT/'results/p7/baseline_days.csv').exists() else None
 z=np.load(ROOT/'results/p5/dataset.npz');t=z['t'][z['split']!=0];dd=pd.to_datetime(t,utc=True).tz_convert('Asia/Shanghai').strftime('%Y-%m-%d')
 u,n=np.unique(dd,return_counts=True);return u[n==96]
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
def d1(x,y,ti,ix,stem,mask,dt=.25):
 qs=[.03,.1,.25,.5,.75,.9,.97];en=y.sum((1,2))*dt
 models=[LGBMRegressor(objective='quantile',alpha=q,n_estimators=65,learning_rate=.04,num_leaves=6,min_child_samples=16,verbosity=-1,n_jobs=1,random_state=42).fit(x[ti],en[ti]) for q in qs];joblib.dump(models,str(stem)+'.joblib');pred=np.sort(np.column_stack([m.predict(x[ix]) for m in models]),1)
 out=[];order=np.argsort(en[ti],kind='stable')
 for j,i in enumerate(ix):
  eq=np.interp((np.arange(32)+.5)/32,qs,pred[j]);nearest=np.abs(en[ti,None]-eq).argmin(0)
  base=y[ti[nearest]].copy()*mask[i];out.append(base*eq[:,None,None]/np.maximum(.01,base.sum((1,2))*dt)[:,None,None])
 return np.array(out).astype('f4')
def metrics(y,v,sun,dt=.25):
 en=y.sum((1,2))*dt;E=v.sum((2,3))*dt;l,h=np.quantile(E,[.05,.95],1);q=np.quantile(E,.1,1);delta=en-q;g=y.sum(-1);pred=v.sum(-1);late=24 if y.shape[1]==96 else 0
 mask=(sun>0).any(-1);err=cr(pred,g)
 out=dict(energy_crps=cr(E,en),energy_MAE=abs(np.median(E,1)-en),low_pinball=np.maximum(.1*delta,-.9*delta),low_below=en<=q,lower=l,upper=h,cover90=(en>=l)&(en<=h),width90=h-l,interval90=h-l+20*(np.maximum(l-en,0)+np.maximum(en-h,0)),daylight_CRPS=(err*mask).sum(1)/np.maximum(1,mask.sum(1)),late_CRPS=err[:,late:].mean(1),late_MAE=abs(np.median(pred,1)-g)[:,late:].mean(1),daylight_MAE=(abs(np.median(pred,1)-g)*mask).sum(1)/np.maximum(1,mask.sum(1)),observed_MWh=en,predicted_MWh=np.median(E,1))
 for h in [1,3,6]:
  b=int(h/dt)
  if g.shape[1]%b==0:out[f'block{h}_CRPS']=cr(pred.reshape(len(g),32,-1,b).sum(-1)*dt,g.reshape(len(g),-1,b).sum(-1)*dt).mean(1)
 for k in range(y.shape[-1]):out[f'site{k}_energy_CRPS']=cr(v[:,:,:,k].sum(2)*dt,y[:,:,k].sum(1)*dt)
 return pd.DataFrame(out)
def main(stage):
 z=dict(np.load(OUT/'day_interface.npz'));rows=[];logs=[];splits=[]
 for fold in ['early','anchor','later']:
  ti,vi,ca,ei=sp(z,fold);ix=np.unique(np.r_[vi,ca,ei]);days=z['days']
  for k,ii in [('fit',ti),('forecast_validation',vi),('decision_calibration',ca),('evaluation',ei)]:splits+= [dict(fold=fold,day=days[i],role=k) for i in ii]
  for name,pm,seed,hour in (([(a,'full',42,0) for a in ['D0','D1','D2','D3','D4']]) if stage=='1' else [('D4',pm,seed,0) for pm,seed in [('full',42),('full',3407),('full',2025),('none',42),('region',42),('shuffle',42)]]+[('D4','full',42,6),('D2','full',42,6)]):
   tag=f'{fold}_{name}_{pm}_{seed}_r{stage}_h{hour}';x=inputs(z,pm,hour);y=z['y'][:,hour*4:];sun=z['sun'][:,hour*4:];mask=solar_mask(sun);stem=CK/tag;t=time.perf_counter()
   if name=='D0':v=donor(z,ix)
   elif name=='D1':v=d1(x,y,ti,ix,stem,mask)
   else:
    log=fit(x,y,ti,vi,name,stem,seed,r2=(stage=='2' and name=='D4'));logs+= [dict(tag=tag,epoch=e,train=l,validation=va) for e,l,va in log];v=sample(stem,x[ix],mask[ix])
    if stage=='2' and name=='D4':
     analog=d1(x,y,ti,ix,CK/(tag+'_energy'),mask);target=np.sort(analog.sum((2,3))*.25,axis=1)
     energy=v.sum((2,3))*.25;rank=np.argsort(np.argsort(energy,axis=1,kind='stable'),axis=1)
     v*=((.5*energy+.5*np.take_along_axis(target,rank,1))/np.maximum(energy,.00001))[:,:,None,None]
   assert np.isfinite(v).all();f=metrics(y[ix],v,sun[ix]);f['day']=days[ix];f['tag']=tag;f['fold']=fold;f['role']=np.select([np.isin(ix,ei),np.isin(ix,ca)],['evaluation','decision_calibration'],'forecast_validation');f['stage']=stage;f['model']=name;f['pm']=pm;f['seed']=seed;f['hour']=hour;rows.append(f)
   np.savez_compressed(OUT/'scenarios'/f'{tag}.npz',days=days[ix],paths=v,energy=v.sum((2,3))*.25,block1=v.reshape(len(v),32,-1,4,v.shape[-1]).sum((3,4))*.25)
   print(tag,len(ti),len(vi),len(ca),len(ei),round(f[f.role=='forecast_validation'].energy_crps.mean(),3),round(time.perf_counter()-t,2),flush=True)
  pd.concat(rows).to_csv(OUT/f'forecast_days_r{stage}.csv',index=False);pd.DataFrame(logs).to_csv(OUT/f'training_r{stage}.csv',index=False)
 pd.DataFrame(splits).to_csv(OUT/'split_dates.csv',index=False)
if __name__=='__main__':main(sys.argv[1])

