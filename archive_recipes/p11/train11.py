from common11 import *
import torch,joblib
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from src.models.p10.donor import tensor,optimize,fit_suite
from src.models.p11.quantile import Tail
def train(wx,folds,mode,seed,rnd,models):
 z=zload();y=z['y'];en=y.sum((1,2))*.25;blk=y.sum(2).reshape(len(y),8,12).sum(2)*.25
 x,w=design(z,wx,mode);ids,don,frac=geometry(z,wx);ds=z['days']
 if rnd==2:
  # Same real shapes; extrapolate donor energy with known target/donor solar opportunity.
  opportunity=np.maximum(0,np.sin(np.deg2rad(z['sun']))).sum((1,2))
  energy=(y[ids]*solar_mask(z['sun'])[:,None]).sum((2,3))*.25
  ratio=np.clip(opportunity[:,None]/opportunity[ids].clip(1),.6,1.65)
  don=np.quantile(energy*ratio,Q,axis=1).T.astype('f4')
 for f in folds:
  ti,vi,ix,ei=getfold(z,f);ok=qualified(z,wx);ti=ti[(ti>=34)&ok[ti]];vi=vi[ok[vi]];ix=ix[ok[ix]];output=np.unique(np.r_[vi,ix])
  for name in models:
   tag=f'{f}_{wx}_{mode}_{name}_r{rnd}_s{seed}';out=P/'forecasts'/(tag+'.npz')
   if out.exists():continue
   clock=time.perf_counter();torch.manual_seed(seed);torch.set_num_threads(1)
   scale=float(np.median(en[ti]));sc=StandardScaler().fit(x[ti]);xx=sc.transform(x).clip(-5,5);features=np.c_[xx,don/scale]
   if name=='M2':
    # Same residual pipeline and P10 retrieval pool; no new architecture.
    x2=context(z,mode,'gfs') if wx=='W1' else x
    ixmap=np.flatnonzero(ok)
    v,h=fit_suite(x2[ok],y[ok],ds[ok],basepaths(z)[ok],np.searchsorted(ixmap,ti),np.searchsorted(ixmap,vi),np.searchsorted(ixmap,output),CK/tag,seed,True,.25,solar_mask(z['sun'])[ok])
    paths=v['B3'];np.savez_compressed(out,idx=output,days=ds[output],paths=paths)
    log=[dict(component=key,**r) for key,rows in h.items() for r in rows]
   else:
    if name in ('M1','M1match'):
     if name=='M1match':
      # Exact flattened M3 input, including deterministic interval solar geometry
      # and the same solar-opportunity-adjusted donor quantiles in round two.
      features=np.column_stack([features,w.reshape(len(w),-1)])
     trees=[LGBMRegressor(objective='quantile',alpha=float(q),num_leaves=6,n_estimators=95,learning_rate=.035,min_child_samples=13,n_jobs=1,verbosity=-1).fit(features[ti],en[ti]) for q in Q]
     qpred=np.maximum(.01,np.sort(np.column_stack([m.booster_.predict(features[output]) for m in trees]),axis=1));tilt=np.zeros((len(output),8));log=[dict(component='trees',trees=len(trees),iterations=95)]
     joblib.dump((trees,sc),CK/(tag+'.joblib'))
    else:
     m=Tail(xx.shape[1],len(Q)).cuda();X,W,A,Y,B,F,qt=map(tensor,[xx,w,don/scale,en/scale,blk/scale,frac,Q]);u=(np.arange(32)+.5)/32
     hi=np.searchsorted(Q,u).clip(1,len(Q)-1);lo=hi-1;weight=tensor((u-Q[lo])/(Q[hi]-Q[lo]))
     def loss(i):
      qp,t=m(X[i],W[i],A[i]);err=Y[i,None]-qp;pin=torch.maximum(qt*err,(qt-1)*err)
      pt=qp[:,lo]*(1-weight)+qp[:,hi]*weight
      adj=F[i]*t.exp()[:,None];adj=adj/adj.sum(2,keepdim=True).clamp_min(.001)
      bm=(adj*pt[:,:,None]).mean(1)
      ls=2*torch.trapz(pin,qt,dim=1).mean()+.55*pin[:,2:4].mean()+.65*(bm-B[i]).abs().mean()+.008*t.square().mean()
      if rnd==2:
       cfg=json.loads((ROOT/'configs/p11/round2.json').read_text());ls=ls+cfg['tail_weight']*(pin[:,[1,2,3,9,10,11]]).mean()+cfg['center_penalty']*(qp[:,6]-A[i,6]).abs().mean()
      return ls
     log=optimize(m,loss,ti,vi,str(CK/(tag+'.pt')),seed,90)
     with torch.no_grad():qp,t=m(X[output],W[output],A[output]);qpred=qp.cpu().numpy()*scale;tilt=t.cpu().numpy()
     joblib.dump(dict(scale=scale,scaler=sc),CK/(tag+'_scale.joblib'))
    p=dict(idx=output,days=ds[output],ids=ids[output],q=qpred,tilt=tilt);np.savez_compressed(out,**p)
    paths=np.stack([sample(z,p,j) for j in range(len(output))])
   def score(v,label):
    t=metrics(y[output],v,z['sun'][output]);values=v.sum((2,3))*.25
    for a in [.025,.05,.10,.50,.90,.95,.975]:
     delta=en[output]-np.quantile(values,a,axis=1);t[f'pin_{a}']=np.maximum(a*delta,(a-1)*delta);t[f'below_{a}']=delta<=0
    t['s04_nominal']=t.site1_energy_CRPS/20;t['s04_train_scale']=t.site1_energy_CRPS/np.quantile(y[ti,:,1],.995)
    t['shape']=label;t['day']=ds[output];t['role']=np.select([np.isin(output,vi),np.isin(output,ei)],['validation','evaluation'],'out_of_time');return t
   scores=[score(paths,'path')]
   if name=='M3':scores.append(score(np.stack([sample(z,p,j,structure=False) for j in range(len(output))]),'unchanged_shape'))
   t=pd.concat(scores);t.to_csv(P/'scores'/(tag+'.csv'),index=False)
   pd.DataFrame(log).to_csv(P/'logs'/(tag+'.csv'),index=False)
   dump(P/'logs'/(tag+'.json'),dict(train=ds[ti].tolist(),validation=ds[vi].tolist(),output=ds[output].tolist(),device=torch.cuda.get_device_name(),seconds=time.perf_counter()-clock,peak_bytes=torch.cuda.max_memory_allocated(),tag=tag));print(tag,'validation CRPS',round(t[(t.role=='validation')&(t['shape']=='path')].energy_crps.mean(),2),flush=True)
if __name__=='__main__':train(sys.argv[1],sys.argv[2].split(','),sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),sys.argv[6].split(','))

