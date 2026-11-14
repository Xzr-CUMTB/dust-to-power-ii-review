import r3env
from common11 import *
from src.operations.p10.budget import scenario_account,pick_plan
from src.dispatch.p11.selected import correct
from engine import GRID
from train import all_dispatch_days
from functools import lru_cache
BUD=np.arange(10,26,2.5);FOLDS=['early','anchor','later']
@lru_cache(40)
def packp(file):return dict(np.load(file))
def packet(z,file,d,n):
 p=packp(str(file));j=p['days'].tolist().index(d)
 if 'paths' in p:
  assert n==32
  return p['paths'][j].astype('f8').sum(-1)
 return sample(z,p,j,n).astype('f8').sum(-1)
def modelpath(f,wx,m,r,s):return P/'forecasts'/f'{f}_{wx}_full_{m}_r{r}_s{s}.npz'
def past_day(d):return str((pd.Timestamp(str(d))-pd.Timedelta('1d15min')).date())
def getoof(d):return 'oof'+str((pd.Timestamp(str(d))-pd.Timestamp('2019-03-25')).days//14)
def components(z,d,wx,m,r,s,n,f=None):
 f=f or getoof(d);return packet(z,modelpath(f,wx,m,r,s),d,n)
def accounts(v,rr,b):
 return np.stack([scenario_account(v,rr,c,b).mean(0) for c in GRID])
def quantcal(v,old):
 E=v.sum(1)*.25;u=(np.arange(len(v))+.5)/len(v);rank=np.argsort(np.argsort(E,kind='stable'),kind='stable')
 if len(old)<4:return v
 residual=np.stack([r[1] for r in old[-28:]])
 offset=np.array([np.quantile(residual[:,j],q) for j,q in enumerate(Q)])*(len(residual)/(len(residual)+14))
 target=np.sort(np.maximum(.01,np.quantile(E,Q)+offset))
 new=np.interp(u,Q,target)[rank]
 return v*(new/E.clip(.001))[:,None]
def scene_development(z,refs):
 from engine import replay,stats
 t=[];ds=list(map(str,sorted(refs)[:8]))
 for m,r in [('M1',1),('M3',2)]:
  for d in ds:
   for n in [32,128,256]:
    v=components(z,d,'W2',m,r,42,n);a=accounts(v,refs[d],25);en=refs[d].sum()*.25*GRID
    for target in [.9,.95]:
     j=pick_plan(a[:,0],en,target,0.);obs=stats(replay('early',d,GRID[j],25.))
     t.append(dict(model=m,day=d,n=n,target=target,c=GRID[j],forecast_gap=a[j,0],**obs))
 t=pd.DataFrame(t);t.to_csv(P/'scene_development.csv',index=False)
 p=t.merge(t[t.n==256],on=['day','model','target'],suffixes=('','_ref'));p['sw']=p.c!=p.c_ref;p['err']=abs(p.forecast_gap-p.forecast_gap_ref)
 s=p.groupby(['model','n'])[['sw','err']].mean().reset_index();s.to_csv(P/'scene_convergence.csv',index=False)
 n=min(k for k,g in s.groupby('n') if g.sw.max()<=.05 and g.err.max()<=.5)
 dump(P/'scene_lock.json',dict(n=int(n),days=ds,prior_to_evaluation=True));return int(n)
def run(wx,m,r,seed,nopt):
 z=zload();ds=z['days'];ys=dict(zip(ds,z['y'].sum((1,2))*.25));rp=np.load(P/'refs.npz');refs=dict(zip(rp['days'],rp['ref']));dev=pd.read_parquet(P/'devgrid.parquet')
 if nopt=='resolve':return scene_development(z,refs)
 n=int(nopt) if nopt!='lock' else int(json.loads((P/'scene_lock.json').read_text())['n'])
 tag=f'{wx}_{m}_r{r}_s{seed}_n{n}';outfile=P/f'issue_{tag}.parquet'
 if outfile.exists():return
 sums=['shortfall','reserve','throughput'];truth={(t.fold,t.day,t.c,t.B):np.array([t.shortfall,t.reserve,t.throughput]) for t in dev.itertuples()}
 cand=[];issues=[];cal=[];dc=[];calfc=[];kout={}
 for f in FOLDS:
  ev=ds[sp(z,f)[3]].tolist();dates=sorted(dev[dev.fold==f].day.unique());tune_days=dates[:max(4,min(8,len(dates)//3))]
  histories={};past=[];oldmargin={};pre={}
  for d in dates+ev:
   test=d in ev;k=(f,d);rr=refs[d];en=rr.sum()*.25*GRID
   v=components(z,d,wx,m,r,seed,n,f if test else None);ener=v.sum(1)*.25;eligible=[a for a in past if a[0]<=past_day(d)]
   vc=quantcal(v,eligible)
   for kver,pv in [('K0',v),('K1',vc)]:
    evv=pv.sum(1)*.25;obs=ys[d];low,med,hi=np.quantile(evv,[.05,.50,.95]);loss=obs-np.quantile(evv,Q)
    calfc.append(dict(fold=f,day=d,eval=test,version=kver,n_mature=len(eligible),latest=eligible[-1][0] if eligible else '',energy_crps=cr(evv[None],np.array([obs]))[0],cover90=low<=obs<=hi,width90=hi-low,interval90=hi-low+20*max(low-obs,0,obs-hi),q05_loss=max(.05*(obs-low),-.95*(obs-low)),q10_loss=max(.1*loss[3],-.9*loss[3]),q50_loss=.5*abs(obs-med)))
   if not test:past.append((d,ys[d]-np.quantile(ener,Q))) # frozen calibration set for each evaluation block
   else:assert not eligible or eligible[-1][0]<d
   for b in BUD:
    for ki,vv in [('K0',v),('K1',vc)]:
     acc=accounts(vv,rr,b);E=vv.sum(1)*.25;med=np.median(E);width=np.quantile(E,.95)-np.quantile(E,.05)
     feat=np.column_stack([np.full(13,med/300),np.full(13,width/200),GRID*2,en/400,np.full(13,b/25),acc[:,0]/en])
     for j,c in enumerate(GRID):cand.append((f,d,test,ki,b,c,en[j],*acc[j,:3]))
     for target in [.90,.95]:
      mark=(ki,b,target)
      if not test and d in tune_days:
       for margin in [0.,.015,.035]:
        j=pick_plan(acc[:,0],en,target,margin);a=truth[f,d,GRID[j],b]
        oldmargin.setdefault((mark,margin),[]).append((d,en[j],a[0]))
      def score(margin):
       hh=[a for a in oldmargin.get((mark,margin),[]) if a[0]<=past_day(d)]
       if not hh:return (0,0,margin)
       pe=sum(a[1] for a in hh);return (max(0,sum(a[2] for a in hh)/pe-(1-target)),-pe,margin)
      margin=min([0.,.015,.035],key=score)
      modes=[('K0',28)] if ki=='K0' else [('K1',28),('uniform',28),('K2global',28),('K2',28),('K2',14)]
      for rule,window in modes:
       key=(b,target,rule,window);hist=histories.setdefault(key,[]);hh=[a for a in hist if a[0]<=past_day(d)]
       correction=correct(hh,feat,window,rule=='K2') if rule.startswith('K2') else np.zeros((13,3))
       pred=acc[:,:3]+correction*np.column_stack([en,np.full(13,25),np.full(13,60)])
       pred[:,0]=pred[:,0].clip(0,en);pred[:,1]=pred[:,1].clip(0,b);pred[:,2]=pred[:,2].clip(0)
       j=pick_plan(pred[:,0],en,target,margin+(.015 if rule=='uniform' else 0));c=GRID[j]
       issues.append(dict(fold=f,day=d,phase='eval' if test else 'OOF',variant=tag,target=target,rule=rule,window=window,c=c,B=b,margin=margin,expected_gap=pred[j,0],expected_reserve=pred[j,1],expected_throughput=pred[j,2],mature=len(hh[-window:]),latest=hh[-1][0] if hh else '',plan=en[j],coefficient_fallback=not np.any(pred[:,0]<=en*(1-target-margin-(.015 if rule=='uniform' else 0)))))
       if not test:
        obs=truth[f,d,c,b];hist.append((d,feat[j],(obs-acc[j,:3])/np.array([en[j],25,60])))
        cal.append(dict(fold=f,day=d,model=tag,target=target,B=b,rule=rule,window=window,c=c,margin=margin,actual_gap=obs[0],actual_reserve=obs[1],actual_throughput=obs[2],pred_gap=pred[j,0],pred_reserve=pred[j,1],pred_throughput=pred[j,2],baseline_gap=acc[j,0],baseline_reserve=acc[j,1],baseline_throughput=acc[j,2],nfit=len(hh[-window:]),latest=hh[-1][0] if hh else ''))
  print(f,tag,len(issues),flush=True)
 pd.DataFrame(cal).to_parquet(P/f'selected_residuals_{tag}.parquet',index=False);pd.DataFrame(cand,columns=['fold','day','eval','cal','B','c','plan','proxy_gap','proxy_reserve','proxy_throughput']).to_parquet(P/f'candidates_{tag}.parquet',index=False)
 pd.DataFrame(calfc).to_csv(P/f'calibrated_forecast_{tag}.csv',index=False);pd.DataFrame(issues).to_parquet(outfile,index=False)
if __name__=='__main__':run(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])

