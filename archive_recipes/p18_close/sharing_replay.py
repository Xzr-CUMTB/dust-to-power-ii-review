"""Symmetric fresh P18 36-day factorial replay; no historical output is reused as actions."""
from close_common import *
import numpy as np,pandas as pd,time
sys.path[:0]=[str(R/'scripts/p16'),str(R/'scripts/p15'),str(R/'scripts/p14'),str(R/'scripts/p17_ext/bridge')]
import plant14
from src.planning.p9.dispatch import execute as original_execute,scenario_gaps

def inputs():
    z=np.load(R/'results/p5/dataset.npz');ix=np.flatnonzero(z['split']!=0)
    v=np.load(R/'results/p6/forecasts/F1_members.npz');assert np.array_equal(v['ids'],ix)
    ds=pd.to_datetime(z['t'][ix],utc=True).tz_convert('Asia/Shanghai').strftime('%Y-%m-%d').to_numpy()
    sites=['00','04','07','08'];cap=z['cap'].astype(float)
    ratios=json.loads((R/'results/p4_h4_storage/protocol.json').read_text())['plan_fit']['ratios']
    days=sorted(d for d in np.unique(ds) if '2019-05-08'<=d<='2019-06-12' and (ds==d).sum()==96)
    assert len(days)==36
    for n,day in enumerate(days):
        i=np.flatnonzero(ds==day)
        orig=v['paths'][i].astype(float)*cap[None,None,:,None]
        obs=z['y'][ix[i],:,0].astype(float)*cap[None,:]
        ref=z['clear'][ix[i]].astype(float)*cap[None,:,None]/1000*np.array([ratios[s] for s in sites])[None,:,None]
        perm=orig.copy();rng=np.random.default_rng(18042+n)
        for k in range(4):perm[:,:,k,:]=perm[:,np.r_[0,rng.permutation(np.arange(1,32))],k,:]
        assert np.array_equal(np.sort(orig,axis=1),np.sort(perm,axis=1))
        x=np.stack([plant14.prepared(s,day)[5] for s in sites],axis=1)
        yield day,sites,cap,orig,perm,obs,ref,x

def run(version):
    out=O/'sharing'/version;out.mkdir(exist_ok=True,parents=True)
    if version=='original':execute=original_execute
    else:
        from revised import execute
    lock(C/'sharing_replay_lock.json',dict(days=json.loads((R/'configs/p18/sharing_C_lock.json').read_text())['days'],
        pairing='original P18 final: member0 held in per-station pairing; isolated input identical',
        resources=dict(E=33.3,P=16.65,R=9.99,B=25.,initial_terminal=3.33),commitment=.7,
        coefficients='frozen soc42 weights with original same-source feature computation',
        versions=['original','revised'],evaluation='fixed original plans only; technical C-class revision',
        weights='32 equal complete trajectories, no special meaning of member0',
        statistical_unit='common calendar week; 2000 paired bootstrap17042'))
    rows=[];manifest=[];t=time.perf_counter()
    for day,sites,cap,orig,perm,obs,ref,x in inputs():
        for method in ('I0','I1','I2','I3'):
            pool=method in ('I2','I3');paths0=perm if method=='I2' else orig
            for k in ([None] if pool else range(4)):
                label='pooled' if pool else sites[k];p=out/'actions'/method/(day+'_'+label+'.npz')
                fac=1. if pool else cap[k]/cap.sum()
                yn=(obs.sum(1) if pool else obs[:,k])/fac
                vn=(paths0.sum(2) if pool else paths0[:,:,k,:])/fac
                rn=(ref.sum(1) if pool else ref[:,k,:])/fac
                xx=np.average(x,axis=1,weights=cap) if pool else x[:,k]
                pl=rn*.7;qs=np.quantile(vn,[.2,.5,.8],axis=1).transpose(1,0,2).copy()
                inn=np.r_[0.,0.,(yn-qs[:,1,0])[:-2]]
                ft=plant14.value_features(dict(x=xx,pm=np.zeros((96,4,6)),ref=rn),vn,np.arange(96),.7,'no')
                w=plant14.coeff(ft,'soc',42)
                a=execute(yn,vn,pl,scenario_gaps(vn,qs,pl,inn),w,qs[:,2,:].copy(),np.full(96,.5),3,plant14.fixed('anchor'),33.3,16.65,9.99,25.,3.33,True)*fac
                p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,a=a)
                residual=np.max(abs(a[:,0]+a[:,3]+a[:,4]+a[:,5]-a[:,1]-a[:,2]-a[:,6]))
                assert residual<1e-6 and abs(a[0,7]-a[-1,8])<1e-6
                old=np.load(R/'results/p18/sharing/actual_actions'/method/(day+'_'+label+'.npz'))['a']
                row=dict(day=day,method=method,site=label,plan=.25*a[:,1].sum(),shortfall=.25*a[:,5].sum(),
                    reserve=.25*a[:,4].sum(),throughput=.25*a[:,2:4].sum(),spill=.25*a[:,6].sum(),
                    initial=a[0,7],terminal=a[-1,8],mass_residual=residual,
                    max_original_action_difference=float(abs(a-old).max()),
                    changed_quarters=int(np.any(abs(a-old)>1e-8,axis=1).sum()))
                row['delivered']=row['plan']-row['shortfall'];row['rate']=row['delivered']/row['plan']
                row['daily95']=row['rate']>=.95;row['daily90']=row['rate']>=.9
                run=longest=0
                for flag in a[:,5]>1e-7:run=run+1 if flag else 0;longest=max(longest,run)
                row['longest_shortfall_hours']=longest*.25
                rows.append(row);manifest.append(dict(path=p.relative_to(R).as_posix(),sha256=sha(p),day=day,method=method,site=label))
        pd.DataFrame(rows).to_csv(out/'by_site.csv',index=False)
        print(version,day,len(rows),'seconds',round(time.perf_counter()-t,1),flush=True)
    df=pd.DataFrame(rows);pd.DataFrame(manifest).to_csv(out/'action_manifest.csv',index=False)
    metrics=['plan','shortfall','reserve','throughput','spill','initial','terminal','delivered']
    daily=df.groupby(['method','day'])[metrics].sum().reset_index();daily['rate']=daily.delivered/daily.plan
    daily['daily95']=daily.rate>=.95;daily['daily90']=daily.rate>=.9
    daily.to_parquet(out/'paired_day_results.parquet',index=False)
    totals=daily.groupby('method')[metrics].sum();totals['rate']=totals.delivered/totals.plan
    totals['daily95']=daily.groupby('method').daily95.mean();totals['daily90']=daily.groupby('method').daily90.mean()
    totals.to_csv(out/'engineering.csv')
    days=sorted(daily.day.unique());weeks=pd.to_datetime(days).to_period('W-SUN').astype(str);u=sorted(set(weeks))
    ww=np.random.default_rng(17042).multinomial(len(u),np.full(len(u),1/len(u)),2000)
    ww=ww[:,[u.index(k) for k in weeks]];effects=[]
    for left,right,name in [('I1','I0','information_under_islands'),('I2','I0','pooling_under_independent'),('I3','I2','information_under_pooling'),('I3','I1','pooling_under_joint')]:
        a=daily[daily.method==left].set_index('day').loc[days]
        b=daily[daily.method==right].set_index('day').loc[days]
        for m in ['plan','delivered','shortfall','reserve','throughput','spill']:
            delta=(a[m]-b[m]).to_numpy();boot=(ww@delta)*len(days)/ww.sum(1)
            effects.append(dict(comparison=name,metric=m,difference=delta.sum(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),unit='MWh',days=36))
    pd.DataFrame(effects).to_csv(out/'paired_effects.csv',index=False)
    write(out/'COMPLETE.json',dict(version=version,days=36,simulations=len(rows),steps=96*len(rows),seconds=time.perf_counter()-t,
        max_original_action_difference=df.max_original_action_difference.max(),changed_quarters=int(df.changed_quarters.sum()),
        max_mass_residual=df.mass_residual.max(),input_lock_sha256=sha(C/'sharing_replay_lock.json'),code_sha256=sha(__file__)))
if __name__=='__main__':run(sys.argv[1])
