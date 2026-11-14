"""Build the issue-time inputs independently of R3 outcome labels."""
from common12 import *
from policy11 import quantcal, accounts, modelpath, getoof, past_day
from common11 import sample, basepaths, cr
from functools import lru_cache
@lru_cache(30)
def load_pack(path):
    return dict(np.load(path))
def paths_at(path,day):
    bundle=load_pack(path)
    return bundle['paths'][list(bundle['days']).index(day)].astype(float).sum(-1)
def prepare(source):
    catalog=dates(); z=zload(); source_days=list(z['days']); E=z['y'].sum((1,2))*.25
    w=pd.read_csv(P11/'weather_all.csv');w=w[w.eligible].groupby(['day','field','h']).value.mean().unstack(['field','h'])
    refs=np.load(P11/'refs.npz'); rr=dict(zip(refs['days'],refs['ref']))
    locks=pd.read_csv(P10/'validation_lock.csv').query("source=='operational_GFS'").set_index('fold')
    base=basepaths(z) if source in ['D0','P10'] else None
    def d0(f,day):
        path=P9/'scenarios'/f'{f}_D0_full_42_r3_h0.npz'
        if day in load_pack(path)['days']:return paths_at(path,day),'P9_D0online'
        i=source_days.index(day); past=np.flatnonzero(z['days']<=past_day(day))[-12:]
        ep=np.median(base[past].sum((2,3))*.25,1)
        ratio=np.exp(np.median(np.log((E[past]+5)/(ep+5)))).clip(.65,1.65)
        return base[i].astype(float).sum(-1)*ratio,'history_D0online'
    meta=[];metric=[];sanity=[]
    for f,(training,evaluation) in catalog.items():
        out=[];contexts=[];raw_energy=[];history=[];model='M1match' if source=='M1' else 'M3'
        for day in training+evaluation:
            is_eval=day in evaluation; i=source_days.index(day); cutoff=past_day(day)
            support=None
            if source=='D0':
                v,origin=d0(f,day)
            elif source=='P10':
                origin=None
                validfolds=[f] if is_eval else list(reversed(FOLDS[:FOLDS.index(f)+1]))
                for oldfold in validfolds:
                    vi=sp(z,oldfold)[1]
                    if z['days'][vi[-1]]>cutoff:continue
                    sel=locks.loc[oldfold]
                    path=P10/'scenarios'/f'{oldfold}_full_{sel.weather}_42_r5_{sel.model}.npz'
                    if day in load_pack(path)['days']:
                        v=paths_at(path,day);origin=path.stem;break
                if origin is None:
                    assert not is_eval
                    v,origin=d0(f,day);origin='before_P10_available_'+origin
            else:
                origin=f if is_eval else getoof(day)
                pp=load_pack(modelpath(origin,'W2',model,2,42));j=list(pp['days']).index(day)
                shape=source!='M3shape'
                v=sample(z,pp,j,structure=shape).astype(float).sum(-1)
                support=len(np.unique(pp['ids'][j]))/32
            raw=v.sum(1)*.25
            eligible=[row for row in history if row[0]<=cutoff]
            if source in ['M1','M3','M3shape']:
                # For the shape intervention transport exactly the same calibrated energy as the full shape.
                if source=='M3shape':
                    full=sample(z,pp,j,structure=True).astype(float).sum(-1)
                    raw=full.sum(1)*.25
                    en=quantcal(full,eligible).sum(1)*.25
                    v*= (en/(v.sum(1)*.25).clip(.00001))[:,None]
                else:v=quantcal(v,eligible)
            if not is_eval:history.append((day,E[i]-np.quantile(raw,Q)))
            e=v.sum(1)*.25; qs=np.quantile(e,[.05,.1,.5,.9,.95])
            support=len(np.unique(e.round(3)))/32 if support is None else support
            pm=z['pm'][i];wind=z['w'][i];weather=np.r_[w.loc[day,'rad'].to_numpy()/700,w.loc[day,'cloud'].to_numpy()/100]
            context=np.r_[weather,np.sin(np.deg2rad(z['sun'][i])).clip(0).mean(),qs/300,(qs[-1]-qs[0])/250,support,pm.mean(0),np.average(pm[:,[1,5]],axis=0,weights=wind+.01)]
            out.append([accounts(v,rr[day],b)[:,:3] for b in BUDGETS]);contexts.append(context);raw_energy.append(np.median(e))
            meta.append(dict(fold=f,day=day,source=source,role='evaluation' if is_eval else 'forward_development',origin=origin,latest_whole_day_allowed=cutoff,available_issue=day+' 00:00+08:00',power_label_available=str(pd.Timestamp(day,tz='Asia/Shanghai')+pd.Timedelta('1d15min')),K1_count=len(eligible) if source.startswith('M') else None))
            lo,hi=qs[0],qs[-1];metric.append(dict(fold=f,day=day,source=source,eval=is_eval,CRPS=cr(e[None],np.array([E[i]]))[0],width90=hi-lo,coverage90=lo<=E[i]<=hi,interval90=hi-lo+20*max(lo-E[i],E[i]-hi,0),q10loss=max(.1*(E[i]-qs[1]),-.9*(E[i]-qs[1])),median=qs[2]))
        days=training+evaluation;plans=np.stack([np.tile(GRID*rr[d].sum()*.25,len(BUDGETS)) for d in days])
        proxy=np.array(out).reshape(len(days),91,3)
        if source in ['M1','M3']:
            old=pd.read_parquet(P11/f'candidates_W2_{model}_r2_s42_n32.parquet')
            old=old[(old.fold==f)&(old.cal=='K1')].sort_values(['day','B','c'])
            diff=abs(proxy-old[['proxy_gap','proxy_reserve','proxy_throughput']].to_numpy().reshape(proxy.shape)).max()
            assert diff<.00005,(source,f,diff);sanity.append(dict(fold=f,max_abs_diff=diff))
        np.savez_compressed(OUT/'predictions'/f'{source}_{f}.npz',days=np.array(days,str),dev_count=len(training),context=contexts,proxy=proxy,plan=plans,median=raw_energy)
    pd.DataFrame(meta).to_csv(OUT/'predictions'/f'{source}_origins.csv',index=False)
    pd.DataFrame(metric).to_csv(OUT/'predictions'/f'{source}_scores.csv',index=False)
    dump(OUT/'predictions'/f'{source}_checks.json',sanity)
    print(source,len(meta),'fold-days',flush=True)
if __name__=='__main__':
    for source in sys.argv[1:]:prepare(source)

