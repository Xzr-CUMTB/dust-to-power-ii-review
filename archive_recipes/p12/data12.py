from common12 import *
from functools import lru_cache
SCALE=np.array([20.,10.,25.])
BGRID=np.repeat(BUDGETS,len(GRID))
CGRID=np.tile(GRID,len(BUDGETS))
TARGETS=[.90,.95]
def largest(gap,plan,target,indices):
    valid=indices[gap[indices]<=plan[indices]*(1-target)+1.e-10]
    return int(valid[-1] if len(valid) else indices[0])
@lru_cache(None)
def packet(source,fold):
    return dict(np.load(OUT/'predictions'/f'{source}_{fold}.npz'))
@lru_cache(None)
def truth(fold):
    dev=pd.read_parquet(P11/'devgrid.parquet')
    actual=pd.concat([dev[dev.fold==fold],pd.read_parquet(OUT/'grid'/f'{fold}.parquet')])
    actual=actual.sort_values(['day','B','c'])
    return {d:g[['shortfall','reserve','throughput']].to_numpy() for d,g in actual.groupby('day')}
@lru_cache(None)
def incumbent(fold):
    past=pd.read_parquet(P11/'incumbent_oof.parquet')
    val=pd.read_parquet(P11/'executed_daily.parquet')
    val=val[val.variant=='D0online']
    return {(r.day,r.target):key_index(25,r.c) for r in pd.concat([past[past.fold==fold],val[val.fold==fold]]).itertuples()}
@lru_cache(None)
def energy_truth():
    z=zload()
    return dict(zip(z['days'],z['y'].sum((1,2))*.25))
def candidates(p,i,history,mode='full'):
    context=p['context'][i].copy()
    # positions 16..23: regional level, anomaly, validity and age; 24..25: wind-weighted anomaly.
    if mode=='none':context[16:]=0
    if mode=='region':context[24:]=0
    plan=p['plan'][i]
    features=np.column_stack([np.tile(context,(91,1)),CGRID,plan/350,BGRID/25,
              p['proxy'][i]/SCALE,np.tile(history,(91,1)),np.tile([3.33/33.3,33.3/50,16.65/25,9.99/15,.95],(91,1))])
    return features
def history_features(records):
    if not records:return np.zeros(4)
    latest=records[-7:]
    return np.r_[np.mean([r['energy_error'] for r in latest])/100,
                 np.mean([r['raw_residual'] for r in latest],axis=0)/SCALE].clip(-5,5)
def historical_features(p,fold,mode):
    labels=truth(fold);visible=[];X=[];Y=[];mask=[]
    for i,d in enumerate(p['days'][:int(p['dev_count'])]):
        eligible=[h for h in visible if age(d,h['day'])>=2]
        X.append(candidates(p,i,history_features(eligible),mode))
        raw=p['proxy'][i];plan=p['plan'][i];res=labels[d]-raw
        # Complete counterfactual grid with each weather day equally weighted.
        chosen=np.array([largest(raw[:,0],plan,t,np.arange(b*13,b*13+13)) for t in TARGETS for b in range(7)])
        w=np.full(91,.5/91);np.add.at(w,chosen,.5/len(chosen));mask.append(w)
        j=largest(raw[:,0],plan,.95,np.arange(78,91))
        visible.append(dict(day=d,raw_residual=res[j],energy_error=energy_truth()[d]-p['median'][i]))
        Y.append(res/SCALE)
    return np.array(X),np.array(Y),np.array(mask)

