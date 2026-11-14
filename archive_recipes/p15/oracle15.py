"""Isolated evaluation-only multiple-choice optimizer on actual R3 outcomes."""
from paths15 import *
import time
import numpy as np
import pandas as pd
from scipy.optimize import Bounds,LinearConstraint,linprog,milp
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor

def solve(args):
    site,t,kind=args
    folder=OUT/'offline_oracle'
    key=['site','fold','day','j']
    p=pd.read_parquet(OLD/'candidate_plans.parquet',filters=[('site','==',site),('register','==','original91')])
    y=pd.read_parquet(OLD/'offline_candidate_outcomes.parquet',filters=[('site','==',site)])
    p=p.merge(y,on=key,validate='one_to_one')
    refs=pd.read_parquet(OLD/'reference/local_locks.parquet',filters=[('site','==',site),('source','==','D0'),('target','==',t)])
    ref=p.merge(refs[['day','fold','original_index']].rename(columns={'original_index':'j'}),on=['day','fold','j'],validate='many_to_one')
    dev=ref[(ref.role=='development')&(ref.fold==('early' if site=='core' else site))]
    base=ref[ref.role=='reviewed_evaluation']
    df=p[p.role=='reviewed_evaluation'].sort_values(['day','j']).copy()
    n=len(base)
    assert n==df.day.nunique() and (df.groupby('day').size()==91).all() and (df.contract_MWh>0).all()
    assert set(base.day)==set(df.day)
    sensitivity=kind in ['loosenGap','loosenReserve','loosenThroughput']
    if kind=='original' or sensitivity:budget=base[['reserve','throughput']].sum().to_numpy()
    elif kind=='B0':budget=n*dev[['reserve','throughput']].mean().to_numpy()
    else:
        rr=pd.read_parquet(OUT/'reference_resource_predictions.parquet')
        rr=rr[(rr.site==site)&(rr.target==t)&(rr.role=='eval')&rr.selected]
        assert len(rr)==n
        budget=rr[['reserve','throughput']].sum().to_numpy()
    df['excess']=df.shortfall-(1-t)*df.contract_MWh
    # Exact within-day dominance; drop equivalent duplicates only after a stable tie break.
    panels=[]
    for day,g in df.groupby('day'):
        h=g.drop_duplicates(['excess','reserve','throughput','delivered'])
        v=h[['excess','reserve','throughput']].to_numpy()
        v=np.column_stack([v,-h.delivered])
        dominates=(v[:,None,:]<=v[None,:,:]).all(2)&(v[:,None,:]<v[None,:,:]).any(2)
        panels.append(h.loc[~dominates.any(0)])
    h=pd.concat(panels,ignore_index=True)
    ids=pd.factorize(h.day,sort=True)[0]
    eq=csr_matrix((np.ones(len(h)),(ids,np.arange(len(h)))),shape=(n,len(h)))
    A=csr_matrix(h[['excess','reserve','throughput']].to_numpy().T)
    bound=np.r_[0.,budget]
    if sensitivity:
        station_scale=1. if site=='core' else json.loads((ROOT/'results/p13/stations'/site/'roles.json').read_text())['capacity']/66.6
        bound[['loosenGap','loosenReserve','loosenThroughput'].index(kind)]+=station_scale
    objective=-h.delivered.to_numpy()
    lp=linprog(objective,A_ub=A,b_ub=bound,A_eq=eq,b_eq=np.ones(n),bounds=(0,1),method='highs')
    start=time.perf_counter()
    sol=milp(objective,integrality=np.ones(len(h)),bounds=Bounds(0,1),constraints=[LinearConstraint(eq,1,1),LinearConstraint(A,-np.inf,bound)],options={'time_limit':30.,'mip_rel_gap':1e-7})
    tag=f'{site}_{t}_{kind}'
    rec=dict(site=site,target=t,budget=kind,days=n,D0_in_91=True,D0_delivery=base.delivered.sum(),D0_reserve=base.reserve.sum(),D0_throughput=base.throughput.sum(),gap_excess_limit=bound[0],reserve_limit=bound[1],throughput_limit=bound[2],status=sol.status,message=sol.message,seconds=time.perf_counter()-start,lp_upper=-lp.fun if lp.success else None,eligible_after_dominance=len(h))
    if lp.success:
        rec.update(dict(zip(['lp_shadow_gap','lp_shadow_reserve','lp_shadow_throughput'],-lp.ineqlin.marginals)))
    if sol.x is not None:
        picked=h.loc[sol.x>.5].copy()
        assert len(picked)==n==picked.day.nunique()
        assert (picked[['excess','reserve','throughput']].sum().to_numpy()<=bound+1e-5).all()
        rec.update(delivered=picked.delivered.sum(),plan=picked.contract_MWh.sum(),shortfall=picked.shortfall.sum(),reserve=picked.reserve.sum(),throughput=picked.throughput.sum(),rate=picked.delivered.sum()/picked.contract_MWh.sum(),upper=min(-float(sol.mip_dual_bound),-lp.fun),gap=sol.mip_gap)
        rec['gain']=rec['delivered']-rec['D0_delivery']
        picked.assign(target=t,budget=kind).to_parquet(folder/(tag+'.parquet'),index=False)
    write_json(folder/(tag+'.json'),rec)

if __name__=='__main__':
    physics=pd.read_csv(OLD/'physical_checks.csv')
    assert physics.terminal.max()<1e-6 and physics.soc.max()<1e-6
    tasks=[(s,t,k) for s in SITES for t in TARGETS for k in sys.argv[1:] if t==.95 or not k.startswith('loosen')]
    with ProcessPoolExecutor(3) as pool:list(pool.map(solve,tasks))

