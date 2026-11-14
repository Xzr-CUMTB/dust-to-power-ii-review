"""C-only finite-candidate hindsight diagnostics, never imported by online policies."""
from pathlib import Path
import os,sys,json,hashlib,time
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[v]='1'
import numpy as np
import pandas as pd
from scipy.optimize import milp,linprog,Bounds,LinearConstraint
from scipy.sparse import csr_matrix
R=Path(__file__).absolute().parents[3]
O=R/'results/p18/opportunity';C=R/'configs/p18/opportunity'
for p in [O,C,O/'grids',O/'selected',O/'actual_actions']:p.mkdir(exist_ok=True,parents=True)
REF='D0_E0_s42_F0_equal_r1_D0plan';MAIN='M3_E0_s42_F0_equal_r1';TREE='M1_E0_s42_F0_equal_r1'
Q=['plan','delivered','shortfall','reserve','throughput','spill']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def scope(site):
    if site=='2107':return R/'results/p17_ext/confirmation/2107/runtime',R/'results/p17_ext/confirmation/2107/replay','CA2107'
    if site in ['00','04','07','08']:return R/'results/p17/local_forward',R/'results/p17/component_replay','PVOD_component'
    return R,R/'results/p17/historical_replay','PVOD_cluster' if site=='core' else 'PVOD_single'
def bind(site):
    runtime,base,identity=scope(site);sys.path.insert(0,str(runtime/'scripts/p16'))
    import policy16 as p
    from physical15 import factor
    assert p.ROOT==runtime
    return p,float(factor(site)),base
def freeze():
    files=[R/'configs/p17/method_lock.json',R/'configs/p17_ext/systems/2107/eligible_calendar_lock.json',R/'results/p17_ext/confirmation/2107/issued_daily.parquet',R/'results/p17/daily_replays.parquet']
    s=dict(classification='C-only previously studied data; evaluation offline boundaries only',sites=['core','01','03','05','06','09','00','04','07','08','2107'],targets=[.95,.9],candidate_order='91: B outer [10,12.5,...25], coefficient inner [.30,.35,...,.90]',O='per-weather-day one candidate; cumulative gap <= (1-target)*commitment; reserve and AC throughput separately <= original D0 actual totals. H not separable because reservations modify actions, no H candidate sums as oracle.',shortfall='Same original day commitments, same two actual-resource caps: minimize sum gap within available finite91 candidates; lower bound to physically possible saving. CA adds separate perfect-information LP-relaxation upper bound.',solver=dict(seconds=60,mip_relative_gap=1e-8,resource_validation_MWh=1e-6),hashes={str(p.relative_to(R)):sha(p) for p in files})
    f=C/'lock.json'
    if f.exists():assert json.loads(f.read_text(encoding='utf8'))==s
    else:write(f,s)
def grid(site):
    assert (C/'lock.json').exists();p,factor,base=bind(site)
    g=pd.read_parquet(base/'issues/O'/site/REF/'0.95.parquet').sort_values('day')
    folder=O/'grids'/site;folder.mkdir(exist_ok=True);rows=[];tic=time.perf_counter();maxerr=0.
    for r in g.itertuples():
        dest=folder/f'{r.day}.npz'
        if dest.exists():
            data=np.load(dest);assert sha(dest)==json.loads(dest.with_suffix('.json').read_text())['sha256'];continue
        a=[]
        for j in range(91):
            v=p.ordinary(site,str(r.fold),r.day,j)*factor
            assert v.shape[0]==96
            err=max(float(abs(v[:,0]+v[:,3]+v[:,4]+v[:,5]-v[:,1]-v[:,2]-v[:,6]).max()),float(abs(v[:,7]+.25*(.95*v[:,2]-v[:,3]/.95)-v[:,8]).max()))
            assert err<1e-6;maxerr=max(maxerr,err)
            a.append(v)
        a=np.array(a)
        vals=np.column_stack([a[:,:,1].sum(1),a[:,:,1].sum(1)-a[:,:,5].sum(1),a[:,:,5].sum(1),a[:,:,4].sum(1),a[:,:,2:4].sum((1,2)),a[:,:,6].sum(1)])*.25
        assert np.max(abs(vals[int(r.j)]-np.array([getattr(r,k) for k in Q])))<1e-6
        # Archived group actions are re-evaluated before any new oracle solve.
        with np.load(base/'actions/eval/O'/site/REF/'0.95.npz') as old:
            ix=list(old['days']).index(r.day);assert np.max(abs(a[int(r.j)]-old['a'][ix]))<1e-6
        np.savez_compressed(dest,values=vals,pv=a[0,:,0],reference_plan=a[int(r.j),:,1],reference_action=a[int(r.j)],factor=factor,fold=str(r.fold),day=r.day)
        write(dest.with_suffix('.json'),dict(site=site,day=r.day,sha256=sha(dest),candidates=91,reference_original_action=int(r.j),rows=96))
    for dest in sorted(folder.glob('*.npz')):
        z=np.load(dest);values=z['values']
        for j,v in enumerate(values):rows.append(dict(site=site,day=str(z['day']),fold=str(z['fold']),j=j,c=.30+.05*(j%13),B=(10+2.5*(j//13))*factor,**dict(zip(Q,v))))
    table=pd.DataFrame(rows);table.to_parquet(O/f'grid_{site}.parquet',index=False)
    write(O/f'grid_{site}_complete.json',dict(site=site,days=len(g),candidates=91,seconds=time.perf_counter()-tic,max_step_error=maxerr,factor=factor,source='fresh ordinary R3 execution; measured PV; no synthesized target'))
def archived(site):
    """Use previously validated full candidate outcomes for old core/5 systems."""
    k=['site','fold','day','j']
    plans=pd.read_parquet(R/'results/p14/candidate_plans.parquet',filters=[('site','==',site),('register','==','original91')])
    outcomes=pd.read_parquet(R/'results/p14/offline_candidate_outcomes.parquet',filters=[('site','==',site)])
    v=plans.merge(outcomes,on=k,validate='one_to_one');v=v[v.role=='reviewed_evaluation'].copy()
    v=v.rename(columns={'contract_MWh':'plan'})
    return v
def solve(site,t,fixed=False):
    runtime,base,identity=scope(site)
    d=pd.read_parquet(base/'issues/O'/site/REF/f'{t}.parquet').sort_values('day')
    if not fixed and site not in ['00','04','07','08','2107']:
        path=R/'results/p15/offline_oracle'/f'{site}_{t}_original.json'
        old=json.loads(path.read_text());assert abs(old['D0_delivery']-d.delivered.sum())<1e-6
        assert abs(old['D0_reserve']-d.reserve.sum())<1e-6 and abs(old['D0_throughput']-d.throughput.sum())<1e-6
        rec=dict(site=site,identity=identity,target=t,type='A_delivery_finite91',days=len(d),independent_new_weather=0,status=old['status'],reused_source=str(path.relative_to(R)),reused_sha256=sha(path),original_D0_plan=d.plan.sum(),original_D0_delivered=d.delivered.sum(),original_D0_gap=d.shortfall.sum(),original_D0_reserve=d.reserve.sum(),original_D0_throughput=d.throughput.sum(),H_oracle_status='not separable; not solved',**{k:old.get(k) for k in ['plan','delivered','shortfall','reserve','throughput','rate','gap','upper']})
        rec['A_delivery_attained']=rec['delivered']-rec['original_D0_delivered'];rec['relative_delivery_gain']=rec['A_delivery_attained']/rec['original_D0_delivered'];rec['A_shortfall_attained']=rec['original_D0_gap']-rec['shortfall']
        write(O/f'{site}_{t}_delivery.json',rec)
        return
    frame=pd.read_parquet(O/f'grid_{site}.parquet') if site in ['00','04','07','08','2107'] else archived(site)
    frame=frame[frame.day.isin(d.day)].sort_values(['day','j']).copy()
    assert set(frame.day)==set(d.day) and (frame.groupby('day').size()==91).all()
    if fixed:
        ref=dict(zip(d.day,d.plan));frame=frame[abs(frame.plan-frame.day.map(ref))<1e-6].copy()
    frame['excess']=frame.shortfall-(1-t)*frame.plan
    h=frame.drop_duplicates(['day','excess','reserve','throughput','delivered']).reset_index(drop=True)
    ix=pd.factorize(h.day,sort=True)[0];n=len(d);eq=csr_matrix((np.ones(len(h)),(ix,np.arange(len(h)))),shape=(n,len(h)))
    fields=['reserve','throughput'] if fixed else ['excess','reserve','throughput']
    A=csr_matrix(h[fields].to_numpy().T);bound=np.array([d.reserve.sum(),d.throughput.sum()]) if fixed else np.array([0.,d.reserve.sum(),d.throughput.sum()])
    c=h.shortfall.to_numpy() if fixed else -h.delivered.to_numpy()
    lp=linprog(c,A_ub=A,b_ub=bound,A_eq=eq,b_eq=np.ones(n),bounds=(0,1),method='highs')
    tic=time.perf_counter();sol=milp(c,integrality=np.ones(len(h)),bounds=Bounds(0,1),constraints=[LinearConstraint(eq,1,1),LinearConstraint(A,-np.inf,bound)],options={'time_limit':60.,'mip_rel_gap':1e-8})
    tag=f'{site}_{t}_'+('fixed_commitment' if fixed else 'delivery')
    rec=dict(site=site,identity=identity,target=t,type='A_shortfall_finite_attainable' if fixed else 'A_delivery_finite91',days=n,independent_new_weather=0,status=int(sol.status),solver_message=sol.message,solver_seconds=time.perf_counter()-tic,original_D0_plan=d.plan.sum(),original_D0_delivered=d.delivered.sum(),original_D0_gap=d.shortfall.sum(),original_D0_reserve=d.reserve.sum(),original_D0_throughput=d.throughput.sum(),H_oracle_status='not separable; not solved',finite_candidate_input='new execution' if site in ['00','04','07','08','2107'] else 'archived exact P14 outcomes',lp_relaxation_value=float(lp.fun) if lp.success else None)
    if sol.x is not None:
        chosen=h[sol.x>.5].copy();assert len(chosen)==n and chosen.day.nunique()==n
        use=chosen[fields].sum().to_numpy();assert np.max(use-bound)<1e-6
        rec.update({k:float(chosen[k].sum()) for k in Q if k in chosen});rec['rate']=rec['delivered']/rec['plan'];rec['A_shortfall_attained']=rec['original_D0_gap']-rec['shortfall'];rec['A_delivery_attained']=rec['delivered']-rec['original_D0_delivered'];rec['relative_delivery_gain']=rec['A_delivery_attained']/rec['original_D0_delivered']
        rec['mip_gap']=float(sol.mip_gap);rec['objective_bound']=float(sol.mip_dual_bound)
        rec['A_shortfall_upper_bound_MWh']=float(d.shortfall.sum()-sol.mip_dual_bound) if fixed else None
        rec['A_delivery_upper_bound_MWh']=float(-sol.mip_dual_bound-d.delivered.sum()) if not fixed else None
        chosen.assign(target=t,role='hindsight_not_deployable').to_parquet(O/'selected'/f'{tag}.parquet',index=False)
    write(O/f'{tag}.json',rec)
def replay(site):
    p,f,base=bind(site);dest=O/'actual_actions'/site;dest.mkdir(exist_ok=True,parents=True);rows=[]
    for file in sorted((O/'selected').glob(site+'_*.parquet')):
        selected=pd.read_parquet(file).sort_values('day');aa=[]
        for r in selected.itertuples():
            v=p.ordinary(site,str(r.fold),r.day,int(r.j))*f
            y=np.array([v[:,1].sum(),v[:,1].sum()-v[:,5].sum(),v[:,5].sum(),v[:,4].sum(),v[:,2:4].sum(),v[:,6].sum()])*.25
            expect=np.array([getattr(r,k,y[ix]) for ix,k in enumerate(Q)])
            assert np.max(abs(y-expect))<1e-6
            aa.append(v)
        np.savez_compressed(dest/(file.stem+'.npz'),days=selected.day,j=selected.j,a=aa,columns=np.array(p.COLS))
        rows.append(dict(site=site,oracle=file.stem,days=len(selected),steps=len(selected)*96,max_output_error=0.))
    pd.DataFrame(rows).to_csv(O/f'replay_{site}.csv',index=False)
if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='freeze':freeze()
    elif cmd=='grid':grid(sys.argv[2])
    elif cmd=='solve':
        for t in [.95,.9]:
            for fixed in [False,True]:solve(sys.argv[2],t,fixed)
    elif cmd=='fixed':
        for t in [.95,.9]:solve(sys.argv[2],t,True)
    elif cmd=='replay':replay(sys.argv[2])
