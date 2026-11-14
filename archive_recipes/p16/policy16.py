from base16 import *
from adapt16 import quoted
from plant16 import hard, outputs, execute_day, COLS, H_COLS, Reservations
from concurrent.futures import ProcessPoolExecutor
from train15 import stage_ids

@lru_cache(100)
def forecast(site,src,head,seed):
    if head in ['E0','S1']:
        if site!='core' and seed!=42 and head=='E0':
            p=np.load(OUT/'forecasts'/f'{site}_{src}_s{seed}_E0.npz')
            return p['mean'],p['scale']
        p=np.load(PRE/'predictions'/f'{site}_{src}_{seed}.npz')
        key='neural' if head=='E0' else 'simple'
        return p[key],p[key+'_scale']
    p=np.load(OUT/'forecasts'/f'{site}_{src}_s{seed}_{head}.npz')
    return p['mean'],p['scale']

@lru_cache(4000)
def ordinary(site,fold,day,j):
    a,_=execute_day(site,fold,day,int(j))
    return a

def protected(site,fold,day,j,cap,branch):
    if branch=='O':return ordinary(site,fold,day,int(j))
    return hard(site,fold,day,int(j),cap)

def timing_path(site,t,phase,indices,b0):
    d=pd.read_parquet(PRE/'predictions'/f'reference_{site}.parquet')
    d=d[(d.target==t)&(d.role==phase)&d.selected].set_index('idx')
    cap=np.array([25.,399.6]);limit=np.minimum(b0,cap-b0);path=[]
    for k,i in enumerate(indices):
        if k==0 and len(indices)%2:value=b0.copy()
        elif (k-len(indices)%2)%2==0:
            shift=np.clip(d.loc[i,['reserve','throughput']].to_numpy(dtype=float)/float(data(site)['factor'])-b0,-limit,limit)
            value=b0+shift
        else:value=b0-shift
        path.append(value)
    return np.array(path)

def opportunity(site,src,phase,i,base,t):
    z=data(site,src);p=z['raw'][i];plan=z['plan'][i];values=[]
    for resource in range(2):
        utilities=[]
        for mult in [.65,1.35]:
            cap=base.copy();cap[resource]*=mult
            center,_,_=quoted(site,src,phase,i,p,np.zeros_like(p),cap)
            gap=center[:,0];utilities.append(np.max(plan-gap-3.*np.maximum(0.,gap-(1-t)*plan)))
        values.append((utilities[1]-utilities[0])/max(.1,.7*base[resource]))
    return np.array(values)

@lru_cache(100)
def weather_history(site,src,t,phase):
    z=data(site,src);split=int(z['split']);n=int(z['n']);stop=split if phase=='validation' else n
    days=z['days'].astype('datetime64[D]')
    prior=np.flatnonzero((np.arange(len(days))<stop)&(days<=days[stop]-np.timedelta64(2,'D')))
    b0=supply(site,t,phase)
    hist=np.stack([opportunity(site,src,phase,i,b0,t) for i in prior])
    median=np.median(hist,0);mad=np.median(abs(hist-median),0)
    return median,np.maximum(.1,1.4826*mad)

def allocation(site,src,phase,i,t,pace,b0,free,slots,scheduled):
    if pace=='fixed':proposed=b0
    elif pace=='timing':proposed=scheduled
    else:
        proposed=free/slots
        if pace=='weather':
            median,dev=weather_history(site,src,t,phase)
            score=opportunity(site,src,phase,i,b0,t)
            proposed=proposed*(1+.30*np.tanh((score-median)/dev))
    return np.minimum(np.minimum(np.maximum(0.,proposed),free),[25.,399.6])

def run(site,t,src,head,seed,fb,pace,rnd,branch,phase,fix=None,force=None,block_ids=None):
    z=data(site,src);center,width=forecast(site,src,head,seed)
    n=int(z['n']);split=int(z['split']);ti=TARGETS.index(t);factor=float(z['factor'])
    ids=np.arange(split,n) if phase=='validation' else np.arange(n,len(z['days']))
    if block_ids is not None:ids=np.array(block_ids,dtype=int)
    dates=z['days'].astype('datetime64[D]');ctx=z['ctx'][:,:12]
    eta,q,rev=settings(site,t);b0=supply(site,t,phase);total=len(ids)*b0
    book=Reservations(total);settled_debt=np.zeros(3);history=[];live={}
    rows=[];panel=[];risk=[];widths=[];steps=[];events=[];more=[]
    timing=timing_path(site,t,phase,ids,b0) if pace=='timing' else [None]*len(ids)
    label=tag(src,head,seed,fb,pace,rnd)+(('_'+force) if force else '')
    root=OUT if block_ids is None else OUT/'block_checks'/str(z['days'][ids[0]])
    path=root/('development' if phase=='validation' else 'issues')/branch/site/label
    path.mkdir(exist_ok=True,parents=True)
    frozen={}
    if force=='fixedO':
        ordinary_path=path.parents[2]/'O'/site/tag(src,head,seed,fb,'equal',rnd)/f'{t}.parquet'
        locked=pd.read_parquet(ordinary_path,columns=['day','j'])
        frozen=dict(zip(locked.day,locked.j))
    def release(k,date,at_close):
        r=live.pop(k);val=r['actual'];j=r['j']
        returned=book.settle(k,val[1:]) if branch=='H' else np.zeros(2)
        settled_debt[:]+=val-r['allowance']
        events.append(dict(label_day=str(dates[k]),available=str(dates[k]+np.timedelta64(1,'D'))+' 00:15',observed_at=date,
            final=at_close,j=j,used_R=val[1]*factor,used_T=val[2]*factor,released_R=returned[0]*factor,released_T=returned[1]*factor))
        js=[j];ys=[val]
        if not at_close and fb=='F1' and j!=int(z['ref'][k,ti]):
            other=int(z['ref'][k,ti]);y2=outputs(protected(site,z['fold'][k],str(dates[k]),other,r['offer'],branch))
            js.append(other);ys.append(y2)
            more.append(dict(day=str(dates[k]),received=date,j=other,gap=y2[0]*factor,reserve=y2[1]*factor,throughput=y2[2]*factor))
        if not at_close:history.append(dict(actions=js,errors=np.array(ys)-r['base'][js],context=ctx[k]))
    for slot,i in enumerate(ids):
        day=str(dates[i]);fold=str(z['fold'][i])
        for k in list(live):
            if dates[k]<=dates[i]-np.timedelta64(2,'D'):release(k,day+' 00:00',False)
        open_dates='|'.join(str(dates[k]) for k in live)
        prev=book.held();free=book.available();settled=book.settled.copy()
        offer=allocation(site,src,phase,i,t,pace,b0,free,len(ids)-slot,timing[slot]) if branch=='H' else b0.copy()
        base=center[i].copy();unc=width[i].copy();cap=ordinary_bound(z['plan'][i])
        # Transfer comparator recommends plans with the ordinary-R3 model, but only
        # its own metered R3H outcomes ever enter its subsequent calibration history.
        if branch=='H' and force!='meteredTransfer':base,unc,cap=quoted(site,src,phase,i,base,unc,offer)
        if fix is not None:base[:,fix['index']]+=fix['shift']
        model=np.clip(base+correction(history,ctx[i]),0,cap)
        pending=sum((r['expected'] for r in live.values()),np.zeros(3))
        booked_model=model.copy()
        booked_model[:,1]=np.clip(model[:,1]+rev['size']*unc[:,1],0,cap[:,1])
        planning_resource=b0 if force=='meteredTransfer' else offer
        j,reason,num,over,lam=choose(z['plan'][i],booked_model,unc,t,int(z['local'][i,ti]),planning_resource,settled_debt,pending,'dual',eta,q,len(history),{})
        if force in ['D0plan','localD0']:
            j=int(z['ref' if force=='D0plan' else 'local'][i,ti]);reason=force
        elif force=='fixedO':j=int(frozen[day]);reason=force
        # Keep the issued point estimate separate from the P15 reserve safety adjustment.
        reservation=np.array([min(offer[0],B[j]),offer[1]]) if branch=='H' else np.zeros(2)
        if branch=='H':book.book(i,reservation)
        plan=z['plan'][i,j];allow=np.r_[(1-t)*plan,b0]
        expected=booked_model[j]-allow
        rec=dict(site=site,run=label,branch=branch,phase=phase,day=day,fold=fold,i=i,target=t,src=src,head=head,seed=seed,feedback=fb,pace=pace,round=rnd,force=force or '',
                 j=j,orig=int(z['ref'][i,ti]),c=C[j],B=B[j]*factor,plan=plan*factor,online_labels=len(history),open_dates=open_dates,eligible=num,reason=reason,
                 pred_gap=model[j,0]*factor,pred_R=model[j,1]*factor,pred_T=model[j,2]*factor,risk_R=booked_model[j,1]*factor,
                 **{key+'_'+k:v[l]*factor for key,v in [('total',total),('B0',b0),('settled',settled),('prior_held',prev),('free',free),('offer',offer),('booked',reservation),('free_after',book.available())] for l,k in enumerate(['R','T'])},
                 **{key+'_'+k:v[l]*mult for key,v,mult in [('debt',settled_debt,factor),('pending',pending,factor),('lambda',lam,1.),('violation',over,factor)] for l,k in enumerate(['gap','R','T'])})
        panel.append(model.astype('f4'));risk.append(booked_model.astype('f4'));widths.append(unc.astype('f4'))
        # Only after the issue has been determined, run the 96 physical steps for this actual plan.
        a=protected(site,fold,day,j,reservation,branch);actual=outputs(a)
        assert abs(a[:,1].sum()*.25-plan)<1.e-5
        assert abs(a[0,7]-3.33)<1.e-6 and abs(a[-1,8]-3.33)<1.e-6
        assert np.max(abs(a[:,0]+a[:,3]+a[:,4]+a[:,5]-a[:,1]-a[:,2]-a[:,6]))<1.e-6
        assert np.max(abs(a[:,7]+.25*(.95*a[:,2]-a[:,3]/.95)-a[:,8]))<1.e-6
        if branch=='H':
            assert (actual[1:]<=reservation+1.e-6).all()
            assert np.max(abs(a[:,14]-.25*(a[:,2]+a[:,3])-a[:,15]))<1.e-6
        rec.update(shortfall=actual[0]*factor,delivered=(plan-actual[0])*factor,reserve=actual[1]*factor,throughput=actual[2]*factor,spill=a[:,6].sum()*.25*factor,terminal=a[-1,8]*factor)
        rows.append(rec)
        steps.append(a.astype('f8')*factor)
        live[i]=dict(actual=actual,j=j,base=base,allowance=allow,expected=expected,offer=offer)
    for k in list(live):
        release(k,str(dates[ids[-1]]+np.timedelta64(1,'D'))+' 00:15',True)
    if branch=='H':
        assert not book.pending and np.allclose(book.settled*factor,np.array([[r['reserve'],r['throughput']] for r in rows]).sum(0))
    df=pd.DataFrame(rows);df.to_parquet(path/f'{t}.parquet',index=False)
    np.savez_compressed(path/f'{t}_panels.npz',days=z['days'][ids],mean=panel,decision_forecast=risk,scale=widths)
    actdir=root/'actions'/phase/branch/site/label;actdir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(actdir/f'{t}.npz',a=steps,columns=np.array(H_COLS if branch=='H' else COLS),days=z['days'][ids],j=df.j)
    logdir=root/'settlements'/phase/branch/site/label;logdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(events).to_parquet(logdir/f'{t}.parquet',index=False)
    if more:pd.DataFrame(more).to_parquet(logdir/f'{t}_simulated_D0.parquet',index=False)
    write_json(path/f'{t}_seal.json',dict(issued=digest(path/f'{t}.parquet'),parameters=digest(CFG/'protocol.json'),actions=digest(actdir/f'{t}.npz')))
    return df

