"""Single-station forward adaptation, with source-normalized R3 and a local all-day donor forecast."""
from policy13 import *
from station_weather import siteframe,meta
from common11 import Q,cr
from src.models.p11.quantile import Tail
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from external_days import prepare
from engine import execute,fixed,coeff,value_features,scenario_gaps
from policy11 import accounts
import joblib
def source(s):
    frame=siteframe(s)[['power','lmd_totalirrad']].rename(columns={'lmd_totalirrad':'irr'})
    groups=frame.power.groupby((frame.index-pd.Timedelta('15min')).normalize())
    obs={d:v.to_numpy() for d,v in groups if len(v)==96 and np.isfinite(v).all() and v.min()>=0}
    cap=float(meta.loc['station'+s,'Capacity'])/1000
    return frame,obs,cap
def solar(d,lat,lon):
    day=pd.Timestamp(d).dayofyear;decl=.409*np.sin(2*np.pi*(day-81)/365)
    phase=2*np.pi*(day-81)/364;eq=9.87*np.sin(2*phase)-7.53*np.cos(phase)-1.5*np.sin(phase)
    hr=(np.arange(96)+.5)/4+(lon-120)/15+eq/60
    return np.sin(np.deg2rad(lat))*np.sin(decl)+np.cos(np.deg2rad(lat))*np.cos(decl)*np.cos(np.pi*(hr-12)/12)
def generation(s,dest):
    frame,obs,cap=source(s);factor=cap/66.6
    wx=pd.read_parquet(O/'stations'/('weather_east.parquet' if s=='01' else 'weather_local.parquet'));wx=wx[wx.site==s].pivot(index='day',columns=['h','field'],values='value')
    ds=pd.read_csv(O/'stations/calendar.csv',dtype={'site':str}).query('site==@s').day.tolist();ds=sorted(set(ds)&set(wx.index))
    W=[];history=[];banks=[];analog=[];YY=[];anchors=[];qual=[]
    for d in ds:
        stamp=pd.Timestamp(d);pool=sorted(k for k in obs if k<=stamp-pd.Timedelta('2d'))[-90:]
        if len(pool)<5:continue
        matched=[k for k in pool if str(k.date()) in wx.index]
        if len(matched)<5:continue
        q=wx.loc[d].unstack().sort_index();rad=q.rad.to_numpy()/700;cloud=q.cloud.to_numpy()/100
        m=meta.loc['station'+s];sun=solar(d,m.Latitude,m.Longitude)
        tm=np.array([sun[max(0,k):min(96,k+24)].mean() for k in [8,32,56,80]])
        w=np.c_[np.tile(rad[:,None],(1,4)),np.tile(cloud[:,None],(1,4)),np.arange(4)/3,np.array([5,11,17,23])/24,tm]
        E=lambda k:obs[k].sum()*.25/factor
        x=np.r_[np.mean([E(v) for v in pool[-7:]])/300,E(pool[-1])/300,np.std([E(v) for v in pool[-7:]])/150,np.maximum(0,sun).mean(),rad,cloud]
        hpool=np.array([E(k) for k in pool]);rank=np.argsort(abs(hpool-E(pool[-1]))+np.arange(len(pool))[::-1]*.5)
        hist=np.stack([obs[pool[j]]/factor for j in np.resize(rank[:32],32)])
        a=wx.loc[[str(k.date()) for k in matched]].to_numpy();distance=np.abs((a-wx.loc[d].values)/np.array([100 if field=='cloud' else 700 for h,field in wx.columns])).mean(1)+.001*np.arange(len(a))[::-1]
        use=[matched[j] for j in np.resize(np.argsort(distance)[:32],32)];bank=np.stack([obs[k]/factor for k in use]);bank[:,sun<-.05]=0
        if np.any(bank.sum(1)<=.000001): # use nearest earlier nonzero shape; equal rule for tree and neural
            good=bank.sum(1)>.000001
            if not good.any():raise ValueError('no valid daylight shape')
            bank[~good]=bank[good][0]
        order=bank.sum(1).argsort();bank=bank[order];shape_energy=bank.sum(1)*.25
        scaled=shape_energy*np.array([np.clip(np.maximum(sun,0).sum()/max(1,np.maximum(solar(str(use[j].date()),m.Latitude,m.Longitude),0).sum()),.6,1.65) for j in order])
        anchors.append(np.quantile(scaled,Q));W.append(w);history.append(x);banks.append(bank);analog.append(hist);qual.append(d);YY.append(obs[stamp]/factor)
    ds=np.array(qual);y=np.array(YY);bank=np.array(banks);anchors=np.array(anchors)
    prior=np.flatnonzero(ds<='2019-04-05');va=prior[-5:];tr=prior[:-6]
    cal=np.flatnonzero((ds>='2019-04-08')&(ds<='2019-05-05'));ev=np.flatnonzero(ds>='2019-05-08')
    assert len(tr)>=10 and len(cal)>=8 and len(ev)>=7
    dump(dest/'roles.json',dict(train=ds[tr].tolist(),validation=ds[va].tolist(),calibration=ds[cal].tolist(),evaluation=ds[ev].tolist(),capacity=cap))
    X=np.array(history);w=np.array(W);en=y.sum(1)*.25;scale=np.median(en[tr]);sc=StandardScaler().fit(X[tr]);xx=sc.transform(X).clip(-5,5)
    ft=np.c_[xx,anchors/scale,w.reshape(len(ds),-1)]
    # Single deterministic tree per quantile; equal date budget and 70 fitting iterations cap.
    trees=[LGBMRegressor(objective='quantile',alpha=float(q),n_estimators=70,num_leaves=4,min_child_samples=4,learning_rate=.04,n_jobs=1,verbosity=-1).fit(ft[tr],en[tr]) for q in Q]
    treeq=np.sort(np.column_stack([t.booster_.predict(ft) for t in trees]),axis=1).clip(.001)
    joblib.dump((trees,sc,scale),K/f'site{s}_trees.joblib')
    torch.manual_seed(42);torch.set_num_threads(1);net=Tail(xx.shape[1],len(Q)).cuda()
    ten=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device='cuda')
    xx_,ww,an,ys,qu=map(ten,[xx,w,anchors/scale,en/scale,Q])
    block=y.reshape(len(ds),8,12).sum(2)*.25/scale
    frac=bank.reshape(len(ds),32,8,12).sum(3)/bank.sum(2)[:,:,None]
    u=(np.arange(32)+.5)/32;hi=np.searchsorted(Q,u).clip(1,len(Q)-1);lo=hi-1
    weight=ten((u-Q[lo])/(Q[hi]-Q[lo]));bl=ten(block);fr=ten(frac)
    def loss(ix):
        q,t=net(xx_[ix],ww[ix],an[ix]);er=ys[ix,None]-q;pin=torch.maximum(qu*er,(qu-1)*er)
        e=q[:,lo]*(1-weight)+q[:,hi]*weight;shape=fr[ix]*t.exp()[:,None];shape=shape/shape.sum(2,keepdim=True)
        return 2*torch.trapz(pin,qu,dim=1).mean()+.55*pin[:,2:4].mean()+.65*((shape*e[:,:,None]).mean(1)-bl[ix]).abs().mean()+.008*t.square().mean()
    opt=torch.optim.AdamW(net.parameters(),lr=.004,weight_decay=.01);best=np.inf;log=[]
    for ep in range(70):
        opt.zero_grad();l=loss(tr);l.backward();opt.step()
        with torch.no_grad():v=float(loss(va))
        log.append(dict(epoch=ep,train=float(l.detach()),validation=v))
        if v<best:best=v;state={k:t.detach().cpu().clone() for k,t in net.state_dict().items()}
    net.load_state_dict(state);torch.save(state,K/f'site{s}_generation.pt')
    with torch.no_grad():q,t=net(xx_,ww,an);nq=q.cpu().numpy()*scale;tilt=t.cpu().numpy()
    paths={'D0':np.array(analog)}
    for name,quant,til in [('M3',nq,tilt),('M1',treeq,np.zeros_like(tilt))]:
        e=np.stack([np.interp(u,Q,row) for row in quant]);a=bank*np.exp(np.repeat(til,12,axis=1))[:,None,:]
        paths[name]=a*(e/(a.sum(2)*.25))[:,:,None]
    pd.DataFrame(log).to_csv(dest/'generation_epochs.csv',index=False);ids=np.r_[cal,ev]
    np.savez_compressed(dest/'generation.npz',days=ds[ids],n=len(cal),x=X[ids],y=y[ids],**{k:v[ids] for k,v in paths.items()})
def dispatch_inputs(s,d,inputs):
    frame,obs,cap=inputs;dt=pd.Timestamp(d);past=[k for k in sorted(obs) if k<=dt-pd.Timedelta('2d')]
    yy,sc,ref0,_,_,cx,_=prepare(dt,cap,obs,frame,past,int(s));rr=ref0/.7;qs=np.quantile(sc,[.2,.5,.8],axis=1).transpose(1,0,2).copy();inn=np.r_[0.,0.,(yy-qs[:,1,0])[:-2]]
    xx=np.zeros((96,147));xx[:,96:112]=cx[:,:16];args={}
    def action(j):
        b,c=divmod(j,13)
        if c not in args:
            pl=rr*GRID[c];w=coeff(value_features(dict(x=xx,pm=np.zeros((96,4,6)),ref=rr),sc,np.arange(96),GRID[c],'no'),'soc',42)
            args[c]=(yy,sc,pl,scenario_gaps(sc,qs,pl,inn),w,qs[:,2,:].copy())
        return execute(*args[c],np.full(96,.5),3,fixed('anchor'),33.3,16.65,9.99,float(BUDGETS[b]),3.33,True)
    return rr[:,0],action
def station(s):
    dest=O/'stations'/s;dest.mkdir(exist_ok=True)
    assert (C/'round2.json').exists();generation(s,dest)
    z=dict(np.load(dest/'generation.npz'));ds=z['days'];n=int(z['n']);inputs=source(s);fac=inputs[2]/66.6
    Y=[];ref=[];plan=[];proxies={m:[] for m in ['D0','M1','M3']}
    for i,d in enumerate(ds):
        rr,act=dispatch_inputs(s,str(d),inputs);pe=np.tile(GRID*rr.sum()*.25,7);plan.append(pe)
        vals=[]
        for j in range(91):
            st=stats(act(j));vals.append([st['shortfall'],st['reserve'],st['throughput']])
        Y.append(vals)
        for m in proxies:proxies[m].append(np.concatenate([accounts(z[m][i],rr,b)[:,:3] for b in BUDGETS]))
        ref.append([largest(proxies['D0'][-1][:,0],pe,t,np.arange(78,91)) for t in TARGETS])
    Y=np.array(Y);ref=np.array(ref);plan=np.array(plan);np.savez_compressed(dest/'grid.npz',days=ds,y=Y,plan=plan,ref=ref,**proxies)
    forecasts=[];locks=[];cands=[];logs=[]
    for m in proxies:
        raw=np.array(proxies[m]);ctx=[]
        for i,d in enumerate(ds):
            past=[j for j in range(i) if age(d,ds[j])>=2][-7:];E=z[m][i].sum(1)*.25;q=np.quantile(E,[.1,.5,.9])
            resid=np.mean([Y[j,ref[j,1]]-raw[j,ref[j,1]] for j in past],0)/SCALE if past else np.zeros(3)
            ctx.append(np.r_[z['x'][i],q/300,(q[2]-q[0])/200,resid])
            actual=z['y'][i].sum()*.25
            forecasts.append(dict(site=s,model=m,day=d,phase='evaluation' if i>=n else 'cal',CRPS=cr(E[None],np.array([actual]))[0]*fac,normalized_CRPS=cr(E[None],np.array([actual]))[0]/66.6,observed_MWh=actual*fac))
        actions=np.stack([np.c_[CGRID,BGRID/25,plan[i]/350,raw[i]/SCALE] for i in range(len(ds))])
        b=dict(days=ds,n=n,ctx=np.array(ctx),act=actions,y=Y,ref=ref,raw=raw,plan=plan)
        # Expanding OOF warmup, separate final local fit strictly predates external evaluation.
        for kind in ['absolute','absnet','multi','pair']:
            pred=np.zeros((len(ds),2,91,3))
            if kind in ['absnet','pair']:
                splits=[]
                for ix in list(range(10,n,7))+[n]:
                    prior=[j for j in range(n) if age(ds[ix],ds[j])>=2];nv=min(5,max(3,len(prior)//4))
                    tr=prior[:-(nv+1)];va=prior[-nv:];name=f'site{s}_{m}_{kind}_{ix}'
                    pred[ix:]=train_head(b,tr,va,name,42,2,kind=='absnet',.5 if kind=='pair' else 0)[ix:];splits.append(dict(issue=ds[ix],fit=ds[tr].tolist(),validation=ds[va].tolist()))
                dump(dest/f'{m}_{kind}_roles.json',splits)
            for t in TARGETS:
                for rule in ['one_sided','tight','near','nominal']:
                    rows,cs,_,_=run_daybank(b,pred,kind,rule,'point',t,True,s)
                    locks.extend([dict(**r,source=m,method=kind) for r in rows]);cands.extend([g.assign(source=m,method=kind) for g in cs])
    pd.DataFrame(forecasts).to_csv(dest/'forecast_scores.csv',index=False);pd.concat(cands).to_parquet(dest/'candidates.parquet',index=False)
    locked=dest/'locked.parquet';pd.DataFrame(locks).to_parquet(locked,index=False);digest=sha(locked)
    rows=[];frames=[]
    for d,g in pd.read_parquet(locked).groupby('day'):
        _,action=dispatch_inputs(s,d,inputs);ii=list(ds).index(d);lookup={}
        for j in np.unique(np.r_[g.idx,g.ref]):
            a=action(int(j))*fac;lookup[j]=stats(a);frames.append(pd.DataFrame(a,columns=COLS).assign(day=d,idx=j,step=np.arange(96)))
            assert abs(lookup[j]['shortfall']-Y[ii,j,0]*fac)<1e-5
        for r in g.itertuples():
            st=lookup[r.idx];rs=lookup[r.ref]
            rows.append(dict(site=s,day=d,target=r.target,regime=r.regime,source=r.source,method=r.method,active=r.active,**st,**{'delta_'+k:st[k]-rs[k] for k in ['plan','delivered','shortfall','reserve','throughput','spill']}))
    pd.concat(frames,ignore_index=True).to_parquet(dest/'actions.parquet',index=False);pd.DataFrame(rows).to_csv(dest/'executed.csv',index=False)
    dump(dest/'complete.json',dict(station=s,days=len(ds)-n,cal=n,lock_sha256=digest,code_sha256=sha(Path(__file__)),site_lock=sha(C/'site_lock.json'),round2_lock=sha(C/'round2.json')));print(s,len(rows),flush=True)
if __name__=='__main__':station(sys.argv[1])

