"""Conditional weighting and coherent residual transport of complete PV donor trajectories."""
import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from torch import nn
from sklearn.preprocessing import StandardScaler

DEVICE = 'cuda'
QUANTILES = np.array([.03,.10,.25,.5,.75,.90,.97])
def tensor(x):
    return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)

def pool_indices(days, maximum=90):
    clock=np.array(days,dtype='datetime64[D]')
    valid=clock[None,:]+np.timedelta64(2,'D')<=clock[:,None]
    donors=np.zeros((len(days),maximum),int)
    mask=np.zeros_like(donors,dtype=bool)
    for i,permitted in enumerate(valid):
        selected=np.flatnonzero(permitted)[-maximum:]
        donors[i,:len(selected)]=selected
        mask[i,:len(selected)]=True
    return donors,mask

def ensemble_crps(paths,truth):
    count=paths.shape[1]
    weight=(2*torch.arange(count,device=paths.device)+1-count)/count**2
    shape=(1,count)+(1,)*(paths.ndim-2)
    return (paths-truth[:,None]).abs().mean(1)-(paths.sort(1).values*weight.reshape(shape)).sum(1)

class Retrieval(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.log_precision=nn.Parameter(torch.full((n_features,),-1.7))
        self.context=nn.Sequential(nn.Linear(n_features,12),nn.Tanh(),nn.Linear(12,n_features))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)
    def forward(self,x,donor_x,age,mask):
        precision=nn.functional.softplus(self.log_precision+.35*self.context(x)).clamp(max=4.)
        distance=((donor_x-x[:,None]).abs().clamp(max=4)*precision[:,None]).sum(2)/np.sqrt(x.shape[1])
        return (-distance-age/48).masked_fill(~mask,-1e6).softmax(1)

def weighted_crps(v,w,y):
    order=v.argsort(1)
    sorted_values=v.gather(1,order)
    weights=w[:,:,None].expand_as(v).gather(1,order)
    dispersion=(sorted_values*weights*(2*weights.cumsum(1)-weights-1)).sum(1)
    return ((v-y[:,None]).abs()*w[:,:,None]).sum(1)-dispersion

class Residual(nn.Module):
    def __init__(self,n_features,rank):
        super().__init__()
        self.rank=rank
        self.network=nn.Sequential(nn.Linear(n_features,24),nn.SiLU(),nn.Linear(24,2+rank))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
    def forward(self,x,anchor,basis,scale):
        params=self.network(x)
        total=anchor.sum((2,3))
        median=total.median(1,keepdim=True).values
        offset=.48*scale*torch.tanh(params[:,:1])
        spread=torch.exp(.55*torch.tanh(params[:,1:2]))
        target=(median+offset+spread*(total-median)).clamp_min(.0001)
        if self.rank:
            logshape=(.55*torch.tanh(params[:,2:]))@basis
            anchor=anchor*torch.exp(logshape)[:,None,:,None]
        return anchor*(target/anchor.sum((2,3)).clamp_min(.0001))[:,:,None,None]

def optimize(net,loss_fn,ti,vi,stem,seed,steps=75):
    rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(net.parameters(),lr=.004,weight_decay=.01)
    best=float('inf');epoch_best=0;log=[]
    for ep in range(steps):
        net.train()
        batches=np.array_split(rng.permutation(ti),max(1,len(ti)//40))
        errors=[]
        for indices in batches:
            loss=loss_fn(indices)
            opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
            errors.append(float(loss.detach()))
        with torch.no_grad():
            net.eval(); val=float(loss_fn(vi))
        log.append(dict(epoch=ep+1,train=float(np.mean(errors)),valid=val))
        if val<best-1.e-5:
            best=val;epoch_best=ep
            torch.save(dict(state=net.state_dict(),epoch=ep+1,seed=seed),stem)
        if ep-epoch_best>=14:break
    net.load_state_dict(torch.load(stem,map_location=DEVICE,weights_only=True)['state'])
    return log

def fit_suite(x,y,days,base,ti,vi,output,stem,seed=42,second=False,dt=.25,sunmask=None):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    ids,mask=pool_indices(days)
    ti=ti[mask[ti].sum(1)>=12]
    assert mask[vi].any(1).all() and mask[output].any(1).all()
    scaler=StandardScaler().fit(x[ti])
    features=scaler.transform(x).clip(-5,5).astype('f4')
    enc=tensor(features)
    yy=tensor(y)
    cap=max(.05,float(np.median(y[ti].sum((1,2)))))
    blocks=yy.reshape(len(y),4,y.shape[1]//4,y.shape[2]).sum((2,3))
    endpoints=torch.cat((blocks.sum(1,keepdim=True),blocks),1)/cap
    donor_values=endpoints[ids]
    donor_x=enc[ids]
    ages=(np.array(days,dtype='datetime64[D]')[:,None]-np.array(days,dtype='datetime64[D]')[ids]).astype(float)
    ages=tensor(ages)
    mm=torch.as_tensor(mask,device=DEVICE)
    model=Retrieval(x.shape[1]).to(DEVICE)
    def objective(indices):
        weights=model(enc[indices],donor_x[indices],ages[indices],mm[indices])
        loss=weighted_crps(donor_values[indices],weights,endpoints[indices])
        return loss[:,0].mean()+.8*loss[:,1:].mean()
    logs={'B1':optimize(model,objective,ti,vi,str(stem)+'_B1.pt',seed)}
    with torch.no_grad():weights=model(enc,donor_x,ages,mm).cpu().numpy()
    # Common stratified sampling of full paths. Rank by donor energy to reduce Monte Carlo noise.
    energy=y.sum((1,2));order=np.argsort(energy[ids],axis=1,kind='stable')
    sorted_w=np.take_along_axis(weights,order,1);cdf=sorted_w.cumsum(1);cdf[:,-1]=1.
    u=(np.arange(32)+.5)/32
    pick=np.stack([ids[i,order[i,np.searchsorted(cdf[i],u).clip(0,ids.shape[1]-1)]] for i in range(len(y))])
    B1=y[pick].astype('f4')
    if sunmask is not None:
        B1*=sunmask[:,None,:,:]
        base=base*sunmask[:,None,:,:]
    result={'B1':B1[output]}
    cache={}
    for name,anchor,rank in [('B2',base,0),('B3',B1,3)]:
        if second and name=='B3':
            # Fixed half-mixture resamples complete members from each source, without averaging coordinates.
            left=np.argsort(B1.sum((2,3)),axis=1)[:,::2]
            right=np.argsort(base.sum((2,3)),axis=1)[:,1::2]
            days_index=np.arange(len(y))[:,None]
            anchor=np.concatenate([B1[days_index,left],base[days_index,right]],axis=1)
        totals=anchor.sum((2,3));summ=np.quantile(totals,[.1,.5,.9],axis=1).T/cap
        condition=np.c_[features,summ].astype('f4')
        inputs=tensor(condition)
        ar=tensor(anchor)
        h=y.shape[1];t=np.linspace(-1,1,h)
        basis=tensor(np.stack([t,np.cos(np.pi*t),np.sin(2*np.pi*t)]))
        residual=Residual(condition.shape[1],rank).to(DEVICE)
        def loss(indices):
            path=residual(inputs[indices],ar[indices],basis,cap)
            b=path.reshape(len(indices),32,4,h//4,y.shape[2]).sum((3,4))/cap
            totals=b.sum(2);truth=endpoints[indices,0]
            ls=ensemble_crps(totals,truth).mean()+.8*ensemble_crps(b,endpoints[indices,1:]).mean()
            if second:
                # Low-tail calibration plus shrinkage toward the identity on limited evidence.
                q=torch.quantile(totals,.1,dim=1);err=truth-q
                ls=ls+.9*torch.maximum(.1*err,-.9*err).mean()+.16*(totals-ar[indices].sum((2,3))/cap).abs().mean()
            return ls
        logs[name]=optimize(residual,loss,ti,vi,str(stem)+'_'+name+'.pt',seed)
        with torch.no_grad():result[name]=residual(inputs[output],ar[output],basis,cap).cpu().numpy()
        cache[name]=summ
    anchor=base.sum((2,3))*dt;med=np.median(anchor,axis=1);norm=cap*dt
    gx=np.c_[features,np.quantile(anchor,[.1,.5,.9],axis=1).T/norm]
    residual_truth=(energy*dt-med)/norm
    trees=[LGBMRegressor(objective='quantile',alpha=float(q),learning_rate=.035,n_estimators=70,num_leaves=5,min_child_samples=14,n_jobs=1,verbosity=-1,random_state=seed).fit(gx[ti],residual_truth[ti]) for q in QUANTILES]
    qvals=np.sort(np.stack([tree.booster_.predict(gx[output]) for tree in trees],axis=1),1)
    target=np.maximum(.00001,med[output,None]+norm*np.stack([np.interp(u,QUANTILES,row) for row in qvals]))
    rank=np.argsort(np.argsort(anchor[output],axis=1,kind='stable'),axis=1)
    new=np.take_along_axis(target,rank,axis=1)
    result['B2tree']=(base[output]*(new/np.maximum(.00001,anchor[output]))[:,:,None,None]).astype('f4')
    joblib.dump(dict(scaler=scaler,trees=trees,cap=cap,dt=dt,days=np.array(days)[ti],val_days=np.array(days)[vi]),str(stem)+'_fit.joblib')
    np.savez_compressed(str(stem)+'_retrieval.npz',days=np.array(days)[output],candidate=ids[output],weights=weights[output],selected=pick[output],valid=mask[output])
    return result,logs

