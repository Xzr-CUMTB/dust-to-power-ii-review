"""Finite-action choice and date-weighted feedback under separate signed constraints."""
import numpy as np
from src.decision.p14.ledger import select as previous
SCALE=np.array([20.,10.,25.])
B=np.repeat(np.arange(10.,26.,2.5),13)
C=np.tile(np.round(np.arange(.30,.901,.05),4),7)

def correction(history,context):
    if not history:return np.zeros((91,3))
    records=history[-14:]
    # Each weather day has unit mass. An identical issued/reference action appears once.
    values=[];support=[]
    for k,h in enumerate(records):
        j=h['actions']
        kernels=np.exp(-np.abs(C[:,None]-C[j])/.24-np.abs(B[:,None]-B[j])/10.)
        age=2**(-(len(records)-1-k)/7)
        weather=np.exp(-np.minimum(abs(context-h['context']),3).mean())
        weights=kernels*(age*weather/len(j))
        values.append(weights@h['errors'])
        support.append(weights.sum(1))
    return np.sum(values,axis=0)/(3.+np.sum(support,axis=0))[:,None]

def choose(plan,pred,scale,t,local,budget,debt,pending,rule,eta,q,count,revision):
    # The second round selectively adjusts the provisional component (or selected reserve bias).
    corrected=pred.copy()
    if revision.get('target')=='reserve_bias':
        corrected[:,1]=np.clip(corrected[:,1]+revision['size']*scale[:,1],0,B)
    if rule!='dual':
        mode='fixed' if rule=='margin' else rule
        j,allow,reason,ok=previous(plan,corrected,t,local,budget,debt,pending,mode,2,None)
        excess=corrected-np.column_stack([(1-t)*plan,np.tile(budget,(len(plan),1))])
        if mode=='fixed':excess+=np.array([2.,.45,1.5])
        return j,reason,int(ok.sum()),excess[j]-allow,np.zeros(3)
    predicted=corrected+q*scale
    pweight=revision.get('size',1.) if revision.get('target')=='pending' else 1.
    signed=debt+pweight*pending
    excess=predicted-np.column_stack([(1-t)*plan,np.tile(budget,(len(plan),1))])
    lam=np.clip(np.array([1.4,.55,.45])+eta*signed/SCALE/np.sqrt(max(1,count)),0.,8.)
    room=np.array([15.,6.,16.])+np.clip(-signed/5.,-np.array([15.,6.,16.]),np.array([30.,8.,18.]))
    over=excess-room;feasible=(over<=1e-9).all(1)
    utility=(plan-predicted[:,0])/50.-(excess/SCALE)@lam
    if feasible.any():
        j=int(np.argmax(np.where(feasible,utility,-np.inf)));reason='maximum_lagrangian'
    else:
        worst=(over/SCALE).max(1);j=int(np.lexsort((-utility,worst))[0]);reason='minimum_excess'
        if worst[local]<=worst[j]+.10:j=int(local);reason='local_fallback'
    return j,reason,int(feasible.sum()),over[j],lam

