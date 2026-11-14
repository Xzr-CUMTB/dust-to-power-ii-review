"""Finite actions with three separate signed energy balances. All arithmetic in MWh."""
import numpy as np
SCALE=np.array([20.,10.,25.])
HORIZON=5.
LIMIT=np.array([30.,8.,18.])
def select(plan,expected,rho,baseline,resource,settled,pending,mode,round_id,correction):
    debt=settled if round_id==1 else settled+pending
    allowance=np.zeros(3) if mode!='ledger' else np.clip(-debt/HORIZON,-LIMIT,LIMIT)
    risk=np.column_stack([expected[:,0]-(1-rho)*plan,expected[:,1:]-resource])
    if mode=='fixed':risk=risk+np.array([2.,.45,1.5])
    if round_id==2 and correction is not None:risk=risk+correction
    excess=risk-allowance
    feasible=(excess<=1e-8).all(1);delivery=plan-expected[:,0]
    if feasible.any():
        index=int(np.argmax(np.where(feasible,delivery,-np.inf)))
        why='highest_predicted_delivery_feasible'
    else:
        maximum=np.max(excess/SCALE,axis=1)
        index=int(np.lexsort((np.arange(len(plan)),-delivery,maximum))[0]);why='minimum_worst_normalized_excess'
        if maximum[baseline]<=maximum[index]+.10:
            index=baseline;why='local_reference_fallback'
    return index,allowance,why,feasible
def calibrate(raw,history,context):
    if len(history)<4:return raw.copy()
    h=history[-28:];delta=np.stack([r['model_error'] for r in h]);ages=np.array([r['ordinal'] for r in h]);dist=np.stack([r['context'] for r in h])-context
    kernel=np.exp(-np.abs(dist).clip(0,4).mean(1));kernels=np.exp(-np.log(2)*(ages.max()-ages)[None,:]/np.array([5,14,28])[:,None])*kernel
    kernels/=kernels.sum(1,keepdims=True);effective=1/(kernels*kernels).sum(1)
    offsets=kernels@delta*(effective/(8+effective))[:,None]
    loss=np.array([r['expert_error'] for r in h]).mean(0);mix=np.exp(-np.sqrt(len(h))*np.minimum(20,loss));mix/=mix.sum(0)
    return raw+(mix*offsets).sum(0)
def expert_offsets(history,context):
    if len(history)<4:return np.zeros((3,3))
    h=history[-28:];age=np.array([r['ordinal'] for r in h]);dist=np.stack([r['context'] for r in h])-context
    w=np.exp(-np.log(2)*(age.max()-age)[None,:]/np.array([5,14,28])[:,None]-abs(dist).clip(0,4).mean(1)[None,:]);w/=w.sum(1,keepdims=True)
    n=1/(w*w).sum(1)
    return (w@np.stack([r['model_error'] for r in h]))*(n/(n+8))[:,None]

