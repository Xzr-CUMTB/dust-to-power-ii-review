"""Resource-constrained causal scenario accounting and immutable day-ahead plan selection."""
import numpy as np
from numba import njit
from src.dispatch.p6.control import project,commands

@njit(cache=True)
def scenario_account(paths,reference,c,B):
    result=np.empty((len(paths),4))
    for j in range(len(paths)):
        s=3.33;b=B;short=0.;through=0.;run=0.;longest=0.
        for k in range(96):
            estimate=max(0.,paths[j,k-2]+c*(reference[k]-reference[k-2])) if k>=2 else 0.
            ch,d,r=commands(c*reference[k]-estimate,s,b,.35,16.65,9.99,33.3)
            s,b,ch,d,r,sh,_=project(s,b,ch,d,r,paths[j,k],c*reference[k],k,33.3,16.65,9.99)
            run=(run+.25) if sh>.001 else 0.;longest=max(run,longest)
            short+=.25*sh;through+=.25*(ch+d)
        result[j]=np.array([short,B-b,through,longest])
    return result

def rfeature(account,c,B,en):
    sh=account[:,0]/max(.01,en)
    return [sh.mean(),np.quantile(sh,.9),np.mean(sh>.05),np.mean(account[:,3])/12,c,B/25]

def pick_plan(loss,energy,target,margin,cycle_pred=None,limit=1e20):
    ok=(loss<=energy*(1-target-margin))
    if cycle_pred is not None:ok&=cycle_pred<=limit
    return int(np.flatnonzero(ok)[-1]) if ok.any() else 0


