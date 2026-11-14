"""P18-CLOSE technical revision: only kind3 objective tie-breaking changes.

All physical transitions, candidate actions, forecasts, coefficients and horizons
are unchanged. The 1e-9 normalized objective-MWh tolerance matches the existing
kind0 numerical scale. No candidate gains priority from its result or member id.
The first candidate within 1e-9 of the true minimum has deterministic priority.
"""
import numpy as np
from numba import njit
from src.dispatch.p6.control import project,commands
from src.dispatch.p8.risk import options,policy_command,select as original_select
from src.dispatch.p7.policy import scenario_gaps
TIE_TOL=1.e-9
@njit(cache=True)
def stable_minimum(obj):
    low=np.min(obj)
    for j in range(len(obj)):
        if obj[j]<=low+TIE_TOL:return j
    return np.argmin(obj)
@njit(cache=True)
def select(metric,kind,lam,stress,slow):
    if kind!=3:return original_select(metric,kind,lam,stress,slow)
    obj=metric[:,0]+slow*.18*metric[:,3]
    mult=lam*(.35+1.65*stress)
    return stable_minimum(obj+mult*(metric[:,1]+metric[:,2]))

@njit(cache=True)
def execute(y,v,pl,gg,w,upper,hazard,kind,lam,E,P,R,B,init,terminal):
 s=init;b=B;a=np.zeros((96,13))
 for k in range(96):
  # When every candidate gives the very same first command the choice has no physical effect.
  act=np.empty((21 if kind else 18,3))
  for j in range(len(act)):
   ty=2*(j//6)+j%2 if j<18 else 3
   act[j]=policy_command(j,0,gg[k,ty,0,0],s,b,E,P,R,pl[k,0],upper[k,min(3,95-k)]>pl[k,min(3,95-k)],upper[k,0])
  chosen=0
  if np.max(np.abs(act-act[0]))>1e-12:
   acts,m=options(k,s,b,v,gg,pl,w,upper,E,P,R,.0001,b,.1*E,False,kind!=0)
   stress=min(1.,max(0.,.35*(1-b/max(B,.01))+.25*(1-(s-.1*E)/(.8*E))+.4*hazard[k]))
   chosen=select(m,kind,lam,stress,False)
  c,d,r=act[chosen]
  sn,bn,ce,de,re,sh,sp=project(s,b,c,d,r,y[k],pl[k,0],k if terminal else k-96,E,P,R)
  a[k]=np.array([y[k],pl[k,0],ce,de,re,sh,sp,s,sn,b,bn,c,d]);s,b=sn,bn
 return a

@njit(cache=True)
def r3h(y,v,pl,gg,w,upper,hazard,lam,nominal,reserve_cap,throughput_cap):
    e,p,rated=33.3,16.65,9.99
    floor=.1*e
    soc=floor
    reserve=min(nominal,reserve_cap)
    nominal_left=nominal
    left=throughput_cap
    a=np.empty((96,19))
    for k in range(96):
        options_now=np.empty((21,3))
        for j in range(21):
            kind=2*(j//6)+j%2 if j<18 else 3
            options_now[j]=policy_command(j,0,gg[k,kind,0,0],soc,nominal_left,e,p,rated,pl[k,0],upper[k,min(3,95-k)]>pl[k,min(3,95-k)],upper[k,0])
        chosen=0
        if np.max(np.abs(options_now-options_now[0]))>1.e-12:
            _,m=options(k,soc,nominal_left,v,gg,pl,w,upper,e,p,rated,.0001,nominal_left,.1*e,False,True)
            stress=min(1.,max(0.,.35*(1-nominal_left/max(nominal,.01))+.25*(1-(soc-floor)/(.8*e))+.4*hazard[k]))
            chosen=select(m,3,lam,stress,False)
        c,d,r=options_now[chosen]
        # P14 reachability formula reserves AC discharge for every MWh added to inventory.
        charge_limit=4*max(0.,left-.95*(soc-floor))/(1+.95*.95)
        sn,rn,ch,di,re,gap,spill=project(soc,reserve,min(c,charge_limit),min(d,4*max(0.,left)),r,y[k],pl[k,0],k,e,p,rated)
        after=left-.25*(ch+di)
        assert after>=.95*(sn-floor)-1.e-6 and rn>=-1.e-7
        a[k]=np.array([y[k],pl[k,0],ch,di,re,gap,spill,soc,sn,reserve,rn,c,d,r,left,after,charge_limit,nominal_left,nominal_left-.25*re])
        soc=sn;left=after;reserve=rn;nominal_left=max(0.,nominal_left-.25*re)
    assert abs(soc-floor)<1.e-6
    return a

