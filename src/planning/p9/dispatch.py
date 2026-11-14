"""Frozen finite-policy R3/SOC. Daily physical projection or a block terminal for actual carry sensitivity."""
import numpy as np
from numba import njit
from src.dispatch.p6.control import project,commands
from src.dispatch.p8.risk import options,select,policy_command
from src.dispatch.p7.policy import scenario_gaps
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
def planning_loss(scen,ref,c,B=25.,E=33.3,P=16.65,R=9.99,init=3.33):
 out=np.zeros(len(scen))
 for j in range(len(scen)):
  s=init;b=B
  for k in range(96):
   hat=max(0.,scen[j,k-2]+c*(ref[k]-ref[k-2])) if k>=2 else 0.
   ch,d,r=commands(c*ref[k]-hat,s,b,.35,P,R,E)
   s,b,ch,d,r,sh,sp=project(s,b,ch,d,r,scen[j,k],c*ref[k],k,E,P,R);out[j]+=.25*sh
 return out

