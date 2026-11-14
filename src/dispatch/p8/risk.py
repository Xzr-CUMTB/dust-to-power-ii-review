"""Finite causal policies with exact equal-weight empirical CVaR. Power MW, stored energy MWh."""
import numpy as np
from numba import njit
from src.dispatch.p6.control import commands,project,value
@njit(cache=True)
def cvar(loss,alpha=.875):
 sample=np.sort(loss);eta=sample[min(len(loss)-1,int(np.ceil(alpha*len(loss)))-1)]
 # Closed form minimizer of eta + sum(u)/((1-alpha)S), u >= loss-eta, u>=0.
 return eta+np.maximum(loss-eta,0.).mean()/(1-alpha)
@njit(cache=True)
def policy_command(j,h,gap,s,b,E,P,R,due,recover,upper):
 if j<18:return commands(gap,s,b,(0.,.65,1.)[(j//2)%3],P,R,E)
 c,d,r=commands(gap,s,b,.35,P,R,E)
 if j==18 and h<2:
  r=min(r,.35*R);d=min(P,max(0.,(s-.1*E)*.95*4),max(0.,gap-r))
 if j==19 and h<4:
  floor=(.16 if recover else .32)*E;d=min(d,max(0.,(s-floor)*.95*4))
  r=min(R,4*b,max(0.,gap-d))
 if j==20 and h<4 and upper>due and d<1e-7:
  c=max(c,min(P,upper-due))
 return c,d,r
@njit(cache=True)
def options(k,s,b,paths,gaps,plans,w,upper,E,P,R,penalty,spend,target,slow,macro):
 S=paths.shape[1];H=min(24,96-k);n=18+3*macro;actions=np.zeros((n,3));metric=np.zeros((n,8))
 horizon_block=min(12-k%12,H)
 for j in range(n):
  typ=2*(j//6)+j%2 if j<18 else 3
  tail=np.zeros(S);excess=np.zeros(S);served=np.zeros(S);depletion=np.zeros(S);total=0.;stock=0.;exhaust_rate=0.
  for m in range(S):
   ss=s;bb=b;available=spend;first=H*.25
   for h in range(H):
    limit=min(bb,available) if slow and h<horizon_block else bb
    recovery=upper[k,min(h+3,H-1)]>plans[k,min(h+3,H-1)]
    c,d,r=policy_command(j,h,gaps[k,typ,m,h],ss,limit,E,P,R,plans[k,h],recovery,upper[k,h])
    if h==0:actions[j,0]=c;actions[j,1]=d;actions[j,2]=r
    ss,bb,c,d,r,sh,sp=project(ss,bb,c,d,r,paths[k,m,h],plans[k,h],k+h,E,P,R)
    available=max(0.,available-.25*r)
    tail[m]+=.25*sh;served[m]+=.25*(plans[k,h]-sh)
    if sh>1e-5 and ss<=.1*E+.01 and bb<.05:
     excess[m]+=.25*sh
     if first==H*.25:first=h*.25;exhaust_rate+=1./S
    total+=.25*(sh+.01*r+penalty*(c+d)+.00001*sp)
    if slow and h+1==horizon_block:stock+=max(0.,target-ss)
   depletion[m]=first
   if k+H<96:total-=value(max(0.,ss-.1*E),bb,w[k])
  metric[j,0]=total/S;metric[j,1]=cvar(tail);metric[j,2]=cvar(excess);metric[j,3]=stock/S
  metric[j,4]=np.sort(served)[int(.05*S)];metric[j,5]=depletion.mean();metric[j,6]=exhaust_rate;metric[j,7]=tail.mean()
 return actions,metric
@njit(cache=True)
def select(metric,kind,lam,stress,slow):
 obj=metric[:,0]+slow*.18*metric[:,3];base=np.argmin(obj);cv=metric[:,1];risk=metric[:,2]
 if len(obj)==18 and kind==0 and not slow:
  base=0
  for j in range(1,18):
   if obj[j]<obj[base]-1.e-9:base=j
  return base
 if kind==1:
  cap=cv[base]-min(1.,lam)*(cv[base]-cv.min())
  return np.argmin(np.where(cv<=cap+1.e-8,obj,1e20))
 if kind>=2:
  mult=lam*(.35+1.65*stress) if kind==3 else lam
  return np.argmin(obj+mult*(cv+risk))
 return base

