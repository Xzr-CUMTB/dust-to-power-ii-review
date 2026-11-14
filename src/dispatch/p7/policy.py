"""Finite-action closed-loop policy and paired continuations. All resources use P6 AC units."""
import numpy as np
from numba import njit
from src.dispatch.p6.control import project,commands,value
@njit(cache=True)
def scenario_gaps(paths,quantiles,plan,innovation):
 gaps=np.empty((96,6,paths.shape[1],24),np.float64)
 for k in range(96):
  for a in range(3):
   for g in range(2):
    for m in range(paths.shape[1]):
     for h in range(24):
      error=innovation[k]*.65**h if h<2 else paths[k,m,h-2]-quantiles[k,1,h-2]
      estimate=quantiles[k,1,h]+(a-1)*.7*max(.25,quantiles[k,2,h]-quantiles[k,0,h])+g*.75*error
      gaps[k,2*a+g,m,h]=plan[k,h]-max(0.,estimate)
 return gaps
@njit(cache=True)
def allocation(k,s,b,paths,gaps,plan,weight,penalty,expand):
 count=18+4*expand
 actions=np.empty((count,3));costs=np.zeros(count)
 for j in range(18):
  a=j//6;g=j%2;share=(0.,.65,1.)[(j//2)%3]
  c,d,r=commands(gaps[k,a*2+g,0,0],s,b,share,16.65,9.99,33.3)
  actions[j,0]=c;actions[j,1]=d;actions[j,2]=r
 identical=True
 for j in range(1,18):
  if abs(actions[j,0]-actions[0,0])+abs(actions[j,1]-actions[0,1])+abs(actions[j,2]-actions[0,2])>1e-8:identical=False;break
 # If all commands are zero, local interpolation cannot create a different action.
 if identical and np.max(actions[0])<1e-9:
  for j in range(18,count):actions[j]=actions[0]
  return actions,0,costs
 horizon=min(24,96-k);members=paths.shape[1];best=0;best18=0
 for j in range(count):
  template=j if j<18 else best18
  a=template//6;g=template%2;share=(0.,.65,1.)[(template//2)%3]
  if j>=18:
   actions[j]=actions[best18].copy()
   if j==18:actions[j,1]*=.5
   if j==19:actions[j,2]*=.5
   if j==20:actions[j,0]*=.5
   if j==21:actions[j,2]=min(9.99,b*4,actions[j,2]+min(2.,max(0.,plan[k,0]-actions[j,2])))
  score=0.
  for m in range(members):
   soc=s;left=b
   for h in range(horizon):
    if h==0:c,d,r=actions[j]
    else:c,d,r=commands(gaps[k,a*2+g,m,h],soc,left,share,16.65,9.99,33.3)
    soc,left,c,d,r,sh,sp=project(soc,left,c,d,r,paths[k,m,h],plan[k,h],k+h,33.3,16.65,9.99)
    score+=.25*(sh+.01*r+penalty*(c+d)+.00001*sp)
   if k+horizon<96:score-=value(max(0.,soc-3.33),left,weight[k])
  costs[j]=score/members
  if costs[j]<costs[best]-1e-9:best=j
  if j==17:best18=best
 return actions,best,costs
@njit(cache=True)
def action_basis(a):
 c,d,r=a/10.
 return np.array((c,d,r,c*c,d*d,r*r,c*r,d*r))
@njit(cache=True)
def state_vector(context,s,b,used,previous,k):
 return np.concatenate((context,np.array(((s-3.33)/26.64,b/25.,used/60.,previous/10.,(96-k)/96.))))
@njit(cache=True)
def choice(k,s,b,used,previous,paths,gaps,plan,w,context,net,learned,expand,penalty,margin):
 actions,base,scenario_cost=allocation(k,s,b,paths,gaps,plan,w,penalty,expand)
 if not learned:return actions[base],base,base
 mu,sd,w1,b1,w2,b2=net
 state=state_vector(context[k],s,b,used,previous,k)
 coef=(np.tanh(((state-mu)/sd)@w1+b1)@w2+b2).reshape(4,9)
 unit=np.array((1.,.01,.00001,penalty))
 best=base;gain=-margin;orig=action_basis(actions[base])
 for j in range(len(actions)):
  diff=(coef[:,:8]@(action_basis(actions[j])-orig))@unit
  if diff<gain-1.e-8:best=j;gain=diff
 return actions[best],best,base
@njit(cache=True)
def replay(truth,paths,gaps,plan,w,context,net,learned,expand,budget,penalty,margin):
 rows=np.empty((96,19));s=3.33;b=budget;used=0.;previous=0.
 for k in range(96):
  cmd,j,base=choice(k,s,b,used,previous,paths,gaps,plan,w,context,net,learned,expand,penalty,margin)
  sn,bn,c,d,r,sh,sp=project(s,b,cmd[0],cmd[1],cmd[2],truth[k],plan[k,0],k,33.3,16.65,9.99)
  rows[k]=np.array((truth[k],plan[k,0],c,d,r,sh,sp,s,sn,b,bn,used,previous,cmd[0],cmd[1],cmd[2],j,base,used+.25*(c+d)))
  s=sn;b=bn;used+=.25*(c+d);previous=r
 return rows
@njit(cache=True)
def continue_day(k,s,b,used,previous,first,truth,paths,gaps,plan,w,context,net,learned,expand):
 costs=np.zeros(4)
 for h in range(k,96):
  if h==k:cmd=first
  else:cmd,_,_=choice(h,s,b,used,previous,paths,gaps,plan,w,context,net,learned,expand,.0001,.0)
  s,b,c,d,r,sh,sp=project(s,b,cmd[0],cmd[1],cmd[2],truth[h],plan[h,0],h,33.3,16.65,9.99)
  previous=r;used+=.25*(c+d)
  costs+=.25*np.array((sh,r,sp,c+d))
 assert abs(s-3.33)<1e-5
 return costs

