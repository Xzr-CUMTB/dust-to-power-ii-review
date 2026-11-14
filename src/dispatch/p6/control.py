"""Finite-policy stochastic MPC. MW for powers, MWh for states, dt=0.25 h.
No truth argument to solve(). Actual power enters only the execution projection or matured innovation.
"""
import numpy as np
from numba import njit
@njit(cache=True)
def project(s,b,c,d,r,pv,plan,k,E,P,R):
 floor=.1*E
 forced=max(0.,(s-floor-(95-k)*P*.25/.95)*.95*4)
 d=min(max(d,forced),P,max(0.,(s-floor)*.95*4))
 r=min(max(0.,r),R,max(0.,b*4))
 c=min(max(0.,c) if d<=1e-7 else 0.,P,max(0.,(.9*E-s)*4/.95),max(0.,pv+d+r-plan))
 # Last-interval recharge would violate the hard terminal; also enforce reachable upper inventory.
 c=min(c,max(0.,(floor+(95-k)*P*.25/.95-s+d*.25/.95)*4/.95))
 sn=s+.25*(.95*c-d/.95);bn=max(0.,b-.25*r)
 net=pv+d+r-c-plan
 return sn,bn,c,d,r,max(0.,-net),max(0.,net)
@njit(cache=True)
def commands(gap,s,b,share,P,R,E):
 # Forecast-only charge/discharge and reserve commands; actual balancing not available here.
 usable=max(0.,(s-.1*E)*.95*4)
 dr=min(P,usable,max(0.,gap)*(1-share))
 rr=min(R,b*4,max(0.,gap-dr))
 dr=min(P,usable,max(0.,gap-rr))
 return min(P,max(0.,-gap)),dr,rr
@njit(cache=True)
def basis(e,r):
 # MWh saved relative to zero inventory. Concave increasing piecewise linear.
 u=np.empty(9)
 for j,knot in enumerate((4.,12.,30.)):
  u[j]=min(e*.95,knot);u[3+j]=min(r,knot);u[6+j]=min(e*.95+r,knot)
 return u
@njit(cache=True)
def value(e,r,w):
 v=0.
 for j,knot in enumerate((4.,12.,30.)):
  v+=w[j]*min(e*.95,knot)+w[3+j]*min(r,knot)+w[6+j]*min(e*.95+r,knot)
 return v
@njit
def solve(v,q,plan,k,s,b,E,P,R,w,innovation,price,adapt,feedback,release=0.,tail=0.):
 n=min(24,96-k);S=len(v);best=1e25;chosen=np.zeros(6)
 # Same finite policy class for C2/C3. Gains and offsets selected without actual future.
 for offset in (-.7,0.,.7):
  for share in (0.,.65,1.):
   for gain in (0.,.75):
    if not feedback and gain>0:continue
    cost=0.;c0=d0=r0=0.
    for j in range(S):
     ss=s;bb=b
     for h in range(n):
      err=innovation*(.65**h)
      if h>=2:err=v[j,h-2]-q[1,h-2]
      # Round two: excess remaining reserve vs known future contracted energy
      # feeds an advance release offset. Current truth is absent.
      free=bb/(5.+bb+.15*tail)
      expected=q[1,h]+(offset+adapt-release*free)*max(.25,q[2,h]-q[0,h])+gain*err
      c,d,r=commands(plan[h]-max(0.,expected),ss,bb,share,P,R,E)
      if h==0:c0,d0,r0=c,d,r
      ss,bb,c,d,r,short,spill=project(ss,bb,c,d,r,v[j,h],plan[h],k+h,E,P,R)
      cost+=.25*(short+price*r+.0001*(c+d)+.00001*spill)
     if k+n<96:cost-=value(max(0.,ss-.1*E),bb,w)
    cost/=S
    if cost<best-1e-9:
     best=cost;chosen[:]=np.array((c0,d0,r0,offset,share,gain))
 return chosen,best
@njit
def simulate(v,qs,truth,plans,coefs,innov,E,P,R,budget,price,adapt,feedback,release=0.):
 a=np.zeros((96,20));s=.1*E;b=budget
 for k in range(96):
  tail=.25*plans[min(k+24,96):,0].sum()
  cmd,cost=solve(v[k],qs[k],plans[k],k,s,b,E,P,R,coefs[k],innov[k],price,adapt,feedback,release,tail)
  sn,bn,c,d,r,short,spill=project(s,b,cmd[0],cmd[1],cmd[2],truth[k],plans[k,0],k,E,P,R)
  a[k]=np.array((truth[k],plans[k,0],c,d,r,short,spill,s,sn,bn,cmd[0],cmd[1],cmd[2],cmd[3],cmd[4],cmd[5],cost,innov[k],b,value(s-.1*E,b,coefs[k])))
  s,b=sn,bn
 assert abs(s-.1*E)<1e-5 and b>=-1e-6
 return a
@njit(cache=True)
def suffix(truth,plan,start,e,b,E,P,R):
 # Deployable simple base policy for ADP targets. Truth[h-2] available, current truth only for plant execution.
 s=.1*E+e;cost=0.
 for h in range(start,96):
  observed=truth[max(0,h-2)] # start >= 24 by construction
  gap=plan[h]-max(0.,observed+plan[h]-plan[h-2])
  c,d,r=commands(gap,s,b,.35,P,R,E)
  s,b,c,d,r,sh,sp=project(s,b,c,d,r,truth[h],plan[h],h,E,P,R)
  cost+=.25*(sh+.01*r+.0001*(c+d)+.00001*sp)
 return cost

