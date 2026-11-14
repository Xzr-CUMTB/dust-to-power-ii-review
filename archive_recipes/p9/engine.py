from intake import *
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/'scripts/p8'));sys.path.append(str(ROOT/'scripts/p7'))
from base import data,dayrows,coeff,value_features,report_day
from src.planning.p9.dispatch import execute,planning_loss,scenario_gaps
from src.dispatch.p5.control import plan as lp
from functools import lru_cache
import hashlib
GRID=np.round(np.arange(.30,.901,.05),4)
COLS=['pv','plan','charge','discharge','reserve','shortfall','spill','soc','soc_after','balance','balance_after','requested_charge','requested_discharge']
@lru_cache(1)
def gmem():return np.load(str(ROOT/'results/p6/forecasts/F1.npy'),mmap_mode='r')
@lru_cache(6)
def hzd(fold):return np.load(ROOT/f'results/p8/risk/hazard_{fold}_full.npy')[:,0]
@lru_cache(1)
def full():return dict(np.load(OUT/'day_interface.npz'))
@lru_cache(20)
def midnight(day):
 i=np.flatnonzero(full()['days']==day)[0];return full()['y'][i].astype(float).sum(1)
@lru_cache(4)
def fixed(fold):return json.loads((ROOT/f'results/p8/training/risk_lock_{fold}.json').read_text())['R3']['lam']
@lru_cache(8)
def rawday(day):
 z=data();ix=dayrows(day);v=gmem()[ix].astype('f8');q=np.quantile(v,[.2,.5,.8],axis=1).transpose(1,0,2).copy()
 y=z['pv'][ix].astype(float);assert np.max(abs(midnight(day)-y))<1.e-4
 inn=np.r_[0.,0.,(y-q[:,1,0])[:-2]];local={k:z[k][ix] for k in ['x','pm','ref']}
 return y,v,q,inn,local
def ref(day):return rawday(day)[4]['ref'][:,0]
@lru_cache(18)
def pack(day,c):
 y,v,q,inn,local=rawday(day);pl=local['ref']*c;gg=scenario_gaps(v,q,pl,inn)
 w=coeff(value_features(local,v,np.arange(96),c,'yes'),'soc',42)
 return y,v,pl,gg,w,q[:,2,:].copy()
def replay(fold,day,c,B,ctrl='R3',soc=3.33,end=True,E=33.3,P=16.65,R=9.99):
 p=pack(day,float(c));a=execute(*p,hzd(fold)[dayrows(day)],3 if ctrl=='R3' else 0,fixed(fold),E,P,R,float(B),float(soc),end)
 assert abs(a[:,2:4].min())<1e-5 and abs(a[-1,8]-.1*E)<1e-5 if end else a[-1,8]>=.1*E-1.e-6
 assert np.max(abs(a[:,0]+a[:,3]+a[:,4]+a[:,5]-a[:,1]-a[:,2]-a[:,6]))<1e-5
 return a
def lookup(fname):
 p=np.load(fname);return dict(zip(p['days'].tolist(),p['paths'].astype(float).sum(-1)))
def choice(loss,en,target,margin): 
 ok=loss<=en*(1-target-margin);return int(np.where(ok)[0][-1]) if ok.any() else 0
def blocks(days):
 groups=[];buf=[]
 for day in sorted(days):
  if buf and (pd.Timestamp(str(day))-pd.Timestamp(str(buf[-1]))).days!=1:buf=[]
  buf.append(str(day))
  if len(buf)==7:groups.append(buf);buf=[]
 return groups
def stats(a):return report_day(a)
def blockbook(score,base,remain,n,scale=1.,cap=50.):
 return min(remain,cap,max(0.,remain*max(1.,score*scale)/(max(1.,score*scale)+(n-1)*max(1.,base))))

