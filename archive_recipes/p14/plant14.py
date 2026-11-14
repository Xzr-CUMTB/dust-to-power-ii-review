from io14 import *
from engine import pack,hzd,fixed,dayrows,execute,coeff,value_features,scenario_gaps
from site13 import source,prepare
from src.decision.p14.hard import bounded
import torch
# The station adapter imports an older experiment which changes Torch's runtime thread count.
# P13 generation sets the count back to one before either dispatch adapter is used.
torch.set_num_threads(1)
@lru_cache(8)
def site_inputs(s):return source(s)
@lru_cache(18)
def prepared(s,d):
    frame,obs,cap=site_inputs(s);dt=pd.Timestamp(str(d));past=[k for k in sorted(obs) if k<=dt-pd.Timedelta('2d')]
    yy,sc,ref0,_,_,cx,_=prepare(dt,cap,obs,frame,past,int(s));rr=ref0/.7
    qs=np.quantile(sc,[.2,.5,.8],axis=1).transpose(1,0,2).copy()
    inn=np.r_[0.,0.,(yy-qs[:,1,0])[:-2]];xx=np.zeros((96,147));xx[:,96:112]=cx[:,:16]
    return yy,sc,rr,qs,inn,xx
@lru_cache(250)
def arguments(s,f,d,c):
    if s=='core':return pack(str(d),float(c)),hzd(f)[dayrows(str(d))],fixed(f)
    y,sc,rr,qs,inn,xx=prepared(s,d);pl=rr*c
    w=coeff(value_features(dict(x=xx,pm=np.zeros((96,4,6)),ref=rr),sc,np.arange(96),c,'no'),'soc',42)
    return (y,sc,pl,scenario_gaps(sc,qs,pl,inn),w,qs[:,2,:].copy()),np.full(96,.5),fixed('anchor')
def action(site,f,d,j,hard,loose=False):
    args,haz,lam=arguments(site,f,str(d),float(CGRID[j]))
    if hard:return bounded(*args,haz,lam,float(BGRID[j]),1.e4 if loose else 40.,1.e4 if loose else 15.)
    return execute(*args,haz,3,lam,33.3,16.65,9.99,float(BGRID[j]),3.33,True)
@lru_cache(24000)
def hard_label(site,f,d,j):
    a=action(site,f,d,j,True)
    return np.array([a[:,5].sum(),a[:,4].sum(),a[:,2:4].sum()])*.25
def smoke():
    import torch
    a=action('core','anchor','2019-05-10',key_index(25,.75),False);b=action('core','anchor','2019-05-10',key_index(25,.75),True,True)
    assert abs(a-b[:,:13]).max()<1e-6
    assert abs(a[:,5].sum()*.25-27.28272796514943)<1e-6
    with torch.autocast(device_type='cuda'):
        x=torch.ones((4,4),device='cuda',requires_grad=True);v=(x@x).sum()
    v.backward()
    dump(OUT14/'environment.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,threads=os.environ['OMP_NUM_THREADS'],loose_cap_parity=float(abs(a-b[:,:13]).max()),historical_gap=float(a[:,5].sum()*.25)))
if __name__=='__main__':smoke()

