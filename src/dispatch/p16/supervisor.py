"""Period supply ledger and physical supervision of the original R3 action recommendation."""
import numpy as np
from numba import njit
from src.dispatch.p6.control import project
from src.dispatch.p8.risk import options,select,policy_command

class Reservations:
    def __init__(self,total):
        self.total=np.asarray(total,dtype=float)
        self.settled=np.zeros(2)
        self.pending={}

    def held(self):
        return sum(self.pending.values(),start=np.zeros(2))

    def available(self):
        return np.maximum(0.,self.total-self.settled-self.held())

    def book(self,key,amount):
        amount=np.asarray(amount,dtype=float)
        if key in self.pending or np.any(amount<0) or np.any(amount>self.available()+1.e-7):
            raise ValueError('Invalid resource reservation')
        self.pending[key]=amount.copy()

    def settle(self,key,meter):
        meter=np.asarray(meter,dtype=float)
        held=self.pending.pop(key)
        assert np.all(meter>=-1.e-7) and np.all(meter<=held+1.e-6)
        self.settled+=meter
        assert np.all(self.settled+self.held()<=self.total+1.e-6)
        return held-meter

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

