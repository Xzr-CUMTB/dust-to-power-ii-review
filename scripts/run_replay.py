"""Reissue selected plans with mature feedback, then verify one archived day of physical execution."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_inference import consequences
from src.dispatch.p15.decision import B,choose,correction
from src.dispatch.p16.supervisor import r3h as original_hard
from src.planning.p9.dispatch import execute as original_ordinary
from scripts.p18_close.revised import execute as revised_ordinary,r3h as revised_hard

def issue(source):
    config=json.loads((ROOT/'configs/operating_point.json').read_text())
    z=np.load(ROOT/f'examples/candidates_{source}.npz')
    days=z['days'].astype('datetime64[D]'); b0=np.array(config['B0'])
    ctx=z['ctx'][:,:12]
    center,width=consequences(source)
    debt=np.zeros(3);pending={};history=[];chosen=[]
    for i,day in enumerate(days):
        for old in list(pending):
            # An observation for d settles at d+1 00:15 local time.
            # The next midnight issue able to read it is d+2.
            if days[old]>day-np.timedelta64(2,'D'):
                continue
            action,expected,allowance=pending.pop(old)
            realized=z['outcomes'][old]
            debt+=realized-allowance
            history.append(dict(actions=[action],errors=(realized-center[old,action])[None],context=ctx[old]))
        assert len(history)==int(z['n_received'][i])
        plan=z['plan'][i]
        cap=np.column_stack((plan,B,np.full(91,16.65*24)))
        booked=np.clip(center[i]+correction(history,ctx[i]),0,cap)
        booked[:,1]=np.clip(booked[:,1]+config['reserve_bias']*width[i,:,1],0,B)
        expected_outstanding=sum((record[1] for record in pending.values()),np.zeros(3))
        decision,*_=choose(plan,booked,width[i],config['target'],int(z['local'][i,1]),b0,debt,expected_outstanding,'dual',config['eta'],config['q'],len(history),{})
        # Metered consequences in the fixture belong to this selected action.
        # Reject a different path before it could consume a mismatched settlement.
        assert int(decision)==int(z['expected_index'][i]),(source,str(day),decision,z['expected_index'][i])
        allowance=np.r_[(1-config['target'])*plan[decision],b0]
        pending[i]=(decision,booked[decision]-allowance,allowance)
        chosen.append(int(decision))
    return dict(matched_issued_days=len(chosen),first_action=chosen[0],last_action=chosen[-1])

def physical(branch):
    z=np.load(ROOT/f'examples/R3_{branch}.npz')
    args=[z[key] for key in ('y','v','pl','gg','w','upper','haz')]
    if branch=='O':
        params=(3,float(z['lam']),33.3,16.65,9.99,float(z['B']),3.33,True)
        functions=[original_ordinary,revised_ordinary]
    else:
        params=(float(z['lam']),float(z['B']),float(z['booked_R'])/float(z['factor']),float(z['booked_T'])/float(z['factor']))
        functions=[original_hard,revised_hard]
    out={}
    for name,fn in zip(['original','revised'],functions):
        a=fn(*args,*params)*float(z['factor'])
        err=float(abs(a-z[name+'_actions']).max())
        np.testing.assert_allclose(a,z[name+'_actions'],atol=1.e-6,rtol=0)
        balance=a[:,0]+a[:,3]+a[:,4]+a[:,5]-a[:,1]-a[:,2]-a[:,6]
        soc=a[:,7]+.25*(.95*a[:,2]-a[:,3]/.95)-a[:,8]
        np.testing.assert_allclose([max(abs(balance)),max(abs(soc)),a[-1,8]-a[0,7]],0,atol=1.e-6)
        assert np.all(a[:,2:4]<=16.65+1.e-7) and np.all(a[:,4]<=9.99+1.e-7)
        np.testing.assert_allclose(a[:,9]-.25*a[:,4],a[:,10],atol=1.e-7)
        if branch=='H':
            assert a[:,4].sum()*.25<=float(z['booked_R'])+1.e-7
            assert a[:,2:4].sum()*.25<=float(z['booked_T'])+1.e-7
        out[name]=dict(max_step_error=err,steps=len(a),max_balance_residual_MW=float(abs(balance).max()),max_SOC_residual_MWh=float(abs(soc).max()))
    return out
if __name__=='__main__':
    print(json.dumps(dict(midnight={m:issue(m) for m in ['D0','M1','M3']},dispatch={b:physical(b) for b in ['O','H']}),indent=2))
