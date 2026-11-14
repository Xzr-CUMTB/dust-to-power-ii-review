"""Standalone NumPy/math verification of saved first-divergence evidence (no project models)."""
from pathlib import Path
import math,json,sys
import numpy as np
R=Path(__file__).absolute().parents[2]
def risk(v):
    q=sorted(float(x) for x in v);eta=q[min(len(q)-1,int(math.ceil(.875*len(q))))-1]
    return eta+math.fsum(max(v-eta,0.) for v in q)/(len(q)*.125)
def scores(z,postfix):
    tr=z['trace'+postfix];ct=z['contributions'+postfix]
    mean=np.array([(math.fsum(ct[j].ravel().tolist())-math.fsum(tr[j,:,1].tolist()))/32 for j in range(21)])
    stress=min(1.,max(0.,.35*(1-float(z['b'])/25)+.25*(1-(float(z['s'])-3.33)/26.64)+.2))
    mult=.05*(.35+1.65*stress)
    return mean+mult*np.array([risk(tr[j,:,2])+risk(tr[j,:,3]) for j in range(21)])
def main():
    rows=[]
    for p in sorted((R/'results/p18_close/trace').glob('*_first_difference.npz')):
        z=dict(np.load(p));perm=z['perm']
        assert np.array_equal(z['v'][:,perm],z['vp'])
        assert np.array_equal(z['gg'][:,:,perm],z['gp'])
        assert np.array_equal(z['trace'][:,perm],z['tracep'])
        assert np.array_equal(z['contributions'][:,perm],z['contributionsp'])
        assert np.array_equal(z['actions'],z['actionsp'])
        a=scores(z,'');b=scores(z,'p');assert np.array_equal(a,b)
        first=lambda v:int(np.flatnonzero(v<=v.min()+1e-9)[0])
        assert first(a)==first(b)
        rows.append(dict(file=p.name,old_choices=[int(np.argmin(z['obj'])),int(np.argmin(z['objp']))],
            stable_choices=[first(a),first(b)],highprecision_objective_difference=float(abs(a-b).max()),
            h0_gap_spread=float(np.ptp(z['gg'][int(z['k']),:,:,0],axis=1).max())))
    assert len(rows)==2
    print(json.dumps(dict(status='passed',cases=rows,dependency='NumPy and Python stdlib only'),indent=2))
if __name__=='__main__':main()
