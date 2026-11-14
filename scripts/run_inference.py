"""Reload the frozen CPU inference paths and reproduce the first eligible weather-day example."""
from pathlib import Path
import json,sys
import numpy as np
import torch
import lightgbm as lgb

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.p11.quantile import Tail
from src.models.p15.readout import ResourceReadout
from src.scenarios.donors import Q,quantcal
from src.operations.p10.budget import scenario_account
from src.dispatch.p15.decision import B,C

def generation():
    example=np.load(ROOT/'examples/M3.npz')
    x,w,anchors=[torch.as_tensor(example[k][None],dtype=torch.float32) for k in ('x','weather','anchors')]
    net=Tail(x.shape[1],len(Q))
    net.load_state_dict(torch.load(ROOT/'checkpoints/M3.pt',map_location='cpu',weights_only=True)['state'])
    net.eval()
    with torch.no_grad():
        quantiles,tilts=net(x,w,anchors)
    q=quantiles.numpy()[0]*float(example['energy_scale'])
    t=tilts.numpy()[0]
    energies=np.interp((np.arange(32)+.5)/32,Q,q)
    shaped=example['donor_bank']*np.repeat(np.exp(t),12)[None,:,None]
    scenarios=(shaped*(energies/(.25*shaped.sum((1,2))))[:,None,None]).astype('f4')
    np.testing.assert_allclose(q,example['q'],atol=1.e-4,rtol=2.e-6)
    np.testing.assert_allclose(scenarios,example['paths'],atol=1.e-4,rtol=2.e-6)
    dates=example['donor_days'].astype('datetime64[D]')
    assert np.all(dates<=np.datetime64(str(example['day']))-np.timedelta64(2,'D'))
    issue=np.load(ROOT/'examples/issue_context.npz')
    residual_days=issue['residual_dates'].astype('datetime64[D]')
    assert np.all(residual_days<=np.datetime64(str(issue['day']))-np.timedelta64(2,'D'))
    scenes=quantcal(scenarios.astype(float).sum(-1),list(zip(residual_days,issue['residuals'])))
    proxies=np.stack([scenario_account(scenes,issue['reference'],c,b).mean(0)[:3] for c,b in zip(C,B)])
    expected=np.load(ROOT/'examples/candidates_M3.npz')['raw'][0]
    np.testing.assert_allclose(proxies,expected,atol=5.e-5,rtol=1.e-6)
    tree=np.load(ROOT/'examples/M1.npz')
    preds=[lgb.Booster(model_str=file.read_text(encoding='utf-8')).predict(tree['features'][None],num_threads=1)[0] for file in sorted((ROOT/'checkpoints/M1').glob('quantile_*.txt'))]
    assert len(preds)==len(Q)
    np.testing.assert_allclose(np.maximum(.01,np.sort(preds)),tree['q'])
    return dict(day=str(example['day']),members=len(scenarios),shape=list(scenarios.shape),max_absolute_path_difference_MW=float(abs(scenarios-example['paths']).max()),maximum_energy_residual_MWh=float(abs(scenarios.astype(float).sum((1,2))*.25-energies).max()),M1_quantiles=len(preds),maximum_candidate_proxy_error_MWh=float(abs(proxies-expected).max()))

def consequences(name):
    z=np.load(ROOT/f'examples/candidates_{name}.npz')
    w=torch.load(ROOT/f'checkpoints/E0_{name}.pt',map_location='cpu',weights_only=True)
    # Broadcast the daily state across the 91 possible plans; append candidate-specific inputs.
    context=np.broadcast_to(z['ctx'][:,None],(len(z['days']),91,z['ctx'].shape[-1]))
    features=np.concatenate((context,z['act']),axis=2)
    normalized=np.clip((features-w['mean'].numpy())/w['sd'].numpy(),-5,5)
    net=ResourceReadout(features.shape[-1])
    net.load_state_dict(w['weights']);net.eval()
    with torch.no_grad():
        loc,scale=net(torch.as_tensor(normalized,dtype=torch.float32))
    units=np.array([20.,10.,25.])
    center=z['raw']+loc.numpy()*units
    width=scale.numpy()*units
    np.testing.assert_allclose(center,z['expected_mean'],rtol=1.e-6,atol=1.e-5)
    np.testing.assert_allclose(width,z['expected_width'],rtol=1.e-6,atol=1.e-5)
    return center,width

if __name__=='__main__':
    torch.set_num_threads(1)
    result=generation()
    for method in ('D0','M1','M3'):
        mean,width=consequences(method)
        z=np.load(ROOT/f'examples/candidates_{method}.npz')
        result[method]=dict(days=len(mean),candidate_count=mean.shape[1],max_mean_difference_MWh=float(abs(mean-z['expected_mean']).max()),max_width_difference_MWh=float(abs(width-z['expected_width']).max()))
    print(json.dumps(result,indent=2))
