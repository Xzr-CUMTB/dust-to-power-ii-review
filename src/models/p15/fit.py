"""Original 48-epoch E0 fit. Configure CK and OUT before running a training experiment."""
import time,json
from pathlib import Path
import numpy as np
import torch
from src.models.p15.readout import ResourceReadout
CK=Path("checkpoints"); OUT=Path("results");
def write_json(p,obj):
    p.write_text(json.dumps(obj,default=str),encoding="utf-8")
def stage_ids(z):
    n,split=int(z['n']),int(z['split'])
    dates=z['days'].astype('datetime64[D]')
    for stage,start,end in [('validation',split,n),('eval',n,len(dates))]:
        tr=np.flatnonzero((np.arange(len(dates))<start)&(dates<=dates[start]-np.timedelta64(2,'D')))
        assert len(tr)>6 and tr[-1]<n
        yield stage,tr,np.arange(start,end)

def neural(x,y,train,infer,seed,name,dates):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    mu=x[train].reshape(-1,x.shape[-1]).mean(0);sd=x[train].reshape(-1,x.shape[-1]).std(0).clip(.08)
    inp=torch.as_tensor(((x-mu)/sd).clip(-5,5),dtype=torch.float32,device='cuda')
    target=torch.as_tensor(y[train],dtype=torch.float32,device='cuda')
    model=ResourceReadout(x.shape[-1]).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.035)
    trace=[]
    for epoch in range(48):
        mean,b=model(inp[train])
        resid=target-mean
        loss=(resid.abs()/b+b.log()).mean()+.25*torch.nn.functional.smooth_l1_loss(mean,target)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.5);opt.step()
        trace.append(float(loss.detach()))
    path=CK/(name+'.pt')
    torch.save(dict(weights=model.state_dict(),mean=torch.tensor(mu),sd=torch.tensor(sd)),path)
    restored=ResourceReadout(x.shape[-1]).cuda()
    restored.load_state_dict(torch.load(path,weights_only=True)['weights'])
    with torch.no_grad():
        ans=restored(inp[infer]);check=model(inp[infer])
        assert max(float((p-q).abs().max()) for p,q in zip(ans,check))<1e-6
    write_json(OUT/'logs'/(name+'.json'),dict(train=dates[train].tolist(),first_prediction=str(dates[infer[0]]),loss=trace,device=torch.cuda.get_device_name(),seed=seed))
    return [p.cpu().numpy() for p in ans]
