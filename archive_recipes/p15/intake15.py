"""Package issue-time signals and keep previous-period training outcomes in separate files."""
from paths15 import *
sys.path.insert(0,str(ROOT/'scripts/p14'))
from io14 import dataset, segments, calibrate, proposed, CGRID, BGRID
from plant14 import prepared
import numpy as np
import pandas as pd

def build(site):
    forecasts=['D0','M1','M3']+(['seed3407','seed2025','M3noPM'] if site=='core' else [])
    for name in forecasts:
        inputs=[]
        earlier=None
        for f in segments(site):
            b=dataset(site,f,'M3' if name=='M3noPM' else name,'none' if name=='M3noPM' else 'full')
            if earlier is None:
                earlier=b
            k=b['n']
            take=np.arange(len(b['days'])) if f==segments(site)[0] else np.arange(k,len(b['days']))
            inputs.append({key:b[key][take] for key in ['days','ctx','raw','plan','ref','act']})
            inputs[-1]['fold']=np.array([f]*len(take))
        data={key:np.concatenate([z[key] for z in inputs]) for key in inputs[0]}
        days=data['days']; n=earlier['n']; split=n//2
        assert np.all(np.diff(np.asarray(days,dtype='datetime64[D]')).astype(int)>0)
        dev=earlier['y'][:n]
        if name=='D0':
            np.savez_compressed(OUT/'development'/f'{site}.npz',days=days[:n],y=dev)
            if site=='core':
                r=np.load(ROOT/'results/p11/refs.npz')
                lookup=dict(zip(r['days'],r['ref']))
                shape=np.stack([np.asarray(lookup[d]).reshape(8,12).sum(1)*.25 for d in days])
            else:
                shape=np.stack([prepared(site,str(d))[2].reshape(96,-1)[:,0].reshape(8,12).sum(1)*.25 for d in days])
            shape/=shape.sum(1,keepdims=True).clip(.001)
            np.save(OUT/'inputs'/f'{site}_shape.npy',shape)
        local=np.zeros((len(days),2),int)
        for ix,g in enumerate([range(split,n),range(n,len(days))]):
            first=g[0];cut=dict(earlier)
            cut['n']=sum(np.datetime64(str(days[first]))-np.datetime64(str(day))>=np.timedelta64(2,'D') for day in days[:first])
            for ti,t in enumerate(TARGETS):
                best,_,_=calibrate(cut,t)
                for i in g:
                    scores=data['raw'][i,78:,0];plan=data['plan'][i,78:]
                    if best['kind']=='fixed':j=78+int(np.argmin(abs(CGRID[78:]-best['parameter'])))
                    else:
                        ok=np.flatnonzero(scores<=plan*max(0,1-t-best['parameter']))
                        j=78+(ok[-1] if len(ok) else 0)
                    local[i,ti]=j
        np.savez_compressed(OUT/'inputs'/f'{site}_{name}.npz',**data,n=n,split=split,factor=earlier['factor'],local=local)
if __name__=='__main__':
    for s in sys.argv[1:] or SITES:build(s)

