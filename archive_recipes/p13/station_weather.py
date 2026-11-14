from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,sys,json
from io import BytesIO
import numpy as np,pandas as pd,requests
from scipy.io import netcdf_file
R=Path(__file__).absolute().parents[2]; dest=R/'results/p13/stations';dest.mkdir(exist_ok=True,parents=True)
raw=R/'data/raw/p13_gfs';raw.mkdir(exist_ok=True,parents=True)
root=R/'data/raw/pv_original/PVODdatasets_v1.0'
stations=['02','01','03','05','06','09']
meta=pd.read_csv(root/'metadata.csv').set_index('Station_ID')
def siteframe(s):
    fr=pd.read_csv(root/f'station{s}.csv');idx=pd.to_datetime(fr.date_time,utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    return fr.set_index(pd.DatetimeIndex(idx)).loc[~idx.duplicated(False).values].sort_index()
def register():
    ds={}
    for s in [f'{i:02}' for i in range(10)]:
        f=siteframe(s);ds[s]=f.power
    wx=pd.read_csv(R/'results/p11/weather_all.csv');good=wx.groupby('day').eligible.all();allowed=set(good[good].index)
    rows=[];days=[]
    for s in stations:
        f=ds[s];g=f.where(np.isfinite(f)&(f>=0)).groupby((f.index-pd.Timedelta('15min')).normalize()).count();full=g[g==96].index.strftime('%Y-%m-%d')
        for d in full:
            if '2019-03-05'<=d<='2019-06-12' and d in allowed:days.append(dict(site=s,day=d))
        m=meta.loc['station'+s];twins=[]
        for other in ds:
            if other==s:continue
            tab=pd.concat([f,ds[other]],axis=1,join='inner').dropna()
            if len(tab) and np.array_equal(tab.iloc[:,0],tab.iloc[:,1]):twins.append(other)
        rows.append(dict(site=s,lat=m.Latitude,lon=m.Longitude,MW=m.Capacity/1000,first_utc=str(f.index[0]-pd.Timedelta('8h')),last_utc=str(f.index[-1]-pd.Timedelta('8h')),full_days=len(full),eligible_subset=sum(r['site']==s for r in days),exact_label_twins=','.join(twins),sha256=hashlib.sha256((root/f'station{s}.csv').read_bytes()).hexdigest(),inverter_model=m.Inverters,device_identity='separate station files and coordinates; individual device serial linkage unavailable',prior_review='P4 coverage/event review; no P13 engineering scores'))
    pd.DataFrame(rows).to_csv(dest/'inventory.csv',index=False);pd.DataFrame(days).to_csv(dest/'calendar.csv',index=False)
def load(task):
    d,h,group=task
    source=pd.read_json(R/f'data/raw/p11_gfs/{d}_{h}.json').iloc[0];p=(R/f'data/raw/p11_gfs/{d}_{h}.nc') if h!=78 else R/f'data/raw/p10_weather/gfs_checked_{d}_3.nc'
    url=source['subset'];assert source.eligible
    if group=='east':
        url=url.replace('east=115.5','east=117.75').replace('west=113','west=117.25').replace('north=40','north=38.5').replace('south=36','south=38')
        assert url.startswith('https://tds.gdex.ucar.edu/thredds/ncss/grid/files/g/d084001/')
        p=raw/f'{d}_{h}.nc'
        if p.exists():
            try:
                with netcdf_file(BytesIO(p.read_bytes()),'r',mmap=False) as check:
                    assert any('Short-Wave' in k for k in check.variables)
            except (ValueError,IndexError,TypeError,AssertionError):
                p.rename(p.with_suffix('.incomplete'))
        if not p.exists():
            url=url.replace('west=117.25','west=117.20')
            with requests.get(url,timeout=(10,90),stream=True) as resp:
                resp.raise_for_status();parts=[]
                for block in resp.iter_content(8192):
                    parts.append(block);assert sum(map(len,parts))<=100000
                data=b''.join(parts)
            assert data[:3]==b'CDF'
            with netcdf_file(BytesIO(data),'r',mmap=False) as check:
                assert all(any(k.startswith(v) for k in check.variables) for v in ['Downward_Short','Total_cloud_cover_entire'])
            p.write_bytes(data)
            p.with_suffix('.url').write_text(url,encoding='utf8')
        if p.with_suffix('.url').exists():url=p.with_suffix('.url').read_text(encoding='utf8')
    data=p.read_bytes();sig=hashlib.sha256(data).hexdigest()
    if group!='east':assert sig==source.sha256
    with netcdf_file(BytesIO(data),'r',mmap=False) as ds:
        lat=next(v[:] for k,v in ds.variables.items() if k.startswith('lat'));lon=next(v[:] for k,v in ds.variables.items() if k.startswith('lon'));rows=[]
        for s in (['01'] if group=='east' else [v for v in stations if v!='01']):
            m=meta.loc['station'+s];i=np.abs(lat-m.Latitude).argmin();j=np.abs(lon-m.Longitude).argmin()
            assert abs(lat[i]-m.Latitude)<=.126 and abs(lon[j]-m.Longitude)<=.126
            for field,prefix in [('rad','Downward_Short'),('cloud','Total_cloud_cover_entire')]:
                v=next(v for k,v in ds.variables.items() if k.startswith(prefix) and '6_Hour_Average' in k);tv=ds.variables[v.dimensions[0]]
                start=pd.to_datetime(tv.units.decode().split(' since ')[1],utc=True);bounds=start+pd.to_timedelta(ds.variables[tv.bounds.decode()][:].ravel(),unit='h')
                assert bounds[0]==pd.Timestamp(source.start) and bounds[-1]==pd.Timestamp(source.end)
                value=float(v[:].reshape(-1,len(lat),len(lon))[0,i,j]);assert 0<=value<=(1500 if field=='rad' else 100)
                rows.append(dict(site=s,day=d,h=h,field=field,value=value,unit=v.units.decode(),grid_lat=lat[i],grid_lon=lon[j],init=source.init,mtime=source.mtime,issue=source.issue,start=source.start,end=source.end,url=url,sha256=sig))
    return rows
def run(group):
    days=sorted(pd.read_csv(dest/'calendar.csv').day.unique());tasks=[(d,h,group) for d in days for h in [72,78,84,90]]
    def safe(t):
        try:return load(t)
        except Exception as e:
            print(t,str(e)[:240],flush=True);return []
    if group=='pilot':return load((days[0],72,'east'))
    with ThreadPoolExecutor(8) as pool:rows=[r for g in pool.map(safe,tasks) for r in g]
    pd.DataFrame(rows).to_parquet(dest/f'weather_{group}.parquet',index=False)
    assert len(rows)==len(tasks)*(2 if group=='east' else 10),(len(rows),len(tasks))
if __name__=='__main__':
    if sys.argv[1]=='register':register()
    else:run(sys.argv[1])

