"""Only public historical UCAR NCSS subsets; no write outside P11."""
from pathlib import Path
import sys,re,json,hashlib,datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'scripts/p10'))
from weather import session,read_http,coords,np,pd,requests
from scipy.io import netcdf_file
out=ROOT/'data/raw/p11_gfs';out.mkdir(parents=True,exist_ok=True)
dest=ROOT/'results/p11';dest.mkdir(parents=True,exist_ok=True)
base='https://tds.gdex.ucar.edu/thredds/'
fields=['Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average','Total_cloud_cover_entire_atmosphere_6_Hour_Average']
def fetch(task):
 d,h=task; stem=out/f'{d}_{h}';meta=stem.with_suffix('.json')
 if meta.exists():return json.loads(meta.read_text())
 init=pd.Timestamp(d,tz='UTC')-pd.Timedelta('3d');issue=pd.Timestamp(d,tz='Asia/Shanghai').tz_convert('UTC')
 key=f'files/g/d084001/{init:%Y}/{init:%Y%m%d}/gfs.0p25.{init:%Y%m%d%H}.f{h:03d}.grib2'
 url=base+'fileServer/'+key
 query=requests.Request('GET',base+'ncss/grid/'+key,params=dict(var=','.join(fields),north=40,south=36,east=115.5,west=113,time='all',accept='netcdf3')).prepare().url
 if h==78:
  m=json.loads((ROOT/f'results/p10/gfs_checked_{d}_3.json').read_text()); raw=(ROOT/f'data/raw/p10_weather/gfs_checked_{d}_3.nc').read_bytes()
  assert m['sha256']==hashlib.sha256(raw).hexdigest();mtime=pd.Timestamp(m['archive_mtime']);acq='P10 cache; original acquisition clock not logged'
 else:
  r=session().head(url,timeout=16);r.raise_for_status();mtime=pd.Timestamp(parsedate_to_datetime(r.headers['Last-Modified']))
  raw,_=read_http(query,ceiling=100000);acq=str(pd.Timestamp.now(tz='UTC'))
 valid=init+pd.Timedelta(hours=h);rows=[]
 with netcdf_file(BytesIO(raw),'r',mmap=False) as ds:
  lat=ds.variables[next(k for k in ds.variables if k.startswith('lat'))][:].copy();lon=ds.variables[next(k for k in ds.variables if k.startswith('lon'))][:].copy()
  for field in fields:
   v=ds.variables[field];tv=ds.variables[v.dimensions[0]];origin=pd.to_datetime(tv.units.decode().split(' since ')[1],utc=True)
   bounds=origin+pd.to_timedelta(ds.variables[tv.bounds.decode()][:].ravel(),unit='h')
   assert len(bounds)==2 and bounds[1]==valid and bounds[1]-bounds[0]==pd.Timedelta('6h')
   dat=v[:].reshape(-1,len(lat),len(lon))[0]
   for s in coords.itertuples():
    i=np.abs(lat-s.pv_latitude).argmin();j=np.abs(lon-s.pv_longitude).argmin()
    assert abs(lat[i]-s.pv_latitude)<=.126 and abs(lon[j]-s.pv_longitude)<=.126
    value=float(dat[i,j]);assert 0<=value<=(1500 if field==fields[0] else 100)
    rows.append(dict(day=d,h=h,station=s.pv_station,field='rad' if field==fields[0] else 'cloud',variable=field,unit=v.units.decode(),value=value,lat=float(lat[i]),lon=float(lon[j]),start=str(bounds[0]),end=str(bounds[1]),init=str(init),issue=str(issue),age_h=(issue-init).total_seconds()/3600,mtime=str(mtime),eligible=mtime<=issue,url=url,subset=query,acquired=acq,sha256=hashlib.sha256(raw).hexdigest()))
 if h!=78:stem.with_suffix('.nc').write_bytes(raw)
 meta.write_text(json.dumps(rows),encoding='utf8');return rows
def safe(task):
 try:return fetch(task)
 except Exception as e:print(task,type(e).__name__,str(e)[:250],flush=True);return []
def main(stage):
 days=pd.read_csv(ROOT/'results/p9/day_labels.csv').day if stage=='all' else ['2019-05-09','2019-05-11','2019-05-13']
 tasks=[(d,h) for d in days for h in (72,78,84,90)]
 with ThreadPoolExecutor(6) as ex:rows=[r for part in ex.map(safe,tasks) for r in part]
 df=pd.DataFrame(rows);df.to_csv(dest/f'weather_{stage}.csv',index=False)
 print(len(rows),'of',len(tasks)*8,flush=True)
 if len(rows)!=len(tasks)*8:
  (dest/f'weather_incomplete_{stage}_{len(rows)}.json').write_text(json.dumps([t for t in tasks if not (out/f'{t[0]}_{t[1]}.json').exists()]));sys.exit(2)
 if stage=='all':(dest/'weather_complete.json').write_text(json.dumps(dict(days=len(days),rows=len(rows),all_eligible=bool(df.eligible.all()))))
if __name__=='__main__':main(sys.argv[1])

