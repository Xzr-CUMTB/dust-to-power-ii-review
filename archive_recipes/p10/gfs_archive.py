"""Conservative GFS archive timing: retain server mtime, initialization, and actual valid time."""
from weather import *
from email.utils import parsedate_to_datetime
from scipy.io import netcdf_file
from io import BytesIO

TDS='https://tds.gdex.ucar.edu/thredds/'
FIELDS=['Downward_Short-Wave_Radiation_Flux_surface','Total_cloud_cover_entire_atmosphere']
def identity(day,age):
    init=pd.Timestamp(day,tz='UTC')-pd.Timedelta(days=age)
    if age==1:init+=pd.Timedelta(hours=12)
    target=pd.Timestamp(day,tz='UTC')+pd.Timedelta(hours=6)
    h=int((target-init).total_seconds()/3600)
    key=f'files/g/d084001/{init:%Y}/{init:%Y%m%d}/gfs.0p25.{init:%Y%m%d%H}.f{h:03d}.grib2'
    return init,target,h,key
def fetch_day(task):
    day,age=task;init,target,h,key=identity(day,age)
    issue=pd.Timestamp(day,tz='Asia/Shanghai').tz_convert('UTC')
    file=OUT/f'gfs_checked_{day}_{age}.json'
    if file.exists():
        cached=json.loads(file.read_text())
        if cached.get('status')=='ok' and len(cached.get('rows',[]))==8:return cached
    report=dict(day=day,age=age,init=str(init),valid=str(target),issue=str(issue),url=TDS+'fileServer/'+key,rows=[])
    try:
        response=session().head(report['url'],timeout=18);response.raise_for_status()
        mtime=pd.Timestamp(parsedate_to_datetime(response.headers['Last-Modified']))
        report.update(archive_mtime=str(mtime),mtime_before_issue=mtime<=issue,mtime_before_06=mtime<=issue+pd.Timedelta('6h'))
        dds,_=read_http(TDS+'dodsC/'+key+'.dds',ceiling=400000)
        names=re.findall(r'Float(?:32|64)\s+([\w-]+)\[',dds.decode())
        fields=[next((k for k in names if k.startswith(prefix)),None) for prefix in FIELDS]
        fields=[v for v in fields if v]
        report['varnames']=fields
        if not fields:raise ValueError('no target variables in DDS: '+dds[:160].decode(errors='replace'))
        query={'var':','.join(fields),'north':40,'south':36,'east':115.5,'west':113.0,'time':'all','accept':'netcdf3'}
        url=requests.Request('GET',TDS+'ncss/grid/'+key,params=query).prepare().url
        raw,_=read_http(url,ceiling=350000)
        with netcdf_file(BytesIO(raw),'r',mmap=False) as ds:
            coords_names=[k for k in ds.variables if k.startswith('lat') or k.startswith('lon')]
            lat=ds.variables[next(k for k in coords_names if k.startswith('lat'))][:].copy()
            lon=ds.variables[next(k for k in coords_names if k.startswith('lon'))][:].copy()
            for name in fields:
                var=ds.variables[name]
                data=var[:].copy().reshape(-1,len(lat),len(lon))[0]
                timevar=ds.variables[var.dimensions[0]]
                unit,origin=timevar.units.decode().split(' since ')
                actual=pd.to_datetime(origin,utc=True)+pd.to_timedelta(float(timevar[:].ravel()[0]),unit={'Hour':'h','hours':'h','Hour(s)':'h','hour':'h'}[unit])
                bounds_name=getattr(timevar,'bounds',b'').decode()
                span=None
                if bounds_name in ds.variables:
                    bv=ds.variables[bounds_name][:].ravel()
                    span=[pd.to_datetime(origin,utc=True)+pd.to_timedelta(float(t),unit='h') for t in bv]
                report['time_check']=dict(coordinate=str(actual),bounds=str(span),expected_end=str(target))
                assert actual==target or (span and span[-1]==target),report['time_check']
                for s in coords.itertuples():
                    i=np.abs(lat-s.pv_latitude).argmin();j=np.abs(lon-s.pv_longitude).argmin()
                    assert abs(lat[i]-s.pv_latitude)<=.126 and abs(lon[j]-s.pv_longitude)<=.126
                    report['rows'].append(dict(station=s.pv_station,grid_lon=float(lon[j]),grid_lat=float(lat[i]),field=name,unit=var.units.decode(),value=float(data[i,j]),time_value=str(actual),time_bounds=str(span)))
        (DEST/f'gfs_checked_{day}_{age}.nc').write_bytes(raw)
        report.update(bytes=len(raw)+len(dds),status='ok',sha256=hashlib.sha256(raw).hexdigest())
    except Exception as exc:report['status']=repr(exc)
    file.write_text(json.dumps(report,indent=2),encoding='utf8');return report
def run(stage):
    if stage=='pilot':tasks=[('2019-05-11',1),('2019-05-11',2)]
    elif stage=='week':tasks=[(d,age) for d in pd.date_range('2019-05-08',periods=7).strftime('%Y-%m-%d') for age in [1,2,3]]
    elif stage=='repair':
        tasks=[]
        for p in OUT.glob('gfs_checked_*_3.json'):
            r=json.loads(p.read_text())
            if r['status']!='ok':tasks.append((r['day'],3))
    else:tasks=[(d,3) for d in pd.read_csv(ROOT/'results/p9/day_labels.csv').day]
    with ThreadPoolExecutor(max_workers=4) as ex:status=list(ex.map(fetch_day,tasks))
    print([(r['day'],r['age'],r.get('mtime_before_issue'),r['status']) for r in status[:10]],flush=True)
if __name__=='__main__':run(sys.argv[1])

