"""Check and prepare separately obtained original PVOD and station-air observations in a scratch root."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,zipfile
import numpy as np,pandas as pd
ROOT=Path(__file__).absolute().parents[1]
PV_SHA='0f726b42235bf99a2c118a3cb7544722150489b01af3a2853a7c9778b538883e'
def prepare_pv(archive,workspace):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=PV_SHA:
        raise ValueError('PVOD archive fingerprint differs from the screened V5 listing.')
    destination=workspace/'data/raw/pv_original/PVODdatasets_v1.0'
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        if source.testzip() is not None:raise ValueError('Invalid ZIP CRC')
        for name in source.namelist():
            filename=Path(name).name
            if filename.endswith('.csv') and not filename.startswith('.'):
                target=destination/filename
                payload=source.read(name)
                if target.exists() and target.read_bytes()!=payload:
                    raise FileExistsError(f'Conflicting input file: {filename}')
                target.write_bytes(payload)
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/audit/audit_original_pv.py'),'--root',str(workspace)],check=True)
def prepare_air(hourly,workspace):
    air=pd.read_csv(hourly,dtype={'site_code':str,'type':str})
    air['observation_time_utc']=pd.to_datetime(air.time_bjt,utc=True)
    keys=['observation_time_utc','site_code','type']
    if air.duplicated(keys).any():raise ValueError('Repeated monitor observation')
    hours=pd.date_range(air.observation_time_utc.min(),air.observation_time_utc.max(),freq='h')
    monitors=sorted(air.site_code.unique())
    if len(monitors)!=31:raise ValueError('Expected 31 registered monitor IDs')
    grid=pd.MultiIndex.from_product([hours,monitors,['PM10','PM2.5']],names=keys)
    air['source_hour_record_present']=True
    x=air.set_index(keys)[['value','source_hour_record_present']].reindex(grid).reset_index()
    x['source_hour_record_present']=x.source_hour_record_present.eq(True)
    x['value_present']=np.isfinite(x.value)
    x['available_time_scenario_utc']=x.observation_time_utc+pd.Timedelta(hours=3)
    output=workspace/'data/processed';output.mkdir(parents=True,exist_ok=True)
    x.to_parquet(output/'air_hourly_monitors.parquet',index=False)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--pvod',type=Path)
    parser.add_argument('--monitor-hourly',type=Path)
    args=parser.parse_args()
    if args.pvod is None and args.monitor_hourly is None:parser.error('Specify an archive or merged monitor input')
    if args.pvod:prepare_pv(args.pvod,args.workspace)
    if args.monitor_hourly:prepare_air(args.monitor_hourly,args.workspace)
