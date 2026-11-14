"""One command runs the inference, accounting, physics and numerics regression tests."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile
ROOT=Path(__file__).absolute().parents[1]
def main():
    ledger=(ROOT/'MANIFEST.sha256').read_text().splitlines()
    expected={}
    for line in ledger:
        checksum,path=line.split('  ',1)
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==checksum,path
        expected[path]=checksum
    outputs={}
    with tempfile.TemporaryDirectory() as cache:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=cache,NUMBA_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for script in ['scripts/run_inference.py','scripts/run_replay.py','scripts/reproduce_tables.py','scripts/p18_close/verify_minimal.py']:
            completed=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/script)],check=True,cwd=ROOT,env=env,stdout=subprocess.PIPE,text=True)
            outputs[script]=json.loads(completed.stdout)
    print(json.dumps(dict(verified_payload_files=len(expected),tests=outputs),indent=2))
if __name__=='__main__':main()
