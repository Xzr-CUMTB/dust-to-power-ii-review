"""Acquire the public ScienceDB V5 PVOD archive and preserve source bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT=Path(os.path.abspath(__file__)).parents[2]
META=ROOT/"data/metadata"
OUT=META/"pv_sources"
RAW=ROOT/"data/raw/pv_original"
DATASET="f8f3d7af144f441795c5781497e56b62"
BASE="https://www.scidb.cn"
EXPECTED_SIZE=7983824
EXPECTED_MD5="5cb8ebfb4cdc99973deacf1bea8bacc3"
LOG=ROOT/"logs/download_pv_original.jsonl"

def log(event,**kwargs):
    row={"utc":datetime.now(timezone.utc).isoformat(),"event":event,**kwargs}
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps(row,ensure_ascii=False),flush=True)

def validate(path):
    blob=path.read_bytes()
    if len(blob)!=EXPECTED_SIZE or hashlib.md5(blob).hexdigest()!=EXPECTED_MD5:
        raise ValueError("Original archive differs from published V5 size/MD5")
    with zipfile.ZipFile(path) as archive:
        bad=archive.testzip()
        if bad:
            raise ValueError("Original archive CRC failed: "+bad)
    return {"bytes":len(blob),"md5":hashlib.md5(blob).hexdigest(),"sha256":hashlib.sha256(blob).hexdigest()}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--refresh-metadata",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    RAW.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    session.headers.update({"User-Agent":"DustToPowerII-public-research/1.0","Referer":BASE+"/en/detail?dataSetId="+DATASET})
    tree_path=OUT/"scidb_public_file_tree.json"
    if args.refresh_metadata or not tree_path.exists():
        url=BASE+"/api/gin-sdb-filetree/public/file/childrenFileListByPath"
        response=session.post(url,json={"dataSetId":DATASET,"version":"V5","path":"/V5","lastIndex":0,"pageSize":100},timeout=(20,60))
        response.raise_for_status()
        tree_path.write_bytes(response.content)
        log("public_listing",url=url,status=response.status_code)
    tree=json.loads(tree_path.read_text(encoding="utf-8"))
    if tree.get("code")!=20000:
        raise RuntimeError("ScienceDB public file listing unavailable; no authentication bypass attempted")
    files=[item for item in tree.get("data",[]) if not item.get("dir") and item.get("suffix")=="zip"]
    if len(files)!=1:
        raise ValueError("Expected one original V5 archive")
    item=files[0]
    if item["size"]!=EXPECTED_SIZE or item["md5"]!=EXPECTED_MD5:
        raise ValueError("Public listing no longer matches recorded source version")
    path=RAW/item["fileName"]
    part=path.with_suffix(path.suffix+".part")
    url="https://china.scidb.cn/download?fileId="+item["id"]
    if not path.exists():
        for attempt in range(1,4):
            try:
                offset=part.stat().st_size if part.exists() else 0
                headers={"Range":f"bytes={offset}-"} if offset else {}
                with session.get(url,stream=True,timeout=(20,60),headers=headers) as response:
                    response.raise_for_status()
                    if "text/html" in response.headers.get("Content-Type",""):
                        (OUT/"scidb_download_landing.html").write_bytes(response.content)
                        raise ValueError("Public download returned HTML, not data")
                    if response.status_code==206:
                        if not response.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                            raise ValueError("Mismatched resume range")
                        mode="ab"
                    else:
                        offset,mode=0,"wb"
                    log("download_start",source=url,attempt=attempt,status=response.status_code,offset=offset)
                    count=0
                    with part.open(mode) as stream:
                        for chunk in response.iter_content(1024**2):
                            if chunk:
                                stream.write(chunk)
                                count+=len(chunk)
                    expected_chunk=response.headers.get("Content-Length")
                    if expected_chunk and count!=int(expected_chunk):
                        raise ValueError("Incomplete HTTP response body")
                validate(part)
                part.replace(path)
                break
            except Exception as exc:
                log("download_error",attempt=attempt,error=str(exc))
                if attempt==3:
                    raise
                time.sleep(attempt*2)
    row={"file":item["fileName"],"dataset_doi":"10.11922/sciencedb.01094","version":"V5/5.0.0",
         "source_page":BASE+"/en/detail?dataSetId="+DATASET,"source":url,
         "expected_bytes":EXPECTED_SIZE,"expected_md5":EXPECTED_MD5,**validate(path),
         "path":str(path.relative_to(ROOT)),"verified":True,"verification_utc":datetime.now(timezone.utc).isoformat()}
    target=RAW/path.stem
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        row["members"]=archive.namelist()
        for member in archive.infolist():
            dest=Path(os.path.abspath(target/member.filename))
            if os.path.commonpath([str(dest),str(target.absolute())])!=str(target.absolute()):
                raise ValueError("Unsafe original archive member")
            if member.is_dir():
                dest.mkdir(parents=True,exist_ok=True)
                continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                blob=dest.read_bytes()
                if len(blob)!=member.file_size or (zlib.crc32(blob)&0xffffffff)!=member.CRC:
                    raise ValueError("Existing raw extracted file modified; refusing overwrite: "+str(dest))
                continue
            with archive.open(member) as source,dest.open("wb") as output:
                shutil.copyfileobj(source,output)
    (META/"pv_original_download_manifest.json").write_text(json.dumps([row],ensure_ascii=False,indent=2),encoding="utf-8")
    log("original_verified",file=row["file"],bytes=row["bytes"],sha256=row["sha256"],source=url)

if __name__=="__main__":
    main()

