"""Download and inspect a bounded CNEMC-derived hourly air-quality sample.

Sources are retained unchanged. Releases of station metadata may be newer than
historical observations: site-code coverage is checked rather than assumed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).absolute().parents[2]
RAW = ROOT / "data/raw/air"
META = ROOT / "data/metadata"
LOG = ROOT / "logs/download_air.jsonl"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def checksum(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid(path, kind):
    head = path.read_bytes()[:2048]
    if kind == "csv":
        if b"<html" in head.lower() or b"<!doctype" in head.lower():
            raise ValueError("HTML received instead of CSV")
        df = pd.read_csv(path, nrows=4, encoding="utf-8-sig")
        if not {"date", "hour", "type"}.issubset(df.columns):
            raise ValueError(f"Unexpected CSV fields: {df.columns.tolist()}")
    elif kind == "xlsx":
        if not head.startswith(b"PK"):
            raise ValueError("Not an XLSX ZIP file")
        pd.read_excel(path, nrows=3)
    elif kind == "json":
        json.loads(path.read_text(encoding="utf-8"))
    elif kind == "pdf":
        if not head.startswith(b"%PDF-"):
            raise ValueError("Not a PDF document")
        import fitz
        with fitz.open(path) as document:
            if not document.page_count:
                raise ValueError("PDF has no pages")


def download(url, destination, kind, expected_md5=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    started = utc_now()
    if destination.exists():
        valid(destination, kind)
        if expected_md5 and checksum(destination, "md5") != expected_md5:
            raise ValueError("Existing file checksum mismatch")
        manifest_path = META / "air_download_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded = next((r for r in previous if r.get("url") == url and r.get("sha256")), None)
            if recorded and recorded["sha256"] != checksum(destination):
                raise ValueError("Existing file differs from recorded SHA256")
        return {"url": url, "path": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size,
                "sha256": checksum(destination), "verified_utc": utc_now(), "status": "verified_existing"}
    last_error = None
    for attempt in range(1, 4):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "DustToPowerII-research-data-intake/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Inconsistent Content-Range")
                    mode = "ab"
                else:
                    offset, mode = 0, "wb"
                received = 0
                with part.open(mode) as fh:
                    for chunk in response.iter_content(1024 * 256):
                        if chunk:
                            fh.write(chunk)
                            received += len(chunk)
                expected = response.headers.get("Content-Length")
                if expected and not response.headers.get("Content-Encoding") and received != int(expected):
                    raise ValueError(f"Length mismatch {received} != {expected}")
                # XLSX must retain its suffix for spreadsheet parser engine detection.
                if kind == "xlsx":
                    import openpyxl
                    with part.open("rb") as fh:
                        book = openpyxl.load_workbook(fh, read_only=True, data_only=True)
                        book.close()
                else:
                    valid(part, kind)
                if expected_md5 and checksum(part, "md5") != expected_md5:
                    raise ValueError("Published MD5 mismatch")
                part.replace(destination)
                result = {"url": url, "resolved_url": response.url, "path": str(destination.relative_to(ROOT)),
                          "bytes": destination.stat().st_size, "sha256": checksum(destination),
                          "published_md5": expected_md5, "content_type": response.headers.get("Content-Type"),
                          "last_modified": response.headers.get("Last-Modified"), "started_utc": started,
                          "verified_utc": utc_now(), "status": "downloaded_verified", "attempt": attempt}
                return result
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            # Tiny HTML error bodies cannot be usefully resumed as real data.
            if part.exists() and part.stat().st_size < 65536:
                part.unlink()
            if attempt < 3:
                time.sleep(attempt * 2)
    return {"url": url, "path": str(destination.relative_to(ROOT)), "started_utc": started,
            "finished_utc": utc_now(), "status": "failed", "error": last_error}


def inspect_daily(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    pollutants = frame[frame["type"].isin(["PM10", "PM2.5"])].copy()
    stations = [c for c in frame.columns if c not in ("date", "hour", "type")]
    values = pollutants[stations].apply(pd.to_numeric, errors="coerce")
    result = {"path": str(path.relative_to(ROOT)), "rows": len(frame), "columns": len(frame.columns),
              "station_count": len(stations), "types": sorted(frame["type"].unique().tolist()),
              "dates": sorted(frame["date"].unique().tolist()),
              "hour_min": int(frame["hour"].min()), "hour_max": int(frame["hour"].max()),
              "pm_rows": len(pollutants), "pm_missing_fraction": float(values.isna().to_numpy().mean()),
              "pm_negative_cells": int((values < 0).sum().sum()),
              "duplicate_date_hour_type": int(frame.duplicated(["date", "hour", "type"]).sum()),
              "first_station_codes": stations[:12], "timestamp_interpretation": "Asia/Shanghai candidate; reporting latency not documented",
              "unit_pm": "microgram/m3 per archive documentation"}
    if len(pollutants):
        sample_codes = stations[:5]
        sample = pollutants[["date", "hour", "type"] + sample_codes].head(20)
        sample.to_csv(META / f"air_sample_{result['dates'][0]}.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="*", default=["20190403", "20190404", "20190405", "20190406",
                                                       "20190517", "20190518", "20190519", "20190520"])
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-sites", action="store_true")
    args = parser.parse_args()
    for directory in (RAW, META, LOG.parent):
        directory.mkdir(parents=True, exist_ok=True)
    dates = list(args.dates)
    if args.start and args.end:
        first, last = date.fromisoformat(args.start), date.fromisoformat(args.end)
        if last < first:
            parser.error("end precedes start")
        dates = [(first + timedelta(days=i)).strftime("%Y%m%d") for i in range((last-first).days+1)]
    jobs = [(f"https://quotsoft.net/air/data/china_sites_{d}.csv", RAW / f"china_sites_{d}.csv", "csv", None) for d in dates]
    if not args.skip_sites:
        jobs += [("https://zenodo.org/records/7152703/files/site_data.xlsx?download=1", RAW / "site_data_zenodo_7152703.xlsx", "xlsx", "56e1abfe2738804ebb8a9ca8979cb09a"),
                 ("https://zenodo.org/api/records/7152703", RAW / "zenodo_7152703_record.json", "json", None)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, 6))) as pool:
        results = list(pool.map(lambda job: download(*job), jobs))
    with LOG.open("a", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    # Keep the complete source register across subset runs and later recovery.
    register = {}
    for line in LOG.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        url = item["url"]
        register.setdefault(url, {}).update(item)
    (META / "air_download_manifest.json").write_text(json.dumps(list(register.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    audits = [inspect_daily(RAW / f"china_sites_{d}.csv") for d in dates if (RAW / f"china_sites_{d}.csv").exists()]
    (META / "air_daily_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    site_path = RAW / "site_data_zenodo_7152703.xlsx"
    if site_path.exists():
        site = pd.read_excel(site_path)
        site.to_csv(META / "air_station_metadata_raw.csv", index=False, encoding="utf-8-sig")
        station_info = {"rows": len(site), "columns": site.columns.tolist(), "sample": site.head(6).fillna("").to_dict(orient="records"),
                        "source": "https://zenodo.org/records/7152703", "published_md5_verified": checksum(site_path, "md5") == "56e1abfe2738804ebb8a9ca8979cb09a"}
        (META / "air_station_schema.json").write_text(json.dumps(station_info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"downloaded_or_verified": sum(r["status"] != "failed" for r in results), "failed": [r for r in results if r["status"] == "failed"],
                      "bytes": sum(r.get("bytes", 0) for r in results), "daily_files_audited": len(audits),
                      "first_date": min((x["dates"][0] for x in audits), default=None),
                      "last_date": max((x["dates"][-1] for x in audits), default=None),
                      "audit_file": str(META / "air_daily_audit.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
