"""Inspect air monitor coordinates and build distance-only PV associations.

No monitor is called upstream from distance alone. Direction, wind and lag
information must be evaluated by subsequent experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[2]
META = ROOT / "data/metadata"


def haversine_km(lon, lat, longitudes, latitudes):
    lon1, lat1 = np.radians(lon), np.radians(lat)
    lon2, lat2 = np.radians(longitudes), np.radians(latitudes)
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    META.mkdir(parents=True, exist_ok=True)
    sites = pd.read_excel(ROOT / "data/raw/air/site_data_zenodo_7152703.xlsx")
    # The published headings are reversed. This is a dataset-level correction,
    # independently confirmed by archive documentation's 1001A example.
    coords = pd.DataFrame({"site_code": sites["Site code"].astype(str),
                           "longitude": pd.to_numeric(sites["Latitude"], errors="coerce"),
                           "latitude": pd.to_numeric(sites["Longitude"], errors="coerce"),
                           "raw_column_Latitude": sites["Latitude"], "raw_column_Longitude": sites["Longitude"],
                           "coordinate_transform": "longitude=raw.Latitude;latitude=raw.Longitude",
                           "source_doi": "10.5281/zenodo.7152703"})
    assert coords["site_code"].is_unique
    assert coords["longitude"].between(73, 136).all()
    assert coords["latitude"].between(18, 54).all()
    assert len(coords) == 1051
    coords.to_csv(META / "air_station_coordinates.csv", index=False)
    coordinate_audit = {"raw_rows": len(sites), "unique_sites": int(coords.site_code.nunique()),
                        "raw_Latitude_above_90": int((sites.Latitude > 90).sum()),
                        "corrected_longitude_range": [float(coords.longitude.min()), float(coords.longitude.max())],
                        "corrected_latitude_range": [float(coords.latitude.min()), float(coords.latitude.max())],
                        "evidence": "https://quotsoft.net/air/ station 1001A example: longitude116.366 latitude39.8673",
                        "scope": "1051-site subset; not the complete 1605-site observation network",
                        "metadata_historical_coordinate_validity": "station relocation/version dates unavailable"}
    (META / "air_coordinate_audit.json").write_text(json.dumps(coordinate_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    pvroot = ROOT / "data/raw/pv_ai_pvod/AIweatherdata/AIweatherdata/station_data"
    associations = []
    original_metadata_path = ROOT / "data/raw/pv_original/PVODdatasets_v1.0/metadata.csv"
    original_metadata = pd.read_csv(original_metadata_path).set_index("Station_ID") if original_metadata_path.exists() else None
    crosschecks = []
    for original in [0, 4, 7, 8]:
        path = pvroot / f"{original}.csv"
        pv = pd.read_csv(path, nrows=1)
        if pv.loc[0, "Location"] != f"station{original:02d}":
            raise ValueError(f"PV source station mismatch: {path}")
        lon, lat = float(pv.loc[0, "Longitude"]), float(pv.loc[0, "Latitude"])
        if original_metadata is not None:
            original_lon = float(original_metadata.loc[f"station{original:02d}", "Longitude"])
            original_lat = float(original_metadata.loc[f"station{original:02d}", "Latitude"])
            agree = bool(np.allclose([original_lon, original_lat], [lon, lat], rtol=0, atol=1e-8))
            if not agree:
                raise ValueError(f"PV coordinate source disagreement: station{original:02d}")
            crosschecks.append({"pv_station": f"{original:02d}", "original_longitude": original_lon,
                                "original_latitude": original_lat, "ai_longitude": lon, "ai_latitude": lat,
                                "coordinates_agree": agree})
        distances = haversine_km(lon, lat, coords.longitude.to_numpy(), coords.latitude.to_numpy())
        pairs = coords[["site_code", "longitude", "latitude"]].copy()
        pairs["pv_station"] = f"{original:02d}"
        pairs["pv_longitude"], pairs["pv_latitude"] = lon, lat
        pairs["distance_km"] = distances
        pairs = pairs.sort_values("distance_km").reset_index(drop=True)
        pairs["distance_rank"] = np.arange(1, len(pairs)+1)
        associations.append(pairs)
    if crosschecks:
        pd.DataFrame(crosschecks).to_csv(META / "air_pv_coordinate_crosscheck.csv", index=False)
    pairs = pd.concat(associations, ignore_index=True)
    pairs.to_csv(META / "air_pv_monitor_distances.csv", index=False)
    nearest = pairs[pairs.distance_rank <= 10].copy()
    nearest.to_csv(META / "air_pv_nearest10.csv", index=False)
    # Use only downloaded/validated files; do not inspect partial downloads.
    wanted = sorted(set(nearest.site_code))
    pieces, code_coverage = [], []
    for path in sorted((ROOT / "data/raw/air").glob("china_sites_*.csv")):
        fields = pd.read_csv(path, nrows=0).columns.tolist()
        found = [s for s in wanted if s in fields]
        code_coverage.append({"file": path.name, "observation_site_count": len(fields)-3,
                              "coordinate_matched_count": len(set(fields) & set(coords.site_code)),
                              "nearest10_codes_present": len(found), "nearest10_codes_total": len(wanted)})
        frame = pd.read_csv(path, usecols=["date", "hour", "type"] + found)
        frame = frame[frame.type.isin(["PM10", "PM2.5"])]
        frame = frame.melt(id_vars=["date", "hour", "type"], var_name="site_code", value_name="value")
        frame["time_bjt"] = (pd.to_datetime(frame["date"].astype(str), format="%Y%m%d") + pd.to_timedelta(frame.hour, unit="h")).dt.tz_localize("Asia/Shanghai")
        frame["unit"] = "ug/m3"
        pieces.append(frame)
    hourly = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    hourly.to_csv(META / "air_pv_nearby_hourly.csv", index=False)
    pd.DataFrame(code_coverage).to_csv(META / "air_station_code_coverage.csv", index=False)
    event_diagnostics = []
    if len(hourly):
        diag = hourly.groupby(["site_code", "type"]).value.agg(["count", "min", "median", "max"]).reset_index()
        diag.to_csv(META / "air_pv_nearby_summary.csv", index=False)
        event_path = META / "dust_event_candidates.csv"
        if event_path.exists():
            events = pd.read_csv(event_path)
            # This table describes exposure proxies only. It does not redefine
            # event membership, declare dust causality or use power/model errors.
            matched = hourly.merge(nearest[["site_code", "pv_station"]], on="site_code", how="inner")
            for _, event in events.iterrows():
                start = pd.Timestamp(event.start_date_bjt).tz_localize("Asia/Shanghai")
                end = pd.Timestamp(event.end_date_bjt_inclusive).tz_localize("Asia/Shanghai") + pd.Timedelta(days=1)
                sub = matched[(matched.time_bjt >= start) & (matched.time_bjt < end)]
                for pv_station in sorted(nearest.pv_station.unique()):
                    pm = sub[sub.pv_station == pv_station].pivot(index=["time_bjt", "site_code"], columns="type", values="value")
                    expected_hours = int((end-start).total_seconds()/3600)
                    if "PM10" in pm:
                        pm10 = pm["PM10"]
                        max_value = None if pm10.dropna().empty else float(pm10.max())
                        median_value = None if pm10.dropna().empty else float(pm10.median())
                        peak_time = None if pm10.dropna().empty else str(pm10.idxmax()[0])
                        available_hours = int(pm10.dropna().reset_index().time_bjt.nunique())
                        available_cells = int(pm10.notna().sum())
                    else:
                        max_value, median_value, peak_time, available_hours, available_cells = None, None, None, 0, 0
                    ratio = None
                    if {"PM10", "PM2.5"}.issubset(pm.columns):
                        ratios = (pm["PM10"] / pm["PM2.5"].where(pm["PM2.5"] > 0)).replace([np.inf, -np.inf], np.nan).dropna()
                        if len(ratios):
                            ratio = float(ratios.median())
                    event_diagnostics.append({"event_id": event.event_id, "pv_station": pv_station,
                                              "expected_hours": expected_hours, "valid_pm10_hours_any_nearby": available_hours,
                                              "nearby_pm10_valid_cells": available_cells,
                                              "nearby_pm10_expected_cells": expected_hours * 10,
                                              "pm10_max_ug_m3": max_value, "pm10_median_ug_m3": median_value,
                                              "pm10_peak_time_bjt": peak_time, "median_pm10_pm25_ratio": ratio,
                                              "hebei_mentioned": event.hebei_mentioned,
                                              "interpretation": "nearby_monitor_proxy;not_pv_site_dust_or_causal_label"})
            pd.DataFrame(event_diagnostics).to_csv(META / "dust_local_monitor_diagnostics.csv", index=False)
    daily_audits = json.loads((META / "air_daily_audit.json").read_text(encoding="utf-8"))
    coverage = pd.DataFrame(code_coverage)
    summary = {"days": len(daily_audits), "rows_total": sum(r["rows"] for r in daily_audits),
               "pm_rows_total": sum(r["pm_rows"] for r in daily_audits),
               "station_counts": sorted(set(r["station_count"] for r in daily_audits)),
               "hours_not_0_23": sum(r["hour_min"] != 0 or r["hour_max"] != 23 for r in daily_audits),
               "duplicate_keys": sum(r["duplicate_date_hour_type"] for r in daily_audits),
               "negative_pm_cells": sum(r["pm_negative_cells"] for r in daily_audits),
               "missing_fraction_minmax": [min(r["pm_missing_fraction"] for r in daily_audits), max(r["pm_missing_fraction"] for r in daily_audits)],
               "metadata_coordinate_match_minmax": [int(coverage.coordinate_matched_count.min()), int(coverage.coordinate_matched_count.max())],
               "nearby_unique_monitors": int(hourly.site_code.nunique()), "nearby_rows": len(hourly),
               "nearby_missing_fraction": float(hourly.value.isna().mean()),
               "time_first": str(hourly.time_bjt.min()), "time_last": str(hourly.time_bjt.max()),
               "event_rows": len(event_diagnostics),
               "event_full_pm10_hours": sum(r["expected_hours"] == r["valid_pm10_hours_any_nearby"] for r in event_diagnostics),
               "nearby_bytes": (META / "air_pv_nearby_hourly.csv").stat().st_size}
    (META / "air_intake_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"coordinate_audit": coordinate_audit, "nearest_monitor_by_station": nearest[nearest.distance_rank == 1].to_dict(orient="records"),
                      "daily_files": len(code_coverage), "nearby_hourly_rows": len(hourly)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
