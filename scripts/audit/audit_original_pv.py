"""Audit source PVOD records and create masked 15-minute grids; never impute power.

Source timestamp semantics and units were checked against the PVOD publication:
date_time UTC, power MW, metadata Capacity kW. Duplicate timestamp groups are
excluded in full from the gridded values and retained in source-record traces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def intervals(mask: np.ndarray, index: pd.DatetimeIndex, station: str) -> list[dict]:
    positions = np.flatnonzero(mask)
    if not len(positions):
        return []
    groups = np.split(positions, np.where(np.diff(positions) != 1)[0] + 1)
    return [{'station_id': station, 'start_utc': str(index[g[0]]),
             'end_utc': str(index[g[-1]]), 'slots': int(len(g)),
             'duration_hours': len(g) / 4} for g in groups]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).absolute().parents[2])
    args = parser.parse_args()
    root = args.root.absolute()
    source_dir = root / 'data/raw/pv_original/PVODdatasets_v1.0'
    output_dir = root / 'data/processed'
    audit_dir = root / 'data/metadata'
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = source_dir / 'metadata.csv'
    metadata_hash = sha256(metadata_path)
    station_metadata = pd.read_csv(metadata_path).set_index('Station_ID')
    station_ids = ['00', '04', '07', '08']
    paper_capacity = {'00': 6.6, '04': 20.0, '07': 20.0, '08': 20.0}
    frames, source_frames, duplicates, missing_slots, missing_intervals = [], [], [], [], []
    excluded_sources, station_audits, duplicate_groups = [], [], []

    for station_id in station_ids:
        source_path = source_dir / f'station{station_id}.csv'
        source_hash = sha256(source_path)
        source = pd.read_csv(source_path)
        native_columns = list(source)
        timestamps = pd.to_datetime(source['date_time'], errors='coerce', utc=True)
        valid_timestamp = timestamps.notna()
        aligned_timestamp = (timestamps.dt.minute.mod(15).eq(0)
                             & timestamps.dt.second.eq(0)
                             & timestamps.dt.microsecond.eq(0))
        duplicated_timestamp = timestamps.duplicated(keep=False) & valid_timestamp
        source['station_id'] = station_id
        source['source_record_index'] = np.arange(len(source), dtype=np.int64)
        source['source_file'] = source_path.relative_to(root).as_posix()
        source['source_sha256'] = source_hash
        source['timestamp_utc'] = timestamps
        source['source_timestamp_valid'] = valid_timestamp
        source['source_timestamp_on_15min_grid'] = aligned_timestamp
        source['source_timestamp_duplicated'] = duplicated_timestamp
        source_frames.append(source.copy())
        duplicates.append(source.loc[duplicated_timestamp].copy())
        excluded_sources.append(source.loc[~valid_timestamp | ~aligned_timestamp].copy())
        for timestamp, group in source.loc[duplicated_timestamp].groupby('timestamp_utc'):
            differing = [column for column in native_columns
                         if group[column].nunique(dropna=False) > 1]
            duplicate_groups.append({
                'station_id': station_id, 'timestamp_utc': str(timestamp),
                'source_record_indices': group.source_record_index.tolist(),
                'source_record_count': len(group), 'differing_columns': differing,
                'power_values_mw': [float(x) if np.isfinite(x) else None
                                    for x in pd.to_numeric(group.power, errors='coerce')],
                'action': 'exclude_all_rows_at_timestamp_from_gridded_values',
            })

        grid_times = pd.date_range(timestamps[valid_timestamp].min().floor('15min'),
                                   timestamps[valid_timestamp].max().ceil('15min'), freq='15min')
        source_counts = timestamps[valid_timestamp & aligned_timestamp].value_counts()
        count = source_counts.reindex(grid_times, fill_value=0).to_numpy(dtype=np.int64)
        retained = source.loc[valid_timestamp & aligned_timestamp & ~duplicated_timestamp].copy()
        retained = retained.set_index('timestamp_utc')
        assert retained.index.is_unique
        grid = retained[native_columns + ['source_record_index']].reindex(grid_times)
        grid.index.name = 'timestamp_utc'
        power = pd.to_numeric(grid['power'], errors='coerce').to_numpy(dtype=float)
        row_metadata = station_metadata.loc[f'station{station_id}']
        capacity_kw = float(row_metadata.Capacity)
        capacity_mw = capacity_kw / 1000
        longitude, latitude = float(row_metadata.Longitude), float(row_metadata.Latitude)
        if capacity_mw != paper_capacity[station_id]:
            raise ValueError(f'Original and paper capacities differ for {station_id}')
        solar = pvlib.solarposition.get_solarposition(grid_times, latitude, longitude)
        elevation = solar.elevation.to_numpy()
        finite = np.isfinite(power)
        value_valid = (count == 1) & finite & (power >= 0)
        output = pd.DataFrame({
            'timestamp_utc': grid_times,
            'timestamp_local': grid_times.tz_convert('Asia/Shanghai'),
            'station_id': station_id,
            'power_mw': power,
            'paper_capacity_mw': paper_capacity[station_id],
            'power_pu_paper_capacity': power / paper_capacity[station_id],
            'metadata_capacity_kw': capacity_kw,
            'metadata_capacity_mw': capacity_mw,
            'solar_elevation_degrees': elevation,
            'longitude': longitude, 'latitude': latitude,
            'source_record_present': count > 0,
            'source_unique_record_present': count == 1,
            'source_record_count': count,
            'excluded_duplicate_timestamp': count > 1,
            'source_value_present': finite,
            'value_valid': value_valid,
            'diagnostic_power_valid': value_valid,
            'source_imputation_status_known': False,
            'source_original_observation_confirmed': False,
            'power_above_metadata_capacity': finite & (power > capacity_mw),
            'power_above_105pct_metadata_capacity': finite & (power > capacity_mw * 1.05),
            'power_negative': finite & (power < 0),
            'metadata_source': metadata_path.relative_to(root).as_posix(),
            'source_file': source_path.relative_to(root).as_posix(),
            'source_sha256': source_hash,
            'source_record_index': pd.array(grid.source_record_index, dtype='Int64'),
        })
        for column in native_columns:
            if column.startswith('lmd_') or column.startswith('nwp_'):
                output[column] = pd.to_numeric(grid[column], errors='coerce').to_numpy()
        output['source_value_status'] = np.select(
            [count == 0, count > 1, ~finite, power < 0],
            ['missing_source_slot', 'duplicate_timestamp_excluded',
             'nonfinite_power', 'negative_power'], default='source_value_available')
        output['night_positive_power_flag'] = ((elevation < -6)
                                               & (power > capacity_mw * 0.01))
        frames.append(output)
        no_source = count == 0
        missing_slots.append(output.loc[no_source, [
            'station_id', 'timestamp_utc', 'timestamp_local', 'source_record_count',
            'source_value_status']].copy())
        missing_intervals.extend(intervals(no_source, grid_times, station_id))
        raw_power = pd.to_numeric(source['power'], errors='coerce').to_numpy(dtype=float)
        numeric_lmd = pd.to_numeric(output['lmd_totalirrad'], errors='coerce').to_numpy()
        night = elevation < -6
        mistaken_local = grid_times.tz_localize(None).tz_localize('Asia/Shanghai')
        mistaken_elevation = pvlib.solarposition.get_solarposition(
            mistaken_local, latitude, longitude).elevation.to_numpy()
        station_audits.append({
            'station_id': station_id,
            'source_file': source_path.relative_to(root).as_posix(),
            'source_sha256': source_hash,
            'source_rows': len(source), 'source_columns': native_columns,
            'source_unique_timestamps': int(timestamps.nunique()),
            'invalid_timestamp_rows': int((~valid_timestamp).sum()),
            'off_grid_timestamp_rows': int((valid_timestamp & ~aligned_timestamp).sum()),
            'source_time_ordered': bool(timestamps.is_monotonic_increasing),
            'duplicate_timestamp_rows_all_members': int(duplicated_timestamp.sum()),
            'duplicate_timestamp_groups': int(timestamps[duplicated_timestamp].nunique()),
            'duplicate_rows_beyond_first': int((valid_timestamp & timestamps.duplicated()).sum()),
            'duplicate_policy': 'exclude_entire_timestamp_group',
            'grid_slots': len(grid_times), 'grid_missing_source_slots': int(no_source.sum()),
            'grid_duplicate_excluded_slots': int((count > 1).sum()),
            'grid_no_retained_unique_source_slots': int((count != 1).sum()),
            'grid_start_utc': str(grid_times.min()), 'grid_end_utc': str(grid_times.max()),
            'grid_start_local': str(grid_times.min().tz_convert('Asia/Shanghai')),
            'grid_end_local': str(grid_times.max().tz_convert('Asia/Shanghai')),
            'metadata_source': metadata_path.relative_to(root).as_posix(),
            'metadata_sha256': metadata_hash,
            'metadata_capacity_kw': capacity_kw, 'metadata_capacity_mw': capacity_mw,
            'paper_capacity_mw': paper_capacity[station_id],
            'longitude': longitude, 'latitude': latitude,
            'source_power_min_mw': float(np.nanmin(raw_power)),
            'source_power_max_mw': float(np.nanmax(raw_power)),
            'source_power_nonfinite_rows': int((~np.isfinite(raw_power)).sum()),
            'source_power_negative_rows': int((raw_power < 0).sum()),
            'grid_value_valid_slots': int(value_valid.sum()),
            'grid_value_valid_hours': float(value_valid.sum() / 4),
            'grid_over_metadata_capacity_slots': int(output.power_above_metadata_capacity.sum()),
            'grid_over_105pct_metadata_capacity_slots': int(output.power_above_105pct_metadata_capacity.sum()),
            'capacity_action': 'flag_only_no_clipping_or_exclusion',
            'night_definition_solar_elevation_degrees_below': -6.0,
            'night_positive_power_threshold_mw': capacity_mw * 0.01,
            'night_any_positive_power_slots': int(np.sum(night & (power > 0))),
            'night_above_1pct_capacity_power_slots': int(output.night_positive_power_flag.sum()),
            'night_irradiance_over50_slots_UTC_interpretation': int(np.sum(night & (numeric_lmd > 50))),
            'night_irradiance_over50_slots_wrong_local_interpretation': int(np.sum((mistaken_elevation < -6) & (numeric_lmd > 50))),
            'source_imputation_status_known': False,
            'source_record_semantics': 'presence_in_published_original_CSV_not_sensor_level_rawness',
            'power_imputed_by_this_pipeline': False,
            'missing_per_native_column': {k: int(v) for k, v in source[native_columns].isna().sum().items()},
        })
        # Conservation and duplicate rejection are ingestion invariants, not model tests.
        assert int(count.sum()) == int((valid_timestamp & aligned_timestamp).sum())
        assert output.loc[output.excluded_duplicate_timestamp, 'power_mw'].isna().all()
        assert not output.loc[output.excluded_duplicate_timestamp, 'diagnostic_power_valid'].any()
        assert output.loc[~output.source_record_present, 'power_mw'].isna().all()
        assert sha256(source_path) == source_hash

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(output_dir / 'pv_four_stations_original.parquet', index=False)
    raw_records = pd.concat(source_frames, ignore_index=True)
    raw_records.to_parquet(output_dir / 'pv_original_source_records.parquet', index=False)
    pd.concat(duplicates, ignore_index=True).to_csv(
        audit_dir / 'pv_original_duplicate_records.csv', index=False, encoding='utf-8-sig')
    pd.concat(excluded_sources, ignore_index=True).to_csv(
        audit_dir / 'pv_original_invalid_timestamp_records.csv', index=False, encoding='utf-8-sig')
    pd.concat(missing_slots, ignore_index=True).to_csv(
        audit_dir / 'pv_original_missing_slots.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(missing_intervals, columns=['station_id', 'start_utc', 'end_utc', 'slots',
                                          'duration_hours']).to_csv(
        audit_dir / 'pv_original_missing_intervals.csv', index=False, encoding='utf-8-sig')
    wide_valid = combined.pivot(index='timestamp_utc', columns='station_id',
                                values='diagnostic_power_valid').astype('boolean').fillna(False).astype(bool)
    jointly_valid = wide_valid[station_ids].all(axis=1)
    valid_times = wide_valid.index[jointly_valid]
    jointly_present = combined.pivot(index='timestamp_utc', columns='station_id',
                                     values='source_record_present').astype('boolean').fillna(False).astype(bool).all(axis=1)
    audit = {
        'dataset': 'PVOD v1.0 published original CSVs',
        'station_ids': station_ids,
        'source_timestamp_column': 'date_time', 'source_timestamp_timezone': 'UTC',
        'power_unit': 'MW', 'metadata_capacity_unit': 'kW',
        'observation_status_caveat': 'Original-CSV row presence establishes source provenance; no sensor-level original/imputation flag is supplied.',
        'value_valid_definition': 'unique source timestamp, finite power, and power >= 0; capacity exceedance flagged only',
        'source_record_present_definition': 'one or more source CSV records at this slot, including excluded duplicate groups',
        'station_audits': station_audits, 'duplicate_groups': duplicate_groups,
        'joint': {
            'grid_rows_total': len(combined), 'raw_source_records_total': len(raw_records),
            'common_source_present_slots': int(jointly_present.sum()),
            'common_value_valid_slots': int(jointly_valid.sum()),
            'common_value_valid_hours': int(jointly_valid.sum()) / 4,
            'common_valid_start_utc': str(valid_times.min()) if len(valid_times) else None,
            'common_valid_end_utc': str(valid_times.max()) if len(valid_times) else None,
            'common_valid_start_local': str(valid_times.min().tz_convert('Asia/Shanghai')) if len(valid_times) else None,
            'common_valid_end_local': str(valid_times.max().tz_convert('Asia/Shanghai')) if len(valid_times) else None,
            'hours_semantics': 'count of valid 15-minute slots / 4; not elapsed span',
        },
        'outputs': {
            'gridded': 'data/processed/pv_four_stations_original.parquet',
            'all_native_records': 'data/processed/pv_original_source_records.parquet',
            'duplicates': 'data/metadata/pv_original_duplicate_records.csv',
            'missing_slots': 'data/metadata/pv_original_missing_slots.csv',
            'missing_intervals': 'data/metadata/pv_original_missing_intervals.csv',
        },
        'power_imputed': False, 'capacity_exceedance_clipped': False,
        'models_evaluated': False,
    }
    write_json(audit_dir / 'pv_original_quality_audit.json', audit)
    pd.DataFrame([{k: v for k, v in item.items() if not isinstance(v, (dict, list))}
                  for item in station_audits]).to_csv(
        audit_dir / 'pv_original_quality_audit.csv', index=False, encoding='utf-8-sig')
    assert sha256(metadata_path) == metadata_hash
    print(json.dumps({'stations': [{k: row[k] for k in [
        'station_id', 'source_rows', 'grid_slots', 'grid_missing_source_slots',
        'grid_duplicate_excluded_slots', 'grid_value_valid_slots', 'source_power_max_mw',
        'grid_over_metadata_capacity_slots', 'night_above_1pct_capacity_power_slots']}
        for row in station_audits], 'joint': audit['joint'],
        'duplicate_groups': duplicate_groups}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
