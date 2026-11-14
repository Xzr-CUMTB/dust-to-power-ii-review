# Reproduction

## Frozen reference case

From the root of the private checkout or extracted supplement, run:

~~~sh
python -B tests/verify_release.py
python -B scripts/auxiliary_tables.py
~~~

The first command verifies the checked-in source, model and data SHA-256s, reloads the CPU models and runs numerical tests. Numba first compiles the original and revised dispatch kernels in a temporary cache. No login, remote request, training or full-system optimization is needed. Exact versions are in environment/frozen.json. The optional data-preparation scripts use the normal pytorch conda interpreter with pvlib available; the reference CPU tests also work without user-site packages.

1. **Generation and proxies.** The frozen early-fold M3 weights are applied to issue-time features for 18 April 2019. Conditional donor quantiles and block tilts are converted to 32 paths of 96 quarter-hours at four sites. Paths retain paired station observations and are normalized to the learned energy. The saved dated development residuals implement the same mature energy calibration used in the operating comparison. The script integrates each calibrated member under the 91 candidate plans, reproducing the archived candidate means for gap, reserve and throughput. Thirteen fixed LightGBM quantile models are also reloaded against the same-day benchmark.
2. **Forecast-to-plan check.** The revised E0 weights are loaded separately for D0, M1 and M3. A CPU pass reconstructs conditional mean and scale for each of 53 × 91 candidate sets. The program chronologically reissues every selected action from the archived candidate inputs. Each selected index is checked before its own simulated outcomes are accepted for later feedback. Outcomes of a day enter the policy two midnight issues later. An alternative candidate would require new physical outcomes; the check stops if the index differs.
3. **Timed dispatch.** The ordinary and hard-budget R3 profiles replay all 96 quarter-hours of an existing plan on observed generation, with original and revised executors independently compared to their saved action arrays. Tests check bus balance, reserve accounting, SOC transitions and terminal inventory. Both numeric near-tie regression cases also replay through the independent precision check. The revised sharing contrast used the same deterministic tie rule. The two near-tie files are numerical regression cases.
4. **Tables.** From 5,100 archived method-day records, the table command reproduces 600 stored metric/interval rows across versions and regimes and outputs the O95 results. It draws 2,000 multinomial samples of common calendar weeks ending Sunday, normalizing each draw to the original count of paired days. Delivery, shortfall, reserve and AC throughput retain separate columns. The sharing contrasts use 36 common days. The historical 91-action opportunity ratio uses the original executor and matched offline bound. Source registration and the repeated-seed comparison retain their original scope.
5. **Auxiliary tasks.** The second command reduces the weather-window experiment (two inputs on 53 matched days), hourly combination (21 Hong Kong meters) and native point-power comparison (119 Utrecht households). The accompanying tables preserve short-horizon joint-risk scores and particulate sensitivity.

## Original source acquisition

Use a separate working directory for the original public archives. PVOD v1.0 on Science Data Bank uses listing version V5 (structured metadata version 5.0.0), zip file PVODdatasets_v1.0.zip, 7,983,824 bytes, SHA-256 0f726b42235bf99a2c118a3cb7544722150489b01af3a2853a7c9778b538883e. The download script reads the registered public file listing; alternatively obtain the ZIP at [Science Data Bank](https://doi.org/10.11922/sciencedb.01094). The following command checks the hash, extracts station CSVs into a fresh directory, builds the masked PV table and retains diagnostic traces.

~~~sh
python -B scripts/prepare_data.py --pvod PVODdatasets_v1.0.zip --workspace scratch
~~~

The original spatial audit uses PVOD metadata and a matching AI-PVOD station-coordinate release. [AI-PVOD](https://huggingface.co/datasets/yujiaA/AI-PVOD/tree/8e2919884afd7e79c89d30be895a103d8f6ab218). AIweatherdata.zip has SHA-256 e12938e84da9f5df5e7c0345647d74c257af729816b708a89854324c5f76ce89. Hourly PM files are available at the daily URLs listed in configs/air_source_files.csv. Monitor metadata sheet site_data.xlsx is archived at Zenodo 7152703, MD5 56e1abfe2738804ebb8a9ca8979cb09a. scripts/download/air_download.py accepts --start and --end dates. scripts/download/air_spatial_audit.py checks the coordinate axes, finds nearest monitors and merges their PM observations. With the resulting merged file:

~~~sh
python -B scripts/prepare_data.py --monitor-hourly air_pv_nearby_hourly.csv --workspace scratch
~~~

The publication-clock sensitivity uses hour-ending observations and an assumed release delay. Source revision histories and actual station receipt times are unavailable. The historical midnight pipeline used for the study is in scripts/p9/intake.py; the 3-hour setting is its default. In a populated archival layout this program reads the processed PV, monitor table and air_pv_monitor_distances.csv, checks all 96 target intervals jointly and builds day_interface.npz. The program takes its root from its own location. 

The weather archive inventory records run, valid interval, file mtime, midnight issue, archive URL, actual NetCDF subset URL and the local subset SHA-256. The checksum is for the downloaded subset, not for the much larger original GRIB2 file. The eligible runs were issued 64 hours before midnight. The four field windows have forecast endpoint leads of 72, 78, 84 and 90 hours. An archive last-modified date serves as a proxy for prior availability.

The simulated action/forecast feature fixtures come from the reported calculations. Source PV records use the CC BY attribution in THIRD_PARTY_NOTICES.md. The national monitor archives and field-by-field raw weather grids are not included. The reference calculations read the compact, derived inputs. Archived routines under archive_recipes preserve the key computations at their historical stages, including candidate formation, M1 and M3 training, station adaptation, the original finite-candidate optimization and the symmetric sharing intervention. They reference the full research archive’s intermediate tables, weights and import structure. They are provided for code inspection, separate from the standalone commands above. Full training, retrieval of every source archive and multi-site exhaustive dispatch were not repeated for this release.

The current release has no assigned software DOI. A private checkout has access controls. For review the same numerical payload is provided as a separate anonymous archive.
