# Methods and numerical entry points

| Quantity | Executable or source | Comparison and interpretation |
|---|---|---|
| PV metering, QC and day boundaries | scripts/prepare_data.py, scripts/audit/audit_original_pv.py, scripts/p9/intake.py | UTC source stamps, local power intervals, duplicate masks, full-day finite observations, original capacity exceedances. |
| Four weather intervals, donor ranking, scenario shape and energy | src/scenarios/donors.py, src/models/p11/quantile.py, scripts/run_inference.py | Weather eligibility precedes training and retrieval. The one-day checkpoint example uses the early fold, round 2, seed 42. |
| Historical reference, matched quantile tree | src/scenarios/donors.py, archive_recipes/p12/inputs12.py, checkpoints/M1 | Distinct historical donor baseline and weather-matched tree benchmark. The tree is reloaded for the same example as M3. |
| Candidate consequence proxies and probability adjustment | src/operations/p10/budget.py, src/scenarios/donors.py, scripts/run_inference.py | Dated development energy residuals calibrate paths; candidate-specific component means are computed from 32 trajectories. |
| E0 and feedback | src/models/p15/readout.py, src/models/p15/fit.py, src/dispatch/p15/decision.py, scripts/run_replay.py | Three small residual heads are fitted separately by forecast source. F0 learns after settlement from the selected plan. |
| R3 and the H supervisor | src/planning/p9/dispatch.py, src/dispatch/p16/supervisor.py, scripts/p18_close/revised.py | Original frozen and revised executors share physical projections and retain distinct numerical choice rules. |
| Physical feasibility and invariance | scripts/run_replay.py, scripts/p18_close/verify_minimal.py | 96 states and actions per case, initial/terminal SOC, AC bus and battery transitions, cap and reservation checks. Two archived near ties exercise the selector. |
| Weather / forecast score, delivery, matching, weekly intervals | scripts/reproduce_tables.py, scripts/auxiliary_tables.py | F03 and F04 represent distinct calibrations. F05 and F06 use revised E0 rows. F08 uses the original repeated fits, with shared seed 42. |
| H budget, original opportunity, external adaptation | reference_results/daily_energy.csv, finite_candidate_bound.csv, original_opportunity_pairs.csv, archive_recipes/opportunity/run.py | F09 and F10 use original execution. For F11 the reissued California system and frozen external adaptation have separate rows. |
| Resource sharing and cross-site pairing | reference_results/sharing_daily.csv, scripts/reproduce_tables.py, archive_recipes/p18_close/sharing_replay.py | F12 compares fixed plans in four arms under the revised selector. |
| Exposure and external prediction | scripts/auxiliary_tables.py, reference_results | F14–F16 concern shorter-horizon risk, PM contrasts or native power prediction; these rows do not enter the four-station O95 denominator. |
