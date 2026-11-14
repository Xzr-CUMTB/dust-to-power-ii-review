# Array and table semantics

| File or field | Shape and unit | Meaning |
|---|---|---|
| M3.npz | 32 × 96 × 4, MW | Masked historical donor power and frozen forecast for 18 April 2019. Local quarter-hours, site order 00, 04, 07, 08. |
| x, weather, anchors | vector, 4 × 11, 13 | Normalized issue context, forecast intervals and scaled donor quantile anchors. |
| issue_context.npz | date, development residuals, plan reference | Dated residuals are eligible through D−2. The historical reference is a within-day power shape. |
| candidates_{D0,M1,M3}.npz | 53 dates × 91 actions × 3 outcomes | Raw consequences, model inputs and archived expected and simulated outcomes. Consequence order is shortfall, reserve use, battery AC throughput, in MWh. |
| Candidate index j | 0–90 | Thirteen coefficients 0.30–0.90 in increments of 0.05 for each of seven daily reserve offers 10–25 MWh in increments of 2.5. |
| R3_O.npz and R3_H.npz | y[96], v[96,32,24], pl[96,24] | Realized PV in MW; near-term scenarios; 24-step local contract lookahead. R3 has separate inputs from the M3 midnight forecast. |
| gg, w, upper, haz | [96,6,32,24], [96,9], [96,24], [96] | Scenario gap inputs, stored value coefficients, upper near-term paths and risk input. |
| Action matrix | 96 × 13 for O; 96 × 19 for H | PV, plan, charging, discharging, reserve, shortfall, spill (MW), opening/closing SOC and reserve (MWh), two requested battery powers. H adds requested reserve, opening/closing throughput balance, charge ceiling and opening/closing nominal reserve. |
| daily_energy.csv | system × date × version × method × regime × target | Daily integration of each archived transaction. Monthly or cluster aggregates must preserve regime and version. |
| generation_scores_daily.csv | system, date, model, seed | Daily CRPS, 90% interval score and width, and 10% quantile loss in MWh. Coverage is binary. |
| weather_scores.csv | matched evaluation day and model | Weather-window analysis before the later online energy calibration used in the frozen forecast-score table. |
| original_opportunity_pairs.csv | system, original O95 | Same-executor online gain paired to the 91-candidate retrospective bound. |
| sharing_daily.csv | 36 common dates × 4 arms | Fixed 0.70 contract, isolated or ideal pooled bus and altered or retained cross-site pairing. |
| short_horizon_scores, particle_ablation, particle_delay | method, horizon or exposure block | Auxiliary scores and recorded intervals; their statistical units are those in the individual source columns. |
| HK_hourly, UT_households | meter/household | Auxiliary native forecast tasks; their common-weather units differ from the engineering system. |
