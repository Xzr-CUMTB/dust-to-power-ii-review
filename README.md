# Midnight commitments for photovoltaic generation

This archive contains executable inference, midnight plan selection, physical dispatch and table calculations for a 66.6 MW four-station photovoltaic cluster. The scientific programs read the included cases offline. `tests/verify_release.py` checks the payload hashes before executing the four main programs. It uses an isolated temporary directory for compiled kernels.

```sh
python -B tests/verify_release.py
python -B scripts/auxiliary_tables.py
```

Use Python 3.9 in the frozen environment specified in [environment/frozen.json](environment/frozen.json). The reference calculations use NumPy, pandas, LightGBM, Numba and PyTorch. CPU inference is sufficient. The original training used an NVIDIA RTX 5070 Ti and PyTorch 2.7.1 with CUDA 12.8. Optional source preparation also uses pvlib, pyarrow and openpyxl.

The main example issues a plan at local midnight, draws 32 trajectories for four sites at 96 intervals and estimates consequences for 91 plans. It uses observed generation to settle the plan under simulated battery and reserve operation. The evidence package retains a separate link between inputs and each figure through [method_to_code.md](docs/method_to_code.md). Read the [reproduction guide](docs/reproduction.md) for the executable scope, [data dictionary](docs/data_dictionary.md) for array semantics and [source notices](THIRD_PARTY_NOTICES.md) for data access and attribution.

The revised O95 comparison yielded 2,597.16 MWh of extra delivery relative to the same-version historical reference over 53 previously examined dates. The forecast-only contrast with the weather-matched tree yielded 246.22 MWh. The original frozen comparison, hard-budget experiments and sharing intervention each have their own baseline and resource conditions.
