# Data contract

No raw data are bundled. All NPZ files use numeric arrays and `allow_pickle=False`.

## Weather window

A forecast at time `t` uses exactly 24 observations from `t-6h` through `t-15min`, sampled every 15 minutes. Flatten oldest-to-latest to `(n,360)` after encoding 11 physical weather variables as 15 channels: four wind speeds (hub, 50m, 30m, 10m), temperature, pressure, humidity, and sine/cosine pairs for the four wind directions. Current/future target weather does not belong in a predictor window.

`preprocessing.py` contains the exact formal-study helpers for weather-tail thresholds (training-period wind 98th percentile, temperature 2nd/98th percentiles, nominal 25m/s cutout), boundary purging, fit-only imputation/scaling, and checks for shared original observations. Domain labels describe weather tails at forecast target times. The normal source and weather-tail target are both supervised in the fitting period.

## Fit archive

The `fit` command expects:

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `Xfit` | `(n,360)` float | Finite weather windows, imputed/scaled using the owning fitting population |
| `yfit` | `(n,)` float | Standardized fitting power labels |
| `n_source` | scalar integer | Number of leading source rows |
| `n_target` | scalar integer | Number of following target rows; counts sum to `n` |
| `label_mean` | scalar | Fitting-label mean in MW |
| `label_scale` | positive scalar | Fitting-label population SD in MW |

Additional split-identity/audit fields may be retained in the fitting archive but are not needed by the core solver. Fit/inference arrays must use the same 360-coordinate imputer/scaler. Do not fit transformations to calibration or assessment rows. The paper's explicit historical exception is preserved in `configs/formal_protocol.json`: originally missing evaluation coordinates retain the original 2019 fitting imputation before complete-2019 rescaling.

The source/target count fields in selected configs record the exact main-study populations. The CLI does not silently resample, truncate or force a new dataset to those sizes. For a different fitting population the effective regularization and relative kernel bandwidth are instantiated from that population; this is a new fit, not exact paper reproduction.

## Inference archive

Exactly one key: `X`, a finite `(n_eval,360)` array standardized with the fitting transform. Labels are rejected. The command saves raw nine-head quantiles, rearranged integrated mean, rearranged median, original tau=0.5 head, and nominal-95% endpoint pair, in MW after reversing the fitting-label standardization.

## Assessment archive

The separate `score` command expects `y` with shape `(n_eval,)` in MW and a positive scalar `capacity_mw`. It reads no fitting or tuning files and cannot change the saved configuration/model.

## Chronology

The four development folds, Q2 calibration and H2 retrospective assessment are recorded in `configs/formal_protocol.json`. Training/validation disjointness is required within each split; expanding fitting populations across development folds naturally overlap. Purge raw observation IDs across split boundaries, not merely row indices of overlapping windows. `audit_arrays` checks these IDs when supplied.
