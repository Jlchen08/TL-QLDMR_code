# TL-QLDMR

Core implementation and experiment configurations for **Transfer Learning Quantile Large-margin Distribution Machine for Regression (TL-QLDMR)** in weather-tail wind-power forecasting.

The model combines domain-weighted pinball loss, centered source-residual variance, and projected source–target alignment in a fixed Nyström RBF representation. The structured primal–dual interior-point solver returns an original primal/feasible-dual gap certificate for each quantile head. The source-residual centering retains its label-dependent linear term.

## Install and check

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/smoke.py
```

Python 3.10+ is required. The core uses NumPy, SciPy and scikit-learn on CPU; no GPU or PyTorch is required for this implementation. The synthetic smoke check verifies the centered objective, domain loss masses, all nine solver certificates and model save/load. It does not reproduce the six-farm paper results.

## Included

- `src/tlqldmr/kernel_model.py`: fitted Nyström map, objective assembly, nine-head fitting and model state.
- `src/tlqldmr/structured_ipm.py`: structured Mehrotra primal–dual solver.
- `src/tlqldmr/weather_representation.py`: history (360), recent (90), mixed (450) and last-lag (15) covariate representations.
- `src/tlqldmr/preprocessing.py`: fit-only imputation/scaling, weather-tail labels, temporal purging and raw-measurement overlap checks extracted from the formal study.
- `src/tlqldmr/metrics.py`: normalized point/interval scores, native-nine-head WIS and nonnegative finite-rank calibration expansion.
- `configs/selected_tl/`: 18 recorded TL-QLDMR farm/task configurations (six farms × point/95% interval/distribution), with final fitting seeds 0, 1 and 2.
- `configs/kernel_search.json` and `configs/baseline_search.json`: registered 96-candidate spaces and baseline recipes. The older 24-row `candidate_schedule` utility retained inside the frozen kernel source is not the formal96-candidate schedule; use these JSON spaces for the revised study. The baseline configuration file documents the original comparator roster; this focused release does not implement every historical comparator pipeline.
- `configs/formal_protocol.json`, `component_experiment.json`, `lofo_experiment.json` and `cost_experiment.json`: chronological development, matched components/loss masses, cross-farm label budgets and computational settings.
- `provenance/source_manifest.json`: frozen source hashes and the precise extraction/import adjustments.

## Fit and predict

Supply the standardized NPZ arrays described in [docs/data_format.md](docs/data_format.md). Fitting rows must be ordered **source first, target second**. The fitting archive contains fitting labels; the inference archive contains `X` only.

```bash
tlqldmr fit \
  --fit /path/to/fit.npz \
  --inference /path/to/inference.npz \
  --config configs/selected_tl/farm1_point.json \
  --seed 0 --output outputs/farm1_point_seed0

tlqldmr predict \
  --model outputs/farm1_point_seed0 \
  --inference /path/to/new_inference.npz \
  --output outputs/new_predictions.npz

tlqldmr score \
  --predictions outputs/farm1_point_seed0/predictions.npz \
  --targets /path/to/assessment_targets.npz \
  --output outputs/farm1_point_seed0/assessment.json
```

The selected point procedure is fitted with all nine heads, matching the formal study. Point forecasts integrate the rearranged quantile function with constant extreme tails. Nominal-95% intervals exchange only the specified endpoint heads; the distribution evaluation rearranges the full native grid. Raw outputs and the original 0.5 head are retained separately. Predictions are returned in the original label units without clipping. Fit/inference and assessment are separate commands.

Use `--method Weighted-SVQR`, `--method NoVariance` or `--method NoAlignment` with the same configuration for matched fixed-map controls. Independently selected component controls and cross-farm procedures require their own development/fit populations; reusing a main-study configuration is not equivalent to that separate experiment.

## Reproducing the study

The paper uses six farms, full eligible fitting populations, four chronological development folds and three final fitting seeds. Registered search spaces and completed availability are distinct: not all registered candidates completed before the development cutoff. The published assessment period was previously inspected and is described as retrospective. The supplied configurations preserve those facts.

This repository is a focused core/configuration release. It does not redistribute raw six-farm observations, forecast arrays, fitted weights, terminal search ledgers or every historical orchestration/report-generation script. Obtain the original data archive cited in the manuscript (Scientific Data, DOI [10.1038/s41597-022-01696-6](https://doi.org/10.1038/s41597-022-01696-6)); construct and verify the source/target populations, 24-lag windows and split identities before applying the recorded configurations. Full numerical reproduction additionally requires the exact study caches and computation record supplied with the revision. See [docs/reproduction_scope.md](docs/reproduction_scope.md).
