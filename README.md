# KSErrorPropagation

Orbital state uncertainty propagation using Kustaanheimo–Stiefel (KS) coordinates and various filtering approaches. Compares multiple error propagation methods against Monte Carlo ground truth.

## Approaches

| Name | Description |
|------|-------------|
| **Monte Carlo (Ground Truth)** | Full nonlinear propagation of many samples; mean and covariance estimated from propagated ensemble. |
| **Cartesian LinCov** | Linearized covariance propagation via state transition matrix (STM) in Cartesian coordinates. |
| **Cartesian UT** | Unscented Transform (2n+1 sigma points) with resampling at each timestep; Cartesian dynamics. |
| **Cartesian CKF** | Cubature Kalman Filter (2n sigma points, spherical-radial rule) with resampling; Cartesian dynamics. |
| **KS CKF** | Same CKF sigma-point rule, but propagates via KS dynamics. |
| **KS Relative CKF** | CKF with linearized KS relative (chief-deputy) dynamics at each step. |
| **KS LinCov** | STM-based covariance propagation in KS coordinates with energy augmentation. |
| **Energy-Stratified KS CKF** | Stratified sampling by orbital energy; per-stratum KS CKF propagation; weighted aggregation. |

## Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage

Scripts accept an optional config path (default: `config/default.jl`) and can be run with or without logging.

**Run comparison** (compares approaches vs Monte Carlo):

```bash
julia --project=. scripts/error_propagation_comparison.jl [config.jl] [save]
```

Pass `save` to write results to `out/` for later plotting. Use `scripts/compare_approaches.jl` to reload from `out/` and regenerate plots without re-running propagation.

**Run with log file** (output to terminal and timestamped log):

```bash
bash run_with_log.sh scripts/error_propagation_comparison.jl config/sanity_check.jl save
```

Logs go to `logs/<script_basename>_<YYYY-MM-DD>_<HH-MM-SS>.log`.

**Other scripts:** `scripts/run_monte_carlo.jl`, `scripts/compare_approaches.jl`, etc. All accept `[config.jl]` as first argument.
