# KSErrorPropagation

## Setup

Install dependencies:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Update package resolution (resolve dependencies):
```bash
julia --project=. -e 'using Pkg; Pkg.resolve()'
```

Run scripts:
```bash
julia --project=. julia_script.jl
```

### Running with a log file

To run a Julia script with output both on the terminal and in a timestamped log file:

```bash
# From repo root:
bash julia/run_with_log.sh julia/scripts/<script_name>.jl
bash julia/run_with_log.sh julia/scripts/<script_name>.jl config/fast.jl   # with config
```

Logs are written to `julia/logs/<script_basename>_<YYYY-MM-DD>_<HH-MM-SS>.log`. Example:

```bash
bash julia/run_with_log.sh julia/scripts/run_monte_carlo_and_save_npz.jl
bash julia/run_with_log.sh julia/scripts/compare_approaches.jl config/sanity_check.jl
```

### Running with a config file argument

Julia exposes command-line arguments in `ARGS`. Scripts that load config can accept an optional config path:

```bash
julia scripts/save_monte_carlo_npz.jl
# uses config/default.jl

julia scripts/save_monte_carlo_npz.jl config/fast.jl
# uses config/fast.jl
```

The script chooses the config path from `ARGS` and includes it before using the config. With no argument, it falls back to `config/default.jl`.

### Error propagation approaches

The following approaches are implemented and compared against Monte Carlo ground truth:

| Name | Description |
|------|-------------|
| **Monte Carlo (Ground Truth)** | Full nonlinear propagation of many samples; mean and covariance estimated from propagated ensemble. |
| **LinCov (Cartesian)** | Linearized covariance propagation via state transition matrix (STM) in Cartesian coordinates. |
| **Cartesian UT** | Unscented Transform (2n+1 sigma points) with resampling at each timestep; Cartesian dynamics. |
| **Cartesian CKF** | Cubature Kalman Filter (2n sigma points, spherical-radial rule) with resampling; Cartesian dynamics. |
| **KS CKF** | Same CKF sigma-point rule, but propagates via KS (Kustaanheimo–Stiefel) dynamics. |
| **KS Relative CKF** | CKF with linearized KS relative (chief-deputy) dynamics at each step. |
| **KS LinCov** | STM-based covariance propagation in KS coordinates with energy augmentation. |
| **Energy-Stratified KS CKF** | Stratified sampling by orbital energy; per-stratum KS CKF propagation; weighted aggregation. |

Results for each approach are saved to `julia/out/` when running the comparison test with the `save` argument. Use `julia/scripts/compare_approaches.jl` to reload from `out/` and regenerate comparison plots without re-running the propagation.

### Error propagation comparison test

`julia/test/test_error_propagation_comparison.jl` compares error propagation methods against Monte Carlo. Arguments:

| Argument   | Description |
|-----------|-------------|
| 1st (optional) | Config file path (e.g. `config/sanity_check.jl`). Default: `config/default.jl`. |
| 2nd (optional) | Save results to `julia/out/`: pass `save` or `true` to enable. Default: do not save. |

Examples:

```bash
# Default config, no saving to out/
julia --project=julia julia/test/test_error_propagation_comparison.jl

# Sanity-check config
julia --project=julia julia/test/test_error_propagation_comparison.jl config/sanity_check.jl

# Save approach results to julia/out/
julia --project=julia julia/test/test_error_propagation_comparison.jl config/sanity_check.jl save
```

With the log script (from repo root or from `julia/`):

```bash
bash julia/run_with_log.sh test/test_error_propagation_comparison.jl config/sanity_check.jl
bash julia/run_with_log.sh test/test_error_propagation_comparison.jl config/sanity_check.jl save
```

Check project status in REPL:
```julia
julia
# In REPL:
] activate .
] st
```
