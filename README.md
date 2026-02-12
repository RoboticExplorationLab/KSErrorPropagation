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

### Running with a config file argument

Julia exposes command-line arguments in `ARGS`. Scripts that load config can accept an optional config path:

```bash
julia scripts/save_monte_carlo_npz.jl
# uses config/default.jl

julia scripts/save_monte_carlo_npz.jl config/fast.jl
# uses config/fast.jl
```

The script chooses the config path from `ARGS` and includes it before using the config. With no argument, it falls back to `config/default.jl`.

Check project status in REPL:
```julia
julia
# In REPL:
] activate .
] st
```
