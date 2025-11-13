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

Check project status in REPL:
```julia
julia
# In REPL:
] activate .
] st
```
