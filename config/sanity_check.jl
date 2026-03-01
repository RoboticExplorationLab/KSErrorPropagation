include(joinpath(@__DIR__, "default.jl"))

OE_INITIAL_STD_SCENARIOS = [
    [1e2, 0.01, deg2rad(0.01), deg2rad(0.01), deg2rad(0.01), deg2rad(0.1)],
]
NUM_MC_SAMPLES = 100
