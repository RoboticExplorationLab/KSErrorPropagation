include(joinpath(@__DIR__, "default.jl"))

OE_INITIAL_STD_SCENARIOS = [
    [1e1, 0.00001, deg2rad(0.001), deg2rad(0.001), deg2rad(0.001), deg2rad(0.001)],
]
NUM_MC_SAMPLES = 100
