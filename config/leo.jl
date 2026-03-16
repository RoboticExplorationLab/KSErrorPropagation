include(joinpath(@__DIR__, "default.jl"))

TEST_ORBITS = [
    (name = "Low Earth orbit (e=0.01, 750 km altitude)",
        a = 750e3 + SIM_PARAMS.R_EARTH,
        e = 0.01,
        i = deg2rad(0.0),
        omega = deg2rad(0.0),
        RAAN = deg2rad(0.0),
        M = deg2rad(0.0),
        description = "LEO",
        id = "leo"),
]

OE_INITIAL_STD_SCENARIOS = [
    [1e3,  1e-3, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)],
    [5e3,  1e-3, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)],
    [10e3, 1e-3, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)],
    [20e3, 1e-3, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)]
]