include(joinpath(@__DIR__, "default.jl"))

TEST_ORBITS = [
    (name = "Molniya orbit (e=0.74, 26600 km semi-major axis)",
        a = 26600e3,
        e = 0.74,
        i = deg2rad(63.4),
        omega = deg2rad(270.0),
        RAAN = deg2rad(0.0),
        M = deg2rad(0.0),
        description = "Molniya",
        id = "mol"),
]

OE_INITIAL_STD_SCENARIOS = [
    [1e3,  1e-5, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)],
    [5e3,  1e-5, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)],
    [10e3, 1e-5, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)],
    [20e3, 1e-5, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)]
]