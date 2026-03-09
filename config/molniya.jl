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

# NUM_MC_SAMPLES = 100

########################################################
# Does not work for any
# OE_INITIAL_STD_SCENARIOS = [
#     [1e3, 0.001, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)],
#     [5e4, 0.000001, deg2rad(0.000001), deg2rad(0.000001), deg2rad(0.000001), deg2rad(0.000001)],
#     [1e5, 0.000001, deg2rad(0.000001), deg2rad(0.000001), deg2rad(0.000001), deg2rad(0.000001)]
# ]
########################################################

# # (1) Works for all
# OE_INITIAL_STD_SCENARIOS = [
#     [1e3, 0.00001, deg2rad(0.001), deg2rad(0.001), deg2rad(0.001), deg2rad(0.001)],
# ]

# # (2) Works for Cartesian LinCov, UT, and Energy-Stratified KS CKF
# OE_INITIAL_STD_SCENARIOS = [
#     [5e3, 0.00001, deg2rad(0.001), deg2rad(0.001), deg2rad(0.001), deg2rad(0.001)],
# ]

# # (3) Works for Cartesian LinCov and Energy-Stratified KS CKF
# OE_INITIAL_STD_SCENARIOS = [
#     [1e4, 0.00001, deg2rad(0.001), deg2rad(0.001), deg2rad(0.001), deg2rad(0.001)],
# ]

OE_INITIAL_STD_SCENARIOS = [
    [1e3, 0.00001, deg2rad(0.001), deg2rad(0.001), deg2rad(0.001), deg2rad(0.001)],    
    [5e3, 0.00001, deg2rad(0.001), deg2rad(0.001), deg2rad(0.001), deg2rad(0.001)],
    [1e4, 0.00001, deg2rad(0.001), deg2rad(0.001), deg2rad(0.001), deg2rad(0.001)],
    [5e4, 0.00001, deg2rad(0.001), deg2rad(0.001), deg2rad(0.001), deg2rad(0.001)],
    [1e5, 0.00001, deg2rad(0.001), deg2rad(0.001), deg2rad(0.001), deg2rad(0.001)]
]