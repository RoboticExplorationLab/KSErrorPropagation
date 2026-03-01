# ============================================================================
# Default simulation configuration
# ============================================================================
# Single source of truth for all simulation parameters, test orbits,
# uncertainty scenarios, and run settings.
#
# Usage (from test/ or scripts/):
#   include(joinpath(@__DIR__, "..", "config", "default.jl"))
#
# After including, the following variables are defined in the caller's scope:
#   SIM_PARAMS              — named tuple of physical/integrator parameters
#   TEST_ORBITS             — vector of orbit named tuples (name, a, e, i, ...)
#   OE_INITIAL_STD_SCENARIOS — vector of 6-element OE initial std [σ_a, σ_e, σ_i, σ_RAAN, σ_ω, σ_M]
#   NUM_ORBITS_LIST         — vector of number of orbital periods to propagate
#   NUM_MC_SAMPLES          — number of MC samples (ground truth / saving)
#   NUM_MC_SAMPLES_BINNING  — number of MC samples for energy-stratified method
#   NUM_ENERGY_BINS         — number of energy bins for comparison test (single run)
#   NUM_ENERGY_BINS_LIST    — vector of bin counts for energy-stratified sweep
#   RANDOM_SEED             — seed for Random.seed! (reproducible runs)
#   KS_RELATIVE_STATE_NORM_THRESHOLD — threshold on norm(ks_state_rel[1:4]) to choose
#       separate vs together callback in KS relative dynamics (default 1e-7)
#
# Helper (in error_propagation.jl):
#   compute_P0_from_oe_samples(oe_vec_0, σ_oe, GM, R_Earth; n_samples=10000) → 6×6 Cartesian P_0
# ============================================================================

# ---- Simulation parameters -------------------------------------------------

SIM_PARAMS = (
    # Physical parameters
    GM = SD.GM_EARTH,
    R_EARTH = SD.R_EARTH,
    J2 = SD.J2_EARTH,
    OMEGA_EARTH = SD.OMEGA_EARTH,
    CD = 2.2,       # drag coefficient
    A = 1.0,        # cross-sectional area (m²)
    m = 100.0,      # mass (kg)
    epoch_0 = Epoch(2000, 1, 1, 12, 0, 0),  # J2000

    # Perturbations
    add_perturbations = true,

    # Sampling time (time step) in seconds
    sampling_time = 30.0,

    # Integrator (from DifferentialEquations.jl)
    integrator = Tsit5,

    # Integrator tolerances
    abstol = 1e-12,
    reltol = 1e-13,

    # Success criteria thresholds
    max_pos_error_threshold = 1.0,    # meters
    max_vel_error_threshold = 0.01,   # m/s

    # Scaling factors (computed from orbit when nothing)
    t_scale = nothing,
    r_scale = nothing,
    v_scale = nothing,
    a_scale = nothing,
    GM_scale = nothing,
)

# ---- Test orbits -----------------------------------------------------------

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

# ---- Scenario / run parameters ---------------------------------------------

# Reproducibility
RANDOM_SEED = 1234

# KS relative dynamics: threshold to choose separate vs together integration callback
KS_RELATIVE_STATE_NORM_THRESHOLD = 1e-7

# OE initial std scenarios: [σ_a (m), σ_e, σ_i, σ_RAAN, σ_ω, σ_M] (rad for angles)
OE_INITIAL_STD_SCENARIOS = [
    [1e3, 0.001, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)],
    [1e4, 0.001, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)],
    [1e5, 0.001, deg2rad(0.1), deg2rad(0.1), deg2rad(0.1), deg2rad(1.0)],
]

# Number of orbital periods to propagate
NUM_ORBITS_LIST = [3.0]

# Monte Carlo sample counts
NUM_MC_SAMPLES = 5000           # ground truth / save script
NUM_MC_SAMPLES_BINNING = 5000   # energy-stratified method

# Energy bins: single value for comparison test, list for sweep
NUM_ENERGY_BINS = 10
NUM_ENERGY_BINS_LIST = [1, 2, 5, 10, 20, 50]
