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
#   POSITION_UNCERTAINTIES  — vector of position uncertainties in meters
#   NUM_ORBITS_LIST         — vector of number of orbital periods to propagate
#   NUM_MC_SAMPLES          — number of MC samples (ground truth / saving)
#   NUM_MC_SAMPLES_BINNING  — number of MC samples for energy binning method
#   NUM_ENERGY_BINS_LIST    — vector of bin counts for energy-binned sweep
#
# Helper functions:
#   compute_velocity_uncertainty(σ_pos, sma, r_vec_0, GM) → σ_vel
#   build_initial_covariance(σ_pos, σ_vel) → 6×6 diagonal P_0
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

# ---- Helper functions -------------------------------------------------------

"""
    compute_velocity_uncertainty(σ_pos, sma, r_vec_0, GM)

Compute 1-sigma velocity uncertainty from position uncertainty via the
vis-viva equation uncertainty propagation law.
"""
function compute_velocity_uncertainty(σ_pos, sma, r_vec_0, GM)
    # Compute velocity uncertainty from sqrt(GM/(sma + sigma_position))
    # n = sqrt(SIM_PARAMS.GM / sma^3)  # Circular orbit velocity
    # σ_vel = n * σ_pos

    # Compute velocity uncertainty from the Uncertainty Propagation Law
    # norm(v) = sqrt(GM * (2/norm(r) - 1/a)) = f(norm(v), a)
    # σ_norm(v) = sqrt( (∂f/∂norm(r))² * σ_norm(r)² + (∂f/∂a)² * σ_a² ), but σ_a = 0
    # ∂f/∂norm(r) = -GM / (sqrt(GM * (2/norm(r) - 1/a)) * norm(r)²)
    σ_vel = sqrt((-GM / (sqrt(GM * (2 / norm(r_vec_0) - 1 / sma)) * norm(r_vec_0)^2))^2 * σ_pos^2)
    return σ_vel
end

"""
    build_initial_covariance(σ_pos, σ_vel)

Build a 6×6 diagonal initial covariance matrix from isotropic position
and velocity uncertainties.
"""
function build_initial_covariance(σ_pos, σ_vel)
    return diagm([σ_pos^2, σ_pos^2, σ_pos^2, σ_vel^2, σ_vel^2, σ_vel^2])
end

# ---- Scenario / run parameters ---------------------------------------------

# Position uncertainty scenarios (meters)
POSITION_UNCERTAINTIES = [1e3, 1e4, 1e5]  # 1 km, 10 km, 100 km

# Number of orbital periods to propagate
NUM_ORBITS_LIST = [3.0]

# Monte Carlo sample counts
NUM_MC_SAMPLES = 5000           # ground truth / save script
NUM_MC_SAMPLES_BINNING = 5000   # energy binning method

# Energy bins sweep
NUM_ENERGY_BINS_LIST = [1, 2, 5, 10, 20, 50]
