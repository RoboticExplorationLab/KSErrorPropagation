using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots
using LinearAlgebra
using SatelliteDynamics
using DifferentialEquations

const SD = SatelliteDynamics

include("../src/cartesian_dynamics.jl")
include("../src/ks_dynamics.jl")
include("../src/error_propagation.jl")
include("../src/utils.jl")

# Define simulation parameters
SIM_PARAMS = (
    # Physical parameters
    GM=SD.GM_EARTH,
    R_EARTH=SD.R_EARTH,
    J2=SD.J2_EARTH,
    OMEGA_EARTH=SD.OMEGA_EARTH,
    CD=2.2, # drag coefficient
    A=1.0, # cross-sectional area (m²)
    m=100.0, # mass (kg)
    epoch_0=Epoch(2000, 1, 1, 12, 0, 0), # initial epoch at 0 TBD seconds after J2000

    # Add perturbations
    add_perturbations=true,

    # Number of orbits to simulate
    num_orbits=1,

    # Sampling time (time step) in seconds
    sampling_time=30.0,  # seconds

    # Integrator to use (from DifferentialEquations.jl)
    integrator=Tsit5,

    # Integrator tolerances
    abstol=1e-12,      # Absolute tolerance
    reltol=1e-13,      # Relative tolerance

    # Success criteria thresholds
    max_pos_error_threshold=1.0,      # meters
    max_vel_error_threshold=0.01,     # m/s

    # Scaling factors (optional, computed from orbit if not provided)
    t_scale=nothing,  # time scale (typically orbital period)
    r_scale=nothing,  # position scale (typically semi-major axis)
    v_scale=nothing,  # velocity scale
    a_scale=nothing,  # acceleration scale
    GM_scale=nothing,  # gravitational constant scale
)

# Define test orbits
test_orbits = [
    (name="Low Earth orbit (e=0.1, 750 km altitude)",
        a=750e3 + SIM_PARAMS.R_EARTH,
        e=0.01,
        i=deg2rad(0.0),
        omega=deg2rad(0.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="LEO"),
    (name="Molniya orbit (e=0.74, 26600 km semi-major axis)",
        a=26600e3,
        e=0.74,
        i=deg2rad(63.4),
        omega=deg2rad(270.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Molniya"),
]

# Position uncertainty scenarios (meters)
position_uncertainties = [1e3, 1e4, 1e5]  # 1km, 10km, 100km

# Number of orbits to test
num_orbits_list = [3.0]#, 3.0, 5.0]

println("="^80)
println("ERROR PROPAGATION COMPARISON TEST")
println("="^80)
println("Comparing error propagation methods against Monte Carlo ground truth")
println("\nTest configuration:")
println("  Number of orbits: ", num_orbits_list)
println("  Position uncertainties: ", position_uncertainties, " m")
println("  Number of test scenarios: ", length(test_orbits) * length(position_uncertainties) * length(num_orbits_list))

# Store results for all scenarios
all_results = []

# Loop over orbits
for (orbit_idx, orbit) in enumerate(test_orbits)
    sma = orbit.a
    e = orbit.e
    i = orbit.i
    ω = orbit.omega
    Ω = orbit.RAAN
    M = orbit.M

    println("\n" * "="^80)
    println("ORBIT: ", orbit.name)
    println("="^80)
    println("Orbit parameters:")
    println("  Semi-major axis: ", sma / 1e3, " km")
    println("  Eccentricity: ", e)
    println("  Inclination: ", rad2deg(i), " deg")
    println("  Argument of periapsis: ", rad2deg(ω), " deg")
    println("  RAAN: ", rad2deg(Ω), " deg")
    println("  Mean anomaly: ", rad2deg(M), " deg")

    # Convert orbital elements to Cartesian state
    oe_vec = [sma, e, i, Ω, ω, M]
    x_vec_0 = SD.sOSCtoCART(oe_vec; GM=SIM_PARAMS.GM, use_degrees=false)
    r_vec_0 = x_vec_0[1:3]
    v_vec_0 = x_vec_0[4:6]

    println("\nInitial conditions:")
    println("  Position: ", r_vec_0)
    println("  Velocity: ", v_vec_0)
    println("  Radius: ", norm(r_vec_0) / 1e3, " km")
    println("  Speed: ", norm(v_vec_0) / 1e3, " km/s")

    # Loop over position uncertainty scenarios
    for σ_pos in position_uncertainties
        # Compute velocity uncertainty from sqrt(GM/(sma + sigma_position))
        # n = sqrt(SIM_PARAMS.GM / sma^3)  # Circular orbit velocity
        # σ_vel = n * σ_pos

        # Compute velocity uncertainty from the Uncertainty Propagation Law
        # norm(v) = sqrt(GM * (2/norm(r) - 1/a)) = f(norm(v), a)
        # σ_norm(v) = sqrt( (∂f/∂norm(r))² * σ_norm(r)² + (∂f/∂a)² * σ_a² ), but σ_a = 0
        # ∂f/∂norm(r) = -GM / (sqrt(GM * (2/norm(r) - 1/a)) * norm(r)²)
        σ_vel = sqrt((-SIM_PARAMS.GM / (sqrt(SIM_PARAMS.GM * (2 / norm(r_vec_0) - 1 / sma)) * norm(r_vec_0)^2))^2 * σ_pos^2)

        println("\n" * "-"^80)
        println("POSITION UNCERTAINTY SCENARIO: σ_pos = ", σ_pos, " m")
        println("-"^80)
        println("  Position uncertainty: ", σ_pos, " m (1-sigma)")
        println("  Velocity uncertainty: ", σ_vel, " m/s (1-sigma)")

        # Initial state covariance
        P_0 = diagm([σ_pos^2, σ_pos^2, σ_pos^2, σ_vel^2, σ_vel^2, σ_vel^2])

        # Loop over number of orbits
        for num_orbits in num_orbits_list
            println("\n" * "="^80)
            println("NUM ORBITS: ", num_orbits)
            println("="^80)

            # Propagation time: multiple orbital periods
            T_orbital = 2π * sqrt(sma^3 / SIM_PARAMS.GM)
            t_0 = 0.0
            t_end = num_orbits * T_orbital
            dt = SIM_PARAMS.sampling_time
            times = collect(t_0:dt:t_end)

            println("\nPropagation parameters:")
            println("  Orbital period: ", T_orbital, " s (", T_orbital / 3600, " hours)")
            println("  Time step: ", dt, " s")
            println("  Number of steps: ", length(times))
            println("  Total propagation time: ", t_end / 3600, " hours")

            # Scaling factors
            t_scale = something(SIM_PARAMS.t_scale, T_orbital)
            r_scale = something(SIM_PARAMS.r_scale, sma)
            v_scale = something(SIM_PARAMS.v_scale, r_scale / t_scale)
            a_scale = something(SIM_PARAMS.a_scale, v_scale / t_scale)
            GM_scale = something(SIM_PARAMS.GM_scale, r_scale^3 / t_scale^2)

            # Scaling matrices for covariance propagation
            S = Diagonal([1 / r_scale, 1 / r_scale, 1 / r_scale, 1 / v_scale, 1 / v_scale, 1 / v_scale])
            S_inv = Diagonal([r_scale, r_scale, r_scale, v_scale, v_scale, v_scale])

            sim_params = merge(SIM_PARAMS, (t_scale=t_scale, r_scale=r_scale, v_scale=v_scale,
                a_scale=a_scale, GM_scale=GM_scale, S=S, S_inv=S_inv))

            # Monte Carlo (ground truth)
            println("\n" * "="^80)
            println("1. MONTE CARLO (Ground Truth)")
            println("="^80)
            x_vec_traj_mean_mc, P_traj_mc = propagate_uncertainty_via_monte_carlo(x_vec_0, P_0, times, sim_params, 5000)
            println("  Completed: ", length(x_vec_traj_mean_mc), " states")

            # Linearized Covariance Propagation (Cartesian)
            println("\n" * "="^80)
            println("2. LINEARIZED COVARIANCE PROPAGATION (Cartesian Coordinates)")
            println("="^80)
            x_vec_traj_mean_lin_cart, P_traj_lin_cart = propagate_uncertainty_via_linearized_cartesian_dynamics(x_vec_0, P_0, times, sim_params)
            println("  Completed: ", length(x_vec_traj_mean_lin_cart), " states")

            # Unscented Transform (Cartesian)
            println("\n" * "="^80)
            println("3. UNSCENTED TRANSFORM (Cartesian Coordinates)")
            println("="^80)
            x_vec_traj_mean_ut_cart, P_traj_ut_cart = propagate_uncertainty_via_cartesian_unscented_transform(x_vec_0, P_0, times, sim_params)
            println("  Completed: ", length(x_vec_traj_mean_ut_cart), " states")

            # Eigen-based Sigma Points (Cartesian)
            println("\n" * "="^80)
            println("4. EIGEN-BASED SIGMA POINTS (Cartesian Coordinates)")
            println("="^80)
            x_vec_traj_mean_eigen_cart, P_traj_eigen_cart = propagate_uncertainty_via_cartesian_sigma_points(x_vec_0, P_0, times, sim_params)
            println("  Completed: ", length(x_vec_traj_mean_eigen_cart), " states")

            # Eigen-based Sigma Points (KS)
            println("\n" * "="^80)
            println("5. EIGEN-BASED SIGMA POINTS (KS Coordinates)")
            println("="^80)
            x_vec_traj_mean_eigen_ks, P_traj_eigen_ks = propagate_uncertainty_via_ks_sigma_points(x_vec_0, P_0, times, sim_params)
            println("  Completed: ", length(x_vec_traj_mean_eigen_ks), " states")

            # Linearized KS Sigma Points
            println("\n" * "="^80)
            println("6. LINEARIZED KS SIGMA POINTS")
            println("="^80)
            x_vec_traj_mean_lin_ks_sigma, P_traj_lin_ks_sigma = propagate_uncertainty_via_linearized_ks_sigma_points(x_vec_0, P_0, times, sim_params)
            println("  Completed: ", length(x_vec_traj_mean_lin_ks_sigma), " states")

            # Linearized KS Dynamics
            println("\n" * "="^80)
            println("7. LINEARIZED KS DYNAMICS")
            println("="^80)
            x_vec_traj_mean_lin_ks_dyn, P_traj_lin_ks_dyn = propagate_uncertainty_via_linearized_ks_dynamics(x_vec_0, P_0, times, sim_params)
            println("  Completed: ", length(x_vec_traj_mean_lin_ks_dyn), " states")

            # Compare against Monte Carlo
            println("\n" * "="^80)
            println("COMPARISON AGAINST MONTE CARLO GROUND TRUTH")
            println("="^80)

            # Compute error metrics using the new function
            metrics_lin_cart = error_metrics(x_vec_traj_mean_mc, P_traj_mc,
                x_vec_traj_mean_lin_cart, P_traj_lin_cart)
            metrics_ut_cart = error_metrics(x_vec_traj_mean_mc, P_traj_mc,
                x_vec_traj_mean_ut_cart, P_traj_ut_cart)
            metrics_eigen_cart = error_metrics(x_vec_traj_mean_mc, P_traj_mc,
                x_vec_traj_mean_eigen_cart, P_traj_eigen_cart)
            metrics_eigen_ks = error_metrics(x_vec_traj_mean_mc, P_traj_mc,
                x_vec_traj_mean_eigen_ks, P_traj_eigen_ks)
            metrics_lin_ks_sigma = error_metrics(x_vec_traj_mean_mc, P_traj_mc,
                x_vec_traj_mean_lin_ks_sigma, P_traj_lin_ks_sigma)
            metrics_lin_ks_dyn = error_metrics(x_vec_traj_mean_mc, P_traj_mc,
                x_vec_traj_mean_lin_ks_dyn, P_traj_lin_ks_dyn)
            N = min(length(x_vec_traj_mean_mc), length(x_vec_traj_mean_lin_cart), length(x_vec_traj_mean_ut_cart), length(x_vec_traj_mean_eigen_cart), length(x_vec_traj_mean_eigen_ks), length(x_vec_traj_mean_lin_ks_sigma), length(x_vec_traj_mean_lin_ks_dyn))

            # Print position errors
            println("\nPosition errors (vs Monte Carlo):")
            println("  Linearized Covariance Propagation (Cartesian):")
            println("    RMSE = ", metrics_lin_cart.pos_rmse, " m, Min = ", metrics_lin_cart.pos_min, " m, Max = ", metrics_lin_cart.pos_max, " m")
            println("  Unscented Transform (Cartesian):")
            println("    RMSE = ", metrics_ut_cart.pos_rmse, " m, Min = ", metrics_ut_cart.pos_min, " m, Max = ", metrics_ut_cart.pos_max, " m")
            println("  Eigen-based Sigma Points (Cartesian):")
            println("    RMSE = ", metrics_eigen_cart.pos_rmse, " m, Min = ", metrics_eigen_cart.pos_min, " m, Max = ", metrics_eigen_cart.pos_max, " m")
            println("  Eigen-based Sigma Points (KS):")
            println("    RMSE = ", metrics_eigen_ks.pos_rmse, " m, Min = ", metrics_eigen_ks.pos_min, " m, Max = ", metrics_eigen_ks.pos_max, " m")
            println("  Linearized KS Sigma Points:")
            println("    RMSE = ", metrics_lin_ks_sigma.pos_rmse, " m, Min = ", metrics_lin_ks_sigma.pos_min, " m, Max = ", metrics_lin_ks_sigma.pos_max, " m")
            println("  Linearized KS Dynamics:")
            println("    RMSE = ", metrics_lin_ks_dyn.pos_rmse, " m, Min = ", metrics_lin_ks_dyn.pos_min, " m, Max = ", metrics_lin_ks_dyn.pos_max, " m")

            # Print velocity errors
            println("\nVelocity errors (vs Monte Carlo):")
            println("  Linearized Covariance Propagation (Cartesian):")
            println("    RMSE = ", metrics_lin_cart.vel_rmse, " m/s, Min = ", metrics_lin_cart.vel_min, " m/s, Max = ", metrics_lin_cart.vel_max, " m/s")
            println("  Unscented Transform (Cartesian):")
            println("    RMSE = ", metrics_ut_cart.vel_rmse, " m/s, Min = ", metrics_ut_cart.vel_min, " m/s, Max = ", metrics_ut_cart.vel_max, " m/s")
            println("  Eigen-based Sigma Points (Cartesian):")
            println("    RMSE = ", metrics_eigen_cart.vel_rmse, " m/s, Min = ", metrics_eigen_cart.vel_min, " m/s, Max = ", metrics_eigen_cart.vel_max, " m/s")
            println("  Eigen-based Sigma Points (KS):")
            println("    RMSE = ", metrics_eigen_ks.vel_rmse, " m/s, Min = ", metrics_eigen_ks.vel_min, " m/s, Max = ", metrics_eigen_ks.vel_max, " m/s")
            println("  Linearized KS Sigma Points:")
            println("    RMSE = ", metrics_lin_ks_sigma.vel_rmse, " m/s, Min = ", metrics_lin_ks_sigma.vel_min, " m/s, Max = ", metrics_lin_ks_sigma.vel_max, " m/s")
            println("  Linearized KS Dynamics:")
            println("    RMSE = ", metrics_lin_ks_dyn.vel_rmse, " m/s, Min = ", metrics_lin_ks_dyn.vel_min, " m/s, Max = ", metrics_lin_ks_dyn.vel_max, " m/s")

            # Print position and velocity uncertainty errors
            println("\nPosition uncertainty errors (vs Monte Carlo):")
            println("  Linearized Covariance Propagation (Cartesian):")
            println("    RMSE = ", metrics_lin_cart.pos_uncertainty_rmse, " m, Min = ", metrics_lin_cart.pos_uncertainty_min, " m, Max = ", metrics_lin_cart.pos_uncertainty_max, " m")
            println("  Unscented Transform (Cartesian):")
            println("    RMSE = ", metrics_ut_cart.pos_uncertainty_rmse, " m, Min = ", metrics_ut_cart.pos_uncertainty_min, " m, Max = ", metrics_ut_cart.pos_uncertainty_max, " m")
            println("  Eigen-based Sigma Points (Cartesian):")
            println("    RMSE = ", metrics_eigen_cart.pos_uncertainty_rmse, " m, Min = ", metrics_eigen_cart.pos_uncertainty_min, " m, Max = ", metrics_eigen_cart.pos_uncertainty_max, " m")
            println("  Eigen-based Sigma Points (KS):")
            println("    RMSE = ", metrics_eigen_ks.pos_uncertainty_rmse, " m, Min = ", metrics_eigen_ks.pos_uncertainty_min, " m, Max = ", metrics_eigen_ks.pos_uncertainty_max, " m")
            println("  Linearized KS Sigma Points:")
            println("    RMSE = ", metrics_lin_ks_sigma.pos_uncertainty_rmse, " m, Min = ", metrics_lin_ks_sigma.pos_uncertainty_min, " m, Max = ", metrics_lin_ks_sigma.pos_uncertainty_max, " m")
            println("  Linearized KS Dynamics:")
            println("    RMSE = ", metrics_lin_ks_dyn.pos_uncertainty_rmse, " m, Min = ", metrics_lin_ks_dyn.pos_uncertainty_min, " m, Max = ", metrics_lin_ks_dyn.pos_uncertainty_max, " m")

            println("\nVelocity uncertainty errors (vs Monte Carlo):")
            println("  Linearized Covariance Propagation (Cartesian):")
            println("    RMSE = ", metrics_lin_cart.vel_uncertainty_rmse, " m/s, Min = ", metrics_lin_cart.vel_uncertainty_min, " m/s, Max = ", metrics_lin_cart.vel_uncertainty_max, " m/s")
            println("  Unscented Transform (Cartesian):")
            println("    RMSE = ", metrics_ut_cart.vel_uncertainty_rmse, " m/s, Min = ", metrics_ut_cart.vel_uncertainty_min, " m/s, Max = ", metrics_ut_cart.vel_uncertainty_max, " m/s")
            println("  Eigen-based Sigma Points (Cartesian):")
            println("    RMSE = ", metrics_eigen_cart.vel_uncertainty_rmse, " m/s, Min = ", metrics_eigen_cart.vel_uncertainty_min, " m/s, Max = ", metrics_eigen_cart.vel_uncertainty_max, " m/s")
            println("  Eigen-based Sigma Points (KS):")
            println("    RMSE = ", metrics_eigen_ks.vel_uncertainty_rmse, " m/s, Min = ", metrics_eigen_ks.vel_uncertainty_min, " m/s, Max = ", metrics_eigen_ks.vel_uncertainty_max, " m/s")
            println("  Linearized KS Sigma Points:")
            println("    RMSE = ", metrics_lin_ks_sigma.vel_uncertainty_rmse, " m/s, Min = ", metrics_lin_ks_sigma.vel_uncertainty_min, " m/s, Max = ", metrics_lin_ks_sigma.vel_uncertainty_max, " m/s")
            println("  Linearized KS Dynamics:")
            println("    RMSE = ", metrics_lin_ks_dyn.vel_uncertainty_rmse, " m/s, Min = ", metrics_lin_ks_dyn.vel_uncertainty_min, " m/s, Max = ", metrics_lin_ks_dyn.vel_uncertainty_max, " m/s")

            # Create plots
            println("\nGenerating plots...")

            # Plot 1: Position error
            p1 = plot(times[1:N] ./ 3600, metrics_lin_cart.pos_errors, label="Linearized Covariance",
                linewidth=2, color=:green, linestyle=:dash, xlabel="Time (hours)",
                ylabel="Position Error (m)", yscale=:log10, legend=:topleft)
            plot!(p1, times[1:N] ./ 3600, metrics_ut_cart.pos_errors, label="Unscented Transform",
                linewidth=2, color=:blue, linestyle=:dot)
            plot!(p1, times[1:N] ./ 3600, metrics_eigen_cart.pos_errors, label="Eigen-based Sigma Points (Cartesian)",
                linewidth=2, color=:red, linestyle=:dashdot)
            plot!(p1, times[1:N] ./ 3600, metrics_eigen_ks.pos_errors, label="Eigen-based Sigma Points (KS)",
                linewidth=2, color=:orange, linestyle=:dashdotdot)
            plot!(p1, times[1:N] ./ 3600, metrics_lin_ks_sigma.pos_errors, label="Linearized KS Sigma Points",
                linewidth=2, color=:purple, linestyle=:dashdotdot)
            plot!(p1, times[1:N] ./ 3600, metrics_lin_ks_dyn.pos_errors, label="Linearized KS Dynamics",
                linewidth=2, color=:brown, linestyle=:dot)

            # Plot 2: Velocity error
            p2 = plot(times[1:N] ./ 3600, metrics_lin_cart.vel_errors, label="Linearized Covariance",
                linewidth=2, color=:green, linestyle=:dash, xlabel="Time (hours)",
                ylabel="Velocity Error (m/s)", yscale=:log10, legend=:topleft)
            plot!(p2, times[1:N] ./ 3600, metrics_ut_cart.vel_errors, label="Unscented Transform",
                linewidth=2, color=:blue, linestyle=:dot)
            plot!(p2, times[1:N] ./ 3600, metrics_eigen_cart.vel_errors, label="Eigen-based Sigma Points (Cartesian)",
                linewidth=2, color=:red, linestyle=:dashdot)
            plot!(p2, times[1:N] ./ 3600, metrics_eigen_ks.vel_errors, label="Eigen-based Sigma Points (KS)",
                linewidth=2, color=:orange, linestyle=:dashdotdot)
            plot!(p2, times[1:N] ./ 3600, metrics_lin_ks_sigma.vel_errors, label="Linearized KS Sigma Points",
                linewidth=2, color=:purple, linestyle=:dashdotdot)
            plot!(p2, times[1:N] ./ 3600, metrics_lin_ks_dyn.vel_errors, label="Linearized KS Dynamics",
                linewidth=2, color=:brown, linestyle=:dot)

            # Plot 3: Position uncertainty error
            p3 = plot(times[1:N] ./ 3600, metrics_lin_cart.pos_uncertainty_errors, label="Linearized Covariance",
                linewidth=2, color=:green, linestyle=:dash, xlabel="Time (hours)",
                ylabel="Position Uncertainty Error (m)", yscale=:log10, legend=:topleft)
            plot!(p3, times[1:N] ./ 3600, metrics_ut_cart.pos_uncertainty_errors, label="Unscented Transform",
                linewidth=2, color=:blue, linestyle=:dot)
            plot!(p3, times[1:N] ./ 3600, metrics_eigen_cart.pos_uncertainty_errors, label="Eigen-based Sigma Points (Cartesian)",
                linewidth=2, color=:red, linestyle=:dashdot)
            plot!(p3, times[1:N] ./ 3600, metrics_eigen_ks.pos_uncertainty_errors, label="Eigen-based Sigma Points (KS)",
                linewidth=2, color=:orange, linestyle=:dashdotdot)
            plot!(p3, times[1:N] ./ 3600, metrics_lin_ks_sigma.pos_uncertainty_errors, label="Linearized KS Sigma Points",
                linewidth=2, color=:purple, linestyle=:dashdotdot)
            plot!(p3, times[1:N] ./ 3600, metrics_lin_ks_dyn.pos_uncertainty_errors, label="Linearized KS Dynamics",
                linewidth=2, color=:brown, linestyle=:dot)

            # Plot 4: Velocity uncertainty error
            p4 = plot(times[1:N] ./ 3600, metrics_lin_cart.vel_uncertainty_errors, label="Linearized Covariance",
                linewidth=2, color=:green, linestyle=:dash, xlabel="Time (hours)",
                ylabel="Velocity Uncertainty Error (m/s)", yscale=:log10, legend=:topleft)
            plot!(p4, times[1:N] ./ 3600, metrics_ut_cart.vel_uncertainty_errors, label="Unscented Transform",
                linewidth=2, color=:blue, linestyle=:dot)
            plot!(p4, times[1:N] ./ 3600, metrics_eigen_cart.vel_uncertainty_errors, label="Eigen-based Sigma Points (Cartesian)",
                linewidth=2, color=:red, linestyle=:dashdot)
            plot!(p4, times[1:N] ./ 3600, metrics_eigen_ks.vel_uncertainty_errors, label="Eigen-based Sigma Points (KS)",
                linewidth=2, color=:orange, linestyle=:dashdotdot)
            plot!(p4, times[1:N] ./ 3600, metrics_lin_ks_sigma.vel_uncertainty_errors, label="Linearized KS Sigma Points",
                linewidth=2, color=:purple, linestyle=:dashdotdot)
            plot!(p4, times[1:N] ./ 3600, metrics_lin_ks_dyn.vel_uncertainty_errors, label="Linearized KS Dynamics",
                linewidth=2, color=:brown, linestyle=:dot)

            # Generate filename based on scenario
            filename = "figs/test_error_propagation_comparison_orb$(orbit_idx)_pos$(Int(σ_pos))m_orbits$(num_orbits).png"

            # Combine plots with increased left margin for y-axis labels
            # Use wider size and set margins explicitly to ensure y-labels are visible
            p_combined = plot(p1, p2, p3, p4, layout=(4, 1), size=(1000, 1600),
                left_margin=50Plots.px)
            savefig(p_combined, filename)
            println("  Saved plot to ", filename)

            # Store results
            push!(all_results, (
                orbit=orbit.name,
                σ_pos=σ_pos,
                σ_vel=σ_vel,
                num_orbits=num_orbits,
                metrics_lin_cart=metrics_lin_cart,
                metrics_ut_cart=metrics_ut_cart,
                metrics_eigen_cart=metrics_eigen_cart,
                metrics_eigen_ks=metrics_eigen_ks,
                metrics_lin_ks_sigma=metrics_lin_ks_sigma,
                metrics_lin_ks_dyn=metrics_lin_ks_dyn,
            ))

            println("\n" * "="^80)
            println("SCENARIO COMPLETE")
            println("="^80)
        end  # end num_orbits loop
    end  # end position uncertainty loop
end  # end orbit loop

println("\n" * "="^80)
println("ALL TESTS COMPLETE")
println("="^80)
println("\nSummary of all scenarios:")
for (idx, result) in enumerate(all_results)
    m_lin = result.metrics_lin_cart
    m_ut = result.metrics_ut_cart
    m_eigen_cart = result.metrics_eigen_cart
    m_eigen_ks = result.metrics_eigen_ks
    m_lin_ks_sigma = result.metrics_lin_ks_sigma
    m_lin_ks_dyn = result.metrics_lin_ks_dyn
    println("\nScenario $idx:")
    println("  Orbit: ", result.orbit)
    println("  Position uncertainty: ", result.σ_pos, " m")
    println("  Velocity uncertainty: ", result.σ_vel, " m/s")
    println("  Number of orbits: ", result.num_orbits)
    println("  Position errors:")
    println("    Linearized - RMSE: ", m_lin.pos_rmse, " m, Min: ", m_lin.pos_min, " m, Max: ", m_lin.pos_max, " m")
    println("    Unscented Transform - RMSE: ", m_ut.pos_rmse, " m, Min: ", m_ut.pos_min, " m, Max: ", m_ut.pos_max, " m")
    println("    Eigen-based Sigma Points (Cartesian) - RMSE: ", m_eigen_cart.pos_rmse, " m, Min: ", m_eigen_cart.pos_min, " m, Max: ", m_eigen_cart.pos_max, " m")
    println("    Eigen-based Sigma Points (KS) - RMSE: ", m_eigen_ks.pos_rmse, " m, Min: ", m_eigen_ks.pos_min, " m, Max: ", m_eigen_ks.pos_max, " m")
    println("    Linearized KS Sigma Points - RMSE: ", m_lin_ks_sigma.pos_rmse, " m, Min: ", m_lin_ks_sigma.pos_min, " m, Max: ", m_lin_ks_sigma.pos_max, " m")
    println("    Linearized KS Dynamics - RMSE: ", m_lin_ks_dyn.pos_rmse, " m, Min: ", m_lin_ks_dyn.pos_min, " m, Max: ", m_lin_ks_dyn.pos_max, " m")
    println("  Velocity errors:")
    println("    Linearized - RMSE: ", m_lin.vel_rmse, " m/s, Min: ", m_lin.vel_min, " m/s, Max: ", m_lin.vel_max, " m/s")
    println("    Unscented Transform - RMSE: ", m_ut.vel_rmse, " m/s, Min: ", m_ut.vel_min, " m/s, Max: ", m_ut.vel_max, " m/s")
    println("    Eigen-based Sigma Points (Cartesian) - RMSE: ", m_eigen_cart.vel_rmse, " m/s, Min: ", m_eigen_cart.vel_min, " m/s, Max: ", m_eigen_cart.vel_max, " m/s")
    println("    Eigen-based Sigma Points (KS) - RMSE: ", m_eigen_ks.vel_rmse, " m/s, Min: ", m_eigen_ks.vel_min, " m/s, Max: ", m_eigen_ks.vel_max, " m/s")
    println("    Linearized KS Sigma Points - RMSE: ", m_lin_ks_sigma.vel_rmse, " m/s, Min: ", m_lin_ks_sigma.vel_min, " m/s, Max: ", m_lin_ks_sigma.vel_max, " m/s")
    println("    Linearized KS Dynamics - RMSE: ", m_lin_ks_dyn.vel_rmse, " m/s, Min: ", m_lin_ks_dyn.vel_min, " m/s, Max: ", m_lin_ks_dyn.vel_max, " m/s")
    println("  Position uncertainty errors:")
    println("    Linearized - RMSE: ", m_lin.pos_uncertainty_rmse, " m, Min: ", m_lin.pos_uncertainty_min, " m, Max: ", m_lin.pos_uncertainty_max, " m")
    println("    Unscented Transform - RMSE: ", m_ut.pos_uncertainty_rmse, " m, Min: ", m_ut.pos_uncertainty_min, " m, Max: ", m_ut.pos_uncertainty_max, " m")
    println("    Eigen-based Sigma Points (Cartesian) - RMSE: ", m_eigen_cart.pos_uncertainty_rmse, " m, Min: ", m_eigen_cart.pos_uncertainty_min, " m, Max: ", m_eigen_cart.pos_uncertainty_max, " m")
    println("    Eigen-based Sigma Points (KS) - RMSE: ", m_eigen_ks.pos_uncertainty_rmse, " m, Min: ", m_eigen_ks.pos_uncertainty_min, " m, Max: ", m_eigen_ks.pos_uncertainty_max, " m")
    println("    Linearized KS Sigma Points - RMSE: ", m_lin_ks_sigma.pos_uncertainty_rmse, " m, Min: ", m_lin_ks_sigma.pos_uncertainty_min, " m, Max: ", m_lin_ks_sigma.pos_uncertainty_max, " m")
    println("    Linearized KS Dynamics - RMSE: ", m_lin_ks_dyn.pos_uncertainty_rmse, " m, Min: ", m_lin_ks_dyn.pos_uncertainty_min, " m, Max: ", m_lin_ks_dyn.pos_uncertainty_max, " m")
    println("  Velocity uncertainty errors:")
    println("    Linearized - RMSE: ", m_lin.vel_uncertainty_rmse, " m/s, Min: ", m_lin.vel_uncertainty_min, " m/s, Max: ", m_lin.vel_uncertainty_max, " m/s")
    println("    Unscented Transform - RMSE: ", m_ut.vel_uncertainty_rmse, " m/s, Min: ", m_ut.vel_uncertainty_min, " m/s, Max: ", m_ut.vel_uncertainty_max, " m/s")
    println("    Eigen-based Sigma Points (Cartesian) - RMSE: ", m_eigen_cart.vel_uncertainty_rmse, " m/s, Min: ", m_eigen_cart.vel_uncertainty_min, " m/s, Max: ", m_eigen_cart.vel_uncertainty_max, " m/s")
    println("    Eigen-based Sigma Points (KS) - RMSE: ", m_eigen_ks.vel_uncertainty_rmse, " m/s, Min: ", m_eigen_ks.vel_uncertainty_min, " m/s, Max: ", m_eigen_ks.vel_uncertainty_max, " m/s")
    println("    Linearized KS Sigma Points - RMSE: ", m_lin_ks_sigma.vel_uncertainty_rmse, " m/s, Min: ", m_lin_ks_sigma.vel_uncertainty_min, " m/s, Max: ", m_lin_ks_sigma.vel_uncertainty_max, " m/s")
    println("    Linearized KS Dynamics - RMSE: ", m_lin_ks_dyn.vel_uncertainty_rmse, " m/s, Min: ", m_lin_ks_dyn.vel_uncertainty_min, " m/s, Max: ", m_lin_ks_dyn.vel_uncertainty_max, " m/s")
end

