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

# Load shared configuration
include(joinpath(@__DIR__, "..", "config", "default.jl"))

# Local aliases (config uses UPPER_CASE constants)
test_orbits = TEST_ORBITS
position_uncertainties = POSITION_UNCERTAINTIES
num_orbits_list = NUM_ORBITS_LIST
num_samples = NUM_MC_SAMPLES

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
        σ_vel = compute_velocity_uncertainty(σ_pos, sma, r_vec_0, SIM_PARAMS.GM)

        println("\n" * "-"^80)
        println("POSITION UNCERTAINTY SCENARIO: σ_pos = ", σ_pos, " m")
        println("-"^80)
        println("  Position uncertainty: ", σ_pos, " m (1-sigma)")
        println("  Velocity uncertainty: ", σ_vel, " m/s (1-sigma)")

        P_0 = build_initial_covariance(σ_pos, σ_vel)

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

            # Helper function to wrap method calls and catch errors
            function run_method_safely(method_name, method_func, x_vec_0, P_0, times, sim_params, args...; kwargs...)
                println("\n" * "="^80)
                println(method_name)
                println("="^80)
                try
                    result = method_func(x_vec_0, P_0, times, sim_params, args...; kwargs...)
                    x_vec_traj, P_traj = result
                    println("  Completed: ", length(x_vec_traj), " states")
                    return (x_vec_traj=x_vec_traj, P_traj=P_traj, failed=false, error_msg=nothing, error=nothing)
                catch e
                    println("  ✗ FAILED: ", method_name)
                    println("  Error: ", typeof(e).name.name, ": ", sprint(showerror, e))
                    return (x_vec_traj=Vector{Vector{Float64}}(), P_traj=Vector{Matrix{Float64}}(), failed=true, error_msg=sprint(showerror, e), error=e)
                end
            end

            # Monte Carlo (ground truth) - don't catch errors for this one, we need it
            println("\n" * "="^80)
            println("1. MONTE CARLO (Ground Truth)")
            println("="^80)
            x_vec_traj_mean_mc, P_traj_mc = propagate_uncertainty_via_monte_carlo(x_vec_0, P_0, times, sim_params, num_samples)
            println("  Completed: ", length(x_vec_traj_mean_mc), " states")

            # Linearized Covariance Propagation (Cartesian)
            result_lin_cart = run_method_safely("2. LINEARIZED COVARIANCE PROPAGATION (Cartesian Coordinates)",
                propagate_uncertainty_via_linearized_cartesian_dynamics, x_vec_0, P_0, times, sim_params)
            x_vec_traj_mean_lin_cart = result_lin_cart.x_vec_traj
            P_traj_lin_cart = result_lin_cart.P_traj

            # Unscented Transform (Cartesian)
            result_ut_cart = run_method_safely("3. UNSCENTED TRANSFORM (Cartesian Coordinates)",
                propagate_uncertainty_via_cartesian_unscented_transform, x_vec_0, P_0, times, sim_params)
            x_vec_traj_mean_ut_cart = result_ut_cart.x_vec_traj
            P_traj_ut_cart = result_ut_cart.P_traj

            # Eigen-based Sigma Points (Cartesian)
            result_eigen_cart = run_method_safely("4. EIGEN-BASED SIGMA POINTS (Cartesian Coordinates)",
                propagate_uncertainty_via_cartesian_sigma_points, x_vec_0, P_0, times, sim_params)
            x_vec_traj_mean_eigen_cart = result_eigen_cart.x_vec_traj
            P_traj_eigen_cart = result_eigen_cart.P_traj

            # Eigen-based Sigma Points (KS)
            result_eigen_ks = run_method_safely("5. EIGEN-BASED SIGMA POINTS (KS Coordinates)",
                propagate_uncertainty_via_ks_sigma_points, x_vec_0, P_0, times, sim_params)
            x_vec_traj_mean_eigen_ks = result_eigen_ks.x_vec_traj
            P_traj_eigen_ks = result_eigen_ks.P_traj

            # Linearized KS Sigma Points
            result_lin_ks_sigma = run_method_safely("6. LINEARIZED KS SIGMA POINTS",
                propagate_uncertainty_via_linearized_ks_sigma_points, x_vec_0, P_0, times, sim_params)
            x_vec_traj_mean_lin_ks_sigma = result_lin_ks_sigma.x_vec_traj
            P_traj_lin_ks_sigma = result_lin_ks_sigma.P_traj

            # Linearized KS Dynamics
            result_lin_ks_dyn = run_method_safely("7. LINEARIZED KS DYNAMICS",
                propagate_uncertainty_via_linearized_ks_dynamics, x_vec_0, P_0, times, sim_params)
            x_vec_traj_mean_lin_ks_dyn = result_lin_ks_dyn.x_vec_traj
            P_traj_lin_ks_dyn = result_lin_ks_dyn.P_traj

            # Supervisor-spec: MC energy binning -> per-bin linearized KS sigma points -> per-step sampling aggregation
            result_mc_binned_ks = run_method_safely("8. MC ENERGY-BINNED + KS SIGMA POINTS + AGGREGATION SAMPLING",
                propagate_uncertainty_via_energy_binned_mc_then_ks_sigma_points, x_vec_0, P_0, times, sim_params;
                num_mc_samples=5000, num_energy_bins=10)
            x_vec_traj_mean_mc_binned_ks = result_mc_binned_ks.x_vec_traj
            P_traj_mc_binned_ks = result_mc_binned_ks.P_traj

            # Compare against Monte Carlo
            println("\n" * "="^80)
            println("COMPARISON AGAINST MONTE CARLO GROUND TRUTH")
            println("="^80)

            # Compute error metrics using the new function (only for successful methods)
            metrics_lin_cart = result_lin_cart.failed ? nothing : error_metrics(x_vec_traj_mean_mc, P_traj_mc, x_vec_traj_mean_lin_cart, P_traj_lin_cart)
            metrics_ut_cart = result_ut_cart.failed ? nothing : error_metrics(x_vec_traj_mean_mc, P_traj_mc, x_vec_traj_mean_ut_cart, P_traj_ut_cart)
            metrics_eigen_cart = result_eigen_cart.failed ? nothing : error_metrics(x_vec_traj_mean_mc, P_traj_mc, x_vec_traj_mean_eigen_cart, P_traj_eigen_cart)
            metrics_eigen_ks = result_eigen_ks.failed ? nothing : error_metrics(x_vec_traj_mean_mc, P_traj_mc, x_vec_traj_mean_eigen_ks, P_traj_eigen_ks)
            metrics_lin_ks_sigma = result_lin_ks_sigma.failed ? nothing : error_metrics(x_vec_traj_mean_mc, P_traj_mc, x_vec_traj_mean_lin_ks_sigma, P_traj_lin_ks_sigma)
            metrics_lin_ks_dyn = result_lin_ks_dyn.failed ? nothing : error_metrics(x_vec_traj_mean_mc, P_traj_mc, x_vec_traj_mean_lin_ks_dyn, P_traj_lin_ks_dyn)
            metrics_mc_binned_ks = result_mc_binned_ks.failed ? nothing : error_metrics(x_vec_traj_mean_mc, P_traj_mc, x_vec_traj_mean_mc_binned_ks, P_traj_mc_binned_ks)

            # Compute N based only on successful methods
            lengths_to_check = [length(x_vec_traj_mean_mc)]
            if !result_lin_cart.failed
                push!(lengths_to_check, length(x_vec_traj_mean_lin_cart))
            end
            if !result_ut_cart.failed
                push!(lengths_to_check, length(x_vec_traj_mean_ut_cart))
            end
            if !result_eigen_cart.failed
                push!(lengths_to_check, length(x_vec_traj_mean_eigen_cart))
            end
            if !result_eigen_ks.failed
                push!(lengths_to_check, length(x_vec_traj_mean_eigen_ks))
            end
            if !result_lin_ks_sigma.failed
                push!(lengths_to_check, length(x_vec_traj_mean_lin_ks_sigma))
            end
            if !result_lin_ks_dyn.failed
                push!(lengths_to_check, length(x_vec_traj_mean_lin_ks_dyn))
            end
            if !result_mc_binned_ks.failed
                push!(lengths_to_check, length(x_vec_traj_mean_mc_binned_ks))
            end
            N = minimum(lengths_to_check)

            # Print position errors
            println("\nPosition errors (vs Monte Carlo):")
            if metrics_lin_cart !== nothing
                println("  Linearized Covariance Propagation (Cartesian):")
                println("    RMSE = ", metrics_lin_cart.pos_rmse, " m, Min = ", metrics_lin_cart.pos_min, " m, Max = ", metrics_lin_cart.pos_max, " m")
            else
                println("  Linearized Covariance Propagation (Cartesian): FAILED")
            end
            if metrics_ut_cart !== nothing
                println("  Unscented Transform (Cartesian):")
                println("    RMSE = ", metrics_ut_cart.pos_rmse, " m, Min = ", metrics_ut_cart.pos_min, " m, Max = ", metrics_ut_cart.pos_max, " m")
            else
                println("  Unscented Transform (Cartesian): FAILED")
            end
            if metrics_eigen_cart !== nothing
                println("  Eigen-based Sigma Points (Cartesian):")
                println("    RMSE = ", metrics_eigen_cart.pos_rmse, " m, Min = ", metrics_eigen_cart.pos_min, " m, Max = ", metrics_eigen_cart.pos_max, " m")
            else
                println("  Eigen-based Sigma Points (Cartesian): FAILED")
            end
            if metrics_eigen_ks !== nothing
                println("  Eigen-based Sigma Points (KS):")
                println("    RMSE = ", metrics_eigen_ks.pos_rmse, " m, Min = ", metrics_eigen_ks.pos_min, " m, Max = ", metrics_eigen_ks.pos_max, " m")
            else
                println("  Eigen-based Sigma Points (KS): FAILED")
            end
            if metrics_lin_ks_sigma !== nothing
                println("  Linearized KS Sigma Points:")
                println("    RMSE = ", metrics_lin_ks_sigma.pos_rmse, " m, Min = ", metrics_lin_ks_sigma.pos_min, " m, Max = ", metrics_lin_ks_sigma.pos_max, " m")
            else
                println("  Linearized KS Sigma Points: FAILED")
            end
            if metrics_lin_ks_dyn !== nothing
                println("  Linearized KS Dynamics:")
                println("    RMSE = ", metrics_lin_ks_dyn.pos_rmse, " m, Min = ", metrics_lin_ks_dyn.pos_min, " m, Max = ", metrics_lin_ks_dyn.pos_max, " m")
            else
                println("  Linearized KS Dynamics: FAILED")
            end
            if metrics_mc_binned_ks !== nothing
                println("  MC energy-binned + linKS sigma + sampling aggregation:")
                println("    RMSE = ", metrics_mc_binned_ks.pos_rmse, " m, Min = ", metrics_mc_binned_ks.pos_min, " m, Max = ", metrics_mc_binned_ks.pos_max, " m")
            else
                println("  MC energy-binned + linKS sigma + sampling aggregation: FAILED")
            end

            # Print velocity errors
            println("\nVelocity errors (vs Monte Carlo):")
            if metrics_lin_cart !== nothing
                println("  Linearized Covariance Propagation (Cartesian):")
                println("    RMSE = ", metrics_lin_cart.vel_rmse, " m/s, Min = ", metrics_lin_cart.vel_min, " m/s, Max = ", metrics_lin_cart.vel_max, " m/s")
            else
                println("  Linearized Covariance Propagation (Cartesian): FAILED")
            end
            if metrics_ut_cart !== nothing
                println("  Unscented Transform (Cartesian):")
                println("    RMSE = ", metrics_ut_cart.vel_rmse, " m/s, Min = ", metrics_ut_cart.vel_min, " m/s, Max = ", metrics_ut_cart.vel_max, " m/s")
            else
                println("  Unscented Transform (Cartesian): FAILED")
            end
            if metrics_eigen_cart !== nothing
                println("  Eigen-based Sigma Points (Cartesian):")
                println("    RMSE = ", metrics_eigen_cart.vel_rmse, " m/s, Min = ", metrics_eigen_cart.vel_min, " m/s, Max = ", metrics_eigen_cart.vel_max, " m/s")
            else
                println("  Eigen-based Sigma Points (Cartesian): FAILED")
            end
            if metrics_eigen_ks !== nothing
                println("  Eigen-based Sigma Points (KS):")
                println("    RMSE = ", metrics_eigen_ks.vel_rmse, " m/s, Min = ", metrics_eigen_ks.vel_min, " m/s, Max = ", metrics_eigen_ks.vel_max, " m/s")
            else
                println("  Eigen-based Sigma Points (KS): FAILED")
            end
            if metrics_lin_ks_sigma !== nothing
                println("  Linearized KS Sigma Points:")
                println("    RMSE = ", metrics_lin_ks_sigma.vel_rmse, " m/s, Min = ", metrics_lin_ks_sigma.vel_min, " m/s, Max = ", metrics_lin_ks_sigma.vel_max, " m/s")
            else
                println("  Linearized KS Sigma Points: FAILED")
            end
            if metrics_lin_ks_dyn !== nothing
                println("  Linearized KS Dynamics:")
                println("    RMSE = ", metrics_lin_ks_dyn.vel_rmse, " m/s, Min = ", metrics_lin_ks_dyn.vel_min, " m/s, Max = ", metrics_lin_ks_dyn.vel_max, " m/s")
            else
                println("  Linearized KS Dynamics: FAILED")
            end
            if metrics_mc_binned_ks !== nothing
                println("  MC energy-binned + linKS sigma + sampling aggregation:")
                println("    RMSE = ", metrics_mc_binned_ks.vel_rmse, " m/s, Min = ", metrics_mc_binned_ks.vel_min, " m/s, Max = ", metrics_mc_binned_ks.vel_max, " m/s")
            else
                println("  MC energy-binned + linKS sigma + sampling aggregation: FAILED")
            end

            # Print position and velocity uncertainty errors
            println("\nPosition uncertainty errors (vs Monte Carlo):")
            if metrics_lin_cart !== nothing
                println("  Linearized Covariance Propagation (Cartesian):")
                println("    RMSE = ", metrics_lin_cart.pos_uncertainty_rmse, " m, Min = ", metrics_lin_cart.pos_uncertainty_min, " m, Max = ", metrics_lin_cart.pos_uncertainty_max, " m")
            else
                println("  Linearized Covariance Propagation (Cartesian): FAILED")
            end
            if metrics_ut_cart !== nothing
                println("  Unscented Transform (Cartesian):")
                println("    RMSE = ", metrics_ut_cart.pos_uncertainty_rmse, " m, Min = ", metrics_ut_cart.pos_uncertainty_min, " m, Max = ", metrics_ut_cart.pos_uncertainty_max, " m")
            else
                println("  Unscented Transform (Cartesian): FAILED")
            end
            if metrics_eigen_cart !== nothing
                println("  Eigen-based Sigma Points (Cartesian):")
                println("    RMSE = ", metrics_eigen_cart.pos_uncertainty_rmse, " m, Min = ", metrics_eigen_cart.pos_uncertainty_min, " m, Max = ", metrics_eigen_cart.pos_uncertainty_max, " m")
            else
                println("  Eigen-based Sigma Points (Cartesian): FAILED")
            end
            if metrics_eigen_ks !== nothing
                println("  Eigen-based Sigma Points (KS):")
                println("    RMSE = ", metrics_eigen_ks.pos_uncertainty_rmse, " m, Min = ", metrics_eigen_ks.pos_uncertainty_min, " m, Max = ", metrics_eigen_ks.pos_uncertainty_max, " m")
            else
                println("  Eigen-based Sigma Points (KS): FAILED")
            end
            if metrics_lin_ks_sigma !== nothing
                println("  Linearized KS Sigma Points:")
                println("    RMSE = ", metrics_lin_ks_sigma.pos_uncertainty_rmse, " m, Min = ", metrics_lin_ks_sigma.pos_uncertainty_min, " m, Max = ", metrics_lin_ks_sigma.pos_uncertainty_max, " m")
            else
                println("  Linearized KS Sigma Points: FAILED")
            end
            if metrics_lin_ks_dyn !== nothing
                println("  Linearized KS Dynamics:")
                println("    RMSE = ", metrics_lin_ks_dyn.pos_uncertainty_rmse, " m, Min = ", metrics_lin_ks_dyn.pos_uncertainty_min, " m, Max = ", metrics_lin_ks_dyn.pos_uncertainty_max, " m")
            else
                println("  Linearized KS Dynamics: FAILED")
            end
            if metrics_mc_binned_ks !== nothing
                println("  MC energy-binned + linKS sigma + sampling aggregation:")
                println("    RMSE = ", metrics_mc_binned_ks.pos_uncertainty_rmse, " m, Min = ", metrics_mc_binned_ks.pos_uncertainty_min, " m, Max = ", metrics_mc_binned_ks.pos_uncertainty_max, " m")
            else
                println("  MC energy-binned + linKS sigma + sampling aggregation: FAILED")
            end

            println("\nVelocity uncertainty errors (vs Monte Carlo):")
            if metrics_lin_cart !== nothing
                println("  Linearized Covariance Propagation (Cartesian):")
                println("    RMSE = ", metrics_lin_cart.vel_uncertainty_rmse, " m/s, Min = ", metrics_lin_cart.vel_uncertainty_min, " m/s, Max = ", metrics_lin_cart.vel_uncertainty_max, " m/s")
            else
                println("  Linearized Covariance Propagation (Cartesian): FAILED")
            end
            if metrics_ut_cart !== nothing
                println("  Unscented Transform (Cartesian):")
                println("    RMSE = ", metrics_ut_cart.vel_uncertainty_rmse, " m/s, Min = ", metrics_ut_cart.vel_uncertainty_min, " m/s, Max = ", metrics_ut_cart.vel_uncertainty_max, " m/s")
            else
                println("  Unscented Transform (Cartesian): FAILED")
            end
            if metrics_eigen_cart !== nothing
                println("  Eigen-based Sigma Points (Cartesian):")
                println("    RMSE = ", metrics_eigen_cart.vel_uncertainty_rmse, " m/s, Min = ", metrics_eigen_cart.vel_uncertainty_min, " m/s, Max = ", metrics_eigen_cart.vel_uncertainty_max, " m/s")
            else
                println("  Eigen-based Sigma Points (Cartesian): FAILED")
            end
            if metrics_eigen_ks !== nothing
                println("  Eigen-based Sigma Points (KS):")
                println("    RMSE = ", metrics_eigen_ks.vel_uncertainty_rmse, " m/s, Min = ", metrics_eigen_ks.vel_uncertainty_min, " m/s, Max = ", metrics_eigen_ks.vel_uncertainty_max, " m/s")
            else
                println("  Eigen-based Sigma Points (KS): FAILED")
            end
            if metrics_lin_ks_sigma !== nothing
                println("  Linearized KS Sigma Points:")
                println("    RMSE = ", metrics_lin_ks_sigma.vel_uncertainty_rmse, " m/s, Min = ", metrics_lin_ks_sigma.vel_uncertainty_min, " m/s, Max = ", metrics_lin_ks_sigma.vel_uncertainty_max, " m/s")
            else
                println("  Linearized KS Sigma Points: FAILED")
            end
            if metrics_lin_ks_dyn !== nothing
                println("  Linearized KS Dynamics:")
                println("    RMSE = ", metrics_lin_ks_dyn.vel_uncertainty_rmse, " m/s, Min = ", metrics_lin_ks_dyn.vel_uncertainty_min, " m/s, Max = ", metrics_lin_ks_dyn.vel_uncertainty_max, " m/s")
            else
                println("  Linearized KS Dynamics: FAILED")
            end
            if metrics_mc_binned_ks !== nothing
                println("  MC energy-binned + linKS sigma + sampling aggregation:")
                println("    RMSE = ", metrics_mc_binned_ks.vel_uncertainty_rmse, " m/s, Min = ", metrics_mc_binned_ks.vel_uncertainty_min, " m/s, Max = ", metrics_mc_binned_ks.vel_uncertainty_max, " m/s")
            else
                println("  MC energy-binned + linKS sigma + sampling aggregation: FAILED")
            end

            # Create plots
            println("\nGenerating plots...")

            # KL divergence vs Monte Carlo (Gaussian approximation)
            # Use the same N horizon as other metrics (min length across methods)
            # Only compute for successful methods
            kl_lin_cart = metrics_lin_cart !== nothing ? [gaussian_kl_divergence("2. LINEARIZED COVARIANCE PROPAGATION (Cartesian Coordinates)", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_lin_cart[i], P_traj_lin_cart[i]) for i in 1:N] : nothing
            kl_ut_cart = metrics_ut_cart !== nothing ? [gaussian_kl_divergence("3. UNSCENTED TRANSFORM (Cartesian Coordinates)", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_ut_cart[i], P_traj_ut_cart[i]) for i in 1:N] : nothing
            kl_eigen_cart = metrics_eigen_cart !== nothing ? [gaussian_kl_divergence("4. EIGEN-BASED SIGMA POINTS (Cartesian Coordinates)", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_eigen_cart[i], P_traj_eigen_cart[i]) for i in 1:N] : nothing
            kl_eigen_ks = metrics_eigen_ks !== nothing ? [gaussian_kl_divergence("5. EIGEN-BASED SIGMA POINTS (KS Coordinates)", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_eigen_ks[i], P_traj_eigen_ks[i]) for i in 1:N] : nothing
            kl_lin_ks_sigma = metrics_lin_ks_sigma !== nothing ? [gaussian_kl_divergence("6. LINEARIZED KS SIGMA POINTS", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_lin_ks_sigma[i], P_traj_lin_ks_sigma[i]) for i in 1:N] : nothing
            kl_lin_ks_dyn = metrics_lin_ks_dyn !== nothing ? [gaussian_kl_divergence("7. LINEARIZED KS DYNAMICS", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_lin_ks_dyn[i], P_traj_lin_ks_dyn[i]) for i in 1:N] : nothing
            kl_mc_binned_ks = metrics_mc_binned_ks !== nothing ? [gaussian_kl_divergence("8. MC ENERGY-BINNED + KS SIGMA + AGG SAMPLING", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_mc_binned_ks[i], P_traj_mc_binned_ks[i]) for i in 1:N] : nothing

            # Plot 1: Position error
            p1 = plot(xlabel="Time (hours)", ylabel="Position Error (m)", yscale=:log10, legend=:topleft)
            if metrics_lin_cart !== nothing
                plot!(p1, times[1:N] ./ 3600, metrics_lin_cart.pos_errors, label="Linearized Covariance",
                    linewidth=2, color=:green, linestyle=:dash)
            end
            if metrics_ut_cart !== nothing
                plot!(p1, times[1:N] ./ 3600, metrics_ut_cart.pos_errors, label="Unscented Transform",
                    linewidth=2, color=:blue, linestyle=:dot)
            end
            if metrics_eigen_cart !== nothing
                plot!(p1, times[1:N] ./ 3600, metrics_eigen_cart.pos_errors, label="Eigen-based Sigma Points (Cartesian)",
                    linewidth=2, color=:red, linestyle=:dashdot)
            end
            if metrics_eigen_ks !== nothing
                plot!(p1, times[1:N] ./ 3600, metrics_eigen_ks.pos_errors, label="Eigen-based Sigma Points (KS)",
                    linewidth=2, color=:orange, linestyle=:dashdotdot)
            end
            if metrics_lin_ks_sigma !== nothing
                plot!(p1, times[1:N] ./ 3600, metrics_lin_ks_sigma.pos_errors, label="Linearized KS Sigma Points",
                    linewidth=2, color=:purple, linestyle=:dashdotdot)
            end
            if metrics_lin_ks_dyn !== nothing
                plot!(p1, times[1:N] ./ 3600, metrics_lin_ks_dyn.pos_errors, label="Linearized KS Dynamics",
                    linewidth=2, color=:brown, linestyle=:dot)
            end
            if metrics_mc_binned_ks !== nothing
                plot!(p1, times[1:N] ./ 3600, metrics_mc_binned_ks.pos_errors, label="MC Energy-binned + KS Sigma + Sampling Aggregation",
                    linewidth=2, color=:black, linestyle=:dash)
            end

            # Plot 2: Velocity error
            p2 = plot(xlabel="Time (hours)", ylabel="Velocity Error (m/s)", yscale=:log10, legend=:topleft)
            if metrics_lin_cart !== nothing
                plot!(p2, times[1:N] ./ 3600, metrics_lin_cart.vel_errors, label="Linearized Covariance",
                    linewidth=2, color=:green, linestyle=:dash)
            end
            if metrics_ut_cart !== nothing
                plot!(p2, times[1:N] ./ 3600, metrics_ut_cart.vel_errors, label="Unscented Transform",
                    linewidth=2, color=:blue, linestyle=:dot)
            end
            if metrics_eigen_cart !== nothing
                plot!(p2, times[1:N] ./ 3600, metrics_eigen_cart.vel_errors, label="Eigen-based Sigma Points (Cartesian)",
                    linewidth=2, color=:red, linestyle=:dashdot)
            end
            if metrics_eigen_ks !== nothing
                plot!(p2, times[1:N] ./ 3600, metrics_eigen_ks.vel_errors, label="Eigen-based Sigma Points (KS)",
                    linewidth=2, color=:orange, linestyle=:dashdotdot)
            end
            if metrics_lin_ks_sigma !== nothing
                plot!(p2, times[1:N] ./ 3600, metrics_lin_ks_sigma.vel_errors, label="Linearized KS Sigma Points",
                    linewidth=2, color=:purple, linestyle=:dashdotdot)
            end
            if metrics_lin_ks_dyn !== nothing
                plot!(p2, times[1:N] ./ 3600, metrics_lin_ks_dyn.vel_errors, label="Linearized KS Dynamics",
                    linewidth=2, color=:brown, linestyle=:dot)
            end
            if metrics_mc_binned_ks !== nothing
                plot!(p2, times[1:N] ./ 3600, metrics_mc_binned_ks.vel_errors, label="MC Energy-binned + KS Sigma + Sampling Aggregation",
                    linewidth=2, color=:black, linestyle=:dash)
            end

            # Plot 3: Position uncertainty error
            p3 = plot(xlabel="Time (hours)", ylabel="Position Uncertainty Error (m)", yscale=:log10, legend=:topleft)
            if metrics_lin_cart !== nothing
                plot!(p3, times[1:N] ./ 3600, metrics_lin_cart.pos_uncertainty_errors, label="Linearized Covariance",
                    linewidth=2, color=:green, linestyle=:dash)
            end
            if metrics_ut_cart !== nothing
                plot!(p3, times[1:N] ./ 3600, metrics_ut_cart.pos_uncertainty_errors, label="Unscented Transform",
                    linewidth=2, color=:blue, linestyle=:dot)
            end
            if metrics_eigen_cart !== nothing
                plot!(p3, times[1:N] ./ 3600, metrics_eigen_cart.pos_uncertainty_errors, label="Eigen-based Sigma Points (Cartesian)",
                    linewidth=2, color=:red, linestyle=:dashdot)
            end
            if metrics_eigen_ks !== nothing
                plot!(p3, times[1:N] ./ 3600, metrics_eigen_ks.pos_uncertainty_errors, label="Eigen-based Sigma Points (KS)",
                    linewidth=2, color=:orange, linestyle=:dashdotdot)
            end
            if metrics_lin_ks_sigma !== nothing
                plot!(p3, times[1:N] ./ 3600, metrics_lin_ks_sigma.pos_uncertainty_errors, label="Linearized KS Sigma Points",
                    linewidth=2, color=:purple, linestyle=:dashdotdot)
            end
            if metrics_lin_ks_dyn !== nothing
                plot!(p3, times[1:N] ./ 3600, metrics_lin_ks_dyn.pos_uncertainty_errors, label="Linearized KS Dynamics",
                    linewidth=2, color=:brown, linestyle=:dot)
            end
            if metrics_mc_binned_ks !== nothing
                plot!(p3, times[1:N] ./ 3600, metrics_mc_binned_ks.pos_uncertainty_errors, label="MC Energy-binned + KS Sigma + Sampling Aggregation",
                    linewidth=2, color=:black, linestyle=:dash)
            end

            # Plot 4: Velocity uncertainty error
            p4 = plot(xlabel="Time (hours)", ylabel="Velocity Uncertainty Error (m/s)", yscale=:log10, legend=:topleft)
            if metrics_lin_cart !== nothing
                plot!(p4, times[1:N] ./ 3600, metrics_lin_cart.vel_uncertainty_errors, label="Linearized Covariance",
                    linewidth=2, color=:green, linestyle=:dash)
            end
            if metrics_ut_cart !== nothing
                plot!(p4, times[1:N] ./ 3600, metrics_ut_cart.vel_uncertainty_errors, label="Unscented Transform",
                    linewidth=2, color=:blue, linestyle=:dot)
            end
            if metrics_eigen_cart !== nothing
                plot!(p4, times[1:N] ./ 3600, metrics_eigen_cart.vel_uncertainty_errors, label="Eigen-based Sigma Points (Cartesian)",
                    linewidth=2, color=:red, linestyle=:dashdot)
            end
            if metrics_eigen_ks !== nothing
                plot!(p4, times[1:N] ./ 3600, metrics_eigen_ks.vel_uncertainty_errors, label="Eigen-based Sigma Points (KS)",
                    linewidth=2, color=:orange, linestyle=:dashdotdot)
            end
            if metrics_lin_ks_sigma !== nothing
                plot!(p4, times[1:N] ./ 3600, metrics_lin_ks_sigma.vel_uncertainty_errors, label="Linearized KS Sigma Points",
                    linewidth=2, color=:purple, linestyle=:dashdotdot)
            end
            if metrics_lin_ks_dyn !== nothing
                plot!(p4, times[1:N] ./ 3600, metrics_lin_ks_dyn.vel_uncertainty_errors, label="Linearized KS Dynamics",
                    linewidth=2, color=:brown, linestyle=:dot)
            end
            if metrics_mc_binned_ks !== nothing
                plot!(p4, times[1:N] ./ 3600, metrics_mc_binned_ks.vel_uncertainty_errors, label="MC Energy-binned + KS Sigma + Sampling Aggregation",
                    linewidth=2, color=:black, linestyle=:dash)
            end

            # Plot 5: KL divergence (Gaussian approx) vs Monte Carlo
            p5 = plot(xlabel="Time (hours)", ylabel="KL Divergence", yscale=:log10, legend=:topleft)
            if kl_lin_cart !== nothing
                plot!(p5, times[1:N] ./ 3600, kl_lin_cart, label="Linearized Covariance",
                    linewidth=2, color=:green, linestyle=:dash)
            end
            if kl_ut_cart !== nothing
                plot!(p5, times[1:N] ./ 3600, kl_ut_cart, label="Unscented Transform",
                    linewidth=2, color=:blue, linestyle=:dot)
            end
            if kl_eigen_cart !== nothing
                plot!(p5, times[1:N] ./ 3600, kl_eigen_cart, label="Eigen-based Sigma Points (Cartesian)",
                    linewidth=2, color=:red, linestyle=:dashdot)
            end
            if kl_eigen_ks !== nothing
                plot!(p5, times[1:N] ./ 3600, kl_eigen_ks, label="Eigen-based Sigma Points (KS)",
                    linewidth=2, color=:orange, linestyle=:dashdotdot)
            end
            if kl_lin_ks_sigma !== nothing
                plot!(p5, times[1:N] ./ 3600, kl_lin_ks_sigma, label="Linearized KS Sigma Points",
                    linewidth=2, color=:purple, linestyle=:dashdotdot)
            end
            if kl_lin_ks_dyn !== nothing
                plot!(p5, times[1:N] ./ 3600, kl_lin_ks_dyn, label="Linearized KS Dynamics",
                    linewidth=2, color=:brown, linestyle=:dot)
            end
            if kl_mc_binned_ks !== nothing
                plot!(p5, times[1:N] ./ 3600, kl_mc_binned_ks, label="MC Energy-binned + KS Sigma + Sampling Aggregation",
                    linewidth=2, color=:black, linestyle=:dash)
            end

            # Generate filename based on scenario
            filename = "figs/test_error_propagation_comparison_$(orbit.id)_num_orbits$(Int(num_orbits))_std_pos$(Int(σ_pos))m_std_vel$(round(σ_vel, digits=6))mps_num_samples$(Int(num_samples)).png"

            # Combine plots with increased left margin for y-axis labels
            # Use wider size and set margins explicitly to ensure y-labels are visible
            p_combined = plot(p1, p2, p3, p4, p5, layout=(5, 1), size=(1000, 2000), left_margin=50Plots.px)
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
                metrics_mc_binned_ks=metrics_mc_binned_ks,
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
    m_mc_binned_ks = result.metrics_mc_binned_ks
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
    println("    MC Energy-binned + KS Sigma + Sampling Aggregation - RMSE: ", m_mc_binned_ks.pos_rmse, " m, Min: ", m_mc_binned_ks.pos_min, " m, Max: ", m_mc_binned_ks.pos_max, " m")
    println("  Velocity errors:")
    println("    Linearized - RMSE: ", m_lin.vel_rmse, " m/s, Min: ", m_lin.vel_min, " m/s, Max: ", m_lin.vel_max, " m/s")
    println("    Unscented Transform - RMSE: ", m_ut.vel_rmse, " m/s, Min: ", m_ut.vel_min, " m/s, Max: ", m_ut.vel_max, " m/s")
    println("    Eigen-based Sigma Points (Cartesian) - RMSE: ", m_eigen_cart.vel_rmse, " m/s, Min: ", m_eigen_cart.vel_min, " m/s, Max: ", m_eigen_cart.vel_max, " m/s")
    println("    Eigen-based Sigma Points (KS) - RMSE: ", m_eigen_ks.vel_rmse, " m/s, Min: ", m_eigen_ks.vel_min, " m/s, Max: ", m_eigen_ks.vel_max, " m/s")
    println("    Linearized KS Sigma Points - RMSE: ", m_lin_ks_sigma.vel_rmse, " m/s, Min: ", m_lin_ks_sigma.vel_min, " m/s, Max: ", m_lin_ks_sigma.vel_max, " m/s")
    println("    Linearized KS Dynamics - RMSE: ", m_lin_ks_dyn.vel_rmse, " m/s, Min: ", m_lin_ks_dyn.vel_min, " m/s, Max: ", m_lin_ks_dyn.vel_max, " m/s")
    println("    MC Energy-binned + KS Sigma + Sampling Aggregation - RMSE: ", m_mc_binned_ks.vel_rmse, " m/s, Min: ", m_mc_binned_ks.vel_min, " m/s, Max: ", m_mc_binned_ks.vel_max, " m/s")
    println("  Position uncertainty errors:")
    println("    Linearized - RMSE: ", m_lin.pos_uncertainty_rmse, " m, Min: ", m_lin.pos_uncertainty_min, " m, Max: ", m_lin.pos_uncertainty_max, " m")
    println("    Unscented Transform - RMSE: ", m_ut.pos_uncertainty_rmse, " m, Min: ", m_ut.pos_uncertainty_min, " m, Max: ", m_ut.pos_uncertainty_max, " m")
    println("    Eigen-based Sigma Points (Cartesian) - RMSE: ", m_eigen_cart.pos_uncertainty_rmse, " m, Min: ", m_eigen_cart.pos_uncertainty_min, " m, Max: ", m_eigen_cart.pos_uncertainty_max, " m")
    println("    Eigen-based Sigma Points (KS) - RMSE: ", m_eigen_ks.pos_uncertainty_rmse, " m, Min: ", m_eigen_ks.pos_uncertainty_min, " m, Max: ", m_eigen_ks.pos_uncertainty_max, " m")
    println("    Linearized KS Sigma Points - RMSE: ", m_lin_ks_sigma.pos_uncertainty_rmse, " m, Min: ", m_lin_ks_sigma.pos_uncertainty_min, " m, Max: ", m_lin_ks_sigma.pos_uncertainty_max, " m")
    println("    Linearized KS Dynamics - RMSE: ", m_lin_ks_dyn.pos_uncertainty_rmse, " m, Min: ", m_lin_ks_dyn.pos_uncertainty_min, " m, Max: ", m_lin_ks_dyn.pos_uncertainty_max, " m")
    println("    MC Energy-binned + KS Sigma + Sampling Aggregation - RMSE: ", m_mc_binned_ks.pos_uncertainty_rmse, " m, Min: ", m_mc_binned_ks.pos_uncertainty_min, " m, Max: ", m_mc_binned_ks.pos_uncertainty_max, " m")
    println("  Velocity uncertainty errors:")
    println("    Linearized - RMSE: ", m_lin.vel_uncertainty_rmse, " m/s, Min: ", m_lin.vel_uncertainty_min, " m/s, Max: ", m_lin.vel_uncertainty_max, " m/s")
    println("    Unscented Transform - RMSE: ", m_ut.vel_uncertainty_rmse, " m/s, Min: ", m_ut.vel_uncertainty_min, " m/s, Max: ", m_ut.vel_uncertainty_max, " m/s")
    println("    Eigen-based Sigma Points (Cartesian) - RMSE: ", m_eigen_cart.vel_uncertainty_rmse, " m/s, Min: ", m_eigen_cart.vel_uncertainty_min, " m/s, Max: ", m_eigen_cart.vel_uncertainty_max, " m/s")
    println("    Eigen-based Sigma Points (KS) - RMSE: ", m_eigen_ks.vel_uncertainty_rmse, " m/s, Min: ", m_eigen_ks.vel_uncertainty_min, " m/s, Max: ", m_eigen_ks.vel_uncertainty_max, " m/s")
    println("    Linearized KS Sigma Points - RMSE: ", m_lin_ks_sigma.vel_uncertainty_rmse, " m/s, Min: ", m_lin_ks_sigma.vel_uncertainty_min, " m/s, Max: ", m_lin_ks_sigma.vel_uncertainty_max, " m/s")
    println("    Linearized KS Dynamics - RMSE: ", m_lin_ks_dyn.vel_uncertainty_rmse, " m/s, Min: ", m_lin_ks_dyn.vel_uncertainty_min, " m/s, Max: ", m_lin_ks_dyn.vel_uncertainty_max, " m/s")
    println("    MC Energy-binned + KS Sigma + Sampling Aggregation - RMSE: ", m_mc_binned_ks.vel_uncertainty_rmse, " m/s, Min: ", m_mc_binned_ks.vel_uncertainty_min, " m/s, Max: ", m_mc_binned_ks.vel_uncertainty_max, " m/s")
end

