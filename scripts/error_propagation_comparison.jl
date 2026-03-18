using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots
using LinearAlgebra
using SatelliteDynamics
using DifferentialEquations
using NPZ

const SD = SatelliteDynamics

include("../src/cartesian_dynamics.jl")
include("../src/ks_dynamics.jl")
include("../src/error_propagation.jl")

# Load configuration (default or user-specified, e.g. config/sanity_check.jl)
if length(ARGS) >= 1
    config_path = ARGS[1]
    if !isabspath(config_path)
        config_path = joinpath(@__DIR__, "..", config_path)
    end
    println("Loading config: ", config_path)
    include(config_path)
else
    include(joinpath(@__DIR__, "..", "config", "default.jl"))
end

# Save approach results to out/ (default: false). Usage: julia script.jl [config.jl] [save]
save_to_out = length(ARGS) >= 2 && (ARGS[2] == "save" || lowercase(ARGS[2]) == "true")

using Random
Random.seed!(RANDOM_SEED)

# Local aliases (config uses UPPER_CASE constants)
test_orbits = TEST_ORBITS
oe_initial_std_scenarios = OE_INITIAL_STD_SCENARIOS
num_orbits_list = NUM_ORBITS_LIST
num_samples = NUM_MC_SAMPLES
num_mc_samples_binning = NUM_MC_SAMPLES_BINNING
num_energy_bins = NUM_ENERGY_BINS
min_samples_threshold = MIN_SAMPLES_THRESHOLD

# Number string for filenames (e.g. 1000, 1e-5)
fname_num(x) = isinteger(x) ? string(Int(x)) : string(x)

println("="^80)
println("ERROR PROPAGATION COMPARISON TEST")
println("="^80)
println("Comparing error propagation methods against Monte Carlo ground truth")
println("\nTest configuration:")
println("  Number of orbits: ", num_orbits_list)
println("  OE initial std scenarios: ", length(oe_initial_std_scenarios))
println("  Number of test scenarios: ", length(test_orbits) * length(oe_initial_std_scenarios) * length(num_orbits_list))
println("  Save results to out/: ", save_to_out)

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
    oe_vec_0 = SD.sCARTtoOSC(x_vec_0; GM=SIM_PARAMS.GM, use_degrees=false)
    r_vec_0 = x_vec_0[1:3]
    v_vec_0 = x_vec_0[4:6]

    println("\nInitial conditions:")
    println("  Position: ", r_vec_0)
    println("  Velocity: ", v_vec_0)
    println("  Radius: ", norm(r_vec_0) / 1e3, " km")
    println("  Speed: ", norm(v_vec_0) / 1e3, " km/s")

    # Loop over OE initial std scenarios
    for (scenario_idx, oe_std) in enumerate(oe_initial_std_scenarios)
        println("\n" * "-"^80)
        println("OE INITIAL STD SCENARIO ", scenario_idx, ": σ_a = ", oe_std[1], " m")
        println("-"^80)
        println("  OE initial std: ", oe_std)

        # Compute P_0 once from OE Monte Carlo samples; all methods use it
        P_0 = compute_P0_from_oe_samples(oe_vec_0, oe_std, SIM_PARAMS.GM, SIM_PARAMS.R_EARTH)

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
                    # Handle 2-value (x, P) or 3-value (x, P, sigma_points/samples) returns
                    if result isa Tuple && length(result) >= 3
                        x_vec_traj, P_traj, sigma_points_traj = result
                    else
                        x_vec_traj, P_traj = result
                        sigma_points_traj = nothing
                    end
                    println("  Completed: ", length(x_vec_traj), " states", (length(x_vec_traj) < length(times) ? " (partial)" : ""))
                    if sigma_points_traj !== nothing
                        println("  Sigma points/samples: ", length(sigma_points_traj[1]), " per timestep")
                    end
                    return (x_vec_traj=x_vec_traj, P_traj=P_traj, sigma_points_traj=sigma_points_traj, failed=false, error_msg=nothing, error=nothing)
                catch e
                    println("  ✗ FAILED: ", method_name)
                    println("  Error: ", typeof(e).name.name, ": ", sprint(showerror, e))
                    return (x_vec_traj=Vector{Vector{Float64}}(), P_traj=Vector{Matrix{Float64}}(), sigma_points_traj=nothing, failed=true, error_msg=sprint(showerror, e), error=e)
                end
            end

            # Monte Carlo (ground truth) - load from npz file if present, else run on the fly
            println("\n" * "="^80)
            data_dir = joinpath(@__DIR__, "..", "data")
            mc_filename = "mc_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_std_e$(fname_num(oe_std[2]))_num_samples$(Int(num_samples)).npz"
            mc_filepath = joinpath(data_dir, mc_filename)
            if isfile(mc_filepath)
                println("1. MONTE CARLO (Ground Truth) - Loading from file")
                println("="^80)
                println("  Loading from: ", mc_filepath)
                npz_data = NPZ.npzread(mc_filepath)
                x_array = npz_data["x"]
                P_array = npz_data["P"]
                timestamp = npz_data["timestamp"]
                N_loaded = size(x_array, 1)
                x_vec_traj_mean_mc = [x_array[i, :] for i in 1:N_loaded]
                P_traj_mc = [P_array[i, :, :] for i in 1:N_loaded]
                if length(timestamp) != length(times) || maximum(abs.(timestamp .- times)) > 1e-6
                    println("  Warning: Loaded timestamp differs from expected times. Using loaded timestamp.")
                    times = timestamp
                end
                println("  Loaded: ", length(x_vec_traj_mean_mc), " states")
            else
                println("1. MONTE CARLO (Ground Truth) - Running on the fly (file not found)")
                println("="^80)
                println("  File not found: ", mc_filepath)
                x_vec_traj_mean_mc, P_traj_mc = propagate_uncertainty_via_monte_carlo(x_vec_0, P_0, times, sim_params, num_samples;
                    oe_std=oe_std)
                println("  Completed: ", length(x_vec_traj_mean_mc), " states")
            end

            # Linearized Covariance Propagation (Cartesian)
            result_lin_cart = run_method_safely("2. LINCOV (Cartesian)",
                propagate_uncertainty_via_linearized_cartesian_dynamics, x_vec_0, P_0, times, sim_params)
            x_vec_traj_mean_lin_cart = result_lin_cart.x_vec_traj
            P_traj_lin_cart = result_lin_cart.P_traj

            # Unscented Transform (Cartesian)
            result_ut_cart = run_method_safely("3. CARTESIAN UT",
                propagate_uncertainty_via_cartesian_unscented_transform, x_vec_0, P_0, times, sim_params;
                return_sigma_points=true)
            x_vec_traj_mean_ut_cart = result_ut_cart.x_vec_traj
            P_traj_ut_cart = result_ut_cart.P_traj

            # Eigen-based Sigma Points (Cartesian)
            result_eigen_cart = run_method_safely("4. CARTESIAN CKF",
                propagate_uncertainty_via_cartesian_sigma_points, x_vec_0, P_0, times, sim_params;
                return_sigma_points=true)
            x_vec_traj_mean_eigen_cart = result_eigen_cart.x_vec_traj
            P_traj_eigen_cart = result_eigen_cart.P_traj

            # Eigen-based Sigma Points (KS)
            result_eigen_ks = run_method_safely("5. KS CKF",
                propagate_uncertainty_via_ks_sigma_points, x_vec_0, P_0, times, sim_params;
                return_sigma_points=true)
            x_vec_traj_mean_eigen_ks = result_eigen_ks.x_vec_traj
            P_traj_eigen_ks = result_eigen_ks.P_traj

            # Linearized KS Sigma Points
            result_lin_ks_sigma = run_method_safely("6. KS RELATIVE CKF",
                propagate_uncertainty_via_linearized_ks_sigma_points, x_vec_0, P_0, times, sim_params;
                return_sigma_points=true)
            x_vec_traj_mean_lin_ks_sigma = result_lin_ks_sigma.x_vec_traj
            P_traj_lin_ks_sigma = result_lin_ks_sigma.P_traj

            # LinCov (KS) - commented out; approach needs fixing and is not ready for running yet
            # result_lin_ks_dyn = run_method_safely("7. KS LINCOV",
            #     propagate_uncertainty_via_linearized_ks_dynamics, x_vec_0, P_0, times, sim_params)
            # x_vec_traj_mean_lin_ks_dyn = result_lin_ks_dyn.x_vec_traj
            # P_traj_lin_ks_dyn = result_lin_ks_dyn.P_traj
            result_lin_ks_dyn = (x_vec_traj=Vector{Vector{Float64}}(), P_traj=Vector{Matrix{Float64}}(), sigma_points_traj=nothing, failed=true, error_msg=nothing, error=nothing)
            x_vec_traj_mean_lin_ks_dyn = Vector{Vector{Float64}}()
            P_traj_lin_ks_dyn = Vector{Matrix{Float64}}()

            # Supervisor-spec: MC energy binning -> per-bin linearized KS sigma points -> per-step sampling aggregation
            result_mc_binned_ks = run_method_safely("8. ENERGY-STRATIFIED KS CKF",
                propagate_uncertainty_via_energy_binned_mc_then_ks_sigma_points, x_vec_0, P_0, times, sim_params;
                num_mc_samples=num_mc_samples_binning, num_energy_bins=num_energy_bins, min_samples_threshold=min_samples_threshold, return_samples=true,
                oe_std=oe_std)
            x_vec_traj_mean_mc_binned_ks = result_mc_binned_ks.x_vec_traj
            P_traj_mc_binned_ks = result_mc_binned_ks.P_traj

            # Save approach results to out/ (only when save_to_out is true)
            if save_to_out
                out_dir = joinpath(@__DIR__, "..", "out")
                mkpath(out_dir)

                function save_approach_npz(out_dir, approach_id, orbit, num_orbits, oe_std, times_save, x_vec_traj, P_traj; sigma_points_traj=nothing)
                    N_save = length(x_vec_traj)
                    x_array = zeros(N_save, 6)
                    P_array = zeros(N_save, 6, 6)
                    for i in 1:N_save
                        x_array[i, :] = x_vec_traj[i]
                        P_array[i, :, :] = P_traj[i]
                    end
                    npz_dict = Dict{String,Any}("x" => x_array, "P" => P_array, "timestamp" => collect(times_save[1:N_save]))
                    if sigma_points_traj !== nothing && length(sigma_points_traj) >= N_save
                        num_points = length(sigma_points_traj[1])
                        for j in 1:num_points
                            s_array = zeros(N_save, 6)
                            for i in 1:N_save
                                s_array[i, :] = sigma_points_traj[i][j]
                            end
                            npz_dict["s$j"] = s_array
                        end
                        println("  Including ", num_points, " sigma points/samples per timestep")
                    end
                    filename = "$(approach_id)_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_std_e$(fname_num(oe_std[2])).npz"
                    filepath = joinpath(out_dir, filename)
                    NPZ.npzwrite(filepath, npz_dict)
                    println("  Saved: ", filepath)
                end

                println("\n" * "="^80)
                println("SAVING APPROACH RESULTS TO out/")
                println("="^80)
                save_approach_npz(out_dir, "mc", orbit, num_orbits, oe_std, times, x_vec_traj_mean_mc, P_traj_mc)
                if !result_lin_cart.failed
                    save_approach_npz(out_dir, "lincov_cart", orbit, num_orbits, oe_std, times, x_vec_traj_mean_lin_cart, P_traj_lin_cart)
                end
                if !result_ut_cart.failed
                    save_approach_npz(out_dir, "ut_cart", orbit, num_orbits, oe_std, times, x_vec_traj_mean_ut_cart, P_traj_ut_cart;
                        sigma_points_traj=result_ut_cart.sigma_points_traj)
                end
                if !result_eigen_cart.failed
                    save_approach_npz(out_dir, "ckf_cart", orbit, num_orbits, oe_std, times, x_vec_traj_mean_eigen_cart, P_traj_eigen_cart;
                        sigma_points_traj=result_eigen_cart.sigma_points_traj)
                end
                if !result_eigen_ks.failed
                    save_approach_npz(out_dir, "ckf_ks", orbit, num_orbits, oe_std, times, x_vec_traj_mean_eigen_ks, P_traj_eigen_ks;
                        sigma_points_traj=result_eigen_ks.sigma_points_traj)
                end
                if !result_lin_ks_sigma.failed
                    save_approach_npz(out_dir, "ckf_ks_rel", orbit, num_orbits, oe_std, times, x_vec_traj_mean_lin_ks_sigma, P_traj_lin_ks_sigma;
                        sigma_points_traj=result_lin_ks_sigma.sigma_points_traj)
                end
                if !result_mc_binned_ks.failed
                    save_approach_npz(out_dir, "stratified_ks", orbit, num_orbits, oe_std, times, x_vec_traj_mean_mc_binned_ks, P_traj_mc_binned_ks;
                        sigma_points_traj=result_mc_binned_ks.sigma_points_traj)
                end
            end

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
            # if !result_lin_ks_dyn.failed
            #     push!(lengths_to_check, length(x_vec_traj_mean_lin_ks_dyn))
            # end
            if !result_mc_binned_ks.failed
                push!(lengths_to_check, length(x_vec_traj_mean_mc_binned_ks))
            end
            N = minimum(lengths_to_check)

            # Print position errors
            println("\nPosition errors (vs Monte Carlo):")
            if metrics_lin_cart !== nothing
                println("  LinCov (Cartesian):")
                println("    RMSE = ", metrics_lin_cart.pos_rmse, " m, Min = ", metrics_lin_cart.pos_min, " m, Max = ", metrics_lin_cart.pos_max, " m")
            else
                println("  LinCov (Cartesian): FAILED")
            end
            if metrics_ut_cart !== nothing
                println("  UT (Cartesian):")
                println("    RMSE = ", metrics_ut_cart.pos_rmse, " m, Min = ", metrics_ut_cart.pos_min, " m, Max = ", metrics_ut_cart.pos_max, " m")
            else
                println("  UT (Cartesian): FAILED")
            end
            if metrics_eigen_cart !== nothing
                println("  CKF (Cartesian):")
                println("    RMSE = ", metrics_eigen_cart.pos_rmse, " m, Min = ", metrics_eigen_cart.pos_min, " m, Max = ", metrics_eigen_cart.pos_max, " m")
            else
                println("  CKF (Cartesian): FAILED")
            end
            if metrics_eigen_ks !== nothing
                println("  CKF (KS):")
                println("    RMSE = ", metrics_eigen_ks.pos_rmse, " m, Min = ", metrics_eigen_ks.pos_min, " m, Max = ", metrics_eigen_ks.pos_max, " m")
            else
                println("  CKF (KS): FAILED")
            end
            if metrics_lin_ks_sigma !== nothing
                println("  CKF (KS Relative):")
                println("    RMSE = ", metrics_lin_ks_sigma.pos_rmse, " m, Min = ", metrics_lin_ks_sigma.pos_min, " m, Max = ", metrics_lin_ks_sigma.pos_max, " m")
            else
                println("  CKF (KS Relative): FAILED")
            end
            if metrics_lin_ks_dyn !== nothing
                println("  LinCov (KS):")
                println("    RMSE = ", metrics_lin_ks_dyn.pos_rmse, " m, Min = ", metrics_lin_ks_dyn.pos_min, " m, Max = ", metrics_lin_ks_dyn.pos_max, " m")
            else
                println("  LinCov (KS): FAILED")
            end
            if metrics_mc_binned_ks !== nothing
                println("  Stratified KS CKF:")
                println("    RMSE = ", metrics_mc_binned_ks.pos_rmse, " m, Min = ", metrics_mc_binned_ks.pos_min, " m, Max = ", metrics_mc_binned_ks.pos_max, " m")
            else
                println("  Stratified KS CKF: FAILED")
            end

            # Print velocity errors
            println("\nVelocity errors (vs Monte Carlo):")
            if metrics_lin_cart !== nothing
                println("  LinCov (Cartesian):")
                println("    RMSE = ", metrics_lin_cart.vel_rmse, " m/s, Min = ", metrics_lin_cart.vel_min, " m/s, Max = ", metrics_lin_cart.vel_max, " m/s")
            else
                println("  LinCov (Cartesian): FAILED")
            end
            if metrics_ut_cart !== nothing
                println("  UT (Cartesian):")
                println("    RMSE = ", metrics_ut_cart.vel_rmse, " m/s, Min = ", metrics_ut_cart.vel_min, " m/s, Max = ", metrics_ut_cart.vel_max, " m/s")
            else
                println("  UT (Cartesian): FAILED")
            end
            if metrics_eigen_cart !== nothing
                println("  CKF (Cartesian):")
                println("    RMSE = ", metrics_eigen_cart.vel_rmse, " m/s, Min = ", metrics_eigen_cart.vel_min, " m/s, Max = ", metrics_eigen_cart.vel_max, " m/s")
            else
                println("  CKF (Cartesian): FAILED")
            end
            if metrics_eigen_ks !== nothing
                println("  CKF (KS):")
                println("    RMSE = ", metrics_eigen_ks.vel_rmse, " m/s, Min = ", metrics_eigen_ks.vel_min, " m/s, Max = ", metrics_eigen_ks.vel_max, " m/s")
            else
                println("  CKF (KS): FAILED")
            end
            if metrics_lin_ks_sigma !== nothing
                println("  CKF (KS Relative):")
                println("    RMSE = ", metrics_lin_ks_sigma.vel_rmse, " m/s, Min = ", metrics_lin_ks_sigma.vel_min, " m/s, Max = ", metrics_lin_ks_sigma.vel_max, " m/s")
            else
                println("  CKF (KS Relative): FAILED")
            end
            if metrics_lin_ks_dyn !== nothing
                println("  LinCov (KS):")
                println("    RMSE = ", metrics_lin_ks_dyn.vel_rmse, " m/s, Min = ", metrics_lin_ks_dyn.vel_min, " m/s, Max = ", metrics_lin_ks_dyn.vel_max, " m/s")
            else
                println("  LinCov (KS): FAILED")
            end
            if metrics_mc_binned_ks !== nothing
                println("  Stratified KS CKF:")
                println("    RMSE = ", metrics_mc_binned_ks.vel_rmse, " m/s, Min = ", metrics_mc_binned_ks.vel_min, " m/s, Max = ", metrics_mc_binned_ks.vel_max, " m/s")
            else
                println("  Stratified KS CKF: FAILED")
            end

            # Print position and velocity uncertainty errors
            println("\nPosition uncertainty errors (vs Monte Carlo):")
            if metrics_lin_cart !== nothing
                println("  LinCov (Cartesian):")
                println("    RMSE = ", metrics_lin_cart.pos_uncertainty_rmse, " m, Min = ", metrics_lin_cart.pos_uncertainty_min, " m, Max = ", metrics_lin_cart.pos_uncertainty_max, " m")
            else
                println("  LinCov (Cartesian): FAILED")
            end
            if metrics_ut_cart !== nothing
                println("  UT (Cartesian):")
                println("    RMSE = ", metrics_ut_cart.pos_uncertainty_rmse, " m, Min = ", metrics_ut_cart.pos_uncertainty_min, " m, Max = ", metrics_ut_cart.pos_uncertainty_max, " m")
            else
                println("  UT (Cartesian): FAILED")
            end
            if metrics_eigen_cart !== nothing
                println("  CKF (Cartesian):")
                println("    RMSE = ", metrics_eigen_cart.pos_uncertainty_rmse, " m, Min = ", metrics_eigen_cart.pos_uncertainty_min, " m, Max = ", metrics_eigen_cart.pos_uncertainty_max, " m")
            else
                println("  CKF (Cartesian): FAILED")
            end
            if metrics_eigen_ks !== nothing
                println("  CKF (KS):")
                println("    RMSE = ", metrics_eigen_ks.pos_uncertainty_rmse, " m, Min = ", metrics_eigen_ks.pos_uncertainty_min, " m, Max = ", metrics_eigen_ks.pos_uncertainty_max, " m")
            else
                println("  CKF (KS): FAILED")
            end
            if metrics_lin_ks_sigma !== nothing
                println("  CKF (KS Relative):")
                println("    RMSE = ", metrics_lin_ks_sigma.pos_uncertainty_rmse, " m, Min = ", metrics_lin_ks_sigma.pos_uncertainty_min, " m, Max = ", metrics_lin_ks_sigma.pos_uncertainty_max, " m")
            else
                println("  CKF (KS Relative): FAILED")
            end
            if metrics_lin_ks_dyn !== nothing
                println("  LinCov (KS):")
                println("    RMSE = ", metrics_lin_ks_dyn.pos_uncertainty_rmse, " m, Min = ", metrics_lin_ks_dyn.pos_uncertainty_min, " m, Max = ", metrics_lin_ks_dyn.pos_uncertainty_max, " m")
            else
                println("  LinCov (KS): FAILED")
            end
            if metrics_mc_binned_ks !== nothing
                println("  Stratified KS CKF:")
                println("    RMSE = ", metrics_mc_binned_ks.pos_uncertainty_rmse, " m, Min = ", metrics_mc_binned_ks.pos_uncertainty_min, " m, Max = ", metrics_mc_binned_ks.pos_uncertainty_max, " m")
            else
                println("  Stratified KS CKF: FAILED")
            end

            println("\nVelocity uncertainty errors (vs Monte Carlo):")
            if metrics_lin_cart !== nothing
                println("  LinCov (Cartesian):")
                println("    RMSE = ", metrics_lin_cart.vel_uncertainty_rmse, " m/s, Min = ", metrics_lin_cart.vel_uncertainty_min, " m/s, Max = ", metrics_lin_cart.vel_uncertainty_max, " m/s")
            else
                println("  LinCov (Cartesian): FAILED")
            end
            if metrics_ut_cart !== nothing
                println("  UT (Cartesian):")
                println("    RMSE = ", metrics_ut_cart.vel_uncertainty_rmse, " m/s, Min = ", metrics_ut_cart.vel_uncertainty_min, " m/s, Max = ", metrics_ut_cart.vel_uncertainty_max, " m/s")
            else
                println("  UT (Cartesian): FAILED")
            end
            if metrics_eigen_cart !== nothing
                println("  CKF (Cartesian):")
                println("    RMSE = ", metrics_eigen_cart.vel_uncertainty_rmse, " m/s, Min = ", metrics_eigen_cart.vel_uncertainty_min, " m/s, Max = ", metrics_eigen_cart.vel_uncertainty_max, " m/s")
            else
                println("  CKF (Cartesian): FAILED")
            end
            if metrics_eigen_ks !== nothing
                println("  CKF (KS):")
                println("    RMSE = ", metrics_eigen_ks.vel_uncertainty_rmse, " m/s, Min = ", metrics_eigen_ks.vel_uncertainty_min, " m/s, Max = ", metrics_eigen_ks.vel_uncertainty_max, " m/s")
            else
                println("  CKF (KS): FAILED")
            end
            if metrics_lin_ks_sigma !== nothing
                println("  CKF (KS Relative):")
                println("    RMSE = ", metrics_lin_ks_sigma.vel_uncertainty_rmse, " m/s, Min = ", metrics_lin_ks_sigma.vel_uncertainty_min, " m/s, Max = ", metrics_lin_ks_sigma.vel_uncertainty_max, " m/s")
            else
                println("  CKF (KS Relative): FAILED")
            end
            if metrics_lin_ks_dyn !== nothing
                println("  LinCov (KS):")
                println("    RMSE = ", metrics_lin_ks_dyn.vel_uncertainty_rmse, " m/s, Min = ", metrics_lin_ks_dyn.vel_uncertainty_min, " m/s, Max = ", metrics_lin_ks_dyn.vel_uncertainty_max, " m/s")
            else
                println("  LinCov (KS): FAILED")
            end
            if metrics_mc_binned_ks !== nothing
                println("  Stratified KS CKF:")
                println("    RMSE = ", metrics_mc_binned_ks.vel_uncertainty_rmse, " m/s, Min = ", metrics_mc_binned_ks.vel_uncertainty_min, " m/s, Max = ", metrics_mc_binned_ks.vel_uncertainty_max, " m/s")
            else
                println("  Stratified KS CKF: FAILED")
            end

            # Create plots
            println("\nGenerating plots...")

            # Per-method lengths: plot each method over its full range (partial methods stop where they failed)
            n_lin = metrics_lin_cart !== nothing ? length(metrics_lin_cart.pos_errors) : 0
            n_ut = metrics_ut_cart !== nothing ? length(metrics_ut_cart.pos_errors) : 0
            n_eigen = metrics_eigen_cart !== nothing ? length(metrics_eigen_cart.pos_errors) : 0
            n_ks = metrics_eigen_ks !== nothing ? length(metrics_eigen_ks.pos_errors) : 0
            n_ks_rel = metrics_lin_ks_sigma !== nothing ? length(metrics_lin_ks_sigma.pos_errors) : 0
            n_lin_ks = metrics_lin_ks_dyn !== nothing ? length(metrics_lin_ks_dyn.pos_errors) : 0
            n_binned = metrics_mc_binned_ks !== nothing ? length(metrics_mc_binned_ks.pos_errors) : 0

            # KL divergence vs Monte Carlo (per-method length)
            kl_lin_cart = n_lin > 0 ? [gaussian_kl_divergence("2. LINCOV (Cartesian)", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_lin_cart[i], P_traj_lin_cart[i]) for i in 1:n_lin] : nothing
            kl_ut_cart = n_ut > 0 ? [gaussian_kl_divergence("3. CARTESIAN UT", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_ut_cart[i], P_traj_ut_cart[i]) for i in 1:n_ut] : nothing
            kl_eigen_cart = n_eigen > 0 ? [gaussian_kl_divergence("4. CARTESIAN CKF", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_eigen_cart[i], P_traj_eigen_cart[i]) for i in 1:n_eigen] : nothing
            kl_eigen_ks = n_ks > 0 ? [gaussian_kl_divergence("5. KS CKF", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_eigen_ks[i], P_traj_eigen_ks[i]) for i in 1:n_ks] : nothing
            kl_lin_ks_sigma = n_ks_rel > 0 ? [gaussian_kl_divergence("6. KS RELATIVE CKF", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_lin_ks_sigma[i], P_traj_lin_ks_sigma[i]) for i in 1:n_ks_rel] : nothing
            kl_lin_ks_dyn = n_lin_ks > 0 ? [gaussian_kl_divergence("7. KS LINCOV", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_lin_ks_dyn[i], P_traj_lin_ks_dyn[i]) for i in 1:n_lin_ks] : nothing
            kl_mc_binned_ks = n_binned > 0 ? [gaussian_kl_divergence("8. ENERGY-STRATIFIED KS CKF", times[i], x_vec_traj_mean_mc[i], P_traj_mc[i], x_vec_traj_mean_mc_binned_ks[i], P_traj_mc_binned_ks[i]) for i in 1:n_binned] : nothing

            # Plot 1: Position error (each method over its full range)
            p1 = plot(xlabel="Time (hours)", ylabel="Position Error (m)", yscale=:log10, legend=:topleft)
            if n_lin > 0
                plot!(p1, times[1:n_lin] ./ 3600, metrics_lin_cart.pos_errors, label="LinCov (Cartesian)",
                    linewidth=2, color=:green, linestyle=:dash)
            end
            if n_ut > 0
                plot!(p1, times[1:n_ut] ./ 3600, metrics_ut_cart.pos_errors, label="UT (Cartesian)",
                    linewidth=2, color=:blue, linestyle=:dot)
            end
            if n_eigen > 0
                plot!(p1, times[1:n_eigen] ./ 3600, metrics_eigen_cart.pos_errors, label="CKF (Cartesian)",
                    linewidth=2, color=:red, linestyle=:dashdot)
            end
            if n_ks > 0
                plot!(p1, times[1:n_ks] ./ 3600, metrics_eigen_ks.pos_errors, label="CKF (KS)",
                    linewidth=2, color=:orange, linestyle=:dashdotdot)
            end
            if n_ks_rel > 0
                plot!(p1, times[1:n_ks_rel] ./ 3600, metrics_lin_ks_sigma.pos_errors, label="CKF (KS Relative)",
                    linewidth=2, color=:purple, linestyle=:dashdotdot)
            end
            if n_lin_ks > 0
                plot!(p1, times[1:n_lin_ks] ./ 3600, metrics_lin_ks_dyn.pos_errors, label="LinCov (KS)",
                    linewidth=2, color=:brown, linestyle=:dot)
            end
            if n_binned > 0
                plot!(p1, times[1:n_binned] ./ 3600, metrics_mc_binned_ks.pos_errors, label="Stratified KS CKF",
                    linewidth=2, color=:black, linestyle=:dash)
            end

            # Plot 2: Velocity error (each method over its full range)
            p2 = plot(xlabel="Time (hours)", ylabel="Velocity Error (m/s)", yscale=:log10, legend=:topleft)
            if n_lin > 0
                plot!(p2, times[1:n_lin] ./ 3600, metrics_lin_cart.vel_errors, label="LinCov (Cartesian)",
                    linewidth=2, color=:green, linestyle=:dash)
            end
            if n_ut > 0
                plot!(p2, times[1:n_ut] ./ 3600, metrics_ut_cart.vel_errors, label="UT (Cartesian)",
                    linewidth=2, color=:blue, linestyle=:dot)
            end
            if n_eigen > 0
                plot!(p2, times[1:n_eigen] ./ 3600, metrics_eigen_cart.vel_errors, label="CKF (Cartesian)",
                    linewidth=2, color=:red, linestyle=:dashdot)
            end
            if n_ks > 0
                plot!(p2, times[1:n_ks] ./ 3600, metrics_eigen_ks.vel_errors, label="CKF (KS)",
                    linewidth=2, color=:orange, linestyle=:dashdotdot)
            end
            if n_ks_rel > 0
                plot!(p2, times[1:n_ks_rel] ./ 3600, metrics_lin_ks_sigma.vel_errors, label="CKF (KS Relative)",
                    linewidth=2, color=:purple, linestyle=:dashdotdot)
            end
            if n_lin_ks > 0
                plot!(p2, times[1:n_lin_ks] ./ 3600, metrics_lin_ks_dyn.vel_errors, label="LinCov (KS)",
                    linewidth=2, color=:brown, linestyle=:dot)
            end
            if n_binned > 0
                plot!(p2, times[1:n_binned] ./ 3600, metrics_mc_binned_ks.vel_errors, label="Stratified KS CKF",
                    linewidth=2, color=:black, linestyle=:dash)
            end

            # Plot 3: Position uncertainty error (each method over its full range)
            p3 = plot(xlabel="Time (hours)", ylabel="Position Uncertainty Error (m)", yscale=:log10, legend=:topleft)
            if n_lin > 0
                plot!(p3, times[1:n_lin] ./ 3600, metrics_lin_cart.pos_uncertainty_errors, label="LinCov (Cartesian)",
                    linewidth=2, color=:green, linestyle=:dash)
            end
            if n_ut > 0
                plot!(p3, times[1:n_ut] ./ 3600, metrics_ut_cart.pos_uncertainty_errors, label="UT (Cartesian)",
                    linewidth=2, color=:blue, linestyle=:dot)
            end
            if n_eigen > 0
                plot!(p3, times[1:n_eigen] ./ 3600, metrics_eigen_cart.pos_uncertainty_errors, label="CKF (Cartesian)",
                    linewidth=2, color=:red, linestyle=:dashdot)
            end
            if n_ks > 0
                plot!(p3, times[1:n_ks] ./ 3600, metrics_eigen_ks.pos_uncertainty_errors, label="CKF (KS)",
                    linewidth=2, color=:orange, linestyle=:dashdotdot)
            end
            if n_ks_rel > 0
                plot!(p3, times[1:n_ks_rel] ./ 3600, metrics_lin_ks_sigma.pos_uncertainty_errors, label="CKF (KS Relative)",
                    linewidth=2, color=:purple, linestyle=:dashdotdot)
            end
            if n_lin_ks > 0
                plot!(p3, times[1:n_lin_ks] ./ 3600, metrics_lin_ks_dyn.pos_uncertainty_errors, label="LinCov (KS)",
                    linewidth=2, color=:brown, linestyle=:dot)
            end
            if n_binned > 0
                plot!(p3, times[1:n_binned] ./ 3600, metrics_mc_binned_ks.pos_uncertainty_errors, label="Stratified KS CKF",
                    linewidth=2, color=:black, linestyle=:dash)
            end

            # Plot 4: Velocity uncertainty error (each method over its full range)
            p4 = plot(xlabel="Time (hours)", ylabel="Velocity Uncertainty Error (m/s)", yscale=:log10, legend=:topleft)
            if n_lin > 0
                plot!(p4, times[1:n_lin] ./ 3600, metrics_lin_cart.vel_uncertainty_errors, label="LinCov (Cartesian)",
                    linewidth=2, color=:green, linestyle=:dash)
            end
            if n_ut > 0
                plot!(p4, times[1:n_ut] ./ 3600, metrics_ut_cart.vel_uncertainty_errors, label="UT (Cartesian)",
                    linewidth=2, color=:blue, linestyle=:dot)
            end
            if n_eigen > 0
                plot!(p4, times[1:n_eigen] ./ 3600, metrics_eigen_cart.vel_uncertainty_errors, label="CKF (Cartesian)",
                    linewidth=2, color=:red, linestyle=:dashdot)
            end
            if n_ks > 0
                plot!(p4, times[1:n_ks] ./ 3600, metrics_eigen_ks.vel_uncertainty_errors, label="CKF (KS)",
                    linewidth=2, color=:orange, linestyle=:dashdotdot)
            end
            if n_ks_rel > 0
                plot!(p4, times[1:n_ks_rel] ./ 3600, metrics_lin_ks_sigma.vel_uncertainty_errors, label="CKF (KS Relative)",
                    linewidth=2, color=:purple, linestyle=:dashdotdot)
            end
            if n_lin_ks > 0
                plot!(p4, times[1:n_lin_ks] ./ 3600, metrics_lin_ks_dyn.vel_uncertainty_errors, label="LinCov (KS)",
                    linewidth=2, color=:brown, linestyle=:dot)
            end
            if n_binned > 0
                plot!(p4, times[1:n_binned] ./ 3600, metrics_mc_binned_ks.vel_uncertainty_errors, label="Stratified KS CKF",
                    linewidth=2, color=:black, linestyle=:dash)
            end

            # Plot 5: KL divergence (each method over its full range)
            p5 = plot(xlabel="Time (hours)", ylabel="KL Divergence", yscale=:log10, legend=:topleft)
            if kl_lin_cart !== nothing
                plot!(p5, times[1:n_lin] ./ 3600, kl_lin_cart, label="LinCov (Cartesian)",
                    linewidth=2, color=:green, linestyle=:dash)
            end
            if kl_ut_cart !== nothing
                plot!(p5, times[1:n_ut] ./ 3600, kl_ut_cart, label="UT (Cartesian)",
                    linewidth=2, color=:blue, linestyle=:dot)
            end
            if kl_eigen_cart !== nothing
                plot!(p5, times[1:n_eigen] ./ 3600, kl_eigen_cart, label="CKF (Cartesian)",
                    linewidth=2, color=:red, linestyle=:dashdot)
            end
            if kl_eigen_ks !== nothing
                plot!(p5, times[1:n_ks] ./ 3600, kl_eigen_ks, label="CKF (KS)",
                    linewidth=2, color=:orange, linestyle=:dashdotdot)
            end
            if kl_lin_ks_sigma !== nothing
                plot!(p5, times[1:n_ks_rel] ./ 3600, kl_lin_ks_sigma, label="CKF (KS Relative)",
                    linewidth=2, color=:purple, linestyle=:dashdotdot)
            end
            if kl_lin_ks_dyn !== nothing
                plot!(p5, times[1:n_lin_ks] ./ 3600, kl_lin_ks_dyn, label="LinCov (KS)",
                    linewidth=2, color=:brown, linestyle=:dot)
            end
            if kl_mc_binned_ks !== nothing
                plot!(p5, times[1:n_binned] ./ 3600, kl_mc_binned_ks, label="Stratified KS CKF",
                    linewidth=2, color=:black, linestyle=:dash)
            end

            # Generate filename based on scenario
            figdir = joinpath(@__DIR__, "..", "figs")
            mkpath(figdir)
            filename = joinpath(figdir, "error_propagation_comparison_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_std_e$(fname_num(oe_std[2]))_num_samples$(Int(num_samples)).png")

            # Combine plots with increased left margin for y-axis labels
            # Use wider size and set margins explicitly to ensure y-labels are visible
            p_combined = plot(p1, p2, p3, p4, p5, layout=(5, 1), size=(1000, 2000), left_margin=50Plots.px)
            savefig(p_combined, filename)
            println("  Saved plot to ", filename)

            # Store results
            push!(all_results, (
                orbit=orbit.name,
                oe_std=oe_std,
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
    end  # end oe_std scenario loop
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
    println("  OE initial std (σ_a) = ", result.oe_std[1], " m  |  orbits = ", result.num_orbits)
    println("  Position errors:")
    if m_lin !== nothing
        println("    LinCov (Cartesian) - RMSE: ", m_lin.pos_rmse, " m, Min: ", m_lin.pos_min, " m, Max: ", m_lin.pos_max, " m")
    else
        println("    LinCov (Cartesian) - FAILED")
    end
    if m_ut !== nothing
        println("    UT (Cartesian) - RMSE: ", m_ut.pos_rmse, " m, Min: ", m_ut.pos_min, " m, Max: ", m_ut.pos_max, " m")
    else
        println("    UT (Cartesian) - FAILED")
    end
    (m_eigen_cart !== nothing) && println("    CKF (Cartesian) - RMSE: ", m_eigen_cart.pos_rmse, " m, Min: ", m_eigen_cart.pos_min, " m, Max: ", m_eigen_cart.pos_max, " m")
    (m_eigen_cart === nothing) && println("    CKF (Cartesian): FAILED")
    (m_eigen_ks !== nothing) && println("    CKF (KS) - RMSE: ", m_eigen_ks.pos_rmse, " m, Min: ", m_eigen_ks.pos_min, " m, Max: ", m_eigen_ks.pos_max, " m")
    (m_eigen_ks === nothing) && println("    CKF (KS): FAILED")
    (m_lin_ks_sigma !== nothing) && println("    CKF (KS Relative) - RMSE: ", m_lin_ks_sigma.pos_rmse, " m, Min: ", m_lin_ks_sigma.pos_min, " m, Max: ", m_lin_ks_sigma.pos_max, " m")
    (m_lin_ks_sigma === nothing) && println("    CKF (KS Relative): FAILED")
    (m_lin_ks_dyn !== nothing) && println("    LinCov (KS) - RMSE: ", m_lin_ks_dyn.pos_rmse, " m, Min: ", m_lin_ks_dyn.pos_min, " m, Max: ", m_lin_ks_dyn.pos_max, " m")
    (m_lin_ks_dyn === nothing) && println("    LinCov (KS): FAILED")
    (m_mc_binned_ks !== nothing) && println("    Stratified KS CKF - RMSE: ", m_mc_binned_ks.pos_rmse, " m, Min: ", m_mc_binned_ks.pos_min, " m, Max: ", m_mc_binned_ks.pos_max, " m")
    (m_mc_binned_ks === nothing) && println("    Stratified KS CKF: FAILED")
    println("  Velocity errors:")
    (m_lin !== nothing) && println("    LinCov (Cartesian) - RMSE: ", m_lin.vel_rmse, " m/s, Min: ", m_lin.vel_min, " m/s, Max: ", m_lin.vel_max, " m/s")
    (m_ut !== nothing) && println("    UT (Cartesian) - RMSE: ", m_ut.vel_rmse, " m/s, Min: ", m_ut.vel_min, " m/s, Max: ", m_ut.vel_max, " m/s")
    (m_eigen_cart !== nothing) && println("    CKF (Cartesian) - RMSE: ", m_eigen_cart.vel_rmse, " m/s, Min: ", m_eigen_cart.vel_min, " m/s, Max: ", m_eigen_cart.vel_max, " m/s")
    (m_eigen_ks !== nothing) && println("    CKF (KS) - RMSE: ", m_eigen_ks.vel_rmse, " m/s, Min: ", m_eigen_ks.vel_min, " m/s, Max: ", m_eigen_ks.vel_max, " m/s")
    (m_lin_ks_sigma !== nothing) && println("    CKF (KS Relative) - RMSE: ", m_lin_ks_sigma.vel_rmse, " m/s, Min: ", m_lin_ks_sigma.vel_min, " m/s, Max: ", m_lin_ks_sigma.vel_max, " m/s")
    (m_lin_ks_dyn !== nothing) && println("    LinCov (KS) - RMSE: ", m_lin_ks_dyn.vel_rmse, " m/s, Min: ", m_lin_ks_dyn.vel_min, " m/s, Max: ", m_lin_ks_dyn.vel_max, " m/s")
    (m_mc_binned_ks !== nothing) && println("    Stratified KS CKF - RMSE: ", m_mc_binned_ks.vel_rmse, " m/s, Min: ", m_mc_binned_ks.vel_min, " m/s, Max: ", m_mc_binned_ks.vel_max, " m/s")
    println("  Position uncertainty errors:")
    (m_lin !== nothing) && println("    LinCov (Cartesian) - RMSE: ", m_lin.pos_uncertainty_rmse, " m, Min: ", m_lin.pos_uncertainty_min, " m, Max: ", m_lin.pos_uncertainty_max, " m")
    (m_ut !== nothing) && println("    UT (Cartesian) - RMSE: ", m_ut.pos_uncertainty_rmse, " m, Min: ", m_ut.pos_uncertainty_min, " m, Max: ", m_ut.pos_uncertainty_max, " m")
    (m_eigen_cart !== nothing) && println("    CKF (Cartesian) - RMSE: ", m_eigen_cart.pos_uncertainty_rmse, " m, Min: ", m_eigen_cart.pos_uncertainty_min, " m, Max: ", m_eigen_cart.pos_uncertainty_max, " m")
    (m_eigen_ks !== nothing) && println("    CKF (KS) - RMSE: ", m_eigen_ks.pos_uncertainty_rmse, " m, Min: ", m_eigen_ks.pos_uncertainty_min, " m, Max: ", m_eigen_ks.pos_uncertainty_max, " m")
    (m_lin_ks_sigma !== nothing) && println("    CKF (KS Relative) - RMSE: ", m_lin_ks_sigma.pos_uncertainty_rmse, " m, Min: ", m_lin_ks_sigma.pos_uncertainty_min, " m, Max: ", m_lin_ks_sigma.pos_uncertainty_max, " m")
    (m_lin_ks_dyn !== nothing) && println("    LinCov (KS) - RMSE: ", m_lin_ks_dyn.pos_uncertainty_rmse, " m, Min: ", m_lin_ks_dyn.pos_uncertainty_min, " m, Max: ", m_lin_ks_dyn.pos_uncertainty_max, " m")
    (m_mc_binned_ks !== nothing) && println("    Stratified KS CKF - RMSE: ", m_mc_binned_ks.pos_uncertainty_rmse, " m, Min: ", m_mc_binned_ks.pos_uncertainty_min, " m, Max: ", m_mc_binned_ks.pos_uncertainty_max, " m")
    println("  Velocity uncertainty errors:")
    (m_lin !== nothing) && println("    LinCov (Cartesian) - RMSE: ", m_lin.vel_uncertainty_rmse, " m/s, Min: ", m_lin.vel_uncertainty_min, " m/s, Max: ", m_lin.vel_uncertainty_max, " m/s")
    (m_ut !== nothing) && println("    UT (Cartesian) - RMSE: ", m_ut.vel_uncertainty_rmse, " m/s, Min: ", m_ut.vel_uncertainty_min, " m/s, Max: ", m_ut.vel_uncertainty_max, " m/s")
    (m_eigen_cart !== nothing) && println("    CKF (Cartesian) - RMSE: ", m_eigen_cart.vel_uncertainty_rmse, " m/s, Min: ", m_eigen_cart.vel_uncertainty_min, " m/s, Max: ", m_eigen_cart.vel_uncertainty_max, " m/s")
    (m_eigen_ks !== nothing) && println("    CKF (KS) - RMSE: ", m_eigen_ks.vel_uncertainty_rmse, " m/s, Min: ", m_eigen_ks.vel_uncertainty_min, " m/s, Max: ", m_eigen_ks.vel_uncertainty_max, " m/s")
    (m_lin_ks_sigma !== nothing) && println("    CKF (KS Relative) - RMSE: ", m_lin_ks_sigma.vel_uncertainty_rmse, " m/s, Min: ", m_lin_ks_sigma.vel_uncertainty_min, " m/s, Max: ", m_lin_ks_sigma.vel_uncertainty_max, " m/s")
    (m_lin_ks_dyn !== nothing) && println("    LinCov (KS) - RMSE: ", m_lin_ks_dyn.vel_uncertainty_rmse, " m/s, Min: ", m_lin_ks_dyn.vel_uncertainty_min, " m/s, Max: ", m_lin_ks_dyn.vel_uncertainty_max, " m/s")
    (m_mc_binned_ks !== nothing) && println("    Stratified KS CKF - RMSE: ", m_mc_binned_ks.vel_uncertainty_rmse, " m/s, Min: ", m_mc_binned_ks.vel_uncertainty_min, " m/s, Max: ", m_mc_binned_ks.vel_uncertainty_max, " m/s")
end

