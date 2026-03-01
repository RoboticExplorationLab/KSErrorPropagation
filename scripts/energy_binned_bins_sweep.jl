using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots
using LinearAlgebra
using SatelliteDynamics
using DifferentialEquations
using Printf
using NPZ

const SD = SatelliteDynamics

include("../src/cartesian_dynamics.jl")
include("../src/ks_dynamics.jl")
include("../src/error_propagation.jl")
include("../src/utils.jl")

# Load shared configuration
include(joinpath(@__DIR__, "..", "config", "default.jl"))

# Save approach results to out/ (default: false). Usage: julia script.jl [save]
save_to_out = length(ARGS) >= 1 && (ARGS[1] == "save" || lowercase(ARGS[1]) == "true")

using Random
Random.seed!(RANDOM_SEED)

# Local aliases (config uses UPPER_CASE constants)
test_orbits = TEST_ORBITS
oe_initial_std_scenarios = OE_INITIAL_STD_SCENARIOS
num_orbits_list = NUM_ORBITS_LIST
num_energy_bins_list = NUM_ENERGY_BINS_LIST
num_mc_samples_ground_truth = NUM_MC_SAMPLES
num_mc_samples_binning = NUM_MC_SAMPLES_BINNING
min_samples_threshold = MIN_SAMPLES_THRESHOLD

fname_num(x) = isinteger(x) ? string(Int(x)) : string(x)

println("="^80)
println("ENERGY-STRATIFIED KS CKF: BINS SWEEP TEST")
println("="^80)
println("Comparing Energy-Stratified KS CKF method against Monte Carlo")
println("while varying the number of energy bins")
println("\nTest configuration:")
println("  Number of orbits: ", num_orbits_list)
println("  OE initial std scenarios: ", length(oe_initial_std_scenarios))
println("  Number of energy bins to test: ", num_energy_bins_list)
println("  MC samples (ground truth, loaded from file): ", num_mc_samples_ground_truth)
println("  MC samples (binning method): ", num_mc_samples_binning)
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

        P_0 = compute_P0_from_oe_samples(oe_vec_0, oe_std, SIM_PARAMS.GM, SIM_PARAMS.R_EARTH)
        σ_vel = sqrt((P_0[4, 4] + P_0[5, 5] + P_0[6, 6]) / 3)  # RMS velocity uncertainty (m/s) from initial covariance

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
                    if length(result) == 3
                        x_vec_traj, P_traj, sigma_points_traj = result
                    else
                        x_vec_traj, P_traj = result
                        sigma_points_traj = nothing
                    end
                    println("  Completed: ", length(x_vec_traj), " states")
                    return (x_vec_traj=x_vec_traj, P_traj=P_traj, sigma_points_traj=sigma_points_traj, failed=false, error_msg=nothing, error=nothing)
                catch e
                    println("  ✗ FAILED: ", method_name)
                    println("  Error: ", typeof(e).name.name, ": ", sprint(showerror, e))
                    return (x_vec_traj=Vector{Vector{Float64}}(), P_traj=Vector{Matrix{Float64}}(), sigma_points_traj=nothing, failed=true, error_msg=sprint(showerror, e), error=e)
                end
            end

            # Monte Carlo (ground truth) - load from npz file
            println("\n" * "="^80)
            println("MONTE CARLO (Ground Truth) - Loading from file")
            println("="^80)
            
            # Construct filename matching save_monte_carlo_npz.jl pattern
            data_dir = joinpath(@__DIR__, "..", "data")
            mc_filename = "mc_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_num_samples$(Int(num_mc_samples_ground_truth)).npz"
            mc_filepath = joinpath(data_dir, mc_filename)
            
            if !isfile(mc_filepath)
                error("Monte Carlo data file not found: $mc_filepath\nPlease run scripts/save_monte_carlo_npz.jl first to generate the data.")
            end
            
            println("  Loading from: ", mc_filepath)
            npz_data = NPZ.npzread(mc_filepath)
            
            # Extract data
            x_array = npz_data["x"]  # (N, 6)
            P_array = npz_data["P"]  # (N, 6, 6)
            timestamp = npz_data["timestamp"]  # (N,)
            
            # Convert to expected format: Vector{Vector{Float64}} and Vector{Matrix{Float64}}
            N_loaded = size(x_array, 1)
            x_vec_traj_mean_mc = [x_array[i, :] for i in 1:N_loaded]
            P_traj_mc = [P_array[i, :, :] for i in 1:N_loaded]
            
            # Verify times match (or use loaded timestamp)
            if length(timestamp) != length(times) || maximum(abs.(timestamp .- times)) > 1e-6
                println("  Warning: Loaded timestamp differs from expected times. Using loaded timestamp.")
                times = timestamp
            end
            
            println("  Loaded: ", length(x_vec_traj_mean_mc), " states")

            # Test Energy-Stratified KS CKF method with different numbers of bins
            results_by_bins = Dict{Int,NamedTuple}()

            for num_bins in num_energy_bins_list
                method_name = "ENERGY-STRATIFIED KS CKF (num_bins=$num_bins)"
                result = run_method_safely(method_name,
                    propagate_uncertainty_via_energy_binned_mc_then_ks_sigma_points,
                    x_vec_0, P_0, times, sim_params;
                    num_mc_samples=num_mc_samples_binning,
                    num_energy_bins=num_bins,
                    min_samples_threshold=min_samples_threshold,
                    verbose=true,
                    return_samples=true,
                    oe_std=oe_std)
                results_by_bins[num_bins] = result
            end

            # Compare against Monte Carlo
            println("\n" * "="^80)
            println("COMPARISON AGAINST MONTE CARLO GROUND TRUTH")
            println("="^80)

            # Compute error metrics for each number of bins
            metrics_by_bins = Dict{Int,Any}()

            for num_bins in num_energy_bins_list
                result = results_by_bins[num_bins]
                if !result.failed
                    metrics = error_metrics(x_vec_traj_mean_mc, P_traj_mc, result.x_vec_traj, result.P_traj)
                    metrics_by_bins[num_bins] = metrics
                else
                    metrics_by_bins[num_bins] = nothing
                end
            end

            # Print summary table
            println("\n" * "-"^80)
            println("SUMMARY: Position Errors (vs Monte Carlo)")
            println("-"^80)
            println("Num Bins | RMSE (m) | Min (m) | Max (m)")
            println("-"^80)
            for num_bins in num_energy_bins_list
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    println("  $num_bins   | ",
                        @sprintf("%8.2f", metrics.pos_rmse), " | ",
                        @sprintf("%8.2f", metrics.pos_min), " | ",
                        @sprintf("%8.2f", metrics.pos_max))
                else
                    println("  $num_bins   | FAILED")
                end
            end

            println("\n" * "-"^80)
            println("SUMMARY: Velocity Errors (vs Monte Carlo)")
            println("-"^80)
            println("Num Bins | RMSE (m/s) | Min (m/s) | Max (m/s)")
            println("-"^80)
            for num_bins in num_energy_bins_list
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    println("  $num_bins   | ",
                        @sprintf("%10.6f", metrics.vel_rmse), " | ",
                        @sprintf("%10.6f", metrics.vel_min), " | ",
                        @sprintf("%10.6f", metrics.vel_max))
                else
                    println("  $num_bins   | FAILED")
                end
            end

            println("\n" * "-"^80)
            println("SUMMARY: Position Uncertainty Errors (vs Monte Carlo)")
            println("-"^80)
            println("Num Bins | RMSE (m) | Min (m) | Max (m)")
            println("-"^80)
            for num_bins in num_energy_bins_list
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    println("  $num_bins   | ",
                        @sprintf("%8.2f", metrics.pos_uncertainty_rmse), " | ",
                        @sprintf("%8.2f", metrics.pos_uncertainty_min), " | ",
                        @sprintf("%8.2f", metrics.pos_uncertainty_max))
                else
                    println("  $num_bins   | FAILED")
                end
            end

            println("\n" * "-"^80)
            println("SUMMARY: Velocity Uncertainty Errors (vs Monte Carlo)")
            println("-"^80)
            println("Num Bins | RMSE (m/s) | Min (m/s) | Max (m/s)")
            println("-"^80)
            for num_bins in num_energy_bins_list
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    println("  $num_bins   | ",
                        @sprintf("%10.6f", metrics.vel_uncertainty_rmse), " | ",
                        @sprintf("%10.6f", metrics.vel_uncertainty_min), " | ",
                        @sprintf("%10.6f", metrics.vel_uncertainty_max))
                else
                    println("  $num_bins   | FAILED")
                end
            end

            # Create plots comparing different numbers of bins
            println("\nGenerating plots...")

            # Find minimum length across all successful methods
            lengths_to_check = [length(x_vec_traj_mean_mc)]
            for num_bins in num_energy_bins_list
                result = results_by_bins[num_bins]
                if !result.failed
                    push!(lengths_to_check, length(result.x_vec_traj))
                end
            end
            N = minimum(lengths_to_check)

            # Color/linestyle palette for bin counts (matches error_propagation_comparison style)
            bin_styles = [
                (color=:green, linestyle=:dash),
                (color=:blue, linestyle=:dot),
                (color=:red, linestyle=:dashdot),
                (color=:orange, linestyle=:dashdotdot),
                (color=:purple, linestyle=:dashdotdot),
                (color=:black, linestyle=:dash),
            ]

            # Plot 1: Position error (same structure as error_propagation_comparison)
            p1 = plot(xlabel="Time (hours)", ylabel="Position Error (m)", yscale=:log10, legend=:topleft)
            for (idx, num_bins) in enumerate(num_energy_bins_list)
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    style = bin_styles[mod1(idx, length(bin_styles))]
                    plot!(p1, times[1:N] ./ 3600, metrics.pos_errors,
                        label="$(num_bins) bins",
                        linewidth=2, color=style.color, linestyle=style.linestyle)
                end
            end

            # Plot 2: Velocity error
            p2 = plot(xlabel="Time (hours)", ylabel="Velocity Error (m/s)", yscale=:log10, legend=:topleft)
            for (idx, num_bins) in enumerate(num_energy_bins_list)
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    style = bin_styles[mod1(idx, length(bin_styles))]
                    plot!(p2, times[1:N] ./ 3600, metrics.vel_errors,
                        label="$(num_bins) bins",
                        linewidth=2, color=style.color, linestyle=style.linestyle)
                end
            end

            # Plot 3: Position uncertainty error
            p3 = plot(xlabel="Time (hours)", ylabel="Position Uncertainty Error (m)", yscale=:log10, legend=:topleft)
            for (idx, num_bins) in enumerate(num_energy_bins_list)
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    style = bin_styles[mod1(idx, length(bin_styles))]
                    plot!(p3, times[1:N] ./ 3600, metrics.pos_uncertainty_errors,
                        label="$(num_bins) bins",
                        linewidth=2, color=style.color, linestyle=style.linestyle)
                end
            end

            # Plot 4: Velocity uncertainty error
            p4 = plot(xlabel="Time (hours)", ylabel="Velocity Uncertainty Error (m/s)", yscale=:log10, legend=:topleft)
            for (idx, num_bins) in enumerate(num_energy_bins_list)
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    style = bin_styles[mod1(idx, length(bin_styles))]
                    plot!(p4, times[1:N] ./ 3600, metrics.vel_uncertainty_errors,
                        label="$(num_bins) bins",
                        linewidth=2, color=style.color, linestyle=style.linestyle)
                end
            end

            # Plot 5: KL divergence
            p5 = plot(xlabel="Time (hours)", ylabel="KL Divergence", yscale=:log10, legend=:topleft)
            for (idx, num_bins) in enumerate(num_energy_bins_list)
                result = results_by_bins[num_bins]
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing && !result.failed
                    kl_vals = [gaussian_kl_divergence("Stratified KS CKF (num_bins=$num_bins)", times[i],
                        x_vec_traj_mean_mc[i], P_traj_mc[i],
                        result.x_vec_traj[i], result.P_traj[i]) for i in 1:N]
                    style = bin_styles[mod1(idx, length(bin_styles))]
                    plot!(p5, times[1:N] ./ 3600, kl_vals,
                        label="$(num_bins) bins",
                        linewidth=2, color=style.color, linestyle=style.linestyle)
                end
            end

            # Generate filename and save (5 vertically stacked plots, same structure as error_propagation_comparison)
            figdir = joinpath(@__DIR__, "..", "figs")
            mkpath(figdir)
            filename = joinpath(figdir, "energy_binned_bins_sweep_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_num_bins$(join(num_energy_bins_list, "-")).png")

            # Combine plots: 5 stacked vertically, size matching error_propagation_comparison
            p_combined = plot(p1, p2, p3, p4, p5, layout=(5, 1), size=(1000, 2000), left_margin=50Plots.px)
            savefig(p_combined, filename)
            println("  Saved plot to ", filename)

            # Save approach results to out/
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
                    filepath = joinpath(out_dir, approach_id * ".npz")
                    NPZ.npzwrite(filepath, npz_dict)
                    println("  Saved: ", filepath)
                end

                println("\n" * "="^80)
                println("SAVING APPROACH RESULTS TO out/")
                println("="^80)
                save_approach_npz(out_dir, "mc_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m", orbit, num_orbits, oe_std, times, x_vec_traj_mean_mc, P_traj_mc)
                for num_bins in num_energy_bins_list
                    result = results_by_bins[num_bins]
                    if !result.failed
                        approach_id = "stratified_ks_num_bins$(num_bins)_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m"
                        save_approach_npz(out_dir, approach_id, orbit, num_orbits, oe_std, times, result.x_vec_traj, result.P_traj; sigma_points_traj=result.sigma_points_traj)
                    end
                end
            end

            # Store results
            push!(all_results, (
                orbit=orbit.name,
                oe_std=oe_std,
                σ_vel=σ_vel,
                num_orbits=num_orbits,
                num_energy_bins_list=num_energy_bins_list,
                results_by_bins=results_by_bins,
                metrics_by_bins=metrics_by_bins,
            ))

            println("\n" * "="^80)
            println("SCENARIO COMPLETE")
            println("="^80)
        end  # end num_orbits loop
    end  # end OE initial std scenario loop
end  # end orbit loop

println("\n" * "="^80)
println("ALL TESTS COMPLETE")
println("="^80)
println("\nSummary of all scenarios:")
for (idx, result) in enumerate(all_results)
    println("\nScenario $idx:")
    println("  Orbit: ", result.orbit)
    println("  OE initial std (σ_a): ", result.oe_std[1], " m")
    println("  Velocity uncertainty: ", result.σ_vel, " m/s")
    println("  Number of orbits: ", result.num_orbits)
    println("  Number of energy bins tested: ", result.num_energy_bins_list)
    println("  Results:")
    for num_bins in result.num_energy_bins_list
        metrics = result.metrics_by_bins[num_bins]
        if metrics !== nothing
            println("    $(num_bins) bins:")
            println("      Position RMSE: ", metrics.pos_rmse, " m")
            println("      Velocity RMSE: ", metrics.vel_rmse, " m/s")
            println("      Position Uncertainty RMSE: ", metrics.pos_uncertainty_rmse, " m")
            println("      Velocity Uncertainty RMSE: ", metrics.vel_uncertainty_rmse, " m/s")
        else
            println("    $(num_bins) bins: FAILED")
        end
    end
end
