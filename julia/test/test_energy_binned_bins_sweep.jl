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
        description="LEO",
        id="leo"),
    (name="Molniya orbit (e=0.74, 26600 km semi-major axis)",
        a=26600e3,
        e=0.74,
        i=deg2rad(63.4),
        omega=deg2rad(270.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Molniya",
        id="mol"),
]

# Position uncertainty scenarios (meters)
position_uncertainties = [1e3, 1e4, 1e5]  # 1km

# Number of orbits to test
num_orbits_list = [3.0]

# Number of energy bins to test
num_energy_bins_list = [1, 2, 5, 10, 20, 50]

# Monte Carlo parameters
num_mc_samples_ground_truth = 5000  # Number of samples in saved MC data file (must match save_monte_carlo_npz.jl)
num_mc_samples_binning = 5000  # For energy binning method

println("="^80)
println("ENERGY-BINNED HYBRID PROPAGATION: BINS SWEEP TEST")
println("="^80)
println("Comparing Hybrid Energy-Binned KS Sigma Points method against Monte Carlo")
println("while varying the number of energy bins")
println("\nTest configuration:")
println("  Number of orbits: ", num_orbits_list)
println("  Position uncertainties: ", position_uncertainties, " m")
println("  Number of energy bins to test: ", num_energy_bins_list)
println("  MC samples (ground truth, loaded from file): ", num_mc_samples_ground_truth)
println("  MC samples (binning method): ", num_mc_samples_binning)
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
        # Compute velocity uncertainty from the Uncertainty Propagation Law
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

            # Monte Carlo (ground truth) - load from npz file
            println("\n" * "="^80)
            println("MONTE CARLO (Ground Truth) - Loading from file")
            println("="^80)
            
            # Construct filename matching save_monte_carlo_npz.jl pattern
            data_dir = joinpath(@__DIR__, "..", "data")
            mc_filename = "mc_$(orbit.id)_num_orbits$(Int(num_orbits))_std_pos$(Int(σ_pos))m_std_vel$(round(σ_vel, digits=6))mps_num_samples$(Int(num_mc_samples_ground_truth)).npz"
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

            # Test Hybrid Energy-Binned KS Sigma Points method with different numbers of bins
            results_by_bins = Dict{Int,NamedTuple}()

            for num_bins in num_energy_bins_list
                method_name = "HYBRID ENERGY-BINNED KS SIGMA POINTS (num_bins=$num_bins)"
                result = run_method_safely(method_name,
                    propagate_uncertainty_via_energy_binned_mc_then_ks_sigma_points,
                    x_vec_0, P_0, times, sim_params;
                    num_mc_samples=num_mc_samples_binning,
                    num_energy_bins=num_bins,
                    drop_edge_bins=true,
                    verbose=true)
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

            # Plot 1: Position error vs number of bins
            p1 = plot(xlabel="Time (hours)", ylabel="Position Error (m)", yscale=:log10, legend=:topleft, title="Position Error vs Number of Energy Bins")
            for (idx, num_bins) in enumerate(num_energy_bins_list)
                result = results_by_bins[num_bins]
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    plot!(p1, times[1:N] ./ 3600, metrics.pos_errors,
                        label="$(num_bins) bins",
                        linewidth=2)
                end
            end

            # Plot 2: Velocity error vs number of bins
            p2 = plot(xlabel="Time (hours)", ylabel="Velocity Error (m/s)", yscale=:log10, legend=:topleft, title="Velocity Error vs Number of Energy Bins")
            for (idx, num_bins) in enumerate(num_energy_bins_list)
                result = results_by_bins[num_bins]
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    plot!(p2, times[1:N] ./ 3600, metrics.vel_errors,
                        label="$(num_bins) bins",
                        linewidth=2)
                end
            end

            # Plot 3: Position uncertainty error vs number of bins
            p3 = plot(xlabel="Time (hours)", ylabel="Position Uncertainty Error (m)", yscale=:log10, legend=:topleft, title="Position Uncertainty Error vs Number of Energy Bins")
            for (idx, num_bins) in enumerate(num_energy_bins_list)
                result = results_by_bins[num_bins]
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    plot!(p3, times[1:N] ./ 3600, metrics.pos_uncertainty_errors,
                        label="$(num_bins) bins",
                        linewidth=2)
                end
            end

            # Plot 4: Velocity uncertainty error vs number of bins
            p4 = plot(xlabel="Time (hours)", ylabel="Velocity Uncertainty Error (m/s)", yscale=:log10, legend=:topleft, title="Velocity Uncertainty Error vs Number of Energy Bins")
            for (idx, num_bins) in enumerate(num_energy_bins_list)
                result = results_by_bins[num_bins]
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing
                    plot!(p4, times[1:N] ./ 3600, metrics.vel_uncertainty_errors,
                        label="$(num_bins) bins",
                        linewidth=2)
                end
            end

            # Plot 5: KL divergence vs number of bins
            p5 = plot(xlabel="Time (hours)", ylabel="KL Divergence", yscale=:log10, legend=:topleft, title="KL Divergence vs Number of Energy Bins")
            for (idx, num_bins) in enumerate(num_energy_bins_list)
                result = results_by_bins[num_bins]
                metrics = metrics_by_bins[num_bins]
                if metrics !== nothing && !result.failed
                    kl_vals = [gaussian_kl_divergence("Hybrid Energy-Binned (num_bins=$num_bins)", times[i],
                        x_vec_traj_mean_mc[i], P_traj_mc[i],
                        result.x_vec_traj[i], result.P_traj[i]) for i in 1:N]
                    plot!(p5, times[1:N] ./ 3600, kl_vals,
                        label="$(num_bins) bins",
                        linewidth=2)
                end
            end

            # Plot 6: RMSE vs number of bins (summary)
            p6 = plot(xlabel="Number of Energy Bins", ylabel="RMSE (m or mm/s)", yscale=:log10, legend=:topleft, title="RMSE vs Number of Energy Bins")
            pos_rmse_vals = [metrics_by_bins[b] !== nothing ? metrics_by_bins[b].pos_rmse : NaN for b in num_energy_bins_list]
            vel_rmse_vals = [metrics_by_bins[b] !== nothing ? metrics_by_bins[b].vel_rmse : NaN for b in num_energy_bins_list]
            plot!(p6, num_energy_bins_list, pos_rmse_vals, label="Position RMSE (m)", marker=:circle, linewidth=2, color=:blue)
            plot!(p6, num_energy_bins_list, vel_rmse_vals .* 1e3, label="Velocity RMSE (mm/s)", marker=:square, linewidth=2, color=:red)

            # Generate filename based on scenario
            filename = "figs/test_energy_binned_bins_sweep_$(orbit.id)_num_orbits$(Int(num_orbits))_std_pos$(Int(σ_pos))m_num_bins$(join(num_energy_bins_list, "-")).png"

            # Combine plots
            p_combined = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1400, 2400), left_margin=50Plots.px)
            savefig(p_combined, filename)
            println("  Saved plot to ", filename)

            # Store results
            push!(all_results, (
                orbit=orbit.name,
                σ_pos=σ_pos,
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
    end  # end position uncertainty loop
end  # end orbit loop

println("\n" * "="^80)
println("ALL TESTS COMPLETE")
println("="^80)
println("\nSummary of all scenarios:")
for (idx, result) in enumerate(all_results)
    println("\nScenario $idx:")
    println("  Orbit: ", result.orbit)
    println("  Position uncertainty: ", result.σ_pos, " m")
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
