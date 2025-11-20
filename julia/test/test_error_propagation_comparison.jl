"""
Test script to compare error propagation methods:
- Monte Carlo (ground truth)
- Sigma Points (KS coordinates)
- Linearized Gaussian (KS coordinates)
- Linearized Gaussian (Cartesian coordinates)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using DifferentialEquations
using SatelliteDynamics
using Plots

const SD = SatelliteDynamics

include("../src/cartesian_dynamics.jl")
include("../src/ks_dynamics.jl")
include("../src/utils.jl")

const GM = SD.GM_EARTH
const R_EARTH = SD.R_EARTH

# Define simulation parameters
const SIM_PARAMS = (
    # Number of orbits to simulate
    num_orbits=1.0,

    # Sampling time (time step) in seconds
    sampling_time=60.0,  # seconds

    # Integrator to use (from DifferentialEquations.jl)
    integrator=Tsit5,

    # Integrator tolerances
    abstol=1e-12,      # Absolute tolerance
    reltol=1e-13,      # Relative tolerance
)

# Define test orbits
test_orbits = [
    (name="Circular LEO (e=0.0, 7128 km altitude)",
        a=7128.1363e3,
        e=0.0,
        i=deg2rad(98.2),
        omega=deg2rad(0.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Circular sun-synchronous orbit"),
]

# Position uncertainty scenarios (meters)
position_uncertainties = [100.0, 1000.0, 10000.0]  # 100m, 1km, 10km

# Number of orbits to test
num_orbits_list = [1.0, 2.0, 5.0]

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
    x_vec_0 = SD.sOSCtoCART(oe_vec; GM=GM, use_degrees=false)
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
        σ_vel = sqrt(GM / (sma + σ_pos))

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
            T_orbital = 2π * sqrt(sma^3 / GM)
            t_0 = 0.0
            t_end = num_orbits * T_orbital
            dt = SIM_PARAMS.sampling_time
            times = collect(t_0:dt:t_end)

            println("\nPropagation parameters:")
            println("  Orbital period: ", T_orbital, " s (", T_orbital / 3600, " hours)")
            println("  Time step: ", dt, " s")
            println("  Number of steps: ", length(times))
            println("  Total propagation time: ", t_end / 3600, " hours")

            # Monte Carlo (ground truth)
            println("\n" * "="^80)
            println("1. MONTE CARLO (Ground Truth)")
            println("="^80)
            x_vec_traj_mean_mc, P_traj_mc, times_mc = propagate_monte_carlo(
                x_vec_0, P_0, times, SIM_PARAMS, GM, 5000)
            println("  Completed: ", length(x_vec_traj_mean_mc), " states")

            # Sigma Points (KS)
            println("\n" * "="^80)
            println("2. SIGMA POINTS (KS Coordinates)")
            println("="^80)
            x_vec_traj_mean_sp, P_traj_sp, times_sp = propagate_ks_keplerian_dynamics_sigma_points(
                x_vec_0, P_0, times, SIM_PARAMS, GM)
            println("  Completed: ", length(x_vec_traj_mean_sp), " states")

            # Linearized Gaussian (KS)
            println("\n" * "="^80)
            println("3. LINEARIZED GAUSSIAN (KS Coordinates)")
            println("="^80)
            x_vec_traj_mean_lin_ks, P_traj_lin_ks, times_lin_ks = propagate_ks_keplerian_dynamics_linearized(
                x_vec_0, P_0, times, SIM_PARAMS, GM)
            println("  Completed: ", length(x_vec_traj_mean_lin_ks), " states")

            # Linearized Gaussian (Cartesian)
            println("\n" * "="^80)
            println("4. LINEARIZED GAUSSIAN (Cartesian Coordinates)")
            println("="^80)
            x_vec_traj_mean_lin_cart, P_traj_lin_cart, times_lin_cart = propagate_cartesian_keplerian_dynamics_linearized(
                x_vec_0, P_0, times, SIM_PARAMS, GM)
            println("  Completed: ", length(x_vec_traj_mean_lin_cart), " states")

            # Compare against Monte Carlo
            println("\n" * "="^80)
            println("COMPARISON AGAINST MONTE CARLO GROUND TRUTH")
            println("="^80)

            # Mean trajectory errors
            println("\nMean trajectory errors (vs Monte Carlo):")
            max_mean_pos_error_sp = 0.0
            max_mean_vel_error_sp = 0.0
            max_mean_pos_error_lin_ks = 0.0
            max_mean_vel_error_lin_ks = 0.0
            max_mean_pos_error_lin_cart = 0.0
            max_mean_vel_error_lin_cart = 0.0

            N = min(length(x_vec_traj_mean_mc), length(x_vec_traj_mean_sp),
                length(x_vec_traj_mean_lin_ks), length(x_vec_traj_mean_lin_cart))

            for i in 1:N
                # Sigma Points
                pos_error_sp = norm(x_vec_traj_mean_sp[i][1:3] - x_vec_traj_mean_mc[i][1:3])
                vel_error_sp = norm(x_vec_traj_mean_sp[i][4:6] - x_vec_traj_mean_mc[i][4:6])
                if pos_error_sp > max_mean_pos_error_sp
                    max_mean_pos_error_sp = pos_error_sp
                    if i <= 5 || pos_error_sp > 1000.0
                        println("    Large error at step $i: pos=$(pos_error_sp) m, sp=$(x_vec_traj_mean_sp[i][1:3]), mc=$(x_vec_traj_mean_mc[i][1:3])")
                    end
                end
                if vel_error_sp > max_mean_vel_error_sp
                    max_mean_vel_error_sp = vel_error_sp
                end

                # Linearized KS
                pos_error_lin_ks = norm(x_vec_traj_mean_lin_ks[i][1:3] - x_vec_traj_mean_mc[i][1:3])
                vel_error_lin_ks = norm(x_vec_traj_mean_lin_ks[i][4:6] - x_vec_traj_mean_mc[i][4:6])
                if pos_error_lin_ks > max_mean_pos_error_lin_ks
                    max_mean_pos_error_lin_ks = pos_error_lin_ks
                end
                if vel_error_lin_ks > max_mean_vel_error_lin_ks
                    max_mean_vel_error_lin_ks = vel_error_lin_ks
                end

                # Linearized Cartesian
                pos_error_lin_cart = norm(x_vec_traj_mean_lin_cart[i][1:3] - x_vec_traj_mean_mc[i][1:3])
                vel_error_lin_cart = norm(x_vec_traj_mean_lin_cart[i][4:6] - x_vec_traj_mean_mc[i][4:6])
                if pos_error_lin_cart > max_mean_pos_error_lin_cart
                    max_mean_pos_error_lin_cart = pos_error_lin_cart
                end
                if vel_error_lin_cart > max_mean_vel_error_lin_cart
                    max_mean_vel_error_lin_cart = vel_error_lin_cart
                end
            end

            println("  Sigma Points:")
            println("    Max position error: ", max_mean_pos_error_sp, " m")
            println("    Max velocity error: ", max_mean_vel_error_sp, " m/s")
            println("  Linearized (KS):")
            println("    Max position error: ", max_mean_pos_error_lin_ks, " m")
            println("    Max velocity error: ", max_mean_vel_error_lin_ks, " m/s")
            println("  Linearized (Cartesian):")
            println("    Max position error: ", max_mean_pos_error_lin_cart, " m")
            println("    Max velocity error: ", max_mean_vel_error_lin_cart, " m/s")

            # Covariance comparison
            println("\nCovariance comparison (Frobenius norm difference from Monte Carlo):")
            max_cov_error_sp = 0.0
            max_cov_error_lin_ks = 0.0
            max_cov_error_lin_cart = 0.0

            cov_error_traj_sp = zeros(N)
            cov_error_traj_lin_ks = zeros(N)
            cov_error_traj_lin_cart = zeros(N)

            for i in 1:N
                # Sigma Points
                cov_diff_sp = P_traj_sp[i] - P_traj_mc[i]
                cov_error_sp = norm(cov_diff_sp)
                cov_error_traj_sp[i] = cov_error_sp
                if cov_error_sp > max_cov_error_sp
                    max_cov_error_sp = cov_error_sp
                end

                # Linearized KS
                cov_diff_lin_ks = P_traj_lin_ks[i] - P_traj_mc[i]
                cov_error_lin_ks = norm(cov_diff_lin_ks)
                cov_error_traj_lin_ks[i] = cov_error_lin_ks
                if cov_error_lin_ks > max_cov_error_lin_ks
                    max_cov_error_lin_ks = cov_error_lin_ks
                end

                # Linearized Cartesian
                cov_diff_lin_cart = P_traj_lin_cart[i] - P_traj_mc[i]
                cov_error_lin_cart = norm(cov_diff_lin_cart)
                cov_error_traj_lin_cart[i] = cov_error_lin_cart
                if cov_error_lin_cart > max_cov_error_lin_cart
                    max_cov_error_lin_cart = cov_error_lin_cart
                end
            end

            println("  Sigma Points:")
            println("    Max covariance error: ", max_cov_error_sp)
            println("  Linearized (KS):")
            println("    Max covariance error: ", max_cov_error_lin_ks)
            println("  Linearized (Cartesian):")
            println("    Max covariance error: ", max_cov_error_lin_cart)

            # Extract 3-sigma bounds
            σ_pos_mc = [sqrt(P_traj_mc[i][1, 1] + P_traj_mc[i][2, 2] + P_traj_mc[i][3, 3]) for i in 1:N]
            σ_pos_sp = [sqrt(P_traj_sp[i][1, 1] + P_traj_sp[i][2, 2] + P_traj_sp[i][3, 3]) for i in 1:N]
            σ_pos_lin_ks = [sqrt(P_traj_lin_ks[i][1, 1] + P_traj_lin_ks[i][2, 2] + P_traj_lin_ks[i][3, 3]) for i in 1:N]
            σ_pos_lin_cart = [sqrt(P_traj_lin_cart[i][1, 1] + P_traj_lin_cart[i][2, 2] + P_traj_lin_cart[i][3, 3]) for i in 1:N]

            σ_vel_mc = [sqrt(P_traj_mc[i][4, 4] + P_traj_mc[i][5, 5] + P_traj_mc[i][6, 6]) for i in 1:N]
            σ_vel_sp = [sqrt(P_traj_sp[i][4, 4] + P_traj_sp[i][5, 5] + P_traj_sp[i][6, 6]) for i in 1:N]
            σ_vel_lin_ks = [sqrt(P_traj_lin_ks[i][4, 4] + P_traj_lin_ks[i][5, 5] + P_traj_lin_ks[i][6, 6]) for i in 1:N]
            σ_vel_lin_cart = [sqrt(P_traj_lin_cart[i][4, 4] + P_traj_lin_cart[i][5, 5] + P_traj_lin_cart[i][6, 6]) for i in 1:N]

            # Create plots
            println("\nGenerating plots...")

            # Plot 1: Mean position comparison
            p1 = plot(times[1:N] ./ 3600, [x_vec_traj_mean_mc[i][1] for i in 1:N],
                label="Monte Carlo (x)", linewidth=2, color=:black, linestyle=:solid)
            plot!(p1, times[1:N] ./ 3600, [x_vec_traj_mean_sp[i][1] for i in 1:N],
                label="Sigma Points (x)", linewidth=2, color=:blue, linestyle=:dash)
            plot!(p1, times[1:N] ./ 3600, [x_vec_traj_mean_lin_ks[i][1] for i in 1:N],
                label="Linearized KS (x)", linewidth=2, color=:red, linestyle=:dot)
            plot!(p1, times[1:N] ./ 3600, [x_vec_traj_mean_lin_cart[i][1] for i in 1:N],
                label="Linearized Cart (x)", linewidth=2, color=:green, linestyle=:dashdot)
            plot!(p1, xlabel="Time (hours)", ylabel="Position X (m)",
                title="Mean Position X Comparison", legend=:topright)

            # Plot 2: Position uncertainty (3-sigma)
            p2 = plot(times[1:N] ./ 3600, 3.0 .* σ_pos_mc, label="Monte Carlo (3σ)",
                linewidth=2, color=:black, linestyle=:solid)
            plot!(p2, times[1:N] ./ 3600, 3.0 .* σ_pos_sp, label="Sigma Points (3σ)",
                linewidth=2, color=:blue, linestyle=:dash)
            plot!(p2, times[1:N] ./ 3600, 3.0 .* σ_pos_lin_ks, label="Linearized KS (3σ)",
                linewidth=2, color=:red, linestyle=:dot)
            plot!(p2, times[1:N] ./ 3600, 3.0 .* σ_pos_lin_cart, label="Linearized Cart (3σ)",
                linewidth=2, color=:green, linestyle=:dashdot)
            plot!(p2, xlabel="Time (hours)", ylabel="Position Uncertainty (m)",
                title="Position Uncertainty (3-sigma)", legend=:topright)

            # Plot 3: Velocity uncertainty (3-sigma)
            p3 = plot(times[1:N] ./ 3600, 3.0 .* σ_vel_mc, label="Monte Carlo (3σ)",
                linewidth=2, color=:black, linestyle=:solid)
            plot!(p3, times[1:N] ./ 3600, 3.0 .* σ_vel_sp, label="Sigma Points (3σ)",
                linewidth=2, color=:blue, linestyle=:dash)
            plot!(p3, times[1:N] ./ 3600, 3.0 .* σ_vel_lin_ks, label="Linearized KS (3σ)",
                linewidth=2, color=:red, linestyle=:dot)
            plot!(p3, times[1:N] ./ 3600, 3.0 .* σ_vel_lin_cart, label="Linearized Cart (3σ)",
                linewidth=2, color=:green, linestyle=:dashdot)
            plot!(p3, xlabel="Time (hours)", ylabel="Velocity Uncertainty (m/s)",
                title="Velocity Uncertainty (3-sigma)", legend=:topright)

            # Plot 4: Covariance error (Frobenius norm)
            p4 = plot(times[1:N] ./ 3600, cov_error_traj_sp, label="Sigma Points",
                linewidth=2, color=:blue, linestyle=:dash)
            plot!(p4, times[1:N] ./ 3600, cov_error_traj_lin_ks, label="Linearized KS",
                linewidth=2, color=:red, linestyle=:dot)
            plot!(p4, times[1:N] ./ 3600, cov_error_traj_lin_cart, label="Linearized Cart",
                linewidth=2, color=:green, linestyle=:dashdot)
            plot!(p4, xlabel="Time (hours)", ylabel="Frobenius Norm",
                title="Covariance Error vs Monte Carlo", legend=:topright, yscale=:log10)

            # Generate filename based on scenario
            filename = "figs/test_error_propagation_comparison_orb$(orbit_idx)_pos$(Int(σ_pos))m_orbits$(num_orbits).png"

            # Combine plots
            p_combined = plot(p1, p2, p3, p4, layout=(4, 1), size=(800, 1600))
            savefig(p_combined, filename)
            println("  Saved plot to ", filename)

            # Store results
            push!(all_results, (
                orbit=orbit.name,
                σ_pos=σ_pos,
                σ_vel=σ_vel,
                num_orbits=num_orbits,
                max_mean_pos_error_sp=max_mean_pos_error_sp,
                max_mean_vel_error_sp=max_mean_vel_error_sp,
                max_mean_pos_error_lin_ks=max_mean_pos_error_lin_ks,
                max_mean_vel_error_lin_ks=max_mean_vel_error_lin_ks,
                max_mean_pos_error_lin_cart=max_mean_pos_error_lin_cart,
                max_mean_vel_error_lin_cart=max_mean_vel_error_lin_cart,
                max_cov_error_sp=max_cov_error_sp,
                max_cov_error_lin_ks=max_cov_error_lin_ks,
                max_cov_error_lin_cart=max_cov_error_lin_cart,
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
    println("  Max position errors - SP: ", result.max_mean_pos_error_sp, " m, Lin KS: ", result.max_mean_pos_error_lin_ks, " m, Lin Cart: ", result.max_mean_pos_error_lin_cart, " m")
    println("  Max velocity errors - SP: ", result.max_mean_vel_error_sp, " m/s, Lin KS: ", result.max_mean_vel_error_lin_ks, " m/s, Lin Cart: ", result.max_mean_vel_error_lin_cart, " m/s")
    println("  Max covariance errors - SP: ", result.max_cov_error_sp, ", Lin KS: ", result.max_cov_error_lin_ks, ", Lin Cart: ", result.max_cov_error_lin_cart)
end

