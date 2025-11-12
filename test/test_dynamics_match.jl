"""
Test script to verify that cartesian and KS dynamics match for gravity-only propagation.
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

const GM = SD.GM_EARTH
const R_EARTH = SD.R_EARTH

# Define test orbits
test_orbits = [
    (name="Circular orbit (750 km altitude)",
        a=750e3 + SD.R_EARTH,
        e=0.0,
        i=deg2rad(98.2),
        omega=deg2rad(0.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Circular orbit"),
    (name="Eccentric orbit (e=0.9, 750 km altitude)",
        a=750e3 + SD.R_EARTH,
        e=0.9,
        i=deg2rad(98.2),
        omega=deg2rad(0.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Highly eccentric orbit"),
    (name="Moderate eccentricity (e=0.5, 750 km altitude)",
        a=750e3 + SD.R_EARTH,
        e=0.5,
        i=deg2rad(98.2),
        omega=deg2rad(0.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Moderate eccentricity"),
    (name="Low Earth orbit (e=0.1, 750 km altitude)",
        a=750e3 + SD.R_EARTH,
        e=0.1,
        i=deg2rad(98.2),
        omega=deg2rad(0.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Low eccentricity"),
]

# Define simulation parameters
const SIM_PARAMS = (
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
)

# Function to test a single orbit
function test_orbit(orbit_name, a, e, i, omega, RAAN, M, GM)
    println("\n" * "="^80)
    println("TESTING: ", orbit_name)
    println("="^80)

    # Convert orbital elements to cartesian using SatelliteDynamics
    oe_vec = [a, e, i, RAAN, omega, M]
    x_vec_0 = SD.sOSCtoCART(oe_vec; GM=GM, use_degrees=false)
    r_vec_0 = x_vec_0[1:3]
    v_vec_0 = x_vec_0[4:6]

    println("\nInitial conditions:")
    println("  Semi-major axis: ", a / 1e3, " km")
    println("  Eccentricity: ", e)
    println("  Inclination: ", i * 180 / π, " deg")
    println("  Argument of periapsis: ", omega * 180 / π, " deg")
    println("  RAAN: ", RAAN * 180 / π, " deg")
    println("  Mean anomaly: ", M * 180 / π, " deg")
    println("  Periapsis distance: ", a * (1 - e) / 1e3, " km")
    println("  Apoapsis distance: ", a * (1 + e) / 1e3, " km")
    println("  Position: ", r_vec_0)
    println("  Velocity: ", v_vec_0)
    println("  Radius: ", norm(r_vec_0) / 1e3, " km")
    println("  Speed: ", norm(v_vec_0) / 1e3, " km/s")

    # Convert to KS coordinates
    ks_state_0 = state_cartesian_to_ks(x_vec_0)
    h_0 = energy_ks(ks_state_0[1:4], ks_state_0[5:8], GM)
    ks_state_augmented_0 = [ks_state_0; h_0]

    # Verify conversion
    x_vec_0_ks_back = state_ks_to_cartesian(ks_state_augmented_0[1:8])
    conv_pos_error = norm(x_vec_0_ks_back[1:3] - r_vec_0)
    conv_vel_error = norm(x_vec_0_ks_back[4:6] - v_vec_0)
    println("\nKS conversion verification:")
    println("  Position error: ", conv_pos_error, " m")
    println("  Velocity error: ", conv_vel_error, " m/s")

    # Propagation time: multiple orbital periods
    T_orbital = 2π * sqrt(a^3 / GM)
    t_0 = 0.0
    t_end = SIM_PARAMS.num_orbits * T_orbital
    dt = SIM_PARAMS.sampling_time
    times = collect(t_0:dt:t_end)

    println("\nPropagation parameters:")
    println("  Orbital period: ", T_orbital, " s (", T_orbital / 3600, " hours)")
    println("  Time step: ", dt, " s")
    println("  Number of steps: ", length(times))

    # Propagate in Cartesian coordinates
    println("\nPropagating in Cartesian coordinates...")
    x_vec_traj, t_traj_cart = propagate_cartesian_keplerian_orbit(x_vec_0, times, SIM_PARAMS, GM)
    println("  Completed: ", length(x_vec_traj), " states")

    # Propagate in KS coordinates
    println("Propagating in KS coordinates...")
    x_vec_traj_ks, ks_state_augmented_traj, t_traj_ks = propagate_ks_keplerian_orbit(ks_state_augmented_0, times, SIM_PARAMS, GM)
    println("  Completed: ", length(ks_state_augmented_traj), " states")

    # Compare the time histories from Cartesian and KS propagations
    time_diffs = [t_traj_cart[i] - t_traj_ks[i] for i in 1:min(length(t_traj_cart), length(t_traj_ks))]
    max_time_diff = maximum(abs.(time_diffs))
    println("\nTime comparison between Cartesian and KS propagations:")
    println("  Maximum absolute time difference: ", max_time_diff, " s")
    println("  Element-wise time differences (first 10): ", time_diffs[1:min(10, end)])

    # Compare trajectories
    max_pos_error = 0.0
    max_vel_error = 0.0
    max_pos_error_idx = 1
    max_vel_error_idx = 1

    for i in 1:min(length(x_vec_traj), length(x_vec_traj_ks))
        x_cart = x_vec_traj[i]
        x_ks = x_vec_traj_ks[i]

        pos_error = norm(x_cart[1:3] - x_ks[1:3])
        vel_error = norm(x_cart[4:6] - x_ks[4:6])

        if pos_error > max_pos_error
            max_pos_error = pos_error
            max_pos_error_idx = i
        end

        if vel_error > max_vel_error
            max_vel_error = vel_error
            max_vel_error_idx = i
        end
    end

    println("\nMaximum errors:")
    println("  Position error: ", max_pos_error, " m (at step ", max_pos_error_idx, ")")
    println("  Velocity error: ", max_vel_error, " m/s (at step ", max_vel_error_idx, ")")

    # Summary
    success = max_pos_error < SIM_PARAMS.max_pos_error_threshold &&
              max_vel_error < SIM_PARAMS.max_vel_error_threshold
    println("\n" * "-"^80)
    if success
        println("✓ SUCCESS: Dynamics match within tolerance!")
    else
        println("⚠ WARNING: Dynamics may not match perfectly")
    end
    println("-"^80)

    return (success=success, max_pos_error=max_pos_error, max_vel_error=max_vel_error, 
        x_vec_traj=x_vec_traj, x_vec_traj_ks=x_vec_traj_ks, times=times)
end

# Run tests for all orbits
println("="^80)
println("KS ERROR PROPAGATION - DYNAMICS MATCHING TEST")
println("="^80)
println("Testing multiple orbits to verify cartesian and KS dynamics match")

results = []
for orbit in test_orbits
    result = test_orbit(orbit.name, orbit.a, orbit.e, orbit.i, orbit.omega, orbit.RAAN, orbit.M, GM)
    push!(results, (orbit=orbit, result=result))
end

# Overall summary
println("\n" * "="^80)
println("OVERALL SUMMARY")
println("="^80)

all_passed = true
for (i, (orbit, result)) in enumerate(results)
    status = result.success ? "✓ PASS" : "✗ FAIL"
    println("\n$(i). $(orbit.name)")
    println("   Status: $(status)")
    println("   Max position error: $(result.max_pos_error) m")
    println("   Max velocity error: $(result.max_vel_error) m/s")
    if !result.success
        global all_passed = false
    end
end

println("\n" * "="^80)
if all_passed
    println("✓ ALL TESTS PASSED!")
else
    println("⚠ SOME TESTS FAILED - Check results above")
end
println("="^80)

# Generate plots for the eccentric orbit (e=0.9)
eccentric_idx = findfirst(o -> o.e == 0.9, test_orbits)
if eccentric_idx !== nothing
    orbit_result = results[eccentric_idx].result
    println("\nGenerating plots for eccentric orbit (e=0.9)...")

    p1 = plot(title="Position Comparison (e=0.9)", xlabel="Time (s)", ylabel="Position (m)")
    plot!(p1, orbit_result.times, [x[1] for x in orbit_result.x_vec_traj],
        label="Cartesian x", linewidth=2)
    plot!(p1, orbit_result.times, [x[1] for x in orbit_result.x_vec_traj_ks],
        label="KS x", linestyle=:dash, linewidth=2)
    plot!(p1, orbit_result.times, [x[2] for x in orbit_result.x_vec_traj],
        label="Cartesian y", linewidth=2)
    plot!(p1, orbit_result.times, [x[2] for x in orbit_result.x_vec_traj_ks],
        label="KS y", linestyle=:dash, linewidth=2)

    p2 = plot(title="Position Error (e=0.9)", xlabel="Time (s)", ylabel="Error (m)")
    pos_errors = [norm(orbit_result.x_vec_traj[i][1:3] - orbit_result.x_vec_traj_ks[i][1:3])
                  for i = 1:length(orbit_result.times)]
    plot!(p2, orbit_result.times, pos_errors, label="Position error", linewidth=2)

    p3 = plot(title="Velocity Error (e=0.9)", xlabel="Time (s)", ylabel="Error (m/s)")
    vel_errors = [norm(orbit_result.x_vec_traj[i][4:6] - orbit_result.x_vec_traj_ks[i][4:6])
                  for i = 1:length(orbit_result.times)]
    plot!(p3, orbit_result.times, vel_errors, label="Velocity error", linewidth=2)

    p_combined = plot(p1, p2, p3, layout=(3, 1), size=(800, 1200))
    savefig(p_combined, "figs/dynamics_comparison.png")
    println("  Saved plot to figs/dynamics_comparison.png")
end

