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
include("../src/utils.jl")

const GM = SD.GM_EARTH
const R_EARTH = SD.R_EARTH

# Define test orbits
test_orbits = [
    (name="Circular orbit (750 km altitude)",
        a=750e3 + R_EARTH,
        e=0.0,
        i=deg2rad(0.0),
        omega=deg2rad(0.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Circular orbit"),
    (name="Eccentric orbit (e=0.9, 750 km altitude)",
        a=750e3 + R_EARTH,
        e=0.9,
        i=deg2rad(0.0),#i=deg2rad(98.2),
        omega=deg2rad(0.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Highly eccentric orbit"),
    (name="Moderate eccentricity (e=0.5, 750 km altitude)",
        a=750e3 + R_EARTH,
        e=0.5,
        i=deg2rad(0.0),
        omega=deg2rad(0.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Moderate eccentricity"),
    (name="Low Earth orbit (e=0.1, 750 km altitude)",
        a=750e3 + R_EARTH,
        e=0.1,
        i=deg2rad(0.0),
        omega=deg2rad(0.0),
        RAAN=deg2rad(0.0),
        M=deg2rad(0.0),
        description="Low eccentricity"),
]

# Define simulation parameters
const SIM_PARAMS = (
    # Number of orbits to simulate
    num_orbits=3,

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

    # Compute analytical solution
    println("Computing analytical solution...")
    oe_vec_0 = [a, e, i, RAAN, omega, M]
    x_vec_traj_analytical = propagate_analytical_keplerian_orbit(oe_vec_0, times, GM)
    println("  Completed: ", length(x_vec_traj_analytical), " states")

    # Compare the time histories from Cartesian and KS propagations
    time_diffs = [t_traj_cart[i] - t_traj_ks[i] for i in 1:min(length(t_traj_cart), length(t_traj_ks))]
    max_time_diff = maximum(abs.(time_diffs))
    println("\nTime comparison between Cartesian and KS propagations:")
    println("  Maximum absolute time difference: ", max_time_diff, " s")
    println("  Element-wise time differences (first 10): ", time_diffs[1:min(10, end)])

    # Compare trajectories: Cartesian vs KS
    max_pos_error_cart_ks = 0.0
    max_vel_error_cart_ks = 0.0
    max_pos_error_idx_cart_ks = 1
    max_vel_error_idx_cart_ks = 1

    # Compare trajectories: Cartesian vs Analytical
    max_pos_error_cart_analytical = 0.0
    max_vel_error_cart_analytical = 0.0
    max_pos_error_idx_cart_analytical = 1
    max_vel_error_idx_cart_analytical = 1

    # Compare trajectories: KS vs Analytical
    max_pos_error_ks_analytical = 0.0
    max_vel_error_ks_analytical = 0.0
    max_pos_error_idx_ks_analytical = 1
    max_vel_error_idx_ks_analytical = 1

    N = min(length(x_vec_traj), length(x_vec_traj_ks), length(x_vec_traj_analytical))
    for i in 1:N
        x_cart = x_vec_traj[i]
        x_ks = x_vec_traj_ks[i]
        x_analytical = x_vec_traj_analytical[i]

        # Cartesian vs KS
        pos_error_cart_ks = norm(x_cart[1:3] - x_ks[1:3])
        vel_error_cart_ks = norm(x_cart[4:6] - x_ks[4:6])
        if pos_error_cart_ks > max_pos_error_cart_ks
            max_pos_error_cart_ks = pos_error_cart_ks
            max_pos_error_idx_cart_ks = i
        end
        if vel_error_cart_ks > max_vel_error_cart_ks
            max_vel_error_cart_ks = vel_error_cart_ks
            max_vel_error_idx_cart_ks = i
        end

        # Cartesian vs Analytical
        pos_error_cart_analytical = norm(x_cart[1:3] - x_analytical[1:3])
        vel_error_cart_analytical = norm(x_cart[4:6] - x_analytical[4:6])
        if pos_error_cart_analytical > max_pos_error_cart_analytical
            max_pos_error_cart_analytical = pos_error_cart_analytical
            max_pos_error_idx_cart_analytical = i
        end
        if vel_error_cart_analytical > max_vel_error_cart_analytical
            max_vel_error_cart_analytical = vel_error_cart_analytical
            max_vel_error_idx_cart_analytical = i
        end

        # KS vs Analytical
        pos_error_ks_analytical = norm(x_ks[1:3] - x_analytical[1:3])
        vel_error_ks_analytical = norm(x_ks[4:6] - x_analytical[4:6])
        if pos_error_ks_analytical > max_pos_error_ks_analytical
            max_pos_error_ks_analytical = pos_error_ks_analytical
            max_pos_error_idx_ks_analytical = i
        end
        if vel_error_ks_analytical > max_vel_error_ks_analytical
            max_vel_error_ks_analytical = vel_error_ks_analytical
            max_vel_error_idx_ks_analytical = i
        end
    end

    println("\nMaximum errors (Cartesian vs KS):")
    println("  Position error: ", max_pos_error_cart_ks, " m (at step ", max_pos_error_idx_cart_ks, ")")
    println("  Velocity error: ", max_vel_error_cart_ks, " m/s (at step ", max_vel_error_idx_cart_ks, ")")

    println("\nMaximum errors (Cartesian vs Analytical):")
    println("  Position error: ", max_pos_error_cart_analytical, " m (at step ", max_pos_error_idx_cart_analytical, ")")
    println("  Velocity error: ", max_vel_error_cart_analytical, " m/s (at step ", max_vel_error_idx_cart_analytical, ")")

    println("\nMaximum errors (KS vs Analytical):")
    println("  Position error: ", max_pos_error_ks_analytical, " m (at step ", max_pos_error_idx_ks_analytical, ")")
    println("  Velocity error: ", max_vel_error_ks_analytical, " m/s (at step ", max_vel_error_idx_ks_analytical, ")")

    # Summary
    success_cart_ks = max_pos_error_cart_ks < SIM_PARAMS.max_pos_error_threshold &&
                      max_vel_error_cart_ks < SIM_PARAMS.max_vel_error_threshold
    success_cart_analytical = max_pos_error_cart_analytical < SIM_PARAMS.max_pos_error_threshold &&
                              max_vel_error_cart_analytical < SIM_PARAMS.max_vel_error_threshold
    success_ks_analytical = max_pos_error_ks_analytical < SIM_PARAMS.max_pos_error_threshold &&
                            max_vel_error_ks_analytical < SIM_PARAMS.max_vel_error_threshold

    println("\n" * "-"^80)
    if success_cart_ks
        println("✓ SUCCESS: Cartesian and KS dynamics match within tolerance!")
    else
        println("⚠ WARNING: Cartesian and KS dynamics may not match perfectly")
    end
    if success_cart_analytical
        println("✓ SUCCESS: Cartesian matches analytical solution within tolerance!")
    else
        println("⚠ WARNING: Cartesian may not match analytical solution perfectly")
    end
    if success_ks_analytical
        println("✓ SUCCESS: KS matches analytical solution within tolerance!")
    else
        println("⚠ WARNING: KS may not match analytical solution perfectly")
    end
    println("-"^80)

    return (success_cart_ks=success_cart_ks,
        success_cart_analytical=success_cart_analytical,
        success_ks_analytical=success_ks_analytical,
        max_pos_error_cart_ks=max_pos_error_cart_ks,
        max_vel_error_cart_ks=max_vel_error_cart_ks,
        max_pos_error_cart_analytical=max_pos_error_cart_analytical,
        max_vel_error_cart_analytical=max_vel_error_cart_analytical,
        max_pos_error_ks_analytical=max_pos_error_ks_analytical,
        max_vel_error_ks_analytical=max_vel_error_ks_analytical,
        x_vec_traj=x_vec_traj,
        x_vec_traj_ks=x_vec_traj_ks,
        x_vec_traj_analytical=x_vec_traj_analytical,
        times=times)
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

all_cart_ks_passed = true
all_cart_analytical_passed = true
all_ks_analytical_passed = true

for (i, (orbit, result)) in enumerate(results)
    status_cart_ks = result.success_cart_ks ? "✓ PASS" : "✗ FAIL"
    status_cart_analytical = result.success_cart_analytical ? "✓ PASS" : "✗ FAIL"
    status_ks_analytical = result.success_ks_analytical ? "✓ PASS" : "✗ FAIL"

    println("\n$(i). $(orbit.name)")
    println("   Cartesian vs KS: $(status_cart_ks)")
    println("      Max position error: $(result.max_pos_error_cart_ks) m")
    println("      Max velocity error: $(result.max_vel_error_cart_ks) m/s")
    println("   Cartesian vs Analytical: $(status_cart_analytical)")
    println("      Max position error: $(result.max_pos_error_cart_analytical) m")
    println("      Max velocity error: $(result.max_vel_error_cart_analytical) m/s")
    println("   KS vs Analytical: $(status_ks_analytical)")
    println("      Max position error: $(result.max_pos_error_ks_analytical) m")
    println("      Max velocity error: $(result.max_vel_error_ks_analytical) m/s")

    if !result.success_cart_ks
        global all_cart_ks_passed = false
    end
    if !result.success_cart_analytical
        global all_cart_analytical_passed = false
    end
    if !result.success_ks_analytical
        global all_ks_analytical_passed = false
    end
end

println("\n" * "="^80)
if all_cart_ks_passed && all_cart_analytical_passed && all_ks_analytical_passed
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

    # Helper function to determine order of magnitude for normalization
    function get_order_of_magnitude(values)
        max_abs = maximum(abs.(values))
        if max_abs == 0.0
            return 0
        end
        return Int(floor(log10(max_abs)))
    end

    # Position data
    pos_x_cart = [x[1] for x in orbit_result.x_vec_traj]
    pos_x_ks = [x[1] for x in orbit_result.x_vec_traj_ks]
    pos_x_analytical = [x[1] for x in orbit_result.x_vec_traj_analytical]
    pos_y_cart = [x[2] for x in orbit_result.x_vec_traj]
    pos_y_ks = [x[2] for x in orbit_result.x_vec_traj_ks]
    pos_y_analytical = [x[2] for x in orbit_result.x_vec_traj_analytical]

    all_pos_values = vcat(pos_x_cart, pos_x_ks, pos_x_analytical, pos_y_cart, pos_y_ks, pos_y_analytical)
    pos_order = get_order_of_magnitude(all_pos_values)
    pos_scale = 10.0^pos_order

    # Position errors
    pos_errors_cart_ks = [norm(orbit_result.x_vec_traj[i][1:3] - orbit_result.x_vec_traj_ks[i][1:3])
                          for i = 1:length(orbit_result.times)]
    pos_errors_cart_analytical = [norm(orbit_result.x_vec_traj[i][1:3] - orbit_result.x_vec_traj_analytical[i][1:3])
                                  for i = 1:length(orbit_result.times)]
    pos_errors_ks_analytical = [norm(orbit_result.x_vec_traj_ks[i][1:3] - orbit_result.x_vec_traj_analytical[i][1:3])
                                for i = 1:length(orbit_result.times)]

    all_pos_errors = vcat(pos_errors_cart_ks, pos_errors_cart_analytical, pos_errors_ks_analytical)
    pos_error_order = get_order_of_magnitude(all_pos_errors)
    pos_error_scale = 10.0^pos_error_order

    # Velocity errors
    vel_errors_cart_ks = [norm(orbit_result.x_vec_traj[i][4:6] - orbit_result.x_vec_traj_ks[i][4:6])
                          for i = 1:length(orbit_result.times)]
    vel_errors_cart_analytical = [norm(orbit_result.x_vec_traj[i][4:6] - orbit_result.x_vec_traj_analytical[i][4:6])
                                  for i = 1:length(orbit_result.times)]
    vel_errors_ks_analytical = [norm(orbit_result.x_vec_traj_ks[i][4:6] - orbit_result.x_vec_traj_analytical[i][4:6])
                                for i = 1:length(orbit_result.times)]

    all_vel_errors = vcat(vel_errors_cart_ks, vel_errors_cart_analytical, vel_errors_ks_analytical)
    vel_error_order = get_order_of_magnitude(all_vel_errors)
    vel_error_scale = 10.0^vel_error_order

    # Position comparison plot
    pos_ylabel = pos_order == 0 ? "Position (m)" : "Position (1e$(pos_order) m)"
    p1 = plot(title="Position Comparison (e=0.9)", xlabel="Time (h)", ylabel=pos_ylabel)
    plot!(p1, orbit_result.times / 3600, pos_x_cart / pos_scale,
        label="Cartesian x", linewidth=2)
    plot!(p1, orbit_result.times / 3600, pos_x_ks / pos_scale,
        label="KS x", linestyle=:dash, linewidth=2)
    plot!(p1, orbit_result.times / 3600, pos_x_analytical / pos_scale,
        label="Analytical x", linestyle=:dot, linewidth=2)
    plot!(p1, orbit_result.times / 3600, pos_y_cart / pos_scale,
        label="Cartesian y", linewidth=2)
    plot!(p1, orbit_result.times / 3600, pos_y_ks / pos_scale,
        label="KS y", linestyle=:dash, linewidth=2)
    plot!(p1, orbit_result.times / 3600, pos_y_analytical / pos_scale,
        label="Analytical y", linestyle=:dot, linewidth=2)

    # Position errors plot
    pos_error_ylabel = pos_error_order == 0 ? "Error (m)" : "Error (1e$(pos_error_order) m)"
    p2 = plot(title="Position Errors (e=0.9)", xlabel="Time (h)", ylabel=pos_error_ylabel)
    plot!(p2, orbit_result.times / 3600, pos_errors_cart_ks / pos_error_scale, label="Cartesian-KS", linewidth=2)
    plot!(p2, orbit_result.times / 3600, pos_errors_cart_analytical / pos_error_scale, label="Cartesian-Analytical", linewidth=2)
    plot!(p2, orbit_result.times / 3600, pos_errors_ks_analytical / pos_error_scale, label="KS-Analytical", linewidth=2)

    # Velocity errors plot
    vel_error_ylabel = vel_error_order == 0 ? "Error (m/s)" : "Error (1e$(vel_error_order) m/s)"
    p3 = plot(title="Velocity Errors (e=0.9)", xlabel="Time (h)", ylabel=vel_error_ylabel)
    plot!(p3, orbit_result.times / 3600, vel_errors_cart_ks / vel_error_scale, label="Cartesian-KS", linewidth=2)
    plot!(p3, orbit_result.times / 3600, vel_errors_cart_analytical / vel_error_scale, label="Cartesian-Analytical", linewidth=2)
    plot!(p3, orbit_result.times / 3600, vel_errors_ks_analytical / vel_error_scale, label="KS-Analytical", linewidth=2)

    p_combined = plot(p1, p2, p3, layout=(3, 1), size=(800, 1200))
    savefig(p_combined, "figs/test_keplerian_dynamics_match.png")
    println("  Saved plot to figs/test_keplerian_dynamics_match.png")
end

