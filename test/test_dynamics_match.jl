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
    osc_elements = [a, e, i, RAAN, omega, M]
    x0_cart = SD.sOSCtoCART(osc_elements; GM=GM, use_degrees=false)
    r0 = x0_cart[1:3]
    v0 = x0_cart[4:6]

    println("\nInitial conditions:")
    println("  Semi-major axis: ", a / 1e3, " km")
    println("  Eccentricity: ", e)
    println("  Inclination: ", i * 180 / π, " deg")
    println("  Argument of periapsis: ", omega * 180 / π, " deg")
    println("  RAAN: ", RAAN * 180 / π, " deg")
    println("  Mean anomaly: ", M * 180 / π, " deg")
    println("  Periapsis distance: ", a * (1 - e) / 1e3, " km")
    println("  Apoapsis distance: ", a * (1 + e) / 1e3, " km")
    println("  Position: ", r0)
    println("  Velocity: ", v0)
    println("  Radius: ", norm(r0) / 1e3, " km")
    println("  Speed: ", norm(v0) / 1e3, " km/s")

    # Convert to KS coordinates
    p_state_0_ks = state_inertial_to_ks(x0_cart)
    h0 = ks_h_energy(p_state_0_ks[1:4], p_state_0_ks[5:8], GM)
    p_state_0_ks_full = [p_state_0_ks; h0]

    # Verify conversion
    x0_ks_back = state_ks_to_inertial(p_state_0_ks)
    conv_pos_error = norm(x0_ks_back[1:3] - r0)
    conv_vel_error = norm(x0_ks_back[4:6] - v0)
    println("\nKS conversion verification:")
    println("  Position error: ", conv_pos_error, " m")
    println("  Velocity error: ", conv_vel_error, " m/s")

    # Propagation time: multiple orbital periods
    T_orbital = 2π * sqrt(a^3 / GM)
    t0 = 0.0
    t_end = SIM_PARAMS.num_orbits * T_orbital
    dt = SIM_PARAMS.sampling_time
    times = collect(t0:dt:t_end)

    println("\nPropagation parameters:")
    println("  Orbital period: ", T_orbital, " s (", T_orbital / 3600, " hours)")
    println("  Time step: ", dt, " s")
    println("  Number of steps: ", length(times))

    # Propagate in Cartesian coordinates
    println("\nPropagating in Cartesian coordinates...")
    function cart_dynamics!(xdot, x, p, t)
        cartesian_gravity_dynamics!(xdot, x, p, t)
    end

    prob_cart = ODEProblem(cart_dynamics!, x0_cart, (t0, t_end), GM)
    sol_cart = solve(prob_cart, SIM_PARAMS.integrator();
        abstol=SIM_PARAMS.abstol,
        reltol=SIM_PARAMS.reltol,
        saveat=times)

    x_traj_cart = [sol_cart.u[k] for k = 1:length(sol_cart.u)]
    println("  Completed: ", length(x_traj_cart), " states")

    # Propagate in KS coordinates
    println("Propagating in KS coordinates...")
    x_traj_ks, p_state_traj_ks = propagate_ks_with_time(p_state_0_ks_full, t0, times, GM)
    println("  Completed: ", length(x_traj_ks), " states")

    # Compare trajectories
    max_pos_error = 0.0
    max_vel_error = 0.0
    max_pos_error_idx = 1
    max_vel_error_idx = 1

    for i in 1:min(length(x_traj_cart), length(x_traj_ks))
        x_cart = x_traj_cart[i]
        x_ks = x_traj_ks[i]

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

    # Check final state
    x_final_cart = x_traj_cart[end]
    x_final_ks = x_traj_ks[end]
    final_pos_error = norm(x_final_cart[1:3] - x_final_ks[1:3])
    final_vel_error = norm(x_final_cart[4:6] - x_final_ks[4:6])

    println("\nFinal state comparison:")
    println("  Position error: ", final_pos_error, " m")
    println("  Velocity error: ", final_vel_error, " m/s")

    # Energy conservation check
    E0_cart = 0.5 * norm(v0)^2 - GM / norm(r0)
    E_final_cart = 0.5 * norm(x_final_cart[4:6])^2 - GM / norm(x_final_cart[1:3])
    energy_change = E_final_cart - E0_cart

    println("\nEnergy conservation:")
    println("  Initial energy: ", E0_cart, " J/kg")
    println("  Final energy: ", E_final_cart, " J/kg")
    println("  Energy change: ", energy_change, " J/kg")

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
        final_pos_error=final_pos_error, final_vel_error=final_vel_error,
        energy_change=energy_change, x_traj_cart=x_traj_cart, x_traj_ks=x_traj_ks, times=times)
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
    plot!(p1, orbit_result.times, [x[1] for x in orbit_result.x_traj_cart],
        label="Cartesian x", linewidth=2)
    plot!(p1, orbit_result.times, [x[1] for x in orbit_result.x_traj_ks],
        label="KS x", linestyle=:dash, linewidth=2)
    plot!(p1, orbit_result.times, [x[2] for x in orbit_result.x_traj_cart],
        label="Cartesian y", linewidth=2)
    plot!(p1, orbit_result.times, [x[2] for x in orbit_result.x_traj_ks],
        label="KS y", linestyle=:dash, linewidth=2)

    p2 = plot(title="Position Error (e=0.9)", xlabel="Time (s)", ylabel="Error (m)")
    pos_errors = [norm(orbit_result.x_traj_cart[i][1:3] - orbit_result.x_traj_ks[i][1:3])
                  for i = 1:length(orbit_result.times)]
    plot!(p2, orbit_result.times, pos_errors, label="Position error", linewidth=2)

    p3 = plot(title="Velocity Error (e=0.9)", xlabel="Time (s)", ylabel="Error (m/s)")
    vel_errors = [norm(orbit_result.x_traj_cart[i][4:6] - orbit_result.x_traj_ks[i][4:6])
                  for i = 1:length(orbit_result.times)]
    plot!(p3, orbit_result.times, vel_errors, label="Velocity error", linewidth=2)

    p_combined = plot(p1, p2, p3, layout=(3, 1), size=(800, 1200))
    savefig(p_combined, "figs/dynamics_comparison.png")
    println("  Saved plot to figs/dynamics_comparison.png")
end

