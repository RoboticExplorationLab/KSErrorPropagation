using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots
using LinearAlgebra
using SatelliteDynamics
using DifferentialEquations

const SD = SatelliteDynamics

include("../src/cartesian_dynamics.jl")
include("../src/ks_dynamics.jl")
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
        a=750e3 + R_EARTH,
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

# Function to test a single orbit
function test_orbit(orbit_name, a, e, i, omega, RAAN, M)
    println("\n" * "="^80)
    println("TESTING: ", orbit_name)
    println("="^80)

    # Convert orbital elements to cartesian using SatelliteDynamics
    oe_vec = [a, e, i, RAAN, omega, M]
    x_vec_0 = SD.sOSCtoCART(oe_vec; GM=SIM_PARAMS.GM, use_degrees=false)
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
    h_0 = energy_ks(ks_state_0[1:4], ks_state_0[5:8], SIM_PARAMS.GM)
    ks_state_augmented_0 = [ks_state_0; h_0]

    # Verify conversion
    x_vec_0_ks_back = state_ks_to_cartesian(ks_state_augmented_0[1:8])
    conv_pos_error = norm(x_vec_0_ks_back[1:3] - r_vec_0)
    conv_vel_error = norm(x_vec_0_ks_back[4:6] - v_vec_0)
    println("\nKS conversion verification:")
    println("  Position error: ", conv_pos_error, " m")
    println("  Velocity error: ", conv_vel_error, " m/s")

    # Propagation time: multiple orbital periods
    T_orbital = 2π * sqrt(a^3 / SIM_PARAMS.GM)
    t_0 = 0.0
    t_end = SIM_PARAMS.num_orbits * T_orbital
    dt = SIM_PARAMS.sampling_time
    times = collect(t_0:dt:t_end)

    println("\nPropagation parameters:")
    println("  Orbital period: ", T_orbital, " s (", T_orbital / 3600, " hours)")
    println("  Time step: ", dt, " s")
    println("  Number of steps: ", length(times))

    # Scaling factors
    t_scale = something(SIM_PARAMS.t_scale, T_orbital)
    r_scale = something(SIM_PARAMS.r_scale, a)
    v_scale = something(SIM_PARAMS.v_scale, r_scale / t_scale)
    a_scale = something(SIM_PARAMS.a_scale, v_scale / t_scale)
    GM_scale = something(SIM_PARAMS.GM_scale, r_scale^3 / t_scale^2)
    sim_params = merge(SIM_PARAMS, (t_scale=t_scale, r_scale=r_scale, v_scale=v_scale, a_scale=a_scale, GM_scale=GM_scale))

    # Propagate in Cartesian coordinates
    println("\nPropagating in Cartesian coordinates...")
    x_vec_traj_cart, t_traj_cart = propagate_cartesian_dynamics(x_vec_0, times, sim_params)
    println("  Completed: ", length(x_vec_traj_cart), " states")

    # Propagate in KS coordinates
    println("Propagating in KS coordinates...")
    x_vec_traj_ks, t_traj_ks = propagate_ks_dynamics(x_vec_0, times, sim_params)
    println("  Completed: ", length(x_vec_traj_ks), " states")

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

    N = min(length(x_vec_traj_cart), length(x_vec_traj_ks))
    for i in 1:N
        x_cart = x_vec_traj_cart[i]
        x_ks = x_vec_traj_ks[i]

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
    end

    println("\nMaximum errors (Cartesian vs KS):")
    println("  Position error: ", max_pos_error_cart_ks, " m (at step ", max_pos_error_idx_cart_ks, ")")
    println("  Velocity error: ", max_vel_error_cart_ks, " m/s (at step ", max_vel_error_idx_cart_ks, ")")

    # Summary
    success_cart_ks = max_pos_error_cart_ks < SIM_PARAMS.max_pos_error_threshold &&
                      max_vel_error_cart_ks < SIM_PARAMS.max_vel_error_threshold

    println("\n" * "-"^80)
    if success_cart_ks
        println("✓ SUCCESS: Cartesian and KS dynamics match within tolerance!")
    else
        println("⚠ WARNING: Cartesian and KS dynamics may not match perfectly")
    end
    println("-"^80)

    return (success_cart_ks=success_cart_ks,
        max_pos_error_cart_ks=max_pos_error_cart_ks,
        max_vel_error_cart_ks=max_vel_error_cart_ks,
        x_vec_traj_cart=x_vec_traj_cart,
        x_vec_traj_ks=x_vec_traj_ks,
        times=times)
end

# Run tests for all orbits
println("="^80)
println("CARTESIAN AND KS DYNAMICS MATCHING TEST")
println("="^80)
println("Testing multiple orbits to verify cartesian and KS dynamics match")

results = []
for orbit in test_orbits
    # Calculate periapsis distance and altitude
    r_peri = orbit.a * (1 - orbit.e)
    alt_peri = (r_peri - SIM_PARAMS.R_EARTH) / 1e3  # km

    # Skip orbits with perigee below Earth's surface
    if alt_peri < 0.0
        println("\n" * "="^80)
        println("SKIPPING: ", orbit.name)
        println("="^80)
        println("  Periapsis altitude: ", alt_peri, " km (below Earth's surface)")
        println("  Periapsis distance: ", r_peri / 1e3, " km")
        println("  Earth radius: ", SIM_PARAMS.R_EARTH / 1e3, " km")
        println("  This orbit would go through the Earth - skipping test")
        println("="^80)
        continue
    end

    result = test_orbit(orbit.name, orbit.a, orbit.e, orbit.i, orbit.omega, orbit.RAAN, orbit.M)
    push!(results, (orbit=orbit, result=result))
end

# Overall summary
println("\n" * "="^80)
println("OVERALL SUMMARY")
println("="^80)

all_cart_ks_passed = true

for (i, (orbit, result)) in enumerate(results)
    status_cart_ks = result.success_cart_ks ? "✓ PASS" : "✗ FAIL"

    println("\n$(i). $(orbit.name)")
    println("   Cartesian vs KS: $(status_cart_ks)")
    println("      Max position error: $(result.max_pos_error_cart_ks) m")
    println("      Max velocity error: $(result.max_vel_error_cart_ks) m/s")

    if !result.success_cart_ks
        global all_cart_ks_passed = false
    end
end

println("\n" * "="^80)
if all_cart_ks_passed
    println("✓ ALL TESTS PASSED!")
else
    println("⚠ SOME TESTS FAILED - Check results above")
end
println("="^80)

# Helper function to determine order of magnitude for normalization
function get_order_of_magnitude(values)
    max_abs = maximum(abs.(values))
    if max_abs == 0.0
        return 0
    end
    return Int(floor(log10(max_abs)))
end

# Function to generate plots for a single orbit
function generate_plot(orbit_name, orbit_result, description)
    # Position data
    pos_x_cart = [x[1] for x in orbit_result.x_vec_traj_cart]
    pos_x_ks = [x[1] for x in orbit_result.x_vec_traj_ks]
    pos_y_cart = [x[2] for x in orbit_result.x_vec_traj_cart]
    pos_y_ks = [x[2] for x in orbit_result.x_vec_traj_ks]

    all_pos_values = vcat(pos_x_cart, pos_x_ks, pos_y_cart, pos_y_ks)
    pos_order = get_order_of_magnitude(all_pos_values)
    pos_scale = 10.0^pos_order

    # Position errors
    pos_errors_cart_ks = [norm(orbit_result.x_vec_traj_cart[i][1:3] - orbit_result.x_vec_traj_ks[i][1:3])
                          for i = 1:length(orbit_result.times)]

    all_pos_errors = vcat(pos_errors_cart_ks)
    pos_error_order = get_order_of_magnitude(all_pos_errors)
    pos_error_scale = 10.0^pos_error_order

    # Velocity errors
    vel_errors_cart_ks = [norm(orbit_result.x_vec_traj_cart[i][4:6] - orbit_result.x_vec_traj_ks[i][4:6])
                          for i = 1:length(orbit_result.times)]

    all_vel_errors = vcat(vel_errors_cart_ks)
    vel_error_order = get_order_of_magnitude(all_vel_errors)
    vel_error_scale = 10.0^vel_error_order

    # Position comparison plot
    pos_ylabel = pos_order == 0 ? "Position (m)" : "Position (1e$(pos_order) m)"
    p1 = plot(title="Position Comparison ($description)", xlabel="Time (h)", ylabel=pos_ylabel)
    plot!(p1, orbit_result.times / 3600, pos_x_cart / pos_scale,
        label="Cartesian x", linewidth=2)
    plot!(p1, orbit_result.times / 3600, pos_x_ks / pos_scale,
        label="KS x", linestyle=:dash, linewidth=2)
    plot!(p1, orbit_result.times / 3600, pos_y_cart / pos_scale,
        label="Cartesian y", linewidth=2)
    plot!(p1, orbit_result.times / 3600, pos_y_ks / pos_scale,
        label="KS y", linestyle=:dash, linewidth=2)

    # Position errors plot
    pos_error_ylabel = pos_error_order == 0 ? "Error (m)" : "Error (1e$(pos_error_order) m)"
    p2 = plot(title="Position Errors ($description)", xlabel="Time (h)", ylabel=pos_error_ylabel)
    plot!(p2, orbit_result.times / 3600, pos_errors_cart_ks / pos_error_scale, label="Cartesian-KS", linewidth=2)

    # Velocity errors plot
    vel_error_ylabel = vel_error_order == 0 ? "Error (m/s)" : "Error (1e$(vel_error_order) m/s)"
    p3 = plot(title="Velocity Errors ($description)", xlabel="Time (h)", ylabel=vel_error_ylabel)
    plot!(p3, orbit_result.times / 3600, vel_errors_cart_ks / vel_error_scale, label="Cartesian-KS", linewidth=2)

    p_combined = plot(p1, p2, p3, layout=(3, 1), size=(800, 1200))

    # Create filename from description
    filename = "figs/test_dynamics_match_$(description).png"
    savefig(p_combined, filename)
    println("  Saved plot to $filename")
end

# Generate plots for each orbit
println("\nGenerating plots for all orbits...")
for (i, (orbit, result)) in enumerate(results)
    println("\nGenerating plot for orbit $(i): $(orbit.name)...")
    generate_plot(orbit.name, result, orbit.description)
end

