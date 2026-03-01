using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots
using LinearAlgebra
using SatelliteDynamics
using DifferentialEquations

const SD = SatelliteDynamics

include("../src/cartesian_dynamics.jl")
include("../src/ks_dynamics.jl")
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

# Chief orbit parameters
sma_c = 750e3 + SD.R_EARTH  # semi-major axis
e_c = 0.05  # eccentricity
i_c = deg2rad(98.2)  # inclination
ω_c = deg2rad(0.0)  # argument of periapsis
Ω_c = deg2rad(0.0)  # right angle of the ascending node
M_c = deg2rad(0.0)  # Mean Anomaly

# Deputy orbit parameters (offset by 0.001 degrees in mean anomaly and inclination)
sma_d = sma_c  # same semi-major axis
e_d = e_c
i_d = i_c + deg2rad(0.001)  # offset by 0.001 degrees in inclination
ω_d = ω_c  # same argument of periapsis
Ω_d = Ω_c  # same RAAN
M_d = M_c + deg2rad(0.001)  # offset by 0.001 degrees in mean anomaly

println("="^80)
println("KS ERROR PROPAGATION - PERTURBED RELATIVE DYNAMICS MATCHING TEST")
println("="^80)
println("Testing KS perturbed relative dynamics against full nonlinear propagation")

println("\nChief orbit:")
println("  Semi-major axis: ", sma_c / 1e3, " km")
println("  Eccentricity: ", e_c)
println("  Inclination: ", i_c * 180 / π, " deg")
println("  Argument of periapsis: ", ω_c * 180 / π, " deg")
println("  RAAN: ", Ω_c * 180 / π, " deg")
println("  Mean anomaly: ", M_c * 180 / π, " deg")

println("\nDeputy orbit:")
println("  Semi-major axis: ", sma_d / 1e3, " km")
println("  Eccentricity: ", e_d)
println("  Inclination: ", i_d * 180 / π, " deg (offset: ", (i_d - i_c) * 180 / π, " deg)")
println("  Argument of periapsis: ", ω_d * 180 / π, " deg")
println("  RAAN: ", Ω_d * 180 / π, " deg")
println("  Mean anomaly: ", M_d * 180 / π, " deg (offset: ", (M_d - M_c) * 180 / π, " deg)")

# Convert orbital elements to Cartesian states
oe_vec_chief = [sma_c, e_c, i_c, Ω_c, ω_c, M_c]
x_vec_chief_0 = SD.sOSCtoCART(oe_vec_chief; GM=SIM_PARAMS.GM, use_degrees=false)
r_vec_chief_0 = x_vec_chief_0[1:3]
v_vec_chief_0 = x_vec_chief_0[4:6]

oe_vec_deputy = [sma_d, e_d, i_d, Ω_d, ω_d, M_d]
x_vec_deputy_0 = SD.sOSCtoCART(oe_vec_deputy; GM=SIM_PARAMS.GM, use_degrees=false)
r_vec_deputy_0 = x_vec_deputy_0[1:3]
v_vec_deputy_0 = x_vec_deputy_0[4:6]

# Initial relative state
x_vec_rel_0 = x_vec_deputy_0 .- x_vec_chief_0
r_vec_rel_0 = x_vec_rel_0[1:3]
v_vec_rel_0 = x_vec_rel_0[4:6]

println("\nInitial conditions:")
println("  Chief position: ", r_vec_chief_0)
println("  Chief velocity: ", v_vec_chief_0)
println("  Chief radius: ", norm(r_vec_chief_0) / 1e3, " km")
println("  Chief speed: ", norm(v_vec_chief_0) / 1e3, " km/s")
println("  Deputy position: ", r_vec_deputy_0)
println("  Deputy velocity: ", v_vec_deputy_0)
println("  Deputy radius: ", norm(r_vec_deputy_0) / 1e3, " km")
println("  Deputy speed: ", norm(v_vec_deputy_0) / 1e3, " km/s")
println("  Relative position: ", r_vec_rel_0)
println("  Relative velocity: ", v_vec_rel_0)
println("  Relative distance: ", norm(r_vec_rel_0), " m")

# Propagation time: multiple orbital periods
T_orbital = 2π * sqrt(sma_c^3 / SIM_PARAMS.GM)
t_0 = 0.0
t_end = SIM_PARAMS.num_orbits * T_orbital
dt = SIM_PARAMS.sampling_time
times = collect(t_0:dt:t_end)

println("\nPropagation parameters:")
println("  Orbital period: ", T_orbital, " s (", T_orbital / 3600, " hours)")
println("  Time step: ", dt, " s")
println("  Number of steps: ", length(times))
println("  Total propagation time: ", t_end / 3600, " hours")

# Scaling factors
t_scale = something(SIM_PARAMS.t_scale, T_orbital)
r_scale = something(SIM_PARAMS.r_scale, sma_c)
v_scale = something(SIM_PARAMS.v_scale, r_scale / t_scale)
a_scale = something(SIM_PARAMS.a_scale, v_scale / t_scale)
GM_scale = something(SIM_PARAMS.GM_scale, r_scale^3 / t_scale^2)
sim_params = merge(SIM_PARAMS, (t_scale=t_scale, r_scale=r_scale, v_scale=v_scale, a_scale=a_scale, GM_scale=GM_scale))

# Full nonlinear propagation: propagate chief and deputy separately with perturbations
println("\nPropagating chief in Cartesian coordinates...")
x_vec_traj_chief_cart, t_traj_chief_cart = propagate_cartesian_dynamics(x_vec_chief_0, times, sim_params)
println("  Completed: ", length(x_vec_traj_chief_cart), " states")

println("Propagating deputy in Cartesian coordinates...")
x_vec_traj_deputy_cart, t_traj_deputy_cart = propagate_cartesian_dynamics(x_vec_deputy_0, times, sim_params)
println("  Completed: ", length(x_vec_traj_deputy_cart), " states")

# Compute relative state from full nonlinear propagation
x_vec_traj_rel_cart = [x_vec_traj_deputy_cart[k] .- x_vec_traj_chief_cart[k] for k = 1:length(times)]
println("  Computed relative states from full nonlinear propagation")

# KS perturbed relative dynamics propagation
println("Propagating using KS perturbed relative dynamics...")
x_vec_traj_chief_ks, x_vec_traj_rel_ks, t_chief_traj, t_deputy_traj = propagate_ks_relative_dynamics(x_vec_chief_0, x_vec_deputy_0, times, sim_params)
println("  Completed: ", length(x_vec_traj_chief_ks), " chief states")
println("  Completed: ", length(x_vec_traj_rel_ks), " relative states")

# Compare time histories
println("\nTime comparison:")
println("  Chief Cartesian times: ", length(t_traj_chief_cart), " points")
println("  Chief KS times: ", length(t_chief_traj), " points")
println("  Deputy Cartesian times: ", length(t_traj_deputy_cart), " points")
println("  Deputy KS times: ", length(t_deputy_traj), " points")

# Check which states were actually saved (not zero time means state was saved)
valid_chief_indices = [i for i in 1:length(t_chief_traj) if abs(t_chief_traj[i] - times[i]) < 1.0]
valid_deputy_indices = [i for i in 1:length(t_deputy_traj) if abs(t_deputy_traj[i] - times[i]) < 1.0]
println("  Valid chief states: ", length(valid_chief_indices), " / ", length(times))
println("  Valid deputy states: ", length(valid_deputy_indices), " / ", length(times))

# Compare trajectories: Chief
max_pos_error_chief = 0.0
max_vel_error_chief = 0.0
max_pos_error_idx_chief = 1
max_vel_error_idx_chief = 1

# Compare trajectories: Relative state
max_pos_error_rel = 0.0
max_vel_error_rel = 0.0
max_pos_error_idx_rel = 1
max_vel_error_idx_rel = 1

N = min(length(x_vec_traj_chief_cart), length(x_vec_traj_chief_ks),
    length(x_vec_traj_rel_cart), length(x_vec_traj_rel_ks))

for i in 1:N
    # Skip if state wasn't saved (time doesn't match)
    if !(i in valid_chief_indices) || !(i in valid_deputy_indices)
        continue
    end
    # Chief comparison
    x_chief_cart = x_vec_traj_chief_cart[i]
    x_chief_ks = x_vec_traj_chief_ks[i]

    pos_error_chief = norm(x_chief_cart[1:3] - x_chief_ks[1:3])
    vel_error_chief = norm(x_chief_cart[4:6] - x_chief_ks[4:6])

    if pos_error_chief > max_pos_error_chief
        global max_pos_error_chief = pos_error_chief
        global max_pos_error_idx_chief = i
    end
    if vel_error_chief > max_vel_error_chief
        global max_vel_error_chief = vel_error_chief
        global max_vel_error_idx_chief = i
    end

    # Relative state comparison
    x_rel_cart = x_vec_traj_rel_cart[i]
    x_rel_ks = x_vec_traj_rel_ks[i]

    pos_error_rel = norm(x_rel_cart[1:3] - x_rel_ks[1:3])
    vel_error_rel = norm(x_rel_cart[4:6] - x_rel_ks[4:6])

    if pos_error_rel > max_pos_error_rel
        global max_pos_error_rel = pos_error_rel
        global max_pos_error_idx_rel = i
    end
    if vel_error_rel > max_vel_error_rel
        global max_vel_error_rel = vel_error_rel
        global max_vel_error_idx_rel = i
    end
end

# RMS error functions
function relative_state_rms_position_error(x_rel_traj_true, x_rel_traj_comp)
    Nk = length(x_rel_traj_true)
    accum = 0.0
    for k = 1:Nk
        diff = x_rel_traj_comp[k][1:3] .- x_rel_traj_true[k][1:3]
        accum += diff'diff
    end
    return sqrt(accum / Nk)
end

function relative_state_rms_velocity_error(x_rel_traj_true, x_rel_traj_comp)
    Nk = length(x_rel_traj_true)
    accum = 0.0
    for k = 1:Nk
        diff = x_rel_traj_comp[k][4:6] .- x_rel_traj_true[k][4:6]
        accum += diff'diff
    end
    return sqrt(accum / Nk)
end

println("\nMaximum errors (Chief: Cartesian vs KS):")
println("  Position error: ", max_pos_error_chief, " m (at step ", max_pos_error_idx_chief, ")")
println("  Velocity error: ", max_vel_error_chief, " m/s (at step ", max_vel_error_idx_chief, ")")

println("\nMaximum errors (Relative state: Full nonlinear vs KS perturbed relative dynamics):")
println("  Position error: ", max_pos_error_rel, " m (at step ", max_pos_error_idx_rel, ")")
println("  Velocity error: ", max_vel_error_rel, " m/s (at step ", max_vel_error_idx_rel, ")")

# Compute RMS errors for relative state
valid_rel_traj_cart = [x_vec_traj_rel_cart[i] for i in 1:N if (i in valid_chief_indices) && (i in valid_deputy_indices)]
valid_rel_traj_ks = [x_vec_traj_rel_ks[i] for i in 1:N if (i in valid_chief_indices) && (i in valid_deputy_indices)]

if length(valid_rel_traj_cart) > 0 && length(valid_rel_traj_ks) > 0
    min_len = min(length(valid_rel_traj_cart), length(valid_rel_traj_ks))
    rms_pos_error_rel = relative_state_rms_position_error(valid_rel_traj_cart[1:min_len], valid_rel_traj_ks[1:min_len])
    rms_vel_error_rel = relative_state_rms_velocity_error(valid_rel_traj_cart[1:min_len], valid_rel_traj_ks[1:min_len])

    println("\nRMS errors (Relative state: Full nonlinear vs KS perturbed relative dynamics):")
    println("  RMS position error: ", rms_pos_error_rel, " m")
    println("  RMS velocity error: ", rms_vel_error_rel, " m/s")
end

# Summary
success_chief = max_pos_error_chief < SIM_PARAMS.max_pos_error_threshold &&
                max_vel_error_chief < SIM_PARAMS.max_vel_error_threshold
success_rel = max_pos_error_rel < SIM_PARAMS.max_pos_error_threshold &&
              max_vel_error_rel < SIM_PARAMS.max_vel_error_threshold

println("\n" * "-"^80)
if success_chief
    println("✓ SUCCESS: Chief propagation matches within tolerance!")
else
    println("⚠ WARNING: Chief propagation may not match perfectly")
end
if success_rel
    println("✓ SUCCESS: Relative dynamics match within tolerance!")
else
    println("⚠ WARNING: Relative dynamics may not match perfectly")
end
println("-"^80)

# Helper function to determine order of magnitude for normalization
function get_order_of_magnitude(values)
    max_abs = maximum(abs.(values))
    if max_abs == 0.0 || !isfinite(max_abs)
        return 0
    end
    return Int(floor(log10(max_abs)))
end

# Function to generate plots for relative dynamics comparison
function generate_relative_dynamics_plot(times, N,
    x_vec_traj_chief_cart, x_vec_traj_chief_ks,
    x_vec_traj_rel_cart, x_vec_traj_rel_ks,
    filename)
    # Chief position data
    pos_x_chief_cart = [x[1] for x in x_vec_traj_chief_cart[1:N]]
    pos_x_chief_ks = [x[1] for x in x_vec_traj_chief_ks[1:N]]
    pos_y_chief_cart = [x[2] for x in x_vec_traj_chief_cart[1:N]]
    pos_y_chief_ks = [x[2] for x in x_vec_traj_chief_ks[1:N]]

    all_chief_pos_values = vcat(pos_x_chief_cart, pos_x_chief_ks, pos_y_chief_cart, pos_y_chief_ks)
    chief_pos_order = get_order_of_magnitude(all_chief_pos_values)
    chief_pos_scale = 10.0^chief_pos_order

    # Relative position data
    pos_x_rel_cart = [x[1] for x in x_vec_traj_rel_cart[1:N]]
    pos_x_rel_ks = [x[1] for x in x_vec_traj_rel_ks[1:N]]
    pos_y_rel_cart = [x[2] for x in x_vec_traj_rel_cart[1:N]]
    pos_y_rel_ks = [x[2] for x in x_vec_traj_rel_ks[1:N]]

    all_rel_pos_values = vcat(pos_x_rel_cart, pos_x_rel_ks, pos_y_rel_cart, pos_y_rel_ks)
    rel_pos_order = get_order_of_magnitude(all_rel_pos_values)
    rel_pos_scale = 10.0^rel_pos_order

    # Chief position errors
    pos_errors_chief = [norm(x_vec_traj_chief_cart[i][1:3] - x_vec_traj_chief_ks[i][1:3])
                        for i = 1:N]
    chief_pos_error_order = get_order_of_magnitude(pos_errors_chief)
    chief_pos_error_scale = 10.0^chief_pos_error_order

    # Relative position errors
    pos_errors_rel = [norm(x_vec_traj_rel_cart[i][1:3] - x_vec_traj_rel_ks[i][1:3])
                      for i = 1:N]
    rel_pos_error_order = get_order_of_magnitude(pos_errors_rel)
    rel_pos_error_scale = 10.0^rel_pos_error_order

    # Relative velocity errors
    vel_errors_rel = [norm(x_vec_traj_rel_cart[i][4:6] - x_vec_traj_rel_ks[i][4:6])
                      for i = 1:N]
    rel_vel_error_order = get_order_of_magnitude(vel_errors_rel)
    rel_vel_error_scale = 10.0^rel_vel_error_order

    # Chief position comparison plot
    chief_pos_ylabel = chief_pos_order == 0 ? "Position (m)" : "Position (1e$(chief_pos_order) m)"
    p1 = plot(title="Chief Position Comparison (Perturbed)", xlabel="Time (h)", ylabel=chief_pos_ylabel)
    plot!(p1, times[1:N] / 3600, pos_x_chief_cart / chief_pos_scale,
        label="Cartesian x", linewidth=2)
    plot!(p1, times[1:N] / 3600, pos_x_chief_ks / chief_pos_scale,
        label="KS x", linestyle=:dash, linewidth=2)
    plot!(p1, times[1:N] / 3600, pos_y_chief_cart / chief_pos_scale,
        label="Cartesian y", linewidth=2)
    plot!(p1, times[1:N] / 3600, pos_y_chief_ks / chief_pos_scale,
        label="KS y", linestyle=:dash, linewidth=2)

    # Chief position errors plot
    chief_pos_error_ylabel = chief_pos_error_order == 0 ? "Error (m)" : "Error (1e$(chief_pos_error_order) m)"
    p2 = plot(title="Chief Position Errors (Perturbed)", xlabel="Time (h)", ylabel=chief_pos_error_ylabel)
    plot!(p2, times[1:N] / 3600, pos_errors_chief / chief_pos_error_scale,
        label="Cartesian-KS", linewidth=2, color=:red)

    # Relative position comparison plot
    rel_pos_ylabel = rel_pos_order == 0 ? "Position (m)" : "Position (1e$(rel_pos_order) m)"
    p3 = plot(title="Relative Position Comparison (Perturbed)", xlabel="Time (h)", ylabel=rel_pos_ylabel)
    plot!(p3, times[1:N] / 3600, pos_x_rel_cart / rel_pos_scale,
        label="Full nonlinear x", linewidth=2)
    plot!(p3, times[1:N] / 3600, pos_x_rel_ks / rel_pos_scale,
        label="KS relative x", linestyle=:dash, linewidth=2)
    plot!(p3, times[1:N] / 3600, pos_y_rel_cart / rel_pos_scale,
        label="Full nonlinear y", linewidth=2)
    plot!(p3, times[1:N] / 3600, pos_y_rel_ks / rel_pos_scale,
        label="KS relative y", linestyle=:dash, linewidth=2)

    # Relative position errors plot
    rel_pos_error_ylabel = rel_pos_error_order == 0 ? "Error (m)" : "Error (1e$(rel_pos_error_order) m)"
    p4 = plot(title="Relative Position Errors (Perturbed)", xlabel="Time (h)", ylabel=rel_pos_error_ylabel)
    plot!(p4, times[1:N] / 3600, pos_errors_rel / rel_pos_error_scale,
        label="Full nonlinear - KS relative", linewidth=2, color=:red)

    # Relative velocity errors plot
    rel_vel_error_ylabel = rel_vel_error_order == 0 ? "Error (m/s)" : "Error (1e$(rel_vel_error_order) m/s)"
    p5 = plot(title="Relative Velocity Errors (Perturbed)", xlabel="Time (h)", ylabel=rel_vel_error_ylabel)
    plot!(p5, times[1:N] / 3600, vel_errors_rel / rel_vel_error_scale,
        label="Full nonlinear - KS relative", linewidth=2, color=:red)

    p_combined = plot(p1, p2, p3, p4, p5, layout=(5, 1), size=(800, 2000))
    savefig(p_combined, filename)
    println("  Saved plot to $filename")
end

# Generate plots
println("\nGenerating plots...")
generate_relative_dynamics_plot(times, N,
    x_vec_traj_chief_cart, x_vec_traj_chief_ks,
    x_vec_traj_rel_cart, x_vec_traj_rel_ks,
    "figs/test_relative_dynamics_match.png")

println("\n" * "="^80)
if success_chief && success_rel
    println("✓ ALL TESTS PASSED!")
else
    println("⚠ SOME TESTS FAILED - Check results above")
end
println("="^80)

