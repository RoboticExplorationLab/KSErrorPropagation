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

# Initial conditions: circular orbit at 7000 km altitude
R_EARTH = SD.R_EARTH
altitude = 7000e3  # meters
r_mag = R_EARTH + altitude
v_mag = sqrt(GM / r_mag)  # circular orbit velocity

# Initial state in Cartesian: [r; v]
r0 = [r_mag, 0.0, 0.0]
v0 = [0.0, v_mag, 0.0]
x0_cart = [r0; v0]

println("Initial Cartesian state:")
println("  Position: ", r0)
println("  Velocity: ", v0)
println("  Radius: ", norm(r0), " m")
println("  Speed: ", norm(v0), " m/s")

# Convert to KS coordinates
p_state_0_ks = state_inertial_to_ks(x0_cart)
h0 = ks_h_energy(p_state_0_ks[1:4], p_state_0_ks[5:8], GM)
p_state_0_ks_full = [p_state_0_ks; h0]

println("\nInitial KS state:")
println("  p: ", p_state_0_ks[1:4])
println("  p_prime: ", p_state_0_ks[5:8])
println("  h: ", h0)

# Verify conversion
x0_ks_back = state_ks_to_inertial(p_state_0_ks)
println("\nVerification - KS to Cartesian conversion:")
println("  Position error: ", norm(x0_ks_back[1:3] - r0), " m")
println("  Velocity error: ", norm(x0_ks_back[4:6] - v0), " m/s")

# Propagation time: one orbital period
T_orbital = 2π * sqrt(r_mag^3 / GM)
t0 = 0.0
t_end = T_orbital
dt = T_orbital / 100
times = collect(t0:dt:t_end)

println("\nPropagation parameters:")
println("  Orbital period: ", T_orbital, " s (", T_orbital/3600, " hours)")
println("  Time step: ", dt, " s")
println("  Number of steps: ", length(times))

# Propagate in Cartesian coordinates
println("\nPropagating in Cartesian coordinates...")
function cart_dynamics!(xdot, x, p, t)
    cartesian_gravity_dynamics!(xdot, x, p, t)
end

prob_cart = ODEProblem(cart_dynamics!, x0_cart, (t0, t_end), GM)
sol_cart = solve(prob_cart, Tsit5(); abstol=1e-12, reltol=1e-13, saveat=times)

x_traj_cart = [sol_cart.u[k] for k = 1:length(sol_cart.u)]

println("  Completed: ", length(x_traj_cart), " states")

# Propagate in KS coordinates
println("\nPropagating in KS coordinates...")
x_traj_ks, p_state_traj_ks = propagate_ks_with_time(p_state_0_ks_full, t0, times, GM)

println("  Completed: ", length(x_traj_ks), " states")

# Compare trajectories
println("\n" * "="^60)
println("COMPARISON RESULTS")
println("="^60)

max_pos_error = 0.0
max_vel_error = 0.0
max_pos_error_idx = 1
max_vel_error_idx = 1

for i in 1:min(length(x_traj_cart), length(x_traj_ks))
    x_cart = x_traj_cart[i]
    x_ks = x_traj_ks[i]
    
    pos_error = norm(x_cart[1:3] - x_ks[1:3])
    vel_error = norm(x_cart[4:6] - x_ks[4:6])
    
    global max_pos_error, max_pos_error_idx, max_vel_error, max_vel_error_idx
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
println("\nFinal state comparison:")
x_final_cart = x_traj_cart[end]
x_final_ks = x_traj_ks[end]
println("  Cartesian final position: ", x_final_cart[1:3])
println("  KS final position:       ", x_final_ks[1:3])
println("  Position difference:     ", x_final_cart[1:3] - x_final_ks[1:3])
println("  Position error norm:     ", norm(x_final_cart[1:3] - x_final_ks[1:3]), " m")

println("\n  Cartesian final velocity: ", x_final_cart[4:6])
println("  KS final velocity:       ", x_final_ks[4:6])
println("  Velocity difference:     ", x_final_cart[4:6] - x_final_ks[4:6])
println("  Velocity error norm:     ", norm(x_final_cart[4:6] - x_final_ks[4:6]), " m/s")

# Energy conservation check
println("\nEnergy conservation:")
E0_cart = 0.5 * norm(v0)^2 - GM / norm(r0)
E_final_cart = 0.5 * norm(x_final_cart[4:6])^2 - GM / norm(x_final_cart[1:3])
println("  Initial energy (Cartesian):   ", E0_cart, " J/kg")
println("  Final energy (Cartesian):     ", E_final_cart, " J/kg")
println("  Energy change (Cartesian):     ", E_final_cart - E0_cart, " J/kg")

# Plot comparison
println("\nGenerating plots...")
p1 = plot(title="Position Comparison", xlabel="Time (s)", ylabel="Position (m)")
plot!(p1, times, [x[1] for x in x_traj_cart], label="Cartesian x", linewidth=2)
plot!(p1, times, [x[1] for x in x_traj_ks], label="KS x", linestyle=:dash, linewidth=2)
plot!(p1, times, [x[2] for x in x_traj_cart], label="Cartesian y", linewidth=2)
plot!(p1, times, [x[2] for x in x_traj_ks], label="KS y", linestyle=:dash, linewidth=2)

p2 = plot(title="Position Error", xlabel="Time (s)", ylabel="Error (m)")
pos_errors = [norm(x_traj_cart[i][1:3] - x_traj_ks[i][1:3]) for i = 1:length(times)]
plot!(p2, times, pos_errors, label="Position error", linewidth=2)

p3 = plot(title="Velocity Error", xlabel="Time (s)", ylabel="Error (m/s)")
vel_errors = [norm(x_traj_cart[i][4:6] - x_traj_ks[i][4:6]) for i = 1:length(times)]
plot!(p3, times, vel_errors, label="Velocity error", linewidth=2)

p_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 1200))
savefig(p_combined, "figs/dynamics_comparison.png")
println("  Saved plot to figs/dynamics_comparison.png")

# Summary
println("\n" * "="^60)
if max_pos_error < 1.0 && max_vel_error < 0.01
    println("✓ SUCCESS: Dynamics match within tolerance!")
    println("  Position error < 1 m")
    println("  Velocity error < 0.01 m/s")
else
    println("⚠ WARNING: Dynamics may not match perfectly")
    println("  Check the errors above")
end
println("="^60)

