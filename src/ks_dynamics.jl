"""
KS (Kustaanheimo-Stiefel) dynamics with gravity only (no J2, no drag).
"""

using LinearAlgebra
using DifferentialEquations

include("ks_transform.jl")

"""
    ks_gravity(ks_state_augmented, s, GM)

KS dynamics with gravity only (point mass gravity).

# Arguments
- `ks_state_augmented`: KS augmented state vector [y_vec; y_vec_prime; h]
- `s`: fictitious time (not used but required for ODE interface)
- `GM`: gravitational parameter (default: GM_EARTH)

# Returns
- `ks_state_augmented_prime`: KS augmented state vector derivative [y_vec_prime; y_vec_pprime; h_prime]
"""
function ks_gravity(ks_state_augmented, s, GM)
    y_vec = ks_state_augmented[1:4]
    y_vec_prime = ks_state_augmented[5:8]
    h = ks_state_augmented[9]

    # Unperturbed KS dynamics (gravity only)
    y_vec_pprime = (-h / 2.0) * y_vec
    h_prime = 0.0

    return [y_vec_prime; y_vec_pprime; h_prime]
end

"""
    ks_gravity!(ks_state_full_prime, ks_state_full, p, s)

In-place version for DifferentialEquations.jl.

# Arguments
- `ks_state_full_prime`: KS full state vector derivative [y_vec_prime; y_vec_pprime; h_prime; t_prime]
- `ks_state_full`: KS full state vector [y_vec; y_vec_prime; h; t]
- `p`: parameters (GM)
- `s`: fictitious time
"""
function ks_gravity!(ks_state_full_prime, ks_state_full, p, s)
    GM = p isa Number ? p : p[1]
    ks_state_augmented = ks_state_full[1:9]

    # KS dynamics
    ks_state_augmented_prime = ks_gravity(ks_state_augmented, s, GM)

    # Time derivative: dt/ds = r = y'y
    y_vec = ks_state_augmented[1:4]
    r_vec_norm = y_vec'y_vec
    t_prime = r_vec_norm

    ks_state_full_prime .= [ks_state_augmented_prime; t_prime]
end

"""
    propagate_ks_keplerian_orbit(ks_state_augmented_0, times, sim_params, GM)

Propagate KS state with time tracking, handling the transformation between
real time t and fictitious time s where dt = r * ds and r = y'y.

# Arguments
- `ks_state_augmented_0`: initial KS augmented state [y_vec; y_vec_prime; h]
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter

# Returns
- `x_vec_traj_ks`: array of Cartesian states [r_vec; v_vec] at each time
- `ks_state_augmented_traj`: array of KS augmented states [y_vec; y_vec_prime; h] at each time
- `t_traj`: array of times
"""
function propagate_ks_keplerian_orbit(ks_state_augmented_0, times, sim_params, GM)
    # KS full state vector: [y_vec; y_vec_prime; h; t]
    ks_state_full_0 = [ks_state_augmented_0; times[1]]

    # Pre-allocate trajectory storage
    ks_state_augmented_traj = [zeros(9) for k = 1:length(times)]
    ks_state_augmented_traj[1] .= ks_state_augmented_0
    t_traj = zeros(length(times))
    t_traj[1] = times[1]

    # Callback to save states at specific real time points
    function condition!(out, ks_state_full, s, sol)
        t_current = ks_state_full[10]  # Real time is the 10th element
        out .= times .- t_current
    end

    function affect!(sol, idx)
        if idx <= length(times)
            ks_state_augmented_traj[idx] .= sol.u[1:9]
            t_traj[idx] = sol.u[10]
        end
    end

    cb = VectorContinuousCallback(condition!, affect!, length(times))

    # Estimate fictitious time span
    y_vec_0 = ks_state_augmented_0[1:4]
    r_vec_norm_0 = y_vec_0'y_vec_0
    s_0 = 0.0
    s_end = (times[end] - times[1]) / r_vec_norm_0 # * 2.0  # Add margin

    prob = ODEProblem(ks_gravity!, ks_state_full_0, (s_0, s_end), GM)
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, callback=cb)

    # Convert KS states to Cartesian
    x_vec_traj_ks = [state_ks_to_cartesian(ks_state_augmented_traj[k][1:8]) for k = 1:length(times)]
    return x_vec_traj_ks, ks_state_augmented_traj, t_traj
end

"""
    propagate_ks_keplerian_orbit_scaled(ks_state_augmented_0, times, sim_params, GM, a)

Propagate KS state with time tracking, handling the transformation between
real time t and fictitious time s where dt = r * ds and r = y'y.

# Arguments
- `ks_state_augmented_0`: initial KS augmented state [y_vec; y_vec_prime; h]
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter
- `a`: semi-major axis (for normalization)

# Returns
- `x_vec_traj_ks`: array of Cartesian states [r_vec; v_vec] at each time
- `ks_state_augmented_traj`: array of KS augmented states [y_vec; y_vec_prime; h] at each time
- `t_traj`: array of times
"""
function propagate_ks_keplerian_orbit_scaled(ks_state_augmented_0, times, sim_params, GM, d_scale, t_scale)
    # Normalize via Cartesian conversion (simpler than direct KS normalization)
    x_vec_0 = state_ks_to_cartesian(ks_state_augmented_0[1:8])
    r_vec_0_scaled = x_vec_0[1:3] / d_scale
    v_vec_0_scaled = x_vec_0[4:6] * t_scale / d_scale
    x_vec_0_scaled = [r_vec_0_scaled; v_vec_0_scaled]

    # Convert normalized Cartesian to normalized KS
    ks_state_0_scaled = state_cartesian_to_ks(x_vec_0_scaled)

    # Normalize energy parameter h
    GM_scaled = GM * t_scale^2 / d_scale^3
    h_0_scaled = energy_ks(ks_state_0_scaled[1:4], ks_state_0_scaled[5:8], GM_scaled)
    ks_state_augmented_0_scaled = [ks_state_0_scaled; h_0_scaled]

    # Normalize times
    times_scaled = times / t_scale

    # KS full state vector: [y_vec; y_vec_prime; h; t]
    ks_state_full_0_scaled = [ks_state_augmented_0_scaled; times_scaled[1]]

    # Pre-allocate trajectory storage
    ks_state_augmented_traj_scaled = [zeros(9) for k = 1:length(times_scaled)]
    ks_state_augmented_traj_scaled[1] .= ks_state_augmented_0_scaled
    t_traj_scaled = zeros(length(times_scaled))
    t_traj_scaled[1] = times_scaled[1]

    # Callback to save states at specific normalized real time points
    function condition!(out, ks_state_full_scaled, s_scaled, sol)
        t_current_scaled = ks_state_full_scaled[10]  # Normalized real time is the 10th element
        out .= times_scaled .- t_current_scaled
    end

    function affect!(sol, idx)
        if idx <= length(times_scaled)
            ks_state_augmented_traj_scaled[idx] .= sol.u[1:9]
            t_traj_scaled[idx] = sol.u[10]
        end
    end

    cb = VectorContinuousCallback(condition!, affect!, length(times_scaled))

    # Estimate normalized fictitious time span
    y_vec_0_scaled = ks_state_augmented_0_scaled[1:4]
    r_vec_norm_0_scaled = y_vec_0_scaled'y_vec_0_scaled
    s_0_scaled = 0.0
    s_end_scaled = (times_scaled[end] - times_scaled[1]) / r_vec_norm_0_scaled

    prob = ODEProblem(ks_gravity!, ks_state_full_0_scaled, (s_0_scaled, s_end_scaled), GM_scaled)
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, callback=cb)

    # Denormalize KS states and convert to Cartesian
    ks_state_augmented_traj = Vector{Vector{Float64}}()
    x_vec_traj_ks = Vector{Vector{Float64}}()

    for k = 1:length(times_scaled)
        # Denormalize via Cartesian conversion
        ks_state_scaled = ks_state_augmented_traj_scaled[k][1:8]
        x_vec_scaled = state_ks_to_cartesian(ks_state_scaled)
        r_vec_scaled = x_vec_scaled[1:3]
        v_vec_scaled = x_vec_scaled[4:6]

        r_vec = r_vec_scaled * d_scale
        v_vec = v_vec_scaled * d_scale / t_scale

        # Convert back to KS to get denormalized KS state
        ks_state = state_cartesian_to_ks([r_vec; v_vec])
        # Recompute h from denormalized state (more reliable than denormalizing h_scaled)
        h = energy_ks(ks_state[1:4], ks_state[5:8], GM)
        ks_state_augmented = [ks_state; h]

        push!(ks_state_augmented_traj, ks_state_augmented)
        push!(x_vec_traj_ks, [r_vec; v_vec])
    end

    t_traj = t_traj_scaled * t_scale

    return x_vec_traj_ks, ks_state_augmented_traj, t_traj
end