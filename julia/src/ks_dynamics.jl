"""
KS (Kustaanheimo-Stiefel) dynamics with gravity only (no J2, no drag).
"""

using LinearAlgebra
using DifferentialEquations

include("ks_transform.jl")
include("cartesian_dynamics.jl")

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
    propagate_ks_keplerian_dynamics(ks_state_augmented_0, times, sim_params, GM)

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
function propagate_ks_keplerian_dynamics(ks_state_augmented_0, times, sim_params, GM)
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
    propagate_ks_keplerian_dynamics_scaled(ks_state_augmented_0, times, sim_params, GM, a)

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
function propagate_ks_keplerian_dynamics_scaled(ks_state_augmented_0, times, sim_params, GM, d_scale, t_scale)
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

"""
    ks_A_matrix(h)

Compute the A matrix for the KS equation.

# Arguments
- `h`: energy parameter

# Returns
- `A`: A matrix
"""
function ks_A_matrix(h)
    return [zeros(4, 4) I(4); -0.5 * h * I(4) zeros(4, 4)]
end

"""
    ks_stm(h, s)

Compute the STM for the KS equation.

# Arguments
- `h`: energy parameter
- `s`: fictitious time

# Returns
- `Φ`: STM
"""
function ks_stm(h, s)
    return exp(ks_A_matrix(h) * s)
end

"""
    propagate_ks_analytical_keplerian_dynamics(ks_state_augmented_0, times, sim_params, GM)

Analytical solution for KS Keplerian orbit propagation.
The KS equation y'' = -h/2 * y has a closed-form solution.

# Arguments
- `ks_state_augmented_0`: initial KS augmented state [y_vec; y_vec_prime; h] (9 elements)
- `times`: array of times to save at
- `sim_params`: simulation parameters (not used for analytical solution)
- `GM`: gravitational parameter

# Returns
- `x_vec_traj_ks`: array of Cartesian states [r_vec; v_vec] at each time
- `ks_state_augmented_traj`: array of KS augmented states [y_vec; y_vec_prime; h] at each time
- `t_traj`: array of times
"""
function propagate_ks_analytical_keplerian_dynamics(ks_state_augmented_0, times, sim_params, GM)
    # Extract initial KS state components
    y_vec_0 = ks_state_augmented_0[1:4]
    y_vec_prime_0 = ks_state_augmented_0[5:8]
    h = ks_state_augmented_0[9]

    # Pre-allocate trajectory storage
    ks_state_augmented_traj = [zeros(9) for k = 1:length(times)]
    ks_state_augmented_traj[1] .= ks_state_augmented_0
    t_traj = zeros(length(times))
    t_traj[1] = times[1]

    # Initialize augmented state: z_aug = [y; y'; t]
    z_aug = [y_vec_0; y_vec_prime_0; times[1]]
    s = 0.0
    k_out = 1  # Index for output times

    # Maximum step size in s (adaptive based on radius)
    max_Δs = Inf

    while k_out < length(times)
        t_target = times[k_out+1]

        # Extract current state
        y_vec = z_aug[1:4]
        y_vec_prime = z_aug[5:8]
        t_current = z_aug[9]

        # Compute current radius for adaptive step sizing
        r_current = y_vec'y_vec

        # Estimate step size in s to approach target time
        # dt/ds = r, so Δs ≈ (t_target - t_current) / r
        if t_target > t_current
            Δs_estimate = (t_target - t_current) / max(r_current, 1e-10)
            # Limit step size for stability (especially near periapsis)
            # Use h to compute step size limit: ω = sqrt(h/2), so limit based on h
            ω_for_step = sqrt(max(h / 2.0, 1e-20))
            Δs = min(Δs_estimate, max_Δs, 0.1 / ω_for_step)
        else
            # Shouldn't happen if times are increasing, but handle it
            break
        end

        # Use ks_stm to propagate [y; y'] forward in s
        Φ_88 = ks_stm(h, Δs)
        ks_state_new = Φ_88 * [y_vec; y_vec_prime]
        y_vec_new = ks_state_new[1:4]
        y_vec_prime_new = ks_state_new[5:8]

        # Compute time increment Δt(Δs) from current state
        # Coefficients computed from current y and y'
        y_dot_y = y_vec'y_vec
        y_dot_yp = y_vec'y_vec_prime
        yp_dot_yp = y_vec_prime'y_vec_prime

        # ω = sqrt(h/2) for time increment calculation
        ω = sqrt(h / 2.0)
        ω_sq = h / 2.0
        Φ_88 = ks_stm(h, Δs)

        # A: average of radius and scaled velocity norm
        # B: difference term
        # C: cross term between position and velocity
        A = 0.5 * (y_dot_y + yp_dot_yp / ω_sq)
        B = 0.5 * (y_dot_y - yp_dot_yp / ω_sq)
        C = y_dot_yp / ω

        # Time increment: Δt(Δs) = A*Δs + (B/(2ω))*sin(2ωΔs) - (C/(2ω))*(cos(2ωΔs) - 1)
        cos_2ωΔs = cos(2 * ω * Δs)
        sin_2ωΔs = sin(2 * ω * Δs)
        Δt = A * Δs + (B / (2 * ω)) * sin_2ωΔs - (C / (2 * ω)) * (cos_2ωΔs - 1.0)

        # Update augmented state: [y; y'; t]
        z_aug_new = [y_vec_new; y_vec_prime_new; t_current + Δt]

        # Check if we've crossed the target time
        if z_aug_new[9] >= t_target
            # Compute exact state at t_target using analytical solution
            # We need to find Δs_exact such that t(s + Δs_exact) = t_target
            # Use Newton's method with good initial guess (the Δs we just used)
            Δs_exact = Δs
            for iter = 1:10
                # Compute time at s + Δs_exact
                cos_2ωΔs_exact = cos(2 * ω * Δs_exact)
                sin_2ωΔs_exact = sin(2 * ω * Δs_exact)

                Δt_exact = A * Δs_exact + (B / (2 * ω)) * sin_2ωΔs_exact - (C / (2 * ω)) * (cos_2ωΔs_exact - 1.0)
                t_at_s_plus_ds = t_current + Δt_exact

                residual = t_at_s_plus_ds - t_target
                if abs(residual) < 1e-12
                    break
                end

                # Derivative: dt/ds = r(s + Δs_exact)
                # Compute state at s + Δs_exact using ks_stm
                Φ_88_exact = ks_stm(h, Δs_exact)
                ks_state_at_s_plus_ds = Φ_88_exact * [y_vec; y_vec_prime]
                y_vec_at_s_plus_ds = ks_state_at_s_plus_ds[1:4]
                r_at_s_plus_ds = y_vec_at_s_plus_ds'y_vec_at_s_plus_ds

                if abs(r_at_s_plus_ds) < 1e-15
                    break
                end

                Δs_exact = Δs_exact - residual / r_at_s_plus_ds
                Δs_exact = clamp(Δs_exact, 0.0, Δs * 1.1)  # Keep it reasonable
            end

            # Compute exact state at s + Δs_exact using ks_stm
            Φ_88_exact = ks_stm(h, Δs_exact)
            ks_state_exact = Φ_88_exact * [y_vec; y_vec_prime]
            y_vec_exact = ks_state_exact[1:4]
            y_vec_prime_exact = ks_state_exact[5:8]

            # Save exact state at target time
            k_out += 1
            ks_state_augmented_traj[k_out] = [y_vec_exact; y_vec_prime_exact; h]
            t_traj[k_out] = t_target

            # Update state for next iteration (use the exact state)
            z_aug = [y_vec_exact; y_vec_prime_exact; t_target]
            s += Δs_exact
        else
            # Haven't reached target yet, continue propagating
            z_aug = z_aug_new
            s += Δs
        end

        # Adaptive step size: reduce if radius is very small (near periapsis)
        # Use h to compute step size limit: ω = sqrt(h/2)
        ω_for_adaptive = sqrt(max(h / 2.0, 1e-20))
        if r_current < 1e6  # Small radius threshold
            max_Δs = 0.01 / ω_for_adaptive
        else
            max_Δs = 0.1 / ω_for_adaptive
        end
    end

    # Convert KS states to Cartesian
    x_vec_traj_ks = [state_ks_to_cartesian(ks_state_augmented_traj[k][1:8]) for k = 1:length(times)]
    return x_vec_traj_ks, ks_state_augmented_traj, t_traj
end

"""
    ks_J2_perturbation(ks_state_augmented, s, GM, R_EARTH, J2)

KS J2 perturbation.

# Arguments
- `ks_state_augmented`: KS augmented state vector [y_vec; y_vec_prime; h]
- `s`: fictitious time (not used but required for ODE interface)
- `GM`: gravitational parameter (default: GM_EARTH)
- `R_EARTH`: Earth radius
- `J2`: J2 coefficient

# Returns
- `ks_state_augmented_prime`: KS augmented state vector derivative [y_vec_prime; y_vec_pprime; h_prime]
"""
function ks_J2_perturbation(ks_state_augmented, s, GM, R_EARTH, J2)
    y_vec = ks_state_augmented[1:4]
    y_vec_prime = ks_state_augmented[5:8]
    ks_state = ks_state_augmented[1:8]

    x_vec = state_ks_to_cartesian(ks_state)
    a_J2 = cartesian_J2_perturbation(x_vec, s, GM, R_EARTH, J2)
    y_vec_pprime = (y_vec'y_vec / 2.0) * (L(y_vec)' * [a_J2; 0.0])

    h_prime = -2 * y_vec_prime' * L(y_vec)' * [a_J2; 0.0]

    return [zeros(4); y_vec_pprime; h_prime]
end

"""
    ks_drag_perturbation(ks_state_augmented, s, epoch, OMEGA_EARTH, CD, A, m)

KS drag perturbation.

# Arguments
- `ks_state_augmented`: KS augmented state vector [y_vec; y_vec_prime; h]
- `s`: fictitious time (not used but required for ODE interface)
- `epoch`: current epoch (Epoch object)
- `OMEGA_EARTH`: Earth rotation rate
- `CD`: drag coefficient
- `A`: cross-sectional area
- `m`: mass
"""
function ks_drag_perturbation(ks_state_augmented, s, epoch, OMEGA_EARTH, CD, A, m)
    y_vec = ks_state_augmented[1:4]
    y_vec_prime = ks_state_augmented[5:8]
    ks_state = ks_state_augmented[1:8]

    x_vec = state_ks_to_cartesian(ks_state)
    a_drag = cartesian_drag_perturbation(x_vec, s, epoch, OMEGA_EARTH, CD, A, m)
    y_vec_pprime = (y_vec'y_vec / 2.0) * (L(y_vec)' * [a_drag; 0.0])

    h_prime = -2 * y_vec_prime' * L(y_vec)' * [a_drag; 0.0]

    return [zeros(4); y_vec_pprime; h_prime]
end

"""
    ks_perturbed_dynamics!(ks_state_augmented_prime, ks_state_augmented, p, s)

In-place version for DifferentialEquations.jl.

# Arguments
- `ks_state_augmented_prime`: output derivative vector
- `ks_state_augmented`: state vector [y_vec; y_vec_prime; h]
- `p`: parameters (GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m)
- `s`: fictitious time
"""
function ks_perturbed_dynamics!(ks_state_full_prime, ks_state_full, p, s)
    # Unpack parameters tuple: (GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m)
    GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m = p

    ks_state_augmented = ks_state_full[1:9]
    t_current = ks_state_full[10]  # Current real time

    # Update epoch to current time (t_current is elapsed time since initial epoch)
    epoch = epoch_0 + t_current

    # KS dynamics
    ks_state_augmented_prime = ks_gravity(ks_state_augmented, s, GM)
    ks_state_augmented_prime += ks_J2_perturbation(ks_state_augmented, s, GM, R_EARTH, J2)
    ks_state_augmented_prime += ks_drag_perturbation(ks_state_augmented, s, epoch, OMEGA_EARTH, CD, A, m)

    # Time derivative: dt/ds = r = y'y
    y_vec = ks_state_augmented[1:4]
    r_vec_norm = y_vec'y_vec
    t_prime = r_vec_norm

    ks_state_full_prime .= [ks_state_augmented_prime; t_prime]
end

"""
    propagate_ks_perturbed_dynamics(ks_state_augmented_0, times, sim_params, GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m)

Propagate KS state with time tracking using perturbed dynamics.

# Arguments
- `ks_state_augmented_0`: initial KS augmented state [y_vec; y_vec_prime; h]
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter
- `R_EARTH`: Earth radius
- `J2`: J2 coefficient
- `epoch_0`: initial epoch (Epoch object)
- `OMEGA_EARTH`: Earth rotation rate
- `CD`: drag coefficient
- `A`: cross-sectional area
- `m`: mass
"""
function propagate_ks_perturbed_dynamics(ks_state_augmented_0, times, sim_params, GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m)
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

    # Use tuple instead of array for better performance with mixed types
    prob = ODEProblem(ks_perturbed_dynamics!, ks_state_full_0, (s_0, s_end), (GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m))
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, callback=cb)

    # Convert KS states to Cartesian
    x_vec_traj_ks = [state_ks_to_cartesian(ks_state_augmented_traj[k][1:8]) for k = 1:length(times)]
    return x_vec_traj_ks, ks_state_augmented_traj, t_traj
end