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

"""
    propagate_ks_analytical_keplerian_orbit(ks_state_augmented_0, times, sim_params, GM)

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
function propagate_ks_analytical_keplerian_orbit(ks_state_augmented_0, times, sim_params, GM)
    # Extract initial conditions
    y_vec_0 = ks_state_augmented_0[1:4]
    y_vec_prime_0 = ks_state_augmented_0[5:8]

    # Energy is constant (use provided h or recompute for consistency)
    h = ks_state_augmented_0[9]

    # Pre-allocate trajectory storage
    ks_state_augmented_traj = [zeros(9) for k = 1:length(times)]
    t_traj = zeros(length(times))

    # Set initial state
    ks_state_augmented_traj[1] = [y_vec_0; y_vec_prime_0; h]
    t_traj[1] = times[1]

    # Analytical solution depends on sign of h
    if h > 0
        # Elliptic case: use STM to propagate [y; y'; t] forward in s
        ω = sqrt(h / 2.0)
        I4 = Matrix{Float64}(I, 4, 4)

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
                Δs = min(Δs_estimate, max_Δs, 0.1 / max(ω, 1e-10))
            else
                # Shouldn't happen if times are increasing, but handle it
                break
            end

            # Build 9×9 STM for step Δs
            # For KS state: [y; y'] propagates via 8×8 block
            cos_ωΔs = cos(ω * Δs)
            sin_ωΔs = sin(ω * Δs)
            cos_2ωΔs = cos(2 * ω * Δs)
            sin_2ωΔs = sin(2 * ω * Δs)

            # 8×8 STM for [y; y']
            Φ_88 = [cos_ωΔs*I4 (1/ω)*sin_ωΔs*I4;
                -ω*sin_ωΔs*I4 cos_ωΔs*I4]

            # Compute time increment Δt(Δs) from current state
            # Coefficients computed from current y and y'
            y_dot_y = y_vec'y_vec
            y_dot_yp = y_vec'y_vec_prime
            yp_dot_yp = y_vec_prime'y_vec_prime

            A = 0.5 * (y_dot_y + yp_dot_yp / (ω^2))
            B = 0.5 * (y_dot_y - yp_dot_yp / (ω^2))
            C = y_dot_yp / ω

            # Time increment: Δt(Δs) = A*Δs + (B/(2ω))*sin(2ωΔs) - (C/(2ω))*(cos(2ωΔs) - 1)
            Δt = A * Δs + (B / (2 * ω)) * sin_2ωΔs - (C / (2 * ω)) * (cos_2ωΔs - 1.0)

            # Build full 9×9 STM
            Φ = zeros(9, 9)
            Φ[1:8, 1:8] = Φ_88
            Φ[9, 9] = 1.0  # t propagates as t_new = t + Δt

            # Apply STM: z_aug_new = Φ * z_aug (but t needs special handling)
            z_aug_new = Φ * z_aug
            z_aug_new[9] = t_current + Δt  # Update time explicitly

            # Check if we've crossed the target time
            if z_aug_new[9] >= t_target
                # Compute exact state at t_target using analytical solution
                # We need to find Δs_exact such that t(s + Δs_exact) = t_target
                # Use Newton's method with good initial guess (the Δs we just used)
                Δs_exact = Δs
                for iter = 1:10
                    # Compute time at s + Δs_exact
                    cos_ωΔs_exact = cos(ω * Δs_exact)
                    sin_ωΔs_exact = sin(ω * Δs_exact)
                    cos_2ωΔs_exact = cos(2 * ω * Δs_exact)
                    sin_2ωΔs_exact = sin(2 * ω * Δs_exact)

                    Δt_exact = A * Δs_exact + (B / (2 * ω)) * sin_2ωΔs_exact - (C / (2 * ω)) * (cos_2ωΔs_exact - 1.0)
                    t_at_s_plus_ds = t_current + Δt_exact

                    residual = t_at_s_plus_ds - t_target
                    if abs(residual) < 1e-12
                        break
                    end

                    # Derivative: dt/ds = r(s + Δs_exact)
                    # Compute state at s + Δs_exact
                    y_vec_at_s_plus_ds = y_vec * cos_ωΔs_exact + (y_vec_prime / ω) * sin_ωΔs_exact
                    r_at_s_plus_ds = y_vec_at_s_plus_ds'y_vec_at_s_plus_ds

                    if abs(r_at_s_plus_ds) < 1e-15
                        break
                    end

                    Δs_exact = Δs_exact - residual / r_at_s_plus_ds
                    Δs_exact = clamp(Δs_exact, 0.0, Δs * 1.1)  # Keep it reasonable
                end

                # Compute exact state at s + Δs_exact
                cos_ωΔs_exact = cos(ω * Δs_exact)
                sin_ωΔs_exact = sin(ω * Δs_exact)
                y_vec_exact = y_vec * cos_ωΔs_exact + (y_vec_prime / ω) * sin_ωΔs_exact
                y_vec_prime_exact = -y_vec * ω * sin_ωΔs_exact + y_vec_prime * cos_ωΔs_exact

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
            if r_current < 1e6  # Small radius threshold
                max_Δs = 0.01 / max(ω, 1e-10)
            else
                max_Δs = 0.1 / max(ω, 1e-10)
            end
        end

    else
        error("Only elliptic case (h > 0) is currently implemented with STM approach")
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
    ks_perturbed_dynamics!(ks_state_augmented_prime, ks_state_augmented, p, s)

In-place version for DifferentialEquations.jl.

# Arguments
- `ks_state_augmented_prime`: output derivative vector
- `ks_state_augmented`: state vector [y_vec; y_vec_prime; h]
- `p`: parameters (GM, R_EARTH, J2)
- `t`: time
"""
function ks_perturbed_dynamics!(ks_state_full_prime, ks_state_full, p, s)
    GM = p isa Number ? p : p[1]
    R_EARTH = p isa Number ? p : p[2]
    J2 = p isa Number ? p : p[3]

    ks_state_augmented = ks_state_full[1:9]

    # KS dynamics
    ks_state_augmented_prime = ks_gravity(ks_state_augmented, s, GM) + ks_J2_perturbation(ks_state_augmented, s, GM, R_EARTH, J2)

    # Time derivative: dt/ds = r = y'y
    y_vec = ks_state_augmented[1:4]
    r_vec_norm = y_vec'y_vec
    t_prime = r_vec_norm

    ks_state_full_prime .= [ks_state_augmented_prime; t_prime]
end

"""
    propagate_ks_perturbed_dynamics(ks_state_augmented_0, times, sim_params, GM, R_EARTH, J2)

Propagate KS state with time tracking using perturbed dynamics.

# Arguments
- `ks_state_augmented_0`: initial KS augmented state [y_vec; y_vec_prime; h]
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter
- `R_EARTH`: Earth radius
- `J2`: J2 coefficient
"""
function propagate_ks_perturbed_dynamics(ks_state_augmented_0, times, sim_params, GM, R_EARTH, J2)
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

    prob = ODEProblem(ks_perturbed_dynamics!, ks_state_full_0, (s_0, s_end), [GM, R_EARTH, J2])
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, callback=cb)

    # Convert KS states to Cartesian
    x_vec_traj_ks = [state_ks_to_cartesian(ks_state_augmented_traj[k][1:8]) for k = 1:length(times)]
    return x_vec_traj_ks, ks_state_augmented_traj, t_traj
end