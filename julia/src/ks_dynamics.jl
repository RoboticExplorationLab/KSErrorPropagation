"""
KS (Kustaanheimo-Stiefel) dynamics with gravity only (no J2, no drag).
"""

using LinearAlgebra
using DifferentialEquations
using ForwardDiff

include("ks_transform.jl")
include("cartesian_dynamics.jl")
include("error_propagation.jl")

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
    T_state = eltype(ks_state_augmented_0)
    T_full = promote_type(T_state, Float64)
    ks_state_full_0 = Vector{T_full}(undef, 10)
    ks_state_full_0[1:9] = ks_state_augmented_0
    ks_state_full_0[10] = times[1]

    # Pre-allocate trajectory storage
    ks_state_augmented_traj = [similar(ks_state_augmented_0) for k = 1:length(times)]
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

    # Ensure final state is saved if callback didn't catch it
    if t_traj[end] != times[end]
        # Interpolate or use final solution state
        ks_state_full_final = sol.u[end]
        ks_state_augmented_traj[end] .= ks_state_full_final[1:9]
        t_traj[end] = ks_state_full_final[10]
    end

    # Convert KS states to Cartesian
    x_vec_traj_ks = [state_ks_to_cartesian(ks_state_augmented_traj[k][1:8]) for k = 1:length(times)]
    return x_vec_traj_ks, ks_state_augmented_traj, t_traj
end

"""
    propagate_ks_keplerian_dynamics_linearized(x_vec_0, P_0, times, sim_params, GM)

Propagate state uncertainty using linearized Gaussian propagation in KS coordinates.

# Arguments
- `x_vec_0`: initial mean state vector [r_vec; v_vec] (6-dimensional, Cartesian)
- `P_0`: initial covariance matrix (6×6, Cartesian)
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter

# Returns
- `x_vec_traj_mean`: array of mean states at each time (Cartesian)
- `P_traj`: array of covariance matrices at each time (Cartesian)
- `times`: array of times
"""
function propagate_ks_keplerian_dynamics_linearized(x_vec_0, P_0, times, sim_params, GM)
    # Convert initial state to KS
    ks_state_0 = state_cartesian_to_ks(x_vec_0)
    h_0 = energy_ks(ks_state_0[1:4], ks_state_0[5:8], GM)
    ks_state_augmented_0 = [ks_state_0; h_0]

    # Propagate mean in KS coordinates
    x_vec_traj_mean_ks, ks_state_augmented_traj, t_traj = propagate_ks_keplerian_dynamics(ks_state_augmented_0, times, sim_params, GM)

    # Pre-allocate covariance trajectory
    P_traj = Vector{Matrix{Float64}}(undef, length(times))
    P_traj[1] = copy(P_0)
    P_traj_ks = Vector{Matrix{Float64}}(undef, length(times))
    P_traj_ks[1] = transform_covariance_cartesian_to_ks(P_0, x_vec_0)

    # Propagate covariance at each time step
    for t_idx in 2:length(times)
        # Get current mean state in Cartesian
        ks_state_augmented_current = ks_state_augmented_traj[t_idx-1]
        dt = times[t_idx] - times[t_idx-1]

        # Compute state transition matrix F in KS space (8x8, excluding energy)
        # Function to compute state at next time given current state
        function state_transition_wrapper(ks_state_input)
            # Extract primal values if ks_state_input contains Dual types
            ks_state_input_primal = ForwardDiff.value.(ks_state_input)

            # Reconstruct augmented state (add energy)
            h_input = energy_ks(ks_state_input_primal[1:4], ks_state_input_primal[5:8], GM)
            ks_state_augmented_input = [ks_state_input_primal; h_input]

            # Propagate from ks_state_augmented_input to next time
            times_small = [times[t_idx-1], times[t_idx]]
            _, ks_state_augmented_traj_small, _ = propagate_ks_keplerian_dynamics(ks_state_augmented_input, times_small, sim_params, GM)

            # Return only the 8-element KS state (excluding energy)
            return ks_state_augmented_traj_small[end][1:8]
        end

        # Get current KS state (8 elements, excluding energy)
        ks_state_current = ks_state_augmented_current[1:8]

        # Compute Jacobian using ForwardDiff (8x8)
        F = ForwardDiff.jacobian(state_transition_wrapper, ks_state_current)

        # Propagate covariance: P(t+dt) = F * P(t) * F'
        P_traj_ks[t_idx] = F * P_traj_ks[t_idx-1] * F'

        # Ensure symmetry
        P_traj_ks[t_idx] = (P_traj_ks[t_idx] + P_traj_ks[t_idx]') / 2.0

        # Convert covariance from KS to Cartesian
        P_traj[t_idx] = transform_covariance_ks_to_cartesian(P_traj_ks[t_idx], ks_state_augmented_traj[t_idx][1:8])

        # Ensure symmetry
        P_traj[t_idx] = (P_traj[t_idx] + P_traj[t_idx]') / 2.0
    end

    return x_vec_traj_mean_ks, P_traj, t_traj
end

"""
    propagate_ks_keplerian_dynamics_sigma_points(x_vec_0, P_0, times, sim_params, GM)

Propagate state uncertainty using sigma point (Unscented Transform) method in KS coordinates.

# Arguments
- `x_vec_0`: initial mean state vector [r_vec; v_vec] (6-dimensional, Cartesian)
- `P_0`: initial covariance matrix (6×6, Cartesian)
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter

# Returns
- `x_vec_traj_mean`: array of mean states at each time (Cartesian)
- `P_traj`: array of covariance matrices at each time (Cartesian)
- `times`: array of times
"""
function propagate_ks_keplerian_dynamics_sigma_points(x_vec_0, P_0, times, sim_params, GM)
    # Generate sigma points in Cartesian space
    sigma_points_0, weights_mean, weights_cov = generate_sigma_points(x_vec_0, P_0)
    num_sigma_points = length(sigma_points_0)

    # Pre-allocate storage for propagated sigma points at each time
    sigma_points_propagated = Vector{Vector{Vector{Float64}}}(undef, length(times))
    for t_idx in 1:length(times)
        sigma_points_propagated[t_idx] = Vector{Vector{Float64}}(undef, num_sigma_points)
    end

    # Propagate each sigma point
    println("  Propagating ", num_sigma_points, " sigma points...")
    for (sp_idx, sigma_point_0) in enumerate(sigma_points_0)
        # Transform to KS
        ks_state_sp_0 = state_cartesian_to_ks_via_newton_method(sigma_point_0)
        h_sp_0 = energy_ks(ks_state_sp_0[1:4], ks_state_sp_0[5:8], GM)
        ks_state_augmented_sp_0 = [ks_state_sp_0; h_sp_0]

        # Propagate through full nonlinear KS dynamics
        x_vec_traj_sp, _, _ = propagate_ks_keplerian_dynamics(ks_state_augmented_sp_0, times, sim_params, GM)

        # Store propagated states (already in Cartesian)
        for t_idx in 1:length(times)
            sigma_points_propagated[t_idx][sp_idx] = x_vec_traj_sp[t_idx]
        end
    end

    # Reconstruct mean and covariance at each time
    x_vec_traj_mean = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))

    for t_idx in 1:length(times)
        x_vec_traj_mean[t_idx], P_traj[t_idx] = reconstruct_from_sigma_points(
            sigma_points_propagated[t_idx], weights_mean, weights_cov)
    end

    return x_vec_traj_mean, P_traj, times
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
    return [zeros(4, 4) I(4); -0.5*h*I(4) zeros(4, 4)]
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

"""
    propagate_ks_keplerian_relative_dynamics(x_vec_chief_0, x_vec_deputy_0, times, sim_params, GM)

Propagate KS relative state with time tracking using Keplerian relative dynamics.

# Arguments
- `x_vec_chief_0`: initial Cartesian state of chief [r_vec; v_vec]
- `x_vec_deputy_0`: initial Cartesian state of deputy [r_vec; v_vec]
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter

# Returns
- `x_vec_traj_chief_ks`: array of Cartesian states of chief [r_vec; v_vec] at each time
- `x_vec_traj_rel_ks`: array of Cartesian relative states [r_vec; v_vec] at each time
- `t_chief_traj`: array of times for chief
- `t_deputy_traj`: array of times for deputy
"""
function propagate_ks_keplerian_relative_dynamics(x_vec_chief_0, x_vec_deputy_0, times, sim_params, GM)
    # Convert Cartesian states to KS states
    ks_state_chief_0 = state_cartesian_to_ks_via_newton_method(x_vec_chief_0)
    ks_state_deputy_0 = state_cartesian_to_ks_via_newton_method(x_vec_deputy_0; y_vec_near=ks_state_chief_0[1:4])
    ks_state_rel_0 = ks_state_deputy_0 .- ks_state_chief_0

    # Compute energy
    h_chief_0 = energy_ks(ks_state_chief_0[1:4], ks_state_chief_0[5:8], GM)
    h_deputy_0 = energy_ks(ks_state_deputy_0[1:4], ks_state_deputy_0[5:8], GM)
    h_rel_0 = h_deputy_0 - h_chief_0

    # ForwardDiff function for KS gravity
    function f_function(ks_state_augmented)
        y_vec = ks_state_augmented[1:4]
        y_vec_prime = ks_state_augmented[5:8]
        h = ks_state_augmented[9]

        # Unperturbed KS dynamics (gravity only)
        y_vec_pprime = (-h / 2.0) * y_vec
        h_prime = 0.0

        return [y_vec_prime; y_vec_pprime; h_prime]
    end

    # Keplerian relative dynamics
    function ks_keplerian_relative_dynamics!(z_prime, z, p, s)
        ks_state_agumented_chief = z[1:9]
        ks_state_agumented_rel = z[10:18]

        # Nonlinear dynamics for chief
        ks_state_agumented_chief_prime = f_function(ks_state_agumented_chief)

        # Linearize around chief position
        ∂f∂ks_state_agumented_chief = ForwardDiff.jacobian(f_function, ks_state_agumented_chief)

        # Linear relative dynamics
        ks_state_agumented_rel_prime = ∂f∂ks_state_agumented_chief * ks_state_agumented_rel
        ks_state_agumented_deputy = ks_state_agumented_chief .+ ks_state_agumented_rel

        # Time derivatives
        t_chief_prime = ks_state_agumented_chief[1:4]'ks_state_agumented_chief[1:4]
        t_deputy_prime = ks_state_agumented_deputy[1:4]'ks_state_agumented_deputy[1:4]

        z_prime .= [ks_state_agumented_chief_prime; ks_state_agumented_rel_prime; t_chief_prime; t_deputy_prime]
    end

    # Initial condition
    z_0 = [ks_state_chief_0; h_chief_0; ks_state_rel_0; h_rel_0; times[1]; times[1]]

    # Pre-allocate trajectory storage
    ks_state_agumented_chief_traj = [zeros(9) for k = 1:length(times)]
    ks_state_agumented_chief_traj[1] .= [ks_state_chief_0; h_chief_0]
    ks_state_agumented_deputy_traj = [zeros(9) for k = 1:length(times)]
    ks_state_agumented_deputy_traj[1] .= [ks_state_deputy_0; h_deputy_0]
    t_deputy_traj = zeros(length(times))
    t_deputy_traj[1] = times[1]
    t_chief_traj = zeros(length(times))
    t_chief_traj[1] = times[1]

    ############################ CALLBACK FUNCTIONS ############################
    # Callback to save states at specific real time points
    # SOLVER HAS ISSUES WHEN RELATIVE STATE IS ZERO
    function condition_separate!(out, z, s, sol)
        t_chief_current = z[end-1]
        t_deputy_current = z[end]
        out .= [times .- t_chief_current; times .- t_deputy_current]
    end

    function affect_separate!(sol, idx)
        if idx <= length(times)
            # Chief timepoint
            ks_state_agumented_chief_traj[idx] .= sol.u[1:9]
            t_chief_traj[idx] = sol.u[end-1]
        else
            # Deputy timepoint
            idx -= length(times)
            ks_state_agumented_deputy_traj[idx] .= sol.u[10:18] .+ sol.u[1:9]
            t_deputy_traj[idx] = sol.u[end]
        end
    end

    function condition_together!(out, z, s, sol)
        t_chief_current = z[end-1]
        out .= times .- t_chief_current
    end

    function affect_together!(sol, idx)
        if idx <= length(times)
            # Chief timepoint
            ks_state_agumented_chief_traj[idx] .= sol.u[1:9]
            t_chief_traj[idx] = sol.u[end-1]

            # Deputy timepoint
            ks_state_agumented_deputy_traj[idx] .= sol.u[10:18] .+ sol.u[1:9]
            t_deputy_traj[idx] = sol.u[end]
        end
    end

    if norm(ks_state_rel_0[1:4]) > 1e-14
        cb = VectorContinuousCallback(condition_separate!, affect_separate!, 2 * length(times))
    else
        cb = VectorContinuousCallback(condition_together!, affect_together!, length(times))
    end
    ############################################################################

    # Estimate fictitious time span
    r_vec_norm_0 = norm(x_vec_chief_0[1:3])
    s_0 = 0.0
    s_end = (times[end] - times[1]) / r_vec_norm_0

    # Numerical solver
    prob = ODEProblem(ks_keplerian_relative_dynamics!, z_0, (s_0, s_end))
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, callback=cb)

    # Convert KS states to Cartesian
    x_vec_traj_chief_ks = [state_ks_to_cartesian(ks_state_agumented_chief_traj[k][1:8]) for k = 1:length(times)]
    x_vec_traj_deputy_ks = [state_ks_to_cartesian(ks_state_agumented_deputy_traj[k][1:8]) for k = 1:length(times)]
    x_vec_traj_rel_ks = [x_vec_traj_deputy_ks[k][1:6] .- x_vec_traj_chief_ks[k][1:6] for k = 1:length(times)]

    return x_vec_traj_chief_ks, x_vec_traj_rel_ks, t_chief_traj, t_deputy_traj
end

"""
    propagate_ks_keplerian_relative_dynamics_sigma_points(x_vec_chief_0, x_vec_deputy_0, P_rel_0, times, sim_params, GM)

Propagate relative state uncertainty using sigma point (Unscented Transform) method.

# Arguments
- `x_vec_chief_0`: initial Cartesian state of chief [r_vec; v_vec]
- `x_vec_deputy_0`: initial Cartesian state of deputy [r_vec; v_vec]
- `P_rel_0`: initial relative state covariance matrix (6×6 in Cartesian)
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter

# Returns
- `x_vec_traj_rel_mean`: array of mean relative states [r_vec; v_vec] at each time
- `P_traj_rel`: array of relative state covariance matrices at each time
- `t_traj`: array of times
"""
function propagate_ks_keplerian_relative_dynamics_sigma_points(x_vec_chief_0, x_vec_deputy_0, P_rel_0, times, sim_params, GM)
    # Initial relative state mean
    x_vec_rel_0 = x_vec_deputy_0 .- x_vec_chief_0

    # Generate sigma points in Cartesian relative space
    sigma_points_rel_0, weights_mean, weights_cov = generate_sigma_points(x_vec_rel_0, P_rel_0)
    num_sigma_points = length(sigma_points_rel_0)

    # Propagate chief separately (nonlinear, no uncertainty)
    x_vec_traj_chief, _, t_traj_chief = propagate_cartesian_keplerian_dynamics(x_vec_chief_0, times, sim_params, GM)

    # Pre-allocate storage for propagated sigma points at each time
    sigma_points_rel_propagated = Vector{Vector{Vector{Float64}}}(undef, length(times))

    # Propagate each sigma point
    for sp_idx in 1:num_sigma_points
        # Convert relative sigma point to absolute deputy state
        x_vec_deputy_sp = x_vec_chief_0 .+ sigma_points_rel_0[sp_idx]

        # Transform to KS
        ks_state_deputy_sp_0 = state_cartesian_to_ks_via_newton_method(x_vec_deputy_sp)
        h_deputy_sp_0 = energy_ks(ks_state_deputy_sp_0[1:4], ks_state_deputy_sp_0[5:8], GM)
        ks_state_augmented_deputy_sp_0 = [ks_state_deputy_sp_0; h_deputy_sp_0]

        # Propagate through full nonlinear dynamics
        x_vec_traj_deputy_sp, _, t_traj_deputy_sp = propagate_ks_keplerian_dynamics(
            ks_state_augmented_deputy_sp_0, times, sim_params, GM)

        # Convert to relative states at each time
        if sp_idx == 1
            # Initialize storage on first iteration
            for t_idx in 1:length(times)
                sigma_points_rel_propagated[t_idx] = Vector{Vector{Float64}}(undef, num_sigma_points)
            end
        end

        for t_idx in 1:length(times)
            # Compute relative state: deputy - chief
            x_vec_rel_sp = x_vec_traj_deputy_sp[t_idx] .- x_vec_traj_chief[t_idx]
            sigma_points_rel_propagated[t_idx][sp_idx] = x_vec_rel_sp
        end
    end

    # Reconstruct mean and covariance at each time
    x_vec_traj_rel_mean = Vector{Vector{Float64}}(undef, length(times))
    P_traj_rel = Vector{Matrix{Float64}}(undef, length(times))

    for t_idx in 1:length(times)
        x_vec_traj_rel_mean[t_idx], P_traj_rel[t_idx] = reconstruct_from_sigma_points(
            sigma_points_rel_propagated[t_idx], weights_mean, weights_cov)
    end

    return x_vec_traj_rel_mean, P_traj_rel, times
end

"""
    propagate_ks_keplerian_relative_dynamics_linearized(x_vec_chief_0, x_vec_deputy_0, P_rel_0, times, sim_params, GM)

Propagate relative state uncertainty using linearized Gaussian propagation.

# Arguments
- `x_vec_chief_0`: initial Cartesian state of chief [r_vec; v_vec]
- `x_vec_deputy_0`: initial Cartesian state of deputy [r_vec; v_vec]
- `P_rel_0`: initial relative state covariance matrix (6×6 in Cartesian)
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter

# Returns
- `x_vec_traj_rel_mean`: array of mean relative states [r_vec; v_vec] at each time
- `P_traj_rel`: array of relative state covariance matrices at each time
- `t_traj`: array of times
"""
function propagate_ks_keplerian_relative_dynamics_linearized(x_vec_chief_0, x_vec_deputy_0, P_rel_0, times, sim_params, GM)
    # Use existing function for mean propagation
    x_vec_traj_chief, x_vec_traj_rel_mean, t_chief_traj, t_deputy_traj = propagate_ks_keplerian_relative_dynamics(
        x_vec_chief_0, x_vec_deputy_0, times, sim_params, GM)

    # Pre-allocate covariance trajectory
    P_traj_rel = Vector{Matrix{Float64}}(undef, length(times))
    P_traj_rel[1] = copy(P_rel_0)

    # Propagate covariance at each time step
    for t_idx in 2:length(times)
        # Get current mean states
        x_vec_chief_current = x_vec_traj_chief[t_idx-1]
        x_vec_rel_current = x_vec_traj_rel_mean[t_idx-1]
        x_vec_deputy_current = x_vec_chief_current .+ x_vec_rel_current

        # Time step
        dt = times[t_idx] - times[t_idx-1]

        # Compute state transition matrix F in Cartesian relative space
        # F = ∂(x_rel(t+dt)) / ∂(x_rel(t))
        # We compute this by linearizing the relative dynamics

        # Function to compute relative state at next time given current relative state
        function relative_dynamics_wrapper(x_rel_input)
            x_deputy_input = x_vec_chief_current .+ x_rel_input

            # Convert to KS
            ks_state_chief = state_cartesian_to_ks_via_newton_method(x_vec_chief_current)
            ks_state_deputy = state_cartesian_to_ks_via_newton_method(x_deputy_input; y_vec_near=ks_state_chief[1:4])

            # Compute energy
            h_chief = energy_ks(ks_state_chief[1:4], ks_state_chief[5:8], GM)
            h_deputy = energy_ks(ks_state_deputy[1:4], ks_state_deputy[5:8], GM)

            # Create augmented states
            ks_state_augmented_chief = [ks_state_chief; h_chief]
            ks_state_augmented_deputy = [ks_state_deputy; h_deputy]

            # Propagate both through small time step
            times_small = [times[t_idx-1], times[t_idx]]
            _, x_vec_traj_chief_small, _ = propagate_ks_keplerian_dynamics(ks_state_augmented_chief, times_small, sim_params, GM)
            _, x_vec_traj_deputy_small, _ = propagate_ks_keplerian_dynamics(ks_state_augmented_deputy, times_small, sim_params, GM)

            # Compute relative state at next time
            x_rel_next = x_vec_traj_deputy_small[2] .- x_vec_traj_chief_small[2]

            return x_rel_next
        end

        # Compute Jacobian using ForwardDiff
        F = ForwardDiff.jacobian(relative_dynamics_wrapper, x_vec_rel_current)

        # Propagate covariance: P(t+dt) = F * P(t) * F'
        P_traj_rel[t_idx] = F * P_traj_rel[t_idx-1] * F'

        # Ensure symmetry
        P_traj_rel[t_idx] = (P_traj_rel[t_idx] + P_traj_rel[t_idx]') / 2.0
    end

    return x_vec_traj_rel_mean, P_traj_rel, times
end

"""
    propagate_ks_perturbed_relative_dynamics(x_vec_chief_0, x_vec_deputy_0, times, sim_params, GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m)

Propagate KS relative state with time tracking using perturbed relative dynamics (J2 + drag).

# Arguments
- `x_vec_chief_0`: initial Cartesian state of chief [r_vec; v_vec]
- `x_vec_deputy_0`: initial Cartesian state of deputy [r_vec; v_vec]
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

# Returns
- `x_vec_traj_chief_ks`: array of Cartesian states of chief [r_vec; v_vec] at each time
- `x_vec_traj_rel_ks`: array of Cartesian relative states [r_vec; v_vec] at each time
- `t_chief_traj`: array of times for chief
- `t_deputy_traj`: array of times for deputy
"""
function propagate_ks_perturbed_relative_dynamics(x_vec_chief_0, x_vec_deputy_0, times, sim_params, GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m)
    # Convert Cartesian states to KS states
    ks_state_chief_0 = state_cartesian_to_ks_via_newton_method(x_vec_chief_0)
    ks_state_deputy_0 = state_cartesian_to_ks_via_newton_method(x_vec_deputy_0; y_vec_near=ks_state_chief_0[1:4])
    ks_state_rel_0 = ks_state_deputy_0 .- ks_state_chief_0

    # Compute energy
    h_chief_0 = energy_ks(ks_state_chief_0[1:4], ks_state_chief_0[5:8], GM)
    h_deputy_0 = energy_ks(ks_state_deputy_0[1:4], ks_state_deputy_0[5:8], GM)
    h_rel_0 = h_deputy_0 - h_chief_0

    # ForwardDiff function for KS perturbed dynamics
    function f_function(ks_state_augmented, t_current)
        y_vec = ks_state_augmented[1:4]
        y_vec_prime = ks_state_augmented[5:8]
        h = ks_state_augmented[9]

        # Gravity (unperturbed KS dynamics) - REQUIRED as base
        y_vec_pprime = (-h / 2.0) * y_vec
        h_prime = 0.0

        # Add J2 perturbation (perturbation ON TOP OF gravity)
        ks_state = ks_state_augmented[1:8]
        x_vec = state_ks_to_cartesian(ks_state)
        r_vec = x_vec[1:3]
        r_vec_norm = norm(r_vec)
        a_J2 = -(3.0 / 2.0) * J2 * (GM / r_vec_norm^2) * (R_EARTH / r_vec_norm)^2 * [
                   (1 - 5 * (r_vec[3] / r_vec_norm)^2) * r_vec[1] / r_vec_norm,
                   (1 - 5 * (r_vec[3] / r_vec_norm)^2) * r_vec[2] / r_vec_norm,
                   (3 - 5 * (r_vec[3] / r_vec_norm)^2) * r_vec[3] / r_vec_norm]
        y_vec_pprime_J2 = (y_vec'y_vec / 2.0) * (L(y_vec)' * [a_J2; 0.0])
        h_prime_J2 = -2 * y_vec_prime' * L(y_vec)' * [a_J2; 0.0]

        # Add drag perturbation
        epoch = epoch_0 + t_current
        v_vec = x_vec[4:6]
        omega = [0.0, 0.0, OMEGA_EARTH]
        v_vec_rel = v_vec - cross(omega, r_vec)
        v_vec_rel_norm = norm(v_vec_rel)

        y_vec_pprime_drag = zeros(4)
        h_prime_drag = 0.0

        if v_vec_rel_norm >= 1e-6
            # # Use NRLMSISE-00 atmospheric density model
            # Extract primal values for SatelliteDynamics functions that don't support Dual types
            # r_vec_primal = ForwardDiff.value.(r_vec)
            # r_ecef = SD.sECItoECEF(epoch, r_vec_primal)
            # geod = SD.sECEFtoGEOD(r_ecef; use_degrees=false)
            # rho = SD.density_nrlmsise00(epoch, geod; use_degrees=false)

            # Use Harris-Priester atmospheric density model
            rho = SD.density_harris_priester(epoch, r_vec)

            a_drag = -0.5 * (CD * A / m) * rho * v_vec_rel_norm * v_vec_rel
            y_vec_pprime_drag = (y_vec'y_vec / 2.0) * (L(y_vec)' * [a_drag; 0.0])
            h_prime_drag = -2 * y_vec_prime' * L(y_vec)' * [a_drag; 0.0]
        end

        # Sum all contributions: gravity + J2 (+ drag if enabled)
        y_vec_pprime_total = y_vec_pprime + y_vec_pprime_J2 + y_vec_pprime_drag
        h_prime_total = h_prime + h_prime_J2 + h_prime_drag

        return [y_vec_prime; y_vec_pprime_total; h_prime_total]
    end

    # Perturbed relative dynamics
    function ks_perturbed_relative_dynamics!(z_prime, z, p, s)
        ks_state_agumented_chief = z[1:9]
        ks_state_agumented_rel = z[10:18]

        # Nonlinear dynamics for chief
        t_chief_current = z[end-1]
        ks_state_agumented_chief_prime = f_function(ks_state_agumented_chief, t_chief_current)

        # Linearize around chief position
        ∂f∂ks_state_agumented_chief = ForwardDiff.jacobian(x -> f_function(x, t_chief_current), ks_state_agumented_chief)

        # Linear relative dynamics
        ks_state_agumented_rel_prime = ∂f∂ks_state_agumented_chief * ks_state_agumented_rel
        ks_state_agumented_deputy = ks_state_agumented_chief .+ ks_state_agumented_rel

        # Time derivatives
        t_chief_prime = ks_state_agumented_chief[1:4]'ks_state_agumented_chief[1:4]
        t_deputy_prime = ks_state_agumented_deputy[1:4]'ks_state_agumented_deputy[1:4]

        z_prime .= [ks_state_agumented_chief_prime; ks_state_agumented_rel_prime; t_chief_prime; t_deputy_prime]
    end

    # Initial condition
    z_0 = [ks_state_chief_0; h_chief_0; ks_state_rel_0; h_rel_0; times[1]; times[1]]

    # Pre-allocate trajectory storage
    ks_state_agumented_chief_traj = [zeros(9) for k = 1:length(times)]
    ks_state_agumented_chief_traj[1] .= [ks_state_chief_0; h_chief_0]
    ks_state_agumented_deputy_traj = [zeros(9) for k = 1:length(times)]
    ks_state_agumented_deputy_traj[1] .= [ks_state_deputy_0; h_deputy_0]
    t_deputy_traj = zeros(length(times))
    t_deputy_traj[1] = times[1]
    t_chief_traj = zeros(length(times))
    t_chief_traj[1] = times[1]

    ############################ CALLBACK FUNCTIONS ############################
    # Callback to save states at specific real time points
    # SOLVER HAS ISSUES WHEN RELATIVE STATE IS ZERO
    function condition_separate!(out, z, s, sol)
        t_chief_current = z[end-1]
        t_deputy_current = z[end]
        out .= [times .- t_chief_current; times .- t_deputy_current]
    end

    function affect_separate!(sol, idx)
        if idx <= length(times)
            # Chief timepoint
            ks_state_agumented_chief_traj[idx] .= sol.u[1:9]
            t_chief_traj[idx] = sol.u[end-1]
        else
            # Deputy timepoint
            idx -= length(times)
            ks_state_agumented_deputy_traj[idx] .= sol.u[10:18] .+ sol.u[1:9]
            t_deputy_traj[idx] = sol.u[end]
        end
    end

    function condition_together!(out, z, s, sol)
        t_chief_current = z[end-1]
        out .= times .- t_chief_current
    end

    function affect_together!(sol, idx)
        if idx <= length(times)
            # Chief timepoint
            ks_state_agumented_chief_traj[idx] .= sol.u[1:9]
            t_chief_traj[idx] = sol.u[end-1]

            # Deputy timepoint
            ks_state_agumented_deputy_traj[idx] .= sol.u[10:18] .+ sol.u[1:9]
            t_deputy_traj[idx] = sol.u[end]
        end
    end

    if norm(ks_state_rel_0[1:4]) > 1e-14
        cb = VectorContinuousCallback(condition_separate!, affect_separate!, 2 * length(times))
    else
        cb = VectorContinuousCallback(condition_together!, affect_together!, length(times))
    end
    ############################################################################

    # Estimate fictitious time span
    r_vec_norm_0 = norm(x_vec_chief_0[1:3])
    s_0 = 0.0
    s_end = (times[end] - times[1]) / r_vec_norm_0

    # Numerical solver
    prob = ODEProblem(ks_perturbed_relative_dynamics!, z_0, (s_0, s_end))
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, callback=cb)

    # Convert KS states to Cartesian
    x_vec_traj_chief_ks = [state_ks_to_cartesian(ks_state_agumented_chief_traj[k][1:8]) for k = 1:length(times)]
    x_vec_traj_deputy_ks = [state_ks_to_cartesian(ks_state_agumented_deputy_traj[k][1:8]) for k = 1:length(times)]
    x_vec_traj_rel_ks = [x_vec_traj_deputy_ks[k][1:6] .- x_vec_traj_chief_ks[k][1:6] for k = 1:length(times)]

    return x_vec_traj_chief_ks, x_vec_traj_rel_ks, t_chief_traj, t_deputy_traj
end