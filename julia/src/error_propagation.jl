using Random
using LinearAlgebra
using ForwardDiff
using DifferentialEquations

function propagate_uncertainty_via_monte_carlo(x_vec_0, P_0, times, sim_params, num_samples=10000; return_samples=false)
    # Sample initial states from multivariate Gaussian
    # Use Cholesky decomposition: if x ~ N(0, I), then L_chol*x + μ ~ N(μ, P) where P = L_chol*L_chol'
    L_chol = cholesky(P_0).L

    # Generate samples
    samples_0 = Vector{Vector{Float64}}(undef, num_samples)
    for i in 1:num_samples
        z = randn(6)  # Standard normal
        samples_0[i] = x_vec_0 .+ L_chol * z
    end

    # Pre-allocate trajectory storage
    samples_propagated = Vector{Vector{Vector{Float64}}}(undef, length(times))
    for t_idx in 1:length(times)
        samples_propagated[t_idx] = Vector{Vector{Float64}}(undef, num_samples)
    end

    # Propagate each sample
    println("  Propagating ", num_samples, " Monte Carlo samples...")
    for (i, sample_0) in enumerate(samples_0)
        if i % 1000 == 0
            println("    Sample ", i, " / ", num_samples)
        end

        # Propagate sample through full nonlinear dynamics
        x_vec_traj_sample, _ = propagate_cartesian_dynamics(sample_0, times, sim_params)

        # Store propagated states
        for t_idx in 1:length(times)
            samples_propagated[t_idx][i] = x_vec_traj_sample[t_idx]
        end
    end

    # Pre-allocate mean and covariance storage
    x_vec_traj = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))

    # Compute mean and covariance at each time
    for t_idx in 1:length(times)
        # Compute mean
        x_mean = zeros(6)
        for i in 1:num_samples
            x_mean .+= samples_propagated[t_idx][i]
        end
        x_mean ./= num_samples
        x_vec_traj[t_idx] = x_mean

        # Compute covariance
        P = zeros(6, 6)
        for i in 1:num_samples
            diff = samples_propagated[t_idx][i] .- x_mean
            P .+= diff * diff'
        end
        P ./= (num_samples - 1)  # Sample covariance (Bessel's correction)
        P_traj[t_idx] = (P + P') / 2.0  # Enforce symmetry
    end

    if return_samples
        return x_vec_traj, P_traj, samples_propagated
    else
        return x_vec_traj, P_traj
    end
end

function propagate_uncertainty_via_linearized_cartesian_dynamics(x_vec_0, P_0, times, sim_params)
    # Step 1: Scale initial state and covariance for numerical stability
    x_vec_0_scaled = [x_vec_0[1:3] / sim_params.r_scale; x_vec_0[4:6] / sim_params.v_scale]
    times_scaled = times / sim_params.t_scale
    P_0_scaled = sim_params.S * P_0 * sim_params.S'

    # Step 2: Propagate mean trajectory using full nonlinear dynamics
    x_vec_traj, _ = propagate_cartesian_dynamics(x_vec_0, times, sim_params)

    # Step 3: Create dynamics wrapper in SCALED coordinates for Jacobian computation
    # Wraps cartesian_dynamics! to return the derivative instead of mutating
    function dynamics_scaled(x_vec_scaled, t_scaled)
        # Pre-allocate output array with same type as input (for ForwardDiff compatibility)
        T = eltype(x_vec_scaled)
        x_vec_dot_scaled = Vector{T}(undef, 6)
        cartesian_dynamics!(x_vec_dot_scaled, x_vec_scaled, sim_params, t_scaled)
        return x_vec_dot_scaled
    end

    # Step 4: Compute state transition matrix (STM) in scaled coordinates
    # The STM satisfies: dΦ_scaled/dt_scaled = A_scaled(t_scaled) · Φ_scaled
    # where A_scaled = ∂f_scaled/∂x_scaled

    # Initialize STM as identity (in scaled coordinates)
    Φ_0_scaled = Matrix{Float64}(I, 6, 6)

    # Combine state and STM into augmented state: [x_scaled; vec(Φ_scaled)]
    # Φ_scaled is 6x6, so we need 36 additional states (total: 6 + 36 = 42)
    augmented_state_0_scaled = [x_vec_0_scaled; vec(Φ_0_scaled)]

    # Define augmented dynamics function in scaled coordinates
    function augmented_dynamics_scaled!(du, u, p, t_scaled)
        x_vec_scaled = u[1:6]
        Φ_vec_scaled = u[7:42]
        Φ_scaled = reshape(Φ_vec_scaled, 6, 6)

        # Compute Jacobian A_scaled = ∂f_scaled/∂x_scaled at current state using ForwardDiff
        A_scaled = ForwardDiff.jacobian(x -> dynamics_scaled(x, t_scaled), x_vec_scaled)

        # State derivative: dx_scaled/dt_scaled = f_scaled(x_scaled)
        du[1:6] = dynamics_scaled(x_vec_scaled, t_scaled)

        # STM derivative: dΦ_scaled/dt_scaled = A_scaled · Φ_scaled
        dΦ_scaled = A_scaled * Φ_scaled
        du[7:42] = vec(dΦ_scaled)
    end

    # Integrate augmented system in scaled coordinates
    prob = ODEProblem(augmented_dynamics_scaled!, augmented_state_0_scaled, (times_scaled[1], times_scaled[end]), sim_params)
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, saveat=times_scaled)

    # Extract STM at each time (in scaled coordinates)
    Φ_traj_scaled = Vector{Matrix{Float64}}(undef, length(times))
    for (i, t_scaled) in enumerate(times_scaled)
        Φ_vec_scaled = sol.u[i][7:42]
        Φ_traj_scaled[i] = reshape(Φ_vec_scaled, 6, 6)
    end

    # Step 5: Propagate covariance in scaled coordinates: 
    # P_scaled(t) = Φ_scaled(t,t₀) · P_scaled(t₀) · Φ_scaled(t,t₀)ᵀ
    P_traj_scaled = Vector{Matrix{Float64}}(undef, length(times))
    P_traj_scaled[1] = P_0_scaled

    for i in 2:length(times)
        P_traj_scaled[i] = Φ_traj_scaled[i] * P_0_scaled * Φ_traj_scaled[i]'
    end

    # Step 6: Unscale covariance: P = S_inv * P_scaled * S_inv'
    P_traj = Vector{Matrix{Float64}}(undef, length(times))
    for i in 1:length(times)
        P_traj[i] = sim_params.S_inv * P_traj_scaled[i] * sim_params.S_inv'
    end

    return x_vec_traj, P_traj
end

function generate_sigma_points_via_unscented_transform(x_vec, P; α=1e-3, β=2.0, κ=0.0)
    # Compute Unscented Transform parameters
    n = length(x_vec)
    λ = α^2 * (n + κ) - n  # Scaling parameter
    γ = sqrt(n + λ)  # Scaling factor for sigma points

    # Generate sigma points using Cholesky decomposition: P = L * L'
    L_chol = cholesky(P).L

    # Generate 2n+1 sigma points
    num_sigma_points = 2 * n + 1
    sigma_points = Vector{Vector{Float64}}(undef, num_sigma_points)

    # First sigma point is the mean
    sigma_points[1] = copy(x_vec)

    # Remaining 2n sigma points: x ± γ * L_i
    # where L_i is the i-th column of the Cholesky factor
    for i in 1:n
        col = L_chol[:, i]
        sigma_points[1+i] = x_vec .+ γ .* col
        sigma_points[n+1+i] = x_vec .- γ .* col
    end

    # Compute weights for mean
    # Mean weights: w_m[1] = λ/(n+λ), w_m[i] = 1/(2(n+λ)) for i > 1
    w_m = zeros(num_sigma_points)
    w_m[1] = λ / (n + λ)
    for i in 2:num_sigma_points
        w_m[i] = 1.0 / (2.0 * (n + λ))
    end

    # Compute weights for covariance
    # Covariance weights: w_c[1] = λ/(n+λ) + (1-α²+β), w_c[i] = 1/(2(n+λ)) for i > 1
    w_c = zeros(num_sigma_points)
    w_c[1] = λ / (n + λ) + (1 - α^2 + β)
    for i in 2:num_sigma_points
        w_c[i] = 1.0 / (2.0 * (n + λ))
    end

    return sigma_points, w_m, w_c
end

function propagate_uncertainty_via_cartesian_unscented_transform(x_vec_0, P_0, times, sim_params; return_sigma_points::Bool=false)
    # Initialize trajectory storage
    x_vec_traj = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))
    x_vec_traj[1] = copy(x_vec_0)
    P_traj[1] = copy(P_0)
    x_vec_current = copy(x_vec_0)
    P_current = copy(P_0)

    # Optional: collect propagated sigma points at every timestep
    if return_sigma_points
        sigma_points_all = Vector{Vector{Vector{Float64}}}(undef, length(times))
        sp_0, _, _ = generate_sigma_points_via_unscented_transform(x_vec_0, P_0)
        sigma_points_all[1] = sp_0  # initial sigma points at t=0
    end

    # Resample and propagate at each timestep
    println("  Propagating via Cartesian UT with resampling at each timestep...")

    for t_idx in 1:(length(times)-1)
        # Generate sigma points from current mean and covariance
        sigma_points, w_m, w_c = generate_sigma_points_via_unscented_transform(x_vec_current, P_current)
        num_sigma_points = length(sigma_points)

        # Propagate each sigma point from times[t_idx] to times[t_idx+1]
        sigma_points_traj = Vector{Vector{Float64}}(undef, num_sigma_points)

        for i in 1:num_sigma_points
            x_vec_traj_sigma, _ = propagate_cartesian_dynamics(sigma_points[i], [times[t_idx], times[t_idx+1]], sim_params)
            sigma_points_traj[i] = x_vec_traj_sigma[end]  # Take the propagated state at times[t_idx+1]
        end

        # Store propagated sigma points (before computing mean/cov)
        if return_sigma_points
            sigma_points_all[t_idx+1] = copy(sigma_points_traj)
        end

        # Compute weighted mean and covariance
        x_vec = zeros(6)
        for i in 1:num_sigma_points
            x_vec .+= w_m[i] .* sigma_points_traj[i]
        end

        P = zeros(6, 6)
        for i in 1:num_sigma_points
            diff = sigma_points_traj[i] .- x_vec
            P .+= w_c[i] .* (diff * diff')
        end

        # Store results and update current state for next iteration
        x_vec_traj[t_idx+1] = x_vec
        P_traj[t_idx+1] = (P + P') / 2.0  # Enforce symmetry
        x_vec_current = x_vec
        P_current = (P + P') / 2.0  # Enforce symmetry
    end

    return return_sigma_points ? (x_vec_traj, P_traj, sigma_points_all) : (x_vec_traj, P_traj)
end

function generate_sigma_points_via_cubature_rule(x_vec, P; use_eigendecomposition=false)
    # Cubature rule (spherical-radial cubature rule)
    # Uses 2n sigma points with equal weights
    # This is a third-degree rule (exact for polynomials up to degree 3)
    n = length(x_vec)  # State dimension
    sqrt_n = sqrt(n)  # Scaling factor for cubature rule

    # Generate 2n sigma points (no center point)
    num_sigma_points = 2 * n
    sigma_points = Vector{Vector{Float64}}(undef, num_sigma_points)

    if use_eigendecomposition
        # Use eigendecomposition: P = Q * Λ * Q'
        eigen_decomp = eigen(P)
        Q = eigen_decomp.vectors  # Eigenvectors (columns)
        λ_eigen = eigen_decomp.values  # Eigenvalues

        # Generate sigma points: x ± sqrt(n) * sqrt(λ_i) * q_i
        # Note: sqrt(n) * sqrt(λ_i) = sqrt(n * λ_i)
        for i in 1:n
            q_i = Q[:, i]  # i-th eigenvector
            sqrt_λ_i = sqrt(λ_eigen[i])
            sigma_points[i] = x_vec .+ sqrt_n .* sqrt_λ_i .* q_i
            sigma_points[n+i] = x_vec .- sqrt_n .* sqrt_λ_i .* q_i
        end
    else
        # Use Cholesky decomposition: P = L * L'
        L_chol = cholesky(P).L

        # Generate sigma points: x ± sqrt(n) * L_i
        # where L_i is the i-th column of the Cholesky factor
        for i in 1:n
            col = L_chol[:, i]
            sigma_points[i] = x_vec .+ sqrt_n .* col
            sigma_points[n+i] = x_vec .- sqrt_n .* col
        end
    end

    # Compute weights (equal for both mean and covariance)
    # Cubature rule uses equal weights: w = 1/(2n)
    w = 1.0 / (2.0 * n)
    w_m = fill(w, num_sigma_points)
    w_c = fill(w, num_sigma_points)

    return sigma_points, w_m, w_c
end

function propagate_uncertainty_via_cartesian_sigma_points(x_vec_0, P_0, times, sim_params; return_sigma_points::Bool=false)
    # Initialize trajectory storage
    x_vec_traj = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))
    x_vec_traj[1] = copy(x_vec_0)
    P_traj[1] = copy(P_0)
    x_vec_current = copy(x_vec_0)
    P_current = copy(P_0)

    # Optional: collect propagated sigma points at every timestep
    if return_sigma_points
        sigma_points_all = Vector{Vector{Vector{Float64}}}(undef, length(times))
        sp_0, _, _ = generate_sigma_points_via_cubature_rule(x_vec_0, P_0)
        sigma_points_all[1] = sp_0
    end

    # Resample and propagate at each timestep
    println("  Propagating via Cartesian CKF with resampling at each timestep...")

    for t_idx in 1:(length(times)-1)
        # Generate sigma points from current mean and covariance
        sigma_points, w_m, w_c = generate_sigma_points_via_cubature_rule(x_vec_current, P_current)
        num_sigma_points = length(sigma_points)

        # Propagate each sigma point from times[t_idx] to times[t_idx+1]
        sigma_points_traj = Vector{Vector{Float64}}(undef, num_sigma_points)

        for i in 1:num_sigma_points
            x_vec_traj_sigma, _ = propagate_cartesian_dynamics(sigma_points[i], [times[t_idx], times[t_idx+1]], sim_params)
            sigma_points_traj[i] = x_vec_traj_sigma[end]  # Take the propagated state at times[t_idx+1]
        end

        # Store propagated sigma points
        if return_sigma_points
            sigma_points_all[t_idx+1] = copy(sigma_points_traj)
        end

        # Compute weighted mean and covariance
        x_vec = zeros(6)
        for i in 1:num_sigma_points
            x_vec .+= w_m[i] .* sigma_points_traj[i]
        end

        P = zeros(6, 6)
        for i in 1:num_sigma_points
            diff = sigma_points_traj[i] .- x_vec
            P .+= w_c[i] .* (diff * diff')
        end

        # Store results and update current state for next iteration
        x_vec_traj[t_idx+1] = x_vec
        P_traj[t_idx+1] = (P + P') / 2.0  # Enforce symmetry
        x_vec_current = x_vec
        P_current = (P + P') / 2.0  # Enforce symmetry
    end

    return return_sigma_points ? (x_vec_traj, P_traj, sigma_points_all) : (x_vec_traj, P_traj)
end

function propagate_uncertainty_via_ks_sigma_points(x_vec_0, P_0, times, sim_params; return_sigma_points::Bool=false)
    # Initialize trajectory storage
    x_vec_traj = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))
    x_vec_traj[1] = copy(x_vec_0)
    P_traj[1] = copy(P_0)
    x_vec_current = copy(x_vec_0)
    P_current = copy(P_0)

    # Optional: collect propagated sigma points at every timestep
    if return_sigma_points
        sigma_points_all = Vector{Vector{Vector{Float64}}}(undef, length(times))
        sp_0, _, _ = generate_sigma_points_via_cubature_rule(x_vec_0, P_0)
        sigma_points_all[1] = sp_0
    end

    # Resample and propagate at each timestep
    println("  Propagating via KS CKF with resampling at each timestep...")

    for t_idx in 1:(length(times)-1)
        # Generate sigma points from current mean and covariance
        sigma_points, w_m, w_c = generate_sigma_points_via_cubature_rule(x_vec_current, P_current)
        num_sigma_points = length(sigma_points)

        # Propagate each sigma point from times[t_idx] to times[t_idx+1]
        sigma_points_traj = Vector{Vector{Float64}}(undef, num_sigma_points)

        for i in 1:num_sigma_points
            x_vec_traj_sigma, _ = propagate_ks_dynamics(sigma_points[i], [0.0, sim_params.sampling_time], sim_params)
            sigma_points_traj[i] = x_vec_traj_sigma[end]  # Take the propagated state at times[t_idx+1]
        end

        # Store propagated sigma points
        if return_sigma_points
            sigma_points_all[t_idx+1] = copy(sigma_points_traj)
        end

        # Compute weighted mean and covariance
        x_vec = zeros(6)
        for i in 1:num_sigma_points
            x_vec .+= w_m[i] .* sigma_points_traj[i]
        end

        P = zeros(6, 6)
        for i in 1:num_sigma_points
            diff = sigma_points_traj[i] .- x_vec
            P .+= w_c[i] .* (diff * diff')
        end

        # Store results and update current state for next iteration
        x_vec_traj[t_idx+1] = x_vec
        P_traj[t_idx+1] = (P + P') / 2.0  # Enforce symmetry
        x_vec_current = x_vec
        P_current = (P + P') / 2.0  # Enforce symmetry
    end

    return return_sigma_points ? (x_vec_traj, P_traj, sigma_points_all) : (x_vec_traj, P_traj)
end

function propagate_uncertainty_via_linearized_ks_sigma_points(x_vec_0, P_0, times, sim_params; return_sigma_points::Bool=false)
    # Initialize trajectory storage
    x_vec_traj = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))
    x_vec_traj[1] = copy(x_vec_0)
    P_traj[1] = copy(P_0)
    x_vec_current = copy(x_vec_0)
    P_current = copy(P_0)

    # Optional: collect propagated sigma points at every timestep
    if return_sigma_points
        sigma_points_all = Vector{Vector{Vector{Float64}}}(undef, length(times))
        sp_0, _, _ = generate_sigma_points_via_cubature_rule(x_vec_0, P_0)
        sigma_points_all[1] = sp_0
    end

    # Resample and propagate at each timestep
    println("  Propagating via KS Relative CKF with resampling at each timestep...")

    for t_idx in 1:(length(times)-1)
        # Generate sigma points from current mean and covariance
        sigma_points, w_m, w_c = generate_sigma_points_via_cubature_rule(x_vec_current, P_current)
        num_sigma_points = length(sigma_points)

        # Propagate each sigma point (deputy) from times[t_idx] to times[t_idx+1] using linearized KS relative dynamics
        # Note: x_vec_current is the chief (mean), sigma_points[i] is the deputy
        sigma_points_traj = Vector{Vector{Float64}}(undef, num_sigma_points)

        for i in 1:num_sigma_points
            x_vec_traj_chief_i, x_vec_traj_rel_i, times_traj_chief_i, times_traj_deputy_i = propagate_ks_relative_dynamics(x_vec_current, sigma_points[i], [0.0, sim_params.sampling_time], sim_params)
            sigma_points_traj[i] = x_vec_traj_chief_i[end] + x_vec_traj_rel_i[end]

            if times_traj_chief_i[end] == 0.0 || times_traj_deputy_i[end] == 0.0
                println("Propagation [", times[t_idx], ", ", times[t_idx+1], "] failed")
                println("  times_traj_chief_i ", times_traj_chief_i, " times_traj_deputy_i ", times_traj_deputy_i)
            end
        end

        # Store propagated sigma points
        if return_sigma_points
            sigma_points_all[t_idx+1] = copy(sigma_points_traj)
        end

        # Compute weighted mean and covariance
        x_vec = zeros(6)
        for i in 1:num_sigma_points
            x_vec .+= w_m[i] .* sigma_points_traj[i]
        end

        P = zeros(6, 6)
        for i in 1:num_sigma_points
            diff = sigma_points_traj[i] .- x_vec
            P .+= w_c[i] .* (diff * diff')
        end

        # Store results and update current state for next iteration
        x_vec_traj[t_idx+1] = x_vec
        P_traj[t_idx+1] = (P + P') / 2.0  # Enforce symmetry
        x_vec_current = x_vec
        P_current = (P + P') / 2.0  # Enforce symmetry
    end

    return return_sigma_points ? (x_vec_traj, P_traj, sigma_points_all) : (x_vec_traj, P_traj)
end

function propagate_uncertainty_via_linearized_ks_dynamics(x_vec_0, P_0, times, sim_params)
    # Step 1: Extract position and velocity and their standard deviations
    r_vec_0 = x_vec_0[1:3]
    v_vec_0 = x_vec_0[4:6]
    σ_pos = sqrt(P_0[1, 1])
    σ_vel = sqrt(P_0[4, 4])

    # Step 2: Compute total orbital energy uncertainty from the Uncertainty Propagation Law
    # h = GM / norm(r) - norm(v)²/2 = f(norm(r), norm(v))
    # σ_h = sqrt( (∂f/∂norm(r))² * σ_norm(r)² + (∂f/∂norm(v))² * σ_norm(v)² )
    # ∂f/∂norm(r) = - GM / norm(r)² and ∂f/∂norm(v) = - norm(v) / norm(r)
    σ_h = sqrt((-sim_params.GM / norm(r_vec_0)^2)^2 * σ_pos^2 + (-norm(v_vec_0))^2 * σ_vel^2)
    println(" Total orbital energy initial state deviation (σ_h): ", σ_h, " m²/s²")
    σ_t = 0.0
    P_0_augmented = [P_0 zeros(6, 2); zeros(1, 6) σ_h^2 0; zeros(1, 7) σ_t^2]

    # Step 2: Scale initial state for numerical stability
    r_vec_0_scaled = r_vec_0 / sim_params.r_scale
    v_vec_0_scaled = v_vec_0 / sim_params.v_scale
    x_vec_0_scaled = [r_vec_0_scaled; v_vec_0_scaled]
    times_scaled = times / sim_params.t_scale

    # Step 3: Solve QP for y_bar and y'_bar as function of r_bar and v_bar
    ks_state_0_scaled = state_cartesian_to_ks(x_vec_0_scaled)
    y_vec_bar_scaled = ks_state_0_scaled[1:4]
    y_vec_prime_bar_scaled = ks_state_0_scaled[5:8]

    # Step 4: Compute Jacobian d([y; y'; h; t])/d([r; v; h; t]) using the KKT conditions
    #=
    function dKS_dCartesian(r_vec, v_vec, y_vec, y_vec_prime)
        Pi = [I(3) zeros(3, 1)]

        A = Pi * L(y_vec)
        dy_dr = 0.5 * A' * inv(A * A')

        B = 2.0 / (y_vec' * y_vec) * Pi * L(y_vec)
        dyp_dv = B' * inv(B * B')

        dyp_dy = 0.5 * R(Pi' * v_vec)
        dyp_dr = dyp_dy * dy_dr

        GM_scaled = sim_params.GM / sim_params.GM_scale
        dh_dr = -GM_scaled * r_vec' / norm(r_vec)^3
        dh_dv = -v_vec'

        # Jacobian d([y; y'; h; t])/d([r; v; h; t]) is 10x8
        # Input: [r; v; h; t] (8 states)
        # Output: [y; y'; h; t] (10 states)
        J = zeros(10, 8)
        J[1:4, 1:3] = dy_dr
        J[5:8, 1:3] = dyp_dr
        J[5:8, 4:6] = dyp_dv
        J[9, 1:3] = dh_dr
        J[9, 4:6] = dh_dv
        J[10, 8] = 1.0  # dt/dt
        return J
    end
    d_yypht_d_rvht = dKS_dCartesian(r_vec_0_scaled, v_vec_0_scaled, y_vec_bar_scaled, y_vec_prime_bar_scaled)
    =#

    function dKS_dCartesian_fd(r_vec, v_vec, h_val, t_val)
        function rvht_to_yypht(x)
            r_local = x[1:3]
            v_local = x[4:6]
            h_local = x[7]
            t_local = x[8]
            ks_state = state_cartesian_to_ks([r_local; v_local])
            y_local = ks_state[1:4]
            yp_local = ks_state[5:8]
            return [y_local; yp_local; h_local; t_local]
        end

        return ForwardDiff.jacobian(rvht_to_yypht, [r_vec; v_vec; h_val; t_val])
    end

    h_bar_scaled = energy_ks(y_vec_bar_scaled, y_vec_prime_bar_scaled, sim_params.GM / sim_params.GM_scale)
    d_yypht_d_rvht = dKS_dCartesian_fd(r_vec_0_scaled, v_vec_0_scaled, h_bar_scaled, times_scaled[1])

    # Step 6: Map covariance from Cartesian to KS space
    # Scale covariance - need to scale h and t appropriately
    # h has units of m²/s², so h_scale = v_scale^2
    h_scale = sim_params.v_scale^2
    # t has units of s, so t_scale is already defined
    S_augmented = [sim_params.S zeros(6, 2);
        zeros(1, 6) 1.0/h_scale 0.0;
        zeros(1, 7) 1.0/sim_params.t_scale]
    P_0_scaled = S_augmented * P_0_augmented * S_augmented'

    # Q_0 = d([y; y'; h; t])/d([r; v; h; t]) * P_scaled * d([y; y'; h; t])/d([r; v; h; t])^T
    # This is the covariance matrix for [y; y'; h; t] (10x10)
    Q_0_scaled = d_yypht_d_rvht * P_0_scaled * d_yypht_d_rvht'

    # Step 7: Compute h_bar and form full KS state z_bar = [y; y'; h; t] for propagation
    # Note: We only propagate covariance Q for [y; y'], but we need h and t in the state for dynamics
    h_bar_scaled = energy_ks(y_vec_bar_scaled, y_vec_prime_bar_scaled, sim_params.GM / sim_params.GM_scale)
    t_bar_scaled_0 = times_scaled[1]
    z_bar_scaled_0 = [y_vec_bar_scaled; y_vec_prime_bar_scaled; h_bar_scaled; t_bar_scaled_0]

    # Step 8: Propagate z_bar using callback (state only, no STM)
    # Pre-allocate trajectory storage for state
    z_bar_traj_scaled = [zeros(10) for k = 1:length(times_scaled)]
    z_bar_traj_scaled[1] .= z_bar_scaled_0
    times_traj_scaled = zeros(length(times_scaled))
    times_traj_scaled[1] = times_scaled[1]
    callback_triggered = falses(length(times_scaled))
    callback_triggered[1] = true  # Initial state is always set

    # Define state-only dynamics function (without STM)
    function ks_state_dynamics_scaled!(du, u, p, s_scaled)
        z_scaled = u  # Full state [y; y'; h; t]

        # State derivative: dz_scaled/ds_scaled = f_ks(z_scaled)
        # Use the actual dynamics function for state propagation
        ks_state_augmented = ks_dynamics(z_scaled, sim_params, s_scaled)
        # Time derivative: dt/ds = r = y'y
        y_vec_current = z_scaled[1:4]
        t_prime_scaled = y_vec_current'y_vec_current
        du .= [ks_state_augmented; t_prime_scaled]
    end

    # Callback to save states at specific real time points
    function condition!(out, u, s_current_scaled, sol)
        t_current_scaled = u[10]  # Real time is the 10th element
        out .= times_scaled .- t_current_scaled
    end

    function affect!(sol, idx)
        if idx <= length(times_scaled)
            z_bar_traj_scaled[idx] .= sol.u
            times_traj_scaled[idx] = sol.u[10]
            callback_triggered[idx] = true
        end
    end

    cb = VectorContinuousCallback(condition!, affect!, length(times_scaled))

    # Estimate fictitious time span
    r_vec_norm_0_scaled = y_vec_bar_scaled'y_vec_bar_scaled
    s_0_scaled = 0.0
    s_end_scaled = (times_scaled[end] - times_scaled[1]) / r_vec_norm_0_scaled

    # Integrate state-only system
    prob_state = ODEProblem(ks_state_dynamics_scaled!, z_bar_scaled_0, (s_0_scaled, s_end_scaled), sim_params)
    sol_state = solve(prob_state, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, callback=cb)

    # Step 8b: Propagate STM separately on s (fictitious time)
    # Initialize STM as identity (10x10)
    Φ_yyp_0_scaled = Matrix{Float64}(I, 10, 10)

    # Combine state and STM into augmented state for STM propagation
    # State: [y; y'; h; t] (10 states)
    # STM: Φ_{[y;y';h;t]} (10x10 = 100 states)
    augmented_state_0_scaled = [z_bar_scaled_0; vec(Φ_yyp_0_scaled)]  # 10 + 100 = 110 states

    # Define augmented dynamics function for STM propagation
    function augmented_ks_dynamics_scaled!(du, u, p, s_scaled)
        z_scaled = u[1:10]  # Full state [y; y'; h; t]
        Φ_vec_scaled = u[11:110]  # STM for [y; y'; h; t] (10x10 = 100 elements)
        Φ_yyp_scaled = reshape(Φ_vec_scaled, 10, 10)  # 10x10 STM

        # Extract current time value (not Dual) before ForwardDiff - similar to ks_dynamics.jl approach
        t_current_scaled_value = z_scaled[10]

        # Compute Jacobian A_yypht_scaled = ∂f_{[y;y';h;t]}/∂[y;y';h;t] at current state (10x10)
        # This is the Jacobian of [y'; y''; h'; t'] with respect to [y; y'; h; t]
        function ks_dyn_full_wrapper(z_local)
            # z_local is [y; y'; h; t] (10 elements)
            y_local = z_local[1:4]
            yp_local = z_local[5:8]
            h_local = z_local[9]
            t_local = z_local[10]

            # Two-body dynamics: y'' = (-h/2) * y
            ypp_local = (-h_local / 2.0) * y_local
            h_prime_local = 0.0

            # J2 + drag if needed
            if sim_params.add_perturbations
                x_vec_local_scaled = state_ks_to_cartesian([y_local; yp_local])
                x_vec_local = [x_vec_local_scaled[1:3] * sim_params.r_scale;
                    x_vec_local_scaled[4:6] * sim_params.v_scale]
                # Use the extracted time value (Float64, not Dual) - matches ks_dynamics.jl pattern
                t_current = t_current_scaled_value * sim_params.t_scale
                epoch = sim_params.epoch_0 + t_current

                a_J2_vec = cartesian_J2_perturbation(x_vec_local, t_current, sim_params)
                a_drag_vec = cartesian_drag_perturbation(x_vec_local, t_current, epoch, sim_params)
                a_pert_vec = a_J2_vec + a_drag_vec
                a_pert_vec_scaled = a_pert_vec / sim_params.a_scale

                ypp_local = ypp_local + (y_local'y_local / 2.0) * (L(y_local)' * [a_pert_vec_scaled; 0.0])
                h_prime_local = h_prime_local - 2 * yp_local' * L(y_local)' * [a_pert_vec_scaled; 0.0]
            end

            # Time derivative: dt/ds = r = y'y
            t_prime_local = y_local'y_local

            return [yp_local; ypp_local; h_prime_local; t_prime_local]  # Return [y'; y''; h'; t']
        end

        # Compute 10x10 Jacobian for [y; y'; h; t]
        A_yypht_scaled = ForwardDiff.jacobian(ks_dyn_full_wrapper, z_scaled)  # 10x10

        # State derivative: dz_scaled/ds_scaled = f_ks(z_scaled)
        # Use the actual dynamics function for state propagation
        ks_state_augmented = ks_dynamics(z_scaled, sim_params, s_scaled)
        # Time derivative: dt/ds = r = y'y
        y_vec_current = z_scaled[1:4]
        t_prime_scaled = y_vec_current'y_vec_current
        du[1:10] = [ks_state_augmented; t_prime_scaled]

        # STM derivative for [y; y'; h; t]: dΦ_{[y;y';h;t]}/ds = A_{[y;y';h;t]} · Φ_{[y;y';h;t]}
        dΦ_yypht_scaled = A_yypht_scaled * Φ_yyp_scaled
        du[11:110] = vec(dΦ_yypht_scaled)
    end

    # Integrate STM on s (fictitious time)
    prob_stm = ODEProblem(augmented_ks_dynamics_scaled!, augmented_state_0_scaled, (s_0_scaled, s_end_scaled), sim_params)
    sol_stm = solve(prob_stm, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol)

    # Match STM points to state points based on closest real time
    # For each z_bar_traj_scaled[i], find sol_stm.u[j] where sol_stm.u[j][10] is closest to times_traj_scaled[i]
    Φ_traj_scaled = [zeros(10, 10) for k = 1:length(times_scaled)]  # 10x10 STM for [y; y'; h; t]
    Φ_traj_scaled[1] .= Φ_yyp_0_scaled

    for i in 2:length(times_scaled)
        # Get real time from saved state trajectory
        t_state = times_traj_scaled[i]

        # Find STM solution point with closest real time
        best_j = 1
        min_time_diff = abs(sol_stm.u[1][10] - t_state)

        for j in 2:length(sol_stm.u)
            t_stm = sol_stm.u[j][10]
            time_diff = abs(t_stm - t_state)
            if time_diff < min_time_diff
                min_time_diff = time_diff
                best_j = j
            end
        end

        # Extract STM from the matched solution point
        Φ_vec_matched = sol_stm.u[best_j][11:110]
        Φ_traj_scaled[i] = reshape(Φ_vec_matched, 10, 10)
    end

    # Step 9: Propagate covariance: Q(s) = Φ_{[y;y';h;t]}(s) · Q_0 · Φ_{[y;y';h;t]}(s)^T
    # Q is 10x10 covariance for [y; y'; h; t]
    Q_traj_scaled = Vector{Matrix{Float64}}(undef, length(times_scaled))
    Q_traj_scaled[1] = Q_0_scaled

    for i in 2:length(times_scaled)
        Q_traj_scaled[i] = Φ_traj_scaled[i] * Q_0_scaled * Φ_traj_scaled[i]'
    end

    # Step 9b: Apply W matrix to "unwarp" time from Q
    # W(bar_z) transforms Δz to Δz̃ where Δt = 0
    # Δz̃ = W(bar_z) Δz, where W = I - (∂z/∂s) * (∂s/∂t) * e_t^T
    # ∂s/∂t = 1/r = 1/(y'y)
    # ∂z/∂s = [y'; y''; h'; t'] = [y'; y''; h'; r]
    Q_traj_unwarped_scaled = Vector{Matrix{Float64}}(undef, length(times_scaled))

    for i in 1:length(times_scaled)
        z_bar_i = z_bar_traj_scaled[i]
        y_vec_i = z_bar_i[1:4]
        yp_vec_i = z_bar_i[5:8]
        h_i = z_bar_i[9]

        # Compute r = y'y
        r_i = y_vec_i'y_vec_i

        # Compute y'' from dynamics
        ypp_vec_i = (-h_i / 2.0) * y_vec_i
        h_prime_i = 0.0

        # Add perturbations if needed
        if sim_params.add_perturbations
            x_vec_i_scaled = state_ks_to_cartesian([y_vec_i; yp_vec_i])
            x_vec_i = [x_vec_i_scaled[1:3] * sim_params.r_scale;
                x_vec_i_scaled[4:6] * sim_params.v_scale]
            t_current = times_traj_scaled[i] * sim_params.t_scale
            epoch = sim_params.epoch_0 + t_current

            a_J2_vec = cartesian_J2_perturbation(x_vec_i, t_current, sim_params)
            a_drag_vec = cartesian_drag_perturbation(x_vec_i, t_current, epoch, sim_params)
            a_pert_vec = a_J2_vec + a_drag_vec
            a_pert_vec_scaled = a_pert_vec / sim_params.a_scale

            ypp_vec_i = ypp_vec_i + (r_i / 2.0) * (L(y_vec_i)' * [a_pert_vec_scaled; 0.0])
            h_prime_i = h_prime_i - 2 * yp_vec_i' * L(y_vec_i)' * [a_pert_vec_scaled; 0.0]
        end

        # ∂z/∂s = [y'; y''; h'; t'] = [y'; y''; h'; r]
        dz_ds = [yp_vec_i; ypp_vec_i; h_prime_i; r_i]

        # ∂s/∂t = 1/r
        ds_dt = 1.0 / r_i  # Avoid division by zero

        # Construct W matrix: W = I - (∂z/∂s) * (∂s/∂t) * e_t^T
        # e_t is the unit vector for the time component (last element)
        e_t = zeros(10)
        e_t[10] = 1.0
        W = Matrix{Float64}(I, 10, 10) - dz_ds * ds_dt * e_t'

        # Apply W to unwarp time: Q̃ = W Q W^T
        Q_traj_unwarped_scaled[i] = W * Q_traj_scaled[i] * W'
    end

    # Step 11: Convert back to Cartesian for output
    x_vec_traj = Vector{Vector{Float64}}(undef, length(times_scaled))
    P_traj = Vector{Matrix{Float64}}(undef, length(times_scaled))

    #=
    function dCartesian_dKS(y_vec, y_vec_prime)
        Pi = [I(3) zeros(3, 1)]
        dr_dy = 2.0 * Pi * L(y_vec)
        dr_dyp = zeros(3, 4)
        dv_dy = -4.0 / (y_vec' * y_vec)^2 * Pi * L(y_vec) * y_vec_prime * y_vec' + 2.0 / (y_vec' * y_vec) * Pi * L(y_vec_prime)
        dv_dyp = 2.0 / (y_vec' * y_vec) * Pi * L(y_vec)
        GM_scaled = sim_params.GM / sim_params.GM_scale
        dh_dy = -2.0 * (GM_scaled - 2.0 * y_vec_prime' * y_vec_prime) * y_vec' / (y_vec' * y_vec)^2
        dh_dyp = -4.0 * y_vec_prime' / (y_vec' * y_vec)

        J = zeros(8, 10)
        J[1:3, 1:4] = dr_dy
        J[1:3, 5:8] = dr_dyp
        J[4:6, 1:4] = dv_dy
        J[4:6, 5:8] = dv_dyp
        J[7, 1:4] = dh_dy
        J[7, 5:8] = dh_dyp
        J[8, 10] = 1.0
        return J
    end
    =#

    function dCartesian_dKS_fd(y_vec, y_vec_prime, h_val, t_val)
        function yypht_to_rvht(z)
            y_local = z[1:4]
            yp_local = z[5:8]
            h_local = z[9]
            t_local = z[10]
            cart_state = state_ks_to_cartesian([y_local; yp_local])
            r_local = cart_state[1:3]
            v_local = cart_state[4:6]
            return [r_local; v_local; h_local; t_local]
        end

        return ForwardDiff.jacobian(yypht_to_rvht, [y_vec; y_vec_prime; h_val; t_val])
    end

    for i in 1:length(times_scaled)
        # Extract KS state
        y_vec_scaled = z_bar_traj_scaled[i][1:4]
        y_vec_prime_scaled = z_bar_traj_scaled[i][5:8]
        h_scaled = z_bar_traj_scaled[i][9]
        t_scaled = z_bar_traj_scaled[i][10]

        # Convert to Cartesian
        x_vec_scaled = state_ks_to_cartesian([y_vec_scaled; y_vec_prime_scaled])
        x_vec_traj[i] = [x_vec_scaled[1:3] * sim_params.r_scale; x_vec_scaled[4:6] * sim_params.v_scale]
        # d_rvht_d_yypht = dCartesian_dKS(y_vec_scaled, y_vec_prime_scaled)
        d_rvht_d_yypht = dCartesian_dKS_fd(y_vec_scaled, y_vec_prime_scaled, h_scaled, t_scaled)

        # Map from Q (which is for [y; y'; h; t]) to augmented Cartesian ([r; v; h; t])
        # Use unwarped covariance Q_traj_unwarped_scaled instead of Q_traj_scaled
        P_scaled = d_rvht_d_yypht * Q_traj_unwarped_scaled[i] * d_rvht_d_yypht'

        # Unscale covariance
        P_traj[i] = sim_params.S_inv * P_scaled[1:6, 1:6] * sim_params.S_inv'
    end

    return x_vec_traj, P_traj
end

function propagate_uncertainty_via_energy_binned_mc_then_ks_sigma_points(x_vec_0, P_0, times, sim_params; num_mc_samples::Int=20000, num_energy_bins::Int=10, drop_edge_bins::Bool=true, verbose::Bool=true, return_samples::Bool=false)
    """
    Rationale:
    - Draw a large set of samples from the initial Cartesian Gaussian (μ₀, P₀).
    - Convert samples to KS coordinates and compute KS energy `h` per sample.
    - Partition samples into uniformly sized energy bins (uniform bin width in `h`); record sample count per bin.
    - For each bin, compute a Cartesian Gaussian approximation (μ₀,k, P₀,k).
    - Propagate each bin independently using `propagate_uncertainty_via_ks_sigma_points`,
    yielding a mean/covariance trajectory per bin (μ_k(t), P_k(t)).
    - At each time step, draw samples from each bin Gaussian in proportion to its bin sample count,
    then aggregate the full sample set to estimate the overall mean/covariance at that time.
    """

    function sample_gaussian(μ::Vector{Float64}, P::Matrix{Float64}, n::Int)
        L = cholesky(P).L
        out = Matrix{Float64}(undef, 6, n)
        for i in 1:n
            out[:, i] .= μ .+ L * randn(6)
        end
        return out
    end

    function mean_and_cov_from_samples(X::Matrix{Float64})
        d, N = size(X)
        μ = vec(sum(X; dims=2) ./ N)
        P = zeros(d, d)
        for i in 1:N
            dx = X[:, i] .- μ
            P .+= dx * dx'
        end
        P ./= (N - 1)
        return μ, (P + P') / 2.0
    end

    if verbose
        println("  Propagating via Energy-Stratified KS CKF")
        println("    num_mc_samples=", num_mc_samples, " num_energy_bins=", num_energy_bins, " drop_edge_bins=", drop_edge_bins)
    end

    # --- Step 1: Monte Carlo samples at t0 ---
    X0 = sample_gaussian(Vector{Float64}(x_vec_0), Matrix{Float64}(P_0), num_mc_samples) # 6 x N

    # --- Step 2-4: compute energies, sort, bin, and compute per-bin μ₀,k, P₀,k ---
    h_vals = Vector{Float64}(undef, num_mc_samples)
    for i in 1:num_mc_samples
        h_vals[i] = energy_ks_from_cartesian(vec(X0[:, i]), sim_params.GM)
    end
    hmin = minimum(h_vals)
    hmax = maximum(h_vals)
    edges = range(hmin, hmax; length=num_energy_bins + 1)
    edges_vec = collect(edges)
    # assign each sample to a bin by energy interval (uniform-width bins in h)
    bin_id = clamp.(searchsortedlast.(Ref(edges_vec), h_vals), 1, num_energy_bins)
    # counts per energy bin (length == num_energy_bins; zeros allowed)
    bin_counts = [count(==(k), bin_id) for k in 1:num_energy_bins]

    # thresholds are just the internal edges (for printing/debugging)
    thresholds = Float64.(edges_vec[2:end-1])

    # Optionally drop the leftmost/rightmost energy bins
    active_bins = collect(1:num_energy_bins)
    if drop_edge_bins && num_energy_bins >= 2
        active_bins = collect(2:(num_energy_bins-1))
    end

    # Drop bins with 6 or fewer samples, ensuring symmetric removal from both sides
    min_samples_threshold = 6
    bins_with_few_samples = [k for k in active_bins if bin_counts[k] <= min_samples_threshold]

    if !isempty(bins_with_few_samples)
        # Count consecutive bins with few samples from left side (within active_bins)
        left_drop_count = 0
        for k in active_bins
            if bin_counts[k] <= min_samples_threshold
                left_drop_count += 1
            else
                break
            end
        end

        # Count consecutive bins with few samples from right side (within active_bins)
        right_drop_count = 0
        for k in reverse(active_bins)
            if bin_counts[k] <= min_samples_threshold
                right_drop_count += 1
            else
                break
            end
        end

        # Drop the same number from both sides to avoid bias
        symmetric_drop_count = max(left_drop_count, right_drop_count)

        # Update active_bins: remove symmetric drops from edges
        if symmetric_drop_count > 0 && length(active_bins) >= 2 * symmetric_drop_count
            # Remove from left (first symmetric_drop_count bins in active_bins)
            left_dropped = active_bins[1:symmetric_drop_count]
            # Remove from right (last symmetric_drop_count bins in active_bins)
            right_dropped = active_bins[(end-symmetric_drop_count+1):end]
            symmetric_dropped = union(left_dropped, right_dropped)
            active_bins = setdiff(active_bins, symmetric_dropped)

            # Also drop any remaining bins with few samples that weren't already dropped
            remaining_few_samples = setdiff(bins_with_few_samples, symmetric_dropped)
            active_bins = setdiff(active_bins, remaining_few_samples)
        else
            # Not enough bins for symmetric drop, or no symmetric drops needed
            # Just drop bins with few samples
            active_bins = setdiff(active_bins, bins_with_few_samples)
        end
    end

    dropped_bins = setdiff(collect(1:num_energy_bins), active_bins)

    μ0_bins = Vector{Vector{Float64}}(undef, num_energy_bins)
    P0_bins = Vector{Matrix{Float64}}(undef, num_energy_bins)
    h_ranges = Vector{Tuple{Float64,Float64}}(undef, num_energy_bins)

    for k in 1:num_energy_bins
        n_k = bin_counts[k]
        if n_k == 0
            μ0_bins[k] = zeros(6)
            P0_bins[k] = zeros(6, 6)
            h_ranges[k] = (NaN, NaN)
            continue
        end

        idxs = findall(==(k), bin_id)
        Xk = Matrix{Float64}(X0[:, idxs])
        μk, Pk = mean_and_cov_from_samples(Xk)
        μ0_bins[k] = μk
        P0_bins[k] = Pk

        hk = h_vals[idxs]
        h_ranges[k] = (minimum(hk), maximum(hk))
    end

    if verbose
        println("    energy thresholds (", length(thresholds), "): ", thresholds)
        for k in 1:num_energy_bins
            println("    bin ", k, ": n=", bin_counts[k], " h∈[", h_ranges[k][1], ", ", h_ranges[k][2], "]")
        end
        if !isempty(dropped_bins)
            dropped_edge = Int[]
            dropped_few_samples = Int[]
            if drop_edge_bins && num_energy_bins >= 2
                dropped_edge = [1, num_energy_bins]
            end
            min_samples_threshold = 6
            bins_with_few_samples_all = [k for k in 1:num_energy_bins if bin_counts[k] <= min_samples_threshold]
            dropped_few_samples = intersect(dropped_bins, bins_with_few_samples_all)
            if !isempty(dropped_edge) && any(k -> k in dropped_bins, dropped_edge)
                println("    dropped edge bins: ", dropped_edge, " counts=", bin_counts[dropped_edge])
            end
            if !isempty(dropped_few_samples)
                println("    dropped bins with ≤$(min_samples_threshold) samples: ", dropped_few_samples, " counts=", bin_counts[dropped_few_samples])
            end
        end
    end

    if verbose
        println("    Per-bin initial covariance diagnostics (P0_bins):")
        for k in 1:num_energy_bins
            if bin_counts[k] == 0
                println("      bin ", k, ": EMPTY")
                continue
            end
            Pk = (P0_bins[k] + P0_bins[k]') / 2.0
            d = diag(Pk)
            λmin = minimum(eigvals(Symmetric(Pk)))
            λmax = maximum(eigvals(Symmetric(Pk)))
            println("      bin ", k, ": n=", bin_counts[k],
                " diag[min,max]=[", minimum(d), ", ", maximum(d), "]",
                " eig[min,max]=[", λmin, ", ", λmax, "]",
                " sym_err=", norm(P0_bins[k] - P0_bins[k]'))
        end
    end

    # --- Step 5: propagate each active bin independently ---
    x_traj_bins = Vector{Vector{Vector{Float64}}}(undef, num_energy_bins)
    P_traj_bins = Vector{Vector{Matrix{Float64}}}(undef, num_energy_bins)
    for k in 1:num_energy_bins
        if !(k in active_bins)
            x_traj_bins[k] = Vector{Vector{Float64}}()
            P_traj_bins[k] = Vector{Matrix{Float64}}()
            continue
        end
        if verbose
            println("    propagating bin ", k, "/", num_energy_bins, " (n=", bin_counts[k], ", h∈[", h_ranges[k][1], ", ", h_ranges[k][2], "]) ...")
        end
        x_traj_bins[k], P_traj_bins[k] = propagate_uncertainty_via_ks_sigma_points(μ0_bins[k], P0_bins[k], times, sim_params)
    end

    # --- Step 6: aggregate at each time by sampling proportional to bin counts ---
    # Aggregate only the number of samples that were propagated
    total_n = sum(bin_counts[active_bins])
    if total_n == 0
        error("All bins were dropped or empty; cannot aggregate (total_n=0).")
    end
    weights = zeros(num_energy_bins)
    for k in active_bins
        weights[k] = bin_counts[k] / total_n
    end
    total_draw = total_n
    n_draw = [round(Int, w * total_draw) for w in weights]
    while sum(n_draw) < total_draw
        n_draw[argmax(weights)] += 1
    end
    while sum(n_draw) > total_draw
        n_draw[argmax(n_draw)] -= 1
    end

    if verbose
        println("    aggregation draw counts per bin: ", n_draw, " (sum=", sum(n_draw), ", total_draw=", total_draw, ")")
    end

    T = length(times)
    x_vec_traj = Vector{Vector{Float64}}(undef, T)
    P_traj = Vector{Matrix{Float64}}(undef, T)

    # Optional: collect aggregated samples at every timestep
    if return_samples
        samples_all = Vector{Vector{Vector{Float64}}}(undef, T)
    end

    for t_idx in 1:T
        Xagg = Matrix{Float64}(undef, 6, total_draw)
        col = 0
        for k in 1:num_energy_bins
            nk = n_draw[k]
            if nk <= 0
                continue
            end
            if !(k in active_bins)
                continue
            end
            μk = Vector{Float64}(x_traj_bins[k][t_idx])
            Pk = Matrix{Float64}(P_traj_bins[k][t_idx])
            Xk = sample_gaussian(μk, Pk, nk)
            Xagg[:, (col+1):(col+nk)] .= Xk
            col += nk
        end

        μ, P = mean_and_cov_from_samples(Xagg)
        x_vec_traj[t_idx] = μ
        P_traj[t_idx] = P

        # Store aggregated samples as vector of state vectors
        if return_samples
            samples_all[t_idx] = [Xagg[:, j] for j in 1:total_draw]
        end
    end

    return return_samples ? (x_vec_traj, P_traj, samples_all) : (x_vec_traj, P_traj)
end

function error_metrics(x_vec_traj_ref, P_traj_ref, x_vec_traj_test, P_traj_test)
    # Ensure trajectories have the same length
    N = min(length(x_vec_traj_ref), length(x_vec_traj_test), length(P_traj_ref), length(P_traj_test))

    # Compute position and velocity errors
    pos_errors = zeros(N)
    vel_errors = zeros(N)

    for i in 1:N
        # Position error (Euclidean norm)
        pos_errors[i] = norm(x_vec_traj_test[i][1:3] - x_vec_traj_ref[i][1:3])
        # Velocity error (Euclidean norm)
        vel_errors[i] = norm(x_vec_traj_test[i][4:6] - x_vec_traj_ref[i][4:6])
    end

    # Position metrics
    pos_rmse = sqrt(sum(pos_errors .^ 2) / N)
    pos_min = minimum(pos_errors)
    pos_max = maximum(pos_errors)

    # Velocity metrics
    vel_rmse = sqrt(sum(vel_errors .^ 2) / N)
    vel_min = minimum(vel_errors)
    vel_max = maximum(vel_errors)

    # Extract standard deviations from covariance matrices
    # Position standard deviations: sqrt(P[i,i]) for i = 1, 2, 3
    # Velocity standard deviations: sqrt(P[i,i]) for i = 4, 5, 6
    σ_pos_ref = zeros(N, 3)  # [σ_x, σ_y, σ_z] for each time step
    σ_pos_test = zeros(N, 3)
    σ_vel_ref = zeros(N, 3)  # [σ_vx, σ_vy, σ_vz] for each time step
    σ_vel_test = zeros(N, 3)

    # Total position and velocity uncertainties (sqrt of trace of covariance submatrix)
    σ_pos_total_ref = zeros(N)
    σ_pos_total_test = zeros(N)
    σ_vel_total_ref = zeros(N)
    σ_vel_total_test = zeros(N)

    for i in 1:N
        # Position standard deviations (component-wise)
        σ_pos_ref[i, 1] = sqrt(max(0.0, P_traj_ref[i][1, 1]))  # σ_x
        σ_pos_ref[i, 2] = sqrt(max(0.0, P_traj_ref[i][2, 2]))  # σ_y
        σ_pos_ref[i, 3] = sqrt(max(0.0, P_traj_ref[i][3, 3]))  # σ_z

        σ_pos_test[i, 1] = sqrt(max(0.0, P_traj_test[i][1, 1]))  # σ_x
        σ_pos_test[i, 2] = sqrt(max(0.0, P_traj_test[i][2, 2]))  # σ_y
        σ_pos_test[i, 3] = sqrt(max(0.0, P_traj_test[i][3, 3]))  # σ_z

        # Total position uncertainty: sqrt(trace of position covariance submatrix)
        σ_pos_total_ref[i] = sqrt(max(0.0, P_traj_ref[i][1, 1] + P_traj_ref[i][2, 2] + P_traj_ref[i][3, 3]))
        σ_pos_total_test[i] = sqrt(max(0.0, P_traj_test[i][1, 1] + P_traj_test[i][2, 2] + P_traj_test[i][3, 3]))

        # Velocity standard deviations (component-wise)
        σ_vel_ref[i, 1] = sqrt(max(0.0, P_traj_ref[i][4, 4]))  # σ_vx
        σ_vel_ref[i, 2] = sqrt(max(0.0, P_traj_ref[i][5, 5]))  # σ_vy
        σ_vel_ref[i, 3] = sqrt(max(0.0, P_traj_ref[i][6, 6]))  # σ_vz

        σ_vel_test[i, 1] = sqrt(max(0.0, P_traj_test[i][4, 4]))  # σ_vx
        σ_vel_test[i, 2] = sqrt(max(0.0, P_traj_test[i][5, 5]))  # σ_vy
        σ_vel_test[i, 3] = sqrt(max(0.0, P_traj_test[i][6, 6]))  # σ_vz

        # Total velocity uncertainty: sqrt(trace of velocity covariance submatrix)
        σ_vel_total_ref[i] = sqrt(max(0.0, P_traj_ref[i][4, 4] + P_traj_ref[i][5, 5] + P_traj_ref[i][6, 6]))
        σ_vel_total_test[i] = sqrt(max(0.0, P_traj_test[i][4, 4] + P_traj_test[i][5, 5] + P_traj_test[i][6, 6]))
    end

    # Compute position uncertainty error trajectory
    pos_uncertainty_errors = [abs(σ_pos_total_test[i] - σ_pos_total_ref[i]) for i in 1:N]
    pos_uncertainty_rmse = sqrt(sum(pos_uncertainty_errors .^ 2) / N)
    pos_uncertainty_min = minimum(pos_uncertainty_errors)
    pos_uncertainty_max = maximum(pos_uncertainty_errors)

    # Compute velocity uncertainty error trajectory
    vel_uncertainty_errors = [abs(σ_vel_total_test[i] - σ_vel_total_ref[i]) for i in 1:N]
    vel_uncertainty_rmse = sqrt(sum(vel_uncertainty_errors .^ 2) / N)
    vel_uncertainty_min = minimum(vel_uncertainty_errors)
    vel_uncertainty_max = maximum(vel_uncertainty_errors)

    return (
        # Position error trajectory and statistics
        pos_errors=pos_errors,
        pos_rmse=pos_rmse,
        pos_min=pos_min,
        pos_max=pos_max,
        # Velocity error trajectory and statistics
        vel_errors=vel_errors,
        vel_rmse=vel_rmse,
        vel_min=vel_min,
        vel_max=vel_max,
        # Position uncertainty error trajectory and statistics
        pos_uncertainty_errors=pos_uncertainty_errors,
        pos_uncertainty_rmse=pos_uncertainty_rmse,
        pos_uncertainty_min=pos_uncertainty_min,
        pos_uncertainty_max=pos_uncertainty_max,
        # Velocity uncertainty error trajectory and statistics
        vel_uncertainty_errors=vel_uncertainty_errors,
        vel_uncertainty_rmse=vel_uncertainty_rmse,
        vel_uncertainty_min=vel_uncertainty_min,
        vel_uncertainty_max=vel_uncertainty_max,
    )
end

function gaussian_kl_divergence(name, t, x_vec_ref, P_ref, x_vec_test, P_test; use_jitter::Bool=false)
    # KL divergence between two multivariate Gaussians (reference vs test), i.e.:
    #   D_KL( N(μ0, Σ0) || N(μ1, Σ1) )
    #
    # D_KL = 0.5 * ( tr(Σ1^{-1} Σ0) + (μ1-μ0)' Σ1^{-1} (μ1-μ0) - k + log(det(Σ1)/det(Σ0)) )
    #
    # Notes:
    # - Covariances coming from numerical propagation can become slightly non-symmetric / indefinite.
    # - We symmetrize and add adaptive diagonal jitter before factoring.
    μ0 = x_vec_ref
    Σ0 = P_ref
    μ1 = x_vec_test
    Σ1 = P_test

    k = length(μ0)

    # Basic shape checks (fail fast with NaN instead of throwing in plotting code)
    if length(μ1) != k || size(Σ0, 1) != k || size(Σ0, 2) != k || size(Σ1, 1) != k || size(Σ1, 2) != k
        println("  ERROR: Gaussian KL divergence: dimension mismatch. Method: ", name, " at time ", t, " s")
        return NaN
    end

    # Symmetrize to reduce numerical asymmetry
    Σ0s = 0.5 * (Σ0 + Σ0')
    Σ1s = 0.5 * (Σ1 + Σ1')

    # Helper: attempt Cholesky on a symmetrized covariance with increasing jitter.
    # Returns (chol, Σjittered, used_jitter)
    function _chol_with_jitter(Σ::AbstractMatrix{<:Real})
        # Scale jitter based on average diagonal magnitude; keep non-negative baseline
        diag_mean = sum(abs, diag(Σ)) / max(1, size(Σ, 1))
        base = max(diag_mean, 1.0)
        jitter = 0.0
        Σj = Matrix{Float64}(Σ)
        I_k = Matrix{Float64}(I, size(Σ, 1), size(Σ, 2))

        # Try without jitter first; if allowed, grow jitter geometrically
        alphas = use_jitter ?
                 (0.0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0) :
                 (0.0,)

        for α in alphas
            jitter = α * base
            Σj = Matrix{Float64}(Σ) .+ jitter .* I_k
            try
                return cholesky(Symmetric(Σj)), Σj, jitter
            catch
                # keep trying
            end
        end
        return nothing, Σj, jitter
    end

    # Factor both covariances (Σ0 and Σ1 must be SPD for KL to be finite)
    chol0, Σ0j, jitter0 = _chol_with_jitter(Σ0s)
    chol1, Σ1j, jitter1 = _chol_with_jitter(Σ1s)

    if chol0 === nothing || chol1 === nothing
        println("  ERROR: Gaussian KL divergence: cholesky decomposition failed. Method: ", name, " at time ", t, " s")
        if chol0 === nothing
            println("  Σ0 (ref) jitter tried: ", jitter0)
            println("  P0s: ", Σ0s)
        end
        if chol1 === nothing
            println("  Σ1 (test) jitter tried: ", jitter1)
            println("  P1s: ", Σ1s)
        end
        return NaN
    end

    # Verbose: report when jitter was required to make covariances SPD
    if use_jitter && (jitter0 > 0.0 || jitter1 > 0.0)
        println("  INFO: Gaussian KL divergence: applied jitter. Method: ", name, " at time ", t, " s",
            " | jitter_ref=", jitter0, " | jitter_test=", jitter1)
    end

    # log(det(Σ)) from Cholesky: det(Σ) = prod(diag(L))^2
    logdet0 = 2.0 * sum(log, diag(chol0.L))
    logdet1 = 2.0 * sum(log, diag(chol1.L))

    # Compute tr(Σ1^{-1} Σ0) via solving Σ1 X = Σ0, then taking tr(X)
    # Use the Cholesky factorization for stable solves.
    X = chol1 \ Σ0j
    tr_term = tr(X)

    # Quadratic term: (μ1-μ0)' Σ1^{-1} (μ1-μ0)
    δ = Vector{Float64}(μ1 .- μ0)
    quad_term = dot(δ, chol1 \ δ)

    kl = 0.5 * (tr_term + quad_term - k + (logdet1 - logdet0))

    # Numerical guard: KL should be >= 0, but tiny negatives can happen from rounding.
    return max(0.0, kl)
end