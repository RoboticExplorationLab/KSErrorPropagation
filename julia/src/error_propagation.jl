using Random
using LinearAlgebra
using ForwardDiff
using DifferentialEquations

Random.seed!(1234)

function propagate_uncertainty_via_monte_carlo(x_vec_0, P_0, times, sim_params, num_samples=10000)
    # Scale initial state and covariance for numerical stability
    x_vec_0_scaled = [x_vec_0[1:3] / sim_params.r_scale; x_vec_0[4:6] / sim_params.v_scale]
    P_0_scaled = sim_params.S * P_0 * sim_params.S'

    # Sample initial states from multivariate Gaussian in scaled coordinates
    # Use Cholesky decomposition: if x ~ N(0, I), then L_chol*x + μ ~ N(μ, P) where P = L_chol*L_chol'
    L_chol_scaled = cholesky(P_0_scaled).L

    # Generate samples in scaled coordinates
    samples_0_scaled = Vector{Vector{Float64}}(undef, num_samples)
    for i in 1:num_samples
        z = randn(6)  # Standard normal
        samples_0_scaled[i] = x_vec_0_scaled .+ L_chol_scaled * z
    end

    # Unscale samples for propagation (propagate_cartesian_dynamics expects unscaled input)
    samples_0 = [sim_params.S_inv * s for s in samples_0_scaled]

    # Pre-allocate trajectory storage (in scaled coordinates for numerical stability)
    samples_propagated_scaled = Vector{Vector{Vector{Float64}}}(undef, length(times))
    for t_idx in 1:length(times)
        samples_propagated_scaled[t_idx] = Vector{Vector{Float64}}(undef, num_samples)
    end

    # Propagate each sample
    println("  Propagating ", num_samples, " Monte Carlo samples...")
    for (i, sample_0) in enumerate(samples_0)
        if i % 1000 == 0
            println("    Sample ", i, " / ", num_samples)
        end

        # Propagate sample through full nonlinear dynamics (returns unscaled)
        x_vec_traj_sample, _ = propagate_cartesian_dynamics(sample_0, times, sim_params)

        # Scale propagated states for numerical stability in covariance computation
        for t_idx in 1:length(times)
            samples_propagated_scaled[t_idx][i] = sim_params.S * x_vec_traj_sample[t_idx]
        end
    end

    # Pre-allocate mean and covariance storage
    x_vec_traj_mean = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))

    # Compute mean and covariance at each time (in scaled coordinates, then unscale)
    for t_idx in 1:length(times)
        # Compute mean in scaled coordinates
        x_mean_scaled = zeros(6)
        for i in 1:num_samples
            x_mean_scaled .+= samples_propagated_scaled[t_idx][i]
        end
        x_mean_scaled ./= num_samples

        # Unscale mean
        x_vec_traj_mean[t_idx] = sim_params.S_inv * x_mean_scaled

        # Compute covariance in scaled coordinates (better numerical stability)
        P_scaled = zeros(6, 6)
        for i in 1:num_samples
            diff_scaled = samples_propagated_scaled[t_idx][i] .- x_mean_scaled
            P_scaled .+= diff_scaled * diff_scaled'
        end
        P_scaled ./= (num_samples - 1)  # Sample covariance (Bessel's correction)

        # Unscale covariance: P = S_inv * P_scaled * S_inv'
        P_traj[t_idx] = sim_params.S_inv * P_scaled * sim_params.S_inv'
    end

    return x_vec_traj_mean, P_traj
end

function propagate_uncertainty_via_linearized_cartesian_dynamics(x_vec_0, P_0, times, sim_params)
    # Step 1: Scale initial state and covariance for numerical stability
    x_vec_0_scaled = [x_vec_0[1:3] / sim_params.r_scale; x_vec_0[4:6] / sim_params.v_scale]
    times_scaled = times / sim_params.t_scale
    P_0_scaled = sim_params.S * P_0 * sim_params.S'

    # Step 2: Propagate mean trajectory using full nonlinear dynamics
    x_vec_traj_mean, _ = propagate_cartesian_dynamics(x_vec_0, times, sim_params)

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

    return x_vec_traj_mean, P_traj
end

function propagate_uncertainty_via_cartesian_unscented_transform(x_vec_0, P_0, times, sim_params; α=1e-3, β=2.0, κ=0.0)
    # Scale initial state and covariance for numerical stability
    x_vec_0_scaled = [x_vec_0[1:3] / sim_params.r_scale; x_vec_0[4:6] / sim_params.v_scale]
    P_0_scaled = sim_params.S * P_0 * sim_params.S'

    # Compute Unscented Transform parameters
    n = 6  # State dimension
    λ = α^2 * (n + κ) - n  # Scaling parameter
    γ = sqrt(n + λ)  # Scaling factor for sigma points

    # Generate sigma points
    # Use Cholesky decomposition: P = L * L'
    L_chol_scaled = cholesky(P_0_scaled).L

    # Generate 2n+1 sigma points
    num_sigma_points = 2 * n + 1
    sigma_points_0_scaled = Vector{Vector{Float64}}(undef, num_sigma_points)

    # First sigma point is the mean
    sigma_points_0_scaled[1] = copy(x_vec_0_scaled)

    # Remaining 2n sigma points: x ± γ * L_i
    for i in 1:n
        col = L_chol_scaled[:, i]
        sigma_points_0_scaled[1+i] = x_vec_0_scaled .+ γ .* col
        sigma_points_0_scaled[n+1+i] = x_vec_0_scaled .- γ .* col
    end

    # Compute weights
    # Mean weights
    w_m = zeros(num_sigma_points)
    w_m[1] = λ / (n + λ)
    for i in 2:num_sigma_points
        w_m[i] = 1.0 / (2.0 * (n + λ))
    end

    # Covariance weights
    w_c = zeros(num_sigma_points)
    w_c[1] = λ / (n + λ) + (1 - α^2 + β)
    for i in 2:num_sigma_points
        w_c[i] = 1.0 / (2.0 * (n + λ))
    end

    # Unscale sigma points for propagation (propagate_cartesian_dynamics expects unscaled input)
    sigma_points_0 = [sim_params.S_inv * s for s in sigma_points_0_scaled]

    # Propagate each sigma point through full nonlinear dynamics
    println("  Propagating ", num_sigma_points, " sigma points via Unscented Transform...")
    sigma_points_propagated = Vector{Vector{Vector{Float64}}}(undef, num_sigma_points)

    for (i, sigma_point_0) in enumerate(sigma_points_0)
        # Propagate sigma point through full nonlinear dynamics (returns unscaled)
        x_vec_traj_sigma, _ = propagate_cartesian_dynamics(sigma_point_0, times, sim_params)
        sigma_points_propagated[i] = x_vec_traj_sigma
    end

    # Compute weighted mean and covariance at each time
    x_vec_traj_mean = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))

    for t_idx in 1:length(times)
        # Scale propagated sigma points for numerical stability in covariance computation
        sigma_points_t_scaled = [sim_params.S * sigma_points_propagated[i][t_idx] for i in 1:num_sigma_points]

        # Compute weighted mean in scaled coordinates
        x_mean_scaled = zeros(6)
        for i in 1:num_sigma_points
            x_mean_scaled .+= w_m[i] .* sigma_points_t_scaled[i]
        end

        # Unscale mean
        x_vec_traj_mean[t_idx] = sim_params.S_inv * x_mean_scaled

        # Compute weighted covariance in scaled coordinates
        P_scaled = zeros(6, 6)
        for i in 1:num_sigma_points
            diff_scaled = sigma_points_t_scaled[i] .- x_mean_scaled
            P_scaled .+= w_c[i] .* (diff_scaled * diff_scaled')
        end

        # Unscale covariance: P = S_inv * P_scaled * S_inv'
        P_traj[t_idx] = sim_params.S_inv * P_scaled * sim_params.S_inv'
    end

    return x_vec_traj_mean, P_traj
end

function propagate_uncertainty_via_cartesian_sigma_points(x_vec_0, P_0, times, sim_params)
    # Scale initial state and covariance for numerical stability
    x_vec_0_scaled = [x_vec_0[1:3] / sim_params.r_scale; x_vec_0[4:6] / sim_params.v_scale]
    P_0_scaled = sim_params.S * P_0 * sim_params.S'

    # Step 1: Compute eigenvalue decomposition: P = Q * Λ * Q'
    eigen_decomp = eigen(P_0_scaled)
    Q = eigen_decomp.vectors  # Eigenvectors (columns)
    λ_eigen = eigen_decomp.values  # Eigenvalues

    # Step 2: Generate sigma points using eigen-based method
    n = 6  # State dimension
    num_sigma_points = 2 * n
    sigma_points_0_scaled = Vector{Vector{Float64}}(undef, num_sigma_points)

    # Generate 2n sigma points: x ± sqrt(n * λ_i) * q_i
    # Using sqrt(n * λ_i) allows us to use equal weights w = 1/(2n) for both mean and covariance
    for i in 1:n
        q_i = Q[:, i]  # i-th eigenvector
        sqrt_n_λ_i = sqrt(n * λ_eigen[i])  # sqrt of n * i-th eigenvalue
        sigma_points_0_scaled[i] = x_vec_0_scaled .+ sqrt_n_λ_i .* q_i
        sigma_points_0_scaled[n+i] = x_vec_0_scaled .- sqrt_n_λ_i .* q_i
    end

    # Step 3: Compute weights for exact mean and covariance matching
    w = 1.0 / (2 * n)  # Equal weights for both mean and covariance
    w_m = fill(w, num_sigma_points)
    w_c = fill(w, num_sigma_points)

    # Step 4: Unscale sigma points for propagation (propagate_cartesian_dynamics expects unscaled input)
    sigma_points_0 = [sim_params.S_inv * s for s in sigma_points_0_scaled]

    # Step 5: Propagate each sigma point through full nonlinear dynamics
    println("  Propagating ", num_sigma_points, " sigma points via Eigen-based method...")
    sigma_points_propagated = Vector{Vector{Vector{Float64}}}(undef, num_sigma_points)

    for (i, sigma_point_0) in enumerate(sigma_points_0)
        # Propagate sigma point through full nonlinear dynamics (returns unscaled)
        x_vec_traj_sigma, _ = propagate_cartesian_dynamics(sigma_point_0, times, sim_params)
        sigma_points_propagated[i] = x_vec_traj_sigma
    end

    # Step 6: Compute weighted mean and covariance at each time
    x_vec_traj_mean = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))

    for t_idx in 1:length(times)
        # Scale propagated sigma points for numerical stability in covariance computation
        sigma_points_t_scaled = [sim_params.S * sigma_points_propagated[i][t_idx] for i in 1:num_sigma_points]

        # Compute weighted mean in scaled coordinates
        x_mean_scaled = zeros(6)
        for i in 1:num_sigma_points
            x_mean_scaled .+= w_m[i] .* sigma_points_t_scaled[i]
        end

        # Unscale mean
        x_vec_traj_mean[t_idx] = sim_params.S_inv * x_mean_scaled

        # Compute weighted covariance in scaled coordinates
        P_scaled = zeros(6, 6)
        for i in 1:num_sigma_points
            diff_scaled = sigma_points_t_scaled[i] .- x_mean_scaled
            P_scaled .+= w_c[i] .* (diff_scaled * diff_scaled')
        end

        # Unscale covariance: P = S_inv * P_scaled * S_inv'
        P_traj[t_idx] = sim_params.S_inv * P_scaled * sim_params.S_inv'
    end

    return x_vec_traj_mean, P_traj
end

function propagate_uncertainty_via_ks_sigma_points(x_vec_0, P_0, times, sim_params)
    # Scale initial state and covariance for numerical stability
    x_vec_0_scaled = [x_vec_0[1:3] / sim_params.r_scale; x_vec_0[4:6] / sim_params.v_scale]
    P_0_scaled = sim_params.S * P_0 * sim_params.S'

    # Step 1: Compute eigenvalue decomposition: P = Q * Λ * Q'
    eigen_decomp = eigen(P_0_scaled)
    Q = eigen_decomp.vectors  # Eigenvectors (columns)
    λ_eigen = eigen_decomp.values  # Eigenvalues

    # Step 2: Generate sigma points using eigen-based method
    n = 6  # State dimension
    num_sigma_points = 2 * n
    sigma_points_0_scaled = Vector{Vector{Float64}}(undef, num_sigma_points)

    # Generate 2n sigma points: x ± sqrt(n * λ_i) * q_i
    # Using sqrt(n * λ_i) allows us to use equal weights w = 1/(2n) for both mean and covariance
    for i in 1:n
        q_i = Q[:, i]  # i-th eigenvector
        sqrt_n_λ_i = sqrt(n * λ_eigen[i])  # sqrt of n * i-th eigenvalue
        sigma_points_0_scaled[i] = x_vec_0_scaled .+ sqrt_n_λ_i .* q_i
        sigma_points_0_scaled[n+i] = x_vec_0_scaled .- sqrt_n_λ_i .* q_i
    end

    # Step 3: Compute weights for exact mean and covariance matching
    w = 1.0 / (2 * n)  # Equal weights for both mean and covariance
    w_m = fill(w, num_sigma_points)
    w_c = fill(w, num_sigma_points)

    # Step 4: Unscale sigma points for propagation (propagate_ks_dynamics expects unscaled input)
    sigma_points_0 = [sim_params.S_inv * s for s in sigma_points_0_scaled]

    # Step 5: Propagate each sigma point through KS dynamics
    println("  Propagating ", num_sigma_points, " sigma points via Eigen-based method (KS dynamics)...")
    sigma_points_propagated = Vector{Vector{Vector{Float64}}}(undef, num_sigma_points)

    for (i, sigma_point_0) in enumerate(sigma_points_0)
        # Propagate sigma point through KS dynamics (returns unscaled Cartesian)
        x_vec_traj_sigma, _ = propagate_ks_dynamics(sigma_point_0, times, sim_params)
        sigma_points_propagated[i] = x_vec_traj_sigma
    end

    # Step 6: Compute weighted mean and covariance at each time
    x_vec_traj_mean = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))

    for t_idx in 1:length(times)
        # Scale propagated sigma points for numerical stability in covariance computation
        sigma_points_t_scaled = [sim_params.S * sigma_points_propagated[i][t_idx] for i in 1:num_sigma_points]

        # Compute weighted mean in scaled coordinates
        x_mean_scaled = zeros(6)
        for i in 1:num_sigma_points
            x_mean_scaled .+= w_m[i] .* sigma_points_t_scaled[i]
        end

        # Unscale mean
        x_vec_traj_mean[t_idx] = sim_params.S_inv * x_mean_scaled

        # Compute weighted covariance in scaled coordinates
        P_scaled = zeros(6, 6)
        for i in 1:num_sigma_points
            diff_scaled = sigma_points_t_scaled[i] .- x_mean_scaled
            P_scaled .+= w_c[i] .* (diff_scaled * diff_scaled')
        end

        # Unscale covariance: P = S_inv * P_scaled * S_inv'
        P_traj[t_idx] = sim_params.S_inv * P_scaled * sim_params.S_inv'
    end

    return x_vec_traj_mean, P_traj
end

function propagate_uncertainty_via_linearized_ks_sigma_points(x_vec_0, P_0, times, sim_params)
    # Scale initial state and covariance for numerical stability
    x_vec_0_scaled = [x_vec_0[1:3] / sim_params.r_scale; x_vec_0[4:6] / sim_params.v_scale]
    P_0_scaled = sim_params.S * P_0 * sim_params.S'

    # Step 1: Compute eigenvalue decomposition: P = Q * Λ * Q'
    eigen_decomp = eigen(P_0_scaled)
    Q = eigen_decomp.vectors  # Eigenvectors (columns)
    λ_eigen = eigen_decomp.values  # Eigenvalues

    # Step 2: Generate sigma points using eigen-based method
    n = 6  # State dimension
    num_sigma_points = 2 * n
    sigma_points_0_scaled = Vector{Vector{Float64}}(undef, num_sigma_points)

    # Generate 2n sigma points: x ± sqrt(n * λ_i) * q_i
    # Using sqrt(n * λ_i) allows us to use equal weights w = 1/(2n) for both mean and covariance
    for i in 1:n
        q_i = Q[:, i]  # i-th eigenvector
        sqrt_n_λ_i = sqrt(n * λ_eigen[i])  # sqrt of n * i-th eigenvalue
        sigma_points_0_scaled[i] = x_vec_0_scaled .+ sqrt_n_λ_i .* q_i
        sigma_points_0_scaled[n+i] = x_vec_0_scaled .- sqrt_n_λ_i .* q_i
    end

    # Step 3: Compute weights for exact mean and covariance matching
    w = 1.0 / (2 * n)  # Equal weights for both mean and covariance
    w_m = fill(w, num_sigma_points)
    w_c = fill(w, num_sigma_points)

    # Step 4: Unscale sigma points for propagation (propagate_ks_relative_dynamics expects unscaled input)
    sigma_points_0 = [sim_params.S_inv * s for s in sigma_points_0_scaled]

    # Step 5: Propagate each sigma point (deputy) using linearized KS relative dynamics
    # Note: propagate_ks_relative_dynamics computes the chief (mean) trajectory internally
    println("  Propagating ", num_sigma_points, " sigma points (deputies) via linearized KS relative dynamics...")
    sigma_points_propagated = Vector{Vector{Vector{Float64}}}(undef, num_sigma_points)

    for (i, sigma_point_0) in enumerate(sigma_points_0)
        if i % 6 == 0
            println("    Sigma point ", i, " / ", num_sigma_points)
        end

        # Propagate using linearized KS relative dynamics
        # x_vec_0 is the chief (mean), sigma_point_0 is the deputy
        x_vec_traj_chief_i, x_vec_traj_rel_i, _, _ = propagate_ks_relative_dynamics(x_vec_0, sigma_point_0, times, sim_params)

        # Compute deputy trajectory: deputy = chief + relative
        x_vec_traj_deputy_i = [x_vec_traj_chief_i[k] + x_vec_traj_rel_i[k] for k in 1:length(times)]
        sigma_points_propagated[i] = x_vec_traj_deputy_i
    end

    # Step 6: Compute weighted mean and covariance at each time
    # The mean should be the weighted mean of all sigma points (not just the chief)
    x_vec_traj_mean = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))

    for t_idx in 1:length(times)
        # Scale propagated sigma points for numerical stability in covariance computation
        sigma_points_t_scaled = [sim_params.S * sigma_points_propagated[i][t_idx] for i in 1:num_sigma_points]

        # Compute weighted mean in scaled coordinates
        x_mean_scaled = zeros(6)
        for i in 1:num_sigma_points
            x_mean_scaled .+= w_m[i] .* sigma_points_t_scaled[i]
        end

        # Unscale mean
        x_vec_traj_mean[t_idx] = sim_params.S_inv * x_mean_scaled

        # Compute weighted covariance in scaled coordinates
        P_scaled = zeros(6, 6)
        for i in 1:num_sigma_points
            diff_scaled = sigma_points_t_scaled[i] .- x_mean_scaled
            P_scaled .+= w_c[i] .* (diff_scaled * diff_scaled')
        end

        # Unscale covariance: P = S_inv * P_scaled * S_inv'
        P_traj[t_idx] = sim_params.S_inv * P_scaled * sim_params.S_inv'
    end

    return x_vec_traj_mean, P_traj
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

    # Step 8: Propagate z_bar and compute state transition matrix Φ(s) = d([y; y'](s))/d([y; y']_0)
    # Initialize STM as identity (10x10)
    Φ_yyp_0_scaled = Matrix{Float64}(I, 10, 10)

    # Combine full state and STM into augmented state
    # State: [y; y'; h; t] (10 states)
    # STM: Φ_{[y;y';h;t]} (10x10 = 100 states)
    augmented_state_0_scaled = [z_bar_scaled_0; vec(Φ_yyp_0_scaled)]  # 10 + 100 = 110 states

    # Define augmented dynamics function
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

    # Pre-allocate trajectory storage
    z_bar_traj_scaled = [zeros(10) for k = 1:length(times_scaled)]
    z_bar_traj_scaled[1] .= z_bar_scaled_0
    Φ_traj_scaled = [zeros(10, 10) for k = 1:length(times_scaled)]  # 10x10 STM for [y; y'; h; t]
    Φ_traj_scaled[1] .= Φ_yyp_0_scaled
    times_traj_scaled = zeros(length(times_scaled))
    times_traj_scaled[1] = times_scaled[1]
    callback_triggered = falses(length(times_scaled))
    callback_triggered[1] = true  # Initial state is always set

    # Callback to save states at specific real time points
    function condition!(out, u, s_current_scaled, sol)
        t_current_scaled = u[10]  # Real time is the 10th element
        out .= times_scaled .- t_current_scaled
    end

    function affect!(sol, idx)
        if idx <= length(times_scaled)
            z_bar_traj_scaled[idx] .= sol.u[1:10]
            Φ_vec_scaled = sol.u[11:110]  # 10x10 STM
            Φ_traj_scaled[idx] .= reshape(Φ_vec_scaled, 10, 10)
            times_traj_scaled[idx] = sol.u[10]
            callback_triggered[idx] = true
        end
    end

    cb = VectorContinuousCallback(condition!, affect!, length(times_scaled))

    # Estimate fictitious time span
    r_vec_norm_0_scaled = y_vec_bar_scaled'y_vec_bar_scaled
    s_0_scaled = 0.0
    s_end_scaled = (times_scaled[end] - times_scaled[1]) / r_vec_norm_0_scaled

    # Integrate augmented system
    prob = ODEProblem(augmented_ks_dynamics_scaled!, augmented_state_0_scaled, (s_0_scaled, s_end_scaled), sim_params)
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, callback=cb)

    # Step 9: Propagate covariance: Q(s) = Φ_{[y;y';h;t]}(s) · Q_0 · Φ_{[y;y';h;t]}(s)^T
    # Q is 10x10 covariance for [y; y'; h; t]
    Q_traj_scaled = Vector{Matrix{Float64}}(undef, length(times_scaled))
    Q_traj_scaled[1] = Q_0_scaled

    for i in 2:length(times_scaled)
        Q_traj_scaled[i] = Φ_traj_scaled[i] * Q_0_scaled * Φ_traj_scaled[i]'
    end

    # Step 11: Convert back to Cartesian for output
    x_vec_traj_mean = Vector{Vector{Float64}}(undef, length(times_scaled))
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
        x_vec_traj_mean[i] = [x_vec_scaled[1:3] * sim_params.r_scale; x_vec_scaled[4:6] * sim_params.v_scale]
        # d_rvht_d_yypht = dCartesian_dKS(y_vec_scaled, y_vec_prime_scaled)
        d_rvht_d_yypht = dCartesian_dKS_fd(y_vec_scaled, y_vec_prime_scaled, h_scaled, t_scaled)

        # Map from Q (which is for [y; y'; h; t]) to augmented Cartesian ([r; v; h; t])
        P_scaled = d_rvht_d_yypht * Q_traj_scaled[i] * d_rvht_d_yypht'

        # Unscale covariance
        P_traj[i] = sim_params.S_inv * P_scaled[1:6, 1:6] * sim_params.S_inv'
    end

    return x_vec_traj_mean, P_traj
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
