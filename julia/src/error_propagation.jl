using Random
using LinearAlgebra
using ForwardDiff
using DifferentialEquations

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
        P_scaled = (P_scaled + P_scaled') / 2.0  # Ensure symmetry

        # Unscale covariance: P = S_inv * P_scaled * S_inv'
        P_traj[t_idx] = sim_params.S_inv * P_scaled * sim_params.S_inv'
        # Ensure symmetry after unscaling
        P_traj[t_idx] = (P_traj[t_idx] + P_traj[t_idx]') / 2.0
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
        # Ensure symmetry (numerical errors can make it slightly non-symmetric)
        P_traj_scaled[i] = (P_traj_scaled[i] + P_traj_scaled[i]') / 2.0
    end

    # Step 6: Unscale covariance: P = S_inv * P_scaled * S_inv'
    P_traj = Vector{Matrix{Float64}}(undef, length(times))
    for i in 1:length(times)
        P_traj[i] = sim_params.S_inv * P_traj_scaled[i] * sim_params.S_inv'
        # Ensure symmetry after unscaling
        P_traj[i] = (P_traj[i] + P_traj[i]') / 2.0
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
    pos_rms = sqrt(sum(pos_errors .^ 2) / N)
    pos_min = minimum(pos_errors)
    pos_max = maximum(pos_errors)

    # Velocity metrics
    vel_rms = sqrt(sum(vel_errors .^ 2) / N)
    vel_min = minimum(vel_errors)
    vel_max = maximum(vel_errors)

    # Extract standard deviations from covariance matrices
    # Position standard deviations: sqrt(P[i,i]) for i = 1, 2, 3
    # Velocity standard deviations: sqrt(P[i,i]) for i = 4, 5, 6
    σ_pos_ref = zeros(N, 3)  # [σ_x, σ_y, σ_z] for each time step
    σ_pos_test = zeros(N, 3)
    σ_vel_ref = zeros(N, 3)  # [σ_vx, σ_vy, σ_vz] for each time step
    σ_vel_test = zeros(N, 3)

    for i in 1:N
        # Position standard deviations
        σ_pos_ref[i, 1] = sqrt(max(0.0, P_traj_ref[i][1, 1]))  # σ_x
        σ_pos_ref[i, 2] = sqrt(max(0.0, P_traj_ref[i][2, 2]))  # σ_y
        σ_pos_ref[i, 3] = sqrt(max(0.0, P_traj_ref[i][3, 3]))  # σ_z

        σ_pos_test[i, 1] = sqrt(max(0.0, P_traj_test[i][1, 1]))  # σ_x
        σ_pos_test[i, 2] = sqrt(max(0.0, P_traj_test[i][2, 2]))  # σ_y
        σ_pos_test[i, 3] = sqrt(max(0.0, P_traj_test[i][3, 3]))  # σ_z

        # Velocity standard deviations
        σ_vel_ref[i, 1] = sqrt(max(0.0, P_traj_ref[i][4, 4]))  # σ_vx
        σ_vel_ref[i, 2] = sqrt(max(0.0, P_traj_ref[i][5, 5]))  # σ_vy
        σ_vel_ref[i, 3] = sqrt(max(0.0, P_traj_ref[i][6, 6]))  # σ_vz

        σ_vel_test[i, 1] = sqrt(max(0.0, P_traj_test[i][4, 4]))  # σ_vx
        σ_vel_test[i, 2] = sqrt(max(0.0, P_traj_test[i][5, 5]))  # σ_vy
        σ_vel_test[i, 3] = sqrt(max(0.0, P_traj_test[i][6, 6]))  # σ_vz
    end

    # Compute covariance standard deviation errors for each component
    # Position components (x, y, z)
    cov_pos_errors = σ_pos_test - σ_pos_ref  # N x 3 matrix
    cov_pos_rmse = [sqrt(sum(cov_pos_errors[:, j] .^ 2) / N) for j in 1:3]
    cov_pos_min = [minimum(cov_pos_errors[:, j]) for j in 1:3]
    cov_pos_max = [maximum(cov_pos_errors[:, j]) for j in 1:3]

    # Velocity components (vx, vy, vz)
    cov_vel_errors = σ_vel_test - σ_vel_ref  # N x 3 matrix
    cov_vel_rmse = [sqrt(sum(cov_vel_errors[:, j] .^ 2) / N) for j in 1:3]
    cov_vel_min = [minimum(cov_vel_errors[:, j]) for j in 1:3]
    cov_vel_max = [maximum(cov_vel_errors[:, j]) for j in 1:3]

    return (
        pos_rms=pos_rms,
        pos_min=pos_min,
        pos_max=pos_max,
        vel_rms=vel_rms,
        vel_min=vel_min,
        vel_max=vel_max,
        cov_pos_rmse=cov_pos_rmse,
        cov_pos_min=cov_pos_min,
        cov_pos_max=cov_pos_max,
        cov_vel_rmse=cov_vel_rmse,
        cov_vel_min=cov_vel_min,
        cov_vel_max=cov_vel_max,
    )
end
