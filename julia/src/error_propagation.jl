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