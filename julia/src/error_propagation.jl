using Random
using LinearAlgebra

function propagate_monte_carlo(x_vec_0, P_0, times, sim_params, num_samples=10000)
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
    x_vec_traj_mean = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))

    # Compute mean and covariance at each time
    for t_idx in 1:length(times)
        # Compute mean
        x_mean = zeros(6)
        for i in 1:num_samples
            x_mean .+= samples_propagated[t_idx][i]
        end
        x_mean ./= num_samples
        x_vec_traj_mean[t_idx] = x_mean

        # Compute covariance
        P = zeros(6, 6)
        for i in 1:num_samples
            diff = samples_propagated[t_idx][i] .- x_mean
            P .+= diff * diff'
        end
        P ./= (num_samples - 1)  # Sample covariance (Bessel's correction)
        P = (P + P') / 2.0  # Ensure symmetry
        P_traj[t_idx] = P
    end

    return x_vec_traj_mean, P_traj, times
end