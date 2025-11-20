"""
Error propagation utilities for sigma point (Unscented Transform) methods and Monte Carlo.
"""

using LinearAlgebra
using Random
using ForwardDiff

# Include transformation functions (will be included by ks_dynamics.jl or cartesian_dynamics.jl)
# These functions need to be available when this file is included

"""
    generate_sigma_points(x_mean, P, alpha=1.0, beta=2.0, kappa=0.0)

Generate 2n+1 sigma points from mean and covariance using Unscented Transform.

# Arguments
- `x_mean`: mean vector (n-dimensional)
- `P`: covariance matrix (n×n)
- `alpha`: scaling parameter (default: 1.0)
- `beta`: scaling parameter for higher order terms (default: 2.0, optimal for Gaussian)
- `kappa`: secondary scaling parameter (default: 0.0)

# Returns
- `sigma_points`: array of 2n+1 sigma points (each is n-dimensional)
- `weights_mean`: weights for mean reconstruction (2n+1 elements)
- `weights_cov`: weights for covariance reconstruction (2n+1 elements)
"""
function generate_sigma_points(x_mean, P, alpha=1.0, beta=2.0, kappa=0.0)
    n = length(x_mean)

    # Compute scaling parameter
    lambda = alpha^2 * (n + kappa) - n

    # Handle edge case when lambda = 0 (happens when alpha=1, kappa=0)
    # Use a small positive value to ensure the center point gets proper weight
    if abs(lambda) < 1e-10
        lambda = 1e-10
    end

    # Compute matrix square root of (n + lambda) * P
    # Using Cholesky decomposition for numerical stability
    local sqrt_P
    try
        sqrt_P = cholesky((n + lambda) * P).L
    catch
        # If Cholesky fails, use eigenvalue decomposition
        eigen_decomp = eigen((n + lambda) * P)
        sqrt_P = eigen_decomp.vectors * diagm(sqrt.(max.(eigen_decomp.values, 0.0)))
    end

    # Generate sigma points
    sigma_points = Vector{Vector{Float64}}(undef, 2 * n + 1)

    # First sigma point is the mean
    sigma_points[1] = copy(x_mean)

    # Remaining 2n sigma points
    for i in 1:n
        sigma_points[i+1] = x_mean .+ sqrt_P[:, i]
        sigma_points[i+n+1] = x_mean .- sqrt_P[:, i]
    end

    # Compute weights
    weights_mean = zeros(2 * n + 1)
    weights_cov = zeros(2 * n + 1)

    # Weight for mean point
    weights_mean[1] = lambda / (n + lambda)
    weights_cov[1] = lambda / (n + lambda) + (1 - alpha^2 + beta)

    # Weights for remaining points
    w = 1.0 / (2 * (n + lambda))
    for i in 2:(2*n+1)
        weights_mean[i] = w
        weights_cov[i] = w
    end

    return sigma_points, weights_mean, weights_cov
end

"""
    reconstruct_from_sigma_points(sigma_points, weights_mean, weights_cov)

Reconstruct mean and covariance from propagated sigma points.

# Arguments
- `sigma_points`: array of propagated sigma points (each is n-dimensional)
- `weights_mean`: weights for mean reconstruction (2n+1 elements)
- `weights_cov`: weights for covariance reconstruction (2n+1 elements)

# Returns
- `x_mean`: reconstructed mean vector (n-dimensional)
- `P`: reconstructed covariance matrix (n×n)
"""
function reconstruct_from_sigma_points(sigma_points, weights_mean, weights_cov)
    n = length(sigma_points[1])
    num_points = length(sigma_points)

    # Reconstruct mean
    x_mean = zeros(n)
    for i in 1:num_points
        x_mean .+= weights_mean[i] .* sigma_points[i]
    end

    # Reconstruct covariance
    P = zeros(n, n)
    for i in 1:num_points
        diff = sigma_points[i] .- x_mean
        P .+= weights_cov[i] .* (diff * diff')
    end

    # Ensure symmetry (numerical precision)
    P = (P + P') / 2.0

    return x_mean, P
end

"""
    propagate_monte_carlo(x_vec_0, P_0, times, sim_params, GM, num_samples=10000)

Propagate state uncertainty using Monte Carlo method (ground truth).

# Arguments
- `x_vec_0`: initial mean state vector [r_vec; v_vec] (6-dimensional)
- `P_0`: initial covariance matrix (6×6)
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter
- `num_samples`: number of Monte Carlo samples (default: 10000)

# Returns
- `x_vec_traj_mean`: array of mean states at each time
- `P_traj`: array of covariance matrices at each time
- `times`: array of times
"""
function propagate_monte_carlo(x_vec_0, P_0, times, sim_params, GM, num_samples=10000)
    # Sample initial states from multivariate Gaussian
    # Use Cholesky decomposition: if x ~ N(0, I), then L_chol*x + μ ~ N(μ, P) where P = L_chol*L_chol'
    local L_chol
    try
        L_chol = cholesky(P_0).L
    catch
        # If Cholesky fails, use eigenvalue decomposition
        eigen_decomp = eigen(P_0)
        L_chol = eigen_decomp.vectors * diagm(sqrt.(max.(eigen_decomp.values, 0.0)))
    end

    # Generate samples
    samples_0 = Vector{Vector{Float64}}(undef, num_samples)
    for i in 1:num_samples
        z = randn(6)  # Standard normal
        samples_0[i] = x_vec_0 .+ L_chol * z
    end

    # Propagate each sample
    samples_propagated = Vector{Vector{Vector{Float64}}}(undef, length(times))
    for t_idx in 1:length(times)
        samples_propagated[t_idx] = Vector{Vector{Float64}}(undef, num_samples)
    end

    println("  Propagating ", num_samples, " Monte Carlo samples...")
    for (i, sample_0) in enumerate(samples_0)
        if i % 1000 == 0
            println("    Sample ", i, " / ", num_samples)
        end

        # Propagate sample through full nonlinear dynamics
        x_vec_traj_sample, _ = propagate_cartesian_keplerian_dynamics(sample_0, times, sim_params, GM)

        # Store propagated states
        for t_idx in 1:length(times)
            samples_propagated[t_idx][i] = x_vec_traj_sample[t_idx]
        end
    end

    # Compute mean and covariance at each time
    x_vec_traj_mean = Vector{Vector{Float64}}(undef, length(times))
    P_traj = Vector{Matrix{Float64}}(undef, length(times))

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

"""
    transform_covariance_cartesian_to_ks(P_cart, x_vec_cart)

Transform covariance matrix from Cartesian to KS coordinates.

# Arguments
- `P_cart`: covariance matrix in Cartesian coordinates (6×6)
- `x_vec_cart`: Cartesian state vector [r_vec; v_vec] (6-dimensional)

# Returns
- `P_ks`: covariance matrix in KS coordinates (8×8)
"""
function transform_covariance_cartesian_to_ks(P_cart, x_vec_cart)
    # Compute Jacobian of transformation: ∂(KS) / ∂(Cartesian)
    # Using ForwardDiff to compute the Jacobian
    function cartesian_to_ks_wrapper(x_cart)
        ks_state = state_cartesian_to_ks(x_cart)
        return ks_state
    end

    J = ForwardDiff.jacobian(cartesian_to_ks_wrapper, x_vec_cart)

    # Transform covariance: P_ks = J * P_cart * J'
    P_ks = J * P_cart * J'

    # Ensure symmetry
    P_ks = (P_ks + P_ks') / 2.0

    return P_ks
end

"""
    transform_covariance_ks_to_cartesian(P_ks, ks_state)

Transform covariance matrix from KS to Cartesian coordinates.

# Arguments
- `P_ks`: covariance matrix in KS coordinates (8×8)
- `ks_state`: KS state vector [y_vec; y_vec_prime] (8-dimensional)

# Returns
- `P_cart`: covariance matrix in Cartesian coordinates (6×6)
"""
function transform_covariance_ks_to_cartesian(P_ks, ks_state)
    # Compute Jacobian of transformation: ∂(Cartesian) / ∂(KS)
    # Using ForwardDiff to compute the Jacobian
    function ks_to_cartesian_wrapper(ks_state_input)
        x_cart = state_ks_to_cartesian(ks_state_input)
        return x_cart
    end

    J = ForwardDiff.jacobian(ks_to_cartesian_wrapper, ks_state)

    # Transform covariance: P_cart = J * P_ks * J'
    P_cart = J * P_ks * J'

    # Ensure symmetry
    P_cart = (P_cart + P_cart') / 2.0

    return P_cart
end

