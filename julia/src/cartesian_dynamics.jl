"""
Cartesian dynamics with gravity only (no J2, no drag).
"""

using LinearAlgebra
using SatelliteDynamics

const SD = SatelliteDynamics

"""
    cartesian_gravity(x_vec, t, GM)

Cartesian dynamics with gravity only (point mass gravity).

# Arguments
- `x_vec`: 6-vector state [r_vec; v_vec] where r_vec is position (3D) and v_vec is velocity (3D)
- `t`: time (not used but required for ODE interface)
- `GM`: gravitational parameter (default: GM_EARTH)

# Returns
- `x_vec_dot`: 6-vector derivative [r_vec_dot; v_vec_dot]
"""
function cartesian_gravity(x_vec, t, GM)
    r_vec = x_vec[1:3]
    v_vec = x_vec[4:6]

    # Unperturbed Cartesian dynamics (gravity only)
    r_vec_norm = norm(r_vec)
    a_grav = -GM * r_vec / r_vec_norm^3

    return a_grav
end

"""
    cartesian_gravity!(x_vec_dot, x_vec, p, t)

In-place version for DifferentialEquations.jl.

# Arguments
- `x_vec_dot`: output derivative vector
- `x_vec`: state vector [r_vec; v_vec]
- `p`: parameters (GM)
- `t`: time
"""
function cartesian_gravity!(x_vec_dot, x_vec, p, t)
    GM = p isa Number ? p : p[1]
    a_grav = cartesian_gravity(x_vec, t, GM)
    x_vec_dot .= [x_vec[4:6]; a_grav]
end

"""
    propagate_cartesian_keplerian_dynamics(x_vec_0, times, sim_params, GM)

Propagate Cartesian state with time tracking.

# Arguments
- `x_vec_0`: initial Cartesian state [r_vec_0; v_vec_0]
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter

# Returns
- `x_vec_traj`: array of Cartesian states [r_vec; v_vec] at each time
- `t_traj`: array of times
"""
function propagate_cartesian_keplerian_dynamics(x_vec_0, times, sim_params, GM)
    prob = ODEProblem(cartesian_gravity!, x_vec_0, (times[1], times[end]), GM)
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, saveat=times)
    x_vec_traj = [sol.u[k] for k = 1:length(sol.u)]
    t_traj = [sol.t[k] for k = 1:length(sol.t)]
    return x_vec_traj, t_traj
end

"""
    propagate_cartesian_keplerian_dynamics_scaled(x_vec_0, times, sim_params, GM, a)

Propagate Cartesian state with time tracking using normalized coordinates.

# Arguments
- `x_vec_0`: initial Cartesian state [r_vec_0; v_vec_0]
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter
- `a`: semi-major axis (for normalization)

# Returns
- `x_vec_traj`: array of Cartesian states [r_vec; v_vec] at each time
- `t_traj`: array of times
"""
function propagate_cartesian_keplerian_dynamics_scaled(x_vec_0, times, sim_params, GM, d_scale, t_scale)
    # Normalize initial state
    r_vec_0_scaled = x_vec_0[1:3] / d_scale
    v_vec_0_scaled = x_vec_0[4:6] * t_scale / d_scale
    x_vec_0_scaled = [r_vec_0_scaled; v_vec_0_scaled]

    # Normalize times
    times_scaled = times / t_scale

    # Normalized GM: GM_scaled = GM * t_scale² / d_scale³
    GM_scaled = GM * t_scale^2 / d_scale^3

    # Integrate in normalized coordinates
    prob = ODEProblem(cartesian_gravity!, x_vec_0_scaled, (times_scaled[1], times_scaled[end]), GM_scaled)
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, saveat=times_scaled)

    # Denormalize results
    x_vec_traj = Vector{Vector{Float64}}()
    t_traj = Vector{Float64}()
    for k = 1:length(sol.u)
        r_vec_scaled = sol.u[k][1:3]
        v_vec_scaled = sol.u[k][4:6]
        r_vec = r_vec_scaled * d_scale
        v_vec = v_vec_scaled * d_scale / t_scale
        push!(x_vec_traj, [r_vec; v_vec])
        push!(t_traj, sol.t[k] * t_scale)
    end

    return x_vec_traj, t_traj
end

"""
    cartesian_J2_perturbation(x_vec, t, GM, R_EARTH, J2)

Cartesian J2 perturbation.

# Arguments
- `x_vec`: Cartesian state [r_vec; v_vec]
- `t`: time (not used but required for ODE interface)
- `GM`: gravitational parameter
- `R_EARTH`: Earth radius
- `J2`: J2 coefficient

# Returns
- `a_J2`: J2 perturbation acceleration
"""
function cartesian_J2_perturbation(x_vec, t, GM, R_EARTH, J2)
    r_vec = x_vec[1:3]
    r_vec_norm = norm(r_vec)
    a_J2 = -(3.0 / 2.0) * J2 * (GM / r_vec_norm^2) * (R_EARTH / r_vec_norm)^2 * [
               (1 - 5 * (r_vec[3] / r_vec_norm)^2) * r_vec[1] / r_vec_norm,
               (1 - 5 * (r_vec[3] / r_vec_norm)^2) * r_vec[2] / r_vec_norm,
               (3 - 5 * (r_vec[3] / r_vec_norm)^2) * r_vec[3] / r_vec_norm]
    return a_J2
end

"""
    cartesian_drag_perturbation(x_vec, t, epoch, OMEGA_EARTH, CD, A, m)

Cartesian drag perturbation using NRLMSISE-00 atmospheric model.

# Arguments
- `x_vec`: Cartesian state [r_vec; v_vec]
- `t`: elapsed time since initial epoch (seconds)
- `epoch`: current epoch (Epoch object)
- `OMEGA_EARTH`: Earth rotation rate (rad/s)
- `CD`: drag coefficient (dimensionless)
- `A`: cross-sectional area (m²)
- `m`: mass (kg)

# Returns
- `a_drag`: drag perturbation acceleration (m/s²)
"""
function cartesian_drag_perturbation(x_vec, t, epoch, OMEGA_EARTH, CD, A, m)
    r_vec = x_vec[1:3]
    v_vec = x_vec[4:6]
    r_vec_norm = norm(r_vec)

    # Earth's rotation vector
    omega = [0.0, 0.0, OMEGA_EARTH]

    # Velocity relative to the Earth's atmosphere (accounting for rotation)
    v_vec_rel = v_vec - cross(omega, r_vec)
    v_vec_rel_norm = norm(v_vec_rel)

    # Skip drag if velocity is negligible
    if v_vec_rel_norm < 1e-6
        return zeros(3)
    end

    # Atmospheric density using NRLMSISE-00 model
    # Convert ECI position to ECEF (accounting for Earth's rotation)
    r_ecef = SD.sECItoECEF(epoch, r_vec)
    # Convert ECEF to geodetic coordinates [latitude, longitude, altitude]
    geod = SD.sECEFtoGEOD(r_ecef; use_degrees=false)
    # density_nrlmsise00 requires Epoch and geodetic coordinates [lat, lon, alt]
    rho = SD.density_nrlmsise00(epoch, geod; use_degrees=false)

    # Drag acceleration: a = -0.5 * (CD * A / m) * ρ * v² * v̂
    a_drag = -0.5 * (CD * A / m) * rho * v_vec_rel_norm * v_vec_rel

    return a_drag
end

"""
    cartesian_perturbed_dynamics!(x_vec_dot, x_vec, p, t)

In-place version for DifferentialEquations.jl.

# Arguments
- `x_vec_dot`: output derivative vector
- `x_vec`: state vector [r_vec; v_vec]
- `p`: parameters (GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m)
- `t`: elapsed time since initial epoch (seconds)
"""
function cartesian_perturbed_dynamics!(x_vec_dot, x_vec, p, t)
    # Unpack parameters tuple: (GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m)
    GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m = p

    # Update epoch to current time (t is elapsed time since initial epoch)
    epoch = epoch_0 + t

    a_grav = cartesian_gravity(x_vec, t, GM)
    a_J2 = cartesian_J2_perturbation(x_vec, t, GM, R_EARTH, J2)
    a_drag = cartesian_drag_perturbation(x_vec, t, epoch, OMEGA_EARTH, CD, A, m)

    x_vec_dot .= [x_vec[4:6]; a_grav + a_J2 + a_drag]
end

"""
    propagate_cartesian_perturbed_dynamics(x_vec_0, times, sim_params, GM, R_EARTH, J2, epoch, OMEGA_EARTH, CD, A, m)

Propagate Cartesian state with time tracking using perturbed dynamics.

# Arguments
- `x_vec_0`: initial Cartesian state [r_vec_0; v_vec_0]
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
function propagate_cartesian_perturbed_dynamics(x_vec_0, times, sim_params, GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m)
    # Use tuple instead of array for better performance with mixed types
    prob = ODEProblem(cartesian_perturbed_dynamics!, x_vec_0, (times[1], times[end]), (GM, R_EARTH, J2, epoch_0, OMEGA_EARTH, CD, A, m))
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, saveat=times)
    x_vec_traj = [sol.u[k] for k = 1:length(sol.u)]
    t_traj = [sol.t[k] for k = 1:length(sol.t)]
    return x_vec_traj, t_traj
end