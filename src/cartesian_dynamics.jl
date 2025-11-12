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
function cartesian_gravity(x_vec, t, GM=SD.GM_EARTH)
    r_vec = x_vec[1:3]
    v_vec = x_vec[4:6]
    
    # Unperturbed Cartesian dynamics (gravity only)
    r_vec_norm = norm(r_vec)
    r_vec_dot = v_vec
    v_vec_dot = -(GM / r_vec_norm^3) * r_vec
    
    return [r_vec_dot; v_vec_dot]
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
    x_vec_dot .= cartesian_gravity(x_vec, t, GM)
end

"""
    propagate_cartesian_keplerian_orbit(x_vec_0, times, sim_params, GM)

Propagate Cartesian state with time tracking.

# Arguments
- `x_vec_0`: initial Cartesian state [r_vec_0; v_vec_0]
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter

# Returns
- `x_vec_traj`: array of Cartesian states [r_vec; v_vec] at each time
"""
function propagate_cartesian_keplerian_orbit(x_vec_0, times, sim_params, GM=SD.GM_EARTH)
    prob = ODEProblem(cartesian_gravity!, x_vec_0, (times[1], times[end]), GM)
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, saveat=times)
    x_vec_traj = [sol.u[k] for k = 1:length(sol.u)]
    return x_vec_traj
end