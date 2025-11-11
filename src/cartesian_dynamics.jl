"""
Cartesian dynamics with gravity only (no J2, no drag).
"""

using LinearAlgebra
using SatelliteDynamics

const SD = SatelliteDynamics

"""
    cartesian_gravity_dynamics(x, t, GM)

Cartesian dynamics with gravity only (point mass gravity).

# Arguments
- `x`: 6-vector state [r; v] where r is position (3D) and v is velocity (3D)
- `t`: time (not used but required for ODE interface)
- `GM`: gravitational parameter (default: GM_EARTH)

# Returns
- `xdot`: 6-vector derivative [rdot; vdot]
"""
function cartesian_gravity_dynamics(x_vec, t, GM=SD.GM_EARTH)
    r_vec = x_vec[1:3]
    v_vec = x_vec[4:6]
    
    r_vec_norm = norm(r_vec)
    
    r_vec_dot = v_vec
    v_vec_dot = -(GM / r_vec_norm^3) * r_vec
    
    return [r_vec_dot; v_vec_dot]
end

"""
    cartesian_gravity_dynamics!(xdot, x, p, t)

In-place version for DifferentialEquations.jl.

# Arguments
- `xdot`: output derivative vector
- `x`: state vector [r; v]
- `p`: parameters (GM)
- `t`: time
"""
function cartesian_gravity_dynamics!(xdot, x, p, t)
    GM = p isa Number ? p : p[1]
    xdot .= cartesian_gravity_dynamics(x, t, GM)
end

