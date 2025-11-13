"""
Utility functions.
"""

using SatelliteDynamics

"""
    propagate_analytical_keplerian_orbit(oe_vec_0, times, GM)

Compute analytical Keplerian orbit solution at given times.

# Arguments
- `oe_vec_0`: initial orbital elements [a, e, i, RAAN, omega, M_0]
- `times`: array of times to compute solution at
- `GM`: gravitational parameter

# Returns
- `x_vec_traj_analytical`: array of Cartesian states [r_vec; v_vec] at each time
"""
function propagate_analytical_keplerian_orbit(oe_vec_0, times, GM)
    a, e, i, RAAN, omega, M_0 = oe_vec_0
    n = sqrt(GM / a^3)  # Mean motion

    x_vec_traj_analytical = Vector{Vector{Float64}}()

    for t in times
        # Mean anomaly at time t: M(t) = M_0 + n*(t - t_0)
        # Since t_0 = times[1], we have M(t) = M_0 + n*(t - times[1])
        M_t = M_0 + n * (t - times[1])

        # Wrap mean anomaly to [0, 2π)
        M_t = mod(M_t, 2π)

        # Compute orbital elements at time t
        oe_vec_t = [a, e, i, RAAN, omega, M_t]

        # Convert to Cartesian coordinates (analytical solution)
        x_vec_t = SatelliteDynamics.sOSCtoCART(oe_vec_t; GM=GM, use_degrees=false)
        push!(x_vec_traj_analytical, x_vec_t)
    end

    return x_vec_traj_analytical
end