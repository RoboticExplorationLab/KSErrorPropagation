import numpy as np
from scipy.integrate import solve_ivp


def cartesian_gravity(x_vec, t, GM):
    """Two-body orbital dynamics ODE (3D)"""
    r_vec, v_vec = x_vec[:3], x_vec[3:]

    # Unperturbed Cartesian dynamics (gravity only)
    r_vec_norm = np.linalg.norm(r_vec)
    a_grav = -GM * r_vec / r_vec_norm**3

    return np.concatenate([v_vec, a_grav])


def propagate_cartesian_keplerian_dynamics(x_vec_0, times, sim_params, GM):
    """
    Propagate Cartesian state with time tracking.
    """
    sol = solve_ivp(
        lambda t, x_vec: cartesian_gravity(x_vec, t, GM),
        (times[0], times[-1]),
        x_vec_0,
        method=sim_params.method,
        t_eval=times,
        rtol=sim_params.rtol,
        atol=sim_params.atol,
    )
    x_vec_traj = sol.y.T  # Transpose to get (N+1, 6) shape
    t_traj = sol.t
    return x_vec_traj, t_traj
