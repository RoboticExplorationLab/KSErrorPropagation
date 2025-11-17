import numpy as np


def two_body_ode(t, state, mu):
    """Two-body orbital dynamics ODE (3D)"""
    r_vec, v_vec = state[:3], state[3:]
    r_vec_norm = np.linalg.norm(r_vec)
    a_vec = -mu * r_vec / r_vec_norm**3
    return np.concatenate([v_vec, a_vec])
