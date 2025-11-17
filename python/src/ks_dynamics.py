import numpy as np


def L_matrix(ks_state):
    """
    Compute the L matrix from the Kustaanheimo-Stiefel state.
    Given a 4-vector ks_state, return the Left multiply Matrix L where
    col(r_vec,0) = L(ks_state) @ ks_state.
    """
    y1, y2, y3, y4 = ks_state

    L = np.array(
        [[y1, -y2, -y3, y4], [y2, y1, -y4, -y3], [y3, y4, y1, y2], [y4, -y3, y2, -y1]]
    )
    return L


def cartesian_coordinates2ks_state(cartesian_coordinates):
    """
    Transform 3D Cartesian coordinates to Kustaanheimo-Stiefel state.
    Matches the logic from ks_transform.jl with conditional handling based on r1.
    """
    x1, x2, x3 = cartesian_coordinates
    r_vec_norm = np.linalg.norm(cartesian_coordinates)

    if x1 >= 0:
        # Transform best suited for vectors with r1 >= 0
        y4 = 0.0
        y1 = np.sqrt(0.5 * (r_vec_norm + x1) - y4**2)
        y2 = (x2 * y1 + x3 * y4) / (x1 + r_vec_norm)
        y3 = (x3 * y1 - x2 * y4) / (x1 + r_vec_norm)
    else:
        # Transform best suited for vectors with r1 < 0
        y3 = 0.0
        y2 = np.sqrt(0.5 * (r_vec_norm - x1) - y3**2)
        y1 = (x2 * y2 + x3 * y3) / (r_vec_norm - x1)
        y4 = (x3 * y2 - x2 * y3) / (r_vec_norm - x1)

    ks_state = np.array([y1, y2, y3, y4])
    return ks_state


def ks_state2cartesian_coordinates(ks_state):
    """
    Transform Kustaanheimo-Stiefel state to Cartesian coordinates.
    Given a 4-vector ks_state, return the cartesian 3-vector.
    """
    L = L_matrix(ks_state)
    cartesian_coordinates = (L @ ks_state)[:3]
    return cartesian_coordinates


def cartesian_velocity2ks_state_dot(cartesian_velocity, ks_state):
    """
    Transform 3D Cartesian velocity to Kustaanheimo-Stiefel fictitious velocity.
    The cartesian velocities are wrt time (t), whereas the ks-transformed velocities
    are wrt fictitious time (s), where dt = ||x||ds.
    """
    x1_dot, x2_dot, x3_dot = cartesian_velocity
    y1, y2, y3, y4 = ks_state

    y1_dot = 0.5 * (y1 * x1_dot + y2 * x2_dot + y3 * x3_dot)
    y2_dot = 0.5 * (-y2 * x1_dot + y1 * x2_dot + y4 * x3_dot)
    y3_dot = 0.5 * (-y3 * x1_dot - y4 * x2_dot + y1 * x3_dot)
    y4_dot = 0.5 * (y4 * x1_dot - y3 * x2_dot + y2 * x3_dot)
    ks_state_dot = np.array([y1_dot, y2_dot, y3_dot, y4_dot])

    return ks_state_dot


def ks_velocity2cartesian_velocity(ks_state, ks_state_dot):
    """
    Transform Kustaanheimo-Stiefel velocity to Cartesian velocity.
    Given 4-vectors ks_state and ks_state_dot, return the 3-vector cartesian velocity.
    """
    L = L_matrix(ks_state)
    yTy = ks_state.T @ ks_state
    cartesian_velocity = ((2.0 / yTy) * (L @ ks_state_dot))[:3]
    return cartesian_velocity


def ks_ode_system(t, ks_augmented_state, mu):
    """
    Kustaanheimo-Stiefel transformed ODE system for two-body problem
    """
    ks_state, ks_state_dot = ks_augmented_state[:4], ks_augmented_state[4:8]
    h = (mu - 2 * ks_state_dot.T @ ks_state_dot) / (ks_state.T @ ks_state)
    ks_state_ddot = -h / 2 * ks_state

    return np.concatenate([ks_state_dot, ks_state_ddot])
