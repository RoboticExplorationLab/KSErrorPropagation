import numpy as np


def L(y_vec):
    """
    Compute the L matrix from the Kustaanheimo-Stiefel state.
    Given a 4-vector y_vec, return the Left multiply Matrix L where
    col(r_vec,0) = L(y_vec) @ y_vec.
    """
    y1, y2, y3, y4 = y_vec

    return np.array(
        [[y1, -y2, -y3, y4], [y2, y1, -y4, -y3], [y3, y4, y1, y2], [y4, -y3, y2, -y1]]
    )


def position_cartesian_to_ks(r_vec):
    """
    Transform 3D Cartesian coordinates to Kustaanheimo-Stiefel state.
    Matches the logic from ks_transform.jl with conditional handling based on r1.
    """
    r1, r2, r3 = r_vec
    r_vec_norm = np.linalg.norm(r_vec)

    if r1 >= 0:
        # Transform best suited for vectors with r1 >= 0
        y4 = 0.0
        y1 = np.sqrt(0.5 * (r_vec_norm + r1) - y4**2)
        y2 = (r2 * y1 + r3 * y4) / (r1 + r_vec_norm)
        y3 = (r3 * y1 - r2 * y4) / (r1 + r_vec_norm)
    else:
        # Transform best suited for vectors with r1 < 0
        y3 = 0.0
        y2 = np.sqrt(0.5 * (r_vec_norm - r1) - y3**2)
        y1 = (r2 * y2 + r3 * y3) / (r_vec_norm - r1)
        y4 = (r3 * y2 - r2 * y3) / (r_vec_norm - r1)

    return np.array([y1, y2, y3, y4])


def position_ks_to_cartesian(y_vec):
    """
    Transform Kustaanheimo-Stiefel state to Cartesian coordinates.
    Given a 4-vector y_vec, return the cartesian 3-vector.
    """
    return (L(y_vec) @ y_vec)[:3]


def velocity_cartesian_to_ks(y_vec, v_vec):
    """
    Transform 3D Cartesian velocity to Kustaanheimo-Stiefel fictitious velocity.
    The cartesian velocities are wrt time (t), whereas the ks-transformed velocities
    are wrt fictitious time (s), where dt = ||x||ds.
    """
    y1, y2, y3, y4 = y_vec
    v1, v2, v3 = v_vec

    y1_prime = 0.5 * (y1 * v1 + y2 * v2 + y3 * v3)
    y2_prime = 0.5 * (-y2 * v1 + y1 * v2 + y4 * v3)
    y3_prime = 0.5 * (-y3 * v1 - y4 * v2 + y1 * v3)
    y4_prime = 0.5 * (y4 * v1 - y3 * v2 + y2 * v3)

    return np.array([y1_prime, y2_prime, y3_prime, y4_prime])


def velocity_ks_to_cartesian(y_vec, y_vec_prime):
    """
    Transform Kustaanheimo-Stiefel velocity to Cartesian velocity.
    Given 4-vectors y_vec and y_vec_prime, return the 3-vector cartesian velocity.
    """
    return ((2.0 / y_vec.T @ y_vec) * (L(y_vec) @ y_vec_prime))[:3]


def state_ks_to_cartesian(ks_state):
    """
    Transform Kustaanheimo-Stiefel state to Cartesian state.
    Given 4-vectors y_vec and y_vec_prime, return the 6-vector cartesian state.
    """
    y_vec, y_vec_prime = ks_state[:4], ks_state[4:]
    r_vec = position_ks_to_cartesian(y_vec)
    v_vec = velocity_ks_to_cartesian(y_vec, y_vec_prime)
    return np.concatenate([r_vec, v_vec])


def state_cartesian_to_ks(x_vec):
    """
    Transform Cartesian state to Kustaanheimo-Stiefel state.
    Given 6-vector x_vec = [r_vec, v_vec], return the 8-vector KS state.
    """
    r_vec, v_vec = x_vec[:3], x_vec[3:]
    y_vec = position_cartesian_to_ks(r_vec)
    y_vec_prime = velocity_cartesian_to_ks(y_vec, v_vec)
    return np.concatenate([y_vec, y_vec_prime])


def energy_ks(y_vec, y_vec_prime, GM):
    """
    Compute the KS energy parameter h from KS position and velocity.
    """
    return (GM - 2 * (y_vec_prime.T @ y_vec_prime)) / (y_vec.T @ y_vec)
