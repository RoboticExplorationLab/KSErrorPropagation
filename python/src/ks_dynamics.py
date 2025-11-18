import numpy as np


def ks_gravity(ks_state_augmented, s, GM):
    """
    Kustaanheimo-Stiefel dynamics with gravity only (point mass gravity).
    """
    y_vec = ks_state_augmented[:4]
    y_vec_prime = ks_state_augmented[4:8]
    h = ks_state_augmented[8]

    # Unperturbed KS dynamics (gravity only)
    y_vec_pprime = (-h / 2.0) * y_vec
    h_prime = 0.0

    return np.concatenate([y_vec_prime, y_vec_pprime, np.array([h_prime])])
