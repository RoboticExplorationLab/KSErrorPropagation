import numpy as np
from scipy.integrate import solve_ivp


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


def ks_gravity_full(ks_state_full, s, GM):
    """
    KS dynamics with time tracking.
    Full state: [y_vec; y_vec_prime; h; t]
    """
    ks_state_augmented = ks_state_full[:9]

    # KS dynamics
    ks_state_augmented_prime = ks_gravity(ks_state_augmented, s, GM)

    # Time derivative: dt/ds = r = y'y
    y_vec = ks_state_augmented[:4]
    r_vec_norm = y_vec.T @ y_vec
    t_prime = r_vec_norm

    return np.concatenate([ks_state_augmented_prime, np.array([t_prime])])


def propagate_ks_keplerian_dynamics(ks_state_augmented_0, times, sim_params, GM):
    """
    Propagate KS state with time tracking, handling the transformation between
    real time t and fictitious time s where dt = r * ds and r = y'y.

    Parameters
    ----------
    ks_state_augmented_0 : numpy array
        Initial KS augmented state [y_vec; y_vec_prime; h] (9 elements)
    times : numpy array
        Array of times to save at
    sim_params : object
        Simulation parameters with attributes: method, rtol, atol
    GM : float
        Gravitational parameter

    Returns
    -------
    x_vec_traj_ks : numpy array
        Array of Cartesian states [r_vec; v_vec] at each time (N, 6)
    ks_state_augmented_traj : numpy array
        Array of KS augmented states [y_vec; y_vec_prime; h] at each time (N, 9)
    t_traj : numpy array
        Array of times (N,)
    """
    from ks_transform import state_ks_to_cartesian

    # KS full state vector: [y_vec; y_vec_prime; h; t]
    ks_state_full_0 = np.concatenate([ks_state_augmented_0, np.array([times[0]])])

    # Pre-allocate trajectory storage
    ks_state_augmented_traj = np.zeros((len(times), 9))
    ks_state_augmented_traj[0] = ks_state_augmented_0
    t_traj = np.zeros(len(times))
    t_traj[0] = times[0]

    # Estimate fictitious time span
    y_vec_0 = ks_state_augmented_0[:4]
    r_vec_norm_0 = y_vec_0 @ y_vec_0  # Use @ for 1D arrays
    s_0 = 0.0
    s_end = (times[-1] - times[0]) / r_vec_norm_0  # Match Julia: no margin factor

    # Storage for event-triggered states
    # Dictionary to store states when events fire: {target_time: (s, state)}
    event_states = {}

    # Define events for each time point (except the first)
    # Each event fires when we cross the target time
    def make_event(target_time, idx):
        def event(s, ks_state_full):
            t_current = ks_state_full[9]  # Real time is the 10th element
            return t_current - target_time

        event.terminal = False
        event.direction = 1  # Only trigger when crossing from below

        # Store the target time and index for later matching
        event.target_time = target_time
        event.target_idx = idx
        return event

    events = [make_event(times[i], i) for i in range(1, len(times))]

    # Integrate with events and dense output
    # Dense output allows evaluation at exact event times without interpolation to find s
    sol = solve_ivp(
        lambda s, ks_state_full: ks_gravity_full(ks_state_full, s, GM),
        (s_0, s_end),
        ks_state_full_0,
        method=sim_params.method,
        rtol=sim_params.rtol,
        atol=sim_params.atol,
        dense_output=True,
        events=events if len(events) > 0 else None,
    )

    # Extract states from events (no interpolation to find s - events give us exact s)
    # Match event times to target times
    if sol.t_events is not None and len(sol.t_events) > 0:
        for event_idx, event_times in enumerate(sol.t_events):
            if len(event_times) > 0 and event_idx < len(events):
                # Get the target time and index for this event
                target_time = events[event_idx].target_time
                target_idx = events[event_idx].target_idx

                # Use the first event time (events fire when crossing, giving exact s)
                s_event = event_times[0]

                # Get state at this exact s value using dense output
                # This evaluates at the exact s where the event fired (no interpolation to find s)
                ks_state_full = sol.sol(s_event)

                # Save the state
                ks_state_augmented_traj[target_idx] = ks_state_full[:9]
                t_traj[target_idx] = ks_state_full[9]

    # Fill in any missing states (if events didn't fire for some times)
    # This should rarely happen if events work correctly, but handle it gracefully
    for i in range(1, len(times)):
        # Check if this time point wasn't filled (t_traj[i] is still initial value or very different from target)
        if abs(t_traj[i] - times[0]) < 1e-10 or abs(t_traj[i] - times[i]) > 1e-6:
            # If integration reached this time, use the state at the closest integration point
            t_final_integrated = sol.y[9, -1]  # Final time reached in integration

            if times[i] <= t_final_integrated:
                # Find the integration point closest to the target time
                t_values = sol.y[9, :]
                closest_idx = np.argmin(np.abs(t_values - times[i]))
                ks_state_augmented_traj[i] = sol.y[:9, closest_idx]
                t_traj[i] = sol.y[9, closest_idx]
            else:
                # Integration didn't reach this time - use final state
                ks_state_augmented_traj[i] = sol.y[:9, -1]
                t_traj[i] = sol.y[9, -1]

    # Convert KS states to Cartesian
    x_vec_traj_ks = np.array(
        [
            state_ks_to_cartesian(ks_state_augmented_traj[k, :8])
            for k in range(len(times))
        ]
    )

    return x_vec_traj_ks, ks_state_augmented_traj, t_traj
