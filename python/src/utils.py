import numpy as np
import pykep as pk
from pyorb.kepler import mean_to_eccentric


from matplotlib.ticker import FuncFormatter


def propagate_analytical_keplerian_dynamics(oe_vec_0, times, GM):
    """
    Compute analytical Keplerian orbit solution at given times.

    Parameters:
    -----------
    oe_vec_0 : list
        Initial orbital elements [a, e, i, RAAN, omega, M]
    times : list
        Times to compute solution at
    GM : float
        Gravitational parameter (km^3/s^2)

    Returns:
    --------
    x_vec_traj_analytical : numpy array
        Cartesian states [r_vec; v_vec] at each time
    """
    a, e, i, RAAN, omega, M_0 = oe_vec_0

    # Mean motion calculation: use absolute value of a for hyperbolic orbits
    n = np.sqrt(GM / np.abs(a) ** 3)  # Mean motion (always positive)

    x_vec_traj_analytical = np.zeros((len(times), 6))
    t_0 = times[0]  # Initial time reference

    for idx, t in enumerate(times):
        # Mean anomaly at time t: M(t) = M_0 + n*(t - t_0)
        M_t = M_0 + n * (t - t_0)

        # Wrap mean anomaly to [0, 2π) (matching Julia implementation)
        M_t = np.mod(M_t, 2 * np.pi)

        # Compute orbital elements at time t
        oe_vec_t = [a, e, i, RAAN, omega, mean_to_eccentric(M_t, e, degrees=False)]

        # Convert to Cartesian coordinates (analytical solution)
        r_vec_t, v_vec_t = pk.par2ic(oe_vec_t, GM)
        # Convert tuples to numpy arrays
        r_vec_t = np.array(r_vec_t)
        v_vec_t = np.array(v_vec_t)
        x_vec_traj_analytical[idx, :] = np.concatenate([r_vec_t, v_vec_t])

    return x_vec_traj_analytical


def r_vecs2several_quantities(r_vec):
    """Convert 3D position vectors to quantities"""
    x = r_vec[:, 0]
    y = r_vec[:, 1]
    z = r_vec[:, 2]
    r_vec_norm = np.linalg.norm(r_vec, axis=1)
    # For 3D, theta is the angle in the xy-plane
    theta = np.arctan2(y, x)
    return x, y, z, theta, r_vec, r_vec_norm


def apply_scientific_tick_labels(ax, axis, arrays, decimals=1):
    """Normalize tick labels to their dominant order of magnitude and annotate axis labels."""

    def _format_label(label_text, magnitude):
        if magnitude == 0:
            return label_text
        prefix = label_text
        unit = ""
        suffix = ""
        if "[" in label_text and "]" in label_text:
            before, rest = label_text.split("[", 1)
            unit_part, after = rest.split("]", 1)
            prefix = before.rstrip()
            unit = unit_part.strip()
            suffix = after
        bracket_content = (
            f"$\\times 10^{{{magnitude}}}$"
            if not unit
            else f"$\\times 10^{{{magnitude}}}$ {unit}"
        )
        if unit:
            return f"{prefix} [{bracket_content}]{suffix}"
        return f"{prefix} ({bracket_content})"

    values = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        values.append(arr.ravel())
    if not values:
        return
    data = np.concatenate(values)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    max_val = np.max(np.abs(data))
    magnitude = 0 if max_val == 0 else int(np.floor(np.log10(max_val)))
    scale = 10.0**magnitude if magnitude != 0 else 1.0

    formatter = FuncFormatter(lambda value, _: f"{value / scale:.{decimals}f}")

    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel(_format_label(ax.get_xlabel(), magnitude))
    elif axis == "y":
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(_format_label(ax.get_ylabel(), magnitude))
    elif axis == "z":
        ax.zaxis.set_major_formatter(formatter)
        base_label = ax.get_zlabel().split(" ($")[0]
        ax.set_zlabel(f"{base_label} ($\\times 10^{{{magnitude}}}$)")


def rk4_integration(ode_system, state0, t0, dt):
    """RK4 integration of the ODE system"""
    k1 = ode_system(t0, state0)
    k2 = ode_system(t0 + dt / 2, state0 + dt / 2 * k1)
    k3 = ode_system(t0 + dt / 2, state0 + dt / 2 * k2)
    k4 = ode_system(t0 + dt, state0 + dt * k3)
    return state0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
