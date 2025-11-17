import numpy as np
import pykep as pk


from matplotlib.ticker import FuncFormatter


def analytical_solution(t, a, e, mu, t0=0.0):
    """
    Analytical Keplerian propagation using pykep.propagate_lagrangian (like old notebook).
    Uses pykep for OSCtoCART conversion and propagation.
    Assumes i = 0, Ω = 0, ω = 0 and starts at periapsis at t = 0.

    Parameters:
    -----------
    t : float
        Time (seconds)
    a : float
        Semi-major axis (km). Can be negative for hyperbolic orbits.
    e : float
        Eccentricity
    mu : float
        Gravitational parameter (km^3/s^2)
    t0 : float
        Initial time (default 0.0)

    Returns:
    --------
    x, y, z, r_vec, r_vec_norm
    """
    # Get initial state at periapsis (M=0) using pykep
    # pykep can work with km units directly when mu is in km^3/s^2
    a_abs = abs(a) if a < 0 else a

    # Initial orbital elements at periapsis: [a, e, i, RAAN, omega, M]
    el0 = [a_abs, e, 0.0, 0.0, 0.0, 0.0]

    # Convert to initial Cartesian state (in km and km/s)
    r0, v0 = pk.par2ic(el0, mu)
    r0 = np.array(r0)  # km
    v0 = np.array(v0)  # km/s

    # Propagate from initial state to time t using pykep.propagate_lagrangian
    # This function handles the Keplerian propagation correctly
    rf, vf = pk.propagate_lagrangian(r0, v0, t, mu)

    # Results are already in km and km/s
    r_vec_3d = np.array(rf)

    # Extract x, y, z coordinates
    x, y, z = r_vec_3d

    r_vec_norm = np.linalg.norm(r_vec_3d)

    return x, y, z, r_vec_3d, r_vec_norm


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
