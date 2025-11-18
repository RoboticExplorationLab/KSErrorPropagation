"""
Run Python propagate_ks_keplerian_dynamics() and save results to data/ directory.
"""

import numpy as np
import pykep as pk
import sys
import os
import csv

# Add python/src to path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "python", "src"))

from ks_dynamics import propagate_ks_keplerian_dynamics
from ks_transform import state_cartesian_to_ks, energy_ks


class SIM_PARAMS:
    """Simulation parameters matching Julia implementation"""
    method = 'RK45'  # Runge-Kutta 4/5 order
    atol = 1e-12
    rtol = 1e-13


def run_python_ks_numerical(times, a, e, mu, output_file):
    """Run Python KS numerical solution and save results to CSV."""
    print(f"Running Python propagate_ks_keplerian_dynamics()...")

    # Initial orbital elements: [a, e, i, RAAN, omega, M_0]
    # Matching Julia's assumptions: i=0, RAAN=0, omega=0, M_0=0
    oe_vec_0 = [a, e, 0.0, 0.0, 0.0, 0.0]

    # Convert to Cartesian initial state
    r_vec_0, v_vec_0 = pk.par2ic(oe_vec_0, mu)
    x_vec_0 = np.concatenate([np.array(r_vec_0), np.array(v_vec_0)])

    # Convert to KS coordinates
    ks_state_0 = state_cartesian_to_ks(x_vec_0)
    h_0 = energy_ks(ks_state_0[:4], ks_state_0[4:8], mu)
    ks_state_augmented_0 = np.concatenate([ks_state_0, np.array([h_0])])

    # Run KS numerical propagation
    x_vec_traj, ks_state_augmented_traj, t_traj = propagate_ks_keplerian_dynamics(
        ks_state_augmented_0, times, SIM_PARAMS, mu
    )

    # Save to CSV format: time, x, y, z, vx, vy, vz
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y", "z", "vx", "vy", "vz"])
        for i, t in enumerate(times):
            x = x_vec_traj[i]
            writer.writerow([t, x[0], x[1], x[2], x[3], x[4], x[5]])

    print(f"  Saved {len(x_vec_traj)} states to {output_file}")


def main():
    """Main function to run Python KS numerical solution."""
    if len(sys.argv) != 6:
        print(
            "Usage: python run_python_ks_numerical.py <a> <e> <mu> <times_file> <output_file>"
        )
        sys.exit(1)

    a = float(sys.argv[1])
    e = float(sys.argv[2])
    mu = float(sys.argv[3])
    times_file = sys.argv[4]
    output_file = sys.argv[5]

    # Read times from file
    times = []
    with open(times_file, "r") as f:
        for line in f:
            times.append(float(line.strip()))

    # Run Python solution
    run_python_ks_numerical(times, a, e, mu, output_file)


if __name__ == "__main__":
    main()


