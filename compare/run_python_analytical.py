"""
Run Python propagate_analytical_keplerian_dynamics() from utils.py and save results to data/ directory.
"""

import numpy as np
import sys
import os
import csv

# Add python/src to path to import utils
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "python", "src"))

from utils import propagate_analytical_keplerian_dynamics


def run_python_solution(times, a, e, mu, output_file):
    """Run Python propagate_analytical_keplerian_dynamics() and save results to CSV."""
    print(f"Running Python propagate_analytical_keplerian_dynamics()...")

    # Initial orbital elements: [a, e, i, RAAN, omega, M_0]
    # Matching Julia's assumptions: i=0, RAAN=0, omega=0, M_0=0
    oe_vec_0 = [a, e, 0.0, 0.0, 0.0, 0.0]

    # Run function
    x_vec_traj = propagate_analytical_keplerian_dynamics(oe_vec_0, times, mu)

    # Save to CSV format: time, x, y, z, vx, vy, vz
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y", "z", "vx", "vy", "vz"])
        for i, t in enumerate(times):
            x = x_vec_traj[i]
            writer.writerow([t, x[0], x[1], x[2], x[3], x[4], x[5]])

    print(f"  Saved {len(x_vec_traj)} states to {output_file}")


def main():
    """Main function to run Python analytical solution."""
    if len(sys.argv) != 6:  # script name + 5 arguments
        print(
            "Usage: python run_python_analytical.py <a> <e> <mu> <times_file> <output_file>"
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
    run_python_solution(times, a, e, mu, output_file)


if __name__ == "__main__":
    main()
