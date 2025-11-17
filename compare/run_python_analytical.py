"""
Run Python analytical_solution() from utils.py and save results to data/ directory.
"""

import numpy as np
import pykep as pk
import sys
import os
import csv

# Add python/src to path to import utils
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "python", "src"))

from utils import analytical_solution


def run_python_solution(times, a, e, mu, output_file):
    """Run Python analytical_solution() and save results to CSV."""
    print(f"Running Python analytical_solution()...")

    positions = []
    velocities = []
    times_actual = []

    a_abs = abs(a) if a < 0 else a
    el0 = [a_abs, e, 0.0, 0.0, 0.0, 0.0]
    r0, v0 = pk.par2ic(el0, mu)
    r0 = np.array(r0)
    v0 = np.array(v0)

    for t in times:
        try:
            # Get position from actual function
            x, y, z, r_vec, r_norm = analytical_solution(t, a, e, mu)

            # Get velocity
            rf, vf = pk.propagate_lagrangian(r0, v0, t, mu)

            positions.append(r_vec)
            velocities.append(np.array(vf))
            times_actual.append(float(t))
        except Exception as e:
            print(f"  Error at t={t:.1f} s: {e}")
            break

    # Save to CSV: time, x, y, z, vx, vy, vz
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y", "z", "vx", "vy", "vz"])
        for i, t in enumerate(times_actual):
            writer.writerow([t] + positions[i].tolist() + velocities[i].tolist())

    print(f"  Saved {len(positions)} states to {output_file}")


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
