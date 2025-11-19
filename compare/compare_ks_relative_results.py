#!/usr/bin/env python3

import math
from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_results(filename: str) -> dict:
    path = DATA_DIR / filename
    chief = []
    rel = []
    t_chief = []
    t_deputy = []
    times = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            t_chief.append(float(row["t_chief"]))
            t_deputy.append(float(row["t_deputy"]))
            chief.append(
                [
                    float(row["chief_x"]),
                    float(row["chief_y"]),
                    float(row["chief_z"]),
                    float(row["chief_vx"]),
                    float(row["chief_vy"]),
                    float(row["chief_vz"]),
                ]
            )
            rel.append(
                [
                    float(row["rel_x"]),
                    float(row["rel_y"]),
                    float(row["rel_z"]),
                    float(row["rel_vx"]),
                    float(row["rel_vy"]),
                    float(row["rel_vz"]),
                ]
            )

    return {
        "times": times,
        "t_chief": t_chief,
        "t_deputy": t_deputy,
        "chief": chief,
        "relative": rel,
    }


def vector_norm(vec):
    return math.sqrt(sum(v * v for v in vec))


def compare_two_results(results1: dict, results2: dict, name1: str, name2: str):
    """Compare two result sets and print summary statistics."""
    chief1 = results1["chief"]
    chief2 = results2["chief"]
    rel1 = results1["relative"]
    rel2 = results2["relative"]

    n_chief = min(len(chief1), len(chief2))
    n_rel = min(len(rel1), len(rel2))

    # Chief comparison
    max_pos = max_vel = 0.0
    idx_pos = idx_vel = 0

    for k in range(n_chief):
        pos_err = vector_norm([chief1[k][i] - chief2[k][i] for i in range(3)])
        vel_err = vector_norm([chief1[k][i] - chief2[k][i] for i in range(3, 6)])
        if pos_err > max_pos:
            max_pos = pos_err
            idx_pos = k + 1
        if vel_err > max_vel:
            max_vel = vel_err
            idx_vel = k + 1

    print(f"\nChief comparison ({name1} vs {name2}):")
    print(f"  Max position difference: {max_pos:.6e} m at step {idx_pos}")
    print(f"  Max velocity difference: {max_vel:.6e} m/s at step {idx_vel}")

    # Relative state comparison
    max_rel_pos = max_rel_vel = 0.0
    idx_rel_pos = idx_rel_vel = 0

    for k in range(n_rel):
        pos_err = vector_norm([rel1[k][i] - rel2[k][i] for i in range(3)])
        vel_err = vector_norm([rel1[k][i] - rel2[k][i] for i in range(3, 6)])
        if pos_err > max_rel_pos:
            max_rel_pos = pos_err
            idx_rel_pos = k + 1
        if vel_err > max_rel_vel:
            max_rel_vel = vel_err
            idx_rel_vel = k + 1

    print(f"\nRelative state comparison ({name1} vs {name2}):")
    print(f"  Max position difference: {max_rel_pos:.6e} m at step {idx_rel_pos}")
    print(f"  Max velocity difference: {max_rel_vel:.6e} m/s at step {idx_rel_vel}")


def main():
    print("=" * 80)
    print("KS RELATIVE DYNAMICS COMPARISON")
    print("=" * 80)

    print("\nLoading results...")
    local_results = load_results("local_results.csv")
    reference_results = load_results("reference_results.csv")
    full_nonlinear_results = load_results("full_nonlinear_results.csv")

    print(f"  Local KS: {len(local_results['times'])} time points")
    print(f"  Reference KS: {len(reference_results['times'])} time points")
    print(f"  Full Nonlinear: {len(full_nonlinear_results['times'])} time points")

    # Compare Local KS vs Reference KS
    print("\n" + "=" * 80)
    print("COMPARISON 1: Local KS vs Reference KS")
    print("=" * 80)
    compare_two_results(local_results, reference_results, "Local KS", "Reference KS")

    # Compare Local KS vs Full Nonlinear
    print("\n" + "=" * 80)
    print("COMPARISON 2: Local KS vs Full Nonlinear")
    print("=" * 80)
    compare_two_results(
        local_results, full_nonlinear_results, "Local KS", "Full Nonlinear"
    )

    # Compare Reference KS vs Full Nonlinear
    print("\n" + "=" * 80)
    print("COMPARISON 3: Reference KS vs Full Nonlinear")
    print("=" * 80)
    compare_two_results(
        reference_results, full_nonlinear_results, "Reference KS", "Full Nonlinear"
    )

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
