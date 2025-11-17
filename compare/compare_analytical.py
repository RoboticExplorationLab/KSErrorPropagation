"""
Compare Python and Julia analytical solutions from saved data files.
"""

import numpy as np
import json
import os




def compare_results(python_file, julia_file, output_file, a, e, mu):
    """Compare the saved results from Python and Julia."""
    print(f"\nComparing results...")

    # Load Python CSV
    python_data = np.loadtxt(python_file, delimiter=",", skiprows=1)
    python_times = python_data[:, 0]
    python_positions = python_data[:, 1:4]
    python_velocities = python_data[:, 4:7]

    # Load Julia CSV
    julia_data = np.loadtxt(julia_file, delimiter=",")
    julia_times = julia_data[:, 0]
    julia_positions = julia_data[:, 1:4]
    julia_velocities = julia_data[:, 4:7]

    # Check if times match
    if len(python_times) != len(julia_times):
        print(f"  Warning: Different number of time steps")
        print(f"    Python: {len(python_times)}, Julia: {len(julia_times)}")
        min_len = min(len(python_times), len(julia_times))
        python_times = python_times[:min_len]
        python_positions = python_positions[:min_len]
        python_velocities = python_velocities[:min_len]
        julia_times = julia_times[:min_len]
        julia_positions = julia_positions[:min_len]
        julia_velocities = julia_velocities[:min_len]

    # Compare positions
    pos_errors = np.linalg.norm(python_positions - julia_positions, axis=1)
    max_pos_error = np.max(pos_errors)
    mean_pos_error = np.mean(pos_errors)
    rms_pos_error = np.sqrt(np.mean(pos_errors**2))

    # Compare velocities
    vel_errors = np.linalg.norm(python_velocities - julia_velocities, axis=1)
    max_vel_error = np.max(vel_errors)
    mean_vel_error = np.mean(vel_errors)
    rms_vel_error = np.sqrt(np.mean(vel_errors**2))

    # Create comparison results
    comparison = {
        "python_file": python_file,
        "julia_file": julia_file,
        "parameters": {
            "a": float(a),
            "e": float(e),
            "mu": float(mu),
        },
        "num_steps": len(python_times),
        "position_errors": {
            "max_km": float(max_pos_error),
            "max_m": float(max_pos_error * 1000),
            "mean_km": float(mean_pos_error),
            "mean_m": float(mean_pos_error * 1000),
            "rms_km": float(rms_pos_error),
            "rms_m": float(rms_pos_error * 1000),
            "all_errors_km": pos_errors.tolist(),
            "all_errors_m": (pos_errors * 1000).tolist(),
        },
        "velocity_errors": {
            "max_km_s": float(max_vel_error),
            "max_m_s": float(max_vel_error * 1000),
            "mean_km_s": float(mean_vel_error),
            "mean_m_s": float(mean_vel_error * 1000),
            "rms_km_s": float(rms_vel_error),
            "rms_m_s": float(rms_vel_error * 1000),
            "all_errors_km_s": vel_errors.tolist(),
            "all_errors_m_s": (vel_errors * 1000).tolist(),
        },
        "sample_comparison": [],
    }

    # Add sample comparisons
    n_samples = min(10, len(python_times))
    indices = np.linspace(0, len(python_times) - 1, n_samples, dtype=int)
    for i in indices:
        r_py = float(np.linalg.norm(python_positions[i]))
        r_jl = float(np.linalg.norm(julia_positions[i]))
        v_py = float(np.linalg.norm(python_velocities[i]))
        v_jl = float(np.linalg.norm(julia_velocities[i]))
        comparison["sample_comparison"].append(
            {
                "time_s": float(python_times[i]),
                "python_r_km": r_py,
                "julia_r_km": r_jl,
                "r_error_m": float(pos_errors[i] * 1000),
                "python_v_km_s": v_py,
                "julia_v_km_s": v_jl,
                "v_error_m_s": float(vel_errors[i] * 1000),
            }
        )

    # Save comparison
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\nParameters:")
    print(f"  a = {comparison['parameters']['a']} km")
    print(f"  e = {comparison['parameters']['e']}")
    print(f"  μ = {comparison['parameters']['mu']} km³/s²")
    print(f"\nPosition Errors:")
    print(f"  Max:  {max_pos_error:.6e} km ({max_pos_error*1000:.6e} m)")
    print(f"  Mean: {mean_pos_error:.6e} km ({mean_pos_error*1000:.6e} m)")
    print(f"  RMS:  {rms_pos_error:.6e} km ({rms_pos_error*1000:.6e} m)")
    print(f"\nVelocity Errors:")
    print(f"  Max:  {max_vel_error:.6e} km/s ({max_vel_error*1000:.6e} m/s)")
    print(f"  Mean: {mean_vel_error:.6e} km/s ({mean_vel_error*1000:.6e} m/s)")
    print(f"  RMS:  {rms_vel_error:.6e} km/s ({rms_vel_error*1000:.6e} m/s)")
    print(f"\nResults saved to: {output_file}")

    return comparison


def main():
    """Main function to compare saved results."""
    GM_earth = 3.986e5  # km^3/s^2

    # Test cases
    test_cases = [
        {"name": "circular", "a": 7000, "e": 0.0},
        {"name": "low_eccentricity", "a": 7000, "e": 0.1},
        {"name": "high_eccentricity", "a": 70000, "e": 0.9},
    ]

    print("=" * 80)
    print("COMPARING PYTHON AND JULIA ANALYTICAL SOLUTIONS")
    print("=" * 80)

    # Get data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    for test_case in test_cases:
        print(f"\n{'-'*80}")
        print(f"Test Case: {test_case['name']}")
        print(f"  a = {test_case['a']} km, e = {test_case['e']}")
        print(f"{'-'*80}")

        a = test_case["a"]
        e = test_case["e"]

        # File names in data directory
        python_file = os.path.join(data_dir, f"results_python_{test_case['name']}.csv")
        julia_file = os.path.join(data_dir, f"results_julia_{test_case['name']}.csv")
        comparison_file = os.path.join(
            data_dir, f"comparison_{test_case['name']}.json"
        )

        # Check if files exist
        if not os.path.exists(python_file):
            print(f"  Warning: Python results file not found: {python_file}")
            print("  Run run_python_analytical.py first")
            continue

        if not os.path.exists(julia_file):
            print(f"  Warning: Julia results file not found: {julia_file}")
            print("  Run run_julia_analytical.py first")
            continue

        # Compare results
        comparison = compare_results(
            python_file, julia_file, comparison_file, a, e, GM_earth
        )

    print(f"\n{'='*80}")
    print("ALL COMPARISONS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
