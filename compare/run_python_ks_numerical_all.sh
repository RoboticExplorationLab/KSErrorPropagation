#!/bin/bash
# Run Python KS numerical solution for all test cases

GM_EARTH=398600.0

# Test cases
declare -a test_cases=("circular:7000:0.0" "low_eccentricity:7000:0.1" "high_eccentricity:70000:0.9")

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$script_dir/data"
python_script="$script_dir/run_python_ks_numerical.py"

# Ensure data directory exists
mkdir -p "$data_dir"

for test_case in "${test_cases[@]}"; do
    IFS=':' read -r name a e <<< "$test_case"
    
    echo "============================================================"
    echo "Test Case: $name"
    echo "  a = $a km, e = $e"
    echo "============================================================"
    
    # Calculate orbital period and create times
    a_abs=$(echo "$a" | awk '{if ($1 < 0) print -$1; else print $1}')
    T=$(python3 -c "import math; print(2 * math.pi * math.sqrt($a_abs**3 / $GM_EARTH))")
    times_file="$data_dir/times_${name}.csv"
    
    # Generate times (2 orbital periods, 100 points)
    python3 -c "
import numpy as np
T = $T
times = np.linspace(0, 2*T, 100)
with open('$times_file', 'w') as f:
    for t in times:
        f.write(f'{t:.15e}\n')
"
    
    output_file="$data_dir/results_python_ks_numerical_${name}.csv"
    
    echo "Running Python KS solution..."
    python3 "$python_script" "$a" "$e" "$GM_EARTH" "$times_file" "$output_file"
    
    # Clean up times file
    rm -f "$times_file"
    
    echo ""
done

echo "============================================================"
echo "ALL PYTHON KS NUMERICAL SOLUTIONS COMPLETE"
echo "Results saved to: $data_dir"
echo "============================================================"


