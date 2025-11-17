#!/usr/bin/env julia
"""
Julia script to run propagate_analytical_keplerian_dynamics and save results to CSV.
Called from Python with command-line arguments.
"""

using Pkg
using DelimitedFiles

# Get project root (assume script is in compare/ directory)
project_root = joinpath(@__DIR__, "..")
julia_project = joinpath(project_root, "julia")
Pkg.activate(julia_project)

include(joinpath(julia_project, "src", "utils.jl"))

# Parse command-line arguments
if length(ARGS) != 5
    error("Usage: julia run_julia_analytical.jl <a> <e> <mu> <times_file> <output_file>")
end

a = parse(Float64, ARGS[1])
e = parse(Float64, ARGS[2])
mu = parse(Float64, ARGS[3])
times_file = ARGS[4]
output_file = ARGS[5]

# Read times from file (one per line)
times_data = readdlm(times_file, Float64)
times = vec(times_data)  # Convert to 1D array

# Initial orbital elements: [a, e, i, RAAN, omega, M_0]
# Matching Python's assumptions: i=0, RAAN=0, omega=0, M_0=0
oe_vec_0 = [a, e, 0.0, 0.0, 0.0, 0.0]

# Run function
x_vec_traj = propagate_analytical_keplerian_dynamics(oe_vec_0, times, mu)

# Extract positions and velocities
n = length(x_vec_traj)

# Save to CSV format: time, x, y, z, vx, vy, vz
open(output_file, "w") do f
    for i in 1:n
        x = x_vec_traj[i]
        println(f, times[i], ",", x[1], ",", x[2], ",", x[3], ",", x[4], ",", x[5], ",", x[6])
    end
end

println("Saved ", n, " states to ", output_file)

