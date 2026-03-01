#!/usr/bin/env julia

using LinearAlgebra
using SatelliteDynamics
using DifferentialEquations

const SIM_PARAMS = (
    num_orbits=1.0,
    sampling_time=30.0,
    integrator=Tsit5,
    abstol=1e-12,
    reltol=1e-13,
    max_pos_error_threshold=1.0,
    max_vel_error_threshold=0.01,
)

"""
    build_ks_relative_scenario(; num_orbits=1.0, dt=30.0)

Construct the initial chief/deputy Cartesian states and common time grid
matching the `test_keplerian_relative_dynamics_match.jl` scenario.
"""
function build_ks_relative_scenario(; num_orbits::Float64=1.0, dt::Float64=30.0)
    GM = SatelliteDynamics.GM_EARTH
    R_EARTH = SatelliteDynamics.R_EARTH

    sma_c = 750e3 + R_EARTH
    e_c = 0.0
    i_c = deg2rad(98.2)
    ω_c = deg2rad(0.0)
    Ω_c = deg2rad(0.0)
    M_c = deg2rad(0.0)

    sma_d = sma_c
    e_d = e_c + 0.3
    i_d = i_c + deg2rad(0.001)
    ω_d = ω_c
    Ω_d = Ω_c
    M_d = M_c + deg2rad(0.001)

    oe_vec_chief = [sma_c, e_c, i_c, Ω_c, ω_c, M_c]
    oe_vec_deputy = [sma_d, e_d, i_d, Ω_d, ω_d, M_d]

    x_vec_chief_0 = SatelliteDynamics.sOSCtoCART(oe_vec_chief; GM=GM, use_degrees=false)
    x_vec_deputy_0 = SatelliteDynamics.sOSCtoCART(oe_vec_deputy; GM=GM, use_degrees=false)

    T_orbital = 2π * sqrt(sma_c^3 / GM)
    times = collect(0.0:dt:num_orbits*T_orbital)

    return (; GM, x_vec_chief_0, x_vec_deputy_0, times, dt, num_orbits, T_orbital)
end

include("../src/cartesian_dynamics.jl")

function run_full_nonlinear_sim()
    scenario = build_ks_relative_scenario(; num_orbits=SIM_PARAMS.num_orbits, dt=SIM_PARAMS.sampling_time)

    println("Propagating chief in Cartesian coordinates...")
    x_vec_traj_chief_cart, t_traj_chief_cart = propagate_cartesian_keplerian_dynamics(
        scenario.x_vec_chief_0,
        scenario.times,
        SIM_PARAMS,
        scenario.GM)
    println("  Completed: ", length(x_vec_traj_chief_cart), " states")

    println("Propagating deputy in Cartesian coordinates...")
    x_vec_traj_deputy_cart, t_traj_deputy_cart = propagate_cartesian_keplerian_dynamics(
        scenario.x_vec_deputy_0,
        scenario.times,
        SIM_PARAMS,
        scenario.GM)
    println("  Completed: ", length(x_vec_traj_deputy_cart), " states")

    # Compute relative state from full nonlinear propagation
    x_vec_traj_rel_cart = [x_vec_traj_deputy_cart[k] .- x_vec_traj_chief_cart[k] for k = 1:length(scenario.times)]
    println("  Computed relative states from full nonlinear propagation")

    return Dict(
        "chief" => x_vec_traj_chief_cart,
        "relative" => x_vec_traj_rel_cart,
        "t_chief" => t_traj_chief_cart,
        "t_deputy" => t_traj_deputy_cart,
        "times" => scenario.times)
end

function write_results_csv(path, results)
    open(path, "w") do io
        header = join([
                "index", "time", "t_chief", "t_deputy",
                "chief_x", "chief_y", "chief_z", "chief_vx", "chief_vy", "chief_vz",
                "rel_x", "rel_y", "rel_z", "rel_vx", "rel_vy", "rel_vz"
            ], ",")
        println(io, header)
        for idx in eachindex(results["times"])
            row = Any[
                idx,
                results["times"][idx],
                results["t_chief"][idx],
                results["t_deputy"][idx],
                results["chief"][idx]...,
                results["relative"][idx]...
            ]
            println(io, join(row, ","))
        end
    end
end

function main()
    results = run_full_nonlinear_sim()
    data_dir = normpath(joinpath(@__DIR__, "..", "..", "compare", "data"))
    out_path_csv = joinpath(data_dir, "full_nonlinear_results.csv")
    write_results_csv(out_path_csv, results)
    println("Saved full nonlinear propagation results to $(out_path_csv)")
end

main()

