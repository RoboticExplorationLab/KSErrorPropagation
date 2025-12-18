using ForwardDiff
using LinearAlgebra
using SatelliteDynamics
using DifferentialEquations

include("ks_transform.jl")
include("cartesian_dynamics.jl")

function ks_dynamics(ks_state_full_scaled, sim_params, s_current_scaled)
    y_vec_scaled = ks_state_full_scaled[1:4]
    y_vec_prime_scaled = ks_state_full_scaled[5:8]
    h_scaled = ks_state_full_scaled[9]

    # Two-body dynamics
    y_vec_pprime_scaled = (-h_scaled / 2.0) * y_vec_scaled
    h_prime_scaled = 0.0

    # J2 + drag
    if sim_params.add_perturbations
        x_vec_scaled = state_ks_to_cartesian(ks_state_full_scaled[1:8])
        x_vec = [x_vec_scaled[1:3] * sim_params.r_scale; x_vec_scaled[4:6] * sim_params.v_scale]
        t_current_scaled = ks_state_full_scaled[10]
        t_current = t_current_scaled * sim_params.t_scale
        epoch = sim_params.epoch_0 + t_current

        a_J2_vec = cartesian_J2_perturbation(x_vec, t_current, sim_params)
        a_drag_vec = cartesian_drag_perturbation(x_vec, t_current, epoch, sim_params)
        a_pert_vec = a_J2_vec + a_drag_vec
        a_pert_vec_scaled = a_pert_vec / sim_params.a_scale

        y_vec_pprime_scaled = y_vec_pprime_scaled + (y_vec_scaled'y_vec_scaled / 2.0) * (L(y_vec_scaled)' * [a_pert_vec_scaled; 0.0])
        h_prime_scaled = h_prime_scaled - 2 * y_vec_prime_scaled' * L(y_vec_scaled)' * [a_pert_vec_scaled; 0.0]
    end

    return [y_vec_prime_scaled; y_vec_pprime_scaled; h_prime_scaled]
end

function ks_dynamics!(ks_state_full_prime_scaled, ks_state_full_scaled, sim_params, s_current_scaled)
    # KS dynamics
    ks_state_augmented_prime_scaled = ks_dynamics(ks_state_full_scaled, sim_params, s_current_scaled)

    # Time derivative: dt/ds = r = y'y
    y_vec_scaled = ks_state_full_scaled[1:4]
    r_vec_norm_scaled = y_vec_scaled'y_vec_scaled
    t_prime_scaled = r_vec_norm_scaled

    ks_state_full_prime_scaled .= [ks_state_augmented_prime_scaled; t_prime_scaled]
end

function propagate_ks_dynamics(x_vec_0, times, sim_params)
    # Scaling
    x_vec_0_scaled = [x_vec_0[1:3] / sim_params.r_scale; x_vec_0[4:6] / sim_params.v_scale]
    times_scaled = times / sim_params.t_scale
    ks_state_0_scaled = state_cartesian_to_ks(x_vec_0_scaled)
    h_0_scaled = energy_ks(ks_state_0_scaled[1:4], ks_state_0_scaled[5:8], sim_params.GM / sim_params.GM_scale)
    ks_state_augmented_0_scaled = [ks_state_0_scaled; h_0_scaled]
    ks_state_full_0_scaled = [ks_state_augmented_0_scaled; times_scaled[1]]

    # Pre-allocate trajectory storage
    ks_state_augmented_traj_scaled = [zeros(9) for k = 1:length(times_scaled)]
    ks_state_augmented_traj_scaled[1] .= ks_state_augmented_0_scaled
    times_traj_scaled = zeros(length(times_scaled))
    times_traj_scaled[1] = times_scaled[1]

    # Callback to save states at specific real time points
    function condition!(out, ks_state_full_scaled, s_current_scaled, sol)
        t_current_scaled = ks_state_full_scaled[10]  # Real time is the 10th element
        out .= times_scaled .- t_current_scaled
    end

    function affect!(sol, idx)
        if idx <= length(times_scaled)
            ks_state_augmented_traj_scaled[idx] .= sol.u[1:9]
            times_traj_scaled[idx] = sol.u[10]
        end
    end

    cb = VectorContinuousCallback(condition!, affect!, length(times_scaled))

    # Estimate fictitious time span so we don't integrate for too long
    y_vec_0_scaled = ks_state_augmented_0_scaled[1:4]
    r_vec_norm_0_scaled = y_vec_0_scaled'y_vec_0_scaled
    s_0_scaled = 0.0
    s_end_scaled = (times_scaled[end] - times_scaled[1]) / r_vec_norm_0_scaled * 2.0  # add margin

    # ODE problem
    prob = ODEProblem(ks_dynamics!, ks_state_full_0_scaled, (s_0_scaled, s_end_scaled), sim_params)
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, callback=cb)

    # Unscaling
    x_vec_traj_scaled = [state_ks_to_cartesian(ks_state_augmented_traj_scaled[k][1:8]) for k = 1:length(times_scaled)]
    x_vec_traj = [[x_vec_traj_scaled[k][1:3] * sim_params.r_scale; x_vec_traj_scaled[k][4:6] * sim_params.v_scale] for k = 1:length(times_scaled)]
    times_traj = times_traj_scaled * sim_params.t_scale
    return x_vec_traj, times_traj
end

function propagate_ks_relative_dynamics(x_vec_chief_0, x_vec_deputy_0, times, sim_params)
    # Scaling
    x_vec_chief_0_scaled = [x_vec_chief_0[1:3] / sim_params.r_scale; x_vec_chief_0[4:6] / sim_params.v_scale]
    x_vec_deputy_0_scaled = [x_vec_deputy_0[1:3] / sim_params.r_scale; x_vec_deputy_0[4:6] / sim_params.v_scale]
    times_scaled = times / sim_params.t_scale

    # Convert Cartesian states to KS states
    ks_state_chief_0_scaled = state_cartesian_to_ks_via_newton_method(x_vec_chief_0_scaled)
    ks_state_deputy_0_scaled = state_cartesian_to_ks_via_newton_method(x_vec_deputy_0_scaled; y_vec_near=ks_state_chief_0_scaled[1:4])
    ks_state_rel_0_scaled = ks_state_deputy_0_scaled .- ks_state_chief_0_scaled

    # Compute energy
    h_chief_0_scaled = energy_ks(ks_state_chief_0_scaled[1:4], ks_state_chief_0_scaled[5:8], sim_params.GM / sim_params.GM_scale)
    h_deputy_0_scaled = energy_ks(ks_state_deputy_0_scaled[1:4], ks_state_deputy_0_scaled[5:8], sim_params.GM / sim_params.GM_scale)
    h_rel_0_scaled = h_deputy_0_scaled - h_chief_0_scaled

    # ForwardDiff function for KS perturbed dynamics
    function f_function(ks_state_augmented_scaled, t_current_scaled)
        y_vec_scaled = ks_state_augmented_scaled[1:4]
        y_vec_prime_scaled = ks_state_augmented_scaled[5:8]
        h_scaled = ks_state_augmented_scaled[9]

        # Two-body dynamics
        y_vec_pprime_scaled = (-h_scaled / 2.0) * y_vec_scaled
        h_prime_scaled = 0.0

        # J2 + drag
        if sim_params.add_perturbations
            x_vec_scaled = state_ks_to_cartesian(ks_state_augmented_scaled[1:8])
            x_vec = [x_vec_scaled[1:3] * sim_params.r_scale; x_vec_scaled[4:6] * sim_params.v_scale]
            t_current = t_current_scaled * sim_params.t_scale
            epoch = sim_params.epoch_0 + t_current

            a_J2_vec = cartesian_J2_perturbation(x_vec, t_current, sim_params)
            a_drag_vec = cartesian_drag_perturbation(x_vec, t_current, epoch, sim_params)
            a_pert_vec = a_J2_vec + a_drag_vec
            a_pert_vec_scaled = a_pert_vec / sim_params.a_scale

            y_vec_pprime_scaled = y_vec_pprime_scaled + (y_vec_scaled'y_vec_scaled / 2.0) * (L(y_vec_scaled)' * [a_pert_vec_scaled; 0.0])
            h_prime_scaled = h_prime_scaled - 2 * y_vec_prime_scaled' * L(y_vec_scaled)' * [a_pert_vec_scaled; 0.0]
        end

        return [y_vec_prime_scaled; y_vec_pprime_scaled; h_prime_scaled]
    end

    # Perturbed relative dynamics
    function ks_perturbed_relative_dynamics!(z_prime, z, p, s)
        ks_state_agumented_chief = z[1:9]
        ks_state_agumented_rel = z[10:18]

        # Nonlinear dynamics for chief
        t_chief_current = z[end-1]
        ks_state_agumented_chief_prime = f_function(ks_state_agumented_chief, t_chief_current)

        # Linearize around chief position
        ∂f∂ks_state_agumented_chief = ForwardDiff.jacobian(x -> f_function(x, t_chief_current), ks_state_agumented_chief)

        # Linear relative dynamics
        ks_state_agumented_rel_prime = ∂f∂ks_state_agumented_chief * ks_state_agumented_rel
        ks_state_agumented_deputy = ks_state_agumented_chief .+ ks_state_agumented_rel

        # Time derivatives
        t_chief_prime = ks_state_agumented_chief[1:4]'ks_state_agumented_chief[1:4]
        t_deputy_prime = ks_state_agumented_deputy[1:4]'ks_state_agumented_deputy[1:4]

        z_prime .= [ks_state_agumented_chief_prime; ks_state_agumented_rel_prime; t_chief_prime; t_deputy_prime]
    end

    # Initial condition
    z_0 = [ks_state_chief_0_scaled; h_chief_0_scaled; ks_state_rel_0_scaled; h_rel_0_scaled; times_scaled[1]; times_scaled[1]]

    # Pre-allocate trajectory storage
    ks_state_agumented_chief_traj_scaled = [zeros(9) for k = 1:length(times_scaled)]
    ks_state_agumented_chief_traj_scaled[1] .= [ks_state_chief_0_scaled; h_chief_0_scaled]
    ks_state_agumented_deputy_traj_scaled = [zeros(9) for k = 1:length(times_scaled)]
    ks_state_agumented_deputy_traj_scaled[1] .= [ks_state_deputy_0_scaled; h_deputy_0_scaled]
    t_deputy_traj_scaled = zeros(length(times_scaled))
    t_deputy_traj_scaled[1] = times_scaled[1]
    t_chief_traj_scaled = zeros(length(times_scaled))
    t_chief_traj_scaled[1] = times_scaled[1]

    ############################ CALLBACK FUNCTIONS ############################
    # Callback to save states at specific real time points
    # SOLVER HAS ISSUES WHEN RELATIVE STATE IS ZERO
    function condition_separate!(out, z, s, sol)
        t_chief_current_scaled = z[end-1]
        t_deputy_current_scaled = z[end]
        out .= [times_scaled .- t_chief_current_scaled; times_scaled .- t_deputy_current_scaled]
    end

    function affect_separate!(sol, idx)
        if idx <= length(times_scaled)
            # Chief timepoint
            ks_state_agumented_chief_traj_scaled[idx] .= sol.u[1:9]
            t_chief_traj_scaled[idx] = sol.u[end-1]
        else
            # Deputy timepoint
            idx -= length(times_scaled)
            ks_state_agumented_deputy_traj_scaled[idx] .= sol.u[10:18] .+ sol.u[1:9]
            t_deputy_traj_scaled[idx] = sol.u[end]
        end
    end

    function condition_together!(out, z, s, sol)
        t_chief_current_scaled = z[end-1]
        out .= times_scaled .- t_chief_current_scaled
    end

    function affect_together!(sol, idx)
        if idx <= length(times_scaled)
            # Chief timepoint
            ks_state_agumented_chief_traj_scaled[idx] .= sol.u[1:9]
            t_chief_traj_scaled[idx] = sol.u[end-1]

            # Deputy timepoint
            ks_state_agumented_deputy_traj_scaled[idx] .= sol.u[10:18] .+ sol.u[1:9]
            t_deputy_traj_scaled[idx] = sol.u[end]
        end
    end

    if norm(ks_state_rel_0_scaled[1:4]) > 1e-14
        cb = VectorContinuousCallback(condition_separate!, affect_separate!, 2 * length(times_scaled))
    else
        cb = VectorContinuousCallback(condition_together!, affect_together!, length(times_scaled))
    end
    ############################################################################

    # Estimate fictitious time span
    r_vec_norm_0_scaled = norm(x_vec_chief_0_scaled[1:3])
    s_0_scaled = 0.0
    s_end_scaled = (times_scaled[end] - times_scaled[1]) / r_vec_norm_0_scaled * 2.0  # add margin

    # Numerical solver
    prob = ODEProblem(ks_perturbed_relative_dynamics!, z_0, (s_0_scaled, s_end_scaled))
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, callback=cb)

    # Convert KS states to Cartesian
    x_vec_traj_chief_scaled = [state_ks_to_cartesian(ks_state_agumented_chief_traj_scaled[k][1:8]) for k = 1:length(times_scaled)]
    x_vec_traj_chief = [[x_vec_traj_chief_scaled[k][1:3] * sim_params.r_scale; x_vec_traj_chief_scaled[k][4:6] * sim_params.v_scale] for k = 1:length(times_scaled)]
    x_vec_traj_deputy_scaled = [state_ks_to_cartesian(ks_state_agumented_deputy_traj_scaled[k][1:8]) for k = 1:length(times_scaled)]
    x_vec_traj_deputy = [[x_vec_traj_deputy_scaled[k][1:3] * sim_params.r_scale; x_vec_traj_deputy_scaled[k][4:6] * sim_params.v_scale] for k = 1:length(times_scaled)]
    x_vec_traj_rel = [x_vec_traj_deputy[k][1:6] .- x_vec_traj_chief[k][1:6] for k = 1:length(times_scaled)]
    t_chief_traj = t_chief_traj_scaled * sim_params.t_scale
    t_deputy_traj = t_deputy_traj_scaled * sim_params.t_scale
    return x_vec_traj_chief, x_vec_traj_rel, t_chief_traj, t_deputy_traj
end