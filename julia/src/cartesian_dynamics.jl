using LinearAlgebra
using SatelliteDynamics

function cartesian_gravity(x_vec, t, GM)
    r_vec = x_vec[1:3]
    v_vec = x_vec[4:6]

    # Unperturbed Cartesian dynamics (gravity only)
    r_vec_norm = norm(r_vec)
    a_grav = -GM * r_vec / r_vec_norm^3

    return a_grav
end

function cartesian_J2_perturbation(x_vec, t, sim_params)
    r_vec = x_vec[1:3]
    r_vec_norm = norm(r_vec)
    a_J2 = -(3.0 / 2.0) * sim_params.J2 * (sim_params.GM / r_vec_norm^2) * (sim_params.R_EARTH / r_vec_norm)^2 * [
               (1 - 5 * (r_vec[3] / r_vec_norm)^2) * r_vec[1] / r_vec_norm,
               (1 - 5 * (r_vec[3] / r_vec_norm)^2) * r_vec[2] / r_vec_norm,
               (3 - 5 * (r_vec[3] / r_vec_norm)^2) * r_vec[3] / r_vec_norm]
    return a_J2
end

function cartesian_drag_perturbation(x_vec, t, epoch, sim_params)
    r_vec = x_vec[1:3]
    v_vec = x_vec[4:6]
    r_vec_norm = norm(r_vec)

    # Earth's rotation vector
    omega = [0.0, 0.0, sim_params.OMEGA_EARTH]

    # Velocity relative to the Earth's atmosphere (accounting for rotation)
    v_vec_rel = v_vec - cross(omega, r_vec)
    v_vec_rel_norm = norm(v_vec_rel)

    # Skip drag if velocity is negligible
    if v_vec_rel_norm < 1e-6
        return zeros(3)
    end

    # Use Harris-Priester atmospheric density model
    rho = SatelliteDynamics.density_harris_priester(epoch, r_vec)

    # Drag acceleration: a = -0.5 * (CD * A / m) * ρ * v² * v̂
    a_drag = -0.5 * (sim_params.CD * sim_params.A / sim_params.m) * rho * v_vec_rel_norm * v_vec_rel

    return a_drag
end

function cartesian_dynamics!(x_vec_dot_scaled, x_vec_scaled, sim_params, t_current_scaled)
    # Unscaling
    r_vec_scaled = x_vec_scaled[1:3]
    v_vec_scaled = x_vec_scaled[4:6]
    x_vec = [r_vec_scaled * sim_params.r_scale; v_vec_scaled * sim_params.v_scale]
    t_current = t_current_scaled * sim_params.t_scale

    # Two-body dynamics
    a_grav_scaled = cartesian_gravity(x_vec_scaled, t_current_scaled, sim_params.GM / sim_params.GM_scale)
    a_vec_scaled = a_grav_scaled

    # J2 + drag
    if sim_params.add_perturbations
        epoch = sim_params.epoch_0 + t_current
        a_J2_vec = cartesian_J2_perturbation(x_vec, t_current, sim_params)
        a_drag_vec = cartesian_drag_perturbation(x_vec, t_current, epoch, sim_params)
        a_vec_pert = a_J2_vec + a_drag_vec
        a_vec_pert_scaled = a_vec_pert / sim_params.a_scale
        a_vec_scaled = a_vec_scaled + a_vec_pert_scaled
    end

    x_vec_dot_scaled .= [v_vec_scaled; a_vec_scaled]
end

function propagate_cartesian_dynamics(x_vec_0, times, sim_params)
    # Scaling
    x_vec_0_scaled = [x_vec_0[1:3] / sim_params.r_scale; x_vec_0[4:6] / sim_params.v_scale]
    times_scaled = times / sim_params.t_scale

    # ODE problem 
    prob = ODEProblem(cartesian_dynamics!, x_vec_0_scaled, (times_scaled[1], times_scaled[end]), sim_params)
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol, saveat=times_scaled)

    # Unscaling
    x_vec_traj = [[sol.u[k][1:3] * sim_params.r_scale; sol.u[k][4:6] * sim_params.v_scale] for k = 1:length(sol.u)]
    times_traj = sol.t * sim_params.t_scale
    return x_vec_traj, times_traj
end