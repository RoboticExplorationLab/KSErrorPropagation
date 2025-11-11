"""
KS (Kustaanheimo-Stiefel) dynamics with gravity only (no J2, no drag).
"""

using LinearAlgebra
using SatelliteDynamics
using DifferentialEquations

const SD = SatelliteDynamics

include("ks_transform.jl")

"""
    ks_gravity_dynamics(p_state, s, GM)

KS dynamics with gravity only (point mass gravity).

# Arguments
- `p_state`: 9-vector KS state [p; p_prime; h] where:
  - `p`: 4-vector KS position
  - `p_prime`: 4-vector KS velocity (wrt fictitious time s)
  - `h`: energy parameter
- `s`: fictitious time (not used but required for ODE interface)
- `GM`: gravitational parameter (default: GM_EARTH)

# Returns
- `p_state_dot`: 9-vector derivative [p_prime; p_pprime; h_prime]

The KS dynamics for gravity only are:
- p' = p_prime (already in state)
- p'' = (-h/2) * p
- h' = 0

where h = (GM - 2*(p_prime'p_prime)) / (p'p)
"""
function ks_gravity_dynamics(p_state, s, GM=SD.GM_EARTH)
    p = p_state[1:4]
    p_prime = p_state[5:8]
    h = p_state[9]
    
    # Unperturbed KS dynamics (gravity only)
    p_pprime = (-h / 2) * p
    h_prime = 0.0
    
    return [p_prime; p_pprime; h_prime]
end

"""
    ks_gravity_dynamics!(p_state_dot, p_state, p, s)

In-place version for DifferentialEquations.jl.

# Arguments
- `p_state_dot`: output derivative vector
- `p_state`: KS state vector [p; p_prime; h]
- `p`: parameters (GM)
- `s`: fictitious time
"""
function ks_gravity_dynamics!(p_state_dot, p_state, p, s)
    GM = p isa Number ? p : p[1]
    p_state_dot .= ks_gravity_dynamics(p_state, s, GM)
end

"""
    propagate_ks_with_time(p_state_0, t0, times, GM)

Propagate KS state with time tracking, handling the transformation between
real time t and fictitious time s where dt = r * ds and r = p'p.

# Arguments
- `p_state_0`: initial KS state [p; p_prime; h]
- `t0`: initial real time
- `times`: array of real times to save at
- `GM`: gravitational parameter

# Returns
- `x_traj`: array of Cartesian states [r; v] at each time
- `p_state_traj`: array of KS states [p; p_prime; h] at each time

The state vector includes time: z = [p; p_prime; h; t]
"""
function propagate_ks_with_time(p_state_0, t0, times, GM=SD.GM_EARTH)
    # Extended state: [p; p_prime; h; t] where t is real time
    z0 = [p_state_0; t0]
    
    function ks_dynamics_with_time!(zdot, z, p, s)
        p_state = z[1:9]
        
        # KS dynamics
        p_state_dot = ks_gravity_dynamics(p_state, s, GM)
        
        # Time derivative: dt/ds = r = p'p
        p_vec = p_state[1:4]
        r = p_vec'p_vec
        tdot = r
        
        zdot .= [p_state_dot; tdot]
    end
    
    # Integrate in fictitious time s
    # Estimate s_end based on approximate orbital period
    p = p_state_0[1:4]
    r_initial = p'p
    # Rough estimate: s_end ~ t_end / r_initial
    s0 = 0.0
    s_end = (times[end] - t0) / r_initial * 2.0  # Add margin
    
    prob = ODEProblem(ks_dynamics_with_time!, z0, (s0, s_end), GM)
    
    # Integrate with dense output for interpolation
    sol = solve(prob, Tsit5(); abstol=1e-12, reltol=1e-13, dense=true)
    
    # Find states at desired times by interpolation
    x_traj = Vector{Vector{Float64}}()
    p_state_traj = Vector{Vector{Float64}}()
    
    for t_target in times
        # Find s such that t(s) = t_target
        # Use bisection or linear search
        s_low = s0
        s_high = s_end
        s_guess = s0
        
        # Simple linear search (could be improved with bisection)
        for _ in 1:100
            z_guess = sol(s_guess)
            t_guess = z_guess[10]
            
            if abs(t_guess - t_target) < 1e-6
                break
            elseif t_guess < t_target
                s_low = s_guess
                s_guess = (s_guess + s_high) / 2
            else
                s_high = s_guess
                s_guess = (s_low + s_guess) / 2
            end
        end
        
        z_final = sol(s_guess)
        p_state = z_final[1:9]
        x = state_ks_to_inertial(p_state[1:8])
        push!(x_traj, x)
        push!(p_state_traj, p_state)
    end
    
    return x_traj, p_state_traj
end

