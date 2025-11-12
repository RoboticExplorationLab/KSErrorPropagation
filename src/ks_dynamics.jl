"""
KS (Kustaanheimo-Stiefel) dynamics with gravity only (no J2, no drag).
"""

using LinearAlgebra
using SatelliteDynamics
using DifferentialEquations

const SD = SatelliteDynamics

include("ks_transform.jl")

"""
    ks_gravity(ks_state_augmented, s, GM)

KS dynamics with gravity only (point mass gravity).

# Arguments
- `ks_state_augmented`: KS augmented state vector [y_vec; y_vec_prime; h]
- `s`: fictitious time (not used but required for ODE interface)
- `GM`: gravitational parameter (default: GM_EARTH)

# Returns
- `ks_state_augmented_prime`: KS augmented state vector derivative [y_vec_prime; y_vec_pprime; h_prime]
"""
function ks_gravity(ks_state_augmented, s, GM=SD.GM_EARTH)
    y_vec = ks_state_augmented[1:4]
    y_vec_prime = ks_state_augmented[5:8]
    h = ks_state_augmented[9]
    
    # Unperturbed KS dynamics (gravity only)
    y_vec_pprime = (-h / 2.0) * y_vec
    h_prime = 0.0
    
    return [y_vec_prime; y_vec_pprime; h_prime]
end

"""
    ks_gravity!(ks_state_augmented_prime, ks_state_augmented, p, s)

In-place version for DifferentialEquations.jl.

# Arguments
- `ks_state_augmented_prime`: KS augmented state vector derivative [y_vec_prime; y_vec_pprime; h_prime]
- `ks_state_augmented`: KS augmented state vector [y_vec; y_vec_prime; h]
- `p`: parameters (GM)
- `s`: fictitious time
"""
# function ks_gravity!(ks_state_augmented_prime, ks_state_augmented, p, s)
#     GM = p isa Number ? p : p[1]
#     ks_state_augmented_prime .= ks_gravity(ks_state_augmented, s, GM)
# end
function ks_gravity!(ks_state_full_prime, ks_state_full, p, s)
    GM = p isa Number ? p : p[1]
    ks_state_augmented = ks_state_full[1:9]
    
    # KS dynamics
    ks_state_augmented_prime = ks_gravity(ks_state_augmented, s, GM)
    
    # Time derivative: dt/ds = r = y'y
    y_vec = ks_state_augmented[1:4]
    r_vec_norm = y_vec'y_vec
    t_prime = r_vec_norm
    
    ks_state_full_prime .= [ks_state_augmented_prime; t_prime]
end

"""
    propagate_ks_keplerian_orbit(ks_state_augmented_0, times, sim_params, GM)

Propagate KS state with time tracking, handling the transformation between
real time t and fictitious time s where dt = r * ds and r = y'y.

# Arguments
- `ks_state_augmented_0`: initial KS augmented state [y_vec; y_vec_prime; h]
- `times`: array of times to save at
- `sim_params`: simulation parameters
- `GM`: gravitational parameter

# Returns
- `x_vec_traj_ks`: array of Cartesian states [r_vec; v_vec] at each time
- `ks_state_augmented_traj`: array of KS augmented states [y_vec; y_vec_prime; h] at each time
"""
function propagate_ks_keplerian_orbit(ks_state_augmented_0, times, sim_params, GM=SD.GM_EARTH)
    ks_state_full_0 = [ks_state_augmented_0; times[1]] # KS full state vector: [y_vec; y_vec_prime; h; t]
    y_vec_0 = ks_state_augmented_0[1:4]
    r_vec_norm_0 = y_vec_0'y_vec_0
    s_0 = 0.0
    s_end = (times[end] - times[1]) / r_vec_norm_0  # Rough estimate: s_end ~ t_end / r_vec_norm_0
    
    prob = ODEProblem(ks_gravity!, ks_state_full_0, (s_0, s_end), GM)
    sol = solve(prob, sim_params.integrator(); abstol=sim_params.abstol, reltol=sim_params.reltol)
    
    # Find states at desired times by interpolation
    x_vec_traj_ks = Vector{Vector{Float64}}()
    ks_state_augmented_traj = Vector{Vector{Float64}}()
    
    for t_target in times
        # Find s such that t(s) = t_target
        s_low = s_0
        s_high = s_end
        s_guess = s_0
        
        for _ in 1:100
            ks_state_full_guess = sol(s_guess)
            t_guess = ks_state_full_guess[10]
            
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
        
        ks_state_full_final = sol(s_guess)
        ks_state_augmented_final = ks_state_full_final[1:9]
        x_vec_final_ks = state_ks_to_cartesian(ks_state_augmented_final[1:8])
        push!(x_vec_traj_ks, x_vec_final_ks)
        push!(ks_state_augmented_traj, ks_state_augmented_final)
    end
    
    return x_vec_traj_ks, ks_state_augmented_traj
end

