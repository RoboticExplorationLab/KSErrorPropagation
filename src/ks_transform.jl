"""
KS (Kustaanheimo-Stiefel) transformation functions for converting between
Cartesian and KS coordinates.
"""

using LinearAlgebra

""" `position_inertial_to_ks(x)`
Given a 3-vector `x`, corresponding to an inertial position,
return the ks-transformed 4-vector `p`.
"""
function position_inertial_to_ks(x)
    x1, _, _ = x
    r = sqrt(x'x)

    if x1 >= 0
        return position_cart_to_ks_plus(x)
    else
        return position_cart_to_ks_minus(x)
    end
end

""" position_cart_to_ks_plus
Convert a vector to ks, uses the transform best suited for vectors with x1 > 0
"""
function position_cart_to_ks_plus(x)
    x1, x2, x3 = x
    r = sqrt(x'x)

    p4 = 0.0
    p1 = sqrt(0.5 * (r + x1) - p4^2)
    p2 = (x2 * p1 + x3 * p4) / (x1 + r)
    p3 = (x3 * p1 - x2 * p4) / (x1 + r)
    return [p1, p2, p3, p4]
end

""" position_cart_to_ks_minus
Convert a vector to ks, uses the transform best suited for vectors with x1 < 0
"""
function position_cart_to_ks_minus(x)
    x1, x2, x3 = x
    r = sqrt(x'x)

    p3 = 0.0
    p2 = sqrt(0.5 * (r - x1) - p3^2)
    p1 = (x2 * p2 + x3 * p3) / (r - x1)
    p4 = (x3 * p2 - x2 * p3) / (r - x1)
    return [p1, p2, p3, p4]
end

""" `position_ks_to_inertial()`
Given a 4-vector `p`, corresponding to a KS position,
return the inertial 3-vector `x`.
"""
function position_ks_to_inertial(p)
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]

    x1 = p1^2 - p2^2 - p3^2 + p4^2
    x2 = 2.0 * (p1 * p2 - p3 * p4)
    x3 = 2.0 * (p1 * p3 + p2 * p4)

    return [x1, x2, x3]
end

""" `velocity_inertial_to_ks(p, x_dot)`
Given a 4-vector `p`, corresponding to the ks position, and a
3-vector `x_dot`, corresponding to inertial velocities, 
return the 4-vector of ks-transformed velocities `p_prime`.
The inertial velocities are wrt real time (t), whereas the ks-transformed velocities
are wrt fictitious time (s), where dt = ||x||ds.
"""
function velocity_inertial_to_ks(p, x_dot)
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]

    x1_dot = x_dot[1]
    x2_dot = x_dot[2]
    x3_dot = x_dot[3]

    p1_prime = 0.5 * (p1 * x1_dot + p2 * x2_dot + p3 * x3_dot)
    p2_prime = 0.5 * (-p2 * x1_dot + p1 * x2_dot + p4 * x3_dot)
    p3_prime = 0.5 * (-p3 * x1_dot - p4 * x2_dot + p1 * x3_dot)
    p4_prime = 0.5 * (p4 * x1_dot - p3 * x2_dot + p2 * x3_dot)

    return [p1_prime, p2_prime, p3_prime, p4_prime]
end

""" `velocity_ks_to_inertial(p, p_prime)`
Given 4-vectors `p` and `p_prime`, corresponding to the ks transformed
position and velocity, respectively, return the 3-vector inertial velocity `x`.
"""
function velocity_ks_to_inertial(p, p_prime)
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]

    p1_prime = p_prime[1]
    p2_prime = p_prime[2]
    p3_prime = p_prime[3]
    p4_prime = p_prime[4]

    pTp = p'p

    x1_dot = (2.0 / pTp) * (p1 * p1_prime - p2 * p2_prime - p3 * p3_prime + p4 * p4_prime)
    x2_dot = (2.0 / pTp) * (p2 * p1_prime + p1 * p2_prime - p4 * p3_prime - p3 * p4_prime)
    x3_dot = (2.0 / pTp) * (p3 * p1_prime + p4 * p2_prime + p1 * p3_prime + p2 * p4_prime)

    return [x1_dot, x2_dot, x3_dot]
end

""" `state_ks_to_inertial(p_state)`
Given an 8-vector `p_state = [p, p_prime]` compute the 6-vector cartesian state `x_state = [x, x_dot]`.
"""
function state_ks_to_inertial(p_state)
    p = p_state[1:4]
    p_prime = p_state[5:8]

    x = position_ks_to_inertial(p)
    x_dot = velocity_ks_to_inertial(p, p_prime)

    return [x; x_dot]
end

""" `state_inertial_to_ks(x_state)`
Given a 6-vector `x_state = [x, x_dot]` compute the 8-vector KS state `p_state = [p, p_prime]`
"""
function state_inertial_to_ks(x_state)
    x = x_state[1:3]
    x_dot = x_state[4:6]

    p = position_inertial_to_ks(x)
    p_prime = velocity_inertial_to_ks(p, x_dot)

    return [p; p_prime]
end

""" `ks_L(p)`
Given a ks quaternion `p`, compute the Left multiply Matrix L(p) where p*r = L(p)r
"""
function ks_L(p)
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]

    return [p1 -p2 -p3 p4
        p2 p1 -p4 -p3
        p3 p4 p1 p2
        p4 -p3 p2 -p1]
end

""" `ks_h_energy(p, p_prime, GM)`
Compute the KS energy parameter h from KS position and velocity.
"""
function ks_h_energy(p, p_prime, GM)
    return (GM - 2 * (p_prime'p_prime)) / (p'p)
end

