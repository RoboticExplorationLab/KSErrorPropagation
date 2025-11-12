"""
KS (Kustaanheimo-Stiefel) transformation functions for converting between
Cartesian and KS coordinates.
"""

using LinearAlgebra

""" `L(y_vec)`
Given a 4-vector `y_vec`, corresponding to a KS position,
return the Left multiply Matrix L(y_vec) where col(r_vec,0) = L(y_vec)y_vec.
"""
function L(y_vec)
    y1, y2, y3, y4 = y_vec

    return [y1 -y2 -y3 y4
            y2 y1 -y4 -y3
            y3 y4 y1 y2
            y4 -y3 y2 -y1]
end

""" `position_cartesian_to_ks(r_vec)`
Given a 3-vector `r_vec`, corresponding to an cartesian position,
return the ks-transformed 4-vector `y_vec`.
"""
# This approach might break
# function position_cartesian_to_ks(r_vec)
#     r1, r2, r3 = r_vec
#     r_norm = sqrt(r_vec'r_vec)

#     y1 = 0.0
#     y4 = sqrt(0.5 * (r_norm + r1) - y1^2)
#     y2 = (r2 * y4 + r3 * y1) / (r_norm + r1)
#     y3 = (r3 * y1 - r2 * y4) / (r_norm + r1)
#     return [y1, y2, y3, y4]
# end
function position_cartesian_to_ks(r_vec)
    r1, r2, r3 = r_vec
    r_vec_norm = norm(r_vec)

    if r1 >= 0
        # Transform best suited for vectors with r1 >= 0
        y4 = 0.0
        y1 = sqrt(0.5 * (r_vec_norm + r1) - y4^2)
        y2 = (r2 * y1 + r3 * y4) / (r1 + r_vec_norm)
        y3 = (r3 * y1 - r2 * y4) / (r1 + r_vec_norm)
        return [y1, y2, y3, y4]
    else
        # Transform best suited for vectors with r1 < 0
        y3 = 0.0
        y2 = sqrt(0.5 * (r_vec_norm - r1) - y3^2)
        y1 = (r2 * y2 + r3 * y3) / (r_vec_norm - r1)
        y4 = (r3 * y2 - r2 * y3) / (r_vec_norm - r1)
        return [y1, y2, y3, y4]
    end
end

""" `position_ks_to_cartesian(y_vec)`
Given a 4-vector `y_vec`, corresponding to a KS position,
return the cartesian 3-vector `r_vec`.
"""
# function position_ks_to_cartesian(y_vec)
#     y1, y2, y3, y4 = y_vec

#     r1 = y1^2 - y2^2 - y3^2 + y4^2
#     r2 = 2.0 * (y1 * y2 - y3 * y4)
#     r3 = 2.0 * (y1 * y3 + y2 * y4)

#     return [r1, r2, r3]
# end
function position_ks_to_cartesian(y_vec)
    return (L(y_vec) * y_vec)[1:3]
end

""" `velocity_cartesian_to_ks(y_vec, v_vec)`
Given a 4-vector `y_vec`, corresponding to the ks position, and a
3-vector `v_vec`, corresponding to cartesian velocities, 
return the 4-vector of ks-transformed velocities `y_vec_prime`.
The cartesian velocities are wrt time (t), whereas the ks-transformed velocities
are wrt fictitious time (s), where dt = ||x||ds.
"""
function velocity_cartesian_to_ks(y_vec, v_vec)
    y1, y2, y3, y4 = y_vec
    v1, v2, v3 = v_vec

    y1_prime = 0.5 * (y1 * v1 + y2 * v2 + y3 * v3)
    y2_prime = 0.5 * (-y2 * v1 + y1 * v2 + y4 * v3)
    y3_prime = 0.5 * (-y3 * v1 - y4 * v2 + y1 * v3)
    y4_prime = 0.5 * (y4 * v1 - y3 * v2 + y2 * v3)

    return [y1_prime, y2_prime, y3_prime, y4_prime]
end

""" `velocity_ks_to_cartesian(y_vec, y_vec_prime)`
Given 4-vectors `y_vec` and `y_vec_prime`, corresponding to the ks transformed
position and velocity, respectively, return the 3-vector cartesian velocity `x`.
"""
# function velocity_ks_to_cartesian(y_vec, y_vec_prime)
#     y1, y2, y3, y4 = y_vec
#     y1_prime, y2_prime, y3_prime, y4_prime = y_vec_prime
#     yTy = y'y

#     v1 = (2.0 / yTy) * (y1 * y1_prime - y2 * y2_prime - y3 * y3_prime + y4 * y4_prime)
#     v2 = (2.0 / yTy) * (y2 * y1_prime + y1 * y2_prime - y4 * y3_prime - y3 * y4_prime)
#     v3 = (2.0 / yTy) * (y3 * y1_prime + y4 * y2_prime + y1 * y3_prime + y2 * y4_prime)

#     return [v1, v2, v3]
# end
function velocity_ks_to_cartesian(y_vec, y_vec_prime)
    return ((2.0 / y_vec'y_vec) * (L(y_vec) * y_vec_prime))[1:3]
end

""" `state_ks_to_cartesian(ks_state)`
Given an 8-vector `ks_state = [y_vec, y_vec_prime]` compute the 6-vector cartesian state `x_vec = [r_vec, v_vec]`.
"""
function state_ks_to_cartesian(ks_state)
    y_vec = ks_state[1:4]
    y_vec_prime = ks_state[5:8]

    r_vec = position_ks_to_cartesian(y_vec)
    v_vec = velocity_ks_to_cartesian(y_vec, y_vec_prime)

    return [r_vec; v_vec]
end

""" `state_cartesian_to_ks(x_vec)`
Given a 6-vector `x_vec = [r_vec, v_vec]` compute the 8-vector KS state `ks_state = [y_vec, y_vec_prime]`
"""
function state_cartesian_to_ks(x_vec)
    r_vec = x_vec[1:3]
    v_vec = x_vec[4:6]

    y_vec = position_cartesian_to_ks(r_vec)
    y_vec_prime = velocity_cartesian_to_ks(y_vec, v_vec)

    return [y_vec; y_vec_prime]
end

""" `energy_ks(y_vec, y_vec_prime, GM)`
Compute the KS energy parameter h from KS position and velocity.
"""
function energy_ks(y_vec, y_vec_prime, GM)
    return (GM - 2 * (y_vec_prime'y_vec_prime)) / (y_vec'y_vec)
end

