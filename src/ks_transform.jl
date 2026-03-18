using LinearAlgebra

function L(y_vec)
    y1, y2, y3, y4 = y_vec

    return [y1 -y2 -y3 y4
        y2 y1 -y4 -y3
        y3 y4 y1 y2
        y4 -y3 y2 -y1]
end

function R(y_vec)
    y1, y2, y3, y4 = y_vec

    return [y1 y2 y3 -y4
        y2 -y1 y4 y3
        y3 -y4 -y1 -y2
        y4 y3 -y2 y1]
end

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

function position_ks_to_cartesian(y_vec)
    return (L(y_vec)*y_vec)[1:3]
end

function velocity_cartesian_to_ks(y_vec, v_vec)
    y1, y2, y3, y4 = y_vec
    v1, v2, v3 = v_vec

    y1_prime = 0.5 * (y1 * v1 + y2 * v2 + y3 * v3)
    y2_prime = 0.5 * (-y2 * v1 + y1 * v2 + y4 * v3)
    y3_prime = 0.5 * (-y3 * v1 - y4 * v2 + y1 * v3)
    y4_prime = 0.5 * (y4 * v1 - y3 * v2 + y2 * v3)

    return [y1_prime, y2_prime, y3_prime, y4_prime]
end

function velocity_ks_to_cartesian(y_vec, y_vec_prime)
    return ((2.0/y_vec'y_vec)*(L(y_vec)*y_vec_prime))[1:3]
end

function state_ks_to_cartesian(ks_state)
    y_vec = ks_state[1:4]
    y_vec_prime = ks_state[5:8]

    r_vec = position_ks_to_cartesian(y_vec)
    v_vec = velocity_ks_to_cartesian(y_vec, y_vec_prime)

    return [r_vec; v_vec]
end

function state_cartesian_to_ks(x_vec)
    r_vec = x_vec[1:3]
    v_vec = x_vec[4:6]

    y_vec = position_cartesian_to_ks(r_vec)
    y_vec_prime = velocity_cartesian_to_ks(y_vec, v_vec)

    return [y_vec; y_vec_prime]
end

function energy_ks(y_vec, y_vec_prime, GM)
    return (GM - 2 * (y_vec_prime'y_vec_prime)) / (y_vec'y_vec)
end

function energy_ks_from_cartesian(x_vec, GM)
    ks_state = state_cartesian_to_ks(x_vec)
    return energy_ks(ks_state[1:4], ks_state[5:8], GM)
end

function position_cartesian_to_ks_via_newton_method(r_vec; y_vec_near=[1.0; 0.0; 0.0; 0.0], tol=1e-12, max_iter=100, return_verbose=false)
    # Projection matrix to project onto the Cartesian space
    Pi = [I(3) zeros(3, 1)]

    # KKT matrix
    function kkt_matrix(y_vec, λ)
        H = I(4) .+ (2 * R(Pi' * λ))'
        A = 2 * Pi * L(y_vec)
        return [H A';
            A zeros(3, 3)]
    end

    # Right-hand side vector
    function rhs(r_vec, y_vec_near, y_vec, λ)
        b1 = (y_vec .- y_vec_near) .+ ((2 * Pi * L(y_vec))'λ)
        b2 = Pi * L(y_vec) * y_vec .- r_vec
        b = [b1; b2]
        return -b
    end

    z = [y_vec_near; zeros(3)]
    i = 0
    while i < max_iter
        i += 1
        A = kkt_matrix(z[1:4], z[5:7])
        b = rhs(r_vec, y_vec_near, z[1:4], z[5:7])
        Δz_vec = A \ b
        z += Δz_vec
        if norm(Δz_vec) < tol
            break
        end
    end

    if return_verbose
        return z, i
    end

    return z[1:4]
end

function state_cartesian_to_ks_via_newton_method(x_state; y_vec_near=[1.0; 0.0; 0.0; 0.0], tol=1e-12, max_iter=100)
    r_vec = x_state[1:3]
    v_vec = x_state[4:6]

    y_vec = position_cartesian_to_ks_via_newton_method(r_vec; y_vec_near=y_vec_near, tol=tol, max_iter=max_iter)
    y_vec_prime = velocity_cartesian_to_ks(y_vec, v_vec)

    return [y_vec; y_vec_prime]
end