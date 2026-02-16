include(joinpath(@__DIR__, "default.jl"))

POSITION_UNCERTAINTIES = [1e0]
NUM_MC_SAMPLES = 100

function compute_velocity_uncertainty(σ_pos, sma, r_vec_0, GM)
    return 1e-3
end