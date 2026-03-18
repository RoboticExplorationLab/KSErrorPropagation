"""
Offline comparison of error propagation approaches.

Loads pre-computed approach results from out/ and Monte Carlo ground truth,
computes error metrics, prints a summary table to stdout, and generates
comparison plots in figs/.

Usage:
    julia scripts/compare_approaches.jl                     # uses config/default.jl
    julia scripts/compare_approaches.jl config/sanity_check.jl  # uses custom config

The out/ directory should contain NPZ files produced by
scripts/error_propagation_comparison.jl with the naming convention:
    {approach_id}_{orbit_id}_num_orbits{N}_oe_std_a{σ_a}m_std_e{σ_e}.npz
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots
using LinearAlgebra
using NPZ
using Printf
using SatelliteDynamics

const SD = SatelliteDynamics

include(joinpath(@__DIR__, "..", "src", "cartesian_dynamics.jl"))
include(joinpath(@__DIR__, "..", "src", "ks_dynamics.jl"))
include(joinpath(@__DIR__, "..", "src", "error_propagation.jl"))

# Load configuration (default or user-specified)
if length(ARGS) >= 1
    config_path = ARGS[1]
    if !isabspath(config_path)
        config_path = joinpath(@__DIR__, "..", config_path)
    end
    println("Loading config: ", config_path)
    include(config_path)
else
    include(joinpath(@__DIR__, "..", "config", "default.jl"))
end

# ── Approach registry ────────────────────────────────────────────────────────
# Each entry: (id, display_name, color, linestyle)
APPROACHES = [
    (id="lincov_cart", name="LinCov (Cartesian)",    color=:green,  linestyle=:dash),
    (id="ut_cart",     name="UT (Cartesian)",        color=:blue,   linestyle=:dot),
    (id="ckf_cart",    name="CKF (Cartesian)",       color=:red,    linestyle=:dashdot),
    (id="ckf_ks",      name="CKF (KS)",             color=:orange, linestyle=:dashdotdot),
    (id="ckf_ks_rel",  name="CKF (KS Relative)",    color=:purple, linestyle=:dashdotdot),
    (id="lincov_ks",   name="LinCov (KS)",          color=:brown,  linestyle=:dot),
    (id="stratified_ks", name="Stratified KS CKF",  color=:black,  linestyle=:dash),
]

# ── Directories ──────────────────────────────────────────────────────────────
out_dir  = joinpath(@__DIR__, "..", "out")
figs_dir = joinpath(@__DIR__, "..", "figs")
mkpath(figs_dir)

# ── Helper: load approach NPZ → (x_vec_traj, P_traj, times) ────────────────
fname_num(x) = isinteger(x) ? string(Int(x)) : string(x)
function load_approach(dir, approach_id, orbit, num_orbits, oe_std)
    filename = "$(approach_id)_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_std_e$(fname_num(oe_std[2])).npz"
    filepath = joinpath(dir, filename)
    if !isfile(filepath)
        return nothing
    end
    npz = NPZ.npzread(filepath)
    x_array   = npz["x"]          # (N, 6)
    P_array   = npz["P"]          # (N, 6, 6)
    timestamp = npz["timestamp"]  # (N,)
    N = size(x_array, 1)
    x_traj = [x_array[i, :] for i in 1:N]
    P_traj = [P_array[i, :, :] for i in 1:N]
    return (x_vec_traj=x_traj, P_traj=P_traj, times=timestamp)
end

# ── Main comparison loop ─────────────────────────────────────────────────────
println("="^80)
println("OFFLINE APPROACH COMPARISON")
println("="^80)
println("Reading results from: ", out_dir)
println("Saving figures to:    ", figs_dir)

for orbit in TEST_ORBITS
    sma = orbit.a
    r_vec_0 = SD.sOSCtoCART([sma, orbit.e, orbit.i, orbit.RAAN, orbit.omega, orbit.M];
        GM=SIM_PARAMS.GM, use_degrees=false)[1:3]

    for oe_std in OE_INITIAL_STD_SCENARIOS
        for num_orbits in NUM_ORBITS_LIST
            println("\n" * "="^80)
            println("SCENARIO: ", orbit.name)
            println("  OE initial std (σ_a) = ", oe_std[1], " m  |  orbits = ", num_orbits)
            println("="^80)

            # Load Monte Carlo ground truth
            mc = load_approach(out_dir, "mc", orbit, num_orbits, oe_std)
            if mc === nothing
                println("  ✗ Monte Carlo ground truth not found — skipping scenario")
                continue
            end
            x_mc = mc.x_vec_traj
            P_mc = mc.P_traj
            times = mc.times

            # Load each approach and compute metrics
            loaded   = []  # (approach_entry, data, metrics)
            for ap in APPROACHES
                data = load_approach(out_dir, ap.id, orbit, num_orbits, oe_std)
                if data === nothing
                    println("  ✗ ", ap.name, " — data not found")
                    continue
                end
                met = error_metrics(x_mc, P_mc, data.x_vec_traj, data.P_traj)
                push!(loaded, (ap=ap, data=data, metrics=met))
                println("  ✓ ", ap.name, " — loaded (", length(data.x_vec_traj), " states)")
            end

            if isempty(loaded)
                println("  No approach data found — skipping scenario")
                continue
            end

            # ── Print summary table ──────────────────────────────────────
            N = minimum(vcat(length(x_mc), [length(l.data.x_vec_traj) for l in loaded]))

            println("\n" * "-"^100)
            @printf("  %-25s  %12s  %12s  %12s  %12s\n",
                "Approach", "Pos RMSE (m)", "Vel RMSE (m/s)", "σ_pos RMSE (m)", "σ_vel RMSE (m/s)")
            println("  " * "-"^97)
            for l in loaded
                m = l.metrics
                @printf("  %-25s  %12.4f  %12.6f  %12.4f  %12.6f\n",
                    l.ap.name, m.pos_rmse, m.vel_rmse, m.pos_uncertainty_rmse, m.vel_uncertainty_rmse)
            end
            println("-"^100)

            # ── KL divergence ────────────────────────────────────────────
            kl_data = []
            for l in loaded
                kl = [gaussian_kl_divergence(l.ap.name, times[i],
                    x_mc[i], P_mc[i],
                    l.data.x_vec_traj[i], l.data.P_traj[i]) for i in 1:N]
                push!(kl_data, (ap=l.ap, kl=kl))
            end

            # ── Generate comparison plots ────────────────────────────────
            println("\nGenerating comparison plot...")

            # Plot 1: Position error
            p1 = plot(xlabel="Time (hours)", ylabel="Position Error (m)",
                yscale=:log10, legend=:topleft)
            for l in loaded
                plot!(p1, times[1:N] ./ 3600, l.metrics.pos_errors,
                    label=l.ap.name, linewidth=2, color=l.ap.color, linestyle=l.ap.linestyle)
            end

            # Plot 2: Velocity error
            p2 = plot(xlabel="Time (hours)", ylabel="Velocity Error (m/s)",
                yscale=:log10, legend=:topleft)
            for l in loaded
                plot!(p2, times[1:N] ./ 3600, l.metrics.vel_errors,
                    label=l.ap.name, linewidth=2, color=l.ap.color, linestyle=l.ap.linestyle)
            end

            # Plot 3: Position uncertainty error
            p3 = plot(xlabel="Time (hours)", ylabel="Position Uncertainty Error (m)",
                yscale=:log10, legend=:topleft)
            for l in loaded
                plot!(p3, times[1:N] ./ 3600, l.metrics.pos_uncertainty_errors,
                    label=l.ap.name, linewidth=2, color=l.ap.color, linestyle=l.ap.linestyle)
            end

            # Plot 4: Velocity uncertainty error
            p4 = plot(xlabel="Time (hours)", ylabel="Velocity Uncertainty Error (m/s)",
                yscale=:log10, legend=:topleft)
            for l in loaded
                plot!(p4, times[1:N] ./ 3600, l.metrics.vel_uncertainty_errors,
                    label=l.ap.name, linewidth=2, color=l.ap.color, linestyle=l.ap.linestyle)
            end

            # Plot 5: KL divergence
            p5 = plot(xlabel="Time (hours)", ylabel="KL Divergence",
                yscale=:log10, legend=:topleft)
            for kld in kl_data
                plot!(p5, times[1:N] ./ 3600, kld.kl,
                    label=kld.ap.name, linewidth=2, color=kld.ap.color, linestyle=kld.ap.linestyle)
            end

            # Combine and save
            p_combined = plot(p1, p2, p3, p4, p5,
                layout=(5, 1), size=(1000, 2000), left_margin=50Plots.px)

            figname = joinpath(figs_dir, "compare_approaches_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_std_e$(fname_num(oe_std[2])).png")
            savefig(p_combined, figname)
            println("  Saved figure: ", figname)

        end  # num_orbits
    end  # oe_std
end  # orbit

println("\n" * "="^80)
println("COMPARISON COMPLETE")
println("="^80)
