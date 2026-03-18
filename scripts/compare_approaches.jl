"""
Offline comparison of error propagation approaches.

Loads pre-computed approach results from out/ and Monte Carlo ground truth,
computes error metrics, prints a summary table to stdout, and generates
comparison plots in figs/.

Usage:
    julia scripts/compare_approaches.jl                     # run both (default config)
    julia scripts/compare_approaches.jl config/leo.jl        # run both (custom config)
    julia scripts/compare_approaches.jl config/leo.jl approaches   # only approach comparison
    julia scripts/compare_approaches.jl config/leo.jl bins         # only bins sweep

The out/ directory should contain NPZ files produced by
scripts/error_propagation_comparison.jl with the naming convention:
    {approach_id}_{orbit_id}_num_orbits{N}_oe_std_a{σ_a}m_std_e{σ_e}.npz

Monte Carlo ground truth is loaded from data/ with the naming convention:
    mc_{orbit_id}_num_orbits{N}_oe_std_a{σ_a}m_std_e{σ_e}_num_samples{N}.npz
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots
using LinearAlgebra
using NPZ
using PrettyTables
using Printf
using SatelliteDynamics

const SD = SatelliteDynamics

include(joinpath(@__DIR__, "..", "src", "cartesian_dynamics.jl"))
include(joinpath(@__DIR__, "..", "src", "ks_dynamics.jl"))
include(joinpath(@__DIR__, "..", "src", "error_propagation.jl"))

# Parse ARGS: [config.jl] [mode]
#   config.jl: optional path (default: config/default.jl)
#   mode: "approaches" | "bins" | "all" (default: "all")
args = copy(ARGS)
config_path = joinpath(@__DIR__, "..", "config", "default.jl")
mode = "all"
if length(args) >= 1 && endswith(args[1], ".jl")
    config_path = args[1]
    if !isabspath(config_path)
        config_path = joinpath(@__DIR__, "..", config_path)
    end
    args = args[2:end]
end
if length(args) >= 1
    mode = lowercase(args[1])
    @assert mode in ("approaches", "bins", "all") "mode must be 'approaches', 'bins', or 'all'"
end
println("Loading config: ", config_path)
include(config_path)

# ── Approach registry ────────────────────────────────────────────────────────
# Each entry: (id, display_name, color, linestyle)
APPROACHES = [
    (id="lincov_cart", name="LinCov (Cartesian)",    color=:green,  linestyle=:dash),
    (id="ut_cart",     name="UT (Cartesian)",        color=:blue,   linestyle=:dot),
    (id="ckf_cart",    name="CKF (Cartesian)",       color=:red,    linestyle=:dashdot),
    (id="ckf_ks",      name="CKF (KS)",             color=:orange, linestyle=:dashdotdot),
    (id="ckf_ks_rel",  name="CKF (KS Relative)",    color=:purple, linestyle=:dashdotdot),
    # (id="lincov_ks",   name="LinCov (KS)",          color=:brown,  linestyle=:dot),  # not implemented yet
    (id="stratified_ks", name="Stratified KS CKF",  color=:black,  linestyle=:dash),
]

# ── Directories ──────────────────────────────────────────────────────────────
out_dir   = joinpath(@__DIR__, "..", "out")
data_dir  = joinpath(@__DIR__, "..", "data")
figs_dir  = joinpath(@__DIR__, "..", "figs")
mkpath(figs_dir)

# ── Helper: load approach NPZ → (x_vec_traj, P_traj, times) ────────────────
fname_num(x) = isinteger(x) ? string(Int(x)) : string(x)

# argmin/best excluding NaN and Inf (for table formatting)
argmin_finite(vals) = (idx = findall(isfinite, vals); isempty(idx) ? 1 : idx[argmin(vals[i] for i in idx)])
best_finite(vals)   = (idx = findall(isfinite, vals); isempty(idx) ? NaN : minimum(vals[i] for i in idx))

# Format a table cell: value, best*, or value (+X%)
function fmt_cell(v, best, is_best, fmt)
    if !isfinite(v)
        return isinf(v) ? "Inf" : "NaN"
    end
    rel(v, b) = (isfinite(v) && isfinite(b) && b > 0) ? 100 * (v - b) / b : 0.0
    is_best && return @sprintf("%s*", fmt(v))
    (isfinite(best) && best > 0) ? @sprintf("%s (+%.3f%%)", fmt(v), rel(v, best)) : fmt(v)
end

# Print summary table using PrettyTables
function print_summary_table(row_labels, col_labels, vals_matrix, best_col_idx; row_suffix=nothing, first_col_header="")
    row_suffix = something(row_suffix, fill("", length(row_labels)))
    nrows, ncols = size(vals_matrix)
    cells = [fmt_cell(vals_matrix[i, j], vals_matrix[best_col_idx[j], j],
                      i == best_col_idx[j], v -> @sprintf("%.3f", v))
              for i in 1:nrows, j in 1:ncols]
    rows = [vcat(rl * row_suffix[i], cells[i, :]) for (i, rl) in enumerate(row_labels)]
    data = permutedims(hcat(rows...))
    pretty_table(data;
        column_labels = vcat(first_col_header, col_labels),
        alignment = vcat(:l, fill(:r, ncols)),
        table_format = TextTableFormat(borders = text_table_borders__unicode_rounded),
        fit_table_in_display_horizontally = false,
    )
    println("  * = best in column (finite values only); (+X%) = relative error to best")
    if any(!isempty, row_suffix)
        println("  † = did not complete full propagation (fewer timesteps than MC)")
    end
end

function load_approach(dir, approach_id, orbit, num_orbits, oe_std; num_samples=nothing)
    if approach_id == "mc" && num_samples !== nothing
        filename = "mc_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_std_e$(fname_num(oe_std[2]))_num_samples$(Int(num_samples)).npz"
    else
        filename = "$(approach_id)_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_std_e$(fname_num(oe_std[2])).npz"
    end
    filepath = joinpath(dir, filename)
    if !isfile(filepath)
        return nothing
    end
    npz = try
        NPZ.npzread(filepath)
    catch e
        @warn "Failed to load $filepath: $e"
        return nothing
    end
    x_array   = npz["x"]          # (N, 6)
    P_array   = npz["P"]          # (N, 6, 6)
    timestamp = npz["timestamp"]  # (N,)
    N = size(x_array, 1)
    x_traj = [x_array[i, :] for i in 1:N]
    P_traj = [P_array[i, :, :] for i in 1:N]
    return (x_vec_traj=x_traj, P_traj=P_traj, times=timestamp)
end

# ── Main comparison loop ─────────────────────────────────────────────────────
if mode in ("approaches", "all")
println("="^80)
println("OFFLINE APPROACH COMPARISON")
println("="^80)
println("Monte Carlo from:     ", data_dir)
println("Approach results from:", out_dir)
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

            # Load Monte Carlo ground truth from data/
            mc = load_approach(data_dir, "mc", orbit, num_orbits, oe_std; num_samples=NUM_MC_SAMPLES)
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

            # ── KL divergence (needed for table and plot) ──────────────────
            # Each approach uses its own successful propagation length (no truncation to minimum)
            kl_data = []
            for l in loaded
                n = min(length(x_mc), length(l.data.x_vec_traj))
                kl = [gaussian_kl_divergence(l.ap.name, times[i],
                    x_mc[i], P_mc[i],
                    l.data.x_vec_traj[i], l.data.P_traj[i]) for i in 1:n]
                kl_finite = [k for k in kl if isfinite(k)]
                kl_rmse = isempty(kl_finite) ? NaN : sqrt(sum(x -> x^2, kl_finite) / length(kl_finite))
                push!(kl_data, (ap=l.ap, kl=kl, kl_rmse=kl_rmse))
            end

            # ── Print summary table ──────────────────────────────────────
            pos_vals  = [l.metrics.pos_rmse for l in loaded]
            vel_vals  = [l.metrics.vel_rmse for l in loaded]
            σpos_vals = [l.metrics.pos_uncertainty_rmse for l in loaded]
            σvel_vals = [l.metrics.vel_uncertainty_rmse for l in loaded]
            kl_vals   = [kld.kl_rmse for kld in kl_data]
            n_mc = length(x_mc)
            row_labels = [l.ap.name for l in loaded]
            row_suffix = [length(l.data.x_vec_traj) < n_mc ? " †" : "" for l in loaded]
            vals = hcat(pos_vals, vel_vals, σpos_vals, σvel_vals, kl_vals)
            best_col_idx = [argmin_finite(vals[:, j]) for j in 1:5]
            col_labels = ["Pos RMSE (m)", "Vel RMSE (m/s)", "σ_pos RMSE (m)", "σ_vel RMSE (m/s)", "KL RMSE"]
            println()
            print_summary_table(row_labels, col_labels, vals, best_col_idx; row_suffix, first_col_header="Approach")

            # ── Generate comparison plots ────────────────────────────────
            println("\nGenerating comparison plot...")

            # Plot 1: Position error (each approach plotted for its full successful duration)
            p1 = plot(xlabel="Time (hours)", ylabel="Position Error (m)",
                yscale=:log10, legend=:topleft)
            for l in loaded
                n = length(l.metrics.pos_errors)
                plot!(p1, times[1:n] ./ 3600, l.metrics.pos_errors,
                    label=l.ap.name, linewidth=2, color=l.ap.color, linestyle=l.ap.linestyle)
            end

            # Plot 2: Velocity error
            p2 = plot(xlabel="Time (hours)", ylabel="Velocity Error (m/s)",
                yscale=:log10, legend=:topleft)
            for l in loaded
                n = length(l.metrics.vel_errors)
                plot!(p2, times[1:n] ./ 3600, l.metrics.vel_errors,
                    label=l.ap.name, linewidth=2, color=l.ap.color, linestyle=l.ap.linestyle)
            end

            # Plot 3: Position uncertainty error
            p3 = plot(xlabel="Time (hours)", ylabel="Position Uncertainty Error (m)",
                yscale=:log10, legend=:topleft)
            for l in loaded
                n = length(l.metrics.pos_uncertainty_errors)
                plot!(p3, times[1:n] ./ 3600, l.metrics.pos_uncertainty_errors,
                    label=l.ap.name, linewidth=2, color=l.ap.color, linestyle=l.ap.linestyle)
            end

            # Plot 4: Velocity uncertainty error
            p4 = plot(xlabel="Time (hours)", ylabel="Velocity Uncertainty Error (m/s)",
                yscale=:log10, legend=:topleft)
            for l in loaded
                n = length(l.metrics.vel_uncertainty_errors)
                plot!(p4, times[1:n] ./ 3600, l.metrics.vel_uncertainty_errors,
                    label=l.ap.name, linewidth=2, color=l.ap.color, linestyle=l.ap.linestyle)
            end

            # Plot 5: KL divergence
            p5 = plot(xlabel="Time (hours)", ylabel="KL Divergence",
                yscale=:log10, legend=:topleft)
            for kld in kl_data
                n = length(kld.kl)
                plot!(p5, times[1:n] ./ 3600, kld.kl,
                    label=kld.ap.name, linewidth=2, color=kld.ap.color, linestyle=kld.ap.linestyle)
            end

            # Combine and save
            p_combined = plot(p1, p2, p3, p4, p5,
                layout=(5, 1), size=(1000, 2000), left_margin=50Plots.px)

            figname = joinpath(figs_dir, "error_propagation_comparison_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_std_e$(fname_num(oe_std[2]))_num_samples$(NUM_MC_SAMPLES).png")
            savefig(p_combined, figname)
            println("  Saved figure: ", figname)

        end  # num_orbits
    end  # oe_std
end  # orbit

println("\n" * "="^80)
println("COMPARISON COMPLETE")
println("="^80)
end  # mode in ("approaches", "all")

# ── Bins sweep: reproduce energy_binned_bins_sweep plots from out/ ────────────
if mode in ("bins", "all")
println("\n" * "="^80)
println("BINS SWEEP: STRATIFIED KS CKF (varying number of energy bins)")
println("="^80)

bin_styles = [
    (color=:green,  linestyle=:dash),
    (color=:blue,   linestyle=:dot),
    (color=:red,    linestyle=:dashdot),
    (color=:orange, linestyle=:dashdotdot),
    (color=:purple, linestyle=:dashdotdot),
    (color=:black,  linestyle=:dash),
]

for orbit in TEST_ORBITS
    for oe_std in OE_INITIAL_STD_SCENARIOS
        for num_orbits in NUM_ORBITS_LIST
            println("\n" * "="^80)
            println("BINS SWEEP SCENARIO: ", orbit.name,
                "  σ_a = ", oe_std[1], " m  |  orbits = ", num_orbits)
            println("="^80)

            # Load Monte Carlo ground truth
            mc = load_approach(data_dir, "mc", orbit, num_orbits, oe_std; num_samples=NUM_MC_SAMPLES)
            if mc === nothing
                println("  ✗ Monte Carlo ground truth not found — skipping")
                continue
            end
            x_mc   = mc.x_vec_traj
            P_mc   = mc.P_traj
            times  = mc.times

            # Load each bin count
            bin_results = []   # (num_bins, data, metrics)
            for num_bins in NUM_ENERGY_BINS_LIST
                ap_id = "stratified_ks_num_bins$(num_bins)"
                data  = load_approach(out_dir, ap_id, orbit, num_orbits, oe_std)
                if data === nothing
                    println("  ✗ stratified_ks num_bins=", num_bins, " — data not found")
                    continue
                end
                met = error_metrics(x_mc, P_mc, data.x_vec_traj, data.P_traj)
                push!(bin_results, (num_bins=num_bins, data=data, metrics=met))
                println("  ✓ num_bins=", num_bins, " — loaded (", length(data.x_vec_traj), " states)")
            end

            if isempty(bin_results)
                println("  No bin data found — skipping scenario")
                continue
            end

            # KL divergence (for table and plot)
            # Each bin config uses its own successful propagation length (no truncation to minimum)
            bin_kl_data = []
            for r in bin_results
                n = min(length(x_mc), length(r.data.x_vec_traj))
                kl_vals = [gaussian_kl_divergence("stratified_ks num_bins=$(r.num_bins)", times[i],
                    x_mc[i], P_mc[i],
                    r.data.x_vec_traj[i], r.data.P_traj[i]) for i in 1:n]
                kl_finite = [k for k in kl_vals if isfinite(k)]
                kl_rmse = isempty(kl_finite) ? NaN : sqrt(sum(x -> x^2, kl_finite) / length(kl_finite))
                push!(bin_kl_data, (kl_vals=kl_vals, kl_rmse=kl_rmse))
            end

            # Print summary table
            n_mc = length(x_mc)
            row_labels = [string(r.num_bins) for r in bin_results]
            row_suffix = [length(r.data.x_vec_traj) < n_mc ? " †" : "" for r in bin_results]
            vals = hcat(
                [r.metrics.pos_rmse for r in bin_results],
                [r.metrics.vel_rmse for r in bin_results],
                [r.metrics.pos_uncertainty_rmse for r in bin_results],
                [r.metrics.vel_uncertainty_rmse for r in bin_results],
                [kld.kl_rmse for kld in bin_kl_data],
            )
            best_col_idx = [argmin_finite(vals[:, j]) for j in 1:5]
            col_labels = ["Pos RMSE (m)", "Vel RMSE (m/s)", "σ_pos RMSE (m)", "σ_vel RMSE (m/s)", "KL RMSE"]
            println()
            print_summary_table(row_labels, col_labels, vals, best_col_idx; row_suffix, first_col_header="Num Bins")

            # Build plots
            p1 = plot(xlabel="Time (hours)", ylabel="Position Error (m)",
                yscale=:log10, legend=:topleft)
            p2 = plot(xlabel="Time (hours)", ylabel="Velocity Error (m/s)",
                yscale=:log10, legend=:topleft)
            p3 = plot(xlabel="Time (hours)", ylabel="Position Uncertainty Error (m)",
                yscale=:log10, legend=:topleft)
            p4 = plot(xlabel="Time (hours)", ylabel="Velocity Uncertainty Error (m/s)",
                yscale=:log10, legend=:topleft)
            p5 = plot(xlabel="Time (hours)", ylabel="KL Divergence",
                yscale=:log10, legend=:topleft)

            for (idx, r) in enumerate(bin_results)
                style = bin_styles[mod1(idx, length(bin_styles))]
                lbl   = "$(r.num_bins) bins"
                m     = r.metrics
                kld   = bin_kl_data[idx]
                n     = length(m.pos_errors)
                plot!(p1, times[1:n] ./ 3600, m.pos_errors,
                    label=lbl, linewidth=2, color=style.color, linestyle=style.linestyle)
                plot!(p2, times[1:n] ./ 3600, m.vel_errors,
                    label=lbl, linewidth=2, color=style.color, linestyle=style.linestyle)
                plot!(p3, times[1:n] ./ 3600, m.pos_uncertainty_errors,
                    label=lbl, linewidth=2, color=style.color, linestyle=style.linestyle)
                plot!(p4, times[1:n] ./ 3600, m.vel_uncertainty_errors,
                    label=lbl, linewidth=2, color=style.color, linestyle=style.linestyle)
                plot!(p5, times[1:n] ./ 3600, kld.kl_vals,
                    label=lbl, linewidth=2, color=style.color, linestyle=style.linestyle)
            end

            p_combined = plot(p1, p2, p3, p4, p5,
                layout=(5, 1), size=(1000, 2000), left_margin=50Plots.px)

            bins_str = join(NUM_ENERGY_BINS_LIST, "-")
            figname  = joinpath(figs_dir,
                "energy_binned_bins_sweep_$(orbit.id)_num_orbits$(Int(num_orbits))_oe_std_a$(fname_num(oe_std[1]))m_std_e$(fname_num(oe_std[2]))_num_bins$(bins_str).png")
            savefig(p_combined, figname)
            println("  Saved figure: ", figname)

        end  # num_orbits
    end  # oe_std
end  # orbit

println("\n" * "="^80)
println("BINS SWEEP COMPLETE")
println("="^80)
end  # mode in ("bins", "all")
