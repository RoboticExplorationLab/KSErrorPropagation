"""
Script to run Monte Carlo propagation and save results as npz file.
Similar to test_error_propagation_comparison.jl but saves individual samples.

Usage:
    # Run as script:
    julia src/save_monte_carlo_npz.jl

    # Or from Julia REPL:
    include("src/save_monte_carlo_npz.jl")
    main()  # Call main() explicitly after including

The script saves data to /data directory at project root with format:
    - x: mean state vector trajectory (N, 6) where N is number of time steps
    - P: covariance matrix trajectory (N, 6, 6)
    - x1, x2, ..., xM: individual sample trajectories (N, 6) each, where M is num_samples
    - timestamp: time array in seconds (N,) - compatible with npzLoader which checks for ['timesteps', 'timestamp', 'time']

Configuration can be modified in the main() function:
    - test_orbits: list of orbits to propagate
    - position_uncertainties: list of position uncertainties (meters)
    - num_orbits_list: list of number of orbits to propagate
    - num_samples: number of Monte Carlo samples
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using SatelliteDynamics
using DifferentialEquations

# Try to load NPZ, add if not available
try
    using NPZ
catch
    println("NPZ.jl not found. Adding NPZ package...")
    Pkg.add("NPZ")
    using NPZ
end

const SD = SatelliteDynamics

include(joinpath(@__DIR__, "..", "src", "cartesian_dynamics.jl"))
include(joinpath(@__DIR__, "..", "src", "error_propagation.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))

# Define simulation parameters (matching test_error_propagation_comparison.jl)
SIM_PARAMS = (
    # Physical parameters
    GM=SD.GM_EARTH,
    R_EARTH=SD.R_EARTH,
    J2=SD.J2_EARTH,
    OMEGA_EARTH=SD.OMEGA_EARTH,
    CD=2.2, # drag coefficient
    A=1.0, # cross-sectional area (m²)
    m=100.0, # mass (kg)
    epoch_0=Epoch(2000, 1, 1, 12, 0, 0), # initial epoch at 0 TBD seconds after J2000

    # Add perturbations
    add_perturbations=true,

    # Sampling time (time step) in seconds
    sampling_time=30.0,  # seconds

    # Integrator to use (from DifferentialEquations.jl)
    integrator=Tsit5,

    # Integrator tolerances
    abstol=1e-12,      # Absolute tolerance
    reltol=1e-13,      # Relative tolerance

    # Success criteria thresholds
    max_pos_error_threshold=1.0,      # meters
    max_vel_error_threshold=0.01,     # m/s

    # Scaling factors (optional, computed from orbit if not provided)
    t_scale=nothing,  # time scale (typically orbital period)
    r_scale=nothing,  # position scale (typically semi-major axis)
    v_scale=nothing,  # velocity scale
    a_scale=nothing,  # acceleration scale
    GM_scale=nothing,  # gravitational constant scale
)


function main()
    # ========== Configuration ==========
    # Define test orbits (matching test_energy_binned_bins_sweep.jl)
    test_orbits = [
        (name="Low Earth orbit (e=0.1, 750 km altitude)",
            a=750e3 + SIM_PARAMS.R_EARTH,
            e=0.01,
            i=deg2rad(0.0),
            omega=deg2rad(0.0),
            RAAN=deg2rad(0.0),
            M=deg2rad(0.0),
            description="LEO",
            id="leo"),
        (name="Molniya orbit (e=0.74, 26600 km semi-major axis)",
            a=26600e3,
            e=0.74,
            i=deg2rad(63.4),
            omega=deg2rad(270.0),
            RAAN=deg2rad(0.0),
            M=deg2rad(0.0),
            description="Molniya",
            id="mol"),
    ]
    
    # Position uncertainty scenarios (meters)
    position_uncertainties = [1e3, 1e4, 1e5]  # 1km, 10km, 100km
    
    # Number of orbits to test
    num_orbits_list = [3.0]
    
    # Number of Monte Carlo samples
    num_samples = 5000
    
    # Output directory (create data directory at project root)
    project_root = joinpath(@__DIR__, "..")
    data_dir = joinpath(project_root, "data")
    if !isdir(data_dir)
        mkpath(data_dir)
        println("Created data directory: ", data_dir)
    end
    
    println("="^80)
    println("MONTE CARLO PROPAGATION AND NPZ SAVE")
    println("="^80)
    println("Configuration:")
    println("  Number of orbits: ", num_orbits_list)
    println("  Position uncertainties: ", position_uncertainties, " m")
    println("  Number of samples: ", num_samples)
    println("  Number of scenarios: ", length(test_orbits) * length(position_uncertainties) * length(num_orbits_list))
    
    # Loop over all orbit/uncertainty/orbit combinations
    for (orbit_idx, orbit) in enumerate(test_orbits)
        sma = orbit.a
        e = orbit.e
        i = orbit.i
        ω = orbit.omega
        Ω = orbit.RAAN
        M = orbit.M
        orbit_name = orbit.name
        
        println("\n" * "="^80)
        println("ORBIT: ", orbit_name)
        println("="^80)
        println("Orbit parameters:")
        println("  Semi-major axis: ", sma / 1e3, " km")
        println("  Eccentricity: ", e)
        println("  Inclination: ", rad2deg(i), " deg")
        println("  Argument of periapsis: ", rad2deg(ω), " deg")
        println("  RAAN: ", rad2deg(Ω), " deg")
        println("  Mean anomaly: ", rad2deg(M), " deg")
        
        # Convert orbital elements to Cartesian state
        oe_vec = [sma, e, i, Ω, ω, M]
        x_vec_0 = SD.sOSCtoCART(oe_vec; GM=SIM_PARAMS.GM, use_degrees=false)
        r_vec_0 = x_vec_0[1:3]
        v_vec_0 = x_vec_0[4:6]
        
        println("\nInitial conditions:")
        println("  Position: ", r_vec_0)
        println("  Velocity: ", v_vec_0)
        println("  Radius: ", norm(r_vec_0) / 1e3, " km")
        println("  Speed: ", norm(v_vec_0) / 1e3, " km/s")
        
        # Loop over position uncertainty scenarios
        for σ_pos in position_uncertainties
            # Compute velocity uncertainty from the Uncertainty Propagation Law
            σ_vel = sqrt((-SIM_PARAMS.GM / (sqrt(SIM_PARAMS.GM * (2 / norm(r_vec_0) - 1 / sma)) * norm(r_vec_0)^2))^2 * σ_pos^2)
            
            println("\n" * "-"^80)
            println("POSITION UNCERTAINTY SCENARIO: σ_pos = ", σ_pos, " m")
            println("-"^80)
            println("  Position uncertainty: ", σ_pos, " m (1-sigma)")
            println("  Velocity uncertainty: ", σ_vel, " m/s (1-sigma)")
            
            # Initial state covariance
            P_0 = diagm([σ_pos^2, σ_pos^2, σ_pos^2, σ_vel^2, σ_vel^2, σ_vel^2])
            
            # Loop over number of orbits
            for num_orbits in num_orbits_list
                println("\n" * "="^80)
                println("NUM ORBITS: ", num_orbits)
                println("="^80)
                
                # Propagation time: multiple orbital periods
                T_orbital = 2π * sqrt(sma^3 / SIM_PARAMS.GM)
                t_0 = 0.0
                t_end = num_orbits * T_orbital
                dt = SIM_PARAMS.sampling_time
                times = collect(t_0:dt:t_end)
                
                println("\nPropagation parameters:")
                println("  Orbital period: ", T_orbital, " s (", T_orbital / 3600, " hours)")
                println("  Time step: ", dt, " s")
                println("  Number of steps: ", length(times))
                println("  Total propagation time: ", t_end / 3600, " hours")
                
                # Scaling factors
                t_scale = something(SIM_PARAMS.t_scale, T_orbital)
                r_scale = something(SIM_PARAMS.r_scale, sma)
                v_scale = something(SIM_PARAMS.v_scale, r_scale / t_scale)
                a_scale = something(SIM_PARAMS.a_scale, v_scale / t_scale)
                GM_scale = something(SIM_PARAMS.GM_scale, r_scale^3 / t_scale^2)
                
                # Scaling matrices for covariance propagation
                S = Diagonal([1 / r_scale, 1 / r_scale, 1 / r_scale, 1 / v_scale, 1 / v_scale, 1 / v_scale])
                S_inv = Diagonal([r_scale, r_scale, r_scale, v_scale, v_scale, v_scale])
                
                sim_params = merge(SIM_PARAMS, (t_scale=t_scale, r_scale=r_scale, v_scale=v_scale,
                    a_scale=a_scale, GM_scale=GM_scale, S=S, S_inv=S_inv))
                
                # Run Monte Carlo propagation with samples
                println("\n" * "="^80)
                println("RUNNING MONTE CARLO PROPAGATION")
                println("="^80)
                x_vec_traj, P_traj, samples_propagated = propagate_uncertainty_via_monte_carlo(
                    x_vec_0, P_0, times, sim_params, num_samples; return_samples=true
                )
                println("  Completed: ", length(x_vec_traj), " states")
                
                # Prepare data for saving
                println("\n" * "="^80)
                println("PREPARING DATA FOR NPZ SAVE")
                println("="^80)
                
                N = length(times)  # Number of time steps
                
                # Convert mean trajectory to array: (N, 6)
                x_array = zeros(N, 6)
                for t_idx in 1:N
                    x_array[t_idx, :] = x_vec_traj[t_idx]
                end
                
                # Convert covariance trajectory to array: (N, 6, 6)
                P_array = zeros(N, 6, 6)
                for t_idx in 1:N
                    P_array[t_idx, :, :] = P_traj[t_idx]
                end
                
                # Convert samples: reorganize from samples_propagated[t_idx][i] to samples[i][t_idx]
                # Each sample trajectory will be (N, 6)
                println("  Reorganizing ", num_samples, " sample trajectories...")
                sample_arrays = Dict{String, Array{Float64, 2}}()
                for i in 1:num_samples
                    sample_traj = zeros(N, 6)
                    for t_idx in 1:N
                        sample_traj[t_idx, :] = samples_propagated[t_idx][i]
                    end
                    sample_arrays["x$(i)"] = sample_traj
                    if i % 1000 == 0
                        println("    Processed ", i, " / ", num_samples, " samples")
                    end
                end
                
                # Create output filename (matching test_error_propagation_comparison.jl pattern)
                output_filename = "mc_$(orbit.id)_num_orbits$(Int(num_orbits))_std_pos$(Int(σ_pos))m_std_vel$(round(σ_vel, digits=6))mps_num_samples$(Int(num_samples)).npz"
                output_path = joinpath(data_dir, output_filename)
                
                println("\n" * "="^80)
                println("SAVING TO NPZ FILE")
                println("="^80)
                println("  Output file: ", output_path)
                println("  Data shapes:")
                println("    x (mean): ", size(x_array))
                println("    P (covariance): ", size(P_array))
                println("    Number of sample arrays: ", length(sample_arrays))
                println("    Sample array shape: ", size(first(values(sample_arrays))))
                
                # Save to npz
                println("  Creating npz data dictionary...")
                # Create dictionary with all data
                # Note: timestamp contains time in seconds (matching Python's times_seconds)
                # The npzLoader checks for ['timesteps', 'timestamp', 'time'] as aliases
                npz_data = Dict{String, Any}(
                    "x" => x_array,
                    "P" => P_array,
                    "timestamp" => times,  # Time array in seconds (npzLoader will find this)
                )
                
                # Add all sample arrays
                println("  Adding ", num_samples, " sample arrays to npz data...")
                for (key, value) in sample_arrays
                    npz_data[key] = value
                end
                
                println("  Writing npz file (this may take a while for large numbers of samples)...")
                NPZ.npzwrite(output_path, npz_data)
                
                println("\n✓ Successfully saved Monte Carlo data to: ", output_path)
                println("  File contains:")
                println("    - x: mean state vector trajectory (", size(x_array), ")")
                println("    - P: covariance matrix trajectory (", size(P_array), ")")
                println("    - x1, x2, ..., x$(num_samples): individual sample trajectories (", size(first(values(sample_arrays))), " each)")
                println("    - timestamp: time array in seconds (", length(times), ")")
                
                println("\n" * "="^80)
                println("SCENARIO COMPLETE")
                println("="^80)
            end  # end num_orbits loop
        end  # end position uncertainty loop
    end  # end orbit loop
    
    println("\n" * "="^80)
    println("ALL SCENARIOS COMPLETE")
    println("="^80)
end

# Run if executed directly as a script
if !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
# Note: When using include(), call main() explicitly: main()
