"""
Test script to verify the Newton method implementation of position_cartesian_to_ks.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using Test

include("../src/ks_transform.jl")

# Test tolerance for position reconstruction
const POS_TOL = 1e-10

# Test cases: various Cartesian positions
test_positions = [
    (name="Positive x-axis", r_vec=[1.0, 0.0, 0.0]),
    (name="Positive y-axis", r_vec=[0.0, 1.0, 0.0]),
    (name="Positive z-axis", r_vec=[0.0, 0.0, 1.0]),
    (name="Negative x-axis", r_vec=[-1.0, 0.0, 0.0]),
    (name="Unit vector in xy-plane", r_vec=[1.0 / sqrt(2), 1.0 / sqrt(2), 0.0]),
    (name="Arbitrary vector", r_vec=[1.0, 2.0, 3.0]),
    (name="Large magnitude", r_vec=[7000e3, 0.0, 0.0]),  # Earth orbit radius
    (name="Small magnitude", r_vec=[1e-3, 1e-3, 1e-3]),
    (name="All components positive", r_vec=[1.0, 2.0, 3.0]),
    (name="Mixed signs", r_vec=[1.0, -2.0, 3.0]),
    (name="Near zero", r_vec=[1e-6, 1e-6, 1e-6]),
]

# Different initial guesses to test
test_initial_guesses = [
    (name="Default", y_vec_near=[1.0, 0.0, 0.0, 0.0]),
    (name="Near solution (positive x)", y_vec_near=[0.707, 0.0, 0.0, 0.0]),
    (name="Random guess", y_vec_near=[0.5, 0.5, 0.5, 0.5]),
    (name="Small guess", y_vec_near=[0.1, 0.1, 0.1, 0.1]),
]

println("="^80)
println("TESTING: position_cartesian_to_ks_via_newton_method")
println("="^80)

# Test 1: Basic functionality - verify constraint satisfaction
println("\n" * "-"^80)
println("TEST 1: Constraint Satisfaction")
println("-"^80)

all_constraint_tests_passed = true
for (i, test_pos) in enumerate(test_positions)
    r_vec = test_pos.r_vec
    try
        y_vec = position_cartesian_to_ks_via_newton_method(r_vec)
        r_vec_reconstructed = position_ks_to_cartesian(y_vec)

        error = norm(r_vec_reconstructed - r_vec)
        passed = error < POS_TOL

        if !passed
            global all_constraint_tests_passed = false
        end

        status = passed ? "✓ PASS" : "✗ FAIL"
        println("$(i). $(test_pos.name): $(status)")
        println("   Input: $(r_vec)")
        println("   Error: $(error) m")
        if !passed
            println("   Reconstructed: $(r_vec_reconstructed)")
        end
    catch e
        global all_constraint_tests_passed = false
        println("$(i). $(test_pos.name): ✗ FAIL (Exception: $(typeof(e).name.name))")
        println("   Input: $(r_vec)")
        println("   Error: $(e)")
    end
end

if all_constraint_tests_passed
    println("\n✓ All constraint tests passed!")
else
    println("\n✗ Some constraint tests failed!")
end

# Test 2: Comparison with direct conversion method
println("\n" * "-"^80)
println("TEST 2: Comparison with Direct Conversion Method")
println("-"^80)

all_comparison_tests_passed = true
for (i, test_pos) in enumerate(test_positions)
    r_vec = test_pos.r_vec
    try
        # Direct conversion
        y_vec_direct = position_cartesian_to_ks(r_vec)

        # Newton method conversion
        y_vec_newton = position_cartesian_to_ks_via_newton_method(r_vec)

        # Both should satisfy the constraint
        r_vec_direct = position_ks_to_cartesian(y_vec_direct)
        r_vec_newton = position_ks_to_cartesian(y_vec_newton)

        error_direct = norm(r_vec_direct - r_vec)
        error_newton = norm(r_vec_newton - r_vec)

        # Compare the KS vectors (they may differ by sign/phase, but should produce same position)
        # Check if they produce the same position
        pos_diff = norm(r_vec_direct - r_vec_newton)

        passed = error_direct < POS_TOL && error_newton < POS_TOL && pos_diff < POS_TOL

        if !passed
            global all_comparison_tests_passed = false
        end

        status = passed ? "✓ PASS" : "✗ FAIL"
        println("$(i). $(test_pos.name): $(status)")
        println("   Direct method error: $(error_direct) m")
        println("   Newton method error: $(error_newton) m")
        println("   Position difference: $(pos_diff) m")
    catch e
        global all_comparison_tests_passed = false
        println("$(i). $(test_pos.name): ✗ FAIL (Exception: $(typeof(e).name.name))")
        println("   Input: $(r_vec)")
        println("   Error: $(e)")
    end
end

if all_comparison_tests_passed
    println("\n✓ All comparison tests passed!")
else
    println("\n✗ Some comparison tests failed!")
end

# Test 3: Different initial guesses
println("\n" * "-"^80)
println("TEST 3: Different Initial Guesses")
println("-"^80)

test_r_vec = [1.0, 2.0, 3.0]  # Use a standard test vector
all_guess_tests_passed = true

for (i, test_guess) in enumerate(test_initial_guesses)
    y_vec_near = test_guess.y_vec_near
    try
        y_vec = position_cartesian_to_ks_via_newton_method(test_r_vec; y_vec_near=y_vec_near)
        r_vec_reconstructed = position_ks_to_cartesian(y_vec)

        error = norm(r_vec_reconstructed - test_r_vec)
        passed = error < POS_TOL

        if !passed
            global all_guess_tests_passed = false
        end

        status = passed ? "✓ PASS" : "✗ FAIL"
        println("$(i). $(test_guess.name): $(status)")
        println("   Initial guess: $(y_vec_near)")
        println("   Error: $(error) m")
    catch e
        global all_guess_tests_passed = false
        println("$(i). $(test_guess.name): ✗ FAIL (Exception: $(typeof(e).name.name))")
        println("   Initial guess: $(y_vec_near)")
        println("   Error: $(e)")
    end
end

if all_guess_tests_passed
    println("\n✓ All initial guess tests passed!")
else
    println("\n✗ Some initial guess tests failed!")
end

# Test 4: Verbose return option
println("\n" * "-"^80)
println("TEST 4: Verbose Return Option")
println("-"^80)

test_r_vec = [1.0, 2.0, 3.0]
verbose_test_passed = false
try
    z, iterations = position_cartesian_to_ks_via_newton_method(test_r_vec; return_verbose=true)
    y_vec = z[1:4]
    λ = z[5:8]

    r_vec_reconstructed = position_ks_to_cartesian(y_vec)
    error = norm(r_vec_reconstructed - test_r_vec)

    println("Test vector: $(test_r_vec)")
    println("Iterations: $(iterations)")
    println("KS position: $(y_vec)")
    println("Lagrange multipliers: $(λ)")
    println("Reconstruction error: $(error) m")

    if error < POS_TOL && iterations > 0 && iterations <= 100
        println("✓ Verbose return test passed!")
        global verbose_test_passed = true
    else
        println("✗ Verbose return test failed!")
    end
catch e
    println("✗ Verbose return test failed! (Exception: $(typeof(e).name.name))")
    println("   Error: $(e)")
end

# Test 5: Convergence behavior
println("\n" * "-"^80)
println("TEST 5: Convergence Behavior")
println("-"^80)

test_r_vec = [1.0, 2.0, 3.0]
tolerances = [1e-6, 1e-9, 1e-12, 1e-15]

println("Testing convergence with different tolerances:")
for tol in tolerances
    try
        z, iterations = position_cartesian_to_ks_via_newton_method(test_r_vec; tol=tol, return_verbose=true)
        y_vec = z[1:4]
        r_vec_reconstructed = position_ks_to_cartesian(y_vec)
        error = norm(r_vec_reconstructed - test_r_vec)

        println("  Tolerance: $(tol) -> Iterations: $(iterations), Final error: $(error) m")
    catch e
        println("  Tolerance: $(tol) -> FAILED (Exception: $(typeof(e).name.name))")
    end
end

# Test 6: Edge cases
println("\n" * "-"^80)
println("TEST 6: Edge Cases")
println("-"^80)

edge_cases = [
    (name="Zero vector", r_vec=[0.0, 0.0, 0.0]),
    (name="Very small vector", r_vec=[1e-10, 1e-10, 1e-10]),
    (name="Very large vector", r_vec=[1e6, 0.0, 0.0]),
]

all_edge_tests_passed = true
for (i, edge_case) in enumerate(edge_cases)
    r_vec = edge_case.r_vec
    try
        y_vec = position_cartesian_to_ks_via_newton_method(r_vec)
        r_vec_reconstructed = position_ks_to_cartesian(y_vec)
        error = norm(r_vec_reconstructed - r_vec)

        # For zero vector, the error might be larger due to numerical issues
        tol = norm(r_vec) < 1e-9 ? 1e-6 : POS_TOL
        passed = error < tol

        if !passed
            global all_edge_tests_passed = false
        end

        status = passed ? "✓ PASS" : "✗ FAIL"
        println("$(i). $(edge_case.name): $(status)")
        println("   Input: $(r_vec)")
        println("   Error: $(error) m")
    catch e
        println("$(i). $(edge_case.name): ✗ FAIL (Exception: $(e))")
        global all_edge_tests_passed = false
    end
end

if all_edge_tests_passed
    println("\n✓ All edge case tests passed!")
else
    println("\n✗ Some edge case tests failed!")
end

# Test 7: Verify constraint satisfaction mathematically
println("\n" * "-"^80)
println("TEST 7: Mathematical Constraint Verification")
println("-"^80)

test_r_vec = [1.0, 2.0, 3.0]
constraint_verification_passed = false
try
    y_vec = position_cartesian_to_ks_via_newton_method(test_r_vec)

    # Verify: L(y_vec) * y_vec should equal [r_vec; 0]
    L_y = L(y_vec)
    L_y_times_y = L_y * y_vec
    expected = [test_r_vec; 0.0]

    constraint_error = norm(L_y_times_y - expected)
    println("Test vector: $(test_r_vec)")
    println("KS vector: $(y_vec)")
    println("L(y) * y: $(L_y_times_y)")
    println("Expected: $(expected)")
    println("Constraint error: $(constraint_error)")

    global constraint_verification_passed = constraint_error < POS_TOL
    if constraint_verification_passed
        println("✓ Constraint verification passed!")
    else
        println("✗ Constraint verification failed!")
    end
catch e
    println("✗ Constraint verification failed! (Exception: $(typeof(e).name.name))")
    println("   Error: $(e)")
end

# Summary
println("\n" * "="^80)
println("TEST SUMMARY")
println("="^80)

all_tests_passed = all_constraint_tests_passed &&
                   all_comparison_tests_passed &&
                   all_guess_tests_passed &&
                   all_edge_tests_passed &&
                   verbose_test_passed &&
                   constraint_verification_passed

if all_tests_passed
    println("✓ ALL TESTS PASSED!")
else
    println("✗ SOME TESTS FAILED - Check results above")
end
println("="^80)

