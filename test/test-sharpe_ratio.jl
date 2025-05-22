using Test
using SparseMaxSR: compute_sr, compute_mve_sr
using LinearAlgebra

# Helper function to compute expected Sharpe ratio
test_sr_manual(weights, μ, Σ) = dot(weights, μ) / sqrt(dot(weights, Σ * weights))

@testset "compute_sr tests" begin
    μ = [0.1, 0.2, 0.15]
    Σ = [0.05 0.01 0.0;
         0.01 0.04 0.005;
         0.0 0.005 0.03]
    weights = [0.3, 0.4, 0.3]

    # Full selection default
    sr_full = compute_sr(weights, μ, Σ)
    @test isapprox(sr_full, test_sr_manual(weights, μ, Σ); atol=1e-12)

    # Subselection
    sel = [1, 3]
    w_sel = weights[sel]
    μ_sel = μ[sel]
    Σ_sel = Σ[sel, sel]
    sr_sub = compute_sr(weights, μ, Σ; selection=sel)
    @test isapprox(sr_sub, test_sr_manual(w_sel, μ_sel, Σ_sel); atol=1e-12)

    # With do_checks=true and valid inputs
    @test compute_sr(weights, μ, Σ; do_checks=true) ≈ sr_full

    # do_checks should throw on mismatched dimensions
    bad_μ = [0.1, 0.2]
    @test_throws AssertionError compute_sr(weights, bad_μ, Σ; do_checks=true)
end

@testset "compute_mve_sr tests" begin
    # Identity covariance: mve_sr = sqrt(sum(μ.^2))
    μ = [1.0, 2.0, 2.0]
    Σ = Matrix{Float64}(I, 3, 3)
    expected_full = sqrt(sum(μ.^2))
    @test compute_mve_sr(μ, Σ) ≈ expected_full

    # Subselection
    sel = [2, 3]
    μ_sel = μ[sel]
    Σ_sel = Σ[sel, sel]
    expected_sub = sqrt(sum(μ_sel.^2))
    @test compute_mve_sr(μ, Σ; selection=sel) ≈ expected_sub

    # With do_checks=true and valid inputs
    @test compute_mve_sr(μ, Σ; do_checks=true) ≈ expected_full

    # do_checks should throw on non-square Σ
    bad_Σ = [1.0 0.0; 0.0 1.0; 0.1 0.1]
    @test_throws AssertionError compute_mve_sr(μ, bad_Σ; do_checks=true)
end
