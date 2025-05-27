using Test
using SparseMaxSR: compute_sr, compute_mve_sr, compute_mve_weights
using LinearAlgebra

# Helper for manual Sharpe
test_sr_manual(w, μ, Σ) = dot(w, μ) / sqrt(dot(w, Σ * w))

@testset "compute_sr tests" begin
    μ = [0.1, 0.2, 0.15]
    Σ = [0.05 0.01 0.0;
         0.01 0.04 0.005;
         0.0  0.005 0.03]
    w = [0.3, 0.4, 0.3]

    # full selection
    sr_full = compute_sr(w, μ, Σ)
    @test isapprox(sr_full, test_sr_manual(w, μ, Σ); atol=1e-12)

    # subselection
    sel = [1,3]
    w_sel = w[sel]; μ_sel = μ[sel]; Σ_sel = Σ[sel, sel]
    sr_sub = compute_sr(w, μ, Σ; selection=sel)
    @test isapprox(sr_sub, test_sr_manual(w_sel, μ_sel, Σ_sel); atol=1e-12)

    # do_checks=true valid
    @test compute_sr(w, μ, Σ; do_checks=true) ≈ sr_full

    # mismatched dims
    bad_μ = [0.1, 0.2]
    @test_throws AssertionError compute_sr(w, bad_μ, Σ; do_checks=true)
end

@testset "compute_mve_sr tests" begin
    # identity Σ => sqrt(μ'μ)
    μ = [1.0, 2.0, 2.0]
    Σ = Matrix{Float64}(I, 3, 3)
    expected_full = sqrt(sum(μ.^2))
    @test compute_mve_sr(μ, Σ) ≈ expected_full

    # subselection
    sel = [2,3]
    μ_sel = μ[sel]; Σ_sel = Σ[sel, sel]
    expected_sub = sqrt(sum(μ_sel.^2))
    @test compute_mve_sr(μ, Σ; selection=sel) ≈ expected_sub

    # do_checks=true valid
    @test compute_mve_sr(μ, Σ; do_checks=true) ≈ expected_full

    # non-square Σ
    bad_Σ = [1.0 0.0; 0.0 1.0; 0.1 0.1]
    @test_throws AssertionError compute_mve_sr(μ, bad_Σ; do_checks=true)
end

@testset "compute_mve_weights tests" begin
    # 1) Full-sample, identity Σ and gamma=1 => weights == μ
    μ = [0.5, -0.2, 0.7]
    Σ = Matrix{Float64}(I, 3, 3)
    w_id = compute_mve_weights(μ, Σ)
    @test isapprox(w_id, μ; atol=1e-12)

    # 2) General Σ, gamma != 1
    μ2 = [0.1, 0.2, 0.3]
    Σ2 = [2.0 0.5 0.0;
          0.5 1.0 0.2;
          0.0 0.2 0.5]
    γ = 2.5
    # manual: w = inv(Σ2) * μ2 / γ
    w_manual = (Σ2 \ μ2) / γ
    @test compute_mve_weights(μ2, Σ2; γ=γ) ≈ w_manual

    # 3) Subselection: only fill selected entries, zero elsewhere
    sel = [1,3]
    w_sub = compute_mve_weights(μ2, Σ2; selection=sel, γ=1.0)
    @test w_sub[2] == 0.0
    @test isapprox(w_sub[sel], (Σ2[sel,sel] \ μ2[sel]); atol=1e-12)

    # 4) do_checks=true valid
    @test compute_mve_weights(μ2, Σ2; γ=γ,do_checks=true) ≈ w_manual

    # 5) Bad inputs
    bad_Σ = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test_throws AssertionError compute_mve_weights(μ2, bad_Σ; do_checks=true)
    @test_throws AssertionError compute_mve_weights(μ2, Σ2; γ=0.0, do_checks=true)
    @test_throws AssertionError compute_mve_weights(μ2, Σ2; selection=[4], do_checks=true)
end
