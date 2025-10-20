# test-SharpeRatio.jl

using Test, LinearAlgebra, Random
using SparseMaxSR
using SparseMaxSR.SharpeRatio  # convenient names; functions are also re-exported at top level

const TOL = 1e-7  # slightly relaxed to avoid spurious failures across BLAS/solver variants

@testset "SharpeRatio" begin
    # --- basic sanity ---------------------------------------------------------
    @testset "basic sanity" begin
        μ = [0.10, 0.20, 0.15]
        A = [0.2 0.05 0.0; 0.05 0.3 0.02; 0.0 0.02 0.25]
        Σ = Symmetric(A)
        w = fill(1/3, 3)

        @test isfinite(compute_sr(w, μ, Σ))
        @test compute_mve_sr(μ, Σ) > 0
        @test length(compute_mve_weights(μ, Σ)) == 3
    end

    # --- selection semantics --------------------------------------------------
    @testset "selection semantics" begin
        μ = [0.1, 0.2, 0.3, 0.4]
        Σ = Symmetric([0.08 0.01 0.00 0.00;
                       0.01 0.09 0.01 0.00;
                       0.00 0.01 0.07 0.02;
                       0.00 0.00 0.02 0.10])
        sel = [1, 3, 4]
        n = length(μ)

        # compute_sr with selection vs explicit sub-matrix call
        w = rand(n); w ./= sum(abs, w) + eps()
        sr_sel = compute_sr(w, μ, Σ; selection=sel, do_checks=true)

        Σ_sub = Symmetric(Matrix(Σ)[sel, sel])
        sr_direct = compute_sr(w[sel], μ[sel], Σ_sub; do_checks=true)
        @test abs(sr_sel - sr_direct) ≤ TOL

        # compute_mve_weights: zeros outside selection
        w_sel = compute_mve_weights(μ, Σ; selection=sel)
        @test count(!iszero, w_sel) == length(sel)
        @test all(i -> (i ∈ sel) || (w_sel[i] == 0.0), eachindex(w_sel))

        # selection = 1:n should equal "no selection"
        sr_all   = compute_mve_sr(μ, Σ)
        sr_fullS = compute_mve_sr(μ, Σ; selection=collect(1:n))
        @test abs(sr_all - sr_fullS) ≤ TOL
    end

    # --- scale behavior -------------------------------------------------------
    @testset "scale behavior" begin
        Random.seed!(1)
        n = 5
        μ = 0.05 .+ 0.10 .* rand(n)
        A = randn(n, n)
        Σ = Symmetric(A*A' + 0.1I)  # NOTE: plain + with I (no broadcast)

        # MVE_SR(αμ, βΣ) = (α / sqrt(β)) * MVE_SR(μ, Σ)
        base = compute_mve_sr(μ, Σ)
        α, β = 2.0, 3.0
        scaled = compute_mve_sr(α .* μ, β * Matrix(Σ))
        @test abs(scaled - (α / sqrt(β)) * base) ≤ 100TOL

        # SR(w; αμ, βΣ) = (α / sqrt(β)) * SR(w; μ, Σ)
        w = rand(n); w ./= sum(abs, w) + eps()
        sr_base  = compute_sr(w, μ, Σ)
        sr_scale = compute_sr(w, α .* μ, β * Matrix(Σ))
        @test abs(sr_scale - (α / sqrt(β)) * sr_base) ≤ 100TOL
    end

    # --- degeneracy & ridge behavior -----------------------------------------
    @testset "degenerate covariance & ridge" begin
        # Rank-1 covariance (singular)
        n = 6
        v = ones(n)
        Σ_sing = Symmetric(v*v')  # rank-1
        μ = collect(0.1:0.1:0.6)

        # Pseudoinverse path should work and be finite
        sr_mve = compute_mve_sr(μ, Σ_sing; epsilon=0.0)
        @test isfinite(sr_mve) && sr_mve ≥ 0

        w = compute_mve_weights(μ, Σ_sing; epsilon=0.0)
        sr_w = compute_sr(w, μ, Σ_sing; epsilon=0.0)
        @test abs(sr_w - sr_mve) ≤ 1e-5

        # Add tiny diagonal and ridge — still finite
        Σ_ns = Symmetric(Matrix(Σ_sing) + 1e-4I) # NOTE: plain + with I
        sr_mve_eps = compute_mve_sr(μ, Σ_ns; epsilon=1e-2)
        @test isfinite(sr_mve_eps)
    end

    # --- zero covariance corner case ------------------------------------------
    @testset "zero covariance corner case" begin
        μ = [1.0, 2.0]
        Σ = zeros(2, 2)
        w = [0.5, 0.5]

        # SR should be NaN (variance zero)
        @test isnan(compute_sr(w, μ, Σ; epsilon=0.0))
        @test isnan(compute_sr(w, μ, Σ; epsilon=1e-2))  # mean diag = 0 ⇒ ridge scale 0

        # MVE_SR should be 0 (pinv(0) == 0)
        @test compute_mve_sr(μ, Σ; epsilon=0.0) == 0.0
        @test compute_mve_sr(μ, Σ; epsilon=1e-2) == 0.0

        w_mve = compute_mve_weights(μ, Σ; epsilon=0.0)
        @test w_mve == zeros(2)
    end

    # --- single-asset edge case ----------------------------------------------
    @testset "single-asset" begin
        μ  = [0.2]
        σ2 = 0.04
        Σ  = [σ2;;]            # 1×1 matrix
        w  = [1.0]

        # Use epsilon=0.0 to match the exact closed-form (no ridge)
        @test abs(compute_sr(w, μ, Σ; epsilon=0.0) - (μ[1] / sqrt(σ2))) ≤ 1e-12
        @test abs(compute_mve_sr(μ, Σ; epsilon=0.0) - (abs(μ[1]) / sqrt(σ2))) ≤ 1e-12
        @test isapprox(abs.(compute_mve_weights(μ, Σ; epsilon=0.0)),
                       [μ[1] / (σ2)];
                       rtol=0, atol=1e-12)
    end

    # --- weights consistency: SR(w*) == MVE_SR --------------------------------
    @testset "weights consistency" begin
        Random.seed!(2)
        n = 7
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n, n)
        Σ = Symmetric(A*A' + 0.2I)  # NOTE: plain + with I
        wstar = compute_mve_weights(μ, Σ)  # no normalization by default
        sr_wstar = compute_sr(wstar, μ, Σ)
        mve = compute_mve_sr(μ, Σ)
        @test abs(sr_wstar - mve) ≤ 1e-7
    end

    # --- stabilize_Σ semantics ------------------------------------------------
    @testset "stabilize_Σ semantics" begin
        # For a nearly-symmetric Σ, symmetrization should not change results materially.
        Random.seed!(3)
        n = 5
        A = randn(n, n)
        Σ_raw = A*A' + 1e-1I
        Σ_pert = Σ_raw + 1e-10 * randn(n, n)  # tiny asymmetry
        μ = rand(n)

        s1 = compute_mve_sr(μ, Σ_pert; stabilize_Σ=true,  epsilon=0.0)
        s2 = compute_mve_sr(μ, Σ_pert; stabilize_Σ=false, epsilon=0.0)
        @test isfinite(s1) && isfinite(s2)
        @test abs(s1 - s2) ≤ 1e-8
    end

    @testset "normalize_weights semantics" begin
        Random.seed!(4)
        n = 8
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n, n)
        Σ = Symmetric(A*A' + 0.15I)

        # Raw weights vs normalized weights via Utils.normalize_weights (relative mode)
        w0 = compute_mve_weights(μ, Σ; normalize_weights=false)
        w1 = compute_mve_weights(μ, Σ; normalize_weights=true)

        denom = max(abs(sum(w0)), 1e-6 * norm(w0, 1), 1e-10)
        @test isapprox.(w1, w0 ./ denom; atol=1e-10) |> all

        # SR is scale-invariant
        @test abs(compute_sr(w0, μ, Σ) - compute_sr(w1, μ, Σ)) ≤ 1e-10

        # selection case: zeros off support and same normalization behavior
        sel = sort(randperm(n)[1:5])
        w_raw = compute_mve_weights(μ, Σ; selection=sel, normalize_weights=false)
        ws    = compute_mve_weights(μ, Σ; selection=sel, normalize_weights=true)

        @test all(i -> (i ∈ sel) || (ws[i] == 0.0), 1:n)

        denom_sel = max(abs(sum(w_raw)), 1e-6 * norm(w_raw, 1), 1e-10)
        @test isapprox.(ws, w_raw ./ denom_sel; atol=1e-10) |> all

        @test abs(compute_sr(ws, μ, Σ) - compute_sr(w_raw, μ, Σ)) ≤ 1e-10
    end

    # --- error paths when do_checks=true --------------------------------------
    @testset "argument checks" begin
        μ = [0.1, 0.2]
        Σ = [0.04 0.01; 0.01 0.09]
        w = [0.5, 0.5, 0.5]  # wrong length
        @test_throws ErrorException compute_sr(w, μ, Σ; do_checks=true)

        bad_sel = [0, 3]     # out-of-bounds
        @test_throws ErrorException compute_mve_sr(μ, Σ; selection=bad_sel, do_checks=true)
        @test_throws ErrorException compute_mve_weights(μ, Σ; selection=bad_sel, do_checks=true)

        @test_throws ErrorException compute_mve_sr(μ, ones(2,3); do_checks=true)  # wrong shape
    end
end
