# test/test-MIQPHeuristicSearch.jl
# Tests for mve_miqp_heuristic_search with refit toggle, exactly_k, bounds, normalization,
# indicator linking, budget feasibility screen, and expansion no-op behavior.
#
# Assumptions preserved:
# - normalize_weights=false ⇒ NO budget constraint in the MIQP.
# - normalize_weights=true  ⇒ MIQP imposes ∑w=1; outputs are post-normalized via Utils.normalize_weights.
# - Refit computes MVE weights/SR on the returned selection using the call's stabilization choice.
#
using Test, Random, LinearAlgebra, Statistics
using SparseMaxSR
using SparseMaxSR.SharpeRatio
using SparseMaxSR.MIQPHeuristicSearch
import MathOptInterface as MOI

# Internal SR recomputation mirroring stabilize_Σ=false path (symmetrize only)
_sr_internal(w, μ, Σ; epsilon=0.0) = begin
    Σs = Symmetric((Σ + Σ')/2)
    compute_sr(w, μ, Σs; epsilon=epsilon, stabilize_Σ=false, do_checks=false)
end

# Closed-form MVE SR on a given selection, with chosen stabilization
_refit_sr(μ, Σ; selection, stabilize_Σ::Bool, epsilon=0.0) = begin
    if stabilize_Σ
        compute_mve_sr(μ, Σ; selection=selection, epsilon=epsilon, stabilize_Σ=true, do_checks=false)
    else
        Σs = Symmetric((Σ + Σ')/2)
        compute_mve_sr(μ, Σs; selection=selection, epsilon=epsilon, stabilize_Σ=false, do_checks=false)
    end
end

# Utility
abs_sum(x) = abs(sum(x))
norm1e(x)  = sum(abs, x)

const ATOL_SR = 1e-9
const ATOL_W  = 1e-12

@testset "MIQPHeuristicSearch.mve_miqp_heuristic_search" begin

    @testset "basic smoke & invariants (non-refit, cardinality band, normalize_weights=true)" begin
        Random.seed!(123)
        n, k, m = 8, 3, 1
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n, n)
        Σ = Symmetric(A*A' + 0.10I)
        fmin = zeros(n)
        fmax = ones(n)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                        fmin=fmin, fmax=fmax,
                                        expand_rounds=1, mipgap=1e-5, time_limit=30.0,
                                        threads=1, stabilize_Σ=false,
                                        compute_weights=true, use_refit=false, do_checks=true,
                                        exactly_k=false, normalize_weights=true)

        @test res.status isa MOI.TerminationStatusCode
        @test isfinite(res.sr)
        @test abs(abs_sum(res.weights) - 1.0) ≤ 1e-10

        # cardinality: m ≤ |S| ≤ k   (support via numerical threshold)
        supp = sum(abs.(res.weights) .> 1e-12)
        @test m ≤ supp ≤ k

        # box bounds respected (nonnegativity here because fmin ≥ 0)
        @test all(res.weights .≤ fmax .+ 1e-12)
        @test all(res.weights .≥ -1e-12)

        # SR consistency with same stabilization choice (false)
        @test abs(_sr_internal(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-7
    end

    @testset "normalize_weights: cross-problem comparisons removed; check invariants per-solve" begin
        Random.seed!(1337)
        n, k, m = 10, 4, 1
        μ = 0.02 .+ 0.04 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.08I)

        r0 = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                       stabilize_Σ=false, compute_weights=true,
                                       use_refit=false, threads=1,
                                       exactly_k=false, normalize_weights=false)

        r1 = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                       stabilize_Σ=false, compute_weights=true,
                                       use_refit=false, threads=1,
                                       exactly_k=false, normalize_weights=true)

        # r1 invariants (budgeted problem)
        @test abs(abs_sum(r1.weights) - 1.0) ≤ 1e-10
        @test isfinite(r1.sr)
        @test length(r1.selection) ≤ k && length(r1.selection) ≥ m

        # Scale-invariance verified within a single solution (r0)
        x0n = r0.weights / max(abs_sum(r0.weights), eps())
        @test abs(compute_sr(x0n, μ, Σ) - compute_sr(r0.weights, μ, Σ)) ≤ 1e-10
    end

    @testset "exactly_k=true enforces |S| == k (non-refit, normalize_weights=true)" begin
        Random.seed!(2025)
        n, k = 12, 5
        μ = 0.01 .+ 0.04 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.06I)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1,
                                        exactly_k=true, normalize_weights=true)

        @test length(res.selection) == k
        @test abs(abs_sum(res.weights) - 1.0) ≤ 1e-10
        # Ensure weights outside selection are (numerically) zero
        @test all(i -> (i ∈ res.selection) || abs(res.weights[i]) ≤ 1e-12, 1:length(res.weights))
        @test abs(_sr_internal(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-8
    end

    @testset "exactly_k=true with refit: SR equals closed-form MVE SR on S" begin
        Random.seed!(2026)
        n, k = 10, 4
        μ = 0.015 .+ 0.03 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.04I)

        r_rf = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                         stabilize_Σ=false, compute_weights=true,
                                         use_refit=true, threads=1,
                                         exactly_k=true, normalize_weights=true)

        @test length(r_rf.selection) == k

        # Expected refit SR (closed-form on S, same stabilization choice)
        sr_expected = _refit_sr(μ, Σ; selection=r_rf.selection, stabilize_Σ=false, epsilon=0.0)
        @test isapprox(r_rf.sr, sr_expected; atol=ATOL_SR, rtol=0)

        # Returned weights should attain that SR
        sr_w = compute_sr(r_rf.weights, μ, Σ; selection=r_rf.selection, stabilize_Σ=false, epsilon=0.0)
        @test isapprox(sr_w, sr_expected; atol=1e-8, rtol=0)

        # Budget normalization (absolute sum equals 1 due to budget constraint)
        @test abs(abs_sum(r_rf.weights) - 1.0) ≤ 1e-10
    end

    @testset "exactly_k=true overrides m (if provided): result still has |S|==k" begin
        Random.seed!(2027)
        n, k, m = 11, 3, 1
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.05I)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1,
                                        exactly_k=true, normalize_weights=true)

        @test length(res.selection) == k
        @test abs(abs_sum(res.weights) - 1.0) ≤ 1e-10
    end

    @testset "k = 1 diagonal-Σ: picks argmax of μ_i - 0.5γσ_i^2 (non-refit)" begin
        μ  = [0.10, 0.28, 0.18, 0.05]
        σ2 = [0.04, 0.20, 0.09, 0.01]
        Σ  = Symmetric(Diagonal(σ2) |> Matrix)
        γ  = 2.0
        k  = 1

        score = μ .- 0.5 .* γ .* σ2
        best  = argmax(score)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, m=1, γ=γ,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1,
                                        exactly_k=true, normalize_weights=true)

        @test sum(abs.(res.weights) .> 1e-12) == 1
        @test findmax(res.weights)[2] == best
        @test abs(res.weights[best] - 1.0) ≤ 1e-10
        @test abs(_sr_internal(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-9
    end

    @testset "use_refit=true: same selection as non-refit; SR equals closed-form refit SR" begin
        Random.seed!(321)
        n, k, m = 10, 4, 0
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.05I)

        r_nr = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                         stabilize_Σ=false, compute_weights=true,
                                         use_refit=false, threads=1,
                                         exactly_k=true, normalize_weights=true)

        r_rf = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                         stabilize_Σ=false, compute_weights=true,
                                         use_refit=true, threads=1,
                                         exactly_k=true, normalize_weights=true)

        @test r_rf.selection == r_nr.selection

        sr_expected = _refit_sr(μ, Σ; selection=r_nr.selection, stabilize_Σ=false, epsilon=0.0)
        @test isapprox(r_rf.sr, sr_expected; atol=ATOL_SR, rtol=0)

        w = r_rf.weights
        sr_refit = compute_sr(w, μ, Σ; selection=r_rf.selection, stabilize_Σ=false, epsilon=0.0)
        sr_star  = compute_mve_sr(μ, Σ; selection=r_rf.selection, stabilize_Σ=false, epsilon=0.0)

        @test isfinite(sr_refit)
        @test abs(sr_refit - sr_star) ≤ 1e-8
        @test r_rf.sr + 1e-12 ≥ r_nr.sr  # refit does not worsen SR
    end

    @testset "use_refit=true with compute_weights=false returns zero vector but correct SR" begin
        Random.seed!(777)
        n, k, m = 9, 3, 0
        μ = 0.02 .+ 0.02 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.03I)

        r_rf0 = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                          stabilize_Σ=false, compute_weights=false,
                                          use_refit=true, threads=1,
                                          exactly_k=true)

        @test all(iszero, r_rf0.weights)
        sr_expected = _refit_sr(μ, Σ; selection=r_rf0.selection, stabilize_Σ=false, epsilon=0.0)
        @test isapprox(r_rf0.sr, sr_expected; atol=ATOL_SR, rtol=0)
    end

    @testset "enforcing m (min cardinality) and k (non-refit, exactly_k=false)" begin
        Random.seed!(321)
        n, k, m = 7, 4, 2
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.05I)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1,
                                        exactly_k=false, normalize_weights=true)

        supp = sum(abs.(res.weights) .> 1e-12)
        @test m ≤ supp ≤ k
        @test abs(abs_sum(res.weights) - 1.0) ≤ 1e-10
        @test abs(_sr_internal(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-7
    end

    @testset "respecting fmin/fmax bounds (non-refit)" begin
        Random.seed!(777)
        n, k, m = 6, 3, 2
        μ = 0.02 .+ 0.02 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.03I)

        fmin = fill(0.0, n); fmin[1:2] .= 0.05  # active ones must be ≥ 5%
        fmax = fill(0.8, n); fmax[3]   = 0.4    # cap asset 3 at 40%

        res = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                        fmin=fmin, fmax=fmax,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1,
                                        exactly_k=false, normalize_weights=true)

        x = res.weights
        @test all(x .≤ fmax .+ 1e-12)
        for i in 1:n
            if x[i] > 1e-10
                @test x[i] + 1e-12 ≥ fmin[i]
            end
        end
        @test abs(abs_sum(x) - 1.0) ≤ 1e-10
        @test isfinite(res.sr)
    end

    @testset "expand_rounds: does not worsen SR when caps bind (non-refit)" begin
        Random.seed!(888)
        n, k = 8, 3
        μ = 0.02 .+ 0.04 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.08I)

        fmin = zeros(n)
        fmax = fill(1.0 / k, n)

        r0  = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                        fmin=fmin, fmax=fmax,
                                        expand_rounds=0, stabilize_Σ=false,
                                        use_refit=false, threads=1,
                                        exactly_k=false, normalize_weights=true)
        r2  = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                        fmin=fmin, fmax=fmax,
                                        expand_rounds=2, expand_factor=2.0, expand_tol=1e-9,
                                        stabilize_Σ=false, use_refit=false, threads=1,
                                        exactly_k=false, normalize_weights=true)

        @test r2.sr + 1e-12 ≥ r0.sr
    end

    @testset "epsilon regularization on near-singular Σ (non-refit)" begin
        Random.seed!(999)
        n, k = 10, 3
        v = ones(n)
        Σsing = Symmetric(v*v')              # rank-1
        μ = 0.01 .+ 0.01 .* rand(n)

        r0 = mve_miqp_heuristic_search(μ, Σsing; k=k, γ=1.0,
                                       epsilon=0.0, stabilize_Σ=false,
                                       use_refit=false, threads=1,
                                       exactly_k=false, normalize_weights=true)
        @test isfinite(r0.sr)

        rE = mve_miqp_heuristic_search(μ, Σsing; k=k, γ=1.0,
                                       epsilon=1e-2, stabilize_Σ=false,
                                       use_refit=false, threads=1,
                                       exactly_k=false, normalize_weights=true)
        @test isfinite(rE.sr)
    end

    @testset "determinism with threads=1 (same inputs ⇒ same outputs) (non-refit)" begin
        Random.seed!(2024)
        n, k = 9, 3
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.05I)

        r1 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                       stabilize_Σ=false, compute_weights=true,
                                       use_refit=false, threads=1,
                                       exactly_k=true, normalize_weights=true)
        r2 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                       stabilize_Σ=false, compute_weights=true,
                                       use_refit=false, threads=1,
                                       exactly_k=true, normalize_weights=true)

        @test isapprox(r1.sr, r2.sr; atol=1e-12, rtol=0)
        @test isapprox(r1.weights, r2.weights; atol=1e-12, rtol=0)
        @test r1.selection == r2.selection
    end

    @testset "warm starts reproduce the same solution (non-refit)" begin
        Random.seed!(42)
        n, k = 8, 3
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.07I)

        r = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                      stabilize_Σ=false, compute_weights=true,
                                      use_refit=false, threads=1,
                                      exactly_k=true, normalize_weights=true)

        v0 = zeros(Int, n); v0[findall(>(1e-10), r.weights)] .= 1

        r_restart = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                              x_start=r.weights, v_start=v0,
                                              stabilize_Σ=false, compute_weights=true,
                                              use_refit=false, threads=1,
                                              exactly_k=true, normalize_weights=true)

        @test r_restart.selection == r.selection
        @test isapprox(r_restart.weights, r.weights; atol=1e-12, rtol=0)
        @test isapprox(r_restart.sr, r.sr; atol=1e-12, rtol=0)
    end

    @testset "argument checks (do_checks=true)" begin
        μ = [0.1, 0.2]
        Σ = [0.04 0.01; 0.01 0.09]
        @test_throws ErrorException mve_miqp_heuristic_search(μ, ones(2,3); k=1, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=0, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=3, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=1, m=2, do_checks=true, exactly_k=false)
        @test_throws ErrorException mve_miqp_heuristic_search([0.1, Inf], Σ; k=1, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, [NaN 0; 0 1]; k=1, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=1, γ=-1.0, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=1, expand_rounds=-1, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=1, expand_factor=0.0, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=1, expand_tol=-1e-3, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=1, mipgap=-1e-4, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=1, threads=-2, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=2, fmin=[0.0], do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=2, fmax=[1.0], do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=2, fmin=[0.2, 0.3], fmax=[0.1, 0.4], do_checks=true)
    end

    # ──────────────────────────────────────────────────────────────────────────
    # NEW: Indicator-linking stress with shorting (negative fmin)
    # ──────────────────────────────────────────────────────────────────────────
    @testset "indicator linking with shorting (non-refit, normalize_weights=false)" begin
        Random.seed!(4545)
        n, k, m = 8, 4, 1
        μ = -0.01 .+ 0.04 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.05I)

        # allow shorting on half of the names
        fmin = fill(-0.3, n); fmin[1:div(n,2)] .= -0.5
        fmax = fill( 0.6, n); fmax[1:div(n,2)] .=  0.4

        r = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                      fmin=fmin, fmax=fmax,
                                      stabilize_Σ=false, compute_weights=true,
                                      use_refit=false, threads=1,
                                      exactly_k=false, normalize_weights=false)

        x = r.weights
        # infer selection from weights (v=0 ⇒ x=0 exactly by indicator; allow tiny tol)
        sel = findall(abs.(x) .> 1e-12)
        @test length(sel) ≥ m && length(sel) ≤ k
        # bounds respected on active names; zeros off support
        for i in 1:n
            if i ∈ sel
                @test x[i] ≤ fmax[i] + 1e-12
                @test x[i] ≥ fmin[i] - 1e-12
            else
                @test abs(x[i]) ≤ 1e-12
            end
        end
        @test isfinite(r.sr)
    end

    # ──────────────────────────────────────────────────────────────────────────
    # NEW: Budget feasibility quick screen (normalize_weights=true + do_checks=true)
    # ──────────────────────────────────────────────────────────────────────────
    @testset "budget feasibility pre-screen triggers error when caps make ∑x=1 impossible" begin
        # Make it provably impossible: N small and sum of max caps < 1
        n, k, m = 5, 4, 1
        μ = 0.01 .+ 0.02 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.02I)
        fmin = zeros(n)
        fmax = fill(0.10, n)  # total max ≤ 0.5

        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                                              fmin=fmin, fmax=fmax,
                                                              normalize_weights=true,
                                                              do_checks=true,
                                                              stabilize_Σ=false,
                                                              use_refit=false, threads=1)
    end

    # ──────────────────────────────────────────────────────────────────────────
    # NEW: No-op expansion (when nothing is near bounds, extra rounds don't change result)
    # ──────────────────────────────────────────────────────────────────────────
    @testset "no-op expansion: expand_rounds>0 equals expand_rounds=0 when bounds are slack" begin
        Random.seed!(1212)
        n, k = 10, 4
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.07I)
        # very loose bounds
        fmin = fill(-10.0, n)
        fmax = fill( 10.0, n)

        r0 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                       fmin=fmin, fmax=fmax,
                                       expand_rounds=0, stabilize_Σ=false,
                                       use_refit=false, threads=1,
                                       exactly_k=true, normalize_weights=false)

        r5 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                       fmin=fmin, fmax=fmax,
                                       expand_rounds=5, stabilize_Σ=false,
                                       use_refit=false, threads=1,
                                       exactly_k=true, normalize_weights=false)

        @test r0.selection == r5.selection
        @test isapprox(r0.sr, r5.sr; atol=1e-12, rtol=0)
        @test isapprox(r0.weights, r5.weights; atol=1e-12, rtol=0)
    end

    @testset "time_limit: finite SR with tight budget (mode-agnostic)" begin
        Random.seed!(1414)
        n, k = 14, 4
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.05I)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                        time_limit=0.1, mipgap=1e-3, threads=1,
                                        stabilize_Σ=false, use_refit=false,
                                        exactly_k=false)  # default normalize_weights=false
        @test isfinite(res.sr)

        res2 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                         time_limit=0.1, mipgap=1e-3, threads=1,
                                         stabilize_Σ=false, use_refit=true,
                                         exactly_k=true)   # default normalize_weights=false
        @test isfinite(res2.sr)
    end
end