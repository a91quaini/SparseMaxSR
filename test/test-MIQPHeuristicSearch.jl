# test-MIQPHeuristicSearch.jl — tests for mve_miqp_heuristic_search with refit toggle + exactly_k

using Test, Random, LinearAlgebra, Statistics
using SparseMaxSR
using SparseMaxSR.SharpeRatio
using SparseMaxSR.MIQPHeuristicSearch
import MathOptInterface as MOI

# Recompute SR the same way the solver does when stabilize_Σ=false:
_sr_internal(w, μ, Σ; epsilon=0.0) = begin
    Σs = Symmetric((Σ + Σ')/2)
    compute_sr(w, μ, Σs; epsilon=epsilon, stabilize_Σ=false, do_checks=false)
end

# Recompute REFIT SR on a given selection using the same Σ stabilization choice as the call
_refit_sr(μ, Σ; selection, stabilize_Σ::Bool, epsilon=0.0) = begin
    if stabilize_Σ
        # Mirror internal stabilization behavior when requested
        compute_mve_sr(μ, Σ; selection=selection, epsilon=epsilon, stabilize_Σ=true, do_checks=false)
    else
        Σs = Symmetric((Σ + Σ')/2)
        compute_mve_sr(μ, Σs; selection=selection, epsilon=epsilon, stabilize_Σ=false, do_checks=false)
    end
end

@testset "MIQPHeuristicSearch.mve_miqp_heuristic_search" begin

    @testset "basic smoke & invariants (non-refit, sum-to-one, cardinality bounds)" begin
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
                                        exactly_k=false)  # explicit: old behavior

        @test res.status isa MOI.TerminationStatusCode
        @test isfinite(res.sr)
        @test abs(sum(res.weights) - 1.0) ≤ 1e-10
        # cardinality: m ≤ |S| ≤ k
        supp = sum(abs.(res.weights) .> 1e-12)
        @test m ≤ supp ≤ k
        # box bounds respected for active coords; and never exceed fmax anywhere
        @test all(res.weights .≤ fmax .+ 1e-12)
        @test all(res.weights[res.weights .> 1e-12] .≥ minimum(fmin) - 1e-12)
        # SR consistency with the same stabilization choice
        @test abs(_sr_internal(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-7
    end

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: exactly_k=true — strict cardinality (non-refit)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "exactly_k=true enforces |S| == k (non-refit)" begin
        Random.seed!(2025)
        n, k = 12, 5
        μ = 0.01 .+ 0.04 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.06I)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1,
                                        exactly_k=true)

        @test length(res.selection) == k
        @test abs(sum(res.weights) - 1.0) ≤ 1e-10
        @test count(!iszero, res.weights) ≥ k  # zeros outside selection allowed, active ≥ k numerically
        # SR recompute
        @test abs(_sr_internal(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-8
    end

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: exactly_k=true — strict cardinality (refit)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "exactly_k=true with refit: |S| == k and SR equals closed-form on S" begin
        Random.seed!(2026)
        n, k = 10, 4
        μ = 0.015 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.04I)

        r_rf = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                         stabilize_Σ=false, compute_weights=true,
                                         use_refit=true, threads=1,
                                         exactly_k=true)

        @test length(r_rf.selection) == k
        # Refitted SR must match the closed-form MVE SR on the same subset
        sr_expected = _refit_sr(μ, Σ; selection=r_rf.selection, stabilize_Σ=false, epsilon=0.0)
        @test isapprox(r_rf.sr, sr_expected; atol=1e-9, rtol=0)
        # Returned weights should also attain that SR
        sr_w = compute_sr(r_rf.weights, μ, Σ; selection=r_rf.selection, stabilize_Σ=false, epsilon=0.0)
        @test isapprox(sr_w, sr_expected; atol=1e-8, rtol=0)
    end

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: exactly_k=true overrides m (if provided inconsistently)
    # ─────────────────────────────────────────────────────────────────────────
    @testset "exactly_k=true overrides/ignores m: result still has |S|==k" begin
        Random.seed!(2027)
        n, k, m = 11, 3, 1   # inconsistent only if someone expects m to bind with equality
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1,
                                        exactly_k=true)

        @test length(res.selection) == k
        @test abs(sum(res.weights) - 1.0) ≤ 1e-10
    end

    @testset "k = 1 diagonal-Σ case selects argmax of μ_i - 0.5γσ_i^2 (non-refit)" begin
        μ = [0.10, 0.28, 0.18, 0.05]
        σ2 = [0.04, 0.20, 0.09, 0.01]
        Σ  = Symmetric(Diagonal(σ2) |> Matrix)
        γ  = 2.0
        k  = 1

        score = μ .- 0.5 .* γ .* σ2
        best  = argmax(score)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, m=1, γ=γ,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1,
                                        exactly_k=true)  # strict 1-of-N

        @test sum(abs.(res.weights) .> 1e-12) == 1
        @test findmax(res.weights)[2] == best
        @test abs(res.weights[best] - 1.0) ≤ 1e-10
        @test abs(_sr_internal(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-9
    end

    @testset "use_refit=true: selection equality, SR equals closed-form refit SR" begin
        Random.seed!(321)
        n, k, m = 10, 4, 0
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.05I)

        r_nr = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                         stabilize_Σ=false, compute_weights=true,
                                         use_refit=false, threads=1,
                                         exactly_k=true)

        r_rf = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                         stabilize_Σ=false, compute_weights=true,
                                         use_refit=true, threads=1,
                                         exactly_k=true)

        @test r_rf.selection == r_nr.selection

        sr_expected = _refit_sr(μ, Σ; selection=r_nr.selection,
                                stabilize_Σ=false, epsilon=0.0)
        @test isapprox(r_rf.sr, sr_expected; atol=1e-9, rtol=0)

        w = r_rf.weights
        sr_refit = SparseMaxSR.compute_sr(w, μ, Σ; selection=r_rf.selection,
                                          stabilize_Σ=false, epsilon=0.0)
        sr_star  = SparseMaxSR.compute_mve_sr(μ, Σ; selection=r_rf.selection,
                                              stabilize_Σ=false, epsilon=0.0)

        @test isfinite(sr_refit)
        @test abs(sr_refit - sr_star) ≤ 1e-8
        @test r_rf.sr + 1e-12 ≥ r_nr.sr
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

        @test isempty(setdiff(findall(!iszero, r_rf0.weights), Int[]))  # all zeros
        sr_expected = _refit_sr(μ, Σ; selection=r_rf0.selection, stabilize_Σ=false, epsilon=0.0)
        @test isapprox(r_rf0.sr, sr_expected; atol=1e-9, rtol=0)
    end

    @testset "enforcing m (minimum cardinality) and k (non-refit, exactly_k=false)" begin
        Random.seed!(321)
        n, k, m = 7, 4, 2
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1,
                                        exactly_k=false)
        supp = sum(abs.(res.weights) .> 1e-12)
        @test m ≤ supp ≤ k
        @test abs(sum(res.weights) - 1.0) ≤ 1e-10
        @test abs(_sr_internal(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-7
    end

    @testset "respecting fmin/fmax bounds (non-refit)" begin
        Random.seed!(777)
        n, k, m = 6, 3, 2
        μ = 0.02 .+ 0.02 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.03I)

        fmin = fill(0.0, n); fmin[1:2] .= 0.05  # if chosen, at least 5% each
        fmax = fill(0.8, n); fmax[3] = 0.4      # cap asset 3 at 40%

        res = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                        fmin=fmin, fmax=fmax,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1,
                                        exactly_k=false)

        x = res.weights
        @test all(x .≤ fmax .+ 1e-12)
        for i in 1:n
            if x[i] > 1e-10
                @test x[i] + 1e-12 ≥ fmin[i]
            end
        end
        @test abs(sum(x) - 1.0) ≤ 1e-10
        @test isfinite(res.sr)
    end

    @testset "expand_rounds improve (or not worsen) SR when caps bind (non-refit)" begin
        Random.seed!(888)
        n, k = 8, 3
        μ = 0.02 .+ 0.04 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.08I)

        fmin = zeros(n)
        fmax = fill(1.0 / k, n)

        r0  = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                        fmin=fmin, fmax=fmax,
                                        expand_rounds=0, stabilize_Σ=false,
                                        use_refit=false, threads=1,
                                        exactly_k=false)
        r2  = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                        fmin=fmin, fmax=fmax,
                                        expand_rounds=2, expand_factor=2.0, expand_tol=1e-9,
                                        stabilize_Σ=false, use_refit=false, threads=1,
                                        exactly_k=false)
        @test r2.sr + 1e-12 ≥ r0.sr
    end

    @testset "epsilon regularization works on near-singular Σ (non-refit)" begin
        Random.seed!(999)
        n, k = 10, 3
        v = ones(n)
        Σsing = Symmetric(v*v')              # rank-1
        μ = 0.01 .+ 0.01 .* rand(n)

        r0 = mve_miqp_heuristic_search(μ, Σsing; k=k, γ=1.0,
                                       epsilon=0.0, stabilize_Σ=false,
                                       use_refit=false, threads=1,
                                       exactly_k=false)
        @test isfinite(r0.sr)

        rE = mve_miqp_heuristic_search(μ, Σsing; k=k, γ=1.0,
                                       epsilon=1e-2, stabilize_Σ=false,
                                       use_refit=false, threads=1,
                                       exactly_k=false)
        @test isfinite(rE.sr)
    end

    @testset "determinism with threads=1 (same inputs => same outputs) (non-refit)" begin
        Random.seed!(2024)
        n, k = 9, 3
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        r1 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                       stabilize_Σ=false, compute_weights=true,
                                       use_refit=false, threads=1,
                                       exactly_k=true)
        r2 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                       stabilize_Σ=false, compute_weights=true,
                                       use_refit=false, threads=1,
                                       exactly_k=true)

        @test isapprox(r1.sr, r2.sr; atol=0, rtol=0)
        @test isapprox(r1.weights, r2.weights; atol=0, rtol=0)
        @test r1.selection == r2.selection
    end

    @testset "warm starts reproduce the same solution (non-refit)" begin
        Random.seed!(42)
        n, k = 8, 3
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.07I)

        r = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                      stabilize_Σ=false, compute_weights=true,
                                      use_refit=false, threads=1,
                                      exactly_k=true)

        v0 = zeros(Int, n); v0[findall(>(1e-10), r.weights)] .= 1

        r_restart = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                              x_start=r.weights, v_start=v0,
                                              stabilize_Σ=false, compute_weights=true,
                                              use_refit=false, threads=1,
                                              exactly_k=true)

        @test r_restart.selection == r.selection
        @test isapprox(r_restart.weights, r.weights; atol=0, rtol=0)
        @test isapprox(r_restart.sr, r.sr; atol=0, rtol=0)
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

    @testset "time_limit: still returns a finite SR with tight time budget (mode-agnostic)" begin
        Random.seed!(1414)
        n, k = 14, 4
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                        time_limit=0.1, mipgap=1e-3, threads=1,
                                        stabilize_Σ=false, use_refit=false,
                                        exactly_k=false)
        @test isfinite(res.sr)

        res2 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                         time_limit=0.1, mipgap=1e-3, threads=1,
                                         stabilize_Σ=false, use_refit=true,
                                         exactly_k=true)
        @test isfinite(res2.sr)
    end
end
