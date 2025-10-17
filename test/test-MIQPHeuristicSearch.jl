# test-MIQPHeuristicSearch.jl — tests for mve_miqp_heuristic_search with refit toggle

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
        # When the search is called with stabilize_Σ=true, it internally uses _prep_S.
        # We don't have _prep_S here; rely on compute_mve_sr with stabilize_Σ=true to mirror that.
        compute_mve_sr(μ, Σ; selection=selection, epsilon=epsilon, stabilize_Σ=true, do_checks=false)
    else
        # When stabilize_Σ=false, the search symmetrizes once and then calls with stabilize_Σ=false.
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
                                        compute_weights=true, use_refit=false, do_checks=true)

        @test res.status isa MOI.TerminationStatusCode
        @test isfinite(res.mve_sr)
        @test abs(sum(res.mve_weights) - 1.0) ≤ 1e-10
        # cardinality: count positive weights approximates chosen support
        supp = sum(abs.(res.mve_weights) .> 1e-12)
        @test m ≤ supp ≤ k
        # box bounds respected for active coords; and never exceed fmax anywhere
        @test all(res.mve_weights .≤ fmax .+ 1e-12)
        @test all(res.mve_weights[res.mve_weights .> 1e-12] .≥ minimum(fmin) - 1e-12)
        # SR consistency with the same stabilization choice
        @test abs(_sr_internal(res.mve_weights, μ, Σ; epsilon=0.0) - res.mve_sr) ≤ 1e-7
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
                                        use_refit=false, threads=1)
        @test sum(abs.(res.mve_weights) .> 1e-12) == 1
        @test findmax(res.mve_weights)[2] == best
        @test abs(res.mve_weights[best] - 1.0) ≤ 1e-10
        @test abs(_sr_internal(res.mve_weights, μ, Σ; epsilon=0.0) - res.mve_sr) ≤ 1e-9
    end

    @testset "use_refit=true: selection equality, SR equals closed-form refit SR" begin
        Random.seed!(321)
        n, k, m = 10, 4, 0
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.05I)

        # First run non-refit to get MIQP x & selection
        r_nr = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1)

        # Now refit on the same instance
        r_rf = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=true, threads=1)

        # Selection should be identical (refit uses MIQP support)
        @test r_rf.mve_selection == r_nr.mve_selection

        # SR should match closed-form refit SR on that support (same stabilization choice)
        sr_expected = _refit_sr(μ, Σ; selection=r_nr.mve_selection,
                                stabilize_Σ=false, epsilon=0.0)
        @test isapprox(r_rf.mve_sr, sr_expected; atol=1e-9, rtol=0)

        # Sharpe ratio of the returned refit weights must equal closed-form SR on that support
        w = r_rf.mve_weights
        sr_refit = SparseMaxSR.compute_sr(w, μ, Σ; selection=r_rf.mve_selection,
                                        stabilize_Σ=false, epsilon=0.0)
        sr_star  = SparseMaxSR.compute_mve_sr(μ, Σ; selection=r_rf.mve_selection,
                                            stabilize_Σ=false, epsilon=0.0)

        @test isfinite(sr_refit)
        @test abs(sr_refit - sr_star) ≤ 1e-8

        # Typically (no box constraints here) refit SR ≥ MIQP SR
        @test r_rf.mve_sr + 1e-12 ≥ r_nr.mve_sr
    end

    @testset "use_refit=true with compute_weights=false returns zero vector but correct SR" begin
        Random.seed!(777)
        n, k, m = 9, 3, 0
        μ = 0.02 .+ 0.02 .* rand(n)
        A = randn(n, n); Σ = Symmetric(A*A' + 0.03I)

        r_rf0 = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                          stabilize_Σ=false, compute_weights=false,
                                          use_refit=true, threads=1)

        @test isempty(setdiff(findall(!iszero, r_rf0.mve_weights), Int[]))  # all zeros
        sr_expected = _refit_sr(μ, Σ; selection=r_rf0.mve_selection, stabilize_Σ=false, epsilon=0.0)
        @test isapprox(r_rf0.mve_sr, sr_expected; atol=1e-9, rtol=0)
    end

    @testset "enforcing m (minimum cardinality) and k (non-refit)" begin
        Random.seed!(321)
        n, k, m = 7, 4, 2
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        res = mve_miqp_heuristic_search(μ, Σ; k=k, m=m, γ=1.0,
                                        stabilize_Σ=false, compute_weights=true,
                                        use_refit=false, threads=1)
        supp = sum(abs.(res.mve_weights) .> 1e-12)
        @test m ≤ supp ≤ k
        @test abs(sum(res.mve_weights) - 1.0) ≤ 1e-10
        @test abs(_sr_internal(res.mve_weights, μ, Σ; epsilon=0.0) - res.mve_sr) ≤ 1e-7
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
                                        use_refit=false, threads=1)

        x = res.mve_weights
        @test all(x .≤ fmax .+ 1e-12)
        for i in 1:n
            if x[i] > 1e-10
                @test x[i] + 1e-12 ≥ fmin[i]
            end
        end
        @test abs(sum(x) - 1.0) ≤ 1e-10
        @test isfinite(res.mve_sr)
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
                                        use_refit=false, threads=1)
        r2  = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                        fmin=fmin, fmax=fmax,
                                        expand_rounds=2, expand_factor=2.0, expand_tol=1e-9,
                                        stabilize_Σ=false, use_refit=false, threads=1)
        @test r2.mve_sr + 1e-12 ≥ r0.mve_sr
    end

    @testset "epsilon regularization works on near-singular Σ (non-refit)" begin
        Random.seed!(999)
        n, k = 10, 3
        v = ones(n)
        Σsing = Symmetric(v*v')              # rank-1
        μ = 0.01 .+ 0.01 .* rand(n)

        r0 = mve_miqp_heuristic_search(μ, Σsing; k=k, γ=1.0,
                                       epsilon=0.0, stabilize_Σ=false,
                                       use_refit=false, threads=1)
        @test isfinite(r0.mve_sr)

        rE = mve_miqp_heuristic_search(μ, Σsing; k=k, γ=1.0,
                                       epsilon=1e-2, stabilize_Σ=false,
                                       use_refit=false, threads=1)
        @test isfinite(rE.mve_sr)
    end

    @testset "determinism with threads=1 (same inputs => same outputs) (non-refit)" begin
        Random.seed!(2024)
        n, k = 9, 3
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        r1 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                       stabilize_Σ=false, compute_weights=true,
                                       use_refit=false, threads=1)
        r2 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                       stabilize_Σ=false, compute_weights=true,
                                       use_refit=false, threads=1)

        @test isapprox(r1.mve_sr, r2.mve_sr; atol=0, rtol=0)
        @test isapprox(r1.mve_weights, r2.mve_weights; atol=0, rtol=0)
        @test r1.mve_selection == r2.mve_selection
    end

    @testset "warm starts reproduce the same solution (non-refit)" begin
        Random.seed!(42)
        n, k = 8, 3
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.07I)

        r = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                      stabilize_Σ=false, compute_weights=true,
                                      use_refit=false, threads=1)

        v0 = zeros(Int, n); v0[findall(>(1e-10), r.mve_weights)] .= 1

        r_restart = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                              x_start=r.mve_weights, v_start=v0,
                                              stabilize_Σ=false, compute_weights=true,
                                              use_refit=false, threads=1)

        @test r_restart.mve_selection == r.mve_selection
        @test isapprox(r_restart.mve_weights, r.mve_weights; atol=0, rtol=0)
        @test isapprox(r_restart.mve_sr, r.mve_sr; atol=0, rtol=0)
    end

    @testset "argument checks (do_checks=true)" begin
        μ = [0.1, 0.2]
        Σ = [0.04 0.01; 0.01 0.09]
        @test_throws ErrorException mve_miqp_heuristic_search(μ, ones(2,3); k=1, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=0, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=3, do_checks=true)
        @test_throws ErrorException mve_miqp_heuristic_search(μ, Σ; k=1, m=2, do_checks=true)
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
                                        stabilize_Σ=false, use_refit=false)
        @test isfinite(res.mve_sr)

        res2 = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                         time_limit=0.1, mipgap=1e-3, threads=1,
                                         stabilize_Σ=false, use_refit=true)
        @test isfinite(res2.mve_sr)
    end
end
