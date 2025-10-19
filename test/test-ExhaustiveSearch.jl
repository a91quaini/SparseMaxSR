
# test-ExhaustiveSearch.jl — extensive tests for mve_exhaustive_search
# Works with the current SparseMaxSR API.

using Test, Random, LinearAlgebra, Statistics
using SparseMaxSR
using SparseMaxSR.SharpeRatio
using Combinatorics: combinations

# --- helpers ------------------------------------------------------------------

# brute-force best SR (for exactly size s)
function _bruteforce_best_sr(μ, Σ; s::Int, epsilon=0.0, stabilize_Σ=true)
    n = length(μ)
    Σs = stabilize_Σ ? Symmetric((Σ + Σ')/2) : Symmetric((Σ + Σ')/2) # keep symmetric either way
    best_sr = -Inf
    best_set = Int[]
    for idxs in combinations(1:n, s)
        sr = compute_mve_sr(μ, Σs; selection=idxs, epsilon=epsilon,
                            stabilize_Σ=false, do_checks=false)
        if sr > best_sr
            best_sr = sr
            best_set = collect(idxs)
        end
    end
    return best_set, best_sr
end

# Exact SR recompute with the package's stabilization rule (no double ridge)
_sr_exact(w, μ, Σ; epsilon=0.0, stabilize_Σ=true) = begin
    S_eff = SparseMaxSR.Utils._prep_S(Matrix(Σ), epsilon, stabilize_Σ)  # Symmetric(...)
    # S_eff is already stabilized, so pass stabilize_Σ=false to avoid double work
    compute_sr(w, μ, S_eff; epsilon=0.0, stabilize_Σ=false, do_checks=false)
end

# --- tests --------------------------------------------------------------------

@testset "ExhaustiveSearch.mve_exhaustive_search" begin

    @testset "basic exact search equals brute force (exactly_k=true)" begin
        Random.seed!(42)
        n, k = 6, 3                       # C(6,3)=20 — small and safe
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n,n)
        Σ = Symmetric(A*A' + 0.10I)

        brute_set, brute_sr = _bruteforce_best_sr(μ, Σ; s=k, epsilon=0.0, stabilize_Σ=true)

        res = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=0,
                                    epsilon=0.0, rng=Random.default_rng(),
                                    stabilize_Σ=true, compute_weights=true, do_checks=true)

        @test res.status == :EXHAUSTIVE
        @test length(res.selection) == k
        @test isapprox(res.sr, brute_sr; atol=1e-10, rtol=0)
        @test res.selection == brute_set  # combinations() order is deterministic
        # weights consistency
        @test count(!iszero, res.weights) == k
        @test abs(_sr_exact(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-7
    end

    @testset "exactly_k=false returns best over sizes 1..k" begin
        Random.seed!(7)
        n, k = 6, 4
        μ = 0.01 .+ 0.02 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        # brute best over all s=1..k
        best_sr = -Inf
        best_set = Int[]
        for s in 1:k
            set_s, sr_s = _bruteforce_best_sr(μ, Σ; s=s, epsilon=0.0, stabilize_Σ=true)
            if sr_s > best_sr
                best_sr, best_set = sr_s, set_s
            end
        end

        res = mve_exhaustive_search(μ, Σ, k; exactly_k=false, max_samples_per_k=0,
                                    epsilon=0.0, stabilize_Σ=true, compute_weights=true)
        @test res.status == :EXHAUSTIVE
        @test res.selection == best_set
        @test isapprox(res.sr, best_sr; atol=1e-10, rtol=0)
        @test count(!iszero, res.weights) == length(best_set)
        @test abs(_sr_exact(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-7
    end

    @testset "sampled mode equals exhaustive when sample cap ≥ total" begin
        Random.seed!(11)
        n, k = 7, 3                       # C(7,3)=35
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.03I)

        res_exh = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=0,
                                        epsilon=0.0, stabilize_Σ=true, compute_weights=true)
        res_cap = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=35,  # ≥ total
                                        epsilon=0.0, stabilize_Σ=true, compute_weights=true)
        @test res_cap.selection == res_exh.selection
        @test isapprox(res_cap.sr, res_exh.sr; atol=0, rtol=0)
    end

    @testset "RNG reproducibility in sampled mode" begin
        Random.seed!(99)
        n, k = 9, 3                       # C(9,3)=84; sample << total
        μ = 0.03 .+ 0.02 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        rng1 = MersenneTwister(123)
        rng2 = MersenneTwister(123)
        r1 = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=15, epsilon=0.0,
                                   rng=rng1, stabilize_Σ=true, compute_weights=true)
        r2 = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=15, epsilon=0.0,
                                   rng=rng2, stabilize_Σ=true, compute_weights=true)
        @test r1.selection == r2.selection
        @test isapprox(r1.sr, r2.sr; atol=0, rtol=0)
    end

    @testset "k=1 picks asset with highest μ/√σ² for diagonal Σ" begin
        μ = [0.1, 0.3, 0.2, 0.05]
        diagσ2 = [0.04, 0.25, 0.09, 0.01]  # variances
        Σ = Diagonal(diagσ2) |> Matrix |> Symmetric
        k = 1

        # best idx by analytic rule
        scores = μ ./ sqrt.(diagσ2)
        best = argmax(scores)

        res = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=0,
                                    epsilon=0.0, stabilize_Σ=true, compute_weights=true)
        @test res.selection == [best]
        @test count(!iszero, res.weights) == 1
        @test abs(_sr_exact(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-9
    end

    @testset "k=n selects all assets; matches global MVE SR" begin
        Random.seed!(1)
        n = 6
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        res = mve_exhaustive_search(μ, Σ, n; exactly_k=true, max_samples_per_k=0,
                                    epsilon=0.0, stabilize_Σ=true, compute_weights=true)
        @test res.selection == collect(1:n)
        mve = compute_mve_sr(μ, Symmetric((Σ + Σ')/2); epsilon=0.0, stabilize_Σ=false)
        @test isapprox(res.sr, mve; atol=1e-9, rtol=0)
        @test abs(_sr_exact(res.weights, μ, Σ; epsilon=0.0) - mve) ≤ 1e-9
    end

    @testset "near-singular Σ and epsilon stabilization" begin
        Random.seed!(5)
        n, k = 8, 3
        v = ones(n); Σsing = Symmetric(v*v')           # rank-1
        μ = 0.01 .+ 0.01 .* rand(n)

        # With epsilon > 0, should be well-behaved
        res_eps = mve_exhaustive_search(μ, Σsing, k; exactly_k=true, max_samples_per_k=0,
                                epsilon=1e-2, stabilize_Σ=true, compute_weights=true)
        @test isfinite(res_eps.sr)
        @test count(!iszero, res_eps.weights) == k

        sr_re = _sr_exact(res_eps.weights, μ, Σsing; epsilon=1e-2, stabilize_Σ=true)
        @test abs(sr_re - res_eps.sr) ≤ 1e-7
        # epsilon=0 should also not error (pinv path); may yield 0 SR depending on μ vs span(1)
        res0 = mve_exhaustive_search(μ, Σsing, k; exactly_k=true, max_samples_per_k=0,
                                     epsilon=0.0, stabilize_Σ=true, compute_weights=true)
        @test isfinite(res0.sr)
    end

    @testset "ties across supports are handled (identical assets)" begin
        n, k = 5, 2
        μ = fill(0.1, n)
        Σ = Symmetric(Matrix(0.04I, n, n))   # instead of Symmetric(0.04I)
        res = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=0,
                                    epsilon=0.0, stabilize_Σ=true, compute_weights=true)
        @test length(res.selection) == k
        # All pairs have same SR = ||μ_sel|| / sqrt(w'Σw) with optimal weights; just check finiteness and consistency
        @test isfinite(res.sr)
        @test count(!iszero, res.weights) == k
        @test abs(_sr_exact(res.weights, μ, Σ; epsilon=0.0) - res.sr) ≤ 1e-9
    end

    @testset "argument checks (do_checks=true)" begin
        μ = [0.1, 0.2]; Σ = [0.04 0.01; 0.01 0.09]
        @test_throws ErrorException mve_exhaustive_search(μ, ones(2,3), 1; do_checks=true)
        @test_throws ErrorException mve_exhaustive_search(μ, Σ, 0; do_checks=true)
        @test_throws ErrorException mve_exhaustive_search(μ, Σ, 3; do_checks=true)
        @test_throws ErrorException mve_exhaustive_search(μ, Σ, 1; max_samples_per_k=-1, do_checks=true)
        @test_throws ErrorException mve_exhaustive_search(μ, Σ, 1; max_combinations=0, do_checks=true)
        @test_throws ErrorException mve_exhaustive_search([0.1, Inf], Σ, 1; do_checks=true)
        @test_throws ErrorException mve_exhaustive_search(μ, [NaN 0; 0 1], 1; do_checks=true)
    end

    @testset "sampled mode does not exceed cap and improves over time (smoke)" begin
        Random.seed!(22)
        n, k = 10, 3                      # total = 120
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.10I)

        # small cap vs larger cap; larger cap should be ≥ SR
        r_small = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=10,
                                        epsilon=0.0, rng=MersenneTwister(1), stabilize_Σ=true)
        r_big   = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=60,
                                        epsilon=0.0, rng=MersenneTwister(1), stabilize_Σ=true)
        @test r_big.sr + 1e-12 ≥ r_small.sr
    end

    @testset "max_combinations enforces truncation (status) and bounds quality" begin
        Random.seed!(33)
        n, k = 8, 4                       # C(8,4)=70
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.10I)

        res_full = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=0,
                                         max_combinations=10_000,  # >> total, so full enumeration
                                         epsilon=0.0, stabilize_Σ=true)
        @test res_full.status == :EXHAUSTIVE

        # Force truncation/sampling via a tight max_combinations
        res_cap  = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=0,
                                         max_combinations=20,      # << total, triggers sampled path
                                         epsilon=0.0, rng=MersenneTwister(9), stabilize_Σ=true)
        @test res_cap.status == :EXHAUSTIVE_SAMPLED
        # The capped run cannot systematically exceed the full exhaustive SR
        @test res_cap.sr ≤ res_full.sr + 1e-12
    end

    @testset "max_combinations ≥ total still returns exact (status = :EXHAUSTIVE)" begin
        Random.seed!(44)
        n, k = 7, 3                       # total = 35
        μ = 0.01 .+ 0.02 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.06I)

        res1 = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=0,
                                     max_combinations=35, epsilon=0.0, stabilize_Σ=true)
        res2 = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=0,
                                     max_combinations=1_000_000, epsilon=0.0, stabilize_Σ=true)
        @test res1.status == :EXHAUSTIVE
        @test res2.status == :EXHAUSTIVE
        @test res1.selection == res2.selection
        @test isapprox(res1.sr, res2.sr; atol=0, rtol=0)
    end

    # NEW ----------------------------------------------------------------------
    @testset "weights_sum1 semantics (normalization & SR invariance)" begin
        Random.seed!(2025)
        n, k = 8, 4
        μ = 0.01 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.08I)

        # Run twice with and without normalization
        r_unnorm = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=0,
                                         epsilon=0.0, stabilize_Σ=true, compute_weights=true,
                                         weights_sum1=false)
        r_norm   = mve_exhaustive_search(μ, Σ, k; exactly_k=true, max_samples_per_k=0,
                                         epsilon=0.0, stabilize_Σ=true, compute_weights=true,
                                         weights_sum1=true)

        # Same support and same SR (scale invariance)
        @test r_unnorm.selection == r_norm.selection
        @test isapprox(r_unnorm.sr, r_norm.sr; atol=0, rtol=0)

        # Normalized weights: sum to one and proportional to unnormalized
        @test abs(sum(r_norm.weights) - 1.0) ≤ 1e-10
        @test norm(r_norm.weights - (r_unnorm.weights / sum(r_unnorm.weights))) ≤ 1e-8

        # Zeros off-support
        sel = r_norm.selection
        @test all(i -> (i ∈ sel) || r_norm.weights[i] == 0.0, 1:n)
        @test all(i -> (i ∈ sel) || r_unnorm.weights[i] == 0.0, 1:n)

        # SR recompute equals reported SR for both
        @test abs(_sr_exact(r_norm.weights, μ, Σ)   - r_norm.sr)   ≤ 1e-9
        @test abs(_sr_exact(r_unnorm.weights, μ, Σ) - r_unnorm.sr) ≤ 1e-9
    end

    @testset "compute_weights=false returns same support & SR (smoke)" begin
        Random.seed!(303)
        n, k = 7, 3
        μ = 0.01 .+ 0.02 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.04I)

        r_w  = mve_exhaustive_search(μ, Σ, k; exactly_k=true, compute_weights=true,
                                     weights_sum1=false, stabilize_Σ=true)
        r_nw = mve_exhaustive_search(μ, Σ, k; exactly_k=true, compute_weights=false,
                                     stabilize_Σ=true)

        @test r_w.selection == r_nw.selection
        @test isapprox(r_w.sr, r_nw.sr; atol=0, rtol=0)

        # Do not assert on the structure of r_nw.weights (implementation may choose empty or zeros)
        @test count(!iszero, r_w.weights) == k
    end
end
