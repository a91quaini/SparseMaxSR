# test-ExhaustiveSearch.jl — tests for mve_exhaustive_search (NamedTuple API)
#
# New API under test:
#   mve_exhaustive_search(μ::AbstractVector, Σ::AbstractMatrix; k::Int;
#                         epsilon=..., stabilize_Σ=..., do_checks=...,
#                         enumerate_all::Bool=true, max_samples::Int=0,
#                         dedup_samples::Bool=true, rng=Random.GLOBAL_RNG,
#                         compute_weights::Bool=true)
#       -> NamedTuple{(:selection, :weights, :sr, :status)}
#
using Test, Random, LinearAlgebra, Statistics
using SparseMaxSR
using SparseMaxSR.SharpeRatio
using Combinatorics: combinations

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Brute-force best SR over exact support size s (for tiny n,k)
function _bruteforce_best_sr(μ, Σ; s::Int, epsilon=0.0, stabilize_Σ=true)
    n = length(μ)
    Σs = Symmetric((Σ + Σ') / 2)  # let compute_mve_sr handle stabilization via epsilon/stabilize_Σ
    best_sr  = -Inf
    best_set = Int[]
    for idxs in combinations(1:n, s)
        sr = compute_mve_sr(μ, Σs;
            selection   = idxs,
            epsilon     = epsilon,
            stabilize_Σ = stabilize_Σ,
            do_checks   = false)
        if sr > best_sr
            best_sr  = sr
            best_set = collect(idxs)
        end
    end
    return best_set, best_sr
end

# Sum of absolute weights off-support (should be ~0 for exact-support solvers)
function _offsupport_l1(w::AbstractVector, sel::AbstractVector, n::Int)
    offs = setdiff(1:n, sel)
    return sum(abs, @view w[offs])
end

# ──────────────────────────────────────────────────────────────────────────────
# Tests for mve_exhaustive_search
# ──────────────────────────────────────────────────────────────────────────────

@testset "ExhaustiveSearch.mve_exhaustive_search" begin

    @testset "basic exact search equals brute force (exactly k)" begin
        Random.seed!(42)
        n, k = 6, 3                       # C(6,3)=20 — tiny
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.10I)

        brute_set, brute_sr = _bruteforce_best_sr(μ, Σ; s=k, epsilon=0.0, stabilize_Σ=true)

        res = mve_exhaustive_search(μ, Σ; 
            k             = k,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            do_checks     = true,
            enumerate_all = true)

        @test res.status == :EXHAUSTIVE
        @test length(res.selection) == k
        @test res.selection == brute_set
        @test isapprox(res.sr, brute_sr; atol=1e-12, rtol=0)
        @test length(res.weights) == n
        @test _offsupport_l1(res.weights, res.selection, n) ≤ 1e-12
    end

    @testset "sampled mode equals exhaustive when sample cap ≥ total" begin
        Random.seed!(11)
        n, k = 7, 3                       # C(7,3)=35
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.03I)

        res_exh = mve_exhaustive_search(μ, Σ; 
            k             = k,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            enumerate_all = true)

        total = length(collect(combinations(1:n, k)))
        res_cap = mve_exhaustive_search(μ, Σ;
            k             = k,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            enumerate_all = false,
            max_samples   = total)

        @test res_cap.status == :SAMPLED
        @test res_cap.selection == res_exh.selection
        @test isapprox(res_cap.sr, res_exh.sr; atol=0, rtol=0)
        @test length(res_cap.weights) == n
        @test _offsupport_l1(res_cap.weights, res_cap.selection, n) ≤ 1e-12
    end

    @testset "RNG reproducibility in sampled mode" begin
        Random.seed!(99)
        n, k = 9, 3                       # total = 84
        μ = 0.03 .+ 0.02 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        rng1 = MersenneTwister(123)
        rng2 = MersenneTwister(123)
        r1 = mve_exhaustive_search(μ, Σ;
            k             = k,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            enumerate_all = false,
            max_samples   = 15,
            rng           = rng1)
        r2 = mve_exhaustive_search(μ, Σ;
            k             = k,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            enumerate_all = false,
            max_samples   = 15,
            rng           = rng2)

        @test r1.status == :SAMPLED == r2.status
        @test r1.selection == r2.selection
        @test isapprox(r1.sr, r2.sr; atol=0, rtol=0)
        @test length(r1.weights) == n == length(r2.weights)
        @test _offsupport_l1(r1.weights, r1.selection, n) ≤ 1e-12
        @test _offsupport_l1(r2.weights, r2.selection, n) ≤ 1e-12
    end

    @testset "k=1 picks asset with highest μ/√σ² for diagonal Σ" begin
        μ = [0.1, 0.3, 0.2, 0.05]
        diagσ2 = [0.04, 0.25, 0.09, 0.01]
        Σ = Symmetric(Diagonal(diagσ2))
        k = 1

        scores = μ ./ sqrt.(diagσ2)
        best = argmax(scores)

        res = mve_exhaustive_search(μ, Σ;
            k             = k,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            enumerate_all = true)

        @test res.status == :EXHAUSTIVE
        @test res.selection == [best]
        # sanity: SR equals brute
        _, brute = _bruteforce_best_sr(μ, Σ; s=k, epsilon=0.0, stabilize_Σ=true)
        @test isapprox(res.sr, brute; atol=1e-12, rtol=0)
        @test _offsupport_l1(res.weights, res.selection, length(μ)) ≤ 1e-12
    end

    @testset "k=n selects all assets; matches global MVE SR" begin
        Random.seed!(1)
        n = 6
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        res = mve_exhaustive_search(μ, Σ;
            k             = n,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            enumerate_all = true)

        @test res.status == :EXHAUSTIVE
        @test res.selection == collect(1:n)

        mve = compute_mve_sr(μ, Symmetric((Σ + Σ')/2); epsilon=0.0, stabilize_Σ=false)
        @test isapprox(res.sr, mve; atol=1e-12, rtol=0)
        @test _offsupport_l1(res.weights, res.selection, n) ≤ 1e-12
    end

    @testset "near-singular Σ and epsilon stabilization" begin
        Random.seed!(5)
        n, k = 8, 3
        v = ones(n); Σsing = Symmetric(v*v')           # rank-1
        μ = 0.01 .+ 0.01 .* rand(n)

        # epsilon > 0 → well-behaved
        res_eps = mve_exhaustive_search(μ, Σsing;
            k             = k,
            epsilon       = 1e-2,
            stabilize_Σ   = true,
            enumerate_all = true)
        @test res_eps.status == :EXHAUSTIVE
        @test isfinite(res_eps.sr)
        @test length(res_eps.selection) == k
        @test _offsupport_l1(res_eps.weights, res_eps.selection, n) ≤ 1e-10

        # epsilon = 0 → pseudo-inverse path; should not error
        res0 = mve_exhaustive_search(μ, Σsing;
            k             = k,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            enumerate_all = true)
        @test res0.status == :EXHAUSTIVE
        @test isfinite(res0.sr)
        @test length(res0.selection) == k
        @test _offsupport_l1(res0.weights, res0.selection, n) ≤ 1e-10
    end

    @testset "ties across supports are handled (identical assets)" begin
        n, k = 5, 2
        μ = fill(0.1, n)
        Σ = Symmetric(Matrix(0.04I, n, n))
        res = mve_exhaustive_search(μ, Σ;
            k             = k,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            enumerate_all = true)
        @test res.status == :EXHAUSTIVE
        @test length(res.selection) == k
        @test isfinite(res.sr)
        @test _offsupport_l1(res.weights, res.selection, n) ≤ 1e-12
    end

    @testset "argument checks (do_checks=true)" begin
        μ = [0.1, 0.2]; Σ = [0.04 0.01; 0.01 0.09]
        @test_throws ErrorException mve_exhaustive_search(μ, ones(2,3); k=1, do_checks=true)
        @test_throws ErrorException mve_exhaustive_search(μ, Σ; k=0, do_checks=true)
        @test_throws ErrorException mve_exhaustive_search(μ, Σ; k=3, do_checks=true)
        @test_throws ErrorException mve_exhaustive_search([0.1, Inf], Σ; k=1, do_checks=true)
        @test_throws ErrorException mve_exhaustive_search(μ, [NaN 0; 0 1]; k=1, do_checks=true)
        # Sampling knob check (only meaningful for enumerate_all=false)
        @test_throws ErrorException mve_exhaustive_search(μ, Σ; k=1, enumerate_all=false, max_samples=-1, do_checks=true)
        # Valid when enumerate_all=true (max_samples ignored); returns NamedTuple
        @test mve_exhaustive_search(μ, Σ; k=1, enumerate_all=true, max_samples=-1, do_checks=true) isa NamedTuple
    end

    @testset "compute_weights=false returns zero vector (length N)" begin
        Random.seed!(77)
        n, k = 7, 3
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.02I)

        res = mve_exhaustive_search(μ, Σ;
            k               = k,
            epsilon         = 0.0,
            stabilize_Σ     = true,
            enumerate_all   = true,
            compute_weights = false)

        @test res.status == :EXHAUSTIVE
        @test length(res.weights) == n
        @test all(iszero, res.weights)
        @test isfinite(res.sr)
        @test length(res.selection) == k
    end

    @testset "sampled mode improves with more samples (smoke)" begin
        Random.seed!(22)
        n, k = 10, 3                      # total = 120
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.10I)

        r_small = mve_exhaustive_search(μ, Σ;
            k             = k,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            enumerate_all = false,
            max_samples   = 10,
            rng           = MersenneTwister(1))

        r_big = mve_exhaustive_search(μ, Σ;
            k             = k,
            epsilon       = 0.0,
            stabilize_Σ   = true,
            enumerate_all = false,
            max_samples   = 60,
            rng           = MersenneTwister(1))

        @test r_small.status == :SAMPLED == r_big.status
        @test r_big.sr + 1e-12 ≥ r_small.sr
        @test length(r_small.selection) == k && length(r_big.selection) == k
        @test length(r_small.weights) == n && length(r_big.weights) == n
    end
end

