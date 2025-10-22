# test-ExhaustiveSearch.jl — tests for mve_exhaustive_search & mve_exhaustive_search_gridk
#
# New API under test:
#   mve_exhaustive_search(μ::AbstractVector, Σ::AbstractMatrix; k::Int;
#                         epsilon=..., stabilize_Σ=..., do_checks=...,
#                         enumerate_all::Bool=true, max_samples::Int=0,
#                         dedup_samples::Bool=true, rng=Random.GLOBAL_RNG)
#       -> (selection::Vector{Int}, sr::Float64)
#
#   mve_exhaustive_search_gridk(μ, Σ, k_grid; kwargs...) 
#       -> Dict{Int, Tuple{Vector{Int}, Float64}}
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

        sel, sr = mve_exhaustive_search(μ, Σ; 
            k            = k,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            do_checks    = true,
            enumerate_all = true)

        @test length(sel) == k
        @test sel == brute_set
        @test isapprox(sr, brute_sr; atol=1e-12, rtol=0)
    end

    @testset "sampled mode equals exhaustive when sample cap ≥ total" begin
        Random.seed!(11)
        n, k = 7, 3                       # C(7,3)=35
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.03I)

        sel_exh, sr_exh = mve_exhaustive_search(μ, Σ; 
            k            = k,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            enumerate_all = true)

        sel_cap, sr_cap = mve_exhaustive_search(μ, Σ;
            k            = k,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            enumerate_all = false,
            max_samples  = 35)  # ≥ total

        @test sel_cap == sel_exh
        @test isapprox(sr_cap, sr_exh; atol=0, rtol=0)
    end

    @testset "RNG reproducibility in sampled mode" begin
        Random.seed!(99)
        n, k = 9, 3                       # total = 84
        μ = 0.03 .+ 0.02 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        rng1 = MersenneTwister(123)
        rng2 = MersenneTwister(123)
        s1, r1 = mve_exhaustive_search(μ, Σ;
            k            = k,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            enumerate_all = false,
            max_samples  = 15,
            rng          = rng1)
        s2, r2 = mve_exhaustive_search(μ, Σ;
            k            = k,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            enumerate_all = false,
            max_samples  = 15,
            rng          = rng2)

        @test s1 == s2
        @test isapprox(r1, r2; atol=0, rtol=0)
    end

    @testset "k=1 picks asset with highest μ/√σ² for diagonal Σ" begin
        μ = [0.1, 0.3, 0.2, 0.05]
        diagσ2 = [0.04, 0.25, 0.09, 0.01]
        Σ = Symmetric(Diagonal(diagσ2))
        k = 1

        scores = μ ./ sqrt.(diagσ2)
        best = argmax(scores)

        sel, sr = mve_exhaustive_search(μ, Σ;
            k            = k,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            enumerate_all = true)

        @test sel == [best]
        # sanity: SR equals brute
        _, brute = _bruteforce_best_sr(μ, Σ; s=k, epsilon=0.0, stabilize_Σ=true)
        @test isapprox(sr, brute; atol=1e-12, rtol=0)
    end

    @testset "k=n selects all assets; matches global MVE SR" begin
        Random.seed!(1)
        n = 6
        μ = 0.02 .+ 0.03 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.05I)

        sel, sr = mve_exhaustive_search(μ, Σ;
            k            = n,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            enumerate_all = true)

        @test sel == collect(1:n)

        mve = compute_mve_sr(μ, Symmetric((Σ + Σ')/2); epsilon=0.0, stabilize_Σ=false)
        @test isapprox(sr, mve; atol=1e-12, rtol=0)
    end

    @testset "near-singular Σ and epsilon stabilization" begin
        Random.seed!(5)
        n, k = 8, 3
        v = ones(n); Σsing = Symmetric(v*v')           # rank-1
        μ = 0.01 .+ 0.01 .* rand(n)

        # epsilon > 0 → well-behaved
        sel_eps, sr_eps = mve_exhaustive_search(μ, Σsing;
            k            = k,
            epsilon      = 1e-2,
            stabilize_Σ  = true,
            enumerate_all = true)
        @test isfinite(sr_eps)
        @test length(sel_eps) == k

        # epsilon = 0 → pseudo-inverse path; should not error
        sel0, sr0 = mve_exhaustive_search(μ, Σsing;
            k            = k,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            enumerate_all = true)
        @test isfinite(sr0)
        @test length(sel0) == k
    end

    @testset "ties across supports are handled (identical assets)" begin
        n, k = 5, 2
        μ = fill(0.1, n)
        Σ = Symmetric(Matrix(0.04I, n, n))
        sel, sr = mve_exhaustive_search(μ, Σ;
            k            = k,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            enumerate_all = true)
        @test length(sel) == k
        @test isfinite(sr)
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
        # Valid when enumerate_all=true (max_samples ignored)
        @test mve_exhaustive_search(μ, Σ; k=1, enumerate_all=true, max_samples=-1, do_checks=true) isa Tuple
    end

    @testset "sampled mode improves with more samples (smoke)" begin
        Random.seed!(22)
        n, k = 10, 3                      # total = 120
        μ = 0.02 .+ 0.05 .* rand(n)
        A = randn(n,n); Σ = Symmetric(A*A' + 0.10I)

        s_small, r_small = mve_exhaustive_search(μ, Σ;
            k            = k,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            enumerate_all = false,
            max_samples  = 10,
            rng          = MersenneTwister(1))

        s_big, r_big = mve_exhaustive_search(μ, Σ;
            k            = k,
            epsilon      = 0.0,
            stabilize_Σ  = true,
            enumerate_all = false,
            max_samples  = 60,
            rng          = MersenneTwister(1))

        @test r_big + 1e-12 ≥ r_small
        @test length(s_small) == k && length(s_big) == k
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Tests for mve_exhaustive_search_gridk
# ──────────────────────────────────────────────────────────────────────────────



