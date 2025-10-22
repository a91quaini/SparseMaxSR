module ExhaustiveSearch

using LinearAlgebra
using Random
using Combinatorics     # for combinations(itr, k)
using ..Utils
using ..SharpeRatio

export mve_exhaustive_search,
       mve_exhaustive_search_gridk

# ──────────────────────────────────────────────────────────────────────────────
# Overview
# ──────────────────────────────────────────────────────────────────────────────
# This module provides exhaustive and sampled search routines over k-subsets of
# assets to maximize the in-sample MVE Sharpe ratio. It is designed to be:
#   • Numerically safe: all covariance handling goes through Utils._prep_S.
#   • Allocation-light: no per-combination vector allocations on hot paths.
#   • Flexible: exact enumeration when feasible, or uniform sampling of supports.
#
# Public API
# ----------
# exhaustive_best_mve_sr(μ, Σ; k, [kwargs...]) -> (selection::Vector{Int}, sr::Float64)
#   Return the best subset of size k (via full enumeration or sampling, depending
#   on kwargs) and its in-sample MVE Sharpe ratio.
#
# exhaustive_grid_mve_sr(μ, Σ; k_grid::AbstractVector{<:Integer}, [kwargs...])
#   Return a Dict(k => (selection, sr)) for each k in k_grid, sharing precomputed
#   stabilized Σ across the grid.
#
# Numerical details
# -----------------
# Given mean vector μ and covariance matrix Σ, for a subset S, the MVE SR is
#     SR(S) = sqrt( μ_S' Σ_S^{-1} μ_S )
# We compute it via SharpeRatio.compute_mve_sr(μ, Σs; selection=S, stabilize_Σ=false)
# where Σs = Utils._prep_S(Σ, ε, stabilize_Σ) is formed once per call/group.
#
# Optimization details (robust/safe)
# ----------------------------------
# 1) Enumeration avoids per-combination allocations: a single Int buffer is
#    reused; only the current-best selection is copied.
# 2) Sampling uses a direct without-replacement subset sampler; no randperm(n).
# 3) Optional dedup of sampled supports ensures exactly m distinct supports.
# 4) Scorers capture μ, Σs, ε in a closure; no re-prep of Σ in the loop.
#
# Notes on reproducibility
# ------------------------
# You may pass a dedicated RNG (e.g. MersenneTwister) for deterministic sampling.
# For batch runs, derive per-k RNGs from a root RNG to stabilize randomness across
# grid sweeps.
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (allocation-light)
# ──────────────────────────────────────────────────────────────────────────────

const SR_TOL = 0.0  # tie-breaking tolerance; keep zero to be deterministic given RNG

"""
    _to_vec!(buf::Vector{Int}, comb) -> Vector{Int}

Copy the k-tuple `comb` (from `combinations`) into the reusable buffer `buf`
and return `buf`. Avoids allocating a fresh Vector for every combination.

Assumes `length(buf) == length(comb)`.
"""
@inline function _to_vec!(buf::Vector{Int}, comb)::Vector{Int}
    @inbounds for i in 1:length(buf)
        buf[i] = comb[i]
    end
    return buf
end

"""
    _rand_k_subset!(rng, out, n, k) -> out

Fill `out` (length k) with a uniformly-sampled k-subset from 1:n (without
replacement) and return `out`. The result is sorted in ascending order.

This avoids allocating `randperm(n)`; complexity is ~O(k log k) due to final sort.
"""
function _rand_k_subset!(rng::AbstractRNG, out::Vector{Int}, n::Int, k::Int)
    @assert length(out) == k
    # Floyd's algorithm with a small Set to avoid duplicates among the k draws
    seen = Set{Int}()
    i = 1
    while i ≤ k
        x = rand(rng, 1:n)
        if !(x in seen)
            push!(seen, x)
            @inbounds out[i] = x
            i += 1
        end
    end
    sort!(out)
    return out
end

"""
    _score_closure(μ, Σs, ε) -> (sel::AbstractVector{<:Integer}) -> Float64

Create an allocation-light scorer closure that computes SR(S) for a subset S
using the already-prepared symmetric covariance Σs.

Note: Σs must be the stabilized/symmetrized covariance. Inside the scorer we
set `stabilize_Σ=false` to avoid re-prepping.
"""
@inline function _score_closure(μ::AbstractVector{<:Real},
                                Σs::Symmetric,
                                ε::Real)
    return (sel::AbstractVector{<:Integer}) ->
        SharpeRatio.compute_mve_sr(μ, Σs;
            selection    = sel,
            epsilon      = ε,
            stabilize_Σ  = false,
            do_checks    = false)
end


# ──────────────────────────────────────────────────────────────────────────────
# Core search routines (enumeration & sampling)
# ──────────────────────────────────────────────────────────────────────────────

"""
    _enumerate_best!(n::Int, k::Int, scorer) -> (best_sr::Float64, best_sel::Vector{Int})

Enumerate all `n choose k` subsets, return the best Sharpe and its selection.
Uses a reusable buffer to avoid per-combination allocations.
"""
function _enumerate_best!(n::Int, k::Int, scorer::Function)
    best_sr  = -Inf
    best_sel = Vector{Int}(undef, 0)
    buf      = Vector{Int}(undef, k)
    for comb in combinations(1:n, k)
        sel = _to_vec!(buf, comb)
        sr  = scorer(sel)
        if sr > best_sr + SR_TOL
            best_sr  = sr
            best_sel = copy(sel)  # copy *only* when we improve
        end
    end
    return best_sr, best_sel
end

"""
    _sample_best!(rng, n::Int, k::Int, m::Int, scorer;
                  dedup::Bool=true) -> (best_sr, best_sel)

Sample `m` supports of size `k` uniformly at random (without replacement within
each support) and return the best Sharpe and its selection. If `dedup=true`,
re-sample until `m` *distinct* supports have been evaluated.

- For `n ≤ 64`, uniqueness is tracked using a UInt64 bitmask.
- For `n > 64`, uniqueness is tracked via NTuple keys.

This avoids O(n) `randperm` per sample.
"""
function _sample_best!(rng::AbstractRNG, n::Int, k::Int, m::Int, scorer::Function; dedup::Bool=true)
    @assert 1 ≤ k ≤ n
    buf      = Vector{Int}(undef, k)
    best_sr  = -Inf
    best_sel = Vector{Int}(undef, 0)

    if !dedup
        @inbounds for _ in 1:m
            _rand_k_subset!(rng, buf, n, k)
            sr = scorer(buf)
            if sr > best_sr + SR_TOL
                best_sr  = sr
                best_sel = copy(buf)
            end
        end
        return best_sr, best_sel
    end

    if n ≤ 64
        seen = Set{UInt64}()
        while length(seen) < m
            _rand_k_subset!(rng, buf, n, k)
            key = zero(UInt64)
            @inbounds for i in buf
                key |= (UInt64(1) << (i - 1))
            end
            if !(key in seen)
                push!(seen, key)
                sr = scorer(buf)
                if sr > best_sr + SR_TOL
                    best_sr  = sr
                    best_sel = copy(buf)
                end
            end
        end
    else
        seen = Set{NTuple{Int,Int}}()
        while length(seen) < m
            _rand_k_subset!(rng, buf, n, k)
            key = ntuple(i -> @inbounds buf[i], k)
            if !(key in seen)
                push!(seen, key)
                sr = scorer(buf)
                if sr > best_sr + SR_TOL
                    best_sr  = sr
                    best_sel = copy(buf)
                end
            end
        end
    end

    return best_sr, best_sel
end


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    mve_exhaustive_search(μ::AbstractVector{<:Real},
                          Σ::AbstractMatrix{<:Real};
                          k::Integer,
                          epsilon::Real = Utils.EPS_RIDGE,
                          stabilize_Σ::Bool = true,
                          do_checks::Bool = false,
                          # enumeration / sampling knobs
                          enumerate_all::Bool = true,
                          max_samples::Int = 0,
                          dedup_samples::Bool = true,
                          rng::AbstractRNG = Random.GLOBAL_RNG
                          ) -> (selection::Vector{Int}, sr::Float64)

Return the best k-asset subset (and its in-sample MVE Sharpe ratio) under either
full enumeration (`enumerate_all=true`) or uniform random sampling over supports
(`enumerate_all=false`, with `max_samples` draws).

Arguments
---------
- `μ`, `Σ`          : asset moments (length N, N×N).
- `k`               : subset size (1 ≤ k ≤ N).
- `epsilon`         : ridge size used in `Utils._prep_S`.
- `stabilize_Σ`     : if true, symmetrize and ridge-stabilize Σ before scoring.
- `do_checks`       : if true, validate inputs (dims, finiteness, ranges).
- `enumerate_all`   : if true, enumerate all N choose k subsets; else sample.
- `max_samples`     : number of sampled supports when `enumerate_all=false`.
- `dedup_samples`   : ensure sampled supports are distinct.
- `rng`             : RNG used for sampling.

Returns
-------
- `selection::Vector{Int}` : the best subset (sorted indices).
- `sr::Float64`            : the corresponding in-sample MVE Sharpe ratio.

Notes
-----
- If `enumerate_all=false` and `max_samples ≤ 0`, the function behaves like
  enumeration when N choose k is reasonably small; otherwise it throws.
- All scoring uses a *single* stabilized covariance `Σs` to avoid rework inside
  loops: `SharpeRatio.compute_mve_sr(μ, Σs; selection, stabilize_Σ=false)`.
"""
function mve_exhaustive_search(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    k::Integer,
    epsilon::Real = Utils.EPS_RIDGE,
    stabilize_Σ::Bool = true,
    do_checks::Bool = false,
    enumerate_all::Bool = true,
    max_samples::Int = 0,
    dedup_samples::Bool = true,
    rng::AbstractRNG = Random.GLOBAL_RNG
)::Tuple{Vector{Int}, Float64}
    N = length(μ)

    if do_checks
        N > 0 || error("μ must be non-empty.")
        size(Σ) == (N, N) || error("Σ must be N×N (got $(size(Σ))).")
        (1 ≤ k ≤ N) || error("k must be between 1 and N.")
        all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite entries in μ or Σ.")
        isfinite(epsilon) || error("epsilon must be finite.")
        if !enumerate_all
            max_samples > 0 || error("When enumerate_all=false, `max_samples` must be > 0.")
        end
    end

    # Prepare symmetric (and optionally stabilized) covariance once
    Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)

    # Scorer closure (no allocations inside)
    scorer = _score_closure(μ, Σs, epsilon)

    if enumerate_all
        best_sr, best_sel = _enumerate_best!(N, k, scorer)
        return best_sel, best_sr
    else
        best_sr, best_sel = _sample_best!(rng, N, k, max_samples, scorer; dedup=dedup_samples)
        return best_sel, best_sr
    end
end

"""
    mve_exhaustive_search_gridk(μ, Σ;
                                k_grid::AbstractVector{<:Integer},
                                epsilon::Real = Utils.EPS_RIDGE,
                                stabilize_Σ::Bool = true,
                                do_checks::Bool = false,
                                enumerate_all::Bool = true,
                                max_samples_per_k::Int = 0,
                                dedup_samples::Bool = true,
                                rng::AbstractRNG = Random.GLOBAL_RNG
                                ) -> Dict{Int,Tuple{Vector{Int},Float64}}

Evaluate the best MVE Sharpe ratio over a grid of k's, sharing the stabilized
covariance across all k to reduce overhead.

Arguments
---------
- `k_grid`          : vector of subset sizes (each 1 ≤ k ≤ N).
- Other arguments   : as in `exhaustive_best_mve_sr`, applied per k.
- `max_samples_per_k` : number of samples to draw at each k if sampling.

Returns
-------
- `Dict(k => (selection::Vector{Int}, sr::Float64))`.

Details
-------
- For reproducibility across k’s, a child RNG is derived from `rng` using a
  random UInt seed per k. This stabilizes results across repeated runs with the
  same parent RNG.
"""
function mve_exhaustive_search_gridk(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    k_grid::AbstractVector{<:Integer},
    epsilon::Real = Utils.EPS_RIDGE,
    stabilize_Σ::Bool = true,
    do_checks::Bool = false,
    enumerate_all::Bool = true,
    max_samples_per_k::Int = 0,
    dedup_samples::Bool = true,
    rng::AbstractRNG = Random.GLOBAL_RNG
)::Dict{Int,Tuple{Vector{Int},Float64}}
    N = length(μ)
    if do_checks
        N > 0 || error("μ must be non-empty.")
        size(Σ) == (N, N) || error("Σ must be N×N.")
        all(1 .≤ k_grid .≤ N) || error("All k in k_grid must satisfy 1 ≤ k ≤ $N.")
        isfinite(epsilon) || error("epsilon must be finite.")
        if !enumerate_all
            max_samples_per_k > 0 || error("When enumerate_all=false, `max_samples_per_k` must be > 0.")
        end
    end

    # Precompute stabilized covariance once for the entire grid
    Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)
    scorer = _score_closure(μ, Σs, epsilon)

    out = Dict{Int,Tuple{Vector{Int},Float64}}()
    for k in k_grid
        if enumerate_all
            best_sr, best_sel = _enumerate_best!(N, k, scorer)
            out[k] = (best_sel, best_sr)
        else
            rng_k = MersenneTwister(rand(rng, UInt))  # derived RNG for reproducibility
            best_sr, best_sel = _sample_best!(rng_k, N, k, max_samples_per_k, scorer; dedup=dedup_samples)
            out[k] = (best_sel, best_sr)
        end
    end
    return out
end

end # module
