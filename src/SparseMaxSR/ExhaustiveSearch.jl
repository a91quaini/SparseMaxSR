module ExhaustiveSearch

using LinearAlgebra
using Random
using Combinatorics     # for combinations(itr, k)
using ..Utils
using ..SharpeRatio

export mve_exhaustive_search

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
                          rng::AbstractRNG = Random.GLOBAL_RNG,
                          # outputs
                          compute_weights::Bool = true
) -> NamedTuple{(:selection, :weights, :sr, :status)}

Search for the best `k`-asset subset maximizing the in-sample MVE Sharpe ratio.

Returns
-------
NamedTuple with fields:
- `selection::Vector{Int}` — indices of the best subset (sorted).
- `weights::Vector{Float64}` — MVE weights on that subset (zeros off-support).
  If `compute_weights=false`, this is `zeros(N)`.
- `sr::Float64` — in-sample MVE Sharpe ratio for `selection`.
- `status::Symbol` — `:EXHAUSTIVE` if full enumeration used, `:SAMPLED` if supports were sampled.

Arguments
---------
- `μ::AbstractVector{<:Real}`: expected excess returns (length `N`).
- `Σ::AbstractMatrix{<:Real}`: covariance matrix (`N×N`).
- `k::Integer`: subset size (`1 ≤ k ≤ N`).
- `epsilon::Real`: ridge used by `Utils._prep_S` for numerical stability.
- `stabilize_Σ::Bool`: if true, symmetrize + ridge-stabilize `Σ` before scoring.
- `do_checks::Bool`: if true, validate inputs (dims, finiteness, ranges).
- `enumerate_all::Bool`: true → enumerate all `N choose k`; false → sample.
- `max_samples::Int`: number of supports when `enumerate_all=false` (must be > 0).
- `dedup_samples::Bool`: ensure sampled supports are distinct (sampling mode).
- `rng::AbstractRNG`: RNG for sampling mode.
- `compute_weights::Bool`: if true, compute and return MVE weights; else return zeros(N).

Notes
-----
- The covariance is prepared **once** as `Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)` and reused.
- Sharpe ratios are computed with `SharpeRatio.compute_mve_sr(μ, Σs; selection=..., stabilize_Σ=false)`.
- Weights (when requested) are computed with the same `Σs` via
  `SharpeRatio.compute_mve_weights(μ, Σs; selection=..., stabilize_Σ=false, do_checks=false)`.
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
    rng::AbstractRNG = Random.GLOBAL_RNG,
    compute_weights::Bool = true
)
    N = length(μ)

    if do_checks
        N > 0 || error("μ must be non-empty.")
        size(Σ) == (N, N) || error("Σ must be N×N (got $(size(Σ))).")
        (1 ≤ k ≤ N) || error("k must be between 1 and N (got $k, N=$N).")
        all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite entries in μ or Σ.")
        isfinite(epsilon) || error("epsilon must be finite.")
        if !enumerate_all
            max_samples > 0 || error("When enumerate_all=false, `max_samples` must be > 0.")
        end
    end

    # One-time stabilization and scorer
    Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)
    scorer = _score_closure(μ, Σs, epsilon)

    # Find best subset
    best_sr::Float64 = -Inf
    best_sel::Vector{Int} = Int[]
    status::Symbol = :EXHAUSTIVE  # declare once

    if enumerate_all
        best_sr, best_sel = _enumerate_best!(N, k, scorer)
        status = :EXHAUSTIVE
    else
        best_sr, best_sel = _sample_best!(rng, N, k, max_samples, scorer; dedup=dedup_samples)
        status = :SAMPLED
    end

    # Compute weights or return zeros
    weights = if compute_weights && !isempty(best_sel)
        SharpeRatio.compute_mve_weights(μ, Σs;
                                        selection=best_sel,
                                        stabilize_Σ=false,
                                        do_checks=false)
    else
        zeros(Float64, N)
    end

    return (selection = best_sel,
            weights   = weights,
            sr        = best_sr,
            status    = status)
end

end # module
