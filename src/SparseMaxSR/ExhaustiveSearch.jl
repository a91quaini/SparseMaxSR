module ExhaustiveSearch

# --- deps ---
using Random
using LinearAlgebra
using Statistics
using Combinatorics: combinations

# Use SharpeRatio utilities + Utils
import ..SharpeRatio: compute_mve_sr, compute_mve_weights
import ..Utils: EPS_RIDGE, _prep_S

export mve_exhaustive_search

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

# Comparisons tolerance for Sharpe improvements.
const SR_TOL = 1e-12

# Decide whether to enumerate or sample for a given (n, s), given user caps.
@inline function _plan_mode(n::Int, s::Int, max_samples_per_k::Int, max_combinations::Int)
    total = binomial(n, s)
    # If user asked for sampling explicitly, sample up to their cap (or total)
    if max_samples_per_k > 0
        m_target = min(max_samples_per_k, total)
        return (:sample, m_target, total)
    end
    # Otherwise we prefer enumeration, but cap by max_combinations
    if total <= max_combinations
        return (:enumerate_full, total, total)
    else
        # either truncated enumeration or sampling; use sampling w/o replacement
        # up to the cap, which is typically more diverse than "first K combos"
        return (:sample, max_combinations, total)
    end
end

# Return (best_sr, best_sel) after evaluating up to m distinct k-subsets.
# Scoring is delegated to `scorer(sel::AbstractVector{<:Integer}) -> Real`.
function sample_and_score_best!(
    rng::AbstractRNG, n::Int, k::Int, m::Int, scorer::Function
)
    @assert 1 ≤ k ≤ n
    total = binomial(n, k)

    # 1) Enumerate all when m ≥ total (no memory blow-up; stream combos)
    if m ≥ total
        best_sr  = -Inf
        best_sel = Vector{Int}()
        for comb in combinations(1:n, k)
            sel = collect(comb)                    # size-k vector
            sr  = scorer(sel)
            if sr > best_sr + SR_TOL
                best_sr  = sr
                best_sel = sel
            end
        end
        return best_sr, best_sel
    end

    # 2) Sample-without-replacement across supports, streaming scoring
    if n ≤ 64
        # compact bitmask dedup
        seen = Set{UInt64}()
        buf  = Vector{Int}(undef, k)
        best_sr  = -Inf
        best_sel = Vector{Int}()

        while length(seen) < m
            p = randperm(rng, n)
            @inbounds copyto!(buf, 1, p, 1, k)
            sort!(buf)

            key = zero(UInt64)
            @inbounds for i in buf
                key |= (UInt64(1) << (i - 1))
            end
            if !(key in seen)
                push!(seen, key)
                sr = scorer(buf)                   # buf is overwritten later; copy if best
                if sr > best_sr + SR_TOL
                    best_sr  = sr
                    best_sel = copy(buf)
                end
            end
        end
        return best_sr, best_sel
    else
        # tuple fallback for large n
        seen = Set{NTuple{N,Int}}() where {N}
        buf  = Vector{Int}(undef, k)
        best_sr  = -Inf
        best_sel = Vector{Int}()

        while length(seen) < m
            p = randperm(rng, n)
            @inbounds copyto!(buf, 1, p, 1, k)
            sort!(buf)
            key = tuple(buf...)
            if !(key in seen)
                push!(seen, key)
                sr = scorer(buf)
                if sr > best_sr + SR_TOL
                    best_sr  = sr
                    best_sel = copy(buf)
                end
            end
        end
        return best_sr, best_sel
    end
end

# =============================================================================
# Public API
# =============================================================================

"""
    mve_exhaustive_search(μ, Σ, k;
        exactly_k=true,
        max_samples_per_k=0,
        max_combinations::Int=10_000_000,
        epsilon=EPS_RIDGE,
        rng=Random.default_rng(),
        γ=1.0,
        stabilize_Σ = true,
        compute_weights::Bool=false,
        weights_sum1::Bool=false,
        do_checks=false,
    ) -> NamedTuple{(:selection, :weights, :sr, :status)}

Exhaustive (or sampled) search over subsets to maximize the MVE Sharpe ratio
on the selected indices. By default searches exactly `k` assets; set
`exactly_k=false` to allow any size in `1:k` and return the overall best.

If `max_samples_per_k == 0`, the method **tries** to enumerate but will cap the
total number of visited combinations per size by `max_combinations`. When the
cap is hit, it switches to sampling without replacement up to the cap.

If `max_samples_per_k > 0`, the method samples up to `max_samples_per_k` supports
for each size (also bounded above by the number of available combinations).

If `compute_weights=true`, the returned weights are computed via
`compute_mve_weights(...; selection=best_set, weights_sum1=weights_sum1)`.
Note that the reported Sharpe ratio is scale-invariant and does not depend on
`weights_sum1`.

The returned `status` is `:EXHAUSTIVE` if all intended supports were fully
enumerated, and `:EXHAUSTIVE_SAMPLED` if any size was truncated or sampled.

Numerical ridge is controlled by `epsilon` and applied inside the SharpeRatio
routines as `Σ_eff = Σ + epsilon * mean(diag(Σ)) * I`.
"""
function mve_exhaustive_search(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    k::Integer;
    exactly_k::Bool = true,
    max_samples_per_k::Int = 0,
    max_combinations::Int = 10_000_000,
    epsilon::Real = EPS_RIDGE,
    rng::AbstractRNG = Random.default_rng(),
    γ::Real = 1.0,
    stabilize_Σ::Bool = true,
    compute_weights::Bool=false,
    weights_sum1::Bool=false,
    do_checks::Bool = false,
)
    # --- input checks ---
    n = length(μ)
    if do_checks
        size(Σ,1) == n && size(Σ,2) == n || error("Σ must be n×n with n = length(μ).")
        1 ≤ k ≤ n                          || error("k must satisfy 1 ≤ k ≤ n.")
        max_samples_per_k ≥ 0              || error("max_samples_per_k must be ≥ 0.")
        max_combinations > 0               || error("max_combinations must be > 0.")
        isfinite(epsilon)                  || error("epsilon must be finite.")
        γ > 0                              || error("γ must be positive.")
        all(isfinite, μ) && all(isfinite, Σ) || error("μ and Σ must be finite.")
    end

    # --- Stabilize Σ once (or just symmetrize if stabilize_Σ=false) ---
    Σs = stabilize_Σ ? _prep_S(Σ, epsilon, true) : Symmetric((Σ + Σ')/2)

    nrange = 1:n
    best_sr  = -Inf
    best_set = Vector{Int}()
    fully_enumerated = true

    # scorer closure
    scorer = sel -> compute_mve_sr(μ, Σs;
                                   selection=sel,
                                   epsilon=epsilon,
                                   stabilize_Σ=false,
                                   do_checks=false)

    if exactly_k
        s = k
        mode, m_target, total = _plan_mode(n, s, max_samples_per_k, max_combinations)

        if mode == :enumerate_full
            # Full enumeration
            for idxs in combinations(nrange, s)
                sr = scorer(idxs)
                if sr > best_sr + SR_TOL
                    best_sr  = sr
                    best_set = collect(idxs)
                end
            end
        else
            # Sample up to m_target (or truncate enumeration to cap)
            fully_enumerated = false
            if max_samples_per_k == 0 && m_target < total
                # Truncated enumeration of the *first* m_target combinations
                cnt = 0
                for idxs in combinations(nrange, s)
                    sr = scorer(idxs)
                    if sr > best_sr + SR_TOL
                        best_sr  = sr
                        best_set = collect(idxs)
                    end
                    cnt += 1
                    cnt >= m_target && break
                end
            else
                # Random sampling without replacement up to m_target
                best_sr, best_set = sample_and_score_best!(rng, n, s, m_target, scorer)
            end
        end
    else
        # sizes = 1:k
        for s in 1:k
            mode, m_target, total = _plan_mode(n, s, max_samples_per_k, max_combinations)

            if mode == :enumerate_full
                for idxs in combinations(nrange, s)
                    sr = scorer(idxs)
                    if sr > best_sr + SR_TOL
                        best_sr  = sr
                        best_set = collect(idxs)
                    end
                end
            else
                fully_enumerated = false
                if max_samples_per_k == 0 && m_target < total
                    # Truncated enumeration (first m_target combos)
                    cnt = 0
                    for idxs in combinations(nrange, s)
                        sr = scorer(idxs)
                        if sr > best_sr + SR_TOL
                            best_sr  = sr
                            best_set = collect(idxs)
                        end
                        cnt += 1
                        cnt >= m_target && break
                    end
                else
                    sr_s, set_s = sample_and_score_best!(rng, n, s, m_target, scorer)
                    if sr_s > best_sr + SR_TOL
                        best_sr  = sr_s
                        best_set = set_s
                    end
                end
            end
        end
    end

    # Compute weights for the best selection (full-length, zeros elsewhere)
    best_w = (isempty(best_set) || !compute_weights) ?
        zeros(Float64, n) :
        compute_mve_weights(μ, Σs;
            selection=best_set,
            weights_sum1=weights_sum1,
            epsilon=epsilon,
            stabilize_Σ=false,
            do_checks=false)

    return (selection = best_set,
            weights   = best_w,
            sr        = best_sr,
            status    = fully_enumerated ? :EXHAUSTIVE : :EXHAUSTIVE_SAMPLED)

end

end # module
