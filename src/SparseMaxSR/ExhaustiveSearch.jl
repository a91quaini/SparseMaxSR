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

# ---------- shared sampler to avoid duplicate definitions ----------
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
        epsilon=EPS_RIDGE,
        rng=Random.default_rng(),
        γ=1.0,
        stabilize_Σ = true,
        compute_weights::Bool=false,
        do_checks=false,
    ) -> (mve_selection, mve_weights, mve_sr, status)

Exhaustive (or sampled) search over subsets to maximize the MVE Sharpe ratio
on the selected indices. By default searches exactly `k` assets; set
`exactly_k=false` to allow any size in `1:k` and return the overall best.

If `max_samples_per_k == 0`, evaluates all combinations of the chosen size(s).
Otherwise, for each size `s`, evaluates up to `max_samples_per_k` random supports
of size `s` (uniformly sampled). If `max_samples_per_k ≥ binomial(n, s)`, the
algorithm switches to exhaustive enumeration for that `s`.

Numerical ridge is controlled by `epsilon` and applied inside the SharpeRatio
routines as `Σ_eff = Σ + epsilon * mean(diag(Σ)) * I`.
"""
function mve_exhaustive_search(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    k::Integer;
    exactly_k::Bool = true,
    max_samples_per_k::Int = 0,
    epsilon::Real = EPS_RIDGE,
    rng::AbstractRNG = Random.default_rng(),
    γ::Real = 1.0,
    stabilize_Σ::Bool = true,
    compute_weights::Bool=false,
    do_checks::Bool = false,
)
    # --- input checks ---
    n = length(μ)
    if do_checks
        size(Σ,1) == n && size(Σ,2) == n || error("Σ must be n×n with n = length(μ).")
        1 ≤ k ≤ n                          || error("k must satisfy 1 ≤ k ≤ n.")
        max_samples_per_k ≥ 0              || error("max_samples_per_k must be ≥ 0.")
        isfinite(epsilon)                  || error("epsilon must be finite.")
        γ > 0                              || error("γ must be positive.")
        all(isfinite, μ) && all(isfinite, Σ) || error("μ and Σ must be finite.")
    end

    # --- Stabilize Σ once (or just symmetrize if stabilize_Σ=false) ---
    # Use the same Σs for both the MIQP objective and the SR computation.
    Σs = stabilize_Σ ? _prep_S(Σ, epsilon, true) : Symmetric((Σ + Σ')/2)


    nrange = 1:n
    best_sr  = -Inf
    best_set = Vector{Int}()

    if exactly_k
        s = k
        total = binomial(n, s)

        if max_samples_per_k == 0 || max_samples_per_k ≥ total
            # exhaustive for size s (idxs are already sorted by combinations)
            for idxs in combinations(nrange, s)
                sr = compute_mve_sr(μ, Σs; selection=idxs, epsilon=epsilon, stabilize_Σ=false, do_checks=false)
                if sr > best_sr + SR_TOL
                    best_sr  = sr
                    best_set = copy(idxs)
                end
            end
        else
            # sampled mode with cap to avoid infinite loop
            target = min(max_samples_per_k, total)
            scorer = sel -> compute_mve_sr(μ, Σs;
                                        selection=sel,
                                        epsilon=epsilon,
                                        stabilize_Σ=false,
                                        do_checks=false)
            best_sr, best_set = sample_and_score_best!(rng, n, s, target, scorer)
        end

    else
        # sizes = 1:k
        for s in 1:k
            total = binomial(n, s)

            if max_samples_per_k == 0 || max_samples_per_k ≥ total
                # exhaustive per size
                for idxs in combinations(nrange, s)
                    sr = compute_mve_sr(μ, Σs; selection=idxs, epsilon=epsilon, stabilize_Σ=false, do_checks=false)
                    if sr > best_sr + SR_TOL
                        best_sr  = sr
                        best_set = copy(idxs)
                    end
                end
            else
                # sampled per size with cap
                target = min(max_samples_per_k, total)
                scorer = sel -> compute_mve_sr(μ, Σs;
                                            selection=sel,
                                            epsilon=epsilon,
                                            stabilize_Σ=false,
                                            do_checks=false)
                sr_s, set_s = sample_and_score_best!(rng, n, s, target, scorer)
                if sr_s > best_sr + SR_TOL
                    best_sr  = sr_s
                    best_set = set_s
                end
            end
        end
    end

    # Compute weights for the best selection (full-length, zeros elsewhere)
    best_w = (isempty(best_set) || !compute_weights) ? 
        zeros(Float64, n) :
        compute_mve_weights(μ, Σs; selection=best_set, epsilon=epsilon, stabilize_Σ=false, do_checks=false)

    return (mve_selection = best_set,
            mve_weights   = best_w,
            mve_sr        = best_sr,
            status        = :EXHAUSTIVE)
end

end # module
