module SharpeRatio
# Utilities for Sharpe ratio, MVE Sharpe ratio, and MVE weights.

using LinearAlgebra
using Statistics
using ..SparseMaxSR: EPS_RIDGE

export compute_sr,
       compute_mve_sr,
       compute_mve_weights

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

# Symmetrize and (optionally) apply a small ridge:
#    Σ_eff = Sym( (Σ + Σ')/2 + ε·mean(diag(Σ))·I )
@inline function _sym_with_ridge(Σ::AbstractMatrix{T}, epsilon::Real=0.0) where {T<:Real}
    Σsym = (Σ + Σ')/T(2)                # regular add/div (no dots needed)
    if epsilon > 0
        s = T(epsilon) * T(mean(diag(Σsym)))
        return Symmetric(Σsym + s*I)    # <-- plain '+' with I works
    else
        return Symmetric(Σsym)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_sr(weights, μ, Σ; selection=Int[], epsilon=EPS_RIDGE, do_checks=false) -> Float64

Sharpe ratio of a given portfolio: `SR = (w' μ) / sqrt(w' Σ w)`.

If `selection` is provided, the SR is computed on that subvector/submatrix,
i.e. only the selected assets are considered. A small ridge can be added
via `epsilon` for numerical stability.

Returns `NaN` if the variance term becomes nonpositive or nonfinite.
"""
function compute_sr(
    weights::AbstractVector{<:Real},
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::AbstractVector{<:Integer}=Int[],
    epsilon::Real=EPS_RIDGE[],
    do_checks::Bool=false
)::Float64
    n = length(μ)
    do_checks && begin
        length(weights) == n || error("weights and μ must have the same length.")
        size(Σ) == (n, n)      || error("Σ must be $n×$n.")
        isempty(selection) || all(1 .≤ selection .≤ n) || error("selection out of bounds 1..$n.")
    end

    Σeff = _sym_with_ridge(Σ, epsilon)

    if isempty(selection) || length(selection) == n
        num = dot(weights, μ)
        v   = dot(weights, Σeff * weights)
    else
        sel = selection
        w   = weights[sel]
        μs  = μ[sel]
        Σs  = Symmetric(Matrix(Σeff)[sel, sel])  # dense submatrix with symmetry hint
        num = dot(w, μs)
        v   = dot(w, Σs * w)
    end

    den = sqrt(v)
    return (isfinite(den) && den > 0) ? float(num / den) : NaN
end


"""
    compute_mve_sr(μ, Σ; selection=Int[], epsilon=EPS_RIDGE, do_checks=false) -> Float64

Maximum Sharpe ratio (MVE) over the selected assets:
`MVE_SR = sqrt( μ' Σ^{-1} μ )`.

Uses a Cholesky solve when `Σ` is SPD; otherwise falls back to a pseudoinverse.
A small ridge (scaled by `mean(diag(Σ))`) can be added via `epsilon`.
"""
function compute_mve_sr(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::AbstractVector{<:Integer}=Int[],
    epsilon::Real=EPS_RIDGE[],
    do_checks::Bool=false
)::Float64
    n = length(μ)
    do_checks && begin
        size(Σ) == (n, n) || error("Σ must be $n×$n.")
        isempty(selection) || all(1 .≤ selection .≤ n) || error("selection out of bounds 1..$n.")
    end

    Σeff = _sym_with_ridge(Σ, epsilon)

    if isempty(selection) || length(selection) == n
        μs = μ
        Σs = Σeff
    else
        sel = selection
        μs = μ[sel]
        Σs = Symmetric(Matrix(Σeff)[sel, sel])
    end

    val = try
        # SPD path
        x = cholesky(Σs) \ μs
        dot(μs, x)
    catch
        # fallback: pseudoinverse (handles semidefinite/ill-conditioned)
        P = pinv(Matrix(Σs))
        dot(μs, P * μs)
    end

    return sqrt(max(float(val), 0.0))
end


"""
    compute_mve_weights(μ, Σ; selection=Int[], γ=1.0, epsilon=EPS_RIDGE, do_checks=false)
    -> Vector{Float64}

Mean-Variance Efficient (unconstrained) weights:
`w = (1/γ) · Σ^{-1} μ`.

Returns a length-`n` vector. If `selection` is provided, entries off-selection
are set to zero. No budget or sign constraints are imposed, and the vector is
**not** normalized to sum to one (do that downstream if desired).
"""
function compute_mve_weights(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::AbstractVector{<:Integer}=Int[],
    γ::Real=1.0,
    epsilon::Real=EPS_RIDGE[],
    do_checks::Bool=false
)::Vector{Float64}
    n = length(μ)
    do_checks && begin
        n > 0 || error("μ must be non-empty.")
        size(Σ) == (n, n) || error("Σ must be $n×$n.")
        γ > 0 || error("γ must be positive.")
        isempty(selection) || all(1 .≤ selection .≤ n) || error("selection out of bounds 1..$n.")
    end

    Σeff = _sym_with_ridge(Σ, epsilon)

    if isempty(selection) || length(selection) == n
        w = try
            cholesky(Σeff) \ μ
        catch
            pinv(Matrix(Σeff)) * μ
        end
        return Vector{Float64}(w ./ γ)
    else
        sel = selection
        μs  = μ[sel]
        Σs  = Symmetric(Matrix(Σeff)[sel, sel])
        ws  = try
            cholesky(Σs) \ μs
        catch
            pinv(Matrix(Σs)) * μs
        end
        w = zeros(Float64, n)
        @inbounds w[sel] .= ws ./ γ
        return w
    end
end

end # module
