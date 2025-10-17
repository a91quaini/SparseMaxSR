module SharpeRatio

using LinearAlgebra
using Statistics
import ..Utils: EPS_RIDGE, _prep_S

export compute_sr,
       compute_mve_sr,
       compute_mve_weights

# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

# Solve Σ x = b using the best available symmetric solver:
# 1) Try Cholesky (SPD).
# 2) Try LDLᵀ (symmetric indefinite/PSD with pivoting).
# 3) Fallback: pseudoinverse (last resort).
@inline function _sym_solve(Σs::Symmetric{T,<:AbstractMatrix{T}}, b::AbstractVector{T}) where {T<:Real}
    try
        return cholesky(Σs) \ b
    catch
        try
            return ldlt(Σs) \ b
        catch
            return pinv(Matrix(Σs)) * b
        end
    end
end

# =============================================================================
# Public API
# =============================================================================

"""
    compute_sr(weights, μ, Σ;
               selection=Int[], epsilon=EPS_RIDGE, stabilize_Σ=true, do_checks=true) -> Float64

Compute the Sharpe ratio of a given portfolio:

SR = (w' μ) / sqrt(w' Σ w).

# Arguments
- `weights`: portfolio weights.
- `μ`: mean vector of asset excess returns.
- `Σ`: covariance matrix of asset excess returns.
- `selection`: (optional) subset of indices on which to compute SR.
- `epsilon`: ridge added during stabilization for numerical safety.
- `stabilize_Σ`: whether to symmetrize and ridge-stabilize Σ.
- `do_checks`: validate dimensions and finiteness.

Returns `NaN` if the variance term is nonpositive or nonfinite.
"""
function compute_sr(
    weights::AbstractVector{<:Real},
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::AbstractVector{<:Integer}=Int[],
    epsilon::Real=EPS_RIDGE,
    stabilize_Σ::Bool=true,
    do_checks::Bool=true
)::Float64
    n = length(μ)

    if do_checks
        length(weights) == n || error("weights and μ must have the same length.")
        size(Σ) == (n, n) || error("Σ must be $n×$n.")
        isempty(selection) || all(1 .≤ selection .≤ n) || error("selection out of bounds 1..$n.")
        all(isfinite, weights) && all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite inputs.")
        isfinite(epsilon) || error("epsilon must be finite.")
    end

    Σeff = _prep_S(Σ, epsilon, stabilize_Σ)

    if isempty(selection) || length(selection) == n
        num = dot(weights, μ)
        v   = dot(weights, Σeff * weights)
    else
        sel = selection
        w   = weights[sel]
        μs  = μ[sel]
        Σs  = Symmetric(Σeff[sel, sel])
        num = dot(w, μs)
        v   = dot(w, Σs * w)
    end

    v = float(v)
    if !isfinite(v) || v <= 0
        return NaN
    end
    return num / sqrt(v)
end


"""
    compute_mve_sr(μ, Σ;
                   selection=Int[], epsilon=EPS_RIDGE, stabilize_Σ=true, do_checks=true) -> Float64

Compute the **maximum Sharpe ratio** achievable over a subset of assets:

MVE_{SR} = sqrt(μ' Σ^{-1} μ).

# Details
Uses numerically stable symmetric linear solves:
- Cholesky for SPD matrices;
- LDLᵀ (pivoted) for PSD/indefinite cases;
- pseudoinverse fallback if both fail.

`stabilize_Σ=true` ensures Σ is symmetrized and ridge-stabilized to aid convergence.
"""
function compute_mve_sr(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::AbstractVector{<:Integer}=Int[],
    epsilon::Real=EPS_RIDGE,
    stabilize_Σ::Bool=true,
    do_checks::Bool=true
)::Float64
    n = length(μ)

    if do_checks
        size(Σ) == (n, n) || error("Σ must be $n×$n.")
        isempty(selection) || all(1 .≤ selection .≤ n) || error("selection out of bounds 1..$n.")
        all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite inputs.")
        isfinite(epsilon) || error("epsilon must be finite.")
    end

    Σeff = _prep_S(Σ, epsilon, stabilize_Σ)

    if isempty(selection) || length(selection) == n
        μs = μ
        Σs = Σeff
    else
        sel = selection
        μs = μ[sel]
        Σs = Symmetric(Σeff[sel, sel])
    end

    x   = _sym_solve(Σs, μs)
    val = dot(μs, x)
    return sqrt(max(float(val), 0.0))
end


"""
    compute_mve_weights(μ, Σ;
                        selection=Int[],
                        epsilon=EPS_RIDGE, stabilize_Σ=true, do_checks=true) -> Vector{Float64}

Compute **mean–variance efficient (MVE) portfolio weights**:


w^* = Σ^{-1} μ.

# Arguments
- `μ`: mean vector of excess returns.
- `Σ`: covariance matrix.
- `selection`: (optional) subset of indices; others are set to zero.
- `epsilon`: ridge stabilization parameter.
- `stabilize_Σ`: symmetrize and ridge-stabilize Σ before inversion.
- `do_checks`: input validation.

# Returns
A vector of MVE weights.
"""
function compute_mve_weights(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real};
    selection::AbstractVector{<:Integer}=Int[],
    epsilon::Real=EPS_RIDGE,
    stabilize_Σ::Bool=true,
    do_checks::Bool=true
)::Vector{Float64}
    n = length(μ)

    if do_checks
        n > 0 || error("μ must be non-empty.")
        size(Σ) == (n, n) || error("Σ must be $n×$n.")
        isempty(selection) || all(1 .≤ selection .≤ n) || error("selection out of bounds 1..$n.")
        all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite inputs.")
        isfinite(epsilon) || error("epsilon must be finite.")
    end

    Σeff = _prep_S(Σ, epsilon, stabilize_Σ)

    if isempty(selection) || length(selection) == n
        w = _sym_solve(Σeff, μ)
    else
        sel = selection
        μs  = μ[sel]
        Σs  = Symmetric(Σeff[sel, sel])
        ws  = _sym_solve(Σs, μs)
        w   = zeros(Float64, n)
        @inbounds w[sel] .= ws
    end

    return w
end

end # module
