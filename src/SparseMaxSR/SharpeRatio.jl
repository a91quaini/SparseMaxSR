module SharpeRatio

using LinearAlgebra
using Statistics
using ..Utils

export compute_sr,
       compute_mve_sr,
       compute_mve_weights

# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

@inline function _sym_solve(Σs::Symmetric{T,<:AbstractMatrix{T}}, b::AbstractVector{T}) where {T<:Real}
    try
        return cholesky(Σs) \ b
    catch
        try
            return ldlt(Σs) \ b
        catch
            # Try pseudoinverse BEFORE any artificial ridge (tests expect pinv(0)=0)
            A = Matrix(Σs)
            x = pinv(A) * b
            if all(isfinite, x)
                return x
            end
            # Last resort: tiny ridge
            δ = max(eps(T), 1e-10 * mean(diag(A)))
            @inbounds for i in 1:size(A,1)
                A[i,i] += δ
            end
            return cholesky!(A) \ b
        end
    end
end

# =============================================================================
# Public API
# =============================================================================

"""
    compute_sr(weights, μ, Σ;
               selection=Int[], epsilon=Utils.EPS_RIDGE, stabilize_Σ=true, do_checks=false) -> Float64

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
function compute_sr(weights::AbstractVector{<:Real},
                    μ::AbstractVector{<:Real},
                    Σ::AbstractMatrix{<:Real};
                    selection::AbstractVector{<:Integer}=Int[],
                    epsilon::Real=Utils.EPS_RIDGE,
                    stabilize_Σ::Bool=true,
                    do_checks::Bool=false)::Float64
    n = length(μ)
    if do_checks
        length(weights) == n || error("weights and μ must have the same length.")
        size(Σ) == (n, n)   || error("Σ must be $n×$n.")
        isempty(selection) || all(1 .≤ selection .≤ n) || error("selection out of bounds 1..$n.")
        all(isfinite, weights) && all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite inputs.")
        isfinite(epsilon) || error("epsilon must be finite.")
    end

    Σeff = Utils._prep_S(Σ, epsilon, stabilize_Σ)

    if isempty(selection) || length(selection) == n
        num = dot(weights, μ)
        tmp = similar(weights, Float64)
        mul!(tmp, Σeff, weights)              # tmp = Σeff * w
        v = dot(weights, tmp)
    else
        @views begin
            sel = selection
            w   = Float64.(weights[sel])      # small copy is fine here
            μs  = Float64.(μ[sel])
            Σs  = Symmetric(Matrix(Σeff[sel, sel]))  # keep symmetry explicit
            num = dot(w, μs)
            tmp = similar(w)
            mul!(tmp, Σs, w)
            v   = dot(w, tmp)
        end
    end

    v = float(v)
    return (isfinite(v) && v > 0) ? num / sqrt(v) : NaN
end

function compute_mve_sr(μ::AbstractVector{<:Real},
                        Σ::AbstractMatrix{<:Real};
                        selection::AbstractVector{<:Integer}=Int[],
                        epsilon::Real=Utils.EPS_RIDGE,
                        stabilize_Σ::Bool=true,
                        do_checks::Bool=false)::Float64
    n = length(μ)
    if do_checks
        size(Σ) == (n, n) || error("Σ must be $n×$n.")
        isempty(selection) || all(1 .≤ selection .≤ n) || error("selection out of bounds 1..$n.")
        all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite inputs.")
        isfinite(epsilon) || error("epsilon must be finite.")
    end

    Σeff = Utils._prep_S(Σ, epsilon, stabilize_Σ)

    if isempty(selection) || length(selection) == n
        μs = Float64.(μ)
        Σs = Σeff
    else
        @views begin
            sel = selection
            μs = Float64.(μ[sel])
            Σs = Symmetric(Matrix(Σeff[sel, sel]))
        end
    end

    x   = _sym_solve(Σs, μs)
    val = dot(μs, x)
    return sqrt(max(float(val), 0.0))
end

"""
    compute_mve_weights(μ, Σ;
                        selection=Int[],
                        normalize_weights::Bool=false,
                        epsilon::Real=Utils.EPS_RIDGE,
                        stabilize_Σ::Bool=true,
                        do_checks::Bool=false) -> Vector{Float64}

Compute **mean–variance efficient (MVE) portfolio weights**:

    w = Σ^{-1} μ

If `normalize_weights=true`, the returned vector is post-processed with
`Utils.normalize_weights(w; mode=:relative, tol=1e-6, do_checks=false)`, i.e.,
relative L1 normalization with a small safeguard.
This does **not** affect Sharpe ratios (scale-invariant).
"""
function compute_mve_weights(μ::AbstractVector{<:Real},
                             Σ::AbstractMatrix{<:Real};
                             selection::AbstractVector{<:Integer}=Int[],
                             normalize_weights::Bool=false,
                             epsilon::Real=Utils.EPS_RIDGE,
                             stabilize_Σ::Bool=true,
                             do_checks::Bool=false)::Vector{Float64}
    n = length(μ)
    if do_checks
        n > 0 || error("μ must be non-empty.")
        size(Σ) == (n, n) || error("Σ must be $n×$n.")
        isempty(selection) || all(1 .≤ selection .≤ n) || error("selection out of bounds 1..$n.")
        all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite inputs.")
        isfinite(epsilon) || error("epsilon must be finite.")
    end

    Σeff = Utils._prep_S(Σ, epsilon, stabilize_Σ)

    w = if isempty(selection) || length(selection) == n
        _sym_solve(Σeff, Float64.(μ))
    else
        sel = selection
        μs  = Float64.(μ[sel])
        Σs  = Symmetric(Matrix(Σeff[sel, sel]))
        ws  = _sym_solve(Σs, μs)
        out = zeros(Float64, n)
        @inbounds out[sel] .= ws
        out
    end

    if normalize_weights
        # Protect against “numerically empty” solution vectors
        if !(norm(w,1) > 1e-12) && !(abs(sum(w)) > 1e-12)
            return zeros(Float64, length(w))
        end
        w = Utils.normalize_weights(w)  # :relative, tol=1e-6
    end
    return w
end

# Fast-path overloads (no symmetrize/ridge each call)
compute_sr(w::AbstractVector{<:Real}, μ::AbstractVector{<:Real}, Σs::Symmetric; kwargs...) =
    compute_sr(w, μ, Matrix(Σs); kwargs...)

compute_mve_sr(μ::AbstractVector{<:Real}, Σs::Symmetric; kwargs...) =
    compute_mve_sr(μ, Matrix(Σs); kwargs...)

compute_mve_weights(μ::AbstractVector{<:Real}, Σs::Symmetric; kwargs...) =
    compute_mve_weights(μ, Matrix(Σs); kwargs...)


end # module
