module Utils

using LinearAlgebra

# ──────────────────────────────────────────────────────────────────────────────
# Numerical knobs shared across SharpeRatio, Lasso, MIQP
# ──────────────────────────────────────────────────────────────────────────────

# Small ridge used to stabilize covariance matrices where requested.
const EPS_RIDGE = 1e-6

# Prepare a symmetric (and optionally ridge-stabilized) covariance:
# If `stabilize` is true:  Σ_eff = Sym( (Σ+Σ')/2 + ε·mean(diag(Σ))·I )
# If `stabilize` is false: Σ_eff = Sym( (Σ+Σ')/2 )  (no ridge)
@inline function _prep_S(Σ::AbstractMatrix{T}, epsilon::Real, stabilize::Bool) where {T<:Real}
    M = Matrix{Float64}(Σ)          # promote once
    A = (M .+ M') ./ T(2)           # symmetrize without assuming Symmetric wrapper
    n = size(A, 1)
    if stabilize && epsilon > 0
        ss = T(epsilon) * T(tr(A) / n)
        return Symmetric(Matrix(A) .+ ss .* I(n))
    else
        return Symmetric(Matrix(A))
    end
end

"""
    normalize_weights(w::AbstractVector;
                      mode::Symbol = :relative,
                      tol::Real = 1e-6,
                      do_checks::Bool = false) -> Vector{Float64}

Normalize a vector of portfolio weights so that its absolute sum is approximately equal to 1.

The vector `w` is rescaled according to the specified normalization `mode`:

- `mode = :absolute`:
    Divides `w` by `max(abs(sum(w)), tol, 1e-10)`.
    This ensures the **absolute sum** of the resulting weights equals 1 (up to tolerance).

- `mode = :relative` (default):
    Divides `w` by `max(abs(sum(w)), tol * norm(w, 1), 1e-10)`.
    This rescales weights **relative to their L1 norm**, improving stability when the sum of weights is near zero
    but individual elements are not negligible.

Optional argument checks (`do_checks = true`) ensure:
- `mode` is either `:absolute` or `:relative`;
- all elements of `w` are finite;
- `tol` is positive and finite.

Always returns a finite `Vector{Float64}` of the same length as `w`.
"""
function normalize_weights(w::AbstractVector;
                           mode::Symbol = :relative,
                           tol::Real = 1e-6,
                           do_checks::Bool = false)::Vector{Float64}
    s = sum(w)

    if do_checks
        (mode === :absolute || mode === :relative) ||
            error("normalize_weights: mode must be :absolute or :relative (got $mode)")
        all(isfinite, w) || error("weights must be finite.")
        isfinite(tol) || error("tol must be finite.")
        tol > 0 || error("tol must be positive.")
    end

    denom = if mode === :absolute
        max(abs(s), tol, 1e-10)
    else
        max(abs(s), tol * norm(w, 1), 1e-10)
    end

    return Float64.(w ./ denom)
end

export EPS_RIDGE, _prep_S, normalize_weights

end # module
