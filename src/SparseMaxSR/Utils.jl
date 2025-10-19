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
    make_weights_sum1(w::AbstractVector;
                      target::Real=1.0,
                      mode::Symbol=:sum,
                      tol::Real=1e-7,
                      do_checks::Bool=false) -> (w_norm::Vector{Float64}, status::Symbol, s::Float64)

Return a **rescaled copy** of `w` so that:
- `mode = :sum`  →  `sum(w_norm) = target`
- `mode = :abs`  →  `abs(sum(w_norm)) = target` (sign of `sum(w)` is preserved)

If `abs(sum(w)) ≤ tol`, returns `zeros(length(w))` and `status = :ZERO_SUM`.
Otherwise returns the rescaled vector and `status = :OK`.

`target` is positive in both modes.
"""
function make_weights_sum1(w::AbstractVector;
                           target::Real=1.0,
                           mode::Symbol=:sum,
                           tol::Real=1e-7,
                           do_checks::Bool=false)
    s = sum(w)
    if do_checks
        (mode === :sum || mode === :abs) ||
            error("make_weights_sum1: mode must be :sum or :abs (got $mode)")
        isfinite(target)      || error("target must be finite.")
        target > 0            || error("target must be positive.")
        all(isfinite, w)      || error("weights must be finite.")
    end

    if !isfinite(s) || abs(s) ≤ tol || !isfinite(target) || target ≤ 0
        return zeros(Float64, length(w)), :ZERO_SUM, s
    end
    scale = if mode === :sum
        target / s
    else
        (sign(s) * target) / s
    end
    return Float64.(w .* scale), :OK, s
end

export EPS_RIDGE, _prep_S, make_weights_sum1

end # module
