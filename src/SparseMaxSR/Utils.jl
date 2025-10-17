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

export EPS_RIDGE, _prep_S

end # module
