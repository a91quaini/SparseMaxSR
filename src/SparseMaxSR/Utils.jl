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
                      tol::Real=1e-12) -> (w_norm::Vector{Float64}, status::Symbol, s::Float64)

Return a **rescaled copy** of `w` so that:
- `mode = :sum`  →  `sum(w_norm) = target`
- `mode = :abs`  →  `abs(sum(w_norm)) = target` (sign of `sum(w)` is preserved)

If `abs(sum(w)) ≤ tol`, returns `zeros(length(w))` and `status = :ZERO_SUM`.
Otherwise returns the rescaled vector and `status = :OK`.

`target` is positive in both modes.

Examples
--------
```julia
w = [0.2, 0.3, 0.1]
w1, st, s = make_weights_sum1(w)             # sum→1.0
w2, st, s = make_weights_sum1(w; mode=:abs)  # |sum|→1.0
w3, st, s = make_weights_sum1(w; target=2.0) # sum→2.0
```
"""
function make_weights_sum1(w::AbstractVector;
                           target::Real=1.0,
                           mode::Symbol=:sum,
                           tol::Real=1e-12)
    s = sum(w)
    if !(mode === :sum || mode === :abs)
        error("make_weights_sum1: mode must be :sum or :abs (got $mode)")
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

"""
    make_weights_sum1!(w::AbstractVector;
                       target::Real=1.0,
                       mode::Symbol=:sum,
                       tol::Real=1e-12) -> (status::Symbol, s::Float64)

**In-place** version. See `make_weights_sum1`.
Mutates `w` if rescaling is feasible; otherwise fills it with zeros and returns `:ZERO_SUM`.
"""
function make_weights_sum1!(w::AbstractVector;
                            target::Real=1.0,
                            mode::Symbol=:sum,
                            tol::Real=1e-12)
    s = sum(w)
    if !(mode === :sum || mode === :abs)
        error("make_weights_sum1!: mode must be :sum or :abs (got $mode)")
    end
    if !isfinite(s) || abs(s) ≤ tol || !isfinite(target) || target ≤ 0
        fill!(w, 0.0)
        return :ZERO_SUM, s
    end
    scale = if mode === :sum
        target / s
    else
        (sign(s) * target) / s
    end
    @inbounds for i in eachindex(w)
        w[i] = Float64(w[i] * scale)
    end
    return :OK, s
end

export EPS_RIDGE, _prep_S, make_weights_sum1, make_weights_sum1!

end # module
