module Utils

using LinearAlgebra
using Statistics

# ──────────────────────────────────────────────────────────────────────────────
# Numerical knobs shared across SharpeRatio, Lasso, MIQP
# ──────────────────────────────────────────────────────────────────────────────

# Small ridge used to stabilize covariance matrices where requested.
const EPS_RIDGE::Float64 = 1e-6

# Prepare a symmetric (and optionally ridge-stabilized) covariance:
# If `stabilize` is true:  Σ_eff = Sym( (Σ+Σ')/2 + ε·mean(diag(Σ))·I )
# If `stabilize` is false: Σ_eff = Sym( (Σ+Σ')/2 )  (no ridge)
@inline function _prep_S(Σ::AbstractMatrix{T}, 
                         epsilon::Real, 
                         stabilize::Bool)::Symmetric{Float64, Matrix{Float64}} where {T<:Real}
    
    A = Matrix{Float64}(Σ)          # promote once
    n = size(A, 1)

    # In-place symmetrization: A := (A + A') / 2
    @inbounds begin
        for j in 1:n
            A[j,j] = (A[j,j] + A[j,j]) * 0.5
            for i in j+1:n
                s = 0.5 * (A[i,j] + A[j,i])
                A[i,j] = s
                A[j,i] = s
            end
        end
    end

    if stabilize && epsilon > 0
        ss = epsilon * (tr(A) / n)
        # add "ss" to the diagonal (or A .+ ss .* I(n) — both are fine)
        @inbounds for i in 1:n
            A[i,i] += ss
        end
    end

    return Symmetric(A)
end

"""
    normalize_weights(w::AbstractVector;
                      mode::Symbol = :relative,
                      tol::Real = 1e-6,
                      do_checks::Bool = false) -> Vector{Float64}

Return a rescaled copy of `w` with stable normalization.

- `mode = :absolute`:
    Scale by `max(abs(sum(w)), tol, 1e-10)`.
    Effect: the **magnitude of the sum** |∑ w_norm| is ≈ 1 (sign of ∑w preserved).

- `mode = :relative` (default):
    Scale by `max(abs(sum(w)), tol*norm(w,1), 1e-10)`.
    Effect: avoids blow-ups when ∑w ≈ 0 but entries are not negligible, by
    tying the scale to the L1 mass if needed.

If both ∑w and ‖w‖₁ are ≲ tol (numerically empty portfolio), returns a zero vector.

Set `do_checks=true` to validate inputs.
"""
function normalize_weights(w::AbstractVector; mode::Symbol=:relative, tol::Real=1e-6, do_checks::Bool=false)
    s  = sum(w)
    d1 = norm(w, 1)

    if do_checks
        (mode === :absolute || mode === :relative) || error("mode must be :absolute or :relative")
        all(isfinite, w) || error("weights must be finite.")
        isfinite(tol) && tol > 0 || error("tol must be positive and finite.")
    end

    denom = (mode === :absolute) ?
        max(abs(s), tol, 1e-10) :
        max(abs(s), tol*d1, 1e-10)

    return Float64.(w) ./ denom
end

export EPS_RIDGE, _prep_S, normalize_weights

end # module
