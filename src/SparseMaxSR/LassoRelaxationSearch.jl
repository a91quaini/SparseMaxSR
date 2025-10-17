module LassoRelaxationSearch

using LinearAlgebra, Statistics
using GLMNet

import ..SharpeRatio: compute_mve_sr, compute_mve_weights, compute_sr
import ..Utils: EPS_RIDGE, _prep_S

export mve_lasso_relaxation_search

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

function _safe_chol(Q::AbstractMatrix; base_bump=1e-10, max_tries=8)
    τ = base_bump * (tr(Q) / size(Q,1))
    for t in 0:max_tries
        try
            return cholesky(Symmetric(Q + (τ * 2.0^t) * I))
        catch
            # escalate bump
        end
    end
    error("Cholesky failed after $(max_tries+1) bumps")
end

# Fit GLMNet path on (X,y), then pick the largest support s ≤ k.
# Returns (best_idx, jstar, βj, path_status, βmat)
function _glmnet_path_select(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    k::Integer,
    nlambda::Int = 100,
    lambda_min_ratio::Real = 1e-3,
    lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
    alpha::Real = 0.95,
    standardize::Bool = false,
    do_checks::Bool = false
)
    T, N = size(X)
    do_checks && (length(y) == T || error("X and y have incompatible sizes."))
    do_checks && (1 ≤ k ≤ N || error("k must be between 1 and N."))

    if lambda !== nothing
        all(isfinite, lambda)   || error("`lambda` contains non-finite values.")
        all(lambda .> 0)        || error("`lambda` must be strictly positive.")
        all(diff(lambda) .< 0)  || error("`lambda` must be strictly decreasing.")
    end

    kwargs = (; alpha=alpha, intercept=false, standardize=standardize,
               nlambda=nlambda, lambda_min_ratio=lambda_min_ratio)
    path = isnothing(lambda) ?
        GLMNet.glmnet(X, y; kwargs...) :
        GLMNet.glmnet(X, y; alpha=alpha, intercept=false, standardize=standardize, lambda=lambda)

    βmat = Array(path.betas)  # N × L
    L = size(βmat, 2)

    best_idx = Int[]
    best_len = -1
    jstar    = 0
    @inbounds for j in L:-1:1
        S = findall(!iszero, @view βmat[:, j])
        s = length(S)
        if s ≤ k && s > best_len
            best_idx = S
            best_len = s
            jstar    = j
            if s == k
                break
            end
        end
    end

    if best_len < 0
        return (Int[], 0, zeros(Float64, N), :LASSO_PATH_ALMOST_K, βmat)
    end

    status = (best_len == k) ? :LASSO_PATH_EXACT_K : :LASSO_PATH_ALMOST_K
    βj = βmat[:, jstar]
    return (best_idx, jstar, βj, status, βmat)
end

# Build normalized-β weights (full length) with |sum(w)| = 1 if possible.
# Returns (w, is_all_empty::Bool) where "all empty" means we cannot enforce |sum|=1.
@inline function _normalized_lasso_weights(
    βj::AbstractVector{<:Real},
    sel::AbstractVector{<:Integer},
    N::Int;
    tol::Real = 1e-12
)
    w = zeros(Float64, N)
    if isempty(sel)
        return w, true
    end
    b = βj[sel]
    if all(iszero, b)
        return w, true
    end
    s_signed = sum(b)
    if !isfinite(s_signed) || abs(s_signed) < tol
        # Cannot scale to enforce |sum(w)|=1 (sum is ~0): declare ALLEMPTY
        return w, true
    end
    # Scale so that abs(sum(w)) = 1  (indeed sum(w) = sign(signed)*1)
    @inbounds w[sel] .= b ./ abs(s_signed)
    return w, false
end

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    mve_lasso_relaxation_search(R::AbstractMatrix{<:Real};
        k::Integer,
        y::Union{Nothing,AbstractVector{<:Real}} = nothing,
        nlambda::Int = 100,
        lambda_min_ratio::Real = 1e-3,
        lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
        alpha::Real = 0.95,
        standardize::Bool = false,
        epsilon::Real = EPS_RIDGE,
        stabilize_Σ::Bool = true,
        compute_weights::Bool = false,
        use_refit::Bool = true,
        do_checks::Bool = false
    ) -> NamedTuple{(:selection, :weights, :sr, :status)}

Path-based elastic net on returns: regress `y` on `R` with no intercept and take the
largest support `s ≤ k` from the λ-path. If `use_refit=true`, refit exact MVE on this
support; otherwise normalize the LASSO coefficients and scale so that `abs(sum(w))=1`.
If the selected coefficients sum to ~0 (cannot scale), the method returns all-zero
weights and `status = :LASSO_ALLEMPTY`.

By default, `y = ones(T)`.
"""
function mve_lasso_relaxation_search(
    R::AbstractMatrix{<:Real};
    k::Integer,
    y::Union{Nothing,AbstractVector{<:Real}} = nothing,
    nlambda::Int = 100,
    lambda_min_ratio::Real = 1e-3,
    lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
    alpha::Real = 0.95,
    standardize::Bool = false,
    epsilon::Real = EPS_RIDGE,
    stabilize_Σ::Bool = true,
    compute_weights::Bool = false,
    use_refit::Bool = true,
    do_checks::Bool = false
)
    T, N = size(R)
    do_checks && (T > 1 || error("R must have at least 2 rows."))
    do_checks && (N > 0 || error("R must have at least 1 column."))
    (1 ≤ k ≤ N) || error("k must be between 1 and N.")

    yy = isnothing(y) ? ones(Float64, T) : y
    do_checks && (length(yy) == T || error("Length of y must equal number of rows of R."))

    best_idx, jstar, βj, path_status, _ = _glmnet_path_select(
        R, yy; k, nlambda, lambda_min_ratio, lambda, alpha, standardize, do_checks
    )

    # quick exit if no support found (keep path status)
    if isempty(best_idx)
        return (selection = best_idx, 
                weights = zeros(Float64, N), 
                sr = 0.0, 
                status = use_refit ? path_status : :LASSO_ALLEMPTY)
    end

    # moments & stabilized Σ (once)
    μ  = vec(mean(R; dims=1))
    Σ  = cov(Matrix(R); corrected=true)
    Σs = _prep_S(Σ, epsilon, stabilize_Σ)

    if use_refit
        # Refit branch: weights depend on compute_weights flag (can legally be all zeros)
        sr = compute_mve_sr(μ, Σs; selection=best_idx,
                            epsilon=epsilon, stabilize_Σ=false, do_checks=false)
        w  = compute_weights ?
             compute_mve_weights(μ, Σs; selection=best_idx,
                                 epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
             zeros(Float64, N)
        return (selection = sort(best_idx), 
                weights = w, 
                sr = sr, 
                status = path_status)
    else
        # Normalize-coeffs branch with |sum(w)| = 1 if possible; else ALLEMPTY
        w, all_empty = _normalized_lasso_weights(βj, best_idx, N)
        status_final = all_empty ? :LASSO_ALLEMPTY : path_status
        sr = all_empty ? 0.0 : compute_sr(w, μ, Σs; stabilize_Σ=false, do_checks=false)
        return (selection = sort(best_idx), 
                weights = w, 
                sr = sr, 
                status = status_final)
    end
end

"""
    mve_lasso_relaxation_search(μ::AbstractVector, Σ::AbstractMatrix, T::Integer; kwargs...)

Moment-only variant. Build synthetic design with
Q = T(Σₛ + μμᵀ), take U from a safe Cholesky of Q, set X = Uᵀ and y = U \\ (Tμ),
then delegate to the shared path selector. In the normalize branch, weights are
scaled so that `abs(sum(w))=1` where possible; otherwise `:LASSO_ALLEMPTY`.
"""
function mve_lasso_relaxation_search(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    T::Integer;
    k::Integer,
    nlambda::Int = 100,
    lambda_min_ratio::Real = 1e-3,
    lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
    alpha::Real = 0.95,
    standardize::Bool = false,
    epsilon::Real = EPS_RIDGE,
    stabilize_Σ::Bool = true,
    compute_weights::Bool = false,
    use_refit::Bool = true,
    do_checks::Bool = false
)
    N = length(μ)
    do_checks && (size(Σ,1) == N && size(Σ,2) == N || error("Σ must be N×N."))
    (1 ≤ k ≤ N) || error("k must be between 1 and N.")

    Σs = _prep_S(Σ, epsilon, stabilize_Σ)
    Q  = T .* (Matrix(Σs) .+ μ*μ')

    # tiny bump for safety
    τ = eps(Float64) * (tr(Q) / size(Q,1))
    @inbounds for i in 1:N
        Q[i,i] += τ
    end
    U = _safe_chol(Q; base_bump=eps(Float64)).U
    X = transpose(U)
    y = U \ (T .* μ)

    best_idx, jstar, βj, path_status, _ = _glmnet_path_select(
        X, y; k, nlambda, lambda_min_ratio, lambda, alpha, standardize, do_checks
    )

    if isempty(best_idx)
        return (selection = best_idx, 
                weights = zeros(Float64, N), 
                sr = 0.0, 
                status = use_refit ? path_status : :LASSO_ALLEMPTY)
    end

    if use_refit
        sr = compute_mve_sr(μ, Σs; selection=best_idx,
                            epsilon=epsilon, stabilize_Σ=false, do_checks=false)
        w  = compute_weights ?
             compute_mve_weights(μ, Σs; selection=best_idx,
                                 epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
             zeros(Float64, N)
        return (selection = sort(best_idx), 
                weights = w, 
                sr = sr, 
                status = path_status)
    else
        w, all_empty = _normalized_lasso_weights(βj, best_idx, N)
        status_final = all_empty ? :LASSO_ALLEMPTY : path_status
        sr = all_empty ? 0.0 : compute_sr(w, μ, Σs; stabilize_Σ=false, do_checks=false)
        return (selection = sort(best_idx), 
                weights = w, 
                sr = sr, 
                status = status_final)
    end
end

end # module
