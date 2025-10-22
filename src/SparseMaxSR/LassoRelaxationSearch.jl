module LassoRelaxationSearch

using LinearAlgebra
using Statistics
using GLMNet
using ..Utils
using ..SharpeRatio

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

    # Friendly handling of user-supplied lambda
    if lambda !== nothing
        all(isfinite, lambda) || error("`lambda` contains non-finite values.")
        all(lambda .> 0)      || error("`lambda` must be strictly positive.")
        # Accept non-increasing and drop duplicates
        if any(diff(lambda) .> 0)
            error("`lambda` must be non-increasing.")
        end
        λ = unique(lambda)  # preserves order, removes duplicates
        if length(λ) == 0
            error("`lambda` is empty after removing duplicates.")
        end
        lambda = λ
    end

    # Hard caps to accelerate: GLMNet will stop when df exceeds dfmax
    dfmax = k
    pmax  = max(k + 5, 2*k)

    kwargs_common = (
        alpha=alpha, intercept=false, standardize=standardize,
        dfmax=dfmax, pmax=pmax
    )

    path = isnothing(lambda) ?
        GLMNet.glmnet(X, y; nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, kwargs_common...) :
        GLMNet.glmnet(X, y; lambda=lambda, kwargs_common...)

    βmat = Array(path.betas)      # N × L (dense view of the path)
    L    = size(βmat, 2)

    best_len = -1
    best_idx = Int[]
    jstar    = 0

    @inbounds for j in L:-1:1
        col = @view βmat[:, j]
        # fast count first (no alloc)
        s = count(!iszero, col)
        if s ≤ k && s > best_len
            # only now collect indices (allocates once for a winner)
            best_idx = findall(!iszero, col)
            best_len = s
            jstar    = j
            if s == k
                break
            end
        end
    end

    if best_len < 0
        return (Int[], 0, zeros(Float64, size(X,2)), :LASSO_PATH_ALMOST_K, βmat)
    end

    status = (best_len == k) ? :LASSO_PATH_EXACT_K : :LASSO_PATH_ALMOST_K
    βj = βmat[:, jstar]
    return (best_idx, jstar, βj, status, βmat)
end

# Build LASSO-vanilla weights on full length.
# If normalize_weights=true: rescale coefficients with Utils.normalize_weights (relative L1 safeguard).
# If normalize_weights=false: return raw coefficients on the selected support.
# Returns (w, is_all_empty::Bool) when selection or coefficients are empty.
@inline function _lasso_weights_from_beta!(
    w::Vector{Float64},
    βj::AbstractVector{<:Real},
    sel::AbstractVector{<:Integer};
    normalize_weights::Bool,
    tol::Real = 1e-6
)
    fill!(w, 0.0)
    if isempty(sel)
        return w, true
    end
    b = βj[sel]
    if all(iszero, b)
        return w, true
    end
    if normalize_weights
        b_norm = Utils.normalize_weights(b; mode=:relative, tol=tol, do_checks=false)
        @inbounds w[sel] .= b_norm
    else
        @inbounds w[sel] .= b
    end
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
        epsilon::Real = Utils.EPS_RIDGE,
        stabilize_Σ::Bool = true,
        compute_weights::Bool = false,
        normalize_weights::Bool = false,
        use_refit::Bool = true,
        do_checks::Bool = false
    ) -> NamedTuple{(:selection, :weights, :sr, :status)}

Path-based **elastic net (LASSO) relaxation** of the mean–variance efficient (MVE) 
portfolio selection problem. Regress the target `y` on the asset return matrix `R`
(no intercept) using the GLMNet path solver, then select the largest support 
`s ≤ k` across the λ-path. This yields a sparse proxy for the optimal MVE support.

Two evaluation modes are available:

- **Refit mode (`use_refit=true`)**:
  - Only the *selected support* is used.
  - Computes the *exact* MVE Sharpe ratio via `SharpeRatio.compute_mve_sr`.
  - If `compute_weights=true`, refits closed-form MVE weights on the same support.
  - Optionally rescales weights if `normalize_weights=true` (using `Utils.normalize_weights`).

- **Vanilla mode (`use_refit=false`)**:
  - Returns the *raw LASSO coefficients* at the chosen λ.
  - If `normalize_weights=true`, coefficients are rescaled by `Utils.normalize_weights`
    for numerical stability (relative L1 safeguard).
  - Sharpe ratio is computed directly on these raw or normalized coefficients.

Normalization does **not** affect the Sharpe ratio (scale-invariant), but improves 
comparability and numerical robustness when coefficients are small or sign-imbalanced.

# Arguments
- `R::Matrix{<:Real}`: T×N matrix of asset excess returns.
- `k::Integer`: target sparsity (maximum support size).
- `y::Vector` (optional): response variable; defaults to a vector of ones.
- `nlambda`, `lambda_min_ratio`, `lambda`: GLMNet regularization path controls.
- `alpha`: elastic-net mixing parameter (α=1 → pure LASSO).
- `standardize`: whether to standardize regressors inside GLMNet.
- `epsilon`, `stabilize_Σ`: ridge stabilization and symmetrization of Σ = cov(R).
- `compute_weights`: return weight vector (full length, zeros off support).
- `normalize_weights`: whether to post-process weights via `Utils.normalize_weights`.
- `use_refit`: use MVE refit on the selected support (true) or vanilla LASSO (false).
- `do_checks`: toggle input validation and dimension checks.

# Returns
A named tuple with:
- `selection::Vector{Int}` – indices of selected assets;
- `weights::Vector{Float64}` – portfolio weights (possibly normalized);
- `sr::Float64` – in-sample Sharpe ratio;
- `status::Symbol` – path status (e.g. `:LASSO_PATH_EXACT_K`, `:LASSO_ALLEMPTY`, etc.).

# Notes
- The choice `y = ones(T)` corresponds to maximizing the mean return 
  relative to covariance structure (standard MVE interpretation).
- Defaults (`lambda_min_ratio=1e-3`, `alpha≈1`) mirror MATLAB and sklearn LASSO setups.
- Setting `use_refit=false` is faster but less accurate; refit mode is 
  typically preferred for reporting Sharpe ratios and portfolio composition.
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
    epsilon::Real = Utils.EPS_RIDGE,
    stabilize_Σ::Bool = true,
    compute_weights::Bool = false,
    normalize_weights::Bool = false,
    use_refit::Bool = true,
    do_checks::Bool = false
)
    T, N = size(R) 

    if do_checks
        T > 1 || error("R must have at least 2 rows.")
        N > 0 || error("R must have at least 1 column.")
        (0 ≤ k ≤ N) || error("k must be between 0 and N.")  # allow k=0 fast path

        if y !== nothing
            length(y) == T || error("Length of y must equal number of rows of R.")
            all(isfinite, y) || error("y contains non-finite values.")
        end

        # lambda policy: positive, NON-INCREASING (duplicates allowed), dedup in-place
        if lambda !== nothing
            all(isfinite, lambda) || error("`lambda` contains non-finite values.")
            all(lambda .> 0)      || error("`lambda` must be strictly positive.")
            any(diff(lambda) .> 0) && error("`lambda` must be non-increasing.")
            lambda = unique(lambda)  # preserve order; remove duplicates
            length(lambda) > 0 || error("`lambda` is empty after removing duplicates.")
        end

        isfinite(epsilon) || error("epsilon must be finite.")
        alpha ≥ 0 && alpha ≤ 1 || error("alpha must be in [0,1].")
    end

    # k == 0 or k == N fast paths
    if k == 0
        return (selection = Int[], weights = zeros(Float64, N), sr = 0.0, status = :LASSO_ALLEMPTY)
    elseif k == N
        μ  = vec(mean(R; dims=1))
        Σ  = cov(Matrix(R); corrected=true)
        Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)
        sr = SharpeRatio.compute_mve_sr(μ, Σs; stabilize_Σ=false)
        w  = compute_weights ? SharpeRatio.compute_mve_weights(μ, Σs; normalize_weights=normalize_weights, stabilize_Σ=false) : zeros(Float64, N)
        return (selection = collect(1:N), weights = w, sr = sr, status = :LASSO_PATH_EXACT_K)
    end

    yy = isnothing(y) ? ones(Float64, T) : y
    do_checks && (length(yy) == T || error("Length of y must equal number of rows of R."))

    best_idx, jstar, βj, path_status, _ = _glmnet_path_select(
        R, yy; k, nlambda, lambda_min_ratio, lambda, alpha, standardize, do_checks
    )

    # quick exit if no support found (keep path status)
    if isempty(best_idx)
        return (selection = best_idx,
                weights   = zeros(Float64, N),
                sr        = 0.0,
                status    = use_refit ? path_status : :LASSO_ALLEMPTY)
    end

    # moments & stabilized Σ (once)
    μ  = vec(mean(R; dims=1))
    Σ  = cov(Matrix(R); corrected=true)
    Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)

    if use_refit
        # Refit branch: SR independent of scaling; weights (if requested) inherit normalize_weights flag
        sr = SharpeRatio.compute_mve_sr(μ, Σs; selection=best_idx,
                            epsilon=epsilon, stabilize_Σ=false, do_checks=false)
        w  = compute_weights ?
             SharpeRatio.compute_mve_weights(μ, Σs; selection=best_idx,
                                 normalize_weights=normalize_weights,
                                 epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
             zeros(Float64, N)
        return (selection = sort(best_idx),
                weights   = w,
                sr        = sr,
                status    = path_status)
    else
        # Vanilla LASSO weights: raw or normalized (relative L1 safeguard)
        w = zeros(Float64, N)
        w, all_empty = _lasso_weights_from_beta!(w, βj, best_idx; normalize_weights=normalize_weights)
        status_final = all_empty ? :LASSO_ALLEMPTY : path_status
        sr = all_empty ? 0.0 : SharpeRatio.compute_sr(w, μ, Σs; stabilize_Σ=false, do_checks=false)
        return (selection = sort(best_idx),
                weights   = w,
                sr        = sr,
                status    = status_final)
    end
end

"""
    mve_lasso_relaxation_search(μ::AbstractVector, Σ::AbstractMatrix, T::Integer; kwargs...)

Moment-only variant. Build synthetic design with
Q = T(Σₛ + μμᵀ), take U from a safe Cholesky of Q, set X = Uᵀ and y = U \\ (Tμ),
then delegate to the shared path selector. In the vanilla branch, coefficients are either
raw (if `normalize_weights=false`) or rescaled with `Utils.normalize_weights` (if `true`).
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
    epsilon::Real = Utils.EPS_RIDGE,
    stabilize_Σ::Bool = true,
    compute_weights::Bool = false,
    normalize_weights::Bool = false,
    use_refit::Bool = true,
    do_checks::Bool = false
)
    N = length(μ)
    if do_checks
        N > 0 || error("μ must be non-empty.")
        size(Σ) == (N, N) || error("Σ must be N×N.")
        (1 ≤ k ≤ N) || error("k must be between 1 and N.")
        T ≥ 1 || error("T must be a positive integer.")

        all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite entries in μ or Σ.")

        # lambda policy: positive, NON-INCREASING (duplicates allowed), dedup
        if lambda !== nothing
            all(isfinite, lambda) || error("`lambda` contains non-finite values.")
            all(lambda .> 0)      || error("`lambda` must be strictly positive.")
            any(diff(lambda) .> 0) && error("`lambda` must be non-increasing.")
            lambda = unique(lambda)
            length(lambda) > 0 || error("`lambda` is empty after removing duplicates.")
        end

        isfinite(epsilon) || error("epsilon must be finite.")
        alpha ≥ 0 && alpha ≤ 1 || error("alpha must be in [0,1].")
    end

    Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)
    Q  = T .* (Matrix(Σs) .+ μ*μ')
    # non-negative, meaningful scale
    μQ = max(mean(diag(Q)), eps(Float64))
    τ  = eps(Float64) * μQ
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
                weights   = zeros(Float64, N),
                sr        = 0.0,
                status    = use_refit ? path_status : :LASSO_ALLEMPTY)
    end

    if use_refit
        sr = SharpeRatio.compute_mve_sr(μ, Σs; selection=best_idx,
                            epsilon=epsilon, stabilize_Σ=false, do_checks=false)
        w  = compute_weights ?
             SharpeRatio.compute_mve_weights(μ, Σs; selection=best_idx,
                                 normalize_weights=normalize_weights,
                                 epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
             zeros(Float64, N)
        return (selection = sort(best_idx),
                weights   = w,
                sr        = sr,
                status    = path_status)
    else
        w = zeros(Float64, N)
        w, all_empty = _lasso_weights_from_beta!(w, βj, best_idx; normalize_weights=normalize_weights)
        status_final = all_empty ? :LASSO_ALLEMPTY : path_status
        sr = all_empty ? 0.0 : SharpeRatio.compute_sr(w, μ, Σs; stabilize_Σ=false, do_checks=false)
        return (selection = sort(best_idx),
                weights   = w,
                sr        = sr,
                status    = status_final)
    end
end

end # module
