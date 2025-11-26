module LassoRelaxationUtils

using LinearAlgebra
using Statistics
using GLMNet
using ..Utils
using ..SharpeRatio

export _safe_chol,
       _validate_lambda!,
       _alpha_mode,
       _moments_from_R,
       _validate_R_based_inputs!,
       _validate_moment_based_inputs!,
       _glmnet_path_select,
       _select_lambda_by_target_k_strict,
       _rss_at_path_column,
       _df_ridge_only,
       _gcv_log,
       _select_alpha_by_gcv,
       _mk_status,
       _finalize_from_selection!,
       _densify_lambda_exact_k,
       _lasso_weights_from_beta!,
       _rolling_folds,
       _design_from_moments

# ────────────────────────── Low-level numerical helpers ───────────────────────

"""
    _safe_chol(Q; base_bump=1e-10, max_tries=8) -> Cholesky

Robust Cholesky with escalating diagonal bump: tries
`Q + τ I, Q + 2τ I, Q + 4τ I, …` up to `max_tries`. Deterministic.
"""
function _safe_chol(Q::AbstractMatrix; base_bump=1e-10, max_tries=8)
    n = size(Q,1)
    τ0 = base_bump * max(tr(Q) / max(n,1), eps(Float64))
    @inbounds for t in 0:max_tries
        τ = τ0 * (2.0^t)
        try
            return cholesky(Symmetric(Q + τ * I))
        catch
            # keep escalating
        end
    end
    error("Cholesky failed after $(max_tries + 1) diagonal bumps.")
end

@inline function _validate_lambda!(lambda)
    if lambda === nothing
        return nothing
    end
    all(isfinite, lambda) || error("`lambda` contains non-finite values.")
    all(lambda .> 0)      || error("`lambda` must be strictly positive.")
    # enforce non-increasing order deterministically
    sort!(lambda; rev=true)
    unique!(lambda)
    !isempty(lambda) || error("`lambda` is empty after removing duplicates.")
    return lambda
end

@inline function _alpha_mode(alpha)
    if alpha isa AbstractVector
        # deterministic, sanitized grid
        agrid = unique(sort(Float64.(alpha)))
        return true, agrid
    else
        return false, Float64[float(alpha)]
    end
end

@inline function _moments_from_R(R::AbstractMatrix)
    @views μ = vec(mean(R; dims=1))
    @views Σ = cov(R; corrected=true)
    return μ, Σ
end

# ───────────────────────────── Shared input validators ───────────────────────

"""
    _validate_R_based_inputs!(R, k, y, epsilon, alpha, lambda, do_checks, cv_folds, alpha_select)
        -> (lambda, is_grid, agrid::Vector{Float64}, alpha_used::Float64)
"""
function _validate_R_based_inputs!(
    R::AbstractMatrix{<:Real},
    k::Integer,
    y,
    epsilon::Real,
    alpha,
    lambda,
    do_checks::Bool,
    cv_folds::Int,
    alpha_select::Symbol
)
    T, N = size(R)
    is_grid, agrid = _alpha_mode(alpha)
    alpha_used = agrid[1]

    if do_checks
        T > 1 || error("R must have at least 2 rows.")
        N > 0 || error("R must have at least 1 column.")
        (0 ≤ k ≤ N) || error("k must be between 0 and N.")
        if y !== nothing
            length(y) == T || error("Length of y must equal number of rows of R.")
            all(isfinite, y) || error("y contains non-finite values.")
        end
        isfinite(epsilon) || error("epsilon must be finite.")

        if alpha_select === :gcv
            is_grid || error("GCV selection requires `alpha` to be a vector grid.")
            length(agrid) ≥ 1 || error("GCV requires a non-empty alpha grid.")
            all(0.0 .≤ agrid .≤ 1.0) || error("alpha grid must be within [0,1].")
        elseif is_grid
            all(0.0 .≤ agrid .≤ 1.0) || error("alpha grid must be within [0,1].")
            length(agrid) ≥ 2 || error("Provide ≥ 2 alpha values to cross-validate.")
            # basic feasibility of folds
            seg = fld(T, cv_folds + 1)
            seg ≥ 2 || error("Not enough observations for cv_folds=$(cv_folds). Need ≥ 2 per validation slice.")
        else
            (0.0 ≤ float(alpha) ≤ 1.0) || error("alpha must be in [0,1].")
        end
    end
    lambda = _validate_lambda!(lambda)
    return lambda, is_grid, agrid, float(alpha_used)
end

"""
    _validate_moment_based_inputs!(μ, Σ, T, R, k, epsilon, alpha, lambda, do_checks, cv_folds, alpha_select)
        -> (lambda, is_grid, agrid::Vector{Float64}, alpha_used::Float64)
"""
function _validate_moment_based_inputs!(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    T::Integer,
    R,
    k::Integer,
    epsilon::Real,
    alpha,
    lambda,
    do_checks::Bool,
    cv_folds::Int,
    alpha_select::Symbol
)
    N = length(μ)
    is_grid, agrid = _alpha_mode(alpha)
    alpha_used = agrid[1]

    if do_checks
        N > 0 || error("μ must be non-empty.")
        size(Σ) == (N, N) || error("Σ must be N×N.")
        (1 ≤ k ≤ N) || error("k must be between 1 and N.")
        T ≥ 1 || error("T must be a positive integer.")
        all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite entries in μ or Σ.")
        isfinite(epsilon) || error("epsilon must be finite.")

        if alpha_select === :gcv
            is_grid || error("GCV selection requires `alpha` to be a vector grid.")
            length(agrid) ≥ 1 || error("GCV requires a non-empty alpha grid.")
            all(0.0 .≤ agrid .≤ 1.0) || error("alpha grid must be within [0,1].")
        elseif is_grid
            all(0.0 .≤ agrid .≤ 1.0) || error("alpha grid must be within [0,1].")
            length(agrid) ≥ 2 || error("Provide ≥ 2 alpha values to cross-validate.")
            R !== nothing || error("alpha grid OOS-CV requires raw returns `R`.")
            TR = size(R,1)
            seg = fld(TR, cv_folds + 1)
            seg ≥ 2 || error("Not enough observations for cv_folds=$(cv_folds). Need ≥ 2 per validation slice.")
            size(R, 2) == N || error("R must have N columns (same as length(μ)).")
            TR ≥ 2 || error("R must have at least 2 rows.")
            all(isfinite, R) || error("Non-finite entries in R.")
        else
            (0.0 ≤ float(alpha) ≤ 1.0) || error("alpha must be in [0,1].")
        end
    end
    lambda = _validate_lambda!(lambda)
    return lambda, is_grid, agrid, float(alpha_used)
end

# Lightweight container for a GLMNet path we build from moments
struct _GLMPath
    lambdas::Vector{Float64}  # length L, descending
    models::Vector{NamedTuple{(:beta,:sel),Tuple{Vector{Float64},Vector{Int}}}}  # length L
end

@inline _nnz_from_model(m) = length(m.sel)

# Fit a GLMNet path from (μ, Σ, T) by building the synthetic design (X,y)
function _fit_glmnet_path(μ::AbstractVector, Σ::AbstractMatrix, T::Integer;
    nlambda::Int,
    lambda_min_ratio::Real,
    lambda,
    alpha::Real,
    alpha_select::Symbol,
    standardize::Bool,
    epsilon::Real,
    stabilize_Σ::Bool,
    cv_folds::Int,
    cv_verbose::Bool,
    gcv_kappa::Real
)
    X, y, _ = _design_from_moments(μ, Σ, T; epsilon=epsilon, stabilize_Σ=stabilize_Σ)

    kwargs_common = (alpha=alpha, intercept=false, standardize=standardize)
    path = isnothing(lambda) ?
        GLMNet.glmnet(X, y; nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, kwargs_common...) :
        GLMNet.glmnet(X, y; lambda=lambda, kwargs_common...)

    βmat = Array(path.betas)               # N×L
    L    = size(βmat, 2)
    models = Vector{NamedTuple{(:beta,:sel),Tuple{Vector{Float64},Vector{Int}}}}(undef, L)
    @inbounds for j in 1:L
        βj  = @views βmat[:, j]
        sel = findall(!iszero, βj)
        models[j] = (beta=copy(βj), sel=sel)
    end
    return _GLMPath(path.lambda, models)
end

# ───────────────────────────────── GLMNet path helpers ───────────────────────
# geometric densification between λ_hi > λ_lo (both positive)
_geometric_span(λ_hi::Float64, λ_lo::Float64, nadd::Int) =
    nadd <= 0 ? Float64[] :
    exp.(range(log(λ_hi), log(λ_lo), length=nadd+2))[2:end-1]  # exclude endpoints

"""
    _glmnet_path_select(μ, Σ, T; k, ...) -> (best_idx, jstar, βj, status, βmat), path

Fit GLMNet (no intercept). Pick the **largest** (≤k) support on the path.
Returns a view `βj` on the dense beta matrix to avoid an extra copy, and the path.
"""
function _glmnet_path_select(μ, Σ, T; k,
    nlambda::Int, lambda_min_ratio::Float64,
    lambda, alpha::Float64, alpha_select::Symbol,
    nadd::Int, nnested::Int,
    standardize::Bool, epsilon::Float64, stabilize_Σ::Bool,
    compute_weights::Bool, normalize_weights::Bool,
    use_refit::Bool, do_checks::Bool,
    cv_folds::Int, cv_verbose::Bool, gcv_kappa::Float64)

    # 1) Base path
    path = _fit_glmnet_path(μ, Σ, T;
        nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, lambda=lambda,
        alpha=alpha, alpha_select=alpha_select,
        standardize=standardize, epsilon=epsilon, stabilize_Σ=stabilize_Σ,
        cv_folds=cv_folds, cv_verbose=cv_verbose, gcv_kappa=gcv_kappa)

    λgrid  = path.lambdas
    models = path.models
    nnzs   = [ _nnz_from_model(m) for m in models ]

    # If the smallest support on the supplied grid is still > k,
    # push λ upward (multiply by a factor) until we bracket ≤ k.
    extend_up_tries = 0
    while minimum(nnzs) > k && extend_up_tries < 3
        extend_up_tries += 1
        # scale λ upward (sparser) by a factor, e.g., 10×
        λ_up = unique(sort(λgrid .* 10.0; rev=true))
        path_up = _fit_glmnet_path(μ, Σ, T;
            nlambda=length(λ_up), lambda_min_ratio=lambda_min_ratio, lambda=λ_up,
            alpha=alpha, alpha_select=alpha_select,
            standardize=standardize, epsilon=epsilon, stabilize_Σ=stabilize_Σ,
            cv_folds=cv_folds, cv_verbose=cv_verbose, gcv_kappa=gcv_kappa)

        # replace grid with the enlarged, sparser set
        λgrid  = path_up.lambdas
        models = path_up.models
        nnzs   = [ _nnz_from_model(m) for m in models ]
        path   = path_up
    end

    # 2) Extend downward if needed
    extend_tries = 0
    while maximum(nnzs) < k && extend_tries < 3
        extend_tries += 1
        lambda_min_ratio /= 10
        path2 = _fit_glmnet_path(μ, Σ, T;
            nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, lambda=nothing,
            alpha=alpha, alpha_select=alpha_select,
            standardize=standardize, epsilon=epsilon, stabilize_Σ=stabilize_Σ,
            cv_folds=cv_folds, cv_verbose=cv_verbose, gcv_kappa=gcv_kappa)
        λgrid  = path2.lambdas
        models = path2.models
        nnzs   = [ _nnz_from_model(m) for m in models ]
        path   = path2
    end

    # Convenience: dense beta matrix aligned with λgrid
    βmat = hcat([m.beta for m in models]...)  # N×L

    # 3) No bracket: pick closest to k
    if all(nnzs .< k) || all(nnzs .> k)
        pick   = argmin(abs.(nnzs .- k))
        best   = models[pick]
        βj     = view(βmat, :, pick)
        status = (length(best.sel) == k) ? :LASSO_PATH_EXACT_K : :LASSO_PATH_ALMOST_K
        best_tuple = (best.sel, pick, copy(βj), status, βmat)
        return best_tuple, (lambda = λgrid,)
    end

    # 4) Bracket–densify rounds
    for _ in 1:max(nnested,0)
        j_lo = findlast(<(k), nnzs)   # last with s<k
        j_hi = findfirst(>=(k), nnzs) # first with s≥k
        if isnothing(j_lo) || isnothing(j_hi) || j_lo >= j_hi
            break
        end

        λ_hi = λgrid[j_lo]  # larger λ → sparser
        λ_lo = λgrid[j_hi]  # smaller λ → denser
        newλ = unique(sort(_geometric_span(λ_hi, λ_lo, nadd); rev=true))
        isempty(newλ) && break

        # Fit exactly at new λ’s (fixed α)
        path_new = _fit_glmnet_path(μ, Σ, T;
            nlambda=length(newλ), lambda_min_ratio=lambda_min_ratio, lambda=newλ,
            alpha=alpha, alpha_select=:fixed,
            standardize=standardize, epsilon=epsilon, stabilize_Σ=stabilize_Σ,
            cv_folds=cv_folds, cv_verbose=cv_verbose, gcv_kappa=gcv_kappa)

        # Merge and sort by λ (desc)
        λgrid  = vcat(λgrid,  path_new.lambdas)
        models = vcat(models, path_new.models)
        p = sortperm(λgrid; rev=true)
        λgrid  = λgrid[p]
        models = models[p]
        nnzs   = [ _nnz_from_model(m) for m in models ]
        βmat   = hcat([m.beta for m in models]...)

        j_eq = findfirst(==(k), nnzs)
        !isnothing(j_eq) && break
    end

    # 5) Select largest ≤k; fallback to closest overall
    idxs = findall(≤(k), nnzs)
    pick = isempty(idxs) ? argmin(abs.(nnzs .- k)) : idxs[argmax(nnzs[idxs])]
    best   = models[pick]
    βj     = view(βmat, :, pick)
    status = (length(best.sel) == k) ? :LASSO_PATH_EXACT_K : :LASSO_PATH_ALMOST_K
    best_tuple = (best.sel, pick, copy(βj), status, βmat)
    return best_tuple, (lambda = λgrid,)
end

"""
    _select_lambda_by_target_k_strict(βmat, k) -> (sel, jstar)
"""
function _select_lambda_by_target_k_strict(βmat::AbstractMatrix, k::Int)
    L = size(βmat, 2)
    best_len = -1
    best_idx = Int[]
    jstar    = 0
    @inbounds for j in L:-1:1
        col = view(βmat, :, j)
        s = count(!iszero, col)
        if s ≤ k && s > best_len
            best_idx = findall(!iszero, col)
            best_len = s
            jstar    = j
            s == k && break
        end
    end
    if best_len < 0
        return Int[], 0
    else
        return best_idx, jstar
    end
end

# ───────────────────────── Residuals, df, and GCV score ──────────────────────

@inline function _rss_at_path_column(X::AbstractMatrix, y::AbstractVector, βj::AbstractVector)
    r = y .- X * βj
    return sum(abs2, r)
end

function _df_ridge_only(X::AbstractMatrix, sel::Vector{Int}, λ::Real, α::Real)
    if isempty(sel)
        return 0.0
    end
    XA = view(X, :, sel)
    SA = XA' * XA
    λ2 = λ * max(1.0 - α, 0.0)
    if λ2 <= 0
        return 0.0
    end
    nA = size(SA, 1)
    M = SA + λ2 * I
    F = _safe_chol(M; base_bump=eps(Float64)).U
    df = 0.0
    I_k = Matrix{Float64}(I, nA, nA)
    for i in 1:nA
        x = F \ (F' \ I_k[:, i])   # x = M^{-1} e_i
        df += dot(view(SA, :, i), x)
    end
    return df
end

@inline function _gcv_log(RSS::Real, df::Real, T::Int; kappa::Real=1.0)
    denom = 1.0 - df / (kappa * T)
    if !(denom > 0)
        return Inf
    end
    return log(max(RSS, eps(Float64))) - 2.0*log(denom) - log(T)
end

function _select_alpha_by_gcv(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    αgrid::AbstractVector{<:Real},
    k::Integer;
    nlambda::Int,
    lambda_min_ratio::Real,
    lambda_override,
    standardize::Bool,
    kappa::Real
)
    best = (crit = Inf, α = αgrid[1], j = 0,
            sel = Int[], βj = zeros(Float64, size(X,2)), λ = 0.0)
    any_feasible = false

    for α in αgrid
        lambda = _validate_lambda!(lambda_override)
        dfmax = k + 5
        pmax  = 2k
        kwargs_common = (alpha=α, intercept=false, standardize=standardize,
                         dfmax=dfmax, pmax=pmax)
        path = isnothing(lambda) ?
            GLMNet.glmnet(X, y; nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, kwargs_common...) :
            GLMNet.glmnet(X, y; lambda=lambda, kwargs_common...)

        βmat = Array(path.betas) # N × L
        sel, j = _select_lambda_by_target_k_strict(βmat, k)
        if isempty(sel)
            @warn "GCV: no λ achieves |A| ≤ k for α=$(α); skipping this α"
            continue
        end
        any_feasible = true

        βj  = view(βmat, :, j)
        RSS = _rss_at_path_column(X, y, βj)
        λ   = path.lambda[j]
        df  = _df_ridge_only(X, sel, λ, α)
        crit = _gcv_log(RSS, df, size(X,1); kappa=kappa)

        if crit < best.crit
            best = (crit = crit, α = α, j = j, sel = sel,
                    βj = copy(βj), λ = λ)
        end
    end

    if !any_feasible
        return αgrid[1], 0, Int[], zeros(Float64, size(X,2)), NaN, :LASSO_GCV_INFEASIBLE
    end
    return best.α, best.j, best.sel, best.βj, best.λ, :LASSO_ALPHA_GCV
end

# ───────────────────────────── Finalization helper ───────────────────────────

@inline function _mk_status(is_empty::Bool, used_cv::Bool, path_status::Symbol, use_refit::Bool)
    return is_empty ? (use_refit ? path_status : :LASSO_ALLEMPTY) :
           (used_cv  ? :LASSO_ALPHA_CV : path_status)
end

function _finalize_from_selection!(
    N::Int, selection::Vector{Int},
    μ::AbstractVector{<:Real}, Σs::Union{AbstractMatrix, Symmetric},
    βj::AbstractVector{<:Real}, normalize_weights::Bool, compute_weights::Bool,
    epsilon::Real, use_refit::Bool, used_cv::Bool, path_status::Symbol
)
    if isempty(selection)
        status_out = _mk_status(true, used_cv, path_status, use_refit)
        return (Int[], zeros(Float64, N), 0.0, status_out)
    end

    if use_refit
        sr = SharpeRatio.compute_mve_sr(μ, Σs; selection=selection,
                                        epsilon=epsilon, stabilize_Σ=false, do_checks=false)
        w  = compute_weights ?
             SharpeRatio.compute_mve_weights(μ, Σs; selection=selection,
                                             normalize_weights=normalize_weights,
                                             epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
             zeros(Float64, N)
        status_out = _mk_status(false, used_cv, path_status, use_refit)
        return (sort(selection), w, sr, status_out)
    else
        w = zeros(Float64, N)
        w, all_empty = _lasso_weights_from_beta!(w, βj, selection; normalize_weights=normalize_weights)
        status_out = all_empty ? :LASSO_ALLEMPTY : _mk_status(false, used_cv, path_status, use_refit)
        sr = if all_empty
            0.0
        else
            SharpeRatio.compute_sr(w, μ, Σs; stabilize_Σ=false, do_checks=false)
        end
        return (sort(selection), w, sr, status_out)
    end
end

# ───────────────────────────── Lambda densification near k ───────────────────

"""
    _densify_lambda_exact_k(X, y, path, βmat, jstar; k, alpha, standardize) 
        -> (best_idx, βj, status, used)

If the current best support has s<k, find the next column with s>k, create
a geometric grid of 20 lambdas between the two bracketing λ's, refit only
those, and try again. Returns `used=false` if no valid bracket exists.
"""
function _densify_lambda_exact_k(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    path,
    βmat::AbstractMatrix{<:Real},
    jstar::Int;
    k::Integer,
    alpha::Real,
    standardize::Bool
)

    if jstar <= 0 || size(βmat, 2) == 0
        N = size(βmat, 1) > 0 ? size(βmat, 1) : size(X, 2)
        return (Int[], zeros(Float64, N), :LASSO_PATH_ALMOST_K, false)
    end

    N, L = size(βmat)
    s_j = count(!iszero, view(βmat, :, jstar))
    s_j < k || return (Int[], zeros(Float64, N), :LASSO_PATH_EXACT_K, false)

    # Find the first column after jstar (smaller λ) where support exceeds k
    jhi = 0
    @inbounds for j in (jstar+1):L
        s = count(!iszero, view(βmat, :, j))
        if s > k
            jhi = j
            break
        end
    end
    if jhi == 0
        return (Int[], zeros(Float64, N), :LASSO_PATH_ALMOST_K, false)
    end

    λ_lo = path.lambda[jstar]   # larger λ (s ≤ k)
    λ_hi = path.lambda[jhi]     # smaller λ (s > k)
    if !(isfinite(λ_lo) && isfinite(λ_hi)) || !(λ_hi < λ_lo)
        return (Int[], zeros(Float64, N), :LASSO_PATH_ALMOST_K, false)
    end

    nadd = 20
    geom = exp.(range(log(λ_lo), log(λ_hi), length=nadd+2))
    λ_new = reverse(geom[2:end-1])  # decreasing order

    λ_vec = vcat(λ_lo, λ_new, λ_hi)

    dfmax = k + 5
    pmax  = 2k
    kwargs_common = (alpha=alpha, intercept=false, standardize=standardize,
                     dfmax=dfmax, pmax=pmax)
    new_path = GLMNet.glmnet(X, y; lambda=λ_vec, kwargs_common...)
    βnew = Array(new_path.betas)

    sel, j = _select_lambda_by_target_k_strict(βnew, k)
    if isempty(sel)
        return (Int[], zeros(Float64, N), :LASSO_PATH_ALMOST_K, true)
    end
    βj = view(βnew, :, j)
    status = (length(sel) == k) ? :LASSO_PATH_EXACT_K : :LASSO_PATH_ALMOST_K
    return (sel, copy(βj), status, true)
end

# Moments-based overload used by the public API
function _densify_lambda_exact_k(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    T::Integer,
    path,                       # expects path.lambda
    βmat::AbstractMatrix{<:Real},
    jstar::Int;
    k::Integer,
    alpha::Real,
    standardize::Bool
)
    X, y, _ = _design_from_moments(μ, Σ, T; epsilon=Utils.EPS_RIDGE, stabilize_Σ=true)
    return _densify_lambda_exact_k(X, y, path, βmat, jstar; k=k, alpha=alpha, standardize=standardize)
end

# ───────────────────────────── Misc utilities ────────────────────────────────

"""
    _lasso_weights_from_beta!(w, βj, sel; normalize_weights, tol=1e-6)
        -> (w, is_all_empty::Bool)
"""
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
    if normalize_weights
        b = βj[sel]
        if all(iszero, b)
            return w, true
        end
        b = Utils.normalize_weights(b; mode=:relative, tol=tol, do_checks=false)
        @inbounds w[sel] .= b
    else
        @inbounds w[sel] .= βj[sel]
        if all(iszero, view(w,sel))
            fill!(w, 0.0)
            return w, true
        end
    end
    return w, false
end

"""
    _rolling_folds(T, K; min_val=2) -> Vector{Tuple{UnitRange,UnitRange}}
"""
function _rolling_folds(T::Int, K::Int; min_val::Int=2)
    K ≥ 1 || error("cv_folds must be ≥ 1")
    min_val ≥ 1 || error("min_val must be ≥ 1")
    seg = fld(T, K + 1)
    seg ≥ min_val || error("Not enough observations for cv_folds=$(K). Need ≥ $(min_val) per validation slice.")
    cuts = collect(seg:seg:seg*(K+1))
    cuts[end] = T
    folds = Tuple{UnitRange{Int},UnitRange{Int}}[]
    for f in 1:K
        train_end = cuts[f]; val_end = cuts[f+1]
        train = 1:train_end
        val   = (train_end+1):val_end
        length(val) ≥ min_val || error("Validation slice too short at fold $(f).")
        push!(folds, (train, val))
    end
    return folds
end

"""
    _design_from_moments(μ, Σ, T; epsilon, stabilize_Σ)
        -> (X, y, Σs)
"""
@inline function _design_from_moments(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    T::Integer;
    epsilon::Real,
    stabilize_Σ::Bool
)
    N  = length(μ)
    Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)
    Q  = T .* (Matrix(Σs) .+ μ * μ')
    μQ = max(mean(diag(Q)), eps(Float64))
    τ  = eps(Float64) * μQ
    @inbounds for i in 1:N
        Q[i,i] += τ
    end
    U = _safe_chol(Q; base_bump=eps(Float64)).U
    X = transpose(U)
    y = U \ (T .* μ)
    return X, y, Σs
end

end # module
