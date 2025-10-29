module LassoRelaxationSearch

using LinearAlgebra
using Statistics
using GLMNet
using ..Utils
using ..SharpeRatio

export mve_lasso_relaxation_search

# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers — clarity, robustness, and shared logic
# ═════════════════════════════════════════════════════════════════════════════

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

Centralizes argument checks for the R-based public API when `do_checks=true`.
Keeps legacy semantics: allow k ∈ [0, N]. Also validates α-grid requirements for GCV.
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
            # OOS-CV mode (status :LASSO_ALPHA_CV)
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

Centralizes argument checks for the moment-based public API when `do_checks=true`.
Keeps legacy semantics: require k ∈ [1, N]. Also validates α-grid requirements for GCV.
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
            # basic feasibility of folds (on R)
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

# ───────────────────────────────── GLMNet path helpers ───────────────────────

"""
    _glmnet_path_select(X, y; k, ...) -> (best_idx, jstar, βj, status, βmat), path

Fit GLMNet (no intercept). Pick the **largest** (≤k) support on the path.
Returns a view `βj` on the dense beta matrix to avoid an extra copy, and the path.
"""
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

    lambda = _validate_lambda!(lambda)

    # df/pmax caps speed up GLMNet path building
    dfmax = k
    pmax  = max(k + 5, 2k)

    kwargs_common = (alpha=alpha, intercept=false, standardize=standardize,
                     dfmax=dfmax, pmax=pmax)

    path = isnothing(lambda) ?
        GLMNet.glmnet(X, y; nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, kwargs_common...) :
        GLMNet.glmnet(X, y; lambda=lambda, kwargs_common...)

    βmat = Array(path.betas) # N × L
    L    = size(βmat, 2)

    best_len = -1
    best_idx = Int[]
    jstar    = 0

    @inbounds for j in L:-1:1
        col = view(βmat, :, j)
        s = count(!iszero, col)             # allocation-free count
        if s ≤ k && s > best_len
            best_idx = findall(!iszero, col)  # allocate only for the current winner
            best_len = s
            jstar    = j
            s == k && break
        end
    end

    if best_len < 0
        return (Int[], 0, zeros(Float64, size(X, 2)), :LASSO_PATH_ALMOST_K, βmat), path
    end

    status = (best_len == k) ? :LASSO_PATH_EXACT_K : :LASSO_PATH_ALMOST_K
    βj = view(βmat, :, jstar)  # no copy
    return (best_idx, jstar, βj, status, βmat), path
end

"""
    _select_lambda_by_target_k_strict(βmat, k) -> (sel, jstar)

Given a GLMNet beta path (columns = λ values), return the column with
support size `s ≤ k` that is *closest to k from below*. If none exists,
return `(Int[], 0)`.
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

"""
    _rss_at_path_column(X, y, βj) -> RSS

Compute residual sum of squares for a given path column.
"""
@inline function _rss_at_path_column(X::AbstractMatrix, y::AbstractVector, βj::AbstractVector)
    r = y .- X * βj
    return sum(abs2, r)
end

"""
    _df_ridge_only(X, sel, λ, α) -> df

Effective degrees of freedom for the **ridge-only** component on a *fixed*
active set A (|A| ≤ k):
    df = tr( S_A * (S_A + λ₂ I)^{-1} ),   S_A = X_A' X_A,   λ₂ = λ * (1-α).
Assumes the same scaling as the GLMNet design (use `standardize=false` to
avoid scale mismatches). For α=1 (lasso), this returns df=|A| * 0 = 0, which
is consistent with "ridge-only" complexity (selection is external).
"""
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
    # df = tr( S (S + λ2 I)^{-1} )
    nA = size(SA, 1)
    M = SA + λ2 * I
    F = _safe_chol(M; base_bump=eps(Float64)).U
    # Compute trace(SA * M^{-1}) via solves
    df = 0.0
    I_k = Matrix{Float64}(I, nA, nA)
    for i in 1:nA
        x = F \ (F' \ I_k[:, i])   # x = M^{-1} e_i
        df += dot(view(SA, :, i), x)
    end
    return df
end

"""
    _gcv_log(RSS, df, T; kappa=1.0) -> crit

Log-scale GCV criterion:
    logGCV = log(RSS) - 2*log(1 - df/(κ T)) - log(T).
Returns +Inf if `df ≥ κ T`.
"""
@inline function _gcv_log(RSS::Real, df::Real, T::Int; kappa::Real=1.0)
    denom = 1.0 - df / (kappa * T)
    if !(denom > 0)
        return Inf
    end
    return log(max(RSS, eps(Float64))) - 2.0*log(denom) - log(T)
end

"""
    _select_alpha_by_gcv(X, y, αgrid, k; nlambda, lambda_min_ratio, lambda_override,
                         standardize, kappa) -> (α_star, j_star, sel_star, βj_star, λ_star, status)

GCV selector over an α-grid. For each α:
  1) fit glmnet path (dfmax=k, pmax≈2k),
  2) pick λ by strict target (|A| ≤ k, closest below k),
  3) compute RSS, df (ridge-only), log-GCV,
  4) keep α with minimal log-GCV.

If no α yields |A|≤k, returns status = :LASSO_GCV_INFEASIBLE and a zero selection.
"""
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
        # Fit path for this α
        lambda = _validate_lambda!(lambda_override)
        dfmax = k
        pmax  = max(k + 5, 2k)
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

"""
    _finalize_from_selection!(N, selection, μ, Σs, βj, normalize_weights,
                              compute_weights, epsilon, use_refit, used_cv, path_status)
        -> (sel, w, sr, status)

One place to finalize outputs from a found support. Keeps refit/vanilla logic in sync.
"""
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

# ═════════════════════════════════════════════════════════════════════════════
# Public API — R-based entrypoint
# ═════════════════════════════════════════════════════════════════════════════

"""
# mve_lasso_relaxation_search — R-based

    mve_lasso_relaxation_search(R::AbstractMatrix{<:Real};
        k::Integer,
        y::Union{Nothing,AbstractVector{<:Real}} = nothing,
        nlambda::Int = 100,
        lambda_min_ratio::Real = 1e-3,
        lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
        alpha::Union{Real,AbstractVector{<:Real}} = 0.95,
        standardize::Bool = false,
        epsilon::Real = Utils.EPS_RIDGE,
        stabilize_Σ::Bool = true,
        compute_weights::Bool = false,
        normalize_weights::Bool = false,
        use_refit::Bool = true,
        do_checks::Bool = false,
        cv_folds::Int = 5,
        cv_verbose::Bool = false,
        # NEW (opt-in) selector for α
        alpha_select::Symbol = :fixed,   # :fixed | :oos_cv | :gcv
        gcv_kappa::Real = 1.0            # κ in log-GCV
    ) -> NamedTuple{(:selection, :weights, :sr, :status, :alpha)}

Elastic-Net relaxation of sparse MVE via GLMNet on (R, y). α can be:
- `:fixed` (default): use scalar α (legacy behavior), or if α is a vector, do OOS-CV over α-grid (status `:LASSO_ALPHA_CV`).
- `:oos_cv`: explicitly force OOS-CV on α-grid using forward-rolling folds (status `:LASSO_ALPHA_CV`).
- `:gcv`: **generalized cross-validation** over α-grid on the *design* used here; for each α, choose λ on the path such that |A| ≤ k (closest from below). If no α yields |A| ≤ k, returns zeros and status `:LASSO_GCV_INFEASIBLE`.

In GCV mode, the criterion is **log-GCV** with **ridge-only df** on the selected set.
"""
function mve_lasso_relaxation_search(
    R::AbstractMatrix{<:Real};
    k::Integer,
    y::Union{Nothing,AbstractVector{<:Real}} = nothing,
    nlambda::Int = 100,
    lambda_min_ratio::Real = 1e-3,
    lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
    alpha::Union{Real,AbstractVector{<:Real}} = 0.95,
    standardize::Bool = false,
    epsilon::Real = Utils.EPS_RIDGE,
    stabilize_Σ::Bool = true,
    compute_weights::Bool = false,
    normalize_weights::Bool = false,
    use_refit::Bool = true,
    do_checks::Bool = false,
    cv_folds::Int = 5,
    cv_verbose::Bool = false,
    alpha_select::Symbol = :fixed,   # :fixed | :oos_cv | :gcv
    gcv_kappa::Real = 1.0
) :: NamedTuple{(:selection, :weights, :sr, :status, :alpha)}
    T, N = size(R)
    lambda, is_grid, agrid, alpha_used = _validate_R_based_inputs!(R, k, y, epsilon, alpha, lambda, do_checks, cv_folds, alpha_select)

    # trivial fast paths
    if k == 0
        return (selection=Int[], weights=zeros(Float64, N), sr=0.0,
            status = (alpha_select === :gcv ? :LASSO_GCV_INFEASIBLE : :LASSO_ALLEMPTY),
            alpha=alpha_used)
    elseif k == N
        μ, Σ = _moments_from_R(R)
        Σs    = Utils._prep_S(Σ, epsilon, stabilize_Σ)
        sr    = SharpeRatio.compute_mve_sr(μ, Σs; stabilize_Σ=false)
        w     = compute_weights ? SharpeRatio.compute_mve_weights(μ, Σs; normalize_weights=normalize_weights, stabilize_Σ=false) : zeros(Float64, N)
        return (selection=collect(1:N), weights=w, sr=sr, status=:LASSO_PATH_EXACT_K, alpha=alpha_used)
    end

    # choose α according to alpha_select
    if alpha_select === :gcv
        yy = isnothing(y) ? ones(Float64, T) : y
        α_star, _, sel_star, βj_star, _, gcv_status = _select_alpha_by_gcv(
            R, yy, agrid, k; nlambda=nlambda, lambda_min_ratio=lambda_min_ratio,
            lambda_override=lambda, standardize=standardize, kappa=gcv_kappa
        )
        if gcv_status === :LASSO_GCV_INFEASIBLE
            @warn "GCV: all α were infeasible (no |A| ≤ k). Returning zero weights."
            return (selection=Int[], weights=zeros(Float64, N), sr=0.0, status=:LASSO_GCV_INFEASIBLE, alpha=agrid[1])
        end

        μ, Σ  = _moments_from_R(R)
        Σs    = Utils._prep_S(Σ, epsilon, stabilize_Σ)
        sel, w, sr, status_out = _finalize_from_selection!(
            N, sel_star, μ, Σs, βj_star, normalize_weights, compute_weights,
            epsilon, use_refit, false, :LASSO_ALPHA_GCV
        )
        return (selection=sel, weights=w, sr=sr, status=status_out, alpha=α_star)
    end

    # OOS-CV over α-grid (if requested or α is a vector and :fixed behavior)
    used_cv = false
    if is_grid && (alpha_select === :oos_cv || alpha_select === :fixed)
        min_val = 2
        folds = _rolling_folds(T, cv_folds; min_val=min_val)

        best_mean_oos = -Inf
        best_alpha    = agrid[1]
        printer = cv_verbose ? println : (_...)->nothing

        for αg in agrid
            ooss = 0.0; cnt = 0
            for (tr, va) in folds
                @views Rtr = R[tr, :]
                yy = isnothing(y) ? ones(Float64, length(tr)) : @views y[tr]

                (best_tuple, _) = _glmnet_path_select(
                    Rtr, yy; k=k, nlambda=nlambda, lambda_min_ratio=lambda_min_ratio,
                    lambda=lambda, alpha=αg, standardize=standardize, do_checks=false
                )
                best_idx, _, βj, _, _ = best_tuple

                μv, Σv = _moments_from_R(@view R[va, :])
                Σvs = Utils._prep_S(Σv, epsilon, stabilize_Σ)

                # Build fold weights (refit or vanilla) for OOS SR
                w = if isempty(best_idx)
                    zeros(Float64, N)
                elseif use_refit
                    μtr, Σtr = _moments_from_R(Rtr)
                    Σtr_s = Utils._prep_S(Σtr, epsilon, stabilize_Σ)
                    SharpeRatio.compute_mve_weights(μtr, Σtr_s; selection=best_idx,
                        normalize_weights=normalize_weights,
                        epsilon=epsilon, stabilize_Σ=false, do_checks=false)
                else
                    ww = zeros(Float64, N)
                    _lasso_weights_from_beta!(ww, βj, best_idx; normalize_weights=normalize_weights) |> first
                end

                sr_va = SharpeRatio.compute_sr(w, μv, Σvs; stabilize_Σ=false, do_checks=false)
                if isfinite(sr_va)
                    ooss += sr_va; cnt += 1
                end
            end
            mean_oos = cnt > 0 ? (ooss / cnt) : -Inf
            printer("α=$(round(αg, digits=4)) ⇒ mean OOS SR=$(round(mean_oos, digits=6)) over $(cnt) folds")
            if mean_oos > best_mean_oos
                best_mean_oos = mean_oos
                best_alpha    = αg
            end
        end
        alpha_used = float(best_alpha)
        used_cv = true
    end

    # Final path fit on full sample with scalar alpha_used
    yy = isnothing(y) ? ones(Float64, T) : y
    (best_tuple, _) = _glmnet_path_select(
        R, yy; k, nlambda, lambda_min_ratio, lambda, alpha=alpha_used, standardize, do_checks
    )
    best_idx, _, βj, path_status, _ = best_tuple

    μ, Σ  = _moments_from_R(R)
    Σs    = Utils._prep_S(Σ, epsilon, stabilize_Σ)
    
    sel, w, sr, status_out = _finalize_from_selection!(
        N, best_idx, μ, Σs, βj, normalize_weights, compute_weights,
        epsilon, use_refit, used_cv, path_status
    )
    return (selection=sel, weights=w, sr=sr, status=status_out, alpha=alpha_used)

end

# ═════════════════════════════════════════════════════════════════════════════
# Public API — Moment-based entrypoint (synthetic design; α-GCV on design)
# ═════════════════════════════════════════════════════════════════════════════

"""
# mve_lasso_relaxation_search — moment-only (with optional R for OOS α-CV)

    mve_lasso_relaxation_search(μ::AbstractVector{<:Real},
                                Σ::AbstractMatrix{<:Real},
                                T::Integer;
        R::Union{Nothing,AbstractMatrix{<:Real}} = nothing,   # OPTIONAL: enables OOS α-grid CV
        k::Integer,
        nlambda::Int = 100,
        lambda_min_ratio::Real = 1e-3,
        lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
        alpha::Union{Real,AbstractVector{<:Real}} = 0.95,
        standardize::Bool = false,
        epsilon::Real = Utils.EPS_RIDGE,
        stabilize_Σ::Bool = true,
        compute_weights::Bool = false,
        normalize_weights::Bool = false,
        use_refit::Bool = true,
        do_checks::Bool = false,
        cv_folds::Int = 5,
        cv_verbose::Bool = false,
        # NEW (opt-in) selector for α
        alpha_select::Symbol = :fixed,   # :fixed | :oos_cv | :gcv
        gcv_kappa::Real = 1.0            # κ in log-GCV
    ) -> NamedTuple{(:selection, :weights, :sr, :status, :alpha)}

Moment-only variant via synthetic design X,y from (μ, Σ, T).
- `:fixed`: scalar α (legacy). If α is a vector and `:fixed`, we do **OOS-CV** using `R` (required) and forward-rolling folds (status `:LASSO_ALPHA_CV`).
- `:oos_cv`: explicitly force OOS-CV on α-grid (requires `R`). 
- `:gcv`: **GCV** over α-grid on the *synthetic design* (X,y) built from (μ,Σ,T); for each α, choose λ on the path such that |A| ≤ k (closest from below). If no α yields |A| ≤ k, returns zeros and status `:LASSO_GCV_INFEASIBLE`.

In GCV mode, the criterion is **log-GCV** with **ridge-only df** on the selected set.
"""
function mve_lasso_relaxation_search(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    T::Integer;
    R::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    k::Integer,
    nlambda::Int = 100,
    lambda_min_ratio::Real = 1e-3,
    lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
    alpha::Union{Real,AbstractVector{<:Real}} = 0.95,
    standardize::Bool = false,
    epsilon::Real = Utils.EPS_RIDGE,
    stabilize_Σ::Bool = true,
    compute_weights::Bool = false,
    normalize_weights::Bool = false,
    use_refit::Bool = true,
    do_checks::Bool = false,
    cv_folds::Int = 5,
    cv_verbose::Bool = false,
    alpha_select::Symbol = :fixed,   # :fixed | :oos_cv | :gcv
    gcv_kappa::Real = 1.0
) :: NamedTuple{(:selection, :weights, :sr, :status, :alpha)}
    lambda, is_grid, agrid, alpha_used = _validate_moment_based_inputs!(μ, Σ, T, R, k, epsilon, alpha, lambda, do_checks, cv_folds, alpha_select)

    # α selection: GCV on synthetic design (X,y)
    if alpha_select === :gcv
        X, y, Σs = _design_from_moments(μ, Σ, T; epsilon=epsilon, stabilize_Σ=stabilize_Σ)
        α_star, j_star, sel_star, βj_star, λ_star, gcv_status = _select_alpha_by_gcv(
            X, y, agrid, k; nlambda=nlambda, lambda_min_ratio=lambda_min_ratio,
            lambda_override=lambda, standardize=standardize, kappa=gcv_kappa
        )
        if gcv_status === :LASSO_GCV_INFEASIBLE
            @warn "GCV: all α were infeasible (no |A| ≤ k). Returning zero weights."
            N = length(μ)
            return (selection=Int[], weights=zeros(Float64, N), sr=0.0, status=:LASSO_GCV_INFEASIBLE, alpha=agrid[1])
        end

        N = length(μ)
        sel, w, sr, status_out = _finalize_from_selection!(
            N, sel_star, μ, Σs, βj_star, normalize_weights, compute_weights,
            epsilon, use_refit, false, :LASSO_ALPHA_GCV
        )
        return (selection=sel, weights=w, sr=sr, status=status_out, alpha=α_star)
    end

    # OOS-CV over α-grid (if requested or α is a vector and :fixed behavior)
    used_cv = false
    if is_grid && (alpha_select === :oos_cv || alpha_select === :fixed)
        R === nothing && error("OOS α-grid CV in the moment-based API requires providing raw returns `R`.")
        T_R, N = size(R)
        min_val = 2
        folds = _rolling_folds(T_R, cv_folds; min_val=min_val)

        best_mean_oos = -Inf
        best_alpha    = agrid[1]
        printer = cv_verbose ? println : (_...)->nothing

        for αg in agrid
            ooss = 0.0; cnt = 0
            for (tr, va) in folds
                @views Rtr = R[tr, :]
                μtr, Σtr   = _moments_from_R(Rtr)
                Ttr        = length(tr)

                Xtr, ytr, Σtr_s = _design_from_moments(μtr, Σtr, Ttr; epsilon=epsilon, stabilize_Σ=stabilize_Σ)
                (best_tuple, _) = _glmnet_path_select(
                    Xtr, ytr; k=k, nlambda=nlambda, lambda_min_ratio=lambda_min_ratio,
                    lambda=lambda, alpha=αg, standardize=standardize, do_checks=false
                )
                best_idx, _, βj, _, _ = best_tuple

                # Build weights for validation SR
                w = if isempty(best_idx)
                    zeros(Float64, N)
                elseif use_refit
                    SharpeRatio.compute_mve_weights(μtr, Σtr_s; selection=best_idx,
                        normalize_weights=normalize_weights,
                        epsilon=epsilon, stabilize_Σ=false, do_checks=false)
                else
                    ww = zeros(Float64, N)
                    _lasso_weights_from_beta!(ww, βj, best_idx; normalize_weights=normalize_weights) |> first
                end

                μva, Σva = _moments_from_R(@view R[va, :])
                Σva_s = Utils._prep_S(Σva, epsilon, stabilize_Σ)
                sr_va = SharpeRatio.compute_sr(w, μva, Σva_s; stabilize_Σ=false, do_checks=false)
                if isfinite(sr_va)
                    ooss += sr_va; cnt += 1
                end
            end
            mean_oos = cnt > 0 ? (ooss / cnt) : -Inf
            printer("α=$(round(αg, digits=4)) ⇒ mean OOS SR=$(round(mean_oos, digits=6)) over $(cnt) folds")
            if mean_oos > best_mean_oos
                best_mean_oos = mean_oos
                best_alpha    = αg
            end
        end
        alpha_used = float(best_alpha)
        used_cv = true
    end

    # Final fit on (μ, Σ, T) with scalar alpha_used
    X, y, Σs = _design_from_moments(μ, Σ, T; epsilon=epsilon, stabilize_Σ=stabilize_Σ)
    (best_tuple, _) = _glmnet_path_select(
        X, y; k, nlambda, lambda_min_ratio, lambda, alpha=alpha_used, standardize, do_checks
    )
    best_idx, _, βj, path_status, _ = best_tuple

    N = length(μ)
    sel, w, sr, status_out = _finalize_from_selection!(
        N, best_idx, μ, Σs, βj, normalize_weights, compute_weights,
        epsilon, use_refit, used_cv, path_status
    )
    return (selection=sel, weights=w, sr=sr, status=status_out, alpha=alpha_used)
end

# ═════════════════════════════════════════════════════════════════════════════
# Utilities used above
# ═════════════════════════════════════════════════════════════════════════════

"""
    _lasso_weights_from_beta!(w, βj, sel; normalize_weights, tol=1e-6)
        -> (w, is_all_empty::Bool)

Populate `w` (full length) from coefficients `βj` on selection `sel`.
If `normalize_weights=true`, rescale via `Utils.normalize_weights(mode=:relative)`.
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

Forward-rolling folds: split 1:T into K+1 blocks; for fold f,
train = 1:cut[f], valid = (cut[f]+1):cut[f+1]. Each validation has ≥ min_val obs.
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

Moment-only synthetic design:
Q = T(Σₛ + μμᵀ), U'U = Q, X = Uᵀ, y = U \\ (Tμ).
Returns stabilized Σₛ for downstream Sharpe/weights use.
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
    # tiny bump (relative to diag mean) before chol
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
