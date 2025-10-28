module LassoRelaxationSearch

using LinearAlgebra
using Statistics
using GLMNet
using ..Utils
using ..SharpeRatio

export mve_lasso_relaxation_search

# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

"""
    _safe_chol(Q; base_bump=1e-10, max_tries=8) -> Cholesky

Robust Cholesky with escalating diagonal bump. Tries
`Q + τ I, Q + 2τ I, Q + 4τ I, …` up to `max_tries`.
"""
function _safe_chol(Q::AbstractMatrix; base_bump=1e-10, max_tries=8)
    τ = base_bump * (tr(Q) / size(Q, 1))
    for t in 0:max_tries
        try
            return cholesky(Symmetric(Q + (τ * 2.0^t) * I))
        catch
            # escalate bump and retry
        end
    end
    error("Cholesky failed after $(max_tries + 1) bumps.")
end

"""
    _glmnet_path_select(X, y; k, nlambda=100, lambda_min_ratio=1e-3,
                        lambda=nothing, alpha=0.95, standardize=false,
                        do_checks=false)
        -> (best_idx, jstar, βj, status, βmat)

Fit GLMNet on `(X, y)` (no intercept), then pick the **largest** λ-path model
whose support size `s` satisfies `s ≤ k`. Returns:
- `best_idx::Vector{Int}` — selected indices
- `jstar::Int`            — column index on the path
- `βj::AbstractVector`    — coefficients at `jstar` (as a **view**)
- `status::Symbol`        — `:LASSO_PATH_EXACT_K` or `:LASSO_PATH_ALMOST_K`
- `βmat::Matrix{Float64}` — dense copy of GLMNet betas (N × L)
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

    # λ policy
    if lambda !== nothing
        all(isfinite, lambda) || error("`lambda` contains non-finite values.")
        all(lambda .> 0)      || error("`lambda` must be strictly positive.")
        issorted(lambda; rev=true) || error("`lambda` must be non-increasing.")
        lambda = unique(lambda)
        !isempty(lambda) || error("`lambda` is empty after removing duplicates.")
    end

    # Hard caps to accelerate: GLMNet halts when df exceeds dfmax
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
        col = @view βmat[:, j]
        s = count(!iszero, col) # fast count, no alloc
        if s ≤ k && s > best_len
            best_idx = findall(!iszero, col)  # allocate only for current winner
            best_len = s
            jstar    = j
            s == k && break
        end
    end

    if best_len < 0
        return (Int[], 0, zeros(Float64, size(X, 2)), :LASSO_PATH_ALMOST_K, βmat)
    end

    status = (best_len == k) ? :LASSO_PATH_EXACT_K : :LASSO_PATH_ALMOST_K
    βj = @view βmat[:, jstar]  # no copy
    return (best_idx, jstar, βj, status, βmat)
end

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
        b_norm = Utils.normalize_weights(b; mode=:relative, tol=tol, do_checks=false)
        @inbounds w[sel] .= b_norm
    else
        @inbounds w[sel] .= βj[sel]
        if all(iszero, @view w[sel])
            fill!(w, 0.0)
            return w, true
        end
    end
    return w, false
end

"""
    _rolling_folds(T, K; min_val=2) -> Vector{Tuple{UnitRange,UnitRange}}

Time-respecting K-folds: split `1:T` into `K+1` equal-ish blocks; for fold `f`,
`train = 1:cut[f]`, `valid = (cut[f]+1):cut[f+1]`. Enforces each validation
slice to have at least `min_val` observations.
"""
function _rolling_folds(T::Int, K::Int; min_val::Int=2)
    K ≥ 1 || error("cv_folds must be ≥ 1")
    min_val ≥ 1 || error("min_val must be ≥ 1")

    seg = fld(T, K + 1)  # floor(T/(K+1))
    seg ≥ min_val || error("Not enough observations for the requested cv_folds: T=$(T), folds=$(K), need ≥ $(min_val) per validation fold.")

    cuts = collect(seg:seg:seg*(K+1))
    cuts[end] = T

    folds = Tuple{UnitRange{Int},UnitRange{Int}}[]
    for f in 1:K
        train_end = cuts[f]
        val_end   = cuts[f+1]
        train = 1:train_end
        val   = (train_end+1):val_end
        length(val) ≥ min_val || error("Validation slice too short (fold=$(f)): got $(length(val)), need ≥ $(min_val).")
        push!(folds, (train, val))
    end
    return folds
end

"""
    _design_from_moments(μ, Σ, T; epsilon, stabilize_Σ)
        -> (X, y, Σs)

Build synthetic design for the moment-only path:
`Q = T(Σₛ + μμᵀ)`, `U'U = Q`, `X = Uᵀ`, `y = U \\ (Tμ)`.
Also returns the stabilized `Σs` for downstream SR/weights.
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

# Small status helper (unified construction)
@inline function _mk_status(is_empty::Bool, used_cv::Bool, path_status::Symbol, use_refit::Bool)
    return is_empty ? (use_refit ? path_status : :LASSO_ALLEMPTY) :
           (used_cv  ? :LASSO_ALPHA_CV : path_status)
end

"""
    _fit_once_for_alpha(R, y; k, alpha, nlambda, lambda_min_ratio, lambda,
                        standardize, epsilon, stabilize_Σ,
                        compute_weights, normalize_weights, use_refit, do_checks)
        -> (selection, weights, status)

Train-time fit on `(R, y)` for a given `alpha`, returning full-length weights
compatible with the public API semantics.
"""
function _fit_once_for_alpha(
    R::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real};
    k::Integer,
    alpha::Real,
    nlambda::Int, lambda_min_ratio::Real, lambda,
    standardize::Bool, epsilon::Real, stabilize_Σ::Bool,
    compute_weights::Bool, normalize_weights::Bool,
    use_refit::Bool, do_checks::Bool
)
    best_idx, jstar, βj, path_status, _ = _glmnet_path_select(
        R, y; k, nlambda, lambda_min_ratio, lambda, alpha, standardize, do_checks
    )

    _, N = size(R)
    if isempty(best_idx)
         return (selection=Int[], weights=zeros(Float64, N),
                 status = use_refit ? path_status : :LASSO_ALLEMPTY)
    end

    @views μ  = vec(mean(R; dims=1))
    @views Σ  = cov(R; corrected=true)
    Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)

    if use_refit
        w = compute_weights ?
            SharpeRatio.compute_mve_weights(μ, Σs; selection=best_idx,
                                            normalize_weights=normalize_weights,
                                            epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
            zeros(Float64, N)
        return (selection=sort(best_idx), weights=w, status=path_status)
    else
        w = zeros(Float64, N)
        w, all_empty = _lasso_weights_from_beta!(w, βj, best_idx; normalize_weights=normalize_weights)
        return (selection=sort(best_idx),
                weights = all_empty ? zeros(Float64, N) : w,
                status  = all_empty ? :LASSO_ALLEMPTY : path_status)
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
        cv_verbose::Bool = false
    ) -> NamedTuple{(:selection, :weights, :sr, :status, :alpha)}

Path-based **elastic net (LASSO) relaxation** of the mean–variance efficient (MVE)
support selection: regress target `y` on returns matrix `R` (no intercept),
fit a GLMNet path, and choose the largest λ-model with support size `≤ k`.

- If `alpha::Real` (default): **no CV**; behavior unchanged vs legacy.
- If `alpha::Vector`: perform **forward-rolling OOS CV** over the provided `alpha` grid
  using **validation Sharpe ratio**; pick `α*` that maximizes mean OOS SR, then refit
  on the full sample with `α*`. Status is tagged `:LASSO_ALPHA_CV`.

Two evaluation modes:

- **Refit (`use_refit=true`)**: compute exact MVE Sharpe (and optional exact weights) **on the selected support**.
- **Vanilla (`use_refit=false`)**: use raw LASSO coefficients (optionally normalized by `Utils.normalize_weights(mode=:relative)`).

Normalization does **not** affect SR (scale-invariance), but improves numerical stability.

## Arguments (selected)
- `R`: T×N matrix of excess returns.
- `k`: maximum support size.
- `alpha`: scalar (no CV) or vector (grid CV).
- `cv_folds`: number of forward folds for OOS CV when `alpha` is a grid.
- `epsilon`, `stabilize_Σ`: ridge + symmetrization for covariance usage.

## Returns
Named tuple with fields:
- `selection::Vector{Int}` — indices of selected assets.
- `weights::Vector{Float64}` — full-length weights (zeros off support).
- `sr::Float64` — in-sample SR evaluated on full sample.
- `status::Symbol` — `:LASSO_PATH_EXACT_K`, `:LASSO_PATH_ALMOST_K`, `:LASSO_ALLEMPTY`, or `:LASSO_ALPHA_CV`.
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
    cv_verbose::Bool = false
) :: NamedTuple{(:selection, :weights, :sr, :status, :alpha)}
    T, N = size(R)
    is_grid = alpha isa AbstractVector
    alpha_used = is_grid ? first(unique(Float64.(alpha))) : float(alpha)

    # ── checks ────────────────────────────────────────────────────────────────
    if do_checks
        T > 1 || error("R must have at least 2 rows.")
        N > 0 || error("R must have at least 1 column.")
        (0 ≤ k ≤ N) || error("k must be between 0 and N.")  # allow k=0

        if y !== nothing
            length(y) == T || error("Length of y must equal number of rows of R.")
            all(isfinite, y) || error("y contains non-finite values.")
        end

        if lambda !== nothing
            all(isfinite, lambda) || error("`lambda` contains non-finite values.")
            all(lambda .> 0)      || error("`lambda` must be strictly positive.")
            issorted(lambda; rev=true) || error("`lambda` must be non-increasing.")
            lambda = unique(lambda)
            !isempty(lambda) || error("`lambda` is empty after removing duplicates.")
        end

        isfinite(epsilon) || error("epsilon must be finite.")

        if is_grid
            agrid = unique(Float64.(alpha))
            all(isfinite, agrid) || error("alpha grid contains non-finite values.")
            (all(agrid .≥ 0.0) && all(agrid .≤ 1.0)) || error("alpha grid must be in [0,1].")
            length(agrid) ≥ 2 || error("Provide ≥ 2 alpha values to cross-validate.")
        else
            (0.0 ≤ float(alpha) ≤ 1.0) || error("alpha must be in [0,1].")
        end
    end

    # ── trivial fast paths ────────────────────────────────────────────────────
    if k == 0
        return (selection=Int[], weights=zeros(Float64, N), sr=0.0, status=:LASSO_ALLEMPTY, alpha=alpha_used)
    elseif k == N
        @views μ  = vec(mean(R; dims=1))
        @views Σ  = cov(R; corrected=true)
        Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)
        sr = SharpeRatio.compute_mve_sr(μ, Σs; stabilize_Σ=false)
        w  = compute_weights ?
             SharpeRatio.compute_mve_weights(μ, Σs; normalize_weights=normalize_weights, stabilize_Σ=false) :
             zeros(Float64, N)
        return (selection=collect(1:N), weights=w, sr=sr, status=:LASSO_PATH_EXACT_K, alpha=alpha_used)
    end

    # ── CV over alpha grid (if provided) ──────────────────────────────────────
    used_cv = false
    if is_grid
        agrid = unique(Float64.(alpha))  # validated above if do_checks=true
        # quick feasibility guard; also enforced by _rolling_folds
        min_val = 2
        seg = fld(T, cv_folds + 1)
        seg ≥ min_val || error("Not enough observations for cv_folds=$(cv_folds). Need ≥ $(min_val) per validation fold.")

        folds = _rolling_folds(T, cv_folds; min_val=min_val)
        best_mean_oos = -Inf
        best_alpha    = first(agrid)
        printer = cv_verbose ? println : (_...)->nothing

        for αg in agrid
            ooss = 0.0; cnt = 0
            for (tr, va) in folds
                @views Rtr = R[tr, :]
                yy = isnothing(y) ? ones(Float64, length(tr)) : @views y[tr]
                fit = _fit_once_for_alpha(Rtr, yy;
                    k=k, alpha=αg, nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, lambda=lambda,
                    standardize=standardize, epsilon=epsilon, stabilize_Σ=stabilize_Σ,
                    compute_weights=true, normalize_weights=normalize_weights,
                    use_refit=use_refit, do_checks=false)

                @views Rva = R[va, :]
                @views μv  = vec(mean(Rva; dims=1))
                @views Σv  = cov(Rva; corrected=true)
                Σvs = Utils._prep_S(Σv, epsilon, stabilize_Σ)
                sr_va = SharpeRatio.compute_sr(fit.weights, μv, Σvs; stabilize_Σ=false, do_checks=false)
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

        alpha  = best_alpha
        used_cv = true
        alpha_used = float(alpha)
    else
        alpha = float(alpha) # ensure scalar
    end

    # ── main fit on full sample with scalar alpha ─────────────────────────────
    yy = isnothing(y) ? ones(Float64, T) : y
    do_checks && (length(yy) == T || error("Length of y must equal number of rows of R."))

    best_idx, jstar, βj, path_status, _ = _glmnet_path_select(
        R, yy; k, nlambda, lambda_min_ratio, lambda, alpha=alpha, standardize, do_checks
    )

    if isempty(best_idx)
        status_out = _mk_status(true, used_cv, path_status, use_refit)
        return (selection=Int[], weights=zeros(Float64, N), sr=0.0, status=status_out, alpha=alpha_used)
    end

    @views μ  = vec(mean(R; dims=1))
    @views Σ  = cov(R; corrected=true)
    Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)

    if use_refit
        sr = SharpeRatio.compute_mve_sr(μ, Σs; selection=best_idx,
                                        epsilon=epsilon, stabilize_Σ=false, do_checks=false)
        w  = compute_weights ?
             SharpeRatio.compute_mve_weights(μ, Σs; selection=best_idx,
                                             normalize_weights=normalize_weights,
                                             epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
             zeros(Float64, N)
        status_out = _mk_status(false, used_cv, path_status, use_refit)
        return (selection=sort(best_idx), weights=w, sr=sr, status=status_out, alpha=alpha_used)
    else
        w = zeros(Float64, N)
        w, all_empty = _lasso_weights_from_beta!(w, βj, best_idx; normalize_weights=normalize_weights)
        status_out = all_empty ? :LASSO_ALLEMPTY : _mk_status(false, used_cv, path_status, use_refit)
        sr = all_empty ? 0.0 : SharpeRatio.compute_sr(w, μ, Σs; stabilize_Σ=false, do_checks=false)
        return (selection=sort(best_idx), weights=w, sr=sr, status=status_out, alpha=alpha_used)
    end
end

# ═════════════════════════════════════════════════════════════════════════════
# Public API — Moment-based entrypoint (optionally with returns for CV)
# ═════════════════════════════════════════════════════════════════════════════

"""
# mve_lasso_relaxation_search — moment-only (with optional R for α-CV)

    mve_lasso_relaxation_search(μ::AbstractVector{<:Real},
                                Σ::AbstractMatrix{<:Real},
                                T::Integer;
        R::Union{Nothing,AbstractMatrix{<:Real}} = nothing,   # OPTIONAL: enables α-grid CV
        k::Integer,
        nlambda::Int = 100,
        lambda_min_ratio::Real = 1e-3,
        lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
        alpha::Union{Real,AbstractVector{<:Real}} = 0.95,      # scalar → legacy; vector → CV (requires R)
        standardize::Bool = false,
        epsilon::Real = Utils.EPS_RIDGE,
        stabilize_Σ::Bool = true,
        compute_weights::Bool = false,
        normalize_weights::Bool = false,
        use_refit::Bool = true,
        do_checks::Bool = false,
        cv_folds::Int = 5,
        cv_verbose::Bool = false
    ) -> NamedTuple{(:selection, :weights, :sr, :status, :alpha)}

Moment-only variant. Constructs a synthetic design from `(μ, Σ, T)`:
`Q = T(Σₛ + μμᵀ)`, `X = Uᵀ`, `y = U \\ (Tμ)` where `U'U = Q`.
If `alpha` is a **grid** and `R` is provided, performs **forward-rolling OOS CV** on the
grid (using OOS SR) to select `α*`, then refits on `(μ, Σ, T)` with `α*`.

If `alpha` is **scalar** (default), behavior matches the legacy moment-only method.
"""
function mve_lasso_relaxation_search(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    T::Integer;
    R::Union{Nothing,AbstractMatrix{<:Real}} = nothing,  # OPTIONAL
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
    cv_verbose::Bool = false
) :: NamedTuple{(:selection, :weights, :sr, :status, :alpha)}
    N = length(μ)
    is_grid = alpha isa AbstractVector
    alpha_used = is_grid ? first(unique(Float64.(alpha))) : float(alpha)

    # ── checks ────────────────────────────────────────────────────────────────
    if do_checks
        N > 0 || error("μ must be non-empty.")
        size(Σ) == (N, N) || error("Σ must be N×N.")
        (1 ≤ k ≤ N) || error("k must be between 1 and N.")
        T ≥ 1 || error("T must be a positive integer.")
        all(isfinite, μ) && all(isfinite, Σ) || error("Non-finite entries in μ or Σ.")

        if lambda !== nothing
            all(isfinite, lambda) || error("`lambda` contains non-finite values.")
            all(lambda .> 0)      || error("`lambda` must be strictly positive.")
            issorted(lambda; rev=true) || error("`lambda` must be non-increasing.")
            lambda = unique(lambda)
            !isempty(lambda) || error("`lambda` is empty after removing duplicates.")
        end

        isfinite(epsilon) || error("epsilon must be finite.")

        if is_grid
            agrid = unique(Float64.(alpha))
            all(isfinite, agrid) || error("alpha grid contains non-finite values.")
            (all(agrid .≥ 0.0) && all(agrid .≤ 1.0)) || error("alpha grid must be in [0,1].")
            length(agrid) ≥ 2 || error("Provide ≥ 2 alpha values to cross-validate.")
            R !== nothing || error("alpha grid CV requires raw returns `R` for time-based validation.")
        else
            (0.0 ≤ float(alpha) ≤ 1.0) || error("alpha must be in [0,1].")
        end

        if R !== nothing
            size(R, 2) == N || error("R must have N columns (same as length(μ)).")
            size(R, 1) ≥ 2  || error("R must have at least 2 rows.")
            all(isfinite, R) || error("Non-finite entries in R.")
        end
    end

    # ── choose α by CV if grid + R provided ──────────────────────────────────
    used_cv = false
    if is_grid
        agrid = unique(Float64.(alpha))  # validated above if do_checks=true
        T_R = size(R, 1)
        min_val = 2
        seg = fld(T_R, cv_folds + 1)
        seg ≥ min_val || error("Not enough observations for cv_folds=$(cv_folds). Need ≥ $(min_val) per validation fold.")
        folds = _rolling_folds(T_R, cv_folds; min_val=min_val)

        best_mean_oos = -Inf
        best_alpha    = first(agrid)
        printer = cv_verbose ? println : (_...)->nothing

        for αg in agrid
            ooss = 0.0; cnt = 0
            for (tr, va) in folds
                @views Rtr = R[tr, :]
                @views μtr = vec(mean(Rtr; dims=1))
                @views Σtr = cov(Rtr; corrected=true)
                Ttr = length(tr)

                # moment-only train fit at αg → weights on train support policy
                Xtr, ytr, Σtr_s = _design_from_moments(μtr, Σtr, Ttr; epsilon=epsilon, stabilize_Σ=stabilize_Σ)
                best_idx, jstar, βj, path_status, _ = _glmnet_path_select(
                    Xtr, ytr; k=k, nlambda=nlambda, lambda_min_ratio=lambda_min_ratio,
                    lambda=lambda, alpha=αg, standardize=standardize, do_checks=false
                )

                # build weights for OOS scoring
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

                @views Rva = R[va, :]
                @views μva = vec(mean(Rva; dims=1))
                @views Σva = cov(Rva; corrected=true)
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

        alpha  = best_alpha
        used_cv = true
        alpha_used = float(alpha)
    else
        alpha = float(alpha) # ensure scalar
    end

    # ── final fit on the PROVIDED (μ, Σ, T) with scalar alpha ────────────────
    X, y, Σs = _design_from_moments(μ, Σ, T; epsilon=epsilon, stabilize_Σ=stabilize_Σ)
    best_idx, jstar, βj, path_status, _ = _glmnet_path_select(
        X, y; k, nlambda, lambda_min_ratio, lambda, alpha=alpha, standardize, do_checks
    )

    if isempty(best_idx)
        status_out = _mk_status(true, used_cv, path_status, use_refit)
        return (selection=Int[], weights=zeros(Float64, N), sr=0.0, status=status_out, alpha=alpha_used)
    end

    if use_refit
        sr = SharpeRatio.compute_mve_sr(μ, Σs; selection=best_idx,
                                        epsilon=epsilon, stabilize_Σ=false, do_checks=false)
        w  = compute_weights ?
             SharpeRatio.compute_mve_weights(μ, Σs; selection=best_idx,
                                             normalize_weights=normalize_weights,
                                             epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
             zeros(Float64, N)
        status_out = _mk_status(false, used_cv, path_status, use_refit)
        return (selection=sort(best_idx), weights=w, sr=sr, status=status_out, alpha=alpha_used)
    else
        w = zeros(Float64, N)
        w, all_empty = _lasso_weights_from_beta!(w, βj, best_idx; normalize_weights=normalize_weights)
        status_out = all_empty ? :LASSO_ALLEMPTY : _mk_status(false, used_cv, path_status, use_refit)
        sr = all_empty ? 0.0 : SharpeRatio.compute_sr(w, μ, Σs; stabilize_Σ=false, do_checks=false)
        return (selection=sort(best_idx), weights=w, sr=sr, status=status_out, alpha=alpha_used)
    end
end

end # module
