module LassoRelaxationSearch

using LinearAlgebra
using Statistics
using GLMNet
using ..Utils
using ..SharpeRatio
using ..LassoRelaxationUtils

export mve_lasso_relaxation_search

# ═════════════════════════════════════════════════════════════════════════════
# Public API — R-based entrypoint
# ═════════════════════════════════════════════════════════════════════════════

"""
# mve_lasso_relaxation_search — R-based

    mve_lasso_relaxation_search(R; k, y=nothing,
        nlambda=100, lambda_min_ratio=1e-3, lambda=nothing,
        alpha=1.0,                          # scalar or grid
        nadd=80, nnested=2,                 # λ densification near target k
        standardize=false, epsilon=Utils.EPS_RIDGE, stabilize_Σ=true,
        compute_weights=true, normalize_weights=false, use_refit=false,
        do_checks=false,
        cv_folds=5, cv_verbose=false,
        alpha_select=:fixed,                # :fixed | :oos_cv | :gcv
        gcv_kappa=1.0
    ) -> (selection, weights, sr, status, alpha)

Select a LASSO portfolio with support ≈ k from a GLMNet path and return:
selection, weights (refitted or vanilla), SR (computed on moments), status, and α.

λ path behavior and densification
- GLMNet builds a geometric λ grid from λ_max (null model) down to
  λ_min = λ_max * lambda_min_ratio. On a finite grid, the exact-k knot may be skipped.
- `nadd` inserts that many geometric λ’s **between** the two path λ’s that **straddle k**
  (last with |A|<k, first with |A|≥k), refitting only those points.
- `nnested` repeats bracket → densify cycles `nnested` times to refine around k.
- If even after densification `max |A| < k`, the routine automatically **extends the path**
  by reducing `lambda_min_ratio` (a few times) and tries again.
- If no λ with |A|≤k exists, the model with |A| closest to k is returned.

Alpha selection
- `alpha_select=:fixed` uses the provided scalar α (or, if `alpha` is a vector and
  `:oos_cv` is requested, chooses α by rolling OOS Sharpe over `cv_folds`).
- `alpha_select=:oos_cv` expects a grid and selects α by rolling OOS SR on the raw `R`.
- `alpha_select=:gcv` expects a grid and uses log-GCV on a synthetic design built from moments.

Notes
- Internally, we pass **(μ, Σ, T)** to the path selector (for numerical stability and to
  align with Utils’ method signatures), and compute SR/weights from moments. We still use
  raw `R` only for α-CV when requested.

Returns `NamedTuple`:
- `selection::Vector{Int}` (sorted), `weights::Vector{Float64}`, `sr::Float64`,
  `status::Symbol`, `alpha::Float64`.
"""
function mve_lasso_relaxation_search(
    R::AbstractMatrix{<:Real};
    k::Integer,
    y::Union{Nothing,AbstractVector{<:Real}} = nothing,
    nlambda::Int = 100,
    lambda_min_ratio::Real = 1e-3,
    lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
    alpha::Union{Real,AbstractVector{<:Real}} = 1.00,
    # NEW: densification controls
    nadd::Int = 80,
    nnested::Int = 2,
    #
    standardize::Bool = false,
    epsilon::Real = Utils.EPS_RIDGE,
    stabilize_Σ::Bool = true,
    compute_weights::Bool = true,
    normalize_weights::Bool = false,
    use_refit::Bool = false,
    do_checks::Bool = false,
    cv_folds::Int = 5,
    cv_verbose::Bool = false,
    alpha_select::Symbol = :fixed,
    gcv_kappa::Real = 1.0
) :: NamedTuple{(:selection, :weights, :sr, :status, :alpha)}
    T, N = size(R)
    lambda, is_grid, agrid, alpha_used = _validate_R_based_inputs!(
        R, k, y, epsilon, alpha, lambda, do_checks, cv_folds, alpha_select
    )

    # trivial fast paths
    if k == 0
        return (selection=Int[], weights=zeros(Float64, N), sr=0.0,
                status = (alpha_select === :gcv ? :LASSO_GCV_INFEASIBLE : :LASSO_ALLEMPTY),
                alpha=alpha_used)
    elseif k == N
        μ, Σ = _moments_from_R(R)
        Σs   = Utils._prep_S(Σ, epsilon, stabilize_Σ)
        sr   = SharpeRatio.compute_mve_sr(μ, Σs; stabilize_Σ=false)
        w    = compute_weights ? SharpeRatio.compute_mve_weights(μ, Σs; normalize_weights=normalize_weights, stabilize_Σ=false) : zeros(Float64, N)
        return (selection=collect(1:N), weights=w, sr=sr, status=:LASSO_PATH_EXACT_K, alpha=alpha_used)
    end

    # α selection via GCV (uses design internally, fine)
    if alpha_select === :gcv
        μ, Σ      = _moments_from_R(R)
        X, yX, Σs = _design_from_moments(μ, Σ, T; epsilon=epsilon, stabilize_Σ=stabilize_Σ)
        α_star, _, sel_star, βj_star, _, gcv_status = _select_alpha_by_gcv(
            X, yX, agrid, k;
            nlambda=nlambda, lambda_min_ratio=lambda_min_ratio,
            lambda_override=lambda, standardize=standardize, kappa=gcv_kappa
        )
        if gcv_status === :LASSO_GCV_INFEASIBLE
            @warn "GCV: all α were infeasible (no |A| ≤ k). Returning zero weights."
            return (selection=Int[], weights=zeros(Float64, N), sr=0.0, status=:LASSO_GCV_INFEASIBLE, alpha=agrid[1])
        end
        sel, w, sr, status_out = _finalize_from_selection!(
            N, sel_star, μ, Σs, βj_star, normalize_weights, compute_weights,
            epsilon, use_refit, false, :LASSO_ALPHA_GCV
        )
        return (selection=sel, weights=w, sr=sr, status=status_out, alpha=α_star)
    end

    # α-grid OOS CV (if requested) — path selection uses moments
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
                μtr, Σtr   = _moments_from_R(Rtr)
                Ttr        = length(tr)
                (best_tuple, _) = _glmnet_path_select(
                    μtr, Σtr, Ttr;
                    k=k,
                    nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, lambda=lambda,
                    alpha=αg, alpha_select=:fixed,            
                    nadd=nadd, nnested=nnested,
                    standardize=standardize,
                    epsilon=epsilon, stabilize_Σ=stabilize_Σ, 
                    compute_weights=false, normalize_weights=false, use_refit=false, 
                    do_checks=false,
                    cv_folds=cv_folds, cv_verbose=false,       
                    gcv_kappa=1.0                              
                )
                best_idx, jstar, βj, path_status, βmat = best_tuple

                # weights for validation SR
                Σtr_s = Utils._prep_S(Σtr, epsilon, stabilize_Σ)
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

    # Final path fit on full sample — pass (μ, Σ, T) to selector
    μ, Σ = _moments_from_R(R)
    Σs   = Utils._prep_S(Σ, epsilon, stabilize_Σ)
    (best_tuple, path) = _glmnet_path_select(
        μ, Σ, T;
        k=k,
        nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, lambda=lambda,
        alpha=alpha_used, alpha_select=:fixed,
        nadd=nadd, nnested=nnested,
        standardize=standardize,
        epsilon=epsilon, stabilize_Σ=stabilize_Σ,
        compute_weights=false, normalize_weights=false, use_refit=false,
        do_checks=do_checks,
        cv_folds=cv_folds, cv_verbose=false,
        gcv_kappa=1.0
    )
    best_idx, jstar, βj, path_status, βmat = best_tuple

    # Optional one-shot densify to hit exactly k if we ended just below k
    if path_status === :LASSO_PATH_ALMOST_K && jstar > 0 && length(best_idx) < k
        sel2, βj2, status2, used = _densify_lambda_exact_k(
            μ, Σ, T, path, βmat, jstar; k=k, alpha=alpha_used, standardize=standardize
        )
        if used && !isempty(sel2)
            best_idx    = sel2
            βj          = βj2
            path_status = status2
        end
    end

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
# mve_lasso_relaxation_search — moment-based

    mve_lasso_relaxation_search(μ, Σ, T; R=nothing, k,
        nlambda=100, lambda_min_ratio=1e-3, lambda=nothing,
        alpha=1.0,
        nadd=80, nnested=2,                 # λ densification near target k
        standardize=false, epsilon=Utils.EPS_RIDGE, stabilize_Σ=true,
        compute_weights=true, normalize_weights=false, use_refit=false,
        do_checks=false,
        cv_folds=5, cv_verbose=false,
        alpha_select=:fixed,                # :fixed | :oos_cv | :gcv
        gcv_kappa=1.0
    ) -> (selection, weights, sr, status, alpha)

Same as the R-based API but you provide moments (μ, Σ, T). If `alpha_select=:oos_cv`,
you must also provide `R` to form rolling validation folds; otherwise OOS-CV is disabled.

Internally the path selector is called with (μ, Σ, T), and SR/weights are computed from moments.
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
    alpha::Union{Real,AbstractVector{<:Real}} = 1.00,
    # NEW: densification controls
    nadd::Int = 80,
    nnested::Int = 2,
    #
    standardize::Bool = false,
    epsilon::Real = Utils.EPS_RIDGE,
    stabilize_Σ::Bool = true,
    compute_weights::Bool = true,
    normalize_weights::Bool = false,
    use_refit::Bool = false,
    do_checks::Bool = false,
    cv_folds::Int = 5,
    cv_verbose::Bool = false,
    alpha_select::Symbol = :fixed,
    gcv_kappa::Real = 1.0
) :: NamedTuple{(:selection, :weights, :sr, :status, :alpha)}
    lambda, is_grid, agrid, alpha_used = _validate_moment_based_inputs!(
        μ, Σ, T, R, k, epsilon, alpha, lambda, do_checks, cv_folds, alpha_select
    )

    # α selection via GCV on synthetic design (kept as in your code)
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

    # α-grid OOS CV (requires R)
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

                (best_tuple, _) = _glmnet_path_select(
                    μtr, Σtr, Ttr;
                    k=k,
                    nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, lambda=lambda,
                    alpha=αg, alpha_select=:fixed,
                    nadd=nadd, nnested=nnested,
                    standardize=standardize,
                    epsilon=epsilon, stabilize_Σ=stabilize_Σ,
                    compute_weights=false, normalize_weights=false, use_refit=false,
                    do_checks=false,
                    cv_folds=cv_folds, cv_verbose=false,
                    gcv_kappa=1.0
                )
                best_idx, _, βj, _, _ = best_tuple

                Σtr_s = Utils._prep_S(Σtr, epsilon, stabilize_Σ)
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

    # Final fit on moments
    Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)
    (best_tuple, path) = _glmnet_path_select(
        μ, Σ, T;
        k=k,
        nlambda=nlambda, lambda_min_ratio=lambda_min_ratio, lambda=lambda,
        alpha=alpha_used, alpha_select=:fixed,
        nadd=nadd, nnested=nnested,
        standardize=standardize,
        epsilon=epsilon, stabilize_Σ=stabilize_Σ,
        compute_weights=false, normalize_weights=false, use_refit=false,
        do_checks=do_checks,
        cv_folds=cv_folds, cv_verbose=false,
        gcv_kappa=1.0
    )
    best_idx, jstar, βj, path_status, βmat = best_tuple

    if path_status === :LASSO_PATH_ALMOST_K && jstar > 0 && length(best_idx) < k
        sel2, βj2, status2, used = _densify_lambda_exact_k(
            μ, Σ, T, path, βmat, jstar; k=k, alpha=alpha_used, standardize=standardize
        )
        if used && !isempty(sel2)
            best_idx    = sel2
            βj          = βj2
            path_status = status2
        end
    end

    N = length(μ)
    sel, w, sr, status_out = _finalize_from_selection!(
        N, best_idx, μ, Σs, βj, normalize_weights, compute_weights,
        epsilon, use_refit, used_cv, path_status
    )
    return (selection=sel, weights=w, sr=sr, status=status_out, alpha=alpha_used)
end

end # module
