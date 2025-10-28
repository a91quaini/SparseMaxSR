
# test/test-LassoRelaxationSearch.jl
# Fully revised tests for LassoRelaxationSearch to match current semantics:
#  • normalize_weights=true → rescale via Utils.normalize_weights (relative L1 safeguard).
#  • use_refit=true        → selection by lasso, then MVE refit on that selection; normalization only rescales; SR invariant.
#  • use_refit=false       → "normalize-coeffs" branch using Utils.normalize_weights; may return :LASSO_ALLEMPTY.
#  • Cross-validation over alpha is supported (when `alpha` is a Vector); the returned NamedTuple now also includes `alpha`.
#  • Do not require cross-problem equalities when normalizations differ; check invariants *within* a solution.
#  • Realistic tolerances (no atol=rtol=0). Guard divisions by tiny sums.
#  • Unique helper names to avoid cross-file redefinition warnings.
#
using Test, Random, LinearAlgebra, Statistics
using SparseMaxSR
using SparseMaxSR.SharpeRatio
using SparseMaxSR.LassoRelaxationSearch

# ---------- helpers (unique names to avoid collisions) -----------------------
_sym(A) = Symmetric((A + A')/2)

function _sr_exact_lasso(w, μ, Σ; epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true)
    S = SparseMaxSR.Utils._prep_S(Matrix(Σ), epsilon, stabilize_Σ)
    # After preparing S once (ridge baked in), set epsilon=0 to avoid double-ridge
    return compute_sr(w, μ, S; epsilon=0.0, stabilize_Σ=false, do_checks=false)
end

abs_sum(x) = abs(sum(x))

# Checks for use_refit=false branch outputs
function _check_normbranch(sel, w, sr, status; N::Int, k::Int)
    @test issorted(sel)
    @test length(sel) ≤ k
    @test length(w) == N
    @test all(iszero, w[setdiff(1:N, sel)])
    if status == :LASSO_ALLEMPTY
        @test all(iszero, w)
        @test sr == 0.0
    else
        @test status in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K)
        @test isfinite(sr) || sr == 0.0
    end
end

const ATOL_SR  = 1e-12
const ATOL_W   = 1e-10
const ATOL_SUM = 1e-10

# ---------- R-based entrypoint -----------------------------------------------
@testset "LassoRelaxationSearch — R-based (default y / custom y; refit & normalize branches)" begin
    Random.seed!(42)
    T, N = 500, 12

    # Factor structure + noise; centered-ish to exercise ALLEMPTY in normalize branch sometimes
    F = randn(T, 3)
    B = 0.4 .* randn(N, 3)
    E = 0.6 .* randn(T, N)
    R = F * B' .+ E

    for (α, label) in ((0.05, "alpha=0.05 (EN)"), (1.0, "alpha=1.0 (LASSO)"))
        @testset "R-based — $(label)" begin
            for k in (1, 3, 5, 8, 12)
                # ---------------- Refit branch ----------------
                sel_r, w_r, sr_r, st_r, alpha_hat = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=true, do_checks=true, normalize_weights=false
                )
                @test issorted(sel_r) && length(sel_r) ≤ k
                @test length(w_r) == N && all(iszero, w_r[setdiff(1:N, sel_r)])
                @test st_r in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K, :LASSO_ALLEMPTY)
                @test isfinite(sr_r) || sr_r == 0.0
                @test alpha_hat ≈ α

                # Same selection/status with compute_weights=false; weights are zeros, SR still reported
                sel_r0, w_r0, sr_r0, st_r0, alpha_hat0 = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=false, use_refit=true, do_checks=true
                )
                @test sel_r0 == sel_r
                @test st_r0 == st_r
                @test all(iszero, w_r0)
                @test isfinite(sr_r0) || sr_r0 == 0.0
                @test alpha_hat0 ≈ α

                # normalize_weights=true only rescales; SR invariant; selection unchanged
                sel_r1, w_r1, sr_r1, st_r1, alpha_hat1 = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=true, do_checks=true, normalize_weights=true
                )
                @test sel_r1 == sel_r && st_r1 == st_r
                @test isapprox(sr_r1, sr_r; atol=ATOL_SR, rtol=0)
                if !all(iszero, w_r)
                    denom = max(abs(sum(w_r)), 1e-6 * norm(w_r, 1), 1e-10)
                    @test isapprox.(w_r1, w_r ./ denom; atol=ATOL_W) |> all
                end
                @test alpha_hat1 ≈ α

                # ---------------- Normalize-coeffs branch ----------------
                sel_n0, w_n0, sr_n0, st_n0, alpha_hat = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=false, do_checks=true, normalize_weights=false
                )
                _check_normbranch(sel_n0, w_n0, sr_n0, st_n0; N=N, k=k)

                sel_n1, w_n1, sr_n1, st_n1, alpha_hat = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=false, do_checks=true, normalize_weights=true
                )
                _check_normbranch(sel_n1, w_n1, sr_n1, st_n1; N=N, k=k)

                # If both are not ALLEMPTY and sum(w_n0) is not tiny, normalize_weights should just rescale
                if st_n0 != :LASSO_ALLEMPTY && st_n1 != :LASSO_ALLEMPTY && abs_sum(w_n0) > 1e-12
                    @test sel_n0 == sel_n1
                    @test isapprox(sr_n0, sr_n1; atol=ATOL_SR, rtol=0)
                    denom = max(abs(sum(w_n0)), 1e-6 * norm(w_n0, 1), 1e-10)
                    @test isapprox.(w_n1, w_n0 ./ denom; atol=ATOL_W) |> all
                    # SR consistency from recomputation
                    μ = vec(mean(R, dims=1))
                    Σ = _sym(cov(R; corrected=true))
                    @test abs(_sr_exact_lasso(w_n0, μ, Σ) - sr_n0) ≤ 1e-9
                    @test abs(_sr_exact_lasso(w_n1, μ, Σ) - sr_n1) ≤ 1e-9
                end
            end
        end
    end

    # default y=ones(T) must match explicit y in refit branch
    k = 4
    sel_a, w_a, sr_a, st_a, alpha_hata = mve_lasso_relaxation_search(R; k=k, alpha=0.2, standardize=false, use_refit=true)
    sel_b, w_b, sr_b, st_b, alpha_hatb = mve_lasso_relaxation_search(R; k=k, alpha=0.2, standardize=false, y=ones(T), use_refit=true)
    @test sel_a == sel_b && st_a == st_b
    @test alpha_hata ≈ 0.2 && alpha_hatb ≈ 0.2
end

@testset "LassoRelaxationSearch — R-based with custom y (normalize branch)" begin
    Random.seed!(101)
    T, N = 400, 10
    R = randn(T, N)
    y = randn(T)  # non-constant
    k = 5
    sel, w, sr, st, alpha_hat = mve_lasso_relaxation_search(R; y=y, k=k, alpha=0.3, standardize=false, use_refit=false)
    _check_normbranch(sel, w, sr, st; N=N, k=k)
    # If not ALLEMPTY, ensure zeros off support and SR finite
    if st != :LASSO_ALLEMPTY
        @test isfinite(sr)
        @test all(iszero, w[setdiff(1:N, sel)])
    end
    @test alpha_hat ≈ 0.3
end

# ---------- Moment-only entrypoint -------------------------------------------
@testset "LassoRelaxationSearch — moment-only (delegation, refit & normalize branches)" begin
    Random.seed!(7)
    N, T = 15, 600

    μ = 0.02 .* randn(N)
    A = randn(N, N)
    Σ = _sym(A * A' + 0.10I)

    for (α, label) in ((0.05, "alpha=0.05 (EN)"), (1.0, "alpha=1.0 (LASSO)"))
        @testset "moment-only — $(label)" begin
            for k in (1, 4, 7, 10, 15)
                # Refit branch
                sel_r0, w_r0, sr_r0, st_r0, alpha_hat = mve_lasso_relaxation_search(
                    μ, Σ, T; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=true, do_checks=true, normalize_weights=false
                )
                @test issorted(sel_r0) && length(sel_r0) ≤ k
                @test length(w_r0) == N && all(iszero, w_r0[setdiff(1:N, sel_r0)])
                @test st_r0 in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K, :LASSO_ALLEMPTY)
                @test isfinite(sr_r0) || sr_r0 == 0.0
                @test alpha_hat ≈ α

                sel_r1, w_r1, sr_r1, st_r1, alpha_hat = mve_lasso_relaxation_search(
                    μ, Σ, T; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=true, do_checks=true, normalize_weights=true
                )
                @test sel_r1 == sel_r0 && st_r1 == st_r0
                @test isapprox(sr_r1, sr_r0; atol=ATOL_SR, rtol=0)
                if !all(iszero, w_r0)
                    denom = max(abs(sum(w_r0)), 1e-6 * norm(w_r0, 1), 1e-10)
                    @test isapprox.(w_r1, w_r0 ./ denom; atol=ATOL_W) |> all
                end

                # Normalize branch
                sel_n0, w_n0, sr_n0, st_n0, alpha_hat = mve_lasso_relaxation_search(
                    μ, Σ, T; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=false, do_checks=true, normalize_weights=false
                )
                _check_normbranch(sel_n0, w_n0, sr_n0, st_n0; N=N, k=k)

                sel_n1, w_n1, sr_n1, st_n1, alpha_hat = mve_lasso_relaxation_search(
                    μ, Σ, T; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=false, do_checks=true, normalize_weights=true
                )
                _check_normbranch(sel_n1, w_n1, sr_n1, st_n1; N=N, k=k)

                if st_n0 != :LASSO_ALLEMPTY && st_n1 != :LASSO_ALLEMPTY && abs_sum(w_n0) > 1e-12
                    @test sel_n0 == sel_n1
                    @test isapprox(sr_n0, sr_n1; atol=ATOL_SR, rtol=0)
                    denom = max(abs(sum(w_n0)), 1e-6 * norm(w_n0, 1), 1e-10)
                    @test isapprox.(w_n1, w_n0 ./ denom; atol=ATOL_W) |> all
                    @test abs(_sr_exact_lasso(w_n0, μ, Σ) - sr_n0) ≤ 1e-9
                    @test abs(_sr_exact_lasso(w_n1, μ, Σ) - sr_n1) ≤ 1e-9
                end
            end
        end
    end
end

# ---------- User-supplied lambda grid validation -----------------------------
@testset "LassoRelaxationSearch — user-supplied λ (monotone & positivity checks)" begin
    Random.seed!(2025)
    T, N, k = 400, 10, 3
    R = randn(T, N)

    lam_ok = [0.5, 0.2, 0.08, 0.03, 0.01]  # strictly decreasing, positive
    sel, w, sr, st, alpha_hat = mve_lasso_relaxation_search(R; k=k, lambda=lam_ok, alpha=0.2, standardize=false)
    @test length(sel) ≤ k && issorted(sel)

    lam_ok = [0.1, 0.1, 0.05]  # non-increasing with a duplicate
    res = mve_lasso_relaxation_search(R; k=k, lambda=lam_ok, alpha=0.2, standardize=false, do_checks=true)

    # basic sanity on return type and fields (now includes :alpha)
    @test isa(res, NamedTuple)
    @test haskey(res, :selection) && haskey(res, :weights) && haskey(res, :sr) && haskey(res, :status) && haskey(res, :alpha)
    @test length(res.selection) ≤ k
    @test isfinite(res.sr) || isnan(res.sr)
    @test res.status in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K, :LASSO_ALLEMPTY)
    @test res.alpha ≈ 0.2
end

# ---------- Edge cases & do_checks -------------------------------------------
@testset "LassoRelaxationSearch — edge cases & do_checks" begin
    # k = 1 minimal case (any path status is acceptable)
    Random.seed!(123)
    T, N = 300, 6
    R = randn(T, N)
    sel, w, sr, st, alpha_hat = mve_lasso_relaxation_search(R; k=1, nlambda=100, lambda_min_ratio=1e-3, compute_weights=true)
    @test length(sel) ≤ 1 && length(w) == N
    @test st in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K, :LASSO_ALLEMPTY)

    # Very weak signal: normalize branch likely returns ALLEMPTY → w=0, sr=0
    Random.seed!(999)
    Rweak = 1e-3 .* randn(T, N)
    sel2, w2, sr2, st2, alpha_hat = mve_lasso_relaxation_search(Rweak; k=4, nlambda=100, lambda_min_ratio=1e-3, compute_weights=true, use_refit=false)
    @test length(sel2) ≤ 4
    @test st2 in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K, :LASSO_ALLEMPTY)
    if st2 == :LASSO_ALLEMPTY
        @test all(iszero, w2)
        @test sr2 == 0.0
    else
        # If normalize_weights=true is used, weights are rescaled; otherwise no requirement on the sum.
        # Here we didn't set it, so we only check finiteness.
        @test isfinite(sr2) || sr2 == 0.0
    end

    # Bad arguments (trigger do_checks)
    @test_throws ErrorException mve_lasso_relaxation_search(randn(1,5); k=1, do_checks=true)
    @test_throws ErrorException mve_lasso_relaxation_search(randn(10,0); k=1, do_checks=true)

    μ = [0.1, 0.2]; Σ = [0.04 0.01; 0.01 0.09]
    @test_throws ErrorException mve_lasso_relaxation_search(μ, ones(2,3), 100; k=1, do_checks=true)
    @test_throws ErrorException mve_lasso_relaxation_search(μ, Σ, 100; k=0, do_checks=true)
end

# ---------- Alpha-grid cross-validation (R-based) -----------------------------
@testset "LassoRelaxationSearch — alpha-grid CV (R-based, refit branch)" begin
    Random.seed!(2026)
    T, N = 480, 16
    k    = 5

    # Factor-ish returns to create signal + noise
    F = randn(T, 3)
    B = 0.5 .* randn(N, 3)
    E = 0.7 .* randn(T, N)
    R = F * B' .+ E

    # Grid over α including LASSO and elastic-net
    alpha_grid = [0.20, 0.50, 1.00]

    sel, w, sr, st, alpha_hat = mve_lasso_relaxation_search(
        R; k=k, alpha=alpha_grid, nlambda=120, lambda_min_ratio=1e-3,
        standardize=false, epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
        compute_weights=true, normalize_weights=false,
        use_refit=true, do_checks=true,
        cv_folds=5, cv_verbose=false
    )

    @test issorted(sel)
    @test length(sel) ≤ k
    @test length(w) == N
    @test all(iszero, w[setdiff(1:N, sel)])
    @test st in (:LASSO_ALPHA_CV, :LASSO_ALLEMPTY)
    if st == :LASSO_ALLEMPTY
        @test all(iszero, w)
        @test sr == 0.0
    else
        @test isfinite(sr) || sr == 0.0
    end
    @test alpha_hat ∈ alpha_grid

    # Determinism with same seed/data/grid
    sel2, w2, sr2, st2, alpha_hat2 = mve_lasso_relaxation_search(
        R; k=k, alpha=alpha_grid, nlambda=120, lambda_min_ratio=1e-3,
        standardize=false, epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
        compute_weights=true, normalize_weights=false,
        use_refit=true, do_checks=true,
        cv_folds=5, cv_verbose=false
    )
    @test sel2 == sel
    @test st2 == st
    @test isapprox.(w2, w; atol=1e-12, rtol=0) |> all
    @test isapprox(sr2, sr; atol=1e-12, rtol=0)
    @test alpha_hat2 == alpha_hat
end

@testset "LassoRelaxationSearch — alpha-grid CV (R-based, vanilla/normalize branch)" begin
    Random.seed!(2027)
    T, N = 420, 14
    k    = 4
    R = randn(T, N)

    alpha_grid = [0.10, 0.40, 0.85]

    # normalize_weights=false
    sel0, w0, sr0, st0, alpha_hat0 = mve_lasso_relaxation_search(
        R; k=k, alpha=alpha_grid, nlambda=80, lambda_min_ratio=1e-3,
        standardize=false, epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
        compute_weights=true, normalize_weights=false,
        use_refit=false, do_checks=true,
        cv_folds=4
    )
    @test issorted(sel0) && length(sel0) ≤ k
    @test length(w0) == N
    @test all(iszero, w0[setdiff(1:N, sel0)])
    @test st0 in (:LASSO_ALPHA_CV, :LASSO_ALLEMPTY)
    if st0 == :LASSO_ALLEMPTY
        @test all(iszero, w0)
        @test sr0 == 0.0
    else
        @test isfinite(sr0) || sr0 == 0.0
    end
    @test alpha_hat0 ∈ alpha_grid

    # normalize_weights=true should only rescale (if non-empty)
    sel1, w1, sr1, st1, alpha_hat1 = mve_lasso_relaxation_search(
        R; k=k, alpha=alpha_grid, nlambda=80, lambda_min_ratio=1e-3,
        standardize=false, epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
        compute_weights=true, normalize_weights=true,
        use_refit=false, do_checks=true,
        cv_folds=4
    )
    @test sel1 == sel0
    @test st1 == st0
    if st0 != :LASSO_ALLEMPTY && abs(sum(w0)) > 1e-12
        denom = max(abs(sum(w0)), 1e-6 * norm(w0, 1), 1e-10)
        @test isapprox.(w1, w0 ./ denom; atol=ATOL_W) |> all
        @test isapprox(sr1, sr0; atol=ATOL_SR, rtol=0)
    end
    @test alpha_hat1 == alpha_hat0
end

@testset "LassoRelaxationSearch — alpha-grid CV (argument checks & folds)" begin
    Random.seed!(2028)
    T, N = 120, 8
    R = randn(T, N)
    k = 3

    # After unique(), grid collapses to len=1 → must error (as designed)
    @test_throws ErrorException mve_lasso_relaxation_search(
        R; k=k, alpha=[0.5, 0.5], nlambda=50, use_refit=true, do_checks=true
    )

    # Too many folds for sample size (no non-empty validation) → error
    @test_throws ErrorException mve_lasso_relaxation_search(
        R; k=k, alpha=[0.2, 0.8], nlambda=50, use_refit=true,
        do_checks=true, cv_folds=T   # pathological: leaves empty val slices
    )

    # Support cap still enforced under CV
    sel, w, sr, st, alpha_hat = mve_lasso_relaxation_search(
        R; k=k, alpha=[0.15, 0.35, 0.95], nlambda=60,
        use_refit=true, compute_weights=true, do_checks=true
    )
    @test length(sel) ≤ k
    @test alpha_hat ∈ [0.15, 0.35, 0.95]
end

@testset "LassoRelaxationSearch — R-based alpha-grid CV (argument checks on fold size)" begin
    Random.seed!(4041)
    T, N = 30, 6
    R = randn(T, N)
    k = 3

    # Small grid just to trigger CV
    alpha_grid = [0.2, 0.8]

    # Too many folds: with min_val=2 default, seg = floor(T/(K+1)) < 2 ⇒ error
    @test_throws ErrorException mve_lasso_relaxation_search(
        R; k=k, alpha=alpha_grid, do_checks=true, cv_folds=T  # K = T ⇒ seg = floor(T/(T+1)) = 0
    )

    # Also expect error when folds make *some* validation slices shorter than 2
    # e.g., pick K so that seg==1
    K_bad = T - 1  # seg = floor(T/(T)) = 1
    @test_throws ErrorException mve_lasso_relaxation_search(
        R; k=k, alpha=alpha_grid, do_checks=true, cv_folds=K_bad
    )

    # A feasible choice (e.g., K=5) should pass
    sel, w, sr, st, alpha_hat = mve_lasso_relaxation_search(
        R; k=k, alpha=alpha_grid, do_checks=true, cv_folds=5
    )
    @test length(sel) ≤ k
    @test length(w) == N
    @test st in (:LASSO_ALPHA_CV, :LASSO_ALLEMPTY)
    @test alpha_hat ∈ alpha_grid
end

@testset "LassoRelaxationSearch — moment+R alpha-grid CV (refit branch)" begin
    Random.seed!(3031)
    T, N = 480, 18
    # factor structure + noise to create some signal
    F = randn(T, 3)
    B = 0.5 .* randn(N, 3)
    E = 0.7 .* randn(T, N)
    R = F * B' .+ E

    μ = vec(mean(R; dims=1))
    Σ = cov(Matrix(R); corrected=true)

    alpha_grid = [0.20, 0.60, 1.00]
    for k in (1, 4, 8, 12)
        sel, w, sr, st, alpha_hat = mve_lasso_relaxation_search(
            μ, Σ, T; R=R, k=k,
            alpha=alpha_grid, nlambda=120, lambda_min_ratio=1e-3,
            standardize=false, epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
            compute_weights=true, normalize_weights=false,
            use_refit=true, do_checks=true,
            cv_folds=5, cv_verbose=false
        )

        @test length(sel) ≤ k && issorted(sel)
        @test length(w) == N
        @test all(iszero, w[setdiff(1:N, sel)])
        @test st in (:LASSO_ALPHA_CV, :LASSO_ALLEMPTY)
        if st == :LASSO_ALLEMPTY
            @test all(iszero, w)
            @test sr == 0.0
        else
            @test isfinite(sr) || sr == 0.0
        end
        @test alpha_hat ∈ alpha_grid

        # Determinism with same inputs
        sel2, w2, sr2, st2, alpha_hat2 = mve_lasso_relaxation_search(
            μ, Σ, T; R=R, k=k,
            alpha=alpha_grid, nlambda=120, lambda_min_ratio=1e-3,
            standardize=false, epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
            compute_weights=true, normalize_weights=false,
            use_refit=true, do_checks=true,
            cv_folds=5, cv_verbose=false
        )
        @test sel2 == sel
        @test st2 == st
        @test isapprox.(w2, w; atol=1e-12, rtol=0) |> all
        @test isapprox(sr2, sr; atol=1e-12, rtol=0)
        @test alpha_hat2 == alpha_hat
    end
end

@testset "LassoRelaxationSearch — moment+R alpha-grid CV (vanilla/normalize branch)" begin
    Random.seed!(3032)
    T, N = 420, 16
    R = randn(T, N)
    μ = vec(mean(R; dims=1))
    Σ = cov(Matrix(R); corrected=true)

    alpha_grid = [0.10, 0.40, 0.85]
    k = 5

    # normalize_weights = false
    sel0, w0, sr0, st0, alpha_hat0 = mve_lasso_relaxation_search(
        μ, Σ, T; R=R, k=k,
        alpha=alpha_grid, nlambda=90, lambda_min_ratio=1e-3,
        standardize=false, epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
        compute_weights=true, normalize_weights=false,
        use_refit=false, do_checks=true,
        cv_folds=4
    )
    @test length(sel0) ≤ k && issorted(sel0)
    @test length(w0) == N
    @test all(iszero, w0[setdiff(1:N, sel0)])
    @test st0 in (:LASSO_ALPHA_CV, :LASSO_ALLEMPTY)
    if st0 == :LASSO_ALLEMPTY
        @test all(iszero, w0)
        @test sr0 == 0.0
    else
        @test isfinite(sr0) || sr0 == 0.0
    end
    @test alpha_hat0 ∈ alpha_grid

    # normalize_weights = true should only rescale if non-empty
    sel1, w1, sr1, st1, alpha_hat1 = mve_lasso_relaxation_search(
        μ, Σ, T; R=R, k=k,
        alpha=alpha_grid, nlambda=90, lambda_min_ratio=1e-3,
        standardize=false, epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
        compute_weights=true, normalize_weights=true,
        use_refit=false, do_checks=true,
        cv_folds=4
    )
    @test sel1 == sel0
    @test st1 == st0
    if st0 != :LASSO_ALLEMPTY && abs(sum(w0)) > 1e-12
        denom = max(abs(sum(w0)), 1e-6 * norm(w0, 1), 1e-10)
        @test isapprox.(w1, w0 ./ denom; atol=ATOL_W) |> all
        @test isapprox(sr1, sr0; atol=ATOL_SR, rtol=0)
    end
    @test alpha_hat1 == alpha_hat0

    # SR consistency (recompute from (μ, Σ))
    if st0 != :LASSO_ALLEMPTY
        Σs = SparseMaxSR.Utils._prep_S(Matrix(Σ), SparseMaxSR.EPS_RIDGE, true)
        sr_chk = SparseMaxSR.SharpeRatio.compute_sr(w0, μ, Σs; stabilize_Σ=false, do_checks=false)
        @test isapprox(sr0, sr_chk; atol=1e-9, rtol=0)
    end
end

@testset "LassoRelaxationSearch — moment+R alpha-grid CV (argument checks)" begin
    Random.seed!(3033)
    T, N = 120, 8
    R = randn(T, N)
    μ = vec(mean(R; dims=1)); Σ = cov(Matrix(R); corrected=true)
    k = 3

    # alpha grid collapses to length 1 after unique() → error
    @test_throws ErrorException mve_lasso_relaxation_search(
        μ, Σ, T; R=R, k=k, alpha=[0.5, 0.5], do_checks=true
    )

    # Too many folds (no non-empty validation slices) → error
    @test_throws ErrorException mve_lasso_relaxation_search(
        μ, Σ, T; R=R, k=k, alpha=[0.2, 0.8], do_checks=true, cv_folds=T
    )

    # Support cap still enforced
    sel, w, sr, st, alpha_hat = mve_lasso_relaxation_search(
        μ, Σ, T; R=R, k=k, alpha=[0.15, 0.35, 0.95], do_checks=true
    )
    @test length(sel) ≤ k
    @test alpha_hat ∈ [0.15, 0.35, 0.95]
end
