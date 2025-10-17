# test-LassoRelaxationSearch.jl — tests aligned with :LASSO_ALLEMPTY + |sum(w)|=1
#
# Highlights:
#  • Normalize-coeffs branch: if β support sums to ~0 → status=:LASSO_ALLEMPTY, w=0, sr=0
#    otherwise enforce abs(sum(w))≈1.
#  • Refit branch unchanged; compute_weights toggles whether weights are computed.
#  • R-based entrypoint with default y=ones(T) and custom y.
#  • Moment-only entrypoint reusing the same selector.
#  • User λ-grid validation (strictly decreasing, positive).
#  • EPS_RIDGE is a scalar constant.

using Test, Random, LinearAlgebra, Statistics
using SparseMaxSR
using SparseMaxSR.LassoRelaxationSearch

_sym(A) = Symmetric((A + A')/2)

# Helper: check normalize-branch weights according to new contract
function _check_normalize_branch_output(sel, w, sr, status; N::Int, k::Int, atol=1e-8)
    @test issorted(sel)
    @test length(sel) ≤ k
    @test length(w) == N
    @test all(iszero, w[setdiff(1:N, sel)])  # zeros off-support
    if status == :LASSO_ALLEMPTY
        @test all(iszero, w)
        @test sr == 0.0
    else
        @test status in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K)
        @test isapprox(abs(sum(w)), 1.0; atol=atol, rtol=0)
        @test isfinite(sr) || sr == 0.0
    end
end

# --------------------------
# R-based entrypoint (default y and custom y)
# --------------------------
@testset "LassoRelaxationSearch — R-based (default y, custom y, alpha)" begin
    Random.seed!(42)
    T, N = 500, 12

    # Centered returns (likely to trigger ALLEMPTY in normalize branch when signs cancel)
    F = randn(T, 2)
    B = 0.3 .* randn(N, 2)
    E = 0.7 .* randn(T, N)
    R = F * B' .+ E

    for (α, label) in ((0.05, "alpha=0.05 (EN)"), (1.0, "alpha=1.0 (LASSO)"))
        @testset "R-based — $(label)" begin
            for k in (1, 3, 5, 8, 12)
                # --- Refit branch (compute_weights toggled)
                sel_r, w_r, sr_r, st_r = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=true, do_checks=true
                )
                @test issorted(sel_r)
                @test length(sel_r) ≤ k
                @test length(w_r) == N
                @test all(iszero, w_r[setdiff(1:N, sel_r)])  # zeros off-support
                @test st_r in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K)
                @test isfinite(sr_r) || sr_r == 0.0

                # Same selection/status if compute_weights=false (weights are zeros)
                sel_r0, w_r0, sr_r0, st_r0 = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=false, use_refit=true, do_checks=true
                )
                @test sel_r0 == sel_r
                @test st_r0 == st_r
                @test all(iszero, w_r0)
                @test isfinite(sr_r0) || sr_r0 == 0.0

                # --- Normalize-coeffs branch (new semantics)
                sel_n, w_n, sr_n, st_n = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=false, do_checks=true
                )
                _check_normalize_branch_output(sel_n, w_n, sr_n, st_n; N=N, k=k)
            end
        end
    end

    # Default y=ones(T) equals giving y explicitly (refit branch only)
    k = 4
    sel_a, w_a, sr_a, st_a = mve_lasso_relaxation_search(
        R; k=k, alpha=0.1, standardize=false, use_refit=true
    )
    sel_b, w_b, sr_b, st_b = mve_lasso_relaxation_search(
        R; k=k, alpha=0.1, standardize=false, y=ones(T), use_refit=true
    )
    @test sel_a == sel_b
    @test st_a == st_b

    # Custom non-constant y: sanity checks (normalize branch semantics)
    yc = randn(T)
    sel_c, w_c, sr_c, st_c = mve_lasso_relaxation_search(
        R; k=k, alpha=0.2, standardize=false, y=yc, use_refit=false
    )
    _check_normalize_branch_output(sel_c, w_c, sr_c, st_c; N=N, k=k)
end

# --------------------------
# Moment-only entrypoint (delegation to selector)
# --------------------------
@testset "LassoRelaxationSearch — moment-only (delegated selection)" begin
    Random.seed!(7)
    N, T = 15, 600

    μ = 0.02 .* randn(N)  # centered-ish; acceptable for now
    A = randn(N, N)
    Σ = _sym(A * A' + 0.10I)  # SPD

    for (α, label) in ((0.05, "alpha=0.05 (EN)"), (1.0, "alpha=1.0 (LASSO)"))
        @testset "moment-only — $(label)" begin
            for k in (1, 4, 7, 10, 15)
                # Refit branch
                sel_r, w_r, sr_r, st_r = mve_lasso_relaxation_search(
                    μ, Σ, T; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=true, do_checks=true
                )
                @test issorted(sel_r)
                @test length(sel_r) ≤ k
                @test length(w_r) == N
                @test all(iszero, w_r[setdiff(1:N, sel_r)])
                @test st_r in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K)
                @test isfinite(sr_r) || sr_r == 0.0

                # Normalize-coeffs branch (new semantics)
                sel_n, w_n, sr_n, st_n = mve_lasso_relaxation_search(
                    μ, Σ, T; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=false, do_checks=true
                )
                _check_normalize_branch_output(sel_n, w_n, sr_n, st_n; N=N, k=k)
            end
        end
    end
end

# --------------------------
# User-supplied lambda grid checks
# --------------------------
@testset "LassoRelaxationSearch — user-supplied λ (monotone checks)" begin
    Random.seed!(2025)
    T, N, k = 400, 10, 3
    R = randn(T, N)

    lam_ok = [0.5, 0.2, 0.08, 0.03, 0.01]
    sel, w, sr, st = mve_lasso_relaxation_search(
        R; k=k, lambda=lam_ok, alpha=0.2, standardize=false
    )
    @test length(sel) ≤ k
    @test issorted(sel)

    lam_bad1 = [0.1, 0.1, 0.05]    # non-strict
    @test_throws ErrorException mve_lasso_relaxation_search(
        R; k=k, lambda=lam_bad1, alpha=0.2, standardize=false
    )

    lam_bad2 = [0.01, 0.05, 0.2]   # increasing
    @test_throws ErrorException mve_lasso_relaxation_search(
        R; k=k, lambda=lam_bad2, alpha=0.2, standardize=false
    )

    lam_bad3 = [0.2, 0.05, 0.0]    # nonpositive
    @test_throws ErrorException mve_lasso_relaxation_search(
        R; k=k, lambda=lam_bad3, alpha=0.2, standardize=false
    )
end

# --------------------------
# Edge cases & argument checks
# --------------------------
@testset "LassoRelaxationSearch — edge cases" begin
    # k = 1 minimal case
    Random.seed!(123)
    T, N = 300, 6
    R = randn(T, N)
    sel, w, sr, st = mve_lasso_relaxation_search(
        R; k=1, nlambda=100, lambda_min_ratio=1e-3, compute_weights=true
    )
    @test length(sel) ≤ 1
    @test length(w) == N
    @test st in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K, :LASSO_ALLEMPTY)

    # Very weak signal: likely ALLEMPTY in normalize branch
    Random.seed!(999)
    Rweak = 1e-3 .* randn(T, N)
    sel2, w2, sr2, st2 = mve_lasso_relaxation_search(
        Rweak; k=4, nlambda=100, lambda_min_ratio=1e-3, compute_weights=true, use_refit=false
    )
    @test length(sel2) ≤ 4
    @test st2 in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K, :LASSO_ALLEMPTY)
    if st2 == :LASSO_ALLEMPTY
        @test all(iszero, w2)
        @test sr2 == 0.0
    else
        @test isapprox(abs(sum(w2)), 1.0; atol=1e-8, rtol=0)
    end

    # Bad arguments (trigger do_checks)
    @test_throws ErrorException mve_lasso_relaxation_search(randn(1,5); k=1, do_checks=true)
    @test_throws ErrorException mve_lasso_relaxation_search(randn(10,0); k=1, do_checks=true)

    μ = [0.1, 0.2]; Σ = [0.04 0.01; 0.01 0.09]
    @test_throws ErrorException mve_lasso_relaxation_search(μ, ones(2,3), 100; k=1, do_checks=true)
    @test_throws ErrorException mve_lasso_relaxation_search(μ, Σ, 100; k=0, do_checks=true)
end


