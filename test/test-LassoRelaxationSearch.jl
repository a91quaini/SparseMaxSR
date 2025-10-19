
# test-LassoRelaxationSearch.jl — tests aligned with :LASSO_ALLEMPTY + weights_sum1 normalization
#
# Highlights:
#  • Normalize-coeffs branch (use_refit=false): if β support sums ≈ 0 → status=:LASSO_ALLEMPTY, w=0, sr=0;
#    otherwise weights are either normalized (weights_sum1=true ⇒ |sum(w)|≈1) or left unnormalized, but SR is invariant.
#  • Refit branch (use_refit=true): selection from Lasso, weights via MVE on the selected set; weights_sum1 toggles normalization.
#  • R-based entrypoint with default y=ones(T) and custom y.
#  • Moment-only entrypoint reusing the same selector.
#  • User λ-grid validation (strictly decreasing, positive).
#  • EPS_RIDGE is a scalar constant.
#
using Test, Random, LinearAlgebra, Statistics
using SparseMaxSR
using SparseMaxSR.LassoRelaxationSearch
using SparseMaxSR.SharpeRatio

_sym(A) = Symmetric((A + A')/2)

# Helper: recompute SR after preparing Σ once (avoid double stabilization)
function _sr_exact(w, μ, Σ; epsilon=0.0, stabilize_Σ=true)
    S = SparseMaxSR.Utils._prep_S(Matrix(Σ), epsilon, stabilize_Σ)
    compute_sr(w, μ, S; epsilon=0.0, stabilize_Σ=false, do_checks=false)
end

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
        @test isfinite(sr) || sr == 0.0
    end
end

# --------------------------
# R-based entrypoint (default y and custom y)
# --------------------------
@testset "LassoRelaxationSearch — R-based (default y, custom y, alpha, weights_sum1)" begin
    Random.seed!(42)
    T, N = 500, 12

    # Centered returns (can trigger ALLEMPTY in normalize branch when signs cancel)
    F = randn(T, 2)
    B = 0.3 .* randn(N, 2)
    E = 0.7 .* randn(T, N)
    R = F * B' .+ E

    for (α, label) in ((0.05, "alpha=0.05 (EN)"), (1.0, "alpha=1.0 (LASSO)"))
        @testset "R-based — $(label)" begin
            for k in (1, 3, 5, 8, 12)
                # --- Refit branch (compute_weights toggled; weights_sum1 invariance of SR)
                sel_r, w_r, sr_r, st_r = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=true, do_checks=true, weights_sum1=false
                )
                @test issorted(sel_r)
                @test length(sel_r) ≤ k
                @test length(w_r) == N
                @test all(iszero, w_r[setdiff(1:N, sel_r)])
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

                # Refit again but with normalization turned on → SR unchanged, weights proportional
                sel_r1, w_r1, sr_r1, st_r1 = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=true, do_checks=true, weights_sum1=true
                )
                @test sel_r1 == sel_r
                @test st_r1 == st_r
                @test isapprox(sr_r1, sr_r; atol=0, rtol=0)
                if !all(iszero, w_r)  # avoid division by zero in pathological case
                    @test abs(sum(w_r1) - 1.0) ≤ 1e-10
                    @test norm(w_r1 - (w_r / sum(w_r))) ≤ 1e-8
                end

                # --- Normalize-coeffs branch (new semantics): run with and without weights_sum1
                sel_n0, w_n0, sr_n0, st_n0 = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=false, do_checks=true, weights_sum1=false
                )
                _check_normalize_branch_output(sel_n0, w_n0, sr_n0, st_n0; N=N, k=k)

                sel_n1, w_n1, sr_n1, st_n1 = mve_lasso_relaxation_search(
                    R; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=false, do_checks=true, weights_sum1=true
                )
                _check_normalize_branch_output(sel_n1, w_n1, sr_n1, st_n1; N=N, k=k)

                # If not ALLEMPTY, normalization only rescales weights and SR is invariant
                if st_n0 != :LASSO_ALLEMPTY && st_n1 != :LASSO_ALLEMPTY
                    @test sel_n0 == sel_n1
                    @test isapprox(sr_n0, sr_n1; atol=0, rtol=0)
                    @test abs(sum(w_n1)) ≈ 1.0 atol=1e-10
                    @test norm(w_n1 - (w_n0 / sum(w_n0))) ≤ 1e-8
                    # Recompute SR from weights equals reported SR
                    # Build μ, Σ from data to evaluate SR(w)
                    μ = vec(mean(R, dims=1))
                    Σ = _sym(cov(R; corrected=true))
                    @test abs(_sr_exact(w_n0, μ, Σ; epsilon=SparseMaxSR.EPS_RIDGE) - sr_n0) ≤ 1e-9
                    @test abs(_sr_exact(w_n1, μ, Σ; epsilon=SparseMaxSR.EPS_RIDGE) - sr_n1) ≤ 1e-9
                end
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
@testset "LassoRelaxationSearch — moment-only (delegated selection & weights_sum1)" begin
    Random.seed!(7)
    N, T = 15, 600

    μ = 0.02 .* randn(N)  # centered-ish; acceptable for now
    A = randn(N, N)
    Σ = _sym(A * A' + 0.10I)  # SPD

    for (α, label) in ((0.05, "alpha=0.05 (EN)"), (1.0, "alpha=1.0 (LASSO)"))
        @testset "moment-only — $(label)" begin
            for k in (1, 4, 7, 10, 15)
                # Refit branch (weights_sum1=false vs true)
                sel_r0, w_r0, sr_r0, st_r0 = mve_lasso_relaxation_search(
                    μ, Σ, T; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=true, do_checks=true, weights_sum1=false
                )
                @test issorted(sel_r0)
                @test length(sel_r0) ≤ k
                @test length(w_r0) == N
                @test all(iszero, w_r0[setdiff(1:N, sel_r0)])
                @test st_r0 in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K)
                @test isfinite(sr_r0) || sr_r0 == 0.0

                sel_r1, w_r1, sr_r1, st_r1 = mve_lasso_relaxation_search(
                    μ, Σ, T; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=true, do_checks=true, weights_sum1=true
                )
                @test sel_r1 == sel_r0
                @test st_r1 == st_r0
                @test isapprox(sr_r1, sr_r0; atol=0, rtol=0)
                if !all(iszero, w_r0)
                    @test abs(sum(w_r1) - 1.0) ≤ 1e-10
                    @test norm(w_r1 - (w_r0 / sum(w_r0))) ≤ 1e-8
                end

                # Normalize-coeffs branch (use_refit=false): again check weights_sum1 invariance
                sel_n0, w_n0, sr_n0, st_n0 = mve_lasso_relaxation_search(
                    μ, Σ, T; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=false, do_checks=true, weights_sum1=false
                )
                _check_normalize_branch_output(sel_n0, w_n0, sr_n0, st_n0; N=N, k=k)

                sel_n1, w_n1, sr_n1, st_n1 = mve_lasso_relaxation_search(
                    μ, Σ, T; k=k, nlambda=100, lambda_min_ratio=1e-3,
                    alpha=α, standardize=false,
                    epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
                    compute_weights=true, use_refit=false, do_checks=true, weights_sum1=true
                )
                _check_normalize_branch_output(sel_n1, w_n1, sr_n1, st_n1; N=N, k=k)

                if st_n0 != :LASSO_ALLEMPTY && st_n1 != :LASSO_ALLEMPTY
                    @test sel_n0 == sel_n1
                    @test isapprox(sr_n0, sr_n1; atol=0, rtol=0)
                    @test abs(sum(w_n1)) ≈ 1.0 atol=1e-10
                    @test norm(w_n1 - (w_n0 / sum(w_n0))) ≤ 1e-8
                    # SR consistency (moment inputs)
                    @test abs(_sr_exact(w_n0, μ, Σ; epsilon=SparseMaxSR.EPS_RIDGE) - sr_n0) ≤ 1e-9
                    @test abs(_sr_exact(w_n1, μ, Σ; epsilon=SparseMaxSR.EPS_RIDGE) - sr_n1) ≤ 1e-9
                end
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

    lam_ok = [0.5, 0.2, 0.08, 0.03, 0.01]             # strictly decreasing, positive
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
        # Normalization default can be either true/false; enforce only |sum(w)|≤1 if normalized
        if abs(sum(w2)) > 1 + 1e-8
            @test true  # unnormalized path; no constraint on the sum
        else
            @test isapprox(abs(sum(w2)), 1.0; atol=1e-8, rtol=0)
        end
    end

    # Bad arguments (trigger do_checks)
    @test_throws ErrorException mve_lasso_relaxation_search(randn(1,5); k=1, do_checks=true)
    @test_throws ErrorException mve_lasso_relaxation_search(randn(10,0); k=1, do_checks=true)

    μ = [0.1, 0.2]; Σ = [0.04 0.01; 0.01 0.09]
    @test_throws ErrorException mve_lasso_relaxation_search(μ, ones(2,3), 100; k=1, do_checks=true)
    @test_throws ErrorException mve_lasso_relaxation_search(μ, Σ, 100; k=0, do_checks=true)
end
