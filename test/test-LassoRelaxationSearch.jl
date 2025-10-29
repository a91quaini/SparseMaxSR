# test/test-LassoRelaxationSearch.jl
# Comprehensive, modular tests for LassoRelaxationSearch
#
# Coverage:
#   • R-based API: fixed α (refit & vanilla), α-grid OOS-CV, α-GCV (strict ≤ k).
#   • Moment-based API: fixed α (refit & vanilla), α-grid OOS-CV, α-GCV.
#   • Edge cases: k=0, k=N, weak signal → :LASSO_ALLEMPTY, user-supplied λ checks.
#   • do_checks argument validation, fold feasibility, α-range errors.
#
using Test, Random, LinearAlgebra, Statistics
using SparseMaxSR
using SparseMaxSR.SharpeRatio
using SparseMaxSR.LassoRelaxationSearch

# ---------- local helpers (unique names to avoid collisions) -----------------
_symLR(A) = Symmetric((A + A')/2)
_abs_sumLR(x) = abs(sum(x))

function _sr_from_muSigmaLR(w, μ, Σ; epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true)
    S = SparseMaxSR.Utils._prep_S(Matrix(Σ), epsilon, stabilize_Σ)
    return compute_sr(w, μ, S; epsilon=0.0, stabilize_Σ=false, do_checks=false)
end

function _check_vanilla_output_LR(sel, w, sr, status; N::Int, k::Int)
    @test issorted(sel)
    @test length(sel) ≤ k
    @test length(w) == N
    @test all(iszero, w[setdiff(1:N, sel)])
    if status == :LASSO_ALLEMPTY
        @test all(iszero, w)
        @test sr == 0.0
    else
        @test status in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K, :LASSO_ALPHA_CV, :LASSO_ALPHA_GCV)
        @test isfinite(sr) || sr == 0.0
    end
end

const ATOL_SR  = 1e-12
const ATOL_W   = 1e-10

# ============================== R-based API ==================================
@testset "LassoRelaxationSearch — R-based API" begin
    Random.seed!(111)
    T, N = 420, 14

    # Construct factor-ish returns to avoid trivial all-zero
    F = randn(T, 3)
    B = 0.6 .* randn(N, 3)
    E = 0.7 .* randn(T, N)
    R = F * B' .+ E

    @testset "fixed α (refit & vanilla), normalize on/off" begin
        for α in (0.2, 1.0), k in (1, 3, 6, 10, 14)
            # --- refit, compute_weights=true ---
            sel_r, w_r, sr_r, st_r, αhat_r = mve_lasso_relaxation_search(
                R; k=k, alpha=α, compute_weights=true, use_refit=true,
                standardize=false, do_checks=true
            )
            @test issorted(sel_r) && length(sel_r) ≤ k
            @test length(w_r) == N && all(iszero, w_r[setdiff(1:N, sel_r)])
            @test st_r in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K, :LASSO_ALLEMPTY)
            @test isfinite(sr_r) || sr_r == 0.0
            @test αhat_r ≈ α

            # --- refit, normalize_weights=true only rescales ---
            sel_rn, w_rn, sr_rn, st_rn, αhat_rn = mve_lasso_relaxation_search(
                R; k=k, alpha=α, compute_weights=true, use_refit=true,
                normalize_weights=true, standardize=false, do_checks=true
            )
            @test sel_rn == sel_r && st_rn == st_r
            @test isapprox(sr_rn, sr_r; atol=ATOL_SR, rtol=0)
            if !all(iszero, w_r)
                denom = max(abs(sum(w_r)), 1e-6 * norm(w_r, 1), 1e-10)
                @test isapprox.(w_rn, w_r ./ denom; atol=ATOL_W) |> all
            end
            @test αhat_rn ≈ α

            # --- vanilla (use_refit=false), normalize off/on ---
            sel_v, w_v, sr_v, st_v, αhat_v = mve_lasso_relaxation_search(
                R; k=k, alpha=α, compute_weights=true, use_refit=false,
                standardize=false, do_checks=true
            )
            _check_vanilla_output_LR(sel_v, w_v, sr_v, st_v; N=N, k=k)
            @test αhat_v ≈ α

            sel_vn, w_vn, sr_vn, st_vn, αhat_vn = mve_lasso_relaxation_search(
                R; k=k, alpha=α, compute_weights=true, use_refit=false,
                normalize_weights=true, standardize=false, do_checks=true
            )
            _check_vanilla_output_LR(sel_vn, w_vn, sr_vn, st_vn; N=N, k=k)
            @test sel_vn == sel_v && st_vn == st_v
            if st_v != :LASSO_ALLEMPTY && _abs_sumLR(w_v) > 1e-12
                denom = max(abs(sum(w_v)), 1e-6 * norm(w_v, 1), 1e-10)
                @test isapprox.(w_vn, w_v ./ denom; atol=ATOL_W) |> all
                @test isapprox(sr_vn, sr_v; atol=ATOL_SR, rtol=0)
            end
            @test αhat_vn ≈ α
        end
    end

    @testset "k edge cases: k=0, k=N (fixed α)" begin
        α = 0.5
        sel0, w0, sr0, st0, α0 = mve_lasso_relaxation_search(R; k=0, alpha=α)
        @test isempty(sel0) && all(iszero, w0) && sr0 == 0.0 && st0 == :LASSO_ALLEMPTY && α0 ≈ α
        selN, wN, srN, stN, αN = mve_lasso_relaxation_search(R; k=size(R,2), alpha=α, compute_weights=true, use_refit=true)
        @test length(selN) == size(R,2) && issorted(selN)
        @test length(wN) == size(R,2) && all(iszero, wN[setdiff(1:size(R,2), selN)])
        @test isfinite(srN) && stN == :LASSO_PATH_EXACT_K && αN ≈ α
    end

    @testset "α-grid OOS CV (default when α is a vector) — refit" begin
        αgrid = [0.2, 0.6, 1.0]
        sel, w, sr, st, αhat = mve_lasso_relaxation_search(
            R; k=5, alpha=αgrid, compute_weights=true, use_refit=true,
            cv_folds=5, standardize=false, do_checks=true
        )
        @test issorted(sel) && length(sel) ≤ 5
        @test st in (:LASSO_ALPHA_CV, :LASSO_ALLEMPTY)
        @test αhat ∈ αgrid
        @test length(w) == N && all(iszero, w[setdiff(1:N, sel)])
    end

    @testset "α-grid GCV (strict ≤ k, ridge-only df, log-GCV)" begin
        αgrid = [0.1, 0.4, 0.8, 1.0]
        # Feasible case
        sel, w, sr, st, αhat = mve_lasso_relaxation_search(
            R; k=4, alpha=αgrid, alpha_select=:gcv,
            compute_weights=true, use_refit=true, standardize=false, do_checks=true
        )
        @test issorted(sel) && length(sel) ≤ 4
        @test st == :LASSO_ALPHA_GCV
        @test αhat ∈ αgrid

        # Infeasible case: k=0 enforces ≤ k=0 (empty support); GCV should return infeasible/zeros
        αgrid2 = [0.2, 0.5, 0.9]
        sel2, w2, sr2, st2, αhat2 = mve_lasso_relaxation_search(
            R; k=0, alpha=αgrid2, alpha_select=:gcv,
            compute_weights=true, use_refit=true, standardize=false, do_checks=true
        )
        @test isempty(sel2) && all(iszero, w2) && sr2 == 0.0
        @test st2 == :LASSO_GCV_INFEASIBLE
        @test αhat2 == αgrid2[1]
    end

    @testset "user-supplied λ grid validation" begin
        k = 3
        lam_ok = [0.5, 0.2, 0.08, 0.03, 0.01]
        sel, w, sr, st, αhat = mve_lasso_relaxation_search(R; k=k, lambda=lam_ok, alpha=0.2, standardize=false)
        @test length(sel) ≤ k

        lam_dup = [0.1, 0.1, 0.05]  # non-increasing with duplicate: allowed, unique() inside
        res = mve_lasso_relaxation_search(R; k=k, lambda=lam_dup, alpha=0.2, standardize=false, do_checks=true)
        @test isa(res, NamedTuple) && haskey(res, :selection)
    end

    @testset "do_checks — argument errors" begin
        @test_throws ErrorException mve_lasso_relaxation_search(randn(1,5); k=1, do_checks=true)
        @test_throws ErrorException mve_lasso_relaxation_search(randn(10,0); k=1, do_checks=true)
        # α out of range
        @test_throws ErrorException mve_lasso_relaxation_search(R; k=2, alpha=1.1, do_checks=true)
        # α-grid with too few unique values under do_checks
        @test_throws ErrorException mve_lasso_relaxation_search(R; k=2, alpha=[0.3, 0.3], do_checks=true)
        # CV folds infeasible
        @test_throws ErrorException mve_lasso_relaxation_search(R; k=2, alpha=[0.2,0.8], do_checks=true, cv_folds=T)
    end
end

# ============================ Moment-based API ================================
@testset "LassoRelaxationSearch — Moment-based API" begin
    Random.seed!(222)
    N, T = 16, 500
    μ = 0.02 .* randn(N)
    A = randn(N, N)
    Σ = _symLR(A * A' + 0.10I)

    # Also make a returns panel to enable OOS α-CV where needed
    R = randn(T, N)

    @testset "fixed α (refit & vanilla)" begin
        for α in (0.15, 1.0), k in (1, 4, 8, 16)
            sel_r, w_r, sr_r, st_r, αhat_r = mve_lasso_relaxation_search(
                μ, Σ, T; k=k, alpha=α, compute_weights=true, use_refit=true,
                standardize=false, do_checks=true
            )
            @test issorted(sel_r) && length(sel_r) ≤ k
            @test length(w_r) == N && all(iszero, w_r[setdiff(1:N, sel_r)])
            @test st_r in (:LASSO_PATH_EXACT_K, :LASSO_PATH_ALMOST_K, :LASSO_ALLEMPTY)
            @test isfinite(sr_r) || sr_r == 0.0
            @test αhat_r ≈ α

            sel_v, w_v, sr_v, st_v, αhat_v = mve_lasso_relaxation_search(
                μ, Σ, T; k=k, alpha=α, compute_weights=true, use_refit=false,
                standardize=false, do_checks=true
            )
            _check_vanilla_output_LR(sel_v, w_v, sr_v, st_v; N=N, k=k)
            @test αhat_v ≈ α
        end
    end

    @testset "α-grid OOS CV (requires R) — refit" begin
        αgrid = [0.2, 0.6, 1.0]
        sel, w, sr, st, αhat = mve_lasso_relaxation_search(
            μ, Σ, T; R=R, k=5, alpha=αgrid, compute_weights=true, use_refit=true,
            cv_folds=4, standardize=false, do_checks=true
        )
        @test issorted(sel) && length(sel) ≤ 5
        @test st in (:LASSO_ALPHA_CV, :LASSO_ALLEMPTY)
        @test αhat ∈ αgrid
        @test length(w) == N && all(iszero, w[setdiff(1:N, sel)])
    end

    @testset "α-grid GCV on synthetic design (strict ≤ k)" begin
        αgrid = [0.1, 0.4, 0.9]
        sel, w, sr, st, αhat = mve_lasso_relaxation_search(
            μ, Σ, T; k=6, alpha=αgrid, alpha_select=:gcv,
            compute_weights=true, use_refit=true, standardize=false, do_checks=true
        )
        @test issorted(sel) && length(sel) ≤ 6
        @test st == :LASSO_ALPHA_GCV
        @test αhat ∈ αgrid
    end

    @testset "argument errors & edge cases" begin
        # Shape errors
        @test_throws ErrorException mve_lasso_relaxation_search(μ, ones(2,3), T; k=1, do_checks=true)
        @test_throws ErrorException mve_lasso_relaxation_search(μ, Σ, T; k=0, do_checks=true) # k must be ≥1 in moment-API
        # OOS-CV requires R
        @test_throws ErrorException mve_lasso_relaxation_search(μ, Σ, T; k=2, alpha=[0.2,0.6], do_checks=true)
    end
end
