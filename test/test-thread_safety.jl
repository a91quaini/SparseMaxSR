# Thread-safety & reentrancy tests for LASSO and MIQP
#
# Strategy:
#  1) Build a fixed synthetic instance for each method (read-only shared data).
#  2) Compute a serial "golden" result.
#  3) Fire many concurrent evaluations of the SAME call via Threads.@threads
#     with (a) LASSO, (b) MIQP, and (c) mixed LASSO+MIQP.
#  4) Assert: no errors, equal selections, ≈ weights & SR to baseline.
#
# Notes:
#  • We force BLAS to 1 thread and MIQP to threads=1 to avoid oversubscription.
#  • Parallelism stress comes from Julia threads, not from inside the solver.
#
using Test, Random, LinearAlgebra, Statistics
using Base.Threads
using SparseMaxSR
using SparseMaxSR.LassoRelaxationSearch
using SparseMaxSR.MIQPHeuristicSearch
using SparseMaxSR.SharpeRatio

# keep BLAS single-threaded to avoid oversubscription during the test
try
    LinearAlgebra.BLAS.set_num_threads(1)
catch
    # BLAS implementation may not expose set_num_threads; ignore
end

# small helpers
_sym(A) = Symmetric((A + A')/2)
abs_sum(x) = abs(sum(x))

# ─────────────────────────────────────────────────────────────────────────────
# LASSO: concurrent calls must match serial baseline
# ─────────────────────────────────────────────────────────────────────────────
@testset "Thread-safety: LASSO concurrent calls == serial baseline" begin
    Random.seed!(20251104)

    # Synthetic returns panel (R-based API)
    T, N = 480, 18
    F = randn(T, 3)
    B = 0.6 .* randn(N, 3)
    E = 0.7 .* randn(T, N)
    R = F * B' .+ E

    k      = 6
    αgrid  = [0.2, 0.6, 1.0]  # exercise path building
    ntrials = max(2nthreads(), 8)  # at least a few concurrent calls

    # Serial baseline (refit=true, CV on α-grid)
    base_sel, base_w, base_sr, base_st, base_α = mve_lasso_relaxation_search(
        R; k=k, alpha=αgrid, compute_weights=true, use_refit=true,
        cv_folds=5, standardize=false, do_checks=true
    )

    @test issorted(base_sel)
    @test length(base_w) == N

    # Parallel hammer: same inputs, many times
    sels  = Vector{Vector{Int}}(undef, ntrials)
    srs   = Vector{Float64}(undef, ntrials)
    ws    = Vector{Vector{Float64}}(undef, ntrials)
    sts   = Vector{Symbol}(undef, ntrials)
    ahats = Vector{Float64}(undef, ntrials)

    @threads for t in 1:ntrials
        sel, w, sr, st, α̂ = mve_lasso_relaxation_search(
            R; k=k, alpha=αgrid, compute_weights=true, use_refit=true,
            cv_folds=5, standardize=false, do_checks=true
        )
        sels[t]  = sel
        ws[t]    = w
        srs[t]   = sr
        sts[t]   = st
        ahats[t] = α̂
    end

    # All concurrent results must match the baseline (reentrancy + determinism)
    for t in 1:ntrials
        @test sels[t] == base_sel
        @test sts[t]  == base_st
        @test isapprox(ahats[t], base_α; atol=0, rtol=0)
        @test isapprox(srs[t],  base_sr; atol=1e-10, rtol=0)
        @test length(ws[t]) == N
        @test isapprox(ws[t],  base_w;  atol=1e-10, rtol=0)
    end

    # Moment-API fixed-α sanity under concurrency (vanilla, no refit)
    μ = 0.02 .* randn(N)
    A = randn(N, N); Σ = _sym(A*A' + 0.10I)
    α = 1.0

    bsel2, bw2, bsr2, bst2, bα2 = mve_lasso_relaxation_search(
        μ, Σ, T; k=k, alpha=α, compute_weights=true, use_refit=false,
        standardize=false, do_checks=true
    )

    sels2 = Vector{Vector{Int}}(undef, ntrials)
    srs2  = Vector{Float64}(undef, ntrials)
    ws2   = Vector{Vector{Float64}}(undef, ntrials)
    sts2  = Vector{Symbol}(undef, ntrials)

    @threads for t in 1:ntrials
        sel, w, sr, st, _ = mve_lasso_relaxation_search(
            μ, Σ, T; k=k, alpha=α, compute_weights=true, use_refit=false,
            standardize=false, do_checks=true
        )
        sels2[t] = sel; ws2[t] = w; srs2[t] = sr; sts2[t] = st
    end

    for t in 1:ntrials
        @test sels2[t] == bsel2
        @test sts2[t]  == bst2
        @test isapprox(srs2[t], bsr2; atol=1e-10, rtol=0)
        @test isapprox(ws2[t],  bw2;  atol=1e-10, rtol=0)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# MIQP: concurrent calls must match serial baseline (solver threads=1)
# ─────────────────────────────────────────────────────────────────────────────
@testset "Thread-safety: MIQP concurrent calls == serial baseline (threads=1)" begin
    Random.seed!(20251104)

    n, k = 12, 4
    μ = 0.02 .+ 0.04 .* rand(n)
    A = randn(n, n); Σ = _sym(A*A' + 0.08I)

    ntrials = max(2nthreads(), 8)

    # Serial baseline (exactly_k=true, normalize=true, use_refit=false)
    base = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                     stabilize_Σ=false, compute_weights=true,
                                     use_refit=false, threads=1,
                                     exactly_k=true, normalize_weights=true)

    @test length(base.selection) == k
    @test abs(abs_sum(base.weights) - 1.0) ≤ 1e-10

    # Parallel hammer
    sels = Vector{Vector{Int}}(undef, ntrials)
    srs  = Vector{Float64}(undef, ntrials)
    ws   = Vector{Vector{Float64}}(undef, ntrials)

    @threads for t in 1:ntrials
        r = mve_miqp_heuristic_search(μ, Σ; k=k, γ=1.0,
                                      stabilize_Σ=false, compute_weights=true,
                                      use_refit=false, threads=1,
                                      exactly_k=true, normalize_weights=true)
        sels[t] = r.selection
        srs[t]  = r.sr
        ws[t]   = r.weights
    end

    for t in 1:ntrials
        @test sels[t] == base.selection
        @test isapprox(srs[t], base.sr; atol=1e-10, rtol=0)
        @test isapprox(ws[t],  base.weights; atol=1e-10, rtol=0)
        @test abs(abs_sum(ws[t]) - 1.0) ≤ 1e-10
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Mixed workload: LASSO and MIQP in parallel shouldn’t interfere
# ─────────────────────────────────────────────────────────────────────────────
@testset "Thread-safety: mixed LASSO + MIQP parallel workload" begin
    Random.seed!(20251104)

    # LASSO instance (R-based)
    T, N = 360, 16
    F = randn(T, 3); B = 0.5 .* randn(N, 3); E = 0.6 .* randn(T, N)
    R = F * B' .+ E
    kL = 5
    αgrid = [0.3, 1.0]

    l_basel_sel, l_basel_w, l_basel_sr, l_basel_st, l_basel_α =
        mve_lasso_relaxation_search(R; k=kL, alpha=αgrid, compute_weights=true,
                                    use_refit=true, cv_folds=4, standardize=false, do_checks=true)

    # MIQP instance
    n, kM = 10, 3
    μ = 0.01 .+ 0.03 .* rand(n)
    A = randn(n, n); Σ = _sym(A*A' + 0.06I)

    m_base = mve_miqp_heuristic_search(μ, Σ; k=kM, γ=1.0,
                                       stabilize_Σ=false, compute_weights=true,
                                       use_refit=false, threads=1,
                                       exactly_k=true, normalize_weights=true)

    # Run both in parallel groups
    nrep = max(2nthreads(), 8)
    l_sels = Vector{Vector{Int}}(undef, nrep)
    l_srs  = Vector{Float64}(undef, nrep)
    l_ws   = Vector{Vector{Float64}}(undef, nrep)
    m_sels = Vector{Vector{Int}}(undef, nrep)
    m_srs  = Vector{Float64}(undef, nrep)
    m_ws   = Vector{Vector{Float64}}(undef, nrep)

    @threads for t in 1:nrep
        # LASSO
        sel, w, sr, st, α̂ = mve_lasso_relaxation_search(
            R; k=kL, alpha=αgrid, compute_weights=true, use_refit=true,
            cv_folds=4, standardize=false, do_checks=true
        )
        l_sels[t] = sel; l_ws[t] = w; l_srs[t] = sr

        # MIQP
        r = mve_miqp_heuristic_search(μ, Σ; k=kM, γ=1.0,
                                      stabilize_Σ=false, compute_weights=true,
                                      use_refit=false, threads=1,
                                      exactly_k=true, normalize_weights=true)
        m_sels[t] = r.selection; m_ws[t] = r.weights; m_srs[t] = r.sr
    end

    # Compare to baselines
    for t in 1:nrep
        # LASSO
        @test l_sels[t] == l_basel_sel
        @test isapprox(l_srs[t], l_basel_sr; atol=1e-10, rtol=0)
        @test isapprox(l_ws[t],  l_basel_w;  atol=1e-10, rtol=0)
        # MIQP
        @test m_sels[t] == m_base.selection
        @test isapprox(m_srs[t], m_base.sr; atol=1e-10, rtol=0)
        @test isapprox(m_ws[t],  m_base.weights; atol=1e-10, rtol=0)
        @test abs(abs_sum(m_ws[t]) - 1.0) ≤ 1e-10
    end
end
