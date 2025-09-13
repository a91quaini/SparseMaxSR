using Test, Random, LinearAlgebra
using SparseMaxSR
using SparseMaxSR.MVESelection
using SparseMaxSR.SharpeRatio: compute_mve_sr
import SparseMaxSR: set_default_optimizer!
using Combinatorics: combinations

# ---------------------------
# Solver availability checks
# ---------------------------
const HAVE_HIGHS   = Base.find_package("HiGHS")      !== nothing
const HAVE_GLPK    = Base.find_package("GLPK")       !== nothing
const HAVE_CPLEX   = Base.find_package("CPLEX")      !== nothing   # MILP
const HAVE_MOSEK   = Base.find_package("MosekTools") !== nothing   # dual optimizer (conic/QCQP)
const HAVE_MILP    = HAVE_HIGHS || HAVE_GLPK || HAVE_CPLEX

# Pick a conic default (dual optimizer) if one is present
if Base.find_package("Clarabel") !== nothing
    import Clarabel; set_default_optimizer!(() -> Clarabel.Optimizer())
elseif Base.find_package("COSMO") !== nothing
    import COSMO;    set_default_optimizer!(() -> COSMO.Optimizer())
elseif Base.find_package("SCS") !== nothing
    import SCS;      set_default_optimizer!(() -> SCS.Optimizer())
end

# Helper: brute-force k-set maximizing MVE SR (epsilon forwarded)
function _bruteforce_sel(μ, Σ; k::Int, epsilon::Real=0.0)
    n = length(μ)
    best_sr = -Inf
    best_sel = nothing
    for sel_tup in combinations(1:n, k)
        sel = collect(sel_tup)
        sr  = compute_mve_sr(μ, Σ; selection=sel, epsilon=epsilon, do_checks=false)
        if isfinite(sr) && sr > best_sr
            best_sr = sr
            best_sel = sel
        end
    end
    return sort!(best_sel), best_sr
end

# Helper: make a random SPD-ish covariance
make_cov(n; jitter=0.05) = Symmetric(begin
    A = randn(n, n)
    A*A' + jitter*I
end)

# ===========================
# Testsets
# ===========================
@testset "MVESelection" begin
    Random.seed!(12345)

    # --------------------------
    # Exhaustive = brute force
    # --------------------------
    @testset "exhaustive small matches brute force" begin
        n, k = 8, 3
        μ = 0.05 .+ 0.10 .* rand(n)
        Σ = make_cov(n; jitter=0.05)

        sel_exh = MVESelection.mve_selection_exhaustive_search(μ, Σ, k;
                                                               exactly_k=true,
                                                               epsilon=0.0,
                                                               do_checks=true)
        brute_sel, brute_sr = _bruteforce_sel(μ, Σ; k=k, epsilon=0.0)
        sr_exh = compute_mve_sr(μ, Σ; selection=sel_exh, epsilon=0.0)

        @test issorted(sel_exh)
        @test sel_exh == brute_sel
        @test isapprox(sr_exh, brute_sr; atol=1e-10, rtol=0)
    end

    # --------------------------
    # k = 1 rule (argmax |μ| / √σii)
    # --------------------------
    @testset "k=1 picks best single asset" begin
        n, k = 10, 1
        μ = 0.05 .+ 0.10 .* rand(n)
        Σ = make_cov(n; jitter=0.02)

        score = abs.(μ) ./ sqrt.(diag(Σ))
        i★ = argmax(score)
        sel = MVESelection.mve_selection_exhaustive_search(μ, Σ, k; epsilon=0.0)
        @test sel == [i★]

        # Also via the public dispatcher (force exhaustive)
        sel2 = MVESelection.compute_mve_selection(μ, Σ, k; method=:exhaustive, epsilon=0.0)
        @test sel2 == [i★]
    end

    # --------------------------
    # k = n returns 1:n
    # --------------------------
    @testset "k=n returns full set" begin
        n, k = 6, 6
        μ = 0.05 .+ 0.10 .* rand(n)
        Σ = make_cov(n; jitter=0.01)
        @test MVESelection.mve_selection_exhaustive_search(μ, Σ, k; epsilon=0.0) == collect(1:n)
        @test MVESelection.compute_mve_selection(μ, Σ, k; method=:exhaustive, epsilon=0.0) == collect(1:n)
    end

    # --------------------------
    # Auto routing: force each path explicitly
    # --------------------------
    @testset "dispatcher :auto toggles paths as requested" begin
        n, k = 10, 3
        μ = 0.05 .+ 0.10 .* rand(n)
        Σ = make_cov(n; jitter=0.05)

        # Force exhaustive path by thresholds
        selE = MVESelection.compute_mve_selection(μ, Σ, k;
            method=:auto, exhaustive_threshold=100, exhaustive_max_combs=typemax(Int),
            epsilon=0.0)
        brute_sel, _ = _bruteforce_sel(μ, Σ; k=k, epsilon=0.0)
        @test selE == brute_sel

        if HAVE_MILP
            # Force cutting-planes by thresholds
            # (We still set an MILP optimizer explicitly below)
            if HAVE_HIGHS
                import HiGHS
                MILP = () -> HiGHS.Optimizer()
            elseif HAVE_GLPK
                import GLPK
                MILP = () -> GLPK.Optimizer()
            else
                MILP = () -> error("unreachable")
            end
            selC = MVESelection.compute_mve_selection(μ, Σ, k;
                method=:auto, exhaustive_threshold=0, exhaustive_max_combs=0,
                optimizer=MILP(), epsilon=0.0)
            # At least its MVE_SR should match the brute optimum (ties allowed).
            srC = compute_mve_sr(μ, Σ; selection=selC, epsilon=0.0)
            _, sr★ = _bruteforce_sel(μ, Σ; k=k, epsilon=0.0)
            @test isapprox(srC, sr★; atol=1e-8, rtol=0)
        else
            @info "Skipping :auto cutting-planes subtest (no MILP solver installed)."
        end
    end

    # --------------------------
    # Ill-conditioned Σ works via ridge & pseudoinverse paths
    # --------------------------
    @testset "ill-conditioned covariance is handled" begin
        n, k = 10, 3
        μ = 0.05 .+ 0.10 .* rand(n)
        # Near-singular covariance
        B = randn(n, 2)
        Σ = Symmetric(B*B')  # rank 2
        # With epsilon=0.0 exhaustive may still work (uses pinv), but keep a small ridge
        sel = MVESelection.mve_selection_exhaustive_search(μ, Σ, k; epsilon=1e-8)
        @test length(sel) == k
        @test issorted(sel)
    end

    # --------------------------
    # Cutting-planes agrees with exhaustive on small instances
    # --------------------------
    @testset "cutting-planes ≈ exhaustive (small instance)" begin
        if !HAVE_MILP
            @info "Skipping: no MILP solver (HiGHS/GLPK/CPLEX) installed."
        else
            n, k = 9, 3
            μ = 0.05 .+ 0.10 .* rand(n)
            Σ = make_cov(n; jitter=0.05)

            # MILP optimizer
            if HAVE_CPLEX
                import CPLEX; MILP = () -> CPLEX.Optimizer()
            elseif HAVE_HIGHS
                import HiGHS; MILP = () -> HiGHS.Optimizer()
            else
                import GLPK;  MILP = () -> GLPK.Optimizer()
            end

            sel_cp = MVESelection.compute_mve_selection(μ, Σ, k;
                                                        method=:cutting_planes,
                                                        optimizer=MILP(),
                                                        epsilon=0.0)
            sel_ex = MVESelection.mve_selection_exhaustive_search(μ, Σ, k; epsilon=0.0)

            sr_cp = compute_mve_sr(μ, Σ; selection=sel_cp, epsilon=0.0)
            sr_ex = compute_mve_sr(μ, Σ; selection=sel_ex, epsilon=0.0)

            # They may tie but pick different argmins under ties; check SR equality
            @test isapprox(sr_cp, sr_ex; atol=1e-8, rtol=0)
        end
    end

    # --------------------------
    # MosekTools as dual optimizer (if available)
    # --------------------------
    @testset "MosekTools as dual optimizer (if present)" begin
        if !HAVE_MILP || !HAVE_MOSEK
            @info "Skipping MosekTools dual test (need MILP + MosekTools)."
        else
            import MosekTools
            # MILP optimizer (any)
            if HAVE_CPLEX
                import CPLEX; MILP = () -> CPLEX.Optimizer()
            elseif HAVE_HIGHS
                import HiGHS; MILP = () -> HiGHS.Optimizer()
            else
                import GLPK;  MILP = () -> GLPK.Optimizer()
            end

            n, k = 10, 3
            μ = 0.05 .+ 0.10 .* rand(n)
            Σ = make_cov(n; jitter=0.03)

            sel = MVESelection.compute_mve_selection(μ, Σ, k;
                                                     method=:cutting_planes,
                                                     optimizer=MILP(),
                                                     dual_optimizer=MosekTools.Optimizer,
                                                     epsilon=0.0)
            @test length(sel) == k
            @test issorted(sel)
        end
    end

    # --------------------------
    # CPLEX as MILP (if present)
    # --------------------------
    @testset "CPLEX as MILP (if present)" begin
        if !HAVE_CPLEX
            @info "Skipping CPLEX MILP test (CPLEX not installed)."
        else
            import CPLEX
            n, k = 10, 3
            μ = 0.05 .+ 0.10 .* rand(n)
            Σ = make_cov(n; jitter=0.05)

            sel = MVESelection.compute_mve_selection(μ, Σ, k;
                                                     method=:cutting_planes,
                                                     optimizer=CPLEX.Optimizer(),
                                                     epsilon=0.0)
            @test length(sel) == k
            @test issorted(sel)
        end
    end

    # --------------------------
    # Argument checks & errors
    # --------------------------
    @testset "argument checks" begin
        n, k = 6, 3
        μ = 0.05 .+ 0.10 .* rand(n)
        Σ = make_cov(n; jitter=0.02)

        @test_throws ErrorException MVESelection.compute_mve_selection(μ, Σ, 0; method=:exhaustive)
        @test_throws ErrorException MVESelection.compute_mve_selection(μ, Σ, n+1; method=:exhaustive)

        # wrong Σ shape
        @test_throws ErrorException MVESelection.compute_mve_selection(μ, randn(n, n+1), k; method=:exhaustive)
    end
end
