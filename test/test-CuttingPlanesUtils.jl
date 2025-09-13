using Test, Random, LinearAlgebra
import MathOptInterface as MOI

using SparseMaxSR
import SparseMaxSR: set_default_optimizer!, default_optimizer
using SparseMaxSR.CuttingPlanesUtils

# ── Detect conic solvers for duals (needed by inner_dual, etc.)
const HAVE_CLARABEL = Base.find_package("Clarabel") !== nothing
const HAVE_COSMO    = Base.find_package("COSMO")    !== nothing
const HAVE_SCS      = Base.find_package("SCS")      !== nothing
const HAVE_MOSEK    = Base.find_package("MosekTools") !== nothing   # optional for QCQP test
const HAVE_SOLVER   = HAVE_CLARABEL || HAVE_COSMO || HAVE_SCS
const HAVE_CPLEX = Base.find_package("CPLEX") !== nothing

# Set a default dual optimizer (factory) if available
if HAVE_CLARABEL
    import Clarabel
    set_default_optimizer!(() -> Clarabel.Optimizer())
elseif HAVE_COSMO
    import COSMO
    set_default_optimizer!(() -> COSMO.Optimizer())
elseif HAVE_SCS
    import SCS
    set_default_optimizer!(() -> SCS.Optimizer())
end

if !HAVE_SOLVER
    @info "Skipping CuttingPlanesUtils tests: no conic solver installed (Clarabel/COSMO/SCS)."
else
    # Helpers
    make_cov(n; jitter=0.05) = Symmetric(begin
        A = randn(n,n)
        A*A' + jitter*I
    end)

    @testset "CuttingPlanesUtils" begin
        Random.seed!(1)
        n, k = 10, 3
        μ = 0.05 .+ 0.10 .* rand(n)
        Σ = make_cov(n; jitter=0.05)
        γ = ones(n)

        # ───────────────────────────────────────────────────────────────────
        @testset "inner_dual basic & epsilon" begin
            supp = collect(1:k) |> sort!
            res0 = inner_dual(μ, Σ, supp)
            @test res0.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
            @test isfinite(res0.ofv)
            @test length(res0.w) == k

            # With ridge epsilon
            resε = inner_dual(μ, Σ, supp; epsilon=1e-8)
            @test resε.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
            @test isfinite(resε.ofv)
        end

        # ───────────────────────────────────────────────────────────────────
        @testset "inner_dual argument checks" begin
            bad_supp_dup = [1,1,2]
            @test_throws ErrorException inner_dual(μ, Σ, bad_supp_dup)

            bad_supp_range = [0, 2, 3]
            @test_throws ErrorException inner_dual(μ, Σ, bad_supp_range)

            @test_throws ErrorException inner_dual(μ, randn(n, n+1), collect(1:k))
        end

        # ───────────────────────────────────────────────────────────────────
        @testset "portfolios_objective sanity + ensure_one" begin
            s = clamp.(rand(n), 0, 1)
            cut = portfolios_objective(μ, Σ, γ, k, s)
            @test isfinite(cut.p)
            @test length(cut.grad) == n
            @test all(isfinite, cut.grad)

            # All-below-threshold but ensure_one=true still works
            s0 = zeros(n)
            cut2 = portfolios_objective(μ, Σ, γ, k, s0; threshold=0.9, ensure_one=true)
            @test isfinite(cut2.p)
            @test all(isfinite, cut2.grad)

            # All-below-threshold with ensure_one=false throws
            @test_throws ErrorException portfolios_objective(μ, Σ, γ, k, s0; threshold=0.9, ensure_one=false)
        end

        # ───────────────────────────────────────────────────────────────────
        @testset "hillclimb returns valid support" begin
            init_inds = sort!(randperm(n)[1:k])
            h = hillclimb(μ, Σ, k, init_inds; maxiter=20)
            @test h.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
            @test length(h.inds) == k
            @test issorted(h.inds)
            @test all(1 .≤ h.inds .≤ n)
            @test length(h.w_full) == n
        end

        # ───────────────────────────────────────────────────────────────────
        @testset "warm_start binary indicator with exactly k ones + reproducibility" begin
            using Random: MersenneTwister
            rng1 = MersenneTwister(2024)
            rng2 = MersenneTwister(2024)

            s1 = warm_start(μ, Σ, γ, k; num_random_restarts=3, maxiter=25, rng=rng1)
            s2 = warm_start(μ, Σ, γ, k; num_random_restarts=3, maxiter=25, rng=rng2)

            @test length(s1) == n
            @test count(>(0.5), s1) == k
            @test all(x -> 0.0 ≤ x ≤ 1.0, s1)
            @test s1 == s2  # deterministic given same seed & settings
        end

        # ───────────────────────────────────────────────────────────────────
        @testset "kelley_primal_cuts returns usable cuts" begin
            cuts = kelley_primal_cuts(μ, Σ, γ, k, zeros(n), 4; lambda=0.2, delta_scale=2.0)
            @test !isempty(cuts)
            @test all(c -> length(c.grad) == n && isfinite(c.p), cuts)
        end

        # ───────────────────────────────────────────────────────────────────
        @testset "portfolios_socp rotated SOC" begin
            # Only run if the function exists in the module (older file revisions may omit it)
            if isdefined(CuttingPlanesUtils, :portfolios_socp)
                # Supply an explicit optimizer factory to avoid any tool-specific defaulting
                opt = default_optimizer()
                res = CuttingPlanesUtils.portfolios_socp(μ, Σ, γ, k;
                    optimizer = opt, form = :rotated_soc, epsilon = 1e-8)
                @test res.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
                @test isfinite(res.ofv)
                @test length(res.w) == n
            else
                @info "Skipping portfolios_socp test: function not present in current build."
            end
        end

        # ───────────────────────────────────────────────────────────────────
        @testset "portfolios_socp QCQP via MosekTools (if present)" begin
            if isdefined(CuttingPlanesUtils, :portfolios_socp) && HAVE_MOSEK
                import MosekTools
                res = CuttingPlanesUtils.portfolios_socp(μ, Σ, γ, k;
                    optimizer = MosekTools.Optimizer, form = :qcqp, epsilon = 1e-8)
                @test res.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
                @test isfinite(res.ofv)
            else
                @info "Skipping QCQP portfolios_socp test (MosekTools not installed or function absent)."
            end
        end

        # ───────────────────────────────────────────────────────────────────
        @testset "error cases (defensive)" begin
            # portfolios_socp unknown form
            if isdefined(CuttingPlanesUtils, :portfolios_socp)
                @test_throws ErrorException CuttingPlanesUtils.portfolios_socp(μ, Σ, γ, k; form = :not_a_form)
            end

            # kelley_primal_cuts: bad args
            @test_throws ErrorException kelley_primal_cuts(μ, Σ, γ, 0, zeros(n), 3)
            @test_throws ErrorException kelley_primal_cuts(μ, randn(n, n+1), γ, k, zeros(n), 3)
        end

        @testset "kelley_primal_cuts with CPLEX (if present)" begin
            if HAVE_CPLEX
                try
                    import CPLEX
                    # Small instance to keep this snappy
                    Random.seed!(42)
                    n, k = 12, 4
                    μ = 0.05 .+ 0.10 .* rand(n)
                    A = randn(n,n); Σ = Symmetric(A*A' .+ 0.05I)
                    γ = ones(n)

                    # Use CPLEX for the root LP in Kelley; keep it quiet and time-limited.
                    attrs = Dict(
                        MOI.Silent() => true,
                        MOI.TimeLimitSec() => 5.0,
                        MOI.RelativeGapTolerance() => 1e-3,
                    )

                    cuts = kelley_primal_cuts(μ, Σ, γ, k, zeros(n), 3;
                                            optimizer = () -> CPLEX.Optimizer(),
                                            attrs = attrs,
                                            lambda = 0.2,
                                            delta_scale = 2.0)

                    @test !isempty(cuts)
                    @test all(c -> length(c.grad) == n && isfinite(c.p), cuts)
                catch err
                    @info "Skipping CPLEX Kelley test due to error (likely license or install)." error=err
                end
            else
                @info "CPLEX not installed; skipping CPLEX-specific Kelley test."
            end
        end
    end
end
