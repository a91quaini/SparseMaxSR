using Test
using SparseMaxSR
using SparseMaxSR.MVESelection: mve_selection_exhaustive_search, mve_selection_cutting_planes
using LinearAlgebra  # for I
using Random         # for reproducibility
import MathOptInterface; const MOI = MathOptInterface

@testset "SparseMaxSR MVE Selection Suite" begin
    # seed for reproducibility
    Random.seed!(1234)

    # small identity-covariance problem
    μ = [0.10, 0.20, 0.15, 0.05]
    n = length(μ)
    Σ = Matrix{Float64}(I, n, n)
    expected_top = sortperm(μ, rev=true)

    # 1) Exhaustive search tests
    @testset "mve_selection_exhaustive_search" begin
        for k in 1:n
            sel_exh = mve_selection_exhaustive_search(μ, Σ, k; do_checks=true)
            @test length(sel_exh) <= k
            @test all(1 .<= sel_exh .<= n)
            @test length(unique(sel_exh)) == length(sel_exh)
            @test all(x -> x in expected_top[1:k], sel_exh)
        end
    end

    # 2) compute_mve_selection harness tests
    @testset "compute_mve_selection" begin
        using SparseMaxSR: compute_mve_selection
        for k in 1:n
            sel_cmp = compute_mve_selection(μ, Σ, k; exhaustive_threshold=n)
            @test sel_cmp == mve_selection_exhaustive_search(μ, Σ, k)
        end
        for k in 1:n
            sel_cp = compute_mve_selection(
                μ, Σ, k;
                exhaustive_threshold=1,
                ΔT_max=5.0, gap=1e-4,
                num_random_restarts=2,
                use_warm_start=true,
                use_socp_lb=false,
                use_heuristic=true,
                use_kelley_primal=false,
                do_checks=false
            )
            @test length(sel_cp) <= k
            @test all(x -> x in expected_top[1:k], sel_cp)
        end
    end

    # 3) cutting-planes selection tests
    @testset "mve_selection_cutting_planes" begin
        Random.seed!(42)
        for k in 1:3
            sel = mve_selection_cutting_planes(
                μ, Σ, k;
                ΔT_max=30.0,
                gap=1e-4,
                num_random_restarts=3,
                use_warm_start=true,
                use_socp_lb=false,
                use_heuristic=true,
                use_kelley_primal=false,
                do_checks=false
            )
            @test length(sel) <= k
            @test all(1 .<= sel .<= n)
            @test length(unique(sel)) == length(sel)
            # @test all(x -> x in expected_top[1:k], sel)
        end
    end
end
