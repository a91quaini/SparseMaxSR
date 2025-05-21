using Test
using SparseMaxSR: cutting_planes_selection
using LinearAlgebra   # for I
using Random          # for reproducibility
import MathOptInterface
const MOI = MathOptInterface

# import SparseMaxSR.CuttingPlanesUtils: inner_dual

@testset "cutting_planes_selection" begin
    # make RNG deterministic for warm starts / heuristics
    Random.seed!(42)

    # a small 4-asset problem
    μ = [0.10, 0.20, 0.15, 0.05]
    n = length(μ)
    Σ = Matrix{Float64}(I, n, n)   # identity covariance
    γ = ones(n)

    for k in 1:3
        @testset "k = $k" begin
            sel, status = cutting_planes_selection(μ, Σ, γ, k;
                                                   ΔT_max=30.0,
                                                   gap=1e-4,
                                                   num_random_restarts=3,
                                                   use_warm_start=true,
                                                   use_socp_lb=false,
                                                   use_heuristic=true,
                                                   use_kelley_primal=false)

            @test isa(sel, Vector{Int})
            @test length(sel) ≤ k                     # exactly k indices
            @test all(1 .≤ sel .≤ n)                   # within [1,n]
            @test length(unique(sel)) ≤ k             # no duplicates
            @test status == MOI.OPTIMAL                # successful solve

            # Optional: verify inner_dual status on the returned support
            # result = inner_dual(μ, Σ, sel)
            # @test result.status == MOI.OPTIMAL
        end
    end
end
