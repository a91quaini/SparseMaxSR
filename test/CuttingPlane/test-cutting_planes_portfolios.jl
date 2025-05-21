using Test
using SparseMaxSR
using LinearAlgebra   # for I
using Random          # for reproducibility

@testset "cutting_planes_portfolios" begin
    # fix RNG so warm starts / heuristics are reproducible
    Random.seed!(1234)

    # tiny 4-asset example
    μ = [0.10, 0.20, 0.15, 0.05]
    n = length(μ)
    Σ = Matrix{Float64}(I, n, n)   # identity covariance
    γ = ones(n)                    # equal penalties

    for k in 1:3
        @testset "k = $k" begin
            z = cutting_planes_portfolios(μ, Σ, γ, k;
                                          ΔT_max=30.0,
                                          gap=1e-4,
                                          num_random_restarts=3,
                                          use_warm_start=true,
                                          use_socp_lb=false,
                                          use_heuristic=true,
                                          use_kelley_primal=false)

            @test isa(z, Vector{Int})
            @test length(z) == n
            @test sum(z) ≤ k
            @test all(x -> x == 0 || x == 1, z)

            # verify that the reported support is feasible and gives
            # a valid inner_dual value
            # inds = findall(z .== 1)
            # result = inner_dual(μ, Σ, inds)
            # @test result.status == MathOptInterface.OPTIMAL
        end
    end
end
