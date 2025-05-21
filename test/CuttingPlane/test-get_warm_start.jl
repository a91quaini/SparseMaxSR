using Test, Random
using SparseMaxSR
using LinearAlgebra
import MathOptInterface; const MOI = MathOptInterface

@testset "get_warm_start" begin
  μ = [0.1,0.2,0.15,0.05]
  n = length(μ)
  Σ = Matrix{Float64}(I, n, n)
  γ = ones(n)
  k = 2

  s0 = get_warm_start(μ, Σ, γ, k; num_random_restarts=3)

  @test length(s0) == n
  @test sum(s0 .> 0.5) == k
  @test all(v -> v in (0.0,1.0), s0)
end