using Test
using SparseMaxSR           # your package
using LinearAlgebra         # for I, findall, etc.
import MathOptInterface     # for OPTIMAL status
const MOI = MathOptInterface

@testset "portfolios_objective" begin
    # a tiny 4-asset example
    μ = [0.1, 0.2, 0.15, 0.05]
    n = length(μ)
    Σ = Matrix{Float64}(I, n, n)    # identity covariance
    γ = ones(n)                     # uniform γ
    k = 2
    s = [1.0, 0.0, 1.0, 0.0]        # picks assets 1 & 3

    # call the function
    cut = portfolios_objective(μ, Σ, γ, k, s)

    # field names
    @test hasproperty(cut, :p)
    @test hasproperty(cut, :grad)
    @test hasproperty(cut, :status)

    # basic sanity checks
    @test isa(cut.p, Float64) && isfinite(cut.p)
    @test length(cut.grad) == n
    @test cut.status == MOI.OPTIMAL

    # cardinality matches
    @test sum(s .> 0.5) == k

    # since ∇sᵢ = -½·γᵢ·wᵢ² ≤ 0, gradients should be non-positive
    @test all(cut.grad .<= 0.0)
end
