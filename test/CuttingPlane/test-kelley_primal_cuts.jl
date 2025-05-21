# test/CuttingPlane/test-kelley_primal_cuts.jl
using Test
using SparseMaxSR           # your package
using LinearAlgebra         # for I, clamp
import MathOptInterface     # for OPTIMAL
const MOI = MathOptInterface

@testset "kelley_primal_cuts" begin
    μ = [0.1, 0.2, 0.15, 0.05]
    n = length(μ)
    Σ = Matrix{Float64}(I, n, n)
    γ = ones(n)
    k = 2

    # start at the zero vector
    stab0 = zeros(n)

    cuts = cplex_misocp_relaxation # ensure this is loaded first
    cuts = SparseMaxSR.CuttingPlane.kelley_primal_cuts(μ, Σ, γ, k, stab0, 3; eps=1e-12)

    @test isa(cuts, Vector)
    @test all( isa(cut.p, Float64) && isfinite(cut.p) for cut in cuts )
    @test all( length(cut.grad) == n for cut in cuts )
    @test all( cut.status == MOI.OPTIMAL for cut in cuts )
    @test length(cuts) ≤ 3
end
