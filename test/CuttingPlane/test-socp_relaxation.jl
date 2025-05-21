using Test
using SparseMaxSR               # your package
using JuMP
using MosekTools
using LinearAlgebra             # for I
import MathOptInterface         # for OPTIMAL
const MOI = MathOptInterface

@testset "socp_relaxation" begin
    # simple 3-asset problem
    μ = [0.1, 0.2, 0.15]
    n = length(μ)
    Σ = Matrix{Float64}(I, n, n)
    γ = ones(n)
    k = 2

    # call the relaxation
    res = portfolios_socp(μ, Σ, γ, k)

    # 1) solver status
    @test res.status == MOI.OPTIMAL

    # 2) objective finite
    @test isa(res.ofv, Float64) && isfinite(res.ofv)

    # 3) correct variable lengths
    @test length(res.α) == n
    @test length(res.w) == n
    @test length(res.v) == n
    @test isa(res.t, Float64)

    # 4) primal feasibility of QCQP constraints
    Σα = Σ * res.α
    for i in 1:n
        @test res.w[i] + 1e-8 ≥ Σα[i] + res.λ           # cut constraint
        @test res.v[i] + res.t + 1e-8 ≥ (γ[i]/2) * res.w[i]^2  # norm constraint
    end

    # 5) nonnegativity of slacks
    @test all(res.v .>= 0)
    @test res.t ≥ 0
end
