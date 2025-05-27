using Test
using LinearAlgebra         # for I
using Random
import MathOptInterface     # for OPTIMAL

const MOI = MathOptInterface

import SparseMaxSR.CuttingPlanesUtils:  
    inner_dual,
    hillclimb,
    portfolios_socp,
    portfolios_objective,
    warm_start,
    cplex_misocp_relaxation,
    kelley_primal_cuts

###############################################
###############################################

@testset "inner_dual" begin
    # a tiny 3‐asset test
    μ   = [0.1, 0.2, 0.15]
    Σ   = Matrix{Float64}(I, 3, 3)     # identity covariance
    inds = [1, 2]                      # support of size 2

    # call the function under test
    result = inner_dual(μ, Σ, inds)

    @test result.status == MOI.OPTIMAL
    @test isa(result.ofv, Float64) && isfinite(result.ofv)

    @test length(result.α) == length(μ)
    @test length(result.w) == length(inds)

    # check complementary slackness: for j∉inds, w not defined;
    # for j∈inds, w[j] should satisfy w[j] ≥ Σ*α + λ
    Σα = Σ * result.α
    for (j,i) in enumerate(inds)
        @test result.w[j] + 1e-8 ≥ Σα[i] + result.λ
    end
end

###############################################
###############################################

@testset "hillclimb" begin
    # simple 3‐asset problem
    μ    = [0.1, 0.2, 0.15]
    Σ    = Matrix{Float64}(I, 3, 3)    # identity covariance
    k    = 2
    inds0 = [1, 2]

    # run hillclimb
    inds, w_full = hillclimb(μ, Σ, k, inds0; maxiter=10)

    # 1) support has size k
    @test length(inds) == k

    # 2) w_full is length n, zero off‐support, positive on‐support
    @test length(w_full) == length(μ)
    @test all(w_full[i] == 0.0 for i in setdiff(1:length(μ), inds))
    @test all(w_full[i] >  0.0 for i in inds)

    # 3) selected inds must be the top-k entries of w_full
    top_inds = sortperm(w_full, rev=true)[1:k]
    @test Set(inds) == Set(top_inds)
end

###############################################
###############################################

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

###############################################
###############################################

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
    @test sum(s .> 0.5) ≤ k

    # since ∇sᵢ = -½·γᵢ·wᵢ² ≤ 0, gradients should be non-positive
    @test all(cut.grad .≤ 0.0)
end

###############################################
###############################################

@testset "warm_start" begin
  μ = [0.1,0.2,0.15,0.05]
  n = length(μ)
  Σ = Matrix{Float64}(I, n, n)
  γ = ones(n)
  k = 2

  s0 = warm_start(μ, Σ, γ, k; num_random_restarts=3)

  @test length(s0) == n
  @test sum(s0 .> 0.5) == k
  @test all(v -> v in (0.0,1.0), s0)
end

###############################################
###############################################

@testset "cplex_misocp_relaxation" begin
    n, k = 5, 3
    z = cplex_misocp_relaxation(n, k; ΔT_max=1.0)

    @test length(z) == n
    @test all(0.0 .≤ z .≤ 1.0)
    @test isapprox(sum(z), k; atol=1e-6)

    # Optionally, check solver status too:
    # (you'd have to modify the function to also return the status)
    # status = last_solver_status()
    # @test status == MOI.OPTIMAL
end

###############################################
###############################################


@testset "kelley_primal_cuts" begin
    μ = [0.1, 0.2, 0.15, 0.05]
    n = length(μ)
    Σ = Matrix{Float64}(I, n, n)
    γ = ones(n)
    k = 2

    # start at the zero vector
    stab0 = zeros(n)

    cuts = cplex_misocp_relaxation # ensure this is loaded first
    cuts = kelley_primal_cuts(μ, Σ, γ, k, stab0, 3)

    @test isa(cuts, Vector)
    @test all( isa(cut.p, Float64) && isfinite(cut.p) for cut in cuts )
    @test all( length(cut.grad) == n for cut in cuts )
    @test all( cut.status == MOI.OPTIMAL for cut in cuts )
    @test length(cuts) ≤ 3
end


