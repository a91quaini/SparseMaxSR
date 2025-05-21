using Test
using SparseMaxSR           # your package
using LinearAlgebra         # for I
import MathOptInterface     # for OPTIMAL
const MOI = MathOptInterface

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

