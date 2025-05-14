using Test
using SparseMaxSR
using Random

@testset "SparseMaxSR smoke tests" begin
    rng = MersenneTwister(42)
    X = rand(rng, 5, 5)
    @test typeof(X) == Matrix{Float64}
end