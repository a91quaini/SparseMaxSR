using Test
using SparseMaxSR

@testset "SparseMaxSR" begin
    include("test-SharpeRatio.jl")
    include("test-ExhaustiveSearch.jl")
    include("test-MIQPHeuristicSearch.jl")
    include("test-LassoRelaxationSearch.jl")
end
