using Test
using SparseMaxSR

@testset "SparseMaxSR" begin
    include("test-Utils.jl")
    include("test-SharpeRatio.jl")
    include("test-ExhaustiveSearch.jl")
    include("test-MIQPHeuristicSearch.jl")
    include("test-LassoRelaxationSearch.jl")
    # include("test-thread_safety.jl")
end
