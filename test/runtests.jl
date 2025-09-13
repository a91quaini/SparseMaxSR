using Test
using SparseMaxSR

@testset "SparseMaxSR" begin
    include("test-SharpeRatio.jl")
    include("test-CuttingPlanesUtils.jl")
    include("test-MVESelection.jl")
end
