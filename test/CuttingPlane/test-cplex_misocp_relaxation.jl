using Test
using SparseMaxSR            # your package
import MathOptInterface
const MOI = MathOptInterface

@testset "cplex_misocp_relaxation" begin
    n, k = 5, 3
    z = cplex_misocp_relaxation(n, k; ΔT_max=1.0)

    @test length(z) == n
    @test all(0.0 .≤ z .≤ 1.0)
    @test isapprox(sum(z), k; atol=1e-6)
end
