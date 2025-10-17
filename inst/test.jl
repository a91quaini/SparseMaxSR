using SparseMaxSR
using Random

μ = randn(5)
Σ = Matrix{Float64}(I, 5, 5)  # or: Matrix(I,5,5) .* 1.0
SparseMaxSR.mve_miqp_heuristic_search(μ, Σ; k=2)