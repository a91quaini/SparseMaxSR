module SparseMaxSR

using JuMP, CPLEX, MosekTools, LinearAlgebra, Random

include("inner_problem.jl")
include("hillclimb.jl")
include("socp_relaxation.jl")
include("cutting_plane.jl")

export
  inner_dual,
  portfolios_hillclimb,
  portfolios_socp,
  cutting_plane_portfolio

end
