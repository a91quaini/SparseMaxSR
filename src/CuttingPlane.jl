# src/CuttingPlane.jl
module CuttingPlane

using JuMP
# using JuMP: callback_node_status, callback_value
# using JuMP: @lazy_constraint, @user_cut, add_lazy_callback, add_cut_callback

using CPLEX
using MosekTools
using Random
using LinearAlgebra
import MathOptInterface                       # the core MOI package
const MOI = MathOptInterface                  # alias for convenience

# include all the implementation files from the subfolder:
include(joinpath(@__DIR__, "CuttingPlane", "inner_dual.jl"))
include(joinpath(@__DIR__, "CuttingPlane", "hillclimb.jl"))
include(joinpath(@__DIR__, "CuttingPlane", "socp_relaxation.jl"))
include(joinpath(@__DIR__, "CuttingPlane", "portfolios_objective.jl"))
include(joinpath(@__DIR__, "CuttingPlane", "get_warm_start.jl"))
include(joinpath(@__DIR__, "CuttingPlane", "cplex_misocp_relaxation.jl"))
include(joinpath(@__DIR__, "CuttingPlane", "kelley_primal_cuts.jl"))
include(joinpath(@__DIR__, "CuttingPlane", "cutting_planes_portfolios.jl"))

# re-export the functions you want users to see:
export inner_dual,
       hillclimb,
       portfolios_socp,
       portfolios_objective,
       get_warm_start,
       cplex_misocp_relaxation,
       kelley_primal_cuts,
       cutting_planes_portfolios


end # module
