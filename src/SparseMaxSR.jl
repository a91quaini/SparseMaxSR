module SparseMaxSR

# bring in the CuttingPlane submodule
include("CuttingPlane.jl")
using .CuttingPlane

# now re-export everything (or selectively if you prefer)
export inner_dual,
       hillclimb,
       portfolios_socp,
       portfolios_objective,
       get_warm_start,
       cplex_misocp_relaxation,
       kelley_primal_cuts,
       cutting_planes_portfolios

end # module
