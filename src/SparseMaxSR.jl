module SparseMaxSR

# bring in submodules
include("SparseMaxSR/SharpeRatio.jl")
using .SharpeRatio: compute_sr, compute_mve_sr

include("SparseMaxSR/CuttingPlanesUtils.jl")
using .CuttingPlanesUtils: 
    inner_dual, 
    hillclimb, 
    portfolios_socp, 
    portfolios_objective,                     
    warm_start, 
    cplex_misocp_relaxation, 
    kelley_primal_cuts

include("SparseMaxSR/CuttingPlanesSelection.jl")
using .CuttingPlanesSelection: cutting_planes_selection

    # export core functions
export compute_sr, 
    compute_mve_sr, 
    cutting_planes_selection

end # module
