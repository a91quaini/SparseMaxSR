module SparseMaxSR

# bring in submodules
include("SparseMaxSR/CuttingPlanesUtils.jl")
using .CuttingPlanesUtils: inner_dual,
                           hillclimb,
                           portfolios_socp,
                           portfolios_objective,
                           warm_start,
                           cplex_misocp_relaxation,
                           kelley_primal_cuts

include("SparseMaxSR/MVESelection.jl")
using .MVESelection: compute_mve_selection

include("SparseMaxSR/SharpeRatio.jl")
using .SharpeRatio: compute_sr, 
                    compute_mve_sr, 
                    compute_mve_weights,
                    compute_mve_sr_decomposition,
                    simulate_mve_sr

# export core functions
export compute_mve_selection, 
       compute_sr, 
       compute_mve_sr,
       compute_mve_weights,
       compute_mve_sr_decomposition,
       simulate_mve_sr

end # module
