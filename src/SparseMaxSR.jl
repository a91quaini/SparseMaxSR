module SparseMaxSR

# Top-level module wiring for SparseMaxSR.
# Includes submodules and re-exports the public API.

# --- Include submodules ---
include("SparseMaxSR/Utils.jl")
include("SparseMaxSR/SharpeRatio.jl")
include("SparseMaxSR/ExhaustiveSearch.jl")
include("SparseMaxSR/MIQPHeuristicSearch.jl")
include("SparseMaxSR/LassoRelaxationSearch.jl")

# Bring submodules into this namespace
using .Utils
using .SharpeRatio
using .ExhaustiveSearch
using .MIQPHeuristicSearch
using .LassoRelaxationSearch

# --- Re-exports ---

# Core Sharpe/Mean-Variance routines
export compute_sr, compute_mve_sr, compute_mve_weights

# Search front-ends
export mve_exhaustive_search,
       mve_exhaustive_search_gridk,
       mve_miqp_heuristic_search,
       mve_lasso_relaxation_search

# Nothing special to do at initialization; Utils.__init__ sets safe defaults.
function __init__() end

end # module
