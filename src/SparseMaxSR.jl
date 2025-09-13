module SparseMaxSR
"""
SparseMaxSR — sparse maximum-Sharpe portfolio utilities.

Public API re-exported from submodules:

  - `compute_sr`              -- Sharpe ratio of a weight vector
  - `compute_mve_sr`          -- Sharpe ratio for a given support/selection
  - `compute_mve_weights`     -- mean–variance efficient weights on a support
  - `compute_mve_selection`   -- select k assets (exhaustive / cutting-planes)

Implementation details live in:
  - `SharpeRatio`         (numerics for SR and MVE weights)
  - `CuttingPlanesUtils`  (cuts, relaxations, warm starts, local search)
  - `MVESelection`        (exhaustive & cutting-planes selection)
"""

# ──────────────────────────────────────────────────────────────────────────────
# Package-wide defaults (defined BEFORE including submodules)
# ──────────────────────────────────────────────────────────────────────────────

"Global default ridge ε (dimensionless). Read at call time via `EPS_RIDGE[]`."
const EPS_RIDGE = Base.RefValue{Float64}(1e-6)

"Set the package-wide default ridge ε (clamped to ≥ 0)."
set_default_ridge!(ε::Real) = (EPS_RIDGE[] = max(0.0, float(ε)); nothing)

"Get the package-wide default ridge ε."
get_default_ridge() = EPS_RIDGE[]

# Default optimizer factory registry (e.g., `() -> Clarabel.Optimizer()`).
const _DEFAULT_OPT = Base.RefValue{Union{Nothing,Function}}(nothing)

"""
    set_default_optimizer!(factory::Function)

Set the package-wide default optimizer **factory**, e.g.

```julia
using Clarabel
SparseMaxSR.set_default_optimizer!(() -> Clarabel.Optimizer())
```
This factory is used by functions when `optimizer === nothing`.
"""
set_default_optimizer!(factory::Function) = (_DEFAULT_OPT[] = factory; nothing)

"""
    default_optimizer() -> Function

Return the currently set optimizer factory. If none is set, try best-effort
fallbacks (Clarabel, COSMO, SCS) if available; otherwise throw an error.
"""
function default_optimizer()::Function
    _DEFAULT_OPT[] !== nothing && return _DEFAULT_OPT[]  # user-provided

    # Best-effort fallbacks by availability
    if Base.find_package("Clarabel") !== nothing
        @eval import Clarabel
        return () -> Clarabel.Optimizer()
    elseif Base.find_package("COSMO") !== nothing
        @eval import COSMO
        return () -> COSMO.Optimizer()
    elseif Base.find_package("SCS") !== nothing
        @eval import SCS
        return () -> SCS.Optimizer()
    else
        error("No default optimizer set. Call SparseMaxSR.set_default_optimizer!(...) or pass optimizer= ...")
    end
end

export EPS_RIDGE, set_default_ridge!, get_default_ridge,
       set_default_optimizer!, default_optimizer

# ──────────────────────────────────────────────────────────────────────────────
# Includes (paths relative to this file)
# ──────────────────────────────────────────────────────────────────────────────

const _SRC_DIR = @__DIR__

include(joinpath(_SRC_DIR, "SparseMaxSR", "SharpeRatio.jl"))
include(joinpath(_SRC_DIR, "SparseMaxSR", "CuttingPlanesUtils.jl"))
include(joinpath(_SRC_DIR, "SparseMaxSR", "MVESelection.jl"))

# ──────────────────────────────────────────────────────────────────────────────
# Import from submodules and re-export public API
# ──────────────────────────────────────────────────────────────────────────────

import .SharpeRatio: compute_sr, compute_mve_sr, compute_mve_weights
import .CuttingPlanesUtils: inner_dual, hillclimb, portfolios_socp,
                            portfolios_objective, warm_start,
                            cplex_misocp_relaxation, kelley_primal_cuts
import .MVESelection: compute_mve_selection,
                      mve_selection_exhaustive_search,
                      mve_selection_cutting_planes

export compute_sr, compute_mve_sr, compute_mve_weights,
       compute_mve_selection

# Optional initialization hook
function __init__() end

end # module SparseMaxSR