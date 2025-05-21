module SparseMaxSR

# bring in the CuttingPlane submodule
include("CuttingPlanes.jl")

export cutting_planes_selection
import .CuttingPlanes: cutting_planes_selection

end # module
