module SparseMaxSRCOSMOExt

import COSMO
import SparseMaxSR: set_default_optimizer!

# Set COSMO as the default conic optimizer for dual subproblems.
function __init__()
    set_default_optimizer!(() -> COSMO.Optimizer())
end

end # module
