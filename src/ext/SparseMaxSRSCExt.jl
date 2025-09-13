module SparseMaxSRSCExt

import SCS
import SparseMaxSR: set_default_optimizer!

# Set SCS as the default conic optimizer for dual subproblems.
function __init__()
    set_default_optimizer!(() -> SCS.Optimizer())
end

end # module
