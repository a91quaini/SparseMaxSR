module SparseMaxSRClarabelExt

import Clarabel
import SparseMaxSR: set_default_optimizer!

# Set Clarabel as the default conic/QP optimizer for dual subproblems.
function __init__()
    set_default_optimizer!(() -> Clarabel.Optimizer())
end

end # module
