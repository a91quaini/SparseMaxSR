module SparseMaxSRMosekToolsExt

import MosekTools
import SparseMaxSR: set_default_optimizer!

# Set MosekTools as the default optimizer for dual subproblems (QP/SOCP/QCQP).
function __init__()
    set_default_optimizer!(() -> MosekTools.Optimizer())
end

end # module
