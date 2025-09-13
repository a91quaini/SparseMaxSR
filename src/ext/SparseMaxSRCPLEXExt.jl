module SparseMaxSRCPLEXExt

import CPLEX
import SparseMaxSR: set_default_optimizer!

# Set CPLEX as the default optimizer for dual subproblems (QP/QCQP capable).
# (For the MILP master, still pass optimizer=() -> CPLEX.Optimizer() in your call.)
function __init__()
    set_default_optimizer!(() -> CPLEX.Optimizer())
end

end # module
