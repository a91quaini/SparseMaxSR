module SparseMaxSRGurobiExt

import Gurobi
import SparseMaxSR: set_default_optimizer!

# Use Gurobi for duals when available. (MILP master can also use Gurobi.)
function __init__()
    set_default_optimizer!(() -> Gurobi.Optimizer())
end

end # module
