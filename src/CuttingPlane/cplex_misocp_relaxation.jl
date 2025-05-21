using JuMP
using CPLEX                              # CPLEX.Optimizer
import MathOptInterface                  # for termination_status if you need it
const MOI = MathOptInterface

"""
    cplex_misocp_relaxation(n, k; ΔT_max=3600.0)

Continuous relaxation of the ℓ₀‐constraint ‖z‖₀ ≤ k by
allowing z ∈ [0,1]ⁿ, enforcing ∑ zᵢ ≤ k, and then maximizing ∑ zᵢ
so that ∑ zᵢ ≈ k at optimality.

# Arguments
- `n::Int` : number of assets  
- `k::Int` : cardinality upper bound (1 ≤ k ≤ n)  
- `ΔT_max::Float64` : time limit (seconds)

# Returns
- `z::Vector{Float64}` : continuous indicator vector of length n,
   with ∑ z ≈ k.
"""
function cplex_misocp_relaxation(n::Int, k::Int; ΔT_max::Float64 = 3600.0)
    @assert 1 ≤ k ≤ n "k must lie between 1 and n"

    model = Model(optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPX_PARAM_SCRIND" => 0,       # suppress screen output
        "CPX_PARAM_TILIM"  => ΔT_max
    ))

    # z_i ∈ [0,1]
    @variable(model, 0 ≤ z[1:n] ≤ 1)
    # cardinality relaxed: sum(z_i) ≤ k
    @constraint(model, sum(z) ≤ k)

    # maximize ∑ z_i so solver pushes ∑ z_i up to k
    @objective(model, Max, sum(z))

    optimize!(model)

    return value.(z)
end
