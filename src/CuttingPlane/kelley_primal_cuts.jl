using JuMP
using CPLEX                              # for CPLEX.Optimizer
import MathOptInterface                  # for MOI.OPTIMAL if needed
const MOI = MathOptInterface
using SparseMaxSR.CuttingPlane: portfolios_objective
using LinearAlgebra                      # for clamp, dot

"""
    kelley_primal_cuts(μ, Σ, γ, k, stab0, num_epochs; eps=1e-10)

Generate up to `num_epochs` outer‐approximation cuts at the root node
via a simple “in–out” stabilization (λ=0.1, δ=2·eps).

# Arguments
- `μ::Vector{Float64}` : expected returns, length n  
- `Σ::Matrix{Float64}` : covariance, n×n  
- `γ::Vector{Float64}` : per‐asset penalty weights, length n  
- `k::Int`            : cardinality bound  
- `stab0::Vector{Float64}` : initial stabilization point (∈[0,1]ⁿ)  
- `num_epochs::Int`   : how many root passes  
- `eps::Float64`      : tolerance for stabilization (default 1e-10)

# Returns
An array of NamedTuples `(p, grad, status)` (in the same format as
`portfolios_objective`) representing the cuts generated.
"""
function kelley_primal_cuts(μ::Vector{Float64},
                            Σ::Matrix{Float64},
                            γ::Vector{Float64},
                            k::Int,
                            stab0::Vector{Float64},
                            num_epochs::Int;
                            eps::Float64 = 1e-10)

    n = length(μ)
    @assert size(Σ) == (n,n)
    @assert length(γ) == n
    @assert length(stab0) == n
    @assert 1 ≤ k ≤ n

    # build the root model
    model = Model(optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPX_PARAM_SCRIND" => 0
    ))
    @variable(model, 0 ≤ s[1:n] ≤ 1)
    @variable(model, t ≥ -1e12)
    @constraint(model, sum(s) ≤ k)
    @constraint(model, sum(s) ≥ 1)
    @objective(model, Min, t)

    # in–out stabilization parameters
    λ = 0.1
    δ = 2eps

    LB = -Inf
    UB = Inf

    cuts = Vector{NamedTuple{(:p,:grad,:status),Tuple{Float64,Vector{Float64},Symbol}}}()

    for epoch in 1:num_epochs
        optimize!(model)

        # raw solution & update lower bound
        objval = objective_value(model)
        zstar = clamp.(value.(s), 0.0, 1.0)
        LB = max(LB, objval)

        # stabilized point
        stab0 .= (stab0 .+ zstar) ./ 2

        # build the next trial point
        z0 = clamp.(λ .* zstar .+ (1-λ) .* stab0 .+ δ, 0.0, 1.0)

        # evaluate cut at z0
        cut = portfolios_objective(μ, Σ, γ, k, z0)

        if cut.status == MOI.OPTIMAL
            UB = min(UB, cut.p)
            @constraint(model, t ≥ cut.p + dot(cut.grad, s .- z0))
            push!(cuts, (p = cut.p, grad = cut.grad, status = cut.status))
        else
            # feasibility cut
            @constraint(model,
                sum(z0[i]*(1 - s[i]) + s[i]*(1 - z0[i]) for i=1:n) ≥ 1.0
            )
        end
    end

    return cuts
end