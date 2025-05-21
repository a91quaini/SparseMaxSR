using JuMP
using MosekTools    # for Mosek.Optimizer

"""
    inner_dual(μ, Σ, inds)

Solve the dual QP for a fixed support `inds` (|inds| = k) in the k-sparse max-Sharpe problem.

# Arguments
- `μ::Vector{Float64}` : expected‐return vector (length n)
- `Σ::Matrix{Float64}` : covariance matrix (n×n)
- `inds::Vector{Int}` : indices of the k nonzero assets

# Returns
A NamedTuple with fields
- `ofv`    : objective value (dual bound)
- `α`      : dual vector α ∈ ℝⁿ
- `λ`      : scalar multiplier λ
- `w`      : cut‐slack vector w ∈ ℝᵏ
- `status` : termination status (`MOI.OPTIMAL`, etc.)
"""
function inner_dual(μ::Vector{Float64},
                    Σ::Matrix{Float64},
                    inds::Vector{Int})
    n = length(μ)
    k = length(inds)

    model = Model(optimizer_with_attributes(
        Mosek.Optimizer,
        # high-precision interior‐point tolerances
        "MSK_DPAR_INTPNT_QO_TOL_PFEAS" => 1e-8,
        "MSK_DPAR_INTPNT_QO_TOL_DFEAS" => 1e-8,
        "MSK_IPAR_LOG"               => 0,
        "MSK_IPAR_MAX_NUM_WARNINGS"  => 0
    ))

    # dual variables
    @variable(model, α[1:n])
    @variable(model, λ)
    @variable(model, w[1:k])

    # precompute Σ*α so we can index into it
    Σα = Σ * α

    # for each j=1:k, asset i = inds[j]:
    #   w[j] ≥ Σα[i] + λ
    @constraint(model, [j=1:k], w[j] ≥ Σα[inds[j]] + λ)

    # objective: Max  −½‖α‖²  −½‖w‖²  + μᵀα + λ
    @objective(model, Max,
        -0.5 * dot(α, α)
        -0.5 * sum(w[j]^2 for j = 1:k)
        + dot(μ, α)
        + λ
    )

    optimize!(model)

    return (
        ofv    = objective_value(model),
        α      = value.(α),
        λ      = value(λ),
        w      = value.(w),
        status = termination_status(model),
    )
end
