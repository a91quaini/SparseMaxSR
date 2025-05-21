using JuMP
using MosekTools       # for Mosek.Optimizer
import MathOptInterface             # for termination_status
const MOI = MathOptInterface

"""
    portfolios_socp(μ, Σ, γ, k)

QCQP‐relaxation of the k-sparse max-Sharpe portfolio problem:

Maximize over (α, λ, w, v, t):
  -½‖α‖² + μᵀα + λ - ∑ v_i - k·t

Subject to:
  w_i ≥ (Σα)_i + λ
  v_i + t ≥ (γ_i/2)*w_i^2
  v_i, t ≥ 0

# Arguments
- `μ::Vector{Float64}` : expected‐return vector, length n  
- `Σ::Matrix{Float64}` : covariance matrix, n×n  
- `γ::Vector{Float64}` : per‐asset penalty weights, length n  
- `k::Int`            : max number of nonzeros (cardinality)

# Returns
A NamedTuple with fields
- `ofv`    : objective value  
- `α`      : dual vector α ∈ ℝⁿ  
- `λ`      : scalar multiplier λ  
- `w`      : slack vector w ∈ ℝⁿ  
- `v`      : slack vector v ∈ ℝⁿ  
- `t`      : scalar t ≥ 0  
- `status` : termination status
"""
function portfolios_socp(μ::Vector{Float64},
                         Σ::Matrix{Float64},
                         γ::Vector{Float64},
                         k::Int)

    n = length(μ)
    @assert size(Σ) == (n,n) "Σ must be n×n"
    @assert length(γ) == n  "γ must be length n"

    model = Model(optimizer_with_attributes(
        Mosek.Optimizer,
        "MSK_DPAR_INTPNT_QO_TOL_PFEAS"    => 1e-8,
        "MSK_DPAR_INTPNT_QO_TOL_DFEAS"    => 1e-8,
        "MSK_IPAR_LOG"                    => 0,
        "MSK_IPAR_MAX_NUM_WARNINGS"       => 0
    ))

    # decision variables
    @variable(model, α[1:n])
    @variable(model, λ)
    @variable(model, w[1:n])
    @variable(model, v[1:n] >= 0)
    @variable(model, t      >= 0)

    # precompute Σ*α
    Σα = Σ * α

    # cut constraints: w_i ≥ Σα[i] + λ
    @constraint(model, [i=1:n], w[i] ≥ Σα[i] + λ)

    # QCQP constraints: v_i + t ≥ (γ[i]/2)*w[i]^2
    @constraint(model, [i=1:n], v[i] + t ≥ (γ[i] / 2) * w[i]^2)

    # objective: Max -½‖α‖² + μᵀα + λ - ∑v_i - k·t
    @objective(model, Max,
        -0.5 * dot(α, α)
        + dot(μ, α)
        + λ
        - sum(v)
        - k * t
    )

    optimize!(model)

    return (
        ofv    = objective_value(model),
        α      = value.(α),
        λ      = value(λ),
        w      = value.(w),
        v      = value.(v),
        t      = value(t),
        status = termination_status(model),
    )
end