using LinearAlgebra          # for findall, broadcasted arithmetic
import MathOptInterface      # for OPTIMAL, if you need to inspect status
const MOI = MathOptInterface

"""
    portfolios_objective(μ, Σ, γ, k, s) -> Cut

Given a 0–1 vector `s` of length `n`, form the outer-approximation cut
for the k-sparse max-Sharpe QP:

1. Extract `inds = findall(x->x>0.5, s)`.  
2. Solve the dual QP on that support via `inner_dual(μ,Σ,inds)`.  
3. Let `p = dual.ofv`, `α = dual.α`, `λ = dual.λ`.  
4. Compute `w_full = Σ*α .+ λ`.  
5. The gradient is  ∇ₛᵢ = −½·γᵢ·w_full[i]^2.  

# Arguments
- `μ::Vector{Float64}` : expected returns (length n)
- `Σ::Matrix{Float64}` : covariance matrix (n×n)
- `γ::Vector{Float64}` : per-asset cut weights (length n)
- `k::Int`            : cardinality bound (must equal `sum(s)`)
- `s::Vector{Float64}`: 0-1 indicator (length n)

# Returns
- `Cut(p, ∇s, status)` where `status` is the MOI termination symbol.
"""
function portfolios_objective(μ::Vector{Float64},
                              Σ::Matrix{Float64},
                              γ::Vector{Float64},
                              k::Int,
                              s::Vector{Float64})

    n = length(s)
    @assert length(μ) == n          "μ must be length n"
    @assert size(Σ) == (n,n)        "Σ must be n×n"
    @assert length(γ) == n          "γ must be length n"
    @assert sum(s .> 0.5) == k      "sum(s)>0.5 must equal k"

    # 1) Which assets are “on”?
    inds = findall(x-> x > 0.5, s)

    # 2) Solve the dual QP on that support
    dual = inner_dual(μ, Σ, inds)
    p    = dual.ofv
    α    = dual.α
    λ    = dual.λ
    status = dual.status

    # 3) Build full slack vector w_i = (Σ α)_i + λ
    w_full = Σ*α .+ λ

    # 4) Gradient wrt each s_i: -½ γ_i w_i^2
    ∇s = @. -0.5 * γ * w_full^2

    return (p = p, grad = ∇s, status = status)
end