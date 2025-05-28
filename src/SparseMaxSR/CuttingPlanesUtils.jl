module CuttingPlanesUtils
# this module contains useful functions for computing the 
# mve portfolio selection using the cutting planes approach
# in Bertsimas & Cory-Wright (2022) 

using JuMP
using CPLEX
using MosekTools
using Random
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface

export inner_dual,
       hillclimb,
       portfolios_socp,
       portfolios_objective,
       warm_start,
       cplex_misocp_relaxation,
       kelley_primal_cuts

#####################################
#### inner dual
#####################################

"""
    inner_dual(μ, Σ, inds)

Solve the dual QP for a fixed support `inds` (|inds| = k) in the k-sparse max-Sharpe problem.

# Arguments
- `μ::AbstractVector{Float64}` : expected‐return vector (length n)
- `Σ::AbstractMatrix{Float64}` : covariance matrix (n×n)
- `inds::AbstractVector{Int}` : indices of the k nonzero assets

# Returns
A NamedTuple with fields
- `ofv`    : objective value (dual bound)
- `α`      : dual vector α ∈ ℝⁿ
- `λ`      : scalar multiplier λ
- `w`      : cut‐slack vector w ∈ ℝᵏ
- `status` : termination status (`MOI.OPTIMAL`, etc.)
"""
function inner_dual(μ::AbstractVector{Float64},
                    Σ::AbstractMatrix{Float64},
                    inds::AbstractVector{Int})
    n = length(μ)
    k = length(inds)

    model = Model(optimizer_with_attributes(
        Mosek.Optimizer,
        # high-precision interior‐point tolerances
        "MSK_DPAR_INTPNT_QO_TOL_PFEAS" => 1e-8,
        "MSK_DPAR_INTPNT_QO_TOL_DFEAS" => 1e-8,
        "MSK_IPAR_LOG"               => 0,
        "MSK_IPAR_MAX_NUM_WARNINGS"  => 0,
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

#####################################
#### hillclimb
#####################################

"""
    hillclimb(μ, Σ, k, inds0; maxiter=50)

Simple discrete first-order heuristic for the ℓ₀-constrained max-Sharpe problem:

1. Solve dual QP on support `inds`.
2. Form full w-vector (wᵢ=0 off-support).
3. Pick top-k assets by wᵢ → new support.
4. Repeat until convergence or `maxiter`.

# Arguments
- `μ::AbstractVector{Float64}` : expected returns (length n)
- `Σ::AbstractMatrix{Float64}` : covariance matrix (n×n)
- `k::Int`            : desired number of nonzeros
- `inds0::AbstractVector{Int}`: initial support of size k
- `maxiter::Int`      : maximum number of hill-climb iterations

# Returns
- `inds::AbstractVector{Int}` : final support (size k)
- `w_full::AbstractVector{Float64}` : full-length slack vector (wᵢ=0 if i∉inds)
"""
function hillclimb(μ::AbstractVector{Float64},
                              Σ::AbstractMatrix{Float64},
                              k::Int,
                              inds0::AbstractVector{Int};
                              maxiter::Int = 50)

    # n = length(μ)
    # @assert size(Σ) == (n,n)
    # @assert length(inds0) == k

    inds = copy(inds0)
    w_full = zeros(length(μ))

    for iter in 1:maxiter
        # 1) solve dual on current support
        res = inner_dual(μ, Σ, inds)
        @assert res.status == MathOptInterface.OPTIMAL

        # 2) build full slack vector
        w_full .= 0
        w_full[inds] = res.w

        # 3) pick the k largest by w_full
        new_inds = sortperm(w_full, rev=true)[1:k]

        # 4) check convergence
        if new_inds == inds
            break
        end

        inds .= new_inds
    end

    return inds, w_full
end

#####################################
#### socp relaxation
#####################################

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
- `μ::AbstractVector{Float64}` : expected‐return vector, length n  
- `Σ::AbstractMatrix{Float64}` : covariance matrix, n×n  
- `γ::AbstractVector{Float64}` : per‐asset penalty weights, length n  
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
function portfolios_socp(μ::AbstractVector{Float64},
                         Σ::AbstractMatrix{Float64},
                         γ::AbstractVector{Float64},
                         k::Int)

    n = length(μ)
    # @assert size(Σ) == (n,n) "Σ must be n×n"
    # @assert length(γ) == n  "γ must be length n"

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

#####################################
#### portfolios objective
#####################################

"""
    portfolios_objective(μ, Σ, γ, k, s) -> NamedTuples

Given a 0–1 vector `s` of length `n`, form the outer-approximation cut
for the k-sparse max-Sharpe QP:

1. Extract `inds = findall(x->x>0.5, s)`.  
2. Solve the dual QP on that support via `inner_dual(μ,Σ,inds)`.  
3. Let `p = dual.ofv`, `α = dual.α`, `λ = dual.λ`.  
4. Compute `w_full = Σ*α .+ λ`.  
5. The gradient is  ∇ₛᵢ = −½·γᵢ·w_full[i]^2.  

# Arguments
- `μ::AbstractVector{Float64}` : expected returns (length n)
- `Σ::AbstractMatrix{Float64}` : covariance matrix (n×n)
- `γ::AbstractVector{Float64}` : per-asset cut weights (length n)
- `k::Int`            : cardinality bound (must equal `sum(s)`)
- `s::AbstractVector{Float64}`: 0-1 indicator (length n)

# Returns
- An array of NamedTuples `(p, grad, status)` where `status` is the MOI termination symbol.
"""
function portfolios_objective(μ::AbstractVector{Float64},
                              Σ::AbstractMatrix{Float64},
                              γ::AbstractVector{Float64},
                              k::Int,
                              s::AbstractVector{Float64})

    n = length(s)
    # @assert length(μ) == n          "μ must be length n"
    # @assert size(Σ) == (n,n)        "Σ must be n×n"
    # @assert length(γ) == n          "γ must be length n"
    @assert sum(s .> 0.5) ≤ k       "sum(s)>0.5 must be equal or less than k"

    # 1) Which assets are “on”?
    inds = findall(x-> x > 0.5, s)

    # 2) Solve the dual QP on that support
    dual = inner_dual(μ, Σ, inds)

    # 3) Build full slack vector w_i = (Σ α)_i + λ
    w_full = Σ*dual.α .+ dual.λ

    # 4) Gradient wrt each s_i: -½ γ_i w_i^2
    ∇s = @. -0.5 * γ * w_full^2

    return (p = dual.ofv, grad = ∇s, status = dual.status)
end

#####################################
#### warm start
#####################################

"""
    warm_start(μ, Σ, γ, k; num_random_restarts=5)

Generate a good initial 0–1 portfolio `s` with ∥s∥₀ = k by
random restarts plus hill‐climbing.

# Arguments
- `μ::AbstractVector{Float64}` : expected‐return vector
- `Σ::AbstractMatrix{Float64}` : covariance matrix
- `γ::AbstractVector{Float64}` : per‐asset penalty weights
- `k::Int`            : cardinality bound
- `num_random_restarts::Int` : how many random seeds to try (default 5)

# Returns
- `s0::AbstractVector{Float64}` : a binary vector of length `length(μ)` with exactly `k` ones
"""
function warm_start(μ::AbstractVector{Float64},
                        Σ::AbstractMatrix{Float64},
                        γ::AbstractVector{Float64},
                        k::Int;
                        num_random_restarts::Int = 5)

    n = length(μ)
    # @assert size(Σ) == (n,n)  "Σ must be n×n"
    # @assert length(γ) == n    "γ must be length n"
    # @assert 1 ≤ k ≤ n         "k must be between 1 and n"

    best_obj = Inf
    best_s   = zeros(Float64, n)

    for _ in 1:num_random_restarts
        # 1) pick a random support of size k
        init_inds = sort(shuffle(1:n)[1:k])

        # 2) hill‐climb to improve it
        inds, _ = hillclimb(μ, Σ, k, init_inds; maxiter=50)

        # 3) form the binary vector
        s = zeros(Float64, n)
        s[inds] .= 1.0

        # 4) evaluate its cut‐value
        cut = portfolios_objective(μ, Σ, γ, k, s)

        # 5) keep the best
        if cut.p < best_obj
            best_obj = cut.p
            best_s   = copy(s)
        end
    end

    return best_s
end

#####################################
#### cplex misocp relaxation
#####################################

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
- `z::AbstractVector{Float64}` : continuous indicator vector of length n,
   with ∑ z ≈ k.
"""
function cplex_misocp_relaxation(n::Int, k::Int; ΔT_max::Float64 = 3600.0)
    # @assert 1 ≤ k ≤ n "k must lie between 1 and n"

    model = Model(optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPX_PARAM_SCRIND" => 0,       # suppress screen output
        "CPX_PARAM_TILIM"  => ΔT_max,
        "CPX_PARAM_SCRIND" => 0,
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


#####################################
#### kelley primal cuts
#####################################

"""
    kelley_primal_cuts(μ, Σ, γ, k, stab0, num_epochs; eps=1e-10)

Generate up to `num_epochs` outer‐approximation cuts at the root node
via a simple “in–out” stabilization (λ=0.1, δ=2·eps).

# Arguments
- `μ::AbstractVector{Float64}` : expected returns, length n  
- `Σ::AbstractMatrix{Float64}` : covariance, n×n  
- `γ::AbstractVector{Float64}` : per‐asset penalty weights, length n  
- `k::Int`            : cardinality bound  
- `stab0::AbstractVector{Float64}` : initial stabilization point (∈[0,1]ⁿ)  
- `num_epochs::Int`   : how many root passes  
- `eps::Float64`      : tolerance for stabilization (default 1e-10)

# Returns
An array of NamedTuples `(p, grad, status)` (in the same format as
`portfolios_objective`) representing the cuts generated.
"""
function kelley_primal_cuts(μ::AbstractVector{Float64},
                            Σ::AbstractMatrix{Float64},
                            γ::AbstractVector{Float64},
                            k::Int,
                            stab0::AbstractVector{Float64},
                            num_epochs::Int;
                            eps::Float64 = 1e-10)

    n = length(μ)
    # @assert size(Σ) == (n,n)
    # @assert length(γ) == n
    # @assert length(stab0) == n
    # @assert 1 ≤ k           "k must be at least 1"
    # @assert k ≤ n           "k must be at most n"

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

    # collect cuts
    cuts = []

    for _ in 1:num_epochs
        optimize!(model)
        zstar = clamp.(value.(s), 0.0, 1.0)

        # stabilized point
        stab0 .= (stab0 .+ zstar) ./ 2

        # build the next trial point
        z0 = clamp.(λ .* zstar .+ (1-λ) .* stab0 .+ δ, 0.0, 1.0)

        # evaluate cut at z0
        cut = portfolios_objective(μ, Σ, γ, k, z0)

        if cut.status == MOI.OPTIMAL
            @constraint(model, t ≥ cut.p + dot(cut.grad, s .- z0))
            push!(cuts, cut)
        else
            # feasibility cut
            @constraint(model,
                sum(z0[i]*(1 - s[i]) + s[i]*(1 - z0[i]) for i=1:n) ≥ 1.0
            )
        end
    end

    return cuts
end


end # end module