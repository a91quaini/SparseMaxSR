module MVESelection
# this module contains the mve portfolio selection functions 
# based on exhaustive search of all combinations of n choose k assets
# and the cutting planes search of Bertsimas & Cory-Wright (2022) 

using LinearAlgebra
using Statistics
using Distributions
import Base.Iterators: combinations
using JuMP
using CPLEX                              # CPLEX.Optimizer
import MathOptInterface                  # for status constants and callbacks
const MOI = MathOptInterface

# bring in the cutting planes utils submodule:
using ..CuttingPlanesUtils:
    inner_dual,
    hillclimb,
    portfolios_socp,
    portfolios_objective,
    warm_start,
    cplex_misocp_relaxation,
    kelley_primal_cuts


using JuMP: @build_constraint, callback_value
import MathOptInterface: LazyConstraint, LazyConstraintCallback, submit

export compute_mve_selection

######################################
#### compute_mve_selection
######################################

"""
    compute_mve_selection(μ, Σ, k; exhaustive_threshold=20, kwargs...)

Selects a maximum Sharpe ratio portfolio with cardinality constraint `k`.
Uses exhaustive search if `n ≤ exhaustive_threshold`, otherwise falls back
to a cutting-plane outer approximation algorithm.

# Arguments
- `μ::Vector{Float64}`: Expected returns.
- `Σ::Matrix{Float64}`: Covariance matrix.
- `k::Int`: Cardinality constraint.
- `exhaustive_threshold::Int`: Upper bound on `n` for which exhaustive search is attempted.
- `kwargs`: Additional keyword arguments passed to each method.

# Returns
- `selection::Vector{Int}`: Optimal subset of selected asset indices.
"""
function compute_mve_selection(
    μ::Vector{Float64},
    Σ::Matrix{Float64},
    k::Int;
    exhaustive_threshold::Int=20,
    kwargs...
)
    n = length(μ)
    if n <= exhaustive_threshold
        return mve_selection_exhaustive_search(μ, Σ, k; kwargs...)
    else
        return mve_selection_cutting_planes(μ, Σ, k; kwargs...)
    end
end

######################################
#### mve_selection_exhaustive_search
######################################

"""
    mve_selection_exhaustive_search(
        μ, Σ, k; max_comb=0, γ=1.0, do_checks=false
    ) -> Vector{Int}

Search subsets of assets up to cardinality `k` to maximize the MVE Sharpe ratio `sqrt(μ_S' * Σ_S^-1 * μ_S)` over selected indices. If `max_comb == 0`, evaluates all combinations for each k; otherwise, randomly samples `max_comb` subsets per k.

# Arguments
- `μ::Vector{Float64}`: Expected returns vector (length n).
- `Σ::Matrix{Float64}`: Covariance matrix (n×n).
- `k::Int`: Maximum subset size (1 ≤ k ≤ n).
- `max_comb::Int`: Number of random subsets per k (0 ⇒ all combinations).
- `γ::Float64`: Risk‐aversion parameter (default = 1.0).
- `do_checks::Bool`: If `true`, validate inputs.

# Returns
- `selection::Vector{Int}`: Indices of the k selected assets.
"""
function mve_selection_exhaustive_search(
    μ::Vector{Float64},
    Σ::Matrix{Float64},
    k::Int;
    max_comb::Int=0,
    γ::Float64=1.0,
    do_checks::Bool=false
)

    n = length(μ)
    if do_checks
        @assert size(Σ) == (n,n) "Σ must be n×n"
        @assert 1 ≤ k ≤ n "k must be between 1 and n"
        @assert max_comb ≥ 0 "max_comb must be non-negative"
        @assert γ > 0 "γ must be positive"
    end

    best_sr = -Inf
    selection = Int[]

    if max_comb == 0
        for k in 1:k
            for sel_tuple in combinations(1:n, k)
                sel = collect(sel_tuple)
                sr = compute_mve_sr(μ, Σ; selection=sel, do_checks=false)
                if sr > best_sr
                    best_sr, selection = sr, sel
                end
            end
        end
    else
        for k in 1:k
            for _ in 1:max_comb
                sel = randperm(n)[1:k]
                sr = compute_mve_sr(μ, Σ; selection=sel, do_checks=false)
                if sr > best_sr
                    best_sr, selection = sr, sel
                end
            end
        end
    end

    return selection
end

######################################
#### mve_selection_cutting_planes
######################################

"""
    mve_selection_cutting_planes(μ, Σ, k;
                                 λ=Float64[],
                                 ΔT_max=600.0,
                                 gap=1e-4,
                                 num_random_restarts=5,
                                 use_warm_start=true,
                                 use_socp_lb=false,
                                 use_heuristic=true,
                                 use_kelley_primal=false,
                                 do_checks=false)

Outer-approximation cutting-plane solver for the ℓ₀-constrained max-Sharpe
problem with ridge regularization.

# Arguments
- `μ::Vector{Float64}`: Expected returns vector of length n.
- `Σ::Matrix{Float64}`: Covariance matrix (n×n) of asset returns.
- `k::Int`: Number of assets to select (must satisfy 1 ≤ k ≤ n).
- `λ::Vector{Float64}` (optional): Ridge regularization parameters (risk aversion coefficients) of length n. If not provided,
  it defaults to a uniform setting `fill(100.0 / sqrt(n), n)`, which is the standard scaling used in Bertsimas & Cory-Wright (2022).
- `ΔT_max::Float64`: Time limit for the solver in seconds.
- `gap::Float64`: Relative MIP optimality gap tolerance.
- `num_random_restarts::Int`: Number of random restarts for the warm-start heuristic.
- `use_warm_start::Bool`: If true, computes an initial feasible solution using random restarts to warm-start the OA cuts.
- `use_socp_lb::Bool`: If true, adds a second-order conic relaxation lower-bound cut via CPLEX's MI-SOCP.
- `use_heuristic::Bool`: If true, applies a hill-climbing heuristic at each OA callback to generate additional cuts.
- `use_kelley_primal::Bool`: If true, adds Kelley’s primal cutting-plane cuts at the root node using a user-specified number of iterations.
- `do_checks::Bool`: If true, asserts that all input arguments meet required conditions (e.g., dimensions, ranges).

# Returns
- `selection::Vector{Int}`: Indices of the k selected assets.

# Ridge Regularization
- If `λ` is unspecified or empty, it is internally set to `fill(100.0 / sqrt(n), n)` as a default, following the recommended
  scaling for balanced penalization.
"""
function mve_selection_cutting_planes(
    μ::Vector{Float64},
    Σ::Matrix{Float64},
    k::Int;
    λ::Vector{Float64}=Float64[],
    ΔT_max::Float64=600.0,
    gap::Float64=1e-4,
    num_random_restarts::Int=5,
    use_warm_start::Bool=true,
    use_socp_lb::Bool=false,
    use_heuristic::Bool=true,
    use_kelley_primal::Bool=false,
    do_checks::Bool=false,
)
    n = length(μ)
    if isempty(λ)
        λ = fill(100.0 / sqrt(n), n)
    end

    if do_checks
        @assert n > 0 "μ must be non-empty"
        @assert size(Σ) == (n,n) "Σ must be n×n"
        @assert length(λ) == n   "λ must be length n"
        @assert 1 ≤ k ≤ n        "k must be between 1 and n"
        @assert ΔT_max > 0       "ΔT_max must be positive"
        @assert gap > 0          "gap must be positive"
    end

    model = Model(optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPX_PARAM_EPGAP" => gap,
        "CPX_PARAM_TILIM" => ΔT_max,
    ))

    @variable(model, z[1:n], Bin)
    @variable(model, t)
    @constraint(model, sum(z) ≤ k)
    @objective(model, Min, t)

    if use_warm_start
        s0 = warm_start(μ, Σ, λ, k; num_random_restarts=num_random_restarts)
        set_start_value.(z, s0)
        cut = portfolios_objective(μ, Σ, λ, k, s0)
        @constraint(model, t ≥ cut.p + dot(cut.grad, z .- s0))
    end

    if use_socp_lb
        frac = cplex_misocp_relaxation(n, k; ΔT_max=ΔT_max)
        topk = sortperm(frac, rev=true)[1:k]
        s_lb = zeros(Float64, n); s_lb[topk] .= 1.0
        cut = portfolios_objective(μ, Σ, λ, k, s_lb)
        @constraint(model, t ≥ cut.p + dot(cut.grad, z .- s_lb))
    end

    function oa_cb(cb_data)
        zf = [callback_value(cb_data, z[i]) for i in 1:n]
        zv = round.(Int, zf)
        if sum(zv) != k
            return
        end

        s_val = Float64.(zv)
        cut = portfolios_objective(μ, Σ, λ, k, s_val)
        con = @build_constraint(t ≥ cut.p + dot(cut.grad, z .- zv))
        submit(model, LazyConstraint(cb_data), con)

        if use_heuristic
            inds2, _ = hillclimb(μ, Σ, k, findall(x -> x == 1, zv); maxiter=20)
            s_h = zeros(Float64, n); s_h[inds2] .= 1.0
            hcut = portfolios_objective(μ, Σ, λ, k, s_h)
            con2 = @build_constraint(t ≥ hcut.p + dot(hcut.grad, z .- s_h))
            submit(model, LazyConstraint(cb_data), con2)
        end
    end
    MOI.set(model, LazyConstraintCallback(), oa_cb)

    if use_kelley_primal
        stab0 = zeros(Float64, n)
        for c in kelley_primal_cuts(μ, Σ, λ, k, stab0, 20)
            @constraint(model, t ≥ c.p + dot(c.grad, z .- stab0))
        end
    end

    optimize!(model)
    # status = termination_status(model)
    zvals = value.(z)
    selection = findall(x -> x > 0.5, zvals)

    return selection #, status
end

end