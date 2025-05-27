module MVESelection
# this module contains the mve portfolio selection functions 
# based on exhaustive search of all combinations of n choose k assets
# and the cutting planes search of Bertsimas & Cory-Wright (2022) 

using LinearAlgebra
using Statistics
using JuMP
using CPLEX                              # CPLEX.Optimizer
import MathOptInterface                  # for status constants and callbacks
const MOI = MathOptInterface

using JuMP: @build_constraint, callback_value
import MathOptInterface: LazyConstraint, LazyConstraintCallback, submit
import Combinatorics: combinations

# bring in the cutting planes utils submodule:
using ..CuttingPlanesUtils: inner_dual,
                            hillclimb,
                            portfolios_socp,
                            portfolios_objective,
                            warm_start,
                            cplex_misocp_relaxation,
                            kelley_primal_cuts

using ..SharpeRatio: compute_sr, 
                     compute_mve_sr, 
                     compute_mve_weights

export compute_mve_selection, compute_mve_sr_decomposition, simulate_mve_sr

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
        for _ in 1:k
            for sel_tuple in combinations(1:n, k)
                sel = collect(sel_tuple)
                sr = compute_mve_sr(μ, Σ; selection=sel, do_checks=false)
                if sr > best_sr
                    best_sr, selection = sr, sel
                end
            end
        end
    else
        for _ in 1:k
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
  it defaults to a uniform setting `ones(n) .* (100.0/sqrt(n))`, which is the standard scaling used in Bertsimas & Cory-Wright (2022).
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
- If `λ` is unspecified or empty, it is internally set to `ones(n) .* (100.0/sqrt(n))` as a default, following the recommended
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
        λ = ones(n) .* (100.0/sqrt(n))
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

############################################
#### compute_mve_sr_decomposition 
############################################

"""
    compute_mve_sr_decomposition(
        μ, Σ, μ_sample, Σ_sample, k; do_checks=false
    ) -> NamedTuple

Compute the mean-variance efficient Sharpe-ratio decomposition into estimation and selection components.

# Arguments
- `μ::Vector{Float64}`: Population mean vector (length n).
- `Σ::Matrix{Float64}`: Population covariance matrix (n×n).
- `μ_sample::Vector{Float64}`: Sample mean vector (length n).
- `Σ_sample::Matrix{Float64}`: Sample covariance matrix (n×n).
- `k::Int`: Maximum cardinality k (1 ≤ k ≤ n) for the selection term.
- `do_checks::Bool`: If `true`, perform input validity checks.

# Returns
A `NamedTuple` with fields:
- `:mve_sr_cardk_est_term` :: `Float64` — estimation component = Sharpe-ratio of sample-MVE weights on population `(μ,Σ)`.
- `:mve_sr_cardk_sel_term` :: `Float64` — selection component = population MVE Sharpe-ratio on the selected assets.
"""
function compute_mve_sr_decomposition(
    μ::Vector{Float64}, Σ::Matrix{Float64},
    μ_sample::Vector{Float64}, Σ_sample::Matrix{Float64},
    k::Int; do_checks::Bool=false
)
    n = length(μ)
    if do_checks
        @assert length(μ_sample) == n "μ_sample must match length of μ"
        @assert size(Σ) == (n,n) "Σ must be n×n"
        @assert size(Σ_sample) == (n,n) "Σ_sample must be n×n"
        @assert 1 ≤ k ≤ n "k must be between 1 and n"
        @assert max_comb ≥ 0 "max_comb must be non-negative"
    end

    # Optional: sample MVE SR (unconstrained)
    # sample_mve_sr = compute_mve_sr(μ_sample, Σ_sample; do_checks=false)

    # Optional: sample MVE SR with cardinality k
    # assumes compute_mve_sr_cardk is defined elsewhere:
    # returns NamedTuple(weight = Vector, selection = Vector{Int}, sr = Float64)
    # result_cardk = compute_mve_sr_cardk(
    #     μ_sample, Σ_sample, k; max_comb=max_comb, do_checks=false
    # )

    # compute optimal sample selection
    selection = mve_selection(μ_sample, Σ_sample, k)

    # compute the optimal sample weights associated to this selection
    weights = compute_mve_weights(μ_sample, Σ_sample; selection=selection)

    # population Sharpe ratio decomposition: estimation term
    est_term = compute_sr(weights, μ, Σ; selection=selection)

    # population Sharpe ratio decomposition: selection term
    sel_term = compute_mve_sr(μ, Σ; selection=selection)

    return (
        # sample_mve_sr          = sample_mve_sr,
        # sample_mve_sr_cardk    = result_cardk.sr,
        mve_sr_cardk_est_term  = est_term,
        mve_sr_cardk_sel_term  = sel_term,
    )
end

############################
#### simulate_mve_sr
############################

"""
    simulate_mve_sr(
        μ, Σ, n_obs, k; max_comb=0, do_checks=false
    ) -> NamedTuple

Simulate samples from a multivariate normal (μ,Σ), compute sample MVE SR decomposition.

# Arguments
- `μ::Vector{Float64}`: True mean vector.
- `Σ::Matrix{Float64}`: True covariance matrix.
- `n_obs::Int`: Number of observations to simulate.
- `k::Int`: Maximum investment cardinality.
- `max_comb::Int`: Maximum combinations to enumerate (0 ⇒ all).
- `do_checks::Bool`: If `true`, perform input checks.

# Returns
A `NamedTuple` with fields:
- `:sample_mve_sr` :: `Float64` — optimal sample MVE Sharpe-ratio (unconstrained).
- `:sample_mve_sr_cardk` :: `Float64` — optimal sample MVE Sharpe-ratio under cardinality `k`.
- `:mve_sr_cardk_est_term` :: `Float64` — estimation component = Sharpe-ratio of sample-MVE weights on population `(μ,Σ)`.
- `:mve_sr_cardk_sel_term` :: `Float64` — selection component = population MVE Sharpe-ratio on the selected assets.
"""
function simulate_mve_sr(
    μ::Vector{Float64}, Σ::Matrix{Float64},
    n_obs::Int, k::Int; max_comb::Int=0, do_checks::Bool=false
)
    if do_checks 
        @assert !isempty(μ) "μ must be non-empty"
        @assert size(Σ) == (length(μ), length(μ)) "Σ must be square"
        @assert n_obs > 0 "n_obs must be positive"
        @assert 1 ≤ k ≤ length(μ) "k must be between 1 and length(μ)"
        @assert max_comb ≥ 1 "max_comb must be non-negative"
    end

    # draw standard normals
    Z = randn(length(μ), n_obs)                    # n × n_obs
    # turn Σ into a square root via cholesky
    L = cholesky(Symmetric(Σ)).L           # lower‐triangular
    # now sample = μ .+ L * Z
    sample = (L * Z) .+ μ[:, ones(n_obs)]
    μ_sample     = vec(mean(sample; dims=2))
    Σ_sample     = cov(eachcol(sample))

    # this needs the Distributions package
    # mvn = MvNormal(μ, Σ)
    # sample = rand(mvn, n_obs)          # size: length(μ) × n_obs
    # mu_sample = vec(mean(sample, dims=2))
    # sigma_sample = cov(transpose(sample))

    return compute_mve_sr_decomposition(
        μ, Σ, μ_sample, Σ_sample,
        k; max_comb=max_comb, do_checks=false
    )
end

end # end module