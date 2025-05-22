module SharpeRatio
# this module contains functions useful for computing Sharpe ratio,
# mve Sharpe ratio, mve weights, ...

using LinearAlgebra
using Random
using Statistics
using Distributions
import Base.Iterators: combinations

# bring in the mve_selection submodule:
using ..MVESelection: mve_selection

export compute_sr, 
       compute_mve_sr, 
       compute_mve_weights,
       compute_mve_sr_decomposition,
       simulate_mve_sr

############################################
#### compute_sr 
############################################

"""
    compute_sr(weights, μ, Σ; selection=Int[], do_checks=false)

Compute the Sharpe ratio of a given portfolio.

# Arguments
- `weights::Vector{Float64}`: Portfolio weights (length n).
- `μ::Vector{Float64}`: Expected returns vector (length n).
- `Σ::Matrix{Float64}`: Covariance matrix (n×n).
- `selection::Vector{Int}` (keyword): Indices of assets to include (default selects all).
- `do_checks::Bool` (keyword): If `true`, perform input argument validations (default = `false`).

# Returns
- `sr::Float64`: The Sharpe ratio `wᵀ μ / sqrt(wᵀ Σ w)` over the selected assets.
"""
function compute_sr(
    weights::Vector{Float64},
    μ::Vector{Float64},
    Σ::Matrix{Float64};
    selection::Vector{Int}=Int[],
    do_checks::Bool=false
)
    if do_checks
        @assert length(weights) == length(μ) "weights and μ must be of the same length"
        @assert size(Σ,1) == size(Σ,2) "Σ must be a square matrix"
        @assert size(Σ,1) == length(weights) "dimensions of Σ must match length of weights"
        if !isempty(selection)
            @assert all(1 .<= selection .<= length(μ)) "selection indices must be within [1, length(μ)]"
        end
    end

    Σ = Symmetric(Σ)

    # select submatrix or full matrix
    if isempty(selection) || length(selection) == length(μ)
        return dot(weights, μ) / sqrt(dot(weights, Σ * weights))
    else
        Σ_sel = Symmetric(Σ[selection, selection])
        μ_sel = μ[selection]
        w_sel = weights[selection]
    end

    # compute Sharpe ratio
    return dot(w_sel, μ_sel) / sqrt(dot(w_sel, Σ_sel * w_sel))
end

############################################
#### compute_mve_sr 
############################################

"""
    compute_mve_sr(μ, Σ; selection=Int[], do_checks=false)

Compute the maximum Sharpe ratio of the mean-variance efficient portfolio.

# Arguments
- `μ::Vector{Float64}`: Expected returns vector (length n).
- `Σ::Matrix{Float64}`: Covariance matrix (n×n).
- `selection::Vector{Int}` (keyword): Indices of assets to include (default selects all).
- `do_checks::Bool` (keyword): If `true`, perform input argument validations (default = `false`).

# Returns
- `mve_sr::Float64`: The maximal Sharpe ratio `sqrt(μᵀ Σ⁻¹ μ)` over the selected assets.
"""
function compute_mve_sr(
    μ::Vector{Float64},
    Σ::Matrix{Float64};
    selection::Vector{Int}=Int[],
    do_checks::Bool=false
)
    if do_checks
        @assert length(μ) == size(Σ,1) "length of μ must equal dimensions of Σ"
        @assert size(Σ,1) == size(Σ,2) "Σ must be a square matrix"
        if !isempty(selection)
            @assert all(1 .<= selection .<= length(μ)) "selection indices must be within [1, length(μ)]"
        end
    end

    # select submatrix or full matrix
    if isempty(selection) || length(selection) == length(μ)
        Σ_sel = Symmetric(Σ)
        μ_sel = μ
    else
        Σ_sel = Symmetric(Σ[selection, selection])
        μ_sel = μ[selection]
    end

    x = isposdef(Σ_sel) ? 
          cholesky(Σ_sel) \ μ_sel :    # SPD case
          pinv(Σ_sel) * μ_sel          # PSD (singular) case
    # compute MVE Sharpe ratio
    return sqrt(dot(μ_sel, x))
end

############################################
#### compute_mve_weights 
############################################

"""
    compute_mve_weights(
        mu, sigma; selection=Int[], gamma=1.0, do_checks=false
    ) -> Vector{Float64}

Compute Mean-Variance Efficient (MVE) portfolio weights:

w = 1/γ * Σ^{-1} μ

# Arguments
- `mu::Vector{Float64}`: First moment vector (length n).
- `sigma::Matrix{Float64}`: Covariance matrix (n×n).
- `selection::Vector{Int}` (keyword): Indices to include (default all).
- `gamma::Float64` (keyword): Risk aversion (default 1.0).
- `do_checks::Bool` (keyword): If `true`, perform input checks (default false).

# Returns
- `w::Vector{Float64}`: Length-n weight vector with zeros for unselected assets.
"""
function compute_mve_weights(
    μ::Vector{Float64},
    Σ::Matrix{Float64};
    selection::Vector{Int}=Int[],
    γ::Float64=1.0,
    do_checks::Bool=false
)
    n = length(μ)
    if do_checks
        @assert n > 0 "μ must be non-empty"
        @assert size(Σ) == (n,n) "sigma must be n×n"
        @assert γ > 0 "γ must be positive"
        if !isempty(selection)
            @assert all(1 .<= selection .<= n) "selection indices out of range"
        end
    end

    # Use Symmetric to ensure efficient SPD solves
    Σ = Symmetric(Σ)
    if isempty(selection) || length(selection) == n
        # full-sample weights
        w = Σ \ μ
        return w / γ
    else
        # weights only on selected assets
        w_full = zeros(Float64, n)
        μ_sel  = μ[selection]
        Σ_sel   = Symmetric(Σ[selection, selection])
        w_sel   = Σ_sel \ μ_sel
        w_full[selection] = w_sel / γ
        return w_full
    end
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

###################################
#### compute_mve_sr_cardk
###################################

"""
    compute_mve_sr_cardk(
        μ, Σ, k; max_comb=0, γ=1.0, do_checks=false
    ) -> NamedTuple{(:sr, :weights, :selection),Tuple{Float64,Vector{Float64},Vector{Int}}}

Search subsets of assets up to cardinality `k` to maximize the MVE Sharpe ratio `sqrt(μ_S' * Σ_S^-1 * μ_S)` over selected indices. If `max_comb == 0`, evaluates all combinations for each k; otherwise, randomly samples `max_comb` subsets per k.

# Arguments
- `μ::Vector{Float64}`: Expected returns vector (length n).
- `Σ::Matrix{Float64}`: Covariance matrix (n×n).
- `k::Int`: Maximum subset size (1 ≤ k ≤ n).
- `max_comb::Int`: Number of random subsets per k (0 ⇒ all combinations).
- `γ::Float64`: Risk‐aversion parameter (default = 1.0).
- `do_checks::Bool`: If `true`, validate inputs.

# Returns
A `NamedTuple` with fields:
- `sr::Float64`: highest Sharpe ratio found.
- `weights::Vector{Float64}`: MVE weights (length n) for the best subset.
- `selection::Vector{Int}`: indices (1-based) of the chosen assets.
"""
function compute_mve_sr_cardk(
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
    best_sel = Int[]

    if max_comb == 0
        for k in 1:k
            for sel_tuple in combinations(1:n, k)
                sel = collect(sel_tuple)
                sr = compute_mve_sr(μ, Σ; selection=sel, do_checks=false)
                if sr > best_sr
                    best_sr, best_sel = sr, sel
                end
            end
        end
    else
        for k in 1:k
            for _ in 1:max_comb
                sel = randperm(n)[1:k]
                sr = compute_mve_sr(μ, Σ; selection=sel, do_checks=false)
                if sr > best_sr
                    best_sr, best_sel = sr, sel
                end
            end
        end
    end

    best_weights = compute_mve_weights(μ, Σ; selection=best_sel, gamma=γ, do_checks=false)
    return (sr=best_sr, weights=best_weights, selection=best_sel)
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

    mvn = MvNormal(μ, Σ)
    sample = rand(mvn, n_obs)          # size: length(μ) × n_obs
    mu_sample = vec(mean(sample, dims=2))
    sigma_sample = cov(transpose(sample))

    return compute_mve_sr_decomposition(
        μ, Σ, mu_sample, sigma_sample,
        k; max_comb=max_comb, do_checks=false
    )
end

end # module SharpeRatio