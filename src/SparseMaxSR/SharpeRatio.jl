module SharpeRatio
# this module contains functions useful for computing Sharpe ratio,
# mve Sharpe ratio, mve weights, ...

using LinearAlgebra
using Random
using Statistics

export compute_sr, 
       compute_mve_sr, 
       compute_mve_weights

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

end # end module