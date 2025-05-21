module SharpeRatio

using LinearAlgebra

export compute_sr, compute_mve_sr

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

    # select submatrix or full matrix
    if isempty(selection) || length(selection) == length(μ)
        Σ_sel = Symmetric(Σ)
        μ_sel = μ
    else
        Σ_sel = Symmetric(Σ[selection, selection])
        μ_sel = μ[selection]
        w_sel = weights[selection]
    end

    # compute Sharpe ratio
    return dot(w_sel, μ_sel) / sqrt(dot(w_sel, Σ_sel * w_sel))
end

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
        w_sel = weights[selection]
    end

    x = isposdef(Σ_sel) ? 
          cholesky(Σ_sel) \ μ_sel :    # SPD case
          pinv(Σ_sel) * μ_sel          # PSD (singular) case
    # compute MVE Sharpe ratio
    return sqrt(dot(μ_sel, x))
end

end # module SharpeRatio