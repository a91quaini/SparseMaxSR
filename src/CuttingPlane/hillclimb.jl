using LinearAlgebra   # for zeros, etc.
using Statistics      # for sortperm

"""
    hillclimb(μ, Σ, k, inds0; maxiter=50)

Simple discrete first-order heuristic for the ℓ₀-constrained max-Sharpe problem:

1. Solve dual QP on support `inds`.
2. Form full w-vector (wᵢ=0 off-support).
3. Pick top-k assets by wᵢ → new support.
4. Repeat until convergence or `maxiter`.

# Arguments
- `μ::Vector{Float64}` : expected returns (length n)
- `Σ::Matrix{Float64}` : covariance matrix (n×n)
- `k::Int`            : desired number of nonzeros
- `inds0::Vector{Int}`: initial support of size k
- `maxiter::Int`      : maximum number of hill-climb iterations

# Returns
- `inds::Vector{Int}` : final support (size k)
- `w_full::Vector{Float64}` : full-length slack vector (wᵢ=0 if i∉inds)
"""
function hillclimb(μ::Vector{Float64},
                              Σ::Matrix{Float64},
                              k::Int,
                              inds0::Vector{Int};
                              maxiter::Int = 50)

    n = length(μ)
    @assert size(Σ) == (n,n)
    @assert length(inds0) == k

    inds = copy(inds0)
    w_full = zeros(n)

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

