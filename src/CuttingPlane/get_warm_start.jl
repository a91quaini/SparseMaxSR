using Random                          # for shuffle
using SparseMaxSR.CuttingPlane: hillclimb, portfolios_objective  # local helpers

"""
    get_warm_start(μ, Σ, γ, k; num_random_restarts=5)

Generate a good initial 0–1 portfolio `s` with ∥s∥₀ = k by
random restarts plus hill‐climbing.

# Arguments
- `μ::Vector{Float64}` : expected‐return vector
- `Σ::Matrix{Float64}` : covariance matrix
- `γ::Vector{Float64}` : per‐asset penalty weights
- `k::Int`            : cardinality bound
- `num_random_restarts::Int` : how many random seeds to try (default 5)

# Returns
- `s0::Vector{Float64}` : a binary vector of length `length(μ)` with exactly `k` ones
"""
function get_warm_start(μ::Vector{Float64},
                        Σ::Matrix{Float64},
                        γ::Vector{Float64},
                        k::Int;
                        num_random_restarts::Int = 5)

    n = length(μ)
    @assert size(Σ) == (n,n)  "Σ must be n×n"
    @assert length(γ) == n    "γ must be length n"
    @assert 1 ≤ k ≤ n         "k must be between 1 and n"

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