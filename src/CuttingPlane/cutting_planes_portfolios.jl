# src/CuttingPlane/cutting_planes_portfolios.jl

using JuMP
using JuMP: @build_constraint, callback_value
using CPLEX                              # CPLEX.Optimizer
import MathOptInterface                  # for callback support
const MOI = MathOptInterface
import MathOptInterface: LazyConstraint, LazyConstraintCallback, submit

"""
    cutting_planes_portfolios(μ, Σ, γ, k;
                              ΔT_max=600.0,
                              gap=1e-4,
                              num_random_restarts=5,
                              use_warm_start=true,
                              use_socp_lb=false,
                              use_heuristic=true,
                              use_kelley_primal=false)

Outer‐approximation cutting‐plane solver for the ℓ₀‐constrained max‐Sharpe
problem. Returns a binary vector z of length n with exactly k ones.

# Arguments
- `μ::Vector{Float64}`   Expected returns (length n)
- `Σ::Matrix{Float64}`   Covariance matrix (n×n)
- `γ::Vector{Float64}`   Penalty weights (length n)
- `k::Int`               Sparsity bound (1 ≤ k ≤ n)

# Keywords
- `ΔT_max::Float64`         MIP time limit (seconds)
- `gap::Float64`            Relative MIP gap
- `num_random_restarts::Int` Hill‐climb warm‐start tries
- `use_warm_start::Bool`      Add one OA cut from hill‐climb start
- `use_socp_lb::Bool`         Add one SOC‐P bound cut
- `use_heuristic::Bool`       Add heuristic cuts in callback
- `use_kelley_primal::Bool`   Add Kelley cuts at root

# Returns
- `Vector{Int}` binary selection z with sum(z)==k
"""
function cutting_planes_portfolios(
    μ::Vector{Float64},
    Σ::Matrix{Float64},
    γ::Vector{Float64},
    k::Int;
    ΔT_max::Float64=600.0,
    gap::Float64=1e-4,
    num_random_restarts::Int=5,
    use_warm_start::Bool=true,
    use_socp_lb::Bool=false,
    use_heuristic::Bool=true,
    use_kelley_primal::Bool=false,
)
    n = length(μ)
    @assert size(Σ)==(n,n) "Σ must be n×n"
    @assert length(γ)==n   "γ must be length n"
    @assert 1 ≤ k ≤ n      "k must be between 1 and n"

    model = Model(optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPX_PARAM_EPGAP" => gap,
        "CPX_PARAM_TILIM" => ΔT_max,
    ))

    @variable(model, z[1:n], Bin)
    @variable(model, t)
    @constraint(model, sum(z) ≤ k)
    @objective(model, Min, t)

    # 1) Warm‐start OA cut
    if use_warm_start
        s0  = get_warm_start(μ, Σ, γ, k; num_random_restarts=num_random_restarts)
        set_start_value.(z, s0)
        cut = portfolios_objective(μ, Σ, γ, k, s0)
        @constraint(model, t ≥ cut.p + dot(cut.grad, z .- s0))
    end

    # 2) SOC‐P bound cut
    if use_socp_lb
        frac = cplex_misocp_relaxation(n, k; ΔT_max=ΔT_max)
        topk = sortperm(frac, rev=true)[1:k]
        s_lb = zeros(Float64, n); s_lb[topk] .= 1.0
        cut = portfolios_objective(μ, Σ, γ, k, s_lb)
        @constraint(model, t ≥ cut.p + dot(cut.grad, z .- s_lb))
    end

    # 3) Lazy OA (and heuristic) callback
    function oa_cb(cb_data)
        # 3a) fetch fractional z's
        zf = [callback_value(cb_data, z[i]) for i in 1:n]
        zv = round.(Int, zf)
        # only proceed on full k‐supports
        if sum(zv) != k
            return
        end

        # 3b) extract the index list of selected assets
        inds = findall(x->x==1, zv)

        # 3c) outer‐approximation cut
        s_val = Float64.(zv)
        cut = portfolios_objective(μ, Σ, γ, k, s_val)
        con = @build_constraint(t ≥ cut.p + dot(cut.grad, z .- zv))
        submit(model, LazyConstraint(cb_data), con)

        # 3d) heuristic cut
        if use_heuristic
            inds2, _ = hillclimb(μ, Σ, k, inds; maxiter=20)
            s_h = zeros(Float64, n); s_h[inds2] .= 1.0
            hcut = portfolios_objective(μ, Σ, γ, k, s_h)
            con2 = @build_constraint(t ≥ hcut.p + dot(hcut.grad, z .- s_h))
            submit(model, LazyConstraint(cb_data), con2)
        end
    end
    MOI.set(model, LazyConstraintCallback(), oa_cb)

    # 4) Root Kelley‐primal cuts
    if use_kelley_primal
        stab0 = zeros(Float64, n)
        for c in kelley_primal_cuts(μ, Σ, γ, k, stab0, 20)
            @constraint(model, t ≥ c.p + dot(c.grad, z .- stab0))
        end
    end

    optimize!(model)
    return Int.(round.(value.(z)))
end
