using LinearAlgebra
using JuMP
using CPLEX                              # CPLEX.Optimizer
import MathOptInterface                  # for status constants and callbacks
const MOI = MathOptInterface

# bring in the utils submodule:
using .CuttingPlanesUtils:
    inner_dual,
    hillclimb,
    portfolios_socp,
    portfolios_objective,
    warm_start,
    cplex_misocp_relaxation,
    kelley_primal_cuts


using JuMP: @build_constraint, callback_value
import MathOptInterface: LazyConstraint, LazyConstraintCallback, submit

"""
    cutting_planes_selection(μ, Σ, γ, k;
                             ΔT_max=600.0,
                             gap=1e-4,
                             num_random_restarts=5,
                             use_warm_start=true,
                             use_socp_lb=false,
                             use_heuristic=true,
                             use_kelley_primal=false)

Outer‐approximation cutting‐plane solver for the ℓ₀‐constrained max‐Sharpe
problem.

# Returns
- `selection::Vector{Int}` : the indices of the k selected assets.
- `status`                : the MOI termination status of the solve.

All keyword arguments and logic (warm‐start, SOC‐P bound, heuristic cuts
and Kelley primal cuts) are identical to the original
`cutting_planes_portfolios`.
"""
function cutting_planes_selection(
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
    @assert size(Σ) == (n,n) "Σ must be n×n"
    @assert length(γ) == n   "γ must be length n"
    @assert 1 ≤ k ≤ n        "k must be between 1 and n"

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
        s0  = warm_start(μ, Σ, γ, k; num_random_restarts=num_random_restarts)
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

    # 3) Lazy‐OA (and heuristic) callback
    function oa_cb(cb_data)
        zf = [callback_value(cb_data, z[i]) for i in 1:n]
        zv = round.(Int, zf)
        if sum(zv) != k
            return  # only process full‐support candidates
        end

        # OA cut
        s_val = Float64.(zv)
        cut = portfolios_objective(μ, Σ, γ, k, s_val)
        con = @build_constraint(t ≥ cut.p + dot(cut.grad, z .- zv))
        submit(model, LazyConstraint(cb_data), con)

        # heuristic cut
        if use_heuristic
            inds2, _ = hillclimb(μ, Σ, k, findall(x->x==1, zv); maxiter=20)
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
    status = termination_status(model)

    # build `selection` as the list of indices where z ≈ 1
    zvals = value.(z)
    selection = findall(x-> x > 0.5, zvals)

    return selection, status
end
