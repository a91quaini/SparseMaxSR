module MVESelection
# MVE selection via exhaustive search and cutting-planes

using LinearAlgebra
using Statistics
using JuMP
using Random
using Combinatorics: binomial, combinations
import MathOptInterface
const MOI = MathOptInterface

using ..CuttingPlanesUtils: hillclimb,
                            cplex_misocp_relaxation,
                            portfolios_objective,
                            warm_start,
                            kelley_primal_cuts

using ..SharpeRatio: compute_sr,
                     compute_mve_sr,
                     compute_mve_weights

using ..SparseMaxSR: EPS_RIDGE                     

# Helpful JuMP utilities used in callbacks
using JuMP: @build_constraint, callback_value

export compute_mve_selection, compute_mve_sr_decomposition, simulate_mve_sr

######################################
#### compute_mve_selection
######################################

"""
    compute_mve_selection(μ, Σ, k;
                          method=:auto,
                          exhaustive_threshold=20,
                          exhaustive_max_combs=200_000,
                          optimizer=nothing,
                          attrs=NamedTuple(),
                          dual_optimizer=nothing,
                          dual_attrs=NamedTuple(),
                          epsilon=EPS_RIDGE,
                          rng=Random.default_rng(),
                          kwargs...)

Select a maximum-Sharpe portfolio with cardinality `k`.

Method selection:
  • `:exhaustive` — exhaustive search  
  • `:cutting_planes` — cutting-planes OA  
  • `:auto` (default) — use exhaustive if `n ≤ exhaustive_threshold`
    **or** `binomial(n,k) ≤ exhaustive_max_combs`; otherwise cutting-planes.

`optimizer`/`attrs` configure the MILP master (cutting-planes path).
`dual_optimizer`/`dual_attrs` configure the QP duals used in cuts and heuristics.
If `epsilon>0`, methods internally use `Σ + ε·mean(diag(Σ))·I`.

Returns a sorted `Vector{Int}` of the selected `k` indices.
"""
function compute_mve_selection(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    k::Integer;
    method::Symbol = :auto,
    exhaustive_threshold::Int = 20,
    exhaustive_max_combs::Int = 200_000,
    optimizer = nothing,
    attrs = NamedTuple(),
    dual_optimizer = nothing,
    dual_attrs = NamedTuple(),
    epsilon::Real = EPS_RIDGE[],
    rng::AbstractRNG = Random.default_rng(),
    kwargs...
)
    n = length(μ)
    (size(Σ, 1) == n && size(Σ, 2) == n) || error("Σ must be $n×$n; got $(size(Σ)).")
    (1 ≤ k ≤ n)                       || error("k must satisfy 1 ≤ k ≤ n.")
    (epsilon ≥ 0)                     || error("epsilon must be ≥ 0.")

    # Gentle symmetrization (ridge handled in callees)
    Σsym = (Σ .+ Σ') / 2

    # Decide the path
    use_exhaustive = if method === :exhaustive
        true
    elseif method === :cutting_planes
        false
    elseif method === :auto
        (n ≤ exhaustive_threshold) || (binomial(n, k) ≤ exhaustive_max_combs)
    else
        error("Unknown method=$(method). Use :auto, :exhaustive, or :cutting_planes.")
    end

    # Run the chosen method
    sel = if use_exhaustive
        mve_selection_exhaustive_search(
            μ, Σsym, k;
            epsilon = epsilon,
            rng = rng,
            kwargs...
        )
    else
        mve_selection_cutting_planes(
            μ, Σsym, k;
            optimizer = optimizer,
            attrs = attrs,
            dual_optimizer = dual_optimizer,
            dual_attrs = dual_attrs,
            epsilon = epsilon,
            rng = rng,
            kwargs...
        )
    end

    # Normalize/validate the result
    sel_vec = sort!(unique!(collect(sel)))
    length(sel_vec) == k || error("Selected set has length $(length(sel_vec)) ≠ k=$k.")
    all(1 .≤ sel_vec .≤ n) || error("Selected indices out of bounds 1..$n.")
    return sel_vec
end


######################################
#### mve_selection_exhaustive_search
######################################

"""
    mve_selection_exhaustive_search(μ, Σ, k;
        exactly_k::Bool = true,
        max_samples_per_k::Int = 0,
        epsilon::Real = EPS_RIDGE,
        rng::AbstractRNG = Random.default_rng(),
        γ::Real = 1.0,                # kept for backward compatibility (ignored)
        do_checks::Bool = true
    ) -> Vector{Int}

Exhaustive (or sampled) search over subsets to maximize the MVE Sharpe ratio
on the selected indices. By default searches **exactly** k assets; set
`exactly_k=false` to allow any size in `1:k` and return the overall best.

If `max_samples_per_k == 0`, evaluates **all** combinations of the chosen size(s).
Otherwise, for each size `s`, it evaluates `max_samples_per_k` random supports
of size `s` (uniformly sampled).

The covariance used in evaluation is `Σ_eff = Σ + ε·mean(diag(Σ))·I` when
`epsilon > 0` for numerical stability.
"""
function mve_selection_exhaustive_search(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    k::Integer;
    exactly_k::Bool = true,
    max_samples_per_k::Int = 0,
    epsilon::Real = EPS_RIDGE[],
    rng::AbstractRNG = Random.default_rng(),
    γ::Real = 1.0,       # ignored
    do_checks::Bool = true
) :: Vector{Int}

    n = length(μ)
    if do_checks
        (size(Σ,1) == n && size(Σ,2) == n) || error("Σ must be $n×$n; got $(size(Σ)).")
        (1 ≤ k ≤ n) || error("k must satisfy 1 ≤ k ≤ n.")
        (max_samples_per_k ≥ 0) || error("max_samples_per_k must be non-negative.")
    end

    # Symmetrize + optional ridge (used consistently in compute_mve_sr)
    Σsym = (Σ .+ Σ') / 2
    Σeff = epsilon > 0 ? Σsym + (float(epsilon) * mean(diag(Σsym))) * I : Σsym

    # Which subset sizes to consider
    sizes = exactly_k ? (k:k) : (1:k)

    best_sr  = -Inf
    best_sel = Int[]

    # Helper: evaluate a given selection
    function eval_sel!(sel::Vector{Int})
        sr = compute_mve_sr(μ, Σeff; selection = sel, do_checks = false)
        if isfinite(sr) && sr > best_sr
            best_sr  = sr
            best_sel = copy(sel)
        end
        return nothing
    end

    # Exhaustive over all combs for size s
    function exhaustive_over_size!(s::Int)
        for tup in combinations(1:n, s)
            sel = collect(tup)
            eval_sel!(sel)
        end
    end

    # Random sampling of supports of size s
    function sampled_over_size!(s::Int, m::Int)
        m ≤ 0 && return
        for _ in 1:m
            sel = sort!(randperm(rng, n)[1:s])
            eval_sel!(sel)
        end
    end

    for s in sizes
        if max_samples_per_k == 0
            exhaustive_over_size!(s)
        else
            sampled_over_size!(s, max_samples_per_k)
        end
    end

    # Ensure canonical, in-bounds output
    best_sel = sort!(unique!(best_sel))
    isempty(best_sel) && error("Failed to find a feasible selection (check inputs).")
    all(1 .≤ best_sel .≤ n) || error("Selection out of bounds 1..$n.")

    # If exactly_k=true, enforce length k (defensive)
    if exactly_k && length(best_sel) != k
        if length(best_sel) > k
            best_sel = best_sel[1:k]
        else
            # pad by best unused indices using a simple score proxy (μ/σ)
            unused = setdiff(1:n, best_sel)
            σ = sqrt.(diag(Σsym) .+ eps(Float64))
            score = μ ./ σ
            add_inds = partialsortperm(view(score, unused), 1:(k - length(best_sel)); rev=true)
            best_sel = sort!(vcat(best_sel, unused[add_inds]))
        end
    end

    return best_sel
end


######################################
#### mve_selection_cutting_planes
######################################

"""
    mve_selection_cutting_planes(μ, Σ, k;
        γ = Float64[],              # per-asset penalties (aka λ in the old API)
        λ = Float64[],              # backward-compat alias; if provided and γ==[], use λ
        optimizer = nothing,        # MILP optimizer (e.g., HiGHS.Optimizer, GLPK.Optimizer, Gurobi.Optimizer, CPLEX.Optimizer)
        attrs = NamedTuple(),       # extra attributes for the MILP master
        dual_optimizer = nothing,   # QP/QCQP optimizer for duals in cuts (Clarabel/COSMO/SCS/Mosek)
        dual_attrs = NamedTuple(),  # extra attributes for the dual optimizer
        ΔT_max::Real = 600.0,       # master time limit (seconds)
        gap::Real = 1e-4,           # master relative MIP gap
        num_random_restarts::Int = 5,
        use_warm_start::Bool = true,
        use_socp_lb::Bool = false,  # kept for compatibility; implemented as LP selection LB
        use_lp_lb::Bool = use_socp_lb,
        use_heuristic::Bool = true,
        use_kelley_primal::Bool = false,
        epsilon::Real = EPS_RIDGE,        # ridge for Σ in duals
        rng::AbstractRNG = Random.default_rng(),
        do_checks::Bool = false
    ) -> Vector{Int}

Outer-approximation cutting-plane solver for the ℓ₀-constrained max-Sharpe problem.

- Master MILP in (z, t) with ∑z == k and cuts:
      minimize t
      subject to 0 ≤ z ≤ 1, ∑ z == k,
                 t ≥ p(s) + ∇p(s)ᵀ (z - s)   for selected s ∈ [0,1]^n.
- Adds cuts at warm starts, an LP-selection lower-bound point, optional Kelley root passes,
  and (if supported) via lazy constraints at incumbent integer points.

`optimizer`/`attrs` configure the MILP master; `dual_optimizer`/`dual_attrs` configure the
dual QP solves inside `portfolios_objective` and `hillclimb`. If `epsilon>0`, duals use
Σ_eff = Σ + ε·mean(diag(Σ))·I.
"""
function mve_selection_cutting_planes(
    μ::AbstractVector{<:Real},
    Σ::AbstractMatrix{<:Real},
    k::Integer;
    γ::AbstractVector{<:Real} = Float64[],
    λ::AbstractVector{<:Real} = Float64[],
    optimizer = nothing,
    attrs = NamedTuple(),
    dual_optimizer = nothing,
    dual_attrs = NamedTuple(),
    ΔT_max::Real = 600.0,
    gap::Real = 1e-4,
    num_random_restarts::Int = 5,
    use_warm_start::Bool = true,
    use_socp_lb::Bool = false,
    use_lp_lb::Bool = use_socp_lb,
    use_heuristic::Bool = true,
    use_kelley_primal::Bool = false,
    epsilon::Real = EPS_RIDGE[],
    rng::AbstractRNG = Random.default_rng(),
    do_checks::Bool = false,
) :: Vector{Int}
    n = length(μ)
    do_checks && begin
        (size(Σ,1) == n && size(Σ,2) == n) || error("Σ must be $n×$n; got $(size(Σ)).")
        (1 ≤ k ≤ n) || error("k must satisfy 1 ≤ k ≤ n.")
        (ΔT_max > 0 && gap ≥ 0) || error("ΔT_max must be >0 and gap ≥ 0.")
    end

    # penalties: γ wins over λ; default if both empty
    if isempty(γ)
        γ = isempty(λ) ? fill(100.0 / sqrt(n), n) : collect(λ)
    else
        γ = collect(γ)
    end
    do_checks && (length(γ) == n || error("γ must have length n=$n."))

    # Symmetrize Σ (ridge is applied inside dual evaluations)
    Σsym = (Σ .+ Σ') / 2

    # Require an explicit MILP optimizer (no hidden imports in src/)
    optimizer === nothing && error("Please provide a MILP `optimizer=` (e.g., HiGHS.Optimizer, GLPK.Optimizer, Gurobi.Optimizer, CPLEX.Optimizer).")

    model = Model(optimizer)
    set_silent(model)

    # Generic, solver-agnostic attributes (ignore if unsupported)
    try set_optimizer_attribute(model, MOI.TimeLimitSec(), float(ΔT_max)) catch; end
    try set_optimizer_attribute(model, MOI.RelativeGapTolerance(), float(gap)) catch; end
    for (k_attr, v_attr) in pairs(attrs)
        set_optimizer_attribute(model, k_attr, v_attr)
    end

    # Variables and base constraints
    @variable(model, z[1:n], Bin)
    @variable(model, t)
    @constraint(model, sum(z) == k)    # exactly k assets
    @objective(model, Min, t)

    # Helper to add a single OA cut at s ∈ [0,1]^n
    function add_cut_at!(s::AbstractVector{<:Real})
        cut = portfolios_objective(μ, Σsym, γ, k, s;
                                   optimizer = dual_optimizer,
                                   attrs = dual_attrs,
                                   epsilon = epsilon)
        if cut.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
            @constraint(model, t ≥ cut.p + sum(cut.grad[i] * (z[i] - s[i]) for i in 1:n))
            return true
        else
            return false
        end
    end

    # Warm start via heuristics + cut
    if use_warm_start
        s0 = warm_start(μ, Σsym, γ, k;
                        num_random_restarts = num_random_restarts,
                        maxiter = 50,
                        optimizer = dual_optimizer,
                        attrs = dual_attrs,
                        epsilon = epsilon,
                        rng = rng)
        @inbounds for i in 1:n
            set_start_value(z[i], s0[i])
        end
        add_cut_at!(s0)
    end

    # LP selection lower bound (score-based LP, then cut at binarized top-k)
    if use_lp_lb
        # Simple SNR-like scores as a generic proxy
        σ = sqrt.(diag(Σsym) .+ eps(Float64))
        scores = μ ./ σ
        sel_relax = cplex_misocp_relaxation(scores, k;
                                            optimizer = dual_optimizer,
                                            ΔT_max = min(ΔT_max, 30.0))
        z_lp = sel_relax.z
        topk = partialsortperm(z_lp, 1:k; rev=true)
        s_lb = zeros(Float64, n); @inbounds s_lb[topk] .= 1.0
        add_cut_at!(s_lb)
    end

    # Optional Kelley root passes (add root cuts)
    if use_kelley_primal
        stab0 = zeros(Float64, n)
        cuts = kelley_primal_cuts(μ, Σsym, γ, k, stab0, 10;
                                  optimizer = optimizer,           # root LP/MIP
                                  epsilon = epsilon)
        for c in cuts
            @constraint(model, t ≥ c.p + sum(c.grad[i] * (z[i] - stab0[i]) for i in 1:n))
        end
    end

    # Lazy OA cuts at integer incumbents (if supported)
    if MOI.supports(model, MOI.LazyConstraintCallback())
        function oa_cb(cb_data)
            zf = [callback_value(cb_data, z[i]) for i in 1:n]
            zv = round.(Int, zf)          # should already satisfy ∑z == k (enforced)
            s_val = Float64.(zv)

            # Main cut
            cut = portfolios_objective(μ, Σsym, γ, k, s_val;
                                       optimizer = dual_optimizer,
                                       attrs = dual_attrs,
                                       epsilon = epsilon)
            if cut.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
                con = @build_constraint(t ≥ cut.p + sum(cut.grad[i] * (z[i] - s_val[i]) for i in 1:n))
                MOI.submit(model, MOI.LazyConstraint(cb_data), con)
            end

            # Heuristic improvement cut
            if use_heuristic
                hres = hillclimb(μ, Σsym, k, findall(==(1), zv);
                                 maxiter = 20,
                                 optimizer = dual_optimizer,
                                 attrs = dual_attrs,
                                 epsilon = epsilon)
                s_h = zeros(Float64, n); @inbounds s_h[hres.inds] .= 1.0
                hcut = portfolios_objective(μ, Σsym, γ, k, s_h;
                                            optimizer = dual_optimizer,
                                            attrs = dual_attrs,
                                            epsilon = epsilon)
                if hcut.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
                    con2 = @build_constraint(t ≥ hcut.p + sum(hcut.grad[i] * (z[i] - s_h[i]) for i in 1:n))
                    MOI.submit(model, MOI.LazyConstraint(cb_data), con2)
                end
            end
        end
        MOI.set(model, MOI.LazyConstraintCallback(), oa_cb)
    else
        @warn "Selected optimizer does not support LazyConstraintCallback(); proceeding without lazy OA cuts."
    end

    optimize!(model)

    # Extract selection (defensive: choose top-k by z value)
    zvals = value.(z)
    sel = sort!(collect(partialsortperm(zvals, 1:k; rev=true)))
    return sel
end

end # end module
