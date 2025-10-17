module MIQPHeuristicSearch

using LinearAlgebra, Statistics
using JuMP, CPLEX
import MathOptInterface as MOI
import ..SharpeRatio: compute_sr, compute_mve_sr, compute_mve_weights
import ..Utils: EPS_RIDGE, _prep_S

export mve_miqp_heuristic_search

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    _solve_once!(μ, Σ, N, k, m, γ, fmin_loc, fmax_loc;
                 x0=nothing, v0=nothing,
                 mipgap=1e-4, time_limit=Inf, threads=0, verbose=false,
                 epsilon=EPS_RIDGE)

Single MIQP solve with cardinality and box constraints:

minimize   0.5·γ·x'Σx − μ'x
subject to ∑ x_i = 1,  m ≤ ∑ v_i ≤ k,
           fmin_i·v_i ≤ x_i ≤ fmax_i·v_i, v_i ∈ {0,1}.

Returns a named tuple `(x, v, sr, status, obj)`, where `sr` is computed from `x`
against `(μ, Σ)` (using the same stabilized Σ as the objective).
"""
function _solve_once!(
    μ::AbstractVector, Σ::AbstractMatrix, N::Int, k::Int, m::Int, γ::Float64,
    fmin_loc::Vector{Float64}, fmax_loc::Vector{Float64};
    x0::Union{Nothing,AbstractVector}=nothing,
    v0::Union{Nothing,AbstractVector}=nothing,
    mipgap::Float64=1e-4, time_limit::Real=Inf, threads::Int=0,
    verbose::Bool=false, epsilon::Real=EPS_RIDGE,
)
    model = Model(CPLEX.Optimizer)
    if !verbose; set_silent(model); end
    if isfinite(time_limit)
        MOI.set(model, MOI.TimeLimitSec(), float(time_limit))
    end
    try set_optimizer_attribute(model, "MIPGap", mipgap) catch end
    if threads > 0
        try set_optimizer_attribute(model, "CPX_PARAM_THREADS", threads) catch end
    end

    @variable(model, x[1:N])
    @variable(model, v[1:N], Bin)
    @constraint(model, sum(x) == 1.0)
    @constraint(model, sum(v) <= k)
    @constraint(model, sum(v) >= m)
    @constraint(model, [i=1:N], x[i] <= fmax_loc[i] * v[i])
    @constraint(model, [i=1:N], x[i] >= fmin_loc[i] * v[i])
    @objective(model, Min, 0.5 * γ * dot(x, Σ * x) - dot(μ, x))

    if x0 !== nothing; @inbounds for i in 1:N; set_start_value(x[i], x0[i]); end end
    if v0 !== nothing; @inbounds for i in 1:N; set_start_value(v[i], v0[i]); end end

    optimize!(model)

    if MOI.get(model, MOI.ResultCount()) == 0
        return (x = zeros(N),
                v = zeros(Int, N),
                sr = -Inf,
                status = termination_status(model),
                obj = NaN)
    end

    xx = value.(x)
    vv = value.(v)
    vv = [ (isfinite(vv[i]) && vv[i] ≥ 0.5) ? 1 : 0 for i in 1:N ]
    sr = compute_sr(xx, μ, Σ; epsilon=epsilon, stabilize_Σ=false, do_checks=false)

    return (x = collect(xx),
            v = vv,
            sr = sr,
            status = termination_status(model),
            obj = objective_value(model))
end

"""
    _expand_bounds!(x, v, fmin_loc, fmax_loc, expand_factor, expand_tol)

If an active weight `x_i` is near its bound (within `expand_tol`), expand that bound
by multiplying it by `expand_factor` (sign-aware for possibly negative `fmin`).
"""
function _expand_bounds!(
    x::Vector{Float64}, v::Vector{Int},
    fmin_loc::Vector{Float64}, fmax_loc::Vector{Float64},
    expand_factor::Float64, expand_tol::Float64,
)
    N = length(x)
    @inbounds for i in 1:N
        if v[i] == 1
            if x[i] ≥ fmax_loc[i] - expand_tol
                fmax_loc[i] *= expand_factor
            elseif x[i] ≤ fmin_loc[i] + expand_tol
                if fmin_loc[i] < 0
                    fmin_loc[i] *= expand_factor
                end
            end
        end
    end
    nothing
end

# =============================================================================
# Public API
# =============================================================================

"""
    mve_miqp_heuristic_search(μ, Σ; k,
                              m::Int=0, γ::Float64=1.0,
                              fmin=zeros(length(μ)), fmax=ones(length(μ)),
                              expand_rounds::Int=2,
                              expand_factor::Float64=2.0,
                              expand_tol::Float64=1e-7,
                              mipgap::Float64=1e-4,
                              time_limit::Real=Inf,
                              threads::Int=0,
                              x_start=nothing, v_start=nothing,
                              compute_weights::Bool=false,
                              use_refit::Bool=true,
                              verbose::Bool=false,
                              epsilon::Real=EPS_RIDGE,
                              stabilize_Σ::Bool=true,
                              do_checks::Bool=false) -> NamedTuple{(:selection, :weights, :sr, :status)}

Heuristic MIQP for mean–variance efficient (MVE) selection with cardinality and
box constraints. The model solves

minimize   0.5·γ·x'Σₛx − μ'x
subject to ∑ x_i = 1,  m ≤ ∑ v_i ≤ k,
           fmin_i·v_i ≤ x_i ≤ fmax_i·v_i,  v_i ∈ {0,1},

where Σₛ is Σ stabilized by `_prep_S(Σ, epsilon, stabilize_Σ)` if `stabilize_Σ=true`,
else the symmetrized Σ. After solving (and optional bound expansion rounds):

- If `use_refit == false`: returns the MIQP portfolio (selection from `v`, SR from `x`,
  and `mve_weights = x` iff `compute_weights=true`).
- If `use_refit == true`: uses the MIQP **selection** only, then **refits** MVE on that
  support via `compute_mve_sr` and (optionally) `compute_mve_weights`. The SR reported
  is the **refit** SR.

Arguments are as usual; `γ` is passed through to `compute_mve_weights` in the refit branch.
"""
function mve_miqp_heuristic_search(
    μ::AbstractVector, Σ::AbstractMatrix; k::Int,
    m::Int=0, γ::Float64=1.0,
    fmin::AbstractVector=zeros(length(μ)),
    fmax::AbstractVector=ones(length(μ)),
    expand_rounds::Int=2, expand_factor::Float64=2.0, expand_tol::Float64=1e-7,
    mipgap::Float64=1e-4, time_limit::Real=Inf, threads::Int=0,
    x_start::Union{Nothing,AbstractVector}=nothing,
    v_start::Union{Nothing,AbstractVector}=nothing,
    compute_weights::Bool=false,
    use_refit::Bool=true,
    verbose::Bool=false, epsilon::Real=EPS_RIDGE,
    stabilize_Σ::Bool=true, do_checks::Bool=false
)
    N = length(μ)
    if do_checks
        size(Σ,1)==N && size(Σ,2)==N || error("Σ must be N×N.")
        1 ≤ k ≤ N                     || error("1 ≤ k ≤ N required.")
        0 ≤ m ≤ k                     || error("0 ≤ m ≤ k required.")
        length(fmin)==N && length(fmax)==N || error("fmin, fmax must be length N.")
        γ > 0                          || error("γ must be positive.")
        expand_rounds ≥ 0              || error("expand_rounds ≥ 0.")
        expand_factor > 0              || error("expand_factor > 0.")
        expand_tol ≥ 0                 || error("expand_tol ≥ 0.")
        mipgap ≥ 0                     || error("mipgap ≥ 0.")
        threads ≥ 0                    || error("threads ≥ 0.")
        x_start === nothing || length(x_start) == N || error("x_start length N.")
        v_start === nothing || length(v_start) == N || error("v_start length N.")
        all(isfinite, μ) && all(isfinite, Σ) || error("μ and Σ must be finite.")
        @inbounds for i in 1:N
            fmin[i] ≤ fmax[i] || error("Require fmin[i] ≤ fmax[i] for all i.")
        end
        # Feasibility quick check
        sum_largest_k = sum(partialsort!(collect(fmax), 1:k; rev=true))
        sum_smallest_k = sum(partialsort!(collect(fmin), 1:k))
        sum_largest_k ≥ 1 || error("Infeasible caps: k * max fmax must allow sum(x)=1.")
        sum_smallest_k ≤ 1 || error("Infeasible lower bounds: sum of k smallest fmin exceeds 1.")
    end

    # Stabilize Σ once (or just symmetrize if stabilize_Σ=false) —
    # Use the same Σₛ for MIQP and any SR/weights computations.
    Σs = stabilize_Σ ? _prep_S(Σ, epsilon, true) : Symmetric((Σ + Σ')/2)

    fmin_work = Float64.(fmin)
    fmax_work = Float64.(fmax)

    sol = _solve_once!(μ, Σs, N, k, m, γ, fmin_work, fmax_work;
                       x0=x_start, v0=v_start,
                       mipgap=mipgap, time_limit=time_limit, threads=threads,
                       verbose=verbose, epsilon=epsilon)

    for _ in 1:expand_rounds
        active = 0
        @inbounds for i in 1:N
            if sol.v[i]==1 &&
               (sol.x[i] ≥ fmax_work[i]-expand_tol || sol.x[i] ≤ fmin_work[i]+expand_tol)
                active += 1
            end
        end
        active==0 && break
        _expand_bounds!(sol.x, sol.v, fmin_work, fmax_work, expand_factor, expand_tol)
        sol = _solve_once!(μ, Σs, N, k, m, γ, fmin_work, fmax_work;
                           x0=sol.x, v0=sol.v,
                           mipgap=mipgap, time_limit=time_limit, threads=threads,
                           verbose=verbose, epsilon=epsilon)
    end

    subset = findall(==(1), sol.v)

    if !use_refit
        # Original behavior: SR from MIQP x; weights optionally returned as x
        w = compute_weights ? sol.x : zeros(Float64, N)
        return (selection=subset,
                weights=w,
                sr=sol.sr,
                status=sol.status)
    else
        # Refit on MIQP support: SR/weights from closed-form MVE on Σs
        if isempty(subset)
            # No active names — nothing to refit
            w = zeros(Float64, N)
            return (selection=subset,
                    weights=w,
                    sr=0.0,
                    status=sol.status)
        end
        sr_refit = compute_mve_sr(μ, Σs; selection=subset,
                                  epsilon=epsilon, stabilize_Σ=false, do_checks=false)
        w_refit = compute_weights ?
                  compute_mve_weights(μ, Σs; selection=subset,
                                      epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
                  zeros(Float64, N)
        return (selection=subset,
                weights=w_refit,
                sr=sr_refit,
                status=sol.status)
    end
end

end # module
