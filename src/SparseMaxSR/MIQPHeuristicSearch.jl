module MIQPHeuristicSearch

using LinearAlgebra
using Statistics
using JuMP
import MathOptInterface as MOI
using CPLEX
using ..Utils
using ..SharpeRatio

export mve_miqp_heuristic_search

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    _solve_once!(μ, Σ, N, k, m, γ, fmin_loc, fmax_loc;
                 budget_constraint::Bool=false,            # default: no budget
                 exactly_k::Bool=false,
                 x0=nothing, v0=nothing,
                 mipgap=1e-4, time_limit=200, threads=0,   # defaults mimic MATLAB setup
                 verbose=false, epsilon=Utils.EPS_RIDGE)

Single MIQP solve with cardinality and box constraints:

minimize   0.5·γ·x'Σx − μ'x
subject to (optional) ∑ x_i = 1  if `budget_constraint=true`,
           and m ≤ ∑ v_i ≤ k (or ∑ v_i = k if `exactly_k=true`),
           fmin_i·v_i ≤ x_i ≤ fmax_i·v_i, v_i ∈ {0,1}.
"""
function _solve_once!(
    μ::AbstractVector, Σ::AbstractMatrix, N::Int, k::Int, m::Int, γ::Float64,
    fmin_loc::Vector{Float64}, fmax_loc::Vector{Float64};
    budget_constraint::Bool=false,
    exactly_k::Bool=false,
    x0::Union{Nothing,AbstractVector}=nothing,
    v0::Union{Nothing,AbstractVector}=nothing,
    mipgap::Float64=1e-4, time_limit::Real=200, threads::Int=0,
    verbose::Bool=false, epsilon::Real=Utils.EPS_RIDGE,
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

    # Budget is OPTIONAL now:
    if budget_constraint
        @constraint(model, sum(x) == 1.0)
    end

    if exactly_k
        @constraint(model, sum(v) == k)
    else
        @constraint(model, sum(v) <= k)
        @constraint(model, sum(v) >= m)
    end
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
    sr = SharpeRatio.compute_sr(xx, μ, Σ; epsilon=epsilon, stabilize_Σ=false, do_checks=false)

    return (x = collect(xx),
            v = vv,
            sr = sr,
            status = termination_status(model),
            obj = objective_value(model))
end

"""
    _expand_bounds!(x, v, fmin_loc, fmax_loc, expand_factor, expand_tol)
"""
function _expand_bounds!(x::Vector{Float64}, v::Vector{Int},
                         fmin_loc::Vector{Float64}, fmax_loc::Vector{Float64},
                         expand_factor::Float64, expand_tol::Float64)
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
        # Cardinality & caps
        exactly_k::Bool = false,
        m::Union{Int,Nothing} = nothing,           # default: max(0, k-1)
        fmin::AbstractVector = zeros(length(μ)),
        fmax::AbstractVector = ones(length(μ)),

        # Heuristic bound expansion (MATLAB-like)
        expand_rounds::Int = 20,
        expand_factor::Float64 = 3.0,
        expand_tol::Float64 = 1e-2,

        # MIP solve controls
        mipgap::Float64 = 1e-4,
        time_limit::Real = 200,
        threads::Int = 0,
        x_start::Union{Nothing,AbstractVector} = nothing,
        v_start::Union{Nothing,AbstractVector} = nothing,

        # Outputs & refit
        compute_weights::Bool = false,
        normalize_weights::Bool = false,   # also controls budget constraint (see Key behaviors)
        use_refit::Bool = true,

        # Numerics & checks
        epsilon::Real = Utils.EPS_RIDGE,
        stabilize_Σ::Bool = true,
        verbose::Bool = false,
        do_checks::Bool = false
    ) -> NamedTuple{(:selection, :weights, :sr, :status)}

Heuristic MIQP for **mean–variance efficient (MVE) selection** with cardinality and
box constraints. The core model solves

    minimize   0.5 · x' Σₛ x  −  μ' x
    subject to (optional)  ∑ᵢ xᵢ = 1            # only if `normalize_weights = true`
               m ≤ ∑ᵢ vᵢ ≤ k    (or ∑ᵢ vᵢ = k if `exactly_k=true`)
               fminᵢ · vᵢ ≤ xᵢ ≤ fmaxᵢ · vᵢ,    vᵢ ∈ {0,1}

where Σₛ = `Utils._prep_S(Σ, epsilon, stabilize_Σ)` if stabilization is enabled; otherwise
Σ is symmetrized once and reused.

The routine optionally performs **bound-expansion rounds**: if any active weight sits
within `expand_tol` of its bound, that bound is multiplied by `expand_factor` and the
MIQP is re-solved (up to `expand_rounds` times). This mimics the behavior of the
original MATLAB code that progressively relaxes caps when the incumbent hits them.

# Key behaviors

- **Budget constraint**:
  - By **default** (`normalize_weights=false`) there is **no** ∑x=1 constraint, matching the MATLAB QCP/MISOCO setup.
  - If `normalize_weights=true`, the model **adds** ∑x=1 and the returned weights are additionally
    (re)scaled via `Utils.normalize_weights(w)` (relative L1 safeguard).

- **Cardinality**:
  - Default is a **band**: `exactly_k=false` with lower bound `m = max(0, k-1)` and upper bound `k`.
  - Set `exactly_k=true` to enforce **exactly k** active names.

- **Refit vs raw MIQP** (`use_refit`):
  - If `use_refit=true` (default): we use the **selection** from the MIQP and then
    compute the **exact MVE Sharpe** on that support via `SharpeRatio.compute_mve_sr(μ, Σₛ; selection=...)`.
    If `compute_weights=true`, weights are produced by
    `SharpeRatio.compute_mve_weights(μ, Σₛ; selection=..., normalize_weights=normalize_weights)`.
  - If `use_refit=false`: we keep the **raw MIQP portfolio `x`**. If `compute_weights=true`,
    the returned weights are `x` (optionally post-normalized if `normalize_weights=true`).

- **Scale & Sharpe**:
  Sharpe ratio `SR = (μ' w) / sqrt(w' Σₛ w)` is **scale-invariant**. Whether or not you
  normalize weights to sum to 1 does not change SR (for positive scaling).

# Arguments

- `μ::AbstractVector`: mean excess returns (length N).
- `Σ::AbstractMatrix`: covariance matrix (N×N).
- `k::Int`: upper bound on cardinality (and the target size if `exactly_k=true`).

- `exactly_k`: enforce ∑v = k when true; otherwise a band `m ≤ ∑v ≤ k`.
- `m`: lower bound on cardinality. If `nothing`, it defaults to `max(0, k-1)`.
- `fmin`, `fmax`: elementwise lower/upper caps used in linking constraints (on active names).

- `expand_rounds`, `expand_factor`, `expand_tol`: bound-expansion heuristic parameters.

- `mipgap`, `time_limit`, `threads`: solver controls passed to CPLEX.
  - `x_start`, `v_start`: optional warm starts for continuous and binary variables.

- `compute_weights`: return a weight vector (full length, zeros off support).
- `normalize_weights`: if true, **(i)** impose ∑x=1 in the MIQP and **(ii)** post-normalize returned
  weights using `Utils.normalize_weights`.
- `use_refit`: use MIQP **selection** then refit closed-form MVE on Σₛ; otherwise keep raw `x`.

- `epsilon`, `stabilize_Σ`: stabilization of Σ via ridge/symmetrization; applied once.
- `verbose`: solver output.
- `do_checks`: input validation toggles (dimensions, finiteness, etc.).

# Returns

A named tuple:
- `selection::Vector{Int}` — indices where `vᵢ=1` in the final MIQP solve.
- `weights::Vector{Float64}` — either refit MVE weights on `selection` (if `use_refit && compute_weights`)
  or the raw MIQP `x` (if `!use_refit && compute_weights`), or an all-zero vector otherwise.
- `sr::Float64` — Sharpe ratio:
  - refit branch: computed with `SharpeRatio.compute_mve_sr(μ, Σₛ; selection=selection)`;
  - vanilla branch: computed directly on the returned `weights` (or `x`), using Σₛ.
- `status` — MOI termination status from the last MIQP solve.

"""
function mve_miqp_heuristic_search(
    μ::AbstractVector, Σ::AbstractMatrix; k::Int,
    exactly_k::Bool=false,
    m::Union{Int,Nothing}=nothing,
    γ::Float64=1.0,
    fmin::AbstractVector=zeros(length(μ)),
    fmax::AbstractVector=ones(length(μ)),
    expand_rounds::Int=20, expand_factor::Float64=3.0, expand_tol::Float64=1e-2,
    mipgap::Float64=1e-4, time_limit::Real=200, threads::Int=0,
    x_start::Union{Nothing,AbstractVector}=nothing,
    v_start::Union{Nothing,AbstractVector}=nothing,
    compute_weights::Bool=false,
    normalize_weights::Bool=false,
    use_refit::Bool=true,
    verbose::Bool=false, epsilon::Real=Utils.EPS_RIDGE,
    stabilize_Σ::Bool=true, do_checks::Bool=false
)
    N = length(μ)
    m_eff = isnothing(m) ? max(0, k-1) : m             # default m := max(0, k-1)
    if exactly_k
        m_eff = k
    end

    if do_checks
        size(Σ,1)==N && size(Σ,2)==N || error("Σ must be N×N.")
        1 ≤ k ≤ N                     || error("1 ≤ k ≤ N required.")
        0 ≤ m_eff ≤ k                 || error("0 ≤ m ≤ k required.")
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
        # NOTE: no budget-feasibility checks here because ∑x=1 may be absent.
    end

    # Stabilize Σ once (or just symmetrize if stabilize_Σ=false)
    Σs = stabilize_Σ ? Utils._prep_S(Σ, epsilon, true) : Symmetric((Σ + Σ')/2)

    fmin_work = Float64.(fmin)
    fmax_work = Float64.(fmax)

    # Budget is tied to normalize_weights: only impose if we intend sum(w)=1 output.
    budget_on = normalize_weights

    sol = _solve_once!(μ, Σs, N, k, m_eff, γ, fmin_work, fmax_work;
                       budget_constraint=budget_on,
                       exactly_k=exactly_k,
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
        sol = _solve_once!(μ, Σs, N, k, m_eff, γ, fmin_work, fmax_work;
                           budget_constraint=budget_on,
                           exactly_k=exactly_k,
                           x0=sol.x, v0=sol.v,
                           mipgap=mipgap, time_limit=time_limit, threads=threads,
                           verbose=verbose, epsilon=epsilon)
    end

    subset = findall(==(1), sol.v)

    if !use_refit
        # Vanilla MIQP portfolio (optionally normalize with relative L1 safeguard)
        w = compute_weights ? copy(sol.x) : zeros(Float64, N)
        if compute_weights && normalize_weights
            w = Utils.normalize_weights(w)   # uses defaults: mode=:relative, tol=1e-6
        end
        return (selection=subset, weights=w, sr=sol.sr, status=sol.status)
    else
        # Refit on selection (Sharpe uses Σs; weight normalization controlled by flag)
        if isempty(subset)
            w = zeros(Float64, N)
            return (selection=subset, weights=w, sr=0.0, status=sol.status)
        end
        sr_refit = SharpeRatio.compute_mve_sr(μ, Σs; selection=subset,
                                  epsilon=epsilon, stabilize_Σ=false, do_checks=false)
        w_refit = compute_weights ?
                  SharpeRatio.compute_mve_weights(μ, Σs; selection=subset,
                                      normalize_weights=normalize_weights,
                                      epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
                  zeros(Float64, N)
        return (selection=subset, weights=w_refit, sr=sr_refit, status=sol.status)
    end
end

end # module
