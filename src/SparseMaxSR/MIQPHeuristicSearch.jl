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
    _solve_once!(μ, Σs, N, k, m, γ, fmin_loc, fmax_loc;
                 budget_constraint::Bool=false,
                 exactly_k::Bool=false,
                 x0=nothing, v0=nothing,
                 mipgap=1e-4, time_limit=200, threads=0,
                 verbose=false, epsilon=Utils.EPS_RIDGE)

Single MIQP solve with cardinality and box constraints:

minimize   0.5·γ·x'Σs x − μ'x
subject to (optional) ∑ x_i = 1                 if `budget_constraint=true`,
           m ≤ ∑ v_i ≤ k (or ∑ v_i = k         if `exactly_k=true`),
           indicator links:  v_i=0 ⇒ x_i=0,
                              v_i=1 ⇒ fmin_i ≤ x_i ≤ fmax_i,
           v_i ∈ {0,1}.

Notes on robustness:
- Uses *indicator constraints* instead of big-M linking; this tightens the LP
  relaxation and improves MIP search stability without changing the logic.
- Applies conservative numeric tolerances for CPLEX (epRHS/epOpt/epInt) and
  `NUMERICALEMPHASIS=1`, which can reduce acceptance of nearly-infeasible incumbents.
- Treats Σ as `Symmetric` explicitly in the quadratic form to avoid asymmetry noise.
"""
function _solve_once!(
    μ::AbstractVector, Σs::AbstractMatrix, N::Int, k::Int, m::Int, γ::Float64,
    fmin_loc::Vector{Float64}, fmax_loc::Vector{Float64};
    budget_constraint::Bool=false,
    exactly_k::Bool=false,
    x0::Union{Nothing,AbstractVector}=nothing,
    v0::Union{Nothing,AbstractVector}=nothing,
    mipgap::Float64=1e-4, time_limit::Real=200, threads::Int=0,
    verbose::Bool=false, epsilon::Real=Utils.EPS_RIDGE,
)
    model = Model(CPLEX.Optimizer)
    if !verbose
        set_silent(model)
    end
    if isfinite(time_limit)
        MOI.set(model, MOI.TimeLimitSec(), float(time_limit))
    end
    # Safe numeric settings (ignored silently if not supported by the local CPLEX)
    try
        set_optimizer_attribute(model, "MIPGap", mipgap)
        set_optimizer_attribute(model, "CPX_PARAM_NUMERICALEMPHASIS", 1)
        set_optimizer_attribute(model, "CPX_PARAM_EPRHS", 1e-9)
        set_optimizer_attribute(model, "CPX_PARAM_EPOPT", 1e-9)
        set_optimizer_attribute(model, "CPX_PARAM_EPINT", 1e-9)
        # mild polishing encouragement
        set_optimizer_attribute(model, "CPX_PARAM_POLISHAFTERNODE", 1)
        set_optimizer_attribute(model, "CPX_PARAM_MIPORDIND", 0)
    catch
        # ignore unsupported attributes
    end
    if threads > 0
        try set_optimizer_attribute(model, "CPX_PARAM_THREADS", threads) catch end
    end

    @variable(model, x[1:N])
    @variable(model, v[1:N], Bin)

    # Optional budget constraint (tied to normalize_weights at the call site)
    if budget_constraint
        @constraint(model, sum(x) == 1.0)
    end

    # Cardinality band or exact k
    if exactly_k
        @constraint(model, sum(v) == k)
    else
        @constraint(model, sum(v) <= k)
        @constraint(model, sum(v) >= m)
    end

    # Linking constraints
    # v_i ∈ {0,1}, and for each i:
    #   x_i ≤ fmax_i * v_i
    #   x_i ≥ fmin_i * v_i
    # When v_i = 0, these force x_i ≤ 0 and x_i ≥ 0 ⇒ x_i = 0, even if fmin_i < 0.
    @constraint(model, [i=1:N], x[i] <= fmax_loc[i] * v[i])
    @constraint(model, [i=1:N], x[i] >= fmin_loc[i] * v[i])
    
    # Explicit symmetric quadratic form (Σs is already stabilized/symmetrized by caller)
    Σsym = Symmetric(Matrix(Σs))
    @objective(model, Min, 0.5 * γ * dot(x, Σsym * x) - dot(μ, x))

    if x0 !== nothing
        @inbounds for i in 1:N
            set_start_value(x[i], x0[i])
        end
    end
    if v0 !== nothing
        @inbounds for i in 1:N
            set_start_value(v[i], v0[i])
        end
    end

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

    # Hard-threshold the binaries for reporting
    vv = [ (isfinite(vv[i]) && vv[i] ≥ 0.5) ? 1 : 0 for i in 1:N ]

    # Clean up negligible weights where v=0 to avoid epsilon noise in SR
    @inbounds for i in 1:N
        if vv[i] == 0 && abs(xx[i]) ≤ 1e-12
            xx[i] = 0.0
        end
    end

    sr = SharpeRatio.compute_sr(xx, μ, Σsym; epsilon=epsilon,
                                stabilize_Σ=false, do_checks=false)
    if !isfinite(sr)
        sr = -Inf
    end

    return (x = collect(xx),
            v = vv,
            sr = sr,
            status = termination_status(model),
            obj = objective_value(model))
end

"""
    _expand_bounds!(x, v, fmin_loc, fmax_loc, expand_factor, expand_tol) -> Bool

Heuristic bound expansion used by the MATLAB-style loop.
Returns `true` if any bound was actually touched (so we should re-solve).
"""
function _expand_bounds!(x::Vector{Float64}, v::Vector{Int},
                         fmin_loc::Vector{Float64}, fmax_loc::Vector{Float64},
                         expand_factor::Float64, expand_tol::Float64)::Bool
    N = length(x)
    touched = false
    @inbounds for i in 1:N
        if v[i] == 1
            if x[i] ≥ fmax_loc[i] - expand_tol
                fmax_loc[i] *= expand_factor
                touched = true
            elseif x[i] ≤ fmin_loc[i] + expand_tol
                if fmin_loc[i] < 0
                    fmin_loc[i] *= expand_factor
                    touched = true
                end
            end
        end
    end
    return touched
end

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    mve_miqp_heuristic_search(μ, Σ; k,
        # Cardinality & caps
        exactly_k::Bool = false,
        m::Union{Int,Nothing} = nothing,           # default: max(0, k-1)
        γ::Float64 = 1.0,
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
box constraints. The core model solves:

    minimize   0.5 · γ · x' Σₛ x  −  μ' x
    subject to (optional)  ∑ᵢ xᵢ = 1                 # only if `normalize_weights = true`
               m ≤ ∑ᵢ vᵢ ≤ k      (or ∑ᵢ vᵢ = k if `exactly_k=true`)
               vᵢ = 0  ⇒  xᵢ = 0
               vᵢ = 1  ⇒  fminᵢ ≤ xᵢ ≤ fmaxᵢ
               vᵢ ∈ {0,1}

where Σₛ is a stabilized/symmetrized covariance used within the optimizer and for Sharpe
evaluations. By default, we compute Σₛ = `Utils._prep_S(Σ, epsilon, true)`; if
`stabilize_Σ=false`, we use a one-time symmetrization `Σₛ = (Σ + Σ')/2`.

# Key behaviors

- **Budget constraint**:
  - By **default** (`normalize_weights=false`) there is **no** ∑x=1 constraint, matching
    the MATLAB MIQP/QCP setup. Returned Sharpe ratios are *scale-invariant*.
  - If `normalize_weights=true`, the model **adds** ∑x=1 and returned weights are also
    rescaled by `Utils.normalize_weights` (relative-L1 safeguard) in the **vanilla**
    branch and by `SharpeRatio.compute_mve_weights(...; normalize_weights=true)` in
    the **refit** branch.

- **Cardinality**:
  - Default is a **band**: `exactly_k=false` with lower bound `m = max(0, k-1)` and
    upper bound `k`. Set `exactly_k=true` to enforce **exactly k**.

- **Refit vs raw MIQP** (`use_refit`):
  - `use_refit=true` (default): use MIQP **selection** then compute the **exact MVE**
    Sharpe on that support via `SharpeRatio.compute_mve_sr(μ, Σₛ; selection=...)`.
    If `compute_weights=true`, weights are `compute_mve_weights(μ, Σₛ; selection=...,
    normalize_weights=normalize_weights)`.
  - `use_refit=false`: keep the **raw MIQP portfolio `x`**. If `compute_weights=true`,
    the returned weights are `x` (post-normalized if `normalize_weights=true`).

- **Scale & Sharpe**:
  `SR = (μ' w) / √(w' Σₛ w)` is scale-invariant, so budgets do not affect the SR
  (for positive rescaling). We ensure Σₛ is consistently used for SR.

# Arguments

- `μ::AbstractVector` (length N): mean excess returns.
- `Σ::AbstractMatrix` (N×N): covariance matrix.
- `k::Int`: cardinality upper bound (and target if `exactly_k=true`).

- `exactly_k::Bool=false`: if true, enforces ∑v = k; else `m ≤ ∑v ≤ k`.
- `m::Union{Int,Nothing}=nothing`: lower cardinality bound; defaults to `max(0, k-1)`.
- `γ::Float64=1.0`: overall risk-aversion scale (just re-scales the quadratic term).
- `fmin, fmax::AbstractVector` (length N): lower/upper caps active when `v_i=1`.

- `expand_rounds::Int=20`, `expand_factor::Float64=3.0`, `expand_tol::Float64=1e-2`:
  bound-expansion heuristic; if a chosen weight sits within `expand_tol` of a bound,
  that bound is multiplied by `expand_factor` and the MIQP is re-solved (up to
  `expand_rounds` times). This mimics MATLAB’s progressive relaxation.

- `mipgap::Float64=1e-4`, `time_limit::Real=200`, `threads::Int=0`:
  CPLEX controls. Warm starts `x_start`, `v_start` are optional and reused across
  expansion rounds when bounds actually change.

- `compute_weights::Bool=false`: if true, returns a full-length weight vector
  (zeros off support). Otherwise returns zeros.
- `normalize_weights::Bool=false`: toggles ∑x=1 in the MIQP and post-normalization of
  outputs (vanilla) or closed-form MVE weights (refit).
- `use_refit::Bool=true`: MIQP selection + closed-form MVE on the support.

- `epsilon::Real=Utils.EPS_RIDGE`, `stabilize_Σ::Bool=true`:
  numerical stabilization of Σ and SR computation.
- `verbose::Bool=false`: solver output.
- `do_checks::Bool=false`: input validation & quick feasibility screens.

# Returns

A named tuple:
- `selection::Vector{Int}` — indices with `vᵢ=1` in the final MIQP solve.
- `weights::Vector{Float64}` — either refit MVE weights on `selection`
  (if `use_refit && compute_weights`) or the raw MIQP `x` (if `!use_refit && compute_weights`),
  or an all-zero vector otherwise.
- `sr::Float64` — Sharpe ratio computed on Σₛ (refit: exact MVE SR; vanilla: SR of `x`).
- `status` — MOI termination status from the last MIQP solve.

# Computational notes (robustness, no API impact)

- CPLEX numeric emphasis and tight tolerances reduce acceptance of nearly-infeasible
  incumbents; these parameters do **not** alter the mathematical optimum.
- Indicator constraints produce a tighter relaxation than big-M linking with the same
  logic (`v=0 ⇒ x=0`, `v=1 ⇒ fmin ≤ x ≤ fmax`), typically speeding up the search and
  improving stability.
- We avoid redundant re-solves in the expansion loop when no bound is actually tight.
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

    # Effective lower cardinality bound (band by default; exact k if requested)
    m_eff = isnothing(m) ? max(0, k-1) : m
    if exactly_k
        m_eff = k
    end

    # Input checks (safe; only triggered if do_checks=true)
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
        # If we plan to enforce ∑x=1, perform a very cheap feasibility screen.
        if normalize_weights
            smin = 0.0
            smax = 0.0
            @inbounds for i in 1:N
                smin += min(0.0, fmin[i])
                smax += max(0.0, fmax[i])
            end
            (1.0 ≥ smin - 1e-12 && 1.0 ≤ smax + 1e-12) ||
                error("Budget ∑x=1 incompatible with caps (quick screen).")
        end
    end

    # Stabilize Σ once (or just symmetrize if requested)
    Σs = stabilize_Σ ? Utils._prep_S(Σ, epsilon, true) : Symmetric((Σ + Σ')/2)

    fmin_work = Float64.(fmin)
    fmax_work = Float64.(fmax)

    # Budget is tied to normalize_weights: only impose if we intend sum(w)=1 output
    budget_on = normalize_weights

    # First solve
    sol = _solve_once!(μ, Σs, N, k, m_eff, γ, fmin_work, fmax_work;
                       budget_constraint=budget_on,
                       exactly_k=exactly_k,
                       x0=x_start, v0=v_start,
                       mipgap=mipgap, time_limit=time_limit, threads=threads,
                       verbose=verbose, epsilon=epsilon)

    # MATLAB-like expansion rounds; only re-solve if a bound was actually touched
    for _ in 1:expand_rounds
        touched = _expand_bounds!(sol.x, sol.v, fmin_work, fmax_work,
                                  expand_factor, expand_tol)
        touched || break
        sol = _solve_once!(μ, Σs, N, k, m_eff, γ, fmin_work, fmax_work;
                           budget_constraint=budget_on,
                           exactly_k=exactly_k,
                           x0=sol.x, v0=sol.v,
                           mipgap=mipgap, time_limit=time_limit, threads=threads,
                           verbose=verbose, epsilon=epsilon)
    end

    subset = findall(==(1), sol.v)

    # VANILLA: keep the MIQP portfolio (optionally normalize)
    if !use_refit
        w = compute_weights ? copy(sol.x) : zeros(Float64, N)
        if compute_weights && normalize_weights
            w = Utils.normalize_weights(w)
        end
        return (selection=subset, weights=w, sr=sol.sr, status=sol.status)
    end

    # REFIT: closed-form MVE on the selected support
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

end # module