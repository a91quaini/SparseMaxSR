
module MIQPHeuristicSearch
# Robust MIQP heuristic for sparse MVE selection
#
# This implementation is defensive against malformed inputs and solver hiccups:
# - No use of uninitialized arrays: all vectors are fully materialized Float64.
# - Bounds (fmin/fmax) are sanitized, ordered elementwise, and widened by a
#   progressive expansion heuristic if solutions stick to caps.
# - Σ is symmetrized and (optionally) ridge-stabilized once via Utils._prep_S.
# - The budget constraint ∑x=1 is toggled by `normalize_weights`.
# - `use_refit=true` recomputes exact MVE SR/weights on the selected support.
# - All solver attributes are set in try/catch blocks for portability.
# - Thread-safe: BLAS threads are temporarily set to 1 during solve and restored.
#
# Public API:
#   mve_miqp_heuristic_search(μ, Σ; k, exactly_k=false, m=1, γ=1.0,
#                              fmin=-0.25, fmax=0.25, expand_rounds=20,
#                              expand_factor=3.0, expand_tol=1e-2,
#                              mipgap=1e-4, time_limit=200, threads=1,
#                              compute_weights=true, normalize_weights=false,
#                              use_refit=false, verbose=false,
#                              epsilon=Utils.EPS_RIDGE, stabilize_Σ=true,
#                              do_checks=false)
#
# Returns NamedTuple{(:selection,:weights,:sr,:status)}.
#

using LinearAlgebra
using JuMP
import MathOptInterface as MOI
import LinearAlgebra.BLAS

# Prefer CPLEX if available in the environment; otherwise fall back to default.
# (The package declares CPLEX as a dependency, but we still guard the import.)
const _has_cplex = try
    @eval using CPLEX
    true
catch
    false
end

using ..Utils
using ..SharpeRatio

export mve_miqp_heuristic_search

# ──────────────────────────────────────────────────────────────────────────────
# Utilities (sanitizers & small helpers)
# ──────────────────────────────────────────────────────────────────────────────

# Return a dense Float64 vector of length N from:
#  - nothing  → fill(default, N)
#  - scalar   → fill(scalar, N)
#  - function → v = f(N); length(v)==N ? v : fill(v[1], N) if nonempty
#  - vector   → length==N ? copy : fill(first, N) if nonempty
function _as_float_vec(x, N::Int, default::Float64)
    if x === nothing
        return fill(default, N)
    elseif x isa Number
        return fill(Float64(x), N)
    elseif x isa Function
        v = x(N)
        if length(v) == N
            return Float64.(v)
        elseif length(v) >= 1
            return fill(Float64(v[1]), N)
        else
            return fill(default, N)
        end
    else
        if length(x) == N
            return Float64.(x)
        elseif length(x) >= 1
            return fill(Float64(x[1]), N)
        else
            return fill(default, N)
        end
    end
end

# Replace non-finite entries with a default value (in-place).
function _sanitize_nonfinite!(v::Vector{Float64}, default::Float64)
    @inbounds for i in eachindex(v)
        vi = v[i]
        v[i] = isfinite(vi) ? vi : default
    end
    return v
end

# Ensure elementwise fmin[i] ≤ fmax[i]; swap when needed (in-place).
function _enforce_order!(fmin::Vector{Float64}, fmax::Vector{Float64})
    @inbounds for i in eachindex(fmin, fmax)
        if fmin[i] > fmax[i]
            tmp = fmin[i]; fmin[i] = fmax[i]; fmax[i] = tmp
        end
    end
    return nothing
end

# Heuristic bound expansion: if x hits a bound within tol, expand symmetric band.
function _expand_bounds!(x::Vector{Float64}, v::Vector{Int},
                         fmin::Vector{Float64}, fmax::Vector{Float64},
                         factor::Float64, tol::Float64)::Bool
    touched = false
    @inbounds for i in eachindex(x)
        if v[i] == 1 && (abs(x[i]-fmin[i]) ≤ tol || abs(x[i]-fmax[i]) ≤ tol)
            range = fmax[i] - fmin[i]
            δ = (factor - 1.0) * range * 0.5
            fmin[i] -= δ
            fmax[i] += δ
            touched = true
        end
    end
    return touched
end

# Fallback result with zeros and a status Symbol (default :ERROR)
_fallback_result(N::Int, status::Symbol=:ERROR) =
    (selection = Int[], weights = zeros(Float64, N), sr = 0.0, status = status)

# ──────────────────────────────────────────────────────────────────────────────
# Core single MIQP solve
# ──────────────────────────────────────────────────────────────────────────────

function _solve_once!(μ::Vector{Float64},
                      Σs::Symmetric{Float64,Matrix{Float64}},
                      N::Int, k::Int, m::Int, γ::Float64,
                      fmin::Vector{Float64}, fmax::Vector{Float64};
                      budget_constraint::Bool,
                      exactly_k::Bool,
                      mipgap::Float64,
                      time_limit::Real,
                      threads::Int,
                      verbose::Bool,
                      epsilon::Float64,
                      x0::Union{Nothing,AbstractVector}=nothing,   
                      v0::Union{Nothing,AbstractVector}=nothing)   

    # Save current BLAS thread count and set to 1 for thread safety
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    
    try
        # Build model with CPLEX if present; otherwise default solver-less Model()
        model = _has_cplex ? Model(CPLEX.Optimizer) : Model()
        if !verbose
            set_silent(model)
        end

        # Generic attribute settings guarded by try/catch — portable across solvers
        try
            if _has_cplex
                set_optimizer_attribute(model, "MIPGap", mipgap)
                set_optimizer_attribute(model, "CPX_PARAM_NUMERICALEMPHASIS", 1)
                set_optimizer_attribute(model, "CPX_PARAM_EPRHS", 1e-9)
                set_optimizer_attribute(model, "CPX_PARAM_EPOPT", 1e-9)
                set_optimizer_attribute(model, "CPX_PARAM_EPINT", 1e-9)
                set_optimizer_attribute(model, "CPX_PARAM_POLISHAFTERNODE", 1)
                set_optimizer_attribute(model, "CPX_PARAM_THREADS", threads)
            else
                MOI.set(model, MOI.RawParameter("MIPGap"), mipgap)
                MOI.set(model, MOI.RawParameter("Threads"), threads)
            end
        catch
            # ignore unsupported attributes
        end
        try
            if isfinite(time_limit)
                MOI.set(model, MOI.TimeLimitSec(), float(time_limit))
            end
        catch
        end

        @variable(model, fmin[i] <= x[i=1:N] <= fmax[i])
        @variable(model, v[i=1:N], Bin)

        # ── Warm starts (if provided)
        if x0 !== nothing
            @inbounds for i in 1:N
                set_start_value(x[i], Float64(x0[i]))
            end
        end
        if v0 !== nothing
            @inbounds for i in 1:N
                set_start_value(v[i], Int(v0[i]))
            end
        end

        # Budget constraint only if we aim to return sum(w)=1 weights
        if budget_constraint
            @constraint(model, sum(x) == 1.0)
        end

        # Cardinality: band or exact k
        if exactly_k
            @constraint(model, sum(v) == k)
        else
            @constraint(model, m <= sum(v) <= k)
        end

        # Linking constraints (big-M style with local bounds)
        @constraint(model, [i=1:N], x[i] <= fmax[i] * v[i])
        @constraint(model, [i=1:N], x[i] >= fmin[i] * v[i])

        # Robust quadratic objective: Min 0.5*γ x'Σs x - μ'x
        @objective(model, Min, 0.5 * γ * dot(x, Σs * x) - dot(μ, x))

        optimize!(model)

        # If no result, return error tuple
        rc = try MOI.get(model, MOI.ResultCount()) catch; 0 end
        if rc == 0
            st = try termination_status(model) catch; :ERROR end
            return (x=zeros(Float64, N), v=zeros(Int, N), sr=-Inf, status=st, obj=NaN)
        end

        # Extract values, sanitize
        xx = try value.(x) catch; fill(NaN, N) end
        vv = try value.(v) catch; fill(NaN, N) end

        # Hard threshold binaries; zero out tiny weights when v=0
        @inbounds for i in 1:N
            vi = (isfinite(vv[i]) && vv[i] ≥ 0.5) ? 1 : 0
            vv[i] = vi
            xi = isfinite(xx[i]) ? xx[i] : 0.0
            if vi == 0 && abs(xi) ≤ 1e-12
                xi = 0.0
            end
            xx[i] = xi
        end

        # Compute SR on Σs (no extra stabilization)
        sr = SharpeRatio.compute_sr(xx, μ, Σs; epsilon=epsilon, stabilize_Σ=false, do_checks=false)
        sr = isfinite(sr) ? sr : -Inf

        st = try termination_status(model) catch; :UNKNOWN end
        obj = try objective_value(model) catch; NaN end
        return (x=Float64.(xx), v=Int.(vv), sr=sr, status=st, obj=obj)
    finally
        # Restore original BLAS thread count
        BLAS.set_num_threads(old_blas_threads)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

"""
    mve_miqp_heuristic_search(μ, Σ;
        k::Int,
        exactly_k::Bool=false,
        m::Union{Nothing,Int}=1,
        γ::Real=1.0,
        fmin=-0.25,
        fmax=0.25,
        expand_rounds::Int=20,
        expand_factor::Float64=3.0,
        expand_tol::Float64=1e-2,
        mipgap::Float64=1e-4,
        time_limit::Real=200,
        threads::Int=1,
        compute_weights::Bool=true,
        normalize_weights::Bool=false,
        use_refit::Bool=false,
        verbose::Bool=false,
        epsilon::Real=Utils.EPS_RIDGE,
        stabilize_Σ::Bool=true,
        do_checks::Bool=false
    ) -> NamedTuple{(:selection,:weights,:sr,:status)}

Mixed‑Integer Quadratic heuristic for mean–variance efficient (MVE) selection.

Model:
    minimize   0.5·γ·x'Σₛ x − μ'x
    s.t.       m ≤ ∑ vᵢ ≤ k     (or ∑ vᵢ = k if `exactly_k`)
               vᵢ ∈ {0,1},  fminᵢ vᵢ ≤ xᵢ ≤ fmaxᵢ vᵢ
               (optionally) ∑ xᵢ = 1   if `normalize_weights=true`

Key behaviors:
- `normalize_weights=false` (default): no budget constraint; SR is scale‑invariant.
- `use_refit=true`: compute exact MVE SR and weights on the selected support.
- Σ is prepared once via `Utils._prep_S(Σ, epsilon, stabilize_Σ)`.

Returns `(selection, weights, sr, status)`.
"""
function mve_miqp_heuristic_search(
    μ::AbstractVector, Σ::AbstractMatrix; k::Int,
    exactly_k::Bool=false,
    m::Union{Int,Nothing}=1,
    γ::Float64=1.0,
    fmin::AbstractVector=fill(-0.25,length(μ)),
    fmax::AbstractVector=fill(0.25,length(μ)),
    expand_rounds::Int=20, expand_factor::Float64=3.0, expand_tol::Float64=1e-2,
    mipgap::Float64=1e-4, time_limit::Real=200, threads::Int=1,
    x_start::Union{Nothing,AbstractVector}=nothing,     
    v_start::Union{Nothing,AbstractVector}=nothing,     
    compute_weights::Bool=true,
    normalize_weights::Bool=false,                      # toggles budget
    use_refit::Bool=false,
    verbose::Bool=false, epsilon::Real=Utils.EPS_RIDGE,
    stabilize_Σ::Bool=true, do_checks::Bool=false
)

    N = length(μ)
    μf = Float64.(μ)

    # Effective cardinality lower bound
    m_eff = exactly_k ? k : (isnothing(m) ? max(0, k-1) : m)    

    if do_checks
        size(Σ,1)==N && size(Σ,2)==N || error("Σ must be N×N.")
        1 ≤ k ≤ N || error("1 ≤ k ≤ N required.")
        0 ≤ m_eff ≤ k || error("0 ≤ m ≤ k required.")
        length(fmin)==N && length(fmax)==N || error("fmin, fmax must be length N.")
        γ > 0 || error("γ must be positive.")
        expand_rounds ≥ 0 || error("expand_rounds ≥ 0.")
        expand_factor > 0 || error("expand_factor > 0.")
        expand_tol ≥ 0 || error("expand_tol ≥ 0.")
        mipgap ≥ 0 || error("mipgap ≥ 0.")
        threads ≥ 0 || error("threads ≥ 0.")
        x_start === nothing || length(x_start) == N || error("x_start length N.")
        v_start === nothing || length(v_start) == N || error("v_start length N.")
        all(isfinite, μ) && all(isfinite, Σ) || error("μ and Σ must be finite.")
        @inbounds for i in 1:N
            fmin[i] ≤ fmax[i] || error("Require fmin[i] ≤ fmax[i] for all i.")
        end
        if normalize_weights
            smin = 0.0; smax = 0.0
            @inbounds for i in 1:N
                smin += min(0.0, fmin[i])
                smax += max(0.0, fmax[i])
            end
            (1.0 ≥ smin - 1e-12 && 1.0 ≤ smax + 1e-12) ||
                error("Budget ∑x=1 incompatible with caps (quick screen).")
        end
    end

    # Prepare Σ once
    Σs = Utils._prep_S(Σ, epsilon, stabilize_Σ)

    # Robust, sanitized bounds
    fmin_work = _as_float_vec(fmin, N, 0.0)
    fmax_work = _as_float_vec(fmax, N, 1.0)
    _sanitize_nonfinite!(fmin_work, 0.0)
    _sanitize_nonfinite!(fmax_work, 1.0)
    _enforce_order!(fmin_work, fmax_work)

    # Budget toggled by normalize_weights
    budget_on = normalize_weights

    # First solve with try/catch guard
    sol = try
        _solve_once!(μ, Σs, N, k, m_eff, γ, fmin_work, fmax_work;
                       budget_constraint=budget_on,
                       exactly_k=exactly_k,
                       x0=x_start, v0=v_start,             # <── pass warm starts
                       mipgap=mipgap, time_limit=time_limit, threads=threads,
                       verbose=verbose, epsilon=epsilon)
    catch err
        @debug "MIQP initial solve failed" err
        return _fallback_result(N, :ERROR)
    end

    # Progressive bound expansion (MATLAB-like); only re-solve if a bound was hit
    for _ in 1:expand_rounds
        touched = _expand_bounds!(sol.x, sol.v, fmin_work, fmax_work, expand_factor, expand_tol)
        touched || break
        sol = try
            _solve_once!(μf, Σs, N, k, m_eff, Float64(γ),
                         fmin_work, fmax_work;
                         budget_constraint=budget_on,
                         exactly_k=exactly_k,
                         mipgap=Float64(mipgap),
                         time_limit=time_limit,
                         threads=threads,
                         verbose=verbose,
                         epsilon=Float64(epsilon))
        catch err
            @debug "MIQP expansion solve failed" err
            return _fallback_result(N, :ERROR)
        end
    end

    sel = findall(==(1), sol.v)

    # VANILLA branch: keep MIQP portfolio x
    if !use_refit
        w = compute_weights ? copy(sol.x) : zeros(Float64, N)
        if compute_weights && normalize_weights
            w = Utils.normalize_weights(w)
        end
        sr_out = isfinite(sol.sr) ? sol.sr : 0.0
        return (selection=sel, weights=w, sr=sr_out, status=sol.status)
    end

    # REFIT branch: exact MVE on support
    if isempty(sel)
        return (selection=Int[], weights=zeros(Float64, N), sr=0.0, status=sol.status)
    end

    sr_refit = SharpeRatio.compute_mve_sr(μf, Σs; selection=sel,
                                          epsilon=epsilon, stabilize_Σ=false, do_checks=false)
    sr_refit = isfinite(sr_refit) ? sr_refit : 0.0

    w_refit = compute_weights ?
        SharpeRatio.compute_mve_weights(μf, Σs; selection=sel,
                                        normalize_weights=normalize_weights,
                                        epsilon=epsilon, stabilize_Σ=false, do_checks=false) :
        zeros(Float64, N)
    if compute_weights && !(sum(abs, w_refit) > 1e-12)
        w_refit .= 0.0
    end

    return (selection=sel, weights=w_refit, sr=sr_refit, status=sol.status)
end

end # module
