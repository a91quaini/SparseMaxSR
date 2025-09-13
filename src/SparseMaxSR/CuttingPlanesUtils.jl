module CuttingPlanesUtils
"""
Utilities for k–sparse max-Sharpe selection via outer-approximation / cutting-planes
(cf. Bertsimas & Cory-Wright, 2022).

This module provides low-level pieces (dual subproblem, SOC/QCQP relaxations,
hill-climb warm starts, Kelley cuts). Concrete solvers are **optional**:
either pass an explicit `optimizer::Function` (e.g. `() -> Clarabel.Optimizer()`)
or rely on `SparseMaxSR.default_optimizer()` which is populated by extensions.
"""

using JuMP
using Random
using LinearAlgebra
import MathOptInterface as MOI
using ..SparseMaxSR: default_optimizer, EPS_RIDGE

export inner_dual,
       hillclimb,
       cplex_misocp_relaxation,
       portfolios_socp,
       portfolios_objective,
       warm_start,
       kelley_primal_cuts

# ──────────────────────────────────────────────────────────────────────────────
# inner_dual
# ──────────────────────────────────────────────────────────────────────────────

"""
    inner_dual(μ, Σ, inds; optimizer=nothing, attrs=NamedTuple(), epsilon=EPS_RIDGE)

Solve the dual QP for a fixed support `inds` (|inds| = k) in the k-sparse
max-Sharpe problem.

Maximize over (α ∈ ℝⁿ, λ ∈ ℝ, w ∈ ℝᵏ):
    -½‖α‖²  -½‖w‖²  + μᵀα + λ
subject to, for j = 1,…,k with i = inds[j]:
    w[j] ≥ (Σ α)[i] + λ

# Arguments
- `μ::AbstractVector{<:Real}`  : expected-return vector (length n)
- `Σ::AbstractMatrix{<:Real}`  : covariance matrix (n×n)
- `inds::AbstractVector{<:Integer}` : indices of the k nonzero assets (distinct)

# Keywords
- `optimizer` : MOI optimizer *factory* or instance. If `nothing`, uses
  `SparseMaxSR.default_optimizer()`.
- `attrs`     : NamedTuple/Dict of optimizer attributes passed to JuMP.
- `epsilon`   : optional ridge (≥ 0). If > 0, solve with `Σ + ε·mean(diag(Σ))·I`.

# Returns
NamedTuple:
- `ofv::Float64`                         : objective value (dual bound; `NaN` if unavailable)
- `α::Vector{Float64}`                   : dual vector
- `λ::Float64`                           : scalar multiplier
- `w::Vector{Float64}`                   : slack vector (length k, aligned with `inds`)
- `status::MOI.TerminationStatusCode`    : solver termination status
"""
function inner_dual(μ::AbstractVector{<:Real},
                    Σ::AbstractMatrix{<:Real},
                    inds::AbstractVector{<:Integer};
                    optimizer = nothing,
                    attrs = NamedTuple(),
                    epsilon::Real = EPS_RIDGE[])

    n = length(μ)
    (size(Σ,1) == n && size(Σ,2) == n) ||
        error("Σ must be $n×$n; got $(size(Σ)).")

    k = length(inds)
    (1 ≤ k ≤ n) || error("length(inds)=k must satisfy 1 ≤ k ≤ n.")
    all(1 .≤ inds .≤ n) || error("indices in `inds` must lie in 1:n.")
    if length(unique(inds)) != k
        error("`inds` contains duplicates; provide k distinct indices.")
    end
    sort!(inds)  # deterministic order

    # Optional ridge for numerical stability: Σeff = Σ + ε·mean(diag(Σ))·I
    Σeff = epsilon > 0 ? (Σ + (float(epsilon) * mean(diag(Σ))) * I) : Σ

    # Pick optimizer (factory or instance)
    optimizer === nothing && (optimizer = default_optimizer())
    model = Model(optimizer)
    set_silent(model)

    # Apply attributes, if any
    for (k_attr, v_attr) in pairs(attrs)
        set_optimizer_attribute(model, k_attr, v_attr)
    end

    # Variables
    @variable(model, α[1:n])
    @variable(model, λ)
    @variable(model, w[1:k])

    # Σα as a JuMP expression (safer than Σ*α for generic AbstractMatrix)
    @expression(model, Σα[i=1:n], sum(Σeff[i,j] * α[j] for j in 1:n))

    # Constraints: w[j] ≥ Σα[inds[j]] + λ
    @constraint(model, [j=1:k], w[j] ≥ Σα[inds[j]] + λ)

    # Objective: Max( -½‖α‖² - ½‖w‖² + μᵀα + λ )
    @objective(model, Max,
        -0.5 * sum(α[i]^2 for i in 1:n) -
        0.5 * sum(w[j]^2 for j in 1:k) +
         sum(μ[i] * α[i] for i in 1:n) +
         λ
    )

    optimize!(model)

    status = MOI.get(model, MOI.TerminationStatus())
    ofv    = try objective_value(model) catch; NaN end
    αv     = try value.(α)            catch; fill(NaN, n) end
    λv     = try value(λ)             catch; NaN end
    wv     = try value.(w)            catch; fill(NaN, k) end

    return (ofv = ofv, α = αv, λ = λv, w = wv, status = status)
end



#####################################
#### hillclimb
#####################################

"""
    hillclimb(μ, Σ, k, inds0;
              maxiter=50, optimizer=nothing, attrs=NamedTuple(),
              epsilon=0.0, tol=1e-12, keep_best=true)

Greedy support-improvement heuristic for the ℓ₀-constrained max-Sharpe problem.

At each iteration:
1. Solve the dual QP on the current support (via `inner_dual`).
2. Build a full-length slack vector `w_full` with `w[inds]=dual.w`, zeros elsewhere.
3. Set the next support to the indices of the top-k entries of `w_full`.
4. Stop when the support stabilizes or `maxiter` is reached.

# Arguments
- `μ::AbstractVector{<:Real}`       : expected returns (length n)
- `Σ::AbstractMatrix{<:Real}`       : covariance (n×n)
- `k::Integer`                      : desired cardinality (1 ≤ k ≤ n)
- `inds0::AbstractVector{<:Integer}`: initial support of size k (distinct)

# Keywords
- `maxiter::Int`     : maximum iterations (default 50)
- `optimizer`        : optimizer factory/instance passed to `inner_dual`
                       (if `nothing`, `default_optimizer()` is used downstream)
- `attrs`            : optimizer attributes forwarded to `inner_dual`
- `epsilon::Real`    : optional ridge in `inner_dual` (default 0.0)
- `tol::Real`        : tolerance parameter (kept for API stability; not used)
- `keep_best::Bool`  : if `true`, keep the best dual bound seen (not only the last)

# Returns
NamedTuple:
- `inds::Vector{Int}`                      : final (or best) support (size k, sorted)
- `w_full::Vector{Float64}`                : last (or best) full slack vector
- `ofv::Float64`                           : best dual objective bound observed
- `status::MOI.TerminationStatusCode`      : status at the best incumbent
- `iters::Int`                             : number of iterations performed
"""
function hillclimb(μ::AbstractVector{<:Real},
                   Σ::AbstractMatrix{<:Real},
                   k::Integer,
                   inds0::AbstractVector{<:Integer};
                   maxiter::Int = 50,
                   optimizer = nothing,
                   attrs = NamedTuple(),
                   epsilon::Real = 0.0,
                   tol::Real = 1e-12,  # kept for compatibility (unused)
                   keep_best::Bool = true)

    n = length(μ)
    (size(Σ,1) == n && size(Σ,2) == n) ||
        error("Σ must be $n×$n; got $(size(Σ)).")
    (1 ≤ k ≤ n) || error("k must satisfy 1 ≤ k ≤ n.")

    # sanitize initial support
    inds = sort!(unique!(collect(inds0)))
    length(inds) == k || error("inds0 must contain k=$k distinct indices.")
    all(1 .≤ inds .≤ n) || error("inds0 contains out-of-range indices (1..$n).")

    w_full = zeros(Float64, n)

    # book-keeping for the best incumbent
    best_inds   = copy(inds)
    best_w_full = similar(w_full)
    best_ofv    = -Inf
    best_status = MOI.OPTIMIZE_NOT_CALLED

    iters = 0
    while iters < maxiter
        iters += 1

        # 1) dual on current support
        res = inner_dual(μ, Σ, inds; optimizer=optimizer, attrs=attrs, epsilon=epsilon)

        # if solver failed badly, stop (keep current/best)
        if res.status ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
            break
        end

        # 2) full slack vector
        fill!(w_full, 0.0)
        @inbounds w_full[inds] .= res.w

        # 3) top-k indices via partial sort (faster than full sort)
        new_inds = partialsortperm(w_full, 1:k; rev=true)
        sort!(new_inds)

        # track best incumbent by dual bound (higher is better)
        if keep_best
            if res.ofv > best_ofv
                best_ofv    = res.ofv
                best_inds   = copy(new_inds)
                best_w_full .= w_full
                best_status = res.status
            end
        else
            best_ofv    = res.ofv
            best_inds   = copy(new_inds)
            best_w_full .= w_full
            best_status = res.status
        end

        # 4) convergence check: support unchanged ⇒ terminate
        if new_inds == inds
            break
        end

        inds = new_inds
    end

    return (inds   = best_inds,
            w_full = best_w_full,
            ofv    = best_ofv,
            status = best_status,
            iters  = iters)
end

#####################################
#### cplex misocp relaxation
#####################################

# Generic LP selection relaxation (legacy name kept for compatibility)
function cplex_misocp_relaxation(scores::AbstractVector{<:Real},
                                 k::Integer;
                                 optimizer = nothing,
                                 attrs = NamedTuple(),
                                 ΔT_max::Real = 60.0,
                                 enforce_equal::Bool = false,
                                 start_z::Union{Nothing,AbstractVector{<:Real}} = nothing)
    n = length(scores)
    (1 ≤ k ≤ n) || error("k must satisfy 1 ≤ k ≤ n (got k=$k, n=$n).")

    optimizer === nothing && (optimizer = default_optimizer())

    model = JuMP.Model(optimizer)
    JuMP.set_silent(model)

    try
        JuMP.set_optimizer_attribute(model, MathOptInterface.TimeLimitSec(), float(ΔT_max))
    catch
    end
    for (ka, va) in pairs(attrs)
        JuMP.set_optimizer_attribute(model, ka, va)
    end

    @variable(model, 0.0 ≤ z[1:n] ≤ 1.0)
    if enforce_equal
        @constraint(model, sum(z) == k)
    else
        @constraint(model, sum(z) ≤ k)
    end

    clean = similar(scores, Float64)
    @inbounds for i in 1:n
        s = float(scores[i])
        clean[i] = isfinite(s) ? s : 0.0
    end
    @objective(model, Max, dot(clean, z))

    if start_z !== nothing
        length(start_z) == n || error("start_z must have length n=$n.")
        @inbounds for i in 1:n
            set_start_value(z[i], clamp(float(start_z[i]), 0.0, 1.0))
        end
    end

    optimize!(model)

    status = MathOptInterface.get(model, MathOptInterface.TerminationStatus())
    ofv    = try objective_value(model) catch; NaN end
    zv     = try value.(z) catch; fill(0.0, n) end
    return (z = zv, ofv = ofv, status = status)
end


#####################################
#### socp / qcqp relaxation
#####################################

"""
    portfolios_socp(μ, Σ, γ, k;
                    optimizer=nothing, attrs=NamedTuple(),
                    epsilon=EPS_RIDGE, form=:qcqp)

Convex relaxation of the k-sparse max-Sharpe selection master problem.

Max over (α ∈ ℝⁿ, λ ∈ ℝ, w ∈ ℝⁿ, v ∈ ℝⁿ₊, t ≥ 0):
    -½‖α‖² + μᵀα + λ - ∑ᵢ vᵢ - k·t

Subject to:
    wᵢ ≥ (Σ α)ᵢ + λ
    (penalty link between v, t, w)  # see `form` below

`form` chooses how to link (v, t, w):
  • `:qcqp` (default):  vᵢ + t ≥ (γᵢ/2)·wᵢ²
  • `:rotated_soc`:      2 vᵢ t ≥ γᵢ wᵢ², vᵢ,t≥0

Returns `(ofv, α, λ, w, v, t, status)`.
"""
function portfolios_socp(μ::AbstractVector{<:Real},
                         Σ::AbstractMatrix{<:Real},
                         γ::AbstractVector{<:Real},
                         k::Integer;
                         optimizer = nothing,
                         attrs = NamedTuple(),
                         epsilon::Real = EPS_RIDGE[],
                         form::Symbol = :qcqp)

    n = length(μ)
    (size(Σ,1) == n && size(Σ,2) == n) ||
        error("Σ must be $n×$n; got $(size(Σ)).")
    length(γ) == n || error("γ must have length n=$n.")
    all(γ .≥ 0) || error("γ must be elementwise non-negative.")
    (k ≥ 0) || error("k must be non-negative.")

    # Optional ridge
    Σeff = epsilon > 0 ? (Σ + (float(epsilon) * mean(diag(Σ))) * I) : Σ

    # Optimizer
    optimizer === nothing && (optimizer = default_optimizer())
    model = Model(optimizer)
    set_silent(model)
    for (k_attr, v_attr) in pairs(attrs)
        set_optimizer_attribute(model, k_attr, v_attr)
    end

    # Variables
    @variable(model, α[1:n])
    @variable(model, λ)
    @variable(model, w[1:n])
    @variable(model, v[1:n] >= 0.0)
    @variable(model, t >= 0.0)

    # Σα expression
    @expression(model, Σα[i=1:n], sum(Σeff[i,j] * α[j] for j in 1:n))

    # Linking cuts: wᵢ ≥ (Σα)ᵢ + λ
    @constraint(model, [i=1:n], w[i] ≥ Σα[i] + λ)

    if form == :qcqp
        # vᵢ + t ≥ (γᵢ/2) · wᵢ²  (convex quadratic)
        @constraint(model, [i=1:n], v[i] + t ≥ (γ[i] / 2) * w[i]^2)
    elseif form == :rotated_soc
        # 2 vᵢ t ≥ γᵢ wᵢ² ⇒ (vᵢ, t, sqrt(γᵢ)·wᵢ) ∈ Qr
        sqrtγ = sqrt.(max.(γ, 0))
        for i in 1:n
            @constraint(model, [v[i], t, sqrtγ[i] * w[i]] in MOI.RotatedSecondOrderCone())
        end
    else
        error("Unknown form=$(form). Use :qcqp or :rotated_soc.")
    end

    # Objective: Max -½‖α‖² + μᵀα + λ - ∑v - k·t
    @objective(model, Max,
        -0.5 * sum(α[i]^2 for i in 1:n) +
         sum(μ[i] * α[i] for i in 1:n) +
         λ - sum(v) - k * t
    )

    optimize!(model)

    status = MOI.get(model, MOI.TerminationStatus())
    ofv    = try objective_value(model) catch; NaN end
    αv     = try value.(α) catch; fill(NaN, n) end
    λv     = try value(λ)  catch; NaN end
    wv     = try value.(w) catch; fill(NaN, n) end
    vv     = try value.(v) catch; fill(NaN, n) end
    tv     = try value(t)  catch; NaN end

    return (ofv = ofv, α = αv, λ = λv, w = wv, v = vv, t = tv, status = status)
end


#####################################
#### portfolios objective
#####################################

"""
    portfolios_objective(μ, Σ, γ, k, s;
                         optimizer=nothing, attrs=NamedTuple(),
                         epsilon=EPS_RIDGE, threshold=0.5, ensure_one=true)

Build an outer-approximation cut at point `s ∈ [0,1]^n`.

Returns `(p, grad, status)` where `p` is the dual value and
`grad[i] = -½ γ[i] * (w_full[i]^2)` with `w_full = (Σeff * α) .+ λ`.
"""
function portfolios_objective(μ::AbstractVector{<:Real},
                              Σ::AbstractMatrix{<:Real},
                              γ::AbstractVector{<:Real},
                              k::Integer,
                              s::AbstractVector{<:Real};
                              optimizer = nothing,
                              attrs = NamedTuple(),
                              epsilon::Real = EPS_RIDGE[],
                              threshold::Real = 0.5,
                              ensure_one::Bool = true)

    n = length(s)
    (length(μ) == n) || error("μ must have length n=$n.")
    (size(Σ,1) == n && size(Σ,2) == n) || error("Σ must be $n×$n; got $(size(Σ)).")
    (length(γ) == n) || error("γ must have length n=$n.")
    all(γ .≥ 0) || error("γ must be elementwise nonnegative.")
    (1 ≤ k ≤ n) || error("k must satisfy 1 ≤ k ≤ n.")

    # 1) Active set from threshold
    inds = findall(i -> s[i] > threshold, 1:n)

    # 2) Nonempty and ≤ k (deterministic)
    if isempty(inds)
        if ensure_one
            push!(inds, argmax(s))
        else
            error("Empty active set at evaluation point; set `ensure_one=true` or lower `threshold`.")
        end
    end
    if length(inds) > k
        topk = partialsortperm(view(s, inds), 1:k; rev=true)
        inds = sort!(collect(inds[topk]))
    else
        sort!(inds)
    end

    # 3) Dual on support
    optimizer === nothing && (optimizer = default_optimizer())
    dual = inner_dual(μ, Σ, inds; optimizer=optimizer, attrs=attrs, epsilon=epsilon)

    # 4) Build w_full with the SAME Σeff used in the dual
    Σeff = epsilon > 0 ? Σ + (float(epsilon) * mean(diag(Σ))) * I : Σ
    w_full = Vector{Float64}(Σeff * dual.α .+ dual.λ)

    # 5) Subgradient wrt s: elementwise square
    grad = @. -0.5 * γ * (w_full.^2)

    return (p = dual.ofv, grad = grad, status = dual.status)
end


#####################################
#### warm start
#####################################

"""
    warm_start(μ, Σ, γ, k;
               num_random_restarts=5, maxiter=50,
               optimizer=nothing, attrs=NamedTuple(),
               epsilon=EPS_RIDGE, rng=Random.default_rng(),
               include_greedy=true)

Generate a good initial indicator vector `s0 ∈ {0,1}^n` with exactly `k` ones
for the k-sparse max-Sharpe problem using a mix of deterministic greedy seeds
and random restarts, each refined by `hillclimb`.

Heuristics tried:
1. **SNR seed**: pick top-k by `μ[i] / √(Σ[ii] + ε̄)` (ε̄ = ridge scale).
2. **MV proxy seed**: compute `x = (Σ + ε̄ I) \\ μ` and pick top-k by `|x[i]|`.
3. `num_random_restarts` random k-subsets.

Each seed is improved with `hillclimb`, then scored by `portfolios_objective`;
the best dual bound `p` is kept.

# Arguments
- `μ::AbstractVector{<:Real}` : expected returns (length n)
- `Σ::AbstractMatrix{<:Real}` : covariance (n×n)
- `γ::AbstractVector{<:Real}` : nonnegative per-asset penalties (length n)
- `k::Integer`                : cardinality (1 ≤ k ≤ n)

# Keywords
- `num_random_restarts::Int` : number of random seeds (default 5)
- `maxiter::Int`             : hillclimb iterations per seed (default 50)
- `optimizer`                : optimizer passed to `hillclimb`/`portfolios_objective`
                              (if `nothing`, defaults are chosen inside those functions)
- `attrs`                    : optimizer attributes
- `epsilon::Real`            : ridge for stability (default 0.0); ε̄=ε·mean(diag(Σ))
- `rng::AbstractRNG`         : RNG for reproducibility (default `Random.default_rng()`)
- `include_greedy::Bool`     : include deterministic seeds (default `true`)

# Returns
- `s0::Vector{Float64}` : binary vector of length n with exactly k ones.
"""
function warm_start(μ::AbstractVector{<:Real},
                    Σ::AbstractMatrix{<:Real},
                    γ::AbstractVector{<:Real},
                    k::Integer;
                    num_random_restarts::Int = 5,
                    maxiter::Int = 50,
                    optimizer = nothing,
                    attrs = NamedTuple(),
                    epsilon::Real = EPS_RIDGE[],
                    rng::AbstractRNG = Random.default_rng(),
                    include_greedy::Bool = true)

    n = length(μ)
    (size(Σ,1) == n && size(Σ,2) == n) || error("Σ must be $n×$n; got $(size(Σ)).")
    length(γ) == n || error("γ must have length n=$n.")
    all(γ .≥ 0) || error("γ must be elementwise nonnegative.")
    (1 ≤ k ≤ n) || error("k must satisfy 1 ≤ k ≤ n.")

    # Ridge scale used for seeds and inner dual
    eps_scale = epsilon > 0 ? float(epsilon) * mean(diag(Σ)) : 0.0

    candidates = Vector{Vector{Int}}()

    if include_greedy
        # 1) SNR seed: μ[i] / sqrt(var_i + eps_scale)
        denom = sqrt.(diag(Σ) .+ eps_scale)
        snr = @. μ / max(denom, eps(Float64))
        push!(candidates, sort!(partialsortperm(snr, 1:k; rev=true)))

        # 2) MV proxy: x = (Σ + eps_scale I) \ μ ; pick largest |x|
        Σeff = eps_scale > 0 ? Σ + eps_scale * I : Σ
        x = try
            Σeff \ μ
        catch
            # fallback: diagonal preconditioner if Σ is ill-conditioned
            @. μ / max(diag(Σeff), eps(Float64))
        end
        push!(candidates, sort!(partialsortperm(abs.(x), 1:k; rev=true)))
    end

    # 3) Random seeds
    for _ in 1:max(0, num_random_restarts)
        push!(candidates, sort!(randperm(rng, n)[1:k]))
    end

    # Track the best seed by dual bound p
    best_p    = -Inf
    best_inds = candidates[1]  # placeholder; will be overwritten
    best_s    = zeros(Float64, n)

    # Reusable buffers
    s = zeros(Float64, n)

    for init_inds in candidates
        # Refine with hillclimb
        res = hillclimb(μ, Σ, k, init_inds;
                        maxiter=maxiter, optimizer=optimizer,
                        attrs=attrs, epsilon=epsilon, keep_best=true)

        # Build binary s from refined support
        fill!(s, 0.0)
        @inbounds s[res.inds] .= 1.0

        # Evaluate cut value at s (dual outer bound)
        cut = portfolios_objective(μ, Σ, γ, k, s;
                                   optimizer=optimizer, attrs=attrs, epsilon=epsilon)

        # Prefer higher p; if cut not optimal, fall back to hillclimb bound
        score = (cut.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)) ? cut.p : res.ofv

        if score > best_p
            best_p    = score
            best_inds = copy(res.inds)
            best_s   .= s
        end
    end

    # Ensure exactly k ones (defensive)
    if count(!iszero, best_s) != k
        fill!(best_s, 0.0)
        @inbounds best_s[best_inds] .= 1.0
    end

    return best_s
end


#####################################
#### kelley primal cuts
#####################################

"""
    kelley_primal_cuts(μ, Σ, γ, k, stab0, num_epochs;
                       optimizer=nothing, attrs=NamedTuple(),
                       epsilon=EPS_RIDGE, threshold=0.5,
                       lambda=0.10, delta_scale=2.0)

Generate up to `num_epochs` outer-approximation cuts at the root node via a
simple in–out stabilization scheme. At epoch r:

1. Solve the current root LP:  min t  s.t.  0≤s≤1,  ∑s ≤ k,  ∑s ≥ 1,  and cuts.
2. Let z★ = clamp(value(s), 0, 1). Update the stabilized center:
       stab ← 0.5*(stab + z★).
3. Trial point:
       z₀ = clamp(lambda*z★ + (1-lambda)*stab + δ, 0, 1),
   where δ = delta_scale*eps (elementwise).
4. Evaluate the outer function at z₀ with `portfolios_objective` and add the cut:
       t ≥ p + ∇ᵀ (s − z₀).

# Arguments
- `μ::AbstractVector{<:Real}`       : expected returns (length n)
- `Σ::AbstractMatrix{<:Real}`       : covariance matrix (n×n)
- `γ::AbstractVector{<:Real}`       : nonnegative per-asset penalties (length n)
- `k::Integer`                       : cardinality bound (1 ≤ k ≤ n)
- `stab0::AbstractVector{<:Real}`    : initial stabilization point in [0,1]^n
- `num_epochs::Integer`              : number of root passes

# Keywords
- `optimizer`  : optimizer factory/instance; if `nothing`, uses `default_optimizer()`
- `attrs`      : optimizer attributes (NamedTuple/Dict) passed to JuMP
- `epsilon`    : ridge used downstream in duals (Σeff = Σ + ε·mean(diag(Σ))·I)
- `threshold`  : threshold for supports inside `portfolios_objective` (default 0.5)
- `lambda`     : in–out mixing weight in (0,1) (default 0.10)
- `delta_scale`: stabilization jitter multiplier (default 2.0 → δ = 2*eps)

# Returns
`Vector{NamedTuple}` of cuts, each `(p, grad, status)` as in `portfolios_objective`.
"""
function kelley_primal_cuts(μ::AbstractVector{<:Real},
                            Σ::AbstractMatrix{<:Real},
                            γ::AbstractVector{<:Real},
                            k::Integer,
                            stab0::AbstractVector{<:Real},
                            num_epochs::Integer;
                            optimizer = nothing,
                            attrs = NamedTuple(),
                            epsilon::Real = EPS_RIDGE[],
                            threshold::Real = 0.5,
                            lambda::Real = 0.10,
                            delta_scale::Real = 2.0)

    n = length(μ)
    (size(Σ,1) == n && size(Σ,2) == n) || error("Σ must be $n×$n; got $(size(Σ)).")
    (length(γ) == n) || error("γ must have length n=$n.")
    all(γ .≥ 0) || error("γ must be elementwise nonnegative.")
    (1 ≤ k ≤ n) || error("k must satisfy 1 ≤ k ≤ n.")
    (length(stab0) == n) || error("stab0 must have length n=$n.")
    (0.0 < lambda < 1.0) || error("lambda must lie in (0,1).")

    # Pick optimizer (factory or instance)
    optimizer === nothing && (optimizer = default_optimizer())

    # Root model: min t  s.t.  0≤s≤1, ∑s ≤ k, ∑s ≥ 1, and cuts added iteratively.
    model = Model(optimizer)
    set_silent(model)
    for (k_attr, v_attr) in pairs(attrs)
        set_optimizer_attribute(model, k_attr, v_attr)
    end

    @variable(model, 0.0 ≤ s[1:n] ≤ 1.0)
    @variable(model, t)

    @constraint(model, sum(s) ≤ k)
    @constraint(model, sum(s) ≥ 1.0)
    @objective(model, Min, t)

    # stabilization vectors
    stab = clamp.(collect(stab0), 0.0, 1.0)
    δ = fill(delta_scale * eps(Float64), n)

    cuts = Vector{NamedTuple}(undef, 0)

    for _ in 1:num_epochs
        optimize!(model)
        zstar = clamp.(value.(s), 0.0, 1.0)

        # in–out stabilization
        @inbounds @. stab = 0.5 * (stab + zstar)
        z0 = clamp.(lambda .* zstar .+ (1 - lambda) .* stab .+ δ, 0.0, 1.0)

        # evaluate outer function & subgradient at z0
        cut = portfolios_objective(μ, Σ, γ, k, z0;
                                   optimizer = optimizer,
                                   attrs = attrs,
                                   epsilon = epsilon,
                                   threshold = threshold,
                                   ensure_one = true)

        if cut.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
            # t ≥ p + grad' * (s - z0)
            @constraint(model, t ≥ cut.p + sum(cut.grad[i] * (s[i] - z0[i]) for i in 1:n))
            push!(cuts, cut)
        else
            # safeguard: simple Hamming-distance cut to avoid cycling on z0
            @constraint(model, sum(z0[i] * (1 - s[i]) + s[i] * (1 - z0[i]) for i in 1:n) ≥ 1.0)
            push!(cuts, cut)
        end
    end

    return cuts
end

end # end module