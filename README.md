# SparseMaxSR.jl

SparseMaxSR is a Julia package for **mean–variance portfolio selection with cardinality constraints**. It focuses on selecting **at most / exactly _k_ assets out of _N_** to **maximize the Sharpe ratio** (or to compute mean‑variance efficient weights on a given support). The toolbox offers:

- **Exhaustive / Random Search** over supports (with hard caps to keep runtime bounded).
- **MIQP Heuristic Search** (via JuMP/MathOptInterface; optional refit; optional budget normalization).
- **LASSO Relaxation Search** (path‑based support discovery; vanilla normalization or refit).
- **Core Sharpe‑ratio utilities** (MVE Sharpe and weights).

It is designed for **reproducible experiments** across small, medium, and large universes.

---

## Installation

### Option A — develop locally (recommended while iterating)

```julia
julia> using Pkg
julia> Pkg.activate(".")           # your project
julia> Pkg.develop(path="/path/to/SparseMaxSR")
julia> using SparseMaxSR
```

### Option B — add from Git (if hosted)

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/<org>/SparseMaxSR.jl")
julia> using SparseMaxSR
```

> If you maintain separate solver backends (SCS/HiGHS/Mosek/CPLEX/…),
> add them to your environment as needed (see **Dependencies**).

---

## Dependencies

Core (direct):

- `LinearAlgebra`, `Statistics`, `Random`
- `Combinatorics` (for combinations)
- `JuMP` and `MathOptInterface` (MIQP heuristic)
- `GLMNet` (LASSO path; used in relaxation search)
- `CPLEX` (MIQP heuristic)

---

## Exported API (overview)

All functions return **named tuples** where multiple outputs are expected
(e.g., `selection`, `weights`, `sr`, `status`).

### SharpeRatio

```julia
compute_sr(w::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix;
           epsilon::Real=EPS_RIDGE, stabilize_Σ::Bool=true, do_checks::Bool=false) -> Real
```
Compute the portfolio Sharpe ratio of weights `w` given `(μ, Σ)`.
If `stabilize_Σ=true`, a ridge‑stabilized covariance `Σ + ε·mean(diag(Σ))·I` is used internally
(controlled by `epsilon`).

```julia
compute_mve_sr(μ::AbstractVector, Σ::AbstractMatrix;
               selection::AbstractVector{<:Integer}=Int[],
               epsilon::Real=EPS_RIDGE, stabilize_Σ::Bool=true, do_checks::Bool=false) -> Real
```
Compute the **maximum** Sharpe ratio achievable by the **mean‑variance efficient** portfolio
**restricted to a support** `selection` (if empty, uses all assets).

```julia
compute_mve_weights(μ::AbstractVector, Σ::AbstractMatrix;
                    selection::AbstractVector{<:Integer}=Int[],
                    epsilon::Real=EPS_RIDGE, stabilize_Σ::Bool=true,
                    weights_sum1::Bool=false, do_checks::Bool=false) -> Vector{Float64}
```
Return the corresponding **MVE weights** on `selection` (full‑length vector with zeros outside the support).
If `weights_sum1=true`, the result is rescaled to satisfy `|∑w|=1` (when feasible).

---

### ExhaustiveSearch

```julia
mve_exhaustive_search(μ::AbstractVector, Σ::AbstractMatrix, k::Integer;
    exactly_k::Bool=true,
    max_samples_per_k::Int=0,
    max_combinations::Int=10_000_000,
    epsilon::Real=EPS_RIDGE,
    rng::AbstractRNG=Random.default_rng(),
    γ::Real=1.0,
    stabilize_Σ::Bool=true,
    compute_weights::Bool=false,
    weights_sum1::Bool=false,
    do_checks::Bool=false
) -> NamedTuple{(:selection, :weights, :sr, :status)}
```
Searches supports to maximize the **MVE Sharpe ratio**:

- If `max_samples_per_k == 0`, the routine **tries** to enumerate all combinations at each size,
  but will **cap** work per size at `max_combinations`. If the cap binds, it switches to
  **sampling without replacement** up to the cap and returns `status = :EXHAUSTIVE_SAMPLED`.
- If `max_samples_per_k > 0`, it samples up to that many supports per size (also bounded by the number of available combinations).
- `exactly_k=true` restricts to size `k`; set `false` to search all sizes in `1:k`.
- If `compute_weights=true`, the routine returns MVE weights on the winning support; set `weights_sum1=true`
  to normalize them to budget **(|∑w|=1)**.

---

### LassoRelaxationSearch

```julia
mve_lasso_relaxation_search(μ::AbstractVector, Σ::AbstractMatrix, T::Integer;
    k::Integer,
    nlambda::Int=100,
    lambda_min_ratio::Real=1e-5,
    alpha::Real=1.0,                  # Elastic Net mixing; 1.0 = LASSO
    standardize::Bool=false,
    epsilon::Real=EPS_RIDGE,
    stabilize_Σ::Bool=true,
    compute_weights::Bool=true,
    use_refit::Bool=true,
    weights_sum1::Bool=false,
    do_checks::Bool=false
) -> NamedTuple{(:selection, :weights, :sr, :status)}
```
- Builds a **path** (via `GLMNet`) and picks **the largest support** of size `≤ k`.
  - If it hits `k` exactly: `status = :LASSO_PATH_EXACT_K`.
  - Otherwise: `status = :LASSO_PATH_ALMOST_K`.
  - If no variables are ever selected: `status = :LASSO_ALLEMPTY`.
- **Vanilla (use_refit=false)**: **normalize the path coefficients into portfolio weights**;
  set `weights_sum1=true` to enforce `|∑w|=1` (when feasible). If the selected coefficients sum ≈ 0,
  the routine returns `w=0`, `sr=0`, `status=:LASSO_ALLEMPTY`.
- **Refit (use_refit=true)**: compute **exact MVE weights on the selected support**; set `weights_sum1=true`
  to rescale the refit weights to `∑w=1`. SR is invariant to this rescaling.

---

### MIQPHeuristicSearch

```julia
mve_miqp_heuristic_search(μ::AbstractVector, Σ::AbstractMatrix;
    k::Integer,
    m::Integer=max(0, k-1),           # lower bound on cardinality (when used)
    γ::Real=1.0,                      # risk‑aversion in the MIQP objective
    fmin::AbstractVector=zeros(length(μ)),
    fmax::AbstractVector=ones(length(μ)),
    # Heuristic expansion controls (MATLAB‑like defaults):
    expand_rounds::Int=20,
    expand_factor::Float64=3.0,
    expand_tol::Float64=1e-2,
    # Solve controls:
    mipgap::Real=1e-4,
    time_limit::Real=200.0,           # seconds
    threads::Int=1,
    # Behavior:
    exactly_k::Bool=false,
    compute_weights::Bool=true,
    use_refit::Bool=false,
    weights_sum1::Bool=false,
    epsilon::Real=EPS_RIDGE,
    stabilize_Σ::Bool=true,
    do_checks::Bool=false,
    # Warm starts (optional):
    x_start::AbstractVector=nothing,  # continuous weights seed
    v_start::AbstractVector=nothing   # binary support seed
) -> NamedTuple{(:selection, :weights, :sr, :status)}
```
- A JuMP/MathOptInterface‑based heuristic for the cardinality‑constrained MVE problem.
- **Budget:** *not enforced by default* (`weights_sum1=false`). Set `weights_sum1=true`
  to rescale final weights to `|∑w|=1` (SR invariant).
- **Cardinality:** use `exactly_k=true` to enforce `|S|=k`. Otherwise, the algorithm respects the bounds `m ≤ |S| ≤ k`.
  If both are provided and inconsistent with equality, `exactly_k=true` takes precedence.
- **Refit:** `use_refit=true` recomputes **closed‑form MVE weights** and SR on the found support.
- **Bounds:** elementwise `fmin ≤ w ≤ fmax` (interpreted in the MIQP stage; the final normalization step may scale within bounds if used with `weights_sum1=true`).

---

## Example: Comparing Exhaustive, LASSO (vanilla/refit), and MIQP (vanilla/refit)

Save as `example.jl`, then run:

```bash
julia --project=. example.jl
```

```julia
#!/usr/bin/env julia

using SparseMaxSR
using Random, LinearAlgebra, Statistics, Printf, Dates
using Combinatorics: binomial
import MathOptInterface as MOI

# --------------------------------
# Helpers
# --------------------------------
timestamp() = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")

# Simulate returns with a mild 2-factor structure + noise
function simulate_returns(T::Int, N::Int; nf::Int=2, beta_scale=0.3, eps_scale=0.7,
                          rng=Random.default_rng())
    F = randn(rng, T, nf)
    B = beta_scale .* randn(rng, N, nf)
    E = eps_scale  .* randn(rng, T, N)
    return F * B' .+ E
end

means_and_cov(R) = (vec(mean(R, dims=1)), cov(R; corrected=true))
cell(sr, t) = isnan(sr) ? "-" : @sprintf("%.4f / %.2fs", sr, t)
_fmt_ks(v::Vector{Int}) = isempty(v) ? "(none)" : join(sort(unique(v)), ", ")

# --------------------------------
# Runners (set weights_sum1=true if you want ∑w=1)
# --------------------------------
function run_exhaustive(μ, Σ, k; exactly_k=true, max_combinations=3_000_000, weights_sum1=false)
    tsec = @elapsed begin
        res = SparseMaxSR.mve_exhaustive_search(μ, Σ, k;
            exactly_k=exactly_k, max_samples_per_k=0, max_combinations=max_combinations,
            epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
            compute_weights=true, weights_sum1=weights_sum1, do_checks=false)
        global sel = res.selection; global w = res.weights
        global sr = res.sr;         global st = res.status
    end
    return sel, w, sr, tsec, st
end

function run_miqp_vanilla(μ, Σ, k; weights_sum1=false)
    tsec = @elapsed begin
        res = SparseMaxSR.mve_miqp_heuristic_search(μ, Σ; k=k,
            compute_weights=true, use_refit=false, weights_sum1=weights_sum1)
        global sel = res.selection; global w = res.weights
        global sr = res.sr;         global st = res.status
    end
    return sel, w, sr, tsec, st
end

function run_miqp_refit(μ, Σ, k; weights_sum1=false)
    tsec = @elapsed begin
        res = SparseMaxSR.mve_miqp_heuristic_search(μ, Σ; k=k,
            compute_weights=true, use_refit=true, weights_sum1=weights_sum1)
        global sel = res.selection; global w = res.weights
        global sr = res.sr;         global st = res.status
    end
    return sel, w, sr, tsec, st
end

# LASSO runners (moment-based entry)
function run_lasso_vanilla(R, μ, Σ, k; alpha=0.95, weights_sum1=false)
    tsec = @elapsed begin
        res = SparseMaxSR.mve_lasso_relaxation_search(μ, Σ, size(R,1);
            k=k, nlambda=100, lambda_min_ratio=1e-5, alpha=alpha, standardize=false,
            epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
            compute_weights=true, use_refit=false, weights_sum1=weights_sum1, do_checks=false)
        global sel = res.selection; global w = res.weights
        global sr = res.sr;         global st = res.status
    end
    return sel, w, sr, tsec, st
end

function run_lasso_refit(R, μ, Σ, k; alpha=0.95, weights_sum1=false)
    tsec = @elapsed begin
        res = SparseMaxSR.mve_lasso_relaxation_search(μ, Σ, size(R,1);
            k=k, nlambda=100, lambda_min_ratio=1e-5, alpha=alpha, standardize=false,
            epsilon=SparseMaxSR.EPS_RIDGE, stabilize_Σ=true,
            compute_weights=true, use_refit=true, weights_sum1=weights_sum1, do_checks=false)
        global sel = res.selection; global w = res.weights
        global sr = res.sr;         global st = res.status
    end
    return sel, w, sr, tsec, st
end
```

> **Tip:** Set `weights_sum1=true` in the runners above if you want weights to sum to one;
> SR values will be identical either way (scale invariance).

---

## Version notes

- **Budget normalization is now optional everywhere** via `weights_sum1` (default `false`).
- **MIQP defaults** align with a MATLAB‑like heuristic setup: `time_limit=200.0`, `expand_rounds=20`, `expand_factor=3.0`, `expand_tol=1e-2`.
- **LASSO path selection** returns the largest support `≤ k`, with statuses `:LASSO_PATH_EXACT_K`, `:LASSO_PATH_ALMOST_K`, `:LASSO_ALLEMPTY`.
- `EPS_RIDGE` is a **scalar constant** used across routines for covariance stabilization.
