# SparseMaxSR.jl

SparseMaxSR is a Julia package for **mean–variance portfolio selection with cardinality constraints**. It focuses on selecting **at most/ exactly _k_ assets out of _N_** to **maximize the Sharpe ratio** (or compute the mean-variance efficient weights on a given support). The toolbox currently offers:

- **Exhaustive/Random Search** over supports (with hard caps to keep runtime bounded).
- **MIQP Heuristic Search** (via JuMP/MathOptInterface; supports optional refit).
- **LASSO Relaxation Search** (path-based support discovery; vanilla normalization or refit).
- **Core Sharpe-ratio utilities** (MVE Sharpe and weights).

It is designed for **reproducible experiments** across small, medium, and very large universes.

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

> If you maintain separate solver backends (SCS/HiGHS/Mosek), add them to your environment as needed (see **Dependencies**).

---

## Dependencies

Core (direct):

- `LinearAlgebra`, `Statistics`, `Random`
- `Combinatorics` (for combinations)
- `JuMP` and `MathOptInterface` (MIQP heuristic)
- `GLMNet` (LASSO path; used in relaxation search)

Suggested / optional:

- One or more conic/MIP solvers (depending on your MIQP heuristic):  
  `HiGHS`, `CPLEX`, `MosekTools`, `Gurobi`, `SCS`, `Clarabel`, `COSMO`, etc.

---

## Exported API (overview)

Below are the **main user-facing functions** you’ll typically use in experiments. All functions return **named tuples** whenever multiple outputs are expected (selection, weights, SR, status).

### SharpeRatio

```julia
compute_sr(w::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix;
           epsilon::Real=EPS_RIDGE, stabilize_Σ::Bool=true, do_checks::Bool=false) -> Real
```
Compute the portfolio Sharpe ratio of weights `w` given `(μ, Σ)`. If `stabilize_Σ=true`, a ridge-stabilized covariance `Σ + ε·mean(diag(Σ))·I` is used internally (controlled by `epsilon`).

```julia
compute_mve_sr(μ::AbstractVector, Σ::AbstractMatrix;
               selection::AbstractVector{<:Integer}=Int[],
               epsilon::Real=EPS_RIDGE, stabilize_Σ::Bool=true, do_checks::Bool=false) -> Real
```
Compute the **maximum** Sharpe ratio achievable by the **mean-variance efficient** portfolio **restricted to a support** `selection` (if empty, uses all assets).

```julia
compute_mve_weights(μ::AbstractVector, Σ::AbstractMatrix;
                    selection::AbstractVector{<:Integer}=Int[],
                    epsilon::Real=EPS_RIDGE, stabilize_Σ::Bool=true, do_checks::Bool=false) -> Vector{Float64}
```
Return the corresponding **MVE weights** on `selection` (full-length vector with zeros outside the support).

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
    do_checks::Bool=false
) -> NamedTuple{(:selection, :weights, :sr, :status)}
```
Searches supports to maximize the **MVE Sharpe ratio**:

- If `max_samples_per_k == 0`, it **tries** to enumerate all combinations at each size, but will **cap the total work** per size at `max_combinations`; when the cap binds, the routine switches to **sampling without replacement** up to the cap.
- If `max_samples_per_k > 0`, it samples up to that many supports per size (also bounded by the number of available combinations).
- `exactly_k=true` restricts to size `k`; set `false` to search all sizes in `1:k`.
- Returns `status = :EXHAUSTIVE` if full enumeration happened for all tried sizes, otherwise `:EXHAUSTIVE_SAMPLED`.

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
    do_checks::Bool=false
) -> NamedTuple{(:selection, :weights, :sr, :status)}
```
- Builds a **path** (via `GLMNet`) and picks **the largest support** of size `≤ k`.  
  - If it hits `k` exactly: `status = :LASSO_PATH_EXACT_K`.  
  - Otherwise: `status = :LASSO_PATH_ALMOST_K`.  
  - If no variables selected along the path: `status = :LASSO_ALLEMPTY`.
- **Vanilla**: set `use_refit=false` to **normalize the coefficients** into portfolio weights (sum-to-one if possible).
- **Refit**: set `use_refit=true` to compute **exact MVE weights on the selected support**.

---

### MIQPHeuristicSearch

```julia
mve_miqp_heuristic_search(μ::AbstractVector, Σ::AbstractMatrix;
    k::Integer,
    compute_weights::Bool=true,
    use_refit::Bool=false,
    epsilon::Real=EPS_RIDGE,
    stabilize_Σ::Bool=true,
    do_checks::Bool=false,
    # ... plus any solver routing / optimizer settings you expose
) -> NamedTuple{(:selection, :weights, :sr, :status)}
```
- A JuMP/MathOptInterface-based heuristic for the cardinality-constrained MVE problem.
- `use_refit=true` recomputes exact MVE weights on the found support.
- `status` typically reflects the solver’s termination status (e.g., `MOI.OPTIMAL`) or a symbolic flag from the heuristic.

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
# Runners
# --------------------------------
function run_exhaustive(μ, Σ, k; exactly_k=true, max_combinations=3_000_000)
    tsec = @elapsed begin
        res = SparseMaxSR.mve_exhaustive_search(μ, Σ, k;
            exactly_k=exactly_k, max_samples_per_k=0, max_combinations=max_combinations,
            epsilon=SparseMaxSR.Utils.EPS_RIDGE, stabilize_Σ=true,
            compute_weights=true, do_checks=false)
        global sel = res.mve_selection; global w = res.mve_weights
        global sr = res.mve_sr;         global st = res.status
    end
    return sel, w, sr, tsec, st
end

function run_miqp_vanilla(μ, Σ, k)
    tsec = @elapsed begin
        res = SparseMaxSR.mve_miqp_heuristic_search(μ, Σ; k=k,
            compute_weights=true, use_refit=false)
        global sel = res.mve_selection; global w = res.mve_weights
        global sr = res.mve_sr;         global st = res.status
    end
    return sel, w, sr, tsec, st
end

function run_miqp_refit(μ, Σ, k)
    tsec = @elapsed begin
        res = SparseMaxSR.mve_miqp_heuristic_search(μ, Σ; k=k,
            compute_weights=true, use_refit=true)
        global sel = res.mve_selection; global w = res.mve_weights
        global sr = res.mve_sr;         global st = res.status
    end
    return sel, w, sr, tsec, st
end

# LASSO runners (moment-based entry)
function run_lasso_vanilla(R, μ, Σ, k; alpha=0.95)
    tsec = @elapsed begin
        res = SparseMaxSR.mve_lasso_relaxation_search(μ, Σ, size(R,1);
            k=k, nlambda=100, lambda_min_ratio=1e-5, alpha=alpha, standardize=false,
            epsilon=SparseMaxSR.Utils.EPS_RIDGE, stabilize_Σ=true,
            compute_weights=true, use_refit=false, do_checks=false)
        global sel = res.mve_selection; global w = res.mve_weights
        global sr = res.mve_sr;         global st = res.status
    end
    return sel, w, sr, tsec, st
end

function run_lasso_refit(R, μ, Σ, k; alpha=0.95)
    tsec = @elapsed begin
        res = SparseMaxSR.mve_lasso_relaxation_search(μ, Σ, size(R,1);
            k=k, nlambda=100, lambda_min_ratio=1e-5, alpha=alpha, standardize=false,
            epsilon=SparseMaxSR.Utils.EPS_RIDGE, stabilize_Σ=true,
            compute_weights=true, use_refit=true, do_checks=false)
        global sel = res.mve_selection; global w = res.mve_weights
        global sr = res.mve_sr;         global st = res.status
    end
    return sel, w, sr, tsec, st
end

# Pretty table
function print_table(title::AbstractString, ks::Vector{Int},
                     methods::Vector{String}, cells::Dict{Tuple{Int,String},String})
    println("\n$title")
    println("-"^max(10, length(title)))
    @printf("%-6s", "k")
    for m in methods
        @printf(" | %-18s", m)
    end
    println()
    println("-"^(6 + length(methods)*(3+18)))
    for k in ks
        @printf("%-6d", k)
        for m in methods
            c = get(cells, (k,m), "-")
            @printf(" | %-18s", c)
        end
        println()
    end
    println()
end

# --------------------------------
# Experiment A: T=500, N=30, k in 1,3,5,7,9
# --------------------------------
Random.seed!(42)
T, N = 500, 30
ks = [1,3,5,7,9]
methods = ["EXHAUSTIVE", "LASSO-VANILLA", "LASSO-REFIT", "MIQP-VANILLA", "MIQP-REFIT"]

cells = Dict{Tuple{Int,String},String}()
lasso_almost = Int[]; lasso_empty = Int[]; miqp_notopt = Int[]

println("SparseMaxSR example — $(timestamp())")
println("Experiment A: T=$T, N=$N; methods=$(join(methods, \", \"))")
R = simulate_returns(T, N)
μ, Σ = means_and_cov(R)

EXH_CAP = 3_000_000  # cap combinations for exhaustive

for k in ks
    # Exhaustive (guarded by cap)
    if binomial(N, k) <= EXH_CAP
        try
            _, _, sr, t, st = run_exhaustive(μ, Σ, k; exactly_k=true, max_combinations=EXH_CAP)
            cells[(k,"EXHAUSTIVE")] = cell(sr, t) * (st == :EXHAUSTIVE ? "" : " *")
        catch
            cells[(k,"EXHAUSTIVE")] = "ERR"
        end
    else
        cells[(k,"EXHAUSTIVE")] = "SKIP"
    end

    # LASSO
    try
        _, _, sr, t, st = run_lasso_vanilla(R, μ, Σ, k; alpha=0.99)
        cells[(k,"LASSO-VANILLA")] = st == :LASSO_ALLEMPTY ?
            @sprintf("%-18s", "EMPTY / $(round(t; digits=2))s") : cell(sr, t)
        st == :LASSO_PATH_ALMOST_K  && push!(lasso_almost, k)
        st == :LASSO_ALLEMPTY       && push!(lasso_empty,  k)
    catch
        cells[(k,"LASSO-VANILLA")] = "ERR"
    end
    try
        _, _, sr, t, st = run_lasso_refit(R, μ, Σ, k; alpha=0.99)
        cells[(k,"LASSO-REFIT")] = cell(sr, t)
        st == :LASSO_PATH_ALMOST_K && push!(lasso_almost, k)
    catch
        cells[(k,"LASSO-REFIT")] = "ERR"
    end

    # MIQP
    try
        _, _, sr, t, st = run_miqp_vanilla(μ, Σ, k)
        cells[(k,"MIQP-VANILLA")] = cell(sr, t)
        st != MOI.OPTIMAL && push!(miqp_notopt, k)
    catch
        cells[(k,"MIQP-VANILLA")] = "ERR"
    end
    try
        _, _, sr, t, st = run_miqp_refit(μ, Σ, k)
        cells[(k,"MIQP-REFIT")] = cell(sr, t)
        st != MOI.OPTIMAL && push!(miqp_notopt, k)
    catch
        cells[(k,"MIQP-REFIT")] = "ERR"
    end
end

print_table("Results — Experiment A (T=500, N=30)", ks, methods, cells)
println("LASSO (both): support size < k for k ∈ {" * _fmt_ks(lasso_almost) * "}")
println("LASSO-VANILLA: ALLEMPTY for k ∈ {" * _fmt_ks(lasso_empty) * "}")
println("MIQP: solver not OPTIMAL for k ∈ {" * _fmt_ks(miqp_notopt) * "}")
println("\nNote: an asterisk '*' next to EXHAUSTIVE means the run hit the max_combinations cap and fell back to sampling.")
```
