# SparseMaxSR.jl

SparseMaxSR is a Julia package for **mean–variance portfolio selection under cardinality constraints**.
It provides a consistent and efficient framework to estimate **sparse maximum‑Sharpe portfolios**, i.e., portfolios composed of at most or exactly *k* assets out of *N*, while preserving the mean–variance efficient (MVE) structure.

Mathematically, the goal is to solve the **sparse Sharpe‑maximization problem**

$$
\max_{w\in\mathbb{R}^N} \frac{w'\mu}{\sqrt{w'\Sigma w}}
\quad \text{s.t.}\quad \|w\|_0 \le k,
$$

where  
- $\mu\in\mathbb{R}^N$ is the vector of expected excess returns,  
- $\Sigma\in\mathbb{R}^{N\times N}$ is the covariance matrix of returns, and  
- $\|w\|_0$ counts the number of nonzero elements in $w$.

The package implements several **complementary search strategies**:

1. **Exhaustive / Random Search** — exact or stochastic enumeration of supports of size *k* (feasible for small *N*).  
2. **LASSO Relaxation Search** — continuous relaxation via LASSO/Elastic‑Net regression, followed by optional MVE refit.  
3. **MIQP Heuristic Search** — mixed‑integer quadratic programming heuristic via JuMP/CPLEX, supporting cardinality bounds and warm starts.  

All methods share a unified interface and numerical backend (`SharpeRatio` module).  
They can be used for empirical comparisons, replication studies, and large‑scale simulation of high‑dimensional portfolio selection.

---

## Installation

### Option A — develop locally (recommended)

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.develop(path="/path/to/SparseMaxSR")
julia> using SparseMaxSR
```

### Option B — add from Git

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/<org>/SparseMaxSR.jl")
julia> using SparseMaxSR
```

> Add solver back‑ends (e.g. `CPLEX`, `HiGHS`, `MosekTools`, `SCS`) to your environment as needed.

---

## Dependencies

Core modules:

- `LinearAlgebra`, `Statistics`, `Random` — numerical primitives  
- `Combinatorics` — support enumeration  
- `JuMP`, `MathOptInterface` — MIQP formulation  
- `GLMNet` — LASSO / Elastic‑Net relaxation  
- `CPLEX` — MIQP heuristic backend

---

## Exported API (overview)

All high‑level functions return **named tuples** `(selection, weights, sr, status)`:

| Field | Meaning |
|:------|:--------|
| `selection` | Indices of selected assets (support set). |
| `weights` | Portfolio weights (full vector, zeros outside support). |
| `sr` | Sharpe ratio achieved by `weights`. |
| `status` | Symbolic solver status (`:OK`, `:LASSO_PATH_EXACT_K`, `:EXHAUSTIVE_SAMPLED`, etc.). |

---

### `compute_sr`

```julia
compute_sr(w::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix;
           epsilon::Real=EPS_RIDGE,
           stabilize_Σ::Bool=true,
           do_checks::Bool=false) -> Real
```

Compute the Sharpe ratio of a given weight vector `w`.

**Formula**

$$
SR(w) = \frac{w'\mu}{\sqrt{w'(\Sigma + \epsilon I)w}},
$$

where a ridge term $\epsilon I$ is added if `stabilize_Σ=true`.

| Argument | Type | Default | Description |
|:----------|:------|:---------|:-------------|
| `w` | Vector | — | Portfolio weights |
| `μ` | Vector | — | Expected returns |
| `Σ` | Matrix | — | Covariance matrix |
| `epsilon` | Real | `EPS_RIDGE` | Ridge regularization magnitude |
| `stabilize_Σ` | Bool | `true` | Whether to apply ridge stabilization |
| `do_checks` | Bool | `false` | Validate dimensions and finiteness |

---

### `compute_mve_sr`

```julia
compute_mve_sr(μ::AbstractVector, Σ::AbstractMatrix;
               selection::AbstractVector{<:Integer}=Int[],
               epsilon::Real=EPS_RIDGE,
               stabilize_Σ::Bool=true,
               do_checks::Bool=false) -> Real
```

Computes the **maximum Sharpe ratio** achievable by the **mean‑variance efficient (MVE)** portfolio restricted to a subset of assets.

If `selection` is empty, all assets are used.

| Argument | Type | Default | Description |
|:----------|:------|:---------|:-------------|
| `μ`, `Σ` | see above | — | Mean and covariance |
| `selection` | Vector{Int} | `Int[]` | Indices of active assets |
| `epsilon`, `stabilize_Σ`, `do_checks` | — | — | Same as above |

---

### `compute_mve_weights`

```julia
compute_mve_weights(μ::AbstractVector, Σ::AbstractMatrix;
                    selection::AbstractVector{<:Integer}=Int[],
                    epsilon::Real=EPS_RIDGE,
                    stabilize_Σ::Bool=true,
                    weights_sum1::Bool=false,
                    do_checks::Bool=false) -> Vector{Float64}
```

Returns the MVE weights corresponding to the subset `selection` (zeros elsewhere).  
If `weights_sum1=true`, weights are rescaled using `Utils.normalize_weights` so that $|\sum_i w_i|≈1$.

---

### `mve_exhaustive_search`

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
    do_checks::Bool=false)
```

Performs exhaustive or sampled search over all supports up to size `k`, selecting the one that maximizes `compute_mve_sr`.

| Argument | Type | Default | Description |
|:----------|:------|:---------|:-------------|
| `μ`, `Σ`, `k` | — | — | Inputs and target sparsity |
| `exactly_k` | Bool | `true` | Restrict to supports of size `k` |
| `max_samples_per_k` | Int | `0` | Number of random supports per size (`0` → full enumeration) |
| `max_combinations` | Int | `10_000_000` | Safety cap on total combinations |
| `compute_weights` | Bool | `false` | Whether to return MVE weights |
| `weights_sum1` | Bool | `false` | Normalize weights (|∑w|≈1) |
| `status` | Symbol | — | Indicates whether full enumeration or sampling was used |

---

### `mve_lasso_relaxation_search`

```julia
mve_lasso_relaxation_search(μ::AbstractVector, Σ::AbstractMatrix, T::Integer;
    k::Integer,
    nlambda::Int=100,
    lambda_min_ratio::Real=1e-3,
    alpha::Real=0.95,
    standardize::Bool=false,
    epsilon::Real=EPS_RIDGE,
    stabilize_Σ::Bool=true,
    compute_weights::Bool=true,
    use_refit::Bool=true,
    weights_sum1::Bool=false,
    do_checks::Bool=false)
```

Applies a **LASSO / Elastic‑Net relaxation** of the MVE problem, using GLMNet to trace the regularization path.  
It picks the largest support of size ≤ `k` and optionally recomputes exact MVE weights.

| Argument | Type | Default | Description |
|:----------|:------|:---------|:-------------|
| `T` | Int | — | Number of observations (used for normalization) |
| `k` | Int | — | Desired maximum support size |
| `nlambda` | Int | `100` | Number of λ grid points |
| `lambda_min_ratio` | Real | `1e-3` | Smallest λ relative to λ_max |
| `alpha` | Real | `0.95` | Elastic‑Net mixing (1.0 = LASSO) |
| `standardize` | Bool | `false` | Standardize regressors |
| `use_refit` | Bool | `true` | Recompute MVE weights on selected support |
| `weights_sum1` | Bool | `false` | Normalize final weights |
| `compute_weights` | Bool | `true` | Return portfolio weights (vs support only) |

**Statuses**:  
`:LASSO_PATH_EXACT_K`, `:LASSO_PATH_ALMOST_K`, `:LASSO_ALLEMPTY`.

---

### `mve_miqp_heuristic_search`

```julia
mve_miqp_heuristic_search(μ::AbstractVector, Σ::AbstractMatrix;
    k::Integer,
    m::Integer=max(0,k-1),
    γ::Real=1.0,
    fmin::AbstractVector=zeros(length(μ)),
    fmax::AbstractVector=ones(length(μ)),
    expand_rounds::Int=20,
    expand_factor::Float64=3.0,
    expand_tol::Float64=1e-2,
    mipgap::Real=1e-4,
    time_limit::Real=200.0,
    threads::Int=1,
    exactly_k::Bool=false,
    compute_weights::Bool=true,
    use_refit::Bool=false,
    weights_sum1::Bool=false,
    epsilon::Real=EPS_RIDGE,
    stabilize_Σ::Bool=true,
    do_checks::Bool=false,
    x_start::AbstractVector=nothing,
    v_start::AbstractVector=nothing)
```

A JuMP/CPLEX‑based heuristic that approximates the cardinality‑constrained MVE solution via MIQP.  
The algorithm alternates between mixed‑integer search and local expansion rounds to escape local minima.

| Argument | Type | Default | Description |
|:----------|:------|:---------|:-------------|
| `k` | Int | — | Upper cardinality bound |
| `m` | Int | `max(0,k-1)` | Lower cardinality bound |
| `γ` | Real | `1.0` | Risk‑aversion coefficient in objective |
| `expand_rounds` | Int | `20` | Number of heuristic expansion passes |
| `expand_factor` | Float | `3.0` | Search radius multiplier |
| `expand_tol` | Float | `1e-2` | Expansion tolerance |
| `mipgap` | Real | `1e-4` | MILP relative optimality gap |
| `time_limit` | Real | `200.0` | Solver time limit (seconds) |
| `threads` | Int | `1` | Parallel threads |
| `exactly_k` | Bool | `false` | Enforce |S|=k |
| `use_refit` | Bool | `false` | Refit exact MVE weights |
| `weights_sum1` | Bool | `false` | Normalize |∑w|≈1 |
| `x_start`, `v_start` | Vector | `nothing` | Warm starts for continuous and binary variables |

---

## Example

The `example.jl` script compares all methods (Exhaustive, LASSO, MIQP) on simulated two‑factor returns.
Run:

```bash
julia --project=. example.jl
```

It prints Sharpe ratios and timing for each k grid.

---
