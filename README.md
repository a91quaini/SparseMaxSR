# SparseMaxSR.jl

**SparseMaxSR** is a Julia package for *mean–variance efficient (MVE)* portfolio selection under **cardinality constraints**.  
It provides unified, numerically robust tools to estimate **sparse maximum‑Sharpe portfolios**, i.e., portfolios composed of at most or exactly *k* assets out of *N*, while preserving the classical mean–variance structure.

---

## 1. Problem statement

We seek to solve the *sparse Sharpe‑maximization problem*:

$$
\max_{w \in \mathbb{R}^N}
\frac{w' \mu}{\sqrt{w' \Sigma w}}
\quad \text{s.t.} \quad \|w\|_0 \le k,
$$

where

- $\mu \in \mathbb{R}^N$: vector of expected excess returns,  
- $\Sigma \in \mathbb{R}^{N\times N}$: covariance matrix of returns,  
- $\|w\|_0$: number of nonzero elements in $w$ (sparsity).

The unconstrained MVE (mean–variance efficient) portfolio is

$$
w_{\text{MVE}} = \Sigma^{-1}\mu, \qquad
SR_{\text{MVE}} = \sqrt{\mu'\Sigma^{-1}\mu}.
$$

SparseMaxSR provides algorithms that approximate this solution when the support size is restricted.

---

## 2. Methods implemented

SparseMaxSR offers three complementary approaches, all sharing a unified API:

| Method | Description | Typical use |
|:--------|:-------------|:-------------|
| **Exhaustive / Random Search** | Enumerates or samples subsets of size *k* and selects the one with maximal in‑sample MVE Sharpe ratio. | Small N (≤ 30–40) or validation of heuristics. |
| **LASSO Relaxation Search** | Solves a continuous relaxation via **Elastic‑Net (GLMNet)** regression; selects the largest support ≤ *k* and optionally refits exact MVE weights. | Large N, fast approximation. |
| **MIQP Heuristic Search** | Mixed‑integer quadratic heuristic via **JuMP + CPLEX**, with optional cardinality band and progressive bound expansion. | Medium/large N; high accuracy with time control. |

All methods call the shared low‑level routines in the [`SharpeRatio`](#sharperatio-module) and [`Utils`](#utils-module) modules for stable covariance handling and Sharpe computation.

---

## 3. Installation

### Option A — local development

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.develop(path="/path/to/SparseMaxSR")
julia> using SparseMaxSR
```

### Option B — clone from Git

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/<org>/SparseMaxSR.jl")
julia> using SparseMaxSR
```

> Add solver backends (`CPLEX`, `HiGHS`, `MosekTools`, `SCS`, etc.) to your environment as required.

---

## 4. Dependencies

| Dependency | Purpose |
|:------------|:---------|
| `LinearAlgebra`, `Statistics`, `Random` | Core numerical routines |
| `Combinatorics` | Exhaustive and random subset generation |
| `GLMNet` | LASSO / Elastic‑Net path solver |
| `JuMP`, `MathOptInterface`, `CPLEX` | MIQP heuristic search |

---

## 5. Exported API

All main routines return **named tuples** `(selection, weights, sr, status)` or `(selection, sr)` depending on context.

| Field | Meaning |
|:------|:---------|
| `selection` | Vector of indices of selected assets |
| `weights` | Portfolio weights (zeros off‑support) |
| `sr` | In‑sample Sharpe ratio |
| `status` | Symbol describing solver outcome (`:OK`, `:LASSO_PATH_EXACT_K`, etc.) |

---

### 🔹 SharpeRatio module

#### `compute_sr`

```julia
compute_sr(w::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix;
           selection::AbstractVector{<:Integer}=Int[],
           epsilon::Real=EPS_RIDGE,
           stabilize_Σ::Bool=true,
           do_checks::Bool=false) -> Float64
```

Computes the Sharpe ratio of a given portfolio:

$$
SR(w) = \frac{w' \mu}{\sqrt{w' (\Sigma + \epsilon I) w}}.
$$

If `selection` is provided, only those indices contribute to the numerator and denominator.  
Returns `NaN` when variance ≤ 0 or not finite.

---

#### `compute_mve_sr`

```julia
compute_mve_sr(μ::AbstractVector, Σ::AbstractMatrix;
               selection::AbstractVector{<:Integer}=Int[],
               epsilon::Real=EPS_RIDGE,
               stabilize_Σ::Bool=true,
               do_checks::Bool=false) -> Float64
```

Computes the **maximum Sharpe ratio** of the mean–variance efficient (MVE) portfolio on a given subset:

$$
SR^*(S) = \sqrt{ \mu_S' \Sigma_S^{-1} \mu_S }.
$$

When `selection` is empty, the full‑universe MVE Sharpe ratio is returned.

---

#### `compute_mve_weights`

```julia
compute_mve_weights(μ::AbstractVector, Σ::AbstractMatrix;
                    selection::AbstractVector{<:Integer}=Int[],
                    normalize_weights::Bool=false,
                    epsilon::Real=EPS_RIDGE,
                    stabilize_Σ::Bool=true,
                    do_checks::Bool=false) -> Vector{Float64}
```

Computes MVE weights $w = \Sigma^{-1}\mu$ (restricted to the chosen support if provided).  
If `normalize_weights=true`, weights are rescaled for numerical stability.  
Normalization does **not** affect Sharpe ratios (scale‑invariant).

---

### 🔹 ExhaustiveSearch module

#### `mve_exhaustive_search`

```julia
mve_exhaustive_search(μ::AbstractVector{<:Real},
                      Σ::AbstractMatrix{<:Real};
                      k::Integer,
                      epsilon::Real = Utils.EPS_RIDGE,
                      stabilize_Σ::Bool = true,
                      do_checks::Bool = false,
                      # enumeration / sampling knobs
                      enumerate_all::Bool = true,
                      max_samples::Int = 0,
                      dedup_samples::Bool = true,
                      rng::AbstractRNG = Random.GLOBAL_RNG,
                      # outputs
                      compute_weights::Bool = true
) -> NamedTuple{(:selection, :weights, :sr, :status)}
```

Enumerates or samples subsets of size `k`, returning the best subset and its MVE Sharpe ratio.  
For feasible $\binom{N}{k}$, set `enumerate_all=true`; otherwise, provide `max_samples`.

| Argument | Description |
|:----------|:-------------|
| `μ`, `Σ` | Mean and covariance |
| `k` | Target subset size |
| `enumerate_all` | Enumerate all combinations (`true`) or sample |
| `max_samples` | Number of samples if not enumerating |
| `dedup_samples` | Ensure sampled supports are unique |
| `rng` | Random number generator |
| `epsilon`, `stabilize_Σ`, `do_checks` | Numerical options |
| `compute_weights` | Output optimal weights |

Returns `(selection, sr)`.

---

### 🔹 LassoRelaxationSearch module

#### `mve_lasso_relaxation_search`

```julia
mve_lasso_relaxation_search(R::AbstractMatrix{<:Real};
                            k::Integer,
                            y::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            nlambda::Int=100,
                            lambda_min_ratio::Real=1e-3,
                            alpha::Real=0.95,
                            standardize::Bool=false,
                            epsilon::Real=EPS_RIDGE,
                            stabilize_Σ::Bool=true,
                            compute_weights::Bool=false,
                            normalize_weights::Bool=false,
                            use_refit::Bool=true,
                            do_checks::Bool=false)
    -> NamedTuple{(:selection,:weights,:sr,:status)}
```

**Path‑based Elastic‑Net relaxation** of the sparse MVE problem:

1. Fits GLMNet path $y = R\beta + \varepsilon$.  
2. Chooses the largest support ≤ `k`.  
3. Either:  
   - `use_refit=true`: compute exact MVE SR and refit weights on the selected support;  
   - `use_refit=false`: return the raw or normalized LASSO coefficients.

**Statuses**:  
`:LASSO_PATH_EXACT_K`, `:LASSO_PATH_ALMOST_K`, `:LASSO_ALLEMPTY`.

---

### 🔹 MIQPHeuristicSearch module

#### `mve_miqp_heuristic_search`

```julia
mve_miqp_heuristic_search(μ::AbstractVector, Σ::AbstractMatrix;
                          k::Integer,
                          exactly_k::Bool=false,
                          m::Union{Int,Nothing}=nothing,
                          γ::Float64=1.0,
                          fmin::AbstractVector=zeros(length(μ)),
                          fmax::AbstractVector=ones(length(μ)),
                          expand_rounds::Int=20,
                          expand_factor::Float64=3.0,
                          expand_tol::Float64=1e-2,
                          mipgap::Float64=1e-4,
                          time_limit::Real=200,
                          threads::Int=0,
                          compute_weights::Bool=false,
                          normalize_weights::Bool=false,
                          use_refit::Bool=true,
                          verbose::Bool=false,
                          epsilon::Real=EPS_RIDGE,
                          stabilize_Σ::Bool=true,
                          do_checks::Bool=false)
```

Heuristic mixed‑integer quadratic programming (MIQP) solver for sparse MVE selection.  
The optimization problem is:

$$
\begin{aligned}
\min_{x,v} \; & \tfrac{1}{2} \gamma x' \Sigma_s x - \mu'x \\
\text{s.t.} \;& m \le \sum_i v_i \le k, \\
& v_i=0 \Rightarrow x_i=0, \\
& v_i=1 \Rightarrow f_{\min,i} \le x_i \le f_{\max,i}.
\end{aligned}
$$

If `normalize_weights=true`, adds a budget constraint $\sum_i x_i = 1$ and rescales outputs.

Key options:
- `expand_rounds`, `expand_factor`, `expand_tol`: progressive bound‑expansion heuristic  
- `mipgap`, `time_limit`: CPLEX tolerances  
- `use_refit`: recompute exact MVE SR/weights on final support  
- `normalize_weights`: toggles ∑x = 1 and post‑normalization

Returns `(selection, weights, sr, status)`.

---

## 6. Example usage

The included [`example.jl`](example.jl) script compares all methods on simulated two‑factor returns:

```bash
julia --project=. example.jl
```

Example output:

```
Results — Experiment A (T=500, N=30)
------------------------------------
k      | EXHAUSTIVE         | LASSO‑VANILLA      | LASSO‑REFIT        | MIQP‑VANILLA       | MIQP‑REFIT
-----------------------------------------------------------------------------------------------------
1      | 0.1538 / 0.00s     | 0.1538 / 0.00s     | 0.1538 / 0.00s     | 0.1538 / 4.7s      | 0.1538 / 0.2s
3      | 0.1962 / 0.07s     | 0.1760 / 0.00s     | 0.1945 / 0.00s     | 0.1930 / 3.1s      | 0.1955 / 0.6s
...
```

---

