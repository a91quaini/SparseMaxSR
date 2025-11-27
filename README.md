# SparseMaxSR.jl

**SparseMaxSR** is a Julia package for *mean–variance efficient (MVE)* portfolio selection under **cardinality constraints**.  
It provides unified, numerically robust tools to estimate **sparse maximum‑Sharpe portfolios**, i.e., portfolios composed of at most or exactly \(k\) assets out of \(N\), while preserving the classical mean–variance structure.

---

## 1. Problem statement

We seek to solve the *sparse Sharpe-maximization problem*:

$$
\max_{w \in \mathbb{R}^N}\;
\frac{w^\top \mu}{\sqrt{w^\top \Sigma\, w}}
\quad \text{s.t.} \quad \lVert w \rVert_0 \le k,
$$

where

- \(\mu \in \mathbb{R}^N\): vector of expected (excess) returns,  
- \(\Sigma \in \mathbb{R}^{N\times N}\): covariance matrix of returns,  
- \(\lVert w \rVert_0\): number of nonzeros in \(w\) (sparsity).

The unconstrained MVE (mean–variance efficient) solution is

$$
w_{\text{MVE}} = \Sigma^{-1}\mu, 
\qquad
SR_{\text{MVE}} = \sqrt{\mu^\top \Sigma^{-1}\mu}\, .
$$

SparseMaxSR provides algorithms that approximate this solution when the support size is restricted.

---

## 2. Methods implemented

SparseMaxSR offers three complementary approaches, all sharing a unified API and robust numerics:

| Method | Description | Typical use |
|:--|:--|:--|
| **Exhaustive / Random Search** | Enumerates or samples \(k\)-subsets and picks the one with maximal in-sample **MVE Sharpe**. | Small \(N\) (≤ 30–40) or ground-truth validation. |
| **LASSO Relaxation Search** | Continuous relaxation via **Elastic‑Net (GLMNet)**; pick the largest support \(\le k\), optionally **refit** exact MVE on that support; supports fixed \(\alpha\), OOS‑CV over an \(\alpha\)-grid, and **GCV** selection. | Large \(N\), fast approximation / screening. |
| **MIQP Heuristic Search** | Mixed‑integer quadratic model via **JuMP + CPLEX** with cardinality‑band or exact‑\(k\) and progressive bound expansion; optional **refit**. | Medium/large \(N\); high‑accuracy under time controls. |

All methods rely on the shared [`SharpeRatio`](#sharperatio-module) and [`Utils`](#utils-module) modules for stable covariance handling and Sharpe computation.

---

## 3. Installation

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
julia> using SparseMaxSR
```

Or via the github repo:
```julia
julia> Pkg.add(url="https://github.com/a91quaini/SparseMaxSR.jl")
```


> Add solver backends you plan to use (e.g., `CPLEX`) to your environment.

---

## 4. Exported API (overview)

All main routines return either a **named tuple** `(selection, weights, sr, status)` (MIQP/LASSO) or a pair `(selection, sr)` (Exhaustive), as detailed below.

| Field | Meaning |
|:--|:--|
| `selection` | Sorted indices of selected assets |
| `weights` | Full-length weight vector (zeros off‑support; present when `compute_weights=true` or by design) |
| `sr` | In‑sample Sharpe ratio of the returned portfolio or of the MVE refit |
| `status` | Symbol describing the outcome (e.g., `:OK`, `:LASSO_PATH_EXACT_K`, MOI termination status, etc.) |

---

## 5. Module & Function Reference

Each function uses a uniform structure: **Name**, **Description** (with formulas / pseudo‑algorithm), **Arguments**, **Returns**.

### SharpeRatio module

#### `compute_sr`

**Description.** Sharpe ratio of a given portfolio \(w\):

$$
SR(w) = \frac{w^\top \mu}{\sqrt{w^\top \Sigma_s\, w}}, 
\qquad 
\Sigma_s := \mathrm{Sym}\!\Big(\tfrac{\Sigma + \Sigma^\top}{2} + \epsilon\,\bar d\, I\Big),
$$

where \(\bar d = \tfrac{1}{N}\operatorname{tr}(\Sigma)\) and the stabilizing ridge is controlled by `epsilon` (if `stabilize_Σ=true`). Optional `selection` restricts both numerator and denominator to a subset.

**Signature.**
```julia
compute_sr(w::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix;
           selection::AbstractVector{<:Integer}=Int[],
           epsilon::Real=Utils.EPS_RIDGE,
           stabilize_Σ::Bool=true,
           do_checks::Bool=false) -> Float64
```

**Arguments.**
- `w`, `μ`, `Σ`: portfolio weights, mean vector, covariance matrix.
- `selection`: subset indices; if empty, use full universe.
- `epsilon`: ridge size for stabilization.
- `stabilize_Σ`: whether to symmetrize and ridge‑stabilize `Σ` internally.
- `do_checks`: validate shapes/finiteness.

**Returns.** `Float64` Sharpe; returns `NaN` if variance ≤ 0 or not finite.

---

#### `compute_mve_sr`

**Description.** Maximum Sharpe (MVE) on a given subset \(S\):

$$
SR^\star(S) = \sqrt{\mu_S^\top \Sigma_S^{-1}\mu_S}\, .
$$

Internally uses the stabilized/symmetrized covariance \(\Sigma_s\) once per call.

**Signature.**
```julia
compute_mve_sr(μ::AbstractVector, Σ::AbstractMatrix;
               selection::AbstractVector{<:Integer}=Int[],
               epsilon::Real=Utils.EPS_RIDGE,
               stabilize_Σ::Bool=true,
               do_checks::Bool=false) -> Float64
```

**Arguments.** As above; `selection` restricts the universe to \(S\).

**Returns.** `Float64` \(SR^\star\).

---

#### `compute_mve_weights`

**Description.** MVE weights \(w=\Sigma^{-1}\mu\) (restricted to `selection` if provided). Optional normalization rescales the vector for numerical safety; this does **not** change Sharpe ratios (scale‑invariant).

**Signature.**
```julia
compute_mve_weights(μ::AbstractVector, Σ::AbstractMatrix;
                    selection::AbstractVector{<:Integer}=Int[],
                    normalize_weights::Bool=false,
                    epsilon::Real=Utils.EPS_RIDGE,
                    stabilize_Σ::Bool=true,
                    do_checks::Bool=false) -> Vector{Float64}
```

**Arguments.** As above; set `normalize_weights=true` to apply `Utils.normalize_weights` (relative‑L1 safeguard).

**Returns.** `Vector{Float64}` full‑length weights (zeros off‑support if `selection` provided).

---

### ExhaustiveSearch module

#### `mve_exhaustive_search`

**Description.** Best‑subset search for MVE Sharpe at fixed cardinality \(k\).  
Two modes: (i) **Enumeration** of all \(\binom{N}{k}\) supports; (ii) **Sampling** of `max_samples` supports (without replacement within a support), optionally deduplicated.

Scoring uses a single stabilized covariance \(\Sigma_s\) and

$$
SR^\star(S) = \sqrt{\mu_S^\top \Sigma_{s,S}^{-1}\mu_S}\, .
$$

**Signature.**
```julia
mve_exhaustive_search(μ::AbstractVector{<:Real},
                      Σ::AbstractMatrix{<:Real};
                      k::Integer,
                      epsilon::Real = Utils.EPS_RIDGE,
                      stabilize_Σ::Bool = true,
                      do_checks::Bool = false,
                      enumerate_all::Bool = true,
                      max_samples::Int = 0,
                      dedup_samples::Bool = true,
                      rng::AbstractRNG = Random.GLOBAL_RNG
) :: Tuple{Vector{Int}, Float64}
```

**Arguments.**
- `μ`, `Σ`: asset moments.
- `k`: subset size (1 ≤ k ≤ N).
- `enumerate_all`: `true` → enumerate; `false` → sample.
- `max_samples`: number of sampled supports if `enumerate_all=false`.
- `dedup_samples`: ensure distinct supports when sampling.
- `rng`: RNG for sampling.
- `epsilon`, `stabilize_Σ`, `do_checks`: numerical/validation options.

**Returns.**
- `(selection::Vector{Int}, sr::Float64)` — best support and its in‑sample MVE Sharpe.

> For grid evaluations across many `k`, a lightweight wrapper can be used.

---

### LassoRelaxationSearch module

The LASSO/Elastic‑Net relaxation is built on a regression path (no intercept) and a **support‑size rule**: choose the column on the path whose support size is the **largest ≤ k** (closest from below). Two entry points exist.

Mathematically, for given \((\alpha,\lambda)\), the GLMNet path solves

$$
\min_{\beta \in \mathbb{R}^N} \;
\frac{1}{2T}\,\lVert y - X\beta \rVert_2^2
\;+\;
\lambda\!\left(\frac{1-\alpha}{2}\lVert\beta\rVert_2^2 + \alpha \lVert\beta\rVert_1\right),
\qquad \alpha \in [0,1].
$$

Selection is by the **largest support ≤ k** along the path, then either:
- **Refit** exact MVE on that support (recommended), or
- Return the **vanilla** LASSO portfolio built from the coefficients (optionally normalized).

Both entry points support three ways to specify \(\alpha\):
- `alpha_select = :fixed` (default): scalar \(\alpha\). If `alpha` is a vector and `:fixed`, an **OOS cross‑validation** over the grid is performed (forward‑rolling folds).
- `alpha_select = :oos_cv`: explicit OOS‑CV over an `alpha`‑grid (requires `R` in the moment‑based API).
- `alpha_select = :gcv`: **generalized cross‑validation** over an `alpha`‑grid (no folds). For each \(\alpha\), pick \(\lambda\) by the strict target‑\(k\) rule; compute log‑GCV using a ridge‑only degrees of freedom on the selected set:

$$
\mathrm{df}(\alpha,\lambda;A) \;=\; \operatorname{tr}\!\big( S_A \, (S_A + \lambda_2 I)^{-1} \big),\quad
S_A = X_A^\top X_A,\ \lambda_2 = \lambda(1-\alpha).
$$

The log‑GCV criterion is

$$
\log\mathrm{GCV} \;=\; \log(\mathrm{RSS}) - 2\log\!\Big(1-\tfrac{\mathrm{df}}{\kappa T}\Big) - \log T,
$$

with stability parameter \(\kappa\) (argument `gcv_kappa`). The \(\alpha\) with the **smallest** log‑GCV is selected. If no \(\lambda\) attains \(|A|\le k\) for any \(\alpha\), the function returns zeros with status `:LASSO_GCV_INFEASIBLE`.

#### `mve_lasso_relaxation_search` — R‑based

**Description.** Builds the path directly on raw returns \(R\) and optional response \(y\) (default: a vector of ones). Supports fixed \(\alpha\), OOS‑CV on an \(\alpha\)-grid, and **GCV** on an \(\alpha\)-grid. Final output is either **refit** MVE on the selected support or **vanilla** LASSO weights from the chosen column.

**Signature.**
```julia
mve_lasso_relaxation_search(R::AbstractMatrix{<:Real};
    k::Integer,
    y::Union{Nothing,AbstractVector{<:Real}} = nothing,
    nlambda::Int = 100,
    lambda_min_ratio::Real = 1e-3,
    lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
    alpha::Union{Real,AbstractVector{<:Real}} = 0.95,
    standardize::Bool = false,
    epsilon::Real = Utils.EPS_RIDGE,
    stabilize_Σ::Bool = true,
    compute_weights::Bool = false,
    normalize_weights::Bool = false,
    use_refit::Bool = true,
    do_checks::Bool = false,
    cv_folds::Int = 5,
    cv_verbose::Bool = false,
    alpha_select::Symbol = :fixed,   # :fixed | :oos_cv | :gcv
    gcv_kappa::Real = 1.0
) :: NamedTuple{(:selection, :weights, :sr, :status, :alpha)}
```

**Arguments.**
- `R` (`T×N`): raw returns matrix; `y` (`T`): optional response (defaults to ones).
- `k`: target support size.
- `alpha`: scalar (fixed) or vector grid; `alpha_select`: `:fixed`, `:oos_cv`, or `:gcv`.
- `lambda`, `nlambda`, `lambda_min_ratio`: path controls (passed to GLMNet).
- `standardize`: pass `true` to GLMNet standardization if desired.
- `epsilon`, `stabilize_Σ`: stabilization used when refitting or scoring.
- `compute_weights`: if `true`, return full‑length weights (zeros off‑support).
- `normalize_weights`: normalize vanilla weights (or refit weights) for numerical stability.
- `use_refit`: if `true`, refit exact MVE on the selected support; else return vanilla LASSO portfolio.
- `cv_folds`, `cv_verbose`: OOS‑CV controls (forward‑rolling folds) for `:fixed` with vector `alpha` and for `:oos_cv` mode.
- `gcv_kappa`: \(\kappa\) in the log‑GCV formula.
- `do_checks`: validate inputs and basic feasibility.

**Returns.**
- `selection::Vector{Int}` — chosen support (largest \(\le k\)).
- `weights::Vector{Float64}` — refit MVE weights (if `use_refit && compute_weights`) or vanilla LASSO weights; zeros if empty support.
- `sr::Float64` — Sharpe of refit MVE (refit) or vanilla portfolio (vanilla).
- `status::Symbol` — e.g., `:LASSO_PATH_EXACT_K`, `:LASSO_PATH_ALMOST_K`, `:LASSO_ALLEMPTY`, `:LASSO_ALPHA_CV`, `:LASSO_ALPHA_GCV`, `:LASSO_GCV_INFEASIBLE`.
- `alpha::Float64` — final \(\alpha\) used.

---

#### `mve_lasso_relaxation_search` — moment‑based

**Description.** Constructs a **synthetic design** \((X,y)\) from \((\mu,\Sigma,T)\) and proceeds as above. Supports fixed \(\alpha\), **OOS‑CV** over an \(\alpha\)-grid **(when `R` is provided)**, and **GCV** over an \(\alpha\)-grid **without needing `R`** (GCV runs on the synthetic design).  
Synthetic design (with stabilized \(\Sigma_s\)) is:

$$
Q = T(\Sigma_s + \mu\mu^\top), \quad U^\top U = Q,\quad X = U^\top,\quad y = U \backslash (T\mu).
$$

**Signature.**
```julia
mve_lasso_relaxation_search(μ::AbstractVector{<:Real},
                            Σ::AbstractMatrix{<:Real},
                            T::Integer;
    R::Union{Nothing,AbstractMatrix{<:Real}} = nothing,   # enables OOS α-CV
    k::Integer,
    nlambda::Int = 100,
    lambda_min_ratio::Real = 1e-3,
    lambda::Union{Nothing,AbstractVector{<:Real}} = nothing,
    alpha::Union{Real,AbstractVector{<:Real}} = 0.95,
    standardize::Bool = false,
    epsilon::Real = Utils.EPS_RIDGE,
    stabilize_Σ::Bool = true,
    compute_weights::Bool = false,
    normalize_weights::Bool = false,
    use_refit::Bool = true,
    do_checks::Bool = false,
    cv_folds::Int = 5,
    cv_verbose::Bool = false,
    alpha_select::Symbol = :fixed,   # :fixed | :oos_cv | :gcv
    gcv_kappa::Real = 1.0
) :: NamedTuple{(:selection, :weights, :sr, :status, :alpha)}
```

**Arguments.**
- `μ`, `Σ`, `T`: moments and (effective) sample size for the synthetic design.
- `R` (optional): raw returns to enable **OOS α‑CV** (`:fixed` with vector `alpha`, or `:oos_cv` mode). **Not required for `:gcv`**.
- Remaining arguments/semantics as in the R‑based entry point (including `alpha_select` and `gcv_kappa`).

**Returns.** Same fields as the R‑based entry point, including the selected `alpha`.

---

### MIQPHeuristicSearch module

#### `mve_miqp_heuristic_search`

**Description.** Heuristic MIQP for sparse MVE **selection** with box bounds and a cardinality **band** or **exact‑k**. The core model (with stabilized/symmetrized \(\Sigma_s\)) is:

$$
\begin{aligned}
\min_{x,v}\quad & \tfrac{1}{2}\,\gamma\, x^\top \Sigma_s x - \mu^\top x \\
\text{s.t.}\quad
& m \le \sum_i v_i \le k \quad (\text{or } \sum_i v_i = k \text{ if exact}),\\
& v_i = 0 \Rightarrow x_i = 0,\qquad
  v_i = 1 \Rightarrow f_{\min,i} \le x_i \le f_{\max,i},\\
& v_i \in \{0,1\}. 
\end{aligned}
$$

If `normalize_weights=true`, the budget \(\sum_i x_i = 1\) is **added** and outputs are normalized accordingly.  
A progressive **bound‑expansion** loop (up to `expand_rounds`) relaxes tight bounds and re‑solves.

**Pseudo‑algorithm (high‑level).**
1. Build \(\Sigma_s\) once; set band \(m\le\sum v_i\le k\) (or exact‑\(k\)).  
2. Solve MIQP with indicator/big‑M linking and caps \([f_{\min},f_{\max}]\).  
3. If some chosen \(x_i\) is near a bound, expand that bound and re‑solve (repeat up to `expand_rounds`).  
4. Extract support \(S=\{i: v_i=1\}\).  
   - **Refit**: compute \(SR^\star(S)\) and (optionally) \(w_{\text{MVE}}(S)\).  
   - **Vanilla**: keep the MIQP weights \(x\) (optionally normalized).

**Signature.**
```julia
mve_miqp_heuristic_search(μ::AbstractVector, Σ::AbstractMatrix;
    k::Integer,
    exactly_k::Bool=false,
    m::Union{Int,Nothing}=1,
    γ::Float64=1.0,
    fmin::AbstractVector=fill(-0.25, length(μ)),
    fmax::AbstractVector=fill(0.25, length(μ)),
    expand_rounds::Int=20,
    expand_factor::Float64=3.0,
    expand_tol::Float64=1e-2,
    mipgap::Float64=1e-4,
    time_limit::Real=200,
    threads::Int=1,
    x_start::Union{Nothing,AbstractVector}=nothing,
    v_start::Union{Nothing,AbstractVector}=nothing,
    compute_weights::Bool=true,
    normalize_weights::Bool=false,   # also toggles budget ∑x=1
    use_refit::Bool=false,
    epsilon::Real=Utils.EPS_RIDGE,
    stabilize_Σ::Bool=true,
    verbose::Bool=false,
    do_checks::Bool=false
) :: NamedTuple{(:selection, :weights, :sr, :status)}
```

**Arguments.**
- `μ`, `Σ`, `k`: asset moments and target cardinality (`k` is an upper bound unless `exactly_k=true`).
- `exactly_k`, `m`: exact‑\(k\) or band (default `m=1`, minimum cardinality ensures at least one asset is selected; when `exactly_k=true`, `m` is overridden to equal `k`).
- `γ`: risk‑aversion scale (just rescales the quadratic term).
- `fmin`, `fmax`: lower/upper caps active when an asset is selected (default: `[-0.25, 0.25]` allows both long and short positions).
- `expand_rounds`, `expand_factor`, `expand_tol`: bound‑expansion controls.
- `mipgap`, `time_limit`, `threads`, `x_start`, `v_start`: MIP controls and warm starts. The `threads` parameter (default `1`) controls both CPLEX and BLAS threading for thread safety.
- `compute_weights`: if `true`, return weights (refit or vanilla as per `use_refit`; default `true`).
- `normalize_weights`: adds \(\sum x = 1\) inside MIQP and normalizes outputs (or refit weights); default `false` for scale‑invariant portfolios.
- `use_refit`: if `true`, compute exact MVE Sharpe/weights on the final support; else keep MIQP portfolio `x` (default `false` for vanilla MIQP weights).
- `epsilon`, `stabilize_Σ`, `verbose`, `do_checks`: numerics and I/O.

**Thread safety.** The function temporarily sets BLAS threads to 1 during the solve and restores the original setting afterward, ensuring predictable CPU usage when `threads=1`.

**Returns.**
- `selection::Vector{Int}` — indices with \(v_i=1\).
- `weights::Vector{Float64}` — refit MVE weights (if `use_refit && compute_weights`) or the raw MIQP `x` (if `!use_refit && compute_weights`); zeros otherwise.
- `sr::Float64` — refit MVE Sharpe on \(S\) (refit) or Sharpe of `x` (vanilla).
- `status` — MOI termination status from the final solve.

---

### Utils module

#### `normalize_weights`

**Description.** Stable post‑scaling of a weight vector.  
Modes:
- `:absolute` — divide by \(\max(|\sum w|,\ \text{tol},\ 1\mathrm{e}{-10})\).  
- `:relative` (default) — divide by \(\max(|\sum w|,\ \text{tol}\,\lVert w\rVert_1,\ 1\mathrm{e}{-10})\).

If both \(\sum w\) and \(\lVert w\rVert_1\) are tiny, returns the zero vector.

**Signature.**
```julia
normalize_weights(w::AbstractVector;
                  mode::Symbol=:relative,
                  tol::Real=1e-6,
                  do_checks::Bool=false) -> Vector{Float64}
```

**Arguments.** `mode` ∈ `(:absolute, :relative)`, `tol` > 0; `do_checks` validates inputs.  
**Returns.** Rescaled copy of `w`.

---

## 6. Example

Run the bundled script comparing EXHAUSTIVE, LASSO (vanilla/refit), and MIQP (vanilla/refit) across several regimes:

```bash
julia --project=. example.jl
```

The script prints compact tables of \(SR\) and timing for each method and \(k\).

---

## 7. Reproducibility & numerics

- All routines symmetrize and (optionally) ridge‑stabilize \(\Sigma\) via a single helper and re‑use the stabilized matrix across inner loops.  
- Sharpe ratios use the same stabilized \(\Sigma_s\) consistently.  
- Randomized components (sampling, CV splits) accept an explicit RNG or use deterministic seeds across grids when appropriate.

---

## 8. License

MIT. See `LICENSE`.
